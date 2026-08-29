"""ロングショート・ポートフォリオの構築とバックテスト (論文 2.2 節).

  L_{t+1} = シグナル上位 q, S_{t+1} = 下位 q
  w = +1/|L| (ロング), -1/|S| (ショート)  → Σw = 0, Σ|w| = 2
  R_{t+1} = Σ_j w_j * r^oc_{j,t+1}
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Params
from .data import DataBundle
from .engine import SignalPanel


def cross_section_weights(signal_row: pd.Series, q: float = 0.3) -> pd.Series:
    """1 日分のシグナルから等ウェイトのロングショート・ウェイトを作る (式 3-6)。"""
    s = signal_row.dropna()
    n = s.size
    k = max(1, int(round(q * n)))
    if 2 * k > n:
        k = n // 2
    if k == 0:
        return pd.Series(0.0, index=signal_row.index)

    ranked = s.sort_values(ascending=False)
    longs, shorts = ranked.index[:k], ranked.index[-k:]

    w = pd.Series(0.0, index=signal_row.index)
    w[longs] = 1.0 / k
    w[shorts] = -1.0 / k
    return w


def double_sort_weights(sig_a: pd.Series, sig_b: pd.Series) -> pd.Series:
    """2x2 ダブルソート (論文 4.3-4)。両方 High をロング、両方 Low をショート。"""
    common = sig_a.dropna().index.intersection(sig_b.dropna().index)
    if len(common) < 4:
        return pd.Series(0.0, index=sig_a.index)
    a, b = sig_a[common], sig_b[common]
    hi_a, hi_b = a > a.median(), b > b.median()

    longs = common[(hi_a & hi_b).to_numpy()]
    shorts = common[(~hi_a & ~hi_b).to_numpy()]

    w = pd.Series(0.0, index=sig_a.index)
    if len(longs):
        w[longs] = 1.0 / len(longs)
    if len(shorts):
        w[shorts] = -1.0 / len(shorts)
    return w


@dataclass
class StrategyResult:
    returns: pd.Series          # 執行日 (t+1) をインデックスとする日次リターン
    weights: pd.DataFrame       # 執行日 x 銘柄
    turnover: pd.Series
    gross_returns: pd.Series    # コスト控除前


def run_strategy(
    signals: pd.DataFrame,
    bundle: DataBundle,
    exec_dates: pd.Series,
    params: Params,
    weight_fn=None,
    second_signals: pd.DataFrame | None = None,
) -> StrategyResult:
    """シグナル panel を受け取り、t+1 の open-to-close で執行した結果を返す。"""
    q = params.quantile
    rows_w, rows_r, rows_gr, rows_to = {}, {}, {}, {}
    prev_w: pd.Series | None = None

    for t in signals.index:
        exec_date = exec_dates.loc[t]
        row = signals.loc[t]
        if second_signals is not None:
            w = double_sort_weights(row, second_signals.loc[t])
        elif weight_fn is not None:
            w = weight_fn(row)
        else:
            w = cross_section_weights(row, q)

        r_oc = bundle.jp_oc.loc[exec_date].reindex(w.index)
        valid = r_oc.notna()
        w = w.where(valid, 0.0)
        gross = float((w * r_oc.fillna(0.0)).sum())

        if prev_w is None:
            turn = float(w.abs().sum())
        else:
            turn = float((w - prev_w.reindex(w.index).fillna(0.0)).abs().sum())
        cost = turn * params.cost_bps / 1e4

        rows_w[exec_date] = w
        rows_gr[exec_date] = gross
        rows_r[exec_date] = gross - cost
        rows_to[exec_date] = turn
        prev_w = w

    return StrategyResult(
        returns=pd.Series(rows_r).sort_index(),
        weights=pd.DataFrame(rows_w).T.sort_index(),
        turnover=pd.Series(rows_to).sort_index(),
        gross_returns=pd.Series(rows_gr).sort_index(),
    )


def market_timing(
    signals: pd.DataFrame,
    bundle: DataBundle,
    exec_dates: pd.Series,
    params: Params,
) -> pd.DataFrame:
    """市場全体の方向予測。

    シグナルの断面平均 mean_j(ẑ_J,t+1) をグローバルファクターの伝播量とみなし、
    その符号で翌日の東京市場 (TOPIX-17 等ウェイト) の日中方向を予測する。
    """
    rows = {}
    for t in signals.index:
        exec_date = exec_dates.loc[t]
        s = signals.loc[t].dropna()
        if s.empty:
            continue
        r_oc = bundle.jp_oc.loc[exec_date].reindex(s.index).dropna()
        if r_oc.empty:
            continue
        mkt = float(r_oc.mean())
        pred = float(s.mean())
        rows[exec_date] = {
            "signal_mean": pred,
            "signal_disp": float(s.std(ddof=1)),
            "market_oc": mkt,
            "pred_up": pred > 0,
            "actual_up": mkt > 0,
            "correct": (pred > 0) == (mkt > 0),
            "timed_return": np.sign(pred) * mkt,
        }
    df = pd.DataFrame(rows).T.sort_index()
    for c in ("signal_mean", "signal_disp", "market_oc", "timed_return"):
        df[c] = df[c].astype(float)
    return df


def run_all(panel: SignalPanel, bundle: DataBundle, params: Params) -> dict:
    """論文の 4 戦略 + 市場方向をまとめて実行する。"""
    ex = panel.execution_date
    results = {
        "MOM": run_strategy(panel.mom, bundle, ex, params),
        "PCA_PLAIN": run_strategy(panel.pca_plain, bundle, ex, params),
        "PCA_SUB": run_strategy(panel.pca_sub, bundle, ex, params),
        "DOUBLE": run_strategy(
            panel.mom, bundle, ex, params, second_signals=panel.pca_sub
        ),
    }
    timing = market_timing(panel.pca_sub, bundle, ex, params)

    # 市場全体の方向は別モデル (direction.py) で推定する。理由はモジュールの
    # docstring を参照 (リードラグ・シグナルは構造上ほぼ市場中立になるため)。
    from .direction import market_factors, naive_direction, rolling_direction

    frame = market_factors(bundle, ex)
    direction = rolling_direction(frame)
    direction_naive = naive_direction(frame)

    # ベンチマーク: TOPIX-17 等ウェイトの日中 (open-to-close) リターン
    bench_dates = ex.to_numpy()
    bench = bundle.jp_oc.loc[bench_dates].mean(axis=1)
    bench.name = "EW_BENCH"

    return {
        "strategies": results,
        "timing": timing,
        "direction": direction,
        "direction_naive": direction_naive,
        "benchmark": bench,
    }
