"""ローリング推定エンジン: 各営業日のシグナルを生成する.

重要な設計方針 (先読み防止)
---------------------------
  * 時点 t のシグナルに使う情報は
      - 窓 W_t = {t-L, ..., t-1} の日米 close-to-close リターン (平均・分散・相関)
      - 当日 t の「米国」close-to-close リターン
    のみ。日本の t の値は一切使わない (日本の t は米国の t より前に引けている
    ので使っても因果的ではあるが、論文の定義に従い米国情報のみを使う)。
  * 執行は t+1 の日本の open-to-close。
  * 事前エクスポージャー C_full は既定では学習期間 (2010-2014) で固定。
    prior_mode="expanding" にすると t 以前のデータのみで逐次再推定する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Params
from .data import DataBundle
from .signal import compute_signal

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 長期相関行列 C_full
# ---------------------------------------------------------------------------
def estimate_c_full(returns: pd.DataFrame, n_us: int, min_periods: int = 250) -> pd.DataFrame:
    """ペアワイズ相関で C_full を推定し、欠損はブロック平均で埋める。

    XLC (2018 上場) や XLRE (2015 上場) のように学習期間に存在しない銘柄が
    あるため、単純な dropna では推定できない。ペアワイズ相関を使い、それでも
    欠損する要素は 米-米 / 日-日 / 米-日 のブロック平均相関で補完する。
    """
    c = returns.corr(min_periods=min_periods)
    tickers = list(returns.columns)
    n = len(tickers)
    is_us = np.array([i < n_us for i in range(n)])
    arr = c.to_numpy(dtype=float).copy()

    off = ~np.eye(n, dtype=bool)
    blocks = {
        "uu": np.outer(is_us, is_us) & off,
        "jj": np.outer(~is_us, ~is_us) & off,
        "uj": (np.outer(is_us, ~is_us) | np.outer(~is_us, is_us)) & off,
    }
    for mask in blocks.values():
        vals = arr[mask]
        finite = vals[np.isfinite(vals)]
        fill = float(finite.mean()) if finite.size else 0.0
        target = mask & ~np.isfinite(arr)
        arr[target] = fill

    np.fill_diagonal(arr, 1.0)
    arr = 0.5 * (arr + arr.T)
    return pd.DataFrame(arr, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# エンジン
# ---------------------------------------------------------------------------
@dataclass
class SignalPanel:
    """各戦略の日次シグナル (index = シグナル算出日 t, columns = 日本 ETF)。

    execution_date[t] = t+1 (実際に日本で執行する営業日)
    """

    pca_sub: pd.DataFrame
    pca_plain: pd.DataFrame
    mom: pd.DataFrame
    execution_date: pd.Series
    factor_scores: pd.DataFrame
    eigenvalues: pd.DataFrame
    diagnostics: pd.DataFrame


class LeadLagEngine:
    def __init__(self, bundle: DataBundle, params: Params | None = None):
        self.bundle = bundle
        self.p = params or Params()

        self.us_tickers = list(bundle.us_cc.columns)
        self.jp_tickers = list(bundle.jp_cc.columns)
        self.n_us = len(self.us_tickers)

        # 米国ブロックが先、日本ブロックが後の結合リターン行列
        self.returns = pd.concat([bundle.us_cc, bundle.jp_cc], axis=1)
        self.returns.columns = self.us_tickers + self.jp_tickers
        self.dates = self.returns.index

        self._c_full_cache: dict[int, pd.DataFrame] = {}
        self._fixed_c_full: pd.DataFrame | None = None

    # -- C_full ------------------------------------------------------------
    def _corr_source(self) -> pd.DataFrame:
        """相関推定に使うリターン行列。lag_jp_in_corr=True なら日本を 1 日ずらす。"""
        if not self.p.lag_jp_in_corr:
            return self.returns
        shifted = self.returns.copy()
        shifted[self.jp_tickers] = shifted[self.jp_tickers].shift(-1)
        return shifted

    def c_full_at(self, i: int) -> pd.DataFrame:
        """時点インデックス i で使える長期相関行列を返す。"""
        src = self._corr_source()
        if self.p.prior_mode == "fixed":
            if self._fixed_c_full is None:
                mask = (src.index >= pd.Timestamp(self.p.train_start)) & (
                    src.index <= pd.Timestamp(self.p.train_end)
                )
                train = src.loc[mask]
                if len(train) < self.p.prior_min_obs:
                    raise ValueError(
                        f"学習期間の観測数が不足しています ({len(train)} < "
                        f"{self.p.prior_min_obs})。train_start/train_end を見直してください。"
                    )
                self._fixed_c_full = estimate_c_full(
                    train, self.n_us, min_periods=min(self.p.prior_min_obs, len(train) // 2)
                )
            return self._fixed_c_full

        # expanding: t 以前のデータのみで定期的に再推定
        step = max(1, self.p.prior_refit_every)
        # 直近の再推定タイミングまで遡る。ただし i を超えない (因果性) 範囲で
        # 最低観測数を確保する。
        key = min(i, max((i // step) * step, self.p.prior_min_obs))
        if key not in self._c_full_cache:
            train = src.iloc[:key]
            if len(train) < self.p.prior_min_obs:
                raise ValueError("expanding モードの学習データが不足しています")
            self._c_full_cache[key] = estimate_c_full(
                train, self.n_us, min_periods=min(self.p.prior_min_obs, len(train) // 2)
            )
        return self._c_full_cache[key]

    # -- 銘柄選択 ----------------------------------------------------------
    def _available(self, i: int) -> tuple[list[str], list[str]]:
        """時点 i で使える米国銘柄と、t+1 に執行できる日本銘柄。"""
        L = self.p.window
        win = self.returns.iloc[i - L : i]

        us_ok = [
            t for t in self.us_tickers
            if win[t].notna().all() and np.isfinite(self.returns.iloc[i][t])
        ]
        jp_next = self.bundle.jp_oc.iloc[i + 1]
        jp_ok = [
            t for t in self.jp_tickers
            if win[t].notna().all() and np.isfinite(jp_next[t])
        ]
        return us_ok, jp_ok

    # -- メイン ------------------------------------------------------------
    def run(self, verbose: bool = False) -> SignalPanel:
        p = self.p
        L = p.window
        start_ts = pd.Timestamp(
            p.backtest_start
            if p.backtest_start
            else (pd.Timestamp(p.train_end) + pd.Timedelta(days=1))
            if p.prior_mode == "fixed"
            else self.dates[max(L, p.prior_min_obs)]
        )

        sub_rows, plain_rows, mom_rows = {}, {}, {}
        fac_rows, eig_rows, diag_rows = {}, {}, {}

        n = len(self.dates)
        for i in range(L, n - 1):          # i+1 の執行日が必要なので n-1 まで
            t = self.dates[i]
            if t < start_ts:
                continue

            us_ok, jp_ok = self._available(i)
            if len(us_ok) < p.min_names or len(jp_ok) < p.min_names:
                continue

            win = self.returns.iloc[i - L : i]
            r_us_win = win[us_ok].to_numpy()
            r_jp_win = win[jp_ok].to_numpy()
            r_us_today = self.returns.iloc[i][us_ok].to_numpy()

            cf = self.c_full_at(i).loc[us_ok + jp_ok, us_ok + jp_ok].to_numpy()

            res_sub = compute_signal(
                r_us_win, r_jp_win, r_us_today, us_ok, jp_ok, cf,
                lam=p.lam, n_factors=p.n_factors,
            )
            res_plain = compute_signal(
                r_us_win, r_jp_win, r_us_today, us_ok, jp_ok, cf,
                lam=0.0, n_factors=p.n_factors,
            )

            sub_rows[t] = pd.Series(res_sub.scores, index=jp_ok)
            plain_rows[t] = pd.Series(res_plain.scores, index=jp_ok)
            mom_rows[t] = win[jp_ok].mean()          # 式(31) 単純モメンタム
            fac_rows[t] = pd.Series(
                res_sub.factor_scores, index=[f"f{k+1}" for k in range(p.n_factors)]
            )
            eig_rows[t] = pd.Series(
                res_sub.eigenvalues, index=[f"l{k+1}" for k in range(p.n_factors)]
            )
            diag_rows[t] = pd.Series(
                {
                    "n_us": len(us_ok),
                    "n_jp": len(jp_ok),
                    "exec_date": self.dates[i + 1],
                    "z_us_norm": float(np.linalg.norm(res_sub.z_us)),
                }
            )

        if not sub_rows:
            raise RuntimeError("シグナルを 1 日も計算できませんでした。期間設定を確認してください。")

        diagnostics = pd.DataFrame(diag_rows).T
        panel = SignalPanel(
            pca_sub=pd.DataFrame(sub_rows).T.reindex(columns=self.jp_tickers),
            pca_plain=pd.DataFrame(plain_rows).T.reindex(columns=self.jp_tickers),
            mom=pd.DataFrame(mom_rows).T.reindex(columns=self.jp_tickers),
            execution_date=diagnostics["exec_date"],
            factor_scores=pd.DataFrame(fac_rows).T,
            eigenvalues=pd.DataFrame(eig_rows).T,
            diagnostics=diagnostics,
        )
        if verbose:
            print(
                f"シグナル生成: {panel.pca_sub.index[0].date()} 〜 "
                f"{panel.pca_sub.index[-1].date()} ({len(panel.pca_sub)} 日)"
            )
        return panel

    # -- 最新 1 日分だけ (日次予測用) --------------------------------------
    def latest(self):
        """最新の米国終値に基づく翌営業日シグナルを返す。"""
        p = self.p
        L = p.window
        i = len(self.dates) - 1

        win = self.returns.iloc[i - L : i]
        us_ok = [t for t in self.us_tickers
                 if win[t].notna().all() and np.isfinite(self.returns.iloc[i][t])]
        jp_ok = [t for t in self.jp_tickers if win[t].notna().all()]
        if len(us_ok) < p.min_names or len(jp_ok) < p.min_names:
            raise RuntimeError("直近データが不足しています")

        cf = self.c_full_at(i).loc[us_ok + jp_ok, us_ok + jp_ok].to_numpy()
        res = compute_signal(
            win[us_ok].to_numpy(),
            win[jp_ok].to_numpy(),
            self.returns.iloc[i][us_ok].to_numpy(),
            us_ok, jp_ok, cf,
            lam=p.lam, n_factors=p.n_factors,
        )
        return self.dates[i], res
