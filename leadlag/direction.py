"""市場全体の方向予測.

なぜ別モジュールなのか
----------------------
論文のリードラグ・シグナル ẑ_J = B z_U は、構造上「市場全体の方向」をほとんど
予測しない。理由は伝播行列 B = V_J V_U' の作られ方にある。

  λ を大きく取ると V^(K) は事前部分空間 span{v1, v2, v3} にほぼ一致する。
  ここで v1 = 全銘柄一様、v2 = 米国 +/日本 - なので

      span{v1, v2} = span{(1_U, 0), (0, 1_J)}

  であり、この 2 次元部分空間への射影行列は日米ブロック対角になる。
  つまり v1 と v2 の日米クロス項はちょうど打ち消し合い、B に残るクロス成分は
  実質的に v3 (シクリカル / ディフェンシブ) だけになる。

その結果 mean_j(ẑ_J) ≈ 0 となり、シグナルは本質的に「業種間の相対」しか
語らない (実際に合成データでも実データでも、断面平均の標準偏差は個別シグナルの
100 分の 1 程度になる)。これは欠陥ではなく、市場中立なロングショート戦略として
設計されているということである。

そこで市場全体の方向は、同じリードラグ仮説をそのまま素直に使う別の推定量
      翌日の東京の日中リターン (等ウェイト)  ~  a + b * 当日の米国リターン (等ウェイト)
をローリング回帰で推定して予測する。係数は常に過去のデータのみで推定する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DataBundle


def market_factors(bundle: DataBundle, exec_dates: pd.Series) -> pd.DataFrame:
    """シグナル日 t をインデックスとして、説明変数と被説明変数を組む。

    x : 当日 t の米国業種 ETF 等ウェイト close-to-close リターン
    y : 翌営業日 t+1 の TOPIX-17 等ウェイト open-to-close リターン
    """
    us_ew = bundle.us_cc.mean(axis=1, skipna=True)
    jp_ew_oc = bundle.jp_oc.mean(axis=1, skipna=True)
    jp_ew_cc = bundle.jp_cc.mean(axis=1, skipna=True)

    rows = {}
    for t in exec_dates.index:
        e = exec_dates.loc[t]
        if t not in us_ew.index or e not in jp_ew_oc.index:
            continue
        rows[t] = {
            "us_ew_cc": float(us_ew.loc[t]),
            "jp_ew_cc_prev": float(jp_ew_cc.loc[t]) if t in jp_ew_cc.index else np.nan,
            "exec_date": e,
            "jp_ew_oc_next": float(jp_ew_oc.loc[e]),
        }
    df = pd.DataFrame(rows).T
    for c in ("us_ew_cc", "jp_ew_cc_prev", "jp_ew_oc_next"):
        df[c] = df[c].astype(float)
    return df


def rolling_direction(
    frame: pd.DataFrame,
    window: int = 250,
    min_obs: int = 120,
    use_jp_prev: bool = True,
) -> pd.DataFrame:
    """ローリング OLS で翌日の東京日中リターンを予測する。

    説明変数: 当日の米国 EW リターン (+ 任意で当日の東京 EW リターン)。
    係数は t 時点までの過去 window 日のみで推定するため先読みは無い。
    """
    cols = ["us_ew_cc"] + (["jp_ew_cc_prev"] if use_jp_prev else [])
    x_all = frame[cols].to_numpy(dtype=float)
    y_all = frame["jp_ew_oc_next"].to_numpy(dtype=float)
    n = len(frame)

    preds = np.full(n, np.nan)
    betas = np.full((n, len(cols) + 1), np.nan)

    for i in range(n):
        lo = max(0, i - window)
        # i 番目の y は「t+1 の実現値」なので、学習には i-1 までしか使えない。
        xs, ys = x_all[lo:i], y_all[lo:i]
        ok = np.isfinite(xs).all(axis=1) & np.isfinite(ys)
        if ok.sum() < min_obs:
            continue
        xd = np.column_stack([np.ones(ok.sum()), xs[ok]])
        coef, *_ = np.linalg.lstsq(xd, ys[ok], rcond=None)
        betas[i] = coef
        if np.isfinite(x_all[i]).all():
            preds[i] = float(coef @ np.concatenate([[1.0], x_all[i]]))

    out = frame.copy()
    out["pred"] = preds
    out["pred_up"] = out["pred"] > 0
    out["actual_up"] = out["jp_ew_oc_next"] > 0
    out["correct"] = out["pred_up"] == out["actual_up"]
    out["timed_return"] = np.sign(out["pred"]) * out["jp_ew_oc_next"]
    for j, c in enumerate(["const"] + cols):
        out[f"beta_{c}"] = betas[:, j]
    return out.dropna(subset=["pred"])


def naive_direction(frame: pd.DataFrame) -> pd.DataFrame:
    """比較用: 当日の米国 EW リターンの符号をそのまま翌日の方向とする。"""
    out = frame.copy()
    out["pred"] = out["us_ew_cc"]
    out["pred_up"] = out["pred"] > 0
    out["actual_up"] = out["jp_ew_oc_next"] > 0
    out["correct"] = out["pred_up"] == out["actual_up"]
    out["timed_return"] = np.sign(out["pred"]) * out["jp_ew_oc_next"]
    return out


def latest_direction(
    bundle: DataBundle, window: int = 250, min_obs: int = 120, use_jp_prev: bool = True
) -> dict:
    """最新の米国終値から翌営業日の東京市場の方向を予測する (日次運用用)。"""
    us_ew = bundle.us_cc.mean(axis=1, skipna=True).dropna()
    jp_ew_oc = bundle.jp_oc.mean(axis=1, skipna=True)
    jp_ew_cc = bundle.jp_cc.mean(axis=1, skipna=True)

    dates = us_ew.index
    t = dates[-1]

    # 学習データ: (t' の米国, t'+1 の東京) のペアを過去から作る
    xs, ys = [], []
    for k in range(max(0, len(dates) - window - 1), len(dates) - 1):
        d0, d1 = dates[k], dates[k + 1]
        row = [us_ew.loc[d0]]
        if use_jp_prev:
            row.append(jp_ew_cc.loc[d0])
        y = jp_ew_oc.loc[d1]
        if np.all(np.isfinite(row)) and np.isfinite(y):
            xs.append(row)
            ys.append(y)

    if len(ys) < min_obs:
        raise RuntimeError("方向予測の学習データが不足しています")

    xd = np.column_stack([np.ones(len(xs)), np.array(xs)])
    coef, *_ = np.linalg.lstsq(xd, np.array(ys), rcond=None)

    x_now = [us_ew.loc[t]] + ([jp_ew_cc.loc[t]] if use_jp_prev else [])
    pred = float(coef @ np.concatenate([[1.0], np.array(x_now)]))

    resid = np.array(ys) - xd @ coef
    return {
        "asof": t,
        "pred": pred,
        "direction": "上昇" if pred > 0 else "下落",
        "us_ew_cc": float(us_ew.loc[t]),
        "coef": coef.tolist(),
        "resid_sd": float(resid.std(ddof=len(coef))),
        "n_train": len(ys),
    }
