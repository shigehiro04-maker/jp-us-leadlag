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

そこで市場全体の地合いは、同じリードラグ仮説をそのまま素直に使う別の推定量
      翌日の東京の日中リターン (等ウェイト)  ~  a + b * 当日の米国リターン (等ウェイト)
をローリング回帰で推定する。係数は常に過去のデータのみで推定する。

ただしこの予測値を「上がる/下がる」として読んではいけない (下の注記を参照)。
東京の日中リターンは恒常的にマイナスなので、正しい読み方は
「日中は売り優位。その圧力が過去と比べて強いか弱いか」である。
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
    bundle: DataBundle,
    window: int = 250,
    min_obs: int = 120,
    use_jp_prev: bool = True,
    asof: pd.Timestamp | None = None,
) -> dict:
    """最新の米国終値から翌営業日の東京市場の方向を予測する (日次運用用)。

    asof を渡すとその日付の米国終値を基準にする。業種シグナル側が
    データ欠損で 1 日さかのぼったときに、両者の基準日をそろえるために使う。
    """
    us_ew = bundle.us_cc.mean(axis=1, skipna=True).dropna()
    jp_ew_oc = bundle.jp_oc.mean(axis=1, skipna=True)
    jp_ew_cc = bundle.jp_cc.mean(axis=1, skipna=True)

    # 米国だけ先に届いている日を基準にする場合。日本の当日バーが無いので
    # 「前日の東京」の説明変数は使えない。米国リターンだけで組み直す。
    x_ahead = None
    if asof is not None and (len(us_ew) == 0 or pd.Timestamp(asof) > us_ew.index[-1]):
        ahead = getattr(bundle, "us_cc_ahead", None)
        if ahead is None or pd.Timestamp(asof) not in ahead.index:
            raise RuntimeError("指定された基準日の米国データがありません")
        x_ahead = float(ahead.loc[pd.Timestamp(asof)].mean(skipna=True))
        use_jp_prev = False

    dates = us_ew.index
    if asof is not None and x_ahead is None:
        pos = dates.searchsorted(pd.Timestamp(asof), side="right") - 1
        if pos < 0:
            raise RuntimeError("指定された基準日より前の米国データがありません")
        dates = dates[: pos + 1]
    t = dates[-1] if x_ahead is None else pd.Timestamp(asof)

    # 学習データ: (t' の米国, t'+1 の東京) のペアを過去から作る。
    # 基準日が共通営業日より先の場合も、学習は共通営業日の範囲で行う。
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

    if x_ahead is not None:
        x_now = [x_ahead]
    else:
        x_now = [us_ew.loc[t]] + ([jp_ew_cc.loc[t]] if use_jp_prev else [])
    pred = float(coef @ np.concatenate([[1.0], np.array(x_now)]))

    y_arr = np.asarray(ys, dtype=float)
    fitted = xd @ coef
    resid = y_arr - fitted
    q = np.quantile(fitted, [0.2, 0.4, 0.6, 0.8])

    return {
        "asof": t,
        "pred": pred,
        "bias": intraday_bias(y_arr),
        "strength": strength_label(pred, q),
        "quantiles": [float(x) for x in q],
        "base_mean": float(y_arr.mean()),
        "base_share_down": float((y_arr < 0).mean()),
        "us_ew_cc": float(x_ahead if x_ahead is not None else us_ew.loc[t]),
        "coef": coef.tolist(),
        "resid_sd": float(resid.std(ddof=len(coef))),
        "n_train": len(ys),
    }


# ---------------------------------------------------------------------------
# 「日中は売り優位、その強弱」という読み方
# ---------------------------------------------------------------------------
# 実データでの検証で分かったこと:
#   * 東京の日中 (寄付き→大引け) リターンは恒常的にマイナス。TOPIX-17 等ウェイトで
#     2015-2026 年の年率は約 -14%。日本株のリターンはほぼ夜間に発生している。
#   * このモデルの的中率 53% は「上下を当てている」のではなく、その恒常的な
#     マイナスを拾っているだけ。実際、常に売り持ちするほうが成績が良い
#     (R/R 1.40 対 1.24)。上昇予想の日の実現リターンも平均 -1.2bp とマイナス。
#   * ただし予測値には情報がある。下落予想の日は平均 -7.7bp、上昇予想の日は
#     -1.2bp と、6bp 以上の差がついている。
# したがって上下の当てものとして見せるのは誤りで、「日中は売り優位」を前提に
# その圧力の強弱として読むのが正しい。
# ---------------------------------------------------------------------------

STRENGTH_LABELS = ["強い", "やや強い", "標準", "やや弱い", "弱い"]


def intraday_bias(realized: np.ndarray) -> str:
    """学習期間の実績から、日中の地合いが売り優位かどうかを判定する。"""
    return "売り優位" if float(np.mean(realized)) < 0 else "買い優位"


def strength_label(pred: float, quantiles) -> str:
    """予測値が過去の分布のどこにあるかで、下押し圧力の強弱を表す。

    予測が小さい (よりマイナス寄り) ほど売り圧力が強い、と読む。
    """
    q20, q40, q60, q80 = quantiles
    if pred <= q20:
        return STRENGTH_LABELS[0]
    if pred <= q40:
        return STRENGTH_LABELS[1]
    if pred <= q60:
        return STRENGTH_LABELS[2]
    if pred <= q80:
        return STRENGTH_LABELS[3]
    return STRENGTH_LABELS[4]
