"""評価指標 (論文 4.2 節).

  AR   = 年率リターン        = ann_factor * mean(R)
  RISK = 年率リスク          = sqrt(ann_factor) * std(R)
  R/R  = AR / RISK
  MDD  = 最大ドローダウン

注意: 論文の式 (27)(28) は年率換算係数が 12 と書かれているが、被説明変数は
日次リターンであり、表2 の水準 (AR 23.79%) と整合するのは 252 である。
ここでは既定を 252 とし、Params.ann_factor で変更できるようにしている。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annual_return(r: pd.Series, ann: int = 252) -> float:
    return float(ann * r.mean())


def annual_risk(r: pd.Series, ann: int = 252) -> float:
    return float(np.sqrt(ann) * r.std(ddof=1))


def risk_return(r: pd.Series, ann: int = 252) -> float:
    risk = annual_risk(r, ann)
    return float(annual_return(r, ann) / risk) if risk > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    """式 (30)。ピークからの最大下落幅 (%表示のため 100 倍、正の値で返す)。"""
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(-dd.min() * 100.0)


def newey_west_tstat(r: pd.Series, lags: int | None = None) -> float:
    """平均が 0 という帰無仮説に対する Newey-West t 値。"""
    x = r.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 10:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    e = x - x.mean()
    s = float(e @ e) / n
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        s += 2.0 * w * float(e[l:] @ e[:-l]) / n
    se = np.sqrt(max(s, 1e-18) / n)
    return float(x.mean() / se)


def summarize(returns: pd.Series, ann: int = 252) -> dict:
    r = returns.dropna()
    return {
        "AR": annual_return(r, ann) * 100.0,
        "RISK": annual_risk(r, ann) * 100.0,
        "R/R": risk_return(r, ann),
        "MDD": max_drawdown(r),
        "t(NW)": newey_west_tstat(r),
        "HitRate(%)": float((r > 0).mean() * 100.0),
        "N": int(r.size),
    }


def summary_table(returns: dict[str, pd.Series], ann: int = 252) -> pd.DataFrame:
    return pd.DataFrame({k: summarize(v, ann) for k, v in returns.items()}).T.round(2)
