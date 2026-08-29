"""論文の理想モデル (式 23-24) に従う合成データ生成器.

実データが無い環境でも実装の正しさを検証できるようにするための道具。

生成モデル
----------
  g_t = rho * g_{t-1} + sqrt(1-rho^2) * u_t          (共通ファクター, AR(1))
  米国 close-to-close :  z_U,t     = V_U* g_t   + eps_U,t
  日本 open-to-close  :  z_J,t+1   = V_J* g_t   + eps_J,t+1
  日本 close-to-close :  当日の open-to-close と同一 (寄付き = 前日終値と仮定)

つまり「米国で t に顕在化した共通ショックが、日本では t+1 の日中に現れる」。
rho > 0 のとき同一日付の日米相関も正になるため、結合相関行列の PCA が
共通モードを拾える (現実のデータでも overnight 経由で同様の構造になる)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadlag.config import JP_TICKERS, US_TICKERS, cyclical_score
from leadlag.data import build_bundle


def _true_loadings(us: list[str], jp: list[str], rng, noise: float = 0.35):
    """事前部分空間 (global / country / cyclical) に近い真のローディングを作る。"""
    tickers = us + jp
    n_us = len(us)
    n = len(tickers)
    is_us = np.array([i < n_us for i in range(n)])

    v1 = np.ones(n)
    v2 = np.where(is_us, 1.0, -1.0)
    v3 = np.array([cyclical_score(t) for t in tickers])
    base = np.column_stack([v1, v2, v3])
    base = base + noise * rng.standard_normal(base.shape)   # 事前からのズレ

    q, _ = np.linalg.qr(base)                                # 列直交化
    return q[:n_us, :], q[n_us:, :]


def make_bundle(
    n_days: int = 2000,
    seed: int = 0,
    rho: float = 0.3,
    signal_strength: float = 1.0,
    noise_us: float = 1.0,
    noise_jp: float = 1.0,
    daily_vol: float = 0.012,
    loading_noise: float = 0.35,
    pure_noise: bool = False,
    factor_strengths: tuple[float, float, float] = (1.0, 0.45, 0.30),
):
    """合成 DataBundle と真のローディングを返す。"""
    rng = np.random.default_rng(seed)
    us, jp = list(US_TICKERS), list(JP_TICKERS)
    n_us, n_jp = len(us), len(jp)
    v_us, v_jp = _true_loadings(us, jp, rng, noise=loading_noise)

    k = v_us.shape[1]
    g = np.zeros((n_days, k))
    for t in range(1, n_days):
        g[t] = rho * g[t - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(k)

    # 実市場と同様、グローバル(第1)ファクターが最も強いという構造を入れる。
    # これが無いと結合相関行列の日米ブロック間相関がほぼ 0 になり、
    # 事前部分空間の global 成分と country 成分が打ち消し合ってしまう。
    g = g * np.asarray(factor_strengths, dtype=float)[: g.shape[1]]

    if pure_noise:
        g[:] = 0.0

    z_us = signal_strength * (g @ v_us.T) + noise_us * rng.standard_normal((n_days, n_us))
    # 日本は 1 日遅れて反応する
    z_jp = np.zeros((n_days, n_jp))
    z_jp[1:] = signal_strength * (g[:-1] @ v_jp.T)
    z_jp += noise_jp * rng.standard_normal((n_days, n_jp))

    r_us = daily_vol * z_us
    r_jp_oc = daily_vol * z_jp

    dates = pd.bdate_range("2008-01-02", periods=n_days)

    us_close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + r_us, axis=0), index=dates, columns=us
    )
    jp_close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + r_jp_oc, axis=0), index=dates, columns=jp
    )
    jp_open = jp_close.shift(1).fillna(100.0)          # 寄付き = 前日終値

    us_panel = pd.concat({t: pd.DataFrame({"Open": us_close[t], "Close": us_close[t]})
                          for t in us}, axis=1)
    jp_panel = pd.concat({t: pd.DataFrame({"Open": jp_open[t], "Close": jp_close[t]})
                          for t in jp}, axis=1)

    bundle = build_bundle(us_panel, jp_panel)
    return bundle, {"v_us": v_us, "v_jp": v_jp, "g": g, "dates": dates}
