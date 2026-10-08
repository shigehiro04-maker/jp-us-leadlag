"""日次予測で並べる複数モデル。

2026-10 の検証（results/verification_2026-10/）を受けて追加した。

- SCS       : 金谷・吉田 (SIG-FIN-037-11)「業種対応シグナル」。日本の各業種に、
              GICS で対応づけた米国業種 ETF の当日リターンをそのまま割り当てる。
              共分散を一切使わない。検証では 2015〜2025 年の R/R が最も高く、
              論文公開 (2026-03-19) 後もプラスを保った唯一のモデル → **メイン**
- PCA_SUB   : 中川ほか (SIG-FIN-036-13) 部分空間正則化 PCA（従来のこのページの予測）
- PCA_L1250 : 金谷・吉田の長期窓 plain PCA（L=1250, λ=0）
- 合成      : 上の 3 つを日ごとに標準化して平均

どのモデルも上位 q・下位 q を等ウェイトでロング/ショートする（q=0.3 → 各 5 業種）。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import JP_TICKERS, Params

MAIN_MODEL = "SCS"
MODEL_ORDER = ["SCS", "PCA_SUB", "PCA_L1250", "合成"]
MODEL_LABEL = {
    "SCS": "業種対応 SCS",
    "PCA_SUB": "部分空間正則化PCA",
    "PCA_L1250": "長期窓PCA",
    "合成": "3モデル合成",
}

# 金谷・吉田 表1: GICS-to-TOPIX-17 対応表（* は主要構成銘柄から推定した業種）
SCS_MAP: dict[str, str] = {
    "1617.T": "XLP",   # 食品 ← 生活必需品
    "1618.T": "XLE",   # エネルギー資源 ← エネルギー
    "1619.T": "XLI",   # 建設・資材 ← 資本財 *
    "1620.T": "XLB",   # 素材・化学 ← 素材
    "1621.T": "XLV",   # 医薬品 ← ヘルスケア
    "1622.T": "XLY",   # 自動車・輸送機 ← 一般消費財
    "1623.T": "XLB",   # 鉄鋼・非鉄 ← 素材
    "1624.T": "XLI",   # 機械 ← 資本財
    "1625.T": "XLK",   # 電機・精密 ← 情報技術 *
    "1626.T": "XLC",   # 情報通信・サービスその他 ← コミュニケーション *
    "1627.T": "XLU",   # 電力・ガス ← 公益
    "1628.T": "XLI",   # 運輸・物流 ← 資本財
    "1629.T": "XLI",   # 商社・卸売 ← 資本財 *
    "1630.T": "XLY",   # 小売 ← 一般消費財
    "1631.T": "XLF",   # 銀行 ← 金融
    "1632.T": "XLF",   # 金融(除く銀行) ← 金融
    "1633.T": "XLRE",  # 不動産 ← 不動産
}


def select_long_short(sig: pd.Series, q: float = 0.3) -> tuple[list[str], list[str]]:
    """上位・下位 q を選ぶ。同点は業種コード順（SCS は同じ米国 ETF に写る業種が同点になる）。"""
    s = sig.dropna()
    s = s.reindex([t for t in JP_TICKERS if t in s.index] + [t for t in s.index if t not in JP_TICKERS])
    n = s.size
    k = max(1, int(round(q * n)))
    if 2 * k > n:
        k = n // 2
    if k == 0:
        return [], []
    ranked = s.sort_values(ascending=False, kind="stable")
    return list(ranked.index[:k]), list(ranked.index[-k:])


def scs_scores(us_today: pd.Series, jp_tickers: list[str]) -> pd.Series:
    """米国当日リターン（close-to-close）から SCS スコアを作る。欠測は 0。"""
    vals = {}
    for j in jp_tickers:
        u = SCS_MAP.get(j)
        v = us_today.get(u, np.nan) if u else np.nan
        vals[j] = float(v) if np.isfinite(v) else 0.0
    return pd.Series(vals)


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * 0.0


def us_return_on(bundle, asof: pd.Timestamp) -> pd.Series:
    """基準日の米国 close-to-close リターン。共通営業日より先行して届いた日にも対応。"""
    ahead = getattr(bundle, "us_cc_ahead", None)
    if ahead is not None and asof in ahead.index:
        return ahead.loc[asof]
    return bundle.us_cc.loc[asof]


def all_models(bundle, params: Params, asof: pd.Timestamp, pca_sub_scores: pd.Series) -> dict[str, pd.Series]:
    """基準日 asof の各モデルのスコア（日本業種ごと）。計算できないモデルは省く。"""
    from .engine import LeadLagEngine   # 循環 import 回避

    jp = list(pca_sub_scores.index)
    out: dict[str, pd.Series] = {}
    out["SCS"] = scs_scores(us_return_on(bundle, asof), jp)
    out["PCA_SUB"] = pca_sub_scores
    try:
        long_params = replace(params, window=1250, lam=0.0)
        if len(bundle.dates) > long_params.window + 1:
            a2, r2 = LeadLagEngine(bundle, long_params).latest()
            if pd.Timestamp(a2) == pd.Timestamp(asof):
                out["PCA_L1250"] = pd.Series(r2.scores, index=r2.jp_tickers).reindex(jp)
    except Exception as exc:   # noqa: BLE001  長期窓が取れないときは省くだけ
        print(f"PCA_L1250 を計算できませんでした: {exc}")
    zs = [_z(s.reindex(jp).astype(float)) for s in out.values()]
    out["合成"] = pd.concat(zs, axis=1).mean(axis=1)
    return {k: out[k] for k in MODEL_ORDER if k in out}
