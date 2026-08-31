"""ユニバース定義と既定パラメータ.

論文: 中川ほか「部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略」
      (SIG-FIN-036-13) の 2 章 / 4.1 節に対応。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List

# ---------------------------------------------------------------------------
# 米国側: S&P500 を GICS 11 業種に分けた Select Sector SPDR ETF
# ---------------------------------------------------------------------------
US_TICKERS: List[str] = [
    "XLB",   # 素材
    "XLC",   # コミュニケーション・サービス (2018-06 上場)
    "XLE",   # エネルギー
    "XLF",   # 金融
    "XLI",   # 資本財
    "XLK",   # 情報技術
    "XLP",   # 生活必需品
    "XLRE",  # 不動産 (2015-10 上場)
    "XLU",   # 公益
    "XLV",   # ヘルスケア
    "XLY",   # 一般消費財
]

US_NAMES: Dict[str, str] = {
    "XLB": "素材",
    "XLC": "コミュニケーション",
    "XLE": "エネルギー",
    "XLF": "金融",
    "XLI": "資本財",
    "XLK": "情報技術",
    "XLP": "生活必需品",
    "XLRE": "不動産",
    "XLU": "公益",
    "XLV": "ヘルスケア",
    "XLY": "一般消費財",
}

# ---------------------------------------------------------------------------
# 日本側: NEXT FUNDS TOPIX-17 業種別 ETF (配当込み指数連動)
# ---------------------------------------------------------------------------
JP_TICKERS: List[str] = [f"{n}.T" for n in range(1617, 1634)]

JP_NAMES: Dict[str, str] = {
    "1617.T": "食品",
    "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄",
    "1624.T": "機械",
    "1625.T": "電機・精密",
    "1626.T": "情報通信・サービスその他",
    "1627.T": "電力・ガス",
    "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",
    "1630.T": "小売",
    "1631.T": "銀行",
    "1632.T": "金融(除く銀行)",
    "1633.T": "不動産",
}

# ---------------------------------------------------------------------------
# 事前部分空間 v3 (シクリカル / ディフェンシブ) のラベル (論文 4.1 節)
#   +1 = シクリカル(景気敏感), -1 = ディフェンシブ, 記載のない銘柄は 0
# ---------------------------------------------------------------------------
US_CYCLICAL: List[str] = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE: List[str] = ["XLK", "XLP", "XLU", "XLV"]
JP_CYCLICAL: List[str] = ["1618.T", "1625.T", "1629.T", "1631.T"]
JP_DEFENSIVE: List[str] = ["1617.T", "1621.T", "1627.T", "1630.T"]


def cyclical_score(ticker: str) -> float:
    if ticker in US_CYCLICAL or ticker in JP_CYCLICAL:
        return 1.0
    if ticker in US_DEFENSIVE or ticker in JP_DEFENSIVE:
        return -1.0
    return 0.0


def display_name(ticker: str) -> str:
    return US_NAMES.get(ticker) or JP_NAMES.get(ticker) or ticker


# ---------------------------------------------------------------------------
# 既定パラメータ
# ---------------------------------------------------------------------------
@dataclass
class Params:
    # --- モデル ---
    window: int = 60            # L: 推定ウィンドウ長 (営業日)
    n_factors: int = 3          # K: 使用する主成分数
    n_prior: int = 3            # K0: 事前部分空間の次元 (v1, v2, v3)
    lam: float = 0.9            # λ: 事前エクスポージャーへの縮約強度 (式13)
    quantile: float = 0.3       # q: ロング/ショートの分位

    # --- 事前エクスポージャー C_full の推定方法 ---
    # "fixed"     : train_start〜train_end の固定期間で 1 度だけ推定 (論文の設定)
    # "expanding" : t 以前の全データで定期的に再推定 (完全に因果的。XLC/XLRE の
    #               上場が遅い問題にも自然に対応する)
    prior_mode: str = "fixed"
    train_start: str = "2010-01-01"
    train_end: str = "2014-12-31"
    prior_refit_every: int = 60      # expanding 時の再推定間隔 (営業日)
    prior_min_obs: int = 250         # expanding 時に必要な最低観測数

    # --- バックテスト ---
    start: str = "2010-01-01"        # データ取得開始
    end: str | None = None           # None なら最新まで
    backtest_start: str | None = None  # None なら train_end の翌日
    cost_bps: float = 0.0            # 片道コスト(bps)。論文は 0 (コスト考慮なし)
    ann_factor: int = 252            # 年率換算係数

    # --- 実装オプション ---
    # True にすると相関行列を作る際に日本側リターンを 1 日前にずらし、
    # 「米国 t → 日本 t+1」の実際のリード・ラグ構造に合わせる (論文外の変種)
    lag_jp_in_corr: bool = False
    min_names: int = 6               # この数を下回る日はスキップ
    # 推定ウィンドウ内で、その銘柄に値が入っている必要のある割合。
    # 1.0 にすると 1 日でも欠ければ 60 営業日ぶんその銘柄が使えなくなる。
    # 無料データは散発的に値が抜けるため、既定では 5% までの欠損を許容する。
    min_window_coverage: float = 0.95

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMS = Params()
