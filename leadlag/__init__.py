"""日米業種リードラグ投資戦略 (部分空間正則化付き PCA) の実装.

論文: 中川 慧, 竹本 悠城, 久保 健治, 加藤 真大
      「部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略」
      人工知能学会 金融情報学研究会 SIG-FIN-036-13
"""

from .config import DEFAULT_PARAMS, JP_TICKERS, US_TICKERS, Params
from .data import DataBundle, build_bundle, load_bundle
from .engine import LeadLagEngine, SignalPanel, estimate_c_full
from .signal import (
    build_prior_basis,
    compute_signal,
    prior_exposure_matrix,
    propagation_matrix,
    regularized_corr,
)

__version__ = "0.1.0"

__all__ = [
    "Params",
    "DEFAULT_PARAMS",
    "US_TICKERS",
    "JP_TICKERS",
    "DataBundle",
    "build_bundle",
    "load_bundle",
    "LeadLagEngine",
    "SignalPanel",
    "estimate_c_full",
    "build_prior_basis",
    "prior_exposure_matrix",
    "regularized_corr",
    "propagation_matrix",
    "compute_signal",
]
