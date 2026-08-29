"""部分空間正則化付き PCA によるリードラグ・シグナル (論文 3 章).

流れ:
  1. 事前部分空間 V0 = [v1, v2, v3] を構築 (3.1 節)
       v1: グローバル (全銘柄に等ウェイト)
       v2: 国スプレッド (米国 +, 日本 -) を v1 に直交化
       v3: シクリカル / ディフェンシブ を v1, v2 に直交化
  2. 長期相関行列 C_full から事前方向の固有値 D0 = diag(V0' C_full V0) を取り、
     ターゲット行列 C0_raw = V0 D0 V0' を作り、相関行列に正規化 (式 10-12)
  3. 各時点の窓内相関 C_t を C_t^reg = (1-λ) C_t + λ C0 に縮約 (式 13)
  4. C_t^reg の上位 K 固有ベクトルを V^(K) とし、米国ブロック V_U と
     日本ブロック V_J に分割 (式 14-16)
  5. 米国の当日標準化リターン z_U,t を射影してファクタースコア
     f_t = V_U' z_U,t、日本側へ復元して ẑ_J,t+1 = V_J f_t (式 18-20)
     すなわち ẑ_J,t+1 = B_t z_U,t,  B_t = V_J V_U'  (rank <= K)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import cyclical_score

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 3.1 事前部分空間
# ---------------------------------------------------------------------------
def _gram_schmidt(vectors: list[np.ndarray], tol: float = 1e-8) -> np.ndarray:
    """列ベクトル群を順に直交化・正規化する。退化した列は捨てる。"""
    basis: list[np.ndarray] = []
    for v in vectors:
        w = np.asarray(v, dtype=float).copy()
        for b in basis:
            w -= (b @ w) * b
        nrm = np.linalg.norm(w)
        if nrm > tol:
            basis.append(w / nrm)
    if not basis:
        raise ValueError("事前部分空間が構築できませんでした")
    return np.column_stack(basis)


def build_prior_basis(tickers: list[str], is_us: np.ndarray) -> np.ndarray:
    """事前固有ベクトル V0 (N x K0, 列直交) を作る。

    Parameters
    ----------
    tickers : 長さ N の銘柄コード列 (米国ブロックが先、日本ブロックが後)
    is_us   : 長さ N の bool 配列。True なら米国銘柄。
    """
    is_us = np.asarray(is_us, dtype=bool)
    n = len(tickers)
    if n != is_us.size:
        raise ValueError("tickers と is_us の長さが一致しません")

    v1 = np.ones(n)                                   # グローバル
    v2 = np.where(is_us, 1.0, -1.0)                   # 国スプレッド
    v3 = np.array([cyclical_score(t) for t in tickers])  # シクリカル/ディフェンシブ
    return _gram_schmidt([v1, v2, v3])


def prior_exposure_matrix(v0: np.ndarray, c_full: np.ndarray) -> np.ndarray:
    """式 (10)-(12): 事前エクスポージャー行列 C0 を作る。"""
    # (10) 事前方向の固有値。推定誤差で負になり得るので 0 で下限を切る。
    d0 = np.diag(np.clip(np.diag(v0.T @ c_full @ v0), 0.0, None))
    c0_raw = v0 @ d0 @ v0.T                           # (11) ターゲット行列
    delta = np.diag(c0_raw).copy()
    delta[delta < _EPS] = _EPS
    inv_sqrt = 1.0 / np.sqrt(delta)
    c0 = c0_raw * np.outer(inv_sqrt, inv_sqrt)        # (12) 相関行列へ正規化
    np.fill_diagonal(c0, 1.0)                         # 対角を厳密に 1 に
    return c0


# ---------------------------------------------------------------------------
# 3.2 部分空間正則化 PCA
# ---------------------------------------------------------------------------
def regularized_corr(c_t: np.ndarray, c0: np.ndarray, lam: float) -> np.ndarray:
    """式 (13): C_t^reg = (1-λ) C_t + λ C0"""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("λ は [0,1] である必要があります")
    return (1.0 - lam) * c_t + lam * c0


def top_eigenvectors(c: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """対称行列の上位 k 固有ベクトル (降順) と固有値を返す。"""
    c = 0.5 * (c + c.T)                               # 数値誤差の対称化
    vals, vecs = np.linalg.eigh(c)
    order = np.argsort(vals)[::-1][:k]
    return vecs[:, order], vals[order]


def propagation_matrix(v_us: np.ndarray, v_jp: np.ndarray) -> np.ndarray:
    """式 (21): B = V_J V_U'  (rank <= K の低ランク伝播行列)"""
    return v_jp @ v_us.T


# ---------------------------------------------------------------------------
# まとめ: 1 時点分のシグナル計算
# ---------------------------------------------------------------------------
@dataclass
class SignalResult:
    """ある t におけるシグナルと中間量。"""

    scores: np.ndarray          # ẑ_J,t+1  (日本側シグナル, 長さ NJ)
    factor_scores: np.ndarray   # f_t      (長さ K)
    z_us: np.ndarray            # z_U,t    (米国の標準化リターン, 長さ NU)
    b_matrix: np.ndarray        # B_t      (NJ x NU)
    eigenvalues: np.ndarray     # 上位 K 固有値
    us_tickers: list[str]
    jp_tickers: list[str]


def compute_signal(
    window_returns_us: np.ndarray,
    window_returns_jp: np.ndarray,
    today_returns_us: np.ndarray,
    us_tickers: list[str],
    jp_tickers: list[str],
    c_full: np.ndarray,
    *,
    lam: float = 0.9,
    n_factors: int = 3,
) -> SignalResult:
    """1 時点分のリードラグ・シグナルを計算する。

    Parameters
    ----------
    window_returns_us : (L, NU) 窓 W_t = {t-L,...,t-1} の米国 close-to-close リターン
    window_returns_jp : (L, NJ) 同じ窓の日本 close-to-close リターン
    today_returns_us  : (NU,)   当日 t の米国 close-to-close リターン
    c_full            : (N, N)  事前エクスポージャー用の長期相関行列
                                (列順は us_tickers + jp_tickers)
    """
    r_us = np.asarray(window_returns_us, dtype=float)
    r_jp = np.asarray(window_returns_jp, dtype=float)
    n_us, n_jp = r_us.shape[1], r_jp.shape[1]

    if r_us.shape[0] != r_jp.shape[0]:
        raise ValueError("米国と日本の窓長が一致しません")
    if today_returns_us.shape[0] != n_us:
        raise ValueError("当日リターンの次元が窓と一致しません")

    tickers = list(us_tickers) + list(jp_tickers)
    is_us = np.array([True] * n_us + [False] * n_jp)

    # --- 窓内の平均・標準偏差で標準化 (式 8-9)。t のデータは一切使わない ---
    r_win = np.hstack([r_us, r_jp])                       # (L, N)
    mu = r_win.mean(axis=0)
    sd = r_win.std(axis=0, ddof=0)
    sd = np.where(sd < _EPS, _EPS, sd)
    z_win = (r_win - mu) / sd

    # --- 窓内相関行列 C_t ---
    c_t = np.corrcoef(z_win, rowvar=False)
    c_t = np.nan_to_num(c_t, nan=0.0)
    np.fill_diagonal(c_t, 1.0)

    # --- 事前部分空間と C0 ---
    v0 = build_prior_basis(tickers, is_us)
    c0 = prior_exposure_matrix(v0, c_full)

    # --- 正則化と固有分解 ---
    c_reg = regularized_corr(c_t, c0, lam)
    v_k, eigvals = top_eigenvectors(c_reg, n_factors)
    v_us = v_k[:n_us, :]                                  # (NU, K)
    v_jp = v_k[n_us:, :]                                  # (NJ, K)

    # --- 当日の米国ショックを窓内統計で標準化 (式 17) ---
    z_us = (np.asarray(today_returns_us, dtype=float) - mu[:n_us]) / sd[:n_us]

    # --- 射影 -> 復元 (式 18-20) ---
    f_t = v_us.T @ z_us
    scores = v_jp @ f_t

    return SignalResult(
        scores=scores,
        factor_scores=f_t,
        z_us=z_us,
        b_matrix=propagation_matrix(v_us, v_jp),
        eigenvalues=eigvals,
        us_tickers=list(us_tickers),
        jp_tickers=list(jp_tickers),
    )
