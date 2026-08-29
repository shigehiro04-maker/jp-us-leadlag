"""実装の検証テスト。

  1. 数学的性質  : 事前基底の直交性、C0 が相関行列であること、rank(B) <= K
  2. 経済的性質  : 理想モデル下で B の推定が真の B* に近づくこと
  3. 実装の健全性: 先読みが無いこと、ノイズのみのデータで収益が出ないこと
  4. 正則化の効果: 短い窓 + 強いノイズで PCA_SUB が PCA_PLAIN を上回ること
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.backtest import cross_section_weights, run_all
from leadlag.config import JP_TICKERS, US_TICKERS, Params
from leadlag.engine import LeadLagEngine, estimate_c_full
from leadlag.metrics import max_drawdown, risk_return
from leadlag.signal import (
    build_prior_basis,
    compute_signal,
    prior_exposure_matrix,
    regularized_corr,
)
from tests.synthetic import make_bundle

TICKERS = list(US_TICKERS) + list(JP_TICKERS)
N_US = len(US_TICKERS)
IS_US = np.array([i < N_US for i in range(len(TICKERS))])


# ---------------------------------------------------------------------------
# 1. 数学的性質
# ---------------------------------------------------------------------------
def test_prior_basis_is_orthonormal():
    v0 = build_prior_basis(TICKERS, IS_US)
    assert v0.shape == (len(TICKERS), 3)
    np.testing.assert_allclose(v0.T @ v0, np.eye(3), atol=1e-10)


def test_prior_basis_directions():
    """v1 はグローバル、v2 は日米で符号が反転していること。"""
    v0 = build_prior_basis(TICKERS, IS_US)
    v1 = v0[:, 0]
    assert np.allclose(v1, v1[0])                      # 全銘柄同じ重み
    v2 = v0[:, 1]
    assert v2[IS_US].mean() * v2[~IS_US].mean() < 0     # 米国と日本で符号が逆


def test_global_and_country_modes_cancel_across_markets():
    """span{v1, v2} = span{(1_U,0), (0,1_J)} なので、この 2 次元への射影行列の
    日米クロスブロックは厳密に 0 になる。B に残るクロス成分は実質 v3 だけ、
    という手法の重要な性質を確認する。"""
    v0 = build_prior_basis(TICKERS, IS_US)
    p12 = v0[:, :2] @ v0[:, :2].T
    np.testing.assert_allclose(p12[:N_US, N_US:], 0.0, atol=1e-12)

    p3 = np.outer(v0[:, 2], v0[:, 2])
    assert np.abs(p3[:N_US, N_US:]).max() > 1e-3   # v3 はクロス成分を持つ


def test_c0_is_a_correlation_matrix():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, len(TICKERS)))
    c_full = np.corrcoef(x, rowvar=False)
    v0 = build_prior_basis(TICKERS, IS_US)
    c0 = prior_exposure_matrix(v0, c_full)

    np.testing.assert_allclose(np.diag(c0), 1.0, atol=1e-12)
    np.testing.assert_allclose(c0, c0.T, atol=1e-12)
    assert np.all(np.abs(c0) <= 1.0 + 1e-9)


def test_regularized_corr_interpolates():
    a = np.eye(3)
    b = np.full((3, 3), 0.5)
    np.fill_diagonal(b, 1.0)
    np.testing.assert_allclose(regularized_corr(a, b, 0.0), a)
    np.testing.assert_allclose(regularized_corr(a, b, 1.0), b)


def test_propagation_matrix_is_low_rank():
    """命題 1: rank(B) <= K"""
    bundle, _ = make_bundle(n_days=400, seed=1)
    eng = LeadLagEngine(bundle, Params(window=60, prior_mode="expanding",
                                       prior_min_obs=200))
    _, res = eng.latest()
    assert np.linalg.matrix_rank(res.b_matrix, tol=1e-8) <= 3
    assert res.b_matrix.shape == (len(res.jp_tickers), len(res.us_tickers))


def test_signal_equals_b_times_z():
    """式 (20): ẑ_J = B z_U が実際に成り立つこと。"""
    bundle, _ = make_bundle(n_days=400, seed=2)
    eng = LeadLagEngine(bundle, Params(window=60, prior_mode="expanding",
                                       prior_min_obs=200))
    _, res = eng.latest()
    np.testing.assert_allclose(res.scores, res.b_matrix @ res.z_us, atol=1e-10)


def test_weights_are_dollar_neutral():
    s = pd.Series(np.arange(17.0), index=JP_TICKERS)
    w = cross_section_weights(s, q=0.3)
    assert abs(w.sum()) < 1e-12                 # Σw = 0
    assert abs(w.abs().sum() - 2.0) < 1e-12     # Σ|w| = 2
    assert w[JP_TICKERS[-1]] > 0                # 最大シグナルがロング
    assert w[JP_TICKERS[0]] < 0                 # 最小シグナルがショート
    assert (w > 0).sum() == (w < 0).sum() == 5  # q=0.3, n=17 -> 5 銘柄ずつ


# ---------------------------------------------------------------------------
# 2. 経済的性質: 理想モデルでの回復
# ---------------------------------------------------------------------------
def test_recovers_true_subspace():
    """理想モデル下で、推定した B が真の V_J V_U' と強く相関すること。"""
    bundle, truth = make_bundle(n_days=3000, seed=3, rho=0.6,
                                noise_us=0.5, noise_jp=0.5, loading_noise=0.1)
    eng = LeadLagEngine(bundle, Params(window=250, prior_mode="expanding",
                                       prior_min_obs=500, n_factors=3))
    _, res = eng.latest()

    b_true = truth["v_jp"] @ truth["v_us"].T
    b_hat = res.b_matrix
    corr = np.corrcoef(b_true.ravel(), b_hat.ravel())[0, 1]
    assert abs(corr) > 0.8, f"真の伝播行列との相関が低い: {corr:.3f}"


def test_signal_predicts_next_day_returns():
    """理想モデル下で、シグナルと翌日実現リターンの相関が正であること。

    ここでは「事前ラベル (シクリカル/ディフェンシブ) が正しい」状況を想定する
    (loading_noise を小さく取る)。λ=0.9 の推定量は事前部分空間に強く縮約される
    ため、予測力は事前情報の正しさに直接依存する。
    """
    bundle, _ = make_bundle(n_days=2000, seed=4, rho=0.4, noise_jp=0.6,
                            loading_noise=0.05,
                            factor_strengths=(1.0, 0.4, 0.8))
    params = Params(window=60, prior_mode="expanding", prior_min_obs=300)
    panel = LeadLagEngine(bundle, params).run()

    ics = []
    for t in panel.pca_sub.index:
        s = panel.pca_sub.loc[t].dropna()
        r = bundle.jp_oc.loc[panel.execution_date.loc[t]].reindex(s.index)
        if r.notna().sum() > 5:
            ics.append(np.corrcoef(s.rank(), r.rank())[0, 1])
    ics = np.asarray(ics)
    ic = float(ics.mean())
    tstat = ic / (ics.std(ddof=1) / np.sqrt(ics.size))
    assert tstat > 4.0, f"IC が有意でない: IC={ic:.4f}, t={tstat:.2f}"


def test_signal_level_is_market_neutral_by_construction():
    """事前部分空間 span{v1, v2} の日米クロス項は厳密に打ち消し合うため、
    シグナルの断面平均 (= 市場方向の情報) はほぼ 0 になる。
    これは実装のバグではなく手法の性質であり、direction.py を別に持つ理由。"""
    bundle, _ = make_bundle(n_days=1500, seed=12, rho=0.4)
    params = Params(window=60, prior_mode="expanding", prior_min_obs=300, lam=0.9)
    panel = LeadLagEngine(bundle, params).run()

    level = panel.pca_sub.mean(axis=1).std()
    cross_sectional = panel.pca_sub.stack().std()
    assert level / cross_sectional < 0.1, (
        f"断面平均が想定より大きい: {level / cross_sectional:.3f}"
    )


# ---------------------------------------------------------------------------
# 3. 実装の健全性
# ---------------------------------------------------------------------------
def test_no_lookahead():
    """将来のデータを差し替えても、過去のシグナルが一切変わらないこと。"""
    bundle, _ = make_bundle(n_days=1200, seed=5)
    params = Params(window=60, prior_mode="expanding", prior_min_obs=300,
                    prior_refit_every=60)

    panel_full = LeadLagEngine(bundle, params).run()

    cut = 900
    rng = np.random.default_rng(99)
    tampered = bundle
    import copy

    tampered = copy.deepcopy(bundle)
    for df in (tampered.us_cc, tampered.jp_cc, tampered.jp_oc):
        df.iloc[cut:] = rng.standard_normal(df.iloc[cut:].shape) * 0.02

    panel_tampered = LeadLagEngine(tampered, params).run()

    cut_date = bundle.dates[cut - 2]        # 改変位置より前
    common = panel_full.pca_sub.index[panel_full.pca_sub.index <= cut_date]
    common = common.intersection(panel_tampered.pca_sub.index)
    assert len(common) > 100

    np.testing.assert_allclose(
        panel_full.pca_sub.loc[common].to_numpy(),
        panel_tampered.pca_sub.loc[common].to_numpy(),
        atol=1e-12,
        err_msg="将来データの変更が過去のシグナルに影響している = 先読みがある",
    )


def test_no_edge_on_pure_noise():
    """予測可能な構造が無いデータでは R/R がほぼ 0 になること。"""
    bundle, _ = make_bundle(n_days=2000, seed=6, pure_noise=True)
    params = Params(window=60, prior_mode="expanding", prior_min_obs=300)
    panel = LeadLagEngine(bundle, params).run()
    res = run_all(panel, bundle, params)
    rr = risk_return(res["strategies"]["PCA_SUB"].returns)
    assert abs(rr) < 0.6, f"ノイズだけのデータで R/R={rr:.2f} は不自然"


def test_execution_date_is_strictly_after_signal_date():
    bundle, _ = make_bundle(n_days=600, seed=7)
    params = Params(window=60, prior_mode="expanding", prior_min_obs=300)
    panel = LeadLagEngine(bundle, params).run()
    assert (panel.execution_date.index < panel.execution_date.to_numpy()).all()


def test_costs_reduce_returns():
    bundle, _ = make_bundle(n_days=1000, seed=8, rho=0.4)
    base = Params(window=60, prior_mode="expanding", prior_min_obs=300)
    panel = LeadLagEngine(bundle, base).run()

    r0 = run_all(panel, bundle, base)["strategies"]["PCA_SUB"].returns
    costly = Params(**{**base.to_dict(), "cost_bps": 10.0})
    r1 = run_all(panel, bundle, costly)["strategies"]["PCA_SUB"].returns
    assert r1.mean() < r0.mean()


# ---------------------------------------------------------------------------
# 4. 正則化の効果
# ---------------------------------------------------------------------------
def test_regularization_helps_under_estimation_error():
    """短い窓・強いノイズという推定誤差が大きい状況で、
    部分空間正則化ありの方が情報係数が高いこと (論文 4.4 節の主張)。"""
    bundle, _ = make_bundle(n_days=2500, seed=11, rho=0.4,
                            noise_us=1.5, noise_jp=1.5, loading_noise=0.15)
    params = Params(window=40, prior_mode="expanding", prior_min_obs=300)
    panel = LeadLagEngine(bundle, params).run()

    def mean_ic(sig: pd.DataFrame) -> float:
        out = []
        for t in sig.index:
            s = sig.loc[t].dropna()
            r = bundle.jp_oc.loc[panel.execution_date.loc[t]].reindex(s.index)
            if r.notna().sum() > 5:
                out.append(np.corrcoef(s.rank(), r.rank())[0, 1])
        return float(np.nanmean(out))

    ic_sub = mean_ic(panel.pca_sub)
    ic_plain = mean_ic(panel.pca_plain)
    assert ic_sub > ic_plain, f"IC: SUB={ic_sub:.4f} <= PLAIN={ic_plain:.4f}"


def test_estimate_c_full_handles_missing_tickers():
    """学習期間に存在しない銘柄 (XLC 等) があっても C_full が作れること。"""
    idx = pd.bdate_range("2010-01-01", periods=400)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((400, len(TICKERS))) * 0.01,
                      index=idx, columns=TICKERS)
    df["XLC"] = np.nan                       # 全欠損
    c = estimate_c_full(df, N_US, min_periods=100)
    assert np.isfinite(c.to_numpy()).all()
    np.testing.assert_allclose(np.diag(c.to_numpy()), 1.0)


def test_max_drawdown_sign_and_scale():
    r = pd.Series([0.1, -0.5, 0.2])
    assert 45.0 < max_drawdown(r) < 55.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
