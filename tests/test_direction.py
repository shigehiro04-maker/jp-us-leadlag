"""市場方向モデル (direction.py) の検証。"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from leadlag.config import Params
from leadlag.direction import (
    latest_direction,
    market_factors,
    naive_direction,
    rolling_direction,
)
from leadlag.engine import LeadLagEngine
from tests.synthetic import make_bundle


def _frame(seed: int = 21, n_days: int = 1500):
    bundle, _ = make_bundle(n_days=n_days, seed=seed, rho=0.4)
    params = Params(window=60, prior_mode="expanding", prior_min_obs=300)
    panel = LeadLagEngine(bundle, params).run()
    return bundle, market_factors(bundle, panel.execution_date)


def test_market_factors_pairs_are_shifted_by_one_day():
    bundle, frame = _frame()
    for t in frame.index[:50]:
        exec_date = frame.loc[t, "exec_date"]
        assert exec_date > t
        expected = bundle.jp_oc.loc[exec_date].mean()
        assert abs(frame.loc[t, "jp_ew_oc_next"] - expected) < 1e-12


def test_rolling_direction_has_no_lookahead():
    """t 時点の予測が、t 以降の実現値に依存しないこと。"""
    _, frame = _frame()
    out = rolling_direction(frame, window=250, min_obs=120)

    cut = len(frame) // 2
    tampered = frame.copy()
    rng = np.random.default_rng(0)
    tampered.iloc[cut:, tampered.columns.get_loc("jp_ew_oc_next")] = (
        rng.standard_normal(len(frame) - cut) * 0.02
    )
    out2 = rolling_direction(tampered, window=250, min_obs=120)

    cut_date = frame.index[cut - 1]        # 改変を始めた位置より前の日付
    idx = out.index[out.index < cut_date].intersection(out2.index)
    assert len(idx) > 100
    np.testing.assert_allclose(
        out.loc[idx, "pred"].to_numpy(),
        out2.loc[idx, "pred"].to_numpy(),
        atol=1e-12,
    )


def test_direction_beats_coin_flip_on_synthetic():
    """合成データには本物のリードラグがあるので、的中率が 50% を有意に上回る。"""
    _, frame = _frame(n_days=2500)
    out = rolling_direction(frame)
    hit = out["correct"].mean()
    n = len(out)
    z = (hit - 0.5) / np.sqrt(0.25 / n)
    assert z > 3.0, f"的中率 {hit:.3f} (z={z:.2f}) が有意でない"


def test_naive_and_regression_agree_directionally():
    _, frame = _frame()
    reg = rolling_direction(frame)
    naive = naive_direction(frame).loc[reg.index]
    agree = (reg["pred_up"] == naive["pred_up"]).mean()
    assert agree > 0.7


def test_latest_direction_respects_asof():
    """業種シグナルがさかのぼったとき、方向モデルも同じ基準日を使うこと。"""
    bundle, _ = make_bundle(n_days=800, seed=23, rho=0.4)
    last = bundle.dates[-1]
    prev = bundle.dates[-2]
    assert latest_direction(bundle, asof=last)["asof"] == last
    assert latest_direction(bundle, asof=prev)["asof"] == prev
    assert latest_direction(bundle, asof=prev)["pred"] != latest_direction(
        bundle, asof=last
    )["pred"]


def test_latest_direction_returns_finite_prediction():
    bundle, _ = make_bundle(n_days=800, seed=22, rho=0.4)
    res = latest_direction(bundle)
    assert np.isfinite(res["pred"])
    assert res["asof"] == bundle.dates[-1]
    assert res["bias"] in ("売り優位", "買い優位")
    assert res["strength"] in ("強い", "やや強い", "標準", "やや弱い", "弱い")
    assert res["quantiles"] == sorted(res["quantiles"])


def test_strength_label_maps_prediction_onto_past_distribution():
    """予測がよりマイナス寄りなほど「売り圧力が強い」と読むこと。"""
    from leadlag.direction import strength_label

    q = [-0.004, -0.002, 0.0, 0.002]
    assert strength_label(-0.010, q) == "強い"
    assert strength_label(-0.003, q) == "やや強い"
    assert strength_label(-0.001, q) == "標準"
    assert strength_label(0.001, q) == "やや弱い"
    assert strength_label(0.010, q) == "弱い"


def test_bias_reflects_realized_intraday_sign():
    from leadlag.direction import intraday_bias

    assert intraday_bias(np.array([-0.001, -0.002, 0.0005])) == "売り優位"
    assert intraday_bias(np.array([0.001, 0.002, -0.0005])) == "買い優位"


def test_strength_uses_only_past_data():
    """強弱の判定に使う分位点が、学習ウィンドウ内の値だけで決まること。"""
    bundle, _ = make_bundle(n_days=800, seed=24, rho=0.4)
    res = latest_direction(bundle, window=250)
    assert res["n_train"] <= 250
    lo, hi = min(res["quantiles"]), max(res["quantiles"])
    assert lo < hi
