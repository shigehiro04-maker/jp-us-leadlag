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


def test_latest_direction_returns_finite_prediction():
    bundle, _ = make_bundle(n_days=800, seed=22, rho=0.4)
    res = latest_direction(bundle)
    assert np.isfinite(res["pred"])
    assert res["direction"] in ("上昇", "下落")
    assert res["asof"] == bundle.dates[-1]
