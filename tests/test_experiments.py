"""検証用スクリプトの補助関数のテスト。

ここで一番怖いのは平滑化に先読みが混ざることで、混ざると「改善した」という
結論そのものが嘘になる。そこだけは機械的に確かめておく。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_experiments import (  # noqa: E402
    cross_sectional_z, rank_flip_stats, smooth,
)


def _panel(n=50, m=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((n, m)),
        index=pd.bdate_range("2024-01-01", periods=n),
        columns=[f"s{i}" for i in range(m)],
    )


def test_smoothing_is_causal():
    """将来の値を変えても、それ以前の平滑化結果が動かないこと。"""
    p = _panel()
    for standardize in (False, True):
        a = smooth(p, halflife=5, standardize=standardize)
        q = p.copy()
        q.iloc[30:] += 100.0                      # 30 日目以降を大きく動かす
        b = smooth(q, halflife=5, standardize=standardize)
        pd.testing.assert_frame_equal(a.iloc[:30], b.iloc[:30])


def test_cross_sectional_z_normalises_scale():
    """日ごとのスケール差が消えること（順位は保たれる）。"""
    p = _panel()
    p.iloc[10] *= 1000.0                          # 1 日だけ桁が違う
    z = cross_sectional_z(p)
    assert abs(float(z.iloc[10].std(ddof=0)) - 1.0) < 1e-9
    assert abs(float(z.iloc[10].mean())) < 1e-9
    # 順位は変わらない
    assert list(z.iloc[10].rank()) == list(p.iloc[10].rank())


def test_smoothing_reduces_sign_flips():
    """符号が毎日反転する系列で、平滑化が反転率を下げること。"""
    n, m = 300, 8
    rng = np.random.default_rng(1)
    axis = rng.standard_normal(m)
    flip = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)   # 1 日おきに符号反転
    raw = pd.DataFrame(
        np.outer(flip, axis) + rng.standard_normal((n, m)) * 0.01,
        index=pd.bdate_range("2024-01-01", periods=n),
        columns=[f"s{i}" for i in range(m)],
    )
    before = rank_flip_stats(raw)
    after = rank_flip_stats(smooth(raw, halflife=5, standardize=True))
    assert before["flip_share(%)"] > 90
    assert after["flip_share(%)"] < before["flip_share(%)"]
    assert before["abs_rho_mean"] > 0.9          # 軸は同じで符号だけ反転


def test_rank_flip_stats_ignores_short_rows():
    p = _panel(n=20, m=3)                         # 銘柄数が 5 未満
    out = rank_flip_stats(p)
    assert out["N"] == 0
