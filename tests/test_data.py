"""データ整形の検証。

実運用で実際に踏んだ不具合の回帰テストを含む:
  yfinance が「まだ値の入っていない当日」を全銘柄 NaN の行として返すことがあり、
  そのままだと最新営業日がその空行になって当日の米国ショックが取れなくなる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.config import Params
from leadlag.data import build_bundle
from leadlag.engine import LeadLagEngine
from tests.synthetic import make_bundle


def _panels(dates, tickers, seed=0):
    rng = np.random.default_rng(seed)
    px = 100 * np.cumprod(
        1 + rng.standard_normal((len(dates), len(tickers))) * 0.01, axis=0
    )
    close = pd.DataFrame(px, index=dates, columns=tickers)
    opn = close.shift(1).fillna(100.0)
    return pd.concat(
        {t: pd.DataFrame({"Open": opn[t], "Close": close[t]}) for t in tickers}, axis=1
    )


def test_all_nan_trailing_row_is_dropped():
    dates = pd.bdate_range("2024-01-01", periods=200)
    us = _panels(dates, ["XLB", "XLE", "XLF"], seed=1)
    jp = _panels(dates, ["1617.T", "1618.T", "1619.T"], seed=2)

    # 米国側にだけ「値の入っていない翌日」を足す (yfinance が返してくる形)
    ghost = dates[-1] + pd.Timedelta(days=1)
    us.loc[ghost] = np.nan

    bundle = build_bundle(us, jp)
    assert ghost not in bundle.dates
    assert bundle.dates[-1] == dates[-1]
    assert bundle.us_cc.loc[bundle.dates[-1]].notna().all()


def test_close_to_close_spans_own_market_holidays():
    """相手国の休場日を挟んでも、自国の連続営業日でリターンが計算されること。"""
    dates = pd.bdate_range("2024-01-01", periods=60)
    us = _panels(dates, ["XLB", "XLE", "XLF"], seed=3)
    jp = _panels(dates.delete(10), ["1617.T", "1618.T", "1619.T"], seed=4)  # 日本が休場

    bundle = build_bundle(us, jp)
    d = bundle.dates[bundle.dates > dates[10]][0]
    prev_close = us[("XLB", "Close")].shift(1).loc[d]
    expected = us[("XLB", "Close")].loc[d] / prev_close - 1
    assert abs(bundle.us_cc.loc[d, "XLB"] - expected) < 1e-12


def test_latest_falls_back_when_newest_day_is_unusable():
    """最新日の米国データが欠けていても、1日さかのぼって予測できること。"""
    bundle, _ = make_bundle(n_days=700, seed=5, rho=0.4)
    params = Params(window=60, prior_mode="expanding", prior_min_obs=300)

    asof_full, _ = LeadLagEngine(bundle, params).latest()

    broken = bundle
    import copy

    broken = copy.deepcopy(bundle)
    broken.us_cc.iloc[-1] = np.nan          # 当日の米国リターンが全欠損

    asof_fallback, res = LeadLagEngine(broken, params).latest()
    assert asof_fallback == bundle.dates[-2]
    assert asof_fallback < asof_full
    assert np.isfinite(res.scores).all()


def test_latest_raises_when_too_many_days_missing():
    bundle, _ = make_bundle(n_days=700, seed=6, rho=0.4)
    import copy

    broken = copy.deepcopy(bundle)
    broken.us_cc.iloc[-6:] = np.nan
    with pytest.raises(RuntimeError, match="不足"):
        LeadLagEngine(broken, Params(window=60, prior_mode="expanding",
                                     prior_min_obs=300)).latest(max_lookback=5)
