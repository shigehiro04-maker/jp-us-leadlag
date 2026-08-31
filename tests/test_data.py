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
    assert bundle.us_last_raw == dates[-1]      # 空行は最終日として数えない


def test_raw_last_dates_expose_one_sided_lag():
    """片方の市場だけ当日ぶんが未配信のとき、それが分かること。"""
    dates = pd.bdate_range("2024-01-01", periods=100)
    us = _panels(dates.delete(-1), ["XLB", "XLE", "XLF"], seed=7)   # 米国だけ1日遅れ
    jp = _panels(dates, ["1617.T", "1618.T", "1619.T"], seed=8)

    bundle = build_bundle(us, jp)
    assert bundle.us_last_raw == dates[-2]
    assert bundle.jp_last_raw == dates[-1]
    assert bundle.dates[-1] == dates[-2]


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


def test_corrupt_price_spike_is_removed():
    """無料データに混じる異常なリターンが除去され、記録されること。

    実運用で 1629.T の年率リターンが 3339% と表示される破損に遭遇したため、
    その回帰テスト。
    """
    dates = pd.bdate_range("2024-01-01", periods=120)
    us = _panels(dates, ["XLB", "XLE", "XLF"], seed=11)
    jp = _panels(dates, ["1617.T", "1618.T", "1629.T"], seed=12)

    # 1629.T に「分割の調整漏れ」相当の跳ねを入れる
    jp.loc[dates[60], ("1629.T", "Close")] *= 10.0
    jp.loc[dates[60], ("1629.T", "Open")] *= 10.0

    bundle = build_bundle(jp_panel=jp, us_panel=us)
    q = bundle.quality_report
    assert len(q) >= 2                                  # 跳ねと戻りの両方
    assert set(q["ticker"]) == {"1629.T"}
    assert not np.isfinite(bundle.jp_cc.loc[dates[60], "1629.T"])

    # 健全な銘柄には手を付けない
    assert bundle.jp_cc["1617.T"].notna().sum() == len(dates) - 1
    # 年率リターンが常識的な範囲に収まること
    assert abs(bundle.summary().loc["1629.T", "Ret(%)"]) < 200


def test_sanitize_keeps_large_but_plausible_moves():
    from leadlag.data import sanitize_returns

    df = pd.DataFrame({"A": [0.0, -0.35, 0.40, 0.0]})
    cleaned, rep = sanitize_returns(df, max_abs=0.5, label="t")
    assert cleaned.notna().all().all()      # ±40% は残す
    assert rep.empty


def test_scattered_gaps_do_not_erase_the_window():
    """散発的な欠損で評価日がまるごと消えないこと。

    実データで 2025年10月に日本側の値が数銘柄ぶん抜けており、「ウィンドウ内に
    1 日でも欠損があればその銘柄を除外する」という作りのせいで、10月から
    翌2月まで約3か月ぶんのシグナルが 1 日も生成されないという不具合が出た。
    """
    from leadlag.backtest import run_all

    bundle, _ = make_bundle(n_days=900, seed=41, rho=0.4)
    params = Params(window=60, prior_mode="expanding", prior_min_obs=300)
    full = LeadLagEngine(bundle, params).run()

    import copy

    holed = copy.deepcopy(bundle)
    rng = np.random.default_rng(7)
    # 実データの 2025年10月と同じ程度（1日あたり 2 銘柄前後）の欠損を
    # 20 営業日ぶん、日替わりで別々の銘柄に入れる
    for day in range(600, 620):
        for tk in rng.choice(holed.jp_cc.columns, size=2, replace=False):
            holed.jp_cc.iloc[day, holed.jp_cc.columns.get_loc(tk)] = np.nan

    holed_panel = LeadLagEngine(holed, params).run()

    lost = len(full.pca_sub) - len(holed_panel.pca_sub)
    assert lost <= 5, f"{lost} 日ぶんの評価日が失われた（欠損は20日ぶんのみ）"

    # 欠損期間の直後にもシグナルが出ていること
    after = holed.dates[640]
    assert (holed_panel.pca_sub.index == after).any()

    # 損益も計算できること
    res = run_all(holed_panel, holed, params)
    assert np.isfinite(res["strategies"]["PCA_SUB"].returns).all()


def test_signal_tolerates_nan_in_window():
    from leadlag.signal import compute_signal

    rng = np.random.default_rng(3)
    us = rng.standard_normal((60, 4)) * 0.01
    jp = rng.standard_normal((60, 5)) * 0.01
    us[10, 1] = np.nan
    jp[22, 3] = np.nan
    c_full = np.eye(9)
    res = compute_signal(us, jp, rng.standard_normal(4) * 0.01,
                         ["XLB", "XLE", "XLF", "XLI"],
                         ["1617.T", "1618.T", "1619.T", "1620.T", "1621.T"],
                         c_full, lam=0.9, n_factors=3)
    assert np.isfinite(res.scores).all()
    assert np.isfinite(res.b_matrix).all()
