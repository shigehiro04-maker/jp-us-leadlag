"""SCS などの追加モデル (leadlag/models.py) と、日次ページへの組み込みの検証。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_page import append_today, next_tokyo_session  # noqa: E402

from leadlag.config import JP_TICKERS, US_TICKERS  # noqa: E402
from leadlag.models import SCS_MAP, scs_scores, select_long_short  # noqa: E402


def test_scs_map_covers_all_sectors_with_valid_us_etfs():
    assert set(SCS_MAP) == set(JP_TICKERS)
    assert set(SCS_MAP.values()) <= set(US_TICKERS)


def test_scs_scores_copy_the_matched_us_return():
    us = pd.Series({t: 0.0 for t in US_TICKERS})
    us["XLV"] = 0.0103
    us["XLI"] = -0.0218
    s = scs_scores(us, JP_TICKERS)
    assert s["1621.T"] == 0.0103                      # 医薬品 ← ヘルスケア
    for j in ("1619.T", "1624.T", "1628.T", "1629.T"):
        assert s[j] == -0.0218                        # 資本財に写る 4 業種


def test_scs_missing_us_etf_is_neutral():
    us = pd.Series({t: 0.01 for t in US_TICKERS if t != "XLC"})
    us["XLC"] = float("nan")
    assert scs_scores(us, JP_TICKERS)["1626.T"] == 0.0


def test_ties_are_broken_by_sector_code():
    # 2026-10-07 の米国終値（実データ）で、XLI に写る 4 業種の同点を業種コード順に裁く
    us = pd.Series({"XLB": -.01508, "XLC": -.00349, "XLE": -.00612, "XLF": -.00481,
                    "XLI": -.0218, "XLK": -.00302, "XLP": -.00122, "XLRE": -.0129,
                    "XLU": -.00024, "XLV": .01029, "XLY": -.00322})
    lo, sh = select_long_short(scs_scores(us, JP_TICKERS), 0.3)
    assert lo == ["1621.T", "1627.T", "1617.T", "1625.T", "1622.T"]
    assert sh == ["1623.T", "1619.T", "1624.T", "1628.T", "1629.T"]


def test_pre_open_prediction_is_not_overwritten_after_open():
    hist = [{"asof": "2026-10-08", "long": ["A"], "short": ["B"], "pre_open": True}]
    out = append_today(hist, {"asof": "2026-10-08", "long": ["C"], "short": ["D"],
                              "pre_open": False})
    assert out[0]["long"] == ["A"]
    # 寄付き前どうしなら最新で置き換える（7:30 の後の 8:30 の追いかけ実行）
    out = append_today(hist, {"asof": "2026-10-08", "long": ["C"], "short": ["D"],
                              "pre_open": True})
    assert out[0]["long"] == ["C"]


def test_next_session_skips_weekend_and_holidays():
    assert str(next_tokyo_session(pd.Timestamp("2026-10-09"))) == "2026-10-13"  # 土日+スポーツの日
    assert str(next_tokyo_session(pd.Timestamp("2026-10-07"))) == "2026-10-08"
    assert str(next_tokyo_session(pd.Timestamp("2026-12-30"))) == "2027-01-04"


def test_history_records_all_models_and_scores_them(tmp_path):
    from build_page import build
    from leadlag.config import Params
    from tests.synthetic import make_bundle
    import copy

    full, _ = make_bundle(n_days=420, seed=7, rho=0.4)
    params = Params(prior_mode="expanding", prior_min_obs=300)
    for n in (410, 411, 412):
        b = copy.deepcopy(full)
        for a in ("us_cc", "jp_cc", "jp_oc", "us_close", "jp_close", "jp_open"):
            setattr(b, a, getattr(full, a).iloc[:n])
        build(tmp_path, params, "./data", bundle=b, holdings={})
    hist = json.loads((tmp_path / "history.json").read_text())
    first = hist[0]
    assert first["main"] == "SCS"
    assert {"SCS", "PCA_SUB", "合成"} <= set(first["models"])
    assert first["resolved"]
    assert "ls_return" in first["models"]["SCS"]
    doc = (tmp_path / "index.html").read_text()
    assert "モデル別の予想" in doc and "モデル別の成績" in doc
