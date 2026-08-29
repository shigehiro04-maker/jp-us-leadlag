"""毎朝の HTML ページ生成 (scripts/build_page.py) の検証。"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_page import (  # noqa: E402
    append_today,
    build,
    resolve_history,
    sparkline,
)

from leadlag.config import Params  # noqa: E402
from tests.synthetic import make_bundle  # noqa: E402


def _truncated(bundle, n: int):
    b = copy.deepcopy(bundle)
    for attr in ("us_cc", "jp_cc", "jp_oc", "us_close", "jp_close", "jp_open"):
        setattr(b, attr, getattr(bundle, attr).iloc[:n])
    return b


@pytest.fixture(scope="module")
def daily_pages(tmp_path_factory):
    """30 営業日ぶん連続で生成し、履歴が溜まる様子を再現する。"""
    out = tmp_path_factory.mktemp("docs")
    full, _ = make_bundle(n_days=760, seed=31, rho=0.4)
    params = Params(prior_mode="expanding", prior_min_obs=300)
    for n in range(730, 760):
        build(out, params, "./data", bundle=_truncated(full, n))
    return out, full


def test_page_is_written_and_self_contained(daily_pages):
    out, _ = daily_pages
    doc = (out / "index.html").read_text()
    assert doc.startswith("<!DOCTYPE html>")
    assert "次の東京立会日の予想" in doc
    # 外部 CDN に一切依存しないこと (機内でも開ける / CSP に引っかからない)。
    # SVG の名前空間 URL だけは取得先ではないので除外する。
    body = doc.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in body
    assert "https://" not in body
    assert (out / ".nojekyll").exists()


def test_page_contains_disclaimer(daily_pages):
    out, _ = daily_pages
    doc = (out / "index.html").read_text()
    assert "投資助言ではありません" in doc


def test_history_accumulates_and_resolves(daily_pages):
    out, _ = daily_pages
    hist = json.loads((out / "history.json").read_text())
    assert len(hist) == 30
    resolved = [h for h in hist if h.get("resolved")]
    # 最新の 1 件だけが未確定 (執行日のデータがまだ無い)
    assert len(resolved) == 29
    assert not hist[-1].get("resolved")
    for h in resolved:
        assert h["exec_date"] > h["asof"]
        assert len(h["long"]) == len(h["short"]) == 5
        assert isinstance(h["direction_correct"], bool)


def test_resolved_return_matches_actual_data(daily_pages):
    """採点結果が実際の open-to-close リターンと一致すること。"""
    out, full = daily_pages
    hist = json.loads((out / "history.json").read_text())
    h = [x for x in hist if x.get("resolved")][-1]
    row = full.jp_oc.loc[pd.Timestamp(h["exec_date"])]
    expected = row[h["long"]].mean() - row[h["short"]].mean()
    assert abs(h["ls_return"] - expected) < 1e-12


def test_rerunning_same_day_does_not_duplicate():
    hist = [{"asof": "2026-01-05", "long": [], "short": [], "market_pred": 0.0}]
    again = append_today(hist, {"asof": "2026-01-05", "long": ["A"], "short": ["B"],
                                "market_pred": 1.0})
    assert len(again) == 1
    assert again[0]["long"] == ["A"]


def test_resolve_skips_records_without_execution_data():
    bundle, _ = make_bundle(n_days=400, seed=32)
    last = str(bundle.dates[-1].date())
    hist = [{"asof": last, "long": ["1617.T"], "short": ["1618.T"],
             "market_pred": 0.0, "resolved": False}]
    out = resolve_history(hist, bundle)
    assert not out[0].get("resolved")     # 翌営業日のデータがまだ無い


def test_sparkline_is_valid_svg():
    assert sparkline([1.0, 1.02, 0.99, 1.05]).startswith("<svg")
    assert sparkline([1.0]) == ""
