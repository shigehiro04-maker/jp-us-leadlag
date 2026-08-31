"""毎朝の HTML ページ生成 (scripts/build_page.py) の検証。"""

from __future__ import annotations

import copy
import json
import re
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


EXTERNAL_RESOURCE = re.compile(
    r'(?:src|href)\s*=\s*"https?://(?!finance\.yahoo\.co\.jp/)', re.I
)


def test_page_loads_no_external_resources(daily_pages):
    """外部から読み込むファイルが無いこと (機内でも開ける / CSP に引っかからない)。

    Yahoo!ファイナンスへのリンクは「押したら遷移する」だけで、ページの表示には
    影響しないので許容する。スクリプトやスタイルの読み込みは許さない。
    """
    out, _ = daily_pages
    doc = (out / "index.html").read_text()
    assert doc.startswith("<!DOCTYPE html>")
    assert "次の東京立会日の予想" in doc
    assert (out / ".nojekyll").exists()

    hits = EXTERNAL_RESOURCE.findall(doc)
    assert not hits, f"外部リソースを読み込んでいる: {hits[:3]}"
    assert "<script src=" not in doc
    assert "<link rel=\"stylesheet\"" not in doc


def test_page_contains_disclaimer(daily_pages):
    out, _ = daily_pages
    doc = (out / "index.html").read_text()
    assert "投資助言ではありません" in doc


def test_page_frames_intraday_as_sell_bias_not_direction(daily_pages):
    """日中の地合いを「上がる/下がる」ではなく売り圧力の強弱として見せること。

    実データの検証で、このモデルの的中率は日中リターンが恒常的にマイナスである
    ことを拾っているだけで、上下の予測としては価値がないと分かったため。
    """
    out, _ = daily_pages
    doc = (out / "index.html").read_text()

    assert "日中の地合い" in doc
    assert "下押し圧力" in doc
    for banned in ("市場全体の方向", "方向予想", "方向の的中率"):
        assert banned not in doc, f"方向の当てものとしての表現が残っている: {banned}"
    assert any(s in doc for s in ("強い", "やや強い", "標準", "やや弱い", "弱い"))


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
        assert isinstance(h["market_return"], float)
        assert isinstance(h["strength"], str)


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


# ---------------------------------------------------------------------------
# 業種ETFの構成銘柄の表示
# ---------------------------------------------------------------------------
def _fake_holdings(n: int = 25, stale: bool = False) -> dict:
    from leadlag.config import JP_TICKERS

    out = {}
    for t in JP_TICKERS:
        rows = [{"code": f"{1300 + i}", "name": f"銘柄{i}", "weight": (n - i) / n * 100}
                for i in range(n)]
        rec = {"ticker": t, "fund_code": t[:4], "as_of": "2026年7月31日",
               "n": n, "holdings": rows, "error": None}
        if stale:
            rec["stale"] = True
        out[t] = rec
    return out


@pytest.fixture(scope="module")
def page_with_holdings(tmp_path_factory):
    out = tmp_path_factory.mktemp("hold")
    full, _ = make_bundle(n_days=420, seed=51, rho=0.4)
    params = Params(prior_mode="expanding", prior_min_obs=300)
    build(out, params, "./data", bundle=full, holdings=_fake_holdings())
    return (out / "index.html").read_text()


def test_shows_top_ten_holdings_and_collapses_the_rest(page_with_holdings):
    doc = page_with_holdings
    assert "銘柄0" in doc and "銘柄9" in doc          # 上位10件は展開済み
    assert "残り15銘柄" in doc                        # 残りは折りたたみ
    assert "全25銘柄" in doc
    assert "2026年7月31日現在" in doc


def test_every_sector_row_is_expandable(page_with_holdings):
    from leadlag.config import JP_TICKERS

    doc = page_with_holdings
    assert doc.count('<summary class="rowsum">') == len(JP_TICKERS)
    assert doc.count('class="holdings"') == len(JP_TICKERS)


def test_weights_are_rendered_as_percent(page_with_holdings):
    assert "100.00%" in page_with_holdings          # 先頭銘柄 (25/25)


def test_missing_holdings_degrade_gracefully(tmp_path):
    full, _ = make_bundle(n_days=420, seed=52, rho=0.4)
    params = Params(prior_mode="expanding", prior_min_obs=300)
    build(tmp_path, params, "./data", bundle=full, holdings={})
    doc = (tmp_path / "index.html").read_text()
    assert "構成銘柄を取得できませんでした" in doc
    assert "業種ランキング" in doc                    # ページ自体は壊れない


def test_stale_holdings_are_flagged(tmp_path):
    full, _ = make_bundle(n_days=420, seed=53, rho=0.4)
    params = Params(prior_mode="expanding", prior_min_obs=300)
    build(tmp_path, params, "./data", bundle=full,
          holdings=_fake_holdings(n=12, stale=True))
    doc = (tmp_path / "index.html").read_text()
    assert "前回取得ぶん" in doc


def test_holdings_page_loads_no_external_resources(page_with_holdings):
    hits = EXTERNAL_RESOURCE.findall(page_with_holdings)
    assert not hits, f"外部リソースを読み込んでいる: {hits[:3]}"


def test_holdings_source_is_credited(page_with_holdings):
    assert "野村アセットマネジメント" in page_with_holdings
    assert "組入全銘柄情報" in page_with_holdings


# ---------------------------------------------------------------------------
# Yahoo!ファイナンスへのリンク
# ---------------------------------------------------------------------------
def test_quote_url_builds_tokyo_ticker():
    from build_page import quote_url

    assert quote_url("6501") == "https://finance.yahoo.co.jp/quote/6501.T"
    assert quote_url("1625.T") == "https://finance.yahoo.co.jp/quote/1625.T"
    assert quote_url(" 2802 ") == "https://finance.yahoo.co.jp/quote/2802.T"


def test_sector_etf_code_links_to_yahoo(page_with_holdings):
    from leadlag.config import JP_TICKERS

    doc = page_with_holdings
    for t in JP_TICKERS:
        assert f'href="https://finance.yahoo.co.jp/quote/{t}"' in doc


def test_constituents_link_to_yahoo(page_with_holdings):
    doc = page_with_holdings
    # 上位10銘柄ぶんのリンク (ダミーの構成銘柄は 1300〜)
    assert 'href="https://finance.yahoo.co.jp/quote/1300.T"' in doc
    assert 'class="hrow2" href="https://finance.yahoo.co.jp/quote/' in doc


def test_links_open_in_a_new_tab_safely(page_with_holdings):
    doc = page_with_holdings
    n_links = doc.count("finance.yahoo.co.jp/quote/")
    assert n_links > 20
    # すべてのリンクに rel="noopener noreferrer" が付いていること
    assert doc.count('rel="noopener noreferrer"') == n_links
    assert doc.count('target="_blank"') == n_links


def test_tapping_the_code_does_not_toggle_the_section(page_with_holdings):
    """業種コードのリンクは折りたたみの中にあるので、開閉を止める必要がある。"""
    doc = page_with_holdings
    assert 'document.querySelectorAll(".rowsum a")' in doc
    assert "stopPropagation" in doc


def test_links_are_labelled_for_screen_readers(page_with_holdings):
    assert "をYahoo!ファイナンスで見る" in page_with_holdings


def test_constituent_rows_are_not_aria_overridden(page_with_holdings):
    """構成銘柄の行に aria-label を付けないこと。

    aria-label はリンクの中身を上書きするため、付けると読み上げ時に
    組入比率が読まれなくなる。行の内容そのものを読ませる。
    """
    doc = page_with_holdings
    for m in re.finditer(r'<a class="hrow2"[^>]*>', doc):
        assert "aria-label" not in m.group(0)
    assert "構成銘柄（タップで Yahoo!ファイナンスへ）" in doc


def test_etf_code_link_keeps_its_label(page_with_holdings):
    """ETFコードだけの表示ではリンク先が分からないので、こちらには残す。"""
    assert 'class="tk etflink"' in page_with_holdings
    assert "をYahoo!ファイナンスで見る" in page_with_holdings
