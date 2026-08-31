"""組入銘柄の取り込みの検証。

実ファイルは取得元 (野村AM) にネットワークが通る環境でしか読めないため、
ここでは想定されるレイアウトを合成して parse_holdings_sheet を検証する。
実ファイルの構造は scripts/probe_holdings.py で確認する。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.holdings import (
    SectorHoldings,
    load_cached,
    parse_holdings_sheet,
    refresh,
)


def _sheet(header_offset: int, cols: list[str], rows: list[list]) -> pd.DataFrame:
    """ヘッダーの前に説明行が入っている、よくある体裁のシートを作る。"""
    width = len(cols)
    pre = [["組入全銘柄情報"] + [None] * (width - 1),
           ["基準日", "2026年7月31日"] + [None] * (width - 2)][:header_offset]
    data = [pre[i] if i < len(pre) else [None] * width for i in range(header_offset)]
    return pd.DataFrame(data + [cols] + rows)


def test_parses_typical_layout():
    df = _sheet(
        2,
        ["No.", "銘柄コード", "銘柄名", "業種", "組入比率(%)"],
        [
            [1, "2802", "味の素", "食品", 12.34],
            [2, "2914", "日本たばこ産業", "食品", 9.87],
            [3, "2503", "キリンホールディングス", "食品", 5.5],
        ],
    )
    rows, as_of = parse_holdings_sheet(df)
    assert [r["code"] for r in rows] == ["2802", "2914", "2503"]
    assert rows[0]["name"] == "味の素"
    assert rows[0]["weight"] == 12.34
    assert as_of is not None and "2026" in as_of


def test_sorts_by_weight_descending():
    df = _sheet(
        1,
        ["コード", "銘柄名", "比率"],
        [["1111", "A", 1.0], ["2222", "B", 30.0], ["3333", "C", 10.0]],
    )
    rows, _ = parse_holdings_sheet(df)
    assert [r["code"] for r in rows] == ["2222", "3333", "1111"]


def test_keeps_order_when_no_weight_column():
    df = _sheet(0, ["銘柄コード", "銘柄名"], [["1111", "A"], ["2222", "B"]])
    rows, _ = parse_holdings_sheet(df)
    assert [r["code"] for r in rows] == ["1111", "2222"]
    assert all(r["weight"] is None for r in rows)


def test_ignores_blank_and_total_rows():
    df = _sheet(
        1,
        ["銘柄コード", "銘柄名", "組入比率"],
        [
            ["2802", "味の素", 12.3],
            [None, None, None],
            ["", "", ""],
            ["2914", "日本たばこ産業", 9.8],
        ],
    )
    rows, _ = parse_holdings_sheet(df)
    assert len(rows) == 2


def test_numeric_codes_are_normalised():
    """Excel から 1332.0 のように読めてしまうコードを整えること。"""
    df = _sheet(1, ["銘柄コード", "銘柄名", "比率"], [[1332.0, "ニッスイ", 3.2]])
    rows, _ = parse_holdings_sheet(df)
    assert rows[0]["code"] == "1332"


def test_percent_sign_is_stripped():
    df = _sheet(1, ["コード", "銘柄名", "ウェイト"], [["2802", "味の素", "12.3%"]])
    rows, _ = parse_holdings_sheet(df)
    assert rows[0]["weight"] == 12.3


def test_returns_nothing_when_header_is_absent():
    df = pd.DataFrame([["まったく関係のない表"], ["1", "2"]])
    rows, _ = parse_holdings_sheet(df)
    assert rows == []


# ---------------------------------------------------------------------------
def test_refresh_falls_back_to_cache_on_failure(tmp_path, monkeypatch):
    """取得に失敗した業種は前回ぶんを残し、ページから銘柄を消さないこと。"""
    import leadlag.holdings as H

    path = tmp_path / "holdings.json"
    path.write_text(json.dumps({
        "_fetched_at": 0,
        "1617.T": {"ticker": "1617.T", "fund_code": "1617", "as_of": "2026年6月30日",
                   "n": 1, "holdings": [{"code": "2802", "name": "味の素", "weight": 9.0}],
                   "error": None},
    }, ensure_ascii=False))

    monkeypatch.setattr(
        H, "fetch_one",
        lambda t, **kw: SectorHoldings(t, t.replace(".T", ""), None, [], error="通信失敗"),
    )
    out = refresh(path, tickers=["1617.T"])
    assert out["1617.T"]["holdings"][0]["code"] == "2802"
    assert out["1617.T"]["stale"] is True


def test_refresh_uses_cache_when_recent(tmp_path, monkeypatch):
    import leadlag.holdings as H

    path = tmp_path / "holdings.json"
    path.write_text(json.dumps({
        "_fetched_at": time.time(),
        "1617.T": {"ticker": "1617.T", "fund_code": "1617", "as_of": None, "n": 1,
                   "holdings": [{"code": "2802", "name": "味の素", "weight": 9.0}],
                   "error": None},
    }, ensure_ascii=False))

    called = []
    monkeypatch.setattr(H, "fetch_one", lambda t, **kw: called.append(t))
    out = refresh(path, tickers=["1617.T"], max_age_days=7)
    assert not called, "新しいキャッシュがあるのに取得しにいっている"
    assert out["1617.T"]["holdings"]


def test_refresh_stores_new_data(tmp_path, monkeypatch):
    import leadlag.holdings as H

    path = tmp_path / "holdings.json"
    monkeypatch.setattr(
        H, "fetch_one",
        lambda t, **kw: SectorHoldings(
            t, t.replace(".T", ""), "2026年7月31日",
            [{"code": "9999", "name": "テスト", "weight": 1.0}],
        ),
    )
    out = refresh(path, tickers=["1617.T"])
    assert out["1617.T"]["holdings"][0]["code"] == "9999"
    assert load_cached(path)["1617.T"]["holdings"][0]["code"] == "9999"


def test_load_cached_tolerates_broken_file(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{ これは JSON ではない")
    assert load_cached(p) == {}


# ---------------------------------------------------------------------------
# 実ファイル (野村AM「保有明細」シート) の体裁をそのまま再現した回帰テスト
# ---------------------------------------------------------------------------
REAL_HEADER = [
    "No.",
    "銘柄コード_x000D_\n（Code）",
    "ISINコード",
    "銘柄_x000D_\n（Name）",
    "Name",
    "株数（※）_x000D_\nNo. of shares",
    "評価金額(円）_x000D_\nValuation",
    "純資産比率_x000D_\n% of NAV",
]


def _real_sheet(rows: list[list]) -> pd.DataFrame:
    pre = [
        [1617, "NEXT FUNDS 食品（TOPIX-17）上場投信", None, None, None, None, None, None],
        [None, "2026年7月31日現在　（as of July 31, 2026）", None, None,
         "ISINコード：", "JP3046560003", None, None],
    ]
    return pd.DataFrame(pre + [REAL_HEADER] + rows)


def test_real_layout_picks_the_right_columns():
    """ISINコード列や英文名列を掴まないこと。"""
    df = _real_sheet([
        [1, 2914, "JP3726800000", "日本たばこ産業", "JAPAN TOBACCO INC.",
         141200, 1007462000, 0.262887],
        [2, 2802, "JP3119600009", "味の素", "AJINOMOTO CO.,INC.",
         112100, 556576500, 0.145233],
    ])
    rows, as_of = parse_holdings_sheet(df)
    assert len(rows) == 2
    assert rows[0]["code"] == "2914"
    assert rows[0]["name"] == "日本たばこ産業"        # 英文名ではない
    assert as_of == "2026年7月31日"                   # 余分な英字を落とす


def test_real_layout_converts_fraction_to_percent():
    """純資産比率は 0.2628 のような小数で入っているので % に直すこと。"""
    df = _real_sheet([
        [1, 2914, "JP3726800000", "日本たばこ産業", "JT", 1, 1, 0.262887],
        [2, 2802, "JP3119600009", "味の素", "AJI", 1, 1, 0.145233],
        [3, 2503, "JP3258000003", "キリンHD", "KIRIN", 1, 1, 0.591880],
    ])
    rows, _ = parse_holdings_sheet(df)
    by_code = {r["code"]: r for r in rows}
    assert abs(by_code["2914"]["weight"] - 26.2887) < 1e-3
    assert abs(sum(r["weight"] for r in rows) - 100.0) < 0.1


def test_percent_style_file_is_left_alone():
    """すでに % 表記のファイルを 100 倍しないこと。"""
    df = _real_sheet([
        [1, 2914, "JP3726800000", "日本たばこ産業", "JT", 1, 1, 26.29],
        [2, 2802, "JP3119600009", "味の素", "AJI", 1, 1, 73.71],
    ])
    rows, _ = parse_holdings_sheet(df)
    by_code = {r["code"]: r for r in rows}
    assert abs(by_code["2914"]["weight"] - 26.29) < 1e-6
    assert abs(by_code["2802"]["weight"] - 73.71) < 1e-6


def test_metadata_sheet_yields_nothing():
    """$MetaData シートを誤って銘柄表として読まないこと。"""
    meta = pd.DataFrame([
        [23245, None, None],
        ["2026-03-19 16:52:06", False, None],
        [True, None, None],
        [None, None, None],
        [7, "014_純資産総額", "実行結果"],
        [22, "220_銘柄一覧", "実行結果"],
    ])
    rows, _ = parse_holdings_sheet(meta)
    assert rows == []
