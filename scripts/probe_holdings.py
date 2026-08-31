#!/usr/bin/env python3
"""組入銘柄ファイルの実際の構造を調べる（一時的な診断用）。

この開発環境からは取得元に接続できないため、GitHub Actions 上で実行して
結果をリポジトリに残す。パーサが実ファイルに合っているかを確認したら消す。
"""

from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leadlag.config import JP_TICKERS                      # noqa: E402
from leadlag.holdings import BASE_URL, parse_holdings_sheet  # noqa: E402

out = []


def log(s: str = "") -> None:
    print(s, flush=True)
    out.append(s)


for ticker in ["1617.T", "1631.T"]:
    code = ticker.replace(".T", "")
    url = BASE_URL.format(code=code)
    log(f"===== {ticker} =====")
    log(url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
        log(f"取得 OK: {len(raw)} バイト")
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, header=None)
        log(f"シート: {list(sheets)}")
        for name, df in sheets.items():
            log(f"--- シート '{name}' shape={df.shape} ---")
            log(df.head(14).to_string(max_colwidth=24))
            rows, as_of = parse_holdings_sheet(df)
            log(f"→ パーサ結果: {len(rows)} 銘柄, 基準日={as_of}")
            if rows:
                log(f"   先頭3件: {rows[:3]}")
                log(f"   末尾2件: {rows[-2:]}")
    except Exception as exc:  # noqa: BLE001
        log(f"失敗: {type(exc).__name__}: {exc}")
    log()

log("===== 全17業種の到達性 =====")
for ticker in JP_TICKERS:
    code = ticker.replace(".T", "")
    try:
        req = urllib.request.Request(BASE_URL.format(code=code),
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, header=None)
        best, as_of = [], None
        for df in sheets.values():
            rr, dd = parse_holdings_sheet(df)
            if len(rr) > len(best):
                best, as_of = rr, dd
        top = best[0] if best else None
        log(f"{ticker}: {len(best):4d} 銘柄  基準日={as_of}  首位={top}")
    except Exception as exc:  # noqa: BLE001
        log(f"{ticker}: 失敗 {type(exc).__name__}: {exc}")

Path("debug").mkdir(exist_ok=True)
Path("debug/holdings_probe.txt").write_text("\n".join(out), encoding="utf-8")
print("wrote debug/holdings_probe.txt")
