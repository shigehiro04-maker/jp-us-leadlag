"""TOPIX-17 業種別 ETF の組入銘柄を取得する.

出所
----
野村アセットマネジメントが「組入全銘柄情報」として ETF ごとに公開している
Excel ファイル:

    https://www.nomura-am.co.jp/fund/monthly_holdings/{code}_brd_data.xlsx

一覧ページは https://nextfunds.jp/monthly_holdings/ 。月次更新で、銘柄コード・
銘柄名・組入比率が入っている。日次の PCF (設定・交換ポートフォリオ CSV) も
あるが、ファイル名に申込日と商品ごとの識別子が入り壊れやすいので採らない。

このモジュールはネットワークに出る唯一の箇所が `fetch_holdings` に閉じており、
取得できなかった業種は空リストを返す。呼び出し側は前回取得ぶんを使い回せる。
"""

from __future__ import annotations

import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import JP_TICKERS

BASE_URL = "https://www.nomura-am.co.jp/fund/monthly_holdings/{code}_brd_data.xlsx"
SOURCE_PAGE = "https://nextfunds.jp/monthly_holdings/"

# 列名の揺れを吸収するための候補。完全一致ではなく部分一致で拾う。
# 実ファイルの見出しは「銘柄コード_x000D_（Code）」「銘柄_x000D_（Name）」
# 「純資産比率_x000D_% of NAV」のように改行が混ざる。部分一致で拾い、
# ISIN コード列や英文名列を誤って掴まないよう avoid で除外する。
CODE_KEYS = ("銘柄コード", "コード", "code")
NAME_KEYS = ("銘柄名", "名称", "銘柄", "name")
WEIGHT_KEYS = ("純資産比率", "組入比率", "比率", "ウエイト", "ウェイト", "weight")


@dataclass
class SectorHoldings:
    ticker: str                 # 1617.T
    fund_code: str              # 1617
    as_of: str | None           # ファイルに書かれていた基準日 (取れれば)
    holdings: list[dict]        # [{"code": "2802", "name": "味の素", "weight": 8.3}, ...]
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "fund_code": self.fund_code,
            "as_of": self.as_of,
            "n": len(self.holdings),
            "holdings": self.holdings,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
def _pick_column(columns, keys, exclude=(), avoid=()) -> str | None:
    """列名の部分一致で列を選ぶ。

    keys は優先順に並べる。「銘柄コード」が「銘柄名」より先に
    銘柄名列として拾われないよう、既に使った列 (exclude) と
    紛らわしい語 (avoid) を除いて探す。
    """
    for k in keys:
        for c in columns:
            if c in exclude:
                continue
            s = str(c)
            if any(a in s for a in avoid):
                continue
            if k in s:
                return c
    return None


def _clean_as_of(text: str | None) -> str | None:
    """「2026年7月31日現在　（as of July 31, 2026）」から日付だけ取り出す。"""
    if not text:
        return None
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if m:
        return f"{int(m.group(1))}年{int(m.group(2))}月{int(m.group(3))}日"
    m = re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text)
    return m.group(0) if m else text.strip()[:40]


def _normalise_weights(rows: list[dict]) -> list[dict]:
    """比率を % に揃える。

    実ファイルの「純資産比率」は 0.2628 のような小数 (=26.28%) で入っている。
    ファイルによっては既に % の可能性もあるので、合計値を見て判断する。
    """
    vals = [r["weight"] for r in rows if r["weight"] is not None]
    if not vals:
        return rows
    total = sum(vals)
    if total <= 2.0:                 # 合計が 1 前後 → 小数表記なので % に直す
        for r in rows:
            if r["weight"] is not None:
                r["weight"] = r["weight"] * 100.0
    return rows


def parse_holdings_sheet(df: pd.DataFrame) -> tuple[list[dict], str | None]:
    """組入銘柄シートを [{code, name, weight}] に正規化する。

    ヘッダー行の位置がファイルによって違うので、コード列と銘柄名列の両方が
    見つかる行をヘッダーとみなして読み直す。
    """
    as_of = None
    header_row = None

    for i in range(min(len(df), 15)):
        row = [str(x) for x in df.iloc[i].tolist()]
        joined = " ".join(row)
        if any(k in joined for k in CODE_KEYS) and any(k in joined for k in NAME_KEYS):
            header_row = i
            break
        if as_of is None and ("基準日" in joined or "年" in joined and "月" in joined):
            for cell in row:
                if cell and cell != "nan" and ("年" in cell or "/" in cell or "-" in cell):
                    as_of = cell.strip()
                    break

    if header_row is None:
        return [], as_of

    body = df.iloc[header_row + 1 :].copy()
    body.columns = [str(x).strip() for x in df.iloc[header_row].tolist()]

    col_code = _pick_column(body.columns, CODE_KEYS, avoid=("ISIN",))
    col_name = _pick_column(body.columns, NAME_KEYS, exclude=(col_code,),
                            avoid=("コード", "code", "ISIN"))
    col_wt = _pick_column(body.columns, WEIGHT_KEYS,
                          exclude=(col_code, col_name))
    if col_code is None or col_name is None:
        return [], as_of

    out: list[dict] = []
    for _, r in body.iterrows():
        code = str(r[col_code]).strip()
        name = str(r[col_name]).strip()
        if not code or code in ("nan", "None") or not name or name == "nan":
            continue
        code = code.split(".")[0]              # 1332.0 のような読み取りを整える
        if not code[:4].isalnum():
            continue
        weight = None
        if col_wt is not None:
            try:
                weight = float(str(r[col_wt]).replace("%", "").replace(",", ""))
            except (TypeError, ValueError):
                weight = None
        out.append({"code": code[:5], "name": name, "weight": weight})

    out = _normalise_weights(out)

    # 比率の降順。比率が無いファイルは元の順序を保つ。
    if any(h["weight"] is not None for h in out):
        out.sort(key=lambda h: (h["weight"] is None, -(h["weight"] or 0.0)))
    return out, _clean_as_of(as_of)


def fetch_one(ticker: str, timeout: int = 30, retries: int = 3) -> SectorHoldings:
    """1 業種ぶんの組入銘柄を取得する。失敗しても例外は投げない。"""
    import urllib.request

    code = ticker.replace(".T", "")
    url = BASE_URL.format(code=code)
    last_err = None

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (compatible; leadlag/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, header=None)
            best: list[dict] = []
            as_of = None
            for df in sheets.values():
                rows, sheet_date = parse_holdings_sheet(df)
                if len(rows) > len(best):
                    best, as_of = rows, sheet_date
            if best:
                return SectorHoldings(ticker, code, as_of, best)
            last_err = "銘柄行が見つかりませんでした"
        except Exception as exc:                    # noqa: BLE001 通信起因は再試行
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < retries - 1:
            time.sleep(3 * (attempt + 1))

    return SectorHoldings(ticker, code, None, [], error=str(last_err)[:200])


def fetch_holdings(tickers: list[str] | None = None) -> dict[str, dict]:
    """全業種ぶんを取得して {ticker: dict} で返す。"""
    tickers = list(tickers or JP_TICKERS)
    out: dict[str, dict] = {}
    for t in tickers:
        res = fetch_one(t)
        out[t] = res.to_dict()
        status = f"{len(res.holdings)} 銘柄" if res.holdings else f"失敗 ({res.error})"
        print(f"  組入銘柄 {t}: {status}", flush=True)
        time.sleep(0.5)                             # 相手先に負荷をかけない
    return out


# ---------------------------------------------------------------------------
def load_cached(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def refresh(path: Path, tickers: list[str] | None = None,
            max_age_days: int = 7) -> dict[str, dict]:
    """必要なら取り直し、取れなかった業種は前回ぶんを使う。

    月次更新のデータなので毎日取りに行く必要はない。取得に失敗した日は
    キャッシュを維持し、ページから銘柄が消えないようにする。
    """
    cached = load_cached(path)
    fresh_enough = False
    if cached.get("_fetched_at"):
        age = time.time() - float(cached["_fetched_at"])
        fresh_enough = age < max_age_days * 86400
    if fresh_enough and all(
        cached.get(t, {}).get("holdings") for t in (tickers or JP_TICKERS)
    ):
        print(f"組入銘柄: キャッシュを使用 ({path.name})", flush=True)
        return cached

    print("組入銘柄を取得中…", flush=True)
    fetched = fetch_holdings(tickers)
    merged: dict = {"_fetched_at": time.time(), "_source": SOURCE_PAGE}
    for t, rec in fetched.items():
        if rec["holdings"]:
            merged[t] = rec
        elif cached.get(t, {}).get("holdings"):
            old = dict(cached[t])
            old["stale"] = True
            merged[t] = old
            print(f"  {t}: 取得に失敗したため前回ぶんを使用", flush=True)
        else:
            merged[t] = rec

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    return merged
