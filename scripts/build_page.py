#!/usr/bin/env python3
"""毎朝の予測を iPhone 向けの 1 枚の HTML にして docs/index.html に書き出す。

GitHub Actions から毎営業日 22:00 UTC (= 翌 07:00 JST) に実行される想定。
外部 CDN に依存しない自己完結の HTML を生成するため、機内モードでなければ
どこからでも開ける。

  python scripts/build_page.py --outdir docs
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leadlag.backtest import cross_section_weights          # noqa: E402
from leadlag.config import Params, display_name             # noqa: E402
from leadlag.data import load_bundle                        # noqa: E402
from leadlag.direction import latest_direction              # noqa: E402
from leadlag.engine import LeadLagEngine                    # noqa: E402
from leadlag.holdings import SOURCE_PAGE                     # noqa: E402
from leadlag.holdings import refresh as refresh_holdings     # noqa: E402

JST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------------------
# 履歴の管理
# ---------------------------------------------------------------------------
def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def resolve_history(history: list[dict], bundle) -> list[dict]:
    """執行日のデータが揃った過去の予測に、実現結果を書き込む。

    執行日は「基準日より後の最初の東京立会日」。判定に日米共通営業日ではなく
    日本の営業日を使うのは、採点に要るのが日本の寄付きと大引けだけだからで、
    同じ日の米国データを待つ理由がない。共通営業日で見ていると米国側の配信が
    遅れた日に採点が止まる。日本が休場で米国だけ開いていた日も正しく飛ばせる。

    なお、これは日々の採点だけの話。バックテスト側は論文の定義どおり
    共通営業日のままにしてある。
    """
    jp_oc = bundle.jp_oc_all if bundle.jp_oc_all is not None else bundle.jp_oc
    dates = jp_oc.index
    for rec in history:
        if rec.get("resolved"):
            continue
        asof = pd.Timestamp(rec["asof"])
        later = dates[dates > asof]
        if len(later) == 0:
            continue
        exec_date = later[0]
        row = jp_oc.loc[exec_date]
        longs = [t for t in rec["long"] if t in row.index and np.isfinite(row[t])]
        shorts = [t for t in rec["short"] if t in row.index and np.isfinite(row[t])]
        if not longs or not shorts:
            continue
        mkt = float(row.dropna().mean())
        ls = float(row[longs].mean() - row[shorts].mean())
        # 上下の的中は記録しない。日中リターンは恒常的にマイナスで、符号を
        # 当てたかどうかは戦略の良し悪しをほとんど表さないため
        # (詳細は leadlag/direction.py の注記)。実現値そのものを残す。
        rec.update(
            {
                "resolved": True,
                "exec_date": str(exec_date.date()),
                "ls_return": ls,
                "market_return": mkt,
            }
        )
    return history


def append_today(history: list[dict], rec: dict) -> list[dict]:
    history = [h for h in history if h["asof"] != rec["asof"]]
    history.append(rec)
    history.sort(key=lambda h: h["asof"])
    return history[-500:]


# ---------------------------------------------------------------------------
# 画面部品
# ---------------------------------------------------------------------------
def sparkline(values: list[float], width: int = 300, height: int = 48) -> str:
    """累積リターンの簡易スパークライン (インライン SVG)。"""
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pts = [
        (width * i / (len(values) - 1), height - (v - lo) / span * (height - 6) - 3)
        for i, v in enumerate(values)
    ]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    zero_y = ""
    if lo < 0 < hi:
        y0 = height - (0 - lo) / span * (height - 6) - 3
        zero_y = f'<line x1="0" y1="{y0:.1f}" x2="{width}" y2="{y0:.1f}" class="zero"/>'
    return (
        f'<svg viewBox="0 0 {width} {height}" class="spark" preserveAspectRatio="none" '
        f'role="img" aria-label="累積リターン">{zero_y}'
        f'<path d="{path}" fill="none" class="sparkline"/></svg>'
    )


def bar(value: float, vmax: float) -> str:
    """シグナルの強さを示す横バー。"""
    pct = 0.0 if vmax <= 0 else min(abs(value) / vmax, 1.0) * 50.0
    side = "pos" if value >= 0 else "neg"
    style = (
        f"left:50%;width:{pct:.1f}%" if value >= 0 else f"right:50%;width:{pct:.1f}%"
    )
    return f'<span class="bar"><i class="{side}" style="{style}"></i></span>'


# Yahoo!ファイナンス (日本) の個別銘柄ページ。東証銘柄は「コード.T」で引ける。
YAHOO_QUOTE = "https://finance.yahoo.co.jp/quote/{code}"


def quote_url(code: str) -> str:
    """東証の銘柄コードから Yahoo!ファイナンスの個別ページ URL を作る。"""
    code = str(code).strip()
    if not code.endswith(".T"):
        code = f"{code}.T"
    return YAHOO_QUOTE.format(code=quote(code, safe=".-"))


def holdings_html(rec: dict | None, top_n: int = 10) -> str:
    """1 業種ぶんの構成銘柄。上位 top_n を出し、残りはさらに折りたたむ。"""
    if not rec or not rec.get("holdings"):
        return '<p class="hnote">構成銘柄を取得できませんでした</p>'

    rows = rec["holdings"]

    def one(h: dict) -> str:
        w = f'{h["weight"]:.2f}%' if h.get("weight") is not None else "—"
        code = html.escape(str(h["code"]))
        name = html.escape(str(h["name"]))
        # aria-label は付けない。付けると行の中身 (コード・銘柄名・組入比率) が
        # 読み上げから消え、比率が読まれなくなるため。リンク先は
        # 一覧の見出しで「Yahoo!ファイナンスへ」と一度伝えれば足りる。
        return (f'<li><a class="hrow2" href="{quote_url(h["code"])}"'
                f' target="_blank" rel="noopener noreferrer">'
                f'<span class="hc">{code}</span>'
                f'<span class="hn">{name}</span>'
                f'<span class="hw">{w}</span></a></li>')

    top = "".join(one(h) for h in rows[:top_n])
    rest = rows[top_n:]
    more = ""
    if rest:
        more = (f'<details class="more"><summary>残り{len(rest)}銘柄</summary>'
                f'<ul class="hlist">{"".join(one(h) for h in rest)}</ul></details>')
    stale = ' <span class="staleflag">前回取得ぶん</span>' if rec.get("stale") else ""
    asof = html.escape(str(rec.get("as_of") or "基準日不明"))
    return (f'<p class="hlead">構成銘柄（タップで Yahoo!ファイナンスへ）</p>'
            f'<ul class="hlist">{top}</ul>{more}'
            f'<p class="hnote">{asof}現在・全{len(rows)}銘柄{stale}</p>')


def row_html(ticker: str, sig: float, vmax: float, tag: str,
             holdings: dict | None = None) -> str:
    cls = {"ロング": "long", "ショート": "short"}.get(tag, "flat")
    # λ が大きいとシグナルが 8 業種に集中し、残りは実質ノイズになる。
    # 上位/下位に選ばれていても弱いものは見た目で分かるようにする。
    weak = ' <span class="weak" title="シグナルが弱く実質ノイズです">弱</span>' \
        if vmax > 0 and abs(sig) < 0.2 * vmax else ""
    return f"""      <li class="rowwrap">
        <details class="sect">
          <summary class="rowsum">
            <span class="row {cls}">
              <a class="tk etflink" href="{quote_url(ticker)}"
                 target="_blank" rel="noopener noreferrer"
                 aria-label="{html.escape(ticker)}をYahoo!ファイナンスで見る"
                 >{html.escape(ticker)}</a>
              <span class="nm">{html.escape(display_name(ticker))}{weak}</span>
              {bar(sig, vmax)}
              <span class="sg">{sig:+.3f}</span>
              <span class="chev" aria-hidden="true">›</span>
            </span>
          </summary>
          <div class="holdings">{holdings_html(holdings)}</div>
        </details>
      </li>"""


# ---------------------------------------------------------------------------
def build(outdir: Path, params: Params, cache: str, synthetic: int = 0,
          bundle=None, holdings: dict | None = None) -> Path:
    bundle_is_synthetic = bundle is not None
    if bundle is not None:
        pass
    elif synthetic:
        from tests.synthetic import make_bundle

        bundle, _ = make_bundle(n_days=synthetic, seed=0, rho=0.4)
        params = Params(**{**params.to_dict(), "prior_mode": "expanding"})
    else:
        bundle = load_bundle(
            start=params.start, end=params.end, cache_dir=cache, refresh=True
        )
    engine = LeadLagEngine(bundle, params)
    asof, res = engine.latest()

    sig = pd.Series(res.scores, index=res.jp_tickers).sort_values(ascending=False)
    w = cross_section_weights(sig, params.quantile)
    longs = [t for t in sig.index if w.get(t, 0) > 0]
    shorts = [t for t in sig.index if w.get(t, 0) < 0]

    md = latest_direction(bundle, asof=asof)

    # 業種別 ETF の構成銘柄 (月次更新なので普段はキャッシュを使う)
    if holdings is None:
        holdings = ({} if (synthetic or bundle_is_synthetic)
                    else refresh_holdings(outdir / "holdings.json"))

    # 次の東京立会日 (概算: 翌営業日。祝日は当日になって確定する)
    next_session = (asof + pd.offsets.BDay(1)).date()

    hist_path = outdir / "history.json"
    history = resolve_history(load_history(hist_path), bundle)
    history = append_today(
        history,
        {
            "asof": str(asof.date()),
            "long": longs,
            "short": shorts,
            "market_pred": float(md["pred"]),
            "strength": md["strength"],
            "us_ew": float(md["us_ew_cc"]),
            "signals": {t: float(sig[t]) for t in sig.index},
            "resolved": False,
        },
    )
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=1))

    done = [h for h in history if h.get("resolved")]
    recent = done[-60:]
    cum = [1.0]
    intraday = []
    for h in recent:
        cum.append(cum[-1] * (1 + h["ls_return"]))
        intraday.append(h["market_return"])
    ls_total = (cum[-1] - 1) * 100 if len(cum) > 1 else float("nan")
    intraday_avg = 1e4 * float(np.mean(intraday)) if intraday else float("nan")

    vmax = float(np.abs(sig.to_numpy()).max()) or 1.0
    us_z = pd.Series(res.z_us, index=res.us_tickers).sort_values(ascending=False)

    # --- 日中の地合い: 上下の当てものではなく、売り圧力の強弱として見せる ---
    strength = md["strength"]
    bias = md["bias"]
    meter_filled = {"強い": 5, "やや強い": 4, "標準": 3, "やや弱い": 2, "弱い": 1}[strength]
    meter = "".join(
        f'<i class="{"on" if k < meter_filled else "off"}"></i>' for k in range(5)
    )
    strength_cls = {"強い": "st5", "やや強い": "st4", "標準": "st3",
                    "やや弱い": "st2", "弱い": "st1"}[strength]

    long_rows = "\n".join(row_html(t, sig[t], vmax, "ロング", holdings.get(t)) for t in longs)
    short_rows = "\n".join(row_html(t, sig[t], vmax, "ショート", holdings.get(t)) for t in shorts)
    mid = [t for t in sig.index if t not in longs and t not in shorts]
    mid_rows = "\n".join(row_html(t, sig[t], vmax, "-", holdings.get(t)) for t in mid)
    us_rows = "\n".join(
        f'      <li class="row plain {"long" if v >= 0 else "short"}">'
        f'<span class="tk">{t}</span>'
        f'<span class="nm">{html.escape(display_name(t))}</span>'
        f'{bar(float(v), float(np.abs(us_z).max()) or 1.0)}'
        f'<span class="sg">{v:+.2f}</span></li>'
        for t, v in us_z.items()
    )

    hist_rows = "\n".join(
        f'      <li class="hrow"><span class="hd">{h["exec_date"][5:]}</span>'
        f'<span class="hv {"pos" if h["ls_return"] >= 0 else "neg"}">'
        f'{h["ls_return"]*100:+.2f}%</span>'
        f'<span class="hv {"pos" if h["market_return"] >= 0 else "neg"}">'
        f'{h["market_return"]*1e4:+.0f}bp</span></li>'
        for h in reversed(recent[-15:])
    )

    # データの鮮度。提供元が当日ぶんをまだ埋めていないと asof が 1 日古くなるので、
    # どの日付まで取得できていたかをページと実行ログの両方に残す。
    us_last = (bundle.us_last_raw or bundle.dates[-1]).date()
    jp_last = (bundle.jp_last_raw or bundle.dates[-1]).date()
    lag = "" if str(us_last) == str(asof.date()) else "（提供元の更新待ちで1日前を使用）"
    print(f"データ最終日: US {us_last} / JP {jp_last} / 使用した米国終値 {asof.date()} {lag}")

    n_hold = sum(len(v.get("holdings") or []) for v in holdings.values()
                 if isinstance(v, dict))
    holdings_src = (f"・全{n_hold}銘柄を収録" if n_hold else "・未取得")
    print(f"構成銘柄: {n_hold} 銘柄を収録")

    generated = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")
    html_doc = PAGE.format(
        asof=asof.date(),
        next_session=next_session,
        bias=bias,
        strength=strength,
        strength_cls=strength_cls,
        meter=meter,
        pred_bp=md["pred"] * 1e4,
        base_bp=md["base_mean"] * 1e4,
        base_down=md["base_share_down"] * 100,
        n_train=md["n_train"],
        resid=md["resid_sd"] * 100,
        us_ew=md["us_ew_cc"] * 100,
        long_rows=long_rows,
        short_rows=short_rows,
        mid_rows=mid_rows,
        us_rows=us_rows,
        hist_rows=hist_rows or '<li class="hrow"><span class="hd">まだ履歴がありません</span></li>',
        spark=sparkline(cum) if len(cum) > 2 else "",
        intraday_avg=f"{intraday_avg:+.0f}bp" if intraday else "—",
        ls_total=f"{ls_total:+.1f}%" if len(cum) > 1 else "—",
        n_hist=len(recent),
        f_scores=", ".join(f"f{i+1}={v:+.2f}" for i, v in enumerate(res.factor_scores)),
        generated=generated,
        asof_iso=asof.date().isoformat(),
        next_iso=next_session.isoformat(),
        us_last=us_last,
        jp_last=jp_last,
        holdings_src=holdings_src,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / ".nojekyll").write_text("")
    path = outdir / "index.html"
    path.write_text(html_doc, encoding="utf-8")
    print(f"wrote {path} (asof {asof.date()}, {len(recent)} resolved history rows)")
    return path


# ---------------------------------------------------------------------------
PAGE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="日米リードラグ">
<meta name="theme-color" content="#0b1020">
<title>日米リードラグ {asof}</title>
<link rel="apple-touch-icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 180 180'%3E%3Crect width='180' height='180' rx='40' fill='%230b1020'/%3E%3Cpath d='M28 120 L64 84 L96 104 L152 52' stroke='%2360a5fa' stroke-width='12' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='152' cy='52' r='13' fill='%23f87171'/%3E%3C/svg%3E">
<style>
:root {{
  --bg:#f6f7fb; --card:#fff; --fg:#14161c; --muted:#6b7280; --line:#e5e7eb;
  --up:#0d9488; --down:#dc2626; --accent:#2563eb;
  --safe-t:env(safe-area-inset-top); --safe-b:env(safe-area-inset-bottom);
}}
@media (prefers-color-scheme:dark) {{
  :root {{ --bg:#0b1020; --card:#151a2d; --fg:#e8eaf2; --muted:#9aa3b8; --line:#252b42;
    --up:#2dd4bf; --down:#f87171; --accent:#60a5fa; }}
}}
* {{ box-sizing:border-box; -webkit-tap-highlight-color:transparent; }}
body {{
  margin:0; background:var(--bg); color:var(--fg);
  font:16px/1.5 -apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans JP",sans-serif;
  padding:calc(var(--safe-t) + 12px) 12px calc(var(--safe-b) + 28px);
  max-width:560px; margin-inline:auto;
}}
header {{ margin:4px 4px 14px; }}
h1 {{ font-size:15px; font-weight:600; margin:0; color:var(--muted); letter-spacing:.02em; }}
.date {{ font-size:26px; font-weight:700; margin:2px 0 0; letter-spacing:-.01em; }}
.sub {{ font-size:13px; color:var(--muted); margin-top:3px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:16px;
  padding:16px; margin-bottom:12px; }}
.card h2 {{ font-size:12px; font-weight:600; color:var(--muted); margin:0 0 12px;
  text-transform:uppercase; letter-spacing:.08em; }}
.dir {{ display:flex; align-items:baseline; gap:10px; }}
.dir .word {{ font-size:38px; font-weight:800; letter-spacing:-.02em; line-height:1; }}
.dir .pct {{ font-size:19px; font-weight:600; }}
.up {{ color:var(--up); }} .down {{ color:var(--down); }}
.flat-dir {{ color:var(--muted); }}
.word.bias {{ font-size:34px; }}
.strengthline {{ display:flex; align-items:baseline; gap:10px; margin-top:8px; }}
.strengthline .lbl {{ font-size:12px; color:var(--muted); }}
.strengthline .pct {{ font-size:22px; font-weight:700; }}
.pct.st5 {{ color:var(--down); }} .pct.st4 {{ color:var(--down); opacity:.85; }}
.pct.st3 {{ color:var(--muted); }} .pct.st2 {{ color:var(--fg); opacity:.75; }}
.pct.st1 {{ color:var(--up); }}
.meter {{ display:flex; gap:5px; margin:12px 0 4px; }}
.meter i {{ flex:1; height:7px; border-radius:3px; }}
.meter i.on {{ background:var(--down); }}
.meter i.off {{ background:var(--line); }}
.meta.small {{ font-size:12px; margin-top:10px; }}
.weak {{ font-size:10px; color:var(--muted); border:1px solid var(--line);
  border-radius:4px; padding:0 3px; margin-left:5px; vertical-align:1px; }}
.meta {{ font-size:13px; color:var(--muted); margin-top:10px; }}
.meta b {{ color:var(--fg); font-weight:600; }}
ul {{ list-style:none; padding:0; margin:0; }}
.rowwrap {{ border-bottom:1px solid var(--line); }}
.rowwrap:last-child {{ border-bottom:0; }}
.rowsum {{ display:block; cursor:pointer; list-style:none; }}
.rowsum::-webkit-details-marker {{ display:none; }}
.rowsum::marker {{ content:""; }}
.row {{ display:grid; grid-template-columns:52px 1fr 68px 50px 12px;
  align-items:center; gap:8px; padding:9px 0; font-size:14px; }}
.chev {{ color:var(--muted); font-size:11px; text-align:right;
  transition:transform .15s; }}
.row.plain {{ grid-template-columns:52px 1fr 76px 52px;
  border-bottom:1px solid var(--line); }}
.row.plain:last-child {{ border-bottom:0; }}
.sect[open] .chev {{ transform:rotate(90deg); }}
@media (prefers-reduced-motion:reduce) {{ .chev {{ transition:none; }} }}
.sect[open] .rowsum .row {{ background:var(--line); border-radius:6px;
  padding-inline:6px; margin-inline:-6px; }}
.holdings {{ padding:2px 0 12px 6px; }}
.hlist {{ list-style:none; padding:0; margin:2px 0 0; }}
.hrow2 {{ display:grid; grid-template-columns:44px 1fr 60px; gap:8px;
  padding:4px 0; font-size:13px; }}
.hrow2 {{ color:inherit; text-decoration:none; -webkit-touch-callout:default; }}
.hrow2:active {{ background:var(--line); border-radius:5px; }}
.hrow2 .hn {{ color:var(--accent); }}
.hc {{ font-variant-numeric:tabular-nums; color:var(--muted); font-size:12px; }}
.hn {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.hw {{ text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }}
.hlead {{ font-size:11px; color:var(--muted); margin:6px 0 2px; }}
.hnote {{ font-size:11px; color:var(--muted); margin:8px 0 0; }}
.staleflag {{ color:var(--down); }}
details.more > summary {{ font-size:12px; color:var(--accent); padding:8px 0 4px;
  cursor:pointer; list-style:none; }}
details.more > summary::-webkit-details-marker {{ display:none; }}
details.more > summary::after {{ content:" ›"; }}
details.more[open] > summary::after {{ content:" ⌄"; }}
.tk {{ font-variant-numeric:tabular-nums; font-size:12px; color:var(--muted); }}
a.etflink {{ color:var(--accent); text-decoration:none; font-weight:600;
  padding:11px 5px; margin:-11px -5px; border-radius:4px; }}
a.etflink:active {{ background:var(--line); }}
a:focus-visible, .rowsum:focus-visible {{ outline:2px solid var(--accent);
  outline-offset:2px; border-radius:4px; }}
.nm {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.sg {{ text-align:right; font-variant-numeric:tabular-nums; font-size:13px; }}
.row.long .sg {{ color:var(--up); font-weight:600; }}
.row.short .sg {{ color:var(--down); font-weight:600; }}
.row.flat {{ opacity:.55; }}
.bar {{ position:relative; display:block; height:6px; background:var(--line);
  border-radius:3px; }}
.bar i {{ position:absolute; top:0; height:6px; border-radius:3px; }}
.bar i.pos {{ background:var(--up); }} .bar i.neg {{ background:var(--down); }}
.label {{ font-size:12px; font-weight:700; letter-spacing:.06em; margin:2px 0 8px; }}
.label.l {{ color:var(--up); }} .label.s {{ color:var(--down); }}
details {{ margin-top:10px; }}
details > summary:not(.rowsum) {{ cursor:pointer; font-size:13px;
  color:var(--accent); list-style:none; padding:8px 0; }}
details > summary::-webkit-details-marker {{ display:none; }}
details > summary:not(.rowsum)::after {{ content:" ›"; }}
details[open] > summary:not(.rowsum)::after {{ content:" ⌄"; }}
.stats {{ display:flex; gap:18px; margin-bottom:10px; }}
.stat .v {{ font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; }}
.stat .k {{ font-size:11px; color:var(--muted); }}
.spark {{ width:100%; height:48px; display:block; margin:6px 0 2px; }}
.sparkline {{ stroke:var(--accent); stroke-width:2; vector-effect:non-scaling-stroke; }}
.zero {{ stroke:var(--line); stroke-width:1; vector-effect:non-scaling-stroke; }}
.hrow {{ display:grid; grid-template-columns:56px 1fr 68px; padding:5px 0;
  font-size:13px; border-bottom:1px solid var(--line); }}
.hd {{ color:var(--muted); font-variant-numeric:tabular-nums; }}
.hv {{ text-align:right; font-variant-numeric:tabular-nums; }}
.hv.pos {{ color:var(--up); }} .hv.neg {{ color:var(--down); }}
.hm {{ text-align:right; color:var(--muted); }}
.stale {{ display:none; background:var(--down); color:#fff; border-radius:12px;
  padding:10px 14px; font-size:13px; margin-bottom:12px; }}
footer {{ font-size:11px; color:var(--muted); line-height:1.6; margin:18px 4px 0; }}
</style>
</head>
<body>
<div class="stale" id="stale"></div>

<header>
  <h1>次の東京立会日の予想</h1>
  <p class="date">{next_session}</p>
  <p class="sub">NY {asof} 終値時点の情報にもとづく（寄付き→大引け）</p>
</header>

<section class="card">
  <h2>日中の地合い</h2>
  <div class="dir">
    <span class="word bias">{bias}</span>
  </div>
  <div class="strengthline">
    <span class="lbl">下押し圧力</span>
    <span class="pct {strength_cls}">{strength}</span>
  </div>
  <div class="meter">{meter}</div>
  <p class="meta">
     翌日の東京EW日中の予測 <b>{pred_bp:+.0f}bp</b> ・
     過去{n_train}営業日の日中平均 <b>{base_bp:+.0f}bp</b>（うち下落 {base_down:.0f}%）・
     当日の米国11業種 等ウェイト <b>{us_ew:+.2f}%</b>
  </p>
  <p class="meta small">
     日本株のリターンはほぼ夜間（前日大引け→翌日寄付き）に発生し、日中は恒常的に
     マイナスです。このモデルは上がるか下がるかを当てるものではなく、
     <b>その下押し圧力が過去と比べて強いか弱いか</b>を見るものです。
     強弱は過去{n_train}営業日の予測値の分布の五分位で判定しています。
     「弱い」は上昇に転じるという意味ではありません。
  </p>
</section>

<section class="card">
  <h2>業種ランキング（相対の強弱）</h2>
  <p class="label l">▲ ロング（強いと予想）</p>
  <ul>
{long_rows}
  </ul>
  <p class="label s" style="margin-top:14px">▼ ショート（弱いと予想）</p>
  <ul>
{short_rows}
  </ul>
  <details>
    <summary>中間の業種を表示</summary>
    <ul>
{mid_rows}
    </ul>
  </details>
  <details>
    <summary>米国業種の当日ショック（窓内標準化 z）</summary>
    <ul>
{us_rows}
    </ul>
    <p class="meta">共通ファクター {f_scores}</p>
  </details>
</section>

<section class="card">
  <h2>直近の実績（ロングショート）</h2>
  <div class="stats">
    <div class="stat"><div class="v">{ls_total}</div><div class="k">直近{n_hist}営業日 累積</div></div>
    <div class="stat"><div class="v">{intraday_avg}</div><div class="k">日中EWの平均<br>（1日あたり）</div></div>
  </div>
  {spark}
  <details>
    <summary>日別の内訳（左=ロングショート、右=日中EW）</summary>
    <ul>
{hist_rows}
    </ul>
  </details>
  <p class="meta">実際に運用した記録ではなく、毎朝このページが出した予想を
     その日の実現リターンで後から採点したものです。取引コストは含みません。<br>
     <b>検証メモ:</b> 2015〜2025年の再現では良好でしたが、①片道4.6bpsで損益が消える
     回転率、②2026年に入っての大幅な悪化、が確認されています。詳細は検証レポートを
     参照してください。</p>
</section>

<footer>
  生成: {generated}　/　取得できたデータの最終日: 米国 {us_last}・日本 {jp_last}<br>
  構成銘柄: 野村アセットマネジメント「組入全銘柄情報」（月次）{holdings_src}<br>
  中川 慧ほか「部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略」
  (SIG-FIN-036-13) の再現実装による出力です。<br>
  <b>投資助言ではありません。</b>バックテスト上の成績は将来の成果を保証しません。
  実際の売買では取引コスト・スリッページ・流動性の影響を受けます。
  投資判断はご自身の責任で行ってください。
</footer>

<script>
(function () {{
  // 業種行の中のリンクを押したときは、折りたたみを開閉させずに遷移させる
  document.querySelectorAll(".rowsum a").forEach(function (a) {{
    a.addEventListener("click", function (e) {{ e.stopPropagation(); }});
  }});

  var el = document.getElementById("stale");
  var asof = new Date("{asof_iso}T21:00:00Z");
  var days = (Date.now() - asof.getTime()) / 86400000;

  // 対象の立会日がすでに過ぎているときは、その旨をはっきり出す。
  // 提供元のデータ待ちで基準日が進まないことがあり、黙っていると
  // 終わった立会日の予想をそのまま見てしまう。
  var target = Date.parse("{next_iso}T00:00:00+09:00");
  var jstNow = new Date(Date.now() + 9 * 3600000);
  var jstMidnight = Date.UTC(jstNow.getUTCFullYear(), jstNow.getUTCMonth(),
                             jstNow.getUTCDate()) - 9 * 3600000;
  if (target < jstMidnight) {{
    el.textContent = "⚠ この予想の対象は {next_session}（すでに終わった立会日）です。"
      + "取得元のデータ待ちで基準日が進んでいません。";
    el.style.display = "block";
  }} else if (days > 4) {{
    el.textContent = "⚠ このページは " + Math.floor(days) +
      " 日前のデータです。自動更新が止まっている可能性があります。";
    el.style.display = "block";
  }}
}})();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs")
    ap.add_argument("--cache", default="./data")
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--factors", type=int, default=3)
    ap.add_argument("--quantile", type=float, default=0.3)
    ap.add_argument("--prior-mode", default="fixed", choices=["fixed", "expanding"])
    ap.add_argument("--synthetic", type=int, default=0,
                    help="ネットワーク不要の合成データで表示確認する")
    a = ap.parse_args()

    params = Params(
        window=a.window, n_factors=a.factors, lam=a.lam,
        quantile=a.quantile, prior_mode=a.prior_mode,
    )
    build(Path(a.outdir), params, a.cache, synthetic=a.synthetic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
