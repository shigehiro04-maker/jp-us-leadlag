#!/usr/bin/env python3
"""毎朝の予測を iPhone 向けの 1 枚の HTML にして docs/index.html に書き出す。

Mac の launchd から平日 7:30 / 8:30 JST（予想）と 16:30 JST（採点）に実行する
（scripts/publish_from_mac.sh）。GitHub Actions の定時実行は数時間遅れて
寄付き後になりがちなため、2026-10 に切り替えた。
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
from leadlag.models import (MAIN_MODEL, MODEL_LABEL, MODEL_ORDER,  # noqa: E402
                            all_models, select_long_short, us_return_on)
from leadlag import viz                                       # noqa: E402

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
    jp_co = getattr(bundle, "jp_co_all", None)
    dates = jp_oc.index
    now = datetime.now(JST)
    for rec in history:
        # 採点済みでも、モデル別の採点や業種別の実績がまだ無いものは埋める
        # （後から追加したモデルを過去分に再計算した記録など）
        need_models = any("ls_return" not in m for m in (rec.get("models") or {}).values())
        if rec.get("resolved") and not need_models and "actual_oc" in rec:
            continue
        if rec.get("exec_date"):
            exec_date = pd.Timestamp(rec["exec_date"])
            if exec_date not in dates:
                continue
        else:
            asof = pd.Timestamp(rec["asof"])
            later = dates[dates > asof]
            if len(later) == 0:
                continue
            exec_date = later[0]
        # 当日の立会中は提供元が途中経過のバーを返すので、大引け後まで採点しない
        if exec_date.date() >= now.date() and (now.hour, now.minute) < (15, 45):
            continue
        row = jp_oc.loc[exec_date]
        co_row = jp_co.loc[exec_date] if jp_co is not None and exec_date in jp_co.index else None
        if not rec.get("resolved"):
            longs = [t for t in rec["long"] if t in row.index and np.isfinite(row[t])]
            shorts = [t for t in rec["short"] if t in row.index and np.isfinite(row[t])]
            if not longs or not shorts:
                continue
            # 上下の的中は記録しない。日中リターンは恒常的にマイナスで、符号を
            # 当てたかどうかは戦略の良し悪しをほとんど表さないため
            # (詳細は leadlag/direction.py の注記)。実現値そのものを残す。
            rec.update({
                "resolved": True,
                "exec_date": str(exec_date.date()),
                "ls_return": float(row[longs].mean() - row[shorts].mean()),
                "market_return": float(row.dropna().mean()),
            })
        # 業種別の実績（答え合わせの図に使う）
        rec["actual_oc"] = {t: round(float(v), 6) for t, v in row.items() if np.isfinite(v)}
        if co_row is not None:
            rec["actual_co"] = {t: round(float(v), 6) for t, v in co_row.items() if np.isfinite(v)}
        # モデル別の採点。寄付き→大引け (ls) と、同じ選択で測った
        # 前日大引け→寄付き (ls_overnight)。後者が大きく前者が小さいときは、
        # 予想が寄付きの時点で先に織り込まれている (2026-10 検証参照)。
        for m in (rec.get("models") or {}).values():
            lo = [t for t in m["long"] if t in row.index and np.isfinite(row[t])]
            sh = [t for t in m["short"] if t in row.index and np.isfinite(row[t])]
            if lo and sh:
                m["ls_return"] = float(row[lo].mean() - row[sh].mean())
            if co_row is not None:
                lo2 = [t for t in m["long"] if t in co_row.index and np.isfinite(co_row[t])]
                sh2 = [t for t in m["short"] if t in co_row.index and np.isfinite(co_row[t])]
                if lo2 and sh2:
                    m["ls_overnight"] = float(co_row[lo2].mean() - co_row[sh2].mean())
    return history


def append_today(history: list[dict], rec: dict) -> list[dict]:
    # 寄付き前に出した予想は、寄付き後の再実行（夕方の採点など）で上書きしない。
    # 上書きすると「寄付き前に公開した予想」の記録が後から変わってしまう。
    prev = next((h for h in history if h["asof"] == rec["asof"]), None)
    if prev is not None and prev.get("pre_open") and not rec.get("pre_open", True):
        return history
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


MOOD_TEXT = {   # 下押し圧力の強弱 → 読み方
    "強い": ("いつもより下げやすい", "down"),
    "やや強い": ("いつもよりやや下げやすい", "down"),
    "標準": ("いつも並み", "flat-dir"),
    "やや弱い": ("いつもよりやや下げにくい", "up"),
    "弱い": ("いつもより下げにくい", "up"),
}


def mood_html(md: dict) -> str:
    """日中（寄付き→大引け）の地合いを、ふだんとの比較として見せる。

    日本株は寄付き後に下がる日が多い（日中は恒常的にマイナス）。上がるか下がるかではなく、
    「今日はふだんより下げやすいか・下げにくいか」を 5 段階の目盛りで示す。
    """
    from leadlag.direction import STRENGTH_LABELS
    strength = md["strength"]
    seg = STRENGTH_LABELS.index(strength)          # 0=下げやすい … 4=下げにくい
    q = [x * 1e4 for x in md.get("quantiles") or []]
    pred = md["pred"] * 1e4
    # 目盛り上の位置（%）。中の区間は境界の値で按分し、両端の区間は隣の区間幅で按分する
    if len(q) == 4:
        edges = [q[0] - (q[1] - q[0]), *q, q[3] + (q[3] - q[2])]
        lo, hi = edges[seg], edges[seg + 1]
        frac = 0.5 if hi <= lo else min(max((pred - lo) / (hi - lo), 0.08), 0.92)
    else:
        frac = 0.5
    pos = (seg + frac) * 20
    word, cls = MOOD_TEXT[strength]
    segs = "".join(f'<i class="s{k}{" on" if k == seg else ""}"></i>' for k in range(5))
    ticks = "".join(f'<span style="left:{(k+1)*20}%">{v:+.0f}</span>' for k, v in enumerate(q)) if len(q) == 4 else ""
    base = md["base_mean"] * 1e4

    def bp(v):
        return "0bp" if round(v) == 0 else f"{v:+.0f}bp"
    lead = ("日本株は寄付き後に下がる日が多く、" if base < 0 and md["base_share_down"] > 0.5 else "")
    return f"""<p class="moodword {cls}">{word}</p>
  <p class="meta" style="margin-top:4px">{lead}過去{md["n_train"]}営業日の
     寄付き→大引けは平均 <b>{bp(base)}</b>（{md["base_share_down"]*100:.0f}%の日が下落）。
     前夜の米国11業種（平均 <b>{md["us_ew_cc"]*100:+.2f}%</b>）から見た今日の予測は <b>{bp(pred)}</b> です。</p>
  <div class="scale" role="img" aria-label="下押し圧力の目盛り: {html.escape(word)}">
    <span class="ptr" style="left:{pos:.1f}%"><b>今日</b>▼</span>
    <div class="segs">{segs}</div>
    <div class="ticks">{ticks}</div>
    <div class="ends"><span>← 下げやすい</span><span>いつも並み</span><span>下げにくい →</span></div>
    <p class="meta small" style="margin-top:2px">目盛りの数字は区切りの予測値（bp、1bp=0.01%）</p>
  </div>
  <details>
    <summary>この目盛りの見方</summary>
    <p class="meta small">下押し圧力（寄付き後の売られやすさ）の強弱です。過去{md["n_train"]}営業日に
       このモデルが出した予測値を低い順に5等分し、今日の予測がどこに入るかを示しています
       （区切りの数字は予測値・bp）。上がるか下がるかを当てるものではなく、
       右端の「下げにくい」でも上昇するという意味ではありません。</p>
  </details>"""


def next_tokyo_session(asof: pd.Timestamp):
    """基準日 (米国の日付) の次の東京立会日。土日・祝日・年末年始 (12/31〜1/3) を飛ばす。"""
    try:
        import jpholiday
    except ImportError:          # 無ければ土日だけ飛ばす
        jpholiday = None
    d = (pd.Timestamp(asof) + pd.Timedelta(days=1)).date()
    while (d.weekday() >= 5 or (d.month, d.day) in {(12, 31), (1, 1), (1, 2), (1, 3)}
           or (jpholiday is not None and jpholiday.is_holiday(d))):
        d += timedelta(days=1)
    return d


def _name(t: str) -> str:
    return html.escape(display_name(t))


def model_compare_html(models_rec: dict) -> str:
    """モデルごとのロング/ショートを並べる。全モデルが一致する業種は太字。"""
    names = [m for m in MODEL_ORDER if m in models_rec]
    if not names:
        return ""
    def common(side: str) -> set:
        sets = [set(models_rec[m][side]) for m in names]
        return set.intersection(*sets) if sets else set()
    cl, cs = common("long"), common("short")
    rows = []
    for m in names:
        lo = "・".join(f"<b>{_name(t)}</b>" if t in cl else _name(t) for t in models_rec[m]["long"])
        sh = "・".join(f"<b>{_name(t)}</b>" if t in cs else _name(t) for t in models_rec[m]["short"])
        tag = ' <span class="weak">メイン</span>' if m == MAIN_MODEL else ""
        rows.append(f'<li class="mrow"><p class="mname">{html.escape(MODEL_LABEL.get(m, m))}{tag}</p>'
                    f'<p class="ml"><span class="up">▲</span> {lo}</p>'
                    f'<p class="ml"><span class="down">▼</span> {sh}</p></li>')
    return "\n".join(rows)


def model_perf_html(history: list[dict]) -> str:
    """モデル別の成績: 指標タイル・累積の折れ線・夜間/日中の比較・答え合わせ・日別の表。"""
    recs = [h for h in history if h.get("resolved") and h.get("models")
            and any("ls_return" in m for m in h["models"].values())]
    if not recs:
        return '<p class="meta">モデル別の実績は、予想の翌営業日の大引け後から表示されます。</p>'
    names = [m for m in MODEL_ORDER if any(m in r["models"] for r in recs)]
    live = [r for r in recs if r.get("pre_open")]
    live_start = live[0]["exec_date"] if live else None

    def agg(rs, m):
        x = [r["models"][m]["ls_return"] for r in rs if m in r["models"] and "ls_return" in r["models"][m]]
        on = [r["models"][m]["ls_overnight"] for r in rs if m in r["models"] and "ls_overnight" in r["models"][m]]
        if not x:
            return None
        return {"name": m, "n": len(x), "cum": (np.prod([1 + v for v in x]) - 1) * 100,
                "win": np.mean([v > 0 for v in x]) * 100, "intraday": float(np.sum(x)) * 100,
                "overnight": float(np.sum(on)) * 100 if on else None}

    st_all = [a for a in (agg(recs, m) for m in names) if a]
    st_live = {a["name"]: a for a in (agg(live, m) for m in names) if a}
    tiles = []
    for a in st_all:
        lv = st_live.get(a["name"])
        lv_txt = (f'寄付き前公開 {lv["n"]}日 {lv["cum"]:+.2f}%' if lv else "寄付き前公開 まだなし")
        main = ' <span class="weak">メイン</span>' if a["name"] == MAIN_MODEL else ""
        tiles.append(
            f'<div class="tile"><p class="tn"><i class="sw m{viz.model_color_index(a["name"])}"></i>'
            f'{html.escape(MODEL_LABEL.get(a["name"], a["name"]))}{main}</p>'
            f'<p class="tv {"up" if a["cum"] >= 0 else "down"}">{a["cum"]:+.1f}%</p>'
            f'<p class="tk2">累積 {a["n"]}日・勝率 {a["win"]:.0f}%</p><p class="tk2">{lv_txt}</p></div>')

    dates = [r["exec_date"] for r in recs]
    series = {m: [r["models"].get(m, {}).get("ls_return") for r in recs] for m in names}
    last = recs[-1]
    last_ls = " ・ ".join(
        f'{html.escape(MODEL_LABEL.get(m, m))} <b class="{"up" if last["models"][m]["ls_return"] >= 0 else "down"}">'
        f'{last["models"][m]["ls_return"]*100:+.2f}%</b>'
        for m in names if "ls_return" in last["models"].get(m, {}))
    backfill_note = (
        '<p class="meta">寄付き前公開の開始より前の日は、各日の米国終値までのデータで'
        '後から計算した参考値です（このページで寄付き前に出していた予想ではありません）。</p>'
        if any(r.get("backfilled") for r in recs) else "")
    def sub(title, body, key, opened=False):
        op = " open" if opened else ""
        return (f'<details class="subfold"{op} data-k="{key}"><summary class="fh3">{title}'
                f'<span class="tri" aria-hidden="true">▼</span></summary>{body}</details>')
    cum_body = f"""{viz.legend(names)}
  {viz.cumulative_chart(dates, series, live_start)}
  {backfill_note}"""
    split_body = f"""<p class="meta" style="margin-top:0">同じ予想を「寄付き→大引け」（取れる部分）と「前日大引け→寄付き」
     （寄付きの時点ですでに動いた部分）で測った合計。下の斜線が大きく上が小さいほど、先回りされています。</p>
  {viz.split_bars(st_all)}"""
    ans_body = f"""<p class="meta" style="margin-top:0">{last_ls}</p>
  {viz.answer_check(last)}"""
    daily_body = f"""{viz.legend(names)}
    {viz.daily_table(recs, names)}"""
    return f"""<div class="tiles">{"".join(tiles)}</div>
  {sub("累積リターン（ロング5−ショート5、寄付き→大引け）", cum_body, "cum", True)}
  {sub("寄付きで先に織り込まれていないか", split_body, "split")}
  {sub(f"前回の答え合わせ（{last['exec_date']}・業種別の寄付き→大引け）", ans_body, "ans")}
  {sub("日別の実績（ロング−ショート %、直近20日）", daily_body, "daily")}
  <p class="meta">取引コスト控除前。実際に運用した記録ではなく、予想をその日の実現リターンで採点したものです。</p>"""


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

    # 次の東京立会日（土日・祝日・年末年始を飛ばす）
    next_session = next_tokyo_session(asof)
    now_jst = datetime.now(JST)
    pre_open = now_jst < datetime(next_session.year, next_session.month,
                                  next_session.day, 9, 0, tzinfo=JST)

    # 各モデルのスコア。メインは SCS（leadlag/models.py の注記参照）
    model_scores = all_models(bundle, params, asof, sig)
    models_rec = {}
    for name, sc in model_scores.items():
        lo, sh = select_long_short(sc, params.quantile)
        models_rec[name] = {"long": lo, "short": sh,
                            "signals": {t: float(sc[t]) for t in sc.dropna().index}}

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
            "models": models_rec,
            "main": MAIN_MODEL,
            "target": str(next_session),
            "created_at": now_jst.strftime("%Y-%m-%d %H:%M"),
            "pre_open": bool(pre_open),
            "resolved": False,
        },
    )
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=1))

    # 寄付き前に出した予想が残っていれば、ページもその内容で描く（予備実行で変えない）
    shown = next((h for h in history if h["asof"] == str(asof.date())), None)
    if shown and shown.get("models"):
        models_rec = shown["models"]
        pre_open = bool(shown.get("pre_open", pre_open))
        created_str = str(shown.get("created_at", ""))[5:].replace("-", "/")
    else:
        created_str = now_jst.strftime("%m/%d %H:%M")

    recent = [h for h in history if h.get("resolved")]

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

    # 業種ランキングの本体はメインモデル (SCS)。米国リターン (%) をそのまま表示する
    main = models_rec.get(MAIN_MODEL) or models_rec["PCA_SUB"]
    msig = pd.Series(main["signals"]).reindex(
        [t for t in res.jp_tickers if t in main["signals"]])
    msig = msig.sort_values(ascending=False, kind="stable")
    disp = msig * (100 if MAIN_MODEL == "SCS" else 1)
    mvmax = float(np.abs(disp.to_numpy()).max()) or 1.0
    m_long, m_short = main["long"], main["short"]
    long_rows = "\n".join(row_html(t, disp[t], mvmax, "ロング", holdings.get(t)) for t in m_long)
    short_rows = "\n".join(row_html(t, disp[t], mvmax, "ショート", holdings.get(t)) for t in m_short)
    mid = [t for t in msig.index if t not in m_long and t not in m_short]
    mid_rows = "\n".join(row_html(t, disp[t], mvmax, "-", holdings.get(t)) for t in mid)
    compare_html = model_compare_html(models_rec)
    perf_html = model_perf_html(history)
    # 前夜の米国 11 業種の騰落（SCS がそのまま使う値）
    try:
        us_today = {t: float(v) for t, v in us_return_on(bundle, asof).items()}
    except KeyError:
        us_today = {}
    us_chart = viz.us_sector_chart(us_today)
    grid_order = list(pd.Series(models_rec.get("合成", main)["signals"])
                      .sort_values(ascending=False, kind="stable").index)
    grid_html = viz.agreement_grid(models_rec, grid_order)


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
        mood=mood_html(md),
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
        f_scores=", ".join(f"f{i+1}={v:+.2f}" for i, v in enumerate(res.factor_scores)),
        generated=generated,
        asof_iso=asof.date().isoformat(),
        next_iso=next_session.isoformat(),
        us_last=us_last,
        jp_last=jp_last,
        holdings_src=holdings_src,
        compare_html=compare_html,
        us_chart=us_chart,
        grid_html=grid_html,
        perf_html=perf_html,
        main_label=MODEL_LABEL.get(MAIN_MODEL, MAIN_MODEL),
        created=created_str,
        pre_open_note=("寄付き前に作成" if pre_open else "寄付き後に作成（参考扱い）"),
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
  --m0:#2a78d6; --m1:#eb6834; --m2:#1baf7a; --m3:#c98500;
  --safe-t:env(safe-area-inset-top); --safe-b:env(safe-area-inset-bottom);
}}
@media (prefers-color-scheme:dark) {{
  :root {{ --bg:#0b1020; --card:#151a2d; --fg:#e8eaf2; --muted:#9aa3b8; --line:#252b42;
    --up:#2dd4bf; --down:#f87171; --accent:#60a5fa;
    --m0:#3987e5; --m1:#d95926; --m2:#199e70; --m3:#c98500; }}
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
.legend {{ display:flex; flex-wrap:wrap; gap:6px 14px; font-size:12px; color:var(--muted); margin:4px 0 6px; }}
.lg {{ display:inline-flex; align-items:center; gap:5px; }}
.sw {{ display:inline-block; width:10px; height:10px; border-radius:3px; vertical-align:-1px; margin-right:4px; }}
.sw.m0, .dot.m0 {{ background:var(--m0); fill:var(--m0); }} .sw.m1, .dot.m1 {{ background:var(--m1); fill:var(--m1); }}
.sw.m2, .dot.m2 {{ background:var(--m2); fill:var(--m2); }} .sw.m3, .dot.m3 {{ background:var(--m3); fill:var(--m3); }}
.chart {{ width:100%; height:auto; display:block; margin:2px 0 6px; overflow:visible; }}
.chart .ln {{ fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }}
.chart .ln.m0 {{ stroke:var(--m0); stroke-width:2.6; }} .chart .ln.m1 {{ stroke:var(--m1); }}
.chart .ln.m2 {{ stroke:var(--m2); }} .chart .ln.m3 {{ stroke:var(--m3); stroke-dasharray:4 3; }}
.chart .gridl {{ stroke:var(--line); stroke-width:1; }} .chart .zero {{ stroke:var(--muted); stroke-width:1; opacity:.6; }}
.chart .live {{ stroke:var(--muted); stroke-dasharray:3 3; }}
.chart .ax {{ font-size:10px; fill:var(--muted); }} .chart .vl {{ font-size:10px; fill:var(--fg); }}
.chart .hit {{ fill:transparent; }} .chart .hit:hover {{ fill:var(--line); }}
.h3 {{ font-size:13px; font-weight:700; margin:18px 0 6px; }}
.tiles {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.tile {{ border:1px solid var(--line); border-radius:12px; padding:9px 10px; }}
.tile p {{ margin:0; }} .tn {{ font-size:12px; font-weight:600; }}
.tv {{ font-size:22px; font-weight:800; font-variant-numeric:tabular-nums; margin:2px 0 !important; }}
.tk2 {{ font-size:11px; color:var(--muted); }}
.dvlist {{ list-style:none; padding:0; margin:4px 0; }}
.dv {{ display:grid; grid-template-columns:92px 1fr 56px 44px; gap:6px; align-items:center; padding:3px 0; font-size:13px; }}
.dl {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }} .dl small {{ color:var(--muted); font-size:10px; }}
.db {{ position:relative; display:block; height:10px; background:var(--line); border-radius:3px; }}
.db i {{ position:absolute; top:0; height:10px; border-radius:3px; }}
.db i.pos {{ background:var(--up); }} .db i.neg {{ background:var(--down); }}
.db i.hatch {{ background-image:repeating-linear-gradient(45deg, rgba(255,255,255,.55) 0 2px, transparent 2px 5px); }}
.dval {{ text-align:right; font-variant-numeric:tabular-nums; font-size:12px; }}
.dvlist.below .dv {{ grid-template-columns:92px 1fr 56px; }}
.dsub {{ grid-column:2 / -1; font-size:11px; color:var(--muted); margin-top:-2px; }}
.dm {{ font-size:11px; color:var(--muted); white-space:nowrap; }} .to {{ font-size:11px; }}
.tag {{ font-size:10px; font-weight:700; }}
.sblist {{ list-style:none; padding:0; margin:4px 0; }}
.sb {{ display:grid; grid-template-columns:96px 1fr 56px; gap:3px 6px; align-items:center; padding:6px 0;
  border-bottom:1px solid var(--line); font-size:12px; }}
.sb:last-child {{ border-bottom:0; }}
.sbn {{ grid-column:1 / -1; font-weight:600; font-size:12px; }} .sbk {{ color:var(--muted); font-size:11px; }}
.scroll {{ overflow-x:auto; }}
table.grid {{ border-collapse:separate; border-spacing:2px; width:100%; font-size:12px; }}
table.grid th {{ font-size:10px; color:var(--muted); font-weight:600; white-space:nowrap; padding:2px; }}
table.grid th.gt {{ text-align:left; color:var(--fg); font-size:12px; font-weight:500; max-width:84px;
  overflow:hidden; text-overflow:ellipsis; }}
table.grid td {{ text-align:center; border-radius:5px; padding:4px 0; font-weight:700; }}
td.gl {{ background:color-mix(in srgb, var(--up) 22%, transparent); color:var(--up); }}
td.gs {{ background:color-mix(in srgb, var(--down) 22%, transparent); color:var(--down); }}
td.gn {{ color:var(--muted); }} td.ga {{ font-size:11px; min-width:30px; }}
table.heat {{ border-collapse:separate; border-spacing:2px; width:100%; font-size:12px; font-variant-numeric:tabular-nums; }}
table.heat th {{ font-size:11px; color:var(--muted); font-weight:500; text-align:left; white-space:nowrap; }}
table.heat td {{ text-align:right; padding:3px 5px; border-radius:4px; }}
td.hp {{ background:color-mix(in srgb, var(--up) calc(var(--a) * 100%), transparent); }}
td.hn {{ background:color-mix(in srgb, var(--down) calc(var(--a) * 100%), transparent); }}
.mut {{ color:var(--muted); }}
details.card {{ margin-top:0; }}
details.fold > summary.fh, details.subfold > summary.fh3 {{ display:flex; align-items:center; justify-content:space-between;
  cursor:pointer; list-style:none; color:inherit; padding:0; }}
details.fold > summary.fh::-webkit-details-marker, details.subfold > summary.fh3::-webkit-details-marker {{ display:none; }}
details.fold > summary.fh::after, details.subfold > summary.fh3::after {{ content:none; }}
details.fold > summary.fh h2 {{ margin:0; }}
details.fold[open] > summary.fh {{ margin-bottom:12px; }}
.tri {{ font-size:11px; color:var(--muted); transition:transform .15s; margin-left:8px; }}
details:not([open]) > summary > .tri {{ transform:rotate(-90deg); }}
@media (prefers-reduced-motion:reduce) {{ .tri {{ transition:none; }} }}
details.subfold {{ border-top:1px solid var(--line); margin-top:12px; padding-top:10px; }}
summary.fh3 {{ font-size:13px; font-weight:700; }}
details.subfold[open] > summary.fh3 {{ margin-bottom:6px; }}
.moodword {{ font-size:28px; font-weight:800; letter-spacing:-.01em; margin:0; line-height:1.25; }}
.moodword.down {{ color:var(--down); }} .moodword.up {{ color:var(--up); }} .moodword.flat-dir {{ color:var(--fg); }}
.scale {{ position:relative; margin:30px 2px 4px; }}
.scale .ptr {{ position:absolute; top:-24px; transform:translateX(-50%); font-size:11px; color:var(--fg);
  white-space:nowrap; display:flex; flex-direction:column; align-items:center; line-height:1.1; }}
.scale .segs {{ display:flex; gap:2px; }}
.scale .segs i {{ flex:1; height:12px; border-radius:3px; opacity:.35; }}
.scale .segs i.on {{ opacity:1; outline:2px solid var(--fg); outline-offset:1px; }}
.scale .s0 {{ background:var(--down); }} .scale .s1 {{ background:color-mix(in srgb, var(--down) 55%, var(--line)); }}
.scale .s2 {{ background:var(--muted); }} .scale .s3 {{ background:color-mix(in srgb, var(--up) 55%, var(--line)); }}
.scale .s4 {{ background:var(--up); }}
.scale .ticks {{ position:relative; height:14px; font-size:10px; color:var(--muted); font-variant-numeric:tabular-nums; }}
.scale .ticks span {{ position:absolute; top:2px; transform:translateX(-50%); }}
.scale .ends {{ display:flex; justify-content:space-between; font-size:11px; color:var(--muted); margin-top:2px; }}
.mrow {{ padding:8px 0; border-bottom:1px solid var(--line); }}
.mrow:last-child {{ border-bottom:0; }}
.mname {{ font-size:13px; font-weight:600; margin:0 0 2px; }}
.ml {{ font-size:13px; margin:1px 0; line-height:1.5; }}
table.perf {{ width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }}
table.perf th, table.perf td {{ padding:5px 4px; border-bottom:1px solid var(--line); text-align:right; }}
table.perf th:first-child, table.perf td:first-child {{ text-align:left; }}
table.perf th {{ font-size:11px; color:var(--muted); font-weight:600; }}
td.pos {{ color:var(--up); }} td.neg {{ color:var(--down); }}
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

<details class="card fold" open data-k="mood">
  <summary class="fh"><h2>今日の日中（寄付き→大引け）の地合い</h2><span class="tri" aria-hidden="true">▼</span></summary>
  {mood}
</details>

<details class="card fold" open data-k="rank">
  <summary class="fh"><h2>業種ランキング（相対の強弱）</h2><span class="tri" aria-hidden="true">▼</span></summary>
  <p class="meta" style="margin-top:0">モデル: <b>{main_label}</b>（対応する米国業種ETFの当日リターン%）・{created} {pre_open_note}</p>
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
</details>

<details class="card fold" data-k="us">
  <summary class="fh"><h2>前夜の米国11業種（{asof}）</h2><span class="tri" aria-hidden="true">▼</span></summary>
  <p class="meta" style="margin-top:0">右は、その動きを受け取る日本の業種（業種対応 SCS の対応表）。</p>
  {us_chart}
  <p class="meta">部分空間正則化PCAの共通ファクター {f_scores}</p>
</details>

<details class="card fold" data-k="models">
  <summary class="fh"><h2>モデル別の予想</h2><span class="tri" aria-hidden="true">▼</span></summary>
  <p class="meta" style="margin-top:0">▲ロング（強い）・▼ショート（弱い）。右端は4モデルのうち何モデルが同じ向きか。
     並びは3モデル合成のスコア順。</p>
  {grid_html}
  <details>
    <summary>モデルごとの一覧で見る</summary>
    <ul>
{compare_html}
    </ul>
  </details>
  <p class="meta">部分空間正則化PCAは論文公開（2026/3/19）後に寄付きで先回りされ、
     3〜7月に大きく崩れました（検証レポート参照）。</p>
</details>

<details class="card fold" open data-k="perf">
  <summary class="fh"><h2>モデル別の成績</h2><span class="tri" aria-hidden="true">▼</span></summary>
  {perf_html}
  <p class="meta"><b>検証メモ（2026-10）:</b> 2015〜2025年はSCS R/R 3.96・PCA_SUB 2.37と良好でしたが、
     論文公開後（3/19〜7/23）にPCA_SUBは −35.8%。同じ予想で前日大引け→寄付きを測ると
     公開後に大きく増えており、寄付きで先に織り込まれています。損益分岐は片道
     2.6bp（PCA_SUB）〜4.0bp（SCS）。詳細は results/verification_2026-10/ を参照。</p>
</details>


<footer>
  生成: {generated}　/　取得できたデータの最終日: 米国 {us_last}・日本 {jp_last}<br>
  構成銘柄: 野村アセットマネジメント「組入全銘柄情報」（月次）{holdings_src}<br>
  中川 慧ほか「部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略」
  (SIG-FIN-036-13)、金谷・吉田「日米リード・ラグ戦略のための業種対応シグナル」
  (SIG-FIN-037-11) の再現実装による出力です。<br>
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

  // 折りたたみの開閉を端末ごとに覚えておく
  document.querySelectorAll("details[data-k]").forEach(function (d) {{
    var k = "fold:" + d.getAttribute("data-k");
    try {{ var v = localStorage.getItem(k); if (v !== null) d.open = (v === "1"); }} catch (e) {{}}
    d.addEventListener("toggle", function () {{
      try {{ localStorage.setItem(k, d.open ? "1" : "0"); }} catch (e) {{}}
    }});
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
