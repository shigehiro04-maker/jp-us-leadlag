"""日次ページ用の図（外部ライブラリ・CDN なしのインライン SVG / HTML）。

色は役割で決める。
- モデルの識別（カテゴリ）: --m0..--m3（順番固定。モデルの並びは models.MODEL_ORDER）
- 上げ下げ（極性）       : --up / --down（ページ既存のトークン）
文字色は常に本文・補助色で、系列色は線・点・凡例の印だけに使う。
"""

from __future__ import annotations

import html
from datetime import date

import numpy as np

from .config import display_name
from .models import MAIN_MODEL, MODEL_LABEL, MODEL_ORDER, SCS_MAP


def _e(x) -> str:
    return html.escape(str(x))


def _fmt_pct(v: float, digits: int = 2) -> str:
    return f"{v:+.{digits}f}%"


def model_color_index(name: str) -> int:
    return MODEL_ORDER.index(name) if name in MODEL_ORDER else 0


def legend(names: list[str]) -> str:
    items = "".join(
        f'<span class="lg"><i class="sw m{model_color_index(n)}"></i>{_e(MODEL_LABEL.get(n, n))}</span>'
        for n in names)
    return f'<div class="legend">{items}</div>'


# ---------------------------------------------------------------------------
# 横棒（HTML）: 中央 0 の発散バー
# ---------------------------------------------------------------------------
def diverging_rows(rows: list[tuple[str, str, float, str]], vmax: float | None = None,
                   unit: str = "%", mark_below: bool = False) -> str:
    """rows = [(左ラベル, 補足, 値, 右の印 HTML)]。値は % 単位。"""
    if not rows:
        return ""
    vmax = vmax or max(abs(r[2]) for r in rows) or 1.0
    out = []
    for lab, sub, v, mark in rows:
        w = min(abs(v) / vmax, 1.0) * 50
        side = "pos" if v >= 0 else "neg"
        pos = f"left:50%;width:{w:.1f}%" if v >= 0 else f"right:50%;width:{w:.1f}%"
        out.append(
            f'<li class="dv" title="{_e(lab)} {v:+.2f}{unit}">'
            f'<span class="dl">{_e(lab)}<small>{_e(sub)}</small></span>'
            f'<span class="db"><i class="{side}" style="{pos}"></i></span>'
            f'<span class="dval">{v:+.2f}{unit}</span>'
            + (f'<span class="dsub">{mark}</span></li>' if mark_below else f'<span class="dm">{mark}</span></li>'))
    cls = "dvlist below" if mark_below else "dvlist"
    return f'<ul class="{cls}">' + "".join(out) + "</ul>"


def us_sector_chart(us_today: dict[str, float]) -> str:
    """前夜の米国 11 業種の騰落と、その動きを受け取る日本の業種（SCS の対応）。"""
    inv: dict[str, list[str]] = {}
    for j, u in SCS_MAP.items():
        inv.setdefault(u, []).append(display_name(j))
    rows = []
    for u, v in sorted(us_today.items(), key=lambda kv: -kv[1]):
        if v is None or not np.isfinite(v):
            continue
        rows.append((f"{display_name(u)}", f" {u}", v * 100,
                     f'<span class="to">→ {_e("・".join(inv.get(u, [])) or "—")}</span>'))
    return diverging_rows(rows, mark_below=True)


# ---------------------------------------------------------------------------
# モデル×業種のヒートマップ（HTML テーブル）
# ---------------------------------------------------------------------------
def agreement_grid(models_rec: dict, jp_order: list[str]) -> str:
    names = [m for m in MODEL_ORDER if m in models_rec]
    head = "".join(f'<th><span class="sw m{model_color_index(m)}"></span>'
                   f'{_e(MODEL_LABEL.get(m, m).replace("部分空間正則化", "正則化"))}</th>' for m in names)
    body = []
    for t in jp_order:
        cells, score = [], 0
        for m in names:
            if t in models_rec[m]["long"]:
                cells.append('<td class="gl" title="ロング">▲</td>'); score += 1
            elif t in models_rec[m]["short"]:
                cells.append('<td class="gs" title="ショート">▼</td>'); score -= 1
            else:
                cells.append('<td class="gn">·</td>')
        agree = (f'<td class="ga up">▲{score}</td>' if score > 0 else
                 f'<td class="ga down">▼{-score}</td>' if score < 0 else '<td class="ga">—</td>')
        body.append(f'<tr><th class="gt">{_e(display_name(t))}</th>{"".join(cells)}{agree}</tr>')
    return (f'<div class="scroll"><table class="grid"><tr><th></th>{head}<th>一致</th></tr>'
            + "".join(body) + "</table></div>")


# ---------------------------------------------------------------------------
# 累積リターンの折れ線（SVG）
# ---------------------------------------------------------------------------
def cumulative_chart(dates: list[str], series: dict[str, list[float]],
                     live_start: str | None = None, width: int = 340, height: int = 190) -> str:
    """series の値は日次 L-S リターン（小数）。累積 % にして描く。"""
    if len(dates) < 2 or not series:
        return '<p class="meta">実績が 2 日以上たまると折れ線が出ます。</p>'
    padl, padr, padt, padb = 38, 48, 10, 22
    W, H = width - padl - padr, height - padt - padb
    cum = {}
    for k, v in series.items():
        c, acc = [], 1.0
        for r in v:
            acc *= 1 + (r if r is not None and np.isfinite(r) else 0.0)
            c.append((acc - 1) * 100)
        cum[k] = [0.0] + c
    n = len(dates) + 1
    lo = min(min(v) for v in cum.values()); hi = max(max(v) for v in cum.values())
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = (hi - lo) or 1.0
    lo -= span * 0.06; hi += span * 0.06; span = hi - lo

    def X(i): return padl + W * i / (n - 1)
    def Y(v): return padt + H * (1 - (v - lo) / span)

    # 目盛り（4 本前後のきりのよい値）
    step = _nice_step(span / 4)
    ticks = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
    grid = "".join(
        f'<line x1="{padl}" x2="{padl+W}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="{"zero" if abs(t) < 1e-9 else "gridl"}"/>'
        f'<text x="{padl-4}" y="{Y(t)+3.5:.1f}" class="ax" text-anchor="end">{"0" if abs(t) < 1e-9 else f"{t:+.0f}"}%</text>'
        for t in ticks)
    xl = (f'<text x="{padl}" y="{height-6}" class="ax">{dates[0][5:]}</text>'
          f'<text x="{padl+W}" y="{height-6}" class="ax" text-anchor="end">{dates[-1][5:]}</text>')
    live = ""
    if live_start and live_start in dates:
        i = dates.index(live_start) + 1
        live = (f'<line x1="{X(i-1):.1f}" x2="{X(i-1):.1f}" y1="{padt}" y2="{padt+H}" class="live"/>'
                f'<text x="{X(i-1)+3:.1f}" y="{padt+9}" class="ax">寄付き前公開→</text>')
    lines, ends = [], []
    for k, c in cum.items():
        ci = model_color_index(k)
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(c))
        lines.append(f'<path d="{d}" class="ln m{ci}"><title>{_e(MODEL_LABEL.get(k, k))} 累積 {c[-1]:+.2f}%</title></path>')
        # 点（ホバーで日付と値）
        pts = "".join(
            f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="6" class="hit"><title>{dates[i-1]} {_e(MODEL_LABEL.get(k, k))} 累積 {v:+.2f}%</title></circle>'
            for i, v in enumerate(c) if i > 0)
        lines.append(pts)
        ends.append((Y(c[-1]), ci, c[-1]))
    # 右端の値ラベル（重なりを避けて縦にずらす）
    ends.sort()
    lab, last = [], -99
    for y, ci, v in ends:
        y = max(y, last + 11); last = y
        lab.append(f'<circle cx="{padl+W+5:.1f}" cy="{y-3:.1f}" r="3.5" class="dot m{ci}"/>'
                   f'<text x="{padl+W+11:.1f}" y="{y:.1f}" class="vl">{v:+.1f}%</text>')
    return (f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="モデル別の累積リターン">'
            f'{grid}{live}{"".join(lines)}{"".join(lab)}{xl}</svg>')


def _nice_step(x: float) -> float:
    if x <= 0:
        return 1.0
    e = 10 ** np.floor(np.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if x <= m * e:
            return m * e
    return 10 * e


# ---------------------------------------------------------------------------
# 寄付き→大引け と 前日大引け→寄付き の比較（HTML 棒）
# ---------------------------------------------------------------------------
def split_bars(stats: list[dict]) -> str:
    """stats = [{name, intraday(%), overnight(%)}]"""
    if not stats:
        return ""
    vmax = max(max(abs(s["intraday"]), abs(s["overnight"] or 0)) for s in stats) or 1.0
    rows = []
    for s in stats:
        ci = model_color_index(s["name"])
        def bar(v, cls):
            if v is None:
                return '<span class="db"></span><span class="dval mut">—</span>'
            w = min(abs(v) / vmax, 1) * 50
            pos = f"left:50%;width:{w:.1f}%" if v >= 0 else f"right:50%;width:{w:.1f}%"
            return (f'<span class="db"><i class="{cls} {"pos" if v >= 0 else "neg"}" style="{pos}"></i></span>'
                    f'<span class="dval">{v:+.2f}%</span>')
        rows.append(
            f'<li class="sb"><span class="sbn"><i class="sw m{ci}"></i>{_e(MODEL_LABEL.get(s["name"], s["name"]))}</span>'
            f'<span class="sbk">寄付→引け</span>{bar(s["intraday"], "solid")}'
            f'<span class="sbk">前日引け→寄付</span>{bar(s["overnight"], "hatch")}</li>')
    return '<ul class="sblist">' + "".join(rows) + "</ul>"


# ---------------------------------------------------------------------------
# 前回の答え合わせ: 実際の業種リターン（寄付→引け）と、メインモデルの予想
# ---------------------------------------------------------------------------
def answer_check(rec: dict) -> str:
    act = rec.get("actual_oc") or {}
    if not act:
        return ""
    main = (rec.get("models") or {}).get(MAIN_MODEL) or {"long": rec.get("long", []), "short": rec.get("short", [])}
    rows = []
    for t, v in sorted(act.items(), key=lambda kv: -kv[1]):
        if t in main["long"]:
            mark = '<span class="tag up">▲予想</span>'
        elif t in main["short"]:
            mark = '<span class="tag down">▼予想</span>'
        else:
            mark = ""
        rows.append((display_name(t), "", v * 100, mark))
    return diverging_rows(rows)


# ---------------------------------------------------------------------------
# 日別の実績（モデル×日のヒート表）
# ---------------------------------------------------------------------------
def daily_table(recs: list[dict], names: list[str], n: int = 20) -> str:
    rows = []
    vals = [abs(r["models"][m]["ls_return"]) * 100 for r in recs for m in names
            if m in r.get("models", {}) and "ls_return" in r["models"][m]]
    vmax = max(vals) if vals else 1.0
    for r in list(reversed(recs))[:n]:
        cells = []
        for m in names:
            mm = r.get("models", {}).get(m, {})
            if "ls_return" not in mm:
                cells.append('<td class="mut">—</td>'); continue
            v = mm["ls_return"] * 100
            a = 0.12 + 0.55 * min(abs(v) / vmax, 1)
            cls = "hp" if v >= 0 else "hn"
            cells.append(f'<td class="{cls}" style="--a:{a:.2f}">{v:+.2f}</td>')
        tag = "" if r.get("pre_open") else ' <small class="mut">参考</small>'
        rows.append(f'<tr><th>{_e(r["exec_date"][5:])}{tag}</th>{"".join(cells)}</tr>')
    head = "".join(f'<th><span class="sw m{model_color_index(m)}"></span></th>' for m in names)
    return (f'<div class="scroll"><table class="heat"><tr><th>執行日</th>{head}</tr>{"".join(rows)}</table></div>')
