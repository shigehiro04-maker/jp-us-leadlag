#!/usr/bin/env python3
"""history.json の過去の予想に、後から追加したモデル（SCS など）の予想を埋める。

各記録の基準日 (asof) までの価格だけを使って計算し直すので、先読みはない。
ただしその日の朝にこのページで公開していた予想ではないため、
"backfilled": true を付け、"pre_open" は付けない（成績の「寄付き前公開」には入らない）。

  python scripts/backfill_models.py --outdir docs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_page import resolve_history                       # noqa: E402
from leadlag.config import Params                             # noqa: E402
from leadlag.data import build_bundle, load_prices            # noqa: E402
from leadlag.engine import LeadLagEngine                      # noqa: E402
from leadlag.models import all_models, select_long_short      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs")
    ap.add_argument("--cache", default="./data")
    a = ap.parse_args()
    path = Path(a.outdir) / "history.json"
    hist = json.loads(path.read_text())
    params = Params()
    us, jp = load_prices(start=params.start, cache_dir=a.cache, refresh=True)
    n = 0
    for rec in hist:
        if rec.get("models"):
            continue
        asof = pd.Timestamp(rec["asof"])
        b = build_bundle(us.loc[:asof], jp.loc[:asof])
        a2, res = LeadLagEngine(b, params).latest()
        if pd.Timestamp(a2) != asof:
            print(f"{rec['asof']}: 基準日が一致しないため省略 ({a2.date()})")
            continue
        sig = pd.Series(res.scores, index=res.jp_tickers)
        models = {}
        for name, sc in all_models(b, params, asof, sig).items():
            lo, sh = select_long_short(sc, params.quantile)
            models[name] = {"long": lo, "short": sh,
                            "signals": {t: float(sc[t]) for t in sc.dropna().index}}
        rec["models"] = models
        rec["main"] = "SCS"
        rec["backfilled"] = True
        n += 1
        print(f"{rec['asof']}: SCS L={models['SCS']['long']}")
    full = build_bundle(us, jp)
    hist = resolve_history(hist, full)
    path.write_text(json.dumps(hist, ensure_ascii=False, indent=1))
    print(f"埋めた記録: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
