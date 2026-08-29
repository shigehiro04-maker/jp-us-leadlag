import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, yfinance
from leadlag.config import Params
from leadlag.data import build_bundle, load_prices
from leadlag.engine import LeadLagEngine

print("yfinance:", yfinance.__version__, "pandas:", pd.__version__)
us, jp = load_prices(start="2010-01-01", cache_dir="./data", refresh=True)
print("US panel:", us.shape, us.index[0].date(), "->", us.index[-1].date())
print("JP panel:", jp.shape, jp.index[0].date(), "->", jp.index[-1].date())
print("US cols:", sorted({c[0] for c in us.columns}))
print("JP cols:", sorted({c[0] for c in jp.columns}))
print("US index tail:", [str(d.date()) for d in us.index[-5:]])
print("JP index tail:", [str(d.date()) for d in jp.index[-5:]])

b = build_bundle(us, jp)
print("common:", len(b.dates), b.dates[0].date(), "->", b.dates[-1].date())
print("--- us_cc tail notna per row ---"); print(b.us_cc.tail(6).notna().sum(axis=1).to_string())
print("--- jp_cc tail notna per row ---"); print(b.jp_cc.tail(6).notna().sum(axis=1).to_string())
print("--- jp_oc tail notna per row ---"); print(b.jp_oc.tail(6).notna().sum(axis=1).to_string())
print("--- us_close tail ---"); print(b.us_close.tail(3).iloc[:, :6].to_string())
print("--- jp_close tail ---"); print(b.jp_close.tail(3).iloc[:, :6].to_string())
print("--- jp_open tail ---"); print(b.jp_open.tail(3).iloc[:, :6].to_string())

p = Params()
e = LeadLagEngine(b, p)
i = len(e.dates) - 1
win = e.returns.iloc[i - p.window:i]
print("window:", win.index[0].date(), "->", win.index[-1].date(), "rows", len(win))
for t in e.us_tickers:
    print(f"US {t:5s} win_ok={bool(win[t].notna().all())} nan_in_win={int(win[t].isna().sum())} today={e.returns.iloc[i][t]}")
for t in e.jp_tickers:
    print(f"JP {t:8s} win_ok={bool(win[t].notna().all())} nan_in_win={int(win[t].isna().sum())} today={e.returns.iloc[i][t]}")
