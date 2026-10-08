import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings('ignore')
O = 'out'
R = pd.read_csv(f'{O}/ret_oc.csv', index_col=0, parse_dates=True)
RO = pd.read_csv(f'{O}/ret_overnight.csv', index_col=0, parse_dates=True)
A = pd.read_csv(f'{O}/active.csv', index_col=0, parse_dates=True)
PUB = pd.Timestamp('2026-03-19'); P62_END = pd.Timestamp('2026-07-23')
STR = ['PCA_SUB','PCA_PLAIN','MOM','DOUBLE','SCS','PCA_L1250']

def stats(r):
    r = r.dropna()
    if len(r) < 5: return {}
    ar = r.mean()*252; risk = r.std()*np.sqrt(252)
    W = (1+r).cumprod(); mdd = (W/W.cummax()-1).min()
    t = r.mean()/r.std()*np.sqrt(len(r))
    return dict(N=len(r), AR=ar*100, RISK=risk*100, RR=ar/risk, MDD=-mdd*100, cum=(W.iloc[-1]-1)*100, t=t, hit=(r>0).mean()*100)

def sub(k, a=None, b=None):
    r = R[k][A[k]]
    if a: r = r[r.index >= a]
    if b: r = r[r.index <= b]
    return r

res = {}
periods = {
  '全期間(2015-2026/10)': (None, None),
  '論文1と同期間(2015-2025)': (None, '2025-12-31'),
  'SCS論文と同期間(2018-2025)': ('2018-01-04', '2025-12-31'),
  '公開前(2015-2026/3/18)': (None, '2026-03-18'),
  '公開後 全期間(3/19-10/7)': ('2026-03-19', None),
  '公開後 廣橋らと同期間(3/19-7/23)': ('2026-03-19', '2026-07-23'),
  '廣橋ら以降の延長(7/24-10/7)': ('2026-07-24', None),
}
for p, (a, b) in periods.items():
    res[p] = {k: stats(sub(k, a, b)) for k in STR}

# 年次
yearly = {k: {str(y): round(sub(k)[sub(k).index.year == y].sum()*100, 2) for y in range(2015, 2027)} for k in STR}

# 月次（2025-）
monthly = {k: {str(m): round(v*100, 2) for m, v in sub(k, '2025-01-01').groupby(sub(k, '2025-01-01').index.to_period('M')).sum().items()} for k in STR}

# オーバーナイト vs 日中（同一ウェイト）
on = {}
for k in ['PCA_SUB', 'SCS', 'PCA_L1250']:
    m = A[k]
    on[k] = {}
    for p, (a, b) in [('公開前', (None, '2026-03-18')), ('公開後', ('2026-03-19', None)), ('延長(7/24-)', ('2026-07-24', None))]:
        x = RO[k][m]; y = R[k][m]
        if a: x, y = x[x.index >= a], y[y.index >= a]
        if b: x, y = x[x.index <= b], y[y.index <= b]
        on[k][p] = dict(overnight=x.mean()*252*100, intraday=y.mean()*252*100, N=len(x))

# プラセボ（廣橋ら式11）: h=公開後日数
plc = {}
for k in ['PCA_SUB', 'SCS', 'PCA_L1250']:
    r = sub(k); h = (r.index >= PUB).sum()
    pos = r.index.get_loc(r.index[r.index >= PUB][0])
    actual = r.iloc[pos:pos+h].mean() - r.iloc[pos-h:pos].mean()
    deltas = [r.iloc[c:c+h].mean() - r.iloc[c-h:c].mean() for c in range(h, pos-h+1)]
    deltas = np.array(deltas)
    plc[k] = dict(h=int(h), actual_ann=actual*252*100, pct_rank=(deltas < actual).mean()*100,
                  z=(actual-deltas.mean())/deltas.std(), min_ann=deltas.min()*252*100)

# 取引コスト：毎日 寄りで建て引けで解消 → 売買代金 = グロス2 × 往復2 = 資本の4倍
cost = {}
for k in STR:
    for p, (a, b) in [('公開前', (None, '2026-03-18')), ('公開後', ('2026-03-19', None))]:
        r = sub(k, a, b)
        be = r.mean()/4*1e4  # 片道あたり損益分岐 bps
        cost.setdefault(k, {})[p] = dict(breakeven_bps=be,
            AR_net_3bp=(r.mean()-4*3e-4)*252*100, AR_net_5bp=(r.mean()-4*5e-4)*252*100, AR_net_10bp=(r.mean()-4*10e-4)*252*100)

# 相関
corr = R[STR].loc[A[STR].all(1)].corr().round(2).to_dict()

# 累積系列（チャート用）
cum = {k: ((1+sub(k)).cumprod()).resample('W').last() for k in STR}
cum_post = {k: ((1+sub(k, '2025-10-01')).cumprod()) for k in ['PCA_SUB', 'SCS', 'PCA_L1250', 'MOM']}
on_post = {f'{k}_{t}': ((1+(RO if t=='on' else R)[k][A[k]].loc['2025-10-01':]).cumprod()) for k in ['PCA_SUB','SCS'] for t in ['on','oc']}

def ser(s): return {'d': [x.strftime('%Y-%m-%d') for x in s.index], 'v': [round(float(v), 5) for v in s.values]}
json.dump(dict(res=res, yearly=yearly, monthly=monthly, on=on, plc=plc, cost=cost, corr=corr,
               cum={k: ser(v) for k, v in cum.items()}, cum_post={k: ser(v) for k, v in cum_post.items()},
               on_post={k: ser(v) for k, v in on_post.items()}),
          open(f'{O}/summary.json', 'w'), ensure_ascii=False, indent=1)

pd.set_option('display.width', 200)
for p in res:
    print('\n==', p); print(pd.DataFrame(res[p]).T.round(2))
print('\n== 年次(%)'); print(pd.DataFrame(yearly).round(1))
print('\n== overnight/intraday'); print(json.dumps(on, ensure_ascii=False, indent=0))
print('\n== placebo'); print(pd.DataFrame(plc).T.round(2))
print('\n== cost'); print(pd.DataFrame({(k,p): v for k in cost for p, v in cost[k].items()}).T.round(2))
print('\n== corr'); print(pd.DataFrame(corr))
