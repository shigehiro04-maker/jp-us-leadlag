"""日米業種リードラグ戦略の再現バックテスト
- 中川ら(2026) PCA_SUB / PCA_PLAIN / MOM / DOUBLE
- 金谷・吉田(2026) SCS（業種対応シグナル） / PCA長窓（L=1250）
- 廣橋ら(2026) 公開後アルファ減衰（ETF寄り付き価格）を延長検証
"""
import sys, json
import numpy as np, pandas as pd

SRC = sys.argv[1]; OUT = sys.argv[2]
US = ['XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY']
JP = [f'{c}.T' for c in range(1617, 1634)]
ALL = US + JP
NU, NJ = len(US), len(JP)
PUB = pd.Timestamp('2026-03-19')     # 公開後最初の日本営業日
LAST_JP = pd.Timestamp('2026-10-07')  # 10/8 は場中データのため除外

D = pd.read_csv(SRC, parse_dates=['date'])
D = D[D.date <= LAST_JP].sort_values(['ticker','date'])
# データ異常の除去：前後5日の中央値から ±50% 以上乖離した終値の行は欠損扱い（例: 1629.T 2026-03-30/31 の Yahoo 異常値）
med = D.groupby('ticker')['Close'].transform(lambda s: s.rolling(11, center=True, min_periods=3).median())
bad = (D['Close'] / med - 1).abs() > 0.5
print('除外行:', D.loc[bad, ['date','ticker','Close']].to_string())
D.loc[bad, ['Open','Close','Adj Close','Volume']] = np.nan
px = {k: D.pivot(index='date', columns='ticker', values=k) for k in ['Open','Close','Adj Close','Volume']}

# ---------- リターン ----------
# C2C（配当調整済み）: 各市場の自国営業日ベース
adj = px['Adj Close']
us_adj = adj[US].dropna(how='all'); us_adj = us_adj[us_adj[['XLK','XLF']].notna().all(axis=1)]
jp_adj = adj[JP].dropna(how='all'); jp_adj = jp_adj[jp_adj['1617.T'].notna()]
us_cc = us_adj.pct_change(fill_method=None)
jp_cc = jp_adj.pct_change(fill_method=None)
# 日本 O2C（約定のあった日のみ）
jo, jc, jv = px['Open'][JP].loc[jp_adj.index], px['Close'][JP].loc[jp_adj.index], px['Volume'][JP].loc[jp_adj.index]
valid = (jv > 0) & (jo > 0) & (jc > 0)
jp_oc = (jc / jo - 1).where(valid)
# 日本 C2O（オーバーナイト、配当落ち調整のため adj 比率で補正）
adjf = (jp_adj / jc)                                   # 調整係数
jp_co = ((jo * adjf) / (jc * adjf).shift(1) - 1).where(valid)

# ---------- 共通営業日パネル（論文1と同じく同一暦日で結合） ----------
common = us_cc.index.intersection(jp_cc.index)
R = pd.concat([us_cc.loc[common], jp_cc.loc[common]], axis=1)[ALL]
R = R.iloc[1:]
# 日本営業日 d → 直前の米国営業日 s(d)
us_dates = us_cc.index
def s_of(d):
    i = us_dates.searchsorted(d) - 1
    return us_dates[i] if i >= 0 else None

# ---------- 事前部分空間 ----------
cyc = {'XLB':1,'XLE':1,'XLF':1,'XLRE':1,'XLK':-1,'XLP':-1,'XLU':-1,'XLV':-1,
       '1618.T':1,'1625.T':1,'1629.T':1,'1631.T':1,'1617.T':-1,'1621.T':-1,'1627.T':-1,'1630.T':-1}
v1 = np.ones(len(ALL)); v2 = np.r_[np.ones(NU), -np.ones(NJ)]; v3 = np.array([cyc.get(a, 0) for a in ALL], float)
V0 = []
for v in [v1, v2, v3]:
    for u in V0: v = v - (v @ u) * u
    V0.append(v / np.linalg.norm(v))
V0 = np.column_stack(V0)

def corr_fill(X):
    """欠測資産は相関0・対角1"""
    C = pd.DataFrame(X).corr(min_periods=20).values
    C = np.nan_to_num(C, nan=0.0); np.fill_diagonal(C, 1.0)
    return C

full = R.loc['2010-01-01':'2014-12-31']
Cfull = corr_fill(full.values)
D0 = np.diag(np.diag(V0.T @ Cfull @ V0))
Craw = V0 @ D0 @ V0.T
dd = np.sqrt(np.diag(Craw)); C0 = Craw / np.outer(dd, dd); np.fill_diagonal(C0, 1.0)

def pca_signal(C, zU, K=3):
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(w)[::-1][:K]]
    return V[NU:] @ (V[:NU].T @ zU)

# ---------- 日次シグナル ----------
SCS_MAP = {'1617.T':'XLP','1618.T':'XLE','1619.T':'XLI','1620.T':'XLB','1621.T':'XLV','1622.T':'XLY',
           '1623.T':'XLB','1624.T':'XLI','1625.T':'XLK','1626.T':'XLC','1627.T':'XLU','1628.T':'XLI',
           '1629.T':'XLI','1630.T':'XLY','1631.T':'XLF','1632.T':'XLF','1633.T':'XLRE'}

Rv = R.values; Ridx = R.index
jp_days = jp_oc.index[(jp_oc.index >= '2015-01-01')]
sig = {k: {} for k in ['PCA_SUB','PCA_PLAIN','PCA_L1250','MOM','SCS']}
for d in jp_days:
    t = s_of(d)
    if t is None or t not in Ridx: continue
    i = Ridx.get_loc(t)
    for name, L, lam in [('PCA_SUB',60,0.9),('PCA_PLAIN',60,0.0),('PCA_L1250',1250,0.0)]:
        if i < L: continue
        W = Rv[i-L:i]
        mu, sd = np.nanmean(W, 0), np.nanstd(W, 0)
        sd = np.where((sd > 0) & np.isfinite(sd), sd, np.nan)
        Z = (W - mu) / sd
        C = corr_fill(Z)
        Ct = (1-lam)*C + lam*C0
        zU = np.nan_to_num((Rv[i, :NU] - mu[:NU]) / sd[:NU])
        sig[name][d] = pca_signal(Ct, zU)
    if i >= 60:
        sig['MOM'][d] = np.nanmean(Rv[i-60:i, NU:], 0)   # 日本側 窓内平均（論文式31）
    sig['SCS'][d] = np.array([np.nan_to_num(us_cc.loc[t, SCS_MAP[j]]) for j in JP])

S = {k: pd.DataFrame.from_dict(v, orient='index', columns=JP).sort_index() for k, v in sig.items()}

def weights(sdf, q=0.3):
    W = pd.DataFrame(0.0, index=sdf.index, columns=JP)
    for d, row in sdf.iterrows():
        ok = valid.loc[d].reindex(JP).fillna(False).values & np.isfinite(row.values)
        idx = np.where(ok)[0]
        if len(idx) < 10: continue
        n = max(1, int(round(q * len(idx))))
        # 同点は業種番号順（安定ソート）
        order = idx[np.argsort(-row.values[idx], kind='stable')]
        W.loc[d, JP[0]] = 0  # noop
        w = np.zeros(NJ); w[order[:n]] = 1/n; w[order[-n:]] = -1/n
        W.loc[d] = w
    return W

Wd = {k: weights(v) for k, v in S.items()}
# DOUBLE: MOM と PCA_SUB の中央値ダブルソート
idx = S['PCA_SUB'].index.intersection(S['MOM'].index)
Wdb = pd.DataFrame(0.0, index=idx, columns=JP)
for d in idx:
    a, b = S['PCA_SUB'].loc[d].values, S['MOM'].loc[d].values
    hi = (a > np.median(a)) & (b > np.median(b)); lo = (a < np.median(a)) & (b < np.median(b))
    hi &= valid.loc[d].values; lo &= valid.loc[d].values
    w = np.zeros(NJ)
    if hi.sum() and lo.sum(): w[hi] = 1/hi.sum(); w[lo] = -1/lo.sum()
    Wdb.loc[d] = w
Wd['DOUBLE'] = Wdb

oc = jp_oc.fillna(0.0); co = jp_co.fillna(0.0)
ret = pd.DataFrame({k: (W * oc.loc[W.index]).sum(1) for k, W in Wd.items()})
ret_on = pd.DataFrame({k: (W * co.loc[W.index]).sum(1) for k, W in Wd.items()})
active = pd.DataFrame({k: (W.abs().sum(1) > 0) for k, W in Wd.items()})

ret.to_csv(f'{OUT}/ret_oc.csv'); ret_on.to_csv(f'{OUT}/ret_overnight.csv'); active.to_csv(f'{OUT}/active.csv')
for k, v in S.items(): v.to_csv(f'{OUT}/signal_{k}.csv')
for k, v in Wd.items(): v.to_csv(f'{OUT}/weights_{k}.csv')
jp_oc.to_csv(f'{OUT}/jp_oc.csv'); jp_co.to_csv(f'{OUT}/jp_co.csv'); jv.to_csv(f'{OUT}/jp_volume.csv')
print('done', ret.index.min().date(), ret.index.max().date(), len(ret))
