import json, base64, io
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
S = json.load(open('out/summary.json'))
R = pd.read_csv('out/ret_oc.csv', index_col=0, parse_dates=True)
RO = pd.read_csv('out/ret_overnight.csv', index_col=0, parse_dates=True)
A = pd.read_csv('out/active.csv', index_col=0, parse_dates=True)
COL = {'SCS':'#1b7f5c','PCA_SUB':'#2a5db0','PCA_L1250':'#8a5cc2','PCA_PLAIN':'#9aa3ad','DOUBLE':'#d08a2c','MOM':'#c0504d'}
PUB = pd.Timestamp('2026-03-19')

def png(fig):
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=130, bbox_inches='tight'); plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()

# 図1 全期間累積（対数）
fig, ax = plt.subplots(figsize=(9, 4.2))
for k in ['SCS','PCA_L1250','PCA_SUB','DOUBLE','PCA_PLAIN','MOM']:
    r = R[k][A[k]]; ax.plot((1+r).cumprod(), label=k, color=COL[k], lw=1.4)
ax.set_yscale('log'); ax.axvline(PUB, color='k', ls='--', lw=0.8); ax.text(PUB, ax.get_ylim()[1]*0.6, ' 論文公開', fontsize=9)
ax.set_title('累積リターン（グロス・寄引け、2015/1〜2026/10/7、対数軸）'); ax.legend(ncol=3, fontsize=8, frameon=False); ax.grid(alpha=.3)
f1 = png(fig)

# 図2 公開前後（2025/10〜）
fig, ax = plt.subplots(figsize=(9, 4))
for k in ['SCS','PCA_SUB','PCA_L1250','MOM']:
    r = R[k][A[k]].loc['2025-10-01':]; ax.plot((1+r).cumprod(), label=k, color=COL[k], lw=1.6)
ax.axvline(PUB, color='k', ls='--', lw=.8); ax.axvline(pd.Timestamp('2026-07-23'), color='gray', ls=':', lw=.8)
ax.text(PUB, 1.27, ' 公開 3/19', fontsize=9); ax.text(pd.Timestamp('2026-07-23'), 1.27, ' 廣橋ら標本終了 7/23', fontsize=9, color='gray')
ax.set_title('公開前後の拡大（2025/10〜、寄引けリターン）'); ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.3)
f2 = png(fig)

# 図3 オーバーナイト vs 日中（PCA_SUB）20日移動平均・年率
fig, ax = plt.subplots(figsize=(9, 3.8))
for k, ls in [('PCA_SUB','-'), ('SCS','--')]:
    m = A[k]
    ax.plot(RO[k][m].loc['2025-06-01':].rolling(20).mean()*252*100, color='#d08a2c', ls=ls, lw=1.4, label=f'{k} オーバーナイト(前日終値→寄付)')
    ax.plot(R[k][m].loc['2025-06-01':].rolling(20).mean()*252*100, color=COL[k], ls=ls, lw=1.4, label=f'{k} 日中(寄付→引け)')
ax.axhline(0, color='k', lw=.6); ax.axvline(PUB, color='k', ls='--', lw=.8)
ax.set_title('同じウェイトで測ったリターンの分解（20日移動平均・年率%）'); ax.legend(fontsize=7.5, frameon=False, ncol=2); ax.grid(alpha=.3)
f3 = png(fig)

def tbl(p):
    d = S['res'][p]; rows = ''
    for k in ['SCS','PCA_L1250','PCA_SUB','DOUBLE','PCA_PLAIN','MOM']:
        x = d[k]
        rows += f"<tr><td>{k}</td><td>{x['AR']:.1f}%</td><td>{x['RISK']:.1f}%</td><td><b>{x['RR']:.2f}</b></td><td>{x['MDD']:.1f}%</td><td>{x['cum']:.1f}%</td><td>{x['t']:.2f}</td><td>{int(x['N'])}</td></tr>"
    return f"<table><tr><th>戦略</th><th>年率</th><th>リスク</th><th>R/R</th><th>MDD</th><th>累積</th><th>t値</th><th>日数</th></tr>{rows}</table>"

on = S['on']; plc = S['plc']; cost = S['cost']
on_rows = ''.join(f"<tr><td>{k}</td>" + ''.join(f"<td>{on[k][p]['overnight']:.0f}%</td><td>{on[k][p]['intraday']:.0f}%</td>" for p in ['公開前','公開後']) + "</tr>" for k in on)
cost_rows = ''.join(f"<tr><td>{k}</td><td>{cost[k]['公開前']['breakeven_bps']:.1f}</td><td>{cost[k]['公開前']['AR_net_3bp']:.1f}%</td><td>{cost[k]['公開前']['AR_net_5bp']:.1f}%</td><td>{cost[k]['公開後']['breakeven_bps']:.1f}</td></tr>" for k in ['SCS','PCA_L1250','PCA_SUB','DOUBLE','PCA_PLAIN','MOM'])
yr = pd.DataFrame(S['yearly'])[['SCS','PCA_L1250','PCA_SUB','DOUBLE','PCA_PLAIN','MOM']]
yr_rows = ''.join(f"<tr><td>{y}</td>" + ''.join(f"<td class='{'neg' if v<0 else ''}'>{v:.1f}</td>" for v in yr.loc[y]) + "</tr>" for y in yr.index)

html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>日米業種リードラグ検証</title><style>
:root{{--bg:#fff;--fg:#1d2329;--mut:#5d6873;--line:#e2e6ea;--acc:#2a5db0;--neg:#c0504d;--card:#f6f8fa}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16191d;--fg:#e6e9ec;--mut:#9aa3ad;--line:#2c3238;--acc:#7aa7ff;--neg:#ff7b72;--card:#1e2329}} img{{background:#fff;border-radius:6px}}}}
body{{background:var(--bg);color:var(--fg);font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;max-width:880px;margin:0 auto;padding:24px 16px;line-height:1.7}}
h1{{font-size:1.5rem;margin:.2em 0}} h2{{font-size:1.15rem;border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:2em}}
.mut{{color:var(--mut);font-size:.88rem}} table{{border-collapse:collapse;width:100%;font-size:.86rem;margin:.6em 0;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}}
.neg{{color:var(--neg)}} img{{width:100%;height:auto}} .card{{background:var(--card);border-radius:8px;padding:12px 16px;margin:12px 0}}
li{{margin:.25em 0}}</style></head><body>
<h1>日米業種リードラグ戦略の再現・検証</h1>
<div class="mut">データ: Yahoo Finance 日足（XLx 11本・NEXT FUNDS TOPIX-17 ETF 17本）2009/6〜2026/10/7 ／ 評価: 2015/1〜2026/10/7 ／ 作成 2026-10-08</div>

<div class="card"><b>結論</b><ul>
<li><b>論文の数字はおおむね再現できた。</b> 2015〜2025年のPCA_SUBはR/R 2.37（論文 2.22）、SCSはR/R 3.96（論文 3.60）。米国業種→翌日の日本業種という予測関係は、データの上では強く、安定している。</li>
<li><b>ただし公開（2026/3/19）後、PCA_SUBは寄引けでは大きく崩れた。</b> 3/19〜7/23の累積は −35.8%（廣橋らの報告は −30.8%）。プラセボ検定でも、過去のどの時点よりも大きな悪化だった。</li>
<li><b>その後（7/24〜10/7）、PCA_SUBは年率+25%に戻している。</b> ただし48営業日しかなく、t値は0.81。「回復した」と言える段階ではない。</li>
<li><b>シグナルの価値は寄付き前にほぼ価格に入っている。</b> 同じウェイトで測ると、オーバーナイト（前日終値→寄付）の取り分は公開前でも年率33〜56%あった。公開後は115〜138%に膨らみ、日中に残る分が減るか、マイナスになった。</li>
<li><b>取引コストが最大の壁。</b> 毎日、寄りで建てて引けで外すため、売買代金は資本の4倍になる。損益分岐は片道わずか2.6bp（PCA_SUB）〜4.0bp（SCS）で、片道5bpかかればすべての戦略がマイナス。さらに大半のETFは1日の売買代金が数千万円程度しかない。</li>
</ul></div>

<h2>1. 累積リターン</h2><img src="{f1}" alt="累積リターン">
<h2>2. 期間別パフォーマンス（グロス）</h2>
<p class="mut">年率は日次平均×252。R/R＝年率÷年率リスク。コスト控除前。</p>
<h3>公開前（2015/1〜2026/3/18）</h3>{tbl('公開前(2015-2026/3/18)')}
<h3>公開後・廣橋らと同じ期間（2026/3/19〜7/23）</h3>{tbl('公開後 廣橋らと同期間(3/19-7/23)')}
<h3>延長期間（2026/7/24〜10/7）</h3>{tbl('廣橋ら以降の延長(7/24-10/7)')}
<h3>公開後の全期間（2026/3/19〜10/7）</h3>{tbl('公開後 全期間(3/19-10/7)')}

<h2>3. 公開前後の拡大</h2><img src="{f2}" alt="公開前後">
<p>PCA_SUBは公開直後から5月にかけて急落し、7月下旬以降は横ばいから持ち直しへ。SCSは公開後もプラスだが、公開前（年率40%）よりは弱い（公開後は年率23%、t=1.24で有意ではない）。</p>

<h2>4. 寄付きで先回りされているか（オーバーナイト/日中の分解）</h2>
<table><tr><th>戦略</th><th>公開前 夜間</th><th>公開前 日中</th><th>公開後 夜間</th><th>公開後 日中</th></tr>{on_rows}</table>
<img src="{f3}" alt="分解">
<p>同じウェイトでも、公開後は「前日終値→寄付」の取り分が大きく膨らみ、「寄付→引け」の取り分が減っている。これは廣橋らの指摘（ETFの寄付きがシグナルの方向に先に動く）と同じ動き。ただし、今回はETFの価格しか見ておらず、TOPIX-17指数側との比較はしていない。そのため、市場全体で米国情報が早く織り込まれるようになった可能性も否定できない。また、図を見ると、PCA_SUBの日中リターンの悪化とオーバーナイトの膨張は、公開日の約1か月前（2026年2月）から始まっている。研究会での発表や事前の告知で、公開日より前に情報が出回っていた可能性もある。なお、PCA系のシグナルを持たないSCSでも夜間の取り分が同じように膨らんでいる。PCA_SUB固有の現象というより、「米国業種の動きを日本の寄付きに当てる」取引全般が混み合った可能性がある。</p>

<h2>5. プラセボ検定（廣橋ら 式11）</h2>
<table><tr><th>戦略</th><th>公開後日数h</th><th>実際の変化（年率）</th><th>疑似公開日の分布内の位置</th><th>z値</th></tr>
{''.join(f"<tr><td>{k}</td><td>{v['h']}</td><td>{v['actual_ann']:.1f}%</td><td>下から{v['pct_rank']:.1f}%</td><td>{v['z']:.2f}</td></tr>" for k,v in plc.items())}</table>
<p>PCA_SUBの悪化は、過去の疑似公開日での変化より必ず大きい（最小値をさらに下回る。z=−4.6）。SCSは通常の振れ幅の範囲内。</p>

<h2>6. 取引コスト感応度</h2>
<p class="mut">毎日、寄りでロング1・ショート1を建て、引けで解消する。売買代金は資本の4倍。片道コストをc bpとすると、1日あたり4c bpのコストになる。</p>
<table><tr><th>戦略</th><th>公開前 損益分岐(片道bp)</th><th>公開前 片道3bp控除後 年率</th><th>公開前 片道5bp控除後 年率</th><th>公開後 損益分岐(bp)</th></tr>{cost_rows}</table>
<p>TOPIX-17 ETFは売買代金が小さい。公開前6か月の1銘柄あたり日次売買代金の中央値は、多くの銘柄で500万〜3,000万円程度だった。公開後はおおむね2倍に増えている（1623は約6倍）。寄付きの板寄せで約定すればスプレッドは払わずに済むが、ロット次第で寄付き価格そのものを動かしてしまう。空売りは一般信用（日計り）が使える銘柄に限られる点にも注意。</p>

<h2>7. 年次リターン（%・単純合計）</h2>
<table><tr><th>年</th>{''.join(f'<th>{c}</th>' for c in yr.columns)}</tr>{yr_rows}</table>

<h2>8. 実装の仕様と論文との違い</h2><ul>
<li>PCA_SUB: L=60、λ=0.9、K=3、q=0.3（各サイド5業種・等ウェイト）。C<sub>full</sub>は2010–2014年。事前部分空間＝グローバル・日米スプレッド・景気敏感/ディフェンシブ（論文1の分類どおり）。</li>
<li>相関の推定には、同じ暦日の日米C2Cリターン（配当調整済み）を使った。日本の営業日dには、その直前の米国営業日s(d)の標準化リターンを当てた。XLRE・XLCの上場前は、相関0・標準化リターン0として扱った。</li>
<li>SCS: 金谷・吉田の表1の対応表を使用。同じスコアになった業種は、業種番号の若い順に並べた。PCA_L1250は、λ=0で窓1,250日のplain PCA。</li>
<li>執行: 日本ETFの始値で建て、終値で解消。出来高0の日は、その銘柄を選択対象から外した。Yahooの異常値（1629.Tの2026/3/30・31）は除外した。</li>
<li>論文と違う点: データ源（Bloombergではなく Yahoo）、評価開始（2015年）、年率化の方法（×252）。そのため水準は完全には一致しない。</li>
</ul>
<p class="mut">本レポートはバックテストの検証結果であり、投資判断の推奨ではありません。数値はすべてコスト控除前のグロスです。</p>
</body></html>"""
open('out/リードラグ検証レポート.html', 'w').write(html)
print('ok', len(html)//1024, 'KB')
