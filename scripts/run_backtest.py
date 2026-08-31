#!/usr/bin/env python3
"""実データでのバックテストを一括実行し、results/ に結果を書き出す。

この環境（および手元の開発環境）からは価格データ提供元に接続できないため、
GitHub Actions 上で実行して結果をリポジトリに残すことを想定している。

  python scripts/run_backtest.py --outdir results

出力:
  summary_main.csv       論文 表2 に対応する 4 戦略の要約
  summary_costs.csv      取引コスト 0/5/10/20bps での PCA_SUB と DOUBLE
  summary_expanding.csv  事前行列を完全に因果的に推定した場合
  yearly.csv             年次リターン
  sweep.csv              λ × K の感応度（R/R と AR）
  daily_returns.csv      日次リターン
  cumulative_returns.png 累積リターン図
  direction.csv          市場方向モデルの日次結果
  REPORT.md              上記のテキスト要約
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leadlag.backtest import run_all, run_strategy                 # noqa: E402
from leadlag.config import Params                                  # noqa: E402
from leadlag.data import load_bundle                               # noqa: E402
from leadlag.engine import LeadLagEngine                           # noqa: E402
from leadlag.metrics import (                                      # noqa: E402
    annual_return, annual_risk, max_drawdown, newey_west_tstat,
    risk_return, summary_table,
)
from leadlag.report import plot_cumulative                         # noqa: E402

PAPER = pd.DataFrame(
    {
        "AR": [5.63, 6.24, 23.79, 18.86],
        "RISK": [10.59, 9.94, 10.70, 11.16],
        "R/R": [0.53, 0.62, 2.22, 1.69],
        "MDD": [16.97, 23.65, 9.58, 12.10],
    },
    index=["MOM", "PCA_PLAIN", "PCA_SUB", "DOUBLE"],
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def data_coverage(us_panel: pd.DataFrame, jp_panel: pd.DataFrame) -> pd.DataFrame:
    """月ごとに、各市場で実際に値が入っている日数を数える。

    無料データはある期間だけまるごと欠けることがあり、そのまま共通営業日を
    取ると評価期間に穴が空く。穴の前後で推定ウィンドウが不連続な日付を
    またぐため、気づかないと結果が歪む。
    """
    def _count(panel: pd.DataFrame) -> pd.Series:
        close = panel.loc[:, [c for c in panel.columns if c[1] == "Close"]]
        has = close.notna().any(axis=1)
        return has.groupby([has.index.year, has.index.month]).sum()

    us, jp = _count(us_panel), _count(jp_panel)
    df = pd.DataFrame({"us": us, "jp": jp}).fillna(0).astype(int)
    df.index = [f"{y}-{m:02d}" for y, m in df.index]
    return df


def yearly_table(returns: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, r in returns.items():
        by_year = r.dropna().groupby(r.dropna().index.year)
        rows[name] = (by_year.apply(lambda s: (1 + s).prod() - 1) * 100).round(2)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--backtest-start", default=None,
                    help="既定は train_end の翌日（2015-01-01）")
    ap.add_argument("--cache", default="./data")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--sweep-lams", default="0,0.3,0.5,0.7,0.9,0.95,1.0")
    ap.add_argument("--sweep-ks", default="2,3,4,5")
    ap.add_argument("--data-only", action="store_true",
                    help="データの取得状況だけを調べて終了する（短時間）")
    a = ap.parse_args()
    if not a.backtest_start:          # ワークフローから空文字で渡ってくる場合
        a.backtest_start = None

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- データ ----------------
    if a.synthetic:
        from tests.synthetic import make_bundle
        bundle, _ = make_bundle(n_days=a.synthetic, seed=0, rho=0.4)
        base_kwargs = dict(prior_mode="expanding", prior_min_obs=300)
    else:
        _log("価格データを取得中…")
        from leadlag.data import build_bundle, load_prices

        us_panel, jp_panel = load_prices(start=a.start, cache_dir=a.cache, refresh=True)
        coverage = data_coverage(us_panel, jp_panel)
        coverage.to_csv(out / "data_coverage.csv")
        gaps = coverage[(coverage["us"] == 0) | (coverage["jp"] == 0)]
        if len(gaps):
            _log(f"⚠ 片方の市場のデータが 1 日も無い月が {len(gaps)} 件あります:")
            print(gaps.to_string(), flush=True)
        bundle = build_bundle(us_panel, jp_panel)
        base_kwargs = dict(prior_mode="fixed")

        common = pd.Series(1, index=bundle.dates)
        gap_days = bundle.dates.to_series().diff().dt.days
        big = gap_days[gap_days > 10]
        if len(big):
            _log(f"⚠ 共通営業日が {len(big)} 箇所で 10 日以上飛んでいます:")
            for dt, n in big.items():
                _log(f"   {dt.date()} の直前に {int(n)} 日の空白")

        if a.data_only:
            _log("--data-only のためここで終了します")
            return 0
    _log(f"共通営業日 {len(bundle.dates)} 日 "
         f"({bundle.dates[0].date()} 〜 {bundle.dates[-1].date()})")

    bundle.summary().to_csv(out / "universe_stats.csv")

    q = bundle.quality_report
    if q is not None and len(q):
        q.to_csv(out / "data_quality.csv", index=False)
        _log(f"異常値として除去したリターン {len(q)} 件:")
        print(q.to_string(), flush=True)
    else:
        (out / "data_quality.csv").write_text("kind,date,ticker,value\n")
        _log("異常値なし")

    base = Params(backtest_start=a.backtest_start, **base_kwargs)

    # ---------------- 本体（論文の設定） ----------------
    _log("メインのバックテスト (λ=0.9, K=3, L=60, q=0.3)…")
    panel = LeadLagEngine(bundle, base).run(verbose=True)
    result = run_all(panel, bundle, base)
    strat = {k: v.returns for k, v in result["strategies"].items()}

    main_tbl = summary_table(strat, ann=base.ann_factor)
    cmp_tbl = main_tbl[["AR", "RISK", "R/R", "MDD"]].join(
        PAPER, rsuffix="_paper"
    )[["AR", "AR_paper", "RISK", "RISK_paper", "R/R", "R/R_paper", "MDD", "MDD_paper"]]
    main_tbl.to_csv(out / "summary_main.csv")
    cmp_tbl.to_csv(out / "summary_vs_paper.csv")

    daily = pd.DataFrame(strat)
    daily["EW_BENCH"] = result["benchmark"]
    daily.to_csv(out / "daily_returns.csv")
    yearly_table({**strat, "EW_BENCH": result["benchmark"]}).to_csv(out / "yearly.csv")
    result["direction"].to_csv(out / "direction.csv")

    curves = dict(strat)
    curves["EW_BENCH"] = result["benchmark"]
    plot_cumulative(curves, out / "cumulative_returns.png",
                    title="JP-US sector lead-lag (L=60, K=3, lambda=0.9, q=0.3)")

    # 回転率
    turn = pd.DataFrame({k: v.turnover for k, v in result["strategies"].items()})
    turn.describe().to_csv(out / "turnover.csv")

    # ---------------- 取引コスト感応度 ----------------
    _log("取引コスト感応度…")
    rows = {}
    for bps in (0.0, 2.0, 5.0, 10.0, 20.0):
        p = Params(**{**base.to_dict(), "cost_bps": bps})
        for name, sig, second in (
            ("PCA_SUB", panel.pca_sub, None),
            ("DOUBLE", panel.mom, panel.pca_sub),
            ("MOM", panel.mom, None),
        ):
            r = run_strategy(sig, bundle, panel.execution_date, p,
                             second_signals=second).returns
            rows[(name, bps)] = {
                "AR": annual_return(r, p.ann_factor) * 100,
                "RISK": annual_risk(r, p.ann_factor) * 100,
                "R/R": risk_return(r, p.ann_factor),
                "MDD": max_drawdown(r),
                "t(NW)": newey_west_tstat(r),
            }
    cost_tbl = pd.DataFrame(rows).T.round(2)
    cost_tbl.index.names = ["Strategy", "cost_bps"]
    cost_tbl.to_csv(out / "summary_costs.csv")

    # ---------------- 完全に因果的な事前行列 ----------------
    _log("expanding（完全因果）版…")
    exp_p = Params(prior_mode="expanding", prior_min_obs=500, prior_refit_every=60,
                   backtest_start=a.backtest_start)
    exp_panel = LeadLagEngine(bundle, exp_p).run(verbose=True)
    exp_res = run_all(exp_panel, bundle, exp_p)
    exp_tbl = summary_table({k: v.returns for k, v in exp_res["strategies"].items()},
                            ann=exp_p.ann_factor)
    exp_tbl.to_csv(out / "summary_expanding.csv")

    # ---------------- λ × K 感応度 ----------------
    _log("λ × K の感応度…")
    lams = [float(x) for x in a.sweep_lams.split(",")]
    ks = [int(x) for x in a.sweep_ks.split(",")]
    sw = []
    for k in ks:
        for lam in lams:
            p = Params(**{**base.to_dict(), "lam": lam, "n_factors": k})
            try:
                pn = LeadLagEngine(bundle, p).run()
                r = run_strategy(pn.pca_sub, bundle, pn.execution_date, p).returns
                sw.append({"K": k, "lambda": lam,
                           "AR": round(annual_return(r, p.ann_factor) * 100, 2),
                           "R/R": round(risk_return(r, p.ann_factor), 3),
                           "MDD": round(max_drawdown(r), 2)})
            except Exception as exc:  # noqa: BLE001
                sw.append({"K": k, "lambda": lam, "AR": np.nan,
                           "R/R": np.nan, "MDD": np.nan, "error": str(exc)[:60]})
            _log(f"  K={k} λ={lam} -> R/R={sw[-1].get('R/R')}")
    sweep = pd.DataFrame(sw)
    sweep.to_csv(out / "sweep.csv", index=False)

    # ---------------- レポート ----------------
    dirn = result["direction"]
    ic = []
    for t in panel.pca_sub.index:
        s = panel.pca_sub.loc[t].dropna()
        r = bundle.jp_oc.loc[panel.execution_date.loc[t]].reindex(s.index)
        ok = r.notna()
        if ok.sum() > 5:
            ic.append(np.corrcoef(s[ok].rank(), r[ok].rank())[0, 1])
    ic = np.asarray(ic)

    md = [
        "# バックテスト結果",
        "",
        f"- データ: {bundle.dates[0].date()} 〜 {bundle.dates[-1].date()}"
        f"（日米共通営業日 {len(bundle.dates)} 日）",
        f"- 評価期間: {daily.index[0].date()} 〜 {daily.index[-1].date()}"
        f"（{len(daily)} 営業日）",
        "- 設定: L=60, K=3, λ=0.9, q=0.3, 事前行列は 2010-2014 で固定",
        "",
        "## 論文 表2 との比較（取引コストなし）",
        "", cmp_tbl.round(2).to_markdown(), "",
        "## 取引コスト感応度（片道 bps）",
        "", cost_tbl.to_markdown(), "",
        "## 年次リターン（%）",
        "", yearly_table({**strat, "EW_BENCH": result["benchmark"]}).to_markdown(), "",
        "## 事前行列を完全に因果的に推定した場合",
        "", exp_tbl.to_markdown(), "",
        "## λ × K 感応度（R/R）",
        "", sweep.pivot(index="K", columns="lambda", values="R/R").to_markdown(), "",
        "## λ × K 感応度（AR %）",
        "", sweep.pivot(index="K", columns="lambda", values="AR").to_markdown(), "",
        "## 情報係数（シグナルと翌日実現リターンの順位相関）",
        "",
        f"- 平均 IC: {ic.mean():.4f}",
        f"- IC の t 値: {ic.mean() / (ic.std(ddof=1) / np.sqrt(ic.size)):.2f}"
        f"（n={ic.size}）",
        "",
        "## 市場方向モデル",
        "",
        f"- 的中率: {dirn['correct'].mean() * 100:.2f}%（n={len(dirn)}）",
        f"- 方向で売買した場合の R/R: "
        f"{risk_return(dirn['timed_return'], base.ann_factor):.2f}",
        f"- 買い持ち（日中）の R/R: "
        f"{risk_return(dirn['jp_ew_oc_next'], base.ann_factor):.2f}",
        "",
        "## 回転率（1日あたり Σ|Δw|、グロス2倍のポートフォリオ）",
        "", turn.describe().round(3).to_markdown(), "",
        "## 除去した異常値",
        "",
        (q.to_markdown(index=False) if q is not None and len(q) else "なし"),
        "",
        "## 期間を区切った PCA_SUB",
        "",
        pd.DataFrame(
            {
                lab: {
                    "AR": annual_return(s, base.ann_factor) * 100,
                    "RISK": annual_risk(s, base.ann_factor) * 100,
                    "R/R": risk_return(s, base.ann_factor),
                    "MDD": max_drawdown(s),
                    "N": len(s),
                }
                for lab, s in {
                    "全期間": strat["PCA_SUB"],
                    "論文の標本内 (〜2025)": strat["PCA_SUB"][
                        strat["PCA_SUB"].index < "2026-01-01"
                    ],
                    "標本外 (2026〜)": strat["PCA_SUB"][
                        strat["PCA_SUB"].index >= "2026-01-01"
                    ],
                }.items()
            }
        ).T.round(2).to_markdown(),
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print()
    print("\n".join(md[:40]))
    _log(f"完了: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
