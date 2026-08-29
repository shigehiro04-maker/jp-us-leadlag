"""コマンドラインインターフェース.

  python -m leadlag.cli backtest            # 論文の再現バックテスト
  python -m leadlag.cli predict             # NY 終値から翌日の東京を予測
  python -m leadlag.cli data --refresh      # 価格データの再取得のみ
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import cross_section_weights, run_all
from .config import DEFAULT_PARAMS, Params, display_name
from .data import bundle_from_csv, load_bundle
from .engine import LeadLagEngine
from .metrics import summary_table
from .report import format_summary, plot_signal_heat, write_report


# ---------------------------------------------------------------------------
def _add_common(ap: argparse.ArgumentParser) -> None:
    d = DEFAULT_PARAMS
    ap.add_argument("--start", default=d.start, help="データ取得開始日")
    ap.add_argument("--end", default=d.end, help="データ取得終了日 (既定: 最新)")
    ap.add_argument("--window", type=int, default=d.window, help="推定ウィンドウ L")
    ap.add_argument("--factors", type=int, default=d.n_factors, help="主成分数 K")
    ap.add_argument("--lam", type=float, default=d.lam, help="正則化強度 λ")
    ap.add_argument("--quantile", type=float, default=d.quantile, help="ロング/ショート分位 q")
    ap.add_argument("--prior-mode", choices=["fixed", "expanding"], default=d.prior_mode)
    ap.add_argument("--train-start", default=d.train_start)
    ap.add_argument("--train-end", default=d.train_end)
    ap.add_argument("--cost-bps", type=float, default=d.cost_bps,
                    help="片道取引コスト(bps)。論文の再現は 0")
    ap.add_argument("--ann-factor", type=int, default=d.ann_factor)
    ap.add_argument("--lag-jp-in-corr", action="store_true",
                    help="相関行列を作る際に日本側を 1 日ずらす (論文外の変種)")
    ap.add_argument("--cache", default="./data", help="価格キャッシュの保存先")
    ap.add_argument("--refresh", action="store_true", help="キャッシュを無視して再取得")
    ap.add_argument("--us-csv", help="自前 CSV を使う場合の米国データ")
    ap.add_argument("--jp-csv", help="自前 CSV を使う場合の日本データ")
    ap.add_argument("--synthetic", type=int, metavar="N_DAYS", default=0,
                    help="ネットワーク不要の合成データで動作確認する (例: --synthetic 1500)")


def _params_from_args(a) -> Params:
    if getattr(a, "synthetic", 0) and a.prior_mode == "fixed":
        # 合成データは日付が実データと揃わないため expanding に切り替える
        a.prior_mode = "expanding"
    return Params(
        window=a.window,
        n_factors=a.factors,
        lam=a.lam,
        quantile=a.quantile,
        prior_mode=a.prior_mode,
        train_start=a.train_start,
        train_end=a.train_end,
        start=a.start,
        end=a.end,
        backtest_start=getattr(a, "backtest_start", None),
        cost_bps=a.cost_bps,
        ann_factor=a.ann_factor,
        lag_jp_in_corr=a.lag_jp_in_corr,
    )


def _load(a):
    if getattr(a, "synthetic", 0):
        from tests.synthetic import make_bundle

        print(f"[合成データモード] {a.synthetic} 営業日ぶんの疑似データで実行します")
        bundle, _ = make_bundle(n_days=a.synthetic, seed=0, rho=0.4)
        return bundle
    if a.us_csv and a.jp_csv:
        return bundle_from_csv(a.us_csv, a.jp_csv)
    return load_bundle(start=a.start, end=a.end, cache_dir=a.cache, refresh=a.refresh)


# ---------------------------------------------------------------------------
def cmd_data(a) -> int:
    bundle = _load(a)
    print(f"共通営業日: {bundle.dates[0].date()} 〜 {bundle.dates[-1].date()} "
          f"({len(bundle.dates)} 日)")
    print()
    print(bundle.summary().to_string())
    return 0


def cmd_backtest(a) -> int:
    params = _params_from_args(a)
    bundle = _load(a)
    print(f"データ: {bundle.dates[0].date()} 〜 {bundle.dates[-1].date()} "
          f"({len(bundle.dates)} 共通営業日)")

    engine = LeadLagEngine(bundle, params)
    panel = engine.run(verbose=True)
    result = run_all(panel, bundle, params)

    outdir = Path(a.outdir)
    rep = write_report(result, params, outdir)

    print("\n=== 各戦略の要約統計 (論文 表2 に対応) ===")
    print(format_summary(rep["table"]))

    print("\n=== 市場全体の方向予測 (ローリング回帰: 米国EW → 翌日東京EWの日中) ===")
    dirn = rep["direction"]
    naive = rep["direction_naive"]
    print(f"的中率 (回帰)       : {dirn['correct'].mean()*100:.2f}%  (n={len(dirn)})")
    print(f"的中率 (米国符号のみ): {naive['correct'].mean()*100:.2f}%")
    print(f"上昇予想の割合      : {dirn['pred_up'].mean()*100:.2f}%")
    print(f"実際の上昇の割合    : {dirn['actual_up'].mean()*100:.2f}%")
    print(f"直近の回帰係数      : " +
          ", ".join(f"{c}={dirn[c].iloc[-1]:+.4f}"
                    for c in dirn.columns if c.startswith("beta_")))
    print(summary_table(
        {
            "TIMING(方向で売買)": dirn["timed_return"],
            "BUY&HOLD(日中)": dirn["jp_ew_oc_next"],
        },
        ann=params.ann_factor,
    ).to_string())

    timing = rep["timing"]
    lvl = timing["signal_mean"].std() / panel.pca_sub.stack().std()
    print("\n--- 参考: リードラグ・シグナル断面平均による方向予測 ---")
    print(f"的中率            : {timing['correct'].mean()*100:.2f}%")
    print(f"断面平均の相対的な大きさ: {lvl:.4f} "
          "(1 に近いほど市場方向の情報を持つ。0 に近いのは設計どおり市場中立だから)")

    if a.cost_bps == 0:
        print("\n注: 取引コスト 0 (論文と同条件)。--cost-bps 5 などで感応度を確認してください。")

    print(f"\n出力: {outdir.resolve()}")
    for k, v in rep["paths"].items():
        print(f"  {k:8s} -> {v.name}")
    return 0


def cmd_sweep(a) -> int:
    """λ (正則化強度) と K (主成分数) を変えて R/R がどう動くかを見る。

    λ=0.9 は事前部分空間への縮約がかなり強く、結果がほぼ
    「シクリカル vs ディフェンシブ」のラベルだけで決まる領域に入る。
    実データでどの水準が妥当かはここで確認すること。
    """
    from .metrics import risk_return

    base = _params_from_args(a)
    bundle = _load(a)
    lams = [float(x) for x in a.lams.split(",")]
    ks = [int(x) for x in a.ks.split(",")]

    rows = []
    for k in ks:
        for lam in lams:
            p = Params(**{**base.to_dict(), "lam": lam, "n_factors": k})
            try:
                panel = LeadLagEngine(bundle, p).run()
                res = run_all(panel, bundle, p)
                r = res["strategies"]["PCA_SUB"].returns
                rows.append({"K": k, "lambda": lam,
                             "R/R": round(risk_return(r, p.ann_factor), 3),
                             "AR(%)": round(r.mean() * p.ann_factor * 100, 2)})
            except Exception as exc:  # noqa: BLE001
                rows.append({"K": k, "lambda": lam, "R/R": np.nan,
                             "AR(%)": np.nan, "error": str(exc)[:40]})
            print(f"  K={k} λ={lam}: {rows[-1].get('R/R')}", flush=True)

    df = pd.DataFrame(rows)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "sweep.csv", index=False)

    print("\n=== R/R の感応度 (行=K, 列=λ) ===")
    print(df.pivot(index="K", columns="lambda", values="R/R").to_string())
    print(f"\n出力: {(outdir / 'sweep.csv').resolve()}")
    return 0


def cmd_predict(a) -> int:
    params = _params_from_args(a)
    bundle = _load(a)
    engine = LeadLagEngine(bundle, params)
    asof, res = engine.latest()

    s = pd.Series(res.scores, index=res.jp_tickers).sort_values(ascending=False)
    w = cross_section_weights(s, params.quantile)

    df = pd.DataFrame(
        {
            "業種": [display_name(t) for t in s.index],
            "シグナル": s.round(4),
            "ウェイト": w.reindex(s.index).round(4),
        }
    )
    df["建玉"] = np.where(df["ウェイト"] > 0, "ロング",
                          np.where(df["ウェイト"] < 0, "ショート", "-"))

    us_z = pd.Series(res.z_us, index=res.us_tickers).sort_values(ascending=False)
    mkt = float(s.mean())
    disp = float(s.std(ddof=1))

    print("=" * 68)
    print(f"基準となる米国終値の日付 : {asof.date()}")
    print(f"予測対象                 : 翌営業日の東京市場 (寄付き→大引け)")
    print(f"モデル                   : L={params.window}, K={params.n_factors}, "
          f"λ={params.lam}, q={params.quantile}, prior={params.prior_mode}")
    print("=" * 68)

    from .direction import latest_direction

    print("\n【市場全体の地合い】(ローリング回帰による別モデル)")
    try:
        md = latest_direction(bundle, asof=asof)
        direction = md["direction"]
        print(f"  当日の米国EWリターン : {md['us_ew_cc']*100:+.2f}%")
        print(f"  翌日の東京EW日中予測 : {md['pred']*100:+.2f}%  → {direction}")
        print(f"  予測の残差標準偏差   : {md['resid_sd']*100:.2f}%  "
              f"(学習 {md['n_train']} 日)")
    except RuntimeError as e:
        direction = "判定不能"
        md = {"pred": float("nan")}
        print(f"  {e}")
    print(f"  ※ 1 日先の方向の的中率はせいぜい 55% 前後です。過信しないでください。")

    print("\n【業種シグナルの補足】")
    print(f"  シグナル断面平均 : {mkt:+.4f} (設計上ほぼ 0。市場方向の情報は持ちません)")
    print(f"  断面ばらつき     : {disp:.4f}  (大きいほど業種間の差が出やすい)")
    print(f"  共通ファクター f : " +
          ", ".join(f"f{i+1}={v:+.3f}" for i, v in enumerate(res.factor_scores)))

    print("\n【米国業種の当日ショック (窓内標準化, z)】")
    for t, v in us_z.items():
        print(f"  {t:5s} {display_name(t):<12s} {v:+.2f}")

    print("\n【翌日の東京・業種ランキング】")
    print(df.to_string())

    if a.json:
        payload = {
            "asof_us_close": str(asof.date()),
            "market_direction": direction,
            "market_pred_return": md.get("pred"),
            "signal_mean": mkt,
            "signal_dispersion": disp,
            "factor_scores": [float(x) for x in res.factor_scores],
            "us_shock_z": {k: float(v) for k, v in us_z.items()},
            "sectors": [
                {
                    "ticker": t,
                    "name": display_name(t),
                    "signal": float(s[t]),
                    "weight": float(w.get(t, 0.0)),
                }
                for t in s.index
            ],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        Path(a.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\nJSON を書き出しました: {a.json}")

    if a.heatmap:
        p = plot_signal_heat(res.b_matrix, res.us_tickers, res.jp_tickers,
                             Path(a.heatmap))
        print(f"伝播行列のヒートマップ: {p}")

    print("\n※ これは論文手法の再現であり、投資助言ではありません。")
    return 0


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="leadlag",
        description="日米業種リードラグ戦略 (部分空間正則化付き PCA)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_data = sub.add_parser("data", help="価格データの取得と基本統計量")
    _add_common(p_data)
    p_data.set_defaults(func=cmd_data)

    p_bt = sub.add_parser("backtest", help="論文の再現バックテスト")
    _add_common(p_bt)
    p_bt.add_argument("--backtest-start", default=None,
                      help="バックテスト開始日 (既定: train_end の翌日)")
    p_bt.add_argument("--outdir", default="./output")
    p_bt.set_defaults(func=cmd_backtest)

    p_sw = sub.add_parser("sweep", help="λ と K の感応度分析")
    _add_common(p_sw)
    p_sw.add_argument("--backtest-start", default=None)
    p_sw.add_argument("--lams", default="0,0.3,0.5,0.7,0.9,0.95,1.0")
    p_sw.add_argument("--ks", default="2,3,4,5")
    p_sw.add_argument("--outdir", default="./output")
    p_sw.set_defaults(func=cmd_sweep)

    p_pr = sub.add_parser("predict", help="最新の NY 終値から翌日の東京を予測")
    _add_common(p_pr)
    p_pr.add_argument("--json", help="結果を JSON で保存するパス")
    p_pr.add_argument("--heatmap", help="伝播行列のヒートマップ PNG 保存先")
    p_pr.set_defaults(func=cmd_predict)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
