#!/usr/bin/env python3
"""改善案の検証（本番モデルには手を入れない、純粋な調査用）。

ライブ運用で観測された問題:
  業種ランキングの前日との順位相関が |ρ| ≈ 1 のまま符号だけ日替わりで反転する。
  つまりランキングは実質 1 本の軸で、その符号が毎日ひっくり返っている。
  これは伝播行列 B = V_J V_U' のクロス成分が実質 v3（シクリカル/ディフェンシブ）
  だけになるという構造から予想されることで、バグではない。

ここで測るのは次の 3 点。

  1. 平滑化がその反転を実際に抑えるか、抑えた結果として成績が改善するか。
     素の EWMA だけでなく、断面 z 化してから EWMA する版も試す。シグナルの
     絶対値は日によって 20 倍も振れるので、素のままだと一部の日が平均を支配する。
  2. λ と K を変えたとき、直近レジーム（2024 年以降、2026 年）で生き残る設定が
     あるか。全期間で最良でも直近で壊れていれば意味がない。
  3. 平滑化で回転率がどれだけ下がり、損益分岐コストがどこまで上がるか。
     現状の損益分岐は片道 4.6bp で、現実的なコスト（10〜30bp）に届いていない。

  python scripts/run_experiments.py --outdir results_exp
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leadlag.backtest import run_strategy                            # noqa: E402
from leadlag.config import Params                                    # noqa: E402
from leadlag.engine import LeadLagEngine                             # noqa: E402
from leadlag.metrics import (                                        # noqa: E402
    annual_return, annual_risk, max_drawdown, newey_west_tstat, risk_return,
)

# 評価する期間の切り方。論文の標本内と、直近レジームを分けて見る。
PERIODS = {
    "全期間": (None, None),
    "〜2023": (None, "2024-01-01"),
    "2024〜": ("2024-01-01", None),
    "2026〜": ("2026-01-01", None),
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 平滑化
# ---------------------------------------------------------------------------
def cross_sectional_z(panel: pd.DataFrame) -> pd.DataFrame:
    """各日のシグナルを断面で標準化する。

    シグナルの絶対値は伝播行列の増幅率次第で日によって桁が変わる。順位だけを
    使う戦略には影響しないが、日をまたいで平均するとスケールの大きい日が
    平均を支配してしまうので、平滑化の前にそろえておく。
    """
    mu = panel.mean(axis=1)
    sd = panel.std(axis=1, ddof=0)
    sd = sd.where(sd > 1e-12, np.nan)
    return panel.sub(mu, axis=0).div(sd, axis=0)


def smooth(panel: pd.DataFrame, halflife: float | None, standardize: bool) -> pd.DataFrame:
    """過去のシグナルだけを使う指数移動平均（先読みなし）。"""
    src = cross_sectional_z(panel) if standardize else panel
    if halflife is None:
        return src
    return src.ewm(halflife=halflife, adjust=False, ignore_na=True).mean()


# ---------------------------------------------------------------------------
# 診断
# ---------------------------------------------------------------------------
def rank_flip_stats(panel: pd.DataFrame) -> dict:
    """前日とのランキングの一致度を測る。

    rho_mean が 0 付近でも安定しているとは限らない。実データでは |rho| が
    ほぼ 1 のまま符号が反転していたので、|rho| の平均と符号が反転した日の
    割合を別々に出す。
    """
    rhos = []
    prev = None
    for _, row in panel.iterrows():
        s = row.dropna()
        if s.size < 5:
            prev = None
            continue
        r = s.rank()
        if prev is not None:
            common = r.index.intersection(prev.index)
            if common.size >= 5:
                a = r[common].to_numpy(dtype=float)
                b = prev[common].to_numpy(dtype=float)
                if a.std() > 0 and b.std() > 0:
                    rhos.append(float(np.corrcoef(a, b)[0, 1]))
        prev = r
    if not rhos:
        return {"rho_mean": np.nan, "abs_rho_mean": np.nan,
                "flip_share(%)": np.nan, "N": 0}
    x = np.asarray(rhos)
    return {
        "rho_mean": float(x.mean()),
        "abs_rho_mean": float(np.abs(x).mean()),
        "flip_share(%)": float((x < 0).mean() * 100.0),
        "N": int(x.size),
    }


def information_coefficient(panel: pd.DataFrame, bundle, exec_dates) -> dict:
    """シグナルと翌日の実現リターンの順位相関（IC）。"""
    ic = []
    for t in panel.index:
        s = panel.loc[t].dropna()
        if s.size < 6:
            continue
        r = bundle.jp_oc.loc[exec_dates.loc[t]].reindex(s.index)
        ok = r.notna()
        if ok.sum() > 5:
            ic.append(float(np.corrcoef(s[ok].rank(), r[ok].rank())[0, 1]))
    if len(ic) < 10:
        return {"IC": np.nan, "IC_t": np.nan, "IC_N": len(ic)}
    x = np.asarray(ic)
    return {
        "IC": float(x.mean()),
        "IC_t": float(x.mean() / (x.std(ddof=1) / np.sqrt(x.size))),
        "IC_N": int(x.size),
    }


def stats(res, ann: int) -> dict:
    r = res.returns.dropna()
    g = res.gross_returns.dropna()
    turn = float(res.turnover.mean())
    # 損益分岐コスト（片道 bps）: グロスの平均日次リターンを回転率で割る
    be = float(g.mean() / turn * 1e4) if turn > 1e-9 else np.nan
    return {
        "AR": annual_return(r, ann) * 100,
        "RISK": annual_risk(r, ann) * 100,
        "R/R": risk_return(r, ann),
        "MDD": max_drawdown(r),
        "t(NW)": newey_west_tstat(r),
        "回転率": turn,
        "損益分岐bp": be,
        "N": int(r.size),
    }


def slice_returns(res, lo, hi):
    """StrategyResult を期間で切り出した簡易オブジェクトを返す。"""
    class _S:
        pass

    s = _S()
    idx = res.returns.index
    m = np.ones(len(idx), dtype=bool)
    if lo is not None:
        m &= idx >= pd.Timestamp(lo)
    if hi is not None:
        m &= idx < pd.Timestamp(hi)
    s.returns = res.returns[m]
    s.gross_returns = res.gross_returns[m]
    s.turnover = res.turnover[m]
    s.weights = res.weights[m]
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_exp")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--backtest-start", default=None)
    ap.add_argument("--cache", default="./data")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--halflives", default="1,2,3,5,10,20")
    ap.add_argument("--lams", default="0.5,0.7,0.9,0.95")
    ap.add_argument("--ks", default="3,4,5")
    ap.add_argument("--costs", default="0,5,10,20")
    a = ap.parse_args()
    if not a.backtest_start:
        a.backtest_start = None

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    if a.synthetic:
        from tests.synthetic import make_bundle
        bundle, _ = make_bundle(n_days=a.synthetic, seed=0, rho=0.4)
        base_kwargs = dict(prior_mode="expanding", prior_min_obs=300)
    else:
        _log("価格データを取得中…")
        from leadlag.data import build_bundle, load_prices
        us_panel, jp_panel = load_prices(start=a.start, cache_dir=a.cache, refresh=True)
        bundle = build_bundle(us_panel, jp_panel)
        base_kwargs = dict(prior_mode="fixed")
    _log(f"共通営業日 {len(bundle.dates)} 日 "
         f"({bundle.dates[0].date()} 〜 {bundle.dates[-1].date()})")

    halflives: list[float | None] = [None] + [float(x) for x in a.halflives.split(",")]
    lams = [float(x) for x in a.lams.split(",")]
    ks = [int(x) for x in a.ks.split(",")]
    costs = [float(x) for x in a.costs.split(",")]

    base = Params(backtest_start=a.backtest_start, **base_kwargs)

    # ----- 1. 平滑化（論文の λ=0.9, K=3 を固定して平滑化だけを動かす） -----
    _log("1/3 平滑化の効果…")
    panel = LeadLagEngine(bundle, base).run(verbose=True)
    ex = panel.execution_date

    flip_rows, smooth_rows = {}, {}
    for standardize in (False, True):
        tag = "z化" if standardize else "素"
        for hl in halflives:
            sig = smooth(panel.pca_sub, hl, standardize)
            name = f"{tag}/HL={'なし' if hl is None else hl}"
            flip_rows[name] = rank_flip_stats(sig)
            flip_rows[name].update(
                information_coefficient(sig, bundle, ex)
            )
            for bps in costs:
                p = Params(**{**base.to_dict(), "cost_bps": bps})
                res = run_strategy(sig, bundle, ex, p)
                for plabel, (lo, hi) in PERIODS.items():
                    sub = slice_returns(res, lo, hi)
                    if len(sub.returns) < 30:
                        continue
                    smooth_rows[(tag, "なし" if hl is None else hl, bps, plabel)] = (
                        stats(sub, p.ann_factor)
                    )
            _log(f"  {name} 完了")

    flip = pd.DataFrame(flip_rows).T.round(3)
    flip.to_csv(out / "rank_stability.csv")

    sm = pd.DataFrame(smooth_rows).T.round(3)
    sm.index.names = ["標準化", "半減期", "コストbp", "期間"]
    sm.to_csv(out / "smoothing.csv")

    # ----- 2. λ × K を直近レジームで評価 -----
    _log("2/3 λ × K を期間別に…")
    grid_rows = {}
    for k in ks:
        for lam in lams:
            p = Params(**{**base.to_dict(), "lam": lam, "n_factors": k})
            try:
                pn = LeadLagEngine(bundle, p).run()
            except Exception as exc:                       # noqa: BLE001
                _log(f"  K={k} λ={lam} 失敗: {exc}")
                continue
            for hl in (None, 5.0):
                sig = smooth(pn.pca_sub, hl, standardize=True)
                res = run_strategy(sig, bundle, pn.execution_date, p)
                for plabel, (lo, hi) in PERIODS.items():
                    sub = slice_returns(res, lo, hi)
                    if len(sub.returns) < 30:
                        continue
                    grid_rows[(k, lam, "なし" if hl is None else hl, plabel)] = (
                        stats(sub, p.ann_factor)
                    )
            _log(f"  K={k} λ={lam} 完了")
    grid = pd.DataFrame(grid_rows).T.round(3)
    grid.index.names = ["K", "lambda", "半減期", "期間"]
    grid.to_csv(out / "grid.csv")

    # ----- 3. レポート -----
    _log("3/3 レポート…")

    def pivot(df, value, **fixed):
        d = df.reset_index()
        for k_, v_ in fixed.items():
            d = d[d[k_] == v_]
        return d

    lines = [
        "# 改善案の検証",
        "",
        f"- データ: {bundle.dates[0].date()} 〜 {bundle.dates[-1].date()}",
        "- 設定: L=60, q=0.3, 事前行列は固定（論文どおり）",
        "- 平滑化は過去のシグナルのみを使う指数移動平均（先読みなし）",
        "",
        "## 1. ランキングの安定性と情報係数",
        "",
        "`abs_rho_mean` が 1 に近く `flip_share` が 50% 付近なら、"
        "「同じ 1 本の軸の符号が毎日ひっくり返っている」状態。",
        "", flip.to_markdown(), "",
        "## 2. 平滑化の効果（コスト 0bp、期間別）",
        "",
        pivot(sm, "R/R", コストbp=0.0)
        .pivot(index=["標準化", "半減期"], columns="期間", values="R/R")
        .to_markdown(),
        "",
        "## 3. 平滑化の効果（現実的なコスト 10bp、期間別 R/R）",
        "",
        pivot(sm, "R/R", コストbp=10.0)
        .pivot(index=["標準化", "半減期"], columns="期間", values="R/R")
        .to_markdown(),
        "",
        "## 4. 回転率と損益分岐コスト（全期間）",
        "",
        pivot(sm, "回転率", コストbp=0.0, 期間="全期間")
        .set_index(["標準化", "半減期"])[["回転率", "損益分岐bp", "AR", "MDD"]]
        .to_markdown(),
        "",
        "## 5. λ × K（z化＋半減期5、期間別 R/R）",
        "",
        pivot(grid, "R/R", 半減期=5.0)
        .pivot(index=["K", "lambda"], columns="期間", values="R/R")
        .to_markdown(),
        "",
        "## 6. λ × K（平滑化なし、期間別 R/R）",
        "",
        pivot(grid, "R/R", 半減期="なし")
        .pivot(index=["K", "lambda"], columns="期間", values="R/R")
        .to_markdown(),
        "",
    ]
    (out / "REPORT_EXPERIMENTS.md").write_text("\n".join(lines), encoding="utf-8")
    print()
    print("\n".join(lines))
    _log(f"完了: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
