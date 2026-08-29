"""結果の出力: 図表 CSV / 累積リターン図 / コンソール表示."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import summary_table  # noqa: E402

# 図のラベルは英語で統一する (日本語フォントが無い環境での文字化けを避けるため)
plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.3})


def plot_cumulative(returns: dict[str, pd.Series], path: Path, title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = {
        "PCA_SUB": dict(color="#2563eb", lw=1.8),
        "DOUBLE": dict(color="#dc2626", lw=1.4),
        "PCA_PLAIN": dict(color="#d97706", lw=1.1, ls="--"),
        "MOM": dict(color="#16a34a", lw=1.1, ls="--"),
        "EW_BENCH": dict(color="#6b7280", lw=1.0, ls=":"),
    }
    for name, r in returns.items():
        wealth = (1.0 + r.dropna()).cumprod()
        ax.plot(wealth.index, wealth.to_numpy(), label=name,
                **styles.get(name, dict(lw=1.2)))
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative wealth (log scale)")
    ax.set_xlabel("Date")
    ax.set_title(title or "Cumulative returns")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_signal_heat(b_matrix, us_tickers, jp_tickers, path: Path) -> Path:
    """伝播行列 B_t のヒートマップ (米国業種 -> 日本業種)。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(b_matrix, cmap="RdBu_r", vmin=-abs(b_matrix).max(),
                   vmax=abs(b_matrix).max())
    ax.set_xticks(range(len(us_tickers)))
    ax.set_xticklabels(us_tickers, rotation=90, fontsize=7)
    ax.set_yticks(range(len(jp_tickers)))
    ax.set_yticklabels(jp_tickers, fontsize=7)
    ax.set_title("Propagation matrix B = V_J V_U'  (US -> JP)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def write_report(result: dict, params, outdir: Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    strat_returns = {k: v.returns for k, v in result["strategies"].items()}
    table = summary_table(strat_returns, ann=params.ann_factor)
    paths["summary"] = outdir / "summary.csv"
    table.to_csv(paths["summary"])

    all_curves = dict(strat_returns)
    all_curves["EW_BENCH"] = result["benchmark"]
    paths["chart"] = plot_cumulative(
        all_curves, outdir / "cumulative_returns.png",
        title=f"JP-US sector lead-lag  (L={params.window}, K={params.n_factors}, "
              f"lambda={params.lam}, q={params.quantile})",
    )

    daily = pd.DataFrame(strat_returns)
    daily["EW_BENCH"] = result["benchmark"]
    paths["daily"] = outdir / "daily_returns.csv"
    daily.to_csv(paths["daily"])

    timing = result["timing"]
    paths["timing"] = outdir / "signal_level_diag.csv"
    timing.to_csv(paths["timing"])

    direction = result["direction"]
    paths["direction"] = outdir / "market_direction.csv"
    direction.to_csv(paths["direction"])

    return {
        "paths": paths,
        "table": table,
        "timing": timing,
        "direction": direction,
        "direction_naive": result["direction_naive"],
    }


def format_summary(table: pd.DataFrame) -> str:
    return table.to_string()
