"""価格データの取得と日米の日付整合.

時間軸の考え方
--------------
  暦日 t の東京市場は 15:00 JST (06:00 UTC) に引け、同じ暦日 t の米国市場は
  16:00 ET (21:00 UTC) に引ける。つまり「米国の t の終値」は「東京の t の終値」
  より後に確定し、東京では翌営業日 t+1 の寄付きから織り込まれる。
  したがって
      シグナル: 米国 t の close-to-close リターン
      執行    : 日本 t+1 の open-to-close リターン
  となる (論文 2.1 節)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import JP_TICKERS, US_TICKERS

DEFAULT_CACHE = Path(os.environ.get("LEADLAG_CACHE", "./data"))


def sanitize_returns(
    df: pd.DataFrame, max_abs: float = 0.5, label: str = ""
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """異常なリターンを NaN にし、取り除いた点の一覧を返す。

    ETF の日次リターンが ±50% を超えることは実質的にあり得ず、無料データでは
    株式分割・分配の調整漏れや誤った気配値がそのまま入ってくることがある。
    1 点でも残ると、その後 60 営業日ぶんの標準偏差と相関が壊れ、さらに
    その日にポジションを持っていれば損益そのものが架空の値になる。

    銘柄まるごとではなく該当日だけを NaN にする。推定ウィンドウが NaN を
    含む銘柄はエンジン側が自動的に除外するので、異常値の周辺だけ
    その銘柄がユニバースから外れる（保守的で望ましい挙動）。
    """
    bad = df.abs() > max_abs
    rows = []
    if bad.to_numpy().any():
        idx = np.where(bad.to_numpy())
        for i, j in zip(*idx):
            rows.append(
                {
                    "kind": label,
                    "date": df.index[i],
                    "ticker": df.columns[j],
                    "value": float(df.iat[i, j]),
                }
            )
    return df.mask(bad), pd.DataFrame(rows)


@dataclass
class DataBundle:
    """日米共通営業日にそろえたリターン行列。"""

    us_cc: pd.DataFrame   # 米国 close-to-close リターン (index=日付, col=US_TICKERS)
    jp_cc: pd.DataFrame   # 日本 close-to-close リターン
    jp_oc: pd.DataFrame   # 日本 open-to-close リターン (執行対象)
    us_close: pd.DataFrame
    jp_close: pd.DataFrame
    jp_open: pd.DataFrame
    # 共通営業日に絞る前の、各市場で実際に価格が取れていた最終日。
    # 提供元の更新遅れ（片方だけ当日ぶんが無い状態）を見分けるために持つ。
    us_last_raw: pd.Timestamp | None = None
    jp_last_raw: pd.Timestamp | None = None
    # 共通営業日より後の米国リターン。米国は日本より後に引けるので、提供元が
    # 日本の当日ぶんをまだ配信していない朝でも、米国の最新終値だけは先に届く。
    # その 1 日を捨てると予測が 1 営業日遅れるため、別に持っておく。
    us_cc_ahead: pd.DataFrame | None = None
    # 異常値として除去したリターンの一覧 (kind, date, ticker, value)
    quality_report: pd.DataFrame | None = None

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.us_cc.index

    def summary(self) -> pd.DataFrame:
        """論文 表1 に相当する基本統計量 (年率)。"""
        rows = []
        for name, df in (("US", self.us_cc), ("JP", self.jp_cc)):
            for t in df.columns:
                s = df[t].dropna()
                if s.empty:
                    continue
                rows.append(
                    {
                        "Ticker": t,
                        "Market": name,
                        "Ret(%)": 252 * s.mean() * 100,
                        "Vol(%)": np.sqrt(252) * s.std(ddof=1) * 100,
                        "Ret/Vol": (252 * s.mean()) / (np.sqrt(252) * s.std(ddof=1)),
                        "Skew": s.skew(),
                        "Kurtosis": s.kurtosis(),
                        "N": int(s.size),
                    }
                )
        return pd.DataFrame(rows).set_index("Ticker").round(2)


# ---------------------------------------------------------------------------
# ダウンロード
# ---------------------------------------------------------------------------
def _download(
    tickers: list[str], start: str, end: str | None, retries: int = 3
) -> pd.DataFrame:
    import time

    import yfinance as yf

    raw = pd.DataFrame()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,   # 分割・分配金調整済み (Open/Close とも同じ係数)
                progress=False,
                group_by="ticker",
                threads=True,
            )
            if not raw.empty:
                break
        except Exception as exc:    # noqa: BLE001  ネットワーク起因は再試行する
            last_err = exc
        if attempt < retries - 1:
            time.sleep(5 * (attempt + 1))

    if raw.empty:
        raise RuntimeError(
            "価格データを取得できませんでした。ネットワーク接続を確認してください。"
            + (f" (最後のエラー: {last_err})" if last_err else "")
        )

    frames = {}
    for t in tickers:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        frames[t] = sub[["Open", "Close"]]
    panel = pd.concat(frames, axis=1)
    panel.index = pd.to_datetime(panel.index).tz_localize(None).normalize()
    return panel


def load_prices(
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE,
    refresh: bool = False,
    us_tickers: list[str] | None = None,
    jp_tickers: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """米国パネルと日本パネル (Open/Close) を返す。キャッシュ付き。"""
    us_tickers = list(us_tickers or US_TICKERS)
    jp_tickers = list(jp_tickers or JP_TICKERS)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{start}_{end or 'latest'}"
    us_path = cache_dir / f"us_{tag}.pkl"
    jp_path = cache_dir / f"jp_{tag}.pkl"

    if not refresh and us_path.exists() and jp_path.exists():
        return pd.read_pickle(us_path), pd.read_pickle(jp_path)

    us = _download(us_tickers, start, end)
    jp = _download(jp_tickers, start, end)
    us.to_pickle(us_path)
    jp.to_pickle(jp_path)
    return us, jp


# ---------------------------------------------------------------------------
# リターン構成
# ---------------------------------------------------------------------------
def build_bundle(
    us_panel: pd.DataFrame,
    jp_panel: pd.DataFrame,
    max_abs_return: float = 0.5,
) -> DataBundle:
    """Open/Close パネルから日米共通営業日のリターン行列を作る。"""
    us_tk = sorted({c[0] for c in us_panel.columns})
    jp_tk = sorted({c[0] for c in jp_panel.columns})

    us_close = us_panel.loc[:, [(t, "Close") for t in us_tk]]
    us_close.columns = us_tk
    jp_close = jp_panel.loc[:, [(t, "Close") for t in jp_tk]]
    jp_close.columns = jp_tk
    jp_open = jp_panel.loc[:, [(t, "Open") for t in jp_tk]]
    jp_open.columns = jp_tk

    # yfinance は取引がまだ確定していない日や祝日について、全銘柄 NaN の
    # 空行を返してくることがある。この行を残すと「最新の営業日」がその空行に
    # なり、当日の米国ショックが取れなくなるので、市場ごとに取り除く。
    us_close = us_close.dropna(how="all")
    jp_close = jp_close.dropna(how="all")
    jp_open = jp_open.reindex(jp_close.index)

    # close-to-close は各市場の連続する自国営業日で計算してから共通日に絞る。
    # (先に共通日で間引くと、休場を跨いだリターンが欠落してしまう)
    us_cc_full = us_close.pct_change()
    jp_cc_full = jp_close.pct_change()
    jp_oc_full = jp_close / jp_open - 1.0

    # 無料データに混じる異常値を取り除く（詳細は sanitize_returns を参照）
    us_cc_full, q1 = sanitize_returns(us_cc_full, max_abs_return, "us_cc")
    jp_cc_full, q2 = sanitize_returns(jp_cc_full, max_abs_return, "jp_cc")
    jp_oc_full, q3 = sanitize_returns(jp_oc_full, max_abs_return, "jp_oc")
    quality = pd.concat([q1, q2, q3], ignore_index=True)

    common = us_close.index.intersection(jp_close.index)
    common = common.sort_values()

    return DataBundle(
        us_cc=us_cc_full.loc[common],
        jp_cc=jp_cc_full.loc[common],
        jp_oc=jp_oc_full.loc[common],
        us_close=us_close.loc[common],
        jp_close=jp_close.loc[common],
        jp_open=jp_open.loc[common],
        us_last_raw=us_close.index[-1] if len(us_close) else None,
        jp_last_raw=jp_close.index[-1] if len(jp_close) else None,
        us_cc_ahead=(us_cc_full.loc[us_cc_full.index > common[-1]]
                     if len(common) else us_cc_full),
        quality_report=quality,
    )


def load_bundle(
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE,
    refresh: bool = False,
) -> DataBundle:
    us, jp = load_prices(start=start, end=end, cache_dir=cache_dir, refresh=refresh)
    return build_bundle(us, jp)


def bundle_from_csv(us_csv: str, jp_csv: str) -> DataBundle:
    """自前の CSV から読み込む。

    期待する形式 (long format):
        date,ticker,open,close
    """
    def _panel(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["date"])
        df.columns = [c.lower() for c in df.columns]
        piv = df.pivot(index="date", columns="ticker", values=["open", "close"])
        piv = piv.swaplevel(axis=1).sort_index(axis=1)
        piv.columns = pd.MultiIndex.from_tuples(
            [(t, k.capitalize()) for t, k in piv.columns]
        )
        piv.index = pd.to_datetime(piv.index).normalize()
        return piv

    return build_bundle(_panel(us_csv), _panel(jp_csv))
