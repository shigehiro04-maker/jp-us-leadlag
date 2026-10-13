#!/usr/bin/env python3
"""2 つの history.json を予想日（asof）ごとにまとめる。

Mac と GitHub Actions（予備）の両方が docs/ を更新するため、push がぶつかると
生成ファイルどうしで git の競合が起きる。行単位の競合解消は意味がないので、
記録単位でまとめ直す。

  python scripts/merge_history.py <手元の history.json> <公開中の history.json>
  → 2 つ目のファイルを、まとめた結果で上書きする

同じ予想日の記録が両方にあるとき:
  1. 寄付き前に作った予想（pre_open）を優先（寄付き後の再計算で置き換えない）
  2. 次に、採点済み（resolved）のもの
  3. それでも同じなら、項目の多いもの（業種別の実績などを持つ）
"""
import json
import sys
from pathlib import Path


def score(r: dict) -> tuple:
    return (bool(r.get("pre_open")), bool(r.get("resolved")), len(json.dumps(r)))


def merge(a: list[dict], b: list[dict]) -> list[dict]:
    by = {}
    for r in list(b) + list(a):            # 同点なら手元（a）を残す
        k = r.get("asof")
        if k is None:
            continue
        if k not in by or score(r) >= score(by[k]):
            by[k] = r
    return [by[k] for k in sorted(by)][-500:]


def load(p: Path) -> list[dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


if __name__ == "__main__":
    local, remote = Path(sys.argv[1]), Path(sys.argv[2])
    out = merge(load(local), load(remote))
    remote.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"history をまとめました: {len(out)} 件")
