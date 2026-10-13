"""scripts/merge_history.py の検証（Mac と予備実行の history をまとめる）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from merge_history import merge  # noqa: E402


def test_pre_open_record_wins_over_later_recompute():
    local = [{"asof": "2026-10-12", "pre_open": True, "long": ["A"], "resolved": False}]
    remote = [{"asof": "2026-10-12", "pre_open": False, "long": ["B"], "resolved": True}]
    out = merge(local, remote)
    assert out[0]["long"] == ["A"]


def test_resolved_record_wins_when_both_pre_open():
    local = [{"asof": "2026-10-12", "pre_open": True, "long": ["A"]}]
    remote = [{"asof": "2026-10-12", "pre_open": True, "long": ["A"], "resolved": True, "ls_return": 0.01}]
    assert merge(local, remote)[0].get("resolved")


def test_union_keeps_both_sides_and_sorts():
    out = merge([{"asof": "2026-10-13"}], [{"asof": "2026-10-09"}, {"asof": "2026-10-12"}])
    assert [r["asof"] for r in out] == ["2026-10-09", "2026-10-12", "2026-10-13"]
