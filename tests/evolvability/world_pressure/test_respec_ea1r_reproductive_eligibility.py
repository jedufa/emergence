from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_ea1r():
    path = Path(__file__).resolve().parents[3] / "tools" / "bin" / "respec_ea1r_reproductive_eligibility.py"
    spec = importlib.util.spec_from_file_location("respec_ea1r_reproductive_eligibility", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("respec_ea1r_reproductive_eligibility", module)
    spec.loader.exec_module(module)
    return module


ea1r = _load_ea1r()


def _row(seed: int, arm: str, zone: str, primary: float, births: float) -> dict:
    return {
        "seed": seed,
        "fixed_arm": arm,
        "zone": zone,
        "selected_parent_events_per_1000_eligible_parent_events": primary,
        "births_per_1000_eligible_parent_events": births,
        "selected_parent_events_per_1000_agent_ticks": primary / 100.0,
        "births_per_1000_agent_ticks": births / 100.0,
        "selected_parent_events": primary,
        "births": births,
        "eligible_parent_events": 1000.0,
        "agent_ticks_cumulative": 10000.0,
    }


def test_respec_ea1r_same_zone_summary_returns_c_when_b_specialist_loses() -> None:
    rows = []
    for seed in ea1r.SEEDS:
        rows.extend(
            [
                _row(seed, "a_favored", "resource_a_zone", 200.0, 700.0),
                _row(seed, "neutral", "resource_a_zone", 50.0, 500.0),
                _row(seed, "b_favored", "resource_a_zone", 60.0, 450.0),
                _row(seed, "a_favored", "resource_b_zone", 30.0, 400.0),
                _row(seed, "neutral", "resource_b_zone", 220.0, 900.0),
                _row(seed, "b_favored", "resource_b_zone", 190.0, 850.0),
            ]
        )

    deltas = ea1r._local_deltas(rows)
    summary = ea1r._same_zone_summary(deltas)

    assert summary["decision"] == "RESPEC-EA1R-C"
    assert summary["primary_comparisons"]["a_favored_gt_neutral_in_a_zone"]["supporting_seeds"] == 4
    assert summary["primary_comparisons"]["b_favored_gt_neutral_in_b_zone"]["supporting_seeds"] == 0
    assert "not ready for evolving-axis selection" in summary["blocking_reason"]


def test_respec_ea1r_report_includes_plain_language_and_no_claims() -> None:
    payloads = {
        "respec_ea1r_final_decision.json": {
            "decision": "RESPEC-EA1R-C",
            "reason": "B failed",
            "no_claims": ea1r.NO_CLAIMS,
        },
        "respec_ea1r_same_zone_niche_summary.json": {
            "primary_comparisons": {
                "a_favored_gt_neutral_in_a_zone": {"supporting_seeds": 4},
                "neutral_gt_b_favored_in_a_zone": {"supporting_seeds": 2},
                "b_favored_gt_neutral_in_b_zone": {"supporting_seeds": 0},
                "neutral_gt_a_favored_in_b_zone": {"supporting_seeds": 4},
            }
        },
        "respec_ea1r_repeat_parity_summary.json": {"status": "PASS"},
    }

    text = ea1r.report_text(payloads)

    assert "Decision: **RESPEC-EA1R-C**" in text
    assert "In simple terms:" in text
    assert "does not claim evolved specialization" in text
