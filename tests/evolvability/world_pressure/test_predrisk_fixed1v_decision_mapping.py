from __future__ import annotations

from tools.bin.predrisk_fixed1_validation_report import ARMS, WORLDS, decision_summary


def _cell(world: str, arm: str, seed: int) -> dict[str, object]:
    return {
        "world_family": world,
        "fixed_probe": arm,
        "seed": seed,
        "status": "ok",
        "missing_required_telemetry_fields": "",
        "exposed_rich_net_return_per_1000_available_resource_agent_ticks": 1.0,
        "safe_covered_selected_parent_events_per_1000_eligible_parent_events_after_zone_use": None,
    }


def test_predrisk_fixed1v_unsupported_reciprocal_demand_maps_to_b_not_f() -> None:
    cells = [
        _cell(world, arm, seed)
        for world in WORLDS
        for arm in ARMS
        for seed in (101, 202, 303, 404)
    ]

    summary = decision_summary(cells, expected_cells=96)

    assert summary["complete"] is True
    assert summary["statuses_ok"] is True
    assert summary["missing_required_telemetry_cell_count"] == 0
    assert summary["decision"] == "PREDRISK-FIXED-1V-B"
    assert summary["decision_reason"] == "reciprocal predator-resource fixed-probe demand not supported"
