from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "tools/bin/branch4_niche_strategy_audit.py"
SPEC = importlib.util.spec_from_file_location("branch4_niche_strategy_audit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
branch4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(branch4)


ZONE_A = "food_high|shelter_low|waste_low|density_low|move_cost_base|recovery_low"
ZONE_B = "food_low|shelter_high|waste_high|density_high|move_cost_base|recovery_high"


def _niche(agent_id: int, zone: str = ZONE_A, *, ticks: int = 24, births: int = 0) -> dict:
    return {
        "schema_version": "branch_niche_observability_v1",
        "run_id": "branch4_niche_strategy__context_patch_observed_clean__init_legacy__seed000101",
        "agent_id": agent_id,
        "founder_ancestry_id": "L1",
        "biological_lineage_id": f"L{agent_id}",
        "parent_ids": [] if agent_id == 1 else [1],
        "birth_generation": 0 if agent_id == 1 else 1,
        "final_generation": 3,
        "total_observed_ticks": ticks,
        "dominant_zone_class": zone,
        "zone_exposure_counts": {zone: ticks},
        "movement_actions_by_zone": {zone: 8},
        "gather_actions_by_zone": {zone: 2},
        "successful_births_by_zone": {zone: births} if births else {},
        "food_collected_by_zone": {zone: 1.5},
        "damage_by_zone": {zone: 0.0},
        "transitions_between_zone_classes": {},
        "shelter_cover_exposure_ticks": ticks if "shelter_high" in zone else 0,
        "movement_action_rate": 8 / max(1, ticks),
        "gather_action_rate": 2 / max(1, ticks),
        "transition_rate": 0.0,
        "successful_birth_rate": births / max(1, ticks),
        "missing_data_coverage": [],
    }


def _lifecycle(agent_id: int, *, raw_strategy: str = "raw", parent: bool = False, lifetime: int = 100) -> dict:
    return {
        "schema_version": "branch_observability_v1",
        "run_id": "branch4_niche_strategy__context_patch_observed_clean__init_legacy__seed000101",
        "agent_id": agent_id,
        "founder_ancestry_id": "L1",
        "biological_lineage_id": f"L{agent_id}",
        "parent_ids": [] if agent_id == 1 else [1],
        "birth_generation": 0 if agent_id == 1 else 1,
        "final_generation": 3,
        "lifetime_ticks": lifetime,
        "final_genome_signature": f"genome-{agent_id}",
        "memory_dim": 64,
        "sensor_radius": 9,
        "successful_births": 1 if parent else 0,
        "successful_children_count": 1 if parent else 0,
        "strategy_id": raw_strategy,
        "alive_at_run_end": True,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_run(root: Path) -> Path:
    run_dir = root / "0001_branch4_niche_strategy__context_patch_observed_clean__init_legacy__seed000101"
    bo = run_dir / "branch_observability"
    _write_jsonl(
        bo / "branch_observability_agent_lifecycle.jsonl",
        [_lifecycle(1, raw_strategy="raw_parent", parent=True), _lifecycle(2, raw_strategy="raw_child_a"), _lifecycle(3, raw_strategy="raw_child_b")],
    )
    _write_jsonl(
        bo / "branch_observability_niche_lifecycle.jsonl",
        [_niche(1, ZONE_A, births=2), _niche(2, ZONE_A), _niche(3, ZONE_B)],
    )
    _write_jsonl(
        bo / "branch_observability_birth_edges.jsonl",
        [
            {
                "schema_version": "branch_observability_v1",
                "event_type": "birth_success",
                "run_id": "branch4_niche_strategy__context_patch_observed_clean__init_legacy__seed000101",
                "birth_event_id": "birth-1",
                "generation": 1,
                "tick": 0,
                "parent_agent_id": 1,
                "child_agent_id": 2,
                "parent_biological_lineage_id": "L1",
                "child_biological_lineage_id": "L2",
                "child_founder_ancestry_id": "L1",
            },
            {
                "schema_version": "branch_observability_v1",
                "event_type": "birth_success",
                "run_id": "branch4_niche_strategy__context_patch_observed_clean__init_legacy__seed000101",
                "birth_event_id": "birth-2",
                "generation": 1,
                "tick": 1,
                "parent_agent_id": 1,
                "child_agent_id": 3,
                "parent_biological_lineage_id": "L1",
                "child_biological_lineage_id": "L3",
                "child_founder_ancestry_id": "L1",
            },
        ],
    )
    (bo / "branch_observability_run_summary.json").write_text(
        json.dumps({"run_id": "branch4_niche_strategy__context_patch_observed_clean__init_legacy__seed000101"}) + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_same_normalized_behavior_ignores_lifetime_genome_and_capacity() -> None:
    a = _niche(1, ZONE_A) | {"lifetime_ticks": 12, "final_genome_signature": "a", "memory_dim": 4}
    b = _niche(2, ZONE_A) | {"lifetime_ticks": 999, "final_genome_signature": "b", "memory_dim": 128}

    assert branch4.niche_signature(a)["strategy_id"] == branch4.niche_signature(b)["strategy_id"]


def test_different_niche_use_separates_strategy() -> None:
    assert branch4.niche_signature(_niche(1, ZONE_A))["strategy_id"] != branch4.niche_signature(_niche(2, ZONE_B))["strategy_id"]


def test_missing_exposure_fails_to_insufficient_data_bucket() -> None:
    sig = branch4.niche_signature(_niche(1, ZONE_A, ticks=0) | {"zone_exposure_counts": {}})

    assert sig["strategy_status"] == "insufficient_data"
    assert "insufficient_data" in sig["signature_payload"]


def test_strategy_signature_is_row_order_independent() -> None:
    row = _niche(1, ZONE_A)
    shuffled = dict(reversed(list(row.items())))

    assert branch4.niche_signature(row)["strategy_id"] == branch4.niche_signature(shuffled)["strategy_id"]


def test_population_increase_without_niche_change_is_not_strategy_gain() -> None:
    rows = [_niche(agent_id, ZONE_A) for agent_id in (1, 2, 3, 4)]
    strategy_ids = {branch4.niche_signature(row)["strategy_id"] for row in rows}

    assert len(strategy_ids) == 1


def test_analyze_run_emits_branch_strategy_tables(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    result = branch4.analyze_runs([run_dir], profile="permissive")

    assert len(result["raw_individual_signatures"]) == 3
    assert len(result["compressed_strategies"]) == 2
    assert result["strategy_coverage"]["branch_strategy_links"] >= 1
    assert result["three_layer_gate"]["classification"] in {
        "reporting_strategy_method_ready",
        "reporting_strategy_not_ready",
    }
