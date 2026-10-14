"""CPU contract checks for the bounded C4 resident repair lane."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
from pathlib import Path

import torch

from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_resident_repair import (
    C4_RESIDENT_REPAIR_CONTRACT_SCHEMA_V1,
    C4_RESIDENT_REPAIR_GATE_CLAIM_V1,
    C4_RESIDENT_REPAIR_PHASE_ORDER_V1,
    C4_RESIDENT_REPAIR_STATUS_V1,
    C4_RESIDENT_RESET_MANIFEST_SCHEMA_V1,
    c4_newborn_dead_slot_reset_coverage_manifest_v1,
    c4_resident_repair_contract_v1,
    validate_c4_resident_repair_contract_v1,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / (
    "_artifacts/tensor_world_c0_c7_v3/c4/"
    "tensor_world_C4_newborn_dead_slot_reset_coverage_manifest.json"
)


def _resolve(root: object, path: str) -> object:
    value = root
    for component in path.split("."):
        value = getattr(value, component)
    return value


def _runtime_tensor_paths(bundle: object, owner_paths: set[str]) -> set[str]:
    paths: set[str] = set()
    for owner_path in owner_paths:
        owner = _resolve(bundle, owner_path)
        assert is_dataclass(owner)
        for field in fields(owner):
            if field.name == "schema_version":
                continue
            value = getattr(owner, field.name)
            if isinstance(value, torch.Tensor):
                paths.add(f"{owner_path}.{field.name}")
    return paths


def test_c4_resident_repair_contract_is_explicit_and_fail_closed() -> None:
    contract = c4_resident_repair_contract_v1()
    assert contract["schema_version"] == C4_RESIDENT_REPAIR_CONTRACT_SCHEMA_V1
    assert contract["status"] == C4_RESIDENT_REPAIR_STATUS_V1
    assert contract["gate_claim"] is C4_RESIDENT_REPAIR_GATE_CLAIM_V1 is False
    assert contract["historical_decision"] == (
        "TW-C4-RESIDENCY-BLOCKED_PRESERVED"
    )
    assert tuple(contract["resident_transaction_phase_order"]) == (
        C4_RESIDENT_REPAIR_PHASE_ORDER_V1
    )
    assert contract["resident_transaction_status"] == "NOT_BOUND_FAIL_CLOSED"
    assert tuple(contract["invariants"])
    assert set(contract["buffer_contract"]) == {
        "death_candidates",
        "free_slots",
        "birth_plan",
        "event_rows",
        "device_shadows",
    }


def test_c4_reset_manifest_classifies_every_v4_tensor_leaf_on_cpu() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=4, active=2, width=4)
    manifest = c4_newborn_dead_slot_reset_coverage_manifest_v1()
    entries = manifest["fields"]
    paths = {str(entry["path"]) for entry in entries}
    owner_paths = {str(entry["owner"]) for entry in entries}

    assert manifest["schema_version"] == C4_RESIDENT_RESET_MANIFEST_SCHEMA_V1
    assert manifest["status"] == C4_RESIDENT_REPAIR_STATUS_V1
    assert manifest["gate_claim"] is False
    assert int(manifest["field_count"]) == len(entries) == len(paths)
    assert paths == _runtime_tensor_paths(bundle, owner_paths)
    assert all(entry["classification"] != "UNCLASSIFIED" for entry in entries)
    assert all(entry["death_action"] for entry in entries)
    assert all(entry["newborn_action"] for entry in entries)


def test_c4_reset_manifest_covers_identity_occupancy_and_newborn_domains() -> None:
    entries = {
        str(entry["path"]): entry
        for entry in c4_newborn_dead_slot_reset_coverage_manifest_v1()["fields"]
    }
    expected = {
        "base.base.core.agents.agent_id": "agent_slot_state",
        "base.base.core.occupancy.slot_grid": "occupancy_projection",
        "base.base.patch.agent_grid_i32": "occupancy_projection",
        "base.base.core.policy.packed_weights": "policy_slot_state",
        "base.genome.reproduction_threshold": "genome_slot_state",
        "base.adaptation.chemistry": "adaptation_slot_state",
        "base.policy_runtime.action_cache_actions": "policy_runtime_slot_state",
        "population.free_queue": "population_transaction_state",
        "events.typed_sequence": "event_tape",
        "scalar_rng.python_random_words": "scalar_rng_continuation",
    }
    for path, classification in expected.items():
        assert entries[path]["classification"] == classification
        assert entries[path]["death_action"] != "not_applicable"
        assert entries[path]["newborn_action"] != "not_applicable"

    required = [
        entry
        for entry in entries.values()
        if entry["resident_status"] != "OUTSIDE_C4_SCOPE"
    ]
    assert required
    assert all(entry["death_action"] != "preserve" for entry in required)


def test_c4_reset_manifest_artifact_matches_generated_cpu_contract() -> None:
    generated = c4_newborn_dead_slot_reset_coverage_manifest_v1()
    artifact = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == generated["schema_version"]
    assert artifact["status"] == generated["status"]
    assert artifact["gate_claim"] is generated["gate_claim"] is False
    assert artifact["source_of_truth"] == generated["source_of_truth"]
    assert artifact["field_count"] == generated["field_count"]
    assert set(artifact["owner_paths"]) == {
        str(entry["owner"]) for entry in generated["fields"]
    }
    counts: dict[str, int] = {}
    for entry in generated["fields"]:
        key = str(entry["classification"])
        counts[key] = counts.get(key, 0) + 1
    assert artifact["classification_counts"] == counts


def test_c4_repair_validator_reports_total_cpu_inventory() -> None:
    result = validate_c4_resident_repair_contract_v1()
    assert result["schema_version"] == C4_RESIDENT_REPAIR_CONTRACT_SCHEMA_V1
    assert result["status"] == C4_RESIDENT_REPAIR_STATUS_V1
    assert result["field_count"] == 263
    assert result["gate_claim"] is False
