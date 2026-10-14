"""Source-bound C4 population contract preparation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_contract import (
    C4_GATE_AUTHORIZATION_V1,
    C4_PARITY_BOUNDARY_SURFACES_V1,
    C4_PHASE_ORDER,
    C4_RESIDENCY_BOUNDARY_INVARIANTS_V1,
    C4_SOURCE_AUDIT_SCHEMA_V1,
    C4_STRESS_FIXTURES_V1,
    validate_c4_population_contract,
)
from engine.tensor_world.c4_native import (
    C4_IDENTITY_NATIVE_SCHEMA_VERSION_V1,
    C4_IDENTITY_STAGING_FIELDS_V1,
    C4_IDENTITY_STAGING_SCHEMA_VERSION_V1,
    TensorWorldC4IdentityPreflightConfigV1,
    c4_identity_native_source_manifest_v1,
    execute_c4_identity_preflight_cuda,
    validate_c4_identity_preflight_staging_v1,
)
from engine.tensor_world.c4_transaction import (
    C4_BOUNDARY_SURFACE_PATHS_V1,
    C4PopulationTransactionError,
    C4_POPULATION_TRANSACTION_RECEIPT_SCHEMA_V1,
    C4_TRANSACTION_ROLLED_BACK_STATUS_V1,
    C4_TRANSACTION_ROLLBACK_SCOPE_V1,
    capture_c4_population_boundary_v1,
    compare_c4_population_boundaries_v1,
    execute_c4_population_transaction_oracle_v1,
)
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from engine.tensor_world.c4_fixture import (
    C4_ADVERSARIAL_FIXTURE_NAMES_V1,
    make_c4_adversarial_fixture_v1,
    make_c4_fixture_v1,
)
from engine.tensor_world.c4_population import (
    TensorWorldC4BirthConfigV1,
    TensorWorldC4PopulationConfigV1,
    apply_c4_death_settlement_oracle_v1,
    commit_c4_ordinary_births_oracle_v1,
    plan_c4_ordinary_births_oracle_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]


def test_c4_population_contract_is_current_and_preparation_only() -> None:
    receipt = validate_c4_population_contract(ROOT)
    assert receipt["schema_version"] == C4_SOURCE_AUDIT_SCHEMA_V1
    assert receipt["gate_authorization"] == C4_GATE_AUTHORIZATION_V1
    assert tuple(receipt["phase_order"]) == C4_PHASE_ORDER
    assert receipt["status"] == "PREPARATION_ONLY"


def test_c4_contract_declares_adversarial_fixtures_boundary_surfaces_and_residency() -> None:
    receipt = validate_c4_population_contract(ROOT)
    assert tuple(receipt["stress_fixtures"]) == C4_STRESS_FIXTURES_V1
    assert receipt["parity_boundary_surfaces"] == {
        name: list(fields) for name, fields in C4_PARITY_BOUNDARY_SURFACES_V1.items()
    }
    assert tuple(
        receipt["residency_boundary_invariants"]
    ) == C4_RESIDENCY_BOUNDARY_INVARIANTS_V1


def test_c4_fixture_remaps_world_offset_ids_to_the_world_local_identity_abi() -> None:
    bundle, _ = make_c4_fixture_v1(batch=2, capacity=8, active=3, width=8)
    assert bundle.base.base.core.agents.agent_id.tolist() == [
        [1, 2, 3, -1, -1, -1, -1, -1],
        [1, 2, 3, -1, -1, -1, -1, -1],
    ]
    assert bundle.population.agent_id_to_slot.tolist() == [
        [-1, 0, 1, 2, -1, -1, -1, -1],
        [-1, 0, 1, 2, -1, -1, -1, -1],
    ]
    bundle.validate()


def test_c4_adversarial_fixture_manifest_materializes_every_required_case() -> None:
    assert C4_ADVERSARIAL_FIXTURE_NAMES_V1 == C4_STRESS_FIXTURES_V1
    for name in C4_ADVERSARIAL_FIXTURE_NAMES_V1:
        bundle, metadata = make_c4_adversarial_fixture_v1(name)
        assert metadata["name"] == name
        assert metadata["route"] in {
            "cpu_oracle_preparation_only",
            "cpu_adjacent_mate_preparation_only",
        }
        bundle.validate()

    mixed, mixed_metadata = make_c4_adversarial_fixture_v1(
        "mixed_world_transaction_counts"
    )
    assert mixed.batch == 2
    assert mixed_metadata["world_local_transaction_counts"] == [1, 2]
    assert mixed.base.base.core.agents.agent_id[:, :3].tolist() == [[1, 2, 3], [1, 2, 3]]


def test_c4_boundary_comparator_covers_declared_identity_and_residency_surfaces() -> None:
    bundle, _ = make_c4_fixture_v1(batch=2, capacity=8, active=3, width=8)
    expected = capture_c4_population_boundary_v1(bundle, phase="pre_transaction")
    actual = capture_c4_population_boundary_v1(bundle, phase="pre_transaction")
    assert tuple(expected.surfaces) == tuple(
        path for path in C4_BOUNDARY_SURFACE_PATHS_V1 if path in expected.surfaces
    )
    comparison = compare_c4_population_boundaries_v1(expected, actual)
    assert comparison["pass"] is True
    assert comparison["mismatches"] == []

    bundle.population.slot_tokens[0, 0] += 1
    drift = capture_c4_population_boundary_v1(bundle, phase="post_commit")
    comparison = compare_c4_population_boundaries_v1(expected, drift)
    assert comparison["pass"] is False
    assert any(
        row["surface"] == "population.slot_tokens"
        and row["reason"] == "value_mismatch"
        for row in comparison["mismatches"]
    )


def test_c4_mixed_world_fixture_runs_world_local_death_transaction_counts() -> None:
    bundle, _ = make_c4_adversarial_fixture_v1(
        "mixed_world_transaction_counts"
    )
    receipt = execute_c4_population_transaction_oracle_v1(bundle)
    assert receipt.status == "PREPARATION_PASS"
    assert receipt.counts_after["death_count"] == (1, 2)
    assert receipt.counts_after["alive_count"] == (2, 1)
    assert receipt.counts_after["event_active_count"] == (1, 2)
    assert receipt.phase_status_map["typed_lifecycle_event_commit"] == "PASS"
    assert receipt.phase_counts["typed_lifecycle_event_commit"][
        "event_active_count"
    ] == (1, 2)
    assert bundle.base.base.core.agents.agent_id.tolist() == [
        [1, -1, 3, -1, -1, -1, -1, -1],
        [-1, -1, 3, -1, -1, -1, -1, -1],
    ]


def test_c4_population_artifact_binds_source_and_identity_invariants() -> None:
    artifact = ROOT / "_artifacts/tensor_world_c0_c7_v3/c4/tensor_world_C4_population_contract.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["schema_version"] == C4_SOURCE_AUDIT_SCHEMA_V1
    assert payload["gate_authorization"] == C4_GATE_AUTHORIZATION_V1
    assert payload["status"] == "PREPARATION_ONLY"
    assert payload["field_group_counts"]["C4_TRANSACTIONAL"] == 40
    assert len(payload["phase_order"]) == len(C4_PHASE_ORDER)
    assert "slot_reuse_does_not_reuse_biological_agent_id" in payload["identity_invariants"]
    assert tuple(payload["stress_fixtures"]) == C4_STRESS_FIXTURES_V1
    assert payload["parity_boundary_surfaces"]["identity_lineage"]


def test_c4_adversarial_fixture_artifact_is_preparation_only() -> None:
    artifact = ROOT / (
        "_artifacts/tensor_world_c0_c7_v3/c4/"
        "tensor_world_C4_adversarial_fixture_manifest.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["status"] == "PREPARATION_ONLY"
    assert payload["decision"] == "NO_GATE_DECISION"
    assert tuple(payload["fixture_names"]) == C4_STRESS_FIXTURES_V1
    assert payload["identity_boundary"]["production_policy_claimed"] is False
    assert payload["transaction_boundary"]["host_observation"] is True
    assert payload["gate_claim"] is False


def test_c4_transaction_coordinator_artifact_binds_receipt_and_rollback_schema() -> None:
    artifact = ROOT / (
        "_artifacts/tensor_world_c0_c7_v3/c4/"
        "tensor_world_C4_transaction_coordinator.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["schema_version"] == C4_POPULATION_TRANSACTION_RECEIPT_SCHEMA_V1
    assert payload["status"] == "PREPARATION_ONLY"
    assert payload["decision"] == "NO_GATE_DECISION"
    assert payload["gate_authorization"] == C4_GATE_AUTHORIZATION_V1
    assert tuple(payload["phase_order"]) == C4_PHASE_ORDER
    assert payload["failure_boundary"]["rollback_scope"] == (
        C4_TRANSACTION_ROLLBACK_SCOPE_V1
    )
    assert payload["failure_boundary"]["restores_optional_c5_replay_tensor_leaves"] is True


def test_c4_death_settlement_orders_by_biological_identity_and_releases_slots() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=4, width=8)
    bundle.base.base.core.agents.energy[0, 1] = 0.0
    bundle.base.base.core.agents.health[0, 2] = 0.0
    bundle.base.lifecycle.death_candidate[0, 1] = True
    bundle.base.lifecycle.death_candidate[0, 2] = True

    apply_c4_death_settlement_oracle_v1(bundle)

    assert bundle.population.death_count.tolist() == [2]
    assert bundle.population.death_agent_ids[0, :2].tolist() == [2, 3]
    assert bundle.population.death_slots[0, :2].tolist() == [1, 2]
    assert bundle.population.death_reason_codes[0, :2].tolist() == [1, 2]
    assert bundle.population.active_count.tolist() == [2]
    assert bundle.population.active_agent_ids[0, :2].tolist() == [1, 4]
    assert bundle.population.active_agent_slots[0, :2].tolist() == [0, 3]
    assert bundle.population.free_queue[0, :6].tolist() == [4, 5, 6, 7, 1, 2]
    assert bundle.population.free_queue_count.tolist() == [6]
    assert bundle.base.base.core.agents.alive[0, 1].item() is False
    assert bundle.base.base.core.agents.agent_id[0, 1].item() == -1
    assert bundle.population.agent_id_to_slot[0, :5].tolist() == [-1, 0, -1, -1, 3]
    bundle.validate()


def test_c4_births_fail_closed_until_the_population_plan_is_bound() -> None:
    try:
        TensorWorldC4PopulationConfigV1(enable_births=True)
    except TensorWorldContractError as exc:
        assert "birth planning" in str(exc)
    else:
        raise AssertionError("C4 birth planning must remain fail-closed in this slice")


def test_c4_ordinary_birth_plan_is_explicit_rng_tape_and_commit_boundary() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    config = TensorWorldC4BirthConfigV1(max_births=3)
    try:
        plan_c4_ordinary_births_oracle_v1(bundle, config)
    except TensorWorldContractError as exc:
        assert "RNG tape" in str(exc)
    else:
        raise AssertionError("birth planning must not invent a C5 RNG stream")

    before_ids = bundle.base.base.core.agents.agent_id.clone()
    before_energy = bundle.base.base.core.agents.energy.clone()
    tape = torch.zeros((1, 8, 3), dtype=torch.float64)
    plan_c4_ordinary_births_oracle_v1(bundle, config, spawn_uniforms=tape)
    assert bundle.population.planned_count.tolist() == [2]
    assert torch.equal(bundle.base.base.core.agents.agent_id, before_ids)
    assert torch.equal(bundle.base.base.core.agents.energy, before_energy)

    commit_c4_ordinary_births_oracle_v1(bundle, config)
    assert bundle.population.active_count.tolist() == [5]
    assert bundle.base.base.core.agents.agent_id[0, :5].tolist() == [1, 2, 3, 4, 5]
    assert bundle.population.lineage_parent_agent_id[0, 3:5].tolist() == [1, 3]
    assert bundle.population.patch_habitat_lineage_id[0, 3:5].tolist() == [1, 3]
    assert bundle.population.lineage_birth_tick[0, 3:5].tolist() == [0, 0]
    assert bundle.base.base.core.agents.health[0, 3:5].tolist() == [12.0, 12.0]
    assert bundle.base.base.core.occupancy.agent_id_grid[0, 0, 1].item() == 4
    assert bundle.base.base.core.occupancy.agent_id_grid[0, 0, 3].item() == 5
    assert bundle.population.free_queue[0, :3].tolist() == [5, 6, 7]
    bundle.validate()


def test_c4_identity_native_preflight_manifest_is_source_bound_and_nonmutating() -> None:
    manifest = c4_identity_native_source_manifest_v1()
    assert manifest["schema_version"] == C4_IDENTITY_NATIVE_SCHEMA_VERSION_V1
    assert manifest["authoritative_semantics"] == (
        "preflight_ordinary_birth_identity_external_scratch_v2"
    )
    assert manifest["mutates_authoritative_state"] is False
    assert manifest["gate_claim"] is False
    assert set(manifest["sources"]) == {
        "binding",
        "kernel",
        "flattened_identity_header",
        "identity_header",
    }
    artifact = ROOT / (
        "_artifacts/tensor_world_c0_c7_v3/c4/"
        "tensor_world_C4_identity_preflight_contract.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["schema_version"] == C4_IDENTITY_NATIVE_SCHEMA_VERSION_V1
    assert payload["status"] == "PREPARATION_ONLY"
    assert payload["gate_claim"] is False
    assert payload["source_references"] == manifest["sources"]


def test_c4_identity_native_preflight_fails_closed_before_cpu_fallback() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=4, active=2, width=4)
    parents = torch.full((1, 4), -1, dtype=torch.int64)
    spawns = torch.full((1, 4, 2), -1, dtype=torch.int32)
    planned = torch.zeros((1,), dtype=torch.int64)
    before_ids = bundle.base.base.core.agents.agent_id.clone()
    with pytest.raises(TensorWorldContractError, match="CUDA state bundle"):
        execute_c4_identity_preflight_cuda(bundle, parents, spawns, planned)
    assert torch.equal(bundle.base.base.core.agents.agent_id, before_ids)


def test_c4_identity_native_preflight_config_rejects_invalid_source_ranges() -> None:
    with pytest.raises(TensorWorldContractError, match="parent_energy_share"):
        TensorWorldC4IdentityPreflightConfigV1(parent_energy_share=1.1)
    with pytest.raises(TensorWorldContractError, match="cooldown_steps"):
        TensorWorldC4IdentityPreflightConfigV1(cooldown_steps=-1)


def test_c4_identity_staging_contract_is_fixed_capacity_and_fail_closed() -> None:
    manifest = c4_identity_native_source_manifest_v1()
    assert manifest["staging_schema_version"] == C4_IDENTITY_STAGING_SCHEMA_VERSION_V1
    assert tuple(manifest["staging_fields"]) == C4_IDENTITY_STAGING_FIELDS_V1
    assert "nonzero_status" in manifest["status_policy"]

    with pytest.raises(TensorWorldContractError, match="CPU staging/fallback"):
        validate_c4_identity_preflight_staging_v1(
            {}, batch=1, capacity=4, device=torch.device("cpu")
        )


def test_c4_identity_staging_rejects_cpu_payload_even_with_cuda_device_claim() -> None:
    outputs = {
        name: torch.empty(
            (1, 4) if name.endswith("slots") or name.endswith("ids") or name.endswith("energy")
            else (1,),
            dtype=torch.float64 if name.endswith("energy") else torch.int64,
        )
        for name in C4_IDENTITY_STAGING_FIELDS_V1
    }
    with pytest.raises(TensorWorldContractError, match="must be a CUDA tensor"):
        validate_c4_identity_preflight_staging_v1(
            outputs, batch=1, capacity=4, device=torch.device("cuda")
        )


def test_c4_transaction_coordinator_reports_order_and_counts() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    bundle.base.base.core.agents.energy[0, 1] = 0.0
    bundle.base.lifecycle.death_candidate[0, 1] = True
    receipt = execute_c4_population_transaction_oracle_v1(
        bundle,
        birth_config=TensorWorldC4BirthConfigV1(max_births=3),
        spawn_uniforms=torch.zeros((1, 8, 3), dtype=torch.float64),
    )

    assert receipt.schema_version == C4_POPULATION_TRANSACTION_RECEIPT_SCHEMA_V1
    assert receipt.status == "PREPARATION_PASS"
    assert receipt.gate_claim is False
    assert receipt.phase_status_map[receipt.phase_order[0]] == "PASS"
    assert receipt.phase_status_map[receipt.phase_order[1]] == "PASS"
    assert receipt.phase_status_map[receipt.phase_order[2]] == "PASS"
    assert receipt.phase_status_map[receipt.phase_order[3]] == "PASS"
    assert receipt.phase_status_map[receipt.phase_order[4]] == "PASS"
    assert receipt.counts_before["alive_count"] == (3,)
    assert receipt.counts_after["death_count"] == (1,)
    assert receipt.counts_after["planned_birth_count"] == (2,)
    assert receipt.counts_after["birth_count"] == (2,)
    assert receipt.counts_after["active_count"] == (4,)
    encoded = receipt.as_dict()
    assert encoded["phase_order"] == list(receipt.phase_order)
    assert encoded["phase_status"][receipt.phase_order[4]] == "PASS"


def test_c4_transaction_coordinator_rolls_back_event_capacity_failure() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    config = TensorWorldC4BirthConfigV1(max_births=3)
    assert bundle.events is not None
    replay = make_c5_pcg64_context_v1(bundle, streams=1, seed=19)
    bundle.events.active_count.fill_(bundle.events.capacity)
    replay.event_sequence.fill_(bundle.events.capacity)
    before = {
        "ids": bundle.base.base.core.agents.agent_id.clone(),
        "energy": bundle.base.base.core.agents.energy.clone(),
        "planned": bundle.population.planned_count.clone(),
        "events": bundle.events.active_count.clone(),
        "rng_state": replay.rng_state_words.clone(),
        "rng_cursors": replay.rng_cursors.clone(),
        "rng_semantic": replay.semantic_draw_counters.clone(),
    }

    with pytest.raises(C4PopulationTransactionError, match="event tape capacity") as error:
        execute_c4_population_transaction_oracle_v1(
            bundle,
            birth_config=config,
            rng_context=replay,
        )

    receipt = error.value.receipt
    assert receipt.status == C4_TRANSACTION_ROLLED_BACK_STATUS_V1
    assert receipt.rollback_scope == C4_TRANSACTION_ROLLBACK_SCOPE_V1
    assert receipt.rollback_applied is True
    assert receipt.failure_phase == "birth_plan_and_reserved_slot_commit"
    assert receipt.phase_status_map[receipt.failure_phase] == "FAILED"
    assert torch.equal(bundle.base.base.core.agents.agent_id, before["ids"])
    assert torch.equal(bundle.base.base.core.agents.energy, before["energy"])
    assert torch.equal(bundle.population.planned_count, before["planned"])
    # The pre-existing full tape is restored as part of the same transaction.
    assert torch.equal(bundle.events.active_count, before["events"])
    assert torch.equal(replay.rng_state_words, before["rng_state"])
    assert torch.equal(replay.rng_cursors, before["rng_cursors"])
    assert torch.equal(replay.semantic_draw_counters, before["rng_semantic"])
    assert receipt.counts_before == receipt.counts_after


def test_c4_transaction_coordinator_rejects_mixed_birth_routes() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=4, active=2, width=4)
    from engine.tensor_world.c4_mating import TensorWorldC4MateBirthConfigV1

    with pytest.raises(TensorWorldContractError, match="combine ordinary and mate"):
        execute_c4_population_transaction_oracle_v1(
            bundle,
            birth_config=TensorWorldC4BirthConfigV1(max_births=1),
            mate_config=TensorWorldC4MateBirthConfigV1(max_births=1),
        )


def test_c4_transaction_coordinator_executes_mate_route_with_c5_context() -> None:
    bundle, _ = make_c4_adversarial_fixture_v1("recombination_path")
    from engine.tensor_world.c4_mating import TensorWorldC4MateBirthConfigV1

    replay = make_c5_pcg64_context_v1(bundle, streams=2, seed=19)
    receipt = execute_c4_population_transaction_oracle_v1(
        bundle,
        mate_config=TensorWorldC4MateBirthConfigV1(max_births=1),
        rng_context=replay,
    )

    assert receipt.status == "PREPARATION_PASS"
    assert receipt.phase_status_map["parent_and_mate_selection_with_declared_rng_boundary"] == "PASS"
    assert receipt.counts_after["planned_birth_count"] == (1,)
    assert receipt.counts_after["birth_count"] == (1,)
    assert receipt.counts_after["active_count"] == (4,)
