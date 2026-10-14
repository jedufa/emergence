from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from evolution.structural_development_v1 import DEVELOPMENTAL_EXPRESSION_BLOCK_KEY
from evolution.structural_sustainable_growth_v1 import module_is_dormant
from perf.cuda_architecture_b_modular_capacity_epoch import (
    MODULAR_CAPACITY_EPOCH_AUTHORIZATION_SCHEMA_VERSION,
    capacity_epoch_authorization_receipt_sha256,
)
from perf.cuda_architecture_b_modular_lifecycle import ModularLifecycleError
from perf.cuda_architecture_b_modular_state_abi import (
    modular_storage_from_projection,
    modular_storage_projection_payload,
    pack_modular_storage,
)
from perf.eiel_phase_f_boundary_actions_v2 import (
    BOUNDARY_ACTION_SCHEMA_SHA256,
    PHASE_E_RECEIPT_BINDING_SCHEMA_VERSION,
    PhaseFBoundaryActionError,
    apply_record_boundary_action,
    assert_capacity_migration_rejected,
    build_checkpoint_schedule_descriptor,
    capture_modular_checkpoint,
    execute_bound_capacity_migration,
    prepare_prune_reuse_boundary,
    require_full_runtime_resume_prerequisites,
    restore_fresh_modular_owner,
)
from perf.eiel_phase_f_forced_combination_fixture_v2 import (
    materialize_phase_f_case_v2,
)
from simulation.capacity_epoch import SAFE_BOUNDARY_KIND


def _case(family: str, profile: str = "b12", role: str = "r0") -> dict:
    return materialize_phase_f_case_v2(f"pf2_{family}_{profile}_{role}")


def _packed(family: str, profile: str = "b12", role: str = "r0"):
    fixture = _case(family, profile, role)
    return fixture, pack_modular_storage(
        [agent["record"] for agent in fixture["agents"]],
        agent_capacity=16,
        profile_id=fixture["profile_id"],
    )


def _missing_identity_admissions(source, replacement) -> dict[str, int]:
    start = len(source.identity_table)
    missing = sorted(set(replacement.identity_table) - set(source.identity_table))
    return {identity: start + index for index, identity in enumerate(missing)}


def _fresh_replacement_packed(profile: str = "b12"):
    fixture = _case("f12", profile)
    record = deepcopy(
        next(
            agent["record"]
            for agent in fixture["agents"]
            if agent["record"]["genome"]["modules"]
        )
    )
    module = record["genome"]["modules"][0]
    module["module_id"] = "phase-f-reuse-m000"
    return pack_modular_storage(
        [record],
        agent_capacity=16,
        profile_id=fixture["profile_id"],
    )


def _resident_projection(source) -> dict:
    projection = modular_storage_projection_payload(source)
    projection["metadata"] = {
        "world": "phase-f-test",
        "generation": 11,
        **projection["metadata"],
    }
    projection["tensors"] = {
        "alive_mask": source.tensors["agent_occupied"].clone(),
        "slot_to_agent_id": source.tensors["agent_id"].clone(),
        "slot_tokens": source.tensors["agent_slot_token"].clone(),
        **projection["tensors"],
    }
    return projection


def _rng() -> tuple[dict, dict]:
    return {"stream_index": 7}, {"state": torch.arange(8, dtype=torch.uint8)}


def _authorization(source_packed_sha256: str, **overrides) -> dict:
    authorization = {
        "schema_version": MODULAR_CAPACITY_EPOCH_AUTHORIZATION_SCHEMA_VERSION,
        "authorization_receipt_id": "phase-f-unit-test-external-receipt",
        "authorization_receipt_sha256": "",
        "authorized": True,
        "authorized_generation": 11,
        "next_generation": 12,
        "source_profile_id": "b2-12x32-v1",
        "target_profile_id": "b2-16x48-v1",
        "safe_boundary": SAFE_BOUNDARY_KIND,
        "rollback_checkpoint_digest": source_packed_sha256,
        "rollback_checkpoint_verified": True,
    }
    authorization.update(overrides)
    authorization["authorization_receipt_sha256"] = (
        capacity_epoch_authorization_receipt_sha256(authorization)
    )
    return authorization


def _binding(fixture: dict, source, authorization: dict) -> dict:
    return {
        "schema_version": PHASE_E_RECEIPT_BINDING_SCHEMA_VERSION,
        "verification_status": "externally_verified",
        "phase_e_final_receipt_sha256": "e" * 64,
        "phase_e_execution_commit": "c" * 40,
        "matrix_sha256": fixture["matrix_sha256"],
        "case_sha256": fixture["case_sha256"],
        "authorization_receipt_sha256": authorization["authorization_receipt_sha256"],
        "rollback_checkpoint_digest": source.packed_sha256,
        "source_profile_id": "b2-12x32-v1",
        "target_profile_id": "b2-16x48-v1",
    }


def test_member_reset_is_exactly_one_named_member_and_rng_free() -> None:
    fixture = _case("f06")
    source = fixture["agents"][0]["record"]
    modules = source["genome"]["modules"]
    target_id = str(modules[0]["module_instance_id"])
    states_before = deepcopy(source["structural_live_module_state"])
    outputs_before = deepcopy(source["structural_live_module_outputs"])
    torch_rng_before = torch.get_rng_state().clone()

    result = apply_record_boundary_action(
        source,
        profile_id=fixture["profile_id"],
        action="reset_member_state",
        target_module_instance_id=target_id,
    )
    repeated = apply_record_boundary_action(
        source,
        profile_id=fixture["profile_id"],
        action="reset_member_state",
        target_module_instance_id=target_id,
    )

    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert result == repeated
    assert result.record["structural_live_module_state"][target_id] == 0.0
    assert result.record["structural_live_module_outputs"][target_id] == 0.0
    assert {
        key: value
        for key, value in result.record["structural_live_module_state"].items()
        if key != target_id
    } == {key: value for key, value in states_before.items() if key != target_id}
    assert {
        key: value
        for key, value in result.record["structural_live_module_outputs"].items()
        if key != target_id
    } == {key: value for key, value in outputs_before.items() if key != target_id}
    assert result.receipt["schema_sha256"] == BOUNDARY_ACTION_SCHEMA_SHA256
    assert result.receipt["rng_draw_count"] == 0


def test_dependency_block_uses_real_prerequisite_and_dormant_source() -> None:
    fixture = _case("f07")
    source = fixture["agents"][0]["record"]
    prerequisite, target = source["genome"]["modules"]
    prerequisite_id = str(prerequisite["module_instance_id"])
    target_id = str(target["module_instance_id"])

    result = apply_record_boundary_action(
        source,
        profile_id=fixture["profile_id"],
        action="block_dependency",
        target_module_instance_id=target_id,
        prerequisite_module_instance_id=prerequisite_id,
    )
    modules = {
        str(module["module_instance_id"]): module
        for module in result.record["genome"]["modules"]
    }
    development = modules[target_id]["params"][DEVELOPMENTAL_EXPRESSION_BLOCK_KEY]
    assert str(prerequisite["homology_id"]) in development["prerequisite_homology_ids"]
    assert development["latch_dependency_readiness"] is False
    assert module_is_dormant(modules[prerequisite_id]) is True
    assert (
        result.record["structural_development_latch_by_instance_id"][target_id] is False
    )
    assert (
        result.record["structural_development_effective_gain_by_instance_id"][target_id]
        == 0.0
    )
    assert result.receipt["affected_module_instance_ids"] == sorted(
        [prerequisite_id, target_id]
    )


def test_dormancy_action_and_record_actions_fail_closed() -> None:
    fixture = _case("f06")
    source = fixture["agents"][0]["record"]
    target_id = str(source["genome"]["modules"][0]["module_instance_id"])
    result = apply_record_boundary_action(
        source,
        profile_id=fixture["profile_id"],
        action="enable_member_dormancy",
        target_module_instance_id=target_id,
    )
    target = next(
        module
        for module in result.record["genome"]["modules"]
        if module["module_instance_id"] == target_id
    )
    assert module_is_dormant(target) is True

    with pytest.raises(PhaseFBoundaryActionError, match="unsupported"):
        apply_record_boundary_action(
            source,
            profile_id=fixture["profile_id"],
            action="pretend_runtime_success",
            target_module_instance_id=target_id,
        )
    with pytest.raises(PhaseFBoundaryActionError, match="distinct prerequisite"):
        apply_record_boundary_action(
            source,
            profile_id=fixture["profile_id"],
            action="block_dependency",
            target_module_instance_id=target_id,
            prerequisite_module_instance_id=target_id,
        )


def test_prepared_prune_reuse_calls_public_lifecycle_and_binds_slots() -> None:
    source_fixture, source = _packed("f09")
    replacement = _fresh_replacement_packed()
    pruned_id = str(
        source.records_sidecar[0]["genome"]["modules"][-1]["module_instance_id"]
    )
    replacement_id = str(
        replacement.records_sidecar[0]["genome"]["modules"][0]["module_instance_id"]
    )
    decisions = [
        {
            "agent_row": 0,
            "pruned_module_instance_id": pruned_id,
            "replacement_agent_row": 0,
            "replacement_module_instance_id": replacement_id,
        }
    ]
    admissions = _missing_identity_admissions(source, replacement)
    source_before = {name: value.clone() for name, value in source.tensors.items()}

    result = prepare_prune_reuse_boundary(
        source,
        replacement,
        decisions=decisions,
        admitted_identity_indices=admissions,
    )
    repeated = prepare_prune_reuse_boundary(
        source,
        replacement,
        decisions=decisions,
        admitted_identity_indices=admissions,
    )

    assert result.receipt == repeated.receipt
    assert result.receipt["action"] == "prepared_prune_reuse"
    assert result.receipt["rng_draw_count"] == 0
    assert result.lifecycle.receipt.cpu_validation_mode is True
    assert result.lifecycle.receipt.decision_count == 1
    assert all(
        torch.equal(source.tensors[name], source_before[name])
        for name in source.tensors
    )
    binding = result.receipt["decision_bindings"][0]
    slot = binding["module_slot"]
    replacement_index = admissions[replacement_id]
    assert result.lifecycle.planes["module_instance_id_index"][0, slot] == (
        replacement_index
    )
    assert result.lifecycle.planes["module_state_present"][0, slot].item() is False
    assert result.lifecycle.planes["module_output_present"][0, slot].item() is False

    post_reuse_fixture = _case("f09", role="r4")
    schedule = build_checkpoint_schedule_descriptor(
        case_id=post_reuse_fixture["case_id"],
        case_sha256=post_reuse_fixture["case_sha256"],
        matrix_sha256=source_fixture["matrix_sha256"],
        profile_id=source_fixture["profile_id"],
        schedule_kind="post_reuse_pre_tick",
        prerequisite_receipt=result.receipt,
    )
    with pytest.raises(PhaseFBoundaryActionError, match="lifecycle result"):
        capture_modular_checkpoint(source, schedule)


def test_prune_reuse_preserves_public_dependency_rejection() -> None:
    _source_fixture, source = _packed("f07")
    replacement = _fresh_replacement_packed()
    root_id = str(
        source.records_sidecar[0]["genome"]["modules"][0]["module_instance_id"]
    )
    replacement_id = str(
        replacement.records_sidecar[0]["genome"]["modules"][0]["module_instance_id"]
    )
    with pytest.raises(ModularLifecycleError, match="developmental dependencies"):
        prepare_prune_reuse_boundary(
            source,
            replacement,
            decisions=[
                {
                    "agent_row": 0,
                    "pruned_module_instance_id": root_id,
                    "replacement_agent_row": 0,
                    "replacement_module_instance_id": replacement_id,
                }
            ],
            admitted_identity_indices=_missing_identity_admissions(source, replacement),
        )


def test_migration_requires_external_binding_then_executes_public_migrator() -> None:
    fixture, source = _packed("f11")
    projection = _resident_projection(source)
    authorization = _authorization(source.packed_sha256)
    rng_state, rng_tensors = _rng()

    with pytest.raises(PhaseFBoundaryActionError, match="receipt binding"):
        execute_bound_capacity_migration(
            source_metadata=projection["metadata"],
            source_tensors=projection["tensors"],
            authorization=authorization,
            phase_e_receipt_binding=None,
            rng_state_before=deepcopy(rng_state),
            rng_state_after=deepcopy(rng_state),
            rng_tensors_before={
                name: value.clone() for name, value in rng_tensors.items()
            },
            rng_tensors_after={
                name: value.clone() for name, value in rng_tensors.items()
            },
        )

    binding = _binding(fixture, source, authorization)
    torch_rng_before = torch.get_rng_state().clone()
    result = execute_bound_capacity_migration(
        source_metadata=projection["metadata"],
        source_tensors=projection["tensors"],
        authorization=authorization,
        phase_e_receipt_binding=binding,
        rng_state_before=deepcopy(rng_state),
        rng_state_after=deepcopy(rng_state),
        rng_tensors_before={name: value.clone() for name, value in rng_tensors.items()},
        rng_tensors_after={name: value.clone() for name, value in rng_tensors.items()},
    )
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert result.receipt["action"] == "accepted_capacity_migration"
    assert result.receipt["rng_draw_count"] == 0
    assert result.migration.receipt.migration_rng_draws == 0
    assert result.migration.receipt.source_profile_id == "b2-12x32-v1"
    assert result.migration.receipt.target_profile_id == "b2-16x48-v1"

    target = modular_storage_from_projection(
        result.migration.metadata, result.migration.tensors
    )
    assert target is not None
    post_migration_fixture = _case("f11", profile="b16", role="r4")
    schedule = build_checkpoint_schedule_descriptor(
        case_id=post_migration_fixture["case_id"],
        case_sha256=post_migration_fixture["case_sha256"],
        matrix_sha256=post_migration_fixture["matrix_sha256"],
        profile_id=post_migration_fixture["profile_id"],
        schedule_kind="post_migration_pre_tick",
        prerequisite_receipt=result.receipt,
    )
    checkpoint = capture_modular_checkpoint(target, schedule)
    assert restore_fresh_modular_owner(checkpoint).owner.packed_sha256 == (
        target.packed_sha256
    )


def test_migration_rejection_is_proved_by_public_migrator() -> None:
    _fixture, source = _packed("f11")
    projection = _resident_projection(source)
    rng_state, rng_tensors = _rng()
    receipt = assert_capacity_migration_rejected(
        source_metadata=projection["metadata"],
        source_tensors=projection["tensors"],
        authorization=None,
        expected_rejection="unauthorized_request",
        rng_state_before=deepcopy(rng_state),
        rng_state_after=deepcopy(rng_state),
        rng_tensors_before={name: value.clone() for name, value in rng_tensors.items()},
        rng_tensors_after={name: value.clone() for name, value in rng_tensors.items()},
    )
    assert receipt["status"] == "rejected_by_public_cpu_migration"
    assert receipt["rng_draw_count"] == 0

    unsafe = _authorization(source.packed_sha256, safe_boundary="mid_generation")
    receipt = assert_capacity_migration_rejected(
        source_metadata=projection["metadata"],
        source_tensors=projection["tensors"],
        authorization=unsafe,
        expected_rejection="wrong_safe_boundary",
        rng_state_before=deepcopy(rng_state),
        rng_state_after=deepcopy(rng_state),
        rng_tensors_before={name: value.clone() for name, value in rng_tensors.items()},
        rng_tensors_after={name: value.clone() for name, value in rng_tensors.items()},
    )
    assert "safe boundary" in receipt["observed_error"]

    with pytest.raises(PhaseFBoundaryActionError, match="exact RNG state"):
        assert_capacity_migration_rejected(
            source_metadata=projection["metadata"],
            source_tensors=projection["tensors"],
            authorization=None,
            expected_rejection="unauthorized_request",
            rng_state_before=deepcopy(rng_state),
            rng_state_after={"stream_index": 8},
            rng_tensors_before={
                name: value.clone() for name, value in rng_tensors.items()
            },
            rng_tensors_after={
                name: value.clone() for name, value in rng_tensors.items()
            },
        )


def test_checkpoint_schedule_capture_and_fresh_owner_restore_are_exact() -> None:
    fixture, source = _packed("f12")
    schedule = build_checkpoint_schedule_descriptor(
        case_id=fixture["case_id"],
        case_sha256=fixture["case_sha256"],
        matrix_sha256=fixture["matrix_sha256"],
        profile_id=fixture["profile_id"],
        schedule_kind="pre_tick_fresh_owner",
    )
    checkpoint = capture_modular_checkpoint(source, schedule)
    restored = restore_fresh_modular_owner(checkpoint)

    assert restored.owner.packed_sha256 == source.packed_sha256
    assert restored.owner.identity_table == source.identity_table
    assert all(
        torch.equal(restored.owner.tensors[name], source.tensors[name])
        for name in source.tensors
    )
    assert all(
        restored.owner.tensors[name].data_ptr() != source.tensors[name].data_ptr()
        for name in source.tensors
    )
    assert restored.receipt["checkpoint_scope"] == ("fresh_modular_storage_owner_only")
    assert restored.receipt["rng_draw_count"] == 0


def test_checkpoint_boundary_prerequisites_and_full_runtime_fail_closed() -> None:
    fixture = _case("f12", profile="b16", role="r1")
    with pytest.raises(PhaseFBoundaryActionError, match="prepared_prune_reuse"):
        build_checkpoint_schedule_descriptor(
            case_id=fixture["case_id"],
            case_sha256=fixture["case_sha256"],
            matrix_sha256=fixture["matrix_sha256"],
            profile_id=fixture["profile_id"],
            schedule_kind="post_reuse_pre_tick",
        )

    with pytest.raises(PhaseFBoundaryActionError, match="prerequisites are missing"):
        require_full_runtime_resume_prerequisites()
    with pytest.raises(PhaseFBoundaryActionError, match="outside the modular"):
        require_full_runtime_resume_prerequisites(
            {
                "resident_owner_checkpoint_projection": object(),
                "simulation_runner_checkpoint_bundle": object(),
                "authoritative_rng_state_and_tensor_projection": object(),
                "generation_and_tick_boundary_metadata": object(),
            }
        )
