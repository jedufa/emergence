from __future__ import annotations

from functools import lru_cache

import pytest
import torch

from evolution.phase_f_v2.matrix import build_phase_f_v2_matrix
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_PROJECTION_PREFIX,
)
from perf.eiel_phase_f_boundary_actions_v2 import (
    build_checkpoint_schedule_descriptor,
    capture_modular_checkpoint,
    restore_fresh_modular_owner,
)
from perf.eiel_phase_f_execution_source_v2 import (
    PhaseFV2ExecutionSourceError,
    build_phase_f_migration_context_v2,
    prepare_phase_f_execution_source_v2,
)
from perf.eiel_phase_f_forced_combination_fixture_v2 import (
    build_phase_f_fixture_registry_v2,
)
from perf.eiel_phase_f_forced_combination_source_v2 import (
    materialize_phase_f_source_v2,
)


@lru_cache(maxsize=1)
def _matrix() -> dict:
    return build_phase_f_v2_matrix()


@lru_cache(maxsize=1)
def _registry() -> dict:
    return build_phase_f_fixture_registry_v2(matrix=_matrix())


def _case(case_id: str) -> dict:
    return next(row for row in _matrix()["cases"] if row["case_id"] == case_id)


def _fixture(case_id: str) -> dict:
    return next(row for row in _registry()["cases"] if row["case_id"] == case_id)


def _migration_context(source_case_id: str) -> dict:
    return build_phase_f_migration_context_v2(
        matrix=_matrix(),
        fixture_registry=_registry(),
        source_case_id=source_case_id,
        phase_e_final_receipt_sha256="e" * 64,
        phase_e_execution_commit="c" * 40,
    )


def _prepare(case_id: str):
    case = _case(case_id)
    source_case_id = case["role_contract"]["boundary_source_case_id"]
    context = (
        None if source_case_id is None else _migration_context(str(source_case_id))
    )
    return prepare_phase_f_execution_source_v2(
        matrix=_matrix(),
        fixture_registry=_registry(),
        case_id=case_id,
        migration_context=context,
    )


def test_unchanged_roles_keep_the_exact_materialized_source() -> None:
    for case_id in (
        "pf2_f09_b12_r0",
        "pf2_f09_b12_r1",
        "pf2_f11_b12_r0",
        "pf2_f11_b12_r1",
        "pf2_f11_b12_r4",
        "pf2_f12_b12_r0",
        "pf2_f12_b16_r4",
    ):
        case = _case(case_id)
        fixture = _fixture(case_id)
        raw = materialize_phase_f_source_v2(
            fixture,
            source_seed=int(case["fixture_blueprint"]["construction_seed"]),
        )
        prepared = _prepare(case_id)
        assert prepared.source.source_sha256 == raw.source_sha256
        assert (
            prepared.owner.packed_sha256 == raw.owner_metadata["packed_storage_sha256"]
        )
        assert prepared.boundary_receipt is None
        assert prepared.preparation_receipt["preparation_kind"] == "unchanged"


@pytest.mark.parametrize(
    "case_id",
    [
        "pf2_f09_b12_r2",
        "pf2_f09_b12_r4",
        "pf2_f09_b16_r2",
        "pf2_f09_b16_r4",
        "pf2_f12_b16_r1",
    ],
)
def test_prune_reuse_returns_one_actual_detached_owner(case_id: str) -> None:
    prepared = _prepare(case_id)
    receipt = prepared.boundary_receipt
    assert receipt is not None
    assert receipt["action"] == "prepared_prune_reuse"
    assert receipt["semantic_repack_verified"] is True
    assert receipt["target_packed_sha256"] == prepared.owner.packed_sha256
    assert prepared.source.owner_metadata["packed_storage_sha256"] == (
        prepared.owner.packed_sha256
    )
    for slot, formal in enumerate(prepared.source.formal_oracle_inputs):
        for name, plane in formal.planes.items():
            generation = prepared.source.generation_tensors[
                f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}"
            ][slot : slot + 1]
            assert torch.equal(plane, generation), name
            assert plane.data_ptr() != generation.data_ptr()


@pytest.mark.parametrize("profile", ["b12", "b16"])
def test_rejected_f9_requests_preserve_source_and_rng(profile: str) -> None:
    case_id = f"pf2_f09_{profile}_r3"
    before = torch.get_rng_state().clone()
    prepared = _prepare(case_id)
    receipt = prepared.boundary_receipt
    assert receipt is not None
    assert receipt["action"] == "rejected_prune_requests"
    assert [(row["status"], row["reason"]) for row in receipt["rejections"]] == [
        ("dependency_blocked", "required_by_development"),
        ("ineligible", "minimum_module_invariant"),
    ]
    assert receipt["source_packed_sha256"] == receipt["result_packed_sha256"]
    assert torch.equal(before, torch.get_rng_state())


def test_f11_rejections_are_public_exact_and_unchanged() -> None:
    expected = {
        "pf2_f11_b12_r2": "unauthorized_request",
        "pf2_f11_b12_r3": "wrong_safe_boundary",
    }
    for case_id, rejection in expected.items():
        prepared = _prepare(case_id)
        receipt = prepared.boundary_receipt
        assert receipt is not None
        assert receipt["action"] == "rejected_capacity_migration"
        assert receipt["expected_rejection"] == rejection
        assert receipt["source_packed_sha256"] == prepared.owner.packed_sha256
        assert prepared.source.case_id == case_id


def test_all_f11_b16_roles_migrate_the_same_role_bound_b12_owner() -> None:
    for role in range(5):
        target_id = f"pf2_f11_b16_r{role}"
        source_id = f"pf2_f11_b12_r{role}"
        prepared = prepare_phase_f_execution_source_v2(
            matrix=_matrix(),
            fixture_registry=_registry(),
            case_id=target_id,
            migration_context=_migration_context(source_id),
        )
        receipt = prepared.boundary_receipt
        assert receipt is not None
        assert receipt["action"] == "accepted_capacity_migration"
        assert receipt["target_packed_sha256"] == prepared.owner.packed_sha256
        assert prepared.owner.layout.profile_id == "b2-16x48-v1"
        provenance = prepared.source.owner_metadata["execution_source_provenance"]
        assert provenance["source_case_id"] == source_id
        assert provenance["target_case_id"] == target_id
        assert prepared.source.case_id == target_id


@pytest.mark.parametrize(
    ("case_id", "source_case_id"),
    [
        ("pf2_f09_b12_r4", None),
        ("pf2_f09_b16_r4", None),
        ("pf2_f11_b16_r4", "pf2_f11_b12_r4"),
        ("pf2_f12_b16_r1", None),
        ("pf2_f12_b16_r2", "pf2_f12_b12_r2"),
    ],
)
def test_all_five_post_boundary_checkpoints_capture_the_actual_owner(
    case_id: str, source_case_id: str | None
) -> None:
    prepared = prepare_phase_f_execution_source_v2(
        matrix=_matrix(),
        fixture_registry=_registry(),
        case_id=case_id,
        migration_context=(
            None if source_case_id is None else _migration_context(source_case_id)
        ),
    )
    case = _case(case_id)
    prerequisite = prepared.checkpoint_prerequisite_receipt
    assert prerequisite is not None
    schedule = build_checkpoint_schedule_descriptor(
        case_id=case_id,
        case_sha256=case["case_sha256"],
        matrix_sha256=_matrix()["matrix_sha256"],
        profile_id=case["profile_id"],
        schedule_kind=case["checkpoint_schedule"]["kind"],
        prerequisite_receipt=prerequisite,
    )
    checkpoint = capture_modular_checkpoint(prepared.owner, schedule)
    restored = restore_fresh_modular_owner(checkpoint)
    assert restored.owner.packed_sha256 == prepared.owner.packed_sha256
    assert all(
        restored.owner.tensors[name].data_ptr()
        != prepared.owner.tensors[name].data_ptr()
        for name in prepared.owner.tensors
    )


def test_migration_context_is_deterministic_and_fail_closed() -> None:
    first = _migration_context("pf2_f12_b12_r2")
    second = _migration_context("pf2_f12_b12_r2")
    assert first["authorization"] == second["authorization"]
    assert first["phase_e_receipt_binding"] == second["phase_e_receipt_binding"]
    assert all(
        torch.equal(first["rng_tensors"][name], second["rng_tensors"][name])
        for name in first["rng_tensors"]
    )
    with pytest.raises(PhaseFV2ExecutionSourceError, match="migration context"):
        prepare_phase_f_execution_source_v2(
            matrix=_matrix(),
            fixture_registry=_registry(),
            case_id="pf2_f12_b16_r2",
        )
    with pytest.raises(PhaseFV2ExecutionSourceError, match="SHA-256"):
        build_phase_f_migration_context_v2(
            matrix=_matrix(),
            fixture_registry=_registry(),
            source_case_id="pf2_f12_b12_r2",
            phase_e_final_receipt_sha256="not-a-digest",
            phase_e_execution_commit="c" * 40,
        )
