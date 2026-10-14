"""CPU-only contract tests for the fail-closed C4 R2 birth adapter."""

from __future__ import annotations

import pytest
import torch

from engine.perf.cuda_architecture_b_birth_transaction import (
    DRIVE_STATE_FIELDS,
    GENOME_FLOAT32_FIELDS,
    GENOME_INT32_FIELDS,
    IDENTITY_STATE_FIELDS,
    SCALAR_RNG_STATE_FIELDS,
)
from engine.tensor_world import c4_resident_birth
from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_population import TensorWorldC4BirthConfigV1
from engine.tensor_world.c4_resident_birth import (
    C4_RESIDENT_BIRTH_POLICY_FIELDS_V1,
    C4_RESIDENT_BIRTH_STATIC_BLOCKERS_V1,
    C4_RESIDENT_BIRTH_UNRESOLVED_SOURCE_FIELDS_V1,
    execute_c4_resident_birth_transaction_cuda_v1,
    inspect_c4_resident_birth_compatibility_v1,
    project_c4_resident_birth_shadow_v1,
    validate_c4_resident_birth_compatibility_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


def _fixture_and_config():
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    config = TensorWorldC4BirthConfigV1(allow_mutation=True)
    return bundle, config


def test_native_field_inventory_is_complete_and_ordinary_only() -> None:
    bundle, _ = _fixture_and_config()
    shadow = project_c4_resident_birth_shadow_v1(bundle)

    assert set(shadow.identity_state) == set(IDENTITY_STATE_FIELDS)
    assert set(shadow.scalar_rng_state) == set(SCALAR_RNG_STATE_FIELDS)
    assert set(shadow.genome_state) == set(
        (*GENOME_INT32_FIELDS, *GENOME_FLOAT32_FIELDS)
    )
    assert set(shadow.drive_state).issubset(DRIVE_STATE_FIELDS)
    assert set(shadow.policy_state).issubset(C4_RESIDENT_BIRTH_POLICY_FIELDS_V1)
    assert "policy_birth_crossover_mask_packed_tape" not in (
        C4_RESIDENT_BIRTH_POLICY_FIELDS_V1
    )


def test_projection_uses_source_aliases_and_never_materializes_spawn_plan() -> None:
    bundle, _ = _fixture_and_config()
    shadow = project_c4_resident_birth_shadow_v1(bundle)

    assert (
        shadow.identity_state["slot_to_agent_id"].untyped_storage().data_ptr()
        == bundle.base.base.core.agents.agent_id.untyped_storage().data_ptr()
    )
    assert (
        shadow.policy_state["mlp_pack"].untyped_storage().data_ptr()
        == bundle.base.base.core.policy.packed_weights.untyped_storage().data_ptr()
    )
    assert shadow.parent_slots.dtype == torch.int32
    assert shadow.spawn_positions is None


def test_current_c4_projection_reports_the_minimal_exact_execution_gaps() -> None:
    bundle, config = _fixture_and_config()
    shadow = project_c4_resident_birth_shadow_v1(bundle)

    result = inspect_c4_resident_birth_compatibility_v1(bundle, shadow, config)

    assert result.compatible is False
    assert "non_cuda_state:bundle" in result.blockers
    assert set(C4_RESIDENT_BIRTH_STATIC_BLOCKERS_V1).issubset(result.blockers)
    assert set(C4_RESIDENT_BIRTH_UNRESOLVED_SOURCE_FIELDS_V1).issubset(
        result.missing_native_fields
    )
    assert "dtype_mismatch:parent_slots_native_int64" in result.blockers
    for field in ("pos_y", "pos_x", "lineage_founder_id"):
        assert any(
            blocker.startswith(f"dtype_mismatch:identity.{field}:")
            for blocker in result.blockers
        )


def test_missing_cpython_source_state_is_rejected_explicitly() -> None:
    bundle, config = _fixture_and_config()
    bundle.scalar_rng = None
    shadow = project_c4_resident_birth_shadow_v1(bundle)

    result = inspect_c4_resident_birth_compatibility_v1(bundle, shadow, config)

    assert all(
        f"missing_source_state:scalar_rng.{name}" in result.blockers
        for name in SCALAR_RNG_STATE_FIELDS
    )


def test_dtype_mismatch_is_fail_closed_before_native_execution() -> None:
    bundle, config = _fixture_and_config()
    shadow = project_c4_resident_birth_shadow_v1(bundle)
    shadow.identity_state = dict(shadow.identity_state)
    shadow.identity_state["slot_tokens"] = shadow.identity_state["slot_tokens"].to(
        torch.int32
    )

    result = inspect_c4_resident_birth_compatibility_v1(bundle, shadow, config)

    assert any(
        blocker.startswith("dtype_mismatch:identity.slot_tokens")
        for blocker in result.blockers
    )


def test_same_dtype_copy_is_rejected_as_an_aliasing_mismatch() -> None:
    bundle, config = _fixture_and_config()
    shadow = project_c4_resident_birth_shadow_v1(bundle)
    shadow.identity_state = dict(shadow.identity_state)
    shadow.identity_state["slot_tokens"] = shadow.identity_state["slot_tokens"].clone()

    result = inspect_c4_resident_birth_compatibility_v1(bundle, shadow, config)

    assert "aliasing_mismatch:identity.slot_tokens" in result.blockers


@pytest.mark.parametrize(
    ("mate_slots", "recombination"),
    ((torch.zeros(1, dtype=torch.int64), False), (None, True)),
)
def test_mate_and_recombination_requests_are_not_simplified(
    mate_slots: torch.Tensor | None, recombination: bool
) -> None:
    bundle, config = _fixture_and_config()
    shadow = project_c4_resident_birth_shadow_v1(bundle)

    result = inspect_c4_resident_birth_compatibility_v1(
        bundle,
        shadow,
        config,
        mate_slots=mate_slots,
        recombination=recombination,
    )

    assert "unsupported_mate_or_recombination" in result.blockers


def test_validator_and_executor_fail_before_loading_native_helper(monkeypatch) -> None:
    bundle, config = _fixture_and_config()
    shadow = project_c4_resident_birth_shadow_v1(bundle)
    called = False

    def forbidden_native_call(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("native helper must not run")

    monkeypatch.setattr(
        c4_resident_birth,
        "execute_fixed_mlp_birth_transaction_cuda",
        forbidden_native_call,
    )

    with pytest.raises(TensorWorldContractError, match="compatibility failed"):
        validate_c4_resident_birth_compatibility_v1(bundle, shadow, config)
    with pytest.raises(TensorWorldContractError, match="compatibility failed"):
        execute_c4_resident_birth_transaction_cuda_v1(
            bundle,
            shadow,
            config,
            sensor_radius_max=3,
        )
    assert called is False


def test_unexpected_shadow_fields_are_rejected() -> None:
    bundle, config = _fixture_and_config()
    shadow = project_c4_resident_birth_shadow_v1(bundle)
    shadow.drive_state = dict(shadow.drive_state)
    shadow.drive_state["invented_rng"] = torch.zeros(1)

    result = inspect_c4_resident_birth_compatibility_v1(bundle, shadow, config)

    assert "unexpected_shadow_field:drive.invented_rng" in result.blockers
