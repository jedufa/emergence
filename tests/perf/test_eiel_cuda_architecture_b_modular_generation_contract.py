from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from perf.cuda_architecture_b_modular_generation_contract import (
    MODULAR_GENERATION_CONFIG_SCALAR_NAMES,
    MODULAR_GENERATION_ENTRY_ERROR,
    MODULAR_GENERATION_ENTRY_SCHEMA_VERSION,
    SUPPORTED_PROFILE_SHAPES,
    ModularGenerationEntryError,
    prepare_modular_generation_entry,
)
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_PROJECTION_PREFIX,
)
from perf.cuda_architecture_b_modular_state_bank import (
    MODULAR_STATE_BANK_FIELD_COUNT,
    build_modular_state_bank_manifest,
)


CAPACITY = 4


def _resident(profile_id: str = "b2-12x32-v1") -> dict[str, torch.Tensor]:
    manifest = build_modular_state_bank_manifest(CAPACITY, profile_id)
    tensors = {
        field.projected_name: torch.full(
            field.shape,
            field.fill_value,
            dtype=field.torch_dtype,
        )
        for field in manifest.fields
    }
    tensors.update(
        {
            "slot_to_agent_id": torch.tensor([7, 11, -1, -1], dtype=torch.int64),
            "slot_tokens": torch.tensor([70, 110, 3, 4], dtype=torch.int64),
            "alive_mask": torch.tensor([True, True, False, False]),
        }
    )
    prefix = MODULAR_STORAGE_PROJECTION_PREFIX
    tensors[f"{prefix}agent_occupied"][:] = tensors["alive_mask"]
    tensors[f"{prefix}agent_id"][:] = tensors["slot_to_agent_id"]
    tensors[f"{prefix}agent_slot_token"][:] = torch.tensor(
        [70, 110, -1, -1], dtype=torch.int64
    )
    return tensors


def _prepare(
    resident: object,
    *,
    enabled: bool = True,
    profile_id: str = "b2-12x32-v1",
    **updates,
):
    values = {
        "agent_capacity": CAPACITY,
        "profile_id": profile_id,
        "policy_family": "fixed_mlp",
        "action_count": 7,
        "observation_width": 27,
        "cache_width": 1,
        "policy_stride": 1,
        "development_mature_start_step": 20,
        "development_late_start_step": 40,
        "max_births": 0,
        "turnover_enabled": False,
        "device": "cpu",
        "allow_cpu_validation": True,
    }
    values.update(updates)
    return prepare_modular_generation_entry(resident, enabled=enabled, **values)


def test_default_off_returns_before_any_modular_or_config_traversal() -> None:
    class MustNotBeRead:
        def __getattribute__(self, name):
            raise AssertionError(f"default-off argument was read: {name}")

    assert prepare_modular_generation_entry(
        MustNotBeRead(),
        enabled=False,
        agent_capacity=-1,
        profile_id="unsupported",
        policy_family="gru",
        action_count=-1,
        observation_width=-1,
        cache_width=-1,
        policy_stride=-1,
        development_mature_start_step=-1,
        development_late_start_step=-1,
        max_births=99,
        turnover_enabled=True,
        device="not-a-device",
        allow_cpu_validation=False,
    ) is None


@pytest.mark.parametrize("profile_id", tuple(SUPPORTED_PROFILE_SHAPES))
def test_manifest_order_bridge_is_exact_for_both_logical_profiles(
    profile_id: str,
) -> None:
    resident = _resident(profile_id)
    entry = _prepare(resident, profile_id=profile_id)
    assert entry is not None
    expected_names = tuple(field.projected_name for field in entry.manifest.fields)
    assert len(expected_names) == MODULAR_STATE_BANK_FIELD_COUNT == 57
    assert entry.schema_version == MODULAR_GENERATION_ENTRY_SCHEMA_VERSION
    assert entry.receipt.tensor_argument_names == expected_names
    assert all(
        resident[name] is argument
        for name, argument in zip(expected_names, entry.tensor_arguments, strict=True)
    )
    assert entry.config_scalar_names == MODULAR_GENERATION_CONFIG_SCALAR_NAMES
    assert entry.config_scalar_values[5:9] == (
        *SUPPORTED_PROFILE_SHAPES[profile_id],
        16,
        48,
    )
    assert entry.config_scalar_values[9:11] == (20, 40)
    assert entry.ordered_call_arguments[:57] == entry.tensor_arguments


def test_receipt_declares_validation_without_execution_or_promotion() -> None:
    entry = _prepare(_resident())
    assert entry is not None
    receipt = entry.receipt.to_dict()
    assert receipt["field_count"] == 57
    assert receipt["resident_arguments_validated"] is True
    assert receipt["execution_performed"] is False
    assert receipt["generation_v3_call_changed"] is False
    assert receipt["birth_lifecycle_supported"] is False
    assert receipt["turnover_lifecycle_supported"] is False
    assert receipt["live_module_execution_claimed"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("policy_family", "gru", "fixed_mlp"),
        ("action_count", 6, "action_count"),
        ("observation_width", 26, "observation_width"),
        ("cache_width", 2, "cache_width"),
        ("policy_stride", 2, "policy_stride"),
        ("profile_id", "b2-13x33-v1", "profile"),
        ("agent_capacity", True, "agent_capacity"),
    ),
)
def test_frozen_policy_shape_and_profile_contract_rejects_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ModularGenerationEntryError, match=message):
        _prepare(_resident(), **{field: value})


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"max_births": 1}, "max_births=0"),
        ({"turnover_enabled": True}, "turnover disabled"),
        ({"turnover_enabled": 0}, "turnover disabled"),
        (
            {"development_mature_start_step": 19},
            "development schedule",
        ),
        (
            {"development_late_start_step": 41},
            "development schedule",
        ),
    ),
)
def test_first_live_entry_rejects_birth_and_turnover_lifecycle(
    updates: dict,
    message: str,
) -> None:
    with pytest.raises(ModularGenerationEntryError, match=message):
        _prepare(_resident(), **updates)


@pytest.mark.parametrize("damage", ("missing", "extra", "mixed"))
def test_plane_set_is_closed_against_partial_extra_and_mixed_versions(
    damage: str,
) -> None:
    resident = _resident()
    name = next(
        key for key in resident if key.startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    )
    if damage == "missing":
        del resident[name]
    elif damage == "extra":
        resident[f"{MODULAR_STORAGE_PROJECTION_PREFIX}future_plane"] = torch.tensor([0])
    else:
        resident["phase_e_modular_v2__agent_occupied"] = torch.zeros(
            CAPACITY, dtype=torch.bool
        )
    with pytest.raises(ModularGenerationEntryError, match=MODULAR_GENERATION_ENTRY_ERROR):
        _prepare(resident)


@pytest.mark.parametrize("damage", ("dtype", "shape", "contiguity", "requires_grad"))
def test_each_resident_plane_requires_exact_descriptor_and_storage(damage: str) -> None:
    resident = _resident()
    manifest = build_modular_state_bank_manifest(CAPACITY)
    float_field = next(
        field for field in manifest.fields if field.torch_dtype == torch.float32 and len(field.shape) > 1
    )
    tensor = resident[float_field.projected_name]
    if damage == "dtype":
        resident[float_field.projected_name] = tensor.to(torch.float64)
    elif damage == "shape":
        resident[float_field.projected_name] = tensor.reshape(-1)
    elif damage == "contiguity":
        resident[float_field.projected_name] = (
            tensor.transpose(0, 1).contiguous().transpose(0, 1)
        )
        assert not resident[float_field.projected_name].is_contiguous()
    else:
        resident[float_field.projected_name] = tensor.requires_grad_(True)
    with pytest.raises(ModularGenerationEntryError, match="invalid resident modular plane"):
        _prepare(resident)


@pytest.mark.parametrize("binding", ("agent_id", "occupancy", "slot_token"))
def test_resident_owner_identity_occupancy_and_slot_tokens_are_authoritative(
    binding: str,
) -> None:
    resident = _resident()
    prefix = MODULAR_STORAGE_PROJECTION_PREFIX
    if binding == "agent_id":
        resident[f"{prefix}agent_id"][0] += 1
    elif binding == "occupancy":
        resident[f"{prefix}agent_occupied"][0] = False
    else:
        resident[f"{prefix}agent_slot_token"][0] += 1
    with pytest.raises(ModularGenerationEntryError, match=MODULAR_GENERATION_ENTRY_ERROR):
        _prepare(resident)


def test_cpu_validation_mode_must_be_explicit_and_consistent() -> None:
    with pytest.raises(ModularGenerationEntryError, match="explicit validation mode"):
        _prepare(_resident(), allow_cpu_validation=False)
    with pytest.raises(ModularGenerationEntryError, match="contradicts a CUDA target"):
        _prepare(_resident(), device="cuda", allow_cpu_validation=True)


def test_non_mapping_enabled_input_fails_closed() -> None:
    with pytest.raises(ModularGenerationEntryError, match="resident_tensors"):
        _prepare(SimpleNamespace())
