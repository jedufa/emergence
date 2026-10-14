from __future__ import annotations

"""Phase E live-generation entry contract without an execution call.

The existing generation-v3 call is intentionally untouched.  This optional
bridge only validates and orders the committed 57-plane resident modular state
bank for a forthcoming CUDA entry point.  It performs no simulation work and
supports no birth, turnover, or modular lifecycle operation.
"""

from dataclasses import dataclass
from typing import Any, Final, Mapping

import torch

from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_PROJECTION_PREFIX,
)
from perf.cuda_architecture_b_modular_state_bank import (
    MODULAR_STATE_BANK_FIELD_COUNT,
    ModularStateBankManifest,
    build_modular_state_bank_manifest,
)
from evolution.structural_development_v1 import (
    LATE_START_STEP,
    MATURE_START_STEP,
)


MODULAR_GENERATION_ENTRY_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_modular_generation_entry_v1"
)
MODULAR_GENERATION_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_modular_generation_entry_receipt_v1"
)
MODULAR_GENERATION_ENTRY_ERROR: Final[str] = (
    "EIEL_CUDA_MODULAR_GENERATION_ENTRY_REJECTED"
)
SUPPORTED_PROFILE_SHAPES: Final[dict[str, tuple[int, int]]] = {
    "b2-12x32-v1": (12, 32),
    "b2-16x48-v1": (16, 48),
}
MODULAR_GENERATION_CONFIG_SCALAR_NAMES: Final[tuple[str, ...]] = (
    "agent_capacity",
    "action_count",
    "observation_width",
    "cache_width",
    "policy_stride",
    "logical_module_slots",
    "logical_connection_slots",
    "physical_module_slots",
    "physical_connection_slots",
    "development_mature_start_step",
    "development_late_start_step",
    "max_births",
    "turnover_enabled",
)


class ModularGenerationEntryError(RuntimeError):
    """The future modular generation entry cannot accept this boundary."""


def _fail(message: str) -> ModularGenerationEntryError:
    return ModularGenerationEntryError(f"{MODULAR_GENERATION_ENTRY_ERROR}: {message}")


def _exact_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(f"{name} must be an exact int >= {minimum}")
    return value


def _matches_device(actual: torch.device, expected: torch.device) -> bool:
    return actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    )


@dataclass(frozen=True)
class ModularGenerationEntryReceipt:
    manifest_sha256: str
    profile_id: str
    agent_capacity: int
    device: str
    cpu_validation_mode: bool
    tensor_argument_names: tuple[str, ...]
    config_scalar_names: tuple[str, ...]
    config_scalar_values: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODULAR_GENERATION_RECEIPT_SCHEMA_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "profile_id": self.profile_id,
            "agent_capacity": self.agent_capacity,
            "device": self.device,
            "cpu_validation_mode": self.cpu_validation_mode,
            "field_count": len(self.tensor_argument_names),
            "tensor_argument_names": list(self.tensor_argument_names),
            "config_scalar_names": list(self.config_scalar_names),
            "config_scalar_values": list(self.config_scalar_values),
            "resident_arguments_validated": True,
            "execution_performed": False,
            "generation_v3_call_changed": False,
            "birth_lifecycle_supported": False,
            "turnover_lifecycle_supported": False,
            "live_module_execution_claimed": False,
        }


@dataclass(frozen=True)
class ModularGenerationEntryContract:
    schema_version: str
    manifest: ModularStateBankManifest
    tensor_arguments: tuple[torch.Tensor, ...]
    config_scalar_names: tuple[str, ...]
    config_scalar_values: tuple[int, ...]
    receipt: ModularGenerationEntryReceipt

    @property
    def ordered_call_arguments(self) -> tuple[Any, ...]:
        """Future-call argument order; reading it performs no execution."""

        return (*self.tensor_arguments, *self.config_scalar_values)


def _validate_owner_bindings(
    resident_tensors: Mapping[str, Any],
    *,
    capacity: int,
    target: torch.device,
) -> None:
    bindings = (
        ("slot_to_agent_id", torch.int64),
        ("slot_tokens", torch.int64),
        ("alive_mask", torch.bool),
    )
    for name, dtype in bindings:
        tensor = resident_tensors.get(name)
        if (
            not isinstance(tensor, torch.Tensor)
            or not _matches_device(tensor.device, target)
            or tensor.dtype != dtype
            or tuple(tensor.shape) != (capacity,)
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise _fail(f"invalid resident owner binding tensor: {name}")

    prefix = MODULAR_STORAGE_PROJECTION_PREFIX
    occupied = resident_tensors[f"{prefix}agent_occupied"]
    agent_id = resident_tensors[f"{prefix}agent_id"]
    agent_slot_token = resident_tensors[f"{prefix}agent_slot_token"]
    alive = resident_tensors["alive_mask"]
    slot_to_agent_id = resident_tensors["slot_to_agent_id"]
    slot_tokens = resident_tensors["slot_tokens"]
    comparisons = (
        (
            torch.all(torch.eq(occupied, alive)),
            "modular occupancy disagrees with resident alive_mask",
        ),
        (
            torch.all(torch.eq(agent_id, slot_to_agent_id)),
            "modular agent IDs disagree with resident slot ownership",
        ),
        (
            torch.all(
                torch.logical_or(
                    torch.logical_not(alive),
                    torch.eq(agent_slot_token, slot_tokens),
                )
            ),
            "modular slot tokens disagree with resident slot authority",
        ),
    )
    for comparison, message in comparisons:
        if target.type == "cuda":
            try:
                torch._assert_async(comparison, message)
            except RuntimeError as exc:
                raise _fail(str(exc)) from exc
        elif not bool(comparison):
            raise _fail(message)


def prepare_modular_generation_entry(
    resident_tensors: Any,
    *,
    enabled: bool,
    agent_capacity: int,
    profile_id: str,
    policy_family: str,
    action_count: int,
    observation_width: int,
    cache_width: int,
    policy_stride: int,
    development_mature_start_step: int,
    development_late_start_step: int,
    max_births: int,
    turnover_enabled: bool,
    device: str | torch.device = "cuda",
    allow_cpu_validation: bool = False,
) -> ModularGenerationEntryContract | None:
    """Validate and order the future modular CUDA generation arguments.

    ``enabled=False`` returns before any other argument, including
    ``resident_tensors``, is inspected.  CPU is allowed only for explicit test
    validation; the production target remains CUDA.
    """

    if enabled is False:
        return None
    if enabled is not True:
        raise _fail("enabled must be an exact bool")
    if not isinstance(resident_tensors, Mapping):
        raise _fail("resident_tensors must be a mapping")
    capacity = _exact_int(agent_capacity, name="agent_capacity", minimum=1)
    if type(profile_id) is not str or profile_id not in SUPPORTED_PROFILE_SHAPES:
        raise _fail("logical modular profile is unsupported")
    if policy_family != "fixed_mlp":
        raise _fail("policy_family must be fixed_mlp")
    if _exact_int(action_count, name="action_count") != 7:
        raise _fail("action_count must equal 7")
    if _exact_int(observation_width, name="observation_width") != 27:
        raise _fail("observation_width must equal 27")
    if _exact_int(cache_width, name="cache_width") != 1:
        raise _fail("cache_width must equal 1")
    if _exact_int(policy_stride, name="policy_stride", minimum=1) != 1:
        raise _fail("policy_stride must equal 1")
    if (
        _exact_int(
            development_mature_start_step,
            name="development_mature_start_step",
            minimum=1,
        )
        != MATURE_START_STEP
        or _exact_int(
            development_late_start_step,
            name="development_late_start_step",
            minimum=2,
        )
        != LATE_START_STEP
    ):
        raise _fail(
            "development schedule must equal the accepted 20/40 boundaries"
        )
    if _exact_int(max_births, name="max_births") != 0:
        raise _fail("first live modular milestone requires max_births=0")
    if type(turnover_enabled) is not bool or turnover_enabled is not False:
        raise _fail("first live modular milestone requires turnover disabled")
    if type(allow_cpu_validation) is not bool:
        raise _fail("allow_cpu_validation must be an exact bool")

    target = torch.device(device)
    if target.type == "cpu":
        if not allow_cpu_validation:
            raise _fail("CPU is permitted only in explicit validation mode")
    elif target.type != "cuda":
        raise _fail("modular generation target must be CUDA")
    elif allow_cpu_validation:
        raise _fail("CPU validation mode contradicts a CUDA target")

    manifest = build_modular_state_bank_manifest(capacity, profile_id)
    logical_shape = SUPPORTED_PROFILE_SHAPES[profile_id]
    if (
        manifest.logical_module_slots != logical_shape[0]
        or manifest.logical_connection_slots != logical_shape[1]
        or manifest.physical_module_slots != 16
        or manifest.physical_connection_slots != 48
        or len(manifest.fields) != MODULAR_STATE_BANK_FIELD_COUNT
    ):
        raise _fail("committed modular manifest shape is not supported")

    expected_names = tuple(field.projected_name for field in manifest.fields)
    expected_set = frozenset(expected_names)
    actual_set = frozenset(
        str(name)
        for name in resident_tensors
        if str(name).startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    )
    mixed_names = sorted(
        str(name)
        for name in resident_tensors
        if str(name).startswith("phase_e_modular_")
        and not str(name).startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    )
    if mixed_names:
        raise _fail(f"mixed modular plane versions are forbidden: {mixed_names!r}")
    if actual_set != expected_set:
        raise _fail(
            "resident modular plane set mismatch "
            f"missing={sorted(expected_set-actual_set)!r};"
            f"extra={sorted(actual_set-expected_set)!r}"
        )

    ordered: list[torch.Tensor] = []
    for field in manifest.fields:
        tensor = resident_tensors[field.projected_name]
        if (
            not isinstance(tensor, torch.Tensor)
            or not _matches_device(tensor.device, target)
            or tensor.dtype != field.torch_dtype
            or tuple(tensor.shape) != field.shape
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise _fail(f"invalid resident modular plane: {field.projected_name}")
        ordered.append(tensor)

    _validate_owner_bindings(
        resident_tensors,
        capacity=capacity,
        target=target,
    )
    scalar_values = (
        capacity,
        action_count,
        observation_width,
        cache_width,
        policy_stride,
        manifest.logical_module_slots,
        manifest.logical_connection_slots,
        manifest.physical_module_slots,
        manifest.physical_connection_slots,
        development_mature_start_step,
        development_late_start_step,
        max_births,
        int(turnover_enabled),
    )
    receipt = ModularGenerationEntryReceipt(
        manifest_sha256=manifest.manifest_sha256,
        profile_id=manifest.profile_id,
        agent_capacity=capacity,
        device=str(target),
        cpu_validation_mode=allow_cpu_validation,
        tensor_argument_names=expected_names,
        config_scalar_names=MODULAR_GENERATION_CONFIG_SCALAR_NAMES,
        config_scalar_values=scalar_values,
    )
    return ModularGenerationEntryContract(
        schema_version=MODULAR_GENERATION_ENTRY_SCHEMA_VERSION,
        manifest=manifest,
        tensor_arguments=tuple(ordered),
        config_scalar_names=MODULAR_GENERATION_CONFIG_SCALAR_NAMES,
        config_scalar_values=scalar_values,
        receipt=receipt,
    )


__all__ = [
    "MODULAR_GENERATION_CONFIG_SCALAR_NAMES",
    "MODULAR_GENERATION_ENTRY_ERROR",
    "MODULAR_GENERATION_ENTRY_SCHEMA_VERSION",
    "MODULAR_GENERATION_RECEIPT_SCHEMA_VERSION",
    "SUPPORTED_PROFILE_SHAPES",
    "ModularGenerationEntryContract",
    "ModularGenerationEntryError",
    "ModularGenerationEntryReceipt",
    "prepare_modular_generation_entry",
]
