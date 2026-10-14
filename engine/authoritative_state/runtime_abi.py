"""Deterministic pre-dispatch observation of the direct-resident runtime ABI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ast
import math
from typing import Any, Final

import torch

from engine.perf.cuda_architecture_b_boundary import EVENT_COLUMN_NAMES
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
    PATCH_P2_STATE_FIELDS_V1,
    PreparedDirectResidentTrajectoryV1,
    validate_prepared_direct_resident_trajectory_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    ORDINARY_BIRTH_DRIVE_FIELDS,
    ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS,
    ORDINARY_BIRTH_GENOME_INT32_FIELDS,
    ORDINARY_BIRTH_IDENTITY_FIELDS,
    ORDINARY_BIRTH_OUTPUT_FIELDS,
    ORDINARY_BIRTH_POLICY_FIELDS,
    ORDINARY_BIRTH_SCALAR_RNG_FIELDS,
    TURNOVER_MUTABLE_STATE_FIELDS,
    TURNOVER_OUTPUT_FIELDS,
    TURNOVER_STATE_FIELDS,
)

from .errors import fail
from .generated.runtime_abi_contracts_v1 import (
    RUNTIME_ABI_PATH_CONTRACTS_V1,
)
from .identity import canonical_sha256


RUNTIME_ABI_RECEIPT_SCHEMA_V1: Final[str] = (
    "eiel_authoritative_state_runtime_abi_observation_v1"
)
RUNTIME_ABI_ERROR: Final[str] = "E_STATE_RUNTIME_ABI"
_ORDINARY_BIRTHS_ENABLED_ARGUMENT_INDEX: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("config.ordinary_births_enabled")
)
_GENERATION_TURNOVER_ENABLED_ARGUMENT_INDEX: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("config.generation_turnover_enabled")
)

_FULL_TICK_STATE_NAMES: Final[tuple[str, ...]] = (
    "age",
    "gru_upkeep_cost",
    "cost_charge_scalar",
    "cost_charge_maturation_scalar",
    "drive_inertia",
    "drive_noise",
    "prev_drive_delta",
    "drive_volatility",
    "drive_jitter",
    "episode0_drive_jitter_sum",
    "last_energy_delta",
    "last_health_delta",
    "last_injury_delta",
    "selfharm_events",
    "episode0_selfharm_events",
    "chemistry",
    "runner_pcg64_state_words",
    "runner_pcg64_buffer_words",
    "pcg64_state_words",
    "pcg64_buffer_words",
    "drive_noise_samples_by_tick_slot",
    "drive_noise_samples_valid_by_tick_slot",
    "slot_free_queue",
    "slot_free_queue_count",
    "residual_ids",
    "residual_slots",
    "residual_count",
    "policy_draws",
    "metrics_draws",
)
_FULL_TICK_OUTPUT_NAMES: Final[tuple[str, ...]] = (
    "chemistry_order_slots",
    "runner_shuffle_draws",
    "runner_shuffle_status",
    "drive_noise_draws",
    "drive_noise_raw64_draws",
    "drive_noise_status",
    "chemistry_draws",
    "chemistry_status",
    "active_mechanics_status",
    "active_mechanics_counts",
    "residual_death_agent_ids",
    "residual_death_slots",
    "residual_death_reason_codes",
    "residual_death_count",
    "residual_release_count",
    "residual_release_status",
    "policy_pcg64_state_after_action",
    "policy_pcg64_buffer_after_action",
    "food_spawn_stats",
    "generation_summary_scalar",
)
_EVENT_STATE_NAMES: Final[tuple[str, ...]] = (
    *EVENT_COLUMN_NAMES,
    "event_active_count",
    "event_status",
    "lineage_founder_id",
    "lineage_depth",
    "initial_agent_ids_by_row",
    "terminal_survival",
    "terminal_rows",
    "event_status_by_tick",
    "event_sequence_base",
)
_ORDINARY_BIRTH_STATE_NAMES: Final[tuple[str, ...]] = (
    *ORDINARY_BIRTH_IDENTITY_FIELDS,
    *ORDINARY_BIRTH_SCALAR_RNG_FIELDS,
    *ORDINARY_BIRTH_GENOME_INT32_FIELDS,
    "policy_mem_dim",
    "policy_use_gru",
    *ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS,
    *ORDINARY_BIRTH_DRIVE_FIELDS,
    *ORDINARY_BIRTH_POLICY_FIELDS,
    *TURNOVER_STATE_FIELDS,
    "policy_birth_tape_cursor_base",
    "python_random_raw_draw_base",
)

_NESTED_ARGUMENTS: Final[dict[int, tuple[tuple[str, ...], str]]] = {
    60: (_FULL_TICK_STATE_NAMES, "read_write"),
    61: (_FULL_TICK_OUTPUT_NAMES, "write_only"),
    62: ((), "read_only"),
    63: ((), "read_only"),
    65: ((), "write_only"),
    66: ((), "write_only"),
    67: (_EVENT_STATE_NAMES, "read_write"),
    68: (_ORDINARY_BIRTH_STATE_NAMES, "read_write"),
    69: (ORDINARY_BIRTH_OUTPUT_FIELDS, "write_only"),
    70: (("lifecycle_scratch_storage",), "write_only"),
    72: (TURNOVER_MUTABLE_STATE_FIELDS, "read_write"),
    73: (
        tuple(f"{name}_shadow" for name in TURNOVER_MUTABLE_STATE_FIELDS),
        "write_only",
    ),
    74: (TURNOVER_OUTPUT_FIELDS, "write_only"),
    75: (("turnover_pointer_table",), "write_only"),
    76: (PATCH_P2_STATE_FIELDS_V1, "read_write"),
    77: (("patch_p2_scratch",), "write_only"),
    78: (("physical_patch_stats",), "write_only"),
}


def _configuration_value(configuration: object, name: str, default: int) -> int:
    value = (
        configuration.get(name, default)
        if isinstance(configuration, Mapping)
        else getattr(configuration, name, default)
    )
    if isinstance(value, bool) or not isinstance(value, int):
        raise fail(
            RUNTIME_ABI_ERROR,
            f"prepared.source.configuration.{name}",
            "must be an integer",
        )
    return value


def _shape(value: torch.Tensor) -> list[int]:
    return [int(dimension) for dimension in value.shape]


def _small_int_vector(value: Any, *, path: str) -> list[int]:
    if not torch.is_tensor(value) or value.numel() > 256:
        raise fail(RUNTIME_ABI_ERROR, path, "must be a tensor with at most 256 values")
    if value.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise fail(RUNTIME_ABI_ERROR, path, "must have an integer dtype")
    return [int(item) for item in value.detach().reshape(-1).cpu().tolist()]


def _outer_mutability(name: str) -> str:
    if name.startswith("config.") or name.startswith("reporting_fitness."):
        return "not_tensor"
    if name.startswith("outputs.") or name.startswith("physical_"):
        return "write_only"
    if name.startswith("live_state.") or name in {
        "active_agent_ids",
        "active_count_state",
    }:
        return "read_write"
    if name == "compact_summary_checkpoint_descriptor_specs":
        return "read_only"
    return "not_tensor"


class _AliasTracker:
    def __init__(self) -> None:
        self._objects: dict[int, str] = {}
        self._storages: dict[tuple[str, int], str] = {}

    def identity(self, tensor: torch.Tensor) -> dict[str, Any]:
        object_key = id(tensor)
        object_alias = self._objects.setdefault(
            object_key, f"object-{len(self._objects):04d}"
        )
        storage = tensor.untyped_storage()
        pointer = int(storage.data_ptr())
        # Distinct zero-sized storages commonly share address zero. Object
        # identity preserves their real non-aliasing relationship.
        storage_key = (
            str(tensor.device),
            pointer if storage.nbytes() > 0 else id(storage),
        )
        storage_alias = self._storages.setdefault(
            storage_key, f"storage-{len(self._storages):04d}"
        )
        return {
            "object_alias": object_alias,
            "storage_alias": storage_alias,
            "storage_offset": int(tensor.storage_offset()),
            "stride": [int(value) for value in tensor.stride()],
        }


def resolve_prepared_direct_resident_trajectory_v1(
    prepared: Any,
) -> PreparedDirectResidentTrajectoryV1:
    """Resolve only sanctioned production wrappers to the canonical ABI owner."""

    if isinstance(prepared, PreparedDirectResidentTrajectoryV1):
        return prepared

    from engine.campaigns.production_engine_native_bindings import (
        PreparedNativeMechanicsOwnerV1,
    )

    if isinstance(prepared, PreparedNativeMechanicsOwnerV1):
        prepared = prepared.physical_window_arguments

    from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
        PreparedDirectResidentPatchMulticellV1,
    )

    if isinstance(prepared, PreparedDirectResidentPatchMulticellV1):
        trajectory = prepared.patch
    else:
        from engine.perf.cuda_architecture_b_direct_resident_gmnw_gru_v1 import (
            PreparedDirectResidentGmnwGruV1,
        )
        from engine.perf.cuda_architecture_b_direct_resident_gru_v3 import (
            PreparedDirectResidentGruV3,
        )
        from engine.perf.cuda_architecture_b_direct_resident_eed_v1 import (
            PreparedDirectResidentEedV1,
        )
        from engine.perf.cuda_architecture_b_direct_resident_jungle_v4 import (
            PreparedDirectResidentJungleV4,
        )
        from engine.perf.cuda_architecture_b_direct_resident_wdl_gru_v3 import (
            PreparedDirectResidentWdlGruV3,
        )
        from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
            PreparedDirectResidentWdlV2,
        )
        from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
            PreparedDirectResidentWdlJungleV1,
        )

        configured_wrappers = (
            PreparedDirectResidentWdlV2,
            PreparedDirectResidentWdlJungleV1,
            PreparedDirectResidentJungleV4,
            PreparedDirectResidentGruV3,
            PreparedDirectResidentEedV1,
            PreparedDirectResidentWdlGruV3,
            PreparedDirectResidentGmnwGruV1,
        )
        if isinstance(prepared, configured_wrappers):
            trajectory = prepared.patch
        else:
            trajectory = None

    if trajectory is None:
        raise fail(
            RUNTIME_ABI_ERROR,
            "prepared",
            "must be a direct-resident owner or a sanctioned typed wrapper",
        )

    if not isinstance(trajectory, PreparedDirectResidentTrajectoryV1):
        raise fail(
            RUNTIME_ABI_ERROR,
            "prepared.patch",
            "typed wrapper does not own a direct-resident trajectory",
        )
    return trajectory


def _shape_inputs(prepared: PreparedDirectResidentTrajectoryV1) -> dict[str, Any]:
    source = prepared.source
    energy = source.live_state["energy"]
    if not torch.is_tensor(energy) or energy.ndim != 2:
        raise fail(
            RUNTIME_ABI_ERROR,
            "prepared.source.live_state.energy",
            "must be a rank-2 tensor",
        )
    batch_cells, slot_capacity = (int(value) for value in energy.shape)
    agent_id_to_slot = source.live_state.get("agent_id_to_slot")
    agent_id_capacity = (
        int(agent_id_to_slot.shape[1])
        if torch.is_tensor(agent_id_to_slot) and agent_id_to_slot.ndim == 2
        else None
    )
    event_capacity = (
        int(source.event_state_tensors[0].shape[1])
        if source.event_state_tensors
        and torch.is_tensor(source.event_state_tensors[0])
        and source.event_state_tensors[0].ndim >= 2
        else None
    )
    chemistry = source.full_tick_state_tensors[15]
    chemistry_width = (
        int(chemistry.shape[2])
        if torch.is_tensor(chemistry) and chemistry.ndim == 3
        else None
    )
    event_status_by_tick = source.event_state_tensors[29]
    output_tick_capacity = (
        int(event_status_by_tick.shape[1])
        if torch.is_tensor(event_status_by_tick)
        and event_status_by_tick.ndim == 2
        else None
    )
    patch = source.patch_p2_state_tensors[0]
    patch_height = int(patch.shape[1]) if torch.is_tensor(patch) and patch.ndim >= 3 else None
    patch_width = int(patch.shape[2]) if torch.is_tensor(patch) and patch.ndim >= 3 else None
    active_counts = _small_int_vector(
        source.active_count_state, path="prepared.source.active_count_state"
    )
    event_counts = _small_int_vector(
        source.event_state_tensors[len(EVENT_COLUMN_NAMES)],
        path="prepared.source.event_state_tensors.event_tape_active_count",
    )
    free_queue = source.live_state.get("slot_free_queue_count")
    free_queue_counts = (
        _small_int_vector(free_queue, path="prepared.source.live_state.slot_free_queue_count")
        if torch.is_tensor(free_queue)
        else None
    )
    pending_free = source.live_state.get("pending_free_count")
    pending_free_counts = (
        _small_int_vector(
            pending_free,
            path="prepared.source.live_state.pending_free_count",
        )
        if torch.is_tensor(pending_free)
        else None
    )
    residual = source.full_tick_state_tensors[26]
    residual_counts = (
        _small_int_vector(
            residual,
            path="prepared.source.full_tick_state_tensors.residual_count",
        )
        if torch.is_tensor(residual)
        and residual.ndim == 1
        and residual.dtype in {torch.int32, torch.int64}
        else None
    )
    noise_mask = source.full_tick_state_tensors[21]
    noise_valid_count = (
        int(noise_mask.detach().sum().item())
        if torch.is_tensor(noise_mask) and noise_mask.dtype == torch.bool
        else None
    )
    drive_width = (
        int(source.full_tick_state_tensors[6].shape[-1])
        if torch.is_tensor(source.full_tick_state_tensors[6])
        and source.full_tick_state_tensors[6].ndim == 3
        else None
    )
    return {
        "batch_cells": batch_cells,
        "slot_capacity": slot_capacity,
        "agent_id_capacity": agent_id_capacity,
        "event_capacity": event_capacity,
        "chemistry_width": chemistry_width,
        "drive_width": drive_width,
        "output_tick_capacity": output_tick_capacity,
        "patch_height": patch_height,
        "patch_width": patch_width,
        "ticks_per_generation": int(source.ticks_per_generation),
        "physical_generation_count": int(source.physical_window_generations),
        "physical_window_ticks": (
            int(source.ticks_per_generation) * int(source.physical_window_generations)
        ),
        "max_births_per_tick": _configuration_value(
            source.configuration, "max_births_per_tick", 0
        ),
        "active_count_by_cell": active_counts,
        "event_active_count_by_cell": event_counts,
        "slot_free_queue_count_by_cell": free_queue_counts,
        "pending_free_count_by_cell": pending_free_counts,
        "residual_count_by_cell": residual_counts,
        "drive_noise_valid_count": noise_valid_count,
    }


def _valid_count(
    *,
    path: str,
    tensor: torch.Tensor,
    mutability: str,
    shape_inputs: Mapping[str, Any],
) -> tuple[int, str]:
    capacity = int(tensor.numel())
    if mutability == "write_only":
        return 0, "pre_dispatch_output_empty"
    contract = RUNTIME_ABI_PATH_CONTRACTS_V1.get(path)
    if contract is None:
        raise fail(
            RUNTIME_ABI_ERROR,
            path,
            "has no generated valid-row contract",
        )
    formula = contract["valid_row_count_formula"]
    shape_formula = contract["shape_formula"].replace(" ", "")

    def total(name: str) -> int:
        values = shape_inputs.get(name)
        if not isinstance(values, list) or any(type(value) is not int for value in values):
            raise fail(
                RUNTIME_ABI_ERROR,
                path,
                f"requires exact {name}",
            )
        return sum(values)

    if formula == "product(shape)":
        valid = capacity
    elif formula == "count(value>=0)":
        valid = int((tensor.detach() >= 0).sum().item())
    elif formula == "count_below_sentinel(initial_agent_ids_by_row)":
        valid = int(
            (tensor.detach() < torch.iinfo(torch.int64).max).sum().item()
        )
    elif formula == "sum(active_count_by_cell)":
        valid = total("active_count_by_cell")
    elif formula == "sum(event_active_count_by_cell)":
        valid = total("event_active_count_by_cell")
    elif formula == "sum(slot_free_queue_count_by_cell)":
        valid = total("slot_free_queue_count_by_cell")
    elif formula == "sum(pending_free_count_by_cell)":
        valid = total("pending_free_count_by_cell")
    elif formula == "sum(residual_count_by_cell)":
        valid = total("residual_count_by_cell")
    elif formula == "count_true(drive_noise_samples_valid_by_tick_slot)":
        count = shape_inputs.get("drive_noise_valid_count")
        if type(count) is not int:
            raise fail(RUNTIME_ABI_ERROR, path, "requires the exact drive-noise mask")
        valid = count
    elif formula == "count_true(drive_noise_samples_valid_by_tick_slot)*D":
        count = shape_inputs.get("drive_noise_valid_count")
        width = shape_inputs.get("drive_width")
        if type(count) is not int or type(width) is not int:
            raise fail(
                RUNTIME_ABI_ERROR,
                path,
                "requires the exact drive-noise mask and width",
            )
        valid = count * width
    elif formula.replace(" ", "").startswith("sum(active_count_by_cell)*"):
        batch = shape_inputs["batch_cells"]
        slots = shape_inputs["slot_capacity"]
        denominator = batch * slots
        if denominator < 1 or capacity % denominator:
            raise fail(
                RUNTIME_ABI_ERROR,
                path,
                "active-prefix contract does not divide tensor capacity",
            )
        valid = total("active_count_by_cell") * (capacity // denominator)
    elif formula == "route_active_prefix_or_full_storage":
        if path.endswith("agent_id_to_slot"):
            valid = int((tensor.detach() >= 0).sum().item())
        elif shape_formula.startswith("[B,S"):
            trailing = math.prod(_shape(tensor)[2:]) if tensor.ndim > 2 else 1
            valid = total("active_count_by_cell") * trailing
        else:
            valid = capacity
    elif formula == "cell":
        valid = int(shape_inputs["batch_cells"])
    elif formula.isdigit():
        valid = int(formula)
    else:
        symbols = _shape_symbols({"shape_formula_inputs": shape_inputs})
        evaluated = _eval_dimension(formula.replace(" ", ""), symbols)
        if evaluated is None:
            raise fail(
                RUNTIME_ABI_ERROR,
                path,
                f"cannot evaluate generated valid-row formula: {formula}",
            )
        valid = evaluated
    if valid < 0 or valid > capacity:
        raise fail(
            RUNTIME_ABI_ERROR,
            path,
            f"valid-row count {valid} is outside capacity {capacity}",
        )
    return valid, formula


def _mapping_scalar_values_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical scalar subset of one ABI mapping argument."""

    projected: dict[str, Any] = {}
    for raw_name, child in sorted(value.items(), key=lambda item: str(item[0])):
        name = str(raw_name)
        if type(child) in {bool, int, float, str} or child is None:
            projected[name] = child
        elif isinstance(child, Mapping):
            nested = _mapping_scalar_values_v1(child)
            if nested:
                projected[name] = nested
        elif isinstance(child, (tuple, list)) and all(
            type(item) in {bool, int, float, str} or item is None for item in child
        ):
            projected[name] = list(child)
    return projected


def _mapping_scalar_vectors_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project bounded one-dimensional control vectors retained in an ABI map."""

    projected: dict[str, Any] = {}
    for raw_name, child in sorted(value.items(), key=lambda item: str(item[0])):
        if (
            not torch.is_tensor(child)
            or child.ndim != 1
            or child.numel() > 256
            or child.dtype not in {torch.float32, torch.float64}
            or not bool(torch.isfinite(child).all().item())
        ):
            continue
        values = [float(item) for item in child.detach().cpu().tolist()]
        projected[str(raw_name)] = {
            "dtype": str(child.dtype).removeprefix("torch."),
            "shape": _shape(child),
            "values": values,
            "values_sha256": canonical_sha256(values),
        }
    return projected


def observe_prepared_direct_resident_abi_v1(
    prepared: PreparedDirectResidentTrajectoryV1,
    *,
    phase: str = "pre_dispatch",
) -> dict[str, Any]:
    """Capture exact retained ABI metadata before or after native dispatch."""

    if phase not in {"pre_dispatch", "post_dispatch"}:
        raise fail(RUNTIME_ABI_ERROR, "phase", "must be pre_dispatch or post_dispatch")
    validate_prepared_direct_resident_trajectory_v1(prepared)
    arguments = tuple(prepared.arguments)
    if len(arguments) != len(DIRECT_RESIDENT_ARGUMENT_NAMES_V1):
        raise fail(
            RUNTIME_ABI_ERROR,
            "prepared.arguments",
            "argument count differs from the direct-resident ABI",
        )
    shape_inputs = _shape_inputs(prepared)
    aliases = _AliasTracker()
    tensor_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []

    def add_tensor(
        tensor: Any,
        *,
        path: str,
        outer_ordinal: int,
        mutability: str,
    ) -> str:
        if not torch.is_tensor(tensor):
            raise fail(RUNTIME_ABI_ERROR, path, "must be a torch tensor")
        valid_count, valid_semantics = _valid_count(
            path=path,
            tensor=tensor,
            mutability=mutability,
            shape_inputs=shape_inputs,
        )
        row = {
            "path": path,
            "outer_ordinal": outer_ordinal,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "rank": int(tensor.ndim),
            "shape": _shape(tensor),
            "shape_formula_inputs": dict(shape_inputs),
            "device": str(tensor.device),
            "mutability": mutability,
            "capacity_count": int(tensor.numel()),
            "valid_count": valid_count,
            "valid_count_semantics": valid_semantics,
            "requires_grad": bool(tensor.requires_grad),
            "is_contiguous": bool(tensor.is_contiguous()),
            "alias_identity": aliases.identity(tensor),
        }
        tensor_rows.append(row)
        return path

    for ordinal, (name, value) in enumerate(
        zip(DIRECT_RESIDENT_ARGUMENT_NAMES_V1, arguments, strict=True)
    ):
        nested = _NESTED_ARGUMENTS.get(ordinal)
        if ordinal == 68 and isinstance(value, Sequence):
            from .contracts.ordinary_turnover_v1 import (
                ordinary_birth_state_names_v1,
            )

            try:
                nested = (
                    ordinary_birth_state_names_v1(len(value)),
                    "read_write",
                )
            except ValueError as exc:
                raise fail(
                    RUNTIME_ABI_ERROR,
                    name,
                    "nested count must be 113, 122, or 139",
                ) from exc
        if (
            ordinal == 69
            and isinstance(value, Sequence)
            and arguments[_ORDINARY_BIRTHS_ENABLED_ARGUMENT_INDEX] is False
        ):
            nested = ((), "write_only")
        if (
            ordinal in {72, 73, 75}
            and isinstance(value, Sequence)
            and arguments[_GENERATION_TURNOVER_ENABLED_ARGUMENT_INDEX] is False
        ):
            mutability = "read_write" if ordinal == 72 else "write_only"
            nested = ((), mutability)
        if nested is not None:
            names, mutability = nested
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise fail(RUNTIME_ABI_ERROR, name, "must be a tensor sequence")
            if len(value) != len(names):
                raise fail(
                    RUNTIME_ABI_ERROR,
                    name,
                    f"nested count must be {len(names)}, got {len(value)}",
                )
            paths = [
                add_tensor(
                    tensor,
                    path=f"{name}.{nested_name}",
                    outer_ordinal=ordinal,
                    mutability=mutability,
                )
                for nested_name, tensor in zip(names, value, strict=True)
            ]
            outer_rows.append(
                {
                    "ordinal": ordinal,
                    "name": name,
                    "kind": "tensor_sequence",
                    "nested_count": len(paths),
                    "tensor_paths": paths,
                }
            )
        elif torch.is_tensor(value):
            path = add_tensor(
                value,
                path=name,
                outer_ordinal=ordinal,
                mutability=_outer_mutability(name),
            )
            outer_rows.append(
                {
                    "ordinal": ordinal,
                    "name": name,
                    "kind": "tensor",
                    "tensor_paths": [path],
                }
            )
        elif isinstance(value, Mapping):
            scalar_values = _mapping_scalar_values_v1(value)
            scalar_vectors = _mapping_scalar_vectors_v1(value)
            outer_rows.append(
                {
                    "ordinal": ordinal,
                    "name": name,
                    "kind": "mapping",
                    "keys": sorted(str(key) for key in value),
                    "scalar_values": scalar_values,
                    "scalar_values_sha256": canonical_sha256(scalar_values),
                    "scalar_vectors": scalar_vectors,
                    "scalar_vectors_sha256": canonical_sha256(scalar_vectors),
                }
            )
        elif type(value) in {bool, int, float}:
            outer_rows.append(
                {
                    "ordinal": ordinal,
                    "name": name,
                    "kind": "scalar",
                    "python_type": type(value).__name__,
                    "value": value,
                }
            )
        else:
            raise fail(
                RUNTIME_ABI_ERROR,
                name,
                f"unsupported outer argument type {type(value).__name__}",
            )

    payload: dict[str, Any] = {
        "schema_version": RUNTIME_ABI_RECEIPT_SCHEMA_V1,
        "phase": phase,
        "route": "patch_fixed_mlp_direct_resident_v1",
        "outer_argument_count": len(outer_rows),
        "outer_argument_names_sha256": canonical_sha256(
            list(DIRECT_RESIDENT_ARGUMENT_NAMES_V1)
        ),
        "shape_formula_inputs": shape_inputs,
        "outer_arguments": outer_rows,
        "tensor_count": len(tensor_rows),
        "tensor_metadata_sha256": canonical_sha256(tensor_rows),
        "tensors": tensor_rows,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def observe_applied_direct_resident_abi_v1(
    prepared: PreparedDirectResidentTrajectoryV1,
) -> dict[str, Any]:
    """Capture the retained direct-resident ABI after native execution settles."""

    return observe_prepared_direct_resident_abi_v1(
        prepared, phase="post_dispatch"
    )


def validate_runtime_abi_order_v1(
    receipt: Mapping[str, Any],
    abi_contract: Mapping[str, Any],
    *,
    require_production_admission: bool = False,
    prepared: PreparedDirectResidentTrajectoryV1 | None = None,
) -> dict[str, str]:
    """Bind one real prepared owner to every applicable exact ABI order."""

    if (
        receipt.get("schema_version") != RUNTIME_ABI_RECEIPT_SCHEMA_V1
        or receipt.get("phase") != "pre_dispatch"
    ):
        raise fail(RUNTIME_ABI_ERROR, "receipt", "is not a pre-dispatch ABI receipt")
    if require_production_admission:
        if prepared is None:
            raise fail(
                RUNTIME_ABI_ERROR,
                "prepared",
                "exact production metadata reconciliation requires the prepared owner",
            )
        from .contracts.full_tick_event_v1 import (
            reconcile_full_tick_event_receipt_v1,
        )
        from .contracts.ordinary_turnover_v1 import (
            reconcile_ordinary_turnover_contracts_v1,
        )

        reconcile_full_tick_event_receipt_v1(receipt, prepared=prepared)
        ordinary_count = len(prepared.source.ordinary_birth_state_tensors)
        turnover_count = len(prepared.source.turnover_state_tensors)
        ordinary_turnover = reconcile_ordinary_turnover_contracts_v1(
            receipt,
            ordinary_variant_count=ordinary_count,
            turnover_variant_count=turnover_count,
        )
        ambiguities = ordinary_turnover["ambiguities"]
        if ambiguities:
            first = ambiguities[0]
            raise fail(
                RUNTIME_ABI_ERROR,
                f"{first['family']}[{first['index']}].{first['name']}",
                "; ".join(str(reason) for reason in first["reasons"]),
            )
    outer = receipt.get("outer_arguments")
    tensors = receipt.get("tensors")
    variants = abi_contract.get("variants")
    if (
        not isinstance(outer, list)
        or not isinstance(tensors, list)
        or not isinstance(variants, list)
    ):
        raise fail(RUNTIME_ABI_ERROR, "receipt", "ABI rows are incomplete")
    observed: dict[str, list[str]] = {
        "direct_resident_outer": [str(row.get("name")) for row in outer],
    }
    prefix_to_family = {
        "full_tick_state_tensors": "full_tick_state",
        "event_state_tensors": "event_state",
        "ordinary_birth_state_tensors": "ordinary_birth_state",
        "ordinary_birth_output_tensors": "ordinary_birth_output",
        "ordinary_birth_scratch_tensors": "ordinary_birth_scratch",
        "turnover_state_tensors": "turnover_mutable_state",
        "turnover_output_tensors": "turnover_output",
        "patch_p2_state_tensors": "patch_p2_state",
    }
    for prefix, family in prefix_to_family.items():
        marker = f"{prefix}."
        observed[family] = [
            str(row["path"])[len(marker) :]
            for row in tensors
            if str(row.get("path", "")).startswith(marker)
        ]
    admitted: dict[str, str] = {}
    by_family_count = {
        (str(row.get("abi_family")), int(row.get("count", -1))): row
        for row in variants
        if isinstance(row, Mapping) and type(row.get("count")) is int
    }
    for family, names in observed.items():
        key = (family, len(names))
        variant = by_family_count.get(key)
        if variant is None:
            raise fail(
                RUNTIME_ABI_ERROR,
                family,
                f"no canonical variant has count {len(names)}",
            )
        expected = [str(row["name"]) for row in variant.get("entries", [])]
        if names != expected:
            mismatch = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(names, expected, strict=False)
                    )
                    if left != right
                ),
                min(len(names), len(expected)),
            )
            raise fail(
                RUNTIME_ABI_ERROR,
                family,
                f"order differs at index {mismatch}",
            )
        digest = canonical_sha256(names)
        if digest != variant.get("order_sha256"):
            raise fail(RUNTIME_ABI_ERROR, family, "order hash differs")
        if (
            require_production_admission
            and variant.get("support_status") != "supported_exact_metadata"
        ):
            raise fail(
                RUNTIME_ABI_ERROR,
                family,
                f"{variant['variant_id']} has exact order but unresolved "
                "authoritative metadata and is forbidden for production",
            )
        if require_production_admission:
            _validate_observed_variant_metadata_v1(
                family=family,
                variant=variant,
                receipt=receipt,
            )
        admitted[family] = str(variant["variant_id"])
    return admitted


def _shape_symbols(receipt: Mapping[str, Any]) -> dict[str, int]:
    inputs = receipt.get("shape_formula_inputs")
    if not isinstance(inputs, Mapping):
        raise fail(RUNTIME_ABI_ERROR, "receipt.shape_formula_inputs", "is absent")
    names = {
        "B": "batch_cells",
        "S": "slot_capacity",
        "I": "agent_id_capacity",
        "E": "event_capacity",
        "H": "patch_height",
        "W": "patch_width",
        "T": "ticks_per_generation",
        "O": "output_tick_capacity",
        "M": "max_births_per_tick",
        "D": "drive_width",
        "CHEM": "chemistry_width",
        "OUT": "output_tick_capacity",
    }
    return {
        symbol: int(inputs[source])
        for symbol, source in names.items()
        if type(inputs.get(source)) is int
    }


def _eval_dimension(expression: str, symbols: Mapping[str, int]) -> int | None:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise fail(
            RUNTIME_ABI_ERROR, "shape_formula", f"is malformed: {expression}"
        ) from exc

    def evaluate(node: ast.AST) -> int | None:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return int(node.value)
        if isinstance(node, ast.Name):
            return symbols.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult)
        ):
            left, right = evaluate(node.left), evaluate(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            return left * right
        raise fail(
            RUNTIME_ABI_ERROR,
            "shape_formula",
            f"uses a forbidden expression: {expression}",
        )

    return evaluate(tree)


def _evaluate_shape_formula(
    formula: object, symbols: Mapping[str, int]
) -> list[int | None] | None:
    if not isinstance(formula, str):
        raise fail(RUNTIME_ABI_ERROR, "shape_formula", "must be a string")
    value = formula.strip()
    if value == "[]":
        return []
    if not (value.startswith("[") and value.endswith("]")):
        return None
    body = value[1:-1]
    if not body:
        return []
    return [_eval_dimension(part, symbols) for part in body.split(",")]


def _admits_replay_backed_policy_post_rng_state_v1(
    *,
    family: str,
    index: int,
    dimension: int,
    expected: int | None,
    actual: object,
    actual_shape: list[Any],
    observed_rows: Sequence[Mapping[str, Any]],
) -> bool:
    if (
        family != "ordinary_birth_state"
        or index != 89
        or dimension != 2
        or expected != 2560
        or actual != 0
        or len(actual_shape) != 3
    ):
        return False
    delta_shape = None
    for row in observed_rows:
        path = str(row.get("path", ""))
        if path.endswith(".policy_birth_delta_tape"):
            delta_shape = row.get("shape")
            break
    if delta_shape is None and len(observed_rows) > 86:
        delta_shape = observed_rows[86].get("shape")
    if not isinstance(delta_shape, list) or len(delta_shape) != 3:
        return False
    return (
        actual_shape[0] == delta_shape[0]
        and actual_shape[1] == delta_shape[1]
        and delta_shape[2] == 1
    )


def _policy_delta_shape_for_replay_diagnostic_v1(
    observed_rows: Sequence[Mapping[str, Any]],
) -> object:
    for row in observed_rows:
        path = str(row.get("path", ""))
        if path.endswith(".policy_birth_delta_tape"):
            return row.get("shape")
    if len(observed_rows) > 86:
        return observed_rows[86].get("shape")
    return None


def _validate_observed_variant_metadata_v1(
    *,
    family: str,
    variant: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    entries = variant.get("entries")
    tensors = receipt.get("tensors")
    outer = receipt.get("outer_arguments")
    if not isinstance(entries, list) or not isinstance(tensors, list):
        raise fail(RUNTIME_ABI_ERROR, family, "metadata rows are absent")
    if any(
        not isinstance(entry, Mapping)
        or entry.get("metadata_status") != "supported_exact"
        for entry in entries
    ):
        raise fail(RUNTIME_ABI_ERROR, family, "contains a non-exact metadata row")
    aliases = {
        "stale_tail_policy": ("stale_tail_semantics", "stale_tail_contract"),
        "provider_participation": (
            "provider_participation",
            "provider_projection",
        ),
        "analysis_participation": (
            "analysis_participation",
            "analysis_projection",
        ),
    }
    direct_keys = (
        "semantic_owner",
        "dtype",
        "rank",
        "shape_formula",
        "valid_row_count_formula",
        "capacity_formula",
        "valid_range",
        "sentinels",
        "mutability",
        "checkpoint_participation",
        "chronicle_participation",
    )
    for index, entry in enumerate(entries):
        contract = entry.get("authority_contract")
        if not isinstance(contract, Mapping):
            raise fail(
                RUNTIME_ABI_ERROR,
                f"{family}[{index}]",
                "has no authority contract",
            )
        expected_physical_type = str(contract.get("kind", "tensor"))
        if entry.get("physical_type") != expected_physical_type:
            raise fail(
                RUNTIME_ABI_ERROR,
                f"{family}[{index}].physical_type",
                "differs from its generated authority contract",
            )
        logical_bit_width = entry.get("bit_width")
        physical_bit_width = contract.get("bit_width")
        if entry.get("dtype") == "bool":
            bit_width_matches = (
                logical_bit_width == 1 and physical_bit_width == 8
            )
        elif contract.get("kind") in {
            "ordered_tensor_container",
            "ordered_mapping",
        }:
            bit_width_matches = (
                logical_bit_width == "container"
                and physical_bit_width is None
            )
        else:
            bit_width_matches = logical_bit_width == physical_bit_width
        if not bit_width_matches:
            raise fail(
                RUNTIME_ABI_ERROR,
                f"{family}[{index}].bit_width",
                "differs from its generated logical/physical authority contract",
            )
        for key in direct_keys:
            if entry.get(key) != contract.get(key):
                raise fail(
                    RUNTIME_ABI_ERROR,
                    f"{family}[{index}].{key}",
                    "differs from its generated authority contract",
                )
        for key, candidates in aliases.items():
            expected = next(
                (contract[name] for name in candidates if name in contract),
                None,
            )
            if entry.get(key) != expected:
                raise fail(
                    RUNTIME_ABI_ERROR,
                    f"{family}[{index}].{key}",
                    "differs from its generated authority contract",
                )
    if family == "direct_resident_outer":
        if not isinstance(outer, list) or len(entries) != len(outer):
            raise fail(RUNTIME_ABI_ERROR, family, "outer metadata count differs")
        kind_map = {
            "ordered_tensor_container": "tensor_sequence",
            "ordered_mapping": "mapping",
        }
        for index, (entry, observed) in enumerate(
            zip(entries, outer, strict=True)
        ):
            contract = entry["authority_contract"]
            expected_kind = kind_map.get(
                str(contract.get("kind")), str(contract.get("kind"))
            )
            if observed.get("kind") != expected_kind:
                raise fail(
                    RUNTIME_ABI_ERROR,
                    f"{family}[{index}]",
                    f"kind differs: expected {expected_kind}",
                )
            ordered_count = contract.get("ordered_count")
            observed_nested_count = observed.get("nested_count")
            if (
                index == 68
                and contract.get("child_contract_family")
                == "ordinary_birth_state"
                and observed_nested_count in {113, 122, 139}
            ):
                # The outer ABI retains 160 arguments while this child
                # container selects one separately validated hereditary ABI.
                continue
            if (
                index == 69
                and contract.get("child_contract_family")
                == "ordinary_birth_output"
                and observed_nested_count
                == (
                    63
                    if outer[_ORDINARY_BIRTHS_ENABLED_ARGUMENT_INDEX].get("value")
                    is True
                    else 0
                )
            ):
                continue
            if (
                index in {72, 73, 75}
                and outer[_GENERATION_TURNOVER_ENABLED_ARGUMENT_INDEX].get(
                    "value"
                )
                is False
                and observed_nested_count == 0
            ):
                continue
            if (
                type(ordered_count) is int
                and observed_nested_count != ordered_count
            ):
                raise fail(
                    RUNTIME_ABI_ERROR,
                    f"{family}[{index}]",
                    "nested count differs",
                )
        return

    prefix = {
        "full_tick_state": "full_tick_state_tensors.",
        "event_state": "event_state_tensors.",
        "ordinary_birth_state": "ordinary_birth_state_tensors.",
        "ordinary_birth_output": "ordinary_birth_output_tensors.",
        "ordinary_birth_scratch": "ordinary_birth_scratch_tensors.",
        "turnover_mutable_state": "turnover_state_tensors.",
        "turnover_output": "turnover_output_tensors.",
        "patch_p2_state": "patch_p2_state_tensors.",
    }[family]
    observed_rows = [
        row
        for row in tensors
        if isinstance(row, Mapping)
        and str(row.get("path", "")).startswith(prefix)
    ]
    if len(observed_rows) != len(entries):
        raise fail(RUNTIME_ABI_ERROR, family, "tensor metadata count differs")
    symbols = _shape_symbols(receipt)
    for index, (entry, observed) in enumerate(
        zip(entries, observed_rows, strict=True)
    ):
        if observed.get("dtype") != entry.get("dtype"):
            raise fail(
                RUNTIME_ABI_ERROR,
                f"{family}[{index}].dtype",
                f"expected {entry.get('dtype')}, got {observed.get('dtype')}",
            )
        rank = entry.get("rank")
        if type(rank) is int and observed.get("rank") != rank:
            raise fail(
                RUNTIME_ABI_ERROR,
                f"{family}[{index}].rank",
                f"expected {rank}, got {observed.get('rank')}",
            )
        expected_shape = _evaluate_shape_formula(
            entry.get("shape_formula"), symbols
        )
        if expected_shape is not None:
            actual_shape = observed.get("shape")
            if not isinstance(actual_shape, list) or len(actual_shape) != len(
                expected_shape
            ):
                raise fail(
                    RUNTIME_ABI_ERROR,
                    f"{family}[{index}].shape",
                    "rank differs from symbolic shape",
                )
            for dimension, (expected, actual) in enumerate(
                zip(expected_shape, actual_shape, strict=True)
            ):
                if expected is not None and expected != actual:
                    if _admits_replay_backed_policy_post_rng_state_v1(
                        family=family,
                        index=index,
                        dimension=dimension,
                        expected=expected,
                        actual=actual,
                        actual_shape=actual_shape,
                        observed_rows=observed_rows,
                    ):
                        continue
                    raise fail(
                        RUNTIME_ABI_ERROR,
                        f"{family}[{index}].shape[{dimension}]",
                        (
                            f"expected {expected}, got {actual}; "
                            "replay-backed post-state requires compact "
                            "policy_birth_delta_tape shape, observed "
                            f"{_policy_delta_shape_for_replay_diagnostic_v1(observed_rows)}"
                        )
                        if (
                            family == "ordinary_birth_state"
                            and index == 89
                            and dimension == 2
                            and actual == 0
                        )
                        else f"expected {expected}, got {actual}",
                    )


__all__ = [
    "RUNTIME_ABI_ERROR",
    "RUNTIME_ABI_RECEIPT_SCHEMA_V1",
    "observe_prepared_direct_resident_abi_v1",
    "resolve_prepared_direct_resident_trajectory_v1",
    "validate_runtime_abi_order_v1",
]
