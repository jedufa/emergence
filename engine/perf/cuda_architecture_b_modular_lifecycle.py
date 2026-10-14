from __future__ import annotations

"""Deterministic resident prune-and-reuse for the Phase E modular envelope."""

from dataclasses import dataclass
from typing import Any, Final, Mapping, MutableMapping

import torch

from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_PROJECTION_PREFIX,
    build_modular_storage_layout,
)


MODULAR_LIFECYCLE_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_modular_lifecycle_v1"
)
MODULAR_LIFECYCLE_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_modular_lifecycle_receipt_v1"
)
SUPPORTED_PROFILE_LIMITS: Final[dict[str, tuple[int, int]]] = {
    "b2-12x32-v1": (12, 32),
    "b2-16x48-v1": (16, 48),
}
PHYSICAL_MODULE_SLOTS: Final[int] = 16
PHYSICAL_CONNECTION_SLOTS: Final[int] = 48

_MODULE_SLOT_FIELDS: Final[tuple[str, ...]] = (
    "module_occupied",
    "module_local_id_index",
    "module_instance_id_index",
    "module_homology_id_index",
    "module_type_index",
    "module_type_code",
    "identity_origin_code",
    "topology_position",
    "operator_code",
    "integrator_parameters",
    "gated_parameters",
    "module_dormant",
    "module_state_present",
    "module_state",
    "module_output_present",
    "module_previous_output",
    "input_binding_count",
    "input_semantic_code",
    "input_semantic_id_index",
    "input_gain",
    "input_enabled",
    "output_binding_count",
    "output_semantic_code",
    "output_semantic_id_index",
    "output_action_index",
    "output_gain",
    "output_enabled",
    "development_explicit",
    "development_stage_gains",
    "development_prerequisite_count",
    "development_prerequisite_homology_index",
    "development_prerequisite_module_slot",
    "development_latch_enabled",
    "development_latch_present",
    "development_latch_state",
    "development_effective_gain_present",
    "development_effective_gain",
)
_CONNECTION_SLOT_FIELDS: Final[tuple[str, ...]] = (
    "connection_occupied",
    "connection_local_id_index",
    "connection_src_module_slot",
    "connection_dst_module_slot",
    "connection_src_slot_index",
    "connection_dst_slot_index",
    "connection_value_type_index",
    "connection_channel_index",
    "connection_weight",
)
_COST_FIELDS: Final[tuple[str, ...]] = (
    "structural_cost_counts",
    "structural_cost_components",
    "structural_cost_total",
)
REPLACEMENT_FIELD_NAMES: Final[tuple[str, ...]] = (
    *_MODULE_SLOT_FIELDS,
    *_COST_FIELDS,
)


class ModularLifecycleError(RuntimeError):
    """The requested resident lifecycle operation is unsafe or malformed."""


@dataclass(frozen=True)
class ModularLifecycleReceipt:
    profile_id: str
    agent_capacity: int
    decision_count: int
    device: str
    cpu_validation_mode: bool
    source_mutated: bool = False
    agent_identity_preserved: bool = True
    owner_slot_binding_preserved: bool = True
    reused_same_physical_slot: bool = True
    pruned_state_output_cleared: bool = True
    pruned_connections_cleared: bool = True
    dependency_edges_cleared: bool = True
    rng_draws: int = 0
    device_to_host_synchronizations: int = 0
    reproduction_claimed: bool = False
    campaign_claimed: bool = False
    intelligence_or_emergence_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODULAR_LIFECYCLE_RECEIPT_SCHEMA_VERSION,
            **self.__dict__,
            "scope": "per_agent_resident_prune_and_same_slot_reuse_only",
        }


@dataclass(frozen=True)
class ModularLifecycleResult:
    planes: MutableMapping[str, torch.Tensor]
    audit_tensors: Mapping[str, torch.Tensor]
    receipt: ModularLifecycleReceipt


def _assert_tensor(condition: torch.Tensor, message: str) -> None:
    try:
        torch._assert_async(condition, message)
    except RuntimeError as exc:
        raise ModularLifecycleError(str(exc)) from exc


def _resolve_planes(
    planes: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(planes, Mapping):
        raise ModularLifecycleError("modular planes must be a mapping")
    logical_names = tuple(field.name for field in build_modular_storage_layout(1).fields)
    logical_set = set(logical_names)
    prefixed_set = {
        f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}" for name in logical_names
    }
    keys = {str(name) for name in planes}
    logical_present = keys & logical_set
    prefixed_present = keys & prefixed_set
    other_modular = {
        name
        for name in keys
        if name.startswith("phase_e_modular_") and name not in prefixed_set
    }
    if other_modular or (logical_present and prefixed_present):
        raise ModularLifecycleError("mixed modular plane namespaces are forbidden")
    if logical_present:
        if keys != logical_set:
            raise ModularLifecycleError("unprefixed modular plane set must be exact")
        return {name: planes[name] for name in logical_names}, "unprefixed"
    if prefixed_present:
        if keys != prefixed_set:
            raise ModularLifecycleError("prefixed modular plane set must be exact")
        return {
            name: planes[f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}"]
            for name in logical_names
        }, MODULAR_STORAGE_PROJECTION_PREFIX
    raise ModularLifecycleError("complete modular plane set is required")


def _matches_device(actual: torch.device, expected: torch.device) -> bool:
    return actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    )


def _validate_plane_contract(
    resolved: Mapping[str, torch.Tensor],
    *,
    profile_id: str,
    cpu_validation_mode: bool,
) -> tuple[int, torch.device, Mapping[str, Any]]:
    if profile_id not in SUPPORTED_PROFILE_LIMITS:
        raise ModularLifecycleError(f"unsupported modular profile={profile_id}")
    if type(cpu_validation_mode) is not bool:
        raise TypeError("cpu_validation_mode must be a bool")
    occupied = resolved.get("agent_occupied")
    if not isinstance(occupied, torch.Tensor) or occupied.ndim != 1:
        raise ModularLifecycleError("agent_occupied must be a rank-one tensor")
    capacity = int(occupied.shape[0])
    if capacity <= 0:
        raise ModularLifecycleError("agent capacity must be positive")
    layout = build_modular_storage_layout(capacity, profile_id)
    device = occupied.device
    if (device.type == "cpu") != cpu_validation_mode:
        raise ModularLifecycleError(
            "CPU tensors require explicit cpu_validation_mode and CUDA tensors forbid it"
        )
    fields = layout.fields_by_name
    for field in layout.fields:
        tensor = resolved[field.name]
        if (
            not isinstance(tensor, torch.Tensor)
            or not _matches_device(tensor.device, device)
            or tensor.dtype != field.torch_dtype
            or tuple(tensor.shape) != field.shape
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ModularLifecycleError(f"invalid modular lifecycle plane: {field.name}")
    return capacity, device, fields


def _validate_owner_bindings(
    resolved: Mapping[str, torch.Tensor],
    owner_bindings: Mapping[str, Any],
    *,
    capacity: int,
    device: torch.device,
) -> None:
    if not isinstance(owner_bindings, Mapping) or set(owner_bindings) != {
        "slot_to_agent_id",
        "slot_tokens",
        "alive_mask",
    }:
        raise ModularLifecycleError("owner binding tensor set must be exact")
    for name, dtype in (
        ("slot_to_agent_id", torch.int64),
        ("slot_tokens", torch.int64),
        ("alive_mask", torch.bool),
    ):
        value = owner_bindings[name]
        if (
            not isinstance(value, torch.Tensor)
            or not _matches_device(value.device, device)
            or value.dtype != dtype
            or tuple(value.shape) != (capacity,)
            or not value.is_contiguous()
            or value.requires_grad
        ):
            raise ModularLifecycleError(f"invalid owner binding tensor: {name}")
    occupied = resolved["agent_occupied"]
    _assert_tensor(
        torch.all(torch.eq(occupied, owner_bindings["alive_mask"])),
        "agent occupancy disagrees with owner alive mask",
    )
    _assert_tensor(
        torch.all(torch.eq(resolved["agent_id"], owner_bindings["slot_to_agent_id"])),
        "agent identity disagrees with owner slots",
    )
    _assert_tensor(
        torch.all(
            torch.logical_or(
                torch.logical_not(occupied),
                torch.eq(resolved["agent_slot_token"], owner_bindings["slot_tokens"]),
            )
        ),
        "agent slot token disagrees with owner slots",
    )


def _assert_unique_ids(
    values: torch.Tensor,
    occupied: torch.Tensor,
    *,
    message: str,
) -> None:
    paired = torch.logical_and(occupied.unsqueeze(2), occupied.unsqueeze(1))
    equal = torch.eq(values.unsqueeze(2), values.unsqueeze(1))
    diagonal = torch.eye(
        values.shape[1], dtype=torch.bool, device=values.device
    ).unsqueeze(0)
    duplicates = torch.logical_and(torch.logical_and(paired, equal), ~diagonal)
    _assert_tensor(torch.logical_not(torch.any(duplicates)), message)


def _validate_resident_graph(
    resolved: Mapping[str, torch.Tensor],
    *,
    logical_modules: int,
    logical_connections: int,
) -> None:
    agent_occupied = resolved["agent_occupied"]
    module_occupied = resolved["module_occupied"]
    connection_occupied = resolved["connection_occupied"]
    module_count = torch.sum(module_occupied.to(dtype=torch.int32), dim=1)
    connection_count = torch.sum(connection_occupied.to(dtype=torch.int32), dim=1)
    _assert_tensor(
        torch.all(torch.eq(module_count, resolved["module_count"])),
        "module count contradicts occupancy",
    )
    _assert_tensor(
        torch.all(torch.eq(connection_count, resolved["connection_count"])),
        "connection count contradicts occupancy",
    )
    _assert_tensor(
        torch.all(
            torch.logical_or(
                torch.logical_not(agent_occupied),
                torch.logical_and(
                    module_count <= logical_modules,
                    connection_count <= logical_connections,
                ),
            )
        ),
        "logical modular profile limit exceeded",
    )
    _assert_tensor(
        torch.logical_not(torch.any(module_occupied[:, logical_modules:])),
        "module occupancy is outside the logical profile",
    )
    _assert_tensor(
        torch.logical_not(torch.any(connection_occupied[:, logical_connections:])),
        "connection occupancy is outside the logical profile",
    )
    _assert_tensor(
        torch.all(torch.logical_or(~module_occupied, agent_occupied.unsqueeze(1))),
        "module occupancy requires an occupied agent",
    )
    src = resolved["connection_src_module_slot"].to(dtype=torch.int64)
    dst = resolved["connection_dst_module_slot"].to(dtype=torch.int64)
    endpoints_in_range = torch.logical_and(
        torch.logical_and(src >= 0, src < PHYSICAL_MODULE_SLOTS),
        torch.logical_and(dst >= 0, dst < PHYSICAL_MODULE_SLOTS),
    )
    src_present = module_occupied.gather(1, src.clamp(0, PHYSICAL_MODULE_SLOTS - 1))
    dst_present = module_occupied.gather(1, dst.clamp(0, PHYSICAL_MODULE_SLOTS - 1))
    _assert_tensor(
        torch.all(
            torch.logical_or(
                ~connection_occupied,
                torch.logical_and(endpoints_in_range, torch.logical_and(src_present, dst_present)),
            )
        ),
        "connection endpoint is missing or out of range",
    )
    for name in (
        "module_local_id_index",
        "module_instance_id_index",
        "module_homology_id_index",
    ):
        _assert_tensor(
            torch.all(torch.logical_or(~module_occupied, resolved[name] >= 0)),
            f"occupied module has out-of-range identity field={name}",
        )
        _assert_unique_ids(
            resolved[name], module_occupied, message=f"duplicate occupied module ID field={name}"
        )
    _assert_tensor(
        torch.all(
            torch.logical_or(
                ~connection_occupied,
                resolved["connection_local_id_index"] >= 0,
            )
        ),
        "occupied connection has out-of-range identity",
    )
    _assert_unique_ids(
        resolved["connection_local_id_index"],
        connection_occupied,
        message="duplicate occupied connection ID",
    )
    connection_order = resolved["connection_execution_order"].to(dtype=torch.int64)
    _assert_tensor(
        torch.all(
            torch.where(
                connection_occupied,
                torch.logical_and(
                    connection_order >= 0,
                    connection_order < PHYSICAL_CONNECTION_SLOTS,
                ),
                connection_order == -1,
            )
        ),
        "connection execution order occupancy is invalid",
    )
    rank_counts = torch.sum(
        torch.nn.functional.one_hot(
            connection_order.clamp(0, PHYSICAL_CONNECTION_SLOTS - 1),
            num_classes=PHYSICAL_CONNECTION_SLOTS,
        ).to(dtype=torch.int32)
        * connection_occupied.unsqueeze(2).to(dtype=torch.int32),
        dim=1,
    )
    expected_ranks = (
        torch.arange(
            PHYSICAL_CONNECTION_SLOTS,
            dtype=torch.int32,
            device=connection_occupied.device,
        ).reshape(1, -1)
        < connection_count.reshape(-1, 1)
    ).to(dtype=torch.int32)
    _assert_tensor(
        torch.all(torch.eq(rank_counts, expected_ranks)),
        "connection execution order must be a contiguous permutation",
    )
    _assert_tensor(
        torch.all(torch.logical_or(~agent_occupied, resolved["agent_id"] >= 0)),
        "occupied agent ID is out of range",
    )
    agent_pair = torch.logical_and(agent_occupied.unsqueeze(1), agent_occupied.unsqueeze(0))
    agent_equal = torch.eq(resolved["agent_id"].unsqueeze(1), resolved["agent_id"].unsqueeze(0))
    agent_diagonal = torch.eye(
        agent_occupied.shape[0], dtype=torch.bool, device=agent_occupied.device
    )
    _assert_tensor(
        torch.logical_not(
            torch.any(torch.logical_and(torch.logical_and(agent_pair, agent_equal), ~agent_diagonal))
        ),
        "duplicate occupied agent ID",
    )


def _validate_decisions(
    resolved: Mapping[str, torch.Tensor],
    replacement: Mapping[str, Any],
    agent_rows: torch.Tensor,
    module_slots: torch.Tensor,
    *,
    capacity: int,
    device: torch.device,
    fields: Mapping[str, Any],
    logical_modules: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        not isinstance(agent_rows, torch.Tensor)
        or not isinstance(module_slots, torch.Tensor)
        or not _matches_device(agent_rows.device, device)
        or not _matches_device(module_slots.device, device)
        or agent_rows.dtype != torch.int64
        or module_slots.dtype != torch.int64
        or agent_rows.ndim != 1
        or tuple(module_slots.shape) != tuple(agent_rows.shape)
        or not agent_rows.is_contiguous()
        or not module_slots.is_contiguous()
        or agent_rows.requires_grad
        or module_slots.requires_grad
    ):
        raise ModularLifecycleError("lifecycle decision tensors are malformed")
    decision_count = int(agent_rows.shape[0])
    if decision_count <= 0:
        raise ModularLifecycleError("at least one lifecycle decision is required")
    safe_rows = agent_rows.clamp(0, capacity - 1)
    safe_slots = module_slots.clamp(0, PHYSICAL_MODULE_SLOTS - 1)
    _assert_tensor(
        torch.all(torch.logical_and(agent_rows >= 0, agent_rows < capacity)),
        "lifecycle agent row is out of range",
    )
    _assert_tensor(
        torch.all(
            torch.logical_and(module_slots >= 0, module_slots < logical_modules)
        ),
        "lifecycle module slot is outside the logical profile",
    )
    row_counts = torch.bincount(safe_rows, minlength=capacity)
    _assert_tensor(torch.all(row_counts <= 1), "duplicate per-agent lifecycle decision")
    _assert_tensor(
        torch.all(resolved["agent_occupied"].gather(0, safe_rows)),
        "lifecycle decision targets a free agent row",
    )
    _assert_tensor(
        torch.all(resolved["module_occupied"][safe_rows, safe_slots]),
        "lifecycle decision targets an unoccupied module slot",
    )
    if not isinstance(replacement, Mapping) or set(replacement) != set(
        REPLACEMENT_FIELD_NAMES
    ):
        raise ModularLifecycleError("replacement plane set must be exact")
    for name in REPLACEMENT_FIELD_NAMES:
        value = replacement[name]
        field = fields[name]
        if name in _MODULE_SLOT_FIELDS:
            shape = (decision_count, *field.shape[2:])
        else:
            shape = (decision_count, *field.shape[1:])
        if (
            not isinstance(value, torch.Tensor)
            or not _matches_device(value.device, device)
            or value.dtype != field.torch_dtype
            or tuple(value.shape) != shape
            or not value.is_contiguous()
            or value.requires_grad
        ):
            raise ModularLifecycleError(f"invalid replacement plane: {name}")
    _assert_tensor(
        torch.all(replacement["module_occupied"]),
        "replacement module must be occupied",
    )
    for name in (
        "module_local_id_index",
        "module_instance_id_index",
        "module_homology_id_index",
    ):
        old = resolved[name][safe_rows, safe_slots]
        _assert_tensor(torch.all(replacement[name] >= 0), f"replacement {name} is out of range")
        _assert_tensor(
            torch.all(torch.ne(replacement[name], old)),
            f"replacement {name} must differ from the pruned identity",
        )
    _assert_tensor(
        torch.all(torch.logical_not(replacement["module_state_present"])),
        "replacement may not inherit pruned state",
    )
    _assert_tensor(
        torch.all(torch.eq(replacement["module_state"], 0.0)),
        "replacement module state must be neutral",
    )
    _assert_tensor(
        torch.all(torch.logical_not(replacement["module_output_present"])),
        "replacement may not inherit pruned output",
    )
    _assert_tensor(
        torch.all(torch.eq(replacement["module_previous_output"], 0.0)),
        "replacement module output must be neutral",
    )
    _assert_tensor(
        torch.all(replacement["development_prerequisite_count"] == 0),
        "first reuse primitive requires cleared replacement dependencies",
    )
    for name in (
        "development_latch_enabled",
        "development_latch_present",
        "development_latch_state",
        "development_effective_gain_present",
    ):
        _assert_tensor(
            torch.all(torch.logical_not(replacement[name])),
            f"replacement lifetime field must be cleared: {name}",
        )
    _assert_tensor(
        torch.all(replacement["development_effective_gain"] == 0.0),
        "replacement effective gain must be neutral",
    )
    _assert_tensor(
        torch.all(replacement["structural_cost_counts"] >= 0),
        "replacement cost counts must be nonnegative",
    )
    _assert_tensor(
        torch.all(
            torch.logical_and(
                torch.isfinite(replacement["structural_cost_components"]),
                replacement["structural_cost_components"] >= 0.0,
            )
        ),
        "replacement cost components must be finite nonnegative",
    )
    _assert_tensor(
        torch.all(
            torch.logical_and(
                torch.isfinite(replacement["structural_cost_total"]),
                torch.eq(
                    replacement["structural_cost_total"],
                    torch.sum(replacement["structural_cost_components"], dim=1),
                ),
            )
        ),
        "replacement cost total must equal finite component sum",
    )
    row_selected = torch.zeros((capacity,), dtype=torch.bool, device=device)
    row_selected.scatter_(0, safe_rows, True)
    selected_slot_by_agent = torch.full(
        (capacity,), -1, dtype=torch.int64, device=device
    )
    selected_slot_by_agent.scatter_(0, safe_rows, safe_slots)
    return safe_rows, safe_slots, row_selected, selected_slot_by_agent


def prune_and_reuse_modular_slots(
    planes: Mapping[str, Any],
    owner_bindings: Mapping[str, Any],
    *,
    profile_id: str,
    agent_rows: torch.Tensor,
    module_slots: torch.Tensor,
    replacement: Mapping[str, Any],
    cpu_validation_mode: bool = False,
) -> ModularLifecycleResult:
    """Prune and replace one supplied module slot per supplied agent row."""

    resolved, namespace = _resolve_planes(planes)
    capacity, device, fields = _validate_plane_contract(
        resolved,
        profile_id=profile_id,
        cpu_validation_mode=cpu_validation_mode,
    )
    logical_modules, logical_connections = SUPPORTED_PROFILE_LIMITS[profile_id]
    _validate_owner_bindings(
        resolved,
        owner_bindings,
        capacity=capacity,
        device=device,
    )
    _validate_resident_graph(
        resolved,
        logical_modules=logical_modules,
        logical_connections=logical_connections,
    )
    safe_rows, safe_slots, row_selected, selected_slot_by_agent = _validate_decisions(
        resolved,
        replacement,
        agent_rows,
        module_slots,
        capacity=capacity,
        device=device,
        fields=fields,
        logical_modules=logical_modules,
    )

    prerequisite_count = resolved["development_prerequisite_count"].to(dtype=torch.int64)
    positions = torch.arange(2, dtype=torch.int64, device=device).reshape(1, 1, 2)
    active_prerequisite = positions < prerequisite_count.unsqueeze(2)
    prerequisite_slots = resolved["development_prerequisite_module_slot"].to(
        dtype=torch.int64
    )
    targets_pruned = torch.eq(
        prerequisite_slots,
        selected_slot_by_agent.reshape(capacity, 1, 1),
    )
    other_modules = torch.logical_and(
        resolved["module_occupied"],
        torch.ne(
            torch.arange(PHYSICAL_MODULE_SLOTS, device=device).reshape(1, -1),
            selected_slot_by_agent.reshape(-1, 1),
        ),
    )
    active_blocker = torch.logical_and(
        row_selected.reshape(-1, 1, 1),
        torch.logical_and(
            other_modules.unsqueeze(2),
            torch.logical_and(active_prerequisite, targets_pruned),
        ),
    )
    _assert_tensor(
        torch.logical_not(torch.any(active_blocker)),
        "pruned module has active developmental dependencies",
    )

    result = {name: value.clone() for name, value in resolved.items()}
    for name in _MODULE_SLOT_FIELDS:
        result[name].index_put_((safe_rows, safe_slots), replacement[name])

    connection_occupied = resolved["connection_occupied"]
    selected = selected_slot_by_agent.reshape(-1, 1)
    incident = torch.logical_and(
        row_selected.reshape(-1, 1),
        torch.logical_and(
            connection_occupied,
            torch.logical_or(
                torch.eq(resolved["connection_src_module_slot"], selected),
                torch.eq(resolved["connection_dst_module_slot"], selected),
            ),
        ),
    )
    for name in _CONNECTION_SLOT_FIELDS:
        field = fields[name]
        mask = incident.reshape(
            capacity,
            PHYSICAL_CONNECTION_SLOTS,
            *([1] * (result[name].ndim - 2)),
        )
        fill = torch.full_like(result[name], field.fill_value)
        result[name] = torch.where(mask, fill, result[name])

    remaining_connections = result["connection_occupied"]
    result["connection_count"] = torch.where(
        row_selected,
        torch.sum(remaining_connections.to(dtype=torch.int32), dim=1),
        result["connection_count"],
    )
    old_order = resolved["connection_execution_order"].to(dtype=torch.int64)
    removed_order = torch.where(
        incident,
        old_order,
        torch.full_like(old_order, PHYSICAL_CONNECTION_SLOTS),
    )
    removed_before = torch.sum(
        (
            removed_order.unsqueeze(2)
            < old_order.unsqueeze(1)
        ).to(dtype=torch.int64),
        dim=1,
    )
    compact_order = torch.where(
        remaining_connections,
        old_order - removed_before,
        torch.full_like(old_order, -1),
    ).to(dtype=result["connection_execution_order"].dtype)
    result["connection_execution_order"] = torch.where(
        row_selected.reshape(-1, 1),
        compact_order,
        result["connection_execution_order"],
    )
    for name in _COST_FIELDS:
        result[name].index_copy_(0, safe_rows, replacement[name])

    _validate_resident_graph(
        result,
        logical_modules=logical_modules,
        logical_connections=logical_connections,
    )
    decision_count = int(agent_rows.shape[0])
    output_planes: MutableMapping[str, torch.Tensor]
    if namespace == "unprefixed":
        output_planes = result
    else:
        output_planes = {
            f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}": value
            for name, value in result.items()
        }
    return ModularLifecycleResult(
        planes=output_planes,
        audit_tensors={
            "selected_agent_rows": row_selected,
            "selected_module_slot_by_agent": selected_slot_by_agent,
            "cleared_connection_mask": incident,
        },
        receipt=ModularLifecycleReceipt(
            profile_id=profile_id,
            agent_capacity=capacity,
            decision_count=decision_count,
            device=str(device),
            cpu_validation_mode=cpu_validation_mode,
        ),
    )


__all__ = [
    "MODULAR_LIFECYCLE_RECEIPT_SCHEMA_VERSION",
    "MODULAR_LIFECYCLE_SCHEMA_VERSION",
    "ModularLifecycleError",
    "ModularLifecycleReceipt",
    "ModularLifecycleResult",
    "REPLACEMENT_FIELD_NAMES",
    "SUPPORTED_PROFILE_LIMITS",
    "prune_and_reuse_modular_slots",
]
