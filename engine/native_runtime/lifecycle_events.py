from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import operator
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from simulation.output_cost_timing import current_flat_output_cost_timing


SCHEMA_VERSION = "eiel_typed_lifecycle_event_v1"
_EVENT_CURSOR_ENVELOPE_SCHEMA_VERSION = "eiel_authoritative_event_cursor_envelope_v1"

SENTINEL_INT = -1


class LifecycleEventType(IntEnum):
    BIRTH_COMMITTED = 1
    MUTATION = 2
    BIRTH_FAILED = 3
    DEATH_COMMITTED = 4


FIELD_DTYPES: dict[str, torch.dtype] = {
    "event_type": torch.int32,
    "generation": torch.int64,
    "tick": torch.int64,
    "sequence": torch.int64,
    "actor_agent_id": torch.int64,
    "target_agent_id": torch.int64,
    "actor_slot": torch.int64,
    "target_slot": torch.int64,
    "parent_agent_id": torch.int64,
    "child_agent_id": torch.int64,
    "lineage_founder_id": torch.int64,
    "lineage_depth": torch.int32,
    "subtype_code": torch.int32,
    "reason_code": torch.int32,
    "success": torch.int8,
    "value": torch.float32,
}

_LEGACY_EVENT_CORE_FIELDS = (
    "event_tape_type",
    "event_tape_actor_id",
    "event_tape_target_id",
    "event_tape_value",
)
_LEGACY_EVENT_OPTIONAL_FIELDS = (
    "event_tape_generation",
    "event_tape_tick",
)
_EVENT_COUNT_METADATA_FIELDS = (
    "typed_lifecycle_event_active_count",
    "typed_lifecycle_event_next_sequence",
    "event_tape_active_count",
    "event_tape_cursor",
    "active_event_count",
    "event_cursor",
    "typed_event_rows",
)

_LIFECYCLE_OUTPUT_CONTRACT_CACHE: dict[str, dict[str, Any] | None] = {}

_BIRTH_FAILURE_REASON_CODES = {
    "birth_fail_no_space": 1,
    "birth_fail_buffer": 2,
    "birth_fail_no_free_slot": 3,
    "birth_fail_no_mate": 4,
    "birth_fail_conception": 5,
    "birth_fail_gestation": 6,
    "birth_fail_injury_gate": 7,
    "birth_fail_scar_gate": 8,
    "birth_fail_non_space": 9,
}

_COMMITTED_DEATH_EVENT_NAMES = {
    "committed_death",
    "death",
    "death_committed",
}

_MUTATION_EVENT_NAMES = {
    "mutation",
    "mutation_committed",
    "policy_mutation",
}

_DEATH_REASON_CODES = {
    "energy_depleted": 1,
    "energy_depletion": 1,
    "energy_exhausted": 1,
    "health_depleted": 2,
    "health_depletion": 2,
    "health_exhausted": 2,
}


def schema_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sentinel_int": SENTINEL_INT,
        "event_types": {
            item.name.lower(): int(item.value) for item in LifecycleEventType
        },
        "fields": {
            name: str(dtype).replace("torch.", "")
            for name, dtype in FIELD_DTYPES.items()
        },
        "ordering": ["generation", "tick", "sequence"],
        "active_prefix": "rows [0:event_tape_active_count) are authoritative",
    }


def empty_typed_lifecycle_event_tensors() -> dict[str, torch.Tensor]:
    return {
        name: torch.empty((0,), dtype=dtype) for name, dtype in FIELD_DTYPES.items()
    }


def _required_tensor(tensors: Mapping[str, Any], name: str) -> torch.Tensor:
    value = tensors.get(name)
    if not hasattr(value, "detach"):
        raise KeyError(f"missing lifecycle event tensor: {name}")
    return value.detach().to("cpu").contiguous()


def _event_metadata_count(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"typed lifecycle projection {name} must be an integer")
    try:
        resolved = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"typed lifecycle projection {name} must be an integer"
        ) from exc
    if resolved < 0:
        raise RuntimeError(f"typed lifecycle projection {name} must be nonnegative")
    return int(resolved)


def _event_column_length(tensors: Mapping[str, Any], name: str) -> int:
    tensor = _required_tensor(tensors, name)
    if tensor.ndim != 1:
        raise RuntimeError(f"typed lifecycle projection column {name} must be rank one")
    return int(tensor.shape[0])


def validated_event_storage_layout(
    projection: Mapping[str, Any],
    *,
    explicit_active_count: Any | None = None,
) -> dict[str, Any]:
    metadata_value = projection.get("metadata") or {}
    tensors_value = projection.get("tensors") or {}
    if not isinstance(metadata_value, Mapping) or not isinstance(
        tensors_value, Mapping
    ):
        raise TypeError(
            "typed lifecycle projection metadata and tensors must be mappings"
        )
    metadata = dict(metadata_value)
    tensors = dict(tensors_value)
    prefix = "typed_lifecycle_event_"
    typed_names = tuple(f"{prefix}{name}" for name in FIELD_DTYPES)
    typed_present = tuple(name for name in typed_names if name in tensors)
    legacy_present = tuple(
        name
        for name in (*_LEGACY_EVENT_CORE_FIELDS, *_LEGACY_EVENT_OPTIONAL_FIELDS)
        if name in tensors
    )
    if typed_present and len(typed_present) != len(typed_names):
        missing = sorted(set(typed_names) - set(typed_present))
        raise RuntimeError(
            f"typed lifecycle projection is missing mandatory columns: {missing}"
        )
    legacy_core_present = tuple(
        name for name in _LEGACY_EVENT_CORE_FIELDS if name in tensors
    )
    if legacy_present and len(legacy_core_present) != len(_LEGACY_EVENT_CORE_FIELDS):
        missing = sorted(set(_LEGACY_EVENT_CORE_FIELDS) - set(legacy_core_present))
        raise RuntimeError(
            f"legacy lifecycle projection is missing mandatory columns: {missing}"
        )

    represented_columns = (*typed_present, *legacy_present)
    lengths = {
        name: _event_column_length(tensors, name) for name in represented_columns
    }
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        raise RuntimeError(
            f"typed lifecycle projection columns have inconsistent lengths: {lengths}"
        )
    capacity = next(iter(unique_lengths), 0)

    external_history = metadata.get("event_history_storage_mode") == (
        "eiel_resident_event_sink_v1"
    )
    declaration_fields = tuple(
        name
        for name in _EVENT_COUNT_METADATA_FIELDS
        if not (external_history and name == "typed_lifecycle_event_next_sequence")
    )
    declarations = {
        name: _event_metadata_count(metadata[name], name=name)
        for name in declaration_fields
        if name in metadata
    }
    if explicit_active_count is not None:
        declarations["explicit_active_count"] = _event_metadata_count(
            explicit_active_count,
            name="explicit_active_count",
        )
    declared_values = set(declarations.values())
    if len(declared_values) > 1:
        raise RuntimeError(
            f"typed lifecycle projection cursor contradiction: {declarations}"
        )
    active_count = next(iter(declared_values), capacity)
    if active_count > capacity:
        raise RuntimeError(
            "typed lifecycle projection active count exceeds column capacity: "
            f"active_count={active_count} capacity={capacity}"
        )
    if not represented_columns and active_count != 0:
        raise RuntimeError(
            "typed lifecycle projection declares event rows without mandatory columns"
        )
    if "event_tape_inactive_tail_count" in metadata:
        inactive_tail = _event_metadata_count(
            metadata["event_tape_inactive_tail_count"],
            name="event_tape_inactive_tail_count",
        )
        if inactive_tail != capacity - active_count:
            raise RuntimeError(
                "typed lifecycle projection inactive-tail contradiction: "
                f"inactive_tail={inactive_tail} capacity={capacity} active_count={active_count}"
            )
    if external_history:
        sequence_base = _event_metadata_count(
            metadata.get("typed_lifecycle_event_sequence_base", 0),
            name="typed_lifecycle_event_sequence_base",
        )
        next_sequence = _event_metadata_count(
            metadata.get("typed_lifecycle_event_next_sequence"),
            name="typed_lifecycle_event_next_sequence",
        )
        frontier = _event_metadata_count(
            metadata.get("event_history_frontier"),
            name="event_history_frontier",
        )
        if next_sequence != sequence_base + active_count or frontier != next_sequence:
            raise RuntimeError(
                "external lifecycle event frontier contradicts pending rows: "
                f"base={sequence_base} active={active_count} "
                f"next={next_sequence} frontier={frontier}"
            )
    else:
        sequence_base = 0
        next_sequence = active_count
    return {
        "representation": "typed"
        if typed_present
        else "legacy"
        if legacy_present
        else "empty",
        "active_count": int(active_count),
        "capacity": int(capacity),
        "declared_counts": declarations,
        "column_lengths": lengths,
        "sequence_base": int(sequence_base),
        "next_sequence": int(next_sequence),
        "external_history": external_history,
    }


def _validated_typed_event_tensors(
    events: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, torch.Tensor], int]:
    tensors = {name: _required_tensor(events, name) for name in FIELD_DTYPES}
    lengths = {
        name: int(tensor.shape[0])
        for name, tensor in tensors.items()
        if tensor.ndim == 1
    }
    if len(lengths) != len(FIELD_DTYPES):
        invalid = sorted(name for name, tensor in tensors.items() if tensor.ndim != 1)
        raise RuntimeError(
            f"{label} typed lifecycle columns must be rank one: {invalid}"
        )
    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            f"{label} typed lifecycle columns have inconsistent lengths: {lengths}"
        )
    return tensors, next(iter(lengths.values()), 0)


def _sentinel_tensor(n: int, *, dtype: torch.dtype) -> torch.Tensor:
    if dtype.is_floating_point:
        return torch.zeros((int(n),), dtype=dtype)
    return torch.full((int(n),), SENTINEL_INT, dtype=dtype)


def _safe_int(value: Any, default: int = SENTINEL_INT) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _lineage_founder_int(value: Any, default: int = SENTINEL_INT) -> int:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized[:1].lower() == "l" and normalized[1:].isdigit():
            return int(normalized[1:])
    return _safe_int(value, default)


def _first_present(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _is_committed_death_context(row: Mapping[str, Any]) -> bool:
    raw = _first_present(row, "event_type", "event", default="")
    if isinstance(raw, (int, IntEnum)):
        return int(raw) == int(LifecycleEventType.DEATH_COMMITTED)
    return str(raw or "").strip().lower() in _COMMITTED_DEATH_EVENT_NAMES


def _reference_context_sort_key(
    item: tuple[int, Mapping[str, Any]],
) -> tuple[int, int, int, int, int]:
    index, row = item
    generation = _safe_int(row.get("generation", 0), 0)
    tick = _safe_int(row.get("tick", 0), 0)
    event_phase = 0 if _is_committed_death_context(row) else 1
    order = _safe_int(_first_present(row, "sequence", "order", default=index), index)
    scan_id = str(row.get("scan_id", "") or "")
    if "sequence" not in row and "order" not in row and scan_id:
        order = _safe_int(scan_id.rsplit(":", 1)[-1], index)
    return generation, tick, event_phase, order, index


def _death_reason_and_value(row: Mapping[str, Any]) -> tuple[int, float]:
    raw_code = _first_present(row, "reason_code", "death_reason_code")
    raw_value = _first_present(row, "value", "death_value")
    if raw_code is None and raw_value is not None:
        value = float(raw_value)
        if math.isfinite(value) and value.is_integer():
            raw_code = int(value)
    if raw_code is None:
        reason_name = (
            str(_first_present(row, "death_reason", "reason", default="") or "")
            .strip()
            .lower()
        )
        raw_code = _DEATH_REASON_CODES.get(reason_name)
    if raw_code is None:
        raise ValueError(
            "committed death context requires an authoritative reason_code/value"
        )
    reason_code = int(raw_code)
    if reason_code not in set(_DEATH_REASON_CODES.values()):
        raise ValueError(f"unsupported committed death reason_code: {reason_code}")
    scalar = float(reason_code if raw_value is None else raw_value)
    if not math.isfinite(scalar) or scalar != float(reason_code):
        raise ValueError("committed death reason_code and value must match exactly")
    return reason_code, scalar


def typed_lifecycle_events_from_legacy_tape(
    projection: Mapping[str, Any],
    *,
    active_count: int | None = None,
) -> dict[str, torch.Tensor]:
    metadata = dict(projection.get("metadata") or {})
    tensors = dict(projection.get("tensors") or {})
    layout = validated_event_storage_layout(
        projection,
        explicit_active_count=active_count,
    )
    if layout["representation"] != "legacy":
        raise RuntimeError(
            "legacy lifecycle projection requires the complete legacy event tensor set"
        )
    n = int(layout["active_count"])
    event_type = _required_tensor(tensors, "event_tape_type").to(dtype=torch.int32)
    actor = _required_tensor(tensors, "event_tape_actor_id").to(dtype=torch.int64)
    target = _required_tensor(tensors, "event_tape_target_id").to(dtype=torch.int64)
    value = _required_tensor(tensors, "event_tape_value").to(dtype=torch.float32)
    tape_generation = tensors.get("event_tape_generation")
    tape_tick = tensors.get("event_tape_tick")
    if tape_generation is not None and hasattr(tape_generation, "detach"):
        tape_generation = (
            tape_generation.detach().to("cpu", dtype=torch.int64).contiguous()
        )
    else:
        tape_generation = None
    if tape_tick is not None and hasattr(tape_tick, "detach"):
        tape_tick = tape_tick.detach().to("cpu", dtype=torch.int64).contiguous()
    else:
        tape_tick = None
    event_type = event_type[:n].contiguous()
    actor = actor[:n].contiguous()
    target = target[:n].contiguous()
    value = value[:n].contiguous()

    out = empty_typed_lifecycle_event_tensors()
    out["event_type"] = event_type
    out["generation"] = (
        tape_generation[:n].contiguous()
        if tape_generation is not None
        else torch.full(
            (n,), int(metadata.get("generation", 0) or 0), dtype=torch.int64
        )
    )
    out["tick"] = (
        tape_tick[:n].contiguous()
        if tape_tick is not None
        else torch.full((n,), int(metadata.get("tick", 0) or 0), dtype=torch.int64)
    )
    out["sequence"] = torch.arange(n, dtype=torch.int64)
    out["actor_agent_id"] = actor
    out["target_agent_id"] = target
    out["actor_slot"] = _sentinel_tensor(n, dtype=torch.int64)
    out["target_slot"] = _sentinel_tensor(n, dtype=torch.int64)
    out["parent_agent_id"] = torch.where(
        event_type == int(LifecycleEventType.BIRTH_COMMITTED),
        actor,
        _sentinel_tensor(n, dtype=torch.int64),
    )
    out["child_agent_id"] = torch.where(
        (event_type == int(LifecycleEventType.BIRTH_COMMITTED))
        | (event_type == int(LifecycleEventType.MUTATION)),
        target,
        _sentinel_tensor(n, dtype=torch.int64),
    )
    out["lineage_founder_id"] = _sentinel_tensor(n, dtype=torch.int64)
    out["lineage_depth"] = _sentinel_tensor(n, dtype=torch.int32)
    out["subtype_code"] = event_type.clone().to(dtype=torch.int32)
    out["reason_code"] = torch.where(
        event_type == int(LifecycleEventType.BIRTH_FAILED),
        value.to(dtype=torch.int32),
        _sentinel_tensor(n, dtype=torch.int32),
    )
    out["success"] = torch.where(
        event_type == int(LifecycleEventType.BIRTH_FAILED),
        torch.zeros((n,), dtype=torch.int8),
        torch.ones((n,), dtype=torch.int8),
    )
    out["value"] = value
    return out


def typed_lifecycle_events_from_reference_contexts(
    contexts: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, torch.Tensor]:
    rows = [
        dict(row)
        for _index, row in sorted(
            enumerate(contexts),
            key=_reference_context_sort_key,
        )
    ]
    n = len(rows)
    out = empty_typed_lifecycle_event_tensors()
    if n == 0:
        return out

    event_type = []
    generation = []
    tick = []
    sequence = []
    actor = []
    target = []
    actor_slot = []
    target_slot = []
    parent = []
    child = []
    founder = []
    lineage_depth = []
    subtype = []
    reason = []
    success = []
    value = []
    for index, row in enumerate(rows):
        raw_event = row.get("event_type", row.get("event", ""))
        event_name = str(raw_event or "").strip().lower()
        is_death = _is_committed_death_context(row)
        is_mutation = (
            isinstance(raw_event, (int, IntEnum))
            and int(raw_event) == int(LifecycleEventType.MUTATION)
        ) or event_name in _MUTATION_EVENT_NAMES
        is_success = (
            event_name in {"birth_success", "successful_birth"}
            or int(row.get("birth_success", 0) or 0) > 0
        )
        parent_id = _safe_int(_first_present(row, "parent_id", default=SENTINEL_INT))
        child_id = _safe_int(_first_present(row, "child_id", default=SENTINEL_INT))
        if is_death:
            if "generation" not in row or "tick" not in row:
                raise ValueError("committed death context requires generation and tick")
            generation_id = _safe_int(row.get("generation"))
            tick_id = _safe_int(row.get("tick"))
            if generation_id < 0 or tick_id < 0:
                raise ValueError(
                    "committed death generation and tick must be non-negative"
                )
            actor_id = _safe_int(
                _first_present(
                    row, "actor_agent_id", "agent_id", "actor_id", default=SENTINEL_INT
                )
            )
            if actor_id < 0:
                raise ValueError(
                    "committed death context requires actor agent identity"
                )
            target_id = _safe_int(
                _first_present(
                    row, "target_agent_id", "target_id", default=SENTINEL_INT
                )
            )
            if target_id != SENTINEL_INT:
                raise ValueError("committed death target_agent_id must be -1")
            reason_code, scalar = _death_reason_and_value(row)
            kind = int(LifecycleEventType.DEATH_COMMITTED)
            ok = 1
            parent_id = SENTINEL_INT
            child_id = SENTINEL_INT
            actor_slot_id = _safe_int(
                _first_present(row, "actor_slot", "slot", default=SENTINEL_INT)
            )
            target_slot_id = SENTINEL_INT
            founder_id = _lineage_founder_int(
                _first_present(
                    row,
                    "lineage_founder_id",
                    "founder_id",
                    "lineage_id",
                    default=SENTINEL_INT,
                )
            )
            lineage_depth_id = _safe_int(
                _first_present(row, "lineage_depth", default=SENTINEL_INT)
            )
        elif is_mutation:
            kind = int(LifecycleEventType.MUTATION)
            reason_code = SENTINEL_INT
            ok = 1
            child_id = _safe_int(
                _first_present(
                    row, "child_id", "child_agent_id", "agent_id", default=SENTINEL_INT
                )
            )
            actor_id = _safe_int(
                _first_present(
                    row, "actor_agent_id", "agent_id", "child_id", default=child_id
                ),
                child_id,
            )
            target_id = _safe_int(
                _first_present(
                    row, "target_agent_id", "child_id", "agent_id", default=child_id
                ),
                child_id,
            )
            parent_id = _safe_int(
                _first_present(
                    row, "parent_agent_id", "parent_id", default=SENTINEL_INT
                )
            )
            scalar = float(
                _first_present(
                    row, "value", "mutation_delta", "delta_total", default=0.0
                )
                or 0.0
            )
            generation_id = int(row.get("generation", 0) or 0)
            tick_id = int(row.get("tick", 0) or 0)
            actor_slot_id = _safe_int(
                _first_present(
                    row, "actor_slot", "child_slot", "slot", default=SENTINEL_INT
                )
            )
            target_slot_id = _safe_int(
                _first_present(
                    row, "target_slot", "child_slot", "slot", default=actor_slot_id
                ),
                actor_slot_id,
            )
            founder_id = _lineage_founder_int(
                _first_present(
                    row,
                    "lineage_founder_id",
                    "founder_id",
                    "lineage_id",
                    default=SENTINEL_INT,
                )
            )
            lineage_depth_id = _safe_int(
                _first_present(row, "lineage_depth", default=SENTINEL_INT)
            )
        elif is_success:
            kind = int(LifecycleEventType.BIRTH_COMMITTED)
            reason_code = SENTINEL_INT
            ok = 1
            target_id = child_id
            scalar = float(row.get("value", row.get("child_energy", 0.0)) or 0.0)
            actor_id = parent_id
            generation_id = int(row.get("generation", 0) or 0)
            tick_id = int(row.get("tick", 0) or 0)
            actor_slot_id = _safe_int(
                _first_present(row, "actor_slot", "parent_slot", default=SENTINEL_INT)
            )
            target_slot_id = _safe_int(
                _first_present(row, "target_slot", "child_slot", default=SENTINEL_INT)
            )
            founder_id = _lineage_founder_int(
                _first_present(
                    row,
                    "lineage_founder_id",
                    "founder_id",
                    "lineage_id",
                    default=SENTINEL_INT,
                )
            )
            lineage_depth_id = _safe_int(
                _first_present(row, "lineage_depth", default=SENTINEL_INT)
            )
        else:
            kind = int(LifecycleEventType.BIRTH_FAILED)
            failure = str(row.get("failure_reason", event_name) or "")
            reason_code = int(
                _BIRTH_FAILURE_REASON_CODES.get(
                    failure, _BIRTH_FAILURE_REASON_CODES["birth_fail_non_space"]
                )
            )
            ok = 0
            target_id = SENTINEL_INT
            scalar = float(reason_code)
            actor_id = parent_id
            generation_id = int(row.get("generation", 0) or 0)
            tick_id = int(row.get("tick", 0) or 0)
            actor_slot_id = _safe_int(
                _first_present(row, "actor_slot", "parent_slot", default=SENTINEL_INT)
            )
            target_slot_id = SENTINEL_INT
            founder_id = _lineage_founder_int(
                _first_present(
                    row,
                    "lineage_founder_id",
                    "founder_id",
                    "lineage_id",
                    default=SENTINEL_INT,
                )
            )
            lineage_depth_id = _safe_int(
                _first_present(row, "lineage_depth", default=SENTINEL_INT)
            )
        event_type.append(kind)
        generation.append(generation_id)
        tick.append(tick_id)
        sequence.append(index)
        actor.append(actor_id)
        target.append(target_id)
        actor_slot.append(actor_slot_id)
        target_slot.append(target_slot_id)
        parent.append(parent_id if (is_success or is_mutation) else SENTINEL_INT)
        child.append(child_id if (is_success or is_mutation) else SENTINEL_INT)
        founder.append(founder_id)
        lineage_depth.append(lineage_depth_id)
        subtype.append(kind)
        reason.append(reason_code)
        success.append(ok)
        value.append(scalar)

    out["event_type"] = torch.tensor(event_type, dtype=torch.int32)
    out["generation"] = torch.tensor(generation, dtype=torch.int64)
    out["tick"] = torch.tensor(tick, dtype=torch.int64)
    out["sequence"] = torch.tensor(sequence, dtype=torch.int64)
    out["actor_agent_id"] = torch.tensor(actor, dtype=torch.int64)
    out["target_agent_id"] = torch.tensor(target, dtype=torch.int64)
    out["actor_slot"] = torch.tensor(actor_slot, dtype=torch.int64)
    out["target_slot"] = torch.tensor(target_slot, dtype=torch.int64)
    out["parent_agent_id"] = torch.tensor(parent, dtype=torch.int64)
    out["child_agent_id"] = torch.tensor(child, dtype=torch.int64)
    out["lineage_founder_id"] = torch.tensor(founder, dtype=torch.int64)
    out["lineage_depth"] = torch.tensor(lineage_depth, dtype=torch.int32)
    out["subtype_code"] = torch.tensor(subtype, dtype=torch.int32)
    out["reason_code"] = torch.tensor(reason, dtype=torch.int32)
    out["success"] = torch.tensor(success, dtype=torch.int8)
    out["value"] = torch.tensor(value, dtype=torch.float32)
    return out


def complete_reference_typed_lifecycle_events(
    events: Mapping[str, Any],
    identities: Mapping[int, Mapping[str, Any]],
    *,
    require_complete: bool,
) -> dict[str, torch.Tensor]:
    """Restore complete reference rows after the legacy compatibility projection."""

    out = {
        name: _required_tensor(events, name).to(dtype=dtype).clone()
        for name, dtype in FIELD_DTYPES.items()
    }
    lengths = {int(tensor.shape[0]) for tensor in out.values()}
    if len(lengths) != 1:
        raise RuntimeError("reference typed lifecycle fields have inconsistent lengths")
    rows = next(iter(lengths), 0)
    if not torch.equal(out["sequence"], torch.arange(rows, dtype=torch.int64)):
        raise RuntimeError("reference typed lifecycle sequence is not contiguous")

    def identity(
        agent_id: int, *, row_index: int, role: str
    ) -> Mapping[str, Any] | None:
        resolved = identities.get(int(agent_id))
        if resolved is None and require_complete:
            raise RuntimeError(
                "reference typed lifecycle identity is unavailable: "
                f"row={row_index} role={role} agent_id={agent_id}"
            )
        return resolved

    def identity_int(
        resolved: Mapping[str, Any] | None,
        field: str,
        *,
        row_index: int,
        role: str,
    ) -> int:
        value = _safe_int(resolved.get(field) if resolved is not None else None)
        if value < 0 and require_complete:
            raise RuntimeError(
                "reference typed lifecycle identity field is unavailable: "
                f"row={row_index} role={role} field={field}"
            )
        return value

    for index in range(rows):
        kind = int(out["event_type"][index].item())
        actor_id = int(out["actor_agent_id"][index].item())
        target_id = int(out["target_agent_id"][index].item())
        if kind in (
            int(LifecycleEventType.BIRTH_COMMITTED),
            int(LifecycleEventType.MUTATION),
        ):
            subject_id = target_id
            subject_role = "child"
        elif kind in (
            int(LifecycleEventType.BIRTH_FAILED),
            int(LifecycleEventType.DEATH_COMMITTED),
        ):
            subject_id = actor_id
            subject_role = "actor"
        else:
            raise RuntimeError(
                f"unsupported reference typed lifecycle event_type: {kind}"
            )

        actor_identity: Mapping[str, Any] | None = None
        subject_identity: Mapping[str, Any] | None = None
        if int(out["actor_slot"][index].item()) < 0:
            actor_identity = identity(actor_id, row_index=index, role="actor")
            out["actor_slot"][index] = identity_int(
                actor_identity,
                "slot",
                row_index=index,
                role="actor",
            )
        if (
            kind
            in (
                int(LifecycleEventType.BIRTH_COMMITTED),
                int(LifecycleEventType.MUTATION),
            )
            and int(out["target_slot"][index].item()) < 0
        ):
            subject_identity = identity(subject_id, row_index=index, role=subject_role)
            out["target_slot"][index] = identity_int(
                subject_identity,
                "slot",
                row_index=index,
                role=subject_role,
            )
        if int(out["lineage_founder_id"][index].item()) < 0:
            if subject_identity is None:
                subject_identity = identity(
                    subject_id, row_index=index, role=subject_role
                )
            out["lineage_founder_id"][index] = identity_int(
                subject_identity,
                "lineage_founder_id",
                row_index=index,
                role=subject_role,
            )
        if int(out["lineage_depth"][index].item()) < 0:
            if subject_identity is None:
                subject_identity = identity(
                    subject_id, row_index=index, role=subject_role
                )
            out["lineage_depth"][index] = identity_int(
                subject_identity,
                "lineage_depth",
                row_index=index,
                role=subject_role,
            )

        if kind == int(LifecycleEventType.MUTATION):
            if int(out["parent_agent_id"][index].item()) < 0:
                if subject_identity is None:
                    subject_identity = identity(
                        subject_id, row_index=index, role=subject_role
                    )
                out["parent_agent_id"][index] = identity_int(
                    subject_identity,
                    "parent_agent_id",
                    row_index=index,
                    role=subject_role,
                )
            out["child_agent_id"][index] = subject_id
        if (
            kind == int(LifecycleEventType.DEATH_COMMITTED)
            and int(out["reason_code"][index].item()) < 0
        ):
            scalar = float(out["value"][index].item())
            reason_code = (
                int(scalar)
                if math.isfinite(scalar) and scalar.is_integer()
                else SENTINEL_INT
            )
            if reason_code not in set(_DEATH_REASON_CODES.values()):
                raise RuntimeError(
                    "reference committed death has no authoritative reason: "
                    f"row={index} value={scalar}"
                )
            out["reason_code"][index] = reason_code

        if require_complete:
            required = {
                "actor_slot": int(out["actor_slot"][index].item()),
                "lineage_founder_id": int(out["lineage_founder_id"][index].item()),
                "lineage_depth": int(out["lineage_depth"][index].item()),
            }
            if kind in (
                int(LifecycleEventType.BIRTH_COMMITTED),
                int(LifecycleEventType.MUTATION),
            ):
                required["target_slot"] = int(out["target_slot"][index].item())
            if kind in (
                int(LifecycleEventType.BIRTH_FAILED),
                int(LifecycleEventType.DEATH_COMMITTED),
            ):
                required["reason_code"] = int(out["reason_code"][index].item())
            missing = sorted(name for name, value in required.items() if value < 0)
            if missing:
                raise RuntimeError(
                    "reference typed lifecycle row remains incomplete: "
                    f"row={index} fields={missing}"
                )
    return out


def concatenate_typed_lifecycle_events(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    normalized: list[dict[str, torch.Tensor]] = []
    counts: list[int] = []
    for label, events in (("first", first), ("second", second)):
        local = {
            name: _required_tensor(events, name).to(dtype=dtype)
            for name, dtype in FIELD_DTYPES.items()
        }
        lengths = {int(tensor.shape[0]) for tensor in local.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"{label} typed lifecycle fields have inconsistent lengths"
            )
        count = next(iter(lengths), 0)
        expected_sequence = torch.arange(count, dtype=torch.int64)
        if not torch.equal(local["sequence"], expected_sequence):
            raise ValueError(
                f"{label} typed lifecycle sequence is not a contiguous zero-based cursor"
            )
        normalized.append(local)
        counts.append(count)

    total = counts[0] + counts[1]
    out = {
        name: torch.cat((normalized[0][name], normalized[1][name]), dim=0).contiguous()
        for name in FIELD_DTYPES
    }
    out["sequence"] = torch.arange(total, dtype=torch.int64)
    return out


def continue_typed_lifecycle_event_accumulator(
    checkpoint_events: Mapping[str, Any],
    reference_events: Mapping[str, Any],
    *,
    checkpoint_cursor: Any,
) -> dict[str, torch.Tensor]:
    try:
        cursor = operator.index(checkpoint_cursor)
    except TypeError as exc:
        raise TypeError("typed lifecycle checkpoint cursor must be an integer") from exc

    continued = concatenate_typed_lifecycle_events(checkpoint_events, reference_events)
    checkpoint_rows = int(_required_tensor(checkpoint_events, "event_type").shape[0])
    if cursor < 0 or cursor != checkpoint_rows:
        raise RuntimeError(
            "typed lifecycle checkpoint cursor contradiction: "
            f"cursor={cursor} checkpoint_rows={checkpoint_rows}"
        )
    return continued


def validated_typed_lifecycle_event_cursor(
    projection: Mapping[str, Any],
    events: Mapping[str, Any],
) -> int:
    layout = validated_event_storage_layout(projection)
    _tensors, event_rows = _validated_typed_event_tensors(events, label="projected")
    if int(layout["active_count"]) != event_rows:
        raise RuntimeError(
            "typed lifecycle projection cursor contradiction: "
            f"active_count={layout['active_count']} event_rows={event_rows}"
        )
    sequence_base = int(layout.get("sequence_base", 0))
    sequence = _tensors["sequence"]
    expected_sequence = torch.arange(
        sequence_base,
        sequence_base + event_rows,
        dtype=torch.int64,
    )
    if not torch.equal(sequence, expected_sequence):
        raise RuntimeError(
            "typed lifecycle projection sequence is not contiguous from its frontier"
        )
    return event_rows


def typed_lifecycle_events_from_projection(
    projection: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    tensors = dict(projection.get("tensors") or {})
    layout = validated_event_storage_layout(projection)
    if layout["representation"] == "empty":
        return empty_typed_lifecycle_event_tensors()
    if layout["representation"] == "legacy":
        return typed_lifecycle_events_from_legacy_tape(projection)
    prefix = "typed_lifecycle_event_"
    out: dict[str, torch.Tensor] = {}
    for name, dtype in FIELD_DTYPES.items():
        out[name] = _required_tensor(tensors, f"{prefix}{name}").to(dtype=dtype)
    active_count = int(layout["active_count"])
    return {name: tensor[:active_count].contiguous() for name, tensor in out.items()}


def typed_lifecycle_event_digest(events: Mapping[str, torch.Tensor]) -> str:
    tensors, _count = _validated_typed_event_tensors(events, label="digest")
    h = hashlib.sha256()
    h.update(SCHEMA_VERSION.encode("utf-8"))
    for name in sorted(FIELD_DTYPES):
        tensor = tensors[name]
        array = np.ascontiguousarray(tensor.detach().to("cpu").numpy())
        h.update(name.encode("utf-8"))
        h.update(str(array.dtype).encode("utf-8"))
        h.update(
            json.dumps([int(dim) for dim in array.shape], separators=(",", ":")).encode(
                "utf-8"
            )
        )
        h.update(array.tobytes())
    return h.hexdigest()


def authoritative_event_cursor_envelope_digest_sha256(
    projection: Mapping[str, Any],
) -> str:
    """Hash the validated event cursors and authoritative active event prefix."""

    layout = validated_event_storage_layout(projection)
    events = typed_lifecycle_events_from_projection(projection)
    active_prefix_event_digest = typed_lifecycle_event_digest(events)
    typed_active_cursor = validated_typed_lifecycle_event_cursor(projection, events)

    metadata = projection.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError(
            "authoritative event cursor projection metadata must be a mapping"
        )

    def exact_nonnegative_integer(value: Any, *, name: str) -> int:
        if type(value) is not int:
            raise TypeError(
                f"authoritative event cursor {name} must be an exact integer"
            )
        if value < 0:
            raise RuntimeError(f"authoritative event cursor {name} must be nonnegative")
        return value

    event_cursor = exact_nonnegative_integer(
        projection.get("event_cursor"),
        name="projection.event_cursor",
    )
    runtime_event_cursor = exact_nonnegative_integer(
        projection.get("runtime_event_cursor"),
        name="projection.runtime_event_cursor",
    )
    next_sequence = exact_nonnegative_integer(
        metadata.get("typed_lifecycle_event_next_sequence"),
        name="metadata.typed_lifecycle_event_next_sequence",
    )
    active_count = int(layout["active_count"])
    cursor_fields = {
        "event_cursor": event_cursor,
        "validated_active_count": active_count,
        "typed_active_cursor": typed_active_cursor,
    }
    if len(set(cursor_fields.values())) != 1:
        raise RuntimeError(f"authoritative event cursor contradiction: {cursor_fields}")

    envelope = {
        "schema_version": _EVENT_CURSOR_ENVELOPE_SCHEMA_VERSION,
        "event_cursor": event_cursor,
        "runtime_event_cursor": runtime_event_cursor,
        "validated_event_storage_layout": {
            "active_count": active_count,
            "capacity": int(layout["capacity"]),
            "representation": str(layout["representation"]),
        },
        "typed_active_cursor": typed_active_cursor,
        "typed_lifecycle_event_sequence_base": int(layout.get("sequence_base", 0)),
        "typed_lifecycle_event_next_sequence": next_sequence,
        "event_history_frontier": int(
            metadata.get("event_history_frontier", next_sequence)
        ),
        "active_prefix_event_digest_sha256": active_prefix_event_digest,
    }
    canonical_ascii_json = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical_ascii_json).hexdigest()


def typed_lifecycle_events_to_rows(
    events: Mapping[str, torch.Tensor],
    *,
    start: int = 0,
    stop: int | None = None,
) -> list[dict[str, Any]]:
    tensors, total = _validated_typed_event_tensors(events, label="row serialization")
    lo = max(0, int(start))
    hi = total if stop is None else max(lo, min(total, int(stop)))
    rows: list[dict[str, Any]] = []
    for index in range(lo, hi):
        row: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
        for name in FIELD_DTYPES:
            value = tensors[name][index].item()
            row[name] = float(value) if isinstance(value, float) else int(value)
        rows.append(row)
    return rows


def _coerce_typed_lifecycle_events(source: Any) -> dict[str, torch.Tensor]:
    if hasattr(source, "metadata") and hasattr(source, "tensors"):
        source = {"metadata": source.metadata, "tensors": source.tensors}
    if isinstance(source, Mapping) and ("metadata" in source or "tensors" in source):
        return typed_lifecycle_events_from_projection(source)
    if not isinstance(source, Mapping):
        raise TypeError("typed lifecycle source must be an event mapping or projection")
    return {
        name: _required_tensor(source, name).to(dtype=dtype)
        for name, dtype in FIELD_DTYPES.items()
    }


def committed_death_semantic_rows(source: Any) -> list[dict[str, int | float]]:
    events = _coerce_typed_lifecycle_events(source)
    rows = typed_lifecycle_events_to_rows(events)
    deaths = [
        row
        for row in rows
        if int(row["event_type"]) == int(LifecycleEventType.DEATH_COMMITTED)
    ]
    deaths.sort(
        key=lambda row: (
            int(row["generation"]),
            int(row["tick"]),
            int(row["sequence"]),
            int(row["actor_agent_id"]),
        )
    )
    normalized: list[dict[str, int | float]] = []
    for index, row in enumerate(deaths):
        generation = int(row["generation"])
        tick = int(row["tick"])
        actor_id = int(row["actor_agent_id"])
        target_id = int(row["target_agent_id"])
        value = float(row["value"])
        reason_code = int(row["reason_code"])
        if reason_code == SENTINEL_INT and math.isfinite(value) and value.is_integer():
            reason_code = int(value)
        if generation < 0 or tick < 0:
            raise ValueError(
                f"committed death row {index} has a negative generation/tick"
            )
        if actor_id < 0:
            raise ValueError(f"committed death row {index} has no actor identity")
        if target_id != SENTINEL_INT:
            raise ValueError(f"committed death row {index} target_agent_id must be -1")
        if int(row["subtype_code"]) != int(LifecycleEventType.DEATH_COMMITTED):
            raise ValueError(f"committed death row {index} has invalid subtype_code")
        if int(row["success"]) != 1:
            raise ValueError(
                f"committed death row {index} must be committed successfully"
            )
        if reason_code not in set(_DEATH_REASON_CODES.values()):
            raise ValueError(
                f"committed death row {index} has unsupported reason_code {reason_code}"
            )
        if not math.isfinite(value) or value != float(reason_code):
            raise ValueError(f"committed death row {index} reason_code/value mismatch")
        normalized.append(
            {
                "event_type": int(LifecycleEventType.DEATH_COMMITTED),
                "generation": generation,
                "tick": tick,
                "actor_agent_id": actor_id,
                "target_agent_id": target_id,
                "reason_code": reason_code,
                "value": value,
            }
        )
    return normalized


def _semantic_rows_digest(rows: list[dict[str, int | float]]) -> str:
    payload = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _first_semantic_mismatch(
    reference_rows: list[dict[str, int | float]],
    candidate_rows: list[dict[str, int | float]],
) -> dict[str, Any] | None:
    shared = min(len(reference_rows), len(candidate_rows))
    for index in range(shared):
        reference = reference_rows[index]
        candidate = candidate_rows[index]
        for field in reference:
            if reference[field] != candidate.get(field):
                return {
                    "path": f"$.committed_deaths[{index}].{field}",
                    "reference": reference[field],
                    "candidate": candidate.get(field),
                }
    if len(reference_rows) != len(candidate_rows):
        return {
            "path": "$.committed_deaths.length",
            "reference": len(reference_rows),
            "candidate": len(candidate_rows),
        }
    return None


def compare_committed_death_semantics(reference: Any, candidate: Any) -> dict[str, Any]:
    errors: dict[str, str] = {}
    try:
        reference_rows = committed_death_semantic_rows(reference)
    except Exception as exc:
        reference_rows = []
        errors["reference"] = f"{type(exc).__name__}: {exc}"
    try:
        candidate_rows = committed_death_semantic_rows(candidate)
    except Exception as exc:
        candidate_rows = []
        errors["candidate"] = f"{type(exc).__name__}: {exc}"
    non_vacuous = bool(reference_rows)
    mismatch = (
        None if errors else _first_semantic_mismatch(reference_rows, candidate_rows)
    )
    if mismatch is None and not errors and not non_vacuous:
        mismatch = {
            "path": "$.committed_deaths.reference_count",
            "reference": ">0",
            "candidate": len(candidate_rows),
        }
    return {
        "schema_version": "eiel_committed_death_semantic_compare_v1",
        "ok": not errors and mismatch is None,
        "non_vacuous": non_vacuous,
        "reference_count": len(reference_rows),
        "candidate_count": len(candidate_rows),
        "reference_digest": _semantic_rows_digest(reference_rows),
        "candidate_digest": _semantic_rows_digest(candidate_rows),
        "first_mismatch": mismatch,
        "errors": errors,
        "reference_rows": reference_rows,
        "candidate_rows": candidate_rows,
    }


def write_typed_lifecycle_events_jsonl(
    path: str | Path,
    events: Mapping[str, torch.Tensor],
    *,
    start: int = 0,
    stop: int | None = None,
) -> int:
    rows = typed_lifecycle_events_to_rows(events, start=start, stop=stop)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if int(start) > 0 and out.exists() else "w"
    timing = current_flat_output_cost_timing()
    if timing is not None:
        with timing.serialization_phase():
            encoded = "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            )
        with timing.write_phase():
            with out.open(mode, encoding="utf-8") as handle:
                handle.write(encoded)
        return int(start) + len(rows)
    with out.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return int(start) + len(rows)


def write_typed_lifecycle_events_csv(
    path: str | Path,
    events: Mapping[str, torch.Tensor],
    *,
    start: int = 0,
    stop: int | None = None,
) -> int:
    rows = typed_lifecycle_events_to_rows(events, start=start, stop=stop)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["schema_version", *FIELD_DTYPES.keys()]
    append = int(start) > 0 and out.exists()
    timing = current_flat_output_cost_timing()
    if timing is not None:
        with timing.serialization_phase():
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
            if not append:
                writer.writeheader()
            writer.writerows(rows)
            encoded = buffer.getvalue()
        with timing.write_phase():
            with out.open(
                "a" if append else "w", encoding="utf-8", newline=""
            ) as handle:
                handle.write(encoded)
        return int(start) + len(rows)
    with out.open("a" if append else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if not append:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return int(start) + len(rows)


def _jsonl_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def _expected_jsonl_prefix(events: Mapping[str, torch.Tensor], count: int) -> bytes:
    rows = typed_lifecycle_events_to_rows(events, start=0, stop=int(count))
    timing = current_flat_output_cost_timing()
    if timing is None:
        return "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ).encode("utf-8")
    with timing.serialization_phase():
        return "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ).encode("utf-8")


def _expected_csv_prefix(events: Mapping[str, torch.Tensor], count: int) -> bytes:
    rows = typed_lifecycle_events_to_rows(events, start=0, stop=int(count))
    timing = current_flat_output_cost_timing()
    if timing is None:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=["schema_version", *FIELD_DTYPES.keys()],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue().encode("utf-8")
    with timing.serialization_phase():
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=["schema_version", *FIELD_DTYPES.keys()],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue().encode("utf-8")


def _lifecycle_output_contract(jsonl: Path) -> dict[str, Any] | None:
    root = jsonl.parent.parent if jsonl.parent.name == "metrics" else jsonl.parent
    root = root.expanduser().resolve()
    key = str(root)
    if key in _LIFECYCLE_OUTPUT_CONTRACT_CACHE:
        return _LIFECYCLE_OUTPUT_CONTRACT_CACHE[key]
    contract_path = root / "artifact_lifecycle_contract.json"
    if not contract_path.is_file():
        _LIFECYCLE_OUTPUT_CONTRACT_CACHE[key] = None
        return None
    try:
        from engine.campaigns.artifact_lifecycle import (
            validate_run_artifact_lifecycle_contract_v1,
        )

        contract = validate_run_artifact_lifecycle_contract_v1(
            json.loads(contract_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise RuntimeError(
            f"invalid bound artifact lifecycle contract for typed lifecycle output: {contract_path}: {exc}"
        ) from exc
    _LIFECYCLE_OUTPUT_CONTRACT_CACHE[key] = contract
    return contract


def _write_lifecycle_output_violation(
    *,
    jsonl: Path,
    contract: Mapping[str, Any],
    code: str,
    attempted_bytes: int,
    limit_bytes: int | None,
) -> None:
    from engine.campaigns.canonical import canonical_sha256

    root = jsonl.parent.parent if jsonl.parent.name == "metrics" else jsonl.parent
    root = root.expanduser().resolve()
    core = {
        "schema_version": "eiel_lifecycle_storage_violation_receipt_v1",
        "run_root": str(root),
        "lifecycle_profile": contract["lifecycle_profile"],
        "lifecycle_contract_sha256": contract["contract_sha256"],
        "artifact_family": "native_runtime.typed_lifecycle_events",
        "owner_writer": "write_typed_lifecycle_public_outputs",
        "artifact_path": "metrics/typed_lifecycle_events.jsonl+csv",
        "code": code,
        "attempted_bytes": int(attempted_bytes),
        "limit_bytes": None if limit_bytes is None else int(limit_bytes),
        "automatic_prune_allowed": False,
        "status": "blocked",
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    root.mkdir(parents=True, exist_ok=True)
    (root / "lifecycle_storage_violation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _enforce_lifecycle_output_before_write(
    *,
    jsonl: Path,
    csv_path: Path,
) -> tuple[dict[str, Any] | None, int | None]:
    contract = _lifecycle_output_contract(jsonl)
    if contract is None:
        return None, None
    event_policy = str(contract["observability"]["debug_event_trace_policy"])
    current = sum(
        int(path.stat().st_size) for path in (jsonl, csv_path) if path.is_file()
    )
    if event_policy == "forbidden":
        _write_lifecycle_output_violation(
            jsonl=jsonl,
            contract=contract,
            code="undeclared_event_trace",
            attempted_bytes=max(1, current),
            limit_bytes=0,
        )
        raise RuntimeError(
            "lifecycle contract forbids native typed lifecycle event outputs"
        )
    limit = int(contract["storage"]["scratch_max_bytes"])
    ring = contract["observability"]["ring_buffer"]
    if event_policy == "bounded" and ring["enabled"]:
        limit = min(limit, int(ring["max_bytes"]))
    if current > limit:
        _write_lifecycle_output_violation(
            jsonl=jsonl,
            contract=contract,
            code="scratch_quota_exceeded",
            attempted_bytes=current,
            limit_bytes=limit,
        )
        raise RuntimeError(
            "lifecycle contract scratch quota blocks native typed lifecycle outputs"
        )
    return contract, limit


def _enforce_lifecycle_output_after_write(
    *,
    jsonl: Path,
    csv_path: Path,
    contract: Mapping[str, Any] | None,
    limit: int | None,
) -> None:
    if contract is None or limit is None:
        return
    current = sum(
        int(path.stat().st_size) for path in (jsonl, csv_path) if path.is_file()
    )
    if current > limit:
        _write_lifecycle_output_violation(
            jsonl=jsonl,
            contract=contract,
            code="scratch_quota_exceeded",
            attempted_bytes=current,
            limit_bytes=limit,
        )
        raise RuntimeError(
            "lifecycle contract scratch quota blocks native typed lifecycle outputs"
        )


def _validate_public_output_prefix(
    path: Path,
    events: Mapping[str, torch.Tensor],
    count: int | None,
    *,
    label: str,
) -> None:
    if count is None:
        return
    expected = (
        _expected_jsonl_prefix(events, count)
        if label == "jsonl"
        else _expected_csv_prefix(events, count)
    )
    if path.read_bytes() != expected:
        raise RuntimeError(
            f"typed lifecycle public output {label} prefix conflicts with canonical tape"
        )


def write_typed_lifecycle_public_outputs(
    jsonl_path: str | Path,
    csv_path: str | Path,
    events: Mapping[str, torch.Tensor],
    *,
    cursor: int | None = None,
) -> dict[str, Any]:
    jsonl = Path(jsonl_path)
    csv_out = Path(csv_path)
    lifecycle_contract, lifecycle_limit = _enforce_lifecycle_output_before_write(
        jsonl=jsonl,
        csv_path=csv_out,
    )
    _tensors, total_rows = _validated_typed_event_tensors(events, label="public output")
    json_count = _jsonl_row_count(jsonl)
    csv_count = _csv_row_count(csv_out)
    if (
        json_count is not None
        and csv_count is not None
        and int(json_count) != int(csv_count)
    ):
        raise RuntimeError("typed lifecycle public output JSONL/CSV row count mismatch")
    if cursor is None:
        start_cursor = int(
            json_count
            if json_count is not None
            else csv_count
            if csv_count is not None
            else 0
        )
        cursor_source = (
            "existing_file_row_count"
            if json_count is not None or csv_count is not None
            else "new_output"
        )
    else:
        start_cursor = int(cursor)
        cursor_source = "provided_cursor"
    if start_cursor < 0 or start_cursor > total_rows:
        raise RuntimeError("typed lifecycle public output cursor outside event range")
    for label, count in (("jsonl", json_count), ("csv", csv_count)):
        if count is not None and int(count) > total_rows:
            raise RuntimeError(
                f"typed lifecycle public output {label} has rows beyond event range"
            )
        if cursor is not None and count is not None and int(count) != start_cursor:
            raise RuntimeError(
                f"typed lifecycle public output {label} row count conflicts with cursor"
            )
    _validate_public_output_prefix(jsonl, events, json_count, label="jsonl")
    _validate_public_output_prefix(csv_out, events, csv_count, label="csv")

    if json_count is None:
        json_cursor = write_typed_lifecycle_events_jsonl(
            jsonl, events, start=0, stop=total_rows
        )
    else:
        json_cursor = write_typed_lifecycle_events_jsonl(
            jsonl, events, start=start_cursor
        )
    if csv_count is None:
        csv_cursor = write_typed_lifecycle_events_csv(
            csv_out, events, start=0, stop=total_rows
        )
    else:
        csv_cursor = write_typed_lifecycle_events_csv(
            csv_out, events, start=start_cursor
        )
    if int(json_cursor) != int(csv_cursor):
        raise RuntimeError("typed lifecycle public output cursor mismatch")
    _enforce_lifecycle_output_after_write(
        jsonl=jsonl,
        csv_path=csv_out,
        contract=lifecycle_contract,
        limit=lifecycle_limit,
    )
    return {
        "cursor_source": cursor_source,
        "start_cursor": int(start_cursor),
        "end_cursor": int(json_cursor),
        "rows_written": int(json_cursor - start_cursor),
        "jsonl_existed": json_count is not None,
        "csv_existed": csv_count is not None,
    }
