"""Canonical fixed-capacity CPU receipts for modular recurrent mutations.

This module is deliberately storage-only. It validates and canonicalizes a
three-entry mutation sequence without invoking mutation execution or CUDA.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import re
from typing import Any, Final

from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    CHANGED_DIMENSIONS,
    OPERATIONS,
    STATUSES,
    SUPPORTED_PROFILE_IDS,
    float64_from_bits,
)


SCHEMA_VERSION: Final[str] = "eiel_modular_recurrent_mutation_sequence_v1"
OVERFLOW_SCHEMA_VERSION: Final[str] = (
    "eiel_modular_recurrent_mutation_sequence_overflow_v1"
)
CAPACITY: Final[int] = 3
MUTATION_SEQUENCE_CAPACITY_V1: Final[int] = CAPACITY

_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "profile_id",
    "source_packed_sha256",
    "capacity",
    "count",
    "entries",
    "sequence_sha256",
)
_ENTRY_BODY_FIELDS: Final[tuple[str, ...]] = (
    "sequence_index",
    "generation",
    "tick",
    "parent_agent_id",
    "parent_slot",
    "parent_slot_token",
    "child_agent_id",
    "child_slot",
    "child_slot_token",
    "draw0_float64_bits",
    "draw1_float64_bits",
    "operation",
    "status",
    "reason",
    "selected_instance_id_index",
    "selected_homology_id_index",
    "changed_dimension",
    "changed",
    "pre_genome_sha256",
    "post_genome_sha256",
    "committed",
)
_ENTRY_FIELDS: Final[tuple[str, ...]] = (*_ENTRY_BODY_FIELDS, "receipt_sha256")
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FLOAT64_BITS = re.compile(r"[0-9a-f]{16}\Z")
_REASON = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,255}\Z")
_PROFILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_INT32_MAX: Final[int] = (1 << 31) - 1
_INT64_MAX: Final[int] = (1 << 63) - 1
_DIMENSIONS_BY_OPERATION: Final[dict[str, frozenset[str]]] = {
    "toggle_dormancy": frozenset({"dormant"}),
    "prune_eligible_module": frozenset({"module"}),
    "refine_descendant_parameter": frozenset(
        CHANGED_DIMENSIONS
        - {
            "none",
            "module",
            "dormant",
            "input_binding_gain",
            "output_binding_gain",
        }
    ),
    "refine_descendant_routing_or_expression": frozenset(
        {"expression_scale", "input_binding_gain", "output_binding_gain"}
    ),
}


class ModularRecurrentMutationSequenceValidationError(ValueError):
    """A mutation sequence failed the strict v1 contract."""


class ModularRecurrentMutationSequenceOverflowError(
    ModularRecurrentMutationSequenceValidationError
):
    """Appending would exceed the fixed sequence capacity."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        self.payload = dict(payload)
        super().__init__("mutation sequence capacity overflow")


@dataclass(frozen=True, slots=True)
class MutationSequenceEntryV1:
    sequence_index: int
    generation: int
    tick: int
    parent_agent_id: int
    parent_slot: int
    parent_slot_token: int
    child_agent_id: int
    child_slot: int
    child_slot_token: int
    draw0_float64_bits: str
    draw1_float64_bits: str
    operation: str
    status: str
    reason: str
    selected_instance_id_index: int
    selected_homology_id_index: int
    changed_dimension: str
    changed: bool
    pre_genome_sha256: str
    post_genome_sha256: str
    committed: bool
    receipt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in _ENTRY_FIELDS}


@dataclass(frozen=True, slots=True)
class ModularRecurrentMutationSequenceV1:
    schema_version: str
    profile_id: str
    source_packed_sha256: str
    capacity: int
    count: int
    entries: tuple[MutationSequenceEntryV1 | None, ...]
    sequence_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "source_packed_sha256": self.source_packed_sha256,
            "capacity": self.capacity,
            "count": self.count,
            "entries": [
                entry.to_dict() if entry is not None else None for entry in self.entries
            ],
            "sequence_sha256": self.sequence_sha256,
        }


def _fail(path: str, message: str) -> None:
    raise ModularRecurrentMutationSequenceValidationError(f"{path} {message}")


def _object(
    value: object, path: str, expected_fields: tuple[str, ...]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    keys = set(value.keys())
    if not all(type(key) is str for key in keys):
        _fail(path, "contains a non-string field name")
    expected = set(expected_fields)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing:
        _fail(path, f"missing required fields: {missing}")
    if unknown:
        _fail(path, f"contains unknown fields: {unknown}")
    return value


def _literal(value: object, path: str, expected: str) -> str:
    if type(value) is not str or value != expected:
        _fail(path, f"must be {expected!r}")
    return value


def _matched_string(value: object, path: str, pattern: re.Pattern[str]) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        _fail(path, "has an invalid canonical string value")
    return value


def _choice(value: object, path: str, choices: frozenset[str] | tuple[str, ...]) -> str:
    if type(value) is not str or value not in choices:
        _fail(path, f"must be one of {sorted(choices)}")
    return value


def _integer(value: object, path: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        _fail(path, f"must be an integer in [{lower}, {upper}]")
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "must be a boolean")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _operation_for_draw0(draw0: float) -> str:
    if draw0 < 0.25:
        return OPERATIONS[0]
    if draw0 < 0.5:
        return OPERATIONS[1]
    if draw0 < 0.75:
        return OPERATIONS[2]
    return OPERATIONS[3]


def _unit_draw(bits: object, path: str) -> tuple[str, float]:
    canonical_bits = _matched_string(bits, path, _FLOAT64_BITS)
    value = float64_from_bits(canonical_bits)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        _fail(path, "must encode a finite value in [0, 1)")
    return canonical_bits, value


def _entry_body(entry: MutationSequenceEntryV1) -> dict[str, object]:
    return {field: getattr(entry, field) for field in _ENTRY_BODY_FIELDS}


def mutation_sequence_entry_sha256_v1(entry: MutationSequenceEntryV1) -> str:
    """Compute one canonical entry digest, excluding its digest field."""

    return hashlib.sha256(_canonical_json_bytes(_entry_body(entry))).hexdigest()


def _validate_entry(value: object, *, index: int) -> MutationSequenceEntryV1:
    path = f"$.entries[{index}]"
    row = _object(value, path, _ENTRY_FIELDS)
    draw0_bits, draw0 = _unit_draw(
        row["draw0_float64_bits"], f"{path}.draw0_float64_bits"
    )
    draw1_bits, _ = _unit_draw(row["draw1_float64_bits"], f"{path}.draw1_float64_bits")
    operation = _choice(row["operation"], f"{path}.operation", OPERATIONS)
    expected_operation = _operation_for_draw0(draw0)
    if operation != expected_operation:
        _fail(
            f"{path}.operation",
            f"does not match draw0; expected {expected_operation!r}",
        )

    status = _choice(row["status"], f"{path}.status", STATUSES)
    changed = _boolean(row["changed"], f"{path}.changed")
    dimension = _choice(
        row["changed_dimension"], f"{path}.changed_dimension", CHANGED_DIMENSIONS
    )
    if changed != (status == "accepted"):
        _fail(path, "must set changed exactly when status is 'accepted'")
    if changed == (dimension == "none"):
        _fail(path, "must use a non-'none' changed_dimension exactly when changed")

    selected_instance = _integer(
        row["selected_instance_id_index"],
        f"{path}.selected_instance_id_index",
        -1,
        _INT32_MAX,
    )
    selected_homology = _integer(
        row["selected_homology_id_index"],
        f"{path}.selected_homology_id_index",
        -1,
        _INT32_MAX,
    )
    if changed and (selected_instance < 0 or selected_homology < 0):
        _fail(path, "accepted mutation must identify its selected module")
    if changed and dimension not in _DIMENSIONS_BY_OPERATION[operation]:
        _fail(path, "changed_dimension is incompatible with operation")

    parent_agent_id = _integer(
        row["parent_agent_id"], f"{path}.parent_agent_id", 0, _INT64_MAX
    )
    child_agent_id = _integer(
        row["child_agent_id"], f"{path}.child_agent_id", 0, _INT64_MAX
    )
    if parent_agent_id == child_agent_id:
        _fail(path, "parent and child agent identities must differ")

    pre_genome_sha256 = _matched_string(
        row["pre_genome_sha256"], f"{path}.pre_genome_sha256", _LOWER_SHA256
    )
    post_genome_sha256 = _matched_string(
        row["post_genome_sha256"], f"{path}.post_genome_sha256", _LOWER_SHA256
    )
    if changed != (pre_genome_sha256 != post_genome_sha256):
        _fail(path, "changed must exactly match a genome digest transition")
    if not _boolean(row["committed"], f"{path}.committed"):
        _fail(f"{path}.committed", "must be true for every active entry")

    entry = MutationSequenceEntryV1(
        sequence_index=_integer(
            row["sequence_index"], f"{path}.sequence_index", 0, CAPACITY - 1
        ),
        generation=_integer(row["generation"], f"{path}.generation", 1, _INT64_MAX),
        tick=_integer(row["tick"], f"{path}.tick", 0, 0),
        parent_agent_id=parent_agent_id,
        parent_slot=_integer(row["parent_slot"], f"{path}.parent_slot", 0, _INT32_MAX),
        parent_slot_token=_integer(
            row["parent_slot_token"], f"{path}.parent_slot_token", 0, _INT64_MAX
        ),
        child_agent_id=child_agent_id,
        child_slot=_integer(row["child_slot"], f"{path}.child_slot", 0, _INT32_MAX),
        child_slot_token=_integer(
            row["child_slot_token"], f"{path}.child_slot_token", 0, _INT64_MAX
        ),
        draw0_float64_bits=draw0_bits,
        draw1_float64_bits=draw1_bits,
        operation=operation,
        status=status,
        reason=_matched_string(row["reason"], f"{path}.reason", _REASON),
        selected_instance_id_index=selected_instance,
        selected_homology_id_index=selected_homology,
        changed_dimension=dimension,
        changed=changed,
        pre_genome_sha256=pre_genome_sha256,
        post_genome_sha256=post_genome_sha256,
        committed=True,
        receipt_sha256=_matched_string(
            row["receipt_sha256"], f"{path}.receipt_sha256", _LOWER_SHA256
        ),
    )
    expected_digest = mutation_sequence_entry_sha256_v1(entry)
    if not hmac.compare_digest(entry.receipt_sha256, expected_digest):
        _fail(f"{path}.receipt_sha256", "does not match the canonical entry body")
    return entry


def _sequence_body(
    *,
    profile_id: str,
    source_packed_sha256: str,
    count: int,
    entries: Sequence[MutationSequenceEntryV1 | None],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile_id,
        "source_packed_sha256": source_packed_sha256,
        "capacity": CAPACITY,
        "count": count,
        "entries": [
            entry.to_dict() if entry is not None else None for entry in entries
        ],
    }


def mutation_sequence_sha256_v1(value: ModularRecurrentMutationSequenceV1) -> str:
    """Compute the sequence digest over its complete fixed-capacity body."""

    body = _sequence_body(
        profile_id=value.profile_id,
        source_packed_sha256=value.source_packed_sha256,
        count=value.count,
        entries=value.entries,
    )
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def validate_modular_recurrent_mutation_sequence_v1(
    value: object,
) -> ModularRecurrentMutationSequenceV1:
    """Validate and normalize a strict three-slot mutation sequence."""

    row = _object(value, "$", _TOP_LEVEL_FIELDS)
    capacity = _integer(row["capacity"], "$.capacity", CAPACITY, CAPACITY)
    count = _integer(row["count"], "$.count", 0, CAPACITY)
    entries_value = row["entries"]
    if type(entries_value) is not list:
        _fail("$.entries", "must be an array")
    if len(entries_value) != CAPACITY:
        _fail("$.entries", f"must contain exactly {CAPACITY} slots")

    entries: list[MutationSequenceEntryV1 | None] = []
    for index, entry_value in enumerate(entries_value):
        if index < count:
            if entry_value is None:
                _fail(f"$.entries[{index}]", "must be active within the count prefix")
            entry = _validate_entry(entry_value, index=index)
            if entry.sequence_index != index:
                _fail(
                    f"$.entries[{index}].sequence_index",
                    f"must be the canonical contiguous index {index}",
                )
            entries.append(entry)
        else:
            if entry_value is not None:
                _fail(f"$.entries[{index}]", "must be null in the inactive suffix")
            entries.append(None)

    active_entries = tuple(entry for entry in entries if entry is not None)
    child_ids: set[int] = set()
    child_owners: set[tuple[int, int]] = set()
    first_generation: int | None = None
    for index, entry in enumerate(active_entries):
        if first_generation is None:
            first_generation = entry.generation
        elif entry.generation != first_generation + index:
            _fail(
                f"$.entries[{index}].generation",
                "must be contiguous across the active prefix",
            )
        if entry.child_agent_id in child_ids:
            _fail(f"$.entries[{index}].child_agent_id", "is duplicated")
        child_owner = (entry.child_slot, entry.child_slot_token)
        if child_owner in child_owners:
            _fail(f"$.entries[{index}]", "duplicates a child slot/token owner")
        child_ids.add(entry.child_agent_id)
        child_owners.add(child_owner)

    profile_id = _choice(
        _matched_string(row["profile_id"], "$.profile_id", _PROFILE),
        "$.profile_id",
        SUPPORTED_PROFILE_IDS,
    )
    sequence = ModularRecurrentMutationSequenceV1(
        schema_version=_literal(
            row["schema_version"], "$.schema_version", SCHEMA_VERSION
        ),
        profile_id=profile_id,
        source_packed_sha256=_matched_string(
            row["source_packed_sha256"], "$.source_packed_sha256", _LOWER_SHA256
        ),
        capacity=capacity,
        count=count,
        entries=tuple(entries),
        sequence_sha256=_matched_string(
            row["sequence_sha256"], "$.sequence_sha256", _LOWER_SHA256
        ),
    )
    expected_digest = mutation_sequence_sha256_v1(sequence)
    if not hmac.compare_digest(sequence.sequence_sha256, expected_digest):
        _fail("$.sequence_sha256", "does not match the canonical sequence body")
    return sequence


def _entry_wire_for_build(
    value: Mapping[str, object] | MutationSequenceEntryV1,
    *,
    index: int,
) -> dict[str, object]:
    if isinstance(value, MutationSequenceEntryV1):
        wire = value.to_dict()
    elif isinstance(value, Mapping):
        wire = dict(value)
    else:
        _fail(f"entries[{index}]", "must be an entry object")

    keys = set(wire)
    body_fields = set(_ENTRY_BODY_FIELDS)
    full_fields = set(_ENTRY_FIELDS)
    if keys == body_fields:
        wire["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(wire)).hexdigest()
    elif keys != full_fields:
        _object(wire, f"entries[{index}]", _ENTRY_FIELDS)
    return _validate_entry(wire, index=index).to_dict()


def build_modular_recurrent_mutation_sequence_v1(
    *,
    profile_id: str,
    source_packed_sha256: str,
    entries: Sequence[Mapping[str, object] | MutationSequenceEntryV1 | None],
    count: int | None = None,
    capacity: int = CAPACITY,
) -> ModularRecurrentMutationSequenceV1:
    """Build a canonical sequence from compact or already padded entries."""

    if type(capacity) is not int or capacity != CAPACITY:
        _fail("capacity", f"must be exactly {CAPACITY}")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        _fail("entries", "must be a sequence")
    entry_values = list(entries)
    if len(entry_values) > CAPACITY:
        _fail("entries", f"cannot exceed fixed capacity {CAPACITY}")

    first_none = next(
        (index for index, entry in enumerate(entry_values) if entry is None),
        len(entry_values),
    )
    if any(entry is not None for entry in entry_values[first_none:]):
        _fail("entries", "must contain an active prefix and inactive null suffix")
    derived_count = first_none
    if count is not None and (type(count) is not int or count != derived_count):
        _fail("count", f"must exactly match the active prefix length {derived_count}")

    active_wires = [
        _entry_wire_for_build(entry, index=index)
        for index, entry in enumerate(entry_values[:derived_count])
        if entry is not None
    ]
    padded_entries: list[dict[str, object] | None] = [*active_wires]
    padded_entries.extend([None] * (CAPACITY - derived_count))
    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile_id,
        "source_packed_sha256": source_packed_sha256,
        "capacity": CAPACITY,
        "count": derived_count,
        "entries": padded_entries,
    }
    body["sequence_sha256"] = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    return validate_modular_recurrent_mutation_sequence_v1(body)


def canonical_modular_recurrent_mutation_sequence_bytes_v1(
    value: Mapping[str, object] | ModularRecurrentMutationSequenceV1,
) -> bytes:
    """Return canonical ASCII JSON only after complete sequence validation."""

    wire = (
        value.to_dict()
        if isinstance(value, ModularRecurrentMutationSequenceV1)
        else value
    )
    sequence = validate_modular_recurrent_mutation_sequence_v1(wire)
    return _canonical_json_bytes(sequence.to_dict())


def modular_recurrent_mutation_sequence_overflow_contract_payload_v1(
    value: Mapping[str, object] | ModularRecurrentMutationSequenceV1,
) -> dict[str, object]:
    """Describe a rejected overflow while preserving the full exact prefix."""

    wire = (
        value.to_dict()
        if isinstance(value, ModularRecurrentMutationSequenceV1)
        else value
    )
    sequence = validate_modular_recurrent_mutation_sequence_v1(wire)
    if sequence.count != CAPACITY:
        _fail("$.count", "must equal capacity for an overflow contract")
    return {
        "schema_version": OVERFLOW_SCHEMA_VERSION,
        "profile_id": sequence.profile_id,
        "source_packed_sha256": sequence.source_packed_sha256,
        "capacity": sequence.capacity,
        "count": sequence.count,
        "attempted_sequence_index": sequence.count,
        "status": "rejected",
        "reason": "capacity_overflow",
        "entries": [entry.to_dict() for entry in sequence.entries if entry is not None],
        "sequence_sha256": sequence.sequence_sha256,
    }


def append_modular_recurrent_mutation_sequence_v1(
    value: Mapping[str, object] | ModularRecurrentMutationSequenceV1,
    entry: Mapping[str, object] | MutationSequenceEntryV1,
) -> ModularRecurrentMutationSequenceV1:
    """Return a new sequence with one entry, failing closed on overflow."""

    wire = (
        value.to_dict()
        if isinstance(value, ModularRecurrentMutationSequenceV1)
        else value
    )
    sequence = validate_modular_recurrent_mutation_sequence_v1(wire)
    if sequence.count == CAPACITY:
        payload = modular_recurrent_mutation_sequence_overflow_contract_payload_v1(
            sequence
        )
        raise ModularRecurrentMutationSequenceOverflowError(payload)
    active_entries = [
        current for current in sequence.entries[: sequence.count] if current is not None
    ]
    return build_modular_recurrent_mutation_sequence_v1(
        profile_id=sequence.profile_id,
        source_packed_sha256=sequence.source_packed_sha256,
        entries=[*active_entries, entry],
    )


def append_modular_recurrent_mutation_sequence_entry_v1(
    value: Mapping[str, object] | ModularRecurrentMutationSequenceV1,
    entry: Mapping[str, object] | MutationSequenceEntryV1,
) -> ModularRecurrentMutationSequenceV1:
    """Alias with an explicit entry name for callers building sequences."""

    return append_modular_recurrent_mutation_sequence_v1(value, entry)


mutation_sequence_overflow_contract_payload_v1 = (
    modular_recurrent_mutation_sequence_overflow_contract_payload_v1
)


__all__ = [
    "CAPACITY",
    "MUTATION_SEQUENCE_CAPACITY_V1",
    "OVERFLOW_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "ModularRecurrentMutationSequenceOverflowError",
    "ModularRecurrentMutationSequenceV1",
    "ModularRecurrentMutationSequenceValidationError",
    "MutationSequenceEntryV1",
    "append_modular_recurrent_mutation_sequence_entry_v1",
    "append_modular_recurrent_mutation_sequence_v1",
    "build_modular_recurrent_mutation_sequence_v1",
    "canonical_modular_recurrent_mutation_sequence_bytes_v1",
    "modular_recurrent_mutation_sequence_overflow_contract_payload_v1",
    "mutation_sequence_entry_sha256_v1",
    "mutation_sequence_overflow_contract_payload_v1",
    "mutation_sequence_sha256_v1",
    "validate_modular_recurrent_mutation_sequence_v1",
]
