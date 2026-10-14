"""Exact CPU replay for fixed-capacity modular mutation sequences."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hmac
from typing import Any

from agents.modular_genome_v1 import (
    canonical_hash_from_genome_dict,
    canonical_payload_from_genome_dict,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    ModularRecurrentMutationReceiptV1,
    ModularRecurrentMutationReceiptValidationError,
    build_modular_recurrent_mutation_receipt_v1,
    turnover_mutation_ancestry_sha256_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_replay_v1 import (
    ModularRecurrentMutationReplayError,
    replay_modular_recurrent_mutation_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_sequence_v1 import (
    ModularRecurrentMutationSequenceV1,
    ModularRecurrentMutationSequenceValidationError,
    MutationSequenceEntryV1,
    validate_modular_recurrent_mutation_sequence_v1,
)
from evolution.structural_observation_routing_v1 import ROUTING_BLOCK_KEY
from evolution.structural_output_routing_v1 import OUTPUT_ROUTING_BLOCK_KEY


_INT64_MAX = (1 << 63) - 1


class ModularRecurrentMutationSequenceReplayError(RuntimeError):
    """A canonical mutation sequence could not be replayed exactly."""


@dataclass(frozen=True, slots=True)
class MutationSequenceReplayEntryResultV1:
    """One independently owned result from the ordered sequence replay."""

    sequence_index: int
    generation: int
    tick: int
    parent_slot: int
    child_slot: int
    parent_agent_id: int
    child_agent_id: int
    origin_slot: int
    pre_genome_sha256: str
    post_genome_sha256: str
    receipt: ModularRecurrentMutationReceiptV1
    parent_record: dict[str, Any]
    child_record: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModularRecurrentMutationSequenceReplayResultV1:
    """Complete replay output suitable for checkpoint rebind."""

    final_records: tuple[dict[str, Any] | None, ...]
    entry_results: tuple[MutationSequenceReplayEntryResultV1, ...]
    origin_slot_by_agent_id: dict[int, int]
    final_origin_slot_by_record_slot: tuple[int | None, ...]
    identity_table: tuple[str, ...]

    @property
    def records(self) -> tuple[dict[str, Any] | None, ...]:
        """Compatibility alias for callers consuming the final sidecar."""

        return self.final_records

    @property
    def replay_results(self) -> tuple[MutationSequenceReplayEntryResultV1, ...]:
        """Compatibility alias for per-entry results."""

        return self.entry_results


def _fail(message: str) -> None:
    raise ModularRecurrentMutationSequenceReplayError(message)


def _exact_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or not 0 <= value <= _INT64_MAX:
        _fail(f"{field} must be an exact nonnegative int64 value")
    return value


def _canonical_genome(
    value: object,
    *,
    field: str,
) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a genome mapping")
    raw = deepcopy(dict(value))
    try:
        canonical = canonical_payload_from_genome_dict(raw)
        digest = canonical_hash_from_genome_dict(raw)
    except Exception as exc:
        raise ModularRecurrentMutationSequenceReplayError(
            f"{field} is not a canonicalizable genome: {exc}"
        ) from exc
    return canonical, digest


def _normalize_records(
    records: Sequence[Mapping[str, Any] | None],
) -> list[dict[str, Any] | None]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        _fail("initial_records must be a fixed-capacity sequence")
    normalized: list[dict[str, Any] | None] = []
    for slot, value in enumerate(records):
        if value is None:
            normalized.append(None)
            continue
        if not isinstance(value, Mapping):
            _fail(f"initial_records[{slot}] must be a record mapping or None")
        record = deepcopy(dict(value))
        _exact_nonnegative_int(
            record.get("agent_id"), field=f"initial_records[{slot}].agent_id"
        )
        _exact_nonnegative_int(
            record.get("agent_slot_token"),
            field=f"initial_records[{slot}].agent_slot_token",
        )
        _canonical_genome(record.get("genome"), field=f"initial_records[{slot}].genome")
        normalized.append(record)
    _validate_unique_live_owners(normalized, context="initial_records")
    return normalized


def _validate_unique_live_owners(
    records: Sequence[dict[str, Any] | None],
    *,
    context: str,
) -> None:
    owner_slot_by_id: dict[int, int] = {}
    for slot, record in enumerate(records):
        if record is None:
            continue
        agent_id = _exact_nonnegative_int(
            record.get("agent_id"), field=f"{context}[{slot}].agent_id"
        )
        prior_slot = owner_slot_by_id.get(agent_id)
        if prior_slot is not None:
            _fail(
                f"{context} contains duplicate live owner agent_id={agent_id} "
                f"at slots {prior_slot} and {slot}"
            )
        owner_slot_by_id[agent_id] = slot


def _identity_strings(
    records: Sequence[dict[str, Any] | None],
    *,
    context: str,
) -> tuple[str, ...]:
    strings: set[str] = set()
    for slot, record in enumerate(records):
        if record is None:
            continue
        genome, _ = _canonical_genome(
            record.get("genome"), field=f"{context}[{slot}].genome"
        )
        for module in genome["modules"]:
            for field in (
                "module_id",
                "module_instance_id",
                "homology_id",
                "module_type",
            ):
                strings.add(str(module.get(field, "")))
            params = module.get("params", {})
            if not isinstance(params, Mapping):
                continue
            for block_name in (ROUTING_BLOCK_KEY, OUTPUT_ROUTING_BLOCK_KEY):
                block = params.get(block_name)
                if not isinstance(block, Mapping):
                    continue
                bindings = block.get("bindings", ())
                if not isinstance(bindings, Sequence) or isinstance(
                    bindings, (str, bytes)
                ):
                    continue
                for binding in bindings:
                    if isinstance(binding, Mapping):
                        strings.add(str(binding.get("semantic_id", "")))
        for connection in genome["connections"]:
            for field in (
                "connection_id",
                "src_slot",
                "dst_slot",
                "value_type",
                "channel",
            ):
                strings.add(str(connection.get(field, "")))
    strings.discard("")
    return tuple(sorted(strings))


def _validate_identity_table(
    identity_table: Sequence[str],
    records: Sequence[dict[str, Any] | None],
) -> tuple[str, ...]:
    if isinstance(identity_table, (str, bytes)) or not isinstance(
        identity_table, Sequence
    ):
        _fail("identity_table must be a sequence of identity strings")
    table = tuple(identity_table)
    if any(type(value) is not str or not value for value in table):
        _fail("identity_table entries must be nonempty strings")
    if tuple(sorted(set(table))) != table:
        _fail("identity_table must be sorted and contain no duplicates")
    derived = _identity_strings(records, context="initial_records")
    if derived != table:
        _fail("identity_table does not exactly describe the initial record sidecar")
    return table


def _require_identity_table_stability(
    records: Sequence[dict[str, Any] | None],
    identity_table: tuple[str, ...],
    *,
    sequence_index: int,
) -> None:
    derived = _identity_strings(
        records, context=f"entries[{sequence_index}].result_records"
    )
    if derived != identity_table:
        _fail(f"entries[{sequence_index}] changed the fixed record identity_table")


def _validate_parent(
    records: Sequence[dict[str, Any] | None],
    entry: MutationSequenceEntryV1,
) -> dict[str, Any]:
    if not 0 <= entry.parent_slot < len(records):
        _fail(
            f"entries[{entry.sequence_index}] parent_slot is outside the record sidecar"
        )
    parent = records[entry.parent_slot]
    if parent is None:
        _fail(f"entries[{entry.sequence_index}] parent_slot has no current record")
    actual_id = _exact_nonnegative_int(
        parent.get("agent_id"),
        field=f"records[{entry.parent_slot}].agent_id",
    )
    actual_token = _exact_nonnegative_int(
        parent.get("agent_slot_token"),
        field=f"records[{entry.parent_slot}].agent_slot_token",
    )
    if actual_id != entry.parent_agent_id:
        _fail(
            f"entries[{entry.sequence_index}] has a stale parent_agent_id: "
            f"expected current {actual_id}, got {entry.parent_agent_id}"
        )
    if actual_token != entry.parent_slot_token:
        _fail(
            f"entries[{entry.sequence_index}] has a stale parent_slot_token: "
            f"expected current {actual_token}, got {entry.parent_slot_token}"
        )
    return parent


def _build_receipt(
    sequence: ModularRecurrentMutationSequenceV1,
    entry: MutationSequenceEntryV1,
) -> ModularRecurrentMutationReceiptV1:
    transaction = {
        "transaction_kind": "turnover",
        "transaction_ordinal": 0,
        "parent_agent_id": entry.parent_agent_id,
        "parent_slot": entry.parent_slot,
        "parent_slot_token": entry.parent_slot_token,
        "child_agent_id": entry.child_agent_id,
        "child_slot": entry.child_slot,
        "child_slot_token": entry.child_slot_token,
        "attempt_ordinal": 0,
        "draw0_float64_bits": entry.draw0_float64_bits,
        "draw1_float64_bits": entry.draw1_float64_bits,
        "operation": entry.operation,
        "status": entry.status,
        "reason": entry.reason,
        "selected_instance_id_index": entry.selected_instance_id_index,
        "selected_homology_id_index": entry.selected_homology_id_index,
        "changed_dimension": entry.changed_dimension,
        "changed": entry.changed,
    }
    try:
        ancestry_sha256 = turnover_mutation_ancestry_sha256_v1(
            parent_slot=entry.parent_slot,
            child_slot=entry.child_slot,
            parent_agent_id=entry.parent_agent_id,
            child_agent_id=entry.child_agent_id,
            child_slot_token=entry.child_slot_token,
        )
        return build_modular_recurrent_mutation_receipt_v1(
            profile_id=sequence.profile_id,
            source_packed_sha256=sequence.source_packed_sha256,
            ancestry_tape_sha256=ancestry_sha256,
            transactions=[transaction],
        )
    except ModularRecurrentMutationReceiptValidationError as exc:
        raise ModularRecurrentMutationSequenceReplayError(
            f"entries[{entry.sequence_index}] cannot construct a canonical receipt: {exc}"
        ) from exc


def replay_modular_recurrent_mutation_sequence_v1(
    sequence: Mapping[str, object] | ModularRecurrentMutationSequenceV1,
    initial_records: Sequence[Mapping[str, Any] | None],
    *,
    identity_table: Sequence[str],
) -> ModularRecurrentMutationSequenceReplayResultV1:
    """Replay a validated mutation journal against a fixed record sidecar."""

    try:
        to_dict = getattr(sequence, "to_dict", None)
        validated = validate_modular_recurrent_mutation_sequence_v1(
            to_dict() if callable(to_dict) else sequence
        )
    except ModularRecurrentMutationSequenceValidationError as exc:
        raise ModularRecurrentMutationSequenceReplayError(
            f"mutation sequence is invalid: {exc}"
        ) from exc

    records = _normalize_records(initial_records)
    table = _validate_identity_table(identity_table, records)
    origins: dict[int, int] = {
        int(record["agent_id"]): slot
        for slot, record in enumerate(records)
        if record is not None
    }
    entry_results: list[MutationSequenceReplayEntryResultV1] = []

    for entry in validated.entries[: validated.count]:
        if entry is None:
            _fail("validated mutation sequence contains an inactive prefix entry")
        _validate_unique_live_owners(
            records, context=f"entries[{entry.sequence_index}].input_records"
        )
        parent = _validate_parent(records, entry)
        if not 0 <= entry.child_slot < len(records):
            _fail(
                f"entries[{entry.sequence_index}] child_slot is outside the record sidecar"
            )
        if entry.child_agent_id in origins:
            _fail(
                f"entries[{entry.sequence_index}] reuses an existing or historical "
                f"child_agent_id={entry.child_agent_id}"
            )

        _, pre_sha256 = _canonical_genome(
            parent.get("genome"),
            field=f"entries[{entry.sequence_index}].parent_genome",
        )
        if not hmac.compare_digest(pre_sha256, entry.pre_genome_sha256):
            _fail(
                f"entries[{entry.sequence_index}] pre_genome_sha256 mismatch: "
                f"expected {entry.pre_genome_sha256}, got {pre_sha256}"
            )

        receipt = _build_receipt(validated, entry)
        try:
            child_genome = replay_modular_recurrent_mutation_v1(
                parent["genome"],
                receipt,
                identity_table=table,
            )
        except ModularRecurrentMutationReplayError as exc:
            raise ModularRecurrentMutationSequenceReplayError(
                f"entries[{entry.sequence_index}] mutation replay failed: {exc}"
            ) from exc

        _, post_sha256 = _canonical_genome(
            child_genome,
            field=f"entries[{entry.sequence_index}].child_genome",
        )
        if not hmac.compare_digest(post_sha256, entry.post_genome_sha256):
            _fail(
                f"entries[{entry.sequence_index}] post_genome_sha256 mismatch: "
                f"expected {entry.post_genome_sha256}, got {post_sha256}"
            )

        parent_origin = origins.get(entry.parent_agent_id)
        if parent_origin is None:
            _fail(
                f"entries[{entry.sequence_index}] parent has no stable origin mapping"
            )
        child_record = deepcopy(parent)
        child_record["agent_id"] = entry.child_agent_id
        child_record["agent_slot_token"] = entry.child_slot_token
        child_record["genome"] = deepcopy(child_genome)
        records[entry.child_slot] = child_record
        origins[entry.child_agent_id] = parent_origin

        _validate_unique_live_owners(
            records, context=f"entries[{entry.sequence_index}].result_records"
        )
        _require_identity_table_stability(
            records, table, sequence_index=entry.sequence_index
        )
        entry_results.append(
            MutationSequenceReplayEntryResultV1(
                sequence_index=entry.sequence_index,
                generation=entry.generation,
                tick=entry.tick,
                parent_slot=entry.parent_slot,
                child_slot=entry.child_slot,
                parent_agent_id=entry.parent_agent_id,
                child_agent_id=entry.child_agent_id,
                origin_slot=parent_origin,
                pre_genome_sha256=pre_sha256,
                post_genome_sha256=post_sha256,
                receipt=receipt,
                parent_record=deepcopy(parent),
                child_record=deepcopy(child_record),
            )
        )

    final_records = deepcopy(tuple(records))
    final_origins = tuple(
        None if record is None else origins[int(record["agent_id"])]
        for record in final_records
    )
    return ModularRecurrentMutationSequenceReplayResultV1(
        final_records=final_records,
        entry_results=tuple(entry_results),
        origin_slot_by_agent_id=dict(origins),
        final_origin_slot_by_record_slot=final_origins,
        identity_table=table,
    )


__all__ = [
    "ModularRecurrentMutationSequenceReplayError",
    "ModularRecurrentMutationSequenceReplayResultV1",
    "MutationSequenceReplayEntryResultV1",
    "replay_modular_recurrent_mutation_sequence_v1",
]
