"""Fail-closed causal transaction records for Genome Upgrade V2."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType
from typing import Any, Final

from engine.authoritative_state.schema import content_sha256


SCHEMA_VERSION: Final[str] = "genome_v2_causal_transaction_projection_v3"
MAX_RECORD_BYTES: Final[int] = 64 * 1024
PROVIDER_ID: Final[str] = "eiel-production-provider-v1"
RUNTIME_PROFILE_ID: Final[str] = "eiel-production-runtime-v1"


class GenomeV2CausalTransactionProjectionError(ValueError):
    """A causal transaction record or join is incomplete or contradictory."""


def _fail(message: str) -> GenomeV2CausalTransactionProjectionError:
    return GenomeV2CausalTransactionProjectionError(message)


def _exact_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise _fail(f"{name} must be a non-empty canonical string")
    return value


def _exact_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(f"{name} must be an exact integer >= {minimum}")
    return value


def _sha256(value: Any, *, name: str) -> str:
    digest = _exact_string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise _fail(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _json_value(value: Any, *, path: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _fail(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str or not key for key in value):
            raise _fail(f"{path} contains an invalid mapping key")
        return {
            key: _json_value(item, path=f"{path}.{key}")
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise _fail(f"{path} contains a non-JSON value")


def _bounded_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{name} must be a mapping")
    normalized = _json_value(value, path=name)
    assert isinstance(normalized, dict)
    if not normalized:
        raise _fail(f"{name} must not be empty")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class GenomeV2CausalTransactionIdentityV1:
    transaction_id: str
    event_kind: str
    event_sequence: int
    child_id: int
    cell_id: str
    generation: int
    tick: int
    destination_slot: int
    parent_slot: int
    mate_parent_slot: int
    parent_id: int
    mate_parent_id: int
    lineage_founder_id: int
    lineage_depth: int
    provider_id: str
    runtime_profile_id: str
    capability_contract_sha256: str
    package_id_sha256: str
    artifact_sha256: str
    execution_route_sha256: str

    @classmethod
    def create(
        cls,
        *,
        event_kind: str,
        event_sequence: int,
        child_id: int,
        cell_id: str,
        generation: int,
        tick: int,
        destination_slot: int,
        parent_slot: int,
        mate_parent_slot: int,
        parent_id: int,
        mate_parent_id: int,
        lineage_founder_id: int,
        lineage_depth: int,
        capability_contract_sha256: str,
        package_id_sha256: str,
        artifact_sha256: str,
        execution_route_sha256: str,
        provider_id: str = PROVIDER_ID,
        runtime_profile_id: str = RUNTIME_PROFILE_ID,
    ) -> GenomeV2CausalTransactionIdentityV1:
        payload = {
            "event_kind": _exact_string(event_kind, name="event_kind"),
            "event_sequence": _exact_int(
                event_sequence, name="event_sequence"
            ),
            "child_id": _exact_int(child_id, name="child_id"),
            "cell_id": _exact_string(cell_id, name="cell_id"),
            "generation": _exact_int(generation, name="generation"),
            "tick": _exact_int(tick, name="tick"),
            "destination_slot": _exact_int(
                destination_slot, name="destination_slot"
            ),
            "parent_slot": _exact_int(parent_slot, name="parent_slot"),
            "mate_parent_slot": _exact_int(
                mate_parent_slot, name="mate_parent_slot"
            ),
            "parent_id": _exact_int(parent_id, name="parent_id"),
            "mate_parent_id": _exact_int(
                mate_parent_id, name="mate_parent_id"
            ),
            "lineage_founder_id": _exact_int(
                lineage_founder_id, name="lineage_founder_id"
            ),
            "lineage_depth": _exact_int(
                lineage_depth, name="lineage_depth"
            ),
            "provider_id": _exact_string(provider_id, name="provider_id"),
            "runtime_profile_id": _exact_string(
                runtime_profile_id, name="runtime_profile_id"
            ),
            "capability_contract_sha256": _sha256(
                capability_contract_sha256,
                name="capability_contract_sha256",
            ),
            "package_id_sha256": _sha256(
                package_id_sha256, name="package_id_sha256"
            ),
            "artifact_sha256": _sha256(
                artifact_sha256, name="artifact_sha256"
            ),
            "execution_route_sha256": _sha256(
                execution_route_sha256,
                name="execution_route_sha256",
            ),
        }
        if payload["provider_id"] != PROVIDER_ID:
            raise _fail("provider_id is not the production provider")
        if payload["runtime_profile_id"] != RUNTIME_PROFILE_ID:
            raise _fail("runtime_profile_id is not the production runtime")
        if payload["mate_parent_id"] == payload["parent_id"]:
            raise _fail("mate_parent_id must differ from parent_id")
        transaction_id = content_sha256(
            {"schema_version": SCHEMA_VERSION, **payload}
        )
        return cls(transaction_id=transaction_id, **payload)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> GenomeV2CausalTransactionIdentityV1:
        if not isinstance(value, Mapping):
            raise _fail("identity must be a mapping")
        expected = {field.name for field in fields(cls)}
        if set(value) != expected:
            raise _fail("identity fields differ")
        transaction_id = _sha256(
            value["transaction_id"], name="transaction_id"
        )
        created = cls.create(
            **{
                name: value[name]
                for name in expected
                if name != "transaction_id"
            }
        )
        if created.transaction_id != transaction_id:
            raise _fail("transaction_id contradicts its content address")
        return created

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenomeV2CausalTransactionRecordV1:
    schema_version: str
    record_phase: str
    identity: GenomeV2CausalTransactionIdentityV1
    genotype_summary: Mapping[str, Any]
    product_summary: Mapping[str, Any]
    regulatory_summary: Mapping[str, Any]
    context_summary: Mapping[str, Any]
    phenotype_summary: Mapping[str, Any]
    cost_summary: Mapping[str, Any]
    applied_consequence: Mapping[str, Any]
    resulting_state: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        *,
        record_phase: str,
        identity: GenomeV2CausalTransactionIdentityV1 | Mapping[str, Any],
        genotype_summary: Mapping[str, Any],
        product_summary: Mapping[str, Any],
        regulatory_summary: Mapping[str, Any],
        context_summary: Mapping[str, Any],
        phenotype_summary: Mapping[str, Any],
        cost_summary: Mapping[str, Any],
        applied_consequence: Mapping[str, Any],
        resulting_state: Mapping[str, Any],
    ) -> GenomeV2CausalTransactionRecordV1:
        if record_phase not in {"event_time", "terminal"}:
            raise _fail("record_phase must be event_time or terminal")
        resolved_identity = (
            identity
            if isinstance(identity, GenomeV2CausalTransactionIdentityV1)
            else GenomeV2CausalTransactionIdentityV1.from_mapping(identity)
        )
        record = cls(
            schema_version=SCHEMA_VERSION,
            record_phase=record_phase,
            identity=resolved_identity,
            genotype_summary=_bounded_mapping(
                genotype_summary, name="genotype_summary"
            ),
            product_summary=_bounded_mapping(
                product_summary, name="product_summary"
            ),
            regulatory_summary=_bounded_mapping(
                regulatory_summary, name="regulatory_summary"
            ),
            context_summary=_bounded_mapping(
                context_summary, name="context_summary"
            ),
            phenotype_summary=_bounded_mapping(
                phenotype_summary, name="phenotype_summary"
            ),
            cost_summary=_bounded_mapping(
                cost_summary, name="cost_summary"
            ),
            applied_consequence=_bounded_mapping(
                applied_consequence, name="applied_consequence"
            ),
            resulting_state=_bounded_mapping(
                resulting_state, name="resulting_state"
            ),
        )
        encoded = json.dumps(
            record.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        if len(encoded) > MAX_RECORD_BYTES:
            raise _fail("causal transaction record exceeds its size bound")
        return record

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> GenomeV2CausalTransactionRecordV1:
        if not isinstance(value, Mapping):
            raise _fail("causal transaction record must be a mapping")
        expected = {
            "schema_version",
            "record_phase",
            "identity",
            "genotype_summary",
            "product_summary",
            "regulatory_summary",
            "context_summary",
            "phenotype_summary",
            "cost_summary",
            "applied_consequence",
            "resulting_state",
        }
        if set(value) != expected or value.get("schema_version") != SCHEMA_VERSION:
            raise _fail("causal transaction record fields differ")
        return cls.create(
            record_phase=value["record_phase"],
            identity=value["identity"],
            genotype_summary=value["genotype_summary"],
            product_summary=value["product_summary"],
            regulatory_summary=value["regulatory_summary"],
            context_summary=value["context_summary"],
            phenotype_summary=value["phenotype_summary"],
            cost_summary=value["cost_summary"],
            applied_consequence=value["applied_consequence"],
            resulting_state=value["resulting_state"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_phase": self.record_phase,
            "identity": self.identity.as_dict(),
            "genotype_summary": dict(self.genotype_summary),
            "product_summary": dict(self.product_summary),
            "regulatory_summary": dict(self.regulatory_summary),
            "context_summary": dict(self.context_summary),
            "phenotype_summary": dict(self.phenotype_summary),
            "cost_summary": dict(self.cost_summary),
            "applied_consequence": dict(self.applied_consequence),
            "resulting_state": dict(self.resulting_state),
        }


@dataclass(frozen=True)
class GenomeV2CausalTransactionJoinV1:
    transaction_id: str
    event_time: GenomeV2CausalTransactionRecordV1
    terminal: GenomeV2CausalTransactionRecordV1


def _records_by_transaction_id(
    records: Sequence[
        GenomeV2CausalTransactionRecordV1 | Mapping[str, Any]
    ],
    *,
    phase: str,
) -> dict[str, GenomeV2CausalTransactionRecordV1]:
    projected: dict[str, GenomeV2CausalTransactionRecordV1] = {}
    natural_keys: set[tuple[Any, ...]] = set()
    for raw in records:
        record = GenomeV2CausalTransactionRecordV1.from_mapping(
            raw.as_dict()
            if isinstance(raw, GenomeV2CausalTransactionRecordV1)
            else raw
        )
        if record.record_phase != phase:
            raise _fail(f"{phase} input contains a {record.record_phase} record")
        identity = record.identity
        if identity.transaction_id in projected:
            raise _fail(
                f"duplicate {phase} transaction_id: {identity.transaction_id}"
            )
        natural_key = (
            identity.cell_id,
            identity.child_id,
            identity.generation,
            identity.tick,
            identity.destination_slot,
        )
        if natural_key in natural_keys:
            raise _fail(f"ambiguous {phase} natural transaction identity")
        natural_keys.add(natural_key)
        projected[identity.transaction_id] = record
    return projected


def join_causal_transaction_records_v1(
    event_time_records: Sequence[
        GenomeV2CausalTransactionRecordV1 | Mapping[str, Any]
    ],
    terminal_records: Sequence[
        GenomeV2CausalTransactionRecordV1 | Mapping[str, Any]
    ],
) -> tuple[GenomeV2CausalTransactionJoinV1, ...]:
    """Require a complete, exact, one-to-one event-time/terminal join."""

    event_time = _records_by_transaction_id(
        event_time_records, phase="event_time"
    )
    terminal = _records_by_transaction_id(
        terminal_records, phase="terminal"
    )
    if not event_time and not terminal:
        raise _fail("causal transaction join has no evaluable records")
    missing_terminal = sorted(set(event_time) - set(terminal))
    missing_event_time = sorted(set(terminal) - set(event_time))
    if missing_terminal or missing_event_time:
        raise _fail(
            "causal transaction join is incomplete: "
            f"missing_terminal={missing_terminal} "
            f"missing_event_time={missing_event_time}"
        )
    joined: list[GenomeV2CausalTransactionJoinV1] = []
    for transaction_id in sorted(event_time):
        event_record = event_time[transaction_id]
        terminal_record = terminal[transaction_id]
        if event_record.identity != terminal_record.identity:
            raise _fail(
                f"causal transaction identity contradicts: {transaction_id}"
            )
        joined.append(
            GenomeV2CausalTransactionJoinV1(
                transaction_id=transaction_id,
                event_time=event_record,
                terminal=terminal_record,
            )
        )
    return tuple(joined)


__all__ = [
    "GenomeV2CausalTransactionIdentityV1",
    "GenomeV2CausalTransactionJoinV1",
    "GenomeV2CausalTransactionProjectionError",
    "GenomeV2CausalTransactionRecordV1",
    "MAX_RECORD_BYTES",
    "PROVIDER_ID",
    "RUNTIME_PROFILE_ID",
    "SCHEMA_VERSION",
    "join_causal_transaction_records_v1",
]
