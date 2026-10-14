"""Pure adapter boundary for canonical CUDA V2 science observations.

The CUDA provider is allowed to publish a route-neutral observation envelope at
this boundary.  This module only validates and lowers that envelope into the
canonical science draft/finalization contracts; it does not inspect tensors,
derive action semantics, or perform any execution effect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from emergence_contracts._validation import require_instance, require_int, require_text
from emergence_contracts.errors import contract_invalid
from emergence_contracts.ids import ContentDigest, SchemaVersion
from emergence_contracts.payloads import CanonicalPayloadV1
from emergence_contracts.science_instrument import (
    _FORBIDDEN_ACTIVE_TIME_KEYS,
    _FORBIDDEN_SEMANTIC_ROUTE_KEYS,
    _record_required_payload_fields,
    ScienceObservationRecordV1,
    ScienceRecordKindV1,
)
from emergence_contracts.science_observations import (
    ScienceObservationDraftV1,
    finalize_science_observation_drafts_v1,
)


SCIENCE_NATIVE_OBSERVATION_ADAPTER_SCHEMA_VERSION: Final = SchemaVersion(
    "eiel_science_native_observation_adapter_v1"
)
SCIENCE_NATIVE_OBSERVATION_OUTPUT_SCHEMA_VERSION: Final = SchemaVersion(
    "eiel_science_native_observation_output_v1"
)

_DRAFT_REQUIRED_FIELDS: Final = frozenset(
    {
        "producer_component_id",
        "producer_contract_version",
        "producer_phase_registry_ordinal",
        "absolute_tick",
        "semantic_source_address",
        "record_kind",
        "record_kind_version",
        "control_binding_sha256",
        "payload",
    }
)
_DRAFT_OPTIONAL_FIELDS: Final = frozenset(
    {
        "source_state_before_sha256",
        "source_state_after_sha256",
        "cause_event_ids",
        "lifecycle_epoch",
        "legacy_compatibility",
    }
)
_DRAFT_FIELDS: Final = _DRAFT_REQUIRED_FIELDS | _DRAFT_OPTIONAL_FIELDS
_OUTPUT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "event_dispatch",
        "state_projection",
        "checkpoint_projection",
        "effects_performed",
    }
)

# These are the only keys which are meaningful to this adapter.  Route and
# physical scheduling identity belongs to the execution envelope, never to a
# science observation row.
_FORBIDDEN_BOUNDARY_FIELDS: Final = frozenset(
    set(_FORBIDDEN_SEMANTIC_ROUTE_KEYS)
    | {
        "backend_id",
        "batch_index",
        "batch_size",
        "carrier_profile_id",
        "cuda_device",
        "device",
        "native_route_id",
        "owner_id",
        "physical_batch_index",
        "physical_batch_size",
        "physical_slot",
        "provider_family",
        "provider_id",
        "route_id",
        "slot",
        "temporal_k",
    }
)
_EVENT_DISPATCH_EXCLUDED_KINDS: Final = frozenset(
    {
        ScienceRecordKindV1.WORLD_STATE_SNAPSHOT,
        ScienceRecordKindV1.AGENT_STATE_SNAPSHOT,
        ScienceRecordKindV1.CHECKPOINT_MARKER,
    }
)
_STATE_PROJECTION_KINDS: Final = frozenset(
    {
        ScienceRecordKindV1.CONTROL_BINDING,
        ScienceRecordKindV1.WORLD_STATE_SNAPSHOT,
        ScienceRecordKindV1.AGENT_STATE_SNAPSHOT,
        ScienceRecordKindV1.DEATH,
        ScienceRecordKindV1.ENVIRONMENT_TRANSITION,
        ScienceRecordKindV1.WORLD_COMPLETION,
    }
)


def _invalid(message: str, *, field: str) -> None:
    raise contract_invalid(message, field=field)


def _as_schema_version(value: object, *, field: str) -> SchemaVersion:
    if isinstance(value, SchemaVersion):
        return value
    if isinstance(value, str):
        try:
            return SchemaVersion(value)
        except Exception as exc:  # pragma: no cover - constructor owns detail
            _invalid(f"invalid schema version: {exc}", field=field)
    _invalid("expected SchemaVersion or string", field=field)


def _as_digest(value: object, *, field: str) -> ContentDigest:
    if isinstance(value, ContentDigest):
        return value
    if not isinstance(value, Mapping) or set(value) != {"algorithm", "hex"}:
        _invalid("expected ContentDigest or its exact wire form", field=field)
    algorithm = value.get("algorithm")
    digest_hex = value.get("hex")
    if not isinstance(algorithm, str) or not isinstance(digest_hex, str):
        _invalid("digest algorithm and hex must be strings", field=field)
    try:
        return ContentDigest(algorithm, digest_hex)
    except Exception as exc:  # pragma: no cover - constructor owns detail
        _invalid(f"invalid content digest: {exc}", field=field)


def _as_payload(value: object, *, field: str) -> CanonicalPayloadV1:
    if isinstance(value, CanonicalPayloadV1):
        return value
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "payload"}:
        _invalid("expected CanonicalPayloadV1 or its exact wire form", field=field)
    schema_version = _as_schema_version(value.get("schema_version"), field=f"{field}.schema_version")
    try:
        return CanonicalPayloadV1.from_value(schema_version, value.get("payload"))
    except Exception as exc:
        _invalid(f"payload is not canonical JSON: {exc}", field=field)


def _as_optional_digest(value: object, *, field: str) -> ContentDigest | None:
    if value is None:
        return None
    return _as_digest(value, field=field)


def _as_cause_event_ids(value: object, *, field: str) -> tuple[ContentDigest, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _invalid("expected a sequence of ContentDigest values", field=field)
    return tuple(_as_digest(item, field=f"{field}[{index}]") for index, item in enumerate(value))


def _as_rows(value: object, *, field: str) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _invalid("expected a sequence of observation row mappings", field=field)
    rows: list[Mapping[str, object]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            _invalid("every observation row must be a mapping", field=f"{field}[{index}]")
        rows.append(row)
    return tuple(rows)


def _validate_payload_fields(
    kind: ScienceRecordKindV1,
    payload: CanonicalPayloadV1,
    *,
    field: str,
) -> None:
    payload_value = payload.to_value()
    if not isinstance(payload_value, Mapping):
        _invalid("science payload must be an object", field=field)
    actual = frozenset(payload_value)
    expected = frozenset(_record_required_payload_fields(kind, payload))
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    detail = f"missing={missing!r}, unknown={unknown!r}"
    if kind is ScienceRecordKindV1.DECISION_ACTION:
        _invalid(
            "MISSING_OBSERVATION_CAPABILITY: decision_action is accepted only "
            f"with explicit selected/executed action and causal fields ({detail})",
            field=field,
        )
    _invalid(f"payload field set does not match {kind.value}: {detail}", field=field)


def _reject_nested_boundary_fields(value: object, *, field: str) -> None:
    """Reject route/physical identity even when nested in a semantic value."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_ACTIVE_TIME_KEYS:
                _invalid(
                    "generation-shaped public time authority is forbidden",
                    field=f"{field}.{key}",
                )
            if key in _FORBIDDEN_BOUNDARY_FIELDS or key.endswith("_route_binding_sha256"):
                _invalid(
                    f"route/physical fields cannot enter semantic observations: {key!r}",
                    field=f"{field}.{key}",
                )
            _reject_nested_boundary_fields(item, field=f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nested_boundary_fields(item, field=f"{field}[{index}]")


def _draft_from_row(row: Mapping[str, object], *, field: str) -> ScienceObservationDraftV1:
    row_keys = set(row)
    boundary_fields = sorted(row_keys & _FORBIDDEN_BOUNDARY_FIELDS)
    if boundary_fields:
        _invalid(
            f"route/physical fields cannot enter semantic observations: {boundary_fields!r}",
            field=field,
        )
    missing = sorted(_DRAFT_REQUIRED_FIELDS - row_keys)
    unknown = sorted(row_keys - _DRAFT_FIELDS)
    if missing:
        _invalid(f"observation row is missing required fields: {missing!r}", field=field)
    if unknown:
        _invalid(f"observation row has unsupported fields: {unknown!r}", field=field)

    producer_component_id = require_text(
        row["producer_component_id"], field=f"{field}.producer_component_id"
    )
    producer_contract_version = _as_schema_version(
        row["producer_contract_version"], field=f"{field}.producer_contract_version"
    )
    phase_ordinal = require_int(
        row["producer_phase_registry_ordinal"],
        field=f"{field}.producer_phase_registry_ordinal",
        minimum=0,
    )
    absolute_tick = require_int(row["absolute_tick"], field=f"{field}.absolute_tick", minimum=0)
    semantic_source_address = require_text(
        row["semantic_source_address"], field=f"{field}.semantic_source_address"
    )
    try:
        record_kind = ScienceRecordKindV1(row["record_kind"])
    except (TypeError, ValueError) as exc:
        _invalid(f"unknown science record kind: {exc}", field=f"{field}.record_kind")
    record_kind_version = require_int(
        row["record_kind_version"], field=f"{field}.record_kind_version", minimum=1
    )
    if record_kind_version != 1:
        _invalid("only record_kind_version 1 is supported", field=f"{field}.record_kind_version")
    control_binding_sha256 = _as_digest(
        row["control_binding_sha256"], field=f"{field}.control_binding_sha256"
    )
    payload = _as_payload(row["payload"], field=f"{field}.payload")
    _validate_payload_fields(record_kind, payload, field=f"{field}.payload")
    _reject_nested_boundary_fields(payload.to_value(), field=f"{field}.payload")

    lifecycle_epoch = row.get("lifecycle_epoch")
    if lifecycle_epoch is not None:
        lifecycle_epoch = require_int(
            lifecycle_epoch, field=f"{field}.lifecycle_epoch", minimum=0
        )
    legacy_compatibility = row.get("legacy_compatibility")
    if legacy_compatibility is not None:
        legacy_compatibility = _as_payload(
            legacy_compatibility, field=f"{field}.legacy_compatibility"
        )
    return ScienceObservationDraftV1(
        producer_component_id=producer_component_id,
        producer_contract_version=producer_contract_version,
        producer_phase_registry_ordinal=phase_ordinal,
        absolute_tick=absolute_tick,
        semantic_source_address=semantic_source_address,
        record_kind=record_kind,
        record_kind_version=record_kind_version,
        control_binding_sha256=control_binding_sha256,
        payload=payload,
        source_state_before_sha256=_as_optional_digest(
            row.get("source_state_before_sha256"), field=f"{field}.source_state_before_sha256"
        ),
        source_state_after_sha256=_as_optional_digest(
            row.get("source_state_after_sha256"), field=f"{field}.source_state_after_sha256"
        ),
        cause_event_ids=_as_cause_event_ids(row.get("cause_event_ids"), field=f"{field}.cause_event_ids"),
        lifecycle_epoch=lifecycle_epoch,
        legacy_compatibility=legacy_compatibility,
    )


@dataclass(frozen=True, slots=True)
class NativeCudaV2ObservationContextV1:
    """Semantic bindings required by canonical record finalization."""

    instrument_manifest_sha256: ContentDigest
    crca_semantic_run_digest: ContentDigest
    science_semantic_acquisition_sha256: ContentDigest
    cell_id: str
    logical_world_id: int
    semantic_world_key: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_instance(
            self.instrument_manifest_sha256, ContentDigest, field="instrument_manifest_sha256"
        )
        require_instance(
            self.crca_semantic_run_digest, ContentDigest, field="crca_semantic_run_digest"
        )
        require_instance(
            self.science_semantic_acquisition_sha256,
            ContentDigest,
            field="science_semantic_acquisition_sha256",
        )
        require_text(self.cell_id, field="cell_id")
        require_int(self.logical_world_id, field="logical_world_id", minimum=0)
        require_instance(self.semantic_world_key, CanonicalPayloadV1, field="semantic_world_key")


@dataclass(frozen=True, slots=True)
class NativeCudaV2ObservationOutputV1:
    """Versioned, route-neutral provider output accepted by the adapter."""

    event_dispatch: tuple[Mapping[str, object], ...] = ()
    state_projection: tuple[Mapping[str, object], ...] = ()
    checkpoint_projection: tuple[Mapping[str, object], ...] = ()
    effects_performed: bool = False
    schema_version: SchemaVersion = SCIENCE_NATIVE_OBSERVATION_OUTPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.schema_version, SchemaVersion, field="schema_version")
        if self.schema_version != SCIENCE_NATIVE_OBSERVATION_OUTPUT_SCHEMA_VERSION:
            _invalid("unsupported native observation output schema", field="schema_version")
        for field in ("event_dispatch", "state_projection", "checkpoint_projection"):
            _as_rows(getattr(self, field), field=field)
        if self.effects_performed is not False:
            _invalid(
                "native observation projection is read-only and cannot report effects_performed",
                field="effects_performed",
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "NativeCudaV2ObservationOutputV1":
        if not isinstance(value, Mapping):
            _invalid("expected native observation output mapping", field="output")
        missing = sorted(_OUTPUT_FIELDS - set(value))
        unknown = sorted(set(value) - _OUTPUT_FIELDS)
        if missing:
            _invalid(f"native observation output is missing fields: {missing!r}", field="output")
        if unknown:
            _invalid(f"native observation output has unsupported fields: {unknown!r}", field="output")
        schema_version = _as_schema_version(value["schema_version"], field="output.schema_version")
        return cls(
            event_dispatch=_as_rows(value["event_dispatch"], field="output.event_dispatch"),
            state_projection=_as_rows(value["state_projection"], field="output.state_projection"),
            checkpoint_projection=_as_rows(
                value["checkpoint_projection"], field="output.checkpoint_projection"
            ),
            effects_performed=value["effects_performed"],
            schema_version=schema_version,
        )


def _project_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    field: str,
    allowed_kinds: frozenset[ScienceRecordKindV1] | None = None,
    excluded_kinds: frozenset[ScienceRecordKindV1] = frozenset(),
) -> tuple[ScienceObservationDraftV1, ...]:
    drafts: list[ScienceObservationDraftV1] = []
    for index, row in enumerate(rows):
        draft = _draft_from_row(row, field=f"{field}[{index}]")
        if allowed_kinds is not None and draft.record_kind not in allowed_kinds:
            _invalid(
                f"record kind {draft.record_kind.value!r} is not a state/checkpoint projection kind",
                field=f"{field}[{index}].record_kind",
            )
        if draft.record_kind in excluded_kinds:
            _invalid(
                f"record kind {draft.record_kind.value!r} belongs to state/checkpoint projection",
                field=f"{field}[{index}].record_kind",
            )
        drafts.append(draft)
    return tuple(drafts)


def project_native_cuda_v2_event_dispatch_v1(
    rows: Sequence[Mapping[str, object]],
) -> tuple[ScienceObservationDraftV1, ...]:
    """Lower explicit provider event rows without inventing causal fields."""

    return _project_rows(rows, field="event_dispatch", excluded_kinds=_EVENT_DISPATCH_EXCLUDED_KINDS)


def project_native_cuda_v2_state_checkpoint_v1(
    state_rows: Sequence[Mapping[str, object]],
    checkpoint_rows: Sequence[Mapping[str, object]] = (),
) -> tuple[ScienceObservationDraftV1, ...]:
    """Lower explicit state rows and checkpoint markers.

    A checkpoint projection is not a tensor dump.  It must already contain a
    complete canonical ``CHECKPOINT_MARKER`` payload, so raw CUDA checkpoint
    maps fail closed as unsupported input.
    """

    state = _project_rows(state_rows, field="state_projection", allowed_kinds=_STATE_PROJECTION_KINDS)
    checkpoints = _project_rows(
        checkpoint_rows,
        field="checkpoint_projection",
        allowed_kinds=frozenset({ScienceRecordKindV1.CHECKPOINT_MARKER}),
    )
    return state + checkpoints


def _as_output(
    output: NativeCudaV2ObservationOutputV1 | Mapping[str, object],
) -> NativeCudaV2ObservationOutputV1:
    if isinstance(output, NativeCudaV2ObservationOutputV1):
        return output
    if isinstance(output, Mapping):
        return NativeCudaV2ObservationOutputV1.from_mapping(output)
    _invalid("expected NativeCudaV2ObservationOutputV1 or exact wire mapping", field="output")


def project_native_cuda_v2_observation_drafts_v1(
    output: NativeCudaV2ObservationOutputV1 | Mapping[str, object],
) -> tuple[ScienceObservationDraftV1, ...]:
    """Project the complete provider envelope into canonical drafts."""

    normalized = _as_output(output)
    return project_native_cuda_v2_event_dispatch_v1(normalized.event_dispatch) + project_native_cuda_v2_state_checkpoint_v1(
        normalized.state_projection, normalized.checkpoint_projection
    )


def finalize_native_cuda_v2_observations_v1(
    output: NativeCudaV2ObservationOutputV1 | Mapping[str, object],
    *,
    context: NativeCudaV2ObservationContextV1,
    start_world_local_sequence: int = 0,
) -> tuple[ScienceObservationRecordV1, ...]:
    """Finalize adapter drafts through the canonical identity/order authority."""

    require_instance(context, NativeCudaV2ObservationContextV1, field="context")
    require_int(start_world_local_sequence, field="start_world_local_sequence", minimum=0)
    drafts = project_native_cuda_v2_observation_drafts_v1(output)
    return finalize_science_observation_drafts_v1(
        drafts,
        instrument_manifest_sha256=context.instrument_manifest_sha256,
        crca_semantic_run_digest=context.crca_semantic_run_digest,
        science_semantic_acquisition_sha256=context.science_semantic_acquisition_sha256,
        cell_id=context.cell_id,
        logical_world_id=context.logical_world_id,
        semantic_world_key=context.semantic_world_key,
        start_world_local_sequence=start_world_local_sequence,
    )


# Descriptive aliases for callers that use "adapt"/"output" terminology.
adapt_native_cuda_v2_output_v1 = project_native_cuda_v2_observation_drafts_v1
finalize_native_cuda_v2_observation_output_v1 = finalize_native_cuda_v2_observations_v1


__all__ = [
    "SCIENCE_NATIVE_OBSERVATION_ADAPTER_SCHEMA_VERSION",
    "SCIENCE_NATIVE_OBSERVATION_OUTPUT_SCHEMA_VERSION",
    "NativeCudaV2ObservationContextV1",
    "NativeCudaV2ObservationOutputV1",
    "adapt_native_cuda_v2_output_v1",
    "finalize_native_cuda_v2_observation_output_v1",
    "finalize_native_cuda_v2_observations_v1",
    "project_native_cuda_v2_event_dispatch_v1",
    "project_native_cuda_v2_observation_drafts_v1",
    "project_native_cuda_v2_state_checkpoint_v1",
]
