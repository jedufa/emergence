"""Opaque science sidecar and checkpoint-continuation gate for CUDA V2.

The CUDA hot path does not import or parse this module.  Observation ``OFF``
therefore remains the only runtime-admitted mode and has no science allocation
or hook.  The additive sidecar implements the exact wire boundary required by
the pinned EIEL Science Instrument V1 contracts: complete route bindings,
canonical observation records, per-world prefix cursors, bounded typed capture,
and parent-aware checkpoint restoration.

This is an architecture/host-adapter seam.  The typed sidecar is now wired to
the engineering CUDA V2 direct-sparse owner through the additive capture
adapter, while the public extension admission remains deliberately separate
and OFF-only.  The integration does not claim that observation ON is
trajectory-neutral beyond the bounded engineering proof, or that the route is
science-qualified.

Producer-supplied ``pre_sequence_key_sha256`` and ``deterministic_event_id``
remain DTO-compatibility fields at this seam.  Shape validation here is not an
independent replay of, or proof against, the pinned canonical draft-finalization
algorithm.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any, Final, Mapping, Sequence


SCIENCE_EXTENSION_CONTRACT_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_science_extension_contract_v1"
)
SCIENCE_CHECKPOINT_EXTENSION_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_science_checkpoint_extension_v1"
)
SCIENCE_CHECKPOINT_IDENTITY_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_science_checkpoint_identity_v1"
)
SCIENCE_TYPED_CAPTURE_INVENTORY_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_typed_science_capture_inventory_v1"
)
SCIENCE_CURSOR_SET_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_science_cursor_set_v1"
)
SCIENCE_RESTORE_GATE_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_science_restore_gate_v1"
)
SCIENCE_ROUTE_BINDING_ADAPTER_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_science_route_binding_adapter_v1"
)
SCIENCE_ROUTE_BINDING_SCHEMA_V1: Final[str] = "ScienceRouteBindingV1"
SCIENCE_ROUTE_BINDING_WIRE_SCHEMA_V1: Final[str] = "eiel_science_route_binding_v1"
SCIENCE_OBSERVATION_RECORD_SCHEMA_V1: Final[str] = "eiel_science_observation_record_v1"
SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_V1: Final[str] = (
    "eiel_science_native_control_binding_payload_v1"
)
SCIENCE_NATIVE_AGENT_STATE_SNAPSHOT_PAYLOAD_SCHEMA_V1: Final[str] = (
    "eiel_science_native_agent_state_snapshot_payload_v1"
)
SCIENCE_OBSERVATION_PREFIX_CURSOR_SCHEMA_V1: Final[str] = (
    "eiel_science_observation_prefix_cursor_v1"
)
SCIENCE_OBSERVATION_PREFIX_CHAIN_SCHEMA_V1: Final[str] = (
    "eiel_science_observation_prefix_chain_v1"
)
SCIENCE_SEMANTIC_EXECUTION_ID_SCHEMA_V1: Final[str] = (
    "eiel_science_semantic_execution_identity_v1"
)
SCIENCE_OBSERVATION_OFF_V1: Final[str] = "OFF"
SCIENCE_OBSERVATION_TYPED_SIDECAR_V1: Final[str] = "TYPED_SIDECAR_V1"
SCIENCE_STATIC_CONTROL_TABLE_V1: Final[str] = (
    "STATIC_PER_WORLD_ABSOLUTE_TICK_CONTROL_TABLE_V1"
)
SCIENCE_LIFECYCLE_PROFILE_V1: Final[str] = "SCIENCE_COMPACT"
SCIENCE_EXTENSION_STATUS_V1: Final[str] = (
    "OPAQUE_CHECKPOINT_RESTORE_SEAM_NATIVE_CAPTURE_NOT_INTEGRATED"
)
SCIENCE_TYPED_RECORD_KINDS_V1: Final[tuple[str, ...]] = (
    "control_binding",
    "world_state_snapshot",
    "agent_state_snapshot",
    "reproduction_opportunity",
    "birth_attempt",
    "birth_committed",
    "mutation_outcome",
    "decision_action",
    "ecological_transaction",
    "death",
    "environment_transition",
    "world_completion",
    "checkpoint_marker",
)
SCIENCE_WORLD_COMPLETION_CLASSES_V1: Final[frozenset[str]] = frozenset(
    {"EMPTY_INITIAL", "EXTINCT", "NO_BIRTHS", "TERMINAL_HORIZON"}
)
SCIENCE_CONTRACT_RELATIVE_PATH_V1: Final[str] = (
    "_artifacts/tw-cuda-full-evolution-reform-20260825/"
    "science_consumer_extension_contract.json"
)

_SHA256_FIELDS_ROUTE_BINDING = (
    "instrument_manifest_sha256",
    "route_profile_sha256",
    "capability_projection_sha256",
    "native_build_closure_sha256",
    "execution_adapter_sha256",
    "control_adapter_sha256",
    "observation_adapter_sha256",
    "checkpoint_adapter_sha256",
    "semantic_conformance_sha256",
    "resource_envelope_sha256",
    "qualified_envelope_sha256",
    "qualification_authority_sha256",
)
_ROUTE_BINDING_FIELDS = {
    "schema_version",
    "route_id",
    "machine_profile_id",
    *_SHA256_FIELDS_ROUTE_BINDING,
}
_CURSOR_FIELDS = {
    "schema_version",
    "instrument_manifest_sha256",
    "semantic_execution_id",
    "crca_semantic_run_digest",
    "science_semantic_acquisition_sha256",
    "cell_id",
    "logical_world_id",
    "world_uid",
    "checkpoint_at_tick",
    "next_absolute_tick",
    "next_world_local_sequence",
    "prefix_record_count",
    "prefix_chain_sha256",
    "last_deterministic_event_id",
    "previous_cursor_sha256",
    "effects_performed",
}
_RECORD_FIELDS = {
    "schema_version",
    "instrument_manifest_sha256",
    "producer_component_id",
    "producer_contract_version",
    "semantic_execution_id",
    "crca_semantic_run_digest",
    "science_semantic_acquisition_sha256",
    "cell_id",
    "logical_world_id",
    "world_uid",
    "absolute_tick",
    "semantic_source_address",
    "pre_sequence_key_sha256",
    "world_local_sequence",
    "deterministic_event_id",
    "record_kind",
    "record_kind_version",
    "control_binding_sha256",
    "payload",
    "source_state_before_sha256",
    "source_state_after_sha256",
    "cause_event_ids",
    "lifecycle_epoch",
    "legacy_compatibility",
}
_RECORD_REQUIRED_PAYLOAD_FIELDS: Final[dict[str, frozenset[str]]] = {
    "control_binding": frozenset(
        {
            "binding_id",
            "semantic_cell_id",
            "seed_bindings",
            "control_values",
            "control_schedules",
            "registered_profile_ids",
            "semantic_envelope_resolution_sha256",
            "semantic_applied_readback_sha256",
            "continuation_parent_execution_identity",
        }
    ),
    "world_state_snapshot": frozenset(
        {
            "world_state_id",
            "field_set_version",
            "qualified_field_values",
            "snapshot_reason",
        }
    ),
    "agent_state_snapshot": frozenset(
        {
            "agent_state_id",
            "agent_id",
            "alive",
            "position",
            "energy",
            "health",
            "injury",
            "age_ticks",
            "genome_state_id",
            "policy_id",
            "founder_id",
            "lineage_id",
            "lineage_depth",
            "hereditary_package_state_ids",
            "linkage_state_id",
        }
    ),
    "reproduction_opportunity": frozenset(
        {
            "opportunity_id",
            "subject_agent_id",
            "subject_agent_state_id",
            "eligible",
            "candidate_parent_ids",
            "selected_parent_ids",
            "selection_mode_id",
            "outcome_class",
            "reason_code",
        }
    ),
    "birth_attempt": frozenset(
        {
            "birth_attempt_id",
            "opportunity_id",
            "parent_agent_ids",
            "parent_genome_state_ids",
            "birth_mode_id",
            "inheritance_operator_id",
            "recombination_operator_id",
            "attempt_result_class",
            "reason_code",
        }
    ),
    "birth_committed": frozenset(
        {
            "birth_attempt_id",
            "opportunity_id",
            "child_agent_id",
            "parent_agent_ids",
            "parent_genome_state_ids",
            "birth_mode_id",
            "inheritance_operator_id",
            "recombination_operator_id",
            "parent_hereditary_package_state_ids",
            "child_pre_mutation_genome_state_id",
            "child_post_birth_genome_state_id",
            "child_hereditary_package_state_ids",
            "child_linkage_state_id",
            "founder_id",
            "lineage_id",
            "lineage_depth",
        }
    ),
    "mutation_outcome": frozenset(
        {
            "mutation_opportunity_id",
            "mutation_attempt_id",
            "subject_agent_id",
            "origin_agent_id",
            "origin_absolute_tick",
            "attempted",
            "result_class",
            "operator_id",
            "mutation_class_id",
            "affected_coordinates",
            "pre_value_or_state_digest",
            "post_value_or_state_digest",
            "bounded_delta",
            "pre_genome_state_id",
            "post_genome_state_id",
            "reason_code",
        }
    ),
    "decision_action": frozenset(
        {
            "decision_id",
            "agent_id",
            "agent_state_before_id",
            "genome_state_id",
            "policy_id",
            "decision_input_state_id",
            "qualified_policy_output_summary",
            "selected_action_id",
            "executed_action_id",
            "execution_status",
            "execution_reason_code",
        }
    ),
    "ecological_transaction": frozenset(
        {
            "transaction_id",
            "decision_id",
            "actor_agent_id",
            "affected_subject_ids",
            "executed_action_id",
            "phenotype_state_id",
            "world_state_before_id",
            "world_state_after_id",
            "resource_deltas",
            "energy_delta",
            "health_delta",
            "injury_delta",
            "movement_cost",
            "outcome_class",
        }
    ),
    "death": frozenset(
        {
            "agent_id",
            "terminal_agent_state_id",
            "genome_state_id",
            "lineage_id",
            "cause_class",
            "reason_code",
            "causing_event_ids",
        }
    ),
    "environment_transition": frozenset(
        {
            "transition_id",
            "transition_class",
            "schedule_binding_id",
            "world_state_before_id",
            "world_state_after_id",
            "changed_fields",
            "cause_event_ids",
            "persistent",
        }
    ),
    "world_completion": frozenset(
        {
            "completion_id",
            "completion_class",
            "terminal_state_sha256",
            "terminal_population_count",
            "terminal_absolute_tick",
            "next_absolute_tick",
            "terminal_record_count",
            "canonical_record_stream_sha256",
            "checkpoint_occurrence_ids",
            "reason_code",
        }
    ),
    "checkpoint_marker": frozenset(
        {
            "checkpoint_occurrence_id",
            "checkpoint_contract_id",
            "authoritative_state_sha256",
            "observation_prefix_sha256",
            "next_world_local_sequence",
            "continuation_parent_identity",
        }
    ),
}
_FORBIDDEN_TIME_KEYS = frozenset(
    {
        "generation",
        "generations",
        "generation_label",
        "generation_index",
        "ticks_per_generation",
        "physical_window_generations",
        "checkpoint_generation",
        "checkpoint_generations",
        "start_generation",
        "end_generation",
        "refill_window_generations",
    }
)
_SCHEMA_VERSION_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_ROUTE_TOKEN_RE = _SCHEMA_VERSION_TOKEN_RE
_INT64_MAX: Final[int] = (1 << 63) - 1


class CudaFullEvolutionScienceExtensionError(ValueError):
    """Raised when a science sidecar identity is incomplete or inconsistent."""


def _validate_json(value: object, *, path: str = "$") -> None:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise CudaFullEvolutionScienceExtensionError(
                f"{path} contains a non-finite number"
            )
        return
    if value_type is dict:
        if any(type(key) is not str for key in value):
            raise CudaFullEvolutionScienceExtensionError(
                f"{path} contains a non-string object key"
            )
        for key, item in value.items():
            _validate_json(item, path=f"{path}.{key}")
        return
    if value_type in {list, tuple}:
        for index, item in enumerate(value):
            _validate_json(item, path=f"{path}[{index}]")
        return
    raise CudaFullEvolutionScienceExtensionError(
        f"{path} contains unsupported canonical JSON type {value_type.__name__}"
    )


def _canonical_bytes(value: object) -> bytes:
    _validate_json(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_hex(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_digest(value: object) -> dict[str, str]:
    return {"algorithm": "sha256", "hex": _canonical_hex(value)}


def _digest(value: object, *, field: str) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"algorithm", "hex"}
        or value.get("algorithm") != "sha256"
        or type(value.get("hex")) is not str
        or len(str(value["hex"])) != 64
        or any(character not in "0123456789abcdef" for character in str(value["hex"]))
    ):
        raise CudaFullEvolutionScienceExtensionError(
            f"{field} must be an exact SHA-256 digest object"
        )
    return {"algorithm": "sha256", "hex": str(value["hex"])}


def _digest_from_hex(value: object, *, field: str) -> dict[str, str]:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CudaFullEvolutionScienceExtensionError(
            f"{field} must be a lowercase SHA-256 hex string"
        )
    return {"algorithm": "sha256", "hex": value}


def _optional_digest(value: object, *, field: str) -> dict[str, str] | None:
    return None if value is None else _digest(value, field=field)


def _clone_json(value: object) -> Any:
    return json.loads(_canonical_bytes(value).decode("ascii"))


def _text(value: object, *, field: str, route_token: bool = False) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise CudaFullEvolutionScienceExtensionError(
            f"{field} must be non-empty unpadded text"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise CudaFullEvolutionScienceExtensionError(
            f"{field} contains a control character"
        )
    if route_token and _ROUTE_TOKEN_RE.fullmatch(value) is None:
        raise CudaFullEvolutionScienceExtensionError(
            f"{field} must be a path-free route token"
        )
    return value


def _schema_version(value: object, *, field: str) -> str:
    version = _text(value, field=field)
    if _SCHEMA_VERSION_TOKEN_RE.fullmatch(version) is None:
        raise CudaFullEvolutionScienceExtensionError(
            f"{field} must be an ASCII schema-version token"
        )
    return version


def _nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0 or value > _INT64_MAX:
        raise CudaFullEvolutionScienceExtensionError(
            f"{field} must be an integer in [0, {_INT64_MAX}]"
        )
    return value


def _positive_int64(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1 or value > _INT64_MAX:
        raise CudaFullEvolutionScienceExtensionError(
            f"{field} must be a positive signed 64-bit integer"
        )
    return value


def _reject_generation_authority(
    value: object,
    *,
    path: str,
    in_legacy_compatibility: bool = False,
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_is_legacy = in_legacy_compatibility or key == "legacy_compatibility"
            if not child_is_legacy and key in _FORBIDDEN_TIME_KEYS:
                raise CudaFullEvolutionScienceExtensionError(
                    f"generation-shaped authority is forbidden at {path}.{key}"
                )
            _reject_generation_authority(
                item,
                path=f"{path}.{key}",
                in_legacy_compatibility=child_is_legacy,
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_generation_authority(
                item,
                path=f"{path}[{index}]",
                in_legacy_compatibility=in_legacy_compatibility,
            )


def empty_science_observation_prefix_sha256_v1() -> dict[str, str]:
    """Return the exact Science Instrument V1 empty-prefix identity."""

    return _canonical_digest(
        {
            "schema_version": SCIENCE_OBSERVATION_PREFIX_CHAIN_SCHEMA_V1,
            "prefix": "EMPTY",
        }
    )


def extend_science_observation_prefix_sha256_v1(
    prefix_sha256: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    """Apply the exact domain-separated record-by-record prefix algorithm."""

    current = _digest(prefix_sha256, field="prefix_sha256")
    normalized_records = [
        validate_science_observation_record_projection_v1(record) for record in records
    ]
    if normalized_records:
        first = normalized_records[0]
        if any(
            record["world_uid"] != first["world_uid"]
            or record["semantic_execution_id"] != first["semantic_execution_id"]
            for record in normalized_records
        ):
            raise CudaFullEvolutionScienceExtensionError(
                "prefix extension mixes worlds"
            )
        sequences = tuple(
            record["world_local_sequence"] for record in normalized_records
        )
        if sequences != tuple(range(sequences[0], sequences[0] + len(sequences))):
            raise CudaFullEvolutionScienceExtensionError(
                "prefix extension sequence is not contiguous"
            )
    for normalized in normalized_records:
        current = _canonical_digest(
            {
                "schema_version": SCIENCE_OBSERVATION_PREFIX_CHAIN_SCHEMA_V1,
                "previous_prefix_sha256": current,
                "world_local_sequence": normalized["world_local_sequence"],
                "semantic_record_sha256": science_semantic_record_sha256_v1(normalized),
            }
        )
    return current


def validate_science_route_binding_v1(
    value: Mapping[str, object],
) -> dict[str, Any]:
    """Validate the complete canonical ``ScienceRouteBindingV1`` wire form."""

    if not isinstance(value, Mapping) or set(value) != _ROUTE_BINDING_FIELDS:
        raise CudaFullEvolutionScienceExtensionError(
            "ScienceRouteBindingV1 has unknown or missing fields"
        )
    if value.get("schema_version") != SCIENCE_ROUTE_BINDING_WIRE_SCHEMA_V1:
        raise CudaFullEvolutionScienceExtensionError(
            "ScienceRouteBindingV1 wire schema differs"
        )
    normalized: dict[str, Any] = {
        "schema_version": SCIENCE_ROUTE_BINDING_WIRE_SCHEMA_V1,
        "route_id": _text(value.get("route_id"), field="route_id", route_token=True),
        "machine_profile_id": _text(
            value.get("machine_profile_id"),
            field="machine_profile_id",
            route_token=True,
        ),
    }
    for field in _SHA256_FIELDS_ROUTE_BINDING:
        normalized[field] = _digest(value.get(field), field=field)
    return normalized


def seal_science_route_binding_adapter_v1(
    value: Mapping[str, object],
) -> dict[str, Any]:
    binding = validate_science_route_binding_v1(value)
    body = {
        "schema_version": SCIENCE_ROUTE_BINDING_ADAPTER_SCHEMA_V1,
        "adapter_class": SCIENCE_ROUTE_BINDING_SCHEMA_V1,
        "source_schema_version": SCIENCE_ROUTE_BINDING_WIRE_SCHEMA_V1,
        "route_binding": binding,
        "route_binding_sha256": _canonical_digest(binding),
    }
    return body


def validate_science_observation_record_projection_v1(
    value: Mapping[str, object],
) -> dict[str, Any]:
    """Validate the pinned canonical observation-record wire boundary."""

    if not isinstance(value, Mapping) or set(value) != _RECORD_FIELDS:
        raise CudaFullEvolutionScienceExtensionError(
            "science observation record has unknown or missing fields"
        )
    if value.get("schema_version") != SCIENCE_OBSERVATION_RECORD_SCHEMA_V1:
        raise CudaFullEvolutionScienceExtensionError(
            "science observation record schema differs"
        )
    record_kind = value.get("record_kind")
    if record_kind not in SCIENCE_TYPED_RECORD_KINDS_V1:
        raise CudaFullEvolutionScienceExtensionError(
            "science observation record kind is not canonical V1"
        )
    record_kind_version = _nonnegative_int(
        value.get("record_kind_version"), field="record_kind_version"
    )
    if record_kind_version != 1:
        raise CudaFullEvolutionScienceExtensionError(
            "science observation record-kind version differs"
        )
    payload = value.get("payload")
    expected_payload_fields = _RECORD_REQUIRED_PAYLOAD_FIELDS[str(record_kind)]
    if (
        isinstance(payload, Mapping)
        and record_kind == "control_binding"
        and payload.get("schema_version")
        == SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_V1
    ):
        expected_payload_fields = expected_payload_fields | frozenset(
            {"applied_native_control"}
        )
    if (
        isinstance(payload, Mapping)
        and record_kind == "agent_state_snapshot"
        and payload.get("schema_version")
        == SCIENCE_NATIVE_AGENT_STATE_SNAPSHOT_PAYLOAD_SCHEMA_V1
    ):
        expected_payload_fields = expected_payload_fields | frozenset(
            {"absolute_birth_tick"}
        )
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema_version", "payload"}
        or type(payload.get("schema_version")) is not str
        or not payload["schema_version"]
        or not isinstance(payload.get("payload"), Mapping)
        or set(payload["payload"]) != expected_payload_fields
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "science observation payload differs from its frozen record-kind contract"
        )
    _schema_version(payload["schema_version"], field="payload.schema_version")
    _reject_generation_authority(payload["payload"], path="payload")
    if (
        record_kind == "world_completion"
        and payload["payload"]["completion_class"]
        not in SCIENCE_WORLD_COMPLETION_CLASSES_V1
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "unsupported world completion class"
        )
    cause_event_ids = value.get("cause_event_ids")
    if not isinstance(cause_event_ids, list):
        raise CudaFullEvolutionScienceExtensionError("cause_event_ids must be a list")
    normalized_causes = [
        _digest(item, field="cause_event_ids[]") for item in cause_event_ids
    ]
    if normalized_causes != sorted(
        normalized_causes, key=lambda item: (item["algorithm"], item["hex"])
    ) or len({item["hex"] for item in normalized_causes}) != len(normalized_causes):
        raise CudaFullEvolutionScienceExtensionError(
            "cause_event_ids must be sorted and unique"
        )
    normalized: dict[str, Any] = {
        "schema_version": SCIENCE_OBSERVATION_RECORD_SCHEMA_V1,
        "instrument_manifest_sha256": _digest(
            value.get("instrument_manifest_sha256"),
            field="instrument_manifest_sha256",
        ),
        "producer_component_id": _text(
            value.get("producer_component_id"), field="producer_component_id"
        ),
        "producer_contract_version": _schema_version(
            value.get("producer_contract_version"),
            field="producer_contract_version",
        ),
        "semantic_execution_id": _digest(
            value.get("semantic_execution_id"), field="semantic_execution_id"
        ),
        "crca_semantic_run_digest": _digest(
            value.get("crca_semantic_run_digest"), field="crca_semantic_run_digest"
        ),
        "science_semantic_acquisition_sha256": _digest(
            value.get("science_semantic_acquisition_sha256"),
            field="science_semantic_acquisition_sha256",
        ),
        "cell_id": _text(value.get("cell_id"), field="cell_id"),
        "logical_world_id": _nonnegative_int(
            value.get("logical_world_id"), field="logical_world_id"
        ),
        "world_uid": _digest(value.get("world_uid"), field="world_uid"),
        "absolute_tick": _nonnegative_int(
            value.get("absolute_tick"), field="absolute_tick"
        ),
        "semantic_source_address": _text(
            value.get("semantic_source_address"), field="semantic_source_address"
        ),
        "pre_sequence_key_sha256": _digest(
            value.get("pre_sequence_key_sha256"), field="pre_sequence_key_sha256"
        ),
        "world_local_sequence": _nonnegative_int(
            value.get("world_local_sequence"), field="world_local_sequence"
        ),
        "deterministic_event_id": _digest(
            value.get("deterministic_event_id"), field="deterministic_event_id"
        ),
        "record_kind": str(record_kind),
        "record_kind_version": 1,
        "control_binding_sha256": _digest(
            value.get("control_binding_sha256"), field="control_binding_sha256"
        ),
        "payload": _clone_json(payload),
        "source_state_before_sha256": _optional_digest(
            value.get("source_state_before_sha256"),
            field="source_state_before_sha256",
        ),
        "source_state_after_sha256": _optional_digest(
            value.get("source_state_after_sha256"),
            field="source_state_after_sha256",
        ),
        "cause_event_ids": normalized_causes,
        "lifecycle_epoch": (
            None
            if value.get("lifecycle_epoch") is None
            else _nonnegative_int(value.get("lifecycle_epoch"), field="lifecycle_epoch")
        ),
        "legacy_compatibility": (
            None
            if value.get("legacy_compatibility") is None
            else _clone_json(value["legacy_compatibility"])
        ),
    }
    if normalized["legacy_compatibility"] is not None:
        legacy = normalized["legacy_compatibility"]
        if not isinstance(legacy, dict) or set(legacy) != {"schema_version", "payload"}:
            raise CudaFullEvolutionScienceExtensionError(
                "legacy_compatibility must use CanonicalPayloadV1 wire shape"
            )
        _schema_version(
            legacy["schema_version"], field="legacy_compatibility.schema_version"
        )
    return normalized


def science_semantic_record_sha256_v1(
    value: Mapping[str, object],
) -> dict[str, str]:
    """Hash the exact semantic form; legacy display metadata is excluded."""

    normalized = validate_science_observation_record_projection_v1(value)
    semantic = dict(normalized)
    del semantic["legacy_compatibility"]
    return _canonical_digest(semantic)


def validate_science_observation_prefix_cursor_v1(
    value: Mapping[str, object],
) -> dict[str, Any]:
    """Validate the exact ``ScienceObservationPrefixCursorV1`` wire form."""

    if not isinstance(value, Mapping) or set(value) != _CURSOR_FIELDS:
        raise CudaFullEvolutionScienceExtensionError(
            "science observation cursor has unknown or missing fields"
        )
    if value.get("schema_version") != SCIENCE_OBSERVATION_PREFIX_CURSOR_SCHEMA_V1:
        raise CudaFullEvolutionScienceExtensionError(
            "science observation cursor schema differs"
        )
    normalized: dict[str, Any] = {
        "schema_version": SCIENCE_OBSERVATION_PREFIX_CURSOR_SCHEMA_V1,
        "instrument_manifest_sha256": _digest(
            value.get("instrument_manifest_sha256"),
            field="instrument_manifest_sha256",
        ),
        "semantic_execution_id": _digest(
            value.get("semantic_execution_id"), field="semantic_execution_id"
        ),
        "crca_semantic_run_digest": _digest(
            value.get("crca_semantic_run_digest"), field="crca_semantic_run_digest"
        ),
        "science_semantic_acquisition_sha256": _digest(
            value.get("science_semantic_acquisition_sha256"),
            field="science_semantic_acquisition_sha256",
        ),
        "cell_id": _text(value.get("cell_id"), field="cell_id"),
        "logical_world_id": _nonnegative_int(
            value.get("logical_world_id"), field="logical_world_id"
        ),
        "world_uid": _digest(value.get("world_uid"), field="world_uid"),
        "checkpoint_at_tick": _nonnegative_int(
            value.get("checkpoint_at_tick"), field="checkpoint_at_tick"
        ),
        "next_absolute_tick": _nonnegative_int(
            value.get("next_absolute_tick"), field="next_absolute_tick"
        ),
        "next_world_local_sequence": _nonnegative_int(
            value.get("next_world_local_sequence"),
            field="next_world_local_sequence",
        ),
        "prefix_record_count": _nonnegative_int(
            value.get("prefix_record_count"), field="prefix_record_count"
        ),
        "prefix_chain_sha256": _digest(
            value.get("prefix_chain_sha256"), field="prefix_chain_sha256"
        ),
        "last_deterministic_event_id": _optional_digest(
            value.get("last_deterministic_event_id"),
            field="last_deterministic_event_id",
        ),
        "previous_cursor_sha256": _optional_digest(
            value.get("previous_cursor_sha256"), field="previous_cursor_sha256"
        ),
        "effects_performed": value.get("effects_performed"),
    }
    if normalized["next_absolute_tick"] != normalized["checkpoint_at_tick"]:
        raise CudaFullEvolutionScienceExtensionError(
            "observation cursor must resume at its absolute checkpoint boundary"
        )
    if normalized["next_world_local_sequence"] != normalized["prefix_record_count"]:
        raise CudaFullEvolutionScienceExtensionError(
            "world sequence must equal the sealed prefix record count"
        )
    expected_semantic_execution = _canonical_digest(
        {
            "schema_version": SCIENCE_SEMANTIC_EXECUTION_ID_SCHEMA_V1,
            "crca_semantic_run_digest": normalized["crca_semantic_run_digest"],
            "science_semantic_acquisition_sha256": normalized[
                "science_semantic_acquisition_sha256"
            ],
        }
    )
    if normalized["semantic_execution_id"] != expected_semantic_execution:
        raise CudaFullEvolutionScienceExtensionError(
            "semantic execution identity differs from cursor inputs"
        )
    if (normalized["last_deterministic_event_id"] is None) != (
        normalized["prefix_record_count"] == 0
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "last event identity must match prefix emptiness"
        )
    if (
        normalized["prefix_record_count"] == 0
        and normalized["prefix_chain_sha256"]
        != empty_science_observation_prefix_sha256_v1()
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "empty observation prefix identity differs"
        )
    if normalized["effects_performed"] is not False:
        raise CudaFullEvolutionScienceExtensionError(
            "observation cursor sealing must be effect-free"
        )
    return normalized


def science_observation_cursor_sha256_v1(
    value: Mapping[str, object],
) -> dict[str, str]:
    return _canonical_digest(validate_science_observation_prefix_cursor_v1(value))


def seal_empty_science_observation_cursor_v1(
    *,
    instrument_manifest_sha256: Mapping[str, object],
    crca_semantic_run_digest: Mapping[str, object],
    science_semantic_acquisition_sha256: Mapping[str, object],
    cell_id: str,
    logical_world_id: int,
    world_uid: Mapping[str, object],
    checkpoint_at_tick: int = 0,
) -> dict[str, Any]:
    crca = _digest(crca_semantic_run_digest, field="crca_semantic_run_digest")
    acquisition = _digest(
        science_semantic_acquisition_sha256,
        field="science_semantic_acquisition_sha256",
    )
    return validate_science_observation_prefix_cursor_v1(
        {
            "schema_version": SCIENCE_OBSERVATION_PREFIX_CURSOR_SCHEMA_V1,
            "instrument_manifest_sha256": _digest(
                instrument_manifest_sha256, field="instrument_manifest_sha256"
            ),
            "semantic_execution_id": _canonical_digest(
                {
                    "schema_version": SCIENCE_SEMANTIC_EXECUTION_ID_SCHEMA_V1,
                    "crca_semantic_run_digest": crca,
                    "science_semantic_acquisition_sha256": acquisition,
                }
            ),
            "crca_semantic_run_digest": crca,
            "science_semantic_acquisition_sha256": acquisition,
            "cell_id": cell_id,
            "logical_world_id": logical_world_id,
            "world_uid": _digest(world_uid, field="world_uid"),
            "checkpoint_at_tick": checkpoint_at_tick,
            "next_absolute_tick": checkpoint_at_tick,
            "next_world_local_sequence": 0,
            "prefix_record_count": 0,
            "prefix_chain_sha256": empty_science_observation_prefix_sha256_v1(),
            "last_deterministic_event_id": None,
            "previous_cursor_sha256": None,
            "effects_performed": False,
        }
    )


def _cursor_projection(
    *, cuda_world_uid: int, cursor: Mapping[str, object]
) -> dict[str, Any]:
    normalized_cuda_world_uid = _positive_int64(cuda_world_uid, field="cuda_world_uid")
    normalized_cursor = validate_science_observation_prefix_cursor_v1(cursor)
    return {
        "cuda_world_uid": normalized_cuda_world_uid,
        "cursor": normalized_cursor,
        "cursor_sha256": _canonical_digest(normalized_cursor),
    }


def _validate_cursor_projections(
    values: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {
            "cuda_world_uid",
            "cursor",
            "cursor_sha256",
        }:
            raise CudaFullEvolutionScienceExtensionError(
                "science cursor projection has an inexact schema"
            )
        projection = _cursor_projection(
            cuda_world_uid=value.get("cuda_world_uid"),  # type: ignore[arg-type]
            cursor=value.get("cursor"),  # type: ignore[arg-type]
        )
        if projection["cursor_sha256"] != _digest(
            value.get("cursor_sha256"), field="cursor_sha256"
        ):
            raise CudaFullEvolutionScienceExtensionError(
                "science cursor projection digest differs"
            )
        normalized.append(projection)
    normalized.sort(key=lambda item: item["cuda_world_uid"])
    if (
        not normalized
        or len({item["cuda_world_uid"] for item in normalized}) != len(normalized)
        or len({item["cursor"]["world_uid"]["hex"] for item in normalized})
        != len(normalized)
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "science cursor projections require unique CUDA and science worlds"
        )
    return normalized


def seal_science_world_mapping_v1(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Bind numeric CUDA UIDs to exact semantic Science Instrument worlds."""

    required = {
        "cuda_world_uid",
        "science_world_uid",
        "semantic_world_id",
        "cell_id",
        "logical_world_id",
        "study_id",
        "test_id",
        "version_id",
        "arm_id",
        "replicate_id",
        "reducer_id",
    }
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise CudaFullEvolutionScienceExtensionError(
                "science world mapping row has an inexact schema"
            )
        cuda_uid = _positive_int64(row.get("cuda_world_uid"), field="cuda_world_uid")
        normalized.append(
            {
                "cuda_world_uid": cuda_uid,
                "science_world_uid": _digest(
                    row.get("science_world_uid"), field="science_world_uid"
                ),
                "semantic_world_id": _text(
                    row.get("semantic_world_id"), field="semantic_world_id"
                ),
                "cell_id": _text(row.get("cell_id"), field="cell_id"),
                "logical_world_id": _nonnegative_int(
                    row.get("logical_world_id"), field="logical_world_id"
                ),
                **{
                    key: _text(row.get(key), field=key)
                    for key in (
                        "study_id",
                        "test_id",
                        "version_id",
                        "arm_id",
                        "replicate_id",
                        "reducer_id",
                    )
                },
            }
        )
    normalized.sort(key=lambda row: row["cuda_world_uid"])
    if not normalized or any(
        len(
            {
                row[field] if field != "science_world_uid" else row[field]["hex"]
                for row in normalized
            }
        )
        != len(normalized)
        for field in (
            "cuda_world_uid",
            "science_world_uid",
            "semantic_world_id",
            "logical_world_id",
        )
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "science world mapping identities must be unique"
        )
    body = {
        "schema_version": "eiel_cuda_full_evolution_science_world_mapping_v1",
        "rows": normalized,
    }
    return {**body, "mapping_sha256": _canonical_digest(body)}


def validate_science_world_mapping_v1(
    value: Mapping[str, object],
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema_version",
            "rows",
            "mapping_sha256",
        }
        or value.get("schema_version")
        != "eiel_cuda_full_evolution_science_world_mapping_v1"
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "science world mapping envelope is invalid"
        )
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise CudaFullEvolutionScienceExtensionError(
            "science mapping rows must be a list"
        )
    rebuilt = seal_science_world_mapping_v1(rows)
    if rebuilt != dict(value):
        raise CudaFullEvolutionScienceExtensionError(
            "science world mapping seal differs"
        )
    return rebuilt


def seal_static_science_control_table_v1(
    rows: Sequence[Mapping[str, object]],
    *,
    terminal_next_absolute_tick: int,
) -> dict[str, Any]:
    """Seal complete, half-open, per-world absolute-tick control intervals."""

    terminal = _nonnegative_int(
        terminal_next_absolute_tick, field="terminal_next_absolute_tick"
    )
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "cuda_world_uid",
            "intervals",
        }:
            raise CudaFullEvolutionScienceExtensionError(
                "static science control row has an inexact schema"
            )
        intervals = row.get("intervals")
        if not isinstance(intervals, list):
            raise CudaFullEvolutionScienceExtensionError(
                "static science control row is invalid"
            )
        cuda_uid = _positive_int64(row.get("cuda_world_uid"), field="cuda_world_uid")
        normalized_intervals: list[dict[str, Any]] = []
        next_start = 0
        for interval in intervals:
            if not isinstance(interval, Mapping) or set(interval) != {
                "start_absolute_tick",
                "end_absolute_tick",
                "control_payload_sha256",
            }:
                raise CudaFullEvolutionScienceExtensionError(
                    "static control interval has an inexact schema"
                )
            start = _nonnegative_int(
                interval.get("start_absolute_tick"), field="start_absolute_tick"
            )
            end = _nonnegative_int(
                interval.get("end_absolute_tick"), field="end_absolute_tick"
            )
            if start != next_start or end <= start:
                raise CudaFullEvolutionScienceExtensionError(
                    "static control intervals must be contiguous half-open ticks"
                )
            normalized_intervals.append(
                {
                    "start_absolute_tick": start,
                    "end_absolute_tick": end,
                    "control_payload_sha256": _digest(
                        interval.get("control_payload_sha256"),
                        field="control_payload_sha256",
                    ),
                }
            )
            next_start = end
        if terminal == 0:
            if normalized_intervals:
                raise CudaFullEvolutionScienceExtensionError(
                    "zero-tick controls cannot contain intervals"
                )
        elif not normalized_intervals or next_start != terminal:
            raise CudaFullEvolutionScienceExtensionError(
                "static controls must cover the full absolute-tick horizon"
            )
        normalized.append(
            {"cuda_world_uid": cuda_uid, "intervals": normalized_intervals}
        )
    normalized.sort(key=lambda row: row["cuda_world_uid"])
    if not normalized or len({row["cuda_world_uid"] for row in normalized}) != len(
        normalized
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "static controls require unique CUDA worlds"
        )
    body = {
        "schema_version": "eiel_cuda_full_evolution_static_control_table_v1",
        "schedule_kind": SCIENCE_STATIC_CONTROL_TABLE_V1,
        "time_authority": "absolute_tick_v1",
        "start_next_absolute_tick": 0,
        "terminal_next_absolute_tick": terminal,
        "rows": normalized,
    }
    return {**body, "control_table_sha256": _canonical_digest(body)}


def validate_static_science_control_table_v1(
    value: Mapping[str, object],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "schedule_kind",
        "time_authority",
        "start_next_absolute_tick",
        "terminal_next_absolute_tick",
        "rows",
        "control_table_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("schema_version")
        != "eiel_cuda_full_evolution_static_control_table_v1"
        or value.get("schedule_kind") != SCIENCE_STATIC_CONTROL_TABLE_V1
        or value.get("time_authority") != "absolute_tick_v1"
        or type(value.get("start_next_absolute_tick")) is not int
        or value.get("start_next_absolute_tick") != 0
        or not isinstance(value.get("rows"), list)
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "static science control table envelope is invalid"
        )
    rebuilt = seal_static_science_control_table_v1(
        value["rows"],  # type: ignore[arg-type]
        terminal_next_absolute_tick=value.get("terminal_next_absolute_tick"),  # type: ignore[arg-type]
    )
    if rebuilt != dict(value):
        raise CudaFullEvolutionScienceExtensionError(
            "static science control table seal differs"
        )
    return rebuilt


class ScienceTypedCaptureBufferV1:
    """Bounded host-only capture of exact canonical science record projections."""

    __slots__ = (
        "_capacity_rows",
        "_starting",
        "_by_science_uid",
        "_records",
        "_record_counts",
        "_overflowed",
    )

    def __init__(
        self,
        *,
        capacity_rows: int,
        starting_cursor_projections: Sequence[Mapping[str, object]],
    ) -> None:
        self._capacity_rows = _nonnegative_int(capacity_rows, field="capacity_rows")
        self._starting = _validate_cursor_projections(starting_cursor_projections)
        self._by_science_uid = {
            projection["cursor"]["world_uid"]["hex"]: projection
            for projection in self._starting
        }
        self._records: list[dict[str, Any]] = []
        self._record_counts: Counter[str] = Counter()
        self._overflowed = False

    @property
    def capacity_rows(self) -> int:
        return self._capacity_rows

    @property
    def committed_rows(self) -> int:
        return len(self._records)

    @property
    def overflowed(self) -> bool:
        return self._overflowed

    def append(self, value: Mapping[str, object]) -> None:
        if self._overflowed:
            raise CudaFullEvolutionScienceExtensionError(
                "typed capture already overflowed; frontier cannot commit"
            )
        if len(self._records) >= self._capacity_rows:
            self._overflowed = True
            raise CudaFullEvolutionScienceExtensionError(
                "typed science capture overflow; frontier not committed"
            )
        record = validate_science_observation_record_projection_v1(value)
        uid_hex = record["world_uid"]["hex"]
        projection = self._by_science_uid.get(uid_hex)
        if projection is None:
            raise CudaFullEvolutionScienceExtensionError(
                "typed capture record names an unmapped science world"
            )
        cursor = projection["cursor"]
        identity_fields = (
            "instrument_manifest_sha256",
            "semantic_execution_id",
            "crca_semantic_run_digest",
            "science_semantic_acquisition_sha256",
            "cell_id",
            "logical_world_id",
            "world_uid",
        )
        if any(record[field] != cursor[field] for field in identity_fields):
            raise CudaFullEvolutionScienceExtensionError(
                "typed capture record identity differs from its starting cursor"
            )
        current_world_count = sum(
            existing["world_uid"] == record["world_uid"] for existing in self._records
        )
        expected_sequence = cursor["next_world_local_sequence"] + current_world_count
        if record["world_local_sequence"] != expected_sequence:
            raise CudaFullEvolutionScienceExtensionError(
                "typed capture world-local sequence is not contiguous"
            )
        if record["absolute_tick"] < cursor["next_absolute_tick"]:
            raise CudaFullEvolutionScienceExtensionError(
                "typed capture record precedes its starting cursor"
            )
        previous_world_record = next(
            (
                existing
                for existing in reversed(self._records)
                if existing["world_uid"] == record["world_uid"]
            ),
            None,
        )
        if previous_world_record is not None and (
            record["absolute_tick"],
            record["world_local_sequence"],
        ) <= (
            previous_world_record["absolute_tick"],
            previous_world_record["world_local_sequence"],
        ):
            raise CudaFullEvolutionScienceExtensionError(
                "typed capture per-world observation order is invalid"
            )
        self._records.append(record)
        self._record_counts[record["record_kind"]] += 1

    def extend(self, values: Sequence[Mapping[str, object]]) -> None:
        for value in values:
            self.append(value)

    def seal(self, *, checkpoint_at_tick: int) -> dict[str, Any]:
        checkpoint_tick = _nonnegative_int(
            checkpoint_at_tick, field="checkpoint_at_tick"
        )
        if self._overflowed:
            raise CudaFullEvolutionScienceExtensionError(
                "typed capture overflow blocks checkpoint frontier commit"
            )
        canonical_records = sorted(
            self._records,
            key=lambda row: (row["world_uid"]["hex"], row["world_local_sequence"]),
        )
        if any(row["absolute_tick"] >= checkpoint_tick for row in canonical_records):
            raise CudaFullEvolutionScienceExtensionError(
                "typed capture must be a half-open prefix before checkpoint_at_tick"
            )
        final_projections: list[dict[str, Any]] = []
        for projection in self._starting:
            cursor = projection["cursor"]
            if checkpoint_tick < cursor["next_absolute_tick"]:
                raise CudaFullEvolutionScienceExtensionError(
                    "checkpoint precedes a starting observation cursor"
                )
            rows = [
                row
                for row in canonical_records
                if row["world_uid"] == cursor["world_uid"]
            ]
            if not rows and checkpoint_tick == cursor["next_absolute_tick"]:
                final_cursor = cursor
            else:
                final_cursor = validate_science_observation_prefix_cursor_v1(
                    {
                        **cursor,
                        "checkpoint_at_tick": checkpoint_tick,
                        "next_absolute_tick": checkpoint_tick,
                        "next_world_local_sequence": cursor["next_world_local_sequence"]
                        + len(rows),
                        "prefix_record_count": cursor["prefix_record_count"]
                        + len(rows),
                        "prefix_chain_sha256": extend_science_observation_prefix_sha256_v1(
                            cursor["prefix_chain_sha256"], rows
                        ),
                        "last_deterministic_event_id": (
                            cursor["last_deterministic_event_id"]
                            if not rows
                            else rows[-1]["deterministic_event_id"]
                        ),
                        "previous_cursor_sha256": projection["cursor_sha256"],
                    }
                )
            final_projections.append(
                _cursor_projection(
                    cuda_world_uid=projection["cuda_world_uid"],
                    cursor=final_cursor,
                )
            )
        body = {
            "schema_version": SCIENCE_TYPED_CAPTURE_INVENTORY_SCHEMA_V1,
            "observation_record_schema_version": SCIENCE_OBSERVATION_RECORD_SCHEMA_V1,
            "observation_cursor_schema_version": SCIENCE_OBSERVATION_PREFIX_CURSOR_SCHEMA_V1,
            "time_authority": "absolute_tick_v1",
            "artifact_lifecycle_profile": SCIENCE_LIFECYCLE_PROFILE_V1,
            "checkpoint_at_tick": checkpoint_tick,
            "capacity_rows": self._capacity_rows,
            "committed_rows": len(canonical_records),
            "capture_overflowed": False,
            "overflow_policy": "fail_closed_no_frontier_commit",
            "capture_effects_performed": False,
            "starting_cursors": self._starting,
            "final_cursors": final_projections,
            "record_kind_counts": {
                kind: self._record_counts.get(kind, 0)
                for kind in SCIENCE_TYPED_RECORD_KINDS_V1
            },
            "records": canonical_records,
            "capture_rows_sha256": _canonical_digest(canonical_records),
        }
        return {**body, "inventory_sha256": _canonical_digest(body)}


def validate_typed_science_capture_inventory_v1(
    value: Mapping[str, object],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "observation_record_schema_version",
        "observation_cursor_schema_version",
        "time_authority",
        "artifact_lifecycle_profile",
        "checkpoint_at_tick",
        "capacity_rows",
        "committed_rows",
        "capture_overflowed",
        "overflow_policy",
        "capture_effects_performed",
        "starting_cursors",
        "final_cursors",
        "record_kind_counts",
        "records",
        "capture_rows_sha256",
        "inventory_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise CudaFullEvolutionScienceExtensionError(
            "typed science capture inventory has an inexact schema"
        )
    if (
        value.get("schema_version") != SCIENCE_TYPED_CAPTURE_INVENTORY_SCHEMA_V1
        or value.get("observation_record_schema_version")
        != SCIENCE_OBSERVATION_RECORD_SCHEMA_V1
        or value.get("observation_cursor_schema_version")
        != SCIENCE_OBSERVATION_PREFIX_CURSOR_SCHEMA_V1
        or value.get("time_authority") != "absolute_tick_v1"
        or value.get("artifact_lifecycle_profile") != SCIENCE_LIFECYCLE_PROFILE_V1
        or value.get("capture_overflowed") is not False
        or value.get("overflow_policy") != "fail_closed_no_frontier_commit"
        or value.get("capture_effects_performed") is not False
        or not isinstance(value.get("starting_cursors"), list)
        or not isinstance(value.get("final_cursors"), list)
        or not isinstance(value.get("records"), list)
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "typed science capture inventory authority differs"
        )
    _nonnegative_int(value.get("committed_rows"), field="committed_rows")
    record_kind_counts = value.get("record_kind_counts")
    if not isinstance(record_kind_counts, Mapping) or set(record_kind_counts) != set(
        SCIENCE_TYPED_RECORD_KINDS_V1
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "typed science capture record-kind counts have an inexact schema"
        )
    for record_kind in SCIENCE_TYPED_RECORD_KINDS_V1:
        _nonnegative_int(
            record_kind_counts[record_kind],
            field=f"record_kind_counts.{record_kind}",
        )
    buffer = ScienceTypedCaptureBufferV1(
        capacity_rows=value.get("capacity_rows"),  # type: ignore[arg-type]
        starting_cursor_projections=value["starting_cursors"],  # type: ignore[arg-type]
    )
    buffer.extend(value["records"])  # type: ignore[arg-type]
    rebuilt = buffer.seal(
        checkpoint_at_tick=value.get("checkpoint_at_tick")  # type: ignore[arg-type]
    )
    if rebuilt != dict(value):
        raise CudaFullEvolutionScienceExtensionError(
            "typed science capture inventory seal or cursor continuity differs"
        )
    return rebuilt


def project_base_checkpoint_identity_v1(checkpoint: object) -> dict[str, Any]:
    """Project boundary fields without granting restore admission.

    This helper is useful for rendering and fixtures.  Authority-facing sealing
    and restoration call ``_validated_base_checkpoint_identity_v1`` below,
    which requires the exact runtime type and executes the canonical checkpoint
    envelope validator (including state, layout, and metadata-seal recomputation).
    """

    fields = (
        "schema_version",
        "time_authority",
        "canonical_state_sha256",
        "metadata_seal_sha256",
        "acknowledged_event_next_tick",
        "world_uids",
    )
    if isinstance(checkpoint, Mapping):
        if set(checkpoint) != set(fields):
            raise CudaFullEvolutionScienceExtensionError(
                "base checkpoint projection has an inexact schema"
            )
        raw = dict(checkpoint)
    else:
        if any(not hasattr(checkpoint, field) for field in fields):
            raise CudaFullEvolutionScienceExtensionError(
                "base checkpoint lacks the immutable CUDA V2 boundary"
            )
        raw = {field: getattr(checkpoint, field) for field in fields}
    world_uids = raw["world_uids"]
    if (
        raw["schema_version"] != "eiel_full_evolution_cuda_checkpoint_v2"
        or raw["time_authority"] != "absolute_tick_v1"
        or not isinstance(world_uids, (list, tuple))
        or not world_uids
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "base checkpoint projection is not canonical CUDA V2"
        )
    normalized_world_uids = [
        _positive_int64(uid, field="world_uids[]") for uid in world_uids
    ]
    if len(set(normalized_world_uids)) != len(normalized_world_uids):
        raise CudaFullEvolutionScienceExtensionError(
            "base checkpoint projection is not canonical CUDA V2"
        )
    body = {
        "schema_version": raw["schema_version"],
        "time_authority": raw["time_authority"],
        "canonical_state_sha256": _digest_from_hex(
            raw["canonical_state_sha256"], field="canonical_state_sha256"
        ),
        "metadata_seal_sha256": _digest_from_hex(
            raw["metadata_seal_sha256"], field="metadata_seal_sha256"
        ),
        "acknowledged_event_next_tick": _nonnegative_int(
            raw["acknowledged_event_next_tick"],
            field="acknowledged_event_next_tick",
        ),
        "world_uids": normalized_world_uids,
    }
    return {**body, "base_checkpoint_identity_sha256": _canonical_digest(body)}


def _validated_base_checkpoint_identity_v1(checkpoint: object) -> dict[str, Any]:
    # Lazy import preserves the hard rule that the observation-OFF CUDA hot path
    # does not import or allocate the science adapter.
    from .runtime import (
        CudaFullEvolutionCheckpointV2,
        _validate_checkpoint_envelope,
    )

    if type(checkpoint) is not CudaFullEvolutionCheckpointV2:
        raise CudaFullEvolutionScienceExtensionError(
            "science checkpoint sealing requires exact CudaFullEvolutionCheckpointV2"
        )
    try:
        _validate_checkpoint_envelope(checkpoint)
    except Exception as exc:
        raise CudaFullEvolutionScienceExtensionError(
            f"base CUDA V2 checkpoint validation failed: {exc}"
        ) from exc
    return project_base_checkpoint_identity_v1(checkpoint)


def observation_off_projection_v1(
    *,
    base_checkpoint_state_sha256: str,
    base_checkpoint_metadata_sha256: str,
    next_absolute_tick: int,
) -> dict[str, Any]:
    """Project explicit zero-allocation observation OFF state outside the hot path."""

    body = {
        "schema_version": "eiel_cuda_full_evolution_observation_off_projection_v1",
        "observation_mode": SCIENCE_OBSERVATION_OFF_V1,
        "base_checkpoint_state_sha256": _digest_from_hex(
            base_checkpoint_state_sha256, field="base_checkpoint_state_sha256"
        ),
        "base_checkpoint_metadata_sha256": _digest_from_hex(
            base_checkpoint_metadata_sha256, field="base_checkpoint_metadata_sha256"
        ),
        "next_absolute_tick": _nonnegative_int(
            next_absolute_tick, field="next_absolute_tick"
        ),
        "capture_enabled": False,
        "science_allocation_bytes": 0,
        "science_hot_path_hooks": 0,
        "trajectory_semantics": "unchanged",
        "science_route_qualified": False,
    }
    return {**body, "projection_sha256": _canonical_digest(body)}


def admit_science_extension_request_v1(request: Mapping[str, object]) -> dict[str, Any]:
    """Admit the legacy extension contract; typed capture uses the V1 carrier seam.

    The public extension contract is intentionally still OFF-only.  Science
    Instrument typed capture is an additive engineering path owned by the
    carrier/provider contract, so callers must not mistake this older helper
    for its admission surface.
    """

    if set(request) != {"observation_mode", "control_schedule_kind"}:
        raise CudaFullEvolutionScienceExtensionError(
            "science extension request has an inexact schema"
        )
    if (
        request.get("observation_mode") != SCIENCE_OBSERVATION_OFF_V1
        or request.get("control_schedule_kind") is not None
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "typed science capture is host-adapter-only and not runtime-admitted"
        )
    return {
        "observation_mode": SCIENCE_OBSERVATION_OFF_V1,
        "control_schedule_kind": None,
        "science_allocation_bytes": 0,
        "science_hot_path_hooks": 0,
        "science_route_qualified": False,
    }


def _cursor_set_sha256(inventory: Mapping[str, object]) -> dict[str, str]:
    projections = inventory.get("final_cursors")
    if not isinstance(projections, list):
        raise CudaFullEvolutionScienceExtensionError(
            "capture inventory lacks final cursor projections"
        )
    normalized = _validate_cursor_projections(projections)
    return _canonical_digest(
        {
            "schema_version": SCIENCE_CURSOR_SET_SCHEMA_V1,
            "cursors": [
                {
                    "cuda_world_uid": projection["cuda_world_uid"],
                    "cursor_sha256": projection["cursor_sha256"],
                }
                for projection in normalized
            ],
        }
    )


def seal_science_checkpoint_extension_v1(
    *,
    base_checkpoint: object,
    science_route_binding: Mapping[str, object],
    world_mapping: Mapping[str, object],
    control_table: Mapping[str, object],
    typed_capture_inventory: Mapping[str, object],
    continuation_parent_checkpoint_sha256: Mapping[str, object] | None,
) -> dict[str, Any]:
    """Seal one opaque extension from complete, validated identity sidecars."""

    base = _validated_base_checkpoint_identity_v1(base_checkpoint)
    route_adapter = seal_science_route_binding_adapter_v1(science_route_binding)
    mapping = validate_science_world_mapping_v1(world_mapping)
    controls = validate_static_science_control_table_v1(control_table)
    inventory = validate_typed_science_capture_inventory_v1(typed_capture_inventory)
    final_cursors = _validate_cursor_projections(inventory["final_cursors"])
    base_worlds = base["world_uids"]
    mapping_worlds = [row["cuda_world_uid"] for row in mapping["rows"]]
    control_worlds = [row["cuda_world_uid"] for row in controls["rows"]]
    cursor_worlds = [row["cuda_world_uid"] for row in final_cursors]
    if not base_worlds == mapping_worlds == control_worlds == cursor_worlds:
        raise CudaFullEvolutionScienceExtensionError(
            "base checkpoint, mapping, controls, and cursors name different worlds"
        )
    next_tick = base["acknowledged_event_next_tick"]
    if (
        inventory["checkpoint_at_tick"] != next_tick
        or next_tick > controls["terminal_next_absolute_tick"]
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "science sidecar cursor differs from the absolute base checkpoint frontier"
        )
    mapping_by_cuda = {row["cuda_world_uid"]: row for row in mapping["rows"]}
    for projection in final_cursors:
        cursor = projection["cursor"]
        mapped = mapping_by_cuda[projection["cuda_world_uid"]]
        if (
            cursor["world_uid"] != mapped["science_world_uid"]
            or cursor["cell_id"] != mapped["cell_id"]
            or cursor["logical_world_id"] != mapped["logical_world_id"]
        ):
            raise CudaFullEvolutionScienceExtensionError(
                "science cursor differs from the numeric-to-semantic UID mapping"
            )
    acquisitions = {
        projection["cursor"]["science_semantic_acquisition_sha256"]["hex"]
        for projection in final_cursors
    }
    instruments = {
        projection["cursor"]["instrument_manifest_sha256"]["hex"]
        for projection in final_cursors
    }
    crca_runs = {
        projection["cursor"]["crca_semantic_run_digest"]["hex"]
        for projection in final_cursors
    }
    semantic_executions = {
        projection["cursor"]["semantic_execution_id"]["hex"]
        for projection in final_cursors
    }
    if len(acquisitions) != 1 or instruments != {
        route_adapter["route_binding"]["instrument_manifest_sha256"]["hex"]
    }:
        raise CudaFullEvolutionScienceExtensionError(
            "science acquisition or instrument identity differs across the extension"
        )
    if len(crca_runs) != 1 or len(semantic_executions) != 1:
        raise CudaFullEvolutionScienceExtensionError(
            "science CRCA run or semantic execution identity differs across the extension"
        )
    parent_digest = (
        None
        if continuation_parent_checkpoint_sha256 is None
        else _digest(
            continuation_parent_checkpoint_sha256,
            field="continuation_parent_checkpoint_sha256",
        )
    )
    cursor_has_parent = {
        projection["cursor"]["previous_cursor_sha256"] is not None
        for projection in final_cursors
    }
    if cursor_has_parent == {False}:
        if parent_digest is not None:
            raise CudaFullEvolutionScienceExtensionError(
                "root science checkpoint cannot name a continuation parent"
            )
        continuation_kind = "ROOT"
    elif cursor_has_parent == {True}:
        if parent_digest is None:
            raise CudaFullEvolutionScienceExtensionError(
                "continued science checkpoint requires an exact parent"
            )
        continuation_kind = "CONTINUATION"
    else:
        raise CudaFullEvolutionScienceExtensionError(
            "science worlds disagree about continuation parent presence"
        )
    body = {
        "schema_version": SCIENCE_CHECKPOINT_EXTENSION_SCHEMA_V1,
        "contract_status": SCIENCE_EXTENSION_STATUS_V1,
        "observation_mode": SCIENCE_OBSERVATION_TYPED_SIDECAR_V1,
        "time_authority": "absolute_tick_v1",
        "base_checkpoint_identity_sha256": base["base_checkpoint_identity_sha256"],
        "next_absolute_tick": next_tick,
        "science_semantic_acquisition_sha256": _digest_from_hex(
            next(iter(acquisitions)), field="science_semantic_acquisition_sha256"
        ),
        "observation_cursor_set_sha256": _cursor_set_sha256(inventory),
        "control_table_sha256": controls["control_table_sha256"],
        "control_schedule_kind": SCIENCE_STATIC_CONTROL_TABLE_V1,
        "control_cursor_next_absolute_tick": next_tick,
        "continuation_kind": continuation_kind,
        "continuation_parent_checkpoint_sha256": parent_digest,
        "world_mapping_sha256": mapping["mapping_sha256"],
        "typed_capture_inventory_sha256": inventory["inventory_sha256"],
        "typed_capture_capacity_rows": inventory["capacity_rows"],
        "typed_capture_committed_rows": inventory["committed_rows"],
        "capture_overflowed": False,
        "overflow_policy": "fail_closed_no_frontier_commit",
        "science_route_binding_schema": SCIENCE_ROUTE_BINDING_WIRE_SCHEMA_V1,
        "science_route_binding_sha256": route_adapter["route_binding_sha256"],
        "artifact_lifecycle_profile": SCIENCE_LIFECYCLE_PROFILE_V1,
        "continuation_identity_status": "SEALED_REQUIRES_RESTORE_GATE",
        "native_capture_runtime_integrated": False,
        "science_route_qualified": False,
        "science_continuable": False,
    }
    extension_sha256 = _canonical_digest(body)
    checkpoint_identity_sha256 = _canonical_digest(
        {
            "schema_version": SCIENCE_CHECKPOINT_IDENTITY_SCHEMA_V1,
            "base_checkpoint_identity_sha256": base["base_checkpoint_identity_sha256"],
            "science_extension_sha256": extension_sha256,
        }
    )
    return {
        **body,
        "extension_sha256": extension_sha256,
        "science_checkpoint_identity_sha256": checkpoint_identity_sha256,
    }


def validate_science_checkpoint_extension_v1(
    envelope: Mapping[str, object],
) -> dict[str, Any]:
    """Validate an opaque seal without granting restore admission."""

    required = {
        "schema_version",
        "contract_status",
        "observation_mode",
        "time_authority",
        "base_checkpoint_identity_sha256",
        "next_absolute_tick",
        "science_semantic_acquisition_sha256",
        "observation_cursor_set_sha256",
        "control_table_sha256",
        "control_schedule_kind",
        "control_cursor_next_absolute_tick",
        "continuation_kind",
        "continuation_parent_checkpoint_sha256",
        "world_mapping_sha256",
        "typed_capture_inventory_sha256",
        "typed_capture_capacity_rows",
        "typed_capture_committed_rows",
        "capture_overflowed",
        "overflow_policy",
        "science_route_binding_schema",
        "science_route_binding_sha256",
        "artifact_lifecycle_profile",
        "continuation_identity_status",
        "native_capture_runtime_integrated",
        "science_route_qualified",
        "science_continuable",
        "extension_sha256",
        "science_checkpoint_identity_sha256",
    }
    if not isinstance(envelope, Mapping) or set(envelope) != required:
        raise CudaFullEvolutionScienceExtensionError(
            "science checkpoint extension has an inexact schema"
        )
    normalized = _clone_json(envelope)
    constants = {
        "schema_version": SCIENCE_CHECKPOINT_EXTENSION_SCHEMA_V1,
        "contract_status": SCIENCE_EXTENSION_STATUS_V1,
        "observation_mode": SCIENCE_OBSERVATION_TYPED_SIDECAR_V1,
        "time_authority": "absolute_tick_v1",
        "control_schedule_kind": SCIENCE_STATIC_CONTROL_TABLE_V1,
        "overflow_policy": "fail_closed_no_frontier_commit",
        "science_route_binding_schema": SCIENCE_ROUTE_BINDING_WIRE_SCHEMA_V1,
        "artifact_lifecycle_profile": SCIENCE_LIFECYCLE_PROFILE_V1,
        "continuation_identity_status": "SEALED_REQUIRES_RESTORE_GATE",
    }
    boolean_constants = {
        "capture_overflowed": False,
        "native_capture_runtime_integrated": False,
        "science_route_qualified": False,
        "science_continuable": False,
    }
    if any(
        normalized.get(key) != expected for key, expected in constants.items()
    ) or any(
        normalized.get(key) is not expected
        for key, expected in boolean_constants.items()
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "science checkpoint extension authority differs"
        )
    next_tick = _nonnegative_int(
        normalized["next_absolute_tick"], field="next_absolute_tick"
    )
    control_cursor_next_tick = _nonnegative_int(
        normalized["control_cursor_next_absolute_tick"],
        field="control_cursor_next_absolute_tick",
    )
    if control_cursor_next_tick != next_tick:
        raise CudaFullEvolutionScienceExtensionError(
            "science checkpoint control cursor differs from checkpoint frontier"
        )
    capacity = _nonnegative_int(
        normalized["typed_capture_capacity_rows"],
        field="typed_capture_capacity_rows",
    )
    committed = _nonnegative_int(
        normalized["typed_capture_committed_rows"],
        field="typed_capture_committed_rows",
    )
    if committed > capacity:
        raise CudaFullEvolutionScienceExtensionError(
            "science checkpoint capture count exceeds capacity"
        )
    for field in (
        "base_checkpoint_identity_sha256",
        "science_semantic_acquisition_sha256",
        "observation_cursor_set_sha256",
        "control_table_sha256",
        "world_mapping_sha256",
        "typed_capture_inventory_sha256",
        "science_route_binding_sha256",
        "extension_sha256",
        "science_checkpoint_identity_sha256",
    ):
        normalized[field] = _digest(normalized[field], field=field)
    parent = normalized["continuation_parent_checkpoint_sha256"]
    if normalized["continuation_kind"] == "ROOT":
        if parent is not None:
            raise CudaFullEvolutionScienceExtensionError(
                "root extension cannot name a parent"
            )
    elif normalized["continuation_kind"] == "CONTINUATION":
        normalized["continuation_parent_checkpoint_sha256"] = _digest(
            parent, field="continuation_parent_checkpoint_sha256"
        )
    else:
        raise CudaFullEvolutionScienceExtensionError(
            "science checkpoint continuation kind is invalid"
        )
    body = {
        key: normalized[key]
        for key in normalized
        if key not in {"extension_sha256", "science_checkpoint_identity_sha256"}
    }
    expected_extension = _canonical_digest(body)
    expected_checkpoint = _canonical_digest(
        {
            "schema_version": SCIENCE_CHECKPOINT_IDENTITY_SCHEMA_V1,
            "base_checkpoint_identity_sha256": normalized[
                "base_checkpoint_identity_sha256"
            ],
            "science_extension_sha256": expected_extension,
        }
    )
    if (
        normalized["extension_sha256"] != expected_extension
        or normalized["science_checkpoint_identity_sha256"] != expected_checkpoint
    ):
        raise CudaFullEvolutionScienceExtensionError(
            "science checkpoint extension or checkpoint identity seal differs"
        )
    return normalized


def admit_science_checkpoint_restore_v1(
    envelope: Mapping[str, object],
    *,
    base_checkpoint: object,
    science_route_binding: Mapping[str, object],
    world_mapping: Mapping[str, object],
    control_table: Mapping[str, object],
    typed_capture_inventory: Mapping[str, object],
    parent_envelope: Mapping[str, object] | None = None,
    parent_typed_capture_inventory: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Admit the opaque restoration boundary only after all identities converge."""

    current = validate_science_checkpoint_extension_v1(envelope)
    rebuilt = seal_science_checkpoint_extension_v1(
        base_checkpoint=base_checkpoint,
        science_route_binding=science_route_binding,
        world_mapping=world_mapping,
        control_table=control_table,
        typed_capture_inventory=typed_capture_inventory,
        continuation_parent_checkpoint_sha256=current[
            "continuation_parent_checkpoint_sha256"
        ],
    )
    if rebuilt != current:
        raise CudaFullEvolutionScienceExtensionError(
            "science restore sidecars do not reconstruct the checkpoint extension"
        )
    inventory = validate_typed_science_capture_inventory_v1(typed_capture_inventory)
    if current["continuation_kind"] == "ROOT":
        if parent_envelope is not None or parent_typed_capture_inventory is not None:
            raise CudaFullEvolutionScienceExtensionError(
                "root science restore cannot supply parent state"
            )
    else:
        if parent_envelope is None or parent_typed_capture_inventory is None:
            raise CudaFullEvolutionScienceExtensionError(
                "continued science restore requires parent envelope and cursor inventory"
            )
        parent = validate_science_checkpoint_extension_v1(parent_envelope)
        parent_inventory = validate_typed_science_capture_inventory_v1(
            parent_typed_capture_inventory
        )
        if (
            current["continuation_parent_checkpoint_sha256"]
            != parent["science_checkpoint_identity_sha256"]
            or parent["next_absolute_tick"] > current["next_absolute_tick"]
            or parent["science_semantic_acquisition_sha256"]
            != current["science_semantic_acquisition_sha256"]
            or parent["science_route_binding_sha256"]
            != current["science_route_binding_sha256"]
            or parent["world_mapping_sha256"] != current["world_mapping_sha256"]
            or parent["control_table_sha256"] != current["control_table_sha256"]
            or parent["typed_capture_inventory_sha256"]
            != parent_inventory["inventory_sha256"]
            or inventory["starting_cursors"] != parent_inventory["final_cursors"]
        ):
            raise CudaFullEvolutionScienceExtensionError(
                "science continuation parent identity or cursor frontier differs"
            )
    receipt_body = {
        "schema_version": SCIENCE_RESTORE_GATE_SCHEMA_V1,
        "science_checkpoint_identity_sha256": current[
            "science_checkpoint_identity_sha256"
        ],
        "base_checkpoint_identity_sha256": current["base_checkpoint_identity_sha256"],
        "next_absolute_tick": current["next_absolute_tick"],
        "continuation_kind": current["continuation_kind"],
        "opaque_extension_restore_admitted": True,
        "science_continuation_identity_complete": True,
        "native_capture_runtime_integrated": False,
        "science_route_qualified": False,
        "science_continuable": False,
        "authority_scope": "HOST_ADAPTER_IDENTITY_ONLY",
    }
    return {**receipt_body, "restore_gate_sha256": _canonical_digest(receipt_body)}


__all__ = [
    "SCIENCE_CHECKPOINT_EXTENSION_SCHEMA_V1",
    "SCIENCE_CHECKPOINT_IDENTITY_SCHEMA_V1",
    "SCIENCE_CONTRACT_RELATIVE_PATH_V1",
    "SCIENCE_EXTENSION_CONTRACT_SCHEMA_V1",
    "SCIENCE_EXTENSION_STATUS_V1",
    "SCIENCE_LIFECYCLE_PROFILE_V1",
    "SCIENCE_NATIVE_AGENT_STATE_SNAPSHOT_PAYLOAD_SCHEMA_V1",
    "SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_V1",
    "SCIENCE_OBSERVATION_OFF_V1",
    "SCIENCE_OBSERVATION_PREFIX_CURSOR_SCHEMA_V1",
    "SCIENCE_OBSERVATION_RECORD_SCHEMA_V1",
    "SCIENCE_OBSERVATION_TYPED_SIDECAR_V1",
    "SCIENCE_ROUTE_BINDING_SCHEMA_V1",
    "SCIENCE_ROUTE_BINDING_WIRE_SCHEMA_V1",
    "SCIENCE_STATIC_CONTROL_TABLE_V1",
    "SCIENCE_TYPED_CAPTURE_INVENTORY_SCHEMA_V1",
    "SCIENCE_TYPED_RECORD_KINDS_V1",
    "CudaFullEvolutionScienceExtensionError",
    "ScienceTypedCaptureBufferV1",
    "admit_science_checkpoint_restore_v1",
    "admit_science_extension_request_v1",
    "empty_science_observation_prefix_sha256_v1",
    "extend_science_observation_prefix_sha256_v1",
    "observation_off_projection_v1",
    "project_base_checkpoint_identity_v1",
    "science_observation_cursor_sha256_v1",
    "science_semantic_record_sha256_v1",
    "seal_empty_science_observation_cursor_v1",
    "seal_science_checkpoint_extension_v1",
    "seal_science_route_binding_adapter_v1",
    "seal_science_world_mapping_v1",
    "seal_static_science_control_table_v1",
    "validate_science_checkpoint_extension_v1",
    "validate_science_observation_prefix_cursor_v1",
    "validate_science_observation_record_projection_v1",
    "validate_science_route_binding_v1",
    "validate_science_world_mapping_v1",
    "validate_static_science_control_table_v1",
    "validate_typed_science_capture_inventory_v1",
]
