"""Pure, immutable contracts for ``EIEL_SCIENCE_INSTRUMENT_V1``.

The objects in this module describe science intent and bind content identities.
They do not compile campaigns, derive CRCA identity, admit execution, read schema
files, build artifacts, or launch a provider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ._validation import (
    INT64_MAX,
    require_instance,
    require_int,
    require_sorted_unique_text,
    require_text,
    require_tuple_instances,
)
from .canonical import canonical_json_bytes, canonical_sha256
from .errors import contract_invalid
from .ids import ContentDigest, SchemaVersion
from .payloads import CanonicalPayloadV1

PRODUCT_ID = "EIEL_SCIENCE_INSTRUMENT_V1"
LEGACY_PRODUCT_ALIAS = "B136_SCIENCE_INSTRUMENT_V1"

SCIENCE_CAMPAIGN_SCHEMA_VERSION = SchemaVersion("eiel_science_campaign_v1")
SCIENCE_CONTROL_DESCRIPTOR_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_control_descriptor_v1"
)
SCIENCE_REDUCER_DESCRIPTOR_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_reducer_descriptor_v1"
)
SCIENCE_INSTRUMENT_MANIFEST_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_instrument_manifest_v1"
)
SCIENCE_ROUTE_BINDING_SCHEMA_VERSION = SchemaVersion("eiel_science_route_binding_v1")
SCIENCE_EXECUTION_PLAN_SCHEMA_VERSION = SchemaVersion("eiel_science_execution_plan_v1")
SCIENCE_REDUCTION_PLAN_SCHEMA_VERSION = SchemaVersion("eiel_science_reduction_plan_v1")
SCIENCE_FINDING_SCHEMA_VERSION = SchemaVersion("eiel_science_finding_v1")
SCIENCE_ADMISSION_RECEIPT_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_admission_receipt_v1"
)
SCIENCE_OBSERVATION_RECORD_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_observation_record_v1"
)
# A native science control binding carries the exact applied mask/vector
# receipt in addition to the long-standing semantic control payload.  This is
# a payload variant, rather than a change to the V1 record kind, so existing
# non-native observations remain readable and native-bound callers fail closed
# if they omit the provenance field.
SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_native_control_binding_payload_v1"
)
# Native carriers may expose the authoritative absolute birth tick without
# changing the route-neutral base agent-state contract used by host-only and
# historical producers.  The additional value is semantic science state, not
# a physical slot or CUDA implementation field.
SCIENCE_NATIVE_AGENT_STATE_SNAPSHOT_PAYLOAD_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_native_agent_state_snapshot_payload_v1"
)
# This is the closed native-control receipt currently accepted by the physical
# CUDA V2 science carrier.  Keeping the wire validation in the low-dependency
# contract layer lets sidecar ingestion and offline reducers enforce the same
# provenance without importing the execution/provider implementation.
SCIENCE_NATIVE_CONTROL_SCHEMA_VERSION = "eiel_science_cuda_v2_native_control_v1"
SCIENCE_NATIVE_MUTATION_CONTROL_ID = "mutation.enabled"
SCIENCE_NATIVE_FULL_CAPABILITY_MASK = 127
SCIENCE_NATIVE_MUTATION_OFF_CAPABILITY_MASK = 95
SCIENCE_EVIDENCE_PARTITION_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_evidence_partition_v1"
)
SCIENCE_PROFILE_RESOURCE_ENVELOPE_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_profile_resource_envelope_v1"
)
SCIENCE_RESULT_RECEIPT_SCHEMA_VERSION = SchemaVersion("eiel_science_result_receipt_v1")

SEMANTIC_ACQUISITION_CORE_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_semantic_acquisition_v1"
)
PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_physical_execution_plan_v1"
)


class ScienceAdmissionOutcomeV1(str, Enum):
    ADMITTED = "ADMITTED"
    OUTSIDE_QUALIFIED_ENVELOPE = "OUTSIDE_QUALIFIED_ENVELOPE"
    MISSING_SCIENTIFIC_CAPABILITY = "MISSING_SCIENTIFIC_CAPABILITY"
    MISSING_OBSERVATION_CAPABILITY = "MISSING_OBSERVATION_CAPABILITY"
    INVALID_CAMPAIGN = "INVALID_CAMPAIGN"
    INSTRUMENT_INTEGRITY_FAILURE = "INSTRUMENT_INTEGRITY_FAILURE"


SCIENCE_ADMISSION_OUTCOME_PRECEDENCE: tuple[ScienceAdmissionOutcomeV1, ...] = (
    ScienceAdmissionOutcomeV1.INSTRUMENT_INTEGRITY_FAILURE,
    ScienceAdmissionOutcomeV1.INVALID_CAMPAIGN,
    ScienceAdmissionOutcomeV1.MISSING_SCIENTIFIC_CAPABILITY,
    ScienceAdmissionOutcomeV1.MISSING_OBSERVATION_CAPABILITY,
    ScienceAdmissionOutcomeV1.OUTSIDE_QUALIFIED_ENVELOPE,
    ScienceAdmissionOutcomeV1.ADMITTED,
)


class ScienceObservationProfileV1(str, Enum):
    MINIMAL = "SCIENCE_MINIMAL_V1"
    STANDARD = "SCIENCE_STANDARD_V1"
    FORENSIC = "SCIENCE_FORENSIC_V1"


class ScienceRecordKindV1(str, Enum):
    CONTROL_BINDING = "control_binding"
    WORLD_STATE_SNAPSHOT = "world_state_snapshot"
    AGENT_STATE_SNAPSHOT = "agent_state_snapshot"
    REPRODUCTION_OPPORTUNITY = "reproduction_opportunity"
    BIRTH_ATTEMPT = "birth_attempt"
    BIRTH_COMMITTED = "birth_committed"
    MUTATION_OUTCOME = "mutation_outcome"
    DECISION_ACTION = "decision_action"
    ECOLOGICAL_TRANSACTION = "ecological_transaction"
    DEATH = "death"
    ENVIRONMENT_TRANSITION = "environment_transition"
    WORLD_COMPLETION = "world_completion"
    CHECKPOINT_MARKER = "checkpoint_marker"


V2_RESERVED_RECORD_KINDS: tuple[str, ...] = (
    "lifetime_state_change",
    "structural_change",
)

_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:/-]*$")
_ROUTE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_CONTROL_SCHEDULE_KINDS = frozenset(
    {
        "piecewise_constant_absolute_ticks",
        "pulse_absolute_ticks",
        "registered_deterministic_trigger",
        "static",
        "switch_absolute_tick",
    }
)
_REDUCER_PURITY_DECLARATIONS = frozenset(
    {
        "CONTENT_ADDRESSED_IMPLEMENTATION",
        "DETERMINISTIC_CANONICAL_OUTPUT",
        "NO_NETWORK",
        "NO_SIMULATOR_IMPORT",
        "NO_TRAJECTORY_MUTATION",
        "OFFLINE_ONLY",
        # These declarations are used by the canonical compact-world atlas
        # reducer to make its sealed-input and terminal-reconciliation gates
        # explicit. They remain descriptive capability declarations, not
        # execution authority.
        "SEALED_ARTIFACT_INPUT_ONLY",
        "TERMINAL_INDEX_RECONCILIATION_REQUIRED",
    }
)
_REQUIRED_REDUCER_PURITY_DECLARATIONS = frozenset(
    {
        "DETERMINISTIC_CANONICAL_OUTPUT",
        "NO_SIMULATOR_IMPORT",
        "NO_TRAJECTORY_MUTATION",
        "OFFLINE_ONLY",
    }
)

_RECORD_REQUIRED_PAYLOAD_FIELDS: dict[ScienceRecordKindV1, frozenset[str]] = {
    ScienceRecordKindV1.CONTROL_BINDING: frozenset(
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
    ScienceRecordKindV1.WORLD_STATE_SNAPSHOT: frozenset(
        {
            "world_state_id",
            "field_set_version",
            "qualified_field_values",
            "snapshot_reason",
        }
    ),
    ScienceRecordKindV1.AGENT_STATE_SNAPSHOT: frozenset(
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
    ScienceRecordKindV1.REPRODUCTION_OPPORTUNITY: frozenset(
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
    ScienceRecordKindV1.BIRTH_ATTEMPT: frozenset(
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
    ScienceRecordKindV1.BIRTH_COMMITTED: frozenset(
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
    ScienceRecordKindV1.MUTATION_OUTCOME: frozenset(
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
    ScienceRecordKindV1.DECISION_ACTION: frozenset(
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
    ScienceRecordKindV1.ECOLOGICAL_TRANSACTION: frozenset(
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
    ScienceRecordKindV1.DEATH: frozenset(
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
    ScienceRecordKindV1.ENVIRONMENT_TRANSITION: frozenset(
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
    ScienceRecordKindV1.WORLD_COMPLETION: frozenset(
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
    ScienceRecordKindV1.CHECKPOINT_MARKER: frozenset(
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


def _record_required_payload_fields(
    record_kind: ScienceRecordKindV1,
    payload: CanonicalPayloadV1,
) -> frozenset[str]:
    """Return the exact payload fields for a record/payload variant."""

    required = _RECORD_REQUIRED_PAYLOAD_FIELDS[record_kind]
    if (
        record_kind is ScienceRecordKindV1.CONTROL_BINDING
        and payload.schema_version
        == SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_VERSION
    ):
        return required | frozenset({"applied_native_control"})
    if (
        record_kind is ScienceRecordKindV1.AGENT_STATE_SNAPSHOT
        and payload.schema_version
        == SCIENCE_NATIVE_AGENT_STATE_SNAPSHOT_PAYLOAD_SCHEMA_VERSION
    ):
        return required | frozenset({"absolute_birth_tick"})
    return required


def validate_science_native_control_receipt_v1(
    value: object,
    *,
    field: str = "applied_native_control",
) -> dict[str, object]:
    """Validate and normalize the closed native-control provenance receipt.

    This helper is deliberately limited to the current native V2 control
    contract.  It does not grant execution authority; it only prevents a
    canonical control-binding row from claiming a mask/vector/digest that do
    not bind to one another.
    """

    receipt = _mapping(value, field=field)
    required = {"schema_version", "capability_mask", "control_vector", "control_digest"}
    if set(receipt) != required:
        raise contract_invalid("native control receipt has an invalid shape", field=field)
    if receipt["schema_version"] != SCIENCE_NATIVE_CONTROL_SCHEMA_VERSION:
        raise contract_invalid("native control schema is unsupported", field=f"{field}.schema_version")
    mask = receipt["capability_mask"]
    if type(mask) is not int:
        raise contract_invalid("native capability mask must be an integer", field=f"{field}.capability_mask")
    vector = _mapping(receipt["control_vector"], field=f"{field}.control_vector")
    if set(vector) != {SCIENCE_NATIVE_MUTATION_CONTROL_ID} or type(
        vector[SCIENCE_NATIVE_MUTATION_CONTROL_ID]
    ) is not bool:
        raise contract_invalid("native control vector is not the closed mutation contract", field=f"{field}.control_vector")
    expected_mask = (
        SCIENCE_NATIVE_FULL_CAPABILITY_MASK
        if vector[SCIENCE_NATIVE_MUTATION_CONTROL_ID]
        else SCIENCE_NATIVE_MUTATION_OFF_CAPABILITY_MASK
    )
    if mask != expected_mask:
        raise contract_invalid("native capability mask differs from mutation.enabled", field=f"{field}.capability_mask")
    control_digest = _digest_from_dict(receipt["control_digest"], field=f"{field}.control_digest")
    expected_digest = canonical_sha256(
        {
            "schema_version": SCIENCE_NATIVE_CONTROL_SCHEMA_VERSION,
            "capability_mask": mask,
            "control_vector": dict(vector),
        }
    )
    if control_digest != expected_digest:
        raise contract_invalid("native control digest does not match mask/vector", field=f"{field}.control_digest")
    return {
        "schema_version": SCIENCE_NATIVE_CONTROL_SCHEMA_VERSION,
        "capability_mask": mask,
        "control_vector": dict(vector),
        "control_digest": control_digest.to_dict(),
    }


class EEDReleaseStatusV1(str, Enum):
    INCLUDED = "INCLUDED"
    EXCLUDED = "EXCLUDED"


_CAMPAIGN_SECTION_FIELDS = (
    "instrument",
    "question",
    "design",
    "world",
    "initialization",
    "controls",
    "seeds",
    "time",
    "observations",
    "execution",
    "reducers",
    "artifact_lifecycle",
    "decision",
)

_CAMPAIGN_SECTION_SCHEMA_VERSIONS = {
    field_name: SchemaVersion(f"eiel_science_{field_name}_v1")
    for field_name in _CAMPAIGN_SECTION_FIELDS
}
_CAMPAIGN_SECTION_SCHEMA_VERSIONS["artifact_lifecycle"] = SchemaVersion(
    "eiel_run_artifact_lifecycle_contract_v1"
)

_FORBIDDEN_ACTIVE_TIME_KEYS = frozenset(
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

_FORBIDDEN_SEMANTIC_ROUTE_KEYS = frozenset(
    {
        "cuda_architecture",
        "execution_adapter_sha256",
        "machine_profile_id",
        "native_build_closure_sha256",
        "physical_batch_count",
        "physical_batch_worlds",
        "provider_id",
        "qualified_envelope_sha256",
        "route_binding_sha256",
        "route_id",
    }
)


def _mapping(
    value: object,
    *,
    field: str,
    exact_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise contract_invalid(
            "value must be an object", field=field, value_type=type(value).__name__
        )
    if exact_keys is not None and set(value) != set(exact_keys):
        raise contract_invalid("object has an invalid shape", field=field)
    if any(type(key) is not str for key in value):
        raise contract_invalid("object keys must be strings", field=field)
    return value


def _require_token(value: object, *, field: str) -> str:
    text = require_text(value, field=field)
    if _TOKEN_RE.fullmatch(text) is None:
        raise contract_invalid("value must be an ASCII token", field=field)
    return text


def _require_route_token(value: object, *, field: str) -> str:
    text = require_text(value, field=field)
    if _ROUTE_TOKEN_RE.fullmatch(text) is None:
        raise contract_invalid(
            "route identity must be a path-free ASCII token", field=field
        )
    return text


def _require_token_tuple(values: object, *, field: str) -> None:
    require_sorted_unique_text(values, field=field)
    assert isinstance(values, tuple)
    for value in values:
        _require_token(value, field=field)


def _digest_from_dict(value: object, *, field: str) -> ContentDigest:
    digest = _mapping(value, field=field, exact_keys={"algorithm", "hex"})
    return ContentDigest(digest["algorithm"], digest["hex"])


def _payload_from_dict(value: object, *, field: str) -> CanonicalPayloadV1:
    try:
        return CanonicalPayloadV1.from_dict(value)
    except (TypeError, ValueError) as error:
        raise contract_invalid("invalid canonical payload", field=field) from error


def _payload_for_section(field: str, value: object) -> CanonicalPayloadV1:
    return CanonicalPayloadV1.from_value(
        _CAMPAIGN_SECTION_SCHEMA_VERSIONS[field], value
    )


def _optional_digest_from_dict(value: object, *, field: str) -> ContentDigest | None:
    return None if value is None else _digest_from_dict(value, field=field)


def _optional_payload_from_dict(
    value: object, *, field: str
) -> CanonicalPayloadV1 | None:
    return None if value is None else _payload_from_dict(value, field=field)


def _validate_schema_version(
    value: SchemaVersion, *, expected: SchemaVersion, field: str = "schema_version"
) -> None:
    require_instance(value, SchemaVersion, field=field)
    if value != expected:
        raise contract_invalid("unsupported schema version", field=field)


def _reject_generation_authority(
    value: object, *, path: str = "$", in_legacy_compatibility: bool = False
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise contract_invalid("object keys must be strings", field=path)
            child_is_legacy = in_legacy_compatibility or key == "legacy_compatibility"
            if not child_is_legacy and key in _FORBIDDEN_ACTIVE_TIME_KEYS:
                raise contract_invalid(
                    "generation-shaped public time authority is forbidden",
                    field=f"{path}.{key}",
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


def _reject_route_authority_from_semantic_core(
    value: object, *, path: str = "semantic_acquisition_core"
) -> None:
    """Keep route selection and implementation identity out of science semantics."""

    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise contract_invalid("object keys must be strings", field=path)
            if key in _FORBIDDEN_SEMANTIC_ROUTE_KEYS or key.endswith(
                "_route_binding_sha256"
            ):
                raise contract_invalid(
                    "route-specific authority is forbidden in semantic acquisition",
                    field=f"{path}.{key}",
                )
            _reject_route_authority_from_semantic_core(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_route_authority_from_semantic_core(item, path=f"{path}[{index}]")


def _payload_value(payload: CanonicalPayloadV1, *, field: str) -> dict[str, object]:
    require_instance(payload, CanonicalPayloadV1, field=field)
    return _mapping(payload.to_value(), field=field)


def _validate_sorted_unique_ticks(
    values: object,
    *,
    field: str,
    start_next_tick: int,
    terminal_tick: int,
) -> None:
    if not isinstance(values, list):
        raise contract_invalid("tick schedule must be a list", field=field)
    ticks = [require_int(item, field=field) for item in values]
    if ticks != sorted(set(ticks)):
        raise contract_invalid("tick schedule must be sorted and unique", field=field)
    if any(item < start_next_tick or item > terminal_tick for item in ticks):
        raise contract_invalid("tick lies outside the run", field=field)


def _validate_campaign_time(payload: CanonicalPayloadV1) -> None:
    value = _payload_value(payload, field="time")
    if value.get("authority") != "absolute_tick_v1":
        raise contract_invalid(
            "absolute_tick_v1 is the sole public time authority",
            field="time.authority",
        )
    start_next_tick = require_int(
        value.get("start_next_tick", 0), field="time.start_next_tick"
    )
    total_ticks = require_int(
        value.get("total_ticks"), field="time.total_ticks", minimum=1
    )
    if start_next_tick > INT64_MAX - total_ticks:
        raise contract_invalid(
            "run tick span overflows int64", field="time.total_ticks"
        )
    terminal_tick = start_next_tick + total_ticks
    for key in ("observation_ticks", "checkpoint_ticks"):
        _validate_sorted_unique_ticks(
            value.get(key, []),
            field=f"time.{key}",
            start_next_tick=start_next_tick,
            terminal_tick=terminal_tick,
        )
    branches = value.get("branches", [])
    if not isinstance(branches, list):
        raise contract_invalid("branches must be a list", field="time.branches")
    for branch in branches:
        _mapping(
            branch,
            field="time.branches[]",
            exact_keys={
                "branch_id",
                "parent_checkpoint_tick",
                "intervention_control_ids",
            },
        )
        require_text(branch["branch_id"], field="time.branches[].branch_id")
        tick = require_int(
            branch["parent_checkpoint_tick"],
            field="time.branches[].parent_checkpoint_tick",
        )
        if tick not in value.get("checkpoint_ticks", []):
            raise contract_invalid(
                "branch must reference a declared absolute checkpoint tick",
                field="time.branches[].parent_checkpoint_tick",
            )
        controls = branch["intervention_control_ids"]
        if not isinstance(controls, list) or any(
            type(item) is not str for item in controls
        ):
            raise contract_invalid(
                "branch intervention controls must be strings",
                field="time.branches[].intervention_control_ids",
            )


@dataclass(frozen=True, slots=True)
class ScienceCampaignV1:
    """Strict normalized rich campaign; named experiments have no dispatch role."""

    campaign_id: str
    instrument: CanonicalPayloadV1
    question: CanonicalPayloadV1
    design: CanonicalPayloadV1
    world: CanonicalPayloadV1
    initialization: CanonicalPayloadV1
    controls: CanonicalPayloadV1
    seeds: CanonicalPayloadV1
    time: CanonicalPayloadV1
    observations: CanonicalPayloadV1
    execution: CanonicalPayloadV1
    reducers: CanonicalPayloadV1
    artifact_lifecycle: CanonicalPayloadV1
    decision: CanonicalPayloadV1
    product_id: str = PRODUCT_ID
    schema_version: SchemaVersion = SCIENCE_CAMPAIGN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.campaign_id, field="campaign_id")
        if self.product_id != PRODUCT_ID:
            raise contract_invalid("unsupported science instrument", field="product_id")
        _validate_schema_version(
            self.schema_version, expected=SCIENCE_CAMPAIGN_SCHEMA_VERSION
        )
        for field_name in _CAMPAIGN_SECTION_FIELDS:
            payload = getattr(self, field_name)
            require_instance(payload, CanonicalPayloadV1, field=field_name)
            _reject_generation_authority(payload.to_value(), path=field_name)
        instrument = _payload_value(self.instrument, field="instrument")
        if instrument.get("product_id") != PRODUCT_ID:
            raise contract_invalid(
                "campaign instrument product_id is not canonical",
                field="instrument.product_id",
            )
        _digest_from_dict(
            instrument.get("instrument_manifest_sha256"),
            field="instrument.instrument_manifest_sha256",
        )
        _validate_campaign_time(self.time)
        lifecycle = _payload_value(self.artifact_lifecycle, field="artifact_lifecycle")
        if lifecycle.get("profile_id") != "SCIENCE_COMPACT":
            raise contract_invalid(
                "normalized rich campaigns require SCIENCE_COMPACT",
                field="artifact_lifecycle.profile_id",
            )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": str(self.schema_version),
            "product_id": self.product_id,
            "campaign_id": self.campaign_id,
        }
        result.update(
            {
                field_name: getattr(self, field_name).to_dict()
                for field_name in _CAMPAIGN_SECTION_FIELDS
            }
        )
        return result

    @classmethod
    def from_dict(cls, value: object) -> "ScienceCampaignV1":
        expected = {
            "schema_version",
            "product_id",
            "campaign_id",
            *_CAMPAIGN_SECTION_FIELDS,
        }
        data = _mapping(value, field="science_campaign", exact_keys=expected)
        kwargs = {
            field_name: _payload_from_dict(data[field_name], field=field_name)
            for field_name in _CAMPAIGN_SECTION_FIELDS
        }
        return cls(
            campaign_id=data["campaign_id"],
            product_id=data["product_id"],
            schema_version=SchemaVersion(data["schema_version"]),
            **kwargs,
        )

    @classmethod
    def from_campaign_document(cls, value: object) -> "ScienceCampaignV1":
        """Normalize a schema-validated rich wire document into immutable payloads."""

        allowed = {
            "schema_version",
            "product_id",
            "campaign_id",
            "instrument",
            "question",
            "design",
            "world",
            "initialization",
            "controls",
            "seeds",
            "time",
            "observations",
            "execution",
            "reducers",
            "artifact_lifecycle",
            "decision",
            "legacy_compatibility",
        }
        data = _mapping(value, field="science_campaign_document")
        if not set(data).issubset(allowed):
            raise contract_invalid(
                "science campaign document contains unknown fields",
                field="science_campaign_document",
            )
        required = allowed - {
            "artifact_lifecycle",
            "decision",
            "legacy_compatibility",
        }
        if not required.issubset(data):
            raise contract_invalid(
                "science campaign document is missing required fields",
                field="science_campaign_document",
            )
        if data["schema_version"] != str(SCIENCE_CAMPAIGN_SCHEMA_VERSION):
            raise contract_invalid(
                "unsupported campaign document schema", field="schema_version"
            )
        if data["product_id"] != PRODUCT_ID:
            raise contract_invalid("unsupported science instrument", field="product_id")
        section_values = {
            field_name: data[field_name]
            for field_name in _CAMPAIGN_SECTION_FIELDS
            if field_name in data
        }
        section_values.setdefault(
            "artifact_lifecycle",
            {
                "profile_id": "SCIENCE_COMPACT",
                "contract_ref": "eiel_run_artifact_lifecycle_contract_v1",
                "explicit_intent_allowed_override": None,
            },
        )
        section_values.setdefault("decision", {})
        return cls(
            campaign_id=data["campaign_id"],
            instrument=_payload_for_section("instrument", section_values["instrument"]),
            question=_payload_for_section("question", section_values["question"]),
            design=_payload_for_section("design", section_values["design"]),
            world=_payload_for_section("world", section_values["world"]),
            initialization=_payload_for_section(
                "initialization", section_values["initialization"]
            ),
            controls=_payload_for_section("controls", section_values["controls"]),
            seeds=_payload_for_section("seeds", section_values["seeds"]),
            time=_payload_for_section("time", section_values["time"]),
            observations=_payload_for_section(
                "observations", section_values["observations"]
            ),
            execution=_payload_for_section("execution", section_values["execution"]),
            reducers=_payload_for_section("reducers", section_values["reducers"]),
            artifact_lifecycle=_payload_for_section(
                "artifact_lifecycle", section_values["artifact_lifecycle"]
            ),
            decision=_payload_for_section("decision", section_values["decision"]),
        )

    def to_campaign_document(self) -> dict[str, object]:
        """Return the normalized wire form accepted by the campaign schema."""

        result: dict[str, object] = {
            "schema_version": str(self.schema_version),
            "product_id": self.product_id,
            "campaign_id": self.campaign_id,
        }
        result.update(
            {
                field_name: getattr(self, field_name).to_value()
                for field_name in _CAMPAIGN_SECTION_FIELDS
            }
        )
        return result

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def campaign_spec_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ScienceControlDescriptorV1:
    control_id: str
    value_contract: CanonicalPayloadV1
    unit: str
    activation_phases: tuple[str, ...]
    heterogeneity_modes: tuple[str, ...]
    schedule_kinds: tuple[str, ...]
    readback_capability_ids: tuple[str, ...]
    interaction_ids: tuple[str, ...]
    component_id: str
    qualified_envelope_subjects: tuple[str, ...]
    schema_version: SchemaVersion = SCIENCE_CONTROL_DESCRIPTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_token(self.control_id, field="control_id")
        require_instance(
            self.value_contract, CanonicalPayloadV1, field="value_contract"
        )
        require_text(self.unit, field="unit")
        _require_token(self.component_id, field="component_id")
        for field_name in (
            "activation_phases",
            "heterogeneity_modes",
            "schedule_kinds",
            "readback_capability_ids",
            "interaction_ids",
            "qualified_envelope_subjects",
        ):
            _require_token_tuple(getattr(self, field_name), field=field_name)
        if not set(self.schedule_kinds).issubset(_CONTROL_SCHEDULE_KINDS):
            raise contract_invalid(
                "unsupported control schedule kind", field="schedule_kinds"
            )
        _validate_schema_version(
            self.schema_version, expected=SCIENCE_CONTROL_DESCRIPTOR_SCHEMA_VERSION
        )
        _reject_generation_authority(
            self.value_contract.to_value(), path="value_contract"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "control_id": self.control_id,
            "value_contract": self.value_contract.to_dict(),
            "unit": self.unit,
            "activation_phases": list(self.activation_phases),
            "heterogeneity_modes": list(self.heterogeneity_modes),
            "schedule_kinds": list(self.schedule_kinds),
            "readback_capability_ids": list(self.readback_capability_ids),
            "interaction_ids": list(self.interaction_ids),
            "component_id": self.component_id,
            "qualified_envelope_subjects": list(self.qualified_envelope_subjects),
        }

    def descriptor_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ScienceReducerDescriptorV1:
    reducer_id: str
    contract_version: SchemaVersion
    implementation_sha256: ContentDigest
    configuration_schema_sha256: ContentDigest
    required_observation_capability_ids: tuple[str, ...]
    output_contract_id: str
    purity_declarations: tuple[str, ...]
    schema_version: SchemaVersion = SCIENCE_REDUCER_DESCRIPTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_token(self.reducer_id, field="reducer_id")
        require_instance(self.contract_version, SchemaVersion, field="contract_version")
        require_instance(
            self.implementation_sha256, ContentDigest, field="implementation_sha256"
        )
        require_instance(
            self.configuration_schema_sha256,
            ContentDigest,
            field="configuration_schema_sha256",
        )
        _require_token_tuple(
            self.required_observation_capability_ids,
            field="required_observation_capability_ids",
        )
        _require_token(self.output_contract_id, field="output_contract_id")
        _require_token_tuple(self.purity_declarations, field="purity_declarations")
        declarations = set(self.purity_declarations)
        if not declarations.issubset(_REDUCER_PURITY_DECLARATIONS):
            raise contract_invalid(
                "unsupported reducer purity declaration",
                field="purity_declarations",
            )
        if not _REQUIRED_REDUCER_PURITY_DECLARATIONS.issubset(declarations):
            raise contract_invalid(
                "required reducer purity declarations are missing",
                field="purity_declarations",
            )
        _validate_schema_version(
            self.schema_version, expected=SCIENCE_REDUCER_DESCRIPTOR_SCHEMA_VERSION
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "reducer_id": self.reducer_id,
            "contract_version": str(self.contract_version),
            "implementation_sha256": self.implementation_sha256.to_dict(),
            "configuration_schema_sha256": self.configuration_schema_sha256.to_dict(),
            "required_observation_capability_ids": list(
                self.required_observation_capability_ids
            ),
            "output_contract_id": self.output_contract_id,
            "purity_declarations": list(self.purity_declarations),
        }

    def descriptor_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ScienceInstrumentManifestV1:
    component_graph_sha256: ContentDigest
    control_catalog_sha256: ContentDigest
    world_profile_catalog_sha256: ContentDigest
    founder_profile_catalog_sha256: ContentDigest
    policy_profile_catalog_sha256: ContentDigest
    mutation_operator_catalog_sha256: ContentDigest
    observation_abi_sha256: ContentDigest
    observation_profile_catalog_sha256: ContentDigest
    reducer_protocol_version: SchemaVersion
    route_binding_protocol_version: SchemaVersion
    eed_release_status: EEDReleaseStatusV1
    product_id: str = PRODUCT_ID
    schema_version: SchemaVersion = SCIENCE_INSTRUMENT_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.product_id != PRODUCT_ID:
            raise contract_invalid("unsupported science instrument", field="product_id")
        for field_name in (
            "component_graph_sha256",
            "control_catalog_sha256",
            "world_profile_catalog_sha256",
            "founder_profile_catalog_sha256",
            "policy_profile_catalog_sha256",
            "mutation_operator_catalog_sha256",
            "observation_abi_sha256",
            "observation_profile_catalog_sha256",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        require_instance(
            self.reducer_protocol_version,
            SchemaVersion,
            field="reducer_protocol_version",
        )
        require_instance(
            self.route_binding_protocol_version,
            SchemaVersion,
            field="route_binding_protocol_version",
        )
        if self.route_binding_protocol_version != SCIENCE_ROUTE_BINDING_SCHEMA_VERSION:
            raise contract_invalid(
                "unsupported route binding protocol",
                field="route_binding_protocol_version",
            )
        require_instance(
            self.eed_release_status, EEDReleaseStatusV1, field="eed_release_status"
        )
        _validate_schema_version(
            self.schema_version, expected=SCIENCE_INSTRUMENT_MANIFEST_SCHEMA_VERSION
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "product_id": self.product_id,
            "component_graph_sha256": self.component_graph_sha256.to_dict(),
            "control_catalog_sha256": self.control_catalog_sha256.to_dict(),
            "world_profile_catalog_sha256": self.world_profile_catalog_sha256.to_dict(),
            "founder_profile_catalog_sha256": self.founder_profile_catalog_sha256.to_dict(),
            "policy_profile_catalog_sha256": self.policy_profile_catalog_sha256.to_dict(),
            "mutation_operator_catalog_sha256": self.mutation_operator_catalog_sha256.to_dict(),
            "observation_abi_sha256": self.observation_abi_sha256.to_dict(),
            "observation_profile_catalog_sha256": self.observation_profile_catalog_sha256.to_dict(),
            "reducer_protocol_version": str(self.reducer_protocol_version),
            "route_binding_protocol_version": str(self.route_binding_protocol_version),
            "eed_release_status": self.eed_release_status.value,
        }

    def manifest_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ScienceRouteBindingV1:
    """Content-addressed physical-route binding for a route-neutral instrument."""

    instrument_manifest_sha256: ContentDigest
    route_id: str
    route_profile_sha256: ContentDigest
    machine_profile_id: str
    capability_projection_sha256: ContentDigest
    native_build_closure_sha256: ContentDigest
    execution_adapter_sha256: ContentDigest
    control_adapter_sha256: ContentDigest
    observation_adapter_sha256: ContentDigest
    checkpoint_adapter_sha256: ContentDigest
    semantic_conformance_sha256: ContentDigest
    resource_envelope_sha256: ContentDigest
    qualified_envelope_sha256: ContentDigest
    qualification_authority_sha256: ContentDigest
    schema_version: SchemaVersion = SCIENCE_ROUTE_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
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
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        _require_route_token(self.route_id, field="route_id")
        _require_route_token(self.machine_profile_id, field="machine_profile_id")
        _validate_schema_version(
            self.schema_version, expected=SCIENCE_ROUTE_BINDING_SCHEMA_VERSION
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "instrument_manifest_sha256": self.instrument_manifest_sha256.to_dict(),
            "route_id": self.route_id,
            "route_profile_sha256": self.route_profile_sha256.to_dict(),
            "machine_profile_id": self.machine_profile_id,
            "capability_projection_sha256": self.capability_projection_sha256.to_dict(),
            "native_build_closure_sha256": self.native_build_closure_sha256.to_dict(),
            "execution_adapter_sha256": self.execution_adapter_sha256.to_dict(),
            "control_adapter_sha256": self.control_adapter_sha256.to_dict(),
            "observation_adapter_sha256": self.observation_adapter_sha256.to_dict(),
            "checkpoint_adapter_sha256": self.checkpoint_adapter_sha256.to_dict(),
            "semantic_conformance_sha256": self.semantic_conformance_sha256.to_dict(),
            "resource_envelope_sha256": self.resource_envelope_sha256.to_dict(),
            "qualified_envelope_sha256": self.qualified_envelope_sha256.to_dict(),
            "qualification_authority_sha256": self.qualification_authority_sha256.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ScienceRouteBindingV1":
        data = _mapping(
            value,
            field="science_route_binding",
            exact_keys={
                "schema_version",
                "instrument_manifest_sha256",
                "route_id",
                "route_profile_sha256",
                "machine_profile_id",
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
            },
        )
        return cls(
            schema_version=SchemaVersion(data["schema_version"]),
            instrument_manifest_sha256=_digest_from_dict(
                data["instrument_manifest_sha256"],
                field="instrument_manifest_sha256",
            ),
            route_id=data["route_id"],
            route_profile_sha256=_digest_from_dict(
                data["route_profile_sha256"], field="route_profile_sha256"
            ),
            machine_profile_id=data["machine_profile_id"],
            capability_projection_sha256=_digest_from_dict(
                data["capability_projection_sha256"],
                field="capability_projection_sha256",
            ),
            native_build_closure_sha256=_digest_from_dict(
                data["native_build_closure_sha256"],
                field="native_build_closure_sha256",
            ),
            execution_adapter_sha256=_digest_from_dict(
                data["execution_adapter_sha256"], field="execution_adapter_sha256"
            ),
            control_adapter_sha256=_digest_from_dict(
                data["control_adapter_sha256"], field="control_adapter_sha256"
            ),
            observation_adapter_sha256=_digest_from_dict(
                data["observation_adapter_sha256"],
                field="observation_adapter_sha256",
            ),
            checkpoint_adapter_sha256=_digest_from_dict(
                data["checkpoint_adapter_sha256"],
                field="checkpoint_adapter_sha256",
            ),
            semantic_conformance_sha256=_digest_from_dict(
                data["semantic_conformance_sha256"],
                field="semantic_conformance_sha256",
            ),
            resource_envelope_sha256=_digest_from_dict(
                data["resource_envelope_sha256"], field="resource_envelope_sha256"
            ),
            qualified_envelope_sha256=_digest_from_dict(
                data["qualified_envelope_sha256"],
                field="qualified_envelope_sha256",
            ),
            qualification_authority_sha256=_digest_from_dict(
                data["qualification_authority_sha256"],
                field="qualification_authority_sha256",
            ),
        )

    def route_binding_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ScienceExecutionPlanV1:
    instrument_manifest_sha256: ContentDigest
    route_binding_sha256: ContentDigest
    crca_semantic_run_digest: ContentDigest
    semantic_acquisition_core: CanonicalPayloadV1
    physical_execution_core: CanonicalPayloadV1
    legacy_campaign_plan_sha256: ContentDigest | None = None
    schema_version: SchemaVersion = SCIENCE_EXECUTION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(
            self.instrument_manifest_sha256,
            ContentDigest,
            field="instrument_manifest_sha256",
        )
        require_instance(
            self.route_binding_sha256,
            ContentDigest,
            field="route_binding_sha256",
        )
        require_instance(
            self.crca_semantic_run_digest,
            ContentDigest,
            field="crca_semantic_run_digest",
        )
        require_instance(
            self.semantic_acquisition_core,
            CanonicalPayloadV1,
            field="semantic_acquisition_core",
        )
        require_instance(
            self.physical_execution_core,
            CanonicalPayloadV1,
            field="physical_execution_core",
        )
        if self.semantic_acquisition_core.schema_version != (
            SEMANTIC_ACQUISITION_CORE_SCHEMA_VERSION
        ):
            raise contract_invalid(
                "semantic acquisition core has the wrong schema version",
                field="semantic_acquisition_core.schema_version",
            )
        if self.physical_execution_core.schema_version != (
            PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION
        ):
            raise contract_invalid(
                "physical execution core has the wrong schema version",
                field="physical_execution_core.schema_version",
            )
        if self.legacy_campaign_plan_sha256 is not None:
            require_instance(
                self.legacy_campaign_plan_sha256,
                ContentDigest,
                field="legacy_campaign_plan_sha256",
            )
        _validate_schema_version(
            self.schema_version, expected=SCIENCE_EXECUTION_PLAN_SCHEMA_VERSION
        )
        _reject_generation_authority(
            self.semantic_acquisition_core.to_value(), path="semantic_acquisition_core"
        )
        _reject_generation_authority(
            self.physical_execution_core.to_value(), path="physical_execution_core"
        )
        _reject_route_authority_from_semantic_core(
            self.semantic_acquisition_core.to_value()
        )

    def science_semantic_acquisition_sha256(self) -> ContentDigest:
        return canonical_sha256(
            {
                "schema_version": str(SEMANTIC_ACQUISITION_CORE_SCHEMA_VERSION),
                "core": self.semantic_acquisition_core.to_value(),
            }
        )

    def physical_execution_plan_sha256(self) -> ContentDigest:
        return canonical_sha256(
            {
                "schema_version": str(PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION),
                "route_binding_sha256": self.route_binding_sha256.to_dict(),
                "core": self.physical_execution_core.to_value(),
            }
        )

    def execution_core_dict(self) -> dict[str, object]:
        return {
            "instrument_manifest_sha256": self.instrument_manifest_sha256.to_dict(),
            "route_binding_sha256": self.route_binding_sha256.to_dict(),
            "crca_semantic_run_digest": self.crca_semantic_run_digest.to_dict(),
            "science_semantic_acquisition_sha256": self.science_semantic_acquisition_sha256().to_dict(),
            "physical_execution_plan_sha256": self.physical_execution_plan_sha256().to_dict(),
        }

    def science_execution_plan_sha256(self) -> ContentDigest:
        return canonical_sha256(
            {
                "schema_version": str(self.schema_version),
                "execution_core": self.execution_core_dict(),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "execution_core": self.execution_core_dict(),
            "semantic_acquisition_core": self.semantic_acquisition_core.to_dict(),
            "physical_execution_core": self.physical_execution_core.to_dict(),
            "identities": {
                "science_semantic_acquisition_sha256": self.science_semantic_acquisition_sha256().to_dict(),
                "physical_execution_plan_sha256": self.physical_execution_plan_sha256().to_dict(),
                "science_execution_plan_sha256": self.science_execution_plan_sha256().to_dict(),
            },
            "legacy_bindings": {
                "legacy_campaign_plan_sha256": (
                    None
                    if self.legacy_campaign_plan_sha256 is None
                    else self.legacy_campaign_plan_sha256.to_dict()
                )
            },
        }

    @classmethod
    def from_dict(cls, value: object) -> "ScienceExecutionPlanV1":
        data = _mapping(
            value,
            field="science_execution_plan",
            exact_keys={
                "schema_version",
                "execution_core",
                "semantic_acquisition_core",
                "physical_execution_core",
                "identities",
                "legacy_bindings",
            },
        )
        execution_core = _mapping(
            data["execution_core"],
            field="execution_core",
            exact_keys={
                "instrument_manifest_sha256",
                "route_binding_sha256",
                "crca_semantic_run_digest",
                "science_semantic_acquisition_sha256",
                "physical_execution_plan_sha256",
            },
        )
        legacy = _mapping(
            data["legacy_bindings"],
            field="legacy_bindings",
            exact_keys={"legacy_campaign_plan_sha256"},
        )
        result = cls(
            instrument_manifest_sha256=_digest_from_dict(
                execution_core["instrument_manifest_sha256"],
                field="instrument_manifest_sha256",
            ),
            route_binding_sha256=_digest_from_dict(
                execution_core["route_binding_sha256"],
                field="route_binding_sha256",
            ),
            crca_semantic_run_digest=_digest_from_dict(
                execution_core["crca_semantic_run_digest"],
                field="crca_semantic_run_digest",
            ),
            semantic_acquisition_core=_payload_from_dict(
                data["semantic_acquisition_core"], field="semantic_acquisition_core"
            ),
            physical_execution_core=_payload_from_dict(
                data["physical_execution_core"], field="physical_execution_core"
            ),
            legacy_campaign_plan_sha256=_optional_digest_from_dict(
                legacy["legacy_campaign_plan_sha256"],
                field="legacy_campaign_plan_sha256",
            ),
            schema_version=SchemaVersion(data["schema_version"]),
        )
        identities = _mapping(
            data["identities"],
            field="identities",
            exact_keys={
                "science_semantic_acquisition_sha256",
                "physical_execution_plan_sha256",
                "science_execution_plan_sha256",
            },
        )
        expected = {
            "science_semantic_acquisition_sha256": result.science_semantic_acquisition_sha256(),
            "physical_execution_plan_sha256": result.physical_execution_plan_sha256(),
            "science_execution_plan_sha256": result.science_execution_plan_sha256(),
        }
        for field_name, digest in expected.items():
            if _digest_from_dict(identities[field_name], field=field_name) != digest:
                raise contract_invalid("identity digest mismatch", field=field_name)
        if data["execution_core"] != result.execution_core_dict():
            raise contract_invalid("execution core mismatch", field="execution_core")
        return result


@dataclass(frozen=True, slots=True)
class ScienceReductionPlanV1:
    reducer_descriptors: tuple[ScienceReducerDescriptorV1, ...]
    normalized_configurations: CanonicalPayloadV1
    evidence_requirements: CanonicalPayloadV1
    offline_decisions: CanonicalPayloadV1
    output_contract_ids: tuple[str, ...]
    schema_version: SchemaVersion = SCIENCE_REDUCTION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_tuple_instances(
            self.reducer_descriptors,
            ScienceReducerDescriptorV1,
            field="reducer_descriptors",
            unique=True,
        )
        reducer_ids = [item.reducer_id for item in self.reducer_descriptors]
        if reducer_ids != sorted(set(reducer_ids)):
            raise contract_invalid(
                "reducers must be sorted and unique by reducer_id",
                field="reducer_descriptors",
            )
        for field_name in (
            "normalized_configurations",
            "evidence_requirements",
            "offline_decisions",
        ):
            require_instance(
                getattr(self, field_name), CanonicalPayloadV1, field=field_name
            )
        require_sorted_unique_text(
            self.output_contract_ids, field="output_contract_ids"
        )
        _validate_schema_version(
            self.schema_version, expected=SCIENCE_REDUCTION_PLAN_SCHEMA_VERSION
        )

    def reduction_core_dict(self) -> dict[str, object]:
        return {
            "reducer_descriptors": [
                item.to_dict() for item in self.reducer_descriptors
            ],
            "normalized_configurations": self.normalized_configurations.to_value(),
            "evidence_requirements": self.evidence_requirements.to_value(),
            "offline_decisions": self.offline_decisions.to_value(),
            "output_contract_ids": list(self.output_contract_ids),
        }

    def reduction_plan_sha256(self) -> ContentDigest:
        return canonical_sha256(
            {
                "schema_version": str(self.schema_version),
                "reduction_core": self.reduction_core_dict(),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "reduction_core": self.reduction_core_dict(),
            "reduction_plan_sha256": self.reduction_plan_sha256().to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ScienceFindingV1:
    code: str
    subject_path: str
    detail: str
    evidence_refs: tuple[str, ...] = ()
    extension_target: str | None = None
    schema_version: SchemaVersion = SCIENCE_FINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.code, field="code")
        require_text(self.subject_path, field="subject_path")
        require_text(self.detail, field="detail")
        require_sorted_unique_text(self.evidence_refs, field="evidence_refs")
        if self.extension_target is not None:
            require_text(self.extension_target, field="extension_target")
        _validate_schema_version(
            self.schema_version, expected=SCIENCE_FINDING_SCHEMA_VERSION
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "code": self.code,
            "subject_path": self.subject_path,
            "detail": self.detail,
            "evidence_refs": list(self.evidence_refs),
            "extension_target": self.extension_target,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ScienceFindingV1":
        data = _mapping(
            value,
            field="science_finding",
            exact_keys={
                "schema_version",
                "code",
                "subject_path",
                "detail",
                "evidence_refs",
                "extension_target",
            },
        )
        evidence_refs = data["evidence_refs"]
        if not isinstance(evidence_refs, list):
            raise contract_invalid(
                "evidence_refs must be a list", field="evidence_refs"
            )
        return cls(
            schema_version=SchemaVersion(data["schema_version"]),
            code=data["code"],
            subject_path=data["subject_path"],
            detail=data["detail"],
            evidence_refs=tuple(evidence_refs),
            extension_target=data["extension_target"],
        )


@dataclass(frozen=True, slots=True)
class ScienceAdmissionReceiptV1:
    campaign_spec_sha256: ContentDigest
    instrument_manifest_sha256: ContentDigest
    route_binding_sha256: ContentDigest
    qualified_envelope_projection_sha256: ContentDigest
    machine_profile_id: str
    machine_profile_sha256: ContentDigest
    lifecycle_contract_sha256: ContentDigest
    campaign_compiler_component_sha256: ContentDigest
    science_semantic_acquisition_sha256: ContentDigest
    physical_execution_plan_sha256: ContentDigest
    science_execution_plan_sha256: ContentDigest
    reduction_plan_sha256: ContentDigest
    compilation_bundle_sha256: ContentDigest
    active_qrr_resolution_receipt_sha256: ContentDigest
    canonical_crca_semantic_identity: ContentDigest
    admission_kernel_receipt_sha256: ContentDigest
    findings: tuple[ScienceFindingV1, ...]
    primary_outcome: ScienceAdmissionOutcomeV1
    legacy_campaign_plan_sha256: ContentDigest | None = None
    rich_to_flat_projection_receipt_sha256: ContentDigest | None = None
    science_qrr_binding_receipt_sha256: ContentDigest | None = None
    effects_performed: bool = False
    schema_version: SchemaVersion = SCIENCE_ADMISSION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "campaign_spec_sha256",
            "instrument_manifest_sha256",
            "route_binding_sha256",
            "qualified_envelope_projection_sha256",
            "machine_profile_sha256",
            "lifecycle_contract_sha256",
            "campaign_compiler_component_sha256",
            "science_semantic_acquisition_sha256",
            "physical_execution_plan_sha256",
            "science_execution_plan_sha256",
            "reduction_plan_sha256",
            "compilation_bundle_sha256",
            "active_qrr_resolution_receipt_sha256",
            "canonical_crca_semantic_identity",
            "admission_kernel_receipt_sha256",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        for field_name in (
            "legacy_campaign_plan_sha256",
            "rich_to_flat_projection_receipt_sha256",
            "science_qrr_binding_receipt_sha256",
        ):
            value = getattr(self, field_name)
            if value is not None:
                require_instance(value, ContentDigest, field=field_name)
        require_text(self.machine_profile_id, field="machine_profile_id")
        require_tuple_instances(
            self.findings, ScienceFindingV1, field="findings", unique=True
        )
        finding_order = [
            (
                item.subject_path,
                item.code,
                item.detail,
                item.extension_target or "",
                item.evidence_refs,
            )
            for item in self.findings
        ]
        if finding_order != sorted(finding_order):
            raise contract_invalid(
                "findings must be in canonical subject order", field="findings"
            )
        require_instance(
            self.primary_outcome,
            ScienceAdmissionOutcomeV1,
            field="primary_outcome",
        )
        if type(self.effects_performed) is not bool or self.effects_performed:
            raise contract_invalid(
                "science admission is effect-free", field="effects_performed"
            )
        _validate_schema_version(
            self.schema_version, expected=SCIENCE_ADMISSION_RECEIPT_SCHEMA_VERSION
        )

    def to_dict(self) -> dict[str, object]:
        digest_fields = (
            "campaign_spec_sha256",
            "instrument_manifest_sha256",
            "route_binding_sha256",
            "qualified_envelope_projection_sha256",
            "machine_profile_sha256",
            "lifecycle_contract_sha256",
            "campaign_compiler_component_sha256",
            "science_semantic_acquisition_sha256",
            "physical_execution_plan_sha256",
            "science_execution_plan_sha256",
            "reduction_plan_sha256",
            "compilation_bundle_sha256",
            "active_qrr_resolution_receipt_sha256",
            "canonical_crca_semantic_identity",
            "admission_kernel_receipt_sha256",
        )
        result: dict[str, object] = {
            "schema_version": str(self.schema_version),
            "machine_profile_id": self.machine_profile_id,
            "findings": [item.to_dict() for item in self.findings],
            "primary_outcome": self.primary_outcome.value,
            "effects_performed": self.effects_performed,
        }
        result.update(
            {
                field_name: getattr(self, field_name).to_dict()
                for field_name in digest_fields
            }
        )
        for field_name in (
            "legacy_campaign_plan_sha256",
            "rich_to_flat_projection_receipt_sha256",
            "science_qrr_binding_receipt_sha256",
        ):
            value = getattr(self, field_name)
            result[field_name] = None if value is None else value.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: object) -> "ScienceAdmissionReceiptV1":
        digest_fields = {
            "campaign_spec_sha256",
            "instrument_manifest_sha256",
            "route_binding_sha256",
            "qualified_envelope_projection_sha256",
            "machine_profile_sha256",
            "lifecycle_contract_sha256",
            "campaign_compiler_component_sha256",
            "science_semantic_acquisition_sha256",
            "physical_execution_plan_sha256",
            "science_execution_plan_sha256",
            "reduction_plan_sha256",
            "compilation_bundle_sha256",
            "active_qrr_resolution_receipt_sha256",
            "canonical_crca_semantic_identity",
            "admission_kernel_receipt_sha256",
        }
        optional_digest_fields = {
            "legacy_campaign_plan_sha256",
            "rich_to_flat_projection_receipt_sha256",
            "science_qrr_binding_receipt_sha256",
        }
        expected = {
            "schema_version",
            "machine_profile_id",
            "findings",
            "primary_outcome",
            "effects_performed",
            *digest_fields,
            *optional_digest_fields,
        }
        data = _mapping(
            value, field="science_admission_receipt", exact_keys=expected
        )
        findings = data["findings"]
        if not isinstance(findings, list):
            raise contract_invalid("findings must be a list", field="findings")
        try:
            primary_outcome = ScienceAdmissionOutcomeV1(data["primary_outcome"])
        except (TypeError, ValueError) as error:
            raise contract_invalid(
                "unsupported science admission outcome", field="primary_outcome"
            ) from error
        return cls(
            **{
                field_name: _digest_from_dict(data[field_name], field=field_name)
                for field_name in digest_fields
            },
            machine_profile_id=data["machine_profile_id"],
            findings=tuple(ScienceFindingV1.from_dict(item) for item in findings),
            primary_outcome=primary_outcome,
            **{
                field_name: _optional_digest_from_dict(
                    data[field_name], field=field_name
                )
                for field_name in optional_digest_fields
            },
            effects_performed=data["effects_performed"],
            schema_version=SchemaVersion(data["schema_version"]),
        )

    def receipt_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ScienceResultReceiptV1:
    campaign_spec_sha256: ContentDigest
    instrument_manifest_sha256: ContentDigest
    route_binding_sha256: ContentDigest
    science_semantic_acquisition_sha256: ContentDigest
    physical_execution_plan_sha256: ContentDigest
    science_execution_plan_sha256: ContentDigest
    evidence_manifest_sha256: ContentDigest
    reduction_plan_sha256: ContentDigest
    reducer_descriptor_sha256: ContentDigest
    reducer_implementation_sha256: ContentDigest
    reducer_configuration_sha256: ContentDigest
    output_contract_id: str
    result_sha256: ContentDigest
    resume_chain_sha256: ContentDigest | None = None
    trajectory_effects_performed: bool = False
    schema_version: SchemaVersion = SCIENCE_RESULT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "campaign_spec_sha256",
            "instrument_manifest_sha256",
            "route_binding_sha256",
            "science_semantic_acquisition_sha256",
            "physical_execution_plan_sha256",
            "science_execution_plan_sha256",
            "evidence_manifest_sha256",
            "reduction_plan_sha256",
            "reducer_descriptor_sha256",
            "reducer_implementation_sha256",
            "reducer_configuration_sha256",
            "result_sha256",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        if self.resume_chain_sha256 is not None:
            require_instance(
                self.resume_chain_sha256,
                ContentDigest,
                field="resume_chain_sha256",
            )
        _require_token(self.output_contract_id, field="output_contract_id")
        if (
            type(self.trajectory_effects_performed) is not bool
            or self.trajectory_effects_performed
        ):
            raise contract_invalid(
                "offline reduction cannot affect trajectory state",
                field="trajectory_effects_performed",
            )
        _validate_schema_version(
            self.schema_version, expected=SCIENCE_RESULT_RECEIPT_SCHEMA_VERSION
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "campaign_spec_sha256": self.campaign_spec_sha256.to_dict(),
            "instrument_manifest_sha256": self.instrument_manifest_sha256.to_dict(),
            "route_binding_sha256": self.route_binding_sha256.to_dict(),
            "science_semantic_acquisition_sha256": self.science_semantic_acquisition_sha256.to_dict(),
            "physical_execution_plan_sha256": self.physical_execution_plan_sha256.to_dict(),
            "science_execution_plan_sha256": self.science_execution_plan_sha256.to_dict(),
            "evidence_manifest_sha256": self.evidence_manifest_sha256.to_dict(),
            "reduction_plan_sha256": self.reduction_plan_sha256.to_dict(),
            "reducer_descriptor_sha256": self.reducer_descriptor_sha256.to_dict(),
            "reducer_implementation_sha256": self.reducer_implementation_sha256.to_dict(),
            "reducer_configuration_sha256": self.reducer_configuration_sha256.to_dict(),
            "output_contract_id": self.output_contract_id,
            "result_sha256": self.result_sha256.to_dict(),
            "resume_chain_sha256": (
                None
                if self.resume_chain_sha256 is None
                else self.resume_chain_sha256.to_dict()
            ),
            "trajectory_effects_performed": self.trajectory_effects_performed,
        }

    def receipt_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ScienceObservationRecordV1:
    instrument_manifest_sha256: ContentDigest
    producer_component_id: str
    producer_contract_version: SchemaVersion
    semantic_execution_id: ContentDigest
    crca_semantic_run_digest: ContentDigest
    science_semantic_acquisition_sha256: ContentDigest
    cell_id: str
    logical_world_id: int
    world_uid: ContentDigest
    absolute_tick: int
    semantic_source_address: str
    pre_sequence_key_sha256: ContentDigest
    world_local_sequence: int
    deterministic_event_id: ContentDigest
    record_kind: ScienceRecordKindV1
    record_kind_version: int
    control_binding_sha256: ContentDigest
    payload: CanonicalPayloadV1
    source_state_before_sha256: ContentDigest | None = None
    source_state_after_sha256: ContentDigest | None = None
    cause_event_ids: tuple[ContentDigest, ...] = ()
    lifecycle_epoch: int | None = None
    legacy_compatibility: CanonicalPayloadV1 | None = None
    schema_version: SchemaVersion = SCIENCE_OBSERVATION_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "instrument_manifest_sha256",
            "semantic_execution_id",
            "crca_semantic_run_digest",
            "science_semantic_acquisition_sha256",
            "world_uid",
            "pre_sequence_key_sha256",
            "deterministic_event_id",
            "control_binding_sha256",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        for field_name in (
            "producer_component_id",
            "cell_id",
            "semantic_source_address",
        ):
            require_text(getattr(self, field_name), field=field_name)
        require_instance(
            self.producer_contract_version,
            SchemaVersion,
            field="producer_contract_version",
        )
        require_int(self.logical_world_id, field="logical_world_id")
        require_int(self.absolute_tick, field="absolute_tick")
        require_int(self.world_local_sequence, field="world_local_sequence")
        require_instance(self.record_kind, ScienceRecordKindV1, field="record_kind")
        require_int(
            self.record_kind_version,
            field="record_kind_version",
            minimum=1,
        )
        if self.record_kind_version != 1:
            raise contract_invalid(
                "unsupported science record-kind version", field="record_kind_version"
            )
        require_instance(self.payload, CanonicalPayloadV1, field="payload")
        for field_name in ("source_state_before_sha256", "source_state_after_sha256"):
            value = getattr(self, field_name)
            if value is not None:
                require_instance(value, ContentDigest, field=field_name)
        require_tuple_instances(
            self.cause_event_ids,
            ContentDigest,
            field="cause_event_ids",
            unique=True,
            sorted_by_string=True,
        )
        if self.lifecycle_epoch is not None:
            require_int(self.lifecycle_epoch, field="lifecycle_epoch")
        if self.legacy_compatibility is not None:
            require_instance(
                self.legacy_compatibility,
                CanonicalPayloadV1,
                field="legacy_compatibility",
            )
        _validate_schema_version(
            self.schema_version, expected=SCIENCE_OBSERVATION_RECORD_SCHEMA_VERSION
        )
        payload_value = self.payload.to_value()
        _reject_generation_authority(payload_value, path="payload")
        payload_mapping = _mapping(payload_value, field="payload")
        required_payload_fields = _record_required_payload_fields(
            self.record_kind, self.payload
        )
        if set(payload_mapping) != set(required_payload_fields):
            raise contract_invalid(
                "record payload does not match its frozen record-kind contract",
                field="payload",
            )
        if self.record_kind is ScienceRecordKindV1.WORLD_COMPLETION:
            if payload_mapping["completion_class"] not in {
                "EMPTY_INITIAL",
                "EXTINCT",
                "NO_BIRTHS",
                "TERMINAL_HORIZON",
            }:
                raise contract_invalid(
                    "unsupported world completion class",
                    field="payload.completion_class",
                )

    def semantic_dict(self) -> dict[str, object]:
        """Identity-bearing record form; legacy display metadata is excluded."""

        return {
            "schema_version": str(self.schema_version),
            "instrument_manifest_sha256": self.instrument_manifest_sha256.to_dict(),
            "producer_component_id": self.producer_component_id,
            "producer_contract_version": str(self.producer_contract_version),
            "semantic_execution_id": self.semantic_execution_id.to_dict(),
            "crca_semantic_run_digest": self.crca_semantic_run_digest.to_dict(),
            "science_semantic_acquisition_sha256": self.science_semantic_acquisition_sha256.to_dict(),
            "cell_id": self.cell_id,
            "logical_world_id": self.logical_world_id,
            "world_uid": self.world_uid.to_dict(),
            "absolute_tick": self.absolute_tick,
            "semantic_source_address": self.semantic_source_address,
            "pre_sequence_key_sha256": self.pre_sequence_key_sha256.to_dict(),
            "world_local_sequence": self.world_local_sequence,
            "deterministic_event_id": self.deterministic_event_id.to_dict(),
            "record_kind": self.record_kind.value,
            "record_kind_version": self.record_kind_version,
            "control_binding_sha256": self.control_binding_sha256.to_dict(),
            "payload": self.payload.to_dict(),
            "source_state_before_sha256": (
                None
                if self.source_state_before_sha256 is None
                else self.source_state_before_sha256.to_dict()
            ),
            "source_state_after_sha256": (
                None
                if self.source_state_after_sha256 is None
                else self.source_state_after_sha256.to_dict()
            ),
            "cause_event_ids": [item.to_dict() for item in self.cause_event_ids],
            "lifecycle_epoch": self.lifecycle_epoch,
        }

    def to_dict(self) -> dict[str, object]:
        value = self.semantic_dict()
        value["legacy_compatibility"] = (
            None
            if self.legacy_compatibility is None
            else self.legacy_compatibility.to_dict()
        )
        return value

    def semantic_record_sha256(self) -> ContentDigest:
        return canonical_sha256(self.semantic_dict())

    @classmethod
    def from_dict(cls, value: object) -> "ScienceObservationRecordV1":
        required = {
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
        data = _mapping(value, field="science_observation_record", exact_keys=required)
        causes = data["cause_event_ids"]
        if not isinstance(causes, list):
            raise contract_invalid(
                "cause_event_ids must be a list", field="cause_event_ids"
            )
        return cls(
            instrument_manifest_sha256=_digest_from_dict(
                data["instrument_manifest_sha256"], field="instrument_manifest_sha256"
            ),
            producer_component_id=data["producer_component_id"],
            producer_contract_version=SchemaVersion(data["producer_contract_version"]),
            semantic_execution_id=_digest_from_dict(
                data["semantic_execution_id"], field="semantic_execution_id"
            ),
            crca_semantic_run_digest=_digest_from_dict(
                data["crca_semantic_run_digest"], field="crca_semantic_run_digest"
            ),
            science_semantic_acquisition_sha256=_digest_from_dict(
                data["science_semantic_acquisition_sha256"],
                field="science_semantic_acquisition_sha256",
            ),
            cell_id=data["cell_id"],
            logical_world_id=data["logical_world_id"],
            world_uid=_digest_from_dict(data["world_uid"], field="world_uid"),
            absolute_tick=data["absolute_tick"],
            semantic_source_address=data["semantic_source_address"],
            pre_sequence_key_sha256=_digest_from_dict(
                data["pre_sequence_key_sha256"], field="pre_sequence_key_sha256"
            ),
            world_local_sequence=data["world_local_sequence"],
            deterministic_event_id=_digest_from_dict(
                data["deterministic_event_id"], field="deterministic_event_id"
            ),
            record_kind=ScienceRecordKindV1(data["record_kind"]),
            record_kind_version=data["record_kind_version"],
            control_binding_sha256=_digest_from_dict(
                data["control_binding_sha256"], field="control_binding_sha256"
            ),
            payload=_payload_from_dict(data["payload"], field="payload"),
            source_state_before_sha256=_optional_digest_from_dict(
                data["source_state_before_sha256"], field="source_state_before_sha256"
            ),
            source_state_after_sha256=_optional_digest_from_dict(
                data["source_state_after_sha256"], field="source_state_after_sha256"
            ),
            cause_event_ids=tuple(
                _digest_from_dict(item, field="cause_event_ids[]") for item in causes
            ),
            lifecycle_epoch=data["lifecycle_epoch"],
            legacy_compatibility=_optional_payload_from_dict(
                data["legacy_compatibility"], field="legacy_compatibility"
            ),
            schema_version=SchemaVersion(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class ScienceEvidencePartitionV1:
    instrument_manifest_sha256: ContentDigest
    science_semantic_acquisition_sha256: ContentDigest
    observation_abi_version: SchemaVersion
    profile: ScienceObservationProfileV1
    world_uid: ContentDigest
    record_kind: ScienceRecordKindV1
    absolute_tick_start: int
    absolute_tick_end: int
    rows: tuple[ScienceObservationRecordV1, ...]
    schema_version: SchemaVersion = SCIENCE_EVIDENCE_PARTITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "instrument_manifest_sha256",
            "science_semantic_acquisition_sha256",
            "world_uid",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        require_instance(
            self.observation_abi_version,
            SchemaVersion,
            field="observation_abi_version",
        )
        require_instance(self.profile, ScienceObservationProfileV1, field="profile")
        require_instance(self.record_kind, ScienceRecordKindV1, field="record_kind")
        require_int(self.absolute_tick_start, field="absolute_tick_start")
        require_int(self.absolute_tick_end, field="absolute_tick_end")
        if self.absolute_tick_end < self.absolute_tick_start:
            raise contract_invalid(
                "partition tick span is reversed", field="absolute_tick_end"
            )
        require_tuple_instances(
            self.rows, ScienceObservationRecordV1, field="rows", unique=True
        )
        keys = [(row.absolute_tick, row.world_local_sequence) for row in self.rows]
        if keys != sorted(set(keys)):
            raise contract_invalid(
                "partition rows must be sorted with unique sequence keys", field="rows"
            )
        for row in self.rows:
            if row.instrument_manifest_sha256 != self.instrument_manifest_sha256:
                raise contract_invalid("row instrument mismatch", field="rows")
            if (
                row.science_semantic_acquisition_sha256
                != self.science_semantic_acquisition_sha256
            ):
                raise contract_invalid("row acquisition mismatch", field="rows")
            if (
                row.world_uid != self.world_uid
                or row.record_kind is not self.record_kind
            ):
                raise contract_invalid("row partition key mismatch", field="rows")
            if (
                not self.absolute_tick_start
                <= row.absolute_tick
                < self.absolute_tick_end
            ):
                raise contract_invalid(
                    "row tick lies outside half-open span", field="rows"
                )
        _validate_schema_version(
            self.schema_version, expected=SCIENCE_EVIDENCE_PARTITION_SCHEMA_VERSION
        )

    @property
    def first_sequence(self) -> int | None:
        return None if not self.rows else self.rows[0].world_local_sequence

    @property
    def last_sequence(self) -> int | None:
        return None if not self.rows else self.rows[-1].world_local_sequence

    def rows_sha256(self) -> ContentDigest:
        return canonical_sha256([row.semantic_dict() for row in self.rows])

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "instrument_manifest_sha256": self.instrument_manifest_sha256.to_dict(),
            "science_semantic_acquisition_sha256": self.science_semantic_acquisition_sha256.to_dict(),
            "observation_abi_version": str(self.observation_abi_version),
            "profile_id": self.profile.value,
            "world_uid": self.world_uid.to_dict(),
            "record_kind": self.record_kind.value,
            "absolute_tick_start": self.absolute_tick_start,
            "absolute_tick_end": self.absolute_tick_end,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "row_count": len(self.rows),
            "rows": [row.to_dict() for row in self.rows],
            "rows_sha256": self.rows_sha256().to_dict(),
        }

    def partition_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "ScienceEvidencePartitionV1":
        expected = {
            "schema_version",
            "instrument_manifest_sha256",
            "science_semantic_acquisition_sha256",
            "observation_abi_version",
            "profile_id",
            "world_uid",
            "record_kind",
            "absolute_tick_start",
            "absolute_tick_end",
            "first_sequence",
            "last_sequence",
            "row_count",
            "rows",
            "rows_sha256",
        }
        data = _mapping(value, field="science_evidence_partition", exact_keys=expected)
        rows = data["rows"]
        if not isinstance(rows, list):
            raise contract_invalid("rows must be a list", field="rows")
        result = cls(
            instrument_manifest_sha256=_digest_from_dict(
                data["instrument_manifest_sha256"], field="instrument_manifest_sha256"
            ),
            science_semantic_acquisition_sha256=_digest_from_dict(
                data["science_semantic_acquisition_sha256"],
                field="science_semantic_acquisition_sha256",
            ),
            observation_abi_version=SchemaVersion(data["observation_abi_version"]),
            profile=ScienceObservationProfileV1(data["profile_id"]),
            world_uid=_digest_from_dict(data["world_uid"], field="world_uid"),
            record_kind=ScienceRecordKindV1(data["record_kind"]),
            absolute_tick_start=data["absolute_tick_start"],
            absolute_tick_end=data["absolute_tick_end"],
            rows=tuple(ScienceObservationRecordV1.from_dict(row) for row in rows),
            schema_version=SchemaVersion(data["schema_version"]),
        )
        if data["row_count"] != len(result.rows):
            raise contract_invalid("row_count mismatch", field="row_count")
        if data["first_sequence"] != result.first_sequence:
            raise contract_invalid("first_sequence mismatch", field="first_sequence")
        if data["last_sequence"] != result.last_sequence:
            raise contract_invalid("last_sequence mismatch", field="last_sequence")
        if (
            _digest_from_dict(data["rows_sha256"], field="rows_sha256")
            != result.rows_sha256()
        ):
            raise contract_invalid("rows_sha256 mismatch", field="rows_sha256")
        return result


@dataclass(frozen=True, slots=True)
class ProfileResourceEnvelopeV1:
    profile: ScienceObservationProfileV1
    max_records_per_tick_per_world_by_kind: tuple[tuple[ScienceRecordKindV1, int], ...]
    max_record_bytes_by_kind: tuple[tuple[ScienceRecordKindV1, int], ...]
    chunk_max_records: int
    chunk_max_bytes: int
    max_in_flight_chunks: int
    fixed_buffer_bytes: int
    per_world_cursor_bytes: int
    per_resident_agent_observation_state_bytes: int
    logical_world_count: int
    resident_capacity: int
    max_resident_observation_bytes: int
    schema_version: SchemaVersion = SCIENCE_PROFILE_RESOURCE_ENVELOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.profile, ScienceObservationProfileV1, field="profile")
        for field_name in (
            "max_records_per_tick_per_world_by_kind",
            "max_record_bytes_by_kind",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise contract_invalid(
                    "capacity table must be a tuple", field=field_name
                )
            rendered: list[str] = []
            for item in values:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise contract_invalid(
                        "capacity row is malformed", field=field_name
                    )
                kind, count = item
                require_instance(kind, ScienceRecordKindV1, field=field_name)
                require_int(count, field=field_name)
                rendered.append(kind.value)
            if rendered != sorted(set(rendered)):
                raise contract_invalid(
                    "capacity rows must be sorted and unique", field=field_name
                )
        for field_name in (
            "chunk_max_records",
            "chunk_max_bytes",
            "max_in_flight_chunks",
            "fixed_buffer_bytes",
            "per_world_cursor_bytes",
            "per_resident_agent_observation_state_bytes",
            "logical_world_count",
            "resident_capacity",
            "max_resident_observation_bytes",
        ):
            require_int(getattr(self, field_name), field=field_name, minimum=0)
        if self.chunk_max_records < 1 or self.chunk_max_bytes < 1:
            raise contract_invalid(
                "chunk capacities must be positive", field="chunk_max_records"
            )
        _validate_schema_version(
            self.schema_version,
            expected=SCIENCE_PROFILE_RESOURCE_ENVELOPE_SCHEMA_VERSION,
        )
        if self.max_resident_observation_bytes != self.computed_resident_bytes():
            raise contract_invalid(
                "resident observation byte equation mismatch",
                field="max_resident_observation_bytes",
            )

    def computed_resident_bytes(self) -> int:
        return (
            self.fixed_buffer_bytes
            + (self.max_in_flight_chunks * self.chunk_max_bytes)
            + (self.logical_world_count * self.per_world_cursor_bytes)
            + (
                self.logical_world_count
                * self.resident_capacity
                * self.per_resident_agent_observation_state_bytes
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "profile_id": self.profile.value,
            "max_records_per_tick_per_world_by_kind": {
                kind.value: count
                for kind, count in self.max_records_per_tick_per_world_by_kind
            },
            "max_record_bytes_by_kind": {
                kind.value: count for kind, count in self.max_record_bytes_by_kind
            },
            "chunk_max_records": self.chunk_max_records,
            "chunk_max_bytes": self.chunk_max_bytes,
            "max_in_flight_chunks": self.max_in_flight_chunks,
            "fixed_buffer_bytes": self.fixed_buffer_bytes,
            "per_world_cursor_bytes": self.per_world_cursor_bytes,
            "per_resident_agent_observation_state_bytes": self.per_resident_agent_observation_state_bytes,
            "logical_world_count": self.logical_world_count,
            "resident_capacity": self.resident_capacity,
            "max_resident_observation_bytes": self.max_resident_observation_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ProfileResourceEnvelopeV1":
        data = _mapping(
            value,
            field="science_profile_resource_envelope",
            exact_keys={
                "schema_version",
                "profile_id",
                "max_records_per_tick_per_world_by_kind",
                "max_record_bytes_by_kind",
                "chunk_max_records",
                "chunk_max_bytes",
                "max_in_flight_chunks",
                "fixed_buffer_bytes",
                "per_world_cursor_bytes",
                "per_resident_agent_observation_state_bytes",
                "logical_world_count",
                "resident_capacity",
                "max_resident_observation_bytes",
            },
        )

        def capacity_table(
            field_name: str,
        ) -> tuple[tuple[ScienceRecordKindV1, int], ...]:
            table = _mapping(data[field_name], field=field_name)
            return tuple(
                (ScienceRecordKindV1(kind), table[kind]) for kind in sorted(table)
            )

        return cls(
            profile=ScienceObservationProfileV1(data["profile_id"]),
            max_records_per_tick_per_world_by_kind=capacity_table(
                "max_records_per_tick_per_world_by_kind"
            ),
            max_record_bytes_by_kind=capacity_table("max_record_bytes_by_kind"),
            chunk_max_records=data["chunk_max_records"],
            chunk_max_bytes=data["chunk_max_bytes"],
            max_in_flight_chunks=data["max_in_flight_chunks"],
            fixed_buffer_bytes=data["fixed_buffer_bytes"],
            per_world_cursor_bytes=data["per_world_cursor_bytes"],
            per_resident_agent_observation_state_bytes=data[
                "per_resident_agent_observation_state_bytes"
            ],
            logical_world_count=data["logical_world_count"],
            resident_capacity=data["resident_capacity"],
            max_resident_observation_bytes=data["max_resident_observation_bytes"],
            schema_version=SchemaVersion(data["schema_version"]),
        )


__all__ = [
    "EEDReleaseStatusV1",
    "LEGACY_PRODUCT_ALIAS",
    "PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION",
    "PRODUCT_ID",
    "ProfileResourceEnvelopeV1",
    "SCIENCE_ADMISSION_OUTCOME_PRECEDENCE",
    "SCIENCE_ADMISSION_RECEIPT_SCHEMA_VERSION",
    "SCIENCE_CAMPAIGN_SCHEMA_VERSION",
    "SCIENCE_CONTROL_DESCRIPTOR_SCHEMA_VERSION",
    "SCIENCE_EVIDENCE_PARTITION_SCHEMA_VERSION",
    "SCIENCE_EXECUTION_PLAN_SCHEMA_VERSION",
    "SCIENCE_FINDING_SCHEMA_VERSION",
    "SCIENCE_INSTRUMENT_MANIFEST_SCHEMA_VERSION",
    "SCIENCE_OBSERVATION_RECORD_SCHEMA_VERSION",
    "SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_VERSION",
    "SCIENCE_NATIVE_AGENT_STATE_SNAPSHOT_PAYLOAD_SCHEMA_VERSION",
    "SCIENCE_NATIVE_CONTROL_SCHEMA_VERSION",
    "SCIENCE_NATIVE_FULL_CAPABILITY_MASK",
    "SCIENCE_NATIVE_MUTATION_CONTROL_ID",
    "SCIENCE_NATIVE_MUTATION_OFF_CAPABILITY_MASK",
    "SCIENCE_PROFILE_RESOURCE_ENVELOPE_SCHEMA_VERSION",
    "SCIENCE_REDUCER_DESCRIPTOR_SCHEMA_VERSION",
    "SCIENCE_REDUCTION_PLAN_SCHEMA_VERSION",
    "SEMANTIC_ACQUISITION_CORE_SCHEMA_VERSION",
    "ScienceAdmissionOutcomeV1",
    "ScienceAdmissionReceiptV1",
    "ScienceCampaignV1",
    "ScienceControlDescriptorV1",
    "ScienceEvidencePartitionV1",
    "ScienceExecutionPlanV1",
    "ScienceFindingV1",
    "ScienceInstrumentManifestV1",
    "ScienceObservationProfileV1",
    "ScienceObservationRecordV1",
    "ScienceRecordKindV1",
    "ScienceReducerDescriptorV1",
    "ScienceReductionPlanV1",
    "V2_RESERVED_RECORD_KINDS",
    "validate_science_native_control_receipt_v1",
]
