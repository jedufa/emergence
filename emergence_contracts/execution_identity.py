"""CRCA immutable layered execution identity and Run/Attempt occurrence laws.

These DTOs are a pure contract kernel.  They do not admit resources, grant
leases, write receipts, launch execution, or make qualification decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from ._validation import (
    require_instance,
    require_int,
    require_text,
    require_tuple_instances,
)
from .artifacts import NamedDigestV1
from .canonical import canonical_sha256
from .errors import contract_invalid
from .ids import (
    AttemptId,
    BackendId,
    ContentDigest,
    ExperimentId,
    LeaseId,
    RunId,
    SchemaVersion,
)


CRCA_IDENTITY_FREEZE_ID = "CRCA-IDENTITY-FREEZE-1.0"
RUN_ATTEMPT_EDGE_CASE_MATRIX_CASES = 24
RUN_ATTEMPT_EDGE_CASE_MATRIX_SHA256 = (
    "6d2856a982117064536fd7b0750df4a78040e183f725d38fa7cb9c9a3d0d41bd"
)

SEMANTIC_RUN_SPEC_SCHEMA_VERSION = SchemaVersion("crca_semantic_run_spec_v1")
EXECUTION_IMPLEMENTATION_SCHEMA_VERSION = SchemaVersion(
    "crca_execution_implementation_v1"
)
ADMISSION_ASSURANCE_BINDING_SCHEMA_VERSION = SchemaVersion(
    "crca_admission_assurance_binding_v1"
)
LIFECYCLE_BINDING_SCHEMA_VERSION = SchemaVersion("crca_lifecycle_binding_v1")
ATTEMPT_ADMISSION_REQUEST_SCHEMA_VERSION = SchemaVersion(
    "crca_attempt_admission_request_v1"
)
CONCRETIZED_EXECUTION_SCHEMA_VERSION = SchemaVersion("crca_concretized_execution_v1")
RUN_OCCURRENCE_SCHEMA_VERSION = SchemaVersion("crca_run_occurrence_v1")
ATTEMPT_ADMISSION_DISPOSITION_SCHEMA_VERSION = SchemaVersion(
    "crca_attempt_admission_disposition_v1"
)
ATTEMPT_FRONTIERS_SCHEMA_VERSION = SchemaVersion("crca_attempt_frontiers_v1")
CANONICAL_CONTINUATION_LEASE_SCHEMA_VERSION = SchemaVersion(
    "crca_canonical_continuation_lease_v1"
)
ATTEMPT_START_SCHEMA_VERSION = SchemaVersion("crca_attempt_start_v1")
ATTEMPT_TERMINAL_SCHEMA_VERSION = SchemaVersion("crca_attempt_terminal_v1")
ATTEMPT_RETRY_BASIS_SCHEMA_VERSION = SchemaVersion("crca_attempt_retry_basis_v1")
EXACT_IMPLEMENTATION_CONFORMANCE_SCHEMA_VERSION = SchemaVersion(
    "crca_exact_implementation_conformance_v1"
)
RESTORE_CHECKPOINT_AUTHORIZATION_SCHEMA_VERSION = SchemaVersion(
    "crca_restore_checkpoint_authorization_v1"
)
CONTINUATION_LEASE_EVENT_SCHEMA_VERSION = SchemaVersion(
    "crca_continuation_lease_event_v1"
)
CONTINUATION_LEASE_EVENT_SCHEMA_VERSION_V2 = SchemaVersion(
    "crca_continuation_lease_event_v2"
)


def _digest(value: ContentDigest) -> dict[str, str]:
    return value.to_dict()


def _optional_digest(value: ContentDigest | None) -> dict[str, str] | None:
    return None if value is None else value.to_dict()


def _validate_schema(
    actual: SchemaVersion, expected: SchemaVersion, *, field: str
) -> None:
    require_instance(actual, SchemaVersion, field=field)
    if actual != expected:
        raise contract_invalid("unsupported CRCA identity schema", field=field)


def _require_canonical_utc(value: object, *, field: str) -> datetime:
    text = require_text(value, field=field)
    if len(text) != 20 or not text.endswith("Z"):
        raise contract_invalid(
            "timestamp must use canonical second-resolution UTC",
            field=field,
        )
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise contract_invalid("timestamp is invalid", field=field) from error
    return parsed


def _validate_digest_tuple(
    values: tuple[NamedDigestV1, ...], *, field: str, required: bool = False
) -> None:
    require_tuple_instances(values, NamedDigestV1, field=field)
    if required and not values:
        raise contract_invalid("at least one digest is required", field=field)
    names = [item.name for item in values]
    if names != sorted(names) or len(names) != len(set(names)):
        raise contract_invalid(
            "named digests must be sorted by unique name", field=field
        )


@dataclass(frozen=True, slots=True)
class SemanticRunSpecV1:
    experiment_id: ExperimentId
    system_assembly_digest: ContentDigest
    semantic_parameters_digest: ContentDigest
    rng_contract_digest: ContentDigest
    seed_digest: ContentDigest
    absolute_start_next_tick: int
    absolute_target_next_tick: int
    trajectory_checkpoint_ticks: tuple[int, ...]
    measurement_behavior_digest: ContentDigest
    control_and_branch_semantics_digest: ContentDigest
    schema_version: SchemaVersion = SEMANTIC_RUN_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.experiment_id, ExperimentId, field="experiment_id")
        for field_name in (
            "system_assembly_digest",
            "semantic_parameters_digest",
            "rng_contract_digest",
            "seed_digest",
            "measurement_behavior_digest",
            "control_and_branch_semantics_digest",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        start = require_int(
            self.absolute_start_next_tick, field="absolute_start_next_tick"
        )
        target = require_int(
            self.absolute_target_next_tick, field="absolute_target_next_tick"
        )
        if target <= start:
            raise contract_invalid(
                "absolute target must be after the start",
                field="absolute_target_next_tick",
            )
        if not isinstance(self.trajectory_checkpoint_ticks, tuple):
            raise contract_invalid(
                "trajectory checkpoints must be a tuple",
                field="trajectory_checkpoint_ticks",
            )
        checkpoints = list(self.trajectory_checkpoint_ticks)
        for value in checkpoints:
            require_int(value, field="trajectory_checkpoint_ticks")
        if checkpoints != sorted(set(checkpoints)):
            raise contract_invalid(
                "trajectory checkpoints must be sorted and unique",
                field="trajectory_checkpoint_ticks",
            )
        if any(value < start or value > target for value in checkpoints):
            raise contract_invalid(
                "trajectory checkpoint lies outside the absolute Run span",
                field="trajectory_checkpoint_ticks",
            )
        _validate_schema(
            self.schema_version,
            SEMANTIC_RUN_SPEC_SCHEMA_VERSION,
            field="schema_version",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "experiment_id": str(self.experiment_id),
            "system_assembly_digest": _digest(self.system_assembly_digest),
            "semantic_parameters_digest": _digest(self.semantic_parameters_digest),
            "rng_contract_digest": _digest(self.rng_contract_digest),
            "seed_digest": _digest(self.seed_digest),
            "absolute_start_next_tick": self.absolute_start_next_tick,
            "absolute_target_next_tick": self.absolute_target_next_tick,
            "trajectory_checkpoint_ticks": list(self.trajectory_checkpoint_ticks),
            "measurement_behavior_digest": _digest(self.measurement_behavior_digest),
            "control_and_branch_semantics_digest": _digest(
                self.control_and_branch_semantics_digest
            ),
        }

    def semantic_spec_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ExecutionImplementationV1:
    source_commit: str
    source_tree_digest: ContentDigest
    dirty_patch_digest: ContentDigest | None
    toolchain_digest: ContentDigest
    environment_digest: ContentDigest
    provider_id: str
    backend_id: BackendId
    runtime_profile_id: str
    native_artifact_digests: tuple[NamedDigestV1, ...]
    abi_digest: ContentDigest
    device_conformance_envelope_digest: ContentDigest
    schema_version: SchemaVersion = EXECUTION_IMPLEMENTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.source_commit, field="source_commit")
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ):
            raise contract_invalid(
                "source commit must be a full lowercase Git object ID",
                field="source_commit",
            )
        for field_name in (
            "source_tree_digest",
            "toolchain_digest",
            "environment_digest",
            "abi_digest",
            "device_conformance_envelope_digest",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        if self.dirty_patch_digest is not None:
            require_instance(
                self.dirty_patch_digest, ContentDigest, field="dirty_patch_digest"
            )
        require_text(self.provider_id, field="provider_id")
        require_instance(self.backend_id, BackendId, field="backend_id")
        require_text(self.runtime_profile_id, field="runtime_profile_id")
        _validate_digest_tuple(
            self.native_artifact_digests,
            field="native_artifact_digests",
        )
        _validate_schema(
            self.schema_version,
            EXECUTION_IMPLEMENTATION_SCHEMA_VERSION,
            field="schema_version",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "source_commit": self.source_commit,
            "source_tree_digest": _digest(self.source_tree_digest),
            "dirty_patch_digest": _optional_digest(self.dirty_patch_digest),
            "toolchain_digest": _digest(self.toolchain_digest),
            "environment_digest": _digest(self.environment_digest),
            "provider_id": self.provider_id,
            "backend_id": str(self.backend_id),
            "runtime_profile_id": self.runtime_profile_id,
            "native_artifact_digests": [
                item.to_dict() for item in self.native_artifact_digests
            ],
            "abi_digest": _digest(self.abi_digest),
            "device_conformance_envelope_digest": _digest(
                self.device_conformance_envelope_digest
            ),
        }

    def implementation_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class AdmissionAssuranceBindingV1:
    intended_claim_id: str
    qualified_envelope_digest: ContentDigest | None
    obligation_digests: tuple[NamedDigestV1, ...]
    authority_reference_digests: tuple[NamedDigestV1, ...]
    review_reference_digests: tuple[NamedDigestV1, ...]
    schema_version: SchemaVersion = ADMISSION_ASSURANCE_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.intended_claim_id, field="intended_claim_id")
        if self.qualified_envelope_digest is not None:
            require_instance(
                self.qualified_envelope_digest,
                ContentDigest,
                field="qualified_envelope_digest",
            )
        for field_name in (
            "obligation_digests",
            "authority_reference_digests",
            "review_reference_digests",
        ):
            _validate_digest_tuple(getattr(self, field_name), field=field_name)
        _validate_schema(
            self.schema_version,
            ADMISSION_ASSURANCE_BINDING_SCHEMA_VERSION,
            field="schema_version",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "intended_claim_id": self.intended_claim_id,
            "qualified_envelope_digest": _optional_digest(
                self.qualified_envelope_digest
            ),
            "obligation_digests": [item.to_dict() for item in self.obligation_digests],
            "authority_reference_digests": [
                item.to_dict() for item in self.authority_reference_digests
            ],
            "review_reference_digests": [
                item.to_dict() for item in self.review_reference_digests
            ],
        }

    def admission_assurance_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class LifecycleBindingV1:
    output_root: str
    artifact_lifecycle_contract_digest: ContentDigest
    inventory_role_digests: tuple[NamedDigestV1, ...]
    initial_hold_digests: tuple[NamedDigestV1, ...]
    retention_profile_id: str
    schema_version: SchemaVersion = LIFECYCLE_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.output_root, field="output_root")
        require_instance(
            self.artifact_lifecycle_contract_digest,
            ContentDigest,
            field="artifact_lifecycle_contract_digest",
        )
        _validate_digest_tuple(
            self.inventory_role_digests, field="inventory_role_digests"
        )
        _validate_digest_tuple(self.initial_hold_digests, field="initial_hold_digests")
        require_text(self.retention_profile_id, field="retention_profile_id")
        _validate_schema(
            self.schema_version,
            LIFECYCLE_BINDING_SCHEMA_VERSION,
            field="schema_version",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "output_root": self.output_root,
            "artifact_lifecycle_contract_digest": _digest(
                self.artifact_lifecycle_contract_digest
            ),
            "inventory_role_digests": [
                item.to_dict() for item in self.inventory_role_digests
            ],
            "initial_hold_digests": [
                item.to_dict() for item in self.initial_hold_digests
            ],
            "retention_profile_id": self.retention_profile_id,
        }

    def lifecycle_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class AttemptAdmissionRequestV1:
    resource_requirements_digest: ContentDigest
    executor_constraints_digest: ContentDigest
    requested_restore_checkpoint_digest: ContentDigest | None
    observational_persistence_cadence_digest: ContentDigest | None
    proposed_reason: str
    schema_version: SchemaVersion = ATTEMPT_ADMISSION_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(
            self.resource_requirements_digest,
            ContentDigest,
            field="resource_requirements_digest",
        )
        require_instance(
            self.executor_constraints_digest,
            ContentDigest,
            field="executor_constraints_digest",
        )
        for field_name in (
            "requested_restore_checkpoint_digest",
            "observational_persistence_cadence_digest",
        ):
            value = getattr(self, field_name)
            if value is not None:
                require_instance(value, ContentDigest, field=field_name)
        require_text(self.proposed_reason, field="proposed_reason")
        _validate_schema(
            self.schema_version,
            ATTEMPT_ADMISSION_REQUEST_SCHEMA_VERSION,
            field="schema_version",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "resource_requirements_digest": _digest(self.resource_requirements_digest),
            "executor_constraints_digest": _digest(self.executor_constraints_digest),
            "requested_restore_checkpoint_digest": _optional_digest(
                self.requested_restore_checkpoint_digest
            ),
            "observational_persistence_cadence_digest": _optional_digest(
                self.observational_persistence_cadence_digest
            ),
            "proposed_reason": self.proposed_reason,
        }

    def attempt_admission_request_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ConcretizedExecutionV1:
    semantic_spec_digest: ContentDigest
    implementation_digest: ContentDigest
    admission_assurance_digest: ContentDigest
    lifecycle_digest: ContentDigest
    attempt_admission_request_digest: ContentDigest
    campaign_plan_digest: ContentDigest
    bound_execution_request_digest: ContentDigest
    run_contract_digest: ContentDigest
    baseline_digest: ContentDigest
    route_contract_digest: ContentDigest
    assembly_digest: ContentDigest
    qualification_key_digest: ContentDigest
    schema_version: SchemaVersion = CONCRETIZED_EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "semantic_spec_digest",
            "implementation_digest",
            "admission_assurance_digest",
            "lifecycle_digest",
            "attempt_admission_request_digest",
            "campaign_plan_digest",
            "bound_execution_request_digest",
            "run_contract_digest",
            "baseline_digest",
            "route_contract_digest",
            "assembly_digest",
            "qualification_key_digest",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        _validate_schema(
            self.schema_version,
            CONCRETIZED_EXECUTION_SCHEMA_VERSION,
            field="schema_version",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            **{
                field_name: _digest(getattr(self, field_name))
                for field_name in (
                    "semantic_spec_digest",
                    "implementation_digest",
                    "admission_assurance_digest",
                    "lifecycle_digest",
                    "attempt_admission_request_digest",
                    "campaign_plan_digest",
                    "bound_execution_request_digest",
                    "run_contract_digest",
                    "baseline_digest",
                    "route_contract_digest",
                    "assembly_digest",
                    "qualification_key_digest",
                )
            },
        }

    def concretized_execution_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


def derive_run_id(
    *,
    experiment_id: ExperimentId,
    cell_or_repeat_ordinal: int,
    semantic_spec_digest: ContentDigest,
    parent_checkpoint_digest: ContentDigest | None,
    branch_spec_digest: ContentDigest | None,
) -> RunId:
    require_instance(experiment_id, ExperimentId, field="experiment_id")
    require_int(cell_or_repeat_ordinal, field="cell_or_repeat_ordinal", minimum=0)
    require_instance(semantic_spec_digest, ContentDigest, field="semantic_spec_digest")
    for field_name, value in (
        ("parent_checkpoint_digest", parent_checkpoint_digest),
        ("branch_spec_digest", branch_spec_digest),
    ):
        if value is not None:
            require_instance(value, ContentDigest, field=field_name)
    digest = canonical_sha256(
        {
            "schema_version": str(RUN_OCCURRENCE_SCHEMA_VERSION),
            "experiment_id": str(experiment_id),
            "cell_or_repeat_ordinal": cell_or_repeat_ordinal,
            "semantic_spec_digest": _digest(semantic_spec_digest),
            "parent_checkpoint_digest": _optional_digest(parent_checkpoint_digest),
            "branch_spec_digest": _optional_digest(branch_spec_digest),
        }
    )
    return RunId(f"crca-run-v1-{digest.hex}")


@dataclass(frozen=True, slots=True)
class RunOccurrenceV1:
    run_id: RunId
    experiment_id: ExperimentId
    cell_or_repeat_ordinal: int
    semantic_spec_digest: ContentDigest
    parent_checkpoint_digest: ContentDigest | None
    branch_spec_digest: ContentDigest | None
    schema_version: SchemaVersion = RUN_OCCURRENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        expected = derive_run_id(
            experiment_id=self.experiment_id,
            cell_or_repeat_ordinal=self.cell_or_repeat_ordinal,
            semantic_spec_digest=self.semantic_spec_digest,
            parent_checkpoint_digest=self.parent_checkpoint_digest,
            branch_spec_digest=self.branch_spec_digest,
        )
        require_instance(self.run_id, RunId, field="run_id")
        if self.run_id != expected:
            raise contract_invalid(
                "RunId differs from occurrence identity", field="run_id"
            )
        _validate_schema(
            self.schema_version, RUN_OCCURRENCE_SCHEMA_VERSION, field="schema_version"
        )

    @classmethod
    def create(
        cls,
        *,
        experiment_id: ExperimentId,
        cell_or_repeat_ordinal: int,
        semantic_spec_digest: ContentDigest,
        parent_checkpoint_digest: ContentDigest | None = None,
        branch_spec_digest: ContentDigest | None = None,
    ) -> "RunOccurrenceV1":
        return cls(
            run_id=derive_run_id(
                experiment_id=experiment_id,
                cell_or_repeat_ordinal=cell_or_repeat_ordinal,
                semantic_spec_digest=semantic_spec_digest,
                parent_checkpoint_digest=parent_checkpoint_digest,
                branch_spec_digest=branch_spec_digest,
            ),
            experiment_id=experiment_id,
            cell_or_repeat_ordinal=cell_or_repeat_ordinal,
            semantic_spec_digest=semantic_spec_digest,
            parent_checkpoint_digest=parent_checkpoint_digest,
            branch_spec_digest=branch_spec_digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "run_id": str(self.run_id),
            "experiment_id": str(self.experiment_id),
            "cell_or_repeat_ordinal": self.cell_or_repeat_ordinal,
            "semantic_spec_digest": _digest(self.semantic_spec_digest),
            "parent_checkpoint_digest": _optional_digest(self.parent_checkpoint_digest),
            "branch_spec_digest": _optional_digest(self.branch_spec_digest),
        }


class AttemptAdmissionDispositionCodeV1(str, Enum):
    WAIT_RESOURCE = "WAIT_RESOURCE"
    REJECTED = "REJECTED"
    LEASE_EXPIRED_PRELAUNCH = "LEASE_EXPIRED_PRELAUNCH"
    START_AUTHORIZED = "START_AUTHORIZED"


@dataclass(frozen=True, slots=True)
class AttemptAdmissionDispositionV1:
    run_id: RunId
    attempt_admission_request_digest: ContentDigest
    code: AttemptAdmissionDispositionCodeV1
    reason: str
    scheduler_assignment_lease_digest: ContentDigest | None
    resource_start_lease_digest: ContentDigest | None
    schema_version: SchemaVersion = ATTEMPT_ADMISSION_DISPOSITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.run_id, RunId, field="run_id")
        require_instance(
            self.attempt_admission_request_digest,
            ContentDigest,
            field="attempt_admission_request_digest",
        )
        require_instance(self.code, AttemptAdmissionDispositionCodeV1, field="code")
        require_text(self.reason, field="reason")
        for field_name in (
            "scheduler_assignment_lease_digest",
            "resource_start_lease_digest",
        ):
            value = getattr(self, field_name)
            if value is not None:
                require_instance(value, ContentDigest, field=field_name)
        requires_start_lease = self.code in {
            AttemptAdmissionDispositionCodeV1.LEASE_EXPIRED_PRELAUNCH,
            AttemptAdmissionDispositionCodeV1.START_AUTHORIZED,
        }
        if requires_start_lease != (self.resource_start_lease_digest is not None):
            raise contract_invalid(
                "admission disposition has an invalid resource-start lease binding",
                field="resource_start_lease_digest",
            )
        _validate_schema(
            self.schema_version,
            ATTEMPT_ADMISSION_DISPOSITION_SCHEMA_VERSION,
            field="schema_version",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "run_id": str(self.run_id),
            "attempt_admission_request_digest": _digest(
                self.attempt_admission_request_digest
            ),
            "code": self.code.value,
            "reason": self.reason,
            "scheduler_assignment_lease_digest": _optional_digest(
                self.scheduler_assignment_lease_digest
            ),
            "resource_start_lease_digest": _optional_digest(
                self.resource_start_lease_digest
            ),
        }

    def disposition_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class AttemptFrontiersV1:
    last_dispatched_next_tick: int
    last_computed_next_tick: int
    durable_checkpoint_next_tick: int
    published_artifact_next_tick: int
    schema_version: SchemaVersion = ATTEMPT_FRONTIERS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "last_dispatched_next_tick",
            "last_computed_next_tick",
            "durable_checkpoint_next_tick",
            "published_artifact_next_tick",
        ):
            require_int(getattr(self, field_name), field=field_name)
        if self.last_computed_next_tick > self.last_dispatched_next_tick:
            raise contract_invalid(
                "computed frontier cannot exceed dispatched frontier",
                field="last_computed_next_tick",
            )
        if self.durable_checkpoint_next_tick > self.last_computed_next_tick:
            raise contract_invalid(
                "durable checkpoint frontier cannot exceed computed frontier",
                field="durable_checkpoint_next_tick",
            )
        if self.published_artifact_next_tick > self.last_computed_next_tick:
            raise contract_invalid(
                "published artifact frontier cannot exceed computed frontier",
                field="published_artifact_next_tick",
            )
        _validate_schema(
            self.schema_version,
            ATTEMPT_FRONTIERS_SCHEMA_VERSION,
            field="schema_version",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "last_dispatched_next_tick": self.last_dispatched_next_tick,
            "last_computed_next_tick": self.last_computed_next_tick,
            "durable_checkpoint_next_tick": self.durable_checkpoint_next_tick,
            "published_artifact_next_tick": self.published_artifact_next_tick,
        }


def derive_attempt_id(
    *, run_id: RunId, monotonic_ordinal: int, nonce: str
) -> AttemptId:
    require_instance(run_id, RunId, field="run_id")
    require_int(monotonic_ordinal, field="monotonic_ordinal", minimum=1)
    require_text(nonce, field="nonce")
    digest = canonical_sha256(
        {
            "schema_version": str(ATTEMPT_START_SCHEMA_VERSION),
            "run_id": str(run_id),
            "monotonic_ordinal": monotonic_ordinal,
            "nonce": nonce,
        }
    )
    return AttemptId(f"crca-attempt-v1-{digest.hex}")


@dataclass(frozen=True, slots=True)
class CanonicalContinuationLeaseV1:
    lease_id: LeaseId
    run_id: RunId
    attempt_id: AttemptId
    restore_checkpoint_digest: ContentDigest | None
    lineage_scope_digest: ContentDigest
    owner_id: str
    lease_epoch: int
    fencing_token_digest: ContentDigest
    prior_lease_digest: ContentDigest | None
    granted_at_utc: str
    expires_at_utc: str
    schema_version: SchemaVersion = CANONICAL_CONTINUATION_LEASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.lease_id, LeaseId, field="lease_id")
        require_instance(self.run_id, RunId, field="run_id")
        require_instance(self.attempt_id, AttemptId, field="attempt_id")
        if self.restore_checkpoint_digest is not None:
            require_instance(
                self.restore_checkpoint_digest,
                ContentDigest,
                field="restore_checkpoint_digest",
            )
        require_instance(
            self.lineage_scope_digest,
            ContentDigest,
            field="lineage_scope_digest",
        )
        require_text(self.owner_id, field="owner_id")
        require_int(self.lease_epoch, field="lease_epoch", minimum=1)
        require_instance(
            self.fencing_token_digest,
            ContentDigest,
            field="fencing_token_digest",
        )
        if self.prior_lease_digest is not None:
            require_instance(
                self.prior_lease_digest, ContentDigest, field="prior_lease_digest"
            )
        granted = _require_canonical_utc(self.granted_at_utc, field="granted_at_utc")
        expires = _require_canonical_utc(self.expires_at_utc, field="expires_at_utc")
        if expires <= granted:
            raise contract_invalid(
                "continuation lease must expire after grant", field="expires_at_utc"
            )
        _validate_schema(
            self.schema_version,
            CANONICAL_CONTINUATION_LEASE_SCHEMA_VERSION,
            field="schema_version",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "lease_id": str(self.lease_id),
            "run_id": str(self.run_id),
            "attempt_id": str(self.attempt_id),
            "restore_checkpoint_digest": _optional_digest(
                self.restore_checkpoint_digest
            ),
            "lineage_scope_digest": _digest(self.lineage_scope_digest),
            "owner_id": self.owner_id,
            "lease_epoch": self.lease_epoch,
            "fencing_token_digest": _digest(self.fencing_token_digest),
            "prior_lease_digest": _optional_digest(self.prior_lease_digest),
            "granted_at_utc": self.granted_at_utc,
            "expires_at_utc": self.expires_at_utc,
        }

    def lease_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


class AttemptRetryBasisCodeV1(str, Enum):
    INITIAL_EXECUTION = "INITIAL_EXECUTION"
    CHANGED_CONDITION = "CHANGED_CONDITION"
    NEW_INFORMATION = "NEW_INFORMATION"
    EXACT_RESTORE_AFTER_FAILURE = "EXACT_RESTORE_AFTER_FAILURE"
    QUALIFICATION_INTEGRATION_ESCAPE = "QUALIFICATION_INTEGRATION_ESCAPE"


@dataclass(frozen=True, slots=True)
class AttemptRetryBasisV1:
    run_id: RunId
    candidate_attempt_id: AttemptId
    prior_attempt_id: AttemptId | None
    code: AttemptRetryBasisCodeV1
    reason: str
    semantic_spec_digest: ContentDigest
    prior_concretized_execution_digest: ContentDigest | None
    candidate_concretized_execution_digest: ContentDigest
    basis_evidence_digests: tuple[NamedDigestV1, ...]
    implementation_conformance_digest: ContentDigest | None
    schema_version: SchemaVersion = ATTEMPT_RETRY_BASIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.run_id, RunId, field="run_id")
        require_instance(
            self.candidate_attempt_id, AttemptId, field="candidate_attempt_id"
        )
        require_instance(self.code, AttemptRetryBasisCodeV1, field="code")
        require_text(self.reason, field="reason")
        require_instance(
            self.semantic_spec_digest, ContentDigest, field="semantic_spec_digest"
        )
        require_instance(
            self.candidate_concretized_execution_digest,
            ContentDigest,
            field="candidate_concretized_execution_digest",
        )
        if self.prior_attempt_id is not None:
            require_instance(self.prior_attempt_id, AttemptId, field="prior_attempt_id")
        if self.prior_concretized_execution_digest is not None:
            require_instance(
                self.prior_concretized_execution_digest,
                ContentDigest,
                field="prior_concretized_execution_digest",
            )
        if self.implementation_conformance_digest is not None:
            require_instance(
                self.implementation_conformance_digest,
                ContentDigest,
                field="implementation_conformance_digest",
            )
        _validate_digest_tuple(
            self.basis_evidence_digests, field="basis_evidence_digests"
        )
        initial = self.code is AttemptRetryBasisCodeV1.INITIAL_EXECUTION
        prior_absent = (
            self.prior_attempt_id is None
            and self.prior_concretized_execution_digest is None
        )
        prior_complete = (
            self.prior_attempt_id is not None
            and self.prior_concretized_execution_digest is not None
        )
        if (initial and not prior_absent) or (not initial and not prior_complete):
            raise contract_invalid(
                "initial execution and prior Attempt bindings disagree",
                field="prior_attempt_id",
            )
        if initial and self.basis_evidence_digests:
            raise contract_invalid(
                "initial execution cannot claim retry evidence",
                field="basis_evidence_digests",
            )
        if not initial and not self.basis_evidence_digests:
            raise contract_invalid(
                "retry requires changed-condition or new-information evidence",
                field="basis_evidence_digests",
            )
        _validate_schema(
            self.schema_version,
            ATTEMPT_RETRY_BASIS_SCHEMA_VERSION,
            field="schema_version",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "run_id": str(self.run_id),
            "candidate_attempt_id": str(self.candidate_attempt_id),
            "prior_attempt_id": (
                None if self.prior_attempt_id is None else str(self.prior_attempt_id)
            ),
            "code": self.code.value,
            "reason": self.reason,
            "semantic_spec_digest": _digest(self.semantic_spec_digest),
            "prior_concretized_execution_digest": _optional_digest(
                self.prior_concretized_execution_digest
            ),
            "candidate_concretized_execution_digest": _digest(
                self.candidate_concretized_execution_digest
            ),
            "basis_evidence_digests": [
                item.to_dict() for item in self.basis_evidence_digests
            ],
            "implementation_conformance_digest": _optional_digest(
                self.implementation_conformance_digest
            ),
        }

    def basis_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ExactImplementationConformanceV1:
    run_id: RunId
    candidate_attempt_id: AttemptId
    prior_implementation_digest: ContentDigest
    candidate_implementation_digest: ContentDigest
    candidate_abi_digest: ContentDigest
    candidate_device_conformance_envelope_digest: ContentDigest
    exact_equivalence_evidence_digests: tuple[NamedDigestV1, ...]
    qualification_delta_digest: ContentDigest
    schema_version: SchemaVersion = EXACT_IMPLEMENTATION_CONFORMANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.run_id, RunId, field="run_id")
        require_instance(
            self.candidate_attempt_id, AttemptId, field="candidate_attempt_id"
        )
        for field_name in (
            "prior_implementation_digest",
            "candidate_implementation_digest",
            "candidate_abi_digest",
            "candidate_device_conformance_envelope_digest",
            "qualification_delta_digest",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        _validate_digest_tuple(
            self.exact_equivalence_evidence_digests,
            field="exact_equivalence_evidence_digests",
            required=True,
        )
        _validate_schema(
            self.schema_version,
            EXACT_IMPLEMENTATION_CONFORMANCE_SCHEMA_VERSION,
            field="schema_version",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "run_id": str(self.run_id),
            "candidate_attempt_id": str(self.candidate_attempt_id),
            "prior_implementation_digest": _digest(self.prior_implementation_digest),
            "candidate_implementation_digest": _digest(
                self.candidate_implementation_digest
            ),
            "candidate_abi_digest": _digest(self.candidate_abi_digest),
            "candidate_device_conformance_envelope_digest": _digest(
                self.candidate_device_conformance_envelope_digest
            ),
            "exact_equivalence_evidence_digests": [
                item.to_dict() for item in self.exact_equivalence_evidence_digests
            ],
            "qualification_delta_digest": _digest(self.qualification_delta_digest),
        }

    def conformance_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class RestoreCheckpointAuthorizationV1:
    checkpoint_digest: ContentDigest
    run_id: RunId
    source_attempt_id: AttemptId
    source_attempt_terminal_digest: ContentDigest
    absolute_next_tick: int
    sealed_bundle_digest: ContentDigest
    completion_marker_digest: ContentDigest
    schema_version: SchemaVersion = RESTORE_CHECKPOINT_AUTHORIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(
            self.checkpoint_digest, ContentDigest, field="checkpoint_digest"
        )
        require_instance(self.run_id, RunId, field="run_id")
        require_instance(self.source_attempt_id, AttemptId, field="source_attempt_id")
        for field_name in (
            "source_attempt_terminal_digest",
            "sealed_bundle_digest",
            "completion_marker_digest",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        require_int(self.absolute_next_tick, field="absolute_next_tick")
        _validate_schema(
            self.schema_version,
            RESTORE_CHECKPOINT_AUTHORIZATION_SCHEMA_VERSION,
            field="schema_version",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "checkpoint_digest": _digest(self.checkpoint_digest),
            "run_id": str(self.run_id),
            "source_attempt_id": str(self.source_attempt_id),
            "source_attempt_terminal_digest": _digest(
                self.source_attempt_terminal_digest
            ),
            "absolute_next_tick": self.absolute_next_tick,
            "sealed_bundle_digest": _digest(self.sealed_bundle_digest),
            "completion_marker_digest": _digest(self.completion_marker_digest),
        }

    def authorization_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


class ContinuationLeaseEventCodeV1(str, Enum):
    HEARTBEAT = "HEARTBEAT"
    RENEWED = "RENEWED"
    RELEASED = "RELEASED"
    REASSIGNED = "REASSIGNED"
    EXPIRED = "EXPIRED"
    RECLAIMED = "RECLAIMED"
    PUBLISH_COMMITTED = "PUBLISH_COMMITTED"
    OWNER_DEAD = "OWNER_DEAD"


@dataclass(frozen=True, slots=True)
class ContinuationLeaseEventV1:
    lease_digest: ContentDigest
    run_id: RunId
    attempt_id: AttemptId
    owner_id: str
    lease_epoch: int
    fencing_token_digest: ContentDigest
    code: ContinuationLeaseEventCodeV1
    event_at_utc: str
    reason: str
    prior_event_digest: ContentDigest | None
    successor_lease_digest: ContentDigest | None
    publication_digest: ContentDigest | None
    owner_dead_proof_digest: ContentDigest | None = None
    schema_version: SchemaVersion = CONTINUATION_LEASE_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.lease_digest, ContentDigest, field="lease_digest")
        require_instance(self.run_id, RunId, field="run_id")
        require_instance(self.attempt_id, AttemptId, field="attempt_id")
        require_text(self.owner_id, field="owner_id")
        require_int(self.lease_epoch, field="lease_epoch", minimum=1)
        require_instance(
            self.fencing_token_digest,
            ContentDigest,
            field="fencing_token_digest",
        )
        require_instance(self.code, ContinuationLeaseEventCodeV1, field="code")
        _require_canonical_utc(self.event_at_utc, field="event_at_utc")
        require_text(self.reason, field="reason")
        for field_name in (
            "prior_event_digest",
            "successor_lease_digest",
            "publication_digest",
        ):
            value = getattr(self, field_name)
            if value is not None:
                require_instance(value, ContentDigest, field=field_name)
        successor_required = self.code in {
            ContinuationLeaseEventCodeV1.RENEWED,
            ContinuationLeaseEventCodeV1.REASSIGNED,
            ContinuationLeaseEventCodeV1.RECLAIMED,
        }
        if successor_required != (self.successor_lease_digest is not None):
            raise contract_invalid(
                "continuation lease event has an invalid successor binding",
                field="successor_lease_digest",
            )
        publication_required = (
            self.code is ContinuationLeaseEventCodeV1.PUBLISH_COMMITTED
        )
        if publication_required != (self.publication_digest is not None):
            raise contract_invalid(
                "continuation lease event has an invalid publication binding",
                field="publication_digest",
            )
        owner_dead = self.code is ContinuationLeaseEventCodeV1.OWNER_DEAD
        if owner_dead != (self.owner_dead_proof_digest is not None):
            raise contract_invalid(
                "dead-owner continuation event has an invalid proof binding",
                field="owner_dead_proof_digest",
            )
        if self.owner_dead_proof_digest is not None:
            require_instance(
                self.owner_dead_proof_digest,
                ContentDigest,
                field="owner_dead_proof_digest",
            )
        expected_schema = (
            CONTINUATION_LEASE_EVENT_SCHEMA_VERSION_V2
            if owner_dead
            else CONTINUATION_LEASE_EVENT_SCHEMA_VERSION
        )
        _validate_schema(self.schema_version, expected_schema, field="schema_version")

    def to_dict(self) -> dict[str, object]:
        value = {
            "schema_version": str(self.schema_version),
            "lease_digest": _digest(self.lease_digest),
            "run_id": str(self.run_id),
            "attempt_id": str(self.attempt_id),
            "owner_id": self.owner_id,
            "lease_epoch": self.lease_epoch,
            "fencing_token_digest": _digest(self.fencing_token_digest),
            "code": self.code.value,
            "event_at_utc": self.event_at_utc,
            "reason": self.reason,
            "prior_event_digest": _optional_digest(self.prior_event_digest),
            "successor_lease_digest": _optional_digest(self.successor_lease_digest),
            "publication_digest": _optional_digest(self.publication_digest),
        }
        if self.owner_dead_proof_digest is not None:
            value["owner_dead_proof_digest"] = _digest(
                self.owner_dead_proof_digest
            )
        return value

    def event_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class AttemptStartV1:
    attempt_id: AttemptId
    run_id: RunId
    monotonic_ordinal: int
    nonce: str
    concretized_execution_digest: ContentDigest
    implementation_digest: ContentDigest
    attempt_admission_request_digest: ContentDigest
    scheduler_assignment_lease_digest: ContentDigest | None
    resource_start_lease_digest: ContentDigest
    post_admission_attempt_plan_digest: ContentDigest
    admission_disposition_digest: ContentDigest
    programme_work_session_assignment_dispatch_binding_digest: ContentDigest
    parent_attempt_id: AttemptId | None
    restore_checkpoint_digest: ContentDigest | None
    canonical_continuation_lease_digest: ContentDigest
    owner_binding_digest: ContentDigest
    execution_side_effect_digest: ContentDigest
    initial_frontiers: AttemptFrontiersV1
    schema_version: SchemaVersion = ATTEMPT_START_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.attempt_id, AttemptId, field="attempt_id")
        require_instance(self.run_id, RunId, field="run_id")
        require_int(self.monotonic_ordinal, field="monotonic_ordinal", minimum=1)
        require_text(self.nonce, field="nonce")
        if self.attempt_id != derive_attempt_id(
            run_id=self.run_id,
            monotonic_ordinal=self.monotonic_ordinal,
            nonce=self.nonce,
        ):
            raise contract_invalid(
                "AttemptId differs from Run/ordinal/nonce identity", field="attempt_id"
            )
        for field_name in (
            "concretized_execution_digest",
            "implementation_digest",
            "attempt_admission_request_digest",
            "resource_start_lease_digest",
            "post_admission_attempt_plan_digest",
            "admission_disposition_digest",
            "programme_work_session_assignment_dispatch_binding_digest",
            "canonical_continuation_lease_digest",
            "owner_binding_digest",
            "execution_side_effect_digest",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        if self.scheduler_assignment_lease_digest is not None:
            require_instance(
                self.scheduler_assignment_lease_digest,
                ContentDigest,
                field="scheduler_assignment_lease_digest",
            )
        if self.parent_attempt_id is not None:
            require_instance(
                self.parent_attempt_id, AttemptId, field="parent_attempt_id"
            )
        if self.restore_checkpoint_digest is not None:
            require_instance(
                self.restore_checkpoint_digest,
                ContentDigest,
                field="restore_checkpoint_digest",
            )
        require_instance(
            self.initial_frontiers, AttemptFrontiersV1, field="initial_frontiers"
        )
        initial_values = (
            self.initial_frontiers.last_dispatched_next_tick,
            self.initial_frontiers.last_computed_next_tick,
            self.initial_frontiers.durable_checkpoint_next_tick,
            self.initial_frontiers.published_artifact_next_tick,
        )
        if self.restore_checkpoint_digest is None and len(set(initial_values)) != 1:
            raise contract_invalid(
                "non-restored Attempt cannot claim unequal pre-existing progress; "
                "initial frontiers must share one fresh absolute start coordinate",
                field="initial_frontiers",
            )
        _validate_schema(
            self.schema_version, ATTEMPT_START_SCHEMA_VERSION, field="schema_version"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "attempt_id": str(self.attempt_id),
            "run_id": str(self.run_id),
            "monotonic_ordinal": self.monotonic_ordinal,
            "nonce": self.nonce,
            "concretized_execution_digest": _digest(self.concretized_execution_digest),
            "implementation_digest": _digest(self.implementation_digest),
            "attempt_admission_request_digest": _digest(
                self.attempt_admission_request_digest
            ),
            "scheduler_assignment_lease_digest": _optional_digest(
                self.scheduler_assignment_lease_digest
            ),
            "resource_start_lease_digest": _digest(self.resource_start_lease_digest),
            "post_admission_attempt_plan_digest": _digest(
                self.post_admission_attempt_plan_digest
            ),
            "admission_disposition_digest": _digest(self.admission_disposition_digest),
            "programme_work_session_assignment_dispatch_binding_digest": _digest(
                self.programme_work_session_assignment_dispatch_binding_digest
            ),
            "parent_attempt_id": (
                None if self.parent_attempt_id is None else str(self.parent_attempt_id)
            ),
            "restore_checkpoint_digest": _optional_digest(
                self.restore_checkpoint_digest
            ),
            "canonical_continuation_lease_digest": _digest(
                self.canonical_continuation_lease_digest
            ),
            "owner_binding_digest": _digest(self.owner_binding_digest),
            "execution_side_effect_digest": _digest(self.execution_side_effect_digest),
            "initial_frontiers": self.initial_frontiers.to_dict(),
        }

    def start_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    def validate_authorized_disposition(
        self, disposition: AttemptAdmissionDispositionV1
    ) -> None:
        require_instance(
            disposition,
            AttemptAdmissionDispositionV1,
            field="admission_disposition",
        )
        if disposition.code is not AttemptAdmissionDispositionCodeV1.START_AUTHORIZED:
            raise contract_invalid(
                "Attempt start requires START_AUTHORIZED disposition",
                field="admission_disposition",
            )
        comparisons = (
            (self.run_id, disposition.run_id, "run_id"),
            (
                self.attempt_admission_request_digest,
                disposition.attempt_admission_request_digest,
                "attempt_admission_request_digest",
            ),
            (
                self.scheduler_assignment_lease_digest,
                disposition.scheduler_assignment_lease_digest,
                "scheduler_assignment_lease_digest",
            ),
            (
                self.resource_start_lease_digest,
                disposition.resource_start_lease_digest,
                "resource_start_lease_digest",
            ),
            (
                self.admission_disposition_digest,
                disposition.disposition_digest(),
                "admission_disposition_digest",
            ),
        )
        for actual, expected, field_name in comparisons:
            if actual != expected:
                raise contract_invalid(
                    "Attempt start differs from authorized admission",
                    field=field_name,
                )


class AttemptTerminalStatusV1(str, Enum):
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"
    QUARANTINED = "QUARANTINED"


@dataclass(frozen=True, slots=True)
class AttemptTerminalV1:
    attempt_id: AttemptId
    attempt_start_digest: ContentDigest
    status: AttemptTerminalStatusV1
    final_frontiers: AttemptFrontiersV1
    result_digest: ContentDigest
    reason: str
    schema_version: SchemaVersion = ATTEMPT_TERMINAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_instance(self.attempt_id, AttemptId, field="attempt_id")
        require_instance(
            self.attempt_start_digest, ContentDigest, field="attempt_start_digest"
        )
        require_instance(self.status, AttemptTerminalStatusV1, field="status")
        require_instance(
            self.final_frontiers, AttemptFrontiersV1, field="final_frontiers"
        )
        require_instance(self.result_digest, ContentDigest, field="result_digest")
        require_text(self.reason, field="reason")
        _validate_schema(
            self.schema_version, ATTEMPT_TERMINAL_SCHEMA_VERSION, field="schema_version"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "attempt_id": str(self.attempt_id),
            "attempt_start_digest": _digest(self.attempt_start_digest),
            "status": self.status.value,
            "final_frontiers": self.final_frontiers.to_dict(),
            "result_digest": _digest(self.result_digest),
            "reason": self.reason,
        }

    def terminal_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    def validate_transition(self, start: AttemptStartV1) -> None:
        require_instance(start, AttemptStartV1, field="attempt_start")
        if self.attempt_id != start.attempt_id:
            raise contract_invalid(
                "terminal AttemptId differs from start", field="attempt_id"
            )
        if self.attempt_start_digest != start.start_digest():
            raise contract_invalid(
                "terminal receipt differs from Attempt start",
                field="attempt_start_digest",
            )
        initial = start.initial_frontiers
        for field_name in (
            "last_dispatched_next_tick",
            "last_computed_next_tick",
            "durable_checkpoint_next_tick",
            "published_artifact_next_tick",
        ):
            if getattr(self.final_frontiers, field_name) < getattr(initial, field_name):
                raise contract_invalid(
                    "terminal frontier cannot move backwards", field=field_name
                )
