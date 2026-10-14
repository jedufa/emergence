"""Pure validation of Scheduler-issued Launchpad Attempt bundles.

Parsing this bundle proves exact identity, lease, and CRCA-law consistency.  It
does not issue a lease, create an Attempt, launch a provider, or itself grant
execution authority.  The only authority-bearing writer is the Scheduler
issuer in ``tools.scheduler.launchpad_attempts``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Mapping

from emergence_contracts import (
    AttemptAdmissionDispositionCodeV1,
    AttemptAdmissionDispositionV1,
    AttemptAdmissionRequestV1,
    AttemptFrontiersV1,
    AttemptId,
    AttemptRetryBasisCodeV1,
    AttemptRetryBasisV1,
    AttemptStartV1,
    AttemptTerminalStatusV1,
    AttemptTerminalV1,
    BackendId,
    CanonicalContinuationLeaseV1,
    CONTINUATION_LEASE_EVENT_SCHEMA_VERSION,
    CONTINUATION_LEASE_EVENT_SCHEMA_VERSION_V2,
    ConcretizedExecutionV1,
    ContentDigest,
    ContinuationLeaseEventCodeV1,
    ContinuationLeaseEventV1,
    ExactImplementationConformanceV1,
    ExecutionImplementationV1,
    ExperimentId,
    LeaseId,
    LifecycleBindingV1,
    NamedDigestV1,
    RestoreCheckpointAuthorizationV1,
    RunId,
    RunOccurrenceV1,
    SemanticRunSpecV1,
    canonical_sha256,
)
from engine.campaigns.cuda_full_evolution_v2_shared_route import (
    CudaFullEvolutionV2SharedRouteError,
    shared_w32_gpu_policy_v1,
)
from tools.scheduler.programme_contracts import (
    GpuResourceTopologyError,
    shared_gpu_lane_candidate_sets_v1,
)
from eiel_composition.attempt_admission import (
    AttemptRecordV1,
    ContinuationLeaseLedgerV1,
    validate_attempt_start_v1,
)


AUTHORITATIVE_BUNDLE_SCHEMA_VERSION = "crca_authoritative_launchpad_attempt_bundle_v1"
AUTHORITATIVE_BUNDLE_SCHEMA_VERSION_V2 = (
    "crca_authoritative_launchpad_attempt_bundle_v2"
)


class LaunchpadAttemptAuthorityError(ValueError):
    """A purported Scheduler bundle is incomplete, forged, or stale."""


def _fail(message: str) -> None:
    raise LaunchpadAttemptAuthorityError(message)


def _exact(raw: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    observed = set(raw)
    if observed != expected:
        _fail(
            f"{label} fields differ; missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )


def _digest(raw: object, *, field: str) -> ContentDigest:
    if not isinstance(raw, Mapping):
        _fail(f"{field} must be a content digest")
    try:
        return ContentDigest(
            algorithm=str(raw.get("algorithm") or ""),
            hex=str(raw.get("hex") or ""),
        )
    except Exception as error:
        raise LaunchpadAttemptAuthorityError(f"{field} is not exact SHA-256") from error


def _optional_digest(raw: object, *, field: str) -> ContentDigest | None:
    return None if raw is None else _digest(raw, field=field)


def _canonical_utc(raw: object, *, field: str) -> datetime:
    if not isinstance(raw, str) or len(raw) != 20 or not raw.endswith("Z"):
        _fail(f"{field} must use canonical second-resolution UTC")
    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise LaunchpadAttemptAuthorityError(f"{field} is invalid") from error


def _text(raw: object, *, field: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        _fail(f"{field} must be non-empty text")
    return raw


def _positive_int(raw: object, *, field: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        _fail(f"{field} must be a positive integer")
    return raw


_CLAIM_DIGEST_FIELDS = frozenset(
    {
        "scheduler_owner_digest",
        "execution_owner_digest",
        "lineage_scope_digest",
        "execution_request_digest",
        "campaign_plan_digest",
        "source_digest",
        "baseline_digest",
        "preflight_receipt_digest",
        "lifecycle_binding_digest",
        "lifecycle_contract_digest",
        "native_artifact_digest",
        "semantic_run_spec_digest",
        "run_occurrence_digest",
        "execution_implementation_digest",
        "concretized_execution_digest",
        "attempt_admission_request_digest",
        "retry_basis_digest",
        "post_admission_attempt_plan_digest",
        "programme_binding_digest",
        "initial_frontiers_digest",
    }
)
_CLAIM_FIELDS = {
    "schema_version",
    "claim_id",
    "scheduler_owner_id",
    "scheduler_owner_digest",
    "execution_owner_id",
    "execution_owner_digest",
    "run_id",
    "attempt_id",
    "lineage_scope_digest",
    "execution_request_digest",
    "campaign_plan_digest",
    "source_digest",
    "baseline_digest",
    "preflight_receipt_digest",
    "lifecycle_binding_digest",
    "lifecycle_contract_digest",
    "run_root",
    "provider_id",
    "route_id",
    "device_id",
    "native_artifact_digest",
    "batch_size",
    "ticks_per_dispatch",
    "execution_mode",
    "semantic_run_spec_digest",
    "run_occurrence_digest",
    "execution_implementation_digest",
    "concretized_execution_digest",
    "attempt_admission_request_digest",
    "retry_basis_digest",
    "post_admission_attempt_plan_digest",
    "programme_binding_digest",
    "initial_frontiers_digest",
    "created_at_utc",
}
_RESOURCE_LEASE_FIELDS = {
    "schema_version",
    "lease_id",
    "scheduler_owner_id",
    "scheduler_owner_digest",
    "execution_claim_digest",
    "run_id",
    "attempt_id",
    "attempt_admission_request_digest",
    "resource_ids",
    "resource_scope_digest",
    "resource_observation_id",
    "resource_observation_digest",
    "resource_recommendation_id",
    "resource_recommendation_digest",
    "live_provider_check_digest",
    "lease_epoch",
    "fencing_token_digest",
    "prior_lease_digest",
    "issued_at_utc",
    "start_before_utc",
    "expires_at_utc",
}
_RESOURCE_EVENT_FIELDS = {
    "schema_version",
    "lease_digest",
    "code",
    "scheduler_owner_id",
    "fencing_token_digest",
    "event_at_utc",
    "reason",
    "prior_event_digest",
    "successor_lease_digest",
}
_START_EFFECT_FIELDS = {
    "schema_version",
    "effect_kind",
    "claim_digest",
    "run_id",
    "attempt_id",
    "resource_start_lease_digest",
    "continuation_lease_digest",
    "scheduler_owner_id",
    "scheduler_owner_digest",
    "process_owner_id",
    "process_owner_digest",
    "fencing_token_digest",
    "started_at_utc",
}
_START_EFFECT_V2_FIELDS = {
    *_START_EFFECT_FIELDS,
    "process_id",
    "process_start_identity",
}


def _launchpad_claim_v1(raw: Mapping[str, Any]) -> ContentDigest:
    _exact(raw, _CLAIM_FIELDS, label="LaunchpadAttemptClaimV1")
    if raw.get("schema_version") != "crca_launchpad_attempt_claim_v1":
        _fail("Launchpad Attempt claim schema is unsupported")
    for field in (
        "claim_id",
        "scheduler_owner_id",
        "execution_owner_id",
        "run_id",
        "attempt_id",
        "run_root",
        "provider_id",
        "route_id",
        "device_id",
        "execution_mode",
    ):
        _text(raw.get(field), field=f"claim.{field}")
    if not Path(str(raw["run_root"])).is_absolute():
        _fail("claim.run_root must be an exact absolute path")
    _positive_int(raw.get("batch_size"), field="claim.batch_size")
    _positive_int(raw.get("ticks_per_dispatch"), field="claim.ticks_per_dispatch")
    for field in _CLAIM_DIGEST_FIELDS:
        _digest(raw.get(field), field=f"claim.{field}")
    _canonical_utc(raw.get("created_at_utc"), field="claim.created_at_utc")
    expected_claim_id = (
        "LPCLAIM-"
        + canonical_sha256(
            {key: value for key, value in raw.items() if key != "claim_id"}
        )
        .hex[:24]
        .upper()
    )
    if raw.get("claim_id") != expected_claim_id:
        _fail("Launchpad claim ID differs from its exact identity")
    return canonical_sha256(dict(raw))


def _launchpad_resource_lease_v1(
    raw: Mapping[str, Any],
    *,
    claim: Mapping[str, Any],
    claim_digest: ContentDigest,
    campaign_plan: Mapping[str, Any],
) -> tuple[ContentDigest, datetime, datetime, datetime]:
    _exact(raw, _RESOURCE_LEASE_FIELDS, label="LaunchpadResourceStartLeaseV1")
    if raw.get("schema_version") != "crca_launchpad_resource_start_lease_v1":
        _fail("Launchpad resource-start lease schema is unsupported")
    for field in (
        "lease_id",
        "scheduler_owner_id",
        "run_id",
        "attempt_id",
        "resource_observation_id",
        "resource_recommendation_id",
    ):
        _text(raw.get(field), field=f"resource_start_lease.{field}")
    resource_ids = raw.get("resource_ids")
    if (
        not isinstance(resource_ids, list)
        or not resource_ids
        or any(
            not isinstance(value, str) or not value.strip() for value in resource_ids
        )
        or resource_ids != sorted(set(resource_ids))
    ):
        _fail("resource_start_lease.resource_ids must be sorted, unique, and non-empty")
    device_id = str(claim["device_id"])
    if not any(
        resource_id == device_id
        or resource_id.endswith(f":{device_id}")
        or resource_id.startswith(f"gpu:{device_id}/lane/")
        for resource_id in resource_ids
    ):
        _fail("resource-start lease resources do not bind the claimed device")
    execution = campaign_plan.get("execution")
    gpu_resource_policy = (
        execution.get("gpu_resource_policy")
        if isinstance(execution, Mapping)
        else None
    )
    if gpu_resource_policy is None:
        expected_resource_ids = [f"gpu:{device_id}"]
        expected_scope = {
            "schema_version": "eiel_exclusive_gpu_scope_v1",
            "resource_ids": expected_resource_ids,
            "run_id": str(claim["run_id"]),
        }
        if (
            resource_ids != expected_resource_ids
            or _digest(
                raw.get("resource_scope_digest"),
                field="resource_start_lease.resource_scope_digest",
            )
            != canonical_sha256(expected_scope)
        ):
            _fail("exclusive GPU resource differs from canonical topology")
    else:
        try:
            policy = shared_w32_gpu_policy_v1()
            if not isinstance(gpu_resource_policy, Mapping) or dict(
                gpu_resource_policy
            ) != policy:
                raise CudaFullEvolutionV2SharedRouteError(
                    "shared W32 GPU policy differs from the canonical contract"
                )
            expected_candidates = shared_gpu_lane_candidate_sets_v1(device_id)
        except (
            CudaFullEvolutionV2SharedRouteError,
            GpuResourceTopologyError,
        ) as exc:
            raise LaunchpadAttemptAuthorityError(str(exc)) from exc
        expected_scope = {
            "schema_version": "eiel_shared_gpu_lane_scope_v1",
            "run_id": str(claim["run_id"]),
            "device_id": device_id,
            "gpu_resource_policy": policy,
            "candidate_resource_sets": [
                list(candidate) for candidate in expected_candidates
            ],
        }
        if (
            tuple(resource_ids) not in expected_candidates
            or _digest(
                raw.get("resource_scope_digest"),
                field="resource_start_lease.resource_scope_digest",
            )
            != canonical_sha256(expected_scope)
        ):
            _fail("shared GPU resource differs from canonical topology")
    _positive_int(raw.get("lease_epoch"), field="resource_start_lease.lease_epoch")
    for field in (
        "scheduler_owner_digest",
        "execution_claim_digest",
        "attempt_admission_request_digest",
        "resource_scope_digest",
        "resource_observation_digest",
        "resource_recommendation_digest",
        "live_provider_check_digest",
        "fencing_token_digest",
    ):
        _digest(raw.get(field), field=f"resource_start_lease.{field}")
    _optional_digest(
        raw.get("prior_lease_digest"),
        field="resource_start_lease.prior_lease_digest",
    )
    issued = _canonical_utc(
        raw.get("issued_at_utc"), field="resource_start_lease.issued_at_utc"
    )
    start_before = _canonical_utc(
        raw.get("start_before_utc"),
        field="resource_start_lease.start_before_utc",
    )
    expires = _canonical_utc(
        raw.get("expires_at_utc"), field="resource_start_lease.expires_at_utc"
    )
    if not issued < start_before <= expires:
        _fail("resource-start lease interval is invalid")
    if (
        _digest(
            raw.get("execution_claim_digest"),
            field="resource_start_lease.execution_claim_digest",
        )
        != claim_digest
        or raw.get("scheduler_owner_id") != claim.get("scheduler_owner_id")
        or raw.get("scheduler_owner_digest") != claim.get("scheduler_owner_digest")
        or raw.get("run_id") != claim.get("run_id")
        or raw.get("attempt_id") != claim.get("attempt_id")
        or raw.get("attempt_admission_request_digest")
        != claim.get("attempt_admission_request_digest")
    ):
        _fail("resource-start lease differs from its exact claim")
    resource_seed = {
        "claim_digest": claim_digest.to_dict(),
        "resource_ids": resource_ids,
        "lease_epoch": raw["lease_epoch"],
        "prior_lease_digest": raw.get("prior_lease_digest"),
        "issued_at_utc": raw["issued_at_utc"],
    }
    seed_digest = canonical_sha256(resource_seed)
    expected_lease_id = f"LPRSTART-{seed_digest.hex[:24].upper()}"
    expected_fence = canonical_sha256(
        {
            "resource_seed": resource_seed,
            "scheduler_owner_digest": claim["scheduler_owner_digest"],
        }
    )
    if (
        raw.get("lease_id") != expected_lease_id
        or _digest(
            raw.get("fencing_token_digest"),
            field="resource_start_lease.fencing_token_digest",
        )
        != expected_fence
    ):
        _fail("resource-start lease identity or fence is not Scheduler-derived")
    return canonical_sha256(dict(raw)), issued, start_before, expires


def _resource_started_event_v1(
    raw: Mapping[str, Any],
    *,
    claim: Mapping[str, Any],
    lease: Mapping[str, Any],
    lease_digest: ContentDigest,
    issued: datetime,
    start_before: datetime,
) -> datetime:
    _exact(raw, _RESOURCE_EVENT_FIELDS, label="ResourceStartLeaseEventV1")
    if (
        raw.get("schema_version") != "crca_resource_start_lease_event_v1"
        or raw.get("code") != "STARTED"
    ):
        _fail("resource-start event is not an exact STARTED event")
    _text(
        raw.get("scheduler_owner_id"), field="resource_start_event.scheduler_owner_id"
    )
    _text(raw.get("reason"), field="resource_start_event.reason")
    event_at = _canonical_utc(
        raw.get("event_at_utc"), field="resource_start_event.event_at_utc"
    )
    if not issued <= event_at < start_before:
        _fail("STARTED event lies outside its exact start window")
    if (
        _digest(raw.get("lease_digest"), field="resource_start_event.lease_digest")
        != lease_digest
        or raw.get("scheduler_owner_id") != claim.get("scheduler_owner_id")
        or raw.get("scheduler_owner_id") != lease.get("scheduler_owner_id")
        or _digest(
            raw.get("fencing_token_digest"),
            field="resource_start_event.fencing_token_digest",
        )
        != _digest(
            lease.get("fencing_token_digest"),
            field="resource_start_lease.fencing_token_digest",
        )
        or raw.get("prior_event_digest") is not None
        or raw.get("successor_lease_digest") is not None
    ):
        _fail("STARTED event owner, fence, or binding differs")
    return event_at


def _launchpad_start_effect_v1(
    raw: Mapping[str, Any],
    *,
    claim: Mapping[str, Any],
    claim_digest: ContentDigest,
    lease_digest: ContentDigest,
    continuation_digest: ContentDigest,
    continuation_fence: Mapping[str, Any],
    started_at: datetime,
) -> ContentDigest:
    schema = raw.get("schema_version")
    _exact(
        raw,
        (
            _START_EFFECT_V2_FIELDS
            if schema == "crca_launchpad_attempt_start_effect_v2"
            else _START_EFFECT_FIELDS
        ),
        label="LaunchpadAttemptStartEffectV1",
    )
    if (
        schema
        not in {
            "crca_launchpad_attempt_start_effect_v1",
            "crca_launchpad_attempt_start_effect_v2",
        }
        or raw.get("effect_kind") != "SCHEDULER_EXECUTION_AUTHORITY_GRANTED"
    ):
        _fail("Launchpad start effect kind or schema is invalid")
    for field in (
        "run_id",
        "attempt_id",
        "scheduler_owner_id",
        "process_owner_id",
    ):
        _text(raw.get(field), field=f"start_effect.{field}")
    for field in (
        "claim_digest",
        "resource_start_lease_digest",
        "continuation_lease_digest",
        "scheduler_owner_digest",
        "process_owner_digest",
        "fencing_token_digest",
    ):
        _digest(raw.get(field), field=f"start_effect.{field}")
    if schema == "crca_launchpad_attempt_start_effect_v2":
        _positive_int(raw.get("process_id"), field="start_effect.process_id")
        process_identity = _text(
            raw.get("process_start_identity"),
            field="start_effect.process_start_identity",
        )
        if not process_identity.startswith("linux-proc:"):
            _fail("Launchpad V2 start effect process identity is not Linux procfs")
    if (
        _canonical_utc(raw.get("started_at_utc"), field="start_effect.started_at_utc")
        != started_at
    ):
        _fail("Launchpad start effect time differs from STARTED event")
    if (
        _digest(raw.get("claim_digest"), field="start_effect.claim_digest")
        != claim_digest
        or _digest(
            raw.get("resource_start_lease_digest"),
            field="start_effect.resource_start_lease_digest",
        )
        != lease_digest
        or _digest(
            raw.get("continuation_lease_digest"),
            field="start_effect.continuation_lease_digest",
        )
        != continuation_digest
        or raw.get("run_id") != claim.get("run_id")
        or raw.get("attempt_id") != claim.get("attempt_id")
        or raw.get("scheduler_owner_id") != claim.get("scheduler_owner_id")
        or raw.get("scheduler_owner_digest") != claim.get("scheduler_owner_digest")
        or raw.get("process_owner_id") != claim.get("execution_owner_id")
        or raw.get("process_owner_digest") != claim.get("execution_owner_digest")
        or raw.get("fencing_token_digest")
        != continuation_fence.get("fencing_token_digest")
    ):
        _fail("Launchpad start effect owner, fence, or binding differs")
    return canonical_sha256(dict(raw))


def _named_digests(raw: object, *, field: str) -> tuple[NamedDigestV1, ...]:
    if not isinstance(raw, list):
        _fail(f"{field} must be an array")
    values: list[NamedDigestV1] = []
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            _fail(f"{field}[{index}] must be an object")
        _exact(row, {"name", "digest"}, label=f"{field}[{index}]")
        values.append(
            NamedDigestV1(
                name=str(row.get("name") or ""),
                digest=_digest(row.get("digest"), field=f"{field}[{index}].digest"),
            )
        )
    return tuple(values)


def _round_trip(value: object, raw: Mapping[str, Any], *, label: str) -> Any:
    if value.to_dict() != dict(raw):
        _fail(f"{label} is not canonical")
    return value


def _semantic(raw: Mapping[str, Any]) -> SemanticRunSpecV1:
    return _round_trip(
        SemanticRunSpecV1(
            experiment_id=ExperimentId(str(raw.get("experiment_id") or "")),
            system_assembly_digest=_digest(
                raw.get("system_assembly_digest"), field="system_assembly_digest"
            ),
            semantic_parameters_digest=_digest(
                raw.get("semantic_parameters_digest"),
                field="semantic_parameters_digest",
            ),
            rng_contract_digest=_digest(
                raw.get("rng_contract_digest"), field="rng_contract_digest"
            ),
            seed_digest=_digest(raw.get("seed_digest"), field="seed_digest"),
            absolute_start_next_tick=int(raw.get("absolute_start_next_tick")),
            absolute_target_next_tick=int(raw.get("absolute_target_next_tick")),
            trajectory_checkpoint_ticks=tuple(
                raw.get("trajectory_checkpoint_ticks") or ()
            ),
            measurement_behavior_digest=_digest(
                raw.get("measurement_behavior_digest"),
                field="measurement_behavior_digest",
            ),
            control_and_branch_semantics_digest=_digest(
                raw.get("control_and_branch_semantics_digest"),
                field="control_and_branch_semantics_digest",
            ),
        ),
        raw,
        label="SemanticRunSpecV1",
    )


def _implementation(raw: Mapping[str, Any]) -> ExecutionImplementationV1:
    return _round_trip(
        ExecutionImplementationV1(
            source_commit=str(raw.get("source_commit") or ""),
            source_tree_digest=_digest(
                raw.get("source_tree_digest"), field="source_tree_digest"
            ),
            dirty_patch_digest=_optional_digest(
                raw.get("dirty_patch_digest"), field="dirty_patch_digest"
            ),
            toolchain_digest=_digest(
                raw.get("toolchain_digest"), field="toolchain_digest"
            ),
            environment_digest=_digest(
                raw.get("environment_digest"), field="environment_digest"
            ),
            provider_id=str(raw.get("provider_id") or ""),
            backend_id=BackendId(str(raw.get("backend_id") or "")),
            runtime_profile_id=str(raw.get("runtime_profile_id") or ""),
            native_artifact_digests=_named_digests(
                raw.get("native_artifact_digests"), field="native_artifact_digests"
            ),
            abi_digest=_digest(raw.get("abi_digest"), field="abi_digest"),
            device_conformance_envelope_digest=_digest(
                raw.get("device_conformance_envelope_digest"),
                field="device_conformance_envelope_digest",
            ),
        ),
        raw,
        label="ExecutionImplementationV1",
    )


def _lifecycle(raw: Mapping[str, Any]) -> LifecycleBindingV1:
    return _round_trip(
        LifecycleBindingV1(
            output_root=str(raw.get("output_root") or ""),
            artifact_lifecycle_contract_digest=_digest(
                raw.get("artifact_lifecycle_contract_digest"),
                field="artifact_lifecycle_contract_digest",
            ),
            inventory_role_digests=_named_digests(
                raw.get("inventory_role_digests"), field="inventory_role_digests"
            ),
            initial_hold_digests=_named_digests(
                raw.get("initial_hold_digests"), field="initial_hold_digests"
            ),
            retention_profile_id=str(raw.get("retention_profile_id") or ""),
        ),
        raw,
        label="LifecycleBindingV1",
    )


def _admission_request(raw: Mapping[str, Any]) -> AttemptAdmissionRequestV1:
    return _round_trip(
        AttemptAdmissionRequestV1(
            resource_requirements_digest=_digest(
                raw.get("resource_requirements_digest"),
                field="resource_requirements_digest",
            ),
            executor_constraints_digest=_digest(
                raw.get("executor_constraints_digest"),
                field="executor_constraints_digest",
            ),
            requested_restore_checkpoint_digest=_optional_digest(
                raw.get("requested_restore_checkpoint_digest"),
                field="requested_restore_checkpoint_digest",
            ),
            observational_persistence_cadence_digest=_optional_digest(
                raw.get("observational_persistence_cadence_digest"),
                field="observational_persistence_cadence_digest",
            ),
            proposed_reason=str(raw.get("proposed_reason") or ""),
        ),
        raw,
        label="AttemptAdmissionRequestV1",
    )


def _concretized(raw: Mapping[str, Any]) -> ConcretizedExecutionV1:
    fields = (
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
    return _round_trip(
        ConcretizedExecutionV1(
            **{field: _digest(raw.get(field), field=field) for field in fields}
        ),
        raw,
        label="ConcretizedExecutionV1",
    )


def _run(raw: Mapping[str, Any]) -> RunOccurrenceV1:
    return _round_trip(
        RunOccurrenceV1(
            run_id=RunId(str(raw.get("run_id") or "")),
            experiment_id=ExperimentId(str(raw.get("experiment_id") or "")),
            cell_or_repeat_ordinal=int(raw.get("cell_or_repeat_ordinal")),
            semantic_spec_digest=_digest(
                raw.get("semantic_spec_digest"), field="semantic_spec_digest"
            ),
            parent_checkpoint_digest=_optional_digest(
                raw.get("parent_checkpoint_digest"), field="parent_checkpoint_digest"
            ),
            branch_spec_digest=_optional_digest(
                raw.get("branch_spec_digest"), field="branch_spec_digest"
            ),
        ),
        raw,
        label="RunOccurrenceV1",
    )


def _frontiers(raw: Mapping[str, Any]) -> AttemptFrontiersV1:
    return _round_trip(
        AttemptFrontiersV1(
            last_dispatched_next_tick=int(raw.get("last_dispatched_next_tick")),
            last_computed_next_tick=int(raw.get("last_computed_next_tick")),
            durable_checkpoint_next_tick=int(raw.get("durable_checkpoint_next_tick")),
            published_artifact_next_tick=int(raw.get("published_artifact_next_tick")),
        ),
        raw,
        label="AttemptFrontiersV1",
    )


def _disposition(raw: Mapping[str, Any]) -> AttemptAdmissionDispositionV1:
    return _round_trip(
        AttemptAdmissionDispositionV1(
            run_id=RunId(str(raw.get("run_id") or "")),
            attempt_admission_request_digest=_digest(
                raw.get("attempt_admission_request_digest"),
                field="attempt_admission_request_digest",
            ),
            code=AttemptAdmissionDispositionCodeV1(str(raw.get("code") or "")),
            reason=str(raw.get("reason") or ""),
            scheduler_assignment_lease_digest=_optional_digest(
                raw.get("scheduler_assignment_lease_digest"),
                field="scheduler_assignment_lease_digest",
            ),
            resource_start_lease_digest=_optional_digest(
                raw.get("resource_start_lease_digest"),
                field="resource_start_lease_digest",
            ),
        ),
        raw,
        label="AttemptAdmissionDispositionV1",
    )


def _retry(raw: Mapping[str, Any]) -> AttemptRetryBasisV1:
    return _round_trip(
        AttemptRetryBasisV1(
            run_id=RunId(str(raw.get("run_id") or "")),
            candidate_attempt_id=AttemptId(str(raw.get("candidate_attempt_id") or "")),
            prior_attempt_id=(
                None
                if raw.get("prior_attempt_id") is None
                else AttemptId(str(raw.get("prior_attempt_id")))
            ),
            code=AttemptRetryBasisCodeV1(str(raw.get("code") or "")),
            reason=str(raw.get("reason") or ""),
            semantic_spec_digest=_digest(
                raw.get("semantic_spec_digest"), field="semantic_spec_digest"
            ),
            prior_concretized_execution_digest=_optional_digest(
                raw.get("prior_concretized_execution_digest"),
                field="prior_concretized_execution_digest",
            ),
            candidate_concretized_execution_digest=_digest(
                raw.get("candidate_concretized_execution_digest"),
                field="candidate_concretized_execution_digest",
            ),
            basis_evidence_digests=_named_digests(
                raw.get("basis_evidence_digests"), field="basis_evidence_digests"
            ),
            implementation_conformance_digest=_optional_digest(
                raw.get("implementation_conformance_digest"),
                field="implementation_conformance_digest",
            ),
        ),
        raw,
        label="AttemptRetryBasisV1",
    )


def _continuation(raw: Mapping[str, Any]) -> CanonicalContinuationLeaseV1:
    return _round_trip(
        CanonicalContinuationLeaseV1(
            lease_id=LeaseId(str(raw.get("lease_id") or "")),
            run_id=RunId(str(raw.get("run_id") or "")),
            attempt_id=AttemptId(str(raw.get("attempt_id") or "")),
            restore_checkpoint_digest=_optional_digest(
                raw.get("restore_checkpoint_digest"),
                field="restore_checkpoint_digest",
            ),
            lineage_scope_digest=_digest(
                raw.get("lineage_scope_digest"), field="lineage_scope_digest"
            ),
            owner_id=str(raw.get("owner_id") or ""),
            lease_epoch=int(raw.get("lease_epoch")),
            fencing_token_digest=_digest(
                raw.get("fencing_token_digest"), field="fencing_token_digest"
            ),
            prior_lease_digest=_optional_digest(
                raw.get("prior_lease_digest"), field="prior_lease_digest"
            ),
            granted_at_utc=str(raw.get("granted_at_utc") or ""),
            expires_at_utc=str(raw.get("expires_at_utc") or ""),
        ),
        raw,
        label="CanonicalContinuationLeaseV1",
    )


def _continuation_event(raw: Mapping[str, Any]) -> ContinuationLeaseEventV1:
    return _round_trip(
        ContinuationLeaseEventV1(
            lease_digest=_digest(raw.get("lease_digest"), field="lease_digest"),
            run_id=RunId(str(raw.get("run_id") or "")),
            attempt_id=AttemptId(str(raw.get("attempt_id") or "")),
            owner_id=str(raw.get("owner_id") or ""),
            lease_epoch=int(raw.get("lease_epoch")),
            fencing_token_digest=_digest(
                raw.get("fencing_token_digest"), field="fencing_token_digest"
            ),
            code=ContinuationLeaseEventCodeV1(str(raw.get("code") or "")),
            event_at_utc=str(raw.get("event_at_utc") or ""),
            reason=str(raw.get("reason") or ""),
            prior_event_digest=_optional_digest(
                raw.get("prior_event_digest"), field="prior_event_digest"
            ),
            successor_lease_digest=_optional_digest(
                raw.get("successor_lease_digest"), field="successor_lease_digest"
            ),
            publication_digest=_optional_digest(
                raw.get("publication_digest"), field="publication_digest"
            ),
            owner_dead_proof_digest=_optional_digest(
                raw.get("owner_dead_proof_digest"),
                field="owner_dead_proof_digest",
            ),
            schema_version=(
                CONTINUATION_LEASE_EVENT_SCHEMA_VERSION_V2
                if raw.get("code") == "OWNER_DEAD"
                else CONTINUATION_LEASE_EVENT_SCHEMA_VERSION
            ),
        ),
        raw,
        label="ContinuationLeaseEventV1",
    )


def _start(raw: Mapping[str, Any]) -> AttemptStartV1:
    return _round_trip(
        AttemptStartV1(
            attempt_id=AttemptId(str(raw.get("attempt_id") or "")),
            run_id=RunId(str(raw.get("run_id") or "")),
            monotonic_ordinal=int(raw.get("monotonic_ordinal")),
            nonce=str(raw.get("nonce") or ""),
            concretized_execution_digest=_digest(
                raw.get("concretized_execution_digest"),
                field="concretized_execution_digest",
            ),
            implementation_digest=_digest(
                raw.get("implementation_digest"), field="implementation_digest"
            ),
            attempt_admission_request_digest=_digest(
                raw.get("attempt_admission_request_digest"),
                field="attempt_admission_request_digest",
            ),
            scheduler_assignment_lease_digest=_optional_digest(
                raw.get("scheduler_assignment_lease_digest"),
                field="scheduler_assignment_lease_digest",
            ),
            resource_start_lease_digest=_digest(
                raw.get("resource_start_lease_digest"),
                field="resource_start_lease_digest",
            ),
            post_admission_attempt_plan_digest=_digest(
                raw.get("post_admission_attempt_plan_digest"),
                field="post_admission_attempt_plan_digest",
            ),
            admission_disposition_digest=_digest(
                raw.get("admission_disposition_digest"),
                field="admission_disposition_digest",
            ),
            programme_work_session_assignment_dispatch_binding_digest=_digest(
                raw.get("programme_work_session_assignment_dispatch_binding_digest"),
                field="programme_work_session_assignment_dispatch_binding_digest",
            ),
            parent_attempt_id=(
                None
                if raw.get("parent_attempt_id") is None
                else AttemptId(str(raw.get("parent_attempt_id")))
            ),
            restore_checkpoint_digest=_optional_digest(
                raw.get("restore_checkpoint_digest"),
                field="restore_checkpoint_digest",
            ),
            canonical_continuation_lease_digest=_digest(
                raw.get("canonical_continuation_lease_digest"),
                field="canonical_continuation_lease_digest",
            ),
            owner_binding_digest=_digest(
                raw.get("owner_binding_digest"), field="owner_binding_digest"
            ),
            execution_side_effect_digest=_digest(
                raw.get("execution_side_effect_digest"),
                field="execution_side_effect_digest",
            ),
            initial_frontiers=_frontiers(dict(raw.get("initial_frontiers") or {})),
        ),
        raw,
        label="AttemptStartV1",
    )


def _terminal(raw: Mapping[str, Any]) -> AttemptTerminalV1:
    return _round_trip(
        AttemptTerminalV1(
            attempt_id=AttemptId(str(raw.get("attempt_id") or "")),
            attempt_start_digest=_digest(
                raw.get("attempt_start_digest"), field="attempt_start_digest"
            ),
            status=AttemptTerminalStatusV1(str(raw.get("status") or "")),
            final_frontiers=_frontiers(dict(raw.get("final_frontiers") or {})),
            result_digest=_digest(raw.get("result_digest"), field="result_digest"),
            reason=str(raw.get("reason") or ""),
        ),
        raw,
        label="AttemptTerminalV1",
    )


def _attempt_record(raw: object, *, label: str) -> AttemptRecordV1:
    if not isinstance(raw, Mapping):
        _fail(f"{label} must be an object")
    _exact(
        raw,
        {"start", "concretized_execution", "terminal"},
        label=label,
    )
    terminal = raw.get("terminal")
    return AttemptRecordV1(
        start=_start(dict(raw["start"])),
        concretized_execution=_concretized(dict(raw["concretized_execution"])),
        terminal=None if terminal is None else _terminal(dict(terminal)),
    )


def _restore(raw: Mapping[str, Any] | None) -> RestoreCheckpointAuthorizationV1 | None:
    if raw is None:
        return None
    return _round_trip(
        RestoreCheckpointAuthorizationV1(
            checkpoint_digest=_digest(
                raw.get("checkpoint_digest"), field="checkpoint_digest"
            ),
            run_id=RunId(str(raw.get("run_id") or "")),
            source_attempt_id=AttemptId(str(raw.get("source_attempt_id") or "")),
            source_attempt_terminal_digest=_digest(
                raw.get("source_attempt_terminal_digest"),
                field="source_attempt_terminal_digest",
            ),
            absolute_next_tick=int(raw.get("absolute_next_tick")),
            sealed_bundle_digest=_digest(
                raw.get("sealed_bundle_digest"), field="sealed_bundle_digest"
            ),
            completion_marker_digest=_digest(
                raw.get("completion_marker_digest"),
                field="completion_marker_digest",
            ),
        ),
        raw,
        label="RestoreCheckpointAuthorizationV1",
    )


def _conformance(
    raw: Mapping[str, Any] | None,
) -> ExactImplementationConformanceV1 | None:
    if raw is None:
        return None
    return _round_trip(
        ExactImplementationConformanceV1(
            run_id=RunId(str(raw.get("run_id") or "")),
            candidate_attempt_id=AttemptId(str(raw.get("candidate_attempt_id") or "")),
            prior_implementation_digest=_digest(
                raw.get("prior_implementation_digest"),
                field="prior_implementation_digest",
            ),
            candidate_implementation_digest=_digest(
                raw.get("candidate_implementation_digest"),
                field="candidate_implementation_digest",
            ),
            candidate_abi_digest=_digest(
                raw.get("candidate_abi_digest"), field="candidate_abi_digest"
            ),
            candidate_device_conformance_envelope_digest=_digest(
                raw.get("candidate_device_conformance_envelope_digest"),
                field="candidate_device_conformance_envelope_digest",
            ),
            exact_equivalence_evidence_digests=_named_digests(
                raw.get("exact_equivalence_evidence_digests"),
                field="exact_equivalence_evidence_digests",
            ),
            qualification_delta_digest=_digest(
                raw.get("qualification_delta_digest"),
                field="qualification_delta_digest",
            ),
        ),
        raw,
        label="ExactImplementationConformanceV1",
    )


@dataclass(frozen=True, slots=True)
class AuthoritativeAttemptBundleV1:
    _bundle_schema_version: ClassVar[str] = AUTHORITATIVE_BUNDLE_SCHEMA_VERSION
    _requires_restore_source_record: ClassVar[bool] = False

    raw: Mapping[str, Any]
    semantic_run_spec: SemanticRunSpecV1
    run_occurrence: RunOccurrenceV1
    execution_implementation: ExecutionImplementationV1
    lifecycle_binding: LifecycleBindingV1
    concretized_execution: ConcretizedExecutionV1
    attempt_admission_request: AttemptAdmissionRequestV1
    admission_disposition: AttemptAdmissionDispositionV1
    retry_basis: AttemptRetryBasisV1
    attempt_start: AttemptStartV1
    continuation_lease: CanonicalContinuationLeaseV1
    continuation_ledger: ContinuationLeaseLedgerV1
    prior_records: tuple[AttemptRecordV1, ...]
    restore_authorization: RestoreCheckpointAuthorizationV1 | None
    implementation_conformance: ExactImplementationConformanceV1 | None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AuthoritativeAttemptBundleV1":
        expected = {
            "schema_version",
            "authority_kind",
            "claim",
            "resource_start_lease",
            "resource_start_event",
            "continuation_lease",
            "continuation_lease_ledger",
            "execution_request",
            "campaign_plan",
            "preflight_receipt",
            "lifecycle_binding",
            "semantic_run_spec",
            "run_occurrence",
            "execution_implementation",
            "concretized_execution",
            "attempt_admission_request",
            "admission_disposition",
            "retry_basis",
            "attempt_start",
            "restore_authorization",
            "implementation_conformance",
            "prior_attempt_records",
            "start_effect",
            "issued_at_utc",
        }
        if cls._requires_restore_source_record:
            expected.add("restore_source_attempt_record")
        _exact(raw, expected, label=cls.__name__)
        if (
            raw.get("schema_version") != cls._bundle_schema_version
            or raw.get("authority_kind") != "SCHEDULER_ISSUED_LAUNCHPAD_ATTEMPT"
        ):
            _fail("bundle is not a Scheduler-issued Launchpad authority bundle")
        claim = dict(raw.get("claim") or {})
        preflight = dict(raw.get("preflight_receipt") or {})
        resource = dict(raw.get("resource_start_lease") or {})
        resource_event = dict(raw.get("resource_start_event") or {})
        start_effect = dict(raw.get("start_effect") or {})
        _exact(
            claim,
            {
                "schema_version",
                "claim_id",
                "scheduler_owner_id",
                "scheduler_owner_digest",
                "execution_owner_id",
                "execution_owner_digest",
                "run_id",
                "attempt_id",
                "lineage_scope_digest",
                "execution_request_digest",
                "campaign_plan_digest",
                "source_digest",
                "baseline_digest",
                "preflight_receipt_digest",
                "lifecycle_binding_digest",
                "lifecycle_contract_digest",
                "run_root",
                "provider_id",
                "route_id",
                "device_id",
                "native_artifact_digest",
                "batch_size",
                "ticks_per_dispatch",
                "execution_mode",
                "semantic_run_spec_digest",
                "run_occurrence_digest",
                "execution_implementation_digest",
                "concretized_execution_digest",
                "attempt_admission_request_digest",
                "retry_basis_digest",
                "post_admission_attempt_plan_digest",
                "programme_binding_digest",
                "initial_frontiers_digest",
                "created_at_utc",
            },
            label="LaunchpadAttemptClaimV1",
        )
        _exact(
            resource,
            {
                "schema_version",
                "lease_id",
                "scheduler_owner_id",
                "scheduler_owner_digest",
                "execution_claim_digest",
                "run_id",
                "attempt_id",
                "attempt_admission_request_digest",
                "resource_ids",
                "resource_scope_digest",
                "resource_observation_id",
                "resource_observation_digest",
                "resource_recommendation_id",
                "resource_recommendation_digest",
                "live_provider_check_digest",
                "lease_epoch",
                "fencing_token_digest",
                "prior_lease_digest",
                "issued_at_utc",
                "start_before_utc",
                "expires_at_utc",
            },
            label="LaunchpadResourceStartLeaseV1",
        )
        _exact(
            resource_event,
            {
                "schema_version",
                "lease_digest",
                "code",
                "scheduler_owner_id",
                "fencing_token_digest",
                "event_at_utc",
                "reason",
                "prior_event_digest",
                "successor_lease_digest",
            },
            label="ResourceStartLeaseEventV1",
        )
        start_effect_schema = start_effect.get("schema_version")
        _exact(
            start_effect,
            (
                _START_EFFECT_V2_FIELDS
                if start_effect_schema
                == "crca_launchpad_attempt_start_effect_v2"
                else _START_EFFECT_FIELDS
            ),
            label="LaunchpadAttemptStartEffectV1",
        )
        if (
            claim.get("schema_version") != "crca_launchpad_attempt_claim_v1"
            or resource.get("schema_version")
            != "crca_launchpad_resource_start_lease_v1"
            or resource_event.get("schema_version")
            != "crca_resource_start_lease_event_v1"
            or resource_event.get("code") != "STARTED"
            or start_effect_schema
            not in {
                "crca_launchpad_attempt_start_effect_v1",
                "crca_launchpad_attempt_start_effect_v2",
            }
            or start_effect.get("effect_kind")
            != "SCHEDULER_EXECUTION_AUTHORITY_GRANTED"
        ):
            _fail("bundle has a thin or foreign claim, lease, event, or effect")
        shared_lane_start = any(
            isinstance(value, str) and "/lane/" in value
            for value in resource.get("resource_ids") or []
        )
        if (
            shared_lane_start
            and start_effect_schema != "crca_launchpad_attempt_start_effect_v2"
        ):
            _fail("shared GPU lane lacks recoverable process identity")
        _exact(
            preflight,
            {
                "schema_version",
                "provider_id",
                "route_id",
                "device_id",
                "native_artifact_digest",
                "batch_size",
                "ticks_per_dispatch",
                "execution_mode",
            },
            label="Launchpad preflight projection",
        )
        if preflight.get("schema_version") != (
            "crca_launchpad_preflight_authority_projection_v1"
        ):
            _fail("Launchpad preflight projection schema is unsupported")
        claim_digest = _launchpad_claim_v1(claim)
        resource_digest, resource_issued, resource_start_before, _resource_expires = (
            _launchpad_resource_lease_v1(
                resource,
                claim=claim,
                claim_digest=claim_digest,
                campaign_plan=dict(raw["campaign_plan"]),
            )
        )
        resource_started_at = _resource_started_event_v1(
            resource_event,
            claim=claim,
            lease=resource,
            lease_digest=resource_digest,
            issued=resource_issued,
            start_before=resource_start_before,
        )
        continuation = _continuation(dict(raw["continuation_lease"]))
        effect_digest = _launchpad_start_effect_v1(
            start_effect,
            claim=claim,
            claim_digest=claim_digest,
            lease_digest=resource_digest,
            continuation_digest=continuation.lease_digest(),
            continuation_fence=dict(raw["continuation_lease"]),
            started_at=resource_started_at,
        )
        if (
            _digest(
                resource.get("execution_claim_digest"), field="execution_claim_digest"
            )
            != claim_digest
            or resource.get("run_id") != claim.get("run_id")
            or resource.get("attempt_id") != claim.get("attempt_id")
            or resource.get("scheduler_owner_id") != claim.get("scheduler_owner_id")
            or resource.get("scheduler_owner_digest")
            != claim.get("scheduler_owner_digest")
            or resource.get("attempt_admission_request_digest")
            != claim.get("attempt_admission_request_digest")
            or _digest(resource_event.get("lease_digest"), field="lease_digest")
            != resource_digest
            or resource_event.get("scheduler_owner_id")
            != claim.get("scheduler_owner_id")
            or resource_event.get("fencing_token_digest")
            != resource.get("fencing_token_digest")
            or resource_event.get("prior_event_digest") is not None
            or resource_event.get("successor_lease_digest") is not None
            or _digest(start_effect.get("claim_digest"), field="claim_digest")
            != claim_digest
            or _digest(
                start_effect.get("resource_start_lease_digest"),
                field="resource_start_lease_digest",
            )
            != resource_digest
            or start_effect.get("run_id") != claim.get("run_id")
            or start_effect.get("attempt_id") != claim.get("attempt_id")
            or start_effect.get("scheduler_owner_id") != claim.get("scheduler_owner_id")
            or start_effect.get("scheduler_owner_digest")
            != claim.get("scheduler_owner_digest")
            or start_effect.get("process_owner_id") != claim.get("execution_owner_id")
            or start_effect.get("process_owner_digest")
            != claim.get("execution_owner_digest")
            or start_effect.get("fencing_token_digest")
            != dict(raw.get("continuation_lease") or {}).get("fencing_token_digest")
            or preflight.get("provider_id") != claim.get("provider_id")
            or preflight.get("route_id") != claim.get("route_id")
            or preflight.get("device_id") != claim.get("device_id")
            or preflight.get("batch_size") != claim.get("batch_size")
            or preflight.get("ticks_per_dispatch") != claim.get("ticks_per_dispatch")
            or preflight.get("execution_mode") != claim.get("execution_mode")
            or preflight.get("native_artifact_digest")
            != claim.get("native_artifact_digest")
        ):
            _fail("bundle owner, claim, resource lease, or start effect differs")
        semantic = _semantic(dict(raw["semantic_run_spec"]))
        run = _run(dict(raw["run_occurrence"]))
        implementation = _implementation(dict(raw["execution_implementation"]))
        lifecycle = _lifecycle(dict(raw["lifecycle_binding"]))
        concretized = _concretized(dict(raw["concretized_execution"]))
        admission_request = _admission_request(dict(raw["attempt_admission_request"]))
        disposition = _disposition(dict(raw["admission_disposition"]))
        retry = _retry(dict(raw["retry_basis"]))
        start = _start(dict(raw["attempt_start"]))
        ledger_raw = dict(raw.get("continuation_lease_ledger") or {})
        _exact(ledger_raw, {"leases", "events"}, label="continuation lease ledger")
        leases = tuple(_continuation(dict(row)) for row in ledger_raw["leases"])
        events = tuple(_continuation_event(dict(row)) for row in ledger_raw["events"])
        ledger = ContinuationLeaseLedgerV1(leases=leases, events=events)
        prior_rows = raw.get("prior_attempt_records")
        if not isinstance(prior_rows, list):
            _fail("prior_attempt_records must be an array")
        prior: list[AttemptRecordV1] = []
        for row in prior_rows:
            prior.append(_attempt_record(row, label="prior Attempt record"))
        restore_source_record = (
            _attempt_record(
                raw.get("restore_source_attempt_record"),
                label="external restore source Attempt record",
            )
            if cls._requires_restore_source_record
            else None
        )
        if (
            cls._requires_restore_source_record
            and restore_source_record is not None
            and restore_source_record.terminal is None
        ):
            _fail("external restore source Attempt lacks a terminal record")
        restore = _restore(
            None
            if raw.get("restore_authorization") is None
            else dict(raw["restore_authorization"])
        )
        conformance = _conformance(
            None
            if raw.get("implementation_conformance") is None
            else dict(raw["implementation_conformance"])
        )
        if (
            canonical_sha256(dict(raw["execution_request"]))
            != _digest(
                claim.get("execution_request_digest"), field="execution_request_digest"
            )
            or canonical_sha256(dict(raw["campaign_plan"]))
            != _digest(claim.get("campaign_plan_digest"), field="campaign_plan_digest")
            or canonical_sha256(dict(raw["preflight_receipt"]))
            != _digest(
                claim.get("preflight_receipt_digest"), field="preflight_receipt_digest"
            )
            or lifecycle.lifecycle_digest()
            != _digest(
                claim.get("lifecycle_binding_digest"), field="lifecycle_binding_digest"
            )
            or lifecycle.artifact_lifecycle_contract_digest
            != _digest(
                claim.get("lifecycle_contract_digest"),
                field="lifecycle_contract_digest",
            )
            or lifecycle.output_root != claim.get("run_root")
            or semantic.semantic_spec_digest()
            != _digest(
                claim.get("semantic_run_spec_digest"), field="semantic_run_spec_digest"
            )
            or canonical_sha256(run.to_dict())
            != _digest(
                claim.get("run_occurrence_digest"), field="run_occurrence_digest"
            )
            or implementation.implementation_digest()
            != _digest(
                claim.get("execution_implementation_digest"),
                field="execution_implementation_digest",
            )
            or implementation.source_tree_digest
            != _digest(claim.get("source_digest"), field="source_digest")
            or concretized.concretized_execution_digest()
            != _digest(
                claim.get("concretized_execution_digest"),
                field="concretized_execution_digest",
            )
            or concretized.semantic_spec_digest
            != _digest(
                claim.get("semantic_run_spec_digest"), field="semantic_run_spec_digest"
            )
            or concretized.implementation_digest
            != _digest(
                claim.get("execution_implementation_digest"),
                field="execution_implementation_digest",
            )
            or concretized.lifecycle_digest
            != _digest(
                claim.get("lifecycle_binding_digest"), field="lifecycle_binding_digest"
            )
            or concretized.attempt_admission_request_digest
            != _digest(
                claim.get("attempt_admission_request_digest"),
                field="attempt_admission_request_digest",
            )
            or concretized.campaign_plan_digest
            != _digest(claim.get("campaign_plan_digest"), field="campaign_plan_digest")
            or concretized.bound_execution_request_digest
            != _digest(
                claim.get("execution_request_digest"), field="execution_request_digest"
            )
            or concretized.baseline_digest
            != _digest(claim.get("baseline_digest"), field="baseline_digest")
            or admission_request.attempt_admission_request_digest()
            != _digest(
                claim.get("attempt_admission_request_digest"),
                field="attempt_admission_request_digest",
            )
            or retry.basis_digest()
            != _digest(claim.get("retry_basis_digest"), field="retry_basis_digest")
            or implementation.provider_id != claim.get("provider_id")
        ):
            _fail(
                "bundle differs from exact request, plan, source, lifecycle, or CRCA projection"
            )
        artifact_digests = {
            item.digest for item in implementation.native_artifact_digests
        }
        if (
            _digest(claim.get("native_artifact_digest"), field="native_artifact_digest")
            not in artifact_digests
        ):
            _fail("selected native artifact is not bound by implementation")
        owner_binding = canonical_sha256(
            {
                "schema_version": "crca_launchpad_attempt_owner_binding_v1",
                "claim_digest": claim_digest.to_dict(),
                "scheduler_owner_id": claim["scheduler_owner_id"],
                "scheduler_owner_digest": claim["scheduler_owner_digest"],
                "execution_owner_id": claim["execution_owner_id"],
                "execution_owner_digest": claim["execution_owner_digest"],
                "resource_start_lease_digest": resource_digest.to_dict(),
                "continuation_lease_digest": continuation.lease_digest().to_dict(),
            }
        )
        if (
            start.owner_binding_digest != owner_binding
            or start.execution_side_effect_digest != effect_digest
            or start.resource_start_lease_digest != resource_digest
            or start.canonical_continuation_lease_digest != continuation.lease_digest()
            or start.post_admission_attempt_plan_digest
            != _digest(
                claim.get("post_admission_attempt_plan_digest"),
                field="post_admission_attempt_plan_digest",
            )
            or start.programme_work_session_assignment_dispatch_binding_digest
            != _digest(
                claim.get("programme_binding_digest"),
                field="programme_binding_digest",
            )
            or canonical_sha256(start.initial_frontiers.to_dict())
            != _digest(
                claim.get("initial_frontiers_digest"),
                field="initial_frontiers_digest",
            )
            or disposition.resource_start_lease_digest != resource_digest
            or _digest(
                start_effect.get("continuation_lease_digest"),
                field="continuation_lease_digest",
            )
            != continuation.lease_digest()
            or resource_event.get("event_at_utc") != raw.get("issued_at_utc")
            or start_effect.get("started_at_utc") != raw.get("issued_at_utc")
        ):
            _fail("AttemptStart differs from Scheduler owner, lease, or effect")
        validation = validate_attempt_start_v1(
            run=run,
            semantic_spec=semantic,
            concretized_execution=concretized,
            implementation=implementation,
            admission_request=admission_request,
            disposition=disposition,
            retry_basis=retry,
            start=start,
            continuation_lease=continuation,
            continuation_lease_ledger=ledger,
            execution_side_effect_at_utc=str(raw.get("issued_at_utc") or ""),
            prior_records=tuple(prior),
            restore_authorization=restore,
            restore_source_record=restore_source_record,
            implementation_conformance=conformance,
        )
        if validation.get("execution_authority_granted") is not False:
            _fail("pure CRCA validator unexpectedly granted execution authority")
        values: dict[str, Any] = {
            "raw": dict(raw),
            "semantic_run_spec": semantic,
            "run_occurrence": run,
            "execution_implementation": implementation,
            "lifecycle_binding": lifecycle,
            "concretized_execution": concretized,
            "attempt_admission_request": admission_request,
            "admission_disposition": disposition,
            "retry_basis": retry,
            "attempt_start": start,
            "continuation_lease": continuation,
            "continuation_ledger": ledger,
            "prior_records": tuple(prior),
            "restore_authorization": restore,
            "implementation_conformance": conformance,
        }
        if cls._requires_restore_source_record:
            values["restore_source_record"] = restore_source_record
        return cls(**values)

    @classmethod
    def from_path(cls, path: str | Path) -> "AuthoritativeAttemptBundleV1":
        target = Path(path).expanduser()
        stat = os.lstat(target)
        if target.is_symlink() or not target.is_file() or stat.st_nlink != 1:
            _fail("authoritative bundle must be one regular non-linked file")
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise LaunchpadAttemptAuthorityError(
                "authoritative bundle is unreadable"
            ) from error
        if not isinstance(raw, Mapping):
            _fail("authoritative bundle must be an object")
        digest = canonical_sha256(dict(raw))
        if target.name != f"{digest.hex}.json":
            _fail("authoritative bundle filename differs from content digest")
        return cls.from_dict(raw)

    def validate(self) -> dict[str, Any]:
        """Return a pure validation receipt; never grant execution authority."""

        claim = dict(self.raw["claim"])
        return {
            "schema_version": (
                "crca_authoritative_launchpad_attempt_validation_v2"
                if self._requires_restore_source_record
                else "crca_authoritative_launchpad_attempt_validation_v1"
            ),
            "status": "AUTHORITATIVE_BUNDLE_VALID",
            "authority_source": "SCHEDULER_ISSUED_BUNDLE",
            "run_id": str(self.run_occurrence.run_id),
            "attempt_id": str(self.attempt_start.attempt_id),
            "provider_id": claim["provider_id"],
            "route_id": claim["route_id"],
            "device_id": claim["device_id"],
            "batch_size": claim["batch_size"],
            "ticks_per_dispatch": claim["ticks_per_dispatch"],
            "execution_mode": claim["execution_mode"],
            "bundle_digest": canonical_sha256(dict(self.raw)).to_dict(),
            "execution_authority_granted": False,
        }


@dataclass(frozen=True, slots=True)
class AuthoritativeAttemptBundleV2(AuthoritativeAttemptBundleV1):
    """Additive child-Run bundle carrying one external source Attempt record."""

    _bundle_schema_version: ClassVar[str] = AUTHORITATIVE_BUNDLE_SCHEMA_VERSION_V2
    _requires_restore_source_record: ClassVar[bool] = True

    restore_source_record: AttemptRecordV1


def authoritative_attempt_bundle_from_dict(
    raw: Mapping[str, Any],
) -> AuthoritativeAttemptBundleV1 | AuthoritativeAttemptBundleV2:
    schema = raw.get("schema_version")
    if schema == AUTHORITATIVE_BUNDLE_SCHEMA_VERSION:
        return AuthoritativeAttemptBundleV1.from_dict(raw)
    if schema == AUTHORITATIVE_BUNDLE_SCHEMA_VERSION_V2:
        return AuthoritativeAttemptBundleV2.from_dict(raw)
    _fail("authoritative bundle schema is unsupported")


def authoritative_attempt_bundle_from_path(
    path: str | Path,
) -> AuthoritativeAttemptBundleV1 | AuthoritativeAttemptBundleV2:
    target = Path(path).expanduser()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaunchpadAttemptAuthorityError(
            "authoritative bundle is unreadable"
        ) from error
    if not isinstance(raw, Mapping):
        _fail("authoritative bundle must be an object")
    schema = raw.get("schema_version")
    if schema == AUTHORITATIVE_BUNDLE_SCHEMA_VERSION:
        return AuthoritativeAttemptBundleV1.from_path(target)
    if schema == AUTHORITATIVE_BUNDLE_SCHEMA_VERSION_V2:
        return AuthoritativeAttemptBundleV2.from_path(target)
    _fail("authoritative bundle schema is unsupported")


__all__ = [
    "AUTHORITATIVE_BUNDLE_SCHEMA_VERSION",
    "AUTHORITATIVE_BUNDLE_SCHEMA_VERSION_V2",
    "AuthoritativeAttemptBundleV1",
    "AuthoritativeAttemptBundleV2",
    "LaunchpadAttemptAuthorityError",
    "authoritative_attempt_bundle_from_dict",
    "authoritative_attempt_bundle_from_path",
]
