from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from eiel_composition.attempt_admission import (
    AttemptAdmissionError,
    AttemptRecordV1,
    ContinuationLeaseLedgerV1,
    append_attempt_terminal_receipt_v1,
    append_occurrence_receipt_v1,
    commit_canonical_publication_v1,
    non_runtime_occurrence_decision_v1,
    require_canonical_publish_authority_v1,
    validate_attempt_start_v1,
    validate_frontier_progression_v1,
    validate_terminal_once_v1,
)
from emergence_contracts import (
    CONTINUATION_LEASE_EVENT_SCHEMA_VERSION,
    CONTINUATION_LEASE_EVENT_SCHEMA_VERSION_V2,
    AdmissionAssuranceBindingV1,
    AttemptAdmissionDispositionCodeV1,
    AttemptAdmissionDispositionV1,
    AttemptAdmissionRequestV1,
    AttemptFrontiersV1,
    AttemptRetryBasisCodeV1,
    AttemptRetryBasisV1,
    AttemptStartV1,
    AttemptTerminalStatusV1,
    AttemptTerminalV1,
    BackendId,
    CanonicalContinuationLeaseV1,
    ConcretizedExecutionV1,
    ContentDigest,
    ContractError,
    ContinuationLeaseEventCodeV1,
    ContinuationLeaseEventV1,
    ExecutionImplementationV1,
    ExactImplementationConformanceV1,
    ExperimentId,
    LeaseId,
    LifecycleBindingV1,
    NamedDigestV1,
    RestoreCheckpointAuthorizationV1,
    RunOccurrenceV1,
    SemanticRunSpecV1,
    canonical_sha256,
    derive_attempt_id,
)


def _digest(label: str) -> ContentDigest:
    return canonical_sha256({"label": label})


def _named(name: str, label: str | None = None) -> NamedDigestV1:
    return NamedDigestV1(name=name, digest=_digest(label or name))


def _semantic(label: str = "a", *, start: int = 0) -> SemanticRunSpecV1:
    return SemanticRunSpecV1(
        experiment_id=ExperimentId("experiment-g3"),
        system_assembly_digest=_digest("assembly"),
        semantic_parameters_digest=_digest(f"parameters-{label}"),
        rng_contract_digest=_digest("rng"),
        seed_digest=_digest("seed"),
        absolute_start_next_tick=start,
        absolute_target_next_tick=1000,
        trajectory_checkpoint_ticks=(start, 500, 1000),
        measurement_behavior_digest=_digest("measurement"),
        control_and_branch_semantics_digest=_digest(f"control-{label}"),
    )


def _implementation(label: str = "a") -> ExecutionImplementationV1:
    commit_character = "1" if label == "a" else "2"
    return ExecutionImplementationV1(
        source_commit=commit_character * 40,
        source_tree_digest=_digest(f"tree-{label}"),
        dirty_patch_digest=None,
        toolchain_digest=_digest("toolchain"),
        environment_digest=_digest(f"environment-{label}"),
        provider_id="provider-v1",
        backend_id=BackendId("resident-cuda"),
        runtime_profile_id="profile-v1",
        native_artifact_digests=(_named("binary", f"binary-{label}"),),
        abi_digest=_digest("abi"),
        device_conformance_envelope_digest=_digest("device-envelope"),
    )


def _admission(
    label: str = "a", *, restore: ContentDigest | None = None
) -> AttemptAdmissionRequestV1:
    return AttemptAdmissionRequestV1(
        resource_requirements_digest=_digest(f"resources-{label}"),
        executor_constraints_digest=_digest(f"executor-{label}"),
        requested_restore_checkpoint_digest=restore,
        observational_persistence_cadence_digest=_digest("cadence"),
        proposed_reason=f"admit-{label}",
    )


def _concretized(
    semantic: SemanticRunSpecV1,
    implementation: ExecutionImplementationV1,
    admission: AttemptAdmissionRequestV1,
) -> ConcretizedExecutionV1:
    assurance = AdmissionAssuranceBindingV1(
        intended_claim_id="technical-only",
        qualified_envelope_digest=None,
        obligation_digests=(),
        authority_reference_digests=(),
        review_reference_digests=(),
    )
    lifecycle = LifecycleBindingV1(
        output_root="/runtime/fresh",
        artifact_lifecycle_contract_digest=_digest("lifecycle"),
        inventory_role_digests=(),
        initial_hold_digests=(),
        retention_profile_id="test-retention",
    )
    return ConcretizedExecutionV1(
        semantic_spec_digest=semantic.semantic_spec_digest(),
        implementation_digest=implementation.implementation_digest(),
        admission_assurance_digest=assurance.admission_assurance_digest(),
        lifecycle_digest=lifecycle.lifecycle_digest(),
        attempt_admission_request_digest=admission.attempt_admission_request_digest(),
        campaign_plan_digest=_digest("plan"),
        bound_execution_request_digest=_digest("bound"),
        run_contract_digest=_digest("run-contract"),
        baseline_digest=_digest("baseline"),
        route_contract_digest=_digest("route"),
        assembly_digest=semantic.system_assembly_digest,
        qualification_key_digest=_digest("qualification-key"),
    )


def _run(
    semantic: SemanticRunSpecV1,
    *,
    ordinal: int = 0,
    parent: ContentDigest | None = None,
    branch: ContentDigest | None = None,
) -> RunOccurrenceV1:
    return RunOccurrenceV1.create(
        experiment_id=semantic.experiment_id,
        cell_or_repeat_ordinal=ordinal,
        semantic_spec_digest=semantic.semantic_spec_digest(),
        parent_checkpoint_digest=parent,
        branch_spec_digest=branch,
    )


def _disposition(
    run: RunOccurrenceV1,
    admission: AttemptAdmissionRequestV1,
    *,
    start_lease: str,
) -> AttemptAdmissionDispositionV1:
    return AttemptAdmissionDispositionV1(
        run_id=run.run_id,
        attempt_admission_request_digest=admission.attempt_admission_request_digest(),
        code=AttemptAdmissionDispositionCodeV1.START_AUTHORIZED,
        reason="resource and provider ready",
        scheduler_assignment_lease_digest=None,
        resource_start_lease_digest=_digest(start_lease),
    )


def _lease(
    run: RunOccurrenceV1,
    attempt_id: object,
    *,
    label: str,
    epoch: int = 1,
    restore: ContentDigest | None = None,
    prior: ContentDigest | None = None,
    granted: str = "2026-08-23T04:00:00Z",
    expires: str = "2026-08-23T04:05:00Z",
    owner: str = "owner-a",
) -> CanonicalContinuationLeaseV1:
    return CanonicalContinuationLeaseV1(
        lease_id=LeaseId(f"lease-{label}"),
        run_id=run.run_id,
        attempt_id=attempt_id,
        restore_checkpoint_digest=restore,
        lineage_scope_digest=_digest("lineage-scope"),
        owner_id=owner,
        lease_epoch=epoch,
        fencing_token_digest=_digest(f"fence-{label}"),
        prior_lease_digest=prior,
        granted_at_utc=granted,
        expires_at_utc=expires,
    )


def _basis(
    *,
    run: RunOccurrenceV1,
    candidate_id: object,
    candidate: ConcretizedExecutionV1,
    code: AttemptRetryBasisCodeV1,
    prior: AttemptRecordV1 | None = None,
    evidence_name: str | None = None,
    conformance: ContentDigest | None = None,
) -> AttemptRetryBasisV1:
    return AttemptRetryBasisV1(
        run_id=run.run_id,
        candidate_attempt_id=candidate_id,
        prior_attempt_id=None if prior is None else prior.start.attempt_id,
        code=code,
        reason=f"basis-{code.value.lower()}",
        semantic_spec_digest=run.semantic_spec_digest,
        prior_concretized_execution_digest=(
            None
            if prior is None
            else prior.concretized_execution.concretized_execution_digest()
        ),
        candidate_concretized_execution_digest=candidate.concretized_execution_digest(),
        basis_evidence_digests=(
            () if evidence_name is None else (_named(evidence_name),)
        ),
        implementation_conformance_digest=conformance,
    )


def _start(
    *,
    run: RunOccurrenceV1,
    ordinal: int,
    nonce: str,
    concretized: ConcretizedExecutionV1,
    implementation: ExecutionImplementationV1,
    admission: AttemptAdmissionRequestV1,
    disposition: AttemptAdmissionDispositionV1,
    lease: CanonicalContinuationLeaseV1,
    parent: object | None = None,
    restore: ContentDigest | None = None,
    frontiers: AttemptFrontiersV1 | None = None,
) -> AttemptStartV1:
    attempt_id = derive_attempt_id(
        run_id=run.run_id, monotonic_ordinal=ordinal, nonce=nonce
    )
    assert attempt_id == lease.attempt_id
    return AttemptStartV1(
        attempt_id=attempt_id,
        run_id=run.run_id,
        monotonic_ordinal=ordinal,
        nonce=nonce,
        concretized_execution_digest=concretized.concretized_execution_digest(),
        implementation_digest=implementation.implementation_digest(),
        attempt_admission_request_digest=admission.attempt_admission_request_digest(),
        scheduler_assignment_lease_digest=disposition.scheduler_assignment_lease_digest,
        resource_start_lease_digest=disposition.resource_start_lease_digest,
        post_admission_attempt_plan_digest=_digest(f"post-plan-{ordinal}"),
        admission_disposition_digest=disposition.disposition_digest(),
        programme_work_session_assignment_dispatch_binding_digest=_digest(
            "control-plane"
        ),
        parent_attempt_id=parent,
        restore_checkpoint_digest=restore,
        canonical_continuation_lease_digest=lease.lease_digest(),
        owner_binding_digest=_digest(f"owner-{ordinal}"),
        execution_side_effect_digest=_digest(f"side-effect-{ordinal}"),
        initial_frontiers=frontiers or AttemptFrontiersV1(0, 0, 0, 0),
    )


def _initial_bundle(
    semantic: SemanticRunSpecV1 | None = None,
) -> tuple[
    SemanticRunSpecV1,
    RunOccurrenceV1,
    ExecutionImplementationV1,
    AttemptAdmissionRequestV1,
    ConcretizedExecutionV1,
    AttemptAdmissionDispositionV1,
    CanonicalContinuationLeaseV1,
    AttemptRetryBasisV1,
    AttemptStartV1,
]:
    semantic = semantic or _semantic()
    run = _run(semantic)
    implementation = _implementation()
    admission = _admission()
    concretized = _concretized(semantic, implementation, admission)
    attempt_id = derive_attempt_id(
        run_id=run.run_id, monotonic_ordinal=1, nonce="initial"
    )
    disposition = _disposition(run, admission, start_lease="start-1")
    lease = _lease(run, attempt_id, label="1")
    basis = _basis(
        run=run,
        candidate_id=attempt_id,
        candidate=concretized,
        code=AttemptRetryBasisCodeV1.INITIAL_EXECUTION,
    )
    start = _start(
        run=run,
        ordinal=1,
        nonce="initial",
        concretized=concretized,
        implementation=implementation,
        admission=admission,
        disposition=disposition,
        lease=lease,
        frontiers=AttemptFrontiersV1(
            semantic.absolute_start_next_tick,
            semantic.absolute_start_next_tick,
            semantic.absolute_start_next_tick,
            semantic.absolute_start_next_tick,
        ),
    )
    return (
        semantic,
        run,
        implementation,
        admission,
        concretized,
        disposition,
        lease,
        basis,
        start,
    )


def _terminal(
    start: AttemptStartV1,
    *,
    status: AttemptTerminalStatusV1,
    frontiers: AttemptFrontiersV1,
    label: str,
) -> AttemptTerminalV1:
    return AttemptTerminalV1(
        attempt_id=start.attempt_id,
        attempt_start_digest=start.start_digest(),
        status=status,
        final_frontiers=frontiers,
        result_digest=_digest(label),
        reason=label,
    )


def _event(
    lease: CanonicalContinuationLeaseV1,
    *,
    code: ContinuationLeaseEventCodeV1,
    at: str,
    prior_event: ContinuationLeaseEventV1 | None = None,
    successor: CanonicalContinuationLeaseV1 | None = None,
    publication: ContentDigest | None = None,
    owner_dead_proof: ContentDigest | None = None,
) -> ContinuationLeaseEventV1:
    return ContinuationLeaseEventV1(
        lease_digest=lease.lease_digest(),
        run_id=lease.run_id,
        attempt_id=lease.attempt_id,
        owner_id=lease.owner_id,
        lease_epoch=lease.lease_epoch,
        fencing_token_digest=lease.fencing_token_digest,
        code=code,
        event_at_utc=at,
        reason=f"event-{code.value.lower()}",
        prior_event_digest=(
            None if prior_event is None else prior_event.event_digest()
        ),
        successor_lease_digest=(
            None if successor is None else successor.lease_digest()
        ),
        publication_digest=publication,
        owner_dead_proof_digest=owner_dead_proof,
        schema_version=(
            CONTINUATION_LEASE_EVENT_SCHEMA_VERSION_V2
            if owner_dead_proof is not None
            else CONTINUATION_LEASE_EVENT_SCHEMA_VERSION
        ),
    )


def _reclaimed_ledger(
    predecessor: CanonicalContinuationLeaseV1,
    successor: CanonicalContinuationLeaseV1,
) -> ContinuationLeaseLedgerV1:
    reclaim = _event(
        predecessor,
        code=ContinuationLeaseEventCodeV1.RECLAIMED,
        at=successor.granted_at_utc,
        successor=successor,
    )
    return ContinuationLeaseLedgerV1.initial(predecessor).transition(reclaim, successor)


def test_prelaunch_wait_is_not_a_runtime_attempt() -> None:
    _, run, _, admission, *_ = _initial_bundle()
    wait = AttemptAdmissionDispositionV1(
        run_id=run.run_id,
        attempt_admission_request_digest=admission.attempt_admission_request_digest(),
        code=AttemptAdmissionDispositionCodeV1.WAIT_RESOURCE,
        reason="GPU temporarily busy",
        scheduler_assignment_lease_digest=None,
        resource_start_lease_digest=None,
    )

    assert "attempt_id" not in wait.to_dict()


def test_initial_execution_requires_exact_leases_and_side_effect_window() -> None:
    (
        _,
        run,
        implementation,
        admission,
        concretized,
        disposition,
        lease,
        basis,
        start,
    ) = _initial_bundle()

    result = validate_attempt_start_v1(
        run=run,
        semantic_spec=_semantic(),
        concretized_execution=concretized,
        implementation=implementation,
        admission_request=admission,
        disposition=disposition,
        retry_basis=basis,
        start=start,
        continuation_lease=lease,
        continuation_lease_ledger=ContinuationLeaseLedgerV1.initial(lease),
        execution_side_effect_at_utc="2026-08-23T04:01:00Z",
    )

    assert result["status"] == "START_VALID"
    assert result["execution_authority_granted"] is False


def test_initial_execution_accepts_semantic_nonzero_absolute_start() -> None:
    semantic = _semantic(start=71)
    (
        _,
        run,
        implementation,
        admission,
        concretized,
        disposition,
        lease,
        basis,
        start,
    ) = _initial_bundle(semantic)

    result = validate_attempt_start_v1(
        run=run,
        semantic_spec=semantic,
        concretized_execution=concretized,
        implementation=implementation,
        admission_request=admission,
        disposition=disposition,
        retry_basis=basis,
        start=start,
        continuation_lease=lease,
        continuation_lease_ledger=ContinuationLeaseLedgerV1.initial(lease),
        execution_side_effect_at_utc="2026-08-23T04:01:00Z",
    )

    assert result["status"] == "START_VALID"
    assert start.initial_frontiers == AttemptFrontiersV1(71, 71, 71, 71)


def test_nonrestored_frontiers_must_equal_semantic_absolute_start() -> None:
    (
        semantic,
        run,
        implementation,
        admission,
        concretized,
        disposition,
        lease,
        basis,
        start,
    ) = _initial_bundle()
    mismatched = replace(start, initial_frontiers=AttemptFrontiersV1(1, 1, 1, 1))

    with pytest.raises(AttemptAdmissionError, match="semantic absolute start"):
        validate_attempt_start_v1(
            run=run,
            semantic_spec=semantic,
            concretized_execution=concretized,
            implementation=implementation,
            admission_request=admission,
            disposition=disposition,
            retry_basis=basis,
            start=mismatched,
            continuation_lease=lease,
            continuation_lease_ledger=ContinuationLeaseLedgerV1.initial(lease),
            execution_side_effect_at_utc="2026-08-23T04:01:00Z",
        )


def test_retry_basis_requires_both_prior_attempt_bindings() -> None:
    *_, basis, start = _initial_bundle()
    with pytest.raises(ContractError, match="prior Attempt bindings"):
        replace(
            basis,
            code=AttemptRetryBasisCodeV1.NEW_INFORMATION,
            prior_attempt_id=start.attempt_id,
            basis_evidence_digests=(_named("new_information"),),
        )


def test_semantic_change_is_never_admitted_as_retry() -> None:
    (
        _,
        run,
        implementation,
        admission,
        concretized,
        disposition,
        lease,
        basis,
        start,
    ) = _initial_bundle()
    changed = replace(
        concretized, semantic_spec_digest=_semantic("changed").semantic_spec_digest()
    )

    with pytest.raises(AttemptAdmissionError, match="semantic change"):
        validate_attempt_start_v1(
            run=run,
            semantic_spec=_semantic(),
            concretized_execution=changed,
            implementation=implementation,
            admission_request=admission,
            disposition=disposition,
            retry_basis=basis,
            start=start,
            continuation_lease=lease,
            continuation_lease_ledger=ContinuationLeaseLedgerV1.initial(lease),
            execution_side_effect_at_utc="2026-08-23T04:01:00Z",
        )


def test_equivalent_expensive_rerun_requires_typed_justification() -> None:
    semantic, run, implementation, admission, concretized, _, lease1, _, start1 = (
        _initial_bundle()
    )
    terminal1 = _terminal(
        start1,
        status=AttemptTerminalStatusV1.COMPLETED,
        frontiers=AttemptFrontiersV1(1000, 1000, 1000, 1000),
        label="completed",
    )
    prior = AttemptRecordV1(start1, concretized, terminal1)
    attempt2 = derive_attempt_id(run_id=run.run_id, monotonic_ordinal=2, nonce="rerun")
    disposition2 = _disposition(run, admission, start_lease="start-2")
    lease2 = _lease(
        run,
        attempt2,
        label="2",
        epoch=2,
        prior=lease1.lease_digest(),
        granted="2026-08-23T04:03:00Z",
        expires="2026-08-23T04:08:00Z",
    )
    released = _event(
        lease1,
        code=ContinuationLeaseEventCodeV1.RELEASED,
        at="2026-08-23T04:02:00Z",
    )
    reassigned = _event(
        lease1,
        code=ContinuationLeaseEventCodeV1.REASSIGNED,
        at=lease2.granted_at_utc,
        prior_event=released,
        successor=lease2,
    )
    ledger = ContinuationLeaseLedgerV1.initial(lease1).append_event(released)
    ledger = ledger.transition(reassigned, lease2)
    start2 = _start(
        run=run,
        ordinal=2,
        nonce="rerun",
        concretized=concretized,
        implementation=implementation,
        admission=admission,
        disposition=disposition2,
        lease=lease2,
        parent=start1.attempt_id,
    )
    unsupported_basis = _basis(
        run=run,
        candidate_id=attempt2,
        candidate=concretized,
        code=AttemptRetryBasisCodeV1.NEW_INFORMATION,
        prior=prior,
        evidence_name="generic_note",
    )

    with pytest.raises(AttemptAdmissionError, match="new_information"):
        validate_attempt_start_v1(
            run=run,
            semantic_spec=semantic,
            concretized_execution=concretized,
            implementation=implementation,
            admission_request=admission,
            disposition=disposition2,
            retry_basis=unsupported_basis,
            start=start2,
            continuation_lease=lease2,
            continuation_lease_ledger=ledger,
            execution_side_effect_at_utc="2026-08-23T04:04:00Z",
            prior_records=(prior,),
        )


def test_precheckpoint_crash_restarts_from_absolute_run_start() -> None:
    semantic, run, implementation, admission, concretized, _, lease1, _, start1 = (
        _initial_bundle()
    )
    terminal1 = _terminal(
        start1,
        status=AttemptTerminalStatusV1.FAILED,
        frontiers=AttemptFrontiersV1(120, 100, 0, 0),
        label="crash-before-checkpoint",
    )
    prior = AttemptRecordV1(start1, concretized, terminal1)
    attempt2 = derive_attempt_id(
        run_id=run.run_id, monotonic_ordinal=2, nonce="restart"
    )
    disposition2 = _disposition(run, admission, start_lease="start-2")
    lease2 = _lease(
        run,
        attempt2,
        label="2",
        epoch=2,
        prior=lease1.lease_digest(),
        granted=lease1.expires_at_utc,
        expires="2026-08-23T04:10:00Z",
    )
    ledger = _reclaimed_ledger(lease1, lease2)
    start2 = _start(
        run=run,
        ordinal=2,
        nonce="restart",
        concretized=concretized,
        implementation=implementation,
        admission=admission,
        disposition=disposition2,
        lease=lease2,
        parent=start1.attempt_id,
    )
    basis2 = _basis(
        run=run,
        candidate_id=attempt2,
        candidate=concretized,
        code=AttemptRetryBasisCodeV1.CHANGED_CONDITION,
        prior=prior,
        evidence_name="changed_condition",
    )

    assert (
        validate_attempt_start_v1(
            run=run,
            semantic_spec=semantic,
            concretized_execution=concretized,
            implementation=implementation,
            admission_request=admission,
            disposition=disposition2,
            retry_basis=basis2,
            start=start2,
            continuation_lease=lease2,
            continuation_lease_ledger=ledger,
            execution_side_effect_at_utc="2026-08-23T04:06:00Z",
            prior_records=(prior,),
        )["attempt_ordinal"]
        == 2
    )
    assert start2.initial_frontiers == AttemptFrontiersV1(0, 0, 0, 0)
    assert semantic.absolute_start_next_tick == 0


def _postcheckpoint_retry() -> tuple[
    RunOccurrenceV1,
    ExecutionImplementationV1,
    AttemptAdmissionRequestV1,
    ConcretizedExecutionV1,
    AttemptRecordV1,
    RestoreCheckpointAuthorizationV1,
    AttemptAdmissionDispositionV1,
    CanonicalContinuationLeaseV1,
    ContinuationLeaseLedgerV1,
    AttemptRetryBasisV1,
    AttemptStartV1,
]:
    _, run, implementation, _, concretized1, _, lease1, _, start1 = _initial_bundle()
    terminal1 = _terminal(
        start1,
        status=AttemptTerminalStatusV1.INTERRUPTED,
        frontiers=AttemptFrontiersV1(700, 650, 500, 0),
        label="crash-after-checkpoint",
    )
    prior = AttemptRecordV1(start1, concretized1, terminal1)
    checkpoint = _digest("checkpoint-500")
    authorization = RestoreCheckpointAuthorizationV1(
        checkpoint_digest=checkpoint,
        run_id=run.run_id,
        source_attempt_id=start1.attempt_id,
        source_attempt_terminal_digest=terminal1.terminal_digest(),
        absolute_next_tick=500,
        sealed_bundle_digest=_digest("sealed-bundle"),
        completion_marker_digest=_digest("completion-marker"),
    )
    admission2 = _admission("restore", restore=checkpoint)
    concretized2 = _concretized(_semantic(), implementation, admission2)
    attempt2 = derive_attempt_id(
        run_id=run.run_id, monotonic_ordinal=2, nonce="restore"
    )
    disposition2 = _disposition(run, admission2, start_lease="start-2")
    lease2 = _lease(
        run,
        attempt2,
        label="2",
        epoch=2,
        restore=checkpoint,
        prior=lease1.lease_digest(),
        granted=lease1.expires_at_utc,
        expires="2026-08-23T04:10:00Z",
    )
    ledger = _reclaimed_ledger(lease1, lease2)
    start2 = _start(
        run=run,
        ordinal=2,
        nonce="restore",
        concretized=concretized2,
        implementation=implementation,
        admission=admission2,
        disposition=disposition2,
        lease=lease2,
        parent=start1.attempt_id,
        restore=checkpoint,
        frontiers=AttemptFrontiersV1(500, 500, 500, 0),
    )
    basis2 = _basis(
        run=run,
        candidate_id=attempt2,
        candidate=concretized2,
        code=AttemptRetryBasisCodeV1.EXACT_RESTORE_AFTER_FAILURE,
        prior=prior,
        evidence_name="restore_checkpoint",
    )
    return (
        run,
        implementation,
        admission2,
        concretized2,
        prior,
        authorization,
        disposition2,
        lease2,
        ledger,
        basis2,
        start2,
    )


def test_postcheckpoint_crash_and_fresh_owner_resume_exact_absolute_tick() -> None:
    (
        run,
        implementation,
        admission,
        concretized,
        prior,
        authorization,
        disposition,
        lease,
        ledger,
        basis,
        start,
    ) = _postcheckpoint_retry()

    result = validate_attempt_start_v1(
        run=run,
        semantic_spec=_semantic(),
        concretized_execution=concretized,
        implementation=implementation,
        admission_request=admission,
        disposition=disposition,
        retry_basis=basis,
        start=start,
        continuation_lease=lease,
        continuation_lease_ledger=ledger,
        execution_side_effect_at_utc="2026-08-23T04:06:00Z",
        prior_records=(prior,),
        restore_authorization=authorization,
    )

    assert result["status"] == "START_VALID"
    assert start.initial_frontiers.durable_checkpoint_next_tick == 500

    mismatched_admission = replace(
        admission,
        requested_restore_checkpoint_digest=_digest("another-checkpoint"),
    )
    mismatched_disposition = replace(
        disposition,
        attempt_admission_request_digest=(
            mismatched_admission.attempt_admission_request_digest()
        ),
    )
    mismatched_concretized = replace(
        concretized,
        attempt_admission_request_digest=(
            mismatched_admission.attempt_admission_request_digest()
        ),
    )
    mismatched_basis = replace(
        basis,
        candidate_concretized_execution_digest=(
            mismatched_concretized.concretized_execution_digest()
        ),
    )
    mismatched_start = replace(
        start,
        concretized_execution_digest=(
            mismatched_concretized.concretized_execution_digest()
        ),
        attempt_admission_request_digest=(
            mismatched_admission.attempt_admission_request_digest()
        ),
        admission_disposition_digest=mismatched_disposition.disposition_digest(),
    )
    with pytest.raises(AttemptAdmissionError, match="request restore checkpoint"):
        validate_attempt_start_v1(
            run=run,
            semantic_spec=_semantic(),
            concretized_execution=mismatched_concretized,
            implementation=implementation,
            admission_request=mismatched_admission,
            disposition=mismatched_disposition,
            retry_basis=mismatched_basis,
            start=mismatched_start,
            continuation_lease=lease,
            continuation_lease_ledger=ledger,
            execution_side_effect_at_utc="2026-08-23T04:06:00Z",
            prior_records=(prior,),
            restore_authorization=authorization,
        )


def test_reboot_with_changed_implementation_requires_exact_conformance() -> None:
    (
        run,
        _,
        admission,
        _,
        prior,
        authorization,
        disposition,
        lease,
        ledger,
        _,
        _,
    ) = _postcheckpoint_retry()
    implementation2 = _implementation("reboot")
    concretized2 = _concretized(_semantic(), implementation2, admission)
    start2 = _start(
        run=run,
        ordinal=2,
        nonce="restore",
        concretized=concretized2,
        implementation=implementation2,
        admission=admission,
        disposition=disposition,
        lease=lease,
        parent=prior.start.attempt_id,
        restore=authorization.checkpoint_digest,
        frontiers=AttemptFrontiersV1(500, 500, 500, 0),
    )
    no_proof = _basis(
        run=run,
        candidate_id=start2.attempt_id,
        candidate=concretized2,
        code=AttemptRetryBasisCodeV1.EXACT_RESTORE_AFTER_FAILURE,
        prior=prior,
        evidence_name="restore_checkpoint",
    )
    with pytest.raises(AttemptAdmissionError, match="conformance"):
        validate_attempt_start_v1(
            run=run,
            semantic_spec=_semantic(),
            concretized_execution=concretized2,
            implementation=implementation2,
            admission_request=admission,
            disposition=disposition,
            retry_basis=no_proof,
            start=start2,
            continuation_lease=lease,
            continuation_lease_ledger=ledger,
            execution_side_effect_at_utc="2026-08-23T04:06:00Z",
            prior_records=(prior,),
            restore_authorization=authorization,
        )
    wrong_conformance = ExactImplementationConformanceV1(
        run_id=run.run_id,
        candidate_attempt_id=start2.attempt_id,
        prior_implementation_digest=(prior.concretized_execution.implementation_digest),
        candidate_implementation_digest=implementation2.implementation_digest(),
        candidate_abi_digest=_digest("wrong-abi"),
        candidate_device_conformance_envelope_digest=(
            implementation2.device_conformance_envelope_digest
        ),
        exact_equivalence_evidence_digests=(_named("exact_binary_equivalence"),),
        qualification_delta_digest=_digest("artifact-equivalence-delta"),
    )
    wrong_basis = replace(
        no_proof,
        implementation_conformance_digest=wrong_conformance.conformance_digest(),
    )
    with pytest.raises(AttemptAdmissionError, match="conformance binding differs"):
        validate_attempt_start_v1(
            run=run,
            semantic_spec=_semantic(),
            concretized_execution=concretized2,
            implementation=implementation2,
            admission_request=admission,
            disposition=disposition,
            retry_basis=wrong_basis,
            start=start2,
            continuation_lease=lease,
            continuation_lease_ledger=ledger,
            execution_side_effect_at_utc="2026-08-23T04:06:00Z",
            prior_records=(prior,),
            restore_authorization=authorization,
            implementation_conformance=wrong_conformance,
        )
    conformance = replace(
        wrong_conformance,
        candidate_abi_digest=implementation2.abi_digest,
    )
    with_proof = replace(
        no_proof,
        implementation_conformance_digest=conformance.conformance_digest(),
    )
    assert (
        validate_attempt_start_v1(
            run=run,
            semantic_spec=_semantic(),
            concretized_execution=concretized2,
            implementation=implementation2,
            admission_request=admission,
            disposition=disposition,
            retry_basis=with_proof,
            start=start2,
            continuation_lease=lease,
            continuation_lease_ledger=ledger,
            execution_side_effect_at_utc="2026-08-23T04:06:00Z",
            prior_records=(prior,),
            restore_authorization=authorization,
            implementation_conformance=conformance,
        )["status"]
        == "START_VALID"
    )


def test_racing_retry_stale_publisher_is_quarantined(tmp_path: Path) -> None:
    run, _, _, _, _, authorization, _, lease2, ledger, _, start2 = (
        _postcheckpoint_retry()
    )
    attempt3 = derive_attempt_id(
        run_id=run.run_id, monotonic_ordinal=3, nonce="racing-retry"
    )
    lease3 = _lease(
        run,
        attempt3,
        label="3",
        epoch=3,
        restore=authorization.checkpoint_digest,
        prior=lease2.lease_digest(),
        granted=lease2.expires_at_utc,
        expires="2026-08-23T04:15:00Z",
        owner="owner-b",
    )
    reclaim = _event(
        lease2,
        code=ContinuationLeaseEventCodeV1.RECLAIMED,
        at=lease2.expires_at_utc,
        prior_event=ledger.events[-1],
        successor=lease3,
    )
    ledger = ledger.transition(reclaim, lease3)

    with pytest.raises(AttemptAdmissionError, match="QUARANTINE_REQUIRED"):
        require_canonical_publish_authority_v1(
            ledger, lease2, publication_at_utc="2026-08-23T04:11:00Z"
        )
    loser_terminal = _terminal(
        start2,
        status=AttemptTerminalStatusV1.QUARANTINED,
        frontiers=AttemptFrontiersV1(600, 600, 500, 0),
        label="stale-publisher-quarantined",
    )
    assert (
        validate_terminal_once_v1(start2, None, loser_terminal, run, _semantic()).status
        is AttemptTerminalStatusV1.QUARANTINED
    )
    require_canonical_publish_authority_v1(
        ledger, lease3, publication_at_utc="2026-08-23T04:11:00Z"
    )
    publication = _event(
        lease3,
        code=ContinuationLeaseEventCodeV1.PUBLISH_COMMITTED,
        at="2026-08-23T04:11:00Z",
        prior_event=reclaim,
        publication=_digest("canonical-publication"),
    )
    committed, pointer = commit_canonical_publication_v1(
        tmp_path, ledger=ledger, lease=lease3, event=publication
    )
    assert pointer.is_file()
    assert committed.canonical_publication_digest == _digest("canonical-publication")
    with pytest.raises(AttemptAdmissionError, match="already published"):
        require_canonical_publish_authority_v1(
            committed, lease3, publication_at_utc="2026-08-23T04:12:00Z"
        )
    competing = replace(
        publication,
        publication_digest=_digest("competing-publication"),
        reason="competing-publication",
    )
    with pytest.raises(AttemptAdmissionError, match="QUARANTINE_REQUIRED"):
        commit_canonical_publication_v1(
            tmp_path, ledger=ledger, lease=lease3, event=competing
        )


def test_heartbeat_renewal_release_and_early_reclaim_are_fenced() -> None:
    _, run, *_ = _initial_bundle()
    attempt1 = derive_attempt_id(
        run_id=run.run_id, monotonic_ordinal=1, nonce="initial"
    )
    lease1 = _lease(run, attempt1, label="1")
    heartbeat = _event(
        lease1,
        code=ContinuationLeaseEventCodeV1.HEARTBEAT,
        at="2026-08-23T04:01:00Z",
    )
    lease2 = _lease(
        run,
        attempt1,
        label="renewed",
        epoch=2,
        prior=lease1.lease_digest(),
        granted="2026-08-23T04:02:00Z",
        expires="2026-08-23T04:07:00Z",
    )
    renewed = _event(
        lease1,
        code=ContinuationLeaseEventCodeV1.RENEWED,
        at=lease2.granted_at_utc,
        prior_event=heartbeat,
        successor=lease2,
    )
    ledger = ContinuationLeaseLedgerV1.initial(lease1).append_event(heartbeat)
    ledger = ledger.transition(renewed, lease2)
    released = _event(
        lease2,
        code=ContinuationLeaseEventCodeV1.RELEASED,
        at="2026-08-23T04:03:00Z",
        prior_event=renewed,
    )
    released_ledger = ledger.append_event(released)
    assert released_ledger.events[-1].code is (ContinuationLeaseEventCodeV1.RELEASED)
    assert released_ledger.is_active is False

    early_successor = replace(
        lease2,
        lease_id=LeaseId("lease-early"),
        lease_epoch=3,
        fencing_token_digest=_digest("fence-early"),
        prior_lease_digest=lease2.lease_digest(),
        granted_at_utc="2026-08-23T04:04:00Z",
        expires_at_utc="2026-08-23T04:09:00Z",
    )
    early_reclaim = _event(
        lease2,
        code=ContinuationLeaseEventCodeV1.RECLAIMED,
        at=early_successor.granted_at_utc,
        prior_event=renewed,
        successor=early_successor,
    )
    with pytest.raises(AttemptAdmissionError, match="before the fencing deadline"):
        ledger.transition(early_reclaim, early_successor)


def test_proof_bound_dead_owner_fence_permits_preexpiry_reclaim() -> None:
    _, run, *_ = _initial_bundle()
    attempt1 = derive_attempt_id(
        run_id=run.run_id, monotonic_ordinal=1, nonce="dead-owner"
    )
    lease1 = _lease(run, attempt1, label="dead-owner")
    dead_owner = _event(
        lease1,
        code=ContinuationLeaseEventCodeV1.OWNER_DEAD,
        at="2026-08-23T04:01:00Z",
        owner_dead_proof=_digest("dead-owner-proof"),
    )
    successor = _lease(
        run,
        derive_attempt_id(
            run_id=run.run_id, monotonic_ordinal=2, nonce="dead-owner-successor"
        ),
        label="dead-owner-successor",
        epoch=2,
        prior=lease1.lease_digest(),
        granted="2026-08-23T04:02:00Z",
        expires="2026-08-23T04:07:00Z",
    )
    reclaim = _event(
        lease1,
        code=ContinuationLeaseEventCodeV1.RECLAIMED,
        at=successor.granted_at_utc,
        prior_event=dead_owner,
        successor=successor,
    )

    ledger = ContinuationLeaseLedgerV1.initial(lease1).append_event(dead_owner)
    reclaimed = ledger.transition(reclaim, successor)

    assert reclaimed.current_lease == successor
    assert [event.code.value for event in reclaimed.events] == [
        "OWNER_DEAD",
        "RECLAIMED",
    ]


def test_frontiers_and_terminal_receipt_are_append_only() -> None:
    semantic, run, *_, start = _initial_bundle()
    validate_frontier_progression_v1(
        AttemptFrontiersV1(100, 90, 50, 0),
        AttemptFrontiersV1(120, 100, 50, 20),
    )
    with pytest.raises(AttemptAdmissionError, match="moved backwards"):
        validate_frontier_progression_v1(
            AttemptFrontiersV1(100, 90, 50, 0),
            AttemptFrontiersV1(99, 90, 50, 0),
        )
    first = _terminal(
        start,
        status=AttemptTerminalStatusV1.FAILED,
        frontiers=AttemptFrontiersV1(100, 90, 50, 0),
        label="first-terminal",
    )
    assert validate_terminal_once_v1(start, None, first, run, semantic) == first
    assert validate_terminal_once_v1(start, first, first, run, semantic) == first
    with pytest.raises(AttemptAdmissionError, match="semantic specification"):
        validate_terminal_once_v1(start, None, first, run, _semantic("changed"))
    conflicting = replace(first, reason="different-terminal")
    with pytest.raises(AttemptAdmissionError, match="different immutable terminal"):
        validate_terminal_once_v1(start, first, conflicting, run, semantic)

    partial_completed = _terminal(
        start,
        status=AttemptTerminalStatusV1.COMPLETED,
        frontiers=AttemptFrontiersV1(999, 999, 500, 500),
        label="partial-completed",
    )
    with pytest.raises(AttemptAdmissionError, match="absolute Run target"):
        validate_terminal_once_v1(start, None, partial_completed, run, semantic)


def test_branch_and_repeat_are_new_run_occurrences_not_retry_attempts() -> None:
    _, parent_run, _, _, parent_concretized, _, _, _, parent_start = _initial_bundle()
    parent_terminal = _terminal(
        parent_start,
        status=AttemptTerminalStatusV1.INTERRUPTED,
        frontiers=AttemptFrontiersV1(500, 500, 500, 0),
        label="parent-checkpoint",
    )
    parent_source = AttemptRecordV1(parent_start, parent_concretized, parent_terminal)
    checkpoint = _digest("branch-checkpoint")
    authorization = RestoreCheckpointAuthorizationV1(
        checkpoint_digest=checkpoint,
        run_id=parent_run.run_id,
        source_attempt_id=parent_start.attempt_id,
        source_attempt_terminal_digest=parent_terminal.terminal_digest(),
        absolute_next_tick=500,
        sealed_bundle_digest=_digest("branch-bundle"),
        completion_marker_digest=_digest("branch-marker"),
    )
    branch_semantic = _semantic("branch")
    child = _run(
        branch_semantic,
        parent=checkpoint,
        branch=_digest("branch-spec"),
    )
    implementation = _implementation()
    admission = _admission("branch", restore=checkpoint)
    concretized = _concretized(branch_semantic, implementation, admission)
    child_attempt = derive_attempt_id(
        run_id=child.run_id, monotonic_ordinal=1, nonce="branch-initial"
    )
    disposition = _disposition(child, admission, start_lease="branch-start")
    lease = _lease(child, child_attempt, label="branch", restore=checkpoint)
    start = _start(
        run=child,
        ordinal=1,
        nonce="branch-initial",
        concretized=concretized,
        implementation=implementation,
        admission=admission,
        disposition=disposition,
        lease=lease,
        restore=checkpoint,
        frontiers=AttemptFrontiersV1(500, 500, 500, 0),
    )
    basis = _basis(
        run=child,
        candidate_id=child_attempt,
        candidate=concretized,
        code=AttemptRetryBasisCodeV1.INITIAL_EXECUTION,
    )
    with pytest.raises(AttemptAdmissionError, match="parent source Attempt"):
        validate_attempt_start_v1(
            run=child,
            semantic_spec=branch_semantic,
            concretized_execution=concretized,
            implementation=implementation,
            admission_request=admission,
            disposition=disposition,
            retry_basis=basis,
            start=start,
            continuation_lease=lease,
            continuation_lease_ledger=ContinuationLeaseLedgerV1.initial(lease),
            execution_side_effect_at_utc="2026-08-23T04:01:00Z",
            restore_authorization=authorization,
        )
    assert (
        validate_attempt_start_v1(
            run=child,
            semantic_spec=branch_semantic,
            concretized_execution=concretized,
            implementation=implementation,
            admission_request=admission,
            disposition=disposition,
            retry_basis=basis,
            start=start,
            continuation_lease=lease,
            continuation_lease_ledger=ContinuationLeaseLedgerV1.initial(lease),
            execution_side_effect_at_utc="2026-08-23T04:01:00Z",
            restore_authorization=authorization,
            restore_source_record=parent_source,
        )["status"]
        == "START_VALID"
    )
    assert child.run_id != parent_run.run_id

    repeat = _run(_semantic(), ordinal=1)
    assert repeat.run_id != parent_run.run_id
    assert repeat.semantic_spec_digest == parent_run.semantic_spec_digest
    assert parent_concretized.semantic_spec_digest == repeat.semantic_spec_digest


@pytest.mark.parametrize(
    "action_kind", ["ARTIFACT_ONLY", "EVIDENCE_REEVALUATION", "REPORTING_ONLY"]
)
def test_non_runtime_derivations_never_create_attempts(action_kind: str) -> None:
    decision = non_runtime_occurrence_decision_v1(action_kind)
    assert decision["runtime_attempt_created"] is False
    assert decision["execution_side_effect_authorized"] is False


def test_q6_integration_escape_cannot_continue_the_frozen_run() -> None:
    _, run, implementation, admission, concretized, _, _, _, start1 = _initial_bundle()
    terminal1 = _terminal(
        start1,
        status=AttemptTerminalStatusV1.FAILED,
        frontiers=AttemptFrontiersV1(100, 90, 0, 0),
        label="q6-integration-defect",
    )
    prior = AttemptRecordV1(start1, concretized, terminal1)
    attempt2 = derive_attempt_id(
        run_id=run.run_id, monotonic_ordinal=2, nonce="q6-escape"
    )
    disposition2 = _disposition(run, admission, start_lease="q6-start")
    lease2 = _lease(run, attempt2, label="q6")
    start2 = _start(
        run=run,
        ordinal=2,
        nonce="q6-escape",
        concretized=concretized,
        implementation=implementation,
        admission=admission,
        disposition=disposition2,
        lease=lease2,
        parent=start1.attempt_id,
    )
    basis2 = _basis(
        run=run,
        candidate_id=attempt2,
        candidate=concretized,
        code=AttemptRetryBasisCodeV1.QUALIFICATION_INTEGRATION_ESCAPE,
        prior=prior,
        evidence_name="qualification_integration_escape",
    )

    with pytest.raises(AttemptAdmissionError, match="successor Run"):
        validate_attempt_start_v1(
            run=run,
            semantic_spec=_semantic(),
            concretized_execution=concretized,
            implementation=implementation,
            admission_request=admission,
            disposition=disposition2,
            retry_basis=basis2,
            start=start2,
            continuation_lease=lease2,
            continuation_lease_ledger=ContinuationLeaseLedgerV1.initial(lease2),
            execution_side_effect_at_utc="2026-08-23T04:01:00Z",
            prior_records=(prior,),
        )


def test_content_addressed_receipts_are_append_only_and_idempotent(
    tmp_path: Path,
) -> None:
    *_, start = _initial_bundle()
    payload = start.to_dict()
    digest = start.start_digest()
    first = append_occurrence_receipt_v1(
        tmp_path, kind="attempt-start", payload=payload, digest=digest
    )
    second = append_occurrence_receipt_v1(
        tmp_path, kind="attempt-start", payload=payload, digest=digest
    )
    assert first == second
    first.write_text("{}\n", encoding="utf-8")
    with pytest.raises(AttemptAdmissionError, match="already differs"):
        append_occurrence_receipt_v1(
            tmp_path, kind="attempt-start", payload=payload, digest=digest
        )


def test_attempt_terminal_pointer_rejects_a_second_terminal(tmp_path: Path) -> None:
    semantic, run, *_, start = _initial_bundle()
    first = _terminal(
        start,
        status=AttemptTerminalStatusV1.FAILED,
        frontiers=AttemptFrontiersV1(100, 90, 50, 0),
        label="first-terminal",
    )
    path = append_attempt_terminal_receipt_v1(
        tmp_path, start=start, terminal=first, run=run, semantic_spec=semantic
    )
    assert path.is_file()
    assert (
        append_attempt_terminal_receipt_v1(
            tmp_path,
            start=start,
            terminal=first,
            run=run,
            semantic_spec=semantic,
        )
        == path
    )
    conflicting = replace(first, reason="conflicting-terminal")
    with pytest.raises(AttemptAdmissionError, match="different terminal"):
        append_attempt_terminal_receipt_v1(
            tmp_path,
            start=start,
            terminal=conflicting,
            run=run,
            semantic_spec=semantic,
        )
