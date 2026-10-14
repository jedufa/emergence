from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from emergence_contracts import (
    AdmissionAssuranceBindingV1,
    AttemptAdmissionDispositionCodeV1,
    AttemptAdmissionDispositionV1,
    AttemptAdmissionRequestV1,
    AttemptFrontiersV1,
    AttemptId,
    AttemptStartV1,
    AttemptTerminalStatusV1,
    AttemptTerminalV1,
    BackendId,
    ConcretizedExecutionV1,
    ContentDigest,
    ContractError,
    ExecutionImplementationV1,
    ExperimentId,
    LifecycleBindingV1,
    LeaseId,
    NamedDigestV1,
    RunOccurrenceV1,
    CanonicalContinuationLeaseV1,
    RUN_ATTEMPT_EDGE_CASE_MATRIX_CASES,
    RUN_ATTEMPT_EDGE_CASE_MATRIX_SHA256,
    SemanticRunSpecV1,
    canonical_sha256,
    derive_attempt_id,
)
from emergence_contracts.resources import (
    CRCA_G0_RATCHET_RECONCILIATION_V1,
    CRCA_RUN_ATTEMPT_EDGE_CASES_V1,
    read_resource_bytes,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULAR_RATCHET_BASELINE = (
    REPO_ROOT
    / "tools"
    / "architecture"
    / "baselines"
    / "eiel_modular_legacy_imports_v1.json"
)


def _digest(label: str) -> ContentDigest:
    return canonical_sha256({"label": label})


def _named(label: str) -> NamedDigestV1:
    return NamedDigestV1(name=label, digest=_digest(label))


def _semantic() -> SemanticRunSpecV1:
    return SemanticRunSpecV1(
        experiment_id=ExperimentId("experiment-a"),
        system_assembly_digest=_digest("assembly"),
        semantic_parameters_digest=_digest("parameters"),
        rng_contract_digest=_digest("rng-v1"),
        seed_digest=_digest("seed-1708"),
        absolute_start_next_tick=0,
        absolute_target_next_tick=960,
        trajectory_checkpoint_ticks=(0, 480, 960),
        measurement_behavior_digest=_digest("measurement"),
        control_and_branch_semantics_digest=_digest("control"),
    )


def _implementation() -> ExecutionImplementationV1:
    return ExecutionImplementationV1(
        source_commit="6d6d4cb0ecf61315d949e5110d024f07eb41e597",
        source_tree_digest=_digest("tree"),
        dirty_patch_digest=None,
        toolchain_digest=_digest("toolchain"),
        environment_digest=_digest("environment"),
        provider_id="production-engine-provider-v1",
        backend_id=BackendId("resident-cuda"),
        runtime_profile_id="b136-resident",
        native_artifact_digests=(_named("extension"),),
        abi_digest=_digest("abi"),
        device_conformance_envelope_digest=_digest("device-envelope"),
    )


def _assurance() -> AdmissionAssuranceBindingV1:
    return AdmissionAssuranceBindingV1(
        intended_claim_id="engineering-shadow-only",
        qualified_envelope_digest=_digest("qualified-envelope"),
        obligation_digests=(_named("absolute-tick"),),
        authority_reference_digests=(_named("authority-register"),),
        review_reference_digests=(),
    )


def _lifecycle() -> LifecycleBindingV1:
    return LifecycleBindingV1(
        output_root="/external/runtime/fresh-root",
        artifact_lifecycle_contract_digest=_digest("lifecycle"),
        inventory_role_digests=(_named("checkpoint-inventory"),),
        initial_hold_digests=(),
        retention_profile_id="CHECKPOINT_TRANSIT",
    )


def _admission_request() -> AttemptAdmissionRequestV1:
    return AttemptAdmissionRequestV1(
        resource_requirements_digest=_digest("resources"),
        executor_constraints_digest=_digest("executor"),
        requested_restore_checkpoint_digest=None,
        observational_persistence_cadence_digest=_digest("cadence"),
        proposed_reason="initial execution",
    )


def _concretized() -> ConcretizedExecutionV1:
    semantic = _semantic()
    implementation = _implementation()
    assurance = _assurance()
    lifecycle = _lifecycle()
    admission = _admission_request()
    return ConcretizedExecutionV1(
        semantic_spec_digest=semantic.semantic_spec_digest(),
        implementation_digest=implementation.implementation_digest(),
        admission_assurance_digest=assurance.admission_assurance_digest(),
        lifecycle_digest=lifecycle.lifecycle_digest(),
        attempt_admission_request_digest=admission.attempt_admission_request_digest(),
        campaign_plan_digest=_digest("campaign-plan"),
        bound_execution_request_digest=_digest("bound-request"),
        run_contract_digest=_digest("run-contract"),
        baseline_digest=_digest("baseline"),
        route_contract_digest=_digest("route"),
        assembly_digest=_digest("assembly"),
        qualification_key_digest=_digest("qualification-key"),
    )


def _run(ordinal: int = 0) -> RunOccurrenceV1:
    semantic = _semantic()
    return RunOccurrenceV1.create(
        experiment_id=semantic.experiment_id,
        cell_or_repeat_ordinal=ordinal,
        semantic_spec_digest=semantic.semantic_spec_digest(),
    )


def test_semantic_digest_is_deterministic_and_absolute_tick_only() -> None:
    first = _semantic()
    second = _semantic()
    assert first.semantic_spec_digest() == second.semantic_spec_digest()
    assert "generation" not in repr(first.to_dict()).lower()
    assert first.to_dict()["trajectory_checkpoint_ticks"] == [0, 480, 960]


def test_semantic_change_changes_only_the_semantic_layer() -> None:
    original = _semantic()
    changed = replace(original, absolute_target_next_tick=961)
    assert original.semantic_spec_digest() != changed.semantic_spec_digest()


def test_repeat_ordinal_creates_sibling_run_occurrence() -> None:
    first = _run(0)
    repeat = _run(1)
    assert first.semantic_spec_digest == repeat.semantic_spec_digest
    assert first.run_id != repeat.run_id


def test_run_and_attempt_identity_reference_vectors_are_stable() -> None:
    run = _run()
    assert str(run.run_id) == (
        "crca-run-v1-2a96f6b2edc1143b8efae32ffbda4568"
        "229a4ec6215581c32df38311b8cbb1ca"
    )
    assert str(
        derive_attempt_id(run_id=run.run_id, monotonic_ordinal=1, nonce="nonce-1")
    ) == (
        "crca-attempt-v1-3d0d1e25db5d9813dbaebcf36397ec4f"
        "f8421edcf31e0896060a2e07f5037cbe"
    )


def test_concretized_identity_binds_all_five_preadmission_layers() -> None:
    original = _concretized()
    changed = replace(original, lifecycle_digest=_digest("different-lifecycle"))
    assert original.semantic_spec_digest == changed.semantic_spec_digest
    assert original.concretized_execution_digest() != changed.concretized_execution_digest()


def test_resource_wait_is_a_preadmission_disposition_not_an_attempt() -> None:
    run = _run()
    disposition = AttemptAdmissionDispositionV1(
        run_id=run.run_id,
        attempt_admission_request_digest=_admission_request().attempt_admission_request_digest(),
        code=AttemptAdmissionDispositionCodeV1.WAIT_RESOURCE,
        reason="GPU lease unavailable",
        scheduler_assignment_lease_digest=None,
        resource_start_lease_digest=None,
    )
    assert "attempt_id" not in disposition.to_dict()


def test_start_authorization_requires_resource_start_lease() -> None:
    with pytest.raises(ContractError, match="resource-start lease"):
        AttemptAdmissionDispositionV1(
            run_id=_run().run_id,
            attempt_admission_request_digest=_digest("request"),
            code=AttemptAdmissionDispositionCodeV1.START_AUTHORIZED,
            reason="provider ready",
            scheduler_assignment_lease_digest=None,
            resource_start_lease_digest=None,
        )


def test_attempt_start_is_bound_to_side_effect_and_leases() -> None:
    run = _run()
    attempt_id = derive_attempt_id(
        run_id=run.run_id, monotonic_ordinal=1, nonce="nonce-1"
    )
    start = AttemptStartV1(
        attempt_id=attempt_id,
        run_id=run.run_id,
        monotonic_ordinal=1,
        nonce="nonce-1",
        concretized_execution_digest=_concretized().concretized_execution_digest(),
        implementation_digest=_implementation().implementation_digest(),
        attempt_admission_request_digest=_admission_request().attempt_admission_request_digest(),
        scheduler_assignment_lease_digest=None,
        resource_start_lease_digest=_digest("resource-start-lease"),
        post_admission_attempt_plan_digest=_digest("post-admission-plan"),
        admission_disposition_digest=_digest("start-authorized"),
        programme_work_session_assignment_dispatch_binding_digest=_digest(
            "control-plane-binding"
        ),
        parent_attempt_id=None,
        restore_checkpoint_digest=None,
        canonical_continuation_lease_digest=_digest("continuation-lease"),
        owner_binding_digest=_digest("owner-binding"),
        execution_side_effect_digest=_digest("process-start"),
        initial_frontiers=AttemptFrontiersV1(0, 0, 0, 0),
    )
    assert start.attempt_id == attempt_id
    assert start.to_dict()["execution_side_effect_digest"] == _digest(
        "process-start"
    ).to_dict()
    assert start.start_digest() == canonical_sha256(start.to_dict())


def test_continuation_lease_binds_one_run_attempt_and_epoch() -> None:
    run = _run()
    attempt_id = derive_attempt_id(
        run_id=run.run_id, monotonic_ordinal=2, nonce="retry"
    )
    lease = CanonicalContinuationLeaseV1(
        lease_id=LeaseId("continuation-lease-2"),
        run_id=run.run_id,
        attempt_id=attempt_id,
        restore_checkpoint_digest=_digest("sealed-checkpoint"),
        lineage_scope_digest=_digest("canonical-lineage"),
        owner_id="scheduler-v1",
        lease_epoch=2,
        fencing_token_digest=_digest("fence-2"),
        prior_lease_digest=_digest("lease-1"),
        granted_at_utc="2026-08-23T01:00:00Z",
        expires_at_utc="2026-08-23T01:05:00Z",
    )
    assert lease.lease_digest() == canonical_sha256(lease.to_dict())


def test_continuation_lease_rejects_nonpositive_validity_window() -> None:
    run = _run()
    with pytest.raises(ContractError, match="expire after grant"):
        CanonicalContinuationLeaseV1(
            lease_id=LeaseId("continuation-lease-2"),
            run_id=run.run_id,
            attempt_id=derive_attempt_id(
                run_id=run.run_id, monotonic_ordinal=2, nonce="retry"
            ),
            restore_checkpoint_digest=_digest("sealed-checkpoint"),
            lineage_scope_digest=_digest("canonical-lineage"),
            owner_id="scheduler-v1",
            lease_epoch=2,
            fencing_token_digest=_digest("fence-2"),
            prior_lease_digest=_digest("lease-1"),
            granted_at_utc="2026-08-23T01:05:00Z",
            expires_at_utc="2026-08-23T01:05:00Z",
        )


def test_terminal_transition_is_bound_to_the_attempt_start() -> None:
    run = _run()
    attempt_id = derive_attempt_id(
        run_id=run.run_id, monotonic_ordinal=1, nonce="nonce-1"
    )
    start = AttemptStartV1(
        attempt_id=attempt_id,
        run_id=run.run_id,
        monotonic_ordinal=1,
        nonce="nonce-1",
        concretized_execution_digest=_digest("concretized"),
        implementation_digest=_digest("implementation"),
        attempt_admission_request_digest=_digest("request"),
        scheduler_assignment_lease_digest=None,
        resource_start_lease_digest=_digest("start-lease"),
        post_admission_attempt_plan_digest=_digest("plan"),
        admission_disposition_digest=_digest("start-authorized"),
        programme_work_session_assignment_dispatch_binding_digest=_digest(
            "control-plane-binding"
        ),
        parent_attempt_id=None,
        restore_checkpoint_digest=None,
        canonical_continuation_lease_digest=_digest("continuation"),
        owner_binding_digest=_digest("owners"),
        execution_side_effect_digest=_digest("side-effect"),
        initial_frontiers=AttemptFrontiersV1(0, 0, 0, 0),
    )
    terminal = AttemptTerminalV1(
        attempt_id=attempt_id,
        attempt_start_digest=start.start_digest(),
        status=AttemptTerminalStatusV1.QUARANTINED,
        final_frontiers=AttemptFrontiersV1(480, 480, 480, 0),
        result_digest=_digest("quarantine-receipt"),
        reason="lost canonical continuation lease",
    )
    assert terminal.terminal_digest() == canonical_sha256(terminal.to_dict())
    terminal.validate_transition(start)


def test_attempt_start_requires_the_exact_authorized_disposition() -> None:
    run = _run()
    disposition = AttemptAdmissionDispositionV1(
        run_id=run.run_id,
        attempt_admission_request_digest=_digest("request"),
        code=AttemptAdmissionDispositionCodeV1.START_AUTHORIZED,
        reason="provider ready",
        scheduler_assignment_lease_digest=_digest("scheduler-lease"),
        resource_start_lease_digest=_digest("start-lease"),
    )
    start = AttemptStartV1(
        attempt_id=derive_attempt_id(
            run_id=run.run_id, monotonic_ordinal=1, nonce="nonce-1"
        ),
        run_id=run.run_id,
        monotonic_ordinal=1,
        nonce="nonce-1",
        concretized_execution_digest=_digest("concretized"),
        implementation_digest=_digest("implementation"),
        attempt_admission_request_digest=disposition.attempt_admission_request_digest,
        scheduler_assignment_lease_digest=disposition.scheduler_assignment_lease_digest,
        resource_start_lease_digest=_digest("start-lease"),
        post_admission_attempt_plan_digest=_digest("plan"),
        admission_disposition_digest=disposition.disposition_digest(),
        programme_work_session_assignment_dispatch_binding_digest=_digest(
            "control-plane-binding"
        ),
        parent_attempt_id=None,
        restore_checkpoint_digest=None,
        canonical_continuation_lease_digest=_digest("continuation"),
        owner_binding_digest=_digest("owners"),
        execution_side_effect_digest=_digest("side-effect"),
        initial_frontiers=AttemptFrontiersV1(0, 0, 0, 0),
    )
    start.validate_authorized_disposition(disposition)
    with pytest.raises(ContractError, match="differs from authorized admission"):
        start.validate_authorized_disposition(
            replace(disposition, reason="different receipt identity")
        )


def test_terminal_frontier_cannot_move_backwards_from_attempt_start() -> None:
    run = _run()
    attempt_id = derive_attempt_id(
        run_id=run.run_id, monotonic_ordinal=2, nonce="restore"
    )
    start = AttemptStartV1(
        attempt_id=attempt_id,
        run_id=run.run_id,
        monotonic_ordinal=2,
        nonce="restore",
        concretized_execution_digest=_digest("concretized"),
        implementation_digest=_digest("implementation"),
        attempt_admission_request_digest=_digest("request"),
        scheduler_assignment_lease_digest=None,
        resource_start_lease_digest=_digest("start-lease"),
        post_admission_attempt_plan_digest=_digest("plan"),
        admission_disposition_digest=_digest("start-authorized"),
        programme_work_session_assignment_dispatch_binding_digest=_digest(
            "control-plane-binding"
        ),
        parent_attempt_id=AttemptId("attempt-parent"),
        restore_checkpoint_digest=_digest("sealed-checkpoint"),
        canonical_continuation_lease_digest=_digest("continuation"),
        owner_binding_digest=_digest("owners"),
        execution_side_effect_digest=_digest("side-effect"),
        initial_frontiers=AttemptFrontiersV1(480, 480, 480, 0),
    )
    terminal = AttemptTerminalV1(
        attempt_id=attempt_id,
        attempt_start_digest=start.start_digest(),
        status=AttemptTerminalStatusV1.FAILED,
        final_frontiers=AttemptFrontiersV1(479, 479, 479, 0),
        result_digest=_digest("failure"),
        reason="invalid rollback",
    )
    with pytest.raises(ContractError, match="cannot move backwards"):
        terminal.validate_transition(start)


def test_attempt_id_mismatch_fails_closed() -> None:
    run = _run()
    with pytest.raises(ContractError, match="AttemptId differs"):
        AttemptStartV1(
            attempt_id=AttemptId("foreign-attempt"),
            run_id=run.run_id,
            monotonic_ordinal=1,
            nonce="nonce-1",
            concretized_execution_digest=_digest("concretized"),
            implementation_digest=_digest("implementation"),
            attempt_admission_request_digest=_digest("request"),
            scheduler_assignment_lease_digest=None,
            resource_start_lease_digest=_digest("start-lease"),
            post_admission_attempt_plan_digest=_digest("plan"),
            admission_disposition_digest=_digest("start-authorized"),
            programme_work_session_assignment_dispatch_binding_digest=_digest(
                "control-plane-binding"
            ),
            parent_attempt_id=None,
            restore_checkpoint_digest=None,
            canonical_continuation_lease_digest=_digest("continuation"),
            owner_binding_digest=_digest("owners"),
            execution_side_effect_digest=_digest("side-effect"),
            initial_frontiers=AttemptFrontiersV1(0, 0, 0, 0),
        )


def test_frontiers_fail_closed_on_impossible_progress() -> None:
    with pytest.raises(ContractError, match="computed frontier"):
        AttemptFrontiersV1(
            last_dispatched_next_tick=10,
            last_computed_next_tick=11,
            durable_checkpoint_next_tick=10,
            published_artifact_next_tick=10,
        )


def test_trajectory_checkpoint_ticks_are_sorted_unique() -> None:
    with pytest.raises(ContractError, match="sorted and unique"):
        replace(_semantic(), trajectory_checkpoint_ticks=(480, 0, 480))


def test_repeat_ordinal_cannot_be_negative() -> None:
    semantic = _semantic()
    with pytest.raises(ContractError, match="integer must be"):
        RunOccurrenceV1.create(
            experiment_id=semantic.experiment_id,
            cell_or_repeat_ordinal=-1,
            semantic_spec_digest=semantic.semantic_spec_digest(),
        )


def test_interpreted_implementation_can_have_no_native_artifact() -> None:
    interpreted = replace(_implementation(), native_artifact_digests=())
    assert interpreted.to_dict()["native_artifact_digests"] == []


def test_nonrestored_attempt_cannot_begin_after_dispatch() -> None:
    run = _run()
    with pytest.raises(ContractError, match="pre-existing progress"):
        AttemptStartV1(
            attempt_id=derive_attempt_id(
                run_id=run.run_id, monotonic_ordinal=1, nonce="nonce-1"
            ),
            run_id=run.run_id,
            monotonic_ordinal=1,
            nonce="nonce-1",
            concretized_execution_digest=_digest("concretized"),
            implementation_digest=_digest("implementation"),
            attempt_admission_request_digest=_digest("request"),
            scheduler_assignment_lease_digest=None,
            resource_start_lease_digest=_digest("start-lease"),
            post_admission_attempt_plan_digest=_digest("plan"),
            admission_disposition_digest=_digest("start-authorized"),
            programme_work_session_assignment_dispatch_binding_digest=_digest(
                "control-plane-binding"
            ),
            parent_attempt_id=None,
            restore_checkpoint_digest=None,
            canonical_continuation_lease_digest=_digest("continuation"),
            owner_binding_digest=_digest("owners"),
            execution_side_effect_digest=_digest("side-effect"),
            initial_frontiers=AttemptFrontiersV1(1, 0, 0, 0),
        )


def test_crca_resource_api_rejects_names_outside_its_allowlist() -> None:
    with pytest.raises(KeyError, match="unknown contract resource name"):
        read_resource_bytes("../07_run_attempt_edge_cases.csv")


def test_all_24_run_attempt_decisions_are_content_frozen() -> None:
    payload = read_resource_bytes(CRCA_RUN_ATTEMPT_EDGE_CASES_V1)
    assert hashlib.sha256(payload).hexdigest() == RUN_ATTEMPT_EDGE_CASE_MATRIX_SHA256
    assert len(payload.decode("utf-8").splitlines()) - 1 == RUN_ATTEMPT_EDGE_CASE_MATRIX_CASES


def test_all_31_live_ratchet_findings_have_owned_exception_records() -> None:
    reconciliation = json.loads(
        read_resource_bytes(CRCA_G0_RATCHET_RECONCILIATION_V1).decode("utf-8")
    )
    baseline = json.loads(MODULAR_RATCHET_BASELINE.read_text("utf-8"))
    classified = set(reconciliation["classified_edges"])
    assert reconciliation["reproduced_live_finding_count"] == 31
    assert len(classified) == 31
    records = {
        f"{row['source']} -> {row['target']}": row
        for row in baseline["temporary_exceptions"]
        if f"{row['source']} -> {row['target']}" in classified
    }
    assert set(records) == classified
    for record in records.values():
        assert record["owner"]
        assert record["reason"]
        assert record["expires_after_slice"]
        assert record["validation_obligation"]
