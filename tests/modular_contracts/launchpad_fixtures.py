"""Synthetic launchpad fixtures extracted unchanged from the V1 tests."""
from __future__ import annotations


import copy


import hashlib


import os


import threading


from dataclasses import fields


from pathlib import Path


from typing import Any, Mapping


import pytest


import tools.scheduler.launchpad_attempts as launchpad_attempts


import tools.scheduler.programme_driver as programme_driver


from eiel_composition.launchpad_attempt_authority import (
    AuthoritativeAttemptBundleV1,
    LaunchpadAttemptAuthorityError,
)


from emergence_contracts import (
    AdmissionAssuranceBindingV1,
    AttemptAdmissionRequestV1,
    AttemptFrontiersV1,
    AttemptRetryBasisCodeV1,
    AttemptRetryBasisV1,
    AttemptTerminalStatusV1,
    AttemptTerminalV1,
    BackendId,
    ConcretizedExecutionV1,
    ContentDigest as ExecutionDigest,
    ExecutionImplementationV1,
    ExperimentId,
    LifecycleBindingV1,
    NamedDigestV1,
    RunOccurrenceV1,
    SemanticRunSpecV1,
    canonical_sha256 as execution_sha256,
    derive_attempt_id,
)


from tools.actors.common import ActorError


from tools.actors.resource_manager import (
    assess_dead_cuda_owner_reclaim_v1,
    assess_legacy_gpu_migration_reclaim_v1,
)


from tools.scheduler.launchpad_attempts import SchedulerLaunchpadAttemptIssuerV1


from tools.scheduler.launcher import process_start_identity


from tools.scheduler.programme_contracts import (
    ContentDigest,
    LaunchpadAttemptClaimV1,
    LaunchpadResourceStartLeaseV1,
    ProgrammeContractError,
    ResourceStartLeaseEventCodeV1,
    ResourceStartLeaseEventV3,
    ResourceStartLeaseV1,
    canonical_sha256,
)


from tools.scheduler.programme_driver import ResourceStartLeaseLedgerV1


T0 = "2026-08-26T10:00:00Z"


T1 = "2026-08-26T10:01:00Z"


T2 = "2026-08-26T10:02:00Z"


T3 = "2026-08-26T10:03:00Z"


T9 = "2026-08-26T10:09:00Z"


T10 = "2026-08-26T10:10:00Z"


T11 = "2026-08-26T10:11:00Z"


TEST_PROCESS_ID = os.getpid()


TEST_PROCESS_START_IDENTITY = process_start_identity(TEST_PROCESS_ID)


assert TEST_PROCESS_START_IDENTITY is not None


def _ed(label: str) -> ExecutionDigest:
    return execution_sha256({"label": label})


def _sd(value: str | ExecutionDigest) -> ContentDigest:
    digest = _ed(value) if isinstance(value, str) else value
    return ContentDigest(algorithm=digest.algorithm, hex=digest.hex)


def _fixture(tmp_path: Path) -> dict[str, Any]:
    state = tmp_path / "state"
    issuer = SchedulerLaunchpadAttemptIssuerV1(
        state, allow_test_shared_gpu_state_root=True
    )
    native = _ed("native-artifact")
    semantic = SemanticRunSpecV1(
        experiment_id=ExperimentId("launchpad-authority-fixture"),
        system_assembly_digest=_ed("assembly"),
        semantic_parameters_digest=_ed("semantic-parameters"),
        rng_contract_digest=_ed("rng"),
        seed_digest=_ed("seed"),
        absolute_start_next_tick=0,
        absolute_target_next_tick=4,
        trajectory_checkpoint_ticks=(2, 4),
        measurement_behavior_digest=_ed("measurement"),
        control_and_branch_semantics_digest=_ed("control-branch"),
    )
    run = RunOccurrenceV1.create(
        experiment_id=semantic.experiment_id,
        cell_or_repeat_ordinal=0,
        semantic_spec_digest=semantic.semantic_spec_digest(),
    )
    attempt_id = derive_attempt_id(
        run_id=run.run_id, monotonic_ordinal=1, nonce="launchpad-initial"
    )
    implementation = ExecutionImplementationV1(
        source_commit="a" * 40,
        source_tree_digest=_ed("source-tree"),
        dirty_patch_digest=None,
        toolchain_digest=_ed("toolchain"),
        environment_digest=_ed("environment"),
        provider_id="eiel-full-evolution-cuda-native-provider-v1",
        backend_id=BackendId("cuda-sm75"),
        runtime_profile_id="eiel_full_evolution_cuda_native_v2",
        native_artifact_digests=(NamedDigestV1("kernel", native),),
        abi_digest=_ed("abi"),
        device_conformance_envelope_digest=_ed("device-envelope"),
    )
    assurance = AdmissionAssuranceBindingV1(
        intended_claim_id="TECHNICAL_ONLY_MICRO_QUALPREP",
        qualified_envelope_digest=None,
        obligation_digests=(),
        authority_reference_digests=(),
        review_reference_digests=(),
    )
    run_root = str((tmp_path / "run-root").resolve())
    lifecycle = LifecycleBindingV1(
        output_root=run_root,
        artifact_lifecycle_contract_digest=_ed("lifecycle-contract"),
        inventory_role_digests=(),
        initial_hold_digests=(),
        retention_profile_id="QUALIFICATION",
    )
    admission = AttemptAdmissionRequestV1(
        resource_requirements_digest=_ed("resource-requirements"),
        executor_constraints_digest=_ed("executor-constraints"),
        requested_restore_checkpoint_digest=None,
        observational_persistence_cadence_digest=None,
        proposed_reason="Initial CUDA-native micro QualPrep Attempt.",
    )
    execution_request = {
        "schema_version": "eiel_execution_request_v2",
        "request_id": "REQ-LAUNCHPAD-AUTHORITY",
    }
    campaign_plan = {
        "schema_version": "eiel_campaign_plan_v1",
        "plan_id": "PLAN-LAUNCHPAD-AUTHORITY",
    }
    baseline = _ed("EIEL_PRODUCTION_BASELINE_V1")
    concretized = ConcretizedExecutionV1(
        semantic_spec_digest=semantic.semantic_spec_digest(),
        implementation_digest=implementation.implementation_digest(),
        admission_assurance_digest=assurance.admission_assurance_digest(),
        lifecycle_digest=lifecycle.lifecycle_digest(),
        attempt_admission_request_digest=admission.attempt_admission_request_digest(),
        campaign_plan_digest=execution_sha256(campaign_plan),
        bound_execution_request_digest=execution_sha256(execution_request),
        run_contract_digest=_ed("run-contract"),
        baseline_digest=baseline,
        route_contract_digest=_ed("route-contract"),
        assembly_digest=_ed("assembly"),
        qualification_key_digest=_ed("qualification-key"),
    )
    retry = AttemptRetryBasisV1(
        run_id=run.run_id,
        candidate_attempt_id=attempt_id,
        prior_attempt_id=None,
        code=AttemptRetryBasisCodeV1.INITIAL_EXECUTION,
        reason="First physical occurrence.",
        semantic_spec_digest=semantic.semantic_spec_digest(),
        prior_concretized_execution_digest=None,
        candidate_concretized_execution_digest=concretized.concretized_execution_digest(),
        basis_evidence_digests=(),
        implementation_conformance_digest=None,
    )
    frontiers = AttemptFrontiersV1(0, 0, 0, 0)
    preflight = {
        "schema_version": "crca_launchpad_preflight_authority_projection_v1",
        "provider_id": implementation.provider_id,
        "route_id": "tensor_world_cuda_full_evolution_v2",
        "device_id": "GPU-0000:01:00.0",
        "native_artifact_digest": native.to_dict(),
        "batch_size": 136,
        "ticks_per_dispatch": 64,
        "execution_mode": "CUDA_GRAPH",
    }
    owner_digest = _sd("provider-process-owner")
    claim = LaunchpadAttemptClaimV1.create(
        scheduler_owner_id="tools.scheduler.launchpad_attempts",
        scheduler_owner_digest=issuer.owner_digest(),
        execution_owner_id="provider-process-owner",
        execution_owner_digest=owner_digest,
        run_id=str(run.run_id),
        attempt_id=str(attempt_id),
        lineage_scope_digest=_sd("lineage-scope"),
        execution_request_digest=_sd(execution_sha256(execution_request)),
        campaign_plan_digest=_sd(execution_sha256(campaign_plan)),
        source_digest=_sd(implementation.source_tree_digest),
        baseline_digest=_sd(baseline),
        preflight_receipt_digest=canonical_sha256(preflight),
        lifecycle_binding_digest=_sd(lifecycle.lifecycle_digest()),
        lifecycle_contract_digest=_sd(lifecycle.artifact_lifecycle_contract_digest),
        run_root=run_root,
        provider_id=implementation.provider_id,
        route_id=preflight["route_id"],
        device_id=preflight["device_id"],
        native_artifact_digest=_sd(native),
        batch_size=136,
        ticks_per_dispatch=64,
        execution_mode="CUDA_GRAPH",
        semantic_run_spec_digest=_sd(semantic.semantic_spec_digest()),
        run_occurrence_digest=canonical_sha256(run.to_dict()),
        execution_implementation_digest=_sd(implementation.implementation_digest()),
        concretized_execution_digest=_sd(concretized.concretized_execution_digest()),
        attempt_admission_request_digest=_sd(
            admission.attempt_admission_request_digest()
        ),
        retry_basis_digest=_sd(retry.basis_digest()),
        post_admission_attempt_plan_digest=_sd("post-admission-plan"),
        programme_binding_digest=_sd("programme-binding"),
        initial_frontiers_digest=canonical_sha256(frontiers.to_dict()),
        created_at_utc=T0,
    )
    prepare = {
        "claim": claim,
        "execution_request": execution_request,
        "campaign_plan": campaign_plan,
        "preflight_receipt": preflight,
        "lifecycle_binding": lifecycle.to_dict(),
        "semantic_run_spec": semantic.to_dict(),
        "run_occurrence": run.to_dict(),
        "execution_implementation": implementation.to_dict(),
        "concretized_execution": concretized.to_dict(),
        "attempt_admission_request": admission.to_dict(),
        "retry_basis": retry.to_dict(),
        "initial_frontiers": frontiers.to_dict(),
        "monotonic_ordinal": 1,
        "nonce": "launchpad-initial",
        "parent_attempt_id": None,
        "resource_ids": ("gpu:GPU-0000:01:00.0",),
        "resource_scope_digest": canonical_sha256(
            {
                "schema_version": "eiel_exclusive_gpu_scope_v1",
                "resource_ids": ["gpu:GPU-0000:01:00.0"],
                "run_id": str(run.run_id),
            }
        ),
        "resource_observation_id": "ROBS-LAUNCHPAD-EXACT",
        "resource_observation": {
            "schema_version": "eiel_resource_observation_v1",
            "device_id": "GPU-0000:01:00.0",
        },
        "resource_observed_at_utc": T0,
        "resource_recommendation_id": "RADM-LAUNCHPAD-EXACT",
        "resource_recommendation": {
            "schema_version": "eiel_resource_recommendation_v1",
            "decision": "ADMIT",
        },
        "live_provider_check": {
            "schema_version": "crca_launchpad_live_provider_check_v1",
            "available": True,
            "device_id": "GPU-0000:01:00.0",
        },
        "at_utc": T0,
        "start_window_seconds": 180,
        "resource_ttl_seconds": 300,
        "continuation_ttl_seconds": 300,
    }
    return {
        "state": state,
        "issuer": issuer,
        "claim": claim,
        "owner_digest": owner_digest,
        "semantic": semantic,
        "run": run,
        "concretized": concretized,
        "attempt_id": str(attempt_id),
        "prepare": prepare,
    }


def _prepare_start(fixture: dict[str, Any]) -> dict[str, Any]:
    fixture["issuer"].prepare(**fixture["prepare"])
    return fixture["issuer"].start(
        attempt_id=fixture["attempt_id"],
        process_owner_id="provider-process-owner",
        process_owner_digest=fixture["owner_digest"],
        process_id=TEST_PROCESS_ID,
        process_start_identity=TEST_PROCESS_START_IDENTITY,
        at_utc=T1,
    )


def _terminal_for_started(
    started: Mapping[str, Any],
    *,
    status: AttemptTerminalStatusV1 = AttemptTerminalStatusV1.INTERRUPTED,
    frontiers: AttemptFrontiersV1 | None = None,
) -> AttemptTerminalV1:
    start = AuthoritativeAttemptBundleV1.from_dict(started["bundle"]).attempt_start
    return AttemptTerminalV1(
        attempt_id=start.attempt_id,
        attempt_start_digest=start.start_digest(),
        status=status,
        final_frontiers=frontiers or AttemptFrontiersV1(0, 0, 0, 0),
        result_digest=_ed("terminal-fixture"),
        reason="bounded exact terminal",
    )

