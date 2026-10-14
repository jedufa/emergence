from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.campaigns.execution_front_door_v1 import (
    ExecutionFrontDoorError,
    _workhorse_terminal_lifecycle_artifacts_v1,
)
from eiel_composition.cuda_full_evolution_v2_launchpad_adapter import (
    CanonicalCudaFullEvolutionV2LaunchpadAdapterV1,
    CudaFullEvolutionV2LaunchpadAdapterError,
)
from emergence_contracts import (
    AttemptFrontiersV1,
    AttemptId,
    AttemptTerminalStatusV1,
    AttemptTerminalV1,
    ContentDigest,
    RunId,
    canonical_sha256,
)
from engine.perf.cuda_full_evolution_v2.workhorse.contract import (
    BACKEND_FUSED_CTA_TEMPORAL_V2,
    BACKEND_REFERENCE_PHASED_V2,
    FullEvolutionCudaWorkhorseContractV2,
    SUPPORTED_WORKHORSE_SHAPES_V2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.semantic_cases import (
    workhorse_semantic_case_operation_contract_v2,
)


SHAPE_ID = SUPPORTED_WORKHORSE_SHAPES_V2[0].shape_id


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    case_id: str,
) -> tuple[
    CanonicalCudaFullEvolutionV2LaunchpadAdapterV1,
    SimpleNamespace,
    dict[str, object],
    dict[str, object],
]:
    semantic_case = workhorse_semantic_case_operation_contract_v2(
        case_id=case_id,
        shape_id=SHAPE_ID,
        semantic_temporal_k=2,
    )
    law = semantic_case["case_law"]
    backend_id = (
        BACKEND_REFERENCE_PHASED_V2
        if law["artifact_role"] == "REFERENCE"
        else BACKEND_FUSED_CTA_TEMPORAL_V2
    )
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=SHAPE_ID,
        backend_id=backend_id,
        B=law["B"],
        temporal_k=law["dispatch_temporal_k"],
        total_ticks=law["total_ticks"],
        start_next_tick=law["start_next_absolute_tick"],
    ).to_dict()
    source_identity = "7" * 64 if law["requires_source_checkpoint"] else None
    external_source = {"source": "fixture"} if source_identity else None
    science = {"binding_sha256": "8" * 64}
    operation = {
        "operation": "SEMANTIC_EVIDENCE_CASE",
        "semantic_case_contract": semantic_case,
        "contract": contract,
        "artifact": {
            "sha256": "a" * 64,
            "sealed_artifact_identity_sha256": "b" * 64,
        },
        "science_checkpoint_compatibility_binding": science,
        "science_continuation_contract_sha256": "c" * 64,
        "source_checkpoint_identity_sha256": source_identity,
        "external_source_checkpoint_binding": external_source,
    }
    plan: dict[str, object] = {
        "plan_sha256": "d" * 64,
        "analysis": {"extension": {"contract": operation}},
        "physical_batches": [{"batch_id": "semantic-case"}],
    }
    run_root = tmp_path / "run"
    batch_root = run_root / "batches" / "semantic-case"
    batch_root.mkdir(parents=True)
    fragment_path = batch_root / "semantic_case_fragment.json"
    fragment_path.write_text("{}", encoding="ascii")
    fragment = {
        "fragment_body_sha256": "e" * 64,
        "canonical_parent_binding": {
            "run_id": "RUN-SEMANTIC",
            "attempt_id": "ATTEMPT-SEMANTIC",
            "operation_contract_sha256": semantic_case[
                "operation_contract_sha256"
            ],
        },
        "base_checkpoint": (
            {
                "identity": {
                    "base_checkpoint_identity_sha256": "f" * 64,
                }
            }
            if law["produces_base_checkpoint"]
            else None
        ),
    }
    monkeypatch.setattr(
        "engine.perf.cuda_full_evolution_v2.workhorse.semantic_cases."
        "reopen_workhorse_semantic_case_fragment_v2",
        lambda _path: (fragment, {}),
    )
    terminal = semantic_case["attempt_terminal_contract"]
    expected_status = terminal["attempt_terminal_status"]
    occurrence_paths: list[str] = []
    produced = None
    if law["produces_base_checkpoint"]:
        produced = {"checkpoint": "fixture"}
        for name in ("occurrence.json", "commit.json"):
            target = batch_root / name
            target.write_text("{}", encoding="ascii")
            occurrence_paths.append(str(target.resolve()))
    result = {
        "schema_version": "eiel_workhorse_semantic_case_provider_result_v2",
        "operation": "SEMANTIC_EVIDENCE_CASE",
        "case_id": case_id,
        "shape_id": contract["shape_id"],
        "backend_id": contract["backend_id"],
        "B": contract["B"],
        "temporal_k": contract["temporal_k"],
        "semantic_temporal_k": 2,
        "artifact_sha256": "a" * 64,
        "sealed_artifact_identity_sha256": "b" * 64,
        "expected_attempt_terminal_status": expected_status,
        "attempt_terminal_contract": terminal,
        "provisional_result_class": (
            "SUCCESS_PENDING_COMPLETED_TERMINAL"
            if expected_status == "COMPLETED"
            else "EXPECTED_NEGATIVE_PENDING_FAILED_TERMINAL"
        ),
        "provider_terminalization_action": (
            "RETURN_FOR_COMPLETED_TERMINALIZATION"
            if expected_status == "COMPLETED"
            else "RETURN_FOR_FAILED_ATTEMPT_TERMINALIZATION"
        ),
        "case_fragment_path": str(fragment_path.resolve()),
        "case_fragment_file_sha256": _sha256_file(fragment_path),
        "case_fragment_body_sha256": "e" * 64,
        "source_checkpoint_identity_sha256": source_identity,
        "external_source_checkpoint_binding": external_source,
        "science_checkpoint_compatibility_binding": science,
        "produced_external_source_checkpoint_binding": produced,
        "checkpoint_occurrence_artifact_paths": occurrence_paths,
        "absolute_frontiers": {
            "start_next_tick": contract["start_next_tick"],
            "target_next_tick": contract["start_next_tick"] + contract["total_ticks"],
            "committed_next_tick": terminal["final_frontiers"][
                "published_artifact_next_tick"
            ],
        },
    }
    (batch_root / "manifest.json").write_text(
        json.dumps({"status": "completed", "result": result}),
        encoding="ascii",
    )
    authority = SimpleNamespace(
        raw={"claim": {"run_root": str(run_root)}},
        attempt_start=SimpleNamespace(attempt_id="ATTEMPT-SEMANTIC"),
        run_occurrence=SimpleNamespace(run_id="RUN-SEMANTIC"),
    )
    adapter = CanonicalCudaFullEvolutionV2LaunchpadAdapterV1(
        repo_root=tmp_path,
        state_root=tmp_path / "state",
        allow_test_state_root=True,
    )
    if produced is not None:
        monkeypatch.setattr(
            adapter,
            "validate_workhorse_external_checkpoint_source_v2",
            lambda value, *, expected_identity_sha256: value,
        )
    return adapter, authority, plan, result


@pytest.mark.parametrize(
    "case_id",
    [
        "parity_reference_k1_sequence",
        "checkpoint_same_restore_sequence",
        "fresh_restore_continuation",
        "invalid_restore_failstop",
        "transaction_failstop",
        "event_reuse_canonicalization_regression",
    ],
)
def test_semantic_provider_projection_uses_exact_closed_case_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
) -> None:
    adapter, authority, plan, result = _semantic_fixture(
        tmp_path, monkeypatch, case_id=case_id
    )
    projection = adapter._semantic_provider_terminal_projection_v2(
        authority=authority,
        campaign_plan=plan,
        execution_receipt={"status": "completed", "plan_sha256": "d" * 64},
    )
    assert projection is not None
    assert projection["attempt_terminal_contract"] == result[
        "attempt_terminal_contract"
    ]


def test_semantic_provider_terminal_echo_substitution_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, authority, plan, result = _semantic_fixture(
        tmp_path, monkeypatch, case_id="transaction_failstop"
    )
    tampered = deepcopy(result)
    tampered["attempt_terminal_contract"]["final_frontiers"][
        "last_computed_next_tick"
    ] += 1
    manifest_path = (
        tmp_path / "run" / "batches" / "semantic-case" / "manifest.json"
    )
    manifest_path.write_text(
        json.dumps({"status": "completed", "result": tampered}),
        encoding="ascii",
    )
    with pytest.raises(
        CudaFullEvolutionV2LaunchpadAdapterError,
        match="provisional result differs",
    ):
        adapter._semantic_provider_terminal_projection_v2(
            authority=authority,
            campaign_plan=plan,
            execution_receipt={"status": "completed", "plan_sha256": "d" * 64},
        )


def test_semantic_scheduler_terminal_crash_replay_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, fixture_authority, plan, _result = _semantic_fixture(
        tmp_path, monkeypatch, case_id="transaction_failstop"
    )
    operation = plan["analysis"]["extension"]["contract"]
    planned = operation["semantic_case_contract"]["attempt_terminal_contract"]
    terminal_frontiers = planned["final_frontiers"]
    attempt_id = AttemptId("ATTEMPT-SEMANTIC")
    start_digest = ContentDigest("sha256", "1" * 64)
    receipt = {"status": "completed", "plan_sha256": "d" * 64}
    terminal = AttemptTerminalV1(
        attempt_id=attempt_id,
        attempt_start_digest=start_digest,
        status=AttemptTerminalStatusV1.FAILED,
        final_frontiers=AttemptFrontiersV1(
            terminal_frontiers["last_dispatched_next_tick"],
            terminal_frontiers["last_computed_next_tick"],
            terminal_frontiers["durable_checkpoint_next_tick"],
            terminal_frontiers["published_artifact_next_tick"],
        ),
        result_digest=canonical_sha256(receipt),
        reason="Canonical Launchpad semantic expected-negative operation failed.",
    ).to_dict()
    terminal_bundle = {
        "attempt_id": str(attempt_id),
        "attempt_terminal": terminal,
        "continuation_event": {"publication_digest": None},
    }
    terminal_path = tmp_path / "scheduler-terminal.json"
    terminal_path.write_text(json.dumps(terminal_bundle), encoding="ascii")
    terminal_digest = canonical_sha256(terminal_bundle).to_dict()

    class Issuer:
        terminalize_called = False

        @staticmethod
        def terminal_attempt_record(_attempt_id: str) -> dict[str, object]:
            return {
                "terminal_bundle": terminal_bundle,
                "terminal_bundle_path": str(terminal_path),
                "terminal_bundle_digest": terminal_digest,
                "restore_source_attempt_record": {},
            }

        def terminalize(self, **_values: object) -> None:
            self.terminalize_called = True
            raise AssertionError("terminal replay must not mutate Scheduler state")

    issuer = Issuer()
    adapter.issuer = issuer
    authority = SimpleNamespace(
        raw={
            "claim": {
                **fixture_authority.raw["claim"],
                "provider_id": (
                    "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
                ),
                "execution_owner_digest": {
                    "algorithm": "sha256",
                    "hex": "2" * 64,
                },
            }
        },
        semantic_run_spec=SimpleNamespace(
            absolute_start_next_tick=operation["contract"]["start_next_tick"],
            absolute_target_next_tick=(
                operation["contract"]["start_next_tick"]
                + operation["contract"]["total_ticks"]
            ),
        ),
        attempt_start=SimpleNamespace(
            attempt_id=attempt_id,
            start_digest=lambda: start_digest,
        ),
        run_occurrence=SimpleNamespace(run_id=RunId("RUN-SEMANTIC")),
    )
    monkeypatch.setattr(adapter, "_authority", lambda _path: authority)

    replay = adapter.terminalize_attempt_authority_v1(
        attempt_authority_path=tmp_path / "unused-pointer.json",
        campaign_plan=plan,
        execution_status="completed",
        execution_receipt=receipt,
        result_binding={},
    )
    assert replay["status"] == "TERMINAL"
    assert replay["workhorse_terminal_binding"]["attempt_terminal_status"] == (
        "FAILED"
    )
    assert issuer.terminalize_called is False


def test_premature_valid_negative_without_terminal_receipt_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ExecutionFrontDoorError,
        match="lacks its terminal semantic receipt",
    ):
        _workhorse_terminal_lifecycle_artifacts_v1(
            tmp_path,
            {"result_class": "VALID_NEGATIVE"},
        )


def test_tampered_valid_negative_terminal_receipt_fails_closed(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "workhorse-semantic-case-receipt.json"
    receipt.write_text("{}", encoding="ascii")
    digest = _sha256_file(receipt)
    lifecycle = {
        "artifact_family": "qualification.workhorse_semantic_case_v2",
        "retention_class": "R3",
        "owner_writer": (
            "cuda_full_evolution_workhorse_v2_single_operation_owner"
        ),
        "release_predicate": "technical semantic case retained",
    }
    finalized = {
        "operation": "SEMANTIC_EVIDENCE_CASE",
        "result_class": "VALID_NEGATIVE",
        "final_receipt_path": str(receipt),
        "final_receipt_file_sha256": digest,
        "final_receipt_content_sha256": "a" * 64,
        "semantic_case_receipt_path": str(receipt),
        "semantic_case_receipt_file_sha256": digest,
        "semantic_case_receipt_content_sha256": "a" * 64,
        "lifecycle_binding": lifecycle,
    }
    with pytest.raises(
        ExecutionFrontDoorError,
        match="terminal semantic receipt cannot reopen",
    ):
        _workhorse_terminal_lifecycle_artifacts_v1(
            tmp_path,
            {
                "result_class": "VALID_NEGATIVE",
                "workhorse_terminalized_operation": finalized,
                "workhorse_terminalized_semantic_case": finalized,
                "crca_attempt_terminal": {
                    "workhorse_terminal_binding": {
                        "attempt_terminal_status": "FAILED"
                    }
                },
            },
        )
