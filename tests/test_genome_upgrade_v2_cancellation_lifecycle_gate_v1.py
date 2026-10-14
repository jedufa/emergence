from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.scripts.genome_upgrade_v2_cancellation_lifecycle_gate_v1 import (
    CancellationLifecycleGateError,
    execute,
)


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _cancel_receipt(
    artifact_manifest: dict[str, Any],
    *,
    request_phase: str,
) -> dict[str, Any]:
    request = {
        "request_id": f"request-{request_phase}",
        "controller_id": f"controller-{request_phase}",
        "disposition": "accepted",
        "state_observed": "tick_running",
    }
    cancellation = {
        "request_observation_cursor": "generation:2:tick:0",
    }
    return {
        "ok": True,
        "decision": "GENOME-V2-NATIVE-CANCELLATION-PASS",
        "artifacts": {"cuda_manifest": artifact_manifest},
        "claims": {
            "checkpoint_on_cancel_exact": True,
            "fresh_owner_restore_after_cancel_exact": True,
        },
        "cancel_arrival_observation": {
            "request_phase": request_phase,
            "qualification_binding_hold_ms": (
                12 if request_phase == "active_generation_transaction" else 0
            ),
            "manifest": {
                "status": (
                    "tick_running"
                    if request_phase == "active_generation_transaction"
                    else "inter_tick_control"
                ),
                "generation_transaction_open": True,
                "inter_tick_control_window_open": (
                    request_phase == "inter_tick_control"
                ),
                "inter_tick_control_wait_timeout_ms": 10,
            },
        },
        "cancel_request_receipt": request,
        "pre_submission_cancel_request_receipt": {
            "request_id": "before-submission",
            "controller_id": "unsubmitted-controller",
            "disposition": "not_running",
            "state_observed": "initialized",
        },
        "pre_dispatch_cancel_request_receipt": {
            "request_id": "before-dispatch",
            "controller_id": "main-controller",
            "disposition": "not_running",
            "state_observed": "initialized",
        },
        "repeated_cancel_request_receipt": {
            "request_id": "repeat",
            "controller_id": request["controller_id"],
            "disposition": "already_requested",
        },
        "terminal_result": {
            "terminal_status": "cancelled",
            "cancellation": cancellation,
        },
        "boundary_acknowledgement": {
            "terminal_status": "cancelled",
            "cancellation": cancellation,
        },
        "parity": {
            "checkpoint_parity": {
                "exact": True,
                "candidate_projection_digest": "a" * 64,
                "baseline_projection_digest": "a" * 64,
                "fresh_owner_projection_digest": "a" * 64,
            }
        },
        "no_post_terminal_mutation": {
            "ok": True,
            "late_cancel_receipt": {
                "request_id": "late",
                "controller_id": request["controller_id"],
                "disposition": "already_terminal",
            },
        },
    }


def _atomic_receipt(
    artifact_manifest: dict[str, Any],
    cases: tuple[str, ...],
    *,
    post_commit: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    return {
        "qualification_status": "PASS",
        "artifact": {"manifest": artifact_manifest},
        "failure_injection_rows": [
            {
                "case_id": case_id,
                "status": "PASS",
                "expected_authority": (
                    "one_complete_committed_transaction"
                    if case_id in post_commit
                    else "exact_pre_commit_publication_state"
                ),
                "all_state_exact": case_id not in post_commit,
                "committed_transaction_exact": case_id in post_commit,
            }
            for case_id in cases
        ],
    }


def _inputs(tmp_path: Path) -> dict[str, Path | str]:
    artifact = tmp_path / "candidate.so"
    artifact.write_bytes(b"candidate")
    from hashlib import sha256

    artifact_hash = sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "artifact": str(artifact),
        "artifact_sha256": artifact_hash,
        "artifact_identity": {
            "runtime_resource_receipt": {
                "authoritative_state_build_provenance_v1": {
                    "build_source_commit": "b" * 40,
                    "compiled_source_snapshot_sha256": "c" * 64,
                }
            }
        },
    }
    return {
        "artifact_manifest_path": _write(tmp_path / "manifest.json", manifest),
        "control_receipt_path": _write(
            tmp_path / "control.json",
            _cancel_receipt(manifest, request_phase="inter_tick_control"),
        ),
        "active_receipt_path": _write(
            tmp_path / "active.json",
            _cancel_receipt(
                manifest,
                request_phase="active_generation_transaction",
            ),
        ),
        "birth_receipt_path": _write(
            tmp_path / "birth.json",
            _atomic_receipt(
                manifest,
                (
                    "after_destination_reservation",
                    "after_genotype_staging",
                    "immediately_after_commit_before_reporting",
                ),
                post_commit=frozenset(
                    {"immediately_after_commit_before_reporting"}
                ),
            ),
        ),
        "turnover_receipt_path": _write(
            tmp_path / "turnover.json",
            _atomic_receipt(
                manifest,
                (
                    "mid_cull",
                    "after_child_staging",
                    "after_turnover_commit_before_checkpoint",
                ),
            ),
        ),
        "checkpoint_receipt_path": _write(
            tmp_path / "checkpoint.json",
            {
                "qualification_status": "PASS",
                "artifact_sha256": artifact_hash,
            },
        ),
        "observability_receipt_path": _write(
            tmp_path / "observability.json",
            {
                "ok": True,
                "decision": "GENOME-V2-OBSERVABILITY-PASS",
                "source_artifact_sha256": artifact_hash,
            },
        ),
        "output_dir": tmp_path / "out",
        "package_candidate_id": "genome-v2-package-candidate-test",
        "route_requirement_id": "genome-v2-route-requirement-test",
    }


def test_gate_emits_all_fifteen_same_artifact_lifecycle_cases(
    tmp_path: Path,
) -> None:
    summary = execute(**_inputs(tmp_path))  # type: ignore[arg-type]

    assert summary["ok"] is True
    assert summary["case_count"] == 15
    assert summary["route_qualification_status"] == "pending_t3"
    output = tmp_path / "out"
    assert (output / "genome_v2_cancellation_matrix.csv").is_file()
    assert (output / "genome_v2_historical_cancellation_supersession.json").is_file()


def test_gate_rejects_mixed_artifact_receipts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    checkpoint_path = inputs["checkpoint_receipt_path"]
    assert isinstance(checkpoint_path, Path)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["artifact_sha256"] = "d" * 64
    _write(checkpoint_path, checkpoint)

    with pytest.raises(
        CancellationLifecycleGateError,
        match="checkpoint receipt is not a same-artifact PASS",
    ):
        execute(**inputs)  # type: ignore[arg-type]


def test_gate_rejects_post_commit_row_without_committed_transaction_exact(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    birth_path = inputs["birth_receipt_path"]
    assert isinstance(birth_path, Path)
    birth = json.loads(birth_path.read_text(encoding="utf-8"))
    for row in birth["failure_injection_rows"]:
        if row["case_id"] == "immediately_after_commit_before_reporting":
            row["committed_transaction_exact"] = False
            break
    _write(birth_path, birth)

    with pytest.raises(
        CancellationLifecycleGateError,
        match="birth atomicity matrix has a failing row",
    ):
        execute(**inputs)  # type: ignore[arg-type]
