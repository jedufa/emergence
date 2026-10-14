from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest

from engine.campaigns.artifact_lifecycle import (
    DEBUG_FORENSIC,
    build_artifact_disposition_manifest_v1,
    build_sealed_artifact_inventory_row_v1,
    lifecycle_profile_contract_v1,
    validate_run_artifact_lifecycle_contract_v1,
)
from tools.scripts import eiel_evidence_certificate_reclaim as reclaim
from tools.scripts import eiel_runtime_storage_prune as prune


DRIVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "_artifacts/eiel-storage-availability-capsules-20260827"
    / "apply_debug_closeout_matrix.py"
)


def _load_driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "eiel_apply_debug_closeout_matrix", DRIVER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Any]:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    git_identity = {
        "repository_root": "/repo/emergence",
        "commit": "1" * 40,
        "working_tree_clean": True,
        "working_tree_status_lines": [],
    }
    raw_path = runtime_root / "scratch/raw.bin"
    request_path = runtime_root / "execution_request.json"
    result_path = runtime_root / "execution_result.json"
    parity_path = runtime_root / "reporting_parity_receipt.json"
    recipe_path = (
        runtime_root
        / "scratch/batch-0000/gpu_tick_anatomy_v1/source_descriptor_recipe.json"
    )
    payloads = {
        raw_path: b"raw-transient-diagnostic",
        request_path: (
            json.dumps(
                {
                    "source_identity": {
                        "git": git_identity,
                        "campaign_spec_path": "/repo/spec.json",
                        "campaign_spec_sha256": "2" * 64,
                        "compiled_plan_sha256": "3" * 64,
                    }
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        result_path: (
            json.dumps(
                {
                    "source_identity": {"git": git_identity},
                    "execution_receipt": {
                        "provider_id": "fixture-provider",
                        "profile_id": "fixture-runtime-profile",
                        "execution_route_sha256": "4" * 64,
                    },
                    "baseline": {
                        "profile_id": "EIEL_PRODUCTION_BASELINE_V1",
                        "overall_status": "CONFORMANT",
                    },
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        parity_path: b'{"status":"passed"}\n',
        recipe_path: b'{"recipe":"fresh-output-only"}\n',
    }
    inventory = []
    for path, payload in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        inventory.append(
            build_sealed_artifact_inventory_row_v1(
                run_root=runtime_root,
                path=path,
                artifact_family="matrix-driver-fixture",
                retention_class="R1" if path == raw_path else "R4",
                owner_writer="matrix-driver-fixture",
            )
        )
    contract = lifecycle_profile_contract_v1(
        DEBUG_FORENSIC,
        total_max_bytes=10_000_000,
        checkpoint_schedule=[1],
    )
    contract["retention"]["unknown_artifact_scope"] = "exact_object"
    contract.pop("contract_sha256", None)
    contract = validate_run_artifact_lifecycle_contract_v1(contract)
    manifest = build_artifact_disposition_manifest_v1(
        {
            "campaign_id": "matrix-fixture-campaign",
            "plan_sha256": "5" * 64,
            "artifact_lifecycle": contract,
        },
        run_root=runtime_root,
        terminal_status="completed",
        live_reference_status="active",
        audit_hold_status="not_required",
        qualification_hold_status="not_required",
        artifact_inventory=inventory,
        inventory_complete=True,
    )
    manifest_path = runtime_root / "artifact_disposition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (runtime_root / "artifact_lifecycle_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    selected_bytes = raw_path.stat().st_size
    target = {
        "campaign_id": manifest["campaign_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "expected_selected_bytes": selected_bytes,
        "decision": "RAW_DIAGNOSTIC_ABANDONED",
        "comparison_authority": None,
    }
    matrix = {
        "schema_version": "eiel_debug_forensic_cleanup_target_matrix_v1",
        "programme_id": "matrix-driver-programme",
        "work_id": "matrix-driver-work",
        "claim_class": "TECHNICAL_ONLY",
        "reviewer": {
            "identity": "same-identity-reviewer",
            "independence": "REVIEWER_INDEPENDENCE_UNAVAILABLE",
        },
        "selection": {
            "retention_classes": ["R0", "R1"],
            "expected_target_count": 1,
            "expected_selected_bytes": selected_bytes,
            "evidence_class": "RAW_TRANSIENT_EVIDENCE",
            "rationale": "raw bytes are superseded by the compact closeout",
        },
        "supporting_artifacts": [
            {"role": "execution-request", "path": "execution_request.json"},
            {"role": "execution-result", "path": "execution_result.json"},
            {
                "role": "reporting-parity",
                "path": "reporting_parity_receipt.json",
            },
            {
                "role": "source-recipe",
                "path": (
                    "scratch/batch-0000/gpu_tick_anatomy_v1/"
                    "source_descriptor_recipe.json"
                ),
            },
        ],
        "common_limitations": [
            "same-identity reduced-assurance review",
            "technical-only diagnostic conclusion",
        ],
        "no_claims": [
            "No scientific claim is granted.",
            "No qualification claim is granted.",
        ],
        "operator_retention_directive": {
            "authority": "operator matrix A",
            "directive": "retain the compact report and release exact raw rows",
            "scope": "this completed fixture diagnostic",
        },
        "review_profiles": {
            "raw-diagnostic-abandoned": {
                "reviewer_status": "accepted_with_limitations",
                "bounded_claim": "the abandoned diagnostic is exactly accounted for",
                "conclusion": "the abandoned raw diagnostic supports no accepted result",
                "criteria": [
                    {
                        "criterion": "accepted result availability",
                        "status": "failed",
                        "detail": "no accepted result exists for this attempt",
                    },
                    {
                        "criterion": "complete inventory accounting",
                        "status": "passed",
                        "detail": "the exact producer manifest accounts for every row",
                    },
                ],
                "reproduction_expected_status": "NOT_ESTABLISHED",
            }
        },
        "review_profile_routing": [
            {
                "decision": "RAW_DIAGNOSTIC_ABANDONED",
                "comparison_status": None,
                "review_profile": "raw-diagnostic-abandoned",
            }
        ],
        "abandoned_target_reviews": {
            manifest["campaign_id"]: {
                "obligation_id": "matrix-driver-abandoned-obligation",
                "conclusion": "the historical attempt has no accepted result",
                "rationale": "a fresh request is required for another attempt",
            }
        },
        "comparison_authorities": {},
        "targets": [target],
        "hard_exclusions": [
            {"campaign_id": "protected-fixture", "reason": "hard exclusion"}
        ],
    }
    return {
        "runtime_root": runtime_root,
        "raw_path": raw_path,
        "manifest": manifest,
        "matrix": matrix,
        "target": target,
        "selected_bytes": selected_bytes,
    }


def test_changed_authenticated_matrix_cannot_resume_prepared_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    driver = _load_driver()
    fixture = _fixture(tmp_path)
    matrix_path = tmp_path / "matrix.json"
    build_root = tmp_path / "build"
    permanent_root = tmp_path / "permanent"
    repo_root = Path(__file__).resolve().parents[1]
    matrix_path.write_text(
        json.dumps(fixture["matrix"], indent=2, sort_keys=True) + "\n"
    )
    matrix_a_sha256 = _file_sha256(matrix_path)
    monkeypatch.setattr(driver, "EXPECTED_TARGET_COUNT", 1)
    monkeypatch.setattr(driver, "EXPECTED_SELECTED_BYTES", fixture["selected_bytes"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(DRIVER_PATH),
            "--matrix",
            str(matrix_path),
            "--expected-matrix-sha256",
            matrix_a_sha256,
            "--repo-root",
            str(repo_root),
            "--build-root",
            str(build_root),
            "--permanent-root",
            str(permanent_root),
        ],
    )
    assert driver.main() == 0
    capsys.readouterr()

    request_path = (
        build_root
        / fixture["manifest"]["campaign_id"]
        / "bundle/diagnostic_closeout_request.json"
    )
    closeout_root = (
        fixture["runtime_root"]
        / "control/evidence-reclamation/storage-availability-20260828"
    )
    close_dry = driver._json_command(
        [
            sys.executable,
            str(repo_root / "tools/scripts/eiel_evidence_certificate_reclaim.py"),
            "close-gate",
            "--request",
            str(request_path),
            "--evidence-root",
            str(fixture["runtime_root"]),
            "--output-root",
            str(closeout_root),
            "--permanent-root",
            str(permanent_root),
            "--max-permanent-capsule-bytes",
            str(driver.PERMANENT_CAPSULE_LIMIT_BYTES),
            "--max-permanent-root-bytes",
            str(driver.PERMANENT_ROOT_LIMIT_BYTES),
        ],
        cwd=repo_root,
    )
    assert close_dry["selected_bytes"] == fixture["selected_bytes"]
    intent_path = closeout_root / "reclamation_intent.json"
    assert json.loads(intent_path.read_text())["status"] == "PREPARED"
    assert fixture["raw_path"].is_file()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(DRIVER_PATH),
            "--matrix",
            str(matrix_path),
            "--expected-matrix-sha256",
            matrix_a_sha256,
            "--repo-root",
            str(repo_root),
            "--build-root",
            str(build_root),
            "--permanent-root",
            str(permanent_root),
        ],
    )
    assert driver.main() == 0
    same_matrix_resume = json.loads(capsys.readouterr().out)
    assert same_matrix_resume["results"][0]["status"] == "ALREADY_PREPARED"
    assert fixture["raw_path"].is_file()

    matrix_b = copy.deepcopy(fixture["matrix"])
    matrix_b["operator_retention_directive"]["authority"] = "operator matrix B"
    matrix_path.write_text(json.dumps(matrix_b, indent=2, sort_keys=True) + "\n")
    matrix_b_sha256 = _file_sha256(matrix_path)
    close_apply_commands: list[list[str]] = []
    original_json_command = driver._json_command

    def record_json_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
        if "close-gate" in command and "--apply" in command:
            close_apply_commands.append(command)
        return original_json_command(command, cwd=cwd)

    monkeypatch.setattr(driver, "_json_command", record_json_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(DRIVER_PATH),
            "--matrix",
            str(matrix_path),
            "--expected-matrix-sha256",
            matrix_b_sha256,
            "--repo-root",
            str(repo_root),
            "--build-root",
            str(build_root),
            "--permanent-root",
            str(permanent_root),
            "--apply",
        ],
    )
    with pytest.raises(
        driver.CloseoutMatrixError,
        match="not bound to the current operator-authenticated matrix",
    ):
        driver.main()
    assert close_apply_commands == []
    assert fixture["raw_path"].is_file()
    assert json.loads(intent_path.read_text())["status"] == "PREPARED"

    matrix_path.write_text(
        json.dumps(fixture["matrix"], indent=2, sort_keys=True) + "\n"
    )
    real_unlink = prune.os.unlink
    failed = False

    def fail_after_staging(path, *call_args, **call_kwargs):
        nonlocal failed
        if call_kwargs.get("dir_fd") is not None and not failed:
            failed = True
            raise OSError("simulated crash after the selected row was staged")
        return real_unlink(path, *call_args, **call_kwargs)

    monkeypatch.setattr(prune.os, "unlink", fail_after_staging)
    partial = reclaim._apply(
        argparse.Namespace(
            request=request_path,
            evidence_root=fixture["runtime_root"],
            output_root=closeout_root,
            permanent_root=permanent_root,
            max_permanent_capsule_bytes=driver.PERMANENT_CAPSULE_LIMIT_BYTES,
            max_permanent_root_bytes=driver.PERMANENT_ROOT_LIMIT_BYTES,
            apply=True,
        )
    )
    assert partial["mode"] == "partial_failure"
    assert failed is True
    assert not fixture["raw_path"].exists()
    release_journal = json.loads(Path(partial["release_journal"]).read_text())
    assert release_journal["progress"][0]["state"] == "STAGED"
    monkeypatch.setattr(prune.os, "unlink", real_unlink)

    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            str(fixture["runtime_root"]),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    close_apply_commands.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(DRIVER_PATH),
            "--matrix",
            str(matrix_path),
            "--expected-matrix-sha256",
            matrix_a_sha256,
            "--repo-root",
            str(repo_root),
            "--build-root",
            str(build_root),
            "--permanent-root",
            str(permanent_root),
            "--apply",
        ],
    )
    try:
        assert driver.main() == 0
    finally:
        child.terminate()
        child.wait(timeout=5)
    recovered = json.loads(capsys.readouterr().out)
    assert recovered["results"][0]["status"] == "APPLIED"
    assert len(close_apply_commands) == 1
    assert json.loads(intent_path.read_text())["status"] == "APPLIED"
    assert not fixture["raw_path"].exists()
