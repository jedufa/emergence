from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

from tools.scripts import eiel_bridge_records as records


ROOT = Path(__file__).resolve().parents[2]
THREAD_URL = (
    "https://chatgpt.com/c/00000000-0000-0000-0000-000000000000"
    "c/00000000-0000-4000-8000-000000000002"
)


def _run(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "bridge/templates").mkdir(parents=True)
    (project / "bridge/prompt_receipts").mkdir()
    (project / "work_packages").mkdir()
    shutil.copy(
        ROOT / "bridge/templates/latest_status_v1.schema.json",
        project / "bridge/templates/latest_status_v1.schema.json",
    )
    shutil.copy(
        ROOT / "bridge/templates/prompt_receipt_v1.schema.json",
        project / "bridge/templates/prompt_receipt_v1.schema.json",
    )
    shutil.copy(
        ROOT / "bridge/templates/latest_status_v1.json",
        project / "bridge/latest_status.json",
    )
    (project / "work_packages/WORK.md").write_text("# Work package\n", encoding="utf-8")
    _run(project, "init", "-b", "main")
    _run(project, "config", "user.name", "Test")
    _run(project, "config", "user.email", "test@example.invalid")
    _run(project, "add", ".")
    _run(project, "commit", "-m", "fixture")
    return project


def _prompt(tmp_path: Path, *, members: tuple[str, ...] = ("prompt.md",)) -> tuple[Path, Path]:
    payload = b"# Approved prompt\n"
    archive = tmp_path / "prompt.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for member in members:
            bundle.writestr(member, payload)
    extracted = tmp_path / "prompt.md"
    extracted.write_bytes(payload)
    return archive, extracted


def _activation(project: Path, archive: Path, extracted: Path, **overrides):
    kwargs = {
        "source_zip": archive,
        "extracted_prompt": extracted,
        "work_item": "EIEL-TEST",
        "work_package_path": "work_packages/WORK.md",
        "target_branch": "main",
        "target_worktree": project,
        "created_at_utc": "2026-08-01T12:00:00Z",
        "operator_instruction_text": "Execute the approved package.",
        "operator_instruction_timestamp": "2026-08-01T12:00:00Z",
        "source_thread_url": THREAD_URL,
        "active_goal_before": "ACTIVE_GOAL.md",
        "execution_policy": "manual_phase2",
        "bridge_policy": "explicit_handoff_only",
        "mail_policy": "no_email",
        "trash_after_start": False,
        "activation_result": "activated_test",
        "limitations": ["Test fixture."],
    }
    kwargs.update(overrides)
    return records.build_activation_receipt(project, **kwargs)


def test_live_audit_has_valid_mutable_projection_and_classifies_history() -> None:
    result = records.audit_records(ROOT)
    assert result["blocking_count"] == 0
    assert result["ok"] is True
    latest = next(row for row in result["records"] if row["role"] == "mutable_latest_status")
    assert latest["valid"] is True
    unsupported = [row for row in result["records"] if row["schema_status"] == "unsupported"]
    assert len(unsupported) == 3
    assert all(row["severity"] == "historical_finding" for row in unsupported)


def test_audit_does_not_rewrite_historical_receipts() -> None:
    before = {path: _sha(path) for path in (ROOT / "bridge/prompt_receipts").glob("*.json")}
    records.audit_records(ROOT)
    after = {path: _sha(path) for path in before}
    assert before == after


def test_audit_cli_is_read_only_and_exposes_no_transport_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = {path: _sha(path) for path in (ROOT / "bridge/prompt_receipts").glob("*.json")}
    assert records.main(["--project-root", str(ROOT), "audit"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["historical_finding_count"] == 10
    assert {path: _sha(path) for path in before} == before
    parser = records.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--project-root", str(ROOT), "send"])


def test_create_activation_is_hash_bound_valid_and_exclusive(tmp_path: Path) -> None:
    project = _project(tmp_path)
    archive, extracted = _prompt(tmp_path)
    value = _activation(project, archive, extracted)
    assert value["source_thread_url"] == THREAD_URL
    assert value["base_commit"] == _run(project, "rev-parse", "HEAD")
    assert value["source_sha256"] == _sha(archive)
    assert value["prompt_sha256"] == _sha(extracted)
    assert records._validation_errors(project, records.ACTIVATION_RECEIPT_VERSION, value) == []

    output = project / "bridge/prompt_receipts/20260801T120000Z_EIEL-TEST.json"
    records.write_activation_receipt(project, output, value)
    assert json.loads(output.read_text()) == value
    with pytest.raises(records.BridgeRecordError, match="overwrite"):
        records.write_activation_receipt(project, output, value)


@pytest.mark.parametrize(
    "members",
    [
        ("one.md", "two.md"),
        ("nested/prompt.md",),
        ("prompt.txt",),
    ],
)
def test_create_activation_rejects_noncanonical_zip(tmp_path: Path, members: tuple[str, ...]) -> None:
    project = _project(tmp_path)
    archive, extracted = _prompt(tmp_path, members=members)
    with pytest.raises(records.BridgeRecordError, match="activation ZIP"):
        _activation(project, archive, extracted)


def test_create_activation_rejects_prompt_drift_branch_mismatch_and_self_supersession(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    archive, extracted = _prompt(tmp_path)
    extracted.write_text("changed\n", encoding="utf-8")
    with pytest.raises(records.BridgeRecordError, match="do not match"):
        _activation(project, archive, extracted)

    archive, extracted = _prompt(tmp_path)
    with pytest.raises(records.BridgeRecordError, match="branch mismatch"):
        _activation(project, archive, extracted, target_branch="other")
    with pytest.raises(records.BridgeRecordError, match="supersede itself"):
        _activation(
            project,
            archive,
            extracted,
            supersedes_receipt="20260801T120000Z_EIEL-TEST",
        )


def test_create_activation_rejects_symlinked_work_package(tmp_path: Path) -> None:
    project = _project(tmp_path)
    archive, extracted = _prompt(tmp_path)
    (project / "work_packages/LINK.md").symlink_to("WORK.md")
    with pytest.raises(records.BridgeRecordError, match="symbolic link"):
        _activation(
            project,
            archive,
            extracted,
            work_package_path="work_packages/LINK.md",
        )


def _status_source(project: Path) -> dict:
    report = project / "report.md"
    evidence = project / "evidence.md"
    report.write_text("# Report\n", encoding="utf-8")
    evidence.write_text("# Evidence\n", encoding="utf-8")
    receipt = project / "bridge/prompt_receipts/receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    head = _run(project, "rev-parse", "HEAD")
    return {
        "schema_version": records.STATUS_SOURCE_SCHEMA_VERSION,
        "status": {
            "updated_at_utc": "2026-08-01T12:00:00Z",
            "work_item": "EIEL-TEST",
            "work_package_path": "work_packages/WORK.md",
            "work_package_sha256": _sha(project / "work_packages/WORK.md"),
            "activation_receipt_path": "bridge/prompt_receipts/receipt.json",
            "activation_receipt_sha256": _sha(receipt),
            "execution_status": "handoff_ready",
            "decision": "TEST-A",
            "branch": "main",
            "worktree": str(project),
            "base_commit": head,
            "current_commit": head,
            "files_changed": [],
            "commands_run": [],
            "tests_run": [],
            "campaigns_run": [],
            "resource_claim_ids": [],
            "report_path": "report.md",
            "evidence_index_path": "evidence.md",
            "bridge_thread_url": THREAD_URL,
            "handoff_bundle_path": None,
            "handoff_bundle_sha256": None,
            "operator_decision_required": False,
            "next_work_item": None,
            "limitations": ["Fixture only."],
            "no_claims_statement": "No claims.",
            "projection_source": "test fixture",
            "projection_validity": "hash-bound fixture",
        },
    }


def test_render_status_is_schema_and_hash_bound(tmp_path: Path) -> None:
    project = _project(tmp_path)
    source = _status_source(project)
    value = records.render_status(project, source)
    assert value["schema_version"] == records.LATEST_STATUS_VERSION
    assert value["record_role"] == "non_authoritative_projection"

    source["status"]["work_package_sha256"] = "0" * 64
    with pytest.raises(records.BridgeRecordError, match="does not match"):
        records.render_status(project, source)


def test_render_status_rejects_unknown_fields_secrets_and_missing_commits(tmp_path: Path) -> None:
    project = _project(tmp_path)
    source = _status_source(project)
    source["status"]["unexpected"] = True
    with pytest.raises(records.BridgeRecordError, match="invalid"):
        records.render_status(project, source)
    source = _status_source(project)
    source["status"]["password"] = "not-a-real-secret"
    with pytest.raises(records.BridgeRecordError, match="forbidden secret"):
        records.render_status(project, source)
    source = _status_source(project)
    source["status"]["current_commit"] = "0" * 40
    with pytest.raises(records.BridgeRecordError, match="not a commit"):
        records.render_status(project, source)


def test_write_status_is_explicit_and_atomic(tmp_path: Path) -> None:
    project = _project(tmp_path)
    value = records.render_status(project, _status_source(project))
    output = tmp_path / "status.json"
    records.write_status(output, value)
    assert json.loads(output.read_text()) == value
    assert not list(tmp_path.glob(".*.tmp"))
