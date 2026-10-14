from __future__ import annotations

import json
import os
import threading
import time
import zipfile
from pathlib import Path

from tools.evidence.librarian.core import (
    AuditConfig,
    build_catalog,
    check_no_regression,
    inspect_zip_safety,
    run_audit,
)


def _init_repo(root: Path) -> None:
    os.system(f"git -C {root} init -q")
    os.system(f"git -C {root} config user.email test@example.invalid")
    os.system(f"git -C {root} config user.name Test")


def _commit_all(root: Path) -> None:
    os.system(f"git -C {root} add .")
    os.system(f"git -C {root} commit -q -m baseline")


def test_zip_safety_rejects_traversal_duplicate_case_and_unicode(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "x")
        zf.writestr("A.txt", "a")
        zf.writestr("a.txt", "b")
        zf.writestr("cafe\u0301.txt", "one")
        zf.writestr("caf\u00e9.txt", "two")

    report = inspect_zip_safety(archive)

    categories = {issue["category"] for issue in report["issues"]}
    assert report["safety_state"] == "unsafe"
    assert "zip_path_traversal_or_non_normal" in categories
    assert "zip_case_collision" in categories
    assert "zip_unicode_nfc_collision" in categories


def test_audit_detects_moving_file_and_does_not_write_inside_evidences(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    evidence = repo / "evidences"
    output = tmp_path / "audit"
    evidence.mkdir(parents=True)
    (repo / "docs").mkdir()
    moving = evidence / "moving.md"
    moving.write_text("first\n", encoding="utf-8")
    _init_repo(repo)
    _commit_all(repo)
    before = sorted(path.relative_to(evidence).as_posix() for path in evidence.rglob("*"))

    def mutate() -> None:
        time.sleep(0.02)
        moving.write_text("second\n", encoding="utf-8")

    thread = threading.Thread(target=mutate)
    thread.start()
    result = run_audit(
        AuditConfig(
            repo_root=repo,
            evidence_root=evidence,
            output_root=output,
            stability_delay_seconds=0.05,
        )
    )
    thread.join()
    after = sorted(path.relative_to(evidence).as_posix() for path in evidence.rglob("*"))

    issues = json.loads((output / "issue_ledger.json").read_text(encoding="utf-8"))
    assert before == after
    assert result["file_count"] == 1
    assert any(issue["category"] == "in_progress_or_unstable_file" for issue in issues)


def test_catalog_generation_is_deterministic_and_does_not_infer_authority(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    evidence = repo / "evidences"
    output = tmp_path / "audit"
    evidence.mkdir(parents=True)
    (repo / "docs").mkdir()
    (evidence / "record.md").write_text("# Report\n", encoding="utf-8")
    _init_repo(repo)
    _commit_all(repo)
    run_audit(AuditConfig(repo_root=repo, evidence_root=evidence, output_root=output))

    first = build_catalog(repo_root=repo, audit_output_root=output, output_root=tmp_path / "cat-a")
    second = build_catalog(repo_root=repo, audit_output_root=output, output_root=tmp_path / "cat-b")
    cat = json.loads(Path(first["catalog_path"]).read_text(encoding="utf-8"))

    assert Path(first["catalog_path"]).read_bytes() == Path(second["catalog_path"]).read_bytes()
    assert Path(first["html_path"]).read_bytes() == Path(second["html_path"]).read_bytes()
    assert cat["records"] == []
    assert cat["claims"] == []
    assert cat["authorities"] == []
    assert cat["authority_policy"]["inference_allowed"] is False


def test_no_regression_check_fails_only_for_new_issue_growth(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    issue = {
        "schema_version": "eiel_issue_v1",
        "issue_id": "eiel-issue-aaaaaaaaaaaaaaaa",
        "severity": "medium",
        "category": "missing_raw_locator",
        "paths": ["a.md"],
        "evidence": {},
        "detection_rule": "test",
        "first_observed": "test",
        "last_observed": "test",
        "owner": "unassigned",
        "disposition": "legacy_observed_unresolved",
        "exception_expiry": None,
        "remediation_state": "not_started"
    }
    baseline.write_text(json.dumps([issue]), encoding="utf-8")
    current.write_text(json.dumps([issue, {**issue, "issue_id": "eiel-issue-bbbbbbbbbbbbbbbb"}]), encoding="utf-8")

    result = check_no_regression(baseline, current, strict=True)

    assert result["status"] == "failed"
    assert result["regressions"] == [
        {"category": "missing_raw_locator", "baseline_count": 1, "current_count": 2}
    ]


def test_schema_files_are_distinct_and_require_no_inferred_authority() -> None:
    schema_root = Path("tools/evidence/schemas/eiel_librarian_v1")
    schemas = sorted(schema_root.glob("*.schema.json"))
    names = [path.name for path in schemas]

    assert "eiel_evidence_record_v1.schema.json" in names
    assert "eiel_authority_assertion_v1.schema.json" in names
    assert len(names) == 7
    for path in schemas:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["type"] == "object"
        assert payload["additionalProperties"] is False
    authority = json.loads((schema_root / "eiel_authority_assertion_v1.schema.json").read_text(encoding="utf-8"))
    assert authority["properties"]["confidence"]["enum"] == ["explicit", "unknown"]
