from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import make_bootstrap_repo, run_bootstrap
from tests.eiel_acceptance_v118_helpers import write_review_proof_for_session
from tools.actors.common import ActorError, file_sha256, revise_record
from tools.actors.portfolio import session_store, work_store
from tools.actors.session_manager import persist_session_record_fixture
from tools.actors.work_manager import persist_work_record_fixture
from tools.actors.work_package import (
    bind_work_package,
    load_work_package,
    validate_changed_paths,
    work_package_contract_sha256,
)


def _package_value(*, work_id: str = "WORK-V18", version: int = 1) -> dict:
    return {
        "schema_version": "eiel_work_package_v1",
        "work_id": work_id,
        "package_version": version,
        "title": "V1.8 hardening fixture",
        "goal": "Implement one bounded component.",
        "scope": ["Change only the approved component."],
        "owned_files": ["tools/bounded_component.py", "tests/test_bounded_component.py"],
        "forbidden_files": ["tools/actors/mission_planner.py"],
        "dependencies": [],
        "acceptance_criteria": ["The bounded component returns the reviewed value."],
        "tests": ["pytest -q tests/test_bounded_component.py"],
        "evidence": ["Focused test output and final commit."],
        "stop_conditions": ["Promote if a shared contract must change."],
        "recommended_authority": "IMPLEMENTATION_AUTHORITY",
        "recommended_model": "GPT-5.3 Spark High",
        "uncertainty_class": "U2",
        "read_contract": {"entry_files": ["tools/bounded_component.py"], "required_documents": [], "search_roots": ["tools"], "excluded_roots": [], "known_symbols": ["VALUE"], "maximum_initial_files": 6, "broad_search_allowed": False},
        "output_policy": {"default_output_mode": "compact", "maximum_display_lines": 120, "test_output": "failures_and_summary", "git_output": "short_status_or_stat", "json_output": "selected_fields", "log_output": "tail_only", "recursive_output_requires_reason": True},
        "execution_budget": {"expected_tool_calls": 25, "handoff_recommended_after_tool_calls": 50, "maximum_compactions": 1, "expected_duration_minutes": 60, "context_health_policy": "advisory"},
    }


def _write_package(repo: Path, value: dict, name: str = "WORK-V18.json", *, commit: bool = True) -> Path:
    path = repo / "work_packages" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if commit:
        subprocess.run(["git", "add", path.relative_to(repo)], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", f"add {name}"], cwd=repo, check=True)
    return path


def _create_bound(repo: Path, state: Path, package: Path, *, generated_at: str) -> dict:
    return run_eiel(
        [
            "workManager", "create", "--work-package", str(package), "--repo-root", str(repo),
            "--status", "ACTIVE", "--next-action", "Execute the bound package.",
            "--state-root", str(state), "--generated-at", generated_at,
        ],
        state_root=state,
    )["work"]


def _start(
    repo: Path,
    state: Path,
    bootstrap_id: str,
    *,
    codex_id: str,
    role: str = "IMPLEMENTATION_AUTHORITY",
    target: str | None = None,
    generated_at: str,
) -> dict:
    args = [
        "sessionManager", "start", "WORK-V18", "--bootstrap-record", bootstrap_id,
        "--codex-id", codex_id, "--repo-root", str(repo), "--role", role,
        "--model", "GPT-5.5 High" if role == "REVIEW_AUTHORITY" else "GPT-5.3 Spark High",
        "--reasoning-level", "HIGH", "--state-root", str(state), "--generated-at", generated_at,
    ]
    if target:
        args.extend(["--review-target-session", target])
    return run_eiel(args, state_root=state)["session"]


def _commit(repo: Path, relative: str, text: str, message: str) -> str:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", relative], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _refresh_bootstrap(repo: Path, state: Path, session_id: str, *, generated_at: str) -> str:
    bootstrap_id = run_bootstrap(
        repo=repo,
        state_root=state,
        generated_at=generated_at,
    )["bootstrap"]["bootstrap_id"]
    run_eiel(
        [
            "sessionManager",
            "attach-bootstrap",
            session_id,
            "--bootstrap-record",
            bootstrap_id,
            "--repo-root",
            str(repo),
            "--state-root",
            str(state),
            "--generated-at",
            generated_at,
        ],
        state_root=state,
    )
    return bootstrap_id


def test_canonical_json_yaml_identity_and_strict_yaml(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "tests").mkdir()
    value = _package_value()
    json_path = repo / "package.json"
    json_path.write_text(json.dumps(value, indent=7) + "\n", encoding="utf-8")
    yaml_path = repo / "package.yaml"
    yaml_path.write_text(
        """schema_version: eiel_work_package_v1
work_id: WORK-V18
package_version: 1
title: V1.8 hardening fixture
goal: Implement one bounded component.
scope:
  - Change only the approved component.
owned_files:
  - tools/bounded_component.py
  - tests/test_bounded_component.py
forbidden_files:
  - tools/actors/mission_planner.py
dependencies: []
acceptance_criteria:
  - The bounded component returns the reviewed value.
tests:
  - pytest -q tests/test_bounded_component.py
evidence:
  - Focused test output and final commit.
stop_conditions:
  - Promote if a shared contract must change.
recommended_authority: IMPLEMENTATION_AUTHORITY
recommended_model: GPT-5.3 Spark High
uncertainty_class: U2
read_contract:
  entry_files:
    - tools/bounded_component.py
  required_documents: []
  search_roots:
    - tools
  excluded_roots: []
  known_symbols:
    - VALUE
  maximum_initial_files: 6
  broad_search_allowed: false
output_policy:
  default_output_mode: compact
  maximum_display_lines: 120
  test_output: failures_and_summary
  git_output: short_status_or_stat
  json_output: selected_fields
  log_output: tail_only
  recursive_output_requires_reason: true
execution_budget:
  expected_tool_calls: 25
  handoff_recommended_after_tool_calls: 50
  maximum_compactions: 1
  expected_duration_minutes: 60
  context_health_policy: advisory
""",
        encoding="utf-8",
    )
    json_binding, json_package = bind_work_package(json_path, repo_root=repo)
    yaml_binding, yaml_package = bind_work_package(yaml_path, repo_root=repo)
    assert json_binding["sha256"] == yaml_binding["sha256"]
    assert json_binding["sha256_kind"] == "canonical_work_package_contract_v2"
    assert work_package_contract_sha256(json_package) == work_package_contract_sha256(yaml_package)

    nonmaterial = dict(value)
    nonmaterial["title"] = "Spelling and display title corrected"
    nonmaterial["recommended_model"] = "A newer human-selected recommendation"
    assert work_package_contract_sha256(nonmaterial) == work_package_contract_sha256(value)
    material = dict(value)
    material["goal"] = "A materially different engineering goal."
    assert work_package_contract_sha256(material) != work_package_contract_sha256(value)

    duplicate = repo / "duplicate.yaml"
    duplicate.write_text("schema_version: eiel_work_package_v1\nschema_version: duplicate\n", encoding="utf-8")
    with pytest.raises(ActorError) as exc:
        load_work_package(duplicate)
    assert exc.value.code == "work_package_duplicate_key"

    alias = repo / "alias.yaml"
    alias.write_text("defaults: &x [a]\nscope: *x\n", encoding="utf-8")
    with pytest.raises(ActorError) as exc:
        load_work_package(alias)
    assert exc.value.code == "work_package_yaml_alias_forbidden"

    ambiguous = repo / "ambiguous.yaml"
    ambiguous.write_text(yaml_path.read_text(encoding="utf-8").replace("package_version: 1", "package_version: 01"), encoding="utf-8")
    with pytest.raises(ActorError) as exc:
        load_work_package(ambiguous)
    assert exc.value.code == "work_package_invalid"


def test_path_fence_normalization_and_security(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "tools" / "real.py").write_text("x=1\n", encoding="utf-8")
    value = _package_value()
    package = _write_package(repo, value, commit=False)
    binding, loaded = bind_work_package(package, repo_root=repo)
    assert binding["sha256"]
    assert validate_changed_paths(["tools/bounded_component.py"], loaded, repo_root=repo)["valid"] is True
    violation = validate_changed_paths(["README.md", "tools/actors/mission_planner.py"], loaded, repo_root=repo)
    assert violation["valid"] is False
    assert violation["outside_owned_files"] == ["README.md", "tools/actors/mission_planner.py"]
    assert violation["forbidden_files_changed"] == ["tools/actors/mission_planner.py"]

    for bad in ("../escape.py", "/tmp/escape.py", "C:/escape.py", "tools\\escape.py"):
        bad_value = _package_value()
        bad_value["owned_files"] = [bad]
        bad_path = _write_package(repo, bad_value, name=f"bad-{len(bad)}.json", commit=False)
        with pytest.raises(ActorError):
            bind_work_package(bad_path, repo_root=repo)

    symlink = repo / "linked"
    symlink.symlink_to(repo / "tools", target_is_directory=True)
    symlink_value = _package_value()
    symlink_value["owned_files"] = ["linked/future.py"]
    symlink_path = _write_package(repo, symlink_value, name="symlink.json", commit=False)
    with pytest.raises(ActorError) as exc:
        bind_work_package(symlink_path, repo_root=repo)
    assert exc.value.code == "work_package_path_symlink_forbidden"

    case_value = _package_value()
    case_value["owned_files"] = ["Tools/real.py"]
    case_path = _write_package(repo, case_value, name="case.json", commit=False)
    with pytest.raises(ActorError) as exc:
        bind_work_package(case_path, repo_root=repo)
    assert exc.value.code == "work_package_path_case_mismatch"


def test_end_to_end_review_repair_acceptance_with_exact_commit_binding(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = _write_package(repo, _package_value())
    bootstrap0 = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-04T01:00:00Z")["bootstrap"]["bootstrap_id"]
    _create_bound(repo, state, package, generated_at="2026-08-04T01:01:00Z")

    implementation = _start(repo, state, bootstrap0, codex_id="spark-context-1", generated_at="2026-08-04T01:02:00Z")
    commit1 = _commit(repo, "tools/bounded_component.py", "VALUE = 1\n", "implementation v1")
    _refresh_bootstrap(repo, state, implementation["session_id"], generated_at="2026-08-04T01:02:30Z")
    completed1 = run_eiel(
        ["sessionManager", "complete", implementation["session_id"], "--delivery-commit", commit1, "--state-root", str(state), "--generated-at", "2026-08-04T01:03:00Z"],
        state_root=state,
    )["session"]
    assert completed1["delivery_evidence"]["implementation_commit"] == commit1
    assert completed1["delivery_evidence"]["changed_files"] == ["tools/bounded_component.py"]

    bootstrap1 = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-04T01:04:00Z")["bootstrap"]["bootstrap_id"]
    review1 = _start(repo, state, bootstrap1, codex_id="review-context-1", role="REVIEW_AUTHORITY", target=implementation["session_id"], generated_at="2026-08-04T01:05:00Z")
    changes = run_eiel(
        [
            "sessionManager", "complete", review1["session_id"], "--review-verdict", "CHANGES_REQUIRED",
            "--review-finding", "VALUE must be 2.", "--repair-disposition", "WITHIN_PACKAGE",
            "--state-root", str(state), "--generated-at", "2026-08-04T01:06:00Z",
        ],
        state_root=state,
    )["session"]
    assert changes["review_evidence"]["implementation_session_id"] == implementation["session_id"]
    assert changes["review_evidence"]["reviewed_commit"] == commit1
    work = run_eiel(["workManager", "inspect", "WORK-V18", "--state-root", str(state)], state_root=state)["work"]
    assert work["derived_status"] == "CHANGES_REQUIRED"

    repair = _start(repo, state, bootstrap1, codex_id="spark-context-2", generated_at="2026-08-04T01:07:00Z")
    commit2 = _commit(repo, "tools/bounded_component.py", "VALUE = 2\n", "bounded repair")
    _refresh_bootstrap(repo, state, repair["session_id"], generated_at="2026-08-04T01:07:30Z")
    completed2 = run_eiel(
        ["sessionManager", "complete", repair["session_id"], "--delivery-commit", commit2, "--state-root", str(state), "--generated-at", "2026-08-04T01:08:00Z"],
        state_root=state,
    )["session"]
    assert completed2["delivery_evidence"]["implementation_commit"] == commit2

    bootstrap2 = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-04T01:09:00Z")["bootstrap"]["bootstrap_id"]
    review2 = _start(repo, state, bootstrap2, codex_id="review-context-2", role="REVIEW_AUTHORITY", target=repair["session_id"], generated_at="2026-08-04T01:10:00Z")
    review2_proof = write_review_proof_for_session(state, review2["session_id"], limitations=["bounded fixture only"])
    accepted_review = run_eiel(
        [
            "sessionManager", "complete", review2["session_id"], "--review-verdict", "ACCEPT",
            "--review-proof", str(review2_proof),
            "--acceptance-reviewed", "The bounded component returns the reviewed value.",
            "--tests-reviewed", "pytest -q tests/test_bounded_component.py",
            "--evidence-reviewed", "Focused test output and final commit.",
            "--known-limitation", "bounded fixture only",
            "--state-root", str(state), "--generated-at", "2026-08-04T01:11:00Z",
        ],
        state_root=state,
    )["session"]
    evidence = accepted_review["review_evidence"]
    assert evidence["implementation_session_id"] == repair["session_id"]
    assert evidence["review_session_id"] == review2["session_id"]
    assert evidence["reviewed_commit"] == commit2
    assert evidence["reviewed_package_sha256"] == completed2["work_package_binding"]["sha256"]
    assert evidence["implementing_codex_identities"] == ["spark-context-2"]
    assert evidence["reviewing_codex_identities"] == ["review-context-2"]

    completed_work = run_eiel(
        ["workManager", "complete", "WORK-V18", "--review-session", review2["session_id"], "--state-root", str(state), "--generated-at", "2026-08-04T01:12:00Z"],
        state_root=state,
    )["work"]
    assert completed_work["status"] == "COMPLETED"
    handoff = run_eiel(["sessionManager", "handoff", review2["session_id"], "--state-root", str(state)], state_root=state)
    assert handoff["handoff"]["review_evidence"]["reviewed_commit"] == commit2
    audit = run_eiel(["workspace", "audit", "--repo-root", str(repo), "--state-root", str(state)], state_root=state)
    assert audit["status"] in {"CONSISTENT", "CONSISTENT_WITH_WARNINGS"}


def test_promotion_requires_revised_material_package_and_preserves_history(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package1 = _write_package(repo, _package_value())
    bootstrap0 = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-04T02:00:00Z")["bootstrap"]["bootstrap_id"]
    _create_bound(repo, state, package1, generated_at="2026-08-04T02:01:00Z")
    session1 = _start(repo, state, bootstrap0, codex_id="spark-promotion", generated_at="2026-08-04T02:02:00Z")
    promoted = run_eiel(
        [
            "sessionManager", "request-promotion", session1["session_id"], "--requested-class", "U4",
            "--reason", "Architecture ownership is unresolved.", "--affected", "tools/bounded_component.py",
            "--safe-state", "No implementation commit was created.", "--state-root", str(state),
            "--generated-at", "2026-08-04T02:03:00Z",
        ],
        state_root=state,
    )["session"]
    old_digest = promoted["work_package_binding"]["sha256"]
    run_eiel(["sessionManager", "discontinue", session1["session_id"], "--state-root", str(state)], state_root=state)

    same_path_value = _package_value(version=2)
    same_path_value["goal"] = "Resolve the architecture and implement it."
    package1.write_text(json.dumps(same_path_value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed = run_eiel(
        ["workManager", "update", "WORK-V18", "--work-package", str(package1), "--repo-root", str(repo), "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert failed["reason_codes"] == ["work_package_existing_binding_invalid"]
    subprocess.run(["git", "checkout", "--", package1.relative_to(repo)], cwd=repo, check=True)

    value2 = _package_value(version=2)
    value2["goal"] = "Implement the architecture selected by Engineering Authority."
    value2["uncertainty_class"] = "U4"
    value2["recommended_authority"] = "ENGINEERING_AUTHORITY"
    value2["decision_basis"] = {
        "reason_summary": "The promotion request requires an Engineering Authority package revision.",
        "unresolved_questions": ["The shared architecture boundary remains to be implemented."],
        "rejected_alternatives": [{
            "alternative": "Continue under the obsolete implementation-only package.",
            "reason_rejected": "The promotion request requires Engineering Authority and a revised material contract.",
        }],
        "source_decision_refs": [f"session:{session1['session_id']}:promotion"],
    }
    package2 = _write_package(repo, value2, name="WORK-V18-v2.json")
    run_eiel(
        ["workManager", "update", "WORK-V18", "--work-package", str(package2), "--repo-root", str(repo), "--state-root", str(state), "--generated-at", "2026-08-04T02:04:00Z"],
        state_root=state,
    )
    current = run_eiel(["workManager", "inspect", "WORK-V18", "--state-root", str(state)], state_root=state)["work"]
    assert current["work_package_binding"]["package_version"] == 2
    assert current["work_package_binding"]["sha256"] != old_digest
    resolution = current["promotion_resolution"]
    assert resolution["source_promotion"]["session_id"] == session1["session_id"]
    assert resolution["previous_package_binding"]["sha256"] == old_digest
    assert resolution["replacement_package_binding"]["sha256"] == current["work_package_binding"]["sha256"]
    assert resolution["deciding_authority"]["role"] == "ENGINEERING_AUTHORITY"
    assert current["package_binding_history"][-1]["reason"] == value2["decision_basis"]["reason_summary"]
    historical = run_eiel(["sessionManager", "inspect", session1["session_id"], "--state-root", str(state)], state_root=state)["session"]
    assert historical["work_package_binding"]["sha256"] == old_digest
    assert historical["valid"] is True


def test_administrative_reconciliation_is_not_acceptance(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = _write_package(repo, _package_value())
    bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-04T03:00:00Z")["bootstrap"]["bootstrap_id"]
    _create_bound(repo, state, package, generated_at="2026-08-04T03:01:00Z")
    session = _start(repo, state, bootstrap, codex_id="corrupt-context", generated_at="2026-08-04T03:02:00Z")
    blocked = run_eiel(
        ["workManager", "reconcile", "WORK-V18", "--disposition", "DISCONTINUE", "--reason", "Historical state is corrupt.", "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert blocked["reason_codes"] == ["work_reconciliation_ack_required"]
    reconciled = run_eiel(
        [
            "workManager", "reconcile", "WORK-V18", "--disposition", "DISCONTINUE",
            "--reason", "Historical state is corrupt.", "--acknowledge-no-acceptance",
            "--state-root", str(state), "--generated-at", "2026-08-04T03:03:00Z",
        ],
        state_root=state,
    )
    assert reconciled["accepted"] is False
    assert reconciled["work"]["status"] == "DISCONTINUED"
    reconciliation = reconciled["work"]["administrative_reconciliation"]
    assert reconciliation["acceptance_implied"] is False
    assert reconciliation["dependencies_released"] is False
    assert session["session_id"] in reconciliation["active_session_ids_at_reconciliation"]
    history = run_eiel(["workManager", "history", "WORK-V18", "--state-root", str(state)], state_root=state)
    latest_snapshot = json.loads(Path(history["history"][-1]["path"]).read_text(encoding="utf-8"))
    assert latest_snapshot["history"][-1]["action"] == "administrative_reconcile"
    assert latest_snapshot["payload"]["administrative_reconciliation"]["acceptance_implied"] is False


def test_nonmaterial_package_edit_can_rebind_same_version_without_rewriting_history(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    value = _package_value()
    package = _write_package(repo, value)
    bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-04T04:00:00Z")["bootstrap"]["bootstrap_id"]
    _create_bound(repo, state, package, generated_at="2026-08-04T04:01:00Z")
    session = _start(repo, state, bootstrap, codex_id="nonmaterial-context", generated_at="2026-08-04T04:02:00Z")
    old_digest = session["work_package_binding"]["sha256"]
    run_eiel(["sessionManager", "discontinue", session["session_id"], "--state-root", str(state)], state_root=state)

    value["title"] = "Corrected display title"
    value["recommended_model"] = "Current advisory model changed"
    package.write_text(json.dumps(value, indent=4, sort_keys=False) + "\n", encoding="utf-8")
    subprocess.run(["git", "add", package.relative_to(repo)], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "non-material package wording"], cwd=repo, check=True)
    updated = run_eiel(
        ["workManager", "update", "WORK-V18", "--work-package", str(package), "--repo-root", str(repo), "--state-root", str(state)],
        state_root=state,
    )["work"]
    assert updated["work_package_binding"]["package_version"] == 1
    assert updated["work_package_binding"]["sha256"] == old_digest
    assert updated["work_package_binding"]["recommended_model"] == "Current advisory model changed"
    historical = run_eiel(["sessionManager", "inspect", session["session_id"], "--state-root", str(state)], state_root=state)["session"]
    assert historical["valid"] is True
    assert historical["work_package_binding"]["recommended_model"] == "GPT-5.3 Spark High"


def test_delivery_completion_fails_closed_on_dirty_or_out_of_scope_changes(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = _write_package(repo, _package_value())
    bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-04T05:00:00Z")["bootstrap"]["bootstrap_id"]
    _create_bound(repo, state, package, generated_at="2026-08-04T05:01:00Z")
    session = _start(repo, state, bootstrap, codex_id="fence-context", generated_at="2026-08-04T05:02:00Z")
    (repo / "tools/bounded_component.py").write_text("VALUE = 1\n", encoding="utf-8")
    dirty = run_eiel(
        ["sessionManager", "complete", session["session_id"], "--delivery-commit", subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stdout=subprocess.PIPE, check=True).stdout.strip(), "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert dirty["reason_codes"] == ["session_delivery_worktree_dirty"]
    subprocess.run(["git", "add", "tools/bounded_component.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "owned change"], cwd=repo, check=True)
    (repo / "README.md").write_text("out of scope\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "out of scope change"], cwd=repo, check=True)
    _refresh_bootstrap(repo, state, session["session_id"], generated_at="2026-08-04T05:03:00Z")
    outside = run_eiel(
        ["sessionManager", "complete", session["session_id"], "--delivery-commit", subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stdout=subprocess.PIPE, check=True).stdout.strip(), "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert outside["reason_codes"] == ["work_package_file_fence_violation"]
    assert any("README.md" in problem for problem in outside["problems"])


def test_new_delivery_commit_invalidates_previous_acceptance(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = _write_package(repo, _package_value())
    bootstrap0 = run_bootstrap(
        repo=repo,
        state_root=state,
        generated_at="2026-08-04T06:00:00Z",
    )["bootstrap"]["bootstrap_id"]
    _create_bound(repo, state, package, generated_at="2026-08-04T06:01:00Z")

    implementation1 = _start(
        repo,
        state,
        bootstrap0,
        codex_id="spark-stale-1",
        generated_at="2026-08-04T06:02:00Z",
    )
    commit1 = _commit(repo, "tools/bounded_component.py", "VALUE = 1\n", "delivery one")
    _refresh_bootstrap(repo, state, implementation1["session_id"], generated_at="2026-08-04T06:02:30Z")
    run_eiel(
        [
            "sessionManager",
            "complete",
            implementation1["session_id"],
            "--delivery-commit",
            commit1,
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T06:03:00Z",
        ],
        state_root=state,
    )

    bootstrap1 = run_bootstrap(
        repo=repo,
        state_root=state,
        generated_at="2026-08-04T06:04:00Z",
    )["bootstrap"]["bootstrap_id"]
    review1 = _start(
        repo,
        state,
        bootstrap1,
        codex_id="review-stale-1",
        role="REVIEW_AUTHORITY",
        target=implementation1["session_id"],
        generated_at="2026-08-04T06:05:00Z",
    )
    review1_proof = write_review_proof_for_session(state, review1["session_id"])
    run_eiel(
        [
            "sessionManager",
            "complete",
            review1["session_id"],
            "--review-verdict",
            "ACCEPT",
            "--review-proof",
            str(review1_proof),
            "--acceptance-reviewed",
            "The bounded component returns the reviewed value.",
            "--tests-reviewed",
            "pytest -q tests/test_bounded_component.py",
            "--evidence-reviewed",
            "Focused test output and final commit.",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T06:06:00Z",
        ],
        state_root=state,
    )

    bootstrap2 = run_bootstrap(
        repo=repo,
        state_root=state,
        generated_at="2026-08-04T06:07:00Z",
    )["bootstrap"]["bootstrap_id"]
    implementation2 = _start(
        repo,
        state,
        bootstrap2,
        codex_id="spark-stale-2",
        generated_at="2026-08-04T06:08:00Z",
    )
    commit2 = _commit(repo, "tools/bounded_component.py", "VALUE = 2\n", "delivery two")
    _refresh_bootstrap(repo, state, implementation2["session_id"], generated_at="2026-08-04T06:08:30Z")
    delivery2 = run_eiel(
        [
            "sessionManager",
            "complete",
            implementation2["session_id"],
            "--delivery-commit",
            commit2,
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T06:09:00Z",
        ],
        state_root=state,
    )["session"]
    assert delivery2["delivery_evidence"]["implementation_commit"] == commit2

    failed = run_eiel(
        [
            "workManager",
            "complete",
            "WORK-V18",
            "--review-session",
            review1["session_id"],
            "--state-root",
            str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert failed["reason_codes"] == ["work_review_not_accepted"]
    assert any("delivery" in problem for problem in failed["problems"])

    inspected = run_eiel(
        ["workManager", "inspect", "WORK-V18", "--state-root", str(state)],
        state_root=state,
    )["work"]
    assert inspected["derived_status"] == "REVIEW_REQUIRED"
    assert inspected["review"]["latest_delivery_session_id"] == implementation2["session_id"]
    assert inspected["review"]["accepted_review_sessions"] == []


def test_package_dependencies_match_work_graph_and_block_implementation(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    dependency = run_eiel(
        [
            "workManager",
            "create",
            "--work-id",
            "WORK-DEP",
            "--title",
            "Required dependency",
            "--summary",
            "A simple successful prerequisite.",
            "--status",
            "ACTIVE",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T07:00:00Z",
        ],
        state_root=state,
    )["work"]
    assert dependency["work_id"] == "WORK-DEP"

    value = _package_value()
    value["dependencies"] = ["WORK-DEP"]
    package = _write_package(repo, value)
    active = run_eiel(
        [
            "workManager",
            "create",
            "--work-package",
            str(package),
            "--repo-root",
            str(repo),
            "--status",
            "ACTIVE",
            "--state-root",
            str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert active["reason_codes"] == ["work_dependencies_unresolved"]

    created = run_eiel(
        [
            "workManager",
            "create",
            "--work-package",
            str(package),
            "--repo-root",
            str(repo),
            "--status",
            "BACKLOG",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T07:01:00Z",
        ],
        state_root=state,
    )["work"]
    assert created["status"] == "BLOCKED"
    assert created["dependency_work_ids"] == ["WORK-DEP"]

    bootstrap = run_bootstrap(
        repo=repo,
        state_root=state,
        generated_at="2026-08-04T07:02:00Z",
    )["bootstrap"]["bootstrap_id"]
    blocked_start = run_eiel(
        [
            "sessionManager",
            "start",
            "WORK-V18",
            "--bootstrap-record",
            bootstrap,
            "--codex-id",
            "blocked-spark",
            "--repo-root",
            str(repo),
            "--role",
            "IMPLEMENTATION_AUTHORITY",
            "--state-root",
            str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert blocked_start["reason_codes"] == ["session_work_dependencies_unresolved"]

    mismatch = run_eiel(
        [
            "workManager",
            "dependency",
            "remove",
            "WORK-V18",
            "WORK-DEP",
            "--state-root",
            str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert mismatch["reason_codes"] == ["work_validation_failed"]
    assert any("do not match" in problem for problem in mismatch["problems"])

    run_eiel(
        [
            "workManager",
            "complete",
            "WORK-DEP",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T07:03:00Z",
        ],
        state_root=state,
    )
    run_eiel(
        [
            "workManager",
            "refresh",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T07:04:00Z",
        ],
        state_root=state,
    )
    ready = run_eiel(
        ["workManager", "inspect", "WORK-V18", "--state-root", str(state)],
        state_root=state,
    )["work"]
    assert ready["status"] == "BLOCKED"
    assert ready["derived_status"] == "READY_TO_RESUME"
    assert ready["dependencies"]["all_satisfied"] is True


def test_legacy_v17_completed_review_remains_auditable_with_warning(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = _write_package(repo, _package_value())
    bootstrap0 = run_bootstrap(
        repo=repo,
        state_root=state,
        generated_at="2026-08-04T08:00:00Z",
    )["bootstrap"]["bootstrap_id"]
    _create_bound(repo, state, package, generated_at="2026-08-04T08:01:00Z")
    implementation = _start(
        repo,
        state,
        bootstrap0,
        codex_id="legacy-implementer",
        generated_at="2026-08-04T08:02:00Z",
    )
    legacy_commit = _commit(repo, "tools/bounded_component.py", "VALUE = 1\n", "legacy delivery")
    _refresh_bootstrap(repo, state, implementation["session_id"], generated_at="2026-08-04T08:02:30Z")
    run_eiel(
        [
            "sessionManager",
            "complete",
            implementation["session_id"],
            "--delivery-commit",
            legacy_commit,
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T08:03:00Z",
        ],
        state_root=state,
    )
    bootstrap1 = run_bootstrap(
        repo=repo,
        state_root=state,
        generated_at="2026-08-04T08:04:00Z",
    )["bootstrap"]["bootstrap_id"]
    review = _start(
        repo,
        state,
        bootstrap1,
        codex_id="legacy-reviewer",
        role="REVIEW_AUTHORITY",
        target=implementation["session_id"],
        generated_at="2026-08-04T08:05:00Z",
    )
    review_proof = write_review_proof_for_session(state, review["session_id"])
    run_eiel(
        [
            "sessionManager",
            "complete",
            review["session_id"],
            "--review-verdict",
            "ACCEPT",
            "--review-proof",
            str(review_proof),
            "--acceptance-reviewed",
            "The bounded component returns the reviewed value.",
            "--tests-reviewed",
            "pytest -q tests/test_bounded_component.py",
            "--evidence-reviewed",
            "Focused test output and final commit.",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T08:06:00Z",
        ],
        state_root=state,
    )
    run_eiel(
        [
            "workManager",
            "complete",
            "WORK-V18",
            "--review-session",
            review["session_id"],
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T08:07:00Z",
        ],
        state_root=state,
    )

    raw_binding = {
        "schema_version": "eiel_work_package_binding_v1",
        "path": package.relative_to(repo).as_posix(),
        "path_kind": "repository_relative",
        "bound_repo_root": str(repo.resolve()),
        "sha256": file_sha256(package),
        "package_schema_version": "eiel_work_package_v1",
        "package_version": 1,
        "work_id": "WORK-V18",
        "recommended_authority": "IMPLEMENTATION_AUTHORITY",
        "recommended_model": "GPT-5.3 Spark High",
        "uncertainty_class": "U2",
    }

    session_records = []
    for session_id in (implementation["session_id"], review["session_id"]):
        record = session_store(state).get(session_id)
        payload = dict(record["payload"])
        payload["work_package_binding"] = dict(raw_binding)
        payload.pop("delivery_evidence", None)
        payload.pop("review_target", None)
        payload.pop("review_evidence", None)
        revised = revise_record(
            record,
            payload=payload,
            status=str(record["status"]),
            action="simulate_v1_7_record",
            generated_at="2026-08-04T08:08:00Z",
            note="test fixture only",
        )
        persist_session_record_fixture(revised, state_root=state, allow_update=True)
        session_records.append(revised)

    work_record = work_store(state).get("WORK-V18")
    work_payload = dict(work_record["payload"])
    work_payload["work_package_binding"] = dict(raw_binding)
    revised_work = revise_record(
        work_record,
        payload=work_payload,
        status="COMPLETED",
        action="simulate_v1_7_record",
        generated_at="2026-08-04T08:09:00Z",
        note="test fixture only",
    )
    persist_work_record_fixture(revised_work, state_root=state, allow_update=True)

    inspected = run_eiel(
        ["workManager", "inspect", "WORK-V18", "--state-root", str(state)],
        state_root=state,
    )["work"]
    accepted = inspected["review"]["accepted_review_sessions"]
    assert accepted == []
    legacy_review = next(
        item for item in inspected["review"]["review_sessions"]
        if item["session_id"] == review["session_id"]
    )
    assert legacy_review["legacy_acceptance"] is True
    assert legacy_review["authoritative"] is False
    assert any("historical compatibility evidence only" in warning for warning in legacy_review["warnings"])

    audit = run_eiel(
        ["workspace", "audit", "--repo-root", str(repo), "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert audit["status"] == "INCONSISTENT"
    assert any("acceptance" in problem.lower() or "legacy" in problem.lower() for problem in audit["audit"]["problems"])
    assert any("legacy V1.7" in warning for warning in audit["audit"]["warnings"])


def test_delivery_rename_cannot_hide_out_of_scope_source(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    (repo / "outside.txt").write_text("outside\n", encoding="utf-8")
    subprocess.run(["git", "add", "outside.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add outside source"], cwd=repo, check=True)
    package = _write_package(repo, _package_value())
    bootstrap = run_bootstrap(
        repo=repo,
        state_root=state,
        generated_at="2026-08-04T09:00:00Z",
    )["bootstrap"]["bootstrap_id"]
    _create_bound(repo, state, package, generated_at="2026-08-04T09:01:00Z")
    session = _start(
        repo,
        state,
        bootstrap,
        codex_id="rename-fence-context",
        generated_at="2026-08-04T09:02:00Z",
    )
    subprocess.run(
        ["git", "mv", "outside.txt", "tools/bounded_component.py"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "commit", "-q", "-m", "rename outside file into owned path"], cwd=repo, check=True)
    _refresh_bootstrap(repo, state, session["session_id"], generated_at="2026-08-04T09:03:00Z")
    result = run_eiel(
        ["sessionManager", "complete", session["session_id"], "--delivery-commit", subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stdout=subprocess.PIPE, check=True).stdout.strip(), "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert result["reason_codes"] == ["work_package_file_fence_violation"]
    assert any("outside.txt" in problem for problem in result["problems"])
