"""End-to-end forensic checks for the operational-continuity doctrine.

Each case uses the public EIEL managers and asserts the actionable result,
rather than treating a lifecycle enum as an execution decision.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import make_bootstrap_repo, run_bootstrap
from tests.eiel_acceptance_v118_helpers import write_review_proof_for_session
from tests.test_eiel_work_package_v1_14 import (
    commit_file,
    create_bound,
    package_value,
    refresh,
    start_session,
    write_package,
)
from tools.actors.bootstrap import _probe_python
from tools.actors.work_package import verify_work_package_binding


def _create(state: Path, work_id: str, *, status: str = "ACTIVE", dependency: str | None = None) -> None:
    args = [
        "workManager", "create", "--work-id", work_id, "--title", work_id,
        "--summary", "forensic continuity fixture", "--status", status,
        "--state-root", str(state),
    ]
    if dependency:
        args.extend(["--dependency", dependency])
    run_eiel(args, state_root=state)


def test_a_terminal_predecessor_creates_idempotent_successor() -> None:
    import tempfile

    state = Path(tempfile.mkdtemp()) / "state"
    _create(state, "WORK-A")
    run_eiel(["workManager", "discontinue", "WORK-A", "--state-root", str(state)], state_root=state)
    first = run_eiel(["workManager", "continue", "WORK-A", "--state-root", str(state)], state_root=state)
    second = run_eiel(["workManager", "continue", "WORK-A", "--state-root", str(state)], state_root=state)
    assert first["blocker"]["operator_intervention_required"] is False
    assert second["created"] is False
    assert first["continuation_work"]["work_id"] == second["continuation_work"]["work_id"]


def test_b_missing_bootstrap_is_advisory_for_known_repository(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    result = run_eiel(["workspace", "open", "--repo-root", str(repo), "--state-root", str(tmp_path / "state")], state_root=tmp_path / "state")
    assert result["execution_blocked"] is False
    assert result["reason_codes"] == ["workspace_bootstrap_missing"]


def test_c_delivery_newer_than_session_bootstrap_closes_without_manual_attach(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    create_bound(repo, state, package)
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    session = start_session(repo, state, bootstrap_id, codex_id="same-codex", at="2026-08-06T20:01:00Z")
    delivery = commit_file(repo, "tools/bounded_component.py", "VALUE = 16\n", "delivery")
    refresh(repo, state, session["session_id"], at="2026-08-06T20:02:00Z")
    completed = run_eiel(["sessionManager", "complete", session["session_id"], "--delivery-commit", delivery, "--state-root", str(state)], state_root=state)
    assert completed["session"]["status"] == "COMPLETED"
    assert completed["session"]["delivery_evidence"]["delivery_commit"] == delivery


def test_d_same_identity_reduced_assurance_review_accepts_and_releases_work(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    create_bound(repo, state, package)
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    implementation = start_session(repo, state, bootstrap_id, codex_id="only-codex", at="2026-08-06T21:01:00Z")
    delivery = commit_file(repo, "tools/bounded_component.py", "VALUE = 16\n", "delivery")
    refresh(repo, state, implementation["session_id"], at="2026-08-06T21:02:00Z")
    run_eiel(["sessionManager", "complete", implementation["session_id"], "--delivery-commit", delivery, "--state-root", str(state)], state_root=state)
    review_bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-06T21:03:00Z")["bootstrap"]["bootstrap_id"]
    review = start_session(repo, state, review_bootstrap, codex_id="only-codex", at="2026-08-06T21:04:00Z", role="REVIEW_AUTHORITY", target=implementation["session_id"])
    proof = write_review_proof_for_session(state, review["session_id"])
    reviewed = run_eiel([
        "sessionManager", "complete", review["session_id"], "--review-verdict", "ACCEPT", "--review-proof", str(proof),
        "--acceptance-reviewed", "bounded acceptance", "--tests-reviewed", "pytest -q bounded", "--evidence-reviewed", "focused evidence",
        "--state-root", str(state),
    ], state_root=state)["session"]
    assert reviewed["review_evidence"]["identity_separation_assurance"] == "REVIEWER_INDEPENDENCE_UNAVAILABLE"
    accepted = run_eiel([
        "workManager", "complete", "WORK-V114", "--review-session", review["session_id"],
        "--state-root", str(state),
    ], state_root=state)["work"]
    assert accepted["status"] == "COMPLETED"
    assert accepted["accepted_review_session_id"] == review["session_id"]


def test_d_engineering_authority_can_adopt_preexisting_delivery(tmp_path: Path) -> None:
    """Adoption is provenance-based, never blocked by the authority role."""
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package_data = package_value()
    package_data["recommended_authority"] = "ENGINEERING_AUTHORITY"
    package = write_package(repo, package_data)
    create_bound(repo, state, package)
    delivery = commit_file(repo, "tools/bounded_component.py", "VALUE = 17\n", "preexisting delivery")
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    engineering = start_session(
        repo,
        state,
        bootstrap_id,
        codex_id="engineering-codex",
        at="2026-08-06T21:10:00Z",
        role="ENGINEERING_AUTHORITY",
    )
    completed = run_eiel([
        "sessionManager", "complete", engineering["session_id"],
        "--delivery-commit", delivery,
        "--adopt-existing-commit",
        "--adoption-reason", "Record the already committed bounded repair.",
        "--external-producer", "prior engineering delivery",
        "--state-root", str(state),
    ], state_root=state)["session"]
    assert completed["status"] == "COMPLETED"
    assert completed["session_outcome"] == "EXTERNAL_DELIVERY_ADOPTED"
    assert completed["delivery_evidence"]["delivery_kind"] == "EXTERNAL_DELIVERY_ADOPTED"


def test_e_deleted_worktree_does_not_prevent_historical_successor(tmp_path: Path) -> None:
    state = tmp_path / "state"
    former_worktree = tmp_path / "deleted-worktree"
    former_worktree.mkdir()
    _create(state, "WORK-E")
    run_eiel(["workManager", "discontinue", "WORK-E", "--state-root", str(state)], state_root=state)
    former_worktree.rmdir()
    result = run_eiel(["workManager", "continue", "WORK-E", "--state-root", str(state)], state_root=state)
    assert result["continuation_work"]["status"] == "READY"


def test_e_terminal_discontinuation_does_not_require_a_fresh_bootstrap(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    create_bound(repo, state, package)
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    session = start_session(repo, state, bootstrap_id, codex_id="same-codex", at="2026-08-06T21:20:00Z")
    commit_file(repo, "tools/bounded_component.py", "VALUE = 18\n", "unadopted delivery")

    discontinued = run_eiel(
        ["sessionManager", "discontinue", session["session_id"], "--state-root", str(state)],
        state_root=state,
    )["session"]
    assert discontinued["status"] == "DISCONTINUED"


def test_f_mechanical_bootstrap_continues_without_authority_taxonomy(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    result = run_eiel(
        [
            "bootstrap", "inspect", "--repo-root", str(repo), "--task-mode", "mechanical",
            "--topic", "bounded mechanical repair", "--operator-instruction-ref", "fixture instruction",
            "--edit-authorized", "--write-scope", "tools", "--expected-validation", "pytest -q",
            "--preflight-exemption-reason", "bounded mechanical repair has no authority claim",
            "--state-root", str(tmp_path / "state"),
        ],
        state_root=tmp_path / "state",
    )
    assert result["status"] in {"READY", "READY_WITH_WARNINGS"}


def test_g_unrelated_dirty_path_is_retained_as_delivery_context(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    create_bound(repo, state, package)
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    session = start_session(repo, state, bootstrap_id, codex_id="impl", at="2026-08-06T22:01:00Z")
    delivery = commit_file(repo, "tools/bounded_component.py", "VALUE = 16\n", "delivery")
    refresh(repo, state, session["session_id"], at="2026-08-06T22:02:00Z")
    (repo / "operator-note.txt").write_text("preserve\n", encoding="utf-8")
    completed = run_eiel(["sessionManager", "complete", session["session_id"], "--delivery-commit", delivery, "--state-root", str(state)], state_root=state)
    assert completed["session"]["delivery_evidence"]["unrelated_dirty_paths"] == ["operator-note.txt"]


def test_h_missing_qualification_routes_a_prerequisite(tmp_path: Path) -> None:
    state = tmp_path / "state"
    run_eiel(["workManager", "create", "--work-id", "WORK-H", "--title", "H", "--work-type", "QUALIFICATION", "--completion-policy", "QUALIFICATION_DECISION", "--status", "ACTIVE", "--state-root", str(state)], state_root=state)
    routed = run_eiel(["workManager", "continue", "WORK-H", "--state-root", str(state)], state_root=state)
    assert routed["blocker"]["reconciliation_action"] == "qualification_prerequisite_routed"
    assert routed["prerequisite_work"]["status"] == "READY"


def test_i_cpu_capable_task_does_not_hard_fail_for_absent_cuda(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    checks, _ = _probe_python(repo, Path(sys.executable), gpu_required=False)
    assert not [row for row in checks if row["name"] == "cuda" and row["status"] == "BLOCKED"]


def test_j_historical_package_contract_survives_stale_source_path(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    work = create_bound(repo, state, package)
    package.unlink()
    verified = verify_work_package_binding(work["work_package_binding"], repo_root=repo, expected_work_id="WORK-V114")
    assert verified["valid"] is True
    assert "immutable contract receipt" in " ".join(verified["warnings"])


def test_j_administrative_reconciliation_retains_historical_package_contract(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    create_bound(repo, state, package)
    package.unlink()

    reconciled = run_eiel([
        "workManager", "reconcile", "WORK-V114", "--disposition", "OBSOLETE",
        "--reason", "Historical source checkout was removed after a successor replaced this Work.",
        "--acknowledge-no-acceptance", "--state-root", str(state),
    ], state_root=state)

    assert reconciled["accepted"] is False
    assert reconciled["work"]["status"] == "OBSOLETE"
    assert reconciled["work"]["valid"] is True
