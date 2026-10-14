from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

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
from tools.actors.common import ActorError, new_record
from tools.actors.store import ActorRecordStore
from tools.actors.work_package import verify_work_package_binding


def head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def test_explicit_delivery_commit_and_unrelated_dirt_are_recorded(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    create_bound(repo, state, package)
    bootstrap_id = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-06T20:00:00Z")["bootstrap"]["bootstrap_id"]
    session = start_session(repo, state, bootstrap_id, codex_id="impl", at="2026-08-06T20:01:00Z")
    delivery_commit = commit_file(repo, "tools/bounded_component.py", "VALUE = 16\n", "v1.16 delivery")
    refresh(repo, state, session["session_id"], at="2026-08-06T20:02:00Z")

    missing = run_eiel(
        ["sessionManager", "complete", session["session_id"], "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert missing["reason_codes"] == ["session_delivery_commit_required"]

    (repo / "operator-note.txt").write_text("unrelated evidence\n", encoding="utf-8")
    completed = run_eiel(
        [
            "sessionManager", "complete", session["session_id"],
            "--delivery-commit", delivery_commit,
            "--state-root", str(state),
            "--generated-at", "2026-08-06T20:03:00Z",
        ],
        state_root=state,
    )["session"]
    evidence = completed["delivery_evidence"]
    # V2 delivery evidence uses one canonical spelling while retaining a
    # compatibility reader for historical implementation_commit receipts.
    assert evidence["delivery_commit"] == delivery_commit
    assert evidence["unrelated_dirty_paths"] == ["operator-note.txt"]
    assert evidence["decision_snapshot"]["delivery_commit"] == delivery_commit



def test_parallel_live_writer_conflict_is_a_narrow_hard_blocker(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    create_bound(repo, state, package)
    for directory in ("tests", "docs", "evidences"):
        (repo / directory / ".keep").write_text("tracked marker\n", encoding="utf-8")
    subprocess.run(["git", "add", "tests/.keep", "docs/.keep", "evidences/.keep"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "track worktree markers"], cwd=repo, check=True)
    repo2 = tmp_path / "delivery-b"
    subprocess.run(["git", "worktree", "add", "-q", "-b", "delivery-b", str(repo2), "HEAD"], cwd=repo, check=True)

    bootstrap_a = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-06T20:10:00Z")["bootstrap"]["bootstrap_id"]
    first = start_session(repo, state, bootstrap_a, codex_id="impl-a", at="2026-08-06T20:11:00Z")
    bootstrap_b = run_bootstrap(repo=repo2, state_root=state, generated_at="2026-08-06T20:12:00Z")["bootstrap"]["bootstrap_id"]
    second = start_session(repo2, state, bootstrap_b, codex_id="impl-b", at="2026-08-06T20:13:00Z", allow_parallel=True)

    delivery_commit = commit_file(repo, "tools/bounded_component.py", "VALUE = 16\n", "shared delivery")
    subprocess.run(["git", "reset", "--hard", "-q", delivery_commit], cwd=repo2, check=True)
    refresh(repo, state, first["session_id"], at="2026-08-06T20:14:00Z")
    refresh(repo2, state, second["session_id"], at="2026-08-06T20:15:00Z")
    conflict = run_eiel(
        ["sessionManager", "complete", first["session_id"], "--delivery-commit", delivery_commit, "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert conflict["reason_codes"] == ["session_delivery_authority_conflict"]
    assert second["session_id"] in " ".join(conflict["problems"])

def test_nonsemantic_terminal_note_does_not_stale_a_bound_review(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    create_bound(repo, state, package)
    bootstrap_id = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-06T21:00:00Z")["bootstrap"]["bootstrap_id"]
    implementation = start_session(repo, state, bootstrap_id, codex_id="impl", at="2026-08-06T21:01:00Z")
    commit = commit_file(repo, "tools/bounded_component.py", "VALUE = 16\n", "implementation")
    refresh(repo, state, implementation["session_id"], at="2026-08-06T21:02:00Z")
    run_eiel(
        ["sessionManager", "complete", implementation["session_id"], "--delivery-commit", commit, "--state-root", str(state)],
        state_root=state,
    )
    review_bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-06T21:03:00Z")["bootstrap"]["bootstrap_id"]
    review = start_session(
        repo, state, review_bootstrap, codex_id="reviewer", at="2026-08-06T21:04:00Z",
        role="REVIEW_AUTHORITY", target=implementation["session_id"],
    )
    review_proof = write_review_proof_for_session(state, review["session_id"])
    # A terminal note changes the exact implementation revision after review selection.
    run_eiel(
        ["sessionManager", "note", implementation["session_id"], "late provenance note", "--state-root", str(state)],
        state_root=state,
    )
    reviewed = run_eiel(
        [
            "sessionManager", "complete", review["session_id"], "--review-verdict", "ACCEPT",
            "--review-proof", str(review_proof),
            "--acceptance-reviewed", "The bounded component returns the reviewed value.",
            "--tests-reviewed", "pytest -q tests/test_bounded_component.py",
            "--evidence-reviewed", "Focused test output and final commit.",
            "--state-root", str(state),
        ],
        state_root=state,
    )["session"]
    assert reviewed["review_evidence"]["reviewed_commit"] == commit


def test_historical_contract_receipt_survives_live_path_removal(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    created = create_bound(repo, state, package)
    binding = created["work_package_binding"]
    assert Path(binding["contract_receipt_path"]).is_file()
    package.unlink()
    historical = verify_work_package_binding(binding, repo_root=repo, expected_work_id="WORK-V114")
    assert historical["valid"] is True
    assert historical["package"]["work_id"] == "WORK-V114"
    assert any("immutable contract receipt" in warning for warning in historical["warnings"])


def test_dependency_refresh_auto_releases_safe_administrative_block(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    run_eiel(
        ["workManager", "create", "--work-id", "WORK-DEP", "--title", "Dependency", "--status", "ACTIVE", "--next-action", "Finish", "--state-root", str(state)],
        state_root=state,
    )
    run_eiel(
        ["workManager", "create", "--work-id", "WORK-CHILD", "--title", "Child", "--dependency", "WORK-DEP", "--status", "BLOCKED", "--next-action", "Wait", "--state-root", str(state)],
        state_root=state,
    )
    run_eiel(["workManager", "complete", "WORK-DEP", "--state-root", str(state)], state_root=state)
    refreshed = run_eiel(["workManager", "refresh", "--state-root", str(state)], state_root=state)
    assert refreshed["mutated"] is True
    assert refreshed["ready_to_resume_work_ids"] == ["WORK-CHILD"]
    child = run_eiel(["workManager", "inspect", "WORK-CHILD", "--state-root", str(state)], state_root=state)["work"]
    assert child["status"] == "READY"
    assert child["derived_status"] == "READY"



def test_work_list_and_inspect_do_not_repair_torn_current_state(tmp_path: Path) -> None:
    state = tmp_path / "state"
    run_eiel(
        ["workManager", "create", "--work-id", "WORK-READONLY", "--title", "Read only", "--status", "ACTIVE", "--next-action", "Inspect", "--state-root", str(state)],
        state_root=state,
    )
    current = state / "work" / "records" / "WORK-READONLY.json"
    assert current.is_file()
    current.unlink()
    listing = run_eiel(["workManager", "list", "--state-root", str(state)], state_root=state)
    assert current.exists() is False
    assert any(
        row["record_id"] == "WORK-READONLY"
        and row["status"] == "RECOVERABLE_STALE"
        and row["repair_action"] == "promote_latest_history"
        for row in listing["work_items"]
    )
    run_eiel(["workManager", "inspect", "WORK-READONLY", "--state-root", str(state)], state_root=state, expected=2)
    assert current.exists() is False

def test_store_put_updates_one_index_row_without_full_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ActorRecordStore(tmp_path / "state", "probe")
    first = new_record(
        actor="probe", record_type="probe", record_id="REC-ONE", status="READY", payload={"value": 1}, generated_at="2026-08-06T00:00:00Z"
    )
    store.put(first)

    def forbidden_rebuild() -> None:
        raise AssertionError("ordinary put must not rebuild the entire namespace index")

    monkeypatch.setattr(store, "rebuild_index_locked", forbidden_rebuild)
    second = new_record(
        actor="probe", record_type="probe", record_id="REC-TWO", status="READY", payload={"value": 2}, generated_at="2026-08-06T00:01:00Z"
    )
    store.put(second)
    assert [row["record_id"] for row in store.list_read_only()] == ["REC-ONE", "REC-TWO"]



def test_delegated_spark_requires_explicit_context_contracts_or_acknowledgement(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    legacy = package_value()
    for field in ("read_contract", "output_policy", "execution_budget"):
        legacy.pop(field, None)
    package = write_package(repo, legacy)
    create_bound(repo, state, package)
    bootstrap_id = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-06T23:00:00Z")["bootstrap"]["bootstrap_id"]
    base = [
        "sessionManager", "start", "WORK-V114", "--bootstrap-record", bootstrap_id,
        "--codex-id", "spark", "--repo-root", str(repo), "--role", "IMPLEMENTATION_AUTHORITY",
        "--model", "GPT-5.3 Spark High", "--state-root", str(state),
    ]
    blocked = run_eiel(base, state_root=state, expected=2)
    assert blocked["reason_codes"] == ["work_package_context_contract_required"]
    started = run_eiel([*base, "--acknowledge-default-context-contracts"], state_root=state)["session"]
    assert started["context_contract_status"]["delegated_ready"] is False
    assert started["context_contract_status"]["defaults_acknowledged"] is True

def test_manual_identity_assurance_is_declared(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    create_bound(repo, state, package)
    bootstrap_id = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-06T22:00:00Z")["bootstrap"]["bootstrap_id"]
    session = start_session(repo, state, bootstrap_id, codex_id="manual-codex", at="2026-08-06T22:01:00Z")
    assert session["identity_assurance"]["level"] == "DECLARED"
    assert session["identity_assurance"]["source"] == "manual_cli"
    assert session["context_contract_status"]["delegated_ready"] is True
