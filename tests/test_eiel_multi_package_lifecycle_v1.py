from __future__ import annotations

import subprocess
from pathlib import Path

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import make_bootstrap_repo, run_bootstrap
from tests.eiel_acceptance_v118_helpers import write_review_proof_for_session
from tests.test_eiel_work_package_v1_8 import (
    _commit,
    _package_value,
    _refresh_bootstrap,
    _start,
    _write_package,
)
from tools.actors.common import revise_record
from tools.actors.portfolio import work_store
from tools.actors.work_manager import persist_work_record_fixture


def _complete_delivery(repo: Path, state: Path, session: dict, *, text: str, message: str, at: str) -> dict:
    commit = _commit(repo, "tools/bounded_component.py", text, message)
    _refresh_bootstrap(repo, state, session["session_id"], generated_at=at)
    return run_eiel(
        [
            "sessionManager", "complete", session["session_id"], "--delivery-commit", commit,
            "--state-root", str(state), "--generated-at", at,
        ],
        state_root=state,
    )["session"]


def _create_chain(
    repo: Path,
    state: Path,
    bootstrap_id: str,
    packages: list[Path],
    *,
    intermediate_review_sequences: list[int] | None = None,
) -> dict:
    args = [
        "workManager", "create", "--status", "ACTIVE", "--repo-root", str(repo),
        "--state-root", str(state), "--generated-at", "2026-08-08T10:01:00Z",
    ]
    for package in packages:
        args.extend(["--work-package", str(package)])
    for sequence in intermediate_review_sequences or []:
        args.extend(["--intermediate-review-sequence", str(sequence)])
    return run_eiel(args, state_root=state)["work"]


def test_three_package_chain_advances_without_intermediate_review_and_final_review_binds_lineage(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    packages = []
    for index in range(1, 4):
        value = _package_value()
        value["goal"] += f" Stage {index}."
        packages.append(_write_package(repo, value, name=f"WORK-V18-WP{index}.json"))
    bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-08T10:00:00Z")["bootstrap"]["bootstrap_id"]
    created = _create_chain(repo, state, bootstrap, packages)
    assert [row["binding"]["path"] for row in created["work_package_chain"]] == [p.relative_to(repo).as_posix() for p in packages]

    wp1 = _start(repo, state, bootstrap, codex_id="chain-impl-1", generated_at="2026-08-08T10:02:00Z")
    _complete_delivery(repo, state, wp1, text="VALUE = 1\n", message="WP1 delivery", at="2026-08-08T10:03:00Z")
    inspected = run_eiel(["workManager", "inspect", "WORK-V18", "--state-root", str(state)], state_root=state)["work"]
    assert inspected["derived_status"] == "ACTIVE"
    assert inspected["review"]["next_package_binding"]["path"] == packages[1].relative_to(repo).as_posix()
    assert inspected["review"]["accepted_review_sessions"] == []

    bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-08T10:04:00Z")["bootstrap"]["bootstrap_id"]
    wp2 = _start(repo, state, bootstrap, codex_id="chain-impl-2", generated_at="2026-08-08T10:05:00Z")
    assert wp2["work_package_binding"]["path"] == packages[1].relative_to(repo).as_posix()
    _complete_delivery(repo, state, wp2, text="VALUE = 2\n", message="WP2 delivery", at="2026-08-08T10:06:00Z")
    inspected = run_eiel(["workManager", "inspect", "WORK-V18", "--state-root", str(state)], state_root=state)["work"]
    assert inspected["review"]["next_package_binding"]["path"] == packages[2].relative_to(repo).as_posix()

    bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-08T10:07:00Z")["bootstrap"]["bootstrap_id"]
    wp3 = _start(repo, state, bootstrap, codex_id="chain-impl-3", generated_at="2026-08-08T10:08:00Z")
    assert wp3["work_package_binding"]["path"] == packages[2].relative_to(repo).as_posix()
    _complete_delivery(repo, state, wp3, text="VALUE = 3\n", message="WP3 delivery", at="2026-08-08T10:09:00Z")

    bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-08T10:10:00Z")["bootstrap"]["bootstrap_id"]
    ready = run_eiel(["workManager", "inspect", "WORK-V18", "--state-root", str(state)], state_root=state)["work"]
    assert ready["derived_status"] == "REVIEW_REQUIRED"
    review = _start(repo, state, bootstrap, codex_id="chain-review-1", role="REVIEW_AUTHORITY", generated_at="2026-08-08T10:11:00Z")
    lineage = review["review_target"]["integrated_delivery_lineage"]
    assert [row["package_sha256"] for row in lineage] == [row["binding"]["sha256"] for row in ready["work_package_chain"]]
    assert [row["implementation_session_id"] for row in lineage] == [wp1["session_id"], wp2["session_id"], wp3["session_id"]]
    review_proof = write_review_proof_for_session(state, review["session_id"])
    reviewed = run_eiel(
        [
            "sessionManager", "complete", review["session_id"], "--review-verdict", "ACCEPT",
            "--review-proof", str(review_proof),
            "--acceptance-reviewed", "The bounded component returns the reviewed value.",
            "--tests-reviewed", "pytest -q tests/test_bounded_component.py",
            "--evidence-reviewed", "Focused test output and final commit.",
            "--state-root", str(state), "--generated-at", "2026-08-08T10:12:00Z",
        ],
        state_root=state,
    )["session"]
    assert reviewed["review_evidence"]["integrated_delivery_lineage"] == lineage
    completed = run_eiel(
        ["workManager", "complete", "WORK-V18", "--review-session", review["session_id"], "--state-root", str(state), "--generated-at", "2026-08-08T10:13:00Z"],
        state_root=state,
    )["work"]
    assert completed["status"] == "COMPLETED"


def test_explicit_intermediate_review_gate_remains_available(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    value1 = _package_value()
    value1["goal"] += " Gate stage 1."
    value2 = _package_value()
    value2["goal"] += " Gate stage 2."
    package1 = _write_package(repo, value1, name="WORK-V18-GATE1.json")
    package2 = _write_package(repo, value2, name="WORK-V18-GATE2.json")
    bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-08T11:00:00Z")["bootstrap"]["bootstrap_id"]
    args = [
        "workManager", "create", "--status", "ACTIVE", "--repo-root", str(repo),
        "--work-package", str(package1), "--work-package", str(package2),
        "--intermediate-review-sequence", "1", "--state-root", str(state),
    ]
    run_eiel(args, state_root=state)
    wp1 = _start(repo, state, bootstrap, codex_id="gate-impl", generated_at="2026-08-08T11:01:00Z")
    _complete_delivery(repo, state, wp1, text="VALUE = 4\n", message="gated WP1", at="2026-08-08T11:02:00Z")
    inspected = run_eiel(["workManager", "inspect", "WORK-V18", "--state-root", str(state)], state_root=state)["work"]
    assert inspected["derived_status"] == "REVIEW_REQUIRED"
    assert inspected["review"]["intermediate_review_required"] is True
    assert inspected["review"]["next_package_binding"] is None


def test_repair_chain_reopens_legacy_intermediate_completion_for_next_package(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    packages = []
    for index in range(1, 4):
        value = _package_value()
        value["goal"] += f" Repair stage {index}."
        packages.append(_write_package(repo, value, name=f"WORK-V18-REPAIR-WP{index}.json"))
    bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-08T12:00:00Z")["bootstrap"]["bootstrap_id"]
    created = _create_chain(
        repo,
        state,
        bootstrap,
        packages,
        intermediate_review_sequences=[2],
    )
    assert created["work_id"] == "WORK-V18"

    wp1 = _start(repo, state, bootstrap, codex_id="repair-impl-1", generated_at="2026-08-08T12:01:00Z")
    _complete_delivery(repo, state, wp1, text="VALUE = 1\n", message="repair WP1", at="2026-08-08T12:02:00Z")
    bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-08T12:03:00Z")["bootstrap"]["bootstrap_id"]
    wp2 = _start(repo, state, bootstrap, codex_id="repair-impl-2", generated_at="2026-08-08T12:04:00Z")
    _complete_delivery(repo, state, wp2, text="VALUE = 2\n", message="repair WP2", at="2026-08-08T12:05:00Z")
    bootstrap = run_bootstrap(repo=repo, state_root=state, generated_at="2026-08-08T12:06:00Z")["bootstrap"]["bootstrap_id"]
    review = _start(
        repo,
        state,
        bootstrap,
        codex_id="repair-review-2",
        role="REVIEW_AUTHORITY",
        target=wp2["session_id"],
        generated_at="2026-08-08T12:07:00Z",
    )
    review_proof = write_review_proof_for_session(state, review["session_id"])
    accepted = run_eiel(
        [
            "sessionManager", "complete", review["session_id"], "--review-verdict", "ACCEPT",
            "--review-proof", str(review_proof),
            "--acceptance-reviewed", "The bounded component returns the reviewed value.",
            "--tests-reviewed", "pytest -q tests/test_bounded_component.py",
            "--evidence-reviewed", "Focused test output and final commit.",
            "--state-root", str(state), "--generated-at", "2026-08-08T12:08:00Z",
        ],
        state_root=state,
    )["session"]

    record = work_store(state).get("WORK-V18")
    legacy_payload = dict(record["payload"])
    legacy_payload["work_package_binding"] = dict(legacy_payload["work_package_chain"][1]["binding"])
    legacy_payload["accepted_review_session_id"] = accepted["session_id"]
    legacy_payload["acceptance_evidence"] = {"historical": "intermediate completion"}
    legacy = revise_record(
        record,
        payload=legacy_payload,
        status="COMPLETED",
        action="legacy_intermediate_completion",
        generated_at="2026-08-08T12:09:00Z",
    )
    persist_work_record_fixture(legacy, state_root=state, allow_update=True)

    repaired = run_eiel(
        [
            "workManager", "repair-chain", "WORK-V18",
            "--reason", "Historical completion occurred after WP2 review; WP3 remains undelivered.",
            "--acknowledge-intermediate-completion",
            "--state-root", str(state), "--generated-at", "2026-08-08T12:10:00Z",
        ],
        state_root=state,
    )["work"]
    assert repaired["status"] == "READY"
    assert repaired["work_package_binding"]["sha256"] == repaired["work_package_chain"][0]["binding"]["sha256"]
    assert repaired["review"]["next_package_binding"]["path"] == packages[2].relative_to(repo).as_posix()
    assert repaired["accepted_review_session_id"] is None
    assert repaired["lifecycle_repair"]["next_package_sequence"] == 3
