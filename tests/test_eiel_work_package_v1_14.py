from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import make_bootstrap_repo, run_bootstrap
from tests.eiel_acceptance_v118_helpers import write_review_proof_for_session
from tools.actors.common import ActorError
from tools.actors.work_package import (
    CANONICAL_HASH_KIND,
    DEFAULT_EXECUTION_BUDGET,
    DEFAULT_OUTPUT_POLICY,
    DEFAULT_READ_CONTRACT,
    VERIFICATION_EXECUTION_STRICT,
    bind_work_package,
    validate_changed_paths,
    validate_work_package,
    verify_work_package_binding,
    work_package_contract_sha256,
)


def package_value(*, work_id: str = "WORK-V114", version: int = 1) -> dict:
    return {
        "schema_version": "eiel_work_package_v1",
        "work_id": work_id,
        "package_version": version,
        "title": "V1.14 Work Package authority fixture",
        "goal": "Implement one bounded component without widening authority.",
        "scope": ["Change only the bounded component."],
        "owned_files": ["tools/bounded_component.py", "tests/test_bounded_component.py"],
        "forbidden_files": ["tools/actors/work_manager.py"],
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


def write_package(repo: Path, value: dict, *, name: str = "WORK-V114.json", commit: bool = True) -> Path:
    path = repo / "work_packages" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if commit:
        subprocess.run(["git", "add", path.relative_to(repo)], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", f"add {name}"], cwd=repo, check=True)
    return path


def create_bound(repo: Path, state: Path, package: Path, *, at: str = "2026-08-06T10:01:00Z") -> dict:
    return run_eiel(
        [
            "workManager", "create", "--work-package", str(package), "--repo-root", str(repo),
            "--status", "ACTIVE", "--next-action", "Execute the bounded package.",
            "--state-root", str(state), "--generated-at", at,
        ],
        state_root=state,
    )["work"]


def start_session(
    repo: Path,
    state: Path,
    bootstrap_id: str,
    *,
    codex_id: str,
    at: str,
    role: str = "IMPLEMENTATION_AUTHORITY",
    target: str | None = None,
    allow_parallel: bool = False,
) -> dict:
    args = [
        "sessionManager", "start", "WORK-V114", "--bootstrap-record", bootstrap_id,
        "--codex-id", codex_id, "--repo-root", str(repo), "--role", role,
        "--model", "GPT-5.5 High" if role == "REVIEW_AUTHORITY" else "GPT-5.3 Spark High",
        "--reasoning-level", "HIGH", "--state-root", str(state), "--generated-at", at,
    ]
    if target:
        args.extend(["--review-target-session", target])
    if allow_parallel:
        args.append("--allow-parallel")
    return run_eiel(args, state_root=state)["session"]


def commit_file(repo: Path, relative: str, text: str, message: str) -> str:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", relative], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def refresh(repo: Path, state: Path, session_id: str, *, at: str) -> None:
    bootstrap_id = run_bootstrap(repo=repo, state_root=state, generated_at=at)["bootstrap"]["bootstrap_id"]
    run_eiel(
        [
            "sessionManager", "attach-bootstrap", session_id, "--bootstrap-record", bootstrap_id,
            "--repo-root", str(repo), "--state-root", str(state), "--generated-at", at,
        ],
        state_root=state,
    )


def deliver(repo: Path, state: Path, session: dict, *, relative: str, value: int, at: str) -> dict:
    commit_file(repo, relative, f"VALUE = {value}\n", f"delivery {value}")
    refresh(repo, state, session["session_id"], at=at)
    delivery_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stdout=subprocess.PIPE, check=True).stdout.strip()
    return run_eiel(
        ["sessionManager", "complete", session["session_id"], "--delivery-commit", delivery_commit, "--state-root", str(state), "--generated-at", at],
        state_root=state,
    )["session"]


def complete_review(
    state: Path,
    session_id: str,
    verdict: str,
    *,
    at: str,
    supersedes: str | None = None,
    limitations: list[str] | None = None,
) -> dict:
    args = [
        "sessionManager", "complete", session_id, "--review-verdict", verdict,
        "--state-root", str(state), "--generated-at", at,
    ]
    if supersedes:
        args.extend(["--supersedes-review", supersedes])
    if verdict == "ACCEPT":
        proof = write_review_proof_for_session(state, session_id, limitations=limitations)
        args.extend(["--review-proof", str(proof)])
    else:
        args.extend(["--review-finding", f"bounded {verdict.lower()} finding"])
    return run_eiel(args, state_root=state)["session"]


def bootstrap(repo: Path, state: Path, *, at: str) -> str:
    return run_bootstrap(repo=repo, state_root=state, generated_at=at)["bootstrap"]["bootstrap_id"]


def test_exact_owned_path_never_expands_to_subtree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    package = package_value()
    package["owned_files"] = ["target"]
    source = write_package(repo, package, commit=False)
    binding, loaded = bind_work_package(source, repo_root=repo)
    assert binding["sha256_kind"] == CANONICAL_HASH_KIND
    assert validate_changed_paths(["target/child.py"], loaded, repo_root=repo)["valid"] is False
    (repo / "target").mkdir()
    assert validate_changed_paths(["target/child.py"], loaded, repo_root=repo)["valid"] is False
    package["owned_files"] = ["target/**"]
    source = write_package(repo, package, name="subtree.json", commit=False)
    _, subtree = bind_work_package(source, repo_root=repo)
    assert validate_changed_paths(["target/child.py"], subtree, repo_root=repo)["valid"] is True


def test_contract_sufficiency_hash_defaults_and_set_order() -> None:
    for field in ("scope", "owned_files", "acceptance_criteria", "tests", "evidence", "stop_conditions"):
        value = package_value()
        value[field] = []
        result = validate_work_package(value)
        assert result["valid"] is False
        assert any(field in error for error in result["errors"])
    review_package = package_value()
    review_package["recommended_authority"] = "REVIEW_AUTHORITY"
    assert validate_work_package(review_package)["valid"] is False
    oversized = package_value()
    oversized["goal"] = "x" * 5000
    oversized_result = validate_work_package(oversized)
    assert oversized_result["valid"] is False
    assert any("too long" in error for error in oversized_result["errors"])

    omitted = package_value()
    explicit = package_value()
    for field in ("read_contract", "output_policy", "execution_budget"):
        omitted.pop(field, None)
        explicit.pop(field, None)
    explicit["read_contract"] = dict(DEFAULT_READ_CONTRACT)
    explicit["output_policy"] = dict(DEFAULT_OUTPUT_POLICY)
    explicit["execution_budget"] = dict(DEFAULT_EXECUTION_BUDGET)
    assert work_package_contract_sha256(omitted) == work_package_contract_sha256(explicit)

    reordered = package_value()
    reordered["owned_files"] = list(reversed(reordered["owned_files"]))
    reordered["tests"] = ["second", *reordered["tests"]]
    baseline = package_value()
    baseline["tests"] = [*baseline["tests"], "second"]
    assert work_package_contract_sha256(reordered) == work_package_contract_sha256(baseline)


def test_active_binding_requires_committed_git_source_and_strict_provenance(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    package = write_package(repo, package_value(), commit=False)
    with pytest.raises(ActorError) as exc:
        bind_work_package(package, repo_root=repo, require_git_provenance=True)
    assert exc.value.code == "work_package_git_provenance_invalid"
    subprocess.run(["git", "add", package.relative_to(repo)], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "track package"], cwd=repo, check=True)
    binding, _ = bind_work_package(package, repo_root=repo, require_git_provenance=True)
    assert binding["source_commit"]
    assert verify_work_package_binding(
        binding, repo_root=repo, expected_work_id="WORK-V114", verification_mode=VERIFICATION_EXECUTION_STRICT
    )["valid"] is True
    package.write_text(package.read_text(encoding="utf-8").replace("V1.14", "V1.14 advisory"), encoding="utf-8")
    check = verify_work_package_binding(
        binding, repo_root=repo, expected_work_id="WORK-V114", verification_mode=VERIFICATION_EXECUTION_STRICT
    )
    assert check["valid"] is False


def test_accept_must_cover_declared_contract(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    create_bound(repo, state, package)
    implementation = start_session(repo, state, bootstrap(repo, state, at="2026-08-06T11:00:00Z"), codex_id="impl", at="2026-08-06T11:01:00Z")
    deliver(repo, state, implementation, relative="tools/bounded_component.py", value=1, at="2026-08-06T11:02:00Z")
    review = start_session(repo, state, bootstrap(repo, state, at="2026-08-06T11:03:00Z"), codex_id="review", role="REVIEW_AUTHORITY", target=implementation["session_id"], at="2026-08-06T11:04:00Z")
    bad_proof = write_review_proof_for_session(
        state,
        review["session_id"],
        obligation_overrides={"acceptance_criterion": ["unrelated criterion"]},
    )
    failed = run_eiel(
        [
            "sessionManager", "complete", review["session_id"], "--review-verdict", "ACCEPT",
            "--review-proof", str(bad_proof), "--state-root", str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert failed["reason_codes"] == ["review_obligation_proof_invalid"]


def test_review_conflict_requires_explicit_supersession_for_same_delivery(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    create_bound(repo, state, package)
    implementation = start_session(repo, state, bootstrap(repo, state, at="2026-08-06T12:00:00Z"), codex_id="impl", at="2026-08-06T12:01:00Z")
    deliver(repo, state, implementation, relative="tools/bounded_component.py", value=1, at="2026-08-06T12:02:00Z")

    first = start_session(repo, state, bootstrap(repo, state, at="2026-08-06T12:03:00Z"), codex_id="review-a", role="REVIEW_AUTHORITY", target=implementation["session_id"], at="2026-08-06T12:04:00Z")
    complete_review(state, first["session_id"], "ACCEPT", at="2026-08-06T12:05:00Z")
    second = start_session(repo, state, bootstrap(repo, state, at="2026-08-06T12:06:00Z"), codex_id="review-b", role="REVIEW_AUTHORITY", target=implementation["session_id"], at="2026-08-06T12:07:00Z")
    complete_review(
        state, second["session_id"], "REJECT", at="2026-08-06T12:08:00Z",
        supersedes=first["session_id"],
    )

    inspected = run_eiel(
        ["workManager", "inspect", "WORK-V114", "--repo-root", str(repo), "--state-root", str(state)],
        state_root=state,
    )["work"]
    assert inspected["review"]["authoritative_review_session_id"] == second["session_id"]
    assert inspected["review"]["latest_verdict"] == "REJECT"
    assert inspected["review"]["accepted_review_sessions"] == []
    failed = run_eiel(
        ["workManager", "complete", "WORK-V114", "--review-session", first["session_id"], "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert failed["reason_codes"] == ["work_review_not_accepted"]


def test_terminal_note_does_not_resurrect_old_delivery(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    create_bound(repo, state, package)
    first = start_session(repo, state, bootstrap(repo, state, at="2026-08-06T13:00:00Z"), codex_id="impl-a", at="2026-08-06T13:01:00Z")
    deliver(repo, state, first, relative="tools/bounded_component.py", value=1, at="2026-08-06T13:02:00Z")
    review1 = start_session(repo, state, bootstrap(repo, state, at="2026-08-06T13:03:00Z"), codex_id="review-a", role="REVIEW_AUTHORITY", target=first["session_id"], at="2026-08-06T13:04:00Z")
    complete_review(state, review1["session_id"], "ACCEPT", at="2026-08-06T13:05:00Z")

    second = start_session(repo, state, bootstrap(repo, state, at="2026-08-06T13:06:00Z"), codex_id="impl-b", at="2026-08-06T13:07:00Z")
    deliver(repo, state, second, relative="tools/bounded_component.py", value=2, at="2026-08-06T13:08:00Z")
    run_eiel(
        ["sessionManager", "note", first["session_id"], "historical annotation", "--state-root", str(state), "--generated-at", "2026-08-06T14:00:00Z"],
        state_root=state,
    )
    inspected = run_eiel(
        ["workManager", "inspect", "WORK-V114", "--repo-root", str(repo), "--state-root", str(state)],
        state_root=state,
    )["work"]
    assert inspected["review"]["latest_delivery_session_id"] == second["session_id"]
    assert inspected["review"]["authoritative_review_session_id"] is None
    assert inspected["review"]["accepted_review_sessions"] == []
    failed = run_eiel(
        ["workManager", "complete", "WORK-V114", "--review-session", review1["session_id"], "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert failed["reason_codes"] == ["work_review_not_accepted"]


def test_divergent_parallel_deliveries_block_review(tmp_path: Path) -> None:
    repo_a = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo_a, package_value())
    create_bound(repo_a, state, package)
    repo_b = tmp_path / "lane-b"
    subprocess.run(["git", "worktree", "add", "-q", "-b", "lane-b", str(repo_b), "HEAD"], cwd=repo_a, check=True)
    for marker in ("tests", "docs", "evidences"):
        (repo_b / marker).mkdir(parents=True, exist_ok=True)

    lane_a = start_session(repo_a, state, bootstrap(repo_a, state, at="2026-08-06T15:00:00Z"), codex_id="lane-a", at="2026-08-06T15:01:00Z", allow_parallel=True)
    lane_b = start_session(repo_b, state, bootstrap(repo_b, state, at="2026-08-06T15:02:00Z"), codex_id="lane-b", at="2026-08-06T15:03:00Z", allow_parallel=True)
    deliver(repo_a, state, lane_a, relative="tools/bounded_component.py", value=1, at="2026-08-06T15:04:00Z")
    deliver(repo_b, state, lane_b, relative="tools/bounded_component.py", value=2, at="2026-08-06T15:05:00Z")

    failed = run_eiel(
        [
            "sessionManager", "start", "WORK-V114", "--bootstrap-record", bootstrap(repo_a, state, at="2026-08-06T15:06:00Z"),
            "--codex-id", "review", "--repo-root", str(repo_a), "--role", "REVIEW_AUTHORITY",
            "--model", "GPT-5.5 High", "--reasoning-level", "HIGH", "--state-root", str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert failed["reason_codes"] == ["review_delivery_lineage_conflict"]


def test_work_binding_resolves_in_equivalent_replacement_root(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package = write_package(repo, package_value())
    create_bound(repo, state, package)
    replacement = tmp_path / "replacement"
    subprocess.run(["git", "clone", "-q", str(repo), str(replacement)], check=True)
    shutil.rmtree(repo)
    inspected = run_eiel(
        ["workManager", "inspect", "WORK-V114", "--repo-root", str(replacement), "--state-root", str(state)],
        state_root=state,
    )["work"]
    assert inspected["validation"]["valid"] is True


def test_delivery_cannot_modify_historical_package_source(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    package1 = write_package(repo, package_value())
    create_bound(repo, state, package1)
    first = start_session(
        repo,
        state,
        bootstrap(repo, state, at="2026-08-06T16:00:00Z"),
        codex_id="historical-session",
        at="2026-08-06T16:01:00Z",
    )
    run_eiel(
        ["sessionManager", "discontinue", first["session_id"], "--state-root", str(state), "--generated-at", "2026-08-06T16:02:00Z"],
        state_root=state,
    )

    value2 = package_value(version=2)
    value2["goal"] = "Implement the revised bounded component while preserving package history."
    value2["owned_files"] = ["tools/bounded_component.py", "work_packages/WORK-V114.json"]
    package2 = write_package(repo, value2, name="WORK-V114-v2.json")
    run_eiel(
        [
            "workManager", "update", "WORK-V114", "--work-package", str(package2), "--repo-root", str(repo),
            "--state-root", str(state), "--generated-at", "2026-08-06T16:03:00Z",
        ],
        state_root=state,
    )
    second = start_session(
        repo,
        state,
        bootstrap(repo, state, at="2026-08-06T16:04:00Z"),
        codex_id="new-session",
        at="2026-08-06T16:05:00Z",
    )
    subprocess.run(["git", "rm", "-q", package1.relative_to(repo)], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "attempt to delete historical package"], cwd=repo, check=True)
    refresh(repo, state, second["session_id"], at="2026-08-06T16:06:00Z")
    failed = run_eiel(
        ["sessionManager", "complete", second["session_id"], "--delivery-commit", subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stdout=subprocess.PIPE, check=True).stdout.strip(), "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert failed["reason_codes"] == ["work_package_history_path_changed"]
