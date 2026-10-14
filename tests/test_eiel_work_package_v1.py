from __future__ import annotations

import json
import subprocess
from pathlib import Path

from jsonschema import Draft202012Validator

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import make_bootstrap_repo, run_bootstrap
from tests.eiel_acceptance_v118_helpers import write_review_proof_for_session
from tests.test_eiel_storage_policy_reform_v1 import _terminal_root
from tools.actors.work_package import load_work_package, validate_work_package

ROOT = Path(__file__).resolve().parents[1]
ACTORS = ROOT / "tools" / "actors"


def _write_package(
    repo: Path,
    *,
    work_id: str = "WORK-PACKAGE",
    uncertainty: str = "U2",
    authority: str = "IMPLEMENTATION_AUTHORITY",
    dependencies: list[str] | None = None,
) -> Path:
    path = repo / "work_packages" / f"{work_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    package = {
        "schema_version": "eiel_work_package_v1",
        "work_id": work_id,
        "package_version": 1,
        "title": "Bound package work",
        "goal": "Implement one bounded repository-native Work Package.",
        "scope": ["Change only the approved bounded files."],
        "owned_files": ["tools/bounded_component.py", "tests/test_bounded_component.py"],
        "forbidden_files": ["tools/actors/mission_planner.py"],
        "dependencies": list(dependencies or []),
        "acceptance_criteria": ["The bounded implementation passes its focused tests."],
        "tests": ["pytest -q tests/test_bounded_component.py"],
        "evidence": ["Focused test output and final commit identifier."],
        "stop_conditions": ["Stop and promote if a shared architecture contract must change."],
        "recommended_authority": authority,
        "recommended_model": "GPT-5.3 Spark High",
        "uncertainty_class": uncertainty,
        "read_contract": {
            "entry_files": [],
            "required_documents": [],
            "search_roots": ["tools", "tests"],
            "excluded_roots": ["evidences"],
            "known_symbols": ["VALUE"],
            "maximum_initial_files": 6,
            "broad_search_allowed": False,
        },
        "output_policy": {
            "default_output_mode": "compact",
            "maximum_display_lines": 80,
            "test_output": "failures_and_summary",
            "git_output": "short_status_or_stat",
            "json_output": "selected_fields",
            "log_output": "tail_only",
            "recursive_output_requires_reason": True,
        },
        "execution_budget": {
            "expected_tool_calls": 25,
            "handoff_recommended_after_tool_calls": 50,
            "maximum_compactions": 1,
            "expected_duration_minutes": 60,
            "context_health_policy": "advisory",
        },
    }
    path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    subprocess.run(["git", "add", path.relative_to(repo)], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", f"add {work_id} package"], cwd=repo, check=True)
    return path


def _create_bound_work(
    *,
    repo: Path,
    state: Path,
    package: Path,
    status: str = "ACTIVE",
    generated_at: str = "2026-08-04T00:01:00Z",
) -> dict:
    return run_eiel(
        [
            "workManager",
            "create",
            "--work-package",
            str(package),
            "--repo-root",
            str(repo),
            "--status",
            status,
            "--next-action",
            "Execute the exact bound Work Package.",
            "--state-root",
            str(state),
            "--generated-at",
            generated_at,
        ],
        state_root=state,
    )


def _start(
    *,
    repo: Path,
    state: Path,
    bootstrap_id: str,
    work_id: str,
    codex_id: str,
    role: str | None = None,
    generated_at: str,
) -> dict:
    args = [
        "sessionManager",
        "start",
        work_id,
        "--bootstrap-record",
        bootstrap_id,
        "--codex-id",
        codex_id,
        "--repo-root",
        str(repo),
        "--model",
        "GPT-5.3 Spark High" if role != "REVIEW_AUTHORITY" else "GPT-5.5 High",
        "--reasoning-level",
        "HIGH",
        "--state-root",
        str(state),
        "--generated-at",
        generated_at,
    ]
    if role:
        args.extend(["--role", role])
    return run_eiel(args, state_root=state)


def _setup_bound_work(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    repo = make_bootstrap_repo(tmp_path)
    package = _write_package(repo)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(
        repo=repo,
        state_root=state,
        generated_at="2026-08-04T00:00:00Z",
    )["bootstrap"]["bootstrap_id"]
    _create_bound_work(repo=repo, state=state, package=package)
    return repo, state, package, bootstrap_id


def test_work_package_schema_and_yaml_example_are_valid() -> None:
    schema = json.loads((ACTORS / "schemas" / "eiel_work_package_v1.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    package = load_work_package(ACTORS / "examples" / "work_package.example.yaml", expected_work_id="WORK-042")
    Draft202012Validator(schema).validate(package)

    session_schema = json.loads((ACTORS / "schemas" / "eiel_codex_session_v1.schema.json").read_text(encoding="utf-8"))
    legacy_session = json.loads((ACTORS / "examples" / "codex_session.example.json").read_text(encoding="utf-8"))
    for field in (
        "session_role",
        "model_assignment",
        "work_package_binding",
        "session_outcome",
        "review_verdict",
        "promotion_request",
    ):
        legacy_session.pop(field, None)
    Draft202012Validator(session_schema).validate(legacy_session)

    invalid = dict(package)
    invalid["current_phase"] = "IMPLEMENTATION"
    validation = validate_work_package(invalid)
    assert validation["valid"] is False
    assert any("unsupported fields" in problem for problem in validation["errors"])

    invalid_list = dict(package)
    invalid_list["owned_files"] = [123]
    validation = validate_work_package(invalid_list)
    assert validation["valid"] is False
    assert any("strings only" in problem for problem in validation["errors"])


def test_work_and_session_bind_the_same_exact_package(tmp_path: Path) -> None:
    repo, state, package, bootstrap_id = _setup_bound_work(tmp_path)
    work = run_eiel(["workManager", "inspect", "WORK-PACKAGE", "--state-root", str(state)], state_root=state)["work"]
    binding = work["work_package_binding"]
    binding_schema = json.loads((ACTORS / "schemas" / "eiel_work_package_binding_v1.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(binding_schema).validate(binding)
    assert binding["path"] == package.relative_to(repo).as_posix()
    assert binding["recommended_authority"] == "IMPLEMENTATION_AUTHORITY"
    assert binding["recommended_model"] == "GPT-5.3 Spark High"
    assert binding["uncertainty_class"] == "U2"

    session = _start(
        repo=repo,
        state=state,
        bootstrap_id=bootstrap_id,
        work_id="WORK-PACKAGE",
        codex_id="codex-spark-1",
        generated_at="2026-08-04T00:02:00Z",
    )["session"]
    assert session["session_role"] == "IMPLEMENTATION_AUTHORITY"
    assert session["model_assignment"]["model"] == "GPT-5.3 Spark High"
    assert session["work_package_binding"]["sha256"] == binding["sha256"]
    handoff = run_eiel(
        ["sessionManager", "handoff", session["session_id"], "--state-root", str(state)],
        state_root=state,
    )
    assert handoff["handoff"]["work_package_binding"]["sha256"] == binding["sha256"]
    assert handoff["handoff"]["session_role"] == "IMPLEMENTATION_AUTHORITY"


def test_package_bound_work_requires_evidence_entailing_review_before_completion(tmp_path: Path) -> None:
    repo, state, _, bootstrap_id = _setup_bound_work(tmp_path)
    implementation = _start(
        repo=repo,
        state=state,
        bootstrap_id=bootstrap_id,
        work_id="WORK-PACKAGE",
        codex_id="codex-spark-1",
        generated_at="2026-08-04T00:02:00Z",
    )["session"]
    component = repo / "tools" / "bounded_component.py"
    component.parent.mkdir(parents=True, exist_ok=True)
    component.write_text("VALUE = 42\n", encoding="utf-8")
    subprocess.run(["git", "add", "tools/bounded_component.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "bounded implementation"], cwd=repo, check=True)
    delivery_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stdout=subprocess.PIPE, check=True
    ).stdout.strip()
    refreshed_bootstrap = run_bootstrap(
        repo=repo, state_root=state, generated_at="2026-08-04T00:02:30Z"
    )["bootstrap"]["bootstrap_id"]
    run_eiel(
        [
            "sessionManager", "attach-bootstrap", implementation["session_id"],
            "--bootstrap-record", refreshed_bootstrap, "--repo-root", str(repo),
            "--state-root", str(state), "--generated-at", "2026-08-04T00:02:30Z",
        ],
        state_root=state,
    )
    bootstrap_id = refreshed_bootstrap
    completed = run_eiel(
        [
            "sessionManager",
            "complete",
            implementation["session_id"],
            "--delivery-commit",
            delivery_commit,
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:03:00Z",
        ],
        state_root=state,
    )["session"]
    assert completed["session_outcome"] == "DELIVERY_RECORDED"
    work = run_eiel(["workManager", "inspect", "WORK-PACKAGE", "--state-root", str(state)], state_root=state)["work"]
    assert work["derived_status"] == "REVIEW_REQUIRED"

    blocked = run_eiel(
        ["workManager", "complete", "WORK-PACKAGE", "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert blocked["reason_codes"] == ["work_review_required"]

    same_identity = run_eiel(
        [
            "sessionManager",
            "start",
            "WORK-PACKAGE",
            "--role",
            "REVIEW_AUTHORITY",
            "--bootstrap-record",
            bootstrap_id,
            "--codex-id",
            "codex-spark-1",
            "--repo-root",
            str(repo),
            "--state-root",
            str(state),
        ],
        state_root=state,
    )["session"]
    assert same_identity["session_role"] == "REVIEW_AUTHORITY"
    run_eiel(
        [
            "sessionManager", "discontinue", same_identity["session_id"],
            "--state-root", str(state),
        ],
        state_root=state,
    )

    review = _start(
        repo=repo,
        state=state,
        bootstrap_id=bootstrap_id,
        work_id="WORK-PACKAGE",
        codex_id="codex-review-1",
        role="REVIEW_AUTHORITY",
        generated_at="2026-08-04T00:04:00Z",
    )["session"]
    reviewing = run_eiel(
        ["workManager", "inspect", "WORK-PACKAGE", "--state-root", str(state)],
        state_root=state,
    )["work"]
    assert reviewing["derived_status"] == "ACTIVE"
    assert any(
        str(action.get("command") or "").endswith(review["session_id"])
        for action in reviewing["next_actions"]
    )
    review_proof = write_review_proof_for_session(state, review["session_id"])
    reviewed = run_eiel(
        [
            "sessionManager",
            "complete",
            review["session_id"],
            "--review-verdict",
            "ACCEPT",
            "--review-proof",
            str(review_proof),
            "--acceptance-reviewed",
            "The bounded implementation passes its focused tests.",
            "--tests-reviewed",
            "pytest -q tests/test_bounded_component.py",
            "--evidence-reviewed",
            "Focused test output and final commit identifier.",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:05:00Z",
        ],
        state_root=state,
    )["session"]
    assert reviewed["session_outcome"] == "REVIEW_COMPLETE"
    assert reviewed["review_verdict"] == "ACCEPT"

    ready = run_eiel(["workManager", "inspect", "WORK-PACKAGE", "--state-root", str(state)], state_root=state)["work"]
    assert ready["derived_status"] == "READY_TO_ACCEPT"
    storage_root, _storage_manifest, storage_paths = _terminal_root(
        tmp_path / "merge-runs",
        name="auto-closeout",
        retained=True,
    )
    (storage_root / "evidence_certificate.json").write_text("certificate\n", encoding="utf-8")
    (storage_root / "closeout_report.md").write_text("report\n", encoding="utf-8")
    (storage_root / "reproduction_recipe.json").write_text("{}\n", encoding="utf-8")
    accepted_result = run_eiel(
        [
            "workManager",
            "complete",
            "WORK-PACKAGE",
            "--review-session",
            review["session_id"],
            "--storage-evidence-root",
            str(storage_root),
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:06:00Z",
        ],
        state_root=state,
    )
    accepted = accepted_result["work"]
    assert accepted["status"] == "COMPLETED"
    assert accepted["accepted_review_session_id"] == review["session_id"]
    assert accepted_result["storage_merge_closeout"]["status"] == "RELEASED"
    assert accepted_result["storage_merge_closeout"]["bytes_reclaimed"] == 4096
    assert not storage_paths["scratch"].exists()
    assert storage_paths["asset"].exists()

    # A merge performed outside the managed completion command can be
    # reconciled later against the same durable integration identity.
    direct_root, _direct_manifest, direct_paths = _terminal_root(
        tmp_path / "merge-runs",
        name="direct-git-reconciliation",
        retained=True,
    )
    (direct_root / "evidence_certificate.json").write_text("certificate\n", encoding="utf-8")
    (direct_root / "closeout_report.md").write_text("report\n", encoding="utf-8")
    (direct_root / "reproduction_recipe.json").write_text("{}\n", encoding="utf-8")
    reconciled = run_eiel(
        [
            "storage",
            "reconcile-merged-work",
            "WORK-PACKAGE",
            "--repo-root",
            str(repo),
            "--storage-evidence-root",
            str(direct_root),
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:07:00Z",
        ],
        state_root=state,
    )
    assert reconciled["direct_git_reconciliation"] is True
    assert reconciled["storage_merge_closeout"]["bytes_reclaimed"] == 4096
    assert reconciled["storage_merge_closeout"]["status"] == "RELEASED"
    assert not direct_paths["scratch"].exists()
    assert direct_paths["asset"].exists()


def test_promotion_stops_the_session_and_prevents_resume(tmp_path: Path) -> None:
    repo, state, package, bootstrap_id = _setup_bound_work(tmp_path)
    session = _start(
        repo=repo,
        state=state,
        bootstrap_id=bootstrap_id,
        work_id="WORK-PACKAGE",
        codex_id="codex-spark-1",
        generated_at="2026-08-04T00:02:00Z",
    )["session"]
    promoted = run_eiel(
        [
            "sessionManager",
            "request-promotion",
            session["session_id"],
            "--requested-class",
            "U4",
            "--reason",
            "The implementation exposes two incompatible ownership contracts.",
            "--affected",
            "tools/shared_contract.py",
            "--safe-state",
            "All bounded changes are committed on the implementation branch and tests remain green.",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:03:00Z",
        ],
        state_root=state,
    )["session"]
    assert promoted["status"] == "BLOCKED"
    assert promoted["session_outcome"] == "PROMOTION_REQUESTED"
    assert promoted["promotion_request"]["requested_uncertainty_class"] == "U4"
    work = run_eiel(["workManager", "inspect", "WORK-PACKAGE", "--state-root", str(state)], state_root=state)["work"]
    assert work["derived_status"] == "PROMOTION_REQUIRED"
    forced_completion = run_eiel(
        [
            "workManager",
            "complete",
            "WORK-PACKAGE",
            "--force",
            "--state-root",
            str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert forced_completion["reason_codes"] == ["work_promotion_unresolved"]
    blocked = run_eiel(
        [
            "sessionManager",
            "resume",
            session["session_id"],
            "--codex-id",
            "codex-spark-1",
            "--repo-root",
            str(repo),
            "--state-root",
            str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert blocked["reason_codes"] == ["session_promotion_unresolved"]

    run_eiel(
        [
            "sessionManager",
            "discontinue",
            session["session_id"],
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:04:00Z",
        ],
        state_root=state,
    )
    still_promoted = run_eiel(
        ["workManager", "inspect", "WORK-PACKAGE", "--state-root", str(state)],
        state_root=state,
    )["work"]
    assert still_promoted["derived_status"] == "PROMOTION_REQUIRED"

    revised_package = json.loads(package.read_text(encoding="utf-8"))
    revised_package["package_version"] = 2
    revised_package["goal"] = "Implement the clarified ownership contract selected by Engineering Authority."
    revised_path = package.with_name("WORK-PACKAGE-v2.json")
    revised_path.write_text(json.dumps(revised_package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    subprocess.run(["git", "add", revised_path.relative_to(repo)], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add revised package"], cwd=repo, check=True)
    run_eiel(
        [
            "workManager",
            "update",
            "WORK-PACKAGE",
            "--work-package",
            str(revised_path),
            "--repo-root",
            str(repo),
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-04T00:05:00Z",
        ],
        state_root=state,
    )
    revised = run_eiel(
        ["workManager", "inspect", "WORK-PACKAGE", "--state-root", str(state)],
        state_root=state,
    )["work"]
    assert revised["derived_status"] != "PROMOTION_REQUIRED"
    assert revised["work_package_binding"]["package_version"] == 2
    assert revised["work_package_binding"]["path"] == revised_path.relative_to(repo).as_posix()

    historical = run_eiel(
        ["sessionManager", "inspect", session["session_id"], "--state-root", str(state)],
        state_root=state,
    )["session"]
    assert historical["valid"] is True
    assert historical["work_package_binding"]["package_version"] == 1
    audit = run_eiel(["workspace", "audit", "--repo-root", str(repo), "--state-root", str(state)], state_root=state)
    assert audit["status"] in {"CONSISTENT", "CONSISTENT_WITH_WARNINGS"}
    assert not any("digest differs" in problem for problem in audit["audit"]["problems"])


def test_package_drift_fails_closed_and_workspace_exposes_review_state(tmp_path: Path) -> None:
    repo, state, package, bootstrap_id = _setup_bound_work(tmp_path)
    implementation = _start(
        repo=repo,
        state=state,
        bootstrap_id=bootstrap_id,
        work_id="WORK-PACKAGE",
        codex_id="codex-spark-1",
        generated_at="2026-08-04T00:02:00Z",
    )["session"]
    component = repo / "tools" / "bounded_component.py"
    component.parent.mkdir(parents=True, exist_ok=True)
    component.write_text("VALUE = 42\n", encoding="utf-8")
    subprocess.run(["git", "add", "tools/bounded_component.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "bounded implementation"], cwd=repo, check=True)
    run_eiel(
        ["sessionManager", "complete", implementation["session_id"], "--delivery-commit", subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stdout=subprocess.PIPE, check=True).stdout.strip(), "--state-root", str(state)],
        state_root=state,
    )
    workspace = run_eiel(
        [
            "workspace",
            "open",
            "--work-id",
            "WORK-PACKAGE",
            "--bootstrap-record",
            bootstrap_id,
            "--repo-root",
            str(repo),
            "--state-root",
            str(state),
        ],
        state_root=state,
    )
    assert workspace["status"] == "WORK_REVIEW_REQUIRED"
    selected = workspace["workspace"]["selected_work"]
    assert selected["summary"]["derived_status"] == "REVIEW_REQUIRED"
    assert selected["payload"]["work_package_binding"]["uncertainty_class"] == "U2"

    value = json.loads(package.read_text(encoding="utf-8"))
    value["goal"] = "Silently changed contract, which must invalidate the binding."
    package.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inspected = run_eiel(
        ["workManager", "inspect", "WORK-PACKAGE", "--state-root", str(state)],
        state_root=state,
    )
    assert inspected["work"]["valid"] is False
    assert any("material contract changed" in problem for problem in inspected["problems"])


def test_manager_owned_package_and_review_fields_cannot_be_patched(tmp_path: Path) -> None:
    repo, state, _, bootstrap_id = _setup_bound_work(tmp_path)
    work_patch = run_eiel(
        [
            "workManager",
            "update",
            "WORK-PACKAGE",
            "--set",
            "work_package_binding.sha256=deadbeef",
            "--state-root",
            str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert work_patch["reason_codes"] == ["work_managed_field_patch_forbidden"]

    session = _start(
        repo=repo,
        state=state,
        bootstrap_id=bootstrap_id,
        work_id="WORK-PACKAGE",
        codex_id="codex-spark-1",
        generated_at="2026-08-04T00:02:00Z",
    )["session"]
    session_patch = run_eiel(
        [
            "sessionManager",
            "checkpoint",
            session["session_id"],
            "--set",
            "session_role=REVIEW_AUTHORITY",
            "--state-root",
            str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert session_patch["reason_codes"] == ["session_managed_field_patch_forbidden"]
