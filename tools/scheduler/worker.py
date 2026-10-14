"""Deterministic fake worker used to qualify the scheduler without Codex."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from tools.actors.bootstrap import inspect_bootstrap
from tools.actors.common import ActorError, file_sha256, utc_now
from tools.actors.portfolio import bootstrap_store, session_store
from tools.actors.work_package import load_work_package, resolve_work_package_binding
from tools.scheduler.contracts import assignment_store, load_assignment_payload
from tools.actors.session_manager import attach_session_bootstrap_after_commit
from tools.scheduler.worker_protocol import open_worker, result_from_repository, write_worker_result


def _run(args: list[str], *, cwd: Path) -> dict[str, Any]:
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_root if not prior else source_root + os.pathsep + prior
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False, env=env)
    if result.returncode != 0:
        raise ActorError(
            "fake worker manager command failed",
            code="fake_worker_command_failed",
            problems=[" ".join(args), result.stdout[-2000:], result.stderr[-2000:]],
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ActorError("fake worker received non-JSON manager output", code="fake_worker_command_failed") from exc


def _manager(state_root: Path, repo_root: Path, *args: str) -> dict[str, Any]:
    # Run the operator tooling from its installed/source package rather than
    # trusting a fixture or partially checked-out target repository to contain
    # the CLI implementation. Repository-sensitive commands carry their bound
    # repo root in Session/Work records.
    source_root = Path(__file__).resolve().parents[2]
    return _run([sys.executable, "-m", "tools.eiel", *args, "--state-root", str(state_root)], cwd=source_root)




def _assignment_work_package(repo_root: Path, assignment: dict[str, Any]) -> dict[str, Any]:
    material = dict(load_assignment_payload(assignment.get("payload"), require_ready=True).get("material") or {})
    work = dict(material.get("work") or {})
    binding = dict(work.get("work_package_binding") or {})
    package_path = resolve_work_package_binding(binding, repo_root=repo_root)
    return load_work_package(package_path, expected_work_id=str(work.get("record_id") or ""))

def _commit_fixture(repo_root: Path, assignment: dict[str, Any], content: str) -> tuple[str, str]:
    package = _assignment_work_package(repo_root, assignment)
    owned = list(package.get("owned_files") or [])
    if not owned:
        raise ActorError("fake implementation package owns no files", code="fake_worker_scope_empty")
    target_rel = next((row for row in owned if not any(ch in row for ch in "*?[]") and not row.endswith("/")), None)
    if not target_rel:
        raise ActorError("fake implementation requires one literal owned file", code="fake_worker_scope_empty")
    target = repo_root / target_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "--", target_rel], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "Fake worker bounded implementation"], cwd=repo_root, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    return target_rel, head


def _refresh_bootstrap_after_commit(state_root: Path, repo_root: Path, session_id: str) -> str:
    """Create a new verify-only receipt for the worker's committed HEAD."""

    session = session_store(state_root).get(session_id)
    bootstrap_id = str(dict(session.get("payload") or {}).get("bootstrap_record_id") or "")
    prior = bootstrap_store(state_root).get(bootstrap_id)
    payload = dict(prior.get("payload") or {})
    task = dict(payload.get("task") or {})
    authority = dict(payload.get("authority") or {})
    preflight = dict(authority.get("preflight") or {})
    scopes = []
    for row in task.get("write_scope") or []:
        if isinstance(row, dict) and row.get("path"):
            scopes.append(str(row["path"]))
        elif isinstance(row, str):
            scopes.append(row)
    external_evidence_roots = []
    for row in task.get("external_evidence_roots") or []:
        if isinstance(row, dict) and row.get("path"):
            external_evidence_roots.append(str(row["path"]))
        elif isinstance(row, str):
            external_evidence_roots.append(row)
    refreshed, record = inspect_bootstrap(
        repo_root_value=repo_root,
        state_root_value=state_root,
        generated_at=None,
        task_mode=str(task.get("mode") or "implementation"),
        topic=task.get("topic"),
        operator_instruction_ref=str(task.get("operator_instruction_ref") or "scheduler Assignment"),
        edit_authorized=bool(task.get("edit_authorized", True)),
        campaign_authorized=bool(task.get("campaign_authorized", False)),
        gpu_required_mode="yes" if task.get("gpu_required") is True else "no" if task.get("gpu_required") is False else "auto",
        python_value=None,
        allow_current_python=True,
        acknowledge_dirty=False,
        write_scope=scopes,
        external_evidence_roots=external_evidence_roots,
        expected_validation=[str(row) for row in task.get("expected_validation") or []],
        preflight_receipt_path=str(preflight.get("receipt_path") or authority.get("preflight_receipt_path") or "") or None,
        preflight_exemption_reason=None,
        authority_domain=task.get("authority_domain"),
        collection=str(preflight.get("collection") or "all"),
        limit=20,
        store_receipt=True,
    )
    if record is None or refreshed.get("readiness") not in {"READY", "READY_WITH_WARNINGS"}:
        raise ActorError("worker could not refresh Bootstrap after commit", code="worker_bootstrap_refresh_failed")
    return str(record.get("record_id"))


def _write_review_proof(
    *,
    assignment: Mapping[str, Any],
    session_id: str,
    state_root: Path,
    result_path: Path,
    codex_id: str,
) -> Path:
    """Create the fake worker's explicit, content-addressed review input."""

    material = dict(load_assignment_payload(assignment.get("payload"), require_ready=True).get("material") or {})
    package = _assignment_work_package(
        Path(str(dict(material.get("repository") or {}).get("repo_root"))), assignment
    )
    review_session = session_store(state_root).get(session_id)
    review_target = dict(dict(review_session.get("payload") or {}).get("review_target") or {})
    delivery_session_id = str(review_target.get("delivery_session_id") or "")
    delivery = session_store(state_root).get(delivery_session_id)
    delivery_evidence = dict(dict(delivery.get("payload") or {}).get("delivery_evidence") or {})
    reviewed_commit = str(review_target.get("reviewed_commit") or delivery_evidence.get("delivery_commit") or "")
    package_sha = str(review_target.get("reviewed_package_sha256") or dict(dict(material.get("work") or {}).get("work_package_binding") or {}).get("sha256") or "")
    proof_dir = result_path.parent
    proof_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = proof_dir / f"{session_id}-review-evidence.txt"
    evidence_path.write_text("fake worker review evidence\n", encoding="utf-8")
    evidence_sha = file_sha256(evidence_path)
    evidence_ref = {"uri": str(evidence_path), "sha256": evidence_sha}
    execution = {
        "schema_version": "eiel_review_execution_receipt_v1",
        "execution_mode": "REVIEWER_EXECUTED",
        "reviewed_commit": reviewed_commit,
        "command_or_source": "fake scheduler review",
        "status": "PASS",
        "output_sha256": evidence_sha,
        "executor_id": codex_id,
        "executed_at_utc": utc_now(),
        "environment": {"worker": "fake"},
        "review_session_id": session_id,
        "evidence_relation": "EXECUTED_IN_REVIEW",
        "source_execution_id": None,
    }
    results: list[dict[str, Any]] = []
    for obligation in package.get("acceptance_criteria") or []:
        results.append({"kind": "acceptance_criterion", "obligation": str(obligation), "status": "PASS", "evidence_refs": [evidence_ref], "execution_receipt": None})
    for kind, field in (("test", "tests"), ("evidence_item", "evidence")):
        for obligation in package.get(field) or []:
            results.append({"kind": kind, "obligation": str(obligation), "status": "PASS", "evidence_refs": [evidence_ref], "execution_receipt": dict(execution)})
    proof = {
        "schema_version": "eiel_review_obligation_proof_v2",
        "work_id": str(dict(material.get("work") or {}).get("record_id") or ""),
        "reviewed_commit": reviewed_commit,
        "reviewed_package_sha256": package_sha,
        "implementation_sufficiency": "SUFFICIENT",
        "obligation_results": results,
        "limitations": [],
    }
    proof_path = proof_dir / f"{session_id}-review-proof.json"
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return proof_path


def run_fake_worker(
    *,
    assignment_id: str,
    lease_id: str,
    state_root: Path,
    result_path: Path,
    worker_id: str,
    behavior: str,
    fixture_content: str,
    sleep_seconds: float = 0.0,
) -> int:
    assignment = assignment_store(state_root).get(assignment_id)
    material = dict(load_assignment_payload(assignment.get("payload"), require_ready=True).get("material") or {})
    repo_root = Path(str(dict(material.get("repository") or {}).get("repo_root"))).resolve()
    session_id = str(dict(material.get("session") or {}).get("record_id") or "")
    codex_id = f"FAKE-THREAD-{assignment_id}"
    started = utc_now()
    # Scheduler mechanically activates before launch. This call is idempotent
    # and acts only as a defensive worker-side integrity check.
    open_worker(assignment_id, lease_id, state_root=state_root, worker_id=worker_id, codex_id=codex_id)
    if sleep_seconds:
        time.sleep(float(sleep_seconds))
    phase = str(dict(material.get("phase") or {}).get("kind") or "implementation")
    validation_commands: list[str] = []
    report_paths: list[str] = []
    blockers: list[str] = []
    validation_status = "NOT_RUN"
    validation_gaps: list[str] = []
    validation_decision_impact: str | None = None

    if behavior == "crash":
        return 9
    if behavior == "no-result":
        _manager(state_root, repo_root, "sessionManager", "checkpoint", session_id, "--next-action", "Resume after fake worker loss.")
        return 0
    if behavior == "promotion":
        _manager(
            state_root, repo_root, "sessionManager", "request-promotion", session_id,
            "--requested-class", "U4", "--reason", "Fake worker found architecture uncertainty.",
            "--affected", "shared architecture contract", "--safe-state", "Repository remains committed and bounded.",
            "--recommended-authority", "ENGINEERING_AUTHORITY",
        )
        outcome = "PROMOTION_REQUIRED"
    elif behavior == "blocked":
        _manager(state_root, repo_root, "sessionManager", "block", session_id, "--reason", "Fake worker bounded blocker.", "--next-action", "Operator decision required.")
        outcome = "BLOCKED"
        blockers = ["Fake worker bounded blocker."]
        validation_status = "BLOCKED"
    elif behavior in {"review-accept", "review-changes", "review-reject"} or phase == "review":
        verdict = {"review-changes": "CHANGES_REQUIRED", "review-reject": "REJECT"}.get(behavior, "ACCEPT")
        package = _assignment_work_package(repo_root, assignment)
        args = [
            "sessionManager", "complete", session_id,
            "--outcome", "REVIEW_COMPLETE", "--review-verdict", verdict,
        ]
        for criterion in package.get("acceptance_criteria") or []:
            args.extend(["--acceptance-reviewed", str(criterion)])
        for test in package.get("tests") or []:
            args.extend(["--tests-reviewed", str(test)])
        for evidence in package.get("evidence") or []:
            args.extend(["--evidence-reviewed", str(evidence)])
        if verdict == "ACCEPT":
            args.extend(["--review-proof", str(_write_review_proof(
                assignment=assignment,
                session_id=session_id,
                state_root=state_root,
                result_path=result_path,
                codex_id=codex_id,
            ))])
        if verdict != "ACCEPT":
            args.extend(["--review-finding", "Fake review requested a bounded repair.", "--repair-disposition", "WITHIN_PACKAGE"])
        _manager(state_root, repo_root, *args)
        outcome = "COMPLETED"
        validation_status = "PASSED" if verdict == "ACCEPT" else "PARTIAL"
        if validation_status == "PARTIAL":
            validation_gaps = ["Review found a bounded acceptance gap requiring repair."]
            validation_decision_impact = "The delivery is not accepted; a bound repair Session and rereview are required."
    elif behavior in {"implementation-complete", "repair-complete"}:
        changed, delivery_commit = _commit_fixture(repo_root, assignment, fixture_content)
        validation_commands = [f"git show --stat --oneline {delivery_commit} ({changed})"]
        validation_status = "PASSED"
        fresh_bootstrap_id = _refresh_bootstrap_after_commit(state_root, repo_root, session_id)
        attach_session_bootstrap_after_commit(
            session_id, state_root=state_root, bootstrap_record_id=fresh_bootstrap_id, repo_root=repo_root
        )
        _manager(
            state_root,
            repo_root,
            "sessionManager",
            "complete",
            session_id,
            "--outcome",
            "IMPLEMENTATION_COMPLETE",
            "--delivery-commit",
            delivery_commit,
            "--final-step",
            "Fake worker bounded implementation complete",
        )
        outcome = "COMPLETED"
    else:
        _manager(
            state_root, repo_root, "sessionManager", "checkpoint", session_id,
            "--current-step", "Fake worker checkpoint complete",
            "--completed-step", "Assignment opened and verified",
            "--next-action", "Continue the bounded Work Package in a fresh Assignment.",
        )
        _manager(
            state_root, repo_root, "sessionManager", "pause", session_id,
            "--next-action", "Continue the bounded Work Package in a fresh Assignment.",
        )
        outcome = "CHECKPOINTED"

    result = result_from_repository(
        assignment_id=assignment_id,
        lease_id=lease_id,
        state_root=state_root,
        worker_identity={"kind": "fake", "worker_id": worker_id, "pid": __import__("os").getpid(), "thread_id": codex_id},
        outcome=outcome,
        validation_status=validation_status,
        validation_commands=validation_commands,
        validation_gaps=validation_gaps,
        validation_decision_impact=validation_decision_impact,
        report_paths=report_paths,
        blockers=blockers,
        started_at_utc=started,
        ended_at_utc=utc_now(),
        exit_code=0,
    )
    write_worker_result(result_path, result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.scheduler.worker")
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--lease-id", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--behavior", default="checkpoint")
    parser.add_argument("--fixture-content", default="scheduler fake worker fixture\n")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_fake_worker(
            assignment_id=args.assignment_id,
            lease_id=args.lease_id,
            state_root=Path(args.state_root).resolve(),
            result_path=Path(args.result_path).resolve(),
            worker_id=args.worker_id,
            behavior=args.behavior,
            fixture_content=args.fixture_content,
            sleep_seconds=args.sleep_seconds,
        )
    except (ActorError, subprocess.CalledProcessError) as exc:
        payload = {"ok": False, "error": str(exc)}
        if isinstance(exc, ActorError):
            payload.update({"code": exc.code, "problems": exc.problems})
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
