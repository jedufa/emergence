"""Command-line surfaces for the optional EIEL Assignment Linker and Scheduler."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.actors.common import (
    ActorArgumentParser,
    ActorError,
    emit_actor_error,
    emit_payload,
    read_json,
    read_structured,
    with_guidance,
)
from tools.scheduler.contracts import (
    assignment_store,
    dispatch_store,
    lease_store,
    records,
    result_store,
)
from tools.scheduler.dispatch import (
    authorize_dispatch,
    claim_assignment,
    dispatch_candidates,
    expire_leases,
    heartbeat_lease,
    recover_preparing,
    request_assignment,
    set_dispatch_enabled,
    submit_worker_result,
    worker_open,
)
from tools.scheduler.launcher import probe_codex_cli
from tools.scheduler.programme_driver import (
    ProgrammeJournalV1,
    claim_local_driver_lease_v1,
    deliver_terminal_notifications_v1,
    executable_action_adapter_v1,
    executable_notification_sender_v1,
    programme_status_v1,
    reconcile_driver_once_v1,
)
from tools.scheduler.runtime import (
    reconcile,
    run_once,
    scheduler_status,
    set_runtime_status,
    start_polling,
    worker_rows,
)
from tools.scheduler.worker_protocol import result_from_repository, write_worker_result

ASSIGNMENT_ACTOR = "assignmentLinker"
SCHEDULER_ACTOR = "scheduler"


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root")
    parser.add_argument("--generated-at")
    parser.add_argument("--output")
    parser.add_argument("--human", action="store_true")


def _emit(payload: Mapping[str, Any], args: argparse.Namespace, *, human: str | None = None) -> int:
    emit_payload(
        payload,
        output=getattr(args, "output", None),
        human_text=human or _human(payload),
        json_output=not getattr(args, "human", False),
    )
    return 0


def _human(payload: Mapping[str, Any]) -> str:
    """Render concise Assignment/Scheduler summaries without hiding guidance."""

    lines = [f"Status: {payload.get('status') or 'UNKNOWN'}"]
    subject_name = None
    subject: Mapping[str, Any] | None = None
    for key, label in (
        ("dispatch_request", "Dispatch Request"),
        ("assignment", "Assignment"),
        ("lease", "Lease"),
        ("worker_result", "Worker Result"),
        ("scheduler", "Scheduler"),
        ("runtime", "Runtime"),
    ):
        value = payload.get(key)
        if isinstance(value, Mapping):
            subject_name = label
            subject = value
            break
    if subject is not None:
        subject_payload = dict(subject.get("payload") or {})
        runtime = dict(subject.get("runtime") or {}) if subject_name == "Scheduler" else {}
        identifier = (
            subject.get("record_id")
            or subject.get("scheduler_id")
            or runtime.get("scheduler_id")
            or subject.get("assignment_id")
            or subject.get("lease_id")
            or subject.get("result_id")
        )
        if identifier:
            lines.append(f"{subject_name}: {identifier}")
        subject_status = subject.get("status") or subject_payload.get("status")
        if subject_status and str(subject_status) != str(payload.get("status")):
            lines.append(f"{subject_name} status: {subject_status}")
        for key, label in (
            ("work_id", "Work"),
            ("phase", "Phase"),
            ("assignment_id", "Assignment"),
            ("lease_id", "Lease"),
            ("session_id", "Session"),
            ("worker_id", "Worker"),
        ):
            value = subject.get(key) or subject_payload.get(key)
            if value:
                lines.append(f"{label}: {value}")
        counts = subject.get("counts")
        if not isinstance(counts, Mapping) and subject_name == "Scheduler":
            counts = subject.get("records")
        if isinstance(counts, Mapping):
            rendered = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
            if rendered:
                lines.append(f"Counts: {rendered}")
        runtime_counters = runtime.get("counters") if runtime else None
        if isinstance(runtime_counters, Mapping):
            rendered = ", ".join(
                f"{key}={runtime_counters[key]}" for key in sorted(runtime_counters)
            )
            if rendered:
                lines.append(f"Runtime counters: {rendered}")
    for key, label in (
        ("count", "Count"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("recovered", "Recovered"),
    ):
        if key in payload and not isinstance(payload.get(key), (Mapping, list)):
            lines.append(f"{label}: {payload.get(key)}")
    warnings = list(payload.get("warnings") or [])
    if warnings:
        lines.append(f"Warnings: {len(warnings)}")
        lines.extend(f"Warning: {warning}" for warning in warnings[:3])
    actions = list(payload.get("next_actions") or [])
    if actions:
        action = dict(actions[0])
        lines.append(f"Next: {action.get('description') or 'inspect the structured result'}")
        if action.get("command"):
            lines.append(f"Command: {action['command']}")
    elif str(payload.get("status") or "").upper() in {
        "RESULT_SUBMITTED",
        "RECONCILED",
        "SCHEDULER_STOPPED",
        "STOPPED",
        "PAUSED",
        "STOP_REQUESTED",
    }:
        lines.append("Next: none")
    return "\n".join(lines) + "\n"


def _record_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record.get("record_id"),
        "record_type": record.get("record_type"),
        "status": record.get("status"),
        "revision": record.get("revision"),
        "updated_at_utc": record.get("updated_at_utc"),
        "record_sha256": record.get("record_sha256"),
    }


def build_assignment_parser() -> argparse.ArgumentParser:
    parser = ActorArgumentParser(
        prog="python -m tools.eiel assignment",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Command groups:
  Operator: authorize, enable, disable, candidates, request, inspect, list, list-dispatch
  Scheduler-internal/advanced: claim, heartbeat, submit, recover-preparing, expire-leases, worker-open, worker-result, reconcile-result""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    authorize = sub.add_parser("authorize", help="Create an explicit human Dispatch Request.")
    _common(authorize)
    authorize.add_argument("--request", help="JSON or YAML Dispatch Request; flags may provide the minimal request instead.")
    authorize.add_argument("--dispatch-id")
    authorize.add_argument("--work-id")
    authorize.add_argument("--repo-root")
    authorize.add_argument("--expected-branch")
    authorize.add_argument("--phase", action="append", dest="phases")
    authorize.add_argument("--launcher", choices=("fake", "codex_exec"), default="fake")
    authorize.add_argument("--model")
    authorize.add_argument("--reasoning-level")
    authorize.add_argument("--profile")
    authorize.add_argument("--sandbox", default="workspace-write")
    authorize.add_argument("--fake-behavior", default="checkpoint")
    authorize.add_argument("--fake-fixture-content", default="scheduler fake worker fixture\n")
    authorize.add_argument("--queue-priority", type=int, default=100)
    authorize.add_argument("--lease-duration-seconds", type=int, default=3600)
    authorize.add_argument("--heartbeat-interval-seconds", type=int, default=300)
    authorize.add_argument("--created-by", default="HUMAN_OPERATOR")
    authorize.add_argument("--note")
    authorize.add_argument("--disabled", action="store_true")
    authorize.add_argument("--no-fresh-session", action="store_true")
    authorize.add_argument("--no-resume", action="store_true")
    authorize.add_argument("--no-independent-review", action="store_true")
    authorize.add_argument("--no-repair", action="store_true")

    enable = sub.add_parser("enable", help="Enable an existing Dispatch Request.")
    _common(enable)
    enable.add_argument("dispatch_id")
    enable.add_argument("--note")

    disable = sub.add_parser("disable", help="Disable an existing Dispatch Request.")
    _common(disable)
    disable.add_argument("dispatch_id")
    disable.add_argument("--note")

    candidates = sub.add_parser("candidates", help="List explicitly authorized deterministic candidates.")
    _common(candidates)
    candidates.add_argument("--eligible-only", action="store_true")

    request = sub.add_parser("request", help="Compile one immutable Assignment.")
    _common(request)
    request.add_argument("--dispatch-id")
    request.add_argument("--work-id")

    inspect = sub.add_parser("inspect", help="Inspect an Assignment record.")
    _common(inspect)
    inspect.add_argument("assignment_id")

    listing = sub.add_parser("list", help="List Assignment records.")
    _common(listing)
    listing.add_argument("--status")

    dispatches = sub.add_parser("list-dispatch", help="List Dispatch Requests.")
    _common(dispatches)
    dispatches.add_argument("--status")

    claim = sub.add_parser("claim", help="Claim a READY Assignment with a temporary Lease.")
    _common(claim)
    claim.add_argument("assignment_id")
    claim.add_argument("--scheduler-id", required=True)

    heartbeat = sub.add_parser("heartbeat", help="Refresh one active Lease.")
    _common(heartbeat)
    heartbeat.add_argument("lease_id")
    heartbeat.add_argument("--worker-id")
    heartbeat.add_argument("--pid", type=int)

    submit = sub.add_parser("submit", help="Validate and store a structured Worker Result.")
    _common(submit)
    submit.add_argument("--result", required=True)

    recover = sub.add_parser("recover-preparing", help="Recover crash-interrupted PREPARING Assignments.")
    _common(recover)

    expire = sub.add_parser("expire-leases", help="Expire elapsed Leases and record worker loss.")
    _common(expire)

    open_worker_parser = sub.add_parser("worker-open", help="Verify Assignment/Lease and activate the exact prepared Session.")
    _common(open_worker_parser)
    open_worker_parser.add_argument("assignment_id")
    open_worker_parser.add_argument("lease_id")
    open_worker_parser.add_argument("--worker-id", required=True)
    open_worker_parser.add_argument("--codex-id", required=True)

    worker_result = sub.add_parser("worker-result", help="Generate a digest-bound Worker Result from live repository state.")
    _common(worker_result)
    worker_result.add_argument("assignment_id")
    worker_result.add_argument("lease_id")
    worker_result.add_argument("--result-path", required=True)
    worker_result.add_argument("--worker-id", required=True)
    worker_result.add_argument("--worker-kind", default="manual")
    worker_result.add_argument("--outcome", required=True)
    worker_result.add_argument("--validation-status", default="NOT_RUN")
    worker_result.add_argument("--validation-command", action="append", default=[])
    worker_result.add_argument("--report-path", action="append", default=[])
    worker_result.add_argument("--blocker", action="append", default=[])
    worker_result.add_argument("--exit-code", type=int, default=0)

    capability = sub.add_parser("codex-capability", help="Probe the optional non-interactive Codex CLI surface.")
    _common(capability)
    capability.add_argument("--executable", default="codex")

    return parser


def _authorize_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.request:
        raw = read_structured(args.request, name="Dispatch Request")
    else:
        if not args.work_id or not args.repo_root:
            raise ActorError("--work-id and --repo-root are required without --request", code="dispatch_request_incomplete")
        raw = {
            "work_id": args.work_id,
            "enabled": not args.disabled,
            "allowed_authority_phases": args.phases or ["implementation"],
            "worker_profile": {
                "launcher_type": args.launcher,
                "model": args.model,
                "reasoning_level": args.reasoning_level,
                "profile": args.profile,
                "sandbox": args.sandbox,
                "fake_behavior": args.fake_behavior,
                "fake_fixture_content": args.fake_fixture_content,
                "ephemeral": True,
            },
            "repository": {"repo_root": args.repo_root, "expected_branch": args.expected_branch},
            "queue": {"priority": args.queue_priority},
            "automation_policy": {
                "allow_fresh_session": not args.no_fresh_session,
                "allow_resume": not args.no_resume,
                "allow_independent_review": not args.no_independent_review,
                "allow_repair": not args.no_repair,
            },
            "lease_policy": {
                "duration_seconds": args.lease_duration_seconds,
                "heartbeat_interval_seconds": args.heartbeat_interval_seconds,
            },
            "created_by": args.created_by,
            "note": args.note,
        }
    return raw


def assignment_main(argv: Sequence[str] | None = None) -> int:
    parser = build_assignment_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args: argparse.Namespace | None = None
    try:
        args = parser.parse_args(raw)
        if getattr(args, "generated_at", None) and os.environ.get("EIEL_SCHEDULER_TEST_MODE") != "1":
            raise ActorError("--generated-at is available only in explicit Scheduler test mode", code="scheduler_test_time_forbidden")
        command = args.command
        if command == "authorize":
            record = authorize_dispatch(_authorize_payload(args), state_root=args.state_root, generated_at=args.generated_at, dispatch_id=args.dispatch_id)
            return _emit(with_guidance({"ok": True, "dispatch_request": record}, status="DISPATCH_AUTHORIZED", related_records=[str(record.get("record_id"))]), args)
        if command in {"enable", "disable"}:
            record = set_dispatch_enabled(args.dispatch_id, enabled=command == "enable", state_root=args.state_root, generated_at=args.generated_at, note=args.note)
            return _emit(with_guidance({"ok": True, "dispatch_request": record}, status=str(record.get("status"))), args)
        if command == "candidates":
            rows = dispatch_candidates(state_root=args.state_root)
            if args.eligible_only:
                rows = [row for row in rows if row.get("eligible")]
            status = "ASSIGNMENT_CANDIDATES" if rows else "NO_AUTHORIZED_CANDIDATES"
            return _emit(with_guidance({"ok": True, "candidates": rows, "count": len(rows)}, status=status), args)
        if command == "request":
            record = request_assignment(state_root=args.state_root, dispatch_id=args.dispatch_id, work_id=args.work_id, generated_at=args.generated_at)
            status = "ASSIGNMENT_READY" if record else "NO_ASSIGNMENT"
            next_actions = [] if record else [{"description": "Authorize or unblock one Work item before requesting an Assignment.", "command": "python -m tools.eiel assignment candidates --eligible-only"}]
            return _emit(with_guidance({"ok": True, "assignment": record}, status=status, next_actions=next_actions), args)
        if command == "inspect":
            record = assignment_store(args.state_root).get(args.assignment_id)
            return _emit(with_guidance({"ok": True, "assignment": record}, status=str(record.get("status"))), args)
        if command == "list":
            rows = records(assignment_store(args.state_root))
            if args.status:
                rows = [row for row in rows if str(row.get("status")) == str(args.status).upper()]
            return _emit(with_guidance({"ok": True, "assignments": [_record_summary(row) for row in rows], "count": len(rows)}, status="ASSIGNMENT_LIST"), args)
        if command == "list-dispatch":
            rows = records(dispatch_store(args.state_root))
            if args.status:
                rows = [row for row in rows if str(row.get("status")) == str(args.status).upper()]
            return _emit(with_guidance({"ok": True, "dispatch_requests": [_record_summary(row) for row in rows], "count": len(rows)}, status="DISPATCH_LIST"), args)
        if command == "claim":
            assignment, lease = claim_assignment(args.assignment_id, scheduler_id=args.scheduler_id, state_root=args.state_root, generated_at=args.generated_at)
            return _emit(with_guidance({"ok": True, "assignment": assignment, "lease": lease}, status="ASSIGNMENT_CLAIMED"), args)
        if command == "heartbeat":
            worker = {"worker_id": args.worker_id, "pid": args.pid} if args.worker_id or args.pid is not None else None
            lease = heartbeat_lease(args.lease_id, state_root=args.state_root, worker=worker, generated_at=args.generated_at)
            return _emit(with_guidance({"ok": True, "lease": lease}, status="LEASE_ACTIVE"), args)
        if command == "submit":
            result = read_json(args.result, name="Worker Result")
            result_record, assignment, lease = submit_worker_result(result, state_root=args.state_root, generated_at=args.generated_at)
            return _emit(with_guidance({"ok": True, "worker_result": result_record, "assignment": assignment, "lease": lease}, status="RESULT_SUBMITTED"), args)
        if command == "recover-preparing":
            rows = recover_preparing(state_root=args.state_root, generated_at=args.generated_at)
            return _emit(with_guidance({"ok": True, "assignments": [_record_summary(row) for row in rows], "count": len(rows)}, status="PREPARING_RECOVERED"), args)
        if command == "expire-leases":
            rows = expire_leases(state_root=args.state_root, generated_at=args.generated_at)
            return _emit(with_guidance({"ok": True, "leases": [_record_summary(row) for row in rows], "count": len(rows)}, status="LEASES_EXPIRED"), args)
        if command == "worker-open":
            session = worker_open(args.assignment_id, args.lease_id, state_root=args.state_root, worker_id=args.worker_id, codex_id=args.codex_id, generated_at=args.generated_at)
            return _emit(with_guidance({"ok": True, "session": session}, status="WORKER_SESSION_ACTIVE"), args)
        if command == "worker-result":
            result = result_from_repository(
                assignment_id=args.assignment_id,
                lease_id=args.lease_id,
                state_root=args.state_root,
                worker_identity={"kind": args.worker_kind, "worker_id": args.worker_id},
                outcome=args.outcome,
                validation_status=args.validation_status,
                validation_commands=args.validation_command,
                report_paths=args.report_path,
                blockers=args.blocker,
                exit_code=args.exit_code,
                ended_at_utc=args.generated_at,
            )
            path = write_worker_result(args.result_path, result)
            return _emit(with_guidance({"ok": True, "worker_result": result, "path": str(path)}, status="WORKER_RESULT_WRITTEN"), args)
        if command == "codex-capability":
            capability = probe_codex_cli(args.executable)
            status = str(capability.get("status"))
            return _emit(with_guidance({"ok": True, "codex_cli": capability}, status=status, warnings=[] if capability.get("ok") else ["The optional Scheduler remains usable with Fake Worker qualification and manual EIEL workflows remain unaffected."]), args)
        raise ActorError("unsupported assignment command", code="cli_usage_error")
    except ActorError as exc:
        command = raw[0] if raw else "help"
        emit_actor_error(
            exc,
            actor=ASSIGNMENT_ACTOR,
            command=command,
            raw_args=raw,
            parsed_args=args,
        )
        return 2


def build_scheduler_parser() -> argparse.ArgumentParser:
    parser = ActorArgumentParser(
        prog="python -m tools.eiel scheduler",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Command groups:
  Operator: run-once, start, status, workers, pause, resume, stop
  Recovery/administration: reconcile, programme-status, programme-reconcile,
                           programme-watch

The Scheduler is optional, explicit-only, foreground, and single-worker.""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-once", help="Execute at most one explicitly authorized Assignment.")
    _common(run)
    run.add_argument("--scheduler-id")
    run.add_argument("--dispatch-id")
    run.add_argument("--work-id")

    start = sub.add_parser("start", help="Run the optional single-worker polling scheduler in the foreground.")
    _common(start)
    start.add_argument("--scheduler-id")
    start.add_argument("--poll-seconds", type=float, default=300.0)
    start.add_argument("--max-assignments", type=int)
    start.add_argument("--max-idle-cycles", type=int)

    status = sub.add_parser("status", help="Show disposable runtime state and durable record counts.")
    _common(status)

    workers = sub.add_parser("workers", help="Show locally observed worker processes.")
    _common(workers)

    reconcile_parser = sub.add_parser("reconcile", help="Reconcile process loss or completed result files after restart.")
    _common(reconcile_parser)

    programme_status = sub.add_parser(
        "programme-status",
        help="Project one authorized programme journal and its exact pending intents.",
    )
    _common(programme_status)
    programme_status.add_argument("--programme-root", required=True)

    programme_reconcile = sub.add_parser(
        "programme-reconcile",
        help="Validate and reconcile one programme journal without inventing effects.",
    )
    _common(programme_reconcile)
    programme_reconcile.add_argument("--programme-root", required=True)

    programme_watch = sub.add_parser(
        "programme-watch",
        help="Run the sparse owner-event/heartbeat programme waiter.",
    )
    _common(programme_watch)
    programme_watch.add_argument("--programme-root", required=True)
    programme_watch.add_argument("--poll-seconds", type=float, default=300.0)
    programme_watch.add_argument("--max-cycles", type=int)
    programme_watch.add_argument("--action-adapter-executable", required=True)
    programme_watch.add_argument(
        "--action-claim-ttl-seconds", type=int, default=900
    )

    programme_notify = sub.add_parser(
        "programme-notify",
        help="Consume terminal intents through one explicit Project Manager adapter.",
    )
    _common(programme_notify)
    programme_notify.add_argument("--programme-root", required=True)
    programme_notify.add_argument("--adapter-executable", required=True)
    programme_notify.add_argument("--provider", default="project-manager")

    for name in ("pause", "resume", "stop"):
        command = sub.add_parser(name, help=f"{name.title()} the optional foreground polling loop.")
        _common(command)

    return parser


def scheduler_main(argv: Sequence[str] | None = None) -> int:
    parser = build_scheduler_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args: argparse.Namespace | None = None
    try:
        args = parser.parse_args(raw)
        if getattr(args, "generated_at", None) and os.environ.get("EIEL_SCHEDULER_TEST_MODE") != "1":
            raise ActorError("--generated-at is available only in explicit Scheduler test mode", code="scheduler_test_time_forbidden")
        if args.command == "run-once":
            event = run_once(state_root=args.state_root, scheduler_id=args.scheduler_id, dispatch_id=args.dispatch_id, work_id=args.work_id)
            return _emit(with_guidance({"ok": True, "scheduler": event}, status=str(event.get("status"))), args)
        if args.command == "start":
            event = start_polling(
                state_root=args.state_root,
                scheduler_id=args.scheduler_id,
                poll_interval_seconds=args.poll_seconds,
                max_assignments=args.max_assignments,
                max_idle_cycles=args.max_idle_cycles,
            )
            return _emit(with_guidance({"ok": True, "scheduler": event}, status="SCHEDULER_STOPPED"), args)
        if args.command == "status":
            value = scheduler_status(state_root=args.state_root)
            return _emit(with_guidance({"ok": True, "scheduler": value}, status=str(value.get("status"))), args)
        if args.command == "workers":
            rows = worker_rows(state_root=args.state_root)
            return _emit(with_guidance({"ok": True, "workers": rows, "count": len(rows)}, status="WORKER_STATUS"), args)
        if args.command == "reconcile":
            value = reconcile(state_root=args.state_root)
            return _emit(with_guidance({"ok": True, "scheduler": value}, status="RECONCILED"), args)
        if args.command == "programme-status":
            value = programme_status_v1(ProgrammeJournalV1(args.programme_root))
            return _emit(
                with_guidance(
                    {"ok": True, "programme": value},
                    status=str(value.get("programme_state") or "UNKNOWN"),
                ),
                args,
            )
        if args.command in {"programme-reconcile", "programme-watch"}:
            journal = ProgrammeJournalV1(args.programme_root)
            driver_lease = None
            if args.command == "programme-watch":
                driver_lease = claim_local_driver_lease_v1(
                    journal,
                    ttl_seconds=max(30, int(float(args.poll_seconds) * 3)),
                )
                observe, execute = executable_action_adapter_v1(
                    args.action_adapter_executable
                )
                value = reconcile_driver_once_v1(
                    journal,
                    driver_lease=driver_lease,
                    observe=observe,
                    execute=execute,
                    action_claim_ttl_seconds=max(
                        1, int(args.action_claim_ttl_seconds)
                    ),
                )
            else:
                value = reconcile_driver_once_v1(journal)
            cycles = 0
            while (
                args.command == "programme-watch"
                and value.get("status") != "PROGRAMME_TERMINAL"
            ):
                cycles += 1
                if args.max_cycles is not None and cycles >= args.max_cycles:
                    break
                if value.get("status") != "PROGRESSED":
                    time.sleep(max(5.0, float(args.poll_seconds)))
                driver_lease = journal.heartbeat_driver_lease(
                    driver_lease,
                    at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    ttl_seconds=max(30, int(float(args.poll_seconds) * 3)),
                )
                value = reconcile_driver_once_v1(
                    journal,
                    driver_lease=driver_lease,
                    observe=observe,
                    execute=execute,
                    action_claim_ttl_seconds=max(
                        1, int(args.action_claim_ttl_seconds)
                    ),
                )
            return _emit(
                with_guidance(
                    {"ok": True, "programme": value},
                    status=str(value.get("status") or "RECONCILED"),
                ),
                args,
            )
        if args.command == "programme-notify":
            journal = ProgrammeJournalV1(args.programme_root)
            outcomes = deliver_terminal_notifications_v1(
                journal,
                send=executable_notification_sender_v1(args.adapter_executable),
                at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                provider=args.provider,
            )
            return _emit(
                with_guidance(
                    {"ok": True, "deliveries": outcomes},
                    status="NOTIFICATIONS_RECONCILED",
                ),
                args,
            )
        if args.command in {"pause", "resume", "stop"}:
            target = {"pause": "PAUSED", "resume": "RUNNING", "stop": "STOP_REQUESTED"}[args.command]
            value = set_runtime_status(target, state_root=args.state_root)
            return _emit(with_guidance({"ok": True, "runtime": value}, status=target), args)
        raise ActorError("unsupported scheduler command", code="cli_usage_error")
    except ActorError as exc:
        command = raw[0] if raw else "help"
        emit_actor_error(
            exc,
            actor=SCHEDULER_ACTOR,
            command=command,
            raw_args=raw,
            parsed_args=args,
        )
        return 2


__all__ = ["assignment_main", "scheduler_main"]
