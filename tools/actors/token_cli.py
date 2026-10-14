"""WRA-1 token capture and Work Item resource-accounting export CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.actors.common import (
    ActorArgumentParser,
    ActorError,
    emit_actor_error,
    emit_payload,
    with_guidance,
)
from tools.actors.portfolio import (
    SESSION_ID_PREFIX,
    SESSION_TERMINAL_STATUSES,
    WORK_ID_PREFIX,
    WORK_TERMINAL_STATUSES,
    session_store,
    sessions_for_work,
    validate_prefixed_id,
    work_store,
)
from tools.actors.resource_accounting import (
    aggregate_work_usage,
    capture_start,
    export_token_report,
    finalize_session,
    resource_accounting_coverage,
    retain_estimator_approval_receipt,
    retain_estimator_review_receipt,
    require_resource_thread_available,
    resource_thread_lock,
    work_accounting_projection,
)
from tools.actors.session_manager import attach_session_resource_accounting
from tools.actors.work_manager import attach_work_resource_accounting

TOOL_NAME = "token"


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root")
    parser.add_argument("--generated-at")
    parser.add_argument("--output")
    parser.add_argument("--human", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = ActorArgumentParser(
        prog="python -m tools.eiel token",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""WRA-1 delegates Codex telemetry parsing to ccusage. EIEL stores only
normalized, immutable evidence and Work/Session bindings. No cost or credit
fields are accepted or emitted.""",
    )
    sub = parser.add_subparsers(dest="command", required=True, parser_class=ActorArgumentParser)

    start = sub.add_parser("capture-start", help="Capture a cumulative ccusage START snapshot for one Session epoch/thread.")
    _common(start)
    start.add_argument("session_id")
    start.add_argument("--ccusage-command", help="Override the ccusage executable/command for this invocation.")
    start.add_argument("--correction-note")

    end = sub.add_parser("capture-end", help="Capture END snapshots and compute immutable per-Session token deltas.")
    _common(end)
    end.add_argument("session_id")
    end.add_argument("--ccusage-command", help="Override the ccusage executable/command for this invocation.")
    end.add_argument("--correction-note")
    end.add_argument("--capture-id", help="Exact active capture corrected or closed; required with --correction-note.")

    finalize = sub.add_parser("finalize-work", help="Create or correct one immutable Work token_usage.json artifact.")
    _common(finalize)
    finalize.add_argument("work_id")
    finalize.add_argument("--correction-note")

    report = sub.add_parser("report", help="Export attached completed Work token_usage.json artifacts.")
    _common(report)
    report.add_argument("--output-dir", required=True)

    coverage = sub.add_parser("coverage", help="Report non-blocking WRA coverage and unmanaged intervals.")
    _common(coverage)

    estimator_review = sub.add_parser("estimator-review", help="Retain an immutable offline estimator review receipt.")
    _common(estimator_review)
    estimator_review.add_argument("--report", required=True)
    estimator_review.add_argument("--rules", required=True)
    estimator_review.add_argument("--artifact-ref", action="append", default=[])

    estimator_approve = sub.add_parser("estimator-approve", help="Record explicit human approval or rejection of estimator rules.")
    _common(estimator_approve)
    estimator_approve.add_argument("--review-ref", required=True)
    estimator_approve.add_argument("--rules-after", required=True)
    estimator_approve.add_argument("--authority", required=True)
    estimator_approve.add_argument("--decision", choices=("APPROVED", "REJECTED"), required=True)
    return parser


def _session_update(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    accounting: Mapping[str, Any],
    action: str,
    generated_at: str | None,
    note: str | None,
) -> dict[str, Any]:
    return attach_session_resource_accounting(
        str(record.get("record_id") or ""),
        accounting,
        state_root=state_root,
        generated_at=generated_at,
        action=action,
        note=note,
    )


def command_capture_start(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    session_id = validate_prefixed_id(args.session_id, SESSION_ID_PREFIX, label="session")
    initial = session_store(args.state_root).get_read_only(session_id)
    work_id = str(dict(initial.get("payload") or {}).get("work_id") or "")
    initial_payload = dict(initial.get("payload") or {})
    thread_id = str(initial_payload.get("current_codex_id") or "").strip()
    with resource_thread_lock(args.state_root, thread_id):
        require_resource_thread_available(
            args.state_root,
            thread_id,
            exclude_session_id=session_id,
        )
        record = session_store(args.state_root).get(session_id)
        if str(record.get("status") or "") in SESSION_TERMINAL_STATUSES:
            raise ActorError(
                "cannot start a new token capture on a terminal Session",
                code="token_capture_terminal_session",
                related_records=[session_id, work_id],
            )
        accounting, capture = capture_start(
            record,
            state_root=args.state_root,
            generated_at=args.generated_at,
            ccusage_command=args.ccusage_command,
            correction_note=args.correction_note,
        )
        revised = _session_update(
            record,
            state_root=args.state_root,
            accounting=accounting,
            action="capture_token_start",
            generated_at=args.generated_at,
            note=args.correction_note or "WRA-1 token START snapshot",
        )
    return with_guidance(
        {"ok": True, "session_id": session_id, "capture": capture, "session_revision": revised.get("revision")},
        status=str(capture.get("status") or "UNAVAILABLE"),
        reason_codes=["wra1_capture_start"],
        problems=list(capture.get("problems") or []),
        next_actions=[],
        related_records=[session_id, work_id],
    ), 0


def command_capture_end(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    session_id = validate_prefixed_id(args.session_id, SESSION_ID_PREFIX, label="session")
    initial = session_store(args.state_root).get_read_only(session_id)
    work_id = str(dict(initial.get("payload") or {}).get("work_id") or "")
    thread_id = str(dict(initial.get("payload") or {}).get("current_codex_id") or "").strip()
    with resource_thread_lock(args.state_root, thread_id):
        record = session_store(args.state_root).get(session_id)
        current_accounting = dict(dict(record.get("payload") or {}).get("resource_accounting") or {})
        active_capture_id = str(current_accounting.get("active_capture_id") or "")
        if args.correction_note and not args.capture_id:
            raise ActorError("resource correction requires --capture-id", code="resource_correction_capture_missing")
        if args.capture_id and str(args.capture_id).strip().upper() != active_capture_id:
            raise ActorError(
                "resource correction/closure does not name the exact active capture",
                code="resource_correction_capture_mismatch",
                problems=[f"expected {active_capture_id or '<none>'}; received {args.capture_id}"],
            )
        terminal = str(record.get("status") or "") in SESSION_TERMINAL_STATUSES
        accounting, artifacts = finalize_session(
            record,
            state_root=args.state_root,
            generated_at=args.generated_at,
            ccusage_command=args.ccusage_command,
            correction_note=args.correction_note,
            unavailable_reason=(
                "Session is already terminal; the historical END boundary cannot be reconstructed from the current cumulative counter"
                if terminal else None
            ),
        )
        revised = _session_update(
            record,
            state_root=args.state_root,
            accounting=accounting,
            action="capture_token_end",
            generated_at=args.generated_at,
            note=args.correction_note or "WRA-1 token END snapshot",
        )
    statuses = [str(row.get("status") or "UNAVAILABLE") for row in artifacts]
    status = "MEASURED" if statuses and all(value == "MEASURED" for value in statuses) else ("PARTIAL" if "MEASURED" in statuses else "UNAVAILABLE")
    return with_guidance(
        {"ok": True, "session_id": session_id, "token_usage_artifacts": artifacts, "session_revision": revised.get("revision")},
        status=status,
        reason_codes=["wra1_capture_end"],
        next_actions=[],
        related_records=[session_id, work_id],
    ), 0


def command_finalize_work(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    record = work_store(args.state_root).get(work_id)
    if str(record.get("status") or "") not in WORK_TERMINAL_STATUSES:
        raise ActorError(
            "manual Work resource finalization requires a terminal Work Item",
            code="token_finalize_work_not_terminal",
            related_records=[work_id],
            next_actions=[
                {
                    "action": "complete_work",
                    "description": "Complete or administratively retire the Work Item before finalizing a correction artifact.",
                }
            ],
        )
    artifact, reference = aggregate_work_usage(
        record,
        sessions_for_work(args.state_root, work_id),
        state_root=args.state_root,
        generated_at=args.generated_at,
        correction_note=args.correction_note,
    )
    payload = dict(record.get("payload") or {})
    projection = work_accounting_projection(payload, artifact_reference=reference, artifact=artifact)
    record = attach_work_resource_accounting(
        work_id,
        projection,
        state_root=args.state_root,
        generated_at=args.generated_at,
        action="finalize_resource_accounting",
        note=args.correction_note or "WRA-1 Work token usage finalized",
    )
    return with_guidance(
        {"ok": True, "work_id": work_id, "token_usage": artifact, "artifact": reference, "work_revision": record.get("revision")},
        status=str(artifact.get("status") or "UNAVAILABLE"),
        reason_codes=["wra1_work_finalized"],
        problems=list(artifact.get("problems") or []),
        next_actions=[],
        related_records=[work_id],
    ), 0


def command_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    records: list[dict[str, Any]] = []
    store = work_store(args.state_root)
    for row in store.list_read_only():
        if row.get("status") != "COMPLETED":
            continue
        try:
            records.append(store.get_read_only(str(row.get("record_id") or "")))
        except ActorError:
            continue
    index, index_path = export_token_report(
        records,
        output_dir=args.output_dir,
        state_root=args.state_root,
        generated_at=args.generated_at,
    )
    return with_guidance(
        {"ok": True, "report": index, "report_index_path": str(index_path)},
        status="TOKEN_REPORT_EXPORTED",
        reason_codes=["wra1_report_exported"],
        next_actions=[],
        related_records=[str(row.get("work_id")) for row in index.get("work_items") or []],
    ), 0


def command_coverage(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    works = []
    work_records = work_store(args.state_root)
    for row in work_records.list_read_only():
        try:
            works.append(work_records.get_read_only(str(row.get("record_id") or "")))
        except ActorError:
            continue
    sessions = []
    session_records = session_store(args.state_root)
    for row in session_records.list_read_only():
        try:
            sessions.append(session_records.get_read_only(str(row.get("record_id") or "")))
        except ActorError:
            continue
    report = resource_accounting_coverage(works, sessions)
    return with_guidance({"ok": True, "coverage": report}, status="WRA_COVERAGE_REPORTED", reason_codes=["wra_coverage_nonblocking"], next_actions=[], related_records=[]), 0


def command_estimator_review(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    refs = []
    for path in args.artifact_ref or []:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ActorError("estimator artifact reference must be an object", code="estimator_review_input_invalid")
        refs.append(dict(value))
    receipt, reference = retain_estimator_review_receipt(
        state_root=args.state_root, report_path=args.report, rules_path=args.rules,
        included_artifacts=refs, generated_at=args.generated_at,
    )
    return with_guidance({"ok": True, "estimator_review": receipt, "artifact": reference}, status="PENDING_HUMAN_APPROVAL", reason_codes=["estimator_review_retained"], next_actions=[], related_records=[]), 0


def command_estimator_approve(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    reference = json.loads(Path(args.review_ref).read_text(encoding="utf-8"))
    if not isinstance(reference, Mapping):
        raise ActorError("estimator review reference must be an object", code="estimator_review_reference_invalid")
    receipt, artifact = retain_estimator_approval_receipt(
        state_root=args.state_root, review_reference=reference, rules_after_path=args.rules_after,
        authority=args.authority, decision=args.decision, generated_at=args.generated_at,
    )
    return with_guidance({"ok": True, "estimator_approval": receipt, "artifact": artifact}, status=str(receipt.get("decision")), reason_codes=["estimator_human_decision_retained"], next_actions=[], related_records=[]), 0


def _human(payload: Mapping[str, Any]) -> str:
    if "capture" in payload:
        capture = dict(payload.get("capture") or {})
        return f"Session {payload.get('session_id')} token START: {capture.get('status')}\n"
    if "token_usage_artifacts" in payload:
        return f"Session {payload.get('session_id')} token END artifacts: {len(payload.get('token_usage_artifacts') or [])}\n"
    if "token_usage" in payload:
        usage = dict(payload.get("token_usage") or {})
        return f"Work {payload.get('work_id')} token usage: {usage.get('status')} total={usage.get('total_tokens')}\n"
    if "report" in payload:
        return f"Token report exported: {dict(payload.get('report') or {}).get('work_item_count', 0)} Work Items\n"
    return json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args: argparse.Namespace | None = None
    try:
        args = build_parser().parse_args(raw)
        handlers = {
            "capture-start": command_capture_start,
            "capture-end": command_capture_end,
            "finalize-work": command_finalize_work,
            "report": command_report,
            "coverage": command_coverage,
            "estimator-review": command_estimator_review,
            "estimator-approve": command_estimator_approve,
        }
        payload, code = handlers[args.command](args)
        emit_payload(payload, output=args.output, human_text=_human(payload), json_output=not args.human)
        return code
    except ActorError as exc:
        emit_actor_error(
            exc,
            actor=TOOL_NAME,
            command=raw[0] if raw else "unknown",
            raw_args=raw,
            parsed_args=args,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
