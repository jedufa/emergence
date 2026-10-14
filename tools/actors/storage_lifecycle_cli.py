"""Operator diagnostics for lifecycle reclamation and merge reconciliation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from tools.actors.common import ActorArgumentParser, ActorError, emit_actor_error, emit_payload
from tools.actors.work_manager import (
    WORK_ID_PREFIX,
    _normalize_payload,
    _revise,
    _summary,
    work_lifecycle_lock,
    work_store,
)
from tools.actors.work_manager import validate_prefixed_id

from .storage_lifecycle import closeout_merged_work_storage_v1


def _parser() -> argparse.ArgumentParser:
    parser = ActorArgumentParser(
        prog="python -m tools.eiel storage",
        description=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True, parser_class=ActorArgumentParser)
    reconcile = sub.add_parser(
        "reconcile-merged-work",
        help="Close out exact transient evidence for an already completed integrated Work item.",
    )
    reconcile.add_argument("work_id")
    reconcile.add_argument("--repo-root", required=True)
    reconcile.add_argument("--state-root")
    reconcile.add_argument("--storage-evidence-root", action="append", default=[])
    reconcile.add_argument("--generated-at")
    reconcile.add_argument("--note")
    reconcile.add_argument("--output")
    reconcile.add_argument("--human", action="store_true")
    return parser


def _human(payload: dict) -> str:
    closeout = payload.get("storage_merge_closeout") or payload
    return (
        f"Storage merge reconciliation: {closeout.get('status', 'UNKNOWN')}\n"
        f"Bytes reclaimed: {closeout.get('bytes_reclaimed', 0)}\n"
        f"Bytes retained: {closeout.get('bytes_retained', 0)}\n"
        f"Receipt: {(closeout.get('receipt') or {}).get('path', closeout.get('receipt_path', 'none'))}\n"
    )


def _reconcile(args: argparse.Namespace) -> tuple[dict, int]:
    work_id = validate_prefixed_id(args.work_id, WORK_ID_PREFIX, label="work item")
    with work_lifecycle_lock(args.state_root, work_id):
        record = work_store(args.state_root, write_scope=True).get(work_id)
        payload = dict(record.get("payload") or {})
        if record.get("status") != "COMPLETED" or payload.get("accepted") is not True:
            raise ActorError(
                "direct-Git reconciliation requires accepted completed Work",
                code="storage_reconcile_work_not_accepted",
                problems=[f"status={record.get('status')}", f"accepted={payload.get('accepted')}"],
            )
        integration = payload.get("repository_integration_receipt")
        if not isinstance(integration, dict):
            raise ActorError(
                "completed Work has no repository integration receipt",
                code="storage_reconcile_integration_missing",
            )
        closeout = closeout_merged_work_storage_v1(
            work_id=work_id,
            payload=payload,
            integration_reference=integration,
            repo_root=args.repo_root,
            state_root=args.state_root,
            evidence_roots=args.storage_evidence_root,
            generated_at=args.generated_at,
        )
        payload["storage_merge_closeout"] = dict(closeout["receipt"])
        payload = _normalize_payload(
            payload,
            work_id=work_id,
            generated_at=str(record.get("created_at_utc") or args.generated_at or ""),
        )
        revised = _revise(
            args,
            record,
            payload=payload,
            status="COMPLETED",
            action="storage_merge_reconcile",
            note=args.note,
        )
        return {
            "ok": True,
            "work": _summary(revised, state_root=args.state_root),
            "storage_merge_closeout": closeout,
            "direct_git_reconciliation": True,
        }, 0


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args: argparse.Namespace | None = None
    try:
        args = _parser().parse_args(raw)
        payload, code = _reconcile(args)
        emit_payload(payload, output=args.output, human_text=_human(payload), json_output=not args.human)
        return code
    except ActorError as exc:
        emit_actor_error(exc, actor="storage", command=raw[0] if raw else "unknown", raw_args=raw, parsed_args=args)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
