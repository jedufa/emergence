"""Argument parser for the Session Manager CLI.

Kept separate from lifecycle policy so command spelling can evolve without
turning the authority-bearing manager into another multi-thousand-line room.
"""

from __future__ import annotations

import argparse

from tools.actors.common import ActorArgumentParser
from tools.actors.portfolio import SESSION_TERMINAL_STATUSES
from tools.actors.work_package import AUTHORITY_ROLES, UNCERTAINTY_CLASSES


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root")
    parser.add_argument("--generated-at")
    parser.add_argument("--output")
    parser.add_argument("--human", action="store_true")


def build_parser(*, session_statuses: tuple[str, ...], session_outcomes: tuple[str, ...], review_verdicts: tuple[str, ...]) -> argparse.ArgumentParser:
    parser = ActorArgumentParser(
        prog="python -m tools.eiel sessionManager",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Command groups:
  Daily: start, list, status, inspect, checkpoint, context-receipt, note, interrupt, resume, handoff
  Lifecycle transitions: pause, block, request-promotion, complete, discontinue, obsolete, supersede
  Recovery/administration: attach-bootstrap, reconcile-terminal-parent, repair-references, history
  Scheduler-internal/advanced: prepare-assignment, activate-assignment, worker-loss

Canonical public spelling: sessionManager. The session-manager alias remains supported.""",
    )
    sub = parser.add_subparsers(dest="command", required=True, parser_class=ActorArgumentParser)

    start = sub.add_parser("start", help="Start a Codex session for one existing work item.")
    _common(start)
    start.add_argument("work_id")
    start.add_argument("--input")
    start.add_argument("--session-id")
    start.add_argument("--bootstrap-record", required=True)
    start.add_argument("--codex-id")
    start.add_argument("--codex-url")
    start.add_argument("--mission-id")
    start.add_argument("--title")
    start.add_argument("--repo-root")
    start.add_argument("--role", choices=AUTHORITY_ROLES)
    start.add_argument("--review-target-session", help="Completed delivery Session reviewed by a Review Authority Session; defaults to latest delivery.")
    start.add_argument("--model", help="Actual model selected by the human operator for this session.")
    start.add_argument("--reasoning-level", help="Actual reasoning level selected by the human operator.")
    start.add_argument("--max-bootstrap-age-hours", type=float)
    start.add_argument("--allow-stale-bootstrap", action="store_true")
    start.add_argument("--stage")
    start.add_argument("--current-step")
    start.add_argument("--next-action")
    start.add_argument("--next-command")
    start.add_argument("--allow-parallel", action="store_true")
    start.add_argument(
        "--acknowledge-default-context-contracts",
        action="store_true",
        help="Legacy escape hatch: acknowledge that delegated Spark is using default rather than explicit context contracts.",
    )
    start.add_argument("--note")

    prepare = sub.add_parser("prepare-assignment", help="Prepare a Session for an immutable Assignment without fabricating Codex identity.")
    _common(prepare)
    prepare.add_argument("work_id")
    prepare.add_argument("--bootstrap-record", required=True)
    prepare.add_argument("--repo-root", required=True)
    prepare.add_argument("--assignment-id", required=True)
    prepare.add_argument("--session-id", required=True)
    prepare.add_argument("--role", choices=AUTHORITY_ROLES, required=True)
    prepare.add_argument("--review-target-session")
    prepare.add_argument("--model")
    prepare.add_argument("--reasoning-level")

    activate = sub.add_parser("activate-assignment", help="Activate a prepared or resumable Session for one exact Assignment and Lease.")
    _common(activate)
    activate.add_argument("session_id")
    activate.add_argument("--assignment-id", required=True)
    activate.add_argument("--assignment-digest", required=True)
    activate.add_argument("--lease-id", required=True)
    activate.add_argument("--worker-id", required=True)
    activate.add_argument("--codex-id", required=True)
    activate.add_argument("--repo-root", required=True)

    worker_loss = sub.add_parser("worker-loss", help="Record mechanical loss of a scheduler worker without changing Work truth.")
    _common(worker_loss)
    worker_loss.add_argument("session_id")
    worker_loss.add_argument("--assignment-id", required=True)
    worker_loss.add_argument("--lease-id")
    worker_loss.add_argument("--reason", required=True)

    list_cmd = sub.add_parser("list", help="List sessions across all or one work item.")
    _common(list_cmd)
    list_cmd.add_argument("--work-id")
    list_cmd.add_argument("--status", choices=session_statuses, action="append")
    list_cmd.add_argument("--open-only", action="store_true")

    inspect = sub.add_parser("inspect", help="Inspect one session and its exact next action.")
    _common(inspect)
    inspect.add_argument("session_id")

    status = sub.add_parser("status", help="Inspect one session or list all resumable sessions.")
    _common(status)
    status.add_argument("session_id", nargs="?")

    checkpoint = sub.add_parser("checkpoint", help="Persist the exact current and next execution state.")
    _common(checkpoint)
    checkpoint.add_argument("session_id")
    checkpoint.add_argument("--patch")
    checkpoint.add_argument("--set", dest="set_values", action="append", default=[])
    checkpoint.add_argument("--current-stage")
    checkpoint.add_argument("--current-step")
    checkpoint.add_argument("--completed-step", action="append", default=[])
    checkpoint.add_argument("--next-action")
    checkpoint.add_argument("--next-command")
    checkpoint.add_argument("--interrupted-during")
    checkpoint.add_argument("--blocked-reason")
    checkpoint.add_argument("--success-command")
    checkpoint.add_argument("--success-result")
    checkpoint.add_argument("--failed-command")
    checkpoint.add_argument("--failed-result")
    checkpoint.add_argument("--validation-status", choices=("PASSED", "FAILED", "PARTIAL", "SKIPPED"))
    checkpoint.add_argument("--validation-command")
    checkpoint.add_argument("--validation-gap", action="append", default=[])
    checkpoint.add_argument("--validation-decision-impact")
    checkpoint.add_argument("--qualification-ref", action="append", default=[])
    checkpoint.add_argument("--resource-ref", action="append", default=[])
    checkpoint.add_argument("--evidence-ref", action="append", default=[])
    checkpoint.add_argument("--artifact-ref", action="append", default=[])
    checkpoint.add_argument("--report-ref", action="append", default=[])
    checkpoint.add_argument("--librarian-document-id", action="append", default=[])
    checkpoint.add_argument("--prompt-ref", action="append", default=[])
    checkpoint.add_argument("--external-url", action="append", default=[])
    checkpoint.add_argument("--tool-call-count", type=int)
    checkpoint.add_argument("--compaction-count", type=int)
    checkpoint.add_argument("--broad-output-events", type=int)
    checkpoint.add_argument("--shell-output-tokens", type=int)
    checkpoint.add_argument("--handoff-recommended", action="store_true")
    checkpoint.add_argument("--epoch-boundary-reason")
    checkpoint.add_argument("--context-receipt-output")
    checkpoint.add_argument("--note")

    attach_bootstrap = sub.add_parser("attach-bootstrap", help="Attach a fresh verify-only Bootstrap after a worker commit.")
    _common(attach_bootstrap)
    attach_bootstrap.add_argument("session_id")
    attach_bootstrap.add_argument("--bootstrap-record", required=True)
    attach_bootstrap.add_argument("--repo-root", required=True)

    receipt = sub.add_parser("context-receipt", help="Generate a compact immutable continuation receipt for this Session.")
    _common(receipt)
    receipt.add_argument("session_id")
    receipt.add_argument("--repo-root")
    receipt.add_argument("--receipt-output")
    receipt.add_argument("--attach", action="store_true", help="Record the receipt identity on the Session without changing authority.")
    receipt.add_argument("--note")

    note = sub.add_parser("note", help="Append a timestamped session note.")
    _common(note)
    note.add_argument("session_id")
    note.add_argument("text")
    note.add_argument("--note")

    pause = sub.add_parser("pause", help="Pause a session with a required next action.")
    _common(pause)
    pause.add_argument("session_id")
    pause.add_argument("--next-action", required=True)
    pause.add_argument("--next-command")
    pause.add_argument("--note")

    interrupt = sub.add_parser("interrupt", help="Record an interruption point and exact recovery action.")
    _common(interrupt)
    interrupt.add_argument("session_id")
    interrupt.add_argument("--interrupted-during", required=True)
    interrupt.add_argument("--next-action", required=True)
    interrupt.add_argument("--next-command")
    interrupt.add_argument("--note")

    block = sub.add_parser("block", help="Block a session on other work or an external condition.")
    _common(block)
    block.add_argument("session_id")
    block.add_argument("--reason", required=True)
    block.add_argument("--blocked-by", action="append", default=[])
    block.add_argument("--next-action")
    block.add_argument("--next-command")
    block.add_argument("--note")

    promote = sub.add_parser("request-promotion", help="Stop when Work Package uncertainty is higher than approved.")
    _common(promote)
    promote.add_argument("session_id")
    promote.add_argument("--requested-class", choices=UNCERTAINTY_CLASSES, required=True)
    promote.add_argument("--reason", required=True)
    promote.add_argument("--affected", action="append", default=[])
    promote.add_argument("--safe-state", required=True)
    promote.add_argument("--recommended-authority", choices=AUTHORITY_ROLES)
    promote.add_argument("--note")

    resume = sub.add_parser("resume", help="Resume with the original Codex or an explicit replacement.")
    _common(resume)
    resume.add_argument("session_id")
    resume.add_argument("--codex-id")
    resume.add_argument("--codex-url")
    resume.add_argument("--bootstrap-record")
    resume.add_argument("--repo-root")
    resume.add_argument("--max-bootstrap-age-hours", type=float)
    resume.add_argument("--allow-stale-bootstrap", action="store_true")
    resume.add_argument("--takeover-reason")
    resume.add_argument("--take-over-prepared", action="store_true", help="Explicitly detach an unstarted Scheduler Assignment and resume manually.")
    resume.add_argument("--allow-parallel", action="store_true")
    resume.add_argument("--note")

    complete = sub.add_parser("complete", help="Complete a session without completing its work item automatically.")
    _common(complete)
    complete.add_argument("session_id")
    complete.add_argument("--final-step")
    complete.add_argument("--outcome", choices=session_outcomes)
    complete.add_argument(
        "--delivery-commit",
        help="Exact delivery commit claimed by a package-bound delivery Session.",
    )
    complete.add_argument(
        "--adopt-existing-commit",
        action="store_true",
        help="Adopt pre-existing unmanaged delivery truth without claiming the Session produced it.",
    )
    complete.add_argument("--adoption-reason")
    complete.add_argument("--external-producer")
    complete.add_argument("--review-verdict", choices=review_verdicts)
    complete.add_argument("--review-proof", help="Structured evidence-entailing proof required for ACCEPT.")
    complete.add_argument("--supersedes-review", help="Explicit prior Review Session superseded by this verdict.")
    complete.add_argument("--acceptance-reviewed", action="append", default=[], help="Legacy label-only coverage; never sufficient for ACCEPT.")
    complete.add_argument("--tests-reviewed", action="append", default=[])
    complete.add_argument("--evidence-reviewed", action="append", default=[])
    complete.add_argument("--known-limitation", action="append", default=[])
    complete.add_argument("--review-finding", action="append", default=[])
    complete.add_argument("--repair-disposition", choices=("WITHIN_PACKAGE", "REVISED_PACKAGE_REQUIRED"))
    complete.add_argument("--note")

    review_progress = sub.add_parser("review-progress", help="Record structured Review progress without completing the Review Session.")
    _common(review_progress)
    review_progress.add_argument("session_id")
    review_progress.add_argument("--state", choices=("NOT_STARTED", "IN_PROGRESS", "DEFERRED", "COMPLETED"), default="IN_PROGRESS")
    review_progress.add_argument("--completed-obligation", action="append", default=[])
    review_progress.add_argument("--remaining-obligation", action="append", default=[])
    review_progress.add_argument("--last-finding-id")
    review_progress.add_argument("--defer-reason")
    review_progress.add_argument("--resume-condition")
    review_progress.add_argument("--note")

    defer_review = sub.add_parser("defer-review", help="Place a Review Session on an explicit machine-readable hold.")
    _common(defer_review)
    defer_review.add_argument("session_id")
    defer_review.add_argument("--reason", required=True)
    defer_review.add_argument("--resume-condition", required=True)
    defer_review.add_argument("--remaining-obligation", action="append", default=[])
    defer_review.add_argument("--note")

    for name, help_text in (("discontinue", "Discontinue a session."), ("obsolete", "Mark a session obsolete.")):
        cmd = sub.add_parser(name, help=help_text)
        _common(cmd)
        cmd.add_argument("session_id")
        cmd.add_argument("--note")

    supersede = sub.add_parser("supersede", help="Supersede one session with another existing session.")
    _common(supersede)
    supersede.add_argument("session_id")
    supersede.add_argument("--with", dest="with_session_id", required=True)
    supersede.add_argument("--note")

    reconcile = sub.add_parser("reconcile-terminal-parent", help="Close one non-terminal session after its work reached terminal status.")
    _common(reconcile)
    reconcile.add_argument("session_id")
    reconcile.add_argument("--status", choices=tuple(sorted(SESSION_TERMINAL_STATUSES)), required=True)
    reconcile.add_argument("--reason", required=True)
    reconcile.add_argument("--bootstrap-record")
    reconcile.add_argument("--repo-root")
    reconcile.add_argument("--max-bootstrap-age-hours", type=float)
    reconcile.add_argument("--allow-stale-bootstrap", action="store_true")
    reconcile.add_argument("--note")

    repair_references = sub.add_parser(
        "repair-references",
        help="Repair references on one terminal historical Session and retain an immutable repair history.",
    )
    _common(repair_references)
    repair_references.add_argument("session_id")
    repair_references.add_argument("--reason", required=True)
    repair_references.add_argument("--remove-evidence-ref", action="append", default=[])
    repair_references.add_argument("--artifact-ref", action="append", default=[])
    repair_references.add_argument("--report-ref", action="append", default=[])

    history = sub.add_parser("history", help="Show immutable session revisions.")
    _common(history)
    history.add_argument("session_id")

    handoff = sub.add_parser("handoff", help="Produce a concise context-recovery handoff.")
    _common(handoff)
    handoff.add_argument("session_id")
    handoff.add_argument("--repo-root")
    handoff.add_argument("--receipt-output")
    return parser

