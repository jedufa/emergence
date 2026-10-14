"""Argument parser for the Work Manager CLI."""

from __future__ import annotations

import argparse

from tools.actors.common import ActorArgumentParser


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root")
    parser.add_argument("--generated-at")
    parser.add_argument("--output")
    parser.add_argument("--human", action="store_true")


def build_parser(*, work_types: tuple[str, ...], priorities: tuple[str, ...], work_statuses: tuple[str, ...]) -> argparse.ArgumentParser:
    parser = ActorArgumentParser(
        prog="python -m tools.eiel workManager",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Command groups:
  Daily: create, list, status, inspect, update, next, handoff
  Lifecycle transitions: pause, block, resume, reopen, continue, complete, repair-chain, discontinue, obsolete, supersede
  Recovery/administration: reconcile, dependency, refresh, graph, history, automation-candidates

Canonical public spelling: workManager. The work-manager alias remains supported.""",
    )
    sub = parser.add_subparsers(dest="command", required=True, parser_class=ActorArgumentParser)

    create = sub.add_parser("create", help="Create one durable work item.")
    _common(create)
    create.add_argument("--input")
    create.add_argument("--work-id")
    create.add_argument("--work-package", action="append", default=[], help="Bind one repository-native YAML or JSON Work Package; repeat for an ordered package chain.")
    create.add_argument("--intermediate-review-sequence", action="append", type=int, default=[], help="Require substantive review after this package sequence; independent identity is preferred but not required.")
    create.add_argument("--repo-root", help="Repository root used to resolve --work-package.")
    create.add_argument("--title")
    create.add_argument("--summary")
    create.add_argument("--work-type", choices=work_types)
    create.add_argument(
        "--completion-policy",
        choices=("REPOSITORY_DELIVERY", "QUALIFICATION_DECISION", "RECORD_ONLY"),
        help="Optional formal completion contract. Unpackaged non-qualification Work defaults to RECORD_ONLY.",
    )
    create.add_argument("--subsystem", help="Optional subsystem used by WRA-1 reporting and estimator review.")
    create.add_argument("--estimated-tokens-low", type=int)
    create.add_argument("--estimated-tokens-expected", type=int)
    create.add_argument("--estimated-tokens-high", type=int)
    create.add_argument("--estimate-confidence", choices=("none", "low", "medium", "high"))
    create.add_argument("--priority", choices=priorities)
    create.add_argument("--status", choices=("BACKLOG", "READY", "ACTIVE", "BLOCKED"))
    create.add_argument("--mission-id")
    create.add_argument("--parent-work-id")
    create.add_argument("--dependency", action="append", default=[])
    create.add_argument("--tag", action="append", default=[])
    create.add_argument("--current-stage")
    create.add_argument("--current-step")
    create.add_argument("--last-completed-step")
    create.add_argument("--next-action")
    create.add_argument("--next-command")
    create.add_argument("--preferred-codex-conversation-id")
    create.add_argument("--preferred-codex-thread-url")
    create.add_argument(
        "--storage-evidence-root",
        action="append",
        default=[],
        help="Explicit runtime evidence root to close out automatically after accepted integration.",
    )
    create.add_argument("--note")

    list_cmd = sub.add_parser("list", help="List the work portfolio.")
    _common(list_cmd)
    list_cmd.add_argument("--status", choices=work_statuses, action="append")
    list_cmd.add_argument("--priority", choices=priorities, action="append")
    list_cmd.add_argument("--open-only", action="store_true")
    list_cmd.add_argument("--view", choices=("fifo", "priority", "recent", "blocked"), default="fifo")

    inspect = sub.add_parser("inspect", help="Inspect one work item, dependencies, sessions, and next action.")
    _common(inspect)
    inspect.add_argument("work_id")
    inspect.add_argument("--repo-root", help="Current authoritative repository root for Work Package verification.")

    status = sub.add_parser("status", help="Inspect one work item or list open work.")
    _common(status)
    status.add_argument("work_id", nargs="?")
    status.add_argument(
        "--repo-root",
        help="Current authoritative repository root for Work Package verification.",
    )

    update = sub.add_parser("update", help="Revise work metadata and retain immutable history.")
    _common(update)
    update.add_argument("work_id")
    update.add_argument("--patch")
    update.add_argument("--set", dest="set_values", action="append", default=[])
    update.add_argument("--work-package", action="append", default=[], help="Replace or extend the ordered Work Package chain after old sessions are terminal.")
    update.add_argument("--intermediate-review-sequence", action="append", type=int, default=[], help="Require substantive review after this package sequence; independent identity is preferred but not required.")
    update.add_argument("--repo-root", help="Repository root used to resolve --work-package.")
    update.add_argument("--title")
    update.add_argument("--summary")
    update.add_argument("--subsystem")
    update.add_argument("--priority", choices=priorities)
    update.add_argument("--mission-id")
    update.add_argument("--current-stage")
    update.add_argument("--current-step")
    update.add_argument("--last-completed-step")
    update.add_argument("--next-action")
    update.add_argument("--next-command")
    update.add_argument("--preferred-codex-conversation-id")
    update.add_argument("--preferred-codex-thread-url")
    update.add_argument(
        "--storage-evidence-root",
        action="append",
        default=[],
        help="Add an exact runtime evidence root for managed merge closeout.",
    )
    update.add_argument("--qualification-ref", action="append", default=[])
    update.add_argument("--resource-ref", action="append", default=[])
    update.add_argument("--evidence-ref", action="append", default=[])
    update.add_argument("--artifact-ref", action="append", default=[])
    update.add_argument("--report-ref", action="append", default=[])
    update.add_argument("--librarian-document-id", action="append", default=[])
    update.add_argument("--external-url", action="append", default=[])
    update.add_argument("--note")

    evidence = sub.add_parser("evidence-requirement", help="Declare whether Evidence Decision records are a gate or provenance only.")
    _common(evidence)
    evidence.add_argument("work_id")
    evidence.add_argument("--mode", choices=("GATE", "PROVENANCE_ONLY"), required=True)
    evidence.add_argument("--assessment-id", action="append", default=[])
    evidence.add_argument("--accepted-decision", action="append", default=[])
    evidence.add_argument("--supports-proposition", required=True)
    evidence.add_argument("--note")

    hold = sub.add_parser("hold", help="Place Work on a typed operator/Review/resource hold.")
    _common(hold)
    hold.add_argument("work_id")
    hold.add_argument("--kind", choices=("REVIEW_DEFERRED", "RESOURCE_WAIT", "OPERATOR_DECISION"), required=True)
    hold.add_argument("--reason", required=True)
    hold.add_argument("--authority-role", required=True)
    hold.add_argument("--authority-identity", required=True)
    hold.add_argument("--resume-condition", required=True)
    hold.add_argument("--resume-after-utc")
    hold.add_argument("--note")

    clear_hold = sub.add_parser("clear-hold", help="Clear a typed Work hold after its retained condition is satisfied.")
    _common(clear_hold)
    clear_hold.add_argument("work_id")
    clear_hold.add_argument("--next-action", required=True)
    clear_hold.add_argument("--note")

    component = sub.add_parser("add-delivery-component", help="Record one parallel Session contribution and its disposition.")
    _common(component)
    component.add_argument("work_id")
    component.add_argument("--source-session-id", required=True)
    component.add_argument("--source-commit")
    component.add_argument("--contribution-path", action="append", default=[])
    component.add_argument("--disposition", choices=("INCLUDED", "MODIFIED", "REFERENCE_ONLY", "REJECTED"), required=True)
    component.add_argument("--reason", required=True)
    component.add_argument("--note")

    pause = sub.add_parser("pause", help="Pause work while retaining its exact next action.")
    _common(pause)
    pause.add_argument("work_id")
    pause.add_argument("--next-action", required=True)
    pause.add_argument("--next-command")
    pause.add_argument("--note")

    block = sub.add_parser("block", help="Block work on one or more dependency work items.")
    _common(block)
    block.add_argument("work_id")
    block.add_argument("--blocked-by", action="append", default=[])
    block.add_argument("--reason", required=True)
    block.add_argument("--next-action")
    block.add_argument("--next-command")
    block.add_argument("--note")

    resume = sub.add_parser("resume", help="Resume work after all dependencies are complete.")
    _common(resume)
    resume.add_argument("work_id")
    resume.add_argument("--ready", action="store_true")
    resume.add_argument("--note")

    reopen = sub.add_parser(
        "reopen",
        help="Reactivate ordinarily discontinued work while preserving immutable closure history.",
    )
    _common(reopen)
    reopen.add_argument("work_id")
    reopen.add_argument("--reason", required=True)
    reopen.add_argument("--next-action")
    reopen.add_argument("--next-command")
    reopen.add_argument("--note")

    continue_cmd = sub.add_parser(
        "continue",
        help="Automatically reconcile safe administrative friction or create/reuse a successor for terminal historical Work.",
    )
    _common(continue_cmd)
    continue_cmd.add_argument("work_id")
    continue_cmd.add_argument("--reason", default="Operational continuity requested by operator.")

    complete = sub.add_parser("complete", help="Complete work; active sessions must already be closed.")
    _common(complete)
    complete.add_argument("work_id")
    complete.add_argument("--last-completed-step")
    complete.add_argument("--review-session", help="Independent completed Review Authority session with ACCEPT verdict.")
    complete.add_argument("--repo-root", help="Current authoritative repository root for Work Package verification.")
    complete.add_argument("--integration-commit", help="Current integration commit when it differs from the reviewed delivery commit.")
    complete.add_argument("--integration-note")
    complete.add_argument(
        "--storage-evidence-root",
        action="append",
        default=[],
        help="Exact runtime evidence root to release after accepted integration closeout.",
    )
    complete.add_argument("--record-only-reason", help="Optional note for non-repository Work completion; never substitutes for repository delivery provenance.")
    complete.add_argument("--force", action="store_true")
    complete.add_argument("--note")

    repair_chain = sub.add_parser(
        "repair-chain",
        help="Reopen a completed Work whose multi-package chain still has an undelivered package.",
    )
    _common(repair_chain)
    repair_chain.add_argument("work_id")
    repair_chain.add_argument("--reason", required=True)
    repair_chain.add_argument(
        "--acknowledge-intermediate-completion",
        action="store_true",
        help="Acknowledge that the historical completion was for an intermediate package.",
    )

    reconcile = sub.add_parser("reconcile", help="Administratively retire corrupted Work state without acceptance.")
    _common(reconcile)
    reconcile.add_argument("work_id")
    reconcile.add_argument("--disposition", choices=("DISCONTINUE", "OBSOLETE"), required=True)
    reconcile.add_argument("--reason", required=True)
    reconcile.add_argument("--acknowledge-no-acceptance", action="store_true")
    reconcile.add_argument("--note")

    for name, help_text in (("discontinue", "Discontinue work by operator decision."), ("obsolete", "Mark work obsolete because its basis no longer applies.")):
        cmd = sub.add_parser(name, help=help_text)
        _common(cmd)
        cmd.add_argument("work_id")
        cmd.add_argument("--force", action="store_true")
        cmd.add_argument("--note")

    supersede = sub.add_parser("supersede", help="Supersede old work with another existing work item.")
    _common(supersede)
    supersede.add_argument("work_id")
    supersede.add_argument("--with", dest="with_work_id", required=True)
    supersede.add_argument("--note")

    dependency = sub.add_parser("dependency", help="Add or remove a work dependency.")
    _common(dependency)
    dependency.add_argument("operation", choices=("add", "remove"))
    dependency.add_argument("work_id")
    dependency.add_argument("dependency_work_id")
    dependency.add_argument("--note")

    refresh = sub.add_parser("refresh", help="Report blocked Work whose dependencies are now complete.")
    _common(refresh)

    graph = sub.add_parser("graph", help="Show dependency edges and blocker state.")
    _common(graph)
    graph.add_argument("work_id", nargs="?")

    history = sub.add_parser("history", help="Show immutable work revisions.")
    _common(history)
    history.add_argument("work_id")

    next_cmd = sub.add_parser("next", help="Show the exact safe next action for one work item.")
    _common(next_cmd)
    next_cmd.add_argument("work_id")

    handoff = sub.add_parser("handoff", help="Create a concise work handoff for another Codex.")
    _common(handoff)
    handoff.add_argument("work_id")

    automation = sub.add_parser("automation-candidates", help="Report deterministic scheduler eligibility without claiming or launching work.")
    _common(automation)
    automation.add_argument("--eligible-only", action="store_true")
    return parser
