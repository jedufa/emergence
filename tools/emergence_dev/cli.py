"""Command-line interface for Emergence development automation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .benchmark import run_benchmark
from .codemods import apply_codemod, plan_codemod
from .core import DevAutomationError, build_context, write_context
from .tasks import (
    begin_task,
    build_change_plan,
    checkpoint_task,
    resume_task,
    scope_add,
    task_status,
)


def _objective(args: argparse.Namespace) -> str:
    if args.objective is not None:
        return args.objective
    try:
        return args.objective_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise DevAutomationError(f"cannot read objective file: {args.objective_file}") from exc


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--json", action="store_true")


def _state_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root", type=Path)


def _emit(payload: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if "context_markdown" in payload:
        print(payload["context_markdown"], end="")
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="emergence-dev", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="build a bounded advisory context packet")
    objective = prepare.add_mutually_exclusive_group(required=True)
    objective.add_argument("--objective")
    objective.add_argument("--objective-file", type=Path)
    prepare.add_argument("--output", type=Path)
    _common(prepare)

    change = subparsers.add_parser("change", help="manage a declared development change")
    change_subparsers = change.add_subparsers(dest="change_command", required=True)
    begin = change_subparsers.add_parser("begin", help="create a task baseline")
    objective = begin.add_mutually_exclusive_group(required=True)
    objective.add_argument("--objective")
    objective.add_argument("--objective-file", type=Path)
    begin.add_argument("--own", action="append", default=[], required=True)
    begin.add_argument("--adopt", action="append", default=[])
    begin.add_argument("--project-root", type=Path)
    _state_argument(begin)
    begin.add_argument("--json", action="store_true")
    status = change_subparsers.add_parser("status", help="inspect task-owned drift")
    status.add_argument("--task", required=True)
    _state_argument(status)
    status.add_argument("--json", action="store_true")
    plan = change_subparsers.add_parser("plan", help="build a validation and relationship plan")
    plan.add_argument("--task", required=True)
    _state_argument(plan)
    plan.add_argument("--json", action="store_true")
    scope = change_subparsers.add_parser("scope-add", help="add a justified task-owned path")
    scope.add_argument("--task", required=True)
    scope.add_argument("--path", required=True)
    scope.add_argument("--reason", required=True)
    scope.add_argument("--justification", required=True)
    scope.add_argument("--approve-scope")
    scope.add_argument("--adopt-sha256")
    _state_argument(scope)
    scope.add_argument("--json", action="store_true")
    validate = change_subparsers.add_parser("validate", help="run the exact CPU-only validation plan")
    validate.add_argument("--task", required=True)
    _state_argument(validate)
    validate.add_argument("--json", action="store_true")
    external_validate = change_subparsers.add_parser(
        "external-validate",
        help="execute and bind one mandatory external validation",
    )
    external_validate.add_argument("--task", required=True)
    external_validate.add_argument("--obligation", required=True)
    external_validate.add_argument(
        "--evidence-kind", choices=("pytest",), required=True
    )
    external_validate.add_argument(
        "--timeout-seconds", type=int, default=1800
    )
    _state_argument(external_validate)
    external_validate.add_argument("--json", action="store_true")
    external_validate.add_argument("external_command", nargs=argparse.REMAINDER)
    triage = change_subparsers.add_parser("triage", help="summarize the latest validation failures")
    triage.add_argument("--task", required=True)
    _state_argument(triage)
    triage.add_argument("--json", action="store_true")
    review = change_subparsers.add_parser("review", help="review final scope, validation, and diff")
    review.add_argument("--task", required=True)
    _state_argument(review)
    review.add_argument("--json", action="store_true")
    finish = change_subparsers.add_parser("finish", help="write the final task receipt")
    finish.add_argument("--task", required=True)
    finish.add_argument("--report", type=Path)
    _state_argument(finish)
    finish.add_argument("--json", action="store_true")

    task = subparsers.add_parser("task", help="checkpoint or resume a development task")
    task_subparsers = task.add_subparsers(dest="task_command", required=True)
    checkpoint = task_subparsers.add_parser("checkpoint")
    checkpoint.add_argument("--task", required=True)
    checkpoint.add_argument("--note", required=True)
    checkpoint.add_argument("--next-command", required=True)
    _state_argument(checkpoint)
    checkpoint.add_argument("--json", action="store_true")
    resume = task_subparsers.add_parser("resume")
    resume.add_argument("--task", required=True)
    _state_argument(resume)
    resume.add_argument("--json", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="run the curated preparation benchmark")
    benchmark.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/fixtures/emergence_dev_benchmark_v1.json"),
    )
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument("--project-root", type=Path)
    _state_argument(benchmark)
    benchmark.add_argument("--json", action="store_true")

    codemod = subparsers.add_parser("codemod", help="plan or apply a narrow mechanical transform")
    codemod_subparsers = codemod.add_subparsers(dest="codemod_command", required=True)
    codemod_plan = codemod_subparsers.add_parser("plan")
    codemod_plan.add_argument("--task", required=True)
    codemod_plan.add_argument("--spec", type=Path, required=True)
    codemod_plan.add_argument("--output", type=Path)
    _state_argument(codemod_plan)
    codemod_plan.add_argument("--json", action="store_true")
    codemod_apply = codemod_subparsers.add_parser(
        "apply",
        help="apply exactly one live output with exception-atomic rollback",
        description=(
            "Apply a reviewed plan containing exactly one live output. "
            "V1 rollback is exception-atomic, not process-crash-atomic."
        ),
    )
    codemod_apply.add_argument("--task", required=True)
    codemod_apply.add_argument("--plan", type=Path, required=True)
    codemod_apply.add_argument("--plan-hash", required=True)
    _state_argument(codemod_apply)
    codemod_apply.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            payload = build_context(
                _objective(args),
                project_root=args.project_root,
                config_path=args.config,
                state_root=args.state_root,
            )
            if args.output:
                write_context(args.output, payload)
            _emit(payload, as_json=args.json)
            return 0
        if args.command == "change":
            if args.change_command == "begin":
                payload = begin_task(
                    _objective(args),
                    args.own,
                    project_root=args.project_root,
                    state_root=args.state_root,
                    adoptions=args.adopt,
                )
            elif args.change_command == "status":
                payload = task_status(args.task, state_root=args.state_root)
            elif args.change_command == "plan":
                payload = build_change_plan(args.task, state_root=args.state_root)
            elif args.change_command == "scope-add":
                payload = scope_add(
                    args.task,
                    args.path,
                    args.reason,
                    justification=args.justification,
                    state_root=args.state_root,
                    approve_scope=args.approve_scope,
                    adopt_sha256=args.adopt_sha256,
                )
            elif args.change_command == "validate":
                from .validation import validate_task

                payload = validate_task(args.task, state_root=args.state_root)
            elif args.change_command == "external-validate":
                from .validation import record_external_validation

                payload = record_external_validation(
                    args.task,
                    args.obligation,
                    args.external_command,
                    evidence_kind=args.evidence_kind,
                    timeout_seconds=args.timeout_seconds,
                    state_root=args.state_root,
                )
            elif args.change_command == "triage":
                from .validation import triage_task

                payload = triage_task(args.task, state_root=args.state_root)
            elif args.change_command == "review":
                from .validation import review_task

                payload = review_task(args.task, state_root=args.state_root)
            elif args.change_command == "finish":
                from .validation import finish_task

                payload = finish_task(
                    args.task,
                    state_root=args.state_root,
                    report_path=args.report,
                )
            else:
                raise DevAutomationError(f"unsupported change command: {args.change_command}")
            _emit(payload, as_json=args.json)
            return (
                0
                if payload.get("status") not in {"blocked", "failed"}
                else 3
            )
        if args.command == "task":
            if args.task_command == "checkpoint":
                payload = checkpoint_task(
                    args.task,
                    note=args.note,
                    next_command=args.next_command,
                    state_root=args.state_root,
                )
            elif args.task_command == "resume":
                payload = resume_task(args.task, state_root=args.state_root)
            else:
                raise DevAutomationError(f"unsupported task command: {args.task_command}")
            _emit(payload, as_json=args.json)
            return 0 if payload.get("ok", True) else 3
        if args.command == "benchmark":
            payload = run_benchmark(
                args.manifest,
                project_root=args.project_root,
                state_root=args.state_root,
                output=args.output,
            )
            _emit(payload, as_json=args.json)
            return 0 if payload["go_no_go"] == "go" else 3
        if args.command == "codemod":
            if args.codemod_command == "plan":
                payload = plan_codemod(
                    args.task,
                    args.spec,
                    state_root=args.state_root,
                    output=args.output,
                )
            elif args.codemod_command == "apply":
                payload = apply_codemod(
                    args.task,
                    args.plan,
                    args.plan_hash,
                    state_root=args.state_root,
                )
            else:
                raise DevAutomationError(f"unsupported codemod command: {args.codemod_command}")
            _emit(payload, as_json=args.json)
            return 0
        raise DevAutomationError(f"unsupported command: {args.command}")
    except DevAutomationError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
