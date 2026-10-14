#!/usr/bin/env python3
"""Plan, materialize, or run the authorized Phase F v2 battery."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = ROOT / "engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from evolution.phase_f_v2.matrix import load_phase_f_v2_matrix  # noqa: E402
from evolution.phase_f_v2.storage import atomic_write_canonical_json  # noqa: E402
from perf.cuda_architecture_b_phase_e_gpu_gate import (  # noqa: E402
    PhaseEGpuGateBindings,
)
from perf.eiel_phase_f_forced_combination_battery_v2 import (  # noqa: E402
    build_phase_f_battery_plan_v2,
    build_phase_f_entry_receipt_v2,
    execute_phase_f_battery_v2,
    execute_phase_f_cuda_owner_observation_v2,
    materialize_phase_f_cpu_manifest_v2,
    validate_phase_e_final_for_phase_f_v2,
)
from perf.eiel_phase_f_execution_source_v2 import (  # noqa: E402
    build_phase_f_migration_context_v2,
)
from perf.eiel_phase_f_forced_combination_fixture_v2 import (  # noqa: E402
    build_phase_f_fixture_registry_v2,
)


CANONICAL_MATRIX = (
    ROOT
    / "_spec"
    / "evolutionary_capacity"
    / "phase_f"
    / "v2"
    / "forced_combination_matrix.json"
)


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="ascii"))


def _checkpoint_prerequisites(path: str | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    value = _json(path)
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(receipt, dict)
        for key, receipt in value.items()
    ):
        raise ValueError("checkpoint prerequisites must map case IDs to receipts")
    return value


def _migration_contexts(
    *,
    matrix: dict[str, Any],
    registry: dict[str, Any],
    phase_e_final_receipt: dict[str, Any],
    source_commit: str,
) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for case in matrix["cases"]:
        source_case_id = case["role_contract"].get("boundary_source_case_id")
        if source_case_id is None:
            continue
        contexts[case["case_id"]] = build_phase_f_migration_context_v2(
            matrix=matrix,
            fixture_registry=registry,
            source_case_id=source_case_id,
            phase_e_final_receipt_sha256=phase_e_final_receipt["receipt_sha256"],
            phase_e_execution_commit=source_commit,
        )
    return contexts


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    matrix = load_phase_f_v2_matrix(args.matrix)
    registry = build_phase_f_fixture_registry_v2(matrix)
    plan = build_phase_f_battery_plan_v2(matrix=matrix, fixture_registry=registry)
    output = Path(args.output_directory)
    atomic_write_canonical_json(output / "battery_plan.json", plan)
    atomic_write_canonical_json(output / "fixture_registry.json", registry)
    return {
        "ok": True,
        "command": "plan",
        "plan_path": str(output / "battery_plan.json"),
        "plan_sha256": plan["plan_sha256"],
        "logical_cases": plan["logical_case_count"],
        "cpu_expectations": plan["cpu_expectation_count"],
        "accelerated_invocations": plan["accelerated_invocation_count"],
        "gpu_executed": False,
    }


def _cpu(args: argparse.Namespace) -> dict[str, Any]:
    matrix = load_phase_f_v2_matrix(args.matrix)
    registry = build_phase_f_fixture_registry_v2(matrix)
    plan = _json(args.plan)
    final_receipt = _json(args.phase_e_final_receipt)
    manifest = materialize_phase_f_cpu_manifest_v2(
        matrix=matrix,
        fixture_registry=registry,
        plan=plan,
        output_directory=args.output_directory,
        checkpoint_prerequisites=_checkpoint_prerequisites(
            args.checkpoint_prerequisites
        ),
        migration_contexts=_migration_contexts(
            matrix=matrix,
            registry=registry,
            phase_e_final_receipt=final_receipt,
            source_commit=args.source_commit,
        ),
    )
    return {
        "ok": True,
        "command": "cpu",
        "manifest_path": str(
            Path(args.output_directory) / "cpu_expectation_manifest.json"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "expectation_count": manifest["expectation_count"],
        "gpu_executed": False,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_directory).resolve()
    repo_root = Path(args.repo_root).resolve()
    if output == repo_root or repo_root in output.parents:
        raise ValueError("run output_directory must be outside the frozen checkout")
    matrix = load_phase_f_v2_matrix(args.matrix)
    registry = build_phase_f_fixture_registry_v2(matrix)
    plan = _json(args.plan)
    cpu_manifest = _json(args.cpu_manifest)
    matrix_receipt = _json(args.matrix_receipt)
    final_receipt = _json(args.phase_e_final_receipt)
    qualification_receipt = _json(args.phase_e_qualification_receipt)
    raw_bindings = _json(args.phase_e_bindings)
    bindings = PhaseEGpuGateBindings(**raw_bindings)
    authorization = validate_phase_e_final_for_phase_f_v2(
        final_receipt=final_receipt,
        qualification_receipt=qualification_receipt,
        bindings=bindings,
        source_commit=args.source_commit,
    )
    entry = build_phase_f_entry_receipt_v2(
        plan=plan,
        matrix_receipt=matrix_receipt,
        source_commit=args.source_commit,
        phase_e_final_receipt=final_receipt,
        phase_e_authorization=authorization,
        cpu_manifest=cpu_manifest,
        source_manifest_sha256=args.source_manifest_sha256,
        repo_root=repo_root,
    )
    atomic_write_canonical_json(output / "execution_entry_receipt.json", entry)
    owner = partial(
        execute_phase_f_cuda_owner_observation_v2,
        build_directory=args.build_directory,
    )
    ledger = execute_phase_f_battery_v2(
        matrix=matrix,
        fixture_registry=registry,
        plan=plan,
        cpu_manifest=cpu_manifest,
        entry_receipt=entry,
        output_directory=output,
        source_commit=args.source_commit,
        matrix_receipt_sha256=matrix_receipt["receipt_sha256"],
        owner_executor=owner,
        checkpoint_prerequisites=_checkpoint_prerequisites(
            args.checkpoint_prerequisites
        ),
        migration_contexts=_migration_contexts(
            matrix=matrix,
            registry=registry,
            phase_e_final_receipt=final_receipt,
            source_commit=args.source_commit,
        ),
    )
    return {
        "ok": ledger["status"] == "passed",
        "command": "run",
        "status": ledger["status"],
        "smoke_status": ledger["smoke_status"],
        "completed_count": ledger["completed_count"],
        "ledger_path": str(output / "resume_ledger.json"),
        "first_difference": ledger["first_difference"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "cpu", "run"):
        command = sub.add_parser(name)
        command.add_argument("--matrix", type=Path, default=CANONICAL_MATRIX)
        command.add_argument("--output-directory", type=Path, required=True)
    cpu = sub.choices["cpu"]
    cpu.add_argument("--plan", type=Path, required=True)
    cpu.add_argument("--checkpoint-prerequisites", type=Path)
    cpu.add_argument("--phase-e-final-receipt", type=Path, required=True)
    cpu.add_argument("--source-commit", required=True)

    run = sub.choices["run"]
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--cpu-manifest", type=Path, required=True)
    run.add_argument("--matrix-receipt", type=Path, required=True)
    run.add_argument("--phase-e-final-receipt", type=Path, required=True)
    run.add_argument("--phase-e-qualification-receipt", type=Path, required=True)
    run.add_argument("--phase-e-bindings", type=Path, required=True)
    run.add_argument("--source-manifest-sha256", required=True)
    run.add_argument("--source-commit", required=True)
    run.add_argument("--repo-root", type=Path, default=ROOT)
    run.add_argument("--build-directory", type=Path)
    run.add_argument("--checkpoint-prerequisites", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = {"plan": _plan, "cpu": _cpu, "run": _run}[args.command](args)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
