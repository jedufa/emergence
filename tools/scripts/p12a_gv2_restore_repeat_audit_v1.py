#!/usr/bin/env python3
"""Run a bounded P12A repeated fresh-owner restore audit.

This is evidence tooling, not a new campaign route.  It reloads one already
materialized primary checkpoint twice through the registered production
provider and writes a compact receipt over the comparison surfaces.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import canonical_sha256, pretty_json  # noqa: E402
from engine.campaigns.orchestration import (  # noqa: E402
    BatchExecutionContextV1,
    _batch_rows,
    _bind_checkpoint_route,
    _compare_branch,
    _execution_route_sha256,
    _narrow_request,
    _provider_result,
    atomic_write_json,
)
from engine.campaigns.providers import (  # noqa: E402
    default_execution_provider_registry,
)


SCHEMA_VERSION = "p12a_gv2_repeated_restore_audit_v1"


def _read_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{name} must be a JSON object: {path}")
    return value


def _source_batch(
    batches: Sequence[Mapping[str, Any]], source_cell_id: str
) -> tuple[Mapping[str, Any], int]:
    matches = [
        (batch, list(batch["cell_ids"]).index(source_cell_id))
        for batch in batches
        if source_cell_id in batch.get("cell_ids", [])
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"source cell must occur in exactly one batch: {source_cell_id}"
        )
    return matches[0]


def _manifest_seed(
    *,
    plan: Mapping[str, Any],
    provider: Any,
    run_label: str,
    source_cell_id: str,
    split_generation: int,
    terminal_generation: int,
    source_checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "p12a_gv2_repeated_restore_run_manifest_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile_id": plan["resolved_profile"],
        "provider_id": provider.provider_id,
        "execution_route_sha256": _execution_route_sha256(provider, plan),
        "run_label": run_label,
        "source_cell_id": source_cell_id,
        "split_generation": split_generation,
        "terminal_generation": terminal_generation,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "status": "running",
        "phase": "restore_dispatch",
        "checkpoint": None,
        "metrics_rows": [],
        "result": None,
        "comparison": None,
        "error": None,
    }


def _run_restore(
    *,
    plan: Mapping[str, Any],
    provider: Any,
    base_request: Any,
    source_index: int,
    source_cell_id: str,
    checkpoint: Mapping[str, Any],
    output_root: Path,
    run_label: str,
    baseline: Mapping[str, Any],
    terminal_generation: int,
    checkpoint_milestones: tuple[int, ...],
    split_generation: int,
    source_checkpoint_sha256: str,
) -> dict[str, Any]:
    manifest_path = output_root / run_label / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"repeat restore output already exists: {manifest_path}")
    manifest = _manifest_seed(
        plan=plan,
        provider=provider,
        run_label=run_label,
        source_cell_id=source_cell_id,
        split_generation=split_generation,
        terminal_generation=terminal_generation,
        source_checkpoint_sha256=source_checkpoint_sha256,
    )
    atomic_write_json(manifest_path, manifest)
    request = _narrow_request(
        base_request,
        branch_id=run_label,
        source_index=source_index,
        source_cell_id=source_cell_id,
        generations=terminal_generation,
        checkpoints=checkpoint_milestones,
        reporting_enabled=True,
    )
    checkpoint_rebinder = getattr(provider, "prepare_fresh_owner_restore_checkpoint", None)
    if callable(checkpoint_rebinder):
        checkpoint = checkpoint_rebinder(
            plan,
            base_request,
            request,
            checkpoint,
        )
    context = BatchExecutionContextV1(
        run_root=output_root,
        batch_root=manifest_path.parent / "continued",
        manifest_path=manifest_path,
        execution_route_sha256=_execution_route_sha256(provider, plan),
    )
    try:
        raw_result = provider.resume_batch(
            plan,
            request,
            copy.deepcopy(dict(checkpoint)),
            context,
        )
        result = _provider_result(
            raw_result,
            execution_route_sha256=_execution_route_sha256(provider, plan),
        )
        current = _read_object(manifest_path, name="repeat restore manifest")
        current["status"] = result["status"]
        current["phase"] = "terminal_completed"
        current["checkpoint"] = dict(result.get("checkpoint") or {})
        current["metrics_rows"] = [dict(row) for row in result.get("metrics_rows", [])]
        current["result"] = dict(result.get("result") or {})
        current["comparison"] = _compare_branch(
            baseline=baseline,
            candidate=current,
            cell_id=source_cell_id,
            provider=provider,
            continuation=True,
        )
        current["status"] = (
            "completed" if current["comparison"]["match"] else "failed"
        )
        atomic_write_json(manifest_path, current)
    except Exception as exc:
        current = _read_object(manifest_path, name="repeat restore manifest")
        current["status"] = "failed"
        current["phase"] = "failed"
        current["error"] = {"type": type(exc).__name__, "message": str(exc)}
        atomic_write_json(manifest_path, current)
        raise
    return _read_object(manifest_path, name="repeat restore manifest")


def run_audit(
    *,
    run_root: Path,
    source_cell_id: str,
    split_generation: int,
    output_root: Path,
) -> dict[str, Any]:
    plan = _read_object(run_root / "execution_plan.json", name="execution plan")
    provider = default_execution_provider_registry().get(str(plan["resolved_profile"]))
    if provider is None:
        raise SystemExit(f"no provider for profile {plan['resolved_profile']!r}")
    batches = _batch_rows(plan)
    source_batch, source_index = _source_batch(batches, source_cell_id)
    primary_manifest = _read_object(
        run_root / "batches" / str(source_batch["batch_id"]) / "manifest.json",
        name="primary batch manifest",
    )
    result = primary_manifest.get("result")
    catalog = result.get("checkpoint_catalog") if isinstance(result, Mapping) else None
    if not isinstance(catalog, Mapping):
        raise SystemExit("primary manifest lacks checkpoint catalog")
    checkpoint = catalog.get(str(split_generation))
    if not isinstance(checkpoint, Mapping):
        raise SystemExit(f"primary manifest lacks G{split_generation} checkpoint")
    route_sha256 = _execution_route_sha256(provider, plan)
    checkpoint = _bind_checkpoint_route(
        checkpoint,
        route_sha256,
        require_existing=True,
    )
    terminal_generation = int(plan["execution"]["generations"])
    if split_generation >= terminal_generation:
        raise SystemExit("split generation must be before terminal generation")
    checkpoint_milestones = tuple(
        sorted(
            value
            for value in plan["execution"]["checkpoints"]
            if type(value) is int and value <= terminal_generation
        )
    )
    base_request = provider.translate_batch(plan, str(source_batch["batch_id"]))
    source_checkpoint_sha256 = canonical_sha256(checkpoint)
    output_root.mkdir(parents=True, exist_ok=True)
    restores = [
        _run_restore(
            plan=plan,
            provider=provider,
            base_request=base_request,
            source_index=source_index,
            source_cell_id=source_cell_id,
            checkpoint=checkpoint,
            output_root=output_root,
            run_label=f"repeat-restore-{index}",
            baseline=primary_manifest,
            terminal_generation=terminal_generation,
            checkpoint_milestones=checkpoint_milestones,
            split_generation=split_generation,
            source_checkpoint_sha256=source_checkpoint_sha256,
        )
        for index in (1, 2)
    ]
    pair_comparison = _compare_branch(
        baseline=restores[0],
        candidate=restores[1],
        cell_id=source_cell_id,
        provider=provider,
        continuation=True,
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed"
            if all(row["status"] == "completed" for row in restores)
            and all(row["comparison"]["match"] for row in restores)
            and pair_comparison["match"]
            else "failed"
        ),
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "provider_id": provider.provider_id,
        "profile_id": plan["resolved_profile"],
        "execution_route_sha256": route_sha256,
        "source_cell_id": source_cell_id,
        "source_batch_id": source_batch["batch_id"],
        "split_generation": split_generation,
        "terminal_generation": terminal_generation,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_checkpoint": {
            "generation": checkpoint.get("generation"),
            "mechanics_checkpoint_digests": checkpoint.get(
                "mechanics_checkpoint_digests"
            ),
            "canonical_checkpoint_storage_digests": checkpoint.get(
                "canonical_checkpoint_storage_digests"
            ),
            "terminal_cell_digests": checkpoint.get("terminal_cell_digests"),
        },
        "restore_runs": [
            {
                "run_label": row["run_label"],
                "manifest_path": str(output_root / row["run_label"] / "manifest.json"),
                "status": row["status"],
                "native_dispatches": (row.get("result") or {}).get(
                    "native_dispatches"
                ),
                "fresh_owner_restore_receipt": (row.get("result") or {}).get(
                    "fresh_owner_restore_receipt"
                ),
                "comparison": row["comparison"],
            }
            for row in restores
        ],
        "repeat_pair_comparison": pair_comparison,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    atomic_write_json(output_root / "p12a_gv2_repeated_restore_audit_v1.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-cell-id", required=True)
    parser.add_argument("--split-generation", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_audit(
        run_root=args.run_root.expanduser().resolve(),
        source_cell_id=args.source_cell_id,
        split_generation=args.split_generation,
        output_root=args.output_root.expanduser().resolve(),
    )
    print(pretty_json({"ok": receipt["status"] == "passed", **receipt}))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
