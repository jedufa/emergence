#!/usr/bin/env python3
"""Reduce a P12B run with the compact terminal descendant oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns import p12b_compact_descendant_oracle as oracle  # noqa: E402
from engine.campaigns.canonical import canonical_sha256, pretty_json  # noqa: E402


SCHEMA_VERSION = "p12b_compact_terminal_closure_reduction_v1"
SCIENTIFIC_MILESTONES = (5, 10, 20, 30)
TECHNICAL_SEGMENTS = (5, 10, 15, 20, 25, 30)
NATIVE_RUNTIME_CHECKPOINT_PROJECTION_FILENAME = "native_runtime_checkpoint_projection.json"
NATIVE_RUNTIME_CHECKPOINT_PROJECTION_GZIP_FILENAME = (
    "native_runtime_checkpoint_projection.json.gz"
)
POSITIVE_IDENTITY_TYPES = (
    "allele_genotype_state_identity",
    "preregistered_package_identity",
)


class P12BCompactTerminalClosureError(RuntimeError):
    """Campaign root cannot be reduced to a terminal P12B decision."""


def _fail(message: str) -> None:
    raise P12BCompactTerminalClosureError(message)


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P12BCompactTerminalClosureError(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{name} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch_manifest_paths(run_root: Path) -> list[Path]:
    paths = sorted((run_root / "batches").glob("batch-*/manifest.json"))
    if not paths:
        _fail(f"run root has no batch manifests: {run_root}")
    return paths


def _sidecar_path(bundle: Mapping[str, Any]) -> Path:
    raw_path = bundle.get("path")
    if type(raw_path) is not str:
        _fail("checkpoint bundle entry lacks path")
    bundle_path = Path(raw_path)
    compressed = bundle_path / NATIVE_RUNTIME_CHECKPOINT_PROJECTION_GZIP_FILENAME
    path = (
        compressed
        if compressed.exists()
        else bundle_path / NATIVE_RUNTIME_CHECKPOINT_PROJECTION_FILENAME
    )
    if not path.is_file():
        _fail(f"checkpoint sidecar is missing: {path}")
    return path


def _checkpoint_sidecars(
    batch_manifest: Mapping[str, Any],
    *,
    cell_id: str,
) -> dict[int, Path]:
    result = batch_manifest.get("result")
    catalog = result.get("checkpoint_catalog") if isinstance(result, Mapping) else None
    if not isinstance(catalog, Mapping):
        _fail("batch manifest lacks checkpoint catalog")
    sidecars: dict[int, Path] = {}
    for raw_generation, checkpoint in catalog.items():
        if not isinstance(checkpoint, Mapping):
            continue
        try:
            generation = int(raw_generation)
        except ValueError:
            continue
        bundles = checkpoint.get("cell_bundles")
        if not isinstance(bundles, list):
            _fail(f"G{generation} checkpoint lacks cell_bundles")
        matches = [
            bundle
            for bundle in bundles
            if isinstance(bundle, Mapping) and bundle.get("cell_id") == cell_id
        ]
        if len(matches) != 1:
            _fail(f"G{generation} checkpoint must contain exactly one bundle for {cell_id}")
        sidecars[generation] = _sidecar_path(matches[0])
    missing = sorted({0, *TECHNICAL_SEGMENTS} - set(sidecars))
    if missing:
        _fail(f"cell {cell_id} lacks required checkpoint sidecars: {missing}")
    return sidecars


def _cell_rows(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cells = plan.get("cells")
    if not isinstance(cells, list):
        _fail("execution plan lacks cells")
    rows = {
        str(cell["cell_id"]): cell
        for cell in cells
        if isinstance(cell, Mapping) and type(cell.get("cell_id")) is str
    }
    if len(rows) != 12:
        _fail(f"expected 12 logical cells, got {len(rows)}")
    return rows


def _segment_paths(sidecars: Mapping[int, Path], target: int) -> tuple[Path, ...]:
    return tuple(sidecars[generation] for generation in TECHNICAL_SEGMENTS if generation <= target)


def _reduce_cell(
    *,
    run_root: Path,
    batch_manifest_path: Path,
    plan: Mapping[str, Any],
    authority: Mapping[str, Any],
    source_commit: str,
    output_root: Path,
) -> dict[str, Any]:
    batch_manifest = _read_json(batch_manifest_path, name="batch manifest")
    cell_ids = batch_manifest.get("cell_ids")
    if not isinstance(cell_ids, list) or len(cell_ids) != 1 or type(cell_ids[0]) is not str:
        _fail("P12B terminal closure expects physical b1 batch manifests")
    cell_id = str(cell_ids[0])
    cell = _cell_rows(plan)[cell_id]
    dimensions = cell.get("dimensions")
    if not isinstance(dimensions, Mapping):
        _fail(f"cell {cell_id} lacks dimensions")
    seed = int(dimensions["ecology_seed"])
    profile = str(dimensions["founder_profile"])
    sidecars = _checkpoint_sidecars(batch_manifest, cell_id=cell_id)
    cell_output = output_root / "cells" / cell_id
    milestone_results: dict[str, Any] = {}
    for milestone in SCIENTIFIC_MILESTONES:
        result = oracle.reduce_compact_descendant_oracle(
            g0_path=sidecars[0],
            target_path=sidecars[milestone],
            segment_paths=_segment_paths(sidecars, milestone),
            target_milestone=milestone,
            authority=authority,
            source_commit=source_commit,
            campaign_id=str(plan["campaign_id"]),
            cell_id=cell_id,
            seed=seed,
            evidence_root=run_root,
        )
        written = oracle.write_oracle_outputs(result, cell_output / f"g{milestone:02d}")
        milestone_results[str(milestone)] = {
            "result_sha256": result["result_sha256"],
            "summary": result["summary"],
            "output_paths": written,
            "reducer_rows": result["reducer_rows"],
        }
    return {
        "cell_id": cell_id,
        "batch_manifest_path": str(batch_manifest_path),
        "profile": profile,
        "seed": seed,
        "checkpoint_sidecars": {
            str(generation): {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for generation, path in sorted(sidecars.items())
            if generation in {0, *TECHNICAL_SEGMENTS}
        },
        "milestones": milestone_results,
    }


def _g30_positive_by_profile(cell_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_identity: dict[str, dict[str, set[int]]] = {
        identity: {} for identity in POSITIVE_IDENTITY_TYPES
    }
    total_positive_cells: set[str] = set()
    for cell in cell_results:
        rows = cell["milestones"]["30"]["reducer_rows"]
        for identity in POSITIVE_IDENTITY_TYPES:
            positive = any(
                row.get("identity_type") == identity
                and int(row.get("positive_ordinary_birth_carrier_count", 0)) > 0
                for row in rows
            )
            if positive:
                profile = str(cell["profile"])
                by_identity[identity].setdefault(profile, set()).add(int(cell["seed"]))
                total_positive_cells.add(str(cell["cell_id"]))
    return {
        "by_identity_profile_seeds": {
            identity: {
                profile: sorted(seeds) for profile, seeds in sorted(profile_map.items())
            }
            for identity, profile_map in by_identity.items()
        },
        "total_positive_g30_cells": len(total_positive_cells),
    }


def _decision(cell_results: Sequence[Mapping[str, Any]], comparisons: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if len(cell_results) != 12:
        failures.append("primary_cells_not_12")
    for cell in cell_results:
        for milestone in SCIENTIFIC_MILESTONES:
            summary = cell["milestones"][str(milestone)]["summary"]
            if summary.get("evaluable") is not True:
                failures.append(f"{cell['cell_id']}:G{milestone}:oracle_not_evaluable")
    if comparisons.get("ok") is False:
        failures.append("comparison_integrity_failed")
    positives = _g30_positive_by_profile(cell_results)
    if failures:
        decision = "P12B-OB-STATE-PERSIST-B12-G30-CONFIRMATION-E"
        label = "INTEGRITY_FAILURE"
    else:
        by_identity = positives["by_identity_profile_seeds"]
        identity_passes = []
        for identity in POSITIVE_IDENTITY_TYPES:
            profile_counts = [len(seeds) for seeds in by_identity.get(identity, {}).values()]
            identity_passes.append(len(profile_counts) >= 2 and all(count >= 3 for count in profile_counts))
        if all(identity_passes):
            decision = "P12B-OB-STATE-PERSIST-B12-G30-CONFIRMATION-A"
            label = "STATE_PERSISTENCE_CONFIRMED"
        elif int(positives["total_positive_g30_cells"]) > 0:
            decision = "P12B-OB-STATE-PERSIST-B12-G30-CONFIRMATION-B"
            label = "MIXED_OR_PROFILE_CONDITIONAL"
        else:
            decision = "P12B-OB-STATE-PERSIST-B12-G30-CONFIRMATION-C"
            label = "STATE_PERSISTENCE_NOT_SUPPORTED"
    return {
        "decision": decision,
        "decision_label": label,
        "integrity_failures": failures,
        "g30_positive_summary": positives,
    }


def _comparison_integrity(
    *,
    run_root: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    comparisons_manifest = run_root / "comparisons_manifest.json"
    reporting_receipt = run_root / "reporting_parity_receipt.json"
    failures: list[str] = []
    manifest: dict[str, Any] | None = None
    if comparisons_manifest.exists():
        manifest = _read_json(comparisons_manifest, name="comparisons manifest")
        if manifest.get("status") != "completed":
            failures.append(f"comparisons_manifest_status:{manifest.get('status')}")
        counts = manifest.get("counts")
        if not isinstance(counts, Mapping):
            failures.append("comparisons_manifest_missing_counts")
        else:
            for status in ("failed", "interrupted", "pending", "running"):
                if int(counts.get(status, 0)) != 0:
                    failures.append(f"comparisons_manifest_{status}:{counts.get(status)}")
        compiled = plan.get("comparisons")
        compiled_branches = (
            compiled.get("branches") if isinstance(compiled, Mapping) else None
        )
        observed_branches = manifest.get("branches")
        if not isinstance(compiled_branches, list) or not isinstance(
            observed_branches, list
        ):
            failures.append("comparisons_manifest_missing_branches")
        else:
            expected = {
                str(row.get("branch_id")): row
                for row in compiled_branches
                if isinstance(row, Mapping)
            }
            observed = {
                str(row.get("branch_id")): row
                for row in observed_branches
                if isinstance(row, Mapping)
            }
            if set(expected) != set(observed):
                failures.append("comparison_branch_identity_mismatch")
            for branch_id, row in sorted(observed.items()):
                if row.get("status") != "completed":
                    failures.append(f"comparison_branch_not_completed:{branch_id}")
    else:
        failures.append("comparisons_manifest_missing")

    reporting: dict[str, Any] | None = None
    if reporting_receipt.exists():
        reporting = _read_json(reporting_receipt, name="reporting parity receipt")
        if reporting.get("qualification_evidence_status") != "reporting_parity_proven":
            failures.append(
                "reporting_parity_not_proven:"
                f"{reporting.get('qualification_evidence_status')}"
            )
        if reporting.get("status") != "supported":
            failures.append(f"reporting_parity_status:{reporting.get('status')}")
    else:
        failures.append("reporting_parity_receipt_missing")

    return {
        "ok": not failures,
        "failures": failures,
        "comparisons_manifest_path": str(comparisons_manifest)
        if comparisons_manifest.exists()
        else None,
        "comparisons_manifest_sha256": _sha256(comparisons_manifest)
        if comparisons_manifest.exists()
        else None,
        "comparisons_manifest_summary": None
        if manifest is None
        else {
            "status": manifest.get("status"),
            "counts": manifest.get("counts"),
            "branches": [
                {
                    "branch_id": row.get("branch_id"),
                    "kind": row.get("kind"),
                    "source_cell_id": row.get("source_cell_id"),
                    "status": row.get("status"),
                }
                for row in manifest.get("branches", [])
                if isinstance(row, Mapping)
            ],
        },
        "reporting_parity_receipt_path": str(reporting_receipt)
        if reporting_receipt.exists()
        else None,
        "reporting_parity_receipt_sha256": _sha256(reporting_receipt)
        if reporting_receipt.exists()
        else None,
        "reporting_parity_summary": None
        if reporting is None
        else {
            "qualification_evidence_status": reporting.get(
                "qualification_evidence_status"
            ),
            "status": reporting.get("status"),
            "evidence_mode": reporting.get("evidence_mode"),
            "comparison_first_difference": reporting.get("comparison_first_difference"),
        },
    }


def reduce_campaign(
    *,
    run_root: Path,
    authority_path: Path,
    source_commit: str,
    output_root: Path,
) -> dict[str, Any]:
    plan = _read_json(run_root / "execution_plan.json", name="execution plan")
    authority = _read_json(authority_path, name="authority binding")
    output_root.mkdir(parents=True, exist_ok=True)
    cell_results = [
        _reduce_cell(
            run_root=run_root,
            batch_manifest_path=path,
            plan=plan,
            authority=authority,
            source_commit=source_commit,
            output_root=output_root,
        )
        for path in _batch_manifest_paths(run_root)
    ]
    comparisons = _comparison_integrity(run_root=run_root, plan=plan)
    decision = _decision(cell_results, comparisons)
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(run_root),
        "campaign_id": plan.get("campaign_id"),
        "plan_sha256": plan.get("plan_sha256"),
        "source_commit": source_commit,
        "authority_path": str(authority_path),
        "authority_sha256": _sha256(authority_path),
        "scientific_milestones": list(SCIENTIFIC_MILESTONES),
        "technical_segments": list(TECHNICAL_SEGMENTS),
        "cell_results": cell_results,
        "comparisons": comparisons,
        "decision": decision,
    }
    result["result_sha256"] = canonical_sha256(result)
    (output_root / "p12b_compact_terminal_closure_reduction_v1.json").write_text(
        pretty_json(result), encoding="utf-8"
    )
    (output_root / "p12b_compact_terminal_closure_decision_v1.json").write_text(
        pretty_json({**decision, "result_sha256": result["result_sha256"]}),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = reduce_campaign(
        run_root=args.run_root.expanduser().resolve(),
        authority_path=args.authority.expanduser().resolve(),
        source_commit=args.source_commit,
        output_root=args.output_root.expanduser().resolve(),
    )
    print(
        pretty_json(
            {
                "ok": True,
                "decision": result["decision"],
                "result_sha256": result["result_sha256"],
                "output_root": str(args.output_root),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
