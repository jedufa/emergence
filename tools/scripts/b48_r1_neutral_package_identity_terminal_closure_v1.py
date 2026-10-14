#!/usr/bin/env python3
"""Reduce the Genome V2 B48-R1 neutral package-identity replication."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import canonical_sha256, pretty_json  # noqa: E402
from tools.scripts import b48_mutation_persistence_b12x4_terminal_closure_v1 as b48  # noqa: E402
from tools.scripts import p12b_compact_terminal_closure_v1 as p12b_terminal  # noqa: E402


SCHEMA_VERSION = "gv2_b48_r1_neutral_package_identity_reduction_v1"
CAMPAIGN_ID = "gv2-b48-r1-neutral-package-identity-b24g30"
EXPECTED_SEEDS = (707, 808, 909, 1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809)
EXPECTED_PROFILE = "neutral"
PRIMARY_IDENTITY_TYPE = "preregistered_package_identity"
SECONDARY_IDENTITY_TYPE = "allele_genotype_state_identity"
SCIENTIFIC_MILESTONE = 30


class B48R1ReductionError(RuntimeError):
    """B48-R1 evidence cannot be reduced to a terminal decision."""


def _fail(message: str) -> None:
    raise B48R1ReductionError(message)


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise B48R1ReductionError(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{name} must be a JSON object: {path}")
    return value


def _write_json_gz(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(pretty_json(value))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty_json(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan_cells(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cells = {
        str(cell["cell_id"]): cell
        for cell in plan.get("cells", [])
        if isinstance(cell, Mapping) and type(cell.get("cell_id")) is str
    }
    if len(cells) != 24:
        _fail(f"R1 expected 24 logical cells, got {len(cells)}")
    return cells


def _batch_rows(run_root: Path) -> list[tuple[Path, Mapping[str, Any], list[str]]]:
    rows: list[tuple[Path, Mapping[str, Any], list[str]]] = []
    for path in p12b_terminal._batch_manifest_paths(run_root):
        manifest = _read_json(path, name="batch manifest")
        cell_ids = manifest.get("cell_ids")
        if manifest.get("status") != "completed":
            _fail(f"batch is not completed: {path}")
        if (
            not isinstance(cell_ids, list)
            or not cell_ids
            or len(cell_ids) > 12
            or not all(type(cell_id) is str for cell_id in cell_ids)
        ):
            _fail(f"R1 expects nonempty physical batches no larger than B12: {path}")
        rows.append((path, manifest, list(cell_ids)))
    if not rows:
        _fail("R1 run root has no completed batch manifests")
    return rows


def _validate_matrix(cell_results: Sequence[Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    if len(cell_results) != 24:
        failures.append(f"primary_cells_not_24:{len(cell_results)}")
    if len({str(cell["cell_id"]) for cell in cell_results}) != len(cell_results):
        failures.append("duplicate_cell_ids")
    regimes = sorted({str(cell["mutation_regime"]) for cell in cell_results})
    profiles = sorted({str(cell["profile"]) for cell in cell_results})
    seeds = sorted({int(cell["seed"]) for cell in cell_results})
    if regimes != ["low_q10", "off"]:
        failures.append(f"mutation_regimes_not_off_and_low:{regimes}")
    if profiles != [EXPECTED_PROFILE]:
        failures.append(f"profile_set_mismatch:{profiles}")
    if seeds != list(EXPECTED_SEEDS):
        failures.append(f"seed_set_mismatch:{seeds}")
    for seed in EXPECTED_SEEDS:
        seed_cells = [
            cell for cell in cell_results if int(cell["seed"]) == seed
        ]
        if len(seed_cells) != 2:
            failures.append(f"seed_pair_count_mismatch:{seed}:{len(seed_cells)}")
    for cell in cell_results:
        if cell["oracle"]["summary"].get("evaluable") is not True:
            failures.append(f"{cell['cell_id']}:oracle_not_evaluable")
    return failures


def _classify_candidates(
    cell_results: Sequence[Mapping[str, Any]],
    *,
    identity_type: str,
) -> list[dict[str, Any]]:
    off_index: dict[int, Mapping[str, Any]] = {}
    for cell in cell_results:
        if cell["mutation_regime"] != "off":
            continue
        seed = int(cell["seed"])
        if seed in off_index:
            _fail(f"duplicate off-control cell for seed {seed}")
        off_index[seed] = cell
    classified: list[dict[str, Any]] = []
    for cell in cell_results:
        if cell["mutation_regime"] != "low_q10":
            continue
        pair = off_index.get(int(cell["seed"]))
        if pair is None:
            _fail(f"missing paired off-control for {cell['cell_id']}")
        for row in cell["candidate_rows"]:
            if row["identity_type"] != identity_type:
                continue
            state_hash = str(row["state_hash"])
            off_event = set(pair["event_state_hashes"][identity_type])
            off_terminal = set(pair["terminal_state_hashes"][identity_type])
            mutation_unique = (
                row["novel_vs_g0_founders"]
                and state_hash not in off_event
                and state_hash not in off_terminal
            )
            positive = (
                mutation_unique
                and int(row["terminal_descendant_carrier_count"]) > 0
            )
            classified.append(
                {
                    **row,
                    "paired_off_cell_id": pair["cell_id"],
                    "absent_from_paired_off_events": state_hash not in off_event,
                    "absent_from_paired_off_terminal": state_hash not in off_terminal,
                    "mutation_unique_against_pair": mutation_unique,
                    "positive_mutation_persistence": positive,
                }
            )
    return classified


def _decision(
    *,
    cell_results: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Any],
    primary_rows: Sequence[Mapping[str, Any]],
    secondary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures = _validate_matrix(cell_results)
    if comparisons.get("ok") is not True:
        failures.append("comparison_integrity_failed")
    primary_positives = [
        row for row in primary_rows if row.get("positive_mutation_persistence") is True
    ]
    secondary_positives = [
        row for row in secondary_rows if row.get("positive_mutation_persistence") is True
    ]
    primary_positive_seeds = sorted({int(row["seed"]) for row in primary_positives})
    if failures:
        decision = "GV2-B48-R1-E"
        label = "INTEGRITY_FAILURE"
    elif len(primary_positive_seeds) > 1:
        decision = "GV2-B48-R1-A"
        label = "HELD_OUT_SIGNAL_REPLICATED"
    elif primary_positives:
        decision = "GV2-B48-R1-B"
        label = "RARE_OR_CONDITIONAL_SIGNAL_REPLICATED"
    else:
        decision = "GV2-B48-R1-C"
        label = "HELD_OUT_SIGNAL_NOT_REPLICATED"
    return {
        "decision": decision,
        "decision_label": label,
        "integrity_failures": failures,
        "primary_identity_type": PRIMARY_IDENTITY_TYPE,
        "secondary_identity_type": SECONDARY_IDENTITY_TYPE,
        "primary_classified_low_mutation_event_identity_rows": len(primary_rows),
        "primary_positive_mutation_persistence_event_count": len(primary_positives),
        "primary_positive_seeds": primary_positive_seeds,
        "primary_positives_by_seed": {
            str(seed): sum(1 for row in primary_positives if int(row["seed"]) == seed)
            for seed in EXPECTED_SEEDS
        },
        "secondary_classified_low_mutation_event_identity_rows": len(secondary_rows),
        "secondary_positive_mutation_persistence_event_count": len(secondary_positives),
        "secondary_positive_seeds": sorted({int(row["seed"]) for row in secondary_positives}),
        "decision_rule": (
            "E if integrity failure; A if independently auditable primary positives "
            "occur across more than one held-out seed; B if valid but one-seed rare; "
            "C if clean no primary support. Independent audit must confirm before "
            "the decision is terminal."
        ),
    }


def _cell_summary(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": cell["cell_id"],
        "batch_manifest_path": cell["batch_manifest_path"],
        "profile": cell["profile"],
        "seed": cell["seed"],
        "mutation_regime": cell["mutation_regime"],
        "mutation_rate_q10": cell["mutation_rate_q10"],
        "mutation_step_q10": cell["mutation_step_q10"],
        "checkpoint_sidecar_count": len(cell["checkpoint_sidecars"]),
        "oracle": cell["oracle"],
        "candidate_row_count": len(cell["candidate_rows"]),
        "primary_event_state_hash_count": len(cell["event_state_hashes"][PRIMARY_IDENTITY_TYPE]),
        "primary_terminal_state_hash_count": len(cell["terminal_state_hashes"][PRIMARY_IDENTITY_TYPE]),
    }


def _jsonable_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(cell),
        "event_state_hashes": {
            identity_type: sorted(values)
            for identity_type, values in cell["event_state_hashes"].items()
        },
        "terminal_state_hashes": {
            identity_type: sorted(values)
            for identity_type, values in cell["terminal_state_hashes"].items()
        },
    }


def reduce_r1(
    *,
    run_roots: Sequence[Path],
    authority_path: Path,
    source_commit: str,
    output_root: Path,
) -> dict[str, Any]:
    resolved_run_roots = [path.expanduser().resolve() for path in run_roots]
    if not resolved_run_roots:
        _fail("R1 reduction requires at least one run root")
    authority_path = authority_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    authority = _read_json(authority_path, name="authority binding")
    all_plan_cells: dict[str, Mapping[str, Any]] = {}
    all_batches: list[tuple[Path, Mapping[str, Any], list[str]]] = []
    seen: set[str] = set()
    cell_results: list[dict[str, Any]] = []
    partition_comparisons = []
    for run_root in resolved_run_roots:
        plan = _read_json(run_root / "execution_plan.json", name="execution plan")
        if plan.get("campaign_id") != CAMPAIGN_ID:
            _fail(f"unexpected campaign_id: {plan.get('campaign_id')!r}")
        plan_cells = {
            str(cell["cell_id"]): cell
            for cell in plan.get("cells", [])
            if isinstance(cell, Mapping) and type(cell.get("cell_id")) is str
        }
        if not plan_cells:
            _fail(f"R1 partition has no plan cells: {run_root}")
        overlap = set(plan_cells).intersection(all_plan_cells)
        if overlap:
            _fail(f"duplicate plan cells across R1 partitions: {sorted(overlap)}")
        all_plan_cells.update(plan_cells)
        batches = _batch_rows(run_root)
        all_batches.extend(batches)
        for batch_manifest_path, batch_manifest, cell_ids in batches:
            for cell_id in cell_ids:
                if cell_id in seen:
                    _fail(f"duplicate physical cell id in batches: {cell_id}")
                if cell_id not in plan_cells:
                    _fail(f"batch cell id absent from partition plan: {cell_id}")
                seen.add(cell_id)
                cell_results.append(
                    b48._reduce_cell(
                        run_root=run_root,
                        batch_manifest=batch_manifest,
                        batch_manifest_path=batch_manifest_path,
                        plan=plan,
                        authority=authority,
                        source_commit=source_commit,
                        output_root=output_root,
                        cell_id=cell_id,
                        expected_plan_cells=len(plan_cells),
                    )
                )
        partition_comparisons.append(
            {
                "run_root": str(run_root),
                "plan_sha256": plan.get("plan_sha256"),
                "comparisons": p12b_terminal._comparison_integrity(
                    run_root=run_root, plan=plan
                ),
            }
        )
    if len(all_plan_cells) != 24:
        _fail(f"R1 expected 24 logical cells across partitions, got {len(all_plan_cells)}")
    if set(seen) != set(all_plan_cells):
        _fail("physical batches do not exactly cover plan cells")
    comparison_failures = [
        f"{row['run_root']}:{failure}"
        for row in partition_comparisons
        for failure in row["comparisons"].get("failures", [])
    ]
    comparisons = {
        "ok": not comparison_failures
        and all(row["comparisons"].get("ok") is True for row in partition_comparisons),
        "failures": comparison_failures,
        "partitions": partition_comparisons,
    }
    primary_rows = _classify_candidates(cell_results, identity_type=PRIMARY_IDENTITY_TYPE)
    secondary_rows = _classify_candidates(cell_results, identity_type=SECONDARY_IDENTITY_TYPE)
    decision = _decision(
        cell_results=cell_results,
        comparisons=comparisons,
        primary_rows=primary_rows,
        secondary_rows=secondary_rows,
    )
    primary_rows_payload = {"rows": primary_rows}
    secondary_rows_payload = {"rows": secondary_rows}
    primary_rows_sha256 = canonical_sha256(primary_rows_payload)
    secondary_rows_sha256 = canonical_sha256(secondary_rows_payload)
    reduction_path = output_root / "gv2_b48_r1_neutral_package_identity_reduction_v1.json.gz"
    primary_rows_path = output_root / "gv2_b48_r1_primary_classified_rows_v1.json.gz"
    secondary_rows_path = output_root / "gv2_b48_r1_secondary_classified_rows_v1.json.gz"
    decision_path = output_root / "gv2_b48_r1_reduction_decision_v1.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "run_roots": [str(path) for path in resolved_run_roots],
        "partition_plan_sha256s": [
            str(row["plan_sha256"]) for row in partition_comparisons
        ],
        "source_commit": source_commit,
        "authority_path": str(authority_path),
        "authority_sha256": _sha256(authority_path),
        "scientific_milestone": SCIENTIFIC_MILESTONE,
        "expected_seeds": list(EXPECTED_SEEDS),
        "expected_profile": EXPECTED_PROFILE,
        "primary_identity_type": PRIMARY_IDENTITY_TYPE,
        "secondary_identity_type": SECONDARY_IDENTITY_TYPE,
        "physical_batches": [
            {
                "batch_manifest_path": str(path),
                "batch_manifest_sha256": _sha256(path),
                "cell_ids": cell_ids,
            }
            for path, _manifest, cell_ids in all_batches
        ],
        "cell_summaries": [_cell_summary(cell) for cell in cell_results],
        "cell_compact_payload": [_jsonable_cell(cell) for cell in cell_results],
        "primary_classified_rows_path": str(primary_rows_path),
        "primary_classified_rows_sha256": primary_rows_sha256,
        "secondary_classified_rows_path": str(secondary_rows_path),
        "secondary_classified_rows_sha256": secondary_rows_sha256,
        "comparisons": comparisons,
        "decision": decision,
        "claim_boundary": (
            "Neutral held-out package-identity replication only; no B72, ecology, "
            "adaptation, cognition, intelligence, emergence, open-ended-evolution, "
            "profile-general, or allele-genotype-general claim."
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    _write_json_gz(reduction_path, result)
    _write_json_gz(primary_rows_path, primary_rows_payload)
    _write_json_gz(secondary_rows_path, secondary_rows_payload)
    _write_json(decision_path, {**decision, "result_sha256": result["result_sha256"]})
    return {
        **result,
        "paths": {
            "reduction": str(reduction_path),
            "primary_rows": str(primary_rows_path),
            "secondary_rows": str(secondary_rows_path),
            "decision": str(decision_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = reduce_r1(
        run_roots=[path.expanduser().resolve() for path in args.run_root],
        authority_path=args.authority,
        source_commit=args.source_commit,
        output_root=args.output_root,
    )
    print(
        pretty_json(
            {
                "ok": True,
                "decision": result["decision"],
                "result_sha256": result["result_sha256"],
                "paths": result["paths"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
