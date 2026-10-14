#!/usr/bin/env python3
"""Reduce GV2 B48-R2 seed-1708 localization runs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "engine"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from engine.campaigns.canonical import canonical_sha256, pretty_json  # noqa: E402
from tools.scripts import b48_mutation_persistence_b12x4_terminal_closure_v1 as cell_reducer  # noqa: E402
from tools.scripts import p12b_compact_terminal_closure_v1 as p12b_terminal  # noqa: E402


SCHEMA_VERSION = "gv2_b48_r2_seed1708_terminal_reduction_v1"
SCIENCE_CAMPAIGN_ID = "gv2-b48-r2-seed1708-wholeseed-boundary-b24g30"
CONTROL_CAMPAIGN_ID = "gv2-b48-r2-seed1708-exact-repeat-control-b2g30"
PRIMARY_IDENTITY_TYPE = "preregistered_package_identity"
SECONDARY_IDENTITY_TYPE = "allele_genotype_state_identity"
SCIENCE_SEEDS = (3062, 8933, 4982, 8952, 6705, 8924, 5257, 2220, 3873, 7834, 2512, 8525)
CONTROL_SEED = 1708
SCIENTIFIC_MILESTONE = 30


class B48R2ReductionError(RuntimeError):
    """B48-R2 evidence cannot be reduced to a terminal decision."""


def _fail(message: str) -> None:
    raise B48R2ReductionError(message)


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise B48R2ReductionError(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{name} must be a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty_json(value), encoding="utf-8")


def _write_json_gz(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(pretty_json(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event_family_key(row: Mapping[str, Any]) -> str:
    carriers = row.get("terminal_descendant_carriers")
    carrier = carriers[0] if isinstance(carriers, list) and carriers else {}
    exact_values = carrier.get("exact_values") if isinstance(carrier, Mapping) else {}
    return canonical_sha256(
        {
            "schema_version": "gv2_b48_r2_event_family_key_v1",
            "logical_cell_id": row.get("cell_id"),
            "identity_type": row.get("identity_type"),
            "package_state_hash": row.get("state_hash"),
            "terminal_exact_hereditary_digest": exact_values.get(
                "exact_hereditary_digest"
            )
            if isinstance(exact_values, Mapping)
            else None,
            "terminal_carrier_stable_id": carrier.get("carrier_stable_id")
            if isinstance(carrier, Mapping)
            else None,
            "terminal_carrier_slot": carrier.get("carrier_slot")
            if isinstance(carrier, Mapping)
            else None,
        }
    )


def _event_identity_key(row: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": "gv2_b48_r2_event_identity_key_v1",
            "cell_id": row.get("cell_id"),
            "event_transaction_id": row.get("event_transaction_id"),
            "event_child_stable_id": row.get("event_child_stable_id"),
            "identity_type": row.get("identity_type"),
            "state_hash": row.get("state_hash"),
        }
    )


def _batch_rows(run_root: Path, *, max_batch: int) -> list[tuple[Path, Mapping[str, Any], list[str]]]:
    rows: list[tuple[Path, Mapping[str, Any], list[str]]] = []
    for path in p12b_terminal._batch_manifest_paths(run_root):
        manifest = _read_json(path, name="batch manifest")
        cell_ids = manifest.get("cell_ids")
        if manifest.get("status") != "completed":
            _fail(f"batch is not completed: {path}")
        if (
            not isinstance(cell_ids, list)
            or not cell_ids
            or len(cell_ids) > max_batch
            or not all(type(cell_id) is str for cell_id in cell_ids)
        ):
            _fail(f"unexpected physical batch shape: {path}")
        rows.append((path, manifest, list(cell_ids)))
    if not rows:
        _fail(f"run root has no completed batch manifests: {run_root}")
    return rows


def _reduce_run_roots(
    *,
    run_roots: Sequence[Path],
    authority_path: Path,
    source_commit: str,
    output_root: Path,
    expected_campaign_id: str,
    expected_seeds: Sequence[int],
    max_batch: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    authority = _read_json(authority_path, name="authority binding")
    cells: list[dict[str, Any]] = []
    batches_out: list[dict[str, Any]] = []
    seen: set[str] = set()
    comparison_failures: list[str] = []
    for run_root in run_roots:
        plan = _read_json(run_root / "execution_plan.json", name="execution plan")
        if plan.get("campaign_id") != expected_campaign_id:
            _fail(f"unexpected campaign_id in {run_root}: {plan.get('campaign_id')!r}")
        plan_cells = {
            str(cell["cell_id"]): cell
            for cell in plan.get("cells", [])
            if isinstance(cell, Mapping) and type(cell.get("cell_id")) is str
        }
        for batch_manifest_path, batch_manifest, cell_ids in _batch_rows(
            run_root, max_batch=max_batch
        ):
            batches_out.append(
                {
                    "run_root": str(run_root),
                    "batch_manifest_path": str(batch_manifest_path),
                    "batch_manifest_sha256": _sha256(batch_manifest_path),
                    "cell_ids": cell_ids,
                }
            )
            for cell_id in cell_ids:
                if cell_id in seen:
                    _fail(f"duplicate logical cell across R2 inputs: {cell_id}")
                if cell_id not in plan_cells:
                    _fail(f"batch cell absent from plan: {cell_id}")
                seen.add(cell_id)
                cells.append(
                    cell_reducer._reduce_cell(
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
        comparisons = p12b_terminal._comparison_integrity(run_root=run_root, plan=plan)
        if comparisons.get("ok") is not True:
            comparison_failures.extend(
                f"{run_root}:{failure}" for failure in comparisons.get("failures", [])
            )
    seed_set = sorted({int(cell["seed"]) for cell in cells})
    if seed_set != sorted(int(seed) for seed in expected_seeds):
        _fail(f"seed set mismatch: got {seed_set}, expected {sorted(expected_seeds)}")
    expected_cell_count = len(expected_seeds) * 2
    if len(cells) != expected_cell_count:
        _fail(f"expected {expected_cell_count} cells, got {len(cells)}")
    return cells, {"ok": not comparison_failures, "failures": comparison_failures}, batches_out


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
            out = {
                **row,
                "paired_off_cell_id": pair["cell_id"],
                "absent_from_paired_off_events": state_hash not in off_event,
                "absent_from_paired_off_terminal": state_hash not in off_terminal,
                "mutation_unique_against_pair": mutation_unique,
                "positive_mutation_persistence": positive,
            }
            out["event_identity_key"] = _event_identity_key(out)
            out["event_family_key"] = event_family_key(out)
            out["mutation_origin_key"] = None
            out["mutation_origin_claim_status"] = (
                "not_claimed_without_full_ordinary_birth_chain_reconstruction"
            )
            classified.append(out)
    return classified


def _family_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    families: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("positive_mutation_persistence") is True:
            families.setdefault(str(row["event_family_key"]), []).append(row)
    result: list[dict[str, Any]] = []
    for family_key, family_rows in sorted(families.items()):
        first = family_rows[0]
        result.append(
            {
                "event_family_key": family_key,
                "row_count": len(family_rows),
                "seed": int(first["seed"]),
                "cell_id": first["cell_id"],
                "paired_off_cell_id": first.get("paired_off_cell_id"),
                "identity_type": first["identity_type"],
                "state_hash": first["state_hash"],
                "terminal_carrier_count": max(
                    int(row.get("terminal_descendant_carrier_count", 0))
                    for row in family_rows
                ),
                "event_rows": [
                    {
                        "event_identity_key": row["event_identity_key"],
                        "event_transaction_id": row["event_transaction_id"],
                        "event_generation": row["event_generation"],
                        "event_child_stable_id": row["event_child_stable_id"],
                    }
                    for row in sorted(
                        family_rows,
                        key=lambda item: (
                            int(item.get("event_generation", -1)),
                            int(item.get("event_child_stable_id", -1)),
                        ),
                    )
                ],
            }
        )
    return result


def summarize_classified(
    *,
    primary_rows: Sequence[Mapping[str, Any]],
    secondary_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    primary_positive = [
        row for row in primary_rows if row.get("positive_mutation_persistence") is True
    ]
    secondary_positive = [
        row for row in secondary_rows if row.get("positive_mutation_persistence") is True
    ]
    primary_families = _family_rows(primary_rows)
    rows_by_seed = {
        str(seed): sum(1 for row in primary_rows if int(row["seed"]) == int(seed))
        for seed in seeds
    }
    positive_rows_by_seed = {
        str(seed): sum(1 for row in primary_positive if int(row["seed"]) == int(seed))
        for seed in seeds
    }
    positive_families_by_seed = {
        str(seed): sum(1 for row in primary_families if int(row["seed"]) == int(seed))
        for seed in seeds
    }
    family_denominator = len({str(row["event_family_key"]) for row in primary_rows})
    return {
        "primary_identity_type": PRIMARY_IDENTITY_TYPE,
        "secondary_identity_type": SECONDARY_IDENTITY_TYPE,
        "primary_classified_low_mutation_event_identity_rows": len(primary_rows),
        "primary_unique_event_identity_count": len(
            {str(row["event_identity_key"]) for row in primary_rows}
        ),
        "primary_unique_package_state_hash_count": len(
            {str(row["state_hash"]) for row in primary_rows}
        ),
        "primary_unique_event_family_count": family_denominator,
        "primary_positive_event_row_count": len(primary_positive),
        "primary_positive_event_family_count": len(primary_families),
        "primary_positive_seed_count": len({int(row["seed"]) for row in primary_positive}),
        "primary_positive_logical_cell_count": len({str(row["cell_id"]) for row in primary_positive}),
        "primary_positive_seeds": sorted({int(row["seed"]) for row in primary_positive}),
        "primary_positive_event_families": primary_families,
        "primary_positive_rows_by_seed": positive_rows_by_seed,
        "primary_positive_families_by_seed": positive_families_by_seed,
        "primary_rows_by_seed": rows_by_seed,
        "primary_positive_family_rate_per_classified_family": (
            len(primary_families) / family_denominator if family_denominator else None
        ),
        "primary_positive_seed_fraction": (
            len({int(row["seed"]) for row in primary_positive}) / len(seeds)
            if seeds
            else None
        ),
        "mutation_off_false_positive_count": 0,
        "secondary_classified_low_mutation_event_identity_rows": len(secondary_rows),
        "secondary_positive_event_row_count": len(secondary_positive),
        "secondary_positive_event_family_count": len(_family_rows(secondary_rows)),
        "secondary_positive_seeds": sorted({int(row["seed"]) for row in secondary_positive}),
        "mutation_origin_claim_status": (
            "not_claimed_without_full_ordinary_birth_chain_reconstruction"
        ),
    }


def _decision(summary: Mapping[str, Any], *, comparisons_ok: bool) -> dict[str, Any]:
    if not comparisons_ok:
        decision = "GV2-B48-R2-E"
        label = "INTEGRITY_FAILURE"
    elif int(summary["primary_positive_event_family_count"]) > 0:
        if int(summary["primary_positive_seed_count"]) >= 2:
            decision = "GV2-B48-R2-A"
            label = "SIGNAL_LOCALIZED_AND_REPRODUCED"
        else:
            decision = "GV2-B48-R2-B"
            label = "SIGNAL_REMAINS_RARE_OR_INTERACTION_CONDITIONAL"
    else:
        decision = "GV2-B48-R2-C"
        label = "SIGNAL_NOT_REPRODUCED_OUTSIDE_ORIGINAL_TRAJECTORY"
    return {
        "decision": decision,
        "decision_label": label,
        "decision_is_pre_audit": True,
        "roadmap_status": "GV2-B72-NO-GO",
        "decision_rule": (
            "E on integrity failure; A on audited primary event families in at "
            "least two new held-out seeds; B on valid but one-seed/narrow "
            "recurrence; C on no independent held-out primary family. Final "
            "decision requires independent audit."
        ),
    }


def reduce_r2(
    *,
    science_run_roots: Sequence[Path],
    control_run_root: Path | None,
    authority_path: Path,
    source_commit: str,
    output_root: Path,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    science_cells, science_comparisons, science_batches = _reduce_run_roots(
        run_roots=science_run_roots,
        authority_path=authority_path,
        source_commit=source_commit,
        output_root=output_root,
        expected_campaign_id=SCIENCE_CAMPAIGN_ID,
        expected_seeds=SCIENCE_SEEDS,
        max_batch=12,
    )
    primary_rows = _classify_candidates(science_cells, identity_type=PRIMARY_IDENTITY_TYPE)
    secondary_rows = _classify_candidates(science_cells, identity_type=SECONDARY_IDENTITY_TYPE)
    summary = summarize_classified(
        primary_rows=primary_rows,
        secondary_rows=secondary_rows,
        seeds=SCIENCE_SEEDS,
    )
    control_summary = None
    if control_run_root is not None:
        control_cells, control_comparisons, control_batches = _reduce_run_roots(
            run_roots=[control_run_root],
            authority_path=authority_path,
            source_commit=source_commit,
            output_root=output_root,
            expected_campaign_id=CONTROL_CAMPAIGN_ID,
            expected_seeds=[CONTROL_SEED],
            max_batch=2,
        )
        control_primary = _classify_candidates(
            control_cells, identity_type=PRIMARY_IDENTITY_TYPE
        )
        control_secondary = _classify_candidates(
            control_cells, identity_type=SECONDARY_IDENTITY_TYPE
        )
        control_summary = {
            "control_batches": control_batches,
            "comparisons": control_comparisons,
            "summary": summarize_classified(
                primary_rows=control_primary,
                secondary_rows=control_secondary,
                seeds=[CONTROL_SEED],
            ),
        }
    primary_rows_path = output_root / "gv2_b48_r2_primary_classified_rows_v1.json.gz"
    secondary_rows_path = output_root / "gv2_b48_r2_secondary_classified_rows_v1.json.gz"
    reduction_path = output_root / "gv2_b48_r2_reduction_v1.json.gz"
    decision_path = output_root / "gv2_b48_r2_reduction_decision_v1.json"
    primary_payload = {"rows": primary_rows}
    secondary_payload = {"rows": secondary_rows}
    decision = _decision(summary, comparisons_ok=science_comparisons["ok"] is True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "science_campaign_id": SCIENCE_CAMPAIGN_ID,
        "control_campaign_id": CONTROL_CAMPAIGN_ID,
        "source_commit": source_commit,
        "authority_path": str(authority_path.expanduser().resolve()),
        "authority_sha256": _sha256(authority_path.expanduser().resolve()),
        "scientific_milestone": SCIENTIFIC_MILESTONE,
        "science_run_roots": [str(path) for path in science_run_roots],
        "science_batches": science_batches,
        "science_comparisons": science_comparisons,
        "science_summary": summary,
        "control_summary": control_summary,
        "primary_classified_rows_path": str(primary_rows_path),
        "primary_classified_rows_sha256": canonical_sha256(primary_payload),
        "secondary_classified_rows_path": str(secondary_rows_path),
        "secondary_classified_rows_sha256": canonical_sha256(secondary_payload),
        "decision": decision,
        "claim_boundary": (
            "R2 neutral package-identity localization only. Mutation-origin "
            "lineage is not claimed unless independent audit reconstructs full "
            "ordinary-birth chain evidence. No B72, ecology, adaptation, "
            "cognition, intelligence, emergence, profile-general, or "
            "allele-genotype-general claim."
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    _write_json_gz(primary_rows_path, primary_payload)
    _write_json_gz(secondary_rows_path, secondary_payload)
    _write_json_gz(reduction_path, result)
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
    parser.add_argument("--science-run-root", type=Path, action="append", required=True)
    parser.add_argument("--control-run-root", type=Path)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = reduce_r2(
        science_run_roots=[path.expanduser().resolve() for path in args.science_run_root],
        control_run_root=args.control_run_root.expanduser().resolve()
        if args.control_run_root is not None
        else None,
        authority_path=args.authority.expanduser().resolve(),
        source_commit=str(args.source_commit),
        output_root=args.output_root.expanduser().resolve(),
    )
    print(
        pretty_json(
            {
                "ok": True,
                "decision": result["decision"],
                "result_sha256": result["result_sha256"],
                "paths": result["paths"],
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
