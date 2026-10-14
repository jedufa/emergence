#!/usr/bin/env python3
"""Reduce the Genome V2 B48 mutation-persistence B12x4 fallback at G30."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns import p12a_gv2_persistence_reducer as p12a  # noqa: E402
from engine.campaigns import p12b_compact_descendant_oracle as oracle  # noqa: E402
from engine.campaigns.canonical import canonical_sha256, pretty_json  # noqa: E402
from tools.scripts import p12b_compact_terminal_closure_v1 as p12b_terminal  # noqa: E402
from tools.scripts import p12b_mutation_persistence_terminal_closure_v1 as p12b_mutation  # noqa: E402


SCHEMA_VERSION = "gv2_b48_mutation_persistence_b12x4_reduction_v1"
SCIENTIFIC_MILESTONE = 30
TECHNICAL_SEGMENTS = (5, 10, 15, 20, 25, 30)
POSITIVE_IDENTITY_TYPES = p12b_mutation.POSITIVE_IDENTITY_TYPES
EXPECTED_PROFILES = (
    "neutral",
    "recomb72_gv2_v1.single_locus_extreme_pair.homozygous_aa_control",
    "recomb72_gv2_v1.single_locus_extreme_pair.balanced_homozygous_aa_bb",
    "recomb72_gv2_v1.single_locus_extreme_pair.all_heterozygous_ab",
)


class B48MutationPersistenceError(RuntimeError):
    """B48 B12x4 evidence cannot be reduced to a terminal decision."""


def _fail(message: str) -> None:
    raise B48MutationPersistenceError(message)


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise B48MutationPersistenceError(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{name} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_gz(path: Path, value: Mapping[str, Any]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(pretty_json(value))


def _read_json_gz(path: Path, *, name: str) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise B48MutationPersistenceError(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{name} must be a JSON object: {path}")
    return value


def _cell_rows(
    plan: Mapping[str, Any],
    *,
    expected_cells: int = 12,
) -> dict[str, Mapping[str, Any]]:
    cells = plan.get("cells")
    if not isinstance(cells, list):
        _fail("execution plan lacks cells")
    rows = {
        str(cell["cell_id"]): cell
        for cell in cells
        if isinstance(cell, Mapping) and type(cell.get("cell_id")) is str
    }
    if len(rows) != expected_cells:
        _fail(
            "B48 partition expected "
            f"{expected_cells} logical cells, got {len(rows)}"
        )
    return rows


def _state_for_slot(
    projection: Mapping[str, Any],
    carrier: Mapping[str, int],
    *,
    identity_type: str,
    capacity: int,
) -> dict[str, Any]:
    try:
        return p12a._state_definition(
            projection,
            carrier,
            identity_type=identity_type,
            capacity=capacity,
        )
    except p12a.P12APersistenceReducerError as exc:
        raise B48MutationPersistenceError(str(exc)) from exc


def _state_hash(identity_type: str, fields: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": "p12b_mutation_persistence_state_identity_v1",
            "identity_type": identity_type,
            "matched_fields": fields,
        }
    )


def _terminal_carrier_index(
    target: Mapping[str, Any],
    *,
    capacity: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {
        identity_type: {} for identity_type in POSITIVE_IDENTITY_TYPES
    }
    for carrier in p12a._live_slots(target):
        for identity_type in POSITIVE_IDENTITY_TYPES:
            state = _state_for_slot(
                target, carrier, identity_type=identity_type, capacity=capacity
            )
            state_hash = _state_hash(identity_type, state["matched_fields"])
            result[identity_type].setdefault(state_hash, []).append(
                {
                    "carrier_stable_id": int(carrier["agent_id"]),
                    "carrier_slot": int(carrier["slot"]),
                    "lineage_depth": int(carrier["lineage_depth"]),
                    "lineage_founder_id": int(carrier["lineage_founder_id"]),
                    "selected_values": state["matched_fields"],
                    "exact_values": {
                        "exact_hereditary_digest": state["exact_hereditary_digest"]
                    },
                }
            )
    return result


def _reduce_cell(
    *,
    run_root: Path,
    batch_manifest: Mapping[str, Any],
    batch_manifest_path: Path,
    plan: Mapping[str, Any],
    authority: Mapping[str, Any],
    source_commit: str,
    output_root: Path,
    cell_id: str,
    expected_plan_cells: int = 12,
) -> dict[str, Any]:
    cell = _cell_rows(plan, expected_cells=expected_plan_cells)[cell_id]
    dimensions = cell.get("dimensions")
    parameters = cell.get("parameters")
    if not isinstance(dimensions, Mapping) or not isinstance(parameters, Mapping):
        _fail(f"cell {cell_id} lacks dimensions or parameters")
    seed = int(dimensions["ecology_seed"])
    profile = str(dimensions["founder_profile"])
    rate_q10 = int(parameters["genome_upgrade_v2.mutation_rate_q10"])
    step_q10 = int(parameters["genome_upgrade_v2.mutation_step_q10"])
    mutation_regime = "low_q10" if rate_q10 > 0 and step_q10 > 0 else "off"
    sidecars = p12b_terminal._checkpoint_sidecars(batch_manifest, cell_id=cell_id)
    validated_authority = oracle._validate_authority(authority)
    g0 = oracle._projection(sidecars[0], milestone=0)
    target = oracle._projection(sidecars[SCIENTIFIC_MILESTONE], milestone=30)
    capacity = len(p12a._tensor(g0, "alive_mask")[0])
    if len(p12a._tensor(target, "alive_mask")[0]) != capacity:
        _fail(f"G0/G{SCIENTIFIC_MILESTONE} tensor capacity mismatch for {cell_id}")
    event_rows: list[dict[str, Any]] = []
    segment_receipts: list[dict[str, Any]] = []
    segment_paths = tuple(sidecars[generation] for generation in TECHNICAL_SEGMENTS)
    for left, right, path in oracle._validate_segments(
        segment_paths=segment_paths,
        target_milestone=SCIENTIFIC_MILESTONE,
    ):
        segment = oracle._projection(path, milestone=right)
        rows, counts = oracle._segment_event_records(
            segment,
            authority=validated_authority,
            cell_id=cell_id,
            left=left,
            right=right,
        )
        event_rows.extend(rows)
        segment_receipts.append(
            {
                "left_generation": left,
                "right_generation": right,
                "path": str(path),
                "sha256": _sha256(path),
                "event_counts": counts,
                "compact_event_rows_observed": len(rows),
            }
        )
    founder_hashes: dict[str, set[str]] = {
        identity_type: set() for identity_type in POSITIVE_IDENTITY_TYPES
    }
    for founder in p12a._live_slots(g0):
        for identity_type in POSITIVE_IDENTITY_TYPES:
            state = _state_for_slot(
                g0, founder, identity_type=identity_type, capacity=capacity
            )
            founder_hashes[identity_type].add(
                _state_hash(identity_type, state["matched_fields"])
            )
    birth_by_child = {
        int(row["child_stable_id"]): row
        for row in event_rows
        if row.get("event_kind") == "ordinary_birth"
    }
    terminal_carriers = _terminal_carrier_index(target, capacity=capacity)
    event_identity_rows: list[dict[str, Any]] = []
    candidate_child_ids: dict[tuple[str, str], set[int]] = {}
    for event in event_rows:
        if event.get("event_kind") != "ordinary_birth":
            continue
        fields_by_identity = event["child_identity_fields"]
        child_id = int(event["child_stable_id"])
        for identity_type in POSITIVE_IDENTITY_TYPES:
            fields = fields_by_identity[identity_type]
            state_hash = _state_hash(identity_type, fields)
            event_identity_rows.append(
                {
                    "event": event,
                    "identity_type": identity_type,
                    "fields": fields,
                    "state_hash": state_hash,
                    "child_id": child_id,
                }
            )
            candidate_child_ids.setdefault((identity_type, state_hash), set()).add(
                child_id
            )
    descendant_counts: dict[tuple[str, str, int], int] = {}
    descendant_samples: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for identity_type in POSITIVE_IDENTITY_TYPES:
        for state_hash, carriers in terminal_carriers[identity_type].items():
            candidate_children = candidate_child_ids.get((identity_type, state_hash))
            if not candidate_children:
                continue
            for carrier in carriers:
                ancestors = p12b_mutation._ancestor_ids(
                    agent_id=int(carrier["carrier_stable_id"]),
                    birth_by_child=birth_by_child,
                )
                for child_id in ancestors.intersection(candidate_children):
                    key = (identity_type, state_hash, child_id)
                    descendant_counts[key] = descendant_counts.get(key, 0) + 1
                    samples = descendant_samples.setdefault(key, [])
                    if len(samples) < 8:
                        samples.append(dict(carrier))
    candidate_rows: list[dict[str, Any]] = []
    for row in event_identity_rows:
        event = row["event"]
        identity_type = str(row["identity_type"])
        state_hash = str(row["state_hash"])
        child_id = int(row["child_id"])
        key = (identity_type, state_hash, child_id)
        candidate_rows.append(
            {
                "cell_id": cell_id,
                "seed": seed,
                "founder_profile": profile,
                "mutation_regime": mutation_regime,
                "mutation_rate_q10": rate_q10,
                "mutation_step_q10": step_q10,
                "identity_type": identity_type,
                "state_hash": state_hash,
                "event_transaction_id": event["transaction_id"],
                "event_generation": event["generation"],
                "event_child_stable_id": event["child_stable_id"],
                "novel_vs_g0_founders": state_hash not in founder_hashes[identity_type],
                "terminal_descendant_carrier_count": descendant_counts.get(key, 0),
                "terminal_descendant_carriers": descendant_samples.get(key, []),
            }
        )
    ordinary_birth_event_count = sum(
        int(receipt["event_counts"]["ordinary_birth"]) for receipt in segment_receipts
    )
    turnover_refill_event_count = sum(
        int(receipt["event_counts"]["turnover_refill"]) for receipt in segment_receipts
    )
    other_event_count = sum(
        int(receipt["event_counts"]["other"]) for receipt in segment_receipts
    )
    oracle_summary = {
        "target_milestone": SCIENTIFIC_MILESTONE,
        "target_alive_denominator": len(p12a._live_slots(target)),
        "segment_count": len(segment_receipts),
        "source_event_row_count": len(event_rows),
        "ordinary_birth_event_row_count": ordinary_birth_event_count,
        "turnover_refill_event_row_count": turnover_refill_event_count,
        "other_event_row_count": other_event_count,
        "reducer_row_count": len(candidate_rows),
        "positive_ordinary_birth_carrier_rows": sum(
            1 for row in candidate_rows if row["terminal_descendant_carrier_count"] > 0
        ),
        "evaluable": True,
        "supported_identity_types": list(POSITIVE_IDENTITY_TYPES),
        "oracle_mode": "b48_b12x4_event_state_terminal_descendant_summary_v1",
    }
    oracle_identity = {
        "schema_version": "b48_b12x4_compact_cell_oracle_identity_v1",
        "source_commit": source_commit,
        "campaign_id": str(plan["campaign_id"]),
        "cell_id": cell_id,
        "seed": seed,
        "authority": validated_authority,
        "checkpoint_sidecar_sha256s": {
            str(generation): _sha256(path)
            for generation, path in sorted(sidecars.items())
            if generation in {0, *TECHNICAL_SEGMENTS}
        },
        "segment_receipts": segment_receipts,
        "summary": oracle_summary,
    }
    oracle_result_sha256 = canonical_sha256(oracle_identity)
    return {
        "cell_id": cell_id,
        "batch_manifest_path": str(batch_manifest_path),
        "profile": profile,
        "seed": seed,
        "mutation_regime": mutation_regime,
        "mutation_rate_q10": rate_q10,
        "mutation_step_q10": step_q10,
        "checkpoint_sidecars": {
            str(generation): {"path": str(path), "sha256": _sha256(path)}
            for generation, path in sorted(sidecars.items())
            if generation in {0, *TECHNICAL_SEGMENTS}
        },
        "oracle": {
            "result_sha256": oracle_result_sha256,
            "summary": oracle_summary,
            "segment_receipts": segment_receipts,
        },
        "event_state_hashes": p12b_mutation._event_state_hashes(event_rows),
        "terminal_state_hashes": {
            identity_type: set(rows) for identity_type, rows in terminal_carriers.items()
        },
        "candidate_rows": candidate_rows,
    }


def _reduce_partition(
    *,
    run_root: Path,
    authority: Mapping[str, Any],
    source_commit: str,
    output_root: Path,
) -> dict[str, Any]:
    plan = _read_json(run_root / "execution_plan.json", name="execution plan")
    batch_paths = p12b_terminal._batch_manifest_paths(run_root)
    if len(batch_paths) != 1:
        _fail(f"B48 B12x4 partition expects one physical batch: {run_root}")
    batch_manifest_path = batch_paths[0]
    batch_manifest = _read_json(batch_manifest_path, name="batch manifest")
    cell_ids = batch_manifest.get("cell_ids")
    if (
        not isinstance(cell_ids, list)
        or len(cell_ids) != 12
        or not all(type(cell_id) is str for cell_id in cell_ids)
    ):
        _fail("B48 B12x4 partition expects a physical b12 batch manifest")
    if batch_manifest.get("status") != "completed":
        _fail(f"partition batch is not completed: {batch_manifest_path}")
    plan_cells = _cell_rows(plan)
    if set(cell_ids) != set(plan_cells):
        _fail(f"partition cell ids do not match plan: {run_root}")
    cell_results = [
        _reduce_cell(
            run_root=run_root,
            batch_manifest=batch_manifest,
            batch_manifest_path=batch_manifest_path,
            plan=plan,
            authority=authority,
            source_commit=source_commit,
            output_root=output_root,
            cell_id=str(cell_id),
        )
        for cell_id in cell_ids
    ]
    return {
        "run_root": str(run_root),
        "campaign_id": plan.get("campaign_id"),
        "plan_sha256": plan.get("plan_sha256"),
        "cell_ids": list(cell_ids),
        "cell_results": cell_results,
        "comparisons": p12b_terminal._comparison_integrity(run_root=run_root, plan=plan),
    }


def _combined_comparisons(partitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    branches = 0
    for partition in partitions:
        comparisons = partition["comparisons"]
        if comparisons.get("ok") is not True:
            failures.extend(
                f"{partition['run_root']}:{failure}"
                for failure in comparisons.get("failures", [])
            )
        summary = comparisons.get("comparisons_manifest_summary")
        if isinstance(summary, Mapping):
            counts = summary.get("counts")
            if isinstance(counts, Mapping):
                branches += int(counts.get("completed", 0))
    return {"ok": not failures, "failures": failures, "completed_reporting_branches": branches}


def _classify_candidates(
    cell_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    off_index: dict[tuple[int, str], Mapping[str, Any]] = {}
    for cell in cell_results:
        if cell["mutation_regime"] != "off":
            continue
        key = (int(cell["seed"]), str(cell["profile"]))
        if key in off_index:
            _fail(f"duplicate off-control cell for seed/profile {key!r}")
        off_index[key] = cell
    classified: list[dict[str, Any]] = []
    for cell in cell_results:
        if cell["mutation_regime"] != "low_q10":
            continue
        pair = off_index.get((int(cell["seed"]), str(cell["profile"])))
        if pair is None:
            _fail(f"missing paired off-control for {cell['cell_id']}")
        for row in cell["candidate_rows"]:
            identity_type = str(row["identity_type"])
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
) -> dict[str, Any]:
    failures: list[str] = []
    if len(cell_results) != 48:
        failures.append(f"primary_cells_not_48:{len(cell_results)}")
    if len({str(cell["cell_id"]) for cell in cell_results}) != len(cell_results):
        failures.append("duplicate_cell_ids")
    regimes = sorted({str(cell["mutation_regime"]) for cell in cell_results})
    profiles = sorted({str(cell["profile"]) for cell in cell_results})
    seeds = sorted({int(cell["seed"]) for cell in cell_results})
    if regimes != ["low_q10", "off"]:
        failures.append(f"mutation_regimes_not_off_and_low:{regimes}")
    if profiles != sorted(EXPECTED_PROFILES):
        failures.append(f"profile_set_mismatch:{profiles}")
    if seeds != [101, 202, 303, 404, 505, 606]:
        failures.append(f"seed_set_mismatch:{seeds}")
    for cell in cell_results:
        if cell["oracle"]["summary"].get("evaluable") is not True:
            failures.append(f"{cell['cell_id']}:oracle_not_evaluable")
    if comparisons.get("ok") is not True:
        failures.append("comparison_integrity_failed")
    classified_rows = _classify_candidates(cell_results)
    positives = [
        row for row in classified_rows if row.get("positive_mutation_persistence") is True
    ]
    positive_profiles = sorted({str(row["founder_profile"]) for row in positives})
    positive_seeds = sorted({int(row["seed"]) for row in positives})
    positive_identity_types = sorted({str(row["identity_type"]) for row in positives})
    positives_by_profile = {
        profile: sum(1 for row in positives if row["founder_profile"] == profile)
        for profile in sorted(EXPECTED_PROFILES)
    }
    positives_by_identity = {
        identity_type: sum(1 for row in positives if row["identity_type"] == identity_type)
        for identity_type in POSITIVE_IDENTITY_TYPES
    }
    positives_by_profile_seed = {
        f"{profile}|{seed}": sum(
            1
            for row in positives
            if row["founder_profile"] == profile and int(row["seed"]) == seed
        )
        for profile in sorted(EXPECTED_PROFILES)
        for seed in seeds
    }
    if failures:
        decision = "GV2-B48-PROGRAMME-E"
        label = "INTEGRITY_FAILURE"
    elif sorted(positive_profiles) == sorted(EXPECTED_PROFILES) and sorted(
        positive_identity_types
    ) == sorted(POSITIVE_IDENTITY_TYPES):
        decision = "GV2-B48-PROGRAMME-A"
        label = "B48_SCIENTIFIC_PATH_SUPPORTED"
    elif positives:
        decision = "GV2-B48-PROGRAMME-B"
        label = "MIXED_OR_CONDITIONAL_SUPPORT"
    else:
        decision = "GV2-B48-PROGRAMME-C"
        label = "PRIMARY_HYPOTHESIS_NOT_SUPPORTED"
    return {
        "decision": decision,
        "decision_label": label,
        "integrity_failures": failures,
        "positive_mutation_persistence_event_count": len(positives),
        "positive_profiles": positive_profiles,
        "positive_seeds": positive_seeds,
        "positive_identity_types": positive_identity_types,
        "positives_by_profile": positives_by_profile,
        "positives_by_identity": positives_by_identity,
        "positives_by_profile_seed": positives_by_profile_seed,
        "classified_low_mutation_event_identity_rows": len(classified_rows),
    }, classified_rows


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
        "event_state_hash_counts": {
            identity_type: len(values)
            for identity_type, values in cell["event_state_hashes"].items()
        },
        "terminal_state_hash_counts": {
            identity_type: len(values)
            for identity_type, values in cell["terminal_state_hashes"].items()
        },
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


def _single_cell_output_path(output_root: Path, cell_id: str) -> Path:
    return output_root / "cell_compact" / f"{cell_id}.json.gz"


def _run_single_cell_subprocess(
    *,
    run_root: Path,
    batch_manifest_path: Path,
    cell_id: str,
    authority_path: Path,
    source_commit: str,
    output_root: Path,
) -> Mapping[str, Any]:
    path = _single_cell_output_path(output_root, cell_id)
    if path.exists():
        return _read_json_gz(path, name=f"cell compact output {cell_id}")
    command = [
        sys.executable,
        "-m",
        "tools.scripts.b48_mutation_persistence_b12x4_terminal_closure_v1",
        "--single-cell-run-root",
        str(run_root),
        "--single-cell-batch-manifest",
        str(batch_manifest_path),
        "--single-cell-id",
        cell_id,
        "--authority",
        str(authority_path),
        "--source-commit",
        source_commit,
        "--output-root",
        str(output_root),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return _read_json_gz(path, name=f"cell compact output {cell_id}")


def reduce_single_cell(
    *,
    run_root: Path,
    batch_manifest_path: Path,
    cell_id: str,
    authority_path: Path,
    source_commit: str,
    output_root: Path,
) -> dict[str, Any]:
    plan = _read_json(run_root / "execution_plan.json", name="execution plan")
    authority = _read_json(authority_path, name="authority binding")
    batch_manifest = _read_json(batch_manifest_path, name="batch manifest")
    cell = _reduce_cell(
        run_root=run_root,
        batch_manifest=batch_manifest,
        batch_manifest_path=batch_manifest_path,
        plan=plan,
        authority=authority,
        source_commit=source_commit,
        output_root=output_root,
        cell_id=cell_id,
    )
    payload = _jsonable_cell(cell)
    path = _single_cell_output_path(output_root, cell_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_gz(path, payload)
    return payload


def reduce_b48_b12x4(
    *,
    run_roots: Sequence[Path],
    authority_path: Path,
    source_commit: str,
    output_root: Path,
) -> dict[str, Any]:
    if len(run_roots) != 4:
        _fail(f"B48 B12x4 reduction requires exactly four partition roots, got {len(run_roots)}")
    _read_json(authority_path, name="authority binding")
    output_root.mkdir(parents=True, exist_ok=True)
    partitions = []
    for run_root in run_roots:
        plan = _read_json(run_root / "execution_plan.json", name="execution plan")
        batch_paths = p12b_terminal._batch_manifest_paths(run_root)
        if len(batch_paths) != 1:
            _fail(f"B48 B12x4 partition expects one physical batch: {run_root}")
        batch_manifest_path = batch_paths[0]
        batch_manifest = _read_json(batch_manifest_path, name="batch manifest")
        cell_ids = batch_manifest.get("cell_ids")
        if (
            not isinstance(cell_ids, list)
            or len(cell_ids) != 12
            or not all(type(cell_id) is str for cell_id in cell_ids)
        ):
            _fail("B48 B12x4 partition expects a physical b12 batch manifest")
        if batch_manifest.get("status") != "completed":
            _fail(f"partition batch is not completed: {batch_manifest_path}")
        plan_cells = _cell_rows(plan)
        if set(cell_ids) != set(plan_cells):
            _fail(f"partition cell ids do not match plan: {run_root}")
        cell_results = [
            _run_single_cell_subprocess(
                run_root=run_root,
                batch_manifest_path=batch_manifest_path,
                cell_id=str(cell_id),
                authority_path=authority_path,
                source_commit=source_commit,
                output_root=output_root,
            )
            for cell_id in cell_ids
        ]
        partitions.append(
            {
                "run_root": str(run_root),
                "campaign_id": plan.get("campaign_id"),
                "plan_sha256": plan.get("plan_sha256"),
                "cell_ids": list(cell_ids),
                "cell_results": cell_results,
                "comparisons": p12b_terminal._comparison_integrity(
                    run_root=run_root, plan=plan
                ),
            }
        )
    cell_results = [
        cell
        for partition in partitions
        for cell in partition["cell_results"]
    ]
    comparisons = _combined_comparisons(partitions)
    decision, classified_rows = _decision(
        cell_results=cell_results, comparisons=comparisons
    )
    classified_rows_payload = {"rows": classified_rows}
    classified_rows_sha256 = canonical_sha256(classified_rows_payload)
    reduction_path = output_root / "gv2_b48_mutation_persistence_b12x4_reduction_v1.json.gz"
    classified_rows_path = (
        output_root / "gv2_b48_mutation_persistence_b12x4_classified_rows_v1.json.gz"
    )
    decision_path = output_root / "gv2_b48_mutation_persistence_b12x4_decision_v1.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_roots": [str(path) for path in run_roots],
        "campaign_id": partitions[0]["campaign_id"],
        "partition_plan_sha256s": [partition["plan_sha256"] for partition in partitions],
        "source_commit": source_commit,
        "authority_path": str(authority_path),
        "authority_sha256": _sha256(authority_path),
        "scientific_milestone": SCIENTIFIC_MILESTONE,
        "technical_segments": list(TECHNICAL_SEGMENTS),
        "positive_identity_types": list(POSITIVE_IDENTITY_TYPES),
        "partitions": [
            {
                "run_root": partition["run_root"],
                "campaign_id": partition["campaign_id"],
                "plan_sha256": partition["plan_sha256"],
                "cell_ids": partition["cell_ids"],
                "cell_summaries": [
                    _cell_summary(cell) for cell in partition["cell_results"]
                ],
                "comparisons": partition["comparisons"],
            }
            for partition in partitions
        ],
        "classified_rows_path": str(classified_rows_path),
        "classified_rows_sha256": classified_rows_sha256,
        "comparisons": comparisons,
        "decision": decision,
    }
    result["result_sha256"] = canonical_sha256(result)
    _write_json_gz(reduction_path, result)
    _write_json_gz(classified_rows_path, classified_rows_payload)
    decision_payload = {**decision, "result_sha256": result["result_sha256"]}
    decision_path.write_text(pretty_json(decision_payload), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append")
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--single-cell-run-root", type=Path)
    parser.add_argument("--single-cell-batch-manifest", type=Path)
    parser.add_argument("--single-cell-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.single_cell_run_root is not None:
        if args.single_cell_batch_manifest is None or args.single_cell_id is None:
            _fail("single-cell mode requires batch manifest and cell id")
        result = reduce_single_cell(
            run_root=args.single_cell_run_root.expanduser().resolve(),
            batch_manifest_path=args.single_cell_batch_manifest.expanduser().resolve(),
            cell_id=str(args.single_cell_id),
            authority_path=args.authority.expanduser().resolve(),
            source_commit=args.source_commit,
            output_root=args.output_root.expanduser().resolve(),
        )
        print(
            pretty_json(
                {
                    "ok": True,
                    "cell_id": result["cell_id"],
                    "output_path": str(
                        _single_cell_output_path(
                            args.output_root.expanduser().resolve(),
                            str(args.single_cell_id),
                        )
                    ),
                }
            )
        )
        return 0
    if not args.run_root:
        _fail("normal reduction mode requires four --run-root values")
    result = reduce_b48_b12x4(
        run_roots=[path.expanduser().resolve() for path in args.run_root],
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
