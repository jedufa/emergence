#!/usr/bin/env python3
"""Analyze the frozen native stable-memory causal matrix and repeat receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Final, Iterable, Mapping


SCHEMA_VERSION: Final[str] = "eiel_native_memory_science_analysis_v1"
PRIMARY_METRICS: Final[tuple[str, ...]] = (
    "reproductive_contribution",
    "descendant_contribution",
    "final_population",
    "survival",
    "births",
    "deaths",
    "energy",
    "health",
    "wdl_damage",
    "historical_mean_offspring_total",
    "base_fitness_total",
)
HISTORICAL_SELECTION_METRICS: Final[tuple[str, ...]] = (
    "final_population",
    "historical_mean_offspring_total",
)
BENEFICIAL_HIGHER_METRICS: Final[tuple[str, ...]] = tuple(
    metric for metric in PRIMARY_METRICS if metric not in {"deaths", "wdl_damage"}
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sum_field(summaries: Iterable[Mapping[str, Any]], name: str) -> float:
    return fmean([0.0]) + sum(float(row.get(name, 0.0)) for row in summaries)


def _extract_cell(receipt_path: Path) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    contract = receipt["contract"]
    summaries_path = Path(receipt["generation_summaries_path"])
    summaries = _read_json(summaries_path)
    final = summaries[-1]
    science = final["memory_science"]
    behavior = final["behavior"]
    lineage = final["lineage"]
    behavior_rows = int(science["behavior_row_count"])
    offspring = float(science["offspring_total"])
    cue_away = _sum_field(summaries, "wdl_cue_exposed_next_move_away_count")
    cue_toward = _sum_field(summaries, "wdl_cue_exposed_next_move_toward_count")
    cue_stay = _sum_field(summaries, "wdl_cue_exposed_stayed_count")
    return {
        "cell_id": receipt["cell_id"],
        "memory_arm": contract["memory_arm"],
        "wdl_arm": contract["wdl_arm"],
        "init_arm": contract["init_arm"],
        "seed": int(contract["seed"]),
        "generations": int(contract["generations"]),
        "extension_sha256": receipt["extension_sha256"],
        "contract_sha256": receipt["contract_sha256"],
        "initial_matched_tensor_digest_sha256": receipt[
            "initial_matched_tensor_digest_sha256"
        ],
        "terminal_checkpoint_digest": receipt["terminal_checkpoint_digest"],
        "generation_summaries_sha256": _sha256(summaries_path),
        "founder_exact": bool(receipt["founder_receipt"]["exact"]),
        "controller_invocations": int(receipt["controller_invocations"]),
        "python_generation_dispatches": int(receipt["python_generation_dispatches"]),
        "hidden_fallback": bool(receipt["hidden_fallback"]),
        "runtime_jit": bool(receipt["runtime_jit"]),
        "reproductive_contribution": offspring,
        "descendant_contribution": offspring,
        # The controller fail-closes unless turnover restores this fixed target.
        "final_population": float(contract["population"]),
        "survival": float(behavior["cohort_survivors"])
        / max(1.0, float(behavior["cohort_size"])),
        "births": _sum_field(summaries, "committed_birth_rows"),
        "deaths": _sum_field(summaries, "death_rows"),
        "energy": float(science["energy_collected_total"]),
        "health": -float(science["injury_end_total"]),
        "wdl_damage": _sum_field(summaries, "wdl_delayed_hazard_damage_total"),
        "historical_mean_offspring_total": offspring / max(1, behavior_rows),
        "base_fitness_total": float(science["base_fitness_total"]),
        "memory_cost_charged_total": _sum_field(
            summaries, "recurrent_upkeep_charge_total"
        ),
        "memory_cost_charged_agent_ticks": _sum_field(
            summaries, "recurrent_upkeep_charged_agent_ticks"
        ),
        "cue_exposure": _sum_field(
            summaries, "wdl_delayed_hazard_agent_cue_exposure_count"
        ),
        "cue_move_away": cue_away,
        "cue_move_toward": cue_toward,
        "cue_stay": cue_stay,
        "cue_conditioned_move_balance": cue_away - cue_toward,
        "hazard_exposure": _sum_field(
            summaries, "wdl_delayed_hazard_agent_active_exposure_count"
        ),
        "active_lineages": float(lineage["active_lineages"]),
        "top1_alive_share": float(lineage["dominant_population_share"]),
        "lineage_entropy": float(lineage["population_entropy"]),
        "summaries": summaries,
    }


def _load_cells(root: Path) -> list[dict[str, Any]]:
    receipts = sorted(root.glob("native_memory__*/cell_receipt.json"))
    return [_extract_cell(path) for path in receipts]


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return str(row["wdl_arm"]), str(row["init_arm"]), int(row["seed"])


def _paired_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normal = {_pair_key(row): row for row in cells if row["memory_arm"] == "memory_normal"}
    ablated = {_pair_key(row): row for row in cells if row["memory_arm"] == "memory_ablated"}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(normal) & set(ablated)):
        n = normal[key]
        a = ablated[key]
        row: dict[str, Any] = {
            "wdl_arm": key[0],
            "init_arm": key[1],
            "seed": key[2],
            "initial_tensor_match": n["initial_matched_tensor_digest_sha256"]
            == a["initial_matched_tensor_digest_sha256"],
        }
        for metric in PRIMARY_METRICS:
            row[f"delta_{metric}"] = float(n[metric]) - float(a[metric])
        row["delta_memory_cost_charged_total"] = (
            float(n["memory_cost_charged_total"])
            - float(a["memory_cost_charged_total"])
        )
        for metric in (
            "cue_exposure",
            "cue_move_away",
            "cue_move_toward",
            "cue_stay",
            "cue_conditioned_move_balance",
            "hazard_exposure",
            "top1_alive_share",
            "active_lineages",
            "lineage_entropy",
        ):
            row[f"delta_{metric}"] = float(n[metric]) - float(a[metric])
        first_divergence = None
        behavior_fields = (
            "wdl_cue_exposed_next_move_away_count",
            "wdl_cue_exposed_next_move_toward_count",
            "wdl_cue_exposed_stayed_count",
        )
        for generation, (normal_summary, ablated_summary) in enumerate(
            zip(n["summaries"], a["summaries"], strict=True)
        ):
            if any(normal_summary.get(field) != ablated_summary.get(field) for field in behavior_fields):
                first_divergence = generation
                break
        row["first_cue_behavior_divergence_generation"] = first_divergence
        rows.append(row)
    return rows


def _aggregate_effects(paired: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for init_arm in ("init_legacy", "init_founder_balanced", "all"):
        selected = paired if init_arm == "all" else [r for r in paired if r["init_arm"] == init_arm]
        enabled = [r for r in selected if r["wdl_arm"] == "wdl_enabled"]
        disabled = [r for r in selected if r["wdl_arm"] == "wdl_disabled"]
        effects: dict[str, Any] = {}
        for metric in PRIMARY_METRICS:
            enabled_effect = fmean(float(r[f"delta_{metric}"]) for r in enabled)
            disabled_effect = fmean(float(r[f"delta_{metric}"]) for r in disabled)
            effects[metric] = {
                "wdl_enabled_normal_minus_ablated_mean": enabled_effect,
                "wdl_disabled_normal_minus_ablated_mean": disabled_effect,
                "memory_by_wdl_interaction": enabled_effect - disabled_effect,
                "wdl_enabled_positive_pairs": sum(
                    float(r[f"delta_{metric}"]) > 0.0 for r in enabled
                ),
                "paired_seed_count": len(enabled),
            }
        result[init_arm] = effects
    return result


def _repeat_parity(cells: list[dict[str, Any]], repeat_root: Path) -> dict[str, Any]:
    repeat = _load_cells(repeat_root)
    primary_by_id = {row["cell_id"]: row for row in cells}
    rows = []
    for candidate in repeat:
        primary = primary_by_id.get(candidate["cell_id"])
        exact = bool(
            primary
            and primary["terminal_checkpoint_digest"]
            == candidate["terminal_checkpoint_digest"]
            and primary["generation_summaries_sha256"]
            == candidate["generation_summaries_sha256"]
        )
        rows.append({"cell_id": candidate["cell_id"], "exact": exact})
    return {
        "schema_version": "eiel_native_memory_completed_repeat_parity_v1",
        "repeat_count": len(rows),
        "exact_count": sum(row["exact"] for row in rows),
        "all_exact": len(rows) >= 5 and all(row["exact"] for row in rows),
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    cells = _load_cells(args.primary_root.resolve())
    paired = _paired_rows(cells)
    effects = _aggregate_effects(paired)
    repeat = _repeat_parity(cells, args.repeat_root.resolve())
    output = args.output_root.resolve()
    integrity_failures = []
    if len(cells) != 80:
        integrity_failures.append(f"completed_cells={len(cells)} expected=80")
    if len(paired) != 40:
        integrity_failures.append(f"matched_pairs={len(paired)} expected=40")
    if not all(row["initial_tensor_match"] for row in paired):
        integrity_failures.append("one or more matched arms have different initial tensors")
    if not all(row["founder_exact"] for row in cells):
        integrity_failures.append("one or more founder receipts are inexact")
    if any(row["hidden_fallback"] or row["runtime_jit"] for row in cells):
        integrity_failures.append("fallback or runtime JIT was observed")
    if any(row["python_generation_dispatches"] != 0 for row in cells):
        integrity_failures.append("Python generation dispatch was observed")

    cost_pairs = [float(row["delta_memory_cost_charged_total"]) for row in paired]
    cost_sanity = {
        "all_normal_costs_exceed_ablated": all(value > 0.0 for value in cost_pairs),
        "positive_pairs": sum(value > 0.0 for value in cost_pairs),
        "pair_count": len(cost_pairs),
        "mean_normal_minus_ablated_cost": fmean(cost_pairs),
    }
    founder_wdl_normal = [
        row
        for row in cells
        if row["init_arm"] == "init_founder_balanced"
        and row["wdl_arm"] == "wdl_enabled"
        and row["memory_arm"] == "memory_normal"
    ]
    dominant = [
        row
        for row in founder_wdl_normal
        if row["top1_alive_share"] >= 0.95 or row["active_lineages"] <= 1.5
    ]
    lineage_sanity = {
        "founder_balance_violations": sum(not row["founder_exact"] for row in cells),
        "founder_balanced_wdl_normal_rows": len(founder_wdl_normal),
        "single_or_dominant_rows": len(dominant),
        "single_or_dominant_fraction": len(dominant) / max(1, len(founder_wdl_normal)),
        "lineage_confounded": len(dominant) / max(1, len(founder_wdl_normal)) >= 0.75,
    }
    all_effects = effects["all"]
    positive_wdl_metrics = [
        metric
        for metric in BENEFICIAL_HIGHER_METRICS
        if all_effects[metric]["wdl_enabled_normal_minus_ablated_mean"] > 0.0
    ]
    specific_metrics = [
        metric
        for metric in positive_wdl_metrics
        if all_effects[metric]["memory_by_wdl_interaction"] > 0.0
    ]
    historical_positive = [
        metric
        for metric in HISTORICAL_SELECTION_METRICS
        if all_effects[metric]["wdl_enabled_normal_minus_ablated_mean"] > 0.0
        and all_effects[metric]["memory_by_wdl_interaction"] > 0.0
    ]
    behavior = {
        "wdl_pair_count": sum(row["wdl_arm"] == "wdl_enabled" for row in paired),
        "pairs_with_cue_behavior_divergence": sum(
            row["wdl_arm"] == "wdl_enabled"
            and row["first_cue_behavior_divergence_generation"] is not None
            for row in paired
        ),
        "first_divergence_resolution": "generation_only",
        "tick_level_timing_available": False,
    }
    if integrity_failures or not repeat["all_exact"] or not cost_sanity[
        "all_normal_costs_exceed_ablated"
    ]:
        decision = "EIEL-NATIVE-MEMORY-REP-C/ONE-PRECISE-PREREQUISITE"
    elif specific_metrics and historical_positive and not lineage_sanity["lineage_confounded"]:
        decision = "EIEL-NATIVE-MEMORY-REP-A/WDL-SPECIFIC-STABLE-MEMORY-SIGNAL"
    elif positive_wdl_metrics:
        decision = "EIEL-NATIVE-MEMORY-REP-B/SIGNAL-PRESENT-SPECIFICITY-INCOMPLETE"
    else:
        decision = "EIEL-NATIVE-MEMORY-REP-D/HISTORICAL-SIGNAL-NOT-REPLICATED"
    decision_record = {
        "schema_version": "eiel_native_memory_replication_decision_v1",
        "decision": decision,
        "primary_cells": len(cells),
        "matched_pairs": len(paired),
        "integrity_failures": integrity_failures,
        "positive_wdl_metrics": positive_wdl_metrics,
        "positive_wdl_specific_metrics": specific_metrics,
        "metric_direction_contract": {
            "beneficial_when_higher": list(BENEFICIAL_HIGHER_METRICS),
            "adverse_when_higher": ["deaths", "wdl_damage"],
        },
        "historical_selection_metrics_positive_and_specific": historical_positive,
        "replication_a": decision.endswith("WDL-SPECIFIC-STABLE-MEMORY-SIGNAL"),
        "robustness_and_jungle_transfer_allowed": decision.endswith(
            "WDL-SPECIFIC-STABLE-MEMORY-SIGNAL"
        ),
        "direct_reward_active": False,
        "cost_sanity": cost_sanity,
        "lineage_sanity": lineage_sanity,
        "behavior_before_consequence": behavior,
        "repeat_parity": repeat,
    }
    _write_csv(output / "native_memory_primary_cell_summary.csv", [
        {key: value for key, value in row.items() if key != "summaries"} for row in cells
    ])
    _write_csv(output / "native_memory_primary_paired_deltas.csv", paired)
    _write_json(output / "native_memory_primary_effects.json", effects)
    _write_json(output / "native_memory_cost_sanity.json", cost_sanity)
    _write_json(output / "native_memory_lineage_sanity.json", lineage_sanity)
    _write_json(output / "native_memory_behavior_before_consequence.json", behavior)
    _write_json(output / "native_memory_completed_repeat_parity.json", repeat)
    _write_json(output / "native_memory_replication_decision.json", decision_record)
    print(json.dumps(decision_record, indent=2, sort_keys=True))
    return 0 if not integrity_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
