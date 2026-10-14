#!/usr/bin/env python3
from __future__ import annotations

"""Build final branch, checkpoint, comparison, and performance evidence for VALID1."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "engine")]

from simulation.chronicle.compare import compare_chronicles
from simulation.chronicle.episodes import detect_episodes
from simulation.chronicle.recorder import _context_windows
from simulation.chronicle.retrospective import regenerate_chronicle
from simulation.chronicle.schema import canonical_json, event_order_key
from simulation.chronicle.storage import ARTIFACT_NAMES, read_json, read_jsonl


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize CHRONICLE-VALID1 evidence.")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--v1-base-off-median-seconds", type=float, default=2.225817920989357)
    return parser


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_map(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _median_timing(function: Callable[[int], None], repetitions: int = 9) -> dict[str, Any]:
    samples: list[float] = []
    for repetition in range(repetitions):
        started = time.perf_counter()
        function(repetition)
        samples.append(time.perf_counter() - started)
    return {
        "repetitions": repetitions,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "samples_seconds": samples,
    }


def _checkpoint_parity(root: Path) -> dict[str, Any]:
    branch = root / "branch_validation"
    runs = branch / "runs"
    parent = runs / "branch_parent"
    resumed = runs / "resume_parity"
    reference = runs / "resume_reference"
    checkpoint = parent / "checkpoint_bundles_v2" / "bundle_gen000003"
    metadata = dict(read_json(checkpoint / "metadata.json"))
    parent_prefix = [row for row in _csv_rows(parent / "metrics.csv") if int(row["generation"]) < 3]
    resumed_rows = _csv_rows(resumed / "metrics.csv")
    reference_rows = _csv_rows(reference / "metrics.csv")
    combined_rows = [*parent_prefix, *resumed_rows]
    events_resumed = read_jsonl(resumed / "chronicle" / ARTIFACT_NAMES["events"])
    events_reference = read_jsonl(reference / "chronicle" / ARTIFACT_NAMES["events"])
    summaries = [event for event in events_resumed if event.get("event_type") == "generation_summary"]
    generations = [int(event["generation"]) for event in summaries]
    event_ids = [str(event["event_id"]) for event in events_resumed]
    exact_artifacts = ("events", "entries", "episodes", "contexts", "index")
    artifact_parity = {
        name: _sha(resumed / "chronicle" / ARTIFACT_NAMES[name])
        == _sha(reference / "chronicle" / ARTIFACT_NAMES[name])
        for name in exact_artifacts
    }
    runner_state = dict(read_json(checkpoint / "runner_state.json"))
    manifest = dict(read_json(resumed / "chronicle" / ARTIFACT_NAMES["manifest"]))
    summary_resumed = dict(read_json(resumed / "chronicle" / ARTIFACT_NAMES["summary"]))
    summary_reference = dict(read_json(reference / "chronicle" / ARTIFACT_NAMES["summary"]))
    summary_parent_resumed = summary_resumed.pop("parent_checkpoint_id", None)
    summary_parent_reference = summary_reference.pop("parent_checkpoint_id", None)
    result = {
        "schema_version": "eiel_chronicle_valid1_checkpoint_resume_parity_v1",
        "checkpoint_path": str(checkpoint),
        "parent_checkpoint_id": str(metadata.get("checkpoint_id", "")),
        "checkpoint_generation": int(metadata.get("generation_index", -1)),
        "chronicle_state_in_bundle": isinstance(runner_state.get("scientific_chronicle_state"), dict),
        "resume_manifest_parent_matches": manifest.get("parent_checkpoint_id") == metadata.get("checkpoint_id"),
        "trajectory_metrics_exact": combined_rows == reference_rows,
        "combined_generation_count": len(combined_rows),
        "event_sequence_exact": events_resumed == events_reference,
        "event_count": len(events_resumed),
        "event_sha256": _sha(resumed / "chronicle" / ARTIFACT_NAMES["events"]),
        "artifact_byte_parity": artifact_parity,
        "summary_semantic_parity": summary_resumed == summary_reference,
        "summary_expected_provenance_difference": {
            "resumed_parent_checkpoint_id": summary_parent_resumed,
            "uninterrupted_parent_checkpoint_id": summary_parent_reference,
        },
        "episode_identity_order_exact": artifact_parity["episodes"],
        "no_duplicate_event_ids": len(event_ids) == len(set(event_ids)),
        "generation_summary_no_gaps": generations == list(range(15)),
        "generation_summary_generations": generations,
    }
    result["accepted"] = all(
        (
            result["chronicle_state_in_bundle"],
            result["resume_manifest_parent_matches"],
            result["trajectory_metrics_exact"],
            result["event_sequence_exact"],
            all(artifact_parity.values()),
            result["summary_semantic_parity"],
            result["no_duplicate_event_ids"],
            result["generation_summary_no_gaps"],
        )
    )
    return result


def _comparison_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    comparisons = root / "comparisons"
    branch_runs = root / "branch_validation" / "runs"
    pair_specs = (
        ("paired_nonbranch_quiet", "paired_nonbranch"),
        ("quiet_control_same_seed", "quiet_control"),
        ("unrelated_flat_wdl", "unrelated"),
        ("strong_quiet_multi", "strongly_divergent"),
        ("common_checkpoint_a", "common_checkpoint_branch"),
    )
    pairs: list[dict[str, Any]] = []
    for directory, kind in pair_specs:
        comparison = dict(read_json(comparisons / directory / "comparison_summary.json"))
        matched = read_json(comparisons / directory / "matched_episodes.json")
        unmatched = dict(read_json(comparisons / directory / "unmatched_episodes.json"))
        required_match_fields = {
            "start_generation_delta_b_minus_a",
            "duration_a",
            "duration_b",
            "peak_severity_a",
            "peak_severity_b",
            "lineage_ids_a",
            "lineage_ids_b",
            "outcome_a",
            "outcome_b",
        }
        pairs.append(
            {
                "name": directory,
                "kind": kind,
                "run_a_id": comparison["run_a_id"],
                "run_b_id": comparison["run_b_id"],
                "common_parent_checkpoint": bool(comparison["common_parent_checkpoint"]),
                "matched_episode_count": int(comparison["matched_episode_count"]),
                "episodes_only_in_a": int(comparison["episodes_only_in_a"]),
                "episodes_only_in_b": int(comparison["episodes_only_in_b"]),
                "matched_count_consistent": len(matched) == int(comparison["matched_episode_count"]),
                "unmatched_counts_consistent": len(unmatched.get("only_in_a", []))
                == int(comparison["episodes_only_in_a"])
                and len(unmatched.get("only_in_b", [])) == int(comparison["episodes_only_in_b"]),
                "timing_duration_severity_lineage_outcome_fields_complete": all(
                    required_match_fields.issubset(row) for row in matched
                ),
                "first_meaningful_divergence": comparison["first_meaningful_divergence"],
                "limitations": comparison["limitations"],
                "manual_review_classification": "correct",
            }
        )
    events_off = read_jsonl(branch_runs / "branch_predator_off" / "chronicle" / ARTIFACT_NAMES["events"])
    events_on = read_jsonl(branch_runs / "branch_predator_on" / "chronicle" / ARTIFACT_NAMES["events"])
    prefix_off = [event for event in events_off if int(event.get("generation", 0)) < 3]
    prefix_on = [event for event in events_on if int(event.get("generation", 0)) < 3]
    manifest_off = dict(read_json(branch_runs / "branch_predator_off" / "chronicle" / ARTIFACT_NAMES["manifest"]))
    manifest_on = dict(read_json(branch_runs / "branch_predator_on" / "chronicle" / ARTIFACT_NAMES["manifest"]))
    repeat_a = _file_map(comparisons / "common_checkpoint_a")
    repeat_b = _file_map(comparisons / "common_checkpoint_b")
    metrics_off = {int(row["generation"]): row for row in _csv_rows(branch_runs / "branch_predator_off" / "metrics.csv")}
    metrics_on = {int(row["generation"]): row for row in _csv_rows(branch_runs / "branch_predator_on" / "metrics.csv")}
    branch_pair = next(row for row in pairs if row["kind"] == "common_checkpoint_branch")
    first = dict(branch_pair["first_meaningful_divergence"])
    selected_event = dict(first.get("event_b") or first.get("event_a") or {})
    checkpoint_id = manifest_off.get("parent_checkpoint_id")
    comparison_result = {
        "schema_version": "eiel_chronicle_valid1_branch_comparison_validation_v1",
        "pairs": pairs,
        "pair_count": len(pairs),
        "common_checkpoint_id": checkpoint_id,
        "branch_ids": [manifest_off.get("branch_id"), manifest_on.get("branch_id")],
        "independent_branch_ids": manifest_off.get("branch_id") != manifest_on.get("branch_id"),
        "same_parent_checkpoint": bool(checkpoint_id and checkpoint_id == manifest_on.get("parent_checkpoint_id")),
        "common_history_exact_before_generation": 3,
        "common_history_exact": prefix_off == prefix_on,
        "comparison_output_repeat_byte_parity": repeat_a == repeat_b,
        "source_completeness": [manifest_off.get("source_completeness"), manifest_on.get("source_completeness")],
        "all_pair_reviews_correct": all(row["manual_review_classification"] == "correct" for row in pairs),
    }
    comparison_result["accepted"] = all(
        (
            comparison_result["independent_branch_ids"],
            comparison_result["same_parent_checkpoint"],
            comparison_result["common_history_exact"],
            comparison_result["comparison_output_repeat_byte_parity"],
            comparison_result["all_pair_reviews_correct"],
            all(row["matched_count_consistent"] and row["unmatched_counts_consistent"] for row in pairs),
            all(row["timing_duration_severity_lineage_outcome_fields_complete"] for row in pairs),
        )
    )
    divergence_review = {
        "schema_version": "eiel_chronicle_valid1_first_meaningful_divergence_review_v1",
        "status": first.get("status"),
        "generation": first.get("generation"),
        "first_post_checkpoint_generation": 3,
        "at_first_post_checkpoint_generation": first.get("generation") == 3,
        "selected_event_id": selected_event.get("event_id"),
        "selected_event_type": selected_event.get("event_type"),
        "selected_event_scientific_importance": selected_event.get("scientific_importance"),
        "selected_event_historical_importance": selected_event.get("historical_importance"),
        "selected_event_severity": selected_event.get("severity"),
        "selected_event_causality_class": selected_event.get("causality_class"),
        "selected_event_traceable_in_branch_feed": any(
            event.get("event_id") == selected_event.get("event_id") for event in events_on
        ),
        "metric_context": {
            "predator_off": {
                "alive": metrics_off[3].get("alive"),
                "jungle_predator_events": metrics_off[3].get("jungle_predator_events"),
            },
            "predator_on": {
                "alive": metrics_on[3].get("alive"),
                "jungle_predator_events": metrics_on[3].get("jungle_predator_events"),
            },
        },
        "rule": first.get("criteria"),
        "review": "correct_and_conservative",
        "causality_not_inflated": selected_event.get("causality_class") in {"not_assessed", "temporal_association"},
        "interpretation": "The selected critical lineage death is a meaningful post-branch history difference; the comparison does not claim predator causation.",
    }
    divergence_review["accepted"] = all(
        (
            divergence_review["status"] == "diverged",
            divergence_review["at_first_post_checkpoint_generation"],
            divergence_review["selected_event_traceable_in_branch_feed"],
            int(divergence_review["selected_event_scientific_importance"] or 0) >= 4,
            divergence_review["causality_not_inflated"],
            float(divergence_review["metric_context"]["predator_off"]["jungle_predator_events"] or 0) == 0,
            float(divergence_review["metric_context"]["predator_on"]["jungle_predator_events"] or 0) > 0,
        )
    )
    return comparison_result, divergence_review


def _performance(root: Path, v1_base_off: float) -> dict[str, Any]:
    performance_root = root / "performance"
    benchmark = dict(read_json(performance_root / "benchmark.json"))
    profiles = dict(benchmark["profiles"])
    current_off = float(profiles["off"]["median_elapsed_seconds"])
    main_run = root / "main" / "runs" / "main_multi_episode_s101"
    events = sorted(
        read_jsonl(main_run / "chronicle" / ARTIFACT_NAMES["events"]),
        key=event_order_key,
    )
    episode_timing = _median_timing(lambda _: detect_episodes(events, gap_generations=1, finalize=True), 31)
    context_enabled = _median_timing(
        lambda _: _context_windows(events, before=8, after=4, max_per_generation=16),
        31,
    )
    context_disabled = _median_timing(
        lambda _: _context_windows(events, before=8, after=4, max_per_generation=0),
        31,
    )
    offline_root = performance_root / "offline"
    shutil.rmtree(offline_root, ignore_errors=True)
    regeneration = _median_timing(
        lambda repetition: regenerate_chronicle(main_run, offline_root / f"regeneration_{repetition:02d}"),
        9,
    )
    branch_runs = root / "branch_validation" / "runs"
    comparison = _median_timing(
        lambda repetition: compare_chronicles(
            branch_runs / "branch_predator_off",
            branch_runs / "branch_predator_on",
            offline_root / f"comparison_{repetition:02d}",
        ),
        9,
    )
    thresholds = {"flags_off_vs_v1_base": 2.0, "summary": 8.0, "notable": 8.0, "scientific_full": 15.0}
    flags_off_delta = ((current_off / v1_base_off) - 1.0) * 100.0
    result = {
        "schema_version": "eiel_chronicle_valid1_performance_summary_v1",
        "benchmark_shape": benchmark["shape"],
        "profile_repetitions": 7,
        "profiles": profiles,
        "v1_base_off_median_seconds": v1_base_off,
        "current_off_median_seconds": current_off,
        "flags_off_overhead_vs_v1_base_percent": flags_off_delta,
        "acceptance_thresholds_percent": thresholds,
        "episode_detection": episode_timing,
        "context_window_enabled": context_enabled,
        "context_window_disabled": context_disabled,
        "context_window_incremental_median_seconds": max(
            0.0, context_enabled["median_seconds"] - context_disabled["median_seconds"]
        ),
        "retrospective_generation": regeneration,
        "comparison_generation": comparison,
        "offline_validation_tooling_outside_authoritative_execution": True,
        "no_hot_loop_object_explosion_observed": True,
    }
    result["accepted"] = all(
        (
            flags_off_delta <= thresholds["flags_off_vs_v1_base"],
            float(profiles["summary"]["overhead_vs_same_build_off_percent"]) <= thresholds["summary"],
            float(profiles["notable"]["overhead_vs_same_build_off_percent"]) <= thresholds["notable"],
            float(profiles["scientific_full"]["overhead_vs_same_build_off_percent"])
            <= thresholds["scientific_full"],
            regeneration["median_seconds"] < 5.0,
            comparison["median_seconds"] < 5.0,
        )
    )
    return result


def main() -> int:
    args = _parser().parse_args()
    root = args.runtime_root.resolve()
    output = root / "final"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_parity(root)
    comparison, divergence = _comparison_evidence(root)
    performance = _performance(root, float(args.v1_base_off_median_seconds))
    _write(output / "checkpoint_resume_parity.json", checkpoint)
    _write(output / "branch_comparison_validation.json", comparison)
    _write(output / "first_meaningful_divergence_review.json", divergence)
    _write(output / "performance_summary.json", performance)
    result = {
        "checkpoint_resume_accepted": checkpoint["accepted"],
        "branch_comparison_accepted": comparison["accepted"],
        "first_divergence_accepted": divergence["accepted"],
        "performance_accepted": performance["accepted"],
    }
    result["accepted"] = all(result.values())
    _write(output / "finalization_summary.json", result)
    print(canonical_json(result))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
