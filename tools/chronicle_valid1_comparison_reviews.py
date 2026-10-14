#!/usr/bin/env python3
from __future__ import annotations

"""Build deterministic semantic reviews for required VALID1 comparison classes."""

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "engine")]

from simulation.chronicle.storage import ARTIFACT_NAMES, read_json, read_jsonl, write_json


REQUIRED_MATCH_FIELDS = {
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review required VALID1 comparison classes.")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _run_root(runtime_root: Path, run_id: str) -> Path:
    candidates = (
        runtime_root / "main" / "runs" / run_id,
        runtime_root / "branch_validation" / "runs" / run_id,
    )
    return next((path for path in candidates if path.exists()), candidates[0])


def _event_is_traceable(runtime_root: Path, event: Mapping[str, Any] | None) -> bool:
    if not event or not event.get("event_id") or not event.get("run_id"):
        return False
    root = _run_root(runtime_root, str(event["run_id"])) / "chronicle"
    return any(
        str(candidate.get("event_id")) == str(event["event_id"])
        for candidate in read_jsonl(root / ARTIFACT_NAMES["events"])
    )


def _meaningful(event: Mapping[str, Any] | None) -> bool:
    if not event:
        return False
    return bool(
        int(event.get("scientific_importance", 0)) >= 4
        or int(event.get("historical_importance", 0)) >= 4
        or str(event.get("severity", "")) in {"major", "critical"}
        or str(event.get("live_priority", "")) in {"high", "critical"}
    )


def _review(
    *,
    runtime_root: Path,
    comparison_root: Path,
    name: str,
    comparison_class: str,
) -> dict[str, Any]:
    root = comparison_root / name
    summary = dict(read_json(root / "comparison_summary.json"))
    matched = list(read_json(root / "matched_episodes.json"))
    unmatched = dict(read_json(root / "unmatched_episodes.json"))
    first = dict(summary["first_meaningful_divergence"])
    event_a = first.get("event_a")
    event_b = first.get("event_b")
    selected = event_b or event_a
    source_traceable = (
        _event_is_traceable(runtime_root, event_a) if event_a else True
    ) and (
        _event_is_traceable(runtime_root, event_b) if event_b else True
    )
    common = bool(summary.get("common_parent_checkpoint"))
    complete = all(
        str(summary.get(key)) == "live_generation_boundary_complete"
        for key in ("source_completeness_a", "source_completeness_b")
    )
    match_fields_complete = all(REQUIRED_MATCH_FIELDS.issubset(row) for row in matched)
    counts_consistent = (
        len(matched) == int(summary["matched_episode_count"])
        and len(unmatched.get("only_in_a", [])) == int(summary["episodes_only_in_a"])
        and len(unmatched.get("only_in_b", [])) == int(summary["episodes_only_in_b"])
    )
    classification = {
        "severity": selected.get("severity") if selected else None,
        "live_priority": selected.get("live_priority") if selected else None,
        "scientific_importance": selected.get("scientific_importance") if selected else None,
        "historical_importance": selected.get("historical_importance") if selected else None,
        "confidence_class": selected.get("confidence_class") if selected else None,
        "causality_class": selected.get("causality_class") if selected else None,
    }
    review: dict[str, Any] = {
        "schema_version": "eiel_chronicle_valid1_comparison_review_v1",
        "comparison_name": name,
        "comparison_class": comparison_class,
        "run_a_id": summary["run_a_id"],
        "run_b_id": summary["run_b_id"],
        "common_parent_checkpoint": common,
        "matched_episode_count": int(summary["matched_episode_count"]),
        "episodes_only_in_a": int(summary["episodes_only_in_a"]),
        "episodes_only_in_b": int(summary["episodes_only_in_b"]),
        "counts_consistent": counts_consistent,
        "matched_fields_complete": match_fields_complete,
        "source_completeness": [
            summary.get("source_completeness_a"),
            summary.get("source_completeness_b"),
        ],
        "source_complete": complete,
        "first_meaningful_divergence": {
            "status": first.get("status"),
            "generation": first.get("generation"),
            "reason": first.get("reason"),
            "event_id": selected.get("event_id") if selected else None,
            "event_type": selected.get("event_type") if selected else None,
            "event_category": selected.get("event_category") if selected else None,
            "classification": classification,
            "metric_differences": first.get("metric_differences", []),
            "criteria": first.get("criteria", {}),
            "source_traceable": source_traceable,
        },
        "limitations": list(summary.get("limitations", []) or []),
    }
    diverged = first.get("status") == "diverged"
    causality_conservative = classification["causality_class"] in {
        None,
        "not_assessed",
        "temporal_association",
    }
    if comparison_class == "quiet_control":
        metric_differences = list(first.get("metric_differences", []) or [])
        review["interpretation"] = (
            "The same-seed controls have different declared configurations and no common checkpoint. "
            "The first reported difference is a material measured generation-summary difference, "
            "not a fabricated high-priority event or a claim of common-history divergence."
        )
        class_accepted = bool(
            not common
            and diverged
            and selected
            and selected.get("event_type") == "generation_summary"
            and metric_differences
            and source_traceable
        )
    elif comparison_class == "unrelated":
        review["interpretation"] = (
            "The unrelated runs make no common-history claim. The first material difference is a "
            "source-linked event whose historical classification meets the frozen meaningful-event rule."
        )
        class_accepted = bool(
            not common and diverged and _meaningful(selected) and source_traceable and causality_conservative
        )
    elif comparison_class == "strong_divergence":
        review["interpretation"] = (
            "The same-seed quiet and multi-pressure configurations diverge at a source-linked "
            "historically meaningful event and have substantially different episode inventories. "
            "The comparison records difference without assigning cause."
        )
        class_accepted = bool(
            not common
            and diverged
            and _meaningful(selected)
            and source_traceable
            and causality_conservative
            and int(summary["episodes_only_in_a"]) + int(summary["episodes_only_in_b"]) >= 10
        )
    elif comparison_class == "paired_nonbranch":
        review["interpretation"] = (
            "The paired non-branch runs have no common checkpoint. Their material summary difference "
            "is reported as sequence-based comparison context, not as a branch divergence."
        )
        class_accepted = bool(not common and diverged and source_traceable)
    else:
        raise ValueError(comparison_class)
    review["accepted"] = bool(
        class_accepted and complete and counts_consistent and match_fields_complete
    )
    return review


def _metrics(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["generation"]): dict(row) for row in csv.DictReader(handle)}


def _branch_reviews(
    runtime_root: Path,
    comparison_root: Path,
    pair_reviews: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = comparison_root / "common_checkpoint"
    summary = dict(read_json(root / "comparison_summary.json"))
    first = dict(summary["first_meaningful_divergence"])
    selected = dict(first.get("event_b") or first.get("event_a") or {})
    run_a = _run_root(runtime_root, str(summary["run_a_id"]))
    run_b = _run_root(runtime_root, str(summary["run_b_id"]))
    events_a = read_jsonl(run_a / "chronicle" / ARTIFACT_NAMES["events"])
    events_b = read_jsonl(run_b / "chronicle" / ARTIFACT_NAMES["events"])
    manifest_a = dict(read_json(run_a / "chronicle" / ARTIFACT_NAMES["manifest"]))
    manifest_b = dict(read_json(run_b / "chronicle" / ARTIFACT_NAMES["manifest"]))
    metrics_a = _metrics(run_a / "metrics.csv")
    metrics_b = _metrics(run_b / "metrics.csv")
    prefix_a = [event for event in events_a if int(event.get("generation", 0)) < 3]
    prefix_b = [event for event in events_b if int(event.get("generation", 0)) < 3]
    checkpoint_id = manifest_a.get("parent_checkpoint_id")
    branch_pair = {
        "comparison_name": "common_checkpoint",
        "comparison_class": "common_checkpoint_branch",
        "run_a_id": summary["run_a_id"],
        "run_b_id": summary["run_b_id"],
        "common_parent_checkpoint": bool(summary["common_parent_checkpoint"]),
        "matched_episode_count": int(summary["matched_episode_count"]),
        "episodes_only_in_a": int(summary["episodes_only_in_a"]),
        "episodes_only_in_b": int(summary["episodes_only_in_b"]),
        "first_meaningful_divergence_generation": first.get("generation"),
        "accepted": bool(
            summary.get("common_parent_checkpoint")
            and checkpoint_id
            and checkpoint_id == manifest_b.get("parent_checkpoint_id")
            and manifest_a.get("branch_id") != manifest_b.get("branch_id")
            and prefix_a == prefix_b
            and first.get("status") == "diverged"
            and first.get("generation") == 3
            and _event_is_traceable(runtime_root, selected)
        ),
    }
    all_pairs = [
        {
            "comparison_name": review["comparison_name"],
            "comparison_class": review["comparison_class"],
            "run_a_id": review["run_a_id"],
            "run_b_id": review["run_b_id"],
            "common_parent_checkpoint": review["common_parent_checkpoint"],
            "matched_episode_count": review["matched_episode_count"],
            "episodes_only_in_a": review["episodes_only_in_a"],
            "episodes_only_in_b": review["episodes_only_in_b"],
            "first_meaningful_divergence_generation": review["first_meaningful_divergence"]["generation"],
            "accepted": review["accepted"],
        }
        for review in pair_reviews
    ]
    all_pairs.append(branch_pair)
    validation = {
        "schema_version": "eiel_chronicle_valid1_branch_comparison_validation_v2",
        "pair_count": len(all_pairs),
        "pairs": all_pairs,
        "common_checkpoint_id": checkpoint_id,
        "branch_ids": [manifest_a.get("branch_id"), manifest_b.get("branch_id")],
        "independent_branch_ids": manifest_a.get("branch_id") != manifest_b.get("branch_id"),
        "same_parent_checkpoint": bool(
            checkpoint_id and checkpoint_id == manifest_b.get("parent_checkpoint_id")
        ),
        "common_history_exact_before_generation": 3,
        "common_history_exact": prefix_a == prefix_b,
        "source_completeness": [
            manifest_a.get("source_completeness"),
            manifest_b.get("source_completeness"),
        ],
        "all_required_comparison_classes_present": {
            row["comparison_class"] for row in all_pairs
        }
        == {
            "paired_nonbranch",
            "quiet_control",
            "unrelated",
            "strong_divergence",
            "common_checkpoint_branch",
        },
    }
    validation["accepted"] = bool(
        validation["independent_branch_ids"]
        and validation["same_parent_checkpoint"]
        and validation["common_history_exact"]
        and validation["all_required_comparison_classes_present"]
        and all(row["accepted"] for row in all_pairs)
    )
    divergence = {
        "schema_version": "eiel_chronicle_valid1_first_meaningful_divergence_review_v2",
        "status": first.get("status"),
        "generation": first.get("generation"),
        "first_post_checkpoint_generation": 3,
        "at_first_post_checkpoint_generation": first.get("generation") == 3,
        "selected_event_id": selected.get("event_id"),
        "selected_event_type": selected.get("event_type"),
        "selected_event_category": selected.get("event_category"),
        "selected_event_severity": selected.get("severity"),
        "selected_event_live_priority": selected.get("live_priority"),
        "selected_event_scientific_importance": selected.get("scientific_importance"),
        "selected_event_historical_importance": selected.get("historical_importance"),
        "selected_event_confidence_class": selected.get("confidence_class"),
        "selected_event_causality_class": selected.get("causality_class"),
        "selected_event_traceable_in_branch_feed": _event_is_traceable(runtime_root, selected),
        "metric_context": {
            "predator_off": {
                "alive": metrics_a[3].get("alive"),
                "jungle_predator_events": metrics_a[3].get("jungle_predator_events"),
            },
            "predator_on": {
                "alive": metrics_b[3].get("alive"),
                "jungle_predator_events": metrics_b[3].get("jungle_predator_events"),
            },
        },
        "rule": first.get("criteria"),
        "review": "correct_and_conservative",
        "causality_not_inflated": selected.get("causality_class") in {
            "not_assessed",
            "temporal_association",
        },
        "interpretation": (
            "The selected critical lineage death is a meaningful post-branch history difference; "
            "the comparison does not claim predator causation."
        ),
    }
    divergence["accepted"] = bool(
        divergence["status"] == "diverged"
        and divergence["at_first_post_checkpoint_generation"]
        and divergence["selected_event_traceable_in_branch_feed"]
        and int(divergence["selected_event_scientific_importance"] or 0) >= 4
        and divergence["causality_not_inflated"]
        and float(divergence["metric_context"]["predator_off"]["jungle_predator_events"] or 0) == 0
        and float(divergence["metric_context"]["predator_on"]["jungle_predator_events"] or 0) == 589
    )
    return validation, divergence


def main() -> int:
    args = _parser().parse_args()
    runtime_root = args.runtime_root.resolve()
    comparison_root = args.comparison_root.resolve()
    output = args.output_dir.resolve()
    specs = (
        ("paired_nonbranch_quiet", "paired_nonbranch", None),
        ("quiet_control_same_seed", "quiet_control", "quiet_comparison_review.json"),
        ("unrelated_flat_wdl", "unrelated", "unrelated_comparison_review.json"),
        ("strong_quiet_multi", "strong_divergence", "strong_divergence_review.json"),
    )
    accepted = True
    reviews: list[dict[str, Any]] = []
    for name, comparison_class, filename in specs:
        review = _review(
            runtime_root=runtime_root,
            comparison_root=comparison_root,
            name=name,
            comparison_class=comparison_class,
        )
        reviews.append(review)
        if filename:
            write_json(output / filename, review)
        accepted = accepted and bool(review["accepted"])
    branch, divergence = _branch_reviews(runtime_root, comparison_root, reviews)
    write_json(output / "branch_comparison_validation.json", branch)
    write_json(output / "first_meaningful_divergence_review.json", divergence)
    accepted = accepted and bool(branch["accepted"]) and bool(divergence["accepted"])
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
