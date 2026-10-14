from __future__ import annotations

"""Structured run/branch comparison for Scientific Chronicle V1."""

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from simulation.chronicle.schema import COMPARISON_SCHEMA_VERSION, canonical_json, canonical_payload, event_order_key
from simulation.chronicle.storage import ARTIFACT_NAMES, atomic_write_text, read_json, read_jsonl, write_json


def _chronicle_root(path: Path | str) -> Path:
    root = Path(path).resolve()
    return root / "chronicle" if (root / "chronicle" / ARTIFACT_NAMES["events"]).exists() else root


def _episode_key(episode: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    return (
        str(episode.get("episode_type", "")),
        str(episode.get("affected_region") or "global"),
        tuple(sorted(str(item) for item in episode.get("lineage_ids", []) or [])),
    )


def _event_identity(event: Mapping[str, Any]) -> str:
    return canonical_json(
        {
            "generation": int(event.get("generation", 0)),
            "event_type": str(event.get("event_type", "")),
            "event_category": str(event.get("event_category", "")),
            "subject_type": str(event.get("subject_type", "")),
            "subject_id": event.get("subject_id"),
            "lineage_ids": list(event.get("lineage_ids", []) or []),
            "region_id": event.get("region_id"),
            "severity": str(event.get("severity", "")),
            "live_priority": str(event.get("live_priority", "")),
            "scientific_importance": int(event.get("scientific_importance", 0)),
            "historical_importance": int(event.get("historical_importance", 0)),
            "confidence_class": str(event.get("confidence_class", "")),
            "causality_class": str(event.get("causality_class", "")),
        }
    )


def _is_meaningful_event(event: Mapping[str, Any]) -> bool:
    if event.get("event_type") == "generation_summary":
        return False
    return bool(
        int(event.get("scientific_importance", 0)) >= 4
        or int(event.get("historical_importance", 0)) >= 4
        or str(event.get("severity", "")) in {"major", "critical"}
        or str(event.get("live_priority", "")) in {"high", "critical"}
    )


def _material_metric_differences(
    event_a: Mapping[str, Any] | None,
    event_b: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if event_a is None or event_b is None:
        return [{"metric": "generation_summary_presence", "value_a": event_a is not None, "value_b": event_b is not None}]
    values_a = dict(event_a.get("measurements", {}) or {})
    values_b = dict(event_b.get("measurements", {}) or {})
    count_metrics = {
        "alive",
        "births",
        "deaths",
        "failed_births",
        "active_lineages",
        "cohort_size",
        "predator_events",
        "jungle_predator_events",
        "jungle_predator_deaths",
        "delayed_hazard_events",
        "wdl_delayed_hazard_event_count",
    }
    fraction_metrics = {
        "cohort_survival_fraction",
        "food_depleted_fraction",
        "patch_food_depleted_frac",
        "crowding_hotspot_fraction",
        "patch_density_hotspot_frac",
        "jungle_predator_coverage_fraction",
    }
    differences: list[dict[str, Any]] = []
    for metric in sorted(count_metrics | fraction_metrics):
        raw_a = values_a.get(metric)
        raw_b = values_b.get(metric)
        if raw_a is None and raw_b is None:
            continue
        try:
            value_a = float(raw_a or 0.0)
            value_b = float(raw_b or 0.0)
        except (TypeError, ValueError):
            continue
        delta = abs(value_b - value_a)
        if metric in fraction_metrics:
            material = delta + 1e-12 >= 0.10
            rule = "absolute_delta_gte_0.10"
        else:
            zero_crossing = (value_a == 0.0) != (value_b == 0.0)
            relative = delta / max(abs(value_a), abs(value_b), 1.0)
            material = zero_crossing or (delta >= 2.0 and relative + 1e-12 >= 0.10)
            rule = "zero_crossing_or_absolute_delta_gte_2_and_relative_delta_gte_0.10"
        if material:
            differences.append(
                {
                    "metric": metric,
                    "value_a": raw_a,
                    "value_b": raw_b,
                    "absolute_delta": delta,
                    "rule": rule,
                }
            )
    return differences


def _first_meaningful_divergence(
    events_a: list[dict[str, Any]],
    events_b: list[dict[str, Any]],
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": "eiel_chronicle_first_divergence_v1",
        "status": "no_meaningful_divergence",
        "index": None,
        "generation": None,
        "event_a": None,
        "event_b": None,
        "metric_differences": [],
        "criteria": {
            "event": "scientific_or_historical_importance_gte_4_or_major_severity_or_high_live_priority",
            "event_comparison": "per_generation_multiset_of_structured_event_and_classification_identity_without_measurements_or_rendered_text",
            "metric": "selected_generation_summary_metrics_with_material_absolute_and_relative_thresholds",
        },
        "reason": "no unequal meaningful event identity or material generation-summary metric was found",
    }
    meaningful_a: dict[int, list[dict[str, Any]]] = {}
    meaningful_b: dict[int, list[dict[str, Any]]] = {}
    summaries_a: dict[int, dict[str, Any]] = {}
    summaries_b: dict[int, dict[str, Any]] = {}
    for event in events_a:
        generation = int(event.get("generation", 0))
        if event.get("event_type") == "generation_summary":
            summaries_a[generation] = event
        elif _is_meaningful_event(event):
            meaningful_a.setdefault(generation, []).append(event)
    for event in events_b:
        generation = int(event.get("generation", 0))
        if event.get("event_type") == "generation_summary":
            summaries_b[generation] = event
        elif _is_meaningful_event(event):
            meaningful_b.setdefault(generation, []).append(event)

    generations = sorted(set(meaningful_a) | set(meaningful_b) | set(summaries_a) | set(summaries_b))
    for generation in generations:
        events_at_a = meaningful_a.get(generation, [])
        events_at_b = meaningful_b.get(generation, [])
        counts_a = Counter(_event_identity(event) for event in events_at_a)
        counts_b = Counter(_event_identity(event) for event in events_at_b)
        if counts_a != counts_b:
            only_a = counts_a - counts_b
            only_b = counts_b - counts_a
            event_a = next((event for event in events_at_a if _event_identity(event) in only_a), None)
            event_b = next((event for event in events_at_b if _event_identity(event) in only_b), None)
            return {
                **base,
                "status": "diverged",
                "index": generation,
                "generation": generation,
                "event_a": event_a,
                "event_b": event_b,
                "reason": "first generation with an unequal multiset of meaningful structured event identities",
            }
        metric_differences = _material_metric_differences(summaries_a.get(generation), summaries_b.get(generation))
        if metric_differences:
            return {
                **base,
                "status": "diverged",
                "index": generation,
                "generation": generation,
                "event_a": summaries_a.get(generation),
                "event_b": summaries_b.get(generation),
                "metric_differences": metric_differences,
                "reason": "first generation with a material selected generation-summary metric difference",
            }
    return base


def _load(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    events = sorted(read_jsonl(root / ARTIFACT_NAMES["events"]), key=event_order_key)
    episodes_payload = read_json(root / ARTIFACT_NAMES["episodes"])
    episodes = [dict(item) for item in episodes_payload] if isinstance(episodes_payload, list) else []
    manifest_path = root / ARTIFACT_NAMES["manifest"]
    manifest = dict(read_json(manifest_path)) if manifest_path.exists() else {}
    return events, episodes, manifest


def compare_chronicles(run_a: Path | str, run_b: Path | str, output_dir: Path | str) -> dict[str, Any]:
    root_a = _chronicle_root(run_a)
    root_b = _chronicle_root(run_b)
    output_root = Path(output_dir).resolve()
    events_a, episodes_a, manifest_a = _load(root_a)
    events_b, episodes_b, manifest_b = _load(root_b)
    unmatched_b = set(range(len(episodes_b)))
    matched: list[dict[str, Any]] = []
    unmatched_a_rows: list[dict[str, Any]] = []
    for episode_a in episodes_a:
        candidates = [
            index
            for index in unmatched_b
            if _episode_key(episodes_b[index]) == _episode_key(episode_a)
        ]
        if not candidates:
            unmatched_a_rows.append(episode_a)
            continue
        chosen = min(
            candidates,
            key=lambda index: (
                abs(int(episodes_b[index].get("start_generation", 0)) - int(episode_a.get("start_generation", 0))),
                str(episodes_b[index].get("episode_id", "")),
            ),
        )
        episode_b = episodes_b[chosen]
        unmatched_b.remove(chosen)
        matched.append(
            {
                "episode_type": str(episode_a.get("episode_type", "")),
                "episode_a_id": str(episode_a.get("episode_id", "")),
                "episode_b_id": str(episode_b.get("episode_id", "")),
                "start_generation_delta_b_minus_a": int(episode_b.get("start_generation", 0)) - int(episode_a.get("start_generation", 0)),
                "duration_a": int(episode_a.get("end_generation", 0)) - int(episode_a.get("start_generation", 0)) + 1,
                "duration_b": int(episode_b.get("end_generation", 0)) - int(episode_b.get("start_generation", 0)) + 1,
                "peak_severity_a": str(episode_a.get("peak_severity", "")),
                "peak_severity_b": str(episode_b.get("peak_severity", "")),
                "lineage_ids_a": list(episode_a.get("lineage_ids", []) or []),
                "lineage_ids_b": list(episode_b.get("lineage_ids", []) or []),
                "outcome_a": str(episode_a.get("outcome", "")),
                "outcome_b": str(episode_b.get("outcome", "")),
            }
        )
    unmatched_b_rows = [episodes_b[index] for index in sorted(unmatched_b)]

    first_divergence = _first_meaningful_divergence(events_a, events_b)

    summaries_a = {int(event["generation"]): event for event in events_a if event.get("event_type") == "generation_summary"}
    summaries_b = {int(event["generation"]): event for event in events_b if event.get("event_type") == "generation_summary"}
    metric_context = {
        "schema_version": "eiel_chronicle_key_metric_context_v1",
        "generations": [
            {
                "generation": generation,
                "measurements_a": dict(summaries_a.get(generation, {}).get("measurements", {}) or {}),
                "measurements_b": dict(summaries_b.get(generation, {}).get("measurements", {}) or {}),
            }
            for generation in sorted(set(summaries_a).union(summaries_b))
        ],
    }
    same_checkpoint = bool(
        manifest_a.get("parent_checkpoint_id")
        and manifest_a.get("parent_checkpoint_id") == manifest_b.get("parent_checkpoint_id")
    )
    limitations: list[str] = []
    if manifest_a.get("source_completeness") != "live_generation_boundary_complete" or manifest_b.get("source_completeness") != "live_generation_boundary_complete":
        limitations.append("At least one Chronicle reports incomplete or retrospective-only sources.")
    if not same_checkpoint:
        limitations.append("No shared parent checkpoint identity was established; divergence is sequence-based only.")
    summary = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "run_a_id": str(manifest_a.get("run_id") or (events_a[0].get("run_id") if events_a else root_a.name)),
        "run_b_id": str(manifest_b.get("run_id") or (events_b[0].get("run_id") if events_b else root_b.name)),
        "common_parent_checkpoint": same_checkpoint,
        "parent_checkpoint_id": manifest_a.get("parent_checkpoint_id") if same_checkpoint else None,
        "matched_episode_count": len(matched),
        "episodes_only_in_a": len(unmatched_a_rows),
        "episodes_only_in_b": len(unmatched_b_rows),
        "first_meaningful_divergence": first_divergence,
        "source_completeness_a": manifest_a.get("source_completeness", "unknown"),
        "source_completeness_b": manifest_b.get("source_completeness", "unknown"),
        "limitations": limitations,
    }
    write_json(output_root / "comparison_summary.json", summary)
    write_json(output_root / "matched_episodes.json", matched)
    write_json(
        output_root / "unmatched_episodes.json",
        {"only_in_a": unmatched_a_rows, "only_in_b": unmatched_b_rows},
    )
    write_json(output_root / "first_meaningful_divergence.json", first_divergence)
    write_json(output_root / "key_metric_context.json", metric_context)
    report = "\n".join(
        (
            "# Scientific Chronicle Comparison",
            "",
            f"- Run A: `{summary['run_a_id']}`",
            f"- Run B: `{summary['run_b_id']}`",
            f"- Common parent checkpoint: `{str(same_checkpoint).lower()}`",
            f"- Matched episodes: {len(matched)}",
            f"- Episodes only in A: {len(unmatched_a_rows)}",
            f"- Episodes only in B: {len(unmatched_b_rows)}",
            f"- First meaningful divergence: `{first_divergence['status']}` at index `{first_divergence['index']}`",
            "",
            "## Limitations",
            "",
            *(f"- {item}" for item in limitations or ["No additional comparison limitations recorded."]),
            "",
        )
    )
    atomic_write_text(output_root / "comparison_report.md", report)
    return canonical_payload(summary)
