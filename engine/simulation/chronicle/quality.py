from __future__ import annotations

"""Deterministic episode-quality review over persisted Chronicle artifacts."""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.chronicle.episodes import EVENT_EPISODE_TYPES, detect_episodes
from simulation.chronicle.schema import canonical_json, canonical_payload, event_order_key
from simulation.chronicle.storage import ARTIFACT_NAMES, read_json, read_jsonl, write_json


EPISODE_CONTRACTS: dict[str, dict[str, Any]] = {
    "regional_food_depletion": {
        "intended_meaning": "A bounded interval in which the depleted-patch fraction remains above the configured depletion condition.",
        "trigger_event_types": ["regional_food_depletion"],
        "closing_event_types": ["regional_food_recovery"],
        "opening_rule": "Open on a persisted regional_food_depletion threshold event.",
        "continuation_rule": "Continue while depleted fraction remains above the recovery threshold.",
        "closing_rule": "Close at the first generation meeting the configured food-recovery threshold, or at run finalization.",
        "minimum_duration_generations": 1,
        "minimum_magnitude": "patch_food_depleted_frac >= food_depleted_fraction_threshold",
        "identity_rule": "region_id, falling back to patch coordinate or global",
        "overlap_rule": "May overlap ecology, hazard, and population episodes; same-region intervals may not overlap each other.",
        "confidence_default": "derived",
        "causality_default": "temporal_association",
        "source_completeness_requirement": "generation summaries plus the opening threshold event",
        "false_positive_risk": "A one-generation threshold excursion can look like a sustained shortage when persistence is configured to one.",
        "false_negative_risk": "Suppression budgets can omit a threshold event even when metrics cross the threshold.",
    },
    "crowding_waste_crisis": {
        "intended_meaning": "A bounded interval with crowding, occupied-patch waste, or direct regrowth suppression above a configured condition.",
        "trigger_event_types": ["crowding_hotspot", "waste_hotspot", "regrowth_suppression"],
        "closing_event_types": [],
        "opening_rule": "Open on the first persisted crowding, waste, or regrowth-suppression event for a region.",
        "continuation_rule": "Continue while any recorded crisis component remains active.",
        "closing_rule": "Close when all recorded crisis components clear, or at run finalization.",
        "minimum_duration_generations": 1,
        "minimum_magnitude": "crowding or waste threshold, or positive regrowth suppression",
        "identity_rule": "region_id, falling back to patch coordinate or global",
        "overlap_rule": "Crowding, waste, and regrowth events for one region merge; different regions do not merge.",
        "confidence_default": "derived",
        "causality_default": "temporal_association",
        "source_completeness_requirement": "generation summaries and at least one crisis trigger event",
        "false_positive_risk": "Unrelated global crowding and waste signals can be grouped when no region identity is available.",
        "false_negative_risk": "A suppressed component can make an interval look shorter than the underlying metrics.",
    },
    "predator_pressure_wave": {
        "intended_meaning": "A bounded interval with predator-event volume at or above the dynamic population-scaled threshold.",
        "trigger_event_types": ["predator_pressure_spike", "predator_damage"],
        "closing_event_types": [],
        "opening_rule": "Open on a persisted predator-pressure spike.",
        "continuation_rule": "Continue while predator events remain at or above the recorded dynamic threshold.",
        "closing_rule": "Close on the first below-threshold generation, or at run finalization.",
        "minimum_duration_generations": 1,
        "minimum_magnitude": "predator_events >= ceil(max(1, alive) * 0.05)",
        "identity_rule": "region_id, falling back to patch coordinate or global",
        "overlap_rule": "Different regions remain separate; a global wave may overlap lineage and bottleneck episodes.",
        "confidence_default": "derived",
        "causality_default": "temporal_association",
        "source_completeness_requirement": "predator metrics plus the opening event; exact agent attribution needs typed or Python context rows",
        "false_positive_risk": "A high event count with negligible damage still qualifies as pressure, not mortality.",
        "false_negative_risk": "Aggregate-only native coverage cannot validate agent-level participants.",
    },
    "ecological_recovery": {
        "intended_meaning": "A bounded interval of positive ecological-recovery activity or an explicit food-depletion recovery transition.",
        "trigger_event_types": ["regional_food_recovery", "ecological_recovery"],
        "closing_event_types": [],
        "opening_rule": "Open on an explicit food recovery or positive recovery-activity event.",
        "continuation_rule": "Continue while positive recovery activity remains recorded.",
        "closing_rule": "Close when positive recovery activity clears; a transition-only recovery closes at its trigger generation.",
        "minimum_duration_generations": 1,
        "minimum_magnitude": "positive recovery total, or a configured food-recovery crossing",
        "identity_rule": "region_id, falling back to patch coordinate or global",
        "overlap_rule": "May overlap the closing boundary of food depletion and other ecological episodes.",
        "confidence_default": "derived",
        "causality_default": "temporal_association",
        "source_completeness_requirement": "recovery metrics plus at least one recovery trigger event",
        "false_positive_risk": "Positive activity does not by itself prove population or ecosystem restoration.",
        "false_negative_risk": "Recovery dimensions absent from compact metrics cannot be reconstructed.",
    },
    "lineage_expansion": {
        "intended_meaning": "A lineage-specific appearance, rapid share increase, or dominance crossing.",
        "trigger_event_types": ["lineage_appearance", "lineage_dominance_crossing", "lineage_rapid_share_increase"],
        "closing_event_types": [],
        "opening_rule": "Open on a lineage-specific expansion transition.",
        "continuation_rule": "Combine same-lineage expansion transitions separated by at most the configured episode gap.",
        "closing_rule": "Close after the event gap or run finalization.",
        "minimum_duration_generations": 1,
        "minimum_magnitude": "the frozen event threshold that produced the source event",
        "identity_rule": "biological lineage ID",
        "overlap_rule": "Different lineages never merge; expansion may overlap decline only as separately linked episodes.",
        "confidence_default": "derived",
        "causality_default": "temporal_association",
        "source_completeness_requirement": "lineage-resolved generation state; unavailable under aggregate-only native lifecycle coverage",
        "false_positive_risk": "Short-lived share increases can be historically unimportant even when threshold-valid.",
        "false_negative_risk": "Aggregate-only native state cannot support a lineage-resolved judgment.",
    },
    "lineage_decline": {
        "intended_meaning": "A lineage-specific rapid share decline or transition into the near-extinction count.",
        "trigger_event_types": ["lineage_near_extinction", "lineage_rapid_share_decline"],
        "closing_event_types": [],
        "opening_rule": "Open on a lineage-specific decline transition.",
        "continuation_rule": "Combine same-lineage decline transitions separated by at most the configured episode gap.",
        "closing_rule": "Close after the event gap, on extinction as a related episode, or at run finalization.",
        "minimum_duration_generations": 1,
        "minimum_magnitude": "rapid-share delta or near-extinction count threshold",
        "identity_rule": "biological lineage ID",
        "overlap_rule": "Different lineages never merge; extinction is retained as a separate terminal family.",
        "confidence_default": "derived",
        "causality_default": "temporal_association",
        "source_completeness_requirement": "lineage-resolved generation state; unavailable under aggregate-only native lifecycle coverage",
        "false_positive_risk": "A small lineage can cross share thresholds because the total population changed.",
        "false_negative_risk": "Gradual decline below the rapid threshold is intentionally not an episode trigger.",
    },
    "lineage_extinction_sequence": {
        "intended_meaning": "The terminal disappearance of one identified lineage, kept separate from other lineages.",
        "trigger_event_types": ["lineage_extinction"],
        "closing_event_types": [],
        "opening_rule": "Open on the lineage extinction event.",
        "continuation_rule": "No continuation is required after the terminal event.",
        "closing_rule": "Close at the extinction event generation.",
        "minimum_duration_generations": 1,
        "minimum_magnitude": "previous living count > 0 and current living count = 0",
        "identity_rule": "biological lineage ID",
        "overlap_rule": "Each lineage remains a separate terminal episode even when several disappear in one generation.",
        "confidence_default": "derived",
        "causality_default": "temporal_association",
        "source_completeness_requirement": "lineage-resolved generation state; unavailable under aggregate-only native lifecycle coverage",
        "false_positive_risk": "Temporary absence caused by incomplete lineage projection would be misleading.",
        "false_negative_risk": "Aggregate-only native state cannot establish lineage disappearance.",
    },
    "population_bottleneck": {
        "intended_meaning": "A bounded interval with cohort survival at or below the configured bottleneck threshold.",
        "trigger_event_types": ["population_bottleneck"],
        "closing_event_types": [],
        "opening_rule": "Open on a persisted population-bottleneck event.",
        "continuation_rule": "Continue while cohort survival remains at or below threshold.",
        "closing_rule": "Close on the first above-threshold generation, or at run finalization.",
        "minimum_duration_generations": 1,
        "minimum_magnitude": "cohort_survival_rate <= bottleneck_survival_fraction",
        "identity_rule": "global population or explicit region",
        "overlap_rule": "May overlap ecological, predator, hazard, and lineage episodes.",
        "confidence_default": "derived",
        "causality_default": "temporal_association",
        "source_completeness_requirement": "cohort size/survivor metrics and the opening event",
        "false_positive_risk": "The event describes cohort survival, not a guaranteed long-term genetic bottleneck.",
        "false_negative_risk": "A later refill can hide within-generation mortality if cohort metrics are missing.",
    },
    "delayed_hazard_cycle": {
        "intended_meaning": "A bounded interval with delayed-hazard consequence events in the WDL substrate.",
        "trigger_event_types": ["delayed_hazard_cycle"],
        "closing_event_types": [],
        "opening_rule": "Open on a persisted generation with one or more delayed-hazard events.",
        "continuation_rule": "Continue while delayed-hazard event count remains positive.",
        "closing_rule": "Close on the first zero-event generation, or at run finalization.",
        "minimum_duration_generations": 1,
        "minimum_magnitude": "wdl_delayed_hazard_event_count > 0",
        "identity_rule": "region_id, falling back to global population",
        "overlap_rule": "Different regions remain separate and may overlap ecological or population episodes.",
        "confidence_default": "derived",
        "causality_default": "temporal_association",
        "source_completeness_requirement": "WDL generation metrics plus the opening event",
        "false_positive_risk": "Event presence does not imply that the hazard selected for any adaptive response.",
        "false_negative_risk": "A disabled or metrics-only WDL surface cannot recover exact context participants.",
    },
}


GLOBAL_INTERVAL_FAMILIES = frozenset(
    {
        "regional_food_depletion",
        "crowding_waste_crisis",
        "predator_pressure_wave",
        "ecological_recovery",
        "population_bottleneck",
        "delayed_hazard_cycle",
    }
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _load_metrics(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return {int(_number(row.get("generation"), index)): row for index, row in enumerate(rows)}


def _episode_group_from_event(event: Mapping[str, Any], family: str) -> str:
    if family.startswith("lineage_"):
        lineages = list(event.get("lineage_ids", []) or [])
        return str(lineages[0] if lineages else event.get("subject_id") or "unknown_lineage")
    if event.get("region_id"):
        return str(event["region_id"])
    coordinates = event.get("patch_coordinates")
    if isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
        return f"patch:{int(coordinates[0])},{int(coordinates[1])}"
    return "global"


def _source_supports_episode_judgment(family: str, source_completeness: str) -> bool:
    return not (
        family.startswith("lineage_")
        and "aggregate" in str(source_completeness)
    )


def _manifest_config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / ARTIFACT_NAMES["manifest"]
    manifest = dict(read_json(manifest_path)) if manifest_path.exists() else {}
    return manifest, dict(manifest.get("config", {}) or {})


def _active(family: str, row: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    if family == "regional_food_depletion":
        return _number(row.get("patch_food_depleted_frac")) > _number(
            config.get("food_recovery_fraction_threshold"), 0.10
        )
    if family == "crowding_waste_crisis":
        return (
            _number(row.get("patch_density_hotspot_frac"))
            >= _number(config.get("crowding_hotspot_fraction_threshold"), 0.20)
            or _number(row.get("occupied_patch_waste_mean"))
            >= _number(config.get("waste_hotspot_threshold"), 0.25)
            or _number(row.get("patch_regrowth_suppressed_total"))
            > _number(config.get("min_magnitude"), 0.0)
        )
    if family == "predator_pressure_wave":
        alive = max(1, int(_number(row.get("alive"), 1)))
        return int(_number(row.get("jungle_predator_events"))) >= max(1, int(math.ceil(alive * 0.05)))
    if family == "ecological_recovery":
        return _number(row.get("patch_recovery_positive_total")) > _number(config.get("min_magnitude"), 0.0)
    if family == "population_bottleneck":
        return _number(row.get("cohort_survival_rate"), 1.0) <= _number(
            config.get("bottleneck_survival_fraction"), 0.50
        )
    if family == "delayed_hazard_cycle":
        return int(_number(row.get("wdl_delayed_hazard_event_count"))) > 0
    return False


def _expected_interval_end(
    family: str,
    start_generation: int,
    metrics: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
    opening_event_type: str,
) -> int:
    if family not in GLOBAL_INTERVAL_FAMILIES:
        return int(start_generation)
    generations = [generation for generation in sorted(metrics) if generation >= int(start_generation)]
    if not generations:
        return int(start_generation)
    if family == "ecological_recovery" and opening_event_type == "regional_food_recovery":
        if not _active(family, metrics.get(start_generation, {}), config):
            return int(start_generation)
    observed_active = False
    last_active = int(start_generation)
    for generation in generations:
        active = _active(family, metrics[generation], config)
        if active:
            observed_active = True
            last_active = int(generation)
            continue
        if observed_active or generation > int(start_generation):
            return int(generation)
    return int(last_active if observed_active else start_generation)


def _opening_metric_valid(
    family: str,
    event: Mapping[str, Any],
    metrics: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> bool:
    event_type = str(event.get("event_type", ""))
    measurements = dict(event.get("measurements", {}) or {})
    if event_type not in EPISODE_CONTRACTS[family]["trigger_event_types"]:
        return False
    if family.startswith("lineage_"):
        if not event.get("lineage_ids"):
            return False
        if event_type == "lineage_appearance":
            return _number(measurements.get("alive_count")) > 0.0
        if event_type == "lineage_dominance_crossing":
            threshold = _number(
                measurements.get("threshold"), config.get("dominance_threshold", 0.50)
            )
            return (
                _number(measurements.get("share_before")) < threshold
                <= _number(measurements.get("share_after"))
            )
        if event_type == "lineage_rapid_share_increase":
            return _number(measurements.get("share_delta")) >= _number(
                config.get("rapid_lineage_share_delta"), 0.15
            )
        if event_type == "lineage_rapid_share_decline":
            return _number(measurements.get("share_delta")) <= -_number(
                config.get("rapid_lineage_share_delta"), 0.15
            )
        if event_type == "lineage_near_extinction":
            threshold = int(_number(config.get("near_extinction_count"), 1))
            return (
                int(_number(measurements.get("alive_before"))) > threshold
                and 0 < int(_number(measurements.get("alive_after"))) <= threshold
            )
        if event_type == "lineage_extinction":
            return (
                int(_number(measurements.get("previous_alive_count"))) > 0
                and int(_number(measurements.get("alive_count"))) == 0
            )
        return False
    if family == "ecological_recovery" and event_type == "regional_food_recovery":
        return _number(measurements.get("depleted_fraction_after"), 1.0) <= _number(
            measurements.get("threshold"), config.get("food_recovery_fraction_threshold", 0.10)
        )
    row = metrics.get(int(event.get("generation", 0)), {})
    if family == "regional_food_depletion":
        threshold = _number(config.get("food_depleted_fraction_threshold"), 0.25)
        return (
            _number(row.get("patch_food_depleted_frac")) >= threshold
            and _number(measurements.get("depleted_fraction_after")) >= threshold
        )
    if family == "crowding_waste_crisis":
        if event_type == "crowding_hotspot":
            return _number(row.get("patch_density_hotspot_frac")) >= _number(
                config.get("crowding_hotspot_fraction_threshold"), 0.20
            )
        if event_type == "waste_hotspot":
            return _number(row.get("occupied_patch_waste_mean")) >= _number(
                config.get("waste_hotspot_threshold"), 0.25
            )
        if event_type == "regrowth_suppression":
            return _number(row.get("patch_regrowth_suppressed_total")) > _number(
                config.get("min_magnitude"), 0.0
            )
        return False
    return _active(family, row, config)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _interval_openings(
    family: str,
    metrics: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[int]:
    openings: list[int] = []
    prior = False
    for generation, row in sorted(metrics.items()):
        if family == "regional_food_depletion":
            depleted = _number(row.get("patch_food_depleted_frac"))
            active = (
                depleted > _number(config.get("food_recovery_fraction_threshold"), 0.10)
                if prior
                else depleted >= _number(config.get("food_depleted_fraction_threshold"), 0.25)
            )
        else:
            active = _active(family, row, config)
        if active and not prior:
            openings.append(int(generation))
        prior = active
    return openings


def _live_retrospective_review(events: list[dict[str, Any]], stored: list[dict[str, Any]], gap: int) -> dict[str, Any]:
    finalized = detect_episodes(events, gap_generations=gap, finalize=True)
    stored_by_id = {str(item["episode_id"]): item for item in stored}
    final_by_id = {str(item["episode_id"]): item for item in finalized}
    stored_matches = canonical_json(stored) == canonical_json(finalized)
    identity_mismatches = 0
    source_membership_mismatches = 0
    provisional_extension_errors = 0
    order_drift = 0
    lost_episodes = 0
    duplicate_provisional = 0
    expected_finalizations = 0
    final_order = [str(item["episode_id"]) for item in finalized]
    for generation in sorted({int(event.get("generation", 0)) for event in events}):
        prefix = [event for event in events if int(event.get("generation", 0)) <= generation]
        provisional = detect_episodes(prefix, gap_generations=gap, finalize=False)
        provisional_ids = [str(item["episode_id"]) for item in provisional]
        duplicate_provisional += len(provisional_ids) - len(set(provisional_ids))
        expected_order = [episode_id for episode_id in final_order if episode_id in set(provisional_ids)]
        if provisional_ids != expected_order:
            order_drift += 1
        expected_started = {
            str(item["episode_id"])
            for item in finalized
            if int(item.get("start_generation", 0)) <= generation
        }
        lost_episodes += len(expected_started.difference(provisional_ids))
        for episode in provisional:
            episode_id = str(episode["episode_id"])
            final_episode = final_by_id.get(episode_id)
            if final_episode is None:
                identity_mismatches += 1
                continue
            prefix_ids = set(str(value) for value in episode.get("source_event_ids", []) or [])
            final_ids = set(str(value) for value in final_episode.get("source_event_ids", []) or [])
            if not prefix_ids.issubset(final_ids):
                source_membership_mismatches += 1
            if (
                str(episode.get("episode_type")) in GLOBAL_INTERVAL_FAMILIES
                and str(episode.get("status")) == "open"
                and int(final_episode.get("end_generation", 0)) >= generation
                and int(episode.get("end_generation", 0)) != generation
            ):
                provisional_extension_errors += 1
            if episode.get("status") == "open" and final_episode.get("status") == "closed":
                expected_finalizations += 1
    return canonical_payload(
        {
            "stored_final_matches_recomputed_final": stored_matches,
            "stored_episode_count": len(stored_by_id),
            "recomputed_episode_count": len(final_by_id),
            "identity_mismatch_count": identity_mismatches,
            "source_membership_mismatch_count": source_membership_mismatches,
            "provisional_extension_error_count": provisional_extension_errors,
            "order_drift_count": order_drift,
            "lost_episode_count": lost_episodes,
            "duplicate_provisional_count": duplicate_provisional,
            "expected_finalization_count": expected_finalizations,
            "confidence_inflation_count": sum(
                1 for item in finalized if str(item.get("confidence_class")) != "derived"
            ),
            "causality_inflation_count": sum(
                1 for item in finalized if str(item.get("causality_class")) != "temporal_association"
            ),
        }
    )


def analyze_validation_matrix(matrix_path: Path | str, runs_root: Path | str, output_dir: Path | str) -> dict[str, Any]:
    matrix = dict(read_json(Path(matrix_path)))
    root = Path(runs_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cells = [dict(cell) for cell in matrix.get("cells", []) or []]
    inventory: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    live_rows: list[dict[str, Any]] = []
    completions: list[dict[str, Any]] = []
    completeness_counts: Counter[str] = Counter()
    family_run_presence: dict[str, set[str]] = defaultdict(set)
    family_scenario_presence: dict[str, set[str]] = defaultdict(set)
    family_seed_presence: dict[str, set[int]] = defaultdict(set)
    negative_control_evidence: dict[tuple[str, str], dict[str, Any]] = {}
    live_by_run: dict[str, dict[str, Any]] = {}

    for cell in sorted(cells, key=lambda item: str(item.get("run_id", ""))):
        run_id = str(cell["run_id"])
        run_dir = root / run_id
        chronicle_root = run_dir / "chronicle"
        required = [chronicle_root / ARTIFACT_NAMES[name] for name in ARTIFACT_NAMES]
        completed = all(path.exists() for path in required) and (run_dir / "metrics.csv").exists()
        completions.append(
            {
                "run_id": run_id,
                "scenario": str(cell.get("scenario", "")),
                "seed": int(cell.get("seed", 0)),
                "completed": completed,
                "missing": [str(path.relative_to(run_dir)) for path in required if not path.exists()],
            }
        )
        if not completed:
            continue
        events = sorted(read_jsonl(chronicle_root / ARTIFACT_NAMES["events"]), key=event_order_key)
        episodes_payload = read_json(chronicle_root / ARTIFACT_NAMES["episodes"])
        episodes = [dict(item) for item in episodes_payload] if isinstance(episodes_payload, list) else []
        manifest, config = _manifest_config(chronicle_root)
        metrics = _load_metrics(run_dir / "metrics.csv")
        completeness = str(manifest.get("source_completeness", "unknown"))
        completeness_counts[completeness] += 1
        event_by_id = {str(event["event_id"]): event for event in events}
        event_types = Counter(str(event.get("event_type", "")) for event in events)
        aggregate_only = "aggregate" in completeness
        for family, contract in EPISODE_CONTRACTS.items():
            trigger_count = sum(event_types[str(event_type)] for event_type in contract["trigger_event_types"])
            metric_openings = (
                _interval_openings(family, metrics, config)
                if family in GLOBAL_INTERVAL_FAMILIES
                else []
            )
            source_supports_judgment = not (
                family.startswith("lineage_") and aggregate_only
            ) and completeness not in {"", "unknown", "metrics_only_partial"}
            negative_control_evidence[(run_id, family)] = {
                "formal_trigger_event_count": trigger_count,
                "independent_metric_opening_count": len(metric_openings),
                "source_completeness": completeness,
                "source_supports_judgment": source_supports_judgment,
                "absence_confirmed": bool(
                    source_supports_judgment
                    and trigger_count == 0
                    and (family not in GLOBAL_INTERVAL_FAMILIES or not metric_openings)
                ),
                "basis": (
                    "no formal trigger events and no independent generation-metric openings"
                    if family in GLOBAL_INTERVAL_FAMILIES
                    else "no formal lineage trigger events under lineage-resolved source coverage"
                ),
            }
        event_episode_membership: dict[str, set[str]] = defaultdict(set)
        for episode in episodes:
            for event_id in episode.get("source_event_ids", []) or []:
                event_episode_membership[str(event_id)].add(str(episode.get("episode_id", "")))

        duplicate_ids = {item for item, count in Counter(str(ep.get("episode_id")) for ep in episodes).items() if count > 1}
        gap = int(config.get("episode_gap_generations", 1) or 1)
        live = _live_retrospective_review(events, episodes, gap)
        live_rows.append({"run_id": run_id, **live})
        live_by_run[run_id] = live

        grouped_episodes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for episode in episodes:
            family = str(episode.get("episode_type", ""))
            if family not in EPISODE_CONTRACTS:
                continue
            source_events = [
                event_by_id[str(event_id)]
                for event_id in episode.get("source_event_ids", []) or []
                if str(event_id) in event_by_id
            ]
            source_events.sort(key=event_order_key)
            opening = source_events[0] if source_events else {}
            group = (
                str(episode.get("lineage_ids", [""])[0])
                if family.startswith("lineage_") and episode.get("lineage_ids")
                else str(episode.get("affected_region") or "global")
            )
            grouped_episodes[(family, group)].append(episode)
            family_run_presence[family].add(run_id)
            family_scenario_presence[family].add(str(cell.get("scenario", "")))
            family_seed_presence[family].add(int(cell.get("seed", 0)))
            trigger_types = set(EPISODE_CONTRACTS[family]["trigger_event_types"])
            support_types = trigger_types.union(EPISODE_CONTRACTS[family].get("closing_event_types", []))
            opening_correct = bool(opening) and str(opening.get("event_type")) in trigger_types
            source_judgment_available = _source_supports_episode_judgment(family, completeness)
            source_sufficient = source_judgment_available and opening_correct and all(
                str(event.get("event_type")) in support_types or str(event.get("event_type")) == "generation_summary"
                for event in source_events
            )
            metric_consistent = bool(opening) and _opening_metric_valid(family, opening, metrics, config)
            expected_end = _expected_interval_end(
                family,
                int(episode.get("start_generation", 0)),
                metrics,
                config,
                str(opening.get("event_type", "")),
            )
            if family.startswith("lineage_") and source_events:
                expected_end = max(int(event.get("generation", 0)) for event in source_events)
            observed_end = int(episode.get("end_generation", 0))
            close_delta = observed_end - expected_end
            closing_correct = close_delta == 0
            temporal = all(
                int(episode.get("start_generation", 0)) <= int(event.get("generation", 0)) <= observed_end
                for event in source_events
            )
            subject_coherent = all(
                _episode_group_from_event(event, family) == group
                for event in source_events
                if str(event.get("event_type")) in trigger_types
            )
            confidence_correct = str(episode.get("confidence_class")) == "derived"
            causality_correct = str(episode.get("causality_class")) == "temporal_association"
            duplicate = str(episode.get("episode_id")) in duplicate_ids
            classification = "correct"
            if not opening_correct or not metric_consistent:
                classification = "false_positive"
            elif close_delta < 0:
                classification = "premature_close"
            elif close_delta > 0:
                classification = "late_close"
            elif not subject_coherent:
                classification = "incorrect_merge"
            elif duplicate:
                classification = "duplicate_episode"
            elif not source_judgment_available:
                classification = "source_incomplete_but_honestly_labeled"
            elif not source_sufficient:
                classification = "insufficient_evidence"
            elif not confidence_correct or not causality_correct:
                classification = "confidence_or_causality_error"
            elif not temporal:
                classification = "temporal_incoherence"
            review = {
                "run_id": run_id,
                "scenario": str(cell.get("scenario", "")),
                "seed": int(cell.get("seed", 0)),
                "episode_id": str(episode.get("episode_id", "")),
                "episode_type": family,
                "group": group,
                "source_event_ids": ";".join(str(value) for value in episode.get("source_event_ids", []) or []),
                "source_event_count": len(source_events),
                "start_generation": int(episode.get("start_generation", 0)),
                "observed_end_generation": observed_end,
                "expected_end_generation": expected_end,
                "opening_correct": opening_correct,
                "closing_correct": closing_correct,
                "temporal_coherence": temporal,
                "subject_or_region_coherence": subject_coherent,
                "source_event_sufficiency": source_sufficient,
                "metric_consistency": metric_consistent,
                "duplicate_episode": duplicate,
                "confidence_label_correct": confidence_correct,
                "causality_label_correct": causality_correct,
                "source_completeness": completeness,
                "source_completeness_honest": completeness not in {"", "unknown"},
                "source_judgment_available": source_judgment_available,
                "overlap_quality": "coherent",
                "split_merge_quality": "coherent",
                "live_retrospective_classification": (
                    "exact_final_match"
                    if live["stored_final_matches_recomputed_final"]
                    and int(live["identity_mismatch_count"]) == 0
                    and int(live["source_membership_mismatch_count"]) == 0
                    else "disagreement"
                ),
                "classification": classification,
            }
            reviews.append(review)
            inventory.append(
                {
                    "run_id": run_id,
                    "scenario": str(cell.get("scenario", "")),
                    "seed": int(cell.get("seed", 0)),
                    "episode_id": str(episode.get("episode_id", "")),
                    "episode_type": family,
                    "status": str(episode.get("status", "")),
                    "start_generation": int(episode.get("start_generation", 0)),
                    "end_generation": observed_end,
                    "source_event_count": len(source_events),
                    "source_event_ids": ";".join(str(value) for value in episode.get("source_event_ids", []) or []),
                    "peak_severity": str(episode.get("peak_severity", "")),
                    "confidence_class": str(episode.get("confidence_class", "")),
                    "causality_class": str(episode.get("causality_class", "")),
                    "source_completeness": completeness,
                }
            )

        suppressed_generations = {
            int(event.get("generation", 0))
            for event in events
            if event.get("event_type") == "chronicle_events_suppressed"
        }
        for family in GLOBAL_INTERVAL_FAMILIES:
            observed_starts = [
                int(episode.get("start_generation", 0))
                for episode in episodes
                if episode.get("episode_type") == family
            ]
            for expected_start in _interval_openings(family, metrics, config):
                if expected_start in observed_starts:
                    continue
                false_negatives.append(
                    {
                        "run_id": run_id,
                        "scenario": str(cell.get("scenario", "")),
                        "seed": int(cell.get("seed", 0)),
                        "episode_type": family,
                        "expected_start_generation": expected_start,
                        "classification": (
                            "source_incomplete_but_honestly_labeled"
                            if expected_start in suppressed_generations
                            else "false_negative"
                        ),
                        "basis": "independent generation-metric condition transition",
                    }
                )
        for event in events:
            family = EVENT_EPISODE_TYPES.get(str(event.get("event_type", "")))
            if family and str(event["event_id"]) not in event_episode_membership:
                false_negatives.append(
                    {
                        "run_id": run_id,
                        "scenario": str(cell.get("scenario", "")),
                        "seed": int(cell.get("seed", 0)),
                        "episode_type": family,
                        "expected_start_generation": int(event.get("generation", 0)),
                        "classification": "false_negative",
                        "basis": f"uncovered formal trigger event {event['event_id']}",
                    }
                )

        for (family, group), selected in grouped_episodes.items():
            ordered = sorted(selected, key=lambda item: (int(item.get("start_generation", 0)), str(item.get("episode_id"))))
            for left, right in zip(ordered, ordered[1:]):
                if int(right.get("start_generation", 0)) <= int(left.get("end_generation", 0)):
                    for review in reviews:
                        if review["episode_id"] in {left["episode_id"], right["episode_id"]}:
                            review["classification"] = "incorrect_split"
                            review["overlap_quality"] = "overlapping_same_identity_intervals"
                            review["split_merge_quality"] = "incorrect_split"

    inventory.sort(key=lambda row: (row["run_id"], row["start_generation"], row["episode_type"], row["episode_id"]))
    reviews.sort(key=lambda row: (row["run_id"], row["start_generation"], row["episode_type"], row["episode_id"]))
    false_negatives.sort(
        key=lambda row: (row["run_id"], row["expected_start_generation"], row["episode_type"], row["classification"])
    )
    classifications = Counter(str(row["classification"]) for row in reviews)
    classifications.update(str(row["classification"]) for row in false_negatives)
    family_summary: dict[str, Any] = {}
    for family in EPISODE_CONTRACTS:
        family_reviews = [row for row in reviews if row["episode_type"] == family]
        family_missing = [row for row in false_negatives if row["episode_type"] == family]
        counts = Counter(str(row["classification"]) for row in family_reviews + family_missing)
        strong = max(
            family_reviews,
            key=lambda row: (
                int(row["observed_end_generation"]) - int(row["start_generation"]),
                int(row["source_event_count"]),
                str(row["episode_id"]),
            ),
            default=None,
        )
        boundary = min(
            family_reviews,
            key=lambda row: (
                int(row["observed_end_generation"]) - int(row["start_generation"]),
                int(row["source_event_count"]),
                str(row["episode_id"]),
            ),
            default=None,
        )
        absent_cells = [
            cell
            for cell in cells
            if not any(
                row["run_id"] == str(cell["run_id"]) and row["episode_type"] == family
                for row in family_reviews
            )
            and negative_control_evidence.get((str(cell["run_id"]), family), {}).get(
                "absence_confirmed"
            )
        ]
        cross_run = []
        for row in family_reviews:
            if not cross_run or row["run_id"] != cross_run[0]["run_id"]:
                cross_run.append(row)
            if len(cross_run) == 2:
                break
        representative_examples = {
            "strong_positive": (
                {
                    "run_id": strong["run_id"],
                    "episode_id": strong["episode_id"],
                    "source_event_ids": strong["source_event_ids"].split(";") if strong["source_event_ids"] else [],
                }
                if strong
                else None
            ),
            "boundary": (
                {
                    "run_id": boundary["run_id"],
                    "episode_id": boundary["episode_id"],
                    "source_event_ids": boundary["source_event_ids"].split(";") if boundary["source_event_ids"] else [],
                }
                if boundary
                else None
            ),
            "negative_or_control": (
                {
                    "run_id": str(absent_cells[0]["run_id"]),
                    "scenario": str(absent_cells[0].get("scenario", "")),
                    **negative_control_evidence[(str(absent_cells[0]["run_id"]), family)],
                }
                if absent_cells
                else {"available": False, "reason": "family occurred in every completed matrix cell"}
            ),
            "cross_run": [
                {"run_id": row["run_id"], "episode_id": row["episode_id"]}
                for row in cross_run
            ],
        }
        live_disagreement_runs = {
            run_id
            for run_id in family_run_presence.get(family, set())
            if not (
                live_by_run[run_id]["stored_final_matches_recomputed_final"]
                and int(live_by_run[run_id]["identity_mismatch_count"]) == 0
                and int(live_by_run[run_id]["source_membership_mismatch_count"]) == 0
            )
        }
        family_summary[family] = {
            "detected_episode_count": len(family_reviews),
            "reviewed_episode_count": len(family_reviews),
            "run_recurrence_count": len(family_run_presence.get(family, set())),
            "scenario_recurrence_count": len(family_scenario_presence.get(family, set())),
            "scenarios": sorted(family_scenario_presence.get(family, set())),
            "seed_recurrence_count": len(family_seed_presence.get(family, set())),
            "seeds": sorted(family_seed_presence.get(family, set())),
            "live_retrospective_disagreement_count": len(live_disagreement_runs),
            "correct_or_acceptable_count": counts["correct"] + counts["acceptable"],
            "false_positive_count": counts["false_positive"],
            "false_negative_count": counts["false_negative"],
            "split_error_count": counts["incorrect_split"],
            "merge_error_count": counts["incorrect_merge"],
            "premature_close_count": counts["premature_close"],
            "late_close_count": counts["late_close"],
            "source_incomplete_count": counts["source_incomplete_but_honestly_labeled"] + counts["insufficient_evidence"],
            "representative_examples": representative_examples,
        }
    all_live_ok = all(
        bool(row["stored_final_matches_recomputed_final"])
        and int(row["identity_mismatch_count"]) == 0
        and int(row["source_membership_mismatch_count"]) == 0
        and int(row["provisional_extension_error_count"]) == 0
        and int(row["order_drift_count"]) == 0
        and int(row["lost_episode_count"]) == 0
        and int(row["duplicate_provisional_count"]) == 0
        and int(row["confidence_inflation_count"]) == 0
        and int(row["causality_inflation_count"]) == 0
        for row in live_rows
    )
    completion_summary = {
        "schema_version": "eiel_chronicle_valid1_run_completion_v1",
        "planned_run_count": len(cells),
        "completed_run_count": sum(1 for row in completions if row["completed"]),
        "failed_or_missing_run_count": sum(1 for row in completions if not row["completed"]),
        "runs": completions,
    }
    quality_summary = {
        "schema_version": "eiel_chronicle_valid1_episode_quality_summary_v1",
        "episode_family_count": len(EPISODE_CONTRACTS),
        "episode_instance_count": len(reviews),
        "reviewed_episode_count": len(reviews),
        "classification_counts": dict(sorted(classifications.items())),
        "families": family_summary,
    }
    false_summary = {
        "schema_version": "eiel_chronicle_valid1_false_positive_negative_summary_v1",
        "false_positive_count": classifications["false_positive"],
        "false_negative_count": classifications["false_negative"],
        "split_error_count": classifications["incorrect_split"],
        "merge_error_count": classifications["incorrect_merge"],
        "premature_close_count": classifications["premature_close"],
        "late_close_count": classifications["late_close"],
        "source_incomplete_count": classifications["source_incomplete_but_honestly_labeled"]
        + classifications["insufficient_evidence"],
        "instances": false_negatives,
    }
    live_summary = {
        "schema_version": "eiel_chronicle_valid1_live_retrospective_comparison_v1",
        "run_count": len(live_rows),
        "accepted": all_live_ok,
        "runs": live_rows,
    }
    source_summary = {
        "schema_version": "eiel_chronicle_valid1_source_completeness_summary_v1",
        "run_counts_by_source_completeness": dict(sorted(completeness_counts.items())),
        "native_rule": "Agent and lineage episode judgments are unavailable under aggregate-only native lifecycle coverage.",
        "aggregate_only_inference_permitted": False,
    }
    write_json(output / "episode_semantics_contract.json", EPISODE_CONTRACTS)
    write_json(output / "run_completion_summary.json", completion_summary)
    write_json(output / "episode_quality_summary.json", quality_summary)
    write_json(output / "false_positive_negative_summary.json", false_summary)
    write_json(output / "live_retrospective_comparison.json", live_summary)
    write_json(output / "source_completeness_summary.json", source_summary)
    _write_csv(
        output / "episode_inventory.csv",
        inventory,
        [
            "run_id",
            "scenario",
            "seed",
            "episode_id",
            "episode_type",
            "status",
            "start_generation",
            "end_generation",
            "source_event_count",
            "source_event_ids",
            "peak_severity",
            "confidence_class",
            "causality_class",
            "source_completeness",
        ],
    )
    _write_csv(
        output / "episode_quality_review.csv",
        reviews,
        [
            "run_id",
            "scenario",
            "seed",
            "episode_id",
            "episode_type",
            "group",
            "source_event_ids",
            "source_event_count",
            "start_generation",
            "observed_end_generation",
            "expected_end_generation",
            "opening_correct",
            "closing_correct",
            "temporal_coherence",
            "subject_or_region_coherence",
            "source_event_sufficiency",
            "metric_consistency",
            "duplicate_episode",
            "confidence_label_correct",
            "causality_label_correct",
            "source_completeness",
            "source_completeness_honest",
            "source_judgment_available",
            "overlap_quality",
            "split_merge_quality",
            "live_retrospective_classification",
            "classification",
        ],
    )
    return canonical_payload(
        {
            "output_dir": str(output),
            "completed_run_count": completion_summary["completed_run_count"],
            "episode_instance_count": len(reviews),
            "classification_counts": dict(sorted(classifications.items())),
            "live_retrospective_accepted": all_live_ok,
        }
    )
