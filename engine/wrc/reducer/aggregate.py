"""Hierarchical seed/level/parameter aggregation for WRC reductions."""

from __future__ import annotations

from collections import defaultdict
import json
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_sha256

from .config import PARAMETER_RESULT_SCHEMA_VERSION, REDUCER_VERSION, ReducerConfig


FEATURES = (
    "terminal",
    "trajectory_mean",
    "maximum",
    "minimum",
    "slope_per_tick",
    "net_change",
)
RANKED_FEATURES = frozenset({"terminal", "trajectory_mean", "maximum", "minimum"})

CANONICAL_LAYER_METRICS: dict[str, tuple[str, ...]] = {
    "ecology": (
        "patch_food_amount_mean",
        "occupied_patch_food_amount_mean",
        "patch_food_regrowth_total",
        "patch_food_harvest_total",
        "patch_waste_mean",
        "patch_density_mean",
        "jungle_predator_events",
        "patch_recovery_activity_mean",
        "patch_regrowth_suppressed_total",
        "wdl_delayed_hazard_damage_total",
    ),
    "organism": (
        "mean_energy_collected_total",
        "mean_fitness",
        "p90_fitness",
        "max_fitness",
        "mean_injury",
        "max_injury",
        "mean_scar",
    ),
    "population": (
        "alive",
        "successful_births",
        "birth_attempts",
        "failed_births",
        "mean_survival_steps_total",
        "cohort_survival_rate",
        "lineage_alive_count_total",
    ),
}


def _canonical_metric_priority(metric: str, layer: str) -> int:
    values = CANONICAL_LAYER_METRICS.get(layer, ())
    try:
        return values.index(metric)
    except ValueError:
        return len(values) + 1000


def _is_canonical_metric(metric: str) -> bool:
    return any(metric in values for values in CANONICAL_LAYER_METRICS.values())


def _interpretability_penalty(metric: str) -> int:
    """Lower is better for human-facing representative metrics.

    Quantiles, extrema, dispersion helpers, and lineage-derived diagnostics remain
    fully available for ranking/drill-down, but should not outrank a plain mean
    or total when both describe the same declared response concept.
    """

    lower = metric.lower()
    penalty = 0
    helper_suffixes = ("_p10", "_p50", "_p90", "_std", "_min", "_max")
    if lower.endswith(helper_suffixes):
        penalty += 2
    if lower.startswith("lineage_"):
        penalty += 2
    if "_slope" in lower or lower.endswith("_variance"):
        penalty += 1
    return penalty


class ParameterReductionError(ValueError):
    """World reductions cannot be combined into one coherent parameter result."""


def _level_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _feature_value(summary: Mapping[str, Any], feature: str) -> float | None:
    if feature == "terminal":
        last = summary.get("last")
        return _finite(last.get("value")) if isinstance(last, Mapping) else None
    if feature == "trajectory_mean":
        return _finite(summary.get("time_weighted_mean"))
    if feature == "maximum":
        maximum = summary.get("maximum")
        return _finite(maximum.get("value")) if isinstance(maximum, Mapping) else None
    if feature == "minimum":
        minimum = summary.get("minimum")
        return _finite(minimum.get("value")) if isinstance(minimum, Mapping) else None
    return _finite(summary.get(feature))


def _paired_summary(
    baseline: Mapping[int, float],
    treatment: Mapping[int, float],
    *,
    scale_floor: float,
) -> dict[str, Any] | None:
    seeds = sorted(set(baseline) & set(treatment))
    if not seeds:
        return None
    deltas = [treatment[seed] - baseline[seed] for seed in seeds]
    baseline_values = [baseline[seed] for seed in seeds]
    treatment_values = [treatment[seed] for seed in seeds]
    mean_delta = statistics.fmean(deltas)
    median_delta = statistics.median(deltas)
    delta_std = _sample_std(deltas)
    baseline_mean = statistics.fmean(baseline_values)
    treatment_mean = statistics.fmean(treatment_values)
    pooled_values = [*baseline_values, *treatment_values]
    pooled_std = _sample_std(pooled_values)
    scale = max(abs(baseline_mean), abs(treatment_mean), pooled_std, scale_floor)
    zero_tolerance = max(scale_floor, scale * 1.0e-12)
    positive = sum(value > zero_tolerance for value in deltas)
    negative = sum(value < -zero_tolerance for value in deltas)
    zero = len(deltas) - positive - negative
    consistency = max(positive, negative, zero) / len(deltas)
    seed_majority_direction = (
        "positive"
        if positive > negative and positive >= zero
        else "negative"
        if negative > positive and negative >= zero
        else "zero_or_mixed"
    )
    mean_direction = (
        "positive"
        if mean_delta > zero_tolerance
        else "negative"
        if mean_delta < -zero_tolerance
        else "zero"
    )
    direction = (
        mean_direction
        if seed_majority_direction in {mean_direction, "zero_or_mixed"}
        else "mixed"
    )
    return {
        "paired_seed_count": len(seeds),
        "paired_seeds": seeds,
        "baseline_mean": baseline_mean,
        "treatment_mean": treatment_mean,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "minimum_delta": min(deltas),
        "maximum_delta": max(deltas),
        "delta_standard_deviation": delta_std,
        "relative_mean_delta": (
            mean_delta / abs(baseline_mean)
            if abs(baseline_mean) > scale_floor
            else None
        ),
        "paired_standardized_effect": (
            mean_delta / delta_std if delta_std > scale_floor else None
        ),
        "normalized_effect": mean_delta / scale,
        "absolute_normalized_effect": abs(mean_delta) / scale,
        "positive_seed_count": positive,
        "negative_seed_count": negative,
        "zero_seed_count": zero,
        "seed_direction_consistency": consistency,
        "mean_direction": mean_direction,
        "seed_majority_direction": seed_majority_direction,
        "direction": direction,
        "effect_score": abs(mean_delta) / scale * consistency,
    }


def _tokenize(text: str) -> set[str]:
    common = {
        "env",
        "cfg",
        "sim",
        "enable",
        "enabled",
        "apply",
        "per",
        "step",
        "steps",
        "tick",
        "ticks",
        "mean",
        "total",
        "max",
        "min",
        "default",
        "scale",
        "weight",
    }
    normalized = "".join(character if character.isalnum() else " " for character in text.lower())
    return {token for token in normalized.split() if len(token) >= 3 and token not in common}


def metric_domains(
    metric: str,
    *,
    parameter_id: str,
    activation_metrics: set[str],
    primary_response_hints: Sequence[str],
) -> dict[str, Any]:
    lower = metric.lower()
    tags: set[str] = set()
    if metric in activation_metrics:
        tags.update({"activation", "direct"})
    parameter_tokens = _tokenize(parameter_id.rsplit(".", 1)[-1])
    metric_tokens = _tokenize(metric)
    hint_token_sets = [_tokenize(item) for item in primary_response_hints]
    hint_tokens = set().union(*hint_token_sets) if hint_token_sets else set()
    hint_overlap = max((len(tokens & metric_tokens) for tokens in hint_token_sets), default=0)
    hint_fraction = max(
        (len(tokens & metric_tokens) / max(len(tokens), 1) for tokens in hint_token_sets),
        default=0.0,
    )
    if parameter_tokens & metric_tokens:
        tags.add("direct")
    if hint_tokens & metric_tokens:
        tags.add("hinted")

    population_tokens = (
        "alive",
        "death",
        "birth",
        "offspring",
        "cohort",
        "survival",
        "lineage",
        "population",
        "repro",
    )
    organism_tokens = (
        "fitness",
        "energy",
        "injury",
        "scar",
        "impairment",
        "health",
        "behavior",
        "drive_",
        "memory",
        "perception",
        "policy",
        "sensor",
    )
    ecology_tokens = (
        "patch_",
        "jungle_",
        "wdl_",
        "eed_",
        "resource",
        "food",
        "waste",
        "scent",
        "cover",
        "shelter",
        "hazard",
        "predator",
        "regrowth",
        "fertility",
        "density",
    )
    if any(token in lower for token in population_tokens):
        tags.add("population")
    if any(token in lower for token in organism_tokens):
        tags.add("organism")
    if any(token in lower for token in ecology_tokens):
        tags.add("ecology")
    if not tags:
        tags.add("other")
    primary = next(
        (
            domain
            for domain in ("activation", "direct", "population", "organism", "ecology", "other")
            if domain in tags
        ),
        "other",
    )
    return {
        "primary": primary,
        "tags": sorted(tags),
        "hint_overlap": hint_overlap,
        "hint_fraction": hint_fraction,
    }


def _level_is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def classify_response_curve(
    level_rows: Sequence[Mapping[str, Any]],
    *,
    domain: str,
    config: ReducerConfig,
) -> dict[str, Any]:
    if len(level_rows) == 2 and all(isinstance(row.get("level"), bool) for row in level_rows):
        ordered = sorted(level_rows, key=lambda row: bool(row["level"]))
        return {
            "shape": "BOOLEAN_CONTRAST",
            "ordered_levels": [row["level"] for row in ordered],
            "ordered_means": [row["mean"] for row in ordered],
            "first_effective_level": True,
            "strongest_response_level": True,
            "largest_step_level": True,
            "threshold_level": None,
            "saturation_level": None,
            "collapse_level": None,
        }
    if len(level_rows) < 3 or not all(_level_is_numeric(row.get("level")) for row in level_rows):
        return {
            "shape": "NOISY_UNRESOLVED",
            "ordered_levels": [row.get("level") for row in level_rows],
            "ordered_means": [row.get("mean") for row in level_rows],
            "reason": "fewer than three ordered numeric levels",
            "first_effective_level": None,
            "strongest_response_level": None,
            "largest_step_level": None,
            "threshold_level": None,
            "saturation_level": None,
            "collapse_level": None,
        }
    ordered = sorted(level_rows, key=lambda row: float(row["level"]))
    means = [float(row["mean"]) for row in ordered]
    levels = [row["level"] for row in ordered]
    within_std = [float(row.get("standard_deviation", 0.0) or 0.0) for row in ordered]
    value_range = max(means) - min(means)
    scale = max(max(abs(value) for value in means), config.scale_floor)
    relative_range = value_range / scale
    pooled_noise = math.sqrt(sum(value * value for value in within_std) / len(within_std))
    signal_to_noise = value_range / max(pooled_noise, config.scale_floor)
    if (
        value_range <= config.flat_absolute_range_threshold
        or relative_range <= config.flat_relative_range_threshold
    ):
        return {
            "shape": "FLAT",
            "ordered_levels": levels,
            "ordered_means": means,
            "relative_range": relative_range,
            "signal_to_noise": signal_to_noise,
            "first_effective_level": None,
            "strongest_response_level": levels[0],
            "largest_step_level": None,
            "threshold_level": None,
            "saturation_level": None,
            "collapse_level": None,
        }
    if signal_to_noise < config.noisy_signal_to_noise_threshold:
        return {
            "shape": "NOISY_UNRESOLVED",
            "ordered_levels": levels,
            "ordered_means": means,
            "relative_range": relative_range,
            "signal_to_noise": signal_to_noise,
            "reason": "between-level range is small relative to paired-seed noise",
            "first_effective_level": None,
            "strongest_response_level": levels[
                max(range(len(means)), key=lambda index: abs(means[index] - means[0]))
            ],
            "largest_step_level": None,
            "threshold_level": None,
            "saturation_level": None,
            "collapse_level": None,
        }
    differences = [right - left for left, right in zip(means, means[1:])]
    tolerance = max(
        value_range * config.monotonic_tolerance_fraction,
        pooled_noise * 0.10,
        config.flat_absolute_range_threshold,
    )
    signs = [1 if value > tolerance else -1 if value < -tolerance else 0 for value in differences]
    nonzero = [value for value in signs if value]
    maximum_step_index = max(range(len(differences)), key=lambda index: abs(differences[index]))
    maximum_step = differences[maximum_step_index]
    total_step = sum(abs(value) for value in differences)
    step_fraction = abs(maximum_step) / total_step if total_step > config.scale_floor else 0.0
    largest_step_level = levels[maximum_step_index + 1]
    first_effective_index = next(
        (
            index
            for index in range(1, len(means))
            if abs(means[index] - means[0]) > tolerance
        ),
        None,
    )
    first_effective_level = (
        levels[first_effective_index] if first_effective_index is not None else None
    )
    strongest_response_level = levels[
        max(range(len(means)), key=lambda index: abs(means[index] - means[0]))
    ]

    # A dominant adjacent jump is a threshold even when every smaller step has
    # the same sign.  Checking this before monotonicity prevents a collapse such
    # as [10, 10, 9.9, 1] from being flattened into a generic monotonic decline.
    # Evenly stepped monotonic curves remain monotonic because no one step owns
    # the configured fraction of total movement.
    if step_fraction >= config.threshold_step_fraction:
        if maximum_step < 0 and domain == "population":
            prior = abs(means[maximum_step_index])
            relative_drop = abs(maximum_step) / max(prior, config.scale_floor)
            shape = (
                "CATASTROPHIC_THRESHOLD_NEGATIVE"
                if relative_drop >= config.catastrophic_relative_drop
                else "THRESHOLD_NEGATIVE"
            )
        else:
            shape = "THRESHOLD_POSITIVE" if maximum_step > 0 else "THRESHOLD_NEGATIVE"
    elif nonzero and all(value > 0 for value in nonzero):
        tail = abs(differences[-1])
        peak = max(abs(value) for value in differences)
        shape = (
            "SATURATING_POSITIVE"
            if len(differences) >= 2 and tail <= peak * config.saturation_tail_fraction
            else "MONOTONIC_POSITIVE"
        )
    elif nonzero and all(value < 0 for value in nonzero):
        tail = abs(differences[-1])
        peak = max(abs(value) for value in differences)
        shape = (
            "SATURATING_NEGATIVE"
            if len(differences) >= 2 and tail <= peak * config.saturation_tail_fraction
            else "MONOTONIC_NEGATIVE"
        )
    else:
        compact = [value for value in signs if value]
        if compact and compact[0] < 0 and compact[-1] > 0 and compact.count(-1) + compact.count(1) == len(compact):
            first_positive = next((index for index, value in enumerate(compact) if value > 0), len(compact))
            shape = "U_SHAPED" if all(value < 0 for value in compact[:first_positive]) and all(value > 0 for value in compact[first_positive:]) else "NON_MONOTONIC"
        elif compact and compact[0] > 0 and compact[-1] < 0:
            first_negative = next((index for index, value in enumerate(compact) if value < 0), len(compact))
            shape = "INVERTED_U" if all(value > 0 for value in compact[:first_negative]) and all(value < 0 for value in compact[first_negative:]) else "NON_MONOTONIC"
        else:
            shape = "NON_MONOTONIC"

    threshold_level = (
        largest_step_level
        if shape in {
            "THRESHOLD_POSITIVE",
            "THRESHOLD_NEGATIVE",
            "CATASTROPHIC_THRESHOLD_NEGATIVE",
        }
        else None
    )
    collapse_level = (
        largest_step_level
        if shape == "CATASTROPHIC_THRESHOLD_NEGATIVE"
        else None
    )
    saturation_level = None
    if shape in {"SATURATING_POSITIVE", "SATURATING_NEGATIVE"}:
        peak = max(abs(value) for value in differences)
        for level_index in range(1, len(levels)):
            remaining = differences[level_index:]
            if remaining and all(
                abs(value) <= peak * config.saturation_tail_fraction
                for value in remaining
            ):
                saturation_level = levels[level_index]
                break
        if saturation_level is None:
            saturation_level = levels[-1]
    return {
        "shape": shape,
        "ordered_levels": levels,
        "ordered_means": means,
        "ordered_standard_deviations": within_std,
        "relative_range": relative_range,
        "signal_to_noise": signal_to_noise,
        "largest_adjacent_step": maximum_step,
        "largest_adjacent_step_fraction": step_fraction,
        "first_effective_level": first_effective_level,
        "strongest_response_level": strongest_response_level,
        "largest_step_level": largest_step_level,
        "threshold_level": threshold_level,
        "saturation_level": saturation_level,
        "collapse_level": collapse_level,
    }


def _cast_reference(value: Any, type_name: str) -> Any:
    if value is None or value == "":
        return None
    if type_name == "bool":
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
    if type_name == "int":
        return int(float(value))
    if type_name == "float":
        return float(value)
    return str(value)


def _ordered_level_keys(
    levels: Sequence[Any],
    by_level: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[str]:
    output = [
        key
        for value in levels
        if (key := _level_key(value)) in by_level
    ]
    output.extend(sorted(key for key in by_level if key not in output))
    return output


def _parameter_leaf(value: str) -> str:
    return value.rsplit(".", 1)[-1].strip()


def _inferred_parameter_domain(parameter_id: str, family: str) -> str:
    text = f"{parameter_id} {family}".lower()
    if any(token in text for token in ("repro", "birth", "offspring", "population", "lineage", "lifecycle")):
        return "population"
    if any(token in text for token in ("memory", "prediction", "policy", "cognition", "fitness", "health", "injury")):
        return "organism"
    if any(token in text for token in ("patch", "jungle", "wdl", "eed", "food", "waste", "resource", "predator", "recovery", "regrowth", "ecolog")):
        return "ecology"
    if "energy" in text or "cost" in text:
        return "organism"
    return "other"


def _control_metric_candidates(
    metric_names: Sequence[str],
    *,
    parameter_id: str,
    runtime_parameter: str,
) -> set[str]:
    leaves = {
        leaf
        for value in (parameter_id, runtime_parameter)
        if (leaf := _parameter_leaf(value))
    }
    return {metric for metric in metric_names if metric in leaves}


def _metric_role(
    metric: str,
    *,
    control_metrics: set[str],
    activation_metrics: set[str],
) -> str:
    if metric in control_metrics:
        return "control_readback"
    if metric in activation_metrics:
        return "activation"
    return "outcome"


def _expected_readback(value: Any, parameter_type: str) -> float | None:
    if parameter_type == "bool":
        return 1.0 if bool(value) else 0.0
    if parameter_type in {"int", "float"} and not isinstance(value, bool):
        resolved = _finite(value)
        return resolved
    return None


def _control_transport_summary(
    *,
    candidates: set[str],
    parameter_type: str,
    ordered_level_keys: Sequence[str],
    level_values: Mapping[str, Any],
    by_level: Mapping[str, Sequence[Mapping[str, Any]]],
    level_feature_rows: Mapping[tuple[str, str, str], Mapping[int, float]],
    config: ReducerConfig,
) -> dict[str, Any]:
    if not candidates:
        return {
            "status": "NOT_AVAILABLE",
            "metric": None,
            "candidate_metrics": [],
            "coverage": 0.0,
            "match_fraction": None,
            "level_readbacks": [],
            "meaning": (
                "No exact numeric runtime-control readback metric was present; "
                "this does not imply that the control was not applied."
            ),
        }

    scored: list[tuple[float, float, str, list[dict[str, Any]], bool]] = []
    for metric in sorted(candidates):
        level_rows: list[dict[str, Any]] = []
        expected_total = 0
        observed_total = 0
        matched_total = 0
        comparable = True
        for level_key in ordered_level_keys:
            requested = level_values[level_key]
            values_by_seed = level_feature_rows.get(
                (metric, "terminal", level_key), {}
            )
            values = list(values_by_seed.values())
            expected_total += len(by_level[level_key])
            observed_total += len(values)
            expected = _expected_readback(requested, parameter_type)
            if expected is None:
                comparable = False
                matched = None
            else:
                tolerance = max(
                    1.0e-9,
                    abs(expected) * 1.0e-9,
                    config.scale_floor * 100.0,
                )
                matched = sum(abs(value - expected) <= tolerance for value in values)
                matched_total += matched
            level_rows.append(
                {
                    "requested_level": requested,
                    "expected_numeric_readback": expected,
                    "observed_seed_count": len(values),
                    "expected_seed_count": len(by_level[level_key]),
                    "observed_mean": statistics.fmean(values) if values else None,
                    "observed_minimum": min(values) if values else None,
                    "observed_maximum": max(values) if values else None,
                    "matching_seed_count": matched,
                }
            )
        coverage = observed_total / expected_total if expected_total else 0.0
        match_fraction = (
            matched_total / observed_total
            if comparable and observed_total
            else 0.0
        )
        scored.append((match_fraction, coverage, metric, level_rows, comparable))

    match_fraction, coverage, metric, level_rows, comparable = max(
        scored,
        key=lambda row: (row[0], row[1], row[2]),
    )
    if coverage < 1.0:
        status = "PARTIAL"
    elif not comparable:
        status = "OBSERVED_UNVALIDATED"
    elif match_fraction >= 1.0:
        status = "CONFIRMED"
    else:
        status = "MISMATCH"
    return {
        "status": status,
        "metric": metric,
        "candidate_metrics": sorted(candidates),
        "coverage": coverage,
        "match_fraction": match_fraction if comparable else None,
        "level_readbacks": level_rows,
        "meaning": (
            "Exact terminal runtime readback compared with the requested Atlas "
            "level for every available paired seed."
        ),
    }


def _layer_status(score: float, consistency: float, metric_count: int) -> str:
    if metric_count < 1:
        return "NOT_AVAILABLE"
    if score >= 0.75 and consistency >= 0.75:
        return "STRONG"
    if score >= 0.35 and consistency >= 0.625:
        return "MODERATE"
    if score >= 0.05:
        return "WEAK"
    return "NO_EFFECT_DETECTED"


def _scientific_effect_score(
    row: Mapping[str, Any],
    *,
    domains: Mapping[str, Any],
    role: str,
    parameter_domain: str,
    config: ReducerConfig,
) -> dict[str, float]:
    """Return a conservative human-facing ranking score.

    ``effect_score`` remains the pure scale-normalized paired effect.  This
    scientific score additionally discounts tiny absolute changes and thin
    evidence, while rewarding declared mechanism relevance.  It is a ranking
    aid only; the raw effect remains available for audit and drill-down.
    """

    raw = float(row.get("effect_score", 0.0) or 0.0)
    absolute_delta = abs(float(row.get("mean_delta", 0.0) or 0.0))
    materiality = min(1.0, absolute_delta / max(config.scientific_materiality_floor, config.scale_floor))
    # Never erase a real but tiny signal entirely; instead prevent a tiny
    # denominator from winning the headline solely through normalization.
    materiality_weight = 0.25 + 0.75 * materiality

    coverage = max(0.0, min(1.0, float(row.get("world_coverage", 0.0) or 0.0)))
    seed_count = max(0, int(row.get("paired_seed_count", 0) or 0))
    seed_weight = min(1.0, seed_count / max(config.scientific_seed_target, 1))
    consistency = max(0.0, min(1.0, float(row.get("seed_direction_consistency", 0.0) or 0.0)))
    confidence_weight = coverage * seed_weight * (0.5 + 0.5 * consistency)

    tags = set(domains.get("tags", ()))
    primary = str(domains.get("primary") or "other")
    relevance = 1.0
    if role == "activation":
        relevance *= 1.35
    elif "direct" in tags:
        relevance *= 1.25
    if "hinted" in tags:
        relevance *= 1.20
    if primary == parameter_domain and parameter_domain != "other":
        relevance *= 1.12
    if _is_canonical_metric(str(row.get("metric") or "")):
        relevance *= 1.10

    score = raw * materiality_weight * confidence_weight * relevance
    return {
        "scientific_effect_score": score,
        "scientific_materiality": materiality,
        "scientific_materiality_weight": materiality_weight,
        "scientific_confidence_weight": confidence_weight,
        "scientific_relevance_weight": relevance,
    }


def _strongest_effect_by_metric(
    effect_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_world_coverage: float,
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in effect_rows:
        if row.get("feature") not in RANKED_FEATURES:
            continue
        if float(row.get("world_coverage", 0.0)) < minimum_world_coverage:
            continue
        metric = str(row.get("metric") or "")
        if not metric or row.get("metric_role") == "control_readback":
            continue
        prior = output.get(metric)
        score = float(row.get("scientific_effect_score", row.get("effect_score", 0.0)))
        prior_score = (
            float(prior.get("scientific_effect_score", prior.get("effect_score", 0.0)))
            if prior is not None
            else -1.0
        )
        if prior is None or score > prior_score:
            output[metric] = row
    return output


def _layer_summaries(
    *,
    ranked_metrics: Sequence[str],
    metric_ranks: Mapping[str, float],
    metric_consistency: Mapping[str, float],
    metric_domains_index: Mapping[str, Mapping[str, Any]],
    metric_roles_index: Mapping[str, str],
    strongest_effects: Mapping[str, Mapping[str, Any]],
    response_curves: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    curve_index = {
        (str(row.get("metric")), str(row.get("feature"))): row
        for row in response_curves
    }
    output: dict[str, Any] = {}
    for layer in ("activation", "direct", "ecology", "organism", "population", "other"):
        candidates: list[str] = []
        for metric in ranked_metrics:
            role = metric_roles_index[metric]
            primary = str(metric_domains_index[metric]["primary"])
            if layer == "activation":
                selected = role == "activation"
            else:
                selected = role != "activation" and primary == layer
            if selected:
                candidates.append(metric)
        strongest_order = sorted(
            candidates,
            key=lambda metric: (-metric_ranks[metric], metric),
        )
        candidates.sort(
            key=lambda metric: (
                0 if "hinted" in metric_domains_index[metric]["tags"] else 1,
                -float(metric_domains_index[metric].get("hint_fraction", 0.0)),
                -int(metric_domains_index[metric].get("hint_overlap", 0)),
                _interpretability_penalty(metric),
                0 if _canonical_metric_priority(metric, layer) < 1000 else 1,
                _canonical_metric_priority(metric, layer),
                -metric_ranks[metric],
                metric,
            )
        )
        top = candidates[:5]
        strongest_metric = strongest_order[0] if strongest_order else None
        # A layer representative is an interpretation choice, not merely the
        # numerically largest score. Prefer declared response hints first, then
        # canonical metrics for that layer, then the strongest remaining metric.
        # This prevents a p90/std helper from becoming the human-facing label
        # when a more direct, explicitly declared mechanism metric is available.
        hinted = [
            metric for metric in candidates
            if "hinted" in metric_domains_index[metric]["tags"]
        ]
        canonical = [
            metric for metric in candidates
            if _canonical_metric_priority(metric, layer) < 1000
        ]
        if hinted:
            representative_metric = max(
                hinted,
                key=lambda metric: (
                    float(metric_domains_index[metric].get("hint_fraction", 0.0)),
                    int(metric_domains_index[metric].get("hint_overlap", 0)),
                    -_interpretability_penalty(metric),
                    metric_ranks[metric],
                    metric,
                ),
            )
        elif canonical:
            representative_metric = min(
                canonical,
                key=lambda metric: (
                    _canonical_metric_priority(metric, layer),
                    -metric_ranks[metric],
                    metric,
                ),
            )
        else:
            representative_metric = strongest_metric
        top_score = (
            metric_ranks[representative_metric] if representative_metric else 0.0
        )
        top_consistency = (
            metric_consistency[representative_metric]
            if representative_metric
            else 0.0
        )
        strongest_detected_score = (
            metric_ranks[strongest_metric] if strongest_metric else 0.0
        )
        strongest_detected_consistency = (
            metric_consistency[strongest_metric] if strongest_metric else 0.0
        )
        top_outcomes: list[dict[str, Any]] = []
        for metric in top:
            effect = strongest_effects.get(metric, {})
            curve = curve_index.get((metric, "terminal")) or curve_index.get(
                (metric, "trajectory_mean"), {}
            )
            top_outcomes.append(
                {
                    "metric": metric,
                    "metric_role": metric_roles_index[metric],
                    "effect_score": metric_ranks[metric],
                    "seed_consistency": metric_consistency[metric],
                    "direction": effect.get("direction"),
                    "strongest_feature": effect.get("feature"),
                    "strongest_treatment_level": effect.get("treatment_level"),
                    "mean_delta": effect.get("mean_delta"),
                    "relative_mean_delta": effect.get("relative_mean_delta"),
                    "paired_standardized_effect": effect.get(
                        "paired_standardized_effect"
                    ),
                    "response_shape": curve.get("shape"),
                    "domain_tags": metric_domains_index[metric]["tags"],
                }
            )
        aggregate = (
            statistics.fmean(metric_ranks[metric] for metric in top[:3])
            if top
            else 0.0
        )
        output[layer] = {
            "status": _layer_status(top_score, top_consistency, len(candidates)),
            "ranked_metric_count": len(candidates),
            "top_effect_score": top_score,
            "top_effect_metric": representative_metric,
            "top_seed_consistency": top_consistency,
            "strongest_detected_metric": strongest_metric,
            "strongest_detected_effect_score": strongest_detected_score,
            "strongest_detected_seed_consistency": strongest_detected_consistency,
            "top_three_mean_effect_score": aggregate,
            "top_outcomes": top_outcomes,
        }
    return output


def _median_first_tick(
    worlds: Sequence[Mapping[str, Any]],
    metric: str,
    key: str,
) -> float | None:
    values = []
    for world in worlds:
        summary = (world.get("metrics") or {}).get(metric)
        if not isinstance(summary, Mapping):
            continue
        value = summary.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            values.append(value)
    return float(statistics.median(values)) if values else None


def _dynamics_summary(
    *,
    top_metric: str | None,
    response_curves: Sequence[Mapping[str, Any]],
    strongest_effects: Mapping[str, Mapping[str, Any]],
    effect_rows: Sequence[Mapping[str, Any]],
    baseline_key: str,
    by_level: Mapping[str, Sequence[Mapping[str, Any]]],
    level_values: Mapping[str, Any],
) -> dict[str, Any]:
    if not top_metric or top_metric not in strongest_effects:
        return {
            "status": "UNRESOLVED",
            "metric": top_metric,
            "pattern": "UNRESOLVED",
            "timing": "UNRESOLVED",
        }
    strongest = strongest_effects[top_metric]
    treatment_key = _level_key(strongest.get("treatment_level"))
    feature_rows = {
        str(row.get("feature")): row
        for row in effect_rows
        if row.get("metric") == top_metric
        and _level_key(row.get("treatment_level")) == treatment_key
    }
    terminal = feature_rows.get("terminal")
    trajectory = feature_rows.get("trajectory_mean")
    maximum = feature_rows.get("maximum")
    minimum = feature_rows.get("minimum")
    terminal_score = float(terminal.get("effect_score", 0.0)) if terminal else 0.0
    trajectory_score = (
        float(trajectory.get("effect_score", 0.0)) if trajectory else 0.0
    )
    peak_score = max(
        float(maximum.get("effect_score", 0.0)) if maximum else 0.0,
        float(minimum.get("effect_score", 0.0)) if minimum else 0.0,
    )
    terminal_direction = terminal.get("direction") if terminal else None
    trajectory_direction = trajectory.get("direction") if trajectory else None
    curve = next(
        (
            row
            for row in response_curves
            if row.get("metric") == top_metric and row.get("feature") == "terminal"
        ),
        None,
    )
    shape = str(curve.get("shape")) if isinstance(curve, Mapping) else "NOISY_UNRESOLVED"
    if shape.startswith("THRESHOLD") or shape.startswith("CATASTROPHIC"):
        pattern = "THRESHOLD_OR_COLLAPSE"
    elif (
        terminal_score >= 0.15
        and trajectory_score >= 0.15
        and terminal_direction == trajectory_direction
        and terminal_direction in {"positive", "negative"}
    ):
        pattern = "PERSISTENT"
    elif peak_score >= 0.25 and terminal_score < 0.10:
        pattern = "TRANSIENT"
    elif terminal_score >= 0.25 and terminal_score > max(trajectory_score, 1.0e-12) * 1.5:
        pattern = "DELAYED_OR_CUMULATIVE"
    else:
        pattern = "MIXED_OR_UNRESOLVED"

    baseline_worlds = by_level.get(baseline_key, ())
    treatment_worlds = by_level.get(treatment_key, ())
    baseline_onset = _median_first_tick(
        baseline_worlds, top_metric, "first_significant_tick"
    )
    treatment_onset = _median_first_tick(
        treatment_worlds, top_metric, "first_significant_tick"
    )
    horizon = max(
        (
            int(world.get("last_tick") or 0)
            for world in [*baseline_worlds, *treatment_worlds]
        ),
        default=0,
    )
    if horizon <= 0:
        timing = "UNRESOLVED"
        timing_basis = "no usable horizon"
    elif treatment_onset is not None:
        if treatment_onset <= horizon * 0.25:
            timing = "EARLY"
        elif treatment_onset <= horizon * 0.75:
            timing = "MID_HORIZON"
        else:
            timing = "LATE"
        timing_basis = "within-world first-significant-tick diagnostic"
    elif pattern in {
        "PERSISTENT",
        "THRESHOLD_OR_COLLAPSE",
        "DELAYED_OR_CUMULATIVE",
        "TRANSIENT",
    } and max(terminal_score, trajectory_score, peak_score) >= 0.15:
        # The effect itself is well supported but this world-level diagnostic
        # did not expose a first-significant tick. Do not mislabel that as
        # absence of timing evidence, and do not invent an onset tick.
        timing = "EFFECT_PRESENT_ONSET_UNRESOLVED"
        timing_basis = "effect supported; onset tick unavailable from retained world summaries"
    else:
        timing = "NOT_DETECTED"
        timing_basis = "no supported temporal onset or material effect"
    return {
        "status": "AVAILABLE",
        "metric": top_metric,
        "treatment_level": level_values.get(treatment_key),
        "response_shape": shape,
        "pattern": pattern,
        "timing": timing,
        "timing_basis": timing_basis,
        "causal_onset_claim": False,
        "horizon_tick": horizon,
        "baseline_first_significant_tick_median": baseline_onset,
        "treatment_first_significant_tick_median": treatment_onset,
        "terminal_effect_score": terminal_score,
        "trajectory_mean_effect_score": trajectory_score,
        "peak_effect_score": peak_score,
        "onset_semantics": (
            "First significant tick is a within-world deviation from that "
            "world's initial observation, not a causal divergence estimate."
        ),
    }


def _response_landmarks(
    *,
    top_metric: str | None,
    response_curves: Sequence[Mapping[str, Any]],
    strongest_effects: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    curve = next(
        (
            row
            for row in response_curves
            if row.get("metric") == top_metric and row.get("feature") == "terminal"
        ),
        None,
    )
    strongest = strongest_effects.get(top_metric or "", {})
    return {
        "metric": top_metric,
        "first_effective_level": curve.get("first_effective_level") if isinstance(curve, Mapping) else None,
        "strongest_effect_level": strongest.get("treatment_level"),
        "strongest_response_level": curve.get("strongest_response_level") if isinstance(curve, Mapping) else None,
        "largest_step_level": curve.get("largest_step_level") if isinstance(curve, Mapping) else None,
        "threshold_level": curve.get("threshold_level") if isinstance(curve, Mapping) else None,
        "saturation_level": curve.get("saturation_level") if isinstance(curve, Mapping) else None,
        "collapse_level": curve.get("collapse_level") if isinstance(curve, Mapping) else None,
        "recommended_operating_range": None,
        "recommendation_status": "NOT_INFERRED_WITHOUT_AN_EXPLICIT_OBJECTIVE",
    }


def reduce_parameter(
    *,
    selected_row: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
    world_results: Sequence[Mapping[str, Any]],
    config: ReducerConfig,
) -> dict[str, Any]:
    parameter_id = str(launch_plan.get("parameter") or selected_row.get("Parameter") or "")
    if not parameter_id:
        raise ParameterReductionError("parameter identity is missing")
    runtime_parameter = str(launch_plan.get("runtime_parameter") or parameter_id)
    family = str(selected_row.get("Family") or "")
    parameter_domain = _inferred_parameter_domain(parameter_id, family)
    parameter_type = str(launch_plan.get("type") or selected_row.get("Type") or "string")
    levels = list(launch_plan.get("levels") or [])
    expected_worlds = int(launch_plan.get("planned_runs") or len(world_results))
    reference = _cast_reference(selected_row.get("Reference default"), parameter_type)
    if reference is None:
        reference = (
            False
            if levels and all(isinstance(value, bool) for value in levels)
            else levels[0]
            if levels
            else None
        )
    if _level_key(reference) not in {_level_key(value) for value in levels} and levels:
        reference = levels[0]
    activation_metrics = {
        str(row.get("metric"))
        for row in (
            launch_plan.get("challenge", {}).get("assertions", [])
            if isinstance(launch_plan.get("challenge"), Mapping)
            else []
        )
        if isinstance(row, Mapping) and row.get("metric")
    }
    primary_hints = [
        item.strip()
        for item in str(selected_row.get("Primary response metrics") or "").split(";")
        if item.strip()
    ]

    by_level: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    level_values: dict[str, Any] = {}
    for world in world_results:
        identity = world.get("identity")
        if not isinstance(identity, Mapping):
            raise ParameterReductionError("world reduction lacks identity")
        if identity.get("parameter_id") != parameter_id:
            raise ParameterReductionError("foreign parameter world reduction")
        key = _level_key(identity.get("parameter_value"))
        level_values[key] = identity.get("parameter_value")
        by_level[key].append(world)

    baseline_key = _level_key(reference)
    if baseline_key not in by_level:
        raise ParameterReductionError(f"baseline level is absent: {reference!r}")
    ordered_level_keys = _ordered_level_keys(levels, by_level)

    metric_names = sorted(
        {
            str(metric)
            for world in world_results
            for metric in (world.get("metrics") or {})
        }
    )
    control_metrics = _control_metric_candidates(
        metric_names,
        parameter_id=parameter_id,
        runtime_parameter=runtime_parameter,
    )
    effect_rows: list[dict[str, Any]] = []
    metric_ranks: dict[str, float] = defaultdict(float)
    metric_consistency: dict[str, float] = defaultdict(float)
    metric_domains_index: dict[str, dict[str, Any]] = {}
    metric_roles_index: dict[str, str] = {}
    level_feature_rows: dict[tuple[str, str, str], dict[int, float]] = {}

    for metric in metric_names:
        domains = metric_domains(
            metric,
            parameter_id=parameter_id,
            activation_metrics=activation_metrics,
            primary_response_hints=primary_hints,
        )
        role = _metric_role(
            metric,
            control_metrics=control_metrics,
            activation_metrics=activation_metrics,
        )
        metric_domains_index[metric] = domains
        metric_roles_index[metric] = role
        for level_key, worlds in by_level.items():
            for feature in FEATURES:
                values: dict[int, float] = {}
                for world in worlds:
                    identity = world["identity"]
                    summary = (world.get("metrics") or {}).get(metric)
                    value = (
                        _feature_value(summary, feature)
                        if isinstance(summary, Mapping)
                        else None
                    )
                    if value is not None:
                        values[int(identity["seed"])] = value
                level_feature_rows[(metric, feature, level_key)] = values
        for feature in FEATURES:
            baseline_values = level_feature_rows.get((metric, feature, baseline_key), {})
            for level_key in ordered_level_keys:
                if level_key == baseline_key:
                    continue
                treatment_values = level_feature_rows.get(
                    (metric, feature, level_key), {}
                )
                summary = _paired_summary(
                    baseline_values,
                    treatment_values,
                    scale_floor=config.scale_floor,
                )
                if summary is None:
                    continue
                coverage = summary["paired_seed_count"] / max(
                    len(by_level[baseline_key]), len(by_level[level_key]), 1
                )
                row = {
                    "metric": metric,
                    "metric_role": role,
                    "feature": feature,
                    "domain": domains["primary"],
                    "domain_tags": domains["tags"],
                    "primary_response_hint_match": "hinted" in domains["tags"],
                    "baseline_level": reference,
                    "treatment_level": level_values[level_key],
                    "world_coverage": coverage,
                    **summary,
                }
                row.update(
                    _scientific_effect_score(
                        row,
                        domains=domains,
                        role=role,
                        parameter_domain=parameter_domain,
                        config=config,
                    )
                )
                effect_rows.append(row)
                if (
                    role != "control_readback"
                    and feature in RANKED_FEATURES
                    and coverage >= config.minimum_metric_world_coverage
                ):
                    metric_ranks[metric] = max(
                        metric_ranks[metric], float(row["scientific_effect_score"])
                    )
                    metric_consistency[metric] = max(
                        metric_consistency[metric],
                        float(summary["seed_direction_consistency"]),
                    )

    strongest_effects = _strongest_effect_by_metric(
        effect_rows,
        minimum_world_coverage=config.minimum_metric_world_coverage,
    )
    metric_ranks = {
        metric: float(row.get("scientific_effect_score", row.get("effect_score", 0.0)))
        for metric, row in strongest_effects.items()
    }
    metric_consistency = {
        metric: float(row["seed_direction_consistency"])
        for metric, row in strongest_effects.items()
    }
    score_order = sorted(
        metric_ranks,
        key=lambda metric: (
            -metric_ranks[metric],
            0 if metric_roles_index[metric] == "activation" else 1,
            0 if "hinted" in metric_domains_index[metric]["tags"] else 1,
            metric,
        ),
    )
    mandatory = {
        metric
        for metric in metric_ranks
        if metric_roles_index[metric] == "activation"
        or "hinted" in metric_domains_index[metric]["tags"]
        or _is_canonical_metric(metric)
    }
    mandatory_order = [metric for metric in score_order if metric in mandatory]
    selected_metrics = set(
        mandatory_order[: config.max_ranked_metrics_per_parameter]
    )
    for metric in score_order:
        if len(selected_metrics) >= config.max_ranked_metrics_per_parameter:
            break
        selected_metrics.add(metric)
    ranked_metrics = [metric for metric in score_order if metric in selected_metrics]

    response_curves: list[dict[str, Any]] = []
    for metric in ranked_metrics:
        domain = metric_domains_index[metric]["primary"]
        for feature in ("terminal", "trajectory_mean"):
            level_rows: list[dict[str, Any]] = []
            for level_key in ordered_level_keys:
                values = list(
                    level_feature_rows.get((metric, feature, level_key), {}).values()
                )
                if not values:
                    continue
                level_rows.append(
                    {
                        "level": level_values[level_key],
                        "mean": statistics.fmean(values),
                        "standard_deviation": _sample_std(values),
                        "seed_count": len(values),
                    }
                )
            curve = classify_response_curve(
                level_rows, domain=domain, config=config
            )
            response_curves.append(
                {
                    "metric": metric,
                    "metric_role": metric_roles_index[metric],
                    "feature": feature,
                    "domain": domain,
                    **curve,
                }
            )

    layers = _layer_summaries(
        ranked_metrics=ranked_metrics,
        metric_ranks=metric_ranks,
        metric_consistency=metric_consistency,
        metric_domains_index=metric_domains_index,
        metric_roles_index=metric_roles_index,
        strongest_effects=strongest_effects,
        response_curves=response_curves,
    )
    top_metric = ranked_metrics[0] if ranked_metrics else None
    top_score = metric_ranks.get(top_metric, 0.0) if top_metric else 0.0
    top_consistency = (
        metric_consistency.get(top_metric, 0.0) if top_metric else 0.0
    )
    activation_rows = [
        row
        for row in effect_rows
        if row["metric"] in activation_metrics
        and row["feature"] in {"terminal", "trajectory_mean", "maximum"}
    ]
    activation_score = max(
        (float(row.get("scientific_effect_score", row.get("effect_score", 0.0))) for row in activation_rows), default=0.0
    )
    activation_consistency = max(
        (float(row["seed_direction_consistency"]) for row in activation_rows),
        default=0.0,
    )
    if not activation_metrics:
        activation_status = "NOT_DECLARED"
    else:
        activation_status = layers["activation"]["status"]

    complete_worlds = len(world_results)
    if complete_worlds < expected_worlds:
        screening = "INCOMPLETE"
    elif activation_status in {"STRONG", "MODERATE"} or (
        top_score >= 0.50 and top_consistency >= 0.625
    ):
        screening = "ACTIVE"
    elif top_score >= 0.15:
        screening = "WEAK"
    elif ranked_metrics:
        screening = "NO_EFFECT_DETECTED"
    else:
        screening = "UNRESOLVED"

    sensitivity = (
        "HIGH"
        if top_score >= 1.0
        else "MEDIUM"
        if top_score >= 0.40
        else "LOW"
        if top_score >= 0.15
        else "NONE_DETECTED"
    )
    consequence_domains = ("ecology", "organism", "population")
    available_consequences = [
        domain
        for domain in consequence_domains
        if layers[domain]["ranked_metric_count"] > 0
    ]
    if parameter_domain in available_consequences:
        primary_consequence_domain = parameter_domain
    elif available_consequences:
        primary_consequence_domain = max(
            available_consequences,
            key=lambda domain: (
                layers[domain]["top_effect_score"],
                layers[domain]["top_seed_consistency"],
                -consequence_domains.index(domain),
            ),
        )
    elif layers["direct"]["ranked_metric_count"]:
        primary_consequence_domain = "direct"
    elif layers["other"]["ranked_metric_count"]:
        primary_consequence_domain = "other"
    else:
        primary_consequence_domain = "unresolved"
    strongest_curve = next(
        (
            row
            for row in response_curves
            if row["metric"] == top_metric and row["feature"] == "terminal"
        ),
        None,
    )
    response_shape = (
        strongest_curve.get("shape") if strongest_curve else "NOISY_UNRESOLVED"
    )
    downstream_consistency = (
        layers[primary_consequence_domain]["top_seed_consistency"]
        if primary_consequence_domain in layers
        else top_consistency
    )
    downstream = (
        "CONSISTENT"
        if downstream_consistency >= 0.75
        else "HETEROGENEOUS"
        if ranked_metrics
        else "UNRESOLVED"
    )
    domain_summary = [
        {
            "domain": domain,
            "status": payload["status"],
            "ranked_metric_count": payload["ranked_metric_count"],
            "top_effect_score": payload["top_effect_score"],
            "top_seed_consistency": payload["top_seed_consistency"],
            "top_three_mean_effect_score": payload[
                "top_three_mean_effect_score"
            ],
        }
        for domain, payload in layers.items()
    ]

    control_transport = _control_transport_summary(
        candidates=control_metrics,
        parameter_type=parameter_type,
        ordered_level_keys=ordered_level_keys,
        level_values=level_values,
        by_level=by_level,
        level_feature_rows=level_feature_rows,
        config=config,
    )
    dynamics = _dynamics_summary(
        top_metric=top_metric,
        response_curves=response_curves,
        strongest_effects=strongest_effects,
        effect_rows=effect_rows,
        baseline_key=baseline_key,
        by_level=by_level,
        level_values=level_values,
    )
    landmarks = _response_landmarks(
        top_metric=top_metric,
        response_curves=response_curves,
        strongest_effects=strongest_effects,
    )

    control_status = str(control_transport.get("status") or "NOT_AVAILABLE")
    control_path_activation = (
        "CONFIRMED"
        if control_status == "CONFIRMED"
        else "MISMATCH"
        if control_status == "MISMATCH"
        else "OBSERVED_UNVALIDATED"
        if control_status == "OBSERVED_UNVALIDATED"
        else "NOT_AVAILABLE"
    )
    activation_layer = layers.get("activation", {})
    direct_layer = layers.get("direct", {})
    activation_layer_status = str(activation_layer.get("status") or "NOT_AVAILABLE")
    direct_layer_status = str(direct_layer.get("status") or "NOT_AVAILABLE")
    if activation_layer_status in {"STRONG", "MODERATE", "WEAK"}:
        mechanism_response = activation_layer_status
        mechanism_metric = activation_layer.get("top_effect_metric")
        mechanism_basis = "declared activation metric"
    elif direct_layer_status in {"STRONG", "MODERATE", "WEAK"}:
        mechanism_response = direct_layer_status
        mechanism_metric = direct_layer.get("top_effect_metric")
        mechanism_basis = "direct response fallback because declared activation metric did not move materially"
    elif activation_layer_status == "NO_EFFECT_DETECTED" or direct_layer_status == "NO_EFFECT_DETECTED":
        mechanism_response = "NO_EFFECT_DETECTED"
        mechanism_metric = activation_layer.get("top_effect_metric") or direct_layer.get("top_effect_metric")
        mechanism_basis = "no material activation/direct mechanism response detected"
    else:
        mechanism_response = "NOT_AVAILABLE"
        mechanism_metric = None
        mechanism_basis = "no activation/direct mechanism metric available"

    interpretation_semantics = {
        "control_path_activation": control_path_activation,
        "control_path_metric": control_transport.get("metric"),
        "mechanism_response": mechanism_response,
        "mechanism_metric": mechanism_metric,
        "mechanism_basis": mechanism_basis,
        "downstream_response": downstream,
        "note": (
            "Control-path activation is transport/readback evidence; mechanism response "
            "is scientific response evidence; downstream response summarizes broader consequences."
        ),
    }

    level_summaries: list[dict[str, Any]] = []
    detail_metrics = (
        set(ranked_metrics[: config.max_detail_metrics_per_parameter])
        | activation_metrics
        | control_metrics
    )
    for level_key in ordered_level_keys:
        worlds = by_level[level_key]
        metric_payload: dict[str, Any] = {}
        for metric in sorted(detail_metrics):
            feature_payload: dict[str, Any] = {}
            for feature in (
                "terminal",
                "trajectory_mean",
                "maximum",
                "minimum",
                "slope_per_tick",
            ):
                values = list(
                    level_feature_rows.get((metric, feature, level_key), {}).values()
                )
                if values:
                    feature_payload[feature] = {
                        "count": len(values),
                        "mean": statistics.fmean(values),
                        "median": statistics.median(values),
                        "minimum": min(values),
                        "maximum": max(values),
                        "standard_deviation": _sample_std(values),
                    }
            if feature_payload:
                metric_payload[metric] = feature_payload
        level_summaries.append(
            {
                "level": level_values[level_key],
                "world_count": len(worlds),
                "seeds": sorted(int(world["identity"]["seed"]) for world in worlds),
                "metrics": metric_payload,
            }
        )

    effect_rows.sort(
        key=lambda row: (
            -float(row["effect_score"]),
            row["metric"],
            row["feature"],
            _level_key(row["treatment_level"]),
        )
    )
    standardized_summary = {
        "schema_version": "eiel_wrc_parameter_card_v1",
        "headline": {
            "screening_classification": screening,
            "activation_status": activation_status,
            "control_path_activation": control_path_activation,
            "mechanism_response": mechanism_response,
            "mechanism_metric": mechanism_metric,
            "sensitivity": sensitivity,
            "response_shape": response_shape,
            "parameter_domain": parameter_domain,
            "primary_consequence_domain": primary_consequence_domain,
            "top_scientific_metric": top_metric,
            "top_effect_score": top_score,
            "seed_consistency": top_consistency,
            "downstream_response": downstream,
        },
        "control_transport": control_transport,
        "interpretation_semantics": interpretation_semantics,
        "layers": layers,
        "dynamics": dynamics,
        "response_landmarks": landmarks,
    }
    result = {
        "schema_version": PARAMETER_RESULT_SCHEMA_VERSION,
        "reducer_version": REDUCER_VERSION,
        "config_sha256": config.sha256,
        "parameter": {
            "id": str(selected_row.get("ID") or launch_plan.get("id") or ""),
            "parameter": parameter_id,
            "runtime_parameter": runtime_parameter,
            "type": parameter_type,
            "family": family,
            "parameter_domain": parameter_domain,
            "world_base": str(selected_row.get("World base") or ""),
            "world_overlays": list(launch_plan.get("world_overlays") or []),
            "reference_default": reference,
            "levels": levels,
            "seeds": list(launch_plan.get("seeds") or []),
            "primary_response_hints": primary_hints,
            "activation_metrics": sorted(activation_metrics),
            "control_readback_metrics": sorted(control_metrics),
        },
        "coverage": {
            "expected_worlds": expected_worlds,
            "complete_worlds": complete_worlds,
            "complete": complete_worlds == expected_worlds,
            "levels_present": [level_values[key] for key in ordered_level_keys],
        },
        "screening": {
            "classification": screening,
            "activation_status": activation_status,
            "control_path_activation": control_path_activation,
            "mechanism_response": mechanism_response,
            "mechanism_metric": mechanism_metric,
            "activation_effect_score": activation_score,
            "activation_seed_consistency": activation_consistency,
            "sensitivity": sensitivity,
            "response_shape": response_shape,
            "primary_domain": primary_consequence_domain,
            "primary_consequence_domain": primary_consequence_domain,
            "top_metric": top_metric,
            "top_effect_score": top_score,
            "top_seed_consistency": top_consistency,
            "downstream_response": downstream,
            "control_transport_status": control_transport["status"],
            "dynamics_pattern": dynamics.get("pattern"),
            "layer_statuses": {
                name: payload["status"] for name, payload in layers.items()
            },
            "domain_summary": domain_summary,
        },
        "standardized_summary": standardized_summary,
        "ranked_metrics": [
            {
                "metric": metric,
                "metric_role": metric_roles_index[metric],
                "effect_score": metric_ranks[metric],
                "seed_consistency": metric_consistency[metric],
                "domain": metric_domains_index[metric],
            }
            for metric in ranked_metrics
        ],
        "level_summaries": level_summaries,
        "response_curves": response_curves,
        "metric_effects": effect_rows,
        "limitations": [
            "This is a screening reduction, not a causal decision or qualification result.",
            "Response-curve labels describe the tested levels only.",
            "Metrics with insufficient paired-seed coverage are not ranked.",
            "Control transport is confirmed only when an exact numeric readback metric exists.",
            "Raw source locators in world reductions remain the drill-back authority.",
        ],
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "FEATURES",
    "ParameterReductionError",
    "classify_response_curve",
    "metric_domains",
    "reduce_parameter",
]
