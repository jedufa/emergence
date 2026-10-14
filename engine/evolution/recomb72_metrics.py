"""Deterministic, reporting-only metrics for the RECOMB-72 audit.

The helpers in this module are deliberately independent of simulation and
campaign orchestration.  They consume already-authoritative scalar values and
event records, use population (not sample) moments, and never call RNG.
"""

from __future__ import annotations

from collections import Counter
import math
from statistics import median
from typing import Any, Hashable, Mapping, Sequence


GENERATION_METRICS_SCHEMA_VERSION = "recomb72_generation_metrics_v1"
PAIRED_ANALYSIS_SCHEMA_VERSION = "recomb72_paired_analysis_v1"


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonempty_floats(values: Sequence[Any], *, name: str) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a nonempty sequence")
    result = [_finite(value, name=f"{name}[{index}]") for index, value in enumerate(values)]
    if not result:
        raise ValueError(f"{name} must be a nonempty sequence")
    return result


def _population_variance(values: Sequence[float]) -> float:
    center = sum(values) / len(values)
    return sum((value - center) ** 2 for value in values) / len(values)


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    """Return the deterministic linear (R type 7) sample quantile."""

    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = float(fraction) * float(len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return float(
        sorted_values[lower]
        + weight * (sorted_values[upper] - sorted_values[lower])
    )


def canonical_scalar_token(value: Any) -> str:
    """Return an exact, stable token for an already-canonical scalar value."""

    number = _finite(value, name="value")
    if number == 0.0:
        number = 0.0
    return number.hex()


def locus_statistics(
    values: Sequence[Any],
    *,
    lower_bound: Any,
    upper_bound: Any,
    neutral_value: Any = 0.0,
    tolerance_fraction: Any = 0.05,
    generation_zero_variance: Any | None = None,
) -> dict[str, Any]:
    """Compute the frozen per-generation statistics for one audited locus."""

    observed = _nonempty_floats(values, name="values")
    lower = _finite(lower_bound, name="lower_bound")
    upper = _finite(upper_bound, name="upper_bound")
    neutral = _finite(neutral_value, name="neutral_value")
    fraction = _finite(tolerance_fraction, name="tolerance_fraction")
    if upper <= lower:
        raise ValueError("upper_bound must be greater than lower_bound")
    if fraction < 0.0 or fraction > 0.5:
        raise ValueError("tolerance_fraction must be in [0, 0.5]")
    if any(value < lower or value > upper for value in observed):
        raise ValueError("observed locus value is outside its legal bounds")

    ordered = sorted(observed)
    count = len(ordered)
    mean = sum(ordered) / count
    variance = _population_variance(ordered)
    tolerance = (upper - lower) * fraction

    result: dict[str, Any] = {
        "count": count,
        "mean": float(mean),
        "variance_population": float(variance),
        "standard_deviation_population": float(math.sqrt(variance)),
        "minimum": float(ordered[0]),
        "quartile_25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.50),
        "quartile_75": _quantile(ordered, 0.75),
        "maximum": float(ordered[-1]),
        "unique_canonical_value_count": len(
            {canonical_scalar_token(value) for value in ordered}
        ),
        "mean_absolute_value": float(sum(abs(value) for value in ordered) / count),
        "tolerance": float(tolerance),
        "fraction_near_lower_extreme": float(
            sum(value <= lower + tolerance for value in ordered) / count
        ),
        "fraction_near_upper_extreme": float(
            sum(value >= upper - tolerance for value in ordered) / count
        ),
        "fraction_near_either_extreme": float(
            sum(
                value <= lower + tolerance or value >= upper - tolerance
                for value in ordered
            )
            / count
        ),
        "fraction_near_neutral": float(
            sum(abs(value - neutral) <= tolerance for value in ordered) / count
        ),
        "fraction_below_neutral": float(
            sum(value < neutral - tolerance for value in ordered) / count
        ),
        "fraction_above_neutral": float(
            sum(value > neutral + tolerance for value in ordered) / count
        ),
    }
    if generation_zero_variance is not None:
        initial = _finite(
            generation_zero_variance, name="generation_zero_variance"
        )
        result["variance_retention_ratio"] = (
            float(variance / initial) if initial > 0.0 else None
        )
    return result


def locus_inheritance_statistics(
    events: Sequence[Mapping[str, Any]],
    *,
    neutral_value: Any = 0.0,
    sign_tolerance: Any = 0.0,
) -> dict[str, Any]:
    """Summarize parent-pair and child distances for one locus."""

    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise ValueError("events must be a nonempty sequence")
    if not events:
        raise ValueError("events must be a nonempty sequence")
    neutral = _finite(neutral_value, name="neutral_value")
    tolerance = _finite(sign_tolerance, name="sign_tolerance")
    if tolerance < 0.0:
        raise ValueError("sign_tolerance must be non-negative")

    parent_distances: list[float] = []
    to_a: list[float] = []
    to_b: list[float] = []
    to_nearest: list[float] = []
    to_midpoint: list[float] = []
    sign_match = 0
    sign_flip = 0
    sign_indeterminate = 0

    def sign(value: float) -> int:
        delta = value - neutral
        return 1 if delta > tolerance else -1 if delta < -tolerance else 0

    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"events[{index}] must be an object")
        parent_a = _finite(event.get("parent_a"), name=f"events[{index}].parent_a")
        parent_b = _finite(event.get("parent_b"), name=f"events[{index}].parent_b")
        child = _finite(event.get("child"), name=f"events[{index}].child")
        distance_a = abs(child - parent_a)
        distance_b = abs(child - parent_b)
        parent_distances.append(abs(parent_a - parent_b))
        to_a.append(distance_a)
        to_b.append(distance_b)
        to_nearest.append(min(distance_a, distance_b))
        to_midpoint.append(abs(child - (parent_a + parent_b) * 0.5))
        parent_signs = {sign(parent_a), sign(parent_b)}
        child_sign = sign(child)
        if child_sign in parent_signs:
            sign_match += 1
        elif (
            len(parent_signs) == 1
            and next(iter(parent_signs)) != 0
            and child_sign == -next(iter(parent_signs))
        ):
            sign_flip += 1
        else:
            sign_indeterminate += 1

    count = len(events)
    return {
        "event_count": count,
        "parent_pair_distance_mean": float(sum(parent_distances) / count),
        "child_to_parent_a_distance_mean": float(sum(to_a) / count),
        "child_to_parent_b_distance_mean": float(sum(to_b) / count),
        "child_to_nearest_parent_distance_mean": float(sum(to_nearest) / count),
        "child_to_parental_midpoint_distance_mean": float(
            sum(to_midpoint) / count
        ),
        "child_sign_matches_either_parent_count": sign_match,
        "child_sign_flip_count": sign_flip,
        "child_sign_indeterminate_count": sign_indeterminate,
        "child_sign_matches_either_parent_fraction": float(sign_match / count),
        "child_sign_flip_fraction": float(sign_flip / count),
        "child_sign_indeterminate_fraction": float(sign_indeterminate / count),
    }


def _package_row(
    row: Mapping[str, Any], loci: Sequence[str], *, name: str
) -> dict[str, float]:
    if not isinstance(row, Mapping):
        raise ValueError(f"{name} must be an object")
    if set(row) != set(loci):
        raise ValueError(f"{name} must contain exactly the audited loci")
    return {locus: _finite(row[locus], name=f"{name}.{locus}") for locus in loci}


def _normalized_package_distance(
    row: Mapping[str, float],
    reference: Mapping[str, float],
    ranges: Mapping[str, float],
    loci: Sequence[str],
) -> float:
    return math.sqrt(
        sum(((row[locus] - reference[locus]) / ranges[locus]) ** 2 for locus in loci)
        / len(loci)
    )


def package_statistics(
    population: Sequence[Mapping[str, Any]],
    *,
    package_a: Mapping[str, Any],
    package_b: Mapping[str, Any],
    bounds: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Classify Pattern-3 packages and compute normalized correlations."""

    if not isinstance(bounds, Mapping) or not bounds:
        raise ValueError("bounds must define at least one audited locus")
    loci = tuple(sorted(str(name) for name in bounds))
    if len(loci) < 2:
        raise ValueError("package metrics require at least two audited loci")
    ranges: dict[str, float] = {}
    for locus in loci:
        pair = bounds[locus]
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"bounds.{locus} must contain lower and upper")
        lower = _finite(pair[0], name=f"bounds.{locus}[0]")
        upper = _finite(pair[1], name=f"bounds.{locus}[1]")
        if upper <= lower:
            raise ValueError(f"bounds.{locus} is invalid")
        ranges[locus] = upper - lower

    a = _package_row(package_a, loci, name="package_a")
    b = _package_row(package_b, loci, name="package_b")
    for locus in loci:
        lower = _finite(bounds[locus][0], name=f"bounds.{locus}[0]")
        upper = _finite(bounds[locus][1], name=f"bounds.{locus}[1]")
        if not lower <= a[locus] <= upper or not lower <= b[locus] <= upper:
            raise ValueError(f"package endpoint for {locus} is outside legal bounds")
    if any(a[locus] == b[locus] for locus in loci):
        raise ValueError("package endpoints must differ at every audited locus")
    if isinstance(population, (str, bytes)) or not isinstance(population, Sequence):
        raise ValueError("population must be a nonempty sequence")
    rows = [
        _package_row(row, loci, name=f"population[{index}]")
        for index, row in enumerate(population)
    ]
    if not rows:
        raise ValueError("population must be a nonempty sequence")
    for row in rows:
        for locus in loci:
            lower = _finite(bounds[locus][0], name=f"bounds.{locus}[0]")
            upper = _finite(bounds[locus][1], name=f"bounds.{locus}[1]")
            if not lower <= row[locus] <= upper:
                raise ValueError(
                    f"population package value for {locus} is outside legal bounds"
                )

    tokens_a = {locus: canonical_scalar_token(a[locus]) for locus in loci}
    tokens_b = {locus: canonical_scalar_token(b[locus]) for locus in loci}
    counts: Counter[str] = Counter()
    distances_a: list[float] = []
    distances_b: list[float] = []
    for row in rows:
        tokens = {locus: canonical_scalar_token(row[locus]) for locus in loci}
        if tokens == tokens_a:
            category = "intact_a"
        elif tokens == tokens_b:
            category = "intact_b"
        elif all(tokens[locus] in {tokens_a[locus], tokens_b[locus]} for locus in loci):
            category = "recombinant"
        else:
            category = "neither"
        counts[category] += 1
        distances_a.append(_normalized_package_distance(row, a, ranges, loci))
        distances_b.append(_normalized_package_distance(row, b, ranges, loci))

    correlations: dict[str, dict[str, float | None]] = {}
    for left in loci:
        left_values = [row[left] for row in rows]
        left_variance = _population_variance(left_values)
        correlations[left] = {}
        for right in loci:
            right_values = [row[right] for row in rows]
            right_variance = _population_variance(right_values)
            denominator = math.sqrt(left_variance * right_variance)
            covariance = sum(
                (x - sum(left_values) / len(rows))
                * (y - sum(right_values) / len(rows))
                for x, y in zip(left_values, right_values)
            ) / len(rows)
            correlations[left][right] = (
                float(covariance / denominator) if denominator > 0.0 else None
            )

    count = len(rows)
    return {
        "audited_loci": list(loci),
        "population_count": count,
        "intact_package_a_count": counts["intact_a"],
        "intact_package_b_count": counts["intact_b"],
        "intact_package_count": counts["intact_a"] + counts["intact_b"],
        "recombinant_count": counts["recombinant"],
        "neither_package_count": counts["neither"],
        "intact_package_a_fraction": float(counts["intact_a"] / count),
        "intact_package_b_fraction": float(counts["intact_b"] / count),
        "intact_package_fraction": float(
            (counts["intact_a"] + counts["intact_b"]) / count
        ),
        "recombinant_fraction": float(counts["recombinant"] / count),
        "neither_package_fraction": float(counts["neither"] / count),
        "distance_to_package_a_mean": float(sum(distances_a) / count),
        "distance_to_package_a_median": float(median(distances_a)),
        "distance_to_package_b_mean": float(sum(distances_b) / count),
        "distance_to_package_b_median": float(median(distances_b)),
        "correlation_matrix": correlations,
    }


def advance_founder_ancestry(
    ancestry_by_agent: Mapping[Hashable, Mapping[Hashable, Any]],
    birth_events: Sequence[Mapping[str, Any]],
) -> dict[Hashable, dict[Hashable, float]]:
    """Return ancestry with deterministic 50/50 child genealogy appended."""

    result: dict[Hashable, dict[Hashable, float]] = {}
    for agent_id, raw in ancestry_by_agent.items():
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError(f"ancestry for agent {agent_id!r} must be nonempty")
        weights = {
            founder: _finite(weight, name=f"ancestry[{agent_id!r}][{founder!r}]")
            for founder, weight in raw.items()
        }
        total = sum(weights.values())
        if total <= 0.0 or any(weight < 0.0 for weight in weights.values()):
            raise ValueError(f"ancestry for agent {agent_id!r} is invalid")
        result[agent_id] = {
            founder: float(weight / total) for founder, weight in weights.items()
        }

    for index, event in enumerate(birth_events):
        if not isinstance(event, Mapping):
            raise ValueError(f"birth_events[{index}] must be an object")
        child = event.get("child_id")
        parent_a = event.get("parent_a_id")
        parent_b = event.get("parent_b_id")
        if child in result:
            raise ValueError(f"duplicate ancestry identity for child {child!r}")
        if parent_a not in result or parent_b not in result:
            raise ValueError(f"birth_events[{index}] references unknown ancestry")
        weights: Counter[Hashable] = Counter()
        for founder, weight in result[parent_a].items():
            weights[founder] += 0.5 * weight
        for founder, weight in result[parent_b].items():
            weights[founder] += 0.5 * weight
        result[child] = {
            founder: float(weights[founder])
            for founder in sorted(weights, key=lambda value: repr(value))
            if weights[founder] > 0.0
        }
    return result


def ancestry_concentration(
    alive_agent_ids: Sequence[Hashable],
    ancestry_by_agent: Mapping[Hashable, Mapping[Hashable, Any]],
) -> dict[str, Any]:
    """Aggregate founder ancestry across the current living population."""

    if isinstance(alive_agent_ids, (str, bytes)) or not alive_agent_ids:
        raise ValueError("alive_agent_ids must be a nonempty sequence")
    aggregate: Counter[Hashable] = Counter()
    for agent_id in alive_agent_ids:
        weights = ancestry_by_agent.get(agent_id)
        if not isinstance(weights, Mapping) or not weights:
            raise ValueError(f"missing ancestry for living agent {agent_id!r}")
        total = sum(_finite(weight, name="ancestry weight") for weight in weights.values())
        if total <= 0.0:
            raise ValueError(f"ancestry for living agent {agent_id!r} is invalid")
        for founder, raw_weight in weights.items():
            weight = _finite(raw_weight, name="ancestry weight")
            if weight < 0.0:
                raise ValueError("ancestry weights must be non-negative")
            aggregate[founder] += weight / total

    population = float(len(alive_agent_ids))
    shares = {
        founder: float(weight / population)
        for founder, weight in aggregate.items()
        if weight > 0.0
    }
    entropy = -sum(share * math.log(share) for share in shares.values())
    top_founder, top_share = sorted(
        shares.items(), key=lambda item: (-item[1], repr(item[0]))
    )[0]
    return {
        "population_count": len(alive_agent_ids),
        "active_founder_ancestry_count": len(shares),
        "founder_ancestry_entropy": float(entropy),
        "effective_founder_ancestry_count": float(math.exp(entropy)),
        "top1_founder_id": top_founder,
        "top1_founder_ancestry_share": float(top_share),
        "founder_ancestry_shares": {
            str(founder): shares[founder]
            for founder in sorted(shares, key=lambda value: repr(value))
        },
    }


def reproduction_concentration(
    contribution_counts: Mapping[Hashable, Any],
) -> dict[str, Any]:
    """Compute concentration without changing or interpreting parent choice."""

    if not isinstance(contribution_counts, Mapping):
        raise ValueError("contribution_counts must be an object")
    counts: dict[Hashable, int] = {}
    for parent, raw_count in contribution_counts.items():
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError("parent contribution counts must be non-negative integers")
        if raw_count:
            counts[parent] = raw_count
    total = sum(counts.values())
    if total == 0:
        return {
            "parent_contribution_total": 0,
            "contributing_parent_count": 0,
            "top1_parent_contribution_share": None,
            "parent_contribution_hhi": None,
            "effective_contributing_parent_count": None,
            "parent_contribution_entropy": None,
        }
    shares = [count / total for count in counts.values()]
    hhi = sum(share * share for share in shares)
    entropy = -sum(share * math.log(share) for share in shares)
    return {
        "parent_contribution_total": total,
        "contributing_parent_count": len(counts),
        "top1_parent_contribution_share": float(max(shares)),
        "parent_contribution_hhi": float(hhi),
        "effective_contributing_parent_count": float(1.0 / hhi),
        "parent_contribution_entropy": float(entropy),
    }


def paired_milestone_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_mode: str,
    alternative_modes: Sequence[str],
    milestones: Sequence[int] = (5, 10, 20, 30),
    expected_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build complete paired alternative-minus-current milestone summaries."""

    alternatives = tuple(str(mode) for mode in alternative_modes)
    if not alternatives or len(set(alternatives)) != len(alternatives):
        raise ValueError("alternative_modes must be unique and nonempty")
    if current_mode in alternatives:
        raise ValueError("current_mode cannot also be an alternative")
    milestone_set = {int(value) for value in milestones}
    expected_seed_set = (
        None if expected_seeds is None else {int(value) for value in expected_seeds}
    )
    indexed: dict[tuple[str, str, int, int, str], float] = {}
    patterns: set[str] = set()
    metrics: set[str] = set()
    observed_seeds: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"rows[{index}] must be an object")
        pattern = str(row.get("pattern", ""))
        mode = str(row.get("mode", ""))
        metric = str(row.get("metric", ""))
        generation = int(row.get("generation"))
        seed = int(row.get("seed"))
        if not pattern or not metric or mode not in {current_mode, *alternatives}:
            raise ValueError(f"rows[{index}] has an invalid analysis identity")
        if generation not in milestone_set:
            continue
        key = (pattern, metric, generation, seed, mode)
        if key in indexed:
            raise ValueError(f"duplicate paired metric row: {key}")
        indexed[key] = _finite(row.get("value"), name=f"rows[{index}].value")
        patterns.add(pattern)
        metrics.add(metric)
        observed_seeds.add(seed)

    seeds = expected_seed_set if expected_seed_set is not None else observed_seeds
    if not seeds:
        raise ValueError("paired analysis has no seeds")
    if expected_seed_set is not None and observed_seeds != expected_seed_set:
        raise ValueError("paired analysis seed bank is incomplete or unexpected")

    summaries: list[dict[str, Any]] = []
    for pattern in sorted(patterns):
        for metric in sorted(metrics):
            relevant = any(
                key[0] == pattern and key[1] == metric for key in indexed
            )
            if not relevant:
                continue
            for generation in sorted(milestone_set):
                for alternative in alternatives:
                    pairs: list[dict[str, Any]] = []
                    for seed in sorted(seeds):
                        current_key = (
                            pattern,
                            metric,
                            generation,
                            seed,
                            current_mode,
                        )
                        alternative_key = (
                            pattern,
                            metric,
                            generation,
                            seed,
                            alternative,
                        )
                        if current_key not in indexed or alternative_key not in indexed:
                            raise ValueError(
                                "paired analysis is missing a required "
                                f"pattern/metric/generation/seed/mode row: "
                                f"{pattern}/{metric}/G{generation}/S{seed}/{alternative}"
                            )
                        current = indexed[current_key]
                        compared = indexed[alternative_key]
                        pairs.append(
                            {
                                "seed": seed,
                                "current": current,
                                "alternative": compared,
                                "delta_alternative_minus_current": compared - current,
                            }
                        )
                    deltas = [row["delta_alternative_minus_current"] for row in pairs]
                    summaries.append(
                        {
                            "pattern": pattern,
                            "metric": metric,
                            "generation": generation,
                            "current_mode": current_mode,
                            "alternative_mode": alternative,
                            "paired_seed_count": len(pairs),
                            "delta_mean": float(sum(deltas) / len(deltas)),
                            "delta_median": float(median(deltas)),
                            "delta_minimum": float(min(deltas)),
                            "delta_maximum": float(max(deltas)),
                            "positive_seed_count": sum(delta > 0.0 for delta in deltas),
                            "zero_seed_count": sum(delta == 0.0 for delta in deltas),
                            "negative_seed_count": sum(delta < 0.0 for delta in deltas),
                            "same_direction_all_seeds": bool(
                                all(delta > 0.0 for delta in deltas)
                                or all(delta < 0.0 for delta in deltas)
                            ),
                            "pairs": pairs,
                        }
                    )
    return {
        "schema_version": PAIRED_ANALYSIS_SCHEMA_VERSION,
        "current_mode": current_mode,
        "alternative_modes": list(alternatives),
        "milestones": sorted(milestone_set),
        "seeds": sorted(seeds),
        "summaries": summaries,
    }


def time_to_homogenization(
    generation_variances: Sequence[Mapping[str, Any]],
    *,
    threshold_ratio: Any = 0.10,
    consecutive_generations: int = 3,
) -> dict[str, Any]:
    """Find the first consecutive low-variance streak relative to G0."""

    threshold = _finite(threshold_ratio, name="threshold_ratio")
    if threshold < 0.0:
        raise ValueError("threshold_ratio must be non-negative")
    if (
        isinstance(consecutive_generations, bool)
        or not isinstance(consecutive_generations, int)
        or consecutive_generations < 1
    ):
        raise ValueError("consecutive_generations must be a positive integer")
    values: dict[int, float] = {}
    for index, row in enumerate(generation_variances):
        if not isinstance(row, Mapping):
            raise ValueError(f"generation_variances[{index}] must be an object")
        generation = int(row.get("generation"))
        if generation in values:
            raise ValueError(f"duplicate variance generation G{generation}")
        variance = _finite(row.get("variance"), name=f"G{generation}.variance")
        if variance < 0.0:
            raise ValueError("variance cannot be negative")
        values[generation] = variance
    if 0 not in values:
        raise ValueError("time-to-homogenization requires generation zero")
    initial = values[0]
    if initial <= 0.0:
        return {
            "status": "not_evaluable",
            "first_generation": None,
            "confirmation_generation": None,
            "initial_variance": initial,
            "threshold_ratio": threshold,
            "consecutive_generations_required": consecutive_generations,
        }

    streak: list[int] = []
    for generation in sorted(value for value in values if value > 0):
        ratio = values[generation] / initial
        if ratio < threshold and (not streak or generation == streak[-1] + 1):
            streak.append(generation)
        elif ratio < threshold:
            streak = [generation]
        else:
            streak = []
        if len(streak) >= consecutive_generations:
            return {
                "status": "reached",
                "first_generation": streak[0],
                "confirmation_generation": streak[consecutive_generations - 1],
                "initial_variance": initial,
                "threshold_ratio": threshold,
                "consecutive_generations_required": consecutive_generations,
            }
    return {
        "status": "not_reached",
        "first_generation": None,
        "confirmation_generation": None,
        "initial_variance": initial,
        "threshold_ratio": threshold,
        "consecutive_generations_required": consecutive_generations,
    }


__all__ = [
    "GENERATION_METRICS_SCHEMA_VERSION",
    "PAIRED_ANALYSIS_SCHEMA_VERSION",
    "advance_founder_ancestry",
    "ancestry_concentration",
    "canonical_scalar_token",
    "locus_inheritance_statistics",
    "locus_statistics",
    "package_statistics",
    "paired_milestone_summaries",
    "reproduction_concentration",
    "time_to_homogenization",
]
