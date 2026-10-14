"""Fail-closed scientific decisions and provenance for RECOMB-72.

This module is reporting-only.  It consumes completed generation rows and
explicit authority evidence; it never executes a simulation, reads runtime
state, or repairs missing evidence.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .recomb72_inheritance import (
    AUDITED_LOCI,
    BOUNDED_INTERPOLATION,
    CURRENT_AUTHORITATIVE,
    PARENT_COPY_PER_LOCUS,
    WHOLE_PACKAGE_SINGLE_PARENT,
)
from .recomb72_metrics import (
    GENERATION_METRICS_SCHEMA_VERSION,
    paired_milestone_summaries,
    time_to_homogenization,
)


DECISION_SCHEMA_VERSION = "recomb72_scientific_decision_v1"
PROVENANCE_SCHEMA_VERSION = "recomb72_cell_provenance_receipt_v1"

PATTERNS = (
    "single_extreme_scalar",
    "resource_specialization_extremes",
    "correlated_multi_trait_packages",
)
MODES = (
    CURRENT_AUTHORITATIVE,
    PARENT_COPY_PER_LOCUS,
    BOUNDED_INTERPOLATION,
    WHOLE_PACKAGE_SINGLE_PARENT,
)
ALTERNATIVE_MODES = MODES[1:]
SEEDS = (101, 202, 303, 404, 505, 606)
MILESTONES = (5, 10, 20, 30)
GENERATIONS = tuple(range(31))
CHECKPOINT_GENERATIONS = (0, 5, 10, 20, 30)

PRIMARY_LOCUS = {
    "single_extreme_scalar": "exploration_bias",
    "resource_specialization_extremes": "resource_handling_axis",
    "correlated_multi_trait_packages": "exploration_bias",
}

REQUIRED_EXTERNAL_GATES = (
    "authoritative_execution",
    "trait_neutrality",
    "no_fallback",
    "checkpoint_parity",
    "reporting_parity",
    "cell_provenance_complete",
    "chronicle_authority",
    "current_mode_matches_production",
)

DECISION_LABELS = {
    "RECOMB-72-A": "CURRENT_INHERITANCE_MATERIALLY_SUPPRESSES_VARIATION",
    "RECOMB-72-B": "CURRENT_INHERITANCE_REDUCES_VARIANCE_BUT_NOT_MATERIALLY",
    "RECOMB-72-C": "MULTI_TRAIT_PACKAGE_BREAKUP_IS_PRIMARY",
    "RECOMB-72-D": "RESOURCE_SPECIALIZATION_INHERITANCE_IS_SPECIFICALLY_CONFOUNDED",
    "RECOMB-72-E": "AUDIT_NOT_INTERPRETABLE",
}


def frozen_analysis_contract() -> dict[str, Any]:
    """Return the preregistered RECOMB-72 analysis and decision contract."""

    return {
        "schema_version": "recomb72_scientific_analysis_contract_v1",
        "expected_cells": 72,
        "generations": list(GENERATIONS),
        "milestones": list(MILESTONES),
        "paired_seeds": list(SEEDS),
        "modes": list(MODES),
        "patterns": list(PATTERNS),
        "primary_locus_by_pattern": dict(PRIMARY_LOCUS),
        "homogenization": {
            "variance_ratio_below": 0.10,
            "consecutive_generations": 3,
        },
        "frozen_definitions": {
            "homogenization": "variance_below_10_percent_for_3_consecutive_generations",
            "extreme_retention": {
                "tolerance_fraction_of_legal_range": 0.05,
                "near_lower": "value_less_than_or_equal_lower_plus_tolerance",
                "near_upper": "value_greater_than_or_equal_upper_minus_tolerance",
                "near_either": "near_lower_or_near_upper",
            },
            "package_retention": {
                "canonical_scalar_identity": "finite_float_hex_with_zero_normalized",
                "intact_a": "all_audited_loci_exactly_equal_package_a",
                "intact_b": "all_audited_loci_exactly_equal_package_b",
                "recombinant": "every_locus_is_an_endpoint_but_package_is_not_intact",
                "neither": "at_least_one_locus_is_not_an_exact_endpoint",
            },
        },
        "required_external_gates": list(REQUIRED_EXTERNAL_GATES),
        "decision_predicates": {
            "RECOMB-72-A": [
                "at_least_two_primary_patterns_suppressed",
                "for_each_suppressed_pattern_one_discrete_alternative",
                "current_exact_extreme_fraction_zero_and_alternative_positive_all_six_seeds",
                "current_homogenizes_before_alternative_all_six_seeds",
            ],
            "RECOMB-72-B": [
                "some_current_primary_g30_variance_decline",
                "no_current_primary_surface_homogenizes",
                "current_extremes_positive_all_patterns_all_six_seeds",
                "current_pattern3_intact_packages_positive_all_six_seeds",
            ],
            "RECOMB-72-C": [
                "no_current_pattern3_locus_homogenizes_in_any_seed",
                "current_intact_packages_zero_and_whole_package_positive_all_six_seeds",
                "current_recombinant_plus_neither_categories_positive_all_six_seeds",
            ],
            "RECOMB-72-D": [
                "resource_is_the_only_suppressed_primary_pattern",
                "current_resource_axis_homogenizes_before_every_comparison_surface_all_six_seeds",
            ],
            "RECOMB-72-E": [
                "any_integrity_or_external_gate_fails",
                "or_no_bounded_a_b_c_d_predicate_is_proven",
            ],
        },
        "decision_precedence": [
            "RECOMB-72-E",
            "RECOMB-72-A",
            "RECOMB-72-C",
            "RECOMB-72-D",
            "RECOMB-72-B",
        ],
        "decision_labels": dict(DECISION_LABELS),
    }


def analysis_contract_sha256() -> str:
    return _sha256(frozen_analysis_contract())


class Recomb72DecisionError(ValueError):
    """Raised for invalid API inputs or malformed provenance contracts."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Recomb72DecisionError("value is not canonical JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise Recomb72DecisionError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Recomb72DecisionError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise Recomb72DecisionError(f"{name} must be a finite number")
    return result


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Recomb72DecisionError(f"{name} must be an integer")
    return value


def _object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Recomb72DecisionError(f"{name} must be an object")
    return value


def _hex(value: Any, *, name: str, length: int) -> str:
    result = str(value or "")
    if len(result) != length or any(character not in "0123456789abcdef" for character in result):
        raise Recomb72DecisionError(
            f"{name} must be {length} lowercase hexadecimal characters"
        )
    return result


def _text(value: Any, *, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise Recomb72DecisionError(f"{name} must be nonempty")
    return result


def _metric(row: Mapping[str, Any], path: str) -> float:
    current: Any = row
    for part in path.split("."):
        current = _object(current, name=path).get(part)
    return _finite(current, name=path)


def _cell_identity(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        _text(row.get("pattern"), name="row.pattern"),
        _text(row.get("mode"), name="row.mode"),
        _integer(row.get("paired_seed"), name="row.paired_seed"),
        _text(row.get("cell_id"), name="row.cell_id"),
    )


def _row_integrity_reasons(
    indexed: Mapping[tuple[str, str, int, str], Mapping[int, Mapping[str, Any]]],
) -> list[str]:
    reasons: set[str] = set()
    expected_scientific = {
        (pattern, mode, seed)
        for pattern in PATTERNS
        for mode in MODES
        for seed in SEEDS
    }
    observed_scientific = {
        (pattern, mode, seed) for pattern, mode, seed, _cell_id in indexed
    }
    if observed_scientific != expected_scientific:
        reasons.add("scientific_matrix_not_exact_4x3x6")
    if len(indexed) != 72:
        reasons.add("cell_count_not_72")

    registry_hashes: set[str] = set()
    mode_hashes: dict[str, set[str]] = defaultdict(set)
    for (pattern, mode, seed, cell_id), generations in indexed.items():
        if set(generations) != set(GENERATIONS):
            reasons.add(f"{cell_id}:generation_coverage_not_g0_g30")
            continue
        initial = generations[0]
        split = _object(
            _object(initial.get("integrity"), name="integrity").get("founder_split"),
            name="integrity.founder_split",
        )
        if split != {"a": 32, "b": 32}:
            reasons.add(f"{cell_id}:founder_split_not_32_32")
        initial_variance = _metric(
            _object(initial.get("loci"), name="loci")[PRIMARY_LOCUS[pattern]],
            "variance_population",
        )
        if initial_variance <= 0.0:
            reasons.add(f"{cell_id}:primary_g0_variance_not_positive")
        if pattern == "correlated_multi_trait_packages" and any(
            _metric(
                _object(
                    _object(initial.get("loci"), name="loci").get(locus),
                    name=f"loci.{locus}",
                ),
                "variance_population",
            )
            <= 0.0
            for locus in AUDITED_LOCI
        ):
            reasons.add(f"{cell_id}:pattern3_g0_locus_variance_not_positive")

        for generation, row in generations.items():
            if row.get("schema_version") != GENERATION_METRICS_SCHEMA_VERSION:
                reasons.add(f"{cell_id}:generation_schema_mismatch")
            population = _object(row.get("population"), name="population")
            integrity = _object(row.get("integrity"), name="integrity")
            if _integer(population.get("after_refill"), name="after_refill") != 64:
                reasons.add(f"{cell_id}:population_not_64")
            if _integer(population.get("mutation_events"), name="mutation_events") != 0:
                reasons.add(f"{cell_id}:mutation_leakage")
            if _integer(population.get("ordinary_births"), name="ordinary_births") != 0:
                reasons.add(f"{cell_id}:ordinary_birth_leakage")
            if _integer(population.get("failed_refill_events"), name="failed_refill_events") != 0:
                reasons.add(f"{cell_id}:failed_refill")
            if integrity.get("genotype_blind_cull") is not True:
                reasons.add(f"{cell_id}:cull_not_genotype_blind")
            if integrity.get("all_survivors_eligible") is not True:
                reasons.add(f"{cell_id}:survivors_not_equally_eligible")
            mutation_contract = _object(
                integrity.get("mutation_off_contract"),
                name="integrity.mutation_off_contract",
            )
            if (
                mutation_contract.get("mutation_events_required") != 0
                or mutation_contract.get("mutation_rng_draws_required") != 0
                or any(
                    mutation_contract.get(key) is not False
                    for key in (
                        "scalar_mutation",
                        "mutation_rate_mutation",
                        "memory_mutation",
                        "policy_weight_mutation",
                        "structural_mutation",
                    )
                )
            ):
                reasons.add(f"{cell_id}:mutation_off_contract_invalid")
            registry_hashes.add(
                _hex(
                    integrity.get("registry_sha256"),
                    name="registry_sha256",
                    length=64,
                )
            )
            mode_hashes[mode].add(
                _hex(
                    integrity.get("mode_sha256"),
                    name="mode_sha256",
                    length=64,
                )
            )
            if generation > 0 and (
                population.get("eligible_parents") != 32
                or population.get("pairings") != 32
                or population.get("children") != 32
                or population.get("selected_parent_events") != 64
            ):
                reasons.add(f"{cell_id}:generation_boundary_reproduction_incomplete")
            loci = _object(row.get("loci"), name="loci")
            if any(locus not in loci for locus in AUDITED_LOCI):
                reasons.add(f"{cell_id}:audited_locus_telemetry_incomplete")
            if pattern == "correlated_multi_trait_packages":
                packages = _object(row.get("packages"), name="packages")
                if any(
                    name not in packages
                    for name in (
                        "intact_package_fraction",
                        "recombinant_fraction",
                        "neither_package_fraction",
                        "correlation_matrix",
                    )
                ):
                    reasons.add(f"{cell_id}:package_telemetry_incomplete")
    if len(registry_hashes) != 1:
        reasons.add("registry_hash_not_single")
    if set(mode_hashes) != set(MODES) or any(
        len(values) != 1 for values in mode_hashes.values()
    ):
        reasons.add("mode_hash_not_single_per_mode")
    if len({next(iter(values)) for values in mode_hashes.values() if values}) != 4:
        reasons.add("mode_hashes_not_distinct")
    return sorted(reasons)


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int, str], dict[int, Mapping[str, Any]]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise Recomb72DecisionError("rows must be a sequence")
    indexed: dict[tuple[str, str, int, str], dict[int, Mapping[str, Any]]] = {}
    scientific_to_cell: dict[tuple[str, str, int], str] = {}
    cell_to_scientific: dict[str, tuple[str, str, int]] = {}
    for index, raw in enumerate(rows):
        row = _object(raw, name=f"rows[{index}]")
        pattern, mode, seed, cell_id = _cell_identity(row)
        generation = _integer(row.get("generation"), name="row.generation")
        if pattern not in PATTERNS or mode not in MODES or seed not in SEEDS:
            raise Recomb72DecisionError(
                f"rows[{index}] has an unexpected pattern/mode/seed"
            )
        scientific = (pattern, mode, seed)
        previous_cell = scientific_to_cell.setdefault(scientific, cell_id)
        if previous_cell != cell_id:
            raise Recomb72DecisionError(
                f"scientific identity {scientific!r} maps to multiple cells"
            )
        previous_scientific = cell_to_scientific.setdefault(cell_id, scientific)
        if previous_scientific != scientific:
            raise Recomb72DecisionError(
                f"cell identity {cell_id!r} maps to multiple scientific identities"
            )
        key = (pattern, mode, seed, cell_id)
        generations = indexed.setdefault(key, {})
        if generation in generations:
            raise Recomb72DecisionError(
                f"duplicate generation row for {cell_id}/G{generation}"
            )
        generations[generation] = row
    return indexed


def _scientific_summaries(
    indexed: Mapping[tuple[str, str, int, str], Mapping[int, Mapping[str, Any]]],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    paired_rows: list[dict[str, Any]] = []
    pattern3_rows: list[dict[str, Any]] = []
    pattern3_correlations: list[dict[str, Any]] = []
    homogenization: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for (pattern, mode, seed, cell_id), generations in sorted(indexed.items()):
        if set(generations) != set(GENERATIONS):
            continue
        loci = _object(generations[0].get("loci"), name="loci")
        for locus_name in AUDITED_LOCI:
            g0_variance = _metric(
                _object(loci.get(locus_name), name=f"loci.{locus_name}"),
                "variance_population",
            )
            variance_series = []
            for generation in GENERATIONS:
                locus = _object(
                    _object(generations[generation].get("loci"), name="loci").get(
                        locus_name
                    ),
                    name=f"loci.{locus_name}",
                )
                variance = _metric(locus, "variance_population")
                variance_series.append(
                    {"generation": generation, "variance": variance}
                )
                if generation not in MILESTONES:
                    continue
                retention = variance / g0_variance if g0_variance > 0.0 else 0.0
                if locus_name == PRIMARY_LOCUS[pattern]:
                    packages = _object(
                        generations[generation].get("packages"), name="packages"
                    )
                    values = {
                        "variance_retention": retention,
                        "extreme_retention": _metric(
                            locus, "fraction_near_either_extreme"
                        ),
                        "package_retention": _metric(
                            packages, "intact_package_fraction"
                        ),
                        "lineage_genetic_decoupling": float(
                            _object(
                                generations[generation].get("ancestry"),
                                name="ancestry",
                            ).get(
                                "genealogy_persists_while_genetic_distinction_disappears"
                            )
                            is True
                        ),
                    }
                    if pattern == "correlated_multi_trait_packages":
                        values.update(
                            {
                                "package_recombinant_fraction": _metric(
                                    packages, "recombinant_fraction"
                                ),
                                "package_neither_fraction": _metric(
                                    packages, "neither_package_fraction"
                                ),
                            }
                        )
                    for metric, value in values.items():
                        paired_rows.append(
                            {
                                "pattern": pattern,
                                "mode": mode,
                                "seed": seed,
                                "generation": generation,
                                "metric": metric,
                                "value": value,
                            }
                        )
                if pattern == "correlated_multi_trait_packages":
                    pattern3_rows.append(
                        {
                            "pattern": pattern,
                            "mode": mode,
                            "seed": seed,
                            "generation": generation,
                            "metric": f"variance_retention.{locus_name}",
                            "value": retention,
                        }
                    )
            homogeneous = time_to_homogenization(variance_series)
            homogenization.append(
                {
                    "cell_id": cell_id,
                    "pattern": pattern,
                    "locus": locus_name,
                    "primary": locus_name == PRIMARY_LOCUS[pattern],
                    "mode": mode,
                    "seed": seed,
                    **homogeneous,
                }
            )
        if pattern == "correlated_multi_trait_packages":
            for generation in MILESTONES:
                correlation = _object(
                    _object(
                        generations[generation].get("packages"), name="packages"
                    ).get("correlation_matrix"),
                    name="packages.correlation_matrix",
                )
                normalized: dict[str, dict[str, float | None]] = {}
                for left in AUDITED_LOCI:
                    left_row = _object(
                        correlation.get(left),
                        name=f"packages.correlation_matrix.{left}",
                    )
                    normalized[left] = {}
                    for right in AUDITED_LOCI:
                        value = left_row.get(right)
                        normalized[left][right] = (
                            None
                            if value is None
                            else _finite(
                                value,
                                name=f"correlation.{left}.{right}",
                            )
                        )
                pattern3_correlations.append(
                    {
                        "cell_id": cell_id,
                        "pattern": pattern,
                        "mode": mode,
                        "seed": seed,
                        "generation": generation,
                        "correlation_matrix": normalized,
                    }
                )
        terminal_ancestry = _object(generations[30].get("ancestry"), name="ancestry")
        lineage.append(
            {
                "cell_id": cell_id,
                "pattern": pattern,
                "mode": mode,
                "seed": seed,
                "effective_alive_lineages": _metric(
                    terminal_ancestry, "effective_alive_lineages"
                ),
                "top1_founder_ancestry_share": _metric(
                    terminal_ancestry, "top1_founder_ancestry_share"
                ),
                "trait_distance_between_branches": _metric(
                    terminal_ancestry, "trait_distance_between_branches"
                ),
                "package_distance_between_branches": _metric(
                    terminal_ancestry, "package_distance_between_branches"
                ),
                "genealogy_persists": terminal_ancestry.get("genealogy_persists")
                is True,
                "genealogy_persists_while_genetic_distinction_disappears": (
                    terminal_ancestry.get(
                        "genealogy_persists_while_genetic_distinction_disappears"
                    )
                    is True
                ),
            }
        )
    paired = paired_milestone_summaries(
        paired_rows,
        current_mode=CURRENT_AUTHORITATIVE,
        alternative_modes=ALTERNATIVE_MODES,
        milestones=MILESTONES,
        expected_seeds=SEEDS,
    )
    pattern3 = paired_milestone_summaries(
        pattern3_rows,
        current_mode=CURRENT_AUTHORITATIVE,
        alternative_modes=ALTERNATIVE_MODES,
        milestones=MILESTONES,
        expected_seeds=SEEDS,
    )
    return (
        paired,
        pattern3,
        pattern3_correlations,
        homogenization,
        _paired_homogenization_summaries(homogenization),
        lineage,
    )


def _paired_homogenization_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (row["pattern"], row["locus"], row["mode"], row["seed"]): row
        for row in rows
    }
    result: list[dict[str, Any]] = []
    for pattern in PATTERNS:
        loci = AUDITED_LOCI if pattern == "correlated_multi_trait_packages" else (
            PRIMARY_LOCUS[pattern],
        )
        for locus in loci:
            for alternative in ALTERNATIVE_MODES:
                pairs = []
                for seed in SEEDS:
                    current = indexed[(pattern, locus, CURRENT_AUTHORITATIVE, seed)]
                    compared = indexed[(pattern, locus, alternative, seed)]
                    current_generation = current["first_generation"]
                    alternative_generation = compared["first_generation"]
                    current_faster = (
                        current["status"] == "reached"
                        and (
                            compared["status"] != "reached"
                            or current_generation < alternative_generation
                        )
                    )
                    pairs.append(
                        {
                            "seed": seed,
                            "current_status": current["status"],
                            "current_first_generation": current_generation,
                            "alternative_status": compared["status"],
                            "alternative_first_generation": alternative_generation,
                            "current_homogenizes_faster": current_faster,
                        }
                    )
                result.append(
                    {
                        "pattern": pattern,
                        "locus": locus,
                        "alternative_mode": alternative,
                        "paired_seed_count": len(pairs),
                        "current_homogenizes_faster_all_seeds": all(
                            pair["current_homogenizes_faster"] for pair in pairs
                        ),
                        "pairs": pairs,
                    }
                )
    return result


def _summary(
    paired: Mapping[str, Any],
    *,
    pattern: str,
    metric: str,
    alternative: str,
    generation: int = 30,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in paired["summaries"]
        if row["pattern"] == pattern
        and row["metric"] == metric
        and row["alternative_mode"] == alternative
        and row["generation"] == generation
    ]
    if len(matches) != 1:
        raise Recomb72DecisionError(
            f"missing unique paired summary for {pattern}/{metric}/{alternative}/G{generation}"
        )
    return matches[0]


def _all_seed_current_zero_alternative_present(
    summary: Mapping[str, Any],
) -> bool:
    return all(
        _finite(pair["current"], name="current") == 0.0
        and _finite(pair["alternative"], name="alternative") > 0.0
        for pair in summary["pairs"]
    )


def _decision(
    paired: Mapping[str, Any],
    homogenization: Sequence[Mapping[str, Any]],
    paired_homogenization: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    discrete = (PARENT_COPY_PER_LOCUS, WHOLE_PACKAGE_SINGLE_PARENT)
    suppressed_patterns: list[str] = []
    evidence: dict[str, Any] = {
        "predicate_contract": (
            "directional_all_six_seeds_plus_preregistered_homogenization_"
            "and_exact_extreme_package_categories"
        ),
        "suppressed_patterns": suppressed_patterns,
    }
    paired_homogenization_index = {
        (row["pattern"], row["locus"], row["alternative_mode"]): row
        for row in paired_homogenization
    }
    for pattern in PATTERNS:
        suppressed = any(
            _all_seed_current_zero_alternative_present(
                _summary(
                    paired,
                    pattern=pattern,
                    metric="extreme_retention",
                    alternative=alternative,
                )
            )
            and paired_homogenization_index[
                (pattern, PRIMARY_LOCUS[pattern], alternative)
            ]["current_homogenizes_faster_all_seeds"]
            for alternative in discrete
        )
        if suppressed:
            suppressed_patterns.append(pattern)

    homogenization_index = {
        (row["pattern"], row["locus"], row["mode"], row["seed"]): row
        for row in homogenization
    }
    pattern3_scalar_not_homogenized = all(
        homogenization_index[
            (
                "correlated_multi_trait_packages",
                locus,
                CURRENT_AUTHORITATIVE,
                seed,
            )
        ]["status"]
        != "reached"
        for locus in AUDITED_LOCI
        for seed in SEEDS
    )
    package = _summary(
        paired,
        pattern="correlated_multi_trait_packages",
        metric="package_retention",
        alternative=WHOLE_PACKAGE_SINGLE_PARENT,
    )
    package_breakup = _all_seed_current_zero_alternative_present(package)
    recombinant_pairs = _summary(
        paired,
        pattern="correlated_multi_trait_packages",
        metric="package_recombinant_fraction",
        alternative=WHOLE_PACKAGE_SINGLE_PARENT,
    )["pairs"]
    neither_pairs = _summary(
        paired,
        pattern="correlated_multi_trait_packages",
        metric="package_neither_fraction",
        alternative=WHOLE_PACKAGE_SINGLE_PARENT,
    )["pairs"]
    breakup_categories = all(
        left["seed"] == right["seed"]
        and _finite(left["current"], name="current recombinant")
        + _finite(right["current"], name="current neither")
        > 0.0
        for left, right in zip(recombinant_pairs, neither_pairs)
    )
    evidence["pattern3_scalar_not_homogenized_all_loci_all_seeds"] = (
        pattern3_scalar_not_homogenized
    )
    evidence["package_breakup_exact_all_seeds"] = package_breakup
    evidence["package_breakup_categories_present_all_seeds"] = breakup_categories

    comparison_surfaces = [
        ("single_extreme_scalar", PRIMARY_LOCUS["single_extreme_scalar"]),
        *[
            ("correlated_multi_trait_packages", locus)
            for locus in AUDITED_LOCI
        ],
    ]
    resource_faster_than_every_surface = all(
        (
            resource_row := homogenization_index[
                (
                    "resource_specialization_extremes",
                    "resource_handling_axis",
                    CURRENT_AUTHORITATIVE,
                    seed,
                )
            ]
        )["status"]
        == "reached"
        and all(
            (
                comparison_row := homogenization_index[
                    (pattern, locus, CURRENT_AUTHORITATIVE, seed)
                ]
            )["status"]
            != "reached"
            or resource_row["first_generation"]
            < comparison_row["first_generation"]
            for pattern, locus in comparison_surfaces
        )
        for seed in SEEDS
    )
    evidence["resource_homogenizes_faster_than_every_surface_all_seeds"] = (
        resource_faster_than_every_surface
    )

    current_primary_rows = [
        homogenization_index[
            (pattern, PRIMARY_LOCUS[pattern], CURRENT_AUTHORITATIVE, seed)
        ]
        for pattern in PATTERNS
        for seed in SEEDS
    ]
    b_no_homogenization = all(
        row["status"] != "reached" for row in current_primary_rows
    )
    b_extremes_available = all(
        _finite(pair["current"], name="current") > 0.0
        for pattern in PATTERNS
        for pair in _summary(
            paired,
            pattern=pattern,
            metric="extreme_retention",
            alternative=PARENT_COPY_PER_LOCUS,
        )["pairs"]
    )
    b_packages_available = all(
        _finite(pair["current"], name="current") > 0.0
        for pair in package["pairs"]
    )
    b_some_decline = any(
        _finite(pair["current"], name="current") < 1.0
        for pattern in PATTERNS
        for pair in _summary(
            paired,
            pattern=pattern,
            metric="variance_retention",
            alternative=PARENT_COPY_PER_LOCUS,
        )["pairs"]
    )
    b_proven = (
        b_some_decline
        and b_no_homogenization
        and b_extremes_available
        and b_packages_available
    )
    evidence["bounded_b_predicate"] = {
        "some_decline": b_some_decline,
        "no_primary_homogenization": b_no_homogenization,
        "extremes_available_all_patterns_all_seeds": b_extremes_available,
        "packages_available_all_seeds": b_packages_available,
    }

    if len(suppressed_patterns) >= 2:
        return "RECOMB-72-A", evidence
    if (
        pattern3_scalar_not_homogenized
        and package_breakup
        and breakup_categories
    ):
        return "RECOMB-72-C", evidence
    if (
        suppressed_patterns == ["resource_specialization_extremes"]
        and resource_faster_than_every_surface
    ):
        return "RECOMB-72-D", evidence
    if b_proven:
        return "RECOMB-72-B", evidence
    return "RECOMB-72-E", evidence


def analyze_recomb72(
    rows: Sequence[Mapping[str, Any]],
    *,
    external_gates: Mapping[str, Any],
    analysis_contract: Mapping[str, Any] | None = None,
    contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Return paired summaries and exactly one fail-closed terminal decision.

    Complete but scientifically invalid rows resolve to decision E.  Malformed
    API values, schema shapes, and duplicate identities raise
    :class:`Recomb72DecisionError`.
    """

    if analysis_contract is not None:
        supplied_contract = dict(
            _object(analysis_contract, name="analysis_contract")
        )
        if supplied_contract != frozen_analysis_contract():
            raise Recomb72DecisionError(
                "analysis contract differs from the frozen RECOMB-72 contract"
            )
        supplied_sha256 = _hex(
            contract_sha256,
            name="analysis_contract_sha256",
            length=64,
        )
        if supplied_sha256 != analysis_contract_sha256():
            raise Recomb72DecisionError("analysis contract SHA-256 mismatch")
    gates = _object(external_gates, name="external_gates")
    indexed = _index_rows(rows)
    row_reasons = _row_integrity_reasons(indexed)
    external_reasons = [
        f"external_gate_failed:{name}"
        for name in REQUIRED_EXTERNAL_GATES
        if gates.get(name) is not True
    ]
    reasons = sorted({*row_reasons, *external_reasons})
    paired: dict[str, Any] | None = None
    pattern3: dict[str, Any] | None = None
    pattern3_correlations: list[dict[str, Any]] = []
    homogenization: list[dict[str, Any]] = []
    paired_homogenization: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    scientific_evidence: dict[str, Any] = {}
    if not row_reasons:
        (
            paired,
            pattern3,
            pattern3_correlations,
            homogenization,
            paired_homogenization,
            lineage,
        ) = _scientific_summaries(indexed)
    if reasons:
        decision = "RECOMB-72-E"
    else:
        assert paired is not None and pattern3 is not None
        decision, scientific_evidence = _decision(
            paired,
            homogenization,
            paired_homogenization,
        )
        if decision == "RECOMB-72-E":
            reasons.append("scientific_predicates_do_not_support_a_b_c_or_d")
    seed_consistency = (
        []
        if paired is None
        else [
            {
                "pattern": row["pattern"],
                "metric": row["metric"],
                "generation": row["generation"],
                "alternative_mode": row["alternative_mode"],
                "positive_seed_count": row["positive_seed_count"],
                "zero_seed_count": row["zero_seed_count"],
                "negative_seed_count": row["negative_seed_count"],
                "same_direction_all_seeds": row["same_direction_all_seeds"],
            }
            for row in paired["summaries"]
        ]
    )
    receipt = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "analysis_contract_sha256": analysis_contract_sha256(),
        "decision": decision,
        "decision_label": DECISION_LABELS[decision],
        "interpretable": decision != "RECOMB-72-E",
        "fail_closed_reasons": reasons,
        "integrity": {
            "row_gates_passed": not row_reasons,
            "row_gate_failures": row_reasons,
            "external_gates": {
                name: gates.get(name) is True for name in REQUIRED_EXTERNAL_GATES
            },
            "external_gate_failures": external_reasons,
            "observed_generation_rows": len(rows),
            "observed_cells": len(indexed),
        },
        "thresholds": {
            "homogenization_variance_ratio": 0.10,
            "homogenization_consecutive_generations": 3,
            "near_extreme_definition": "provider_frozen_category",
            "package_definition": "provider_exact_intact_category",
        },
        "paired_milestones": paired,
        "pattern3_locus_paired_milestones": pattern3,
        "pattern3_correlation_matrices": pattern3_correlations,
        "seed_consistency": seed_consistency,
        "time_to_homogenization": homogenization,
        "paired_time_to_homogenization": paired_homogenization,
        "lineage_genetic_decoupling": lineage,
        "scientific_decision_evidence": scientific_evidence,
    }
    return {**receipt, "receipt_sha256": _sha256(receipt)}


def _reference(
    value: Any,
    *,
    name: str,
    expected_cell_id: str | None = None,
) -> dict[str, Any]:
    reference = _object(value, name=name)
    result = {
        "path": _text(reference.get("path"), name=f"{name}.path"),
        "sha256": _hex(reference.get("sha256"), name=f"{name}.sha256", length=64),
    }
    if expected_cell_id is not None:
        cell_id = _text(reference.get("cell_id"), name=f"{name}.cell_id")
        if cell_id != expected_cell_id:
            raise Recomb72DecisionError(f"{name}.cell_id mismatch")
        result["cell_id"] = cell_id
    return result


def build_cell_provenance_receipt(
    *,
    cell_manifest: Mapping[str, Any],
    config_snapshot: Mapping[str, Any],
    execution_record: Mapping[str, Any],
    completion: Mapping[str, Any],
    source_commit: str,
    artifact_sha256: str,
    provider_sha256: str,
    inheritance_mode_sha256: str,
    checkpoint_contract_sha256: str,
    generation_metrics_reference: Mapping[str, Any],
    checkpoint_references: Sequence[Mapping[str, Any]],
    chronicle_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one deterministic, identity-bound per-cell provenance receipt."""

    manifest = _object(cell_manifest, name="cell_manifest")
    config = _object(config_snapshot, name="config_snapshot")
    execution = _object(execution_record, name="execution_record")
    done = _object(completion, name="completion")
    cell = _object(manifest.get("cell"), name="cell_manifest.cell")
    cell_id = _text(cell.get("cell_id"), name="cell_id")
    mode = _text(
        _object(cell.get("dimensions"), name="cell.dimensions").get(
            "inheritance_mode"
        ),
        name="inheritance_mode",
    )
    if mode not in MODES:
        raise Recomb72DecisionError("cell inheritance mode is unsupported")
    for name, payload in (
        ("config_snapshot", config),
        ("execution_record", execution),
        ("completion", done),
    ):
        if payload.get("cell_id") != cell_id:
            raise Recomb72DecisionError(f"{name}.cell_id mismatch")
    if execution.get("status") != "completed" or done.get("status") != "completed":
        raise Recomb72DecisionError("execution and completion must be completed")

    plan_hash = _hex(manifest.get("plan_sha256"), name="plan_sha256", length=64)
    route_hash = _hex(
        manifest.get("execution_route_sha256"),
        name="execution_route_sha256",
        length=64,
    )
    profile_id = _text(manifest.get("profile_id"), name="profile_id")
    provider_id = _text(manifest.get("provider_id"), name="provider_id")
    for name, payload in (
        ("execution_record", execution),
        ("completion", done),
    ):
        if payload.get("plan_sha256") != plan_hash:
            raise Recomb72DecisionError(f"{name}.plan_sha256 mismatch")
        if payload.get("execution_route_sha256") != route_hash:
            raise Recomb72DecisionError(
                f"{name}.execution_route_sha256 mismatch"
            )

    metrics = _reference(
        generation_metrics_reference,
        name="generation_metrics_reference",
        expected_cell_id=cell_id,
    )
    metrics["row_count"] = _integer(
        generation_metrics_reference.get("row_count"), name="metrics.row_count"
    )
    if metrics["row_count"] != 31:
        raise Recomb72DecisionError("generation metrics must contain G0-G30")

    checkpoints: list[dict[str, Any]] = []
    seen_generations: set[int] = set()
    for index, raw in enumerate(checkpoint_references):
        reference = _reference(
            raw, name=f"checkpoint_references[{index}]", expected_cell_id=cell_id
        )
        generation = _integer(raw.get("generation"), name="checkpoint.generation")
        if generation in seen_generations:
            raise Recomb72DecisionError("duplicate checkpoint generation")
        seen_generations.add(generation)
        checkpoints.append({**reference, "generation": generation})
    if seen_generations != set(CHECKPOINT_GENERATIONS):
        raise Recomb72DecisionError("checkpoint references must cover G0/G5/G10/G20/G30")
    checkpoints.sort(key=lambda row: row["generation"])

    chronicle = _reference(
        chronicle_reference,
        name="chronicle_reference",
        expected_cell_id=cell_id,
    )
    chronicle["world_id"] = _text(
        chronicle_reference.get("world_id"), name="chronicle_reference.world_id"
    )
    if chronicle["world_id"] != cell_id:
        raise Recomb72DecisionError("Chronicle world_id mismatch")

    receipt = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "cell_id": cell_id,
        "campaign_id": _text(manifest.get("campaign_id"), name="campaign_id"),
        "scientific_identity": {
            "inheritance_mode": mode,
            "founder_pattern": _text(
                cell["dimensions"].get("founder_pattern"), name="founder_pattern"
            ),
            "paired_seed": _integer(
                cell["dimensions"].get("paired_seed"), name="paired_seed"
            ),
        },
        "source_commit": _hex(source_commit, name="source_commit", length=40),
        "bindings": {
            "plan_sha256": plan_hash,
            "execution_route_sha256": route_hash,
            "profile_id": profile_id,
            "provider_id": provider_id,
            "artifact_sha256": _hex(
                artifact_sha256, name="artifact_sha256", length=64
            ),
            "provider_sha256": _hex(
                provider_sha256, name="provider_sha256", length=64
            ),
            "inheritance_mode_sha256": _hex(
                inheritance_mode_sha256,
                name="inheritance_mode_sha256",
                length=64,
            ),
            "checkpoint_contract_sha256": _hex(
                checkpoint_contract_sha256,
                name="checkpoint_contract_sha256",
                length=64,
            ),
        },
        "records": {
            "cell_manifest": {
                "sha256": _sha256(manifest),
                "payload": dict(manifest),
            },
            "config_snapshot": {
                "sha256": _sha256(config),
                "payload": dict(config),
            },
            "execution_record": {
                "sha256": _sha256(execution),
                "payload": dict(execution),
            },
            "completion": {
                "sha256": _sha256(done),
                "payload": dict(done),
            },
        },
        "references": {
            "generation_metrics": metrics,
            "checkpoints": checkpoints,
            "chronicle": chronicle,
        },
    }
    return {**receipt, "receipt_sha256": _sha256(receipt)}


__all__ = [
    "ALTERNATIVE_MODES",
    "CHECKPOINT_GENERATIONS",
    "DECISION_LABELS",
    "DECISION_SCHEMA_VERSION",
    "MILESTONES",
    "PROVENANCE_SCHEMA_VERSION",
    "REQUIRED_EXTERNAL_GATES",
    "Recomb72DecisionError",
    "analyze_recomb72",
    "build_cell_provenance_receipt",
]
