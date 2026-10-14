#!/usr/bin/env python3
"""Deterministic, analysis-only inheritance variance fixtures for RECOMB-1.

This helper deliberately does not import the production reproduction engine.  It
is a neutral shadow model used to compare the variance consequences of several
inheritance rules without selection or continuing mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "recomb1_inheritance_variance_audit_v1"
DEFAULT_SEEDS = (7100, 7101, 7102)
DEFAULT_GENERATIONS = 10
DEFAULT_POPULATION = 96
CHECKPOINT_GENERATION = 2
NOVELTY_GENERATIONS = (1, 2, 5, 10)
MODELS = (
    "current_production_midpoint_shadow",
    "exact_midpoint",
    "uniform_interpolation",
    "parent_copy_per_locus",
    "linkage_block_copy",
    "analysis_only_diploid",
)


def canonical_bytes(value: object) -> bytes:
    """Return the audit's canonical byte representation."""

    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def deterministic_u01(*parts: object) -> float:
    """Frozen SHA-256 counter draw in [0, 1), independent of runtime RNG state."""

    material = canonical_bytes(list(parts))
    integer = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return float(integer) / float(2**64)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(sum(items) / len(items)) if items else 0.0


def _variance(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    center = _mean(items)
    return float(sum((value - center) ** 2 for value in items) / len(items))


def _covariance(rows: Sequence[Sequence[float]], left: int, right: int) -> float:
    if not rows:
        return 0.0
    left_mean = _mean(row[left] for row in rows)
    right_mean = _mean(row[right] for row in rows)
    return float(
        sum(
            (row[left] - left_mean) * (row[right] - right_mean) for row in rows
        )
        / len(rows)
    )


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def analytic_predictions() -> dict[str, object]:
    """Machine-readable one-generation predictions for iid random mating.

    V is the parental marginal variance and C is cross-locus covariance.  The
    diploid prediction is for the first generation from homozygous founders and
    additive (allele-mean) phenotype expression.
    """

    return {
        "assumptions": [
            "infinite population expectation",
            "iid random mating",
            "no selection",
            "no mutation after the generation-zero novelty injection",
            "equal parental contribution",
        ],
        "symbols": {"V": "parental marginal variance", "C": "parental covariance"},
        "models": {
            "current_production_midpoint_shadow": {
                "offspring_rule": "(parent_a + parent_b) / 2",
                "marginal_variance_multiplier": 0.5,
                "cross_locus_covariance_multiplier": 0.5,
            },
            "exact_midpoint": {
                "offspring_rule": "(parent_a + parent_b) / 2",
                "marginal_variance_multiplier": 0.5,
                "cross_locus_covariance_multiplier": 0.5,
            },
            "uniform_interpolation": {
                "offspring_rule": "parent_a + alpha * (parent_b - parent_a); shared alpha ~ U[0,1]",
                "marginal_variance_multiplier": 2.0 / 3.0,
                "cross_locus_covariance_multiplier": 2.0 / 3.0,
            },
            "parent_copy_per_locus": {
                "offspring_rule": "independent fair parent donor at every locus",
                "marginal_variance_multiplier": 1.0,
                "cross_locus_covariance_multiplier": 0.5,
            },
            "linkage_block_copy": {
                "offspring_rule": "one fair parent donor for the complete trait vector",
                "marginal_variance_multiplier": 1.0,
                "cross_locus_covariance_multiplier": 1.0,
            },
            "analysis_only_diploid": {
                "offspring_rule": (
                    "one independently segregated allele per locus from each parent; "
                    "phenotype is allele mean"
                ),
                "marginal_variance_multiplier_from_homozygous_founders": 0.5,
                "cross_locus_covariance_multiplier_from_homozygous_founders": 0.5,
                "scope": "analysis-only comparator; not a production proposal",
            },
        },
    }


def _individual(values: Sequence[float], lineage: str) -> dict[str, object]:
    return {
        "values": [float(value) for value in values],
        "genotype": [[float(value), float(value)] for value in values],
        "lineage_weights": {lineage: 1.0},
    }


def _phenotype(individual: Mapping[str, object], model: str) -> list[float]:
    if model == "analysis_only_diploid":
        genotype = individual["genotype"]
        assert isinstance(genotype, list)
        return [0.5 * (float(pair[0]) + float(pair[1])) for pair in genotype]
    values = individual["values"]
    assert isinstance(values, list)
    return [float(value) for value in values]


def _merge_lineages(
    parent_a: Mapping[str, object], parent_b: Mapping[str, object]
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for parent in (parent_a, parent_b):
        weights = parent["lineage_weights"]
        assert isinstance(weights, dict)
        for name, weight in weights.items():
            merged[str(name)] = merged.get(str(name), 0.0) + 0.5 * float(weight)
    return {name: merged[name] for name in sorted(merged)}


def _child(
    parent_a: Mapping[str, object],
    parent_b: Mapping[str, object],
    *,
    fixture_name: str,
    model: str,
    seed: int,
    generation: int,
    child_index: int,
) -> dict[str, object]:
    values_a = _phenotype(parent_a, model)
    values_b = _phenotype(parent_b, model)
    lineage_weights = _merge_lineages(parent_a, parent_b)

    if model in {"current_production_midpoint_shadow", "exact_midpoint"}:
        values = [0.5 * (a + b) for a, b in zip(values_a, values_b)]
        genotype = [[value, value] for value in values]
    elif model == "uniform_interpolation":
        alpha = deterministic_u01(
            SCHEMA_VERSION, fixture_name, model, seed, generation, child_index, "alpha"
        )
        values = [a + alpha * (b - a) for a, b in zip(values_a, values_b)]
        genotype = [[value, value] for value in values]
    elif model == "parent_copy_per_locus":
        values = [
            a
            if deterministic_u01(
                SCHEMA_VERSION,
                fixture_name,
                model,
                seed,
                generation,
                child_index,
                "donor",
                locus,
            )
            < 0.5
            else b
            for locus, (a, b) in enumerate(zip(values_a, values_b))
        ]
        genotype = [[value, value] for value in values]
    elif model == "linkage_block_copy":
        donor = (
            parent_a
            if deterministic_u01(
                SCHEMA_VERSION,
                fixture_name,
                model,
                seed,
                generation,
                child_index,
                "block_donor",
            )
            < 0.5
            else parent_b
        )
        values = _phenotype(donor, model)
        genotype = [[value, value] for value in values]
    elif model == "analysis_only_diploid":
        genotype_a = parent_a["genotype"]
        genotype_b = parent_b["genotype"]
        assert isinstance(genotype_a, list) and isinstance(genotype_b, list)
        genotype = []
        for locus, (alleles_a, alleles_b) in enumerate(zip(genotype_a, genotype_b)):
            allele_a_index = int(
                deterministic_u01(
                    SCHEMA_VERSION,
                    fixture_name,
                    model,
                    seed,
                    generation,
                    child_index,
                    "gamete_a",
                    locus,
                )
                >= 0.5
            )
            allele_b_index = int(
                deterministic_u01(
                    SCHEMA_VERSION,
                    fixture_name,
                    model,
                    seed,
                    generation,
                    child_index,
                    "gamete_b",
                    locus,
                )
                >= 0.5
            )
            genotype.append(
                [float(alleles_a[allele_a_index]), float(alleles_b[allele_b_index])]
            )
        values = [0.5 * (pair[0] + pair[1]) for pair in genotype]
    else:
        raise ValueError(f"unsupported model: {model}")

    return {
        "values": values,
        "genotype": genotype,
        "lineage_weights": lineage_weights,
    }


def _metrics(
    population: Sequence[Mapping[str, object]],
    *,
    model: str,
    parent_distances: Sequence[tuple[float, float]] = (),
    strategies: Sequence[Sequence[float]] = (),
    novelty_reference: float | None = None,
) -> dict[str, object]:
    rows = [_phenotype(individual, model) for individual in population]
    dimensions = len(rows[0]) if rows else 0
    means = [_mean(row[locus] for row in rows) for locus in range(dimensions)]
    variances = [_variance(row[locus] for row in rows) for locus in range(dimensions)]
    flattened = [value for row in rows for value in row]
    unique_rows = {tuple(round(value, 12) for value in row) for row in rows}
    extreme_fraction = (
        float(sum(abs(value) >= 0.9 for value in flattened) / len(flattened))
        if flattened
        else 0.0
    )
    center_fraction = (
        float(sum(abs(value) <= 0.1 for value in flattened) / len(flattened))
        if flattened
        else 0.0
    )

    lineage_names = sorted(
        {
            str(name)
            for individual in population
            for name in individual["lineage_weights"]
        }
    )
    lineage_means: dict[str, list[float]] = {}
    for name in lineage_names:
        weights = [
            float(individual["lineage_weights"].get(name, 0.0))
            for individual in population
        ]
        denominator = sum(weights)
        lineage_means[name] = [
            float(sum(weight * row[locus] for weight, row in zip(weights, rows)) / denominator)
            if denominator
            else 0.0
            for locus in range(dimensions)
        ]

    out: dict[str, object] = {
        "population_size": len(population),
        "mean_by_locus": means,
        "variance_by_locus": variances,
        "mean_variance": _mean(variances),
        "extreme_fraction": extreme_fraction,
        "center_fraction": center_fraction,
        "bimodality_proxy": extreme_fraction - center_fraction,
        "unique_phenotype_count": len(unique_rows),
        "lineage_conditioned_mean_by_locus": lineage_means,
        "parent_offspring_distance": {
            "mean_to_parent_a": _mean(pair[0] for pair in parent_distances),
            "mean_to_parent_b": _mean(pair[1] for pair in parent_distances),
            "mean_to_nearest_parent": _mean(min(pair) for pair in parent_distances),
        },
    }
    if dimensions >= 2:
        covariance = _covariance(rows, 0, 1)
        denominator = math.sqrt(max(0.0, variances[0] * variances[1]))
        out["covariance_locus_0_1"] = covariance
        out["correlation_locus_0_1"] = covariance / denominator if denominator else 0.0

    if strategies:
        canonical_strategies = {tuple(float(value) for value in row) for row in strategies}
        parent_matches = sum(tuple(row) in canonical_strategies for row in rows)
        parent_fraction = float(parent_matches / len(rows)) if rows else 0.0
        out["package_mixing"] = {
            "complete_parent_package_fraction": parent_fraction,
            "mixed_or_intermediate_package_fraction": 1.0 - parent_fraction,
        }
    if novelty_reference is not None:
        phenotypes = [row[0] for row in rows]
        carrier_count = 0
        for individual in population:
            genotype = individual["genotype"]
            assert isinstance(genotype, list)
            if any(
                abs(float(allele) - novelty_reference) <= 1e-12
                for allele in genotype[0]
            ):
                carrier_count += 1
        out["novelty_survival"] = {
            "phenotypic_signal_fraction": float(
                sum(value > 1e-12 for value in phenotypes) / len(phenotypes)
            ),
            "strong_signal_fraction": float(
                sum(value >= 0.5 * novelty_reference for value in phenotypes)
                / len(phenotypes)
            ),
            "exact_injected_phenotype_fraction": float(
                sum(abs(value - novelty_reference) <= 1e-12 for value in phenotypes)
                / len(phenotypes)
            ),
            "injected_allele_carrier_fraction": float(carrier_count / len(population)),
            "maximum_phenotype": max(phenotypes),
        }
    return out


def _parent_indices(
    *, fixture_name: str, model: str, seed: int, generation: int, child_index: int, size: int
) -> tuple[int, int]:
    parent_a = int(
        deterministic_u01(
            SCHEMA_VERSION,
            fixture_name,
            "parent_schedule",
            seed,
            generation,
            child_index,
            "a",
        )
        * size
    ) % size
    parent_b = int(
        deterministic_u01(
            SCHEMA_VERSION,
            fixture_name,
            "parent_schedule",
            seed,
            generation,
            child_index,
            "b",
        )
        * size
    ) % size
    if parent_b == parent_a:
        parent_b = (parent_b + 1) % size
    # model is accepted intentionally to make the public formula explicit: all
    # arms receive the same parent schedule for a fixture/seed.
    _ = model
    return parent_a, parent_b


def advance_generation(
    population: Sequence[Mapping[str, object]],
    *,
    fixture_name: str,
    model: str,
    seed: int,
    generation: int,
) -> tuple[list[dict[str, object]], list[tuple[float, float]]]:
    children: list[dict[str, object]] = []
    distances: list[tuple[float, float]] = []
    size = len(population)
    for child_index in range(size):
        parent_a_index, parent_b_index = _parent_indices(
            fixture_name=fixture_name,
            model=model,
            seed=seed,
            generation=generation,
            child_index=child_index,
            size=size,
        )
        parent_a = population[parent_a_index]
        parent_b = population[parent_b_index]
        child = _child(
            parent_a,
            parent_b,
            fixture_name=fixture_name,
            model=model,
            seed=seed,
            generation=generation,
            child_index=child_index,
        )
        child_values = _phenotype(child, model)
        distances.append(
            (
                _euclidean(child_values, _phenotype(parent_a, model)),
                _euclidean(child_values, _phenotype(parent_b, model)),
            )
        )
        children.append(child)
    return children, distances


def fixture_definitions(population_size: int = DEFAULT_POPULATION) -> dict[str, dict[str, object]]:
    if population_size != DEFAULT_POPULATION:
        raise ValueError(f"RECOMB-1 fixtures require population_size={DEFAULT_POPULATION}")
    continuous_left = [(-0.90 + (index % 12) * 0.01,) for index in range(48)]
    continuous_right = [(0.79 + (index % 12) * 0.01,) for index in range(48)]
    return {
        "single_trait_bipolar": {
            "initial": [(-1.0,)] * 48 + [(1.0,)] * 48,
            "lineages": ["negative"] * 48 + ["positive"] * 48,
        },
        "single_trait_three_cluster": {
            "initial": [(-1.0,)] * 32 + [(0.0,)] * 32 + [(1.0,)] * 32,
            "lineages": ["negative"] * 32 + ["central"] * 32 + ["positive"] * 32,
        },
        "single_trait_continuous_two_cluster": {
            "initial": continuous_left + continuous_right,
            "lineages": ["negative"] * 48 + ["positive"] * 48,
        },
        "single_trait_bounded_edge": {
            "initial": [(-0.99,)] * 48 + [(0.99,)] * 48,
            "lineages": ["lower_bound"] * 48 + ["upper_bound"] * 48,
        },
        "multi_trait_two_locus": {
            "initial": [(1.0, 1.0)] * 48 + [(-1.0, -1.0)] * 48,
            "lineages": ["strategy_a"] * 48 + ["strategy_b"] * 48,
            "strategies": [(1.0, 1.0), (-1.0, -1.0)],
        },
        "multi_trait_four_locus": {
            "initial": [(1.0, 1.0, -1.0, -1.0)] * 48
            + [(-1.0, -1.0, 1.0, 1.0)] * 48,
            "lineages": ["strategy_a"] * 48 + ["strategy_b"] * 48,
            "strategies": [(1.0, 1.0, -1.0, -1.0), (-1.0, -1.0, 1.0, 1.0)],
        },
    }


def initial_population(fixture: Mapping[str, object]) -> list[dict[str, object]]:
    initial = fixture["initial"]
    lineages = fixture["lineages"]
    assert isinstance(initial, list) and isinstance(lineages, list)
    return [_individual(values, str(lineage)) for values, lineage in zip(initial, lineages)]


def run_model(
    *,
    fixture_name: str,
    fixture: Mapping[str, object],
    model: str,
    seed: int,
    generations: int = DEFAULT_GENERATIONS,
    start_population: Sequence[Mapping[str, object]] | None = None,
    start_generation: int = 0,
    novelty_reference: float | None = None,
) -> dict[str, object]:
    if model not in MODELS:
        raise ValueError(f"unsupported model: {model}")
    population = (
        [dict(individual) for individual in start_population]
        if start_population is not None
        else initial_population(fixture)
    )
    strategies = fixture.get("strategies", [])
    assert isinstance(strategies, list)
    history: list[dict[str, object]] = [
        {
            "generation": start_generation,
            **_metrics(
                population,
                model=model,
                strategies=strategies,
                novelty_reference=novelty_reference,
            ),
        }
    ]
    for generation in range(start_generation + 1, generations + 1):
        population, distances = advance_generation(
            population,
            fixture_name=fixture_name,
            model=model,
            seed=seed,
            generation=generation,
        )
        history.append(
            {
                "generation": generation,
                **_metrics(
                    population,
                    model=model,
                    parent_distances=distances,
                    strategies=strategies,
                    novelty_reference=novelty_reference,
                ),
            }
        )
    return {"history": history, "final_population": population}


def checkpoint_bytes(
    *, fixture_name: str, model: str, seed: int, generation: int, population: object
) -> bytes:
    return canonical_bytes(
        {
            "schema_version": "recomb1_pure_fixture_checkpoint_v1",
            "fixture": fixture_name,
            "model": model,
            "seed": seed,
            "generation": generation,
            "population": population,
        }
    )


def load_checkpoint(payload: bytes) -> dict[str, object]:
    decoded = json.loads(payload.decode("utf-8"))
    if decoded.get("schema_version") != "recomb1_pure_fixture_checkpoint_v1":
        raise ValueError("unsupported RECOMB-1 fixture checkpoint")
    return decoded


def checkpoint_resume_proof(
    *, fixture_name: str, fixture: Mapping[str, object], model: str, seed: int
) -> dict[str, object]:
    full = run_model(
        fixture_name=fixture_name,
        fixture=fixture,
        model=model,
        seed=seed,
        generations=DEFAULT_GENERATIONS,
    )
    prefix = run_model(
        fixture_name=fixture_name,
        fixture=fixture,
        model=model,
        seed=seed,
        generations=CHECKPOINT_GENERATION,
    )
    serialized = checkpoint_bytes(
        fixture_name=fixture_name,
        model=model,
        seed=seed,
        generation=CHECKPOINT_GENERATION,
        population=prefix["final_population"],
    )
    restored = load_checkpoint(serialized)
    resumed = run_model(
        fixture_name=fixture_name,
        fixture=fixture,
        model=model,
        seed=seed,
        generations=DEFAULT_GENERATIONS,
        start_population=restored["population"],
        start_generation=int(restored["generation"]),
    )
    # Generation 2 itself was already completed before the checkpoint.  Compare
    # only newly computed generations; the resumed history's first row is the
    # restored boundary row and intentionally has no parent-distance event.
    full_continuation = {
        "history": full["history"][CHECKPOINT_GENERATION + 1 :],
        "final_population": full["final_population"],
    }
    resumed_continuation = {
        "history": resumed["history"][1:],
        "final_population": resumed["final_population"],
    }
    return {
        "checkpoint_generation": CHECKPOINT_GENERATION,
        "checkpoint_sha256": hashlib.sha256(serialized).hexdigest(),
        "full_continuation_sha256": _hash(full_continuation),
        "resumed_continuation_sha256": _hash(resumed_continuation),
        "byte_identical_continuation": canonical_bytes(full_continuation)
        == canonical_bytes(resumed_continuation),
    }


def _novelty_fixture() -> dict[str, object]:
    # A scalar mutation is injected once into one founder.  No further mutation
    # occurs, isolating inheritance-mediated survival of the new value.
    return {
        "initial": [(1.0,)] + [(0.0,)] * (DEFAULT_POPULATION - 1),
        "lineages": ["injected_mutant"] + ["resident"] * (DEFAULT_POPULATION - 1),
    }


def run_audit(
    *, seeds: Sequence[int] = DEFAULT_SEEDS, generations: int = DEFAULT_GENERATIONS
) -> dict[str, object]:
    if generations != DEFAULT_GENERATIONS:
        raise ValueError(f"RECOMB-1 audit requires generations={DEFAULT_GENERATIONS}")
    fixtures = fixture_definitions()
    results: dict[str, object] = {}
    repeat_checks: dict[str, bool] = {}
    for fixture_name, fixture in fixtures.items():
        model_results: dict[str, object] = {}
        for model in MODELS:
            seed_results: dict[str, object] = {}
            for seed in seeds:
                first = run_model(
                    fixture_name=fixture_name,
                    fixture=fixture,
                    model=model,
                    seed=int(seed),
                    generations=generations,
                )
                second = run_model(
                    fixture_name=fixture_name,
                    fixture=fixture,
                    model=model,
                    seed=int(seed),
                    generations=generations,
                )
                public = {"history": first["history"]}
                repeat_key = f"{fixture_name}/{model}/{seed}"
                repeat_checks[repeat_key] = canonical_bytes(first) == canonical_bytes(second)
                seed_results[str(seed)] = public
            model_results[model] = seed_results
        results[fixture_name] = model_results

    novelty_results: dict[str, object] = {}
    novelty_fixture = _novelty_fixture()
    for model in MODELS:
        novelty_results[model] = {}
        for seed in seeds:
            run = run_model(
                fixture_name="scalar_mutation_novelty_survival",
                fixture=novelty_fixture,
                model=model,
                seed=int(seed),
                generations=generations,
                novelty_reference=1.0,
            )
            novelty_results[model][str(seed)] = {
                str(generation): run["history"][generation]
                for generation in NOVELTY_GENERATIONS
            }

    proof_fixture_name = "single_trait_bipolar"
    checkpoint_proofs = {
        model: checkpoint_resume_proof(
            fixture_name=proof_fixture_name,
            fixture=fixtures[proof_fixture_name],
            model=model,
            seed=int(seeds[0]),
        )
        for model in MODELS
    }

    bipolar = results["single_trait_bipolar"]
    four_locus = results["multi_trait_four_locus"]
    checks = {
        "current_shadow_equals_exact_midpoint": all(
            bipolar["current_production_midpoint_shadow"][str(seed)]["history"]
            == bipolar["exact_midpoint"][str(seed)]["history"]
            for seed in seeds
        ),
        "midpoint_loses_bipolar_variance": all(
            bipolar["exact_midpoint"][str(seed)]["history"][-1]["mean_variance"]
            < 0.05
            for seed in seeds
        ),
        "parent_copy_retains_more_bipolar_variance": all(
            bipolar["parent_copy_per_locus"][str(seed)]["history"][-1]["mean_variance"]
            > bipolar["exact_midpoint"][str(seed)]["history"][-1]["mean_variance"]
            for seed in seeds
        ),
        "linkage_retains_more_bipolar_variance": all(
            bipolar["linkage_block_copy"][str(seed)]["history"][-1]["mean_variance"]
            > bipolar["exact_midpoint"][str(seed)]["history"][-1]["mean_variance"]
            for seed in seeds
        ),
        "per_locus_copy_mixes_four_locus_packages": all(
            four_locus["parent_copy_per_locus"][str(seed)]["history"][-1][
                "package_mixing"
            ]["mixed_or_intermediate_package_fraction"]
            > 0.5
            for seed in seeds
        ),
        "linkage_copy_preserves_four_locus_packages": all(
            four_locus["linkage_block_copy"][str(seed)]["history"][-1][
                "package_mixing"
            ]["complete_parent_package_fraction"]
            == 1.0
            for seed in seeds
        ),
        "all_exact_repeat_checks_pass": all(repeat_checks.values()),
        "all_checkpoint_resume_checks_pass": all(
            proof["byte_identical_continuation"] for proof in checkpoint_proofs.values()
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "analysis-only neutral inheritance fixtures; production behavior unchanged",
        "rules": {
            "population_size": DEFAULT_POPULATION,
            "seeds": [int(seed) for seed in seeds],
            "generations": generations,
            "selection": "none",
            "mutation": "single generation-zero novelty injection only in scalar novelty fixture",
            "bounds": [-1.0, 1.0],
            "parent_schedule": (
                "SHA256(canonical JSON of schema, fixture, parent_schedule, seed, "
                "generation, child index, parent side); model-independent"
            ),
            "model_draws": "SHA256 counter draws with canonical JSON fields; no process RNG state",
            "bimodality_proxy": "extreme_fraction(abs(x)>=0.9) - center_fraction(abs(x)<=0.1)",
            "unique_precision_decimal_places": 12,
        },
        "analytic_predictions": analytic_predictions(),
        "fixtures": results,
        "scalar_mutation_novelty_survival": {
            "description": (
                "one +1 scalar mutation injected into one of 96 generation-zero "
                "founders; no continuing mutation"
            ),
            "measurement_generations": list(NOVELTY_GENERATIONS),
            "results": novelty_results,
        },
        "determinism": {
            "exact_repeat_checks": repeat_checks,
            "checkpoint_resume_fixture": proof_fixture_name,
            "checkpoint_resume_proofs": checkpoint_proofs,
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    payload = run_audit(seeds=tuple(args.seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")


if __name__ == "__main__":
    main()
