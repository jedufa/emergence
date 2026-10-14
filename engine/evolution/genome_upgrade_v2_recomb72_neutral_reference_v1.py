"""Frozen analysis-only neutral-drift reference for RECOMB-72-GV2-v1.

The reference models the campaign's four two-allele founder states under the
neutral generation-boundary retain-32/refill-32 schedule. It is not a
production inheritance mechanic and consumes no live campaign outcomes.
"""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import math
import random
from typing import Any, Final, Sequence

from engine.evolution.phase_f_v2.canonical import (
    canonical_json_bytes,
    canonical_json_sha256,
)


SCHEMA_VERSION: Final[str] = "genome_v2_recomb72_neutral_reference_v1"
RECEIPT_SCHEMA_VERSION: Final[str] = "genome_v2_recomb72_neutral_calibration_receipt_v1"
RNG_NAMESPACE: Final[str] = "genome-v2-recomb72-neutral-drift-v1"
ALLELE_A: Final[int] = 0
ALLELE_B: Final[int] = 1
POPULATION_SIZE: Final[int] = 64
DIPLOID_COPY_COUNT: Final[int] = POPULATION_SIZE * 2
SURVIVOR_COUNT: Final[int] = 32
OFFSPRING_COUNT: Final[int] = 32
TERMINAL_GENERATION: Final[int] = 30
MAX_TOTAL_REPLICATES: Final[int] = 10_000

ALL_HETEROZYGOUS_AB: Final[str] = "all_heterozygous_ab"
BALANCED_HOMOZYGOUS_AA_BB: Final[str] = "balanced_homozygous_aa_bb"
HOMOZYGOUS_AA_CONTROL: Final[str] = "homozygous_aa_control"
HOMOZYGOUS_BB_CONTROL: Final[str] = "homozygous_bb_control"
FOUNDER_STATES: Final[tuple[str, ...]] = (
    ALL_HETEROZYGOUS_AB,
    BALANCED_HOMOZYGOUS_AA_BB,
    HOMOZYGOUS_AA_CONTROL,
    HOMOZYGOUS_BB_CONTROL,
)

METRIC_FIELDS: Final[tuple[str, ...]] = (
    "a_allele_count",
    "b_allele_count",
    "a_allele_frequency",
    "b_allele_frequency",
    "a_retained",
    "b_retained",
    "a_fixed",
    "b_fixed",
    "observed_heterozygosity",
    "expected_heterozygosity",
    "surviving_founder_count",
    "founder_concentration",
)

# (allele, founder_identity), paired into one diploid individual.
_Homolog = tuple[int, int]
_Diploid = tuple[_Homolog, _Homolog]

_CONTRACT: Final[dict[str, Any]] = {
    "schema_version": SCHEMA_VERSION,
    "campaign_id": "RECOMB-72-GV2-v1",
    "purpose": "analysis_only_neutral_drift_calibration",
    "production_inheritance_mechanic": False,
    "campaign_outcomes_consumed": False,
    "model_scope": {
        "marker": "one_abstract_two_allele_diploid_marker",
        "alleles": {"A": ALLELE_A, "B": ALLELE_B},
        "homolog_state": ["allele", "founder_identity"],
        "founder_identity": ("both_g0_homologs_of_founder_i_carry_founder_identity_i"),
        "excluded": [
            "locus_linkage",
            "genome_structure",
            "mutation",
            "ecology",
            "selection",
            "production_rng_parity",
        ],
    },
    "founder_states": {
        ALL_HETEROZYGOUS_AB: "64_AB_diploids",
        BALANCED_HOMOZYGOUS_AA_BB: "32_AA_then_32_BB_diploids",
        HOMOZYGOUS_AA_CONTROL: "64_AA_diploids",
        HOMOZYGOUS_BB_CONTROL: "64_BB_diploids",
    },
    "population_schedule": {
        "generation_range_inclusive": [0, TERMINAL_GENERATION],
        "population_size": POPULATION_SIZE,
        "ploidy": 2,
        "survivors_per_transition": SURVIVOR_COUNT,
        "offspring_per_transition": OFFSPRING_COUNT,
        "survivor_selection": "uniform_without_replacement_genotype_blind",
        "mating_pool": "retained_survivors_only",
        "mating": "two_distinct_parents_uniform_without_replacement_per_offspring",
        "gamete": "one_homolog_uniform_from_each_parent",
        "mutation": "disabled",
    },
    "metric_contract": {
        "a_allele_count": "A_homolog_count_out_of_128",
        "b_allele_count": "B_homolog_count_out_of_128",
        "a_allele_frequency": "a_allele_count_divided_by_128",
        "b_allele_frequency": "b_allele_count_divided_by_128",
        "a_retained": "one_when_a_allele_count_is_positive",
        "b_retained": "one_when_b_allele_count_is_positive",
        "a_fixed": "one_when_a_allele_count_equals_128",
        "b_fixed": "one_when_b_allele_count_equals_128",
        "observed_heterozygosity": "AB_diploid_count_divided_by_64",
        "expected_heterozygosity": "two_times_a_frequency_times_b_frequency",
        "surviving_founder_count": "distinct_g0_founder_identities_on_live_homologs",
        "founder_concentration": "sum_squared_live_founder_identity_frequencies",
    },
    "calibration_band_contract": {
        "quantile_method": "r_type_7_linear",
        "per_state": True,
        "per_generation": True,
        "reported_points": [
            "observed_minimum",
            "q025",
            "median",
            "q975",
            "observed_maximum",
        ],
        "replicates": "caller_seed_count_times_replicates_per_seed_per_state",
        "maximum_total_state_replicates": MAX_TOTAL_REPLICATES,
    },
    "rng_contract": {
        "generator": "python_random.Random",
        "paired_across_founder_states": True,
        "replicate_seed_derivation": (
            "first_64_bits_big_endian_of_sha256_canonical_json"
        ),
        "replicate_seed_inputs": [
            "schema_version",
            "contract_sha256",
            "rng_namespace",
            "base_seed",
            "replicate_ordinal",
        ],
    },
    "claims_boundary": (
        "neutral finite-population reference only; no production parity, "
        "ecological selection, fitness, evolvability, cognition, or campaign "
        "outcome claim"
    ),
}


def neutral_reference_contract_v1() -> dict[str, Any]:
    """Return a detached copy of the frozen analysis contract."""

    return copy.deepcopy(_CONTRACT)


CONTRACT_SHA256: Final[str] = canonical_json_sha256(_CONTRACT)


def _validate_seed(seed: Any, *, name: str) -> int:
    if type(seed) is not int or seed < 0 or seed > (1 << 63) - 1:
        raise ValueError(f"{name} must be an integer in [0, 2^63 - 1]")
    return seed


def _validate_founder_state(founder_state: Any) -> str:
    state = str(founder_state)
    if state not in FOUNDER_STATES:
        raise ValueError(f"unsupported neutral founder state: {state!r}")
    return state


def _derived_seed(base_seed: int, replicate_ordinal: int) -> int:
    material = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "rng_namespace": RNG_NAMESPACE,
        "base_seed": base_seed,
        "replicate_ordinal": replicate_ordinal,
    }
    return int.from_bytes(
        hashlib.sha256(canonical_json_bytes(material)).digest()[:8],
        "big",
    )


def _founder_genotype(founder_state: str, founder: int) -> tuple[int, int]:
    if founder_state == ALL_HETEROZYGOUS_AB:
        return ALLELE_A, ALLELE_B
    if founder_state == BALANCED_HOMOZYGOUS_AA_BB:
        return (
            (ALLELE_A, ALLELE_A)
            if founder < POPULATION_SIZE // 2
            else (ALLELE_B, ALLELE_B)
        )
    if founder_state == HOMOZYGOUS_AA_CONTROL:
        return ALLELE_A, ALLELE_A
    return ALLELE_B, ALLELE_B


def _initial_population(founder_state: str) -> list[_Diploid]:
    population: list[_Diploid] = []
    for founder in range(POPULATION_SIZE):
        left, right = _founder_genotype(founder_state, founder)
        population.append(((left, founder), (right, founder)))
    return population


def _generation_metrics(
    population: Sequence[_Diploid],
    *,
    generation: int,
) -> dict[str, int | float]:
    if len(population) != POPULATION_SIZE:
        raise RuntimeError("neutral reference population size drifted")
    homologs = [homolog for genotype in population for homolog in genotype]
    if len(homologs) != DIPLOID_COPY_COUNT:
        raise RuntimeError("neutral reference diploid copy count drifted")
    if any(allele not in (ALLELE_A, ALLELE_B) for allele, _founder in homologs):
        raise RuntimeError("neutral reference observed an unsupported allele")

    allele_counts = Counter(allele for allele, _founder in homologs)
    founder_counts = Counter(founder for _allele, founder in homologs)
    a_count = allele_counts[ALLELE_A]
    b_count = allele_counts[ALLELE_B]
    a_frequency = a_count / DIPLOID_COPY_COUNT
    b_frequency = b_count / DIPLOID_COPY_COUNT
    founder_concentration = sum(
        (count / DIPLOID_COPY_COUNT) ** 2 for count in founder_counts.values()
    )
    return {
        "generation": generation,
        "a_allele_count": a_count,
        "b_allele_count": b_count,
        "a_allele_frequency": float(a_frequency),
        "b_allele_frequency": float(b_frequency),
        "a_retained": int(a_count > 0),
        "b_retained": int(b_count > 0),
        "a_fixed": int(a_count == DIPLOID_COPY_COUNT),
        "b_fixed": int(b_count == DIPLOID_COPY_COUNT),
        "observed_heterozygosity": float(
            sum(left[0] != right[0] for left, right in population) / POPULATION_SIZE
        ),
        "expected_heterozygosity": float(2.0 * a_frequency * b_frequency),
        "surviving_founder_count": len(founder_counts),
        "founder_concentration": float(founder_concentration),
    }


def simulate_neutral_drift_replicate_v1(
    *,
    founder_state: str,
    seed: int,
    replicate_ordinal: int = 0,
) -> dict[str, Any]:
    """Simulate one exact G0..G30 founder-state reference replicate."""

    state = _validate_founder_state(founder_state)
    base_seed = _validate_seed(seed, name="seed")
    if type(replicate_ordinal) is not int or replicate_ordinal < 0:
        raise ValueError("replicate_ordinal must be a non-negative integer")
    derived_seed = _derived_seed(base_seed, replicate_ordinal)
    rng = random.Random(derived_seed)
    population = _initial_population(state)
    generations = [_generation_metrics(population, generation=0)]

    for generation in range(1, TERMINAL_GENERATION + 1):
        survivor_indices = sorted(rng.sample(range(POPULATION_SIZE), SURVIVOR_COUNT))
        survivors = [population[index] for index in survivor_indices]
        offspring: list[_Diploid] = []
        for _offspring_ordinal in range(OFFSPRING_COUNT):
            parent_a_index, parent_b_index = rng.sample(range(SURVIVOR_COUNT), 2)
            parent_a = survivors[parent_a_index]
            parent_b = survivors[parent_b_index]
            offspring.append((parent_a[rng.randrange(2)], parent_b[rng.randrange(2)]))
        population = survivors + offspring
        generations.append(_generation_metrics(population, generation=generation))

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "founder_state": state,
        "base_seed": base_seed,
        "replicate_ordinal": replicate_ordinal,
        "derived_seed": derived_seed,
        "generations": generations,
    }
    return {**unsigned, "replicate_sha256": canonical_json_sha256(unsigned)}


def _quantile(ordered: Sequence[int | float], fraction: float) -> float:
    if not ordered:
        raise ValueError("quantile requires observations")
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return float(ordered[lower] + weight * (ordered[upper] - ordered[lower]))


def _band(values: Sequence[int | float]) -> dict[str, int | float]:
    ordered = sorted(values)
    return {
        "observed_minimum": ordered[0],
        "q025": _quantile(ordered, 0.025),
        "median": _quantile(ordered, 0.5),
        "q975": _quantile(ordered, 0.975),
        "observed_maximum": ordered[-1],
    }


def _state_calibration(
    *,
    founder_state: str,
    seeds: Sequence[int],
    replicate_count: int,
) -> dict[str, Any]:
    replicates = [
        simulate_neutral_drift_replicate_v1(
            founder_state=founder_state,
            seed=seed,
            replicate_ordinal=replicate_ordinal,
        )
        for seed in seeds
        for replicate_ordinal in range(replicate_count)
    ]
    generation_bands = []
    for generation in range(TERMINAL_GENERATION + 1):
        rows = [replicate["generations"][generation] for replicate in replicates]
        generation_bands.append(
            {
                "generation": generation,
                "bands": {
                    name: _band([row[name] for row in rows]) for name in METRIC_FIELDS
                },
            }
        )
    terminal_rows = [replicate["generations"][-1] for replicate in replicates]
    return {
        "generation_bands": generation_bands,
        "terminal_fixation_summary": {
            name: {
                "fixed_replicates": sum(int(row[name]) for row in terminal_rows),
                "total_replicates": len(replicates),
                "observed_rate": float(
                    sum(int(row[name]) for row in terminal_rows) / len(replicates)
                ),
            }
            for name in ("a_fixed", "b_fixed")
        },
        "replicate_sha256": [replicate["replicate_sha256"] for replicate in replicates],
    }


def calibrate_neutral_drift_reference_v1(
    *,
    seeds: Sequence[int],
    replicate_count: int,
) -> dict[str, Any]:
    """Return content-addressed per-state, per-generation calibration bands.

    ``replicate_count`` is the number of paired replicates per caller seed and
    founder state. The same derived RNG stream is used for all four states.
    """

    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise ValueError("seeds must be a non-empty sequence of integers")
    canonical_seeds = tuple(
        sorted(
            _validate_seed(seed, name=f"seeds[{index}]")
            for index, seed in enumerate(seeds)
        )
    )
    if not canonical_seeds:
        raise ValueError("seeds must be a non-empty sequence of integers")
    if len(set(canonical_seeds)) != len(canonical_seeds):
        raise ValueError("seeds must be unique")
    if type(replicate_count) is not int or replicate_count <= 0:
        raise ValueError("replicate_count must be a positive integer")
    replicates_per_state = len(canonical_seeds) * replicate_count
    total_state_replicates = replicates_per_state * len(FOUNDER_STATES)
    if total_state_replicates > MAX_TOTAL_REPLICATES:
        raise ValueError(
            f"total state replicate count must not exceed {MAX_TOTAL_REPLICATES}"
        )

    state_calibrations = {
        state: _state_calibration(
            founder_state=state,
            seeds=canonical_seeds,
            replicate_count=replicate_count,
        )
        for state in FOUNDER_STATES
    }
    unsigned = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "reference_schema_version": SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "caller_seeds": list(canonical_seeds),
        "replicates_per_seed_per_state": replicate_count,
        "replicates_per_state": replicates_per_state,
        "total_state_replicates": total_state_replicates,
        "state_calibrations": state_calibrations,
        "claims_boundary": _CONTRACT["claims_boundary"],
    }
    return {**unsigned, "receipt_sha256": canonical_json_sha256(unsigned)}


__all__ = [
    "ALL_HETEROZYGOUS_AB",
    "BALANCED_HOMOZYGOUS_AA_BB",
    "CONTRACT_SHA256",
    "DIPLOID_COPY_COUNT",
    "FOUNDER_STATES",
    "HOMOZYGOUS_AA_CONTROL",
    "HOMOZYGOUS_BB_CONTROL",
    "MAX_TOTAL_REPLICATES",
    "METRIC_FIELDS",
    "OFFSPRING_COUNT",
    "POPULATION_SIZE",
    "RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SURVIVOR_COUNT",
    "TERMINAL_GENERATION",
    "calibrate_neutral_drift_reference_v1",
    "neutral_reference_contract_v1",
    "simulate_neutral_drift_replicate_v1",
]
