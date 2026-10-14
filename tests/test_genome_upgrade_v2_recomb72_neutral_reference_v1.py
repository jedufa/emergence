from __future__ import annotations

import pytest

from engine.evolution.genome_upgrade_v2_recomb72_neutral_reference_v1 import (
    ALL_HETEROZYGOUS_AB,
    BALANCED_HOMOZYGOUS_AA_BB,
    CONTRACT_SHA256,
    FOUNDER_STATES,
    HOMOZYGOUS_AA_CONTROL,
    HOMOZYGOUS_BB_CONTROL,
    MAX_TOTAL_REPLICATES,
    METRIC_FIELDS,
    calibrate_neutral_drift_reference_v1,
    neutral_reference_contract_v1,
    simulate_neutral_drift_replicate_v1,
)
from engine.evolution.phase_f_v2.canonical import canonical_json_sha256


def _replicate(state: str) -> dict[str, object]:
    return simulate_neutral_drift_replicate_v1(
        founder_state=state,
        seed=101,
        replicate_ordinal=3,
    )


def test_contract_freezes_exact_recomb72_gv2_v1_states_and_schedule() -> None:
    contract = neutral_reference_contract_v1()

    assert (
        CONTRACT_SHA256
        == "1d7d5b2e51abaf54cfedc625f9a7e7420634ba63ea7a5977b6800d3b2007d70c"
    )
    assert canonical_json_sha256(contract) == CONTRACT_SHA256
    assert contract["campaign_id"] == "RECOMB-72-GV2-v1"
    assert contract["production_inheritance_mechanic"] is False
    assert contract["campaign_outcomes_consumed"] is False
    assert tuple(contract["founder_states"]) == FOUNDER_STATES
    assert contract["founder_states"] == {
        "all_heterozygous_ab": "64_AB_diploids",
        "balanced_homozygous_aa_bb": "32_AA_then_32_BB_diploids",
        "homozygous_aa_control": "64_AA_diploids",
        "homozygous_bb_control": "64_BB_diploids",
    }
    assert contract["population_schedule"] == {
        "generation_range_inclusive": [0, 30],
        "population_size": 64,
        "ploidy": 2,
        "survivors_per_transition": 32,
        "offspring_per_transition": 32,
        "survivor_selection": "uniform_without_replacement_genotype_blind",
        "mating_pool": "retained_survivors_only",
        "mating": "two_distinct_parents_uniform_without_replacement_per_offspring",
        "gamete": "one_homolog_uniform_from_each_parent",
        "mutation": "disabled",
    }


def test_g0_metrics_match_the_four_exact_founder_states() -> None:
    expected = {
        ALL_HETEROZYGOUS_AB: (64, 64, 1.0, 0.5),
        BALANCED_HOMOZYGOUS_AA_BB: (64, 64, 0.0, 0.5),
        HOMOZYGOUS_AA_CONTROL: (128, 0, 0.0, 0.0),
        HOMOZYGOUS_BB_CONTROL: (0, 128, 0.0, 0.0),
    }
    for state, (
        a_count,
        b_count,
        observed,
        expected_heterozygosity,
    ) in expected.items():
        row = _replicate(state)["generations"][0]
        assert row["a_allele_count"] == a_count
        assert row["b_allele_count"] == b_count
        assert row["observed_heterozygosity"] == observed
        assert row["expected_heterozygosity"] == expected_heterozygosity
        assert row["surviving_founder_count"] == 64
        assert row["founder_concentration"] == 1.0 / 64.0


@pytest.mark.parametrize(
    ("state", "a_count", "b_count", "a_fixed", "b_fixed"),
    [
        (HOMOZYGOUS_AA_CONTROL, 128, 0, 1, 0),
        (HOMOZYGOUS_BB_CONTROL, 0, 128, 0, 1),
    ],
)
def test_homozygous_controls_are_absorbing_for_every_generation(
    state: str,
    a_count: int,
    b_count: int,
    a_fixed: int,
    b_fixed: int,
) -> None:
    replicate = _replicate(state)

    assert len(replicate["generations"]) == 31
    for generation, row in enumerate(replicate["generations"]):
        assert row["generation"] == generation
        assert row["a_allele_count"] == a_count
        assert row["b_allele_count"] == b_count
        assert row["a_fixed"] == a_fixed
        assert row["b_fixed"] == b_fixed
        assert row["observed_heterozygosity"] == 0.0
        assert row["expected_heterozygosity"] == 0.0


@pytest.mark.parametrize(
    "state",
    [ALL_HETEROZYGOUS_AB, BALANCED_HOMOZYGOUS_AA_BB],
)
def test_variable_state_replicate_is_deterministic_and_well_formed(state: str) -> None:
    first = _replicate(state)
    second = _replicate(state)

    assert first == second
    unsigned = dict(first)
    supplied = unsigned.pop("replicate_sha256")
    assert canonical_json_sha256(unsigned) == supplied
    assert tuple(first["generations"][0].keys()) == ("generation", *METRIC_FIELDS)
    for row in first["generations"]:
        assert row["a_allele_count"] + row["b_allele_count"] == 128
        assert row["a_allele_frequency"] + row["b_allele_frequency"] == 1.0
        assert row["a_retained"] == int(row["a_allele_count"] > 0)
        assert row["b_retained"] == int(row["b_allele_count"] > 0)
        assert row["a_fixed"] + row["b_fixed"] <= 1
        assert 0.0 <= row["observed_heterozygosity"] <= 1.0
        assert 0.0 <= row["expected_heterozygosity"] <= 0.5
        assert 1 <= row["surviving_founder_count"] <= 64
        assert 1.0 / 64.0 <= row["founder_concentration"] <= 1.0


def test_founder_states_share_one_paired_random_schedule() -> None:
    derived_seeds = {_replicate(state)["derived_seed"] for state in FOUNDER_STATES}

    assert len(derived_seeds) == 1


def test_calibration_is_canonical_and_exposes_per_generation_state_bands() -> None:
    first = calibrate_neutral_drift_reference_v1(
        seeds=[202, 101],
        replicate_count=2,
    )
    second = calibrate_neutral_drift_reference_v1(
        seeds=[101, 202],
        replicate_count=2,
    )

    assert first == second
    assert first["caller_seeds"] == [101, 202]
    assert first["replicates_per_seed_per_state"] == 2
    assert first["replicates_per_state"] == 4
    assert first["total_state_replicates"] == 16
    assert tuple(first["state_calibrations"]) == FOUNDER_STATES
    assert (
        first["receipt_sha256"]
        == "6ecf14eac3f01f4730979d03a468eba8f3d88824ef2db34e60d34db17b7f6a05"
    )
    unsigned = dict(first)
    supplied = unsigned.pop("receipt_sha256")
    assert canonical_json_sha256(unsigned) == supplied

    for state in FOUNDER_STATES:
        calibration = first["state_calibrations"][state]
        assert len(calibration["generation_bands"]) == 31
        assert len(calibration["replicate_sha256"]) == 4
        for generation, row in enumerate(calibration["generation_bands"]):
            assert row["generation"] == generation
            assert set(row["bands"]) == set(METRIC_FIELDS)
            for band in row["bands"].values():
                assert (
                    band["observed_minimum"]
                    <= band["q025"]
                    <= band["median"]
                    <= band["q975"]
                    <= band["observed_maximum"]
                )


@pytest.mark.parametrize(
    ("state", "fixed_metric", "lost_metric"),
    [
        (HOMOZYGOUS_AA_CONTROL, "a_fixed", "b_fixed"),
        (HOMOZYGOUS_BB_CONTROL, "b_fixed", "a_fixed"),
    ],
)
def test_control_calibration_bands_remain_exact(
    state: str,
    fixed_metric: str,
    lost_metric: str,
) -> None:
    calibration = calibrate_neutral_drift_reference_v1(
        seeds=[101],
        replicate_count=2,
    )["state_calibrations"][state]

    for row in calibration["generation_bands"]:
        assert set(row["bands"][fixed_metric].values()) == {1, 1.0}
        assert set(row["bands"][lost_metric].values()) == {0, 0.0}
        assert set(row["bands"]["observed_heterozygosity"].values()) == {0.0}
        assert set(row["bands"]["expected_heterozygosity"].values()) == {0.0}
    assert (
        calibration["terminal_fixation_summary"][fixed_metric]["observed_rate"] == 1.0
    )
    assert calibration["terminal_fixation_summary"][lost_metric]["observed_rate"] == 0.0


def test_replicate_rejects_unknown_founder_state() -> None:
    with pytest.raises(ValueError, match="unsupported neutral founder state"):
        simulate_neutral_drift_replicate_v1(
            founder_state="unknown",
            seed=101,
        )


@pytest.mark.parametrize(
    ("seeds", "replicate_count"),
    [
        ([], 1),
        ([True], 1),
        ([-1], 1),
        ([1, 1], 1),
        ([1], 0),
        ([1], True),
        ([0], MAX_TOTAL_REPLICATES),
    ],
)
def test_calibration_rejects_invalid_or_unbounded_requests(
    seeds: list[int],
    replicate_count: int,
) -> None:
    with pytest.raises(ValueError):
        calibrate_neutral_drift_reference_v1(
            seeds=seeds,
            replicate_count=replicate_count,
        )
