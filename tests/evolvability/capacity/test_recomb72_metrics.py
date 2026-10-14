from __future__ import annotations

import math

import pytest

from evolution.recomb72_metrics import (
    advance_founder_ancestry,
    ancestry_concentration,
    canonical_scalar_token,
    locus_inheritance_statistics,
    locus_statistics,
    package_statistics,
    paired_milestone_summaries,
    reproduction_concentration,
    time_to_homogenization,
)


def test_locus_statistics_freezes_population_moments_and_tolerances() -> None:
    result = locus_statistics(
        (-1.0, -1.0, 1.0, 1.0),
        lower_bound=-1.0,
        upper_bound=1.0,
        generation_zero_variance=1.0,
    )

    assert result == {
        "count": 4,
        "mean": 0.0,
        "variance_population": 1.0,
        "standard_deviation_population": 1.0,
        "minimum": -1.0,
        "quartile_25": -1.0,
        "median": 0.0,
        "quartile_75": 1.0,
        "maximum": 1.0,
        "unique_canonical_value_count": 2,
        "mean_absolute_value": 1.0,
        "tolerance": 0.1,
        "fraction_near_lower_extreme": 0.5,
        "fraction_near_upper_extreme": 0.5,
        "fraction_near_either_extreme": 1.0,
        "fraction_near_neutral": 0.0,
        "fraction_below_neutral": 0.5,
        "fraction_above_neutral": 0.5,
        "variance_retention_ratio": 1.0,
    }
    assert canonical_scalar_token(-0.0) == canonical_scalar_token(0.0)


def test_locus_statistics_rejects_nonfinite_and_out_of_bounds_values() -> None:
    with pytest.raises(ValueError, match="outside"):
        locus_statistics((2.0,), lower_bound=-1.0, upper_bound=1.0)
    with pytest.raises(ValueError, match="finite"):
        locus_statistics((math.nan,), lower_bound=-1.0, upper_bound=1.0)


def test_locus_inheritance_statistics_records_distance_and_sign_behavior() -> None:
    result = locus_inheritance_statistics(
        (
            {"parent_a": -1.0, "parent_b": 1.0, "child": 0.0},
            {"parent_a": -1.0, "parent_b": -0.5, "child": 0.5},
        )
    )

    assert result["parent_pair_distance_mean"] == 1.25
    assert result["child_to_nearest_parent_distance_mean"] == 1.0
    assert result["child_to_parental_midpoint_distance_mean"] == 0.625
    assert result["child_sign_matches_either_parent_count"] == 0
    assert result["child_sign_flip_count"] == 1
    assert result["child_sign_indeterminate_count"] == 1


def test_pattern3_package_classification_and_correlation_are_exact() -> None:
    package_a = {"x": -1.0, "y": -2.0}
    package_b = {"x": 1.0, "y": 2.0}
    result = package_statistics(
        (
            package_a,
            package_b,
            {"x": -1.0, "y": 2.0},
            {"x": 0.0, "y": 0.0},
        ),
        package_a=package_a,
        package_b=package_b,
        bounds={"x": (-1.0, 1.0), "y": (-2.0, 2.0)},
    )

    assert result["intact_package_a_count"] == 1
    assert result["intact_package_b_count"] == 1
    assert result["recombinant_count"] == 1
    assert result["neither_package_count"] == 1
    assert result["intact_package_fraction"] == 0.5
    assert result["correlation_matrix"]["x"]["x"] == pytest.approx(1.0)
    assert result["correlation_matrix"]["x"]["y"] == pytest.approx(5.0 / 11.0)
    assert result["correlation_matrix"]["y"]["x"] == pytest.approx(5.0 / 11.0)


def test_founder_ancestry_reconstruction_and_concentration_are_reporting_only() -> None:
    ancestry = advance_founder_ancestry(
        {"a": {"fa": 1.0}, "b": {"fb": 1.0}},
        ({"child_id": "c", "parent_a_id": "a", "parent_b_id": "b"},),
    )
    ancestry = advance_founder_ancestry(
        ancestry,
        ({"child_id": "d", "parent_a_id": "c", "parent_b_id": "a"},),
    )

    assert ancestry["c"] == {"fa": 0.5, "fb": 0.5}
    assert ancestry["d"] == {"fa": 0.75, "fb": 0.25}
    summary = ancestry_concentration(("c", "d"), ancestry)
    assert summary["founder_ancestry_shares"] == {"fa": 0.625, "fb": 0.375}
    assert summary["top1_founder_id"] == "fa"
    assert summary["top1_founder_ancestry_share"] == 0.625
    assert summary["effective_founder_ancestry_count"] == pytest.approx(
        math.exp(-(0.625 * math.log(0.625) + 0.375 * math.log(0.375)))
    )


def test_reproduction_concentration_handles_skew_and_no_contributions() -> None:
    result = reproduction_concentration({"a": 3, "b": 1, "unused": 0})

    assert result["parent_contribution_total"] == 4
    assert result["contributing_parent_count"] == 2
    assert result["top1_parent_contribution_share"] == 0.75
    assert result["parent_contribution_hhi"] == 0.625
    assert result["effective_contributing_parent_count"] == 1.6
    assert reproduction_concentration({})["parent_contribution_hhi"] is None


def _paired_rows() -> list[dict[str, object]]:
    rows = []
    for generation in (5, 10, 20, 30):
        for seed in (101, 202):
            for mode, value in (
                ("current", 0.1 * seed),
                ("copy", 0.1 * seed + generation),
                ("package", 0.1 * seed + 2 * generation),
            ):
                rows.append(
                    {
                        "pattern": "single_extreme",
                        "mode": mode,
                        "seed": seed,
                        "generation": generation,
                        "metric": "variance_retention",
                        "value": value,
                    }
                )
    return rows


def test_paired_milestone_summaries_preserve_every_seed_and_alternative() -> None:
    result = paired_milestone_summaries(
        list(reversed(_paired_rows())),
        current_mode="current",
        alternative_modes=("copy", "package"),
        expected_seeds=(101, 202),
    )

    assert len(result["summaries"]) == 8
    g30_copy = next(
        row
        for row in result["summaries"]
        if row["generation"] == 30 and row["alternative_mode"] == "copy"
    )
    assert g30_copy["delta_mean"] == pytest.approx(30.0)
    assert g30_copy["delta_minimum"] == pytest.approx(30.0)
    assert g30_copy["positive_seed_count"] == 2
    assert g30_copy["same_direction_all_seeds"] is True
    assert [row["seed"] for row in g30_copy["pairs"]] == [101, 202]


def test_paired_milestone_summaries_fail_closed_on_missing_or_duplicate_pairs() -> None:
    missing = _paired_rows()
    missing.pop()
    with pytest.raises(ValueError, match="missing a required"):
        paired_milestone_summaries(
            missing,
            current_mode="current",
            alternative_modes=("copy", "package"),
            expected_seeds=(101, 202),
        )

    duplicate = _paired_rows()
    duplicate.append(dict(duplicate[0]))
    with pytest.raises(ValueError, match="duplicate paired"):
        paired_milestone_summaries(
            duplicate,
            current_mode="current",
            alternative_modes=("copy", "package"),
        )


def test_time_to_homogenization_requires_three_consecutive_generations() -> None:
    reached = time_to_homogenization(
        (
            {"generation": 0, "variance": 1.0},
            {"generation": 1, "variance": 0.09},
            {"generation": 2, "variance": 0.08},
            {"generation": 3, "variance": 0.11},
            {"generation": 4, "variance": 0.09},
            {"generation": 5, "variance": 0.08},
            {"generation": 6, "variance": 0.07},
        )
    )
    assert reached["status"] == "reached"
    assert reached["first_generation"] == 4
    assert reached["confirmation_generation"] == 6

    not_evaluable = time_to_homogenization(
        ({"generation": 0, "variance": 0.0},)
    )
    assert not_evaluable["status"] == "not_evaluable"
