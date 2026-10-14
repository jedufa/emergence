from __future__ import annotations

import pytest

from tools.scripts.eiel_eed_b1_reducer_v1 import (
    ECOLOGY_SEEDS,
    FOUNDERS,
    REQUIRED_METRICS,
    TREATMENTS,
    ZERO_POPULATION_UNDEFINED_METRICS,
    EedB1ReductionError,
    _positive_admission_index,
    paired_contrasts,
    paired_summary,
    validate_required_metrics,
)


def test_paired_summary_preserves_values_and_sign_counts() -> None:
    result = paired_summary([-2.0, -1.0, 0.0, 1.0, 2.0, 6.0])

    assert result == {
        "count": 6,
        "values": [-2.0, -1.0, 0.0, 1.0, 2.0, 6.0],
        "mean": 1.0,
        "median": 0.5,
        "minimum": -2.0,
        "maximum": 6.0,
        "positive": 3,
        "zero": 1,
        "negative": 2,
    }


def _cells() -> dict[str, dict[tuple[str, int], dict[str, float]]]:
    result: dict[str, dict[tuple[str, int], dict[str, float]]] = {}
    for treatment_ordinal, treatment in enumerate(TREATMENTS):
        rows: dict[tuple[str, int], dict[str, float]] = {}
        for founder_ordinal, founder in enumerate(FOUNDERS):
            for seed in ECOLOGY_SEEDS:
                rows[(founder, seed)] = {
                    metric: float(
                        treatment_ordinal * 1000
                        + founder_ordinal * 100
                        + seed
                        + metric_ordinal
                    )
                    for metric_ordinal, metric in enumerate(REQUIRED_METRICS)
                }
        result[treatment] = rows
    return result


def _metric_row(*, alive: int) -> dict[str, float | None]:
    row = {metric: 1.0 for metric in REQUIRED_METRICS}
    row["population.alive"] = float(alive)
    if alive == 0:
        for metric in ZERO_POPULATION_UNDEFINED_METRICS:
            row[metric] = None
    return row


def test_required_metrics_allow_only_zero_population_undefined_per_capita() -> None:
    extinct = validate_required_metrics(_metric_row(alive=0), name="extinct")

    assert extinct["zero_population_terminal"] is True
    assert set(extinct["undefined_zero_population_metrics"]) == set(
        ZERO_POPULATION_UNDEFINED_METRICS
    )

    live = _metric_row(alive=3)
    live["eed.resource_a_stock_per_capita"] = None
    with pytest.raises(
        EedB1ReductionError,
        match="eed.resource_a_stock_per_capita is missing or undefined",
    ):
        validate_required_metrics(live, name="live")


def test_paired_contrasts_keep_founders_and_seeds_paired() -> None:
    result = paired_contrasts(_cells())

    terminal = result["total_enabled_impact"]
    assert terminal["left_treatment"] == "impact_on"
    assert terminal["right_treatment"] == "impact_off"
    for founder in FOUNDERS:
        summary = terminal["founders"][founder]["population.alive"]
        assert summary["count"] == 6
        assert summary["values"] == [-1000.0] * 6
        assert summary["negative"] == 6


def test_paired_contrasts_classify_extinction_undefined_pairs() -> None:
    cells = _cells()
    for treatment in TREATMENTS:
        cells[treatment][("a_high", 101)] = _metric_row(alive=0)
    result = paired_contrasts(cells, allow_zero_population_undefined=True)

    per_capita = result["total_enabled_impact"]["founders"]["a_high"][
        "eed.resource_a_stock_per_capita"
    ]
    assert per_capita["expected_count"] == 6
    assert per_capita["finite_pair_count"] == 5
    assert per_capita["undefined_pair_count"] == 1
    assert per_capita["undefined_pairs"] == [
        {
            "founder_profile": "a_high",
            "ecology_seed": 101,
            "left_zero_population_undefined": True,
            "right_zero_population_undefined": True,
        }
    ]


def test_paired_contrasts_fail_closed_on_missing_seed() -> None:
    cells = _cells()
    del cells["phenotype_fixed"][("symmetric_mix", 606)]

    with pytest.raises(EedB1ReductionError, match="pairing mismatch"):
        paired_contrasts(cells)


def test_positive_admission_manifest_requires_frozen_censoring_rules() -> None:
    manifest = {
        "schema_version": "eiel_eed_b1_positive_scientific_admission_manifest_v1",
        "status": "frozen",
        "scientific_cell_count": 72,
        "censoring_and_interruption_rules": {},
        "admitted_items": [],
    }

    with pytest.raises(
        EedB1ReductionError,
        match="censoring and interruption rules",
    ):
        _positive_admission_index(manifest)
