from __future__ import annotations

import math

from tools.evidence.paired import ComparisonSpec, compare_paired


def _rows() -> list[dict]:
    rows: list[dict] = []
    for context, normal_values, ablated_values in (
        ("flat", [10.0, 12.0, 14.0], [7.0, 8.0, 9.0]),
        ("pressure", [20.0, 22.0], [15.0, 17.0]),
    ):
        for idx, value in enumerate(normal_values):
            rows.append(
                {
                    "run_id": f"{context}-normal-{idx}",
                    "ecology_context": context,
                    "init_arm": "legacy",
                    "perception_condition": "normal",
                    "seed": idx,
                    "alive": value,
                    "mean_offspring_total": value / 10.0,
                }
            )
        for idx, value in enumerate(ablated_values):
            rows.append(
                {
                    "run_id": f"{context}-ablated-{idx}",
                    "ecology_context": context,
                    "init_arm": "legacy",
                    "perception_condition": "ablated",
                    "seed": idx,
                    "alive": value,
                    "mean_offspring_total": value / 10.0,
                }
            )
    rows.append(
        {
            "run_id": "pressure-normal-missing",
            "ecology_context": "pressure",
            "init_arm": "legacy",
            "perception_condition": "normal",
            "seed": 99,
            "alive": 30.0,
            "mean_offspring_total": 3.0,
        }
    )
    return rows


def test_normal_minus_ablated_deltas_are_descriptive_and_deterministic() -> None:
    spec = ComparisonSpec(
        name="normal_vs_ablated",
        metric_names=("alive", "mean_offspring_total"),
        group_by_dimensions=("ecology_context", "init_arm"),
        pair_by_dimensions=("seed",),
        left_filter={"perception_condition": "normal"},
        right_filter={"perception_condition": "ablated"},
        delta_label="normal_minus_ablated",
    )

    result = compare_paired(_rows(), spec)
    flat_summary = next(row for row in result["summary"] if row["ecology_context"] == "flat")
    assert flat_summary["alive"]["mean_delta"] == 4.0
    assert flat_summary["alive"]["median_delta"] == 4.0
    assert flat_summary["alive"]["min_delta"] == 3.0
    assert flat_summary["alive"]["max_delta"] == 5.0
    assert math.isclose(flat_summary["alive"]["std_delta"], 1.0)
    assert flat_summary["alive"]["direction_consistency"] == 1.0

    pressure_summary = next(row for row in result["summary"] if row["ecology_context"] == "pressure")
    assert pressure_summary["alive"]["paired_seed_coverage"] == 2
    assert pressure_summary["missing_pairs"] == [{"ecology_context": "pressure", "init_arm": "legacy", "seed": 99}]


def test_legacy_vs_founder_balanced_comparison_uses_generic_filters() -> None:
    rows = [
        {"run_id": "l1", "ecology_context": "flat", "init_arm": "legacy", "seed": 1, "alive": 5},
        {"run_id": "b1", "ecology_context": "flat", "init_arm": "founder", "seed": 1, "alive": 8},
        {"run_id": "l2", "ecology_context": "flat", "init_arm": "legacy", "seed": 2, "alive": 6},
        {"run_id": "b2", "ecology_context": "flat", "init_arm": "founder", "seed": 2, "alive": 9},
    ]
    spec = ComparisonSpec(
        name="founder_vs_legacy",
        metric_names=("alive",),
        group_by_dimensions=("ecology_context",),
        pair_by_dimensions=("seed",),
        left_filter={"init_arm": "founder"},
        right_filter={"init_arm": "legacy"},
        delta_label="founder_minus_legacy",
    )
    result = compare_paired(rows, spec)
    assert result["summary"][0]["alive"]["mean_delta"] == 3.0
    assert [row["founder_minus_legacy_alive"] for row in result["per_pair_rows"]] == [3.0, 3.0]
