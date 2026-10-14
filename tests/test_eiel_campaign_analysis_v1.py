from __future__ import annotations

import copy

import pytest

from engine.campaigns.analysis import (
    CampaignAnalysisError,
    analyze_campaign,
    render_analysis_markdown,
)


def _plan(*, aggregate: str = "mean", minimum_coverage: float = 1.0) -> dict:
    cells = []
    ordinal = 0
    for seed in (11, 22):
        for arm in ("control", "treatment"):
            cells.append(
                {
                    "ordinal": ordinal,
                    "cell_id": f"cell-{ordinal}",
                    "branch_id": f"branch-{ordinal}",
                    "rng_lineage_id": f"rng-{ordinal}",
                    "dimensions": {"arm": arm, "ecology_seed": seed},
                    "parameters": {},
                    "seeds": {"ecology": seed, "task": 7},
                }
            )
            ordinal += 1
    return {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": "analysis-contract",
        "plan_sha256": "a" * 64,
        "cells": cells,
        "analysis": {
            "required_metrics": ["outcome"],
            "optional_metrics": ["ratio"],
            "rules": [
                {
                    "id": "outcome-rule",
                    "metric": "outcome",
                    "aggregate": aggregate,
                    "group_by": ["arm"],
                    "pair_by": ["seed.ecology"],
                    "operator": ">=",
                    "threshold": 0,
                    "minimum_coverage": minimum_coverage,
                }
            ],
        },
    }


def _rows(values: tuple[int | float | None, ...], ratios: tuple[float | None, ...] | None = None) -> list[dict]:
    ratios = ratios or (None,) * len(values)
    return [
        {"cell_id": f"cell-{index}", "metrics": {"outcome": value, "ratio": ratios[index]}}
        for index, value in enumerate(values)
    ]


@pytest.mark.parametrize(
    ("aggregate", "values", "expected"),
    [
        ("sum", (0, 2, 0, 4), {"control": 0, "treatment": 6}),
        ("mean", (0, 2, 0, 4), {"control": 0, "treatment": 3}),
        ("count", (0, 2, 0, 4), {"control": 2, "treatment": 2}),
    ],
)
def test_typed_aggregates_group_by_compiled_dimensions_and_measure_zero(
    aggregate: str, values: tuple[int, ...], expected: dict[str, int]
) -> None:
    result = analyze_campaign(_plan(aggregate=aggregate), _rows(values))

    groups = {group["key"]["arm"]: group for group in result["rules"][0]["groups"]}
    assert {arm: row["value"] for arm, row in groups.items()} == expected
    assert all(row["measured"] == 2 for row in groups.values())
    assert result["decision"] == "pass"


def test_null_ratio_stays_unavailable_and_optional_metric_does_not_false_fail() -> None:
    result = analyze_campaign(
        _plan(),
        _rows((0, 1, 0, 1), ratios=(None, None, 0.0, None)),
    )

    assert result["metric_coverage"]["ratio"] == {
        "required": False,
        "expected": 4,
        "measured": 1,
        "coverage": 0.25,
        "status": "not_measured",
    }
    assert result["decision"] == "pass"


def test_missing_required_coverage_is_not_measured_not_pass() -> None:
    result = analyze_campaign(_plan(minimum_coverage=0.5), _rows((0, 1, None, 1)))

    assert result["rules"][0]["status"] == "pass"
    assert result["metric_coverage"]["outcome"]["status"] == "not_measured"
    assert result["decision"] == "not_measured"


def test_threshold_failure_wins_over_other_unmeasured_groups() -> None:
    plan = _plan(minimum_coverage=1.0)
    plan["analysis"]["rules"][0]["threshold"] = 1
    result = analyze_campaign(plan, _rows((0, None, 0, 0)))

    assert result["rules"][0]["status"] == "fail"
    assert result["decision"] == "fail"


def test_paired_delta_uses_pair_seed_and_deterministic_sorted_arm_direction() -> None:
    plan = _plan(aggregate="paired_delta")
    plan["analysis"]["rules"][0]["threshold"] = 2
    result = analyze_campaign(plan, _rows((1, 4, 2, 6)))

    rule = result["rules"][0]
    assert rule["status"] == "pass"
    assert rule["groups"] == [
        {
            "key": {"left": {"arm": "control"}, "right": {"arm": "treatment"}},
            "expected": 2,
            "measured": 2,
            "coverage": 1.0,
            "value": 3.5,
            "status": "pass",
        }
    ]


def test_paired_delta_fails_closed_when_pair_key_is_not_unique() -> None:
    plan = _plan(aggregate="paired_delta")
    plan["analysis"]["rules"][0]["pair_by"] = ["seed.task"]
    result = analyze_campaign(plan, _rows((1, 4, 2, 6)))

    assert result["rules"][0]["status"] == "not_measured"
    assert result["rules"][0]["reason"] == "pair_keys_do_not_identify_exactly_one_cell_per_arm"
    assert result["decision"] == "not_measured"


def test_analysis_is_deterministic_across_row_order() -> None:
    rows = _rows((0, 1, 2, 3))
    left = analyze_campaign(_plan(), rows)
    right = analyze_campaign(copy.deepcopy(_plan()), list(reversed(rows)))

    assert left == right
    assert render_analysis_markdown(left) == render_analysis_markdown(right)
    assert "Decision: **pass**" in render_analysis_markdown(left)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.append(copy.deepcopy(rows[0])), "duplicate metric row"),
        (lambda rows: rows[0].update(cell_id="unknown"), "not a compiled cell"),
        (lambda rows: rows[0]["metrics"].update(outcome=True), "finite number or null"),
        (lambda rows: rows[0].update(extra=True), "unsupported fields"),
    ],
)
def test_metric_rows_fail_closed(mutation, message: str) -> None:
    rows = _rows((0, 1, 2, 3))
    mutation(rows)

    with pytest.raises(CampaignAnalysisError, match=message):
        analyze_campaign(_plan(), rows)


def test_unknown_dimension_or_seed_key_fails_closed() -> None:
    plan = _plan()
    plan["analysis"]["rules"][0]["group_by"] = ["missing"]

    with pytest.raises(CampaignAnalysisError, match="absent from compiled cell"):
        analyze_campaign(plan, _rows((0, 1, 2, 3)))
