from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from engine.qualification_plans import (
    QualificationPlanError,
    compile_matrix,
    load_matrix,
    load_plan,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "registries/qualification_plans"


def _plan() -> dict[str, object]:
    return {
        "schema_version": "eiel_qualification_plan_v1",
        "plan_id": "unit-matrix-v1",
        "description": "Unit-only matrix.",
        "capability_requirements": ["base-capability"],
        "dimensions": [
            {"id": "horizon", "values": [2, 8]},
            {"id": "mode", "values": ["exact"]},
        ],
        "cases": [
            {
                "id": "short",
                "assignments": {"horizon": 2, "mode": "exact"},
            },
            {
                "id": "long",
                "assignments": {"horizon": 8, "mode": "exact"},
                "prerequisites": ["short"],
                "capability_requirements": ["long-horizon"],
            },
        ],
    }


def test_json_and_yaml_have_one_canonical_model(tmp_path: Path) -> None:
    raw = _plan()
    json_path = tmp_path / "plan.json"
    yaml_path = tmp_path / "plan.yaml"
    json_path.write_text(json.dumps(raw), encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert load_plan(json_path) == load_plan(yaml_path) == validate_plan(raw)


def test_representative_registry_plans_load_without_execution_fields() -> None:
    loaded = [
        load_plan(REGISTRY / "resident_horizon_ladder_v1.yaml"),
        load_plan(REGISTRY / "flat_batch_order_v1.json"),
    ]

    assert [plan["plan_id"] for plan in loaded] == [
        "resident-horizon-ladder-v1",
        "flat-batch-order-v1",
    ]
    rendered = json.dumps(loaded, sort_keys=True)
    for forbidden in ("argv", "command", "executable", "production_routing"):
        assert forbidden not in rendered


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [
        (
            ".json",
            '{"schema_version":"eiel_qualification_plan_v1",'
            '"schema_version":"duplicate","plan_id":"x","dimensions":[],"cases":[]}',
        ),
        (
            ".yaml",
            "schema_version: eiel_qualification_plan_v1\n"
            "schema_version: duplicate\nplan_id: x\ndimensions: []\ncases: []\n",
        ),
    ],
)
def test_load_rejects_duplicate_fields(tmp_path: Path, suffix: str, payload: str) -> None:
    path = tmp_path / f"duplicate{suffix}"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(QualificationPlanError, match="duplicate"):
        load_plan(path)


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [
        (
            ".json",
            '{"schema_version":"eiel_qualification_plan_v1","plan_id":"x",'
            '"dimensions":[{"id":"size","values":[NaN]}],'
            '"cases":[{"id":"case","assignments":{"size":NaN}}]}',
        ),
        (
            ".yaml",
            "schema_version: eiel_qualification_plan_v1\nplan_id: x\n"
            "dimensions:\n  - id: size\n    values: [.nan]\n"
            "cases:\n  - id: case\n    assignments: {size: .nan}\n",
        ),
    ],
)
def test_load_rejects_non_finite_numbers(
    tmp_path: Path, suffix: str, payload: str
) -> None:
    path = tmp_path / f"nonfinite{suffix}"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(QualificationPlanError, match="non-finite|finite"):
        load_plan(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda plan: plan.update(command=["run"]), "unsupported fields"),
        (lambda plan: plan.update(schema_version="v2"), "schema_version"),
        (lambda plan: plan["dimensions"].append(copy.deepcopy(plan["dimensions"][0])), "duplicate dimension"),
        (lambda plan: plan["cases"].append(copy.deepcopy(plan["cases"][0])), "duplicate case"),
        (lambda plan: plan["cases"][0]["assignments"].pop("mode"), "assignments mismatch"),
        (lambda plan: plan["cases"][0]["assignments"].update(horizon=32), "unsupported"),
        (lambda plan: plan["cases"][1].update(prerequisites=["missing"]), "unknown prerequisite"),
    ],
)
def test_model_fails_closed_on_unknown_or_inconsistent_data(mutate, message: str) -> None:
    plan = _plan()
    mutate(plan)
    with pytest.raises(QualificationPlanError, match=message):
        validate_plan(plan)


def test_model_rejects_prerequisite_cycles() -> None:
    plan = _plan()
    plan["cases"][0]["prerequisites"] = ["long"]
    with pytest.raises(QualificationPlanError, match="cycle"):
        validate_plan(plan)


def test_long_prerequisite_chain_uses_iterative_cycle_validation() -> None:
    cases = []
    for index in range(1_500):
        case_id = f"case-{index}"
        cases.append(
            {
                "id": case_id,
                "assignments": {"position": index},
                "prerequisites": [] if index == 0 else [f"case-{index - 1}"],
            }
        )
    plan = {
        "schema_version": "eiel_qualification_plan_v1",
        "plan_id": "long-chain-v1",
        "dimensions": [{"id": "position", "values": list(range(1_500))}],
        "cases": cases,
    }

    canonical = validate_plan(plan)

    assert len(canonical["cases"]) == 1_500


def test_canonical_output_has_stable_ordering() -> None:
    plan = _plan()
    plan["dimensions"].reverse()
    plan["cases"].reverse()
    plan["capability_requirements"] = ["z-capability", "a-capability"]

    canonical = validate_plan(plan)

    assert [item["id"] for item in canonical["dimensions"]] == ["horizon", "mode"]
    assert [item["id"] for item in canonical["cases"]] == ["long", "short"]
    assert canonical["capability_requirements"] == ["a-capability", "z-capability"]


def _matrix() -> dict[str, object]:
    return {
        "schema_version": "eiel_qualification_matrix_v1",
        "plan_id": "compiled-matrix-v1",
        "description": "Declarative compiled matrix.",
        "capability_requirements": ["base-capability"],
        "case_id_prefix": "matrix-case",
        "dimensions": [
            {"id": "horizon", "values": [8, 2]},
            {"id": "mode", "values": ["second", "first"]},
        ],
        "include": [],
        "exclude": [],
        "templates": [],
    }


def _case_by_assignment(
    plan: dict[str, object], **assignment: object
) -> dict[str, object]:
    return next(
        case
        for case in plan["cases"]  # type: ignore[union-attr]
        if case["assignments"] == assignment
    )


def test_matrix_compilation_is_stable_across_dimension_and_value_order() -> None:
    first = _matrix()
    reordered = copy.deepcopy(first)
    reordered["dimensions"].reverse()  # type: ignore[union-attr]
    for dimension in reordered["dimensions"]:  # type: ignore[union-attr]
        dimension["values"].reverse()

    compiled = compile_matrix(first)
    repeated = compile_matrix(reordered)

    assert compiled == repeated
    assert len(compiled["cases"]) == 4
    assert compiled["schema_version"] == "eiel_qualification_plan_v1"
    assert all(case["id"].startswith("matrix-case-") for case in compiled["cases"])


def test_versioned_matrix_example_loads_and_compiles() -> None:
    compiled = load_matrix(REGISTRY / "bounded_capacity_matrix_v1.yaml")

    assert compiled["plan_id"] == "bounded-capacity-matrix-v1"
    assert len(compiled["cases"]) == 5


def test_include_and_exclude_are_declarative_equality_selectors() -> None:
    matrix = _matrix()
    matrix["include"] = [{"horizon": 8}, {"mode": "first"}]
    matrix["exclude"] = [{"horizon": 8, "mode": "second"}]

    compiled = compile_matrix(matrix)

    assert [case["assignments"] for case in compiled["cases"]] == [
        {"horizon": 2, "mode": "first"},
        {"horizon": 8, "mode": "first"},
    ]


def test_fully_specified_include_avoids_unselected_cartesian_expansion() -> None:
    matrix = _matrix()
    matrix["dimensions"] = [
        {"id": f"axis_{axis}", "values": list(range(10))}
        for axis in range(8)
    ]
    matrix["include"] = [
        {f"axis_{axis}": value for axis in range(8)}
        for value in (1, 7)
    ]
    matrix["max_cases"] = 2

    compiled = compile_matrix(matrix)

    assert {
        tuple(case["assignments"].items()) for case in compiled["cases"]
    } == {
        tuple({f"axis_{axis}": value for axis in range(8)}.items())
        for value in (1, 7)
    }


def test_partial_include_still_fails_closed_on_oversized_cartesian_space() -> None:
    matrix = _matrix()
    matrix["dimensions"] = [
        {"id": f"axis_{axis}", "values": list(range(10))}
        for axis in range(8)
    ]
    matrix["include"] = [{"axis_0": 1}]
    matrix["max_cases"] = 2

    with pytest.raises(
        QualificationPlanError,
        match="matrix Cartesian product exceeds max_cases",
    ):
        compile_matrix(matrix)


def test_templates_add_capabilities_description_and_exact_prerequisite() -> None:
    matrix = _matrix()
    matrix["templates"] = [
        {
            "selector": {"horizon": 8},
            "capability_requirements": ["long-horizon"],
        },
        {
            "selector": {"horizon": 8, "mode": "first"},
            "description": "First long case.",
            "prerequisites": [{"horizon": 2, "mode": "first"}],
        },
    ]

    compiled = compile_matrix(matrix)
    short = _case_by_assignment(compiled, horizon=2, mode="first")
    long = _case_by_assignment(compiled, horizon=8, mode="first")
    other_long = _case_by_assignment(compiled, horizon=8, mode="second")

    assert compiled["capability_requirements"] == ["base-capability"]
    assert long["description"] == "First long case."
    assert long["prerequisites"] == [short["id"]]
    assert long["capability_requirements"] == ["long-horizon"]
    assert other_long["capability_requirements"] == ["long-horizon"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schema_version="v2"), "schema_version"),
        (lambda value: value.update(command=["run"]), "unsupported fields"),
        (lambda value: value.update(include=[{}]), "must not be empty"),
        (lambda value: value.update(include=[{"missing": 1}]), "unknown dimensions"),
        (lambda value: value.update(include=[{"horizon": 99}]), "unsupported"),
        (
            lambda value: value.update(
                include=[{"horizon": 2}], exclude=[{"horizon": 2}]
            ),
            "no cases",
        ),
    ],
)
def test_matrix_and_selectors_fail_closed(mutate, message: str) -> None:
    matrix = _matrix()
    mutate(matrix)

    with pytest.raises(QualificationPlanError, match=message):
        compile_matrix(matrix)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_matrix_case_limit_must_be_a_positive_integer(limit: object) -> None:
    matrix = _matrix()
    matrix["max_cases"] = limit

    with pytest.raises(QualificationPlanError, match="max_cases"):
        compile_matrix(matrix)


def test_matrix_case_limit_is_checked_before_expansion() -> None:
    matrix = _matrix()
    matrix["max_cases"] = 3

    with pytest.raises(QualificationPlanError, match="Cartesian product"):
        compile_matrix(matrix)


def test_prerequisite_selector_must_resolve_exactly_one_generated_case() -> None:
    matrix = _matrix()
    matrix["templates"] = [
        {
            "selector": {"horizon": 8},
            "prerequisites": [{"horizon": 2}],
        }
    ]

    with pytest.raises(QualificationPlanError, match="exactly one"):
        compile_matrix(matrix)


def test_templates_reject_conflicting_descriptions() -> None:
    matrix = _matrix()
    matrix["templates"] = [
        {"selector": {"horizon": 8}, "description": "Long."},
        {
            "selector": {"horizon": 8, "mode": "first"},
            "description": "Different.",
        },
    ]

    with pytest.raises(QualificationPlanError, match="conflicting"):
        compile_matrix(matrix)


def test_compiled_prerequisite_cycles_fail_closed() -> None:
    matrix = _matrix()
    matrix["templates"] = [
        {
            "selector": {"horizon": 2, "mode": "first"},
            "prerequisites": [{"horizon": 8, "mode": "first"}],
        },
        {
            "selector": {"horizon": 8, "mode": "first"},
            "prerequisites": [{"horizon": 2, "mode": "first"}],
        },
    ]

    with pytest.raises(QualificationPlanError, match="cycle"):
        compile_matrix(matrix)
