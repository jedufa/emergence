from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from engine.campaigns.compiler import load_campaign_spec

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "registries/campaigns/eiel_campaign_v1.schema.json"
SPEC_ROOT = ROOT / "_spec/campaigns/v1"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _fixture(name: str) -> dict:
    value = load_campaign_spec(SPEC_ROOT / name)
    execution = value.get("execution", {})
    if "generations" in execution and "execution_route_identity" not in execution:
        # The schema requires new G/T documents to identify the compatibility
        # route explicitly. This adapter preserves validation coverage for the
        # pre-discriminator fixtures without making implicit G/T copyable.
        execution["execution_route_identity"] = "legacy_generation_route_v1"
    return value


def _validator():
    if importlib.util.find_spec("jsonschema") is None:
        pytest.skip("jsonschema is not installed")
    from jsonschema import Draft202012Validator

    schema = _schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_schema_is_versioned_strict_and_matches_top_level_surface() -> None:
    schema = _schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"]["const"] == "eiel_campaign_v1"
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {
        "schema_version",
        "campaign_id",
        "description",
        "no_claims",
        "world",
        "policy",
        "selection",
        "parameters",
        "matrix",
        "execution",
        "checkpoints",
        "comparisons",
        "artifacts",
        "analysis",
        "output",
        "artifact_lifecycle",
        "run_contract",
        "authority",
        "_legacy_lifecycle_derived",
    }


def test_schema_exposes_continuation_cells_without_making_it_required() -> None:
    comparisons = _schema()["$defs"]["comparisons"]

    assert comparisons["additionalProperties"] is False
    assert comparisons["properties"]["continuation_cells"] == {
        "type": "integer",
        "minimum": 0,
    }
    assert "required" not in comparisons


def test_schema_exposes_optional_reporting_parity() -> None:
    comparisons = _schema()["$defs"]["comparisons"]

    assert comparisons["properties"]["reporting_parity"] == {"type": "boolean"}
    assert "required" not in comparisons


def test_schema_exposes_optional_checkpoint_storage_modes() -> None:
    checkpoints = _schema()["$defs"]["checkpoints"]

    assert checkpoints["additionalProperties"] is False
    assert checkpoints["properties"]["storage_mode"] == {
        "enum": ["disk", "disk_and_memory"]
    }
    assert "required" not in checkpoints


def test_schema_exposes_bounded_optional_acs_slots() -> None:
    checkpoints = _schema()["$defs"]["checkpoints"]

    assert checkpoints["properties"]["acs_slots"] == {
        "type": "integer",
        "enum": [0, 1, 2, 3],
    }
    assert "required" not in checkpoints


def test_schema_exposes_optional_acs_capture_instrumentation() -> None:
    checkpoints = _schema()["$defs"]["checkpoints"]

    assert checkpoints["properties"]["acs_capture_instrumentation"] == {
        "type": "boolean"
    }
    assert "required" not in checkpoints


def test_schema_exposes_optional_typed_technical_gate_schedule() -> None:
    technical_gate = _schema()["$defs"]["execution"]["properties"]["technical_gate"]

    assert technical_gate["additionalProperties"] is False
    assert technical_gate["properties"]["generations"] == {
        "$ref": "#/$defs/positiveInteger"
    }
    assert technical_gate["properties"]["checkpoint_milestones"] == {
        "type": "array",
        "uniqueItems": True,
        "items": {"type": "integer", "minimum": 0},
    }
    assert "generations" not in technical_gate["required"]
    assert "checkpoint_milestones" not in technical_gate["required"]


@pytest.mark.parametrize(
    "name",
    [
        "patch_fixed_g14_irregular.yaml",
        "wdl_gru_g37_irregular.yaml",
        "patch_fixed_b72g100.yaml",
    ],
)
def test_current_campaign_fixtures_validate(name: str) -> None:
    _validator().validate(_fixture(name))


def test_continuation_cells_fixture_validates() -> None:
    fixture = _fixture("wdl_gru_g37_irregular.yaml")
    fixture["comparisons"]["continuation_cells"] = 1

    _validator().validate(fixture)


def test_declarative_technical_gate_schedule_fixture_validates() -> None:
    fixture = _fixture("patch_fixed_g14_irregular.yaml")
    fixture["execution"]["technical_gate"] = {
        "selectors": [
            {
                "ecology_seed": 101,
                "task_seed": 1001,
                "founder_mode": "balanced",
            }
        ],
        "partition": False,
        "generations": 10,
        "checkpoint_milestones": [0, 5, 10],
    }

    _validator().validate(fixture)


def test_canonical_batch_omits_inapplicable_nullable_fields() -> None:
    bounded = _fixture("patch_fixed_g14_irregular.yaml")
    assert bounded["execution"]["batch"] == {"mode": "bounded", "maximum": 2}

    explicit = copy.deepcopy(bounded)
    explicit["execution"]["batch"] = {"mode": "explicit", "cells": 2}
    _validator().validate(explicit)


@pytest.mark.parametrize(
    ("mutation", "expected_path"),
    [
        (lambda value: value.update({"unexpected": True}), "unexpected"),
        (
            lambda value: value["comparisons"].update({"continuation_cells": -1}),
            "continuation_cells",
        ),
        (
            lambda value: value["execution"]["batch"].update({"cells": 3}),
            "batch",
        ),
        (
            lambda value: value["checkpoints"].update({"storage_mode": "memory"}),
            "storage_mode",
        ),
    ],
)
def test_schema_rejects_invalid_fixture_shapes(mutation, expected_path: str) -> None:
    fixture = copy.deepcopy(_fixture("patch_fixed_g14_irregular.yaml"))
    mutation(fixture)

    errors = list(_validator().iter_errors(fixture))
    assert errors
    assert expected_path in " ".join(
        [*map(str, errors[0].absolute_path), errors[0].message]
    )
