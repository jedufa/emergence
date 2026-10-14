from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from emergence_contracts.canonical import canonical_json_bytes, canonical_sha256
from emergence_contracts.errors import ContractError
from emergence_contracts.science_request_v1 import (
    BATCH_POLICY_AT_MOST,
    BATCH_POLICY_EXACT,
    EielScienceRequestV1,
    expand_eiel_science_request_v1,
    validate_eiel_science_request_v1,
)
from tests.test_eiel_science_request_v1_compiler import _selected_world_request
from engine.campaigns.artifact_lifecycle import (
    lifecycle_profile_contract_v1,
    validate_run_artifact_lifecycle_contract_v1,
)


ROOT = Path(__file__).parents[1]
EXAMPLE_ROOT = ROOT / "_spec/campaigns/v1/eiel_science_request_v1/examples"
SCHEMA_PATH = ROOT / "_spec/science_instrument/v1/eiel_science_request_v1.schema.json"
SELECTION_NAME = "EIEL_SCIENCE_REQUEST_V1_selection_window_w32_b1024_example.json"
MUTATION_NAME = (
    "EIEL_SCIENCE_REQUEST_V1_mutation_pressure_w128x80_multibatch_example.json"
)
TEMPLATE_NAME = "EIEL_SCIENCE_REQUEST_V1_template_w32.json"
G5_1024_NAME = (
    "EIEL_SCIENCE_REQUEST_V1_expedition_g5_engineering_1024_w32_b64_v1.json"
)
G5_10000_NAME = (
    "EIEL_SCIENCE_REQUEST_V1_expedition_g5_engineering_10000_w32_b64_v1.json"
)
EXPECTED_MEMBERS = {
    SELECTION_NAME: {
        "sha256": "22c0859df66c4af6930c7e11e2cf17510e0e219f3fe30a8ccafd72699b98997f",
        "size_bytes": 4363,
    },
    MUTATION_NAME: {
        "sha256": "3037d36bfab72cfca09de08b3aa00a7e34207b8b00e74f2d4dd2ac594aab683a",
        "size_bytes": 4152,
    },
}


def _load(name: str) -> dict[str, object]:
    value = json.loads((EXAMPLE_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _placeholder(name: str) -> str:
    return "$" + "{" + name + "}"


def _small_selection() -> dict[str, object]:
    value = _load(SELECTION_NAME)
    value["dimensions"] = {
        "pressure_onset": [0, 2500],
        "pressure_intensity": [0.0, 0.5],
        "mutation_enabled": [False],
        "mutation_step": [1],
        "seed": [1103, 2207],
    }
    value["run"]["batch"] = {"policy": "exact", "worlds": 8}
    value["expect_worlds"] = 8
    return value


def test_examples_are_schema_valid_and_archive_exact() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    provenance = json.loads(
        (EXAMPLE_ROOT / "archive_provenance_v1.json").read_text(encoding="utf-8")
    )

    assert provenance["source_archive"] == {
        "basename": "EIEL_SCIENCE_REQUEST_V1_campaign_examples.zip",
        "sha256": "4e102dde3a7617417f8f40accb8928f99dc9be4e3b10787d5d7db9042dfd0495",
    }
    by_name = {row["member_name"]: row for row in provenance["members"]}
    assert set(by_name) == set(EXPECTED_MEMBERS)
    for name, expected in EXPECTED_MEMBERS.items():
        content = (EXAMPLE_ROOT / name).read_bytes()
        assert len(content) == expected["size_bytes"]
        assert hashlib.sha256(content).hexdigest() == expected["sha256"]
        assert by_name[name]["size_bytes"] == expected["size_bytes"]
        assert by_name[name]["sha256"] == expected["sha256"]
        validator.validate(json.loads(content))


def test_minimal_template_is_schema_valid_and_uses_at_most_ceiling() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    value = _load(TEMPLATE_NAME)
    validator.validate(value)
    contract = validate_eiel_science_request_v1(value)
    plan = contract.batch_plan()
    assert contract.expect_worlds == 1
    assert plan.policy == BATCH_POLICY_AT_MOST
    assert plan.worlds == 1024
    assert plan.batch_sizes == (1,)


@pytest.mark.parametrize(
    ("name", "dimension_width", "logical_worlds", "batch_sizes"),
    [
        (G5_1024_NAME, 32, 1024, (64,) * 16),
        (G5_10000_NAME, 100, 10000, (64,) * 156 + (16,)),
    ],
)
def test_g5_engineering_fixtures_bind_compact_dimensions_and_lifecycle(
    name: str,
    dimension_width: int,
    logical_worlds: int,
    batch_sizes: tuple[int, ...],
) -> None:
    raw = _load(name)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(raw)

    contract = validate_eiel_science_request_v1(raw)
    plan = contract.batch_plan()
    assert raw["question"]["no_claims"] is True
    assert raw["run"] == {
        "ticks": 64,
        "batch": 64,
        "checkpoint_ticks": [64],
    }
    assert raw["observe"]["snapshot_ticks"] == [0, 64]
    assert raw["dimensions"] == {
        "condition": raw["dimensions"]["condition"],
        "control_enabled": [True],
        "replicate": raw["dimensions"]["replicate"],
    }
    assert len(raw["dimensions"]["condition"]) == dimension_width
    assert len(raw["dimensions"]["replicate"]) == dimension_width
    assert raw["world_template"]["controls"] == {
        "mutation.enabled": "${control_enabled}"
    }
    assert raw["world_template"]["seeds"] == {
        "ecology": "${replicate}",
        "founder": "${replicate}",
        "initialization": "${replicate}",
        "intervention": "${replicate}",
        "mutation": "${replicate}",
        "policy": "${condition}",
        "task_environment": "${replicate}",
    }
    assert contract.expect_worlds == logical_worlds
    assert plan.policy == BATCH_POLICY_AT_MOST
    assert plan.worlds == 64
    assert plan.batch_sizes == batch_sizes

    lifecycle = lifecycle_profile_contract_v1(
        "SCIENCE_COMPACT",
        total_max_bytes=64 * 1024**2,
        checkpoint_schedule=tuple(raw["run"]["checkpoint_ticks"]),
        checkpoint_authority="absolute_tick",
    )
    assert validate_run_artifact_lifecycle_contract_v1(lifecycle) == lifecycle
    assert lifecycle["observability"]["output_profile"] == "compact_default"
    assert lifecycle["observability"]["raw_tick_trace_policy"] == "forbidden"
    assert lifecycle["checkpoints"] == {
        "authority": "absolute_tick",
        "checkpoint_ticks": [64],
        "roles": [
            {
                "identity": "checkpoint-tick-000000000064",
                "checkpoint_tick": 64,
                "role": "terminal",
                "retention_class": "R4",
            }
        ],
    }
    assert lifecycle["retention"]["qualification_hold_binding"] == "not_required"
    assert b"generation" not in canonical_json_bytes(raw).lower()
    assert b"phenotype" not in canonical_json_bytes(raw).lower()
    assert b"generation" not in canonical_json_bytes(lifecycle).lower()


def test_selected_world_continuation_is_schema_bound_and_absolute_tick_exact() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    raw = _selected_world_request()
    validator.validate(raw)
    request = validate_eiel_science_request_v1(raw)

    assert request.batch_plan().batch_sizes == (1,)
    assert request.expand_worlds()[0].world_template.to_value() == {
        "controls": {
            "environment.resource.multiplier": 1.5,
            "mutation.rate": 0.25,
        }
    }

    early = deepcopy(raw)
    early["world_template"]["controls"]["mutation.rate"] = {
        "value": 0.25,
        "schedule": {
            "kind": "step_absolute_tick",
            "onset_tick": 24,
            "baseline": 0.125,
        },
    }
    with pytest.raises(ContractError, match=r"\[25, 30\]"):
        validate_eiel_science_request_v1(early)

    multiple = deepcopy(raw)
    multiple["dimensions"]["replicate"] = [0, 1]
    multiple["expect_worlds"] = 2
    multiple["run"]["batch"]["worlds"] = 2
    with pytest.raises(ContractError, match="one unexcluded singleton world"):
        validate_eiel_science_request_v1(multiple)


def test_selection_expands_last_dimension_fastest_with_exact_types() -> None:
    contract = validate_eiel_science_request_v1(_load(SELECTION_NAME))
    worlds = contract.expand_worlds()

    assert contract.analysis_dimensions == (
        "pressure_onset",
        "pressure_intensity",
        "mutation_enabled",
        "mutation_step",
        "seed",
    )
    assert len(worlds) == contract.expect_worlds == 1024
    assert worlds[0].dimension_values.to_value() == {
        "pressure_onset": 0,
        "pressure_intensity": 0.0,
        "mutation_enabled": False,
        "mutation_step": 1,
        "seed": 1103,
    }
    assert worlds[1].dimension_values.to_value()["seed"] == 2207
    assert worlds[7].dimension_values.to_value()["seed"] == 8837
    assert worlds[8].dimension_values.to_value()["mutation_step"] == 4

    template = worlds[0].world_template.to_value()
    controls = template["controls"]
    assert controls["mutation.enabled"] is False
    assert type(controls["mutation.step"]) is int
    assert type(controls["selection.pressure"]["value"]) is float
    assert type(controls["selection.pressure"]["schedule"]["onset_tick"]) is int
    assert template["seeds"]["ecology"] == 1103

    plan = contract.batch_plan()
    assert plan.policy == BATCH_POLICY_EXACT
    assert plan.worlds == 1024
    assert plan.batch_sizes == (1024,)


def test_mutation_partial_exclusion_and_at_most_batches_are_exact() -> None:
    contract = EielScienceRequestV1.from_dict(_load(MUTATION_NAME))
    worlds = expand_eiel_science_request_v1(contract)

    assert len(worlds) == contract.expect_worlds == 240
    assert worlds[0].dimension_values.to_value() == {
        "selection_pressure": 0.0,
        "mutation_rate": 0.0078125,
        "founder_profile": "native-neutral-v1",
        "seed": 1103,
    }
    assert all(
        not (
            row.dimension_values.to_value()["selection_pressure"] == 0.0
            and row.dimension_values.to_value()["mutation_rate"] == 0.0
        )
        for row in worlds
    )
    assert {row.world_ordinal for row in worlds} == set(range(240))
    plan = contract.batch_plan()
    assert plan.policy == BATCH_POLICY_AT_MOST
    assert plan.worlds == 32
    assert plan.batch_sizes == (32, 32, 32, 32, 32, 32, 32, 16)


def test_at_most_batch_ceiling_can_exceed_small_experiment() -> None:
    raw = _small_selection()
    raw["run"]["batch"] = 1024
    raw["expect_worlds"] = 8
    contract = validate_eiel_science_request_v1(raw)
    plan = contract.batch_plan()
    assert plan.policy == BATCH_POLICY_AT_MOST
    assert plan.worlds == 1024
    assert plan.batch_sizes == (8,)


def test_analysis_dimensions_not_object_order_control_expansion() -> None:
    original = _small_selection()
    reordered = deepcopy(original)
    reordered["dimensions"] = dict(reversed(list(reordered["dimensions"].items())))

    first = validate_eiel_science_request_v1(original)
    second = validate_eiel_science_request_v1(reordered)
    assert [row.to_dict() for row in first.expand_worlds()] == [
        row.to_dict() for row in second.expand_worlds()
    ]
    assert first.request_sha256() == second.request_sha256()

    changed_order = deepcopy(original)
    changed_order["reduce"]["configuration"]["analysis_dimensions"] = [
        "pressure_intensity",
        "pressure_onset",
        "mutation_enabled",
        "mutation_step",
        "seed",
    ]
    changed = validate_eiel_science_request_v1(changed_order)
    assert (
        changed.expand_worlds()[2].dimension_values.to_value()["pressure_onset"] == 2500
    )
    assert (
        first.expand_worlds()[2].dimension_values.to_value()["pressure_intensity"]
        == 0.5
    )
    assert changed.request_sha256() != first.request_sha256()


def test_canonical_digest_is_stable_and_input_is_defensively_frozen() -> None:
    raw = _small_selection()
    contract = validate_eiel_science_request_v1(raw)
    expected = canonical_sha256(raw)
    raw["question"]["description"] = "mutated after validation"

    assert contract.request_sha256() == expected
    assert (
        contract.to_dict()["question"]["description"] != raw["question"]["description"]
    )
    assert validate_eiel_science_request_v1(contract) is contract


def test_exclusions_require_exact_json_types() -> None:
    raw = _load(MUTATION_NAME)
    raw["exclude"][0]["selection_pressure"] = 0
    with pytest.raises(ContractError, match="exact typed dimension member"):
        validate_eiel_science_request_v1(raw)


@pytest.mark.parametrize(
    "replacement, message",
    (
        ("prefix-" + _placeholder("seed"), "whole string"),
        (_placeholder("unknown_dimension"), "unknown dimension"),
    ),
)
def test_placeholders_are_whole_string_and_known(
    replacement: str, message: str
) -> None:
    raw = _small_selection()
    raw["world_template"]["seeds"]["ecology"] = replacement
    with pytest.raises(ContractError, match=message):
        validate_eiel_science_request_v1(raw)


def test_every_dimension_must_drive_the_template() -> None:
    raw = _small_selection()
    raw["world_template"]["controls"]["mutation.step"] = 1
    with pytest.raises(ContractError, match="every dimension"):
        validate_eiel_science_request_v1(raw)


def test_expect_worlds_is_recomputed_after_exclusions() -> None:
    raw = _load(MUTATION_NAME)
    raw["expect_worlds"] = 256
    with pytest.raises(ContractError, match="expanded world count 240"):
        validate_eiel_science_request_v1(raw)


def test_exact_batch_rejects_a_partial_tail() -> None:
    raw = _load(MUTATION_NAME)
    raw["run"]["batch"] = {"policy": "exact", "worlds": 32}
    with pytest.raises(ContractError, match="without a remainder"):
        validate_eiel_science_request_v1(raw)


def test_batch_object_cannot_spell_at_most_policy() -> None:
    raw = _small_selection()
    raw["run"]["batch"] = {"policy": "at_most", "worlds": 8}
    with pytest.raises(ContractError, match="only the exact policy"):
        validate_eiel_science_request_v1(raw)


@pytest.mark.parametrize(
    "mutate, message",
    (
        (
            lambda value: value["run"].update(checkpoint_ticks=[50000, 25000, 100000]),
            "sorted and unique",
        ),
        (
            lambda value: value["run"].update(checkpoint_ticks=[25000, 50000]),
            "end at run.ticks",
        ),
        (
            lambda value: value["observe"].update(snapshot_ticks=[100, 100000]),
            "start at absolute tick 0",
        ),
        (
            lambda value: value["dimensions"].update(pressure_onset=[0, 100001]),
            "integer must be in",
        ),
    ),
)
def test_absolute_tick_contract_fails_closed(mutate, message: str) -> None:
    raw = _load(SELECTION_NAME)
    mutate(raw)
    with pytest.raises(ContractError, match=message):
        validate_eiel_science_request_v1(raw)


@pytest.mark.parametrize(
    "mutate, message",
    (
        (
            lambda value: value["reduce"]["configuration"].update(
                analysis_dimensions=[
                    "pressure_onset",
                    "pressure_intensity",
                    "mutation_enabled",
                    "mutation_step",
                ]
            ),
            "exact permutation",
        ),
        (
            lambda value: value["reduce"]["configuration"].update(
                axis_b="missing_axis"
            ),
            "axes must name dimensions",
        ),
        (
            lambda value: value.update(replicate_dimension="pressure_onset"),
            "distinct from reducer axes",
        ),
        (
            lambda value: value.update(replicate_dimension="missing"),
            "must name a dimension",
        ),
    ),
)
def test_reducer_replicate_and_dimension_cross_checks(mutate, message: str) -> None:
    raw = _small_selection()
    mutate(raw)
    with pytest.raises(ContractError, match=message):
        validate_eiel_science_request_v1(raw)


@pytest.mark.parametrize(
    "field",
    ("generation", "ticks_per_generation", "physical_window_generations"),
)
def test_generation_fields_are_forbidden_recursively(field: str) -> None:
    raw = _small_selection()
    raw["world_template"]["controls"][field] = 1
    with pytest.raises(ContractError, match="generation-shaped fields"):
        validate_eiel_science_request_v1(raw)


def test_unknown_fields_and_non_json_template_values_fail_closed() -> None:
    raw = _small_selection()
    raw["executable_hook"] = "module:function"
    with pytest.raises(ContractError, match="strict contract"):
        validate_eiel_science_request_v1(raw)

    raw = _small_selection()
    raw["world_template"]["controls"]["invalid"] = None
    with pytest.raises(ContractError, match="cannot contain null"):
        validate_eiel_science_request_v1(raw)


@pytest.mark.parametrize(
    "field",
    (
        "require_direct_genotype_reproductive_attribution",
        "require_reproduction_funnel_denominators",
    ),
)
def test_reproductive_attribution_requirements_are_exact_booleans(field: str) -> None:
    raw = _small_selection()
    raw["reduce"]["configuration"][field] = "true"
    with pytest.raises(ContractError, match="exact boolean"):
        validate_eiel_science_request_v1(raw)
