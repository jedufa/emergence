from __future__ import annotations

from copy import deepcopy

import pytest

from agents.modular_genome_v1 import canonical_hash_from_genome_dict
from evolution.phase_f_v2.matrix import CASE_COUNT, build_phase_f_v2_matrix
from evolution.structural_development_v1 import DEVELOPMENTAL_EXPRESSION_BLOCK_KEY
from evolution.structural_gated_recurrent_v1 import GATED_OPERATOR, GATED_RECORD_KEY
from evolution.structural_observation_routing_v1 import ROUTING_BLOCK_KEY
from evolution.structural_output_routing_v1 import OUTPUT_ROUTING_BLOCK_KEY
from evolution.structural_sustainable_growth_v1 import ACTIVITY_RECORD_KEY
from perf.cuda_architecture_b_modular_state_abi import pack_modular_storage
from perf.eiel_phase_f_forced_combination_fixture_v2 import (
    MODULAR_RNG_DRAW_COUNT,
    PANEL_AGENT_COUNT,
    PhaseFV2FixtureError,
    build_phase_f_fixture_registry_v2,
    materialize_phase_f_case_v2,
    phase_f_fixture_sha256_v2,
    validate_phase_f_fixture_case_v2,
    validate_phase_f_fixture_registry_v2,
)


@pytest.fixture(scope="module")
def matrix() -> dict:
    return build_phase_f_v2_matrix()


@pytest.fixture(scope="module")
def registry(matrix: dict) -> dict:
    return build_phase_f_fixture_registry_v2(matrix)


def _digest_without(value: dict, field: str) -> str:
    payload = deepcopy(value)
    payload.pop(field)
    return phase_f_fixture_sha256_v2(payload)


def _modules(agent: dict) -> list[dict]:
    return list(agent["record"]["genome"]["modules"])


def _params(agent: dict, index: int) -> dict:
    return dict(_modules(agent)[index]["params"])


def _operators(agent: dict) -> list[str]:
    return [
        "G" if module["params"].get("live_operator") == GATED_OPERATOR else "I"
        for module in _modules(agent)
    ]


def _dormant(agent: dict, index: int) -> bool:
    return bool(_params(agent, index).get(ACTIVITY_RECORD_KEY, {}).get("dormant"))


def _bindings(agent: dict, index: int, key: str) -> list[tuple[str, float]]:
    return [
        (str(row["semantic_id"]), float(row["gain"]))
        for row in _params(agent, index).get(key, {}).get("bindings", [])
    ]


def _development(agent: dict, index: int) -> dict:
    return dict(_params(agent, index).get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY, {}))


def _runtime(agent: dict, field: str, index: int) -> float | bool | None:
    instance_id = str(_modules(agent)[index]["module_instance_id"])
    return agent["record"][field].get(instance_id)


def _connections(agent: dict) -> list[tuple[str, str, float]]:
    return [
        (
            str(row["src_module_id"]),
            str(row["dst_module_id"]),
            float(row["weight"]),
        )
        for row in agent["record"]["genome"]["connections"]
    ]


def _assert_f01(variant: int, agent: dict) -> None:
    modules = _modules(agent)
    if variant in {0, 1}:
        assert modules == []
    elif variant == 2:
        assert _operators(agent) == ["I"]
        assert _bindings(agent, 0, OUTPUT_ROUTING_BLOCK_KEY) == []
        assert _params(agent, 0)["live_output_gain"] == 0.0
        assert _runtime(agent, "structural_live_module_state", 0) == 0.0
        assert _runtime(agent, "structural_live_module_outputs", 0) == 0.0
    elif variant == 3:
        assert _operators(agent) == ["G"]
        gated = _params(agent, 0)[GATED_RECORD_KEY]
        assert all(
            float(value) == 0.0
            for key, value in gated.items()
            if key != "schema_version"
        )
        assert _runtime(agent, "structural_live_module_outputs", 0) == 0.0
    elif variant == 4:
        assert _bindings(agent, 0, ROUTING_BLOCK_KEY) == [("self_energy", 0.0)]
        assert _bindings(agent, 0, OUTPUT_ROUTING_BLOCK_KEY) == [("move_north", 0.5)]
    elif variant == 5:
        assert _bindings(agent, 0, ROUTING_BLOCK_KEY) == [("self_energy", 0.5)]
        assert _bindings(agent, 0, OUTPUT_ROUTING_BLOCK_KEY) == [("move_north", 0.0)]
        assert _development(agent, 0)["stage_gains"] == [0.0, 0.0, 0.0]
        assert (
            _runtime(agent, "structural_development_effective_gain_by_instance_id", 0)
            == 0.0
        )
    elif variant == 6:
        assert _dormant(agent, 0)
        assert _runtime(agent, "structural_live_module_state", 0) != 0.0
        assert _runtime(agent, "structural_live_module_outputs", 0) != 0.0
    else:
        assert len(modules) == 2 and _operators(agent) == ["I", "I"]
        assert modules[0]["params"] == modules[1]["params"]
        assert modules[1]["ancestry"]["origin_operator"] == "duplicate_with_decay"
        assert modules[1]["ancestry"]["parent_module_instance_ids"] == [
            modules[0]["module_instance_id"]
        ]
        assert all(
            _runtime(agent, "structural_live_module_outputs", index) == 0.0
            for index in range(2)
        )


def _assert_f02(variant: int, agent: dict) -> None:
    expected = {
        0: ("self_energy", 0.5, 1),
        1: ("self_energy", -0.5, -1),
        2: ("self_health", 0.5, -1),
        3: ("self_health", -0.5, 1),
    }
    if variant < 4:
        semantic, gain, state_sign = expected[variant]
        assert _bindings(agent, 0, ROUTING_BLOCK_KEY) == [(semantic, gain)]
        state = float(_runtime(agent, "structural_live_module_state", 0))
        assert (state > 0) is (state_sign > 0)
    elif variant in {4, 5}:
        assert len(_modules(agent)) == 1
        assert _connections(agent) == [("m000", "m000", 0.5 if variant == 4 else -0.5)]
        previous = float(_runtime(agent, "structural_live_module_outputs", 0))
        assert (previous > 0) is (variant == 4)
    elif variant == 6:
        assert _development(agent, 0)["stage_gains"] == [0.5, 0.75, 1.0]
        assert (
            _runtime(agent, "structural_development_effective_gain_by_instance_id", 0)
            == 0.5
        )
        assert _bindings(agent, 0, OUTPUT_ROUTING_BLOCK_KEY)[0][0].startswith("move_")
    else:
        assert _bindings(agent, 0, ROUTING_BLOCK_KEY) == [("self_health", -0.5)]
        assert _bindings(agent, 0, OUTPUT_ROUTING_BLOCK_KEY) == [("rest", 0.5)]
        assert float(_runtime(agent, "structural_live_module_state", 0)) < 0


def _assert_f03(variant: int, agent: dict) -> None:
    gated_index = 0 if variant < 6 else 1
    gated = _params(agent, gated_index)[GATED_RECORD_KEY]
    if variant == 0:
        assert gated["update_bias"] <= -4.0
    elif variant == 1:
        assert gated["update_bias"] >= 4.0
    elif variant == 2:
        assert gated["reset_bias"] <= -4.0
        assert gated["update_bias"] >= 4.0
    elif variant == 3:
        assert gated["candidate_bias"] > 0.0
    elif variant == 4:
        assert gated["candidate_bias"] < 0.0
    elif variant == 5:
        assert _bindings(agent, 0, ROUTING_BLOCK_KEY)[0][0] == "self_energy"
    elif variant == 6:
        assert _operators(agent) == ["I", "G"]
        assert _bindings(agent, 1, ROUTING_BLOCK_KEY)[0][0] == "self_health"
        assert _development(agent, 1)["prerequisite_homology_ids"] == [
            _modules(agent)[0]["homology_id"]
        ]
        assert (
            _runtime(agent, "structural_development_effective_gain_by_instance_id", 1)
            == 1.0
        )
    else:
        assert _operators(agent) == ["I", "G"]
        assert _dormant(agent, 0)
        assert _connections(agent) == [("m000", "m001", -0.5)]
        assert _bindings(agent, 1, OUTPUT_ROUTING_BLOCK_KEY) == [("rest", 0.5)]
        assert (
            _runtime(agent, "structural_development_effective_gain_by_instance_id", 1)
            == 0.0
        )


def _assert_f04(variant: int, agent: dict) -> None:
    assert len(_modules(agent)) == 2 and len(_connections(agent)) == 1
    first = _bindings(agent, 0, OUTPUT_ROUTING_BLOCK_KEY)
    second = _bindings(agent, 1, OUTPUT_ROUTING_BLOCK_KEY)
    weight = _connections(agent)[0][2]
    if variant == 0:
        assert _operators(agent) == ["I", "I"] and first and not second
    elif variant == 1:
        assert _operators(agent) == ["I", "G"] and not first and second
        assert weight > 0
    elif variant == 2:
        assert _operators(agent) == ["G", "I"]
        assert first[0][0] == second[0][0]
    elif variant == 3:
        assert _operators(agent) == ["G", "G"]
        assert first[0][0] != second[0][0] and weight > 0
    elif variant == 4:
        assert _operators(agent) == ["G", "G"] and weight < 0
    elif variant == 5:
        assert first[0][0] == second[0][0]
        assert first[0][1] == -second[0][1]
    elif variant == 6:
        assert _dormant(agent, 0)
    else:
        assert _development(agent, 1)["stage_gains"] == [0.0, 0.0, 0.0]
        assert (
            _runtime(agent, "structural_development_effective_gain_by_instance_id", 1)
            == 0.0
        )


def _assert_f05(variant: int, agent: dict) -> None:
    assert len(_modules(agent)) == 3
    edges = {(source, target) for source, target, _ in _connections(agent)}
    if variant == 0:
        assert edges == {("m000", "m001"), ("m001", "m002")}
    elif variant == 1:
        assert edges == {("m000", "m002"), ("m001", "m002")}
    elif variant == 2:
        assert edges == {("m000", "m001"), ("m000", "m002")}
    elif variant == 3:
        assert set(_development(agent, 2)["prerequisite_homology_ids"]) == {
            _modules(agent)[0]["homology_id"],
            _modules(agent)[1]["homology_id"],
        }
    elif variant == 4:
        assert _dormant(agent, 0)
    elif variant == 5:
        assert _development(agent, 1)["stage_gains"] == [0.0, 0.0, 0.0]
        assert (
            _runtime(agent, "structural_development_effective_gain_by_instance_id", 1)
            == 0.0
        )
    elif variant == 6:
        assert set(_operators(agent)) == {"I", "G"}
        semantics = {
            binding[0]
            for index in range(3)
            for binding in _bindings(agent, index, ROUTING_BLOCK_KEY)
        }
        assert semantics == {"self_energy", "self_health"}
    else:
        outputs = {
            binding[0]
            for index in range(3)
            for binding in _bindings(agent, index, OUTPUT_ROUTING_BLOCK_KEY)
        }
        assert "rest" in outputs and any(value.startswith("move_") for value in outputs)


def _assert_f06(variant: int, agent: dict) -> None:
    weights = [weight for _, _, weight in _connections(agent)]
    if variant == 0:
        assert weights == [0.5, 0.5]
    elif variant == 1:
        assert weights == [-0.5, -0.5]
    elif variant == 2:
        assert weights == [0.5, -0.5]
    elif variant == 3:
        assert weights == [0.5, 0.5, 0.5]
    elif variant == 4:
        assert weights == [-0.5, -0.5, -0.5]
    elif variant == 5:
        assert weights == [0.5, 0.5, -1.0] and sum(weights) == 0.0
    elif variant == 6:
        assert _dormant(agent, 1)
    else:
        previous = [
            float(_runtime(agent, "structural_live_module_outputs", index))
            for index in range(3)
        ]
        assert any(value > 0 for value in previous)
        assert any(value < 0 for value in previous)


def _assert_f07(variant: int, agent: dict) -> None:
    assert len(_modules(agent)) == 2
    root_gain = _runtime(
        agent, "structural_development_effective_gain_by_instance_id", 0
    )
    leaf_gain = _runtime(
        agent, "structural_development_effective_gain_by_instance_id", 1
    )
    if variant == 0:
        assert _development(agent, 1)["stage_gains"] == [1.0, 0.0, 0.0]
        assert leaf_gain == 1.0
    elif variant == 1:
        assert _development(agent, 1)["stage_gains"] == [0.0, 1.0, 1.0]
        assert leaf_gain == 1.0
    elif variant == 2:
        assert _development(agent, 1)["stage_gains"] == [0.0, 0.0, 1.0]
        assert leaf_gain == 0.0
        assert _runtime(agent, "structural_live_module_state", 1) == pytest.approx(0.4)
    elif variant == 3:
        assert _dormant(agent, 0) and leaf_gain == 0.0
    elif variant == 4:
        assert not _dormant(agent, 0) and root_gain == leaf_gain == 1.0
    elif variant == 5:
        assert _development(agent, 1)["latch_dependency_readiness"] is True
        assert _runtime(agent, "structural_development_latch_by_instance_id", 1) is True
    elif variant == 6:
        assert _dormant(agent, 1)
        assert _runtime(agent, "structural_development_latch_by_instance_id", 1) is True
    else:
        assert _development(agent, 0)["stage_gains"] == [0.0, 0.0, 0.0]
        assert root_gain == 0.0
        assert _development(agent, 1)["stage_gains"] == [0.5, 0.5, 0.5]
        assert leaf_gain == 0.5 and not _dormant(agent, 1)


def _assert_f08(variant: int, agent: dict) -> None:
    if variant == 0:
        assert _operators(agent) == ["I"] and _dormant(agent, 0)
        assert _runtime(agent, "structural_live_module_state", 0) != 0.0
    elif variant == 1:
        assert _operators(agent) == ["G"] and _dormant(agent, 0)
    elif variant == 2:
        assert _dormant(agent, 0)
        assert _runtime(agent, "structural_live_module_outputs", 0) == pytest.approx(
            0.35
        )
    elif variant == 3:
        assert _dormant(agent, 0)
        assert _bindings(agent, 0, ROUTING_BLOCK_KEY)
        assert _bindings(agent, 0, OUTPUT_ROUTING_BLOCK_KEY)
    elif variant == 4:
        assert _dormant(agent, 0) and not _dormant(agent, 1)
        assert _connections(agent)
    elif variant == 5:
        assert not _dormant(agent, 0) and _dormant(agent, 1)
        assert _connections(agent)
    elif variant == 6:
        assert _operators(agent) == ["I"] and not _dormant(agent, 0)
    else:
        assert _operators(agent) == ["G"] and not _dormant(agent, 0)


def _assert_f09(variant: int, agent: dict) -> None:
    recipe = agent["prepared_boundary_recipe"]
    assert recipe["prepared_only"] is True
    assert recipe["lifecycle_executed"] is False
    if variant == 0:
        assert len(_modules(agent)) > 1 and _dormant(agent, 1)
    elif variant == 1:
        gated = _params(agent, 1)[GATED_RECORD_KEY]
        assert gated["output_gain"] == 0.0 and not _dormant(agent, 1)
    elif variant == 2:
        assert _development(agent, 2)["prerequisite_homology_ids"] == [
            _modules(agent)[1]["homology_id"]
        ]
    elif variant == 3:
        assert len(_modules(agent)) == 1
    elif variant == 4:
        assert _bindings(agent, 1, ROUTING_BLOCK_KEY)
        assert _bindings(agent, 1, OUTPUT_ROUTING_BLOCK_KEY)
    elif variant == 5:
        assert len(_connections(agent)) == 2
    elif variant == 6:
        assert [module["module_id"] for module in _modules(agent)] == [
            "m000",
            "m001",
            "m002",
            "m003",
        ]
    else:
        modules = _modules(agent)
        assert modules[2]["ancestry"]["origin_operator"] == "duplicate_with_decay"
        assert modules[2]["ancestry"]["parent_module_instance_ids"] == [
            modules[1]["module_instance_id"]
        ]


def _assert_f10(variant: int, agent: dict) -> None:
    if variant == 0:
        assert _operators(agent) == ["I"]
    elif variant == 1:
        assert _operators(agent) == ["G"]
    elif variant == 2:
        assert len(_modules(agent)) == 3
    elif variant == 3:
        assert _connections(agent) and all(
            weight == 0.0 for _, _, weight in _connections(agent)
        )
    elif variant == 4:
        assert _bindings(agent, 0, ROUTING_BLOCK_KEY)
        assert _bindings(agent, len(_modules(agent)) - 1, OUTPUT_ROUTING_BLOCK_KEY)
    elif variant == 5:
        assert (
            len(
                _development(agent, len(_modules(agent)) - 1)[
                    "prerequisite_homology_ids"
                ]
            )
            == 1
        )
    elif variant == 6:
        assert _dormant(agent, len(_modules(agent)) - 1)
        assert sum(_dormant(agent, index) for index in range(len(_modules(agent)))) == 1
    else:
        assert len(_modules(agent)) == 8
    assert all(
        not _bindings(agent, index, OUTPUT_ROUTING_BLOCK_KEY)
        or all(
            gain == 0.0 for _, gain in _bindings(agent, index, OUTPUT_ROUTING_BLOCK_KEY)
        )
        for index in range(len(_modules(agent)))
    )


def _assert_f11(variant: int, agent: dict) -> None:
    recipe = agent["prepared_boundary_recipe"]
    assert recipe["prepared_only"] is True
    assert recipe["migration_executed"] is False
    if variant in {0, 1}:
        assert _operators(agent) == ["I", "G"]
        assert _bindings(agent, 0, ROUTING_BLOCK_KEY)[0][0] == "self_energy"
        assert _bindings(agent, 1, ROUTING_BLOCK_KEY)[0][0] == "self_health"
        assert _connections(agent) == [("m000", "m001", 0.5)]
    elif variant in {2, 3}:
        assert len(_modules(agent)) == 2
    elif variant == 4:
        assert len(_modules(agent)) < 12
    elif variant == 5:
        assert len(_modules(agent)) == 3 and len(_connections(agent)) == 2
        assert len(_bindings(agent, 2, OUTPUT_ROUTING_BLOCK_KEY)) == 2
    elif variant == 6:
        assert _dormant(agent, 0)
        assert _development(agent, 1)["prerequisite_homology_ids"] == [
            _modules(agent)[0]["homology_id"]
        ]
        assert (
            _runtime(agent, "structural_development_effective_gain_by_instance_id", 1)
            == 0.5
        )
    else:
        assert len(_modules(agent)) == 2 and _connections(agent)


def _assert_f12(variant: int, agent: dict) -> None:
    recipe = agent["prepared_boundary_recipe"]
    assert recipe["prepared_only"] is True
    assert recipe["checkpoint_authority_claimed"] is False
    if variant == 0:
        assert _modules(agent) == []
    elif variant == 1:
        assert _operators(agent) == ["G"]
    elif variant == 2:
        assert len(_modules(agent)) == 2 and len(_connections(agent)) == 1
    elif variant == 3:
        assert len(_development(agent, 1)["prerequisite_homology_ids"]) == 1
    elif variant == 4:
        assert _dormant(agent, 0)
    elif variant == 5:
        assert len(_modules(agent)) == 4
        assert set(_operators(agent)) == {"I", "G"}
        assert all(
            not _bindings(agent, index, OUTPUT_ROUTING_BLOCK_KEY) for index in range(4)
        )
    elif variant == 6:
        assert _dormant(agent, 1)
        assert _modules(agent)[2]["ancestry"]["origin_operator"] == (
            "duplicate_with_decay"
        )
    else:
        assert _operators(agent) == ["I", "G"] and _connections(agent)


_FAMILY_SEMANTIC_ASSERTIONS = {
    "f01": _assert_f01,
    "f02": _assert_f02,
    "f03": _assert_f03,
    "f04": _assert_f04,
    "f05": _assert_f05,
    "f06": _assert_f06,
    "f07": _assert_f07,
    "f08": _assert_f08,
    "f09": _assert_f09,
    "f10": _assert_f10,
    "f11": _assert_f11,
    "f12": _assert_f12,
}


def test_registry_materializes_all_120_cases_in_matrix_order(
    matrix: dict, registry: dict
) -> None:
    assert registry["case_count"] == CASE_COUNT == 120
    assert [case["case_id"] for case in registry["cases"]] == [
        case["case_id"] for case in matrix["cases"]
    ]
    assert registry["matrix_sha256"] == matrix["matrix_sha256"]
    assert registry["registry_sha256"] == _digest_without(registry, "registry_sha256")


def test_case_and_agent_digests_are_deterministic_and_bound_to_matrix(
    matrix: dict, registry: dict
) -> None:
    selected = registry["cases"][37]
    repeated = materialize_phase_f_case_v2(selected["case_id"], matrix=matrix)
    assert repeated == selected
    assert selected["case_sha256"] == matrix["cases"][37]["case_sha256"]
    assert selected["fixture_sha256"] == _digest_without(selected, "fixture_sha256")
    for agent in selected["agents"]:
        assert agent["agent_fixture_sha256"] == _digest_without(
            agent, "agent_fixture_sha256"
        )
        assert agent["materialized_genome_sha256"] == canonical_hash_from_genome_dict(
            agent["record"]["genome"]
        )


def test_every_panel_has_eight_targets_and_eight_intact_twins(
    registry: dict,
) -> None:
    for case in registry["cases"]:
        agents = case["agents"]
        assert len(agents) == PANEL_AGENT_COUNT == 16
        assert [agent["agent_slot"] for agent in agents] == list(range(16))
        assert [agent["variant_id"] for agent in agents[:8]] == [
            f"v{index}" for index in range(8)
        ]
        assert [agent["variant_id"] for agent in agents[8:]] == [
            f"v{index}" for index in range(8)
        ]
        for index in range(8):
            target = agents[index]
            twin = agents[index + 8]
            assert target["panel_half"] == "target"
            assert twin["panel_half"] == "intact_twin"
            assert (target["twin_slot"], twin["twin_slot"]) == (index + 8, index)
            assert target["intact_genome_sha256"] == twin["intact_genome_sha256"]
            assert twin["materialized_genome_sha256"] == twin["intact_genome_sha256"]


def test_interventions_are_explicit_and_consume_no_rng(
    registry: dict,
) -> None:
    for case in registry["cases"]:
        assert case["modular_rng_draw_count"] == MODULAR_RNG_DRAW_COUNT == 0
        for index, agent in enumerate(case["agents"]):
            intervention = agent["intervention"]
            assert intervention["modular_rng_draw_count"] == 0
            assert intervention["targeted"] is (index < 8)
            assert intervention["action"] == (
                case["role_action"] if index < 8 else "intact_twin"
            )
            assert isinstance(intervention["operations"], list)


def test_member_state_reset_changes_exactly_one_named_member(registry: dict) -> None:
    cases = [
        case
        for case in registry["cases"]
        if case["family_id"] == "f06" and case["role_id"] == "r4"
    ]
    assert len(cases) == 2
    for case in cases:
        for agent in case["agents"][:8]:
            operations = agent["intervention"]["operations"]
            assert len(operations) == 1
            assert operations[0]["operation"] == "reset_runtime_state"
            assert len(operations[0]["module_instance_ids"]) == 1


def test_every_family_variant_and_matrix_action_recipe_is_represented(
    matrix: dict, registry: dict
) -> None:
    expected_actions = {
        (case["family_id"], case["profile_id"], case["role_id"]): case["role_contract"][
            "action"
        ]
        for case in matrix["cases"]
    }
    observed_actions = {
        (case["family_id"], case["profile_id"], case["role_id"]): case["role_action"]
        for case in registry["cases"]
    }
    assert observed_actions == expected_actions
    assert {case["family_id"] for case in registry["cases"]} == {
        f"f{index:02d}" for index in range(1, 13)
    }
    for case in registry["cases"]:
        assert {agent["variant_id"] for agent in case["agents"]} == {
            f"v{index}" for index in range(8)
        }
        assert all(agent["variant_description"] for agent in case["agents"])


def test_all_96_family_variants_have_concrete_semantics_in_both_profiles(
    registry: dict,
) -> None:
    checked: set[tuple[str, str, int]] = set()
    for profile_code in ("b12", "b16"):
        for family_id, assertion in _FAMILY_SEMANTIC_ASSERTIONS.items():
            case = next(
                row
                for row in registry["cases"]
                if row["profile_code"] == profile_code
                and row["family_id"] == family_id
                and row["role_id"] == "r0"
            )
            for variant, agent in enumerate(case["agents"][8:]):
                assertion(variant, agent)
                checked.add((profile_code, family_id, variant))
    assert len(checked) == 2 * 12 * 8


def test_batch_execution_recipe_distinguishes_f1_disabled_and_enabled_empty(
    registry: dict,
) -> None:
    for case in registry["cases"]:
        recipe = case["execution_recipe"]
        assert recipe["scope"] == "whole_single_cell_invocation"
        assert recipe["simulation_executed"] is False
        expected_enabled = not (case["family_id"] == "f01" and case["role_id"] == "r0")
        assert recipe["modular_enabled"] is expected_enabled
    for profile_code in ("b12", "b16"):
        disabled = next(
            case
            for case in registry["cases"]
            if case["family_id"] == "f01"
            and case["profile_code"] == profile_code
            and case["role_id"] == "r0"
        )
        enabled = next(
            case
            for case in registry["cases"]
            if case["family_id"] == "f01"
            and case["profile_code"] == profile_code
            and case["role_id"] == "r1"
        )
        assert all(
            not agent["record"]["genome"]["modules"]
            for agent in disabled["agents"]
            if agent["variant_id"] in {"v0", "v1"}
        )
        assert all(
            not agent["record"]["genome"]["modules"]
            for agent in enabled["agents"]
            if agent["variant_id"] in {"v0", "v1"}
        )


def test_named_f1_f4_and_f7_actions_are_real_interventions(registry: dict) -> None:
    for profile_code in ("b12", "b16"):
        f1 = next(
            case
            for case in registry["cases"]
            if case["profile_code"] == profile_code
            and case["family_id"] == "f01"
            and case["role_id"] == "r4"
        )
        for slot in range(8):
            target = f1["agents"][slot]
            twin = f1["agents"][slot + 8]
            if target["variant_id"] in {"v0", "v1"}:
                assert target["record"]["genome"] == twin["record"]["genome"]
                for field in (
                    "structural_live_module_state",
                    "structural_live_module_outputs",
                    "structural_development_latch_by_instance_id",
                    "structural_development_effective_gain_by_instance_id",
                ):
                    assert target["record"][field] == twin["record"][field]
                continue
            assert len(target["record"]["genome"]["modules"]) == (
                len(twin["record"]["genome"]["modules"]) + 1
            )
            assert target["intervention"]["operations"] == [
                {
                    "operation": "add_neutral_duplicate",
                    "parent_module_instance_id": twin["record"]["genome"]["modules"][0][
                        "module_instance_id"
                    ],
                    "duplicate_module_id": target["record"]["genome"]["modules"][-1][
                        "module_id"
                    ],
                    "duplicate_module_instance_id": target["record"]["genome"][
                        "modules"
                    ][-1]["module_instance_id"],
                }
            ]

        f4 = next(
            case
            for case in registry["cases"]
            if case["profile_code"] == profile_code
            and case["family_id"] == "f04"
            and case["role_id"] == "r3"
        )
        assert any(
            operation["operation"] == "remove_output_bindings"
            for agent in f4["agents"][:8]
            for operation in agent["intervention"]["operations"]
        )

        f7 = next(
            case
            for case in registry["cases"]
            if case["profile_code"] == profile_code
            and case["family_id"] == "f07"
            and case["role_id"] == "r2"
        )
        assert all(
            any(
                operation["operation"] == "zero_developmental_expression"
                for operation in agent["intervention"]["operations"]
            )
            for agent in f7["agents"][:8]
        )


def _without_param_keys(module: dict, keys: set[str]) -> dict:
    value = deepcopy(module)
    params = dict(value["params"])
    for key in keys:
        params.pop(key, None)
    value["params"] = params
    return value


def test_every_intervention_is_exactly_isolated_to_its_named_surface(
    registry: dict,
) -> None:
    for case in registry["cases"]:
        for slot in range(8):
            target = case["agents"][slot]
            twin = case["agents"][slot + 8]
            operations = target["intervention"]["operations"]
            target_record = target["record"]
            twin_record = twin["record"]
            target_genome = target_record["genome"]
            twin_genome = twin_record["genome"]

            if not operations or all(
                operation["operation"]
                in {"boundary_or_reporting_recipe", "prepared_boundary_recipe"}
                for operation in operations
            ):
                assert target_genome == twin_genome
                for field in (
                    "structural_live_module_state",
                    "structural_live_module_outputs",
                    "structural_development_latch_by_instance_id",
                    "structural_development_effective_gain_by_instance_id",
                ):
                    assert target_record[field] == twin_record[field]
                continue

            allowed_params: dict[str, set[str]] = {}
            removed_connections: set[str] = set()
            allowed_state: set[str] = set()
            allowed_output: set[str] = set()
            allowed_latch: set[str] = set()
            allowed_effective: set[str] = set()
            added_module_ids: set[str] = set()
            added_instance_ids: set[str] = set()
            removed_module_ids: set[str] = set()
            removed_instance_ids: set[str] = set()
            twin_by_local = {
                str(module["module_id"]): module for module in twin_genome["modules"]
            }
            twin_by_instance = {
                str(module["module_instance_id"]): module
                for module in twin_genome["modules"]
            }
            for operation in operations:
                name = operation["operation"]
                if name == "remove_input_bindings":
                    for module_id in operation["module_ids"]:
                        allowed_params.setdefault(module_id, set()).add(
                            ROUTING_BLOCK_KEY
                        )
                elif name == "remove_output_bindings":
                    for module_id in operation["module_ids"]:
                        allowed_params.setdefault(module_id, set()).add(
                            OUTPUT_ROUTING_BLOCK_KEY
                        )
                elif name == "reset_runtime_state":
                    allowed_state.update(operation["module_instance_ids"])
                    allowed_output.update(operation["module_instance_ids"])
                elif name == "zero_developmental_expression":
                    module_id = operation["module_id"]
                    allowed_params.setdefault(module_id, set()).add(
                        DEVELOPMENTAL_EXPRESSION_BLOCK_KEY
                    )
                    allowed_effective.add(
                        str(twin_by_local[module_id]["module_instance_id"])
                    )
                elif name == "enable_dormancy":
                    allowed_params.setdefault(operation["module_id"], set()).add(
                        ACTIVITY_RECORD_KEY
                    )
                elif name == "block_dependency_readiness":
                    prerequisite_id = operation["prerequisite_module_id"]
                    target_id = operation["target_module_id"]
                    allowed_params.setdefault(prerequisite_id, set()).add(
                        ACTIVITY_RECORD_KEY
                    )
                    target_instance = str(
                        twin_by_local[target_id]["module_instance_id"]
                    )
                    allowed_effective.add(target_instance)
                    allowed_latch.add(target_instance)
                elif name == "remove_connection":
                    removed_connections.add(operation["connection_id"])
                elif name == "add_neutral_duplicate":
                    added_module_ids.add(operation["duplicate_module_id"])
                    duplicate_instance_id = operation["duplicate_module_instance_id"]
                    added_instance_ids.add(duplicate_instance_id)
                    allowed_state.add(duplicate_instance_id)
                    allowed_output.add(duplicate_instance_id)
                elif name == "prepared_fixture_prune":
                    removed_instance_id = operation["removed_module_instance_id"]
                    removed_instance_ids.add(removed_instance_id)
                    removed_module_ids.add(
                        str(twin_by_instance[removed_instance_id]["module_id"])
                    )
                    allowed_state.add(removed_instance_id)
                    allowed_output.add(removed_instance_id)
                    allowed_latch.add(removed_instance_id)
                    allowed_effective.add(removed_instance_id)
                else:
                    raise AssertionError(f"unclassified intervention operation {name}")

            assert {
                str(module["module_instance_id"]) for module in target_genome["modules"]
            } == (set(twin_by_instance) - removed_instance_ids) | added_instance_ids
            for module in target_genome["modules"]:
                module_id = str(module["module_id"])
                if module_id in added_module_ids:
                    assert module["params"]["live_input_gain"] == 0.0
                    assert module["params"]["live_output_gain"] == 0.0
                    continue
                assert _without_param_keys(
                    module, allowed_params.get(module_id, set())
                ) == _without_param_keys(
                    twin_by_local[module_id], allowed_params.get(module_id, set())
                )
            expected_connections = [
                connection
                for connection in twin_genome["connections"]
                if connection["connection_id"] not in removed_connections
                and connection["src_module_id"] not in removed_module_ids
                and connection["dst_module_id"] not in removed_module_ids
            ]
            assert target_genome["connections"] == expected_connections
            for field, allowed in (
                ("structural_live_module_state", allowed_state),
                ("structural_live_module_outputs", allowed_output),
                ("structural_development_latch_by_instance_id", allowed_latch),
                (
                    "structural_development_effective_gain_by_instance_id",
                    allowed_effective,
                ),
            ):
                target_values = {
                    key: value
                    for key, value in target_record[field].items()
                    if key not in allowed
                }
                twin_values = {
                    key: value
                    for key, value in twin_record[field].items()
                    if key not in allowed
                }
                assert target_values == twin_values


def test_both_profiles_use_fixed_16_by_48_physical_envelope_and_b12_tail_is_zero(
    registry: dict,
) -> None:
    assert {
        (case["physical_module_slots"], case["physical_connection_slots"])
        for case in registry["cases"]
    } == {(16, 48)}
    assert all(case["canonical_tail_verified"] is True for case in registry["cases"])

    b12 = next(case for case in registry["cases"] if case["profile_code"] == "b12")
    packed = pack_modular_storage(
        [agent["record"] for agent in b12["agents"]],
        agent_capacity=16,
        profile_id="b2-12x32-v1",
    )
    assert packed.layout.logical_module_slots == 12
    assert packed.layout.logical_connection_slots == 32
    for field in packed.layout.fields:
        tensor = packed.tensors[field.name]
        if len(field.shape) >= 2 and field.shape[1] == 16:
            tail = tensor[:, 12:]
        elif len(field.shape) >= 2 and field.shape[1] == 48:
            tail = tensor[:, 32:]
        else:
            continue
        assert bool((tail == field.fill_value).all().item()), field.name


def test_intact_and_sham_target_records_remain_equal_to_their_twins(
    registry: dict,
) -> None:
    controls = [case for case in registry["cases"] if case["role_id"] in {"r0", "r1"}]
    assert len(controls) == 48
    for case in controls:
        for index in range(8):
            target = case["agents"][index]["record"]
            twin = case["agents"][index + 8]["record"]
            assert target["genome"] == twin["genome"]
            assert (
                target["structural_live_module_state"]
                == twin["structural_live_module_state"]
            )
            assert (
                target["structural_live_module_outputs"]
                == twin["structural_live_module_outputs"]
            )


def test_case_validator_rejects_unknown_altered_and_noncanonical_cases(
    matrix: dict, registry: dict
) -> None:
    with pytest.raises(PhaseFV2FixtureError, match="unknown Phase F case"):
        materialize_phase_f_case_v2("pf2_f99_b12_r0", matrix=matrix)

    altered_matrix_case = deepcopy(matrix["cases"][0])
    altered_matrix_case["role_id"] = "r9"
    with pytest.raises(PhaseFV2FixtureError, match="frozen matrix binding"):
        materialize_phase_f_case_v2(altered_matrix_case, matrix=matrix)

    assert (
        validate_phase_f_fixture_case_v2(registry["cases"][0], matrix=matrix)
        == registry["cases"][0]
    )
    altered = deepcopy(registry["cases"][0])
    altered["agents"][0]["intervention"]["modular_rng_draw_count"] = 1
    with pytest.raises(PhaseFV2FixtureError, match="does not match"):
        validate_phase_f_fixture_case_v2(altered, matrix=matrix)
    with pytest.raises(PhaseFV2FixtureError, match="string-keyed"):
        validate_phase_f_fixture_case_v2([], matrix=matrix)


def test_registry_validator_fails_closed_on_missing_unknown_or_changed_content(
    matrix: dict, registry: dict
) -> None:
    assert validate_phase_f_fixture_registry_v2(registry, matrix=matrix) == registry

    missing = deepcopy(registry)
    missing["cases"].pop()
    with pytest.raises(PhaseFV2FixtureError, match="incomplete, altered"):
        validate_phase_f_fixture_registry_v2(missing, matrix=matrix)

    extra = deepcopy(registry)
    extra["unexpected"] = True
    with pytest.raises(PhaseFV2FixtureError, match="incomplete, altered"):
        validate_phase_f_fixture_registry_v2(extra, matrix=matrix)
