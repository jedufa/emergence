from __future__ import annotations

from copy import deepcopy

import pytest

from evolution.phase_f_v2.canonical import canonical_sha256
from evolution.phase_f_v2.matrix import build_phase_f_v2_matrix
from perf.eiel_phase_f_forced_combination_fixture_v2 import (
    build_phase_f_fixture_registry_v2,
)
from perf.eiel_phase_f_relationship_analysis_v2 import (
    CAUSAL_SCHEMA,
    CPU_BINDING_SCHEMA,
    DECISION_A,
    DECISION_B,
    DECISION_C,
    DECISION_D,
    OBSERVATION_BINDING_SCHEMA,
    REPEAT_SCHEMA,
    aggregate_phase_f_relationship_terminal_v2,
    bind_phase_f_accelerated_observation_v2,
    build_phase_f_causal_relationships_v2,
    build_phase_f_checkpoint_relationships_v2,
    build_phase_f_repeat_relationships_v2,
    expected_effect_slots_v2,
    select_phase_f_terminal_decision_v2,
)


@pytest.fixture(scope="module")
def matrix():
    return build_phase_f_v2_matrix()


@pytest.fixture(scope="module")
def registry(matrix):
    return build_phase_f_fixture_registry_v2(matrix)


def _finish(value):
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _observation(
    case_id: str,
    *,
    effect: bool = False,
    effect_slot: int = 0,
    checkpoint_kind="none",
):
    zeros = [0.0] * 16
    module = list(zeros)
    residual = [[0.0] * 7 for _ in range(16)]
    state = [{"before": [0.0], "after": [0.0]} for _ in range(16)]
    dormancy = [[False] for _ in range(16)]
    if effect:
        module[effect_slot] = 1.0
        residual[effect_slot][0] = 1.0
        state[effect_slot] = {"before": [1.0], "after": [1.0]}
        dormancy[effect_slot] = [True]
    return {
        "actions": [0] * 16,
        "checkpoint_projection": {
            "expectation_kind": checkpoint_kind,
            "source_sha256": f"{case_id}:source",
            "execution_source_sha256": f"{case_id}:execution",
            "planes_before": [],
            "planes_after": [],
        },
        "dependency_readiness": [[False] for _ in range(16)],
        "developmental_stage": [0] * 16,
        "dormancy": dormancy,
        "fallback_count": 0,
        "free_slot_maps": [{"module_free": [], "connection_free": []}] * 16,
        "homology": [[] for _ in range(16)],
        "identities": [{"agent_id": index} for index in range(16)],
        "modular_rng_draw_count": 0,
        "module_and_connection_presence": [{} for _ in range(16)],
        "profile_and_owner_ids": {
            "case_id": case_id,
            "profile_id": "profile",
            "owner_id": "owner",
            "lane_id": "lane",
        },
        "semantic_ids": [{} for _ in range(16)],
        "slot_maps": [{} for _ in range(16)],
        "validity_masks": [{} for _ in range(16)],
        "action_logits": residual,
        "developmental_effective_gain": zeros,
        "module_output": module,
        "module_state": module,
        "previous_output": zeros,
        "previous_tick_connection_input": zeros,
        "routed_action_residual": residual,
        "routed_observation": [[0.0] for _ in range(16)],
        "state_before_and_after": state,
        "structural_cost_components_and_total": [
            {"components": [0.0], "total": 0.0} for _ in range(16)
        ],
    }


def _cpu(case, observation, kind="uninterrupted", fixture_sha=None):
    value = {
        "schema_version": CPU_BINDING_SCHEMA,
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "expectation_kind": kind,
        "fixture_sha256": fixture_sha
        or f"{int(case['case_sha256'], 16):064x}"[-64:],
        "source_sha256": "1" * 64,
        "expectation_sha256": canonical_sha256(observation),
        "oracle_receipt_sha256": "2" * 64,
        "outcome": "passed",
        "expectation": deepcopy(observation),
    }
    return _finish(value)


def _accelerated(
    case, observation, repeat, kind="uninterrupted", fixture_sha=None
):
    value = {
        "schema_version": OBSERVATION_BINDING_SCHEMA,
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "execution_id": f"{case['case_id']}.{kind}.{repeat}",
        "invocation_kind": kind,
        "repeat_index": repeat,
        "fixture_sha256": fixture_sha
        or f"{int(case['case_sha256'], 16):064x}"[-64:],
        "source_sha256": "1" * 64,
        "expected_sha256": canonical_sha256(observation),
        "observed_sha256": canonical_sha256(observation),
        "owner_id": "owner",
        "profile_id": case["profile_id"],
        "lane_id": "lane",
        "outcome": "passed",
        "invocation_receipt_sha256": f"{repeat + 3:064x}",
        "observation": deepcopy(observation),
    }
    return _finish(value)


def _case_map(matrix):
    return {case["case_id"]: case for case in matrix["cases"]}


def test_effect_slot_table_is_exact(matrix) -> None:
    cases = _case_map(matrix)
    expected = {
        "pf2_f02_b12_r2": (0, 1, 2, 3, 6, 7),
        "pf2_f03_b12_r2": (0, 1, 2, 3, 4, 5, 6),
        "pf2_f03_b12_r3": (6, 7),
        "pf2_f04_b12_r3": (1, 2, 3, 4, 5, 6, 7),
        "pf2_f05_b12_r3": (3,),
        "pf2_f07_b12_r2": tuple(range(8)),
    }
    for case_id, slots in expected.items():
        assert expected_effect_slots_v2(cases[case_id]) == slots
        twin = case_id.replace("_b12_", "_b16_")
        assert expected_effect_slots_v2(cases[twin]) == slots


def test_repeat_relationship_requires_exact_observed_digest(matrix) -> None:
    case = matrix["cases"][0]
    observation = _observation(case["case_id"])
    bindings = [
        _accelerated(case, observation, repeat)
        for repeat in (0, 1)
    ]
    relationships = build_phase_f_repeat_relationships_v2(
        matrix=matrix, accelerated_bindings=bindings
    )
    assert len(relationships) == 1
    assert relationships[0]["outcome"] == "passed"

    changed = deepcopy(bindings[1])
    changed["observation"]["module_output"][0] = 1.0
    changed["observed_sha256"] = canonical_sha256(changed["observation"])
    changed.pop("receipt_sha256")
    changed = _finish(changed)
    failed = build_phase_f_repeat_relationships_v2(
        matrix=matrix, accelerated_bindings=[bindings[0], changed]
    )
    assert failed[0]["outcome"] == "failed"
    assert failed[0]["checks"]["observation_exact"] is False


@pytest.mark.parametrize(
    ("raw_outcome", "expected"),
    (("no_mismatch", "passed"), ("first_difference", "failed")),
)
def test_actual_invocation_outcomes_are_normalized(raw_outcome, expected) -> None:
    observation = _observation("case")
    invocation = {
        "case_id": "case",
        "case_sha256": "1" * 64,
        "execution_id": "case.uninterrupted.0",
        "invocation_kind": "uninterrupted",
        "repeat_index": 0,
        "fixture_sha256": "2" * 64,
        "source_sha256": "3" * 64,
        "expected_sha256": "4" * 64,
        "observed_sha256": canonical_sha256(observation),
        "owner_id": "owner",
        "profile_id": "profile",
        "lane_id": "lane",
        "outcome": raw_outcome,
    }
    invocation = _finish(invocation)
    binding = bind_phase_f_accelerated_observation_v2(
        invocation_receipt=invocation, observation=observation
    )
    assert binding["outcome"] == expected


def _causal_inputs(matrix, registry, case_id, *, effect):
    cases = _case_map(matrix)
    fixtures = {row["case_id"]: row for row in registry["cases"]}
    lesion = cases[case_id]
    ids = (
        lesion["relationship"]["base_case_id"],
        lesion["relationship"]["neutral_case_id"],
        case_id,
    )
    cpu = []
    accelerated = []
    for index, related_id in enumerate(ids):
        case = cases[related_id]
        observation = _observation(related_id, effect=effect and index == 2)
        fixture_sha = fixtures[related_id]["fixture_sha256"]
        cpu.append(_cpu(case, observation, fixture_sha=fixture_sha))
        accelerated.extend(
            _accelerated(
                case, observation, repeat, fixture_sha=fixture_sha
            )
            for repeat in (0, 1)
        )
    repeats = build_phase_f_repeat_relationships_v2(
        matrix=matrix, accelerated_bindings=accelerated
    )
    return cpu, accelerated, repeats


def test_no_op_lesion_is_derived_failure(matrix, registry) -> None:
    case_id = "pf2_f04_b12_r3"
    cpu, accelerated, repeats = _causal_inputs(
        matrix, registry, case_id, effect=False
    )
    relationships = build_phase_f_causal_relationships_v2(
        matrix=matrix,
        fixture_registry=registry,
        cpu_bindings=cpu,
        accelerated_bindings=accelerated,
        repeat_relationships=repeats,
    )
    assert len(relationships) == 1
    assert relationships[0]["outcome"] == "failed"
    assert (
        relationships[0]["checks"][
            "cpu_required_effect:named_routed_residual"
        ]
        is False
    )


def test_unrelated_fixture_byte_change_is_derived_failure(
    matrix, registry
) -> None:
    case_id = "pf2_f02_b12_r4"
    cpu, accelerated, repeats = _causal_inputs(
        matrix, registry, case_id, effect=True
    )
    altered = deepcopy(registry)
    fixture = next(case for case in altered["cases"] if case["case_id"] == case_id)
    fixture["agents"][15]["record"]["agent_id"] += 1
    relationships = build_phase_f_causal_relationships_v2(
        matrix=matrix,
        fixture_registry=altered,
        cpu_bindings=cpu,
        accelerated_bindings=accelerated,
        repeat_relationships=repeats,
    )
    assert relationships[0]["outcome"] == "failed"
    assert relationships[0]["checks"]["unrelated_fixture_bytes_exact"] is False


def test_checkpoint_relationship_detects_normalized_mismatch(matrix) -> None:
    case = next(
        row
        for row in matrix["cases"]
        if row["checkpoint_schedule"]["kind"] != "none"
    )
    ordinary = _observation(case["case_id"], checkpoint_kind="uninterrupted")
    resumed = _observation(case["case_id"], checkpoint_kind="resumed")
    cpu = [_cpu(case, ordinary), _cpu(case, resumed, kind="resumed")]
    accelerated = [
        *(_accelerated(case, ordinary, repeat) for repeat in (0, 1)),
        *(
            _accelerated(case, resumed, repeat, kind="resumed")
            for repeat in (0, 1)
        ),
    ]
    repeats = build_phase_f_repeat_relationships_v2(
        matrix=matrix, accelerated_bindings=accelerated
    )
    passed = build_phase_f_checkpoint_relationships_v2(
        matrix=matrix,
        cpu_bindings=cpu,
        accelerated_bindings=accelerated,
        repeat_relationships=repeats,
    )
    assert passed[0]["outcome"] == "passed"

    altered = deepcopy(accelerated)
    for row in altered[2:]:
        row["observation"]["module_state"][0] = 2.0
        row["observed_sha256"] = canonical_sha256(row["observation"])
        row.pop("receipt_sha256")
        _finish(row)
    repeats = build_phase_f_repeat_relationships_v2(
        matrix=matrix, accelerated_bindings=altered
    )
    failed = build_phase_f_checkpoint_relationships_v2(
        matrix=matrix,
        cpu_bindings=cpu,
        accelerated_bindings=altered,
        repeat_relationships=repeats,
    )
    assert failed[0]["outcome"] == "failed"
    assert failed[0]["checks"]["gpu_checkpoint_projection_exact"] is False


@pytest.mark.parametrize(
    ("integrity", "composition", "incomplete", "expected"),
    (
        (False, False, False, DECISION_A),
        (False, False, True, DECISION_D),
        (False, True, False, DECISION_B),
        (False, True, True, DECISION_B),
        (True, False, False, DECISION_C),
        (True, False, True, DECISION_C),
        (True, True, False, DECISION_C),
        (True, True, True, DECISION_C),
    ),
)
def test_terminal_precedence_truth_table(
    integrity, composition, incomplete, expected
) -> None:
    assert (
        select_phase_f_terminal_decision_v2(
            integrity_failed=integrity,
            composition_failed=composition,
            incomplete=incomplete,
        )
        == expected
    )


def _relationship(schema, case_id, outcome="passed", kind=None):
    value = {
        "schema_version": schema,
        "case_id": case_id,
        "outcome": outcome,
        "checks": {"synthetic_derived_check": outcome == "passed"},
    }
    if schema == REPEAT_SCHEMA:
        value["invocation_kind"] = kind
    return _finish(value)


def test_full_derived_inventory_counts_are_exact(matrix, registry) -> None:
    fixtures = {row["case_id"]: row for row in registry["cases"]}
    cpu = []
    accelerated = []
    for case in matrix["cases"]:
        case_id = case["case_id"]
        is_causal = (
            case["role_contract"]["claim_kind"] == "matched_causal_lesion"
        )
        effect_slot = (
            expected_effect_slots_v2(case)[0] if is_causal else 0
        )
        observation = _observation(
            case_id, effect=is_causal, effect_slot=effect_slot
        )
        fixture_sha = fixtures[case_id]["fixture_sha256"]
        cpu.append(_cpu(case, observation, fixture_sha=fixture_sha))
        accelerated.extend(
            _accelerated(
                case,
                observation,
                repeat,
                fixture_sha=fixture_sha,
            )
            for repeat in (0, 1)
        )
        if case["checkpoint_schedule"]["kind"] != "none":
            resumed = _observation(case_id, checkpoint_kind="resumed")
            cpu.append(
                _cpu(
                    case,
                    resumed,
                    kind="resumed",
                    fixture_sha=fixture_sha,
                )
            )
            accelerated.extend(
                _accelerated(
                    case,
                    resumed,
                    repeat,
                    kind="resumed",
                    fixture_sha=fixture_sha,
                )
                for repeat in (0, 1)
            )
    repeats = build_phase_f_repeat_relationships_v2(
        matrix=matrix, accelerated_bindings=accelerated
    )
    causal = build_phase_f_causal_relationships_v2(
        matrix=matrix,
        fixture_registry=registry,
        cpu_bindings=cpu,
        accelerated_bindings=accelerated,
        repeat_relationships=repeats,
    )
    checkpoints = build_phase_f_checkpoint_relationships_v2(
        matrix=matrix,
        cpu_bindings=cpu,
        accelerated_bindings=accelerated,
        repeat_relationships=repeats,
    )
    assert (len(cpu), len(accelerated)) == (134, 268)
    assert len(repeats) == 134
    assert len(causal) == 42
    assert len(checkpoints) == 14


def test_terminal_inventory_missing_is_d_and_failures_override_it(matrix) -> None:
    incomplete = aggregate_phase_f_relationship_terminal_v2(
        matrix=matrix,
        cpu_bindings=[],
        accelerated_bindings=[],
        repeat_relationships=[],
        causal_relationships=[],
        checkpoint_relationships=[],
    )
    assert incomplete["decision"] == DECISION_D
    assert incomplete["missing_counts"] == {
        "cpu_bindings": 134,
        "accelerated_bindings": 268,
        "repeat_relationships": 134,
        "causal_relationships": 42,
        "checkpoint_relationships": 14,
    }

    causal_case = next(
        case
        for case in matrix["cases"]
        if case["role_contract"]["claim_kind"] == "matched_causal_lesion"
    )
    composition = aggregate_phase_f_relationship_terminal_v2(
        matrix=matrix,
        cpu_bindings=[],
        accelerated_bindings=[],
        repeat_relationships=[],
        causal_relationships=[
            _relationship(CAUSAL_SCHEMA, causal_case["case_id"], "failed")
        ],
        checkpoint_relationships=[],
    )
    assert composition["decision"] == DECISION_B

    first_case = matrix["cases"][0]
    observation = _observation(first_case["case_id"])
    failed_accelerated = _accelerated(first_case, observation, 0)
    failed_accelerated["outcome"] = "first_difference"
    failed_accelerated.pop("receipt_sha256")
    _finish(failed_accelerated)
    integrity = aggregate_phase_f_relationship_terminal_v2(
        matrix=matrix,
        cpu_bindings=[],
        accelerated_bindings=[failed_accelerated],
        repeat_relationships=[],
        causal_relationships=[
            _relationship(CAUSAL_SCHEMA, causal_case["case_id"], "failed")
        ],
        checkpoint_relationships=[],
    )
    assert integrity["decision"] == DECISION_C
