from __future__ import annotations

from copy import deepcopy
import json

import pytest

from agents.structural_identity_v1 import new_native_identity
from evolution.operators_c1 import (
    _op_duplicate_with_decay,
    _op_fusion_merge_compression,
    _op_motif_duplication,
    _op_promoter_expression_mutation,
    apply_gated_recurrent_operator_attempt,
    apply_operator_pipeline,
    mutate_gated_recurrent_operator,
)
from evolution.recombination_c2 import recombine_lineage_payloads
from evolution.structural_development_v1 import (
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
    DEVELOPMENTAL_STAGE_PROFILE_VERSION,
)
from evolution.structural_gated_recurrent_v1 import (
    GATED_EQUATION_VERSION,
    GATED_NUMERIC_VERSION,
    GATED_OPERATOR,
    GATED_RECORD_SCHEMA_VERSION,
    neutral_gated_recurrent_record,
)
from evolution.structural_observation_routing_v1 import (
    ROUTING_REGISTRY_VERSION,
    ROUTING_SCHEMA_VERSION,
)
from evolution.structural_output_routing_v1 import (
    ACTION_SEMANTIC_REGISTRY_VERSION,
    OUTPUT_ROUTING_SCHEMA_VERSION,
)
from simulation.rng_manager import RNGManager


_OPERATOR = GATED_OPERATOR
_PARAM_SCHEMA = GATED_RECORD_SCHEMA_VERSION
_EQUATION = GATED_EQUATION_VERSION
_NUMERIC = GATED_NUMERIC_VERSION


def _identity(index: int, role: str) -> dict:
    return new_native_identity(
        context={
            "campaign_id": "escap7-lifecycle-test",
            "branch_id": "fixed-width-gated",
            "generation": 1,
            "tick": 0,
            "child_agent_id": int(index),
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap7-lifecycle-lineage",
        },
        operator_id="fixture",
        attempt_index=int(index),
        operator_role=str(role),
    )


def _gated_record(**updates: float | int | str) -> dict:
    record: dict = neutral_gated_recurrent_record()
    record["output_gain"] = 1.25
    record.update(updates)
    return record


def _development(*, prerequisites: tuple[str, ...] = ()) -> dict:
    return {
        "schema_version": DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
        "stage_profile_version": DEVELOPMENTAL_STAGE_PROFILE_VERSION,
        "stage_gains": [1.0, 1.0, 1.0],
        "prerequisite_homology_ids": list(prerequisites),
        "latch_dependency_readiness": False,
    }


def _module(
    index: int,
    *,
    module_id: str | None = None,
    gated_record: dict | None = None,
    prerequisites: tuple[str, ...] = (),
) -> dict:
    identity = _identity(index, f"module-{index}")
    params = {
        "live_operator": _OPERATOR,
        "expression_scale": 0.0,
        "live_retention": 0.5,
        "live_input_gain": 0.5,
        "live_output_gain": 0.0,
        "gated_recurrent": deepcopy(gated_record or _gated_record()),
        DEVELOPMENTAL_EXPRESSION_BLOCK_KEY: _development(
            prerequisites=prerequisites
        ),
        "observation_routing": {
            "schema_version": ROUTING_SCHEMA_VERSION,
            "registry_version": ROUTING_REGISTRY_VERSION,
            "bindings": [],
        },
        "output_routing": {
            "schema_version": OUTPUT_ROUTING_SCHEMA_VERSION,
            "registry_version": ACTION_SEMANTIC_REGISTRY_VERSION,
            "bindings": [],
        },
    }
    return {
        "module_id": module_id or f"m{index:03d}",
        "module_type": "generic",
        "params": params,
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **identity,
        "latent_pos": list(identity["topology_pos"]),
    }


def _payload(modules: list[dict], connections: list[dict] | None = None) -> dict:
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": deepcopy(modules),
        "connections": deepcopy(connections or []),
    }


def _operator_context(index: int = 0) -> dict:
    return {
        "operator_id": "escap7-test",
        "attempt_index": int(index),
        "identity_context": {
            "campaign_id": "escap7-lifecycle-test",
            "branch_id": "fixed-width-gated",
            "generation": 2,
            "tick": 0,
            "child_agent_id": 100 + int(index),
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap7-child-lineage",
        },
    }


class _CountingRng:
    def __init__(self, value: float = 0.25) -> None:
        self.value = float(value)
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        return self.value


class _SequenceRng:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.draws = 0

    def random(self) -> float:
        if self.draws >= len(self.values):
            raise AssertionError("unexpected hidden RNG draw")
        value = self.values[self.draws]
        self.draws += 1
        return float(value)


def _integrator(index: int, *, expression: float, output_gain: float) -> dict:
    module = _module(index)
    params = module["params"]
    params.pop("gated_recurrent")
    params["live_operator"] = "bounded_scalar_integrator_v1"
    params["expression_scale"] = float(expression)
    params["live_output_gain"] = float(output_gain)
    return module


def _modules_by_homology(modules: list[dict]) -> dict[str, dict]:
    return {str(module["homology_id"]): module for module in modules}


def test_neutral_acquisition_preserves_full_module_and_rejects_active_integrator() -> None:
    neutral = _integrator(100, expression=0.0, output_gain=1.0)
    before = deepcopy(neutral)
    accepted = mutate_gated_recurrent_operator(
        [neutral],
        0.1,
        0.5,
        enabled=True,
    )
    assert accepted.telemetry["status"] == "accepted"
    acquired = accepted.modules[0]
    assert acquired["params"]["live_operator"] == GATED_OPERATOR
    assert acquired["params"]["gated_recurrent"] == neutral_gated_recurrent_record()
    for key in acquired:
        if key != "params":
            assert acquired[key] == before[key]
    for key in before["params"]:
        if key != "live_operator":
            assert acquired["params"][key] == before["params"][key]

    active = _integrator(101, expression=1.0, output_gain=1.0)
    rejected = mutate_gated_recurrent_operator(
        [active],
        0.1,
        0.5,
        enabled=True,
    )
    assert rejected.telemetry["status"] == "ineligible"
    assert rejected.modules == [active]


@pytest.mark.parametrize(
    ("draw0", "expected_operation"),
    (
        (0.0, "acquire_neutral"),
        (0.199999, "acquire_neutral"),
        (0.2, "mutate_reset_parameter"),
        (0.399999, "mutate_reset_parameter"),
        (0.4, "mutate_update_parameter"),
        (0.599999, "mutate_update_parameter"),
        (0.6, "mutate_candidate_parameter"),
        (0.799999, "mutate_candidate_parameter"),
        (0.8, "mutate_output_gain"),
        (0.999999, "mutate_output_gain"),
    ),
)
def test_two_draw_operator_has_exact_five_bin_boundaries(
    draw0: float,
    expected_operation: str,
) -> None:
    module = (
        _integrator(110, expression=0.0, output_gain=1.0)
        if expected_operation == "acquire_neutral"
        else _module(110)
    )
    result = mutate_gated_recurrent_operator(
        [module],
        draw0,
        0.37,
        enabled=True,
    )
    assert result.telemetry["operation"] == expected_operation
    assert result.telemetry["draw_budget"] == {
        "declared_supplied_draws": 2,
        "internal_rng_draws": 0,
    }


@pytest.mark.parametrize(
    ("enabled", "modules", "draws", "status"),
    (
        (False, [_module(120)], [0.25, 0.5], "disabled"),
        (True, [], [0.25, 0.5], "ineligible"),
        (True, [_integrator(121, expression=1.0, output_gain=1.0)], [0.1, 0.5], "ineligible"),
        (True, [_integrator(122, expression=0.0, output_gain=1.0)], [0.1, 0.5], "accepted"),
    ),
)
def test_every_wrapper_outcome_consumes_exactly_two_supplied_draws(
    enabled: bool,
    modules: list[dict],
    draws: list[float],
    status: str,
) -> None:
    rng = _SequenceRng(draws)
    result = apply_gated_recurrent_operator_attempt(
        deepcopy(modules),
        rng=rng,
        enabled=enabled,
    )
    assert rng.draws == 2
    assert result.telemetry["status"] == status
    assert result.telemetry["draw_budget"] == {
        "declared_supplied_draws": 2,
        "internal_rng_draws": 0,
    }


def test_second_draw_selection_uses_stable_biological_order_not_local_packing() -> None:
    first = _module(130)
    second = _module(131)
    expected_homology = min(str(first["homology_id"]), str(second["homology_id"]))
    forward = mutate_gated_recurrent_operator(
        [deepcopy(first), deepcopy(second)],
        0.25,
        0.1,
        enabled=True,
    )
    reversed_result = mutate_gated_recurrent_operator(
        [deepcopy(second), deepcopy(first)],
        0.25,
        0.1,
        enabled=True,
    )
    assert forward.telemetry["selected_homology_id"] == expected_homology
    assert reversed_result.telemetry["selected_homology_id"] == expected_homology
    assert forward.telemetry["selected_field"] == "reset_bias"
    assert forward.telemetry["value_after"] == reversed_result.telemetry["value_after"]
    assert _modules_by_homology(forward.modules) == _modules_by_homology(
        reversed_result.modules
    )


def test_disabled_c1_schedule_is_exact_and_enabled_lane_adds_only_two_draws() -> None:
    parent = _payload([_integrator(140, expression=0.0, output_gain=1.0)])
    historical_rng = _SequenceRng([])
    historical, historical_telemetry = apply_operator_pipeline(
        payload=deepcopy(parent),
        rng=historical_rng,
        max_ops=0,
    )
    disabled_rng = _SequenceRng([])
    disabled, disabled_telemetry = apply_operator_pipeline(
        payload=deepcopy(parent),
        rng=disabled_rng,
        max_ops=0,
        structural_gated_recurrent_mutation_enable=False,
        structural_gated_recurrent_operator_attempts=0,
    )
    assert historical_rng.draws == disabled_rng.draws == 0
    assert historical == disabled
    assert historical_telemetry == disabled_telemetry
    assert "structural_gated_recurrent_mutation" not in disabled_telemetry

    active_rng = _SequenceRng([0.1, 0.5])
    active, active_telemetry = apply_operator_pipeline(
        payload=deepcopy(parent),
        rng=active_rng,
        max_ops=0,
        structural_gated_recurrent_mutation_enable=True,
        structural_gated_recurrent_operator_attempts=1,
    )
    assert active_rng.draws == 2
    assert active_telemetry["structural_gated_recurrent_mutation"][
        "draws_per_attempt"
    ] == 2
    assert active["modules"][0]["params"]["live_operator"] == GATED_OPERATOR


def test_gated_lane_is_locked_in_runtime_rng_budget_contract() -> None:
    manager = RNGManager(global_seed=7)

    assert manager.expected_delta_for_boundary(
        "mutation.c1.operator.gated_scalar_recurrent_mutation"
    ) == {"mutation.py": 2}


def test_gated_promoter_preserves_operator_record_routing_development_and_identity() -> None:
    module = _module(1, gated_record=_gated_record(reset_bias=1.25))
    payload = _payload([module])
    before = deepcopy(payload["modules"][0])

    child, changed = _op_promoter_expression_mutation(payload, [0.0, 0.75], {})

    assert changed is True
    after = child["modules"][0]
    assert after["params"]["expression_scale"] == 1.25
    assert after["params"]["live_operator"] == _OPERATOR
    assert json.dumps(
        after["params"]["gated_recurrent"], sort_keys=True, separators=(",", ":")
    ) == json.dumps(
        before["params"]["gated_recurrent"], sort_keys=True, separators=(",", ":")
    )
    for key in ("observation_routing", "output_routing", DEVELOPMENTAL_EXPRESSION_BLOCK_KEY):
        assert after["params"][key] == before["params"][key]
    for key in (
        "module_instance_id",
        "homology_id",
        "ancestry",
        "topology_pos",
        "latent_pos",
    ):
        assert after[key] == before[key]


def test_gated_single_duplicate_is_output_neutral_deep_copy_with_new_identity() -> None:
    original = _module(2, gated_record=_gated_record(output_gain=-1.5))
    parent = _payload([original])

    child, changed = _op_duplicate_with_decay(
        parent,
        [0.0, 0.5, 0.5],
        _operator_context(2),
    )

    assert changed is True
    assert len(child["modules"]) == 2
    old, duplicate = child["modules"]
    assert old == original
    assert duplicate["params"]["live_operator"] == _OPERATOR
    assert duplicate["params"]["gated_recurrent"]["output_gain"] == 0.0
    assert old["params"]["gated_recurrent"]["output_gain"] == -1.5
    assert duplicate["module_instance_id"] != old["module_instance_id"]
    assert duplicate["homology_id"] != old["homology_id"]
    duplicate["params"]["gated_recurrent"]["reset_bias"] = -4.0
    assert old["params"]["gated_recurrent"]["reset_bias"] == 0.0


def test_gated_motif_duplicates_all_have_zero_output_without_aliasing() -> None:
    first = _module(3, gated_record=_gated_record(output_gain=0.75))
    second = _module(4, gated_record=_gated_record(output_gain=-0.5))
    parent = _payload([first, second])

    child, changed = _op_motif_duplication(
        parent,
        [0.0, 0.5, 0.5],
        _operator_context(3),
    )

    assert changed is True
    assert len(child["modules"]) == 4
    originals = child["modules"][:2]
    duplicates = child["modules"][2:]
    assert [m["params"]["gated_recurrent"]["output_gain"] for m in originals] == [
        0.75,
        -0.5,
    ]
    assert [m["params"]["gated_recurrent"]["output_gain"] for m in duplicates] == [
        0.0,
        0.0,
    ]
    duplicates[0]["params"]["gated_recurrent"]["candidate_bias"] = 4.0
    assert originals[0]["params"]["gated_recurrent"]["candidate_bias"] == 0.0


def test_dependency_invalidating_fusion_rejects_without_partial_commit_or_exception() -> None:
    prerequisite = _module(10)
    dependent = _module(
        11,
        prerequisites=(str(prerequisite["homology_id"]),),
    )
    parent = _payload([prerequisite, dependent])
    before = deepcopy(parent)
    context: dict = {}

    # Keep the second module and attempt to drop the first prerequisite.
    child, changed = _op_fusion_merge_compression(parent, [0.75, 0.0], context)

    assert changed is False
    assert child == before
    assert "dependency" in str(context.get("result_reason", ""))


def test_fusion_created_live_cycle_rejects_full_proposal_and_preserves_parent() -> None:
    keep = _module(12)
    drop = _module(13)
    target = _module(14)
    connections = [
        {
            "connection_id": "c000",
            "src_module_id": drop["module_id"],
            "dst_module_id": target["module_id"],
            "src_slot": "out",
            "dst_slot": "in",
            "value_type": "scalar",
            "channel": "typed",
            "weight": 0.5,
        },
        {
            "connection_id": "c001",
            "src_module_id": target["module_id"],
            "dst_module_id": keep["module_id"],
            "src_slot": "out",
            "dst_slot": "in",
            "value_type": "scalar",
            "channel": "typed",
            "weight": 0.5,
        },
    ]
    parent = _payload([keep, drop, target], connections)
    before = deepcopy(parent)
    context: dict = {}

    # Two supplied historical fusion draws select keep=m012 and drop=m013.
    # Remapping m013->m014 to m012->m014 would create m012<->m014.
    child, changed = _op_fusion_merge_compression(parent, [0.0, 0.4], context)

    assert changed is False
    assert child == before
    assert context["result_reason"] == "fusion_contract_rejected"
    assert "CYCLE" in str(context.get("fusion_rejection_detail", ""))


@pytest.mark.parametrize(
    ("fault", "value"),
    (
        ("schema_version", "unsupported"),
        ("missing_reset_bias", None),
        ("extra_field", 1.0),
        ("reset_bias", float("nan")),
        ("output_gain", 3.0),
    ),
)
def test_c2_incompatible_or_invalid_gated_record_rejects_before_donor_draw(
    fault: str,
    value: object,
) -> None:
    shared = _identity(20, "shared")
    first = _module(20)
    second = deepcopy(first)
    for module in (first, second):
        module.update(deepcopy(shared))
        module["latent_pos"] = list(shared["topology_pos"])
    if fault == "missing_reset_bias":
        second["params"]["gated_recurrent"].pop("reset_bias")
    else:
        second["params"]["gated_recurrent"][fault] = value
    rng = _CountingRng()

    with pytest.raises((RuntimeError, ValueError)):
        recombine_lineage_payloads(
            parent_a_payload=_payload([first]),
            parent_b_payload=_payload([second]),
            rng=rng,
        )

    assert rng.draws == 0


def test_c2_integrator_gated_rejects_before_draw_and_compatible_gated_is_atomic() -> None:
    shared = _identity(30, "shared")
    gated_a = _module(30, gated_record=_gated_record(reset_bias=-1.0))
    gated_b = _module(31, gated_record=_gated_record(reset_bias=2.0))
    for module in (gated_a, gated_b):
        module.update(deepcopy(shared))
        module["latent_pos"] = list(shared["topology_pos"])

    integrator = deepcopy(gated_a)
    integrator["params"].pop("gated_recurrent")
    integrator["params"]["live_operator"] = "bounded_scalar_integrator_v1"
    mismatch_rng = _CountingRng()
    with pytest.raises(RuntimeError, match="INCOMPATIBLE_LIVE_MODULE"):
        recombine_lineage_payloads(
            parent_a_payload=_payload([gated_a]),
            parent_b_payload=_payload([integrator]),
            rng=mismatch_rng,
        )
    assert mismatch_rng.draws == 0

    atomic_rng = _CountingRng(0.9)
    child, summary = recombine_lineage_payloads(
        parent_a_payload=_payload([gated_a]),
        parent_b_payload=_payload([gated_b]),
        rng=atomic_rng,
    )
    assert atomic_rng.draws == summary.budget_draws == 1
    inherited = child["modules"][0]["params"]["gated_recurrent"]
    assert inherited in (
        gated_a["params"]["gated_recurrent"],
        gated_b["params"]["gated_recurrent"],
    )
    assert inherited not in (
        {
            **gated_a["params"]["gated_recurrent"],
            "reset_bias": 0.5,
        },
    )
    assert "structural_live_module_state" not in child
    assert "structural_live_module_outputs" not in child
