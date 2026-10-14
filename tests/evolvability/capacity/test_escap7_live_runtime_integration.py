from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from agents.structural_identity_v1 import new_native_identity
from evolution.structural_gated_recurrent_v1 import (
    GATED_OPERATOR,
    neutral_gated_recurrent_record,
    step_gated_scalar,
)
from evolution.structural_live_modules_v1 import (
    LIVE_PROFILE,
    execute_agent_live_modules,
    execute_live_modules_batch,
)
from simulation.runner_pkg.runner_config import SimConfig


def _identity(index: int) -> dict:
    return new_native_identity(
        context={
            "campaign_id": "escap7-live-test",
            "branch_id": "fixed-width-gated",
            "generation": 1,
            "tick": 0,
            "child_agent_id": int(index),
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap7-live-lineage",
        },
        operator_id="fixture",
        attempt_index=int(index),
        operator_role=f"module-{index}",
    )


def _record(**updates: float) -> dict:
    record = neutral_gated_recurrent_record()
    record.update(updates)
    return record


def _module(
    index: int,
    *,
    expression: float = 1.0,
    record: dict | None = None,
) -> dict:
    identity = _identity(index)
    return {
        "module_id": f"m{index:03d}",
        "module_type": "generic",
        "params": {
            "live_operator": GATED_OPERATOR,
            "expression_scale": float(expression),
            "gated_recurrent": deepcopy(record or _record()),
        },
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **identity,
        "latent_pos": list(identity["topology_pos"]),
    }


def _connection(source: dict, target: dict, *, weight: float = 0.5) -> dict:
    return {
        "connection_id": "c000",
        "src_module_id": str(source["module_id"]),
        "dst_module_id": str(target["module_id"]),
        "src_slot": "out",
        "dst_slot": "in",
        "value_type": "scalar",
        "channel": "typed",
        "weight": float(weight),
    }


def _agent(modules: list[dict], connections: list[dict] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        genome=SimpleNamespace(
            modules=deepcopy(modules),
            connections=deepcopy(connections or []),
        ),
        structural_live_module_state={},
        structural_live_module_outputs={},
        structural_live_module_telemetry={},
        structural_development_latch_by_instance_id={},
        structural_development_stage_id="",
        structural_development_effective_gain_by_instance_id={},
    )


def _cfg(*, gated: bool) -> SimConfig:
    cfg = SimConfig()
    cfg.device = "cpu"
    cfg.bank_device = "cpu"
    cfg.policy_tick_stride = 1
    cfg.structural_live_modules_enable = True
    cfg.structural_live_modules_profile = LIVE_PROFILE
    cfg.structural_gated_recurrent_enable = bool(gated)
    cfg.structural_gated_recurrent_profile = (
        "escap7_fixed_width_gated_recurrent_v1" if gated else "none"
    )
    return cfg


def _plan(module: dict, gain: float) -> SimpleNamespace:
    instance_id = str(module["module_instance_id"])
    return SimpleNamespace(
        effective_gain_by_instance_id={instance_id: float(gain)},
        latch_by_instance_id={},
        stage_id="mature",
        telemetry={"stage_id": "mature", "random_draws": 0},
    )


def test_gated_runtime_exact_trace_state_output_width_cost_and_rng() -> None:
    record = _record(
        reset_bias=-0.5,
        reset_input_gain=1.0,
        reset_state_gain=-0.75,
        update_bias=0.25,
        update_input_gain=-1.25,
        update_state_gain=0.5,
        candidate_bias=-0.125,
        candidate_input_gain=1.5,
        candidate_state_gain=-1.0,
        output_gain=1.25,
    )
    module = _module(1, expression=-0.75, record=record)
    instance_id = str(module["module_instance_id"])
    agent = _agent([module])
    agent.structural_live_module_state = {instance_id: -0.4}
    agent.structural_live_module_outputs = {instance_id: 0.0}
    expected = step_gated_scalar(
        external_input=0.6,
        routed_previous=0.0,
        previous_state=-0.4,
        expression_scale=-0.75,
        developmental_gain=1.0,
        record=record,
    )

    result = execute_agent_live_modules(
        agent,
        fatigue=0.6,
        gated_recurrent_enabled=True,
    )

    assert agent.structural_live_module_state == {instance_id: expected.state_after}
    assert agent.structural_live_module_outputs == {instance_id: expected.effective_output}
    row = result.summary["state_rows"][0]
    assert row["live_operator"] == GATED_OPERATOR
    assert row["gated_input"] == expected.input_aggregate
    assert row["reset_pre"] == expected.reset_pre
    assert row["reset_gate"] == expected.reset_gate
    assert row["update_pre"] == expected.update_pre
    assert row["update_gate"] == expected.update_gate
    assert row["candidate_pre"] == expected.candidate_pre
    assert row["candidate"] == expected.candidate
    assert row["state_after"] == expected.state_after
    assert row["gated_base_output"] == expected.base_output
    assert row["gated_expressed_output"] == expected.expressed_output
    assert row["gated_effective_output"] == expected.effective_output
    assert result.summary["owned_state_width_per_module"] == 1
    assert result.summary["gated_execution_operation_estimate"] == 32
    assert result.summary["reported_compute_cost"] == 32
    assert result.summary["gated_execution_rng_draws"] == 0
    assert result.summary["random_draws"] == 0
    assert not any(
        name in vars(agent)
        for name in (
            "structural_gated_state",
            "structural_gated_recurrent_state",
            "gated_state_array",
        )
    )


def test_gated_operator_rejects_when_feature_disabled_and_batch_reads_flag() -> None:
    module = _module(2, record=_record(output_gain=1.0, candidate_bias=1.0))
    direct = _agent([module])
    with pytest.raises(RuntimeError, match="ESCAP7_DISABLED_OPERATOR"):
        execute_agent_live_modules(
            direct,
            fatigue=0.5,
            gated_recurrent_enabled=False,
        )

    disabled_batch = _agent([module])
    with pytest.raises(RuntimeError, match="ESCAP7_DISABLED_OPERATOR"):
        execute_live_modules_batch(
            [disabled_batch],
            np.asarray([0.5], dtype=np.float32),
            sim_cfg=_cfg(gated=False),
            device="cpu",
        )

    enabled_batch = _agent([module])
    residual = execute_live_modules_batch(
        [enabled_batch],
        np.asarray([0.5], dtype=np.float32),
        sim_cfg=_cfg(gated=True),
        device="cpu",
    )
    assert residual.shape == (1,)
    assert enabled_batch.structural_live_module_telemetry[
        "gated_recurrent_enabled"
    ] is True


def test_neutral_hidden_state_evolves_but_downstream_reads_exact_zero_output() -> None:
    source = _module(3, expression=1.0, record=_record(output_gain=0.0))
    sink = _module(4, expression=1.0, record=_record(output_gain=0.0))
    agent = _agent([source, sink], [_connection(source, sink, weight=1.0)])
    source_id = str(source["module_instance_id"])
    sink_id = str(sink["module_instance_id"])
    agent.structural_live_module_state = {source_id: 0.8, sink_id: -0.4}
    agent.structural_live_module_outputs = {source_id: 0.0, sink_id: 0.0}

    first = execute_agent_live_modules(
        agent,
        fatigue=0.0,
        gated_recurrent_enabled=True,
    )
    second = execute_agent_live_modules(
        agent,
        fatigue=0.0,
        gated_recurrent_enabled=True,
    )

    assert agent.structural_live_module_state[source_id] == pytest.approx(0.2)
    assert agent.structural_live_module_outputs == {source_id: 0.0, sink_id: 0.0}
    rows = {row["module_id"]: row for row in second.summary["state_rows"]}
    assert rows[sink["module_id"]]["typed_input_previous_outputs"] == 0.0
    assert rows[sink["module_id"]]["gated_input"] == 0.0
    assert first.rest_residual == second.rest_residual == 0.0


def test_previous_tick_connection_is_the_only_structural_connection_input() -> None:
    source = _module(
        5,
        record=_record(candidate_bias=1.0, update_bias=4.0, output_gain=1.0),
    )
    sink = _module(
        6,
        record=_record(candidate_input_gain=1.0, update_bias=4.0, output_gain=1.0),
    )
    agent = _agent([source, sink], [_connection(source, sink, weight=0.5)])

    first = execute_agent_live_modules(
        agent,
        fatigue=0.0,
        gated_recurrent_enabled=True,
    )
    first_rows = {row["module_id"]: row for row in first.summary["state_rows"]}
    assert first_rows[sink["module_id"]]["typed_input_previous_outputs"] == 0.0
    source_output = first_rows[source["module_id"]]["typed_output"]
    assert source_output != 0.0

    second = execute_agent_live_modules(
        agent,
        fatigue=0.0,
        gated_recurrent_enabled=True,
    )
    second_rows = {row["module_id"]: row for row in second.summary["state_rows"]}
    assert second_rows[sink["module_id"]][
        "typed_input_previous_outputs"
    ] == pytest.approx(0.5 * source_output)
    assert second_rows[sink["module_id"]]["gated_input"] == pytest.approx(
        0.5 * source_output
    )


def test_development_inactive_freezes_state_zeros_output_and_reactivates_cleanly() -> None:
    record = _record(candidate_bias=1.0, update_bias=1.0, output_gain=1.0)
    module = _module(7, record=record)
    instance_id = str(module["module_instance_id"])
    agent = _agent([module])
    agent.structural_live_module_state = {instance_id: 0.8}
    agent.structural_live_module_outputs = {instance_id: 0.6}

    inactive = execute_agent_live_modules(
        agent,
        fatigue=0.5,
        gated_recurrent_enabled=True,
        developmental_plan=_plan(module, 0.0),
    )
    inactive_row = inactive.summary["state_rows"][0]
    assert agent.structural_live_module_state == {instance_id: 0.8}
    assert agent.structural_live_module_outputs == {instance_id: 0.0}
    assert inactive.rest_residual == 0.0
    assert inactive_row["inactive_state_frozen"] is True
    assert inactive_row["reset_gate"] is None
    assert inactive.summary["gated_execution_rng_draws"] == 0

    expected = step_gated_scalar(
        external_input=0.5,
        routed_previous=0.0,
        previous_state=0.8,
        expression_scale=1.0,
        developmental_gain=1.0,
        record=record,
    )
    active = execute_agent_live_modules(
        agent,
        fatigue=0.5,
        gated_recurrent_enabled=True,
        developmental_plan=_plan(module, 1.0),
    )
    assert agent.structural_live_module_state == {instance_id: expected.state_after}
    assert agent.structural_live_module_outputs == {
        instance_id: expected.effective_output
    }
    assert active.summary["state_rows"][0]["state_before"] == 0.8


def test_expression_connection_and_state_reset_lesions_are_causal_and_rng_free() -> None:
    source = _module(
        8,
        record=_record(candidate_bias=1.0, update_bias=4.0, output_gain=1.0),
    )
    sink = _module(
        9,
        record=_record(candidate_input_gain=1.0, update_bias=4.0, output_gain=1.0),
    )
    connection = _connection(source, sink, weight=1.0)

    expression_agent = _agent([source])
    expression = execute_agent_live_modules(
        expression_agent,
        fatigue=0.0,
        gated_recurrent_enabled=True,
        lesion_expression=True,
    )
    assert expression.rest_residual == 0.0
    assert next(iter(expression_agent.structural_live_module_state.values())) != 0.0

    connected = _agent([source, sink], [connection])
    execute_agent_live_modules(
        connected,
        fatigue=0.0,
        gated_recurrent_enabled=True,
    )
    lesioned = execute_agent_live_modules(
        connected,
        fatigue=0.0,
        gated_recurrent_enabled=True,
        lesion_connection_ids=[connection["connection_id"]],
    )
    sink_row = next(
        row for row in lesioned.summary["state_rows"] if row["module_id"] == sink["module_id"]
    )
    assert sink_row["typed_input_previous_outputs"] == 0.0

    source_id = str(source["module_instance_id"])
    reset_expected = step_gated_scalar(
        external_input=0.0,
        routed_previous=0.0,
        previous_state=0.0,
        expression_scale=1.0,
        developmental_gain=1.0,
        record=source["params"]["gated_recurrent"],
    )
    reset = execute_agent_live_modules(
        connected,
        fatigue=0.0,
        gated_recurrent_enabled=True,
        reset_instance_ids=[source_id],
        lesion_connection_ids=[connection["connection_id"]],
    )
    reset_row = next(
        row for row in reset.summary["state_rows"] if row["module_id"] == source["module_id"]
    )
    assert reset_row["state_before"] == 0.0
    assert reset_row["state_after"] == reset_expected.state_after
    assert reset.summary["random_draws"] == 0


def test_nonfinite_gated_record_rejects_before_runtime_state_commit() -> None:
    module = _module(10)
    instance_id = str(module["module_instance_id"])
    agent = _agent([module])
    agent.structural_live_module_state = {instance_id: 0.25}
    agent.structural_live_module_outputs = {instance_id: 0.125}
    before_state = deepcopy(agent.structural_live_module_state)
    before_outputs = deepcopy(agent.structural_live_module_outputs)
    agent.genome.modules[0]["params"]["gated_recurrent"]["reset_bias"] = float(
        "nan"
    )

    with pytest.raises(RuntimeError, match="ESCAP7_NONFINITE"):
        execute_agent_live_modules(
            agent,
            fatigue=0.5,
            gated_recurrent_enabled=True,
        )

    assert agent.structural_live_module_state == before_state
    assert agent.structural_live_module_outputs == before_outputs


def test_local_slot_repacking_cannot_change_adversarial_output_accumulation() -> None:
    modules = [
        _module(20, record=_record(output_gain=2.0)),
        _module(21, record=_record(output_gain=2.0e-16)),
        _module(22, record=_record(output_gain=2.0)),
    ]
    first = _agent(modules)
    first.structural_live_module_state = {
        str(modules[0]["module_instance_id"]): 1.0,
        str(modules[1]["module_instance_id"]): 1.0,
        str(modules[2]["module_instance_id"]): -1.0,
    }
    first.structural_live_module_outputs = {
        key: 0.0 for key in first.structural_live_module_state
    }

    repacked_modules = deepcopy(modules)
    for module, local_id in zip(repacked_modules, ("m200", "m000", "m100")):
        module["module_id"] = local_id
    repacked = _agent(repacked_modules)
    repacked.structural_live_module_state = deepcopy(first.structural_live_module_state)
    repacked.structural_live_module_outputs = deepcopy(first.structural_live_module_outputs)

    first_step = execute_agent_live_modules(
        first,
        fatigue=0.0,
        gated_recurrent_enabled=True,
    )
    repacked_step = execute_agent_live_modules(
        repacked,
        fatigue=0.0,
        gated_recurrent_enabled=True,
    )

    assert first_step.rest_residual == repacked_step.rest_residual
    assert first.structural_live_module_state == repacked.structural_live_module_state
    assert first.structural_live_module_outputs == repacked.structural_live_module_outputs


def test_connection_id_and_local_slot_repacking_cannot_change_gated_input_sum() -> None:
    sources = [_module(30), _module(31), _module(32)]
    target = _module(
        33,
        record=_record(candidate_input_gain=4.0, output_gain=1.0),
    )
    modules = [*sources, target]
    connections = []
    for index, source in enumerate(sources):
        connection = _connection(source, target, weight=1.0)
        connection["connection_id"] = f"c{index:03d}"
        connections.append(connection)
    previous_by_instance = {
        str(sources[0]["module_instance_id"]): 1.0,
        str(sources[1]["module_instance_id"]): 1.0e-16,
        str(sources[2]["module_instance_id"]): -1.0,
        str(target["module_instance_id"]): 0.0,
    }

    first = _agent(modules, connections)
    first.structural_live_module_state = {
        str(module["module_instance_id"]): 0.0 for module in modules
    }
    first.structural_live_module_outputs = deepcopy(previous_by_instance)

    repacked_modules = deepcopy(modules)
    local_ids = ("m300", "m100", "m200", "m000")
    local_by_instance: dict[str, str] = {}
    for module, local_id in zip(repacked_modules, local_ids):
        module["module_id"] = local_id
        local_by_instance[str(module["module_instance_id"])] = local_id
    repacked_connections = deepcopy(connections)
    for index, connection in enumerate(repacked_connections):
        source_instance = str(sources[index]["module_instance_id"])
        connection["src_module_id"] = local_by_instance[source_instance]
        connection["dst_module_id"] = local_by_instance[str(target["module_instance_id"])]
        connection["connection_id"] = f"c{2 - index:03d}"
    repacked = _agent(repacked_modules, repacked_connections)
    repacked.structural_live_module_state = deepcopy(first.structural_live_module_state)
    repacked.structural_live_module_outputs = deepcopy(previous_by_instance)

    first_step = execute_agent_live_modules(
        first,
        fatigue=0.0,
        gated_recurrent_enabled=True,
    )
    repacked_step = execute_agent_live_modules(
        repacked,
        fatigue=0.0,
        gated_recurrent_enabled=True,
    )
    target_instance = str(target["module_instance_id"])
    first_row = next(
        row for row in first_step.summary["state_rows"]
        if row["module_instance_id"] == target_instance
    )
    repacked_row = next(
        row for row in repacked_step.summary["state_rows"]
        if row["module_instance_id"] == target_instance
    )

    assert first_row["gated_input"] == repacked_row["gated_input"]
    assert first_row["state_after"] == repacked_row["state_after"]
    assert first_row["typed_output"] == repacked_row["typed_output"]
