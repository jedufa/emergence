from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from perf.cxxcore18_live_summary_tape_metrics import (
    CONSUMER_MEMORY_ACTION,
    CONSUMER_PREDICTION_DEMAND,
    build_live_summary_tape,
    capture_pre_tick_snapshot,
)
from perf.cxxcore19_native_summary_tape_resident import (
    DEFAULT_DECISION,
    INPUT_SCHEMA_VERSION,
    PHASE_TO_CODE,
    config_from_sim_cfg,
    native_contract_audit_payload,
    native_result_payload,
    prepare_native_inputs_from_tape,
    result_payload,
    rows_from_native_result,
    run_native_consumer_shadow_compare,
    run_native_summary_tape_python_reference,
)
from simulation.memory_action_window import MemoryActionWindowInstrumentation
from simulation.prediction_demand_action_context import PredictionDemandActionContextInstrumentation
from simulation.runner_pkg.runner_config import SimConfig


def _runner(tmp_path: Path, *, n: int = 4):
    sim_cfg = SimpleNamespace(
        seed=1919,
        memory_action_window_instrumentation_enable=True,
        memory_action_window_emit_jsonl=True,
        memory_action_window_include_lineage=True,
        memory_action_window_include_memory_state=True,
        memory_action_window_max_events_per_run=0,
        memory_action_window_schema_version="memory_action_window_v1",
        memory_action_window_trait_arm="trait",
        memory_action_window_world_context="world",
        memory_action_window_init_arm="init",
        prediction_demand_action_context_enable=True,
        prediction_demand_action_context_schema_version="prediction_demand_action_context_v1",
        prediction_demand_action_context_window_radius=2,
        prediction_demand_action_context_include_lineage=True,
        prediction_demand_action_context_write_events=False,
        prediction_demand_action_context_max_events_per_run=0,
        prediction_demand_action_context_sign_summary_enable=False,
        prediction_demand_action_context_memory_arm="mem",
        prediction_demand_action_context_prediction_arm="pred",
        prediction_demand_action_context_init_arm="init",
        cxxcore19_native_summary_tape_enable=False,
        cxxcore19_native_summary_tape_mode="off",
        cxxcore19_native_summary_tape_compare_enable=True,
        cxxcore19_native_summary_tape_fail_closed=False,
        cxxcore19_native_summary_tape_first_divergence_limit=1,
    )
    env_cfg = SimpleNamespace(patch_world_enable=True, prediction_demand_enable=True)
    env = SimpleNamespace()
    env.grid = np.zeros((6, 6), dtype=np.int8)
    env.agent_pos_y = np.asarray([(i % 4) + 1 for i in range(n)], dtype=np.int32)
    env.agent_pos_x = np.asarray([((i * 2) % 4) + 1 for i in range(n)], dtype=np.int32)
    env.agent_alive = np.ones((n,), dtype=bool)
    env.patch_delayed_hazard_cue = np.zeros((6, 6), dtype=np.float32)
    env.patch_delayed_hazard_active = np.zeros((6, 6), dtype=np.float32)
    env.patch_delayed_hazard_timer = np.full((6, 6), -1, dtype=np.int32)
    env.patch_delayed_hazard_prior_cue = np.zeros((6, 6), dtype=np.int8)
    env.patch_delayed_hazard_cue[1:3, 1:3] = 1.0
    env.patch_delayed_hazard_active[3:5, 3:5] = 1.0
    env.patch_delayed_hazard_timer[5, 5] = 1
    env.patch_delayed_hazard_prior_cue[5, 5] = 1
    env.wdl_delayed_hazard_cue_cell_count_last = 4
    env.wdl_delayed_hazard_active_cell_count_last = 4
    env.wdl_delayed_hazard_active_without_prior_cue_count_last = 0
    env.patch_prediction_demand_event_id = np.zeros((6, 6), dtype=np.int64)
    env.patch_prediction_demand_event_id[1:5, 1:5] = 9
    env.patch_prediction_demand_cue = np.zeros((6, 6), dtype=np.float32)
    env.patch_prediction_demand_cue[1:5, 1:5] = 1.0
    env.patch_prediction_demand_active = np.zeros((6, 6), dtype=np.float32)
    env.patch_prediction_demand_trend = np.zeros((6, 6), dtype=np.float32)
    env.patch_prediction_demand_trend[1:5, 1:5] = 0.5
    env.patch_prediction_demand_timer = np.full((6, 6), 3, dtype=np.int32)
    env.patch_prediction_demand_age = np.zeros((6, 6), dtype=np.int32)
    env.patch_prediction_demand_delay = np.full((6, 6), 4, dtype=np.int32)
    env.patch_prediction_demand_prior_cue = np.ones((6, 6), dtype=np.int8)
    env.patch_prediction_demand_transition_sign = np.ones((6, 6), dtype=np.int8)
    agents = {
        aid: SimpleNamespace(
            agent_id=aid,
            energy=100.0,
            health=10.0,
            injury=0.0,
            scar=0.0,
            alive=True,
            lineage_id=f"L{aid % 2}",
            founder_id=f"F{aid % 2}",
            slot=aid,
            genome=None,
        )
        for aid in range(n)
    }
    return SimpleNamespace(run_dir=str(tmp_path), sim_cfg=sim_cfg, env_cfg=env_cfg, env=env, pop=SimpleNamespace(agents=agents))


def _capture_tape(runner, *, n: int = 4):
    alive = [runner.pop.agents[i] for i in range(n)]
    agent_ids = np.arange(n, dtype=np.int32)
    actions = np.asarray([0, 1, 4, 5][:n], dtype=np.int32)
    snapshot = capture_pre_tick_snapshot(
        runner,
        generation=1,
        tick=2,
        alive=alive,
        agent_ids_np=agent_ids,
        actions_np=actions,
    )
    runner.pop.agents[0].health = 8.5
    runner.env.agent_pos_y[0] = min(5, int(runner.env.agent_pos_y[0]) + 1)
    return build_live_summary_tape(runner, snapshot)


def _python_rows(runner, tape):
    memory = MemoryActionWindowInstrumentation(runner)
    pred = PredictionDemandActionContextInstrumentation(runner)
    return {
        CONSUMER_MEMORY_ACTION: memory.rows_from_summary_tape(runner, tape),
        CONSUMER_PREDICTION_DEMAND: pred.rows_from_summary_tape(runner, tape),
    }, memory, pred


def test_cxxcore19_flags_default_off() -> None:
    cfg = config_from_sim_cfg(SimConfig())

    assert cfg.cxxcore19_native_summary_tape_enable is False
    assert cfg.cxxcore19_native_summary_tape_mode == "off"
    assert cfg.cxxcore19_native_summary_tape_compare_enable is True
    assert cfg.cxxcore19_native_summary_tape_fail_closed is False


def test_cxxcore19_input_contract_and_phase_codes(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tape = _capture_tape(runner)
    inputs = prepare_native_inputs_from_tape(tape)

    assert inputs.schema_version == INPUT_SCHEMA_VERSION
    assert inputs.agent_ids.tolist() == [0, 1, 2, 3]
    assert inputs.phase_codes.dtype == np.int32
    assert set(inputs.phase_codes.tolist()).issubset(set(PHASE_TO_CODE.values()))
    assert inputs.digest


def test_cxxcore19_python_reference_rows_match_cxxcore18_rows(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tape = _capture_tape(runner)
    inputs = prepare_native_inputs_from_tape(tape)
    expected_rows, memory, pred = _python_rows(runner, tape)
    native = run_native_summary_tape_python_reference(inputs)
    actual_rows = rows_from_native_result(inputs=inputs, result=native, runner=runner, memory_instrumentation=memory, prediction_instrumentation=pred)

    assert expected_rows == actual_rows
    assert native_result_payload(native)["summary_counts"][7] == 4


def test_cxxcore19_shadow_compare_with_python_backend_passes(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tape = _capture_tape(runner)
    inputs = prepare_native_inputs_from_tape(tape)
    expected_rows, memory, pred = _python_rows(runner, tape)

    result = run_native_consumer_shadow_compare(
        inputs=inputs,
        runner=runner,
        memory_instrumentation=memory,
        prediction_instrumentation=pred,
        python_rows=expected_rows,
        native_backend=lambda native_inputs: run_native_summary_tape_python_reference(native_inputs),
        mode="native_summary_tape_shadow_compare",
    )
    payload = result_payload(result)

    assert payload["status"] == "ok"
    assert payload["parity"]["ok"] is True
    assert payload["fallback_reason"] is None


def test_cxxcore19_missing_native_fallback(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tape = _capture_tape(runner)
    inputs = prepare_native_inputs_from_tape(tape)
    expected_rows, memory, pred = _python_rows(runner, tape)

    result = run_native_consumer_shadow_compare(
        inputs=inputs,
        runner=runner,
        memory_instrumentation=memory,
        prediction_instrumentation=pred,
        python_rows=expected_rows,
        native_backend=None,
        mode="native_summary_tape_shadow_compare",
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "native_unavailable"


def test_cxxcore19_native_exception_fallback(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tape = _capture_tape(runner)
    inputs = prepare_native_inputs_from_tape(tape)
    expected_rows, memory, pred = _python_rows(runner, tape)

    def _raising(_inputs):
        raise RuntimeError("injected")

    result = run_native_consumer_shadow_compare(
        inputs=inputs,
        runner=runner,
        memory_instrumentation=memory,
        prediction_instrumentation=pred,
        python_rows=expected_rows,
        native_backend=_raising,
        mode="native_summary_tape_shadow_compare",
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "native_exception"


def test_cxxcore19_injected_divergence_localizes(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tape = _capture_tape(runner)
    inputs = prepare_native_inputs_from_tape(tape)
    expected_rows, memory, pred = _python_rows(runner, tape)

    result = run_native_consumer_shadow_compare(
        inputs=inputs,
        runner=runner,
        memory_instrumentation=memory,
        prediction_instrumentation=pred,
        python_rows=expected_rows,
        native_backend=lambda native_inputs: run_native_summary_tape_python_reference(native_inputs),
        mode="native_summary_tape_shadow_compare",
        inject_divergence=True,
    )

    assert result.status == "divergence"
    assert result.first_divergence is not None
    assert result.first_divergence["consumer"] in {CONSUMER_MEMORY_ACTION, CONSUMER_PREDICTION_DEMAND}


def test_cxxcore19_schema_sentinels_unchanged() -> None:
    audit = native_contract_audit_payload()

    assert DEFAULT_DECISION == "CXXCORE19-INCONCLUSIVE"
    assert audit["checkpoint_schema_changed"] is False
    assert audit["metrics_schema_changed"] is False
    assert audit["native_rng_added"] is False
    assert audit["authoritative_state_mutation"] is False
