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
from perf.cxxcore20_compact_summary_consumers import (
    DEFAULT_DECISION,
    INPUT_SCHEMA_VERSION,
    compare_compact_to_rows,
    compact_consumer_boundary_audit_payload,
    compact_diagnostic_summary,
    config_from_sim_cfg,
    prepare_compact_inputs_from_tape,
    run_compact_shadow_compare,
    run_python_compact_reference,
)
from simulation.memory_action_window import MemoryActionWindowInstrumentation
from simulation.prediction_demand_action_context import PredictionDemandActionContextInstrumentation
from simulation.runner_pkg.runner_config import SimConfig


def _runner(tmp_path: Path, *, n: int = 6):
    sim_cfg = SimpleNamespace(
        seed=2020,
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
        cxxcore20_compact_summary_enable=False,
        cxxcore20_compact_summary_mode="off",
        cxxcore20_compact_summary_compare_enable=True,
        cxxcore20_compact_summary_fail_closed=False,
    )
    env_cfg = SimpleNamespace(patch_world_enable=True, prediction_demand_enable=True)
    env = SimpleNamespace()
    env.grid = np.zeros((7, 7), dtype=np.int8)
    env.agent_pos_y = np.asarray([(i % 5) + 1 for i in range(n)], dtype=np.int32)
    env.agent_pos_x = np.asarray([((i * 2) % 5) + 1 for i in range(n)], dtype=np.int32)
    env.agent_alive = np.ones((n,), dtype=bool)
    env.patch_delayed_hazard_cue = np.zeros((7, 7), dtype=np.float32)
    env.patch_delayed_hazard_active = np.zeros((7, 7), dtype=np.float32)
    env.patch_delayed_hazard_timer = np.full((7, 7), -1, dtype=np.int32)
    env.patch_delayed_hazard_prior_cue = np.zeros((7, 7), dtype=np.int8)
    env.patch_delayed_hazard_cue[1:3, 1:3] = 1.0
    env.patch_delayed_hazard_active[3:5, 3:5] = 1.0
    env.patch_delayed_hazard_timer[5, 5] = 1
    env.patch_delayed_hazard_prior_cue[5, 5] = 1
    env.wdl_delayed_hazard_cue_cell_count_last = 4
    env.wdl_delayed_hazard_active_cell_count_last = 4
    env.wdl_delayed_hazard_active_without_prior_cue_count_last = 0
    env.patch_prediction_demand_event_id = np.zeros((7, 7), dtype=np.int64)
    env.patch_prediction_demand_event_id[1:6, 1:6] = 12
    env.patch_prediction_demand_cue = np.zeros((7, 7), dtype=np.float32)
    env.patch_prediction_demand_cue[1:6, 1:6] = 1.0
    env.patch_prediction_demand_active = np.zeros((7, 7), dtype=np.float32)
    env.patch_prediction_demand_trend = np.zeros((7, 7), dtype=np.float32)
    env.patch_prediction_demand_trend[1:6, 1:6] = 0.5
    env.patch_prediction_demand_timer = np.full((7, 7), 3, dtype=np.int32)
    env.patch_prediction_demand_age = np.zeros((7, 7), dtype=np.int32)
    env.patch_prediction_demand_delay = np.full((7, 7), 4, dtype=np.int32)
    env.patch_prediction_demand_prior_cue = np.ones((7, 7), dtype=np.int8)
    env.patch_prediction_demand_transition_sign = np.ones((7, 7), dtype=np.int8)
    agents = {
        aid: SimpleNamespace(
            agent_id=aid,
            energy=100.0,
            health=10.0,
            injury=0.0,
            scar=0.0,
            alive=True,
            lineage_id=f"L{aid % 3}",
            founder_id=f"F{aid % 2}",
            slot=aid,
            genome=None,
        )
        for aid in range(n)
    }
    return SimpleNamespace(run_dir=str(tmp_path), sim_cfg=sim_cfg, env_cfg=env_cfg, env=env, pop=SimpleNamespace(agents=agents))


def _capture_tape(runner, *, n: int = 6):
    alive = [runner.pop.agents[i] for i in range(n)]
    agent_ids = np.arange(n, dtype=np.int32)
    actions = np.asarray([0, 1, 2, 3, 4, 5][:n], dtype=np.int32)
    snapshot = capture_pre_tick_snapshot(
        runner,
        generation=1,
        tick=2,
        alive=alive,
        agent_ids_np=agent_ids,
        actions_np=actions,
    )
    runner.pop.agents[0].health = 8.5
    runner.pop.agents[4].health = 9.0
    runner.env.agent_pos_y[0] = min(6, int(runner.env.agent_pos_y[0]) + 1)
    return build_live_summary_tape(runner, snapshot)


def _rows(runner, tape):
    memory = MemoryActionWindowInstrumentation(runner)
    pred = PredictionDemandActionContextInstrumentation(runner)
    return {
        CONSUMER_MEMORY_ACTION: memory.rows_from_summary_tape(runner, tape),
        CONSUMER_PREDICTION_DEMAND: pred.rows_from_summary_tape(runner, tape),
    }


def test_cxxcore20_flags_default_off() -> None:
    cfg = config_from_sim_cfg(SimConfig())

    assert cfg.cxxcore20_compact_summary_enable is False
    assert cfg.cxxcore20_compact_summary_mode == "off"
    assert cfg.cxxcore20_compact_summary_compare_enable is True
    assert cfg.cxxcore20_compact_summary_fail_closed is False


def test_cxxcore20_input_contract(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tape = _capture_tape(runner)
    inputs = prepare_compact_inputs_from_tape(tape, grid_shape=runner.env.grid.shape)

    assert inputs.schema_version == INPUT_SCHEMA_VERSION
    assert inputs.agent_ids.tolist() == [0, 1, 2, 3, 4, 5]
    assert inputs.has_prediction.sum() == 6
    assert inputs.lineage_labels == ("L0", "L1", "L2")
    assert inputs.digest


def test_cxxcore20_python_compact_reference_matches_rows(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tape = _capture_tape(runner)
    rows = _rows(runner, tape)
    inputs = prepare_compact_inputs_from_tape(tape, grid_shape=runner.env.grid.shape)
    compact = run_python_compact_reference(inputs)
    parity = compare_compact_to_rows(
        compact,
        memory_rows=rows[CONSUMER_MEMORY_ACTION],
        prediction_rows=rows[CONSUMER_PREDICTION_DEMAND],
    )

    assert parity["ok"] is True
    assert compact_diagnostic_summary(compact)["prediction_groups"]


def test_cxxcore20_shadow_compare_with_python_backend_passes(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tape = _capture_tape(runner)
    inputs = prepare_compact_inputs_from_tape(tape, grid_shape=runner.env.grid.shape)

    result = run_compact_shadow_compare(
        inputs=inputs,
        native_backend=lambda native_inputs: run_python_compact_reference(native_inputs),
        mode="native_compact_shadow_compare",
    )

    assert result.status == "ok"
    assert result.parity["ok"] is True


def test_cxxcore20_missing_native_fallback(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tape = _capture_tape(runner)
    inputs = prepare_compact_inputs_from_tape(tape, grid_shape=runner.env.grid.shape)

    result = run_compact_shadow_compare(inputs=inputs, native_backend=None, mode="native_compact_shadow_compare")

    assert result.status == "fallback"
    assert result.fallback_reason == "native_unavailable"


def test_cxxcore20_native_exception_fallback(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tape = _capture_tape(runner)
    inputs = prepare_compact_inputs_from_tape(tape, grid_shape=runner.env.grid.shape)

    def _raising(_inputs):
        raise RuntimeError("injected")

    result = run_compact_shadow_compare(inputs=inputs, native_backend=_raising, mode="native_compact_shadow_compare")

    assert result.status == "fallback"
    assert result.fallback_reason == "native_exception"


def test_cxxcore20_injected_divergence_localizes(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tape = _capture_tape(runner)
    inputs = prepare_compact_inputs_from_tape(tape, grid_shape=runner.env.grid.shape)

    result = run_compact_shadow_compare(
        inputs=inputs,
        native_backend=lambda native_inputs: run_python_compact_reference(native_inputs),
        mode="native_compact_shadow_compare",
        inject_divergence=True,
    )

    assert result.status == "divergence"
    assert result.first_divergence is not None
    assert "memory_agent_count" in result.first_divergence["field"]


def test_cxxcore20_schema_sentinels_unchanged() -> None:
    audit = compact_consumer_boundary_audit_payload()

    assert DEFAULT_DECISION == "CXXCORE20-INCONCLUSIVE"
    assert audit["checkpoint_schema_changed"] is False
    assert audit["metrics_schema_changed"] is False
    assert audit["native_rng_added"] is False
    assert audit["authoritative_state_mutation"] is False
