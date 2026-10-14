from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from perf.cxxcore18_live_summary_tape_metrics import (
    CONSUMER_MEMORY_ACTION,
    CONSUMER_PREDICTION_DEMAND,
    DECISION,
    TAPE_SCHEMA_VERSION,
    build_live_summary_tape,
    capture_pre_tick_snapshot,
    compare_consumer_rows,
    config_from_sim_cfg,
    coupling_audit_payload,
    result_payload,
    run_consumer_shadow_compare,
    tape_payload,
)
from simulation.memory_action_window import MemoryActionWindowInstrumentation, memory_action_window_schema
from simulation.prediction_demand_action_context import (
    PredictionDemandActionContextInstrumentation,
    prediction_demand_action_context_schema,
)
from simulation.runner_pkg.runner_config import SimConfig


def _runner(tmp_path: Path):
    sim_cfg = SimpleNamespace(
        seed=1818,
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
        prediction_demand_action_context_window_radius=1,
        prediction_demand_action_context_include_lineage=True,
        prediction_demand_action_context_write_events=False,
        prediction_demand_action_context_max_events_per_run=0,
        prediction_demand_action_context_sign_summary_enable=False,
        prediction_demand_action_context_memory_arm="mem",
        prediction_demand_action_context_prediction_arm="pred",
        prediction_demand_action_context_init_arm="init",
        cxxcore_summary_tape_enable=False,
        cxxcore_summary_tape_shadow_compare_enable=False,
        cxxcore_summary_tape_authoritative_metrics_enable=False,
        cxxcore_summary_tape_consumers=(),
        cxxcore_summary_tape_first_divergence_limit=1,
        cxxcore_summary_tape_mode="off",
    )
    env_cfg = SimpleNamespace(patch_world_enable=True, prediction_demand_enable=True)
    env = SimpleNamespace()
    env.grid = np.zeros((5, 5), dtype=np.int8)
    env.agent_pos_y = np.asarray([1, 2, 4], dtype=np.int32)
    env.agent_pos_x = np.asarray([1, 2, 4], dtype=np.int32)
    env.agent_alive = np.asarray([True, True, True], dtype=bool)
    env.patch_delayed_hazard_cue = np.zeros((5, 5), dtype=np.float32)
    env.patch_delayed_hazard_active = np.zeros((5, 5), dtype=np.float32)
    env.patch_delayed_hazard_timer = np.full((5, 5), -1, dtype=np.int32)
    env.patch_delayed_hazard_prior_cue = np.zeros((5, 5), dtype=np.int8)
    env.patch_delayed_hazard_cue[1, 1] = 1.0
    env.patch_delayed_hazard_active[2, 2] = 1.0
    env.patch_delayed_hazard_timer[3, 3] = 1
    env.patch_delayed_hazard_prior_cue[3, 3] = 1
    env.wdl_delayed_hazard_cue_cell_count_last = 1
    env.wdl_delayed_hazard_active_cell_count_last = 1
    env.wdl_delayed_hazard_active_without_prior_cue_count_last = 0

    env.patch_prediction_demand_event_id = np.zeros((5, 5), dtype=np.int64)
    env.patch_prediction_demand_event_id[1:4, 1:4] = 7
    env.patch_prediction_demand_cue = np.zeros((5, 5), dtype=np.float32)
    env.patch_prediction_demand_cue[1:4, 1:4] = 1.0
    env.patch_prediction_demand_active = np.zeros((5, 5), dtype=np.float32)
    env.patch_prediction_demand_trend = np.zeros((5, 5), dtype=np.float32)
    env.patch_prediction_demand_trend[1:4, 1:4] = 0.75
    env.patch_prediction_demand_timer = np.full((5, 5), 3, dtype=np.int32)
    env.patch_prediction_demand_age = np.zeros((5, 5), dtype=np.int32)
    env.patch_prediction_demand_delay = np.full((5, 5), 4, dtype=np.int32)
    env.patch_prediction_demand_prior_cue = np.ones((5, 5), dtype=np.int8)
    env.patch_prediction_demand_transition_sign = np.ones((5, 5), dtype=np.int8)

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
        for aid in range(3)
    }
    pop = SimpleNamespace(agents=agents)
    return SimpleNamespace(run_dir=str(tmp_path), sim_cfg=sim_cfg, env_cfg=env_cfg, env=env, pop=pop)


def _capture_tape(runner):
    alive = [runner.pop.agents[i] for i in range(3)]
    agent_ids = np.asarray([0, 1, 2], dtype=np.int32)
    actions = np.asarray([0, 4, 5], dtype=np.int32)
    snapshot = capture_pre_tick_snapshot(
        runner,
        generation=1,
        tick=2,
        alive=alive,
        agent_ids_np=agent_ids,
        actions_np=actions,
    )
    runner.pop.agents[0].health = 8.5
    runner.env.agent_pos_y[0] = 2
    runner.env.agent_pos_x[0] = 1
    return alive, agent_ids, actions, build_live_summary_tape(runner, snapshot)


def test_cxxcore18_flags_default_off() -> None:
    cfg = config_from_sim_cfg(SimConfig())

    assert cfg.cxxcore_summary_tape_enable is False
    assert cfg.cxxcore_summary_tape_shadow_compare_enable is False
    assert cfg.cxxcore_summary_tape_authoritative_metrics_enable is False
    assert cfg.cxxcore_summary_tape_consumers == ()
    assert cfg.cxxcore_summary_tape_mode == "off"


def test_cxxcore18_tape_schema_and_deterministic_order(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    _, _, _, tape_a = _capture_tape(runner)
    runner2 = _runner(tmp_path / "b")
    _, _, _, tape_b = _capture_tape(runner2)

    assert tape_a.schema_version == TAPE_SCHEMA_VERSION
    assert tape_payload(tape_a)["rows"] == 3
    assert tape_a.agent_ids.tolist() == [0, 1, 2]
    assert tape_a.digest == tape_b.digest


def test_cxxcore18_memory_action_legacy_vs_tape_parity(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    inst = MemoryActionWindowInstrumentation(runner)
    legacy_rows = []
    inst._write_event = lambda row: legacy_rows.append(deepcopy(dict(row)))  # type: ignore[method-assign]
    alive, agent_ids, actions, tape = _capture_tape(runner)

    runner_legacy = _runner(tmp_path / "legacy")
    runner_legacy.run_dir = runner.run_dir
    inst_legacy = MemoryActionWindowInstrumentation(runner_legacy)
    inst_legacy._write_event = lambda row: legacy_rows.append(deepcopy(dict(row)))  # type: ignore[method-assign]
    alive_legacy = [runner_legacy.pop.agents[i] for i in range(3)]
    inst_legacy.record_pre_env(runner_legacy, generation=1, tick=2, alive=alive_legacy, agent_ids_np=agent_ids, actions_np=actions)
    runner_legacy.pop.agents[0].health = 8.5
    runner_legacy.env.agent_pos_y[0] = 2
    runner_legacy.env.agent_pos_x[0] = 1
    inst_legacy.record_post_env(runner_legacy, generation=1, tick=2)

    tape_rows = inst.rows_from_summary_tape(runner, tape)
    parity = compare_consumer_rows(legacy_rows, tape_rows, consumer=CONSUMER_MEMORY_ACTION)

    assert parity["ok"] is True
    assert len(tape_rows) == len(legacy_rows)


def test_cxxcore18_prediction_demand_legacy_vs_tape_parity(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    alive = [runner.pop.agents[i] for i in range(3)]
    agent_ids = np.asarray([0, 1, 2], dtype=np.int32)
    actions = np.asarray([0, 4, 5], dtype=np.int32)
    snapshot = capture_pre_tick_snapshot(
        runner,
        generation=1,
        tick=2,
        alive=alive,
        agent_ids_np=agent_ids,
        actions_np=actions,
    )
    inst = PredictionDemandActionContextInstrumentation(runner)
    inst.record_pre_env(runner, generation=1, tick=2, alive=alive, agent_ids_np=agent_ids, actions_np=actions)
    legacy_rows = list(inst._generation_rows.get(1, []))
    runner.pop.agents[0].health = 8.5
    runner.env.agent_pos_y[0] = 2
    runner.env.agent_pos_x[0] = 1
    tape = build_live_summary_tape(runner, snapshot)

    inst2 = PredictionDemandActionContextInstrumentation(runner)
    tape_rows = inst2.rows_from_summary_tape(runner, tape)
    parity = compare_consumer_rows(legacy_rows, tape_rows, consumer=CONSUMER_PREDICTION_DEMAND)

    assert parity["ok"] is True
    assert len(tape_rows) == len(legacy_rows)


def test_cxxcore18_missing_tape_fallback(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    inst = MemoryActionWindowInstrumentation(runner)
    result = run_consumer_shadow_compare(
        consumer=CONSUMER_MEMORY_ACTION,
        instrumentation=inst,
        runner=runner,
        tape=None,
        legacy_rows=[],
        mode="summary_tape_shadow_compare",
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "missing_tape"


def test_cxxcore18_malformed_tape_fallback(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    inst = MemoryActionWindowInstrumentation(runner)
    result = run_consumer_shadow_compare(
        consumer=CONSUMER_MEMORY_ACTION,
        instrumentation=inst,
        runner=runner,
        tape=object(),
        legacy_rows=[],
        mode="summary_tape_shadow_compare",
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "malformed_tape"


def test_cxxcore18_injected_divergence_first_localization(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    alive, agent_ids, actions, tape = _capture_tape(runner)
    inst = PredictionDemandActionContextInstrumentation(runner)
    inst.record_pre_env(runner, generation=1, tick=2, alive=alive, agent_ids_np=agent_ids, actions_np=actions)
    legacy_rows = list(inst._generation_rows.get(1, []))

    result = run_consumer_shadow_compare(
        consumer=CONSUMER_PREDICTION_DEMAND,
        instrumentation=inst,
        runner=runner,
        tape=tape,
        legacy_rows=legacy_rows,
        mode="summary_tape_shadow_compare",
        inject_divergence=True,
    )

    assert result.status == "divergence"
    assert result.first_divergence is not None
    assert result.first_divergence["consumer"] == CONSUMER_PREDICTION_DEMAND


def test_cxxcore18_authoritative_diagnostics_result_payload(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    alive, agent_ids, actions, tape = _capture_tape(runner)
    inst = MemoryActionWindowInstrumentation(runner)
    legacy_rows = inst.rows_from_summary_tape(runner, tape)
    result = run_consumer_shadow_compare(
        consumer=CONSUMER_MEMORY_ACTION,
        instrumentation=inst,
        runner=runner,
        tape=tape,
        legacy_rows=legacy_rows,
        mode="summary_tape_authoritative_diagnostics",
    )
    payload = result_payload(result)

    assert payload["status"] == "ok"
    assert payload["legacy_row_count"] == payload["tape_row_count"]


def test_cxxcore18_schema_sentinels_unchanged() -> None:
    audit = coupling_audit_payload()

    assert audit["decision"] == DECISION
    assert audit["checkpoint_schema_changed"] is False
    assert audit["metrics_schema_changed"] is False
    assert audit["native_rng_added"] is False
    assert memory_action_window_schema()["schema_version"] == "memory_action_window_v1"
    assert prediction_demand_action_context_schema()["schema_version"] == "prediction_demand_action_context_v1"
