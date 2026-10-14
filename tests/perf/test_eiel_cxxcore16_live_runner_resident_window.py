from __future__ import annotations

import numpy as np

from env.gridworld_pkg.world import GridWorld
from perf.cxxcore15_resident_soa_tick_island import ResidentSoAConfig, as_resident_state, run_resident_python_reference
from perf.cxxcore16_live_runner_resident_window import (
    Cxxcore16ResidentWindowConfig,
    capture_restore_token,
    classify_eligibility,
    config_from_sim_cfg,
    live_runner_resident_window_audit,
    prepare_live_window_inputs,
    restore_from_token,
    result_payload,
    run_live_resident_window,
)
from simulation.rng_manager import RNGManager
from simulation.runner_pkg.runner_config import SimConfig
from tools.scripts.eiel_cxxcore4_cost_layout_backend_probe import _agent, _charged_cfg, _env_cfg


def _fixture(*, n: int = 8, width: int = 8, height: int = 8, ticks: int = 5, seed: int = 16001):
    rm = RNGManager(global_seed=seed)
    env_cfg = _env_cfg(width, height)
    env = GridWorld(env_cfg, seed=seed, rng=rm.env_np)
    agents = [_agent(aid) for aid in range(n)]
    for aid in range(n):
        env.add_agent(aid, pos=(int(aid // width), int(aid % width)))
        agents[aid].energy = 120.0
        agents[aid].health = 10.0
    env.grid[:, :] = 0
    ids = np.arange(n, dtype=np.int32)
    actions = np.empty((ticks, n), dtype=np.int32)
    for tick in range(ticks):
        actions[tick, :] = ((ids * 3 + tick) % 7).astype(np.int32)
    if ticks > 0:
        gather0 = ids[actions[0] == 4]
        env.grid[env.agent_pos_y[gather0], env.agent_pos_x[gather0]] = 1
    return env, env_cfg, _charged_cfg(), agents, actions


def _config(mode: str, **kwargs):
    return Cxxcore16ResidentWindowConfig(
        cxxcore16_resident_window_enable=True,
        cxxcore16_resident_window_authoritative_enable=True,
        cxxcore16_resident_window_mode=mode,
        **kwargs,
    )


def _native_reference(inputs):
    return run_resident_python_reference(as_resident_state(inputs), config=ResidentSoAConfig(sparse_interval=2))


def test_cxxcore16_flags_off_default() -> None:
    cfg = config_from_sim_cfg(SimConfig())
    env, env_cfg, sim_cfg, agents, actions = _fixture()

    result = run_live_resident_window(
        env=env,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        alive=agents,
        actions_by_tick=actions,
        config=Cxxcore16ResidentWindowConfig(),
        native_backend=_native_reference,
    )

    assert cfg.cxxcore16_resident_window_enable is False
    assert result.status == "disabled"
    assert result.fallback_reason == "resident_window_disabled"


def test_cxxcore16_audit_declares_live_runner_boundary_too_wide() -> None:
    audit = live_runner_resident_window_audit()

    assert audit["boundary_decision"] == "CXXCORE16-B-LIVE-RUNNER-BOUNDARY-TOO-WIDE"
    assert audit["checkpoint_schema_changed"] is False
    assert audit["metrics_schema_changed"] is False
    assert audit["native_rng_added"] is False
    assert "internal drives update after env tick" in audit["metrics_diagnostics_dependencies"]


def test_cxxcore16_eligible_live_window_materializes_final_state() -> None:
    env, env_cfg, sim_cfg, agents, actions = _fixture(seed=16002)
    before = capture_restore_token(env, agents).digest

    result = run_live_resident_window(
        env=env,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        alive=agents,
        actions_by_tick=actions,
        config=_config("resident_window_authoritative_no_rng_guarded"),
        native_backend=_native_reference,
    )

    assert result.status == "authoritative_parity"
    assert result.python_env_step_skipped is True
    assert result.parity is not None and result.parity["ok"]
    assert capture_restore_token(env, agents).digest != before
    assert result_payload(result)["diagnostics"]["checkpoint_schema_changed"] is False


def test_cxxcore16_shadow_compare_does_not_materialize_live_state() -> None:
    env, env_cfg, sim_cfg, agents, actions = _fixture(seed=16003)
    before = capture_restore_token(env, agents).digest

    result = run_live_resident_window(
        env=env,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        alive=agents,
        actions_by_tick=actions,
        config=_config("resident_window_shadow_compare"),
        native_backend=_native_reference,
    )

    assert result.status == "shadow_parity"
    assert result.python_env_step_skipped is False
    assert capture_restore_token(env, agents).digest == before


def test_cxxcore16_no_compare_mode_skips_compare_and_materializes() -> None:
    env, env_cfg, sim_cfg, agents, actions = _fixture(seed=16004)

    result = run_live_resident_window(
        env=env,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        alive=agents,
        actions_by_tick=actions,
        config=_config("resident_window_authoritative_no_rng_no_compare", cxxcore16_resident_window_compare_enable=False),
        native_backend=_native_reference,
    )

    assert result.status == "authoritative_parity"
    assert result.parity is not None and result.parity["comparison_skipped"] is True
    assert result.python_env_step_skipped is True


def test_cxxcore16_unsupported_checkpoint_metrics_and_native_missing_fallbacks() -> None:
    env, env_cfg, sim_cfg, agents, actions = _fixture(seed=16005)

    ck = run_live_resident_window(
        env=env,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        alive=agents,
        actions_by_tick=actions,
        config=_config("resident_window_authoritative_no_rng_guarded"),
        native_backend=_native_reference,
        checkpoint_boundary=True,
    )
    metrics = run_live_resident_window(
        env=env,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        alive=agents,
        actions_by_tick=actions,
        config=_config("resident_window_authoritative_no_rng_guarded"),
        native_backend=_native_reference,
        metrics_require_per_tick_python=True,
    )
    native_missing = run_live_resident_window(
        env=env,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        alive=agents,
        actions_by_tick=actions,
        config=_config("resident_window_authoritative_no_rng_guarded"),
        native_backend=None,
    )

    assert ck.fallback_reason == "unsupported_checkpoint_boundary"
    assert metrics.fallback_reason == "unsupported_metrics_dependency"
    assert native_missing.fallback_reason == "native_missing"


def test_cxxcore16_malformed_actions_fallback_before_mutation() -> None:
    env, env_cfg, sim_cfg, agents, actions = _fixture(seed=16006)
    before = capture_restore_token(env, agents).digest

    result = run_live_resident_window(
        env=env,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        alive=agents,
        actions_by_tick=actions.reshape(-1),
        config=_config("resident_window_authoritative_no_rng_guarded"),
        native_backend=_native_reference,
    )

    assert result.fallback_reason == "malformed_actions_or_costs"
    assert capture_restore_token(env, agents).digest == before


def test_cxxcore16_injected_divergence_restores_and_localizes() -> None:
    env, env_cfg, sim_cfg, agents, actions = _fixture(seed=16007)
    before = capture_restore_token(env, agents).digest

    result = run_live_resident_window(
        env=env,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        alive=agents,
        actions_by_tick=actions,
        config=_config("resident_window_authoritative_no_rng_guarded"),
        native_backend=_native_reference,
        inject_divergence=True,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "divergence_detected"
    assert result.first_divergence is not None
    assert result.first_divergence["field"] == "new_energy"
    assert capture_restore_token(env, agents).digest == before


def test_cxxcore16_native_exception_restores_when_not_fail_closed() -> None:
    env, env_cfg, sim_cfg, agents, actions = _fixture(seed=16008)
    before = capture_restore_token(env, agents).digest

    result = run_live_resident_window(
        env=env,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        alive=agents,
        actions_by_tick=actions,
        config=_config("resident_window_authoritative_no_rng_guarded", cxxcore16_resident_window_fail_closed=False),
        native_backend=_native_reference,
        raise_exception=True,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "native_exception"
    assert result.python_state_restored is True
    assert capture_restore_token(env, agents).digest == before


def test_cxxcore16_prepare_live_window_inputs_uses_live_gridworld_state() -> None:
    env, env_cfg, sim_cfg, agents, actions = _fixture(seed=16009)
    inputs, _, diag = prepare_live_window_inputs(
        env=env,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        alive=agents,
        actions_by_tick=actions,
    )
    eligibility = classify_eligibility(
        env=env,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        actions_by_tick=actions,
        native_backend=_native_reference,
    )

    assert inputs.backend.runner.state.actions_by_tick.shape == actions.shape
    assert diag["agents"] == len(agents)
    assert eligibility["classification"] == "eligible_no_rng_resident_window"
