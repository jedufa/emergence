from __future__ import annotations

from pathlib import Path

import numpy as np

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _config(*, enabled: bool, generations: int = 2) -> SimConfig:
    return SimConfig(
        seed=421,
        generations=generations,
        steps_per_generation=3,
        device="cpu",
        bank_device="cpu",
        torch_threads=1,
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=1,
        chronicle_enable=enabled,
        chronicle_profile="notable",
        chronicle_run_id="runner-parity",
    )


def _runner(root: Path, *, enabled: bool, generations: int = 2) -> SimulationRunner:
    return SimulationRunner(
        env_cfg=EnvConfig(width=8, height=8, view_radius=1),
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=_config(enabled=enabled, generations=generations),
        logger=NoOpMetricsLogger(),
        run_dir=str(root),
        perf=False,
    )


def _trajectory_snapshot(runner: SimulationRunner) -> dict:
    alive = []
    for agent in sorted(runner.pop.alive_agents(), key=lambda item: int(item.agent_id)):
        alive.append(
            (
                int(agent.agent_id),
                round(float(agent.energy), 12),
                round(float(agent.health), 12),
                round(float(agent.injury), 12),
                int(getattr(agent.stats, "offspring_count", 0)),
            )
        )
    snapshot = runner.rng_manager.capture_snapshot(generation_index=2)
    return {
        "alive": alive,
        "grid": np.asarray(runner.env.grid).tolist(),
        "agent_grid": np.asarray(runner.env.agent_grid).tolist(),
        "rng": snapshot,
    }


def test_chronicle_on_off_does_not_change_trajectory_or_rng(tmp_path: Path) -> None:
    disabled = _runner(tmp_path / "off", enabled=False)
    disabled.run()
    disabled_snapshot = _trajectory_snapshot(disabled)

    enabled = _runner(tmp_path / "on", enabled=True)
    enabled.run()

    assert disabled_snapshot == _trajectory_snapshot(enabled)
    assert not (tmp_path / "off" / "chronicle").exists()
    assert (tmp_path / "on" / "chronicle" / "formal_events.jsonl").exists()


def test_checkpoint_bundle_restores_chronicle_prefix_and_continues(tmp_path: Path) -> None:
    full = _runner(tmp_path / "full", enabled=True, generations=2)
    full.run()

    first_leg = _runner(tmp_path / "part", enabled=True, generations=1)
    first_leg.run()
    bundle = checkpoint_bundle_v2_save(
        first_leg,
        str(tmp_path / "bundle_gen000001"),
        generation_index=1,
        strict_replay_mode=True,
    )
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    assert "scientific_chronicle_state" in payload["runner_state"]

    resumed = _runner(tmp_path / "resumed", enabled=True, generations=1)
    checkpoint_bundle_v2_apply(resumed, payload, restore_rng=True)
    resumed.run(start_gen=1, skip_init=True)

    full_events = (tmp_path / "full" / "chronicle" / "formal_events.jsonl").read_bytes()
    resumed_events = (tmp_path / "resumed" / "chronicle" / "formal_events.jsonl").read_bytes()
    assert resumed_events == full_events
    assert _trajectory_snapshot(resumed) == _trajectory_snapshot(full)
