from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig
from simulation.tick_native_sampling import TickNativeSamplingError


def _make_sim(run_dir: Path, *, interval: int | None) -> SimulationRunner:
    config = SimConfig(
        generations=2,
        steps_per_generation=160,
        seed=7123,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        science_sample_interval_ticks=interval,
    )
    config.policy_deterministic_actions = True
    config.cuda_architecture_b_production_enable = False
    return SimulationRunner(
        env_cfg=EnvConfig(width=10, height=10),
        pop_cfg=PopConfig(population_size=8),
        sim_cfg=config,
        logger=NoOpMetricsLogger(),
        run_dir=str(run_dir),
        perf=False,
    )


def _terminal_fingerprint(sim: SimulationRunner) -> str:
    digest = hashlib.sha256()
    for name in ("grid", "agent_grid", "agent_alive", "agent_pos_x", "agent_pos_y"):
        value = np.ascontiguousarray(np.asarray(getattr(sim.env, name)))
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    for agent_id, agent in sorted(sim.pop.agents.items()):
        digest.update(
            json.dumps(
                {
                    "agent_id": int(agent_id),
                    "alive": bool(agent.alive),
                    "energy": float(agent.energy),
                    "health": float(agent.health),
                    "injury": float(agent.injury),
                    "genome": str(agent.genome.canonical_hash()),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
    digest.update(json.dumps(sim.rng_manager.capture_snapshot(), sort_keys=True).encode("ascii"))
    return digest.hexdigest()


def _rows(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "metrics" / "tick_native_science_samples.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_tn2_scientific_sampling_cadences_preserve_real_cpu_trajectory(tmp_path: Path) -> None:
    baseline = _make_sim(tmp_path / "off", interval=None)
    baseline.run(start_gen=0, skip_init=False)
    expected = _terminal_fingerprint(baseline)
    assert _rows(tmp_path / "off") == []

    for interval, expected_count in ((10, 32), (80, 4), (160, 2), (320, 1)):
        run_dir = tmp_path / f"sci{interval}"
        sampled = _make_sim(run_dir, interval=interval)
        sampled.run(start_gen=0, skip_init=False)

        assert _terminal_fingerprint(sampled) == expected
        rows = _rows(run_dir)
        assert len(rows) == expected_count
        assert [int(row["next_tick"]) for row in rows] == list(
            range(interval, 321, interval)
        )
        assert [int(row["absolute_tick"]) for row in rows] == [
            int(row["next_tick"]) - 1 for row in rows
        ]
        assert all(int(row["total_ticks"]) == 320 for row in rows)


@pytest.mark.parametrize("interval", (-1, True, "10"))
def test_tn2_rejects_invalid_sampling_intervals(tmp_path: Path, interval: object) -> None:
    sim = _make_sim(tmp_path / "invalid", interval=interval)  # type: ignore[arg-type]
    with pytest.raises(TickNativeSamplingError, match="science_sample_interval_ticks"):
        sim.run(start_gen=0, skip_init=False)


def test_tn2_rejects_array_authority_until_that_path_is_qualified(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "array", interval=10)
    sim.sim_cfg.runnercore1_array_tick_authority_enable = True

    with pytest.raises(RuntimeError, match="not qualified for an array/native"):
        sim.run(start_gen=0, skip_init=False)
