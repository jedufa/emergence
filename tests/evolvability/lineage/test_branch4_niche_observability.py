from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.branch_observability import NICHE_LIFECYCLE_FILENAME
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tiny_sim(
    run_dir: Path,
    *,
    seed: int = 123,
    branch_enabled: bool = True,
    niche_enabled: bool = False,
    patch_world: bool = True,
    generations: int = 2,
) -> SimulationRunner:
    torch.manual_seed(int(seed))
    env_cfg = EnvConfig(width=8, height=8)
    env_cfg.patch_world_enable = bool(patch_world)
    env_cfg.patch_world_observation_enable = bool(patch_world)
    env_cfg.reproduce_requires_mate = False
    env_cfg.reproduce_try_prob = 1.0
    env_cfg.reproduce_cooldown_steps = 0
    env_cfg.reproduce_parent_energy_share = 0.10
    pop_cfg = PopConfig(population_size=8, parent_fraction=0.5, cull_fraction=0.0, slot_factor=6)
    sim_cfg = SimConfig(
        seed=int(seed),
        generations=int(generations),
        steps_per_generation=4,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
    )
    sim_cfg.branch_observability_enable = bool(branch_enabled)
    sim_cfg.branch_niche_observability_enable = bool(niche_enabled)
    sim_cfg.branch_observability_run_id = "branch4_test_run"
    return SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(run_dir), perf=False)


def _fingerprint(sim: SimulationRunner) -> str:
    payload = []
    for agent in sorted(sim.pop.agents.values(), key=lambda item: int(item.agent_id)):
        st = agent.stats
        payload.append(
            {
                "agent_id": int(agent.agent_id),
                "alive": bool(agent.alive),
                "energy": round(float(agent.energy), 8),
                "health": round(float(agent.health), 8),
                "injury": round(float(getattr(agent, "injury", 0.0)), 8),
                "lineage_id": str(st.lineage_id),
                "offspring_count": int(st.offspring_count),
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def test_branch_niche_observability_disabled_writes_no_niche_file(tmp_path: Path) -> None:
    sim = _tiny_sim(tmp_path / "disabled", niche_enabled=False)
    sim.run(start_gen=0, skip_init=False)

    assert not (tmp_path / "disabled" / "branch_observability" / NICHE_LIFECYCLE_FILENAME).exists()


def test_branch_niche_observability_emits_compact_rows(tmp_path: Path) -> None:
    sim = _tiny_sim(tmp_path / "enabled", niche_enabled=True)
    sim.run(start_gen=0, skip_init=False)
    root = tmp_path / "enabled" / "branch_observability"
    rows = _jsonl(root / NICHE_LIFECYCLE_FILENAME)
    lifecycle = _jsonl(root / "branch_observability_agent_lifecycle.jsonl")
    summary = json.loads((root / "branch_observability_run_summary.json").read_text(encoding="utf-8"))

    assert summary["branch_niche_observability_enable"] == 1
    assert summary["niche_lifecycle_rows"] == len(rows)
    assert len(rows) == len(lifecycle)
    assert all(row["schema_version"] == "branch_niche_observability_v1" for row in rows)
    assert any(row["total_observed_ticks"] > 0 for row in rows)
    assert any(row["zone_exposure_counts"] for row in rows)
    assert any(row["food_collected_by_zone"] for row in rows)
    assert all("genome_signature" not in row for row in rows)


def test_branch_niche_observability_preserves_trajectory(tmp_path: Path) -> None:
    baseline = _tiny_sim(tmp_path / "baseline", seed=555, branch_enabled=True, niche_enabled=False)
    baseline.run(start_gen=0, skip_init=False)
    enabled = _tiny_sim(tmp_path / "enabled", seed=555, branch_enabled=True, niche_enabled=True)
    enabled.run(start_gen=0, skip_init=False)

    assert _fingerprint(baseline) == _fingerprint(enabled)


def test_branch_niche_observability_json_bytes_are_repeat_stable(tmp_path: Path) -> None:
    a = _tiny_sim(tmp_path / "a", seed=777, niche_enabled=True)
    a.run(start_gen=0, skip_init=False)
    b = _tiny_sim(tmp_path / "b", seed=777, niche_enabled=True)
    b.run(start_gen=0, skip_init=False)

    assert (tmp_path / "a" / "branch_observability" / NICHE_LIFECYCLE_FILENAME).read_bytes() == (
        tmp_path / "b" / "branch_observability" / NICHE_LIFECYCLE_FILENAME
    ).read_bytes()


def test_branch_niche_observability_checkpoint_resume_extends_existing_rows(tmp_path: Path) -> None:
    part = _tiny_sim(tmp_path / "part", seed=818, niche_enabled=True, generations=1)
    part.run(start_gen=0, skip_init=False)
    part_root = tmp_path / "part" / "branch_observability"
    part_rows = _jsonl(part_root / NICHE_LIFECYCLE_FILENAME)
    bundle = checkpoint_bundle_v2_save(part, str(tmp_path / "bundle_gen1"), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    resume_dir = tmp_path / "resume"
    shutil.copytree(part_root, resume_dir / "branch_observability")
    resumed = _tiny_sim(resume_dir, seed=818, niche_enabled=True, generations=2)
    checkpoint_bundle_v2_apply(resumed, payload, restore_rng=True)
    resumed.run(start_gen=1, skip_init=True)

    resume_rows = _jsonl(resume_dir / "branch_observability" / NICHE_LIFECYCLE_FILENAME)
    assert {row["agent_id"] for row in part_rows}.issubset({row["agent_id"] for row in resume_rows})
    assert sum(row["total_observed_ticks"] for row in resume_rows) > sum(
        row["total_observed_ticks"] for row in part_rows
    )
