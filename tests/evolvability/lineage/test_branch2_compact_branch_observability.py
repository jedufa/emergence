from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.branch_observability import (
    BIRTH_EDGES_FILENAME,
    LIFECYCLE_FILENAME,
    SUMMARY_FILENAME,
    branch_observability_schema,
)
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save

import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "tools/bin/branch1r_evolutionary_branch_audit.py"
SPEC = importlib.util.spec_from_file_location("branch1r_evolutionary_branch_audit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
branch1r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(branch1r)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tiny_sim(run_dir: Path, *, seed: int = 123, enabled: bool = False, generations: int = 2) -> SimulationRunner:
    torch.manual_seed(int(seed))
    env_cfg = EnvConfig(width=8, height=8)
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
    sim_cfg.branch_observability_enable = bool(enabled)
    sim_cfg.branch_observability_run_id = "branch2_test_run"
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
                "lineage_id": str(st.lineage_id),
                "offspring_count": int(st.offspring_count),
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def test_branch_observability_defaults_off_no_files_and_trajectory_parity(tmp_path: Path) -> None:
    assert branch_observability_schema()["default_off"] is True
    disabled = _tiny_sim(tmp_path / "disabled", enabled=False)
    disabled.run(start_gen=0, skip_init=False)
    repeated = _tiny_sim(tmp_path / "repeated", enabled=False)
    repeated.run(start_gen=0, skip_init=False)

    assert _fingerprint(disabled) == _fingerprint(repeated)
    assert not (tmp_path / "disabled" / "branch_observability").exists()
    assert not (tmp_path / "repeated" / "branch_observability").exists()


def test_branch_observability_emits_compact_birth_edges_and_lifecycle(tmp_path: Path) -> None:
    sim = _tiny_sim(tmp_path / "enabled", enabled=True)
    sim.run(start_gen=0, skip_init=False)
    root = tmp_path / "enabled" / "branch_observability"
    edges = _jsonl(root / BIRTH_EDGES_FILENAME)
    lifecycle = _jsonl(root / LIFECYCLE_FILENAME)
    summary = json.loads((root / SUMMARY_FILENAME).read_text(encoding="utf-8"))

    assert summary["mechanics_effect"] == "none_reporting_only"
    assert summary["birth_edge_rows"] == len(edges)
    assert summary["lifecycle_rows"] == len(lifecycle)
    assert len(lifecycle) >= 8
    assert len({row["agent_id"] for row in lifecycle}) == len(lifecycle)
    assert all(row["schema_version"] == "branch_observability_v1" for row in edges + lifecycle)
    assert all(row["birth_event_id"].startswith("bo_birth_") for row in edges)
    assert len({row["birth_event_id"] for row in edges}) == len(edges)
    for row in edges:
        assert row["event_type"] == "birth_success"
        assert row["child_agent_id"] > 0
        assert row["primary_parent_agent_id"] > 0
        assert row["parent_agent_id"] == row["primary_parent_agent_id"]
        assert row["deterministic_child_genome_signature"]
        assert row["mutation_event_coverage"] in {"not_available", "available"}


def test_branch_observability_json_bytes_are_repeat_stable(tmp_path: Path) -> None:
    a = _tiny_sim(tmp_path / "a", seed=777, enabled=True)
    a.run(start_gen=0, skip_init=False)
    b = _tiny_sim(tmp_path / "b", seed=777, enabled=True)
    b.run(start_gen=0, skip_init=False)

    for name in (BIRTH_EDGES_FILENAME, LIFECYCLE_FILENAME, SUMMARY_FILENAME):
        left = (tmp_path / "a" / "branch_observability" / name).read_bytes()
        right = (tmp_path / "b" / "branch_observability" / name).read_bytes()
        assert left == right


def test_branch1r_helper_consumes_branch_observability_files(tmp_path: Path) -> None:
    sim = _tiny_sim(tmp_path / "helper", seed=444, enabled=True)
    sim.run(start_gen=0, skip_init=False)
    rows = branch1r.load_branch_observability_rows(tmp_path / "helper" / "branch_observability")
    result = branch1r.analyze_event_rows(rows)

    assert result["coverage"]["parent_child_edge_rows"] >= 1
    assert result["coverage"]["missing_parent_child_edges"] is False
    assert result["three_layer_gate"]["classification"] in {
        "healthy_branching_search",
        "founder_concentrated_but_branching",
        "founder_diverse_but_strategy_converged",
        "branch_stagnation",
        "branch_collapse",
        "strategy_collapse",
        "insufficient_branch_data",
    }


def test_branch_observability_checkpoint_resume_merges_existing_compact_rows(tmp_path: Path) -> None:
    part = _tiny_sim(tmp_path / "part", seed=818, enabled=True, generations=1)
    part.run(start_gen=0, skip_init=False)
    part_root = tmp_path / "part" / "branch_observability"
    part_edges = _jsonl(part_root / BIRTH_EDGES_FILENAME)
    part_lifecycle_ids = {row["agent_id"] for row in _jsonl(part_root / LIFECYCLE_FILENAME)}
    bundle = checkpoint_bundle_v2_save(part, str(tmp_path / "bundle_gen1"), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    resume_dir = tmp_path / "resume"
    shutil.copytree(part_root, resume_dir / "branch_observability")
    resumed = _tiny_sim(resume_dir, seed=818, enabled=True, generations=2)
    checkpoint_bundle_v2_apply(resumed, payload, restore_rng=True)
    resumed.run(start_gen=1, skip_init=True)

    resume_root = resume_dir / "branch_observability"
    resume_edges = _jsonl(resume_root / BIRTH_EDGES_FILENAME)
    assert {row["birth_event_id"] for row in part_edges}.issubset({row["birth_event_id"] for row in resume_edges})
    assert len({row["birth_event_id"] for row in resume_edges}) == len(resume_edges)
    resume_lifecycle_ids = {row["agent_id"] for row in _jsonl(resume_root / LIFECYCLE_FILENAME)}
    assert part_lifecycle_ids.issubset(resume_lifecycle_ids)
