from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch

from agents.cognition import (
    MEMORY_LADDER,
    cognitive_trait_ids,
    cognitive_trait_registry,
    compute_cognitive_budget_ledger,
    express_cognitive_phenotype,
    get_cognitive_trait,
)
from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
from simulation.metrics import MetricsLogger
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _genome(*, memory_dim: int, use_gru: bool | None = None) -> Genome:
    md = int(memory_dim)
    return Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=30.0,
        mutation_rate=0.03,
        action_temperature=1.0,
        use_gru=bool(md > 0) if use_gru is None else bool(use_gru),
        memory_dim=md,
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.1,
        pred_lr=0.08,
        developmental_program=canonicalize_developmental_program(None),
    )


def _make_sim(
    run_dir: Path,
    *,
    seed: int = 42,
    generations: int = 2,
    steps: int = 4,
    pop_size: int = 8,
    metrics: bool = False,
) -> SimulationRunner:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    env_cfg = EnvConfig(width=10, height=10)
    pop_cfg = PopConfig(population_size=int(pop_size))
    sim_cfg = SimConfig(
        seed=int(seed),
        generations=int(generations),
        steps_per_generation=int(steps),
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
    )
    sim_cfg.policy_deterministic_actions = True
    sim_cfg.checkpoint_bundle_v2_period_gen = 1
    logger = MetricsLogger(str(run_dir)) if metrics else NoOpMetricsLogger()
    return SimulationRunner(env_cfg=env_cfg, pop_cfg=pop_cfg, sim_cfg=sim_cfg, logger=logger, run_dir=str(run_dir), perf=False)


def _last_metrics_row(run_dir: Path) -> dict[str, str]:
    with (run_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    return dict(rows[-1])


def _cognitive_subset(row: dict[str, str]) -> dict[str, str]:
    keys = [
        "cog_capacity_mean",
        "memory_enabled_fraction",
        "memory_dim_mean",
        "memory_bucket_0_fraction",
        "memory_bucket_64_fraction",
        "gru_enabled_fraction",
        "prediction_lr_mean",
        "cog_cost_mean",
        "direct_cognition_reward_present_flag",
    ]
    return {k: row[k] for k in keys}


def _genome_hashes(sim: SimulationRunner) -> dict[int, str]:
    return {int(aid): str(a.genome.canonical_hash()) for aid, a in sorted(sim.pop.agents.items())}


def _energies(sim: SimulationRunner) -> dict[int, float]:
    return {int(aid): round(float(a.energy), 9) for aid, a in sorted(sim.pop.agents.items())}


def test_stage0_registry_stable_and_honest() -> None:
    ids = cognitive_trait_ids()
    assert ids == tuple(t.trait_id for t in cognitive_trait_registry())
    assert ids[:5] == (
        "memory_dim",
        "policy_memory_enabled",
        "policy_hidden_size",
        "action_temperature",
        "exploration_bias",
    )
    assert "future_introspection_channel" in ids
    assert "future_self_regulation_channel" in ids
    assert get_cognitive_trait("memory_dim").allowed_ladder == MEMORY_LADDER
    assert get_cognitive_trait("policy_hidden_size").status == "fixed"
    assert get_cognitive_trait("policy_hidden_size").mutable is False
    assert get_cognitive_trait("future_introspection_channel").status == "future_reserved"
    assert get_cognitive_trait("late_prediction_selection_cost").status == "partially_active"


def test_stage0_phenotype_expression_memory_disabled_enabled_and_deterministic() -> None:
    disabled = express_cognitive_phenotype(_genome(memory_dim=0, use_gru=False))
    enabled_a = express_cognitive_phenotype(_genome(memory_dim=64))
    enabled_b = express_cognitive_phenotype(_genome(memory_dim=64))

    assert disabled.memory_enabled is False
    assert disabled.policy_memory_enabled is False
    assert disabled.effective_memory_capacity == 0
    assert enabled_a.memory_enabled is True
    assert enabled_a.policy_memory_enabled is True
    assert enabled_a.effective_memory_capacity == 64
    assert enabled_a == enabled_b


def test_stage0_budget_ledger_reports_existing_cost_without_new_charge() -> None:
    env_cfg = EnvConfig(width=8, height=8)
    sim_cfg = SimConfig(seed=1)

    low = express_cognitive_phenotype(_genome(memory_dim=0), env_cfg=env_cfg, sim_cfg=sim_cfg)
    high = express_cognitive_phenotype(_genome(memory_dim=64), env_cfg=env_cfg, sim_cfg=sim_cfg)
    low_ledger = compute_cognitive_budget_ledger(low, env_cfg=env_cfg, sim_cfg=sim_cfg)
    high_ledger = compute_cognitive_budget_ledger(high, env_cfg=env_cfg, sim_cfg=sim_cfg)

    assert high_ledger.total_potential_cognitive_capacity == low_ledger.total_potential_cognitive_capacity
    assert high_ledger.total_active_cognitive_capacity > low_ledger.total_active_cognitive_capacity
    assert high_ledger.memory_upkeep_cost > low_ledger.memory_upkeep_cost
    assert high_ledger.charged_energy_delta == 0.0
    assert high_ledger.experimental_charge_enabled is False


def test_stage0_metrics_output_includes_cognitive_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "metrics_columns"
    sim = _make_sim(run_dir, seed=11, generations=1, steps=3, pop_size=6, metrics=True)
    sim.run(start_gen=0, skip_init=False)
    row = _last_metrics_row(run_dir)

    for col in (
        "cog_capacity_mean",
        "memory_enabled_fraction",
        "gru_enabled_fraction",
        "prediction_lr_mean",
        "cog_cost_mean",
        "cognitive_strategy_count_approx",
        "direct_cognition_reward_present_flag",
    ):
        assert col in row


def test_stage0_same_seed_produces_same_cognitive_metrics(tmp_path: Path) -> None:
    run_a = tmp_path / "same_seed_a"
    run_b = tmp_path / "same_seed_b"
    sim_a = _make_sim(run_a, seed=12, generations=2, steps=4, pop_size=8, metrics=True)
    sim_a.run(start_gen=0, skip_init=False)
    sim_b = _make_sim(run_b, seed=12, generations=2, steps=4, pop_size=8, metrics=True)
    sim_b.run(start_gen=0, skip_init=False)
    assert _cognitive_subset(_last_metrics_row(run_a)) == _cognitive_subset(_last_metrics_row(run_b))


def test_stage0_checkpoint_resume_preserves_cognitive_status(tmp_path: Path) -> None:
    run_a = tmp_path / "ckpt_a"
    sim_a = _make_sim(run_a, seed=13, generations=1, steps=4, pop_size=8, metrics=False)
    sim_a.run(start_gen=0, skip_init=False)
    before = compute_cognitive_status_metrics(
        sim_a.pop.alive_agents(),
        selected_parent_agents=getattr(sim_a, "_selected_parent_agents_gen", []),
        env_cfg=sim_a.env_cfg,
        sim_cfg=sim_a.sim_cfg,
    )

    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    run_b = tmp_path / "ckpt_b"
    sim_b = _make_sim(run_b, seed=13, generations=1, steps=4, pop_size=8, metrics=False)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    after = compute_cognitive_status_metrics(
        sim_b.pop.alive_agents(),
        selected_parent_agents=getattr(sim_b, "_selected_parent_agents_gen", []),
        env_cfg=sim_b.env_cfg,
        sim_cfg=sim_b.sim_cfg,
    )
    assert before == after


def test_stage0_status_computation_is_non_mutating(tmp_path: Path) -> None:
    run_dir = tmp_path / "non_mutating"
    sim = _make_sim(run_dir, seed=14, generations=1, steps=4, pop_size=8, metrics=False)
    sim.run(start_gen=0, skip_init=False)

    hashes_before = _genome_hashes(sim)
    energies_before = _energies(sim)
    compute_cognitive_status_metrics(
        sim.pop.alive_agents(),
        selected_parent_agents=getattr(sim, "_selected_parent_agents_gen", []),
        env_cfg=sim.env_cfg,
        sim_cfg=sim.sim_cfg,
    )
    assert _genome_hashes(sim) == hashes_before
    assert _energies(sim) == energies_before


def test_stage0_no_new_direct_cognition_reward_terms() -> None:
    root = Path(__file__).resolve().parents[3]
    selection_sources = [
        root / "engine" / "simulation" / "runner_pkg" / "runner_selection.py",
        root / "engine" / "evolution" / "population_pkg" / "core.py",
        root / "engine" / "evolution" / "selection.py",
    ]
    text = "\n".join(p.read_text(encoding="utf-8") for p in selection_sources)
    forbidden = (
        "memory_dim",
        "hidden_size",
        "cog_capacity",
        "cognitive_capacity",
        "introspection",
        "self_regulation",
    )
    for token in forbidden:
        assert token not in text
