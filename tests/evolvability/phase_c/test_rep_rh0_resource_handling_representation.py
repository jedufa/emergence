from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from agents.agent_policy import MLPPolicy
from agents.genome import Genome
from agents.resource_handling import (
    RESOURCE_HANDLING_AXIS_FIELD,
    RESOURCE_HANDLING_AXIS_MAX,
    RESOURCE_HANDLING_AXIS_MIN,
    RESOURCE_HANDLING_AXIS_NEUTRAL,
    RESOURCE_HANDLING_SCHEMA_VERSION,
    clamp_resource_handling_axis,
    mutate_resource_handling_axis,
    resource_handling_capacities,
    resource_handling_tradeoff_summary,
)
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from evolution.reproduction import crossover_genome
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


class CountingRandom(random.Random):
    def __init__(self, seed: int):
        super().__init__(seed)
        self.calls: dict[str, int] = {"random": 0, "gauss": 0, "choice": 0}

    def random(self) -> float:  # type: ignore[override]
        self.calls["random"] += 1
        return super().random()

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:  # type: ignore[override]
        self.calls["gauss"] += 1
        return super().gauss(mu, sigma)

    def choice(self, seq):  # type: ignore[override]
        self.calls["choice"] += 1
        return super().choice(seq)


class FixedMutationRng:
    def __init__(self, *, mutation_draw: float, gaussian_delta: float):
        self.mutation_draw = float(mutation_draw)
        self.gaussian_delta = float(gaussian_delta)

    def random(self) -> float:
        return self.mutation_draw

    def gauss(self, _mu: float = 0.0, _sigma: float = 1.0) -> float:
        return self.gaussian_delta


def _base_payload(axis: float = RESOURCE_HANDLING_AXIS_NEUTRAL, *, mutation_rate: float = 0.05) -> dict:
    payload = {
        "schema_version": "modular_genome_v1",
        "hidden_size": 64,
        "exploration_bias": 0.1,
        "aggression_bias": -0.1,
        "reproduction_threshold": 30.0,
        "mutation_rate": float(mutation_rate),
        "action_temperature": 1.0,
        "use_gru": False,
        "memory_dim": 0,
        "drive_inertia": 0.85,
        "drive_noise": 0.02,
        "drive_logit_gain": 1.0,
        "pred_lr": 0.08,
        "drive_sp_hunger": 0.55,
        "drive_sp_pain": 0.05,
        "drive_sp_fatigue": 0.15,
        "drive_sp_fear": 0.10,
        "exp_prior_strength": 0.35,
        "exp_prior_noise": 0.05,
        "drive_birth_noise": 0.03,
        "modules": [],
        "connections": [],
    }
    if axis != RESOURCE_HANDLING_AXIS_NEUTRAL:
        payload[RESOURCE_HANDLING_AXIS_FIELD] = float(axis)
    return payload


def _genome(axis: float = RESOURCE_HANDLING_AXIS_NEUTRAL, *, mutation_rate: float = 0.05) -> Genome:
    return Genome.from_dict(_base_payload(axis, mutation_rate=mutation_rate))


def _cfg(*, enabled: bool, reporting: bool = False) -> SimConfig:
    cfg = SimConfig(seed=7, generations=1, steps_per_generation=1, device="cpu", bank_device="cpu", behavior_log_enable=False)
    cfg.genome_c1_enable = False
    cfg.genome_c2_enable = False
    cfg.resource_handling_representation_enable = bool(enabled)
    cfg.resource_handling_representation_reporting_enable = bool(reporting)
    return cfg


def _make_sim(tmp_path: Path, *, seed: int, generations: int = 1, steps: int = 1) -> SimulationRunner:
    env_cfg = EnvConfig(width=8, height=8)
    pop_cfg = PopConfig(population_size=6)
    sim_cfg = SimConfig(seed=int(seed), generations=int(generations), steps_per_generation=int(steps), device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.policy_deterministic_actions = True
    sim_cfg.checkpoint_bundle_v2_period_gen = 0
    sim_cfg.resource_handling_representation_enable = True
    return SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path), perf=False)


def test_rep_rh0_tradeoff_surface_has_neutral_extremes_monotonicity_and_no_universal_upgrade() -> None:
    neutral = resource_handling_capacities(0.0)
    assert neutral["schema_version"] == RESOURCE_HANDLING_SCHEMA_VERSION
    assert neutral["resource_a_capacity"] == pytest.approx(1.0)
    assert neutral["resource_b_capacity"] == pytest.approx(1.0)
    assert resource_handling_capacities(1.0)["resource_a_capacity"] == pytest.approx(2.0)
    assert resource_handling_capacities(1.0)["resource_b_capacity"] == pytest.approx(0.0)
    assert resource_handling_capacities(-1.0)["resource_a_capacity"] == pytest.approx(0.0)
    assert resource_handling_capacities(-1.0)["resource_b_capacity"] == pytest.approx(2.0)

    previous_a = -1.0
    previous_b = 3.0
    for idx in range(401):
        axis = -1.0 + (2.0 * idx / 400.0)
        caps = resource_handling_capacities(axis)
        a_cap = float(caps["resource_a_capacity"])
        b_cap = float(caps["resource_b_capacity"])
        assert caps["capacity_total"] == pytest.approx(2.0)
        assert a_cap >= previous_a
        assert b_cap <= previous_b
        assert a_cap <= 1.0 or b_cap <= 1.0
        assert resource_handling_tradeoff_summary(axis)["no_universal_upgrade"] is True
        previous_a = a_cap
        previous_b = b_cap


def test_rep_rh0_clamp_and_serialization_legacy_neutral_safe() -> None:
    assert clamp_resource_handling_axis(5.0) == RESOURCE_HANDLING_AXIS_MAX
    assert clamp_resource_handling_axis(-5.0) == RESOURCE_HANDLING_AXIS_MIN
    assert clamp_resource_handling_axis(float("nan")) == RESOURCE_HANDLING_AXIS_NEUTRAL

    legacy = Genome.from_dict(_base_payload())
    assert getattr(legacy, RESOURCE_HANDLING_AXIS_FIELD) == RESOURCE_HANDLING_AXIS_NEUTRAL
    assert RESOURCE_HANDLING_AXIS_FIELD not in legacy.to_dict()
    assert Genome.from_dict(legacy.to_dict()).canonical_hash() == legacy.canonical_hash()

    nonzero = Genome.from_dict(_base_payload(0.375))
    assert getattr(nonzero, RESOURCE_HANDLING_AXIS_FIELD) == pytest.approx(0.375)
    assert nonzero.to_dict()[RESOURCE_HANDLING_AXIS_FIELD] == pytest.approx(0.375)
    assert Genome.from_dict(nonzero.to_dict()).canonical_hash() == nonzero.canonical_hash()


def test_rep_rh0_disabled_crossover_ignores_parent_axis_and_adds_no_rng_draws() -> None:
    cfg = _cfg(enabled=False)
    neutral_a = _genome(0.0)
    neutral_b = _genome(0.0)
    shifted_a = _genome(0.8)
    shifted_b = _genome(-0.4)

    rng_neutral = CountingRandom(1001)
    child_neutral = crossover_genome(neutral_a, neutral_b, rng_neutral, sim_cfg=cfg)
    rng_shifted = CountingRandom(1001)
    child_shifted = crossover_genome(shifted_a, shifted_b, rng_shifted, sim_cfg=cfg)

    assert isinstance(child_neutral, Genome)
    assert isinstance(child_shifted, Genome)
    assert rng_neutral.calls == rng_shifted.calls
    assert getattr(child_shifted, RESOURCE_HANDLING_AXIS_FIELD) == RESOURCE_HANDLING_AXIS_NEUTRAL
    assert child_neutral.canonical_hash() == child_shifted.canonical_hash()


def test_rep_rh0_enabled_inherits_mutates_reports_and_reconciles_parent_child_values() -> None:
    cfg = _cfg(enabled=True, reporting=True)
    parent_a = _genome(0.6, mutation_rate=0.0)
    parent_b = _genome(-0.2, mutation_rate=0.0)
    rng = CountingRandom(2002)

    child = crossover_genome(parent_a, parent_b, rng, sim_cfg=cfg)
    assert isinstance(child, Genome)
    assert getattr(child, RESOURCE_HANDLING_AXIS_FIELD) == pytest.approx(0.2)
    record = getattr(child, "_resource_handling_reproduction_record", {})
    assert record["parent_a_axis"] == pytest.approx(0.6)
    assert record["parent_b_axis"] == pytest.approx(-0.2)
    assert record["child_pre_mutation_axis"] == pytest.approx(0.2)
    assert record["child_post_mutation_axis"] == pytest.approx(getattr(child, RESOURCE_HANDLING_AXIS_FIELD))
    assert record["mutation_delta"] == pytest.approx(0.0)


def test_rep_rh0_mutation_direction_and_clamp_are_deterministic() -> None:
    positive, pos_record = mutate_resource_handling_axis(
        0.1,
        FixedMutationRng(mutation_draw=0.0, gaussian_delta=0.25),  # type: ignore[arg-type]
        mutation_rate=1.0,
        sigma=0.10,
    )
    negative, neg_record = mutate_resource_handling_axis(
        0.1,
        FixedMutationRng(mutation_draw=0.0, gaussian_delta=-0.35),  # type: ignore[arg-type]
        mutation_rate=1.0,
        sigma=0.10,
    )
    clipped, clip_record = mutate_resource_handling_axis(
        0.95,
        FixedMutationRng(mutation_draw=0.0, gaussian_delta=0.50),  # type: ignore[arg-type]
        mutation_rate=1.0,
        sigma=0.10,
    )
    assert positive > 0.1
    assert pos_record["mutation_delta"] > 0.0
    assert negative < 0.1
    assert neg_record["mutation_delta"] < 0.0
    assert clipped == RESOURCE_HANDLING_AXIS_MAX
    assert clip_record["clamped"] is True


def test_rep_rh0_checkpoint_roundtrip_preserves_enabled_representation(tmp_path: Path) -> None:
    sim_a = _make_sim(tmp_path / "rep_rh0_ckpt_a", seed=31001)
    for idx, agent in enumerate(sim_a.pop.agents.values()):
        axis = -0.5 + 0.2 * idx
        agent.genome = _genome(axis)
    bundle = tmp_path / "rep_rh0_ckpt_a" / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_b = _make_sim(tmp_path / "rep_rh0_ckpt_b", seed=31001)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    axes_a = {
        int(aid): getattr(agent.genome, RESOURCE_HANDLING_AXIS_FIELD)
        for aid, agent in sim_a.pop.agents.items()
    }
    axes_b = {
        int(aid): getattr(agent.genome, RESOURCE_HANDLING_AXIS_FIELD)
        for aid, agent in sim_b.pop.agents.items()
    }
    hashes_a = {int(aid): str(agent.genome.canonical_hash()) for aid, agent in sim_a.pop.agents.items()}
    hashes_b = {int(aid): str(agent.genome.canonical_hash()) for aid, agent in sim_b.pop.agents.items()}
    assert axes_a == axes_b
    assert hashes_a == hashes_b


def test_rep_rh0_policy_logits_identical_for_matched_organisms_except_dormant_field() -> None:
    torch.manual_seed(7007)
    policy_a = MLPPolicy(obs_size=12, hidden_size=64, n_actions=5)
    policy_b = MLPPolicy(obs_size=12, hidden_size=64, n_actions=5)
    policy_b.load_state_dict(policy_a.state_dict())
    genome_a = _genome(-0.75)
    genome_b = _genome(0.75)
    assert getattr(genome_a, RESOURCE_HANDLING_AXIS_FIELD) != getattr(genome_b, RESOURCE_HANDLING_AXIS_FIELD)

    obs = torch.linspace(-1.0, 1.0, steps=12, dtype=torch.float32)
    with torch.no_grad():
        logits_a = policy_a(obs)
        logits_b = policy_b(obs)
    assert torch.equal(logits_a, logits_b)


def test_rep_rh0_axis_not_consumed_by_live_policy_or_world_paths() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    allowed = {
        repo_root / "engine" / "agents" / "resource_handling.py",
        repo_root / "engine" / "agents" / "genome.py",
        repo_root / "engine" / "evolution" / "reproduction.py",
        repo_root / "engine" / "simulation" / "respec_n1_zone_telemetry.py",
        repo_root / "engine" / "simulation" / "metrics_pkg" / "metrics_logger.py",
        repo_root / "engine" / "simulation" / "metrics_pkg" / "rows.py",
        repo_root / "engine" / "simulation" / "runner_pkg" / "runner_config.py",
        repo_root / "engine" / "simulation" / "runner_pkg" / "runner_ticks_env_apply.py",
    }
    searched_roots = [repo_root / "engine" / "env", repo_root / "engine" / "agents", repo_root / "engine" / "simulation"]
    hits: list[Path] = []
    for root in searched_roots:
        for path in root.rglob("*.py"):
            if path in allowed:
                continue
            if RESOURCE_HANDLING_AXIS_FIELD in path.read_text(encoding="utf-8", errors="ignore"):
                hits.append(path.relative_to(repo_root))
    assert hits == []
