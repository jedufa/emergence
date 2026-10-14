from __future__ import annotations

import random
from pathlib import Path

import pytest

from agents.family_head_residuals import (
    FAMILY_HEAD_RESIDUAL_FIELD,
    FAMILY_HEAD_RESIDUAL_FAMILY_ORDER,
    FAMILY_HEAD_RESIDUAL_MUTATION_OPERATOR_ID,
    canonical_family_head_residual_block,
    center_project_family_residuals,
    family_head_residual_canonical_digest,
    family_head_residual_semantic_digest,
    mutate_family_head_residual_block,
    recombine_family_head_residual_blocks,
)
from agents.genome import Genome
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
        self.random_calls = 0

    def random(self) -> float:  # type: ignore[override]
        self.random_calls += 1
        return super().random()


def _genome(block=None) -> Genome:
    payload = {
        "hidden_size": 64,
        "exploration_bias": 0.1,
        "aggression_bias": 0.0,
        "reproduction_threshold": 30.0,
        "mutation_rate": 0.0,
        "action_temperature": 1.0,
        "use_gru": False,
        "memory_dim": 0,
        "drive_inertia": 0.85,
        "drive_noise": 0.02,
        "drive_logit_gain": 1.0,
        "pred_lr": 0.08,
    }
    if block is not None:
        payload[FAMILY_HEAD_RESIDUAL_FIELD] = block
    return Genome.from_dict(payload)


def _cfg(*, genome: bool = False, mutation: bool = False, reporting: bool = False) -> SimConfig:
    cfg = SimConfig(seed=777, generations=1, steps_per_generation=1, device="cpu", bank_device="cpu", behavior_log_enable=False)
    cfg.genome_c1_enable = False
    cfg.genome_c2_enable = False
    cfg.policy_family_head_residual_genome_enable = bool(genome)
    cfg.policy_family_head_residual_mutation_enable = bool(mutation)
    cfg.policy_family_head_residual_reporting_enable = bool(reporting)
    cfg.policy_family_head_residual_mutation_step = 0.1
    return cfg


def test_polarch3_projection_removes_offset_and_enforces_caps() -> None:
    projected = center_project_family_residuals({"movement": 4.0, "gather": 0.0, "rest": 0.0, "attack": 0.0})

    assert projected["sum_residual"] == pytest.approx(0.0, abs=1e-12)
    assert projected["l1_magnitude"] <= 2.0 + 1e-12
    assert max(abs(v) for v in projected["effective_residuals"].values()) <= 1.0
    assert projected["cap_binding"] is True


def test_polarch3_genome_serialization_and_semantic_hashes_are_stable() -> None:
    common_offset = canonical_family_head_residual_block({"movement": 5.2, "gather": 4.8, "rest": 4.8, "attack": 4.8})
    centered = canonical_family_head_residual_block({"movement": 0.3, "gather": -0.1, "rest": -0.1, "attack": -0.1})
    g1 = _genome(common_offset)
    g2 = _genome(centered)

    assert g1.to_dict()[FAMILY_HEAD_RESIDUAL_FIELD] == g2.to_dict()[FAMILY_HEAD_RESIDUAL_FIELD]
    assert g1.canonical_hash() == g2.canonical_hash()
    assert family_head_residual_semantic_digest(common_offset) == family_head_residual_semantic_digest(centered)
    assert family_head_residual_canonical_digest(common_offset) == family_head_residual_canonical_digest(centered)


def test_polarch3_legacy_missing_block_loads_without_serialized_block() -> None:
    g = _genome()

    assert FAMILY_HEAD_RESIDUAL_FIELD not in g.to_dict()
    assert getattr(g, FAMILY_HEAD_RESIDUAL_FIELD) is None
    assert canonical_family_head_residual_block(None)["stored_residual_raw"] == [0.0, 0.0, 0.0, 0.0]


def test_polarch3_founder_initialization_is_zero_when_enabled(tmp_path: Path) -> None:
    env_cfg = EnvConfig(width=8, height=8)
    pop_cfg = PopConfig(population_size=4)
    sim_cfg = SimConfig(seed=313, generations=1, steps_per_generation=1, device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.policy_family_head_residual_genome_enable = True
    sim = SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path / "founders"), perf=False)
    sim.run(start_gen=0, skip_init=False)

    expected = canonical_family_head_residual_block(None)
    for agent in sim.pop.agents.values():
        assert agent.genome.to_dict()[FAMILY_HEAD_RESIDUAL_FIELD] == expected


def test_polarch3_one_parent_and_midpoint_recombination_preserve_center() -> None:
    a = canonical_family_head_residual_block({"movement": 0.6, "gather": -0.2, "rest": -0.2, "attack": -0.2})
    b = canonical_family_head_residual_block({"movement": -0.2, "gather": 0.6, "rest": -0.2, "attack": -0.2})
    rec = recombine_family_head_residual_blocks(a, b)
    child = rec["child"]["stored_residual_raw"]

    assert rec["draw_budget"] == {"mutation.py": 0}
    assert sum(child) == pytest.approx(0.0, abs=1e-12)
    assert child == pytest.approx([0.2, 0.2, -0.2, -0.2])


def test_polarch3_pairwise_mutation_uses_two_draws_and_records_event() -> None:
    rng = CountingRandom(11)
    after, event = mutate_family_head_residual_block(None, rng, step=0.1)

    assert rng.random_calls == 2
    assert event["operator_id"] == FAMILY_HEAD_RESIDUAL_MUTATION_OPERATOR_ID
    assert event["draw_budget"] == {"mutation.py": 2}
    assert sum(after["stored_residual_raw"]) == pytest.approx(0.0, abs=1e-12)
    assert event["semantic_change"] is True


def test_polarch3_crossover_default_off_does_not_attach_block_or_extra_draws() -> None:
    g1 = _genome(canonical_family_head_residual_block({"movement": 0.4, "gather": -0.4}))
    g2 = _genome(canonical_family_head_residual_block({"rest": 0.4, "attack": -0.4}))
    rng = CountingRandom(101)
    child = crossover_genome(g1, g2, rng, sim_cfg=_cfg(genome=False, mutation=False))

    assert FAMILY_HEAD_RESIDUAL_FIELD not in child.to_dict()
    assert not hasattr(child, "_family_head_residual_reproduction_record")


def test_polarch3_crossover_inherits_and_optionally_mutates_with_fixed_extra_draws() -> None:
    g1 = _genome(canonical_family_head_residual_block({"movement": 0.4, "gather": -0.4}))
    g2 = _genome(canonical_family_head_residual_block({"rest": 0.4, "attack": -0.4}))
    rng_no_mut = CountingRandom(202)
    child_no_mut = crossover_genome(g1, g2, rng_no_mut, sim_cfg=_cfg(genome=True, mutation=False, reporting=True))
    rng_mut = CountingRandom(202)
    child_mut = crossover_genome(g1, g2, rng_mut, sim_cfg=_cfg(genome=True, mutation=True, reporting=True))

    assert FAMILY_HEAD_RESIDUAL_FIELD in child_no_mut.to_dict()
    assert rng_mut.random_calls - rng_no_mut.random_calls == 2
    record = getattr(child_mut, "_family_head_residual_reproduction_record")
    assert record["draw_budget"] == {"mutation.py": 2}
    assert record["mutation_event"]["operator_id"] == FAMILY_HEAD_RESIDUAL_MUTATION_OPERATOR_ID
    assert sum(child_mut.to_dict()[FAMILY_HEAD_RESIDUAL_FIELD]["stored_residual_raw"]) == pytest.approx(0.0, abs=1e-12)


def test_polarch3_checkpoint_roundtrip_preserves_nonzero_dormant_block(tmp_path: Path) -> None:
    env_cfg = EnvConfig(width=8, height=8)
    pop_cfg = PopConfig(population_size=4)
    sim_cfg = SimConfig(seed=303, generations=1, steps_per_generation=1, device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.policy_deterministic_actions = True
    sim = SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path / "a"), perf=False)
    sim.run(start_gen=0, skip_init=False)
    block = canonical_family_head_residual_block({"movement": 0.4, "gather": -0.4})
    for agent in sim.pop.agents.values():
        agent.genome = Genome.from_dict({**agent.genome.to_dict(), FAMILY_HEAD_RESIDUAL_FIELD: block})

    bundle = tmp_path / "a" / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    restored = SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path / "b"), perf=False)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    for aid, agent in sim.pop.agents.items():
        assert restored.pop.agents[aid].genome.to_dict()[FAMILY_HEAD_RESIDUAL_FIELD] == agent.genome.to_dict()[FAMILY_HEAD_RESIDUAL_FIELD]


def test_polarch3_family_order_is_fixed() -> None:
    assert FAMILY_HEAD_RESIDUAL_FAMILY_ORDER == ("movement", "gather", "rest", "attack")
