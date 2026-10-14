from __future__ import annotations

import csv
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from env.env_config import EnvConfig
from evolution.population import Population, PopConfig
from evolution.population_pkg.founder_balance import validate_founder_balance_config
from simulation.metrics import MetricsLogger
from simulation.runner_pkg.runner_config import SimConfig


def _population(*, enable: bool, groups: int = 8, min_per_group: int = 8, size: int = 64) -> Population:
    pop = Population(
        env_cfg=EnvConfig(width=16, height=16),
        pop_cfg=PopConfig(population_size=size, parent_fraction=0.25, cull_fraction=0.5),
        rng=random.Random(123),
        np_rng=np.random.default_rng(123),
    )
    cfg = SimConfig(seed=123)
    cfg.lineage_founder_balance_enable = bool(enable)
    cfg.lineage_founder_group_count = int(groups)
    cfg.lineage_founder_min_agents_per_group = int(min_per_group)
    cfg.lineage_founder_assignment_mode = "deterministic_round_robin"
    cfg.lineage_founder_balance_schema_version = 1
    pop.sim_cfg = cfg
    pop.init_random()
    return pop


def _lineage_counts(pop: Population) -> dict[str, int]:
    counts: dict[str, int] = {}
    for agent in pop.alive_agents():
        lineage_id = str(agent.stats.lineage_id)
        counts[lineage_id] = counts.get(lineage_id, 0) + 1
    return counts


def test_founder_balance_disabled_preserves_one_founder_per_initial_agent() -> None:
    pop = _population(enable=False)

    counts = _lineage_counts(pop)

    assert len(counts) == 64
    assert sorted(counts.values()) == [1] * 64
    assert pop.agents[1].stats.lineage_id == "L00000001"
    assert pop.agents[64].stats.lineage_id == "L00000064"
    assert pop.agents[1].stats.patch_habitat_lineage_id == 1
    assert pop.agents[64].stats.patch_habitat_lineage_id == 64


def test_founder_balance_enabled_assigns_deterministic_balanced_groups() -> None:
    first = _population(enable=True)
    second = _population(enable=True)

    counts = _lineage_counts(first)
    assignment_a = [(agent.agent_id, agent.stats.lineage_id, agent.stats.patch_habitat_lineage_id) for agent in first.alive_agents()]
    assignment_b = [(agent.agent_id, agent.stats.lineage_id, agent.stats.patch_habitat_lineage_id) for agent in second.alive_agents()]

    assert len(counts) == 8
    assert sorted(counts.values()) == [8] * 8
    assert counts["L00000001"] == 8
    assert counts["L00000008"] == 8
    assert assignment_a == assignment_b
    assert first.agents[1].stats.lineage_id == "L00000001"
    assert first.agents[9].stats.lineage_id == "L00000001"
    assert first.agents[8].stats.lineage_id == "L00000008"


@pytest.mark.parametrize(
    ("groups", "min_per_group", "message"),
    [
        (0, 1, "group_count"),
        (65, 1, "cannot exceed population_size"),
        (8, 9, "cannot exceed population_size"),
        (8, 0, "min_agents"),
    ],
)
def test_founder_balance_invalid_config_fails_clearly(groups: int, min_per_group: int, message: str) -> None:
    cfg = SimConfig(seed=123)
    cfg.lineage_founder_balance_enable = True
    cfg.lineage_founder_group_count = groups
    cfg.lineage_founder_min_agents_per_group = min_per_group

    with pytest.raises(ValueError, match=message):
        validate_founder_balance_config(population_size=64, sim_cfg=cfg)


def test_founder_balance_checkpoint_style_roundtrip_preserves_lineage_ids() -> None:
    pop = _population(enable=True)
    before = [(agent.agent_id, agent.stats.lineage_id, agent.stats.founder_id) for agent in pop.alive_agents()]

    restored = pickle.loads(pickle.dumps(pop))
    after = [(agent.agent_id, agent.stats.lineage_id, agent.stats.founder_id) for agent in restored.alive_agents()]

    assert after == before
    assert _lineage_counts(restored) == _lineage_counts(pop)


def test_founder_balance_metrics_emit_cleanly(tmp_path: Path) -> None:
    pop = _population(enable=True)
    logger = MetricsLogger(str(tmp_path))

    logger.log_generation(
        generation=1,
        agents=pop.alive_agents(),
        deaths_this_gen=0,
        selected_parent_agents=pop.alive_agents(),
        sim_cfg=pop.sim_cfg,
        env_cfg=pop.env_cfg,
    )

    with (tmp_path / "metrics.csv").open("r", newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[-1]

    assert row["founder_balance_enable"] == "1"
    assert row["founder_group_count"] == "8"
    assert row["founder_min_agents_per_group"] == "8"
    assert float(row["founder_initial_effective_lineages"]) == pytest.approx(8.0)
    assert float(row["founder_initial_top1_share"]) == pytest.approx(0.125)
    assert float(row["early_lineage_effective_gen_1"]) == pytest.approx(8.0)
    assert float(row["early_lineage_top1_share_gen_1"]) == pytest.approx(0.125)


def test_founder_balance_does_not_enable_intervention_flags() -> None:
    cfg = SimpleNamespace(
        lineage_founder_balance_enable=True,
        lineage_founder_group_count=8,
        lineage_founder_min_agents_per_group=8,
        lineage_founder_assignment_mode="deterministic_round_robin",
        lineage_founder_balance_schema_version=1,
    )

    assert not bool(getattr(cfg, "lineage_protection_enable", False))
    assert not bool(getattr(cfg, "lineage_rescue_enable", False))
    assert not bool(getattr(cfg, "diversity_reward_enable", False))
    assert not bool(getattr(cfg, "archive_reseed_enable", False))
