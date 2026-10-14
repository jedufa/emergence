from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from env.env_config import EnvConfig
from evolution.population import PopConfig, Population
from simulation.metrics import MetricsLogger
from simulation.metrics_pkg.lineage_forensics import (
    LineageCollapseForensicsState,
    compute_lineage_collapse_forensics,
    validate_lineage_collapse_forensics_config,
)
from simulation.runner_pkg.runner_config import SimConfig


def _agent(agent_id: int, lineage_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=agent_id,
        alive=True,
        stats=SimpleNamespace(
            lineage_id=lineage_id,
            founder_id=agent_id,
            parent_ids=(),
            parent_lineage_ids=(),
            birth_generation=0,
            birth_tick=0,
        ),
        drives=[0.0, 0.0, 0.0, 0.0],
        chemistry=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


def _agents(lineages: list[str]) -> list[SimpleNamespace]:
    return [_agent(i + 1, lineage_id) for i, lineage_id in enumerate(lineages)]


def _enabled_cfg() -> SimConfig:
    cfg = SimConfig(seed=123)
    cfg.lineage_collapse_forensics_enable = True
    return cfg


def _population(size: int = 16) -> Population:
    pop = Population(
        env_cfg=EnvConfig(width=16, height=16),
        pop_cfg=PopConfig(population_size=size, parent_fraction=0.25, cull_fraction=0.5),
        rng=random.Random(123),
        np_rng=np.random.default_rng(123),
    )
    pop.sim_cfg = SimConfig(seed=123)
    pop.init_random()
    return pop


def _last_metrics_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))[-1]


def test_lineage_forensics_config_defaults_false() -> None:
    cfg = SimConfig()

    assert cfg.lineage_collapse_forensics_enable is False
    assert cfg.lineage_collapse_forensics_schema_version == 1


def test_lineage_forensics_invalid_schema_fails_clearly() -> None:
    cfg = _enabled_cfg()
    cfg.lineage_collapse_forensics_schema_version = 99

    with pytest.raises(ValueError, match="lineage_collapse_forensics_schema_version"):
        validate_lineage_collapse_forensics_config(cfg)


def test_lineage_forensics_records_first_collapse_generations() -> None:
    cfg = _enabled_cfg()
    state = LineageCollapseForensicsState()

    _, state = compute_lineage_collapse_forensics(
        generation=1,
        agents=_agents(["L00000001", "L00000001", "L00000002", "L00000003"]),
        sim_cfg=cfg,
        state=state,
    )
    _, state = compute_lineage_collapse_forensics(
        generation=2,
        agents=_agents(["L00000001", "L00000001", "L00000001", "L00000002"]),
        sim_cfg=cfg,
        state=state,
    )
    metrics, state = compute_lineage_collapse_forensics(
        generation=3,
        agents=_agents(["L00000001", "L00000001", "L00000001", "L00000001"]),
        sim_cfg=cfg,
        state=state,
    )

    assert metrics["first_effective_lineage_collapse_generation"] == 3
    assert metrics["first_top1_dominance_generation"] == 3
    assert metrics["first_single_alive_lineage_generation"] == 3
    assert metrics["collapse_generation_observed"] == 1
    events = json.loads(metrics["lineage_bottleneck_events_json"])["events"]
    assert [event["event"] for event in events if event["generation"] == 3] == [
        "first_effective_lineage_collapse",
        "first_single_alive_lineage",
        "first_top1_dominance",
    ]


def test_lineage_forensics_records_birth_skew_and_failure_context() -> None:
    cfg = _enabled_cfg()
    state = LineageCollapseForensicsState()

    _, state = compute_lineage_collapse_forensics(
        generation=1,
        agents=_agents(["L00000001", "L00000002"]),
        sim_cfg=cfg,
        state=state,
        birth_contexts=[{"lineage_id": "L00000001"}, {"lineage_id": "L00000002"}],
    )
    metrics, _ = compute_lineage_collapse_forensics(
        generation=2,
        agents=_agents(["L00000001", "L00000002"]),
        sim_cfg=cfg,
        state=state,
        birth_contexts=[{"lineage_id": "L00000001"}, {"lineage_id": "L00000001"}],
        failed_births=3,
        birth_failure_counts={"birth_fail_no_space": 2, "birth_fail_no_mate": 1},
    )

    assert metrics["first_top1_birth_dominance_generation"] == 2
    assert metrics["minority_lineage_birth_window"] == 1
    failures = json.loads(metrics["lineage_birth_failure_context_json"])["contexts"]
    assert failures == [
        {"context": "birth_fail_no_mate", "count": 1},
        {"context": "birth_fail_no_space", "count": 2},
    ]


def test_lineage_forensics_records_death_context_by_lineage() -> None:
    cfg = _enabled_cfg()

    metrics, _ = compute_lineage_collapse_forensics(
        generation=1,
        agents=_agents(["L00000001", "L00000002"]),
        sim_cfg=cfg,
        death_contexts=[
            {"lineage_id": "L00000001", "death_cause": "predator"},
            {"lineage_id": "L00000002", "death_cause": "env"},
        ],
    )

    assert json.loads(metrics["lineage_death_context_json"])["contexts"] == [
        {"context": "env", "count": 1},
        {"context": "predator", "count": 1},
    ]
    assert json.loads(metrics["predator_deaths_by_lineage_json"])["lineages"] == [
        {"lineage_id": "L00000001", "count": 1},
    ]
    assert json.loads(metrics["nonpredator_deaths_by_lineage_json"])["lineages"] == [
        {"lineage_id": "L00000002", "count": 1},
    ]


def test_lineage_forensics_records_patch_occupancy_context_and_branch_timing() -> None:
    cfg = _enabled_cfg()
    descriptors = [
        {
            "lineage_id": "L00000001",
            "lineage_alive_count": 3,
            "lineage_patch_food_amount_mean": 0.5,
            "lineage_patch_waste_mean": 0.1,
            "lineage_patch_density_mean": 2.0,
            "lineage_patch_shelter_mean": 0.2,
            "lineage_patch_recovery_activity_mean": 0.3,
        },
        {
            "lineage_id": "L00000002",
            "lineage_alive_count": 1,
            "lineage_patch_food_amount_mean": 0.7,
            "lineage_patch_waste_mean": 0.0,
            "lineage_patch_density_mean": 1.0,
            "lineage_patch_shelter_mean": 0.4,
            "lineage_patch_recovery_activity_mean": 0.6,
        },
    ]

    metrics, _ = compute_lineage_collapse_forensics(
        generation=1,
        agents=_agents(["L00000001", "L00000001", "L00000001", "L00000002"]),
        sim_cfg=cfg,
        lineage_strategy_descriptors_json=json.dumps(descriptors),
        lineage_branch_candidates_json=json.dumps([{"lineage_id": "L00000002"}]),
    )

    contexts = json.loads(metrics["lineage_patch_occupancy_context_json"])["contexts"]
    assert [row["lineage_id"] for row in contexts] == ["L00000001", "L00000002"]
    assert metrics["top_lineage_patch_occupancy_share"] == pytest.approx(0.75)
    assert metrics["lineages_with_distinct_patch_contexts"] == 2
    assert metrics["branch_candidates_before_collapse"] == 1
    assert metrics["descriptor_lineages_before_collapse"] == 2


def test_lineage_forensics_state_roundtrip_preserves_observed_curves() -> None:
    cfg = _enabled_cfg()
    metrics, state = compute_lineage_collapse_forensics(
        generation=1,
        agents=_agents(["L00000001", "L00000002"]),
        sim_cfg=cfg,
        birth_contexts=[{"lineage_id": "L00000001"}],
    )

    restored = LineageCollapseForensicsState.from_dict(state.to_dict())
    metrics_after, _ = compute_lineage_collapse_forensics(
        generation=2,
        agents=_agents(["L00000001"]),
        sim_cfg=cfg,
        state=restored,
    )

    assert json.loads(metrics["lineage_alive_counts_by_generation_json"])["generations"][0]["generation"] == 1
    assert json.loads(metrics_after["lineage_alive_counts_by_generation_json"])["generations"][0]["generation"] == 1
    assert metrics_after["first_single_alive_lineage_generation"] == 2


def test_metrics_logger_forensics_default_off_and_enabled_neutral(tmp_path: Path) -> None:
    pop = _population()
    before = [(agent.agent_id, agent.stats.lineage_id) for agent in pop.alive_agents()]

    off_logger = MetricsLogger(str(tmp_path / "off"))
    off_cfg = pop.sim_cfg
    off_logger.log_generation(
        generation=1,
        agents=pop.alive_agents(),
        deaths_this_gen=2,
        successful_births=3,
        failed_births=1,
        selected_parent_agents=pop.alive_agents(),
        sim_cfg=off_cfg,
        env_cfg=pop.env_cfg,
    )

    on_logger = MetricsLogger(str(tmp_path / "on"))
    on_cfg = SimConfig(seed=123)
    on_cfg.lineage_collapse_forensics_enable = True
    on_logger.log_generation(
        generation=1,
        agents=pop.alive_agents(),
        deaths_this_gen=2,
        successful_births=3,
        failed_births=1,
        birth_fail_no_space=1,
        selected_parent_agents=pop.alive_agents(),
        sim_cfg=on_cfg,
        env_cfg=pop.env_cfg,
        birth_contexts=[{"lineage_id": "L00000001"}],
    )

    off_row = _last_metrics_row(tmp_path / "off" / "metrics.csv")
    on_row = _last_metrics_row(tmp_path / "on" / "metrics.csv")
    after = [(agent.agent_id, agent.stats.lineage_id) for agent in pop.alive_agents()]

    assert off_row["lineage_forensics_enabled"] == "0"
    assert on_row["lineage_forensics_enabled"] == "1"
    assert off_row["alive"] == on_row["alive"]
    assert off_row["successful_births"] == on_row["successful_births"]
    assert off_row["failed_births"] == on_row["failed_births"]
    assert off_row["deaths"] == on_row["deaths"]
    assert before == after
    assert json.loads(on_row["lineage_birth_failure_context_json"])["contexts"] == [
        {"context": "birth_fail_no_space", "count": 1},
    ]
