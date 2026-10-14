from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from agents.developmental_program_v1 import default_program
from agents.genome import Genome
from evolution.mutation_weather import mutation_weather_state
from evolution.reproduction import crossover_genome
from simulation.metrics_pkg.lineage_reproduction_funnel import LineageReproductionFunnelWriter
from simulation.runner_pkg.runner_config import SimConfig


def _cfg() -> SimConfig:
    cfg = SimConfig(seed=123)
    cfg.genome_c1_enable = False
    cfg.genome_c2_enable = False
    cfg.lineage_reproduction_funnel_reporting_enable = True
    return cfg


def _genome(*, mutation_rate: float = 1.0) -> Genome:
    return Genome(
        hidden_size=64,
        exploration_bias=0.0,
        aggression_bias=0.0,
        reproduction_threshold=40.0,
        mutation_rate=mutation_rate,
        action_temperature=1.0,
        use_gru=False,
        memory_dim=0,
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.0,
        pred_lr=0.08,
        developmental_program=default_program(),
    )


def _scalar_tuple(g: Genome) -> tuple[float, ...]:
    return (
        float(g.exploration_bias),
        float(g.aggression_bias),
        float(g.reproduction_threshold),
        float(g.action_temperature),
        float(g.drive_inertia),
        float(g.drive_noise),
        float(g.drive_logit_gain),
        float(g.pred_lr),
    )


def test_mutation_weather_disabled_preserves_child_genome_values() -> None:
    parent_a = _genome()
    parent_b = _genome()
    baseline_cfg = _cfg()
    disabled_cfg = _cfg()
    disabled_cfg.mutation_weather_enabled = False
    disabled_cfg.mutation_weather_profile = "safe_pulse_v0"
    disabled_cfg._mutation_weather_generation = 1

    baseline = crossover_genome(parent_a, parent_b, random.Random(7), sim_cfg=baseline_cfg)
    disabled = crossover_genome(parent_a, parent_b, random.Random(7), sim_cfg=disabled_cfg)

    assert _scalar_tuple(disabled) == _scalar_tuple(baseline)
    assert not hasattr(disabled, "mutation_weather_enabled")


def test_mutation_weather_schedule_is_deterministic() -> None:
    cfg = _cfg()
    cfg.mutation_weather_enabled = True
    cfg.mutation_weather_profile = "safe_pulse_v0"
    cfg.mutation_weather_seed_salt = "mutrad1-test"

    a = mutation_weather_state(cfg, 17)
    b = mutation_weather_state(cfg, 17)

    assert a == b


def test_fixture_pulse_generation_uses_multiplier_and_nonpulse_is_one() -> None:
    cfg = _cfg()
    cfg.mutation_weather_enabled = True
    cfg.mutation_weather_profile = "safe_pulse_v0"
    cfg.mutation_weather_max_scalar_multiplier = 3.0
    cfg._mutation_weather_fixture_generations = (1,)

    pulse = mutation_weather_state(cfg, 1)
    off = mutation_weather_state(cfg, 2)

    assert pulse.active is True
    assert pulse.scalar_multiplier == pytest.approx(3.0)
    assert off.active is False
    assert off.scalar_multiplier == pytest.approx(1.0)


def test_pulse_generation_multiplies_scalar_sigma_only() -> None:
    parent_a = _genome()
    parent_b = _genome()
    baseline_cfg = _cfg()
    baseline_cfg._mutation_weather_generation = 1
    weather_cfg = _cfg()
    weather_cfg.mutation_weather_enabled = True
    weather_cfg.mutation_weather_profile = "safe_pulse_v0"
    weather_cfg.mutation_weather_max_scalar_multiplier = 3.0
    weather_cfg.mutation_weather_fixture_generations = "[1]"
    weather_cfg._mutation_weather_generation = 1

    baseline = crossover_genome(parent_a, parent_b, random.Random(11), sim_cfg=baseline_cfg)
    weather = crossover_genome(parent_a, parent_b, random.Random(11), sim_cfg=weather_cfg)

    assert abs(float(weather.exploration_bias)) > abs(float(baseline.exploration_bias))
    assert getattr(weather, "mutation_weather_active") == 1
    assert getattr(weather, "mutation_weather_scalar_multiplier") == pytest.approx(3.0)
    assert getattr(weather, "mutation_weather_structural_enabled") == 0
    assert len(getattr(weather, "modules", []) or []) == 0
    assert len(getattr(weather, "connections", []) or []) == 0


def test_structural_weather_fails_closed() -> None:
    cfg = _cfg()
    cfg.mutation_weather_enabled = True
    cfg.mutation_weather_profile = "safe_pulse_v0"
    cfg.mutation_weather_structural_enabled = True
    cfg.mutation_weather_fixture_generations = "[1]"

    state = mutation_weather_state(cfg, 1)

    assert state.active is True
    assert state.structural_enabled is False


def test_birth_event_rows_include_weather_metadata_and_summary(tmp_path) -> None:
    writer = LineageReproductionFunnelWriter(str(tmp_path))
    event = {
        "schema_version": 4,
        "event_schema_version": 4,
        "generation": 1,
        "tick": 2,
        "event": "successful_birth",
        "event_type": "birth_success",
        "lineage_id": "L00000001",
        "child_lineage_id": "L00000001",
        "parent_lineage_id": "L00000001",
        "parent_id": 1,
        "child_id": 2,
        "birth_attempt_count": 1,
        "birth_attempted": 1,
        "birth_success": 1,
        "parent_child_genome_distance_mean": 0.03,
        "parent_child_genome_distance_max": 0.05,
        "parent_child_genome_scalar_distance_mean": 0.04,
        "parent_child_genome_memory_field_delta": 0.0,
        "parent_child_genome_prediction_field_delta": 0.01,
        "parent_child_genome_module_gate_delta": 0.0,
        "mutation_weather_enabled": 1,
        "mutation_weather_profile": "safe_pulse_v0",
        "mutation_weather_active": 1,
        "mutation_weather_event_id": "fixture:safe_pulse_v0:g1:x3",
        "mutation_weather_generation": 1,
        "mutation_weather_severity": "fixture",
        "mutation_weather_scalar_multiplier": 3.0,
        "mutation_weather_count_multiplier": 1.0,
        "mutation_weather_structural_enabled": 0,
        "mutation_weather_affected_offspring": 1,
        "child_mutation_weather_active": 1,
        "child_mutation_weather_event_id": "fixture:safe_pulse_v0:g1:x3",
        "child_mutation_weather_scalar_multiplier": 3.0,
    }
    agent = SimpleNamespace(
        agent_id=1,
        alive=True,
        stats=SimpleNamespace(lineage_id="L00000001"),
    )

    writer.write_generation(
        generation=1,
        pre_turnover_snapshot={
            "alive_before_selection_counts": {"L00000001": 1},
            "eligible_parent_counts": {"L00000001": 1},
            "selected_parent_counts": {"L00000001": 1},
            "successful_birth_counts": {"L00000001": 1},
            "birth_contexts": [event],
            "birth_event_contexts": [event],
        },
        post_refill_agents=[agent],
        turnover_snapshot={"pre_cull_alive_counts": {"L00000001": 1}},
        write_events=True,
        event_schema_version=4,
    )

    events = [
        json.loads(line)
        for line in (tmp_path / "metrics" / "lineage_reproduction_funnel_events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    success = [row for row in events if row["event_type"] == "birth_success"][0]
    summary = json.loads((tmp_path / "metrics" / "lineage_reproduction_funnel_run_summary.json").read_text())

    assert success["mutation_weather_enabled"] == 1
    assert success["child_mutation_weather_active"] == 1
    assert success["child_mutation_weather_scalar_multiplier"] == pytest.approx(3.0)
    assert summary["weather_active_generation_count"] == 1
    assert summary["weather_event_count"] == 1
    assert summary["offspring_affected_by_weather_count"] == 1
    assert summary["parent_child_genome_distance_mean_weather_on"] == pytest.approx(0.03)
    assert summary["prediction_field_delta_events_weather_on"] == 1
    assert summary["module_gate_delta_events_weather_on"] == 0
