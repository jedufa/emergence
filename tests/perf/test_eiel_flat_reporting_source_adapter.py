from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import pytest

from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from env.env_config import EnvConfig
from simulation.metrics_pkg import cognitive_status
from simulation.env_hotpath_expression_diagnostics import (
    EnvHotpathExpressionDiagnostics,
    reset_current_diagnostics,
    set_current_diagnostics,
)
from simulation.metrics_pkg.population_reporting import FlatPopulationReportingRecordsV1
from simulation.runner_pkg.runner_config import SimConfig


def _genome(*, memory_dim: int = 16) -> Genome:
    return Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=30.0,
        mutation_rate=0.03,
        action_temperature=1.0,
        use_gru=bool(memory_dim),
        memory_dim=int(memory_dim),
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.1,
        pred_lr=0.08,
        developmental_program=canonicalize_developmental_program(None),
    )


def _agent(agent_id: int, *, memory_dim: int = 16) -> SimpleNamespace:
    stats = SimpleNamespace(
        founder_id=int(agent_id),
        lineage_id=f"L{agent_id:08d}",
        offspring_count=int(agent_id % 2),
        pred_err_count=2,
        pred_err_sum=0.25 * float(agent_id),
        prediction_update_count=0,
        prediction_state_delta_abs_sum=0.0,
        prediction_state_delta_abs_max=0.0,
    )
    return SimpleNamespace(
        agent_id=int(agent_id),
        alive=True,
        energy=40.0 + float(agent_id),
        health=12.0,
        drives=[0.4, 0.1, 0.2, 0.3],
        genome=_genome(memory_dim=memory_dim),
        stats=stats,
    )


def _configs() -> tuple[EnvConfig, SimConfig]:
    return EnvConfig(width=8, height=8, view_radius=3), SimConfig(seed=17)


def _plain_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in rows]


def _json_bytes(rows: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        rows,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")


def test_source_adapter_matches_existing_agent_records_by_value_and_bytes() -> None:
    env_cfg, sim_cfg = _configs()
    agents = [_agent(1, memory_dim=16), _agent(2, memory_dim=32)]
    expected = cognitive_status._agent_records(
        agents,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )

    records = cognitive_status.flat_population_reporting_records_from_sources(
        agents,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )
    actual = _plain_rows(records.rows)

    assert actual == expected
    assert _json_bytes(actual) == _json_bytes(expected)


def test_source_adapter_preserves_selected_parent_rows() -> None:
    env_cfg, sim_cfg = _configs()
    agents = [_agent(1, memory_dim=16), _agent(2, memory_dim=32)]
    expected_parent_rows = cognitive_status._agent_records(
        [agents[1]],
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )

    records = cognitive_status.flat_population_reporting_records_from_sources(
        agents,
        selected_parent_sources=[agents[1]],
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )

    assert [row["agent_id"] for row in records.rows] == [1, 2]
    assert _plain_rows(records.selected_parent_rows) == expected_parent_rows
    assert records.selected_parent_rows[0] == records.rows[1]


def test_escap8_reporting_keeps_flat_v1_frozen_and_uses_side_context() -> None:
    env_cfg, sim_cfg = _configs()
    sim_cfg.structural_sustainable_growth_enable = True
    sim_cfg.trait_capacity_saturation_reporting_enable = True
    source = _agent(1)
    source.structural_live_module_telemetry = {
        "executing_modules": 3,
        "dormant_modules": 2,
        "nonzero_output_modules": 1,
        "neutral_duplicate_modules": 2,
        "redundant_modules": 2,
        "prunable_modules": 1,
        "structural_cost_total": 55.0,
    }
    source.genome.escap8_growth_attempt_count = 4
    source.genome.escap8_blocked_duplication_attempt_count = 2
    source.genome.escap8_blocked_prune_attempt_count = 1
    source.genome.escap8_cap_contact_count = 2
    source.genome.escap8_cap_pressure_streak = 3
    source.genome.escap8_specialization_metrics_json = '{"specialized":true}'
    source.genome.modules = [{"module_id": f"m{index}"} for index in range(5)]
    source.genome.connections = []

    records = cognitive_status.flat_population_reporting_records_from_sources(
        [source], env_cfg=env_cfg, sim_cfg=sim_cfg
    )
    assert "executing_modules" not in records.rows[0]
    assert "escap8_growth_attempt_count" not in records.rows[0]

    metrics = cognitive_status.compute_cognitive_status_metrics(
        [source], env_cfg=env_cfg, sim_cfg=sim_cfg, generation=3
    )
    structural = json.loads(
        metrics["trait_capacity_saturation_structural_payload_json"]
    )
    summary = json.loads(metrics["trait_capacity_saturation_summary_json"])
    growth = summary["capacity_headroom"]["sustainable_growth"]
    assert structural["executing_module_count_max"] == pytest.approx(3.0)
    assert structural["dormant_module_count_max"] == pytest.approx(2.0)
    assert structural["escap8_growth_attempt_count"] == pytest.approx(4.0)
    assert structural["escap8_cap_contact_count"] == pytest.approx(2.0)
    assert structural["specialization_metrics_record_count"] == 1
    assert growth["blocked_duplication_attempts"] == 2
    assert growth["blocked_prune_attempts"] == 1
    assert growth["cap_pressure_streak"] == 3
    assert growth["structural_cost_max"] == pytest.approx(55.0)
    assert growth["bloat_classification"]["metrics"]["cap_contact_count"] == 2
    assert growth["bloat_classification"]["metrics"]["causal_activity_measured"] is False
    assert (
        growth["bloat_classification"]["causal_activity_basis"]
        == "probe_required_not_classified_live"
    )


def test_source_adapter_returns_an_immutable_source_owned_snapshot() -> None:
    env_cfg, sim_cfg = _configs()
    source = _agent(1)
    records = cognitive_status.flat_population_reporting_records_from_sources(
        [source],
        selected_parent_sources=[source],
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )
    expected_row = dict(records.rows[0])

    source.energy = -100.0
    source.stats.pred_err_sum = 999.0

    assert isinstance(records, FlatPopulationReportingRecordsV1)
    assert dict(records.rows[0]) == expected_row
    assert dict(records.selected_parent_rows[0]) == expected_row
    assert isinstance(records.rows[0]["strategy"], tuple)
    with pytest.raises(TypeError):
        records.rows[0]["energy"] = -200.0  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        records.rows = ()  # type: ignore[misc]


@pytest.mark.parametrize("sources", [None, (), iter(())])
def test_source_adapter_accepts_empty_sources(sources: Iterable[Any] | None) -> None:
    records = cognitive_status.flat_population_reporting_records_from_sources(
        sources,
        selected_parent_sources=None,
    )

    assert isinstance(records, FlatPopulationReportingRecordsV1)
    assert records.rows == ()
    assert records.selected_parent_rows == ()


def test_reducer_accepts_canonical_records_without_observing_later_source_mutation() -> None:
    env_cfg, sim_cfg = _configs()
    source = _agent(1)
    records = cognitive_status.flat_population_reporting_records_from_sources(
        [source],
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )
    row_before = dict(records.rows[0])
    metrics_before = cognitive_status.compute_cognitive_status_metrics_from_records(
        records,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        generation=3,
    )

    source.energy = 4000.0
    source.stats.pred_err_sum = 500.0
    fresh_records = cognitive_status.flat_population_reporting_records_from_sources(
        [source],
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )
    metrics_after = cognitive_status.compute_cognitive_status_metrics_from_records(
        records,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        generation=3,
    )

    assert dict(fresh_records.rows[0]) != row_before
    assert dict(records.rows[0]) == row_before
    assert metrics_after == metrics_before


def test_source_adapter_does_not_mutate_active_expression_diagnostics(tmp_path) -> None:
    env_cfg, sim_cfg = _configs()
    source = _agent(1)
    diagnostics = EnvHotpathExpressionDiagnostics(
        run_dir=str(tmp_path),
        decomposition_enabled=True,
        phenotype_decomposition_enabled=True,
    )
    token = set_current_diagnostics(diagnostics)
    try:
        records = cognitive_status.flat_population_reporting_records_from_sources(
            [source],
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
        )
    finally:
        reset_current_diagnostics(token)

    assert len(records.rows) == 1
    assert diagnostics.decomp_counts == {}
    assert diagnostics.phenotype_counts == {}
    assert diagnostics.event_count == 0
