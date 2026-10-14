from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from typing import Any

import pytest

from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from env.env_config import EnvConfig
from simulation.metrics_pkg import cognitive_status
from simulation.metrics_pkg.population_reporting import (
    FLAT_POPULATION_REPORTING_SCHEMA_VERSION,
    FLAT_REPORTING_REQUIRED_FIELDS,
    FlatPopulationReportingRecordsV1,
    validate_flat_population_reporting_records,
)
from simulation.runner_pkg.runner_config import SimConfig
from simulation.output_cost_timing import (
    begin_flat_output_cost_timing,
    finalize_flat_output_cost_timing,
)


EXPECTED_SCHEMA_VERSION = "flat_population_reporting_v1"


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


def _canonical_rows(agents: list[SimpleNamespace]) -> list[dict[str, Any]]:
    env_cfg, sim_cfg = _configs()
    rows = cognitive_status._agent_records(agents, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert rows
    return rows


def _row(agent_id: int) -> dict[str, Any]:
    row = _canonical_rows([_agent(agent_id)])[0]
    assert row["agent_id"] == agent_id
    assert {"agent", "phenotype", "ledger"}.isdisjoint(row)
    return row


def _records(
    rows: list[dict[str, Any]],
    *,
    selected_parent_rows: list[dict[str, Any]] | None = None,
) -> FlatPopulationReportingRecordsV1:
    records = FlatPopulationReportingRecordsV1(
        rows=rows,
        selected_parent_rows=selected_parent_rows or (),
    )
    validate_flat_population_reporting_records(records)
    return records


def _validate(
    rows: list[dict[str, Any]],
    *,
    selected_parent_rows: list[dict[str, Any]] | None = None,
) -> None:
    records = FlatPopulationReportingRecordsV1(
        rows=rows,
        selected_parent_rows=selected_parent_rows or (),
    )
    validate_flat_population_reporting_records(records)


def test_schema_constant_and_empty_records_are_exact() -> None:
    records = _records([])

    assert FLAT_POPULATION_REPORTING_SCHEMA_VERSION == EXPECTED_SCHEMA_VERSION
    assert records.schema_version == FLAT_POPULATION_REPORTING_SCHEMA_VERSION
    assert records.rows == ()
    assert records.selected_parent_rows == ()


def test_one_canonical_row_validates_without_schema_loss() -> None:
    source = _row(1)
    records = _records([source], selected_parent_rows=[copy.deepcopy(source)])

    assert tuple(source) == FLAT_REPORTING_REQUIRED_FIELDS
    assert len(records.rows) == 1
    assert dict(records.rows[0]) == source
    assert tuple(map(dict, records.selected_parent_rows)) == (source,)


def test_more_than_64_rows_have_no_hidden_batch_cap() -> None:
    template = _row(1)
    rows = []
    for agent_id in range(1, 66):
        row = copy.deepcopy(template)
        row["agent_id"] = agent_id
        rows.append(row)

    records = _records(
        rows,
        selected_parent_rows=[copy.deepcopy(rows[index]) for index in (0, 32, 64)],
    )

    assert len(records.rows) == 65
    assert tuple(row["agent_id"] for row in records.rows) == tuple(range(1, 66))


@pytest.mark.parametrize("agent_ids", [(1, 1), (2, 1)])
def test_duplicate_or_unsorted_agent_identity_is_rejected(agent_ids: tuple[int, int]) -> None:
    rows = [_row(agent_id) for agent_id in agent_ids]

    with pytest.raises(ValueError):
        _validate(rows)


def test_missing_required_field_is_rejected() -> None:
    row = _row(1)
    row.pop("capacity")

    with pytest.raises(ValueError):
        _validate([row])


def test_rows_with_unequal_field_sets_are_rejected() -> None:
    first = _row(1)
    second = _row(2)
    second["capacity_copy"] = second.pop("capacity")

    assert len(first) == len(second)
    assert set(first) != set(second)
    with pytest.raises(ValueError):
        _validate([first, second])


@pytest.mark.parametrize("forbidden_field", ["agent", "phenotype", "ledger"])
def test_live_object_fields_are_forbidden(forbidden_field: str) -> None:
    row = _row(1)
    row[forbidden_field] = object()

    with pytest.raises(ValueError):
        _validate([row])


def test_selected_parent_rows_must_be_a_unique_population_subset() -> None:
    rows = [_row(1), _row(2), _row(3)]

    _validate(rows, selected_parent_rows=[copy.deepcopy(rows[0]), copy.deepcopy(rows[2])])
    with pytest.raises(ValueError):
        _validate(rows, selected_parent_rows=[_row(4)])
    with pytest.raises(ValueError):
        _validate(rows, selected_parent_rows=[copy.deepcopy(rows[0]), copy.deepcopy(rows[0])])


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_non_finite_numeric_values_are_rejected(non_finite: float) -> None:
    row = _row(1)
    row["capacity"] = non_finite

    with pytest.raises(ValueError):
        _validate([row])


def test_records_are_immutable_and_deep_copy_source_rows() -> None:
    source_rows = [_row(1)]
    expected = copy.deepcopy(source_rows[0])
    selected_parent_rows = [copy.deepcopy(source_rows[0])]
    records = _records(source_rows, selected_parent_rows=selected_parent_rows)

    source_rows[0]["capacity"] = -999.0
    source_rows.append(_row(2))
    selected_parent_rows[0]["capacity"] = -998.0
    assert len(records.rows) == 1
    assert dict(records.rows[0]) == expected
    assert dict(records.selected_parent_rows[0]) == expected

    with pytest.raises(TypeError):
        records.rows[0]["capacity"] = -1.0  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        records.rows = ()  # type: ignore[misc]


def test_record_reducer_exactly_matches_agent_reducer_for_reference_fixture() -> None:
    reduce_records = getattr(cognitive_status, "compute_cognitive_status_metrics_from_records", None)
    if reduce_records is None:
        pytest.skip("record reducer is not present in this checkout")

    env_cfg, sim_cfg = _configs()
    agents = [_agent(1, memory_dim=16), _agent(2, memory_dim=32)]
    selected_parents = [agents[1]]
    rows = cognitive_status._agent_records(agents, env_cfg=env_cfg, sim_cfg=sim_cfg)
    selected_parent_rows = cognitive_status._agent_records(
        selected_parents,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )
    records = _records(rows, selected_parent_rows=selected_parent_rows)

    expected = cognitive_status.compute_cognitive_status_metrics(
        agents,
        selected_parent_agents=selected_parents,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        generation=3,
    )
    actual = reduce_records(
        records.rows,
        selected_parent_records=records.selected_parent_rows,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        generation=3,
    )

    assert actual == expected


def test_reference_reporting_entry_point_records_projection_and_reducer_timing(tmp_path) -> None:
    env_cfg, sim_cfg = _configs()
    agents = [_agent(1), _agent(2)]
    timing = begin_flat_output_cost_timing(tmp_path / "reference-output-cost.json")
    assert timing is not None

    cognitive_status.compute_cognitive_status_metrics(
        agents,
        selected_parent_agents=[agents[1]],
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        generation=0,
    )
    for context in (
        timing.simulation_phase,
        timing.serialization_phase,
        timing.write_phase,
        timing.flush_phase,
    ):
        with context():
            pass
    payload = finalize_flat_output_cost_timing(timing)

    assert payload is not None
    assert payload["complete"] is True
    assert payload["call_counts"]["reporting_projection"] == 1
    assert payload["call_counts"]["reporting_reducer"] == 1
