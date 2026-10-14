from __future__ import annotations

import copy
import math
import pickle
from types import SimpleNamespace
from typing import Any

import pytest

from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from env.env_config import EnvConfig
from simulation.metrics_pkg import cognitive_status
from simulation.metrics_pkg.population_reporting import (
    FLAT_REPORTING_FIELD_DTYPES,
    FLAT_REPORTING_REQUIRED_FIELDS,
    FlatPopulationReportingRecordsV1,
)
from simulation.runner_pkg.runner_config import SimConfig
from tools.performance import build_flat_resident_reporting_contract


_PYTHON_TYPES_BY_DTYPE = {
    "bool": bool,
    "float64": float,
    "int64": int,
    "string": str,
    "tuple": tuple,
}


def _reference_row(agent_id: int = 1) -> dict[str, Any]:
    genome = Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=30.0,
        mutation_rate=0.03,
        action_temperature=1.0,
        use_gru=True,
        memory_dim=16,
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.1,
        pred_lr=0.08,
        developmental_program=canonicalize_developmental_program(None),
    )
    stats = SimpleNamespace(
        founder_id=agent_id,
        lineage_id=f"L{agent_id:08d}",
        offspring_count=1,
        pred_err_count=2,
        pred_err_sum=0.25 * float(agent_id),
        prediction_update_count=0,
        prediction_state_delta_abs_sum=0.0,
        prediction_state_delta_abs_max=0.0,
    )
    agent = SimpleNamespace(
        agent_id=agent_id,
        alive=True,
        energy=40.0 + float(agent_id),
        health=12.0,
        drives=[0.4, 0.1, 0.2, 0.3],
        genome=genome,
        stats=stats,
    )
    rows = cognitive_status._agent_records(
        [agent],
        env_cfg=EnvConfig(width=8, height=8, view_radius=3),
        sim_cfg=SimConfig(seed=17),
    )
    assert len(rows) == 1
    return rows[0]


def _validate(
    row: dict[str, Any],
    *,
    selected_parent_rows: tuple[dict[str, Any], ...] = (),
) -> FlatPopulationReportingRecordsV1:
    return FlatPopulationReportingRecordsV1(
        rows=(row,),
        selected_parent_rows=selected_parent_rows,
    )


def test_dtype_contract_matches_generated_schema_and_reference_row_exactly() -> None:
    row = _reference_row()

    assert tuple(FLAT_REPORTING_FIELD_DTYPES) == FLAT_REPORTING_REQUIRED_FIELDS
    assert tuple(row) == FLAT_REPORTING_REQUIRED_FIELDS
    for field_name, dtype in FLAT_REPORTING_FIELD_DTYPES.items():
        assert type(row[field_name]) is _PYTHON_TYPES_BY_DTYPE[dtype]

    _, generated_schema = build_flat_resident_reporting_contract._build_payloads()
    generated_dtypes = {
        field["key"]: field["dtype"] for field in generated_schema["fields"]
    }
    assert generated_dtypes == {
        field_name: FLAT_REPORTING_FIELD_DTYPES[field_name]
        for field_name in generated_dtypes
    }

    records = _validate(row, selected_parent_rows=(copy.deepcopy(row),))
    assert pickle.dumps(dict(records.rows[0]), protocol=5) == pickle.dumps(
        row,
        protocol=5,
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("capacity", None),
        ("capacity", True),
        ("capacity", 1),
        ("capacity", "1.0"),
        ("capacity", "nan"),
        ("capacity", math.nan),
        ("capacity", math.inf),
        ("capacity", -math.inf),
        ("agent_id", None),
        ("agent_id", False),
        ("agent_id", 1.0),
        ("agent_id", "1"),
    ],
)
def test_numeric_fields_reject_substitutions_and_nonfinite_values(
    field_name: str,
    invalid_value: object,
) -> None:
    row = _reference_row()
    row[field_name] = invalid_value

    with pytest.raises(ValueError, match=rf"rows\[0\]\.{field_name}:"):
        _validate(row)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("memory_enabled", 1),
        ("memory_enabled", "true"),
        ("dev_stage", False),
        ("dev_stage", ("default",)),
        ("sensor_channel_ids", ["self_energy"]),
        ("sensor_channel_ids", "self_energy"),
        ("sensor_channel_ids", (None,)),
        ("strategy", (16, 1, math.nan)),
    ],
)
def test_bool_string_and_tuple_fields_require_declared_primitive_types(
    field_name: str,
    invalid_value: object,
) -> None:
    row = _reference_row()
    row[field_name] = invalid_value

    with pytest.raises(ValueError, match=rf"rows\[0\]\.{field_name}"):
        _validate(row)


def test_unexpected_fields_are_rejected_even_when_all_rows_contain_them() -> None:
    row = _reference_row()
    row["capacity_copy"] = row["capacity"]

    with pytest.raises(ValueError, match=r"rows\[0\]\.capacity_copy: field set differs"):
        _validate(row)


def test_selected_parent_row_must_be_value_identical_to_population_row() -> None:
    row = _reference_row()
    selected_parent_row = copy.deepcopy(row)
    selected_parent_row["capacity"] += 1.0

    with pytest.raises(
        ValueError,
        match=r"selected_parent_rows\[0\]: values differ from rows agent_id 1",
    ):
        _validate(row, selected_parent_rows=(selected_parent_row,))
