from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from native_runtime.flat_population_reporting import native_flat_population_reporting_records
from native_runtime.public_metrics_overlay import BEHAVIOR_ROW_FIELDS, public_metrics_overlay
from simulation.metrics_pkg import cognitive_status
from tests.perf.test_eiel_native_flat_reporting_projection import (
    _behavior_rows,
    _configs,
    _owner_projection,
    _reference_agents,
)


def _scope_fixture(
    agent_ids: list[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any, Any]:
    ids = list(agent_ids or [30, 10, 20])
    projection = _owner_projection(ids)
    behavior_rows = _behavior_rows(ids, selected_ids={ids[0]})
    env_cfg, sim_cfg = _configs()
    sim_cfg.lineage_observability_enable = False
    return projection, behavior_rows, env_cfg, sim_cfg


def _records(
    projection: dict[str, Any],
    behavior_rows: Any,
    env_cfg: Any,
    sim_cfg: Any,
):
    return native_flat_population_reporting_records(
        tensors=projection["tensors"],
        behavior_rows=behavior_rows,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metadata=projection["metadata"],
    )


def test_truthful_provenance_preserves_canonical_records_and_reducer_output() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _scope_fixture()
    projection["tensors"]["offspring_count"][0] = 2

    records = _records(projection, behavior_rows, env_cfg, sim_cfg)
    expected_rows = cognitive_status._agent_records(
        _reference_agents(projection),
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )
    reduced = cognitive_status.compute_cognitive_status_metrics_from_records(
        records,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        generation=4,
    )
    overlay = public_metrics_overlay(
        {"behavior_rows": behavior_rows},
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )

    assert tuple(map(dict, records.rows)) == tuple(expected_rows)
    assert {key: overlay[key] for key in reduced} == dict(reduced)
    assert reduced["reproduced_maturity_age_mean"] == 20.0
    assert {row["maturity_age"] for row in records.rows} == {20.0}


@pytest.mark.parametrize(
    "field",
    [
        "world",
        "native_runtime_fixed_policy_family",
        "generation",
        "capacity",
        "alive_count",
        "developmental_maturity_age",
        "policy_energy_observation_contract_schema_version",
    ],
)
def test_required_scope_locked_projection_metadata_is_explicit(field: str) -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _scope_fixture()
    projection["metadata"].pop(field)

    with pytest.raises(ValueError, match=rf"required metadata.*{field}|{field}.*required metadata"):
        _records(projection, behavior_rows, env_cfg, sim_cfg)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("world", "patch", "world=flat"),
        ("native_runtime_fixed_policy_family", "gru", "fixed-MLP"),
        ("generation", True, "metadata.generation.*integer"),
        ("generation", -1, "metadata.generation.*nonnegative"),
        ("capacity", 99, "capacity disagrees"),
        ("alive_count", 0, "alive_count disagrees"),
        ("developmental_maturity_age", 21, "maturity_age authority disagrees"),
        (
            "policy_energy_observation_contract_schema_version",
            "unsupported",
            "unsupported policy energy observation schema",
        ),
    ],
)
def test_invalid_scope_locked_projection_metadata_fails_closed(
    field: str,
    value: Any,
    message: str,
) -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _scope_fixture()
    projection["metadata"][field] = value

    with pytest.raises(ValueError, match=message):
        _records(projection, behavior_rows, env_cfg, sim_cfg)


def test_maturity_age_is_shared_config_metadata_not_a_resident_tensor() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _scope_fixture()
    records = _records(projection, behavior_rows, env_cfg, sim_cfg)

    assert "maturity_age" not in projection["tensors"]
    assert {row["maturity_age"] for row in records.rows} == {20.0}


@pytest.mark.parametrize("field", BEHAVIOR_ROW_FIELDS)
def test_every_behavior_field_is_required(field: str) -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _scope_fixture([30])
    behavior_rows[0].pop(field)

    with pytest.raises(ValueError, match=rf"missing fields.*{field}"):
        _records(projection, behavior_rows, env_cfg, sim_cfg)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generation", 3, "behavior/projection generation mismatch"),
        ("agent_id", "30", "agent_id.*integer"),
        ("cohort", "", "cohort.*non-empty string"),
        ("move_n", -1, "move_n.*nonnegative"),
        ("fitness", float("nan"), "fitness.*finite"),
        ("alive_end", 2, "alive_end.*0 or 1"),
    ],
)
def test_invalid_behavior_required_fields_fail_closed(
    field: str,
    value: Any,
    message: str,
) -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _scope_fixture([30])
    behavior_rows[0][field] = value

    with pytest.raises(ValueError, match=message):
        _records(projection, behavior_rows, env_cfg, sim_cfg)


def test_duplicate_and_malformed_behavior_rows_fail_closed() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _scope_fixture([30])
    duplicate = [behavior_rows[0], copy.deepcopy(behavior_rows[0])]
    with pytest.raises(ValueError, match="duplicate agent_id"):
        _records(projection, duplicate, env_cfg, sim_cfg)

    with pytest.raises(ValueError, match="is not a row"):
        _records(projection, ["malformed"], env_cfg, sim_cfg)

    short_row = [behavior_rows[0][field] for field in BEHAVIOR_ROW_FIELDS[:-1]]
    with pytest.raises(ValueError, match="must have 21 fields"):
        _records(projection, [short_row], env_cfg, sim_cfg)


def test_metrics_generation_must_match_projection_and_behavior_generation() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _scope_fixture()

    with pytest.raises(ValueError, match="projection/metrics generation mismatch"):
        public_metrics_overlay(
            {"behavior_rows": behavior_rows},
            checkpoint_projection=projection,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            metrics_generation=5,
        )


def test_completed_generation_accepts_only_explicit_next_boundary_projection() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _scope_fixture()
    projection["metadata"].update(
        {
            "generation": 5,
            "tick": 0,
            "agent_grid_runner_phase": "generation_boundary",
        }
    )

    overlay = public_metrics_overlay(
        {"behavior_rows": behavior_rows},
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )

    assert overlay["alive"] == 3


@pytest.mark.parametrize(
    "metadata_update",
    [
        {},
        {"tick": 1, "agent_grid_runner_phase": "partial_generation"},
        {"tick": 0, "agent_grid_runner_phase": "partial_generation"},
    ],
)
def test_next_generation_projection_without_exact_boundary_phase_fails_closed(
    metadata_update: dict[str, Any],
) -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _scope_fixture()
    projection["metadata"]["generation"] = 5
    projection["metadata"].update(metadata_update)

    with pytest.raises(ValueError, match="boundary metadata|not at a generation boundary"):
        public_metrics_overlay(
            {"behavior_rows": behavior_rows},
            checkpoint_projection=projection,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            metrics_generation=4,
        )


@pytest.mark.parametrize(
    "missing_input",
    ["survival_steps", "pos_y", "pos_x", "biome_grid"],
)
def test_scope_locked_biome_reporting_requires_complete_authority(
    missing_input: str,
) -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _scope_fixture([30])
    projection["tensors"].pop(missing_input)

    with pytest.raises(ValueError, match=rf"incomplete authority.*{missing_input}"):
        public_metrics_overlay(
            {"behavior_rows": behavior_rows},
            checkpoint_projection=projection,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            metrics_generation=4,
        )


@pytest.mark.parametrize(
    "missing_input",
    ["survival_steps", "pos_y", "pos_x", "biome_grid"],
)
def test_non_scope_biome_reporting_preserves_legacy_safe_fallback(
    missing_input: str,
) -> None:
    tensors: dict[str, Any] = {
        "alive_mask": [True],
        "slot_to_agent_id": [30],
        "survival_steps": [3],
        "pos_y": [0],
        "pos_x": [1],
        "biome_grid": [[0, 1]],
    }
    tensors.pop(missing_input)

    overlay = public_metrics_overlay(checkpoint_projection={"tensors": tensors})

    assert overlay["biome_safe_frac_alive"] == 1.0
    assert overlay["biome_wild_frac_alive"] == 0.0


def test_scope_locked_biome_authority_must_cover_every_live_slot() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _scope_fixture([30, 10])
    projection["tensors"]["pos_x"] = torch.tensor([0], dtype=torch.int32)

    with pytest.raises(ValueError, match="does not cover live slot"):
        public_metrics_overlay(
            {"behavior_rows": behavior_rows},
            checkpoint_projection=projection,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            metrics_generation=4,
        )


def test_scope_locked_biome_reporting_preserves_zero_age_timing() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _scope_fixture([30, 10])
    projection["tensors"]["survival_steps"] = torch.tensor([0, 1], dtype=torch.int32)
    projection["tensors"]["pos_y"] = torch.tensor([0, 0], dtype=torch.int32)
    projection["tensors"]["pos_x"] = torch.tensor([1, 1], dtype=torch.int32)
    projection["tensors"]["biome_grid"] = torch.tensor([[0, 1]], dtype=torch.int32)

    overlay = public_metrics_overlay(
        {"behavior_rows": behavior_rows},
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )

    assert overlay["biome_safe_frac_alive"] == 0.5
    assert overlay["biome_wild_frac_alive"] == 0.5
