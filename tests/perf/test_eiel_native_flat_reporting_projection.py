from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from agents.modular_genome_v1 import canon_float
from env.env_config import EnvConfig
from native_runtime.flat_population_reporting import native_flat_population_reporting_records
from native_runtime.public_metrics_overlay import BEHAVIOR_ROW_FIELDS
from simulation.metrics_pkg import cognitive_status
from simulation.metrics_pkg.population_reporting import FlatPopulationReportingRecordsV1
from simulation.runner_pkg.runner_config import SimConfig


_GENOME_INT_TENSORS = {
    "genome_hidden_size": 64,
    "genome_memory_dim": 0,
    "genome_sensor_channel_mask": 1,
    "genome_sensor_radius": 1,
    "genome_introspection_channel_mask": 32767,
    "genome_self_regulation_control_mask": 31,
    "genome_self_modification_target_mask": 31,
    "genome_directed_inheritance_channel_mask": 63,
}

_GENOME_FLOAT_TENSORS = {
    "genome_exploration_bias": 0.12345678,
    "genome_aggression_bias": -0.23456789,
    "genome_mutation_rate": 0.12345678,
    "genome_reproduction_threshold": 30.12345678,
    "genome_action_temperature": 1.12345678,
    "genome_drive_inertia": 0.87654321,
    "genome_drive_noise": 0.02345678,
    "genome_drive_logit_gain": 1.23456788,
    "genome_pred_lr": 0.08765432,
    "genome_drive_sp_hunger": 0.55432111,
    "genome_drive_sp_pain": 0.05432111,
    "genome_drive_sp_fatigue": 0.15432112,
    "genome_drive_sp_fear": 0.10432111,
    "genome_exp_prior_strength": 0.35432112,
    "genome_exp_prior_noise": 0.05432111,
    "genome_drive_birth_noise": 0.03432111,
}


def _configs() -> tuple[EnvConfig, SimConfig]:
    env_cfg = EnvConfig(width=8, height=8, view_radius=3)
    env_cfg.enable_biomes = False
    env_cfg.patch_world_enable = False
    env_cfg.reproduce_requires_mate = False
    sim_cfg = SimConfig(
        seed=17,
        device="cpu",
        bank_device="cpu",
        native_runtime_fixed_policy_family="mlp",
        native_runtime_cxx26_scope_lock_enable=True,
        force_prediction_disabled=True,
    )
    sim_cfg.enable_developmental_cognition = False
    sim_cfg.enable_evolvable_sensor_channel_mask = False
    sim_cfg.enable_evolvable_sensor_radius = False
    sim_cfg.genome_c1_enable = False
    sim_cfg.genome_c2_enable = False
    sim_cfg.chemistry_functional_coupling_enable = False
    sim_cfg.maintenance_enable = False
    return env_cfg, sim_cfg


def _owner_projection(agent_ids: list[int]) -> dict[str, Any]:
    count = len(agent_ids)
    slot = torch.arange(count, dtype=torch.int32)
    drives = torch.tensor(
        [
            [
                0.123456789 + 0.01 * float(index % 3),
                0.234567891,
                0.456789017,
                0.067890123 + 0.01 * float(index % 2),
            ]
            for index in range(count)
        ],
        dtype=torch.float32,
    )
    tensors: dict[str, torch.Tensor] = {
        "alive_mask": torch.ones(count, dtype=torch.bool),
        "slot_to_agent_id": torch.tensor(agent_ids, dtype=torch.int64),
        # The float32 mechanics energy is deliberately different from the
        # float64 policy-boundary authority used by public reporting.
        "energy": torch.arange(count, dtype=torch.float32) + 1.0,
        "policy_energy_observation": torch.arange(count, dtype=torch.float64)
        + 33.1234567890123,
        "health": torch.arange(count, dtype=torch.float32) * 0.125 + 10.0,
        "drives": drives,
        "survival_steps": (slot * 19 + 7).remainder(151).to(torch.int32),
        "pos_y": torch.zeros(count, dtype=torch.int32),
        "pos_x": torch.zeros(count, dtype=torch.int32),
        "biome_grid": torch.zeros((8, 8), dtype=torch.int32),
        "patch_habitat_lineage_id": torch.tensor(
            [700 + int(agent_id) % 5 for agent_id in agent_ids],
            dtype=torch.int32,
        ),
        "lineage_founder_id": torch.tensor(
            [900 + int(agent_id) % 3 for agent_id in agent_ids],
            dtype=torch.int32,
        ),
        "offspring_count": slot.remainder(3).to(torch.int32),
        "pred_err_count": slot.remainder(3).to(torch.int32),
        "pred_err_sum": slot.to(torch.float32) * 0.125,
        "policy_use_gru": torch.zeros(count, dtype=torch.bool),
    }
    tensors.update(
        {
            name: torch.full((count,), value, dtype=torch.int32)
            for name, value in _GENOME_INT_TENSORS.items()
        }
    )
    tensors.update(
        {
            name: torch.full((count,), value, dtype=torch.float32)
            for name, value in _GENOME_FLOAT_TENSORS.items()
        }
    )
    return {
        "schema_version": "cxxbackbone10_checkpoint_projection_v1",
        "metadata": {
            "world": "flat",
            "native_runtime_fixed_policy_family": "mlp",
            "generation": 4,
            "capacity": count,
            "alive_count": count,
            "developmental_maturity_age": 20,
            "policy_energy_observation_contract_schema_version": (
                "eiel_policy_energy_observation_shadow_v1"
            ),
            "policy_energy_observation_authority": (
                "float64_python_visible_energy_at_flat_policy_boundary"
            ),
            "policy_energy_observation_live_finite_required": True,
        },
        "tensors": tensors,
    }


def _behavior_row(agent_id: int, *, kept_next: int = 0, offspring: int = 999) -> dict[str, Any]:
    values: dict[str, Any] = {name: 0 for name in BEHAVIOR_ROW_FIELDS}
    values.update(
        {
            "generation": 4,
            "agent_id": int(agent_id),
            "cohort": "start",
            "lifespan_steps": 999,
            "offspring": int(offspring),
            "alive_end": 1,
            "eligible_for_selection": int(not kept_next),
            "kept_next": int(kept_next),
        }
    )
    return values


def _behavior_rows(agent_ids: list[int], *, selected_ids: set[int] | None = None) -> list[dict[str, Any]]:
    selected = selected_ids or set()
    return [
        _behavior_row(agent_id, kept_next=int(agent_id in selected))
        for agent_id in reversed(agent_ids)
    ]


def _records(
    projection: dict[str, Any],
    behavior_rows: list[dict[str, Any]],
) -> FlatPopulationReportingRecordsV1:
    env_cfg, sim_cfg = _configs()
    return native_flat_population_reporting_records(
        tensors=projection["tensors"],
        behavior_rows=behavior_rows,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metadata=projection["metadata"],
    )


def _genome_from_slot(tensors: dict[str, torch.Tensor], slot: int) -> Genome:
    def integer(name: str) -> int:
        return int(tensors[name][slot].item())

    def scalar(name: str) -> float:
        field = name.removeprefix("genome_")
        return canon_float(tensors[name][slot].item(), field=field)

    return Genome(
        hidden_size=integer("genome_hidden_size"),
        exploration_bias=scalar("genome_exploration_bias"),
        aggression_bias=scalar("genome_aggression_bias"),
        reproduction_threshold=scalar("genome_reproduction_threshold"),
        mutation_rate=scalar("genome_mutation_rate"),
        action_temperature=scalar("genome_action_temperature"),
        use_gru=bool(tensors["policy_use_gru"][slot].item()),
        memory_dim=integer("genome_memory_dim"),
        drive_inertia=scalar("genome_drive_inertia"),
        drive_noise=scalar("genome_drive_noise"),
        drive_logit_gain=scalar("genome_drive_logit_gain"),
        pred_lr=scalar("genome_pred_lr"),
        sensor_channel_mask=integer("genome_sensor_channel_mask"),
        sensor_radius=integer("genome_sensor_radius"),
        introspection_channel_mask=integer("genome_introspection_channel_mask"),
        self_regulation_control_mask=integer("genome_self_regulation_control_mask"),
        self_modification_target_mask=integer("genome_self_modification_target_mask"),
        directed_inheritance_channel_mask=integer("genome_directed_inheritance_channel_mask"),
        drive_sp_hunger=scalar("genome_drive_sp_hunger"),
        drive_sp_pain=scalar("genome_drive_sp_pain"),
        drive_sp_fatigue=scalar("genome_drive_sp_fatigue"),
        drive_sp_fear=scalar("genome_drive_sp_fear"),
        exp_prior_strength=scalar("genome_exp_prior_strength"),
        exp_prior_noise=scalar("genome_exp_prior_noise"),
        drive_birth_noise=scalar("genome_drive_birth_noise"),
        developmental_program=canonicalize_developmental_program(None),
    )


def _reference_agents(projection: dict[str, Any]) -> list[SimpleNamespace]:
    tensors = projection["tensors"]
    assert isinstance(tensors, dict)
    agents = []
    for slot, raw_agent_id in enumerate(tensors["slot_to_agent_id"]):
        if not bool(tensors["alive_mask"][slot].item()):
            continue
        agent_id = int(raw_agent_id.item())
        pred_count = int(tensors["pred_err_count"][slot].item())
        habitat_lineage_id = int(tensors["patch_habitat_lineage_id"][slot].item())
        founder_id = int(tensors["lineage_founder_id"][slot].item())
        stats = SimpleNamespace(
            survival_steps=int(tensors["survival_steps"][slot].item()),
            offspring_count=int(tensors["offspring_count"][slot].item()),
            patch_habitat_lineage_id=habitat_lineage_id,
            lineage_id=habitat_lineage_id,
            founder_lineage_id=founder_id,
            founder_id=founder_id,
            pred_err_count=pred_count,
            pred_err_sum=float(tensors["pred_err_sum"][slot].item()),
        )
        agents.append(
            SimpleNamespace(
                agent_id=agent_id,
                alive=True,
                energy=float(tensors["policy_energy_observation"][slot].item()),
                health=float(tensors["health"][slot].item()),
                drives=[float(value.item()) for value in tensors["drives"][slot]],
                genome=_genome_from_slot(tensors, slot),
                stats=stats,
            )
        )
    return sorted(agents, key=lambda agent: int(agent.agent_id))


def _assert_projection_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    assert actual["schema_version"] == expected["schema_version"]
    assert actual["metadata"] == expected["metadata"]
    assert actual["tensors"].keys() == expected["tensors"].keys()
    for name, tensor in actual["tensors"].items():
        assert torch.equal(tensor, expected["tensors"][name]), name


def test_more_than_64_live_rows_are_sorted_by_agent_id_and_select_parents_via_kept_next() -> None:
    agent_ids = list(reversed(range(1000, 1065)))
    selected_ids = {1000, 1032, 1064}
    records = _records(
        _owner_projection(agent_ids),
        _behavior_rows(agent_ids, selected_ids=selected_ids),
    )

    assert isinstance(records, FlatPopulationReportingRecordsV1)
    assert len(records.rows) == 65
    assert tuple(row["agent_id"] for row in records.rows) == tuple(range(1000, 1065))
    assert tuple(row["agent_id"] for row in records.selected_parent_rows) == tuple(
        sorted(selected_ids)
    )
    population_by_id = {row["agent_id"]: row for row in records.rows}
    assert all(
        dict(parent) == dict(population_by_id[parent["agent_id"]])
        for parent in records.selected_parent_rows
    )


def test_projection_authorities_and_fixed_flat_transformation_are_exact() -> None:
    projection = _owner_projection([30, 10, 20])
    records = _records(projection, _behavior_rows([30, 10, 20]))
    rows = {row["agent_id"]: row for row in records.rows}
    tensors = projection["tensors"]
    slot = 0
    row = rows[30]

    assert row["energy"] == float(tensors["policy_energy_observation"][slot].item())
    assert row["energy"] != float(tensors["energy"][slot].item())
    assert row["mutation_rate"] == 0.12345678
    assert row["mutation_rate"] == canon_float(
        tensors["genome_mutation_rate"][slot].item(),
        field="mutation_rate",
    )

    drives = np.asarray(tensors["drives"][slot].tolist(), dtype=np.float32)
    assert tensors["drives"].dtype == torch.float32
    expected_conflict = float(abs(float(drives[0]) - float(drives[2]))) + float(
        abs(float(drives[1]) - float(drives[3]))
    )
    assert row["drive_conflict"] == expected_conflict
    assert row["internal_drive_conflict"] == float(np.max(drives) - np.mean(drives))

    assert row["age"] == float(tensors["survival_steps"][slot].item())
    assert row["lineage"] == int(tensors["patch_habitat_lineage_id"][slot].item())
    assert row["lineage"] != int(tensors["lineage_founder_id"][slot].item())
    assert row["offspring"] == float(tensors["offspring_count"][slot].item())
    assert row["offspring"] != 999.0

    # Evolvable mask/radius controls are off: Flat reporting must express the
    # fixed full local window, not copy the raw radius=1/mask=1 genome fields.
    assert row["sensor_radius"] == 3.0
    assert row["observed_cells"] == 49.0
    assert row["active_cell_fraction"] == 1.0
    assert row["cell_channel_capacity"] == 49.0 * row["sensor_channel_count"]
    assert row["sensor_radius"] != float(tensors["genome_sensor_radius"][slot].item())


@pytest.mark.parametrize(
    "missing_tensor",
    [
        "alive_mask",
        "slot_to_agent_id",
        "policy_energy_observation",
        "health",
        "drives",
        "survival_steps",
        "patch_habitat_lineage_id",
        "lineage_founder_id",
        "offspring_count",
        "policy_use_gru",
        "genome_hidden_size",
        "genome_memory_dim",
        "genome_mutation_rate",
        "genome_sensor_radius",
    ],
)
def test_missing_authoritative_tensors_fail_closed(missing_tensor: str) -> None:
    projection = _owner_projection([1, 2])
    del projection["tensors"][missing_tensor]

    with pytest.raises(ValueError, match=missing_tensor):
        _records(projection, _behavior_rows([1, 2]))


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "health",
            torch.tensor([10.0], dtype=torch.float32),
            "health.*(length|capacity)|(length|capacity).*health",
        ),
        ("drives", torch.zeros((2, 3), dtype=torch.float32), "drives"),
        (
            "policy_energy_observation",
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            "policy_energy_observation.*float64|float64.*policy_energy_observation",
        ),
        ("drives", torch.zeros((2, 4), dtype=torch.float64), "drives.*float32|float32.*drives"),
        ("health", torch.tensor([10.0, 11.0], dtype=torch.float64), "health.*float32|float32.*health"),
    ],
)
def test_inconsistent_shapes_or_authority_dtypes_fail_closed(
    field: str,
    replacement: torch.Tensor,
    message: str,
) -> None:
    projection = _owner_projection([1, 2])
    projection["tensors"][field] = replacement

    with pytest.raises(ValueError, match=message):
        _records(projection, _behavior_rows([1, 2]))


def test_duplicate_live_agent_ids_fail_closed() -> None:
    projection = _owner_projection([7, 7])

    with pytest.raises(ValueError, match="duplicate.*agent|agent.*duplicate"):
        _records(projection, _behavior_rows([7]))


@pytest.mark.parametrize("unsupported", ["non_flat", "family", "gru", "memory"])
def test_unsupported_world_gru_or_memory_projection_fails_closed_without_mutation(
    unsupported: str,
) -> None:
    projection = _owner_projection([1, 2])
    if unsupported == "non_flat":
        projection["metadata"]["world"] = "patch"
        message = "Patch|patch|profile"
    elif unsupported == "family":
        projection["metadata"]["native_runtime_fixed_policy_family"] = "gru"
        message = "policy.*family|family.*policy|mlp"
    elif unsupported == "gru":
        projection["tensors"]["policy_use_gru"][0] = True
        message = "GRU|gru|policy_use_gru"
    else:
        projection["tensors"]["genome_memory_dim"][0] = 8
        message = "memory|genome_memory_dim"
    behavior_rows = _behavior_rows([1, 2])
    before_projection = copy.deepcopy(projection)
    before_behavior = copy.deepcopy(behavior_rows)

    with pytest.raises(ValueError, match=message):
        _records(projection, behavior_rows)

    _assert_projection_equal(projection, before_projection)
    assert behavior_rows == before_behavior


def test_successful_projection_does_not_mutate_owner_tensors_or_behavior_rows() -> None:
    projection = _owner_projection([3, 1, 2])
    assert {
        "prediction_update_count",
        "prediction_state_delta_abs_sum",
        "prediction_state_delta_abs_max",
    }.isdisjoint(projection["tensors"])
    behavior_rows = _behavior_rows([3, 1, 2], selected_ids={1, 3})
    before_projection = copy.deepcopy(projection)
    before_behavior = copy.deepcopy(behavior_rows)

    _records(projection, behavior_rows)

    _assert_projection_equal(projection, before_projection)
    assert behavior_rows == before_behavior


def test_shared_reducer_equals_reference_canonical_fixture() -> None:
    projection = _owner_projection([29, 5, 17])
    behavior_rows = _behavior_rows([29, 5, 17], selected_ids={5, 29})
    records = _records(projection, behavior_rows)
    env_cfg, sim_cfg = _configs()
    agents = _reference_agents(projection)
    selected_agents = [agent for agent in agents if agent.agent_id in {5, 29}]
    expected_rows = cognitive_status._agent_records(
        agents,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )
    expected_parent_rows = cognitive_status._agent_records(
        selected_agents,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )

    assert tuple(map(dict, records.rows)) == tuple(expected_rows)
    assert tuple(map(dict, records.selected_parent_rows)) == tuple(expected_parent_rows)

    expected_metrics = cognitive_status.compute_cognitive_status_metrics(
        agents,
        selected_parent_agents=selected_agents,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        generation=4,
    )
    actual_metrics = cognitive_status.compute_cognitive_status_metrics_from_records(
        records,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        generation=4,
    )

    assert actual_metrics == expected_metrics
