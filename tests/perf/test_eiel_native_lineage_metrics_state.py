from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.modular_genome_v1 import canon_float
from native_runtime.public_metrics_overlay import (
    NATIVE_LINEAGE_METRICS_INPUT_KEY,
    public_metrics_overlay,
)
from simulation.metrics_pkg.lineage import LineageMetricsWriter
from simulation.metrics_pkg.metrics_logger import MetricsLogger


def _behavior_row(
    agent_id: int,
    *,
    lifespan: int,
    energy_collected: float,
    offspring: int,
    injury: float,
    aggression: float,
) -> dict[str, object]:
    return {
        "generation": 0,
        "agent_id": agent_id,
        "cohort": "start" if agent_id in {1, 3} else "born",
        "move_n": 0,
        "move_s": 0,
        "move_e": 0,
        "move_w": 0,
        "gather": 1,
        "rest": 0,
        "attack": 0,
        "lifespan_steps": lifespan,
        "energy_collected": energy_collected,
        "offspring": offspring,
        "attacks_made": 0,
        "fitness": 1.0,
        "alive_end": 1,
        "eligible_for_selection": 1,
        "kept_next": 0,
        "injury_end": injury,
        "injury_delta": 0.0,
        "aggression_bias": aggression,
    }


def _native_fixture() -> tuple[dict[str, object], dict[str, object]]:
    raw: dict[str, object] = {
        "behavior_rows": [
            _behavior_row(1, lifespan=10, energy_collected=5.0, offspring=1, injury=0.1, aggression=-0.2),
            _behavior_row(2, lifespan=2, energy_collected=0.0, offspring=0, injury=0.0, aggression=-0.2),
            _behavior_row(3, lifespan=8, energy_collected=3.0, offspring=1, injury=0.2, aggression=0.4),
            _behavior_row(4, lifespan=1, energy_collected=0.0, offspring=0, injury=0.0, aggression=0.4),
        ],
    }
    projection: dict[str, object] = {
        "schema_version": "cxxbackbone10_checkpoint_projection_v1",
        "metadata": {
            "world": "flat",
            "generation": 1,
            "start_health": 12.0,
            "native_runtime_fixed_policy_family": "mlp",
        },
        "tensors": {
            "alive_mask": [True, True, True, True],
            "slot_to_agent_id": [1, 2, 3, 4],
            "pos_y": [0, 0, 1, 1],
            "pos_x": [0, 1, 0, 1],
            "energy": [40.0, 20.0, 35.0, 18.0],
            "policy_energy_observation": [40.0000000003, 20.0000000002, 35.0000000001, 18.0000000004],
            "health": [12.0, 12.0, 11.5, 12.0],
            "injury": [0.1, 0.0, 0.2, 0.0],
            "drives": [
                [0.4, 0.1, 0.2, 0.3],
                [0.5, 0.0, 0.1, 0.2],
                [0.3, 0.2, 0.2, 0.4],
                [0.6, 0.0, 0.1, 0.1],
            ],
            "lineage_founder_id": [1, 1, 3, 3],
            "lineage_depth": [0, 1, 0, 1],
            "lineage_birth_generation": [0, 0, 0, 0],
            "lineage_birth_tick": [0, 1, 0, 1],
            "offspring_count": [1, 0, 1, 0],
            "first_repro_step": [2, -1, 3, -1],
            "survival_steps": [10, 2, 8, 1],
            "energy_collected": [5.0, 0.0, 3.0, 0.0],
            "pred_err_count": [2, 0, 1, 0],
            "pred_err_sum": [0.5, 0.0, 0.2, 0.0],
            "episode0_pred_err_count": [0, 0, 0, 0],
            "episode0_pred_err_sum": [0.0, 0.0, 0.0, 0.0],
            "drive_jitter": [0.05, 0.0, 0.08, 0.0],
            "genome_hidden_size": [64, 64, 32, 32],
            "genome_memory_dim": [0, 0, 0, 0],
            "policy_use_gru": [False, False, False, False],
            "genome_exploration_bias": [0.1, 0.1, -0.3, -0.3],
            "genome_aggression_bias": [-0.2, -0.2, 0.4, 0.4],
        },
    }
    return raw, projection


def _reference_agents(projection: dict[str, object]) -> list[SimpleNamespace]:
    tensors = projection["tensors"]
    assert isinstance(tensors, dict)
    agents = []
    for slot, agent_id in enumerate(tensors["slot_to_agent_id"]):
        founder_id = int(tensors["lineage_founder_id"][slot])
        first_repro = int(tensors["first_repro_step"][slot])
        stats = SimpleNamespace(
            lineage_id=f"L{founder_id:08d}",
            founder_id=founder_id,
            parent_ids=(),
            parent_lineage_ids=(),
            birth_generation=int(tensors["lineage_birth_generation"][slot]),
            birth_tick=int(tensors["lineage_birth_tick"][slot]),
            lineage_depth=int(tensors["lineage_depth"][slot]),
            offspring_count=int(tensors["offspring_count"][slot]),
            survival_steps=float(tensors["survival_steps"][slot]),
            energy_collected=float(tensors["energy_collected"][slot]),
            first_repro_step=None if first_repro < 0 else first_repro,
            pred_err_count=int(tensors["pred_err_count"][slot]),
            pred_err_sum=float(tensors["pred_err_sum"][slot]),
            episode0_pred_err_mean=0.0,
            identity_drift_ema=0.0,
            drive_jitter=float(tensors["drive_jitter"][slot]),
            ecological_role="general_agent",
            role_source="default",
            role_confidence=None,
        )
        genome = SimpleNamespace(
            hidden_size=int(tensors["genome_hidden_size"][slot]),
            memory_dim=int(tensors["genome_memory_dim"][slot]),
            use_gru=bool(tensors["policy_use_gru"][slot]),
            exploration_bias=canon_float(
                tensors["genome_exploration_bias"][slot],
                field="exploration_bias",
            ),
            aggression_bias=canon_float(
                tensors["genome_aggression_bias"][slot],
                field="aggression_bias",
            ),
        )
        agents.append(
            SimpleNamespace(
                agent_id=int(agent_id),
                energy=float(tensors["policy_energy_observation"][slot]),
                health=float(tensors["health"][slot]),
                injury=float(tensors["injury"][slot]),
                drives=list(tensors["drives"][slot]),
                stats=stats,
                genome=genome,
                y=int(tensors["pos_y"][slot]),
                x=int(tensors["pos_x"][slot]),
            )
        )
    return agents


def _sim_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        lineage_observability_enable=True,
        lineage_qd_archive_enable=True,
        lineage_qd_archive_top_n=8,
        lineage_qd_archive_min_score=0.25,
        lineage_qd_archive_cooldown_generations=0,
        lineage_dominance_threshold=0.80,
        lineage_dominance_streak_required=5,
        lineage_min_active_lineages=2,
        lineage_entropy_minimum=0.10,
        lineage_reproduction_dominance_threshold=0.80,
        lineage_reproduction_dominance_streak_required=5,
        diversity_collapse_threshold=0.01,
        diversity_collapse_streak_required=3,
    )


def _pre_cull_birth_fixture() -> tuple[dict[str, object], dict[str, object], list[SimpleNamespace]]:
    raw = {
        "behavior_rows": [
            _behavior_row(1, lifespan=1, energy_collected=20.0, offspring=1, injury=0.0, aggression=-0.2),
            _behavior_row(5, lifespan=0, energy_collected=0.0, offspring=0, injury=0.0, aggression=-0.2),
        ],
        "parent_agent_ids": [1],
        "mate_agent_ids": [-1],
        "child_agent_ids": [5],
        "child_slots": [1],
        "child_positions": [[0, 1]],
        "child_drive_rows": [[5, 1, 0.6, 0.05, 0.15, 0.1]],
        "child_hash_chain_rows": [[5, 1, 1, "inherited_scalar_developmental", 2910598162661279558]],
        "birth_genome_trace_rows": [],
    }
    projection = {
        "schema_version": "cxxbackbone10_checkpoint_projection_v1",
        "metadata": {
            "world": "flat",
            "generation": 1,
            "start_health": 12.0,
            "native_runtime_fixed_policy_family": "mlp",
            "typed_lifecycle_event_active_count": 1,
        },
        "tensors": {
            "alive_mask": [True],
            "slot_to_agent_id": [1],
            "pos_y": [0],
            "pos_x": [0],
            "energy": [30.0],
            "policy_energy_observation": [30.0],
            "health": [12.0],
            "injury": [0.0],
            "drives": [[0.4, 0.05, 0.15, 0.1]],
            "lineage_founder_id": [1],
            "lineage_depth": [0],
            "lineage_birth_generation": [0],
            "lineage_birth_tick": [0],
            "offspring_count": [1],
            "first_repro_step": [1],
            "survival_steps": [1],
            "energy_collected": [20.0],
            "pred_err_count": [0],
            "pred_err_sum": [0.0],
            "episode0_pred_err_count": [0],
            "episode0_pred_err_sum": [0.0],
            "drive_jitter": [0.1],
            "genome_hidden_size": [64],
            "genome_memory_dim": [0],
            "policy_use_gru": [False],
            "genome_sensor_channel_mask": [1023],
            "genome_sensor_radius": [2],
            "genome_introspection_channel_mask": [32767],
            "genome_self_regulation_control_mask": [31],
            "genome_self_modification_target_mask": [31],
            "genome_directed_inheritance_channel_mask": [63],
            "genome_exploration_bias": [0.1],
            "genome_aggression_bias": [-0.2],
            "genome_mutation_rate": [0.05],
            "genome_reproduction_threshold": [30.0],
            "genome_action_temperature": [1.0],
            "genome_drive_inertia": [0.85],
            "genome_drive_noise": [0.02],
            "genome_drive_logit_gain": [1.0],
            "genome_pred_lr": [0.08],
            "genome_drive_sp_hunger": [0.55],
            "genome_drive_sp_pain": [0.05],
            "genome_drive_sp_fatigue": [0.15],
            "genome_drive_sp_fear": [0.1],
            "genome_exp_prior_strength": [0.35],
            "genome_exp_prior_noise": [0.05],
            "genome_drive_birth_noise": [0.03],
            "typed_lifecycle_event_event_type": [1],
            "typed_lifecycle_event_success": [1],
            "typed_lifecycle_event_child_agent_id": [5],
            "typed_lifecycle_event_parent_agent_id": [1],
            "typed_lifecycle_event_lineage_founder_id": [1],
            "typed_lifecycle_event_lineage_depth": [1],
            "typed_lifecycle_event_generation": [0],
            "typed_lifecycle_event_tick": [0],
            "typed_lifecycle_event_value": [15.0],
        },
    }

    def agent(agent_id: int, *, child: bool) -> SimpleNamespace:
        return SimpleNamespace(
            agent_id=agent_id,
            energy=15.0 if child else 30.0,
            health=12.0,
            injury=0.0,
            drives=[0.6, 0.05, 0.15, 0.1] if child else [0.4, 0.05, 0.15, 0.1],
            genome=SimpleNamespace(
                hidden_size=64,
                memory_dim=0,
                use_gru=False,
                aggression_bias=canon_float(-0.2, field="aggression_bias"),
                exploration_bias=canon_float(0.1, field="exploration_bias"),
            ),
            stats=SimpleNamespace(
                lineage_id="L00000001",
                founder_id=1,
                parent_ids=(1,) if child else (),
                parent_lineage_ids=("L00000001",) if child else (),
                birth_generation=0,
                birth_tick=0,
                lineage_depth=1 if child else 0,
                offspring_count=0 if child else 1,
                survival_steps=0 if child else 1,
                energy_collected=0.0 if child else 20.0,
                first_repro_step=None if child else 1,
                pred_err_count=0,
                pred_err_sum=0.0,
                episode0_pred_err_mean=0.0,
                identity_drift_ema=0.0,
                drive_jitter=0.0 if child else 0.1,
                ecological_role="general_agent",
                role_source="default",
                role_confidence=None,
            ),
            y=0,
            x=1 if child else 0,
        )

    return raw, projection, [agent(1, child=False), agent(5, child=True)]


def test_native_lineage_overlay_matches_reference_state_and_artifacts(tmp_path: Path) -> None:
    raw, projection = _native_fixture()
    overlay = public_metrics_overlay(raw, checkpoint_projection=projection, sim_cfg=_sim_cfg())
    assert NATIVE_LINEAGE_METRICS_INPUT_KEY in overlay

    reference = MetricsLogger(str(tmp_path / "reference"))
    native = MetricsLogger(str(tmp_path / "native"))
    reference.log_generation(
        generation=0,
        agents=_reference_agents(projection),
        selected_parent_agents=[],
        deaths_this_gen=0,
        sim_cfg=_sim_cfg(),
    )
    stale_python_agent = SimpleNamespace(
        agent_id=999,
        stats=SimpleNamespace(lineage_id="L00000999", founder_id=999),
        genome=SimpleNamespace(),
    )
    native.log_generation(
        generation=0,
        agents=[stale_python_agent],
        selected_parent_agents=[],
        deaths_this_gen=0,
        sim_cfg=_sim_cfg(),
        native_public_metrics_overlay={
            "source": "native_owner_checkpoint_projection",
            "values": overlay,
        },
    )

    reference_state = reference.export_state()["theme5_lineage_state"]
    native_state = native.export_state()["theme5_lineage_state"]
    assert native_state == reference_state
    assert native_state["archived_entry_ids"]
    assert native_state["lineage_presence_counts"] == {"L00000001": 1, "L00000003": 1}
    assert "L00000999" not in native_state["seen_lineages"]
    for relative in (
        "metrics/lineage_summary.csv",
        "metrics/lineage_metrics.jsonl",
        "metrics/lineage_descriptors.jsonl",
        "metrics/branch_candidates.jsonl",
        "metrics/lineage_events.jsonl",
    ):
        assert (tmp_path / "native" / relative).read_bytes() == (tmp_path / "reference" / relative).read_bytes()


def test_native_lineage_overlay_reconstructs_pre_cull_child_without_agent_sync(tmp_path: Path) -> None:
    raw, projection, reference_agents = _pre_cull_birth_fixture()
    overlay = public_metrics_overlay(raw, checkpoint_projection=projection, sim_cfg=_sim_cfg())
    payload = overlay[NATIVE_LINEAGE_METRICS_INPUT_KEY]
    assert payload["agent_ids"] == [1, 5]

    reference = LineageMetricsWriter(str(tmp_path / "reference"))
    native = LineageMetricsWriter(str(tmp_path / "native"))
    reference.log_generation(generation=0, agents=reference_agents, sim_cfg=_sim_cfg())
    native.log_generation(
        generation=0,
        agents=[],
        sim_cfg=_sim_cfg(),
        native_lineage_input=payload,
    )

    assert native.export_state() == reference.export_state()
    assert native.export_state()["lineage_presence_counts"] == {"L00000001": 1}
    assert native.export_state()["archived_entry_ids"]


@pytest.mark.parametrize(
    ("missing_tensor", "message"),
    [
        ("lineage_founder_id", "incomplete lineage founder identity"),
        ("lineage_depth", "missing required tensors.*lineage_depth"),
        ("policy_energy_observation", "missing required tensors.*policy_energy_observation"),
    ],
)
def test_native_lineage_overlay_fails_closed_when_projection_data_is_missing(
    missing_tensor: str,
    message: str,
) -> None:
    raw, projection = _native_fixture()
    broken = copy.deepcopy(projection)
    del broken["tensors"][missing_tensor]

    with pytest.raises(ValueError, match=message):
        public_metrics_overlay(raw, checkpoint_projection=broken, sim_cfg=_sim_cfg())


def test_native_lineage_writer_rejects_malformed_input_before_state_mutation(tmp_path: Path) -> None:
    raw, projection = _native_fixture()
    overlay = public_metrics_overlay(raw, checkpoint_projection=projection, sim_cfg=_sim_cfg())
    payload = copy.deepcopy(overlay[NATIVE_LINEAGE_METRICS_INPUT_KEY])
    del payload["records"][0]["stats"]["lineage_depth"]
    writer = LineageMetricsWriter(str(tmp_path))
    initial_state = writer.export_state()

    with pytest.raises(ValueError, match="missing required field lineage_depth"):
        writer.log_generation(
            generation=0,
            agents=[],
            sim_cfg=_sim_cfg(),
            native_lineage_input=payload,
        )

    assert writer.export_state() == initial_state
    assert (tmp_path / "metrics" / "lineage_summary.csv").read_text(encoding="utf-8").count("\n") == 1
