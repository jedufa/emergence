from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from agents.cognition import developmental_effective_projection, express_cognitive_phenotype
from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.directed_inheritance import implemented_directed_inheritance_channel_mask
from agents.genome import Genome
from agents.introspection import implemented_introspection_channel_mask
from agents.perception import implemented_sensor_channel_mask
from agents.self_modification import implemented_self_modification_target_mask
from agents.self_regulation import implemented_self_regulation_control_mask
from env.env_config import EnvConfig
from simulation.runner_pkg.runner_config import SimConfig


PHENOTYPE_FIELD_TO_PROJECTION_KEY = {
    "prediction_learning_rate": "pred_lr",
    "drive_logit_gain": "drive_logit_gain",
    "drive_inertia": "drive_inertia",
    "drive_noise": "drive_noise",
    "action_temperature": "action_temperature",
    "aggression_bias": "aggression_bias",
    "reproduction_threshold": "reproduction_threshold",
}


def _genome(**overrides: object) -> Genome:
    kwargs = {
        "hidden_size": 64,
        "exploration_bias": 0.1,
        "aggression_bias": 0.2,
        "reproduction_threshold": 30.0,
        "mutation_rate": 0.03,
        "action_temperature": 1.0,
        "use_gru": True,
        "memory_dim": 32,
        "sensor_channel_mask": implemented_sensor_channel_mask(),
        "sensor_radius": 2,
        "introspection_channel_mask": implemented_introspection_channel_mask(),
        "self_regulation_control_mask": implemented_self_regulation_control_mask(),
        "self_modification_target_mask": implemented_self_modification_target_mask(),
        "directed_inheritance_channel_mask": implemented_directed_inheritance_channel_mask(),
        "drive_inertia": 0.85,
        "drive_noise": 0.02,
        "drive_logit_gain": 1.1,
        "pred_lr": 0.2,
        "developmental_program": canonicalize_developmental_program(None),
    }
    kwargs.update(overrides)
    return Genome(**kwargs)


def _agent(genome: Genome, *, age_steps: int = 10) -> SimpleNamespace:
    stats = SimpleNamespace(
        survival_steps=int(age_steps),
        pred_err_count=0,
        pred_err_sum=0.0,
        prediction_update_count=0,
        offspring_count=0,
        patch_habitat_lineage_id=1,
    )
    return SimpleNamespace(
        agent_id=7,
        genome=genome,
        energy=80.0,
        health=9.0,
        alive=True,
        drives=np.asarray([0.8, 0.2, 0.4, 0.6], dtype=np.float32),
        stats=stats,
        directed_inheritance_payload=None,
    )


def _phenotype(genome: Genome, *, age_steps: int = 10):
    return express_cognitive_phenotype(
        genome,
        agent=_agent(genome, age_steps=age_steps),
        env_cfg=EnvConfig(width=8, height=8, view_radius=2),
        sim_cfg=SimConfig(seed=11),
        age_steps=int(age_steps),
    )


def _phenotype_projection_values(phenotype: object) -> dict[str, float]:
    return {
        key: float(getattr(phenotype, phenotype_field))
        for phenotype_field, key in PHENOTYPE_FIELD_TO_PROJECTION_KEY.items()
    }


def _mismatched_projection_fields(phenotype: object, projection: dict[str, object]) -> set[str]:
    consumed = _phenotype_projection_values(phenotype)
    return {
        key
        for key in sorted(PHENOTYPE_FIELD_TO_PROJECTION_KEY.values())
        if key in projection and float(consumed[key]) != float(projection[key])
    }


def _assert_values_close(actual: dict[str, float], expected: dict[str, float]) -> None:
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert actual[key] == pytest.approx(float(value))


def test_flat_developmental_scalar_map_is_consumed_by_phenotype() -> None:
    genome = _genome()
    flat = {
        "pred_lr": 0.5,
        "drive_logit_gain": 2.0,
        "drive_inertia": 0.7,
        "drive_noise": 0.12,
        "action_temperature": 1.8,
        "aggression_bias": 0.4,
        "reproduction_threshold": 12.0,
    }

    def _resolve(self: Genome, *, age_steps: int) -> dict[str, object]:
        return dict(flat)

    genome.resolve_developmental_expression = MethodType(_resolve, genome)  # type: ignore[method-assign]

    phenotype = _phenotype(genome)

    assert _phenotype_projection_values(phenotype) == flat
    assert developmental_effective_projection(flat) == flat
    assert _mismatched_projection_fields(phenotype, flat) == set()


def test_nested_developmental_effective_falls_back_to_genome_scalars() -> None:
    genome = _genome(
        pred_lr=0.08,
        drive_logit_gain=1.0,
        drive_inertia=0.85,
        drive_noise=0.02,
        action_temperature=1.0,
        aggression_bias=0.0,
        reproduction_threshold=30.0,
    )
    nested_effective = {
        "pred_lr": 0.5,
        "drive_logit_gain": 2.0,
        "drive_inertia": 0.7,
        "drive_noise": 0.12,
        "action_temperature": 1.8,
        "aggression_bias": 0.4,
        "reproduction_threshold": 12.0,
    }

    def _resolve(self: Genome, *, age_steps: int) -> dict[str, object]:
        return {
            "stage_id": "default",
            "active_by_module_id": {},
            "inactive_reason_by_module_id": {},
            "active_module_count": 0,
            "module_count": 0,
            "effective": dict(nested_effective),
        }

    genome.resolve_developmental_expression = MethodType(_resolve, genome)  # type: ignore[method-assign]

    phenotype = _phenotype(genome)
    projection = developmental_effective_projection(genome.resolve_developmental_expression(age_steps=10))

    _assert_values_close(_phenotype_projection_values(phenotype), {
        "pred_lr": 0.08,
        "drive_logit_gain": 1.0,
        "drive_inertia": 0.85,
        "drive_noise": 0.02,
        "action_temperature": 1.0,
        "aggression_bias": 0.0,
        "reproduction_threshold": 30.0,
    })
    assert projection == nested_effective
    assert _mismatched_projection_fields(phenotype, projection) == set(nested_effective)


def test_real_genome_active_developmental_module_projection_is_not_phenotype_input() -> None:
    program = canonicalize_developmental_program(
        {
            "stages": [
                {"stage_id": "juvenile", "start_step": 0, "end_step": 4},
                {"stage_id": "adult", "start_step": 5, "end_step": -1},
            ],
            "module_windows": [
                {"module_id": "m_identity", "stage_id": "juvenile", "active": False},
                {"module_id": "m_identity", "stage_id": "adult", "active": True},
                {"module_id": "m_life", "stage_id": "juvenile", "active": False},
                {"module_id": "m_life", "stage_id": "adult", "active": True},
            ],
        }
    )
    genome = _genome(
        pred_lr=0.08,
        drive_logit_gain=1.0,
        drive_inertia=0.85,
        drive_noise=0.02,
        action_temperature=1.0,
        aggression_bias=0.0,
        reproduction_threshold=30.0,
        developmental_program=program,
        modules=[
            {
                "module_id": "m_identity",
                "module_type": "identity_regulation",
                "params": {
                    "pred_lr": 0.5,
                    "drive_logit_gain": 2.0,
                    "drive_inertia": 0.7,
                    "drive_noise": 0.12,
                },
            },
            {
                "module_id": "m_life",
                "module_type": "life_history",
                "params": {"reproduction_threshold": 12.0},
            },
        ],
    )

    juvenile_expr = genome.resolve_developmental_expression(age_steps=2)
    adult_expr = genome.resolve_developmental_expression(age_steps=10)
    juvenile_projection = developmental_effective_projection(juvenile_expr)
    adult_projection = developmental_effective_projection(adult_expr)
    phenotype = _phenotype(genome, age_steps=10)

    assert juvenile_expr["stage_id"] == "juvenile"
    assert adult_expr["stage_id"] == "adult"
    assert juvenile_projection["pred_lr"] == 0.08
    assert juvenile_projection["drive_logit_gain"] == 1.0
    assert juvenile_projection["reproduction_threshold"] == 30.0
    assert adult_projection["pred_lr"] == 0.5
    assert adult_projection["drive_logit_gain"] == 2.0
    assert adult_projection["drive_inertia"] == pytest.approx(0.7)
    assert adult_projection["drive_noise"] == pytest.approx(0.12)
    assert adult_projection["reproduction_threshold"] == 12.0
    assert _phenotype_projection_values(phenotype)["pred_lr"] == 0.08
    assert _phenotype_projection_values(phenotype)["drive_logit_gain"] == 1.0
    assert _phenotype_projection_values(phenotype)["reproduction_threshold"] == 30.0
    assert _mismatched_projection_fields(phenotype, adult_projection) == {
        "drive_inertia",
        "drive_logit_gain",
        "drive_noise",
        "pred_lr",
        "reproduction_threshold",
    }


def test_missing_nested_effective_fields_fall_back_per_field() -> None:
    genome = _genome(pred_lr=0.08, drive_logit_gain=1.0, reproduction_threshold=30.0)
    nested_effective = {"pred_lr": 0.5}

    def _resolve(self: Genome, *, age_steps: int) -> dict[str, object]:
        return {"stage_id": "default", "effective": dict(nested_effective)}

    genome.resolve_developmental_expression = MethodType(_resolve, genome)  # type: ignore[method-assign]

    phenotype = _phenotype(genome)
    projection = developmental_effective_projection(genome.resolve_developmental_expression(age_steps=10))

    assert projection == {"pred_lr": 0.5}
    assert _phenotype_projection_values(phenotype)["pred_lr"] == 0.08
    assert _phenotype_projection_values(phenotype)["drive_logit_gain"] == 1.0
    assert _phenotype_projection_values(phenotype)["reproduction_threshold"] == 30.0
    assert _mismatched_projection_fields(phenotype, projection) == {"pred_lr"}
