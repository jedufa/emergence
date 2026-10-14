from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, Mapping

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


CONTRACT_FIELDS = (
    "pred_lr",
    "drive_logit_gain",
    "drive_inertia",
    "drive_noise",
    "action_temperature",
    "aggression_bias",
    "reproduction_threshold",
)

PHENOTYPE_FIELD_TO_CONTRACT_FIELD = {
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
        "aggression_bias": 0.0,
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
        "drive_logit_gain": 1.0,
        "pred_lr": 0.08,
        "developmental_program": canonicalize_developmental_program(None),
    }
    kwargs.update(overrides)
    return Genome(**kwargs)


def _agent(genome: Genome, *, age_steps: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=17,
        genome=genome,
        energy=80.0,
        health=9.0,
        alive=True,
        drives=np.asarray([0.8, 0.2, 0.4, 0.6], dtype=np.float32),
        stats=SimpleNamespace(
            survival_steps=int(age_steps),
            pred_err_count=0,
            pred_err_sum=0.0,
            prediction_update_count=0,
            offspring_count=0,
            patch_habitat_lineage_id=1,
        ),
        directed_inheritance_payload=None,
    )


def _phenotype(genome: Genome, *, age_steps: int = 10) -> object:
    return express_cognitive_phenotype(
        genome,
        agent=_agent(genome, age_steps=age_steps),
        env_cfg=EnvConfig(width=8, height=8, view_radius=2),
        sim_cfg=SimConfig(seed=17),
        age_steps=int(age_steps),
    )


def _phenotype_contract_values(phenotype: object) -> dict[str, float]:
    return {
        contract_field: float(getattr(phenotype, phenotype_field))
        for phenotype_field, contract_field in PHENOTYPE_FIELD_TO_CONTRACT_FIELD.items()
    }


def _genome_contract_values(genome: Genome) -> dict[str, float]:
    return {field: float(getattr(genome, field)) for field in CONTRACT_FIELDS}


def _mismatches(left: Mapping[str, float], right: Mapping[str, Any]) -> set[str]:
    return {
        field
        for field in CONTRACT_FIELDS
        if field in right and float(left[field]) != pytest.approx(float(right[field]))
    }


def _assert_contract_values_close(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    assert set(actual) == set(expected)
    for field, expected_value in expected.items():
        assert float(actual[field]) == pytest.approx(float(expected_value))


def _diagnostic_flat_phenotype_input_from_projection(
    genome: Genome,
    expression: Mapping[str, Any] | None,
) -> dict[str, float]:
    flat = _genome_contract_values(genome)
    flat.update(
        {
            field: float(value)
            for field, value in developmental_effective_projection(expression).items()
            if field in CONTRACT_FIELDS
        }
    )
    return flat


def _nested_expression(nested_effective: Mapping[str, float]) -> dict[str, object]:
    return {
        "stage_id": "adult",
        "active_by_module_id": {"m_identity": True, "m_life": True, "m_behavior": True},
        "inactive_reason_by_module_id": {},
        "active_module_count": 3,
        "module_count": 3,
        "module_structure_signature": "diagnostic",
        "effective": dict(nested_effective),
    }


def test_envdevconsume_current_decision_keeps_flat_fallback_not_nested_projection_consumption() -> None:
    genome = _genome()
    nested_effective = {
        "pred_lr": 0.5,
        "drive_logit_gain": 2.0,
        "drive_inertia": 0.7,
        "drive_noise": 0.09,
        "action_temperature": 1.8,
        "aggression_bias": 0.4,
        "reproduction_threshold": 12.0,
    }

    def _resolve(self: Genome, *, age_steps: int) -> dict[str, object]:
        return _nested_expression(nested_effective)

    genome.resolve_developmental_expression = MethodType(_resolve, genome)  # type: ignore[method-assign]

    phenotype_values = _phenotype_contract_values(_phenotype(genome))
    projection_values = developmental_effective_projection(genome.resolve_developmental_expression(age_steps=10))

    assert phenotype_values == _genome_contract_values(genome)
    assert projection_values == nested_effective
    assert _mismatches(phenotype_values, projection_values) == set(CONTRACT_FIELDS)


def test_envdevconsume_explicit_flat_values_remain_the_only_phenotype_input_parity_contract() -> None:
    genome = _genome()
    flat_effective = {
        "pred_lr": 0.4,
        "drive_logit_gain": 1.7,
        "drive_inertia": 0.6,
        "drive_noise": 0.05,
        "action_temperature": 1.4,
        "aggression_bias": -0.3,
        "reproduction_threshold": 18.0,
    }

    def _resolve(self: Genome, *, age_steps: int) -> dict[str, object]:
        return dict(flat_effective)

    genome.resolve_developmental_expression = MethodType(_resolve, genome)  # type: ignore[method-assign]

    phenotype_values = _phenotype_contract_values(_phenotype(genome))

    assert phenotype_values == flat_effective
    assert developmental_effective_projection(flat_effective) == flat_effective


def test_envdevconsume_diagnostic_projection_flattening_adapter_would_be_a_behavior_change() -> None:
    genome = _genome()
    nested_effective = {
        "pred_lr": 0.5,
        "drive_logit_gain": 2.0,
        "drive_inertia": 0.7,
        "drive_noise": 0.09,
        "action_temperature": 1.8,
        "aggression_bias": 0.4,
        "reproduction_threshold": 12.0,
    }
    expression = _nested_expression(nested_effective)

    def _resolve(self: Genome, *, age_steps: int) -> dict[str, object]:
        return dict(expression)

    genome.resolve_developmental_expression = MethodType(_resolve, genome)  # type: ignore[method-assign]

    current_values = _phenotype_contract_values(_phenotype(genome))
    migration_candidate_values = _diagnostic_flat_phenotype_input_from_projection(genome, expression)

    assert current_values == _genome_contract_values(genome)
    assert migration_candidate_values == nested_effective
    assert _mismatches(current_values, migration_candidate_values) == set(CONTRACT_FIELDS)


def test_envdevconsume_non_default_developmental_program_projection_has_no_phenotype_parity_contract() -> None:
    program = canonicalize_developmental_program(
        {
            "stages": [
                {"stage_id": "juvenile", "start_step": 0, "end_step": 4},
                {"stage_id": "adult", "start_step": 5, "end_step": -1},
            ],
            "module_windows": [
                {"module_id": "m_behavior", "stage_id": "juvenile", "active": False},
                {"module_id": "m_behavior", "stage_id": "adult", "active": True},
                {"module_id": "m_identity", "stage_id": "juvenile", "active": False},
                {"module_id": "m_identity", "stage_id": "adult", "active": True},
                {"module_id": "m_life", "stage_id": "juvenile", "active": False},
                {"module_id": "m_life", "stage_id": "adult", "active": True},
            ],
        }
    )
    genome = _genome(
        developmental_program=program,
        modules=[
            {
                "module_id": "m_behavior",
                "module_type": "behavior_bias",
                "params": {"action_temperature": 1.8, "aggression_bias": 0.4},
            },
            {
                "module_id": "m_identity",
                "module_type": "identity_regulation",
                "params": {
                    "pred_lr": 0.5,
                    "drive_logit_gain": 2.0,
                    "drive_inertia": 0.7,
                    "drive_noise": 0.09,
                },
            },
            {
                "module_id": "m_life",
                "module_type": "life_history",
                "params": {"reproduction_threshold": 12.0},
            },
        ],
    )

    juvenile_projection = developmental_effective_projection(genome.resolve_developmental_expression(age_steps=2))
    adult_expression = genome.resolve_developmental_expression(age_steps=10)
    adult_projection = developmental_effective_projection(adult_expression)
    phenotype_values = _phenotype_contract_values(_phenotype(genome, age_steps=10))

    assert juvenile_projection == _genome_contract_values(genome)
    expected_adult_projection = {
        "pred_lr": 0.5,
        "drive_logit_gain": 2.0,
        "drive_inertia": 0.7,
        "drive_noise": 0.09,
        "action_temperature": 1.8,
        "aggression_bias": 0.4,
        "reproduction_threshold": 12.0,
    }
    _assert_contract_values_close(adult_projection, expected_adult_projection)
    assert phenotype_values == _genome_contract_values(genome)
    _assert_contract_values_close(_diagnostic_flat_phenotype_input_from_projection(genome, adult_expression), adult_projection)
    assert _mismatches(phenotype_values, adult_projection) == set(CONTRACT_FIELDS)
