from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from agents.structural_identity_v1 import new_native_identity
from evolution.capacity_contract import build_capacity_contract
from evolution.recombination_c2 import recombine_lineage_payloads
from evolution.structural_gated_recurrent_v1 import GATED_PROFILE
from evolution.structural_live_modules_v1 import LIVE_MODULE_SCHEMA_VERSION, LIVE_PROFILE
from evolution.structural_sustainable_growth_v1 import (
    ACTIVITY_RECORD_KEY,
    ACTIVITY_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    COST_COEFFICIENTS,
    COST_MODE,
    COST_SCHEMA_VERSION,
    DUPLICATION_NEUTRALIZATION_VERSION,
    FIXED_ENVELOPE_SHAPE,
    FIXED_ENVELOPE_SHAPE_DIGEST,
    GROWTH_MUTATION_ATTEMPTS,
    GROWTH_MUTATION_BINS,
    GROWTH_MUTATION_DRAWS,
    REFINEMENT_OPERATOR_VERSION,
    SPECIALIZATION_TELEMETRY_VERSION,
    SUSTAINABLE_GROWTH_PROFILE,
)
from simulation.checkpoint_bundle_v2 import (
    CheckpointStrictError,
    _agent_to_dict,
    _build_metadata,
    _escap8_sustainable_growth_metadata,
    checkpoint_bundle_v2_apply,
)
from simulation.runner_pkg.runner_base import (
    _validate_escap8_sustainable_growth_config,
)
from simulation.runner_pkg.runner_config import SimConfig


class _CountingRng:
    def __init__(self, value: float = 0.9) -> None:
        self.value = float(value)
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        return self.value


def _identity(index: int) -> dict:
    return new_native_identity(
        context={
            "campaign_id": "escap8-c2-checkpoint",
            "branch_id": "sustainable-growth",
            "generation": 1,
            "tick": 0,
            "child_agent_id": int(index),
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap8-c2-checkpoint-lineage",
        },
        operator_id="fixture",
        attempt_index=int(index),
        operator_role=f"module-{index}",
    )


def _module(index: int, *, dormant: bool | None = None) -> dict:
    params: dict = {
        "live_operator": "bounded_scalar_integrator_v1",
        "expression_scale": 1.0,
        "live_retention": 0.5,
        "live_input_gain": 1.0,
        "live_output_gain": 1.0,
    }
    if dormant is not None:
        params[ACTIVITY_RECORD_KEY] = {
            "schema_version": ACTIVITY_SCHEMA_VERSION,
            "dormant": bool(dormant),
        }
    identity = _identity(index)
    return {
        "module_id": f"m{index:03d}",
        "module_type": "generic",
        "params": params,
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **identity,
        "latent_pos": list(identity["topology_pos"]),
    }


def _payload(modules: list[dict]) -> dict:
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": deepcopy(modules),
        "connections": [],
    }


def _cfg(*, enabled: bool) -> SimConfig:
    cfg = SimConfig()
    cfg.structural_live_modules_enable = True
    cfg.structural_live_modules_profile = LIVE_PROFILE
    cfg.structural_live_modules_schema_version = LIVE_MODULE_SCHEMA_VERSION
    cfg.structural_gated_recurrent_enable = True
    cfg.structural_gated_recurrent_profile = GATED_PROFILE
    cfg.structural_sustainable_growth_enable = bool(enabled)
    cfg.structural_sustainable_growth_profile = (
        SUSTAINABLE_GROWTH_PROFILE if enabled else "none"
    )
    cfg.device = "cpu"
    cfg.bank_device = "cpu"
    cfg.policy_tick_stride = 1
    return cfg


def _metadata(cfg: object) -> dict:
    return _build_metadata(
        SimpleNamespace(sim_cfg=cfg),
        8,
        True,
        "world",
        "population",
        "rng",
        "runner",
        "bank",
        "resident",
        "native",
        "events",
        "wal",
    )


def test_disabled_checkpoint_is_exactly_escap7_and_omits_escap8_metadata() -> None:
    disabled = _cfg(enabled=False)
    historical = SimpleNamespace(
        **{
            name: deepcopy(value)
            for name, value in vars(disabled).items()
            if not name.startswith("structural_sustainable_growth_")
        }
    )

    disabled_metadata = _metadata(disabled)
    historical_metadata = _metadata(historical)

    assert "structural_sustainable_growth" not in disabled_metadata
    assert disabled_metadata == historical_metadata
    assert disabled_metadata["checkpoint_id"] == historical_metadata["checkpoint_id"]


def test_disabled_capacity_epoch_metadata_and_checkpoint_identity_are_exact_escap7() -> None:
    disabled = _cfg(enabled=False)
    disabled.capacity_epoch_metadata_enable = True
    historical = SimpleNamespace(
        **{
            name: deepcopy(value)
            for name, value in vars(disabled).items()
            if not name.startswith("structural_sustainable_growth_")
        }
    )

    disabled_metadata = _metadata(disabled)
    historical_metadata = _metadata(historical)

    assert disabled_metadata == historical_metadata
    assert disabled_metadata["checkpoint_id"] == historical_metadata["checkpoint_id"]
    assert disabled_metadata["capacity_epoch"]["capacity_contract_hash"] == (
        historical_metadata["capacity_epoch"]["capacity_contract_hash"]
    )
    disabled_contract = build_capacity_contract(sim_cfg=disabled)
    historical_contract = build_capacity_contract(sim_cfg=historical)
    assert disabled_contract == historical_contract
    assert "sustainable_growth" not in disabled_contract["structure"]


def test_enabled_checkpoint_metadata_freezes_complete_recomputable_contract() -> None:
    cfg = _cfg(enabled=True)
    metadata = _metadata(cfg)
    growth = metadata["structural_sustainable_growth"]

    assert growth == _escap8_sustainable_growth_metadata(cfg)
    assert growth["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert growth["activity_schema_version"] == ACTIVITY_SCHEMA_VERSION
    assert (
        growth["duplication_neutralization_version"]
        == DUPLICATION_NEUTRALIZATION_VERSION
    )
    assert growth["refinement_operator_version"] == REFINEMENT_OPERATOR_VERSION
    assert growth["specialization_telemetry_version"] == SPECIALIZATION_TELEMETRY_VERSION
    assert growth["cost_schema_version"] == COST_SCHEMA_VERSION
    assert growth["cost_mode"] == COST_MODE == "reporting_only"
    assert growth["cost_coefficients"] == {
        str(key): float(value) for key, value in sorted(COST_COEFFICIENTS.items())
    }
    assert growth["cost_telemetry_continuation"] == "recomputed_reporting_only"
    assert growth["c1_operator_attempts"] == GROWTH_MUTATION_ATTEMPTS == 1
    assert growth["c1_operator_draws_per_attempt"] == GROWTH_MUTATION_DRAWS == 2
    assert growth["c1_lane_map"] == [
        {
            "lower_inclusive": float(lower),
            "upper_exclusive": float(upper),
            "operation": operation,
        }
        for lower, upper, operation in GROWTH_MUTATION_BINS
    ]
    assert growth["fixed_envelope_shape"] == dict(FIXED_ENVELOPE_SHAPE)
    assert growth["fixed_envelope_shape_digest"] == FIXED_ENVELOPE_SHAPE_DIGEST
    assert growth["migration_rng_draws"] == 0


@pytest.mark.parametrize(
    "field",
    (
        "structural_sustainable_growth_checkpoint_schema_version",
        "structural_sustainable_growth_duplication_neutralization_version",
        "structural_sustainable_growth_refinement_operator_version",
        "structural_sustainable_growth_specialization_telemetry_version",
    ),
)
def test_atomic_preflight_version_mismatch_changes_identity_and_rejects(field: str) -> None:
    control = _cfg(enabled=True)
    changed = _cfg(enabled=True)
    setattr(changed, field, "different-version")

    assert _metadata(changed)["checkpoint_id"] != _metadata(control)["checkpoint_id"]
    with pytest.raises(RuntimeError, match="ESCAP8_CONFIG_MISMATCH"):
        _validate_escap8_sustainable_growth_config(
            changed,
            policy_device=torch.device("cpu"),
            bank_device=torch.device("cpu"),
        )


def test_enabled_lane_reuses_escap7_fail_closed_helper_and_disabled_adds_none() -> None:
    enabled = _cfg(enabled=True)
    _validate_escap8_sustainable_growth_config(
        enabled,
        policy_device=torch.device("cpu"),
        bank_device=torch.device("cpu"),
    )
    enabled.native_runtime_enable = True
    with pytest.raises(RuntimeError, match="ESCAP8_UNSUPPORTED_EXECUTION_LANE"):
        _validate_escap8_sustainable_growth_config(
            enabled,
            policy_device=torch.device("cpu"),
            bank_device=torch.device("cpu"),
        )

    disabled = _cfg(enabled=False)
    disabled.native_runtime_enable = True
    _validate_escap8_sustainable_growth_config(
        disabled,
        policy_device=torch.device("cuda"),
        bank_device=torch.device("cuda"),
    )


def test_strict_restore_rejects_cost_coefficient_mismatch_before_apply() -> None:
    cfg = _cfg(enabled=True)
    metadata = _metadata(cfg)
    coefficient = sorted(COST_COEFFICIENTS)[0]
    metadata["structural_sustainable_growth"]["cost_coefficients"][coefficient] += 1.0
    sim = SimpleNamespace(
        sim_cfg=cfg,
        device=torch.device("cpu"),
        bank_device=torch.device("cpu"),
    )

    with pytest.raises(CheckpointStrictError, match="metadata/config mismatch"):
        checkpoint_bundle_v2_apply(sim, {"metadata": metadata})


def test_checkpoint_save_rejects_stale_nonzero_dormant_output() -> None:
    cfg = _cfg(enabled=True)
    dormant = _module(30, dormant=True)
    instance_id = str(dormant["module_instance_id"])
    agent = SimpleNamespace(
        agent_id=30,
        genome=SimpleNamespace(modules=[dormant], to_dict=lambda: _payload([dormant])),
        structural_live_module_state={instance_id: 0.75},
        structural_live_module_outputs={instance_id: 0.25},
        structural_live_module_telemetry={},
        stats=None,
    )

    with pytest.raises(RuntimeError, match="DORMANT_OUTPUT_NONZERO"):
        _agent_to_dict(
            agent,
            structural_development_enabled=False,
            structural_development_sim_cfg=cfg,
        )


def test_c2_activity_schema_rejects_before_donor_draw() -> None:
    first = _module(1, dormant=False)
    second = deepcopy(first)
    second["params"][ACTIVITY_RECORD_KEY]["schema_version"] = "unsupported"
    rng = _CountingRng()

    with pytest.raises((RuntimeError, ValueError)):
        recombine_lineage_payloads(
            parent_a_payload=_payload([first]),
            parent_b_payload=_payload([second]),
            rng=rng,
            sustainable_growth_enabled=True,
        )

    assert rng.draws == 0


def test_c2_compatible_homolog_atomically_inherits_dormancy_and_reverses() -> None:
    active = _module(2, dormant=False)
    dormant = deepcopy(active)
    dormant["params"][ACTIVITY_RECORD_KEY]["dormant"] = True

    rng_ab = _CountingRng(0.9)
    child_ab, summary_ab = recombine_lineage_payloads(
        parent_a_payload=_payload([active]),
        parent_b_payload=_payload([dormant]),
        rng=rng_ab,
        sustainable_growth_enabled=True,
    )
    rng_ba = _CountingRng(0.9)
    child_ba, summary_ba = recombine_lineage_payloads(
        parent_a_payload=_payload([dormant]),
        parent_b_payload=_payload([active]),
        rng=rng_ba,
        sustainable_growth_enabled=True,
    )

    assert child_ab == child_ba
    assert summary_ab == summary_ba
    assert rng_ab.draws == rng_ba.draws == summary_ab.budget_draws == 1
    inherited = child_ab["modules"][0]["params"][ACTIVITY_RECORD_KEY]
    assert inherited in (
        active["params"][ACTIVITY_RECORD_KEY],
        dormant["params"][ACTIVITY_RECORD_KEY],
    )


def test_c2_prospective_unmatched_union_cap_rejects_before_draw() -> None:
    parent_a = [_module(index, dormant=False) for index in range(7)]
    parent_b = [_module(index + 100, dormant=True) for index in range(7)]
    rng = _CountingRng()

    with pytest.raises(RuntimeError, match="PROSPECTIVE_CAP_EXCEEDED"):
        recombine_lineage_payloads(
            parent_a_payload=_payload(parent_a),
            parent_b_payload=_payload(parent_b),
            rng=rng,
            sustainable_growth_enabled=True,
        )

    assert rng.draws == 0
