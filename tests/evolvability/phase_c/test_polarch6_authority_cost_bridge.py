from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from agents.family_head_residuals import FAMILY_HEAD_RESIDUAL_FIELD, canonical_family_head_residual_block
from evolution.policy_family_head_authority_runtime import (
    TECHNICAL_COST_COEFFICIENT,
    apply_authority_to_logits_torch,
    authority_log_probs_from_logits_torch,
    binary_opportunity_conditioned_residuals_torch,
    charge_family_head_residual_cost,
    validate_family_head_authority_config,
)
from simulation.runner_pkg.runner_config import SimConfig


def _cfg(**kwargs) -> SimConfig:
    cfg = SimConfig(seed=1, generations=1, steps_per_generation=1, device="cpu", bank_device="cpu", behavior_log_enable=False)
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def _agent(agent_id: int, block=None, *, energy: float = 10.0, alive: bool = True) -> SimpleNamespace:
    genome = SimpleNamespace(**{FAMILY_HEAD_RESIDUAL_FIELD: canonical_family_head_residual_block(block)})
    return SimpleNamespace(agent_id=agent_id, genome=genome, energy=energy, alive=alive)


def test_polarch6_config_dependencies_and_coefficient_validation() -> None:
    assert validate_family_head_authority_config(_cfg(policy_family_head_residual_authority_enable=True)) == [
        "authority_requires_family_head_residual_genome_enable"
    ]
    assert validate_family_head_authority_config(
        _cfg(policy_family_head_residual_genome_enable=True, policy_family_head_residual_cost_coefficient=0.005)
    ) == ["positive_cost_coefficient_requires_cost_enable"]
    assert validate_family_head_authority_config(
        _cfg(
            policy_family_head_residual_genome_enable=True,
            policy_family_head_residual_authority_enable=True,
            policy_family_head_residual_cost_enable=True,
            policy_family_head_residual_cost_coefficient=TECHNICAL_COST_COEFFICIENT,
        )
    ) == []
    assert validate_family_head_authority_config(
        _cfg(policy_family_head_residual_genome_enable=True, policy_family_head_binary_opportunity_gate_enable=True)
    ) == ["binary_opportunity_gate_requires_family_head_residual_authority_enable"]


def test_polarch6_zero_residual_preserves_argmax_and_gumbel_choice() -> None:
    logits = torch.tensor([[0.2, -0.1, 0.0, 0.15, 0.45, 0.05, -0.25]], dtype=torch.float32)
    residuals = torch.zeros((1, 4), dtype=torch.float32)
    transformed = authority_log_probs_from_logits_torch(logits, residuals)
    gumbel = torch.tensor([[0.01, 0.02, 0.0, -0.01, 0.03, -0.02, 0.0]], dtype=torch.float32)

    assert torch.argmax(transformed, dim=1).item() == torch.argmax(logits, dim=1).item()
    assert torch.argmax(transformed + gumbel, dim=1).item() == torch.argmax(logits + gumbel, dim=1).item()
    assert torch.softmax(transformed, dim=1).sum().item() == pytest.approx(1.0)


def test_polarch6_nonzero_authority_changes_primitive_distribution_without_extra_draws() -> None:
    alive = [_agent(0, {"attack": 0.4, "movement": -0.4})]
    cfg = _cfg(policy_family_head_residual_genome_enable=True, policy_family_head_residual_authority_enable=True)
    logits = torch.tensor([[0.2, -0.1, 0.0, 0.15, 0.45, 0.05, -0.25]], dtype=torch.float32)
    transformed, summary = apply_authority_to_logits_torch(logits, alive=alive, row_indices=[0], sim_cfg=cfg)

    assert summary is not None
    assert summary["random_draws_added"] == 0
    assert summary["residual_gain"] == pytest.approx(1.0)
    assert torch.softmax(transformed, dim=1)[0, 6].item() > torch.softmax(logits, dim=1)[0, 6].item()


def test_polarch6_cost_charges_once_per_living_tick_and_excludes_dead() -> None:
    alive = [
        _agent(0, {"movement": 0.5, "gather": -0.5}, energy=10.0),
        _agent(1, None, energy=10.0),
        _agent(2, {"attack": 1.0, "movement": -1.0}, energy=10.0, alive=False),
    ]
    cfg = _cfg(
        policy_family_head_residual_genome_enable=True,
        policy_family_head_residual_cost_enable=True,
        policy_family_head_residual_cost_coefficient=TECHNICAL_COST_COEFFICIENT,
    )
    runner = SimpleNamespace(
        sim_cfg=cfg,
        env=SimpleNamespace(agent_alive=[True, True, False], agent_energy=[10.0, 10.0, 10.0]),
    )
    summary = charge_family_head_residual_cost(runner, alive, generation=0, tick=0)

    assert summary["charged_organism_ticks"] == 2
    assert summary["neutral_charges"] == 1
    assert summary["duplicate_charges"] == 0
    assert summary["omitted_charges"] == 0
    assert summary["total_cost"] == pytest.approx(0.005)
    assert alive[0].energy == pytest.approx(9.995)
    assert alive[1].energy == pytest.approx(10.0)
    assert alive[2].energy == pytest.approx(10.0)


def test_polarch6_authority_disabled_returns_original_logits() -> None:
    alive = [_agent(0, {"attack": 0.4, "movement": -0.4})]
    cfg = _cfg(policy_family_head_residual_genome_enable=True, policy_family_head_residual_authority_enable=False)
    logits = torch.tensor([[0.2, -0.1, 0.0, 0.15, 0.45, 0.05, -0.25]], dtype=torch.float32)
    transformed, summary = apply_authority_to_logits_torch(logits, alive=alive, row_indices=[0], sim_cfg=cfg)

    assert transformed is logits
    assert summary is None


def test_polarch12_binary_gate_raw_centered_probability_equivalence() -> None:
    logits = torch.tensor([[0.2, -0.1, 0.0, 0.15, 0.45, 0.05, -0.25]], dtype=torch.float32)
    residuals = torch.tensor([[-0.3, 0.3, 0.0, 0.0]], dtype=torch.float32)
    opportunities = torch.tensor([[1.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    valid = torch.ones((1, 4), dtype=torch.float32)

    raw, centered = binary_opportunity_conditioned_residuals_torch(residuals, opportunities, valid)
    raw_probs = torch.softmax(authority_log_probs_from_logits_torch(logits, raw), dim=1)
    centered_probs = torch.softmax(authority_log_probs_from_logits_torch(logits, centered), dim=1)

    assert raw[0, 1].item() == pytest.approx(0.0)
    assert centered[0, 1].item() != pytest.approx(0.0)
    assert torch.max(torch.abs(raw_probs - centered_probs)).item() == pytest.approx(0.0, abs=1e-6)


def test_polarch12_binary_gate_all_true_equals_unconditioned_and_all_false_historical() -> None:
    alive = [_agent(0, {"movement": -0.3, "gather": 0.3})]
    cfg_gate = _cfg(
        policy_family_head_residual_genome_enable=True,
        policy_family_head_residual_authority_enable=True,
        policy_family_head_binary_opportunity_gate_enable=True,
    )
    cfg_ungated = _cfg(policy_family_head_residual_genome_enable=True, policy_family_head_residual_authority_enable=True)
    logits = torch.tensor([[0.2, -0.1, 0.0, 0.15, 0.45, 0.05, -0.25]], dtype=torch.float32)
    all_true = torch.ones((1, 4), dtype=torch.float32)
    all_false = torch.zeros((1, 4), dtype=torch.float32)
    valid = torch.ones((1, 4), dtype=torch.float32)
    valid_actions = torch.ones((1, 7), dtype=torch.float32)

    gated_true, summary_true = apply_authority_to_logits_torch(
        logits,
        alive=alive,
        row_indices=[0],
        sim_cfg=cfg_gate,
        opportunity_vectors=all_true,
        valid_family_mask=valid,
        valid_action_mask=valid_actions,
    )
    ungated, _ = apply_authority_to_logits_torch(logits, alive=alive, row_indices=[0], sim_cfg=cfg_ungated)
    gated_false, summary_false = apply_authority_to_logits_torch(
        logits,
        alive=alive,
        row_indices=[0],
        sim_cfg=cfg_gate,
        opportunity_vectors=all_false,
        valid_family_mask=valid,
        valid_action_mask=valid_actions,
    )

    assert summary_true["authority_mode"] == "binary_executable_opportunity_conditioned"
    assert summary_true["raw_centered_probability_error"] == pytest.approx(0.0, abs=1e-6)
    assert torch.max(torch.abs(torch.softmax(gated_true, dim=1) - torch.softmax(ungated, dim=1))).item() == pytest.approx(0.0, abs=1e-6)
    assert torch.max(torch.abs(torch.softmax(gated_false, dim=1) - torch.softmax(logits, dim=1))).item() == pytest.approx(0.0, abs=1e-6)
    assert summary_false["raw_gated_residual_sum"] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_polarch12_l1_point_six_cost_is_independent_of_gate_state() -> None:
    alive_gate = [_agent(0, {"movement": -0.3, "gather": 0.3}, energy=10.0)]
    alive_ungated = [_agent(0, {"movement": -0.3, "gather": 0.3}, energy=10.0)]
    cfg_gate = _cfg(
        policy_family_head_residual_genome_enable=True,
        policy_family_head_residual_authority_enable=True,
        policy_family_head_binary_opportunity_gate_enable=True,
        policy_family_head_residual_cost_enable=True,
        policy_family_head_residual_cost_coefficient=TECHNICAL_COST_COEFFICIENT,
    )
    cfg_ungated = _cfg(
        policy_family_head_residual_genome_enable=True,
        policy_family_head_residual_authority_enable=True,
        policy_family_head_residual_cost_enable=True,
        policy_family_head_residual_cost_coefficient=TECHNICAL_COST_COEFFICIENT,
    )
    runner_gate = SimpleNamespace(sim_cfg=cfg_gate, env=SimpleNamespace(agent_alive=[True], agent_energy=[10.0]))
    runner_ungated = SimpleNamespace(sim_cfg=cfg_ungated, env=SimpleNamespace(agent_alive=[True], agent_energy=[10.0]))

    gate_summary = charge_family_head_residual_cost(runner_gate, alive_gate, generation=0, tick=0)
    ungated_summary = charge_family_head_residual_cost(runner_ungated, alive_ungated, generation=0, tick=0)

    assert gate_summary["total_cost"] == pytest.approx(0.003)
    assert ungated_summary["total_cost"] == pytest.approx(0.003)
    assert alive_gate[0].energy == pytest.approx(alive_ungated[0].energy)
