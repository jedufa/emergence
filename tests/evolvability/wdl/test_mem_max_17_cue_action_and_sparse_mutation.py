from __future__ import annotations

import copy

import torch

from agents.agent_policy import MLPPolicy
from agents.gru_policy import GRUPolicy
from evolution.cue_action_taxonomy import classify_cue_action, effective_runtime_cue_action_probe
from evolution.policy_lineage_telemetry import policy_hash
from evolution.reproduction import mutate_policy_weights
from simulation.rng_manager import RNGManager
from simulation.runner_pkg.runner_config import SimConfig


def _sparse_cfg(seed: int = 1701, *, rate: float = 0.05, std: float = 0.02) -> SimConfig:
    cfg = SimConfig(seed=seed)
    cfg.policy_weight_mutation_enable = True
    cfg.policy_weight_mutation_mode = "sparse"
    cfg.policy_weight_mutation_rate = rate
    cfg.policy_weight_mutation_std = std
    return cfg


def test_effective_runtime_probe_useful_gru_can_retain_cue() -> None:
    policy = GRUPolicy(obs_size=8, hidden_dim=64, n_actions=7)
    probe = effective_runtime_cue_action_probe(
        policy,
        obs_size=8,
        cue_a_index=0,
        cue_b_index=1,
        memory_dim=8,
        cue_enabled=True,
        memory_ablated=False,
        hidden_steps=2,
    )
    result = classify_cue_action(
        raw_probe={},
        effective_probe=probe,
        runtime_semantics={"cue_enabled": True, "observation_enabled": True, "memory_ablated": False},
        policy_type="GRUPolicy",
        memory_dim=8,
    )

    assert probe["available"] is True
    assert result["level2_effective_runtime_retention"] is True


def test_effective_runtime_probe_fails_closed_for_ablation_cue_off_and_mlp() -> None:
    gru = GRUPolicy(obs_size=8, hidden_dim=64, n_actions=7)
    mlp = MLPPolicy(obs_size=8, hidden_size=16, n_actions=7)

    ablated = effective_runtime_cue_action_probe(
        gru,
        obs_size=8,
        cue_a_index=0,
        cue_b_index=1,
        memory_dim=8,
        cue_enabled=True,
        memory_ablated=True,
        hidden_steps=2,
    )
    cue_off = effective_runtime_cue_action_probe(
        gru,
        obs_size=8,
        cue_a_index=0,
        cue_b_index=1,
        memory_dim=8,
        cue_enabled=False,
        memory_ablated=False,
        hidden_steps=2,
    )
    no_memory = effective_runtime_cue_action_probe(
        gru,
        obs_size=8,
        cue_a_index=0,
        cue_b_index=1,
        memory_dim=0,
        cue_enabled=True,
        memory_ablated=False,
        hidden_steps=2,
    )
    mlp_probe = effective_runtime_cue_action_probe(
        mlp,
        obs_size=8,
        cue_a_index=0,
        cue_b_index=1,
        memory_dim=8,
        cue_enabled=True,
        memory_ablated=False,
        hidden_steps=2,
    )

    for probe, semantics, policy_type, memory_dim in (
        (ablated, {"cue_enabled": True, "observation_enabled": True, "memory_ablated": True}, "GRUPolicy", 8),
        (cue_off, {"cue_enabled": False, "observation_enabled": True, "memory_ablated": False}, "GRUPolicy", 8),
        (no_memory, {"cue_enabled": True, "observation_enabled": True, "memory_ablated": False}, "GRUPolicy", 0),
        (mlp_probe, {"cue_enabled": True, "observation_enabled": True, "memory_ablated": False}, "MLPPolicy", 8),
    ):
        result = classify_cue_action(
            raw_probe={},
            effective_probe=probe,
            runtime_semantics=semantics,
            policy_type=policy_type,
            memory_dim=memory_dim,
        )
        assert result["level2_effective_runtime_retention"] is False


def test_sparse_policy_mutation_is_default_off_and_deterministic() -> None:
    base = MLPPolicy(obs_size=8, hidden_size=16, n_actions=7)
    off = copy.deepcopy(base)
    off_summary = mutate_policy_weights(off, rng_manager=RNGManager(1702), sim_cfg=SimConfig(seed=1702))

    left = copy.deepcopy(base)
    right = copy.deepcopy(base)
    left_summary = mutate_policy_weights(left, rng_manager=RNGManager(1703), sim_cfg=_sparse_cfg(1703))
    right_summary = mutate_policy_weights(right, rng_manager=RNGManager(1703), sim_cfg=_sparse_cfg(1703))

    assert off_summary["mutated"] is False
    assert policy_hash(off) == policy_hash(base)
    assert left_summary == right_summary
    assert policy_hash(left) == policy_hash(right)
    assert 0 < left_summary["mutated_parameter_count"] < left_summary["parameter_count"]
    assert policy_hash(left) != policy_hash(base)


def test_sparse_policy_mutation_is_zero_mean_not_directional() -> None:
    base = MLPPolicy(obs_size=8, hidden_size=16, n_actions=7)
    mutated = copy.deepcopy(base)
    before = {name: param.detach().clone() for name, param in mutated.named_parameters()}
    summary = mutate_policy_weights(mutated, rng_manager=RNGManager(1704), sim_cfg=_sparse_cfg(1704, rate=0.25, std=0.02))
    deltas = []
    for name, param in mutated.named_parameters():
        delta = (param.detach() - before[name]).flatten()
        deltas.append(delta[delta != 0.0])
    changed = torch.cat(deltas)

    assert summary["mode"] == "sparse"
    assert int(changed.numel()) == int(summary["mutated_parameter_count"])
    assert bool(torch.any(changed > 0.0))
    assert bool(torch.any(changed < 0.0))
