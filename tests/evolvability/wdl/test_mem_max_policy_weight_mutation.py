from __future__ import annotations

import torch

from agents.gru_policy import GRUPolicy
from evolution.reproduction import mutate_policy_weights
from simulation.rng_manager import RNGManager
from simulation.runner_pkg.runner_config import SimConfig


def _cfg(*, enabled: bool, std: float = 0.01) -> SimConfig:
    cfg = SimConfig(seed=101)
    cfg.policy_weight_mutation_enable = bool(enabled)
    cfg.policy_weight_mutation_std = float(std)
    return cfg


def _clone_policy(policy: GRUPolicy) -> GRUPolicy:
    clone = GRUPolicy(policy.obs_size, policy.hidden_dim, policy.n_actions)
    clone.load_state_dict(policy.state_dict())
    return clone


def _state(policy: GRUPolicy) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in policy.state_dict().items()}


def test_policy_weight_mutation_is_default_off_and_preserves_weights() -> None:
    policy = GRUPolicy(obs_size=12, hidden_dim=16, n_actions=7)
    before = _state(policy)

    summary = mutate_policy_weights(policy, rng_manager=RNGManager(101), sim_cfg=_cfg(enabled=False))

    assert summary["mutated"] is False
    for key, value in policy.state_dict().items():
        assert torch.equal(value, before[key])


def test_policy_weight_mutation_changes_recurrent_and_action_head_weights_when_enabled() -> None:
    policy = GRUPolicy(obs_size=12, hidden_dim=16, n_actions=7)
    before = _state(policy)

    summary = mutate_policy_weights(policy, rng_manager=RNGManager(101), sim_cfg=_cfg(enabled=True, std=0.05))

    assert summary["mutated"] is True
    assert summary["parameter_count"] > 0
    assert not torch.equal(policy.gru.weight_hh, before["gru.weight_hh"])
    assert not torch.equal(policy.head.weight, before["head.weight"])


def test_policy_weight_mutation_is_same_seed_deterministic() -> None:
    source = GRUPolicy(obs_size=12, hidden_dim=16, n_actions=7)
    left = _clone_policy(source)
    right = _clone_policy(source)

    mutate_policy_weights(left, rng_manager=RNGManager(777), sim_cfg=_cfg(enabled=True, std=0.02))
    mutate_policy_weights(right, rng_manager=RNGManager(777), sim_cfg=_cfg(enabled=True, std=0.02))

    for key, value in left.state_dict().items():
        assert torch.equal(value, right.state_dict()[key]), key
