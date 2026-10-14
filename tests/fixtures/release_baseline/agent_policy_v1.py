from __future__ import annotations

"""agents/agent_policy.py

MLP policy definition used by non-GRU agents.

Notes
-----
- This is a simple feed-forward policy network used in the parameter-bank path.
- Evolution operates by crossover/mutation of genomes and, for MLP agents, by
  crossover of policy parameters (see evolution/reproduction.py).

We keep this intentionally small and boring. Complexity belongs in the world
pressure, not in your ego.
"""

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def clone_parameters(src: nn.Module, dst: nn.Module) -> None:
    """Copy parameters from src to dst in-place."""
    for p_src, p_dst in zip(src.parameters(), dst.parameters()):
        p_dst.data.copy_(p_src.data)
