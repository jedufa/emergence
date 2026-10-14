from __future__ import annotations

"""
agents/gru_policy.py

Defines the GRUPolicy module used by GRU agents.

What this file does:
- Implements a GRUCell-based policy network:
    obs -> encoder -> GRUCell -> head -> action logits
- Exists so the runner can extract parameters consistently into a per-slot GRU
  weight bank for fast batched inference.

Critical invariant for banked execution:
- The runner stacks GRU parameters across agents (torch.stack), so all GRU policies
  must share the same hidden_dim.
- This project fixes GRU hidden_dim = 64. Evolution controls *effective* memory
  via genome.memory_dim (masking in runner_policy.py) and pays upkeep per step.
- Therefore, this module enforces hidden_dim=64 regardless of what callers request.
"""

import torch
import torch.nn as nn


_FIXED_GRU_HIDDEN = 64


class GRUPolicy(nn.Module):
    """
    Inference-time GRU policy.
    NOTE: Runner uses a *batched weight bank* + forward_batched_gru for speed.
    This module exists to define structure and to make parameter extraction consistent.
    """
    def __init__(self, obs_size: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.obs_size = int(obs_size)

        # Enforce fixed hidden size for banked stacking
        self.hidden_dim = int(_FIXED_GRU_HIDDEN)

        # Keep what was requested for debugging (optional)
        self.requested_hidden_dim = int(hidden_dim)

        self.n_actions = int(n_actions)

        self.encoder = nn.Linear(self.obs_size, self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, self.n_actions)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.tanh(self.encoder(x))
        h_next = self.gru(z, h)
        logits = self.head(h_next)
        return logits, h_next

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
