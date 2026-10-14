# agents/batched_policy.py
from __future__ import annotations
import torch

from env.gridworld_pkg.constants import ACT_ATTACK


def _batched_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bmm(w, x.unsqueeze(-1)).squeeze(-1) + b


@torch.no_grad()
def forward_batched_mlp(
    x: torch.Tensor,
    w1: torch.Tensor, b1: torch.Tensor,
    w2: torch.Tensor, b2: torch.Tensor,
    w3: torch.Tensor, b3: torch.Tensor,
    attack_bias: torch.Tensor | None = None,
    action_temperature: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Returns action logits.

    attack_bias:
        Added directly to ACT_ATTACK logit (per-agent).

    action_temperature:
        Per-agent logits temperature.
        Applied as: logits / temperature
    """
    h1 = torch.tanh(_batched_linear(x, w1, b1))
    h2 = torch.tanh(_batched_linear(h1, w2, b2))
    out = _batched_linear(h2, w3, b3)

    if attack_bias is not None:
        out[:, ACT_ATTACK] += attack_bias

    # Sprint C: evolvable stochasticity
    if action_temperature is not None:
        # Clamp for numerical safety (defensive, not evolutionary)
        temp = torch.clamp(action_temperature, min=0.05)
        out = out / temp.unsqueeze(-1)

    return out
