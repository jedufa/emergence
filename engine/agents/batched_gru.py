#Batched_GRU.py

from __future__ import annotations
import torch

from env.gridworld_pkg.constants import ACT_ATTACK


def _batched_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # x: (N, D)  w: (N, M, D)  b: (N, M)
    return torch.bmm(w, x.unsqueeze(-1)).squeeze(-1) + b


@torch.no_grad()
def forward_batched_gru(
    *,
    x: torch.Tensor,
    h_prev: torch.Tensor,
    encW: torch.Tensor,
    encb: torch.Tensor,
    Wih: torch.Tensor,
    bih: torch.Tensor,
    Whh: torch.Tensor,
    bhh: torch.Tensor,
    headW: torch.Tensor,
    headb: torch.Tensor,
    attack_bias: torch.Tensor | None = None,
    action_temperature: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched GRUCell forward with per-agent parameters.

    Shapes:
      x:     (N, O)
      h_prev:(N, H)
      encW:  (N, H, O)
      encb:  (N, H)
      Wih:   (N, 3H, H)
      bih:   (N, 3H)
      Whh:   (N, 3H, H)
      bhh:   (N, 3H)
      headW: (N, A, H)
      headb: (N, A)
    """
    # Encode obs -> H
    z_in = torch.tanh(_batched_linear(x, encW, encb))  # (N, H)

    # Compute input and hidden projections separately for GRU math
    i = _batched_linear(z_in, Wih, bih)      # (N, 3H)
    h = _batched_linear(h_prev, Whh, bhh)    # (N, 3H)

    # Split gates: r, z, n
    i_r, i_z, i_n = i.chunk(3, dim=1)
    h_r, h_z, h_n = h.chunk(3, dim=1)

    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)

    h_next = (1.0 - z) * n + z * h_prev  # (N, H)

    logits = _batched_linear(h_next, headW, headb)  # (N, A)

    if attack_bias is not None:
        logits[:, ACT_ATTACK] += attack_bias

    if action_temperature is not None:
        temp = torch.clamp(action_temperature, min=0.05)
        logits = logits / temp.unsqueeze(-1)

    return logits, h_next
