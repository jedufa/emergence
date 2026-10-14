from __future__ import annotations

"""Position-aware target-alignment diagnostics for MEM-MAX runs."""

from typing import Any

import torch

from env.gridworld_pkg.constants import ACT_ATTACK, ACT_GATHER, ACT_MOVE_E, ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_W, ACT_REST


SCHEMA_VERSION = "mem_max_position_aware_target_alignment_v1"
MOVE_DELTAS = {
    ACT_MOVE_N: (-1, 0),
    ACT_MOVE_S: (1, 0),
    ACT_MOVE_E: (0, 1),
    ACT_MOVE_W: (0, -1),
}
NEUTRAL_ACTIONS = {ACT_GATHER, ACT_REST, ACT_ATTACK}


def manhattan(pos: tuple[int, int], target: tuple[int, int]) -> int:
    return abs(int(pos[0]) - int(target[0])) + abs(int(pos[1]) - int(target[1]))


def action_target_delta(
    action: int,
    *,
    agent_pos: tuple[int, int],
    target_a: tuple[int, int],
    target_b: tuple[int, int],
    height: int,
    width: int,
    wrap: bool = False,
    blocked: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    action = int(action)
    y, x = int(agent_pos[0]), int(agent_pos[1])
    if action in MOVE_DELTAS:
        dy, dx = MOVE_DELTAS[action]
        ny = y + int(dy)
        nx = x + int(dx)
        if bool(wrap):
            ny %= int(height)
            nx %= int(width)
        if ny < 0 or nx < 0 or ny >= int(height) or nx >= int(width) or (blocked and (ny, nx) in blocked):
            ny, nx = y, x
    else:
        ny, nx = y, x
    before_a = manhattan((y, x), target_a)
    before_b = manhattan((y, x), target_b)
    after_a = manhattan((ny, nx), target_a)
    after_b = manhattan((ny, nx), target_b)
    da = int(before_a - after_a)
    db = int(before_b - after_b)
    if da > 0 and db <= 0:
        cls = "toward_a"
    elif db > 0 and da <= 0:
        cls = "toward_b"
    elif da > 0 and db > 0:
        cls = "toward_both"
    elif da == 0 and db == 0:
        cls = "neutral"
    else:
        cls = "away_or_blocked"
    return {
        "action": action,
        "next_pos": [int(ny), int(nx)],
        "delta_distance_a": da,
        "delta_distance_b": db,
        "target_class": cls,
        "neutral_action": bool(action in NEUTRAL_ACTIONS),
    }


def action_class_map(
    *,
    agent_pos: tuple[int, int],
    target_a: tuple[int, int],
    target_b: tuple[int, int],
    height: int,
    width: int,
    wrap: bool = False,
    blocked: set[tuple[int, int]] | None = None,
) -> dict[int, dict[str, Any]]:
    return {
        int(action): action_target_delta(
            int(action),
            agent_pos=agent_pos,
            target_a=target_a,
            target_b=target_b,
            height=height,
            width=width,
            wrap=wrap,
            blocked=blocked,
        )
        for action in (ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_E, ACT_MOVE_W, ACT_GATHER, ACT_REST, ACT_ATTACK)
    }


def score_logits_for_targets(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    agent_pos: tuple[int, int],
    target_a: tuple[int, int],
    target_b: tuple[int, int],
    height: int,
    width: int,
    wrap: bool = False,
) -> dict[str, Any]:
    classes = action_class_map(
        agent_pos=agent_pos,
        target_a=target_a,
        target_b=target_b,
        height=height,
        width=width,
        wrap=wrap,
    )
    pa = torch.softmax(logits_a.detach().cpu().float(), dim=0)
    pb = torch.softmax(logits_b.detach().cpu().float(), dim=0)

    def mass(probs: torch.Tensor, target_class: str) -> float:
        return float(sum(float(probs[action].item()) for action, meta in classes.items() if meta["target_class"] == target_class))

    def expected_delta(probs: torch.Tensor, key: str) -> float:
        return float(sum(float(probs[action].item()) * float(meta[key]) for action, meta in classes.items()))

    top_a = int(torch.argmax(logits_a.detach().cpu()).item())
    top_b = int(torch.argmax(logits_b.detach().cpu()).item())
    cue_a_toward_a = mass(pa, "toward_a") + mass(pa, "toward_both")
    cue_a_toward_b = mass(pa, "toward_b") + mass(pa, "toward_both")
    cue_b_toward_a = mass(pb, "toward_a") + mass(pb, "toward_both")
    cue_b_toward_b = mass(pb, "toward_b") + mass(pb, "toward_both")
    signed_score = float((cue_a_toward_a - cue_a_toward_b) + (cue_b_toward_b - cue_b_toward_a))
    vals_a = torch.topk(logits_a.detach().cpu().float(), k=min(2, int(logits_a.numel()))).values
    vals_b = torch.topk(logits_b.detach().cpu().float(), k=min(2, int(logits_b.numel()))).values
    return {
        "schema_version": SCHEMA_VERSION,
        "agent_pos": [int(agent_pos[0]), int(agent_pos[1])],
        "target_a": [int(target_a[0]), int(target_a[1])],
        "target_b": [int(target_b[0]), int(target_b[1])],
        "action_classes": {str(k): v for k, v in sorted(classes.items())},
        "cue_a_toward_a_mass": cue_a_toward_a,
        "cue_a_toward_b_mass": cue_a_toward_b,
        "cue_b_toward_a_mass": cue_b_toward_a,
        "cue_b_toward_b_mass": cue_b_toward_b,
        "signed_cue_consistent_target_score": signed_score,
        "top_action_cue_a": top_a,
        "top_action_cue_b": top_b,
        "top_action_cue_a_target_class": str(classes.get(top_a, {}).get("target_class", "unknown")),
        "top_action_cue_b_target_class": str(classes.get(top_b, {}).get("target_class", "unknown")),
        "top_action_changed": bool(top_a != top_b),
        "top_action_margin_cue_a": float(vals_a[0].item() - vals_a[1].item()) if int(vals_a.numel()) > 1 else 0.0,
        "top_action_margin_cue_b": float(vals_b[0].item() - vals_b[1].item()) if int(vals_b.numel()) > 1 else 0.0,
        "expected_delta_a_under_cue_a": expected_delta(pa, "delta_distance_a"),
        "expected_delta_b_under_cue_b": expected_delta(pb, "delta_distance_b"),
        "position_aware_level4": bool(signed_score > 0.02),
    }
