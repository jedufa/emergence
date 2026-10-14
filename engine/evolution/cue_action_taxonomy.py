from __future__ import annotations

"""Layered cue-action taxonomy for MEM-MAX diagnostics.

The levels here deliberately separate raw counterfactual policy responsiveness
from effective runtime memory retention.  The classifier fails closed whenever
the probe was not run with actual cue exposure, effective memory, and a recurrent
policy.
"""

from typing import Any

import torch

from agents.gru_policy import GRUPolicy
from env.gridworld_pkg.constants import ACT_MOVE_E, ACT_MOVE_W


SCHEMA_VERSION = "mem_max_cue_action_taxonomy_v1"
DEFAULT_THRESHOLDS = {
    "raw_logit_l2_min": 1.0e-6,
    "hidden_state_l2_min": 1.0e-6,
    "target_probability_delta_min": 0.02,
    "target_aligned_score_min": 0.02,
}


def _softmax_prob(logits: torch.Tensor, idx: int) -> float:
    probs = torch.softmax(logits.detach().cpu().float(), dim=0)
    return float(probs[int(idx)].item()) if 0 <= int(idx) < int(probs.numel()) else 0.0


def effective_runtime_cue_action_probe(
    policy: object,
    *,
    obs_size: int,
    cue_a_index: int | None,
    cue_b_index: int | None,
    memory_dim: int,
    cue_enabled: bool,
    memory_ablated: bool,
    hidden_steps: int = 2,
    move_w_index: int = ACT_MOVE_W,
    move_e_index: int = ACT_MOVE_E,
) -> dict[str, Any]:
    """Run a deterministic cue -> hidden-window probe with runtime memory gates.

    This does not draw RNG and does not inspect outcome metrics.  It applies the
    same core memory restrictions used by runtime execution: MLP policies cannot
    retain hidden state; `memory_dim <= 0` cannot pass effective memory; reset
    controls erase recurrent state each step.
    """

    size = max(1, int(obs_size))
    if policy is None or not hasattr(policy, "state_dict"):
        return {"schema_version": SCHEMA_VERSION, "available": False, "reason": "missing_policy"}
    if cue_a_index is None or cue_b_index is None:
        return {"schema_version": SCHEMA_VERSION, "available": False, "reason": "missing_cue_indices"}
    ca = int(cue_a_index)
    cb = int(cue_b_index)
    if not (0 <= ca < size and 0 <= cb < size):
        return {"schema_version": SCHEMA_VERSION, "available": False, "reason": "cue_index_out_of_bounds"}

    obs_a = torch.zeros(size, dtype=torch.float32)
    obs_b = torch.zeros(size, dtype=torch.float32)
    if bool(cue_enabled):
        obs_a[ca] = 1.0
        obs_b[cb] = 1.0
    obs_hidden = torch.zeros(size, dtype=torch.float32)

    if not isinstance(policy, GRUPolicy):
        with torch.no_grad():
            la = policy(obs_a)
            lb = policy(obs_b)
            final_a = policy(obs_hidden)
            final_b = policy(obs_hidden)
        return {
            "schema_version": SCHEMA_VERSION,
            "available": True,
            "probe_kind": "effective_runtime_hidden_window",
            "policy_type": type(policy).__name__,
            "cue_enabled": bool(cue_enabled),
            "memory_ablated": bool(memory_ablated),
            "memory_dim": int(memory_dim),
            "hidden_steps": int(hidden_steps),
            "raw_action_logit_l2_divergence": float(torch.norm(la.detach().cpu().float() - lb.detach().cpu().float()).item()),
            "hidden_state_l2_after_cue": 0.0,
            "hidden_state_l2_after_hidden": 0.0,
            "action_logit_l2_after_hidden": float(
                torch.norm(final_a.detach().cpu().float() - final_b.detach().cpu().float()).item()
            ),
            "top1_after_cue_a": int(torch.argmax(final_a).item()),
            "top1_after_cue_b": int(torch.argmax(final_b).item()),
            "target_a_direction_probability_proxy": _softmax_prob(final_a, move_w_index),
            "target_b_direction_probability_proxy": _softmax_prob(final_b, move_e_index),
            "target_aligned_score_proxy": 0.0,
        }

    hidden_dim = int(getattr(policy, "hidden_dim", 64) or 64)
    keep = max(0, min(int(memory_dim), hidden_dim))
    mask = torch.zeros(hidden_dim, dtype=torch.float32)
    if keep > 0:
        mask[:keep] = 1.0

    def run_sequence(first_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            h = torch.zeros(hidden_dim, dtype=torch.float32)
            if bool(memory_ablated):
                h = torch.zeros_like(h)
            out_cue, h = policy(first_obs, h)
            h = h.detach().cpu().float() * mask
            h_after_cue = h.clone()
            for _ in range(max(0, int(hidden_steps))):
                if bool(memory_ablated):
                    h = torch.zeros_like(h)
                out, h = policy(obs_hidden, h)
                h = h.detach().cpu().float() * mask
            return out_cue.detach().cpu().float(), out.detach().cpu().float(), h

    raw_a, final_a, h_a = run_sequence(obs_a)
    raw_b, final_b, h_b = run_sequence(obs_b)
    hidden_sep = float(torch.norm(h_a - h_b).item())
    target_a_prob = _softmax_prob(final_a, move_w_index)
    target_b_prob = _softmax_prob(final_b, move_e_index)
    cross_a_prob = _softmax_prob(final_a, move_e_index)
    cross_b_prob = _softmax_prob(final_b, move_w_index)
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "probe_kind": "effective_runtime_hidden_window",
        "policy_type": type(policy).__name__,
        "cue_enabled": bool(cue_enabled),
        "memory_ablated": bool(memory_ablated),
        "memory_dim": int(memory_dim),
        "hidden_steps": int(hidden_steps),
        "raw_action_logit_l2_divergence": float(torch.norm(raw_a - raw_b).item()),
        "hidden_state_l2_after_cue": hidden_sep,
        "hidden_state_l2_after_hidden": hidden_sep,
        "action_logit_l2_after_hidden": float(torch.norm(final_a - final_b).item()),
        "top1_after_cue_a": int(torch.argmax(final_a).item()),
        "top1_after_cue_b": int(torch.argmax(final_b).item()),
        "target_a_direction_probability_proxy": target_a_prob,
        "target_b_direction_probability_proxy": target_b_prob,
        "target_aligned_score_proxy": float((target_a_prob - cross_a_prob) + (target_b_prob - cross_b_prob)),
    }


def classify_cue_action(
    *,
    raw_probe: dict[str, Any] | None = None,
    effective_probe: dict[str, Any] | None = None,
    runtime_semantics: dict[str, Any] | None = None,
    policy_type: str = "",
    memory_dim: int = 0,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    t = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    raw = raw_probe or {}
    eff = effective_probe or {}
    sem = runtime_semantics or {}

    level0 = bool(raw.get("available", False) or eff.get("available", False))
    raw_l2 = float(raw.get("action_logit_l2_divergence", eff.get("raw_action_logit_l2_divergence", 0.0)) or 0.0)
    raw_top_changed = bool(
        int(raw.get("top1_cue_a", eff.get("top1_after_cue_a", -1)) or -1)
        != int(raw.get("top1_cue_b", eff.get("top1_after_cue_b", -1)) or -1)
    )
    level1 = bool(level0 and (raw_l2 > float(t["raw_logit_l2_min"]) or raw_top_changed))

    actual_cue_exposure = bool(sem.get("cue_enabled", eff.get("cue_enabled", False))) and bool(
        sem.get("observation_enabled", True)
    )
    memory_effective = (
        str(policy_type or eff.get("policy_type", "")) == "GRUPolicy"
        and int(memory_dim if memory_dim is not None else eff.get("memory_dim", 0) or 0) > 0
        and not bool(sem.get("memory_ablated", eff.get("memory_ablated", False)))
        and not bool(sem.get("reset_memory_each_tick", False))
    )
    hidden_sep = float(eff.get("hidden_state_l2_after_hidden", 0.0) or 0.0)
    level2 = bool(
        eff.get("available", False)
        and eff.get("probe_kind") == "effective_runtime_hidden_window"
        and actual_cue_exposure
        and memory_effective
        and hidden_sep > float(t["hidden_state_l2_min"])
    )

    action_l2 = float(eff.get("action_logit_l2_after_hidden", 0.0) or 0.0)
    top_changed = int(eff.get("top1_after_cue_a", -1) or -1) != int(eff.get("top1_after_cue_b", -1) or -1)
    target_delta = abs(float(eff.get("target_a_direction_probability_proxy", 0.0) or 0.0) - float(eff.get("target_b_direction_probability_proxy", 0.0) or 0.0))
    level3 = bool(level2 and (top_changed or action_l2 > float(t["target_probability_delta_min"]) or target_delta > float(t["target_probability_delta_min"])))
    aligned_score = float(eff.get("target_aligned_score_proxy", 0.0) or 0.0)
    level4 = bool(level3 and aligned_score > float(t["target_aligned_score_min"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "level0_probe_available": bool(level0),
        "level1_raw_responsive": bool(level1),
        "level2_effective_runtime_retention": bool(level2),
        "level3_action_divergent": bool(level3),
        "level4_target_aligned": bool(level4),
        "actual_cue_exposure": bool(actual_cue_exposure),
        "memory_effective": bool(memory_effective),
        "raw_logit_l2": raw_l2,
        "hidden_state_l2_after_hidden": hidden_sep,
        "action_logit_l2_after_hidden": action_l2,
        "target_aligned_score_proxy": aligned_score,
    }


def level_counts(classifications: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "level0_probe_available",
        "level1_raw_responsive",
        "level2_effective_runtime_retention",
        "level3_action_divergent",
        "level4_target_aligned",
    )
    return {key: int(sum(1 for row in classifications if bool(row.get(key, False)))) for key in keys}
