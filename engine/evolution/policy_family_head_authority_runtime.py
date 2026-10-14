from __future__ import annotations

"""Runtime helpers for default-off family-head residual authority and cost."""

import math
from typing import Any, Iterable

import numpy as np
import torch

from agents.family_head_residuals import (
    FAMILY_HEAD_RESIDUAL_FIELD,
    FAMILY_HEAD_RESIDUAL_FAMILY_ORDER,
    family_head_residual_effective_map,
    family_head_residual_semantic_digest,
)
from evolution.policy_action_family import FAMILY_TO_ACTION_IDS


AUTHORITY_SCHEMA_VERSION = "family_head_residual_authority_v1"
MAINTENANCE_COST_SCHEMA_VERSION = "family_head_residual_l1_maintenance_cost_v1"
BINARY_OPPORTUNITY_AUTHORITY_SCHEMA_VERSION = "binary_opportunity_conditioned_family_authority_v1"
TECHNICAL_COST_COEFFICIENT = 0.005
FAMILY_ORDER = FAMILY_HEAD_RESIDUAL_FAMILY_ORDER


def authority_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "policy_family_head_residual_authority_enable", False))


def binary_opportunity_gate_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "policy_family_head_binary_opportunity_gate_enable", False))


def cost_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "policy_family_head_residual_cost_enable", False))


def reporting_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "policy_family_head_residual_authority_reporting_enable", False))


def validate_family_head_authority_config(sim_cfg: Any) -> list[str]:
    reasons: list[str] = []
    representation = bool(getattr(sim_cfg, "policy_family_head_residual_genome_enable", False))
    authority = authority_enabled(sim_cfg)
    cost = cost_enabled(sim_cfg)
    coeff = float(getattr(sim_cfg, "policy_family_head_residual_cost_coefficient", 0.0) or 0.0)
    reporting = reporting_enabled(sim_cfg)
    binary_gate = binary_opportunity_gate_enabled(sim_cfg)
    shadow = bool(
        getattr(sim_cfg, "policy_family_head_residual_shadow_enable", False)
        or getattr(sim_cfg, "policy_family_head_shadow_enable", False)
        or getattr(sim_cfg, "policy_family_head_shadow_record_events", False)
    )
    if authority and not representation:
        reasons.append("authority_requires_family_head_residual_genome_enable")
    if cost and not representation:
        reasons.append("cost_requires_family_head_residual_genome_enable")
    if binary_gate and not authority:
        reasons.append("binary_opportunity_gate_requires_family_head_residual_authority_enable")
    if binary_gate and not representation:
        reasons.append("binary_opportunity_gate_requires_family_head_residual_genome_enable")
    if not math.isfinite(coeff) or coeff < 0.0:
        reasons.append("cost_coefficient_must_be_finite_nonnegative")
    if coeff > 0.0 and not cost:
        reasons.append("positive_cost_coefficient_requires_cost_enable")
    if reporting and not (authority or shadow):
        reasons.append("authority_reporting_requires_authority_or_shadow_context")
    return reasons


def residual_tensor_for_rows(alive: list[Any], row_indices: Iterable[int], *, device: Any, dtype: Any) -> torch.Tensor:
    rows = list(row_indices)
    values: list[list[float]] = []
    for row in rows:
        agent = alive[int(row)]
        genome = getattr(agent, "genome", None)
        block = getattr(genome, FAMILY_HEAD_RESIDUAL_FIELD, None)
        residual = family_head_residual_effective_map(block)
        values.append([float(residual[family]) for family in FAMILY_ORDER])
    if not values:
        return torch.empty((0, len(FAMILY_ORDER)), device=device, dtype=dtype)
    return torch.as_tensor(values, device=device, dtype=dtype)


def authority_log_probs_from_logits_torch(
    logits: torch.Tensor,
    residuals: torch.Tensor,
    valid_action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return primitive log-probability scores for the accepted authority equation.

    The returned tensor can be fed to the existing primitive argmax/Gumbel-Max
    chooser. Zero residuals differ from the original logits only by one row-wise
    additive constant, so deterministic and stochastic choices remain identical.
    """

    if logits.ndim != 2 or int(logits.shape[1]) != 7:
        raise ValueError("family-head authority requires logits shape [N, 7]")
    if residuals.ndim != 2 or int(residuals.shape[1]) != len(FAMILY_ORDER) or int(residuals.shape[0]) != int(logits.shape[0]):
        raise ValueError("family-head authority residual tensor shape mismatch")
    if valid_action_mask is not None:
        if valid_action_mask.shape != logits.shape:
            raise ValueError("family-head authority valid action mask shape mismatch")
        logits = torch.where(
            valid_action_mask.to(device=logits.device, dtype=torch.bool),
            logits,
            torch.full_like(logits, -torch.inf),
        )
    logits = _ensure_finite_action_row(logits)
    family_logits = []
    for family in FAMILY_ORDER:
        idx = list(FAMILY_TO_ACTION_IDS[family])
        family_logits.append(torch.logsumexp(logits[:, idx], dim=1))
    family_logits_t = torch.stack(family_logits, dim=1)
    shifted = family_logits_t + residuals.to(device=logits.device, dtype=logits.dtype)
    family_log_probs = shifted - torch.logsumexp(shifted, dim=1, keepdim=True)
    out = torch.empty_like(logits)
    for family_idx, family in enumerate(FAMILY_ORDER):
        idx = list(FAMILY_TO_ACTION_IDS[family])
        within = logits[:, idx] - torch.logsumexp(logits[:, idx], dim=1, keepdim=True)
        out[:, idx] = family_log_probs[:, family_idx:family_idx + 1] + within
    return _ensure_finite_action_row(out)


def _ensure_finite_action_row(logits: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0 or logits.ndim != 2:
        return logits
    logits = torch.where(torch.isnan(logits) | torch.isposinf(logits), torch.full_like(logits, -torch.inf), logits)
    finite_any = torch.isfinite(logits).any(dim=1)
    if bool(torch.all(finite_any)):
        return logits
    out = logits.clone()
    bad_rows = torch.nonzero(~finite_any, as_tuple=False).reshape((-1,))
    if int(bad_rows.numel()) > 0:
        out[bad_rows, :] = -torch.inf
        out[bad_rows, 5] = 0.0
    return out


def binary_opportunity_conditioned_residuals_torch(
    residuals: torch.Tensor,
    opportunities: torch.Tensor,
    valid_family_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw gated and centered expressed residuals for POLARCH-12.

    `opportunities` and `valid_family_mask` are [N, 4] binary/boolean tensors
    ordered by FAMILY_ORDER. Invalid families are excluded from centering and
    receive zero expressed residual for reporting and scoring.
    """

    if residuals.ndim != 2 or int(residuals.shape[1]) != len(FAMILY_ORDER):
        raise ValueError("binary opportunity gate residuals require shape [N, 4]")
    if opportunities.shape != residuals.shape or valid_family_mask.shape != residuals.shape:
        raise ValueError("binary opportunity gate tensors must share residual shape")
    opp = opportunities.to(device=residuals.device, dtype=residuals.dtype).clamp(min=0.0, max=1.0)
    valid = valid_family_mask.to(device=residuals.device, dtype=residuals.dtype).clamp(min=0.0, max=1.0)
    raw = residuals * opp
    denom = torch.clamp(torch.sum(valid, dim=1, keepdim=True), min=1.0)
    mean = torch.sum(raw * valid, dim=1, keepdim=True) / denom
    centered = torch.where(valid > 0.0, raw - mean, torch.zeros_like(raw))
    return raw, centered


def _raw_centered_probability_error(
    logits: torch.Tensor,
    raw_residuals: torch.Tensor,
    centered_residuals: torch.Tensor,
    valid_action_mask: torch.Tensor | None = None,
) -> float:
    if logits.numel() == 0:
        return 0.0
    raw_scores = authority_log_probs_from_logits_torch(logits, raw_residuals, valid_action_mask=valid_action_mask)
    centered_scores = authority_log_probs_from_logits_torch(logits, centered_residuals, valid_action_mask=valid_action_mask)
    raw_probs = torch.softmax(raw_scores, dim=1)
    centered_probs = torch.softmax(centered_scores, dim=1)
    err = torch.max(torch.abs(raw_probs - centered_probs))
    return float(err.detach().to(torch.float64).cpu().item())


def apply_authority_to_logits_torch(
    logits: torch.Tensor,
    *,
    alive: list[Any],
    row_indices: Iterable[int],
    sim_cfg: Any,
    opportunity_vectors: torch.Tensor | None = None,
    valid_family_mask: torch.Tensor | None = None,
    valid_action_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    if not authority_enabled(sim_cfg):
        return logits, None
    reasons = validate_family_head_authority_config(sim_cfg)
    if reasons:
        raise RuntimeError("POLARCH6-AUTHORITY-CONFIG-INVALID: " + ",".join(reasons))
    residuals = residual_tensor_for_rows(alive, row_indices, device=logits.device, dtype=logits.dtype)
    mode = "unconditioned_family_head"
    raw_gated_residuals = None
    centered_residuals = residuals
    raw_centered_probability_error = 0.0
    if binary_opportunity_gate_enabled(sim_cfg):
        mode = "binary_executable_opportunity_conditioned"
        if opportunity_vectors is None or valid_family_mask is None:
            raise RuntimeError("POLARCH12-BINARY-GATE-OPPORTUNITY-INCOMPLETE")
        raw_gated_residuals, centered_residuals = binary_opportunity_conditioned_residuals_torch(
            residuals,
            opportunity_vectors,
            valid_family_mask,
        )
        if valid_action_mask is None:
            raise RuntimeError("POLARCH12-BINARY-GATE-ACTION-MASK-INCOMPLETE")
        raw_centered_probability_error = _raw_centered_probability_error(
            logits,
            raw_gated_residuals,
            centered_residuals,
            valid_action_mask=valid_action_mask,
        )
    authority_logits = authority_log_probs_from_logits_torch(logits, centered_residuals, valid_action_mask=valid_action_mask if mode == "binary_executable_opportunity_conditioned" else None)
    family_probs_before = _family_probabilities_from_logits(logits, valid_action_mask=valid_action_mask if mode == "binary_executable_opportunity_conditioned" else None)
    family_probs_after = _family_probabilities_from_logits(authority_logits)
    family_delta = family_probs_after - family_probs_before
    summary = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "authority_mode": mode,
        "binary_opportunity_schema_version": BINARY_OPPORTUNITY_AUTHORITY_SCHEMA_VERSION if mode == "binary_executable_opportunity_conditioned" else "",
        "evaluated_decisions": int(logits.shape[0]),
        "random_draws_added": 0,
        "residual_gain": 1.0,
        "family_order": list(FAMILY_ORDER),
        "family_probability_before_sum": _tensor_sum_list(family_probs_before),
        "family_probability_after_sum": _tensor_sum_list(family_probs_after),
        "family_probability_delta_sum": _tensor_sum_list(family_delta),
        "residual_sum": _tensor_sum_list(residuals),
        "stored_residual_sum": _tensor_sum_list(residuals),
        "raw_gated_residual_sum": _tensor_sum_list(raw_gated_residuals) if raw_gated_residuals is not None else [0.0 for _ in FAMILY_ORDER],
        "centered_expressed_residual_sum": _tensor_sum_list(centered_residuals),
        "opportunity_sum": _tensor_sum_list(opportunity_vectors.to(device=logits.device, dtype=logits.dtype)) if opportunity_vectors is not None else [0.0 for _ in FAMILY_ORDER],
        "valid_family_count_sum": _tensor_sum_list(valid_family_mask.to(device=logits.device, dtype=logits.dtype)) if valid_family_mask is not None else [0.0 for _ in FAMILY_ORDER],
        "raw_centered_probability_error": float(raw_centered_probability_error),
        "max_abs_log_score_delta": float(torch.max(torch.abs(authority_logits - logits)).detach().cpu().item()) if logits.numel() else 0.0,
        "probability_error": _probability_error(authority_logits),
    }
    return authority_logits, summary


def _family_probabilities_from_logits(logits: torch.Tensor, *, valid_action_mask: torch.Tensor | None = None) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.empty((0, len(FAMILY_ORDER)), device=logits.device, dtype=logits.dtype)
    if valid_action_mask is not None:
        if valid_action_mask.shape != logits.shape:
            raise ValueError("family-head authority valid action mask shape mismatch")
        logits = torch.where(
            valid_action_mask.to(device=logits.device, dtype=torch.bool),
            logits,
            torch.full_like(logits, -torch.inf),
        )
    logits = _ensure_finite_action_row(logits)
    probs = torch.softmax(logits, dim=1)
    family_probs = []
    for family in FAMILY_ORDER:
        idx = list(FAMILY_TO_ACTION_IDS[family])
        family_probs.append(torch.sum(probs[:, idx], dim=1))
    return torch.stack(family_probs, dim=1)


def _tensor_sum_list(value: torch.Tensor) -> list[float]:
    if value.numel() == 0:
        return [0.0 for _ in FAMILY_ORDER]
    return [float(x) for x in torch.sum(value.detach().to(torch.float64), dim=0).cpu().tolist()]


def _probability_error(log_prob_scores: torch.Tensor) -> float:
    if log_prob_scores.numel() == 0:
        return 0.0
    log_prob_scores = _ensure_finite_action_row(log_prob_scores)
    probs = torch.softmax(log_prob_scores, dim=1)
    err = torch.max(torch.abs(torch.sum(probs, dim=1) - 1.0))
    return float(err.detach().cpu().item())


def residual_l1_for_agent(agent: Any) -> float:
    genome = getattr(agent, "genome", None)
    block = getattr(genome, FAMILY_HEAD_RESIDUAL_FIELD, None)
    residual = family_head_residual_effective_map(block)
    return float(sum(abs(float(residual[family])) for family in FAMILY_ORDER))


def charge_family_head_residual_cost(
    runner: Any,
    alive: list[Any],
    *,
    generation: int,
    tick: int,
) -> dict[str, Any]:
    sim_cfg = getattr(runner, "sim_cfg", None)
    if not cost_enabled(sim_cfg):
        return {
            "schema_version": MAINTENANCE_COST_SCHEMA_VERSION,
            "enabled": False,
            "charged_organism_ticks": 0,
            "total_cost": 0.0,
        }
    reasons = validate_family_head_authority_config(sim_cfg)
    if reasons:
        raise RuntimeError("POLARCH6-COST-CONFIG-INVALID: " + ",".join(reasons))
    coeff = float(getattr(sim_cfg, "policy_family_head_residual_cost_coefficient", 0.0) or 0.0)
    env_alive = getattr(getattr(runner, "env", None), "agent_alive", None)
    env_energy = getattr(getattr(runner, "env", None), "agent_energy", None)
    charged = 0
    omitted = 0
    neutral = 0
    total = 0.0
    for agent in list(alive):
        if not bool(getattr(agent, "alive", True)):
            continue
        aid = int(getattr(agent, "agent_id", -1))
        if env_alive is not None and (aid < 0 or aid >= len(env_alive) or not bool(env_alive[aid])):
            continue
        l1 = residual_l1_for_agent(agent)
        cost = float(coeff) * float(l1)
        if abs(cost) <= 1e-12:
            neutral += 1
        if not math.isfinite(cost) or cost < 0.0:
            omitted += 1
            continue
        if cost > 0.0:
            agent.energy = float(getattr(agent, "energy", 0.0)) - cost
            if env_energy is not None and 0 <= aid < len(env_energy):
                env_energy[aid] = np.asarray(env_energy).dtype.type(float(env_energy[aid]) - cost)
        charged += 1
        total += cost
    summary = {
        "schema_version": MAINTENANCE_COST_SCHEMA_VERSION,
        "enabled": True,
        "generation": int(generation),
        "tick": int(tick),
        "coefficient": float(coeff),
        "living_organism_ticks": int(sum(1 for agent in alive if bool(getattr(agent, "alive", True)))),
        "charged_organism_ticks": int(charged),
        "omitted_charges": int(omitted),
        "duplicate_charges": 0,
        "neutral_charges": int(neutral),
        "total_cost": float(total),
        "cost_timing": "post_env_pre_internal_maintenance",
    }
    _merge_cost_summary(runner, summary)
    return summary


def _merge_cost_summary(runner: Any, summary: dict[str, Any]) -> None:
    total = getattr(runner, "_policy_family_head_residual_cost_total", None)
    if not isinstance(total, dict):
        total = {
            "schema_version": MAINTENANCE_COST_SCHEMA_VERSION,
            "enabled": True,
            "living_organism_ticks": 0,
            "charged_organism_ticks": 0,
            "omitted_charges": 0,
            "duplicate_charges": 0,
            "neutral_charges": 0,
            "total_cost": 0.0,
        }
    for key in ("living_organism_ticks", "charged_organism_ticks", "omitted_charges", "duplicate_charges", "neutral_charges"):
        total[key] = int(total.get(key, 0) or 0) + int(summary.get(key, 0) or 0)
    total["total_cost"] = float(total.get("total_cost", 0.0) or 0.0) + float(summary.get("total_cost", 0.0) or 0.0)
    runner._policy_family_head_residual_cost_total = total


def config_hash_fields() -> dict[str, Any]:
    return {
        "policy_family_head_residual_authority_enable": False,
        "policy_family_head_residual_cost_enable": False,
        "policy_family_head_residual_cost_coefficient": 0.0,
        "policy_family_head_binary_opportunity_gate_enable": False,
        "policy_family_head_binary_opportunity_gate_reporting_enable": False,
        "policy_family_head_binary_opportunity_gate_event_sample_limit": 2048,
        "policy_family_head_residual_authority_reporting_enable": False,
    }


def residual_digest_for_agent(agent: Any) -> str:
    genome = getattr(agent, "genome", None)
    block = getattr(genome, FAMILY_HEAD_RESIDUAL_FIELD, None)
    return family_head_residual_semantic_digest(block)
