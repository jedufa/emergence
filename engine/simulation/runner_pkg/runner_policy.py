# simulation/runner_pkg/runner_policy.py
# Last modified: 2026-02-22 23:47:28 Europe/Vienna
"""Policy execution (inference) for SimulationRunner.

This module runs batched policy inference for both:
- MLP agents (fast parameter-bank path)
- GRU agents (stateful bank path)

Identity Pressure additions
--------------------------
- Internal drives neuromodulation:
    hunger  -> bias GATHER
    pain    -> bias REST and away from ATTACK
    fatigue -> bias REST
    fear    -> bias away from ATTACK

Default observation size remains unchanged. Patch World Step 6 may explicitly
increase observation size behind its separate observation flag. Drives are
injected directly into logits before sampling.

Additionally, each agent has an evolvable scaling gene:
- genome.drive_logit_gain
This makes the degree of neuromodulation selectable.
"""

from __future__ import annotations

from pathlib import Path

from typing import Dict, Optional, List, Any
import time
import os
import sys
import json
import hashlib
import numpy as np
import torch

from agents.batched_policy import forward_batched_mlp
from agents.batched_gru import forward_batched_gru
from agents.agent import ACTION_VOCABULARY
from agents.chemistry_coupling import observation_fidelity_and_bias_from_chem
from agents.cognition import effective_memory_dim_for_genome, memory_ablation_flags
from agents.gru_policy import GRUPolicy
from agents.perception import apply_perception_masks
from evolution.policy_family_head_authority_runtime import (
    FAMILY_ORDER as FAMILY_HEAD_AUTHORITY_ORDER,
    apply_authority_to_logits_torch,
    authority_enabled as family_head_authority_enabled,
    binary_opportunity_gate_enabled as family_head_binary_opportunity_gate_enabled,
)
from evolution.policy_action_family import FAMILY_TO_ACTION_IDS
from evolution.structural_live_modules_v1 import (
    add_rest_logit_residual_,
    execute_live_modules_batch,
    live_modules_enabled,
)
from evolution.structural_observation_routing_v1 import (
    route_observations_batch,
    routing_enabled as observation_routing_enabled,
)
from evolution.structural_output_routing_v1 import (
    add_output_routing_residuals_,
    compile_output_routing_batch,
    output_routing_enabled,
)
from env.gridworld_pkg.patch_world import (
    habacclim_observation_enabled,
    habacclim_observation_scalar_index,
    patch_world_local_observation_size,
    patch_world_total_observation_size,
)
from perf.cxxrewrite4_upstream_resident import (
    prepare_policy_materialization as cxxrewrite4_prepare_policy_materialization,
)
from .runner_action_opportunity import (
    counterfactual_policy_scalars_for_agent,
    record_family_head_authority_choice_margin,
    record_counterfactual_action_response,
    record_policy_action_family_decisions,
    record_policy_margins,
    _valid_actions_for_agent,
)

# Action indices (keep consistent with env constants)
ACT_GATHER = 4
ACT_REST = 5
ACT_ATTACK = 6


def _record_output_routing_telemetry(
    alive: list[Any],
    row_indices: list[int],
    *,
    batch: Any,
    base_logits: torch.Tensor,
) -> None:
    """Attach reporting-only ESCAP-5 arbitration data without changing policy."""

    base_rows = base_logits.detach().to(torch.float32).cpu().numpy()
    for local_row, alive_index in enumerate(row_indices):
        agent = alive[int(alive_index)]
        result = batch.rows[int(alive_index)]
        residual = np.asarray(batch.residuals[int(alive_index)], dtype=np.float32)
        base = np.asarray(base_rows[int(local_row)], dtype=np.float32)
        base_l1 = float(np.sum(np.abs(base), dtype=np.float64))
        residual_l1 = float(np.sum(np.abs(residual), dtype=np.float64))
        summary = dict(result.summary)
        summary.update(
            {
                "per_action_residual_totals": {
                    str(ACTION_VOCABULARY[index]): float(residual[index])
                    for index in range(len(ACTION_VOCABULARY))
                },
                "saturated_action_semantic_ids": [
                    str(row["semantic_id"])
                    for row in list(summary.get("action_rows", []) or [])
                    if bool(row.get("aggregate_clipped", False))
                ],
                "clipped_bindings": 0,
                "rejected_bindings": 0,
                "base_logit_l1": base_l1,
                "structural_residual_l1": residual_l1,
                "residual_base_logit_l1_ratio": float(residual_l1 / max(base_l1, 1.0e-12)),
                "residual_base_ratio_denominator_floor": 1.0e-12,
                "reporting_only": True,
                "telemetry_by_instance_id": dict(result.telemetry_by_instance_id),
            }
        )
        live = dict(getattr(agent, "structural_live_module_telemetry", {}) or {})
        live["output_routing"] = summary
        agent.structural_live_module_telemetry = live


def _np_batched_linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.matmul(w, x[:, :, None]).squeeze(-1) + b).astype(np.float32, copy=False)


def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def _torch_cpu_numpy_view(value: Any) -> np.ndarray | None:
    if not isinstance(value, torch.Tensor):
        return None
    t = value.detach()
    if bool(t.is_cuda) or not bool(t.is_contiguous()):
        return None
    return t.numpy()


def _polarch6_merge_authority_summary(runner: Any, summary: dict[str, Any] | None, *, policy_kind: str) -> None:
    if not summary:
        return
    total = getattr(runner, "_policy_family_head_residual_authority_total", None)
    if not isinstance(total, dict):
        total = {
            "schema_version": "family_head_residual_authority_summary_v1",
            "evaluated_decisions": 0,
            "random_draws_added": 0,
            "policy_kind_counts": {},
            "authority_mode_counts": {},
            "family_order": list(summary.get("family_order", [])),
            "family_probability_before_sum": [0.0 for _ in list(summary.get("family_order", []))],
            "family_probability_after_sum": [0.0 for _ in list(summary.get("family_order", []))],
            "family_probability_delta_sum": [0.0 for _ in list(summary.get("family_order", []))],
            "residual_sum": [0.0 for _ in list(summary.get("family_order", []))],
            "stored_residual_sum": [0.0 for _ in list(summary.get("family_order", []))],
            "raw_gated_residual_sum": [0.0 for _ in list(summary.get("family_order", []))],
            "centered_expressed_residual_sum": [0.0 for _ in list(summary.get("family_order", []))],
            "opportunity_sum": [0.0 for _ in list(summary.get("family_order", []))],
            "valid_family_count_sum": [0.0 for _ in list(summary.get("family_order", []))],
            "max_probability_error": 0.0,
            "max_raw_centered_probability_error": 0.0,
        }
    total["evaluated_decisions"] = int(total.get("evaluated_decisions", 0) or 0) + int(summary.get("evaluated_decisions", 0) or 0)
    total["random_draws_added"] = int(total.get("random_draws_added", 0) or 0) + int(summary.get("random_draws_added", 0) or 0)
    counts = dict(total.get("policy_kind_counts", {}) or {})
    counts[str(policy_kind)] = int(counts.get(str(policy_kind), 0) or 0) + int(summary.get("evaluated_decisions", 0) or 0)
    total["policy_kind_counts"] = counts
    mode_counts = dict(total.get("authority_mode_counts", {}) or {})
    mode = str(summary.get("authority_mode", "unconditioned_family_head") or "unconditioned_family_head")
    mode_counts[mode] = int(mode_counts.get(mode, 0) or 0) + int(summary.get("evaluated_decisions", 0) or 0)
    total["authority_mode_counts"] = mode_counts
    for key in (
        "family_probability_before_sum",
        "family_probability_after_sum",
        "family_probability_delta_sum",
        "residual_sum",
        "stored_residual_sum",
        "raw_gated_residual_sum",
        "centered_expressed_residual_sum",
        "opportunity_sum",
        "valid_family_count_sum",
    ):
        current = list(total.get(key, []) or [])
        incoming = list(summary.get(key, []) or [])
        width = max(len(current), len(incoming))
        current += [0.0] * (width - len(current))
        incoming += [0.0] * (width - len(incoming))
        total[key] = [float(current[idx]) + float(incoming[idx]) for idx in range(width)]
    decisions = max(1, int(total.get("evaluated_decisions", 0) or 0))
    for src, dst in (
        ("family_probability_before_sum", "family_probability_before_mean"),
        ("family_probability_after_sum", "family_probability_after_mean"),
        ("family_probability_delta_sum", "family_probability_delta_mean"),
        ("residual_sum", "residual_mean"),
        ("stored_residual_sum", "stored_residual_mean"),
        ("raw_gated_residual_sum", "raw_gated_residual_mean"),
        ("centered_expressed_residual_sum", "centered_expressed_residual_mean"),
        ("opportunity_sum", "opportunity_mean"),
        ("valid_family_count_sum", "valid_family_count_mean"),
    ):
        total[dst] = [float(value) / float(decisions) for value in list(total.get(src, []) or [])]
    total["max_probability_error"] = max(float(total.get("max_probability_error", 0.0) or 0.0), float(summary.get("probability_error", 0.0) or 0.0))
    total["max_raw_centered_probability_error"] = max(
        float(total.get("max_raw_centered_probability_error", 0.0) or 0.0),
        float(summary.get("raw_centered_probability_error", 0.0) or 0.0),
    )
    runner._policy_family_head_residual_authority_total = total


def _polarch12_binary_opportunity_inputs(
    runner: Any,
    alive: list[Any],
    row_indices: list[int],
    *,
    device: Any,
    dtype: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    cfg = getattr(runner, "sim_cfg", None)
    if not family_head_binary_opportunity_gate_enabled(cfg):
        return None, None, None
    opportunity_rows: list[list[float]] = []
    valid_rows: list[list[float]] = []
    action_rows: list[list[float]] = []
    for row in list(row_indices):
        valid_actions, availability = _valid_actions_for_agent(runner, alive, int(row))
        valid_set = {int(action) for action in valid_actions}
        if not valid_set:
            valid_set = {ACT_REST}
        opportunity_rows.append(
            [
                1.0 if int(availability.get("legal_move_count", 0) or 0) > 0 else 0.0,
                1.0 if int(availability.get("gather_resource_available", 0) or 0) else 0.0,
                1.0 if int(availability.get("rest_need_present", 0) or 0) else 0.0,
                1.0
                if (
                    int(availability.get("attack_target_available", 0) or 0)
                    or int(availability.get("attack_executable", 0) or 0)
                )
                else 0.0,
            ]
        )
        valid_rows.append(
            [
                1.0 if any(int(action) in valid_set for action in FAMILY_TO_ACTION_IDS[family]) else 0.0
                for family in FAMILY_HEAD_AUTHORITY_ORDER
            ]
        )
        action_rows.append([1.0 if action in valid_set else 0.0 for action in range(7)])
    if not opportunity_rows:
        return (
            torch.empty((0, len(FAMILY_HEAD_AUTHORITY_ORDER)), device=device, dtype=dtype),
            torch.empty((0, len(FAMILY_HEAD_AUTHORITY_ORDER)), device=device, dtype=dtype),
            torch.empty((0, 7), device=device, dtype=dtype),
        )
    return (
        torch.as_tensor(opportunity_rows, device=device, dtype=dtype),
        torch.as_tensor(valid_rows, device=device, dtype=dtype),
        torch.as_tensor(action_rows, device=device, dtype=dtype),
    )


def _policy_boundary_hash_jsonable(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _policy_boundary_tensor_summary(value: Optional["torch.Tensor"]) -> Dict[str, Any]:
    if value is None:
        return {"present": False}
    return {
        "present": True,
        "shape": [int(x) for x in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def policy_forward_boundary_rng_snapshot(runner: Any, *, device: str = "") -> Dict[str, Any]:
    """Return a compact no-draw RNG snapshot for BMC policy-boundary probes.

    This helper is intentionally public at module scope so BMC harnesses can
    inspect the policy boundary without reaching into private RNG internals.
    It is not used by normal campaign execution.
    """

    out: Dict[str, Any] = {"device": str(device or ""), "rng_manager_present": False}
    rm = getattr(runner, "rng_manager", None)
    if rm is None:
        return out
    out["rng_manager_present"] = True
    try:
        snap = rm.capture_snapshot(generation_index=getattr(runner, "gen", None))
        policy = (snap.get("streams", {}) or {}).get("policy", {}) or {}
        out["policy_indices"] = policy.get("indices", {})
        out["policy_state_sha256"] = _policy_boundary_hash_jsonable(policy.get("states", {}))
    except Exception as exc:
        out["rng_manager_error"] = repr(exc)
    try:
        debug = rm.debug_snapshot()
        if isinstance(debug, dict):
            out["policy_np_debug"] = debug
    except Exception:
        pass
    return out


def prepare_policy_forward_boundary(
    *,
    kind: str,
    tensors: Dict[str, Optional["torch.Tensor"]],
    runner: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a test-only explicit policy-forward boundary record.

    The record keeps tensor references so the caller can execute the same
    underlying forward operation through `execute_policy_forward_boundary`.
    Public summaries should use the `tensor_summaries` field rather than
    serializing the tensors themselves.
    """

    clean_kind = str(kind)
    if clean_kind not in {"mlp", "gru"}:
        raise ValueError(f"unsupported policy boundary kind: {clean_kind}")
    first = next((value for value in tensors.values() if value is not None), None)
    device = str(getattr(first, "device", "")) if first is not None else ""
    return {
        "schema_version": "eiel_policy_forward_boundary_v1",
        "kind": clean_kind,
        "metadata": dict(metadata or {}),
        "tensor_summaries": {name: _policy_boundary_tensor_summary(value) for name, value in tensors.items()},
        "pre_forward_rng": policy_forward_boundary_rng_snapshot(runner, device=device) if runner is not None else {},
        "tensors": tensors,
    }


def execute_policy_forward_boundary(
    boundary: Dict[str, Any],
    *,
    mlp_forward: Any = None,
    gru_forward: Any = None,
) -> Dict[str, Any]:
    """Execute a prepared policy-forward boundary without sampling actions.

    This is a BMC/test-only helper. It does not mutate runner state, write GRU
    memory, sample actions, transfer actions, or apply env actions.
    """

    kind = str(boundary.get("kind", ""))
    tensors = boundary.get("tensors", {}) or {}
    started = time.perf_counter()
    if kind == "mlp":
        fn = mlp_forward or forward_batched_mlp
        logits = fn(
            tensors["x"],
            tensors["w1"],
            tensors["b1"],
            tensors["w2"],
            tensors["b2"],
            tensors["w3"],
            tensors["b3"],
            attack_bias=tensors.get("attack_bias"),
            action_temperature=tensors.get("action_temperature"),
        )
        h_next = None
    elif kind == "gru":
        fn = gru_forward or forward_batched_gru
        logits, h_next = fn(
            x=tensors["x"],
            h_prev=tensors["h_prev"],
            encW=tensors["encW"],
            encb=tensors["encb"],
            Wih=tensors["Wih"],
            bih=tensors["bih"],
            Whh=tensors["Whh"],
            bhh=tensors["bhh"],
            headW=tensors["headW"],
            headb=tensors["headb"],
            attack_bias=tensors.get("attack_bias"),
            action_temperature=tensors.get("action_temperature"),
        )
    else:
        raise ValueError(f"unsupported policy boundary kind: {kind}")
    return {
        "schema_version": "eiel_policy_forward_boundary_result_v1",
        "kind": kind,
        "elapsed_seconds": float(time.perf_counter() - started),
        "logits": logits,
        "h_next": h_next,
        "output_summaries": {
            "logits": _policy_boundary_tensor_summary(logits),
            "h_next": _policy_boundary_tensor_summary(h_next),
        },
    }


def sample_policy_forward_boundary_actions(runner: Any, logits: "torch.Tensor") -> Dict[str, Any]:
    """Sample actions for a prepared boundary using the runner's policy stream.

    The helper mirrors `_batched_actions` sampling and records pre/post RNG
    snapshots. It is not called by the normal path.
    """

    before = policy_forward_boundary_rng_snapshot(runner, device=str(logits.device))
    if getattr(getattr(runner, "sim_cfg", None), "policy_deterministic_actions", False):
        actions = torch.argmax(logits, dim=1)
        backend = "deterministic_argmax"
    else:
        rm = getattr(runner, "rng_manager", None)
        if rm is None:
            raise RuntimeError("Policy boundary sampling requires runner.rng_manager")
        tr = rm.torch_rng("policy", str(logits.device))
        if tr is None:
            raise RuntimeError(f"Policy boundary sampling missing torch policy RNG for device={str(logits.device)}")
        u = tr.rand(tuple(int(x) for x in logits.shape), device=logits.device, dtype=logits.dtype)
        g = -torch.log(-torch.log(u.clamp(min=1e-6, max=1.0 - 1e-6)))
        actions = torch.argmax(logits + g, dim=1)
        backend = "rng_manager_torch_policy_gumbel_max"
    after = policy_forward_boundary_rng_snapshot(runner, device=str(logits.device))
    return {
        "schema_version": "eiel_policy_forward_boundary_sampling_v1",
        "backend": backend,
        "pre_sampling_rng": before,
        "post_sampling_rng": after,
        "actions": actions,
        "action_summary": _policy_boundary_tensor_summary(actions),
    }


def commit_policy_forward_boundary_metadata(
    *,
    kind: str,
    slots: Optional["torch.Tensor"] = None,
    actions: Optional["torch.Tensor"] = None,
    h_next: Optional["torch.Tensor"] = None,
) -> Dict[str, Any]:
    """Return split/writeback metadata for BMC boundary probes.

    This helper intentionally does not mutate runner state. Existing normal
    path code remains responsible for GRU Hmem writeback and action transfer.
    """

    return {
        "schema_version": "eiel_policy_forward_boundary_commit_metadata_v1",
        "kind": str(kind),
        "slots": _policy_boundary_tensor_summary(slots),
        "actions": _policy_boundary_tensor_summary(actions),
        "h_next": _policy_boundary_tensor_summary(h_next),
    }


class PolicyMixin:
    def _reset_memory_runtime_telemetry(self) -> None:
        self._memory_runtime_telemetry_gen = {
            "events": 0.0,
            "state_abs_sum": 0.0,
            "state_abs_p90_sum": 0.0,
            "delta_abs_sum": 0.0,
            "delta_abs_p90_sum": 0.0,
            "active_slice_abs_sum": 0.0,
            "inactive_slice_abs_sum": 0.0,
            "mask_leakage_sum": 0.0,
            "nonzero_fraction_sum": 0.0,
            "effective_utilization_sum": 0.0,
            "saturation_proxy_sum": 0.0,
            "reset_count_total": 0.0,
        }

    def _memory_runtime_telemetry_summary(self) -> dict[str, float]:
        acc = getattr(self, "_memory_runtime_telemetry_gen", None)
        if not isinstance(acc, dict):
            return {}
        events = float(acc.get("events", 0.0) or 0.0)
        if events <= 0.0:
            return {
                "memory_state_abs_mean": 0.0,
                "memory_state_abs_p90": 0.0,
                "memory_state_delta_mean": 0.0,
                "memory_state_delta_p90": 0.0,
                "memory_active_slice_abs_mean": 0.0,
                "memory_inactive_slice_abs_mean": 0.0,
                "memory_mask_leakage_mean": 0.0,
                "memory_nonzero_fraction_mean": 0.0,
                "memory_effective_utilization_mean": 0.0,
                "memory_saturation_proxy_mean": 0.0,
                "memory_reset_count_total": float(acc.get("reset_count_total", 0.0) or 0.0),
                "memory_use_measurement_available_flag": 0.0,
            }
        return {
            "memory_state_abs_mean": float(acc.get("state_abs_sum", 0.0) or 0.0) / events,
            "memory_state_abs_p90": float(acc.get("state_abs_p90_sum", 0.0) or 0.0) / events,
            "memory_state_delta_mean": float(acc.get("delta_abs_sum", 0.0) or 0.0) / events,
            "memory_state_delta_p90": float(acc.get("delta_abs_p90_sum", 0.0) or 0.0) / events,
            "memory_active_slice_abs_mean": float(acc.get("active_slice_abs_sum", 0.0) or 0.0) / events,
            "memory_inactive_slice_abs_mean": float(acc.get("inactive_slice_abs_sum", 0.0) or 0.0) / events,
            "memory_mask_leakage_mean": float(acc.get("mask_leakage_sum", 0.0) or 0.0) / events,
            "memory_nonzero_fraction_mean": float(acc.get("nonzero_fraction_sum", 0.0) or 0.0) / events,
            "memory_effective_utilization_mean": float(acc.get("effective_utilization_sum", 0.0) or 0.0) / events,
            "memory_saturation_proxy_mean": float(acc.get("saturation_proxy_sum", 0.0) or 0.0) / events,
            "memory_reset_count_total": float(acc.get("reset_count_total", 0.0) or 0.0),
            "memory_use_measurement_available_flag": 1.0,
        }

    def _record_memory_runtime_telemetry(
        self,
        *,
        h_prev: torch.Tensor,
        h_next: torch.Tensor,
        mask: torch.Tensor,
        reset_count: int,
    ) -> None:
        if not bool(getattr(getattr(self, "sim_cfg", None), "enable_cognitive_status_metrics", True)):
            return
        try:
            acc = getattr(self, "_memory_runtime_telemetry_gen", None)
            if not isinstance(acc, dict):
                self._reset_memory_runtime_telemetry()
                acc = getattr(self, "_memory_runtime_telemetry_gen", {})
            with torch.no_grad():
                h_prev_d = h_prev.detach()
                h_next_d = h_next.detach()
                mask_d = mask.detach()
                delta = h_next_d - h_prev_d
                active = mask_d > 0.0
                inactive = ~active
                abs_next = torch.abs(h_next_d)
                abs_delta = torch.abs(delta)

                def _mean(t: torch.Tensor) -> float:
                    return float(t.mean().item()) if int(t.numel()) > 0 else 0.0

                def _p90(t: torch.Tensor) -> float:
                    if int(t.numel()) <= 0:
                        return 0.0
                    flat = t.reshape(-1).to(device="cpu")
                    k = int(max(0, min(int(flat.numel()) - 1, int(np.ceil(0.90 * float(flat.numel())) - 1))))
                    vals, _ = torch.sort(flat)
                    return float(vals[k].item())

                active_vals = abs_next[active] if bool(active.any().item()) else abs_next.new_empty((0,))
                inactive_vals = abs_next[inactive] if bool(inactive.any().item()) else abs_next.new_empty((0,))
                active_nonzero = float((active_vals > 1e-7).sum().item()) if int(active_vals.numel()) > 0 else 0.0
                active_total = float(active_vals.numel()) if int(active_vals.numel()) > 0 else 0.0
                acc["events"] = float(acc.get("events", 0.0) or 0.0) + 1.0
                acc["state_abs_sum"] = float(acc.get("state_abs_sum", 0.0) or 0.0) + _mean(abs_next)
                acc["state_abs_p90_sum"] = float(acc.get("state_abs_p90_sum", 0.0) or 0.0) + _p90(abs_next)
                acc["delta_abs_sum"] = float(acc.get("delta_abs_sum", 0.0) or 0.0) + _mean(abs_delta)
                acc["delta_abs_p90_sum"] = float(acc.get("delta_abs_p90_sum", 0.0) or 0.0) + _p90(abs_delta)
                acc["active_slice_abs_sum"] = float(acc.get("active_slice_abs_sum", 0.0) or 0.0) + _mean(active_vals)
                acc["inactive_slice_abs_sum"] = float(acc.get("inactive_slice_abs_sum", 0.0) or 0.0) + _mean(inactive_vals)
                acc["mask_leakage_sum"] = float(acc.get("mask_leakage_sum", 0.0) or 0.0) + _mean(inactive_vals)
                acc["nonzero_fraction_sum"] = float(acc.get("nonzero_fraction_sum", 0.0) or 0.0) + (
                    float((abs_next > 1e-7).sum().item()) / float(max(1, int(abs_next.numel())))
                )
                acc["effective_utilization_sum"] = float(acc.get("effective_utilization_sum", 0.0) or 0.0) + (
                    active_nonzero / active_total if active_total > 0.0 else 0.0
                )
                acc["saturation_proxy_sum"] = float(acc.get("saturation_proxy_sum", 0.0) or 0.0) + (
                    float((abs_next > 0.95).sum().item()) / float(max(1, int(abs_next.numel())))
                )
                acc["reset_count_total"] = float(acc.get("reset_count_total", 0.0) or 0.0) + float(reset_count)
        except Exception:
            return

    def _apply_a3_observation_coupling(self, alive, local: np.ndarray, energy_norm: np.ndarray, health_norm: np.ndarray) -> None:
        """Apply deterministic bounded observation degradation from chemistry."""
        if not bool(getattr(self.sim_cfg, "chemistry_functional_coupling_enable", False)):
            return
        fidelity_min = float(getattr(self.sim_cfg, "chemistry_observation_fidelity_min", 0.70))
        fidelity_max = float(getattr(self.sim_cfg, "chemistry_observation_fidelity_max", 1.00))
        noise_abs_max = float(getattr(self.sim_cfg, "chemistry_observation_noise_abs_max", 0.15))
        fidelity_strength = float(getattr(self.sim_cfg, "chemistry_observation_fidelity_strength", 0.9))
        noise_strength = float(getattr(self.sim_cfg, "chemistry_observation_noise_strength", 0.8))

        for i, a in enumerate(alive):
            fidelity, bias = observation_fidelity_and_bias_from_chem(
                getattr(a, "chemistry", None),
                fidelity_min=fidelity_min,
                fidelity_max=fidelity_max,
                noise_abs_max=noise_abs_max,
                fidelity_strength=fidelity_strength,
                noise_strength=noise_strength,
            )
            row = local[i, :]
            local[i, :] = np.clip(row * np.float32(fidelity) + np.float32(bias), -1.0, 1.0)
            energy_norm[i] = np.float32(min(1.0, max(0.0, float(energy_norm[i]) * fidelity + bias)))
            health_norm[i] = np.float32(min(1.0, max(0.0, float(health_norm[i]) * fidelity + bias)))
            try:
                st = getattr(a, "stats", None)
                if st is not None:
                    st.chemistry_observation_fidelity = float(fidelity)
                    st.chemistry_observation_bias = float(bias)
            except Exception:
                pass

    def _is_gru_policy(self, policy_obj) -> bool:
        """Return True if policy_obj is a GRU policy (robust to import-path / wrappers).

        Phase-3 determinism: avoid fragile isinstance(GRUPolicy) when the class can be
        loaded from different module roots or wrapped/proxied.
        """
        try:
            if policy_obj is None:
                return False
            # Feature detection first (preferred)
            if hasattr(policy_obj, "gru"):
                return True
            # Fallback: direct type check
            return isinstance(policy_obj, GRUPolicy)
        except Exception:
            return False

    def _policy_slots_log(self, msg: str) -> None:
        """Emit policy-slots debug messages.

        Controlled via env vars:
        - EIEL_DEBUG_POLICY_SLOTS=1 enables generation of messages (existing behavior).
        - EIEL_DEBUG_POLICY_SLOTS_QUIET=1 suppresses stderr spam and writes to
          <run_dir>/debug/policy_slots.log instead.
        """
        try:
            quiet = os.environ.get("EIEL_DEBUG_POLICY_SLOTS_QUIET", "0").strip() == "1"
            if not quiet:
                print(msg, file=sys.stderr)
                return
            run_dir = getattr(self, "run_dir", None) or getattr(self, "run_dir_path", None) or ""
            if not run_dir:
                # No run dir, fall back to stderr.
                print(msg, file=sys.stderr)
                return
            dbg = os.path.join(str(run_dir), "debug")
            os.makedirs(dbg, exist_ok=True)
            out_path = os.path.join(dbg, "policy_slots.log")
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            # Never break the sim because of debug logging.
            try:
                print(msg, file=sys.stderr)
            except Exception:
                pass



    def _eiel_foreensic_v4_enabled(self) -> bool:
        """Return True if forensic device probes are enabled (EIEL_FORENSIC_V4=1)."""
        try:
            return os.environ.get("EIEL_FORENSIC_V4", "0").strip() == "1"
        except Exception:
            return False

    def _eiel_foreensic_log(self, msg: str) -> None:
        """Write a forensic log line (never throws)."""
        try:
            if not self._eiel_foreensic_v4_enabled():
                return
            run_dir = getattr(self, "run_dir", None) or getattr(self, "run_dir_path", None) or ""
            if run_dir:
                dbg = os.path.join(str(run_dir), "debug")
                os.makedirs(dbg, exist_ok=True)
                out_path = os.path.join(dbg, "forensic_bank_device.log")
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            else:
                print(msg, file=sys.stderr)
        except Exception:
            try:
                print(msg, file=sys.stderr)
            except Exception:
                pass

    def _eiel_bank_device_sanity_probe(
        self,
        *,
        stage: str,
        bank_name: str,
        bank: Optional[torch.Tensor],
        slots: Optional[torch.Tensor],
    ) -> None:
        """Log and (optionally) fail-fast on bank/slot device mismatch.

        Enabled only when EIEL_FORENSIC_V4=1.

        This is intentionally minimal and must not introduce nondeterminism.
        """
        if not self._eiel_foreensic_v4_enabled():
            return
        bdev = str(getattr(bank, "device", "None"))
        bdt = str(getattr(bank, "dtype", "None"))
        bsh = list(getattr(bank, "shape", [])) if bank is not None else []
        idev = str(getattr(slots, "device", "None"))
        idt = str(getattr(slots, "dtype", "None"))
        ish = list(getattr(slots, "shape", [])) if slots is not None else []
        cdev = str(getattr(self, "device", "None"))
        self._eiel_foreensic_log(
            f"[EIEL_FORENSIC_V4] {stage} bank={bank_name} bank_dev={bdev} bank_dtype={bdt} bank_shape={bsh} "
            f"slots_dev={idev} slots_dtype={idt} slots_shape={ish} compute_dev={cdev}"
        )
        # Optional strictness: raise immediately if mismatch survives.
        if bank is not None and slots is not None:
            if slots.device != bank.device:
                raise RuntimeError(
                    f"FDP device mismatch at {stage}: bank '{bank_name}' on {str(bank.device)} but slots on {str(slots.device)}"
                )
            if slots.dtype != torch.long:
                raise RuntimeError(
                    f"FDP index dtype mismatch at {stage}: bank '{bank_name}' expects torch.long slots, got {str(slots.dtype)}"
                )

    def _eiel_to_compute(self, t: torch.Tensor, *, name: str, stage: str) -> torch.Tensor:
        """Move tensor to policy compute device (self.device) if needed."""
        dev = getattr(self, "device", None)
        if dev is None or t is None:
            return t
        if t.device == dev:
            return t
        if self._eiel_foreensic_v4_enabled():
            try:
                self._eiel_foreensic_log(
                    f"[EIEL_FORENSIC_V4] {stage} move_to_compute {name} {str(t.device)}->{str(dev)} shape={list(t.shape)} dtype={str(t.dtype)}"
                )
            except Exception:
                pass
        return t.to(dev, non_blocking=False)

    def _eiel_to_bank(self, t: torch.Tensor, *, like: torch.Tensor, name: str, stage: str) -> torch.Tensor:
        """Move tensor to a bank device (like.device) if needed."""
        if t is None or like is None:
            return t
        if t.device == like.device:
            return t
        if self._eiel_foreensic_v4_enabled():
            try:
                self._eiel_foreensic_log(
                    f"[EIEL_FORENSIC_V4] {stage} move_to_bank {name} {str(t.device)}->{str(like.device)} shape={list(t.shape)} dtype={str(t.dtype)}"
                )
            except Exception:
                pass
        return t.to(like.device, non_blocking=False)

    def _eiel_policy_exec_probe(
        self,
        stage: str,
        alive: List["Agent"],
        mlp_idx: List[int],
        gru_idx: List[int],
        k: int,
        cache_hit: bool,
        extra: Optional[Dict[str, Any]] = None,
        *,
        mlp_slots: Optional[List[int]] = None,
        mlp_tokens: Optional[List[int]] = None,
        gru_slots: Optional[List[int]] = None,
        gru_tokens: Optional[List[int]] = None,
        actions_out: Optional["np.ndarray"] = None,
        logits: Optional["torch.Tensor"] = None,
    ) -> None:
        """Emit a compact JSONL record to debug/policy_exec_probe.jsonl for forensic debugging.

        Why this exists:
        - determinism_probe.jsonl is great at telling you *when* the two runs diverge.
        - policy_exec_probe.jsonl is meant to quickly tell you whether divergence is caused by
          *policy-side* nondeterminism (e.g., logits/actions differ), without digging through
          enormous logs.

        Enablement / scope:
        - Set EIEL_POLICY_EXEC_PROBE=1 to enable.
        - For backwards compatibility, EIEL_DETERMINISM_PROBE=1 also enables it.
        - Default scope when enabled via EIEL_DETERMINISM_PROBE: gen=1, ticks 0-15.
        - Override scope with:
            * EIEL_POLICY_EXEC_PROBE_GENS   e.g. "1" or "0-2" or "1,3,5"
            * EIEL_POLICY_EXEC_PROBE_TICKS  e.g. "0-20" or "0,5,10"
        - Set EIEL_POLICY_EXEC_PROBE_ALL=1 to log for all gens/ticks (not recommended).

        The record is intentionally hash-heavy and payload-light:
        - stable digests for ID lists, slot/token lists, RNG states
        - (optionally) stable digests for logits and action outputs
        - small samples for quick eyeballing

        This function must never throw; failures are swallowed and do not affect the sim.
        """
        try:
            # --- gating -----------------------------------------------------
            enable = bool(int(os.getenv("EIEL_POLICY_EXEC_PROBE", "0"))) or bool(
                int(os.getenv("EIEL_DETERMINISM_PROBE", "0"))
            )
            if not enable:
                return

            allow_all = bool(int(os.getenv("EIEL_POLICY_EXEC_PROBE_ALL", "0")))
            cur_gen = int(getattr(self, "gen", -1))
            cur_tick = int(getattr(self, "tick", -1))

            def _parse_int_ranges(spec: Optional[str]) -> Optional[set]:
                """Parse '0-3,7,9' into a set[int]. Returns None for '*' (meaning all)."""
                if spec is None:
                    return set()
                spec = str(spec).strip()
                if not spec:
                    return set()
                if spec == "*":
                    return None
                out: set = set()
                for part in spec.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if part == "*":
                        return None
                    if "-" in part:
                        a_s, b_s = part.split("-", 1)
                        a, b = int(a_s), int(b_s)
                        if b < a:
                            a, b = b, a
                        # keep this bounded to avoid someone accidentally requesting 10^9 ints.
                        b = min(b, a + 100_000)
                        out.update(range(a, b + 1))
                    else:
                        out.add(int(part))
                return out

            if not allow_all:
                gen_spec = os.getenv("EIEL_POLICY_EXEC_PROBE_GENS", "1")
                tick_spec = os.getenv("EIEL_POLICY_EXEC_PROBE_TICKS")
                if tick_spec is None:
                    # If we're in determinism debug mode, expand coverage a little.
                    tick_spec = "0-15" if bool(int(os.getenv("EIEL_DETERMINISM_PROBE", "0"))) else "0"

                gens = _parse_int_ranges(gen_spec)
                ticks = _parse_int_ranges(tick_spec)

                if gens is not None and cur_gen not in gens:
                    return
                if ticks is not None and cur_tick not in ticks:
                    return

            # --- helpers ----------------------------------------------------
            def _sha256_bytes(b: bytes) -> str:
                return hashlib.sha256(b).hexdigest()

            def _digest_ndarray(arr: "np.ndarray") -> str:
                """Stable digest for ndarray values (includes dtype + shape)."""
                a = np.asarray(arr)
                if not a.flags.get("C_CONTIGUOUS", False):
                    a = np.ascontiguousarray(a)
                h = hashlib.sha256()
                h.update(str(a.dtype).encode("utf-8"))
                h.update(b"|")
                h.update(",".join(str(x) for x in a.shape).encode("utf-8"))
                h.update(b"|")
                h.update(a.tobytes(order="C"))
                return h.hexdigest()

            def _digest_int_list(xs: Optional[List[int]]) -> Optional[str]:
                if xs is None:
                    return None
                a = np.asarray(xs, dtype=np.int64)
                return _digest_ndarray(a)

            def _digest_torch_tensor(t: "torch.Tensor") -> str:
                """Stable digest for torch tensors (device-agnostic, includes dtype + shape)."""
                tt = t.detach()
                if tt.is_cuda:
                    tt = tt.to("cpu")
                tt = tt.contiguous()
                try:
                    a = tt.numpy()
                except Exception:
                    a = tt.float().numpy()
                return _digest_ndarray(a)

            def _head(xs: Optional[List[int]], n: int = 16) -> Optional[List[int]]:
                if xs is None:
                    return None
                return list(xs[:n])

            # --- build record ------------------------------------------------
            alive_ids = np.asarray([alive[i].agent_id for i in range(len(alive))], dtype=np.int64)
            mlp_ids = np.asarray([alive[i].agent_id for i in mlp_idx], dtype=np.int64)
            gru_ids = np.asarray([alive[i].agent_id for i in gru_idx], dtype=np.int64)

            digests: Dict[str, str] = {
                "alive_ids": _digest_ndarray(alive_ids),
                "mlp_ids": _digest_ndarray(mlp_ids),
                "gru_ids": _digest_ndarray(gru_ids),
                "torch_cpu_rng": _sha256_bytes(torch.get_rng_state().detach().cpu().numpy().tobytes()),
            }
            if torch.cuda.is_available():
                try:
                    digests["torch_cuda_rng"] = _sha256_bytes(
                        torch.cuda.get_rng_state().detach().cpu().numpy().tobytes()
                    )
                except Exception:
                    # Some environments don't have a current CUDA context; swallow it.
                    pass

            # Optional slot/token lists (highly informative for cache/bank determinism).
            for name, xs in [
                ("mlp_slots", mlp_slots),
                ("mlp_tokens", mlp_tokens),
                ("gru_slots", gru_slots),
                ("gru_tokens", gru_tokens),
            ]:
                d = _digest_int_list(xs)
                if d is not None:
                    digests[name] = d

            # Optional logits/actions digests.
            if logits is not None:
                digests["logits"] = _digest_torch_tensor(logits)
            if actions_out is not None:
                digests["actions_out"] = _digest_ndarray(actions_out)

            rec = {
                "schema_version": "policy_exec_probe_v2",
                "stage": stage,
                "tick": int(cur_tick),
                "gen": int(cur_gen),
                "k": int(k),
                "cache_hit": bool(cache_hit),
                "n_alive": len(alive),
                "n_mlp": len(mlp_idx),
                "n_gru": len(gru_idx),
                "digests": digests,
                "samples": {
                    "alive_ids_head": alive_ids[:16].tolist(),
                    "mlp_ids_head": mlp_ids[:16].tolist(),
                    "gru_ids_head": gru_ids[:16].tolist(),
                    "mlp_slots_head": _head(mlp_slots),
                    "gru_slots_head": _head(gru_slots),
                    "mlp_tokens_head": _head(mlp_tokens),
                    "gru_tokens_head": _head(gru_tokens),
                    "actions_out_head": actions_out.reshape(-1).astype(int).tolist()[:16]
                    if actions_out is not None
                    else None,
                },
                "extra": extra or {},
            }

            run_dir = (
                getattr(self, "run_dir", None)
                or getattr(self, "run_dir_path", None)
                or getattr(self, "out_dir", None)
                or ""
            )
            if not run_dir:
                return
            out_dir = Path(str(run_dir))
            p = out_dir / "debug" / "policy_exec_probe.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

        except Exception:
            return
    def _batched_actions(self, alive, buckets: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Compute one action per alive agent using batched inference.

        Performance goals
        -----------------
        This function is called every tick and is therefore extremely hot.

        The biggest throughput killers at small batch sizes are:
        - repeated allocations (NumPy + Torch) every tick
        - repeated CPU->GPU transfers every tick
        - repeated GPU->CPU transfers / implicit CUDA synchronization

        This implementation targets those directly:
        - Reuse preallocated CPU/GPU staging buffers (grow-only)
        - One host->device transfer for the whole alive batch
        - One device->host transfer for the final actions
        - Avoid softmax+multinomial sampling by using Gumbel-Max sampling

        Semantics are intentionally unchanged:
        - Same observation values
        - Same per-agent genes
        - Same drive neuromodulation and sampling distribution
        - Same GRU masking / memory update
        """

        n = len(alive)
        if n == 0:
            return np.empty((0,), dtype=np.int32)
        if bool(
            getattr(
                getattr(self, "sim_cfg", None),
                "recomb72_neutral_actions_enable",
                False,
            )
        ):
            action_id = int(getattr(self.sim_cfg, "recomb72_neutral_action_id", 5))
            if action_id != 5:
                raise RuntimeError(
                    "RECOMB72_AUDIT_CONTRACT: neutral action must be ACT_REST"
                )
            self._recomb72_neutral_action_agent_ticks = int(
                getattr(self, "_recomb72_neutral_action_agent_ticks", 0)
            ) + n
            return np.full((n,), action_id, dtype=np.int32)

        # Phase 0 perf: capture coarse policy timing buckets (values are seconds).
        t_policy0 = time.perf_counter()
        t_obs0 = time.perf_counter()

        local_size = patch_world_local_observation_size(self.env_cfg)
        obs_size = patch_world_total_observation_size(self.env_cfg)

        # ---- Observations (env-side) ----
        agent_ids_np = np.asarray([a.agent_id for a in alive], dtype=np.int32)
        local = self.env.get_local_observations(agent_ids_np)

        # ---- Build per-agent scalars in one pass (CPU) ----
        # Base obs: local patch + [energy_norm, health_norm]
        obs_np = np.empty((n, obs_size), dtype=np.float32)
        obs_np[:, :local_size] = local

        e_norm = float(self.env_cfg.max_energy)
        h_norm = float(self.env_cfg.start_health)

        energy_norm = np.empty((n,), dtype=np.float32)
        health_norm = np.empty((n,), dtype=np.float32)
        aggr_np = np.empty((n,), dtype=np.float32)
        temp_np = np.empty((n,), dtype=np.float32)

        # Drives: [hunger, pain, fatigue, fear]
        drives_np = np.zeros((n, 4), dtype=np.float32)
        gain_np = np.ones((n,), dtype=np.float32)
        counterfactual_response_enabled = bool(
            getattr(
                self.sim_cfg,
                "structural_counterfactual_action_response_reporting_enable",
                False,
            )
        )
        cf_aggr_np = np.empty((n,), dtype=np.float32) if counterfactual_response_enabled else None
        cf_temp_np = np.empty((n,), dtype=np.float32) if counterfactual_response_enabled else None
        cf_gain_np = np.empty((n,), dtype=np.float32) if counterfactual_response_enabled else None
        cf_status = ["reporting_disabled"] * n if counterfactual_response_enabled else None

        # Partition MLP vs GRU in the same pass
        mlp_idx: List[int] = []
        gru_idx: List[int] = []

        # Slots/tokens for safe caching of GPU index tensors
        mlp_slots_list: List[int] = []
        mlp_tokens_list: List[int] = []
        gru_slots_list: List[int] = []
        gru_tokens_list: List[int] = []

        cxxrewrite4_policy_heads = cxxrewrite4_prepare_policy_materialization(
            self,
            alive=alive,
            agent_ids_np=agent_ids_np,
            e_norm=e_norm,
            h_norm=h_norm,
            stage="actions1",
        )
        cxxrewrite4_use_policy_heads = bool(
            cxxrewrite4_policy_heads is not None and cxxrewrite4_policy_heads.get("use_resident_policy_heads")
        )
        cxxrewrite4_use_drives = bool(
            cxxrewrite4_policy_heads is not None and cxxrewrite4_policy_heads.get("drives_np") is not None
        )
        cxxrewrite4_slots = cxxrewrite4_policy_heads.get("slots") if cxxrewrite4_policy_heads is not None else None
        cxxrewrite4_tokens = cxxrewrite4_policy_heads.get("tokens") if cxxrewrite4_policy_heads is not None else None
        if cxxrewrite4_use_policy_heads:
            energy_norm[:] = np.asarray(cxxrewrite4_policy_heads["energy_norm"], dtype=np.float32)
            health_norm[:] = np.asarray(cxxrewrite4_policy_heads["health_norm"], dtype=np.float32)
        if cxxrewrite4_use_drives:
            drives_np[:, :] = np.asarray(cxxrewrite4_policy_heads["drives_np"], dtype=np.float32)

        for i, a in enumerate(alive):
            if not cxxrewrite4_use_policy_heads:
                energy_norm[i] = float(a.energy) / e_norm
                health_norm[i] = float(a.health) / h_norm

            g = getattr(a, "genome", None)
            if g is not None:
                _scalars = (
                    self.resolve_developmental_effective_scalars_for_agent(a)
                    if hasattr(self, "resolve_developmental_effective_scalars_for_agent")
                    else None
                )
                if _scalars is not None:
                    aggr_np[i] = float(_scalars[0])
                    temp_np[i] = float(_scalars[1])
                else:
                    _dev = self.resolve_developmental_expression_for_agent(a)
                    _eff = _dev.get("effective", {}) if isinstance(_dev, dict) else {}
                    aggr_np[i] = float(_eff.get("aggression_bias", getattr(g, "aggression_bias", 0.0)))
                    temp_np[i] = float(_eff.get("action_temperature", getattr(g, "action_temperature", 1.0)))
                # ---------------------------------------------------------
                # Chaos Week 2: Identity Gate (minimal v1)
                # ---------------------------------------------------------
                # Goal: If an agent is "coherent" (low identity drift), reduce
                # exploratory randomness by lowering its action temperature.
                #
                # Implementation notes:
                # - Coherence is derived from AgentStats.identity_drift_ema.
                # - We normalize drift by sim_cfg.identity_drift_cap and map to
                #   coherence in [0,1] as: coherence = 1 - clamp(drift/cap).
                # - When enabled, we scale temperature by (1 - strength*coherence).
                # - identity_gate_adult_only gates this to post-first-reproduction.
                #
                # This is intentionally simple and deterministic: the smoke tests
                # need a directional signal, not a dissertation.
                try:
                    scfg = self.sim_cfg
                    if getattr(scfg, "identity_gate_enable", False):
                        mode = str(getattr(scfg, "identity_gate_mode", "none") or "none").strip().lower()
                        if mode in ("action_noise", "temperature", "temp"):
                            adult_only = bool(getattr(scfg, "identity_gate_adult_only", True))
                            st = getattr(a, "stats", None)
                            is_adult = False
                            if st is not None:
                                fr = getattr(st, "first_repro_step", None)
                                if fr is not None:
                                    try:
                                        is_adult = int(getattr(st, "survival_steps", 0)) > int(fr)
                                    except Exception:
                                        is_adult = True
                            if (not adult_only) or is_adult:
                                drift = float(getattr(st, "identity_drift_ema", 0.0)) if st is not None else 0.0
                                cap = float(getattr(scfg, "identity_drift_cap", 3.0) or 3.0)
                                if cap <= 0.0:
                                    cap = 3.0
                                norm = drift / cap
                                if norm < 0.0:
                                    norm = 0.0
                                elif norm > 1.0:
                                    norm = 1.0
                                coherence = 1.0 - norm
                                strength = float(getattr(scfg, "identity_gate_strength", 0.0) or 0.0)
                                if strength > 0.0 and coherence > 0.0:
                                    mult = 1.0 - strength * coherence
                                    if mult < 0.05:
                                        mult = 0.05
                                    temp_np[i] = float(temp_np[i]) * float(mult)
                except Exception:
                    pass

                try:
                    if _scalars is not None:
                        gain_np[i] = float(_scalars[2])
                    else:
                        gain_np[i] = float(_eff.get("drive_logit_gain", getattr(g, "drive_logit_gain", 1.0)))
                except Exception:
                    gain_np[i] = 1.0
            else:
                aggr_np[i] = 0.0
                temp_np[i] = 1.0
                gain_np[i] = 1.0
            if counterfactual_response_enabled:
                cf_aggr, cf_temp, cf_gain, cf_reason = counterfactual_policy_scalars_for_agent(
                    self,
                    a,
                    actual_aggression_bias=float(aggr_np[i]),
                    actual_action_temperature=float(temp_np[i]),
                    actual_drive_logit_gain=float(gain_np[i]),
                )
                cf_aggr_np[i] = np.float32(cf_aggr)  # type: ignore[index]
                cf_temp_np[i] = np.float32(cf_temp)  # type: ignore[index]
                cf_gain_np[i] = np.float32(cf_gain)  # type: ignore[index]
                cf_status[i] = str(cf_reason)  # type: ignore[index]

            if not cxxrewrite4_use_drives:
                # Runner-level drive arrays (fast path)
                try:
                    if hasattr(self, "_sync_drive_slot"):
                        self._sync_drive_slot(a)
                    slot = int(getattr(a, "slot", -1))
                    if slot >= 0 and hasattr(self, "_drive_hunger") and self._drive_hunger is not None:
                        drives_np[i, 0] = float(self._drive_hunger[slot])
                        drives_np[i, 1] = float(self._drive_pain[slot])
                        drives_np[i, 2] = float(self._drive_fatigue[slot])
                        drives_np[i, 3] = float(self._drive_fear[slot])
                except Exception:
                    pass

            if cxxrewrite4_slots is not None:
                slot_i = int(cxxrewrite4_slots[i])
            else:
                slot_i = int(getattr(a, "slot", -1))
            if cxxrewrite4_tokens is not None:
                token_i = int(cxxrewrite4_tokens[i])
            else:
                token_i = int(getattr(a, "slot_token", 0))

            # Phase 3 determinism:
            # Do NOT partition MLP vs GRU based only on `agent.policy`.
            # Policy objects are not part of the canonical checkpoint state (they are
            # often None right after resume), while `genome.use_gru` *is* canonical.
            # If we mis-partition even one agent, the torch sampling path changes
            # (different rand tensor shapes) and torch RNG diverges at tick0.
            is_gru = False
            try:
                g = getattr(a, "genome", None)
                if g is not None and hasattr(g, "use_gru"):
                    is_gru = bool(getattr(g, "use_gru", False))
                else:
                    is_gru = bool(self._is_gru_policy(getattr(a, "policy", None)))
            except Exception:
                is_gru = bool(self._is_gru_policy(getattr(a, "policy", None)))

            if is_gru:
                gru_idx.append(i)
                gru_slots_list.append(slot_i)
                gru_tokens_list.append(token_i)
            else:
                mlp_idx.append(i)
                mlp_slots_list.append(slot_i)
                mlp_tokens_list.append(token_i)

        self._apply_a3_observation_coupling(alive, local, energy_norm, health_norm)
        if (
            bool(getattr(self.sim_cfg, "enable_evolvable_sensor_channel_mask", False))
            or bool(getattr(self.sim_cfg, "ablate_sensor_channel_mask", False))
            or bool(getattr(self.sim_cfg, "force_all_sensor_channels_enabled", False))
            or bool(getattr(self.sim_cfg, "force_minimal_sensor_channels_enabled", False))
            or bool(getattr(self.sim_cfg, "enable_evolvable_sensor_radius", False))
            or bool(getattr(self.sim_cfg, "ablate_sensor_radius", False))
            or bool(getattr(self.sim_cfg, "force_min_sensor_radius", False))
            or bool(getattr(self.sim_cfg, "force_max_sensor_radius", False))
            or bool(getattr(self.sim_cfg, "force_default_sensor_radius", False))
        ):
            apply_perception_masks(
                local=local,
                energy_norm=energy_norm,
                health_norm=health_norm,
                agents=alive,
                env_cfg=self.env_cfg,
                sim_cfg=self.sim_cfg,
            )
        obs_np[:, :local_size] = local
        obs_np[:, local_size] = energy_norm
        obs_np[:, local_size + 1] = health_norm
        if habacclim_observation_enabled(self.env_cfg):
            obs_np[:, habacclim_observation_scalar_index(self.env_cfg)] = np.asarray(
                [float(getattr(a, "habitat_acclimation_state", 0.0) or 0.0) for a in alive],
                dtype=np.float32,
            )

        # ESCAP-4 resolves only existing named scalar meanings from the final,
        # already-masked policy frame.  Direct observation bytes/shape remain
        # untouched and disabled profiles do not enter the routing helper.
        escap4_routing_batches = None
        if observation_routing_enabled(self.sim_cfg):
            escap4_routing_batches = route_observations_batch(
                alive,
                obs_np,
                local_size=local_size,
                sim_cfg=self.sim_cfg,
            )

        # ESCAP-3 executes after existing drives have been mirrored and before
        # policy sampling. It is CPU/reference-only and consumes zero RNG.
        escap3_rest_residual_np = execute_live_modules_batch(
            alive,
            drives_np[:, 2],
            sim_cfg=self.sim_cfg,
            device=self.device,
            routing_batches=escap4_routing_batches,
        )

        # ESCAP-5 is deliberately not entered while disabled: the established
        # ESCAP-3 scalar path below then retains its exact allocation,
        # arithmetic, telemetry, and RNG behavior.  When enabled, compile the
        # legacy REST contribution and explicit output bindings into one
        # per-action residual matrix so REST is clipped and added only once.
        escap5_output_routing_batch = None
        if output_routing_enabled(self.sim_cfg):
            escap5_output_routing_batch = compile_output_routing_batch(
                alive,
                sim_cfg=self.sim_cfg,
                legacy_rest_residuals=escap3_rest_residual_np,
            )

        if buckets is not None:
            buckets["obs_build"] = buckets.get("obs_build", 0.0) + (time.perf_counter() - t_obs0)

        # -----------------------------
        # Per-tick index cache (GPU)
        # -----------------------------
        tick_cache = getattr(self, "_policy_tick_cache", None)
        if tick_cache is None:
            tick_cache = {}
            setattr(self, "_policy_tick_cache", tick_cache)

        alive_ids_key = tuple(int(a.agent_id) for a in alive)
        key = (
            alive_ids_key,
            tuple(mlp_idx),
            tuple(gru_idx),
            tuple(mlp_slots_list),
            tuple(mlp_tokens_list),
            tuple(gru_slots_list),
            tuple(gru_tokens_list),
            str(self.device),
            str(getattr(self, "_bank_device", getattr(self, "bank_device", self.device))),
        )
        cache_hit = tick_cache.get("key") == key

        # FDP (Phase 3): ordering proof (deterministic forward call trace).
        # Emits a compact trace of the alive agent order and GRU/MLP partition.
        try:
            enable_fwd = bool(int(os.getenv('EIEL_FDP_FORWARD_TRACE', '0') or '0')) or bool(int(os.getenv('EIEL_DETERMINISM_PROBE', '0') or '0'))
            if enable_fwd:
                cur_gen = int(getattr(self, 'gen', getattr(self, '_eiel_dbg_gen', -1)))
                cur_tick = int(getattr(self, 'tick', getattr(self, '_eiel_dbg_tick', -1)))

                def _parse_int_ranges(spec: str) -> set | None:
                    spec = str(spec or '').strip()
                    if not spec:
                        return set()
                    if spec == '*':
                        return None
                    out: set = set()
                    for part in spec.split(','):
                        part = part.strip()
                        if not part:
                            continue
                        if '-' in part:
                            a, b = part.split('-', 1)
                            try:
                                lo = int(a)
                                hi = int(b)
                            except Exception:
                                continue
                            if hi < lo:
                                lo, hi = hi, lo
                            out.update(range(lo, hi + 1))
                        else:
                            try:
                                out.add(int(part))
                            except Exception:
                                continue
                    return out

                gens_spec = os.getenv('EIEL_FDP_FORWARD_TRACE_GENS', '').strip()
                ticks_spec = os.getenv('EIEL_FDP_FORWARD_TRACE_TICKS', '').strip()

                # Default: when enabled indirectly by determinism probe, only gen=1 tick=0.
                if not gens_spec and not ticks_spec and not bool(int(os.getenv('EIEL_FDP_FORWARD_TRACE', '0') or '0')):
                    gens_spec = '1'
                    ticks_spec = '0'

                allow_all = bool(int(os.getenv('EIEL_FDP_FORWARD_TRACE_ALL', '0') or '0'))
                gset = _parse_int_ranges(gens_spec)
                tset = _parse_int_ranges(ticks_spec)
                ok_gen = True if allow_all or gset is None else (cur_gen in gset if gset else False)
                ok_tick = True if allow_all or tset is None else (cur_tick in tset if tset else False)

                if ok_gen and ok_tick:
                    run_dir = getattr(self, "run_dir", None) or getattr(self, "run_dir_path", None) or ""
                    if run_dir:
                        dbg = os.path.join(str(run_dir), 'debug')
                        os.makedirs(dbg, exist_ok=True)
                        outp = os.path.join(dbg, 'fdp_forward_trace.jsonl')

                        # Collect alive order (in the exact order used for obs/actions arrays).
                        rows = []
                        for j, a in enumerate(alive):
                            g = getattr(a, 'genome', None)
                            rows.append({
                                'i': int(j),
                                'agent_id': int(getattr(a, 'agent_id', 0) or 0),
                                'slot': int(getattr(a, 'slot', -1) or -1),
                                'slot_token': int(getattr(a, 'slot_token', 0) or 0),
                                'use_gru': bool(getattr(g, 'use_gru', False)) if g is not None else bool(self._is_gru_policy(getattr(a, 'policy', None))),
                                'memory_dim': int(getattr(g, 'memory_dim', 0) or 0) if g is not None else 0,
                                'bank_token': int(getattr(a, '_bank_token', -999999)),
                                'gru_bank_token': int(getattr(a, '_gru_bank_token', -999999)),
                            })

                        rec = {
                            'schema_version': 'fdp_forward_trace_v1',
                            'gen': int(cur_gen),
                            'tick': int(cur_tick),
                            'stage': 'actions.entry',
                            'n_alive': int(n),
                            'alive_order': rows,
                            'mlp_idx': [int(x) for x in mlp_idx],
                            'gru_idx': [int(x) for x in gru_idx],
                            'mlp_slots': [int(x) for x in mlp_slots_list],
                            'mlp_tokens': [int(x) for x in mlp_tokens_list],
                            'gru_slots': [int(x) for x in gru_slots_list],
                            'gru_tokens': [int(x) for x in gru_tokens_list],
                        }

                        with open(outp, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(rec, sort_keys=True, separators=(',', ':'), ensure_ascii=True))
                            f.write('\n')
        except Exception:
            pass

        # Phase 3 determinism: policy execution probe (gen1/tick0)
        self._eiel_policy_exec_probe(
            "actions1.entry",
            alive=alive,
            mlp_idx=mlp_idx,
            gru_idx=gru_idx,
            k=1,
            cache_hit=bool(cache_hit),
            mlp_slots=mlp_slots_list,
            mlp_tokens=mlp_tokens_list,
            gru_slots=gru_slots_list,
            gru_tokens=gru_tokens_list,
        )


        # ------------------------------------------------------------
        # Phase 3 determinism debug: stride K-trace (sequence shape audit)
        #
        # Enable with:
        #   EIEL_DEBUG_STRIDE_KTRACE=1
        # Optional:
        #   EIEL_DEBUG_STRIDE_TICKS=0,1,2,3
        # ------------------------------------------------------------
        if os.environ.get("EIEL_DEBUG_STRIDE_KTRACE", "0").strip() == "1":
            try:
                ticks_raw = os.environ.get("EIEL_DEBUG_STRIDE_TICKS", "0,1,2,3")
                _kt_ticks = {int(x) for x in ticks_raw.split(",") if x.strip()}
            except Exception:
                _kt_ticks = {0, 1, 2, 3}
            _kt_gen = int(getattr(self, "_eiel_dbg_gen", -1))
            _kt_tick = int(getattr(self, "_eiel_dbg_tick", -1))
            if _kt_tick in _kt_ticks:
                try:
                    _ids_head = [int(getattr(a, "agent_id")) for a in alive[:16]]
                except Exception:
                    _ids_head = []
                print(
                    f"[stride_ktrace] gen={_kt_gen} tick={_kt_tick} k={k} n={n} "
                    f"mlp_n={len(mlp_idx)} gru_n={len(gru_idx)} cache_hit={int(bool(cache_hit))} "
                    f"alive_head={_ids_head}",
                    flush=True,
                )

        if not cache_hit:
            tick_cache.clear()
            tick_cache["key"] = key

        # Neuromodulation weights (safe defaults)
        w_hunger_gather = float(getattr(self.sim_cfg, "drive_w_hunger_gather", 1.2))
        w_pain_rest = float(getattr(self.sim_cfg, "drive_w_pain_rest", 0.9))
        w_fatigue_rest = float(getattr(self.sim_cfg, "drive_w_fatigue_rest", 1.0))
        w_pain_attack = float(getattr(self.sim_cfg, "drive_w_pain_attack", 1.2))
        w_fear_attack = float(getattr(self.sim_cfg, "drive_w_fear_attack", 1.4))

        # -----------------------------
        # Staging buffers (reuse, grow-only)
        # -----------------------------
        t_stage0 = time.perf_counter()
        packed_cols = obs_size + 1 + 1 + 4 + 1

        # Cache buffers under a separate namespace key (so index cache doesn't get blown away)
        buf = tick_cache.get("buffers")
        if buf is None:
            buf = {}
            tick_cache["buffers"] = buf

        cap = int(buf.get("cap", 0) or 0)
        if cap < n or int(buf.get("packed_cols", 0) or 0) != packed_cols:
            # Grow to next power-ish to reduce realloc frequency
            new_cap = max(n, int(max(128, cap * 2)))
            buf["cap"] = new_cap
            buf["packed_cols"] = packed_cols

            # CPU staging (pinned if CUDA)
            pin = bool(self.device.type == "cuda")
            buf["packed_cpu"] = torch.empty((new_cap, packed_cols), device="cpu", dtype=torch.float32, pin_memory=pin)
            buf["packed_gpu"] = torch.empty((new_cap, packed_cols), device=self.device, dtype=torch.float32)

            buf["actions_all"] = torch.empty((new_cap,), device=self.device, dtype=torch.int64)

            # Convenience NumPy view into the pinned CPU tensor
            buf["packed_cpu_np"] = buf["packed_cpu"].numpy()

        packed_cpu_np = buf["packed_cpu_np"]
        packed_cpu_np[:n, :obs_size] = obs_np
        c = obs_size
        packed_cpu_np[:n, c] = aggr_np; c += 1
        packed_cpu_np[:n, c] = temp_np; c += 1
        packed_cpu_np[:n, c:c+4] = drives_np; c += 4
        packed_cpu_np[:n, c] = gain_np

        # One host->device copy (async when CUDA)
        packed_gpu = buf["packed_gpu"]
        t_copy0 = time.perf_counter()
        # Policy prep includes obs build + packing into pinned CPU buffers.
        if buckets is not None:
            buckets["policy_prep_ms"] = buckets.get("policy_prep_ms", 0.0) + (t_copy0 - t_policy0)
        packed_gpu[:n].copy_(buf["packed_cpu"][:n], non_blocking=True)
        if buckets is not None:
            buckets["h2d_copy_ms"] = buckets.get("h2d_copy_ms", 0.0) + (time.perf_counter() - t_copy0)

        t_fwd0 = time.perf_counter()

        x_all = packed_gpu[:n, :obs_size]
        aggr_all = packed_gpu[:n, obs_size]
        temp_all = packed_gpu[:n, obs_size + 1]
        drives_all = packed_gpu[:n, obs_size + 2: obs_size + 6]
        gain_all = packed_gpu[:n, obs_size + 6].clamp(min=0.0)
        cf_aggr_all = cf_temp_all = cf_gain_all = None
        if counterfactual_response_enabled and cf_aggr_np is not None and cf_temp_np is not None and cf_gain_np is not None:
            cf_aggr_all = torch.from_numpy(cf_aggr_np).to(self.device, non_blocking=True)
            cf_temp_all = torch.from_numpy(cf_temp_np).to(self.device, non_blocking=True)
            cf_gain_all = torch.from_numpy(cf_gain_np).to(self.device, non_blocking=True).clamp(min=0.0)

        if buckets is not None:
            buckets["tensor_stage"] = buckets.get("tensor_stage", 0.0) + (time.perf_counter() - t_stage0)

        actions_all = buf["actions_all"]
        capture_mem_max_level4 = bool(getattr(self.sim_cfg, "mem_max_live_level4_instrumentation_enable", False))
        logits_all = None
        if capture_mem_max_level4:
            logits_all = torch.zeros((n, int(self._out)), device=self.device, dtype=self._bank_dtype)

        # -----------------------------
        # Helpers: categorical sampling via Gumbel-Max
        # -----------------------------
        def _sample_actions_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
            # Sample from Categorical(softmax(logits)) without explicitly computing softmax.
            # Gumbel-Max trick: argmax(logits + gumbel(0,1))
            if getattr(self.sim_cfg, "policy_deterministic_actions", False):
                return torch.argmax(logits, dim=1), None
            rm = getattr(self, "rng_manager", None)
            if rm is None:
                raise RuntimeError(
                    "[DETERMINISM_VIOLATION]\n"
                    "code=DET_FORBIDDEN_RNG_API\n"
                    "context=PolicyMixin missing rng_manager for torch sampling\n"
                )
            tr = rm.torch_rng("policy", str(logits.device))
            if tr is None:
                raise RuntimeError(
                    "[DETERMINISM_VIOLATION]\n"
                    "code=DET_RNG_UNKNOWN_STREAM\n"
                    f"context=missing torch RNG stream policy device={str(logits.device)}\n"
                )
            u = tr.rand(tuple(int(x) for x in logits.shape), device=logits.device, dtype=logits.dtype)
            g = -torch.log(-torch.log(u.clamp(min=1e-6, max=1.0 - 1e-6)))
            return torch.argmax(logits + g, dim=1), g

        # -----------------------------
        # MLP fast path
        # -----------------------------
        if mlp_idx:
            t0 = time.perf_counter()
            mlp_agents = [alive[i] for i in mlp_idx]
            self._sync_bank_for_agents(mlp_agents, buckets=buckets)

            if cache_hit and "mlp_rows" in tick_cache and "mlp_slots" in tick_cache:
                mlp_idx_np = tick_cache["mlp_idx_np"]
                mlp_rows = tick_cache["mlp_rows"]
                slots = tick_cache["mlp_slots"]
            else:
                mlp_idx_np = np.asarray(mlp_idx, dtype=np.int64)
                mlp_rows = torch.from_numpy(mlp_idx_np).to(self.device, non_blocking=True)
                slots_np = np.asarray(mlp_slots_list, dtype=np.int64)
                slots = torch.from_numpy(slots_np).to(device=getattr(self, '_bank_device', self.device), dtype=torch.long, non_blocking=True)
                tick_cache["mlp_idx_np"] = mlp_idx_np
                tick_cache["mlp_rows"] = mlp_rows
                tick_cache["mlp_slots"] = slots

            x = x_all.index_select(0, mlp_rows)
            aggr = aggr_all.index_select(0, mlp_rows)
            temps = temp_all.index_select(0, mlp_rows)
            d = drives_all.index_select(0, mlp_rows)
            gain = gain_all.index_select(0, mlp_rows)
            cf_aggr = cf_temps = cf_gain = None
            if counterfactual_response_enabled and cf_aggr_all is not None and cf_temp_all is not None and cf_gain_all is not None:
                cf_aggr = cf_aggr_all.index_select(0, mlp_rows)
                cf_temps = cf_temp_all.index_select(0, mlp_rows)
                cf_gain = cf_gain_all.index_select(0, mlp_rows)

            # Bank gather (single index_select)
            # ---- EIEL debug: slot index diagnostics (resume/gen0) ----
            # Enabled by env var: EIEL_DEBUG_POLICY_SLOTS=1
            dbg_slots = bool(int(os.getenv("EIEL_DEBUG_POLICY_SLOTS", "0")))
            if dbg_slots and getattr(self, "_mlp_pack", None) is not None:
                try:
                    cap0 = int(self._mlp_pack.shape[0])
                    slots_cpu = slots.detach().to("cpu")
                    if int(slots_cpu.numel()) > 0 and cap0 > 0:
                        smin0 = int(slots_cpu.min().item())
                        smax0 = int(slots_cpu.max().item())
                        bad_pos0 = int((slots_cpu >= cap0).sum().item())
                        bad_neg0 = int((slots_cpu < -cap0).sum().item())
                        sample0 = slots_cpu[: min(16, int(slots_cpu.numel()))].tolist()
                        self._policy_slots_log(f"[EIEL_DEBUG_POLICY_SLOTS pre] cap={cap0} smin={smin0} smax={smax0} bad_pos={bad_pos0} bad_neg={bad_neg0} sample={sample0}")
                        need = cap0
                        if smax0 >= cap0:
                            need = max(need, smax0 + 1)
                        if smin0 < -cap0:
                            need = max(need, abs(smin0) + 1)
                        if need > cap0 and hasattr(self, "_grow_mlp_bank"):
                            self._grow_mlp_bank(max_slots=int(need))
                            cap1 = int(self._mlp_pack.shape[0])
                            bad_pos1 = int((slots_cpu >= cap1).sum().item()) if cap1 > 0 else -1
                            bad_neg1 = int((slots_cpu < -cap1).sum().item()) if cap1 > 0 else -1
                            self._policy_slots_log(f"[EIEL_DEBUG_POLICY_SLOTS post] cap={cap1} need={need} bad_pos={bad_pos1} bad_neg={bad_neg1}")
                        # Optional fail-fast with clear message (default ON when debug enabled)
                        failfast = bool(int(os.getenv("EIEL_DEBUG_POLICY_SLOTS_FAILFAST", "1")))
                        cap_chk = int(self._mlp_pack.shape[0])
                        if failfast and cap_chk > 0:
                            if smax0 >= cap_chk or smin0 < -cap_chk:
                                raise RuntimeError(
                                    f"Invalid slot indices for MLP bank: cap={cap_chk} smin={smin0} smax={smax0} "
                                    f"(bad_pos={int((slots_cpu >= cap_chk).sum().item())}, bad_neg={int((slots_cpu < -cap_chk).sum().item())})"
                                )
                except Exception as _e:
                    # Re-raise under debug to stop at the first bad tick with context.
                    raise
            # ---- end EIEL debug ----
            self._eiel_bank_device_sanity_probe(stage="bank_select.mlp_pack", bank_name="mlp_pack", bank=getattr(self, '_mlp_pack', None), slots=slots)
            pack = self._mlp_pack.index_select(0, slots)
            pack = self._eiel_to_compute(pack, name="mlp_pack_rows", stage="bank_select.mlp_pack")
            H = int(self._hidden)
            O = int(self._obs_size)
            A = int(self._out)
            off = 0
            W1 = pack[:, off:off + (H * O)].view(-1, H, O); off += (H * O)
            b1 = pack[:, off:off + H].view(-1, H); off += H
            W2 = pack[:, off:off + (H * H)].view(-1, H, H); off += (H * H)
            b2 = pack[:, off:off + H].view(-1, H); off += H
            W3 = pack[:, off:off + (A * H)].view(-1, A, H); off += (A * H)
            b3 = pack[:, off:off + A].view(-1, A)

            logits = forward_batched_mlp(
                x,
                W1,
                b1,
                W2,
                b2,
                W3,
                b3,
                attack_bias=aggr,
                action_temperature=temps,
            )

            # Drive neuromodulation (in logits space)
            if logits.numel() != 0:
                logits[:, ACT_GATHER] = logits[:, ACT_GATHER] + gain * (w_hunger_gather * d[:, 0])
                logits[:, ACT_REST] = logits[:, ACT_REST] + gain * (w_pain_rest * d[:, 1] + w_fatigue_rest * d[:, 2])
                logits[:, ACT_ATTACK] = logits[:, ACT_ATTACK] - gain * (w_pain_attack * d[:, 1] + w_fear_attack * d[:, 3])
                if escap5_output_routing_batch is not None:
                    base_logits = logits.detach().clone()
                    residuals = torch.from_numpy(
                        escap5_output_routing_batch.residuals[mlp_idx_np]
                    ).to(device=logits.device, dtype=logits.dtype)
                    add_output_routing_residuals_(logits, residuals)
                    _record_output_routing_telemetry(
                        alive,
                        list(mlp_idx),
                        batch=escap5_output_routing_batch,
                        base_logits=base_logits,
                    )
                elif live_modules_enabled(self.sim_cfg):
                    residual = torch.from_numpy(escap3_rest_residual_np[mlp_idx_np]).to(
                        device=logits.device, dtype=logits.dtype
                    )
                    add_rest_logit_residual_(logits, residual, rest_index=ACT_REST)

            self._eiel_policy_exec_probe(
                "actions1.pre_sample_mlp",
                alive=alive,
                mlp_idx=mlp_idx,
                gru_idx=gru_idx,
                k=1,
                cache_hit=bool(cache_hit),
                extra={"rand_shape": [int(logits.shape[0]), int(logits.shape[1])]},
                logits=logits,
            )
            opportunity_vectors, valid_family_mask, valid_action_mask = _polarch12_binary_opportunity_inputs(
                self,
                alive,
                list(mlp_idx),
                device=logits.device,
                dtype=logits.dtype,
            )
            choice_logits, authority_summary = apply_authority_to_logits_torch(
                logits,
                alive=alive,
                row_indices=list(mlp_idx),
                sim_cfg=self.sim_cfg,
                opportunity_vectors=opportunity_vectors,
                valid_family_mask=valid_family_mask,
                valid_action_mask=valid_action_mask,
            )
            _polarch6_merge_authority_summary(self, authority_summary, policy_kind="mlp")
            record_policy_margins(self, alive=alive, row_indices=list(mlp_idx), logits=choice_logits)
            actions_mlp, action_noise = _sample_actions_from_logits(choice_logits)
            record_family_head_authority_choice_margin(
                self,
                alive=alive,
                row_indices=list(mlp_idx),
                neutral_logits=logits,
                directional_logits=choice_logits,
                actions=actions_mlp,
                sample_noise=action_noise,
                policy_kind="mlp",
                action_source_class="neural_policy_direct",
            )
            record_policy_action_family_decisions(
                self,
                alive=alive,
                row_indices=list(mlp_idx),
                logits=choice_logits,
                actions=actions_mlp,
                sample_noise=action_noise,
                policy_kind="mlp",
                drives=d,
                action_temperature=temps,
                action_source_class="neural_policy_direct",
            )
            if logits_all is not None:
                logits_all.index_copy_(0, mlp_rows, logits.detach().to(dtype=self._bank_dtype))
            if counterfactual_response_enabled and cf_aggr is not None and cf_temps is not None and cf_gain is not None:
                cf_logits = forward_batched_mlp(
                    x,
                    W1,
                    b1,
                    W2,
                    b2,
                    W3,
                    b3,
                    attack_bias=cf_aggr,
                    action_temperature=cf_temps,
                )
                if cf_logits.numel() != 0:
                    cf_logits[:, ACT_GATHER] = cf_logits[:, ACT_GATHER] + cf_gain * (w_hunger_gather * d[:, 0])
                    cf_logits[:, ACT_REST] = cf_logits[:, ACT_REST] + cf_gain * (w_pain_rest * d[:, 1] + w_fatigue_rest * d[:, 2])
                    cf_logits[:, ACT_ATTACK] = cf_logits[:, ACT_ATTACK] - cf_gain * (w_pain_attack * d[:, 1] + w_fear_attack * d[:, 3])
                record_counterfactual_action_response(
                    self,
                    alive=alive,
                    row_indices=list(mlp_idx),
                    actual_logits=choice_logits,
                    counterfactual_logits=cf_logits,
                    actual_actions=actions_mlp,
                    sample_noise=action_noise,
                    status_by_row=[cf_status[i] for i in mlp_idx] if cf_status is not None else None,
                    policy_kind="mlp",
                    action_source_class="neural_policy_direct",
                )
            self._eiel_policy_exec_probe(
                "actions1.post_sample_mlp",
                alive=alive,
                mlp_idx=mlp_idx,
                gru_idx=gru_idx,
                k=1,
                cache_hit=bool(cache_hit),
            )
            actions_all.index_copy_(0, mlp_rows, actions_mlp)

            if buckets is not None:
                buckets["policy_forward_mlp"] = buckets.get("policy_forward_mlp", 0.0) + (time.perf_counter() - t0)

        # -----------------------------
        # GRU stateful path with masking
        # -----------------------------
        if gru_idx:
            t0 = time.perf_counter()

            gru_agents = [alive[i] for i in gru_idx]
            self._sync_gru_bank_for_agents(gru_agents, buckets=buckets)

            if cache_hit and "gru_rows" in tick_cache and "gru_slots" in tick_cache:
                gru_idx_np = tick_cache["gru_idx_np"]
                gru_rows = tick_cache["gru_rows"]
                slots = tick_cache["gru_slots"]
            else:
                gru_idx_np = np.asarray(gru_idx, dtype=np.int64)
                gru_rows = torch.from_numpy(gru_idx_np).to(self.device, non_blocking=True)
                slots_np = np.asarray(gru_slots_list, dtype=np.int64)
                slots = torch.from_numpy(slots_np).to(device=getattr(self, '_bank_device', self.device), dtype=torch.long, non_blocking=True)
                tick_cache["gru_idx_np"] = gru_idx_np
                tick_cache["gru_rows"] = gru_rows
                tick_cache["gru_slots"] = slots
            mem_dims_np = np.asarray(
                [int(effective_memory_dim_for_genome(getattr(a, "genome", None), sim_cfg=self.sim_cfg)) for a in gru_agents],
                dtype=np.int64,
            )
            mem_dims = torch.from_numpy(mem_dims_np).to(self.device, non_blocking=True)

            x = x_all.index_select(0, gru_rows)
            aggr = aggr_all.index_select(0, gru_rows)
            temps = temp_all.index_select(0, gru_rows)
            d = drives_all.index_select(0, gru_rows)
            gain = gain_all.index_select(0, gru_rows)
            cf_aggr = cf_temps = cf_gain = None
            if counterfactual_response_enabled and cf_aggr_all is not None and cf_temp_all is not None and cf_gain_all is not None:
                cf_aggr = cf_aggr_all.index_select(0, gru_rows)
                cf_temps = cf_temp_all.index_select(0, gru_rows)
                cf_gain = cf_gain_all.index_select(0, gru_rows)

            self._eiel_bank_device_sanity_probe(stage="bank_select.gru_hmem", bank_name="Hmem", bank=getattr(self, '_Hmem', None), slots=slots)
            h_prev = self._Hmem.index_select(0, slots)
            h_prev = self._eiel_to_compute(h_prev, name="gru_h_prev", stage="bank_select.gru_hmem")
            Hm = int(h_prev.shape[1])
            mem_dims = torch.clamp(mem_dims, 0, Hm)

            idx = tick_cache.get("mask_idx")
            if idx is None or int(idx.shape[1]) != Hm:
                idx = torch.arange(Hm, device=self.device, dtype=torch.long).unsqueeze(0)
                tick_cache["mask_idx"] = idx
            mask = (idx < mem_dims.unsqueeze(1)).to(h_prev.dtype)
            h_prev = h_prev * mask
            reset_count = 0
            if bool(memory_ablation_flags(self.sim_cfg).get("reset_each_tick", False)):
                reset_count = int(h_prev.shape[0])
                h_prev = torch.zeros_like(h_prev)

            self._eiel_bank_device_sanity_probe(stage="bank_select.gru_pack", bank_name="gru_pack", bank=getattr(self, '_gru_pack', None), slots=slots)
            pack = self._gru_pack.index_select(0, slots)
            pack = self._eiel_to_compute(pack, name="gru_pack_rows", stage="bank_select.gru_pack")
            O = int(self._obs_size)
            A = int(self._out)
            Hm = int(self._gru_hidden)
            off = 0
            encW = pack[:, off:off + (Hm * O)].view(-1, Hm, O); off += (Hm * O)
            encb = pack[:, off:off + Hm].view(-1, Hm); off += Hm
            Wih = pack[:, off:off + (3 * Hm * Hm)].view(-1, 3 * Hm, Hm); off += (3 * Hm * Hm)
            bih = pack[:, off:off + (3 * Hm)].view(-1, 3 * Hm); off += (3 * Hm)
            Whh = pack[:, off:off + (3 * Hm * Hm)].view(-1, 3 * Hm, Hm); off += (3 * Hm * Hm)
            bhh = pack[:, off:off + (3 * Hm)].view(-1, 3 * Hm); off += (3 * Hm)
            headW = pack[:, off:off + (A * Hm)].view(-1, A, Hm); off += (A * Hm)
            headb = pack[:, off:off + A].view(-1, A)

            logits, h_next = forward_batched_gru(
                x=x,
                h_prev=h_prev,
                encW=encW,
                encb=encb,
                Wih=Wih,
                bih=bih,
                Whh=Whh,
                bhh=bhh,
                headW=headW,
                headb=headb,
                attack_bias=aggr,
                action_temperature=temps,
            )

            if logits.numel() != 0:
                logits[:, ACT_GATHER] = logits[:, ACT_GATHER] + gain * (w_hunger_gather * d[:, 0])
                logits[:, ACT_REST] = logits[:, ACT_REST] + gain * (w_pain_rest * d[:, 1] + w_fatigue_rest * d[:, 2])
                logits[:, ACT_ATTACK] = logits[:, ACT_ATTACK] - gain * (w_pain_attack * d[:, 1] + w_fear_attack * d[:, 3])
                if escap5_output_routing_batch is not None:
                    base_logits = logits.detach().clone()
                    residuals = torch.from_numpy(
                        escap5_output_routing_batch.residuals[gru_idx_np]
                    ).to(device=logits.device, dtype=logits.dtype)
                    add_output_routing_residuals_(logits, residuals)
                    _record_output_routing_telemetry(
                        alive,
                        list(gru_idx),
                        batch=escap5_output_routing_batch,
                        base_logits=base_logits,
                    )
                elif live_modules_enabled(self.sim_cfg):
                    residual = torch.from_numpy(escap3_rest_residual_np[gru_idx_np]).to(
                        device=logits.device, dtype=logits.dtype
                    )
                    add_rest_logit_residual_(logits, residual, rest_index=ACT_REST)

            h_next = h_next * mask
            self._record_memory_runtime_telemetry(
                h_prev=h_prev,
                h_next=h_next,
                mask=mask,
                reset_count=reset_count,
            )
            h_next_bank = self._eiel_to_bank(h_next, like=self._Hmem, name="gru_h_next", stage="bank_update.gru_hmem")
            self._Hmem.index_copy_(0, slots, h_next_bank)

            self._eiel_policy_exec_probe(
                "actions1.pre_sample_gru",
                alive=alive,
                mlp_idx=mlp_idx,
                gru_idx=gru_idx,
                k=1,
                cache_hit=bool(cache_hit),
                extra={"rand_shape": [int(logits.shape[0]), int(logits.shape[1])]},
                logits=logits,
            )
            opportunity_vectors, valid_family_mask, valid_action_mask = _polarch12_binary_opportunity_inputs(
                self,
                alive,
                list(gru_idx),
                device=logits.device,
                dtype=logits.dtype,
            )
            choice_logits, authority_summary = apply_authority_to_logits_torch(
                logits,
                alive=alive,
                row_indices=list(gru_idx),
                sim_cfg=self.sim_cfg,
                opportunity_vectors=opportunity_vectors,
                valid_family_mask=valid_family_mask,
                valid_action_mask=valid_action_mask,
            )
            _polarch6_merge_authority_summary(self, authority_summary, policy_kind="gru")
            record_policy_margins(self, alive=alive, row_indices=list(gru_idx), logits=choice_logits)
            actions_gru, action_noise = _sample_actions_from_logits(choice_logits)
            record_family_head_authority_choice_margin(
                self,
                alive=alive,
                row_indices=list(gru_idx),
                neutral_logits=logits,
                directional_logits=choice_logits,
                actions=actions_gru,
                sample_noise=action_noise,
                policy_kind="gru",
                action_source_class="neural_policy_direct",
            )
            record_policy_action_family_decisions(
                self,
                alive=alive,
                row_indices=list(gru_idx),
                logits=choice_logits,
                actions=actions_gru,
                sample_noise=action_noise,
                policy_kind="gru",
                drives=d,
                action_temperature=temps,
                action_source_class="neural_policy_direct",
            )
            if logits_all is not None:
                logits_all.index_copy_(0, gru_rows, logits.detach().to(dtype=self._bank_dtype))
            if counterfactual_response_enabled and cf_aggr is not None and cf_temps is not None and cf_gain is not None:
                cf_logits, _cf_h_next = forward_batched_gru(
                    x=x,
                    h_prev=h_prev,
                    encW=encW,
                    encb=encb,
                    Wih=Wih,
                    bih=bih,
                    Whh=Whh,
                    bhh=bhh,
                    headW=headW,
                    headb=headb,
                    attack_bias=cf_aggr,
                    action_temperature=cf_temps,
                )
                if cf_logits.numel() != 0:
                    cf_logits[:, ACT_GATHER] = cf_logits[:, ACT_GATHER] + cf_gain * (w_hunger_gather * d[:, 0])
                    cf_logits[:, ACT_REST] = cf_logits[:, ACT_REST] + cf_gain * (w_pain_rest * d[:, 1] + w_fatigue_rest * d[:, 2])
                    cf_logits[:, ACT_ATTACK] = cf_logits[:, ACT_ATTACK] - cf_gain * (w_pain_attack * d[:, 1] + w_fear_attack * d[:, 3])
                record_counterfactual_action_response(
                    self,
                    alive=alive,
                    row_indices=list(gru_idx),
                    actual_logits=choice_logits,
                    counterfactual_logits=cf_logits,
                    actual_actions=actions_gru,
                    sample_noise=action_noise,
                    status_by_row=[cf_status[i] for i in gru_idx] if cf_status is not None else None,
                    policy_kind="gru",
                    action_source_class="neural_policy_direct",
                )
            self._eiel_policy_exec_probe(
                "actions1.post_sample_gru",
                alive=alive,
                mlp_idx=mlp_idx,
                gru_idx=gru_idx,
                k=1,
                cache_hit=bool(cache_hit),
            )
            actions_all.index_copy_(0, gru_rows, actions_gru)

            if buckets is not None:
                buckets["policy_forward_gru"] = buckets.get("policy_forward_gru", 0.0) + (time.perf_counter() - t0)

        if buckets is not None:
            buckets["policy_fwd_ms"] = buckets.get("policy_fwd_ms", 0.0) + (time.perf_counter() - t_fwd0)

        # One device->host transfer for the final action vector.
        # If device is CPU, this is just a view conversion.
        t_output0 = time.perf_counter()
        actions_np = actions_all[:n].to("cpu").numpy().astype(np.int32, copy=False)
        if logits_all is not None:
            logits_np = logits_all[:n].detach().to("cpu").numpy().astype(np.float32, copy=True)
            probs_np = torch.softmax(logits_all[:n].detach().to(torch.float32), dim=1).to("cpu").numpy().astype(np.float32, copy=True)
            agent_ids = [int(getattr(agent, "agent_id")) for agent in alive]
            self._mem_max_live_level4_last_policy = {
                "agent_ids": agent_ids,
                "logits_by_agent_id": {int(aid): logits_np[i].copy() for i, aid in enumerate(agent_ids)},
                "probs_by_agent_id": {int(aid): probs_np[i].copy() for i, aid in enumerate(agent_ids)},
            }
        if buckets is not None:
            buckets["policy_output_transfer"] = buckets.get("policy_output_transfer", 0.0) + (
                time.perf_counter() - t_output0
            )
        self._eiel_policy_exec_probe(
            "actions1.exit",
            alive=alive,
            mlp_idx=mlp_idx,
            gru_idx=gru_idx,
            k=1,
            cache_hit=bool(cache_hit),
            actions_out=actions_np,
        )
        return actions_np


    def _policy_cpu_numpy_forward_requested(self) -> bool:
        if bool(getattr(self.sim_cfg, "structural_counterfactual_action_response_reporting_enable", False)):
            return False
        if bool(getattr(self.sim_cfg, "structural_policy_action_family_mapping_reporting_enable", False)):
            return False
        if bool(getattr(self.sim_cfg, "structural_action_consequence_telemetry_reporting_enable", False)):
            return False
        if not bool(getattr(self.sim_cfg, "perf_policy_cpu_numpy_forward_enable", False)):
            return False
        if family_head_authority_enabled(getattr(self, "sim_cfg", None)):
            raise RuntimeError("POLARCH6-W6-UNSUPPORTED: numpy policy fast path does not implement family-head authority")
        device = getattr(self, "device", None)
        if str(device) != "cpu" and str(getattr(device, "type", "")) != "cpu":
            return False
        bank_device = getattr(self, "_bank_device", getattr(self, "bank_device", device))
        if str(bank_device) != "cpu" and str(getattr(bank_device, "type", "")) != "cpu":
            return False
        return True

    def _batched_actions_k_cpu_numpy(
        self,
        *,
        alive,
        k: int,
        n: int,
        obs_np: np.ndarray,
        aggr_np: np.ndarray,
        temp_np: np.ndarray,
        drives_np: np.ndarray,
        gain_np: np.ndarray,
        mlp_idx: List[int],
        gru_idx: List[int],
        mlp_slots_list: List[int],
        gru_slots_list: List[int],
        buckets: Optional[Dict[str, float]],
        w_hunger_gather: float,
        w_pain_rest: float,
        w_fatigue_rest: float,
        w_pain_attack: float,
        w_fear_attack: float,
    ) -> np.ndarray | None:
        if not self._policy_cpu_numpy_forward_requested():
            return None

        mlp_pack_np = _torch_cpu_numpy_view(getattr(self, "_mlp_pack", None)) if mlp_idx else None
        gru_pack_np = _torch_cpu_numpy_view(getattr(self, "_gru_pack", None)) if gru_idx else None
        hmem_np = _torch_cpu_numpy_view(getattr(self, "_Hmem", None)) if gru_idx else None
        if (mlp_idx and mlp_pack_np is None) or (gru_idx and (gru_pack_np is None or hmem_np is None)):
            return None
        mlp_slots_np = np.asarray(mlp_slots_list, dtype=np.int64) if mlp_idx else np.empty((0,), dtype=np.int64)
        gru_slots_np = np.asarray(gru_slots_list, dtype=np.int64) if gru_idx else np.empty((0,), dtype=np.int64)
        if mlp_idx and (
            mlp_pack_np is None
            or bool(np.any(mlp_slots_np < 0))
            or bool(np.any(mlp_slots_np >= int(mlp_pack_np.shape[0])))
        ):
            return None
        if gru_idx and (
            gru_pack_np is None
            or hmem_np is None
            or bool(np.any(gru_slots_np < 0))
            or bool(np.any(gru_slots_np >= int(gru_pack_np.shape[0])))
            or bool(np.any(gru_slots_np >= int(hmem_np.shape[0])))
        ):
            return None

        total_started = time.perf_counter()
        actions_all_k = np.empty((int(n), int(k)), dtype=np.int32)
        deterministic = bool(getattr(self.sim_cfg, "policy_deterministic_actions", False))

        def _sample_k_from_logits(logits: np.ndarray, k_samples: int) -> np.ndarray:
            logits = np.asarray(logits, dtype=np.float32)
            if deterministic:
                a = np.argmax(logits, axis=1).astype(np.int32, copy=False)
                return np.repeat(a[:, None], int(k_samples), axis=1).astype(np.int32, copy=False)
            bsz, actions = logits.shape
            u = self.rng_manager.policy_np.random(int(bsz) * int(k_samples) * int(actions), dtype=np.float32)
            u = u.reshape((int(bsz), int(k_samples), int(actions)))
            g = -np.log(-np.log(np.clip(u, np.float32(1.0e-6), np.float32(1.0 - 1.0e-6))))
            return np.argmax(logits[:, None, :] + g.astype(np.float32, copy=False), axis=2).astype(np.int32, copy=False)

        def _sample_from_logits(logits: np.ndarray) -> np.ndarray:
            logits = np.asarray(logits, dtype=np.float32)
            if deterministic:
                return np.argmax(logits, axis=1).astype(np.int32, copy=False)
            bsz, actions = logits.shape
            u = self.rng_manager.policy_np.random(int(bsz) * int(actions), dtype=np.float32).reshape(
                (int(bsz), int(actions))
            )
            g = -np.log(-np.log(np.clip(u, np.float32(1.0e-6), np.float32(1.0 - 1.0e-6))))
            return np.argmax(logits + g.astype(np.float32, copy=False), axis=1).astype(np.int32, copy=False)

        if mlp_idx:
            t0 = time.perf_counter()
            mlp_agents = [alive[i] for i in mlp_idx]
            self._sync_bank_for_agents(mlp_agents, buckets=buckets)
            rows_np = np.asarray(mlp_idx, dtype=np.int64)
            slots_np = mlp_slots_np
            pack = np.asarray(mlp_pack_np[slots_np], dtype=np.float32)
            x = np.asarray(obs_np[rows_np], dtype=np.float32)
            aggr = np.asarray(aggr_np[rows_np], dtype=np.float32)
            temps = np.asarray(temp_np[rows_np], dtype=np.float32)
            d = np.asarray(drives_np[rows_np], dtype=np.float32)
            gain = np.asarray(gain_np[rows_np], dtype=np.float32)
            h = int(self._hidden)
            obs = int(self._obs_size)
            actions = int(self._out)
            off = 0
            w1 = pack[:, off : off + (h * obs)].reshape((-1, h, obs)); off += h * obs
            b1 = pack[:, off : off + h].reshape((-1, h)); off += h
            w2 = pack[:, off : off + (h * h)].reshape((-1, h, h)); off += h * h
            b2 = pack[:, off : off + h].reshape((-1, h)); off += h
            w3 = pack[:, off : off + (actions * h)].reshape((-1, actions, h)); off += actions * h
            b3 = pack[:, off : off + actions].reshape((-1, actions))
            h1 = np.tanh(_np_batched_linear(x, w1, b1)).astype(np.float32, copy=False)
            h2 = np.tanh(_np_batched_linear(h1, w2, b2)).astype(np.float32, copy=False)
            logits = _np_batched_linear(h2, w3, b3)
            logits[:, ACT_ATTACK] = logits[:, ACT_ATTACK] + aggr
            logits = logits / np.maximum(temps, np.float32(0.05))[:, None]
            if logits.size:
                logits[:, ACT_GATHER] = logits[:, ACT_GATHER] + gain * (np.float32(w_hunger_gather) * d[:, 0])
                logits[:, ACT_REST] = logits[:, ACT_REST] + gain * (
                    np.float32(w_pain_rest) * d[:, 1] + np.float32(w_fatigue_rest) * d[:, 2]
                )
                logits[:, ACT_ATTACK] = logits[:, ACT_ATTACK] - gain * (
                    np.float32(w_pain_attack) * d[:, 1] + np.float32(w_fear_attack) * d[:, 3]
                )
            record_policy_margins(self, alive=alive, row_indices=list(mlp_idx), logits=logits)
            actions_all_k[rows_np, :] = _sample_k_from_logits(logits, int(k))
            if buckets is not None:
                buckets["policy_forward_mlp"] = buckets.get("policy_forward_mlp", 0.0) + (time.perf_counter() - t0)

        if gru_idx:
            t0 = time.perf_counter()
            gru_agents = [alive[i] for i in gru_idx]
            self._sync_gru_bank_for_agents(gru_agents, buckets=buckets)
            rows_np = np.asarray(gru_idx, dtype=np.int64)
            slots_np = gru_slots_np
            x = np.asarray(obs_np[rows_np], dtype=np.float32)
            aggr = np.asarray(aggr_np[rows_np], dtype=np.float32)
            temps = np.asarray(temp_np[rows_np], dtype=np.float32)
            d = np.asarray(drives_np[rows_np], dtype=np.float32)
            gain = np.asarray(gain_np[rows_np], dtype=np.float32)
            h_prev = np.asarray(hmem_np[slots_np], dtype=np.float32).copy()
            hm = int(self._gru_hidden)
            mem_dims_np = np.asarray(
                [int(effective_memory_dim_for_genome(getattr(a, "genome", None), sim_cfg=self.sim_cfg)) for a in gru_agents],
                dtype=np.int64,
            )
            mem_dims_np = np.clip(mem_dims_np, 0, hm)
            mask = (np.arange(hm, dtype=np.int64)[None, :] < mem_dims_np[:, None]).astype(np.float32, copy=False)
            h_prev = (h_prev * mask).astype(np.float32, copy=False)
            reset_count = 0
            if bool(memory_ablation_flags(self.sim_cfg).get("reset_each_tick", False)):
                reset_count = int(h_prev.shape[0])
                h_prev = np.zeros_like(h_prev, dtype=np.float32)
            pack = np.asarray(gru_pack_np[slots_np], dtype=np.float32)
            obs = int(self._obs_size)
            actions = int(self._out)
            off = 0
            enc_w = pack[:, off : off + (hm * obs)].reshape((-1, hm, obs)); off += hm * obs
            enc_b = pack[:, off : off + hm].reshape((-1, hm)); off += hm
            wih = pack[:, off : off + (3 * hm * hm)].reshape((-1, 3 * hm, hm)); off += 3 * hm * hm
            bih = pack[:, off : off + (3 * hm)].reshape((-1, 3 * hm)); off += 3 * hm
            whh = pack[:, off : off + (3 * hm * hm)].reshape((-1, 3 * hm, hm)); off += 3 * hm * hm
            bhh = pack[:, off : off + (3 * hm)].reshape((-1, 3 * hm)); off += 3 * hm
            head_w = pack[:, off : off + (actions * hm)].reshape((-1, actions, hm)); off += actions * hm
            head_b = pack[:, off : off + actions].reshape((-1, actions))
            z_in = np.tanh(_np_batched_linear(x, enc_w, enc_b)).astype(np.float32, copy=False)
            i_proj = _np_batched_linear(z_in, wih, bih)
            h_proj = _np_batched_linear(h_prev, whh, bhh)
            i_r, i_z, i_n = np.split(i_proj, 3, axis=1)
            h_r, h_z, h_n = np.split(h_proj, 3, axis=1)
            r = _np_sigmoid(i_r + h_r)
            z = _np_sigmoid(i_z + h_z)
            n_gate = np.tanh(i_n + r * h_n).astype(np.float32, copy=False)
            h_next = ((np.float32(1.0) - z) * n_gate + z * h_prev).astype(np.float32, copy=False)
            logits = _np_batched_linear(h_next, head_w, head_b)
            logits[:, ACT_ATTACK] = logits[:, ACT_ATTACK] + aggr
            logits = logits / np.maximum(temps, np.float32(0.05))[:, None]
            if logits.size:
                logits[:, ACT_GATHER] = logits[:, ACT_GATHER] + gain * (np.float32(w_hunger_gather) * d[:, 0])
                logits[:, ACT_REST] = logits[:, ACT_REST] + gain * (
                    np.float32(w_pain_rest) * d[:, 1] + np.float32(w_fatigue_rest) * d[:, 2]
                )
                logits[:, ACT_ATTACK] = logits[:, ACT_ATTACK] - gain * (
                    np.float32(w_pain_attack) * d[:, 1] + np.float32(w_fear_attack) * d[:, 3]
                )
            record_policy_margins(self, alive=alive, row_indices=list(gru_idx), logits=logits)
            h_next = (h_next * mask).astype(np.float32, copy=False)
            if bool(getattr(getattr(self, "sim_cfg", None), "enable_cognitive_status_metrics", True)):
                self._record_memory_runtime_telemetry(
                    h_prev=torch.from_numpy(h_prev),
                    h_next=torch.from_numpy(h_next),
                    mask=torch.from_numpy(mask),
                    reset_count=reset_count,
                )
            hmem_np[slots_np, :] = h_next
            actions_gru = _sample_from_logits(logits)
            actions_all_k[rows_np, :] = np.repeat(actions_gru[:, None], int(k), axis=1)
            if buckets is not None:
                buckets["policy_forward_gru"] = buckets.get("policy_forward_gru", 0.0) + (time.perf_counter() - t0)

        if buckets is not None:
            buckets["policy_fwd_ms"] = buckets.get("policy_fwd_ms", 0.0) + (time.perf_counter() - total_started)
        return actions_all_k.astype(np.int32, copy=False)


    def _batched_actions_k(
        self,
        alive,
        k: int,
        buckets: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Compute *k* actions per alive agent using batched inference.

        This powers multi-tick policy batching (SimConfig.policy_tick_stride > 1).

        Behavior
        --------
        - MLP agents: compute logits once and sample *k* actions open-loop.
        - GRU agents: perform ONE GRU update and repeat the resulting action *k*
          times (macro-action / latent-time abstraction).

        Notes
        -----
        - Slot/token safety is enforced by the tick-loop cache (slot_token guards).
        - When k==1 this delegates to `_batched_actions` to preserve exact legacy semantics.

        Returns
        -------
        np.ndarray:
            Shape (n, k), dtype int32.
        """
        k = int(k)
        if k < 1:
            k = 1

        n = len(alive)
        if n == 0:
            return np.empty((0, k), dtype=np.int32)
        if bool(
            getattr(
                getattr(self, "sim_cfg", None),
                "recomb72_neutral_actions_enable",
                False,
            )
        ):
            action_id = int(getattr(self.sim_cfg, "recomb72_neutral_action_id", 5))
            if action_id != 5:
                raise RuntimeError(
                    "RECOMB72_AUDIT_CONTRACT: neutral action must be ACT_REST"
                )
            self._recomb72_neutral_action_agent_ticks = int(
                getattr(self, "_recomb72_neutral_action_agent_ticks", 0)
            ) + n * k
            return np.full((n, k), action_id, dtype=np.int32)

        # Phase 0 perf: capture coarse policy timing buckets (values are seconds).
        t_policy0 = time.perf_counter()

        if k == 1:
            a1 = self._batched_actions(alive, buckets=buckets)
            return a1.reshape(-1, 1).astype(np.int32, copy=False)

        t_obs0 = time.perf_counter()

        local_size = patch_world_local_observation_size(self.env_cfg)
        obs_size = patch_world_total_observation_size(self.env_cfg)

        # ---- Observations (env-side) ----
        agent_ids_np = np.asarray([a.agent_id for a in alive], dtype=np.int32)
        local = self.env.get_local_observations(agent_ids_np)

        # ---- Build per-agent scalars in one pass (CPU) ----
        obs_np = np.empty((n, obs_size), dtype=np.float32)
        obs_np[:, :local_size] = local

        e_norm = float(self.env_cfg.max_energy)
        h_norm = float(self.env_cfg.start_health)

        energy_norm = np.empty((n,), dtype=np.float32)
        health_norm = np.empty((n,), dtype=np.float32)
        aggr_np = np.empty((n,), dtype=np.float32)
        temp_np = np.empty((n,), dtype=np.float32)

        # Drives: [hunger, pain, fatigue, fear]
        drives_np = np.zeros((n, 4), dtype=np.float32)
        gain_np = np.ones((n,), dtype=np.float32)
        counterfactual_response_enabled = bool(
            getattr(
                self.sim_cfg,
                "structural_counterfactual_action_response_reporting_enable",
                False,
            )
        )
        cf_aggr_np = np.empty((n,), dtype=np.float32) if counterfactual_response_enabled else None
        cf_temp_np = np.empty((n,), dtype=np.float32) if counterfactual_response_enabled else None
        cf_gain_np = np.empty((n,), dtype=np.float32) if counterfactual_response_enabled else None
        cf_status = ["reporting_disabled"] * n if counterfactual_response_enabled else None

        # Partition MLP vs GRU in the same pass
        mlp_idx: List[int] = []
        gru_idx: List[int] = []

        # Slots/tokens for safe caching of GPU index tensors
        mlp_slots_list: List[int] = []
        mlp_tokens_list: List[int] = []
        gru_slots_list: List[int] = []
        gru_tokens_list: List[int] = []

        cxxrewrite4_policy_heads = cxxrewrite4_prepare_policy_materialization(
            self,
            alive=alive,
            agent_ids_np=agent_ids_np,
            e_norm=e_norm,
            h_norm=h_norm,
            stage="actionsK",
        )
        cxxrewrite4_use_policy_heads = bool(
            cxxrewrite4_policy_heads is not None and cxxrewrite4_policy_heads.get("use_resident_policy_heads")
        )
        cxxrewrite4_use_drives = bool(
            cxxrewrite4_policy_heads is not None and cxxrewrite4_policy_heads.get("drives_np") is not None
        )
        cxxrewrite4_slots = cxxrewrite4_policy_heads.get("slots") if cxxrewrite4_policy_heads is not None else None
        cxxrewrite4_tokens = cxxrewrite4_policy_heads.get("tokens") if cxxrewrite4_policy_heads is not None else None
        if cxxrewrite4_use_policy_heads:
            energy_norm[:] = np.asarray(cxxrewrite4_policy_heads["energy_norm"], dtype=np.float32)
            health_norm[:] = np.asarray(cxxrewrite4_policy_heads["health_norm"], dtype=np.float32)
        if cxxrewrite4_use_drives:
            drives_np[:, :] = np.asarray(cxxrewrite4_policy_heads["drives_np"], dtype=np.float32)

        for i, a in enumerate(alive):
            if not cxxrewrite4_use_policy_heads:
                energy_norm[i] = float(a.energy) / e_norm
                health_norm[i] = float(a.health) / h_norm

            g = getattr(a, "genome", None)
            if g is not None:
                _scalars = (
                    self.resolve_developmental_effective_scalars_for_agent(a)
                    if hasattr(self, "resolve_developmental_effective_scalars_for_agent")
                    else None
                )
                if _scalars is not None:
                    aggr_np[i] = float(_scalars[0])
                    temp_np[i] = float(_scalars[1])
                else:
                    _dev = self.resolve_developmental_expression_for_agent(a)
                    _eff = _dev.get("effective", {}) if isinstance(_dev, dict) else {}
                    aggr_np[i] = float(_eff.get("aggression_bias", getattr(g, "aggression_bias", 0.0)))
                    temp_np[i] = float(_eff.get("action_temperature", getattr(g, "action_temperature", 1.0)))
                try:
                    if _scalars is not None:
                        gain_np[i] = float(_scalars[2])
                    else:
                        gain_np[i] = float(_eff.get("drive_logit_gain", getattr(g, "drive_logit_gain", 1.0)))
                except Exception:
                    gain_np[i] = 1.0
            else:
                aggr_np[i] = 0.0
                temp_np[i] = 1.0
                gain_np[i] = 1.0
            if counterfactual_response_enabled:
                cf_aggr, cf_temp, cf_gain, cf_reason = counterfactual_policy_scalars_for_agent(
                    self,
                    a,
                    actual_aggression_bias=float(aggr_np[i]),
                    actual_action_temperature=float(temp_np[i]),
                    actual_drive_logit_gain=float(gain_np[i]),
                )
                cf_aggr_np[i] = np.float32(cf_aggr)  # type: ignore[index]
                cf_temp_np[i] = np.float32(cf_temp)  # type: ignore[index]
                cf_gain_np[i] = np.float32(cf_gain)  # type: ignore[index]
                cf_status[i] = str(cf_reason)  # type: ignore[index]

            if not cxxrewrite4_use_drives:
                # Runner-level drive arrays (fast path)
                try:
                    if hasattr(self, "_sync_drive_slot"):
                        self._sync_drive_slot(a)
                    slot = int(getattr(a, "slot", -1))
                    if slot >= 0 and hasattr(self, "_drive_hunger") and self._drive_hunger is not None:
                        drives_np[i, 0] = float(self._drive_hunger[slot])
                        drives_np[i, 1] = float(self._drive_pain[slot])
                        drives_np[i, 2] = float(self._drive_fatigue[slot])
                        drives_np[i, 3] = float(self._drive_fear[slot])
                except Exception:
                    pass

            if cxxrewrite4_slots is not None:
                slot_i = int(cxxrewrite4_slots[i])
            else:
                slot_i = int(getattr(a, "slot", -1))
            if cxxrewrite4_tokens is not None:
                token_i = int(cxxrewrite4_tokens[i])
            else:
                token_i = int(getattr(a, "slot_token", 0))

            # Phase 3 determinism:
            # Same partition rule as `_batched_actions` (above). Using genome.use_gru
            # keeps fresh and resume runs aligned even when policy objects are absent.
            is_gru = False
            try:
                g = getattr(a, "genome", None)
                if g is not None and hasattr(g, "use_gru"):
                    is_gru = bool(getattr(g, "use_gru", False))
                else:
                    is_gru = bool(self._is_gru_policy(getattr(a, "policy", None)))
            except Exception:
                is_gru = bool(self._is_gru_policy(getattr(a, "policy", None)))

            if is_gru:
                gru_idx.append(i)
                gru_slots_list.append(slot_i)
                gru_tokens_list.append(token_i)
            else:
                mlp_idx.append(i)
                mlp_slots_list.append(slot_i)
                mlp_tokens_list.append(token_i)

        self._apply_a3_observation_coupling(alive, local, energy_norm, health_norm)
        obs_np[:, :local_size] = local
        obs_np[:, local_size] = energy_norm
        obs_np[:, local_size + 1] = health_norm
        if habacclim_observation_enabled(self.env_cfg):
            obs_np[:, habacclim_observation_scalar_index(self.env_cfg)] = np.asarray(
                [float(getattr(a, "habitat_acclimation_state", 0.0) or 0.0) for a in alive],
                dtype=np.float32,
            )

        if buckets is not None:
            buckets["obs_build"] = buckets.get("obs_build", 0.0) + (time.perf_counter() - t_obs0)

        # -----------------------------
        # Per-tick index cache (GPU)
        # -----------------------------
        tick_cache = getattr(self, "_policy_tick_cache", None)
        if tick_cache is None:
            tick_cache = {}
            setattr(self, "_policy_tick_cache", tick_cache)

        alive_ids_key = tuple(int(a.agent_id) for a in alive)
        key = (
            alive_ids_key,
            tuple(mlp_idx),
            tuple(gru_idx),
            tuple(mlp_slots_list),
            tuple(mlp_tokens_list),
            tuple(gru_slots_list),
            tuple(gru_tokens_list),
            str(self.device),
            str(getattr(self, "_bank_device", getattr(self, "bank_device", self.device))),
        )
        cache_hit = tick_cache.get("key") == key

        # Phase 3 determinism: policy execution probe (gen1/tick0)
        self._eiel_policy_exec_probe(
            "actionsK.entry",
            alive=alive,
            mlp_idx=mlp_idx,
            gru_idx=gru_idx,
            k=int(k),
            cache_hit=bool(cache_hit),
            mlp_slots=mlp_slots_list,
            mlp_tokens=mlp_tokens_list,
            gru_slots=gru_slots_list,
            gru_tokens=gru_tokens_list,
        )

        # Neuromodulation weights (safe defaults)
        w_hunger_gather = float(getattr(self.sim_cfg, "drive_w_hunger_gather", 1.2))
        w_pain_rest = float(getattr(self.sim_cfg, "drive_w_pain_rest", 0.9))
        w_fatigue_rest = float(getattr(self.sim_cfg, "drive_w_fatigue_rest", 1.0))
        w_pain_attack = float(getattr(self.sim_cfg, "drive_w_pain_attack", 1.2))
        w_fear_attack = float(getattr(self.sim_cfg, "drive_w_fear_attack", 1.4))

        cpu_numpy_actions = self._batched_actions_k_cpu_numpy(
            alive=alive,
            k=int(k),
            n=int(n),
            obs_np=obs_np,
            aggr_np=aggr_np,
            temp_np=temp_np,
            drives_np=drives_np,
            gain_np=gain_np,
            mlp_idx=mlp_idx,
            gru_idx=gru_idx,
            mlp_slots_list=mlp_slots_list,
            gru_slots_list=gru_slots_list,
            buckets=buckets,
            w_hunger_gather=w_hunger_gather,
            w_pain_rest=w_pain_rest,
            w_fatigue_rest=w_fatigue_rest,
            w_pain_attack=w_pain_attack,
            w_fear_attack=w_fear_attack,
        )
        if cpu_numpy_actions is not None:
            if buckets is not None:
                buckets["policy_cpu_numpy_forward_count"] = buckets.get("policy_cpu_numpy_forward_count", 0.0) + 1.0
            return cpu_numpy_actions

        # -----------------------------
        # Staging buffers (grow-only)
        # -----------------------------
        buf = getattr(self, "_policy_buf", None)
        if buf is None:
            buf = {}
            setattr(self, "_policy_buf", buf)

        t_stage0 = time.perf_counter()
        packed_k_staging = bool(getattr(self.sim_cfg, "perf_policy_drive_packed_k_staging_enable", False))

        if packed_k_staging:
            packed_cols = obs_size + 1 + 1 + 4 + 1
            cap = int(buf.get("packed_k_cap", 0) or 0)
            packed_cpu = buf.get("packed_k_cpu")
            packed_gpu = buf.get("packed_k_gpu")
            if (
                packed_cpu is None
                or packed_gpu is None
                or cap < n
                or int(buf.get("packed_k_cols", 0) or 0) != packed_cols
                or str(getattr(packed_gpu, "device", "")) != str(self.device)
            ):
                new_cap = max(n, int(max(128, cap * 2)))
                buf["packed_k_cap"] = new_cap
                buf["packed_k_cols"] = packed_cols
                pin = bool(self.device.type == "cuda")
                buf["packed_k_cpu"] = torch.empty(
                    (new_cap, packed_cols),
                    device="cpu",
                    dtype=torch.float32,
                    pin_memory=pin,
                )
                buf["packed_k_gpu"] = torch.empty((new_cap, packed_cols), dtype=torch.float32, device=self.device)
                buf["packed_k_cpu_np"] = buf["packed_k_cpu"].numpy()

            packed_cpu_np = buf["packed_k_cpu_np"]
            packed_cpu_np[:n, :obs_size] = obs_np
            c = obs_size
            packed_cpu_np[:n, c] = aggr_np
            c += 1
            packed_cpu_np[:n, c] = temp_np
            c += 1
            packed_cpu_np[:n, c : c + 4] = drives_np
            c += 4
            packed_cpu_np[:n, c] = gain_np

            packed_gpu = buf["packed_k_gpu"]
            t_copy0 = time.perf_counter()
            # Policy prep includes obs build + packing into CPU staging buffers.
            if buckets is not None:
                buckets["policy_prep_ms"] = buckets.get("policy_prep_ms", 0.0) + (t_copy0 - t_policy0)
            packed_gpu[:n].copy_(buf["packed_k_cpu"][:n], non_blocking=True)
            if buckets is not None:
                buckets["h2d_copy_ms"] = buckets.get("h2d_copy_ms", 0.0) + (time.perf_counter() - t_copy0)

            x_all = packed_gpu[:n, :obs_size]
            aggr_all = packed_gpu[:n, obs_size]
            temp_all = packed_gpu[:n, obs_size + 1]
            drives_all = packed_gpu[:n, obs_size + 2 : obs_size + 6]
            gain_all = packed_gpu[:n, obs_size + 6]
        else:
            if (buf.get("x_cpu") is None) or (buf["x_cpu"].shape[0] < n) or (buf.get("x_cpu").shape[1] != obs_size):
                buf["x_cpu"] = np.empty((n, obs_size), dtype=np.float32)
                buf["aggr_cpu"] = np.empty((n,), dtype=np.float32)
                buf["temp_cpu"] = np.empty((n,), dtype=np.float32)
                buf["drives_cpu"] = np.empty((n, 4), dtype=np.float32)
                buf["gain_cpu"] = np.empty((n,), dtype=np.float32)

            buf["x_cpu"][:n, :] = obs_np
            buf["aggr_cpu"][:n] = aggr_np
            buf["temp_cpu"][:n] = temp_np
            buf["drives_cpu"][:n, :] = drives_np
            buf["gain_cpu"][:n] = gain_np

            if (buf.get("x_all") is None) or (buf["x_all"].shape[0] < n) or (buf["x_all"].shape[1] != obs_size) or (str(buf["x_all"].device) != str(self.device)):
                buf["x_all"] = torch.empty((n, obs_size), dtype=torch.float32, device=self.device)
                buf["aggr_all"] = torch.empty((n,), dtype=torch.float32, device=self.device)
                buf["temp_all"] = torch.empty((n,), dtype=torch.float32, device=self.device)
                buf["drives_all"] = torch.empty((n, 4), dtype=torch.float32, device=self.device)
                buf["gain_all"] = torch.empty((n,), dtype=torch.float32, device=self.device)

            x_all = buf["x_all"][:n]
            aggr_all = buf["aggr_all"][:n]
            temp_all = buf["temp_all"][:n]
            drives_all = buf["drives_all"][:n]
            gain_all = buf["gain_all"][:n]

            t_copy0 = time.perf_counter()
            # Policy prep includes obs build + packing into CPU staging buffers.
            if buckets is not None:
                buckets["policy_prep_ms"] = buckets.get("policy_prep_ms", 0.0) + (t_copy0 - t_policy0)
            x_all.copy_(torch.from_numpy(buf["x_cpu"][:n, :]), non_blocking=True)
            aggr_all.copy_(torch.from_numpy(buf["aggr_cpu"][:n]), non_blocking=True)
            temp_all.copy_(torch.from_numpy(buf["temp_cpu"][:n]), non_blocking=True)
            drives_all.copy_(torch.from_numpy(buf["drives_cpu"][:n, :]), non_blocking=True)
            gain_all.copy_(torch.from_numpy(buf["gain_cpu"][:n]), non_blocking=True)
            if buckets is not None:
                buckets["h2d_copy_ms"] = buckets.get("h2d_copy_ms", 0.0) + (time.perf_counter() - t_copy0)

        gain_all = gain_all.clamp(min=0.0)
        cf_aggr_all = cf_temp_all = cf_gain_all = None
        if counterfactual_response_enabled and cf_aggr_np is not None and cf_temp_np is not None and cf_gain_np is not None:
            cf_aggr_all = torch.from_numpy(cf_aggr_np).to(self.device, non_blocking=True)
            cf_temp_all = torch.from_numpy(cf_temp_np).to(self.device, non_blocking=True)
            cf_gain_all = torch.from_numpy(cf_gain_np).to(self.device, non_blocking=True).clamp(min=0.0)

        # Grow-only K-actions tensor (depends on k)
        if (buf.get("actions_all_k") is None) or (buf["actions_all_k"].shape[0] < n) or (buf["actions_all_k"].shape[1] < k) or (str(buf["actions_all_k"].device) != str(self.device)):
            buf["actions_all_k"] = torch.empty((n, k), dtype=torch.int64, device=self.device)

        actions_all_k = buf["actions_all_k"][:n, :k]

        t_fwd0 = time.perf_counter()

        if buckets is not None:
            buckets["tensor_stage"] = buckets.get("tensor_stage", 0.0) + (time.perf_counter() - t_stage0)

        # -----------------------------
        # Helpers: categorical sampling via Gumbel-Max (k samples)
        # -----------------------------
        def _policy_uniform(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            """Deterministic uniform(0,1) using RNGManager policy_np.

            Why:
            - P0-A determinism enforcement forbids torch.rand/torch.rand_like in
              simulation code paths because it draws from torch's global RNG.
            - We route sampling through the centralized RNGManager (NumPy stream)
              and then transfer to torch.
            """
            # NumPy draws are deterministic under RNGManager stream discipline.
            n = 1
            for d in shape:
                n *= int(d)
            arr = self.rng_manager.policy_np.random(int(n), dtype=np.float32).reshape(shape)
            t = torch.from_numpy(arr).to(device=device, non_blocking=True)
            if t.dtype != dtype:
                t = t.to(dtype=dtype)
            return t

        def _sample_k_actions_from_logits(logits: torch.Tensor, k_samples: int) -> tuple[torch.Tensor, torch.Tensor | None]:
            """Sample k actions per row from Categorical(softmax(logits))."""
            if getattr(self.sim_cfg, "policy_deterministic_actions", False):
                a = torch.argmax(logits, dim=1)
                return a.unsqueeze(1).expand(-1, k_samples), None
            B, A = logits.shape
            u = _policy_uniform((int(B), int(k_samples), int(A)), device=logits.device, dtype=logits.dtype)
            g = -torch.log(-torch.log(u.clamp(min=1e-6, max=1.0 - 1e-6)))
            return torch.argmax(logits.unsqueeze(1) + g, dim=2), g

        def _sample_actions_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
            if getattr(self.sim_cfg, "policy_deterministic_actions", False):
                return torch.argmax(logits, dim=1), None
            u = _policy_uniform(tuple(int(x) for x in logits.shape), device=logits.device, dtype=logits.dtype)
            g = -torch.log(-torch.log(u.clamp(min=1e-6, max=1.0 - 1e-6)))
            return torch.argmax(logits + g, dim=1), g

        # -----------------------------
        # MLP fast path
        # -----------------------------
        if mlp_idx:
            t0 = time.perf_counter()
            mlp_agents = [alive[i] for i in mlp_idx]
            self._sync_bank_for_agents(mlp_agents, buckets=buckets)

            if cache_hit and "mlp_rows" in tick_cache and "mlp_slots" in tick_cache:
                mlp_idx_np = tick_cache["mlp_idx_np"]
                mlp_rows = tick_cache["mlp_rows"]
                slots = tick_cache["mlp_slots"]
            else:
                mlp_idx_np = np.asarray(mlp_idx, dtype=np.int64)
                mlp_rows = torch.from_numpy(mlp_idx_np).to(self.device, non_blocking=True)
                slots_np = np.asarray(mlp_slots_list, dtype=np.int64)
                slots = torch.from_numpy(slots_np).to(device=getattr(self, '_bank_device', self.device), dtype=torch.long, non_blocking=True)
                tick_cache["mlp_idx_np"] = mlp_idx_np
                tick_cache["mlp_rows"] = mlp_rows
                tick_cache["mlp_slots"] = slots

            x = x_all.index_select(0, mlp_rows)
            aggr = aggr_all.index_select(0, mlp_rows)
            temps = temp_all.index_select(0, mlp_rows)
            d = drives_all.index_select(0, mlp_rows)
            gain = gain_all.index_select(0, mlp_rows)
            cf_aggr = cf_temps = cf_gain = None
            if counterfactual_response_enabled and cf_aggr_all is not None and cf_temp_all is not None and cf_gain_all is not None:
                cf_aggr = cf_aggr_all.index_select(0, mlp_rows)
                cf_temps = cf_temp_all.index_select(0, mlp_rows)
                cf_gain = cf_gain_all.index_select(0, mlp_rows)

            # ---- EIEL debug: slot index diagnostics (resume/gen0) ----
            # Enabled by env var: EIEL_DEBUG_POLICY_SLOTS=1
            dbg_slots = bool(int(os.getenv("EIEL_DEBUG_POLICY_SLOTS", "0")))
            if dbg_slots and getattr(self, "_mlp_pack", None) is not None:
                try:
                    cap0 = int(self._mlp_pack.shape[0])
                    slots_cpu = slots.detach().to("cpu")
                    if int(slots_cpu.numel()) > 0 and cap0 > 0:
                        smin0 = int(slots_cpu.min().item())
                        smax0 = int(slots_cpu.max().item())
                        bad_pos0 = int((slots_cpu >= cap0).sum().item())
                        bad_neg0 = int((slots_cpu < -cap0).sum().item())
                        sample0 = slots_cpu[: min(16, int(slots_cpu.numel()))].tolist()
                        self._policy_slots_log(f"[EIEL_DEBUG_POLICY_SLOTS pre] cap={cap0} smin={smin0} smax={smax0} bad_pos={bad_pos0} bad_neg={bad_neg0} sample={sample0}")
                        need = cap0
                        if smax0 >= cap0:
                            need = max(need, smax0 + 1)
                        if smin0 < -cap0:
                            need = max(need, abs(smin0) + 1)
                        if need > cap0 and hasattr(self, "_grow_mlp_bank"):
                            self._grow_mlp_bank(max_slots=int(need))
                            cap1 = int(self._mlp_pack.shape[0])
                            bad_pos1 = int((slots_cpu >= cap1).sum().item()) if cap1 > 0 else -1
                            bad_neg1 = int((slots_cpu < -cap1).sum().item()) if cap1 > 0 else -1
                            self._policy_slots_log(f"[EIEL_DEBUG_POLICY_SLOTS post] cap={cap1} need={need} bad_pos={bad_pos1} bad_neg={bad_neg1}")
                        # Optional fail-fast with clear message (default ON when debug enabled)
                        failfast = bool(int(os.getenv("EIEL_DEBUG_POLICY_SLOTS_FAILFAST", "1")))
                        cap_chk = int(self._mlp_pack.shape[0])
                        if failfast and cap_chk > 0:
                            if smax0 >= cap_chk or smin0 < -cap_chk:
                                raise RuntimeError(
                                    f"Invalid slot indices for MLP bank: cap={cap_chk} smin={smin0} smax={smax0} "
                                    f"(bad_pos={int((slots_cpu >= cap_chk).sum().item())}, bad_neg={int((slots_cpu < -cap_chk).sum().item())})"
                                )
                except Exception as _e:
                    # Re-raise under debug to stop at the first bad tick with context.
                    raise
            # ---- end EIEL debug ----
            self._eiel_bank_device_sanity_probe(stage="bank_select.mlp_pack", bank_name="mlp_pack", bank=getattr(self, '_mlp_pack', None), slots=slots)
            pack = self._mlp_pack.index_select(0, slots)
            pack = self._eiel_to_compute(pack, name="mlp_pack_rows", stage="bank_select.mlp_pack")
            H = int(self._hidden)
            O = int(self._obs_size)
            A = int(self._out)
            off = 0
            W1 = pack[:, off:off + (H * O)].view(-1, H, O); off += (H * O)
            b1 = pack[:, off:off + H].view(-1, H); off += H
            W2 = pack[:, off:off + (H * H)].view(-1, H, H); off += (H * H)
            b2 = pack[:, off:off + H].view(-1, H); off += H
            W3 = pack[:, off:off + (A * H)].view(-1, A, H); off += (A * H)
            b3 = pack[:, off:off + A].view(-1, A)

            logits = forward_batched_mlp(
                x,
                W1,
                b1,
                W2,
                b2,
                W3,
                b3,
                attack_bias=aggr,
                action_temperature=temps,
            )

            if logits.numel() != 0:
                logits[:, ACT_GATHER] = logits[:, ACT_GATHER] + gain * (w_hunger_gather * d[:, 0])
                logits[:, ACT_REST] = logits[:, ACT_REST] + gain * (w_pain_rest * d[:, 1] + w_fatigue_rest * d[:, 2])
                logits[:, ACT_ATTACK] = logits[:, ACT_ATTACK] - gain * (w_pain_attack * d[:, 1] + w_fear_attack * d[:, 3])

            self._eiel_policy_exec_probe(
                "actionsK.pre_sample_mlp",
                alive=alive,
                mlp_idx=mlp_idx,
                gru_idx=gru_idx,
                k=int(k),
                cache_hit=bool(cache_hit),
                extra={"rand_shape": [int(logits.shape[0]), int(k), int(logits.shape[1])]},
                logits=logits,
            )
            opportunity_vectors, valid_family_mask, valid_action_mask = _polarch12_binary_opportunity_inputs(
                self,
                alive,
                list(mlp_idx),
                device=logits.device,
                dtype=logits.dtype,
            )
            choice_logits, authority_summary = apply_authority_to_logits_torch(
                logits,
                alive=alive,
                row_indices=list(mlp_idx),
                sim_cfg=self.sim_cfg,
                opportunity_vectors=opportunity_vectors,
                valid_family_mask=valid_family_mask,
                valid_action_mask=valid_action_mask,
            )
            _polarch6_merge_authority_summary(self, authority_summary, policy_kind="mlp_stride")
            record_policy_margins(self, alive=alive, row_indices=list(mlp_idx), logits=choice_logits)
            actions_mlp_k, action_noise_k = _sample_k_actions_from_logits(choice_logits, k)
            record_family_head_authority_choice_margin(
                self,
                alive=alive,
                row_indices=list(mlp_idx),
                neutral_logits=logits,
                directional_logits=choice_logits,
                actions=actions_mlp_k,
                sample_noise=action_noise_k,
                policy_kind="mlp",
                action_source_class="neural_policy_stride",
            )
            record_policy_action_family_decisions(
                self,
                alive=alive,
                row_indices=list(mlp_idx),
                logits=choice_logits,
                actions=actions_mlp_k,
                sample_noise=action_noise_k,
                policy_kind="mlp",
                drives=d,
                action_temperature=temps,
                action_source_class="neural_policy_stride",
            )
            if counterfactual_response_enabled and cf_aggr is not None and cf_temps is not None and cf_gain is not None:
                cf_logits = forward_batched_mlp(
                    x,
                    W1,
                    b1,
                    W2,
                    b2,
                    W3,
                    b3,
                    attack_bias=cf_aggr,
                    action_temperature=cf_temps,
                )
                if cf_logits.numel() != 0:
                    cf_logits[:, ACT_GATHER] = cf_logits[:, ACT_GATHER] + cf_gain * (w_hunger_gather * d[:, 0])
                    cf_logits[:, ACT_REST] = cf_logits[:, ACT_REST] + cf_gain * (w_pain_rest * d[:, 1] + w_fatigue_rest * d[:, 2])
                    cf_logits[:, ACT_ATTACK] = cf_logits[:, ACT_ATTACK] - cf_gain * (w_pain_attack * d[:, 1] + w_fear_attack * d[:, 3])
                record_counterfactual_action_response(
                    self,
                    alive=alive,
                    row_indices=list(mlp_idx),
                    actual_logits=choice_logits,
                    counterfactual_logits=cf_logits,
                    actual_actions=actions_mlp_k[:, 0],
                    sample_noise=action_noise_k[:, 0, :] if action_noise_k is not None else None,
                    status_by_row=[cf_status[i] for i in mlp_idx] if cf_status is not None else None,
                    policy_kind="mlp",
                    action_source_class="neural_policy_stride",
                )
            self._eiel_policy_exec_probe(
                "actionsK.post_sample_mlp",
                alive=alive,
                mlp_idx=mlp_idx,
                gru_idx=gru_idx,
                k=int(k),
                cache_hit=bool(cache_hit),
            )
            actions_all_k.index_copy_(0, mlp_rows, actions_mlp_k)

            if buckets is not None:
                buckets["policy_forward_mlp"] = buckets.get("policy_forward_mlp", 0.0) + (time.perf_counter() - t0)

        # -----------------------------
        # GRU stateful path with masking (one step, then repeat)
        # -----------------------------
        if gru_idx:
            t0 = time.perf_counter()

            gru_agents = [alive[i] for i in gru_idx]
            self._sync_gru_bank_for_agents(gru_agents, buckets=buckets)

            if cache_hit and "gru_rows" in tick_cache and "gru_slots" in tick_cache:
                gru_idx_np = tick_cache["gru_idx_np"]
                gru_rows = tick_cache["gru_rows"]
                slots = tick_cache["gru_slots"]
            else:
                gru_idx_np = np.asarray(gru_idx, dtype=np.int64)
                gru_rows = torch.from_numpy(gru_idx_np).to(self.device, non_blocking=True)
                slots_np = np.asarray(gru_slots_list, dtype=np.int64)
                slots = torch.from_numpy(slots_np).to(device=getattr(self, '_bank_device', self.device), dtype=torch.long, non_blocking=True)
                tick_cache["gru_idx_np"] = gru_idx_np
                tick_cache["gru_rows"] = gru_rows
                tick_cache["gru_slots"] = slots
            mem_dims_np = np.asarray(
                [int(effective_memory_dim_for_genome(getattr(a, "genome", None), sim_cfg=self.sim_cfg)) for a in gru_agents],
                dtype=np.int64,
            )
            mem_dims = torch.from_numpy(mem_dims_np).to(self.device, non_blocking=True)

            x = x_all.index_select(0, gru_rows)
            aggr = aggr_all.index_select(0, gru_rows)
            temps = temp_all.index_select(0, gru_rows)
            d = drives_all.index_select(0, gru_rows)
            gain = gain_all.index_select(0, gru_rows)
            cf_aggr = cf_temps = cf_gain = None
            if counterfactual_response_enabled and cf_aggr_all is not None and cf_temp_all is not None and cf_gain_all is not None:
                cf_aggr = cf_aggr_all.index_select(0, gru_rows)
                cf_temps = cf_temp_all.index_select(0, gru_rows)
                cf_gain = cf_gain_all.index_select(0, gru_rows)

            self._eiel_bank_device_sanity_probe(stage="bank_select.gru_hmem", bank_name="Hmem", bank=getattr(self, '_Hmem', None), slots=slots)
            h_prev = self._Hmem.index_select(0, slots)
            h_prev = self._eiel_to_compute(h_prev, name="gru_h_prev", stage="bank_select.gru_hmem")
            Hm = int(h_prev.shape[1])
            mem_dims = torch.clamp(mem_dims, 0, Hm)

            idx = tick_cache.get("mask_idx")
            if idx is None or int(idx.shape[1]) != Hm:
                idx = torch.arange(Hm, device=self.device, dtype=torch.long).unsqueeze(0)
                tick_cache["mask_idx"] = idx
            mask = (idx < mem_dims.unsqueeze(1)).to(h_prev.dtype)
            h_prev = h_prev * mask
            reset_count = 0
            if bool(memory_ablation_flags(self.sim_cfg).get("reset_each_tick", False)):
                reset_count = int(h_prev.shape[0])
                h_prev = torch.zeros_like(h_prev)

            self._eiel_bank_device_sanity_probe(stage="bank_select.gru_pack", bank_name="gru_pack", bank=getattr(self, '_gru_pack', None), slots=slots)
            pack = self._gru_pack.index_select(0, slots)
            pack = self._eiel_to_compute(pack, name="gru_pack_rows", stage="bank_select.gru_pack")
            O = int(self._obs_size)
            A = int(self._out)
            Hm = int(self._gru_hidden)
            off = 0
            encW = pack[:, off:off + (Hm * O)].view(-1, Hm, O); off += (Hm * O)
            encb = pack[:, off:off + Hm].view(-1, Hm); off += Hm
            Wih = pack[:, off:off + (3 * Hm * Hm)].view(-1, 3 * Hm, Hm); off += (3 * Hm * Hm)
            bih = pack[:, off:off + (3 * Hm)].view(-1, 3 * Hm); off += (3 * Hm)
            Whh = pack[:, off:off + (3 * Hm * Hm)].view(-1, 3 * Hm, Hm); off += (3 * Hm * Hm)
            bhh = pack[:, off:off + (3 * Hm)].view(-1, 3 * Hm); off += (3 * Hm)
            headW = pack[:, off:off + (A * Hm)].view(-1, A, Hm); off += (A * Hm)
            headb = pack[:, off:off + A].view(-1, A)

            logits, h_next = forward_batched_gru(
                x=x,
                h_prev=h_prev,
                encW=encW,
                encb=encb,
                Wih=Wih,
                bih=bih,
                Whh=Whh,
                bhh=bhh,
                headW=headW,
                headb=headb,
                attack_bias=aggr,
                action_temperature=temps,
            )

            if logits.numel() != 0:
                logits[:, ACT_GATHER] = logits[:, ACT_GATHER] + gain * (w_hunger_gather * d[:, 0])
                logits[:, ACT_REST] = logits[:, ACT_REST] + gain * (w_pain_rest * d[:, 1] + w_fatigue_rest * d[:, 2])
                logits[:, ACT_ATTACK] = logits[:, ACT_ATTACK] - gain * (w_pain_attack * d[:, 1] + w_fear_attack * d[:, 3])

            h_next = h_next * mask
            self._record_memory_runtime_telemetry(
                h_prev=h_prev,
                h_next=h_next,
                mask=mask,
                reset_count=reset_count,
            )
            h_next_bank = self._eiel_to_bank(h_next, like=self._Hmem, name="gru_h_next", stage="bank_update.gru_hmem")
            self._Hmem.index_copy_(0, slots, h_next_bank)

            self._eiel_policy_exec_probe(
                "actionsK.pre_sample_gru",
                alive=alive,
                mlp_idx=mlp_idx,
                gru_idx=gru_idx,
                k=int(k),
                cache_hit=bool(cache_hit),
                extra={"rand_shape": [int(logits.shape[0]), int(logits.shape[1])]},
                logits=logits,
            )
            opportunity_vectors, valid_family_mask, valid_action_mask = _polarch12_binary_opportunity_inputs(
                self,
                alive,
                list(gru_idx),
                device=logits.device,
                dtype=logits.dtype,
            )
            choice_logits, authority_summary = apply_authority_to_logits_torch(
                logits,
                alive=alive,
                row_indices=list(gru_idx),
                sim_cfg=self.sim_cfg,
                opportunity_vectors=opportunity_vectors,
                valid_family_mask=valid_family_mask,
                valid_action_mask=valid_action_mask,
            )
            _polarch6_merge_authority_summary(self, authority_summary, policy_kind="gru_stride")
            record_policy_margins(self, alive=alive, row_indices=list(gru_idx), logits=choice_logits)
            actions_gru, action_noise = _sample_actions_from_logits(choice_logits)  # (B,)
            actions_gru_rep = actions_gru.view(-1, 1).repeat(1, k)
            record_family_head_authority_choice_margin(
                self,
                alive=alive,
                row_indices=list(gru_idx),
                neutral_logits=logits,
                directional_logits=choice_logits,
                actions=actions_gru_rep,
                sample_noise=action_noise,
                policy_kind="gru",
                action_source_class="neural_policy_stride",
            )
            record_policy_action_family_decisions(
                self,
                alive=alive,
                row_indices=list(gru_idx),
                logits=choice_logits,
                actions=actions_gru_rep,
                sample_noise=action_noise,
                policy_kind="gru",
                drives=d,
                action_temperature=temps,
                action_source_class="neural_policy_stride",
            )
            if counterfactual_response_enabled and cf_aggr is not None and cf_temps is not None and cf_gain is not None:
                cf_logits, _cf_h_next = forward_batched_gru(
                    x=x,
                    h_prev=h_prev,
                    encW=encW,
                    encb=encb,
                    Wih=Wih,
                    bih=bih,
                    Whh=Whh,
                    bhh=bhh,
                    headW=headW,
                    headb=headb,
                    attack_bias=cf_aggr,
                    action_temperature=cf_temps,
                )
                if cf_logits.numel() != 0:
                    cf_logits[:, ACT_GATHER] = cf_logits[:, ACT_GATHER] + cf_gain * (w_hunger_gather * d[:, 0])
                    cf_logits[:, ACT_REST] = cf_logits[:, ACT_REST] + cf_gain * (w_pain_rest * d[:, 1] + w_fatigue_rest * d[:, 2])
                    cf_logits[:, ACT_ATTACK] = cf_logits[:, ACT_ATTACK] - cf_gain * (w_pain_attack * d[:, 1] + w_fear_attack * d[:, 3])
                record_counterfactual_action_response(
                    self,
                    alive=alive,
                    row_indices=list(gru_idx),
                    actual_logits=choice_logits,
                    counterfactual_logits=cf_logits,
                    actual_actions=actions_gru,
                    sample_noise=action_noise,
                    status_by_row=[cf_status[i] for i in gru_idx] if cf_status is not None else None,
                    policy_kind="gru",
                    action_source_class="neural_policy_stride",
                )
            self._eiel_policy_exec_probe(
                "actionsK.post_sample_gru",
                alive=alive,
                mlp_idx=mlp_idx,
                gru_idx=gru_idx,
                k=int(k),
                cache_hit=bool(cache_hit),
            )
            actions_all_k.index_copy_(0, gru_rows, actions_gru_rep)

            if buckets is not None:
                buckets["policy_forward_gru"] = buckets.get("policy_forward_gru", 0.0) + (time.perf_counter() - t0)

        tick_cache["key"] = key

        t_output0 = time.perf_counter()
        actions_np_k = actions_all_k.to("cpu").numpy().astype(np.int32, copy=False)
        if buckets is not None:
            buckets["policy_output_transfer"] = buckets.get("policy_output_transfer", 0.0) + (
                time.perf_counter() - t_output0
            )
        self._eiel_policy_exec_probe(
            "actionsK.exit",
            alive=alive,
            mlp_idx=mlp_idx,
            gru_idx=gru_idx,
            k=int(k),
            cache_hit=bool(cache_hit),
            actions_out=actions_np_k,
        )
        return actions_np_k



    @torch.no_grad()
    def _probe_action_probs(self, alive) -> np.ndarray:
        """Compute a fixed policy "probe signature" for each alive agent.

        The probe is a deterministic synthetic observation (all zeros) that does NOT
        touch the environment. This is used for Phase 4 identity drift pressure:

            baseline_probs = softmax(logits(probe_obs)) captured at episode-0 end
            drift = mean(abs(probs_now - baseline_probs))

        Important invariants:
        - Does NOT update GRU hidden state (we discard h_next).
        - Does NOT apply drive neuromodulation (probe is intended to reflect internal
          organization, not changing internal drives).
        - Deterministic for a given parameter bank + current GRU hidden state.

        Args:
            alive: List of alive agents, in the same ordering used for per-tick work.

        Returns:
            probs_np: float32 array of shape (N, A) aligned to `alive`.
        """
        n = len(alive)
        if n == 0:
            return np.zeros((0, int(getattr(self, "_out", 7) or 7)), dtype=np.float32)

        # Fixed synthetic observation. No env reads, no host->device copies.
        obs_size = int(getattr(self, "_obs_size", getattr(self, "obs_size", 0)) or 0)
        if obs_size <= 0:
            # Safe fallback
            obs_size = patch_world_total_observation_size(self.env_cfg)

        A = int(getattr(self, "_out", 7) or 7)

        # Build per-agent gene scalars (CPU, cheap)
        aggr_np = np.empty((n,), dtype=np.float32)
        temp_np = np.empty((n,), dtype=np.float32)

        mlp_idx = []
        gru_idx = []
        mlp_slots_list = []
        gru_slots_list = []

        for i, a in enumerate(alive):
            g = getattr(a, "genome", None)
            if g is not None:
                _scalars = (
                    self.resolve_developmental_effective_scalars_for_agent(a)
                    if hasattr(self, "resolve_developmental_effective_scalars_for_agent")
                    else None
                )
                if _scalars is not None:
                    aggr_np[i] = float(_scalars[0])
                    temp_np[i] = float(_scalars[1])
                else:
                    _dev = self.resolve_developmental_expression_for_agent(a)
                    _eff = _dev.get("effective", {}) if isinstance(_dev, dict) else {}
                    aggr_np[i] = float(_eff.get("aggression_bias", getattr(g, "aggression_bias", 0.0)))
                    temp_np[i] = float(_eff.get("action_temperature", getattr(g, "action_temperature", 1.0)))
            else:
                aggr_np[i] = 0.0
                temp_np[i] = 1.0

            slot_i = int(getattr(a, "slot", -1))
            if self._is_gru_policy(getattr(a, "policy", None)):
                gru_idx.append(i)
                gru_slots_list.append(slot_i)
            else:
                mlp_idx.append(i)
                mlp_slots_list.append(slot_i)

        probs_out = np.zeros((n, A), dtype=np.float32)

        # Common tensors on GPU
        x_all = torch.zeros((n, obs_size), device=self.device, dtype=self._bank_dtype)
        aggr_t = torch.from_numpy(aggr_np).to(self.device, non_blocking=True)
        temp_t = torch.from_numpy(temp_np).to(self.device, non_blocking=True)

        # MLP probe
        if mlp_idx:
            mlp_agents = [alive[i] for i in mlp_idx]
            self._sync_bank_for_agents(mlp_agents, buckets=None)

            mlp_rows = torch.from_numpy(np.asarray(mlp_idx, dtype=np.int64)).to(self.device, non_blocking=True)
            slots = torch.from_numpy(np.asarray(mlp_slots_list, dtype=np.int64)).to(device=getattr(self, '_bank_device', self.device), dtype=torch.long, non_blocking=True)

            x = x_all.index_select(0, mlp_rows)
            aggr = aggr_t.index_select(0, mlp_rows)
            temps = temp_t.index_select(0, mlp_rows)

            self._eiel_bank_device_sanity_probe(stage="bank_select.mlp_pack", bank_name="mlp_pack", bank=getattr(self, '_mlp_pack', None), slots=slots)
            pack = self._mlp_pack.index_select(0, slots)
            pack = self._eiel_to_compute(pack, name="mlp_pack_rows", stage="bank_select.mlp_pack")
            H = int(self._hidden)
            O = int(self._obs_size)
            off = 0
            W1 = pack[:, off: off + H * O].view(-1, H, O); off += H * O
            b1 = pack[:, off: off + H].view(-1, H); off += H
            W2 = pack[:, off: off + H * H].view(-1, H, H); off += H * H
            b2 = pack[:, off: off + H].view(-1, H); off += H
            W3 = pack[:, off: off + A * H].view(-1, A, H); off += A * H
            b3 = pack[:, off: off + A].view(-1, A)

            logits = forward_batched_mlp(
                x=x,
                w1=W1, b1=b1,
                w2=W2, b2=b2,
                w3=W3, b3=b3,
                attack_bias=aggr,
                action_temperature=temps,
            )
            probs = torch.softmax(logits, dim=1).to("cpu").numpy().astype(np.float32, copy=False)
            probs_out[np.asarray(mlp_idx, dtype=np.int32)] = probs

        # GRU probe
        if gru_idx:
            gru_agents = [alive[i] for i in gru_idx]
            self._sync_gru_bank_for_agents(gru_agents, buckets=None)

            gru_rows = torch.from_numpy(np.asarray(gru_idx, dtype=np.int64)).to(self.device, non_blocking=True)
            slots = torch.from_numpy(np.asarray(gru_slots_list, dtype=np.int64)).to(device=getattr(self, '_bank_device', self.device), dtype=torch.long, non_blocking=True)

            x = x_all.index_select(0, gru_rows)
            aggr = aggr_t.index_select(0, gru_rows)
            temps = temp_t.index_select(0, gru_rows)

            self._eiel_bank_device_sanity_probe(stage="bank_select.gru_pack", bank_name="gru_pack", bank=getattr(self, '_gru_pack', None), slots=slots)
            pack = self._gru_pack.index_select(0, slots)
            pack = self._eiel_to_compute(pack, name="gru_pack_rows", stage="bank_select.gru_pack")
            Hm = int(self._gru_hidden)
            O = int(self._obs_size)
            off = 0
            encW = pack[:, off: off + Hm * O].view(-1, Hm, O); off += Hm * O
            encb = pack[:, off: off + Hm].view(-1, Hm); off += Hm
            Wih = pack[:, off: off + (3 * Hm) * Hm].view(-1, 3 * Hm, Hm); off += (3 * Hm) * Hm
            bih = pack[:, off: off + 3 * Hm].view(-1, 3 * Hm); off += 3 * Hm
            Whh = pack[:, off: off + (3 * Hm) * Hm].view(-1, 3 * Hm, Hm); off += (3 * Hm) * Hm
            bhh = pack[:, off: off + 3 * Hm].view(-1, 3 * Hm); off += 3 * Hm
            headW = pack[:, off: off + A * Hm].view(-1, A, Hm); off += A * Hm
            headb = pack[:, off: off + A].view(-1, A)

            self._eiel_bank_device_sanity_probe(stage="bank_select.gru_hmem", bank_name="Hmem", bank=getattr(self, '_Hmem', None), slots=slots)
            h_prev = self._Hmem.index_select(0, slots)
            h_prev = self._eiel_to_compute(h_prev, name="gru_h_prev", stage="bank_select.gru_hmem")

            logits, _h_next = forward_batched_gru(
                x=x,
                h_prev=h_prev,
                encW=encW,
                encb=encb,
                Wih=Wih,
                bih=bih,
                Whh=Whh,
                bhh=bhh,
                headW=headW,
                headb=headb,
                attack_bias=aggr,
                action_temperature=temps,
            )
            probs = torch.softmax(logits, dim=1).to("cpu").numpy().astype(np.float32, copy=False)
            probs_out[np.asarray(gru_idx, dtype=np.int32)] = probs

        return probs_out
