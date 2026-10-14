"""Checkpoint settlement for prediction state disabled by the runtime profile."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import torch


DISABLED_PREDICTION_STATE_FIELDS_V1: Final[tuple[str, ...]] = (
    "adult_pred_err_ema", "adult_pred_err_ema_exact", "episode0_pred_err_count",
    "episode0_pred_err_sum", "episode0_pred_err_sum_exact", "exp_dE", "exp_dInj",
    "late_pred_err_count", "late_pred_err_sum", "late_pred_err_sum_exact",
    "pred_err_count", "pred_err_sum", "pred_err_sum_exact", "pred_lr_exact",
    "prediction_queue_action", "prediction_queue_count", "prediction_queue_due",
    "prediction_queue_energy", "prediction_queue_exp_de", "prediction_queue_exp_dinj",
    "prediction_queue_injury", "prediction_reset_count",
    "prediction_state_delta_abs_max", "prediction_state_delta_abs_max_exact",
    "prediction_state_delta_abs_sum", "prediction_state_delta_abs_sum_exact",
    "prediction_update_count",
)


class DisabledPredictionSettlementError(RuntimeError):
    """Dormant prediction state cannot be reconstructed exactly."""


def materialize_disabled_prediction_state_v1(
    initial_projection: Any,
    persistent_state: Mapping[str, Any],
    *,
    reset_count_max: int,
) -> dict[str, Any]:
    """Preserve survivors and apply canonical reset values to changed slots."""

    metadata = (
        initial_projection.get("metadata")
        if isinstance(initial_projection, Mapping)
        else getattr(initial_projection, "metadata", None)
    )
    tensors = (
        initial_projection.get("tensors")
        if isinstance(initial_projection, Mapping)
        else getattr(initial_projection, "tensors", None)
    )
    if not isinstance(metadata, Mapping) or not isinstance(tensors, Mapping):
        raise DisabledPredictionSettlementError("initial checkpoint projection is incomplete")
    if not (
        metadata.get("ablate_prediction") is True
        or metadata.get("force_prediction_disabled") is True
    ):
        raise DisabledPredictionSettlementError(
            "dormant prediction settlement requires prediction to be disabled"
        )
    if type(reset_count_max) is not int or reset_count_max < 0:
        raise DisabledPredictionSettlementError("reset_count_max must be nonnegative")
    missing = sorted(set(DISABLED_PREDICTION_STATE_FIELDS_V1) - set(tensors))
    if missing:
        raise DisabledPredictionSettlementError(
            f"initial prediction state is missing: {missing}"
        )
    current_ids = persistent_state.get("slot_to_agent_id")
    alive = persistent_state.get("alive_mask")
    pred_lr = persistent_state.get("pred_lr")
    survival_steps = persistent_state.get("survival_steps")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (current_ids, alive, pred_lr, survival_steps)
    ):
        raise DisabledPredictionSettlementError(
            "resident identity or prediction-rate state is absent"
        )
    if current_ids.ndim != 2 or alive.shape != current_ids.shape:
        raise DisabledPredictionSettlementError("resident identity shape mismatch")
    if pred_lr.shape != current_ids.shape or survival_steps.shape != alive.shape:
        raise DisabledPredictionSettlementError("resident prediction shape mismatch")
    device = current_ids.device
    initial_ids = tensors["slot_to_agent_id"].to(device=device).unsqueeze(0)
    unchanged = alive & current_ids.eq(initial_ids)
    out = dict(persistent_state)
    for name in DISABLED_PREDICTION_STATE_FIELDS_V1:
        initial = tensors[name].to(device=device).unsqueeze(0)
        reset_value = -1 if name in {"prediction_queue_action", "prediction_queue_due"} else 0
        settled = torch.full_like(initial, reset_value)
        if name == "pred_lr_exact":
            # Checkpoint projection is already a host boundary. Canonicalize
            # here because CUDA division can differ from CPU by one binary ULP.
            canonical = pred_lr.detach().to(device="cpu", dtype=torch.float64)
            canonical = torch.round(canonical * 100_000_000.0)
            canonical = (canonical / 100_000_000.0).to(device=device)
            settled = torch.where(
                unchanged,
                initial,
                torch.where(alive, canonical, torch.zeros_like(canonical)),
            )
        elif name == "prediction_reset_count":
            settled = torch.where(
                alive,
                torch.clamp(survival_steps, max=reset_count_max).to(dtype=initial.dtype),
                torch.zeros_like(initial),
            )
        out[name] = settled.contiguous()
    return out


__all__ = [
    "DISABLED_PREDICTION_STATE_FIELDS_V1",
    "DisabledPredictionSettlementError",
    "materialize_disabled_prediction_state_v1",
]
