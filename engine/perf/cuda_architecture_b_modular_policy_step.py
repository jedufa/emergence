from __future__ import annotations

"""Device-resident Phase E modular tick plus fixed-MLP policy composer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, MutableMapping

import torch

from perf.cuda_architecture_b_fixed_mlp import (
    CudaFixedMlpResult,
    fixed_mlp_refill_counted_with_residual_cuda,
)
from perf.cuda_architecture_b_modular_generation_contract import (
    ModularGenerationEntryReceipt,
    prepare_modular_generation_entry,
)
from perf.cuda_architecture_b_modular_tick import (
    ACTION_WIDTH,
    OBSERVATION_WIDTH,
    ModularTickReceipt,
    execute_modular_tick,
)


MODULAR_POLICY_STEP_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_modular_policy_step_v1"
)
MODULAR_POLICY_STEP_LANE: Final[str] = (
    "phase_e_device_resident_modular_tick_then_fixed_mlp_v1"
)


class ModularPolicyStepError(RuntimeError):
    """The bounded modular policy-step contract is invalid."""


@dataclass(frozen=True)
class ModularPolicyStepReceipt:
    profile_id: str
    agent_capacity: int
    device: str
    early_to_adult_step: int
    adult_to_late_step: int
    compact_row_count: int
    modular_manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODULAR_POLICY_STEP_SCHEMA_VERSION,
            "lane": MODULAR_POLICY_STEP_LANE,
            "profile_id": self.profile_id,
            "agent_capacity": self.agent_capacity,
            "device": self.device,
            "early_to_adult_step": self.early_to_adult_step,
            "adult_to_late_step": self.adult_to_late_step,
            "compact_row_count": self.compact_row_count,
            "modular_manifest_sha256": self.modular_manifest_sha256,
            "action_width": ACTION_WIDTH,
            "observation_width": OBSERVATION_WIDTH,
            "cache_width": 1,
            "policy_stride": 1,
            "max_births": 0,
            "turnover_enabled": False,
            "all_active_policy_rows_are_misses": True,
            "residual_applied_before_action_selection": True,
            "device_to_host_synchronizations": 0,
            "hidden_fallback_count": 0,
            "lifecycle_supported": False,
            "campaign_claimed": False,
            "checkpoint_closure_claimed": False,
        }


@dataclass(frozen=True)
class ModularPolicyStepResult:
    policy_result: CudaFixedMlpResult | Any
    entry_receipt: ModularGenerationEntryReceipt
    modular_tick_receipt: ModularTickReceipt | Any
    modular_tick_audit_tensors: dict[str, torch.Tensor]
    audit_tensors: dict[str, torch.Tensor]
    receipt: ModularPolicyStepReceipt


def _exact_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ModularPolicyStepError(f"{name} must be an exact int >= {minimum}")
    return value


def _assert_device(condition: torch.Tensor, message: str) -> None:
    try:
        torch._assert_async(condition, message)
    except RuntimeError as exc:
        raise ModularPolicyStepError(str(exc)) from exc


def _require_tensor(
    value: Any,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device != device
        or value.dtype != dtype
        or tuple(value.shape) != shape
        or not value.is_contiguous()
        or value.requires_grad
    ):
        raise ModularPolicyStepError(f"invalid resident policy-step tensor: {name}")
    return value


def execute_modular_policy_step(
    resident_tensors: Any,
    *,
    enabled: bool = False,
    profile_id: str,
    observations: Any,
    fatigue: Any,
    miss_slots: Any,
    active_count: Any,
    hidden_size: int,
    action_count: int,
    cache_width: int,
    policy_stride: int,
    max_births: int,
    turnover_enabled: bool,
    early_to_adult_step: int,
    adult_to_late_step: int,
    deterministic: bool,
    rng_seed: int = 0,
    rng_counter: int = 0,
    build_directory: str | Path | None = None,
    allow_cpu_validation: bool = False,
) -> ModularPolicyStepResult | None:
    """Run one modular tick and feed compact residuals into fixed-MLP choice.

    Disabled mode returns before inspecting any modular or policy argument.
    The enabled production path remains on one device and performs no lifecycle
    operation, host transfer, or synchronization.
    """

    if enabled is False:
        return None
    if enabled is not True:
        raise ModularPolicyStepError("enabled must be an exact bool")
    if not isinstance(resident_tensors, MutableMapping):
        raise ModularPolicyStepError("resident_tensors must be a mutable mapping")
    if not isinstance(observations, torch.Tensor):
        raise ModularPolicyStepError("observations must be a tensor")
    first_threshold = _exact_int(
        early_to_adult_step, name="early_to_adult_step", minimum=1
    )
    second_threshold = _exact_int(
        adult_to_late_step, name="adult_to_late_step", minimum=2
    )
    if first_threshold >= second_threshold:
        raise ModularPolicyStepError("stage thresholds must be strictly ordered")
    device = observations.device
    capacity = int(observations.shape[0]) if observations.ndim > 0 else 0
    entry = prepare_modular_generation_entry(
        resident_tensors,
        enabled=True,
        agent_capacity=capacity,
        profile_id=profile_id,
        policy_family="fixed_mlp",
        action_count=action_count,
        observation_width=(
            int(observations.shape[1]) if observations.ndim == 2 else -1
        ),
        cache_width=cache_width,
        policy_stride=policy_stride,
        development_mature_start_step=early_to_adult_step,
        development_late_start_step=adult_to_late_step,
        max_births=max_births,
        turnover_enabled=turnover_enabled,
        device=device,
        allow_cpu_validation=allow_cpu_validation,
    )
    if entry is None:
        raise ModularPolicyStepError("enabled modular generation entry disappeared")
    _exact_int(hidden_size, name="hidden_size", minimum=1)
    if type(deterministic) is not bool:
        raise ModularPolicyStepError("deterministic must be an exact bool")

    full_observations = _require_tensor(
        observations,
        name="observations",
        device=device,
        dtype=torch.float32,
        shape=(capacity, OBSERVATION_WIDTH),
    )
    full_fatigue = _require_tensor(
        fatigue,
        name="fatigue",
        device=device,
        dtype=torch.float32,
        shape=(capacity,),
    )
    survival_steps = _require_tensor(
        resident_tensors.get("survival_steps"),
        name="survival_steps",
        device=device,
        dtype=torch.int32,
        shape=(capacity,),
    )
    if not isinstance(miss_slots, torch.Tensor) or miss_slots.ndim != 1:
        raise ModularPolicyStepError("miss_slots must be a rank-one tensor")
    compact_rows = int(miss_slots.shape[0])
    compact_slots = _require_tensor(
        miss_slots,
        name="miss_slots",
        device=device,
        dtype=torch.int32,
        shape=(compact_rows,),
    )
    resident_active_count = _require_tensor(
        active_count,
        name="active_count",
        device=device,
        dtype=torch.int64,
        shape=(1,),
    )
    _assert_device(
        torch.eq(resident_active_count[0], compact_rows),
        "every compact miss row must be active",
    )
    _assert_device(
        torch.all(torch.logical_and(compact_slots >= 0, compact_slots < capacity)),
        "miss_slots must be within fixed capacity",
    )
    slot_index = compact_slots.to(dtype=torch.int64)
    slot_counts = torch.bincount(slot_index, minlength=capacity)
    _assert_device(
        torch.all(slot_counts <= 1),
        "miss_slots must contain each active slot exactly once",
    )
    _assert_device(
        torch.all(resident_tensors["alive_mask"].index_select(0, slot_index)),
        "every active policy miss row must address an alive slot",
    )
    _assert_device(
        torch.eq(
            torch.sum(resident_tensors["alive_mask"].to(dtype=torch.int64)),
            resident_active_count[0],
        ),
        "every alive policy row must be a cache miss",
    )
    _assert_device(torch.all(survival_steps >= 0), "survival_steps must be nonnegative")

    stage_indices = torch.where(
        survival_steps >= second_threshold,
        torch.full_like(survival_steps, 2),
        torch.where(
            survival_steps >= first_threshold,
            torch.ones_like(survival_steps),
            torch.zeros_like(survival_steps),
        ),
    ).contiguous()
    modular = execute_modular_tick(
        resident_tensors,
        full_observations,
        full_fatigue,
        stage_indices,
        action_width=action_count,
    )
    full_residuals = _require_tensor(
        modular.action_residuals,
        name="modular_action_residuals",
        device=device,
        dtype=torch.float32,
        shape=(capacity, ACTION_WIDTH),
    )
    compact_observations = full_observations.index_select(0, slot_index).contiguous()
    compact_residuals = full_residuals.index_select(0, slot_index).contiguous()
    policy_result = fixed_mlp_refill_counted_with_residual_cuda(
        observations=compact_observations,
        slots=compact_slots,
        active_count=resident_active_count,
        logit_residuals=compact_residuals,
        mlp_pack=resident_tensors["mlp_pack"],
        drives=resident_tensors["drives"],
        attack_bias=resident_tensors["attack_bias"],
        action_temperature=resident_tensors["action_temperature"],
        drive_logit_gain=resident_tensors["drive_logit_gain"],
        slot_tokens=resident_tensors["slot_tokens"],
        action_cache_actions=resident_tensors["action_cache_actions"],
        action_cache_token=resident_tensors["action_cache_token"],
        action_cache_cursor=resident_tensors["action_cache_cursor"],
        action_cache_next=resident_tensors["action_cache_next"],
        pcg64_state_words=resident_tensors["pcg64_state_words"],
        pcg64_buffer_words=resident_tensors["pcg64_buffer_words"],
        hidden_size=hidden_size,
        action_count=action_count,
        cache_width=cache_width,
        deterministic=deterministic,
        rng_seed=rng_seed,
        rng_counter=rng_counter,
        build_directory=build_directory,
    )
    audit = {
        "stage_indices": stage_indices,
        "miss_slots": compact_slots,
        "compact_observations": compact_observations,
        "full_action_residuals": full_residuals,
        "compact_action_residuals": compact_residuals,
    }
    receipt = ModularPolicyStepReceipt(
        profile_id=profile_id,
        agent_capacity=capacity,
        device=str(device),
        early_to_adult_step=first_threshold,
        adult_to_late_step=second_threshold,
        compact_row_count=compact_rows,
        modular_manifest_sha256=entry.manifest.manifest_sha256,
    )
    return ModularPolicyStepResult(
        policy_result=policy_result,
        entry_receipt=entry.receipt,
        modular_tick_receipt=modular.receipt,
        modular_tick_audit_tensors=dict(modular.audit_tensors),
        audit_tensors=audit,
        receipt=receipt,
    )


__all__ = [
    "MODULAR_POLICY_STEP_LANE",
    "MODULAR_POLICY_STEP_SCHEMA_VERSION",
    "ModularPolicyStepError",
    "ModularPolicyStepReceipt",
    "ModularPolicyStepResult",
    "execute_modular_policy_step",
]
