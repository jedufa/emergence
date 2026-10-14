from __future__ import annotations

"""CUDA-resident fixed-MLP inference and action-cache refill for Architecture B."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline

from perf.cuda_architecture_b_pcg64 import pcg64_state_tensors


ROOT = Path(__file__).resolve().parents[2]
CUDA_SOURCE = ROOT / "engine" / "perf" / "native" / "cuda_architecture_b_fixed_mlp.cu"
_MODULE: Any | None = None

CUDA_FIXED_MLP_CAPABILITIES = {
    "schema_version": "eiel_cuda_architecture_b_fixed_mlp_capabilities_v1",
    "decision": "ARCHB-CUDA-STAGE1A/FIXED-MLP-DEVICE-RESIDENT",
    "fixed_mlp_policy_resident": True,
    "policy_input_cuda": True,
    "policy_cache_cuda": True,
    "fixed_capacity_counted_active_rows": True,
    "device_resident_active_count": True,
    "host_sampling": False,
    "python_row_loop": False,
    "deterministic_argmax_parity": "exact_actions_bounded_float_logits",
    "stochastic_sampling": "numpy_pcg64_gumbel_exact_or_explicit_splitmix64_prototype",
    "stochastic_cross_backend_parity": "exact_when_pcg64_state_is_provided",
    "gru_supported": False,
    "full_generation_resident": False,
    "promotion_qualified": False,
}


@dataclass(frozen=True)
class CudaFixedMlpResult:
    logits: torch.Tensor
    actions: torch.Tensor
    action_cache_actions: torch.Tensor
    action_cache_token: torch.Tensor
    action_cache_cursor: torch.Tensor
    action_cache_next: torch.Tensor
    next_rng_counter: torch.Tensor
    pcg64_state_words: torch.Tensor
    pcg64_buffer_words: torch.Tensor

    def pcg64_state_dict(self) -> dict[str, object] | None:
        if self.pcg64_state_words.numel() == 0:
            return None
        state_words = [int(word) for word in self.pcg64_state_words.detach().cpu().tolist()]
        buffer_words = [int(word) for word in self.pcg64_buffer_words.detach().cpu().tolist()]
        return {
            "bit_generator": "PCG64",
            "state": {
                "state": state_words[0] | (state_words[1] << 64),
                "inc": state_words[2] | (state_words[3] << 64),
            },
            "has_uint32": buffer_words[0],
            "uinteger": buffer_words[1],
        }


def load_cuda_fixed_mlp_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("Architecture B fixed MLP requires an available CUDA device")
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    cpp_source = r"""
    #include <torch/extension.h>

    std::vector<torch::Tensor> eiel_cuda_fixed_mlp_refill_v1(
        torch::Tensor observations,
        torch::Tensor slots,
        torch::Tensor mlp_pack,
        torch::Tensor drives,
        torch::Tensor attack_bias,
        torch::Tensor action_temperature,
        torch::Tensor drive_logit_gain,
        torch::Tensor slot_tokens,
        torch::Tensor action_cache_actions,
        torch::Tensor action_cache_token,
        torch::Tensor action_cache_cursor,
        torch::Tensor action_cache_next,
        torch::Tensor pcg64_state_words,
        torch::Tensor pcg64_buffer_words,
        const int64_t hidden_size,
        const int64_t action_count,
        const int64_t cache_width,
        const bool deterministic,
        const int64_t rng_seed,
        const int64_t rng_counter,
        const double w_hunger_gather,
        const double w_pain_rest,
        const double w_fatigue_rest,
        const double w_pain_attack,
        const double w_fear_attack
    );

    std::vector<torch::Tensor> eiel_cuda_fixed_mlp_refill_counted_v1(
        torch::Tensor observations,
        torch::Tensor slots,
        torch::Tensor active_count,
        torch::Tensor mlp_pack,
        torch::Tensor drives,
        torch::Tensor attack_bias,
        torch::Tensor action_temperature,
        torch::Tensor drive_logit_gain,
        torch::Tensor slot_tokens,
        torch::Tensor action_cache_actions,
        torch::Tensor action_cache_token,
        torch::Tensor action_cache_cursor,
        torch::Tensor action_cache_next,
        torch::Tensor pcg64_state_words,
        torch::Tensor pcg64_buffer_words,
        const int64_t hidden_size,
        const int64_t action_count,
        const int64_t cache_width,
        const bool deterministic,
        const int64_t rng_seed,
        const int64_t rng_counter,
        const double w_hunger_gather,
        const double w_pain_rest,
        const double w_fatigue_rest,
        const double w_pain_attack,
        const double w_fear_attack
    );

    std::vector<torch::Tensor> eiel_cuda_fixed_mlp_refill_counted_with_residual_v1(
        torch::Tensor observations,
        torch::Tensor slots,
        torch::Tensor active_count,
        torch::Tensor logit_residuals,
        torch::Tensor mlp_pack,
        torch::Tensor drives,
        torch::Tensor attack_bias,
        torch::Tensor action_temperature,
        torch::Tensor drive_logit_gain,
        torch::Tensor slot_tokens,
        torch::Tensor action_cache_actions,
        torch::Tensor action_cache_token,
        torch::Tensor action_cache_cursor,
        torch::Tensor action_cache_next,
        torch::Tensor pcg64_state_words,
        torch::Tensor pcg64_buffer_words,
        const int64_t hidden_size,
        const int64_t action_count,
        const int64_t cache_width,
        const bool deterministic,
        const int64_t rng_seed,
        const int64_t rng_counter,
        const double w_hunger_gather,
        const double w_pain_rest,
        const double w_fatigue_rest,
        const double w_pain_attack,
        const double w_fear_attack
    );

    std::vector<torch::Tensor> eiel_cuda_fixed_mlp_refill_batched_counted_v1(
        torch::Tensor observations,
        torch::Tensor slots,
        torch::Tensor active_count,
        torch::Tensor mlp_pack,
        torch::Tensor drives,
        torch::Tensor attack_bias,
        torch::Tensor action_temperature,
        torch::Tensor drive_logit_gain,
        torch::Tensor slot_tokens,
        torch::Tensor action_cache_actions,
        torch::Tensor action_cache_token,
        torch::Tensor action_cache_cursor,
        torch::Tensor action_cache_next,
        torch::Tensor pcg64_state_words,
        torch::Tensor pcg64_buffer_words,
        const int64_t hidden_size,
        const int64_t action_count,
        const int64_t cache_width,
        const bool deterministic,
        const int64_t rng_seed,
        const int64_t rng_counter,
        const double w_hunger_gather,
        const double w_pain_rest,
        const double w_fatigue_rest,
        const double w_pain_attack,
        const double w_fear_attack
    );
    """
    _MODULE = load_inline(
        name="eiel_cuda_architecture_b_fixed_mlp_ext",
        cpp_sources=[cpp_source],
        cuda_sources=[CUDA_SOURCE.read_text(encoding="utf-8")],
        functions=[
            "eiel_cuda_fixed_mlp_refill_v1",
            "eiel_cuda_fixed_mlp_refill_counted_v1",
            "eiel_cuda_fixed_mlp_refill_counted_with_residual_v1",
            "eiel_cuda_fixed_mlp_refill_batched_counted_v1",
        ],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        extra_include_paths=[str(CUDA_SOURCE.parent)],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def fixed_mlp_refill_cuda(
    *,
    observations: torch.Tensor,
    slots: torch.Tensor,
    mlp_pack: torch.Tensor,
    drives: torch.Tensor,
    attack_bias: torch.Tensor,
    action_temperature: torch.Tensor,
    drive_logit_gain: torch.Tensor,
    slot_tokens: torch.Tensor,
    action_cache_actions: torch.Tensor,
    action_cache_token: torch.Tensor,
    action_cache_cursor: torch.Tensor,
    action_cache_next: torch.Tensor,
    hidden_size: int,
    action_count: int,
    cache_width: int,
    deterministic: bool,
    rng_seed: int = 0,
    rng_counter: int = 0,
    pcg64_state: dict[str, object] | None = None,
    pcg64_state_words: torch.Tensor | None = None,
    pcg64_buffer_words: torch.Tensor | None = None,
    w_hunger_gather: float = 1.2,
    w_pain_rest: float = 0.9,
    w_fatigue_rest: float = 1.0,
    w_pain_attack: float = 1.2,
    w_fear_attack: float = 1.4,
    policy_family: str = "fixed_mlp",
    build_directory: str | Path | None = None,
) -> CudaFixedMlpResult:
    if policy_family != "fixed_mlp":
        raise RuntimeError("Architecture B CUDA policy slice supports fixed_mlp only; GRU is excluded")
    if pcg64_state is not None and (pcg64_state_words is not None or pcg64_buffer_words is not None):
        raise ValueError("provide serialized PCG64 state or resident PCG64 tensors, not both")
    if pcg64_state is not None:
        pcg64_state_words, pcg64_buffer_words = pcg64_state_tensors(
            pcg64_state,
            device=observations.device,
        )
    if (pcg64_state_words is None) != (pcg64_buffer_words is None):
        raise ValueError("resident PCG64 state and buffer tensors must be provided together")
    if pcg64_state_words is None:
        pcg64_state_words = torch.empty(0, dtype=torch.uint64, device=observations.device)
        pcg64_buffer_words = torch.empty(0, dtype=torch.int64, device=observations.device)
    module = load_cuda_fixed_mlp_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_fixed_mlp_refill_v1(
        observations,
        slots,
        mlp_pack,
        drives,
        attack_bias,
        action_temperature,
        drive_logit_gain,
        slot_tokens,
        action_cache_actions,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        pcg64_state_words,
        pcg64_buffer_words,
        int(hidden_size),
        int(action_count),
        int(cache_width),
        bool(deterministic),
        int(rng_seed),
        int(rng_counter),
        float(w_hunger_gather),
        float(w_pain_rest),
        float(w_fatigue_rest),
        float(w_pain_attack),
        float(w_fear_attack),
    )
    return CudaFixedMlpResult(*outputs)


def fixed_mlp_refill_counted_cuda(
    *,
    observations: torch.Tensor,
    slots: torch.Tensor,
    active_count: torch.Tensor,
    mlp_pack: torch.Tensor,
    drives: torch.Tensor,
    attack_bias: torch.Tensor,
    action_temperature: torch.Tensor,
    drive_logit_gain: torch.Tensor,
    slot_tokens: torch.Tensor,
    action_cache_actions: torch.Tensor,
    action_cache_token: torch.Tensor,
    action_cache_cursor: torch.Tensor,
    action_cache_next: torch.Tensor,
    hidden_size: int,
    action_count: int,
    cache_width: int,
    deterministic: bool,
    rng_seed: int = 0,
    rng_counter: int = 0,
    pcg64_state: dict[str, object] | None = None,
    pcg64_state_words: torch.Tensor | None = None,
    pcg64_buffer_words: torch.Tensor | None = None,
    w_hunger_gather: float = 1.2,
    w_pain_rest: float = 0.9,
    w_fatigue_rest: float = 1.0,
    w_pain_attack: float = 1.2,
    w_fear_attack: float = 1.4,
    policy_family: str = "fixed_mlp",
    build_directory: str | Path | None = None,
) -> CudaFixedMlpResult:
    if policy_family != "fixed_mlp":
        raise RuntimeError("Architecture B CUDA policy slice supports fixed_mlp only; GRU is excluded")
    if not active_count.is_cuda or active_count.device != observations.device:
        raise ValueError("active_count must be resident on the observation CUDA device")
    if active_count.dtype != torch.int64 or tuple(active_count.shape) != (1,):
        raise ValueError("active_count must be contiguous CUDA int64[1]")
    if not active_count.is_contiguous():
        raise ValueError("active_count must be contiguous CUDA int64[1]")
    if pcg64_state is not None and (pcg64_state_words is not None or pcg64_buffer_words is not None):
        raise ValueError("provide serialized PCG64 state or resident PCG64 tensors, not both")
    if pcg64_state is not None:
        pcg64_state_words, pcg64_buffer_words = pcg64_state_tensors(
            pcg64_state,
            device=observations.device,
        )
    if (pcg64_state_words is None) != (pcg64_buffer_words is None):
        raise ValueError("resident PCG64 state and buffer tensors must be provided together")
    if pcg64_state_words is None:
        pcg64_state_words = torch.empty(0, dtype=torch.uint64, device=observations.device)
        pcg64_buffer_words = torch.empty(0, dtype=torch.int64, device=observations.device)
    module = load_cuda_fixed_mlp_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_fixed_mlp_refill_counted_v1(
        observations,
        slots,
        active_count,
        mlp_pack,
        drives,
        attack_bias,
        action_temperature,
        drive_logit_gain,
        slot_tokens,
        action_cache_actions,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        pcg64_state_words,
        pcg64_buffer_words,
        int(hidden_size),
        int(action_count),
        int(cache_width),
        bool(deterministic),
        int(rng_seed),
        int(rng_counter),
        float(w_hunger_gather),
        float(w_pain_rest),
        float(w_fatigue_rest),
        float(w_pain_attack),
        float(w_fear_attack),
    )
    return CudaFixedMlpResult(*outputs)


def fixed_mlp_refill_counted_with_residual_cuda(
    *,
    observations: torch.Tensor,
    slots: torch.Tensor,
    active_count: torch.Tensor,
    logit_residuals: torch.Tensor,
    mlp_pack: torch.Tensor,
    drives: torch.Tensor,
    attack_bias: torch.Tensor,
    action_temperature: torch.Tensor,
    drive_logit_gain: torch.Tensor,
    slot_tokens: torch.Tensor,
    action_cache_actions: torch.Tensor,
    action_cache_token: torch.Tensor,
    action_cache_cursor: torch.Tensor,
    action_cache_next: torch.Tensor,
    hidden_size: int,
    action_count: int,
    cache_width: int,
    deterministic: bool,
    rng_seed: int = 0,
    rng_counter: int = 0,
    pcg64_state: dict[str, object] | None = None,
    pcg64_state_words: torch.Tensor | None = None,
    pcg64_buffer_words: torch.Tensor | None = None,
    w_hunger_gather: float = 1.2,
    w_pain_rest: float = 0.9,
    w_fatigue_rest: float = 1.0,
    w_pain_attack: float = 1.2,
    w_fear_attack: float = 1.4,
    policy_family: str = "fixed_mlp",
    build_directory: str | Path | None = None,
) -> CudaFixedMlpResult:
    """Refill fixed-MLP actions after adding resident modular residuals."""

    if policy_family != "fixed_mlp":
        raise RuntimeError("Architecture B CUDA policy slice supports fixed_mlp only; GRU is excluded")
    if not active_count.is_cuda or active_count.device != observations.device:
        raise ValueError("active_count must be resident on the observation CUDA device")
    if active_count.dtype != torch.int64 or tuple(active_count.shape) != (1,):
        raise ValueError("active_count must be contiguous CUDA int64[1]")
    if not active_count.is_contiguous():
        raise ValueError("active_count must be contiguous CUDA int64[1]")
    if (
        not logit_residuals.is_cuda
        or logit_residuals.device != observations.device
        or logit_residuals.dtype != torch.float32
        or tuple(logit_residuals.shape) != (observations.shape[0], action_count)
        or not logit_residuals.is_contiguous()
        or logit_residuals.requires_grad
    ):
        raise ValueError(
            "logit_residuals must be contiguous, no-grad CUDA float32[rows,action_count] "
            "on the observation device"
        )
    if pcg64_state is not None and (pcg64_state_words is not None or pcg64_buffer_words is not None):
        raise ValueError("provide serialized PCG64 state or resident PCG64 tensors, not both")
    if pcg64_state is not None:
        pcg64_state_words, pcg64_buffer_words = pcg64_state_tensors(
            pcg64_state,
            device=observations.device,
        )
    if (pcg64_state_words is None) != (pcg64_buffer_words is None):
        raise ValueError("resident PCG64 state and buffer tensors must be provided together")
    if pcg64_state_words is None:
        pcg64_state_words = torch.empty(0, dtype=torch.uint64, device=observations.device)
        pcg64_buffer_words = torch.empty(0, dtype=torch.int64, device=observations.device)
    module = load_cuda_fixed_mlp_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_fixed_mlp_refill_counted_with_residual_v1(
        observations,
        slots,
        active_count,
        logit_residuals,
        mlp_pack,
        drives,
        attack_bias,
        action_temperature,
        drive_logit_gain,
        slot_tokens,
        action_cache_actions,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        pcg64_state_words,
        pcg64_buffer_words,
        int(hidden_size),
        int(action_count),
        int(cache_width),
        bool(deterministic),
        int(rng_seed),
        int(rng_counter),
        float(w_hunger_gather),
        float(w_pain_rest),
        float(w_fatigue_rest),
        float(w_pain_attack),
        float(w_fear_attack),
    )
    return CudaFixedMlpResult(*outputs)


def fixed_mlp_refill_batched_counted_cuda(
    *,
    observations: torch.Tensor,
    slots: torch.Tensor,
    active_count: torch.Tensor,
    mlp_pack: torch.Tensor,
    drives: torch.Tensor,
    attack_bias: torch.Tensor,
    action_temperature: torch.Tensor,
    drive_logit_gain: torch.Tensor,
    slot_tokens: torch.Tensor,
    action_cache_actions: torch.Tensor,
    action_cache_token: torch.Tensor,
    action_cache_cursor: torch.Tensor,
    action_cache_next: torch.Tensor,
    hidden_size: int,
    action_count: int,
    cache_width: int,
    deterministic: bool,
    rng_seed: int = 0,
    rng_counter: int = 0,
    pcg64_state_words: torch.Tensor | None = None,
    pcg64_buffer_words: torch.Tensor | None = None,
    w_hunger_gather: float = 1.2,
    w_pain_rest: float = 0.9,
    w_fatigue_rest: float = 1.0,
    w_pain_attack: float = 1.2,
    w_fear_attack: float = 1.4,
    build_directory: str | Path | None = None,
) -> CudaFixedMlpResult:
    if observations.ndim != 3:
        raise ValueError("observations must be [cells,rows,features]")
    cells = int(observations.shape[0])
    if tuple(active_count.shape) != (cells, 1) or active_count.dtype != torch.int64:
        raise ValueError("active_count must be int64[cells,1]")
    if pcg64_state_words is None:
        pcg64_state_words = torch.empty(
            0, dtype=torch.uint64, device=observations.device
        )
        pcg64_buffer_words = torch.empty(
            0, dtype=torch.int64, device=observations.device
        )
    if pcg64_buffer_words is None:
        raise ValueError("resident PCG64 state and buffer tensors must be provided together")
    module = load_cuda_fixed_mlp_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_fixed_mlp_refill_batched_counted_v1(
        observations,
        slots,
        active_count,
        mlp_pack,
        drives,
        attack_bias,
        action_temperature,
        drive_logit_gain,
        slot_tokens,
        action_cache_actions,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        pcg64_state_words,
        pcg64_buffer_words,
        int(hidden_size),
        int(action_count),
        int(cache_width),
        bool(deterministic),
        int(rng_seed),
        int(rng_counter),
        float(w_hunger_gather),
        float(w_pain_rest),
        float(w_fatigue_rest),
        float(w_pain_attack),
        float(w_fear_attack),
    )
    return CudaFixedMlpResult(*outputs)


__all__ = [
    "CUDA_FIXED_MLP_CAPABILITIES",
    "CudaFixedMlpResult",
    "fixed_mlp_refill_batched_counted_cuda",
    "fixed_mlp_refill_counted_cuda",
    "fixed_mlp_refill_counted_with_residual_cuda",
    "fixed_mlp_refill_cuda",
    "load_cuda_fixed_mlp_extension",
]
