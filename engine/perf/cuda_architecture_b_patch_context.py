from __future__ import annotations

"""Bounded resident-CUDA Patch context updates.

The base surface owns pending trace decay/emission and base food regrowth. The
P2 surface additionally owns occupancy/density emissions and exact waste and
density regrowth consequences. Neither surface claims actions, policy,
lifecycle, checkpointing, multi-cell batching, Jungle, WDL, or GRU support.
"""

from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CUDA = ROOT / "engine" / "perf" / "native" / "cuda_architecture_b_patch_context.cu"
PATCH_CONTEXT_STAT_FIELDS: Final[tuple[str, ...]] = (
    "scent_emitted_total",
    "scent_decayed_total",
    "waste_emitted_total",
    "waste_decayed_total",
    "food_regrowth_total",
    "regrowth_base_total",
    "regrowth_effective_total",
    "regrowth_suppressed_total",
    "food_depleted_frac",
    "food_full_frac",
)
PATCH_CONTEXT_P2_STAT_FIELDS: Final[tuple[str, ...]] = (
    "scent_emitted_total",
    "scent_decayed_total",
    "waste_emitted_total",
    "waste_decayed_total",
    "food_regrowth_total",
    "regrowth_base_total",
    "regrowth_effective_total",
    "regrowth_suppressed_total",
    "regrowth_multiplier_mean",
    "regrowth_multiplier_min",
    "regrowth_multiplier_p10",
    "regrowth_multiplier_p90",
    "waste_regrowth_penalty_mean",
    "waste_regrowth_penalty_max",
    "density_regrowth_penalty_mean",
    "density_regrowth_penalty_max",
    "high_waste_regrowth_suppression_frac",
    "high_density_regrowth_suppression_frac",
    "regrowth_after_waste_total",
    "regrowth_after_recovery_total",
    "recovery_added_regrowth_total",
    "food_depleted_frac",
    "food_full_frac",
    "density_hotspot_frac",
    "regrowth_consequence_steps",
)
PATCH_CONTEXT_CAPABILITY_RECEIPT: Final[dict[str, bool]] = {
    "single_cell_cuda_resident_state": True,
    "pending_trace_decay_and_emission": True,
    "base_food_regrowth": True,
    "binary_food_projection": True,
    "occupancy_or_density_emission": True,
    "waste_or_density_regrowth_consequence": True,
    "actions_or_policy": False,
    "lifecycle_or_checkpoint": False,
    "multi_cell_batching": False,
    "jungle": False,
    "wdl": False,
    "gru": False,
    "throughput_qualification": False,
}

_MODULE: Any | None = None


def load_cuda_patch_context_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    cuda_source = SOURCE_CUDA.read_text(encoding="utf-8")
    module_suffix = sha256(cuda_source.encode("utf-8")).hexdigest()[:12]
    cpp_source = """
    #include <torch/extension.h>

    torch::Tensor eiel_cuda_architecture_b_patch_context_v1(
        torch::Tensor patch_food_amount,
        torch::Tensor patch_fertility,
        torch::Tensor patch_scent,
        torch::Tensor patch_waste,
        torch::Tensor patch_scent_emit_pending,
        torch::Tensor patch_waste_emit_pending,
        torch::Tensor food_grid,
        int64_t steps,
        double patch_scent_decay_per_step,
        double patch_waste_decay_per_step,
        double patch_scent_max,
        double patch_waste_max,
        double patch_food_amount_max,
        double patch_food_regrow_per_step,
        bool patch_food_regrow_use_fertility,
        bool patch_food_regrow_cap_by_fertility,
        double patch_food_presence_threshold
    );

    torch::Tensor eiel_cuda_architecture_b_patch_context_p2_v1(
        torch::Tensor patch_food_amount,
        torch::Tensor patch_fertility,
        torch::Tensor patch_scent,
        torch::Tensor patch_waste,
        torch::Tensor patch_scent_emit_pending,
        torch::Tensor patch_waste_emit_pending,
        torch::Tensor agent_grid,
        torch::Tensor patch_density,
        torch::Tensor food_grid,
        int64_t steps,
        double patch_scent_decay_per_step,
        double patch_waste_decay_per_step,
        double patch_scent_presence_emit,
        double patch_waste_presence_emit,
        double patch_waste_density_emit_scale,
        double patch_scent_max,
        double patch_waste_max,
        double patch_food_amount_max,
        double patch_food_regrow_per_step,
        bool patch_food_regrow_use_fertility,
        bool patch_food_regrow_cap_by_fertility,
        double patch_food_presence_threshold,
        int64_t patch_density_radius,
        bool patch_density_include_self,
        bool patch_density_normalize,
        bool patch_world_waste_consequence_enable,
        bool patch_waste_regrowth_penalty_enable,
        double patch_waste_regrowth_penalty_scale,
        double patch_waste_regrowth_penalty_max,
        bool patch_density_regrowth_penalty_enable,
        double patch_density_regrowth_penalty_scale,
        double patch_density_regrowth_penalty_max,
        double patch_regrowth_min_multiplier
    );
    """
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load_inline(
        name=f"eiel_cuda_architecture_b_patch_context_{module_suffix}",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=[
            "eiel_cuda_architecture_b_patch_context_v1",
            "eiel_cuda_architecture_b_patch_context_p2_v1",
        ],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def execute_patch_context_cuda(
    *,
    patch_food_amount: torch.Tensor,
    patch_fertility: torch.Tensor,
    patch_scent: torch.Tensor,
    patch_waste: torch.Tensor,
    patch_scent_emit_pending: torch.Tensor,
    patch_waste_emit_pending: torch.Tensor,
    food_grid: torch.Tensor,
    steps: int = 1,
    patch_scent_decay_per_step: float = 0.05,
    patch_waste_decay_per_step: float = 0.02,
    patch_scent_max: float = 1.0,
    patch_waste_max: float = 1.0,
    patch_food_amount_max: float = 1.0,
    patch_food_regrow_per_step: float = 0.02,
    patch_food_regrow_use_fertility: bool = True,
    patch_food_regrow_cap_by_fertility: bool = False,
    patch_food_presence_threshold: float = 0.5,
    build_directory: str | Path | None = None,
) -> torch.Tensor:
    """Advance base-Patch context state in place and return CUDA stats by step.

    `steps=1` is the tick surface. Larger values execute the same bounded
    update repeatedly in one native call without host projection between steps.
    """
    module = load_cuda_patch_context_extension(build_directory=build_directory)
    return module.eiel_cuda_architecture_b_patch_context_v1(
        patch_food_amount,
        patch_fertility,
        patch_scent,
        patch_waste,
        patch_scent_emit_pending,
        patch_waste_emit_pending,
        food_grid,
        int(steps),
        float(patch_scent_decay_per_step),
        float(patch_waste_decay_per_step),
        float(patch_scent_max),
        float(patch_waste_max),
        float(patch_food_amount_max),
        float(patch_food_regrow_per_step),
        bool(patch_food_regrow_use_fertility),
        bool(patch_food_regrow_cap_by_fertility),
        float(patch_food_presence_threshold),
    )


def execute_patch_context_p2_cuda(
    *,
    patch_food_amount: torch.Tensor,
    patch_fertility: torch.Tensor,
    patch_scent: torch.Tensor,
    patch_waste: torch.Tensor,
    patch_scent_emit_pending: torch.Tensor,
    patch_waste_emit_pending: torch.Tensor,
    agent_grid: torch.Tensor,
    patch_density: torch.Tensor,
    food_grid: torch.Tensor,
    steps: int = 1,
    patch_scent_decay_per_step: float = 0.05,
    patch_waste_decay_per_step: float = 0.02,
    patch_scent_presence_emit: float = 0.01,
    patch_waste_presence_emit: float = 0.005,
    patch_waste_density_emit_scale: float = 0.0,
    patch_scent_max: float = 1.0,
    patch_waste_max: float = 1.0,
    patch_food_amount_max: float = 1.0,
    patch_food_regrow_per_step: float = 0.02,
    patch_food_regrow_use_fertility: bool = True,
    patch_food_regrow_cap_by_fertility: bool = False,
    patch_food_presence_threshold: float = 0.5,
    patch_density_radius: int = 1,
    patch_density_include_self: bool = True,
    patch_density_normalize: bool = False,
    patch_world_waste_consequence_enable: bool = True,
    patch_waste_regrowth_penalty_enable: bool = True,
    patch_waste_regrowth_penalty_scale: float = 0.5,
    patch_waste_regrowth_penalty_max: float = 0.5,
    patch_density_regrowth_penalty_enable: bool = True,
    patch_density_regrowth_penalty_scale: float = 0.5,
    patch_density_regrowth_penalty_max: float = 0.5,
    patch_regrowth_min_multiplier: float = 0.1,
    build_directory: str | Path | None = None,
) -> torch.Tensor:
    """Advance the accepted P2 context in place without host projection."""
    module = load_cuda_patch_context_extension(build_directory=build_directory)
    return module.eiel_cuda_architecture_b_patch_context_p2_v1(
        patch_food_amount,
        patch_fertility,
        patch_scent,
        patch_waste,
        patch_scent_emit_pending,
        patch_waste_emit_pending,
        agent_grid,
        patch_density,
        food_grid,
        int(steps),
        float(patch_scent_decay_per_step),
        float(patch_waste_decay_per_step),
        float(patch_scent_presence_emit),
        float(patch_waste_presence_emit),
        float(patch_waste_density_emit_scale),
        float(patch_scent_max),
        float(patch_waste_max),
        float(patch_food_amount_max),
        float(patch_food_regrow_per_step),
        bool(patch_food_regrow_use_fertility),
        bool(patch_food_regrow_cap_by_fertility),
        float(patch_food_presence_threshold),
        int(patch_density_radius),
        bool(patch_density_include_self),
        bool(patch_density_normalize),
        bool(patch_world_waste_consequence_enable),
        bool(patch_waste_regrowth_penalty_enable),
        float(patch_waste_regrowth_penalty_scale),
        float(patch_waste_regrowth_penalty_max),
        bool(patch_density_regrowth_penalty_enable),
        float(patch_density_regrowth_penalty_scale),
        float(patch_density_regrowth_penalty_max),
        float(patch_regrowth_min_multiplier),
    )


__all__ = [
    "PATCH_CONTEXT_CAPABILITY_RECEIPT",
    "PATCH_CONTEXT_P2_STAT_FIELDS",
    "PATCH_CONTEXT_STAT_FIELDS",
    "execute_patch_context_cuda",
    "execute_patch_context_p2_cuda",
    "load_cuda_patch_context_extension",
]
