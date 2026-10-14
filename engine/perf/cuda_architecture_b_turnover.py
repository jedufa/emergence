from __future__ import annotations

"""Exact resident-CUDA generation-turnover selection planning."""

from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CUDA = ROOT / "engine" / "perf" / "native" / "cuda_architecture_b_turnover.cu"
NATIVE_ROOT = ROOT / "engine" / "perf" / "native"
SOURCE_PCG64 = NATIVE_ROOT / "cuda_architecture_b_pcg64.cuh"

TURNOVER_PLAN_STATUS: Final[dict[int, str]] = {
    0: "ok",
    1: "invalid_config",
    2: "invalid_resident_state",
    3: "nonuniform_early_life_weight",
    4: "low_population_reinitialization_unsupported",
    5: "post_cull_low_population_reinitialization_unsupported",
    6: "insufficient_refill_parent_pool",
}

TURNOVER_CULL_STATUS: Final[dict[int, str]] = {
    0: "ok",
    1: "turnover_plan_failed",
    2: "invalid_resident_state",
    3: "free_queue_capacity_exceeded",
    4: "release_state_contradiction",
}

TURNOVER_REFILL_PLAN_STATUS: Final[dict[int, str]] = {
    0: "ok",
    1: "turnover_plan_failed",
    2: "invalid_resident_state",
    3: "invalid_parent_pool",
    4: "no_spawn_space",
}

TURNOVER_PLAN_CAPABILITIES: Final[dict[str, bool | str]] = {
    "schema_version": "eiel_cuda_architecture_b_turnover_plan_capabilities_v1",
    "decision": "ARCHB-CUDA/GENERATION-TURNOVER-SELECTION-PLAN",
    "resident_cuda_state": True,
    "fixed_mlp_flat_patch": True,
    "exact_population_effective_fitness": True,
    "early_injury_gained_term": True,
    "exact_agent_id_tie_break": True,
    "hardcap_selection": True,
    "cull_refill_selection": True,
    "uniform_early_life_weight_only": True,
    "low_population_reinitialization": False,
    "state_mutation": "cull_release_only",
    "refill_rng": True,
    "exact_cpython_parent_pair_selection": True,
    "exact_numpy_pcg64_global_spawn": True,
    "failure_atomic_refill_planning": True,
    "checkpoint_closure": False,
    "multicell": False,
    "promotion_qualified": False,
}

_MODULE: Any | None = None


def load_cuda_turnover_plan_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("Architecture B turnover planning requires CUDA")
    source = SOURCE_CUDA.read_text(encoding="utf-8")
    suffix = sha256(SOURCE_CUDA.read_bytes() + SOURCE_PCG64.read_bytes()).hexdigest()[:12]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    cpp_source = r"""
    #include <torch/extension.h>

    std::vector<torch::Tensor> eiel_cuda_architecture_b_turnover_plan_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor alive_mask,
        torch::Tensor survival_steps,
        torch::Tensor energy_collected_exact,
        torch::Tensor offspring_count,
        torch::Tensor attacks,
        torch::Tensor early_survival_steps,
        torch::Tensor early_energy_collected_exact,
        torch::Tensor early_injury_gained,
        torch::Tensor early_life_weight,
        torch::Tensor action_switches,
        torch::Tensor action_oscillations,
        torch::Tensor drive_volatility,
        torch::Tensor selfharm_events,
        torch::Tensor pred_err_sum,
        torch::Tensor pred_err_count,
        int64_t population_size,
        double cull_fraction,
        double parent_fraction,
        double coherence_thrashing_weight,
        double coherence_drive_volatility_weight,
        double coherence_selfharm_weight,
        double coherence_pred_err_weight
    );

    std::vector<torch::Tensor> eiel_cuda_architecture_b_turnover_cull_release_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor agent_id_to_slot,
        torch::Tensor alive_mask,
        torch::Tensor pos_y,
        torch::Tensor pos_x,
        torch::Tensor agent_grid,
        torch::Tensor energy,
        torch::Tensor health,
        torch::Tensor policy_energy_observation,
        torch::Tensor action_cache_token,
        torch::Tensor action_cache_cursor,
        torch::Tensor action_cache_next,
        torch::Tensor action_cache_actions,
        torch::Tensor slot_free_queue,
        torch::Tensor slot_free_queue_count,
        torch::Tensor pending_free_agent_ids,
        torch::Tensor pending_free_slots,
        torch::Tensor pending_free_count,
        torch::Tensor culled_agent_ids,
        torch::Tensor culled_slots,
        torch::Tensor culled_count,
        torch::Tensor turnover_plan_status
    );

    std::vector<torch::Tensor> eiel_cuda_architecture_b_turnover_refill_plan_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor alive_mask,
        torch::Tensor agent_grid,
        torch::Tensor parent_pool_slots,
        torch::Tensor parent_pool_count,
        torch::Tensor refill_count,
        torch::Tensor turnover_plan_status,
        torch::Tensor python_random_words,
        torch::Tensor python_random_index,
        torch::Tensor python_random_raw_draws,
        torch::Tensor mutation_py_semantic_draws,
        torch::Tensor env_pcg64_state_words,
        torch::Tensor env_pcg64_buffer_words,
        torch::Tensor policy_np_semantic_draws,
        bool env_policy_alias
    );
    """
    _MODULE = load_inline(
        name=f"eiel_cuda_architecture_b_turnover_plan_{suffix}",
        cpp_sources=[cpp_source],
        cuda_sources=[source],
        functions=[
            "eiel_cuda_architecture_b_turnover_plan_v1",
            "eiel_cuda_architecture_b_turnover_cull_release_v1",
            "eiel_cuda_architecture_b_turnover_refill_plan_v1",
        ],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        extra_include_paths=[str(NATIVE_ROOT)],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def execute_turnover_cull_release_cuda(
    *,
    slot_to_agent_id: torch.Tensor,
    agent_id_to_slot: torch.Tensor,
    alive_mask: torch.Tensor,
    pos_y: torch.Tensor,
    pos_x: torch.Tensor,
    agent_grid: torch.Tensor,
    energy: torch.Tensor,
    health: torch.Tensor,
    policy_energy_observation: torch.Tensor,
    action_cache_token: torch.Tensor,
    action_cache_cursor: torch.Tensor,
    action_cache_next: torch.Tensor,
    action_cache_actions: torch.Tensor,
    slot_free_queue: torch.Tensor,
    slot_free_queue_count: torch.Tensor,
    pending_free_agent_ids: torch.Tensor,
    pending_free_slots: torch.Tensor,
    pending_free_count: torch.Tensor,
    culled_agent_ids: torch.Tensor,
    culled_slots: torch.Tensor,
    culled_count: torch.Tensor,
    turnover_plan_status: torch.Tensor,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Cull planned rows and publish pending/culled slots in CPU-owner order."""

    module = load_cuda_turnover_plan_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_turnover_cull_release_v1(
        slot_to_agent_id,
        agent_id_to_slot,
        alive_mask,
        pos_y,
        pos_x,
        agent_grid,
        energy,
        health,
        policy_energy_observation,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        action_cache_actions,
        slot_free_queue,
        slot_free_queue_count,
        pending_free_agent_ids,
        pending_free_slots,
        pending_free_count,
        culled_agent_ids,
        culled_slots,
        culled_count,
        turnover_plan_status,
    )
    names = (
        "released_agent_ids",
        "released_slots",
        "released_count",
        "pending_released_count",
        "culled_released_count",
        "slot_free_queue_count",
        "status",
    )
    return dict(zip(names, outputs, strict=True))


def plan_generation_turnover_refill_cuda(
    *,
    slot_to_agent_id: torch.Tensor,
    alive_mask: torch.Tensor,
    agent_grid: torch.Tensor,
    parent_pool_slots: torch.Tensor,
    parent_pool_count: torch.Tensor,
    refill_count: torch.Tensor,
    turnover_plan_status: torch.Tensor,
    python_random_words: torch.Tensor,
    python_random_index: torch.Tensor,
    python_random_raw_draws: torch.Tensor,
    mutation_py_semantic_draws: torch.Tensor,
    env_pcg64_state_words: torch.Tensor,
    env_pcg64_buffer_words: torch.Tensor,
    policy_np_semantic_draws: torch.Tensor,
    env_policy_alias: bool,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Plan exact two-parent turnover refill and commit its RNG transitions."""

    module = load_cuda_turnover_plan_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_turnover_refill_plan_v1(
        slot_to_agent_id,
        alive_mask,
        agent_grid,
        parent_pool_slots,
        parent_pool_count,
        refill_count,
        turnover_plan_status,
        python_random_words,
        python_random_index,
        python_random_raw_draws,
        mutation_py_semantic_draws,
        env_pcg64_state_words,
        env_pcg64_buffer_words,
        policy_np_semantic_draws,
        bool(env_policy_alias),
    )
    names = (
        "parent_slots",
        "mate_slots",
        "spawn_positions",
        "spawn_attempts_by_birth",
        "planned_count",
        "python_raw_draws_added",
        "python_semantic_draws_added",
        "spawn_raw_draws_added",
        "spawn_semantic_draws_added",
        "status",
    )
    return dict(zip(names, outputs, strict=True))


def plan_generation_turnover_cuda(
    *,
    slot_to_agent_id: torch.Tensor,
    alive_mask: torch.Tensor,
    survival_steps: torch.Tensor,
    energy_collected_exact: torch.Tensor,
    offspring_count: torch.Tensor,
    attacks: torch.Tensor,
    early_survival_steps: torch.Tensor,
    early_energy_collected_exact: torch.Tensor,
    early_injury_gained: torch.Tensor,
    early_life_weight: torch.Tensor,
    action_switches: torch.Tensor,
    action_oscillations: torch.Tensor,
    drive_volatility: torch.Tensor,
    selfharm_events: torch.Tensor,
    pred_err_sum: torch.Tensor,
    pred_err_count: torch.Tensor,
    population_size: int,
    cull_fraction: float,
    parent_fraction: float,
    coherence_thrashing_weight: float = 0.15,
    coherence_drive_volatility_weight: float = 0.08,
    coherence_selfharm_weight: float = 3.0,
    coherence_pred_err_weight: float = 1.25,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Build the authoritative turnover ranking without mutating resident state."""

    module = load_cuda_turnover_plan_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_turnover_plan_v1(
        slot_to_agent_id,
        alive_mask,
        survival_steps,
        energy_collected_exact,
        offspring_count,
        attacks,
        early_survival_steps,
        early_energy_collected_exact,
        early_injury_gained,
        early_life_weight,
        action_switches,
        action_oscillations,
        drive_volatility,
        selfharm_events,
        pred_err_sum,
        pred_err_count,
        int(population_size),
        float(cull_fraction),
        float(parent_fraction),
        float(coherence_thrashing_weight),
        float(coherence_drive_volatility_weight),
        float(coherence_selfharm_weight),
        float(coherence_pred_err_weight),
    )
    names = (
        "ranked_slots",
        "ranked_agent_ids",
        "effective_fitness",
        "ranked_count",
        "culled_slots",
        "culled_agent_ids",
        "culled_count",
        "parent_pool_slots",
        "parent_pool_agent_ids",
        "parent_pool_count",
        "survivor_count",
        "refill_count",
        "hardcap",
        "status",
    )
    return dict(zip(names, outputs, strict=True))


__all__ = [
    "TURNOVER_CULL_STATUS",
    "TURNOVER_PLAN_CAPABILITIES",
    "TURNOVER_PLAN_STATUS",
    "TURNOVER_REFILL_PLAN_STATUS",
    "load_cuda_turnover_plan_extension",
    "execute_turnover_cull_release_cuda",
    "plan_generation_turnover_cuda",
    "plan_generation_turnover_refill_cuda",
]
