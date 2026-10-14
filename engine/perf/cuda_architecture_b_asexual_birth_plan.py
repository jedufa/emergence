from __future__ import annotations

"""Exact resident CUDA planner for ordinary Flat/Patch asexual births.

This is the pre-commit transaction only. It owns parent ordering, cooldown
decrement/filtering, spawn selection, reservation, energy-buffer checks, free
slot planning, and the birth-spawn PCG64 transition. Child identity, genome,
policy, events, and grid placement are separate transactions.
"""

from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = ROOT / "engine" / "perf" / "native"
SOURCE_CUDA = NATIVE_ROOT / "cuda_architecture_b_asexual_birth_plan.cu"
PCG64_HEADER = NATIVE_ROOT / "cuda_architecture_b_pcg64.cuh"

ASEXUAL_BIRTH_PLAN_STATUS: Final[dict[int, str]] = {
    0: "ok",
    1: "invalid_resident_state",
    2: "invalid_agent_ids",
    3: "unsupported_gru",
    4: "invalid_pcg64_state",
    5: "invalid_config",
    6: "counter_overflow",
}

ASEXUAL_BIRTH_REASON: Final[dict[int, str]] = {
    0: "accepted",
    1: "birth_fail_no_space",
    2: "birth_fail_buffer",
    3: "birth_fail_no_free_slot",
    5: "birth_fail_conception",
    6: "birth_fail_gestation",
    7: "birth_pre_health",
    8: "birth_pre_cooldown",
    9: "birth_pre_no_energy",
}

ASEXUAL_BIRTH_PLAN_CAPABILITY_RECEIPT: Final[dict[str, bool | str]] = {
    "schema_version": "eiel_cuda_architecture_b_asexual_birth_plan_capabilities_v1",
    "decision": "ARCHB-CUDA-STANDALONE-ASEXUAL-BIRTH-PLAN",
    "ordinary_non_turnover_flat_patch_fixed_mlp": True,
    "stable_agent_id_parent_order": True,
    "cooldown_decrement_and_filter": True,
    "north_south_west_east_candidate_order": True,
    "numpy_pcg64_shuffle": "bit_exact",
    "selected_first_reserved_tile_failure": True,
    "energy_buffer_viability": True,
    "free_slot_capacity": True,
    "mate_gate": "explicitly_rejected",
    "injury_gate": "explicitly_rejected",
    "conception_gate": "exact_runner_pcg64",
    "gestation_gate": "exact_runner_pcg64",
    "device_resident_authoritative_state": True,
    "whole_plan_fail_closed": True,
    "identity_commit": False,
    "genome_commit": False,
    "policy_commit": False,
    "event_publication": False,
    "generation_turnover": False,
    "gru": False,
    "jungle": False,
    "wdl": False,
    "promotion_qualified": False,
}

_MODULE: Any | None = None


def load_cuda_asexual_birth_plan_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("Architecture B asexual birth planning requires CUDA")
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    cuda_source = SOURCE_CUDA.read_text(encoding="utf-8")
    source_hash = sha256(
        cuda_source.encode("utf-8") + PCG64_HEADER.read_bytes()
    ).hexdigest()[:12]
    cpp_source = r"""
    #include <torch/extension.h>

    std::vector<torch::Tensor> eiel_cuda_architecture_b_asexual_birth_plan_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor alive_mask,
        torch::Tensor policy_use_gru,
        torch::Tensor pos_y,
        torch::Tensor pos_x,
        torch::Tensor agent_grid,
        torch::Tensor energy,
        torch::Tensor policy_energy_observation,
        torch::Tensor health,
        torch::Tensor injury,
        torch::Tensor scar,
        torch::Tensor repro_cooldown,
        torch::Tensor repro_threshold,
        torch::Tensor slot_free_queue,
        torch::Tensor slot_free_queue_count,
        torch::Tensor slot_free_queue_head,
        torch::Tensor pcg64_state_words,
        torch::Tensor pcg64_buffer_words,
        torch::Tensor policy_np_semantic_draws,
        torch::Tensor runner_pcg64_state_words,
        torch::Tensor runner_pcg64_buffer_words,
        torch::Tensor metrics_np_semantic_draws,
        double parent_energy_share,
        double min_energy_buffer,
        double injury_repro_k_injury,
        double injury_repro_k_scar,
        double injury_birth_fail_scar,
        double injury_birth_fail_injury,
        int64_t max_births,
        bool env_policy_alias,
        bool requires_mate,
        bool injury_gate_active,
        bool conception_gate_active,
        bool gestation_gate_active
    );
    """
    _MODULE = load_inline(
        name=f"eiel_cuda_architecture_b_asexual_birth_plan_{source_hash}",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=["eiel_cuda_architecture_b_asexual_birth_plan_v1"],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(NATIVE_ROOT)],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def execute_asexual_birth_plan_cuda(
    *,
    slot_to_agent_id: torch.Tensor,
    alive_mask: torch.Tensor,
    policy_use_gru: torch.Tensor,
    pos_y: torch.Tensor,
    pos_x: torch.Tensor,
    agent_grid: torch.Tensor,
    energy: torch.Tensor,
    policy_energy_observation: torch.Tensor,
    health: torch.Tensor,
    injury: torch.Tensor,
    scar: torch.Tensor,
    repro_cooldown: torch.Tensor,
    repro_threshold: torch.Tensor,
    slot_free_queue: torch.Tensor,
    slot_free_queue_head: torch.Tensor,
    slot_free_queue_count: torch.Tensor | None = None,
    pcg64_state_words: torch.Tensor,
    pcg64_buffer_words: torch.Tensor,
    policy_np_semantic_draws: torch.Tensor,
    runner_pcg64_state_words: torch.Tensor,
    runner_pcg64_buffer_words: torch.Tensor,
    metrics_np_semantic_draws: torch.Tensor,
    parent_energy_share: float,
    min_energy_buffer: float,
    injury_repro_k_injury: float = 0.0,
    injury_repro_k_scar: float = 0.0,
    injury_birth_fail_scar: float = 0.0,
    injury_birth_fail_injury: float = 0.0,
    max_births: int,
    env_policy_alias: bool = True,
    requires_mate: bool = False,
    injury_gate_active: bool = False,
    conception_gate_active: bool = False,
    gestation_gate_active: bool = False,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Plan ordinary fixed-MLP births without committing child state."""
    if slot_free_queue_count is None:
        slot_free_queue_count = torch.full(
            (1,),
            slot_free_queue.numel(),
            dtype=torch.int64,
            device=slot_free_queue.device,
        )
    module = load_cuda_asexual_birth_plan_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_asexual_birth_plan_v1(
        slot_to_agent_id,
        alive_mask,
        policy_use_gru,
        pos_y,
        pos_x,
        agent_grid,
        energy,
        policy_energy_observation,
        health,
        injury,
        scar,
        repro_cooldown,
        repro_threshold,
        slot_free_queue,
        slot_free_queue_count,
        slot_free_queue_head,
        pcg64_state_words,
        pcg64_buffer_words,
        policy_np_semantic_draws,
        runner_pcg64_state_words,
        runner_pcg64_buffer_words,
        metrics_np_semantic_draws,
        float(parent_energy_share),
        float(min_energy_buffer),
        float(injury_repro_k_injury),
        float(injury_repro_k_scar),
        float(injury_birth_fail_scar),
        float(injury_birth_fail_injury),
        int(max_births),
        bool(env_policy_alias),
        bool(requires_mate),
        bool(injury_gate_active),
        bool(conception_gate_active),
        bool(gestation_gate_active),
    )
    names = (
        "repro_cooldown",
        "slot_free_queue_head",
        "pcg64_state_words",
        "pcg64_buffer_words",
        "policy_np_semantic_draws",
        "parent_slots",
        "child_slots",
        "spawn_positions",
        "child_energy",
        "decision_parent_slots",
        "decision_reason_codes",
        "decision_candidate_counts",
        "decision_draw_starts",
        "decision_draw_ends",
        "decision_selected_positions",
        "decision_selected_reserved",
        "reason_counts",
        "planned_count",
        "decision_count",
        "spawn_rng_draws_added",
        "spawn_rng_semantic_draws_added",
        "status",
        "runner_pcg64_state_words",
        "runner_pcg64_buffer_words",
        "metrics_np_semantic_draws",
        "runner_rng_draws_added",
        "runner_rng_semantic_draws_added",
    )
    return dict(zip(names, outputs, strict=True))


__all__ = [
    "ASEXUAL_BIRTH_PLAN_CAPABILITY_RECEIPT",
    "ASEXUAL_BIRTH_PLAN_STATUS",
    "ASEXUAL_BIRTH_REASON",
    "execute_asexual_birth_plan_cuda",
    "load_cuda_asexual_birth_plan_extension",
]
