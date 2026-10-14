"""Conservative resident-CUDA lifecycle slices for Architecture B.

This surface owns action-time death cull and generation-pending free identity
bookkeeping, controlled asexual birth allocation/identity, and exact fixed-MLP
scalar inheritance. Fresh Gaussian values come from a bounded host-libm tape
prepared at the generation boundary; CUDA still owns branch decisions and the
    authoritative MT19937 transition. Dedicated generation-turnover entrypoints
    add two-parent identity and scalar inheritance, but do not claim the complete
    turnover transaction, reporting, multi-cell batching, Jungle, WDL, or GRU.
"""

from __future__ import annotations

import math
import random
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CUDA = ROOT / "engine" / "perf" / "native" / "cuda_architecture_b_lifecycle.cu"
MAX_CPYTHON_GAUSS_RAW_CAPACITY: Final[int] = 10_549_248
LIFECYCLE_STATUS: Final[dict[int, str]] = {
    0: "ok",
    1: "invalid_pending_count",
    2: "pending_capacity_exceeded",
    10: "invalid_birth_plan",
    11: "invalid_parent_order",
    12: "parent_not_viable",
    13: "no_free_slot",
    14: "agent_id_capacity_exceeded",
    15: "spawn_reservation_failed",
    16: "invalid_free_queue_count",
    20: "invalid_scalar_birth_plan",
    21: "invalid_python_random_state",
    22: "gauss_tape_required_or_exhausted",
    30: "invalid_free_release_state",
    31: "free_release_capacity_exceeded",
    32: "free_release_state_contradiction",
}
FIRST_REMAINING_BIRTH_MISMATCH: Final[str] = (
    "child drives/PCG64 initialization is the first remaining birth mismatch; "
    "fixed-MLP policy rows and Torch RNG initialization follow that boundary"
)
LIFECYCLE_CAPABILITY_RECEIPT: Final[dict[str, bool]] = {
    "single_cell_cuda_resident_state": True,
    "deterministic_action_time_death_cull": True,
    "grid_map_alive_cleanup": True,
    "stable_agent_id_death_order": True,
    "generation_pending_free_bookkeeping": True,
    "ordinary_birth_free_queue_release": True,
    "controlled_birth_allocation_identity_commit": True,
    "deterministic_parent_birth_order": True,
    "free_slot_and_agent_id_allocation": True,
    "spawn_reservation": True,
    "stable_birth_identity_maps_alive_order": True,
    "birth_capacity_fail_closed": True,
    "fixed_mlp_scalar_genome_no_fresh_gauss_commit": True,
    "device_resident_cpython_random_transition": True,
    "fresh_gauss_scalar_mutation": True,
    "generation_boundary_gauss_tape": True,
    "child_scalar_genome_commit": True,
    "turnover_two_parent_identity_commit": True,
    "turnover_two_parent_scalar_inheritance": True,
    "child_drives_pcg64_commit": False,
    "fixed_mlp_policy_row_commit": False,
    "controlled_birth_commit": False,
    "child_genome_policy_rng_commit": False,
    "mutation": False,
    "typed_lifecycle_events": False,
    "generation_turnover": False,
    "checkpoint_continuation": False,
    "multi_cell_batching": False,
    "jungle": False,
    "wdl": False,
    "gru": False,
    "throughput_qualification": False,
}

_MODULE: Any | None = None


def load_cuda_lifecycle_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    cuda_source = SOURCE_CUDA.read_text(encoding="utf-8")
    module_suffix = sha256(cuda_source.encode("utf-8")).hexdigest()[:12]
    cpp_source = r"""
    #include <torch/extension.h>

    std::vector<torch::Tensor> eiel_cuda_architecture_b_lifecycle_death_cull_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor agent_id_to_slot,
        torch::Tensor alive_mask,
        torch::Tensor pos_y,
        torch::Tensor pos_x,
        torch::Tensor agent_grid,
        torch::Tensor energy,
        torch::Tensor health,
        torch::Tensor policy_energy_observation,
        torch::Tensor rest_streak,
        torch::Tensor action_cache_token,
        torch::Tensor action_cache_cursor,
        torch::Tensor action_cache_next,
        torch::Tensor pending_free_agent_ids,
        torch::Tensor pending_free_slots,
        torch::Tensor pending_free_count
    );

    std::vector<torch::Tensor> eiel_cuda_architecture_b_controlled_birth_identity_commit_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor agent_id_to_slot,
        torch::Tensor slot_tokens,
        torch::Tensor alive_mask,
        torch::Tensor pos_y,
        torch::Tensor pos_x,
        torch::Tensor agent_grid,
        torch::Tensor energy,
        torch::Tensor health,
        torch::Tensor injury,
        torch::Tensor scar,
        torch::Tensor age,
        torch::Tensor repro_cooldown,
        torch::Tensor repro_threshold,
        torch::Tensor offspring_count,
        torch::Tensor lineage_founder_id,
        torch::Tensor lineage_depth,
        torch::Tensor lineage_birth_generation,
        torch::Tensor lineage_birth_tick,
        torch::Tensor patch_habitat_lineage_id,
        torch::Tensor policy_energy_observation,
        torch::Tensor survival_steps,
        torch::Tensor rest_streak,
        torch::Tensor action_cache_token,
        torch::Tensor action_cache_cursor,
        torch::Tensor action_cache_next,
        torch::Tensor slot_free_queue,
        torch::Tensor slot_free_queue_count,
        torch::Tensor population_next_id,
        torch::Tensor parent_slots,
        torch::Tensor spawn_positions,
        torch::Tensor planned_count,
        double parent_energy_share,
        double minimum_energy_buffer,
        double start_health,
        int64_t cooldown_steps,
        int64_t generation,
        int64_t tick
    );

    std::vector<torch::Tensor> eiel_cuda_architecture_b_release_pending_free_slots_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor alive_mask,
        torch::Tensor slot_free_queue,
        torch::Tensor slot_free_queue_count,
        torch::Tensor pending_free_slots,
        torch::Tensor pending_free_count,
        torch::Tensor pending_release_cursor
    );

    std::vector<torch::Tensor> eiel_cuda_architecture_b_turnover_refill_identity_commit_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor agent_id_to_slot,
        torch::Tensor slot_tokens,
        torch::Tensor alive_mask,
        torch::Tensor pos_y,
        torch::Tensor pos_x,
        torch::Tensor agent_grid,
        torch::Tensor energy,
        torch::Tensor health,
        torch::Tensor injury,
        torch::Tensor scar,
        torch::Tensor age,
        torch::Tensor repro_cooldown,
        torch::Tensor repro_threshold,
        torch::Tensor offspring_count,
        torch::Tensor lineage_founder_id,
        torch::Tensor lineage_depth,
        torch::Tensor lineage_birth_generation,
        torch::Tensor lineage_birth_tick,
        torch::Tensor patch_habitat_lineage_id,
        torch::Tensor policy_energy_observation,
        torch::Tensor survival_steps,
        torch::Tensor rest_streak,
        torch::Tensor action_cache_token,
        torch::Tensor action_cache_cursor,
        torch::Tensor action_cache_next,
        torch::Tensor slot_free_queue,
        torch::Tensor slot_free_queue_count,
        torch::Tensor population_next_id,
        torch::Tensor parent_slots,
        torch::Tensor mate_slots,
        torch::Tensor spawn_positions,
        torch::Tensor planned_count,
        double start_energy,
        double start_health,
        int64_t generation
    );

    std::vector<torch::Tensor> eiel_cuda_architecture_b_fixed_mlp_child_scalar_genome_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor alive_mask,
        torch::Tensor parent_slots,
        torch::Tensor child_slots,
        torch::Tensor planned_count,
        torch::Tensor python_random_words,
        torch::Tensor python_random_index,
        torch::Tensor python_random_has_gauss,
        torch::Tensor python_random_gauss_next,
        torch::Tensor python_random_gauss_primary_tape,
        torch::Tensor python_random_gauss_secondary_tape,
        torch::Tensor python_random_raw_draws,
        torch::Tensor mutation_py_semantic_draws,
        std::vector<torch::Tensor> genome_int32_fields,
        torch::Tensor policy_mem_dim,
        torch::Tensor policy_use_gru,
        std::vector<torch::Tensor> genome_float32_fields,
        int64_t sensor_radius_max
    );

    std::vector<torch::Tensor> eiel_cuda_architecture_b_turnover_fixed_mlp_child_scalar_genome_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor alive_mask,
        torch::Tensor parent_slots,
        torch::Tensor mate_slots,
        torch::Tensor child_slots,
        torch::Tensor planned_count,
        torch::Tensor python_random_words,
        torch::Tensor python_random_index,
        torch::Tensor python_random_has_gauss,
        torch::Tensor python_random_gauss_next,
        torch::Tensor python_random_gauss_primary_tape,
        torch::Tensor python_random_gauss_secondary_tape,
        torch::Tensor python_random_raw_draws,
        torch::Tensor mutation_py_semantic_draws,
        std::vector<torch::Tensor> genome_int32_fields,
        torch::Tensor policy_mem_dim,
        torch::Tensor policy_use_gru,
        std::vector<torch::Tensor> genome_float32_fields,
        int64_t sensor_radius_max
    );
    """
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load_inline(
        name=f"eiel_cuda_architecture_b_lifecycle_{module_suffix}",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=[
            "eiel_cuda_architecture_b_lifecycle_death_cull_v1",
            "eiel_cuda_architecture_b_release_pending_free_slots_v1",
            "eiel_cuda_architecture_b_controlled_birth_identity_commit_v1",
            "eiel_cuda_architecture_b_fixed_mlp_child_scalar_genome_v1",
            "eiel_cuda_architecture_b_turnover_refill_identity_commit_v1",
            "eiel_cuda_architecture_b_turnover_fixed_mlp_child_scalar_genome_v1",
        ],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def execute_lifecycle_death_cull_cuda(
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
    rest_streak: torch.Tensor,
    action_cache_token: torch.Tensor,
    action_cache_cursor: torch.Tensor,
    action_cache_next: torch.Tensor,
    pending_free_agent_ids: torch.Tensor,
    pending_free_slots: torch.Tensor,
    pending_free_count: torch.Tensor,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Cull all terminal rows in one native call without a host projection."""
    module = load_cuda_lifecycle_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_lifecycle_death_cull_v1(
        slot_to_agent_id,
        agent_id_to_slot,
        alive_mask,
        pos_y,
        pos_x,
        agent_grid,
        energy,
        health,
        policy_energy_observation,
        rest_streak,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        pending_free_agent_ids,
        pending_free_slots,
        pending_free_count,
    )
    names = (
        "death_agent_ids",
        "death_slots",
        "death_reason_codes",
        "death_count",
        "alive_agent_ids_by_order",
        "alive_count",
        "status",
    )
    return dict(zip(names, outputs, strict=True))


def release_pending_free_slots_cuda(
    *,
    slot_to_agent_id: torch.Tensor,
    alive_mask: torch.Tensor,
    slot_free_queue: torch.Tensor,
    slot_free_queue_count: torch.Tensor,
    pending_free_slots: torch.Tensor,
    pending_free_count: torch.Tensor,
    pending_release_cursor: torch.Tensor,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Publish newly dead slots to the ordinary-birth free queue in order."""

    module = load_cuda_lifecycle_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_release_pending_free_slots_v1(
        slot_to_agent_id,
        alive_mask,
        slot_free_queue,
        slot_free_queue_count,
        pending_free_slots,
        pending_free_count,
        pending_release_cursor,
    )
    return dict(
        zip(
            (
                "slot_free_queue_count",
                "pending_release_cursor",
                "appended_count",
                "status",
            ),
            outputs,
            strict=True,
        )
    )


def execute_controlled_birth_commit_cuda(
    *,
    slot_to_agent_id: torch.Tensor,
    agent_id_to_slot: torch.Tensor,
    slot_tokens: torch.Tensor,
    alive_mask: torch.Tensor,
    pos_y: torch.Tensor,
    pos_x: torch.Tensor,
    agent_grid: torch.Tensor,
    energy: torch.Tensor,
    health: torch.Tensor,
    injury: torch.Tensor,
    scar: torch.Tensor,
    age: torch.Tensor,
    repro_cooldown: torch.Tensor,
    repro_threshold: torch.Tensor,
    offspring_count: torch.Tensor,
    lineage_founder_id: torch.Tensor,
    lineage_depth: torch.Tensor,
    lineage_birth_generation: torch.Tensor,
    lineage_birth_tick: torch.Tensor,
    patch_habitat_lineage_id: torch.Tensor,
    policy_energy_observation: torch.Tensor,
    survival_steps: torch.Tensor,
    rest_streak: torch.Tensor,
    action_cache_token: torch.Tensor,
    action_cache_cursor: torch.Tensor,
    action_cache_next: torch.Tensor,
    slot_free_queue: torch.Tensor,
    slot_free_queue_count: torch.Tensor,
    population_next_id: torch.Tensor,
    parent_slots: torch.Tensor,
    spawn_positions: torch.Tensor,
    parent_energy_share: float,
    minimum_energy_buffer: float,
    start_health: float,
    cooldown_steps: int,
    generation: int,
    tick: int,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Commit the exact controlled birth allocation/identity sub-transaction.

    Parent order and spawn coordinates are authoritative plan inputs. The call
    validates every parent, slot, ID, and reservation before mutating state.
    Child genome, policy, RNG, events, and reporting remain a separate blocked
    transaction and must complete before this state can enter trajectory use.
    """
    module = load_cuda_lifecycle_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_controlled_birth_identity_commit_v1(
        slot_to_agent_id,
        agent_id_to_slot,
        slot_tokens,
        alive_mask,
        pos_y,
        pos_x,
        agent_grid,
        energy,
        health,
        injury,
        scar,
        age,
        repro_cooldown,
        repro_threshold,
        offspring_count,
        lineage_founder_id,
        lineage_depth,
        lineage_birth_generation,
        lineage_birth_tick,
        patch_habitat_lineage_id,
        policy_energy_observation,
        survival_steps,
        rest_streak,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        slot_free_queue,
        slot_free_queue_count,
        population_next_id,
        parent_slots,
        spawn_positions,
        torch.full(
            (1,),
            parent_slots.numel(),
            dtype=torch.int64,
            device=parent_slots.device,
        ),
        float(parent_energy_share),
        float(minimum_energy_buffer),
        float(start_health),
        int(cooldown_steps),
        int(generation),
        int(tick),
    )
    names = (
        "parent_agent_ids",
        "child_agent_ids",
        "child_slots",
        "committed_count",
        "alive_agent_ids_by_order",
        "alive_count",
        "status",
    )
    return dict(zip(names, outputs, strict=True))


def execute_turnover_refill_identity_commit_cuda(
    *,
    slot_to_agent_id: torch.Tensor,
    agent_id_to_slot: torch.Tensor,
    slot_tokens: torch.Tensor,
    alive_mask: torch.Tensor,
    pos_y: torch.Tensor,
    pos_x: torch.Tensor,
    agent_grid: torch.Tensor,
    energy: torch.Tensor,
    health: torch.Tensor,
    injury: torch.Tensor,
    scar: torch.Tensor,
    age: torch.Tensor,
    repro_cooldown: torch.Tensor,
    repro_threshold: torch.Tensor,
    offspring_count: torch.Tensor,
    lineage_founder_id: torch.Tensor,
    lineage_depth: torch.Tensor,
    lineage_birth_generation: torch.Tensor,
    lineage_birth_tick: torch.Tensor,
    patch_habitat_lineage_id: torch.Tensor,
    policy_energy_observation: torch.Tensor,
    survival_steps: torch.Tensor,
    rest_streak: torch.Tensor,
    action_cache_token: torch.Tensor,
    action_cache_cursor: torch.Tensor,
    action_cache_next: torch.Tensor,
    slot_free_queue: torch.Tensor,
    slot_free_queue_count: torch.Tensor,
    population_next_id: torch.Tensor,
    parent_slots: torch.Tensor,
    mate_slots: torch.Tensor,
    spawn_positions: torch.Tensor,
    planned_count: torch.Tensor,
    start_energy: float,
    start_health: float,
    generation: int,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Commit exact two-parent turnover identities without parent energy debit."""

    module = load_cuda_lifecycle_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_turnover_refill_identity_commit_v1(
        slot_to_agent_id,
        agent_id_to_slot,
        slot_tokens,
        alive_mask,
        pos_y,
        pos_x,
        agent_grid,
        energy,
        health,
        injury,
        scar,
        age,
        repro_cooldown,
        repro_threshold,
        offspring_count,
        lineage_founder_id,
        lineage_depth,
        lineage_birth_generation,
        lineage_birth_tick,
        patch_habitat_lineage_id,
        policy_energy_observation,
        survival_steps,
        rest_streak,
        action_cache_token,
        action_cache_cursor,
        action_cache_next,
        slot_free_queue,
        slot_free_queue_count,
        population_next_id,
        parent_slots,
        mate_slots,
        spawn_positions,
        planned_count,
        float(start_energy),
        float(start_health),
        int(generation),
    )
    names = (
        "parent_agent_ids",
        "mate_agent_ids",
        "child_agent_ids",
        "child_slots",
        "committed_count",
        "alive_agent_ids_by_order",
        "alive_count",
        "status",
    )
    return dict(zip(names, outputs, strict=True))


def _cpython_gauss_tape(
    state: tuple[object, ...],
    raw_capacity: int,
    *,
    raw_offset: int = 0,
) -> tuple[list[float], list[float]]:
    if raw_capacity < 0:
        raise ValueError("Gaussian tape raw capacity must be nonnegative")
    if raw_offset < 0:
        raise ValueError("Gaussian tape raw offset must be nonnegative")
    if raw_offset + raw_capacity > MAX_CPYTHON_GAUSS_RAW_CAPACITY:
        raise ValueError("Gaussian tape raw capacity exceeds the bounded limit")
    if raw_capacity == 0:
        return [], []

    source = random.Random()
    source.setstate(state)
    raw_words = [source.getrandbits(32) for _ in range(raw_capacity + 3)]
    # Device RNG counters are absolute across checkpoint boundaries, while this
    # tape is generated from the boundary's already-advanced CPython state.
    # Preserve absolute indexing without skipping the boundary-local words.
    primary: list[float] = [0.0] * raw_offset
    secondary: list[float] = [0.0] * raw_offset
    double_scale = 1.0 / 9007199254740992.0
    for offset in range(raw_capacity):
        first = (
            (raw_words[offset] >> 5) * 67108864.0
            + (raw_words[offset + 1] >> 6)
        ) * double_scale
        second = (
            (raw_words[offset + 2] >> 5) * 67108864.0
            + (raw_words[offset + 3] >> 6)
        ) * double_scale
        x2pi = first * math.tau
        radius = math.sqrt(-2.0 * math.log(1.0 - second))
        primary.append(math.cos(x2pi) * radius)
        secondary.append(math.sin(x2pi) * radius)
    return primary, secondary


def pack_cpython_random_state_cuda(
    state: tuple[object, ...],
    *,
    device: torch.device | str,
    semantic_draws: int = 0,
    gauss_tape_raw_capacity: int = 0,
    raw_draws: int = 0,
    raw_draw_base: int = 0,
) -> dict[str, torch.Tensor]:
    """Pack ``random.Random.getstate()`` at a host/device ownership boundary."""
    if len(state) != 3 or int(state[0]) != 3:
        raise ValueError("CPython random state version 3 is required")
    internal = tuple(state[1])
    if len(internal) != 625:
        raise ValueError("CPython random state must contain 624 words and an index")
    index = int(internal[-1])
    if not 0 <= index <= 624:
        raise ValueError("CPython random state index is out of range")
    if isinstance(raw_draws, bool) or not isinstance(raw_draws, int) or raw_draws < 0:
        raise ValueError("CPython random raw draw count must be a non-negative integer")
    if (
        isinstance(raw_draw_base, bool)
        or not isinstance(raw_draw_base, int)
        or raw_draw_base < 0
    ):
        raise ValueError("CPython random raw draw base must be a non-negative integer")
    if raw_draw_base > raw_draws:
        raise ValueError("CPython random raw draw base must not exceed raw draw count")
    gauss_next = state[2]
    gauss_primary, gauss_secondary = _cpython_gauss_tape(
        state,
        int(gauss_tape_raw_capacity),
        raw_offset=int(raw_draws - raw_draw_base),
    )
    return {
        "python_random_words": torch.tensor(
            internal[:-1], dtype=torch.uint32, device=device
        ).contiguous(),
        "python_random_index": torch.tensor(
            [index], dtype=torch.int64, device=device
        ),
        "python_random_has_gauss": torch.tensor(
            [gauss_next is not None], dtype=torch.bool, device=device
        ),
        "python_random_gauss_next": torch.tensor(
            [0.0 if gauss_next is None else float(gauss_next)],
            dtype=torch.float64,
            device=device,
        ),
        "python_random_gauss_primary_tape": torch.tensor(
            gauss_primary,
            dtype=torch.float64,
            device=device,
        ),
        "python_random_gauss_secondary_tape": torch.tensor(
            gauss_secondary,
            dtype=torch.float64,
            device=device,
        ),
        "python_random_raw_draws": torch.tensor(
            [raw_draws], dtype=torch.int64, device=device
        ),
        "mutation_py_semantic_draws": torch.tensor(
            [int(semantic_draws)], dtype=torch.int64, device=device
        ),
    }


def execute_fixed_mlp_child_scalar_genome_cuda(
    *,
    slot_to_agent_id: torch.Tensor,
    alive_mask: torch.Tensor,
    parent_slots: torch.Tensor,
    child_slots: torch.Tensor,
    python_random_words: torch.Tensor,
    python_random_index: torch.Tensor,
    python_random_has_gauss: torch.Tensor,
    python_random_gauss_next: torch.Tensor,
    python_random_gauss_primary_tape: torch.Tensor,
    python_random_gauss_secondary_tape: torch.Tensor,
    python_random_raw_draws: torch.Tensor,
    mutation_py_semantic_draws: torch.Tensor,
    genome_hidden_size: torch.Tensor,
    genome_memory_dim: torch.Tensor,
    genome_sensor_channel_mask: torch.Tensor,
    genome_sensor_radius: torch.Tensor,
    genome_introspection_channel_mask: torch.Tensor,
    genome_self_regulation_control_mask: torch.Tensor,
    genome_self_modification_target_mask: torch.Tensor,
    genome_directed_inheritance_channel_mask: torch.Tensor,
    policy_mem_dim: torch.Tensor,
    policy_use_gru: torch.Tensor,
    genome_exploration_bias: torch.Tensor,
    genome_aggression_bias: torch.Tensor,
    genome_mutation_rate: torch.Tensor,
    genome_reproduction_threshold: torch.Tensor,
    genome_action_temperature: torch.Tensor,
    genome_drive_inertia: torch.Tensor,
    genome_drive_noise: torch.Tensor,
    genome_drive_logit_gain: torch.Tensor,
    genome_pred_lr: torch.Tensor,
    genome_drive_sp_hunger: torch.Tensor,
    genome_drive_sp_pain: torch.Tensor,
    genome_drive_sp_fatigue: torch.Tensor,
    genome_drive_sp_fear: torch.Tensor,
    genome_exp_prior_strength: torch.Tensor,
    genome_exp_prior_noise: torch.Tensor,
    genome_drive_birth_noise: torch.Tensor,
    repro_threshold: torch.Tensor,
    gru_upkeep_cost: torch.Tensor,
    policy_attack_bias: torch.Tensor,
    policy_action_temperature: torch.Tensor,
    policy_drive_logit_gain: torch.Tensor,
    drive_inertia: torch.Tensor,
    drive_noise: torch.Tensor,
    pred_lr: torch.Tensor,
    sensor_radius_max: int,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Commit exact asexual fixed-MLP scalar inheritance in one native call.

    The device preflights the complete birth batch on a private MT19937 copy.
    Fresh Gaussian values are selected by the exact raw-word offset from a
    bounded host-libm tape prepared at the generation boundary. A missing or
    exhausted tape returns status 22 without changing genome or RNG state.
    """
    module = load_cuda_lifecycle_extension(build_directory=build_directory)
    int32_fields = [
        genome_hidden_size,
        genome_memory_dim,
        genome_sensor_channel_mask,
        genome_sensor_radius,
        genome_introspection_channel_mask,
        genome_self_regulation_control_mask,
        genome_self_modification_target_mask,
        genome_directed_inheritance_channel_mask,
    ]
    float32_fields = [
        genome_exploration_bias,
        genome_aggression_bias,
        genome_mutation_rate,
        genome_reproduction_threshold,
        genome_action_temperature,
        genome_drive_inertia,
        genome_drive_noise,
        genome_drive_logit_gain,
        genome_pred_lr,
        genome_drive_sp_hunger,
        genome_drive_sp_pain,
        genome_drive_sp_fatigue,
        genome_drive_sp_fear,
        genome_exp_prior_strength,
        genome_exp_prior_noise,
        genome_drive_birth_noise,
        repro_threshold,
        gru_upkeep_cost,
        policy_attack_bias,
        policy_action_temperature,
        policy_drive_logit_gain,
        drive_inertia,
        drive_noise,
        pred_lr,
    ]
    outputs = module.eiel_cuda_architecture_b_fixed_mlp_child_scalar_genome_v1(
        slot_to_agent_id,
        alive_mask,
        parent_slots,
        child_slots,
        torch.full(
            (1,),
            parent_slots.numel(),
            dtype=torch.int64,
            device=parent_slots.device,
        ),
        python_random_words,
        python_random_index,
        python_random_has_gauss,
        python_random_gauss_next,
        python_random_gauss_primary_tape,
        python_random_gauss_secondary_tape,
        python_random_raw_draws,
        mutation_py_semantic_draws,
        int32_fields,
        policy_mem_dim,
        policy_use_gru,
        float32_fields,
        int(sensor_radius_max),
    )
    return dict(zip(("committed_count", "semantic_draws_added", "status"), outputs, strict=True))


def execute_turnover_fixed_mlp_child_scalar_genome_cuda(
    *,
    slot_to_agent_id: torch.Tensor,
    alive_mask: torch.Tensor,
    parent_slots: torch.Tensor,
    mate_slots: torch.Tensor,
    child_slots: torch.Tensor,
    planned_count: torch.Tensor,
    python_random_words: torch.Tensor,
    python_random_index: torch.Tensor,
    python_random_has_gauss: torch.Tensor,
    python_random_gauss_next: torch.Tensor,
    python_random_gauss_primary_tape: torch.Tensor,
    python_random_gauss_secondary_tape: torch.Tensor,
    python_random_raw_draws: torch.Tensor,
    mutation_py_semantic_draws: torch.Tensor,
    genome_hidden_size: torch.Tensor,
    genome_memory_dim: torch.Tensor,
    genome_sensor_channel_mask: torch.Tensor,
    genome_sensor_radius: torch.Tensor,
    genome_introspection_channel_mask: torch.Tensor,
    genome_self_regulation_control_mask: torch.Tensor,
    genome_self_modification_target_mask: torch.Tensor,
    genome_directed_inheritance_channel_mask: torch.Tensor,
    policy_mem_dim: torch.Tensor,
    policy_use_gru: torch.Tensor,
    genome_exploration_bias: torch.Tensor,
    genome_aggression_bias: torch.Tensor,
    genome_mutation_rate: torch.Tensor,
    genome_reproduction_threshold: torch.Tensor,
    genome_action_temperature: torch.Tensor,
    genome_drive_inertia: torch.Tensor,
    genome_drive_noise: torch.Tensor,
    genome_drive_logit_gain: torch.Tensor,
    genome_pred_lr: torch.Tensor,
    genome_drive_sp_hunger: torch.Tensor,
    genome_drive_sp_pain: torch.Tensor,
    genome_drive_sp_fatigue: torch.Tensor,
    genome_drive_sp_fear: torch.Tensor,
    genome_exp_prior_strength: torch.Tensor,
    genome_exp_prior_noise: torch.Tensor,
    genome_drive_birth_noise: torch.Tensor,
    repro_threshold: torch.Tensor,
    gru_upkeep_cost: torch.Tensor,
    policy_attack_bias: torch.Tensor,
    policy_action_temperature: torch.Tensor,
    policy_drive_logit_gain: torch.Tensor,
    drive_inertia: torch.Tensor,
    drive_noise: torch.Tensor,
    pred_lr: torch.Tensor,
    sensor_radius_max: int,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Commit exact two-parent fixed-MLP turnover scalar inheritance."""

    module = load_cuda_lifecycle_extension(build_directory=build_directory)
    int32_fields = [
        genome_hidden_size,
        genome_memory_dim,
        genome_sensor_channel_mask,
        genome_sensor_radius,
        genome_introspection_channel_mask,
        genome_self_regulation_control_mask,
        genome_self_modification_target_mask,
        genome_directed_inheritance_channel_mask,
    ]
    float32_fields = [
        genome_exploration_bias,
        genome_aggression_bias,
        genome_mutation_rate,
        genome_reproduction_threshold,
        genome_action_temperature,
        genome_drive_inertia,
        genome_drive_noise,
        genome_drive_logit_gain,
        genome_pred_lr,
        genome_drive_sp_hunger,
        genome_drive_sp_pain,
        genome_drive_sp_fatigue,
        genome_drive_sp_fear,
        genome_exp_prior_strength,
        genome_exp_prior_noise,
        genome_drive_birth_noise,
        repro_threshold,
        gru_upkeep_cost,
        policy_attack_bias,
        policy_action_temperature,
        policy_drive_logit_gain,
        drive_inertia,
        drive_noise,
        pred_lr,
    ]
    outputs = module.eiel_cuda_architecture_b_turnover_fixed_mlp_child_scalar_genome_v1(
        slot_to_agent_id,
        alive_mask,
        parent_slots,
        mate_slots,
        child_slots,
        planned_count,
        python_random_words,
        python_random_index,
        python_random_has_gauss,
        python_random_gauss_next,
        python_random_gauss_primary_tape,
        python_random_gauss_secondary_tape,
        python_random_raw_draws,
        mutation_py_semantic_draws,
        int32_fields,
        policy_mem_dim,
        policy_use_gru,
        float32_fields,
        int(sensor_radius_max),
    )
    return dict(zip(("committed_count", "semantic_draws_added", "status"), outputs, strict=True))


__all__ = [
    "LIFECYCLE_CAPABILITY_RECEIPT",
    "LIFECYCLE_STATUS",
    "FIRST_REMAINING_BIRTH_MISMATCH",
    "execute_controlled_birth_commit_cuda",
    "execute_fixed_mlp_child_scalar_genome_cuda",
    "execute_turnover_fixed_mlp_child_scalar_genome_cuda",
    "execute_turnover_refill_identity_commit_cuda",
    "execute_lifecycle_death_cull_cuda",
    "release_pending_free_slots_cuda",
    "load_cuda_lifecycle_extension",
    "pack_cpython_random_state_cuda",
]
