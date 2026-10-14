from __future__ import annotations

"""Atomic resident CUDA fixed-MLP birth transactions for Architecture B."""

from hashlib import sha256
from pathlib import Path
from typing import Any, Final, Mapping

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
PERF_ROOT = ROOT / "engine" / "perf"
NATIVE_ROOT = PERF_ROOT / "native"
SOURCE_CPP = NATIVE_ROOT / "cuda_architecture_b_birth_transaction.cpp"
SOURCE_LIFECYCLE = NATIVE_ROOT / "cuda_architecture_b_lifecycle.cu"
SOURCE_DRIVES = NATIVE_ROOT / "cuda_architecture_b_child_drives.cu"
SOURCE_POLICY = NATIVE_ROOT / "cuda_architecture_b_child_policy.cu"
SOURCE_PCG64 = NATIVE_ROOT / "cuda_architecture_b_pcg64.cuh"
SOURCE_TORCH_RNG_PACK = PERF_ROOT / "cuda_architecture_b_torch_rng_state_pack_v1.cuh"

IDENTITY_STATE_FIELDS: Final[tuple[str, ...]] = (
    "slot_to_agent_id",
    "agent_id_to_slot",
    "slot_tokens",
    "alive_mask",
    "pos_y",
    "pos_x",
    "agent_grid",
    "energy",
    "health",
    "injury",
    "scar",
    "age",
    "repro_cooldown",
    "repro_threshold",
    "offspring_count",
    "lineage_founder_id",
    "lineage_depth",
    "lineage_birth_generation",
    "lineage_birth_tick",
    "patch_habitat_lineage_id",
    "policy_energy_observation",
    "survival_steps",
    "rest_streak",
    "action_cache_token",
    "action_cache_cursor",
    "action_cache_next",
    "slot_free_queue",
    "slot_free_queue_count",
    "population_next_id",
    "last_action",
    "last_last_action",
    "action_switches",
    "action_oscillations",
    "episode0_switches",
    "episode0_oscillations",
)

SCALAR_RNG_STATE_FIELDS: Final[tuple[str, ...]] = (
    "python_random_words",
    "python_random_index",
    "python_random_has_gauss",
    "python_random_gauss_next",
    "python_random_gauss_primary_tape",
    "python_random_gauss_secondary_tape",
    "python_random_raw_draws",
    "mutation_py_semantic_draws",
)

GENOME_INT32_FIELDS: Final[tuple[str, ...]] = (
    "genome_hidden_size",
    "genome_memory_dim",
    "genome_sensor_channel_mask",
    "genome_sensor_radius",
    "genome_introspection_channel_mask",
    "genome_self_regulation_control_mask",
    "genome_self_modification_target_mask",
    "genome_directed_inheritance_channel_mask",
)

GENOME_FLOAT32_FIELDS: Final[tuple[str, ...]] = (
    "genome_exploration_bias",
    "genome_aggression_bias",
    "genome_mutation_rate",
    "genome_reproduction_threshold",
    "genome_action_temperature",
    "genome_drive_inertia",
    "genome_drive_noise",
    "genome_drive_logit_gain",
    "genome_pred_lr",
    "genome_drive_sp_hunger",
    "genome_drive_sp_pain",
    "genome_drive_sp_fatigue",
    "genome_drive_sp_fear",
    "genome_exp_prior_strength",
    "genome_exp_prior_noise",
    "genome_drive_birth_noise",
    "repro_threshold",
    "gru_upkeep_cost",
    "policy_attack_bias",
    "policy_action_temperature",
    "policy_drive_logit_gain",
    "drive_inertia",
    "drive_noise",
    "pred_lr",
)

DRIVE_STATE_FIELDS: Final[tuple[str, ...]] = (
    "drives",
    "birth_reset_shock",
    "pcg64_state_words",
    "pcg64_buffer_words",
    "mutation_np_semantic_draws",
    "chemistry",
)

POLICY_STATE_FIELDS: Final[tuple[str, ...]] = (
    "mlp_pack",
    "hmem",
    "rng_manager_mutation_torch_cpu_state",
    "policy_birth_delta_tape",
    "policy_birth_mutation_mask_tape",
    "policy_birth_crossover_mask_packed_tape",
    "policy_birth_post_rng_state_packed_tape",
    "policy_birth_semantic_draws_tape",
    "policy_birth_tape_cursor",
    "mutation_torch_cpu_semantic_draws",
)

_ORDINARY_POLICY_STATE_FIELDS: Final[tuple[str, ...]] = tuple(
    field
    for field in POLICY_STATE_FIELDS
    if field != "policy_birth_crossover_mask_packed_tape"
)

BIRTH_TRANSACTION_STATUS: Final[dict[int, str]] = {
    0: "ok",
    1: "identity_or_slot_allocation_failed",
    2: "scalar_inheritance_or_cpython_rng_failed",
    3: "drive_initialization_or_pcg64_failed",
    4: "policy_inheritance_or_torch_rng_failed",
}

BIRTH_TRANSACTION_CAPABILITIES: Final[dict[str, bool | str]] = {
    "schema_version": "eiel_cuda_architecture_b_birth_transaction_capabilities_v1",
    "decision": "ARCHB-CUDA-STANDALONE-FIXED-MLP-BIRTH-TRANSACTION",
    "authoritative_order": "identity_scalar_drives_policy",
    "resident_cuda_state": True,
    "one_native_transaction_call": True,
    "python_per_birth_loop": False,
    "device_resident_plan_count": True,
    "per_tick_host_sync": False,
    "stable_authoritative_tensor_pointers": True,
    "whole_batch_atomic_commit": True,
    "generation_turnover_refill_transaction": True,
    "generation_turnover_two_parent_crossover": True,
    "generation_turnover_parent_energy_debit": False,
    "fresh_agent_stats_reset": False,
    "selection_stat_accumulator_reset": False,
    "exact_cpython_rng_continuation": True,
    "exact_numpy_pcg64_continuation": True,
    "exact_torch_cpu_mutation_rng_continuation": True,
    "fixed_mlp_flat_patch": True,
    "generation_boundary_random_tapes": True,
    "typed_birth_events": False,
    "generation_integration": False,
    "gru": False,
    "jungle": False,
    "wdl": False,
    "multicell": False,
    "promotion_qualified": False,
}

_MODULE: Any | None = None


def _ordered_tensors(
    state: Mapping[str, torch.Tensor],
    fields: tuple[str, ...],
    *,
    group: str,
) -> list[torch.Tensor]:
    missing = [field for field in fields if field not in state]
    if missing:
        raise KeyError(f"{group} is missing fields: {', '.join(missing)}")
    return [state[field] for field in fields]


def load_cuda_birth_transaction_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("Architecture B birth transaction requires CUDA")
    sources = (
        SOURCE_CPP,
        SOURCE_LIFECYCLE,
        SOURCE_DRIVES,
        SOURCE_POLICY,
    )
    source_hash = sha256(
        b"\0".join(
            source.read_bytes()
            for source in (*sources, SOURCE_PCG64, SOURCE_TORCH_RNG_PACK)
        )
    ).hexdigest()[:12]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=f"eiel_cuda_architecture_b_birth_transaction_{source_hash}",
        sources=[str(source) for source in sources],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(NATIVE_ROOT), str(PERF_ROOT)],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def execute_fixed_mlp_birth_transaction_cuda(
    *,
    identity_state: Mapping[str, torch.Tensor],
    parent_slots: torch.Tensor,
    spawn_positions: torch.Tensor,
    planned_count: torch.Tensor | None = None,
    scalar_rng_state: Mapping[str, torch.Tensor],
    genome_state: Mapping[str, torch.Tensor],
    policy_mem_dim: torch.Tensor,
    policy_use_gru: torch.Tensor,
    drive_state: Mapping[str, torch.Tensor],
    policy_state: Mapping[str, torch.Tensor],
    parent_energy_share: float,
    minimum_energy_buffer: float,
    start_health: float,
    cooldown_steps: int,
    generation: int,
    tick: int,
    sensor_radius_max: int,
    max_energy: float,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Commit one exact fixed-MLP birth batch or leave all state unchanged.

    The native extension stages identity/slot allocation, scalar inheritance,
    newborn drives, and policy-row inheritance in CPU-owner order. Device-side
    status gating copies staged values into the authoritative tensors only when
    all four component transactions succeed. Random tapes are prepared only at
    generation boundaries; there is no Python loop over births. ``planned_count``
    may remain CUDA-resident while the fixed-capacity plan tail is ignored.
    """
    repro_threshold = identity_state.get("repro_threshold")
    if repro_threshold is None or genome_state.get("repro_threshold") is not repro_threshold:
        raise ValueError(
            "identity_state and genome_state must share the authoritative "
            "repro_threshold tensor"
        )
    module = load_cuda_birth_transaction_extension(build_directory=build_directory)
    if planned_count is None:
        planned_count = torch.full(
            (1,),
            parent_slots.numel(),
            dtype=torch.int64,
            device=parent_slots.device,
        )
    outputs = module.eiel_cuda_architecture_b_fixed_mlp_birth_transaction_v1(
        _ordered_tensors(identity_state, IDENTITY_STATE_FIELDS, group="identity_state"),
        parent_slots,
        spawn_positions,
        planned_count,
        _ordered_tensors(
            scalar_rng_state,
            SCALAR_RNG_STATE_FIELDS,
            group="scalar_rng_state",
        ),
        _ordered_tensors(genome_state, GENOME_INT32_FIELDS, group="genome_state"),
        policy_mem_dim,
        policy_use_gru,
        _ordered_tensors(genome_state, GENOME_FLOAT32_FIELDS, group="genome_state"),
        _ordered_tensors(drive_state, DRIVE_STATE_FIELDS, group="drive_state"),
        _ordered_tensors(
            policy_state,
            _ORDINARY_POLICY_STATE_FIELDS,
            group="policy_state",
        ),
        float(parent_energy_share),
        float(minimum_energy_buffer),
        float(start_health),
        int(cooldown_steps),
        int(generation),
        int(tick),
        int(sensor_radius_max),
        float(max_energy),
    )
    names = (
        "parent_agent_ids",
        "child_agent_ids",
        "child_slots",
        "committed_count",
        "alive_agent_ids_by_order",
        "alive_count",
        "scalar_semantic_draws_added",
        "population_birth_noise_rows_consumed",
        "population_exp_prior_noise_rows_consumed",
        "native_population_standard_normal_rows_generated",
        "drive_semantic_draws_added",
        "drive_raw64_draws_added",
        "policy_semantic_draws_added",
        "policy_mutation_events",
        "policy_mutated_elements",
        "policy_mutation_delta_total",
        "policy_mutation_occurred_by_birth",
        "policy_mutation_delta_by_birth",
        "identity_status",
        "scalar_status",
        "drive_status",
        "policy_status",
        "status",
    )
    return dict(zip(names, outputs, strict=True))


def execute_fixed_mlp_turnover_refill_transaction_cuda(
    *,
    identity_state: Mapping[str, torch.Tensor],
    parent_pool_slots: torch.Tensor,
    parent_pool_count: torch.Tensor,
    spawn_positions: torch.Tensor,
    planned_count: torch.Tensor,
    scalar_rng_state: Mapping[str, torch.Tensor],
    genome_state: Mapping[str, torch.Tensor],
    policy_mem_dim: torch.Tensor,
    policy_use_gru: torch.Tensor,
    drive_state: Mapping[str, torch.Tensor],
    policy_state: Mapping[str, torch.Tensor],
    start_energy: float,
    start_health: float,
    generation: int,
    sensor_radius_max: int,
    max_energy: float,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Commit one exact two-parent turnover refill batch atomically."""

    repro_threshold = identity_state.get("repro_threshold")
    if repro_threshold is None or genome_state.get("repro_threshold") is not repro_threshold:
        raise ValueError(
            "identity_state and genome_state must share the authoritative "
            "repro_threshold tensor"
        )
    module = load_cuda_birth_transaction_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_fixed_mlp_turnover_refill_transaction_v1(
        _ordered_tensors(identity_state, IDENTITY_STATE_FIELDS, group="identity_state"),
        parent_pool_slots,
        parent_pool_count,
        spawn_positions,
        planned_count,
        _ordered_tensors(
            scalar_rng_state,
            SCALAR_RNG_STATE_FIELDS,
            group="scalar_rng_state",
        ),
        _ordered_tensors(genome_state, GENOME_INT32_FIELDS, group="genome_state"),
        policy_mem_dim,
        policy_use_gru,
        _ordered_tensors(genome_state, GENOME_FLOAT32_FIELDS, group="genome_state"),
        _ordered_tensors(drive_state, DRIVE_STATE_FIELDS, group="drive_state"),
        _ordered_tensors(policy_state, POLICY_STATE_FIELDS, group="policy_state"),
        float(start_energy),
        float(start_health),
        int(generation),
        int(sensor_radius_max),
        float(max_energy),
    )
    names = (
        "parent_agent_ids",
        "mate_agent_ids",
        "child_agent_ids",
        "child_slots",
        "committed_count",
        "alive_agent_ids_by_order",
        "alive_count",
        "scalar_semantic_draws_added",
        "population_birth_noise_rows_consumed",
        "population_exp_prior_noise_rows_consumed",
        "native_population_standard_normal_rows_generated",
        "drive_semantic_draws_added",
        "drive_raw64_draws_added",
        "policy_semantic_draws_added",
        "policy_mutation_events",
        "policy_mutated_elements",
        "policy_mutation_delta_total",
        "policy_mutation_occurred_by_birth",
        "policy_mutation_delta_by_birth",
        "identity_status",
        "scalar_status",
        "drive_status",
        "policy_status",
        "status",
    )
    return dict(zip(names, outputs, strict=True))


__all__ = [
    "BIRTH_TRANSACTION_CAPABILITIES",
    "BIRTH_TRANSACTION_STATUS",
    "DRIVE_STATE_FIELDS",
    "GENOME_FLOAT32_FIELDS",
    "GENOME_INT32_FIELDS",
    "IDENTITY_STATE_FIELDS",
    "POLICY_STATE_FIELDS",
    "SCALAR_RNG_STATE_FIELDS",
    "execute_fixed_mlp_birth_transaction_cuda",
    "execute_fixed_mlp_turnover_refill_transaction_cuda",
    "load_cuda_birth_transaction_extension",
]
