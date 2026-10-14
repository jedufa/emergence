from __future__ import annotations

"""Atomic resident CUDA ordinary-birth microcycle for Flat and Patch."""

from hashlib import sha256
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import torch
from torch.utils.cpp_extension import load

from perf.cuda_architecture_b_birth_transaction import (
    DRIVE_STATE_FIELDS,
    GENOME_FLOAT32_FIELDS,
    GENOME_INT32_FIELDS,
    IDENTITY_STATE_FIELDS,
    POLICY_STATE_FIELDS,
    SCALAR_RNG_STATE_FIELDS,
)


_ORDINARY_POLICY_STATE_FIELDS: Final[tuple[str, ...]] = tuple(
    field
    for field in POLICY_STATE_FIELDS
    if field != "policy_birth_crossover_mask_packed_tape"
)


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "engine" / "perf" / "native"
PERF = NATIVE.parent
SOURCE_MICROCYCLE = NATIVE / "cuda_architecture_b_birth_microcycle.cpp"
SOURCE_PLANNER = NATIVE / "cuda_architecture_b_asexual_birth_plan.cu"
SOURCE_LIFECYCLE = NATIVE / "cuda_architecture_b_lifecycle.cu"
SOURCE_TRANSACTION = NATIVE / "cuda_architecture_b_birth_transaction.cpp"
SOURCE_DRIVES = NATIVE / "cuda_architecture_b_child_drives.cu"
SOURCE_POLICY = NATIVE / "cuda_architecture_b_child_policy.cu"
SOURCE_BIRTH_EVENTS = NATIVE / "cuda_architecture_b_birth_lifecycle_events.cu"
SOURCE_EVENT_TAPE = NATIVE / "cuda_architecture_b_event_tape.cu"
SOURCE_PCG64 = NATIVE / "cuda_architecture_b_pcg64.cuh"
SOURCE_TORCH_RNG_STATE_PACK = PERF / "cuda_architecture_b_torch_rng_state_pack_v1.cuh"

COMPILE_SOURCES: Final[tuple[Path, ...]] = (
    SOURCE_MICROCYCLE,
    SOURCE_PLANNER,
    SOURCE_LIFECYCLE,
    SOURCE_TRANSACTION,
    SOURCE_DRIVES,
    SOURCE_POLICY,
    SOURCE_BIRTH_EVENTS,
    SOURCE_EVENT_TAPE,
)

BIRTH_MICROCYCLE_STATUS: Final[dict[int, str]] = {
    0: "ok",
    1: "asexual_birth_plan_failed",
    2: "child_transaction_failed",
    3: "birth_lifecycle_event_publication_failed",
    4: "external_generation_gate_failed",
}

BIRTH_MICROCYCLE_CAPABILITIES: Final[dict[str, bool | str]] = {
    "schema_version": "eiel_cuda_architecture_b_birth_microcycle_capabilities_v1",
    "decision": "ARCHB-CUDA/ATOMIC-ORDINARY-BIRTH-MICROCYCLE",
    "action_death_pending_release_before_birth": False,
    "authoritative_free_queue_count_bound": True,
    "device_resident_plan_count": True,
    "fixed_capacity_inactive_tail": True,
    "decision_ordered_failure_mutation_birth_events": True,
    "whole_microcycle_atomic_commit": True,
    "external_generation_gate": True,
    "per_tick_host_sync": False,
    "ordinary_non_turnover_flat_patch_fixed_mlp": True,
    "generation_integration": False,
    "generation_turnover": False,
    "gru": False,
    "jungle": False,
    "wdl": False,
    "promotion_qualified": False,
}

_MODULE: Any | None = None


def _ordered(
    state: Mapping[str, torch.Tensor],
    fields: tuple[str, ...],
    *,
    group: str,
) -> list[torch.Tensor]:
    missing = [name for name in fields if name not in state]
    if missing:
        raise KeyError(f"{group} is missing fields: {', '.join(missing)}")
    return [state[name] for name in fields]


def load_cuda_birth_microcycle_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("Architecture B birth microcycle requires CUDA")
    digest = sha256()
    for source in (*COMPILE_SOURCES, SOURCE_PCG64, SOURCE_TORCH_RNG_STATE_PACK):
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=f"eiel_cuda_architecture_b_birth_microcycle_{digest.hexdigest()[:16]}",
        sources=[str(source) for source in COMPILE_SOURCES],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(NATIVE), str(PERF)],
        extra_cflags=[
            "-O2",
            "-ffp-contract=off",
            "-DEIEL_CUDA_ARCHITECTURE_B_NO_PYBIND",
        ],
        extra_cuda_cflags=[
            "-O2",
            "--fmad=false",
            "-DEIEL_CUDA_ARCHITECTURE_B_NO_PYBIND",
        ],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def execute_birth_microcycle_cuda(
    *,
    identity_state: Mapping[str, torch.Tensor],
    scalar_rng_state: Mapping[str, torch.Tensor],
    genome_state: Mapping[str, torch.Tensor],
    policy_mem_dim: torch.Tensor,
    policy_use_gru: torch.Tensor,
    drive_state: Mapping[str, torch.Tensor],
    policy_state: Mapping[str, torch.Tensor],
    planner_pcg64_state_words: torch.Tensor,
    planner_pcg64_buffer_words: torch.Tensor,
    policy_np_semantic_draws: torch.Tensor,
    runner_pcg64_state_words: torch.Tensor,
    runner_pcg64_buffer_words: torch.Tensor,
    metrics_np_semantic_draws: torch.Tensor,
    event_tape_columns: Sequence[torch.Tensor],
    event_tape_active_count: torch.Tensor,
    event_tape_status: torch.Tensor,
    parent_energy_share: float,
    minimum_energy_buffer: float,
    start_health: float,
    cooldown_steps: int,
    generation: int,
    tick: int,
    sensor_radius_max: int,
    max_energy: float,
    max_births: int,
    injury_repro_k_injury: float = 0.0,
    injury_repro_k_scar: float = 0.0,
    injury_birth_fail_scar: float = 0.0,
    injury_birth_fail_injury: float = 0.0,
    env_policy_alias: bool = True,
    requires_mate: bool = False,
    injury_gate_active: bool = False,
    conception_gate_active: bool = False,
    gestation_gate_active: bool = False,
    external_gate_status: torch.Tensor | None = None,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Plan, commit, and publish one ordinary-birth microcycle."""
    repro_threshold = identity_state.get("repro_threshold")
    if repro_threshold is None or genome_state.get("repro_threshold") is not repro_threshold:
        raise ValueError(
            "identity_state and genome_state must share the authoritative "
            "repro_threshold tensor"
        )
    columns = list(event_tape_columns)
    if len(columns) != 22:
        raise ValueError("event_tape_columns must contain 22 tensors")
    if external_gate_status is None:
        external_gate_status = torch.zeros_like(event_tape_status)
    if (
        external_gate_status.dtype != torch.int32
        or tuple(external_gate_status.shape) != (1,)
        or external_gate_status.device != event_tape_status.device
        or not external_gate_status.is_contiguous()
    ):
        raise ValueError("external_gate_status must be contiguous int32[1]")
    module = load_cuda_birth_microcycle_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_birth_microcycle_v1(
        _ordered(identity_state, IDENTITY_STATE_FIELDS, group="identity_state"),
        _ordered(scalar_rng_state, SCALAR_RNG_STATE_FIELDS, group="scalar_rng_state"),
        _ordered(genome_state, GENOME_INT32_FIELDS, group="genome_state"),
        policy_mem_dim,
        policy_use_gru,
        _ordered(genome_state, GENOME_FLOAT32_FIELDS, group="genome_state"),
        _ordered(drive_state, DRIVE_STATE_FIELDS, group="drive_state"),
        _ordered(
            policy_state,
            _ORDINARY_POLICY_STATE_FIELDS,
            group="policy_state",
        ),
        planner_pcg64_state_words,
        planner_pcg64_buffer_words,
        policy_np_semantic_draws,
        runner_pcg64_state_words,
        runner_pcg64_buffer_words,
        metrics_np_semantic_draws,
        columns,
        event_tape_active_count,
        event_tape_status,
        external_gate_status,
        float(parent_energy_share),
        float(minimum_energy_buffer),
        float(start_health),
        int(cooldown_steps),
        int(generation),
        int(tick),
        int(sensor_radius_max),
        float(max_energy),
        int(max_births),
        float(injury_repro_k_injury),
        float(injury_repro_k_scar),
        float(injury_birth_fail_scar),
        float(injury_birth_fail_injury),
        bool(env_policy_alias),
        bool(requires_mate),
        bool(injury_gate_active),
        bool(conception_gate_active),
        bool(gestation_gate_active),
    )
    names = (
        "planned_parent_slots",
        "planned_child_slots",
        "planned_spawn_positions",
        "planned_child_energy",
        "decision_parent_slots",
        "decision_reason_codes",
        "reason_counts",
        "planned_count",
        "decision_count",
        "spawn_rng_draws_added",
        "spawn_rng_semantic_draws_added",
        "planner_status",
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
        "transaction_status",
        "event_tape_active_count",
        "event_tape_status",
        "status",
        "runner_rng_draws_added",
        "runner_rng_semantic_draws_added",
        "metrics_np_semantic_draws",
    )
    return dict(zip(names, outputs, strict=True))


__all__ = [
    "BIRTH_MICROCYCLE_CAPABILITIES",
    "BIRTH_MICROCYCLE_STATUS",
    "execute_birth_microcycle_cuda",
    "load_cuda_birth_microcycle_extension",
]
