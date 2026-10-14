"""Executable C4 R3 resident death-to-ordinary-birth slot reuse.

R3 creates a device-local post-death planning shadow, runs the R2 ordinary
birth planner against that shadow, and performs one combined authoritative
commit only after both preflight domains succeed.  It never sequentially
mutates live state through the R1 and R2 adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load

from .c4_population import TensorWorldC4BirthConfigV1
from .c4_resident_birth_r2 import (
    TensorWorldC4ResidentBirthR2StagingV1,
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _primary_tensors,
    _zero_tensors,
    make_c4_resident_birth_r2_staging_v1,
)
from .c4_state import TensorWorldStateBundleV4
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = ROOT / "engine" / "tensor_world" / "native"
PERF_ROOT = ROOT / "engine" / "perf"
NATIVE_BINDING = NATIVE_ROOT / "tensor_world_c4_resident_turnover_r3_binding.cpp"
NATIVE_KERNEL = NATIVE_ROOT / "tensor_world_c4_resident_turnover_r3.cu"
R1_SOURCE = NATIVE_ROOT / "tensor_world_c4_resident_death.cu"
R2_SOURCE = NATIVE_ROOT / "tensor_world_c4_resident_birth_r2.cu"
CPU_SOURCE = ROOT / "engine" / "tensor_world" / "c4_population.py"
EVENT_SOURCE = ROOT / "engine" / "tensor_world" / "c4_events.py"

C4_RESIDENT_TURNOVER_R3_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_turnover_r3_v1"
)
C4_RESIDENT_TURNOVER_R3_STAGING_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_turnover_r3_staging_v1"
)
C4_RESIDENT_TURNOVER_R3_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_TURNOVER_R3_STATUS_V1: Final[dict[int, str]] = {
    0: "ok",
    20: "invalid_pending_or_queue_state",
    21: "pending_or_event_capacity_exceeded",
    22: "duplicate_or_contradictory_free_slot",
    23: "death_event_semantics_invalid",
    24: "birth_preflight_failed",
    25: "r3_slot_reuse_admission_failed",
}


@dataclass(slots=True)
class TensorWorldC4ResidentTurnoverR3StagingV1:
    """Fixed-capacity death rows, planning shadow, and nested R2 staging."""

    birth: TensorWorldC4ResidentBirthR2StagingV1
    death_agent_ids: torch.Tensor
    death_slots: torch.Tensor
    death_reason_codes: torch.Tensor
    death_count: torch.Tensor
    virtual_free_queue: torch.Tensor
    virtual_free_count: torch.Tensor
    death_status: torch.Tensor
    shadow_alive: torch.Tensor
    shadow_agent_id: torch.Tensor
    shadow_pos_y: torch.Tensor
    shadow_pos_x: torch.Tensor
    shadow_energy: torch.Tensor
    shadow_health: torch.Tensor
    shadow_occupancy_agent: torch.Tensor
    shadow_occupancy_slot: torch.Tensor
    shadow_patch_agent: torch.Tensor
    shadow_agent_id_to_slot: torch.Tensor
    shadow_repro_cooldown: torch.Tensor
    shadow_policy_energy_observation: torch.Tensor
    shadow_event_active_count: torch.Tensor
    shadow_event_status: torch.Tensor
    schema_version: str = C4_RESIDENT_TURNOVER_R3_STAGING_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.death_agent_ids.device

    def validate(self, *, batch: int, capacity: int, grid_size: int) -> None:
        if self.schema_version != C4_RESIDENT_TURNOVER_R3_STAGING_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 R3 staging schema is unsupported")
        self.birth.validate(batch=batch, capacity=capacity)
        if self.device.type != "cuda" or self.birth.device != self.device:
            raise TensorWorldContractError("C4 R3 staging requires one CUDA device")
        slots = (batch, capacity)
        grid = (batch, grid_size, grid_size)
        expected = (
            (self.death_agent_ids, slots, torch.int64),
            (self.death_slots, slots, torch.int64),
            (self.death_reason_codes, slots, torch.int32),
            (self.death_count, (batch,), torch.int64),
            (self.virtual_free_queue, slots, torch.int64),
            (self.virtual_free_count, (batch,), torch.int64),
            (self.death_status, (batch,), torch.int32),
            (self.shadow_alive, slots, torch.bool),
            (self.shadow_agent_id, slots, torch.int64),
            (self.shadow_pos_y, slots, torch.int64),
            (self.shadow_pos_x, slots, torch.int64),
            (self.shadow_energy, slots, torch.float32),
            (self.shadow_health, slots, torch.float32),
            (self.shadow_occupancy_agent, grid, torch.int64),
            (self.shadow_occupancy_slot, grid, torch.int64),
            (self.shadow_patch_agent, grid, torch.int32),
            (self.shadow_agent_id_to_slot, slots, torch.int32),
            (self.shadow_repro_cooldown, slots, torch.int32),
            (self.shadow_policy_energy_observation, slots, torch.float64),
            (self.shadow_event_active_count, (batch,), torch.int64),
            (self.shadow_event_status, (batch,), torch.int32),
        )
        devices = {self.device}
        for tensor, shape, dtype in expected:
            if tuple(tensor.shape) != shape or tensor.dtype != dtype or not tensor.is_contiguous():
                raise TensorWorldContractError("C4 R3 staging tensor ABI mismatch")
            devices.add(tensor.device)
        if len(devices) != 1:
            raise TensorWorldContractError("C4 R3 staging must share one device")


def make_c4_resident_turnover_r3_staging_v1(
    *, batch: int, capacity: int, grid_size: int, device: torch.device | str
) -> TensorWorldC4ResidentTurnoverR3StagingV1:
    target = torch.device(device)
    if target.type != "cuda":
        raise TensorWorldContractError("C4 R3 staging requires CUDA; CPU fallback forbidden")
    slots = (batch, capacity)
    grid = (batch, grid_size, grid_size)
    staging = TensorWorldC4ResidentTurnoverR3StagingV1(
        birth=make_c4_resident_birth_r2_staging_v1(
            batch=batch, capacity=capacity, device=target
        ),
        death_agent_ids=torch.full(slots, -1, dtype=torch.int64, device=target),
        death_slots=torch.full(slots, -1, dtype=torch.int64, device=target),
        death_reason_codes=torch.zeros(slots, dtype=torch.int32, device=target),
        death_count=torch.zeros(batch, dtype=torch.int64, device=target),
        virtual_free_queue=torch.full(slots, -1, dtype=torch.int64, device=target),
        virtual_free_count=torch.zeros(batch, dtype=torch.int64, device=target),
        death_status=torch.zeros(batch, dtype=torch.int32, device=target),
        shadow_alive=torch.zeros(slots, dtype=torch.bool, device=target),
        shadow_agent_id=torch.full(slots, -1, dtype=torch.int64, device=target),
        shadow_pos_y=torch.full(slots, -1, dtype=torch.int64, device=target),
        shadow_pos_x=torch.full(slots, -1, dtype=torch.int64, device=target),
        shadow_energy=torch.zeros(slots, dtype=torch.float32, device=target),
        shadow_health=torch.zeros(slots, dtype=torch.float32, device=target),
        shadow_occupancy_agent=torch.full(grid, -1, dtype=torch.int64, device=target),
        shadow_occupancy_slot=torch.full(grid, -1, dtype=torch.int64, device=target),
        shadow_patch_agent=torch.full(grid, -1, dtype=torch.int32, device=target),
        shadow_agent_id_to_slot=torch.full(slots, -1, dtype=torch.int32, device=target),
        shadow_repro_cooldown=torch.zeros(slots, dtype=torch.int32, device=target),
        shadow_policy_energy_observation=torch.zeros(
            slots, dtype=torch.float64, device=target
        ),
        shadow_event_active_count=torch.zeros(batch, dtype=torch.int64, device=target),
        shadow_event_status=torch.zeros(batch, dtype=torch.int32, device=target),
    )
    staging.validate(batch=batch, capacity=capacity, grid_size=grid_size)
    return staging


_MODULE: Any | None = None


def c4_resident_turnover_r3_source_receipt_v1() -> dict[str, object]:
    sources = {
        "adapter": Path(__file__), "binding": NATIVE_BINDING, "kernel": NATIVE_KERNEL,
        "r1_native": R1_SOURCE, "r2_native": R2_SOURCE,
        "cpu_population_oracle": CPU_SOURCE, "typed_event_oracle": EVENT_SOURCE,
    }
    return {
        "schema_version": C4_RESIDENT_TURNOVER_R3_SCHEMA_VERSION_V1,
        "route": "R3_B1_DEATH_RELEASE_ORDINARY_BIRTH_SLOT_REUSE",
        "decision": C4_RESIDENT_TURNOVER_R3_DECISION_V1,
        "preflight": "post_death_device_shadow_then_R2_plan",
        "authoritative_commit": "one_status_guarded_multi_kernel_commit_phase",
        "commit_phase": (
            "inherit_reset_kernels_then_combined_identity_event_commit; "
            "no host-visible boundary and no fallible validation after merged status"
        ),
        "rung_admission": "death_and_birth_and_newly_dead_child_slot_required",
        "event_order": "death_rows_then_birth_rows",
        "biological_identity": "monotonic_new_id_never_dead_id_reuse",
        "mutation_mate_recombination_gru": "fail_closed",
        "static_identity_and_jungle_rng": "preserved",
        "gate_claim": False,
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path.read_bytes()).hexdigest()}
            for name, path in sources.items()
        },
    }


def load_tensor_world_c4_resident_turnover_r3_extension(
    *, build_directory: str | Path | None = None, verbose: bool = False
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise TensorWorldContractError("C4 R3 requires CUDA")
    digest = sha256(
        b"\0".join(path.read_bytes() for path in (NATIVE_BINDING, NATIVE_KERNEL, R2_SOURCE))
    ).hexdigest()[:16]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=f"tensor_world_c4_resident_turnover_r3_{digest}",
        sources=[str(NATIVE_BINDING), str(NATIVE_KERNEL)],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(PERF_ROOT), str(NATIVE_ROOT)],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def _death_authoritative(bundle: TensorWorldStateBundleV4) -> list[torch.Tensor]:
    return [
        bundle.base.lifecycle.death_candidate, bundle.base.lifecycle.death_reason,
        bundle.population.pending_free_agent_ids, bundle.population.pending_free_slots,
        bundle.population.pending_free_count, bundle.population.death_agent_ids,
        bundle.population.death_slots, bundle.population.death_reason_codes,
        bundle.population.death_count, bundle.base.agent.rest_streak,
        bundle.base.policy_runtime.action_cache_token,
        bundle.base.policy_runtime.action_cache_cursor,
        bundle.base.policy_runtime.action_cache_next,
    ]


def _turnover_staging_tensors(
    staging: TensorWorldC4ResidentTurnoverR3StagingV1,
) -> list[torch.Tensor]:
    return [
        staging.death_agent_ids, staging.death_slots, staging.death_reason_codes,
        staging.death_count, staging.virtual_free_queue, staging.virtual_free_count,
        staging.death_status, staging.shadow_alive, staging.shadow_agent_id,
        staging.shadow_pos_y, staging.shadow_pos_x, staging.shadow_energy,
        staging.shadow_health, staging.shadow_occupancy_agent,
        staging.shadow_occupancy_slot, staging.shadow_patch_agent,
        staging.shadow_agent_id_to_slot, staging.shadow_repro_cooldown,
        staging.shadow_policy_energy_observation, staging.shadow_event_active_count,
        staging.shadow_event_status,
    ]


def execute_c4_resident_turnover_r3_cuda_v1(
    bundle: TensorWorldStateBundleV4,
    staging: TensorWorldC4ResidentTurnoverR3StagingV1,
    config: TensorWorldC4BirthConfigV1,
    *,
    mate_slots: torch.Tensor | None = None,
    recombination: bool = False,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> TensorWorldC4ResidentTurnoverR3StagingV1:
    if not isinstance(bundle, TensorWorldStateBundleV4) or bundle.device.type != "cuda":
        raise TensorWorldContractError("C4 R3 requires a CUDA V4 bundle")
    if bundle.batch != 1:
        raise TensorWorldContractError("C4 R3 is bounded to B1")
    if not isinstance(config, TensorWorldC4BirthConfigV1) or config.allow_mutation:
        raise TensorWorldContractError("C4 R3 mutation is unsupported and fail-closed")
    if mate_slots is not None or recombination:
        raise TensorWorldContractError("C4 R3 mate/recombination is unsupported and fail-closed")
    if bundle.events is None:
        raise TensorWorldContractError("C4 R3 typed event tape is required")
    bundle.validate()
    staging.validate(batch=1, capacity=bundle.capacity, grid_size=bundle.grid_size)
    if staging.device != bundle.device:
        raise TensorWorldContractError("C4 R3 staging must share the bundle device")
    max_births = bundle.capacity if config.max_births is None else config.max_births
    module = load_tensor_world_c4_resident_turnover_r3_extension(
        build_directory=build_directory, verbose=verbose
    )
    b = staging.birth
    module.turnover_transaction(
        _primary_tensors(bundle), _death_authoritative(bundle),
        _inherit_tensors(bundle), _zero_tensors(bundle), _minus_one_tensors(bundle),
        _half_tensors(bundle), _one_tensors(bundle),
        [
            b.spawn_uniforms, b.parent_slots, b.child_slots, b.child_pos_y,
            b.child_pos_x, b.child_energy, b.planned_count,
            b.decision_parent_slots, b.decision_adjacent_counts,
            b.decision_reason_codes, b.decision_count, b.planned_repro_cooldown,
            b.spawn_rng_draws_added, b.spawn_rng_semantic_draws_added,
            b.child_agent_ids, b.queue_cursor, b.event_sequence, b.status,
        ],
        _turnover_staging_tensors(staging),
        int(max_births), float(config.parent_energy_share),
        float(config.minimum_energy_buffer), int(config.cooldown_steps),
        float(config.start_health), float(config.max_energy),
        int(config.life_history_mode), int(config.maturity_start_ticks),
        int(config.maturity_rise_ticks), int(config.fertility_decline_start_ticks),
        int(config.fertility_decline_ticks), float(config.elder_fertility_floor),
        int(config.generation), int(bundle.base.tick),
    )
    return staging


__all__ = [
    "C4_RESIDENT_TURNOVER_R3_DECISION_V1", "C4_RESIDENT_TURNOVER_R3_SCHEMA_VERSION_V1",
    "C4_RESIDENT_TURNOVER_R3_STAGING_SCHEMA_VERSION_V1",
    "C4_RESIDENT_TURNOVER_R3_STATUS_V1", "TensorWorldC4ResidentTurnoverR3StagingV1",
    "c4_resident_turnover_r3_source_receipt_v1",
    "execute_c4_resident_turnover_r3_cuda_v1",
    "load_tensor_world_c4_resident_turnover_r3_extension",
    "make_c4_resident_turnover_r3_staging_v1",
]
