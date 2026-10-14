"""First resident C4 population rung: batched CUDA death transaction.

This adapter deliberately stops at the R0/R1 death-only boundary.  It uses
caller-owned fixed-capacity CUDA staging, runs a device-side preflight, and
launches the authoritative commit only when every world passed preflight.
There is no CPU retry or per-world/per-agent Python dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load

from .c4_state import TensorWorldStateBundleV4
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
PERF_ROOT = ROOT / "engine" / "perf"
NATIVE_ROOT = ROOT / "engine" / "tensor_world" / "native"
NATIVE_BINDING = NATIVE_ROOT / "tensor_world_c4_resident_death_binding.cpp"
NATIVE_KERNEL = NATIVE_ROOT / "tensor_world_c4_resident_death.cu"
SOURCE_LIFECYCLE = PERF_ROOT / "native" / "cuda_architecture_b_lifecycle.cu"
SOURCE_EVENT_HEADER = PERF_ROOT / "cuda_architecture_b_persistent_lifecycle_events.cuh"

C4_RESIDENT_DEATH_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_death_transaction_v1"
)
C4_RESIDENT_DEATH_STAGING_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_death_staging_v1"
)
C4_RESIDENT_DEATH_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_DEATH_STATUS_V1: Final[dict[int, str]] = {
    0: "ok",
    1: "invalid_pending_count",
    2: "pending_capacity_exceeded",
    3: "invalid_free_queue_count",
    4: "duplicate_free_slot",
    5: "free_slot_state_contradiction",
    6: "event_capacity_exceeded",
    7: "event_semantics_invalid",
    8: "duplicate_death_identity",
}


@dataclass(slots=True)
class TensorWorldC4ResidentDeathStagingV1:
    """Caller-owned fixed-capacity device scratch for one death transaction."""

    death_agent_ids: torch.Tensor
    death_slots: torch.Tensor
    death_reason_codes: torch.Tensor
    death_count: torch.Tensor
    survivor_agent_ids: torch.Tensor
    survivor_slots: torch.Tensor
    survivor_count: torch.Tensor
    status: torch.Tensor
    schema_version: str = C4_RESIDENT_DEATH_STAGING_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.death_agent_ids.device

    @property
    def batch(self) -> int:
        return int(self.death_agent_ids.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.death_agent_ids.shape[1])

    def validate(self, *, batch: int | None = None, capacity: int | None = None) -> None:
        if self.schema_version != C4_RESIDENT_DEATH_STAGING_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 resident death staging schema is unsupported")
        if self.death_agent_ids.ndim != 2:
            raise TensorWorldContractError("C4 resident death staging must be rank two")
        expected_batch, expected_capacity = self.death_agent_ids.shape
        if expected_batch <= 0 or expected_capacity <= 0:
            raise TensorWorldContractError("C4 resident death staging dimensions must be positive")
        if batch is not None and expected_batch != batch:
            raise TensorWorldContractError("C4 resident death staging batch mismatch")
        if capacity is not None and expected_capacity != capacity:
            raise TensorWorldContractError("C4 resident death staging capacity mismatch")
        if self.device.type != "cuda":
            raise TensorWorldContractError(
                "C4 resident death staging requires CUDA; CPU fallback is forbidden"
            )
        slots = (expected_batch, expected_capacity)
        expected = (
            ("death_agent_ids", self.death_agent_ids, slots, torch.int64),
            ("death_slots", self.death_slots, slots, torch.int64),
            ("death_reason_codes", self.death_reason_codes, slots, torch.int32),
            ("death_count", self.death_count, (expected_batch,), torch.int64),
            ("survivor_agent_ids", self.survivor_agent_ids, slots, torch.int64),
            ("survivor_slots", self.survivor_slots, slots, torch.int32),
            ("survivor_count", self.survivor_count, (expected_batch,), torch.int64),
            ("status", self.status, (expected_batch,), torch.int32),
        )
        devices = {self.device}
        for name, tensor, shape, dtype in expected:
            if tuple(tensor.shape) != shape or tensor.dtype != dtype or not tensor.is_contiguous():
                raise TensorWorldContractError(
                    f"C4 resident death staging {name} requires contiguous "
                    f"shape={shape!r} dtype={dtype}"
                )
            devices.add(tensor.device)
        if len(devices) != 1:
            raise TensorWorldContractError("C4 resident death staging must share one device")


def make_c4_resident_death_staging_v1(
    *, batch: int, capacity: int, device: torch.device | str
) -> TensorWorldC4ResidentDeathStagingV1:
    """Allocate fixed-capacity staging once, outside the resident transaction."""

    if type(batch) is not int or batch <= 0 or type(capacity) is not int or capacity <= 0:
        raise TensorWorldContractError("C4 resident death staging dimensions must be positive ints")
    target = torch.device(device)
    if target.type != "cuda":
        raise TensorWorldContractError(
            "C4 resident death staging requires CUDA; CPU fallback is forbidden"
        )
    slots = (batch, capacity)
    staging = TensorWorldC4ResidentDeathStagingV1(
        death_agent_ids=torch.full(slots, -1, dtype=torch.int64, device=target),
        death_slots=torch.full(slots, -1, dtype=torch.int64, device=target),
        death_reason_codes=torch.zeros(slots, dtype=torch.int32, device=target),
        death_count=torch.zeros(batch, dtype=torch.int64, device=target),
        survivor_agent_ids=torch.full(slots, -1, dtype=torch.int64, device=target),
        survivor_slots=torch.full(slots, -1, dtype=torch.int32, device=target),
        survivor_count=torch.zeros(batch, dtype=torch.int64, device=target),
        status=torch.zeros(batch, dtype=torch.int32, device=target),
    )
    staging.validate(batch=batch, capacity=capacity)
    return staging


_MODULE: Any | None = None


def c4_resident_death_native_source_manifest_v1() -> dict[str, object]:
    """Bind the bounded adapter to its implementation and source precedents."""

    sources = {
        "adapter": Path(__file__),
        "binding": NATIVE_BINDING,
        "kernel": NATIVE_KERNEL,
        "architecture_b_lifecycle": SOURCE_LIFECYCLE,
        "architecture_b_typed_events": SOURCE_EVENT_HEADER,
    }
    return {
        "schema_version": C4_RESIDENT_DEATH_SCHEMA_VERSION_V1,
        "staging_schema_version": C4_RESIDENT_DEATH_STAGING_SCHEMA_VERSION_V1,
        "decision": C4_RESIDENT_DEATH_DECISION_V1,
        "route": "R0_R1_CUDA_DEATH_ONLY",
        "batch_commit": "all_worlds_preflight_then_globally_guarded_commit",
        "typed_death_events": "required_attached_resident_tape",
        "cpu_fallback": False,
        "python_world_or_agent_loops": False,
        "caller_owned_fixed_capacity_staging": True,
        "gate_claim": False,
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


def load_tensor_world_c4_resident_death_extension(
    *, build_directory: str | Path | None = None, verbose: bool = False
) -> Any:
    """Build/load the bounded resident death transaction extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise TensorWorldContractError("C4 resident death transaction requires CUDA")
    digest = sha256(
        b"\0".join(path.read_bytes() for path in (NATIVE_BINDING, NATIVE_KERNEL))
    ).hexdigest()[:16]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=f"tensor_world_c4_resident_death_{digest}",
        sources=[str(NATIVE_BINDING), str(NATIVE_KERNEL)],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(PERF_ROOT)],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def _authoritative_tensors(bundle: TensorWorldStateBundleV4) -> list[torch.Tensor]:
    core = bundle.base.base.core
    c3 = bundle.base
    p = bundle.population
    tape = bundle.events
    if tape is None:
        raise TensorWorldContractError(
            "C4 resident death transaction requires an attached typed event tape; "
            "the unsupported event boundary is fail-closed"
        )
    return [
        core.agents.alive,
        core.agents.agent_id,
        core.agents.pos_y,
        core.agents.pos_x,
        core.agents.energy,
        core.agents.health,
        core.occupancy.agent_id_grid,
        core.occupancy.slot_grid,
        c3.base.patch.agent_grid_i32,
        p.agent_id_to_slot,
        c3.lifecycle.death_candidate,
        c3.lifecycle.death_reason,
        c3.agent.policy_energy_observation,
        c3.agent.rest_streak,
        c3.policy_runtime.action_cache_token,
        c3.policy_runtime.action_cache_cursor,
        c3.policy_runtime.action_cache_next,
        p.active_agent_ids,
        p.active_agent_slots,
        p.active_count,
        p.pending_free_agent_ids,
        p.pending_free_slots,
        p.pending_free_count,
        p.death_agent_ids,
        p.death_slots,
        p.death_reason_codes,
        p.death_count,
        p.survivor_agent_ids,
        p.free_queue,
        p.free_queue_count,
        p.lineage_founder_id,
        p.lineage_depth,
        p.birth_parent_ids,
        p.birth_child_ids,
        p.birth_count,
        p.mutation_occurred,
        p.mutation_delta_total,
        tape.event_type,
        tape.generation,
        tape.tick,
        tape.actor_id,
        tape.target_id,
        tape.value,
        tape.typed_event_type,
        tape.typed_generation,
        tape.typed_tick,
        tape.typed_sequence,
        tape.typed_actor_id,
        tape.typed_target_id,
        tape.typed_actor_slot,
        tape.typed_target_slot,
        tape.typed_parent_id,
        tape.typed_child_id,
        tape.typed_founder_id,
        tape.typed_lineage_depth,
        tape.typed_subtype,
        tape.typed_reason,
        tape.typed_success,
        tape.typed_value,
        tape.active_count,
        tape.status,
    ]


def execute_c4_resident_death_transaction_cuda_v1(
    bundle: TensorWorldStateBundleV4,
    staging: TensorWorldC4ResidentDeathStagingV1,
    *,
    generation: int = 0,
    tick: int | None = None,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> TensorWorldC4ResidentDeathStagingV1:
    """Launch preflight and batch-wide all-or-none death commit on CUDA.

    The returned status remains device resident.  Callers may inspect it at a
    measurement/test boundary, but the transaction itself performs no host
    materialization or synchronization and never retries on CPU.
    """

    if not isinstance(bundle, TensorWorldStateBundleV4):
        raise TensorWorldContractError("C4 resident death transaction requires a C4 bundle")
    if bundle.device.type != "cuda":
        raise TensorWorldContractError(
            "C4 resident death transaction requires CUDA; CPU fallback is forbidden"
        )
    if bundle.events is None:
        raise TensorWorldContractError(
            "C4 resident death transaction requires an attached typed event tape; "
            "the unsupported event boundary is fail-closed"
        )
    if not isinstance(staging, TensorWorldC4ResidentDeathStagingV1):
        raise TensorWorldContractError("C4 resident death staging object is required")
    staging.validate(batch=bundle.batch, capacity=bundle.capacity)
    if staging.device != bundle.device:
        raise TensorWorldContractError("C4 resident death staging must share the bundle device")
    if type(generation) is not int or generation < 0:
        raise TensorWorldContractError("C4 resident death generation must be nonnegative int")
    effective_tick = int(bundle.base.tick) if tick is None else tick
    if type(effective_tick) is not int or effective_tick < 0:
        raise TensorWorldContractError("C4 resident death tick must be nonnegative int")
    module = load_tensor_world_c4_resident_death_extension(
        build_directory=build_directory, verbose=verbose
    )
    module.death_transaction(
        _authoritative_tensors(bundle),
        [
            staging.death_agent_ids,
            staging.death_slots,
            staging.death_reason_codes,
            staging.death_count,
            staging.survivor_agent_ids,
            staging.survivor_slots,
            staging.survivor_count,
            staging.status,
        ],
        int(generation),
        int(effective_tick),
    )
    return staging


__all__ = [
    "C4_RESIDENT_DEATH_DECISION_V1",
    "C4_RESIDENT_DEATH_SCHEMA_VERSION_V1",
    "C4_RESIDENT_DEATH_STAGING_SCHEMA_VERSION_V1",
    "C4_RESIDENT_DEATH_STATUS_V1",
    "TensorWorldC4ResidentDeathStagingV1",
    "c4_resident_death_native_source_manifest_v1",
    "execute_c4_resident_death_transaction_cuda_v1",
    "load_tensor_world_c4_resident_death_extension",
    "make_c4_resident_death_staging_v1",
]
