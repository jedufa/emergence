"""Source-bound resident C4 identity/slot reservation preflight.

The C4 population oracle remains the semantic reference while the complete
population transaction is being integrated.  This module exposes the first
resident seam separately: the source Architecture-B identity preflight runs
on CUDA, writes only caller-owned staging tensors, and returns a status plus
the reserved child identities/slots.  It never commits occupancy, identity,
lineage, energy, or policy state.

Keeping this boundary separate is important.  A successful preflight is not a
C4 gate result and does not authorize a CPU fallback or a partial birth commit.
The later resident transaction can consume these staged rows after its genome,
policy, RNG, and event preflights also pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load

from .c4_state import TensorWorldStateBundleV4
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
PERF_ROOT = ROOT / "engine" / "perf"
NATIVE_ROOT = ROOT / "engine" / "tensor_world" / "native"
NATIVE_BINDING = NATIVE_ROOT / "tensor_world_c4_identity_binding.cpp"
NATIVE_KERNEL = NATIVE_ROOT / "tensor_world_c4_identity.cu"
SOURCE_HEADER = PERF_ROOT / "cuda_architecture_b_flattened_birth_identity_v2.cuh"
SOURCE_IDENTITY = PERF_ROOT / "cuda_architecture_b_persistent_birth_identity.cuh"

C4_IDENTITY_NATIVE_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_identity_preflight_native_v1"
)
C4_IDENTITY_NATIVE_ENTRYPOINT_V1: Final[str] = "identity_preflight"
C4_IDENTITY_STAGING_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_identity_preflight_staging_v1"
)
C4_IDENTITY_STAGING_STATUS_POLICY_V1: Final[str] = (
    "nonzero_status_is_fail_closed_no_cpu_fallback_or_partial_commit"
)
C4_IDENTITY_STAGING_FIELDS_V1: Final[tuple[str, ...]] = (
    "staged_parent_slots",
    "staged_child_slots",
    "staged_child_agent_ids",
    "staged_child_energy",
    "staged_count",
    "queue_head",
    "queue_count",
    "status",
)
C4_IDENTITY_NATIVE_STATUS_V1: Final[dict[int, str]] = {
    0: "ok",
    10: "invalid_birth_plan",
    11: "invalid_parent_order",
    12: "parent_not_viable",
    13: "no_free_slot",
    14: "agent_id_capacity",
    15: "spawn_reservation",
    16: "invalid_free_queue_count",
}

# This is the device-side ABI returned by ``identity_preflight``.  Shapes are
# intentionally expressed as templates rather than inferred from a returned
# tensor: a native implementation must not be able to silently widen or
# truncate a fixed-capacity row.
C4_IDENTITY_STAGING_FIELD_SPECS_V1: Final[dict[str, dict[str, object]]] = {
    "staged_parent_slots": {
        "shape": "[batch,capacity]",
        "dtype": "torch.int64",
        "role": "parent slot reservations",
    },
    "staged_child_slots": {
        "shape": "[batch,capacity]",
        "dtype": "torch.int64",
        "role": "child slot reservations",
    },
    "staged_child_agent_ids": {
        "shape": "[batch,capacity]",
        "dtype": "torch.int64",
        "role": "monotonic external child identities",
    },
    "staged_child_energy": {
        "shape": "[batch,capacity]",
        "dtype": "torch.float64",
        "role": "staged child energy",
    },
    "staged_count": {
        "shape": "[batch]",
        "dtype": "torch.int64",
        "role": "reserved birth count",
    },
    "queue_head": {
        "shape": "[batch]",
        "dtype": "torch.int64",
        "role": "transaction-local free queue head",
    },
    "queue_count": {
        "shape": "[batch]",
        "dtype": "torch.int64",
        "role": "remaining free queue count",
    },
    "status": {
        "shape": "[batch]",
        "dtype": "torch.int32",
        "role": "per-world fail-closed preflight status",
    },
}


@dataclass(frozen=True, slots=True)
class TensorWorldC4IdentityPreflightConfigV1:
    """Scalar source configuration for the resident identity preflight."""

    parent_energy_share: float = 0.5
    minimum_energy_buffer: float = 0.0
    start_health: float = 12.0
    cooldown_steps: int = 0
    generation: int = 0
    tick: int = 0

    def __post_init__(self) -> None:
        numeric = (
            ("parent_energy_share", self.parent_energy_share, 0.0, 1.0),
            ("minimum_energy_buffer", self.minimum_energy_buffer, 0.0, float("inf")),
            ("start_health", self.start_health, 0.0, float("inf")),
        )
        for name, value, lower, upper in numeric:
            if not isinstance(value, (int, float)):
                raise TensorWorldContractError(f"C4 identity {name} must be numeric")
            value_f = float(value)
            if not torch.isfinite(torch.tensor(value_f)) or not lower <= value_f <= upper:
                raise TensorWorldContractError(
                    f"C4 identity {name} is outside its source range"
                )
        for name in ("cooldown_steps", "generation", "tick"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TensorWorldContractError(
                    f"C4 identity {name} must be a nonnegative int"
                )


_MODULE: Any | None = None


def c4_identity_native_source_manifest_v1() -> dict[str, object]:
    """Return the exact source lineage used by the native preflight."""

    return {
        "schema_version": C4_IDENTITY_NATIVE_SCHEMA_VERSION_V1,
        "entrypoint": C4_IDENTITY_NATIVE_ENTRYPOINT_V1,
        "staging_schema_version": C4_IDENTITY_STAGING_SCHEMA_VERSION_V1,
        "staging_fields": {
            name: dict(spec) for name, spec in C4_IDENTITY_STAGING_FIELD_SPECS_V1.items()
        },
        "staging_device": "same_cuda_device_as_inputs",
        "staging_capacity": "fixed_[batch,capacity]_and_[batch]_rows",
        "status_policy": C4_IDENTITY_STAGING_STATUS_POLICY_V1,
        "sources": {
            "binding": {
                "path": str(NATIVE_BINDING.relative_to(ROOT)),
                "sha256": sha256(NATIVE_BINDING.read_bytes()).hexdigest(),
            },
            "kernel": {
                "path": str(NATIVE_KERNEL.relative_to(ROOT)),
                "sha256": sha256(NATIVE_KERNEL.read_bytes()).hexdigest(),
            },
            "flattened_identity_header": {
                "path": str(SOURCE_HEADER.relative_to(ROOT)),
                "sha256": sha256(SOURCE_HEADER.read_bytes()).hexdigest(),
            },
            "identity_header": {
                "path": str(SOURCE_IDENTITY.relative_to(ROOT)),
                "sha256": sha256(SOURCE_IDENTITY.read_bytes()).hexdigest(),
            },
        },
        "authoritative_semantics": (
            "preflight_ordinary_birth_identity_external_scratch_v2"
        ),
        "mutates_authoritative_state": False,
        "gate_claim": False,
    }


def load_tensor_world_c4_identity_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    """Build/load the source-bound C4 identity preflight extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise TensorWorldContractError(
            "C4 resident identity preflight requires CUDA"
        )
    if not NATIVE_BINDING.is_file() or not NATIVE_KERNEL.is_file():
        raise TensorWorldContractError("C4 native identity sources are missing")
    digest = sha256(
        b"\0".join(
            source.read_bytes()
            for source in (NATIVE_BINDING, NATIVE_KERNEL, SOURCE_HEADER, SOURCE_IDENTITY)
        )
    ).hexdigest()[:16]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=f"tensor_world_c4_identity_preflight_{digest}",
        sources=[str(NATIVE_BINDING), str(NATIVE_KERNEL)],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(PERF_ROOT)],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def _require_cuda_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device != device
        or tensor.dtype != dtype
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
    ):
        raise TensorWorldContractError(
            f"C4 identity {name} must be a CUDA tensor (requires CUDA "
            f"contiguous shape={shape!r} dtype={dtype})"
        )


def validate_c4_identity_preflight_staging_v1(
    outputs: Mapping[str, torch.Tensor],
    *,
    batch: int,
    capacity: int,
    device: torch.device,
    require_success: bool = False,
) -> dict[str, torch.Tensor]:
    """Validate the fixed-capacity device-local native staging ABI.

    The validator is deliberately independent of extension loading, so the
    contract can be checked in CPU-only CI.  CPU tensors are rejected even
    when their shapes look correct: copying a resident preflight to host
    memory is not a permitted fallback.  ``require_success`` is the explicit
    admission boundary a transaction caller must use before authoritative
    commit; a nonzero status never authorizes a CPU retry or partial commit.
    """

    if type(batch) is not int or batch <= 0:
        raise TensorWorldContractError("C4 identity staging batch must be a positive int")
    if type(capacity) is not int or capacity <= 0:
        raise TensorWorldContractError(
            "C4 identity staging capacity must be a positive int"
        )
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise TensorWorldContractError(
            "C4 identity staging requires a CUDA device; CPU staging/fallback is forbidden"
        )
    if not isinstance(outputs, Mapping):
        raise TensorWorldContractError("C4 identity staging output must be a mapping")

    expected_names = set(C4_IDENTITY_STAGING_FIELDS_V1)
    actual_names = set(outputs)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise TensorWorldContractError(
            "C4 identity staging output ABI fields are not exact: "
            f"missing={missing}, extra={extra}"
        )

    shape_by_name: dict[str, tuple[int, ...]] = {
        "staged_parent_slots": (batch, capacity),
        "staged_child_slots": (batch, capacity),
        "staged_child_agent_ids": (batch, capacity),
        "staged_child_energy": (batch, capacity),
        "staged_count": (batch,),
        "queue_head": (batch,),
        "queue_count": (batch,),
        "status": (batch,),
    }
    dtype_by_name: dict[str, torch.dtype] = {
        "staged_parent_slots": torch.int64,
        "staged_child_slots": torch.int64,
        "staged_child_agent_ids": torch.int64,
        "staged_child_energy": torch.float64,
        "staged_count": torch.int64,
        "queue_head": torch.int64,
        "queue_count": torch.int64,
        "status": torch.int32,
    }
    for name in C4_IDENTITY_STAGING_FIELDS_V1:
        _require_cuda_tensor(
            outputs[name],
            name=f"staging.{name}",
            shape=shape_by_name[name],
            dtype=dtype_by_name[name],
            device=device,
        )

    status = outputs["status"]
    unknown_status = torch.ones_like(status, dtype=torch.bool)
    for code in C4_IDENTITY_NATIVE_STATUS_V1:
        unknown_status &= status != code
    if bool(torch.any(unknown_status).item()):
        values = sorted(
            int(value)
            for value in status.detach().to(device="cpu").unique().tolist()
        )
        raise TensorWorldContractError(
            f"C4 identity staging has unknown status code(s): {values}"
        )
    if require_success and bool(torch.any(status != 0).item()):
        values = [int(value) for value in status.detach().to(device="cpu").tolist()]
        failed = [
            {"world": index, "code": value, "name": C4_IDENTITY_NATIVE_STATUS_V1[value]}
            for index, value in enumerate(values)
            if value != 0
        ]
        raise TensorWorldContractError(
            "C4 identity preflight status is nonzero; fail-closed admission "
            f"rejects CPU fallback and partial commit: {failed}"
        )
    return dict(outputs)


def require_c4_identity_preflight_success_v1(
    outputs: Mapping[str, torch.Tensor],
    *,
    batch: int,
    capacity: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Admit only all-OK device staging; never fall back to a CPU route."""

    return validate_c4_identity_preflight_staging_v1(
        outputs,
        batch=batch,
        capacity=capacity,
        device=device,
        require_success=True,
    )


def execute_c4_identity_preflight_cuda(
    bundle: TensorWorldStateBundleV4,
    parent_slots: torch.Tensor,
    spawn_positions: torch.Tensor,
    planned_count: torch.Tensor,
    *,
    config: TensorWorldC4IdentityPreflightConfigV1 | None = None,
    build_directory: str | Path | None = None,
    verbose: bool = False,
    require_success: bool = False,
) -> dict[str, torch.Tensor]:
    """Run the non-mutating resident C4 identity/slot preflight.

    ``parent_slots`` and ``spawn_positions`` are fixed-capacity plan rows. The
    source preflight writes only returned staging tensors. The output ABI is
    validated before return. Set ``require_success`` at the transaction
    admission boundary to reject any nonzero per-world status; no CPU retry or
    partial authoritative commit is permitted.
    """

    if not isinstance(bundle, TensorWorldStateBundleV4):
        raise TensorWorldContractError("C4 identity preflight requires a C4 state bundle")
    bundle.validate()
    if bundle.device.type != "cuda":
        raise TensorWorldContractError(
            "C4 resident identity preflight requires a CUDA state bundle"
        )
    config = config or TensorWorldC4IdentityPreflightConfigV1()
    device = bundle.device
    batch, capacity = bundle.batch, bundle.capacity
    _require_cuda_tensor(
        parent_slots,
        name="parent_slots",
        shape=(batch, capacity),
        dtype=torch.int64,
        device=device,
    )
    _require_cuda_tensor(
        spawn_positions,
        name="spawn_positions",
        shape=(batch, capacity, 2),
        dtype=torch.int32,
        device=device,
    )
    _require_cuda_tensor(
        planned_count,
        name="planned_count",
        shape=(batch,),
        dtype=torch.int64,
        device=device,
    )

    core = bundle.base.base.core
    agents = core.agents
    population = bundle.population
    _require_cuda_tensor(
        bundle.base.base.patch.agent_grid_i32,
        name="agent_grid",
        shape=(batch, bundle.grid_size, bundle.grid_size),
        dtype=torch.int32,
        device=device,
    )
    module = load_tensor_world_c4_identity_extension(
        build_directory=build_directory, verbose=verbose
    )
    outputs = module.identity_preflight(
        agents.agent_id,
        agents.alive,
        bundle.base.base.patch.agent_grid_i32,
        agents.energy,
        agents.health,
        population.repro_cooldown,
        bundle.base.agent.repro_threshold,
        bundle.base.agent.policy_energy_observation,
        population.free_queue,
        population.free_queue_count,
        population.next_agent_id,
        parent_slots,
        spawn_positions,
        planned_count,
        float(config.parent_energy_share),
        float(config.minimum_energy_buffer),
        float(config.start_health),
        int(config.cooldown_steps),
        int(config.generation),
        int(config.tick),
    )
    names = (
        "staged_parent_slots",
        "staged_child_slots",
        "staged_child_agent_ids",
        "staged_child_energy",
        "staged_count",
        "queue_head",
        "queue_count",
        "status",
    )
    if len(outputs) != len(names):
        raise TensorWorldContractError("C4 native identity output ABI is incomplete")
    mapped = dict(zip(names, outputs, strict=True))
    return validate_c4_identity_preflight_staging_v1(
        mapped,
        batch=batch,
        capacity=capacity,
        device=device,
        require_success=require_success,
    )


__all__ = [
    "C4_IDENTITY_NATIVE_ENTRYPOINT_V1",
    "C4_IDENTITY_NATIVE_SCHEMA_VERSION_V1",
    "C4_IDENTITY_NATIVE_STATUS_V1",
    "C4_IDENTITY_STAGING_FIELDS_V1",
    "C4_IDENTITY_STAGING_FIELD_SPECS_V1",
    "C4_IDENTITY_STAGING_SCHEMA_VERSION_V1",
    "C4_IDENTITY_STAGING_STATUS_POLICY_V1",
    "TensorWorldC4IdentityPreflightConfigV1",
    "c4_identity_native_source_manifest_v1",
    "execute_c4_identity_preflight_cuda",
    "load_tensor_world_c4_identity_extension",
    "require_c4_identity_preflight_success_v1",
    "validate_c4_identity_preflight_staging_v1",
]
