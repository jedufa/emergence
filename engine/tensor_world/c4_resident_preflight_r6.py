"""R6 all-live semantic preflight wrapped around the frozen resident R5 route."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load

from .c4_mating import TensorWorldC4MateBirthConfigV1
from .c4_resident_birth_r2 import (
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _primary_tensors,
    _zero_tensors,
)
from .c4_resident_mutation_r4 import (
    _replay_tensors,
    _scalar_rng_tensors,
    _staging_tensors,
    _validate_replay_structure,
)
from .c4_resident_population_r5 import (
    C4_RESIDENT_POPULATION_R5_SCHEMA_VERSION_V1,
    TensorWorldC4ResidentPopulationR5StagingV1,
    _r5_turnover_tensors,
    make_c4_resident_population_r5_staging_v1,
)
from .c4_resident_turnover_r3 import _death_authoritative
from .c4_state import TensorWorldStateBundleV4
from .c5_determinism import TensorWorldC5ReplayContextV1
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = ROOT / "engine" / "tensor_world" / "native"
PERF_ROOT = ROOT / "engine" / "perf"
NATIVE_BINDING = NATIVE_ROOT / "tensor_world_c4_resident_preflight_r6_binding.cpp"
NATIVE_KERNEL = NATIVE_ROOT / "tensor_world_c4_resident_preflight_r6.cu"
R5_ADAPTER = ROOT / "engine" / "tensor_world" / "c4_resident_population_r5.py"
R5_NATIVE = NATIVE_ROOT / "tensor_world_c4_resident_population_r5.cu"

C4_RESIDENT_PREFLIGHT_R6_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_preflight_r6_v1"
)
C4_RESIDENT_PREFLIGHT_R6_STAGING_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_preflight_r6_staging_v1"
)
C4_RESIDENT_PREFLIGHT_R6_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_PREFLIGHT_R6_STATUS_V1: Final[dict[int, str]] = {
    0: "ok",
    50: "invalid_or_duplicate_free_queue",
    51: "live_identity_or_id_map_inconsistent",
    52: "occupancy_or_patch_mirror_inconsistent",
    53: "batch_global_peer_failed",
}


@dataclass(slots=True)
class TensorWorldC4ResidentPreflightR6StagingV1:
    population: TensorWorldC4ResidentPopulationR5StagingV1
    status: torch.Tensor
    schema_version: str = C4_RESIDENT_PREFLIGHT_R6_STAGING_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.status.device

    def validate(
        self, *, batch: int, capacity: int, grid_size: int, streams: int
    ) -> None:
        if self.schema_version != C4_RESIDENT_PREFLIGHT_R6_STAGING_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 R6 staging schema is unsupported")
        if batch != 4:
            raise TensorWorldContractError("C4 R6 staging is bounded to B4")
        self.population.validate(
            batch=batch, capacity=capacity, grid_size=grid_size, streams=streams
        )
        if (
            tuple(self.status.shape) != (batch,)
            or self.status.dtype != torch.int32
            or not self.status.is_contiguous()
            or self.status.device != self.population.device
            or self.status.device.type != "cuda"
        ):
            raise TensorWorldContractError(
                "C4 R6 status requires contiguous CUDA int32 [B]"
            )


def make_c4_resident_preflight_r6_staging_v1(
    *,
    batch: int,
    capacity: int,
    grid_size: int,
    streams: int,
    device: torch.device | str,
) -> TensorWorldC4ResidentPreflightR6StagingV1:
    """Allocate caller-owned R6 status and the frozen R5 staging envelope."""

    target = torch.device(device)
    if type(batch) is not int or batch != 4:
        raise TensorWorldContractError("C4 R6 staging is bounded to B4")
    if target.type != "cuda":
        raise TensorWorldContractError("C4 R6 staging requires CUDA; CPU fallback forbidden")
    staging = TensorWorldC4ResidentPreflightR6StagingV1(
        population=make_c4_resident_population_r5_staging_v1(
            batch=batch,
            capacity=capacity,
            grid_size=grid_size,
            streams=streams,
            device=target,
        ),
        status=torch.zeros(batch, dtype=torch.int32, device=target),
    )
    staging.validate(
        batch=batch, capacity=capacity, grid_size=grid_size, streams=streams
    )
    return staging


_MODULE: Any | None = None


def load_tensor_world_c4_resident_preflight_r6_extension(
    *, build_directory: str | Path | None = None, verbose: bool = False
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise TensorWorldContractError("C4 R6 resident preflight requires CUDA")
    digest = sha256(
        b"\0".join(path.read_bytes() for path in (NATIVE_BINDING, NATIVE_KERNEL, R5_NATIVE))
    ).hexdigest()[:16]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=f"tensor_world_c4_resident_preflight_r6_{digest}",
        sources=[str(NATIVE_BINDING), str(NATIVE_KERNEL)],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(PERF_ROOT), str(NATIVE_ROOT)],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def execute_c4_resident_preflight_r6_cuda_v1(
    bundle: TensorWorldStateBundleV4,
    staging: TensorWorldC4ResidentPreflightR6StagingV1,
    config: TensorWorldC4MateBirthConfigV1,
    *,
    rng_context: TensorWorldC5ReplayContextV1,
    policy_vector_recombination: bool = False,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> TensorWorldC4ResidentPreflightR6StagingV1:
    """Run all-live preflight and the frozen R5 plan/commit under one device guard."""

    if not isinstance(bundle, TensorWorldStateBundleV4):
        raise TensorWorldContractError("C4 R6 requires a V4 bundle")
    if not isinstance(config, TensorWorldC4MateBirthConfigV1):
        raise TensorWorldContractError("C4 R6 mate birth config is required")
    if not config.allow_mutation:
        raise TensorWorldContractError("C4 R6 requires source-current scalar mutation")
    if policy_vector_recombination:
        raise TensorWorldContractError("C4 V4 has no policy-vector recombination contract")
    if config.life_history_mode != 0:
        raise TensorWorldContractError("C4 R6 non-default life-history is unsupported")
    if bundle.device.type != "cuda":
        raise TensorWorldContractError("C4 R6 requires a CUDA V4 bundle")
    if bundle.batch != 4:
        raise TensorWorldContractError("C4 R6 is bounded to B4")
    if bundle.events is None:
        raise TensorWorldContractError("C4 R6 typed event tape is required")
    bundle.validate()
    streams = _validate_replay_structure(bundle, rng_context)
    if config.environment_stream == config.runner_stream:
        raise TensorWorldContractError("C4 R6 environment and runner streams must differ")
    if config.environment_stream >= streams or config.runner_stream >= streams:
        raise TensorWorldContractError("C4 R6 stream mapping is outside replay state")
    staging.validate(
        batch=4, capacity=bundle.capacity, grid_size=bundle.grid_size, streams=streams
    )
    if staging.device != bundle.device:
        raise TensorWorldContractError("C4 R6 staging must share the bundle device")
    scalar_tensors = _scalar_rng_tensors(bundle)
    for tensor in scalar_tensors:
        if tensor.device != bundle.device or not tensor.is_contiguous():
            raise TensorWorldContractError("C4 R6 CPython RNG must be resident and contiguous")
    max_births = bundle.capacity if config.max_births is None else config.max_births
    module = load_tensor_world_c4_resident_preflight_r6_extension(
        build_directory=build_directory, verbose=verbose
    )
    r5_staging = staging.population
    module.population_transaction(
        _primary_tensors(bundle), _death_authoritative(bundle),
        _inherit_tensors(bundle), _zero_tensors(bundle), _minus_one_tensors(bundle),
        _half_tensors(bundle), _one_tensors(bundle), scalar_tensors,
        _replay_tensors(rng_context), _staging_tensors(r5_staging.mutation),
        _r5_turnover_tensors(r5_staging), staging.status, int(max_births),
        float(config.parent_energy_share), float(config.minimum_energy_buffer),
        int(config.cooldown_steps), float(config.start_health), float(config.max_energy),
        int(config.life_history_mode), int(config.maturity_start_ticks),
        int(config.maturity_rise_ticks), int(config.fertility_decline_start_ticks),
        int(config.fertility_decline_ticks), float(config.elder_fertility_floor),
        int(config.generation), int(bundle.base.tick), int(config.environment_stream),
        int(config.runner_stream),
    )
    return staging


def c4_resident_preflight_r6_source_receipt_v1() -> dict[str, object]:
    sources = {
        "adapter": Path(__file__),
        "binding": NATIVE_BINDING,
        "kernel": NATIVE_KERNEL,
        "frozen_r5_adapter": R5_ADAPTER,
        "frozen_r5_native": R5_NATIVE,
    }
    return {
        "schema_version": C4_RESIDENT_PREFLIGHT_R6_SCHEMA_VERSION_V1,
        "route": "R6_B4_ALL_LIVE_PREFLIGHT_THEN_FROZEN_R5",
        "decision": C4_RESIDENT_PREFLIGHT_R6_DECISION_V1,
        "batch": 4,
        "caller_owned_status": "contiguous_cuda_int32_B",
        "preflight_scope": [
            "every_live_slot_identity_and_dense_id_map",
            "every_occupancy_cell_agent_slot_and_patch_mirror",
            "free_queue_range_occupancy_and_duplicate_rejection",
        ],
        "atomicity": "device_status_merged_before_any_authoritative_commit",
        "frozen_r5_schema_version": C4_RESIDENT_POPULATION_R5_SCHEMA_VERSION_V1,
        "r5_source_modified": False,
        "runtime_fault_atomicity": "not_claimed_for_multi_kernel_launch_faults",
        "production_seed_binding_claim": False,
        "gate_claim": False,
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


__all__ = [
    "C4_RESIDENT_PREFLIGHT_R6_DECISION_V1",
    "C4_RESIDENT_PREFLIGHT_R6_SCHEMA_VERSION_V1",
    "C4_RESIDENT_PREFLIGHT_R6_STAGING_SCHEMA_VERSION_V1",
    "C4_RESIDENT_PREFLIGHT_R6_STATUS_V1",
    "TensorWorldC4ResidentPreflightR6StagingV1",
    "c4_resident_preflight_r6_source_receipt_v1",
    "execute_c4_resident_preflight_r6_cuda_v1",
    "load_tensor_world_c4_resident_preflight_r6_extension",
    "make_c4_resident_preflight_r6_staging_v1",
]
