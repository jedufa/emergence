"""B48 one-tick resident C4 population transaction over frozen R5/R6 semantics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load

from .c4_mating import TensorWorldC4MateBirthConfigV1
from .c4_resident_birth_r2 import (
    _half_tensors, _inherit_tensors, _minus_one_tensors, _one_tensors,
    _primary_tensors, _zero_tensors,
)
from .c4_resident_mutation_r4 import (
    TensorWorldC4ResidentMutationR4StagingV1,
    _replay_tensors, _scalar_rng_tensors, _staging_tensors,
    _validate_replay_structure, make_c4_resident_mutation_r4_staging_v1,
)
from .c4_resident_turnover_r3 import (
    TensorWorldC4ResidentTurnoverR3StagingV1,
    _death_authoritative, _turnover_staging_tensors,
    make_c4_resident_turnover_r3_staging_v1,
)
from .c4_state import TensorWorldStateBundleV4
from .c5_determinism import TensorWorldC5ReplayContextV1
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = ROOT / "engine" / "tensor_world" / "native"
PERF_ROOT = ROOT / "engine" / "perf"
NATIVE_BINDING = NATIVE_ROOT / "tensor_world_c4_resident_population_r9_binding.cpp"
NATIVE_KERNEL = NATIVE_ROOT / "tensor_world_c4_resident_population_r9.cu"
R5_NATIVE = NATIVE_ROOT / "tensor_world_c4_resident_population_r5.cu"
R6_NATIVE = NATIVE_ROOT / "tensor_world_c4_resident_preflight_r6.cu"

C4_RESIDENT_POPULATION_R9_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_population_r9_v1"
)
C4_RESIDENT_POPULATION_R9_STAGING_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_population_r9_staging_v1"
)
C4_RESIDENT_POPULATION_R9_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_POPULATION_R9_BATCH_V1: Final[int] = 48


@dataclass(slots=True)
class TensorWorldC4ResidentPopulationR9StagingV1:
    turnover: TensorWorldC4ResidentTurnoverR3StagingV1
    mutation: TensorWorldC4ResidentMutationR4StagingV1
    shadow_event_sequence: torch.Tensor
    preflight_status: torch.Tensor
    schema_version: str = C4_RESIDENT_POPULATION_R9_STAGING_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.mutation.device

    def validate(
        self, *, batch: int, capacity: int, grid_size: int, streams: int
    ) -> None:
        if self.schema_version != C4_RESIDENT_POPULATION_R9_STAGING_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 R9 staging schema is unsupported")
        if batch != C4_RESIDENT_POPULATION_R9_BATCH_V1:
            raise TensorWorldContractError("C4 R9 staging is bounded to B48")
        self.turnover.validate(batch=batch, capacity=capacity, grid_size=grid_size)
        self.mutation.validate(batch=batch, capacity=capacity, streams=streams)
        expected = (
            (self.shadow_event_sequence, torch.int64),
            (self.preflight_status, torch.int32),
        )
        devices = {self.device, self.turnover.device}
        for tensor, dtype in expected:
            if (
                tuple(tensor.shape) != (batch,)
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise TensorWorldContractError("C4 R9 batch status ABI mismatch")
            devices.add(tensor.device)
        if self.device.type != "cuda" or len(devices) != 1:
            raise TensorWorldContractError("C4 R9 staging requires one CUDA device")


def make_c4_resident_population_r9_staging_v1(
    *, batch: int, capacity: int, grid_size: int, streams: int,
    device: torch.device | str,
) -> TensorWorldC4ResidentPopulationR9StagingV1:
    target = torch.device(device)
    if type(batch) is not int or batch != C4_RESIDENT_POPULATION_R9_BATCH_V1:
        raise TensorWorldContractError("C4 R9 staging is bounded to B48")
    if target.type != "cuda":
        raise TensorWorldContractError("C4 R9 staging requires CUDA; CPU fallback forbidden")
    staging = TensorWorldC4ResidentPopulationR9StagingV1(
        turnover=make_c4_resident_turnover_r3_staging_v1(
            batch=batch, capacity=capacity, grid_size=grid_size, device=target
        ),
        mutation=make_c4_resident_mutation_r4_staging_v1(
            batch=batch, capacity=capacity, streams=streams, device=target
        ),
        shadow_event_sequence=torch.zeros(batch, dtype=torch.int64, device=target),
        preflight_status=torch.zeros(batch, dtype=torch.int32, device=target),
    )
    staging.validate(
        batch=batch, capacity=capacity, grid_size=grid_size, streams=streams
    )
    return staging


def _r9_turnover_tensors(
    staging: TensorWorldC4ResidentPopulationR9StagingV1,
) -> list[torch.Tensor]:
    return [
        *_turnover_staging_tensors(staging.turnover),
        staging.shadow_event_sequence,
    ]


_MODULE: Any | None = None


def load_tensor_world_c4_resident_population_r9_extension(
    *, build_directory: str | Path | None = None, verbose: bool = False
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise TensorWorldContractError("C4 R9 resident population requires CUDA")
    digest = sha256(
        b"\0".join(path.read_bytes() for path in (NATIVE_BINDING, NATIVE_KERNEL, R6_NATIVE))
    ).hexdigest()[:16]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=f"tensor_world_c4_resident_population_r9_{digest}",
        sources=[str(NATIVE_BINDING), str(NATIVE_KERNEL)],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(PERF_ROOT), str(NATIVE_ROOT)],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def execute_c4_resident_population_r9_cuda_v1(
    bundle: TensorWorldStateBundleV4,
    staging: TensorWorldC4ResidentPopulationR9StagingV1,
    config: TensorWorldC4MateBirthConfigV1,
    *, rng_context: TensorWorldC5ReplayContextV1,
    build_directory: str | Path | None = None, verbose: bool = False,
) -> TensorWorldC4ResidentPopulationR9StagingV1:
    """Execute one host-launched B48 transaction without host semantic planning."""

    if not isinstance(bundle, TensorWorldStateBundleV4):
        raise TensorWorldContractError("C4 R9 requires a V4 bundle")
    if not isinstance(config, TensorWorldC4MateBirthConfigV1) or not config.allow_mutation:
        raise TensorWorldContractError("C4 R9 requires source-current mate mutation")
    if config.life_history_mode != 0:
        raise TensorWorldContractError("C4 R9 non-default life-history is unsupported")
    if bundle.device.type != "cuda":
        raise TensorWorldContractError("C4 R9 requires a CUDA V4 bundle")
    if bundle.batch != C4_RESIDENT_POPULATION_R9_BATCH_V1:
        raise TensorWorldContractError("C4 R9 is bounded to B48")
    if bundle.events is None:
        raise TensorWorldContractError("C4 R9 typed event tape is required")
    bundle.validate()
    streams = _validate_replay_structure(bundle, rng_context)
    if config.environment_stream == config.runner_stream:
        raise TensorWorldContractError("C4 R9 environment and runner streams must differ")
    if config.environment_stream >= streams or config.runner_stream >= streams:
        raise TensorWorldContractError("C4 R9 stream mapping is outside replay state")
    staging.validate(
        batch=bundle.batch, capacity=bundle.capacity,
        grid_size=bundle.grid_size, streams=streams,
    )
    if staging.device != bundle.device:
        raise TensorWorldContractError("C4 R9 staging must share the bundle device")
    scalar_tensors = _scalar_rng_tensors(bundle)
    for tensor in scalar_tensors:
        if tensor.device != bundle.device or not tensor.is_contiguous():
            raise TensorWorldContractError("C4 R9 CPython RNG must be resident and contiguous")
    max_births = bundle.capacity if config.max_births is None else config.max_births
    module = load_tensor_world_c4_resident_population_r9_extension(
        build_directory=build_directory, verbose=verbose
    )
    module.population_transaction(
        _primary_tensors(bundle), _death_authoritative(bundle),
        _inherit_tensors(bundle), _zero_tensors(bundle), _minus_one_tensors(bundle),
        _half_tensors(bundle), _one_tensors(bundle), scalar_tensors,
        _replay_tensors(rng_context), _staging_tensors(staging.mutation),
        _r9_turnover_tensors(staging), staging.preflight_status, int(max_births),
        float(config.parent_energy_share), float(config.minimum_energy_buffer),
        int(config.cooldown_steps), float(config.start_health), float(config.max_energy),
        int(config.life_history_mode), int(config.maturity_start_ticks),
        int(config.maturity_rise_ticks), int(config.fertility_decline_start_ticks),
        int(config.fertility_decline_ticks), float(config.elder_fertility_floor),
        int(config.generation), int(bundle.base.tick), int(config.environment_stream),
        int(config.runner_stream),
    )
    return staging


def c4_resident_population_r9_source_receipt_v1() -> dict[str, object]:
    sources = {
        "adapter": Path(__file__), "binding": NATIVE_BINDING, "kernel": NATIVE_KERNEL,
        "frozen_r5_native": R5_NATIVE, "frozen_r6_native": R6_NATIVE,
    }
    return {
        "schema_version": C4_RESIDENT_POPULATION_R9_SCHEMA_VERSION_V1,
        "route": "R9_B48_ONE_TICK_ALL_LIVE_PREFLIGHT_MATE_MUTATION",
        "decision": C4_RESIDENT_POPULATION_R9_DECISION_V1,
        "batch": C4_RESIDENT_POPULATION_R9_BATCH_V1,
        "launch_geometry": "dynamic_batch_blocks",
        "r5_r6_source_modified": False,
        "execution_classification": "host_launched_device_local_not_cuda_graph_bound",
        "runtime_boundary_counters": "unmeasured",
        "runtime_fault_atomicity": "not_claimed_for_multi_kernel_launch_faults",
        "gate_claim": False,
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path.read_bytes()).hexdigest()}
            for name, path in sources.items()
        },
    }


__all__ = [
    "C4_RESIDENT_POPULATION_R9_BATCH_V1",
    "C4_RESIDENT_POPULATION_R9_DECISION_V1",
    "C4_RESIDENT_POPULATION_R9_SCHEMA_VERSION_V1",
    "C4_RESIDENT_POPULATION_R9_STAGING_SCHEMA_VERSION_V1",
    "TensorWorldC4ResidentPopulationR9StagingV1",
    "c4_resident_population_r9_source_receipt_v1",
    "execute_c4_resident_population_r9_cuda_v1",
    "load_tensor_world_c4_resident_population_r9_extension",
    "make_c4_resident_population_r9_staging_v1",
]
