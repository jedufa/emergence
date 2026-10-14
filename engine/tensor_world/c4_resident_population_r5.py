"""B4 resident C4 population transaction over R3 death and R4 birth shadows."""

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
    TensorWorldC4ResidentMutationR4StagingV1,
    _replay_tensors,
    _scalar_rng_tensors,
    _staging_tensors,
    _validate_replay_structure,
    make_c4_resident_mutation_r4_staging_v1,
)
from .c4_resident_turnover_r3 import (
    TensorWorldC4ResidentTurnoverR3StagingV1,
    _death_authoritative,
    _turnover_staging_tensors,
    make_c4_resident_turnover_r3_staging_v1,
)
from .c4_state import TensorWorldStateBundleV4
from .c5_determinism import TensorWorldC5ReplayContextV1
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = ROOT / "engine" / "tensor_world" / "native"
PERF_ROOT = ROOT / "engine" / "perf"
NATIVE_BINDING = NATIVE_ROOT / "tensor_world_c4_resident_population_r5_binding.cpp"
NATIVE_KERNEL = NATIVE_ROOT / "tensor_world_c4_resident_population_r5.cu"
R3_SOURCE = NATIVE_ROOT / "tensor_world_c4_resident_turnover_r3.cu"
R4_SOURCE = NATIVE_ROOT / "tensor_world_c4_resident_mutation_r4.cu"
CPU_POPULATION = ROOT / "engine" / "tensor_world" / "c4_population.py"
CPU_MATING = ROOT / "engine" / "tensor_world" / "c4_mating.py"
CPU_GENOME = ROOT / "engine" / "tensor_world" / "c4_genome.py"

C4_RESIDENT_POPULATION_R5_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_population_r5_v1"
)
C4_RESIDENT_POPULATION_R5_STAGING_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_population_r5_staging_v1"
)
C4_RESIDENT_POPULATION_R5_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_POPULATION_R5_STATUS_V1: Final[dict[int, str]] = {
    0: "ok",
    40: "batch_global_peer_failed",
    41: "invalid_pending_free_or_death_state",
    42: "death_or_event_capacity_exceeded",
    43: "unsupported_identity_capacity",
    44: "death_identity_or_occupancy_inconsistent",
}


@dataclass(slots=True)
class TensorWorldC4ResidentPopulationR5StagingV1:
    """Caller-owned B4 death shadow and mate/mutation transaction buffers."""

    turnover: TensorWorldC4ResidentTurnoverR3StagingV1
    mutation: TensorWorldC4ResidentMutationR4StagingV1
    shadow_event_sequence: torch.Tensor
    schema_version: str = C4_RESIDENT_POPULATION_R5_STAGING_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.mutation.device

    def validate(
        self, *, batch: int, capacity: int, grid_size: int, streams: int
    ) -> None:
        if self.schema_version != C4_RESIDENT_POPULATION_R5_STAGING_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 R5 staging schema is unsupported")
        if batch != 4:
            raise TensorWorldContractError("C4 R5 staging is bounded to B4")
        self.turnover.validate(batch=batch, capacity=capacity, grid_size=grid_size)
        self.mutation.validate(batch=batch, capacity=capacity, streams=streams)
        if (
            tuple(self.shadow_event_sequence.shape) != (batch,)
            or self.shadow_event_sequence.dtype != torch.int64
            or not self.shadow_event_sequence.is_contiguous()
        ):
            raise TensorWorldContractError("C4 R5 shadow event sequence ABI mismatch")
        devices = {
            self.device,
            self.turnover.device,
            self.shadow_event_sequence.device,
        }
        if self.device.type != "cuda" or len(devices) != 1:
            raise TensorWorldContractError("C4 R5 staging requires one CUDA device")


def make_c4_resident_population_r5_staging_v1(
    *,
    batch: int,
    capacity: int,
    grid_size: int,
    streams: int,
    device: torch.device | str,
) -> TensorWorldC4ResidentPopulationR5StagingV1:
    """Allocate the reusable B4 transaction envelope without a CPU fallback."""

    if type(batch) is not int or batch != 4:
        raise TensorWorldContractError("C4 R5 staging is bounded to B4")
    target = torch.device(device)
    if target.type != "cuda":
        raise TensorWorldContractError("C4 R5 staging requires CUDA; CPU fallback forbidden")
    staging = TensorWorldC4ResidentPopulationR5StagingV1(
        turnover=make_c4_resident_turnover_r3_staging_v1(
            batch=batch, capacity=capacity, grid_size=grid_size, device=target
        ),
        mutation=make_c4_resident_mutation_r4_staging_v1(
            batch=batch, capacity=capacity, streams=streams, device=target
        ),
        shadow_event_sequence=torch.zeros(batch, dtype=torch.int64, device=target),
    )
    staging.validate(
        batch=batch, capacity=capacity, grid_size=grid_size, streams=streams
    )
    return staging


def _r5_turnover_tensors(
    staging: TensorWorldC4ResidentPopulationR5StagingV1,
) -> list[torch.Tensor]:
    return [*_turnover_staging_tensors(staging.turnover), staging.shadow_event_sequence]


_MODULE: Any | None = None


def load_tensor_world_c4_resident_population_r5_extension(
    *, build_directory: str | Path | None = None, verbose: bool = False
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise TensorWorldContractError("C4 R5 resident population requires CUDA")
    digest = sha256(
        b"\0".join(path.read_bytes() for path in (NATIVE_BINDING, NATIVE_KERNEL, R4_SOURCE))
    ).hexdigest()[:16]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=f"tensor_world_c4_resident_population_r5_{digest}",
        sources=[str(NATIVE_BINDING), str(NATIVE_KERNEL)],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(PERF_ROOT), str(NATIVE_ROOT)],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def execute_c4_resident_population_r5_cuda_v1(
    bundle: TensorWorldStateBundleV4,
    staging: TensorWorldC4ResidentPopulationR5StagingV1,
    config: TensorWorldC4MateBirthConfigV1,
    *,
    rng_context: TensorWorldC5ReplayContextV1,
    policy_vector_recombination: bool = False,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> TensorWorldC4ResidentPopulationR5StagingV1:
    """Execute one B4 batch-global resident death/mate/mutation transaction."""

    if not isinstance(bundle, TensorWorldStateBundleV4):
        raise TensorWorldContractError("C4 R5 requires a V4 bundle")
    if not isinstance(config, TensorWorldC4MateBirthConfigV1):
        raise TensorWorldContractError("C4 R5 mate birth config is required")
    if not config.allow_mutation:
        raise TensorWorldContractError("C4 R5 requires source-current scalar mutation")
    if policy_vector_recombination:
        raise TensorWorldContractError("C4 V4 has no policy-vector recombination contract")
    if config.life_history_mode != 0:
        raise TensorWorldContractError("C4 R5 non-default life-history is unsupported")
    if bundle.device.type != "cuda":
        raise TensorWorldContractError("C4 R5 requires a CUDA V4 bundle")
    if bundle.batch != 4:
        raise TensorWorldContractError("C4 R5 is bounded to B4")
    if bundle.events is None:
        raise TensorWorldContractError("C4 R5 typed event tape is required")
    bundle.validate()
    streams = _validate_replay_structure(bundle, rng_context)
    if config.environment_stream == config.runner_stream:
        raise TensorWorldContractError("C4 R5 environment and runner streams must differ")
    if config.environment_stream >= streams or config.runner_stream >= streams:
        raise TensorWorldContractError("C4 R5 stream mapping is outside replay state")
    staging.validate(
        batch=4, capacity=bundle.capacity, grid_size=bundle.grid_size, streams=streams
    )
    if staging.device != bundle.device:
        raise TensorWorldContractError("C4 R5 staging must share the bundle device")
    scalar_tensors = _scalar_rng_tensors(bundle)
    for tensor in scalar_tensors:
        if tensor.device != bundle.device or not tensor.is_contiguous():
            raise TensorWorldContractError("C4 R5 CPython RNG must be resident and contiguous")
    max_births = bundle.capacity if config.max_births is None else config.max_births
    module = load_tensor_world_c4_resident_population_r5_extension(
        build_directory=build_directory, verbose=verbose
    )
    module.population_transaction(
        _primary_tensors(bundle), _death_authoritative(bundle),
        _inherit_tensors(bundle), _zero_tensors(bundle), _minus_one_tensors(bundle),
        _half_tensors(bundle), _one_tensors(bundle), scalar_tensors,
        _replay_tensors(rng_context), _staging_tensors(staging.mutation),
        _r5_turnover_tensors(staging), int(max_births),
        float(config.parent_energy_share), float(config.minimum_energy_buffer),
        int(config.cooldown_steps), float(config.start_health), float(config.max_energy),
        int(config.life_history_mode), int(config.maturity_start_ticks),
        int(config.maturity_rise_ticks), int(config.fertility_decline_start_ticks),
        int(config.fertility_decline_ticks), float(config.elder_fertility_floor),
        int(config.generation), int(bundle.base.tick), int(config.environment_stream),
        int(config.runner_stream),
    )
    return staging


def c4_resident_population_r5_source_receipt_v1() -> dict[str, object]:
    sources = {
        "adapter": Path(__file__), "binding": NATIVE_BINDING, "kernel": NATIVE_KERNEL,
        "r3_post_death_shadow": R3_SOURCE, "r4_mate_mt_transaction": R4_SOURCE,
        "cpu_population_oracle": CPU_POPULATION, "cpu_mate_oracle": CPU_MATING,
        "cpu_scalar_oracle": CPU_GENOME,
    }
    return {
        "schema_version": C4_RESIDENT_POPULATION_R5_SCHEMA_VERSION_V1,
        "route": "R5_B4_MIXED_WORLD_DEATH_MATE_MUTATION",
        "decision": C4_RESIDENT_POPULATION_R5_DECISION_V1,
        "batch": 4,
        "preflight": "R3_post_death_shadow_then_R4_mate_MT_plan",
        "atomicity": "batch_global_guarded_multi_kernel_commit_boundary",
        "runtime_fault_atomicity": "not_claimed_for_multi_kernel_launch_faults",
        "event_order": "death_then_mutation_then_birth_per_world",
        "authoritative_parity": "full_V4_death_birth_reset_rng_and_typed_event_surfaces",
        "identity_capacity": "bounded_agent_id_to_slot_domain_fail_closed_no_wrap",
        "policy_vector_recombination": "unsupported_fail_closed",
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
    "C4_RESIDENT_POPULATION_R5_DECISION_V1",
    "C4_RESIDENT_POPULATION_R5_SCHEMA_VERSION_V1",
    "C4_RESIDENT_POPULATION_R5_STAGING_SCHEMA_VERSION_V1",
    "C4_RESIDENT_POPULATION_R5_STATUS_V1",
    "TensorWorldC4ResidentPopulationR5StagingV1",
    "c4_resident_population_r5_source_receipt_v1",
    "execute_c4_resident_population_r5_cuda_v1",
    "load_tensor_world_c4_resident_population_r5_extension",
    "make_c4_resident_population_r5_staging_v1",
]
