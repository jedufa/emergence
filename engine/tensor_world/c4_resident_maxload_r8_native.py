"""Native B4/C32 maximum-load probe for the frozen C4 R8 envelope."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load

from .c4_population import TensorWorldC4BirthConfigV1
from .c4_resident_birth_r2 import (
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _primary_tensors,
    _zero_tensors,
)
from .c4_resident_mutation_r4 import _validate_replay_structure
from .c4_resident_turnover_r3 import (
    TensorWorldC4ResidentTurnoverR3StagingV1,
    _death_authoritative,
    _turnover_staging_tensors,
)
from .c4_state import TensorWorldStateBundleV4
from .c5_determinism import TensorWorldC5ReplayContextV1
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = ROOT / "engine" / "tensor_world" / "native"
PERF_ROOT = ROOT / "engine" / "perf"
NATIVE_BINDING = NATIVE_ROOT / "tensor_world_c4_resident_maxload_r8_binding.cpp"
NATIVE_KERNEL = NATIVE_ROOT / "tensor_world_c4_resident_maxload_r8.cu"
R3_KERNEL = NATIVE_ROOT / "tensor_world_c4_resident_turnover_r3.cu"
R2_KERNEL = NATIVE_ROOT / "tensor_world_c4_resident_birth_r2.cu"
R5_ADAPTER = ROOT / "engine" / "tensor_world" / "c4_resident_population_r5.py"
R5_KERNEL = NATIVE_ROOT / "tensor_world_c4_resident_population_r5.cu"
CPU_ORACLE = ROOT / "engine" / "tensor_world" / "c4_population.py"
CPU_FIXTURE = ROOT / "engine" / "tensor_world" / "c4_resident_maxload_r8.py"

C4_RESIDENT_MAXLOAD_R8_NATIVE_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_maxload_r8_native_probe_v1"
)
C4_RESIDENT_MAXLOAD_R8_NATIVE_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_MAXLOAD_R8_NATIVE_STATUS_V1: Final[dict[int, str]] = {
    0: "ok",
    80: "r8_exact_8_death_8_birth_reuse_or_event_sequence_admission_failed",
}


def _birth_staging_tensors(
    staging: TensorWorldC4ResidentTurnoverR3StagingV1,
) -> list[torch.Tensor]:
    birth = staging.birth
    return [
        birth.spawn_uniforms, birth.parent_slots, birth.child_slots,
        birth.child_pos_y, birth.child_pos_x, birth.child_energy,
        birth.planned_count, birth.decision_parent_slots,
        birth.decision_adjacent_counts, birth.decision_reason_codes,
        birth.decision_count, birth.planned_repro_cooldown,
        birth.spawn_rng_draws_added, birth.spawn_rng_semantic_draws_added,
        birth.child_agent_ids, birth.queue_cursor, birth.event_sequence,
        birth.status,
    ]


def c4_resident_maxload_r8_native_source_receipt_v1() -> dict[str, object]:
    """Bind the probe to its actual ordinary-turnover and reviewed R5 seam."""

    sources = {
        "adapter": Path(__file__),
        "binding": NATIVE_BINDING,
        "kernel": NATIVE_KERNEL,
        "ordinary_turnover_primitive": R3_KERNEL,
        "ordinary_birth_primitive": R2_KERNEL,
        "reviewed_r5_adapter": R5_ADAPTER,
        "reviewed_r5_kernel": R5_KERNEL,
        "cpu_oracle": CPU_ORACLE,
        "cpu_fixture": CPU_FIXTURE,
    }
    return {
        "schema_version": C4_RESIDENT_MAXLOAD_R8_NATIVE_SCHEMA_VERSION_V1,
        "route": "R8_B4_C32_EXACT_8_DEATH_8_ORDINARY_BIRTH_MAXLOAD_PROBE",
        "decision": C4_RESIDENT_MAXLOAD_R8_NATIVE_DECISION_V1,
        "frozen_r5_seam": (
            "public_R5_is_mate_mutation_only_and_cannot_match_the_frozen_ordinary_"
            "no_mutation_R8_oracle; probe_reuses_the_shared_R3_R2_ordinary_primitives"
        ),
        "admission": "every_world_exactly_8_deaths_8_births_and_all_children_reuse_new_deaths",
        "event_rng_ownership": (
            "death_then_birth_typed_events; replay_event_sequence_committed_on_device; "
            "spawn_tape_only; mutation_and_PCG_state_preserved"
        ),
        "rollback": "batch_global_no_authoritative_mutation_on_any_preflight_failure",
        "execution_classification": "host_launched_component_probe_not_production_residency",
        "gate_claim": False,
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


_MODULE: Any | None = None


def load_tensor_world_c4_resident_maxload_r8_extension(
    *, build_directory: str | Path | None = None, verbose: bool = False
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise TensorWorldContractError("C4 native R8 requires CUDA")
    digest = sha256(
        b"\0".join(
            path.read_bytes()
            for path in (NATIVE_BINDING, NATIVE_KERNEL, R3_KERNEL, R2_KERNEL)
        )
    ).hexdigest()[:16]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=f"tensor_world_c4_resident_maxload_r8_{digest}",
        sources=[str(NATIVE_BINDING), str(NATIVE_KERNEL)],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(PERF_ROOT), str(NATIVE_ROOT)],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def execute_c4_resident_maxload_r8_cuda_v1(
    bundle: TensorWorldStateBundleV4,
    staging: TensorWorldC4ResidentTurnoverR3StagingV1,
    config: TensorWorldC4BirthConfigV1,
    *,
    replay: TensorWorldC5ReplayContextV1,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> TensorWorldC4ResidentTurnoverR3StagingV1:
    """Execute exactly the frozen B4/C32, 8+8 ordinary turnover envelope."""

    if not isinstance(bundle, TensorWorldStateBundleV4) or bundle.device.type != "cuda":
        raise TensorWorldContractError("C4 native R8 requires a CUDA V4 bundle")
    if bundle.batch != 4 or bundle.capacity != 32 or bundle.grid_size != 32:
        raise TensorWorldContractError("C4 native R8 is bounded to B4/C32/grid32")
    if bundle.events is None or bundle.events.capacity != 64:
        raise TensorWorldContractError("C4 native R8 requires 64 typed event rows")
    if not isinstance(config, TensorWorldC4BirthConfigV1):
        raise TensorWorldContractError("C4 native R8 ordinary birth config is required")
    if config.allow_mutation:
        raise TensorWorldContractError("C4 native R8 mutation is unsupported and fail-closed")
    if config.max_births != 8 or not config.require_spawn_rng_tape:
        raise TensorWorldContractError("C4 native R8 requires explicit tape and max_births=8")
    bundle.validate()
    _validate_replay_structure(bundle, replay)
    staging.validate(batch=4, capacity=32, grid_size=32)
    if staging.device != bundle.device:
        raise TensorWorldContractError("C4 native R8 staging must share the bundle device")
    module = load_tensor_world_c4_resident_maxload_r8_extension(
        build_directory=build_directory, verbose=verbose
    )
    module.maxload_transaction(
        _primary_tensors(bundle),
        _death_authoritative(bundle),
        _inherit_tensors(bundle),
        _zero_tensors(bundle),
        _minus_one_tensors(bundle),
        _half_tensors(bundle),
        _one_tensors(bundle),
        _birth_staging_tensors(staging),
        _turnover_staging_tensors(staging),
        replay.event_sequence,
        int(config.max_births),
        float(config.parent_energy_share),
        float(config.minimum_energy_buffer),
        int(config.cooldown_steps),
        float(config.start_health),
        float(config.max_energy),
        int(config.life_history_mode),
        int(config.maturity_start_ticks),
        int(config.maturity_rise_ticks),
        int(config.fertility_decline_start_ticks),
        int(config.fertility_decline_ticks),
        float(config.elder_fertility_floor),
        int(config.generation),
        int(bundle.base.tick),
    )
    return staging


__all__ = [
    "C4_RESIDENT_MAXLOAD_R8_NATIVE_DECISION_V1",
    "C4_RESIDENT_MAXLOAD_R8_NATIVE_SCHEMA_VERSION_V1",
    "C4_RESIDENT_MAXLOAD_R8_NATIVE_STATUS_V1",
    "c4_resident_maxload_r8_native_source_receipt_v1",
    "execute_c4_resident_maxload_r8_cuda_v1",
    "load_tensor_world_c4_resident_maxload_r8_extension",
]
