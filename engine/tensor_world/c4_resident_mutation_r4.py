"""Executable C4 R4 component: B1 resident mate birth and scalar mutation.

The source-current C4 oracle has two independent continuation owners.  Mate
selection and placement consume C5-compatible PCG64 streams; scalar genome
mutation consumes the C4 CPython MT19937/Gaussian-tape state.  This component
keeps both states caller-owned and commits them only after one device-side
preflight has staged every birth, genome row, reset, and typed event.

Policy-vector crossover is not part of the V4 CPU oracle.  The newborn policy
row and hidden-size gene therefore inherit from the primary parent exactly;
requests for any other policy inheritance mode fail before native execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load

from .c4_mating import TensorWorldC4MateBirthConfigV1
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
from .c5_determinism import TensorWorldC5ReplayContextV1
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = ROOT / "engine" / "tensor_world" / "native"
PERF_ROOT = ROOT / "engine" / "perf"
NATIVE_BINDING = NATIVE_ROOT / "tensor_world_c4_resident_mutation_r4_binding.cpp"
NATIVE_KERNEL = NATIVE_ROOT / "tensor_world_c4_resident_mutation_r4.cu"
R2_NATIVE = NATIVE_ROOT / "tensor_world_c4_resident_birth_r2.cu"
CPU_MATE = ROOT / "engine" / "tensor_world" / "c4_mating.py"
CPU_GENOME = ROOT / "engine" / "tensor_world" / "c4_genome.py"
CPU_POPULATION = ROOT / "engine" / "tensor_world" / "c4_population.py"
PCG_NATIVE = PERF_ROOT / "cuda_architecture_b_persistent_mate_birth_plan_v1.cuh"
MT_NATIVE = PERF_ROOT / "cuda_architecture_b_persistent_birth_scalar_genome.cuh"

C4_RESIDENT_MUTATION_R4_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_mutation_r4_v1"
)
C4_RESIDENT_MUTATION_R4_STAGING_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_mutation_r4_staging_v1"
)
C4_RESIDENT_MUTATION_R4_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_MUTATION_R4_STATUS_V1: Final[dict[int, str]] = {
    0: "ok",
    30: "invalid_configuration_or_stream_mapping",
    31: "invalid_resident_pcg64_state",
    32: "invalid_parent_mate_or_free_queue_state",
    33: "identity_or_event_capacity_exceeded",
    34: "unsupported_gru_parent_or_mate",
    35: "invalid_cpython_mt19937_state",
    36: "gaussian_tape_exhausted",
}


@dataclass(slots=True)
class TensorWorldC4ResidentMutationR4StagingV1:
    """Fixed-capacity plan, post-RNG shadows, and child scalar rows."""

    birth: TensorWorldC4ResidentBirthR2StagingV1
    mate_slots: torch.Tensor
    decision_mate_slots: torch.Tensor
    runner_rng_draws_added: torch.Tensor
    runner_rng_semantic_draws_added: torch.Tensor
    pcg64_post_state_words: torch.Tensor
    pcg64_post_buffer_words: torch.Tensor
    pcg64_post_cursors: torch.Tensor
    pcg64_post_semantic_draws: torch.Tensor
    python_post_words: torch.Tensor
    python_post_index: torch.Tensor
    python_post_has_gauss: torch.Tensor
    python_post_gauss_next: torch.Tensor
    python_post_raw_draws: torch.Tensor
    python_post_semantic_draws: torch.Tensor
    child_scalar_values: torch.Tensor
    mutation_occurred: torch.Tensor
    mutation_delta_total: torch.Tensor
    shadow_pos_y_i32: torch.Tensor
    shadow_pos_x_i32: torch.Tensor
    shadow_energy_f64: torch.Tensor
    planned_energy_f64: torch.Tensor
    planned_parent_energy_after: torch.Tensor
    planned_mate_energy_after: torch.Tensor
    spawn_positions_i32: torch.Tensor
    decision_spawn_candidate_counts: torch.Tensor
    reason_counts: torch.Tensor
    free_queue_head: torch.Tensor
    planned_parent_slots_i64: torch.Tensor
    planned_mate_slots_i64: torch.Tensor
    planned_child_slots_i64: torch.Tensor
    planned_child_energy_f64: torch.Tensor
    schema_version: str = C4_RESIDENT_MUTATION_R4_STAGING_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.mate_slots.device

    @property
    def streams(self) -> int:
        return int(self.pcg64_post_state_words.shape[1])

    def validate(self, *, batch: int, capacity: int, streams: int) -> None:
        if self.schema_version != C4_RESIDENT_MUTATION_R4_STAGING_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 R4 staging schema is unsupported")
        self.birth.validate(batch=batch, capacity=capacity)
        slots = (batch, capacity)
        expected = (
            ("mate_slots", self.mate_slots, slots, torch.int32),
            ("decision_mate_slots", self.decision_mate_slots, slots, torch.int64),
            ("runner_rng_draws_added", self.runner_rng_draws_added, (batch,), torch.int64),
            ("runner_rng_semantic_draws_added", self.runner_rng_semantic_draws_added, (batch,), torch.int64),
            ("pcg64_post_state_words", self.pcg64_post_state_words, (batch, streams, 4), torch.int64),
            ("pcg64_post_buffer_words", self.pcg64_post_buffer_words, (batch, streams, 2), torch.int64),
            ("pcg64_post_cursors", self.pcg64_post_cursors, (batch, streams), torch.int64),
            ("pcg64_post_semantic_draws", self.pcg64_post_semantic_draws, (batch, streams), torch.int64),
            ("python_post_words", self.python_post_words, (batch, 624), torch.uint32),
            ("python_post_index", self.python_post_index, (batch,), torch.int64),
            ("python_post_has_gauss", self.python_post_has_gauss, (batch,), torch.bool),
            ("python_post_gauss_next", self.python_post_gauss_next, (batch,), torch.float64),
            ("python_post_raw_draws", self.python_post_raw_draws, (batch,), torch.int64),
            ("python_post_semantic_draws", self.python_post_semantic_draws, (batch,), torch.int64),
            ("child_scalar_values", self.child_scalar_values, (batch, capacity, 16), torch.float32),
            ("mutation_occurred", self.mutation_occurred, slots, torch.bool),
            ("mutation_delta_total", self.mutation_delta_total, slots, torch.float32),
            ("shadow_pos_y_i32", self.shadow_pos_y_i32, slots, torch.int32),
            ("shadow_pos_x_i32", self.shadow_pos_x_i32, slots, torch.int32),
            ("shadow_energy_f64", self.shadow_energy_f64, slots, torch.float64),
            ("planned_energy_f64", self.planned_energy_f64, slots, torch.float64),
            ("planned_parent_energy_after", self.planned_parent_energy_after, slots, torch.float64),
            ("planned_mate_energy_after", self.planned_mate_energy_after, slots, torch.float64),
            ("spawn_positions_i32", self.spawn_positions_i32, (batch, capacity, 2), torch.int32),
            ("decision_spawn_candidate_counts", self.decision_spawn_candidate_counts, slots, torch.int32),
            ("reason_counts", self.reason_counts, (batch, 12), torch.int64),
            ("free_queue_head", self.free_queue_head, (batch,), torch.int64),
            ("planned_parent_slots_i64", self.planned_parent_slots_i64, slots, torch.int64),
            ("planned_mate_slots_i64", self.planned_mate_slots_i64, slots, torch.int64),
            ("planned_child_slots_i64", self.planned_child_slots_i64, slots, torch.int64),
            ("planned_child_energy_f64", self.planned_child_energy_f64, slots, torch.float64),
        )
        devices = {self.device, self.birth.device}
        for name, tensor, shape, dtype in expected:
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise TensorWorldContractError(
                    f"C4 R4 staging {name} requires contiguous shape={shape!r} dtype={dtype}"
                )
            devices.add(tensor.device)
        if self.device.type != "cuda" or len(devices) != 1:
            raise TensorWorldContractError("C4 R4 staging requires one CUDA device")


def make_c4_resident_mutation_r4_staging_v1(
    *, batch: int, capacity: int, streams: int, device: torch.device | str
) -> TensorWorldC4ResidentMutationR4StagingV1:
    """Allocate the reusable R4 transaction envelope."""

    target = torch.device(device)
    if (
        type(batch) is not int
        or batch <= 0
        or type(capacity) is not int
        or capacity <= 0
        or type(streams) is not int
        or streams < 2
    ):
        raise TensorWorldContractError("C4 R4 dimensions require positive B/C and at least two streams")
    if target.type != "cuda":
        raise TensorWorldContractError("C4 R4 staging requires CUDA; CPU fallback is forbidden")
    slots = (batch, capacity)
    staging = TensorWorldC4ResidentMutationR4StagingV1(
        birth=make_c4_resident_birth_r2_staging_v1(
            batch=batch, capacity=capacity, device=target
        ),
        mate_slots=torch.full(slots, -1, dtype=torch.int32, device=target),
        decision_mate_slots=torch.full(slots, -1, dtype=torch.int64, device=target),
        runner_rng_draws_added=torch.zeros(batch, dtype=torch.int64, device=target),
        runner_rng_semantic_draws_added=torch.zeros(batch, dtype=torch.int64, device=target),
        pcg64_post_state_words=torch.zeros((batch, streams, 4), dtype=torch.int64, device=target),
        pcg64_post_buffer_words=torch.zeros((batch, streams, 2), dtype=torch.int64, device=target),
        pcg64_post_cursors=torch.zeros((batch, streams), dtype=torch.int64, device=target),
        pcg64_post_semantic_draws=torch.zeros((batch, streams), dtype=torch.int64, device=target),
        python_post_words=torch.zeros((batch, 624), dtype=torch.uint32, device=target),
        python_post_index=torch.zeros(batch, dtype=torch.int64, device=target),
        python_post_has_gauss=torch.zeros(batch, dtype=torch.bool, device=target),
        python_post_gauss_next=torch.zeros(batch, dtype=torch.float64, device=target),
        python_post_raw_draws=torch.zeros(batch, dtype=torch.int64, device=target),
        python_post_semantic_draws=torch.zeros(batch, dtype=torch.int64, device=target),
        child_scalar_values=torch.zeros((batch, capacity, 16), dtype=torch.float32, device=target),
        mutation_occurred=torch.zeros(slots, dtype=torch.bool, device=target),
        mutation_delta_total=torch.zeros(slots, dtype=torch.float32, device=target),
        shadow_pos_y_i32=torch.zeros(slots, dtype=torch.int32, device=target),
        shadow_pos_x_i32=torch.zeros(slots, dtype=torch.int32, device=target),
        shadow_energy_f64=torch.zeros(slots, dtype=torch.float64, device=target),
        planned_energy_f64=torch.zeros(slots, dtype=torch.float64, device=target),
        planned_parent_energy_after=torch.zeros(slots, dtype=torch.float64, device=target),
        planned_mate_energy_after=torch.zeros(slots, dtype=torch.float64, device=target),
        spawn_positions_i32=torch.full((batch, capacity, 2), -1, dtype=torch.int32, device=target),
        decision_spawn_candidate_counts=torch.zeros(slots, dtype=torch.int32, device=target),
        reason_counts=torch.zeros((batch, 12), dtype=torch.int64, device=target),
        free_queue_head=torch.zeros(batch, dtype=torch.int64, device=target),
        planned_parent_slots_i64=torch.full(slots, -1, dtype=torch.int64, device=target),
        planned_mate_slots_i64=torch.full(slots, -1, dtype=torch.int64, device=target),
        planned_child_slots_i64=torch.full(slots, -1, dtype=torch.int64, device=target),
        planned_child_energy_f64=torch.zeros(slots, dtype=torch.float64, device=target),
    )
    staging.validate(batch=batch, capacity=capacity, streams=streams)
    return staging


def _validate_replay_structure(
    bundle: TensorWorldStateBundleV4,
    replay: TensorWorldC5ReplayContextV1,
) -> int:
    """Validate the resident ABI without reading or synchronizing CUDA values."""

    if not isinstance(replay, TensorWorldC5ReplayContextV1):
        raise TensorWorldContractError("C4 R4 requires a C5 replay context")
    if replay.rng_algorithm != "pcg64_words_v1":
        raise TensorWorldContractError("C4 R4 requires pcg64_words_v1")
    if replay.rng_buffer_words is None or replay.semantic_draw_counters is None:
        raise TensorWorldContractError("C4 R4 requires resident PCG64 buffer and semantic counters")
    streams = int(replay.rng_state_words.shape[1]) if replay.rng_state_words.ndim == 3 else -1
    expected = (
        (replay.rng_state_words, (bundle.batch, streams, 4), torch.int64),
        (replay.rng_buffer_words, (bundle.batch, streams, 2), torch.int64),
        (replay.rng_cursors, (bundle.batch, streams), torch.int64),
        (replay.semantic_draw_counters, (bundle.batch, streams), torch.int64),
        (replay.event_sequence, (bundle.batch,), torch.int64),
        (replay.event_order_keys, (bundle.batch, bundle.capacity), torch.int64),
    )
    devices = {bundle.device}
    for tensor, shape, dtype in expected:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype or not tensor.is_contiguous():
            raise TensorWorldContractError("C4 R4 replay tensor ABI mismatch")
        devices.add(tensor.device)
    if streams < 2 or len(devices) != 1:
        raise TensorWorldContractError("C4 R4 replay state requires at least two colocated streams")
    return streams


def _scalar_rng_tensors(bundle: TensorWorldStateBundleV4) -> list[torch.Tensor]:
    state = bundle.scalar_rng
    if state is None:
        raise TensorWorldContractError("C4 R4 requires explicit CPython RNG state")
    tensors = [
        state.python_random_words,
        state.python_random_index,
        state.python_random_has_gauss,
        state.python_random_gauss_next,
        state.python_random_gauss_primary_tape,
        state.python_random_gauss_secondary_tape,
        state.python_random_raw_draws,
        state.mutation_py_semantic_draws,
    ]
    tape_capacity = int(state.python_random_gauss_primary_tape.shape[1])
    expected = (
        (tensors[0], (bundle.batch, 624), torch.uint32),
        (tensors[1], (bundle.batch,), torch.int64),
        (tensors[2], (bundle.batch,), torch.bool),
        (tensors[3], (bundle.batch,), torch.float64),
        (tensors[4], (bundle.batch, tape_capacity), torch.float64),
        (tensors[5], (bundle.batch, tape_capacity), torch.float64),
        (tensors[6], (bundle.batch,), torch.int64),
        (tensors[7], (bundle.batch,), torch.int64),
    )
    for tensor, shape, dtype in expected:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise TensorWorldContractError("C4 R4 CPython RNG tensor ABI mismatch")
    return tensors


def _replay_tensors(replay: TensorWorldC5ReplayContextV1) -> list[torch.Tensor]:
    assert replay.rng_buffer_words is not None
    assert replay.semantic_draw_counters is not None
    return [
        replay.rng_state_words,
        replay.rng_buffer_words,
        replay.rng_cursors,
        replay.semantic_draw_counters,
        replay.event_sequence,
        replay.event_order_keys,
    ]


def _staging_tensors(staging: TensorWorldC4ResidentMutationR4StagingV1) -> list[torch.Tensor]:
    b = staging.birth
    return [
        b.spawn_uniforms, b.parent_slots, b.child_slots, b.child_pos_y,
        b.child_pos_x, b.child_energy, b.planned_count,
        b.decision_parent_slots, b.decision_adjacent_counts,
        b.decision_reason_codes, b.decision_count, b.planned_repro_cooldown,
        b.spawn_rng_draws_added, b.spawn_rng_semantic_draws_added,
        b.child_agent_ids, b.queue_cursor, b.event_sequence, b.status,
        staging.mate_slots, staging.decision_mate_slots,
        staging.runner_rng_draws_added, staging.runner_rng_semantic_draws_added,
        staging.pcg64_post_state_words, staging.pcg64_post_buffer_words,
        staging.pcg64_post_cursors, staging.pcg64_post_semantic_draws,
        staging.python_post_words, staging.python_post_index,
        staging.python_post_has_gauss, staging.python_post_gauss_next,
        staging.python_post_raw_draws, staging.python_post_semantic_draws,
        staging.child_scalar_values, staging.mutation_occurred,
        staging.mutation_delta_total,
        staging.shadow_pos_y_i32, staging.shadow_pos_x_i32,
        staging.shadow_energy_f64, staging.planned_energy_f64,
        staging.planned_parent_energy_after, staging.planned_mate_energy_after,
        staging.spawn_positions_i32, staging.decision_spawn_candidate_counts,
        staging.reason_counts, staging.free_queue_head,
        staging.planned_parent_slots_i64, staging.planned_mate_slots_i64,
        staging.planned_child_slots_i64, staging.planned_child_energy_f64,
    ]


_MODULE: Any | None = None


def load_tensor_world_c4_resident_mutation_r4_extension(
    *, build_directory: str | Path | None = None, verbose: bool = False
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise TensorWorldContractError("C4 R4 resident mutation requires CUDA")
    digest = sha256(
        b"\0".join(path.read_bytes() for path in (NATIVE_BINDING, NATIVE_KERNEL, R2_NATIVE))
    ).hexdigest()[:16]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=f"tensor_world_c4_resident_mutation_r4_{digest}",
        sources=[str(NATIVE_BINDING), str(NATIVE_KERNEL)],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(PERF_ROOT), str(NATIVE_ROOT)],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def execute_c4_resident_mutation_r4_cuda_v1(
    bundle: TensorWorldStateBundleV4,
    staging: TensorWorldC4ResidentMutationR4StagingV1,
    config: TensorWorldC4MateBirthConfigV1,
    *,
    rng_context: TensorWorldC5ReplayContextV1,
    policy_vector_recombination: bool = False,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> TensorWorldC4ResidentMutationR4StagingV1:
    """Execute one batch-global resident mate/mutation transaction."""

    if not isinstance(bundle, TensorWorldStateBundleV4):
        raise TensorWorldContractError("C4 R4 requires a V4 bundle")
    if not isinstance(config, TensorWorldC4MateBirthConfigV1):
        raise TensorWorldContractError("C4 R4 mate birth config is required")
    if not config.allow_mutation:
        raise TensorWorldContractError("C4 R4 requires source-current scalar mutation/recombination")
    if policy_vector_recombination:
        raise TensorWorldContractError("C4 V4 has no policy-vector recombination contract")
    if config.life_history_mode != 0:
        raise TensorWorldContractError(
            "C4 R4 non-default life-history mate fertility is not yet source-exact"
        )
    if bundle.device.type != "cuda":
        raise TensorWorldContractError("C4 R4 requires a CUDA V4 bundle")
    if bundle.batch != 1:
        raise TensorWorldContractError("C4 R4 is bounded to B1")
    if bundle.events is None:
        raise TensorWorldContractError("C4 R4 typed event tape is required")
    streams = _validate_replay_structure(bundle, rng_context)
    if config.environment_stream == config.runner_stream:
        raise TensorWorldContractError("C4 R4 environment and runner streams must be distinct")
    if config.environment_stream >= streams or config.runner_stream >= streams:
        raise TensorWorldContractError("C4 R4 stream mapping is outside the replay context")
    staging.validate(batch=1, capacity=bundle.capacity, streams=streams)
    if staging.device != bundle.device:
        raise TensorWorldContractError("C4 R4 staging must share the bundle device")
    scalar_tensors = _scalar_rng_tensors(bundle)
    for tensor in scalar_tensors:
        if tensor.device != bundle.device or not tensor.is_contiguous():
            raise TensorWorldContractError("C4 R4 CPython RNG must be contiguous and resident")
    max_births = bundle.capacity if config.max_births is None else config.max_births
    module = load_tensor_world_c4_resident_mutation_r4_extension(
        build_directory=build_directory, verbose=verbose
    )
    module.mutation_transaction(
        _primary_tensors(bundle), _inherit_tensors(bundle), _zero_tensors(bundle),
        _minus_one_tensors(bundle), _half_tensors(bundle), _one_tensors(bundle),
        scalar_tensors, _replay_tensors(rng_context), _staging_tensors(staging),
        int(max_births), float(config.parent_energy_share),
        float(config.minimum_energy_buffer), int(config.cooldown_steps),
        float(config.start_health), float(config.max_energy),
        int(config.life_history_mode), int(config.maturity_start_ticks),
        int(config.maturity_rise_ticks), int(config.fertility_decline_start_ticks),
        int(config.fertility_decline_ticks), float(config.elder_fertility_floor),
        int(config.generation), int(bundle.base.tick), int(config.environment_stream),
        int(config.runner_stream),
    )
    return staging


def c4_resident_mutation_r4_source_receipt_v1() -> dict[str, object]:
    sources = {
        "adapter": Path(__file__), "binding": NATIVE_BINDING, "kernel": NATIVE_KERNEL,
        "r2_reset_commit_surface": R2_NATIVE, "cpu_mate_oracle": CPU_MATE,
        "cpu_scalar_oracle": CPU_GENOME, "cpu_population_oracle": CPU_POPULATION,
        "pcg64_native_contract": PCG_NATIVE, "mt19937_native_contract": MT_NATIVE,
    }
    return {
        "schema_version": C4_RESIDENT_MUTATION_R4_SCHEMA_VERSION_V1,
        "route": "R4_B1_MATE_SCALAR_MUTATION_RECOMBINATION",
        "decision": C4_RESIDENT_MUTATION_R4_DECISION_V1,
        "continuation_owners": ["C5_PCG64_MATE_PLACEMENT", "C4_CPYTHON_MT19937_SCALAR"],
        "hidden_and_policy_inheritance": "primary_parent_copy",
        "policy_vector_recombination": "unsupported_fail_closed",
        "life_history_mode": "default_zero_only_nondefault_fail_closed",
        "atomicity": "all_device_preflights_before_batch_guarded_commit",
        "newborn_reset": "R2_complete_V4_reset_surface",
        "typed_events": True,
        "production_seed_binding_claim": False,
        "gate_claim": False,
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path.read_bytes()).hexdigest()}
            for name, path in sources.items()
        },
    }


__all__ = [
    "C4_RESIDENT_MUTATION_R4_DECISION_V1",
    "C4_RESIDENT_MUTATION_R4_SCHEMA_VERSION_V1",
    "C4_RESIDENT_MUTATION_R4_STAGING_SCHEMA_VERSION_V1",
    "C4_RESIDENT_MUTATION_R4_STATUS_V1",
    "TensorWorldC4ResidentMutationR4StagingV1",
    "c4_resident_mutation_r4_source_receipt_v1",
    "execute_c4_resident_mutation_r4_cuda_v1",
    "load_tensor_world_c4_resident_mutation_r4_extension",
    "make_c4_resident_mutation_r4_staging_v1",
]
