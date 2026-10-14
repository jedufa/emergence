"""Bounded CUDA-event autotuning for a sealed Full Evolution V2 prebuilt.

This is a technical-only microbenchmark.  It times only the device transition
between CUDA events.  Event draining, content validation, sink acknowledgement,
and all host projections are deliberately outside that interval.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from statistics import median
from time import perf_counter
from typing import Any, Callable, Final, Sequence

from .contract import (
    CUDA_NATIVE_ROUTE_V2,
    CUDA_NATIVE_TARGET_WTPS_100X,
    MAX_SIGNED_INT64,
    RESIDENT_CAPACITY_C_V2,
    CudaFullEvolutionNativeContractV2,
    cuda_native_required_packed_tick_ms_100x_v2,
    cuda_native_required_packed_ticks_per_second_100x_v2,
)
from .layout import (
    CudaNativeBatchAdmissionV2,
    CudaNativeResourceLedgerV2,
    estimate_cuda_native_resources_v2,
    select_cuda_native_batch_v2,
)
from .native_loader import (
    CudaFullEvolutionNativeLoadError,
    LoadedCudaFullEvolutionNativeV2,
    PROJECT_ROOT,
    cuda_device_identity_v2,
    file_sha256,
    load_prebuilt_cuda_full_evolution_v2,
    validate_prebuilt_cuda_full_evolution_v2,
)
from .runtime import (
    COUNTER_NAMES_V2,
    EXECUTION_MODES_V2,
    EXECUTION_PLAN_RECEIPT_SCHEMA_V2,
    EXECUTION_PLAN_TOPOLOGY_SCHEMA_V2,
    FULL_CAPABILITY_MASK_V2,
    FULL_CAPACITY_BENCHMARK_FIXTURE_V2,
    NATIVE_EXECUTION_RECEIPT_SCHEMA_V2,
    CudaFullEvolutionCheckpointV2,
    CudaFullEvolutionEventDispatchV2,
    CudaFullEvolutionRuntimeV2,
    seal_event_sink_receipt_v2,
)

BENCHMARK_SCHEMA_V2: Final[str] = "eiel_full_evolution_cuda_event_autotune_v1"
BENCHMARK_CLAIM_CLASS_V2: Final[str] = "TECHNICAL_ONLY"
BENCHMARK_SINK_ID_V2: Final[str] = "cuda_event_autotune_validated_discard_sink_v1"
STATIC_BATCH_ANCHORS_V2: Final[tuple[int, ...]] = (
    1,
    2,
    4,
    8,
    16,
    32,
    68,
    136,
)
# Omit --batches to derive a bounded, measured-device B grid.  Seven required
# K values and two required execution modes leave four exact B slots per stage.
DEFAULT_BATCH_CANDIDATES_V2: Final[tuple[int, ...] | None] = None
DEFAULT_TEMPORAL_K_CANDIDATES_V2: Final[tuple[int, ...]] = (1, 2, 4, 8, 16, 32, 64)
CUDA_GRAPH_KERNEL_V2: Final[str] = "advance_optimized_kernel"
CUDA_GRAPH_EXECUTION_MODE_V2: Final[str] = "cuda_graph_monolith"
EAGER_EXECUTION_MODE_V2: Final[str] = "eager_monolith"
DEFAULT_WARMUP_DISPATCHES_V2: Final[int] = 1
DEFAULT_SAMPLES_PER_CANDIDATE_V2: Final[int] = 3
DEFAULT_RESERVE_FLOOR_BYTES_V2: Final[int] = 256 << 20
MAX_CANDIDATES_PER_PLAN_V2: Final[int] = 64
MAX_BATCH_CANDIDATES_PER_PLAN_V2: Final[int] = MAX_CANDIDATES_PER_PLAN_V2 // (
    len(DEFAULT_TEMPORAL_K_CANDIDATES_V2) * len(EXECUTION_MODES_V2)
)
MAX_BATCH_CANDIDATES_PER_ENVELOPE_V2: Final[int] = 64
STAGED_ENVELOPE_SCHEMA_V2: Final[str] = "eiel_cuda_event_autotune_staged_envelope_v1"
RESOLVED_ENVELOPE_SCHEMA_V2: Final[str] = (
    "eiel_cuda_event_autotune_resolved_envelope_v1"
)
B136_CAPACITY_SOURCE_SCHEMA_V2: Final[str] = "eiel_cuda_event_autotune_b136_capacity_v1"
MAX_DISPATCHES_PER_CANDIDATE_V2: Final[int] = 8
MAX_TICKS_PER_CANDIDATE_V2: Final[int] = 512
AUTOTUNE_ARTIFACT_ROOT_V2: Final[Path] = (
    PROJECT_ROOT / "_artifacts" / "tw-cuda-full-evolution-reform-20260825"
)
PERFORMANCE_RESTRUCTURE_CONTRACT_V2: Final[Path] = (
    AUTOTUNE_ARTIFACT_ROOT_V2 / "performance_restructure_contract.json"
)
VALIDATION_MATRIX_V2: Final[Path] = AUTOTUNE_ARTIFACT_ROOT_V2 / "validation_matrix.json"
SOURCE_VALIDATED_ORIGINAL_SWEEP_AUTHORITY_V2: Final[str] = (
    "SOURCE_VALIDATED_ORIGINAL_SWEEP_NON_AUTHORITATIVE"
)
LAUNCH_ORIGIN_SCHEMA_V2: Final[str] = "eiel_cuda_event_autotune_launch_origin_v1"
LAUNCH_PLAN_IDENTITY_SCHEMA_V2: Final[str] = (
    "eiel_cuda_event_autotune_launch_plan_identity_v1"
)
VRAM_OBSERVATION_KEYS_V2: Final[frozenset[str]] = frozenset(
    {"free_bytes", "total_bytes", "allocated_bytes", "reserved_bytes"}
)
MAX_LAUNCH_ORIGIN_BYTES_V2: Final[int] = 1 << 20


class CudaFullEvolutionAutotuneError(RuntimeError):
    """Raised when a bounded V2 autotune request is not safe or exact."""


def _strict_positive(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise CudaFullEvolutionAutotuneError(f"{name} must be a positive integer")
    return value


def _strict_nonnegative(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise CudaFullEvolutionAutotuneError(f"{name} must be a nonnegative integer")
    return value


def _strict_candidate_tuple(
    values: tuple[int, ...], *, name: str, maximum: int | None = None
) -> tuple[int, ...]:
    if type(values) is not tuple or not values:
        raise CudaFullEvolutionAutotuneError(f"{name} must be a nonempty tuple")
    if any(type(value) is not int or value <= 0 for value in values):
        raise CudaFullEvolutionAutotuneError(f"{name} must contain positive integers")
    if maximum is not None and any(value > maximum for value in values):
        raise CudaFullEvolutionAutotuneError(f"{name} contains a value above {maximum}")
    if tuple(sorted(set(values))) != values:
        raise CudaFullEvolutionAutotuneError(
            f"{name} must be strictly increasing and duplicate-free"
        )
    return values


def _maximum_working_bytes_v2(available_vram_bytes: int, reserve_bytes: int) -> int:
    _strict_nonnegative(available_vram_bytes, "available_vram_bytes")
    _strict_nonnegative(reserve_bytes, "reserve_bytes")
    if available_vram_bytes > MAX_SIGNED_INT64 or reserve_bytes > MAX_SIGNED_INT64:
        raise CudaFullEvolutionAutotuneError(
            "VRAM quantities must fit signed int64 admission arithmetic"
        )
    usable = max(0, available_vram_bytes - reserve_bytes)
    # Match select_cuda_native_batch_v2's >=25% headroom arithmetic exactly.
    return (usable // 4) * 3 + ((usable % 4) * 3) // 4


def _checked_candidate_count_v2(
    batch_candidates: tuple[int, ...], temporal_k_candidates: tuple[int, ...]
) -> None:
    del temporal_k_candidates
    if len(batch_candidates) > MAX_BATCH_CANDIDATES_PER_ENVELOPE_V2:
        raise CudaFullEvolutionAutotuneError(
            "autotune envelope exceeds the bounded B candidate count"
        )


@dataclass(frozen=True, slots=True)
class CudaEventAutotunePlanV2:
    artifact_path: Path
    batch_candidates: tuple[int, ...] | None = DEFAULT_BATCH_CANDIDATES_V2
    temporal_k_candidates: tuple[int, ...] = DEFAULT_TEMPORAL_K_CANDIDATES_V2
    warmup_dispatches: int = DEFAULT_WARMUP_DISPATCHES_V2
    samples_per_candidate: int = DEFAULT_SAMPLES_PER_CANDIDATE_V2
    seed: int = 71
    device: str = "cuda"
    reserve_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_path, Path):
            raise CudaFullEvolutionAutotuneError("artifact_path must be a Path")
        if self.batch_candidates is not None:
            _strict_candidate_tuple(self.batch_candidates, name="batch_candidates")
        _strict_candidate_tuple(
            self.temporal_k_candidates, name="temporal_k_candidates", maximum=64
        )
        if self.temporal_k_candidates != DEFAULT_TEMPORAL_K_CANDIDATES_V2:
            raise CudaFullEvolutionAutotuneError(
                "temporal_k_candidates must be the exact required K grid "
                "(1, 2, 4, 8, 16, 32, 64)"
            )
        _strict_nonnegative(self.warmup_dispatches, "warmup_dispatches")
        _strict_positive(self.samples_per_candidate, "samples_per_candidate")
        _strict_nonnegative(self.seed, "seed")
        if not isinstance(self.device, str) or not self.device.strip():
            raise CudaFullEvolutionAutotuneError("device must be a nonempty string")
        if self.reserve_bytes is not None:
            _strict_nonnegative(self.reserve_bytes, "reserve_bytes")
        if self.batch_candidates is not None:
            _checked_candidate_count_v2(
                self.batch_candidates, self.temporal_k_candidates
            )
        if self.dispatches_per_candidate > MAX_DISPATCHES_PER_CANDIDATE_V2:
            raise CudaFullEvolutionAutotuneError(
                "autotune plan exceeds the bounded dispatch count per candidate"
            )
        if (
            self.dispatches_per_candidate * max(self.temporal_k_candidates)
            > MAX_TICKS_PER_CANDIDATE_V2
        ):
            raise CudaFullEvolutionAutotuneError(
                "autotune plan exceeds the bounded absolute-tick span per candidate"
            )

    @property
    def dispatches_per_candidate(self) -> int:
        return self.warmup_dispatches + self.samples_per_candidate


@dataclass(frozen=True, slots=True)
class CudaAutotuneCandidateV2:
    batch_size: int
    temporal_k: int


@dataclass(frozen=True, slots=True)
class CudaAutotuneCapacityV2:
    candidate: CudaAutotuneCandidateV2
    free_vram_bytes: int
    total_vram_bytes: int
    reserve_bytes: int
    ledger: CudaNativeResourceLedgerV2
    admission: CudaNativeBatchAdmissionV2

    @property
    def admitted(self) -> bool:
        return self.admission.admitted


@dataclass(frozen=True, slots=True)
class CudaActivePopulationSummaryV2:
    world_count: int
    total_active_population: int
    minimum_active_population: int
    maximum_active_population: int


@dataclass(frozen=True, slots=True)
class CudaEventTimingSampleV2:
    cuda_event_elapsed_ms: float
    committed_world_ticks: int
    drain_projection_seconds: float
    drain_reported_seconds: float
    content_hash_seconds: float
    acknowledgement_seconds: float
    start_population: CudaActivePopulationSummaryV2
    end_population: CudaActivePopulationSummaryV2
    start_population_projection_seconds: float
    end_population_projection_seconds: float
    event_content_sha256: str = ""
    final_state_sha256: str = ""


@dataclass(frozen=True, slots=True)
class CudaCausalFixtureEvidenceV2:
    """Non-timed causal proof bound to the sealed repeated starting state."""

    fixture_id: str
    starting_checkpoint_metadata_sha256: str
    starting_state_sha256: str
    starting_counters: dict[str, int]
    final_counters: dict[str, int]
    starting_population: CudaActivePopulationSummaryV2
    final_population: CudaActivePopulationSummaryV2
    event_content_sha256s: tuple[str, ...]
    final_state_sha256s: tuple[str, ...]
    valid: bool
    invalid_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CudaExecutionModeEvidenceV2:
    """Copied native receipt for one eager launch or one real graph replay."""

    execution_mode: str
    topology: dict[str, Any]
    launch_receipt: dict[str, Any]
    topology_sha256: str
    launch_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class CudaC512SaturationEvidenceV2:
    """Independent timing from a sealed exact-C512 starting fixture."""

    candidate: CudaAutotuneCandidateV2
    execution_mode: str
    artifact_sha256: str
    manifest_sha256: str
    device_identity: dict[str, Any]
    seed: int
    population_fixture_id: str
    fixture_id: str
    starting_checkpoint_metadata_sha256: str
    starting_state_sha256: str
    starting_counters: dict[str, int]
    final_counters: dict[str, int]
    starting_population: CudaActivePopulationSummaryV2
    final_population: CudaActivePopulationSummaryV2
    sample: CudaEventTimingSampleV2
    execution_evidence: CudaExecutionModeEvidenceV2
    fixture_lineage_sha256: str
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class CudaAutotuneMeasurementV2:
    candidate: CudaAutotuneCandidateV2
    status: str
    capacity: CudaAutotuneCapacityV2
    execution_mode: str = EAGER_EXECUTION_MODE_V2
    samples: tuple[CudaEventTimingSampleV2, ...] = ()
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    peak_device_process_bytes: int | None = None
    allocated_after_initialization_bytes: int | None = None
    reserved_after_initialization_bytes: int | None = None
    device_process_after_initialization_bytes: int | None = None
    initial_population: CudaActivePopulationSummaryV2 | None = None
    initial_population_projection_seconds: float | None = None
    final_population: CudaActivePopulationSummaryV2 | None = None
    fixture: CudaCausalFixtureEvidenceV2 | None = None
    execution_evidence: tuple[CudaExecutionModeEvidenceV2, ...] = ()
    stage_index: int | None = None

    @property
    def committed_world_ticks(self) -> int:
        return sum(sample.committed_world_ticks for sample in self.samples)

    @property
    def cuda_event_elapsed_ms(self) -> float:
        return sum(sample.cuda_event_elapsed_ms for sample in self.samples)

    @property
    def committed_world_ticks_per_second(self) -> float | None:
        if not self.samples or self.cuda_event_elapsed_ms <= 0:
            return None
        return self.committed_world_ticks * 1000.0 / self.cuda_event_elapsed_ms

    @property
    def median_sample_world_ticks_per_second(self) -> float | None:
        if not self.samples:
            return None
        return float(
            median(
                sample.committed_world_ticks * 1000.0 / sample.cuda_event_elapsed_ms
                for sample in self.samples
            )
        )

    @property
    def p10_sample_world_ticks_per_second(self) -> float | None:
        percentiles = self.sample_percentiles_world_ticks_per_second()
        return None if percentiles is None else percentiles["p10"]

    def sample_percentiles_world_ticks_per_second(self) -> dict[str, float] | None:
        if not self.samples:
            return None
        values = sorted(
            sample.committed_world_ticks * 1000.0 / sample.cuda_event_elapsed_ms
            for sample in self.samples
        )
        return {
            "p10": _percentile_v2(values, 0.10),
            "p50": _percentile_v2(values, 0.50),
            "p90": _percentile_v2(values, 0.90),
        }


@dataclass(frozen=True, slots=True)
class CudaAutotuneStageV2:
    """One deterministic <=64-arm portion of the complete B/K envelope."""

    stage_index: int
    batch_candidates: tuple[int, ...]
    candidates: tuple[CudaAutotuneCandidateV2, ...]


@dataclass(frozen=True, slots=True)
class CudaEventAutotuneResultV2:
    artifact_path: str
    artifact_sha256: str
    manifest_sha256: str
    device_identity: dict[str, Any]
    initial_vram: dict[str, int]
    reserve_bytes: int
    plan: CudaEventAutotunePlanV2
    resolved_batch_candidates: tuple[int, ...]
    maximum_vram_admitted_batch_size: int
    measurements: tuple[CudaAutotuneMeasurementV2, ...]
    selected: CudaAutotuneMeasurementV2
    harness_sha256: str
    performance_restructure_contract_sha256: str = ""
    validation_matrix_sha256: str = ""
    topology_identity: dict[str, Any] = field(default_factory=dict)
    target_verdict: dict[str, Any] = field(default_factory=dict)
    b136_comparison: dict[str, Any] = field(default_factory=dict)
    staged_envelope: dict[str, Any] = field(default_factory=dict)
    execution_mode_comparison: dict[str, Any] = field(default_factory=dict)
    c512_saturation_evidence: CudaC512SaturationEvidenceV2 | None = None
    schema_version: str = BENCHMARK_SCHEMA_V2
    claim_class: str = BENCHMARK_CLAIM_CLASS_V2
    launch_origin: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def capacity(value: CudaAutotuneCapacityV2) -> dict[str, Any]:
            return {
                "candidate": asdict(value.candidate),
                "free_vram_bytes": value.free_vram_bytes,
                "total_vram_bytes": value.total_vram_bytes,
                "reserve_bytes": value.reserve_bytes,
                "ledger": asdict(value.ledger),
                "admission": asdict(value.admission),
            }

        def measurement(value: CudaAutotuneMeasurementV2) -> dict[str, Any]:
            return {
                "authority_class": "PRELIMINARY_CLAIMANT_NON_AUTHORITATIVE",
                "authoritative": False,
                "candidate": asdict(value.candidate),
                "execution_mode": value.execution_mode,
                "status": value.status,
                "capacity": capacity(value.capacity),
                "horizon_dependent_resident_bytes": (
                    _horizon_dependent_resident_bytes_v2(value.candidate)
                ),
                "resident_allocation": {
                    "resident_state_bytes": value.capacity.ledger.resident_state_bytes,
                    "bounded_dispatch_event_bytes": (
                        value.capacity.ledger.bounded_dispatch_event_bytes
                    ),
                    "law": "total_ticks does not alter resident allocation; dispatch span K is separately bounded",
                },
                "samples": [asdict(sample) for sample in value.samples],
                "committed_world_ticks": value.committed_world_ticks,
                "cuda_event_elapsed_ms": value.cuda_event_elapsed_ms,
                "committed_world_ticks_per_second": (
                    value.committed_world_ticks_per_second
                ),
                "median_sample_world_ticks_per_second": (
                    value.median_sample_world_ticks_per_second
                ),
                "world_ticks_per_second_percentiles": (
                    value.sample_percentiles_world_ticks_per_second()
                ),
                "peak_allocated_bytes": value.peak_allocated_bytes,
                "peak_reserved_bytes": value.peak_reserved_bytes,
                "peak_device_process_bytes": value.peak_device_process_bytes,
                "allocated_after_initialization_bytes": (
                    value.allocated_after_initialization_bytes
                ),
                "reserved_after_initialization_bytes": (
                    value.reserved_after_initialization_bytes
                ),
                "device_process_after_initialization_bytes": (
                    value.device_process_after_initialization_bytes
                ),
                "initial_population": (
                    None
                    if value.initial_population is None
                    else asdict(value.initial_population)
                ),
                "initial_population_projection_seconds": (
                    value.initial_population_projection_seconds
                ),
                "final_population": (
                    None
                    if value.final_population is None
                    else asdict(value.final_population)
                ),
                "fixture": (None if value.fixture is None else asdict(value.fixture)),
                "execution_evidence": [
                    asdict(evidence) for evidence in value.execution_evidence
                ],
                "stage_index": value.stage_index,
                "required_world_ticks_per_second_100x": str(
                    CUDA_NATIVE_TARGET_WTPS_100X
                ),
                "required_packed_ticks_per_second_100x": str(
                    cuda_native_required_packed_ticks_per_second_100x_v2(
                        value.candidate.batch_size
                    )
                ),
                "required_packed_tick_ms_100x": str(
                    cuda_native_required_packed_tick_ms_100x_v2(
                        value.candidate.batch_size
                    )
                ),
            }

        return {
            "schema_version": self.schema_version,
            "claim_class": self.claim_class,
            "authority_class": "PRELIMINARY_CLAIMANT_NON_AUTHORITATIVE",
            "authoritative": False,
            "route_id": CUDA_NATIVE_ROUTE_V2,
            "artifact": {
                "path": self.artifact_path,
                "sha256": self.artifact_sha256,
                "manifest_sha256": self.manifest_sha256,
            },
            "device_identity": self.device_identity,
            "initial_vram": self.initial_vram,
            "launch_origin": self.launch_origin,
            "reserve_bytes": self.reserve_bytes,
            "peak_allocation": {
                "authority_class": "PRELIMINARY_CLAIMANT_NON_AUTHORITATIVE",
                "authoritative": False,
                "maximum_allocated_bytes": max(
                    (
                        value.peak_allocated_bytes
                        for value in self.measurements
                        if value.peak_allocated_bytes is not None
                    ),
                    default=None,
                ),
                "maximum_reserved_bytes": max(
                    (
                        value.peak_reserved_bytes
                        for value in self.measurements
                        if value.peak_reserved_bytes is not None
                    ),
                    default=None,
                ),
                "maximum_device_process_bytes": max(
                    (
                        value.peak_device_process_bytes
                        for value in self.measurements
                        if value.peak_device_process_bytes is not None
                    ),
                    default=None,
                ),
            },
            "plan": {
                "batch_candidates": (
                    "dynamic"
                    if self.plan.batch_candidates is None
                    else list(self.plan.batch_candidates)
                ),
                "resolved_batch_candidates": list(self.resolved_batch_candidates),
                "maximum_vram_admitted_batch_size": (
                    self.maximum_vram_admitted_batch_size
                ),
                "temporal_k_candidates": list(self.plan.temporal_k_candidates),
                "warmup_dispatches": self.plan.warmup_dispatches,
                "samples_per_candidate": self.plan.samples_per_candidate,
                "seed": self.plan.seed,
                "device": self.plan.device,
            },
            "timing_contract": {
                "clock": "cuda_event_elapsed_ms",
                "timed_scope": "optimized CUDA transition only",
                "committed_only": True,
                "selection": {
                    "primary": "maximum p10 committed world-ticks per second",
                    "within_one_percent": (
                        "lower measured (allocated, reserved, device-process) peak tuple, then lower B, then lower temporal_k"
                    ),
                },
                "event_buffer": {
                    "device_event_row_buffers_per_owner": 1,
                    "device_event_count_buffers_per_owner": 1,
                    "double_buffered": False,
                    "topology_identity": self.topology_identity,
                    "resident_capacity_C": RESIDENT_CAPACITY_C_V2,
                    "shape": "[B, 2*C*temporal_k, 48] uint8",
                },
                "excluded_from_timing": [
                    "event drain and CPU projection",
                    "event content hash",
                    "sink receipt acknowledgement",
                    "completion validation",
                    "allocator statistics",
                ],
            },
            "measurements": [measurement(value) for value in self.measurements],
            "selected": measurement(self.selected),
            "harness_sha256": self.harness_sha256,
            "performance_restructure_contract_sha256": (
                self.performance_restructure_contract_sha256
            ),
            "validation_matrix_sha256": self.validation_matrix_sha256,
            "target_verdict": self.target_verdict,
            "b136_comparison": {
                **self.b136_comparison,
                "authority_class": "PRELIMINARY_CLAIMANT_NON_AUTHORITATIVE",
                "authoritative": False,
            },
            "staged_envelope": {
                **self.staged_envelope,
                "authority_class": "PRELIMINARY_CLAIMANT_NON_AUTHORITATIVE",
                "authoritative": False,
            },
            "execution_mode_comparison": {
                **self.execution_mode_comparison,
                "authority_class": "PRELIMINARY_CLAIMANT_NON_AUTHORITATIVE",
                "authoritative": False,
            },
            "c512_saturation_evidence": (
                None
                if self.c512_saturation_evidence is None
                else {
                    **asdict(self.c512_saturation_evidence),
                    "authority_class": "PRELIMINARY_CLAIMANT_NON_AUTHORITATIVE",
                    "authoritative": False,
                }
            ),
        }


def candidates_for_plan_v2(
    plan: CudaEventAutotunePlanV2,
    *,
    resolved_batch_candidates: tuple[int, ...] | None = None,
) -> tuple[CudaAutotuneCandidateV2, ...]:
    batches = (
        resolved_batch_candidates
        if resolved_batch_candidates is not None
        else plan.batch_candidates
    )
    if batches is None:
        raise CudaFullEvolutionAutotuneError(
            "dynamic batch candidates require a measured device identity and VRAM"
        )
    _strict_candidate_tuple(batches, name="resolved_batch_candidates")
    _checked_candidate_count_v2(batches, plan.temporal_k_candidates)
    return tuple(
        CudaAutotuneCandidateV2(batch_size=batch, temporal_k=temporal_k)
        for batch in batches
        for temporal_k in plan.temporal_k_candidates
    )


def candidate_stages_for_plan_v2(
    plan: CudaEventAutotunePlanV2,
    *,
    resolved_batch_candidates: tuple[int, ...],
) -> tuple[CudaAutotuneStageV2, ...]:
    """Split a complete exact envelope into deterministic <=64-arm rounds."""
    _strict_candidate_tuple(resolved_batch_candidates, name="resolved_batch_candidates")
    _checked_candidate_count_v2(resolved_batch_candidates, plan.temporal_k_candidates)
    stages: list[CudaAutotuneStageV2] = []
    for stage_index, start in enumerate(
        range(0, len(resolved_batch_candidates), MAX_BATCH_CANDIDATES_PER_PLAN_V2)
    ):
        batches = resolved_batch_candidates[
            start : start + MAX_BATCH_CANDIDATES_PER_PLAN_V2
        ]
        candidates = tuple(
            CudaAutotuneCandidateV2(batch_size=batch, temporal_k=temporal_k)
            for batch in batches
            for temporal_k in plan.temporal_k_candidates
        )
        if len(candidates) * len(EXECUTION_MODES_V2) > MAX_CANDIDATES_PER_PLAN_V2:
            raise CudaFullEvolutionAutotuneError(
                "staged autotune round exceeds the 64-arm contract"
            )
        stages.append(
            CudaAutotuneStageV2(
                stage_index=stage_index,
                batch_candidates=batches,
                candidates=candidates,
            )
        )
    if not stages:
        raise CudaFullEvolutionAutotuneError("staged autotune envelope is empty")
    return tuple(stages)


def _canonical_sha256_v2(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _canonical_vram_observation_v2(value: object, *, name: str) -> dict[str, int]:
    if type(value) is not dict or set(value) != VRAM_OBSERVATION_KEYS_V2:
        raise CudaFullEvolutionAutotuneError(
            f"{name} must contain exactly free_bytes, total_bytes, "
            "allocated_bytes, and reserved_bytes"
        )
    free_bytes = _strict_nonnegative(value.get("free_bytes"), f"{name} free_bytes")
    total_bytes = _strict_nonnegative(value.get("total_bytes"), f"{name} total_bytes")
    allocated_bytes = _strict_nonnegative(
        value.get("allocated_bytes"), f"{name} allocated_bytes"
    )
    reserved_bytes = _strict_nonnegative(
        value.get("reserved_bytes"), f"{name} reserved_bytes"
    )
    if max(free_bytes, allocated_bytes, reserved_bytes, total_bytes) > MAX_SIGNED_INT64:
        raise CudaFullEvolutionAutotuneError(f"{name} quantities must fit signed int64")
    if free_bytes > total_bytes:
        raise CudaFullEvolutionAutotuneError(
            f"{name} free_bytes cannot exceed total_bytes"
        )
    if allocated_bytes > reserved_bytes or reserved_bytes > total_bytes:
        raise CudaFullEvolutionAutotuneError(
            f"{name} must satisfy allocated_bytes <= reserved_bytes <= total_bytes"
        )
    return {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "allocated_bytes": allocated_bytes,
        "reserved_bytes": reserved_bytes,
    }


def _launch_plan_identity_v2(plan: CudaEventAutotunePlanV2) -> dict[str, Any]:
    body = {
        "schema_version": LAUNCH_PLAN_IDENTITY_SCHEMA_V2,
        "artifact_path": str(plan.artifact_path.resolve()),
        "batch_candidates": (
            None if plan.batch_candidates is None else list(plan.batch_candidates)
        ),
        "temporal_k_candidates": list(plan.temporal_k_candidates),
        "warmup_dispatches": plan.warmup_dispatches,
        "samples_per_candidate": plan.samples_per_candidate,
        "dispatches_per_candidate": plan.dispatches_per_candidate,
        "seed": plan.seed,
        "device": plan.device,
        "reserve_bytes": plan.reserve_bytes,
    }
    return {**body, "plan_sha256": _canonical_sha256_v2(body)}


def _launch_timestamp_utc_v2() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _validated_launch_timestamp_v2(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise CudaFullEvolutionAutotuneError(
            "launch origin timestamp must be canonical UTC"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CudaFullEvolutionAutotuneError(
            "launch origin timestamp must be canonical UTC"
        ) from exc
    canonical = parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if canonical != value:
        raise CudaFullEvolutionAutotuneError(
            "launch origin timestamp must use the exact canonical UTC encoding"
        )
    return value


def _write_all_v2(file_descriptor: int, payload: bytes) -> None:
    position = 0
    while position < len(payload):
        written = os.write(file_descriptor, payload[position:])
        if written <= 0:
            raise OSError("launch origin write made no progress")
        position += written


def _read_exact_descriptor_v2(file_descriptor: int, size: int) -> bytes:
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(file_descriptor, min(remaining, 64 << 10))
        if not chunk:
            raise CudaFullEvolutionAutotuneError(
                "result publication inode was truncated while it was verified"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(file_descriptor, 1):
        raise CudaFullEvolutionAutotuneError(
            "result publication inode grew while it was verified"
        )
    return b"".join(chunks)


def _unlink_exact_inode_v2(path: Path, identity: tuple[int, int]) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if (metadata.st_dev, metadata.st_ino) != identity:
        return False
    path.unlink()
    return True


def _fsync_directory_v2(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    file_descriptor = os.open(path, flags)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _publish_launch_origin_v2(
    path: Path,
    *,
    artifact_path: Path,
    artifact_sha256: str,
    manifest_sha256: str,
    device_identity: dict[str, Any],
    initial_vram: dict[str, int],
    plan: CudaEventAutotunePlanV2,
    harness_sha256: str,
    performance_restructure_contract_sha256: str,
    validation_matrix_sha256: str,
) -> dict[str, str]:
    if not isinstance(path, Path):
        raise CudaFullEvolutionAutotuneError("launch origin path must be a Path")
    launch_path = path
    if (
        not launch_path.is_absolute()
        or launch_path.resolve() != launch_path
        or launch_path.name in {"", ".", ".."}
    ):
        raise CudaFullEvolutionAutotuneError(
            "launch origin path must be canonical, absolute, and symlink-free"
        )
    source_hashes = (
        artifact_sha256,
        manifest_sha256,
        harness_sha256,
        performance_restructure_contract_sha256,
        validation_matrix_sha256,
    )
    if any(not _is_sha256_v2(value) for value in source_hashes):
        raise CudaFullEvolutionAutotuneError(
            "launch origin source hashes must be SHA-256 values"
        )
    canonical_vram = _canonical_vram_observation_v2(
        initial_vram, name="launch VRAM observation"
    )
    pid = _strict_positive(os.getpid(), "launch origin PID")
    body = {
        "schema_version": LAUNCH_ORIGIN_SCHEMA_V2,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "device_identity": device_identity,
        "initial_vram": canonical_vram,
        "plan_identity": _launch_plan_identity_v2(plan),
        "harness_sha256": harness_sha256,
        "performance_restructure_contract_sha256": (
            performance_restructure_contract_sha256
        ),
        "validation_matrix_sha256": validation_matrix_sha256,
        "pid": pid,
        "launch_timestamp_utc": _launch_timestamp_utc_v2(),
    }
    body_sha256 = _canonical_sha256_v2(body)
    payload = {**body, "body_sha256": body_sha256}
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("ascii")
    if len(encoded) > MAX_LAUNCH_ORIGIN_BYTES_V2:
        raise CudaFullEvolutionAutotuneError("launch origin receipt is too large")

    launch_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    file_descriptor: int | None = None
    created = False
    created_identity: tuple[int, int] | None = None
    try:
        # The creating descriptor remains writable even though the directory
        # entry is born 0400.  A concurrent same-UID opener can therefore
        # never retain a second writable descriptor across the seal boundary.
        file_descriptor = os.open(launch_path, flags, stat.S_IRUSR)
        created = True
        _write_all_v2(file_descriptor, encoded)
        os.fchmod(file_descriptor, stat.S_IRUSR)
        os.fsync(file_descriptor)
        sealed_stat = os.fstat(file_descriptor)
        created_identity = (sealed_stat.st_dev, sealed_stat.st_ino)
        if (
            not stat.S_ISREG(sealed_stat.st_mode)
            or sealed_stat.st_nlink != 1
            or stat.S_IMODE(sealed_stat.st_mode) & 0o222
            or sealed_stat.st_size != len(encoded)
        ):
            raise CudaFullEvolutionAutotuneError(
                "launch origin did not seal as a single-link read-only file"
            )
        _fsync_directory_v2(launch_path.parent)
        try:
            published_stat = os.lstat(launch_path)
        except OSError as exc:
            raise CudaFullEvolutionAutotuneError(
                f"launch origin disappeared during publication: {exc}"
            ) from exc
        if (
            launch_path.resolve() != launch_path
            or not stat.S_ISREG(published_stat.st_mode)
            or (published_stat.st_dev, published_stat.st_ino) != created_identity
            or published_stat.st_size != sealed_stat.st_size
            or published_stat.st_nlink != sealed_stat.st_nlink
            or stat.S_IMODE(published_stat.st_mode) != stat.S_IMODE(sealed_stat.st_mode)
        ):
            raise CudaFullEvolutionAutotuneError(
                "launch origin path changed during publication"
            )
        os.close(file_descriptor)
        file_descriptor = None
    except FileExistsError as exc:
        raise CudaFullEvolutionAutotuneError(
            f"launch origin must not overwrite evidence: {launch_path}"
        ) from exc
    except Exception:
        if created and created_identity is not None:
            try:
                current = os.lstat(launch_path)
                if (current.st_dev, current.st_ino) == created_identity:
                    launch_path.unlink()
                    _fsync_directory_v2(launch_path.parent)
            except OSError:
                pass
        raise
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)

    file_sha256 = hashlib.sha256(encoded).hexdigest()
    return {
        "path": str(launch_path),
        "file_sha256": file_sha256,
        "body_sha256": body_sha256,
    }


def _read_sealed_launch_origin_v2(path: Path) -> bytes:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise CudaFullEvolutionAutotuneError(
            f"cannot reopen launch origin sidecar: {exc}"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o222
        or before.st_size <= 0
        or before.st_size > MAX_LAUNCH_ORIGIN_BYTES_V2
    ):
        raise CudaFullEvolutionAutotuneError(
            "launch origin must be a bounded single-link read-only regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(path, flags)
    except OSError as exc:
        raise CudaFullEvolutionAutotuneError(
            f"cannot safely open launch origin sidecar: {exc}"
        ) from exc
    try:
        opened = os.fstat(file_descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise CudaFullEvolutionAutotuneError(
                "launch origin changed while it was opened"
            )
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(file_descriptor, min(remaining, 64 << 10))
            if not chunk:
                raise CudaFullEvolutionAutotuneError(
                    "launch origin was truncated while it was read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(file_descriptor, 1):
            raise CudaFullEvolutionAutotuneError("launch origin grew while it was read")
    finally:
        os.close(file_descriptor)
    try:
        after = os.lstat(path)
    except OSError as exc:
        raise CudaFullEvolutionAutotuneError(
            f"launch origin disappeared after it was read: {exc}"
        ) from exc
    if (
        (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        or after.st_nlink != 1
        or stat.S_IMODE(after.st_mode) & 0o222
        or after.st_size != before.st_size
    ):
        raise CudaFullEvolutionAutotuneError(
            "launch origin changed while it was validated"
        )
    return b"".join(chunks)


def _validated_launch_origin_v2(
    result: CudaEventAutotuneResultV2,
    *,
    artifact_path: Path,
    artifact_sha256: str,
    manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    reference = result.launch_origin
    if type(reference) is not dict or set(reference) != {
        "path",
        "file_sha256",
        "body_sha256",
    }:
        raise CudaFullEvolutionAutotuneError(
            "authoritative validation requires an exact launch origin reference"
        )
    if not all(isinstance(value, str) for value in reference.values()) or not all(
        _is_sha256_v2(reference[key]) for key in ("file_sha256", "body_sha256")
    ):
        raise CudaFullEvolutionAutotuneError(
            "launch origin reference hashes are invalid"
        )
    launch_path = Path(reference["path"])
    if not launch_path.is_absolute() or launch_path.resolve() != launch_path:
        raise CudaFullEvolutionAutotuneError(
            "launch origin reference path must be canonical and absolute"
        )
    encoded = _read_sealed_launch_origin_v2(launch_path)
    if hashlib.sha256(encoded).hexdigest() != reference["file_sha256"]:
        raise CudaFullEvolutionAutotuneError("launch origin file hash changed")
    try:
        payload = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CudaFullEvolutionAutotuneError(
            "launch origin is not canonical JSON"
        ) from exc
    expected_keys = {
        "schema_version",
        "artifact_path",
        "artifact_sha256",
        "manifest_sha256",
        "device_identity",
        "initial_vram",
        "plan_identity",
        "harness_sha256",
        "performance_restructure_contract_sha256",
        "validation_matrix_sha256",
        "pid",
        "launch_timestamp_utc",
        "body_sha256",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise CudaFullEvolutionAutotuneError(
            "launch origin has an invalid exact DTO shape"
        )
    canonical_encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode(
        "ascii"
    )
    if encoded != canonical_encoded:
        raise CudaFullEvolutionAutotuneError(
            "launch origin is not in its exact canonical encoding"
        )
    body = {key: value for key, value in payload.items() if key != "body_sha256"}
    canonical_body_sha256 = _canonical_sha256_v2(body)
    if (
        payload.get("body_sha256") != canonical_body_sha256
        or reference["body_sha256"] != canonical_body_sha256
    ):
        raise CudaFullEvolutionAutotuneError("launch origin body seal changed")
    initial_vram = _canonical_vram_observation_v2(
        payload.get("initial_vram"), name="launch origin VRAM observation"
    )
    result_vram = _canonical_vram_observation_v2(
        result.initial_vram, name="result launch VRAM observation"
    )
    _validated_launch_timestamp_v2(payload.get("launch_timestamp_utc"))
    if (
        payload.get("schema_version") != LAUNCH_ORIGIN_SCHEMA_V2
        or payload.get("artifact_path") != str(artifact_path.resolve())
        or payload.get("artifact_sha256") != artifact_sha256
        or payload.get("manifest_sha256") != manifest_sha256
        or payload.get("device_identity") != result.device_identity
        or initial_vram != result_vram
        or payload.get("plan_identity") != _launch_plan_identity_v2(result.plan)
        or payload.get("harness_sha256") != result.harness_sha256
        or payload.get("performance_restructure_contract_sha256")
        != result.performance_restructure_contract_sha256
        or payload.get("validation_matrix_sha256") != result.validation_matrix_sha256
        or payload.get("pid") != os.getpid()
    ):
        raise CudaFullEvolutionAutotuneError(
            "launch origin does not match the exact result and live source"
        )
    return payload, dict(reference)


def staged_envelope_receipt_v2(
    *,
    artifact_sha256: str,
    manifest_sha256: str,
    device_identity: dict[str, Any],
    temporal_k_candidates: tuple[int, ...],
    stages: tuple[CudaAutotuneStageV2, ...],
    measurements: Sequence[CudaAutotuneMeasurementV2] | None = None,
    resolved_envelope: dict[str, Any] | None = None,
    initial_vram: dict[str, int] | None = None,
    reserve_bytes: int | None = None,
    dispatches_per_candidate: int | None = None,
    samples_per_candidate: int | None = None,
) -> dict[str, Any]:
    """Seal every declared/measured stage to the source artifact and device."""
    if not _is_sha256_v2(artifact_sha256) or not _is_sha256_v2(manifest_sha256):
        raise CudaFullEvolutionAutotuneError(
            "staged envelope source artifact or manifest hash is invalid"
        )
    if (
        not isinstance(device_identity, dict)
        or not isinstance(device_identity.get("uuid"), str)
        or not device_identity["uuid"].strip()
        or not isinstance(device_identity.get("sm"), str)
        or not device_identity["sm"].strip()
    ):
        raise CudaFullEvolutionAutotuneError(
            "staged envelope device identity is not source-bound"
        )
    _strict_candidate_tuple(
        temporal_k_candidates, name="temporal_k_candidates", maximum=64
    )
    if measurements is not None:
        _strict_positive(
            dispatches_per_candidate, "staged envelope dispatches_per_candidate"
        )
        _strict_positive(samples_per_candidate, "staged envelope samples_per_candidate")
    elif dispatches_per_candidate is not None or samples_per_candidate is not None:
        _strict_positive(
            dispatches_per_candidate, "staged envelope dispatches_per_candidate"
        )
        _strict_positive(samples_per_candidate, "staged envelope samples_per_candidate")
    resolved_batches = _validated_resolved_envelope_v2(
        resolved_envelope,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        device_identity=device_identity,
        initial_vram=initial_vram,
        reserve_bytes=reserve_bytes,
        temporal_k_candidates=temporal_k_candidates,
    )
    measurements_by_candidate_mode: dict[
        tuple[CudaAutotuneCandidateV2, str], CudaAutotuneMeasurementV2
    ] = {}
    if measurements is not None:
        for measurement in measurements:
            key = (measurement.candidate, measurement.execution_mode)
            if key in measurements_by_candidate_mode:
                raise CudaFullEvolutionAutotuneError(
                    "staged envelope measurements contain a duplicate B/K/mode candidate"
                )
            if measurement.execution_mode not in EXECUTION_MODES_V2:
                raise CudaFullEvolutionAutotuneError(
                    "staged envelope measurement execution mode is invalid"
                )
            measurements_by_candidate_mode[key] = measurement
    seen: set[CudaAutotuneCandidateV2] = set()
    stage_records: list[dict[str, Any]] = []
    all_batches: list[int] = []
    for expected_index, stage in enumerate(stages):
        if stage.stage_index != expected_index:
            raise CudaFullEvolutionAutotuneError(
                "staged envelope indices are not contiguous"
            )
        _strict_candidate_tuple(
            stage.batch_candidates,
            name=f"staged envelope B candidates for stage {stage.stage_index}",
        )
        all_batches.extend(stage.batch_candidates)
        expected_candidates = tuple(
            CudaAutotuneCandidateV2(batch_size=batch, temporal_k=temporal_k)
            for batch in stage.batch_candidates
            for temporal_k in temporal_k_candidates
        )
        if (
            stage.candidates != expected_candidates
            or len(stage.candidates) * len(EXECUTION_MODES_V2)
            > MAX_CANDIDATES_PER_PLAN_V2
        ):
            raise CudaFullEvolutionAutotuneError(
                "staged envelope does not exactly cover its declared B/K round"
            )
        if len(set(stage.candidates)) != len(stage.candidates) or any(
            candidate in seen for candidate in stage.candidates
        ):
            raise CudaFullEvolutionAutotuneError(
                "staged envelope contains a duplicate B/K candidate"
            )
        seen.update(stage.candidates)
        measured_records: list[dict[str, Any]] = []
        if measurements is not None:
            for candidate in stage.candidates:
                for execution_mode in EXECUTION_MODES_V2:
                    measurement = measurements_by_candidate_mode.get(
                        (candidate, execution_mode)
                    )
                    if (
                        measurement is None
                        or measurement.stage_index != stage.stage_index
                    ):
                        raise CudaFullEvolutionAutotuneError(
                            "staged envelope candidate lacks its exact measured B/K/mode stage record"
                        )
                    measured_records.append(
                        {
                            "candidate": asdict(measurement.candidate),
                            "execution_mode": measurement.execution_mode,
                            "status": measurement.status,
                            "stage_index": measurement.stage_index,
                            "committed_world_ticks": measurement.committed_world_ticks,
                            "p10_sample_world_ticks_per_second": (
                                measurement.p10_sample_world_ticks_per_second
                            ),
                            "peak_allocated_bytes": measurement.peak_allocated_bytes,
                            "peak_reserved_bytes": measurement.peak_reserved_bytes,
                            "peak_device_process_bytes": (
                                measurement.peak_device_process_bytes
                            ),
                            "fixture_id": (
                                None
                                if measurement.fixture is None
                                else measurement.fixture.fixture_id
                            ),
                        }
                    )
        stage_body = {
            "stage_index": stage.stage_index,
            "batch_candidates": list(stage.batch_candidates),
            "candidates": [asdict(candidate) for candidate in stage.candidates],
            "candidate_count": len(stage.candidates),
            "measurements": measured_records,
            "measurement_count": len(measured_records),
        }
        stage_records.append(
            {
                **stage_body,
                "stage_sha256": _canonical_sha256_v2(
                    {
                        "artifact_sha256": artifact_sha256,
                        "manifest_sha256": manifest_sha256,
                        "device_identity": device_identity,
                        **stage_body,
                    }
                ),
            }
        )
    if not stage_records:
        raise CudaFullEvolutionAutotuneError("staged envelope receipt is empty")
    _strict_candidate_tuple(
        tuple(all_batches), name="staged envelope aggregate B candidates"
    )
    if tuple(all_batches) != resolved_batches:
        raise CudaFullEvolutionAutotuneError(
            "staged envelope does not exactly cover the source-bound resolved envelope"
        )
    expected_measurement_keys = {
        (candidate, execution_mode)
        for candidate in seen
        for execution_mode in EXECUTION_MODES_V2
    }
    if (
        measurements is not None
        and set(measurements_by_candidate_mode) != expected_measurement_keys
    ):
        raise CudaFullEvolutionAutotuneError(
            "staged envelope measurements do not exactly cover declared B/K/mode candidates"
        )
    body = {
        "schema_version": STAGED_ENVELOPE_SCHEMA_V2,
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "device_identity": device_identity,
        "temporal_k_candidates": list(temporal_k_candidates),
        "execution_modes": list(EXECUTION_MODES_V2),
        "resolved_envelope": resolved_envelope,
        "measurement_contract": {
            "dispatches_per_candidate": dispatches_per_candidate,
            "samples_per_candidate": samples_per_candidate,
        },
        "execution_status": (
            "MEASURED_COMPLETE" if measurements is not None else "DECLARED_ONLY"
        ),
        "stages": stage_records,
    }
    return {**body, "aggregate_sha256": _canonical_sha256_v2(body)}


def _require_torch_v2() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment boundary
        raise CudaFullEvolutionAutotuneError(f"PyTorch is unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise CudaFullEvolutionAutotuneError("a physical CUDA device is required")
    return torch


def _live_vram_v2(torch: Any, device: Any) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return _canonical_vram_observation_v2(
        {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        },
        name="live VRAM observation",
    )


def _effective_reserve_bytes_v2(plan: CudaEventAutotunePlanV2, total_bytes: int) -> int:
    if plan.reserve_bytes is not None:
        return plan.reserve_bytes
    return max(DEFAULT_RESERVE_FLOOR_BYTES_V2, total_bytes // 10)


def capacity_for_candidate_v2(
    candidate: CudaAutotuneCandidateV2,
    *,
    free_vram_bytes: int,
    total_vram_bytes: int,
    reserve_bytes: int,
    dispatches_per_candidate: int,
) -> CudaAutotuneCapacityV2:
    _strict_positive(dispatches_per_candidate, "dispatches_per_candidate")
    contract = CudaFullEvolutionNativeContractV2(
        B=candidate.batch_size,
        total_ticks=dispatches_per_candidate * candidate.temporal_k,
        temporal_k=candidate.temporal_k,
    )
    ledger = estimate_cuda_native_resources_v2(
        contract, include_reference_scratch=False
    )
    admission = select_cuda_native_batch_v2(
        contract,
        available_vram_bytes=free_vram_bytes,
        reserve_bytes=reserve_bytes,
        include_reference_scratch=False,
    )
    return CudaAutotuneCapacityV2(
        candidate=candidate,
        free_vram_bytes=free_vram_bytes,
        total_vram_bytes=total_vram_bytes,
        reserve_bytes=reserve_bytes,
        ledger=ledger,
        admission=admission,
    )


def _horizon_dependent_resident_bytes_v2(
    candidate: CudaAutotuneCandidateV2,
) -> int:
    """Prove from the resource ledger that absolute horizon is non-resident."""
    one_dispatch = CudaFullEvolutionNativeContractV2(
        B=candidate.batch_size,
        total_ticks=candidate.temporal_k,
        temporal_k=candidate.temporal_k,
    )
    two_dispatches = CudaFullEvolutionNativeContractV2(
        B=candidate.batch_size,
        total_ticks=2 * candidate.temporal_k,
        temporal_k=candidate.temporal_k,
    )
    first = estimate_cuda_native_resources_v2(
        one_dispatch, include_reference_scratch=False
    )
    second = estimate_cuda_native_resources_v2(
        two_dispatches, include_reference_scratch=False
    )
    delta = second.resident_state_bytes - first.resident_state_bytes
    if delta != 0:
        raise CudaFullEvolutionAutotuneError(
            "resident allocation unexpectedly depends on absolute tick horizon"
        )
    return delta


def maximum_vram_admitted_batch_size_v2(
    *,
    free_vram_bytes: int,
    reserve_bytes: int,
    temporal_k: int,
) -> int:
    """Largest exact B admitted by the current ledger at one temporal_k."""
    probe = CudaFullEvolutionNativeContractV2(
        B=1, total_ticks=temporal_k, temporal_k=temporal_k
    )
    ledger = estimate_cuda_native_resources_v2(probe, include_reference_scratch=False)
    per_world = ledger.total_bytes - ledger.owner_allowance_bytes
    if per_world <= 0:
        raise CudaFullEvolutionAutotuneError(
            "CUDA Native V2 per-world ledger is invalid"
        )
    maximum_working = _maximum_working_bytes_v2(free_vram_bytes, reserve_bytes)
    if maximum_working < ledger.owner_allowance_bytes:
        return 0
    # CUDA-native world mapping is CTA-aligned.  Never emit a fractional warp
    # tail as the advertised VRAM maximum; B136 remains a separate required
    # comparison even when this floor is smaller.
    raw_maximum = (maximum_working - ledger.owner_allowance_bytes) // per_world
    return (raw_maximum // 8) * 8


def _round_up_to_eight_v2(value: int) -> int:
    _strict_positive(value, "SM occupancy candidate")
    return ((value + 7) // 8) * 8


def dynamic_batch_candidates_v2(
    *,
    sm_count: int,
    free_vram_bytes: int,
    reserve_bytes: int,
    temporal_k: int,
) -> tuple[int, ...]:
    """Deterministically derive anchors, q=1..4 SM candidates, and VRAM max."""
    _strict_positive(sm_count, "sm_count")
    maximum = maximum_vram_admitted_batch_size_v2(
        free_vram_bytes=free_vram_bytes,
        reserve_bytes=reserve_bytes,
        temporal_k=temporal_k,
    )
    if maximum <= 0:
        raise CudaFullEvolutionAutotuneError(
            "no CTA-aligned batch is admitted by measured VRAM"
        )
    # Construct the complete exact set.  Static anchors and every admitted
    # q=1..4 SM point are mandatory construction members; the caller converts
    # an oversized cross-product into deterministic <=64-arm stages.
    static_admitted = tuple(
        candidate for candidate in STATIC_BATCH_ANCHORS_V2 if candidate <= maximum
    )
    sm_scaled = tuple(
        candidate
        for candidate in (_round_up_to_eight_v2(q * sm_count) for q in range(1, 5))
        if candidate <= maximum
    )
    resolved = tuple(sorted({*static_admitted, *sm_scaled, maximum}))
    _strict_candidate_tuple(resolved, name="dynamic batch candidates")
    return resolved


def dynamic_batch_candidates_for_k_grid_v2(
    *,
    sm_count: int,
    free_vram_bytes: int,
    reserve_bytes: int,
    temporal_k_candidates: tuple[int, ...],
) -> tuple[tuple[int, ...], dict[int, int]]:
    """Construct the union of every per-K exact candidate requirement."""
    _strict_positive(sm_count, "sm_count")
    _strict_candidate_tuple(
        temporal_k_candidates, name="temporal_k_candidates", maximum=64
    )
    maxima = {
        temporal_k: maximum_vram_admitted_batch_size_v2(
            free_vram_bytes=free_vram_bytes,
            reserve_bytes=reserve_bytes,
            temporal_k=temporal_k,
        )
        for temporal_k in temporal_k_candidates
    }
    admitted_maximum = max(maxima.values(), default=0)
    if admitted_maximum <= 0:
        raise CudaFullEvolutionAutotuneError(
            "no CTA-aligned batch is admitted by measured VRAM"
        )
    static_admitted = tuple(
        candidate
        for candidate in STATIC_BATCH_ANCHORS_V2
        if candidate <= admitted_maximum
    )
    sm_scaled = tuple(
        candidate
        for candidate in (_round_up_to_eight_v2(q * sm_count) for q in range(1, 5))
        if candidate <= admitted_maximum
    )
    resolved = tuple(
        sorted(
            {
                *static_admitted,
                *sm_scaled,
                *(value for value in maxima.values() if value > 0),
            }
        )
    )
    _strict_candidate_tuple(resolved, name="dynamic batch candidates")
    _checked_candidate_count_v2(resolved, temporal_k_candidates)
    return resolved, maxima


def _device_envelope_binding_v2(
    device_identity: object, *, require_sm_count: bool
) -> dict[str, Any]:
    if not isinstance(device_identity, dict):
        raise CudaFullEvolutionAutotuneError("device identity must be a mapping")
    uuid = device_identity.get("uuid")
    compute_capability = device_identity.get("sm")
    if (
        not isinstance(uuid, str)
        or not uuid.strip()
        or not isinstance(compute_capability, str)
        or not compute_capability.strip()
    ):
        raise CudaFullEvolutionAutotuneError(
            "device UUID and compute capability are required for envelope binding"
        )
    binding: dict[str, Any] = {
        "device_uuid": uuid,
        "compute_capability": compute_capability,
    }
    if require_sm_count:
        sm_count = device_identity.get("multiprocessor_count")
        binding["multiprocessor_count"] = _strict_positive(
            sm_count, "device multiprocessor_count"
        )
    return binding


def _initial_vram_binding_v2(initial_vram: object) -> dict[str, int]:
    return _canonical_vram_observation_v2(initial_vram, name="initial VRAM binding")


def _canonical_required_batch_envelope_v2(
    *,
    device_identity: dict[str, Any],
    initial_vram: dict[str, int],
    reserve_bytes: int,
    temporal_k_candidates: tuple[int, ...],
) -> tuple[tuple[int, ...], dict[int, int]]:
    if temporal_k_candidates != DEFAULT_TEMPORAL_K_CANDIDATES_V2:
        raise CudaFullEvolutionAutotuneError(
            "resolved envelope requires the exact required K grid"
        )
    binding = _device_envelope_binding_v2(device_identity, require_sm_count=True)
    return dynamic_batch_candidates_for_k_grid_v2(
        sm_count=binding["multiprocessor_count"],
        free_vram_bytes=initial_vram["free_bytes"],
        reserve_bytes=reserve_bytes,
        temporal_k_candidates=temporal_k_candidates,
    )


def resolved_envelope_v2(
    *,
    artifact_sha256: str,
    manifest_sha256: str,
    device_identity: dict[str, Any],
    initial_vram: dict[str, int],
    reserve_bytes: int,
    temporal_k_candidates: tuple[int, ...],
    resolved_batch_candidates: tuple[int, ...],
    plan_mode: str,
) -> dict[str, Any]:
    """Seal the complete live-derived B envelope before staged measurement."""
    if not _is_sha256_v2(artifact_sha256) or not _is_sha256_v2(manifest_sha256):
        raise CudaFullEvolutionAutotuneError(
            "resolved envelope source artifact or manifest hash is invalid"
        )
    _strict_candidate_tuple(
        temporal_k_candidates, name="temporal_k_candidates", maximum=64
    )
    _strict_candidate_tuple(resolved_batch_candidates, name="resolved_batch_candidates")
    _checked_candidate_count_v2(resolved_batch_candidates, temporal_k_candidates)
    vram_binding = _initial_vram_binding_v2(initial_vram)
    reserve_bytes = _strict_nonnegative(reserve_bytes, "reserve_bytes")
    device_binding = _device_envelope_binding_v2(device_identity, require_sm_count=True)
    required_batches, per_k_maxima = _canonical_required_batch_envelope_v2(
        device_identity=device_identity,
        initial_vram=vram_binding,
        reserve_bytes=reserve_bytes,
        temporal_k_candidates=temporal_k_candidates,
    )
    if plan_mode not in {
        "DYNAMIC_FULL_ENVELOPE",
        "EXPLICIT_REQUIRED_SUPERSET",
    }:
        raise CudaFullEvolutionAutotuneError("resolved envelope plan mode is invalid")
    if (
        plan_mode == "DYNAMIC_FULL_ENVELOPE"
        and resolved_batch_candidates != required_batches
    ):
        raise CudaFullEvolutionAutotuneError(
            "dynamic resolved envelope differs from its live-derived required candidates"
        )
    if not set(required_batches).issubset(resolved_batch_candidates):
        missing = tuple(
            candidate
            for candidate in required_batches
            if candidate not in resolved_batch_candidates
        )
        raise CudaFullEvolutionAutotuneError(
            "resolved envelope omits required live-derived B candidates: "
            + ",".join(str(candidate) for candidate in missing)
        )
    body = {
        "schema_version": RESOLVED_ENVELOPE_SCHEMA_V2,
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "device_binding": device_binding,
        "initial_vram": vram_binding,
        "reserve_bytes": reserve_bytes,
        "temporal_k_candidates": list(temporal_k_candidates),
        "plan_mode": plan_mode,
        "required_batch_candidates": list(required_batches),
        "resolved_batch_candidates": list(resolved_batch_candidates),
        "per_k_maximum_vram_admitted_batch_size": [
            {
                "temporal_k": temporal_k,
                "maximum_vram_admitted_batch_size": per_k_maxima[temporal_k],
            }
            for temporal_k in temporal_k_candidates
        ],
    }
    return {**body, "envelope_sha256": _canonical_sha256_v2(body)}


def _validated_resolved_envelope_v2(
    envelope: object,
    *,
    artifact_sha256: str,
    manifest_sha256: str,
    device_identity: dict[str, Any],
    initial_vram: dict[str, int] | None,
    reserve_bytes: int | None,
    temporal_k_candidates: tuple[int, ...],
) -> tuple[int, ...]:
    if not isinstance(envelope, dict):
        raise CudaFullEvolutionAutotuneError(
            "staged envelope requires a source-bound resolved envelope"
        )
    if initial_vram is None or reserve_bytes is None:
        raise CudaFullEvolutionAutotuneError(
            "staged envelope requires the initial live-VRAM source binding"
        )
    raw_resolved = envelope.get("resolved_batch_candidates")
    if not isinstance(raw_resolved, list):
        raise CudaFullEvolutionAutotuneError(
            "resolved envelope batch candidates are missing"
        )
    expected = resolved_envelope_v2(
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        device_identity=device_identity,
        initial_vram=initial_vram,
        reserve_bytes=reserve_bytes,
        temporal_k_candidates=temporal_k_candidates,
        resolved_batch_candidates=tuple(raw_resolved),
        plan_mode=envelope.get("plan_mode"),
    )
    if envelope != expected:
        raise CudaFullEvolutionAutotuneError(
            "resolved envelope is not an exact source/device-bound live envelope"
        )
    return tuple(raw_resolved)


def _validate_explicit_batch_candidates_v2(
    batch_candidates: tuple[int, ...],
    *,
    maximum_vram_admitted_batch_size: int | None = None,
    per_k_maximum_vram_admitted_batch_size: dict[int, int] | None = None,
    required_batch_candidates: tuple[int, ...] | None = None,
) -> None:
    """Require the complete live-derived envelope for explicit benchmark plans."""
    _strict_candidate_tuple(batch_candidates, name="batch_candidates")
    if required_batch_candidates is not None:
        _strict_candidate_tuple(
            required_batch_candidates, name="required_batch_candidates"
        )
        missing = tuple(
            candidate
            for candidate in required_batch_candidates
            if candidate not in batch_candidates
        )
        if missing:
            raise CudaFullEvolutionAutotuneError(
                "explicit batch candidates omit required live-derived anchors, "
                "SM occupancy candidates, or per-K maxima: "
                + ",".join(str(candidate) for candidate in missing)
            )
    maxima = (
        per_k_maximum_vram_admitted_batch_size.values()
        if per_k_maximum_vram_admitted_batch_size is not None
        else (maximum_vram_admitted_batch_size,)
    )
    admitted = False
    for maximum in maxima:
        _strict_nonnegative(maximum, "maximum_vram_admitted_batch_size")
        admitted = admitted or maximum >= 136
    if admitted and 136 not in batch_candidates:
        raise CudaFullEvolutionAutotuneError(
            "explicit batch candidates omit mandatory B136 despite measured VRAM admission"
        )


def _b136_source_binding_v2(
    *,
    device_identity: dict[str, Any],
    initial_vram: dict[str, int],
    reserve_bytes: int,
    dispatches_per_candidate: int,
) -> dict[str, Any]:
    device_binding = _device_envelope_binding_v2(
        device_identity, require_sm_count=False
    )
    vram_binding = _initial_vram_binding_v2(initial_vram)
    return {
        "device_uuid": device_binding["device_uuid"],
        "compute_capability": device_binding["compute_capability"],
        "initial_free_vram_bytes": vram_binding["free_bytes"],
        "initial_total_vram_bytes": vram_binding["total_bytes"],
        "initial_allocated_vram_bytes": vram_binding["allocated_bytes"],
        "initial_reserved_vram_bytes": vram_binding["reserved_bytes"],
        "reserve_bytes": _strict_nonnegative(reserve_bytes, "reserve_bytes"),
        "dispatches_per_candidate": _strict_positive(
            dispatches_per_candidate, "dispatches_per_candidate"
        ),
    }


def _b136_capacity_record_v2(
    *,
    capacity: CudaAutotuneCapacityV2,
    source_binding: dict[str, Any],
    temporal_k: int,
) -> dict[str, Any]:
    body = {
        "schema_version": B136_CAPACITY_SOURCE_SCHEMA_V2,
        "source_binding": {**source_binding, "temporal_k": temporal_k},
        "candidate": asdict(
            CudaAutotuneCandidateV2(batch_size=136, temporal_k=temporal_k)
        ),
        "admitted": capacity.admitted,
        "ledger": asdict(capacity.ledger),
        "admission": asdict(capacity.admission),
    }
    return {**body, "capacity_sha256": _canonical_sha256_v2(body)}


def _b136_comparison_v2(
    *,
    per_k_maximum_vram_admitted_batch_size: dict[int, int],
    resolved_batch_candidates: tuple[int, ...],
    measurements: Sequence[CudaAutotuneMeasurementV2],
    capacities: dict[int, CudaAutotuneCapacityV2],
    device_identity: dict[str, Any],
    initial_vram: dict[str, int],
    reserve_bytes: int,
    dispatches_per_candidate: int,
) -> dict[str, Any]:
    """Record exact B136 per-K capacity against one live source binding."""
    source_binding = _b136_source_binding_v2(
        device_identity=device_identity,
        initial_vram=initial_vram,
        reserve_bytes=reserve_bytes,
        dispatches_per_candidate=dispatches_per_candidate,
    )
    _strict_candidate_tuple(
        tuple(sorted(per_k_maximum_vram_admitted_batch_size)),
        name="B136 temporal_k_candidates",
        maximum=64,
    )
    required = any(
        maximum >= 136 for maximum in per_k_maximum_vram_admitted_batch_size.values()
    )
    present = 136 in resolved_batch_candidates
    if required and not present:
        raise CudaFullEvolutionAutotuneError(
            "mandatory B136 comparison is omitted despite measured VRAM admission"
        )
    measurements_by_k = {
        temporal_k: [
            value
            for value in measurements
            if value.candidate
            == CudaAutotuneCandidateV2(batch_size=136, temporal_k=temporal_k)
        ]
        for temporal_k in per_k_maximum_vram_admitted_batch_size
    }
    per_k: list[dict[str, Any]] = []
    for temporal_k in sorted(per_k_maximum_vram_admitted_batch_size):
        capacity = capacities.get(temporal_k)
        if capacity is None:
            raise CudaFullEvolutionAutotuneError(
                "B136 per-K capacity ledger is missing"
            )
        expected_maximum = maximum_vram_admitted_batch_size_v2(
            free_vram_bytes=source_binding["initial_free_vram_bytes"],
            reserve_bytes=source_binding["reserve_bytes"],
            temporal_k=temporal_k,
        )
        if per_k_maximum_vram_admitted_batch_size[temporal_k] != expected_maximum:
            raise CudaFullEvolutionAutotuneError(
                "B136 per-K maximum is not the exact initial live-VRAM admission"
            )
        expected_capacity = capacity_for_candidate_v2(
            CudaAutotuneCandidateV2(batch_size=136, temporal_k=temporal_k),
            free_vram_bytes=source_binding["initial_free_vram_bytes"],
            total_vram_bytes=source_binding["initial_total_vram_bytes"],
            reserve_bytes=source_binding["reserve_bytes"],
            dispatches_per_candidate=source_binding["dispatches_per_candidate"],
        )
        if capacity != expected_capacity:
            raise CudaFullEvolutionAutotuneError(
                "B136 per-K capacity ledger is not canonical for the live source binding"
            )
        mode_measurements = measurements_by_k[temporal_k]
        measurement: CudaAutotuneMeasurementV2 | None = None
        if capacity.admitted:
            if {value.execution_mode for value in mode_measurements} != set(
                EXECUTION_MODES_V2
            ):
                raise CudaFullEvolutionAutotuneError(
                    f"mandatory B136 K={temporal_k} mode measurements are incomplete"
                )
            measurement = select_best_measurement_v2(mode_measurements)
        capacity_record = _b136_capacity_record_v2(
            capacity=capacity,
            source_binding=source_binding,
            temporal_k=temporal_k,
        )
        if capacity.admitted:
            if measurement is None:
                raise CudaFullEvolutionAutotuneError(
                    f"mandatory B136 K={temporal_k} measurement is omitted"
                )
            status = f"MEASURED_{measurement.status}"
            reason = "exact B136 candidate was admitted and measured"
        else:
            status = "CAPACITY_SKIPPED"
            reason = "B136 is not admitted by the exact per-K resource ledger"
        per_k.append(
            {
                "temporal_k": temporal_k,
                "candidate": asdict(
                    CudaAutotuneCandidateV2(batch_size=136, temporal_k=temporal_k)
                ),
                "maximum_vram_admitted_batch_size": (
                    per_k_maximum_vram_admitted_batch_size[temporal_k]
                ),
                "status": status,
                "reason": reason,
                "measurement_status": (
                    None if measurement is None else measurement.status
                ),
                "measurement": (
                    None
                    if measurement is None
                    else {
                        "candidate": asdict(measurement.candidate),
                        "execution_mode": measurement.execution_mode,
                        "stage_index": measurement.stage_index,
                        "status": measurement.status,
                        "committed_world_ticks": measurement.committed_world_ticks,
                        "fixture_id": (
                            None
                            if measurement.fixture is None
                            else measurement.fixture.fixture_id
                        ),
                    }
                ),
                "capacity": capacity_record,
            }
        )
    return {
        "source_binding": source_binding,
        "required_by_any_measured_vram": required,
        "present_in_resolved_batch_candidates": present,
        "per_k": per_k,
        "status": "PER_K_B136_RECEIPT_BOUND",
    }


def _device_process_memory_used_v2(torch: Any, device: Any) -> int:
    """Read the driver-visible process/device bytes at an explicit boundary."""
    observed = getattr(torch.cuda, "device_memory_used", None)
    if not callable(observed):
        raise CudaFullEvolutionAutotuneError(
            "CUDA device-process memory observation is unavailable"
        )
    value = observed(device)
    _strict_nonnegative(value, "CUDA device-process memory observation")
    return int(value)


def _percentile_v2(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise CudaFullEvolutionAutotuneError("percentile input is invalid")
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    return float(values[lower] + (values[upper] - values[lower]) * (position - lower))


def active_population_summary_v2(
    owner: CudaFullEvolutionRuntimeV2,
) -> CudaActivePopulationSummaryV2:
    """Read only active-count header words at an explicit non-timed boundary."""
    try:
        values = owner.active_counts
    except Exception as exc:
        raise CudaFullEvolutionAutotuneError(
            f"cannot project active population outside the timed interval: {exc}"
        ) from exc
    if not values or any(
        value < 0 or value > RESIDENT_CAPACITY_C_V2 for value in values
    ):
        raise CudaFullEvolutionAutotuneError("active population projection is invalid")
    return CudaActivePopulationSummaryV2(
        world_count=len(values),
        total_active_population=sum(values),
        minimum_active_population=min(values),
        maximum_active_population=max(values),
    )


def _measure_population_v2(
    owner: CudaFullEvolutionRuntimeV2,
) -> tuple[CudaActivePopulationSummaryV2, float]:
    started = perf_counter()
    population = active_population_summary_v2(owner)
    return population, perf_counter() - started


def _acknowledge_dispatch_v2(
    owner: CudaFullEvolutionRuntimeV2,
    *,
    sink_receipt_id: str,
) -> tuple[CudaFullEvolutionEventDispatchV2, float, float]:
    projection_started = perf_counter()
    dispatch = owner.drain_events(to_cpu=True)
    projection_seconds = perf_counter() - projection_started
    acknowledgement_started = perf_counter()
    receipt = seal_event_sink_receipt_v2(
        dispatch,
        sink_id=BENCHMARK_SINK_ID_V2,
        sink_receipt_id=sink_receipt_id,
    )
    owner.acknowledge_event_drain(receipt)
    acknowledgement_seconds = perf_counter() - acknowledgement_started
    if owner.has_pending_dispatch:
        raise CudaFullEvolutionAutotuneError(
            "timed dispatch was not committed after sink acknowledgement"
        )
    return dispatch, projection_seconds, acknowledgement_seconds


def _run_untimed_dispatch_v2(
    owner: CudaFullEvolutionRuntimeV2,
    *,
    ticks: int,
    dispatch_index: int,
) -> None:
    owner.advance(ticks)
    _acknowledge_dispatch_v2(owner, sink_receipt_id=f"warmup-{dispatch_index:04d}")


def _run_timed_dispatch_v2(
    owner: CudaFullEvolutionRuntimeV2,
    *,
    torch: Any,
    ticks: int,
    batch_size: int,
    dispatch_index: int,
    expected_advanced_ticks: int,
) -> CudaEventTimingSampleV2:
    start_population, start_population_seconds = _measure_population_v2(owner)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    owner.advance(ticks)
    end.record()
    # This waits for completion only; elapsed_time is exclusively CUDA-event time.
    end.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise CudaFullEvolutionAutotuneError(
            "CUDA event timing must be finite and strictly positive"
        )
    dispatch, projection_seconds, acknowledgement_seconds = _acknowledge_dispatch_v2(
        owner, sink_receipt_id=f"sample-{dispatch_index:04d}"
    )
    if owner.advanced_ticks != expected_advanced_ticks:
        raise CudaFullEvolutionAutotuneError(
            "sink-acknowledged tick count differs from the timed dispatch contract"
        )
    if dispatch.tick_count != ticks:
        raise CudaFullEvolutionAutotuneError(
            "drained dispatch tick count differs from the timed dispatch"
        )
    end_population, end_population_seconds = _measure_population_v2(owner)
    return CudaEventTimingSampleV2(
        cuda_event_elapsed_ms=elapsed_ms,
        committed_world_ticks=batch_size * ticks,
        drain_projection_seconds=projection_seconds,
        drain_reported_seconds=float(dispatch.drain_elapsed_seconds),
        content_hash_seconds=float(dispatch.content_hash_elapsed_seconds),
        acknowledgement_seconds=acknowledgement_seconds,
        start_population=start_population,
        end_population=end_population,
        start_population_projection_seconds=start_population_seconds,
        end_population_projection_seconds=end_population_seconds,
        event_content_sha256=str(dispatch.content_sha256),
    )


def _capture_execution_mode_evidence_v2(
    owner: CudaFullEvolutionRuntimeV2,
) -> CudaExecutionModeEvidenceV2:
    """Copy the native topology and committed launch receipt after one sample."""
    provider = getattr(owner, "execution_plan_receipt_v2", None)
    execution_mode = getattr(owner, "execution_mode", None)
    if not callable(provider) or execution_mode not in EXECUTION_MODES_V2:
        raise CudaFullEvolutionAutotuneError(
            "runtime does not expose exact execution-plan evidence"
        )
    raw = provider()
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != EXECUTION_PLAN_RECEIPT_SCHEMA_V2
        or not isinstance(raw.get("topology"), dict)
        or not isinstance(raw.get("last_execution"), dict)
        or raw.get("closed") is not False
    ):
        raise CudaFullEvolutionAutotuneError(
            "runtime execution-plan evidence is incomplete"
        )
    topology = json.loads(json.dumps(raw["topology"], sort_keys=True))
    launch_receipt = json.loads(json.dumps(raw["last_execution"], sort_keys=True))
    return CudaExecutionModeEvidenceV2(
        execution_mode=execution_mode,
        topology=topology,
        launch_receipt=launch_receipt,
        topology_sha256=_canonical_sha256_v2(topology),
        launch_receipt_sha256=_canonical_sha256_v2(launch_receipt),
    )


def _default_owner_factory_v2(
    contract: CudaFullEvolutionNativeContractV2,
    *,
    artifact_path: Path,
    seed: int,
    device: str,
    execution_mode: str,
) -> CudaFullEvolutionRuntimeV2:
    return CudaFullEvolutionRuntimeV2.initialize(
        contract,
        seed=seed,
        artifact_path=artifact_path,
        device=device,
        enable_reference=False,
        first_world_uid=1,
        execution_mode=execution_mode,
    )


def _restore_owner_v2(
    checkpoint: CudaFullEvolutionCheckpointV2,
    *,
    artifact_path: Path,
    device: str,
    execution_mode: str,
) -> CudaFullEvolutionRuntimeV2:
    return CudaFullEvolutionRuntimeV2.from_checkpoint(
        checkpoint,
        artifact_path=artifact_path,
        device=device,
        enable_reference=False,
        execution_mode=execution_mode,
    )


def _aggregate_counters_v2(owner: CudaFullEvolutionRuntimeV2) -> dict[str, int]:
    try:
        per_world = owner.counters()
    except Exception as exc:
        raise CudaFullEvolutionAutotuneError(
            f"cannot project causal counters outside the timed interval: {exc}"
        ) from exc
    totals = {name: 0 for name in COUNTER_NAMES_V2}
    for row in per_world:
        if set(row) != set(COUNTER_NAMES_V2):
            raise CudaFullEvolutionAutotuneError("causal counter projection is invalid")
        for name in COUNTER_NAMES_V2:
            value = row[name]
            if type(value) is not int or value < 0:
                raise CudaFullEvolutionAutotuneError(
                    "causal counter projection is invalid"
                )
            totals[name] += value
    return totals


def _fixture_identity_v2(
    *,
    candidate: CudaAutotuneCandidateV2,
    seed: int,
    checkpoint: CudaFullEvolutionCheckpointV2,
) -> str:
    return _fixture_receipt_identity_v2(
        candidate=candidate,
        seed=seed,
        checkpoint_metadata_sha256=checkpoint.metadata_seal_sha256,
        checkpoint_state_sha256=checkpoint.canonical_state_sha256,
    )


def _fixture_receipt_identity_v2(
    *,
    candidate: CudaAutotuneCandidateV2,
    seed: int,
    checkpoint_metadata_sha256: str,
    checkpoint_state_sha256: str,
) -> str:
    body = {
        "schema_version": "eiel_cuda_full_evolution_forced_causal_fixture_v1",
        "candidate": asdict(candidate),
        "seed": seed,
        "checkpoint_metadata_sha256": checkpoint_metadata_sha256,
        "checkpoint_state_sha256": checkpoint_state_sha256,
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _fixture_evidence_v2(
    *,
    candidate: CudaAutotuneCandidateV2,
    seed: int,
    checkpoint: CudaFullEvolutionCheckpointV2,
    starting_counters: dict[str, int],
    starting_population: CudaActivePopulationSummaryV2,
    final_counters: dict[str, int],
    final_population: CudaActivePopulationSummaryV2,
    samples: tuple[CudaEventTimingSampleV2, ...],
) -> CudaCausalFixtureEvidenceV2:
    deltas = {
        name: final_counters[name] - starting_counters[name]
        for name in COUNTER_NAMES_V2
    }
    state_digests = tuple(sample.final_state_sha256 for sample in samples)
    event_digests = tuple(sample.event_content_sha256 for sample in samples)
    invalid: list[str] = []
    missing = [name for name, value in deltas.items() if value <= 0]
    if missing:
        invalid.append("nonpositive_capability_deltas:" + ",".join(missing))
    if (
        starting_population.total_active_population <= 0
        or final_population.total_active_population <= 0
    ):
        invalid.append("extinction")
    if not event_digests or any(len(value) != 64 for value in event_digests):
        invalid.append("missing_event_digest")
    if not state_digests or any(len(value) != 64 for value in state_digests):
        invalid.append("missing_state_digest")
    if len(set(event_digests)) != 1:
        invalid.append("repeated_event_digest_mismatch")
    if len(set(state_digests)) != 1:
        invalid.append("repeated_state_digest_mismatch")
    if state_digests and state_digests[0] == checkpoint.canonical_state_sha256:
        invalid.append("no_state_change")
    return CudaCausalFixtureEvidenceV2(
        fixture_id=_fixture_identity_v2(
            candidate=candidate, seed=seed, checkpoint=checkpoint
        ),
        starting_checkpoint_metadata_sha256=checkpoint.metadata_seal_sha256,
        starting_state_sha256=checkpoint.canonical_state_sha256,
        starting_counters=starting_counters,
        final_counters=final_counters,
        starting_population=starting_population,
        final_population=final_population,
        event_content_sha256s=event_digests,
        final_state_sha256s=state_digests,
        valid=not invalid,
        invalid_reasons=tuple(invalid),
    )


def measure_candidate_v2(
    candidate: CudaAutotuneCandidateV2,
    *,
    plan: CudaEventAutotunePlanV2,
    torch: Any,
    free_vram_bytes: int,
    total_vram_bytes: int,
    reserve_bytes: int,
    execution_mode: str = EAGER_EXECUTION_MODE_V2,
    stage_index: int | None = None,
    owner_factory: Callable[
        ..., CudaFullEvolutionRuntimeV2
    ] = _default_owner_factory_v2,
    restore_owner: Callable[..., CudaFullEvolutionRuntimeV2] = _restore_owner_v2,
) -> CudaAutotuneMeasurementV2:
    if execution_mode not in EXECUTION_MODES_V2:
        raise CudaFullEvolutionAutotuneError("execution_mode is invalid")
    if stage_index is not None:
        _strict_nonnegative(stage_index, "stage_index")
    capacity = capacity_for_candidate_v2(
        candidate,
        free_vram_bytes=free_vram_bytes,
        total_vram_bytes=total_vram_bytes,
        reserve_bytes=reserve_bytes,
        dispatches_per_candidate=plan.dispatches_per_candidate,
    )
    if not capacity.admitted:
        return CudaAutotuneMeasurementV2(
            candidate=candidate,
            status="SKIPPED_CAPACITY",
            capacity=capacity,
            execution_mode=execution_mode,
            stage_index=stage_index,
        )
    device = plan.device
    contract = CudaFullEvolutionNativeContractV2(
        B=candidate.batch_size,
        # Every warmup/sample restores this exact contract.  It describes one
        # causal trajectory, never a concatenation of timing repetitions.
        total_ticks=candidate.temporal_k,
        temporal_k=candidate.temporal_k,
    )
    # The C++ entrypoint obtains the current stream under its own CUDAGuard.
    # Keep both CUDA events on the requested device/stream around only advance().
    fixture_owner: CudaFullEvolutionRuntimeV2 | None = None
    owner: CudaFullEvolutionRuntimeV2 | None = None
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        fixture_owner = owner_factory(
            contract,
            artifact_path=plan.artifact_path,
            seed=plan.seed,
            device=device,
            execution_mode=execution_mode,
        )
        allocated_after_initialization = int(torch.cuda.memory_allocated(device))
        reserved_after_initialization = int(torch.cuda.memory_reserved(device))
        device_process_after_initialization = _device_process_memory_used_v2(
            torch, device
        )
        peak_device_process = device_process_after_initialization
        try:
            fixture_owner.force_test_conditions()
            checkpoint = fixture_owner.checkpoint()
            initial_population, initial_population_seconds = _measure_population_v2(
                fixture_owner
            )
            initial_counters = _aggregate_counters_v2(fixture_owner)
            fixture_owner = None
            # Warmup is deliberately a different restored owner.  Each warmup
            # starts at the sealed tick-zero fixture; it cannot consume a timed
            # sample's trajectory or extend unique ticks beyond temporal_k <= 64.
            for index in range(plan.warmup_dispatches):
                owner = restore_owner(
                    checkpoint,
                    artifact_path=plan.artifact_path,
                    device=device,
                    execution_mode=execution_mode,
                )
                _run_untimed_dispatch_v2(
                    owner, ticks=candidate.temporal_k, dispatch_index=index
                )
                peak_device_process = max(
                    peak_device_process, _device_process_memory_used_v2(torch, device)
                )
                owner = None
            samples_list: list[CudaEventTimingSampleV2] = []
            execution_evidence: list[CudaExecutionModeEvidenceV2] = []
            final_counters: dict[str, int] | None = None
            final_population: CudaActivePopulationSummaryV2 | None = None
            for index in range(plan.samples_per_candidate):
                owner = restore_owner(
                    checkpoint,
                    artifact_path=plan.artifact_path,
                    device=device,
                    execution_mode=execution_mode,
                )
                sample = _run_timed_dispatch_v2(
                    owner,
                    torch=torch,
                    ticks=candidate.temporal_k,
                    batch_size=candidate.batch_size,
                    dispatch_index=index,
                    expected_advanced_ticks=candidate.temporal_k,
                )
                execution_evidence.append(_capture_execution_mode_evidence_v2(owner))
                peak_device_process = max(
                    peak_device_process, _device_process_memory_used_v2(torch, device)
                )
                if owner.remaining_ticks != 0:
                    raise CudaFullEvolutionAutotuneError(
                        "restored sample did not complete its bounded absolute-tick contract"
                    )
                final_checkpoint = owner.checkpoint()
                sample = replace(
                    sample, final_state_sha256=final_checkpoint.canonical_state_sha256
                )
                samples_list.append(sample)
                counters = _aggregate_counters_v2(owner)
                population = sample.end_population
                if final_counters is None:
                    final_counters = counters
                    final_population = population
                elif final_counters != counters or final_population != population:
                    raise CudaFullEvolutionAutotuneError(
                        "identical sealed fixture did not reproduce counters or population"
                    )
                owner = None
            samples = tuple(samples_list)
            assert final_counters is not None and final_population is not None
            fixture = _fixture_evidence_v2(
                candidate=candidate,
                seed=plan.seed,
                checkpoint=checkpoint,
                starting_counters=initial_counters,
                starting_population=initial_population,
                final_counters=final_counters,
                final_population=final_population,
                samples=samples,
            )
            torch.cuda.synchronize(device)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            peak_device_process = max(
                peak_device_process, _device_process_memory_used_v2(torch, device)
            )
            admitted_working_bytes = _maximum_working_bytes_v2(
                free_vram_bytes, reserve_bytes
            )
            status = "VALID"
            if not fixture.valid:
                status = "INVALID_CAUSAL_FIXTURE"
            elif (
                peak_allocated > admitted_working_bytes
                or peak_reserved > admitted_working_bytes
                or peak_device_process > admitted_working_bytes
            ):
                status = "INVALID_MEASURED_PEAK"
            return CudaAutotuneMeasurementV2(
                candidate=candidate,
                status=status,
                capacity=capacity,
                execution_mode=execution_mode,
                samples=samples,
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
                peak_device_process_bytes=peak_device_process,
                allocated_after_initialization_bytes=allocated_after_initialization,
                reserved_after_initialization_bytes=reserved_after_initialization,
                device_process_after_initialization_bytes=(
                    device_process_after_initialization
                ),
                initial_population=initial_population,
                initial_population_projection_seconds=initial_population_seconds,
                final_population=final_population,
                fixture=fixture,
                execution_evidence=tuple(execution_evidence),
                stage_index=stage_index,
            )
        finally:
            owner = None
            fixture_owner = None
            torch.cuda.empty_cache()


def measure_c512_saturation_v2(
    selected: CudaAutotuneMeasurementV2,
    *,
    plan: CudaEventAutotunePlanV2,
    torch: Any,
    artifact_sha256: str,
    manifest_sha256: str,
    device_identity: dict[str, Any],
    owner_factory: Callable[
        ..., CudaFullEvolutionRuntimeV2
    ] = _default_owner_factory_v2,
    restore_owner: Callable[..., CudaFullEvolutionRuntimeV2] = _restore_owner_v2,
) -> CudaC512SaturationEvidenceV2:
    """Time one independent exact-C512 dispatch from its own sealed fixture."""
    candidate = selected.candidate
    execution_mode = selected.execution_mode
    contract = CudaFullEvolutionNativeContractV2(
        B=candidate.batch_size,
        total_ticks=candidate.temporal_k,
        temporal_k=candidate.temporal_k,
    )
    device = plan.device
    fixture_owner: CudaFullEvolutionRuntimeV2 | None = None
    owner: CudaFullEvolutionRuntimeV2 | None = None
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        fixture_owner = owner_factory(
            contract,
            artifact_path=plan.artifact_path,
            seed=plan.seed,
            device=device,
            execution_mode=execution_mode,
        )
        try:
            provider = getattr(
                fixture_owner, "force_full_capacity_benchmark_fixture_v2", None
            )
            if not callable(provider):
                raise CudaFullEvolutionAutotuneError(
                    "runtime lacks force_full_capacity_benchmark_fixture_v2()"
                )
            checkpoint = provider()
            # A bound method owns its instance.  Drop it immediately after the
            # fixture checkpoint is sealed so fixture_owner = None below really
            # releases the multi-GiB event/state allocation before restore.
            provider = None
            if not isinstance(checkpoint, CudaFullEvolutionCheckpointV2) and not all(
                hasattr(checkpoint, name)
                for name in (
                    "metadata_seal_sha256",
                    "canonical_state_sha256",
                    "artifact_sha256",
                    "manifest_sha256",
                    "device_identity",
                    "population_fixture_id",
                )
            ):
                raise CudaFullEvolutionAutotuneError(
                    "full-capacity fixture did not return a sealed checkpoint"
                )
            if (
                checkpoint.artifact_sha256 != artifact_sha256
                or checkpoint.manifest_sha256 != manifest_sha256
                or dict(checkpoint.device_identity) != device_identity
                or checkpoint.population_fixture_id
                != FULL_CAPACITY_BENCHMARK_FIXTURE_V2
            ):
                raise CudaFullEvolutionAutotuneError(
                    "full-capacity checkpoint is not source/device bound"
                )
            starting_population = active_population_summary_v2(fixture_owner)
            if not _is_c512_population_v2(starting_population):
                raise CudaFullEvolutionAutotuneError(
                    "full-capacity fixture did not produce exact C512"
                )
            starting_counters = _aggregate_counters_v2(fixture_owner)
            fixture_owner = None
            torch.cuda.empty_cache()
            owner = restore_owner(
                checkpoint,
                artifact_path=plan.artifact_path,
                device=device,
                execution_mode=execution_mode,
            )
            if active_population_summary_v2(owner) != starting_population:
                raise CudaFullEvolutionAutotuneError(
                    "restored full-capacity fixture changed population"
                )
            sample = _run_timed_dispatch_v2(
                owner,
                torch=torch,
                ticks=candidate.temporal_k,
                batch_size=candidate.batch_size,
                dispatch_index=0,
                expected_advanced_ticks=candidate.temporal_k,
            )
            execution_evidence = _capture_execution_mode_evidence_v2(owner)
            final_checkpoint = owner.checkpoint()
            sample = replace(
                sample, final_state_sha256=final_checkpoint.canonical_state_sha256
            )
            final_population = sample.end_population
            final_counters = _aggregate_counters_v2(owner)
            fixture_id = _c512_fixture_id_v2(
                candidate=candidate,
                execution_mode=execution_mode,
                artifact_sha256=artifact_sha256,
                manifest_sha256=manifest_sha256,
                device_identity=device_identity,
                seed=plan.seed,
                checkpoint_metadata_sha256=checkpoint.metadata_seal_sha256,
                checkpoint_state_sha256=checkpoint.canonical_state_sha256,
            )
            provisional = CudaC512SaturationEvidenceV2(
                candidate=candidate,
                execution_mode=execution_mode,
                artifact_sha256=artifact_sha256,
                manifest_sha256=manifest_sha256,
                device_identity=device_identity,
                seed=plan.seed,
                population_fixture_id=FULL_CAPACITY_BENCHMARK_FIXTURE_V2,
                fixture_id=fixture_id,
                starting_checkpoint_metadata_sha256=checkpoint.metadata_seal_sha256,
                starting_state_sha256=checkpoint.canonical_state_sha256,
                starting_counters=starting_counters,
                final_counters=final_counters,
                starting_population=starting_population,
                final_population=final_population,
                sample=sample,
                execution_evidence=execution_evidence,
                fixture_lineage_sha256="",
                evidence_sha256="",
            )
            with_lineage = replace(
                provisional,
                fixture_lineage_sha256=_canonical_sha256_v2(
                    _c512_lineage_body_v2(provisional)
                ),
            )
            return replace(
                with_lineage,
                evidence_sha256=_canonical_sha256_v2(
                    _c512_evidence_body_v2(with_lineage)
                ),
            )
        finally:
            owner = None
            fixture_owner = None
            torch.cuda.empty_cache()


def select_best_measurement_v2(
    measurements: Sequence[CudaAutotuneMeasurementV2],
) -> CudaAutotuneMeasurementV2:
    eligible = [
        value
        for value in measurements
        if value.status == "VALID"
        and value.execution_mode in EXECUTION_MODES_V2
        and value.p10_sample_world_ticks_per_second is not None
        and math.isfinite(float(value.p10_sample_world_ticks_per_second))
        and float(value.p10_sample_world_ticks_per_second) > 0
        and value.peak_allocated_bytes is not None
        and value.peak_reserved_bytes is not None
        and value.peak_device_process_bytes is not None
    ]
    if not eligible:
        raise CudaFullEvolutionAutotuneError(
            "no VALID exact B/K candidate completed the causal and allocator gates"
        )
    # Execution mode is a direct implementation comparison for one exact B/K.
    # Keep only its faster valid p10 mode before applying the cross-candidate
    # one-percent resource tie break.
    by_candidate: dict[CudaAutotuneCandidateV2, list[CudaAutotuneMeasurementV2]] = {}
    for value in eligible:
        by_candidate.setdefault(value.candidate, []).append(value)
    candidate_mode_winners = [
        min(
            values,
            key=lambda value: (
                -float(value.p10_sample_world_ticks_per_second or 0.0),
                value.peak_allocated_bytes,
                value.peak_reserved_bytes,
                value.peak_device_process_bytes,
                EXECUTION_MODES_V2.index(value.execution_mode),
            ),
        )
        for values in by_candidate.values()
    ]
    maximum_p10 = max(
        float(value.p10_sample_world_ticks_per_second or 0.0)
        for value in candidate_mode_winners
    )
    within_one_percent = [
        value
        for value in candidate_mode_winners
        if float(value.p10_sample_world_ticks_per_second or 0.0) >= maximum_p10 * 0.99
    ]
    return min(
        within_one_percent,
        key=lambda value: (
            (
                value.peak_allocated_bytes
                if value.peak_allocated_bytes is not None
                else float("inf")
            ),
            (
                value.peak_reserved_bytes
                if value.peak_reserved_bytes is not None
                else float("inf")
            ),
            (
                value.peak_device_process_bytes
                if value.peak_device_process_bytes is not None
                else float("inf")
            ),
            value.candidate.batch_size,
            value.candidate.temporal_k,
            EXECUTION_MODES_V2.index(value.execution_mode),
        ),
    )


def _is_sha256_v2(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_c512_population_v2(value: CudaActivePopulationSummaryV2 | None) -> bool:
    return (
        value is not None
        and value.world_count > 0
        and value.minimum_active_population == RESIDENT_CAPACITY_C_V2
        and value.maximum_active_population == RESIDENT_CAPACITY_C_V2
        and value.total_active_population == value.world_count * RESIDENT_CAPACITY_C_V2
    )


def _finite_nonnegative_v2(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _execution_mode_evidence_closure_v2(
    measurement: CudaAutotuneMeasurementV2,
    *,
    artifact_sha256: object,
    manifest_sha256: object,
    device_identity: object,
) -> tuple[bool, tuple[str, ...]]:
    """Validate actual monolith launch receipts without invented node timing."""
    failures: list[str] = []
    mode = measurement.execution_mode
    if mode not in EXECUTION_MODES_V2:
        return False, ("execution_mode_is_invalid",)
    if (
        not _is_sha256_v2(artifact_sha256)
        or not _is_sha256_v2(manifest_sha256)
        or not isinstance(device_identity, dict)
        or type(device_identity.get("index")) is not int
    ):
        return False, ("execution_evidence_outer_source_is_invalid",)
    evidence = measurement.execution_evidence
    if len(evidence) != len(measurement.samples) or not evidence:
        return False, ("execution_receipt_not_captured_for_every_sample",)
    graph_mode = mode == CUDA_GRAPH_EXECUTION_MODE_V2
    expected_nodes = ["kernel"] if graph_mode else []
    for index, value in enumerate(evidence):
        prefix = f"sample_{index}_"
        topology = value.topology
        launch = value.launch_receipt
        if value.execution_mode != mode:
            failures.append(prefix + "execution_mode_mismatch")
        if (
            not isinstance(topology, dict)
            or value.topology_sha256 != _canonical_sha256_v2(topology)
            or not _is_sha256_v2(value.topology_sha256)
        ):
            failures.append(prefix + "topology_seal_mismatch")
            continue
        if (
            topology.get("schema_version") != EXECUTION_PLAN_TOPOLOGY_SCHEMA_V2
            or topology.get("execution_mode") != mode
            or topology.get("artifact_sha256") != artifact_sha256
            or topology.get("manifest_sha256") != manifest_sha256
            or topology.get("device_identity") != device_identity
            or topology.get("device_index") != device_identity.get("index")
            or topology.get("B") != measurement.candidate.batch_size
            or topology.get("dispatch_temporal_k") != measurement.candidate.temporal_k
            or topology.get("event_capacity_k") != measurement.candidate.temporal_k
            or topology.get("capability_mask") != FULL_CAPABILITY_MASK_V2
            or topology.get("execution_kernel_count") != 1
            or topology.get("graph_node_count") != (1 if graph_mode else 0)
            or topology.get("node_types") != expected_nodes
            or topology.get("dependency_edges") != []
            or topology.get("runtime_inspected") is not graph_mode
            or topology.get("declaration_is_execution_evidence") is not False
            or topology.get("strong_tensor_references") is not True
            or topology.get("pointer_substitution_allowed") is not False
        ):
            failures.append(prefix + "topology_is_not_exact_source_bound_monolith")
        kernel_nodes = topology.get("kernel_nodes")
        if (
            not isinstance(kernel_nodes, list)
            or len(kernel_nodes) != 1
            or not isinstance(kernel_nodes[0], dict)
            or kernel_nodes[0].get("name") != CUDA_GRAPH_KERNEL_V2
            or kernel_nodes[0].get("node_type") != "kernel"
            or kernel_nodes[0].get("runtime_graph_node") is not graph_mode
        ):
            failures.append(prefix + "execution_kernel_is_not_exact_monolith")
        handle_fields = (
            topology.get("construction_stream_handle"),
            topology.get("graph_handle"),
            topology.get("graph_exec_handle"),
        )
        if any(type(handle) is not int or handle < 0 for handle in handle_fields):
            failures.append(prefix + "execution_handles_are_invalid")
        elif graph_mode:
            if handle_fields[1] == 0 or handle_fields[2] == 0:
                failures.append(prefix + "graph_handles_are_missing")
        elif handle_fields[1:] != (0, 0):
            failures.append(prefix + "eager_receipt_fabricates_graph_handles")
        if (
            not isinstance(launch, dict)
            or value.launch_receipt_sha256 != _canonical_sha256_v2(launch)
            or not _is_sha256_v2(value.launch_receipt_sha256)
        ):
            failures.append(prefix + "launch_receipt_seal_mismatch")
            continue
        sample = measurement.samples[index]
        if (
            launch.get("schema_version") != NATIVE_EXECUTION_RECEIPT_SCHEMA_V2
            or launch.get("execution_mode") != mode
            or launch.get("artifact_sha256") != artifact_sha256
            or launch.get("manifest_sha256") != manifest_sha256
            or launch.get("device_identity") != device_identity
            or launch.get("device_index") != device_identity.get("index")
            or launch.get("B") != measurement.candidate.batch_size
            or launch.get("tick_count") != measurement.candidate.temporal_k
            or launch.get("dispatch_temporal_k") != measurement.candidate.temporal_k
            or launch.get("event_capacity_k") != measurement.candidate.temporal_k
            or launch.get("start_next_absolute_tick") != 0
            or launch.get("end_next_absolute_tick") != measurement.candidate.temporal_k
            or launch.get("capability_mask") != FULL_CAPABILITY_MASK_V2
            or launch.get("enqueued_world_ticks")
            != measurement.candidate.batch_size * measurement.candidate.temporal_k
            or launch.get("committed_world_ticks") != sample.committed_world_ticks
            or launch.get("committed_next_absolute_tick")
            != measurement.candidate.temporal_k
            or launch.get("kernel") != CUDA_GRAPH_KERNEL_V2
            or launch.get("graph_launched") is not graph_mode
            or launch.get("topology_graph_node_count") != (1 if graph_mode else 0)
            or launch.get("topology_execution_kernel_count") != 1
            or launch.get("topology_node_types") != expected_nodes
            or launch.get("topology_dependency_edges") != []
            or launch.get("runtime_inspected") is not graph_mode
            or launch.get("runtime_status") != "committed"
            or launch.get("event_status") != "exact_content_sink_acknowledged"
            or launch.get("event_content_sha256") != sample.event_content_sha256
            or launch.get("per_node_timing") is not None
            or launch.get("inter_node_gap_timing") is not None
            or launch.get("timing_fabrication_allowed") is not False
        ):
            failures.append(prefix + "launch_receipt_is_not_exact_committed_monolith")
        if launch.get("topology_kernel_nodes") != kernel_nodes:
            failures.append(prefix + "launch_topology_kernel_mismatch")
        if launch.get("pointer_bindings") != topology.get("pointer_bindings"):
            failures.append(prefix + "launch_pointer_binding_mismatch")
        for digest_name in (
            "plan_identity_sha256",
            "pointer_binding_identity_sha256",
            "capture_identity_sha256",
            "sink_receipt_sha256",
        ):
            if not _is_sha256_v2(launch.get(digest_name)):
                failures.append(prefix + digest_name + "_is_invalid")
    return not failures, tuple(failures)


def _fixture_equivalence_projection_v2(
    measurement: CudaAutotuneMeasurementV2,
) -> dict[str, Any]:
    fixture = measurement.fixture
    if fixture is None:
        raise CudaFullEvolutionAutotuneError("mode comparison fixture is missing")
    return {
        "fixture_id": fixture.fixture_id,
        "starting_checkpoint_metadata_sha256": (
            fixture.starting_checkpoint_metadata_sha256
        ),
        "starting_state_sha256": fixture.starting_state_sha256,
        "starting_counters": fixture.starting_counters,
        "final_counters": fixture.final_counters,
        "starting_population": asdict(fixture.starting_population),
        "final_population": asdict(fixture.final_population),
        "event_content_sha256s": list(fixture.event_content_sha256s),
        "final_state_sha256s": list(fixture.final_state_sha256s),
    }


def execution_mode_comparison_v2(
    selected: CudaAutotuneMeasurementV2,
    measurements: Sequence[CudaAutotuneMeasurementV2],
    *,
    artifact_sha256: str,
    manifest_sha256: str,
    device_identity: dict[str, Any],
) -> dict[str, Any]:
    siblings = {
        value.execution_mode: value
        for value in measurements
        if value.candidate == selected.candidate and value.status == "VALID"
    }
    if set(siblings) != set(EXECUTION_MODES_V2):
        raise CudaFullEvolutionAutotuneError(
            "selected B/K requires valid eager and graph measurements"
        )
    for mode, sibling in siblings.items():
        eligible, failures = _execution_mode_evidence_closure_v2(
            sibling,
            artifact_sha256=artifact_sha256,
            manifest_sha256=manifest_sha256,
            device_identity=device_identity,
        )
        if not eligible:
            raise CudaFullEvolutionAutotuneError(
                f"{mode} execution evidence is invalid: {','.join(failures)}"
            )
    projections = {
        mode: _fixture_equivalence_projection_v2(sibling)
        for mode, sibling in siblings.items()
    }
    if (
        projections[EAGER_EXECUTION_MODE_V2]
        != projections[CUDA_GRAPH_EXECUTION_MODE_V2]
    ):
        raise CudaFullEvolutionAutotuneError(
            "eager and graph measurements do not share exact causal results"
        )
    if selected != siblings.get(selected.execution_mode):
        raise CudaFullEvolutionAutotuneError(
            "selected execution mode is not bound to its measured sibling"
        )
    if select_best_measurement_v2(tuple(siblings.values())) != selected:
        raise CudaFullEvolutionAutotuneError(
            "selected execution mode is not the faster valid p10 mode"
        )
    graph_topology = (
        siblings[CUDA_GRAPH_EXECUTION_MODE_V2].execution_evidence[0].topology
    )
    body = {
        "schema_version": "eiel_cuda_full_evolution_execution_mode_comparison_v2",
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "device_identity": device_identity,
        "candidate": asdict(selected.candidate),
        "selected_execution_mode": selected.execution_mode,
        "selection_statistic": "p10_committed_world_ticks_per_second",
        "p10_by_execution_mode": {
            mode: siblings[mode].p10_sample_world_ticks_per_second
            for mode in EXECUTION_MODES_V2
        },
        "fixture_equivalence_sha256": _canonical_sha256_v2(
            projections[EAGER_EXECUTION_MODE_V2]
        ),
        "graph_topology": graph_topology,
        "graph_topology_sha256": _canonical_sha256_v2(graph_topology),
        "eager_graph_replay_evidence_attached": False,
        "graph_equivalence_sentinel_retained": True,
        "status": "EXACT_MONOLITH_EQUIVALENCE_BOUND",
    }
    return {**body, "comparison_sha256": _canonical_sha256_v2(body)}


def _population_history_entry_is_live_v2(
    value: CudaActivePopulationSummaryV2 | None, *, batch_size: int
) -> bool:
    return (
        value is not None
        and value.world_count == batch_size
        and value.total_active_population > 0
        and 0 < value.minimum_active_population <= value.maximum_active_population
        and value.maximum_active_population <= RESIDENT_CAPACITY_C_V2
        and value.world_count * value.minimum_active_population
        <= value.total_active_population
        <= value.world_count * value.maximum_active_population
    )


def _fixture_closure_v2(
    measurement: CudaAutotuneMeasurementV2,
    *,
    seed: object,
) -> tuple[bool, tuple[str, ...]]:
    """Require the full causal/population receipt, not merely a graph replay."""
    fixture = measurement.fixture
    failures: list[str] = []
    if fixture is None:
        return False, ("selected_causal_fixture_is_missing",)
    if fixture.valid is not True:
        failures.append("selected_causal_fixture_is_not_valid")
    if (
        type(seed) is not int
        or seed < 0
        or fixture.fixture_id
        != _fixture_receipt_identity_v2(
            candidate=measurement.candidate,
            seed=seed if type(seed) is int else -1,
            checkpoint_metadata_sha256=(fixture.starting_checkpoint_metadata_sha256),
            checkpoint_state_sha256=fixture.starting_state_sha256,
        )
    ):
        failures.append("selected_causal_fixture_identity_is_not_source_bound")
    if fixture.invalid_reasons != ():
        failures.append("selected_causal_fixture_invalid_reasons_are_not_empty")
    if set(fixture.starting_counters) != set(COUNTER_NAMES_V2) or set(
        fixture.final_counters
    ) != set(COUNTER_NAMES_V2):
        failures.append("selected_fixture_does_not_bind_all_seven_counters")
    else:
        counter_values = (
            *(fixture.starting_counters[name] for name in COUNTER_NAMES_V2),
            *(fixture.final_counters[name] for name in COUNTER_NAMES_V2),
        )
        if any(type(value) is not int or value < 0 for value in counter_values):
            failures.append("selected_fixture_has_invalid_capability_counter")
        elif any(
            fixture.final_counters[name] <= fixture.starting_counters[name]
            for name in COUNTER_NAMES_V2
        ):
            failures.append("selected_fixture_has_nonpositive_capability_delta")
    if (
        not fixture.final_state_sha256s
        or fixture.starting_state_sha256 == fixture.final_state_sha256s[0]
    ):
        failures.append("selected_fixture_has_no_state_change")
    history = (
        fixture.starting_population,
        fixture.final_population,
        measurement.initial_population,
        measurement.final_population,
        *(sample.start_population for sample in measurement.samples),
        *(sample.end_population for sample in measurement.samples),
    )
    if not measurement.samples or any(
        not _population_history_entry_is_live_v2(
            value, batch_size=measurement.candidate.batch_size
        )
        for value in history
    ):
        failures.append("selected_fixture_population_history_is_incomplete_or_extinct")
    if (
        measurement.initial_population != fixture.starting_population
        or measurement.final_population != fixture.final_population
    ):
        failures.append("selected_fixture_population_history_is_not_bound")
    if len(fixture.event_content_sha256s) != len(measurement.samples) or len(
        fixture.final_state_sha256s
    ) != len(measurement.samples):
        failures.append("selected_fixture_digest_history_is_incomplete")
    elif (
        tuple(sample.event_content_sha256 for sample in measurement.samples)
        != fixture.event_content_sha256s
        or tuple(sample.final_state_sha256 for sample in measurement.samples)
        != fixture.final_state_sha256s
        or not _is_sha256_v2(fixture.starting_checkpoint_metadata_sha256)
        or not _is_sha256_v2(fixture.starting_state_sha256)
        or any(
            not _is_sha256_v2(value)
            for value in (*fixture.event_content_sha256s, *fixture.final_state_sha256s)
        )
    ):
        failures.append("selected_fixture_digest_history_is_not_source_bound")
    elif (
        len(set(fixture.event_content_sha256s)) != 1
        or len(set(fixture.final_state_sha256s)) != 1
    ):
        failures.append("selected_fixture_repeated_digest_history_is_not_exact")
    return not failures, tuple(failures)


def _c512_fixture_id_v2(
    *,
    candidate: CudaAutotuneCandidateV2,
    execution_mode: str,
    artifact_sha256: str,
    manifest_sha256: str,
    device_identity: dict[str, Any],
    seed: int,
    checkpoint_metadata_sha256: str,
    checkpoint_state_sha256: str,
) -> str:
    return _canonical_sha256_v2(
        {
            "schema_version": "eiel_cuda_full_evolution_c512_fixture_v2",
            "candidate": asdict(candidate),
            "execution_mode": execution_mode,
            "artifact_sha256": artifact_sha256,
            "manifest_sha256": manifest_sha256,
            "device_identity": device_identity,
            "seed": seed,
            "population_fixture_id": FULL_CAPACITY_BENCHMARK_FIXTURE_V2,
            "checkpoint_metadata_sha256": checkpoint_metadata_sha256,
            "checkpoint_state_sha256": checkpoint_state_sha256,
        }
    )


def _c512_lineage_body_v2(value: CudaC512SaturationEvidenceV2) -> dict[str, Any]:
    return {
        "candidate": asdict(value.candidate),
        "execution_mode": value.execution_mode,
        "artifact_sha256": value.artifact_sha256,
        "manifest_sha256": value.manifest_sha256,
        "device_identity": value.device_identity,
        "seed": value.seed,
        "population_fixture_id": value.population_fixture_id,
        "fixture_id": value.fixture_id,
        "starting_checkpoint_metadata_sha256": (
            value.starting_checkpoint_metadata_sha256
        ),
        "starting_state_sha256": value.starting_state_sha256,
        "starting_counters": value.starting_counters,
        "final_counters": value.final_counters,
        "starting_population": asdict(value.starting_population),
        "final_population": asdict(value.final_population),
        "sample": asdict(value.sample),
        "execution_evidence": asdict(value.execution_evidence),
    }


def _c512_evidence_body_v2(value: CudaC512SaturationEvidenceV2) -> dict[str, Any]:
    return {
        **_c512_lineage_body_v2(value),
        "fixture_lineage_sha256": value.fixture_lineage_sha256,
    }


def _c512_saturation_closure_v2(
    value: CudaC512SaturationEvidenceV2 | None,
    selected: CudaAutotuneMeasurementV2,
    *,
    artifact_sha256: object,
    manifest_sha256: object,
    device_identity: object,
    seed: object,
) -> tuple[bool, tuple[str, ...]]:
    if value is None:
        return False, ("C512_saturation_measurement_is_missing",)
    failures: list[str] = []
    if (
        value.candidate != selected.candidate
        or value.execution_mode != selected.execution_mode
    ):
        failures.append("C512_candidate_or_execution_mode_is_not_selected")
    if (
        value.artifact_sha256 != artifact_sha256
        or value.manifest_sha256 != manifest_sha256
        or value.device_identity != device_identity
        or value.seed != seed
    ):
        failures.append("C512_source_binding_is_not_outer_exact")
    if value.population_fixture_id != FULL_CAPACITY_BENCHMARK_FIXTURE_V2:
        failures.append("C512_runtime_fixture_identity_is_not_exact")
    if not _is_c512_population_v2(
        value.starting_population
    ) or not _is_c512_population_v2(value.final_population):
        failures.append("C512_population_is_not_exactly_saturated")
    if (
        value.starting_population.world_count != value.candidate.batch_size
        or value.final_population.world_count != value.candidate.batch_size
        or value.sample.start_population != value.starting_population
        or value.sample.end_population != value.final_population
        or not _timing_sample_is_exact_v2(value.sample, candidate=value.candidate)
    ):
        failures.append("C512_timing_sample_is_not_exactly_fixture_bound")
    if (
        set(value.starting_counters) != set(COUNTER_NAMES_V2)
        or set(value.final_counters) != set(COUNTER_NAMES_V2)
        or any(
            type(counter) is not int or counter < 0
            for counter in (
                *value.starting_counters.values(),
                *value.final_counters.values(),
            )
        )
        or any(
            value.final_counters[name] < value.starting_counters[name]
            for name in COUNTER_NAMES_V2
        )
    ):
        failures.append("C512_counters_are_not_exact_monotonic_projections")
    if (
        not _is_sha256_v2(value.starting_checkpoint_metadata_sha256)
        or not _is_sha256_v2(value.starting_state_sha256)
        or value.fixture_id
        != _c512_fixture_id_v2(
            candidate=value.candidate,
            execution_mode=value.execution_mode,
            artifact_sha256=value.artifact_sha256,
            manifest_sha256=value.manifest_sha256,
            device_identity=value.device_identity,
            seed=value.seed,
            checkpoint_metadata_sha256=value.starting_checkpoint_metadata_sha256,
            checkpoint_state_sha256=value.starting_state_sha256,
        )
    ):
        failures.append("C512_fixture_identity_is_not_recomputed_exactly")
    if value.fixture_lineage_sha256 != _canonical_sha256_v2(
        _c512_lineage_body_v2(value)
    ):
        failures.append("C512_fixture_lineage_seal_is_invalid")
    if value.evidence_sha256 != _canonical_sha256_v2(_c512_evidence_body_v2(value)):
        failures.append("C512_evidence_seal_is_invalid")
    if selected.fixture is None or (
        value.fixture_id == selected.fixture.fixture_id
        or value.starting_checkpoint_metadata_sha256
        == selected.fixture.starting_checkpoint_metadata_sha256
        or value.starting_state_sha256 == selected.fixture.starting_state_sha256
    ):
        failures.append("C512_fixture_is_not_distinct_from_selected_P256_fixture")
    synthetic = replace(
        selected,
        samples=(value.sample,),
        execution_mode=value.execution_mode,
        execution_evidence=(value.execution_evidence,),
    )
    execution_eligible, execution_failures = _execution_mode_evidence_closure_v2(
        synthetic,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        device_identity=device_identity,
    )
    if not execution_eligible:
        failures.extend(f"C512_{failure}" for failure in execution_failures)
    return not failures, tuple(failures)


def _b136_capacity_record_is_exact_v2(
    value: object,
    *,
    temporal_k: int,
    admitted: bool,
    source_binding: dict[str, Any],
) -> bool:
    expected_capacity = capacity_for_candidate_v2(
        CudaAutotuneCandidateV2(batch_size=136, temporal_k=temporal_k),
        free_vram_bytes=source_binding["initial_free_vram_bytes"],
        total_vram_bytes=source_binding["initial_total_vram_bytes"],
        reserve_bytes=source_binding["reserve_bytes"],
        dispatches_per_candidate=source_binding["dispatches_per_candidate"],
    )
    if expected_capacity.admitted is not admitted:
        return False
    return value == _b136_capacity_record_v2(
        capacity=expected_capacity,
        source_binding=source_binding,
        temporal_k=temporal_k,
    )


def _b136_closure_v2(
    b136_comparison: dict[str, Any] | None,
    *,
    temporal_k_candidates: tuple[int, ...],
    measurements: Sequence[CudaAutotuneMeasurementV2] | None = None,
    source_binding: dict[str, Any] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Require a complete valid-or-capacity-sentinel B136 record for each K."""
    if not isinstance(b136_comparison, dict):
        return False, ("B136_per_K_receipt_is_missing",)
    rows = b136_comparison.get("per_k")
    if not isinstance(rows, list):
        return False, ("B136_per_K_rows_are_missing",)
    failures: list[str] = []
    if source_binding is None:
        failures.append("B136_expected_live_source_binding_is_missing")
        expected_source: dict[str, Any] | None = None
    else:
        try:
            expected_source = _b136_source_binding_v2(
                device_identity={
                    "uuid": source_binding.get("device_uuid"),
                    "sm": source_binding.get("compute_capability"),
                },
                initial_vram={
                    "free_bytes": source_binding.get("initial_free_vram_bytes"),
                    "total_bytes": source_binding.get("initial_total_vram_bytes"),
                    "allocated_bytes": source_binding.get(
                        "initial_allocated_vram_bytes"
                    ),
                    "reserved_bytes": source_binding.get("initial_reserved_vram_bytes"),
                },
                reserve_bytes=source_binding.get("reserve_bytes"),
                dispatches_per_candidate=source_binding.get("dispatches_per_candidate"),
            )
        except CudaFullEvolutionAutotuneError:
            failures.append("B136_expected_live_source_binding_is_invalid")
            expected_source = None
    if (
        expected_source is not None
        and b136_comparison.get("source_binding") != expected_source
    ):
        failures.append("B136_live_source_binding_mismatch")
    if b136_comparison.get("status") != "PER_K_B136_RECEIPT_BOUND":
        failures.append("B136_per_K_receipt_status_is_missing")
    required_by_any = b136_comparison.get("required_by_any_measured_vram")
    present_in_envelope = b136_comparison.get("present_in_resolved_batch_candidates")
    if type(required_by_any) is not bool or type(present_in_envelope) is not bool:
        failures.append("B136_per_K_receipt_obligation_is_missing")
    by_k: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("temporal_k")) is not int:
            failures.append("B136_per_K_row_has_invalid_identity")
            continue
        temporal_k = row["temporal_k"]
        if temporal_k in by_k:
            failures.append(f"B136_K{temporal_k}_row_is_duplicated")
            continue
        by_k[temporal_k] = row
    unexpected = set(by_k) - set(temporal_k_candidates)
    if unexpected:
        failures.append("B136_per_K_receipt_has_unexpected_K_rows")
    maximums: list[int] = []
    for temporal_k in temporal_k_candidates:
        row = by_k.get(temporal_k)
        if row is None:
            failures.append(f"B136_K{temporal_k}_row_is_missing")
            continue
        status = row.get("status")
        capacity = row.get("capacity")
        maximum = row.get("maximum_vram_admitted_batch_size")
        if type(maximum) is not int or maximum < 0 or maximum % 8 != 0:
            failures.append(f"B136_K{temporal_k}_maximum_admission_is_invalid")
        else:
            maximums.append(maximum)
            if (
                expected_source is not None
                and maximum
                != maximum_vram_admitted_batch_size_v2(
                    free_vram_bytes=expected_source["initial_free_vram_bytes"],
                    reserve_bytes=expected_source["reserve_bytes"],
                    temporal_k=temporal_k,
                )
            ):
                failures.append(f"B136_K{temporal_k}_maximum_is_not_live_source_bound")
        if row.get("candidate") != {
            "batch_size": 136,
            "temporal_k": temporal_k,
        }:
            failures.append(f"B136_K{temporal_k}_candidate_identity_is_not_exact")
        if status == "CAPACITY_SKIPPED":
            if (
                not isinstance(row.get("reason"), str)
                or not row["reason"].strip()
                or expected_source is None
                or not _b136_capacity_record_is_exact_v2(
                    capacity,
                    temporal_k=temporal_k,
                    admitted=False,
                    source_binding=expected_source,
                )
                or maximum >= 136
            ):
                failures.append(f"B136_K{temporal_k}_capacity_skip_is_not_explicit")
            continue
        if (
            status != "MEASURED_VALID"
            or expected_source is None
            or not _b136_capacity_record_is_exact_v2(
                capacity,
                temporal_k=temporal_k,
                admitted=True,
                source_binding=expected_source,
            )
            or maximum < 136
        ):
            failures.append(f"B136_K{temporal_k}_is_not_validly_measured_or_skipped")
            continue
        receipt_measurement = row.get("measurement")
        if (
            row.get("measurement_status") != "VALID"
            or not isinstance(receipt_measurement, dict)
            or receipt_measurement.get("candidate")
            != {"batch_size": 136, "temporal_k": temporal_k}
            or receipt_measurement.get("execution_mode") not in EXECUTION_MODES_V2
            or receipt_measurement.get("status") != "VALID"
            or type(receipt_measurement.get("stage_index")) is not int
            or type(receipt_measurement.get("committed_world_ticks")) is not int
            or receipt_measurement["committed_world_ticks"] <= 0
            or not _is_sha256_v2(receipt_measurement.get("fixture_id"))
        ):
            failures.append(f"B136_K{temporal_k}_measured_receipt_is_not_exact")
            continue
        if measurements is not None:
            matching = [
                measurement
                for measurement in measurements
                if measurement.candidate
                == CudaAutotuneCandidateV2(batch_size=136, temporal_k=temporal_k)
                and measurement.execution_mode == receipt_measurement["execution_mode"]
            ]
            if (
                len(matching) != 1
                or matching[0].status != "VALID"
                or matching[0].stage_index != receipt_measurement["stage_index"]
                or matching[0].committed_world_ticks
                != receipt_measurement["committed_world_ticks"]
                or matching[0].fixture is None
                or matching[0].fixture.fixture_id != receipt_measurement["fixture_id"]
                or expected_source is None
                or matching[0].capacity
                != capacity_for_candidate_v2(
                    CudaAutotuneCandidateV2(batch_size=136, temporal_k=temporal_k),
                    free_vram_bytes=expected_source["initial_free_vram_bytes"],
                    total_vram_bytes=expected_source["initial_total_vram_bytes"],
                    reserve_bytes=expected_source["reserve_bytes"],
                    dispatches_per_candidate=expected_source[
                        "dispatches_per_candidate"
                    ],
                )
            ):
                failures.append(f"B136_K{temporal_k}_measurement_is_not_bound")
    if maximums:
        computed_required = any(maximum >= 136 for maximum in maximums)
        if required_by_any is not computed_required:
            failures.append("B136_per_K_receipt_obligation_does_not_match_capacity")
        if computed_required and present_in_envelope is not True:
            failures.append("B136_mandatory_comparison_is_not_present")
    return not failures, tuple(failures)


def _population_summary_is_well_formed_v2(
    value: CudaActivePopulationSummaryV2 | None, *, batch_size: int
) -> bool:
    return (
        value is not None
        and value.world_count == batch_size
        and type(value.total_active_population) is int
        and type(value.minimum_active_population) is int
        and type(value.maximum_active_population) is int
        and 0
        <= value.minimum_active_population
        <= value.maximum_active_population
        <= RESIDENT_CAPACITY_C_V2
        and value.world_count * value.minimum_active_population
        <= value.total_active_population
        <= value.world_count * value.maximum_active_population
    )


def _timing_sample_is_exact_v2(
    sample: CudaEventTimingSampleV2, *, candidate: CudaAutotuneCandidateV2
) -> bool:
    expected_ticks = candidate.batch_size * candidate.temporal_k
    nonnegative_timings = (
        sample.drain_projection_seconds,
        sample.drain_reported_seconds,
        sample.content_hash_seconds,
        sample.acknowledgement_seconds,
        sample.start_population_projection_seconds,
        sample.end_population_projection_seconds,
    )
    return (
        isinstance(sample.cuda_event_elapsed_ms, (int, float))
        and math.isfinite(float(sample.cuda_event_elapsed_ms))
        and float(sample.cuda_event_elapsed_ms) > 0
        and type(sample.committed_world_ticks) is int
        and sample.committed_world_ticks == expected_ticks
        and all(_finite_nonnegative_v2(value) for value in nonnegative_timings)
        and _population_summary_is_well_formed_v2(
            sample.start_population, batch_size=candidate.batch_size
        )
        and _population_summary_is_well_formed_v2(
            sample.end_population, batch_size=candidate.batch_size
        )
        and _is_sha256_v2(sample.event_content_sha256)
        and _is_sha256_v2(sample.final_state_sha256)
    )


def _measurement_semantics_closure_v2(
    measurement: CudaAutotuneMeasurementV2,
    *,
    expected_candidate: CudaAutotuneCandidateV2,
    expected_execution_mode: str,
    expected_stage_index: int | None,
    artifact_sha256: str,
    manifest_sha256: str,
    device_identity: dict[str, Any],
    seed: int,
    free_vram_bytes: int,
    total_vram_bytes: int,
    reserve_bytes: int,
    dispatches_per_candidate: int,
    samples_per_candidate: int,
) -> tuple[bool, tuple[str, ...]]:
    """Recompute one staged row instead of trusting its sealed summary."""
    failures: list[str] = []
    if measurement.candidate != expected_candidate:
        return False, ("candidate_identity_is_not_exact",)
    if (
        expected_execution_mode not in EXECUTION_MODES_V2
        or measurement.execution_mode != expected_execution_mode
    ):
        failures.append("execution_mode_is_not_exact")
    if measurement.stage_index != expected_stage_index:
        failures.append("stage_index_is_not_exact")
    capacity = measurement.capacity
    if (
        capacity.candidate != expected_candidate
        or type(capacity.free_vram_bytes) is not int
        or type(capacity.total_vram_bytes) is not int
        or type(capacity.reserve_bytes) is not int
        or capacity.free_vram_bytes < 0
        or capacity.free_vram_bytes > capacity.total_vram_bytes
        or capacity.free_vram_bytes != free_vram_bytes
        or capacity.total_vram_bytes != total_vram_bytes
        or capacity.reserve_bytes != reserve_bytes
    ):
        failures.append("capacity_live_source_fields_are_not_exact")
        return False, tuple(failures)
    try:
        expected_capacity = capacity_for_candidate_v2(
            expected_candidate,
            free_vram_bytes=capacity.free_vram_bytes,
            total_vram_bytes=total_vram_bytes,
            reserve_bytes=reserve_bytes,
            dispatches_per_candidate=dispatches_per_candidate,
        )
    except CudaFullEvolutionAutotuneError:
        failures.append("capacity_recomputation_failed")
        return False, tuple(failures)
    if capacity != expected_capacity:
        failures.append("capacity_ledger_or_admission_is_not_recomputed_exactly")
    if not expected_capacity.admitted:
        if measurement.status != "SKIPPED_CAPACITY":
            failures.append("capacity_rejection_status_is_not_exact")
        if (
            measurement.samples
            or measurement.peak_allocated_bytes is not None
            or measurement.peak_reserved_bytes is not None
            or measurement.peak_device_process_bytes is not None
            or measurement.allocated_after_initialization_bytes is not None
            or measurement.reserved_after_initialization_bytes is not None
            or measurement.device_process_after_initialization_bytes is not None
            or measurement.initial_population is not None
            or measurement.initial_population_projection_seconds is not None
            or measurement.final_population is not None
            or measurement.fixture is not None
            or measurement.execution_evidence
        ):
            failures.append("capacity_rejection_carries_measured_payload")
        return not failures, tuple(failures)
    if measurement.status == "SKIPPED_CAPACITY":
        failures.append("admitted_candidate_is_marked_SKIPPED_CAPACITY")
    if len(measurement.samples) != samples_per_candidate:
        failures.append("sample_count_does_not_match_plan")
    for index, sample in enumerate(measurement.samples):
        if not _timing_sample_is_exact_v2(sample, candidate=expected_candidate):
            failures.append(f"sample_{index}_semantics_are_not_exact")
    fixture = measurement.fixture
    if fixture is None or type(fixture.valid) is not bool:
        failures.append("causal_fixture_status_is_missing")
    else:
        if (
            measurement.initial_population != fixture.starting_population
            or measurement.final_population != fixture.final_population
            or any(
                sample.start_population != fixture.starting_population
                or sample.end_population != fixture.final_population
                for sample in measurement.samples
            )
        ):
            failures.append("sample_population_history_is_not_fixture_bound")
        if fixture.valid:
            fixture_eligible, _ = _fixture_closure_v2(measurement, seed=seed)
            if not fixture_eligible:
                failures.append("valid_fixture_payload_is_not_exact")
    execution_eligible, execution_failures = _execution_mode_evidence_closure_v2(
        measurement,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        device_identity=device_identity,
    )
    if not execution_eligible:
        failures.extend(execution_failures)
    allocator_values = (
        measurement.peak_allocated_bytes,
        measurement.peak_reserved_bytes,
        measurement.peak_device_process_bytes,
        measurement.allocated_after_initialization_bytes,
        measurement.reserved_after_initialization_bytes,
        measurement.device_process_after_initialization_bytes,
    )
    if any(
        type(value) is not int or value < 0 for value in allocator_values
    ) or not _finite_nonnegative_v2(measurement.initial_population_projection_seconds):
        failures.append("allocator_or_population_projection_fields_are_invalid")
        return False, tuple(failures)
    (
        peak_allocated,
        peak_reserved,
        peak_process,
        init_allocated,
        init_reserved,
        init_process,
    ) = allocator_values
    if (
        peak_allocated < init_allocated
        or peak_reserved < init_reserved
        or peak_process < init_process
    ):
        failures.append("measured_peak_does_not_cover_initialization")
    admitted_working_bytes = _maximum_working_bytes_v2(
        capacity.free_vram_bytes, reserve_bytes
    )
    measured_peak_invalid = any(
        value > admitted_working_bytes
        for value in (peak_allocated, peak_reserved, peak_process)
    )
    expected_status = (
        "INVALID_CAUSAL_FIXTURE"
        if fixture is None or fixture.valid is not True
        else "INVALID_MEASURED_PEAK" if measured_peak_invalid else "VALID"
    )
    if measurement.status != expected_status:
        failures.append("measurement_status_is_not_recomputed_exactly")
    return not failures, tuple(failures)


def _selected_measurement_closure_v2(
    measurement: CudaAutotuneMeasurementV2,
    measurements: Sequence[CudaAutotuneMeasurementV2] | None,
    *,
    initial_vram: dict[str, int] | None,
    reserve_bytes: int | None,
    dispatches_per_candidate: int | None,
    samples_per_candidate: int | None,
) -> tuple[bool, tuple[str, ...]]:
    """Keep direct verdict calls as strict as the runner's selection gate."""
    failures: list[str] = []
    if measurements is None:
        failures.append("selected_measurement_set_is_missing")
    else:
        matching = [
            value
            for value in measurements
            if value.candidate == measurement.candidate
            and value.execution_mode == measurement.execution_mode
        ]
        if len(matching) != 1 or matching[0] != measurement:
            failures.append("selected_measurement_is_not_bound_to_staged_measurements")
        try:
            recomputed = select_best_measurement_v2(measurements)
        except (
            CudaFullEvolutionAutotuneError,
            TypeError,
            ValueError,
            ZeroDivisionError,
            OverflowError,
        ):
            failures.append("selected_measurement_cannot_be_recomputed")
        else:
            if recomputed != measurement:
                failures.append("selected_measurement_is_not_the_recomputed_winner")
    if measurement.status != "VALID":
        failures.append("selected_measurement_status_is_not_valid")
    capacity = measurement.capacity
    if capacity.candidate != measurement.candidate or not capacity.admitted:
        failures.append("selected_measurement_capacity_is_not_admitted")
        return False, tuple(failures)
    if (
        type(capacity.free_vram_bytes) is not int
        or type(capacity.total_vram_bytes) is not int
        or type(capacity.reserve_bytes) is not int
        or capacity.free_vram_bytes < 0
        or capacity.total_vram_bytes < capacity.free_vram_bytes
        or capacity.reserve_bytes < 0
    ):
        failures.append("selected_measurement_live_vram_is_invalid")
        return False, tuple(failures)
    if (
        not isinstance(initial_vram, dict)
        or type(initial_vram.get("free_bytes")) is not int
        or type(initial_vram.get("total_bytes")) is not int
        or type(reserve_bytes) is not int
        or type(dispatches_per_candidate) is not int
        or type(samples_per_candidate) is not int
    ):
        failures.append("selected_measurement_plan_or_live_source_is_missing")
        return False, tuple(failures)
    try:
        expected_capacity = capacity_for_candidate_v2(
            measurement.candidate,
            free_vram_bytes=initial_vram["free_bytes"],
            total_vram_bytes=initial_vram["total_bytes"],
            reserve_bytes=reserve_bytes,
            dispatches_per_candidate=dispatches_per_candidate,
        )
    except CudaFullEvolutionAutotuneError:
        failures.append("selected_measurement_capacity_recomputation_failed")
        return False, tuple(failures)
    if capacity != expected_capacity:
        failures.append("selected_measurement_capacity_is_not_outer_live_source_bound")
    if len(measurement.samples) != samples_per_candidate:
        failures.append("selected_measurement_sample_count_does_not_match_plan")
    allocator_values = {
        "peak_allocated_bytes": measurement.peak_allocated_bytes,
        "peak_reserved_bytes": measurement.peak_reserved_bytes,
        "peak_device_process_bytes": measurement.peak_device_process_bytes,
        "allocated_after_initialization_bytes": (
            measurement.allocated_after_initialization_bytes
        ),
        "reserved_after_initialization_bytes": (
            measurement.reserved_after_initialization_bytes
        ),
        "device_process_after_initialization_bytes": (
            measurement.device_process_after_initialization_bytes
        ),
    }
    if any(type(value) is not int or value < 0 for value in allocator_values.values()):
        failures.append("selected_measurement_allocator_or_live_fields_are_missing")
        return False, tuple(failures)
    admitted_working_bytes = _maximum_working_bytes_v2(
        capacity.free_vram_bytes, capacity.reserve_bytes
    )
    if any(
        allocator_values[name] > admitted_working_bytes
        for name in (
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "peak_device_process_bytes",
        )
    ):
        failures.append("selected_measurement_peak_exceeds_admitted_working_bytes")
    if (
        allocator_values["peak_allocated_bytes"]
        < allocator_values["allocated_after_initialization_bytes"]
        or allocator_values["peak_reserved_bytes"]
        < allocator_values["reserved_after_initialization_bytes"]
        or allocator_values["peak_device_process_bytes"]
        < allocator_values["device_process_after_initialization_bytes"]
    ):
        failures.append("selected_measurement_peak_does_not_cover_live_initialization")
    return not failures, tuple(failures)


def _staged_envelope_closure_v2(
    staged_envelope: object,
    resolved_envelope: object,
    *,
    artifact_sha256: object,
    manifest_sha256: object,
    device_identity: object,
    initial_vram: object,
    reserve_bytes: object,
    dispatches_per_candidate: object,
    samples_per_candidate: object,
    seed: object,
    temporal_k_candidates: tuple[int, ...],
    measurements: Sequence[CudaAutotuneMeasurementV2] | None,
) -> tuple[bool, tuple[str, ...]]:
    """Reconstruct the aggregate receipt from live-bound inputs before closure."""
    if measurements is None:
        return False, ("staged_envelope_measurement_set_is_missing",)
    if not isinstance(staged_envelope, dict) or not isinstance(resolved_envelope, dict):
        return False, ("staged_envelope_or_resolved_envelope_is_missing",)
    if (
        not isinstance(artifact_sha256, str)
        or not isinstance(manifest_sha256, str)
        or not isinstance(device_identity, dict)
        or not isinstance(initial_vram, dict)
        or type(reserve_bytes) is not int
        or type(dispatches_per_candidate) is not int
        or type(samples_per_candidate) is not int
        or type(seed) is not int
    ):
        return False, ("staged_envelope_live_source_context_is_invalid",)
    try:
        resolved_batches = _validated_resolved_envelope_v2(
            resolved_envelope,
            artifact_sha256=artifact_sha256,
            manifest_sha256=manifest_sha256,
            device_identity=device_identity,
            initial_vram=initial_vram,
            reserve_bytes=reserve_bytes,
            temporal_k_candidates=temporal_k_candidates,
        )
        initial_total_vram_bytes = _strict_nonnegative(
            initial_vram.get("total_bytes"), "initial total VRAM"
        )
        _strict_positive(
            dispatches_per_candidate, "staged envelope dispatches_per_candidate"
        )
        _strict_positive(samples_per_candidate, "staged envelope samples_per_candidate")
        warmup_dispatches = dispatches_per_candidate - samples_per_candidate
        _strict_nonnegative(warmup_dispatches, "staged envelope warmup_dispatches")
        reconstruction_plan = CudaEventAutotunePlanV2(
            artifact_path=Path("/nonexistent/source-bound-autotune-artifact"),
            batch_candidates=resolved_batches,
            temporal_k_candidates=temporal_k_candidates,
            warmup_dispatches=warmup_dispatches,
            samples_per_candidate=samples_per_candidate,
            seed=seed,
        )
        expected_stages = candidate_stages_for_plan_v2(
            reconstruction_plan, resolved_batch_candidates=resolved_batches
        )
        measurements_by_candidate_mode = {
            (value.candidate, value.execution_mode): value for value in measurements
        }
        if len(measurements_by_candidate_mode) != len(measurements):
            return False, ("staged_envelope_measurements_are_not_unique",)
        semantic_failures: list[str] = []
        for stage in expected_stages:
            for candidate in stage.candidates:
                for execution_mode in EXECUTION_MODES_V2:
                    staged_measurement = measurements_by_candidate_mode.get(
                        (candidate, execution_mode)
                    )
                    prefix = (
                        f"B{candidate.batch_size}_K{candidate.temporal_k}_"
                        f"{execution_mode}_"
                    )
                    if staged_measurement is None:
                        semantic_failures.append(prefix + "measurement_is_missing")
                        continue
                    _, candidate_failures = _measurement_semantics_closure_v2(
                        staged_measurement,
                        expected_candidate=candidate,
                        expected_execution_mode=execution_mode,
                        expected_stage_index=stage.stage_index,
                        artifact_sha256=artifact_sha256,
                        manifest_sha256=manifest_sha256,
                        device_identity=device_identity,
                        seed=seed,
                        free_vram_bytes=initial_vram["free_bytes"],
                        total_vram_bytes=initial_total_vram_bytes,
                        reserve_bytes=reserve_bytes,
                        dispatches_per_candidate=dispatches_per_candidate,
                        samples_per_candidate=samples_per_candidate,
                    )
                    semantic_failures.extend(
                        prefix + failure for failure in candidate_failures
                    )
        expected_candidates = {
            (candidate, execution_mode)
            for stage in expected_stages
            for candidate in stage.candidates
            for execution_mode in EXECUTION_MODES_V2
        }
        if set(measurements_by_candidate_mode) != expected_candidates:
            semantic_failures.append(
                "staged_envelope_measurements_do_not_match_expected_candidates"
            )
        if semantic_failures:
            return False, tuple(semantic_failures)
        expected_receipt = staged_envelope_receipt_v2(
            artifact_sha256=artifact_sha256,
            manifest_sha256=manifest_sha256,
            device_identity=device_identity,
            temporal_k_candidates=temporal_k_candidates,
            stages=expected_stages,
            measurements=measurements,
            resolved_envelope=resolved_envelope,
            initial_vram=initial_vram,
            reserve_bytes=reserve_bytes,
            dispatches_per_candidate=dispatches_per_candidate,
            samples_per_candidate=samples_per_candidate,
        )
    except CudaFullEvolutionAutotuneError:
        return False, ("staged_envelope_live_source_reconstruction_failed",)
    if staged_envelope != expected_receipt:
        return False, ("staged_envelope_is_not_the_exact_measured_aggregate",)
    return True, ()


def _topology_source_binding_closure_v2(
    topology_identity: object,
    *,
    artifact_sha256: object,
    manifest_sha256: object,
    device_identity: object,
    execution_mode_comparison: object,
) -> tuple[bool, tuple[str, ...]]:
    """Bind the retained real graph sentinel to the outer live source."""
    if (
        not isinstance(topology_identity, dict)
        or not _is_sha256_v2(artifact_sha256)
        or not _is_sha256_v2(manifest_sha256)
        or not isinstance(device_identity, dict)
        or not isinstance(device_identity.get("uuid"), str)
        or not device_identity["uuid"].strip()
        or not isinstance(device_identity.get("sm"), str)
        or not device_identity["sm"].strip()
    ):
        return False, ("topology_outer_source_context_is_invalid",)
    if not isinstance(execution_mode_comparison, dict):
        return False, ("execution_mode_comparison_is_missing",)
    failures: list[str] = []
    if topology_identity != execution_mode_comparison.get(
        "graph_topology"
    ) or execution_mode_comparison.get("graph_topology_sha256") != _canonical_sha256_v2(
        topology_identity
    ):
        failures.append("topology_is_not_the_retained_graph_sentinel")
    if (
        topology_identity.get("artifact_sha256") != artifact_sha256
        or topology_identity.get("manifest_sha256") != manifest_sha256
        or topology_identity.get("device_identity") != device_identity
        or topology_identity.get("execution_mode") != CUDA_GRAPH_EXECUTION_MODE_V2
        or topology_identity.get("graph_node_count") != 1
        or topology_identity.get("dependency_edges") != []
    ):
        failures.append("topology_is_not_outer_source_bound_real_graph")
    return not failures, tuple(failures)


def _execution_mode_comparison_closure_v2(
    value: object,
    selected: CudaAutotuneMeasurementV2,
    measurements: Sequence[CudaAutotuneMeasurementV2] | None,
    *,
    artifact_sha256: object,
    manifest_sha256: object,
    device_identity: object,
) -> tuple[bool, tuple[str, ...]]:
    if (
        not isinstance(value, dict)
        or measurements is None
        or not isinstance(artifact_sha256, str)
        or not isinstance(manifest_sha256, str)
        or not isinstance(device_identity, dict)
    ):
        return False, ("execution_mode_comparison_context_is_missing",)
    try:
        expected = execution_mode_comparison_v2(
            selected,
            measurements,
            artifact_sha256=artifact_sha256,
            manifest_sha256=manifest_sha256,
            device_identity=device_identity,
        )
    except (CudaFullEvolutionAutotuneError, TypeError, ValueError):
        return False, ("execution_mode_comparison_reconstruction_failed",)
    if value != expected:
        return False, ("execution_mode_comparison_is_not_exact",)
    return True, ()


def _target_verdict_v2(
    measurement: CudaAutotuneMeasurementV2,
    topology_identity: dict[str, Any],
    *,
    b136_comparison: dict[str, Any] | None = None,
    temporal_k_candidates: tuple[int, ...] = DEFAULT_TEMPORAL_K_CANDIDATES_V2,
    measurements: Sequence[CudaAutotuneMeasurementV2] | None = None,
    b136_source_binding: dict[str, Any] | None = None,
    resolved_envelope: dict[str, Any] | None = None,
    staged_envelope: dict[str, Any] | None = None,
    execution_mode_comparison: dict[str, Any] | None = None,
    c512_saturation_evidence: CudaC512SaturationEvidenceV2 | None = None,
    artifact_sha256: str | None = None,
    manifest_sha256: str | None = None,
    device_identity: dict[str, Any] | None = None,
    initial_vram: dict[str, int] | None = None,
    reserve_bytes: int | None = None,
    dispatches_per_candidate: int | None = None,
    samples_per_candidate: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    try:
        percentiles = measurement.sample_percentiles_world_ticks_per_second()
    except (CudaFullEvolutionAutotuneError, TypeError, ValueError, ZeroDivisionError):
        percentiles = None
    p10 = None if percentiles is None else percentiles["p10"]
    required = float(CUDA_NATIVE_TARGET_WTPS_100X)
    threshold_met = p10 is not None and p10 >= required
    selected_eligible, selected_failures = _selected_measurement_closure_v2(
        measurement,
        measurements,
        initial_vram=initial_vram,
        reserve_bytes=reserve_bytes,
        dispatches_per_candidate=dispatches_per_candidate,
        samples_per_candidate=samples_per_candidate,
    )
    staged_eligible, staged_failures = _staged_envelope_closure_v2(
        staged_envelope,
        resolved_envelope,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        device_identity=device_identity,
        initial_vram=initial_vram,
        reserve_bytes=reserve_bytes,
        dispatches_per_candidate=dispatches_per_candidate,
        samples_per_candidate=samples_per_candidate,
        seed=seed,
        temporal_k_candidates=temporal_k_candidates,
        measurements=measurements,
    )
    mode_eligible, mode_failures = _execution_mode_comparison_closure_v2(
        execution_mode_comparison,
        measurement,
        measurements,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        device_identity=device_identity,
    )
    topology_eligible, topology_failures = _topology_source_binding_closure_v2(
        topology_identity,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        device_identity=device_identity,
        execution_mode_comparison=execution_mode_comparison,
    )
    execution_eligible, execution_failures = _execution_mode_evidence_closure_v2(
        measurement,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        device_identity=device_identity,
    )
    fixture_eligible, fixture_failures = _fixture_closure_v2(measurement, seed=seed)
    c512_eligible, c512_failures = _c512_saturation_closure_v2(
        c512_saturation_evidence,
        measurement,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        device_identity=device_identity,
        seed=seed,
    )
    expected_b136_source_binding: dict[str, Any] | None = None
    b136_binding_failures: tuple[str, ...] = ()
    try:
        if (
            not isinstance(device_identity, dict)
            or not isinstance(initial_vram, dict)
            or type(reserve_bytes) is not int
            or type(dispatches_per_candidate) is not int
        ):
            raise CudaFullEvolutionAutotuneError(
                "B136 target source context is incomplete"
            )
        expected_b136_source_binding = _b136_source_binding_v2(
            device_identity=device_identity,
            initial_vram=initial_vram,
            reserve_bytes=reserve_bytes,
            dispatches_per_candidate=dispatches_per_candidate,
        )
    except CudaFullEvolutionAutotuneError:
        b136_binding_failures = ("B136_target_live_source_context_is_invalid",)
    if b136_source_binding != expected_b136_source_binding:
        b136_binding_failures = (
            *b136_binding_failures,
            "B136_target_live_source_binding_is_missing_or_mismatched",
        )
    b136_eligible, b136_failures = _b136_closure_v2(
        b136_comparison,
        temporal_k_candidates=temporal_k_candidates,
        measurements=measurements,
        source_binding=expected_b136_source_binding,
    )
    closure_eligible = (
        selected_eligible
        and staged_eligible
        and mode_eligible
        and topology_eligible
        and execution_eligible
        and fixture_eligible
        and c512_eligible
        and not b136_binding_failures
        and b136_eligible
    )
    closure_failures = (
        *selected_failures,
        *staged_failures,
        *mode_failures,
        *topology_failures,
        *execution_failures,
        *fixture_failures,
        *c512_failures,
        *b136_binding_failures,
        *b136_failures,
    )
    if threshold_met and closure_eligible:
        status = "TARGET_CLOSURE_PASS_PRELIMINARY_NON_AUTHORITATIVE"
    elif threshold_met:
        status = "PERFORMANCE_THRESHOLD_PASS_PRELIMINARY"
    else:
        status = "PERFORMANCE_THRESHOLD_FAIL"
    return {
        "target_world_ticks_per_second": required,
        "selection_statistic": "p10_committed_world_ticks_per_second",
        "selected_p10_world_ticks_per_second": p10,
        "status": status,
        "closure_status": (
            "PRELIMINARY_DATACLASS_CLOSURE_ELIGIBLE"
            if closure_eligible
            else "PRELIMINARY_DATACLASS_CLOSURE_INELIGIBLE"
        ),
        "closure_failures": list(closure_failures),
        "authoritative": False,
        "authority_limitation": (
            "Dataclass composition cannot grant FULL_TARGET_PASS; the public "
            "validator must reopen the artifact, reload the current device, and "
            "recompute every outer binding."
        ),
        "technical_only": True,
    }


def run_cuda_event_autotune_v2(
    plan: CudaEventAutotunePlanV2,
    *,
    launch_origin_path: Path,
    torch_module: Any | None = None,
    owner_factory: Callable[
        ..., CudaFullEvolutionRuntimeV2
    ] = _default_owner_factory_v2,
    restore_owner: Callable[..., CudaFullEvolutionRuntimeV2] = _restore_owner_v2,
) -> CudaEventAutotuneResultV2:
    """Run only the plan's tiny B/K envelope and return a source-bound receipt."""
    if not plan.artifact_path.is_file():
        raise CudaFullEvolutionAutotuneError(
            f"prebuilt artifact is absent: {plan.artifact_path}"
        )
    torch = _require_torch_v2() if torch_module is None else torch_module
    loaded: LoadedCudaFullEvolutionNativeV2 = load_prebuilt_cuda_full_evolution_v2(
        plan.artifact_path, device=plan.device
    )
    torch.cuda.synchronize(plan.device)
    initial_vram = _canonical_vram_observation_v2(
        _live_vram_v2(torch, plan.device), name="launch VRAM observation"
    )
    harness_sha256 = file_sha256(Path(__file__))
    performance_restructure_contract_sha256 = file_sha256(
        PERFORMANCE_RESTRUCTURE_CONTRACT_V2
    )
    validation_matrix_sha256 = file_sha256(VALIDATION_MATRIX_V2)
    launch_origin = _publish_launch_origin_v2(
        launch_origin_path,
        artifact_path=loaded.artifact_path,
        artifact_sha256=loaded.artifact_sha256,
        manifest_sha256=loaded.manifest_sha256,
        device_identity=dict(loaded.device_identity),
        initial_vram=initial_vram,
        plan=plan,
        harness_sha256=harness_sha256,
        performance_restructure_contract_sha256=(
            performance_restructure_contract_sha256
        ),
        validation_matrix_sha256=validation_matrix_sha256,
    )
    reserve_bytes = _effective_reserve_bytes_v2(plan, initial_vram["total_bytes"])
    per_k_maximum_vram_batch = {
        temporal_k: maximum_vram_admitted_batch_size_v2(
            free_vram_bytes=initial_vram["free_bytes"],
            reserve_bytes=reserve_bytes,
            temporal_k=temporal_k,
        )
        for temporal_k in plan.temporal_k_candidates
    }
    maximum_vram_batch = max(per_k_maximum_vram_batch.values())
    required_batches, dynamic_per_k_maximum = dynamic_batch_candidates_for_k_grid_v2(
        sm_count=int(loaded.device_identity["multiprocessor_count"]),
        free_vram_bytes=initial_vram["free_bytes"],
        reserve_bytes=reserve_bytes,
        temporal_k_candidates=plan.temporal_k_candidates,
    )
    if dynamic_per_k_maximum != per_k_maximum_vram_batch:
        raise CudaFullEvolutionAutotuneError(
            "per-K dynamic candidate admission changed during envelope construction"
        )
    if plan.batch_candidates is not None:
        _validate_explicit_batch_candidates_v2(
            plan.batch_candidates,
            per_k_maximum_vram_admitted_batch_size=per_k_maximum_vram_batch,
            required_batch_candidates=required_batches,
        )
        resolved_batches = plan.batch_candidates
        plan_mode = "EXPLICIT_REQUIRED_SUPERSET"
    else:
        resolved_batches = required_batches
        plan_mode = "DYNAMIC_FULL_ENVELOPE"
    assert resolved_batches is not None
    resolved_envelope = resolved_envelope_v2(
        artifact_sha256=loaded.artifact_sha256,
        manifest_sha256=loaded.manifest_sha256,
        device_identity=dict(loaded.device_identity),
        initial_vram=initial_vram,
        reserve_bytes=reserve_bytes,
        temporal_k_candidates=plan.temporal_k_candidates,
        resolved_batch_candidates=resolved_batches,
        plan_mode=plan_mode,
    )
    stages = candidate_stages_for_plan_v2(
        plan, resolved_batch_candidates=resolved_batches
    )
    measurements: list[CudaAutotuneMeasurementV2] = []
    for stage in stages:
        if len(stage.candidates) * len(EXECUTION_MODES_V2) > MAX_CANDIDATES_PER_PLAN_V2:
            raise CudaFullEvolutionAutotuneError(
                "staged autotune round exceeds the 64-arm contract"
            )
        for candidate in stage.candidates:
            for execution_mode in EXECUTION_MODES_V2:
                measurements.append(
                    measure_candidate_v2(
                        candidate,
                        plan=plan,
                        torch=torch,
                        free_vram_bytes=initial_vram["free_bytes"],
                        total_vram_bytes=initial_vram["total_bytes"],
                        reserve_bytes=reserve_bytes,
                        execution_mode=execution_mode,
                        stage_index=stage.stage_index,
                        owner_factory=owner_factory,
                        restore_owner=restore_owner,
                    )
                )
    selected = select_best_measurement_v2(measurements)
    try:
        execution_mode_comparison = execution_mode_comparison_v2(
            selected,
            measurements,
            artifact_sha256=loaded.artifact_sha256,
            manifest_sha256=loaded.manifest_sha256,
            device_identity=dict(loaded.device_identity),
        )
    except CudaFullEvolutionAutotuneError:
        execution_mode_comparison = {}
    topology_identity = dict(execution_mode_comparison.get("graph_topology", {}))
    c512_saturation_evidence = measure_c512_saturation_v2(
        selected,
        plan=plan,
        torch=torch,
        artifact_sha256=loaded.artifact_sha256,
        manifest_sha256=loaded.manifest_sha256,
        device_identity=dict(loaded.device_identity),
        owner_factory=owner_factory,
        restore_owner=restore_owner,
    )
    b136_capacities = {
        temporal_k: capacity_for_candidate_v2(
            CudaAutotuneCandidateV2(batch_size=136, temporal_k=temporal_k),
            free_vram_bytes=initial_vram["free_bytes"],
            total_vram_bytes=initial_vram["total_bytes"],
            reserve_bytes=reserve_bytes,
            dispatches_per_candidate=plan.dispatches_per_candidate,
        )
        for temporal_k in plan.temporal_k_candidates
    }
    b136_source_binding = _b136_source_binding_v2(
        device_identity=dict(loaded.device_identity),
        initial_vram=initial_vram,
        reserve_bytes=reserve_bytes,
        dispatches_per_candidate=plan.dispatches_per_candidate,
    )
    b136_comparison = _b136_comparison_v2(
        per_k_maximum_vram_admitted_batch_size=per_k_maximum_vram_batch,
        resolved_batch_candidates=resolved_batches,
        measurements=measurements,
        capacities=b136_capacities,
        device_identity=dict(loaded.device_identity),
        initial_vram=initial_vram,
        reserve_bytes=reserve_bytes,
        dispatches_per_candidate=plan.dispatches_per_candidate,
    )
    staged_envelope = staged_envelope_receipt_v2(
        artifact_sha256=loaded.artifact_sha256,
        manifest_sha256=loaded.manifest_sha256,
        device_identity=dict(loaded.device_identity),
        temporal_k_candidates=plan.temporal_k_candidates,
        stages=stages,
        measurements=measurements,
        resolved_envelope=resolved_envelope,
        initial_vram=initial_vram,
        reserve_bytes=reserve_bytes,
        dispatches_per_candidate=plan.dispatches_per_candidate,
        samples_per_candidate=plan.samples_per_candidate,
    )
    return CudaEventAutotuneResultV2(
        artifact_path=str(loaded.artifact_path),
        artifact_sha256=loaded.artifact_sha256,
        manifest_sha256=loaded.manifest_sha256,
        device_identity=dict(loaded.device_identity),
        initial_vram=initial_vram,
        reserve_bytes=reserve_bytes,
        plan=plan,
        resolved_batch_candidates=resolved_batches,
        maximum_vram_admitted_batch_size=maximum_vram_batch,
        measurements=tuple(measurements),
        selected=selected,
        harness_sha256=harness_sha256,
        performance_restructure_contract_sha256=(
            performance_restructure_contract_sha256
        ),
        validation_matrix_sha256=validation_matrix_sha256,
        topology_identity=topology_identity,
        target_verdict=_target_verdict_v2(
            selected,
            topology_identity,
            b136_comparison=b136_comparison,
            temporal_k_candidates=plan.temporal_k_candidates,
            measurements=measurements,
            b136_source_binding=b136_source_binding,
            resolved_envelope=resolved_envelope,
            staged_envelope=staged_envelope,
            execution_mode_comparison=execution_mode_comparison,
            c512_saturation_evidence=c512_saturation_evidence,
            artifact_sha256=loaded.artifact_sha256,
            manifest_sha256=loaded.manifest_sha256,
            device_identity=dict(loaded.device_identity),
            initial_vram=initial_vram,
            reserve_bytes=reserve_bytes,
            dispatches_per_candidate=plan.dispatches_per_candidate,
            samples_per_candidate=plan.samples_per_candidate,
            seed=plan.seed,
        ),
        b136_comparison=b136_comparison,
        staged_envelope=staged_envelope,
        execution_mode_comparison=execution_mode_comparison,
        c512_saturation_evidence=c512_saturation_evidence,
        launch_origin=launch_origin,
    )


def _c512_deterministic_projection_v2(
    value: CudaC512SaturationEvidenceV2,
) -> dict[str, Any]:
    """Project replay-stable C512 facts; wall/event durations are freshly attested."""
    return {
        "candidate": asdict(value.candidate),
        "execution_mode": value.execution_mode,
        "artifact_sha256": value.artifact_sha256,
        "manifest_sha256": value.manifest_sha256,
        "device_identity": value.device_identity,
        "seed": value.seed,
        "population_fixture_id": value.population_fixture_id,
        "fixture_id": value.fixture_id,
        "starting_checkpoint_metadata_sha256": (
            value.starting_checkpoint_metadata_sha256
        ),
        "starting_state_sha256": value.starting_state_sha256,
        "starting_counters": value.starting_counters,
        "final_counters": value.final_counters,
        "starting_population": asdict(value.starting_population),
        "final_population": asdict(value.final_population),
        "committed_world_ticks": value.sample.committed_world_ticks,
        "event_content_sha256": value.sample.event_content_sha256,
        "final_state_sha256": value.sample.final_state_sha256,
    }


def _authoritative_measurement_projection_v2(
    value: CudaAutotuneMeasurementV2,
    *,
    artifact_sha256: str,
    manifest_sha256: str,
    device_identity: dict[str, Any],
) -> dict[str, Any]:
    """Serialize only a freshly replayed measurement as authoritative evidence."""
    body = {
        "schema_version": "eiel_cuda_event_autotune_authoritative_measurement_v2",
        "authority_class": "AUTHORITATIVE_FRESH_LIVE_REPLAY",
        "authoritative": True,
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "device_identity": device_identity,
        "candidate": asdict(value.candidate),
        "execution_mode": value.execution_mode,
        "status": value.status,
        "capacity": asdict(value.capacity),
        "samples": [asdict(sample) for sample in value.samples],
        "committed_world_ticks": value.committed_world_ticks,
        "cuda_event_elapsed_ms": value.cuda_event_elapsed_ms,
        "committed_world_ticks_per_second": (value.committed_world_ticks_per_second),
        "median_sample_world_ticks_per_second": (
            value.median_sample_world_ticks_per_second
        ),
        "world_ticks_per_second_percentiles": (
            value.sample_percentiles_world_ticks_per_second()
        ),
        "selected_statistic_p10_world_ticks_per_second": (
            value.p10_sample_world_ticks_per_second
        ),
        "peak_allocated_bytes": value.peak_allocated_bytes,
        "peak_reserved_bytes": value.peak_reserved_bytes,
        "peak_device_process_bytes": value.peak_device_process_bytes,
        "allocated_after_initialization_bytes": (
            value.allocated_after_initialization_bytes
        ),
        "reserved_after_initialization_bytes": (
            value.reserved_after_initialization_bytes
        ),
        "device_process_after_initialization_bytes": (
            value.device_process_after_initialization_bytes
        ),
        "initial_population": (
            None
            if value.initial_population is None
            else asdict(value.initial_population)
        ),
        "initial_population_projection_seconds": (
            value.initial_population_projection_seconds
        ),
        "final_population": (
            None if value.final_population is None else asdict(value.final_population)
        ),
        "fixture": None if value.fixture is None else asdict(value.fixture),
        "execution_evidence": [
            asdict(evidence) for evidence in value.execution_evidence
        ],
    }
    return {
        **body,
        "measurement_sha256": _canonical_sha256_v2(body),
    }


def _authoritative_c512_projection_v2(
    value: CudaC512SaturationEvidenceV2,
) -> dict[str, Any]:
    """Serialize the independent fresh C512 replay, never its claimant copy."""
    body = {
        "schema_version": "eiel_cuda_event_autotune_authoritative_c512_v2",
        "authority_class": "AUTHORITATIVE_FRESH_LIVE_REPLAY",
        "authoritative": True,
        **asdict(value),
    }
    return {
        **body,
        "authoritative_projection_sha256": _canonical_sha256_v2(body),
    }


def _recomputed_b136_comparison_v2(
    result: CudaEventAutotuneResultV2,
    *,
    device_identity: dict[str, Any],
    initial_vram: dict[str, int],
    reserve_bytes: int,
) -> dict[str, Any]:
    """Rebuild the B136 receipt from the validated source and full sweep."""
    per_k_maximum = {
        temporal_k: maximum_vram_admitted_batch_size_v2(
            free_vram_bytes=initial_vram["free_bytes"],
            reserve_bytes=reserve_bytes,
            temporal_k=temporal_k,
        )
        for temporal_k in result.plan.temporal_k_candidates
    }
    capacities = {
        temporal_k: capacity_for_candidate_v2(
            CudaAutotuneCandidateV2(batch_size=136, temporal_k=temporal_k),
            free_vram_bytes=initial_vram["free_bytes"],
            total_vram_bytes=initial_vram["total_bytes"],
            reserve_bytes=reserve_bytes,
            dispatches_per_candidate=result.plan.dispatches_per_candidate,
        )
        for temporal_k in result.plan.temporal_k_candidates
    }
    return _b136_comparison_v2(
        per_k_maximum_vram_admitted_batch_size=per_k_maximum,
        resolved_batch_candidates=result.resolved_batch_candidates,
        measurements=result.measurements,
        capacities=capacities,
        device_identity=device_identity,
        initial_vram=initial_vram,
        reserve_bytes=reserve_bytes,
        dispatches_per_candidate=result.plan.dispatches_per_candidate,
    )


def _authoritative_live_replay_v2(
    result: CudaEventAutotuneResultV2,
    *,
    torch: Any,
    validation_vram: dict[str, int],
) -> dict[str, Any]:
    """Freshly replay the selected B/K modes and C512 fixture on the live GPU."""
    validation_vram = _canonical_vram_observation_v2(
        validation_vram, name="validation-time VRAM observation"
    )
    selected = result.selected
    originals = {
        value.execution_mode: value
        for value in result.measurements
        if value.candidate == selected.candidate and value.status == "VALID"
    }
    if set(originals) != set(EXECUTION_MODES_V2):
        raise CudaFullEvolutionAutotuneError(
            "authoritative replay requires both valid selected B/K modes"
        )
    current_capacity = capacity_for_candidate_v2(
        selected.candidate,
        free_vram_bytes=validation_vram["free_bytes"],
        total_vram_bytes=validation_vram["total_bytes"],
        reserve_bytes=result.reserve_bytes,
        dispatches_per_candidate=result.plan.dispatches_per_candidate,
    )
    if not current_capacity.admitted:
        raise CudaFullEvolutionAutotuneError(
            "current validation VRAM cannot admit the selected B/K replay"
        )
    replays = tuple(
        measure_candidate_v2(
            selected.candidate,
            plan=result.plan,
            torch=torch,
            free_vram_bytes=validation_vram["free_bytes"],
            total_vram_bytes=validation_vram["total_bytes"],
            reserve_bytes=result.reserve_bytes,
            execution_mode=mode,
        )
        for mode in EXECUTION_MODES_V2
    )
    if any(value.status != "VALID" for value in replays):
        raise CudaFullEvolutionAutotuneError(
            "authoritative selected B/K replay did not remain valid"
        )
    replay_by_mode = {value.execution_mode: value for value in replays}
    for mode in EXECUTION_MODES_V2:
        replay_eligible, replay_failures = _measurement_semantics_closure_v2(
            replay_by_mode[mode],
            expected_candidate=selected.candidate,
            expected_execution_mode=mode,
            expected_stage_index=None,
            artifact_sha256=result.artifact_sha256,
            manifest_sha256=result.manifest_sha256,
            device_identity=result.device_identity,
            seed=result.plan.seed,
            free_vram_bytes=validation_vram["free_bytes"],
            total_vram_bytes=validation_vram["total_bytes"],
            reserve_bytes=result.reserve_bytes,
            dispatches_per_candidate=result.plan.dispatches_per_candidate,
            samples_per_candidate=result.plan.samples_per_candidate,
        )
        if not replay_eligible:
            raise CudaFullEvolutionAutotuneError(
                f"authoritative {mode} measurement is invalid: "
                + ",".join(replay_failures)
            )
        if _fixture_equivalence_projection_v2(
            originals[mode]
        ) != _fixture_equivalence_projection_v2(replay_by_mode[mode]):
            raise CudaFullEvolutionAutotuneError(
                f"authoritative {mode} fixture counters or digests changed"
            )
    replay_winner = select_best_measurement_v2(replays)
    replay_p10 = replay_winner.p10_sample_world_ticks_per_second
    if (
        replay_winner.execution_mode != selected.execution_mode
        or replay_p10 is None
        or replay_p10 < float(CUDA_NATIVE_TARGET_WTPS_100X)
    ):
        raise CudaFullEvolutionAutotuneError(
            "authoritative live p10 or execution-mode selection did not close"
        )
    replay_c512 = measure_c512_saturation_v2(
        replay_winner,
        plan=result.plan,
        torch=torch,
        artifact_sha256=result.artifact_sha256,
        manifest_sha256=result.manifest_sha256,
        device_identity=result.device_identity,
    )
    if result.c512_saturation_evidence is None or _c512_deterministic_projection_v2(
        result.c512_saturation_evidence
    ) != _c512_deterministic_projection_v2(replay_c512):
        raise CudaFullEvolutionAutotuneError(
            "authoritative C512 fixture counters or digests changed"
        )
    c512_eligible, c512_failures = _c512_saturation_closure_v2(
        replay_c512,
        replay_winner,
        artifact_sha256=result.artifact_sha256,
        manifest_sha256=result.manifest_sha256,
        device_identity=result.device_identity,
        seed=result.plan.seed,
    )
    if not c512_eligible:
        raise CudaFullEvolutionAutotuneError(
            "authoritative C512 replay is invalid: " + ",".join(c512_failures)
        )
    authoritative_measurements = [
        _authoritative_measurement_projection_v2(
            replay_by_mode[mode],
            artifact_sha256=result.artifact_sha256,
            manifest_sha256=result.manifest_sha256,
            device_identity=result.device_identity,
        )
        for mode in EXECUTION_MODES_V2
    ]
    authoritative_by_mode = {
        value["execution_mode"]: value for value in authoritative_measurements
    }
    authoritative_selected = authoritative_by_mode[replay_winner.execution_mode]
    graph_topology = (
        replay_by_mode[CUDA_GRAPH_EXECUTION_MODE_V2].execution_evidence[0].topology
    )
    peak_allocation = {
        "authority_class": "AUTHORITATIVE_FRESH_LIVE_REPLAY",
        "authoritative": True,
        "maximum_allocated_bytes": max(
            int(value.peak_allocated_bytes) for value in replays
        ),
        "maximum_reserved_bytes": max(
            int(value.peak_reserved_bytes) for value in replays
        ),
        "maximum_device_process_bytes": max(
            int(value.peak_device_process_bytes) for value in replays
        ),
    }
    body = {
        "schema_version": "eiel_cuda_event_autotune_live_replay_v2",
        "authority_class": "AUTHORITATIVE_FRESH_LIVE_REPLAY",
        "authoritative": True,
        "candidate": asdict(selected.candidate),
        "selected_execution_mode": replay_winner.execution_mode,
        "selected_p10_world_ticks_per_second": replay_p10,
        "p10_by_execution_mode": {
            mode: replay_by_mode[mode].p10_sample_world_ticks_per_second
            for mode in EXECUTION_MODES_V2
        },
        "selected_fixture_sha256": _canonical_sha256_v2(
            _fixture_equivalence_projection_v2(replay_winner)
        ),
        "c512_deterministic_projection_sha256": _canonical_sha256_v2(
            _c512_deterministic_projection_v2(replay_c512)
        ),
        "c512_cuda_event_elapsed_ms": replay_c512.sample.cuda_event_elapsed_ms,
        "measurements": authoritative_measurements,
        "selected": authoritative_selected,
        "peak_allocation": peak_allocation,
        "graph_topology": graph_topology,
        "graph_topology_sha256": _canonical_sha256_v2(graph_topology),
        "c512_saturation_evidence": _authoritative_c512_projection_v2(replay_c512),
        "status": "LIVE_REPLAY_EXACT_AND_P10_CLOSED",
    }
    return {**body, "replay_sha256": _canonical_sha256_v2(body)}


def validate_cuda_event_autotune_result_v2(
    result: CudaEventAutotuneResultV2,
    *,
    gpu_validation_requested: bool = True,
) -> dict[str, Any]:
    """Reopen live inputs and, with a physical GPU, grant the only full PASS."""
    if type(result) is not CudaEventAutotuneResultV2:
        raise CudaFullEvolutionAutotuneError(
            "authoritative validation requires the exact V2 result dataclass"
        )
    if type(gpu_validation_requested) is not bool:
        raise CudaFullEvolutionAutotuneError(
            "gpu_validation_requested must be an exact boolean"
        )
    try:
        artifact, _, manifest = validate_prebuilt_cuda_full_evolution_v2(
            result.artifact_path
        )
    except (CudaFullEvolutionNativeLoadError, OSError, ValueError) as exc:
        raise CudaFullEvolutionAutotuneError(
            f"cannot reopen source-bound artifact and manifest: {exc}"
        ) from exc
    actual_artifact_sha256 = file_sha256(artifact)
    actual_manifest_sha256 = manifest.get("manifest_sha256")
    if (
        Path(result.artifact_path).resolve() != artifact
        or result.plan.artifact_path.resolve() != artifact
        or result.artifact_sha256 != actual_artifact_sha256
        or result.artifact_sha256 != manifest.get("output", {}).get("sha256")
        or result.manifest_sha256 != actual_manifest_sha256
        or result.schema_version != BENCHMARK_SCHEMA_V2
        or result.claim_class != BENCHMARK_CLAIM_CLASS_V2
        or result.harness_sha256 != file_sha256(Path(__file__))
        or result.performance_restructure_contract_sha256
        != file_sha256(PERFORMANCE_RESTRUCTURE_CONTRACT_V2)
        or result.validation_matrix_sha256 != file_sha256(VALIDATION_MATRIX_V2)
    ):
        raise CudaFullEvolutionAutotuneError(
            "result source, harness, or contract binding does not match live files"
        )

    launch_origin_payload, launch_origin_reference = _validated_launch_origin_v2(
        result,
        artifact_path=artifact,
        artifact_sha256=actual_artifact_sha256,
        manifest_sha256=actual_manifest_sha256,
    )
    live_device_identity = result.device_identity
    # The launch-time VRAM snapshot is part of the sealed candidate envelope.
    # Free/allocated/reserved bytes are observations, not stable device
    # identity: this validator's own module load and CUDA allocator activity can
    # legitimately change them before the final evidence write.  Reconstruct
    # the measured envelope from its original snapshot and record a separate
    # validation-time observation for physical replay admission.
    envelope_vram = _canonical_vram_observation_v2(
        launch_origin_payload["initial_vram"], name="launch envelope VRAM observation"
    )
    validation_vram = envelope_vram
    if gpu_validation_requested:
        # The authoritative path intentionally has no caller-supplied torch or
        # device adapter: both loader and identity/VRAM queries must reach the
        # current physical CUDA runtime.
        torch = _require_torch_v2()
        try:
            loaded = load_prebuilt_cuda_full_evolution_v2(
                artifact, device=result.plan.device
            )
            device_index = loaded.device_identity.get("index")
            if type(device_index) is not int:
                raise CudaFullEvolutionAutotuneError(
                    "loaded physical device index is invalid"
                )
            queried_identity = cuda_device_identity_v2(torch, device_index)
            queried_vram = _canonical_vram_observation_v2(
                _live_vram_v2(torch, result.plan.device),
                name="validation-time VRAM observation",
            )
        except (CudaFullEvolutionNativeLoadError, OSError, ValueError) as exc:
            raise CudaFullEvolutionAutotuneError(
                f"cannot revalidate the current physical CUDA target: {exc}"
            ) from exc
        live_device_identity = dict(loaded.device_identity)
        validation_vram = queried_vram
        if (
            queried_identity != live_device_identity
            or result.device_identity != live_device_identity
            or envelope_vram["total_bytes"] != queried_vram["total_bytes"]
        ):
            raise CudaFullEvolutionAutotuneError(
                "result device identity or total VRAM does not match the current GPU"
            )
    if not isinstance(live_device_identity, dict):
        raise CudaFullEvolutionAutotuneError(
            "result device or VRAM source binding is invalid"
        )
    expected_reserve = _effective_reserve_bytes_v2(
        result.plan, envelope_vram.get("total_bytes")
    )
    per_k_maximum = {
        temporal_k: maximum_vram_admitted_batch_size_v2(
            free_vram_bytes=envelope_vram.get("free_bytes"),
            reserve_bytes=expected_reserve,
            temporal_k=temporal_k,
        )
        for temporal_k in result.plan.temporal_k_candidates
    }
    required_batches, expected_per_k_maximum = dynamic_batch_candidates_for_k_grid_v2(
        sm_count=live_device_identity.get("multiprocessor_count"),
        free_vram_bytes=envelope_vram.get("free_bytes"),
        reserve_bytes=expected_reserve,
        temporal_k_candidates=result.plan.temporal_k_candidates,
    )
    if expected_per_k_maximum != per_k_maximum:
        raise CudaFullEvolutionAutotuneError(
            "live per-K capacity reconstruction is inconsistent"
        )
    if result.plan.batch_candidates is None:
        expected_batches = required_batches
        plan_mode = "DYNAMIC_FULL_ENVELOPE"
    else:
        _validate_explicit_batch_candidates_v2(
            result.plan.batch_candidates,
            per_k_maximum_vram_admitted_batch_size=per_k_maximum,
            required_batch_candidates=required_batches,
        )
        expected_batches = result.plan.batch_candidates
        plan_mode = "EXPLICIT_REQUIRED_SUPERSET"
    expected_resolved_envelope = resolved_envelope_v2(
        artifact_sha256=actual_artifact_sha256,
        manifest_sha256=actual_manifest_sha256,
        device_identity=live_device_identity,
        initial_vram=envelope_vram,
        reserve_bytes=expected_reserve,
        temporal_k_candidates=result.plan.temporal_k_candidates,
        resolved_batch_candidates=expected_batches,
        plan_mode=plan_mode,
    )
    if (
        result.reserve_bytes != expected_reserve
        or result.resolved_batch_candidates != expected_batches
        or result.maximum_vram_admitted_batch_size != max(per_k_maximum.values())
        or result.staged_envelope.get("resolved_envelope") != expected_resolved_envelope
    ):
        raise CudaFullEvolutionAutotuneError(
            "result envelope or capacity does not match the current live source"
        )
    expected_b136_source = _b136_source_binding_v2(
        device_identity=live_device_identity,
        initial_vram=envelope_vram,
        reserve_bytes=expected_reserve,
        dispatches_per_candidate=result.plan.dispatches_per_candidate,
    )
    expected_b136_comparison = _recomputed_b136_comparison_v2(
        result,
        device_identity=live_device_identity,
        initial_vram=envelope_vram,
        reserve_bytes=expected_reserve,
    )
    if result.b136_comparison != expected_b136_comparison:
        raise CudaFullEvolutionAutotuneError(
            "B136 comparison is not the exact source-validated original sweep receipt"
        )
    preliminary = _target_verdict_v2(
        result.selected,
        result.topology_identity,
        b136_comparison=result.b136_comparison,
        temporal_k_candidates=result.plan.temporal_k_candidates,
        measurements=result.measurements,
        b136_source_binding=expected_b136_source,
        resolved_envelope=expected_resolved_envelope,
        staged_envelope=result.staged_envelope,
        execution_mode_comparison=result.execution_mode_comparison,
        c512_saturation_evidence=result.c512_saturation_evidence,
        artifact_sha256=actual_artifact_sha256,
        manifest_sha256=actual_manifest_sha256,
        device_identity=live_device_identity,
        initial_vram=envelope_vram,
        reserve_bytes=expected_reserve,
        dispatches_per_candidate=result.plan.dispatches_per_candidate,
        samples_per_candidate=result.plan.samples_per_candidate,
        seed=result.plan.seed,
    )
    if result.target_verdict != preliminary:
        raise CudaFullEvolutionAutotuneError(
            "stored preliminary verdict does not match authoritative recomputation"
        )
    if preliminary["status"] != "TARGET_CLOSURE_PASS_PRELIMINARY_NON_AUTHORITATIVE":
        raise CudaFullEvolutionAutotuneError(
            "result is not eligible for authoritative full-target validation"
        )
    if not gpu_validation_requested:
        return {
            "schema_version": (
                "eiel_cuda_event_autotune_source_validated_preliminary_v2"
            ),
            "authority_class": "PRELIMINARY_CLAIMANT_NON_AUTHORITATIVE",
            "status": "SOURCE_VALIDATED_PRELIMINARY_NON_AUTHORITATIVE",
            "authoritative": False,
            "technical_only": True,
            "preliminary_verdict": preliminary,
            "preliminary_closure_sha256": _canonical_sha256_v2(preliminary),
        }
    live_replay = _authoritative_live_replay_v2(
        result, torch=torch, validation_vram=validation_vram
    )
    authority_body = {
        "schema_version": "eiel_cuda_event_autotune_authoritative_validation_v2",
        "artifact_path": str(artifact),
        "artifact_sha256": actual_artifact_sha256,
        "manifest_sha256": actual_manifest_sha256,
        "device_identity": live_device_identity,
        "initial_vram": envelope_vram,
        "validation_vram": validation_vram,
        "launch_origin": launch_origin_reference,
        "reserve_bytes": expected_reserve,
        "harness_sha256": file_sha256(Path(__file__)),
        "performance_restructure_contract_sha256": file_sha256(
            PERFORMANCE_RESTRUCTURE_CONTRACT_V2
        ),
        "validation_matrix_sha256": file_sha256(VALIDATION_MATRIX_V2),
        "preliminary_closure_sha256": _canonical_sha256_v2(preliminary),
        "live_replay": live_replay,
    }
    preliminary_reference = {
        "authority_class": "PRELIMINARY_CLAIMANT_NON_AUTHORITATIVE",
        "authoritative": False,
        "receipt_sha256": _canonical_sha256_v2(result.to_dict()),
        "rows_included_in_authoritative_projection": False,
        "omitted_claimant_fields": [
            "measurements",
            "selected",
            "peak_allocation",
            "execution_mode_comparison",
            "c512_saturation_evidence",
        ],
        "non_authoritative_reporting_sections": [
            "b136_comparison",
            "staged_envelope",
        ],
    }
    verdict_body = {
        "schema_version": "eiel_cuda_event_autotune_authoritative_target_verdict_v2",
        "authority_class": "AUTHORITATIVE_FRESH_LIVE_REPLAY",
        "target_world_ticks_per_second": float(CUDA_NATIVE_TARGET_WTPS_100X),
        "selection_statistic": "p10_committed_world_ticks_per_second",
        "selected_p10_world_ticks_per_second": live_replay[
            "selected_p10_world_ticks_per_second"
        ],
        "selected_execution_mode": live_replay["selected_execution_mode"],
        "status": "FULL_TARGET_PASS",
        "closure_status": "AUTHORITATIVE_LIVE_SOURCE_AND_GPU_REVALIDATED",
        "closure_failures": [],
        "authoritative": True,
        "authority_limitation": None,
        "technical_only": True,
        "authoritative_validation": {
            **authority_body,
            "validation_sha256": _canonical_sha256_v2(authority_body),
        },
        "preliminary_claimant_receipt": preliminary_reference,
    }
    return {
        **verdict_body,
        "target_verdict_sha256": _canonical_sha256_v2(verdict_body),
    }


def _original_sweep_measurement_summary_v2(
    value: CudaAutotuneMeasurementV2,
) -> dict[str, Any]:
    """Bounded audit summary for one source-validated original sweep row."""
    body = {
        "authority_class": SOURCE_VALIDATED_ORIGINAL_SWEEP_AUTHORITY_V2,
        "authoritative": False,
        "candidate": asdict(value.candidate),
        "execution_mode": value.execution_mode,
        "status": value.status,
        "stage_index": value.stage_index,
        "sample_count": len(value.samples),
        "committed_world_ticks": value.committed_world_ticks,
        "cuda_event_elapsed_ms": value.cuda_event_elapsed_ms,
        "world_ticks_per_second_percentiles": (
            value.sample_percentiles_world_ticks_per_second()
        ),
        "p10_sample_world_ticks_per_second": (value.p10_sample_world_ticks_per_second),
        "peak_allocated_bytes": value.peak_allocated_bytes,
        "peak_reserved_bytes": value.peak_reserved_bytes,
        "peak_device_process_bytes": value.peak_device_process_bytes,
        "fixture_id": None if value.fixture is None else value.fixture.fixture_id,
        "original_measurement_dataclass_sha256": _canonical_sha256_v2(asdict(value)),
    }
    return {
        **body,
        "reporting_row_sha256": _canonical_sha256_v2(body),
    }


def _source_validated_original_sweep_reporting_v2(
    result: CudaEventAutotuneResultV2,
    *,
    validation: dict[str, Any],
    live_replay: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project the complete original sweep for audit without granting authority."""
    exact_b136 = _recomputed_b136_comparison_v2(
        result,
        device_identity=validation["device_identity"],
        initial_vram=validation["initial_vram"],
        reserve_bytes=validation["reserve_bytes"],
    )
    if result.b136_comparison != exact_b136:
        raise CudaFullEvolutionAutotuneError(
            "original B136 sweep receipt changed after source validation"
        )
    plan_mode = (
        "DYNAMIC_FULL_ENVELOPE"
        if result.plan.batch_candidates is None
        else "EXPLICIT_REQUIRED_SUPERSET"
    )
    exact_resolved_envelope = resolved_envelope_v2(
        artifact_sha256=validation["artifact_sha256"],
        manifest_sha256=validation["manifest_sha256"],
        device_identity=validation["device_identity"],
        initial_vram=validation["initial_vram"],
        reserve_bytes=validation["reserve_bytes"],
        temporal_k_candidates=result.plan.temporal_k_candidates,
        resolved_batch_candidates=result.resolved_batch_candidates,
        plan_mode=plan_mode,
    )
    exact_stages = candidate_stages_for_plan_v2(
        result.plan,
        resolved_batch_candidates=result.resolved_batch_candidates,
    )
    exact_staged = staged_envelope_receipt_v2(
        artifact_sha256=validation["artifact_sha256"],
        manifest_sha256=validation["manifest_sha256"],
        device_identity=validation["device_identity"],
        temporal_k_candidates=result.plan.temporal_k_candidates,
        stages=exact_stages,
        measurements=result.measurements,
        resolved_envelope=exact_resolved_envelope,
        initial_vram=validation["initial_vram"],
        reserve_bytes=validation["reserve_bytes"],
        dispatches_per_candidate=result.plan.dispatches_per_candidate,
        samples_per_candidate=result.plan.samples_per_candidate,
    )
    if result.staged_envelope != exact_staged:
        raise CudaFullEvolutionAutotuneError(
            "original staged sweep receipt changed after source validation"
        )
    fresh_selected = live_replay["selected"]
    original_selected = _original_sweep_measurement_summary_v2(result.selected)
    if (
        fresh_selected.get("candidate") != asdict(result.selected.candidate)
        or fresh_selected.get("execution_mode") != result.selected.execution_mode
    ):
        raise CudaFullEvolutionAutotuneError(
            "original sweep winner does not cross-link to the fresh replay winner"
        )
    cross_link_body = {
        "authority_class": SOURCE_VALIDATED_ORIGINAL_SWEEP_AUTHORITY_V2,
        "authoritative": False,
        "original_sweep_selected": original_selected,
        "fresh_replay_candidate": fresh_selected["candidate"],
        "fresh_replay_execution_mode": fresh_selected["execution_mode"],
        "fresh_replay_selected_p10_world_ticks_per_second": live_replay[
            "selected_p10_world_ticks_per_second"
        ],
        "fresh_replay_measurement_sha256": fresh_selected["measurement_sha256"],
        "fresh_replay_sha256": live_replay["replay_sha256"],
        "candidate_and_execution_mode_exact_match": True,
    }
    selection_cross_link = {
        **cross_link_body,
        "selection_cross_link_sha256": _canonical_sha256_v2(cross_link_body),
    }
    stage_sha_by_candidate: dict[tuple[int, int], str] = {}
    for stage in exact_staged["stages"]:
        for candidate in stage["candidates"]:
            stage_sha_by_candidate[
                (candidate["batch_size"], candidate["temporal_k"])
            ] = stage["stage_sha256"]
    b136_performance: list[dict[str, Any]] = []
    mode_order = {mode: index for index, mode in enumerate(EXECUTION_MODES_V2)}
    for receipt_row in exact_b136["per_k"]:
        temporal_k = receipt_row["temporal_k"]
        candidate = CudaAutotuneCandidateV2(batch_size=136, temporal_k=temporal_k)
        mode_measurements = sorted(
            (value for value in result.measurements if value.candidate == candidate),
            key=lambda value: mode_order[value.execution_mode],
        )
        valid_measurements = [
            value for value in mode_measurements if value.status == "VALID"
        ]
        original_winner = (
            None
            if not valid_measurements
            else select_best_measurement_v2(valid_measurements)
        )
        performance_body = {
            "authority_class": SOURCE_VALIDATED_ORIGINAL_SWEEP_AUTHORITY_V2,
            "authoritative": False,
            "candidate": asdict(candidate),
            "b136_receipt_status": receipt_row["status"],
            "staged_stage_sha256": stage_sha_by_candidate.get(
                (candidate.batch_size, candidate.temporal_k)
            ),
            "measurements": [
                _original_sweep_measurement_summary_v2(value)
                for value in mode_measurements
            ],
            "original_sweep_selected_execution_mode": (
                None if original_winner is None else original_winner.execution_mode
            ),
            "original_sweep_selected_p10_world_ticks_per_second": (
                None
                if original_winner is None
                else original_winner.p10_sample_world_ticks_per_second
            ),
            "fresh_replay_winner_measurement_sha256": fresh_selected[
                "measurement_sha256"
            ],
        }
        b136_performance.append(
            {
                **performance_body,
                "performance_row_sha256": _canonical_sha256_v2(performance_body),
            }
        )
    b136_body = {
        **exact_b136,
        "authority_class": SOURCE_VALIDATED_ORIGINAL_SWEEP_AUTHORITY_V2,
        "authoritative": False,
        "contained_rows_authority_class": (
            SOURCE_VALIDATED_ORIGINAL_SWEEP_AUTHORITY_V2
        ),
        "contained_rows_authoritative": False,
        "source_validation_sha256": validation["validation_sha256"],
        "original_receipt_sha256": _canonical_sha256_v2(exact_b136),
        "selection_cross_link": selection_cross_link,
        "per_k_performance": b136_performance,
    }
    b136_reporting = {
        **b136_body,
        "source_validated_reporting_sha256": _canonical_sha256_v2(b136_body),
    }
    staged_body = {
        **exact_staged,
        "authority_class": SOURCE_VALIDATED_ORIGINAL_SWEEP_AUTHORITY_V2,
        "authoritative": False,
        "contained_rows_authority_class": (
            SOURCE_VALIDATED_ORIGINAL_SWEEP_AUTHORITY_V2
        ),
        "contained_rows_authoritative": False,
        "source_validation_sha256": validation["validation_sha256"],
        "original_receipt_sha256": _canonical_sha256_v2(exact_staged),
        "selection_cross_link": selection_cross_link,
    }
    staged_reporting = {
        **staged_body,
        "source_validated_reporting_sha256": _canonical_sha256_v2(staged_body),
    }
    return b136_reporting, staged_reporting


def _authoritative_result_projection_v2(
    result: CudaEventAutotuneResultV2,
    target_verdict: dict[str, Any],
) -> dict[str, Any]:
    """Build the writer's result solely from sealed fresh live-replay evidence."""
    if not isinstance(target_verdict, dict):
        raise CudaFullEvolutionAutotuneError(
            "authoritative validator did not return a verdict object"
        )
    verdict_body = {
        key: value
        for key, value in target_verdict.items()
        if key != "target_verdict_sha256"
    }
    if (
        target_verdict.get("schema_version")
        != "eiel_cuda_event_autotune_authoritative_target_verdict_v2"
        or target_verdict.get("status") != "FULL_TARGET_PASS"
        or target_verdict.get("authoritative") is not True
        or target_verdict.get("authority_class") != "AUTHORITATIVE_FRESH_LIVE_REPLAY"
        or target_verdict.get("target_verdict_sha256")
        != _canonical_sha256_v2(verdict_body)
    ):
        raise CudaFullEvolutionAutotuneError(
            "authoritative target verdict seal or status is invalid"
        )
    validation = target_verdict.get("authoritative_validation")
    if not isinstance(validation, dict):
        raise CudaFullEvolutionAutotuneError("authoritative validation body is missing")
    validation_body = {
        key: value for key, value in validation.items() if key != "validation_sha256"
    }
    live_replay = validation.get("live_replay")
    validation_initial_vram = _canonical_vram_observation_v2(
        validation.get("initial_vram"), name="authoritative launch VRAM observation"
    )
    validation_current_vram = _canonical_vram_observation_v2(
        validation.get("validation_vram"),
        name="authoritative validation-time VRAM observation",
    )
    if (
        validation.get("schema_version")
        != "eiel_cuda_event_autotune_authoritative_validation_v2"
        or validation.get("validation_sha256") != _canonical_sha256_v2(validation_body)
        or not isinstance(live_replay, dict)
        or validation_initial_vram
        != _canonical_vram_observation_v2(
            result.initial_vram, name="result launch VRAM observation"
        )
        or validation_current_vram["total_bytes"]
        != validation_initial_vram["total_bytes"]
        or validation.get("launch_origin") != result.launch_origin
    ):
        raise CudaFullEvolutionAutotuneError(
            "authoritative live-source validation seal is invalid"
        )
    replay_body = {
        key: value for key, value in live_replay.items() if key != "replay_sha256"
    }
    selected_p10 = live_replay.get("selected_p10_world_ticks_per_second")
    selected_mode = live_replay.get("selected_execution_mode")
    measurements = live_replay.get("measurements")
    selected = live_replay.get("selected")
    peak_allocation = live_replay.get("peak_allocation")
    c512 = live_replay.get("c512_saturation_evidence")
    preliminary_reference = target_verdict.get("preliminary_claimant_receipt")
    if (
        live_replay.get("status") != "LIVE_REPLAY_EXACT_AND_P10_CLOSED"
        or live_replay.get("authoritative") is not True
        or live_replay.get("authority_class") != "AUTHORITATIVE_FRESH_LIVE_REPLAY"
        or live_replay.get("replay_sha256") != _canonical_sha256_v2(replay_body)
        or not isinstance(selected_p10, (int, float))
        or not math.isfinite(float(selected_p10))
        or float(selected_p10) < float(CUDA_NATIVE_TARGET_WTPS_100X)
        or target_verdict.get("selected_p10_world_ticks_per_second") != selected_p10
        or target_verdict.get("selected_execution_mode") != selected_mode
        or selected_mode not in EXECUTION_MODES_V2
        or not isinstance(measurements, list)
        or len(measurements) != len(EXECUTION_MODES_V2)
        or not isinstance(selected, dict)
        or not isinstance(peak_allocation, dict)
        or not isinstance(c512, dict)
        or not isinstance(preliminary_reference, dict)
        or preliminary_reference.get("authoritative") is not False
        or preliminary_reference.get("rows_included_in_authoritative_projection")
        is not False
        or preliminary_reference.get("receipt_sha256")
        != _canonical_sha256_v2(result.to_dict())
    ):
        raise CudaFullEvolutionAutotuneError(
            "authoritative replay projection is incomplete or inconsistent"
        )
    measurement_modes: set[str] = set()
    for measurement in measurements:
        if not isinstance(measurement, dict):
            raise CudaFullEvolutionAutotuneError(
                "authoritative replay measurement is not an object"
            )
        measurement_body = {
            key: value
            for key, value in measurement.items()
            if key != "measurement_sha256"
        }
        mode = measurement.get("execution_mode")
        if (
            mode not in EXECUTION_MODES_V2
            or measurement.get("authoritative") is not True
            or measurement.get("authority_class") != "AUTHORITATIVE_FRESH_LIVE_REPLAY"
            or measurement.get("measurement_sha256")
            != _canonical_sha256_v2(measurement_body)
            or measurement.get("artifact_sha256") != validation.get("artifact_sha256")
            or measurement.get("manifest_sha256") != validation.get("manifest_sha256")
            or measurement.get("device_identity") != validation.get("device_identity")
        ):
            raise CudaFullEvolutionAutotuneError(
                "authoritative replay measurement seal or source binding is invalid"
            )
        measurement_modes.add(mode)
    selected_rows = [
        measurement
        for measurement in measurements
        if measurement.get("execution_mode") == selected_mode
    ]
    c512_body = {
        key: value
        for key, value in c512.items()
        if key != "authoritative_projection_sha256"
    }
    if (
        measurement_modes != set(EXECUTION_MODES_V2)
        or selected_rows != [selected]
        or selected.get("selected_statistic_p10_world_ticks_per_second") != selected_p10
        or peak_allocation.get("authoritative") is not True
        or peak_allocation.get("authority_class") != "AUTHORITATIVE_FRESH_LIVE_REPLAY"
        or c512.get("authoritative") is not True
        or c512.get("authority_class") != "AUTHORITATIVE_FRESH_LIVE_REPLAY"
        or c512.get("authoritative_projection_sha256")
        != _canonical_sha256_v2(c512_body)
        or c512.get("artifact_sha256") != validation.get("artifact_sha256")
        or c512.get("manifest_sha256") != validation.get("manifest_sha256")
        or c512.get("device_identity") != validation.get("device_identity")
        or c512.get("execution_mode") != selected_mode
        or c512.get("candidate") != selected.get("candidate")
    ):
        raise CudaFullEvolutionAutotuneError(
            "authoritative selected, peak, or C512 projection is inconsistent"
        )
    b136_reporting, staged_reporting = _source_validated_original_sweep_reporting_v2(
        result,
        validation=validation,
        live_replay=live_replay,
    )
    result_body = {
        "schema_version": "eiel_cuda_event_autotune_authoritative_result_v2",
        "claim_class": BENCHMARK_CLAIM_CLASS_V2,
        "authority_class": (
            "AUTHORITATIVE_FRESH_LIVE_REPLAY_WITH_NON_AUTHORITATIVE_ORIGINAL_SWEEP"
        ),
        "authoritative": True,
        "status": "FULL_TARGET_PASS",
        "route_id": CUDA_NATIVE_ROUTE_V2,
        "target_world_ticks_per_second": float(CUDA_NATIVE_TARGET_WTPS_100X),
        "selected_p10_world_ticks_per_second": selected_p10,
        "artifact": {
            "path": validation["artifact_path"],
            "sha256": validation["artifact_sha256"],
            "manifest_sha256": validation["manifest_sha256"],
        },
        "device_identity": validation["device_identity"],
        "initial_vram": validation["initial_vram"],
        "validation_vram": validation["validation_vram"],
        "launch_origin": validation["launch_origin"],
        "reserve_bytes": validation["reserve_bytes"],
        "plan": {
            "authority_class": "VALIDATED_FRESH_LIVE_REPLAY_INPUT",
            "candidate": live_replay["candidate"],
            "selected_execution_mode": selected_mode,
            "temporal_k_candidates": list(result.plan.temporal_k_candidates),
            "warmup_dispatches": result.plan.warmup_dispatches,
            "samples_per_candidate": result.plan.samples_per_candidate,
            "seed": result.plan.seed,
            "device": result.plan.device,
        },
        "measurements": measurements,
        "selected": selected,
        "peak_allocation": peak_allocation,
        "topology_identity": live_replay["graph_topology"],
        "execution_mode_comparison": {
            "authority_class": "AUTHORITATIVE_FRESH_LIVE_REPLAY",
            "authoritative": True,
            "selection_statistic": "p10_committed_world_ticks_per_second",
            "selected_execution_mode": selected_mode,
            "p10_by_execution_mode": live_replay["p10_by_execution_mode"],
            "graph_topology": live_replay["graph_topology"],
            "graph_topology_sha256": live_replay["graph_topology_sha256"],
        },
        "c512_saturation_evidence": c512,
        "b136_comparison": b136_reporting,
        "staged_envelope": staged_reporting,
        "target_verdict": target_verdict,
        "preliminary_claimant_receipt": preliminary_reference,
    }
    return {
        **result_body,
        "result_sha256": _canonical_sha256_v2(result_body),
    }


def _validated_autotune_output_path_v2(output_path: Path) -> Path:
    if not isinstance(output_path, Path):
        raise CudaFullEvolutionAutotuneError("output_path must be a Path")
    output = output_path.resolve()
    artifact_root = AUTOTUNE_ARTIFACT_ROOT_V2.resolve()
    if artifact_root not in output.parents:
        raise CudaFullEvolutionAutotuneError(
            f"output must be beneath the V2 artifact fence: {artifact_root}"
        )
    if output.exists():
        raise CudaFullEvolutionAutotuneError(
            "autotune output must not overwrite evidence"
        )
    return output


def _launch_origin_path_for_output_v2(output_path: Path) -> Path:
    output = _validated_autotune_output_path_v2(output_path)
    if output.suffix == ".json":
        return output.with_name(f"{output.stem}.launch.json")
    return output.with_name(f"{output.name}.launch.json")


def write_cuda_event_autotune_result_v2(
    result: CudaEventAutotuneResultV2, output_path: Path
) -> Path:
    output = _validated_autotune_output_path_v2(output_path)
    if result.launch_origin:
        expected_launch_path = _launch_origin_path_for_output_v2(output)
        if result.launch_origin.get("path") != str(expected_launch_path):
            raise CudaFullEvolutionAutotuneError(
                "launch origin is not bound to the exact output namespace"
            )
    payload = result.to_dict()
    if result.target_verdict.get("status") in {
        "FULL_TARGET_PASS",
        "TARGET_CLOSURE_PASS_PRELIMINARY_NON_AUTHORITATIVE",
    }:
        target_verdict = validate_cuda_event_autotune_result_v2(result)
        payload = _authoritative_result_projection_v2(result, target_verdict)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("ascii")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    file_descriptor: int | None = None
    temporary_created = False
    output_linked = False
    created_identity: tuple[int, int] | None = None
    linked_output_identity: tuple[int, int] | None = None
    try:
        file_descriptor = os.open(temporary, flags, stat.S_IRUSR)
        temporary_created = True
        created = os.fstat(file_descriptor)
        created_identity = (created.st_dev, created.st_ino)
        if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
            raise CudaFullEvolutionAutotuneError(
                "result temporary is not an exclusive single-link regular file"
            )
        _write_all_v2(file_descriptor, encoded)
        os.fchmod(file_descriptor, stat.S_IRUSR)
        os.fsync(file_descriptor)
        sealed = os.fstat(file_descriptor)
        try:
            temporary_stat = os.lstat(temporary)
        except OSError as exc:
            raise CudaFullEvolutionAutotuneError(
                f"result temporary disappeared before publication: {exc}"
            ) from exc
        if (
            (sealed.st_dev, sealed.st_ino) != created_identity
            or (temporary_stat.st_dev, temporary_stat.st_ino) != created_identity
            or not stat.S_ISREG(sealed.st_mode)
            or not stat.S_ISREG(temporary_stat.st_mode)
            or sealed.st_nlink != 1
            or temporary_stat.st_nlink != 1
            or stat.S_IMODE(sealed.st_mode) != stat.S_IRUSR
            or stat.S_IMODE(temporary_stat.st_mode) != stat.S_IRUSR
            or sealed.st_size != len(encoded)
            or temporary_stat.st_size != len(encoded)
            or _read_exact_descriptor_v2(file_descriptor, len(encoded)) != encoded
        ):
            raise CudaFullEvolutionAutotuneError(
                "result temporary changed before no-replace publication"
            )
        try:
            os.link(temporary, output, follow_symlinks=False)
            output_linked = True
        except FileExistsError as exc:
            raise CudaFullEvolutionAutotuneError(
                "autotune output must not overwrite evidence"
            ) from exc
        try:
            published_stat = os.lstat(output)
            temporary_stat = os.lstat(temporary)
        except OSError as exc:
            raise CudaFullEvolutionAutotuneError(
                f"result publication path changed during no-replace link: {exc}"
            ) from exc
        if (
            (published_stat.st_dev, published_stat.st_ino)
            == (temporary_stat.st_dev, temporary_stat.st_ino)
            and published_stat.st_nlink >= 2
            and temporary_stat.st_nlink >= 2
        ):
            linked_output_identity = (published_stat.st_dev, published_stat.st_ino)
        linked = os.fstat(file_descriptor)
        if (
            (linked.st_dev, linked.st_ino) != created_identity
            or (temporary_stat.st_dev, temporary_stat.st_ino) != created_identity
            or (published_stat.st_dev, published_stat.st_ino) != created_identity
            or not stat.S_ISREG(linked.st_mode)
            or not stat.S_ISREG(temporary_stat.st_mode)
            or not stat.S_ISREG(published_stat.st_mode)
            or linked.st_nlink != 2
            or temporary_stat.st_nlink != 2
            or published_stat.st_nlink != 2
            or stat.S_IMODE(linked.st_mode) != stat.S_IRUSR
            or stat.S_IMODE(temporary_stat.st_mode) != stat.S_IRUSR
            or stat.S_IMODE(published_stat.st_mode) != stat.S_IRUSR
            or linked.st_size != len(encoded)
            or temporary_stat.st_size != len(encoded)
            or published_stat.st_size != len(encoded)
            or _read_exact_descriptor_v2(file_descriptor, len(encoded)) != encoded
        ):
            raise CudaFullEvolutionAutotuneError(
                "result publication did not preserve the exact created inode and content"
            )
        _fsync_directory_v2(output.parent)
        if not _unlink_exact_inode_v2(temporary, created_identity):
            raise CudaFullEvolutionAutotuneError(
                "result temporary changed before publication closeout"
            )
        temporary_created = False
        _fsync_directory_v2(output.parent)
        final_stat = os.lstat(output)
        final_inode = os.fstat(file_descriptor)
        if (
            (final_inode.st_dev, final_inode.st_ino) != created_identity
            or (final_stat.st_dev, final_stat.st_ino) != created_identity
            or not stat.S_ISREG(final_inode.st_mode)
            or not stat.S_ISREG(final_stat.st_mode)
            or final_inode.st_nlink != 1
            or final_stat.st_nlink != 1
            or stat.S_IMODE(final_inode.st_mode) != stat.S_IRUSR
            or stat.S_IMODE(final_stat.st_mode) != stat.S_IRUSR
            or final_inode.st_size != len(encoded)
            or final_stat.st_size != len(encoded)
            or _read_exact_descriptor_v2(file_descriptor, len(encoded)) != encoded
        ):
            raise CudaFullEvolutionAutotuneError(
                "result publication changed during final inode verification"
            )
        os.close(file_descriptor)
        file_descriptor = None
    except Exception:
        if output_linked:
            for identity in (created_identity, linked_output_identity):
                if identity is not None and _unlink_exact_inode_v2(output, identity):
                    break
        if temporary_created and created_identity is not None:
            _unlink_exact_inode_v2(temporary, created_identity)
        raise
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
    return output


def _parse_candidates(value: str, *, name: str) -> tuple[int, ...]:
    parts = tuple(part.strip() for part in value.split(","))
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            f"{name} must be a nonempty comma-separated list"
        )
    try:
        parsed = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be comma-separated integers"
        ) from exc
    try:
        return _strict_candidate_tuple(
            parsed, name=name, maximum=64 if name == "temporal-ks" else None
        )
    except CudaFullEvolutionAutotuneError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batches",
        type=lambda value: _parse_candidates(value, name="batches"),
        default=DEFAULT_BATCH_CANDIDATES_V2,
        help="optional exact B list; omit for live SM/VRAM-derived candidates",
    )
    parser.add_argument(
        "--temporal-ks",
        type=lambda value: _parse_candidates(value, name="temporal-ks"),
        default=DEFAULT_TEMPORAL_K_CANDIDATES_V2,
    )
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUP_DISPATCHES_V2)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_PER_CANDIDATE_V2)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reserve-bytes", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = _validated_autotune_output_path_v2(args.output)
    launch_origin_path = _launch_origin_path_for_output_v2(output)
    if launch_origin_path.exists() or launch_origin_path.is_symlink():
        raise CudaFullEvolutionAutotuneError(
            f"launch origin must not overwrite evidence: {launch_origin_path}"
        )
    plan = CudaEventAutotunePlanV2(
        artifact_path=args.artifact,
        batch_candidates=args.batches,
        temporal_k_candidates=args.temporal_ks,
        warmup_dispatches=args.warmups,
        samples_per_candidate=args.samples,
        seed=args.seed,
        device=args.device,
        reserve_bytes=args.reserve_bytes,
    )
    result = run_cuda_event_autotune_v2(plan, launch_origin_path=launch_origin_path)
    output = write_cuda_event_autotune_result_v2(result, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
