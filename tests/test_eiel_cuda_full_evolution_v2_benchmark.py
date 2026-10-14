from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace
import weakref

import pytest

from engine.perf.cuda_full_evolution_v2 import benchmark

ARTIFACT_SHA = "a" * 64
MANIFEST_SHA = "b" * 64
HARNESS_SHA = "c" * 64
CONTRACT_SHA = "d" * 64
MATRIX_SHA = "e" * 64
SEED = 71


def _device(*, uuid: str = "GPU-test", sm_count: int = 1) -> dict[str, object]:
    return {
        "index": 0,
        "name": "Test GPU",
        "uuid": uuid,
        "pci_domain_id": 0,
        "pci_bus_id": 1,
        "pci_device_id": 0,
        "total_memory_bytes": 16 << 20,
        "multiprocessor_count": sm_count,
        "sm": "sm_90",
        "cuda_runtime": "13.0",
    }


def _vram(
    *,
    free_bytes: int = 16 << 20,
    total_bytes: int = 16 << 20,
    allocated_bytes: int = 0,
    reserved_bytes: int = 0,
) -> dict[str, int]:
    return {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "allocated_bytes": allocated_bytes,
        "reserved_bytes": reserved_bytes,
    }


class _FakeEvent:
    def __init__(self, log: list[str], ordinal: int) -> None:
        self.log, self.ordinal = log, ordinal

    def record(self) -> None:
        self.log.append(f"event:{self.ordinal}:record")

    def synchronize(self) -> None:
        self.log.append(f"event:{self.ordinal}:synchronize")

    def elapsed_time(self, other: object) -> float:
        assert isinstance(other, _FakeEvent)
        self.log.append(f"event:{self.ordinal}:elapsed")
        return 2.5


class _FakeCuda:
    def __init__(self, log: list[str], *, peak: int = 303) -> None:
        self.log, self.events, self.peak = log, 0, peak

    def Event(self, *, enable_timing: bool) -> _FakeEvent:
        assert enable_timing
        event = _FakeEvent(self.log, self.events)
        self.events += 1
        return event

    @contextmanager
    def device(self, device: str):
        self.log.append(f"device:{device}:enter")
        try:
            yield
        finally:
            self.log.append(f"device:{device}:exit")

    def synchronize(self, device: str) -> None:
        self.log.append(f"cuda:synchronize:{device}")

    def reset_peak_memory_stats(self, device: str) -> None:
        self.log.append(f"cuda:reset_peak:{device}")

    def memory_allocated(self, device: str) -> int:
        return 101

    def memory_reserved(self, device: str) -> int:
        return 202

    def mem_get_info(self, device: str) -> tuple[int, int]:
        return (16 << 20, 16 << 20)

    def max_memory_allocated(self, device: str) -> int:
        return self.peak

    def max_memory_reserved(self, device: str) -> int:
        return self.peak + 1

    def device_memory_used(self, device: str) -> int:
        return self.peak + 2

    def empty_cache(self) -> None:
        self.log.append("cuda:empty_cache")


class _FakeTorch:
    def __init__(self, log: list[str], *, peak: int = 303) -> None:
        self.cuda = _FakeCuda(log, peak=peak)


class _FakeOwner:
    def __init__(
        self,
        total_ticks: int,
        batch_size: int,
        log: list[str],
        *,
        no_causal: bool = False,
        execution_mode: str = benchmark.EAGER_EXECUTION_MODE_V2,
    ) -> None:
        self.total_ticks, self.batch_size, self.log = total_ticks, batch_size, log
        self.no_causal, self.execution_mode = no_causal, execution_mode
        self.pending = False
        self.last_ticks = 0
        self.advanced_ticks = 0

    @property
    def has_pending_dispatch(self) -> bool:
        return self.pending

    @property
    def remaining_ticks(self) -> int:
        return self.total_ticks - self.advanced_ticks

    @property
    def active_counts(self) -> tuple[int, ...]:
        return (256,) * self.batch_size

    def force_test_conditions(self) -> None:
        self.log.append("owner:force_fixture")

    def checkpoint(self) -> SimpleNamespace:
        return SimpleNamespace(
            canonical_state_sha256=("1" if self.advanced_ticks else "0") * 64,
            metadata_seal_sha256="a" * 64,
            total_ticks=self.total_ticks,
            batch_size=self.batch_size,
            no_causal=self.no_causal,
        )

    def counters(self) -> list[dict[str, int]]:
        delta = 0 if self.no_causal else self.advanced_ticks
        return [
            {name: delta for name in benchmark.COUNTER_NAMES_V2}
            for _ in range(self.batch_size)
        ]

    def advance(self, ticks: int) -> None:
        assert not self.pending
        self.log.append(f"owner:advance:{ticks}")
        self.pending, self.last_ticks = True, ticks

    def drain_events(self, *, to_cpu: bool) -> SimpleNamespace:
        assert to_cpu and self.pending
        self.log.append("owner:drain")
        return SimpleNamespace(
            tick_count=self.last_ticks,
            drain_elapsed_seconds=0.0125,
            content_hash_elapsed_seconds=0.00375,
            content_sha256="e" * 64,
        )

    def acknowledge_event_drain(self, receipt: object) -> None:
        assert receipt is not None and self.pending
        self.log.append("owner:acknowledge")
        self.pending = False
        self.advanced_ticks += self.last_ticks

    def execution_plan_receipt_v2(self) -> dict[str, object]:
        return {
            "schema_version": benchmark.EXECUTION_PLAN_RECEIPT_SCHEMA_V2,
            "topology": {"execution_mode": self.execution_mode},
            "prepared_dispatch_temporal_k": [self.total_ticks],
            "last_execution": {"execution_mode": self.execution_mode},
            "closed": False,
        }


def _population(
    candidate: benchmark.CudaAutotuneCandidateV2, active: int = 256
) -> benchmark.CudaActivePopulationSummaryV2:
    return benchmark.CudaActivePopulationSummaryV2(
        candidate.batch_size,
        candidate.batch_size * active,
        active,
        active,
    )


def _sample(
    candidate: benchmark.CudaAutotuneCandidateV2,
    rate: float,
    *,
    active: int = 256,
    event_digest: str = "e" * 64,
    state_digest: str = "f" * 64,
) -> benchmark.CudaEventTimingSampleV2:
    committed = candidate.batch_size * candidate.temporal_k
    population = _population(candidate, active)
    return benchmark.CudaEventTimingSampleV2(
        committed * 1000.0 / rate,
        committed,
        0.1,
        0.05,
        0.01,
        0.02,
        population,
        population,
        0.001,
        0.001,
        event_digest,
        state_digest,
    )


def _fixture(
    candidate: benchmark.CudaAutotuneCandidateV2,
    samples: tuple[benchmark.CudaEventTimingSampleV2, ...],
) -> benchmark.CudaCausalFixtureEvidenceV2:
    population = _population(candidate)
    starting = {name: 0 for name in benchmark.COUNTER_NAMES_V2}
    final = {name: 1 for name in benchmark.COUNTER_NAMES_V2}
    return benchmark.CudaCausalFixtureEvidenceV2(
        benchmark._fixture_receipt_identity_v2(
            candidate=candidate,
            seed=SEED,
            checkpoint_metadata_sha256="a" * 64,
            checkpoint_state_sha256="0" * 64,
        ),
        "a" * 64,
        "0" * 64,
        starting,
        final,
        population,
        population,
        tuple(sample.event_content_sha256 for sample in samples),
        tuple(sample.final_state_sha256 for sample in samples),
        True,
    )


def _execution_evidence(
    candidate: benchmark.CudaAutotuneCandidateV2,
    mode: str,
    sample: benchmark.CudaEventTimingSampleV2,
    *,
    device: dict[str, object] | None = None,
    artifact_sha: str = ARTIFACT_SHA,
    manifest_sha: str = MANIFEST_SHA,
) -> benchmark.CudaExecutionModeEvidenceV2:
    device = _device() if device is None else device
    graph = mode == benchmark.CUDA_GRAPH_EXECUTION_MODE_V2
    pointers = {"state": 11, "events": 12, "event_counts": 13}
    nodes = [
        {
            "name": benchmark.CUDA_GRAPH_KERNEL_V2,
            "node_type": "kernel",
            "runtime_graph_node": graph,
            "function_pointer_verified": True,
        }
    ]
    topology = {
        "schema_version": benchmark.EXECUTION_PLAN_TOPOLOGY_SCHEMA_V2,
        "execution_mode": mode,
        "artifact_sha256": artifact_sha,
        "manifest_sha256": manifest_sha,
        "device_identity": device,
        "device_index": device["index"],
        "B": candidate.batch_size,
        "dispatch_temporal_k": candidate.temporal_k,
        "event_capacity_k": candidate.temporal_k,
        "capability_mask": benchmark.FULL_CAPABILITY_MASK_V2,
        "execution_kernel_count": 1,
        "graph_node_count": 1 if graph else 0,
        "node_types": ["kernel"] if graph else [],
        "dependency_edges": [],
        "kernel_nodes": nodes,
        "runtime_inspected": graph,
        "declaration_is_execution_evidence": False,
        "strong_tensor_references": True,
        "pointer_substitution_allowed": False,
        "construction_stream_handle": 9,
        "graph_handle": 21 if graph else 0,
        "graph_exec_handle": 22 if graph else 0,
        "pointer_bindings": pointers,
    }
    launch = {
        "schema_version": benchmark.NATIVE_EXECUTION_RECEIPT_SCHEMA_V2,
        "execution_mode": mode,
        "artifact_sha256": artifact_sha,
        "manifest_sha256": manifest_sha,
        "device_identity": device,
        "device_index": device["index"],
        "B": candidate.batch_size,
        "tick_count": candidate.temporal_k,
        "dispatch_temporal_k": candidate.temporal_k,
        "event_capacity_k": candidate.temporal_k,
        "start_next_absolute_tick": 0,
        "end_next_absolute_tick": candidate.temporal_k,
        "capability_mask": benchmark.FULL_CAPABILITY_MASK_V2,
        "enqueued_world_ticks": candidate.batch_size * candidate.temporal_k,
        "committed_world_ticks": sample.committed_world_ticks,
        "committed_next_absolute_tick": candidate.temporal_k,
        "kernel": benchmark.CUDA_GRAPH_KERNEL_V2,
        "graph_launched": graph,
        "topology_graph_node_count": 1 if graph else 0,
        "topology_execution_kernel_count": 1,
        "topology_node_types": ["kernel"] if graph else [],
        "topology_kernel_nodes": nodes,
        "topology_dependency_edges": [],
        "runtime_inspected": graph,
        "runtime_status": "committed",
        "event_status": "exact_content_sink_acknowledged",
        "event_content_sha256": sample.event_content_sha256,
        "per_node_timing": None,
        "inter_node_gap_timing": None,
        "timing_fabrication_allowed": False,
        "pointer_bindings": pointers,
        "plan_identity_sha256": "1" * 64,
        "pointer_binding_identity_sha256": "2" * 64,
        "capture_identity_sha256": "3" * 64,
        "sink_receipt_sha256": "4" * 64,
    }
    return benchmark.CudaExecutionModeEvidenceV2(
        mode,
        topology,
        launch,
        benchmark._canonical_sha256_v2(topology),
        benchmark._canonical_sha256_v2(launch),
    )


def _capacity(
    candidate: benchmark.CudaAutotuneCandidateV2,
    *,
    vram: dict[str, int] | None = None,
) -> benchmark.CudaAutotuneCapacityV2:
    vram = _vram() if vram is None else vram
    return benchmark.capacity_for_candidate_v2(
        candidate,
        free_vram_bytes=vram["free_bytes"],
        total_vram_bytes=vram["total_bytes"],
        reserve_bytes=0,
        dispatches_per_candidate=4,
    )


def _measurement(
    candidate: benchmark.CudaAutotuneCandidateV2,
    mode: str,
    *,
    rates: tuple[float, ...] = (600_000.0,) * 3,
    stage_index: int = 0,
    vram: dict[str, int] | None = None,
    device: dict[str, object] | None = None,
) -> benchmark.CudaAutotuneMeasurementV2:
    capacity = _capacity(candidate, vram=vram)
    if not capacity.admitted:
        return benchmark.CudaAutotuneMeasurementV2(
            candidate,
            "SKIPPED_CAPACITY",
            capacity,
            execution_mode=mode,
            stage_index=stage_index,
        )
    samples = tuple(_sample(candidate, rate) for rate in rates)
    population = _population(candidate)
    return benchmark.CudaAutotuneMeasurementV2(
        candidate,
        "VALID",
        capacity,
        execution_mode=mode,
        samples=samples,
        peak_allocated_bytes=100,
        peak_reserved_bytes=200,
        peak_device_process_bytes=300,
        allocated_after_initialization_bytes=10,
        reserved_after_initialization_bytes=20,
        device_process_after_initialization_bytes=30,
        initial_population=population,
        initial_population_projection_seconds=0.001,
        final_population=population,
        fixture=_fixture(candidate, samples),
        execution_evidence=tuple(
            _execution_evidence(candidate, mode, sample, device=device)
            for sample in samples
        ),
        stage_index=stage_index,
    )


def _c512(
    selected: benchmark.CudaAutotuneMeasurementV2,
    *,
    device: dict[str, object] | None = None,
) -> benchmark.CudaC512SaturationEvidenceV2:
    device = _device() if device is None else device
    candidate = selected.candidate
    sample = _sample(
        candidate,
        550_000.0,
        active=512,
        event_digest="9" * 64,
        state_digest="8" * 64,
    )
    fixture_id = benchmark._c512_fixture_id_v2(
        candidate=candidate,
        execution_mode=selected.execution_mode,
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=device,
        seed=SEED,
        checkpoint_metadata_sha256="6" * 64,
        checkpoint_state_sha256="7" * 64,
    )
    start = {name: 0 for name in benchmark.COUNTER_NAMES_V2}
    final = {name: 1 for name in benchmark.COUNTER_NAMES_V2}
    provisional = benchmark.CudaC512SaturationEvidenceV2(
        candidate,
        selected.execution_mode,
        ARTIFACT_SHA,
        MANIFEST_SHA,
        device,
        SEED,
        benchmark.FULL_CAPACITY_BENCHMARK_FIXTURE_V2,
        fixture_id,
        "6" * 64,
        "7" * 64,
        start,
        final,
        _population(candidate, 512),
        _population(candidate, 512),
        sample,
        _execution_evidence(candidate, selected.execution_mode, sample, device=device),
        "",
        "",
    )
    lined = replace(
        provisional,
        fixture_lineage_sha256=benchmark._canonical_sha256_v2(
            benchmark._c512_lineage_body_v2(provisional)
        ),
    )
    return replace(
        lined,
        evidence_sha256=benchmark._canonical_sha256_v2(
            benchmark._c512_evidence_body_v2(lined)
        ),
    )


def _reseal_c512(
    value: benchmark.CudaC512SaturationEvidenceV2,
) -> benchmark.CudaC512SaturationEvidenceV2:
    value = replace(value, fixture_lineage_sha256="", evidence_sha256="")
    value = replace(
        value,
        fixture_lineage_sha256=benchmark._canonical_sha256_v2(
            benchmark._c512_lineage_body_v2(value)
        ),
    )
    return replace(
        value,
        evidence_sha256=benchmark._canonical_sha256_v2(
            benchmark._c512_evidence_body_v2(value)
        ),
    )


def _complete(*, vram: dict[str, int] | None = None) -> dict[str, object]:
    device = _device()
    vram = _vram() if vram is None else vram
    plan = benchmark.CudaEventAutotunePlanV2(
        Path("/tmp/v2.so"), reserve_bytes=0, seed=SEED
    )
    batches, maxima = benchmark.dynamic_batch_candidates_for_k_grid_v2(
        sm_count=1,
        free_vram_bytes=vram["free_bytes"],
        reserve_bytes=0,
        temporal_k_candidates=plan.temporal_k_candidates,
    )
    resolved = benchmark.resolved_envelope_v2(
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=device,
        initial_vram=vram,
        reserve_bytes=0,
        temporal_k_candidates=plan.temporal_k_candidates,
        resolved_batch_candidates=batches,
        plan_mode="DYNAMIC_FULL_ENVELOPE",
    )
    stages = benchmark.candidate_stages_for_plan_v2(
        plan, resolved_batch_candidates=batches
    )
    measurements = []
    winner = benchmark.CudaAutotuneCandidateV2(1, 1)
    for stage in stages:
        for candidate in stage.candidates:
            for mode in benchmark.EXECUTION_MODES_V2:
                rates = (600_000.0,) * 3
                if candidate == winner:
                    rates = (
                        (1_000_000.0,) * 3
                        if mode == benchmark.EAGER_EXECUTION_MODE_V2
                        else (900_000.0,) * 3
                    )
                measurements.append(
                    _measurement(
                        candidate,
                        mode,
                        rates=rates,
                        stage_index=stage.stage_index,
                        vram=vram,
                        device=device,
                    )
                )
    selected = benchmark.select_best_measurement_v2(measurements)
    comparison = benchmark.execution_mode_comparison_v2(
        selected,
        measurements,
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=device,
    )
    c512 = _c512(selected, device=device)
    b136_capacities = {
        k: _capacity(benchmark.CudaAutotuneCandidateV2(136, k), vram=vram)
        for k in plan.temporal_k_candidates
    }
    b136 = benchmark._b136_comparison_v2(
        per_k_maximum_vram_admitted_batch_size=maxima,
        resolved_batch_candidates=batches,
        measurements=measurements,
        capacities=b136_capacities,
        device_identity=device,
        initial_vram=vram,
        reserve_bytes=0,
        dispatches_per_candidate=plan.dispatches_per_candidate,
    )
    staged = benchmark.staged_envelope_receipt_v2(
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=device,
        temporal_k_candidates=plan.temporal_k_candidates,
        stages=stages,
        measurements=measurements,
        resolved_envelope=resolved,
        initial_vram=vram,
        reserve_bytes=0,
        dispatches_per_candidate=plan.dispatches_per_candidate,
        samples_per_candidate=plan.samples_per_candidate,
    )
    context = {
        "b136_comparison": b136,
        "temporal_k_candidates": plan.temporal_k_candidates,
        "measurements": tuple(measurements),
        "b136_source_binding": benchmark._b136_source_binding_v2(
            device_identity=device,
            initial_vram=vram,
            reserve_bytes=0,
            dispatches_per_candidate=plan.dispatches_per_candidate,
        ),
        "resolved_envelope": resolved,
        "staged_envelope": staged,
        "execution_mode_comparison": comparison,
        "c512_saturation_evidence": c512,
        "artifact_sha256": ARTIFACT_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "device_identity": device,
        "initial_vram": vram,
        "reserve_bytes": 0,
        "dispatches_per_candidate": plan.dispatches_per_candidate,
        "samples_per_candidate": plan.samples_per_candidate,
        "seed": SEED,
    }
    verdict = benchmark._target_verdict_v2(
        selected, comparison["graph_topology"], **context
    )
    return locals()


def _with_preliminary_selected_rate(
    c: dict[str, object],
    rate: float,
    *,
    peak_bytes: int | None = None,
    c512_elapsed_ms: float | None = None,
) -> dict[str, object]:
    original = c["selected"]
    samples = tuple(
        replace(
            sample,
            cuda_event_elapsed_ms=sample.committed_world_ticks * 1000.0 / rate,
        )
        for sample in original.samples
    )
    selected = replace(
        original,
        samples=samples,
        peak_allocated_bytes=(
            original.peak_allocated_bytes if peak_bytes is None else peak_bytes
        ),
        peak_reserved_bytes=(
            original.peak_reserved_bytes if peak_bytes is None else peak_bytes
        ),
        peak_device_process_bytes=(
            original.peak_device_process_bytes if peak_bytes is None else peak_bytes
        ),
    )
    measurements = tuple(
        selected if value == original else value for value in c["measurements"]
    )
    comparison = benchmark.execution_mode_comparison_v2(
        selected,
        measurements,
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=c["device"],
    )
    staged = benchmark.staged_envelope_receipt_v2(
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=c["device"],
        temporal_k_candidates=c["plan"].temporal_k_candidates,
        stages=c["stages"],
        measurements=measurements,
        resolved_envelope=c["resolved"],
        initial_vram=c["vram"],
        reserve_bytes=0,
        dispatches_per_candidate=c["plan"].dispatches_per_candidate,
        samples_per_candidate=c["plan"].samples_per_candidate,
    )
    c512 = c["c512"]
    if c512_elapsed_ms is not None:
        c512 = _reseal_c512(
            replace(
                c512,
                sample=replace(
                    c512.sample,
                    cuda_event_elapsed_ms=c512_elapsed_ms,
                ),
            )
        )
    context = {
        **c["context"],
        "measurements": measurements,
        "staged_envelope": staged,
        "execution_mode_comparison": comparison,
        "c512_saturation_evidence": c512,
    }
    verdict = benchmark._target_verdict_v2(
        selected, comparison["graph_topology"], **context
    )
    return {
        **c,
        "measurements": measurements,
        "selected": selected,
        "comparison": comparison,
        "staged": staged,
        "c512": c512,
        "context": context,
        "verdict": verdict,
    }


def _result(
    c: dict[str, object],
    artifact: Path,
    *,
    launch_origin_path: Path | None = None,
    publish_launch_origin: bool = True,
) -> benchmark.CudaEventAutotuneResultV2:
    result = benchmark.CudaEventAutotuneResultV2(
        str(artifact.resolve()),
        ARTIFACT_SHA,
        MANIFEST_SHA,
        c["device"],
        c["vram"],
        0,
        replace(c["plan"], artifact_path=artifact),
        c["batches"],
        max(c["maxima"].values()),
        tuple(c["measurements"]),
        c["selected"],
        HARNESS_SHA,
        CONTRACT_SHA,
        MATRIX_SHA,
        c["comparison"]["graph_topology"],
        c["verdict"],
        c["b136"],
        c["staged"],
        c["comparison"],
        c["c512"],
    )
    if not publish_launch_origin:
        return result
    launch_origin_path = (
        artifact.with_suffix(".launch.json")
        if launch_origin_path is None
        else launch_origin_path
    )
    launch_origin = benchmark._publish_launch_origin_v2(
        launch_origin_path,
        artifact_path=artifact,
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=c["device"],
        initial_vram=c["vram"],
        plan=result.plan,
        harness_sha256=HARNESS_SHA,
        performance_restructure_contract_sha256=CONTRACT_SHA,
        validation_matrix_sha256=MATRIX_SHA,
    )
    return replace(result, launch_origin=launch_origin)


def _fake_live_replay(
    result: benchmark.CudaEventAutotuneResultV2,
    *,
    selected_p10: float = 500_000.0,
    peak_base: int = 401,
    c512_elapsed_ms: float = 0.125,
) -> dict[str, object]:
    originals = {
        value.execution_mode: value
        for value in result.measurements
        if value.candidate == result.selected.candidate and value.status == "VALID"
    }
    live_measurements = []
    for mode in benchmark.EXECUTION_MODES_V2:
        value = originals[mode]
        rate = selected_p10 if mode == result.selected.execution_mode else 490_000.0
        samples = tuple(
            replace(
                sample,
                cuda_event_elapsed_ms=(sample.committed_world_ticks * 1000.0 / rate),
            )
            for sample in value.samples
        )
        live_measurements.append(
            replace(
                value,
                samples=samples,
                peak_allocated_bytes=peak_base,
                peak_reserved_bytes=peak_base + 1,
                peak_device_process_bytes=peak_base + 2,
                stage_index=None,
            )
        )
    live_by_mode = {value.execution_mode: value for value in live_measurements}
    selected = live_by_mode[result.selected.execution_mode]
    projections = [
        benchmark._authoritative_measurement_projection_v2(
            live_by_mode[mode],
            artifact_sha256=result.artifact_sha256,
            manifest_sha256=result.manifest_sha256,
            device_identity=result.device_identity,
        )
        for mode in benchmark.EXECUTION_MODES_V2
    ]
    projection_by_mode = {value["execution_mode"]: value for value in projections}
    assert result.c512_saturation_evidence is not None
    live_c512 = _reseal_c512(
        replace(
            result.c512_saturation_evidence,
            sample=replace(
                result.c512_saturation_evidence.sample,
                cuda_event_elapsed_ms=c512_elapsed_ms,
            ),
        )
    )
    graph_topology = (
        live_by_mode[benchmark.CUDA_GRAPH_EXECUTION_MODE_V2]
        .execution_evidence[0]
        .topology
    )
    body = {
        "schema_version": "eiel_cuda_event_autotune_live_replay_v2",
        "authority_class": "AUTHORITATIVE_FRESH_LIVE_REPLAY",
        "authoritative": True,
        "candidate": {
            "batch_size": selected.candidate.batch_size,
            "temporal_k": selected.candidate.temporal_k,
        },
        "selected_execution_mode": selected.execution_mode,
        "selected_p10_world_ticks_per_second": (
            selected.p10_sample_world_ticks_per_second
        ),
        "p10_by_execution_mode": {
            mode: live_by_mode[mode].p10_sample_world_ticks_per_second
            for mode in benchmark.EXECUTION_MODES_V2
        },
        "selected_fixture_sha256": benchmark._canonical_sha256_v2(
            benchmark._fixture_equivalence_projection_v2(selected)
        ),
        "c512_deterministic_projection_sha256": benchmark._canonical_sha256_v2(
            benchmark._c512_deterministic_projection_v2(live_c512)
        ),
        "c512_cuda_event_elapsed_ms": live_c512.sample.cuda_event_elapsed_ms,
        "measurements": projections,
        "selected": projection_by_mode[selected.execution_mode],
        "peak_allocation": {
            "authority_class": "AUTHORITATIVE_FRESH_LIVE_REPLAY",
            "authoritative": True,
            "maximum_allocated_bytes": peak_base,
            "maximum_reserved_bytes": peak_base + 1,
            "maximum_device_process_bytes": peak_base + 2,
        },
        "graph_topology": graph_topology,
        "graph_topology_sha256": benchmark._canonical_sha256_v2(graph_topology),
        "c512_saturation_evidence": (
            benchmark._authoritative_c512_projection_v2(live_c512)
        ),
        "status": "LIVE_REPLAY_EXACT_AND_P10_CLOSED",
    }
    return {**body, "replay_sha256": benchmark._canonical_sha256_v2(body)}


def _sealed_authoritative_verdict(
    result: benchmark.CudaEventAutotuneResultV2,
    artifact: Path,
    live_replay: dict[str, object],
) -> dict[str, object]:
    authority_body = {
        "schema_version": "eiel_cuda_event_autotune_authoritative_validation_v2",
        "artifact_path": str(artifact.resolve()),
        "artifact_sha256": ARTIFACT_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "device_identity": result.device_identity,
        "initial_vram": result.initial_vram,
        "validation_vram": result.initial_vram,
        "launch_origin": result.launch_origin,
        "reserve_bytes": result.reserve_bytes,
        "harness_sha256": HARNESS_SHA,
        "performance_restructure_contract_sha256": CONTRACT_SHA,
        "validation_matrix_sha256": MATRIX_SHA,
        "preliminary_closure_sha256": benchmark._canonical_sha256_v2(
            result.target_verdict
        ),
        "live_replay": live_replay,
    }
    preliminary_reference = {
        "authority_class": "PRELIMINARY_CLAIMANT_NON_AUTHORITATIVE",
        "authoritative": False,
        "receipt_sha256": benchmark._canonical_sha256_v2(result.to_dict()),
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
        "target_world_ticks_per_second": float(benchmark.CUDA_NATIVE_TARGET_WTPS_100X),
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
            "validation_sha256": benchmark._canonical_sha256_v2(authority_body),
        },
        "preliminary_claimant_receipt": preliminary_reference,
    }
    return {
        **verdict_body,
        "target_verdict_sha256": benchmark._canonical_sha256_v2(verdict_body),
    }


def _authority_fakes(
    monkeypatch: pytest.MonkeyPatch,
    result: benchmark.CudaEventAutotuneResultV2,
    artifact: Path,
) -> None:
    manifest = {
        "manifest_sha256": MANIFEST_SHA,
        "output": {"sha256": ARTIFACT_SHA},
    }
    expected_fixture = result.selected.fixture
    expected_c512 = result.c512_saturation_evidence

    def validate(path: object):
        if Path(path).resolve() != artifact.resolve():
            raise benchmark.CudaFullEvolutionNativeLoadError("foreign artifact")
        return artifact.resolve(), artifact.with_suffix(".manifest.json"), manifest

    def digest(path: Path) -> str:
        resolved = Path(path).resolve()
        mapping = {
            artifact.resolve(): ARTIFACT_SHA,
            Path(benchmark.__file__).resolve(): HARNESS_SHA,
            benchmark.PERFORMANCE_RESTRUCTURE_CONTRACT_V2.resolve(): CONTRACT_SHA,
            benchmark.VALIDATION_MATRIX_V2.resolve(): MATRIX_SHA,
        }
        return mapping[resolved]

    monkeypatch.setattr(benchmark, "validate_prebuilt_cuda_full_evolution_v2", validate)
    monkeypatch.setattr(
        benchmark,
        "load_prebuilt_cuda_full_evolution_v2",
        lambda *args, **kwargs: SimpleNamespace(device_identity=result.device_identity),
    )
    monkeypatch.setattr(benchmark, "file_sha256", digest)
    monkeypatch.setattr(
        benchmark,
        "cuda_device_identity_v2",
        lambda torch, index: result.device_identity,
    )
    monkeypatch.setattr(
        benchmark, "_live_vram_v2", lambda torch, device: result.initial_vram
    )
    monkeypatch.setattr(benchmark, "_require_torch_v2", lambda: _FakeTorch([]))

    def replay(value, *, torch, validation_vram):
        assert validation_vram["total_bytes"] == result.initial_vram["total_bytes"]
        if (
            value.selected.fixture != expected_fixture
            or value.c512_saturation_evidence != expected_c512
        ):
            raise benchmark.CudaFullEvolutionAutotuneError(
                "live fixture replay mismatch"
            )
        return _fake_live_replay(value)

    monkeypatch.setattr(benchmark, "_authoritative_live_replay_v2", replay)


def test_plan_exact_grid_and_dual_mode_stage_bound() -> None:
    c = _complete()
    assert c["plan"].temporal_k_candidates == (1, 2, 4, 8, 16, 32, 64)
    assert benchmark.MAX_BATCH_CANDIDATES_PER_PLAN_V2 == 4
    assert all(
        len(stage.candidates) * len(benchmark.EXECUTION_MODES_V2) <= 64
        for stage in c["stages"]
    )
    assert c["staged"]["execution_modes"] == list(benchmark.EXECUTION_MODES_V2)


def test_capacity_skip_never_silently_downsizes() -> None:
    candidate = benchmark.CudaAutotuneCandidateV2(136, 64)
    capacity = _capacity(candidate)
    assert not capacity.admitted
    assert capacity.ledger.batch_size == 136
    assert 0 < capacity.admission.selected_batch_size < 136


def test_timed_dispatch_excludes_drain_and_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []
    owner = _FakeOwner(2, 1, log)
    monkeypatch.setattr(
        benchmark, "seal_event_sink_receipt_v2", lambda dispatch, **kwargs: object()
    )
    sample = benchmark._run_timed_dispatch_v2(
        owner,
        torch=_FakeTorch(log),
        ticks=2,
        batch_size=4,
        dispatch_index=0,
        expected_advanced_ticks=2,
    )
    assert sample.committed_world_ticks == 8
    assert log.index("event:1:synchronize") < log.index("owner:drain")
    assert log.index("owner:drain") < log.index("owner:acknowledge")


def test_measurement_restores_every_repetition(monkeypatch: pytest.MonkeyPatch) -> None:
    log, restored = [], []
    plan = benchmark.CudaEventAutotunePlanV2(
        Path("/tmp/v2.so"),
        batch_candidates=(3,),
        warmup_dispatches=1,
        samples_per_candidate=2,
    )

    def factory(contract, **kwargs):
        return _FakeOwner(
            contract.total_ticks,
            contract.B,
            log,
            execution_mode=kwargs["execution_mode"],
        )

    def restore(checkpoint, **kwargs):
        restored.append(checkpoint)
        return _FakeOwner(
            checkpoint.total_ticks,
            checkpoint.batch_size,
            log,
            no_causal=checkpoint.no_causal,
            execution_mode=kwargs["execution_mode"],
        )

    monkeypatch.setattr(
        benchmark, "seal_event_sink_receipt_v2", lambda dispatch, **kwargs: object()
    )
    measured = benchmark.measure_candidate_v2(
        benchmark.CudaAutotuneCandidateV2(3, 2),
        plan=plan,
        torch=_FakeTorch(log),
        free_vram_bytes=16 << 20,
        total_vram_bytes=16 << 20,
        reserve_bytes=0,
        owner_factory=factory,
        restore_owner=restore,
    )
    assert measured.status == "VALID"
    assert len(restored) == 3
    assert all(value is restored[0] for value in restored)
    assert len(measured.execution_evidence) == 2


def test_selection_and_threshold_use_p10_not_median() -> None:
    outlier = _measurement(
        benchmark.CudaAutotuneCandidateV2(1, 1),
        benchmark.EAGER_EXECUTION_MODE_V2,
        rates=(1.0, 1_000_000.0, 1_000_000.0),
    )
    stable = _measurement(
        benchmark.CudaAutotuneCandidateV2(2, 1),
        benchmark.EAGER_EXECUTION_MODE_V2,
        rates=(300_000.0,) * 3,
    )
    assert outlier.median_sample_world_ticks_per_second == 1_000_000.0
    assert outlier.p10_sample_world_ticks_per_second == pytest.approx(200_000.8)
    assert benchmark.select_best_measurement_v2((outlier, stable)) == stable
    verdict = benchmark._target_verdict_v2(outlier, {})
    assert verdict["status"] == "PERFORMANCE_THRESHOLD_FAIL"
    assert verdict["selected_p10_world_ticks_per_second"] == pytest.approx(200_000.8)


def test_selection_uses_faster_valid_mode_for_exact_candidate() -> None:
    candidate = benchmark.CudaAutotuneCandidateV2(1, 1)
    eager = replace(
        _measurement(
            candidate,
            benchmark.EAGER_EXECUTION_MODE_V2,
            rates=(999_500.0,) * 3,
        ),
        peak_allocated_bytes=1,
        peak_reserved_bytes=1,
        peak_device_process_bytes=1,
    )
    graph = replace(
        _measurement(
            candidate,
            benchmark.CUDA_GRAPH_EXECUTION_MODE_V2,
            rates=(1_000_000.0,) * 3,
        ),
        peak_allocated_bytes=100,
        peak_reserved_bytes=200,
        peak_device_process_bytes=300,
    )
    assert benchmark.select_best_measurement_v2((eager, graph)) == graph


def test_complete_dataclass_closure_is_non_authoritative() -> None:
    c = _complete()
    assert c["verdict"]["status"] == "TARGET_CLOSURE_PASS_PRELIMINARY_NON_AUTHORITATIVE"
    assert c["verdict"]["closure_failures"] == []
    assert c["verdict"]["authoritative"] is False
    receipt = _result(c, Path("/tmp/v2.so"), publish_launch_origin=False).to_dict()
    assert receipt["authority_class"] == "PRELIMINARY_CLAIMANT_NON_AUTHORITATIVE"
    assert receipt["authoritative"] is False
    assert all(value["authoritative"] is False for value in receipt["measurements"])
    assert receipt["b136_comparison"]["authoritative"] is False
    assert receipt["c512_saturation_evidence"]["authoritative"] is False


def test_target_recomputes_selected_winner_from_full_measurement_set() -> None:
    c = _complete()
    wrong = next(
        value
        for value in c["measurements"]
        if value.status == "VALID" and value != c["selected"]
    )
    verdict = benchmark._target_verdict_v2(
        wrong, c["comparison"]["graph_topology"], **c["context"]
    )
    assert verdict["status"] != "TARGET_CLOSURE_PASS_PRELIMINARY_NON_AUTHORITATIVE"
    assert (
        "selected_measurement_is_not_the_recomputed_winner"
        in verdict["closure_failures"]
    )


def test_real_monolith_topology_and_eager_winner() -> None:
    c = _complete()
    selected, comparison = c["selected"], c["comparison"]
    topology = comparison["graph_topology"]
    assert selected.execution_mode == benchmark.EAGER_EXECUTION_MODE_V2
    assert topology["graph_node_count"] == 1
    assert topology["dependency_edges"] == []
    assert [node["name"] for node in topology["kernel_nodes"]] == [
        "advance_optimized_kernel"
    ]
    eager_launch = selected.execution_evidence[0].launch_receipt
    assert eager_launch["graph_launched"] is False
    assert eager_launch["per_node_timing"] is None
    assert eager_launch["inter_node_gap_timing"] is None
    assert comparison["graph_equivalence_sentinel_retained"] is True


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        ("node_count", "topology"),
        ("edges", "topology"),
        ("handles", "graph_handles"),
        ("graph_flag", "launch_receipt"),
        ("artifact", "launch_receipt"),
        ("manifest", "source"),
        ("device", "source"),
        ("timing", "launch_receipt"),
    ],
)
def test_execution_receipt_rejects_forgery(mutation: str, fragment: str) -> None:
    c = _complete()
    selected = c["selected"]
    evidence = selected.execution_evidence[0]
    topology, launch = dict(evidence.topology), dict(evidence.launch_receipt)
    if mutation == "node_count":
        topology["graph_node_count"] = 2
    elif mutation == "edges":
        topology["dependency_edges"] = [[0, 1]]
    elif mutation == "handles":
        topology["graph_handle"] = 9
    elif mutation == "graph_flag":
        launch["graph_launched"] = True
    elif mutation == "artifact":
        launch["artifact_sha256"] = "9" * 64
    elif mutation == "manifest":
        topology["manifest_sha256"] = "9" * 64
    elif mutation == "device":
        topology["device_identity"] = _device(uuid="GPU-foreign")
    else:
        launch["per_node_timing"] = {"advance_optimized_kernel": 0.1}
    forged = replace(
        evidence,
        topology=topology,
        launch_receipt=launch,
        topology_sha256=benchmark._canonical_sha256_v2(topology),
        launch_receipt_sha256=benchmark._canonical_sha256_v2(launch),
    )
    selected = replace(
        selected,
        execution_evidence=(forged, *selected.execution_evidence[1:]),
    )
    eligible, failures = benchmark._execution_mode_evidence_closure_v2(
        selected,
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=c["device"],
    )
    assert not eligible
    assert any(fragment in failure for failure in failures)


def test_mode_comparison_requires_both_exact_modes() -> None:
    c = _complete()
    selected = c["selected"]
    incomplete = tuple(
        value
        for value in c["measurements"]
        if not (
            value.candidate == selected.candidate
            and value.execution_mode == benchmark.CUDA_GRAPH_EXECUTION_MODE_V2
        )
    )
    with pytest.raises(
        benchmark.CudaFullEvolutionAutotuneError, match="eager and graph"
    ):
        benchmark.execution_mode_comparison_v2(
            selected,
            incomplete,
            artifact_sha256=ARTIFACT_SHA,
            manifest_sha256=MANIFEST_SHA,
            device_identity=c["device"],
        )


def test_c512_is_distinct_saturation_measurement() -> None:
    c = _complete()
    selected, c512 = c["selected"], c["c512"]
    assert c512.starting_population.minimum_active_population == 512
    assert c512.fixture_id != selected.fixture.fixture_id
    assert c512.starting_state_sha256 != selected.fixture.starting_state_sha256
    assert benchmark._c512_saturation_closure_v2(
        c512,
        selected,
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=c["device"],
        seed=SEED,
    ) == (True, ())


def test_c512_rejects_copied_selected_timing() -> None:
    c = _complete()
    copied = _reseal_c512(replace(c["c512"], sample=c["selected"].samples[0]))
    eligible, failures = benchmark._c512_saturation_closure_v2(
        copied,
        c["selected"],
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=c["device"],
        seed=SEED,
    )
    assert not eligible
    assert "C512_timing_sample_is_not_exactly_fixture_bound" in failures


@pytest.mark.parametrize("field", ["artifact", "manifest", "device"])
def test_c512_rejects_foreign_outer_source(field: str) -> None:
    c = _complete()
    changes: dict[str, object]
    if field == "artifact":
        changes = {"artifact_sha256": "9" * 64}
    elif field == "manifest":
        changes = {"manifest_sha256": "9" * 64}
    else:
        changes = {"device_identity": _device(uuid="GPU-foreign")}
    forged = _reseal_c512(replace(c["c512"], **changes))
    eligible, failures = benchmark._c512_saturation_closure_v2(
        forged,
        c["selected"],
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=c["device"],
        seed=SEED,
    )
    assert not eligible
    assert "C512_source_binding_is_not_outer_exact" in failures


def test_c512_measurement_calls_dedicated_full_capacity_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _measurement(
        benchmark.CudaAutotuneCandidateV2(1, 1),
        benchmark.EAGER_EXECUTION_MODE_V2,
    )
    log: list[str] = []
    device = _device()

    class FullOwner(_FakeOwner):
        def __init__(self, *args, saturated: bool = False, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.saturated = saturated

        @property
        def active_counts(self) -> tuple[int, ...]:
            return ((512 if self.saturated else 256),) * self.batch_size

        def force_full_capacity_benchmark_fixture_v2(self):
            self.saturated = True
            log.append("owner:force_c512")
            return SimpleNamespace(
                metadata_seal_sha256="6" * 64,
                canonical_state_sha256="7" * 64,
                artifact_sha256=ARTIFACT_SHA,
                manifest_sha256=MANIFEST_SHA,
                device_identity=device,
                population_fixture_id=benchmark.FULL_CAPACITY_BENCHMARK_FIXTURE_V2,
                total_ticks=self.total_ticks,
                batch_size=self.batch_size,
            )

    def factory(contract, **kwargs):
        return FullOwner(
            contract.total_ticks,
            contract.B,
            log,
            execution_mode=kwargs["execution_mode"],
        )

    def restore(checkpoint, **kwargs):
        return FullOwner(
            checkpoint.total_ticks,
            checkpoint.batch_size,
            log,
            saturated=True,
            execution_mode=kwargs["execution_mode"],
        )

    monkeypatch.setattr(
        benchmark, "seal_event_sink_receipt_v2", lambda dispatch, **kwargs: object()
    )
    evidence = benchmark.measure_c512_saturation_v2(
        selected,
        plan=benchmark.CudaEventAutotunePlanV2(
            Path("/tmp/v2.so"), batch_candidates=(1,)
        ),
        torch=_FakeTorch(log),
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=device,
        owner_factory=factory,
        restore_owner=restore,
    )
    assert log.count("owner:force_c512") == 1
    assert evidence.starting_population.minimum_active_population == 512
    assert evidence.sample.committed_world_ticks == 1
    assert evidence.fixture_id != selected.fixture.fixture_id


def test_c512_releases_fixture_owner_before_allocating_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _measurement(
        benchmark.CudaAutotuneCandidateV2(1, 1),
        benchmark.EAGER_EXECUTION_MODE_V2,
    )
    log: list[str] = []
    device = _device()
    fixture_ref: weakref.ReferenceType[_FakeOwner] | None = None

    class FullOwner(_FakeOwner):
        def __init__(self, *args, saturated: bool = False, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.saturated = saturated

        @property
        def active_counts(self) -> tuple[int, ...]:
            return ((512 if self.saturated else 256),) * self.batch_size

        def force_full_capacity_benchmark_fixture_v2(self):
            self.saturated = True
            return SimpleNamespace(
                metadata_seal_sha256="6" * 64,
                canonical_state_sha256="7" * 64,
                artifact_sha256=ARTIFACT_SHA,
                manifest_sha256=MANIFEST_SHA,
                device_identity=device,
                population_fixture_id=benchmark.FULL_CAPACITY_BENCHMARK_FIXTURE_V2,
                total_ticks=self.total_ticks,
                batch_size=self.batch_size,
            )

    def factory(contract, **kwargs):
        nonlocal fixture_ref
        owner = FullOwner(
            contract.total_ticks,
            contract.B,
            log,
            execution_mode=kwargs["execution_mode"],
        )
        fixture_ref = weakref.ref(owner)
        return owner

    def restore(checkpoint, **kwargs):
        assert fixture_ref is not None and fixture_ref() is None
        return FullOwner(
            checkpoint.total_ticks,
            checkpoint.batch_size,
            log,
            saturated=True,
            execution_mode=kwargs["execution_mode"],
        )

    monkeypatch.setattr(
        benchmark, "seal_event_sink_receipt_v2", lambda dispatch, **kwargs: object()
    )
    evidence = benchmark.measure_c512_saturation_v2(
        selected,
        plan=benchmark.CudaEventAutotunePlanV2(
            Path("/tmp/v2.so"), batch_candidates=(1,)
        ),
        torch=_FakeTorch(log),
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=device,
        owner_factory=factory,
        restore_owner=restore,
    )
    assert evidence.sample.committed_world_ticks == 1


@pytest.mark.parametrize(
    "mutation",
    ["huge_ticks", "one_sample", "capacity", "false_skip", "wrong_status"],
)
def test_staged_envelope_rejects_semantic_forgery(mutation: str) -> None:
    c = _complete()
    measurements = list(c["measurements"])
    selected = c["selected"]
    index, forged = measurements.index(selected), selected
    if mutation == "huge_ticks":
        sample = replace(forged.samples[0], committed_world_ticks=10**12)
        forged = replace(forged, samples=(sample, *forged.samples[1:]))
    elif mutation == "one_sample":
        forged = replace(
            forged,
            samples=forged.samples[:1],
            execution_evidence=forged.execution_evidence[:1],
        )
    elif mutation == "capacity":
        forged = replace(
            forged,
            capacity=replace(
                forged.capacity,
                free_vram_bytes=2**50,
                total_vram_bytes=2**50,
                reserve_bytes=123,
            ),
        )
    elif mutation == "false_skip":
        forged = benchmark.CudaAutotuneMeasurementV2(
            forged.candidate,
            "SKIPPED_CAPACITY",
            forged.capacity,
            execution_mode=forged.execution_mode,
            stage_index=forged.stage_index,
        )
    else:
        forged = replace(forged, status="INVALID_MEASURED_PEAK")
    measurements[index] = forged
    staged = benchmark.staged_envelope_receipt_v2(
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=c["device"],
        temporal_k_candidates=c["plan"].temporal_k_candidates,
        stages=c["stages"],
        measurements=measurements,
        resolved_envelope=c["resolved"],
        initial_vram=c["vram"],
        reserve_bytes=0,
        dispatches_per_candidate=4,
        samples_per_candidate=3,
    )
    eligible, failures = benchmark._staged_envelope_closure_v2(
        staged,
        c["resolved"],
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=c["device"],
        initial_vram=c["vram"],
        reserve_bytes=0,
        dispatches_per_candidate=4,
        samples_per_candidate=3,
        seed=SEED,
        temporal_k_candidates=c["plan"].temporal_k_candidates,
        measurements=measurements,
    )
    assert not eligible
    assert failures


def test_b136_rejects_capacity_from_another_k() -> None:
    device = _device(sm_count=41)
    vram = _vram(free_bytes=128 << 20, total_bytes=1 << 36)
    ks = (1, 64)
    maxima = {
        k: benchmark.maximum_vram_admitted_batch_size_v2(
            free_vram_bytes=vram["free_bytes"], reserve_bytes=0, temporal_k=k
        )
        for k in ks
    }
    capacities = {
        k: _capacity(benchmark.CudaAutotuneCandidateV2(136, k), vram=vram) for k in ks
    }
    measurements = tuple(
        replace(
            _measurement(
                benchmark.CudaAutotuneCandidateV2(136, 1),
                mode,
                vram=vram,
                device=device,
            ),
            capacity=capacities[1],
        )
        for mode in benchmark.EXECUTION_MODES_V2
    )
    receipt = benchmark._b136_comparison_v2(
        per_k_maximum_vram_admitted_batch_size=maxima,
        resolved_batch_candidates=(1, 136),
        measurements=measurements,
        capacities=capacities,
        device_identity=device,
        initial_vram=vram,
        reserve_bytes=0,
        dispatches_per_candidate=4,
    )
    rows = [
        (
            {**row, "capacity": receipt["per_k"][0]["capacity"]}
            if row["temporal_k"] == 64
            else row
        )
        for row in receipt["per_k"]
    ]
    eligible, failures = benchmark._b136_closure_v2(
        {**receipt, "per_k": rows},
        temporal_k_candidates=ks,
        measurements=measurements,
        source_binding=benchmark._b136_source_binding_v2(
            device_identity=device,
            initial_vram=vram,
            reserve_bytes=0,
            dispatches_per_candidate=4,
        ),
    )
    assert not eligible
    assert any("B136_K64" in failure for failure in failures)


def test_b136_k1_rejects_b1_k1_capacity_ledger() -> None:
    c = _complete()
    source = c["context"]["b136_source_binding"]
    b1_capacity = _capacity(benchmark.CudaAutotuneCandidateV2(1, 1), vram=c["vram"])
    rows = [
        (
            {
                **row,
                "capacity": benchmark._b136_capacity_record_v2(
                    capacity=b1_capacity,
                    source_binding=source,
                    temporal_k=1,
                ),
            }
            if row["temporal_k"] == 1
            else row
        )
        for row in c["b136"]["per_k"]
    ]
    eligible, failures = benchmark._b136_closure_v2(
        {**c["b136"], "per_k": rows},
        temporal_k_candidates=c["plan"].temporal_k_candidates,
        measurements=c["measurements"],
        source_binding=source,
    )
    assert not eligible
    assert "B136_K1_capacity_skip_is_not_explicit" in failures


@pytest.mark.parametrize("field", ["artifact", "manifest", "device"])
def test_target_rejects_foreign_outer_source(field: str) -> None:
    c = _complete()
    context = dict(c["context"])
    if field == "artifact":
        context["artifact_sha256"] = "9" * 64
    elif field == "manifest":
        context["manifest_sha256"] = "9" * 64
    else:
        context["device_identity"] = _device(uuid="GPU-foreign")
    verdict = benchmark._target_verdict_v2(
        c["selected"], c["comparison"]["graph_topology"], **context
    )
    assert verdict["status"] != "TARGET_CLOSURE_PASS_PRELIMINARY_NON_AUTHORITATIVE"


def test_authoritative_validator_reopens_source_and_current_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    artifact = tmp_path / "v2.so"
    artifact.touch()
    result = _result(c, artifact)
    _authority_fakes(monkeypatch, result, artifact)
    verdict = benchmark.validate_cuda_event_autotune_result_v2(result)
    assert verdict["status"] == "FULL_TARGET_PASS"
    assert verdict["authoritative"] is True
    assert verdict["selected_p10_world_ticks_per_second"] == 500_000.0
    assert verdict["preliminary_claimant_receipt"]["authoritative"] is False
    host = benchmark.validate_cuda_event_autotune_result_v2(
        result, gpu_validation_requested=False
    )
    assert host["status"] == "SOURCE_VALIDATED_PRELIMINARY_NON_AUTHORITATIVE"
    assert host["authority_class"] == "PRELIMINARY_CLAIMANT_NON_AUTHORITATIVE"


def test_authoritative_validator_accepts_validation_time_free_vram_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    artifact = tmp_path / "v2.so"
    artifact.touch()
    result = _result(c, artifact)
    _authority_fakes(monkeypatch, result, artifact)
    validation_vram = {
        **result.initial_vram,
        "free_bytes": result.initial_vram["free_bytes"] - 4096,
        "allocated_bytes": 2048,
        "reserved_bytes": 4096,
    }
    monkeypatch.setattr(
        benchmark, "_live_vram_v2", lambda torch, device: validation_vram
    )

    verdict = benchmark.validate_cuda_event_autotune_result_v2(result)

    authority = verdict["authoritative_validation"]
    assert authority["initial_vram"] == result.initial_vram
    assert authority["validation_vram"] == validation_vram


def test_authoritative_validator_rejects_validation_time_total_vram_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    artifact = tmp_path / "v2.so"
    artifact.touch()
    result = _result(c, artifact)
    _authority_fakes(monkeypatch, result, artifact)
    monkeypatch.setattr(
        benchmark,
        "_live_vram_v2",
        lambda torch, device: {
            **result.initial_vram,
            "total_bytes": result.initial_vram["total_bytes"] + 1,
        },
    )

    with pytest.raises(
        benchmark.CudaFullEvolutionAutotuneError,
        match="total VRAM",
    ):
        benchmark.validate_cuda_event_autotune_result_v2(result)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing",
        "negative",
        "huge",
        "allocated_above_reserved",
        "reserved_above_total",
    ],
)
def test_authoritative_validator_rejects_noncanonical_launch_vram(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    artifact = tmp_path / "v2.so"
    artifact.touch()
    result = _result(c, artifact)
    _authority_fakes(monkeypatch, result, artifact)
    forged = dict(result.initial_vram)
    if mutation == "extra":
        forged["claimant_extra"] = 1
    elif mutation == "missing":
        del forged["reserved_bytes"]
    elif mutation == "negative":
        forged["allocated_bytes"] = -1
    elif mutation == "huge":
        forged["total_bytes"] = benchmark.MAX_SIGNED_INT64 + 1
    elif mutation == "allocated_above_reserved":
        forged["allocated_bytes"] = 2
        forged["reserved_bytes"] = 1
    else:
        forged["reserved_bytes"] = forged["total_bytes"] + 1

    with pytest.raises(benchmark.CudaFullEvolutionAutotuneError):
        benchmark.validate_cuda_event_autotune_result_v2(
            replace(result, initial_vram=forged)
        )


def test_authoritative_validator_rejects_rebuilt_foreign_free_vram_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "v2.so"
    artifact.touch()
    honest = _result(_complete(), artifact)
    forged_vram = {
        **honest.initial_vram,
        "free_bytes": honest.initial_vram["free_bytes"] - 4096,
    }
    forged_context = _complete(vram=forged_vram)
    forged = _result(forged_context, artifact, publish_launch_origin=False)
    forged = replace(forged, launch_origin=honest.launch_origin)
    _authority_fakes(monkeypatch, forged, artifact)

    with pytest.raises(
        benchmark.CudaFullEvolutionAutotuneError,
        match="launch origin",
    ):
        benchmark.validate_cuda_event_autotune_result_v2(forged)


def test_authoritative_validator_requires_persisted_launch_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    artifact = tmp_path / "v2.so"
    artifact.touch()
    result = _result(c, artifact, publish_launch_origin=False)
    _authority_fakes(monkeypatch, result, artifact)

    with pytest.raises(
        benchmark.CudaFullEvolutionAutotuneError,
        match="launch origin reference",
    ):
        benchmark.validate_cuda_event_autotune_result_v2(result)


@pytest.mark.parametrize(
    "attack", ["tamper", "symlink", "hardlink", "writable", "replacement"]
)
def test_authoritative_validator_rejects_hostile_launch_sidecar(
    attack: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    artifact = tmp_path / "v2.so"
    artifact.touch()
    result = _result(c, artifact)
    _authority_fakes(monkeypatch, result, artifact)
    launch_path = Path(result.launch_origin["path"])
    original = launch_path.read_bytes()
    if attack == "tamper":
        os.chmod(launch_path, 0o600)
        payload = json.loads(original)
        payload["pid"] += 1
        launch_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="ascii",
        )
        os.chmod(launch_path, 0o400)
    elif attack == "symlink":
        target = tmp_path / "copied-launch.json"
        target.write_bytes(original)
        os.chmod(target, 0o400)
        launch_path.unlink()
        launch_path.symlink_to(target)
    elif attack == "hardlink":
        os.link(launch_path, tmp_path / "second-launch-link.json")
    elif attack == "writable":
        os.chmod(launch_path, 0o600)
    else:
        replacement = tmp_path / "replacement-launch.json"
        payload = json.loads(original)
        payload["launch_timestamp_utc"] = "2000-01-01T00:00:00.000000Z"
        replacement.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="ascii",
        )
        os.chmod(replacement, 0o400)
        os.replace(replacement, launch_path)

    with pytest.raises(benchmark.CudaFullEvolutionAutotuneError):
        benchmark.validate_cuda_event_autotune_result_v2(result)


def test_launch_origin_is_exclusive_single_link_and_read_only(tmp_path: Path) -> None:
    c = _complete()
    artifact = tmp_path / "v2.so"
    artifact.touch()
    result = _result(c, artifact)
    launch_path = Path(result.launch_origin["path"])
    metadata = launch_path.stat()
    assert stat.S_ISREG(metadata.st_mode)
    assert metadata.st_nlink == 1
    assert stat.S_IMODE(metadata.st_mode) & 0o222 == 0
    encoded = launch_path.read_bytes()
    payload = json.loads(encoded)
    body = {key: value for key, value in payload.items() if key != "body_sha256"}
    assert payload["artifact_sha256"] == ARTIFACT_SHA
    assert payload["manifest_sha256"] == MANIFEST_SHA
    assert payload["device_identity"] == c["device"]
    assert payload["initial_vram"] == c["vram"]
    assert payload["plan_identity"] == benchmark._launch_plan_identity_v2(result.plan)
    assert payload["harness_sha256"] == HARNESS_SHA
    assert payload["performance_restructure_contract_sha256"] == CONTRACT_SHA
    assert payload["validation_matrix_sha256"] == MATRIX_SHA
    assert payload["pid"] == os.getpid()
    assert payload["body_sha256"] == benchmark._canonical_sha256_v2(body)
    assert result.launch_origin == {
        "path": str(launch_path),
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
        "body_sha256": payload["body_sha256"],
    }

    with pytest.raises(
        benchmark.CudaFullEvolutionAutotuneError,
        match="must not overwrite",
    ):
        benchmark._publish_launch_origin_v2(
            launch_path,
            artifact_path=artifact,
            artifact_sha256=ARTIFACT_SHA,
            manifest_sha256=MANIFEST_SHA,
            device_identity=c["device"],
            initial_vram=c["vram"],
            plan=result.plan,
            harness_sha256=HARNESS_SHA,
            performance_restructure_contract_sha256=CONTRACT_SHA,
            validation_matrix_sha256=MATRIX_SHA,
        )


def test_launch_origin_never_exposes_a_writable_directory_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    artifact = tmp_path / "v2.so"
    artifact.touch()
    launch_path = tmp_path / "result.launch.json"
    plan = replace(c["plan"], artifact_path=artifact)
    original_write_all = benchmark._write_all_v2
    checked = False

    def checked_write(file_descriptor: int, payload: bytes) -> None:
        nonlocal checked
        metadata = os.lstat(launch_path)
        assert stat.S_IMODE(metadata.st_mode) & 0o222 == 0
        with pytest.raises(PermissionError):
            contender = os.open(launch_path, os.O_WRONLY)
            os.close(contender)
        checked = True
        original_write_all(file_descriptor, payload)

    monkeypatch.setattr(benchmark, "_write_all_v2", checked_write)
    benchmark._publish_launch_origin_v2(
        launch_path,
        artifact_path=artifact,
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=c["device"],
        initial_vram=c["vram"],
        plan=plan,
        harness_sha256=HARNESS_SHA,
        performance_restructure_contract_sha256=CONTRACT_SHA,
        validation_matrix_sha256=MATRIX_SHA,
    )
    assert checked is True
    assert stat.S_IMODE(launch_path.stat().st_mode) == stat.S_IRUSR


def test_launch_origin_rejects_midwrite_path_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    artifact = tmp_path / "v2.so"
    artifact.touch()
    launch_path = tmp_path / "result.launch.json"
    plan = replace(c["plan"], artifact_path=artifact)
    original_fsync_directory = benchmark._fsync_directory_v2
    replaced = False

    def replace_path_after_fsync(path: Path) -> None:
        nonlocal replaced
        original_fsync_directory(path)
        if not replaced:
            replaced = True
            launch_path.unlink()
            launch_path.write_bytes(b"foreign replacement\n")
            os.chmod(launch_path, 0o400)

    monkeypatch.setattr(benchmark, "_fsync_directory_v2", replace_path_after_fsync)

    with pytest.raises(
        benchmark.CudaFullEvolutionAutotuneError,
        match="changed during publication",
    ):
        benchmark._publish_launch_origin_v2(
            launch_path,
            artifact_path=artifact,
            artifact_sha256=ARTIFACT_SHA,
            manifest_sha256=MANIFEST_SHA,
            device_identity=c["device"],
            initial_vram=c["vram"],
            plan=plan,
            harness_sha256=HARNESS_SHA,
            performance_restructure_contract_sha256=CONTRACT_SHA,
            validation_matrix_sha256=MATRIX_SHA,
        )
    assert launch_path.read_bytes() == b"foreign replacement\n"


def test_authoritative_verdict_never_reports_forged_preliminary_billion_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _with_preliminary_selected_rate(_complete(), 1_000_000_000.0)
    artifact = tmp_path / "v2.so"
    artifact.touch()
    result = _result(c, artifact)
    assert result.target_verdict["selected_p10_world_ticks_per_second"] == (
        1_000_000_000.0
    )
    _authority_fakes(monkeypatch, result, artifact)

    verdict = benchmark.validate_cuda_event_autotune_result_v2(result)

    assert verdict["status"] == "FULL_TARGET_PASS"
    assert verdict["selected_p10_world_ticks_per_second"] == 500_000.0
    assert "1000000000.0" not in json.dumps(verdict, sort_keys=True)


def test_authoritative_live_replay_rechecks_modes_fixtures_and_c512(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    artifact = tmp_path / "v2.so"
    result = _result(c, artifact)
    originals = {
        value.execution_mode: value
        for value in result.measurements
        if value.candidate == result.selected.candidate and value.status == "VALID"
    }
    calls: list[str] = []

    def measure(candidate, *, execution_mode, **kwargs):
        calls.append(execution_mode)
        return replace(originals[execution_mode], stage_index=None)

    monkeypatch.setattr(benchmark, "measure_candidate_v2", measure)
    monkeypatch.setattr(
        benchmark,
        "measure_c512_saturation_v2",
        lambda *args, **kwargs: result.c512_saturation_evidence,
    )
    replay = benchmark._authoritative_live_replay_v2(
        result, torch=_FakeTorch([]), validation_vram=result.initial_vram
    )
    assert calls == list(benchmark.EXECUTION_MODES_V2)
    assert replay["status"] == "LIVE_REPLAY_EXACT_AND_P10_CLOSED"
    assert replay["selected_execution_mode"] == result.selected.execution_mode
    assert replay["selected_p10_world_ticks_per_second"] == (
        result.selected.p10_sample_world_ticks_per_second
    )
    assert all(value["authoritative"] for value in replay["measurements"])
    assert replay["c512_saturation_evidence"]["authoritative"] is True


def test_authoritative_live_replay_uses_current_vram_capacity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    result = _result(c, tmp_path / "v2.so")
    current_vram = {
        **result.initial_vram,
        "free_bytes": result.initial_vram["free_bytes"] - 4096,
        "allocated_bytes": 2048,
        "reserved_bytes": 4096,
    }
    originals = {
        value.execution_mode: value
        for value in result.measurements
        if value.candidate == result.selected.candidate and value.status == "VALID"
    }
    calls: list[tuple[int, int]] = []

    def measure(candidate, *, execution_mode, free_vram_bytes, total_vram_bytes, **kw):
        calls.append((free_vram_bytes, total_vram_bytes))
        capacity = benchmark.capacity_for_candidate_v2(
            candidate,
            free_vram_bytes=free_vram_bytes,
            total_vram_bytes=total_vram_bytes,
            reserve_bytes=result.reserve_bytes,
            dispatches_per_candidate=result.plan.dispatches_per_candidate,
        )
        return replace(originals[execution_mode], capacity=capacity, stage_index=None)

    monkeypatch.setattr(benchmark, "measure_candidate_v2", measure)
    monkeypatch.setattr(
        benchmark,
        "measure_c512_saturation_v2",
        lambda *args, **kwargs: result.c512_saturation_evidence,
    )

    replay = benchmark._authoritative_live_replay_v2(
        result, torch=_FakeTorch([]), validation_vram=current_vram
    )

    assert replay["status"] == "LIVE_REPLAY_EXACT_AND_P10_CLOSED"
    assert calls == [
        (current_vram["free_bytes"], current_vram["total_bytes"]),
        (current_vram["free_bytes"], current_vram["total_bytes"]),
    ]


def test_authoritative_live_replay_rejects_current_vram_non_admission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    result = _result(c, tmp_path / "v2.so")
    called = False

    def measure(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("measurement must not start after failed admission")

    monkeypatch.setattr(benchmark, "measure_candidate_v2", measure)
    current_vram = {
        "free_bytes": 0,
        "total_bytes": result.initial_vram["total_bytes"],
        "allocated_bytes": 0,
        "reserved_bytes": 0,
    }

    with pytest.raises(
        benchmark.CudaFullEvolutionAutotuneError,
        match="current validation VRAM cannot admit",
    ):
        benchmark._authoritative_live_replay_v2(
            result, torch=_FakeTorch([]), validation_vram=current_vram
        )
    assert called is False


@pytest.mark.parametrize(
    "mutation", ["path", "hash", "device", "vram", "counter", "digest"]
)
def test_authoritative_validator_rejects_claimant_fakes(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    artifact = tmp_path / "v2.so"
    artifact.touch()
    result = _result(c, artifact)
    _authority_fakes(monkeypatch, result, artifact)
    if mutation == "path":
        result = replace(result, artifact_path=str(tmp_path / "foreign.so"))
    elif mutation == "hash":
        result = replace(result, artifact_sha256="9" * 64)
    elif mutation == "device":
        result = replace(result, device_identity=_device(uuid="GPU-foreign"))
    elif mutation == "vram":
        result = replace(
            result,
            initial_vram={
                **result.initial_vram,
                "free_bytes": 2**50,
                "total_bytes": 2**50,
            },
        )
    else:
        fixture = (
            replace(result.selected.fixture, final_counters={})
            if mutation == "counter"
            else replace(result.selected.fixture, final_state_sha256s=("9" * 64,) * 3)
        )
        selected = replace(result.selected, fixture=fixture)
        measurements = tuple(
            selected if value == result.selected else value
            for value in result.measurements
        )
        result = replace(result, selected=selected, measurements=measurements)
    with pytest.raises(benchmark.CudaFullEvolutionAutotuneError):
        benchmark.validate_cuda_event_autotune_result_v2(result)


@pytest.mark.parametrize("section", ["b136", "staged"])
def test_original_sweep_cannot_inject_authoritative_rows(
    section: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    artifact = tmp_path / "v2.so"
    artifact.touch()
    result = _result(c, artifact)
    _authority_fakes(monkeypatch, result, artifact)
    if section == "b136":
        rows = list(result.b136_comparison["per_k"])
        rows[0] = {
            **rows[0],
            "authority_class": "AUTHORITATIVE_FRESH_LIVE_REPLAY",
            "authoritative": True,
        }
        result = replace(
            result,
            b136_comparison={**result.b136_comparison, "per_k": rows},
        )
    else:
        stages = list(result.staged_envelope["stages"])
        stages[0] = {
            **stages[0],
            "authority_class": "AUTHORITATIVE_FRESH_LIVE_REPLAY",
            "authoritative": True,
        }
        result = replace(
            result,
            staged_envelope={**result.staged_envelope, "stages": stages},
        )

    with pytest.raises(benchmark.CudaFullEvolutionAutotuneError):
        benchmark.validate_cuda_event_autotune_result_v2(result)


def test_writer_requires_authoritative_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    claimant_peak = 9_876_543
    claimant_c512_elapsed = 123_456_789.0
    c = _with_preliminary_selected_rate(
        _complete(),
        1_000_000_000.0,
        peak_bytes=claimant_peak,
        c512_elapsed_ms=claimant_c512_elapsed,
    )
    artifact = tmp_path / "v2.so"
    artifact.touch()
    root = tmp_path / "receipts"
    requested_output = root / "autotune.json"
    result = _result(
        c,
        artifact,
        launch_origin_path=root / "autotune.launch.json",
    )
    live_replay = _fake_live_replay(
        result,
        selected_p10=500_000.0,
        peak_base=401,
        c512_elapsed_ms=0.125,
    )
    authoritative_verdict = _sealed_authoritative_verdict(result, artifact, live_replay)
    called = []

    def validate(value):
        called.append(value)
        return authoritative_verdict

    monkeypatch.setattr(benchmark, "AUTOTUNE_ARTIFACT_ROOT_V2", root)
    monkeypatch.setattr(benchmark, "validate_cuda_event_autotune_result_v2", validate)
    output = benchmark.write_cuda_event_autotune_result_v2(result, requested_output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    published_stat = output.stat()
    assert stat.S_ISREG(published_stat.st_mode)
    assert published_stat.st_nlink == 1
    assert stat.S_IMODE(published_stat.st_mode) == stat.S_IRUSR
    assert called == [result]
    assert payload["schema_version"] == (
        "eiel_cuda_event_autotune_authoritative_result_v2"
    )
    assert payload["authoritative"] is True
    assert payload["launch_origin"] == result.launch_origin
    assert payload["initial_vram"] == result.initial_vram
    assert payload["validation_vram"] == result.initial_vram
    assert payload["selected_p10_world_ticks_per_second"] == 500_000.0
    assert payload["target_verdict"]["status"] == "FULL_TARGET_PASS"
    assert payload["target_verdict"]["selected_p10_world_ticks_per_second"] == (
        500_000.0
    )
    assert (
        payload["selected"]["selected_statistic_p10_world_ticks_per_second"]
        == 500_000.0
    )
    assert payload["peak_allocation"]["maximum_device_process_bytes"] == 403
    assert (
        payload["c512_saturation_evidence"]["sample"]["cuda_event_elapsed_ms"] == 0.125
    )
    assert all(value["authoritative"] for value in payload["measurements"])
    assert payload["preliminary_claimant_receipt"]["authoritative"] is False
    authority = benchmark.SOURCE_VALIDATED_ORIGINAL_SWEEP_AUTHORITY_V2
    b136 = payload["b136_comparison"]
    staged = payload["staged_envelope"]
    for section, original in ((b136, c["b136"]), (staged, c["staged"])):
        assert section["authority_class"] == authority
        assert section["authoritative"] is False
        assert section["contained_rows_authority_class"] == authority
        assert section["contained_rows_authoritative"] is False
        assert section["original_receipt_sha256"] == (
            benchmark._canonical_sha256_v2(original)
        )
        reporting_body = {
            key: value
            for key, value in section.items()
            if key != "source_validated_reporting_sha256"
        }
        assert section["source_validated_reporting_sha256"] == (
            benchmark._canonical_sha256_v2(reporting_body)
        )
        cross_link = section["selection_cross_link"]
        cross_link_body = {
            key: value
            for key, value in cross_link.items()
            if key != "selection_cross_link_sha256"
        }
        assert cross_link["selection_cross_link_sha256"] == (
            benchmark._canonical_sha256_v2(cross_link_body)
        )
        assert (
            cross_link["original_sweep_selected"]["p10_sample_world_ticks_per_second"]
            == 1_000_000_000.0
        )
        assert (
            cross_link["fresh_replay_selected_p10_world_ticks_per_second"] == 500_000.0
        )
        assert cross_link["candidate_and_execution_mode_exact_match"] is True
    assert len(b136["per_k_performance"]) == len(result.plan.temporal_k_candidates)
    for row in b136["per_k_performance"]:
        assert row["authority_class"] == authority
        assert row["authoritative"] is False
        row_body = {
            key: value for key, value in row.items() if key != "performance_row_sha256"
        }
        assert row["performance_row_sha256"] == benchmark._canonical_sha256_v2(row_body)
        for measurement in row["measurements"]:
            assert measurement["authority_class"] == authority
            assert measurement["authoritative"] is False
            assert "p10_sample_world_ticks_per_second" in measurement
            measurement_body = {
                key: value
                for key, value in measurement.items()
                if key != "reporting_row_sha256"
            }
            assert measurement["reporting_row_sha256"] == (
                benchmark._canonical_sha256_v2(measurement_body)
            )
    assert staged["aggregate_sha256"] == c["staged"]["aggregate_sha256"]
    assert payload["selected"]["peak_device_process_bytes"] == 403
    assert (
        payload["selected"]["selected_statistic_p10_world_ticks_per_second"]
        == 500_000.0
    )
    serialized = output.read_text(encoding="utf-8")
    assert "1000000000.0" in serialized
    assert str(claimant_peak) in serialized
    assert str(int(claimant_c512_elapsed)) not in serialized
    with pytest.raises(benchmark.CudaFullEvolutionAutotuneError):
        benchmark.write_cuda_event_autotune_result_v2(result, output)


def test_writer_never_replaces_concurrently_created_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    artifact = tmp_path / "v2.so"
    artifact.touch()
    root = tmp_path / "receipts"
    output = root / "autotune.json"
    result = _result(
        c,
        artifact,
        launch_origin_path=root / "autotune.launch.json",
    )
    verdict = _sealed_authoritative_verdict(result, artifact, _fake_live_replay(result))
    monkeypatch.setattr(benchmark, "AUTOTUNE_ARTIFACT_ROOT_V2", root)

    def competing_writer(value):
        assert value is result
        output.write_text("competing evidence\n", encoding="utf-8")
        return verdict

    monkeypatch.setattr(
        benchmark, "validate_cuda_event_autotune_result_v2", competing_writer
    )

    with pytest.raises(
        benchmark.CudaFullEvolutionAutotuneError,
        match="must not overwrite",
    ):
        benchmark.write_cuda_event_autotune_result_v2(result, output)
    assert output.read_text(encoding="utf-8") == "competing evidence\n"
    assert not (root / ".autotune.json.tmp").exists()


def test_writer_rejects_temporary_path_replacement_without_false_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = _complete()
    artifact = tmp_path / "v2.so"
    artifact.touch()
    root = tmp_path / "receipts"
    output = root / "autotune.json"
    temporary = root / ".autotune.json.tmp"
    result = _result(
        c,
        artifact,
        launch_origin_path=root / "autotune.launch.json",
    )
    verdict = _sealed_authoritative_verdict(result, artifact, _fake_live_replay(result))
    monkeypatch.setattr(benchmark, "AUTOTUNE_ARTIFACT_ROOT_V2", root)
    monkeypatch.setattr(
        benchmark, "validate_cuda_event_autotune_result_v2", lambda value: verdict
    )
    original_link = os.link

    def replace_temporary_then_link(source, destination, *, follow_symlinks=True):
        assert Path(source) == temporary
        temporary.unlink()
        temporary.write_text("foreign race payload\n", encoding="ascii")
        os.chmod(temporary, stat.S_IRUSR)
        return original_link(
            source,
            destination,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(benchmark.os, "link", replace_temporary_then_link)

    with pytest.raises(
        benchmark.CudaFullEvolutionAutotuneError,
        match="exact created inode and content",
    ):
        benchmark.write_cuda_event_autotune_result_v2(result, output)

    assert not output.exists()
    assert temporary.read_text(encoding="ascii") == "foreign race payload\n"


def test_run_rejects_incomplete_explicit_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "v2.so"
    artifact.touch()
    loaded = SimpleNamespace(
        artifact_path=artifact,
        artifact_sha256=ARTIFACT_SHA,
        manifest_sha256=MANIFEST_SHA,
        device_identity=_device(),
        manifest={},
    )
    monkeypatch.setattr(
        benchmark, "load_prebuilt_cuda_full_evolution_v2", lambda *a, **k: loaded
    )
    plan = benchmark.CudaEventAutotunePlanV2(
        artifact, batch_candidates=(1,), reserve_bytes=0
    )
    launch_origin = tmp_path / "incomplete.launch.json"
    with pytest.raises(benchmark.CudaFullEvolutionAutotuneError, match="required"):
        benchmark.run_cuda_event_autotune_v2(
            plan,
            launch_origin_path=launch_origin,
            torch_module=_FakeTorch([]),
        )
    assert launch_origin.is_file()
    assert stat.S_IMODE(launch_origin.stat().st_mode) & 0o222 == 0
