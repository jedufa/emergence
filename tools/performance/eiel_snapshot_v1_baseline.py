#!/usr/bin/env python3
"""Bounded host microbenchmark for the authoritative V1 snapshot writer.

This tool deliberately does not model a campaign boundary, asynchronous handoff,
GPU transfer, lifecycle registration, or full durable Science Instrument run.
It supplies deterministic CPU-resident native-layout inputs to the real
``write_compact_science_snapshot_v1`` implementation and reports that writer's
host-side baseline cost only.  Denominator samples are always untimed.  Optional
phase diagnostics are collected in a separate, alternating-order peer stream.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch

import engine.native_runtime.cuda_v2_science_capture_v1 as capture_adapter
from engine.campaigns.orchestration import atomic_write_json
from engine.native_runtime.cuda_v2_compact_science_capture_v1 import (
    COMPACT_SCIENCE_SNAPSHOT_TIMING_SCHEMA_V1,
    canonical_sha256,
    write_compact_science_snapshot_v1,
)
from engine.perf.cuda_full_evolution_v2.direct_sparse_runtime import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
)


SCHEMA_VERSION = "eiel_snapshot_v1_host_baseline_report_v1"
CLAIM_CLASS = "TECHNICAL_ONLY"
CONTROL_SET_SHA256 = "3" * 64
DEFAULT_OCCUPANCIES = ("zero", "sparse", "all")
MEBIBYTE = 1024 * 1024
OBSERVER_OVERHEAD_SCHEMA_VERSION = "eiel_snapshot_v1_phase_observer_overhead_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(MEBIBYTE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_layout() -> dict[str, int]:
    layout = capture_adapter._STATE_LAYOUT
    capacity = int(layout.field("agent_id").shape[0])
    alive_word_count = int(layout.field("alive_words").shape[0])
    state_bytes_per_world = int(capture_adapter._STATE_BYTES)
    if capacity <= 0 or alive_word_count * 32 < capacity:
        raise ValueError("native state layout has an invalid alive capacity")
    return {
        "state_bytes_per_world": state_bytes_per_world,
        "agent_capacity_per_world": capacity,
        "alive_word_count_per_world": alive_word_count,
        "metric_field_count": len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1),
    }


def _native_layout_projection() -> dict[str, Any]:
    layout = capture_adapter._STATE_LAYOUT
    body: dict[str, Any] = {
        **_native_layout(),
        "fields": [
            {
                "name": field.name,
                "dtype": field.dtype,
                "shape": list(field.shape),
                "offset_bytes": int(field.offset_bytes),
                "bytes": int(field.bytes),
            }
            for field in layout.fields
        ],
        "metric_fields": list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1),
    }
    return {**body, "projection_sha256": canonical_sha256(body)}


def _pack_field(
    raw: bytearray,
    *,
    world_base: int,
    field: str,
    wire_type: str,
    index: int,
    value: int,
) -> None:
    size = struct.calcsize(wire_type)
    struct.pack_into(
        wire_type,
        raw,
        world_base + capture_adapter._field_offset(field) + size * index,
        value,
    )


def _is_alive(*, occupancy: str, slot: int, sparse_stride: int) -> bool:
    if occupancy == "zero":
        return False
    if occupancy == "all":
        return True
    if occupancy == "sparse":
        return slot % sparse_stride == 0
    raise ValueError(f"unsupported occupancy: {occupancy}")


def _synthetic_inputs(
    *,
    batch_size: int,
    occupancy: str,
    absolute_tick: int,
    sparse_stride: int,
) -> dict[str, Any]:
    layout = _native_layout()
    state_bytes_per_world = layout["state_bytes_per_world"]
    capacity = layout["agent_capacity_per_world"]
    alive_word_count = layout["alive_word_count_per_world"]
    metric_fields = tuple(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
    raw = bytearray(batch_size * state_bytes_per_world)
    active_per_world = sum(
        _is_alive(
            occupancy=occupancy,
            slot=slot,
            sparse_stride=sparse_stride,
        )
        for slot in range(capacity)
    )

    policy_offset = capture_adapter._field_offset("policy_alleles")
    trait_offset = capture_adapter._field_offset("trait_alleles")
    for world_index in range(batch_size):
        world_base = world_index * state_bytes_per_world
        world_uid = world_index + 1
        _pack_field(
            raw,
            world_base=world_base,
            field="scalar_header_counters",
            wire_type="<Q",
            index=0,
            value=world_uid,
        )
        _pack_field(
            raw,
            world_base=world_base,
            field="scalar_header_counters",
            wire_type="<Q",
            index=1,
            value=absolute_tick,
        )
        _pack_field(
            raw,
            world_base=world_base,
            field="scalar_header_counters",
            wire_type="<Q",
            index=3,
            value=active_per_world,
        )
        alive_words = [0] * alive_word_count
        for slot in range(capacity):
            agent_id = world_index * capacity + slot + 1
            alive = _is_alive(
                occupancy=occupancy,
                slot=slot,
                sparse_stride=sparse_stride,
            )
            if alive:
                alive_words[slot // 32] |= 1 << (slot % 32)
            _pack_field(
                raw,
                world_base=world_base,
                field="agent_id",
                wire_type="<Q",
                index=slot,
                value=agent_id,
            )
            _pack_field(
                raw,
                world_base=world_base,
                field="position",
                wire_type="<H",
                index=slot,
                value=(slot * 7 + world_index) % 1024,
            )
            _pack_field(
                raw,
                world_base=world_base,
                field="energy",
                wire_type="<h",
                index=slot,
                value=100 - slot % 200,
            )
            _pack_field(
                raw,
                world_base=world_base,
                field="health",
                wire_type="<h",
                index=slot,
                value=80 - slot % 160,
            )
            _pack_field(
                raw,
                world_base=world_base,
                field="age",
                wire_type="<H",
                index=slot,
                value=slot + 1,
            )
            _pack_field(
                raw,
                world_base=world_base,
                field="absolute_birth_tick",
                wire_type="<Q",
                index=slot,
                value=max(0, absolute_tick - slot),
            )
            raw[world_base + policy_offset + slot] = (agent_id % 101) & 0xFF
            raw[world_base + policy_offset + capacity + slot] = (
                -(agent_id % 31)
            ) & 0xFF
            struct.pack_into(
                "<h",
                raw,
                world_base + trait_offset + 2 * slot,
                -(agent_id % 1000),
            )
            struct.pack_into(
                "<h",
                raw,
                world_base + trait_offset + 2 * (capacity + slot),
                agent_id % 1000,
            )
        for index, value in enumerate(alive_words):
            _pack_field(
                raw,
                world_base=world_base,
                field="alive_words",
                wire_type="<I",
                index=index,
                value=value,
            )

    metrics = torch.zeros((batch_size, capacity, len(metric_fields)), dtype=torch.int64)
    agent_ids = torch.arange(1, batch_size * capacity + 1, dtype=torch.int64).reshape(
        batch_size, capacity
    )
    metrics[:, :, 0] = agent_ids * 10
    metrics[:, :, -1] = torch.arange(capacity, dtype=torch.int64) + 1000
    metrics_raw = metrics.numpy().astype("<i8", copy=False).tobytes(order="C")
    world_uids = tuple(range(1, batch_size + 1))
    world_cells = {
        f"cell-{world_uid:06d}": {
            "science_logical_world_id": world_uid,
            "science_ownership": {
                "study_id": "snapshot-v1-host-baseline",
                "arm_id": occupancy,
            },
            "science_world_sha256": {
                "algorithm": "sha256",
                "hex": f"{world_uid % 16:x}" * 64,
            },
            "science_acquisition_sha256": {
                "algorithm": "sha256",
                "hex": f"{(world_uid + 1) % 16:x}" * 64,
            },
            "science_acquisition": {"snapshot_ticks": [absolute_tick]},
        }
        for world_uid in reversed(world_uids)
    }
    return {
        "state": bytes(raw),
        "active_metrics_projection": {
            "schema_version": ("eiel_full_evolution_cuda_active_science_metrics_v1"),
            "next_absolute_tick": absolute_tick,
            "world_uids": list(world_uids),
            "metric_fields": list(metric_fields),
            "shape": list(metrics.shape),
            "content_sha256": _sha256_bytes(metrics_raw),
            "tensor": metrics,
        },
        "world_cells": world_cells,
        "active_agent_count": batch_size * active_per_world,
        "nonzero_agent_id_count": batch_size * capacity,
        "state_input_bytes": len(raw),
        "metrics_input_bytes": len(metrics_raw),
    }


def _nearest_rank_p95(values: Sequence[float | int]) -> float | int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _summary(values: Sequence[float | int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p95": _nearest_rank_p95(values),
        "max": max(values),
    }


def _phase_aggregates(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    timed = [
        sample
        for sample in samples
        if isinstance(sample.get("timing_diagnostics"), Mapping)
    ]
    if not timed:
        return {
            "enabled": False,
            "schema_version": COMPACT_SCIENCE_SNAPSHOT_TIMING_SCHEMA_V1,
            "sample_count": 0,
            "sample_stream": "phase_timed_samples",
            "eligible_as_denominator": False,
            "phases": {},
            "derived": {},
        }
    phase_names = sorted(
        {
            name
            for sample in timed
            for name in sample["timing_diagnostics"].get("phases", {})
        }
    )
    phases: dict[str, Any] = {}
    for name in phase_names:
        rows = [
            sample["timing_diagnostics"]["phases"].get(
                name,
                {"calls": 0, "wall_time_ns": 0, "cpu_time_ns": 0},
            )
            for sample in timed
        ]
        phases[name] = {
            "calls_total": sum(int(row.get("calls", 0)) for row in rows),
            "wall_time_ns": _summary([int(row.get("wall_time_ns", 0)) for row in rows]),
            "cpu_time_ns": _summary([int(row.get("cpu_time_ns", 0)) for row in rows]),
        }
    derived_names = sorted(
        {
            name
            for sample in timed
            for name in sample["timing_diagnostics"].get("derived", {})
        }
    )
    derived = {
        name: _summary(
            [
                int(sample["timing_diagnostics"]["derived"].get(name, 0))
                for sample in timed
            ]
        )
        for name in derived_names
    }
    return {
        "enabled": True,
        "schema_version": COMPACT_SCIENCE_SNAPSHOT_TIMING_SCHEMA_V1,
        "sample_count": len(timed),
        "sample_stream": "phase_timed_samples",
        "eligible_as_denominator": False,
        "phases": phases,
        "derived": derived,
    }


def _run_sample(
    *,
    destination: Path,
    synthetic: Mapping[str, Any],
    absolute_tick: int,
    include_genome_catalog: bool,
    phase_timing: bool,
    sample_index: int,
    warmup: bool,
) -> dict[str, Any]:
    timing_diagnostics: dict[str, Any] | None = {} if phase_timing else None
    started_ns = time.perf_counter_ns()
    receipt = write_compact_science_snapshot_v1(
        destination,
        state=synthetic["state"],
        active_metrics_projection=synthetic["active_metrics_projection"],
        world_cells=synthetic["world_cells"],
        absolute_tick=absolute_tick,
        native_world_control_set_sha256=CONTROL_SET_SHA256,
        include_genome_catalog=include_genome_catalog,
        timing_diagnostics=timing_diagnostics,
    )
    writer_elapsed_ns = time.perf_counter_ns() - started_ns
    compressed_path = Path(str(receipt["path"])) / str(receipt["data_file"])
    compressed = compressed_path.read_bytes()
    uncompressed = gzip.decompress(compressed)
    if _sha256_bytes(uncompressed) != receipt["uncompressed_content_sha256"]:
        raise RuntimeError("snapshot uncompressed content digest differs")
    if _sha256_bytes(compressed) != receipt["compressed_file_sha256"]:
        raise RuntimeError("snapshot compressed file digest differs")
    if len(compressed) != receipt["compressed_bytes"]:
        raise RuntimeError("snapshot compressed byte count differs")
    elapsed_seconds = writer_elapsed_ns / 1_000_000_000
    row_count = int(receipt["row_count"])
    return {
        "sample_index": sample_index,
        "warmup": warmup,
        "measurement_role": (
            "phase_diagnostic_timed" if phase_timing else "denominator_untimed"
        ),
        "writer_elapsed_ns": writer_elapsed_ns,
        "writer_elapsed_seconds": elapsed_seconds,
        "row_count": row_count,
        "captured_world_count": int(receipt["captured_world_count"]),
        "active_agent_count": sum(
            int(row["active_agent_count"]) for row in receipt["captured_worlds"]
        ),
        # The benchmark supplies CPU-resident bytes/tensors.  This is zero
        # physical device-to-host traffic, even though the writer phase keeps
        # its production-facing D2H-compatible name.
        "d2h_bytes": 0,
        "state_input_bytes": int(synthetic["state_input_bytes"]),
        "metrics_input_bytes": int(synthetic["metrics_input_bytes"]),
        "host_input_bytes": int(synthetic["state_input_bytes"])
        + int(synthetic["metrics_input_bytes"]),
        "uncompressed_bytes": len(uncompressed),
        "compressed_bytes": len(compressed),
        "records_per_second": row_count / elapsed_seconds,
        "uncompressed_mebibytes_per_second": (
            len(uncompressed) / MEBIBYTE / elapsed_seconds
        ),
        "compressed_mebibytes_per_second": (
            len(compressed) / MEBIBYTE / elapsed_seconds
        ),
        "artifact_sha256": receipt["artifact_sha256"],
        "uncompressed_content_sha256": receipt["uncompressed_content_sha256"],
        "compressed_file_sha256": receipt["compressed_file_sha256"],
        **(
            {"timing_diagnostics": timing_diagnostics}
            if timing_diagnostics is not None
            else {}
        ),
    }


def _run_interleaved_samples(
    *,
    destination: Path,
    synthetic: Mapping[str, Any],
    absolute_tick: int,
    include_genome_catalog: bool,
    phase_timing: bool,
    sample_count: int,
    sequence_offset: int,
    warmup: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run an untimed denominator stream and an optional timed peer stream.

    When phase timing is requested, each logical sample is a pair over the
    exact same input.  Pair order alternates across warm-up and measured
    samples so monotonic thermal, cache, or filesystem drift does not always
    favor the same stream.  Only the untimed stream is eligible as a
    performance denominator.
    """

    untimed_samples: list[dict[str, Any]] = []
    timed_samples: list[dict[str, Any]] = []
    sample_class = "warmup" if warmup else "measured"
    for sample_index in range(sample_count):
        sequence_index = sequence_offset + sample_index
        order = (False, True) if sequence_index % 2 == 0 else (True, False)
        order_name = "untimed_then_timed" if order[0] is False else "timed_then_untimed"
        if not phase_timing:
            order = (False,)
            order_name = "untimed_only"
        for timed in order:
            stream_name = "phase-timed" if timed else "denominator-untimed"
            sample = _run_sample(
                destination=(
                    destination / f"{sample_class}-{sample_index:04d}-{stream_name}"
                ),
                synthetic=synthetic,
                absolute_tick=absolute_tick,
                include_genome_catalog=include_genome_catalog,
                phase_timing=timed,
                sample_index=sample_index,
                warmup=warmup,
            )
            sample["interleave_sequence_index"] = sequence_index
            sample["pair_execution_order"] = order_name
            if timed:
                timed_samples.append(sample)
            else:
                untimed_samples.append(sample)
    return untimed_samples, timed_samples


def _observer_overhead(
    untimed_samples: Sequence[Mapping[str, Any]],
    timed_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe the measured observer effect without contaminating T1."""

    base: dict[str, Any] = {
        "schema_version": OBSERVER_OVERHEAD_SCHEMA_VERSION,
        "denominator_stream": "measured_samples",
        "phase_stream": "phase_timed_samples",
        "denominator_observer_timing": False,
        "phase_measurements_eligible_as_denominator": False,
        "interleave_policy": (
            "alternate exact-input sample pairs across the full warmup and "
            "measured sequence: even sequence indices run untimed then timed; "
            "odd indices run timed then untimed"
        ),
        "denominator_sample_count": len(untimed_samples),
        "phase_sample_count": len(timed_samples),
    }
    if not timed_samples:
        return {
            **base,
            "enabled": False,
            "sample_pair_count": 0,
            "untimed_writer_elapsed_ns": _summary(
                [int(sample["writer_elapsed_ns"]) for sample in untimed_samples]
            ),
            "timed_writer_elapsed_ns": None,
            "paired_observer_overhead_ns": None,
            "paired_timed_over_untimed_ratio": None,
            "median_observer_overhead_ns": None,
            "median_timed_over_untimed_ratio": None,
            "ratio_of_stream_medians_timed_over_untimed": None,
        }
    if len(untimed_samples) != len(timed_samples):
        raise RuntimeError("phase observer streams have different lengths")
    for untimed, timed in zip(untimed_samples, timed_samples, strict=True):
        if (
            untimed.get("sample_index") != timed.get("sample_index")
            or untimed.get("warmup") != timed.get("warmup")
            or untimed.get("artifact_sha256") != timed.get("artifact_sha256")
            or untimed.get("uncompressed_content_sha256")
            != timed.get("uncompressed_content_sha256")
            or untimed.get("compressed_file_sha256")
            != timed.get("compressed_file_sha256")
        ):
            raise RuntimeError("phase observer pair identity differs")
    untimed_ns = [int(sample["writer_elapsed_ns"]) for sample in untimed_samples]
    timed_ns = [int(sample["writer_elapsed_ns"]) for sample in timed_samples]
    paired_overhead_ns = [
        timed - untimed for untimed, timed in zip(untimed_ns, timed_ns, strict=True)
    ]
    paired_ratios = [
        timed / untimed for untimed, timed in zip(untimed_ns, timed_ns, strict=True)
    ]
    untimed_summary = _summary(untimed_ns)
    timed_summary = _summary(timed_ns)
    paired_overhead_summary = _summary(paired_overhead_ns)
    paired_ratio_summary = _summary(paired_ratios)
    return {
        **base,
        "enabled": True,
        "sample_pair_count": len(untimed_samples),
        "untimed_writer_elapsed_ns": untimed_summary,
        "timed_writer_elapsed_ns": timed_summary,
        "paired_observer_overhead_ns": paired_overhead_summary,
        "paired_timed_over_untimed_ratio": paired_ratio_summary,
        "median_observer_overhead_ns": paired_overhead_summary["median"],
        "median_timed_over_untimed_ratio": paired_ratio_summary["median"],
        "ratio_of_stream_medians_timed_over_untimed": (
            float(timed_summary["median"]) / float(untimed_summary["median"])
        ),
    }


def _cell_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "writer_elapsed_ns",
        "writer_elapsed_seconds",
        "row_count",
        "captured_world_count",
        "active_agent_count",
        "d2h_bytes",
        "state_input_bytes",
        "metrics_input_bytes",
        "host_input_bytes",
        "uncompressed_bytes",
        "compressed_bytes",
        "records_per_second",
        "uncompressed_mebibytes_per_second",
        "compressed_mebibytes_per_second",
    )
    return {
        "p95_method": "nearest_rank",
        **{field: _summary([sample[field] for sample in samples]) for field in fields},
    }


def _git_identity(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "status_porcelain_sha256": _sha256_bytes(status.encode("utf-8")),
        "changed_paths": [line[3:] for line in status.splitlines()],
    }


def run_snapshot_v1_host_baseline(
    *,
    output: Path,
    batch_size: int,
    repetitions: int,
    warmups: int,
    occupancies: Sequence[str],
    sparse_stride: int,
    absolute_tick: int,
    include_genome_catalog: bool,
    phase_timing: bool,
    command_argv: Sequence[str],
    command_source: str,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if warmups < 0:
        raise ValueError("warmups must be nonnegative")
    if sparse_stride <= 0:
        raise ValueError("sparse_stride must be positive")
    if absolute_tick < 0:
        raise ValueError("absolute_tick must be nonnegative")
    normalized_occupancies = tuple(occupancies)
    if not normalized_occupancies or len(set(normalized_occupancies)) != len(
        normalized_occupancies
    ):
        raise ValueError("occupancies must be a nonempty unique sequence")
    if any(value not in DEFAULT_OCCUPANCIES for value in normalized_occupancies):
        raise ValueError("occupancies contain an unsupported value")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    benchmark_started_ns = time.perf_counter_ns()
    cells: list[dict[str, Any]] = []
    all_measured_samples: list[Mapping[str, Any]] = []
    all_phase_timed_samples: list[Mapping[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix=".eiel-snapshot-v1-host-baseline-", dir=output.parent
    ) as scratch_name:
        scratch_root = Path(scratch_name)
        scratch_device = int(scratch_root.stat().st_dev)
        for occupancy in normalized_occupancies:
            fixture_started_ns = time.perf_counter_ns()
            synthetic = _synthetic_inputs(
                batch_size=batch_size,
                occupancy=occupancy,
                absolute_tick=absolute_tick,
                sparse_stride=sparse_stride,
            )
            fixture_elapsed_ns = time.perf_counter_ns() - fixture_started_ns
            warmup_samples, phase_timed_warmup_samples = _run_interleaved_samples(
                destination=scratch_root / occupancy,
                synthetic=synthetic,
                absolute_tick=absolute_tick,
                include_genome_catalog=include_genome_catalog,
                phase_timing=phase_timing,
                sample_count=warmups,
                sequence_offset=0,
                warmup=True,
            )
            measured_samples, phase_timed_samples = _run_interleaved_samples(
                destination=scratch_root / occupancy,
                synthetic=synthetic,
                absolute_tick=absolute_tick,
                include_genome_catalog=include_genome_catalog,
                phase_timing=phase_timing,
                sample_count=repetitions,
                sequence_offset=warmups,
                warmup=False,
            )
            all_measured_samples.extend(measured_samples)
            all_phase_timed_samples.extend(phase_timed_samples)
            cells.append(
                {
                    "occupancy": occupancy,
                    "fixture_construction_elapsed_ns": fixture_elapsed_ns,
                    "expected_active_agent_count": synthetic["active_agent_count"],
                    "nonzero_agent_id_count": synthetic["nonzero_agent_id_count"],
                    "warmup_repetitions": warmups,
                    "measured_repetitions": repetitions,
                    "warmup_samples": warmup_samples,
                    "measured_samples": measured_samples,
                    "phase_timed_warmup_samples": phase_timed_warmup_samples,
                    "phase_timed_samples": phase_timed_samples,
                    "warmup_summary": (
                        _cell_summary(warmup_samples) if warmup_samples else None
                    ),
                    "measured_summary": _cell_summary(measured_samples),
                    "phase_aggregates": _phase_aggregates(phase_timed_samples),
                    "observer_overhead": _observer_overhead(
                        measured_samples, phase_timed_samples
                    ),
                }
            )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_class": CLAIM_CLASS,
        "benchmark_gate": "G1_CPU_BASELINE",
        "benchmark_scope": "HOST_MICRO_WRITER_V1",
        "implementation": "write_compact_science_snapshot_v1_python_v1",
        "science_result": False,
        "qualification_authority_effect": False,
        "prequalification_started": False,
        "qualification_started": False,
        "gpu_used": False,
        "generated_at_utc": _utc_now(),
        "measurement_boundary": {
            "included": (
                "write_compact_science_snapshot_v1 call using CPU-resident "
                "native-layout state and metric tensors"
            ),
            "excluded": [
                "synthetic fixture construction",
                "physical GPU projection",
                "physical device-to-host transfer",
                "campaign snapshot boundary",
                "bounded asynchronous handoff or backpressure",
                "lifecycle or evidence registration",
                "real science execution",
                "benchmark report publication",
            ],
            "T1": "BASELINE_ONLY_NO_SPEEDUP_CLAIM",
            "T2": "NOT_MEASURED",
            "T3": "NOT_MEASURED",
            "denominator": (
                "measured_samples and measured_summary are always collected "
                "with writer phase timing disabled"
            ),
            "phase_diagnostics": (
                "when requested, phase_timed_samples are separate exact-input "
                "peers and are never eligible as the performance denominator"
            ),
            "durability_scope": (
                "writer data and manifest file publication only; no campaign "
                "or lifecycle durable-completion claim"
            ),
        },
        "configuration": {
            "batch_size": batch_size,
            "repetitions": repetitions,
            "warmups": warmups,
            "occupancies": list(normalized_occupancies),
            "sparse_stride": sparse_stride,
            "absolute_tick": absolute_tick,
            "include_genome_catalog": include_genome_catalog,
            "phase_timing": phase_timing,
            "denominator_phase_timing": False,
            "phase_timed_repetitions": repetitions if phase_timing else 0,
            "phase_timed_warmups": warmups if phase_timing else 0,
            "phase_interleave_order": (
                "alternating_exact_input_pairs" if phase_timing else "untimed_only"
            ),
            "occupancy_semantics": (
                "alive-bit distribution; every native slot carries one "
                "deterministic nonzero agent identity"
            ),
            "output": str(output),
        },
        "layout": _native_layout_projection(),
        "command": {
            "source": command_source,
            "argv": list(command_argv),
            "shell": shlex.join(command_argv),
        },
        "source_identity": {
            "repository": str(repo_root),
            "git": _git_identity(repo_root),
            "tool_path": str(Path(__file__).resolve()),
            "tool_file_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "torch_default_device": str(torch.get_default_device()),
            "scratch_parent": str(output.parent),
            "scratch_filesystem_device": scratch_device,
        },
        "timer": {
            "clock": "time.perf_counter_ns",
            "p95_method": "nearest_rank",
            "benchmark_elapsed_ns": time.perf_counter_ns() - benchmark_started_ns,
        },
        "cells": cells,
        "phase_aggregates_all_measured_cells": _phase_aggregates(
            all_phase_timed_samples
        ),
        "observer_overhead_all_measured_cells": _observer_overhead(
            all_measured_samples, all_phase_timed_samples
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--occupancy",
        action="append",
        choices=DEFAULT_OCCUPANCIES,
        help="repeat to select cells; default: zero, sparse, all",
    )
    parser.add_argument("--sparse-stride", type=int, default=16)
    parser.add_argument("--absolute-tick", type=int, default=1)
    parser.add_argument("--include-genome-catalog", action="store_true")
    parser.add_argument("--phase-timing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_argv)
    if argv is None and getattr(sys, "orig_argv", None):
        command_argv = list(sys.orig_argv)
        command_source = "runtime_sys_orig_argv"
    else:
        command_argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            *raw_argv,
        ]
        command_source = "synthesized_for_programmatic_main"
    report = run_snapshot_v1_host_baseline(
        output=args.output,
        batch_size=args.batch_size,
        repetitions=args.repetitions,
        warmups=args.warmups,
        occupancies=(
            tuple(args.occupancy) if args.occupancy is not None else DEFAULT_OCCUPANCIES
        ),
        sparse_stride=args.sparse_stride,
        absolute_tick=args.absolute_tick,
        include_genome_catalog=args.include_genome_catalog,
        phase_timing=args.phase_timing,
        command_argv=command_argv,
        command_source=command_source,
    )
    output = args.output.expanduser().resolve()
    atomic_write_json(output, report)
    print(
        json.dumps(
            {
                "ok": True,
                "schema_version": SCHEMA_VERSION,
                "claim_class": CLAIM_CLASS,
                "output": str(output),
                "output_file_sha256": _sha256_file(output),
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CLAIM_CLASS",
    "DEFAULT_OCCUPANCIES",
    "SCHEMA_VERSION",
    "main",
    "run_snapshot_v1_host_baseline",
)
