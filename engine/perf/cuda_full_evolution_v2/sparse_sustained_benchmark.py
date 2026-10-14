"""Sustained wall-clock benchmark for the exact sparse Full Evolution event path."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from time import perf_counter
from typing import Any, Final, Sequence

from .contract import (
    CUDA_NATIVE_TARGET_WTPS_100X,
    FULL_EVOLUTION_CAPABILITIES_V2,
    CudaFullEvolutionNativeContractV2,
)
from .native_loader import cuda_device_identity_v2, file_sha256
from .runtime import (
    FULL_CAPACITY_BENCHMARK_FIXTURE_V2,
    CudaFullEvolutionRuntimeV2,
    CudaFullEvolutionSparseEventDispatchV3,
    seal_sparse_event_sink_receipt_v3,
)

BENCHMARK_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_sparse_sustained_benchmark_v1"
)
ENVELOPE_SCHEMA_V1: Final[str] = "eiel_cuda_full_evolution_sparse_durable_envelope_v1"
ENVELOPE_MAGIC_V1: Final[bytes] = b"EIELSEV3"
CLAIM_CLASS: Final[str] = "TECHNICAL_ONLY"


class SparseSustainedBenchmarkError(RuntimeError):
    """Raised when a sustained measurement cannot remain authoritative."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise SparseSustainedBenchmarkError("durable envelope write stalled")
        written += count


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_sparse_envelope(
    dispatch: CudaFullEvolutionSparseEventDispatchV3,
    *,
    output_directory: Path,
    sample_id: str,
) -> dict[str, Any]:
    """Publish one immutable envelope with no replace/overwrite authority."""
    metadata = {
        "schema_version": ENVELOPE_SCHEMA_V1,
        "claim_class": CLAIM_CLASS,
        "sample_id": sample_id,
        "dispatch_schema_version": dispatch.schema_version,
        "dispatch_time_authority": dispatch.time_authority,
        "wire_abi": dispatch.wire_abi,
        "dispatch_content_sha256": dispatch.content_sha256,
        "world_uids": list(dispatch.world_uids),
        "start_next_absolute_tick": dispatch.start_next_absolute_tick,
        "end_next_absolute_tick": dispatch.end_next_absolute_tick,
        "tick_count": dispatch.tick_count,
        "event_counts": list(dispatch.event_counts),
        "final_next_agent_ids": list(dispatch.final_next_agent_ids),
        "world_offset_count": len(dispatch.world_offsets),
        "packed_event_count": dispatch.event_count,
        "packed_event_nbytes": len(dispatch.packed_events),
        "artifact_sha256": dispatch.artifact_sha256,
        "manifest_sha256": dispatch.manifest_sha256,
    }
    header = _canonical_json_bytes(metadata)
    offsets = b"".join(
        value.to_bytes(8, byteorder="little", signed=False)
        for value in dispatch.world_offsets
    )
    segments = (
        ENVELOPE_MAGIC_V1,
        len(header).to_bytes(8, byteorder="little", signed=False),
        header,
        offsets,
        dispatch.packed_events,
    )
    envelope_sha256 = hashlib.sha256()
    envelope_nbytes = 0
    for segment in segments:
        envelope_sha256.update(segment)
        envelope_nbytes += len(segment)

    final_path = output_directory / f"{sample_id}.sparse-events-v3.bin"
    temporary_path = output_directory / f".{sample_id}.{os.getpid()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary_path, flags, 0o400)
    linked = False
    try:
        for segment in segments:
            _write_all(descriptor, segment)
        os.fsync(descriptor)
        file_state = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_state.st_mode)
            or file_state.st_nlink != 1
            or file_state.st_size != envelope_nbytes
            or stat.S_IMODE(file_state.st_mode) & 0o222
        ):
            raise SparseSustainedBenchmarkError(
                "sealed sparse envelope temporary file is unsafe"
            )
        os.link(temporary_path, final_path, follow_symlinks=False)
        linked = True
        _fsync_directory(output_directory)
    finally:
        os.close(descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        if linked:
            _fsync_directory(output_directory)
    final_state = final_path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(final_state.st_mode)
        or final_state.st_nlink != 1
        or final_state.st_size != envelope_nbytes
        or stat.S_IMODE(final_state.st_mode) & 0o222
    ):
        raise SparseSustainedBenchmarkError(
            "published sparse envelope identity is unsafe"
        )
    return {
        "path": str(final_path.resolve()),
        "bytes": envelope_nbytes,
        "sha256": envelope_sha256.hexdigest(),
        "mode": oct(stat.S_IMODE(final_state.st_mode)),
        "device": final_state.st_dev,
        "inode": final_state.st_ino,
        "link_count": final_state.st_nlink,
    }


def _publish_result(result: dict[str, Any], output_directory: Path) -> Path:
    payload = json.dumps(result, sort_keys=True, indent=2).encode("ascii") + b"\n"
    final_path = output_directory / "result.json"
    temporary_path = output_directory / f".result.{os.getpid()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary_path, flags, 0o400)
    linked = False
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.link(temporary_path, final_path, follow_symlinks=False)
        linked = True
        _fsync_directory(output_directory)
    finally:
        os.close(descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        if linked:
            _fsync_directory(output_directory)
    return final_path


def _aggregate_counters(rows: list[dict[str, int]]) -> dict[str, int]:
    if not rows or any(tuple(row) != FULL_EVOLUTION_CAPABILITIES_V2 for row in rows):
        raise SparseSustainedBenchmarkError(
            "runtime did not expose the exact seven-capability counter identity"
        )
    return {
        capability: sum(row[capability] for row in rows)
        for capability in FULL_EVOLUTION_CAPABILITIES_V2
    }


def run_sparse_sustained_benchmark(
    *,
    artifact_path: Path,
    output_directory: Path,
    batch_size: int,
    temporal_k: int,
    samples: int,
    warmups: int,
    seed: int,
    execution_mode: str,
) -> dict[str, Any]:
    if batch_size <= 0 or not 1 <= temporal_k <= 64:
        raise SparseSustainedBenchmarkError(
            "B must be positive and K must be in [1,64]"
        )
    if samples < 1 or warmups < 0:
        raise SparseSustainedBenchmarkError(
            "samples must be positive and warmups must be nonnegative"
        )
    artifact_path = artifact_path.resolve(strict=True)
    if output_directory.exists():
        raise SparseSustainedBenchmarkError("output directory must not already exist")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(mode=0o700)
    _fsync_directory(output_directory.parent)

    import torch

    device_index = torch.cuda.current_device()
    device_identity = cuda_device_identity_v2(torch, device_index)
    initial_free_vram, total_vram = torch.cuda.mem_get_info(device_index)
    contract = CudaFullEvolutionNativeContractV2(
        B=batch_size,
        total_ticks=temporal_k,
        temporal_k=temporal_k,
    )
    fixture_owner = CudaFullEvolutionRuntimeV2.initialize(
        contract,
        seed=seed,
        artifact_path=artifact_path,
        execution_mode=execution_mode,
    )
    fixture_checkpoint = fixture_owner.force_full_capacity_benchmark_fixture_v2()
    if fixture_checkpoint.population_fixture_id != FULL_CAPACITY_BENCHMARK_FIXTURE_V2:
        raise SparseSustainedBenchmarkError("exact C512 fixture identity is missing")
    fixture_owner.close()
    del fixture_owner
    gc.collect()
    torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    total_runs = warmups + samples
    for ordinal in range(total_runs):
        measured = ordinal >= warmups
        sample_number = ordinal - warmups if measured else ordinal
        sample_id = (
            f"sample-{sample_number:03d}" if measured else f"warmup-{sample_number:03d}"
        )
        owner = CudaFullEvolutionRuntimeV2.from_checkpoint(
            fixture_checkpoint,
            artifact_path=artifact_path,
            execution_mode=execution_mode,
        )
        torch.cuda.reset_peak_memory_stats(device_index)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_started = perf_counter()
        start_event.record()
        owner.advance(temporal_k)
        end_event.record()
        dispatch = owner.drain_events_sparse_v3()
        publication_started = perf_counter()
        envelope = _publish_sparse_envelope(
            dispatch,
            output_directory=output_directory,
            sample_id=sample_id,
        )
        publication_elapsed_seconds = perf_counter() - publication_started
        receipt = seal_sparse_event_sink_receipt_v3(
            dispatch,
            sink_id="immutable-local-sparse-envelope-v1",
            sink_receipt_id=f"{Path(envelope['path']).name}:{envelope['sha256']}",
        )
        acknowledgement_started = perf_counter()
        owner.acknowledge_sparse_event_drain_v3(receipt)
        acknowledgement_elapsed_seconds = perf_counter() - acknowledgement_started
        wall_elapsed_seconds = perf_counter() - wall_started
        committed_world_ticks = batch_size * temporal_k
        if owner.advanced_ticks != temporal_k or owner.has_pending_dispatch:
            raise SparseSustainedBenchmarkError(
                "runtime did not commit the exact measured tick frontier"
            )
        counters = _aggregate_counters(owner.counters())
        if any(value <= 0 for value in counters.values()):
            raise SparseSustainedBenchmarkError(
                "C512 measurement did not exercise every Full Evolution capability"
            )
        row = {
            "sample_id": sample_id,
            "measured": measured,
            "committed_world_ticks": committed_world_ticks,
            "wall_elapsed_seconds": wall_elapsed_seconds,
            "committed_world_ticks_per_second": (
                committed_world_ticks / wall_elapsed_seconds
            ),
            "cuda_event_kernel_elapsed_ms_diagnostic_only": start_event.elapsed_time(
                end_event
            ),
            "event_count": dispatch.event_count,
            "events_per_world_tick": dispatch.event_count / committed_world_ticks,
            "sparse_wire_bytes": dispatch.wire_nbytes,
            "event_content_sha256": dispatch.content_sha256,
            "event_count_projection_seconds": (
                dispatch.count_projection_elapsed_seconds
            ),
            "event_compaction_transfer_validation_seconds": (
                dispatch.compaction_and_transfer_elapsed_seconds
            ),
            "event_content_hash_seconds": dispatch.content_hash_elapsed_seconds,
            "event_drain_seconds_including_kernel_wait": dispatch.drain_elapsed_seconds,
            "durable_publication_seconds": publication_elapsed_seconds,
            "acknowledgement_seconds": acknowledgement_elapsed_seconds,
            "sink_receipt_sha256": receipt.receipt_sha256,
            "durable_envelope": envelope,
            "capability_counters": counters,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
            "runtime_execution_receipt": owner.execution_plan_receipt_v2(),
        }
        rows.append(row)
        owner.close()
        del owner, dispatch, start_event, end_event
        gc.collect()
        torch.cuda.empty_cache()

    measured_rows = [row for row in rows if row["measured"]]
    throughputs = sorted(
        float(row["committed_world_ticks_per_second"]) for row in measured_rows
    )
    nearest_rank_p10 = throughputs[max(0, math.ceil(0.10 * len(throughputs)) - 1)]
    minimum = throughputs[0]
    target = max(float(CUDA_NATIVE_TARGET_WTPS_100X), 446_719.311_516)
    result = {
        "schema_version": BENCHMARK_SCHEMA_V1,
        "claim_class": CLAIM_CLASS,
        "status": (
            "FULL_TARGET_PASS" if minimum >= target else "FULL_TARGET_NOT_DEMONSTRATED"
        ),
        "target_world_ticks_per_second": target,
        "criterion": (
            "minimum measured wall-clock committed throughput from immediately before "
            "advance through durable envelope fsync+atomic publish+parent fsync, exact "
            "sink receipt seal, runtime rehash, and ACK must meet target"
        ),
        "candidate": {
            "batch_size": batch_size,
            "temporal_k": temporal_k,
            "execution_mode": execution_mode,
            "world_width": 32,
            "initial_population": 256,
            "resident_capacity": 512,
            "fixture_id": FULL_CAPACITY_BENCHMARK_FIXTURE_V2,
        },
        "samples": samples,
        "warmups": warmups,
        "minimum_committed_world_ticks_per_second": minimum,
        "p10_committed_world_ticks_per_second": nearest_rank_p10,
        "median_committed_world_ticks_per_second": (
            throughputs[len(throughputs) // 2]
            if len(throughputs) % 2
            else (
                throughputs[len(throughputs) // 2 - 1]
                + throughputs[len(throughputs) // 2]
            )
            / 2
        ),
        "maximum_committed_world_ticks_per_second": throughputs[-1],
        "all_seven_capabilities_positive_each_sample": all(
            all(value > 0 for value in row["capability_counters"].values())
            for row in measured_rows
        ),
        "full_evolution_capabilities": list(FULL_EVOLUTION_CAPABILITIES_V2),
        "absolute_tick_authority": True,
        "horizon_dependent_resident_bytes": 0,
        "event_transport": {
            "production_wire": "eiel_full_evolution_cuda_sparse_lifecycle_event_v3",
            "dense_v2_role": "tiny_gpu_differential_oracle_only",
            "durability": "single_envelope_fsync_atomic_no_replace_parent_fsync_then_ack",
        },
        "artifact": {
            "path": str(artifact_path),
            "sha256": file_sha256(artifact_path),
            "manifest_path": str(
                artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")
            ),
            "manifest_sha256": file_sha256(
                artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")
            ),
        },
        "fixture_checkpoint": {
            "canonical_state_sha256": fixture_checkpoint.canonical_state_sha256,
            "metadata_seal_sha256": fixture_checkpoint.metadata_seal_sha256,
            "population_fixture_id": fixture_checkpoint.population_fixture_id,
        },
        "device_identity": device_identity,
        "initial_vram": {
            "free_bytes": int(initial_free_vram),
            "total_bytes": int(total_vram),
        },
        "measurement_rows": rows,
    }
    result["result_body_sha256"] = hashlib.sha256(
        _canonical_json_bytes(result)
    ).hexdigest()
    result_path = _publish_result(result, output_directory)
    result["result_path"] = str(result_path.resolve())
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=896)
    parser.add_argument("--temporal-k", type=int, default=64)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--execution-mode",
        choices=("eager_monolith", "cuda_graph_monolith"),
        default="cuda_graph_monolith",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_sparse_sustained_benchmark(
        artifact_path=args.artifact,
        output_directory=args.output,
        batch_size=args.batch_size,
        temporal_k=args.temporal_k,
        samples=args.samples,
        warmups=args.warmups,
        seed=args.seed,
        execution_mode=args.execution_mode,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] == "FULL_TARGET_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
