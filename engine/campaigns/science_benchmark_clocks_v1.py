"""Observational three-clock algebra for science-representative FE benchmarks.

This module never instruments the simulation hot loop.  It derives T_SIM,
T_RELEASE, T_ACS_TAIL, and GPU-RELEASE A/B/C from shipped tick-complete,
segment-dispatch, and ACS capture-barrier receipts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from statistics import median
from typing import Any, Final

from .tick_native import TICK_NATIVE_EXECUTION_ROUTE_V1

SCIENCE_BENCHMARK_CLOCKS_SCHEMA_VERSION: Final[str] = (
    "eiel_science_benchmark_clocks_v1"
)
SCIENCE_PACKED_TICKS_V1: Final[int] = 5000
INTERNAL_LIFECYCLE_INTERVAL_TICKS_V1: Final[int] = 160
_GPU_IDLE_SCOPE_MARKERS: Final[tuple[str, ...]] = (
    "gpu-idle",
    "gpu_idle",
    "post-sync host capture time is a gpu-idle proxy",
    "post_sync_gpu_idle_proxy_time_ns",
)


class ScienceBenchmarkClockError(ValueError):
    """Shipped receipts cannot support the three-clock science benchmark."""


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScienceBenchmarkClockError(f"{field} must be an object")
    return dict(value)


def _int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ScienceBenchmarkClockError(f"{field} must be an integer >= {minimum}")
    return value


def _optional_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _int(value, field=field, minimum=0)


def _ns_to_seconds(value: int) -> float:
    return value / 1_000_000_000.0


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ScienceBenchmarkClockError("cannot summarize an empty sample set")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def terminal_acs_restore_validity_v1(batch_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Prove terminal ACS snapshot identity is digest-bound and published."""

    manifest = _mapping(batch_manifest, field="batch_manifest")
    result = _mapping(manifest.get("result"), field="batch_manifest.result")
    captures = result.get("acs_capture_measurements")
    if not isinstance(captures, list) or not captures:
        raise ScienceBenchmarkClockError("acs_capture_measurements are required")
    capture = _mapping(captures[-1], field="acs_capture_measurements[-1]")
    snapshot = capture.get("immutable_snapshot_digest_sha256s")
    if not isinstance(snapshot, Mapping) or not snapshot:
        raise ScienceBenchmarkClockError("immutable snapshot digests are required")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        checkpoint = _mapping(
            result.get("checkpoint_catalog"), field="checkpoint_catalog"
        )
        if checkpoint:
            checkpoint = _mapping(next(iter(checkpoint.values())), field="catalog entry")
    terminal = {}
    bundles = []
    if isinstance(checkpoint, Mapping):
        raw_terminal = checkpoint.get("terminal_projection_digest_sha256s") or checkpoint.get(
            "terminal_cell_digests"
        )
        if isinstance(raw_terminal, Mapping):
            terminal = dict(raw_terminal)
        raw_bundles = checkpoint.get("cell_bundles")
        if isinstance(raw_bundles, list):
            bundles = [row for row in raw_bundles if isinstance(row, Mapping)]
    mismatches = sorted(
        name
        for name, digest in snapshot.items()
        if terminal.get(name) not in (None, digest)
    )
    missing_bundles = [
        name
        for name in snapshot
        if not any(str(row.get("cell_id")) == str(name) for row in bundles)
    ]
    passed = not mismatches and (not bundles or not missing_bundles)
    return {
        "status": "PASS" if passed else "FAIL",
        "snapshot_cells": len(snapshot),
        "bundle_count": len(bundles),
        "digest_mismatches": mismatches,
        "missing_bundles": missing_bundles,
    }


def packed_tick_horizon_v1(plan: Mapping[str, Any] | Any) -> int:
    """Return the public packed-tick horizon from a compiled plan or payload."""

    payload = plan.payload(include_hash=False) if hasattr(plan, "payload") else plan
    execution = _mapping(payload.get("execution"), field="execution")
    if execution.get("execution_route_identity") == TICK_NATIVE_EXECUTION_ROUTE_V1:
        return _int(execution.get("total_ticks"), field="execution.total_ticks", minimum=1)
    generations = _int(execution.get("generations"), field="execution.generations", minimum=1)
    ticks = _int(
        execution.get("ticks_per_generation"),
        field="execution.ticks_per_generation",
        minimum=1,
    )
    return generations * ticks


def terminal_only_acs_policy_v1(plan: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Return whether compiled ACS/checkpoint policy is terminal-only."""

    payload = plan.payload(include_hash=False) if hasattr(plan, "payload") else plan
    execution = _mapping(payload.get("execution"), field="execution")
    if execution.get("execution_route_identity") == TICK_NATIVE_EXECUTION_ROUTE_V1:
        total_ticks = _int(
            execution.get("total_ticks"), field="execution.total_ticks", minimum=1
        )
        checkpoint_ticks = execution.get("checkpoint_ticks")
        if not isinstance(checkpoint_ticks, list) or checkpoint_ticks != [total_ticks]:
            terminal = False
        else:
            # Native checkpoint authority is already absolute-tick based.  A
            # generation digest switch must not be required or serialized.
            terminal = True
        interior = (
            []
            if not isinstance(checkpoint_ticks, list)
            else [tick for tick in checkpoint_ticks if tick != total_ticks]
        )
        return {
            "terminal_only": terminal,
            "digest_every_generation": None,
            "checkpoint_ticks": list(checkpoint_ticks or []),
            "interior_checkpoint_ticks": interior,
            "horizon_packed_ticks": total_ticks,
        }
    digest = execution.get("digest_every_generation")
    if type(digest) is not bool:
        raise ScienceBenchmarkClockError(
            "execution.digest_every_generation must be boolean"
        )
    generations = _int(execution.get("generations"), field="execution.generations", minimum=1)
    milestones = execution.get("checkpoints")
    if not isinstance(milestones, list):
        raise ScienceBenchmarkClockError("execution.checkpoints must be an array")
    interior = [item for item in milestones if item != generations]
    ticks = _int(
        execution.get("ticks_per_generation"),
        field="execution.ticks_per_generation",
        minimum=1,
    )
    return {
        "terminal_only": digest is False and milestones == [generations],
        "digest_every_generation": digest,
        "checkpoint_ticks": [int(item) * ticks for item in milestones if type(item) is int],
        "interior_checkpoint_ticks": interior,
        "horizon_packed_ticks": generations * ticks,
    }


def _gpu_memory_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    keys = ("memory_allocated_bytes", "memory_reserved_bytes", "max_memory_allocated_bytes")
    return all(before.get(name) == after.get(name) for name in keys)


def _scope_says_gpu_idle(capture: Mapping[str, Any]) -> bool:
    scope = str(capture.get("measurement_scope") or "").lower()
    limitations = capture.get("limitations")
    limitation_text = " ".join(
        str(item).lower() for item in limitations if isinstance(limitations, list)
    )
    blob = f"{scope} {limitation_text}"
    return any(marker in blob for marker in _GPU_IDLE_SCOPE_MARKERS)


def _stage_seconds(rows: Mapping[str, Mapping[str, Any]], name: str) -> float | None:
    values: list[float] = []
    for telemetry in rows.values():
        if not isinstance(telemetry, Mapping):
            continue
        raw = telemetry.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        values.append(float(raw))
    if not values:
        return None
    return sum(values)


def _pause_counts(window_seconds: Sequence[float]) -> dict[str, int]:
    return {
        "pauses_gt_100ms": sum(item > 0.1 for item in window_seconds),
        "pauses_gt_1s": sum(item > 1.0 for item in window_seconds),
        "pauses_gt_10s": sum(item > 10.0 for item in window_seconds),
    }


def _gpu_sample_unix_ns(sample: Mapping[str, Any], *, index: int) -> int:
    raw_unix_ns = sample.get("observed_unix_ns")
    if type(raw_unix_ns) is int and raw_unix_ns >= 0:
        return raw_unix_ns
    observed_at = sample.get("observed_at")
    if isinstance(observed_at, str):
        try:
            parsed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ScienceBenchmarkClockError(
                f"gpu_samples[{index}].observed_at is not ISO-8601"
            ) from exc
        if parsed.tzinfo is None:
            raise ScienceBenchmarkClockError(
                f"gpu_samples[{index}].observed_at must include a timezone"
            )
        delta = parsed.astimezone(timezone.utc) - datetime(
            1970, 1, 1, tzinfo=timezone.utc
        )
        return (
            (delta.days * 86_400 + delta.seconds) * 1_000_000_000
            + delta.microseconds * 1_000
        )
    raise ScienceBenchmarkClockError(
        f"gpu_samples[{index}] requires observed_unix_ns or timezone-bound observed_at"
    )


def _gpu_metric_summary(
    samples: Sequence[Mapping[str, Any]], *fields: str
) -> dict[str, float | int] | None:
    values: list[float] = []
    for sample in samples:
        raw = next(
            (sample.get(field) for field in fields if sample.get(field) is not None),
            None,
        )
        if not isinstance(raw, bool) and isinstance(raw, (int, float)):
            values.append(float(raw))
    if not values:
        return None
    return {
        "mean": sum(values) / len(values),
        "maximum": max(values),
        "sample_count": len(values),
    }


def derive_science_benchmark_clocks_v1(
    *,
    packed_ticks: int,
    packed_worlds: int,
    batch_manifest: Mapping[str, Any],
    gpu_samples: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Derive non-overlapping T_SIM / T_RELEASE / T_ACS_TAIL from shipped receipts."""

    packed_ticks = _int(packed_ticks, field="packed_ticks", minimum=1)
    packed_worlds = _int(packed_worlds, field="packed_worlds", minimum=1)
    manifest = _mapping(batch_manifest, field="batch_manifest")
    result = _mapping(manifest.get("result"), field="batch_manifest.result")
    receipts = result.get("native_execution_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ScienceBenchmarkClockError("native_execution_receipts are required")
    starts: list[int] = []
    stops: list[int] = []
    window_seconds: list[float] = []
    window_ticks: list[int] = []
    host_control_calls = 0
    status_d2h = 0
    event_reclaim_copies = 0
    host_rng_generation = 0
    future_rng_h2d = 0
    tape_bytes = 0
    graph_replays = 0
    graph_recaptures = 0
    for index, raw in enumerate(receipts):
        receipt = _mapping(raw, field=f"native_execution_receipts[{index}]")
        started = receipt.get("segment_dispatch_started_unix_ns")
        finished = receipt.get("segment_dispatch_finished_unix_ns")
        if type(started) is not int or type(finished) is not int:
            raise ScienceBenchmarkClockError(
                "segment dispatch unix timestamps are required on native receipts"
            )
        if finished < started:
            raise ScienceBenchmarkClockError("segment dispatch timestamps overlap backwards")
        starts.append(started)
        stops.append(finished)
        start_tick = receipt.get("start_next_tick")
        end_tick = receipt.get("end_next_tick")
        if type(start_tick) is int and type(end_tick) is int and end_tick > start_tick:
            ticks = end_tick - start_tick
        else:
            start_generation = receipt.get("start_generation")
            end_generation = receipt.get("end_generation")
            if type(start_generation) is int and type(end_generation) is int:
                ticks = max(0, end_generation - start_generation) * (
                    packed_ticks if packed_ticks < INTERNAL_LIFECYCLE_INTERVAL_TICKS_V1 else INTERNAL_LIFECYCLE_INTERVAL_TICKS_V1
                )
            else:
                ticks = 0
        if ticks > 0:
            window_ticks.append(ticks)
            window_seconds.append(_ns_to_seconds(finished - started))
        host_rng_generation += _int(
            receipt.get("gaussian_segment_refill_calls_inside_tick_loop") or 0,
            field="gaussian_segment_refill_calls_inside_tick_loop",
        )
        future_rng_h2d += _int(
            receipt.get("intermediate_host_to_device_operations") or 0,
            field="intermediate_host_to_device_operations",
        )
        future_rng_h2d += _int(
            receipt.get("gaussian_segment_host_to_device_operations_lower_bound") or 0,
            field="gaussian_segment_host_to_device_operations_lower_bound",
        )
        status_d2h += _int(
            receipt.get("intermediate_device_to_host_operations") or 0,
            field="intermediate_device_to_host_operations",
        )
        graph_replays += _int(receipt.get("graph_replays") or 0, field="graph_replays")
        graph_recaptures += _int(
            receipt.get("graph_recaptures") or 0, field="graph_recaptures"
        )
        host_control_calls += _int(
            receipt.get("hot_loop_host_control_calls") or 0,
            field="hot_loop_host_control_calls",
        )
        event_reclaim_copies += _int(
            receipt.get("hot_loop_event_reclaim_copies") or 0,
            field="hot_loop_event_reclaim_copies",
        )
        tape_bytes += _int(receipt.get("tape_bytes") or 0, field="tape_bytes")
    t_sim_start_unix_ns = min(starts)
    t_sim_stop_unix_ns = max(stops)
    t_sim_ns = t_sim_stop_unix_ns - t_sim_start_unix_ns
    if t_sim_ns <= 0:
        raise ScienceBenchmarkClockError("T_SIM must be positive")

    captures = result.get("acs_capture_measurements")
    if not isinstance(captures, list) or not captures:
        raise ScienceBenchmarkClockError("acs_capture_measurements are required")
    capture = _mapping(captures[-1], field="acs_capture_measurements[-1]")
    publications = result.get("acs_catalog_publication_measurements")
    if not isinstance(publications, list) or not publications:
        raise ScienceBenchmarkClockError(
            "acs_catalog_publication_measurements are required"
        )
    publication = _mapping(
        publications[-1], field="acs_catalog_publication_measurements[-1]"
    )
    staging = _mapping(
        result.get("acs_checkpoint_staging"), field="result.acs_checkpoint_staging"
    )
    gpu_before = _mapping(
        capture.get("gpu_resources_before"), field="gpu_resources_before"
    )
    gpu_after = _mapping(
        capture.get("gpu_resources_after"), field="gpu_resources_after"
    )
    host_before = _mapping(
        capture.get("host_resources_before"), field="host_resources_before"
    )
    host_after = _mapping(
        capture.get("host_resources_after"), field="host_resources_after"
    )
    transfer = _mapping(
        capture.get("native_transfer_receipt"), field="native_transfer_receipt"
    )
    capture_completed = _optional_int(
        capture.get("capture_completed_unix_ns"),
        field="capture_completed_unix_ns",
    )
    explicitly_safe_release = _optional_int(
        capture.get("gpu_safely_releasable_unix_ns"),
        field="gpu_safely_releasable_unix_ns",
    )
    if (
        explicitly_safe_release is not None
        and capture_completed is not None
        and explicitly_safe_release < capture_completed
    ):
        raise ScienceBenchmarkClockError(
            "declared GPU-safe release precedes bounded capture completion"
        )
    durable_completed = _optional_int(
        publication.get("durable_completed_unix_ns"),
        field="durable_completed_unix_ns",
    )
    if durable_completed is None:
        raise ScienceBenchmarkClockError("ACS durable unix timestamp is required")
    if durable_completed < t_sim_stop_unix_ns:
        raise ScienceBenchmarkClockError("ACS durable completion precedes T_SIM stop")

    slots = _int(staging.get("slots") or 0, field="acs_checkpoint_staging.slots")
    writer_busy_ns = _int(
        staging.get("writer_busy_time_ns") or 0,
        field="writer_busy_time_ns",
    )
    freeze_ns = _int(
        capture.get("host_capture_freeze_time_ns") or 0,
        field="host_capture_freeze_time_ns",
    )
    final_sync_ns = _int(
        capture.get("native_final_stream_synchronization_time_ns") or 0,
        field="native_final_stream_synchronization_time_ns",
    )
    final_d2h = _int(
        transfer.get("final_device_to_host_operations") or 0,
        field="final_device_to_host_operations",
    )
    async_persist = slots >= 1
    gpu_idle_proxy = _scope_says_gpu_idle(capture)
    memory_unchanged = _gpu_memory_unchanged(gpu_before, gpu_after)

    if explicitly_safe_release is not None:
        decision = "B"
        gpu_release_unix_ns = explicitly_safe_release
        dependency = (
            "bounded terminal capture receipt explicitly proves GPU-safe release "
            "after immutable detachment and before asynchronous persistence completes"
        )
    elif async_persist and gpu_idle_proxy and memory_unchanged:
        decision = "A"
        gpu_release_unix_ns = t_sim_stop_unix_ns
        dependency = (
            "immutable host snapshot is digest-bound after final stream sync; "
            "ACS persist is CPU/storage via ACSCheckpointStaging"
        )
    elif (final_d2h > 0 or freeze_ns > 0) and capture_completed is not None:
        decision = "B"
        gpu_release_unix_ns = capture_completed
        dependency = (
            "bounded GPU capture/D2H or host freeze after the final tick; "
            "release after capture_completed_unix_ns"
        )
    else:
        decision = "C"
        gpu_release_unix_ns = durable_completed
        dependency = (
            "ACS persistence evidence does not prove GPU detach before durable "
            f"completion (slots={slots}, writer_busy_ns={writer_busy_ns})"
        )
    if gpu_release_unix_ns < t_sim_stop_unix_ns:
        raise ScienceBenchmarkClockError("GPU release precedes T_SIM stop")
    if durable_completed < gpu_release_unix_ns:
        raise ScienceBenchmarkClockError("ACS durable completion precedes GPU release")

    t_release_ns = gpu_release_unix_ns - t_sim_stop_unix_ns
    t_acs_tail_ns = durable_completed - gpu_release_unix_ns
    t_job_ns = t_sim_ns + t_release_ns + t_acs_tail_ns
    if t_job_ns != (durable_completed - t_sim_start_unix_ns):
        # Allow a capture-only gap already classified into T_RELEASE/T_ACS_TAIL.
        span_ns = durable_completed - t_sim_start_unix_ns
        if t_job_ns != span_ns:
            raise ScienceBenchmarkClockError(
                "T_JOB must equal T_SIM + T_RELEASE + T_ACS_TAIL and the unix span"
            )

    t_sim_s = _ns_to_seconds(t_sim_ns)
    seconds_per_1000 = t_sim_s * 1000.0 / packed_ticks
    ticks_per_second = packed_ticks / t_sim_s
    world_ticks_per_second = packed_worlds * ticks_per_second

    window_stats: dict[str, float | int | None]
    if window_seconds:
        ordered = sorted(window_seconds)
        per_tick = [
            seconds / ticks
            for seconds, ticks in zip(window_seconds, window_ticks)
            if ticks > 0
        ]
        ordered_ticks = sorted(per_tick) if per_tick else ordered
        window_stats = {
            "sample_count": len(ordered_ticks),
            "median_seconds": float(median(ordered_ticks)),
            "p95_seconds": _percentile(ordered_ticks, 0.95),
            "p99_seconds": _percentile(ordered_ticks, 0.99),
            "slowest_seconds": float(ordered_ticks[-1]),
            "pause_counts": _pause_counts(window_seconds),
            "limitation": (
                None
                if len(ordered_ticks) >= packed_ticks
                else "per-tick CUDA-event samples were not retained; stats are observational segment windows"
            ),
        }
    else:
        window_stats = {
            "sample_count": 0,
            "median_seconds": None,
            "p95_seconds": None,
            "p99_seconds": None,
            "slowest_seconds": None,
            "pause_counts": None,
            "limitation": "no observational segment windows were present",
        }

    checkpoint = manifest.get("checkpoint")
    stage_rows: dict[str, Mapping[str, Any]] = {}
    if isinstance(checkpoint, Mapping):
        staging_block = checkpoint.get("acs_checkpoint_staging")
        if isinstance(staging_block, Mapping):
            raw_stages = staging_block.get("cell_stage_telemetry_seconds")
            if isinstance(raw_stages, Mapping):
                stage_rows = {
                    str(name): row
                    for name, row in raw_stages.items()
                    if isinstance(row, Mapping)
                }
    checkpoint_bytes = None
    if isinstance(checkpoint, Mapping):
        staging_block = checkpoint.get("acs_checkpoint_staging")
        if isinstance(staging_block, Mapping) and type(
            staging_block.get("checkpoint_bytes")
        ) is int:
            checkpoint_bytes = staging_block["checkpoint_bytes"]
    if checkpoint_bytes is None and type(staging.get("checkpoint_bytes")) is int:
        checkpoint_bytes = staging["checkpoint_bytes"]

    gpu_util = None
    gpu_power = None
    primary_gpu_samples: list[Mapping[str, Any]] = []
    post_simulation_gpu_samples: list[Mapping[str, Any]] = []
    out_of_scope_gpu_sample_count = 0
    if gpu_samples:
        for index, sample in enumerate(gpu_samples):
            if not isinstance(sample, Mapping):
                raise ScienceBenchmarkClockError(f"gpu_samples[{index}] must be an object")
            sample_unix_ns = _gpu_sample_unix_ns(sample, index=index)
            if t_sim_start_unix_ns <= sample_unix_ns <= gpu_release_unix_ns:
                primary_gpu_samples.append(sample)
            elif gpu_release_unix_ns < sample_unix_ns <= durable_completed:
                post_simulation_gpu_samples.append(sample)
            else:
                out_of_scope_gpu_sample_count += 1
        util_summary = _gpu_metric_summary(
            primary_gpu_samples, "utilization_percent", "sm_percent"
        )
        power_summary = _gpu_metric_summary(
            primary_gpu_samples, "power_watts", "power_draw_w"
        )
        if util_summary:
            gpu_util = {
                "mean_percent": util_summary["mean"],
                "maximum_percent": util_summary["maximum"],
                "sample_count": util_summary["sample_count"],
            }
        if power_summary:
            gpu_power = {
                "mean_watts": power_summary["mean"],
                "maximum_watts": power_summary["maximum"],
                "sample_count": power_summary["sample_count"],
            }

    tail_gpu_util = _gpu_metric_summary(
        post_simulation_gpu_samples, "utilization_percent", "sm_percent"
    )
    tail_gpu_power = _gpu_metric_summary(
        post_simulation_gpu_samples, "power_watts", "power_draw_w"
    )

    rss_values = [
        value
        for value in (host_before.get("rss_bytes"), host_after.get("rss_bytes"))
        if type(value) is int
    ]
    return {
        "schema_version": SCIENCE_BENCHMARK_CLOCKS_SCHEMA_VERSION,
        "packed_ticks": packed_ticks,
        "packed_worlds": packed_worlds,
        "t_sim_unix_ns": {
            "start": t_sim_start_unix_ns,
            "stop": t_sim_stop_unix_ns,
        },
        "t_release_unix_ns": {
            "start": t_sim_stop_unix_ns,
            "stop": gpu_release_unix_ns,
        },
        "t_acs_tail_unix_ns": {
            "start": gpu_release_unix_ns,
            "stop": durable_completed,
        },
        "T_SIM_seconds": t_sim_s,
        "T_RELEASE_seconds": _ns_to_seconds(t_release_ns),
        "T_ACS_TAIL_seconds": _ns_to_seconds(t_acs_tail_ns),
        "T_JOB_seconds": _ns_to_seconds(t_job_ns),
        "SCIENCE_SECONDS_PER_1000_TICKS": seconds_per_1000,
        "SCIENCE_TICKS_PER_SECOND": ticks_per_second,
        "SCIENCE_WORLD_TICKS_PER_SECOND": world_ticks_per_second,
        "window_stats": window_stats,
        "graph_replays": graph_replays,
        "graph_recaptures": graph_recaptures,
        "host_rng_generation": host_rng_generation,
        "future_rng_h2d": future_rng_h2d,
        "tape_bytes": tape_bytes,
        "hot_loop_host_control_calls": host_control_calls,
        "hot_loop_status_d2h": status_d2h,
        "hot_loop_event_reclaim_copies": event_reclaim_copies,
        "gpu_utilization": gpu_util,
        "gpu_power": gpu_power,
        "gpu_performance_window": {
            "start_unix_ns": t_sim_start_unix_ns,
            "stop_unix_ns": gpu_release_unix_ns,
            "duration_seconds": _ns_to_seconds(
                gpu_release_unix_ns - t_sim_start_unix_ns
            ),
            "sample_count": len(primary_gpu_samples),
            "post_simulation_acs_tail_excluded": True,
        },
        "post_simulation_acs_tail_gpu_activity": {
            "start_unix_ns": gpu_release_unix_ns,
            "stop_unix_ns": durable_completed,
            "duration_seconds": _ns_to_seconds(t_acs_tail_ns),
            "sample_count": len(post_simulation_gpu_samples),
            "gpu_utilization": (
                {
                    "mean_percent": tail_gpu_util["mean"],
                    "maximum_percent": tail_gpu_util["maximum"],
                    "sample_count": tail_gpu_util["sample_count"],
                }
                if tail_gpu_util
                else None
            ),
            "gpu_power": (
                {
                    "mean_watts": tail_gpu_power["mean"],
                    "maximum_watts": tail_gpu_power["maximum"],
                    "sample_count": tail_gpu_power["sample_count"],
                }
                if tail_gpu_power
                else None
            ),
            "excluded_from_gpu_performance": True,
        },
        "gpu_samples_outside_job_window": out_of_scope_gpu_sample_count,
        "peak_vram": {
            "allocated_bytes": gpu_after.get("memory_allocated_bytes"),
            "reserved_bytes": gpu_after.get("memory_reserved_bytes"),
            "max_allocated_bytes": gpu_after.get("max_memory_allocated_bytes"),
        },
        "GPU_RELEASE_DECISION": decision,
        "gpu_release_dependency": dependency,
        "gpu_safely_releasable_unix_ns": gpu_release_unix_ns,
        "final_gpu_use_unix_ns": t_sim_stop_unix_ns,
        "final_tick_unix_ns": t_sim_stop_unix_ns,
        "immutable_snapshot_unix_ns": capture_completed,
        "acs_durable_unix_ns": durable_completed,
        "terminal_acs": {
            "snapshot_capture_seconds": _ns_to_seconds(freeze_ns + final_sync_ns),
            "gpu_dependent_tail_seconds": _ns_to_seconds(t_release_ns),
            "d2h_bytes": transfer.get("final_transfer_bytes"),
            "host_rss_before_bytes": host_before.get("rss_bytes"),
            "host_rss_after_bytes": host_after.get("rss_bytes"),
            "host_rss_peak_bytes": max(rss_values) if rss_values else None,
            "serialization_seconds": _stage_seconds(stage_rows, "checkpoint_serialize"),
            "compression_seconds": _stage_seconds(stage_rows, "checkpoint_projection_gzip"),
            "hashing_seconds": _stage_seconds(stage_rows, "checkpoint_hash"),
            "filesystem_seconds": _stage_seconds(stage_rows, "checkpoint_write"),
            "fsync_seconds": _stage_seconds(stage_rows, "checkpoint_staged_tree_fsync"),
            "durable_completion_seconds": _ns_to_seconds(t_acs_tail_ns),
            "checkpoint_bytes": checkpoint_bytes,
            "bundle_count": (
                len(checkpoint.get("cell_bundles"))
                if isinstance(checkpoint, Mapping)
                and isinstance(checkpoint.get("cell_bundles"), list)
                else capture.get("captured_cell_count")
            ),
            "captured_bytes": capture.get("captured_bytes"),
            "writer_busy_seconds": _ns_to_seconds(writer_busy_ns),
            "final_stream_sync_seconds": _ns_to_seconds(final_sync_ns),
        },
        "estimates_from_t_sim": {
            "ticks_10000_seconds": seconds_per_1000 * 10,
            "ticks_50000_seconds": seconds_per_1000 * 50,
            "ticks_100000_seconds": seconds_per_1000 * 100,
            "terminal_acs_separate_seconds": _ns_to_seconds(t_release_ns + t_acs_tail_ns),
        },
        "non_overlapping": True,
        "t_job_is_not_tick_throughput": True,
    }


def render_science_benchmark_report_v1(
    *,
    status: str,
    qualified_science_route: str,
    b_w_p_c: str,
    source_sha: str,
    artifact_sha: str,
    clocks: Mapping[str, Any],
    restore_validity: str,
    interpretation: Sequence[str],
    b140_unused_reason: str,
) -> str:
    """Render the prompt-required human report from derived clocks."""

    payload = _mapping(clocks, field="clocks")
    windows = _mapping(payload.get("window_stats"), field="window_stats")
    pauses = windows.get("pause_counts") or {}
    gpu_util = payload.get("gpu_utilization") or {}
    gpu_power = payload.get("gpu_power") or {}
    peak = _mapping(payload.get("peak_vram"), field="peak_vram")
    acs = _mapping(payload.get("terminal_acs"), field="terminal_acs")
    lines = [
        f"BENCHMARK STATUS: {status}",
        f"QUALIFIED SCIENCE ROUTE: {qualified_science_route}",
        f"B/W/P/C: {b_w_p_c}",
        f"SOURCE SHA: {source_sha}",
        f"ARTIFACT SHA: {artifact_sha}",
        "",
        "SCIENCE PERFORMANCE",
        f"ticks: {payload.get('packed_ticks')}",
        f"T_SIM: {payload.get('T_SIM_seconds')}",
        f"seconds/1000 ticks: {payload.get('SCIENCE_SECONDS_PER_1000_TICKS')}",
        f"ticks/s: {payload.get('SCIENCE_TICKS_PER_SECOND')}",
        f"world-ticks/s: {payload.get('SCIENCE_WORLD_TICKS_PER_SECOND')}",
        f"median: {windows.get('median_seconds')}",
        f"p95: {windows.get('p95_seconds')}",
        f"p99: {windows.get('p99_seconds')}",
        f"slowest: {windows.get('slowest_seconds')}",
        f"pause counts: {pauses}",
        f"GPU utilization: {gpu_util}",
        f"GPU power: {gpu_power}",
        f"peak VRAM: {peak}",
        "",
        "TERMINAL GPU RELEASE",
        f"T_RELEASE: {payload.get('T_RELEASE_seconds')}",
        f"GPU_RELEASE_DECISION: {payload.get('GPU_RELEASE_DECISION')}",
        f"GPU safely releasable at: {payload.get('gpu_safely_releasable_unix_ns')}",
        f"final GPU use at: {payload.get('final_gpu_use_unix_ns')}",
        "",
        "TERMINAL ACS",
        f"T_ACS_TAIL: {payload.get('T_ACS_TAIL_seconds')}",
        f"checkpoint bytes: {acs.get('checkpoint_bytes')}",
        f"host RSS peak: {acs.get('host_rss_peak_bytes')}",
        f"serialization: {acs.get('serialization_seconds')}",
        f"compression: {acs.get('compression_seconds')}",
        f"hashing: {acs.get('hashing_seconds')}",
        f"filesystem: {acs.get('filesystem_seconds')}",
        f"durable completion: {acs.get('durable_completion_seconds')}",
        f"restore-validity: {restore_validity}",
        "",
        "TOTAL JOB",
        f"T_JOB: {payload.get('T_JOB_seconds')}",
        "",
        "B140 UNUSED: " + b140_unused_reason,
        "",
        "INTERPRETATION",
    ]
    lines.extend(f"- {item}" for item in interpretation)
    return "\n".join(lines) + "\n"


def extrapolate_horizon_seconds_v1(
    clocks: Mapping[str, Any], *, packed_ticks: int
) -> float:
    """Return estimated wall for a longer horizon using T_SIM only, plus ACS."""

    payload = _mapping(clocks, field="clocks")
    per_1000 = payload.get("SCIENCE_SECONDS_PER_1000_TICKS")
    if isinstance(per_1000, bool) or not isinstance(per_1000, (int, float)):
        raise ScienceBenchmarkClockError("SCIENCE_SECONDS_PER_1000_TICKS is missing")
    acs = _mapping(payload.get("estimates_from_t_sim"), field="estimates_from_t_sim")
    acs_seconds = acs.get("terminal_acs_separate_seconds")
    if isinstance(acs_seconds, bool) or not isinstance(acs_seconds, (int, float)):
        acs_seconds = 0.0
    return float(per_1000) * (packed_ticks / 1000.0) + float(acs_seconds)
