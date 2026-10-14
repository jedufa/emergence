"""Fail-closed GPU admission and process-attributed telemetry for ACS runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Protocol


ACS_GPU_TELEMETRY_SCHEMA_VERSION = "eiel_acs_gpu_telemetry_receipt_v1"
ACS_GPU_SAMPLE_STREAM_SCHEMA_VERSION = "eiel_acs_gpu_sample_stream_v1"
ACS_GPU_ACTIVITY_SUMMARY_SCHEMA_VERSION = "eiel_acs_gpu_activity_summary_v1"
_CANONICAL_GPU_PERFORMANCE_SCOPE_V1 = (
    "first_simulation_dispatch_through_terminal_gpu_safe_release_v1"
)
class ACSGPUGateError(RuntimeError):
    """The GPU gate could not produce trustworthy telemetry."""


class GPUSampler(Protocol):
    def sample(self) -> Mapping[str, Any]: ...


class GuardedProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class _SampleStreamV1:
    """Append-only telemetry that remains useful after interruption or OOM."""

    def __init__(self, path: Path, *, header: Mapping[str, Any]):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._handle = self.path.open("x", encoding="utf-8")
        except OSError as exc:
            raise ACSGPUGateError(f"cannot create GPU sample stream: {exc}") from exc
        self._sequence = 0
        self._write(
            {
                "schema_version": ACS_GPU_SAMPLE_STREAM_SCHEMA_VERSION,
                "record_type": "header",
                **dict(header),
            }
        )

    def _write(self, value: Mapping[str, Any]) -> None:
        try:
            self._handle.write(
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            )
            self._handle.flush()
        except OSError as exc:
            raise ACSGPUGateError(f"cannot persist GPU sample stream: {exc}") from exc

    def append(self, *, phase: str, sample: Mapping[str, Any]) -> None:
        self._write(
            {
                "schema_version": ACS_GPU_SAMPLE_STREAM_SCHEMA_VERSION,
                "record_type": "sample",
                "sequence": self._sequence,
                "phase": phase,
                "sample": dict(sample),
            }
        )
        self._sequence += 1

    def finish(self, *, status: str, summary: Mapping[str, Any] | None) -> None:
        self._write(
            {
                "schema_version": ACS_GPU_SAMPLE_STREAM_SCHEMA_VERSION,
                "record_type": "terminal",
                "status": status,
                "sample_count": self._sequence,
                "post_analysis_summary": dict(summary) if summary is not None else None,
            }
        )

    def close(self) -> None:
        try:
            os.fsync(self._handle.fileno())
        finally:
            self._handle.close()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil((percentile / 100.0) * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _metric_summary(
    samples: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    values = [float(sample[field]) for sample in samples if sample.get(field) is not None]
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "max": max(values),
    }


def summarize_gpu_activity_v1(
    samples: Sequence[Mapping[str, Any]],
    *,
    requested_sample_interval_seconds: float,
    idle_sm_threshold_percent: float,
) -> dict[str, Any]:
    """Summarize sampled GPU activity without claiming unsampled intervals."""

    timestamps = [int(sample["monotonic_time_ns"]) for sample in samples]
    intervals = [
        (later - earlier) / 1_000_000_000
        for earlier, later in zip(timestamps, timestamps[1:], strict=False)
    ]
    active = [
        float(sample["sm_percent"]) > idle_sm_threshold_percent for sample in samples
    ]
    runs: list[dict[str, Any]] = []
    if samples:
        start = 0
        for index in range(1, len(samples) + 1):
            if index == len(samples) or active[index] != active[start]:
                runs.append(
                    {
                        "active": active[start],
                        "sample_count": index - start,
                        "observed_span_seconds": (
                            timestamps[index - 1] - timestamps[start]
                        )
                        / 1_000_000_000,
                    }
                )
                start = index
    active_runs = [run for run in runs if run["active"]]
    idle_runs = [run for run in runs if not run["active"]]
    count = len(samples)
    active_count = sum(active)
    return {
        "schema_version": ACS_GPU_ACTIVITY_SUMMARY_SCHEMA_VERSION,
        "sample_count": count,
        "requested_sample_interval_seconds": requested_sample_interval_seconds,
        "observed_span_seconds": (
            (timestamps[-1] - timestamps[0]) / 1_000_000_000 if count > 1 else 0.0
        ),
        "observed_interval_seconds": {
            "mean": sum(intervals) / len(intervals) if intervals else None,
            "p50": _percentile(intervals, 50.0),
            "p95": _percentile(intervals, 95.0),
            "max": max(intervals) if intervals else None,
        },
        "gpu_sm_percent": _metric_summary(samples, "sm_percent"),
        "memory_used_mib": _metric_summary(samples, "memory_used_mib"),
        "power_draw_w": _metric_summary(samples, "power_draw_w"),
        "idle_sm_threshold_percent": idle_sm_threshold_percent,
        "active_sample_count": active_count,
        "active_sample_fraction": active_count / count if count else None,
        "idle_sample_count": count - active_count,
        "idle_sample_fraction": (count - active_count) / count if count else None,
        "active_burst_count": len(active_runs),
        "longest_active_burst_observed_span_seconds": max(
            (run["observed_span_seconds"] for run in active_runs), default=None
        ),
        "idle_gap_count": len(idle_runs),
        "longest_idle_gap_observed_span_seconds": max(
            (run["observed_span_seconds"] for run in idle_runs), default=None
        ),
        "interpretation": (
            "Burst and gap spans are lower bounds between observed samples; "
            "nvidia-smi sampling can miss activity shorter than the measured cadence."
        ),
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _percent(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ACSGPUGateError(f"{field} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or not 0.0 <= rendered <= 100.0:
        raise ACSGPUGateError(f"{field} must be between zero and 100")
    return rendered


def _parse_metric(value: str) -> float:
    stripped = value.strip()
    if stripped in {"-", "N/A", "[N/A]"}:
        return 0.0
    try:
        return float(stripped)
    except ValueError as exc:
        raise ACSGPUGateError(f"invalid nvidia-smi metric: {value!r}") from exc


class NvidiaSMISamplerV1:
    """Collect one global sample plus per-process pmon attribution."""

    def __init__(self, *, gpu_index: int = 0, timeout_seconds: float = 10.0):
        if type(gpu_index) is not int or gpu_index < 0:
            raise ACSGPUGateError("gpu_index must be a non-negative integer")
        if timeout_seconds <= 0:
            raise ACSGPUGateError("timeout_seconds must be positive")
        self._gpu_index = gpu_index
        self._timeout_seconds = timeout_seconds

    def _run(self, command: Sequence[str]) -> str:
        try:
            completed = subprocess.run(
                list(command),
                check=True,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ACSGPUGateError(f"nvidia-smi sampling failed: {exc}") from exc
        return completed.stdout

    def sample(self) -> Mapping[str, Any]:
        sample_started_monotonic_ns = time.perf_counter_ns()
        query = self._run(
            [
                "nvidia-smi",
                f"--id={self._gpu_index}",
                "--query-gpu=index,utilization.gpu,utilization.memory,"
                "utilization.encoder,utilization.decoder,memory.used,memory.total,"
                "power.draw,pstate",
                "--format=csv,noheader,nounits",
            ]
        )
        rows = [line for line in query.splitlines() if line.strip()]
        if len(rows) != 1:
            raise ACSGPUGateError("nvidia-smi GPU query did not return exactly one row")
        fields = [field.strip() for field in rows[0].split(",")]
        if len(fields) != 9:
            raise ACSGPUGateError("nvidia-smi GPU query returned an unexpected shape")
        try:
            observed_index = int(fields[0])
        except ValueError as exc:
            raise ACSGPUGateError("nvidia-smi returned an invalid GPU index") from exc
        if observed_index != self._gpu_index:
            raise ACSGPUGateError("nvidia-smi returned the wrong GPU index")

        pmon = self._run(
            [
                "nvidia-smi",
                "pmon",
                "-i",
                str(self._gpu_index),
                "-c",
                "1",
                "-s",
                "um",
            ]
        )
        processes: list[dict[str, Any]] = []
        for line in pmon.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            columns = line.split()
            if len(columns) < 12:
                raise ACSGPUGateError("nvidia-smi pmon returned an unexpected shape")
            try:
                process_gpu = int(columns[0])
                pid = int(columns[1])
            except ValueError as exc:
                raise ACSGPUGateError(
                    "nvidia-smi pmon returned an invalid process row"
                ) from exc
            if process_gpu != self._gpu_index:
                continue
            processes.append(
                {
                    "pid": pid,
                    "type": columns[2],
                    "sm_percent": _parse_metric(columns[3]),
                    "memory_percent": _parse_metric(columns[4]),
                    "encoder_percent": _parse_metric(columns[5]),
                    "decoder_percent": _parse_metric(columns[6]),
                    "framebuffer_memory_mib": _parse_metric(columns[9]),
                    "command": " ".join(columns[11:]),
                }
            )
        return {
            "sample_started_monotonic_ns": sample_started_monotonic_ns,
            "gpu_index": observed_index,
            "sm_percent": _parse_metric(fields[1]),
            "memory_percent": _parse_metric(fields[2]),
            "encoder_percent": _parse_metric(fields[3]),
            "decoder_percent": _parse_metric(fields[4]),
            "memory_used_mib": _parse_metric(fields[5]),
            "memory_total_mib": _parse_metric(fields[6]),
            "power_draw_w": _parse_metric(fields[7]),
            "pstate": fields[8],
            "processes": processes,
        }


def process_tree_pids_v1(root_pid: int) -> set[int]:
    """Return the live Linux process tree rooted at ``root_pid``."""

    pending = [root_pid]
    observed: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in observed:
            continue
        observed.add(pid)
        children_path = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            children = children_path.read_text(encoding="ascii").split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for child in children:
            try:
                pending.append(int(child))
            except ValueError:
                continue
    return observed


def _sample_row(
    sampler: GPUSampler,
    *,
    monotonic_ns: Callable[[], int],
    utc_now: Callable[[], str],
) -> dict[str, Any]:
    raw = dict(sampler.sample())
    processes = raw.get("processes")
    if not isinstance(processes, list):
        raise ACSGPUGateError("GPU sample processes must be a list")
    for field in ("sm_percent", "memory_percent", "encoder_percent", "decoder_percent"):
        raw[field] = _percent(raw.get(field), field=field)
    raw["monotonic_time_ns"] = monotonic_ns()
    sample_started_monotonic_ns = raw.get("sample_started_monotonic_ns")
    if sample_started_monotonic_ns is None:
        raw["sample_window_clock_bound"] = False
    elif (
        type(sample_started_monotonic_ns) is not int
        or sample_started_monotonic_ns < 0
        or sample_started_monotonic_ns > raw["monotonic_time_ns"]
    ):
        raise ACSGPUGateError("GPU sample start clock is invalid")
    else:
        raw["sample_window_clock_bound"] = True
    raw["observed_at"] = utc_now()
    return raw


def _process_activity(process: Mapping[str, Any]) -> float:
    return max(
        _percent(process.get(field), field=f"process.{field}")
        for field in (
            "sm_percent",
            "memory_percent",
            "encoder_percent",
            "decoder_percent",
        )
    )


def _baseline_reasons(
    samples: Sequence[Mapping[str, Any]],
    *,
    max_sm_percent: float,
    max_encoder_percent: float,
    max_decoder_percent: float,
    max_foreign_process_percent: float,
    allowed_baseline_process_commands: frozenset[str],
) -> list[str]:
    reasons: set[str] = set()
    for sample in samples:
        if float(sample["sm_percent"]) > max_sm_percent:
            reasons.add("GLOBAL_SM_UTILIZATION_ABOVE_LIMIT")
        if float(sample["encoder_percent"]) > max_encoder_percent:
            reasons.add("GLOBAL_ENCODER_UTILIZATION_ABOVE_LIMIT")
        if float(sample["decoder_percent"]) > max_decoder_percent:
            reasons.add("GLOBAL_DECODER_UTILIZATION_ABOVE_LIMIT")
        for process in sample["processes"]:
            command = process.get("command")
            if (
                command not in allowed_baseline_process_commands
                and _process_activity(process) > max_foreign_process_percent
            ):
                reasons.add("ACTIVE_FOREIGN_GPU_PROCESS")
    return sorted(reasons)


def _read_bound_manifest(path: Path) -> tuple[dict[str, Any], int]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACSGPUGateError(f"cannot bind completed batch manifest: {exc}") from exc
    if not isinstance(value, dict) or value.get("status") != "completed":
        raise ACSGPUGateError("bound batch manifest is not completed")
    result = value.get("result")
    captures = (
        result.get("acs_capture_measurements") if isinstance(result, dict) else None
    )
    if not isinstance(captures, list) or not captures:
        raise ACSGPUGateError("bound batch manifest has no ACS capture measurements")
    boundaries = [
        capture.get("last_useful_kernel_completion_observed_host_time_ns")
        for capture in captures
        if isinstance(capture, dict)
    ]
    if not boundaries or any(
        type(boundary) is not int or boundary < 0 for boundary in boundaries
    ):
        raise ACSGPUGateError(
            "bound batch manifest has invalid ACS GPU-idle boundaries"
        )
    return value, max(boundaries)


def _read_canonical_gpu_performance_window_v1(
    manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Read the optional tick-native window that defines primary GPU metrics.

    The timeline binding is the sole authority for this boundary.  A manifest
    that has no ``gpu_performance_window`` field is an older manifest and keeps
    the historical whole-runtime summary.  Once the field is present, even a
    null or incomplete value is a declared contract and must fail closed.
    """

    result = manifest.get("result")
    if not isinstance(result, Mapping):
        return None
    binding = result.get("tick_native_performance_timeline_binding")
    if binding is None:
        return None
    if not isinstance(binding, Mapping):
        raise ACSGPUGateError(
            "tick-native performance timeline binding must be an object"
        )
    timeline = binding.get("timeline")
    if not isinstance(timeline, Mapping):
        raise ACSGPUGateError(
            "tick-native performance timeline binding lacks a timeline object"
        )
    if "gpu_performance_window" not in timeline:
        return None
    raw_window = timeline.get("gpu_performance_window")
    if not isinstance(raw_window, Mapping):
        raise ACSGPUGateError(
            "gpu_performance_window must be an object when declared"
        )
    start_ns = raw_window.get("start_ns")
    end_ns = raw_window.get("end_ns")
    if type(start_ns) is not int or start_ns < 0:
        raise ACSGPUGateError(
            "gpu_performance_window.start_ns must be a nonnegative integer"
        )
    if type(end_ns) is not int or end_ns < 0:
        raise ACSGPUGateError(
            "gpu_performance_window.end_ns must be a nonnegative integer"
        )
    if end_ns <= start_ns:
        raise ACSGPUGateError(
            "gpu_performance_window timestamps must be strictly increasing"
        )
    if raw_window.get("post_simulation_acs_tail_excluded") is not True:
        raise ACSGPUGateError(
            "gpu_performance_window must explicitly exclude the post-simulation ACS tail"
        )
    if raw_window.get("measurement_scope") != _CANONICAL_GPU_PERFORMANCE_SCOPE_V1:
        raise ACSGPUGateError(
            "gpu_performance_window has an unsupported measurement scope"
        )
    terminal_checkpoint_tick = raw_window.get("terminal_checkpoint_tick")
    if type(terminal_checkpoint_tick) is not int or terminal_checkpoint_tick < 1:
        raise ACSGPUGateError(
            "gpu_performance_window.terminal_checkpoint_tick must be a positive integer"
        )
    duration_ns = raw_window.get("duration_ns")
    if duration_ns is not None and (
        type(duration_ns) is not int or duration_ns != end_ns - start_ns
    ):
        raise ACSGPUGateError(
            "gpu_performance_window.duration_ns does not match its boundaries"
        )
    seconds = raw_window.get("seconds")
    if seconds is not None:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise ACSGPUGateError("gpu_performance_window.seconds must be numeric")
        if not math.isfinite(float(seconds)) or not math.isclose(
            float(seconds), (end_ns - start_ns) / 1_000_000_000.0
        ):
            raise ACSGPUGateError(
                "gpu_performance_window.seconds does not match its boundaries"
            )
    return {
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ns": end_ns - start_ns,
        "seconds": (end_ns - start_ns) / 1_000_000_000.0,
        "terminal_checkpoint_tick": terminal_checkpoint_tick,
        "post_simulation_acs_tail_excluded": True,
        "measurement_scope": _CANONICAL_GPU_PERFORMANCE_SCOPE_V1,
    }


def _read_terminal_gpu_release_boundary_v1(
    manifest: Mapping[str, Any],
    *,
    gpu_performance_window: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind the terminal CUDA reset to the canonical performance-window end."""

    if not isinstance(gpu_performance_window, Mapping):
        raise ACSGPUGateError(
            "post-release residency enforcement requires a canonical GPU performance window"
        )
    result = manifest.get("result")
    if not isinstance(result, Mapping):
        raise ACSGPUGateError(
            "post-release residency enforcement requires a bound result"
        )
    terminal_tick = gpu_performance_window.get("terminal_checkpoint_tick")
    window_end_ns = gpu_performance_window.get("end_ns")
    captures = result.get("acs_capture_measurements")
    if not isinstance(captures, list):
        raise ACSGPUGateError(
            "post-release residency enforcement requires ACS capture measurements"
        )
    terminal_captures = [
        row
        for row in captures
        if isinstance(row, Mapping) and row.get("coordinate") == terminal_tick
    ]
    if len(terminal_captures) != 1:
        raise ACSGPUGateError(
            "post-release residency enforcement requires exactly one terminal capture"
        )
    capture_release_ns = terminal_captures[0].get(
        "gpu_safely_releasable_perf_ns"
    )
    reset_receipts = result.get("terminal_cuda_context_release_receipts")
    if not isinstance(reset_receipts, list) or len(reset_receipts) != 1:
        raise ACSGPUGateError(
            "post-release residency enforcement requires exactly one terminal CUDA reset receipt"
        )
    reset_receipt = reset_receipts[0]
    if not isinstance(reset_receipt, Mapping):
        raise ACSGPUGateError("terminal CUDA reset receipt must be an object")
    reset_release_ns = reset_receipt.get("released_perf_ns")
    if (
        reset_receipt.get("schema_version")
        != "eiel_terminal_cuda_primary_context_release_v1"
        or reset_receipt.get("cuda_device_reset_result") != 0
        or type(capture_release_ns) is not int
        or capture_release_ns < 0
        or type(reset_release_ns) is not int
        or reset_release_ns < 0
        or capture_release_ns != reset_release_ns
        or capture_release_ns != window_end_ns
    ):
        raise ACSGPUGateError(
            "terminal CUDA reset, capture release, and performance-window end do not bind exactly"
        )
    return {
        "schema_version": "eiel_terminal_gpu_release_boundary_v1",
        "terminal_checkpoint_tick": terminal_tick,
        "released_perf_ns": capture_release_ns,
        "cuda_device_reset_return_code": 0,
        "performance_window_end_bound": True,
    }


def _post_release_target_gpu_residency_v1(
    samples: Sequence[Mapping[str, Any]],
    *,
    released_perf_ns: int,
    launched_pid: int,
    attributed_target_gpu_pids: set[int],
) -> dict[str, Any]:
    """Prove that the launched process tree does not reacquire framebuffer memory."""

    target_pids = set(attributed_target_gpu_pids)
    target_pids.add(launched_pid)
    observation_not_before_perf_ns = released_perf_ns
    post_release_samples = [
        sample
        for sample in samples
        if sample.get("sample_window_clock_bound") is True
        and int(sample["sample_started_monotonic_ns"])
        > observation_not_before_perf_ns
    ]
    straddling_samples = [
        sample
        for sample in samples
        if sample.get("sample_window_clock_bound") is True
        and int(sample["sample_started_monotonic_ns"]) <= released_perf_ns
        < int(sample["monotonic_time_ns"])
    ]
    resident_rows: list[dict[str, Any]] = []
    for window_relation, sample in (
        [("straddles_release", sample) for sample in straddling_samples]
        + [("after_release", sample) for sample in post_release_samples]
    ):
        for process in sample["processes"]:
            pid = process.get("pid")
            framebuffer_memory_mib = process.get("framebuffer_memory_mib")
            if (
                type(pid) is int
                and pid in target_pids
                and isinstance(framebuffer_memory_mib, (int, float))
                and not isinstance(framebuffer_memory_mib, bool)
                and float(framebuffer_memory_mib) > 0.0
            ):
                resident_rows.append(
                    {
                        "monotonic_time_ns": int(sample["monotonic_time_ns"]),
                        "sample_started_monotonic_ns": int(
                            sample["sample_started_monotonic_ns"]
                        ),
                        "window_relation": window_relation,
                        "pid": pid,
                        "framebuffer_memory_mib": float(framebuffer_memory_mib),
                        "sm_percent": _percent(
                            process.get("sm_percent"), field="process.sm_percent"
                        ),
                        "command": process.get("command"),
                    }
                )
    return {
        "schema_version": "eiel_post_release_target_gpu_residency_v1",
        "released_perf_ns": released_perf_ns,
        "settlement_grace_seconds": 0.0,
        "observation_not_before_perf_ns": observation_not_before_perf_ns,
        "sample_window_clock_bound": True,
        "straddling_sample_count": len(straddling_samples),
        "straddling_resident_sample_count": sum(
            row["window_relation"] == "straddles_release" for row in resident_rows
        ),
        "launched_pid": launched_pid,
        "attributed_target_gpu_pids": sorted(attributed_target_gpu_pids),
        "post_release_sample_count": len(post_release_samples),
        "resident_sample_count": len(resident_rows),
        "max_framebuffer_memory_mib": max(
            (row["framebuffer_memory_mib"] for row in resident_rows),
            default=0.0,
        ),
        "resident_rows": resident_rows,
        "passed": bool(post_release_samples) and not resident_rows,
    }


def _target_process_sm_percent(sample: Mapping[str, Any]) -> float:
    target_pids = set(sample.get("target_gpu_pids") or ())
    return max(
        (
            float(process.get("sm_percent", 0.0))
            for process in sample.get("processes", ())
            if isinstance(process, Mapping) and process.get("pid") in target_pids
        ),
        default=0.0,
    )


def _async_checkpoint_overlap_evidence_v1(
    manifest: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    *,
    active_sm_threshold_percent: float,
) -> dict[str, Any]:
    result = manifest.get("result")
    captures = (
        result.get("acs_capture_measurements")
        if isinstance(result, Mapping)
        else None
    )
    terminal_tick = (
        result.get("terminal_next_absolute_tick") if isinstance(result, Mapping) else None
    )

    def exact_tick_list(value: Any) -> list[int] | None:
        if (
            not isinstance(value, list)
            or any(type(item) is not int or item < 0 for item in value)
            or len(set(value)) != len(value)
            or value != sorted(value)
        ):
            return None
        return list(value)

    expected_transition_ticks = exact_tick_list(
        result.get("expected_resident_owner_transition_ticks")
        if isinstance(result, Mapping)
        else None
    )
    observed_transition_ticks = exact_tick_list(
        result.get("observed_resident_owner_transition_ticks")
        if isinstance(result, Mapping)
        else None
    )
    transfer_receipts_raw = (
        result.get("resident_owner_epoch_transfer_receipts")
        if isinstance(result, Mapping)
        else None
    )
    transfer_receipts = (
        [row for row in transfer_receipts_raw if isinstance(row, Mapping)]
        if isinstance(transfer_receipts_raw, list)
        else []
    )
    transfer_receipt_ticks = exact_tick_list(
        [row.get("boundary_next_absolute_tick") for row in transfer_receipts]
    )
    transfer_physical_owner_ids = {
        row.get("physical_owner_identity") for row in transfer_receipts
    }
    transfer_receipt_rows_valid = bool(
        isinstance(transfer_receipts_raw, list)
        and transfer_receipts
        and len(transfer_receipts) == len(transfer_receipts_raw)
        and transfer_receipt_ticks is not None
        and all(
            row.get("schema_version") == "eiel_resident_owner_epoch_transfer_v1"
            and isinstance(row.get("physical_owner_identity"), str)
            and bool(row.get("physical_owner_identity"))
            and isinstance(row.get("prior_owner_identity"), str)
            and bool(row.get("prior_owner_identity"))
            and isinstance(row.get("successor_owner_identity"), str)
            and bool(row.get("successor_owner_identity"))
            and row.get("prior_owner_identity")
            != row.get("successor_owner_identity")
            and type(row.get("boundary_next_absolute_tick")) is int
            and row["boundary_next_absolute_tick"] >= 0
            and type(row.get("successor_span_end_tick")) is int
            and row["successor_span_end_tick"]
            > row["boundary_next_absolute_tick"]
            and row.get("physical_owner_reused") is True
            and row.get("cuda_allocation_recreated") is False
            and row.get("simultaneous_physical_owner_count") == 1
            for row in transfer_receipts
        )
        and len(transfer_physical_owner_ids) == 1
        and all(
            previous.get("successor_owner_identity")
            == current.get("prior_owner_identity")
            and previous.get("successor_span_end_tick")
            == current.get("boundary_next_absolute_tick")
            for previous, current in zip(
                transfer_receipts,
                transfer_receipts[1:],
                strict=False,
            )
        )
    )
    windows: list[dict[str, Any]] = []
    if isinstance(captures, list):
        for capture in captures:
            if not isinstance(capture, Mapping):
                continue
            coordinate = capture.get("coordinate")
            capture_started = capture.get("capture_started_perf_ns")
            submitted = capture.get("detached_capture_submitted_perf_ns")
            writer_started = capture.get("detached_writer_started_perf_ns")
            writer_finished = capture.get("detached_writer_finished_perf_ns")
            capture_completed = capture.get("capture_completed_perf_ns")
            capture_barrier_time = capture.get("capture_barrier_time_ns")
            gpu_safely_releasable = capture.get("gpu_safely_releasable_perf_ns")
            source_required = capture.get("detached_source_consumer_required")
            source_released = capture.get("detached_source_released_perf_ns")
            successor_prepare_started = capture.get(
                "successor_owner_prepare_started_perf_ns"
            )
            successor_prepare_completed = capture.get(
                "successor_owner_prepare_completed_perf_ns"
            )
            successor_dispatch_started = capture.get(
                "successor_dispatch_started_perf_ns"
            )
            successor_dispatch_completed = capture.get(
                "successor_dispatch_completed_perf_ns"
            )
            successor_last_useful_kernel_completion = capture.get(
                "successor_last_useful_kernel_completion_observed_host_time_ns"
            )
            successor_final_stream_synchronization_time = capture.get(
                "successor_final_stream_synchronization_time_ns"
            )
            successor_final_host_synchronizations = capture.get(
                "successor_final_host_synchronizations"
            )
            execution_device = capture.get("execution_device")
            owner_transition = capture.get("owner_transition")
            if not all(
                type(value) is int and value >= 0
                for value in (
                    coordinate,
                    capture_started,
                    submitted,
                    writer_started,
                    writer_finished,
                    capture_completed,
                    capture_barrier_time,
                )
            ):
                continue
            ordered = (
                capture_started
                <= capture_completed
                <= submitted
                <= writer_started
                < writer_finished
            )
            capture_clock_ordered = bool(
                capture_started
                <= capture_completed
                <= submitted
                <= writer_started
                < writer_finished
            )
            writer_window_samples = [
                sample
                for sample in samples
                if writer_started <= int(sample["monotonic_time_ns"]) <= writer_finished
            ]
            writer_active_target_samples = [
                sample
                for sample in writer_window_samples
                if _target_process_sm_percent(sample) > active_sm_threshold_percent
            ]
            nonterminal = type(terminal_tick) is int and coordinate < terminal_tick
            if "gpu_safely_releasable_perf_ns" in capture:
                if type(gpu_safely_releasable) is int:
                    release_authority = gpu_safely_releasable
                    release_authority_kind = "gpu_safe"
                else:
                    release_authority = None
                    release_authority_kind = None
            elif source_required is True and type(source_released) is int:
                release_authority = source_released
                release_authority_kind = "source"
            else:
                release_authority = None
                release_authority_kind = None
            gpu_safe_release_ordered = bool(
                type(release_authority) is int
                and release_authority_kind
                == ("source" if nonterminal else "gpu_safe")
                and capture_started
                <= capture_completed
                <= submitted
                <= writer_started
                < writer_finished
                <= release_authority
            )
            source_lease_ordered = (
                source_required is True
                and type(source_released) is int
                and capture_started <= source_released
            )
            successor_dispatch_ordered = (
                type(successor_dispatch_started) is int
                and type(successor_dispatch_completed) is int
                and capture_started <= successor_dispatch_started
                < successor_dispatch_completed
            )
            source_handoff_ordered = bool(
                source_lease_ordered
                and owner_transition is True
                and type(successor_prepare_started) is int
                and type(successor_prepare_completed) is int
                and successor_dispatch_ordered
                and capture_started <= successor_prepare_started
                <= successor_prepare_completed
                <= successor_dispatch_started
                < successor_dispatch_completed
                <= source_released
            )
            prepare_overlap_started = (
                max(writer_started, successor_prepare_started)
                if type(successor_prepare_started) is int
                else None
            )
            prepare_overlap_finished = (
                min(writer_finished, successor_prepare_completed)
                if type(successor_prepare_completed) is int
                else None
            )
            successor_prepare_overlapped_writer = bool(
                type(prepare_overlap_started) is int
                and type(prepare_overlap_finished) is int
                and prepare_overlap_started < prepare_overlap_finished
            )
            dispatch_overlap_started = (
                max(writer_started, successor_dispatch_started)
                if successor_dispatch_ordered
                else None
            )
            dispatch_overlap_finished = (
                min(writer_finished, successor_dispatch_completed)
                if successor_dispatch_ordered
                else None
            )
            successor_dispatch_overlapped_writer = bool(
                type(dispatch_overlap_started) is int
                and type(dispatch_overlap_finished) is int
                and dispatch_overlap_started < dispatch_overlap_finished
            )
            dispatch_overlap_samples = (
                [
                    sample
                    for sample in samples
                    if dispatch_overlap_started
                    <= int(sample["monotonic_time_ns"])
                    <= dispatch_overlap_finished
                ]
                if successor_dispatch_overlapped_writer
                else []
            )
            active_dispatch_overlap_samples = [
                sample
                for sample in dispatch_overlap_samples
                if _target_process_sm_percent(sample) > active_sm_threshold_percent
            ]
            native_gpu_completion_ordered = bool(
                successor_dispatch_ordered
                and isinstance(execution_device, str)
                and execution_device.startswith("cuda")
                and type(successor_last_useful_kernel_completion) is int
                and successor_dispatch_started
                <= successor_last_useful_kernel_completion
                <= successor_dispatch_completed
                and type(successor_final_stream_synchronization_time) is int
                and successor_final_stream_synchronization_time >= 0
                and successor_final_host_synchronizations == 1
            )
            native_gpu_completion_overlapped_writer = bool(
                native_gpu_completion_ordered
                and writer_started
                <= successor_last_useful_kernel_completion
                <= writer_finished
            )
            writer_active_at_successor_dispatch = bool(
                successor_dispatch_ordered
                and successor_dispatch_started < writer_finished
            )
            sampled_gpu_activity_overlap_observed = bool(
                writer_active_at_successor_dispatch
                and active_dispatch_overlap_samples
            )
            exact_gpu_overlap_observed = bool(
                writer_active_at_successor_dispatch
                and (
                    sampled_gpu_activity_overlap_observed
                    or native_gpu_completion_overlapped_writer
                )
            )
            transition_handoff_nonblocking = bool(
                source_handoff_ordered
                and (
                    writer_finished <= successor_prepare_started
                    or successor_prepare_overlapped_writer
                    or exact_gpu_overlap_observed
                )
            )
            exact_async_progress_observed = bool(
                exact_gpu_overlap_observed
                or source_lease_ordered
                and successor_prepare_overlapped_writer
            )
            windows.append(
                {
                    "coordinate": coordinate,
                    "nonterminal": nonterminal,
                    "owner_transition": owner_transition,
                    "capture_then_writer_ordered": ordered,
                    "capture_clock_ordered": capture_clock_ordered,
                    "capture_started_perf_ns": capture_started,
                    "capture_completed_perf_ns": capture_completed,
                    "capture_barrier_time_ns": capture_barrier_time,
                    "capture_submitted_perf_ns": submitted,
                    "gpu_safely_releasable_perf_ns": gpu_safely_releasable,
                    "gpu_safe_release_ordered": gpu_safe_release_ordered,
                    "release_authority_perf_ns": release_authority,
                    "release_authority_kind": release_authority_kind,
                    "writer_started_perf_ns": writer_started,
                    "writer_finished_perf_ns": writer_finished,
                    "writer_span_seconds": (writer_finished - writer_started)
                    / 1_000_000_000,
                    "writer_window_sample_count": len(writer_window_samples),
                    "writer_window_active_target_sample_count": len(
                        writer_active_target_samples
                    ),
                    "source_consumer_required": source_required,
                    "source_lease_ordered": source_lease_ordered,
                    "source_handoff_ordered": source_handoff_ordered,
                    "successor_owner_prepare_started_perf_ns": (
                        successor_prepare_started
                    ),
                    "successor_owner_prepare_completed_perf_ns": (
                        successor_prepare_completed
                    ),
                    "successor_dispatch_started_perf_ns": (
                        successor_dispatch_started
                    ),
                    "successor_dispatch_completed_perf_ns": (
                        successor_dispatch_completed
                    ),
                    "successor_dispatch_ordered": (
                        successor_dispatch_ordered
                    ),
                    "source_released_perf_ns": source_released,
                    "prepare_writer_intersection_started_perf_ns": (
                        prepare_overlap_started
                    ),
                    "prepare_writer_intersection_finished_perf_ns": (
                        prepare_overlap_finished
                    ),
                    "successor_prepare_overlapped_writer": (
                        successor_prepare_overlapped_writer
                    ),
                    "dispatch_writer_intersection_started_perf_ns": (
                        dispatch_overlap_started
                    ),
                    "dispatch_writer_intersection_finished_perf_ns": (
                        dispatch_overlap_finished
                    ),
                    "successor_dispatch_overlapped_writer": (
                        successor_dispatch_overlapped_writer
                    ),
                    "dispatch_overlap_sample_count": len(
                        dispatch_overlap_samples
                    ),
                    "dispatch_overlap_active_target_sample_count": len(
                        active_dispatch_overlap_samples
                    ),
                    "sampled_gpu_activity_overlap_observed": (
                        sampled_gpu_activity_overlap_observed
                    ),
                    "successor_last_useful_kernel_completion_observed_host_time_ns": (
                        successor_last_useful_kernel_completion
                    ),
                    "successor_final_stream_synchronization_time_ns": (
                        successor_final_stream_synchronization_time
                    ),
                    "successor_final_host_synchronizations": (
                        successor_final_host_synchronizations
                    ),
                    "execution_device": execution_device,
                    "native_gpu_completion_ordered": native_gpu_completion_ordered,
                    "native_gpu_completion_overlapped_writer": (
                        native_gpu_completion_overlapped_writer
                    ),
                    "gpu_simulation_overlap_observed": exact_gpu_overlap_observed,
                    "gpu_overlap_proof_method": (
                        "native_stream_completion_and_sampled_process_activity_v1"
                        if native_gpu_completion_overlapped_writer
                        and sampled_gpu_activity_overlap_observed
                        else "native_stream_completion_v1"
                        if native_gpu_completion_overlapped_writer
                        else "sampled_process_activity_v1"
                        if sampled_gpu_activity_overlap_observed
                        else None
                    ),
                    "writer_active_at_successor_dispatch": (
                        writer_active_at_successor_dispatch
                    ),
                    "asynchronous_progress_observed": (
                        exact_async_progress_observed
                    ),
                    "transition_handoff_nonblocking": (
                        transition_handoff_nonblocking
                    ),
                }
            )

    staging_rows: list[Mapping[str, Any]] = []
    final_staging: Mapping[str, Any] | None = None
    if isinstance(result, Mapping):
        raw_final_staging = result.get("acs_checkpoint_staging")
        if isinstance(raw_final_staging, Mapping):
            final_staging = raw_final_staging
            staging_rows.append(raw_final_staging)
        catalog = result.get("checkpoint_catalog")
        if isinstance(catalog, Mapping):
            for checkpoint in catalog.values():
                staging = (
                    checkpoint.get("acs_checkpoint_staging")
                    if isinstance(checkpoint, Mapping)
                    else None
                )
                if isinstance(staging, Mapping):
                    staging_rows.append(staging)
    occupancy_high_water = max(
        (
            int(row["queue_occupancy_high_water"])
            for row in staging_rows
            if type(row.get("queue_occupancy_high_water")) is int
        ),
        default=0,
    )
    slots = max(
        (int(row["slots"]) for row in staging_rows if type(row.get("slots")) is int),
        default=0,
    )
    worker_accepts_cuda_objects = (
        False
        if staging_rows
        and all(
            row.get("worker_accepts_cuda_objects") is False
            for row in staging_rows
        )
        else (
            True
            if any(
                row.get("worker_accepts_cuda_objects") is True
                for row in staging_rows
            )
            else None
        )
    )
    worker_tensor_device_policy = (
        "forced_cpu_context_v1"
        if staging_rows
        and all(
            row.get("worker_tensor_device_policy") == "forced_cpu_context_v1"
            for row in staging_rows
        )
        else None
    )
    worker_cuda_free = bool(
        worker_accepts_cuda_objects is False
        and worker_tensor_device_policy == "forced_cpu_context_v1"
    )
    worker_accepts_callbacks = (
        final_staging.get("worker_accepts_callbacks")
        if isinstance(final_staging, Mapping)
        else None
    )
    worker_callbacks_free = worker_accepts_callbacks is False
    temporary_file_high_water = max(
        (
            int(row["temporary_capture_file_high_water"])
            for row in staging_rows
            if type(row.get("temporary_capture_file_high_water")) is int
        ),
        default=0,
    )
    source_file_lifecycle_bounded = bool(
        isinstance(final_staging, Mapping)
        and final_staging.get("detached_source_lease_count") == 0
        and final_staging.get("source_consumer_leases_active") == 0
        and final_staging.get("source_lease_spool_bytes_active") == 0
        and final_staging.get("temporary_capture_files_active") == 0
        and temporary_file_high_water <= slots
        and final_staging.get("capture_pool_count") == 1
        and final_staging.get("capture_pool_model")
        == "one_shared_two_slab_pool_not_per_slot"
    )
    staging_queued_cuda_references = (
        final_staging.get("queued_cuda_references")
        if isinstance(final_staging, Mapping)
        else None
    )
    staging_queued_cuda_callbacks = (
        final_staging.get("queued_cuda_callbacks")
        if isinstance(final_staging, Mapping)
        else None
    )
    queued_cuda_references_explicit_zero = (
        type(staging_queued_cuda_references) is int
        and staging_queued_cuda_references == 0
    )
    queued_cuda_callbacks_explicit_zero = (
        type(staging_queued_cuda_callbacks) is int
        and staging_queued_cuda_callbacks == 0
    )
    owner_max = (
        result.get("resident_owner_max_simultaneous")
        if isinstance(result, Mapping)
        else None
    )
    terminal_owner_count = (
        result.get("resident_owner_live_count_terminal")
        if isinstance(result, Mapping)
        else None
    )
    owner_lifetimes = (
        result.get("resident_owner_lifetime_measurements")
        if isinstance(result, Mapping)
        else None
    )
    owner_lifetime_rows = (
        [row for row in owner_lifetimes if isinstance(row, Mapping)]
        if isinstance(owner_lifetimes, list)
        else []
    )
    owner_lifetime_rows_valid = bool(
        isinstance(owner_lifetimes, list)
        and owner_lifetimes
        and len(owner_lifetime_rows) == len(owner_lifetimes)
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("owner_identity"), str)
            and type(row.get("started_perf_ns")) is int
            and type(row.get("released_perf_ns")) is int
            and row["started_perf_ns"] < row["released_perf_ns"]
            and row.get("live_owner_count_before_release") == 1
            and row.get("live_owner_count_after_release") == 0
            for row in owner_lifetime_rows
        )
        and len(
            {
                str(row["owner_identity"])
                for row in owner_lifetime_rows
            }
        )
        == len(owner_lifetime_rows)
        and all(
            int(previous["released_perf_ns"]) <= int(current["started_perf_ns"])
            for previous, current in zip(
                owner_lifetime_rows,
                owner_lifetime_rows[1:],
                strict=False,
            )
        )
    )
    owner_lifecycle_measured = bool(
        owner_max == 1
        and terminal_owner_count == 0
        and owner_lifetime_rows_valid
    )
    physical_owner_continuity_measured = bool(
        owner_lifecycle_measured
        and len(owner_lifetime_rows) == 1
        and transfer_receipt_rows_valid
    )
    qualifying_windows = [
        row
        for row in windows
        if row["nonterminal"]
        and row["capture_then_writer_ordered"]
        and row["gpu_simulation_overlap_observed"]
    ]
    owner_transition_windows = [
        row
        for row in windows
        if row["source_consumer_required"] is True
        and row["owner_transition"] is True
    ]
    owner_transition_window_ticks = [
        int(row["coordinate"]) for row in owner_transition_windows
    ]
    exact_transition_coverage = bool(
        expected_transition_ticks is not None
        and expected_transition_ticks
        and observed_transition_ticks == expected_transition_ticks
        and transfer_receipt_ticks == expected_transition_ticks
        and transfer_receipt_rows_valid
    )
    passed = (
        exact_transition_coverage
        and bool(qualifying_windows)
        and all(
            row["nonterminal"]
            and row["capture_clock_ordered"]
            and row["gpu_safe_release_ordered"]
            and row["capture_then_writer_ordered"]
            and row["source_lease_ordered"]
            and row["source_handoff_ordered"]
            and row["transition_handoff_nonblocking"]
            for row in owner_transition_windows
        )
        and slots == 2
        and occupancy_high_water == slots
        and worker_cuda_free
        and worker_callbacks_free
        and source_file_lifecycle_bounded
        and queued_cuda_references_explicit_zero
        and queued_cuda_callbacks_explicit_zero
        and physical_owner_continuity_measured
    )
    return {
        "schema_version": "eiel_acs_async_overlap_evidence_v5",
        "passed": passed,
        "active_sm_threshold_percent": active_sm_threshold_percent,
        "terminal_next_absolute_tick": terminal_tick,
        "slots": slots,
        "queue_occupancy_high_water": occupancy_high_water,
        "worker_accepts_cuda_objects": worker_accepts_cuda_objects,
        "worker_tensor_device_policy": worker_tensor_device_policy,
        "worker_accepts_callbacks": worker_accepts_callbacks,
        "queued_cuda_references": staging_queued_cuda_references,
        "queued_cuda_callbacks": staging_queued_cuda_callbacks,
        "temporary_capture_file_high_water": temporary_file_high_water,
        "source_file_lifecycle_bounded": source_file_lifecycle_bounded,
        "resident_owner_max_simultaneous": owner_max,
        "resident_owner_live_count_terminal": terminal_owner_count,
        "resident_owner_lifetime_rows_valid": owner_lifetime_rows_valid,
        "owner_lifecycle_measured": owner_lifecycle_measured,
        "physical_owner_continuity_measured": physical_owner_continuity_measured,
        "expected_owner_transition_ticks": expected_transition_ticks,
        "observed_owner_transition_ticks": observed_transition_ticks,
        "owner_epoch_transfer_ticks": transfer_receipt_ticks,
        "owner_epoch_transfer_count": len(transfer_receipts),
        "owner_epoch_transfer_rows_valid": transfer_receipt_rows_valid,
        "physical_owner_identity_count": len(transfer_physical_owner_ids),
        "owner_transition_window_ticks": owner_transition_window_ticks,
        "exact_owner_transition_coverage": exact_transition_coverage,
        "checkpoint_windows": windows,
        "qualifying_nonterminal_overlap_count": len(qualifying_windows),
        "owner_transition_window_count": len(owner_transition_windows),
        "owner_transition_overlap_count": sum(
            1
            for row in owner_transition_windows
            if row["transition_handoff_nonblocking"]
        ),
        "limitations": [
            "NVML process activity is sampled corroboration and can lag short native dispatch windows",
            "native overlap requires the route's exact final CUDA stream completion boundary inside the writer interval",
            "GPU overlap is counted only inside the exact successor-dispatch and writer intersection",
            "logical owner-epoch coverage is proven by in-place transfer receipts and is independent of checkpoint capture",
            "when persistence finishes before successor dispatch, exact successor-prepare and writer overlap proves producer progress without requiring impossible GPU overlap",
            "a passing window proves persistence overlap after capture, not zero capture-barrier cost",
            "physical B136 acceptance requires both detached slots to become occupied",
        ],
    }


def _terminate_process_group(process: GuardedProcess, *, grace_seconds: float) -> int:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, OSError):
        process.terminate()
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, OSError):
            process.kill()
        return process.wait(timeout=grace_seconds)


def run_guarded_command_v1(
    command: Sequence[str],
    *,
    manifest_path: Path,
    source_commit: str,
    sampler: GPUSampler,
    baseline_sample_count: int = 6,
    sample_interval_seconds: float = 5.0,
    max_baseline_sm_percent: float = 5.0,
    max_baseline_encoder_percent: float = 0.0,
    max_baseline_decoder_percent: float = 0.0,
    max_foreign_process_percent: float = 0.0,
    idle_sm_threshold_percent: float = 5.0,
    termination_grace_seconds: float = 10.0,
    allowed_baseline_process_commands: Sequence[str] = (),
    stdout_log_path: Path | None = None,
    stderr_log_path: Path | None = None,
    sample_stream_path: Path | None = None,
    execution_request_path: Path | None = None,
    require_async_checkpoint_overlap: bool = False,
    require_post_release_zero_residency: bool = False,
    launcher: Callable[[Sequence[str]], GuardedProcess] | None = None,
    process_tree: Callable[[int], set[int]] = process_tree_pids_v1,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    utc_now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    """Admit once, run once, and return a manifest-bound telemetry receipt."""

    argv = list(command)
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ACSGPUGateError("command must contain non-empty argv strings")
    if type(baseline_sample_count) is not int or baseline_sample_count <= 0:
        raise ACSGPUGateError("baseline_sample_count must be positive")
    if sample_interval_seconds <= 0 or termination_grace_seconds <= 0:
        raise ACSGPUGateError("sample interval and termination grace must be positive")
    if type(require_async_checkpoint_overlap) is not bool:
        raise ACSGPUGateError("require_async_checkpoint_overlap must be an exact bool")
    if type(require_post_release_zero_residency) is not bool:
        raise ACSGPUGateError(
            "require_post_release_zero_residency must be an exact bool"
        )
    thresholds = {
        "max_baseline_sm_percent": _percent(
            max_baseline_sm_percent, field="max_baseline_sm_percent"
        ),
        "max_baseline_encoder_percent": _percent(
            max_baseline_encoder_percent, field="max_baseline_encoder_percent"
        ),
        "max_baseline_decoder_percent": _percent(
            max_baseline_decoder_percent, field="max_baseline_decoder_percent"
        ),
        "max_foreign_process_percent": _percent(
            max_foreign_process_percent, field="max_foreign_process_percent"
        ),
        "idle_sm_threshold_percent": _percent(
            idle_sm_threshold_percent, field="idle_sm_threshold_percent"
        ),
    }
    allowed_commands = frozenset(allowed_baseline_process_commands)
    if any(not isinstance(command, str) or not command for command in allowed_commands):
        raise ACSGPUGateError(
            "allowed_baseline_process_commands must contain non-empty strings"
        )
    command_identity = {"argv": argv}
    base: dict[str, Any] = {
        "schema_version": ACS_GPU_TELEMETRY_SCHEMA_VERSION,
        "source_commit": source_commit,
        "command": command_identity,
        "command_sha256": _canonical_sha256(command_identity),
        "manifest_path": str(manifest_path.resolve()),
        "thresholds": thresholds,
        "baseline_sample_count": baseline_sample_count,
        "sample_interval_seconds": sample_interval_seconds,
        "sample_clock": "python_perf_counter_ns_v1",
        "allowed_baseline_process_commands": sorted(allowed_commands),
        "started_at": utc_now(),
        "require_async_checkpoint_overlap": require_async_checkpoint_overlap,
        "require_post_release_zero_residency": (
            require_post_release_zero_residency
        ),
    }
    if execution_request_path is not None:
        request_path = execution_request_path.resolve()
        if not request_path.is_file():
            raise ACSGPUGateError(
                f"execution request is not a file: {request_path}"
            )
        base["execution_request_path"] = str(request_path)
        base["execution_request_sha256"] = _file_sha256(request_path)
    stream = (
        _SampleStreamV1(
            sample_stream_path,
            header={
                "source_commit": source_commit,
                "command": command_identity,
                "command_sha256": base["command_sha256"],
                "manifest_path": base["manifest_path"],
                "execution_request_path": base.get("execution_request_path"),
                "execution_request_sha256": base.get("execution_request_sha256"),
                "requested_sample_interval_seconds": sample_interval_seconds,
                "sample_clock": base["sample_clock"],
                "started_at": base["started_at"],
            },
        )
        if sample_stream_path is not None
        else None
    )

    baseline: list[dict[str, Any]] = []
    try:
        for index in range(baseline_sample_count):
            row = _sample_row(sampler, monotonic_ns=monotonic_ns, utc_now=utc_now)
            baseline.append(row)
            if stream is not None:
                stream.append(phase="baseline", sample=row)
            if index + 1 < baseline_sample_count:
                sleeper(sample_interval_seconds)
    except BaseException:
        if stream is not None:
            try:
                stream.finish(status="sampling_failed_before_launch", summary=None)
            finally:
                stream.close()
        raise
    reasons = _baseline_reasons(
        baseline,
        max_sm_percent=thresholds["max_baseline_sm_percent"],
        max_encoder_percent=thresholds["max_baseline_encoder_percent"],
        max_decoder_percent=thresholds["max_baseline_decoder_percent"],
        max_foreign_process_percent=thresholds["max_foreign_process_percent"],
        allowed_baseline_process_commands=allowed_commands,
    )
    baseline_pids = sorted(
        {
            int(process["pid"])
            for sample in baseline
            for process in sample["processes"]
            if type(process.get("pid")) is int
            and process.get("command") in allowed_commands
        }
    )
    if reasons:
        receipt = {
            **base,
            "status": "rejected_before_launch",
            "baseline_admitted": False,
            "launched": False,
            "rejection_reasons": reasons,
            "baseline_gpu_process_pids": baseline_pids,
            "baseline_samples": baseline,
            "runtime_samples": [],
            "sample_count": 0,
            "no_competing_gpu_workloads": False,
            "process_attributed": False,
            "exclusive_gpu_window": False,
            "finished_at": utc_now(),
        }
        if stream is not None:
            stream.finish(status=receipt["status"], summary=None)
            stream.close()
            receipt["sample_stream_path"] = str(stream.path)
            receipt["sample_stream_sha256"] = _file_sha256(stream.path)
        return receipt

    stdout_handle: Any | None = None
    stderr_handle: Any | None = None
    try:
        if launcher is None:
            if stdout_log_path is not None:
                stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
                stdout_handle = stdout_log_path.open("x", encoding="utf-8")
            if stderr_log_path is not None:
                stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
                stderr_handle = stderr_log_path.open("x", encoding="utf-8")

            def default_launcher(args: Sequence[str]) -> GuardedProcess:
                return subprocess.Popen(
                    list(args),
                    start_new_session=True,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                )

            launcher = default_launcher
        process = launcher(argv)
    except BaseException:
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()
        if stream is not None:
            try:
                stream.finish(status="launch_failed", summary=None)
            finally:
                stream.close()
        raise
    runtime: list[dict[str, Any]] = []
    target_gpu_pids: set[int] = set()
    foreign_gpu_pids: set[int] = set()
    command_exit_code: int | None = None
    runtime_error: str | None = None
    try:
        while process.poll() is None:
            sample = _sample_row(
                sampler,
                monotonic_ns=monotonic_ns,
                utc_now=utc_now,
            )
            tree = process_tree(process.pid)
            sample_target_pids: set[int] = set()
            sample_foreign_pids: set[int] = set()
            for process_row in sample["processes"]:
                pid = process_row.get("pid")
                if type(pid) is not int:
                    raise ACSGPUGateError("GPU process PID must be an integer")
                if pid in tree:
                    sample_target_pids.add(pid)
                elif (
                    pid not in baseline_pids
                    and _process_activity(process_row)
                    > thresholds["max_foreign_process_percent"]
                ):
                    sample_foreign_pids.add(pid)
            sample["target_process_tree_pids"] = sorted(tree)
            sample["target_gpu_pids"] = sorted(sample_target_pids)
            sample["foreign_gpu_pids"] = sorted(sample_foreign_pids)
            runtime.append(sample)
            if stream is not None:
                stream.append(phase="runtime", sample=sample)
            target_gpu_pids.update(sample_target_pids)
            foreign_gpu_pids.update(sample_foreign_pids)
            if sample_foreign_pids:
                runtime_error = "FOREIGN_GPU_PROCESS_OBSERVED"
                command_exit_code = _terminate_process_group(
                    process, grace_seconds=termination_grace_seconds
                )
                break
            if process.poll() is None:
                sleeper(sample_interval_seconds)
        if command_exit_code is None:
            command_exit_code = process.wait()
    except BaseException:
        if process.poll() is None:
            _terminate_process_group(process, grace_seconds=termination_grace_seconds)
        if stream is not None:
            try:
                stream.finish(status="runtime_sampling_failed", summary=None)
            finally:
                stream.close()
        raise

    manifest_sha256: str | None = None
    bound_manifest: dict[str, Any] | None = None
    post_sync_boundary_ns: int | None = None
    post_sync_samples: list[dict[str, Any]] = []
    gpu_performance_window: dict[str, Any] | None = None
    terminal_gpu_release_boundary: dict[str, Any] | None = None
    if command_exit_code == 0 and runtime_error is None:
        try:
            bound_manifest, post_sync_boundary_ns = _read_bound_manifest(manifest_path)
            manifest_sha256 = _canonical_sha256(bound_manifest)
            gpu_performance_window = _read_canonical_gpu_performance_window_v1(
                bound_manifest
            )
            if gpu_performance_window is not None:
                window_start_ns = int(gpu_performance_window["start_ns"])
                window_end_ns = int(gpu_performance_window["end_ns"])
                if not (
                    window_start_ns
                    <= post_sync_boundary_ns
                    <= window_end_ns
                ):
                    raise ACSGPUGateError(
                        "gpu_performance_window does not contain the terminal useful-kernel boundary"
                    )
            if require_post_release_zero_residency:
                terminal_gpu_release_boundary = (
                    _read_terminal_gpu_release_boundary_v1(
                        bound_manifest,
                        gpu_performance_window=gpu_performance_window,
                    )
                )
            post_sync_samples = [
                sample
                for sample in runtime
                if sample["monotonic_time_ns"] >= post_sync_boundary_ns
            ]
            if not post_sync_samples:
                runtime_error = "NO_POST_SYNC_GPU_SAMPLE"
        except ACSGPUGateError as exc:
            runtime_error = str(exc)
    async_overlap_evidence = None
    if runtime_error is None and bound_manifest is not None:
        async_overlap_evidence = _async_checkpoint_overlap_evidence_v1(
            bound_manifest,
            runtime,
            active_sm_threshold_percent=thresholds["idle_sm_threshold_percent"],
        )
        if require_async_checkpoint_overlap and not async_overlap_evidence["passed"]:
            runtime_error = "ASYNC_CHECKPOINT_OVERLAP_NOT_PROVEN"
    post_release_target_gpu_residency = None
    if require_post_release_zero_residency and terminal_gpu_release_boundary is not None:
        post_release_target_gpu_residency = _post_release_target_gpu_residency_v1(
            runtime,
            released_perf_ns=int(
                terminal_gpu_release_boundary["released_perf_ns"]
            ),
            launched_pid=process.pid,
            attributed_target_gpu_pids=target_gpu_pids,
        )
        if runtime_error is None:
            if post_release_target_gpu_residency["post_release_sample_count"] == 0:
                runtime_error = "NO_POST_RELEASE_GPU_RESIDENCY_SAMPLE"
            elif post_release_target_gpu_residency["resident_sample_count"]:
                runtime_error = "POST_RELEASE_TARGET_GPU_RESIDENCY_OBSERVED"
    idle_fraction = (
        sum(
            sample["sm_percent"] <= thresholds["idle_sm_threshold_percent"]
            for sample in post_sync_samples
        )
        / len(post_sync_samples)
        if post_sync_samples
        else None
    )
    process_attributed = bool(target_gpu_pids)
    no_competitors = not foreign_gpu_pids
    if gpu_performance_window is None:
        primary_activity_samples = runtime
        post_simulation_acs_tail_samples: list[dict[str, Any]] = []
        post_simulation_acs_tail_activity: dict[str, Any] | None = None
    else:
        window_start_ns = int(gpu_performance_window["start_ns"])
        window_end_ns = int(gpu_performance_window["end_ns"])
        primary_activity_samples = [
            sample
            for sample in runtime
            if window_start_ns
            <= int(sample["monotonic_time_ns"])
            <= window_end_ns
        ]
        post_simulation_acs_tail_samples = [
            sample
            for sample in runtime
            if int(sample["monotonic_time_ns"]) > window_end_ns
        ]
        post_simulation_acs_tail_activity = summarize_gpu_activity_v1(
            post_simulation_acs_tail_samples,
            requested_sample_interval_seconds=sample_interval_seconds,
            idle_sm_threshold_percent=thresholds["idle_sm_threshold_percent"],
        )
        if not primary_activity_samples and runtime_error is None:
            runtime_error = "NO_PRIMARY_GPU_PERFORMANCE_WINDOW_SAMPLE"
    activity_summary = summarize_gpu_activity_v1(
        primary_activity_samples,
        requested_sample_interval_seconds=sample_interval_seconds,
        idle_sm_threshold_percent=thresholds["idle_sm_threshold_percent"],
    )
    passed = (
        command_exit_code == 0
        and runtime_error is None
        and process_attributed
        and no_competitors
        and manifest_sha256 is not None
    )
    post_analysis_summary = {
        "schema_version": "eiel_acs_gpu_post_analysis_summary_v1",
        "gpu_activity": activity_summary,
        "gpu_performance_window": gpu_performance_window,
        "post_simulation_acs_tail_gpu_activity": post_simulation_acs_tail_activity,
        "terminal_gpu_release_boundary": terminal_gpu_release_boundary,
        "post_release_target_gpu_residency": (
            post_release_target_gpu_residency
        ),
        "async_checkpoint_overlap": async_overlap_evidence,
    }
    receipt = {
        **base,
        "status": "passed" if passed else "failed",
        "baseline_admitted": True,
        "launched": True,
        "rejection_reasons": [],
        "baseline_gpu_process_pids": baseline_pids,
        "baseline_samples": baseline,
        "runtime_samples": runtime,
        "sample_count": len(runtime),
        "launched_pid": process.pid,
        "target_gpu_pids": sorted(target_gpu_pids),
        "foreign_gpu_pids": sorted(foreign_gpu_pids),
        "no_competing_gpu_workloads": no_competitors,
        "process_attributed": process_attributed,
        "exclusive_gpu_window": not baseline_pids and no_competitors,
        "command_exit_code": command_exit_code,
        "runtime_error": runtime_error,
        "manifest_sha256": manifest_sha256,
        "gpu_performance_window": gpu_performance_window,
        "primary_activity_sample_count": len(primary_activity_samples),
        "post_simulation_acs_tail_sample_count": len(
            post_simulation_acs_tail_samples
        ),
        "post_simulation_acs_tail_gpu_activity": post_simulation_acs_tail_activity,
        "terminal_gpu_release_boundary": terminal_gpu_release_boundary,
        "post_release_target_gpu_residency": (
            post_release_target_gpu_residency
        ),
        "post_sync_boundary_monotonic_ns": post_sync_boundary_ns,
        "post_sync_sample_count": len(post_sync_samples),
        "idle_sm_threshold_percent": thresholds["idle_sm_threshold_percent"],
        "gpu_idle_fraction": idle_fraction,
        "activity_summary": activity_summary,
        "async_checkpoint_overlap": async_overlap_evidence,
        "post_analysis_summary": post_analysis_summary,
        "finished_at": utc_now(),
    }
    if stdout_log_path is not None:
        receipt["command_stdout_log"] = str(stdout_log_path.resolve())
    if stderr_log_path is not None:
        receipt["command_stderr_log"] = str(stderr_log_path.resolve())
    if stdout_handle is not None:
        stdout_handle.close()
    if stderr_handle is not None:
        stderr_handle.close()
    if stream is not None:
        try:
            stream.finish(status=receipt["status"], summary=post_analysis_summary)
        finally:
            stream.close()
        receipt["sample_stream_path"] = str(stream.path)
        receipt["sample_stream_sha256"] = _file_sha256(stream.path)
    return receipt


__all__ = [
    "ACS_GPU_TELEMETRY_SCHEMA_VERSION",
    "ACS_GPU_SAMPLE_STREAM_SCHEMA_VERSION",
    "ACS_GPU_ACTIVITY_SUMMARY_SCHEMA_VERSION",
    "ACSGPUGateError",
    "NvidiaSMISamplerV1",
    "process_tree_pids_v1",
    "run_guarded_command_v1",
    "summarize_gpu_activity_v1",
]
