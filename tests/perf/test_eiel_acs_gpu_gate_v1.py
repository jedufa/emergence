from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import pytest

from engine.perf.acs_gpu_gate_v1 import (
    run_guarded_command_v1,
    summarize_gpu_activity_v1,
)


def _sample(
    *,
    sm: float = 0.0,
    encoder: float = 0.0,
    processes: list[dict[str, Any]] | None = None,
    sample_started_ns: int | None = None,
) -> dict[str, Any]:
    row = {
        "gpu_index": 0,
        "sm_percent": sm,
        "memory_percent": 0.0,
        "encoder_percent": encoder,
        "decoder_percent": 0.0,
        "memory_used_mib": 100.0,
        "memory_total_mib": 6_144.0,
        "pstate": "P8",
        "processes": processes or [],
    }
    if sample_started_ns is not None:
        row["sample_started_monotonic_ns"] = sample_started_ns
    return row


def _process(pid: int, *, sm: float) -> dict[str, Any]:
    return {
        "pid": pid,
        "type": "C",
        "sm_percent": sm,
        "memory_percent": 1.0 if sm else 0.0,
        "encoder_percent": 0.0,
        "decoder_percent": 0.0,
        "framebuffer_memory_mib": 100.0,
        "command": "python",
    }


class _Sampler:
    def __init__(self, samples: list[dict[str, Any]]):
        self._samples = iter(samples)

    def sample(self) -> dict[str, Any]:
        return next(self._samples)


class _Process:
    pid = 9_999_999

    def __init__(self) -> None:
        self.poll_count = 0
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        self.poll_count += 1
        return None if self.poll_count == 1 and not self.terminated else 0

    def wait(self, timeout: float | None = None) -> int:
        return 143 if self.terminated else 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class _AlreadyExitedProcess:
    pid = 0

    def poll(self) -> int:
        return 0

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def terminate(self) -> None:
        raise AssertionError("must not terminate an exited process")

    def kill(self) -> None:
        raise AssertionError("must not kill an exited process")


class _TwoRuntimeSampleProcess(_Process):
    def poll(self) -> int | None:
        self.poll_count += 1
        return None if self.poll_count <= 3 and not self.terminated else 0


class _NRuntimeSampleProcess(_Process):
    def __init__(self, runtime_sample_count: int) -> None:
        super().__init__()
        self.runtime_sample_count = runtime_sample_count

    def poll(self) -> int | None:
        self.poll_count += 1
        return (
            None
            if self.poll_count <= self.runtime_sample_count and not self.terminated
            else 0
        )


def _manifest(
    path: Path,
    *,
    boundary_ns: int = 250,
    gpu_performance_window: dict[str, Any] | None = None,
) -> None:
    result: dict[str, Any] = {
        "acs_capture_measurements": [
            {
                "last_useful_kernel_completion_observed_host_time_ns": boundary_ns
            }
        ]
    }
    if gpu_performance_window is not None:
        result["tick_native_performance_timeline_binding"] = {
            "schema_version": "eiel_tick_native_performance_timeline_binding_v1",
            "timeline": {
                "schema_version": "eiel_tick_native_performance_timeline_v1",
                "gpu_performance_window": gpu_performance_window,
            },
        }
    path.write_text(
        json.dumps({"status": "completed", "result": result}),
        encoding="utf-8",
    )


def test_gpu_gate_rejects_before_launch_when_baseline_is_busy(tmp_path: Path) -> None:
    launched = False

    def launcher(_command: list[str]) -> _Process:
        nonlocal launched
        launched = True
        return _Process()

    receipt = run_guarded_command_v1(
        ["never-launched"],
        manifest_path=tmp_path / "manifest.json",
        source_commit="a" * 40,
        sampler=_Sampler([_sample(sm=20), _sample(encoder=5)]),
        baseline_sample_count=2,
        sample_interval_seconds=1,
        launcher=launcher,
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200]).__next__,
        utc_now=lambda: "2026-08-13T00:00:00Z",
    )

    assert launched is False
    assert receipt["status"] == "rejected_before_launch"
    assert receipt["launched"] is False
    assert receipt["rejection_reasons"] == [
        "GLOBAL_ENCODER_UTILIZATION_ABOVE_LIMIT",
        "GLOBAL_SM_UTILIZATION_ABOVE_LIMIT",
    ]


def test_gpu_gate_passes_only_with_process_and_manifest_binding(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    target = _Process()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="b" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(),
                _sample(sm=90, processes=[_process(target.pid, sm=90)]),
            ]
        ),
        baseline_sample_count=2,
        sample_interval_seconds=1,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-13T00:00:00Z",
    )

    assert receipt["status"] == "passed"
    assert receipt["process_attributed"] is True
    assert receipt["no_competing_gpu_workloads"] is True
    assert receipt["post_sync_sample_count"] == 1
    assert receipt["gpu_idle_fraction"] == 0.0
    assert len(receipt["manifest_sha256"]) == 64
    assert len(receipt["command_sha256"]) == 64


def test_gpu_gate_allows_only_the_named_persistent_baseline_processes(
    tmp_path: Path,
) -> None:
    allowed = _process(10, sm=10)
    allowed["command"] = "gnome-remote-de"
    unexpected = _process(11, sm=10)
    unexpected["command"] = "python"
    common = {
        "command": ["never-launched"],
        "manifest_path": tmp_path / "manifest.json",
        "source_commit": "d" * 40,
        "baseline_sample_count": 1,
        "sample_interval_seconds": 1,
        "max_baseline_sm_percent": 100,
        "max_baseline_encoder_percent": 100,
        "max_baseline_decoder_percent": 100,
        "max_foreign_process_percent": 0,
        "allowed_baseline_process_commands": ("gnome-remote-de",),
        "sleeper": lambda _seconds: None,
        "monotonic_ns": iter([100, 200]).__next__,
        "utc_now": lambda: "2026-08-13T00:00:00Z",
    }

    permitted = run_guarded_command_v1(
        sampler=_Sampler([_sample(sm=10, processes=[allowed])]),
        launcher=lambda _command: _AlreadyExitedProcess(),
        **common,
    )
    rejected = run_guarded_command_v1(
        sampler=_Sampler([_sample(sm=10, processes=[unexpected])]),
        launcher=lambda _command: _AlreadyExitedProcess(),
        **common,
    )

    assert permitted["baseline_admitted"] is True
    assert permitted["baseline_gpu_process_pids"] == [10]
    assert rejected["status"] == "rejected_before_launch"
    assert rejected["rejection_reasons"] == ["ACTIVE_FOREIGN_GPU_PROCESS"]


def test_gpu_gate_does_not_reclassify_an_admitted_active_baseline_at_runtime(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    target = _TwoRuntimeSampleProcess()
    baseline = _process(10, sm=10)
    baseline["command"] = "gnome-remote-de"

    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="e" * 40,
        sampler=_Sampler(
            [
                _sample(sm=10, encoder=10, processes=[baseline]),
                _sample(
                    sm=90,
                    processes=[baseline, _process(target.pid, sm=90)],
                ),
                _sample(sm=10, encoder=10, processes=[baseline]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        max_baseline_sm_percent=20,
        max_baseline_encoder_percent=20,
        max_foreign_process_percent=0,
        idle_sm_threshold_percent=20,
        allowed_baseline_process_commands=("gnome-remote-de",),
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-13T00:00:00Z",
    )

    assert receipt["status"] == "passed"
    assert receipt["foreign_gpu_pids"] == []
    assert receipt["baseline_gpu_process_pids"] == [10]


def test_gpu_gate_terminates_own_process_on_foreign_gpu_activity(
    tmp_path: Path,
) -> None:
    target = _Process()
    foreign_pid = 8_888_888
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=tmp_path / "manifest.json",
        source_commit="c" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(
                    sm=95,
                    processes=[
                        _process(target.pid, sm=90),
                        _process(foreign_pid, sm=5),
                    ],
                ),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200]).__next__,
        utc_now=lambda: "2026-08-13T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "FOREIGN_GPU_PROCESS_OBSERVED"
    assert receipt["foreign_gpu_pids"] == [foreign_pid]
    assert receipt["no_competing_gpu_workloads"] is False
    assert target.terminated is True


def test_gpu_gate_writes_child_logs_for_the_default_launcher(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    stdout_path = tmp_path / "child.stdout.log"
    stderr_path = tmp_path / "child.stderr.log"

    receipt = run_guarded_command_v1(
        ["/bin/sh", "-c", "printf out; printf err >&2; sleep 0.03"],
        manifest_path=manifest_path,
        source_commit="d" * 40,
        sampler=_Sampler([_sample() for _ in range(32)]),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        process_tree=lambda _pid: {1},
        sleeper=time.sleep,
        monotonic_ns=iter(range(100, 10_000)).__next__,
        utc_now=lambda: "2026-08-13T00:00:00Z",
        stdout_log_path=stdout_path,
        stderr_log_path=stderr_path,
    )

    assert receipt["launched"] is True
    assert stdout_path.read_text(encoding="utf-8") == "out"
    assert stderr_path.read_text(encoding="utf-8") == "err"


def test_gpu_gate_streams_ordered_samples_and_binds_request_hashes(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    request_path = tmp_path / "execution-request.json"
    request_path.write_text('{"request":"test"}\n', encoding="utf-8")
    sample_stream_path = tmp_path / "gpu-samples.jsonl"
    target = _Process()
    command = ["python", "run.py"]

    receipt = run_guarded_command_v1(
        command,
        manifest_path=manifest_path,
        source_commit="f" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(),
                _sample(sm=90, processes=[_process(target.pid, sm=90)]),
            ]
        ),
        baseline_sample_count=2,
        sample_interval_seconds=1,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-13T00:00:00Z",
        sample_stream_path=sample_stream_path,
        execution_request_path=request_path,
    )

    records = [
        json.loads(line)
        for line in sample_stream_path.read_text(encoding="utf-8").splitlines()
    ]
    command_identity = {"argv": command}
    expected_command_hash = hashlib.sha256(
        json.dumps(
            command_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    expected_request_hash = hashlib.sha256(request_path.read_bytes()).hexdigest()

    assert receipt["status"] == "passed"
    assert [record["record_type"] for record in records] == [
        "header",
        "sample",
        "sample",
        "sample",
        "terminal",
    ]
    assert [record["phase"] for record in records[1:4]] == [
        "baseline",
        "baseline",
        "runtime",
    ]
    assert [record["sequence"] for record in records[1:4]] == [0, 1, 2]
    assert records[0]["source_commit"] == "f" * 40
    assert records[0]["manifest_path"] == str(manifest_path.resolve())
    assert records[0]["command_sha256"] == expected_command_hash
    assert records[0]["execution_request_path"] == str(request_path.resolve())
    assert records[0]["execution_request_sha256"] == expected_request_hash
    assert receipt["execution_request_sha256"] == expected_request_hash
    assert receipt["sample_stream_sha256"] == hashlib.sha256(
        sample_stream_path.read_bytes()
    ).hexdigest()
    assert records[-1]["status"] == "passed"
    assert records[-1]["sample_count"] == 3
    assert records[-1]["post_analysis_summary"] == receipt[
        "post_analysis_summary"
    ]


def test_gpu_activity_summary_reports_cadence_quantiles_and_observed_runs() -> None:
    samples = [
        {**_sample(sm=0), "monotonic_time_ns": 0},
        {**_sample(sm=10), "monotonic_time_ns": 1_000_000_000},
        {**_sample(sm=10), "monotonic_time_ns": 3_000_000_000},
        {**_sample(sm=0), "monotonic_time_ns": 4_000_000_000},
        {**_sample(sm=0), "monotonic_time_ns": 7_000_000_000},
        {**_sample(sm=10), "monotonic_time_ns": 10_000_000_000},
    ]

    summary = summarize_gpu_activity_v1(
        samples,
        requested_sample_interval_seconds=1.0,
        idle_sm_threshold_percent=5.0,
    )

    assert summary["sample_count"] == 6
    assert summary["requested_sample_interval_seconds"] == 1.0
    assert summary["observed_span_seconds"] == 10.0
    assert summary["observed_interval_seconds"] == {
        "mean": 2.0,
        "p50": 2.0,
        "p95": 3.0,
        "max": 3.0,
    }
    assert summary["gpu_sm_percent"] == {
        "mean": 5.0,
        "p50": 0.0,
        "p95": 10.0,
        "max": 10.0,
    }
    assert summary["active_sample_count"] == 3
    assert summary["active_sample_fraction"] == 0.5
    assert summary["idle_sample_count"] == 3
    assert summary["idle_sample_fraction"] == 0.5
    assert summary["active_burst_count"] == 2
    assert summary["longest_active_burst_observed_span_seconds"] == 2.0
    assert summary["idle_gap_count"] == 2
    assert summary["longest_idle_gap_observed_span_seconds"] == 3.0


def _canonical_gpu_performance_window(
    *, start_ns: int = 100, end_ns: int = 200
) -> dict[str, Any]:
    return {
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ns": end_ns - start_ns,
        "seconds": (end_ns - start_ns) / 1_000_000_000.0,
        "terminal_checkpoint_tick": 320,
        "post_simulation_acs_tail_excluded": True,
        "measurement_scope": (
            "first_simulation_dispatch_through_terminal_gpu_safe_release_v1"
        ),
    }


def _bind_terminal_gpu_release(
    path: Path, *, terminal_tick: int = 320, released_perf_ns: int = 200
) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = manifest["result"]
    capture = result["acs_capture_measurements"][0]
    capture["coordinate"] = terminal_tick
    capture["gpu_safely_releasable_perf_ns"] = released_perf_ns
    result["terminal_cuda_context_release_receipts"] = [
        {
            "schema_version": "eiel_terminal_cuda_primary_context_release_v1",
            "cuda_device_reset_result": 0,
            "released_perf_ns": released_perf_ns,
        }
    ]
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_gpu_activity_primary_window_excludes_long_terminal_acs_tail(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _manifest(
        manifest_path,
        boundary_ns=150,
        gpu_performance_window=_canonical_gpu_performance_window(),
    )
    target = _NRuntimeSampleProcess(runtime_sample_count=20)
    runtime_samples = [
        _sample(sm=90, processes=[_process(target.pid, sm=90)]),
        _sample(sm=80, processes=[_process(target.pid, sm=80)]),
        *[
            _sample(sm=0, processes=[_process(target.pid, sm=0)])
            for _ in range(8)
        ],
    ]
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="8" * 40,
        sampler=_Sampler([_sample(), *runtime_samples]),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([0, *range(100, 1_100, 100)]).__next__,
        utc_now=lambda: "2026-08-23T00:00:00Z",
    )

    assert receipt["status"] == "passed"
    assert receipt["sample_count"] == 10
    assert len(receipt["runtime_samples"]) == 10
    assert receipt["primary_activity_sample_count"] == 2
    assert receipt["post_simulation_acs_tail_sample_count"] == 8
    assert receipt["activity_summary"]["sample_count"] == 2
    assert receipt["activity_summary"]["gpu_sm_percent"]["mean"] == 85.0
    assert receipt["activity_summary"]["active_sample_fraction"] == 1.0
    tail = receipt["post_simulation_acs_tail_gpu_activity"]
    assert tail["sample_count"] == 8
    assert tail["gpu_sm_percent"]["mean"] == 0.0
    assert tail["active_sample_fraction"] == 0.0
    assert receipt["post_analysis_summary"][
        "post_simulation_acs_tail_gpu_activity"
    ] == tail


def test_gpu_gate_proves_zero_target_residency_after_terminal_release(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _manifest(
        manifest_path,
        boundary_ns=150,
        gpu_performance_window=_canonical_gpu_performance_window(),
    )
    _bind_terminal_gpu_release(manifest_path)
    target = _NRuntimeSampleProcess(runtime_sample_count=8)
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="1" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(
                    sm=90,
                    processes=[_process(target.pid, sm=90)],
                    sample_started_ns=50,
                ),
                _sample(
                    sm=80,
                    processes=[_process(target.pid, sm=80)],
                    sample_started_ns=150,
                ),
                _sample(sample_started_ns=201),
                _sample(sample_started_ns=301),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_post_release_zero_residency=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([0, 100, 200, 300, 400]).__next__,
        utc_now=lambda: "2026-08-24T00:00:00Z",
    )

    assert receipt["status"] == "passed"
    release = receipt["terminal_gpu_release_boundary"]
    assert release["released_perf_ns"] == 200
    assert release["performance_window_end_bound"] is True
    residency = receipt["post_release_target_gpu_residency"]
    assert residency["post_release_sample_count"] == 2
    assert residency["resident_sample_count"] == 0
    assert residency["settlement_grace_seconds"] == 0.0
    assert residency["sample_window_clock_bound"] is True
    assert residency["passed"] is True


def test_gpu_gate_rejects_same_pid_framebuffer_reacquisition_after_release(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _manifest(
        manifest_path,
        boundary_ns=150,
        gpu_performance_window=_canonical_gpu_performance_window(),
    )
    _bind_terminal_gpu_release(manifest_path)
    target = _NRuntimeSampleProcess(runtime_sample_count=8)
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="2" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(
                    sm=90,
                    processes=[_process(target.pid, sm=90)],
                    sample_started_ns=50,
                ),
                _sample(
                    sm=80,
                    processes=[_process(target.pid, sm=80)],
                    sample_started_ns=150,
                ),
                _sample(
                    processes=[_process(target.pid, sm=0)],
                    sample_started_ns=201,
                ),
                _sample(sample_started_ns=301),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_post_release_zero_residency=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([0, 100, 200, 300, 400]).__next__,
        utc_now=lambda: "2026-08-24T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == (
        "POST_RELEASE_TARGET_GPU_RESIDENCY_OBSERVED"
    )
    residency = receipt["post_release_target_gpu_residency"]
    assert residency["resident_sample_count"] == 1
    assert residency["resident_rows"][0]["pid"] == target.pid
    assert residency["max_framebuffer_memory_mib"] == 100.0


def test_gpu_gate_rejects_positive_residency_in_release_straddling_sample(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _manifest(
        manifest_path,
        boundary_ns=150,
        gpu_performance_window=_canonical_gpu_performance_window(),
    )
    _bind_terminal_gpu_release(manifest_path)
    target = _NRuntimeSampleProcess(runtime_sample_count=6)
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="4" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(
                    sm=90,
                    processes=[_process(target.pid, sm=90)],
                    sample_started_ns=50,
                ),
                _sample(
                    processes=[_process(target.pid, sm=0)],
                    sample_started_ns=150,
                ),
                _sample(sample_started_ns=251),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_post_release_zero_residency=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([0, 100, 250, 300]).__next__,
        utc_now=lambda: "2026-08-24T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == (
        "POST_RELEASE_TARGET_GPU_RESIDENCY_OBSERVED"
    )
    residency = receipt["post_release_target_gpu_residency"]
    assert residency["straddling_sample_count"] == 1
    assert residency["straddling_resident_sample_count"] == 1
    assert residency["resident_rows"][0]["window_relation"] == "straddles_release"


def test_gpu_gate_rejects_missing_post_release_residency_sample(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _manifest(
        manifest_path,
        boundary_ns=150,
        gpu_performance_window=_canonical_gpu_performance_window(),
    )
    _bind_terminal_gpu_release(manifest_path)
    target = _NRuntimeSampleProcess(runtime_sample_count=4)
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="3" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(
                    sm=90,
                    processes=[_process(target.pid, sm=90)],
                    sample_started_ns=50,
                ),
                _sample(
                    sm=80,
                    processes=[_process(target.pid, sm=80)],
                    sample_started_ns=150,
                ),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_post_release_zero_residency=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([0, 100, 200]).__next__,
        utc_now=lambda: "2026-08-24T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "NO_POST_RELEASE_GPU_RESIDENCY_SAMPLE"
    assert receipt["post_release_target_gpu_residency"][
        "post_release_sample_count"
    ] == 0


def test_gpu_gate_keeps_foreign_process_monitoring_outside_primary_window(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _manifest(
        manifest_path,
        boundary_ns=150,
        gpu_performance_window=_canonical_gpu_performance_window(),
    )
    target = _NRuntimeSampleProcess(runtime_sample_count=4)
    foreign = _process(123_456, sm=10)
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="9" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=90, processes=[_process(target.pid, sm=90)]),
                _sample(
                    sm=0,
                    processes=[_process(target.pid, sm=0), foreign],
                ),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([0, 100, 300]).__next__,
        utc_now=lambda: "2026-08-23T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "FOREIGN_GPU_PROCESS_OBSERVED"
    assert receipt["sample_count"] == 2
    assert receipt["runtime_samples"][-1]["foreign_gpu_pids"] == [foreign["pid"]]


def test_gpu_gate_fails_closed_when_declared_primary_window_has_no_samples(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _manifest(
        manifest_path,
        boundary_ns=150,
        gpu_performance_window=_canonical_gpu_performance_window(),
    )
    target = _NRuntimeSampleProcess(runtime_sample_count=2)
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="b" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=0, processes=[_process(target.pid, sm=0)]),
                _sample(sm=0, processes=[_process(target.pid, sm=0)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([0, 300, 400]).__next__,
        utc_now=lambda: "2026-08-23T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "NO_PRIMARY_GPU_PERFORMANCE_WINDOW_SAMPLE"
    assert receipt["primary_activity_sample_count"] == 0
    assert receipt["post_simulation_acs_tail_sample_count"] == 1


@pytest.mark.parametrize("malformation", ["missing_end", "inverted"])
def test_gpu_gate_fails_closed_on_malformed_declared_gpu_window(
    tmp_path: Path, malformation: str
) -> None:
    manifest_path = tmp_path / "manifest.json"
    window = _canonical_gpu_performance_window()
    if malformation == "missing_end":
        del window["end_ns"]
    else:
        window["end_ns"] = window["start_ns"]
        window["duration_ns"] = 0
        window["seconds"] = 0.0
    _manifest(
        manifest_path,
        boundary_ns=150,
        gpu_performance_window=window,
    )
    target = _Process()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="a" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=90, processes=[_process(target.pid, sm=90)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([0, 150]).__next__,
        utc_now=lambda: "2026-08-23T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] is not None
    assert "gpu_performance_window" in receipt["runtime_error"]


def _async_manifest(path: Path, *, slots: int = 2) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "result": {
                    "terminal_next_absolute_tick": 320,
                    "resident_owner_max_simultaneous": 1,
                    "resident_owner_live_count_terminal": 0,
                    "resident_owner_lifetime_measurements": [
                        {
                            "owner_identity": "b" * 64,
                            "started_perf_ns": 10,
                            "released_perf_ns": 500,
                            "live_owner_count_before_release": 1,
                            "live_owner_count_after_release": 0,
                        },
                    ],
                    "expected_resident_owner_transition_ticks": [160],
                    "observed_resident_owner_transition_ticks": [160],
                    "resident_owner_epoch_transfer_receipts": [
                        {
                            "schema_version": "eiel_resident_owner_epoch_transfer_v1",
                            "physical_owner_identity": "f" * 64,
                            "prior_owner_identity": "a" * 64,
                            "successor_owner_identity": "b" * 64,
                            "boundary_next_absolute_tick": 160,
                            "successor_span_end_tick": 320,
                            "physical_owner_reused": True,
                            "cuda_allocation_recreated": False,
                            "simultaneous_physical_owner_count": 1,
                        }
                    ],
                    "acs_checkpoint_staging": {
                        "slots": slots,
                        "queue_occupancy_high_water": slots,
                "worker_accepts_cuda_objects": False,
                "worker_tensor_device_policy": "forced_cpu_context_v1",
                "worker_accepts_callbacks": False,
                        "queued_cuda_references": 0,
                        "queued_cuda_callbacks": 0,
                        "detached_source_lease_count": 0,
                        "source_consumer_leases_active": 0,
                        "source_lease_spool_bytes_active": 0,
                        "temporary_capture_files_active": 0,
                        "temporary_capture_file_high_water": slots,
                        "capture_pool_count": 1,
                        "capture_pool_model": "one_shared_two_slab_pool_not_per_slot",
                    },
                    "checkpoint_catalog": {},
                    "acs_capture_measurements": [
                        {
                            "coordinate": 160,
                            "execution_device": "cuda:0",
                            "capture_started_perf_ns": 100,
                            "capture_completed_perf_ns": 150,
                            "capture_barrier_time_ns": 10,
                            "detached_capture_submitted_perf_ns": 160,
                            "detached_writer_started_perf_ns": 175,
                            "detached_writer_finished_perf_ns": 350,
                            "detached_source_consumer_required": True,
                            "owner_transition": True,
                            "successor_owner_prepare_started_perf_ns": 180,
                            "successor_owner_prepare_completed_perf_ns": 240,
                            "detached_source_released_perf_ns": 450,
                            "successor_dispatch_started_perf_ns": 250,
                            "successor_dispatch_completed_perf_ns": 400,
                            "last_useful_kernel_completion_observed_host_time_ns": 150,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def test_gpu_gate_requires_measured_nonterminal_async_checkpoint_overlap(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="1" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=75, processes=[_process(target.pid, sm=75)]),
                _sample(sm=80, processes=[_process(target.pid, sm=80)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "passed"
    overlap = receipt["async_checkpoint_overlap"]
    assert overlap["passed"] is True
    assert overlap["queue_occupancy_high_water"] == 2
    assert overlap["worker_accepts_cuda_objects"] is False
    assert overlap["qualifying_nonterminal_overlap_count"] == 1
    assert overlap["owner_transition_window_count"] == 1
    assert overlap["owner_transition_overlap_count"] == 1
    assert overlap["resident_owner_max_simultaneous"] == 1
    window = overlap["checkpoint_windows"][0]
    assert window["capture_then_writer_ordered"] is True
    assert window["source_lease_ordered"] is True
    assert window["successor_prepare_overlapped_writer"] is True


def test_gpu_gate_separates_in_place_owner_epoch_transfer_from_checkpoint_overlap(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = manifest["result"]["acs_capture_measurements"][0]
    capture["coordinate"] = 200
    capture["owner_transition"] = False
    capture["detached_source_consumer_required"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="e" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=75, processes=[_process(target.pid, sm=75)]),
                _sample(sm=80, processes=[_process(target.pid, sm=80)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    overlap = receipt["async_checkpoint_overlap"]
    assert receipt["status"] == "passed"
    assert overlap["exact_owner_transition_coverage"] is True
    assert overlap["owner_epoch_transfer_ticks"] == [160]
    assert overlap["owner_epoch_transfer_rows_valid"] is True
    assert overlap["physical_owner_continuity_measured"] is True
    assert overlap["owner_transition_window_count"] == 0
    assert overlap["qualifying_nonterminal_overlap_count"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("physical_owner_reused", False),
        ("cuda_allocation_recreated", True),
        ("simultaneous_physical_owner_count", 2),
    ),
)
def test_gpu_gate_rejects_invalid_in_place_owner_epoch_transfer(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"]["resident_owner_epoch_transfer_receipts"][0][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="f" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=75, processes=[_process(target.pid, sm=75)]),
                _sample(sm=80, processes=[_process(target.pid, sm=80)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    overlap = receipt["async_checkpoint_overlap"]
    assert overlap["owner_epoch_transfer_rows_valid"] is False
    assert overlap["physical_owner_continuity_measured"] is False


def test_gpu_gate_accepts_exact_native_stream_completion_when_nvml_lags(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = manifest["result"]["acs_capture_measurements"][0]
    capture.update(
        {
            "successor_last_useful_kernel_completion_observed_host_time_ns": 300,
            "successor_final_stream_synchronization_time_ns": 25,
            "successor_final_host_synchronizations": 1,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="2" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=0, processes=[_process(target.pid, sm=0)]),
                _sample(sm=0, processes=[_process(target.pid, sm=0)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "passed"
    window = receipt["async_checkpoint_overlap"]["checkpoint_windows"][0]
    assert window["sampled_gpu_activity_overlap_observed"] is False
    assert window["native_gpu_completion_ordered"] is True
    assert window["native_gpu_completion_overlapped_writer"] is True
    assert window["gpu_simulation_overlap_observed"] is True
    assert window["gpu_overlap_proof_method"] == "native_stream_completion_v1"


@pytest.mark.parametrize(
    ("completion", "synchronizations", "execution_device"),
    ((360, 1, "cuda:0"), (300, 0, "cuda:0"), (401, 1, "cuda:0"), (300, 1, "cpu")),
)
def test_gpu_gate_rejects_invalid_or_nonoverlapping_native_completion(
    tmp_path: Path,
    completion: int,
    synchronizations: int,
    execution_device: str,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = manifest["result"]["acs_capture_measurements"][0]
    capture.update(
        {
            "execution_device": execution_device,
            "successor_last_useful_kernel_completion_observed_host_time_ns": (
                completion
            ),
            "successor_final_stream_synchronization_time_ns": 25,
            "successor_final_host_synchronizations": synchronizations,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="9" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=0, processes=[_process(target.pid, sm=0)]),
                _sample(sm=0, processes=[_process(target.pid, sm=0)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "ASYNC_CHECKPOINT_OVERLAP_NOT_PROVEN"
    window = receipt["async_checkpoint_overlap"]["checkpoint_windows"][0]
    assert window["native_gpu_completion_overlapped_writer"] is False
    assert window["gpu_simulation_overlap_observed"] is False


def test_gpu_gate_combines_safe_owner_handoff_with_adjacent_gpu_writer_overlap(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    owner_capture = manifest["result"]["acs_capture_measurements"][0]
    owner_capture["detached_writer_finished_perf_ns"] = 240
    manifest["result"]["acs_capture_measurements"].append(
        {
            "coordinate": 319,
            "capture_started_perf_ns": 490,
            "capture_completed_perf_ns": 500,
            "capture_barrier_time_ns": 10,
            "detached_capture_submitted_perf_ns": 510,
            "detached_writer_started_perf_ns": 525,
            "detached_writer_finished_perf_ns": 700,
            "detached_source_consumer_required": False,
            "owner_transition": False,
            "successor_dispatch_started_perf_ns": 550,
            "successor_dispatch_completed_perf_ns": 650,
            "last_useful_kernel_completion_observed_host_time_ns": 500,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="c" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=0, processes=[_process(target.pid, sm=0)]),
                _sample(sm=80, processes=[_process(target.pid, sm=80)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 600]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "passed", {
        "runtime_error": receipt.get("runtime_error"),
        "exit_code": receipt.get("exit_code"),
        "async_checkpoint_overlap": receipt.get("async_checkpoint_overlap"),
    }
    overlap = receipt["async_checkpoint_overlap"]
    assert overlap["passed"] is True
    assert overlap["exact_owner_transition_coverage"] is True
    assert overlap["owner_transition_overlap_count"] == 1
    assert overlap["qualifying_nonterminal_overlap_count"] == 1
    owner_window, adjacent_window = overlap["checkpoint_windows"]
    assert owner_window["transition_handoff_nonblocking"] is True
    assert owner_window["gpu_simulation_overlap_observed"] is False
    assert owner_window["capture_clock_ordered"] is True
    assert owner_window["capture_started_perf_ns"] == 100
    assert owner_window["capture_completed_perf_ns"] == 150
    assert owner_window["capture_submitted_perf_ns"] == 160
    assert owner_window["gpu_safe_release_ordered"] is True
    assert owner_window["release_authority_perf_ns"] == 450
    assert owner_window["release_authority_kind"] == "source"
    assert adjacent_window["owner_transition"] is False
    assert adjacent_window["gpu_simulation_overlap_observed"] is True


def test_gpu_gate_requires_two_slot_high_water_for_physical_async_acceptance(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"]["acs_checkpoint_staging"][
        "queue_occupancy_high_water"
    ] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="3" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=75, processes=[_process(target.pid, sm=75)]),
                _sample(sm=80, processes=[_process(target.pid, sm=80)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "ASYNC_CHECKPOINT_OVERLAP_NOT_PROVEN"
    assert receipt["async_checkpoint_overlap"]["queue_occupancy_high_water"] == 1
    assert receipt["async_checkpoint_overlap"]["owner_transition_window_count"] == 1
    assert receipt["async_checkpoint_overlap"]["owner_transition_overlap_count"] == 1
    assert receipt["async_checkpoint_overlap"]["resident_owner_max_simultaneous"] == 1


@pytest.mark.parametrize("malformation", [
    "missing_queued_cuda_references",
    "missing_queued_cuda_callbacks",
    "missing_capture_barrier",
    "explicit_gpu_release_before_writer",
    "capture_completed_after_submission",
])
def test_gpu_gate_requires_explicit_queue_zero_and_capture_clock_authority(
    tmp_path: Path, malformation: str
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if malformation == "missing_capture_barrier":
        manifest["result"]["acs_capture_measurements"][0].pop(
            "capture_barrier_time_ns"
        )
    elif malformation == "explicit_gpu_release_before_writer":
        manifest["result"]["acs_capture_measurements"][0][
            "gpu_safely_releasable_perf_ns"
        ] = 100
    elif malformation == "capture_completed_after_submission":
        manifest["result"]["acs_capture_measurements"][0][
            "capture_completed_perf_ns"
        ] = 161
    else:
        manifest["result"]["acs_checkpoint_staging"].pop(
            "queued_cuda_references"
            if malformation == "missing_queued_cuda_references"
            else "queued_cuda_callbacks"
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="e" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=75, processes=[_process(target.pid, sm=75)]),
                _sample(sm=80, processes=[_process(target.pid, sm=80)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "ASYNC_CHECKPOINT_OVERLAP_NOT_PROVEN"
    assert receipt["async_checkpoint_overlap"]["passed"] is False


def test_gpu_gate_rejects_terminal_source_lease_or_third_temporary_file(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    staging = manifest["result"]["acs_checkpoint_staging"]
    staging["detached_source_lease_count"] = 1
    staging["source_consumer_leases_active"] = 1
    staging["source_lease_spool_bytes_active"] = 4096
    staging["temporary_capture_files_active"] = 1
    staging["temporary_capture_file_high_water"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="4" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=75, processes=[_process(target.pid, sm=75)]),
                _sample(sm=80, processes=[_process(target.pid, sm=80)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    overlap = receipt["async_checkpoint_overlap"]
    assert overlap["source_file_lifecycle_bounded"] is False
    assert overlap["temporary_capture_file_high_water"] == 3


def test_gpu_gate_rejects_target_activity_outside_exact_dispatch_writer_intersection(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="8" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=75, processes=[_process(target.pid, sm=75)]),
                _sample(sm=0, processes=[_process(target.pid, sm=0)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "ASYNC_CHECKPOINT_OVERLAP_NOT_PROVEN"
    window = receipt["async_checkpoint_overlap"]["checkpoint_windows"][0]
    assert window["writer_window_active_target_sample_count"] == 1
    assert window["dispatch_overlap_sample_count"] == 1
    assert window["dispatch_overlap_active_target_sample_count"] == 0
    assert window["gpu_simulation_overlap_observed"] is False
    assert window["asynchronous_progress_observed"] is True
    assert window["transition_handoff_nonblocking"] is True
    assert (
        receipt["async_checkpoint_overlap"][
            "qualifying_nonterminal_overlap_count"
        ]
        == 0
    )


@pytest.mark.parametrize(
    ("expected_ticks", "observed_ticks"),
    (
        ([], [160]),
        ([160, 320], [160]),
        ([160], [160, 320]),
    ),
)
def test_gpu_gate_rejects_missing_or_extra_owner_boundary_coverage(
    tmp_path: Path,
    expected_ticks: list[int],
    observed_ticks: list[int],
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"]["expected_resident_owner_transition_ticks"] = expected_ticks
    manifest["result"]["observed_resident_owner_transition_ticks"] = observed_ticks
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="9" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=75, processes=[_process(target.pid, sm=75)]),
                _sample(sm=80, processes=[_process(target.pid, sm=80)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "ASYNC_CHECKPOINT_OVERLAP_NOT_PROVEN"
    overlap = receipt["async_checkpoint_overlap"]
    assert overlap["exact_owner_transition_coverage"] is False
    assert overlap["expected_owner_transition_ticks"] == expected_ticks
    assert overlap["observed_owner_transition_ticks"] == observed_ticks


def test_gpu_gate_rejects_hardcoded_owner_max_without_lifetime_measurements(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["result"]["resident_owner_lifetime_measurements"]
    # A hardcoded max of one is not a measured owner lifetime proof.
    assert manifest["result"]["resident_owner_max_simultaneous"] == 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="a" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=75, processes=[_process(target.pid, sm=75)]),
                _sample(sm=80, processes=[_process(target.pid, sm=80)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "ASYNC_CHECKPOINT_OVERLAP_NOT_PROVEN"
    overlap = receipt["async_checkpoint_overlap"]
    assert overlap["resident_owner_max_simultaneous"] == 1
    assert overlap["resident_owner_lifetime_rows_valid"] is False
    assert overlap["owner_lifecycle_measured"] is False


def test_gpu_gate_fails_when_async_slots_exist_without_measured_overlap(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path, slots=1)
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="2" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=0, processes=[_process(target.pid, sm=0)]),
                _sample(sm=0, processes=[_process(target.pid, sm=0)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "ASYNC_CHECKPOINT_OVERLAP_NOT_PROVEN"
    assert receipt["async_checkpoint_overlap"]["passed"] is False


def test_gpu_gate_rejects_missing_owner_transition_source_lease_proof(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = manifest["result"]["acs_capture_measurements"][0]
    del capture["detached_source_released_perf_ns"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="5" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=75, processes=[_process(target.pid, sm=75)]),
                _sample(sm=80, processes=[_process(target.pid, sm=80)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "ASYNC_CHECKPOINT_OVERLAP_NOT_PROVEN"
    overlap = receipt["async_checkpoint_overlap"]
    assert overlap["passed"] is False
    assert overlap["owner_transition_window_count"] == 1
    assert overlap["owner_transition_overlap_count"] == 0


def test_gpu_gate_rejects_disordered_owner_transition_lease_clocks(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = manifest["result"]["acs_capture_measurements"][0]
    capture["successor_owner_prepare_completed_perf_ns"] = 170
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="6" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=75, processes=[_process(target.pid, sm=75)]),
                _sample(sm=80, processes=[_process(target.pid, sm=80)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "ASYNC_CHECKPOINT_OVERLAP_NOT_PROVEN"
    overlap = receipt["async_checkpoint_overlap"]
    assert overlap["passed"] is False
    assert overlap["owner_transition_window_count"] == 1
    assert overlap["owner_transition_overlap_count"] == 0


def test_gpu_gate_rejects_source_release_before_successor_owner_consumes_it(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = manifest["result"]["acs_capture_measurements"][0]
    capture["detached_source_released_perf_ns"] = 260
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="d" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=75, processes=[_process(target.pid, sm=75)]),
                _sample(sm=80, processes=[_process(target.pid, sm=80)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "ASYNC_CHECKPOINT_OVERLAP_NOT_PROVEN"
    window = receipt["async_checkpoint_overlap"]["checkpoint_windows"][0]
    assert window["source_lease_ordered"] is True
    assert window["source_handoff_ordered"] is False
    assert window["transition_handoff_nonblocking"] is False


def test_gpu_gate_accepts_completed_writer_handoff_but_still_requires_gpu_overlap(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _async_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = manifest["result"]["acs_capture_measurements"][0]
    capture["detached_writer_finished_perf_ns"] = 179
    capture["successor_owner_prepare_started_perf_ns"] = 180
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _TwoRuntimeSampleProcess()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="7" * 40,
        sampler=_Sampler(
            [
                _sample(),
                _sample(sm=75, processes=[_process(target.pid, sm=75)]),
                _sample(sm=80, processes=[_process(target.pid, sm=80)]),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        require_async_checkpoint_overlap=True,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200, 300]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert receipt["runtime_error"] == "ASYNC_CHECKPOINT_OVERLAP_NOT_PROVEN"
    overlap = receipt["async_checkpoint_overlap"]
    assert overlap["passed"] is False
    assert overlap["owner_transition_window_count"] == 1
    assert overlap["owner_transition_overlap_count"] == 1
    assert (
        overlap["checkpoint_windows"][0]["transition_handoff_nonblocking"]
        is True
    )
    assert overlap["qualifying_nonterminal_overlap_count"] == 0


def test_gpu_gate_allows_quiescent_foreign_graphics_context_during_runtime(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path, boundary_ns=150)
    target = _Process()
    receipt = run_guarded_command_v1(
        ["python", "run.py"],
        manifest_path=manifest_path,
        source_commit="4" * 40,
        sampler=_Sampler(
            [
                _sample(processes=[_process(99, sm=0)]),
                _sample(
                    sm=75,
                    processes=[
                        _process(99, sm=0),
                        _process(target.pid, sm=75),
                    ],
                ),
            ]
        ),
        baseline_sample_count=1,
        sample_interval_seconds=1,
        launcher=lambda _command: target,
        process_tree=lambda _pid: {target.pid},
        sleeper=lambda _seconds: None,
        monotonic_ns=iter([100, 200]).__next__,
        utc_now=lambda: "2026-08-22T00:00:00Z",
    )

    assert receipt["status"] == "passed"
    assert receipt["foreign_gpu_pids"] == []
