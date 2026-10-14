from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

from tools import resource_monitor as erm


def _completed(
    stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=returncode, stdout=stdout, stderr=""
    )


def _gpu_runner(*, gpu_stdout: str, process_stdout: str = ""):
    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        joined = " ".join(command)
        if "--query-compute-apps" in joined:
            return _completed(process_stdout)
        return _completed(gpu_stdout)

    return run


def _missing_gpu_runner(
    command: list[str], **_: object
) -> subprocess.CompletedProcess[str]:
    return _completed("", returncode=1)


def _write_proc(
    proc: Path, *, process_count: int = 0, repo_root: Path | None = None
) -> None:
    proc.mkdir(parents=True)
    (proc / "cpuinfo").write_text(
        "\n".join(
            [
                "processor\t: 0",
                "physical id\t: 0",
                "core id\t\t: 0",
                "model name\t: Test CPU",
                "",
                "processor\t: 1",
                "physical id\t: 0",
                "core id\t\t: 1",
                "model name\t: Test CPU",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (proc / "meminfo").write_text(
        "\n".join(
            [
                "MemTotal:       16777216 kB",
                "MemAvailable:   8388608 kB",
                "SwapTotal:       4194304 kB",
                "SwapFree:        4194304 kB",
            ]
        ),
        encoding="utf-8",
    )
    (proc / "stat").write_text(
        "cpu  100 0 100 800 0 0 0 0 0 0\nbtime 1000\n", encoding="utf-8"
    )
    for pid in range(2000, 2000 + process_count):
        pid_dir = proc / str(pid)
        pid_dir.mkdir()
        (pid_dir / "cmdline").write_bytes(
            b"python\0tools/scripts/run_registry_parallel.py\0--batch-tag\0test-campaign\0--device\0cuda\0"
        )
        (pid_dir / "stat").write_text(
            f"{pid} (python) S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100\n",
            encoding="utf-8",
        )
        if repo_root is not None:
            (pid_dir / "cwd").symlink_to(repo_root)


def _config(
    tmp_path: Path, *, min_safe_storage_bytes: int = 0
) -> erm.MonitorConfig:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    live = runtime / "live/simulation-slots"
    proc = tmp_path / "proc"
    repo.mkdir()
    live.mkdir(parents=True)
    _write_proc(proc, repo_root=repo)
    return erm.MonitorConfig(
        repo_root=repo,
        runtime_root=runtime,
        live_slots_root=live,
        proc_root=proc,
        cpu_sample_interval_s=0,
        min_safe_storage_bytes=min_safe_storage_bytes,
    )


def test_parser_behavior() -> None:
    mem = erm.parse_meminfo("MemTotal: 1024 kB\nMemAvailable: 512 kB\n")
    assert mem == {"MemTotal": 1024 * 1024, "MemAvailable": 512 * 1024}
    assert erm.parse_nvidia_smi_gpu_csv("GTX 1660 Ti, 7, 1024, 4096, 5120, 61")[0] == {
        "index": 0,
        "model": "GTX 1660 Ti",
        "utilization_percent": 7.0,
        "vram_used_bytes": 1024 * 1024**2,
        "vram_free_bytes": 4096 * 1024**2,
        "vram_total_bytes": 5120 * 1024**2,
        "temperature_c": 61.0,
    }
    assert erm.parse_nvidia_smi_process_csv("1234, python, 2048")[0]["pid"] == 1234
    assert erm.parse_nvidia_smi_process_csv(
        "GPU-TEST, 1234, python, 2048"
    )[0]["gpu_uuid"] == "GPU-TEST"
    assert erm.cpu_utilization_percent((100, 80), (200, 120)) == 60.0


def test_process_identity_observer_is_pid_reuse_and_unknown_safe(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    boot = proc / "sys/kernel/random"
    boot.mkdir(parents=True)
    (boot / "boot_id").write_text("boot-test\n", encoding="utf-8")
    absent = erm.observe_process_identity_v1(
        4242, "linux-proc:boot-test:123", proc_root=proc
    )
    assert absent["state"] == "ABSENT"

    pid_dir = proc / "4242"
    pid_dir.mkdir()
    (pid_dir / "stat").write_text(
        "4242 (python worker) " + " ".join(["S", *(["0"] * 18), "123"]),
        encoding="utf-8",
    )
    live = erm.observe_process_identity_v1(
        4242, "linux-proc:boot-test:123", proc_root=proc
    )
    assert live["state"] == "LIVE"
    reused = erm.observe_process_identity_v1(
        4242, "linux-proc:boot-test:122", proc_root=proc
    )
    assert reused["state"] == "REUSED"

    (boot / "boot_id").unlink()
    unknown = erm.observe_process_identity_v1(
        4242, "linux-proc:boot-test:123", proc_root=proc
    )
    assert unknown["state"] == "UNKNOWN"


def test_legacy_process_start_ticks_observer_preserves_historical_identity(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    absent = erm.observe_legacy_process_start_ticks_v1(4242, 123, proc_root=proc)
    assert absent["state"] == "ABSENT"

    pid_dir = proc / "4242"
    pid_dir.mkdir()
    (pid_dir / "stat").write_text(
        "4242 (python worker) " + " ".join(["S", *(["0"] * 18), "123"]),
        encoding="utf-8",
    )
    live = erm.observe_legacy_process_start_ticks_v1(4242, 123, proc_root=proc)
    assert live["state"] == "LIVE"
    reused = erm.observe_legacy_process_start_ticks_v1(4242, 122, proc_root=proc)
    assert reused["state"] == "REUSED"

    (pid_dir / "stat").write_text("malformed", encoding="utf-8")
    unknown = erm.observe_legacy_process_start_ticks_v1(4242, 123, proc_root=proc)
    assert unknown["state"] == "UNKNOWN"


def test_gpu_process_query_failure_is_not_authoritative_empty() -> None:
    calls = 0

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if "--query-compute-apps" in " ".join(command):
            return _completed("", returncode=1)
        return _completed("GPU-TEST, GTX 1660 Ti, 0, 256, 5888, 6144, 45")

    gpu = erm.collect_gpu(runner=runner)
    assert calls == 2
    assert gpu["available"] is True
    assert gpu["processes"] == []
    assert gpu["process_query_complete"] is False


def test_malformed_successful_gpu_process_query_is_not_authoritative_empty() -> None:
    gpu = erm.collect_gpu(
        runner=_gpu_runner(
            gpu_stdout="GPU-TEST, GTX 1660 Ti, 0, 256, 5888, 6144, 45",
            process_stdout="GPU-TEST, not-a-pid, python, 64",
        )
    )

    assert gpu["available"] is True
    assert gpu["processes"] == []
    assert gpu["process_query_complete"] is False
    assert gpu["error"] == "nvidia-smi compute-process query contained malformed rows"


def test_gpu_capacity_profile_is_carried_into_observation() -> None:
    gpu = erm.collect_gpu(
        runner=_gpu_runner(
            gpu_stdout="GTX 1660 Ti, 0, 256, 5888, 6144, 45"
        ),
        usable_vram_gib=5.6,
    )

    assert gpu["gpus"][0]["usable_vram_bytes"] == int(5.6 * 1024**3)


def test_gpu_classification_requires_observed_compute_pid() -> None:
    webui_command = (
        "/tmp/emergence-test/user/.venvs/emergence-py314-cuda/bin/python "
        "-m infra.webui.server --port 8791"
    )

    assert erm._classify_device(webui_command, set(), 454721) == "unknown"
    assert erm._classify_device(webui_command, {454721}, 454721) == "gpu"


def test_missing_gpu_degrades_to_unknown(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = erm.collect_report(
        config, runner=_missing_gpu_runner, sleep=lambda _: None
    )
    assert report["gpu"]["available"] is False
    assert report["recommendation"]["value"] == "UNKNOWN"
    assert report["recommendation"]["confidence"] == "LOW"
    assert report["blockers"][0]["kind"] == "gpu_telemetry_unavailable"
    assert report["blocker_classification"] == ["UNKNOWN"]


def test_missing_ste_degrades_gracefully(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert erm.collect_ste(config) == {
        "available": False,
        "status": "STE unavailable",
        "resource_efficiency": {},
        "scientific_efficiency": {},
        "scientific_discovery_efficiency": {},
        "primary_science_speed": {},
        "scientific_throughput": {},
        "qualification_throughput": {},
        "evidence_throughput": {},
        "throughput_taxonomy": {},
    }


def test_no_running_campaigns_is_safe_when_headroom_exists(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = erm.collect_report(
        config,
        runner=_gpu_runner(gpu_stdout="GTX 1660 Ti, 0, 256, 5888, 6144, 45"),
        sleep=lambda _: None,
    )
    assert report["running_eiel_workloads"] == []
    assert report["slot_heartbeats"] == []
    assert report["recommendation"] == {
        "value": "SAFE",
        "confidence": "HIGH",
        "reason": "GPU is idle, RAM headroom is sufficient, and no conflicting CUDA workload was detected.",
    }


def test_one_running_campaign_heartbeat_yields_caution(tmp_path: Path) -> None:
    config = _config(tmp_path)
    heartbeat = {
        "pid": 123,
        "status": "running",
        "device": "gpu",
        "campaign_id": "w32-qualification",
        "run_id": "w32-qualification:batch-0001",
        "current_generation": 10,
        "target_generation": 20,
        "eta_seconds": 50.0,
        "updated_at": "2026-08-02T14:00:00Z",
    }
    (config.live_slots_root / "slot.json").write_text(
        json.dumps(heartbeat), encoding="utf-8"
    )
    report = erm.collect_report(
        config,
        runner=_gpu_runner(gpu_stdout="GTX 1660 Ti, 0, 256, 5888, 6144, 45"),
        sleep=lambda _: None,
        now=dt.datetime(2026, 8, 2, 14, 0, 30, tzinfo=dt.timezone.utc),
    )
    assert len(report["slot_heartbeats"]) == 1
    assert report["slot_heartbeats"][0]["eta_seconds"] == 50.0
    assert report["slot_heartbeats"][0]["elapsed_runtime_seconds"] == "UNKNOWN"
    assert report["slot_heartbeats"][0]["checkpoint_progress"] == "UNKNOWN"
    assert report["gpu"]["occupancy_state"] == "CONTENDED"
    assert report["gpu"]["compute_process_count"] == 0
    assert report["recommendation"]["value"] == "WAIT"
    assert report["recommendation"]["confidence"] == "HIGH"
    assert report["blocker_classification"] == ["GPU_BLOCKER"]
    assert report["recommended_poll_interval_seconds"] == 30
    assert report["next_check_after_seconds"] == 30


def test_multiple_running_campaigns_are_reported_without_eta_invention(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    for index in range(2):
        (config.live_slots_root / f"slot-{index}.json").write_text(
            json.dumps(
                {
                    "pid": 100 + index,
                    "status": "running",
                    "device": "gpu",
                    "campaign_id": f"campaign-{index}",
                    "run_id": f"campaign-{index}:batch",
                    "current_generation": index,
                    "target_generation": 10,
                    "updated_at": "2026-08-02T14:00:00Z",
                }
            ),
            encoding="utf-8",
        )
    report = erm.collect_report(
        config,
        runner=_gpu_runner(gpu_stdout="GTX 1660 Ti, 0, 256, 5888, 6144, 45"),
        sleep=lambda _: None,
        now=dt.datetime(2026, 8, 2, 14, 0, 30, tzinfo=dt.timezone.utc),
    )
    assert len(report["slot_heartbeats"]) == 2
    assert {row["eta_seconds"] for row in report["slot_heartbeats"]} == {"UNKNOWN"}
    assert report["recommendation"]["value"] == "WAIT"


def test_cpu_caution_is_distinct_from_gpu_blocker(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (config.live_slots_root / "slot.json").write_text(
        json.dumps(
            {
                "pid": 123,
                "status": "running",
                "device": "cpu",
                "campaign_id": "cpu-qualification",
                "run_id": "cpu-qualification:batch",
                "current_generation": 5,
                "target_generation": 20,
                "updated_at": "2026-08-02T14:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    report = erm.collect_report(
        config,
        runner=_gpu_runner(gpu_stdout="GTX 1660 Ti, 0, 256, 5888, 6144, 45"),
        sleep=lambda _: None,
        now=dt.datetime(2026, 8, 2, 14, 0, 30, tzinfo=dt.timezone.utc),
    )
    assert report["recommendation"]["value"] == "CAUTION"
    assert report["blocker_classification"] == ["CPU_CAUTION"]
    assert "No GPU blocker" in report["recommendation"]["reason"]


def test_unattributed_compute_process_is_not_reported_as_idle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report: dict[str, Any] = {
        "gpu": {
            "available": True,
            "gpus": [{"vram_free_bytes": 4 * 1024**3, "vram_total_bytes": 6 * 1024**3}],
            "processes": [{"pid": 4242, "process_name": "python", "vram_used_bytes": 2 * 1024**3}],
        },
        "memory": {"ram_available_bytes": 8 * 1024**3},
        "storage": {"roots": []},
        "running_eiel_workloads": [],
        "slot_heartbeats": [],
    }

    blockers = erm.build_blockers(report, config)

    assert erm.gpu_occupancy_state(report) == "CONTENDED"
    assert any(row["kind"] == "unattributed_gpu_process" for row in blockers)
    assert "UNKNOWN" in erm.blocker_classification(blockers)


def test_ram_vram_storage_blocker_classes_are_reported(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        min_safe_storage_bytes=erm.DEFAULT_MIN_SAFE_STORAGE_BYTES,
    )
    report: dict[str, Any] = {
        "gpu": {
            "available": True,
            "gpus": [{"vram_free_bytes": 1 * 1024**3}],
            "processes": [],
        },
        "memory": {"ram_available_bytes": 1 * 1024**3, "swap_used_bytes": 1},
        "storage": {
            "roots": [
                {
                    "available": True,
                    "path": str(config.runtime_root),
                    "free_bytes": 1 * 1024**3,
                }
            ]
        },
        "running_eiel_workloads": [],
        "slot_heartbeats": [],
    }
    blockers = erm.build_blockers(report, config)
    assert erm.blocker_classification(blockers) == [
        "RAM_CAUTION",
        "VRAM_CAUTION",
        "STORAGE_CAUTION",
    ]


def test_next_check_is_fixed_poll_interval_not_eta(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (config.live_slots_root / "slot.json").write_text(
        json.dumps(
            {
                "pid": 123,
                "status": "running",
                "device": "gpu",
                "campaign_id": "gpu-campaign",
                "run_id": "gpu-campaign:batch",
                "eta_seconds": 420,
                "updated_at": "2026-08-02T14:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    report = erm.collect_report(
        config,
        runner=_gpu_runner(gpu_stdout="GTX 1660 Ti, 0, 256, 5888, 6144, 45"),
        sleep=lambda _: None,
        now=dt.datetime(2026, 8, 2, 14, 0, 30, tzinfo=dt.timezone.utc),
        polling_interval_s=17,
    )
    assert report["blockers"][0]["eta_seconds"] == 420
    assert report["recommended_poll_interval_seconds"] == 17
    assert report["next_check_after_seconds"] == 17


def test_missing_heartbeat_eta_is_unknown_not_estimated(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (config.live_slots_root / "slot.json").write_text(
        json.dumps(
            {
                "status": "running",
                "device": "gpu",
                "campaign_id": "no-eta",
                "current_generation": 8,
                "target_generation": 10,
                "updated_at": "2026-08-02T14:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    report = erm.collect_report(
        config,
        runner=_gpu_runner(gpu_stdout="GTX 1660 Ti, 0, 256, 5888, 6144, 45"),
        sleep=lambda _: None,
        now=dt.datetime(2026, 8, 2, 14, 0, 30, tzinfo=dt.timezone.utc),
    )
    assert report["slot_heartbeats"][0]["eta_seconds"] == "UNKNOWN"
    assert report["blockers"][0]["eta_seconds"] == "UNKNOWN"


def test_stale_heartbeat_with_dead_owner_and_terminal_run_is_ignored(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run_root = config.runtime_root / "runs" / "failed-run"
    run_root.mkdir(parents=True)
    (run_root / "execution_state.json").write_text(
        json.dumps({"schema_version": "eiel_execution_state_v1", "status": "failed"}),
        encoding="utf-8",
    )
    (run_root / "execution_result.json").write_text(
        json.dumps({"schema_version": "eiel_execution_result_v2", "status": "failed"}),
        encoding="utf-8",
    )
    (config.live_slots_root / "slot.json").write_text(
        json.dumps(
            {
                "pid": 922497,
                "pid_start_ticks": 88989402,
                "status": "running",
                "device": "gpu",
                "campaign_id": "failed-run",
                "run_id": "failed-run:batch-0000",
                "run_root": str(run_root),
                "updated_at": "2026-08-02T14:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    report = erm.collect_report(
        config,
        runner=_gpu_runner(gpu_stdout="GTX 1660 Ti, 0, 256, 5888, 6144, 45"),
        sleep=lambda _: None,
        now=dt.datetime(2026, 8, 2, 14, 10, tzinfo=dt.timezone.utc),
    )

    assert report["slot_heartbeats"] == []
    assert not any(
        blocker.get("kind") == "stale_gpu_heartbeat"
        for blocker in report["blockers"]
    )


def test_interrupted_heartbeat_status_is_terminal_and_ignored(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    (config.live_slots_root / "slot.json").write_text(
        json.dumps(
            {
                "pid": 922497,
                "status": "interrupted",
                "stage": "interrupted",
                "device": "gpu",
                "campaign_id": "interrupted-run",
                "run_id": "interrupted-run:batch-0000",
                "updated_at": "2026-08-02T14:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    rows = erm.discover_slot_heartbeats(
        config,
        now=dt.datetime(2026, 8, 2, 14, 10, tzinfo=dt.timezone.utc),
    )

    assert rows == []


def test_terminal_state_without_matching_result_remains_fail_closed(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run_root = config.runtime_root / "runs" / "partial-terminal-run"
    run_root.mkdir(parents=True)
    (run_root / "execution_state.json").write_text(
        json.dumps({"schema_version": "eiel_execution_state_v1", "status": "failed"}),
        encoding="utf-8",
    )
    (config.live_slots_root / "slot.json").write_text(
        json.dumps(
            {
                "pid": 922497,
                "status": "running",
                "device": "gpu",
                "run_root": str(run_root),
                "updated_at": "2026-08-02T14:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    rows = erm.discover_slot_heartbeats(
        config,
        now=dt.datetime(2026, 8, 2, 14, 10, tzinfo=dt.timezone.utc),
    )

    assert len(rows) == 1
    assert rows[0]["stale"] is True


def test_terminal_run_does_not_hide_stale_heartbeat_with_live_owner(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    pid_dir = config.proc_root / "2000"
    pid_dir.mkdir()
    (pid_dir / "cmdline").write_bytes(
        b"python\0tools/scripts/run_registry_parallel.py\0--device\0cuda\0"
    )
    (pid_dir / "stat").write_text(
        "2000 (python) S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100\n",
        encoding="utf-8",
    )
    (pid_dir / "cwd").symlink_to(config.repo_root)
    run_root = config.runtime_root / "runs" / "finishing-run"
    run_root.mkdir(parents=True)
    (run_root / "execution_state.json").write_text(
        json.dumps(
            {"schema_version": "eiel_execution_state_v1", "status": "completed"}
        ),
        encoding="utf-8",
    )
    (config.live_slots_root / "slot.json").write_text(
        json.dumps(
            {
                "pid": 2000,
                "status": "running",
                "device": "gpu",
                "campaign_id": "finishing-run",
                "run_id": "finishing-run:batch-0000",
                "run_root": str(run_root),
                "updated_at": "2026-08-02T14:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    rows = erm.discover_slot_heartbeats(
        config,
        now=dt.datetime(2026, 8, 2, 14, 10, tzinfo=dt.timezone.utc),
    )

    assert len(rows) == 1
    assert rows[0]["stale"] is True


def test_terminal_state_outside_runtime_root_cannot_suppress_heartbeat(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    outside = tmp_path / "outside-run"
    outside.mkdir()
    (outside / "execution_state.json").write_text(
        json.dumps({"schema_version": "eiel_execution_state_v1", "status": "failed"}),
        encoding="utf-8",
    )
    (config.live_slots_root / "slot.json").write_text(
        json.dumps(
            {
                "pid": 922497,
                "status": "running",
                "device": "gpu",
                "campaign_id": "outside-run",
                "run_id": "outside-run:batch-0000",
                "run_root": str(outside),
                "updated_at": "2026-08-02T14:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    rows = erm.discover_slot_heartbeats(
        config,
        now=dt.datetime(2026, 8, 2, 14, 10, tzinfo=dt.timezone.utc),
    )

    assert len(rows) == 1
    assert rows[0]["stale"] is True


def test_compact_output_includes_operational_summary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = erm.collect_report(
        config,
        runner=_gpu_runner(gpu_stdout="GTX 1660 Ti, 0, 256, 5888, 6144, 45"),
        sleep=lambda _: None,
        now=dt.datetime(2026, 8, 2, 14, 0, 30, tzinfo=dt.timezone.utc),
    )
    compact = erm.render_compact(report)
    assert "14:00:30" in compact
    assert "SAFE" in compact
    assert "GPU Campaigns:\n0" in compact
    assert "ETA:\nUNKNOWN" in compact
    assert "Blockers:\nnone" in compact


def test_json_compatibility_fields_are_preserved(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = erm.collect_report(
        config,
        runner=_gpu_runner(gpu_stdout="GTX 1660 Ti, 0, 256, 5888, 6144, 45"),
        sleep=lambda _: None,
    )
    for old_key in (
        "cpu",
        "memory",
        "gpu",
        "storage",
        "running_eiel_workloads",
        "slot_heartbeats",
        "campaign_manifests",
        "ste",
        "recommendation",
    ):
        assert old_key in report
    for new_key in (
        "blockers",
        "confidence",
        "next_check_after_seconds",
        "blocker_classification",
        "recommended_poll_interval_seconds",
    ):
        assert new_key in report


def test_human_and_json_cli_output(monkeypatch, capsys) -> None:
    payload = {
        "schema_version": erm.SCHEMA_VERSION,
        "cpu": {
            "utilization_percent": 1.0,
            "load_average": {},
            "physical_cores": 1,
            "logical_threads": 2,
        },
        "gpu": {
            "available": True,
            "gpus": [
                {
                    "index": 0,
                    "model": "Test GPU",
                    "utilization_percent": 0,
                    "vram_used_bytes": 0,
                    "vram_free_bytes": 4 * 1024**3,
                    "vram_total_bytes": 4 * 1024**3,
                    "temperature_c": 40,
                }
            ],
            "processes": [],
        },
        "memory": {
            "ram_used_bytes": 1,
            "ram_available_bytes": 2,
            "swap_used_bytes": 0,
            "swap_available_bytes": 3,
        },
        "storage": {"roots": []},
        "running_eiel_workloads": [],
        "slot_heartbeats": [],
        "ste": {"available": False},
        "recommendation": {"value": "SAFE", "confidence": "HIGH", "reason": "test"},
        "confidence": "HIGH",
        "blockers": [],
        "blocker_classification": [],
        "recommended_poll_interval_seconds": "UNKNOWN",
        "next_check_after_seconds": "UNKNOWN",
    }
    monkeypatch.setattr(erm, "collect_report", lambda config, **_: payload)

    assert erm.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["recommendation"]["confidence"] == "HIGH"
    assert erm.main([]) == 0
    human = capsys.readouterr().out
    assert "EIEL Resource Monitor" in human
    assert "SAFE confidence=HIGH" in human


def test_compact_cli_output(monkeypatch, capsys) -> None:
    payload = {
        "schema_version": erm.SCHEMA_VERSION,
        "generated_at_utc": "2026-08-02T14:00:30Z",
        "gpu": {
            "available": True,
            "gpus": [
                {
                    "vram_used_bytes": 1 * 1024**3,
                    "vram_total_bytes": 6 * 1024**3,
                }
            ],
        },
        "running_eiel_workloads": [],
        "slot_heartbeats": [],
        "recommendation": {"value": "SAFE", "confidence": "HIGH", "reason": "test"},
        "blocker_classification": [],
        "recommended_poll_interval_seconds": "UNKNOWN",
        "next_check_after_seconds": "UNKNOWN",
    }
    monkeypatch.setattr(erm, "collect_report", lambda config, **_: payload)
    assert erm.main(["--compact"]) == 0
    assert "VRAM:\n1.0 / 6.0 GB" in capsys.readouterr().out


def test_watch_mode_polls_without_background_state(monkeypatch, capsys) -> None:
    calls = 0
    sleeps: list[float] = []

    def collect(config: erm.MonitorConfig, **_: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "schema_version": erm.SCHEMA_VERSION,
            "generated_at_utc": "2026-08-02T14:00:30Z",
            "gpu": {"available": True, "gpus": []},
            "running_eiel_workloads": [],
            "slot_heartbeats": [],
            "recommendation": {"value": "SAFE", "confidence": "HIGH", "reason": "test"},
            "blocker_classification": [],
            "recommended_poll_interval_seconds": "UNKNOWN",
            "next_check_after_seconds": "UNKNOWN",
        }

    assert (
        erm.main(
            ["--watch", "5", "--compact", "--max-iterations", "2"],
            collect_report_fn=collect,
            sleep_fn=sleeps.append,
        )
        == 0
    )
    assert calls == 2
    assert sleeps == [5.0]
    assert capsys.readouterr().out.count("SAFE") == 2


def test_wait_until_safe_polls_until_match(capsys) -> None:
    values = ["WAIT", "SAFE"]
    intervals: list[float] = []

    def collect(config: erm.MonitorConfig, **kwargs: object) -> dict[str, Any]:
        value = values.pop(0)
        return {
            "schema_version": erm.SCHEMA_VERSION,
            "generated_at_utc": "2026-08-02T14:00:30Z",
            "gpu": {"available": True, "gpus": []},
            "running_eiel_workloads": [],
            "slot_heartbeats": [],
            "recommendation": {"value": value, "confidence": "HIGH", "reason": value},
            "blocker_classification": [],
            "recommended_poll_interval_seconds": kwargs.get(
                "polling_interval_s", "UNKNOWN"
            ),
            "next_check_after_seconds": kwargs.get("polling_interval_s", "UNKNOWN"),
        }

    assert (
        erm.main(
            ["--wait-until", "SAFE", "--interval", "3", "--compact"],
            collect_report_fn=collect,
            sleep_fn=intervals.append,
        )
        == 0
    )
    assert intervals == [3.0]
    output = capsys.readouterr().out
    assert "\nSAFE\n\nGPU" in output
    assert "WAIT" not in output
