from __future__ import annotations

from engine.perf.hapb_contended_shared_gpu_v1 import (
    C6_CHECKPOINT_TRANSIENT_GIB_V1,
    EXCLUSIVE_CUDA_LABEL_V1,
    EXCLUSIVE_CUDA_MIN_FREE_MIB_V1,
    HAPB_ABSOLUTE_CEILING_GIB_V1,
    HAPB_FULL_H4_WORKING_SET_GIB_V1,
    HAPB_STEADY_WORKING_SET_GIB_V1,
    RUN_LABEL_V1,
    GpuSnapshotV1,
    admit_hapb_exclusive_cuda_launch_v1,
    hapb_host_admission_from_samples_v1,
    hapb_memory_admission_from_samples_v1,
    host_sample_v1,
    memory_sample_from_snapshot_v1,
    admit_hapb_shared_launch_v1,
    c6_present_in_snapshot_v1,
    snapshot_from_resource_monitor_gpu_v1,
    stop_hapb_if_resource_unsafe_v1,
)


def _snap(*, used: int, free: int, c6: bool = True) -> GpuSnapshotV1:
    apps = ((14110, "/x/run_tensor_world_c6_qualification.py", used),) if c6 else ()
    return GpuSnapshotV1(total_mib=6144, used_mib=used, free_mib=free, apps=apps)


def test_c6_detection_and_full_h4_refused_while_c6_present() -> None:
    snap = _snap(used=2182, free=3551, c6=True)
    assert c6_present_in_snapshot_v1(snap) is True
    receipt = admit_hapb_shared_launch_v1(
        snapshot=snap,
        hapb_working_set_gib=HAPB_FULL_H4_WORKING_SET_GIB_V1,
    )
    assert receipt["admitted"] is False
    assert receipt["run_label"] == RUN_LABEL_V1
    assert "hapb_working_set_exceeds_1.0_gib_ceiling_while_c6_present" in receipt[
        "refuse_reasons"
    ]


def test_steady_hapb_075_admitted_beside_c6_qualprep() -> None:
    snap = _snap(used=2182, free=3551, c6=True)
    receipt = admit_hapb_shared_launch_v1(
        snapshot=snap,
        hapb_working_set_gib=HAPB_STEADY_WORKING_SET_GIB_V1,
    )
    assert receipt["admitted"] is True
    assert receipt["projected_free_gib"] >= 1.1
    assert receipt["never_kill_c6"] is True


def test_refuse_when_c6_checkpoint_imminent() -> None:
    snap = _snap(used=2182, free=3551, c6=True)
    receipt = admit_hapb_shared_launch_v1(
        snapshot=snap,
        hapb_working_set_gib=HAPB_STEADY_WORKING_SET_GIB_V1,
        c6_checkpoint_or_restore_imminent=True,
    )
    assert receipt["admitted"] is False
    assert "c6_checkpoint_or_restore_imminent_hapb_must_release" in receipt[
        "refuse_reasons"
    ]
    assert C6_CHECKPOINT_TRANSIENT_GIB_V1 == 0.9


def test_stop_below_half_gib_free() -> None:
    snap = _snap(used=5800, free=400, c6=True)
    trip = stop_hapb_if_resource_unsafe_v1(snap)
    assert trip["stop"] is True
    assert trip["blind_retry_forbidden"] is True
    assert admit_hapb_shared_launch_v1(
        snapshot=snap,
        hapb_working_set_gib=HAPB_STEADY_WORKING_SET_GIB_V1,
    )["admitted"] is False


def test_exclusive_idle_allows_full_h4() -> None:
    snap = _snap(used=17, free=5733, c6=False)
    receipt = admit_hapb_shared_launch_v1(
        snapshot=snap,
        hapb_working_set_gib=HAPB_FULL_H4_WORKING_SET_GIB_V1,
    )
    assert receipt["admitted"] is True
    assert receipt["run_label"] == "EXCLUSIVE_OR_IDLE"
    assert HAPB_ABSOLUTE_CEILING_GIB_V1 == 1.0


def test_exclusive_cuda_refuses_busy_or_unavailable_gpu() -> None:
    busy = admit_hapb_exclusive_cuda_launch_v1(
        snapshot=_snap(used=4360, free=1372, c6=True),
        cuda_available=True,
    )
    assert busy["admitted"] is False
    assert busy["run_label"] == EXCLUSIVE_CUDA_LABEL_V1
    assert busy["wait_resource_not_perf_b"] is True
    assert "c6_present_exclusive_required" in busy["refuse_reasons"]
    assert "other_compute_apps_present" in busy["refuse_reasons"]
    assert "free_vram_below_exclusive_minimum" in busy["refuse_reasons"]
    assert admit_hapb_exclusive_cuda_launch_v1(
        snapshot=_snap(used=17, free=5733, c6=False),
        cuda_available=False,
    )["refuse_reasons"] == ["cuda_unavailable"]


def test_exclusive_cuda_admits_idle_device_with_headroom() -> None:
    receipt = admit_hapb_exclusive_cuda_launch_v1(
        snapshot=_snap(used=17, free=5733, c6=False),
        cuda_available=True,
    )
    assert receipt["admitted"] is True
    assert receipt["compute_app_count"] == 0
    assert receipt["free_mib"] >= EXCLUSIVE_CUDA_MIN_FREE_MIB_V1


def test_host_admission_from_samples_records_rss_margin() -> None:
    live = host_sample_v1(label="now")
    assert live["rss_bytes"] > 0
    admission = hapb_host_admission_from_samples_v1((live,))
    assert admission["status"] == "MEASURED"
    assert admission["peak_rss_bytes"] == live["rss_bytes"]
    assert admission["physical_margin_bytes"] == live["mem_available_bytes"]
    assert hapb_host_admission_from_samples_v1(())["rule_pass"] is False


def test_memory_admission_from_samples_applies_ten_percent_reserve() -> None:
    idle = memory_sample_from_snapshot_v1(_snap(used=17, free=5733, c6=False), label="idle")
    peak = memory_sample_from_snapshot_v1(_snap(used=4360, free=1372, c6=False), label="peak")
    admission = hapb_memory_admission_from_samples_v1((idle, peak))
    assert admission["status"] == "MEASURED"
    assert admission["minimum_free_at_peak_bytes"] == 1372 * 1024 * 1024
    assert admission["peak_used_bytes"] == 4360 * 1024 * 1024
    assert admission["rule_pass"] is True
    assert hapb_memory_admission_from_samples_v1(())["rule_pass"] is False


def test_snapshot_from_resource_monitor_gpu_payload() -> None:
    snapshot = snapshot_from_resource_monitor_gpu_v1(
        {
            "available": True,
            "gpus": [
                {
                    "vram_total_bytes": 6144 * 1024 * 1024,
                    "vram_used_bytes": 4360 * 1024 * 1024,
                    "vram_free_bytes": 1372 * 1024 * 1024,
                }
            ],
            "processes": [
                {
                    "pid": 801704,
                    "process_name": "/x/run_tensor_world_c6_qualification.py",
                    "vram_used_bytes": 4360 * 1024 * 1024,
                }
            ],
        }
    )
    assert snapshot.free_mib == 1372
    assert snapshot.apps[0][0] == 801704
    assert c6_present_in_snapshot_v1(snapshot) is True
