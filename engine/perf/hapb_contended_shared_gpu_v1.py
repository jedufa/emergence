"""Cooperative GTX 1660 Ti sharing between HAPB QualPrep and Tensor World C6.

This is an admission/stop policy only. It never resets the GPU, never kills
C6, and never starts a third CUDA workload. Formal C6 qualification remains
exclusive; this policy applies only to C6 QualPrep/rehearsal coexistence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

SCHEMA_VERSION_V1: Final[str] = "eiel_hapb_contended_shared_gpu_v1"
RUN_LABEL_V1: Final[str] = "CONTENDED_SHARED_GPU"
DEVICE_NAME_V1: Final[str] = "NVIDIA GeForce GTX 1660 Ti"

# Cooperative limits (GiB). Desktop now uses Intel graphics; no 10% VRAM reserve.
HAPB_STEADY_WORKING_SET_GIB_V1: Final[float] = 0.75
HAPB_ABSOLUTE_CEILING_GIB_V1: Final[float] = 1.0
COMBINED_GPU_MEMORY_CEILING_GIB_V1: Final[float] = 4.5
MIN_FREE_AFTER_ADMIT_GIB_V1: Final[float] = 1.1
STOP_FREE_GIB_V1: Final[float] = 0.5
C6_CHECKPOINT_TRANSIENT_GIB_V1: Final[float] = 0.9

# Historical full-H4 HAPB working set observed on this device.
HAPB_FULL_H4_WORKING_SET_GIB_V1: Final[float] = 1.66
EXCLUSIVE_CUDA_MIN_FREE_MIB_V1: Final[int] = 4000
EXCLUSIVE_CUDA_LABEL_V1: Final[str] = "EXCLUSIVE_CUDA_REQUIRED"


@dataclass(frozen=True, slots=True)
class GpuSnapshotV1:
    total_mib: int
    used_mib: int
    free_mib: int
    apps: tuple[tuple[int, str, int], ...]

    @property
    def total_gib(self) -> float:
        return self.total_mib / 1024.0

    @property
    def used_gib(self) -> float:
        return self.used_mib / 1024.0

    @property
    def free_gib(self) -> float:
        return self.free_mib / 1024.0


def c6_present_in_snapshot_v1(snapshot: GpuSnapshotV1) -> bool:
    needles = (
        "run_tensor_world_c6",
        "tensor_world_c6",
        "tensor-world-c6",
        "tensor_world/native",
    )
    for _pid, name, _mem in snapshot.apps:
        lowered = name.lower()
        if any(needle in lowered for needle in needles):
            return True
    return False


def admit_hapb_shared_launch_v1(
    *,
    snapshot: GpuSnapshotV1,
    hapb_working_set_gib: float,
    c6_checkpoint_or_restore_imminent: bool = False,
) -> dict[str, Any]:
    """Return a fail-closed admission receipt. Never mutates GPU state."""

    c6_present = c6_present_in_snapshot_v1(snapshot)
    label = RUN_LABEL_V1 if c6_present else "EXCLUSIVE_OR_IDLE"
    refuse_reasons: list[str] = []
    if hapb_working_set_gib > HAPB_ABSOLUTE_CEILING_GIB_V1 and c6_present:
        refuse_reasons.append(
            "hapb_working_set_exceeds_1.0_gib_ceiling_while_c6_present"
        )
    if c6_present and hapb_working_set_gib > HAPB_STEADY_WORKING_SET_GIB_V1:
        # Above steady target but at/under ceiling is allowed only if the
        # caller explicitly stays <= 1.0 GiB; still record the breach.
        if hapb_working_set_gib > HAPB_ABSOLUTE_CEILING_GIB_V1:
            refuse_reasons.append("hapb_working_set_exceeds_steady_and_ceiling")
    projected_used = snapshot.used_gib + hapb_working_set_gib
    if c6_checkpoint_or_restore_imminent:
        projected_used += C6_CHECKPOINT_TRANSIENT_GIB_V1
        refuse_reasons.append("c6_checkpoint_or_restore_imminent_hapb_must_release")
    projected_free = snapshot.total_gib - projected_used
    if projected_used > COMBINED_GPU_MEMORY_CEILING_GIB_V1:
        refuse_reasons.append("combined_gpu_memory_would_exceed_4.5_gib")
    if projected_free < MIN_FREE_AFTER_ADMIT_GIB_V1:
        refuse_reasons.append("projected_free_below_1.1_gib")
    if snapshot.free_gib < STOP_FREE_GIB_V1:
        refuse_reasons.append("current_free_below_0.5_gib_stop")
    admitted = not refuse_reasons
    return {
        "schema_version": SCHEMA_VERSION_V1,
        "run_label": label,
        "c6_present": c6_present,
        "formal_c6_qualification_exclusive": True,
        "sharing_applies_to": "c6_qualprep_or_rehearsal_only",
        "hapb_working_set_gib": hapb_working_set_gib,
        "hapb_steady_limit_gib": HAPB_STEADY_WORKING_SET_GIB_V1,
        "hapb_ceiling_gib": HAPB_ABSOLUTE_CEILING_GIB_V1,
        "projected_used_gib": projected_used,
        "projected_free_gib": projected_free,
        "admitted": admitted,
        "refuse_reasons": refuse_reasons,
        "timings_vram_non_gating": True,
        "must_release_before_c6_checkpoint_restore_resume": True,
        "never_reset_gpu": True,
        "never_kill_c6": True,
        "never_start_third_cuda_workload": True,
        "separate_evidence_roots_required": True,
    }


def snapshot_from_resource_monitor_gpu_v1(gpu: Mapping[str, Any]) -> GpuSnapshotV1:
    """Project an ERM/resource-monitor GPU payload onto the HAPB snapshot."""

    rows = tuple(gpu.get("gpus") or ())
    first = rows[0] if rows and isinstance(rows[0], Mapping) else {}

    def _mib(key: str) -> int:
        raw = first.get(key) if isinstance(first, Mapping) else None
        if raw is None:
            return 0
        return int(int(raw) // (1024 * 1024))

    apps: list[tuple[int, str, int]] = []
    for proc in gpu.get("processes") or ():
        if not isinstance(proc, Mapping):
            continue
        pid = proc.get("pid")
        if pid in {None, "", "[N/A]"}:
            continue
        try:
            used_raw = proc.get("vram_used_bytes")
            used_mib = 0 if used_raw is None else int(int(used_raw) // (1024 * 1024))
            apps.append((int(pid), str(proc.get("process_name") or ""), used_mib))
        except (TypeError, ValueError):
            continue
    return GpuSnapshotV1(
        total_mib=_mib("vram_total_bytes"),
        used_mib=_mib("vram_used_bytes"),
        free_mib=_mib("vram_free_bytes"),
        apps=tuple(apps),
    )


def admit_hapb_exclusive_cuda_launch_v1(
    *,
    snapshot: GpuSnapshotV1,
    cuda_available: bool,
    min_free_mib: int = EXCLUSIVE_CUDA_MIN_FREE_MIB_V1,
    ignore_pids: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Fail-closed exclusive CUDA admission. Wait is never PERF-B."""

    refuse_reasons: list[str] = []
    if not cuda_available:
        refuse_reasons.append("cuda_unavailable")
    ignored = set(ignore_pids)
    foreign = tuple(app for app in snapshot.apps if app[0] not in ignored)
    if foreign:
        refuse_reasons.append("other_compute_apps_present")
    c6_present = c6_present_in_snapshot_v1(snapshot)
    if c6_present:
        refuse_reasons.append("c6_present_exclusive_required")
    if snapshot.free_mib < int(min_free_mib):
        refuse_reasons.append("free_vram_below_exclusive_minimum")
    return {
        "schema_version": SCHEMA_VERSION_V1,
        "run_label": EXCLUSIVE_CUDA_LABEL_V1,
        "admitted": not refuse_reasons,
        "refuse_reasons": refuse_reasons,
        "wait_resource_not_perf_b": True,
        "cuda_available": bool(cuda_available),
        "c6_present": c6_present,
        "compute_app_count": len(foreign),
        "foreign_compute_apps": [
            {"pid": pid, "name": name, "used_mib": used_mib}
            for pid, name, used_mib in foreign
        ],
        "free_mib": snapshot.free_mib,
        "min_free_mib": int(min_free_mib),
        "never_kill_c6": True,
        "never_reset_gpu": True,
        "never_start_third_cuda_workload": True,
    }


def memory_sample_from_snapshot_v1(snapshot: GpuSnapshotV1, *, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "device_total_bytes": int(snapshot.total_mib) * 1024 * 1024,
        "used_bytes": int(snapshot.used_mib) * 1024 * 1024,
        "free_bytes": int(snapshot.free_mib) * 1024 * 1024,
    }


def hapb_memory_admission_from_samples_v1(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Gate-8 memory row from exclusive-run samples. Missing samples stay unavailable."""

    if not samples:
        return {
            "status": "UNAVAILABLE",
            "physical_margin_bytes": None,
            "rule_pass": False,
        }
    try:
        totals = [int(sample["device_total_bytes"]) for sample in samples]
        frees = [int(sample["free_bytes"]) for sample in samples]
        useds = [int(sample["used_bytes"]) for sample in samples]
    except (KeyError, TypeError, ValueError):
        return {
            "status": "UNAVAILABLE",
            "physical_margin_bytes": None,
            "rule_pass": False,
        }
    if not totals or min(totals) <= 0:
        return {
            "status": "UNAVAILABLE",
            "physical_margin_bytes": None,
            "rule_pass": False,
        }
    total = totals[0]
    minimum_free = min(frees)
    peak_used = max(useds)
    reserve = math.ceil(total * 0.10)
    margin = minimum_free - reserve
    return {
        "status": "MEASURED",
        "device_total_bytes": total,
        "minimum_free_at_peak_bytes": minimum_free,
        "peak_used_bytes": peak_used,
        "ten_percent_reserve_bytes": reserve,
        "physical_margin_bytes": margin,
        "rule_pass": margin >= 0,
        "sample_count": len(samples),
    }


def host_sample_v1(*, label: str) -> dict[str, Any]:
    import resource
    from pathlib import Path

    rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    available = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError):
        available = None
    return {
        "label": label,
        "rss_bytes": rss_kb * 1024,
        "mem_available_bytes": available,
    }


def hapb_host_admission_from_samples_v1(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not samples:
        return {
            "status": "UNAVAILABLE",
            "physical_margin_bytes": None,
            "rule_pass": False,
        }
    try:
        rss = [int(sample["rss_bytes"]) for sample in samples]
        available = [
            int(sample["mem_available_bytes"])
            for sample in samples
            if sample.get("mem_available_bytes") is not None
        ]
    except (KeyError, TypeError, ValueError):
        return {
            "status": "UNAVAILABLE",
            "physical_margin_bytes": None,
            "rule_pass": False,
        }
    if not rss:
        return {
            "status": "UNAVAILABLE",
            "physical_margin_bytes": None,
            "rule_pass": False,
        }
    peak = max(rss)
    min_available = min(available) if available else None
    return {
        "status": "MEASURED",
        "peak_rss_bytes": peak,
        "minimum_available_bytes": min_available,
        "physical_margin_bytes": min_available,
        "rule_pass": min_available is not None and min_available > 0,
        "sample_count": len(samples),
    }


def stop_hapb_if_resource_unsafe_v1(snapshot: GpuSnapshotV1) -> dict[str, Any]:
    """Live stop check. On trip, caller must preserve evidence and not retry."""

    stop = snapshot.free_gib < STOP_FREE_GIB_V1
    return {
        "schema_version": SCHEMA_VERSION_V1,
        "stop": stop,
        "reason": "free_below_0.5_gib" if stop else None,
        "free_gib": snapshot.free_gib,
        "preserve_evidence": stop,
        "blind_retry_forbidden": True,
    }


def query_nvidia_smi_snapshot_v1() -> GpuSnapshotV1:
    import subprocess

    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    total, used, free = (int(float(part.strip())) for part in gpu.split(","))
    apps_raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    apps: list[tuple[int, str, int]] = []
    if apps_raw:
        for line in apps_raw.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3 or parts[0] in {"", "[N/A]"}:
                continue
            try:
                apps.append((int(parts[0]), parts[1], int(float(parts[2]))))
            except ValueError:
                continue
    return GpuSnapshotV1(total, used, free, tuple(apps))
