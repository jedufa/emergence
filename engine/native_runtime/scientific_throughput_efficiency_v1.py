"""Backend-independent Scientific Throughput Efficiency reporting.

This module derives engineering telemetry from already-existing execution
receipts and host counters. It must not dispatch work, synchronize devices, or
alter simulation state.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

SCHEMA_VERSION: Final[str] = "eiel_scientific_throughput_efficiency_v1"
ENGINEERING_MODEL_VERSION: Final[str] = "eiel_ste_engineering_model_v2"
DISCOVERY_EFFICIENCY_SCHEMA_VERSION: Final[str] = (
    "eiel_scientific_discovery_efficiency_v1"
)
FEATURE_FLAG: Final[str] = "EIEL_STE_TELEMETRY"
UNKNOWN: Final[str] = "unknown"
TREND_METRIC_KEYS: Final[tuple[str, ...]] = (
    "scientific_worlds_completed_per_hour",
    "world_generations_per_hour",
    "primary_world_generations_per_hour",
    "qualification_generations_per_hour",
    "parameter_evaluations_per_hour",
    "qualification_cells_per_hour",
    "evidence_packages_per_hour",
    "evidence_bytes_per_hour",
)
LEGACY_THROUGHPUT_ALIASES: Final[dict[str, str]] = {
    "worlds_completed_per_second": "scientific_throughput.worlds_completed_per_second",
    "ticks_per_second": "scientific_throughput.world_ticks_per_second",
    "generations_per_hour": "scientific_throughput.world_generations_per_hour",
    "checkpoint_branches_per_hour": "qualification_throughput.checkpoint_branches_per_hour",
    "worlds_per_hour": "scientific_throughput.worlds_completed_per_hour",
}
EXCLUDED_DISCOVERY_SCORE_METRICS: Final[tuple[str, ...]] = (
    "intelligence",
    "emergence",
    "cognition",
    "novelty",
    "scientific_importance",
)
STE_REPORT_JSON_SCHEMA: Final[dict[str, Any]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "eiel_scientific_throughput_efficiency_report_v2",
    "type": "object",
    "required": [
        "schema_version",
        "metric_name",
        "engineering_model",
        "resource_efficiency",
        "scientific_efficiency",
        "scientific_discovery_efficiency",
        "claims_boundary",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "metric_name": {"const": "Scientific Throughput Efficiency"},
        "backend": {"type": "string"},
        "engineering_model": {"type": "object"},
        "resource_efficiency": {"type": "object"},
        "scientific_efficiency": {"type": "object"},
        "scientific_discovery_efficiency": {"type": "object"},
        "primary_science_speed": {"type": "object"},
        "scientific_throughput": {"type": "object"},
        "qualification_throughput": {"type": "object"},
        "evidence_throughput": {"type": "object"},
        "throughput_taxonomy": {"type": "object"},
        "legacy_metric_aliases": {"type": "object"},
        "science_throughput": {"type": "object"},
        "memory_utilization": {"type": "object"},
        "storage_utilization": {"type": "object"},
        "scheduler_efficiency": {"type": "object"},
        "execution_efficiency": {"type": "object"},
        "backend_specific": {"type": "object"},
        "resource_utilization_time_series": {"type": "object"},
        "dashboard_payload": {"type": "object"},
    },
    "additionalProperties": True,
}
_KNOWN_CUDA_MODELS: Final[tuple[tuple[str, dict[str, Any]], ...]] = (
    (
        "gtx 1660 ti",
        {
            "sm_count": 24,
            "cuda_cores": 1536,
            "warp_size": 32,
            "max_resident_warps_per_sm": 32,
            "max_resident_blocks_per_sm": 16,
            "max_threads_per_sm": 1024,
            "shared_memory_per_sm_bytes": 64 * 1024,
            "registers_per_sm": 65536,
            "memory_bandwidth_gbps": 288.0,
            "clock_frequency_mhz": {"base_sm_mhz": 1500, "boost_sm_mhz": 1770},
        },
    ),
)


def scientific_throughput_efficiency_enabled(
    env: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if env is None else env
    return str(source.get(FEATURE_FLAG, "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "disabled",
    }


def scientific_throughput_efficiency_json_schema() -> dict[str, Any]:
    return dict(STE_REPORT_JSON_SCHEMA)


def build_resource_utilization_time_series_sample(
    *,
    timestamp_seconds: float | None = None,
    cpu_utilization_percent: float | None = None,
    rss_bytes: int | None = None,
    ram_used_bytes: int | None = None,
    ram_total_bytes: int | None = None,
    gpu_utilization_percent: float | None = None,
    vram_used_bytes: int | None = None,
    vram_total_bytes: int | None = None,
    storage_bytes_written: int | None = None,
    source: str = "explicit_input",
) -> dict[str, Any]:
    return {
        "schema_version": "eiel_ste_resource_utilization_time_series_sample_v1",
        "timestamp_seconds": _round(timestamp_seconds, 9),
        "cpu_utilization_percent": _round(cpu_utilization_percent),
        "rss_bytes": _int(rss_bytes),
        "ram_used_bytes": _int(ram_used_bytes),
        "ram_total_bytes": _int(ram_total_bytes),
        "ram_utilization": _round(_ratio(ram_used_bytes, ram_total_bytes)),
        "gpu_utilization_percent": _round(gpu_utilization_percent),
        "vram_used_bytes": _int(vram_used_bytes),
        "vram_total_bytes": _int(vram_total_bytes),
        "vram_utilization": _round(_ratio(vram_used_bytes, vram_total_bytes)),
        "storage_bytes_written": _int(storage_bytes_written),
        "source": str(source),
        "claims_boundary": "optional telemetry sample only; not scheduler input by itself",
    }


def _resource_time_series(
    samples: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    materialized = [
        dict(sample) for sample in samples or () if isinstance(sample, Mapping)
    ]
    return {
        "schema_version": "eiel_ste_resource_utilization_time_series_v1",
        "enabled": bool(materialized),
        "sample_count": len(materialized),
        "samples": materialized,
        "trend_metric_keys": [
            "cpu_utilization_percent",
            "rss_bytes",
            "ram_utilization",
            "gpu_utilization_percent",
            "vram_utilization",
            "storage_bytes_written",
        ],
        "scheduler_use": "future_observational_input_only",
        "plotting_implemented": False,
    }


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace("%", "")
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _int(value: Any) -> int | None:
    number = _float(value)
    if number is None:
        return None
    return int(number)


def _ratio(
    numerator: float | int | None, denominator: float | int | None
) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _mean(values: Iterable[float | int | None]) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    if not materialized:
        return None
    return sum(materialized) / float(len(materialized))


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _per_second(
    count: int | float | None, elapsed_seconds: float | None
) -> float | None:
    return _round(_ratio(count, elapsed_seconds))


def _per_hour(count: int | float | None, elapsed_seconds: float | None) -> float | None:
    if count is None or elapsed_seconds in (None, 0):
        return None
    return _round(float(count) * 3600.0 / float(elapsed_seconds))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _proc_status_kib(path: Path = Path("/proc/self/status")) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in _read_text(path).splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        match = re.search(r"(-?\d+)", raw)
        if match:
            out[key] = int(match.group(1))
    return out


def _proc_meminfo_bytes(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in _read_text(path).splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        match = re.search(r"(-?\d+)", raw)
        if match:
            out[key] = int(match.group(1)) * 1024
    return out


def _smaps_rollup_kib(path: Path = Path("/proc/self/smaps_rollup")) -> dict[str, int]:
    keys = {"Rss", "Pss", "Anonymous", "Shared_Clean", "Private_Clean"}
    out: dict[str, int] = {}
    for line in _read_text(path).splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        if key not in keys:
            continue
        match = re.search(r"(-?\d+)", raw)
        if match:
            out[key] = int(match.group(1))
    return out


def _psi(path: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for line in _read_text(path).splitlines():
        parts = line.split()
        if not parts:
            continue
        row: dict[str, float] = {}
        for item in parts[1:]:
            name, sep, value = item.partition("=")
            if sep:
                parsed = _float(value)
                if parsed is not None:
                    row[name] = parsed
        rows[parts[0]] = row
    return rows


def detect_cpu_hardware_descriptor() -> dict[str, Any]:
    cpuinfo = _read_text(Path("/proc/cpuinfo"))
    model = platform.processor()
    physical_pairs: set[tuple[str, str]] = set()
    current: dict[str, str] = {}
    for line in cpuinfo.splitlines() + [""]:
        if not line.strip():
            if "physical id" in current and "core id" in current:
                physical_pairs.add((current["physical id"], current["core id"]))
            current = {}
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        current[key.strip()] = value.strip()
        if key.strip() == "model name" and not model:
            model = value.strip()

    meminfo = _proc_meminfo_bytes()
    return {
        "schema_version": "eiel_cpu_hardware_descriptor_v1",
        "backend": "cpu",
        "detector_status": "detected",
        "cpu_model": model or UNKNOWN,
        "physical_core_count": len(physical_pairs) or None,
        "logical_thread_count": os.cpu_count(),
        "cache_sizes": {},
        "ram_bytes": meminfo.get("MemTotal"),
        "platform": platform.platform(),
    }


def sample_cpu_process_utilization() -> dict[str, Any]:
    status = _proc_status_kib()
    smaps = _smaps_rollup_kib()
    usage = None
    try:
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF)
        usage = {
            "minor_page_faults": int(raw.ru_minflt),
            "major_page_faults": int(raw.ru_majflt),
            "swap_count": int(raw.ru_nswap),
            "peak_rss_bytes": int(raw.ru_maxrss) * 1024,
        }
    except Exception:
        usage = None
    return {
        "schema_version": "eiel_cpu_process_utilization_sample_v1",
        "rss_bytes": status.get("VmRSS", 0) * 1024 if "VmRSS" in status else None,
        "peak_rss_bytes": status.get("VmHWM", 0) * 1024 if "VmHWM" in status else None,
        "pss_bytes": smaps.get("Pss", 0) * 1024 if "Pss" in smaps else None,
        "anonymous_bytes": (
            smaps.get("Anonymous", 0) * 1024 if "Anonymous" in smaps else None
        ),
        "file_backed_cache_bytes": (
            (smaps.get("Shared_Clean", 0) + smaps.get("Private_Clean", 0)) * 1024
            if smaps
            else None
        ),
        "resource_usage": usage,
        "cpu_pressure": _psi(Path("/proc/pressure/cpu")),
        "memory_pressure": _psi(Path("/proc/pressure/memory")),
    }


def detect_cuda_hardware_descriptor() -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "schema_version": "eiel_cuda_hardware_descriptor_v1",
        "backend": "cuda",
        "detector_status": "unavailable",
        "gpu_model": UNKNOWN,
        "compute_capability": None,
        "sm_count": None,
        "cuda_cores": None,
        "warp_size": None,
        "max_resident_warps_per_sm": None,
        "max_resident_blocks_per_sm": None,
        "max_threads_per_sm": None,
        "shared_memory_per_sm_bytes": None,
        "registers_per_sm": None,
        "total_vram_bytes": None,
        "memory_bandwidth_gbps": None,
        "clock_frequency_mhz": None,
        "sources": [],
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            major = int(getattr(props, "major", 0))
            minor = int(getattr(props, "minor", 0))
            sm_count = int(getattr(props, "multi_processor_count", 0) or 0)
            warp_size = int(getattr(props, "warp_size", 0) or 0)
            max_threads_sm = int(
                getattr(props, "max_threads_per_multi_processor", 0) or 0
            )
            descriptor.update(
                {
                    "detector_status": "detected",
                    "gpu_model": str(getattr(props, "name", UNKNOWN)),
                    "compute_capability": [major, minor],
                    "sm_count": sm_count or None,
                    "cuda_cores": _cuda_cores_for_capability(major, minor, sm_count),
                    "warp_size": warp_size or None,
                    "max_resident_warps_per_sm": (
                        max_threads_sm // warp_size
                        if max_threads_sm and warp_size
                        else None
                    ),
                    "max_resident_blocks_per_sm": int(
                        getattr(props, "max_blocks_per_multi_processor", 0) or 0
                    )
                    or None,
                    "max_threads_per_sm": max_threads_sm or None,
                    "shared_memory_per_sm_bytes": int(
                        getattr(props, "shared_memory_per_multiprocessor", 0) or 0
                    )
                    or None,
                    "registers_per_sm": int(
                        getattr(props, "regs_per_multiprocessor", 0) or 0
                    )
                    or None,
                    "total_vram_bytes": int(getattr(props, "total_memory", 0) or 0)
                    or None,
                    "sources": ["torch.cuda"],
                }
            )
    except Exception as exc:
        descriptor["torch_error"] = f"{type(exc).__name__}: {exc}"

    smi = _nvidia_smi_descriptor()
    if smi:
        descriptor["sources"] = sorted(
            set(descriptor.get("sources", [])) | {"nvidia-smi"}
        )
        descriptor["detector_status"] = (
            "detected" if descriptor["detector_status"] == "detected" else "partial"
        )
        descriptor["gpu_model"] = (
            descriptor["gpu_model"]
            if descriptor["gpu_model"] != UNKNOWN
            else smi.get("gpu_model", UNKNOWN)
        )
        descriptor["compute_capability"] = descriptor["compute_capability"] or smi.get(
            "compute_capability"
        )
        descriptor["total_vram_bytes"] = descriptor["total_vram_bytes"] or smi.get(
            "total_vram_bytes"
        )
        descriptor["clock_frequency_mhz"] = smi.get("clock_frequency_mhz")
        descriptor["gpu_utilization_percent"] = smi.get("gpu_utilization_percent")
    return _enrich_known_cuda_model(descriptor)


def _enrich_known_cuda_model(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    enriched = dict(descriptor)
    model = str(enriched.get("gpu_model", "")).lower()
    for pattern, defaults in _KNOWN_CUDA_MODELS:
        if pattern not in model:
            continue
        for key, value in defaults.items():
            if enriched.get(key) is None:
                enriched[key] = value
        sources = list(enriched.get("sources") or [])
        if "model_table" not in sources:
            sources.append("model_table")
        enriched["sources"] = sorted(sources)
        return enriched
    return enriched


def _nvidia_smi_descriptor() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,clocks.sm,clocks.max.sm,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return {}
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
    if len(fields) < 6:
        return {}
    capability = None
    if fields[1]:
        parts = fields[1].split(".")
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            capability = [int(parts[0]), int(parts[1])]
    return {
        "gpu_model": fields[0],
        "compute_capability": capability,
        "clock_frequency_mhz": {
            "current_sm_mhz": _int(fields[2]),
            "max_sm_mhz": _int(fields[3]),
        },
        "total_vram_bytes": (_int(fields[4]) or 0) * 1024 * 1024 or None,
        "gpu_utilization_percent": _float(fields[5]),
    }


def _cuda_cores_for_capability(major: int, minor: int, sm_count: int) -> int | None:
    cores_per_sm = {
        (7, 5): 64,
        (8, 6): 128,
        (8, 9): 128,
        (9, 0): 128,
    }.get((major, minor))
    if cores_per_sm is None or sm_count <= 0:
        return None
    return int(cores_per_sm * sm_count)


def build_cpu_scientific_throughput_efficiency_report(
    *,
    resident_worlds: int,
    active_worlds: int,
    completed_worlds: int,
    elapsed_seconds: float | None,
    ticks_completed: int | None = None,
    generations_completed: int | None = None,
    checkpoint_branches: int | None = None,
    parameter_evaluations: int | None = None,
    storage_bytes_written: int | None = None,
    hardware_descriptor: Mapping[str, Any] | None = None,
    process_sample: Mapping[str, Any] | None = None,
    discovery_metrics: Mapping[str, Any] | None = None,
    time_series_samples: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    hardware = dict(
        detect_cpu_hardware_descriptor()
        if hardware_descriptor is None
        else hardware_descriptor
    )
    process = dict(
        sample_cpu_process_utilization() if process_sample is None else process_sample
    )
    physical = _int(hardware.get("physical_core_count"))
    logical = _int(hardware.get("logical_thread_count"))
    return _base_report(
        backend="cpu",
        hardware=hardware,
        resident_worlds=resident_worlds,
        active_worlds=active_worlds,
        completed_worlds=completed_worlds,
        elapsed_seconds=elapsed_seconds,
        ticks_completed=ticks_completed,
        generations_completed=generations_completed,
        checkpoint_branches=checkpoint_branches,
        parameter_evaluations=parameter_evaluations,
        memory_utilization={
            "resident_memory_efficiency": None,
            "working_set_bytes": process.get("rss_bytes"),
            "peak_memory_bytes": process.get("peak_rss_bytes"),
            "peak_file_cache_bytes": process.get("file_backed_cache_bytes"),
            "peak_anonymous_memory_bytes": process.get("anonymous_bytes"),
            "pss_bytes": process.get("pss_bytes"),
            "swap_usage": (
                dict(process["resource_usage"]).get("swap_count")
                if isinstance(process.get("resource_usage"), Mapping)
                else None
            ),
            "memory_pressure": process.get("memory_pressure"),
        },
        storage_utilization={
            "checkpoint_throughput_bytes_per_second": _ratio(
                storage_bytes_written, elapsed_seconds
            ),
            "evidence_write_throughput_bytes_per_second": _ratio(
                storage_bytes_written, elapsed_seconds
            ),
            "average_checkpoint_latency_seconds": None,
        },
        scheduler_efficiency={
            "scheduler_overhead_estimate": None,
            "synchronization_overhead_estimate": None,
            "average_runnable_threads": None,
            "blocked_thread_estimate": None,
            "scheduler_wait_estimate": None,
        },
        backend_specific={
            "cpu_resource_utilization": {
                "physical_core_count": physical,
                "logical_thread_count": logical,
                "worlds_per_physical_core": _round(_ratio(resident_worlds, physical)),
                "worlds_per_logical_thread": _round(_ratio(resident_worlds, logical)),
                "average_cpu_utilization_percent": None,
                "peak_cpu_utilization_percent": None,
                "average_active_physical_cores": None,
                "average_active_logical_threads": None,
                "process_sample": process,
            }
        },
        discovery_metrics=discovery_metrics,
        time_series_samples=time_series_samples,
    )


def build_cuda_scientific_throughput_efficiency_report(
    *,
    direct_result: Mapping[str, Any],
    production_route_receipt: Mapping[str, Any],
    completed_generations: int,
    compact_summary_rows: int,
    hardware_descriptor: Mapping[str, Any] | None = None,
    discovery_metrics: Mapping[str, Any] | None = None,
    time_series_samples: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    hardware = dict(
        hardware_descriptor or _hardware_from_route(production_route_receipt)
    )
    if not hardware or hardware.get("backend") != "cuda":
        hardware = {**detect_cuda_hardware_descriptor(), **hardware}
    resident_worlds = (
        _int(direct_result.get("grid_blocks"))
        or _int(direct_result.get("batch_cells"))
        or 0
    )
    kernel_launches = (
        _int(direct_result.get("persistent_hot_loop_launch_count"))
        or _int(direct_result.get("kernel_launches"))
        or _int(
            (direct_result.get("native_controller_receipt") or {}).get(
                "total_kernel_launches"
            )
        )
        or 0
    )
    threads_per_block = _int(direct_result.get("threads_per_block")) or 0
    elapsed_ns = _int(direct_result.get("cuda_event_elapsed_ns"))
    elapsed_seconds = (float(elapsed_ns) / 1_000_000_000.0) if elapsed_ns else None
    bounded_ticks = _int(direct_result.get("bounded_device_ticks"))
    sm_count = _int(hardware.get("sm_count"))
    warp_size = _int(hardware.get("warp_size")) or 32
    max_blocks_sm = _int(hardware.get("max_resident_blocks_per_sm")) or 16
    max_warps_sm = _int(hardware.get("max_resident_warps_per_sm")) or 64
    max_threads_sm = _int(hardware.get("max_threads_per_sm")) or (
        max_warps_sm * warp_size
    )
    average_blocks_per_launch = float(resident_worlds) if resident_worlds else 0.0
    warps_per_block = (
        (threads_per_block + warp_size - 1) // warp_size
        if threads_per_block and warp_size
        else None
    )
    theoretical_blocks = sm_count * max_blocks_sm if sm_count else None
    active_sm_fraction = (
        min(1.0, average_blocks_per_launch / float(sm_count)) if sm_count else None
    )
    warp_occupancy = (
        min(
            1.0,
            average_blocks_per_launch
            * float(warps_per_block)
            / float(sm_count * max_warps_sm),
        )
        if sm_count and warps_per_block
        else None
    )
    thread_occupancy = (
        min(
            1.0,
            average_blocks_per_launch
            * float(threads_per_block)
            / float(sm_count * max_threads_sm),
        )
        if sm_count and threads_per_block
        else None
    )
    peak_allocated = _int(
        direct_result.get("torch_observed_interval_peak_allocated_bytes")
    )
    total_vram = _int(hardware.get("total_vram_bytes"))
    gpu_utilization = hardware.get("gpu_utilization_percent")
    classification = _classify(
        active_sm_fraction, average_blocks_per_launch, theoretical_blocks
    )
    return _base_report(
        backend="cuda",
        hardware=hardware,
        resident_worlds=resident_worlds,
        active_worlds=resident_worlds,
        completed_worlds=resident_worlds if completed_generations else 0,
        elapsed_seconds=elapsed_seconds,
        ticks_completed=bounded_ticks,
        generations_completed=completed_generations * resident_worlds,
        checkpoint_branches=1,
        parameter_evaluations=None,
        memory_utilization={
            "resident_memory_efficiency": _round(_ratio(peak_allocated, total_vram)),
            "working_set_bytes": peak_allocated,
            "peak_memory_bytes": peak_allocated,
            "peak_file_cache_bytes": None,
            "peak_anonymous_memory_bytes": None,
            "vram_utilization": _round(_ratio(peak_allocated, total_vram)),
            "vram_efficiency": _round(_ratio(peak_allocated, total_vram)),
        },
        storage_utilization={
            "checkpoint_throughput_bytes_per_second": None,
            "evidence_write_throughput_bytes_per_second": None,
            "average_checkpoint_latency_seconds": None,
        },
        scheduler_efficiency={
            "scheduler_overhead_estimate": None,
            "synchronization_overhead_estimate": None,
        },
        backend_specific={
            "gpu_resource_utilization": {
                "sm_occupancy": _round(active_sm_fraction),
                "resident_worlds": resident_worlds,
                "worlds_per_sm": _round(_ratio(resident_worlds, sm_count)),
                "estimated_idle_sms": (
                    _round(max(0.0, float(sm_count) * (1.0 - active_sm_fraction)))
                    if sm_count and active_sm_fraction is not None
                    else None
                ),
                "block_saturation": _round(
                    _ratio(average_blocks_per_launch, theoretical_blocks)
                ),
                "warp_occupancy_estimate": _round(warp_occupancy),
                "thread_occupancy_estimate": _round(thread_occupancy),
                "average_kernel_duration_seconds": _round(
                    _ratio(elapsed_seconds, kernel_launches), 9
                ),
                "average_blocks_per_world": 1.0 if resident_worlds else None,
                "average_warps_per_block": warps_per_block,
                "average_threads_per_block": threads_per_block or None,
                "kernel_launches": kernel_launches,
                "gpu_utilization_correlation": {
                    "gpu_utilization_percent": gpu_utilization,
                    "correlation_role": "observational_only",
                },
                "occupancy_classification": classification,
            }
        },
        discovery_metrics=discovery_metrics,
        time_series_samples=time_series_samples,
    )


def _hardware_from_route(route: Mapping[str, Any]) -> dict[str, Any]:
    hardware = dict(route.get("hardware") or {})
    if not hardware:
        return {}
    return _enrich_known_cuda_model(
        {
            "schema_version": "eiel_cuda_hardware_descriptor_v1",
            "backend": "cuda",
            "detector_status": hardware.get("detector_status", UNKNOWN),
            "gpu_model": hardware.get("name", hardware.get("gpu_model", UNKNOWN)),
            "compute_capability": hardware.get("compute_capability"),
            "total_vram_bytes": hardware.get("vram_bytes"),
            "sm_count": hardware.get("sm_count"),
            "warp_size": hardware.get("warp_size"),
            "max_resident_warps_per_sm": hardware.get("max_resident_warps_per_sm"),
            "max_resident_blocks_per_sm": hardware.get("max_resident_blocks_per_sm"),
            "max_threads_per_sm": hardware.get("max_threads_per_sm"),
            "sources": ["route_receipt"],
        }
    )


def _classify(
    active_sm_fraction: float | None,
    average_blocks_per_launch: float,
    theoretical_blocks: int | None,
) -> str:
    if theoretical_blocks and average_blocks_per_launch > theoretical_blocks:
        return "oversubscribed"
    if active_sm_fraction is None:
        return UNKNOWN
    if active_sm_fraction < 0.75:
        return "under_filled"
    if active_sm_fraction >= 0.95:
        return "near_saturation"
    return "partially_filled"


def _base_report(
    *,
    backend: str,
    hardware: Mapping[str, Any],
    resident_worlds: int,
    active_worlds: int,
    completed_worlds: int,
    elapsed_seconds: float | None,
    ticks_completed: int | None,
    generations_completed: int | None,
    checkpoint_branches: int | None,
    parameter_evaluations: int | None,
    memory_utilization: Mapping[str, Any],
    storage_utilization: Mapping[str, Any],
    scheduler_efficiency: Mapping[str, Any],
    backend_specific: Mapping[str, Any],
    discovery_metrics: Mapping[str, Any] | None = None,
    time_series_samples: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    supplied_discovery = dict(discovery_metrics or {})
    primary_generations = _int(
        _metric(supplied_discovery, "primary_world_generations_completed")
    )
    primary_ticks = _int(_metric(supplied_discovery, "primary_world_ticks_completed"))
    primary_elapsed = _float(
        _metric(supplied_discovery, "primary_world_elapsed_seconds")
    )
    if completed_worlds == 1:
        primary_generations = (
            primary_generations
            if primary_generations is not None
            else generations_completed
        )
        primary_ticks = primary_ticks if primary_ticks is not None else ticks_completed
        primary_elapsed = (
            primary_elapsed if primary_elapsed is not None else elapsed_seconds
        )
    primary_science_speed = {
        "schema_version": "eiel_ste_primary_science_speed_v1",
        "definition": (
            "speed of one continuously evolving primary scientific world only; "
            "excludes replay, parity, qualification branches, and evidence work"
        ),
        "primary_world_generations_completed": primary_generations,
        "primary_world_ticks_completed": primary_ticks,
        "primary_world_elapsed_seconds": _round(primary_elapsed, 9),
        "primary_world_generations_per_hour": _per_hour(
            primary_generations, primary_elapsed
        ),
        "primary_world_ticks_per_second": _per_second(primary_ticks, primary_elapsed),
        "inference_status": (
            "explicit_or_single_world"
            if primary_generations is not None or primary_ticks is not None
            else "unknown_multi_trajectory_or_not_supplied"
        ),
    }
    scientific_throughput = {
        "schema_version": "eiel_ste_scientific_throughput_v1",
        "definition": "aggregate scientific simulation work performed per time",
        "worlds_completed": completed_worlds,
        "worlds_completed_per_second": _per_second(completed_worlds, elapsed_seconds),
        "worlds_completed_per_hour": _per_hour(completed_worlds, elapsed_seconds),
        "world_generations_completed": generations_completed,
        "world_generations_per_hour": _per_hour(generations_completed, elapsed_seconds),
        "world_ticks_completed": ticks_completed,
        "world_ticks_per_second": _per_second(ticks_completed, elapsed_seconds),
        "parameter_evaluations": parameter_evaluations,
        "parameter_evaluations_per_hour": _per_hour(
            parameter_evaluations, elapsed_seconds
        ),
    }
    qualification_generations = _int(
        _metric(supplied_discovery, "qualification_generations_completed")
    )
    if (
        qualification_generations is None
        and _int(_metric(supplied_discovery, "completed_qualification_cells"))
        is not None
    ):
        qualification_generations = generations_completed
    qualification_worlds = _int(
        _metric(supplied_discovery, "qualification_worlds_completed")
    )
    if (
        qualification_worlds is None
        and _int(_metric(supplied_discovery, "completed_qualification_cells"))
        is not None
    ):
        qualification_worlds = completed_worlds
    qualification_throughput = {
        "schema_version": "eiel_ste_qualification_throughput_v1",
        "definition": (
            "aggregate qualification workload per time; may include replay, parity, "
            "checkpoint validation, comparison branches, and evidence activities"
        ),
        "qualification_worlds_completed": qualification_worlds,
        "qualification_worlds_per_hour": _per_hour(
            qualification_worlds, elapsed_seconds
        ),
        "qualification_generations_completed": qualification_generations,
        "qualification_generations_per_hour": _per_hour(
            qualification_generations, elapsed_seconds
        ),
        "checkpoint_branches": checkpoint_branches,
        "checkpoint_branches_per_hour": _per_hour(checkpoint_branches, elapsed_seconds),
        "qualification_cells_per_hour": _per_hour(
            _int(_metric(supplied_discovery, "completed_qualification_cells")),
            elapsed_seconds,
        ),
        "inference_status": (
            "explicit"
            if qualification_generations is not None or qualification_worlds is not None
            else "unknown_not_supplied"
        ),
    }
    evidence_packages = _int(_metric(supplied_discovery, "evidence_packages"))
    evidence_volume_bytes = _int(_metric(supplied_discovery, "evidence_volume_bytes"))
    evidence_throughput = {
        "schema_version": "eiel_ste_evidence_throughput_v1",
        "definition": "reporting and evidence production per time",
        "evidence_packages": evidence_packages,
        "evidence_packages_per_hour": _per_hour(evidence_packages, elapsed_seconds),
        "evidence_volume_bytes": evidence_volume_bytes,
        "evidence_bytes_per_hour": _per_hour(evidence_volume_bytes, elapsed_seconds),
        "qualification_reports_per_hour": _per_hour(
            _int(_metric(supplied_discovery, "qualification_reports")), elapsed_seconds
        ),
    }
    science = {
        "resident_worlds": resident_worlds,
        "active_worlds": active_worlds,
        "completed_worlds": completed_worlds,
        "elapsed_seconds": _round(elapsed_seconds, 9),
        "worlds_completed_per_second": scientific_throughput[
            "worlds_completed_per_second"
        ],
        "ticks_per_second": scientific_throughput["world_ticks_per_second"],
        "generations_per_hour": scientific_throughput["world_generations_per_hour"],
        "checkpoint_branches_per_hour": qualification_throughput[
            "checkpoint_branches_per_hour"
        ],
        "parameter_evaluations_per_hour": scientific_throughput[
            "parameter_evaluations_per_hour"
        ],
        "deprecated_legacy_aliases": dict(LEGACY_THROUGHPUT_ALIASES),
    }
    execution = {
        "hardware_occupancy": _first_present(
            backend_specific,
            (
                ("gpu_resource_utilization", "sm_occupancy"),
                ("cpu_resource_utilization", "average_cpu_utilization_percent"),
            ),
        ),
        "idle_resource_fraction": None,
        "synchronization_overhead_estimate": scheduler_efficiency.get(
            "synchronization_overhead_estimate"
        ),
        "scheduling_overhead_estimate": scheduler_efficiency.get(
            "scheduler_overhead_estimate"
        ),
    }
    occupancy = _float(execution["hardware_occupancy"])
    if occupancy is not None:
        execution["idle_resource_fraction"] = _round(1.0 - min(1.0, occupancy))
    discovery = _scientific_discovery_efficiency(
        science=science,
        elapsed_seconds=elapsed_seconds,
        default_worlds_completed=completed_worlds,
        default_generations_completed=generations_completed,
        default_ticks_completed=ticks_completed,
        default_checkpoint_branches=checkpoint_branches,
        default_parameter_evaluations=parameter_evaluations,
        metrics={
            **supplied_discovery,
            "primary_world_generations_per_hour": primary_science_speed[
                "primary_world_generations_per_hour"
            ],
            "qualification_generations_per_hour": qualification_throughput[
                "qualification_generations_per_hour"
            ],
            "world_generations_per_hour": scientific_throughput[
                "world_generations_per_hour"
            ],
            "scientific_worlds_completed_per_hour": scientific_throughput[
                "worlds_completed_per_hour"
            ],
            "evidence_packages_per_hour": evidence_throughput[
                "evidence_packages_per_hour"
            ],
            "evidence_bytes_per_hour": evidence_throughput["evidence_bytes_per_hour"],
        },
    )
    resource_efficiency = {
        "level": 1,
        "label": "Resource Efficiency",
        "execution_efficiency": execution,
        "memory_utilization": dict(memory_utilization),
        "storage_utilization": dict(storage_utilization),
        "scheduler_efficiency": dict(scheduler_efficiency),
        "backend_specific": dict(backend_specific),
    }
    scientific_efficiency = {
        "level": 2,
        "label": "Scientific Efficiency",
        **science,
    }
    time_series = _resource_time_series(time_series_samples)
    dashboard_payload = _dashboard_payload_from_parts(
        backend=backend,
        resource_efficiency=resource_efficiency,
        scientific_efficiency=scientific_efficiency,
        discovery_efficiency=discovery,
        time_series=time_series,
        primary_science_speed=primary_science_speed,
        scientific_throughput=scientific_throughput,
        qualification_throughput=qualification_throughput,
        evidence_throughput=evidence_throughput,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_name": "Scientific Throughput Efficiency",
        "backend": backend,
        "engineering_model": _engineering_model_descriptor(),
        "hardware_descriptor": dict(hardware),
        "resource_efficiency": resource_efficiency,
        "scientific_efficiency": scientific_efficiency,
        "scientific_discovery_efficiency": discovery,
        "primary_science_speed": primary_science_speed,
        "scientific_throughput": scientific_throughput,
        "qualification_throughput": qualification_throughput,
        "evidence_throughput": evidence_throughput,
        "throughput_taxonomy": _throughput_taxonomy_descriptor(),
        "legacy_metric_aliases": dict(LEGACY_THROUGHPUT_ALIASES),
        "resource_utilization_time_series": time_series,
        "dashboard_payload": dashboard_payload,
        "science_throughput": science,
        "memory_utilization": dict(memory_utilization),
        "storage_utilization": dict(storage_utilization),
        "scheduler_efficiency": dict(scheduler_efficiency),
        "execution_efficiency": execution,
        "backend_specific": dict(backend_specific),
        "overall_scientific_throughput_efficiency": _overall_grade(science, execution),
        "claims_boundary": (
            "engineering productivity telemetry only; not qualification authority, "
            "scientific success score, or scientific claim"
        ),
    }


def _engineering_model_descriptor() -> dict[str, Any]:
    return {
        "schema_version": ENGINEERING_MODEL_VERSION,
        "levels": [
            {
                "level": 1,
                "name": "Resource Efficiency",
                "question": "How efficiently is the hardware being used?",
            },
            {
                "level": 2,
                "name": "Scientific Efficiency",
                "question": "How much scientific work is being performed per time?",
            },
            {
                "level": 3,
                "name": "Scientific Discovery Efficiency",
                "question": (
                    "How efficiently is computation converted into reproducible "
                    "scientific output?"
                ),
            },
        ],
        "excluded_score_metrics": list(EXCLUDED_DISCOVERY_SCORE_METRICS),
    }


def _throughput_taxonomy_descriptor() -> dict[str, Any]:
    return {
        "schema_version": "eiel_ste_throughput_taxonomy_v1",
        "principle": "a throughput metric answers exactly one question",
        "primary_science_speed": (
            "one continuously evolving primary scientific world; no replay, "
            "parity, qualification branch, or evidence-generation work"
        ),
        "scientific_throughput": "aggregate scientific simulation work performed",
        "qualification_throughput": (
            "aggregate qualification workload, including replay, parity, "
            "comparison, checkpoint validation, and related branches"
        ),
        "evidence_throughput": "reporting and evidence production",
        "legacy_policy": (
            "legacy ambiguous keys are compatibility aliases only and are marked "
            "deprecated; new consumers should use explicit taxonomy fields"
        ),
    }


def _metric(
    metrics: Mapping[str, Any],
    key: str,
    default: int | float | str | None = None,
) -> Any:
    return metrics[key] if key in metrics else default


def _status(value: Any) -> str:
    if value is None:
        return UNKNOWN
    text = str(value).strip()
    return text if text else UNKNOWN


def _scientific_discovery_efficiency(
    *,
    science: Mapping[str, Any],
    elapsed_seconds: float | None,
    default_worlds_completed: int | None,
    default_generations_completed: int | None,
    default_ticks_completed: int | None,
    default_checkpoint_branches: int | None,
    default_parameter_evaluations: int | None,
    metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    supplied = dict(metrics or {})
    worlds_completed = _int(
        _metric(supplied, "worlds_completed", default_worlds_completed)
    )
    generations_completed = _int(
        _metric(supplied, "generations_completed", default_generations_completed)
    )
    ticks_completed = _int(
        _metric(supplied, "ticks_completed", default_ticks_completed)
    )
    checkpoint_branches = _int(
        _metric(supplied, "checkpoint_branches", default_checkpoint_branches)
    )
    parameter_evaluations = _int(
        _metric(supplied, "parameter_evaluations", default_parameter_evaluations)
    )
    evidence_packages = _int(_metric(supplied, "evidence_packages"))
    evidence_volume_bytes = _int(_metric(supplied, "evidence_volume_bytes"))
    qualification_cells = _int(_metric(supplied, "completed_qualification_cells"))
    elapsed_hours = (
        float(elapsed_seconds) / 3600.0 if elapsed_seconds not in (None, 0) else None
    )
    scientific_worlds_per_hour = _round(
        None
        if worlds_completed is None or elapsed_hours in (None, 0)
        else float(worlds_completed) / float(elapsed_hours)
    )
    world_generations_per_hour = _metric(
        supplied, "world_generations_per_hour", science.get("generations_per_hour")
    )
    primary_generations_per_hour = _metric(
        supplied, "primary_world_generations_per_hour"
    )
    qualification_generations_per_hour = _metric(
        supplied, "qualification_generations_per_hour"
    )
    parameter_evaluations_per_hour = science.get("parameter_evaluations_per_hour")
    qualification_cells_per_hour = _round(
        None
        if qualification_cells is None or elapsed_hours in (None, 0)
        else float(qualification_cells) / float(elapsed_hours)
    )
    evidence_packages_per_hour = _metric(
        supplied,
        "evidence_packages_per_hour",
        _round(
            None
            if evidence_packages is None or elapsed_hours in (None, 0)
            else float(evidence_packages) / float(elapsed_hours)
        ),
    )
    evidence_bytes_per_hour = _metric(
        supplied,
        "evidence_bytes_per_hour",
        _round(
            None
            if evidence_volume_bytes is None or elapsed_hours in (None, 0)
            else float(evidence_volume_bytes) / float(elapsed_hours)
        ),
    )
    trend_metrics = {
        "scientific_worlds_completed_per_hour": scientific_worlds_per_hour,
        "world_generations_per_hour": world_generations_per_hour,
        "primary_world_generations_per_hour": primary_generations_per_hour,
        "qualification_generations_per_hour": qualification_generations_per_hour,
        "parameter_evaluations_per_hour": parameter_evaluations_per_hour,
        "qualification_cells_per_hour": qualification_cells_per_hour,
        "evidence_packages_per_hour": evidence_packages_per_hour,
        "evidence_bytes_per_hour": evidence_bytes_per_hour,
        "deprecated_legacy_aliases": dict(LEGACY_THROUGHPUT_ALIASES),
        "worlds_per_hour": _round(
            None
            if worlds_completed is None or elapsed_hours in (None, 0)
            else float(worlds_completed) / float(elapsed_hours)
        ),
        "generations_per_hour": science.get("generations_per_hour"),
    }
    return {
        "schema_version": DISCOVERY_EFFICIENCY_SCHEMA_VERSION,
        "level": 3,
        "label": "Scientific Discovery Efficiency",
        "worlds_completed": worlds_completed,
        "generations_completed": generations_completed,
        "ticks_completed": ticks_completed,
        "checkpoint_branches": checkpoint_branches,
        "evidence_packages": evidence_packages,
        "evidence_volume_bytes": evidence_volume_bytes,
        "parameter_evaluations": parameter_evaluations,
        "branch_continuations": _int(_metric(supplied, "branch_continuations")),
        "qualified_campaigns_completed": _int(
            _metric(supplied, "qualified_campaigns_completed")
        ),
        "completed_parameter_matrices": _int(
            _metric(supplied, "completed_parameter_matrices")
        ),
        "completed_branch_evaluations": _int(
            _metric(supplied, "completed_branch_evaluations")
        ),
        "successful_replay_validations": _int(
            _metric(supplied, "successful_replay_validations")
        ),
        "completed_qualification_cells": qualification_cells,
        "completed_deterministic_replay_checks": _int(
            _metric(supplied, "completed_deterministic_replay_checks")
        ),
        "campaign_completion_fraction": _round(
            _float(_metric(supplied, "campaign_completion_fraction"))
        ),
        "reproducible_campaign_completion_rate": _round(
            _float(_metric(supplied, "reproducible_campaign_completion_rate"))
        ),
        "replay_validation": _status(_metric(supplied, "replay_validation")),
        "evidence_reproducibility": _status(
            _metric(supplied, "evidence_reproducibility")
        ),
        "trend_metrics": trend_metrics,
        "trend_metric_keys": list(TREND_METRIC_KEYS),
        "historical_trend_support": {
            "stable_keys": list(TREND_METRIC_KEYS),
            "schema_stable_for_campaign_series": True,
            "plotting_implemented": False,
        },
        "excluded_score_metrics": list(EXCLUDED_DISCOVERY_SCORE_METRICS),
        "claims_boundary": (
            "measures engineering productivity only; not scientific scoring; does "
            "not score intelligence, emergence, cognition, novelty, importance, "
            "or scientific correctness"
        ),
    }


def _first_present(source: Mapping[str, Any], paths: Sequence[tuple[str, str]]) -> Any:
    for outer, inner in paths:
        value = source.get(outer)
        if isinstance(value, Mapping) and value.get(inner) is not None:
            return value[inner]
    return None


def _overall_grade(science: Mapping[str, Any], execution: Mapping[str, Any]) -> str:
    throughput = _float(science.get("worlds_completed_per_second"))
    idle = _float(execution.get("idle_resource_fraction"))
    if throughput is None:
        return UNKNOWN
    if idle is None:
        return "C"
    if idle <= 0.10:
        return "A"
    if idle <= 0.25:
        return "B"
    if idle <= 0.50:
        return "C"
    return "D"


def aggregate_scientific_throughput_efficiency_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not reports:
        return None
    backends = sorted({str(report.get("backend", UNKNOWN)) for report in reports})
    throughputs = [
        _float(
            (report.get("science_throughput") or {}).get("worlds_completed_per_second")
        )
        for report in reports
    ]
    primary_generation_rates = [
        _float(
            (report.get("primary_science_speed") or {}).get(
                "primary_world_generations_per_hour"
            )
        )
        for report in reports
    ]
    world_generation_rates = [
        _float(
            (report.get("scientific_throughput") or {}).get(
                "world_generations_per_hour"
            )
        )
        for report in reports
    ]
    qualification_generation_rates = [
        _float(
            (report.get("qualification_throughput") or {}).get(
                "qualification_generations_per_hour"
            )
        )
        for report in reports
    ]
    occupancies = [
        _float((report.get("execution_efficiency") or {}).get("hardware_occupancy"))
        for report in reports
    ]
    discovery_reports = [
        dict(report.get("scientific_discovery_efficiency") or {})
        for report in reports
        if isinstance(report.get("scientific_discovery_efficiency"), Mapping)
    ]
    time_series_samples: list[dict[str, Any]] = []
    for report in reports:
        series = report.get("resource_utilization_time_series")
        if not isinstance(series, Mapping):
            continue
        samples = series.get("samples")
        if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
            time_series_samples.extend(
                dict(sample) for sample in samples if isinstance(sample, Mapping)
            )
    science = {
        "mean_worlds_completed_per_second": _round(_mean(throughputs)),
        "max_worlds_completed_per_second": _round(
            max([value for value in throughputs if value is not None], default=0.0)
        ),
    }
    primary_science_speed = {
        "schema_version": "eiel_ste_primary_science_speed_aggregate_v1",
        "definition": "aggregate of per-report primary science speed observations",
        "mean_primary_world_generations_per_hour": _round(
            _mean(primary_generation_rates)
        ),
    }
    scientific_throughput = {
        "schema_version": "eiel_ste_scientific_throughput_aggregate_v1",
        "definition": "aggregate scientific simulation throughput across reports",
        **science,
        "mean_world_generations_per_hour": _round(_mean(world_generation_rates)),
    }
    qualification_throughput = {
        "schema_version": "eiel_ste_qualification_throughput_aggregate_v1",
        "definition": "aggregate qualification throughput across reports",
        "mean_qualification_generations_per_hour": _round(
            _mean(qualification_generation_rates)
        ),
    }
    evidence_throughput = {
        "schema_version": "eiel_ste_evidence_throughput_aggregate_v1",
        "definition": "aggregate evidence throughput across reports",
    }
    execution = {
        "mean_hardware_occupancy": _round(_mean(occupancies)),
    }
    discovery = _aggregate_discovery_reports(discovery_reports)
    return {
        "schema_version": "eiel_scientific_throughput_efficiency_campaign_aggregate_v1",
        "metric_name": "Scientific Throughput Efficiency",
        "backend": backends[0] if len(backends) == 1 else "mixed",
        "engineering_model": _engineering_model_descriptor(),
        "backend_reports": len(reports),
        "backends": backends,
        "resource_efficiency": {
            "level": 1,
            "label": "Resource Efficiency",
            "execution_efficiency": execution,
            "time_series_sample_count": len(time_series_samples),
        },
        "scientific_efficiency": {
            "level": 2,
            "label": "Scientific Efficiency",
            **science,
        },
        "primary_science_speed": primary_science_speed,
        "scientific_throughput": scientific_throughput,
        "qualification_throughput": qualification_throughput,
        "evidence_throughput": evidence_throughput,
        "throughput_taxonomy": _throughput_taxonomy_descriptor(),
        "legacy_metric_aliases": dict(LEGACY_THROUGHPUT_ALIASES),
        "science_throughput": science,
        "execution_efficiency": execution,
        "scientific_discovery_efficiency": discovery,
        "resource_utilization_time_series": _resource_time_series(time_series_samples),
        "dashboard_payload": _dashboard_payload_from_parts(
            backend=backends[0] if len(backends) == 1 else "mixed",
            resource_efficiency={
                "level": 1,
                "label": "Resource Efficiency",
                "execution_efficiency": execution,
                "time_series_sample_count": len(time_series_samples),
            },
            scientific_efficiency={
                "level": 2,
                "label": "Scientific Efficiency",
                **science,
            },
            discovery_efficiency=discovery,
            time_series=_resource_time_series(time_series_samples),
            primary_science_speed=primary_science_speed,
            scientific_throughput=scientific_throughput,
            qualification_throughput=qualification_throughput,
            evidence_throughput=evidence_throughput,
        ),
        "claims_boundary": (
            "campaign aggregate of engineering telemetry only; not qualification authority"
        ),
    }


def _sum_int(reports: Sequence[Mapping[str, Any]], key: str) -> int | None:
    values = [_int(report.get(key)) for report in reports]
    present = [value for value in values if value is not None]
    if not present:
        return None
    return int(sum(present))


def _aggregate_discovery_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    trend_rows = [
        dict(report.get("trend_metrics") or {})
        for report in reports
        if isinstance(report.get("trend_metrics"), Mapping)
    ]
    mean_trend_metrics = {
        key: _round(_mean(_float(row.get(key)) for row in trend_rows))
        for key in TREND_METRIC_KEYS
    }
    for legacy_key in LEGACY_THROUGHPUT_ALIASES:
        legacy_values = [_float(row.get(legacy_key)) for row in trend_rows]
        if any(value is not None for value in legacy_values):
            mean_trend_metrics[legacy_key] = _round(_mean(legacy_values))
    mean_trend_metrics["deprecated_legacy_aliases"] = dict(LEGACY_THROUGHPUT_ALIASES)
    return {
        "schema_version": "eiel_scientific_discovery_efficiency_campaign_aggregate_v1",
        "level": 3,
        "label": "Scientific Discovery Efficiency",
        "source_reports": len(reports),
        "worlds_completed": _sum_int(reports, "worlds_completed"),
        "generations_completed": _sum_int(reports, "generations_completed"),
        "ticks_completed": _sum_int(reports, "ticks_completed"),
        "checkpoint_branches": _sum_int(reports, "checkpoint_branches"),
        "evidence_packages": _sum_int(reports, "evidence_packages"),
        "evidence_volume_bytes": _sum_int(reports, "evidence_volume_bytes"),
        "parameter_evaluations": _sum_int(reports, "parameter_evaluations"),
        "completed_qualification_cells": _sum_int(
            reports, "completed_qualification_cells"
        ),
        "mean_trend_metrics": mean_trend_metrics,
        "trend_metric_keys": list(TREND_METRIC_KEYS),
        "claims_boundary": (
            "aggregate engineering productivity telemetry only; not scientific scoring"
        ),
    }


def collect_run_scientific_throughput_efficiency_reports(
    run_dirs: Iterable[Path],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        manifest = run_dir / "cuda_architecture_b_output_manifest.json"
        if not manifest.is_file():
            continue
        try:
            import json

            payload = json.loads(manifest.read_text(encoding="ascii"))
        except Exception:
            continue
        report = payload.get("scientific_throughput_efficiency")
        if isinstance(report, Mapping):
            reports.append(dict(report))
    return reports


def _dashboard_payload_from_parts(
    *,
    backend: str,
    resource_efficiency: Mapping[str, Any],
    scientific_efficiency: Mapping[str, Any],
    discovery_efficiency: Mapping[str, Any],
    time_series: Mapping[str, Any],
    primary_science_speed: Mapping[str, Any] | None = None,
    scientific_throughput: Mapping[str, Any] | None = None,
    qualification_throughput: Mapping[str, Any] | None = None,
    evidence_throughput: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resource_execution = dict(resource_efficiency.get("execution_efficiency") or {})
    primary = dict(primary_science_speed or {})
    scientific = dict(scientific_throughput or {})
    qualification = dict(qualification_throughput or {})
    evidence = dict(evidence_throughput or {})
    return {
        "schema_version": "eiel_ste_dashboard_payload_v1",
        "title": "Scientific Throughput Efficiency",
        "backend": backend,
        "throughput_taxonomy": _throughput_taxonomy_descriptor(),
        "cards": {
            "resource_efficiency": {
                "label": "Resource Efficiency",
                "hardware_occupancy": (
                    resource_execution.get("hardware_occupancy")
                    if "hardware_occupancy" in resource_execution
                    else resource_execution.get("mean_hardware_occupancy")
                ),
                "idle_resource_fraction": resource_execution.get(
                    "idle_resource_fraction"
                ),
            },
            "primary_science_speed": {
                "label": "Primary Science Speed",
                "definition": primary.get(
                    "definition",
                    "one continuously evolving primary scientific world",
                ),
                "primary_world_generations_per_hour": primary.get(
                    "primary_world_generations_per_hour",
                    primary.get("mean_primary_world_generations_per_hour"),
                ),
                "primary_world_ticks_per_second": primary.get(
                    "primary_world_ticks_per_second"
                ),
                "inference_status": primary.get("inference_status", UNKNOWN),
            },
            "scientific_throughput": {
                "label": "Scientific Throughput",
                "definition": scientific.get(
                    "definition", "aggregate scientific simulation work"
                ),
                "worlds_completed_per_second": scientific.get(
                    "worlds_completed_per_second",
                    scientific.get("mean_worlds_completed_per_second"),
                ),
                "worlds_completed_per_hour": scientific.get(
                    "worlds_completed_per_hour"
                ),
                "world_generations_per_hour": scientific.get(
                    "world_generations_per_hour",
                    scientific.get("mean_world_generations_per_hour"),
                ),
                "world_ticks_per_second": scientific.get("world_ticks_per_second"),
                "parameter_evaluations_per_hour": scientific.get(
                    "parameter_evaluations_per_hour"
                ),
            },
            "qualification_throughput": {
                "label": "Qualification Throughput",
                "definition": qualification.get(
                    "definition", "aggregate qualification workload"
                ),
                "qualification_generations_per_hour": qualification.get(
                    "qualification_generations_per_hour",
                    qualification.get("mean_qualification_generations_per_hour"),
                ),
                "qualification_worlds_per_hour": qualification.get(
                    "qualification_worlds_per_hour"
                ),
                "qualification_cells_per_hour": qualification.get(
                    "qualification_cells_per_hour"
                ),
                "checkpoint_branches_per_hour": qualification.get(
                    "checkpoint_branches_per_hour"
                ),
            },
            "evidence_throughput": {
                "label": "Evidence Throughput",
                "definition": evidence.get(
                    "definition", "reporting and evidence production"
                ),
                "evidence_packages_per_hour": evidence.get(
                    "evidence_packages_per_hour"
                ),
                "evidence_bytes_per_hour": evidence.get("evidence_bytes_per_hour"),
                "qualification_reports_per_hour": evidence.get(
                    "qualification_reports_per_hour"
                ),
            },
            "scientific_efficiency": {
                "label": "Scientific Efficiency (legacy aggregate aliases)",
                "worlds_completed_per_second": scientific_efficiency.get(
                    "worlds_completed_per_second"
                ),
                "mean_worlds_completed_per_second": scientific_efficiency.get(
                    "mean_worlds_completed_per_second"
                ),
                "generations_per_hour": scientific_efficiency.get(
                    "generations_per_hour"
                ),
                "deprecated_legacy_aliases": dict(LEGACY_THROUGHPUT_ALIASES),
            },
            "scientific_discovery_efficiency": {
                "label": "Scientific Discovery Efficiency",
                "worlds_completed": discovery_efficiency.get("worlds_completed"),
                "generations_completed": discovery_efficiency.get(
                    "generations_completed"
                ),
                "parameter_evaluations": discovery_efficiency.get(
                    "parameter_evaluations"
                ),
                "evidence_packages": discovery_efficiency.get("evidence_packages"),
                "completed_qualification_cells": discovery_efficiency.get(
                    "completed_qualification_cells"
                ),
                "replay_validation": discovery_efficiency.get("replay_validation"),
                "evidence_reproducibility": discovery_efficiency.get(
                    "evidence_reproducibility"
                ),
            },
        },
        "time_series": dict(time_series),
        "trend_ready": True,
        "scheduler_action": "none",
        "claims_boundary": (
            "dashboard payload is engineering telemetry only; no optimization, "
            "ranking, scheduling, or scientific scoring"
        ),
    }


def build_scientific_throughput_efficiency_dashboard_payload(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    payload = report.get("dashboard_payload")
    if isinstance(payload, Mapping):
        return dict(payload)
    return _dashboard_payload_from_parts(
        backend=str(report.get("backend", UNKNOWN)),
        resource_efficiency=dict(report.get("resource_efficiency") or {}),
        scientific_efficiency=dict(report.get("scientific_efficiency") or {}),
        discovery_efficiency=dict(report.get("scientific_discovery_efficiency") or {}),
        time_series=dict(report.get("resource_utilization_time_series") or {}),
        primary_science_speed=dict(report.get("primary_science_speed") or {}),
        scientific_throughput=dict(report.get("scientific_throughput") or {}),
        qualification_throughput=dict(report.get("qualification_throughput") or {}),
        evidence_throughput=dict(report.get("evidence_throughput") or {}),
    )


def write_scientific_throughput_efficiency_report_files(
    report: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "scientific_throughput_efficiency_report.json"
    report_md = output_dir / "scientific_throughput_efficiency_report.md"
    dashboard_json = output_dir / "scientific_throughput_efficiency_dashboard.json"
    report_payload = dict(report)
    report_payload["dashboard_payload"] = (
        build_scientific_throughput_efficiency_dashboard_payload(report)
    )
    report_json.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_md.write_text(render_markdown(report_payload) + "\n", encoding="utf-8")
    dashboard_json.write_text(
        json.dumps(report_payload["dashboard_payload"], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return {
        "report_json": str(report_json),
        "report_markdown": str(report_md),
        "dashboard_json": str(dashboard_json),
    }


def estimate_backend_scaling(
    *,
    backend: str,
    hardware_descriptor: Mapping[str, Any],
    resident_world_counts: Sequence[int],
    threads_per_block: int = 128,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if backend == "cuda":
        hardware = _enrich_known_cuda_model(hardware_descriptor)
        sm_count = _int(hardware.get("sm_count"))
        max_blocks_sm = _int(hardware.get("max_resident_blocks_per_sm")) or 16
        max_warps_sm = _int(hardware.get("max_resident_warps_per_sm")) or 64
        warp_size = _int(hardware.get("warp_size")) or 32
        max_threads_sm = _int(hardware.get("max_threads_per_sm")) or (
            max_warps_sm * warp_size
        )
        warps_per_block = (
            (threads_per_block + warp_size - 1) // warp_size
            if threads_per_block and warp_size
            else None
        )
        for worlds in resident_world_counts:
            average_blocks = float(worlds)
            theoretical_blocks = sm_count * max_blocks_sm if sm_count else None
            occupancy = min(1.0, average_blocks / float(sm_count)) if sm_count else None
            rows.append(
                {
                    "backend": "cuda",
                    "resident_worlds": int(worlds),
                    "estimated_occupied_sms": (
                        _round(float(sm_count) * occupancy)
                        if sm_count and occupancy is not None
                        else None
                    ),
                    "estimated_idle_sms": (
                        _round(float(sm_count) * (1.0 - occupancy))
                        if sm_count and occupancy is not None
                        else None
                    ),
                    "estimated_occupancy": _round(occupancy),
                    "worlds_per_sm": _round(_ratio(worlds, sm_count)),
                    "warp_occupancy_estimate": _round(
                        min(
                            1.0,
                            average_blocks
                            * float(warps_per_block or 0)
                            / float(sm_count * max_warps_sm),
                        )
                        if sm_count and warps_per_block
                        else None
                    ),
                    "thread_occupancy_estimate": _round(
                        min(
                            1.0,
                            average_blocks
                            * float(threads_per_block)
                            / float(sm_count * max_threads_sm),
                        )
                        if sm_count and threads_per_block
                        else None
                    ),
                    "classification": _classify(
                        occupancy,
                        average_blocks,
                        theoretical_blocks,
                    ),
                }
            )
        return rows
    physical = _int(hardware_descriptor.get("physical_core_count"))
    logical = _int(hardware_descriptor.get("logical_thread_count"))
    for worlds in resident_world_counts:
        denominator = logical or physical
        occupancy = (
            min(1.0, float(worlds) / float(denominator)) if denominator else None
        )
        rows.append(
            {
                "backend": "cpu",
                "resident_worlds": int(worlds),
                "worlds_per_physical_core": _round(_ratio(worlds, physical)),
                "worlds_per_logical_thread": _round(_ratio(worlds, logical)),
                "estimated_occupancy": _round(occupancy),
                "classification": _classify(occupancy, float(worlds), denominator),
            }
        )
    return rows


def render_markdown(report: Mapping[str, Any]) -> str:
    science = dict(report.get("science_throughput") or {})
    primary_speed = dict(report.get("primary_science_speed") or {})
    scientific_throughput = dict(report.get("scientific_throughput") or {})
    qualification_throughput = dict(report.get("qualification_throughput") or {})
    evidence_throughput = dict(report.get("evidence_throughput") or {})
    execution = dict(report.get("execution_efficiency") or {})
    discovery = dict(report.get("scientific_discovery_efficiency") or {})
    resource = dict(report.get("resource_efficiency") or {})
    time_series = dict(report.get("resource_utilization_time_series") or {})
    lines = [
        "## Scientific Throughput Efficiency",
        "",
        f"Metric: {report.get('metric_name', 'Scientific Throughput Efficiency')}",
        f"Backend: {report.get('backend', UNKNOWN)}",
        "",
        "## Resource Efficiency",
        "",
        f"Hardware occupancy: {execution.get('hardware_occupancy', UNKNOWN)}",
        f"Mean hardware occupancy: {execution.get('mean_hardware_occupancy', UNKNOWN)}",
        f"Idle resource fraction: {execution.get('idle_resource_fraction', UNKNOWN)}",
        f"Time-series samples: {time_series.get('sample_count', 0)}",
        f"Overall execution efficiency: {report.get('overall_scientific_throughput_efficiency', UNKNOWN)}",
        f"Resource level: {resource.get('level', UNKNOWN)}",
        "",
        "## Primary Science Speed",
        "",
        "Definition: one continuously evolving primary scientific world; excludes replay, parity, qualification branches, and evidence work.",
        f"Primary world generations/hour: {primary_speed.get('primary_world_generations_per_hour', primary_speed.get('mean_primary_world_generations_per_hour', UNKNOWN))}",
        f"Primary world ticks/second: {primary_speed.get('primary_world_ticks_per_second', UNKNOWN)}",
        f"Inference status: {primary_speed.get('inference_status', UNKNOWN)}",
        "",
        "## Scientific Throughput",
        "",
        "Definition: aggregate scientific simulation work performed.",
        f"Worlds completed/second: {scientific_throughput.get('worlds_completed_per_second', scientific_throughput.get('mean_worlds_completed_per_second', UNKNOWN))}",
        f"Worlds completed/hour: {scientific_throughput.get('worlds_completed_per_hour', UNKNOWN)}",
        f"World-generations/hour: {scientific_throughput.get('world_generations_per_hour', scientific_throughput.get('mean_world_generations_per_hour', UNKNOWN))}",
        f"World-ticks/second: {scientific_throughput.get('world_ticks_per_second', UNKNOWN)}",
        f"Parameter evaluations/hour: {scientific_throughput.get('parameter_evaluations_per_hour', UNKNOWN)}",
        "",
        "## Qualification Throughput",
        "",
        "Definition: aggregate qualification workload, including replay, parity, checkpoint validation, and comparison branches.",
        f"Qualification generations/hour: {qualification_throughput.get('qualification_generations_per_hour', qualification_throughput.get('mean_qualification_generations_per_hour', UNKNOWN))}",
        f"Qualification worlds/hour: {qualification_throughput.get('qualification_worlds_per_hour', UNKNOWN)}",
        f"Qualification cells/hour: {qualification_throughput.get('qualification_cells_per_hour', UNKNOWN)}",
        f"Checkpoint branches/hour: {qualification_throughput.get('checkpoint_branches_per_hour', UNKNOWN)}",
        f"Inference status: {qualification_throughput.get('inference_status', UNKNOWN)}",
        "",
        "## Evidence Throughput",
        "",
        "Definition: reporting and evidence production.",
        f"Evidence packages/hour: {evidence_throughput.get('evidence_packages_per_hour', UNKNOWN)}",
        f"Evidence bytes/hour: {evidence_throughput.get('evidence_bytes_per_hour', UNKNOWN)}",
        f"Qualification reports/hour: {evidence_throughput.get('qualification_reports_per_hour', UNKNOWN)}",
        "",
        "## Scientific Efficiency (Legacy Aggregate Aliases)",
        "",
        f"Worlds/sec: {science.get('worlds_completed_per_second', science.get('mean_worlds_completed_per_second', UNKNOWN))}",
        f"Generations/hour: {science.get('generations_per_hour', UNKNOWN)}",
        f"Parameter evaluations/hour: {science.get('parameter_evaluations_per_hour', UNKNOWN)}",
        f"Checkpoint branches/hour: {science.get('checkpoint_branches_per_hour', UNKNOWN)}",
        f"Deprecated aliases: {science.get('deprecated_legacy_aliases', dict(LEGACY_THROUGHPUT_ALIASES))}",
        "",
        "## Scientific Discovery Efficiency",
        "",
        f"Worlds completed: {discovery.get('worlds_completed', UNKNOWN)}",
        f"Generations completed: {discovery.get('generations_completed', UNKNOWN)}",
        f"Checkpoint branches: {discovery.get('checkpoint_branches', UNKNOWN)}",
        f"Evidence packages: {discovery.get('evidence_packages', UNKNOWN)}",
        f"Qualification cells: {discovery.get('completed_qualification_cells', UNKNOWN)}",
        f"Parameter evaluations: {discovery.get('parameter_evaluations', UNKNOWN)}",
        f"Campaign completion: {discovery.get('campaign_completion_fraction', UNKNOWN)}",
        f"Replay validation: {discovery.get('replay_validation', UNKNOWN)}",
        f"Evidence reproducibility: {discovery.get('evidence_reproducibility', UNKNOWN)}",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "FEATURE_FLAG",
    "DISCOVERY_EFFICIENCY_SCHEMA_VERSION",
    "ENGINEERING_MODEL_VERSION",
    "SCHEMA_VERSION",
    "STE_REPORT_JSON_SCHEMA",
    "aggregate_scientific_throughput_efficiency_reports",
    "build_resource_utilization_time_series_sample",
    "build_scientific_throughput_efficiency_dashboard_payload",
    "build_cpu_scientific_throughput_efficiency_report",
    "build_cuda_scientific_throughput_efficiency_report",
    "collect_run_scientific_throughput_efficiency_reports",
    "detect_cpu_hardware_descriptor",
    "detect_cuda_hardware_descriptor",
    "estimate_backend_scaling",
    "render_markdown",
    "sample_cpu_process_utilization",
    "scientific_throughput_efficiency_json_schema",
    "scientific_throughput_efficiency_enabled",
    "write_scientific_throughput_efficiency_report_files",
]
