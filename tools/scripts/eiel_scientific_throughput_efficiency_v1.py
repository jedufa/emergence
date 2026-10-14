"""Offline Scientific Throughput Efficiency estimator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_STE_PATH = (
    PROJECT_ROOT
    / "engine"
    / "native_runtime"
    / "scientific_throughput_efficiency_v1.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "eiel_scientific_throughput_efficiency_module_v1",
    _STE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load STE module: {_STE_PATH}")
_STE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STE)

build_cpu_scientific_throughput_efficiency_report = (
    _STE.build_cpu_scientific_throughput_efficiency_report
)
build_cuda_scientific_throughput_efficiency_report = (
    _STE.build_cuda_scientific_throughput_efficiency_report
)
build_resource_utilization_time_series_sample = (
    _STE.build_resource_utilization_time_series_sample
)
estimate_backend_scaling = _STE.estimate_backend_scaling
render_markdown = _STE.render_markdown
scientific_throughput_efficiency_json_schema = (
    _STE.scientific_throughput_efficiency_json_schema
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate EIEL Scientific Throughput Efficiency from explicit inputs.",
    )
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--resident-worlds", type=_positive_int, required=True)
    parser.add_argument("--completed-worlds", type=_positive_int, default=0)
    parser.add_argument("--elapsed-seconds", type=_positive_float, default=0.0)
    parser.add_argument("--ticks-completed", type=_positive_int)
    parser.add_argument("--generations-completed", type=_positive_int)
    parser.add_argument("--primary-world-generations-completed", type=_positive_int)
    parser.add_argument("--primary-world-ticks-completed", type=_positive_int)
    parser.add_argument("--primary-world-elapsed-seconds", type=_positive_float)
    parser.add_argument("--qualification-generations-completed", type=_positive_int)
    parser.add_argument("--qualification-worlds-completed", type=_positive_int)
    parser.add_argument("--parameter-evaluations", type=_positive_int)
    parser.add_argument("--storage-bytes-written", type=_positive_int)
    parser.add_argument("--time-series-timestamp-seconds", type=_positive_float)
    parser.add_argument("--cpu-utilization-percent", type=_positive_float)
    parser.add_argument("--rss-bytes", type=_positive_int)
    parser.add_argument("--ram-used-bytes", type=_positive_int)
    parser.add_argument("--ram-total-bytes", type=_positive_int)
    parser.add_argument("--gpu-utilization-percent", type=_positive_float)
    parser.add_argument("--vram-used-bytes", type=_positive_int)
    parser.add_argument("--checkpoint-branches", type=_positive_int)
    parser.add_argument("--evidence-packages", type=_positive_int)
    parser.add_argument("--evidence-volume-bytes", type=_positive_int)
    parser.add_argument("--branch-continuations", type=_positive_int)
    parser.add_argument("--qualified-campaigns-completed", type=_positive_int)
    parser.add_argument("--completed-parameter-matrices", type=_positive_int)
    parser.add_argument("--completed-branch-evaluations", type=_positive_int)
    parser.add_argument("--successful-replay-validations", type=_positive_int)
    parser.add_argument("--completed-qualification-cells", type=_positive_int)
    parser.add_argument("--completed-deterministic-replay-checks", type=_positive_int)
    parser.add_argument("--campaign-completion-fraction", type=float)
    parser.add_argument("--reproducible-campaign-completion-rate", type=float)
    parser.add_argument("--replay-validation", default=None)
    parser.add_argument("--evidence-reproducibility", default=None)
    parser.add_argument("--sm-count", type=_positive_int)
    parser.add_argument("--threads-per-block", type=_positive_int, default=128)
    parser.add_argument("--kernel-launches", type=_positive_int, default=1)
    parser.add_argument("--max-blocks-per-sm", type=_positive_int, default=16)
    parser.add_argument("--max-warps-per-sm", type=_positive_int, default=64)
    parser.add_argument("--max-threads-per-sm", type=_positive_int, default=2048)
    parser.add_argument("--total-vram-bytes", type=_positive_int)
    parser.add_argument("--peak-vram-bytes", type=_positive_int)
    parser.add_argument("--physical-cores", type=_positive_int)
    parser.add_argument("--logical-threads", type=_positive_int)
    parser.add_argument(
        "--scaling-worlds",
        default="",
        help="comma-separated resident world counts for offline scaling estimates",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument(
        "--schema", action="store_true", help="emit the report JSON schema"
    )
    parser.add_argument("--output", type=Path)
    return parser


def _discovery_metrics(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "worlds_completed": args.completed_worlds,
        "generations_completed": args.generations_completed,
        "ticks_completed": args.ticks_completed,
        "primary_world_generations_completed": (
            args.primary_world_generations_completed
        ),
        "primary_world_ticks_completed": args.primary_world_ticks_completed,
        "primary_world_elapsed_seconds": args.primary_world_elapsed_seconds,
        "qualification_generations_completed": (
            args.qualification_generations_completed
        ),
        "qualification_worlds_completed": args.qualification_worlds_completed,
        "checkpoint_branches": args.checkpoint_branches,
        "evidence_packages": args.evidence_packages,
        "evidence_volume_bytes": args.evidence_volume_bytes,
        "parameter_evaluations": args.parameter_evaluations,
        "branch_continuations": args.branch_continuations,
        "qualified_campaigns_completed": args.qualified_campaigns_completed,
        "completed_parameter_matrices": args.completed_parameter_matrices,
        "completed_branch_evaluations": args.completed_branch_evaluations,
        "successful_replay_validations": args.successful_replay_validations,
        "completed_qualification_cells": args.completed_qualification_cells,
        "completed_deterministic_replay_checks": (
            args.completed_deterministic_replay_checks
        ),
        "campaign_completion_fraction": args.campaign_completion_fraction,
        "reproducible_campaign_completion_rate": (
            args.reproducible_campaign_completion_rate
        ),
        "replay_validation": args.replay_validation,
        "evidence_reproducibility": args.evidence_reproducibility,
    }


def _time_series_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    values = [
        args.time_series_timestamp_seconds,
        args.cpu_utilization_percent,
        args.rss_bytes,
        args.ram_used_bytes,
        args.ram_total_bytes,
        args.gpu_utilization_percent,
        args.vram_used_bytes,
        args.peak_vram_bytes,
        args.total_vram_bytes,
        args.storage_bytes_written,
    ]
    if all(value is None for value in values):
        return []
    return [
        build_resource_utilization_time_series_sample(
            timestamp_seconds=args.time_series_timestamp_seconds,
            cpu_utilization_percent=args.cpu_utilization_percent,
            rss_bytes=args.rss_bytes,
            ram_used_bytes=args.ram_used_bytes,
            ram_total_bytes=args.ram_total_bytes,
            gpu_utilization_percent=args.gpu_utilization_percent,
            vram_used_bytes=args.vram_used_bytes or args.peak_vram_bytes,
            vram_total_bytes=args.total_vram_bytes,
            storage_bytes_written=args.storage_bytes_written,
            source="offline_cli_input",
        )
    ]


def estimate(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend == "cpu":
        return build_cpu_scientific_throughput_efficiency_report(
            resident_worlds=args.resident_worlds,
            active_worlds=args.resident_worlds,
            completed_worlds=args.completed_worlds,
            elapsed_seconds=args.elapsed_seconds or None,
            ticks_completed=args.ticks_completed,
            generations_completed=args.generations_completed,
            checkpoint_branches=args.checkpoint_branches,
            parameter_evaluations=args.parameter_evaluations,
            storage_bytes_written=args.storage_bytes_written,
            hardware_descriptor={
                "schema_version": "eiel_cpu_hardware_descriptor_v1",
                "backend": "cpu",
                "detector_status": "offline_input",
                "cpu_model": "offline",
                "physical_core_count": args.physical_cores,
                "logical_thread_count": args.logical_threads,
                "cache_sizes": {},
                "ram_bytes": None,
                "platform": "offline",
            },
            process_sample={},
            discovery_metrics=_discovery_metrics(args),
            time_series_samples=_time_series_samples(args),
        )
    elapsed_ns = (
        int(args.elapsed_seconds * 1_000_000_000) if args.elapsed_seconds else 0
    )
    return build_cuda_scientific_throughput_efficiency_report(
        direct_result={
            "grid_blocks": args.resident_worlds,
            "threads_per_block": args.threads_per_block,
            "persistent_hot_loop_launch_count": args.kernel_launches,
            "cuda_event_elapsed_ns": elapsed_ns,
            "bounded_device_ticks": args.ticks_completed,
            "torch_observed_interval_peak_allocated_bytes": args.peak_vram_bytes,
        },
        production_route_receipt={},
        completed_generations=args.generations_completed or 0,
        compact_summary_rows=args.generations_completed or 0,
        hardware_descriptor={
            "schema_version": "eiel_cuda_hardware_descriptor_v1",
            "backend": "cuda",
            "detector_status": "offline_input",
            "gpu_model": "offline",
            "sm_count": args.sm_count,
            "warp_size": 32,
            "max_resident_blocks_per_sm": args.max_blocks_per_sm,
            "max_resident_warps_per_sm": args.max_warps_per_sm,
            "max_threads_per_sm": args.max_threads_per_sm,
            "total_vram_bytes": args.total_vram_bytes,
        },
        discovery_metrics=_discovery_metrics(args),
        time_series_samples=_time_series_samples(args),
    )


def _parse_scaling_worlds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    args = build_parser().parse_args()
    if args.schema:
        payload = (
            json.dumps(
                scientific_throughput_efficiency_json_schema(),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    report = estimate(args)
    scaling = (
        estimate_backend_scaling(
            backend=args.backend,
            hardware_descriptor=report["hardware_descriptor"],
            resident_world_counts=_parse_scaling_worlds(args.scaling_worlds),
            threads_per_block=getattr(args, "threads_per_block", 128),
        )
        if args.scaling_worlds
        else []
    )
    payload_object = {"report": report, "scaling_table": scaling}
    payload = json.dumps(payload_object, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if args.json:
        print(payload, end="")
    else:
        print(render_markdown(report))
        if scaling:
            print("Scaling table:")
            for row in scaling:
                print(f"- {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
