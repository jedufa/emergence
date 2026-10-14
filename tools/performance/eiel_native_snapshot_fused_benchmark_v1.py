#!/usr/bin/env python3
"""Compare the legacy V1 writer with the native capsule publication stages.

This is a technical-only host benchmark.  It uses the real V1 writer and the
real sealed native capsule extension on identical synthetic native-layout
inputs.  It deliberately reports three boundaries separately:

* the legacy Python writer (data, manifest, and publication root);
* native compact capsule stage plus native exact-V1 converter; and
* the fused native compact-to-exact-V1 stage writer.

The native stage measurements stop at a fsynced unpublished data stage.  The
ordinary publication service adds the existing Python manifest/evidence and
atomic publication steps, so this tool never presents a stage-only result as a
full durable campaign result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch

from engine.campaigns.orchestration import atomic_write_json
from engine.native_runtime.cuda_snapshot_plan_v1 import (
    compile_snapshot_plan_v1 as compile_semantic_snapshot_plan_v1,
)
from engine.native_runtime.cuda_snapshot_v1 import (
    SNAPSHOT_COMPATIBILITY_STAGE_ENCODING_POLICY_V1,
    compile_snapshot_plan_v1,
    convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1,
    load_prebuilt_snapshot_extension_v1,
    write_owned_compact_snapshot_capsule_stage_file_v1,
    write_owned_compact_snapshot_to_v1_gzip_stage_file_v1,
)
from engine.native_runtime.cuda_v2_compact_science_capture_v1 import (
    write_compact_science_snapshot_v1,
)
from tools.performance.eiel_snapshot_v1_baseline import _synthetic_inputs


SCHEMA_VERSION = "eiel_native_snapshot_fused_benchmark_report_v1"
CLAIM_CLASS = "TECHNICAL_ONLY_UNQUALIFIED_DEVELOPMENT"
CONTROL_SET_SHA256 = "3" * 64
PUBLICATION_CONTEXT_SHA256 = "4" * 64
MEBIBYTE = 1024 * 1024


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


def _summary(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p95": ordered[max(0, int((0.95 * len(ordered) + 0.999999999)) - 1)],
        "max": max(values),
    }


def _native_state_tensor(synthetic: Mapping[str, Any]) -> torch.Tensor:
    raw = bytearray(synthetic["state"])
    world_count = len(synthetic["active_metrics_projection"]["world_uids"])
    if world_count <= 0 or len(raw) % world_count:
        raise ValueError("synthetic state does not have a world-aligned shape")
    return torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(
        (world_count, len(raw) // world_count)
    )


def _sample_legacy(
    root: Path,
    synthetic: Mapping[str, Any],
    *,
    absolute_tick: int,
    include_genome_catalog: bool,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    receipt = write_compact_science_snapshot_v1(
        root,
        state=synthetic["state"],
        active_metrics_projection=synthetic["active_metrics_projection"],
        world_cells=synthetic["world_cells"],
        absolute_tick=absolute_tick,
        native_world_control_set_sha256=CONTROL_SET_SHA256,
        include_genome_catalog=include_genome_catalog,
    )
    elapsed = time.perf_counter_ns() - started
    data_path = Path(str(receipt["path"])) / str(receipt["data_file"])
    compressed = data_path.read_bytes()
    uncompressed = gzip.decompress(compressed)
    return {
        "elapsed_ns": elapsed,
        "row_count": int(receipt["row_count"]),
        "uncompressed_bytes": len(uncompressed),
        "compressed_bytes": len(compressed),
        "uncompressed_content_sha256": _sha256_bytes(uncompressed),
        "compressed_file_sha256": _sha256_bytes(compressed),
        "compressed": compressed,
    }


def _sample_native_two_stage(
    native_plan: Any,
    state: torch.Tensor,
    metrics: torch.Tensor,
    root: Path,
    *,
    absolute_tick: int,
) -> dict[str, Any]:
    capsule_path = root / "snapshot.compact.capsule"
    gzip_path = root / "snapshot.jsonl.gz"
    started = time.perf_counter_ns()
    compact = write_owned_compact_snapshot_capsule_stage_file_v1(
        native_plan,
        state=state,
        metrics=metrics,
        absolute_tick=absolute_tick,
        stage_path=capsule_path,
    )
    converted = convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
        native_plan,
        capsule_stage_path=capsule_path,
        gzip_stage_path=gzip_path,
    )
    elapsed = time.perf_counter_ns() - started
    compressed = gzip_path.read_bytes()
    uncompressed = gzip.decompress(compressed)
    return {
        "elapsed_ns": elapsed,
        "row_count": int(converted["row_count"]),
        "uncompressed_bytes": len(uncompressed),
        "compressed_bytes": len(compressed),
        "uncompressed_content_sha256": _sha256_bytes(uncompressed),
        "compressed_file_sha256": _sha256_bytes(compressed),
        "compact_capsule_bytes": int(compact["capsule_bytes"]),
        "compressed": compressed,
    }


def _sample_native_compact_only(
    native_plan: Any,
    state: torch.Tensor,
    metrics: torch.Tensor,
    root: Path,
    *,
    absolute_tick: int,
) -> dict[str, Any]:
    """Time only native projection/capsule materialization.

    The exact V1 converter is run after the timed region solely to establish
    parity and output byte counts; its cost is reported by the two-stage row.
    """

    capsule_path = root / "snapshot.compact.capsule"
    gzip_path = root / "snapshot.jsonl.gz"
    started = time.perf_counter_ns()
    compact = write_owned_compact_snapshot_capsule_stage_file_v1(
        native_plan,
        state=state,
        metrics=metrics,
        absolute_tick=absolute_tick,
        stage_path=capsule_path,
    )
    elapsed = time.perf_counter_ns() - started
    converted = convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
        native_plan,
        capsule_stage_path=capsule_path,
        gzip_stage_path=gzip_path,
    )
    compressed = gzip_path.read_bytes()
    uncompressed = gzip.decompress(compressed)
    return {
        "elapsed_ns": elapsed,
        "row_count": int(compact["row_count"]),
        "uncompressed_bytes": len(uncompressed),
        "compressed_bytes": len(compressed),
        "uncompressed_content_sha256": _sha256_bytes(uncompressed),
        "compressed_file_sha256": _sha256_bytes(compressed),
        "compact_capsule_bytes": int(compact["capsule_bytes"]),
        "conversion_elapsed_excluded": True,
        "compressed": compressed,
        "converted_row_count": int(converted["row_count"]),
    }


def _sample_native_fused(
    native_plan: Any,
    state: torch.Tensor,
    metrics: torch.Tensor,
    root: Path,
    *,
    absolute_tick: int,
) -> dict[str, Any]:
    gzip_path = root / "snapshot.jsonl.gz"
    started = time.perf_counter_ns()
    fused = write_owned_compact_snapshot_to_v1_gzip_stage_file_v1(
        native_plan,
        state=state,
        metrics=metrics,
        absolute_tick=absolute_tick,
        stage_path=gzip_path,
        encoding_policy=SNAPSHOT_COMPATIBILITY_STAGE_ENCODING_POLICY_V1,
    )
    elapsed = time.perf_counter_ns() - started
    compressed = gzip_path.read_bytes()
    uncompressed = gzip.decompress(compressed)
    return {
        "elapsed_ns": elapsed,
        "row_count": int(fused["row_count"]),
        "uncompressed_bytes": len(uncompressed),
        "compressed_bytes": len(compressed),
        "uncompressed_content_sha256": _sha256_bytes(uncompressed),
        "compressed_file_sha256": _sha256_bytes(compressed),
        "capsule_bytes": int(fused["capsule_bytes"]),
        "native_timing": dict(fused["timing"]),
        "compressed": compressed,
    }


def run_benchmark(
    *,
    output: Path,
    extension_path: Path | None,
    batch_size: int,
    repetitions: int,
    warmups: int,
    occupancy: str,
    sparse_stride: int,
    absolute_tick: int,
    include_genome_catalog: bool,
) -> dict[str, Any]:
    if batch_size <= 0 or repetitions <= 0 or warmups < 0 or sparse_stride <= 0:
        raise ValueError("batch size/repetitions/stride must be positive and warmups nonnegative")
    repo_root = Path(__file__).resolve().parents[2]
    synthetic = _synthetic_inputs(
        batch_size=batch_size,
        occupancy=occupancy,
        absolute_tick=absolute_tick,
        sparse_stride=sparse_stride,
    )
    world_uids = tuple(synthetic["active_metrics_projection"]["world_uids"])
    semantic_plan = compile_semantic_snapshot_plan_v1(
        world_uids=world_uids,
        world_cells=synthetic["world_cells"],
        native_world_control_set_sha256=CONTROL_SET_SHA256,
        execution_publication_context_sha256=PUBLICATION_CONTEXT_SHA256,
        catalog_absolute_tick=absolute_tick if include_genome_catalog else None,
    )
    loaded = load_prebuilt_snapshot_extension_v1(extension_path)
    native_plan = compile_snapshot_plan_v1(
        semantic_plan,
        loaded_extension=loaded,
    )
    state = _native_state_tensor(synthetic)
    metrics = synthetic["active_metrics_projection"]["tensor"].contiguous()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    samples: dict[str, list[dict[str, Any]]] = {
        "legacy_python_v1": [],
        "native_compact_capsule_only_v1": [],
        "native_compact_plus_converter_v1": [],
        "native_capsule_fused_v1": [],
    }
    benchmark_started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(
        prefix=".eiel-native-snapshot-fused-benchmark-", dir=output.parent
    ) as raw_scratch:
        scratch = Path(raw_scratch)

        def run_one(label: str, sample_index: int, warmup: bool) -> dict[str, Any]:
            phase = "warmup" if warmup else "measured"
            root = scratch / f"{phase}-{label}-{sample_index:04d}"
            root.mkdir()
            if label == "legacy_python_v1":
                result = _sample_legacy(
                    root / "legacy",
                    synthetic,
                    absolute_tick=absolute_tick,
                    include_genome_catalog=include_genome_catalog,
                )
            elif label == "native_compact_capsule_only_v1":
                result = _sample_native_compact_only(
                    native_plan,
                    state,
                    metrics,
                    root,
                    absolute_tick=absolute_tick,
                )
            elif label == "native_compact_plus_converter_v1":
                result = _sample_native_two_stage(
                    native_plan,
                    state,
                    metrics,
                    root,
                    absolute_tick=absolute_tick,
                )
            else:
                result = _sample_native_fused(
                    native_plan,
                    state,
                    metrics,
                    root,
                    absolute_tick=absolute_tick,
                )
            result["sample_index"] = sample_index
            result["warmup"] = warmup
            result["records_per_second"] = result["row_count"] / (
                result["elapsed_ns"] / 1_000_000_000
            )
            result["uncompressed_mebibytes_per_second"] = (
                result["uncompressed_bytes"]
                / MEBIBYTE
                / (result["elapsed_ns"] / 1_000_000_000)
            )
            result["compressed_mebibytes_per_second"] = (
                result["compressed_bytes"]
                / MEBIBYTE
                / (result["elapsed_ns"] / 1_000_000_000)
            )
            # The compressed bytes are retained only for this exact-input
            # parity check and are removed with the bounded scratch root.
            result.pop("compressed", None)
            return result

        for index in range(warmups):
            for label in samples:
                run_one(label, index, True)
        for index in range(repetitions):
            labels = list(samples)
            rotation = index % len(labels)
            labels = labels[rotation:] + labels[:rotation]
            peer_results: dict[str, dict[str, Any]] = {}
            for label in labels:
                peer_results[label] = run_one(label, index, False)
            oracle = peer_results["legacy_python_v1"]
            for label in labels:
                if label == "legacy_python_v1":
                    continue
                if (
                    peer_results[label]["compressed_file_sha256"]
                    != oracle["compressed_file_sha256"]
                    or peer_results[label]["uncompressed_content_sha256"]
                    != oracle["uncompressed_content_sha256"]
                    or peer_results[label]["compressed_bytes"]
                    != oracle["compressed_bytes"]
                    or peer_results[label]["uncompressed_bytes"]
                    != oracle["uncompressed_bytes"]
                ):
                    raise RuntimeError(f"exact V1 parity failed for {label}")
                peer_results[label]["parity_against_legacy"] = True
            oracle["parity_against_legacy"] = True
            for label, result in peer_results.items():
                samples[label].append(result)

    summaries: dict[str, Any] = {}
    for label, rows in samples.items():
        summaries[label] = {
            "timing_ns": _summary([int(row["elapsed_ns"]) for row in rows]),
            "records_per_second": _summary(
                [float(row["records_per_second"]) for row in rows]
            ),
            "uncompressed_mebibytes_per_second": _summary(
                [float(row["uncompressed_mebibytes_per_second"]) for row in rows]
            ),
            "compressed_mebibytes_per_second": _summary(
                [float(row["compressed_mebibytes_per_second"]) for row in rows]
            ),
            "row_count": rows[0]["row_count"],
            "uncompressed_bytes": rows[0]["uncompressed_bytes"],
            "compressed_bytes": rows[0]["compressed_bytes"],
            "samples": rows,
        }
    legacy_median = float(summaries["legacy_python_v1"]["timing_ns"]["median"])
    speedups = {
        label: legacy_median / float(summary["timing_ns"]["median"])
        for label, summary in summaries.items()
        if label != "legacy_python_v1"
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_class": CLAIM_CLASS,
        "benchmark_gate": "G7_NATIVE_HOST_COMPARISON_TECHNICAL_ONLY",
        "science_result": False,
        "qualification_authority_effect": False,
        "prequalification_started": False,
        "qualification_started": False,
        "gpu_used": False,
        "generated_at_utc": _utc_now(),
        "measurement_boundary": {
            "legacy_python_v1": "full V1 writer data plus manifest publication root",
            "native_compact_capsule_only_v1": "native projection and typed capsule stage only; exact converter runs after timing for parity and is explicitly excluded",
            "native_compact_plus_converter_v1": "native compact stage plus exact V1 gzip stage; Python manifest/publication excluded",
            "native_capsule_fused_v1": "native compact projection, capsule authentication, exact JSONL, gzip, hashing, and fsynced unpublished data stage",
            "durable_campaign_completion": "not measured by this host benchmark; the ordinary service retains Python manifest/evidence authority",
            "parity_policy": "every measured native sample must match legacy compressed bytes and decompressed canonical bytes",
        },
        "configuration": {
            "batch_size": batch_size,
            "repetitions": repetitions,
            "warmups": warmups,
            "occupancy": occupancy,
            "sparse_stride": sparse_stride,
            "absolute_tick": absolute_tick,
            "include_genome_catalog": include_genome_catalog,
            "native_plan_sha256": native_plan.identities()["plan_sha256"],
        },
        "extension": {
            "artifact_path": str(loaded.artifact_path),
            "artifact_sha256": loaded.manifest["output"]["sha256"],
            "manifest_path": str(loaded.manifest_path),
            "manifest_sha256": loaded.manifest["manifest_sha256"],
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
        },
        "timing": {
            "clock": "time.perf_counter_ns",
            "benchmark_elapsed_ns": time.perf_counter_ns() - benchmark_started,
            "warmups_excluded_from_summaries": True,
            "sample_order": "rotating legacy/compact/fused order per measured sample",
        },
        "parity": {
            "status": "PASS",
            "measured_native_samples": repetitions * 2,
            "comparison": "compressed_file_sha256, uncompressed_content_sha256, and byte counts",
        },
        "summaries": summaries,
        "median_speedup_over_legacy": speedups,
    }
    report["report_sha256"] = _sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    atomic_write_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--occupancy", choices=("zero", "sparse", "all"), default="all")
    parser.add_argument("--sparse-stride", type=int, default=16)
    parser.add_argument("--absolute-tick", type=int, default=5)
    parser.add_argument("--include-genome-catalog", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_benchmark(
        output=args.output,
        extension_path=args.extension_path,
        batch_size=args.batch_size,
        repetitions=args.repetitions,
        warmups=args.warmups,
        occupancy=args.occupancy,
        sparse_stride=args.sparse_stride,
        absolute_tick=args.absolute_tick,
        include_genome_catalog=args.include_genome_catalog,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output.expanduser().resolve()),
                "report_sha256": report["report_sha256"],
                "median_speedup_over_legacy": report["median_speedup_over_legacy"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "run_benchmark"]
