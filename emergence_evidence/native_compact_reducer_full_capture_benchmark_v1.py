#!/usr/bin/env python3
"""Measure one exact sealed capture through native canonical publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping

from emergence_contracts import canonical_json_bytes
from emergence_evidence.native_compact_reducer_v1 import (
    load_native_compact_reducer_v1,
)
from emergence_evidence.science_compact_world_atlas_v1 import (
    CompactWorldAtlasIncrementalReducerV1,
)


class FullCaptureBenchmarkError(ValueError):
    """The benchmark input, result, or publication boundary is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FullCaptureBenchmarkError(f"{label} is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullCaptureBenchmarkError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise FullCaptureBenchmarkError(f"{label} is not an object")
    return value


def _publish_bytes(path: Path, value: bytes) -> dict[str, Any]:
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    if path.exists() or path.is_symlink() or temporary.exists():
        raise FullCaptureBenchmarkError(
            f"publication target already exists: {path}"
        )
    try:
        with temporary.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "size_bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }


def _result_digest(value: Mapping[str, Any], *, label: str) -> str:
    digest = value.get("result_sha256")
    if (
        not isinstance(digest, Mapping)
        or digest.get("algorithm") != "sha256"
        or not isinstance(digest.get("hex"), str)
    ):
        raise FullCaptureBenchmarkError(f"{label} result digest is absent")
    return str(digest["hex"])


def execute_full_capture_benchmark_v1(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FullCaptureBenchmarkError("output root must be a fresh path")
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter_ns()
    capture_index = _load_object(args.capture_index.resolve(), label="capture index")
    native_artifact = load_native_compact_reducer_v1(
        args.native_artifact.resolve(),
        args.native_manifest.resolve(),
        require_production_qualified=False,
        expected_manifest_sha256=args.native_manifest_sha256,
    )
    reducer = CompactWorldAtlasIncrementalReducerV1(
        args.world_bundle.resolve(),
        max_workers=args.workers,
        native_artifact=native_artifact,
    )
    atlas_result = reducer.finalize(capture_index)
    strategy_result = reducer.strategy_attribution_result
    if not isinstance(strategy_result, dict):
        raise FullCaptureBenchmarkError(
            "native fused strategy attribution result is absent"
        )
    expected_strategy = _load_object(
        args.expected_strategy_result.resolve(),
        label="expected strategy result",
    )
    if strategy_result != expected_strategy:
        raise FullCaptureBenchmarkError(
            "native strategy result differs from the retained Python reference"
        )

    output.mkdir(mode=0o755)
    atlas_publication = _publish_bytes(
        output / "compact_world_atlas_result.json",
        canonical_json_bytes(atlas_result) + b"\n",
    )
    strategy_publication = _publish_bytes(
        output / "strategy_attribution_result.json",
        canonical_json_bytes(strategy_result) + b"\n",
    )
    result_publication_finished = time.perf_counter_ns()
    elapsed_seconds = (result_publication_finished - started) / 1e9
    dispatches = capture_index.get("dispatch_artifacts")
    snapshots = capture_index.get("snapshot_artifacts")
    if not isinstance(dispatches, list) or not isinstance(snapshots, list):
        raise FullCaptureBenchmarkError("capture artifact inventory is invalid")
    event_count = sum(
        int(pointer["event_count"])
        for pointer in dispatches
        if isinstance(pointer, Mapping)
    )
    if event_count != 56_535_520:
        raise FullCaptureBenchmarkError(
            f"full capture event count differs: {event_count}"
        )
    hard_gate_passed = elapsed_seconds <= args.hard_limit_seconds
    receipt = {
        "schema": "eiel_native_compact_reducer_full_capture_benchmark_v1",
        "schema_version": 1,
        "claim_class": "ENGINEERING_ONLY",
        "result": "PASS" if hard_gate_passed else "FAIL",
        "complete_path_seconds": elapsed_seconds,
        "hard_limit_seconds": args.hard_limit_seconds,
        "hard_gate_passed": hard_gate_passed,
        "native_worker_threads": args.workers,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "complete_path_includes": [
            "native artifact and manifest authentication",
            "capture-index authentication",
            "dispatch manifest authentication",
            "gzip payload authentication and decompression",
            "snapshot authentication and reconstruction",
            "native genome and lineage reconstruction",
            "fused native strategy attribution",
            "canonical atlas rendering",
            "canonical strategy rendering",
            "exact retained Python-reference comparison",
            "fsync and immutable canonical result publication"
        ],
        "input": {
            "capture_index": str(args.capture_index.resolve()),
            "capture_index_file_sha256": _sha256(args.capture_index.resolve()),
            "capture_index_sha256": capture_index.get("index_sha256"),
            "world_bundle": str(args.world_bundle.resolve()),
            "world_bundle_file_sha256": _sha256(args.world_bundle.resolve()),
            "native_artifact_sha256": native_artifact.artifact_sha256,
            "native_manifest_sha256": native_artifact.manifest_sha256,
            "expected_strategy_result": str(
                args.expected_strategy_result.resolve()
            ),
            "expected_strategy_file_sha256": _sha256(
                args.expected_strategy_result.resolve()
            ),
            "dispatch_count": len(dispatches),
            "snapshot_count": len(snapshots),
            "event_count": event_count,
            "logical_world_count": len(capture_index.get("ordered_cell_ids", [])),
            "terminal_next_absolute_tick": capture_index.get(
                "terminal_next_absolute_tick"
            ),
            "dropped_record_count": capture_index.get("dropped_record_count"),
        },
        "output": {
            "atlas": atlas_publication,
            "strategy_attribution": strategy_publication,
            "atlas_result_sha256": _result_digest(
                atlas_result,
                label="atlas",
            ),
            "strategy_result_sha256": _result_digest(
                strategy_result,
                label="strategy",
            ),
            "exact_python_reference_parity": True,
            "second_python_attribution_pass": False,
        },
        "qualification": {
            "started": False,
            "evidence_class": "NOT_QUALIFICATION_EVIDENCE",
            "existing_qualification_touched": False,
        },
    }
    receipt_publication = _publish_bytes(
        output / "benchmark_receipt.json",
        canonical_json_bytes(receipt) + b"\n",
    )
    completed_seconds = (time.perf_counter_ns() - started) / 1e9
    return {
        **receipt,
        "process_complete_seconds": completed_seconds,
        "receipt_publication": receipt_publication,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-index", required=True, type=Path)
    parser.add_argument("--world-bundle", required=True, type=Path)
    parser.add_argument("--native-artifact", required=True, type=Path)
    parser.add_argument("--native-manifest", required=True, type=Path)
    parser.add_argument("--native-manifest-sha256", required=True)
    parser.add_argument("--expected-strategy-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hard-limit-seconds", type=float, default=60.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.workers < 1:
        raise FullCaptureBenchmarkError("workers must be positive")
    if args.hard_limit_seconds <= 0:
        raise FullCaptureBenchmarkError("hard limit must be positive")
    result = execute_full_capture_benchmark_v1(args)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["hard_gate_passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FullCaptureBenchmarkError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
