#!/usr/bin/env python3
"""Run one ACS command only after a fail-closed GPU-idle admission gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from engine.campaigns.orchestration import atomic_write_json  # noqa: E402
from engine.perf.acs_gpu_gate_v1 import (  # noqa: E402
    ACSGPUGateError,
    NvidiaSMISamplerV1,
    run_guarded_command_v1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eiel-acs-gpu-guarded-run")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--summary",
        type=Path,
        help="standalone activity summary (defaults beside the receipt)",
    )
    parser.add_argument(
        "--sample-stream",
        type=Path,
        help="append-only JSONL samples (defaults beside the receipt)",
    )
    parser.add_argument(
        "--execution-request",
        type=Path,
        help="construct and bind a canonical `eiel execute REQUEST --human` launch",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--baseline-samples", type=int, default=6)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--max-baseline-sm-percent", type=float, default=5.0)
    parser.add_argument("--max-baseline-encoder-percent", type=float, default=0.0)
    parser.add_argument("--max-baseline-decoder-percent", type=float, default=0.0)
    parser.add_argument("--max-foreign-process-percent", type=float, default=0.0)
    parser.add_argument("--idle-sm-threshold-percent", type=float, default=5.0)
    parser.add_argument("--require-async-checkpoint-overlap", action="store_true")
    parser.add_argument(
        "--require-post-release-zero-residency",
        action="store_true",
        help=(
            "fail unless PID-attributed telemetry proves zero target framebuffer "
            "residency after the manifest-bound terminal CUDA release"
        ),
    )
    parser.add_argument("--allow-baseline-process-command", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _source_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ACSGPUGateError(f"cannot resolve source commit: {exc}") from exc
    return completed.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    execution_request_path = (
        args.execution_request.expanduser().resolve()
        if args.execution_request is not None
        else None
    )
    if execution_request_path is not None:
        if command:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "execution request and explicit command are mutually exclusive",
                    },
                    sort_keys=True,
                )
            )
            return 1
        command = [
            str(_REPOSITORY_ROOT / "tools" / "bin" / "eiel"),
            "execute",
            str(execution_request_path),
            "--human",
        ]
    receipt_path = args.receipt.expanduser().resolve()
    summary_path = (
        args.summary.expanduser().resolve()
        if args.summary is not None
        else receipt_path.parent / "gpu_activity_summary_v1.json"
    )
    sample_stream_path = (
        args.sample_stream.expanduser().resolve()
        if args.sample_stream is not None
        else receipt_path.parent / "gpu_activity_samples_v1.jsonl"
    )
    occupied_paths = [
        path for path in (receipt_path, summary_path, sample_stream_path) if path.exists()
    ]
    if occupied_paths:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "telemetry output already exists",
                    "paths": [str(path) for path in occupied_paths],
                },
                sort_keys=True,
            )
        )
        return 1
    try:
        receipt = run_guarded_command_v1(
            command,
            manifest_path=args.manifest.expanduser().resolve(),
            source_commit=_source_commit(),
            sampler=NvidiaSMISamplerV1(gpu_index=args.gpu_index),
            baseline_sample_count=args.baseline_samples,
            sample_interval_seconds=args.sample_interval_seconds,
            max_baseline_sm_percent=args.max_baseline_sm_percent,
            max_baseline_encoder_percent=args.max_baseline_encoder_percent,
            max_baseline_decoder_percent=args.max_baseline_decoder_percent,
            max_foreign_process_percent=args.max_foreign_process_percent,
            idle_sm_threshold_percent=args.idle_sm_threshold_percent,
            allowed_baseline_process_commands=args.allow_baseline_process_command,
            stdout_log_path=receipt_path.parent / "acs_command.stdout.log",
            stderr_log_path=receipt_path.parent / "acs_command.stderr.log",
            sample_stream_path=sample_stream_path,
            execution_request_path=execution_request_path,
            require_async_checkpoint_overlap=args.require_async_checkpoint_overlap,
            require_post_release_zero_residency=(
                args.require_post_release_zero_residency
            ),
        )
        atomic_write_json(receipt_path, receipt)
        atomic_write_json(summary_path, receipt.get("post_analysis_summary", {}))
    except (ACSGPUGateError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "ok": receipt["status"] == "passed",
                "status": receipt["status"],
                "launched": receipt["launched"],
                "receipt": str(receipt_path),
                "summary": str(summary_path),
                "sample_stream": str(sample_stream_path),
                "rejection_reasons": receipt.get("rejection_reasons", []),
                "runtime_error": receipt.get("runtime_error"),
            },
            sort_keys=True,
        )
    )
    if receipt["status"] == "passed":
        return 0
    return 3 if not receipt["launched"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
