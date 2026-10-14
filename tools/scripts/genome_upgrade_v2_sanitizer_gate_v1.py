#!/usr/bin/env python3
"""Run CUDA and host sanitizers for the fixed Genome V2 package envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


SCHEMA_VERSION = "genome_v2_sanitizer_qualification_receipt_v1"
CUDA_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
ERROR_SUMMARY = re.compile(r"ERROR SUMMARY:\s+(\d+)\s+errors?")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: list[str],
    *,
    log_path: Path,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    matches = ERROR_SUMMARY.findall(completed.stdout)
    error_count = sum(int(value) for value in matches)
    return {
        "command": command,
        "returncode": completed.returncode,
        "log_path": str(log_path),
        "log_sha256": _sha256(log_path),
        "reported_error_count": error_count,
        "error_summary_observed": bool(matches),
        "ok": completed.returncode == 0 and error_count == 0,
    }


def execute(args: argparse.Namespace) -> int:
    root = args.repository_root.resolve()
    artifact = args.artifact.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_sha256") != _sha256(artifact):
        raise RuntimeError("sanitizer artifact manifest is stale")
    source_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=root, text=True
    ).strip()
    committed = manifest.get("committed_source_snapshot")
    if (
        dirty
        or not isinstance(committed, dict)
        or committed.get("build_source_commit") != source_commit
    ):
        raise RuntimeError(
            "sanitizer qualification requires a clean exact-HEAD artifact"
        )

    cuda_source = root / "tests/perf/genome_upgrade_v2_sanitizer_probe.cu"
    host_source = root / "tests/perf/genome_upgrade_v2_host_sanitizer_probe.cpp"
    cuda_binary = output / "genome_v2_cuda_sanitizer_probe"
    host_binary = output / "genome_v2_host_sanitizer_probe"
    compile_rows = []
    compile_rows.append(
        _run(
            [
                args.nvcc,
                "-std=c++17",
                "-O1",
                "-g",
                "-lineinfo",
                "-gencode=arch=compute_75,code=sm_75",
                f"-I{root / 'engine/perf'}",
                str(cuda_source),
                "-o",
                str(cuda_binary),
            ],
            log_path=output / "cuda_compile.log",
            cwd=root,
        )
    )
    compile_rows.append(
        _run(
            [
                args.cxx,
                "-std=c++20",
                "-O1",
                "-g",
                "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer",
                f"-I{root / 'engine/perf'}",
                str(host_source),
                "-o",
                str(host_binary),
            ],
            log_path=output / "host_compile.log",
            cwd=root,
        )
    )
    if not all(row["ok"] for row in compile_rows):
        raise RuntimeError("Genome V2 sanitizer probe compilation failed")

    cuda_rows = []
    for tool in CUDA_TOOLS:
        command = [
            args.compute_sanitizer,
            "--tool",
            tool,
            "--error-exitcode",
            "99",
            "--target-processes",
            "all",
        ]
        if tool == "memcheck":
            command.extend(["--leak-check", "full"])
        command.append(str(cuda_binary))
        row = _run(
            command,
            log_path=output / f"compute_sanitizer_{tool}.log",
            cwd=root,
        )
        row["tool"] = tool
        cuda_rows.append(row)

    environment = dict(os.environ)
    environment.update(
        {
            "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1",
            "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
        }
    )
    host_row = _run(
        [str(host_binary)],
        log_path=output / "host_asan_ubsan.log",
        cwd=root,
        environment=environment,
    )
    integrated_receipts = []
    for raw in args.integrated_receipt:
        path = Path(raw).expanduser().resolve()
        integrated_receipts.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "non_sanitizer_transaction_evidence": True,
            }
        )
    passed = (
        all(row["ok"] for row in cuda_rows)
        and host_row["ok"]
        and len(integrated_receipts) >= 4
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "qualification_status": "PASS" if passed else "FAIL",
        "source_commit": source_commit,
        "source_clean": True,
        "artifact_sha256": _sha256(artifact),
        "manifest_sha256": _sha256(manifest_path),
        "artifact_target": manifest.get("cuda_arch"),
        "tool_versions": {
            "compute_sanitizer": subprocess.check_output(
                (args.compute_sanitizer, "--version"),
                text=True,
                stderr=subprocess.STDOUT,
            ).strip(),
            "nvcc": subprocess.check_output(
                (args.nvcc, "--version"),
                text=True,
                stderr=subprocess.STDOUT,
            ).strip(),
            "cxx": subprocess.check_output(
                (args.cxx, "--version"),
                text=True,
                stderr=subprocess.STDOUT,
            ).splitlines()[0],
        },
        "coverage": {
            "cuda_probe": [
                "b72_capacity256_resident_arrays",
                "expression_and_runtime_projection",
                "ordinary_birth",
                "turnover_child_commit",
                "rollback_and_overflow_rejection",
                "near_capacity_slot_write",
            ],
            "host_probe": [
                "resident_array_indexing",
                "ordinary_birth_staging_commit_and_overflow",
                "cancellation_request",
                "checkpoint_boundary_terminal_sealing",
                "post_terminal_rejection",
            ],
            "integrated_receipts": integrated_receipts,
        },
        "compile_rows": compile_rows,
        "cuda_sanitizer_rows": cuda_rows,
        "host_asan_ubsan": host_row,
        "suppressions": [],
        "unsupported_checks": [],
        "warning_policy": "any sanitizer warning or error blocks promotion",
    }
    path = output / "genome_v2_sanitizer_qualification_receipt.json"
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--integrated-receipt", action="append", default=[])
    parser.add_argument("--compute-sanitizer", default="compute-sanitizer")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cxx", default="g++")
    return execute(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
