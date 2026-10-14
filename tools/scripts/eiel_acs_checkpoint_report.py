#!/usr/bin/env python3
"""Build fail-closed ACS0 or ACS4 reports from completed batch manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from engine.perf.acs_checkpoint_report_v1 import (  # noqa: E402
    ACSCheckpointReportError,
    build_acs0_report_v1,
    build_acs4_report_v1,
)


def _read_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACSCheckpointReportError(f"cannot read {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ACSCheckpointReportError(f"{name} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eiel-acs-checkpoint-report")
    subparsers = parser.add_subparsers(dest="command", required=True)
    acs0 = subparsers.add_parser("acs0")
    acs0.add_argument("--manifest", type=Path, required=True)
    acs0.add_argument("--gpu-telemetry-receipt", type=Path)
    acs0.add_argument("--output", type=Path, required=True)
    acs4 = subparsers.add_parser("acs4")
    acs4.add_argument("--synchronous-manifest", type=Path, required=True)
    acs4.add_argument("--acs-manifest", type=Path, required=True)
    acs4.add_argument("--restore-receipt", type=Path)
    acs4.add_argument("--synchronous-gpu-telemetry-receipt", type=Path)
    acs4.add_argument("--acs-gpu-telemetry-receipt", type=Path)
    acs4.add_argument("--horizon-ticks", type=int, default=50_000)
    acs4.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "acs0":
            report = build_acs0_report_v1(
                _read_mapping(args.manifest, name="ACS0 batch manifest"),
                gpu_telemetry_receipt=(
                    _read_mapping(
                        args.gpu_telemetry_receipt,
                        name="ACS0 GPU telemetry receipt",
                    )
                    if args.gpu_telemetry_receipt is not None
                    else None
                ),
            )
        else:
            restore = (
                _read_mapping(args.restore_receipt, name="ACS restore receipt")
                if args.restore_receipt is not None
                else None
            )
            report = build_acs4_report_v1(
                _read_mapping(
                    args.synchronous_manifest,
                    name="synchronous batch manifest",
                ),
                _read_mapping(args.acs_manifest, name="ACS batch manifest"),
                restore_receipt=restore,
                synchronous_gpu_telemetry_receipt=(
                    _read_mapping(
                        args.synchronous_gpu_telemetry_receipt,
                        name="synchronous GPU telemetry receipt",
                    )
                    if args.synchronous_gpu_telemetry_receipt is not None
                    else None
                ),
                acs_gpu_telemetry_receipt=(
                    _read_mapping(
                        args.acs_gpu_telemetry_receipt,
                        name="ACS GPU telemetry receipt",
                    )
                    if args.acs_gpu_telemetry_receipt is not None
                    else None
                ),
                horizon_ticks=args.horizon_ticks,
            )
        _atomic_json(args.output, report)
    except ACSCheckpointReportError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "ok": report.get("measurement_status") == "COMPLETE",
                "measurement_status": report.get("measurement_status"),
                "decision": report.get("decision"),
                "output": str(args.output.expanduser().resolve()),
                "report_sha256": report.get("report_sha256"),
            },
            sort_keys=True,
        )
    )
    return 0 if report.get("measurement_status") == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
