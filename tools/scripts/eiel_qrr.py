#!/usr/bin/env python3
"""Repository-side builder for Qualified Runtime Release V1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

# The builder may execute against a clean qualified-source worktree that
# predates QRR itself. Extend only the already-selected ``engine`` package with
# this script's QRR tooling; all campaign/runtime imports remain owned by the
# qualified worktree earlier on PYTHONPATH.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_ENGINE_ROOT = _REPOSITORY_ROOT / "engine"
if _LOCAL_ENGINE_ROOT.is_dir() and str(_LOCAL_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_ENGINE_ROOT))

import engine as _engine_package  # noqa: E402

_QRR_TOOLING_ENGINE = Path(__file__).resolve().parents[2] / "engine"
if str(_QRR_TOOLING_ENGINE) not in _engine_package.__path__:
    _engine_package.__path__.append(str(_QRR_TOOLING_ENGINE))

from engine.campaigns.canonical import pretty_json  # noqa: E402
from engine.qrr.historical_escape_suite_v1 import (  # noqa: E402
    run_historical_escape_suite_v1,
)
from engine.qrr.qualified_runtime_release_v1 import (  # noqa: E402
    DEFAULT_QRR_ROOT,
    build_qualified_runtime_release_v1,
    load_qualified_runtime_release_v1,
)
from engine.qrr.runability_closure_v1 import (  # noqa: E402
    classify_runability_failure_v1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eiel_qrr")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--base-spec", type=Path, required=True)
    build.add_argument("--preflight-receipt", type=Path, required=True)
    build.add_argument("--release-root", type=Path, default=DEFAULT_QRR_ROOT)
    build.add_argument("--source-root", type=Path)
    build.add_argument(
        "--runtime-python",
        type=Path,
        help=(
            "durable exact Python environment sealed into the QRR launcher; "
            "worktree-local interpreters are rejected"
        ),
    )
    build.add_argument(
        "--tick-native-execution",
        type=Path,
        help=(
            "absolute-tick reference execution JSON used by the qualified "
            "preflight; the legacy base-spec horizon remains internal-only"
        ),
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("release", type=Path)
    historical = subparsers.add_parser("historical-suite")
    historical.add_argument("--source-root", type=Path, default=Path.cwd())
    historical.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            destination = build_qualified_runtime_release_v1(
                base_spec_path=args.base_spec,
                preflight_receipt_path=args.preflight_receipt,
                release_root=args.release_root,
                source_root=args.source_root,
                tick_native_execution_path=args.tick_native_execution,
                runtime_python_executable=args.runtime_python,
            )
            release = load_qualified_runtime_release_v1(destination)
            result = {
                "ok": True,
                "schema_version": "eiel_qrr_build_receipt_v1",
                "release_id": release.release_id,
                "release_root": str(destination),
                "manifest_sha256": release.payload()["manifest_sha256"],
            }
        elif args.command == "verify":
            release = load_qualified_runtime_release_v1(args.release)
            result = {
                "ok": True,
                "schema_version": "eiel_qrr_verification_v1",
                "release_id": release.release_id,
                "release_root": str(args.release.expanduser().resolve()),
            }
        else:
            result = run_historical_escape_suite_v1(args.source_root)
            if args.output is not None:
                output = args.output.expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(output.suffix + ".tmp")
                temporary.write_text(pretty_json(result), encoding="utf-8")
                temporary.replace(output)
    except Exception as exc:
        result = {
            "ok": False,
            "schema_version": "eiel_qrr_failure_v1",
            "failure_class": classify_runability_failure_v1(exc, stage=args.command),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    print(pretty_json(result), end="")
    return 0 if result.get("ok") is True or result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
