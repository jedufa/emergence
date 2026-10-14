#!/usr/bin/env python3
"""Produce advisory historical storage classifications without mutation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.artifact_lifecycle import (
    ArtifactLifecycleError,
    validate_artifact_disposition_manifest_v1,
)
from tools.scripts.eiel_runtime_storage_prune import build_prune_plan


def inspect_root(root: Path) -> dict[str, Any]:
    runtime_root = root.expanduser().resolve()
    manifest_path = runtime_root / "artifact_disposition_manifest.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        try:
            manifest = validate_artifact_disposition_manifest_v1(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ArtifactLifecycleError) as exc:
            return {
                "root": str(runtime_root),
                "status": "blocked",
                "reason": f"invalid disposition manifest: {type(exc).__name__}: {exc}",
            }
        plan = build_prune_plan(
            runtime_root,
            {
                "event_jsonl",
                "developmental_traces",
                "intermediate_checkpoints",
                "debug_traces",
                "scratch",
                "reproducible_derived",
            },
            disposition_manifest=manifest,
        )
        return {
            "root": str(runtime_root),
            "status": (
                "legacy_manifest_requires_review"
                if isinstance(manifest.get("legacy_reconstruction"), dict)
                and manifest["legacy_reconstruction"].get("status") != "ACCEPTED"
                else "manifest_bound_advisory"
            ),
            "manifest_sha256": manifest["manifest_sha256"],
            "selected_bytes": int(plan["selected_bytes"]),
            "selected_count": int(plan["selected_count"]),
            "automatic_prune_authorized": plan.get("automatic_prune_authorized") is True,
            "plan": plan,
        }
    plan = build_prune_plan(
        runtime_root,
        {
            "event_jsonl",
            "developmental_traces",
            "intermediate_checkpoints",
            "debug_traces",
            "scratch",
            "reproducible_derived",
        },
    )
    core = {
        "root": str(runtime_root),
        "status": "legacy_requires_review",
        "reason": (
            "no lifecycle disposition manifest is present; reconstruct bounded evidence "
            "and obtain an accepted legacy review before prune/archive"
        ),
        "reconstruction_command": (
            "./tools/bin/emergence-python tools/scripts/eiel_legacy_evidence_reconstruct.py "
            f"{runtime_root} --write"
        ),
        "plan": plan,
    }
    return {**core, "report_sha256": canonical_sha256(core)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advisory-only historical storage normalizer")
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--roots-list", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    roots: list[Path] = []
    if args.root is not None:
        roots.append(args.root)
    if args.roots_list is not None:
        roots.extend(
            Path(line.strip())
            for line in args.roots_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not roots:
        raise SystemExit("provide a root or --roots-list")
    report = {
        "schema_version": "eiel_runtime_storage_normalization_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mutation": "none",
        "roots": [inspect_root(root) for root in roots],
    }
    print(json.dumps(report, indent=2, sort_keys=True) + "\n", end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
