#!/usr/bin/env python3
"""Reduce exact Genome V2 environment-switch evidence into immutable outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.authoritative_state.schema import content_sha256  # noqa: E402
from engine.evolution.genome_upgrade_v2_environment_switch_gate_v1 import (  # noqa: E402
    evaluate_environment_switch_v1,
    preregistration_payload_v1,
)


def _read(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(
    *,
    authority: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    output_directory: Path,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=False)
    preregistration_path = output_directory / (
        "genome_v2_environment_switch_preregistration.json"
    )
    decision_path = output_directory / "genome_v2_environment_switch_decision.json"
    manifest_path = output_directory / "genome_v2_environment_switch_manifest.json"
    decision = evaluate_environment_switch_v1(evidence, authority=authority)
    _write(preregistration_path, preregistration_payload_v1())
    _write(decision_path, decision)
    manifest = {
        "schema_version": "genome_v2_p10_environment_switch_runner_v1",
        "decision": decision["decision"],
        "decision_sha256": decision["decision_sha256"],
        "artifacts": {
            "preregistration": {
                "path": preregistration_path.name,
                "sha256": _sha256(preregistration_path),
            },
            "decision": {
                "path": decision_path.name,
                "sha256": _sha256(decision_path),
            },
        },
    }
    manifest["manifest_sha256"] = content_sha256(manifest)
    _write(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        authority = _read(args.authority, label="authority")
        evidence = _read(args.evidence, label="evidence")
        if not isinstance(authority, Mapping):
            raise ValueError("authority must be an object")
        if (
            isinstance(evidence, (str, bytes))
            or not isinstance(evidence, Sequence)
        ):
            raise ValueError("evidence must be an array")
        manifest = materialize(
            authority=authority,
            evidence=evidence,
            output_directory=args.output_directory,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
