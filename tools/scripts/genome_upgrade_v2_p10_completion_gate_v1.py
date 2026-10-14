#!/usr/bin/env python3
"""Publish the five immutable Genome V2 P1.0 completion artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.authoritative_state.schema import content_sha256  # noqa: E402
from engine.evolution.genome_upgrade_v2_p10_completion_gate_v1 import (  # noqa: E402
    OUTPUT_NAMES,
    reduce_p10_completion_v1,
)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"completion evidence is not readable JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("completion evidence must be an object")
    return dict(value)


def materialize(
    *,
    evidence: Mapping[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    output_directory = output_directory.expanduser().absolute()
    if output_directory.exists() or output_directory.is_symlink():
        raise ValueError("completion output directory already exists")
    artifacts = reduce_p10_completion_v1(evidence)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.",
            dir=output_directory.parent,
        )
    )
    try:
        for filename in OUTPUT_NAMES.values():
            value = artifacts[filename]
            path = temporary / filename
            if isinstance(value, str):
                path.write_text(value, encoding="utf-8")
            else:
                path.write_text(
                    json.dumps(
                        value,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=True,
                    )
                    + "\n",
                    encoding="ascii",
                )
        temporary.rename(output_directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    audit = artifacts[OUTPUT_NAMES["audit"]]
    return {
        "schema_version": "genome_v2_p10_completion_runner_v1",
        "decision": audit["decision"],
        "audit_sha256": audit["audit_sha256"],
        "output_directory": str(output_directory),
        "artifact_names": list(OUTPUT_NAMES.values()),
        "artifact_set_sha256": content_sha256(
            {
                name: (output_directory / name).read_bytes().hex()
                for name in OUTPUT_NAMES.values()
            }
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = materialize(
            evidence=_read(args.evidence),
            output_directory=args.output_directory,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
