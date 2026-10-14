#!/usr/bin/env python3
"""Measure the exact EED B48 shared owner without dispatching science ticks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tools.scripts import (
    eiel_b48_non_eed_wdl_jungle_allocation_diagnostic_v1 as _base,
)


SCHEMA_VERSION = "eiel_b48_wdl_jungle_eed_allocation_diagnostic_v1"


def run_allocation_diagnostic_v1(
    spec_path: Path,
    *,
    output_root: Path,
    cells: int = 48,
    artifact_override: Path | None = None,
    manifest_override: Path | None = None,
    artifact_source_spec: Path | None = None,
) -> dict[str, object]:
    """Prepare one B48 EED/WDL/Jungle owner; execution remains unavailable."""

    if cells != 48:
        raise _base.AllocationDiagnosticError(
            "the EED fit diagnostic admits only one physical B48 batch"
        )
    return _base.run_allocation_diagnostic_v1(
        spec_path,
        output_root=output_root,
        cells=cells,
        eed_overlay=True,
        artifact_override=artifact_override,
        manifest_override=manifest_override,
        artifact_source_spec=artifact_source_spec,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-override", type=Path)
    parser.add_argument("--manifest-override", type=Path)
    parser.add_argument("--artifact-source-spec", type=Path)
    args = parser.parse_args(argv)
    result = run_allocation_diagnostic_v1(
        args.spec,
        output_root=args.output_root,
        artifact_override=args.artifact_override,
        manifest_override=args.manifest_override,
        artifact_source_spec=args.artifact_source_spec,
    )
    print(
        json.dumps(
            {"outcome": result["outcome"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
