#!/usr/bin/env python3
"""Emit the static HAPB B-ladder resource projection.  No admission claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Final


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.perf.hapb_v2.resource_projector_v1 import project_b_ladder_v1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-unique-per-cell", type=float)
    parser.add_argument("--measured-unique-global", type=float)
    parser.add_argument("--optimistic-ratio", type=float, default=0.25)
    parser.add_argument("--pessimistic-ratio", type=float, default=0.85)
    parser.add_argument("--device-capacity-bytes", type=int)
    parser.add_argument("--host-admission-bytes", type=int)
    args = parser.parse_args()
    report = project_b_ladder_v1(
        measured_unique_per_cell=args.measured_unique_per_cell,
        measured_unique_global=args.measured_unique_global,
        assumed_unique_ratio_optimistic=args.optimistic_ratio,
        assumed_unique_ratio_pessimistic=args.pessimistic_ratio,
        host_admission_bytes=args.host_admission_bytes,
        device_capacity_bytes=args.device_capacity_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
