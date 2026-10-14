#!/usr/bin/env python3
"""Inspect CPU-readable policy banks or synthetic fixtures for uniqueness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Final

import numpy as np


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.perf.hapb_v2.uniqueness_v1 import (
    analyze_canonical_bytes_v1,
    analyze_policy_bank_v1,
    analyze_source_projection_root_v1,
    project_pool_bytes_v1,
)


def _synthetic_bank() -> np.ndarray:
    rng = np.random.default_rng(0x554E4951)
    bank = rng.standard_normal((2, 8, 9991), dtype=np.float32)
    bank[0, 3] = bank[0, 0]
    bank[1, 1] = bank[0, 0]
    return bank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical-bytes", type=Path)
    parser.add_argument("--source-projection-root", type=Path)
    parser.add_argument("--cells", type=int, default=48)
    parser.add_argument("--slots", type=int, default=512)
    parser.add_argument("--source", default="synthetic_self_check")
    args = parser.parse_args()
    if args.source_projection_root is not None:
        report = analyze_source_projection_root_v1(args.source_projection_root)
    elif args.canonical_bytes is not None:
        report = analyze_canonical_bytes_v1(
            args.canonical_bytes.read_bytes(),
            cells=args.cells,
            slots=args.slots,
            source=str(args.canonical_bytes),
        )
    else:
        report = analyze_policy_bank_v1(_synthetic_bank(), source=args.source)
    report["pool_projection"] = project_pool_bytes_v1(
        cells=int(report["cells"]),
        unique_per_cell=float(np.mean(report["per_cell_unique_contents"])),
        unique_global=float(report["global_unique_contents"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
