#!/usr/bin/env python3
"""Decompose an existing HAPB midpoint/checkpoint artifact on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Final


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.perf.hapb_v2.checkpoint_decompose_v1 import (
    decompose_torch_zip_v1,
    streaming_release_plan_v1,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = decompose_torch_zip_v1(args.input)
    report["streaming_plan"] = streaming_release_plan_v1(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
