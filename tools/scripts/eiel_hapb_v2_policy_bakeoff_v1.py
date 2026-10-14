#!/usr/bin/env python3
"""Future GPU bake-off: V1 N×1 vs V2 N×64 vs documented GEMM candidate.

CPU-safe by default.  --execute-gpu with EIEL_HAPB_V2_EXPERIMENTAL=1 runs the
real experimental driver: load the isolated extension, evaluate fixtures,
compare to the frozen contract, and write the receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Final


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.perf.hapb_v2.experimental import experimental_enabled_v1
from engine.perf.hapb_v2.policy_bakeoff_v1 import (
    HapbV2BakeoffError,
    build_cpu_receipt_v1,
    run_gpu_bakeoff_v1,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--v1-module")
    parser.add_argument("--policy-kernel-fraction", type=float)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    if args.execute_gpu:
        if not experimental_enabled_v1():
            print(
                "refusing GPU bake-off without EIEL_HAPB_V2_EXPERIMENTAL=1",
                file=sys.stderr,
            )
            return 2
        try:
            receipt = run_gpu_bakeoff_v1(
                output=args.output,
                warmup=args.warmup,
                iters=args.iters,
                policy_kernel_fraction=args.policy_kernel_fraction,
                build_dir=args.build_dir,
            )
        except HapbV2BakeoffError as exc:
            print(str(exc), file=sys.stderr)
            return 4
        print(args.output)
        print(json.dumps({"exact_logits_parity": receipt.get("exact_logits_parity")}))
        return 0
    receipt = build_cpu_receipt_v1()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
