#!/usr/bin/env python3
"""Compile the isolated HAPB V2 N×64 experimental extension.

This is not the production HAPB artifact.  Compilation is allowed without
exclusive GPU access.  The resulting module must not be executed while
Tensor World owns the device unless the operator explicitly runs the GPU
proof packet later.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Final


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.perf.hapb_v2 import HAPB_V2_EXPERIMENTAL_ENV_V1
from engine.perf.hapb_v2.experimental_extension_v1 import (
    compile_experimental_extension_v1,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "_artifacts" / "hapb_v2_cpu_prep_20260816" / "experimental_build",
    )
    args = parser.parse_args()
    if os.environ.get(HAPB_V2_EXPERIMENTAL_ENV_V1) != "1":
        print(
            f"{HAPB_V2_EXPERIMENTAL_ENV_V1} is not set; refusing to compile "
            "the isolated experimental module",
            file=sys.stderr,
        )
        return 2
    compiled = compile_experimental_extension_v1(output_dir=args.output_dir)
    print(getattr(compiled, "__file__", args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
