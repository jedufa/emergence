#!/usr/bin/env python3
"""Run bounded Q3 probes for the exact combined B48 WDL/Jungle/EED owner."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from tools.scripts import (
    eiel_b48_non_eed_wdl_jungle_qualprep_runtime_v1 as _base,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate to the shared harness while requiring the EED owner."""

    values = list(sys.argv[1:] if argv is None else argv)
    return _base.main([*values, "--eed-overlay"])


if __name__ == "__main__":
    raise SystemExit(main())
