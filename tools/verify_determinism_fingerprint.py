#!/usr/bin/env python3
"""Compatibility wrapper for shared determinism fingerprint verification."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MONOREPO_ROOT = Path(__file__).resolve().parents[4]
if str(_MONOREPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MONOREPO_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments_shared.determinism.verify_determinism_fingerprint import *  # noqa


if __name__ == "__main__":
    raise SystemExit(main())
