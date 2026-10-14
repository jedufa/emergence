#!/usr/bin/env python3
"""Launch the standalone Emergence simulation slot monitor."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.slot_monitor.server import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
