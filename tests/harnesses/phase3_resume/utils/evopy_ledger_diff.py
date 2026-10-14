#!/usr/bin/env python3
"""tools/evopy_ledger_diff.py

FDP v3 helper: deterministic diff of two evolution_py ledger JSON files.

Why this exists
---------------
Humans are great at missing the one line that matters.
This script prints:
  - total_calls delta
  - top per-key deltas (sorted by absolute delta, then key)

Usage
-----
python tools/evopy_ledger_diff.py path/to/fresh.json path/to/resume.json

Notes
-----
- Ignores any non-deterministic metadata fields not required for comparison.
- Assumes ledger JSON schema:
    {
      "schema_version": "...",   (optional)
      "epoch": "...",           (optional)
      "gen": 0,                  (optional)
      "tick": 54,                (optional)
      "total_calls": 123,        (optional)
      "counts": {"key": 1, ...} OR direct mapping {"key": 1, ...}
    }
"""

from __future__ import annotations

import json
import sys
from typing import Dict, Tuple


def _load(path: str) -> Tuple[Dict[str, int], int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Accept either {"counts": {...}} or direct mapping.
    if isinstance(obj, dict) and "counts" in obj and isinstance(obj["counts"], dict):
        counts = obj["counts"]
        total = int(obj.get("total_calls", sum(int(v) for v in counts.values())))
    elif isinstance(obj, dict):
        # Filter out metadata-looking keys.
        counts = {k: v for k, v in obj.items() if isinstance(v, int)}
        total = int(obj.get("total_calls", sum(int(v) for v in counts.values())))
    else:
        raise ValueError(f"Unexpected JSON shape in {path}")

    norm: Dict[str, int] = {}
    for k, v in counts.items():
        if not isinstance(k, str):
            continue
        try:
            norm[str(k)] = int(v)
        except Exception:
            continue
    return norm, total


def main(argv) -> int:
    if len(argv) != 3:
        print("Usage: python tools/evopy_ledger_diff.py <fresh_ledger.json> <resume_ledger.json>")
        return 2

    a_counts, a_total = _load(argv[1])
    b_counts, b_total = _load(argv[2])

    keys = set(a_counts.keys()) | set(b_counts.keys())
    deltas = []
    for k in keys:
        da = int(a_counts.get(k, 0))
        db = int(b_counts.get(k, 0))
        d = db - da
        if d != 0:
            deltas.append((abs(d), d, k))

    deltas.sort(key=lambda t: (t[0], t[2]))

    print(f"total_calls fresh={a_total} resume={b_total} delta={(b_total - a_total)}")

    if not deltas:
        print("per-key deltas: NONE")
        return 0

    print("per-key deltas (resume - fresh):")
    for _, d, k in reversed(deltas[:50]):
        print(f"{d:+d}\t{k}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
