from __future__ import annotations

"""scripts/phase3_find_first_rng_drift.py

Find the first tick+stage where RNG drift probe dumps differ between two runs.

Inputs
------
--run-a PATH
--run-b PATH

Searches in <run>/state_dumps for files:
  state_<gen>__rng_drift_<stage>_tXXX.json

Compares runner_np_rng_sha256 (and other fields) and reports the earliest mismatch.

This script is best-effort and intentionally dependency-free.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple


_RE = re.compile(r"state_(?P<gen>\d+)__rng_drift_(?P<stage>.+)_t(?P<tick>\d+)\.json$")


def _load(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _index(run_dir: Path) -> Dict[Tuple[int, int, str], Path]:
    out: Dict[Tuple[int, int, str], Path] = {}
    sd = run_dir / "state_dumps"
    if not sd.exists():
        return out
    for p in sd.glob("state_*__rng_drift_*_t*.json"):
        m = _RE.search(p.name)
        if not m:
            continue
        gen = int(m.group("gen"))
        tick = int(m.group("tick"))
        stage = str(m.group("stage"))
        out[(gen, tick, stage)] = p
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    args = ap.parse_args()

    ra = Path(args.run_a)
    rb = Path(args.run_b)
    ia = _index(ra)
    ib = _index(rb)

    keys = sorted(set(ia.keys()) & set(ib.keys()))
    if not keys:
        print("No overlapping rng_drift dumps found.")
        return 2

    for k in keys:
        pa = ia[k]
        pb = ib[k]
        da = _load(pa) or {}
        db = _load(pb) or {}
        if da.get("runner_np_rng_sha256") != db.get("runner_np_rng_sha256"):
            gen, tick, stage = k
            report = {
                "first_drift": {
                    "gen": gen,
                    "tick": tick,
                    "stage": stage,
                    "run_a": {
                        "path": str(pa),
                        "runner_np_rng_sha256": da.get("runner_np_rng_sha256", ""),
                    },
                    "run_b": {
                        "path": str(pb),
                        "runner_np_rng_sha256": db.get("runner_np_rng_sha256", ""),
                    },
                }
            }
            print(json.dumps(report, indent=2, sort_keys=False))
            return 0

    print("No runner_np_rng_sha256 drift found across overlapping dumps.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
