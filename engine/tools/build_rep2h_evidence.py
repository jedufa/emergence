from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolution.structural_tick_reproduction_selection import build_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Build REP-2H tick-loop reproductive selection evidence.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    result = build_evidence(
        Path(args.out_dir),
        runtime_root=Path(args.runtime_root),
        workers=int(args.workers),
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
