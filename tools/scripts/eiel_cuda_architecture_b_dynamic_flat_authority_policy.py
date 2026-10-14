#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.perf.cuda_architecture_b_dynamic_flat_authority import (  # noqa: E402
    AUTHORITY_POLICY_ID,
    emit_authority_policy_evidence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit the ratified Dynamic Flat authority-policy receipts."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = emit_authority_policy_evidence(args.output_dir)
    print(
        json.dumps(
            {
                "ok": True,
                "policy_id": AUTHORITY_POLICY_ID,
                "paths": paths,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
