#!/usr/bin/env python3
from __future__ import annotations

"""Generate compact REP-2F existing-world action opportunity evidence."""

import argparse
import json
from pathlib import Path
from typing import Any

from evolution.structural_action_opportunity_gate import build_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate REP-2F evidence.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--runtime-root", default="")
    parser.add_argument("--test-command-summary-json", default="")
    args = parser.parse_args()
    commands: list[dict[str, Any]] = []
    if args.test_command_summary_json:
        commands = json.loads(Path(args.test_command_summary_json).read_text(encoding="utf-8"))
    build_evidence(
        Path(args.out_dir),
        command_results=commands,
        runtime_root=Path(args.runtime_root) if args.runtime_root else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
