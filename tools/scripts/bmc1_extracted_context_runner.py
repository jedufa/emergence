#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one EIEL cell through the extracted BMC-1 context API")
    parser.add_argument("--result-json", required=True, help="Path for the small execution result JSON")
    parser.add_argument("engine_args", nargs=argparse.REMAINDER, help="Arguments forwarded to engine/main.py")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = _project_root()
    engine_dir = project_root / "engine"
    for path in (str(engine_dir), str(project_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    forwarded = list(args.engine_args or [])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    import main as engine_main  # noqa: WPS433

    plan = engine_main.build_run_plan_from_argv(forwarded)
    result = engine_main.execute_run_plan(plan)
    payload = {
        "status": result.status,
        "returncode": int(result.returncode),
        "run_dir": result.run_dir,
        "metrics_path": result.metrics_path,
        "run_config_path": result.run_config_path,
        "manifest_path": result.manifest_path,
        "wall_seconds": float(result.wall_seconds),
        "plan": {
            "run_id": plan.run_id,
            "runs_dir": plan.runs_dir,
            "seed": int(plan.seed),
            "device": plan.device,
            "bank_device": plan.bank_device,
        },
    }
    out = Path(args.result_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
