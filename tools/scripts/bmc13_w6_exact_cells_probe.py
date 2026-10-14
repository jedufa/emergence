#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "eiel_bmc13_w6_exact_cells_probe_v1"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_bmc2_module():
    return _load_module("bmc2_same_process_isolation_for_bmc13_w6", _project_root() / "tools/scripts/bmc2_same_process_isolation.py")


def _load_bmc4_module():
    return _load_module("bmc4_same_shape_group_probe_for_bmc13_w6", _project_root() / "tools/scripts/bmc4_same_shape_group_probe.py")


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_against_pf1_baselines(rows: list[dict[str, Any]], pf1_summary: dict[str, Any], *, bmc2: Any) -> list[dict[str, Any]]:
    if not pf1_summary:
        return []
    baselines = pf1_summary.get("baselines", {}) or {}
    out: list[dict[str, Any]] = []
    for row in rows:
        baseline = baselines.get(str(row.get("cell")), {})
        compare = bmc2.compare_summaries(baseline, row) if baseline else {"parity": False, "mismatches": ["missing_pf1_baseline"]}
        if int(baseline.get("returncode", 1)) != int(row.get("returncode", 1)):
            compare["parity"] = False
            compare.setdefault("mismatches", []).append("returncode")
        out.append(
            {
                "cell": row.get("cell"),
                "cell_name": row.get("cell_name"),
                "parity": bool(compare.get("parity")),
                "mismatches": list(compare.get("mismatches", [])),
                "pf1_baseline_run_dir": baseline.get("run_dir"),
                "w6_run_dir": row.get("run_dir"),
            }
        )
    return out


def _capture_nvidia_smi() -> dict[str, Any]:
    import subprocess

    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, timeout=15, check=False)
        return {"command": cmd, "returncode": int(cp.returncode), "stdout": cp.stdout.strip(), "stderr": cp.stderr.strip()}
    except Exception as exc:  # pragma: no cover - host dependent.
        return {"command": cmd, "error": repr(exc)}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact BMC same-shape cells through W6 subprocess throughput")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--shape", default="acceptance")
    parser.add_argument("--group", choices=["A", "B"], required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--bank-device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--pf1-summary", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    bmc2 = _load_bmc2_module()
    bmc4 = _load_bmc4_module()
    runs_root = Path(args.runs_root).resolve()
    if runs_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Refusing to reuse existing runs root without --overwrite: {runs_root}")
        shutil.rmtree(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    logs_dir = runs_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    matrix = bmc4.plan_group(
        group=args.group,
        shape=args.shape,
        sequence=args.sequence,
        runs_root=runs_root,
        device=args.device,
        bank_device=args.bank_device,
    )
    planned = list(matrix["baselines"].values())
    workers = max(1, int(args.workers))
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(bmc2.run_baseline, plan, logs_dir=logs_dir): plan for plan in planned}
        for future in as_completed(futures):
            rows.append(future.result())
    total_wall = float(time.perf_counter() - started)
    rows = sorted(rows, key=lambda row: str(row.get("cell", "")))
    pf1_summary = _load_json(Path(args.pf1_summary)) if str(args.pf1_summary).strip() else {}
    comparisons = _compare_against_pf1_baselines(rows, pf1_summary, bmc2=bmc2)
    failures = [row for row in rows if int(row.get("returncode", 1)) != 0]
    parity_failures = [row for row in comparisons if not row.get("parity")]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runs_root": str(runs_root),
        "shape": args.shape,
        "group": args.group,
        "sequence": matrix["sequence"],
        "sequence_cells": [
            {"label": cell.label, "group": cell.group, "seed": cell.seed, "kind": cell.kind}
            for cell in matrix["cells"]
        ],
        "workers": workers,
        "device": args.device,
        "bank_device": args.bank_device,
        "total_wall_seconds": total_wall,
        "sum_child_wall_seconds": float(sum(float(row.get("wall_seconds", 0.0) or 0.0) for row in rows)),
        "mean_child_wall_seconds": float(sum(float(row.get("wall_seconds", 0.0) or 0.0) for row in rows) / max(1, len(rows))),
        "max_child_wall_seconds": float(max((float(row.get("wall_seconds", 0.0) or 0.0) for row in rows), default=0.0)),
        "cells": rows,
        "pf1_summary": str(Path(args.pf1_summary).resolve()) if str(args.pf1_summary).strip() else "",
        "pf1_baseline_comparisons": comparisons,
        "nvidia_smi_after": _capture_nvidia_smi(),
        "status": "passed" if not failures and not parity_failures else "failed",
        "failures": failures,
        "parity_failures": parity_failures,
    }
    _write_json(runs_root / "bmc13_w6_exact_cells_summary.json", payload)
    print(json.dumps({"summary": str(runs_root / "bmc13_w6_exact_cells_summary.json"), "status": payload["status"], "total_wall_seconds": total_wall}, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
