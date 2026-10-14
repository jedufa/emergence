#!/usr/bin/env python3
"""tests/integration/phase5b_e2e/control_interface_sweep_k_resume_matrix.py

Phase 5B Sweep Harness: determinism heatmap under Trait Injection.

What it does
------------
Runs the Phase 5B E2E driver (tests.integration.phase5b_e2e.control_interface_e2e) repeatedly across a grid of
(harness parameters) to find boundary-sensitive determinism failures:

- K (checkpoint interval) and therefore Mode 1 resume boundaries at multiples of K
- C (Mode 0 checkpoint/resume boundary)
- G (total generations)
- seed
- modes (0/1/2 or all)

Design constraints
------------------
- Orchestration only: no engine changes.
- Each case is isolated under a sweep root directory.
- Captures stdout into per-case logs.
- Produces a CSV matrix (plus a small JSON summary).

Usage
-----
python -m tests.integration.phase5b_e2e.control_interface_sweep_k_resume_matrix \
  --ci-fast --modes all --seed 42 \
  --G 100 --K-list 6,8,10,12 --C-list 5,10,15 \
  --keep-zips failures

Notes
-----
This script assumes tests.integration.phase5b_e2e.control_interface_e2e exists and supports the same
Phase4 harness knobs (--mode/--G/--K/--C/--seed/--ci-fast).

Last modified: 2026-02-24 Europe/Vienna
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


MISMATCH_RE = re.compile(r"^ERROR:\s+determinism mismatch in mode\s+(?P<mode>\d+)\s+at resume gen\s+(?P<gen>\d+)\s*$")


@dataclass
class CaseResult:
    case_id: str
    seed: int
    G: int
    K: int
    C: int
    modes: str
    exit_code: int
    status: str
    seconds: float
    mismatch_mode: str
    mismatch_resume_gen: str
    log_path: str


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_repo_root() -> Path:
    # Intended to be run from repo root; fall back to cwd.
    return Path.cwd()


def _parse_int_list(csv_list: str) -> List[int]:
    out: List[int] = []
    for part in [p.strip() for p in csv_list.split(",") if p.strip()]:
        out.append(int(part))
    if not out:
        raise ValueError("empty list")
    return out


def _run_one(
    *,
    repo_root: Path,
    sweep_root: Path,
    seed: int,
    G: int,
    K: int,
    C: int,
    modes: str,
    ci_fast: bool,
    keep_zips: str,
    extra_args: List[str],
) -> CaseResult:
    case_id = f"S{seed:03d}_G{G:04d}_K{K:03d}_C{C:03d}_M{modes}"
    case_dir = sweep_root / case_id
    logs_dir = case_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Provide isolated sweep root to the phase5b_e2e driver (it should forward to harness).
    # If the driver ignores it, isolation still holds for logs.
    env = os.environ.copy()
    env["PHASE5B_SWEEP_CASE_DIR"] = str(case_dir)

    cmd: List[str] = [
        sys.executable,
        "-m",
        "tests.integration.phase5b_e2e.control_interface_e2e",
        "--seed",
        str(seed),
        "--G",
        str(G),
        "--K",
        str(K),
        "--C",
        str(C),
        "--mode",
        str(modes),
    ]
    if ci_fast:
        cmd.append("--ci-fast")

    if keep_zips in ("all", "failures"):
        cmd += ["--keep-zips", keep_zips]

    if extra_args:
        cmd += ["--"] + list(extra_args)

    log_path = logs_dir / f"{case_id}__{_now_stamp()}.log"

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    mismatch_mode = ""
    mismatch_gen = ""

    assert proc.stdout is not None
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"[SWEEP] case_id={case_id}\n")
        lf.write(f"[SWEEP] cmd={' '.join(cmd)}\n")
        lf.flush()

        for line in proc.stdout:
            lf.write(line)
            m = MISMATCH_RE.match(line.strip())
            if m and not mismatch_mode:
                mismatch_mode = m.group("mode")
                mismatch_gen = m.group("gen")

    rc = int(proc.wait())
    dt = time.time() - t0

    status = "PASS" if rc == 0 else "FAIL"
    return CaseResult(
        case_id=case_id,
        seed=seed,
        G=G,
        K=K,
        C=C,
        modes=modes,
        exit_code=rc,
        status=status,
        seconds=dt,
        mismatch_mode=mismatch_mode,
        mismatch_resume_gen=mismatch_gen,
        log_path=str(log_path),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Phase 5B sweep heatmap (K/C boundary sensitivity)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--G", type=int, default=100)
    p.add_argument("--K-list", default="5", help="comma-separated list of K values")
    p.add_argument("--C-list", default="5", help="comma-separated list of C values")
    p.add_argument("--modes", default="all", choices=["0", "1", "2", "all"])
    p.add_argument("--ci-fast", action="store_true")
    p.add_argument("--keep-zips", default="failures", choices=["none", "failures", "all"])
    p.add_argument("--sweep-root", default="", help="override sweep output dir")
    p.add_argument("--cases-max", type=int, default=0, help="optional cap on number of cases")

    # passthrough to simulator (after --)
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    extra_args: List[str] = []
    if "--" in argv:
        idx = argv.index("--")
        extra_args = argv[idx + 1 :]
        argv = argv[:idx]

    args = p.parse_args(argv)

    repo_root = _resolve_repo_root()

    K_list = _parse_int_list(args.K_list)
    C_list = _parse_int_list(args.C_list)

    sweep_root = Path(args.sweep_root) if args.sweep_root else (repo_root / "runs" / f"sweep_phase5b_{_now_stamp()}")
    sweep_root.mkdir(parents=True, exist_ok=True)

    out_csv = sweep_root / "phase5b_sweep_matrix.csv"
    out_json = sweep_root / "phase5b_sweep_summary.json"

    results: List[CaseResult] = []

    case_count = 0
    for K in K_list:
        for C in C_list:
            case_count += 1
            if args.cases_max and len(results) >= int(args.cases_max):
                break
            r = _run_one(
                repo_root=repo_root,
                sweep_root=sweep_root,
                seed=int(args.seed),
                G=int(args.G),
                K=int(K),
                C=int(C),
                modes=str(args.modes),
                ci_fast=bool(args.ci_fast),
                keep_zips=str(args.keep_zips),
                extra_args=extra_args,
            )
            results.append(r)
            # steady progress
            print(f"[SWEEP] {len(results):04d}/{len(K_list)*len(C_list):04d} {r.case_id} -> {r.status} ({r.seconds:.1f}s)")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "case_id",
            "seed",
            "G",
            "K",
            "C",
            "modes",
            "status",
            "exit_code",
            "seconds",
            "mismatch_mode",
            "mismatch_resume_gen",
            "log_path",
        ])
        for r in results:
            w.writerow([
                r.case_id,
                r.seed,
                r.G,
                r.K,
                r.C,
                r.modes,
                r.status,
                r.exit_code,
                f"{r.seconds:.3f}",
                r.mismatch_mode,
                r.mismatch_resume_gen,
                r.log_path,
            ])

    summary: Dict[str, object] = {
        "sweep_root": str(sweep_root),
        "total_cases": len(results),
        "pass": sum(1 for r in results if r.status == "PASS"),
        "fail": sum(1 for r in results if r.status != "PASS"),
        "fails": [
            {
                "case_id": r.case_id,
                "K": r.K,
                "C": r.C,
                "mismatch_mode": r.mismatch_mode,
                "mismatch_resume_gen": r.mismatch_resume_gen,
                "log_path": r.log_path,
            }
            for r in results
            if r.status != "PASS"
        ],
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[SWEEP] CSV: {out_csv}")
    print(f"[SWEEP] JSON: {out_json}")

    return 0 if all(r.status == "PASS" for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
