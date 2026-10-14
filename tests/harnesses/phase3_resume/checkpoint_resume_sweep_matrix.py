#!/usr/bin/env python3
"""
tests/determinism/harnesses/checkpoint_resume_sweep_matrix.py

Automated sweep harness to build a PASS/FAIL matrix for checkpoint/resume determinism.

Goal
----
Systematically run the existing Phase 3 E2E harness (bash) across a grid of:
- GEN_TOTAL (total generations for Run A)
- resume_gen (checkpoint generation to resume from)

It records:
- exit code per case
- PASS/FAIL classification
- produced zip path (if any)
- detected Run A / Run B directories (best-effort)

Design constraints
------------------
- Does NOT modify engine logic; it only orchestrates repeated runs.
- Produces steady console progress so the operator can see activity.
- Tries to keep disk usage sane by optionally deleting PASS zips.

Usage examples
--------------
# Full heatmap for small range (can be many runs):
python -m tests.determinism.harnesses.checkpoint_resume_sweep_matrix \
  --gen-min 5 --gen-max 30 --resume-mode all --keep-zips failures

# Target tail-edge pattern:
python -m tests.determinism.harnesses.checkpoint_resume_sweep_matrix \
  --gen-min 5 --gen-max 60 --resume-mode tail2 --keep-zips failures

Environment variables
---------------------
This script intentionally inherits the caller environment, so you can keep using your
usual forensic env var set (EIEL_* etc). You can override standard harness knobs via:
- SEED, POP, DEVICE, BANK_DEVICE, PERIOD
- SWEEP_ROOT_DIR (default: runs/sweep_<timestamp>)
- CLEAN_RUNS (ignored for sweep; each case uses isolated RUNS_DIR)
- ZIP_OUT_DIR (overridden per case)

Notes
-----
- The underlying bash harness always writes Run A / Run B run dirs under RUNS_DIR.
- Each case uses its own RUNS_DIR to prevent cross-contamination.
- The bash harness is expected at:
  tests/determinism/harnesses/checkpoint_resume_determinism_e2e.sh

Last modified: 2026-02-24 Europe/Vienna
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RUN_A_RE = re.compile(r"^\[INFO\]\s+Run A:\s+(?P<path>.+?)\s*$", re.MULTILINE)
RUN_B_RE = re.compile(r"^\[INFO\]\s+Run B\(resume_gen=(?P<gen>\d+)\):\s+(?P<path>.+?)\s*$", re.MULTILINE)
ARCHIVE_RE = re.compile(r"^\[INFO\]\s+Archive:\s+(?P<path>.+?)\s*$", re.MULTILINE)


@dataclass
class CaseResult:
    case_id: str
    gen_total: int
    resume_gen: int
    exit_code: int
    status: str
    seconds: float
    run_a: str
    run_b: str
    zip_path: str
    log_path: str


def _now_vienna_stamp() -> str:
    # Vienna local time is assumed to be system-local on the operator box.
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_repo_root() -> Path:
    # Best-effort: assume this file is executed as a module from repo root.
    # If not, use current working directory.
    return Path.cwd()


def _compute_resume_list(gen_total: int, mode: str, step: int, *, policy_tick_stride: int, tail_window: int) -> List[int]:
    if gen_total <= 1:
        return []
    if mode == "all":
        return list(range(1, gen_total))
    if mode == "tail2":
        # Only the classic "near-end" boundary
        g = gen_total - 2
        return [g] if g >= 1 else []
    if mode == "tailk":
        out = []
        for k in (2, 3, 4, 5):
            g = gen_total - k
            if g >= 1:
                out.append(g)
        return sorted(set(out))
    if mode == "step":
        return list(range(1, gen_total, max(1, step)))

    # Hypothesis mode: compare tail-2 vs tail-1 parity (e.g., GEN_TOTAL=12 => resume 10 vs 11)
    if mode == "hyp_tail2_parity":
        out = []
        for g in (gen_total - 2, gen_total - 1):
            if g >= 1 and g < gen_total:
                out.append(g)
        return sorted(set(out))

    # Hypothesis mode: stride-aligned resume points plus their immediate neighbor (+1).
    # This rapidly tests whether failures correlate with resume_gen % policy_tick_stride == 0.
    if mode == "hyp_stride_adjacent":
        stride = max(1, int(policy_tick_stride))
        out = []
        lo = 1
        hi = gen_total - 1
        if int(tail_window) > 0:
            lo = max(lo, gen_total - int(tail_window))
        for g in range(lo, hi + 1):
            if g % stride == 0:
                out.append(g)
                if (g + 1) < gen_total:
                    out.append(g + 1)
        return sorted(set([x for x in out if 1 <= x < gen_total]))

    raise ValueError(f"Unknown resume mode: {mode}")


def _run_one_case(
    *,
    repo_root: Path,
    sweep_root: Path,
    seed: int,
    pop: int,
    device: str,
    bank_device: str,
    period: int,
    gen_total: int,
    resume_gen: int,
    keep_zips: str,
    extra_args: List[str],
) -> CaseResult:
    case_id = f"G{gen_total:04d}_R{resume_gen:04d}"
    case_dir = sweep_root / case_id
    runs_dir = case_dir / "runs"
    zips_dir = case_dir / "zips"
    logs_dir = case_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    zips_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Force isolation per case.
    env = os.environ.copy()
    env["SEED"] = str(seed)
    env["POP"] = str(pop)
    env["GEN_TOTAL"] = str(gen_total)
    env["PERIOD"] = str(period)
    env["DEVICE"] = str(device)
    env["BANK_DEVICE"] = str(bank_device)

    # Ensure the bash harness writes into the isolated directories
    env["RUNS_DIR"] = str(runs_dir)
    env["ZIP_OUT_DIR"] = str(zips_dir)

    # Do not let the harness wipe shared runs/ unintentionally.
    env["CLEAN_RUNS"] = "0"

    harness = repo_root / "tests" / "determinism" / "harnesses" / "checkpoint_resume_determinism_e2e.sh"
    if not harness.exists():
        raise FileNotFoundError(f"Missing harness: {harness}")

    cmd = [
        "bash",
        str(harness),
        "--membership-trace",
        "--membership-keep",
        "64",
        "--membership-harness",
        "--resume-gen",
        str(resume_gen),
        "--resume-gens",
        str(resume_gen),
    ]
    if extra_args:
        cmd += ["--"] + extra_args

    log_path = logs_dir / f"{case_id}__{_now_vienna_stamp()}.log"

    t0 = time.time()
    # Stream output to console AND to file (tee-like).
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"[SWEEP] case_id={case_id} gen_total={gen_total} resume_gen={resume_gen}\n")
        lf.write(f"[SWEEP] cmd={' '.join(cmd)}\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        proc.wait()

    dt = time.time() - t0
    out_txt = log_path.read_text(encoding="utf-8", errors="replace")

    run_a = ""
    run_b = ""
    zip_path = ""

    m = RUN_A_RE.search(out_txt)
    if m:
        run_a = m.group("path").strip()

    # The harness can run multiple Bs; but we run single resume per case.
    m = RUN_B_RE.search(out_txt)
    if m:
        run_b = m.group("path").strip()

    m = ARCHIVE_RE.search(out_txt)
    if m:
        zip_path = m.group("path").strip()

    exit_code = int(proc.returncode or 0)
    status = "PASS" if exit_code == 0 else "FAIL"

    # Cleanup policy for PASS zips to keep disk usage sane.
    if zip_path and keep_zips in ("failures", "none"):
        try:
            zp = Path(zip_path)
            if zp.exists():
                if keep_zips == "none" or (keep_zips == "failures" and status == "PASS"):
                    zp.unlink(missing_ok=True)
                    zip_path = ""
        except Exception:
            pass

    return CaseResult(
        case_id=case_id,
        gen_total=int(gen_total),
        resume_gen=int(resume_gen),
        exit_code=exit_code,
        status=status,
        seconds=float(dt),
        run_a=run_a,
        run_b=run_b,
        zip_path=zip_path,
        log_path=str(log_path),
    )



def _prune_checkpoint_bundles_v2(root: Path) -> int:
    """Delete all checkpoint_bundles_v2 directories under root.

    Returns the number of directories removed. This must not raise.
    """
    removed = 0
    try:
        for p in root.rglob("checkpoint_bundles_v2"):
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    removed += 1
            except Exception:
                continue
    except Exception:
        return removed
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep GEN_TOTAL x resume_gen determinism matrix.")
    ap.add_argument("--gen-min", type=int, default=5)
    ap.add_argument("--gen-max", type=int, default=30)
    ap.add_argument("--resume-mode", type=str, default="all", choices=["all", "tail2", "tailk", "step", "hyp_tail2_parity", "hyp_stride_adjacent"])
    ap.add_argument("--resume-step", type=int, default=1, help="Used only when --resume-mode=step")

    ap.add_argument("--policy-tick-stride", type=int, default=int(os.environ.get("POLICY_TICK_STRIDE", "2") or "2"),
                    help="Policy tick stride (used by hypothesis modes). Default from env POLICY_TICK_STRIDE or 2.")
    ap.add_argument("--tail-window", type=int, default=int(os.environ.get("SWEEP_TAIL_WINDOW", "0") or "0"),
                    help="If >0, hypothesis modes only test resume_gen >= gen_total - tail_window.")
    ap.add_argument("--max-cases", type=int, default=int(os.environ.get("SWEEP_MAX_CASES", "0") or "0"),
                    help="If >0, stop after this many cases (for quick hypothesis checks).")

    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42") or "42"))
    ap.add_argument("--pop", type=int, default=int(os.environ.get("POP", "32") or "32"))
    ap.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cuda") or "cuda")
    ap.add_argument("--bank-device", type=str, default=os.environ.get("BANK_DEVICE", "cuda") or "cuda")
    ap.add_argument("--period", type=int, default=int(os.environ.get("PERIOD", "20") or "20"))
    ap.add_argument("--keep-zips", type=str, default="failures", choices=["all", "failures", "none"])
    ap.add_argument("--delete-checkpoints-if-pass", action="store_true", help="If set: PASS cases are deleted entirely; FAIL cases keep case dir but prune checkpoint_bundles_v2 inside runs.")
    ap.add_argument("--out-dir", type=str, default=os.environ.get("SWEEP_ROOT_DIR", "") or "")
    ap.add_argument("--extra-args", type=str, default="", help="Extra args forwarded to engine.main (after --).")
    args = ap.parse_args()

    repo_root = _resolve_repo_root()

    stamp = _now_vienna_stamp()
    sweep_root = Path(args.out_dir) if args.out_dir else (repo_root / "runs" / f"sweep_{stamp}")
    sweep_root.mkdir(parents=True, exist_ok=True)

    extra_args = [x for x in args.extra_args.split() if x.strip()] if args.extra_args else []

    # Build case list
    cases: List[Tuple[int, int]] = []
    for gen_total in range(int(args.gen_min), int(args.gen_max) + 1):
        for resume_gen in _compute_resume_list(gen_total, args.resume_mode, args.resume_step, policy_tick_stride=int(args.policy_tick_stride), tail_window=int(args.tail_window)):
            if resume_gen >= gen_total:
                continue
            cases.append((gen_total, resume_gen))

    if int(args.max_cases) > 0:
        cases = cases[: int(args.max_cases)]

    if not cases:
        print("ERROR: no cases to run")
        return 2

    results_csv = sweep_root / f"sweep_matrix__{stamp}.csv"
    results_json = sweep_root / f"sweep_matrix__{stamp}.json"
    print(f"[SWEEP] root={sweep_root}")
    print(f"[SWEEP] cases={len(cases)} gen=[{args.gen_min}..{args.gen_max}] resume_mode={args.resume_mode} keep_zips={args.keep_zips} stride={args.policy_tick_stride} tail_window={args.tail_window} max_cases={args.max_cases}")

    rows: List[Dict[str, str]] = []
    t0 = time.time()
    fail_count = 0

    for idx, (gen_total, resume_gen) in enumerate(cases, start=1):
        print("")
        print(f"[SWEEP] ===== Case {idx}/{len(cases)}: GEN_TOTAL={gen_total} resume_gen={resume_gen} =====")
        case_t0 = time.time()
        try:
            res = _run_one_case(
                repo_root=repo_root,
                sweep_root=sweep_root,
                seed=int(args.seed),
                pop=int(args.pop),
                device=str(args.device),
                bank_device=str(args.bank_device),
                period=int(args.period),
                gen_total=int(gen_total),
                resume_gen=int(resume_gen),
                keep_zips=str(args.keep_zips),
                extra_args=extra_args,
            )
        except Exception as e:
            # Hard error running harness. Record as FAIL.
            dt = time.time() - case_t0
            res = CaseResult(
                case_id=f"G{gen_total:04d}_R{resume_gen:04d}",
                gen_total=int(gen_total),
                resume_gen=int(resume_gen),
                exit_code=99,
                status="ERROR",
                seconds=float(dt),
                run_a="",
                run_b="",
                zip_path="",
                log_path=str(sweep_root / "logs" / f"ERROR_{gen_total}_{resume_gen}.txt"),
            )
            Path(res.log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(res.log_path).write_text(str(e), encoding="utf-8")

        if res.status != "PASS":
            fail_count += 1

        rows.append(
            {
                "case_id": res.case_id,
                "gen_total": str(res.gen_total),
                "resume_gen": str(res.resume_gen),
                "status": res.status,
                "exit_code": str(res.exit_code),
                "seconds": f"{res.seconds:.2f}",
                "zip_path": res.zip_path,
                "log_path": res.log_path,
                "run_a": res.run_a,
                "run_b": res.run_b,
            }
        )

        # Write incremental results so you never lose progress.
        with results_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        elapsed = time.time() - t0
        print(f"[SWEEP] case_result={res.status} exit={res.exit_code} time={res.seconds:.2f}s (elapsed_total={elapsed/60.0:.1f}m fails={fail_count})")
        # Optional cleanup to control disk usage.
        if bool(getattr(args, "delete_checkpoints_if_pass", False)):
            case_dir = sweep_root / res.case_id
            if res.status == "PASS":
                try:
                    shutil.rmtree(case_dir, ignore_errors=True)
                    print(f"[SWEEP] cleanup: PASS -> deleted case_dir={case_dir}")
                except Exception:
                    pass
            else:
                try:
                    removed = _prune_checkpoint_bundles_v2(case_dir)
                    if removed > 0:
                        print(f"[SWEEP] cleanup: FAIL -> pruned {removed} checkpoint_bundles_v2 dirs under {case_dir}")
                except Exception:
                    pass


    payload = {
        "schema_version": "sweep_matrix_v1",
        "created_at": stamp,
        "root_dir": str(sweep_root),
        "seed": int(args.seed),
        "pop": int(args.pop),
        "device": str(args.device),
        "bank_device": str(args.bank_device),
        "period": int(args.period),
        "gen_min": int(args.gen_min),
        "gen_max": int(args.gen_max),
        "resume_mode": str(args.resume_mode),
        "resume_step": int(args.resume_step),
        "policy_tick_stride": int(args.policy_tick_stride),
        "tail_window": int(args.tail_window),
        "max_cases": int(args.max_cases),
        "keep_zips": str(args.keep_zips),
        "cases_total": len(cases),
        "fails": fail_count,
        "results_csv": str(results_csv),
        "results_rows": rows,
    }
    results_json.write_text(__import__("json").dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("")
    print(f"[SWEEP] DONE. csv={results_csv} json={results_json} fails={fail_count}/{len(cases)}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
