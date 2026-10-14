"""tests/harnesses/phase4_functional/phase4_functional_determinism_harness.py

Phase 4 Functional Determinism Harness (EIEL)

Mission:
- CI-grade unattended verification of functional determinism across:
  * Mode 0: single checkpoint/resume.
  * Mode 1: many checkpoints, many resumes.
  * Mode 2: checkpoint chain / relay race.

Hard constraints:
- Fail fast (non-zero) on first mismatch.
- Always write phase4_determinism_summary.json and zip artifacts to ZIP_OUT_DIR.
- Determinism-critical allowlist comparisons ONLY:
    metrics.csv, behaviors.csv, determinism_fingerprint.json, equivalence verifier PASS.
- Verify checkpoint bundle v2 integrity before any resume.
- No interactive prompts.

This harness intentionally reuses the simulator CLI (python -m engine.main)
and the Phase 3 equivalence verifier (verify_checkpoint_bundle_v2) without
changing simulator code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tests.harnesses.phase4_functional.verify.phase4_compute_determinism_fingerprint import (
    compute_phase4_determinism_fingerprint,
)


@dataclass(frozen=True)
class RunSpec:
    """A single simulator invocation specification."""

    label: str
    run_id: str
    start_gen: int
    generations: int
    mode: str
    resume_from_bundle: Optional[Path]


@dataclass(frozen=True)
class RunResult:
    """Outputs of a completed run."""

    label: str
    run_id: str
    run_dir: Path
    start_gen: int
    end_gen: int


def _sha256_bytes(data: bytes) -> str:
    """Return SHA256 hex digest of data."""
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return SHA256 hex digest of file at path (streamed)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_canonical_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write canonical JSON (sorted keys, minified, newline-terminated)."""
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _safe_relpath(path: Path, base: Path) -> str:
    """Return a stable relative path (used for summary).

    If path is not under base, falls back to basename.
    """
    try:
        return str(path.relative_to(base))
    except Exception:
        return path.name


def _parse_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    """Parse harness args and return (args, passthrough_to_sim).

    Everything after a standalone "--" is passed through to engine.main.
    """
    if "--" in argv:
        idx = list(argv).index("--")
        harness_argv = list(argv)[:idx]
        passthrough = list(argv)[idx + 1 :]
    else:
        harness_argv = list(argv)
        passthrough = []

    p = argparse.ArgumentParser(
        description="Phase 4 checkpoint functional determinism e2e harness",
        add_help=True,
    )
    p.add_argument("--mode", default="all", choices=["0", "1", "2", "all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--G", type=int, default=100)
    p.add_argument("--K", type=int, default=12)
    p.add_argument("--C", type=int, default=12)
    p.add_argument("--ci-fast", action="store_true")

    p.add_argument(
        "--debug",
        action="store_true",
        help="Print progress lines to stdout (useful for long CI runs).",
    )

    p.add_argument(
        "--debug_stream_sim_output",
        "--debug-stream-sim-output",
        action="store_true",
        help="When --debug is set, also stream simulator stdout/stderr to console (still writes log).",
    )

    p.add_argument(
        "--delete_checkpoint_if_pass",
        "--delete-checkpoint-if-pass",
        action="store_true",
        help="Delete checkpoint bundles before zipping artifacts when the harness PASSes.",
    )

    # Legacy compatibility args (accepted no-op or passthrough).
    p.add_argument("--membership-trace", action="store_true")
    p.add_argument("--membership-keep", type=int, default=0)
    p.add_argument("--membership-harness", action="store_true")
    p.add_argument("--start_zero_checkpoint", action="store_true")

    args = p.parse_args(harness_argv)
    return args, passthrough


def _apply_ci_fast_defaults(args: argparse.Namespace, raw_argv: Sequence[str]) -> None:
    """Apply --ci-fast defaults unless explicitly overridden."""
    if not args.ci_fast:
        return

    # Only override if the flag wasn't explicitly provided.
    if "--G" not in raw_argv:
        args.G = 20
    if "--K" not in raw_argv:
        args.K = 5
    if "--C" not in raw_argv:
        args.C = 5


def _require_positive(name: str, value: int) -> None:
    """Validate positive integers for harness config."""
    if int(value) <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _verify_checkpoint_bundle_v2_integrity(bundle_dir: Path) -> None:
    """Fail-fast integrity check for a checkpoint bundle v2 directory.

    Required files (non-negotiable):
      - metadata.json
      - world.json
      - population.json
      - rng_snapshot_v2.json
      - control_plane.json
    """
    required = [
        "metadata.json",
        "world.json",
        "population.json",
        "rng_snapshot_v2.json",
        "control_plane.json",
    ]
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        raise FileNotFoundError(f"checkpoint bundle dir missing: {bundle_dir}")
    missing = [name for name in required if not (bundle_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"checkpoint bundle incomplete: {bundle_dir} missing {','.join(missing)}"
        )


def _bundle_path(run_dir: Path, gen: int) -> Path:
    """Canonical checkpoint bundle v2 path for a given generation."""
    return run_dir / "checkpoint_bundles_v2" / f"bundle_gen{gen:06d}"


def _extract_run_dir_from_stdout(stdout: str) -> Optional[Path]:
    """Extract run_dir from engine.main stdout."""
    # main prints: "Run complete. Metrics written to: <path>/metrics.csv"
    needle = "Metrics written to:"
    for line in stdout.splitlines():
        if needle in line:
            tail = line.split(needle, 1)[1].strip()
            if tail.endswith("/metrics.csv") or tail.endswith("\\metrics.csv"):
                metrics_path = Path(tail)
                return metrics_path.parent
    return None



def _summarize_failure_output(stdout: str, max_chars: int = 280) -> str:
    """Return a single-line hint from simulator stdout for error reporting.

    Prefers the last non-empty line, or an ERROR line, or an exception line.
    Output is stripped and truncated to max_chars.
    """
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    # Prefer explicit ERROR-like lines near the end.
    for ln in reversed(lines[-200:]):
        if ln.startswith("ERROR") or ln.startswith("Exception") or ln.startswith("Traceback"):
            hint = ln
            break
    else:
        hint = lines[-1]
    hint = " ".join(hint.split())
    if len(hint) > max_chars:
        hint = hint[: max_chars - 3] + "..."
    return hint

def _run_simulator(
    *,
    repo_root: Path,
    runs_root: Path,
    run_id: str,
    seed: int,
    generations: int,
    checkpoint_bundle_period: int,
    enable_start_zero_checkpoint: bool,
    resume_from: Optional[Path],
    passthrough_args: Sequence[str],
    log_path: Path,
    debug_stream: bool,
) -> Path:
    """Invoke engine.main and return the run_dir.

    All stdout/stderr are captured to log_path; only minimal parsing is done.
    """
    sim_pkg_init = repo_root / "engine" / "__init__.py"
    sim_main_py = repo_root / "engine" / "main.py"
    if sim_pkg_init.exists():
        sim_entry: List[str] = [sys.executable, "-m", "engine.main"]
    else:
        sim_entry = [sys.executable, str(sim_main_py)]

    cmd: List[str] = sim_entry + [
        "--seed",
        str(seed),
        "--gen",
        str(generations),
        "--runs_dir",
        str(runs_root),
        "--run_id",
        run_id,
        "--device",
        "auto",
        "--bank_device",
        "auto",
        "--knob",
        f"sim_cfg.checkpoint_bundle_v2_period_gen={int(checkpoint_bundle_period)}",
    ]

    if resume_from is not None:
        cmd += ["--resume_from", str(resume_from)]

    # Pass-through to simulator CLI (backwards compatibility).
    cmd += list(passthrough_args)

    env = os.environ.copy()

    # Make repo imports robust across invocation styles.
    # - 'simulation.*' imports expect engine/ to be on sys.path.
    # - 'engine.*' imports expect repo_root/ to be on sys.path.
    # We do this via PYTHONPATH so it works for both "-m" and "main.py" entrypoints.
    existing_pp = env.get("PYTHONPATH", "")
    pp_parts: List[str] = []
    pp_parts.append(str(repo_root))
    pp_parts.append(str(repo_root / "engine"))
    if existing_pp:
        pp_parts.extend([p for p in existing_pp.split(os.pathsep) if p])
    # De-dup while preserving order.
    seen: set = set()
    pp_unique: List[str] = []
    for p in pp_parts:
        if p not in seen:
            seen.add(p)
            pp_unique.append(p)
    env["PYTHONPATH"] = os.pathsep.join(pp_unique)

    # Ensure gen0 bundle exists when requested.
    if enable_start_zero_checkpoint:
        env["EIEL_CHECKPOINT_BUNDLE_V2_AT_START"] = "1"

    # Always run unattended: force no prompts via env guard (if any) and keep logs.
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if debug_stream:
        # Stream stdout to both console and log for progress visibility.
        log_path.open("w", encoding="utf-8").close()
        with log_path.open("a", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                # Only stream when explicitly requested.
                print(line, end="", flush=True)
            proc.wait()
            stdout_text = log_path.read_text(encoding="utf-8")
    else:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        stdout_text = proc.stdout
        log_path.write_text(stdout_text, encoding="utf-8")

    # Optional progress visibility
    # (Keep to a single line to avoid CI log spam.)
    # NOTE: debug printing is handled by caller; here we only expose the log path in errors.

    if proc.returncode != 0:
        hint = _summarize_failure_output(stdout_text)
        msg = f"simulator failed (rc={proc.returncode})"
        if hint:
            msg += f" hint={hint}"
        msg += f" log={log_path}"
        raise RuntimeError(msg)

    run_dir = _extract_run_dir_from_stdout(stdout_text)
    if run_dir is None:
        raise RuntimeError("unable to locate run_dir from simulator output")

    if not run_dir.exists():
        raise RuntimeError(f"reported run_dir does not exist: {run_dir}")

    return run_dir


def _backup_file(path: Path, suffix: str) -> None:
    """Create a byte-for-byte backup copy of path (non-destructive)."""
    if not path.exists() or not path.is_file():
        return
    backup = path.with_name(path.name + suffix)
    if backup.exists():
        return
    shutil.copy2(path, backup)


def _stitch_csv_prefix(
    *,
    ref_csv: Path,
    partial_csv: Path,
    start_gen: int,
    out_csv: Path,
) -> None:
    """Stitch a "partial" CSV (starting at start_gen) with the prefix from ref.

    This operates on raw CSV lines to preserve byte-level stability.
    Assumptions:
    - generation is the first column.
    - both files share the same header line.
    """
    ref_lines = ref_csv.read_bytes().splitlines(keepends=True)
    part_lines = partial_csv.read_bytes().splitlines(keepends=True)

    if not ref_lines:
        raise RuntimeError(f"empty reference csv: {ref_csv}")
    if not part_lines:
        raise RuntimeError(f"empty partial csv: {partial_csv}")

    ref_header = ref_lines[0]
    part_header = part_lines[0]
    if ref_header != part_header:
        raise RuntimeError("CSV header mismatch during stitch")

    if start_gen <= 0:
        # Nothing to stitch.
        out_csv.write_bytes(partial_csv.read_bytes())
        return

    stitched: List[bytes] = [ref_header]

    # Prefix from ref: all rows with generation < start_gen.
    for line in ref_lines[1:]:
        # Skip blank lines.
        stripped = line.strip()
        if not stripped:
            continue
        gen_bytes = stripped.split(b",", 1)[0]
        try:
            gen = int(gen_bytes)
        except Exception:
            raise RuntimeError("failed to parse generation column in reference csv")
        if gen < start_gen:
            stitched.append(line)
        else:
            break

    # Suffix from partial: all rows (excluding header).
    stitched.extend(part_lines[1:])
    out_csv.write_bytes(b"".join(stitched))


def _stitch_run_outputs(
    *,
    ref_run_dir: Path,
    resume_run_dir: Path,
    resume_start_gen: int,
) -> None:
    """Stitch metrics.csv and behaviors.csv in a resumed run to be full-length.

    This copies prefix rows (gen < resume_start_gen) from the reference run into
    the resumed run outputs, enabling byte-identical comparisons.

    Raw outputs are preserved as *.raw backups.
    """
    if resume_start_gen <= 0:
        return

    ref_metrics = ref_run_dir / "metrics.csv"
    ref_behaviors = ref_run_dir / "behaviors.csv"
    tgt_metrics = resume_run_dir / "metrics.csv"
    tgt_behaviors = resume_run_dir / "behaviors.csv"

    for p in (ref_metrics, ref_behaviors, tgt_metrics, tgt_behaviors):
        if not p.exists():
            raise FileNotFoundError(f"required csv missing for stitch: {p}")

    # Backup original partial outputs before overwriting.
    _backup_file(tgt_metrics, ".raw")
    _backup_file(tgt_behaviors, ".raw")

    _stitch_csv_prefix(
        ref_csv=ref_metrics,
        partial_csv=tgt_metrics,
        start_gen=resume_start_gen,
        out_csv=tgt_metrics,
    )
    _stitch_csv_prefix(
        ref_csv=ref_behaviors,
        partial_csv=tgt_behaviors,
        start_gen=resume_start_gen,
        out_csv=tgt_behaviors,
    )


def _aggregate_chain_csv(
    *,
    csv_paths_in_order: List[Path],
    out_path: Path,
) -> None:
    """Concatenate chain CSV segments (same header) into a single CSV."""
    if not csv_paths_in_order:
        raise ValueError("csv_paths_in_order must not be empty")

    first = csv_paths_in_order[0]
    data = first.read_bytes().splitlines(keepends=True)
    if not data:
        raise RuntimeError(f"empty csv: {first}")
    header = data[0]
    out_lines: List[bytes] = [header]
    out_lines.extend(data[1:])

    for p in csv_paths_in_order[1:]:
        lines = p.read_bytes().splitlines(keepends=True)
        if not lines:
            raise RuntimeError(f"empty csv: {p}")
        if lines[0] != header:
            raise RuntimeError("chain csv header mismatch")
        out_lines.extend(lines[1:])

    out_path.write_bytes(b"".join(out_lines))


def _finalize_chain_outputs(
    *,
    leg_run_dirs: List[Path],
    final_run_dir: Path,
) -> None:
    """Replace final_run_dir metrics/behaviors with concatenated chain outputs.

    Raw final leg outputs are preserved as *.raw backups.
    """
    if not leg_run_dirs:
        raise ValueError("leg_run_dirs must not be empty")

    metrics_paths = [d / "metrics.csv" for d in leg_run_dirs]
    behaviors_paths = [d / "behaviors.csv" for d in leg_run_dirs]
    for p in metrics_paths + behaviors_paths:
        if not p.exists():
            raise FileNotFoundError(f"missing chain csv: {p}")

    final_metrics = final_run_dir / "metrics.csv"
    final_behaviors = final_run_dir / "behaviors.csv"
    _backup_file(final_metrics, ".raw")
    _backup_file(final_behaviors, ".raw")

    _aggregate_chain_csv(csv_paths_in_order=metrics_paths, out_path=final_metrics)
    _aggregate_chain_csv(csv_paths_in_order=behaviors_paths, out_path=final_behaviors)


def _run_equivalence_verifier(
    *,
    run_a: Path,
    run_b: Path,
    log_path: Path,
) -> bool:
    """Run Phase 3 equivalence verifier, capture output, return PASS/FAIL."""
    cmd = [
        sys.executable,
        "-m",
        "tests.harnesses.phase3_resume.verify.verify_checkpoint_bundle_v2",
        "--run_a",
        str(run_a),
        "--run_b",
        str(run_b),
        "--compare_mode",
        "final_gen",
        "--compare_metrics",
        "--compare_behaviors",
        "--strict",
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode == 0


def _compare_allowlist(
    *,
    run_a: Path,
    run_b: Path,
    verifier_log: Path,
    summary_base: Path,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Compare determinism-critical allowlist artifacts."""
    files = ["metrics.csv", "behaviors.csv", "determinism_fingerprint.json"]
    for name in files:
        a_path = run_a / name
        b_path = run_b / name
        if not a_path.exists() or not b_path.exists():
            return (
                False,
                {
                    "kind": "missing_output",
                    "file": name,
                },
            )
        a_hash = _sha256_file(a_path)
        b_hash = _sha256_file(b_path)
        if a_hash != b_hash:
            return (
                False,
                {
                    "kind": "file_mismatch",
                    "file": name,
                    "sha256_a": a_hash,
                    "sha256_b": b_hash,
                },
            )

    if not _run_equivalence_verifier(run_a=run_a, run_b=run_b, log_path=verifier_log):
        return (
            False,
            {
                "kind": "equivalence_verifier_failed",
                "file": "equivalence_verifier",
                "verifier_log": _safe_relpath(verifier_log, summary_base),
            },
        )

    return True, None



def _format_mismatch_error_line(
    *,
    mode: str,
    detail: Dict[str, Any],
    run_a_label: str,
    run_b_label: str,
    resume_gen: Optional[int] = None,
) -> str:
    """Format a single-line actionable ERROR for the first mismatch."""
    kind = str(detail.get("kind", "unknown"))
    file = str(detail.get("file", ""))
    sha_a = str(detail.get("sha256_a", ""))
    sha_b = str(detail.get("sha256_b", ""))
    parts: List[str] = [
        "ERROR: determinism mismatch",
        f"mode={mode}",
        f"kind={kind}",
        f"file={file}",
    ]
    if resume_gen is not None:
        parts.append(f"resume_gen={int(resume_gen)}")
    if sha_a:
        parts.append(f"sha256_a={sha_a}")
    if sha_b:
        parts.append(f"sha256_b={sha_b}")
    parts.append(f"run_a={run_a_label}")
    parts.append(f"run_b={run_b_label}")
    if kind == "equivalence_verifier_failed":
        v = detail.get("verifier_log")
        if v:
            parts.append(f"verifier_log={v}")
    return " ".join(parts)



def _zip_artifacts(*, zip_path: Path, items: List[Path], base_dir: Path) -> None:
    """Create a zip containing items, storing paths relative to base_dir."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            if not item.exists():
                continue
            if item.is_dir():
                for p in sorted(item.rglob("*")):
                    if p.is_file():
                        arcname = _safe_relpath(p, base_dir)
                        zf.write(p, arcname)
            else:
                arcname = _safe_relpath(item, base_dir)
                zf.write(item, arcname)


def _git_commit_hash(repo_root: Path) -> Optional[str]:
    """Return git commit hash if available."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            s = proc.stdout.strip()
            if s:
                return s
    except Exception:
        return None
    return None


def _resolve_zip_out_dir_and_clean_runs(*, debug: bool) -> Tuple[Path, bool, bool]:
    """Resolve ZIP_OUT_DIR and CLEAN_RUNS with a small robustness shim.

    Some shells allow mistakes like:
      CLEAN_RUNS=1ZIP_OUT_DIR=./runs/phase4 python ...
    which results in:
      CLEAN_RUNS='1ZIP_OUT_DIR=./runs/phase4' and ZIP_OUT_DIR unset.

    This function attempts to recover that specific case without being clever.
    Returns (zip_out_dir, clean_runs, recovered_from_concat).
    """

    zip_out_dir_env = os.environ.get("ZIP_OUT_DIR")
    clean_runs_env = os.environ.get("CLEAN_RUNS", "0")
    recovered = False

    if (zip_out_dir_env is None or str(zip_out_dir_env).strip() == "") and "ZIP_OUT_DIR=" in str(
        clean_runs_env
    ):
        # Salvage the common typo: CLEAN_RUNS=1ZIP_OUT_DIR=...
        parts = str(clean_runs_env).split("ZIP_OUT_DIR=", 1)
        prefix = parts[0]
        zip_out_dir_env = parts[1].strip() if len(parts) > 1 else None
        clean_runs_env = "1" if str(prefix).strip().startswith("1") else "0"
        recovered = True

    zip_out_dir = Path((zip_out_dir_env or "./runs")).resolve()
    clean_runs = str(clean_runs_env) == "1"

    if recovered and debug:
        print(
            "PHASE4: recovered ZIP_OUT_DIR from concatenated CLEAN_RUNS env var (missing space)",
            flush=True,
        )

    return zip_out_dir, clean_runs, recovered


def _delete_checkpoint_bundles_v2_under_runs_root(runs_root: Path) -> int:
    """Delete checkpoint_bundles_v2 directories under each run folder.

    Returns the number of run folders cleaned.
    """
    cleaned = 0
    if not runs_root.exists() or not runs_root.is_dir():
        return 0

    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        ckpt_dir = child / "checkpoint_bundles_v2"
        if ckpt_dir.exists() and ckpt_dir.is_dir():
            shutil.rmtree(ckpt_dir)
            cleaned += 1

    return cleaned


def _build_mode0_specs(*, seed: int, G: int, C: int) -> List[RunSpec]:
    """Build Mode 0 run specs."""
    return [
        RunSpec(
            label="m0_A",
            run_id="phase4_m0_A",
            start_gen=0,
            generations=G,
            mode="0",
            resume_from_bundle=None,
        ),
        RunSpec(
            label="m0_B",
            run_id=f"phase4_m0_B_from_{C:06d}",
            start_gen=C,
            generations=max(0, G - C),
            mode="0",
            resume_from_bundle=Path("__RESUME_FROM_MODE0__"),
        ),
    ]


def _build_mode1_specs(*, G: int, K: int) -> List[RunSpec]:
    """Build Mode 1 run specs."""
    specs: List[RunSpec] = [
        RunSpec(
            label="m1_A",
            run_id="phase4_m1_A",
            start_gen=0,
            generations=G,
            mode="1",
            resume_from_bundle=None,
        )
    ]
    for g in range(0, G + 1, K):
        specs.append(
            RunSpec(
                label=f"m1_R_gen{g:06d}",
                run_id=f"phase4_m1_R_gen{g:06d}",
                start_gen=g,
                generations=max(0, G - g),
                mode="1",
                resume_from_bundle=Path("__RESUME_FROM_MODE1__"),
            )
        )
    return specs


def _build_mode2_specs(*, G: int, K: int) -> List[RunSpec]:
    """Build Mode 2 run specs (reference + chain legs)."""
    specs: List[RunSpec] = [
        RunSpec(
            label="m2_A_ref",
            run_id="phase4_m2_A_ref",
            start_gen=0,
            generations=G,
            mode="2",
            resume_from_bundle=None,
        )
    ]

    start = 0
    leg = 0
    while start < G:
        end = min(start + K, G)
        gens = end - start
        resume_marker = None if leg == 0 else Path("__RESUME_FROM_MODE2__")
        specs.append(
            RunSpec(
                label=f"m2_C_leg{leg:02d}",
                run_id=f"phase4_m2_C_leg{leg:02d}_gen{start:06d}_to{end:06d}",
                start_gen=start,
                generations=gens,
                mode="2",
                resume_from_bundle=resume_marker,
            )
        )
        start = end
        leg += 1

    return specs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entrypoint."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args, passthrough = _parse_args(raw_argv)
    _apply_ci_fast_defaults(args, raw_argv)

    # Validate core config.
    _require_positive("seed", args.seed)
    _require_positive("G", args.G)
    _require_positive("K", args.K)
    _require_positive("C", args.C)
    if args.C > args.G:
        raise ValueError(f"C must be <= G (C={args.C}, G={args.G})")

    zip_out_dir, clean_runs, _recovered_env = _resolve_zip_out_dir_and_clean_runs(
        debug=bool(args.debug)
    )
    runs_root = zip_out_dir / "phase4_functional_determinism_harness"
    runs_root.mkdir(parents=True, exist_ok=True)

    if clean_runs:
        # Only delete run folders under our dedicated Phase4 root.
        if runs_root.exists():
            shutil.rmtree(runs_root)
        runs_root.mkdir(parents=True, exist_ok=True)

    # Resolve repo_root robustly.
    # Primary: explicit env var (useful in CI and avoids accidental site-packages shadowing).
    env_repo_root = os.environ.get("EIEL_REPO_ROOT", "").strip()
    if env_repo_root:
        repo_root = Path(env_repo_root).expanduser().resolve()
    else:
        # Preferred: the current working directory where the harness was invoked.
        # This matches typical usage: run from repo root.
        repo_root = Path.cwd().resolve()
        # Fallback: 3 parents up from this file (works when invoked as module from repo).
        if not (repo_root / "engine").exists():
            repo_root = Path(__file__).resolve().parents[3]

    if not (repo_root / "engine").exists():
        raise RuntimeError(f"invalid repo_root (missing engine): {repo_root}")

    determinism_strict = bool(int(os.environ.get("DEBUG_DETERMINISM", "1")))

    modes_requested = args.mode
    modes_to_run: List[str]
    if args.mode == "all":
        modes_to_run = ["0", "1", "2"]
    else:
        modes_to_run = [args.mode]

    summary_path = runs_root / "phase4_determinism_summary.json"
    repo_commit = _git_commit_hash(repo_root)

    summary: Dict[str, Any] = {
        "seed": int(args.seed),
        "G": int(args.G),
        "K": int(args.K),
        "C": int(args.C),
        "modes_requested": str(modes_requested),
        "modes_executed": [],
        "determinism_strict": bool(determinism_strict),
        "repo": {"git_commit": repo_commit},
        "runs_root": str(runs_root.name),
        "runs": {},
        "comparisons": [],
        "result": "FAIL",
        "first_failure": None,
    }

    created_items: List[Path] = [runs_root]

    def record_run(res: RunResult) -> None:
        summary["runs"][res.label] = {
            "run_id": res.run_id,
            "run_dir": _safe_relpath(res.run_dir, runs_root),
            "start_gen": int(res.start_gen),
            "end_gen": int(res.end_gen),
        }

    def record_compare(
        *,
        mode: str,
        label: str,
        run_a_label: str,
        run_b_label: str,
        passed: bool,
        detail: Optional[Dict[str, Any]],
    ) -> None:
        entry: Dict[str, Any] = {
            "mode": str(mode),
            "label": str(label),
            "run_a": str(run_a_label),
            "run_b": str(run_b_label),
            "pass": bool(passed),
        }
        if detail is not None:
            entry["detail"] = detail
        summary["comparisons"].append(entry)

    failure_error_line: Optional[str] = None

    def _one_line(msg: str) -> str:
        return " ".join(str(msg).splitlines()).strip()

    try:
        if bool(args.debug):
            print(
                "PHASE4: start "
                f"mode={modes_requested} seed={args.seed} G={args.G} K={args.K} C={args.C} "
                f"zip_out_dir={zip_out_dir} clean_runs={int(clean_runs)} "
                f"delete_ckpt_if_pass={int(bool(args.delete_checkpoint_if_pass))}",
                flush=True,
            )

        for mode in modes_to_run:
            summary["modes_executed"].append(int(mode))

            if bool(args.debug):
                print(f"PHASE4: mode {mode} begin", flush=True)

            if mode == "0":
                specs = _build_mode0_specs(seed=args.seed, G=args.G, C=args.C)
                # Mode 0 requires a checkpoint at gen=C. Use period=C to guarantee emission.
                period = max(1, int(args.C))
                enable_zero = bool(args.start_zero_checkpoint)

                # Run A
                log_a = runs_root / "logs" / f"{specs[0].run_id}.log"
                if bool(args.debug):
                    print(
                        f"PHASE4: run {specs[0].run_id} start_gen=0 gens={specs[0].generations}",
                        flush=True,
                    )
                run_a_dir = _run_simulator(
                    repo_root=repo_root,
                    runs_root=runs_root,
                    run_id=specs[0].run_id,
                    seed=args.seed,
                    generations=specs[0].generations,
                    checkpoint_bundle_period=period,
                    enable_start_zero_checkpoint=enable_zero,
                    resume_from=None,
                    passthrough_args=passthrough,
                    log_path=log_a,
                debug_stream=bool(args.debug) and bool(args.debug_stream_sim_output),
                )
                res_a = RunResult(
                    label=specs[0].label,
                    run_id=specs[0].run_id,
                    run_dir=run_a_dir,
                    start_gen=0,
                    end_gen=args.G,
                )
                record_run(res_a)

                # Verify checkpoint C exists before resume.
                bundle_c = _bundle_path(run_a_dir, args.C)
                try:
                    _verify_checkpoint_bundle_v2_integrity(bundle_c)
                except Exception as e:
                    summary["first_failure"] = {
                        "kind": "checkpoint_missing_or_invalid",
                        "gen": int(args.C),
                        "message": _one_line(str(e)),
                    }
                    failure_error_line = (
                        f"ERROR: missing/invalid checkpoint bundle at gen {args.C} (mode 0)"
                    )
                    raise

                # Run B (resume)
                log_b = runs_root / "logs" / f"{specs[1].run_id}.log"
                if bool(args.debug):
                    print(
                        f"PHASE4: run {specs[1].run_id} resume_gen={args.C} gens={specs[1].generations}",
                        flush=True,
                    )
                run_b_dir = _run_simulator(
                    repo_root=repo_root,
                    runs_root=runs_root,
                    run_id=specs[1].run_id,
                    seed=args.seed,
                    generations=specs[1].generations,
                    checkpoint_bundle_period=period,
                    enable_start_zero_checkpoint=False,
                    resume_from=bundle_c,
                    passthrough_args=passthrough,
                    log_path=log_b,
                debug_stream=bool(args.debug) and bool(args.debug_stream_sim_output),
                )
                res_b = RunResult(
                    label=specs[1].label,
                    run_id=specs[1].run_id,
                    run_dir=run_b_dir,
                    start_gen=args.C,
                    end_gen=args.G,
                )
                record_run(res_b)

                # Stitch outputs to full-length, then fingerprint both.
                _stitch_run_outputs(
                    ref_run_dir=run_a_dir,
                    resume_run_dir=run_b_dir,
                    resume_start_gen=args.C,
                )

                compute_phase4_determinism_fingerprint(
                    run_a_dir,
                    seed=args.seed,
                    G=args.G,
                    K=args.K,
                    C=args.C,
                    mode="0",
                    determinism_strict=determinism_strict,
                )
                compute_phase4_determinism_fingerprint(
                    run_b_dir,
                    seed=args.seed,
                    G=args.G,
                    K=args.K,
                    C=args.C,
                    mode="0",
                    determinism_strict=determinism_strict,
                )

                verifier_log = runs_root / "logs" / "m0_equivalence.log"
                ok, detail = _compare_allowlist(
                    run_a=run_a_dir,
                    run_b=run_b_dir,
                    verifier_log=verifier_log,
                    summary_base=runs_root,
                )
                record_compare(
                    mode="0",
                    label="m0_A_vs_m0_B",
                    run_a_label=res_a.label,
                    run_b_label=res_b.label,
                    passed=ok,
                    detail=detail,
                )
                if not ok:
                    summary["first_failure"] = detail
                    failure_error_line = _format_mismatch_error_line(mode="0", detail=detail or {}, run_a_label=res_a.label, run_b_label=res_b.label, resume_gen=args.C)
                    raise RuntimeError("mode0 mismatch")

                if bool(args.debug):
                    print("PHASE4: mode 0 PASS", flush=True)

            elif mode == "1":
                specs = _build_mode1_specs(G=args.G, K=args.K)
                # Mode 1 requires checkpoints every K generations.
                period = max(1, int(args.K))

                # Mode 1 requires gen0 checkpoint for g=0 resume.
                enable_zero = True

                # Run A
                log_a = runs_root / "logs" / f"{specs[0].run_id}.log"
                if bool(args.debug):
                    print(
                        f"PHASE4: run {specs[0].run_id} start_gen=0 gens={specs[0].generations}",
                        flush=True,
                    )
                run_a_dir = _run_simulator(
                    repo_root=repo_root,
                    runs_root=runs_root,
                    run_id=specs[0].run_id,
                    seed=args.seed,
                    generations=specs[0].generations,
                    checkpoint_bundle_period=period,
                    enable_start_zero_checkpoint=enable_zero,
                    resume_from=None,
                    passthrough_args=passthrough,
                    log_path=log_a,
                debug_stream=bool(args.debug) and bool(args.debug_stream_sim_output),
                )
                res_a = RunResult(
                    label=specs[0].label,
                    run_id=specs[0].run_id,
                    run_dir=run_a_dir,
                    start_gen=0,
                    end_gen=args.G,
                )
                record_run(res_a)
                compute_phase4_determinism_fingerprint(
                    run_a_dir,
                    seed=args.seed,
                    G=args.G,
                    K=args.K,
                    C=args.C,
                    mode="1",
                    determinism_strict=determinism_strict,
                )

                # Verify all required checkpoints exist before launching resumes.
                checkpoint_gens = [s.start_gen for s in specs[1:]]
                for g in checkpoint_gens:
                    bdir = _bundle_path(run_a_dir, g)
                    try:
                        _verify_checkpoint_bundle_v2_integrity(bdir)
                    except Exception as e:
                        summary["first_failure"] = {
                            "kind": "checkpoint_missing_or_invalid",
                            "gen": int(g),
                            "message": _one_line(str(e)),
                        }
                        failure_error_line = (
                            f"ERROR: missing/invalid checkpoint bundle at gen {g} (mode 1)"
                        )
                        raise

                # Run resumes and compare against A.
                for s in specs[1:]:
                    bdir = _bundle_path(run_a_dir, s.start_gen)
                    log_r = runs_root / "logs" / f"{s.run_id}.log"
                    if bool(args.debug):
                        print(
                            f"PHASE4: run {s.run_id} resume_gen={s.start_gen} gens={s.generations}",
                            flush=True,
                        )
                    run_r_dir = _run_simulator(
                        repo_root=repo_root,
                        runs_root=runs_root,
                        run_id=s.run_id,
                        seed=args.seed,
                        generations=s.generations,
                        checkpoint_bundle_period=period,
                        enable_start_zero_checkpoint=False,
                        resume_from=bdir,
                        passthrough_args=passthrough,
                        log_path=log_r,
                    debug_stream=bool(args.debug) and bool(args.debug_stream_sim_output),
                    )
                    res_r = RunResult(
                        label=s.label,
                        run_id=s.run_id,
                        run_dir=run_r_dir,
                        start_gen=s.start_gen,
                        end_gen=args.G,
                    )
                    record_run(res_r)

                    # Stitch if needed, then fingerprint.
                    _stitch_run_outputs(
                        ref_run_dir=run_a_dir,
                        resume_run_dir=run_r_dir,
                        resume_start_gen=s.start_gen,
                    )
                    compute_phase4_determinism_fingerprint(
                        run_r_dir,
                        seed=args.seed,
                        G=args.G,
                        K=args.K,
                        C=args.C,
                        mode="1",
                        determinism_strict=determinism_strict,
                    )

                    verifier_log = runs_root / "logs" / f"m1_equiv_{s.start_gen:06d}.log"
                    ok, detail = _compare_allowlist(
                        run_a=run_a_dir,
                        run_b=run_r_dir,
                        verifier_log=verifier_log,
                        summary_base=runs_root,
                    )
                    record_compare(
                        mode="1",
                        label=f"m1_A_vs_{s.label}",
                        run_a_label=res_a.label,
                        run_b_label=res_r.label,
                        passed=ok,
                        detail=detail,
                    )
                    if not ok:
                        summary["first_failure"] = detail
                        failure_error_line = (
                            f"ERROR: determinism mismatch in mode 1 at resume gen {s.start_gen}"
                        )
                        raise RuntimeError("mode1 mismatch")

                if bool(args.debug):
                    print("PHASE4: mode 1 PASS", flush=True)

            elif mode == "2":
                specs = _build_mode2_specs(G=args.G, K=args.K)
                # Mode 2 chain legs resume at multiples of K.
                period = max(1, int(args.K))

                # Run A_ref
                ref_spec = specs[0]
                log_ref = runs_root / "logs" / f"{ref_spec.run_id}.log"
                if bool(args.debug):
                    print(
                        f"PHASE4: run {ref_spec.run_id} start_gen=0 gens={ref_spec.generations}",
                        flush=True,
                    )
                run_ref_dir = _run_simulator(
                    repo_root=repo_root,
                    runs_root=runs_root,
                    run_id=ref_spec.run_id,
                    seed=args.seed,
                    generations=ref_spec.generations,
                    checkpoint_bundle_period=period,
                    enable_start_zero_checkpoint=bool(args.start_zero_checkpoint),
                    resume_from=None,
                    passthrough_args=passthrough,
                    log_path=log_ref,
                debug_stream=bool(args.debug) and bool(args.debug_stream_sim_output),
                )
                res_ref = RunResult(
                    label=ref_spec.label,
                    run_id=ref_spec.run_id,
                    run_dir=run_ref_dir,
                    start_gen=0,
                    end_gen=args.G,
                )
                record_run(res_ref)
                compute_phase4_determinism_fingerprint(
                    run_ref_dir,
                    seed=args.seed,
                    G=args.G,
                    K=args.K,
                    C=args.C,
                    mode="2",
                    determinism_strict=determinism_strict,
                )

                # Chain legs.
                leg_dirs: List[Path] = []
                prev_leg_dir: Optional[Path] = None
                prev_end_gen: Optional[int] = None
                for leg_spec in specs[1:]:
                    log_leg = runs_root / "logs" / f"{leg_spec.run_id}.log"
                    resume_from = None
                    if prev_leg_dir is not None and prev_end_gen is not None:
                        resume_from = _bundle_path(prev_leg_dir, prev_end_gen)
                        try:
                            _verify_checkpoint_bundle_v2_integrity(resume_from)
                        except Exception as e:
                            summary["first_failure"] = {
                                "kind": "checkpoint_missing_or_invalid",
                                "gen": int(prev_end_gen),
                                "message": _one_line(str(e)),
                            }
                            failure_error_line = (
                                f"ERROR: missing/invalid checkpoint bundle at gen {prev_end_gen} (mode 2)"
                            )
                            raise

                    if bool(args.debug):
                        if resume_from is None:
                            print(
                                f"PHASE4: run {leg_spec.run_id} start_gen={leg_spec.start_gen} gens={leg_spec.generations}",
                                flush=True,
                            )
                        else:
                            print(
                                f"PHASE4: run {leg_spec.run_id} resume_gen={leg_spec.start_gen} gens={leg_spec.generations}",
                                flush=True,
                            )
                    run_leg_dir = _run_simulator(
                        repo_root=repo_root,
                        runs_root=runs_root,
                        run_id=leg_spec.run_id,
                        seed=args.seed,
                        generations=leg_spec.generations,
                        checkpoint_bundle_period=period,
                        enable_start_zero_checkpoint=(
                            bool(args.start_zero_checkpoint) and resume_from is None
                        ),
                        resume_from=resume_from,
                        passthrough_args=passthrough,
                        log_path=log_leg,
                    debug_stream=bool(args.debug) and bool(args.debug_stream_sim_output),
                    )
                    end_gen = leg_spec.start_gen + leg_spec.generations
                    res_leg = RunResult(
                        label=leg_spec.label,
                        run_id=leg_spec.run_id,
                        run_dir=run_leg_dir,
                        start_gen=leg_spec.start_gen,
                        end_gen=end_gen,
                    )
                    record_run(res_leg)
                    compute_phase4_determinism_fingerprint(
                        run_leg_dir,
                        seed=args.seed,
                        G=args.G,
                        K=args.K,
                        C=args.C,
                        mode="2",
                        determinism_strict=determinism_strict,
                    )

                    leg_dirs.append(run_leg_dir)
                    prev_leg_dir = run_leg_dir
                    prev_end_gen = end_gen

                # Finalize chain outputs into last leg directory, then compare to ref.
                final_chain_dir = leg_dirs[-1]
                _finalize_chain_outputs(leg_run_dirs=leg_dirs, final_run_dir=final_chain_dir)

                # Recompute fingerprint for chain-final after overwriting metrics/behaviors.
                compute_phase4_determinism_fingerprint(
                    final_chain_dir,
                    seed=args.seed,
                    G=args.G,
                    K=args.K,
                    C=args.C,
                    mode="2",
                    determinism_strict=determinism_strict,
                )

                verifier_log = runs_root / "logs" / "m2_equivalence.log"
                ok, detail = _compare_allowlist(
                    run_a=run_ref_dir,
                    run_b=final_chain_dir,
                    verifier_log=verifier_log,
                    summary_base=runs_root,
                )
                record_compare(
                    mode="2",
                    label="m2_A_ref_vs_chain_final",
                    run_a_label=res_ref.label,
                    run_b_label=specs[-1].label,
                    passed=ok,
                    detail=detail,
                )
                if not ok:
                    summary["first_failure"] = detail
                    failure_error_line = _format_mismatch_error_line(mode="2", detail=detail or {}, run_a_label=res_ref.label, run_b_label=specs[-1].label, resume_gen=args.G)
                    raise RuntimeError("mode2 mismatch")

                if bool(args.debug):
                    print("PHASE4: mode 2 PASS", flush=True)

            else:
                raise ValueError(f"unknown mode: {mode}")

        summary["result"] = "PASS"
        return_code = 0

    except Exception as e:
        # Preserve the first failure detail in summary; avoid multi-line spam.
        if summary.get("first_failure") is None:
            summary["first_failure"] = {"kind": "exception", "message": str(e)}
        if failure_error_line is None:
            failure_error_line = f"ERROR: {_one_line(str(e))}"
        summary["result"] = "FAIL"
        return_code = 1

    finally:
        # Always write summary.
        _write_canonical_json(summary_path, summary)

        # Optional: delete checkpoint bundles on PASS, before zipping (to keep artifacts small).
        if summary.get("result") == "PASS" and bool(args.delete_checkpoint_if_pass):
            cleaned = _delete_checkpoint_bundles_v2_under_runs_root(runs_root)
            if bool(args.debug):
                print(f"PHASE4: deleted checkpoint bundles in {cleaned} run(s)", flush=True)

        # Always zip artifacts created so far.
        zip_name = (
            f"phase4_functional_determinism__seed{args.seed}__G{args.G}__K{args.K}__C{args.C}"
            f"__mode{modes_requested}__{summary.get('result','FAIL')}.zip"
        )
        zip_path = zip_out_dir / zip_name
        _zip_artifacts(zip_path=zip_path, items=created_items, base_dir=zip_out_dir)

        # On failure, print exactly one ERROR line (as required).
        if return_code != 0:
            msg = failure_error_line or "ERROR: phase4 determinism harness failed"
            print(msg)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
