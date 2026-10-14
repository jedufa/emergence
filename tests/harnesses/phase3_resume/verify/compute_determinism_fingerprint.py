#!/usr/bin/env python3
"""tools/verify/compute_determinism_fingerprint.py

Determinism fingerprint artifact (Phase 4 hardening).

Creates a stable `determinism_fingerprint.json` inside a given run directory.

Fingerprint contents (v1):
  - sha256(metrics.csv)
  - sha256(behaviors.csv)
  - sha256(final_state_dump)

Some runs (e.g., gen=0 bundle creator runs used for resume determinism tests)
do not emit metrics.csv / behaviors.csv. For those runs, this tool can be
invoked with --allow-missing-csv to omit the CSV hashes while still producing
the fingerprint artifact.

"final_state_dump" selection policy (deterministic):
  1) Latest checkpoint bundle v2 runner_state.json under:
        checkpoint_bundles_v2/bundle_gen*/runner_state.json
     using the highest gen number found in folder names.
  2) Latest checkpoint v1 meta.json under:
        checkpoints/ckpt_gen*/meta.json
     using the highest gen number found in folder names.
  3) Latest state dump JSON under:
        state_dumps/state_*.json
     using lexicographically greatest filename.

This tool is stdlib-only and designed for CI usage.

Usage:
  python -m tests.harnesses.phase3_resume.verify.compute_determinism_fingerprint --run-dir <RUN_DIR>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


_GEN_RE = re.compile(r"gen(\d+)")


def _sha256_file(path: Path) -> str:
    """Return sha256 hex digest for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_gen_int(name: str) -> Optional[int]:
    """Extract generation integer from a string like 'bundle_gen000123'."""
    m = _GEN_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _pick_final_state_dump(run_dir: Path) -> Tuple[Path, str]:
    """Pick the deterministic 'final state dump' file path + a label."""

    # 1) Bundle v2 runner_state.json (preferred)
    v2_root = run_dir / "checkpoint_bundles_v2"
    if v2_root.exists() and v2_root.is_dir():
        best: Tuple[int, Path] | None = None
        for p in v2_root.iterdir():
            if not p.is_dir():
                continue
            gi = _extract_gen_int(p.name)
            if gi is None:
                continue
            cand = p / "runner_state.json"
            if not cand.exists():
                continue
            if best is None or gi > best[0]:
                best = (gi, cand)
        if best is not None:
            return best[1], "bundle_v2.runner_state"

    # 2) Checkpoint v1 meta.json
    v1_root = run_dir / "checkpoints"
    if v1_root.exists() and v1_root.is_dir():
        best2: Tuple[int, Path] | None = None
        for p in v1_root.iterdir():
            if not p.is_dir():
                continue
            gi = _extract_gen_int(p.name)
            if gi is None:
                continue
            cand = p / "meta.json"
            if not cand.exists():
                continue
            if best2 is None or gi > best2[0]:
                best2 = (gi, cand)
        if best2 is not None:
            return best2[1], "checkpoint_v1.meta"

    # 3) state_dumps fallback (lexicographic latest)
    sd_root = run_dir / "state_dumps"
    if sd_root.exists() and sd_root.is_dir():
        cands = sorted([p for p in sd_root.glob("state_*.json") if p.is_file()])
        if cands:
            return cands[-1], "state_dumps.latest"

    raise FileNotFoundError(
        "Could not locate final state dump. Tried: checkpoint_bundles_v2/*/runner_state.json, "
        "checkpoints/*/meta.json, state_dumps/state_*.json"
    )


def _sha256_filtered_csv_by_min_gen(path: Path, *, min_gen: int) -> str:
    """Return sha256 for a CSV filtered by a minimum generation.

    The CSV must have a first column named "generation".

    The hash is computed over a normalized byte stream:
      - UTF-8 encoded text
      - preserved header row
      - included data rows where int(generation) >= min_gen
      - '\n' line endings
      - a trailing final '\n'

    This avoids writing any temporary files while keeping the fingerprint stable.

    Args:
        path: CSV file path.
        min_gen: Minimum generation (inclusive) to include.
    """
    if min_gen <= 0:
        return _sha256_file(path)

    h = hashlib.sha256()
    with path.open("r", encoding="utf-8", errors="replace", newline=None) as f:
        lines = f.read().splitlines()

    if not lines:
        # Hash the empty normalized payload (just a trailing newline for consistency).
        h.update(b"\n")
        return h.hexdigest()

    header = lines[0]
    h.update((header + "\n").encode("utf-8"))

    # Filter rows by generation in first column.
    for row in lines[1:]:
        if not row:
            continue
        first = row.split(",", 1)[0]
        try:
            gi = int(first)
        except Exception:
            # If parsing fails, include the row (conservative, avoids accidental drops).
            gi = min_gen
        if gi >= min_gen:
            h.update((row + "\n").encode("utf-8"))

    # Ensure trailing newline even if no rows matched.
    # (Already ensured by header write above.)
    return h.hexdigest()


def compute_fingerprint(
    run_dir: Path, *, allow_missing_csv: bool = False, min_generation: int = 0
) -> Dict[str, object]:
    """Compute fingerprint dict for a run directory.

    Args:
        run_dir: Path to the run directory.
        allow_missing_csv: If True, do not fail when metrics.csv / behaviors.csv are missing.
                           Instead, omit those artifacts from the fingerprint.
        min_generation: If > 0, hash metrics.csv and behaviors.csv after filtering out
                        rows with generation < min_generation (header preserved).
    """
    run_dir = run_dir.resolve()
    metrics = run_dir / "metrics.csv"
    behaviors = run_dir / "behaviors.csv"

    have_metrics = metrics.exists()
    have_behaviors = behaviors.exists()
    if not have_metrics and not allow_missing_csv:
        raise FileNotFoundError(f"missing metrics.csv: {metrics}")
    if not have_behaviors and not allow_missing_csv:
        raise FileNotFoundError(f"missing behaviors.csv: {behaviors}")

    state_path, state_kind = _pick_final_state_dump(run_dir)

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(run_dir))
        except Exception:
            return str(p.name)

    artifacts: Dict[str, object] = {
        "final_state_dump": {
            "path": rel(state_path),
            "sha256": _sha256_file(state_path),
            "kind": state_kind,
        }
    }

    if have_metrics:
        artifacts["metrics.csv"] = {
            "path": rel(metrics),
            "sha256": _sha256_filtered_csv_by_min_gen(metrics, min_gen=int(min_generation)),
        }
    if have_behaviors:
        artifacts["behaviors.csv"] = {
            "path": rel(behaviors),
            "sha256": _sha256_filtered_csv_by_min_gen(behaviors, min_gen=int(min_generation)),
        }

    fp: Dict[str, object] = {
        "fingerprint_version": 1,
        "artifacts": artifacts,
    }
    return fp


def write_fingerprint(
    run_dir: Path, *, allow_missing_csv: bool = False, min_generation: int = 0
) -> Path:
    """Compute and write determinism_fingerprint.json into run_dir.

    Args:
        run_dir: Run directory.
        allow_missing_csv: If True, permit metrics.csv / behaviors.csv to be missing.
    """
    run_dir = run_dir.resolve()
    fp = compute_fingerprint(
        run_dir, allow_missing_csv=allow_missing_csv, min_generation=int(min_generation)
    )
    out_path = run_dir / "determinism_fingerprint.json"
    payload = json.dumps(fp, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    out_path.write_text(payload, encoding="utf-8")
    return out_path


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path, help="Run directory")
    p.add_argument(
        "--allow-missing-csv",
        action="store_true",
        help="Allow metrics.csv / behaviors.csv to be missing (bundle-only runs).",
    )
    p.add_argument(
        "--min-generation",
        type=int,
        default=0,
        help=(
            "If > 0, hash metrics.csv and behaviors.csv after filtering out rows with "
            "generation < min_generation (header preserved)."
        ),
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    try:
        out = write_fingerprint(
            args.run_dir,
            allow_missing_csv=bool(args.allow_missing_csv),
            min_generation=int(args.min_generation),
        )
    except Exception as e:
        print(f"[FINGERPRINT] FAIL: {e}", file=sys.stderr)
        return 1

    print(f"[FINGERPRINT] OK: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
