#!/usr/bin/env python3
"""
tools/verify/verify_checkpoint_bundle_v2.py

Purpose
-------
Verify that Checkpoint Bundle v2 checkpoints were produced as expected during a run, and optionally
compare two completed runs for equivalence.

This script supports three modes:

1) Single-run verification (original behavior)
2) Two-run equivalence comparison: final_bundle (requires bundles in BOTH runs)
3) Two-run equivalence comparison: metrics_only (does NOT require bundles)

Usage
-----
Single-run:
  python -m tests.harnesses.phase3_resume.verify.verify_checkpoint_bundle_v2 --run_dir <path> --period 10 --strict

Two-run compare (final bundle fingerprints):
  python -m tests.harnesses.phase3_resume.verify.verify_checkpoint_bundle_v2 --run_a <pathA> --run_b <pathB> --compare_mode final_bundle --strict

Two-run compare (metrics only):
  python -m tests.harnesses.phase3_resume.verify.verify_checkpoint_bundle_v2 --run_a <pathA> --run_b <pathB> --compare_mode metrics_only --strict

Exit Codes
----------
0 = PASS
2 = FAIL (verification error)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
# -----------------------------------------------------------------------------
# Pretty console output (ANSI). Disable via: EIEL_COLOR=0
# -----------------------------------------------------------------------------
import os as _os

_USE_COLOR = _os.environ.get("EIEL_COLOR", "1") == "1" and sys.stdout.isatty()

def _c(code: str) -> str:
    return f"\033[{code}m" if _USE_COLOR else ""

_RESET = _c("0")
_BOLD = _c("1")
_GREEN = _c("32")
_RED = _c("31")
_YELLOW = _c("33")
_BLUE = _c("34")
_CYAN = _c("36")
_MAGENTA = _c("35")

def _section(title: str) -> None:
    print(f"\n{_BOLD}{_CYAN}========== {title} =========={_RESET}")

def _kv(k: str, v: str) -> None:
    print(f"{_BOLD}{k}:{_RESET} {v}")

def _ok(msg: str) -> None:
    print(f"{_GREEN}[OK]{_RESET} {msg}")

def _warn(msg: str) -> None:
    print(f"{_YELLOW}[WARN]{_RESET} {msg}")

def _fail(msg: str) -> None:
    print(f"{_RED}[FAIL]{_RESET} {msg}")



BUNDLE_SCHEMA = "checkpoint_bundle_v2"


@dataclass(frozen=True)
class BundleInfo:
    """Represents a discovered checkpoint bundle and key parsed fields."""
    bundle_dir: Path
    generation_index: int
    checkpoint_id: str
    strict_replay_mode: bool
    metadata: Dict[str, object]


def _read_json(path: Path) -> Dict[str, object]:
    """Read a UTF-8 JSON file as a dictionary."""
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 over file bytes and return hex digest."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _discover_bundles(run_dir: Path) -> List[Path]:
    """Discover checkpoint bundle directories under a run directory."""
    bundles: List[Path] = []
    for meta_path in run_dir.rglob("metadata.json"):
        try:
            md = _read_json(meta_path)
        except Exception:
            continue
        if md.get("schema_version") == BUNDLE_SCHEMA:
            bundles.append(meta_path.parent)
    return sorted({p.resolve() for p in bundles})


def _parse_bundle(bundle_dir: Path) -> BundleInfo:
    """Parse metadata.json from a bundle and return BundleInfo."""
    meta_path = bundle_dir / "metadata.json"
    if not meta_path.exists():
        raise ValueError(f"Missing metadata.json in {bundle_dir}")
    md = _read_json(meta_path)

    schema = md.get("schema_version")
    if schema != BUNDLE_SCHEMA:
        raise ValueError(f"Unsupported schema_version in {meta_path}: {schema!r}")

    gen = md.get("generation_index")
    if not isinstance(gen, int):
        raise ValueError(f"metadata.generation_index must be int in {meta_path}")

    ckpt_id = md.get("checkpoint_id")
    if not isinstance(ckpt_id, str) or not ckpt_id.strip():
        raise ValueError(f"metadata.checkpoint_id must be non-empty str in {meta_path}")

    strict = md.get("strict_replay_mode")
    if not isinstance(strict, bool):
        if strict in (0, 1):
            strict = bool(strict)
        else:
            raise ValueError(f"metadata.strict_replay_mode must be bool in {meta_path}")

    return BundleInfo(
        bundle_dir=bundle_dir,
        generation_index=gen,
        checkpoint_id=ckpt_id,
        strict_replay_mode=bool(strict),
        metadata=md,
    )


def _required_files() -> List[str]:
    return [
        "metadata.json",
        "world.json",
        "population.json",
        "rng_snapshot_v2.json",
        "control_plane.json",
        "events_hash.txt",
    ]


def _verify_bundle_layout(b: BundleInfo) -> List[str]:
    errors: List[str] = []
    for rel in _required_files():
        if not (b.bundle_dir / rel).exists():
            errors.append(f"[LAYOUT] Missing {rel} in {b.bundle_dir}")

    weights_dir = b.bundle_dir / "weights"
    if not weights_dir.exists() or not weights_dir.is_dir():
        errors.append(f"[LAYOUT] Missing weights/ directory in {b.bundle_dir}")
    else:
        if not list(weights_dir.glob("*.bin")):
            errors.append(f"[LAYOUT] weights/ exists but contains no *.bin files in {b.bundle_dir}")

    m = re.search(r"bundle_gen(\d+)$", b.bundle_dir.name)
    if m:
        try:
            name_gen = int(m.group(1))
            if name_gen != b.generation_index:
                errors.append(
                    f"[LAYOUT] Bundle dir name gen {name_gen} != metadata.generation_index {b.generation_index} in {b.bundle_dir}"
                )
        except Exception:
            pass

    return errors


def _verify_metadata_fields(b: BundleInfo, strict: bool) -> List[str]:
    md = b.metadata
    required = [
        "schema_version",
        "checkpoint_id",
        "generation_index",
        "branch_id",
        "campaign_id",
        "code_commit_hash",
        "python_version",
        "numpy_version",
        "torch_version",
        "population_hash",
        "world_hash",
        "rng_snapshot_hash",
        "event_log_hash",
        "wal_hash",
        "submission_sequence_number",
        "strict_replay_mode",
    ]
    errors: List[str] = []
    for k in required:
        if k not in md:
            errors.append(f"[METADATA] Missing field {k!r} in {b.bundle_dir / 'metadata.json'}")

    if strict:
        for k in ("python_version", "numpy_version", "torch_version", "code_commit_hash"):
            v = md.get(k)
            if not isinstance(v, str) or not v.strip():
                errors.append(f"[METADATA] Strict: {k} must be non-empty string in {b.bundle_dir / 'metadata.json'}")

    return errors


def _verify_events_hash_consistency(b: BundleInfo) -> List[str]:
    errors: List[str] = []
    meta_hash = b.metadata.get("event_log_hash")
    if not isinstance(meta_hash, str) or not meta_hash.strip():
        errors.append(f"[EVENTS] metadata.event_log_hash missing/invalid in {b.bundle_dir / 'metadata.json'}")
        return errors

    events_hash_path = b.bundle_dir / "events_hash.txt"
    if not events_hash_path.exists():
        errors.append(f"[EVENTS] Missing events_hash.txt in {b.bundle_dir}")
        return errors

    txt = events_hash_path.read_text(encoding="utf-8").strip()
    if txt != meta_hash:
        errors.append(f"[EVENTS] events_hash.txt mismatch in {b.bundle_dir}: file={txt} metadata={meta_hash}")
    return errors


def _verify_file_hashes(b: BundleInfo) -> List[str]:
    errors: List[str] = []
    checks = [
        ("world.json", "world_hash"),
        ("population.json", "population_hash"),
        ("rng_snapshot_v2.json", "rng_snapshot_hash"),
    ]
    for rel, meta_key in checks:
        p = b.bundle_dir / rel
        if not p.exists():
            continue
        expected = b.metadata.get(meta_key)
        if not isinstance(expected, str) or not expected.strip():
            errors.append(f"[HASH] metadata.{meta_key} missing/invalid in {b.bundle_dir / 'metadata.json'}")
            continue
        actual = _sha256_file(p)
        if actual != expected:
            errors.append(f"[HASH] {rel} sha256 mismatch in {b.bundle_dir}: actual={actual} metadata={expected}")
    return errors


def _verify_no_tmp_dirs(run_dir: Path) -> List[str]:
    errors: List[str] = []
    for tmp in run_dir.rglob("checkpoint_bundles_v2"):
        if not tmp.is_dir():
            continue
        for p in sorted([p for p in tmp.iterdir() if p.is_dir() and p.name.endswith(".tmp")]):
            errors.append(f"[ATOMIC] Leftover tmp bundle dir (save did not finalize): {p}")
    return errors


def _verify_periodicity(gens: List[int], period: int, strict: bool) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if period <= 0:
        warnings.append("[PERIOD] period<=0 provided; skipping spacing checks.")
        return errors, warnings

    gens_sorted = sorted(gens)
    min_gen = gens_sorted[0]
    max_gen = gens_sorted[-1]

    if min_gen % period != 0:
        warnings.append(f"[PERIOD] First checkpoint generation {min_gen} is not a multiple of period={period}.")

    expected = [g for g in range((min_gen // period) * period, max_gen + 1, period) if g >= min_gen]
    missing = sorted(set(expected) - set(gens_sorted))
    extra = sorted(set(gens_sorted) - set(expected))

    if missing:
        (errors if strict else warnings).append(f"[PERIOD] Missing expected checkpoints for period={period}: {missing}")
    if extra:
        warnings.append(f"[PERIOD] Found extra checkpoints outside expected periodicity: {extra}")

    diffs = [gens_sorted[i + 1] - gens_sorted[i] for i in range(len(gens_sorted) - 1)]
    if diffs and any(d != period for d in diffs):
        (errors if strict else warnings).append(f"[PERIOD] Non-uniform spacing detected, diffs={diffs} expected={period}")

    return errors, warnings


def _format_report(title: str, items: List[str]) -> str:
    if not items:
        return f"{title}: none"
    return title + "\n" + "\n".join(f"- {x}" for x in items)


def _select_final_bundle(bundles: List[BundleInfo]) -> BundleInfo:
    if not bundles:
        raise ValueError("No bundles to select from.")
    return sorted(bundles, key=lambda b: b.generation_index)[-1]


def _compare_metrics_csv(run_a: Path, run_b: Path, strict: bool) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    ma = run_a / "metrics.csv"
    mb = run_b / "metrics.csv"
    if not ma.exists() or not mb.exists():
        msg = f"[COMPARE] metrics.csv missing: A_exists={ma.exists()} ({ma}) B_exists={mb.exists()} ({mb})"
        (errors if strict else warnings).append(msg)
        return errors, warnings

    ha = _sha256_file(ma)
    hb = _sha256_file(mb)
    if ha != hb:
        msg = f"[COMPARE] metrics.csv sha256 mismatch: A={ha} B={hb}"
        (errors if strict else warnings).append(msg)
    return errors, warnings


def _read_csv_as_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a CSV file into (fieldnames, rows) using the stdlib csv module."""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows: List[Dict[str, str]] = []
        for row in reader:
            # DictReader can return None values if row is shorter than header.
            rows.append({k: ("" if v is None else v) for k, v in row.items()})
    return fieldnames, rows


def _infer_gen_column(fieldnames: List[str]) -> Optional[str]:
    """Infer the generation column name from a CSV header."""
    candidates = ["gen", "generation", "generation_index", "generation_idx"]
    for c in candidates:
        if c in fieldnames:
            return c
    return None


def _index_rows_by_gen(rows: List[Dict[str, str]], gen_col: str) -> Dict[int, Dict[str, str]]:
    """Index CSV dict rows by integer generation key."""
    out: Dict[int, Dict[str, str]] = {}
    for r in rows:
        if gen_col not in r:
            continue
        raw = r.get(gen_col, "").strip()
        if raw == "":
            continue
        try:
            g = int(float(raw))
        except Exception:
            continue
        out[g] = r
    return out


def _compare_final_generation_row(
    run_a: Path,
    run_b: Path,
    *,
    strict: bool,
    csv_name: str,
    gen_col: Optional[str] = None,
    ignore_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Compare the row for the final generation in two CSV files.

    This is intentionally different from hashing the whole file: resumed runs often
    do not contain earlier generations, but the *end state* should match.

    Rules:
      - Both CSVs must exist.
      - Determine max generation present in each.
      - If max gens differ: FAIL in strict mode, WARN otherwise.
      - If max gens match: compare the full row dictionaries (minus ignored columns).
    """
    errors: List[str] = []
    warnings: List[str] = []

    pa = run_a / csv_name
    pb = run_b / csv_name
    if not pa.exists() or not pb.exists():
        msg = f"[COMPARE_FINAL_GEN] {csv_name} missing: A_exists={pa.exists()} ({pa}) B_exists={pb.exists()} ({pb})"
        (errors if strict else warnings).append(msg)
        return errors, warnings

    fa, ra = _read_csv_as_rows(pa)
    fb, rb = _read_csv_as_rows(pb)

    gc = gen_col or _infer_gen_column(fa) or _infer_gen_column(fb)
    if not gc:
        msg = f"[COMPARE_FINAL_GEN] Could not infer generation column for {csv_name}. A_cols={fa} B_cols={fb}"
        (errors if strict else warnings).append(msg)
        return errors, warnings

    ia = _index_rows_by_gen(ra, gc)
    ib = _index_rows_by_gen(rb, gc)
    if not ia or not ib:
        msg = f"[COMPARE_FINAL_GEN] Could not find any usable generation rows in {csv_name} (gen_col={gc})"
        (errors if strict else warnings).append(msg)
        return errors, warnings

    ga = max(ia.keys())
    gb = max(ib.keys())
    if ga != gb:
        msg = f"[COMPARE_FINAL_GEN] final generation differs for {csv_name}: A={ga} B={gb}"
        (errors if strict else warnings).append(msg)
        # If gens differ, row comparison is meaningless.
        return errors, warnings

    row_a = dict(ia[ga])
    row_b = dict(ib[gb])

    ignore = set(ignore_cols or [])
    # Always ignore blank header keys if any.
    ignore.add("")

    # Normalize: compare only intersection of keys to avoid false diffs if one side has extra columns.
    keys = sorted(set(row_a.keys()) & set(row_b.keys()) - ignore)
    diffs: List[str] = []
    for k in keys:
        av = row_a.get(k, "")
        bv = row_b.get(k, "")
        if av != bv:
            diffs.append(k)

    if diffs:
        # Show a few columns for quick triage without dumping the universe.
        preview = ", ".join(diffs[:12]) + (" ..." if len(diffs) > 12 else "")
        msg = f"[COMPARE_FINAL_GEN] {csv_name} differs at gen={ga}; differing columns: {preview}"
        (errors if strict else warnings).append(msg)
    return errors, warnings


def _compare_final_bundles(a: BundleInfo, b: BundleInfo, strict: bool, compare_metrics: bool) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    must_match = [
        "world_hash",
        "population_hash",
        "rng_snapshot_hash",
        "event_log_hash",
        "wal_hash",
        "submission_sequence_number",
        "strict_replay_mode",
    ]

    nice_to_match = [
        "code_commit_hash",
        "python_version",
        "numpy_version",
        "torch_version",
        "branch_id",
        "campaign_id",
    ]

    for k in must_match:
        av = a.metadata.get(k)
        bv = b.metadata.get(k)
        if av != bv:
            errors.append(f"[COMPARE] {k} mismatch: A={av!r} (gen {a.generation_index}) vs B={bv!r} (gen {b.generation_index})")

    if a.generation_index != b.generation_index:
        msg = f"[COMPARE] final generation_index differs: A={a.generation_index} B={b.generation_index}"
        (errors if strict else warnings).append(msg)

    for k in nice_to_match:
        av = a.metadata.get(k)
        bv = b.metadata.get(k)
        if av != bv:
            msg = f"[COMPARE] {k} differs: A={av!r} vs B={bv!r}"
            (errors if strict else warnings).append(msg)

    if compare_metrics:
        run_a_root = a.bundle_dir.parent.parent
        run_b_root = b.bundle_dir.parent.parent
        ce, cw = _compare_metrics_csv(run_a_root, run_b_root, strict=strict)
        errors += ce
        warnings += cw

    return errors, warnings


def _verify_single_run(run_dir: Path, period: int, strict: bool, print_bundles: bool) -> Tuple[List[BundleInfo], List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    errors += _verify_no_tmp_dirs(run_dir)

    bundle_dirs = _discover_bundles(run_dir)
    if print_bundles:
        print("Discovered bundles:")
        for d in bundle_dirs:
            print(f"  - {d}")

    if not bundle_dirs:
        errors.append(f"FAIL: No checkpoint_bundle_v2 metadata.json found under {run_dir}")
        return [], errors, warnings

    bundles: List[BundleInfo] = []
    for d in bundle_dirs:
        try:
            bi = _parse_bundle(d)
            bundles.append(bi)
        except Exception as exc:
            errors.append(f"[DISCOVERY] Failed to parse bundle at {d}: {exc}")
            continue

        errors += _verify_bundle_layout(bi)
        errors += _verify_metadata_fields(bi, strict=strict)
        errors += _verify_events_hash_consistency(bi)
        errors += _verify_file_hashes(bi)

    gens = [b.generation_index for b in bundles]
    perr, pwarn = _verify_periodicity(gens, period, strict=strict)
    errors += perr
    warnings += pwarn

    return bundles, errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify Checkpoint Bundle v2 outputs in one or two run directories.")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run_dir", help="Single run directory to scan.")
    g.add_argument("--run_a", help="Run A directory for equivalence comparison.")
    ap.add_argument("--run_b", help="Run B directory for equivalence comparison (required with --run_a).")

    ap.add_argument("--period", type=int, default=0, help="Expected checkpoint period in generations (e.g., 10).")
    ap.add_argument("--strict", action="store_true", help="Fail on any periodicity deviation and on stricter metadata/compare checks.")
    ap.add_argument("--print_bundles", action="store_true", help="Print discovered bundle directories.")

    ap.add_argument(
        "--compare_mode",
        choices=["none", "final_bundle", "metrics_only", "final_gen"],
        default="none",
        help=(
            "Comparison mode with --run_a/--run_b. "
            "'final_bundle' requires bundles in both runs; "
            "'metrics_only' compares metrics.csv only by SHA-256; "
            "'final_gen' compares the final-generation row in metrics.csv (and optionally behaviors.csv)."
        ),
    )
    ap.add_argument(
        "--compare_metrics",
        action="store_true",
        help="When comparing final bundles, also compare run-level metrics.csv by SHA-256 if present.",
    )

    ap.add_argument(
        "--compare_behaviors",
        action="store_true",
        help="When using --compare_mode final_gen, also compare the final-generation row in behaviors.csv.",
    )
    ap.add_argument(
        "--gen_col",
        default="",
        help="Optional explicit generation column name for final_gen comparisons (metrics/behaviors).",
    )
    ap.add_argument(
        "--ignore_cols",
        default="",
        help="Comma-separated list of column names to ignore in final_gen comparisons (e.g., wallclock,elapsed_s).",
    )

    args = ap.parse_args()

    if args.run_a and not args.run_b:
        print("FAIL: --run_b is required when using --run_a", file=sys.stderr)
        return 2
    if args.run_dir and args.run_b:
        print("FAIL: --run_b is only valid with --run_a", file=sys.stderr)
        return 2

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            print(f"FAIL: run_dir does not exist or is not a directory: {run_dir}", file=sys.stderr)
            return 2

        bundles, errors, warnings = _verify_single_run(run_dir, args.period, args.strict, args.print_bundles)

        gens = [b.generation_index for b in bundles]
        _section("CHECKPOINT BUNDLE V2 VERIFY")
        _kv("Run", str(run_dir))
        _kv("Bundles found", f"{len(bundles)}; generations: {sorted(gens)}")
        print(_format_report("WARNINGS", warnings))
        print(_format_report("ERRORS", errors))

        if errors:
            _fail("RESULT: FAIL")
            return 2
        _ok("RESULT: PASS")
        return 0

    run_a = Path(args.run_a).expanduser().resolve()
    run_b = Path(args.run_b).expanduser().resolve()
    if not run_a.exists() or not run_a.is_dir():
        print(f"FAIL: run_a does not exist or is not a directory: {run_a}", file=sys.stderr)
        return 2
    if not run_b.exists() or not run_b.is_dir():
        print(f"FAIL: run_b does not exist or is not a directory: {run_b}", file=sys.stderr)
        return 2

    errors: List[str] = []
    warnings: List[str] = []

    if args.compare_mode == "metrics_only":
        ce, cw = _compare_metrics_csv(run_a, run_b, strict=args.strict)
        errors += ce
        warnings += cw

        _section("EQUIVALENCE (metrics_only)")
        _kv("Run A", str(run_a))
        _kv("Run B", str(run_b))
        print(_format_report("WARNINGS", warnings))
        print(_format_report("ERRORS", errors))
        print("RESULT: FAIL" if errors else "RESULT: PASS")
        return 2 if errors else 0

    if args.compare_mode == "final_gen":
        ignore_cols = [c.strip() for c in (args.ignore_cols or "").split(",") if c.strip()]
        gen_col = args.gen_col.strip() or None

        ce, cw = _compare_final_generation_row(
            run_a,
            run_b,
            strict=args.strict,
            csv_name="metrics.csv",
            gen_col=gen_col,
            ignore_cols=ignore_cols,
        )
        errors += ce
        warnings += cw

        if args.compare_behaviors:
            be, bw = _compare_final_generation_row(
                run_a,
                run_b,
                strict=args.strict,
                csv_name="behaviors.csv",
                gen_col=gen_col,
                ignore_cols=ignore_cols,
            )
            errors += be
            warnings += bw

        _section("EQUIVALENCE (final_gen)")
        _kv("Run A", str(run_a))
        _kv("Run B", str(run_b))
        print(_format_report("WARNINGS", warnings))
        print(_format_report("ERRORS", errors))
        print("RESULT: FAIL" if errors else "RESULT: PASS")
        return 2 if errors else 0

    bundles_a, errors_a, warnings_a = _verify_single_run(run_a, args.period, args.strict, args.print_bundles)
    bundles_b, errors_b, warnings_b = _verify_single_run(run_b, args.period, args.strict, args.print_bundles)

    errors += [f"[A]{e}" for e in errors_a] + [f"[B]{e}" for e in errors_b]
    warnings += [f"[A]{w}" for w in warnings_a] + [f"[B]{w}" for w in warnings_b]

    if args.compare_mode == "final_bundle":
        try:
            final_a = _select_final_bundle(bundles_a)
            final_b = _select_final_bundle(bundles_b)
            ce, cw = _compare_final_bundles(final_a, final_b, strict=args.strict, compare_metrics=args.compare_metrics)
            errors += ce
            warnings += cw
        except Exception as exc:
            errors.append(f"[COMPARE] Failed to compare final bundles: {exc}")

    gens_a = sorted([b.generation_index for b in bundles_a])
    gens_b = sorted([b.generation_index for b in bundles_b])
    _section("EQUIVALENCE (final_bundle)")
    _kv("Run A", str(run_a))
    _kv("Run B", str(run_b))
    print(f"Bundles A: {len(bundles_a)}; generations: {gens_a}")
    print(f"Bundles B: {len(bundles_b)}; generations: {gens_b}")
    print(_format_report("WARNINGS", warnings))
    print(_format_report("ERRORS", errors))
    print("RESULT: FAIL" if errors else "RESULT: PASS")
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
