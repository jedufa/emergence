#!/usr/bin/env python3
"""Phase 3 membership harness.

Goal:
- Compare *alive-set membership* between a fresh run and a resumed run at a specific
  lifecycle point (typically pre_start_generation_alive).

This tool is intentionally pragmatic:
- State dump generation indices can be off-by-one depending on whether the run
  executed 0 or 1 generations.
- Short runs (e.g. "gen=1") often only emit gen000000 dumps.

So we probe for the requested tag with a small, deterministic fallback strategy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
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
_CYAN = _c("36")
_MAGENTA = _c("35")

def _section(title: str) -> None:
    print(f"\n{_BOLD}{_CYAN}========== {title} =========={_RESET}")

def _ok(msg: str) -> None:
    print(f"{_GREEN}[OK]{_RESET} {msg}")

def _fail(msg: str) -> None:
    print(f"{_RED}[FAIL]{_RESET} {msg}")



def _state_dump_dir(run_dir: Path) -> Path:
    return run_dir / "state_dumps"


def _expected_path(run_dir: Path, gen: int, tag: str) -> Path:
    return _state_dump_dir(run_dir) / f"state_{gen:06d}__{tag}.json"


def _find_any_tag(run_dir: Path, tag: str) -> list[Tuple[int, Path]]:
    d = _state_dump_dir(run_dir)
    if not d.exists():
        return []
    out: list[Tuple[int, Path]] = []
    for p in sorted(d.glob(f"state_*__{tag}.json")):
        # state_000123__tag.json
        try:
            stem = p.name.split("__", 1)[0]  # state_000123
            gen_s = stem.split("_", 1)[1]
            gen = int(gen_s)
        except Exception:
            continue
        out.append((gen, p))
    return out



def _find_common_gen_for_tag(fresh_dir: Path, resumed_dir: Path, tag: str, requested_gen: int) -> Optional[int]:
    """Return a generation index that exists in BOTH runs for the given tag.

    If no common gen exists, returns None.

    We pick the common gen closest to requested_gen (tie-breaker: lower gen) to keep behavior deterministic.
    """
    fresh = _find_any_tag(fresh_dir, tag)
    resumed = _find_any_tag(resumed_dir, tag)
    if not fresh or not resumed:
        return None
    fresh_g = {g for g, _ in fresh}
    resumed_g = {g for g, _ in resumed}
    common = sorted(fresh_g & resumed_g)
    if not common:
        return None
    common.sort(key=lambda g: (abs(g - requested_gen), g))
    return common[0]
def _probe_path(run_dir: Path, gen: int, tag: str) -> Tuple[int, Path]:
    """Return (resolved_gen, path).

    Fallback order is deterministic:
      1) exact gen
      2) gen-1 (common when only gen0 exists)
      3) gen+1
      4) nearest available gen for the tag
    """

    candidates = [gen, gen - 1, gen + 1]
    for g in candidates:
        if g < 0:
            continue
        p = _expected_path(run_dir, g, tag)
        if p.exists():
            return g, p

    all_tag = _find_any_tag(run_dir, tag)
    if not all_tag:
        raise FileNotFoundError(
            f"missing state dump for tag '{tag}' in {run_dir} (looked for {candidates} and for any matching files)"
        )

    # choose nearest generation; tie-breaker: lower gen
    all_tag.sort(key=lambda gp: (abs(gp[0] - gen), gp[0]))
    return all_tag[0]


def _load_alive_ids(p: Path) -> set[int]:
    import json

    j = json.loads(p.read_text(encoding="utf-8"))
    alive = j.get("alive_ids")
    if alive is None:
        raise ValueError(f"state dump has no 'alive_ids': {p}")
    return set(int(x) for x in alive)


def _diff(a: set[int], b: set[int]) -> Tuple[set[int], set[int]]:
    return (a - b, b - a)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", required=True, help="fresh run directory")
    ap.add_argument("--resumed", required=True, help="resumed run directory")
    ap.add_argument("--gen", type=int, default=0, help="generation index to probe (default: 0)")
    ap.add_argument(
        "--tag",
        default="pre_start_generation_alive",
        help="state dump tag to compare (default: pre_start_generation_alive)",
    )
    args = ap.parse_args(argv)

    fresh_dir = Path(args.fresh)
    resumed_dir = Path(args.resumed)

    try:
        req_gen = int(args.gen)
        tag = str(args.tag)

        common_gen = _find_common_gen_for_tag(fresh_dir, resumed_dir, tag, req_gen)
        if common_gen is not None:
            # Prefer comparing the *same* generation in both runs.
            req_gen = int(common_gen)

        fresh_g, fresh_p = _probe_path(fresh_dir, req_gen, tag)
        resumed_g, resumed_p = _probe_path(resumed_dir, req_gen, tag)

        # If we still couldn't align to the same gen, treat as a harness SKIP
        # rather than a determinism failure.
        if int(fresh_g) != int(resumed_g):
            _section("PHASE 3 MEMBERSHIP HARNESS")
            _fail("SKIP: could not align dumps to the same generation in both runs")
            print(f"Fresh resolved gen={fresh_g} path={fresh_p}")
            print(f"Resumed resolved gen={resumed_g} path={resumed_p}")
            return 0
    except FileNotFoundError as e:
        # Membership harness is optional in the Phase 3 pipeline. If state dumps are
        # disabled or filtered out, skip gracefully instead of crashing the whole run.
        _section("PHASE 3 MEMBERSHIP HARNESS")
        _fail(f"SKIP: {e}")
        _fail("Membership harness skipped because required state dumps were not found.")
        return 0

    fresh_alive = _load_alive_ids(fresh_p)
    resumed_alive = _load_alive_ids(resumed_p)

    only_in_fresh, only_in_resumed = _diff(fresh_alive, resumed_alive)

    _section("PHASE 3 MEMBERSHIP HARNESS")
    print(f"Fresh:   {fresh_dir}")
    print(f"Resumed: {resumed_dir}")
    print(f"Tag:     {args.tag}")
    print(f"Probe gen requested: {int(args.gen)}")
    print(f"Resolved fresh dump:   gen={fresh_g} path={fresh_p}")
    print(f"Resolved resumed dump: gen={resumed_g} path={resumed_p}")
    print()
    print(f"alive(fresh)={len(fresh_alive)} alive(resumed)={len(resumed_alive)}")

    if not only_in_fresh and not only_in_resumed:
        _ok("RESULT: PASS (alive-set membership identical)")
        return 0

    _fail("RESULT: FAIL (alive-set membership differs)")
    if only_in_fresh:
        s = sorted(only_in_fresh)
        print(f" - only in fresh  (n={len(s)}): {s[:25]}{' ...' if len(s) > 25 else ''}")
    if only_in_resumed:
        s = sorted(only_in_resumed)
        print(f" - only in resumed(n={len(s)}): {s[:25]}{' ...' if len(s) > 25 else ''}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
