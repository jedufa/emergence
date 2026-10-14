"""tests/harnesses/phase4_functional/verify/phase4_compute_determinism_fingerprint.py

Phase 4 determinism fingerprint generator.

Requirements (Phase 4):
- Produce a stable, canonical JSON file at <run_dir>/determinism_fingerprint.json.
- Include SHA256 of metrics.csv bytes and behaviors.csv bytes.
- Optionally include SHA256 of a stable state dump via --include-state-dump (default off for cross-run comparability).
- Include resolved config (seed, G, K, C, mode) and determinism_strict flag.
- No timestamps, no host-dependent fields, sorted keys, minified JSON.

This is intentionally Phase-4-specific to avoid destabilizing Phase 3 tooling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _sha256_file(path: Path) -> str:
    """Return SHA256 hex digest of the file at path (streamed)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _relpath(child: Path, parent: Path) -> str:
    """Return a stable relative path string for child relative to parent."""
    try:
        return str(child.relative_to(parent))
    except Exception:
        # Fallback: basename only (should not happen in harness-managed layouts).
        return child.name


def _pick_state_dump(run_dir: Path, expected_gen: Optional[int]) -> Optional[Tuple[str, Path]]:
    """Pick a stable state-dump-like artifact.

    Preference order:
    1) checkpoint_bundles_v2/bundle_genXXXXXX/runner_state.json at expected_gen (if provided)
    2) latest checkpoint_bundles_v2/bundle_gen*/runner_state.json

    Returns:
        (kind, path) or None.
    """
    bundles_root = run_dir / "checkpoint_bundles_v2"
    if bundles_root.exists() and bundles_root.is_dir():
        if expected_gen is not None:
            candidate = bundles_root / f"bundle_gen{expected_gen:06d}" / "runner_state.json"
            if candidate.exists() and candidate.is_file():
                return "checkpoint_bundle_v2_runner_state", candidate

        # Fallback: latest bundle by generation index.
        latest_gen = None
        latest_path = None
        for d in bundles_root.glob("bundle_gen*"):
            if not d.is_dir():
                continue
            name = d.name
            if not name.startswith("bundle_gen"):
                continue
            try:
                gen = int(name.replace("bundle_gen", ""))
            except Exception:
                continue
            p = d / "runner_state.json"
            if not (p.exists() and p.is_file()):
                continue
            if latest_gen is None or gen > latest_gen:
                latest_gen = gen
                latest_path = p

        if latest_path is not None:
            return "checkpoint_bundle_v2_runner_state", latest_path

    return None


def compute_phase4_determinism_fingerprint(
    run_dir: Path,
    *,
    seed: int,
    G: int,
    K: int,
    C: int,
    mode: str,
    determinism_strict: bool,
    include_state_dump: bool = False,
) -> Dict[str, Any]:
    """Compute and write the Phase 4 determinism fingerprint.

    Args:
        run_dir: Simulator run directory.
        seed: RNG seed.
        G: Total generations for the harness scenario.
        K: Checkpoint interval (modes 1/2).
        C: Mode 0 checkpoint generation.
        mode: Mode identifier ("0", "1", "2", or "all").
        determinism_strict: Whether strict determinism is enabled.

    Returns:
        The fingerprint dict that was written.
    """
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.csv"
    behaviors_path = run_dir / "behaviors.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv missing: {metrics_path}")
    if not behaviors_path.exists():
        raise FileNotFoundError(f"behaviors.csv missing: {behaviors_path}")

    state_block: Dict[str, Any]
    if not include_state_dump:
        state_block = {"kind": None, "path": None, "sha256": None}
    else:
        state = _pick_state_dump(run_dir, expected_gen=G)
        if state is None:
            state_block = {"kind": None, "path": None, "sha256": None}
        else:
            kind, state_path = state
            state_block = {
                "kind": kind,
                "path": _relpath(state_path, run_dir),
                "sha256": _sha256_file(state_path),
            }

    fp: Dict[str, Any] = {
        "fingerprint_version": 2,
        "determinism_strict": bool(determinism_strict),
        "resolved_config": {
            "seed": int(seed),
            "G": int(G),
            "K": int(K),
            "C": int(C),
            "mode": str(mode),
        },
        "artifacts": {
            "metrics.csv": {
                "path": "metrics.csv",
                "sha256": _sha256_file(metrics_path),
            },
            "behaviors.csv": {
                "path": "behaviors.csv",
                "sha256": _sha256_file(behaviors_path),
            },
            "state_dump": state_block,
        },
    }

    out_path = run_dir / "determinism_fingerprint.json"
    out_path.write_text(
        json.dumps(fp, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return fp


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Phase 4 determinism fingerprint generator")
    p.add_argument("--run_dir", required=True, help="Simulator run directory")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--G", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--C", type=int, required=True)
    p.add_argument("--mode", required=True)
    p.add_argument(
        "--determinism_strict",
        type=int,
        default=None,
        help="1/0; if omitted, inferred from DEBUG_DETERMINISM env var",
    )
    p.add_argument(
        "--include_state_dump",
        action="store_true",
        help="Include stable state dump hash in fingerprint (may break cross-run fingerprint equality if availability differs).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint."""
    args = _parse_args(argv)
    if args.determinism_strict is None:
        determinism_strict = bool(int(os.environ.get("DEBUG_DETERMINISM", "1")))
    else:
        determinism_strict = bool(int(args.determinism_strict))

    compute_phase4_determinism_fingerprint(
        Path(args.run_dir),
        seed=args.seed,
        G=args.G,
        K=args.K,
        C=args.C,
        mode=args.mode,
        determinism_strict=determinism_strict,
        include_state_dump=bool(args.include_state_dump),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
