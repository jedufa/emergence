"""scripts/control_interface_e2e.py

Phase 5B end-to-end (E2E) validation runner.

What this does
-------------
- Builds a TraitInjectionCommand JSON file deterministically.
- Runs the existing Phase 4 functional determinism harness in a selected mode
  (0/1/2 or "all") while seeding the ControlPlane inbox at start.
- This validates the Phase 5B gate: "determinism remains intact with injection scheduled".

Why this exists
---------------
Humans should not have to remember a 14-flag incantation to validate a feature.
This script is a thin orchestration wrapper and must not touch simulation physics.

Usage
-----
python -m scripts.phase5b_e2e --ci-fast --mode 0
python -m scripts.phase5b_e2e --ci-fast --mode all
python -m scripts.phase5b_e2e --ci-fast --mode all --verify-phase5c-branching

Notes
-----
- This script passes the injection into engine.main using:
    --control_inbox_seed_json <file>
  which is copied into <run_dir>/inbox/ immediately after run_dir creation.
- That CLI option is intentionally orchestration-only.
- Any arguments after "--" are passed through to engine.main (via harness convention).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _canonical_json_dumps(obj: Dict[str, Any]) -> str:
    """Return canonical JSON string (sorted keys, minified, newline-terminated)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _build_trait_injection_command_json(
    *,
    command_uuid: str,
    target_generation: int,
    apply_phase: str,
    duration_generations: int,
) -> Dict[str, Any]:
    """Build TraitInjectionCommand JSON payload (ControlPlane inbox)."""
    # IMPORTANT: command submission is via ControlPlane only.
    # This JSON is parsed by simulation.control_plane_pkg.commands.parse_command.
    return {
        "command_type": "TraitInjectionCommand",
        "schema_version": "cmd_v2",
        "command_uuid": str(command_uuid),
        "target_generation": int(target_generation),
        "trait_injection": {
            "schema_version": "trait_injection_v2",
            "duration_generations": int(duration_generations),
            "apply_phase": str(apply_phase),
            "locus_tags": ["memory_dim"],
            "allele_values": {"memory_dim": 64},
            "target_agent_ids": [0, 1],
        },
    }


def _split_passthrough(argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split argv into (driver_args, passthrough_args) using '--' sentinel."""
    argv_list = list(argv)
    if "--" not in argv_list:
        return argv_list, []
    idx = argv_list.index("--")
    return argv_list[:idx], argv_list[idx + 1 :]


def _parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse CLI args for phase5b_e2e."""
    p = argparse.ArgumentParser(description="Phase 5B E2E validation wrapper (Phase4 harness + injection)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--G", type=int, default=100, help="Total generations for the harness (passed to Phase4 harness)")
    p.add_argument("--K", type=int, default=12, help="Checkpoint period used by Phase4 harness")
    p.add_argument("--C", type=int, default=12, help="Resume count used by Phase4 harness")
    p.add_argument("--ci-fast", action="store_true", help="Use CI-fast profile (passed to Phase4 harness)")
    p.add_argument(
        "--mode",
        "--modes",
        dest="mode",
        default="1",
        choices=["0", "1", "2", "all"],
        help='Phase4 harness mode to run ("0", "1", "2", or "all"). Default: "1".',
    )
    p.add_argument(
        "--keep-zips",
        default="none",
        choices=["none", "failures", "all"],
        help="Controls whether the E2E driver preserves any produced artifact bundles (driver-side; Phase4 harness may ignore).",
    )

    p.add_argument(
        "--verify-phase5c-branching",
        action="store_true",
        help="After Phase4 harness completes, also run the Phase5c branching pytest (determinism + isolation + metadata).",
    )
    p.add_argument(
        "--apply-phase",
        dest="apply_phase",
        default="post-selection-pre-mutation",
        choices=["post-reproduction-pre-selection", "post-selection-pre-mutation"],
    )
    p.add_argument("--inject-at-gen", type=int, default=1)
    p.add_argument("--duration", type=int, default=1)
    p.add_argument(
        "--uuid",
        default="00000000-0000-0000-0000-0000000000B5",
        help="Deterministic UUID for the injection command (default is fixed).",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Entry point."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    driver_argv, passthrough_argv = _split_passthrough(raw_argv)
    args = _parse_args(driver_argv)

    repo_root = Path(__file__).resolve().parents[3]

    # Build a deterministic command JSON in a temp file.
    with tempfile.TemporaryDirectory(prefix="phase5b_e2e_") as td:
        td_path = Path(td)
        cmd_json_path = td_path / "trait_injection_cmd.json"
        payload = _build_trait_injection_command_json(
            command_uuid=args.uuid,
            target_generation=args.inject_at_gen,
            apply_phase=args.apply_phase,
            duration_generations=args.duration,
        )
        cmd_json_path.write_text(_canonical_json_dumps(payload), encoding="utf-8")

        # Invoke Phase 4 harness with the selected mode(s), passing through seed json + knob for apply_phase.
        harness_mod = "tests.harnesses.phase4_functional.phase4_functional_determinism_harness"
        cmd: List[str] = [
            sys.executable,
            "-m",
            harness_mod,
            "--mode",
            str(args.mode),
            "--seed",
            str(args.seed),
            "--G",
            str(args.G),
            "--K",
            str(args.K),
            "--C",
            str(args.C),
        ]
        if args.ci_fast:
            cmd.append("--ci-fast")
        # NOTE: Phase4 harness may not support '--keep-zips'. We intentionally do NOT forward it.

        # Pass-through args to engine.main (via harness convention: everything after '--').
        cmd += [
            "--",
            "--control_inbox_seed_json",
            str(cmd_json_path),
            "--knob",
            f"sim_cfg.trait_injection_apply_phase={args.apply_phase}",
        ]
        if passthrough_argv:
            cmd += list(passthrough_argv)

        proc = subprocess.run(cmd, cwd=str(repo_root), text=True)
        rc = int(proc.returncode)

        # Optional Phase 5c verification: run the dedicated pytest that asserts
        # deterministic branching + strict isolation + metadata correctness.
        if rc == 0 and bool(getattr(args, "verify_phase5c_branching", False)):
            p5c_cmd: List[str] = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/integration/branching/test_branching_determinism.py",
            ]
            proc2 = subprocess.run(p5c_cmd, cwd=str(repo_root), text=True)
            rc = int(proc2.returncode)

        return rc




if __name__ == "__main__":
    raise SystemExit(main())
