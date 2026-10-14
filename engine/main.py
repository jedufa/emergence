# main.py
from __future__ import annotations


import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from evolution.structural_live_semantics import validate_structural_live_semantics_config

# ------------------------------------------------------------
# Determinism env alias (P0-A)
#
# P0-A introduces EIEL_STRICT_DETERMINISM=1 as the canonical "strict" toggle.
# The existing codebase uses DEBUG_DETERMINISM=1.
# We support both, but EIEL_STRICT_DETERMINISM takes precedence.
# ------------------------------------------------------------
try:
    from simulation.determinism_enforcer import sync_env_vars_from_eiel_strict

    sync_env_vars_from_eiel_strict()
except Exception:
    pass

# ------------------------------------------------------------
# Determinism default (startup)
#
# Policy:
# - Strict determinism must be explicit for local exploratory runs.
# - CI runs default to strict unless explicitly opted out.
#
# Activation signals:
# - EIEL_STRICT_DETERMINISM=1 (canonical)
# - DEBUG_DETERMINISM=1 (legacy)
#
# Optional override:
# - EMERGENCE_DEBUG_DETERMINISM_DEFAULT can set a default ONLY when
#   DEBUG_DETERMINISM is otherwise unset.
# ------------------------------------------------------------
try:
    if "DEBUG_DETERMINISM" not in os.environ:
        if "EMERGENCE_DEBUG_DETERMINISM_DEFAULT" in os.environ:
            os.environ["DEBUG_DETERMINISM"] = (os.environ.get("EMERGENCE_DEBUG_DETERMINISM_DEFAULT", "0") or "0").strip()
        else:
            # Default strict ON in CI, default OFF elsewhere.
            ci = (os.environ.get("CI", "") or "").strip().lower()
            os.environ["DEBUG_DETERMINISM"] = "1" if ci in ("1", "true", "yes", "on") else "0"
except Exception:
    pass

# Enforce strict-mode startup invariants as early as possible.
try:
    from simulation.determinism_enforcer import enforce_startup_invariants

    enforce_startup_invariants()
except ImportError:
    # Should never happen in-repo, but do not break startup if packaging is odd.
    pass

# ------------------------------------------------------------
# Phase 3 determinism: constrain thread pools *early*
#
# Why:
# - Even with torch deterministic algorithms enabled, CPU multi-threaded
#   reductions (BLAS / OpenMP) can produce tiny numerical drift.
# - That drift can flip stochastic sampling decisions given identical RNG,
#   causing fresh vs resume divergence.
#
# Strict mode (DEBUG_DETERMINISM=1) prioritizes bitwise reproducibility over
# speed, so we default to single-thread execution for common backends.
# ------------------------------------------------------------
try:
    if (os.environ.get("DEBUG_DETERMINISM", "0") or "0").strip() == "1":
        for _k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            # Respect explicit user overrides.
            os.environ.setdefault(_k, "1")
except Exception:
    pass

import signal

def _early_signal_exit(signum: int, frame) -> None:
    """Exit cleanly if interrupted during Python startup imports.

    Why this exists:
    - Users sometimes hit Ctrl-C while importing heavy deps (torch).
    - The default behavior is an ugly KeyboardInterrupt traceback.
    - We prefer a clean exit code (130) and a single log line.
    """
    # Keep this extremely lightweight: no checkpointing is possible yet.
    print("[interrupt] SIGINT caught during startup. Exiting...", flush=True)
    raise SystemExit(130)


# Install *very early* so it also covers interruptions during `import torch`.
signal.signal(signal.SIGINT, _early_signal_exit)
signal.signal(signal.SIGTERM, _early_signal_exit)


import time
import json
import argparse
import torch
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence


def _seed_torch_rng(global_seed: int) -> dict:
    """Seed torch's legacy global RNG for module initialization.

    Runtime stochastic decisions must still route through RNGManager-managed
    streams. Policy module constructors, however, still use PyTorch's global
    initializer RNG via nn.Linear/GRUCell reset paths. If that global RNG starts
    from OS entropy, two runs with the same simulation seed can begin with
    different policies before RNGManager-managed runtime streams take over.
    """
    seed_u32 = int(global_seed) & 0xFFFFFFFF
    info = {
        "torch_seed_u32": int(seed_u32),
        "seeded_cpu": False,
        "seeded_cuda": False,
        "mode": "bootstrap_policy_initialization_only",
    }
    try:
        torch.manual_seed(int(seed_u32))
        info["seeded_cpu"] = True
    except Exception as e:  # pragma: no cover
        info["torch_seed_error"] = str(e)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed_u32))
            info["seeded_cuda"] = True
    except Exception as e:  # pragma: no cover
        info["torch_cuda_seed_error"] = str(e)
    return info


def _seed_python_numpy_global(global_seed: int) -> dict:
    """Seed Python `random` and NumPy *global* RNGs.

    Phase-3 requirement:
    - Even if the simulation uses RNGManager streams, the checkpoint bundle captures
      legacy global RNG states (random module and numpy.random RandomState).
    - Without explicit seeding, those globals may start from OS entropy and differ
      between identical runs, causing bundle hash mismatches.

    This is intentionally minimal and should not affect simulation behavior
    if global RNGs are not used.

    Returns
    -------
    dict: summary of what we seeded.
    """
    info: dict = {}
    try:
        import random as _random
        _random.seed(int(global_seed) & 0xFFFFFFFF)
        info["python_random_seed_u32"] = int(global_seed) & 0xFFFFFFFF
    except Exception as e:  # pragma: no cover
        info["python_random_seed_error"] = str(e)

    try:
        import numpy as _np
        _np.random.seed(int(global_seed) & 0xFFFFFFFF)
        info["numpy_random_seed_u32"] = int(global_seed) & 0xFFFFFFFF
    except Exception as e:  # pragma: no cover
        info["numpy_random_seed_error"] = str(e)

    return info


def _apply_torch_strict_determinism() -> dict:
    """Apply strict torch determinism settings when DEBUG_DETERMINISM=1.

    Why this exists (simple version)
    --------------------------------
    Seeding Python/NumPy is not enough. Torch (especially on GPU) can use
    nondeterministic kernels unless explicitly forced into deterministic mode.

    Returns
    -------
    dict: a small record of what was applied, for manifest/debugging.
    """
    det_info = {
        "strict": bool(int(os.getenv("DEBUG_DETERMINISM", "0"))),
        "torch_deterministic_algorithms": False,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
        "allow_tf32_matmul": None,
        "allow_tf32_cudnn": None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "notes": [],
    }
    if not det_info["strict"]:
        return det_info

    # Recommended for CUDA determinism. Must be set before the first CUDA matmul.
    # Setting it here is still early enough in practice because CUDA context
    # initializes lazily (on first torch.cuda call).
    if not det_info["cublas_workspace_config"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        det_info["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]

    # Force deterministic algorithms. If torch hits a nondeterministic op,
    # we WANT it to error in strict mode (so the culprit is obvious).
    try:
        torch.use_deterministic_algorithms(True)
        det_info["torch_deterministic_algorithms"] = True
    except Exception as e:
        det_info["notes"].append(f"torch.use_deterministic_algorithms failed: {e!r}")

    # cuDNN knobs (relevant when using conv/recurrent ops etc.)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        det_info["cudnn_deterministic"] = True
        det_info["cudnn_benchmark"] = False
    except Exception as e:
        det_info["notes"].append(f"torch.backends.cudnn flags failed: {e!r}")

    # TF32 can cause numerical drift; disable it in strict mode.
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        det_info["allow_tf32_matmul"] = False
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False
        det_info["allow_tf32_cudnn"] = False
    except Exception:
        pass

    return det_info

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from env.env_config import EnvConfig
from evolution.population import PopConfig
from evolution.inheritance_registry_v1 import (
    DIPLOID_INHERITANCE_PROFILE,
    LEGACY_INHERITANCE_PROFILE,
)
from simulation.runner import SimConfig, SimulationRunner
from simulation.metrics import MetricsLogger
from simulation.checkpoint import checkpoint_load, checkpoint_apply
from knobs_override import apply_knob_overrides, verify_knob_override_readback
from simulation.performance_profiles import apply_performance_profile


@dataclass(frozen=True)
class RunPlan:
    """Pure per-cell execution plan resolved from CLI-style arguments."""

    argv: tuple[str, ...]
    args: argparse.Namespace
    run_id: str
    runs_dir: str
    seed: int
    device: str
    bank_device: str


@dataclass
class RunContext:
    """Live per-cell context for one simulation run."""

    plan: RunPlan
    run_dir: str
    env_cfg: EnvConfig
    pop_cfg: PopConfig
    sim_cfg: SimConfig
    logger: MetricsLogger
    runner: SimulationRunner
    payload: dict | None
    start_gen: int
    started_wall: float
    engine_logger: object | None = None
    execution_global_rng_state: dict[str, Any] | None = None
    execution_torch_global_settings: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunExecutionResult:
    """Small serializable result from executing one run plan."""

    status: str
    returncode: int
    run_dir: str
    metrics_path: str
    run_config_path: str
    manifest_path: str
    wall_seconds: float


def _git_info() -> dict:
    """Collect basic git provenance information (best-effort).

    Returns:
        Dict with keys like commit, describe, dirty. If git isn't available,
        values will be empty strings.
    """
    import subprocess

    def _run(cmd: list[str]) -> str:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore").strip()
            return out
        except Exception:
            return ""

    commit = _run(["git", "rev-parse", "HEAD"])
    describe = _run(["git", "describe", "--tags", "--always", "--dirty"])
    dirty = "dirty" if describe.endswith("-dirty") else ""
    return {"commit": commit, "describe": describe, "dirty": dirty}


def _plan_freeze_enforced_from_env() -> bool:
    """Return whether control-plane plan-freeze enforcement is enabled for this run."""
    try:
        return bool(int(os.getenv("EIEL_ENFORCE_PLAN_FREEZE", "0") or "0"))
    except Exception:
        return False


def _write_manifest_json(run_dir: str, args: argparse.Namespace) -> None:
    """Write a lightweight manifest.json into the run directory.

    This is intentionally small and human-readable. It complements config.json
    (which is the full sim config snapshot).

    Args:
        run_dir: Run output directory.
        args: Parsed CLI args.
    """
    import platform
    import socket
    import getpass
    from datetime import datetime, timezone

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git": _git_info(),
        "cli_args": vars(args),
        "determinism": {
            "strict": bool(int(os.getenv("DEBUG_DETERMINISM", "0"))),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
            # Torch seeding is recorded on the args object when available.
            "torch_seed_u32": int(getattr(args, "_torch_seed_u32", 0) or 0),
        },
        "control_plane": {
            "plan_freeze_enforced": _plan_freeze_enforced_from_env(),
        },
    }
    try:
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    except Exception:
        # Never crash the sim because a manifest couldn't be written.
        pass


def _unique_run_dir(base_dir: str) -> str:
    """Ensure run directory name uniqueness by suffixing _N if needed."""
    if not os.path.exists(base_dir):
        return base_dir
    i = 1
    while True:
        cand = f"{base_dir}_{i}"
        if not os.path.exists(cand):
            return cand
        i += 1

def _is_planned_skeleton_run_dir(run_dir: str) -> bool:
    """Return True if run_dir looks like a WebUI planned skeleton.

    This is used to prevent suffixing (e.g. _1) when WebUI already created
    the run directory and expects the engine to write into it directly.

    Detection criteria (minimal + deterministic):
    - run_dir exists
    - run_dir/viz/run_status.json exists
    - JSON contains either:
        - status == "planned" (case-insensitive)
        - branch_state == "PLANNED"
    """
    try:
        if not run_dir or not os.path.isdir(run_dir):
            return False
        rs = os.path.join(run_dir, "viz", "run_status.json")
        if not os.path.isfile(rs):
            return False
        try:
            import json
            obj = json.loads(open(rs, "r", encoding="utf-8").read())
        except Exception:
            return False
        if not isinstance(obj, dict):
            return False
        status = str(obj.get("status", "") or "").strip().lower()
        branch_state = str(obj.get("branch_state", "") or "").strip()
        return (status == "planned") or (branch_state == "PLANNED")
    except Exception:
        return False



def _dataclass_from_snapshot(cls, snap: dict):
    """Create a dataclass instance from a snapshot dict.

    This helper is intentionally forgiving:
    - Unknown keys are ignored.
    - Missing keys fall back to dataclass defaults.

    Parameters
    ----------
    cls:
        Dataclass type to instantiate.
    snap:
        Snapshot dict (typically from checkpoint meta.json).

    Returns
    -------
    Any
        Instance of cls.
    """
    try:
        from dataclasses import fields
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in (snap or {}).items() if k in allowed}
        obj = cls(**kwargs)
        # Carry over any remaining keys (best-effort forward-compat).
        for k, v in (snap or {}).items():
            if k not in allowed:
                try:
                    setattr(obj, k, v)
                except Exception:
                    pass
        return obj
    except Exception:
        # Last resort: default construct then setattr.
        obj = cls()  # type: ignore
        for k, v in (snap or {}).items():
            try:
                setattr(obj, k, v)
            except Exception:
                pass
        return obj


def write_run_config_csv(run_dir: str, env_cfg: EnvConfig, pop_cfg: PopConfig, sim_cfg: SimConfig) -> None:
    """
    Writes <runs_dir>/<run>/run_config.csv in the format expected by scripts/view_metrics.py
    """
    import csv

    path = os.path.join(run_dir, "run_config.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])

        for k, v in sim_cfg.__dict__.items():
            w.writerow(["sim", k, v])
        for k, v in pop_cfg.__dict__.items():
            w.writerow(["pop", k, v])
        for k, v in env_cfg.__dict__.items():
            w.writerow(["env", k, v])


#def default_runs_dir() -> str:
#    here = os.path.dirname(os.path.abspath(__file__))
#    parent = os.path.dirname(here)
#    return os.path.join(parent, "runs")

def default_runs_dir() -> str:
    """
    Default runs directory.

    Priority:
    1) EIEL_RUNS_ROOT (runtime layout)
    2) EMERGENCE_RUNS_DIR (legacy)
    3) EMERGENCE_OUTPUTS_DIR (legacy/alternate)
    4) repo_root/runtime/adhoc
    """
    env = os.getenv("EIEL_RUNS_ROOT") or os.getenv("EMERGENCE_RUNS_DIR") or os.getenv("EMERGENCE_OUTPUTS_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))

    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    return os.path.join(parent, "runtime", "adhoc")


def _sanitize_label(label: str) -> str:
    """Sanitize a human-provided label for safe use in a folder name.

    The goal is boring, stable names that work across Windows/WSL/Linux.

    Parameters
    ----------
    label:
        Proposed label (e.g., "smoke", "nightly", "resume").

    Returns
    -------
    str
        Sanitized label containing only [A-Za-z0-9._-], with whitespace
        converted to underscores.
    """
    if not label:
        return ""
    out = "".join((c if c.isalnum() or c in "._-" else "_") for c in str(label).strip())
    # collapse repeated underscores
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _default_run_id(args: argparse.Namespace) -> str:
    """Build a stable, information-dense default run folder name.

    Naming scheme (local project timezone):
        YYYYMMDD_HHMMSS__<label>__g<gitshort>__s<seed>

    Examples:
        20260208_214233__smoke__g1a2b3c4__s000042

    Notes:
    - This is only used when the user did not provide --run_id.
    - The full truth still lives in manifest.json + config.json.
    """
    # Timestamp: prefer Europe/Vienna (project's operating timezone),
    # fall back to local timezone if zoneinfo isn't available.
    if ZoneInfo is not None:
        try:
            tz = ZoneInfo("Europe/Vienna")
        except Exception:
            tz = datetime.now().astimezone().tzinfo or timezone.utc
    else:
        tz = datetime.now().astimezone().tzinfo or timezone.utc

    ts = datetime.now(tz).strftime("%Y%m%d_%H%M%S")

    # Prefer --run_tag as a human label (it's already used by wrappers / nox).
    label = _sanitize_label(str(getattr(args, "run_tag", "") or "")) or "run"

    git = _git_info() or {}
    commit = str(git.get("commit", "") or "")
    gitshort = commit[:7] if commit else "nogit"

    seed = getattr(args, "seed", None)
    try:
        seed_i = int(seed)
        if 0 <= seed_i < 1_000_000:
            seed_str = f"{seed_i:06d}"
        else:
            seed_str = str(seed_i)
    except Exception:
        seed_str = "unknown"

    return f"{ts}__{label}__g{gitshort}__s{seed_str}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evo grid runner")

    # -----------------------------
    # Wrapper-only compatibility flags
    # -----------------------------
    # Some wrappers may pass flags that are intended to be
    # handled outside of main.py (like keeping the run directory after zipping).
    # We accept these as no-ops so argparse does not abort when wrappers
    # accidentally forward them.
    p.add_argument(
        "--keep_run_dir",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--run_folder_permanent",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )

    # -----------------------------
    # Resume / restore
    # -----------------------------
    p.add_argument(
        "--resume_from",
        "--resume-from",
        dest="resume_from",
        type=str,
        default="",
        help=(
            "Resume from a checkpoint directory. Example: "
            "runs/<run_id>/checkpoint_bundles_v2/bundle_gen000100"
        ),
    )

    # -----------------------------
    # Phase 5B/5D: ControlPlane inbox seeding (orchestration-only)
    # -----------------------------
    p.add_argument(
        "--control_inbox_seed_json",
        action="append",
        default=[],
        help=(
            "Path to a ControlPlane command JSON to copy into <run_dir>/inbox at startup. "
            "Repeatable. Orchestration-only; does not change simulation semantics."
        ),
    )

    # Core sim
    p.add_argument("--gen", "--generations", dest="generations", type=int, default=100)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument(
        "--execution-route",
        choices=("legacy_generation_route_v1", "tick_native_execution_route_v1"),
        default="legacy_generation_route_v1",
        help="Execution semantics. The default preserves existing generation-qualified runs.",
    )
    p.add_argument(
        "--total-ticks",
        type=int,
        default=None,
        help="Required direct horizon for tick_native_execution_route_v1.",
    )
    p.add_argument(
        "--science-every-ticks",
        type=int,
        default=None,
        help="Passive scientific sample interval for tick_native_execution_route_v1.",
    )
    p.add_argument(
        "--report-every-ticks",
        type=int,
        default=0,
        help="Optional positive tick-native report cadence; 0 disables reports.",
    )
    p.add_argument(
        "--checkpoint-every-ticks",
        type=int,
        default=0,
        help="Optional positive tick-native checkpoint cadence; 0 disables checkpoints.",
    )
    p.add_argument(
        "--runtime-partition-ticks",
        type=int,
        default=None,
        help=(
            "Optional positive tick-native execution partition. This is a runtime "
            "dispatch boundary only and does not create a biological boundary."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--inheritance_profile",
        "--inheritance-profile",
        dest="inheritance_profile",
        choices=[
            LEGACY_INHERITANCE_PROFILE,
            DIPLOID_INHERITANCE_PROFILE,
        ],
        default=LEGACY_INHERITANCE_PROFILE,
        help="Select the versioned inheritance mechanics profile.",
    )


    # -----------------------------
    # Generic knob override (repeatable)
    # -----------------------------
    # Allows the harness/registry to set any supported knob via:
    #   --knob section.field=value
    # Example:
    #   --knob genome.aggression_bias=0.5
    # Notes:
    # - Applied once in main() after normal CLI flags, before run_dir/RNG/derived config.
    # - Fail-fast: unknown keys or bad types abort the run.
    p.add_argument(
        "--knob",
        action="append",
        default=[],
        help="Override a knob as section.field=value (repeatable). Example: --knob genome.aggression_bias=0.5",
    )
    p.add_argument(
        "--performance_profile",
        "--performance-profile",
        dest="performance_profile",
        type=str,
        default="",
        help="Apply a named default-off implementation performance profile after generic --knob overrides.",
    )
    p.add_argument(
        "--native_runtime_enable",
        "--native-runtime-enable",
        dest="native_runtime_enable",
        action="store_true",
        help="Enable the persistent native runtime generation lane.",
    )
    p.add_argument(
        "--native_runtime_cxx26_scope_lock_enable",
        "--native-runtime-cxx26-scope-lock-enable",
        dest="native_runtime_cxx26_scope_lock_enable",
        action="store_true",
        help="Select the narrow CXX26 Flat CPU native envelope and keep unsupported optional operators off.",
    )
    p.add_argument(
        "--native_runtime_fixed_policy_family",
        "--native-runtime-fixed-policy-family",
        dest="native_runtime_fixed_policy_family",
        type=str,
        default="",
        choices=["", "mlp"],
        help="Fixed policy family for the native runtime envelope.",
    )
    p.add_argument(
        "--native_runtime_allow_runtime_jit_build",
        "--native-runtime-allow-runtime-jit-build",
        dest="native_runtime_allow_runtime_jit_build",
        action="store_true",
        help="Allow runtime JIT for development only; measured native runs should use a prebuilt extension.",
    )
    p.add_argument(
        "--native_runtime_max_births_per_generation",
        "--native-runtime-max-births-per-generation",
        dest="native_runtime_max_births_per_generation",
        type=int,
        default=None,
        help="Maximum native resident birth commits per generation.",
    )
    p.add_argument(
        "--policy_weight_mutation_enable",
        "--policy-weight-mutation-enable",
        dest="policy_weight_mutation_enable",
        action="store_true",
        help="Enable policy weight mutation during reproduction.",
    )
    p.add_argument(
        "--policy_weight_mutation_mode",
        "--policy-weight-mutation-mode",
        dest="policy_weight_mutation_mode",
        type=str,
        default=None,
        choices=["dense", "sparse", "gaussian"],
        help="Policy weight mutation mode. gaussian is accepted as a dense alias by the native runtime.",
    )
    p.add_argument(
        "--policy_weight_mutation_rate",
        "--policy-weight-mutation-rate",
        dest="policy_weight_mutation_rate",
        type=float,
        default=None,
        help="Sparse policy weight mutation probability.",
    )
    p.add_argument(
        "--policy_weight_mutation_std",
        "--policy-weight-mutation-std",
        dest="policy_weight_mutation_std",
        type=float,
        default=None,
        help="Policy weight mutation standard deviation.",
    )
    p.add_argument(
        "--structural_payload_activation_enable",
        "--structural-payload-activation-enable",
        dest="structural_payload_activation_enable",
        action="store_true",
        help="Enable default-off REP-1C founder structural payload activation for controlled runs.",
    )
    p.add_argument(
        "--structural_payload_seed_profile",
        "--structural-payload-seed-profile",
        dest="structural_payload_seed_profile",
        type=str,
        default=None,
        choices=["none", "minimal_canonical_v1"],
        help="REP-1C structural seed profile. Use with --structural_payload_activation_enable.",
    )
    p.add_argument(
        "--structural_payload_seed_schema_version",
        "--structural-payload-seed-schema-version",
        dest="structural_payload_seed_schema_version",
        type=str,
        default=None,
        help="REP-1C structural seed schema version.",
    )
    p.add_argument(
        "--structural_payload_seed_ablation",
        "--structural-payload-seed-ablation",
        dest="structural_payload_seed_ablation",
        type=str,
        default=None,
        choices=["none", "zero_connection_gates", "remove_connections"],
        help="REP-2E diagnostic matched seed ablation control. Default is none.",
    )
    p.add_argument(
        "--structural_live_semantics_enable",
        "--structural-live-semantics-enable",
        dest="structural_live_semantics_enable",
        action="store_true",
        help="Enable default-off REP-2B live structural semantics for controlled runs.",
    )
    p.add_argument(
        "--structural_live_semantics_profile",
        "--structural-live-semantics-profile",
        dest="structural_live_semantics_profile",
        type=str,
        default=None,
        choices=["none", "connection_field_compiler_v1"],
        help="REP-2B live structural semantics profile. Use with --structural_live_semantics_enable.",
    )
    p.add_argument(
        "--structural_live_semantics_schema_version",
        "--structural-live-semantics-schema-version",
        dest="structural_live_semantics_schema_version",
        type=str,
        default=None,
        help="REP-2B live structural semantics schema version.",
    )
    p.add_argument(
        "--structural_live_effect_scale",
        "--structural-live-effect-scale",
        dest="structural_live_effect_scale",
        type=float,
        default=None,
        help="REP-2P fixed pre-clipping structural residual scale. Default 1.0; live bound [0.0, 2.0].",
    )
    p.add_argument(
        "--structural_action_opportunity_reporting_enable",
        "--structural-action-opportunity-reporting-enable",
        dest="structural_action_opportunity_reporting_enable",
        action="store_true",
        help="Enable default-off REP-2F structural action opportunity telemetry.",
    )
    p.add_argument(
        "--structural_action_margin_reporting_enable",
        "--structural-action-margin-reporting-enable",
        dest="structural_action_margin_reporting_enable",
        action="store_true",
        help="Enable default-off REP-2FR compact structural action margin telemetry.",
    )
    p.add_argument(
        "--structural_selection_funnel_reporting_enable",
        "--structural-selection-funnel-reporting-enable",
        dest="structural_selection_funnel_reporting_enable",
        action="store_true",
        help="Enable default-off REP-2G reporting-only structural selection funnel telemetry.",
    )
    p.add_argument(
        "--structural_tick_reproduction_funnel_reporting_enable",
        "--structural-tick-reproduction-funnel-reporting-enable",
        dest="structural_tick_reproduction_funnel_reporting_enable",
        action="store_true",
        help="Enable default-off REP-2H reporting-only tick-loop reproduction funnel telemetry.",
    )

    # Policy batching
    p.add_argument("--policy_tick_stride", type=int, default=2,
                   help="Call policy once every K ticks (1=legacy). MLP caches K actions; GRU repeats a macro-action for K ticks.")

    # Execution
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Policy execution device (auto=cuda if available).")
    p.add_argument("--bank_device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Parameter bank device (auto=same as --device).")
    p.add_argument("--torch_threads", type=int, default=0,
                   help="Torch CPU threads (0=auto/default).")
    p.add_argument("--torch_compile_policy", action="store_true",
                   help="Enable torch.compile for policy forward passes (experimental).")

    p.add_argument("--perf", action="store_true",
                   help="Enable per-generation perf bucket timing and write perf_buckets.csv.")

    # Profiler
    p.add_argument("--enable_profiler", action="store_true")

    # -----------------------------
    # Episode-0 selection knob (THIS WAS MISSING)
    # -----------------------------
    p.add_argument(
        "--episode0_pred_err_fitness_weight",
        type=float,
        default=None,
        help="Selection weight on episode-0 prediction error (0 = off).",
    )

    # Step 3: analysis
    p.add_argument(
        "--devcoupling_enable",
        action="store_true",
        help="Enable developmental coupling analysis (devcoupling.csv).",
    )

    # Option C: late-life coherence tax
    # Option C: selection-time late-life predictive coherence penalty
    p.add_argument("--late_pred_cost_enable", action="store_true")
    p.add_argument("--late_pred_cost_weight", type=float, default=None)

    # Step 6: life-history tradeoff knob (post-repro coherence efficiency bonus)
    p.add_argument("--late_coherence_eff_enable", action="store_true",
                   help="Enable post-reproduction coherence efficiency bonus (cheaper actions when coherent).")
    p.add_argument("--late_coherence_eff_weight", type=float, default=None,
                   help="Strength of late-life coherence efficiency bonus (e.g., 0.01–0.05).")
    p.add_argument("--late_coherence_eff_scale", type=float, default=None,
                   help="Normalization scale for late-life prediction error mean.")
    p.add_argument("--late_coherence_eff_min_count", type=int, default=None,
                   help="Min post-repro pred error samples before bonus applies.")
    p.add_argument("--late_coherence_eff_floor", type=float, default=None,
                   help="Floor on action-cost multiplier for the efficiency bonus.")

    # Deprecated legacy: per-tick energy tax (world-dynamics confound)
    p.add_argument("--late_pred_energy_tax_enable", action="store_true")
    p.add_argument("--late_pred_energy_tax_weight", type=float, default=None)
    p.add_argument("--late_pred_energy_tax_cap", type=float, default=None)
    p.add_argument("--late_pred_energy_tax_start_step", type=int, default=None)

    # Backward-compat CLI aliases (kept so old scripts still run):
    # These map to the legacy energy tax fields.
    p.add_argument("--late_pred_cost_cap", type=float, default=None)
    p.add_argument("--late_pred_cost_start_step", type=int, default=None)

    # Part B: adult coherence -> movement cost coupling (deterministic capability pressure)
    p.add_argument(
        "--adult_move_penalty_enable",
        action="store_true",
        help="Enable adult coherence-gated movement energy penalty (Part B).",
    )
    p.add_argument(
        "--adult_move_penalty_start_step",
        type=int,
        default=None,
        help="Life step at which adult movement penalty begins (None = SimConfig default).",
    )
    p.add_argument(
        "--adult_pred_err_ema_alpha",
        type=float,
        default=None,
        help="EMA alpha for adult prediction error tracking (None = SimConfig default).",
    )
    p.add_argument(
        "--adult_move_penalty_scale",
        type=float,
        default=None,
        help="Scale for mapping pred error EMA -> penalty (None = SimConfig default).",
    )
    p.add_argument(
        "--adult_move_penalty_strength",
        type=float,
        default=None,
        help="Strength of movement cost multiplier (None = SimConfig default).",
    )
    p.add_argument(
        "--adult_move_penalty_cap",
        type=float,
        default=None,
        help="Max movement cost multiplier (None = SimConfig default).",
    )

    # Phase 4: temporal consistency pressure (identity drift -> capability penalty)
    p.add_argument(
        "--identity_drift_enable",
        action="store_true",
        help="Enable identity drift pressure (probe signature drift -> movement cost multiplier).",
    )
    p.add_argument(
        "--identity_probe_every_k",
        type=int,
        default=None,
        help="Measure identity probe every K ticks (None = SimConfig default).",
    )
    p.add_argument(
        "--identity_drift_start_step",
        type=int,
        default=None,
        help="Life step at which identity drift penalty begins (None = SimConfig default).",
    )
    p.add_argument(
        "--identity_drift_scale",
        type=float,
        default=None,
        help="Scale for mapping drift -> penalty (None = SimConfig default).",
    )
    p.add_argument(
        "--identity_drift_strength",
        type=float,
        default=None,
        help="Strength of identity drift movement cost multiplier (None = SimConfig default).",
    )
    p.add_argument(
        "--identity_drift_cap",
        type=float,
        default=None,
        help="Max identity drift movement cost multiplier (None = SimConfig default).",
    )
    p.add_argument(
        "--identity_drift_ema_alpha",
        type=float,
        default=None,
        help="EMA alpha for identity drift tracking (None = SimConfig default).",
    )


    # Population
    p.add_argument("--pop", type=int, default=256)
    p.add_argument("--parents", type=float, default=0.20)
    p.add_argument("--cull", type=float, default=0.60)

    # Env overrides
    p.add_argument("--injury_per_damage", type=float, default=None)
    p.add_argument("--injury_repro_block", type=float, default=None)
    # Grid scaling convenience: sets both EnvConfig.width and EnvConfig.height.
    # This is intentionally a single knob so wrappers can scale runs easily.
    p.add_argument(
        "--grid_size",
        type=int,
        default=None,
        help="Square grid size (sets both env width and height). Example: --grid_size 64",
    )


    # -----------------------------
    # Tier 1: Spatial ecology (static heterogeneity)
    # -----------------------------
    p.add_argument("--spatial_ecology_enable", action="store_true", help="Enable Tier 1 spatial ecology maps")
    p.add_argument("--spatial_ecology_seed", type=int, default=0, help="Seed for spatial ecology map generation")
    p.add_argument("--resource_regrowth_map_spec", type=str, default=None, help="JSON MapSpec (string) or @path.json")
    p.add_argument("--resource_spawn_density_map_spec", type=str, default=None, help="JSON MapSpec (string) or @path.json")
    p.add_argument("--hazard_rate_map_spec", type=str, default=None, help="JSON MapSpec (string) or @path.json")
    p.add_argument("--spatial_ecology_log_formats", type=str, default=None, help="Comma list: npy,png,csv")
    p.add_argument("--spatial_ecology_log_region_id_map", action="store_true", help="Also log region-id maps when supported")
    p.add_argument("--spatial_ecology_no_log_stats", action="store_true", help="Disable summary stats writing for maps")

    # -----------------------------
    # Tier 2: Spatial droughts (regional stress events)
    # -----------------------------
    p.add_argument("--spatial_droughts_enable", action="store_true", help="Enable Tier 2 spatial droughts (regional stress)")
    p.add_argument("--spatial_droughts_spec", type=str, default=None, help="JSON drought spec (string) or @path.json")
    p.add_argument("--spatial_droughts_log_timeseries_every", type=int, default=None, help="Log drought intensity summary every N steps (default 1)")
    p.add_argument("--spatial_droughts_log_periodic_every", type=int, default=None, help="Log periodic drought multiplier stats every N steps (default 100)")

    # Shock system
    


    # === Tier 3 change: CLI wiring ===
    p.add_argument("--migration_pressure_enable", action="store_true",
                   help="Enable Tier 3 migration pressure system (master gate).")
    p.add_argument("--tier3_regions_spec", type=str, default=None,
                   help="Tier 3 region partition spec JSON (inline) or @path.json")
    p.add_argument("--tier3_regional_carrying_capacity", type=str, default=None,
                   help="Tier 3 carrying capacity spec JSON (inline) or @path.json")
    p.add_argument("--tier3_movement_cost_map", type=str, default=None,
                   help="Tier 3 movement cost map spec JSON (inline) or @path.json")
    p.add_argument("--tier3_region_transition_risk", type=str, default=None,
                   help="Tier 3 region transition risk spec JSON (inline) or @path.json")
    p.add_argument("--tier3_validation_force_random_walk", action="store_true",
                   help="Validation-only: override policies with random walk actions.")
    # === Tier 3 change: end CLI wiring ===
    # -----------------------------
    # Tier 4: juvenile spatial vulnerability (life-history pressure, age-only)
    # -----------------------------
    p.add_argument("--tier4_juvenile_vulnerability_enable", action="store_true",
                   help="Enable Tier 4 juvenile vulnerability (age-based; no hard repro bans).")
    p.add_argument("--tier4_juvenile_steps", type=int, default=0,
                   help="Juvenile window length in steps. Juvenile iff age_steps < this. 0 disables.")
    p.add_argument("--tier4_juvenile_grace_steps", type=int, default=0,
                   help="Optional grace window in steps where hazard amplification is suppressed.")
    p.add_argument("--tier4_juvenile_hazard_multiplier", type=float, default=1.0,
                   help="Hazard multiplier for juveniles (clamped [1,10]).")
    p.add_argument("--tier4_juvenile_movement_penalty", type=float, default=1.0,
                   help="Movement cost multiplier for juveniles (clamped [1,10]).")
    p.add_argument("--tier4_juvenile_harvest_multiplier", type=float, default=1.0,
                   help="Harvest gain multiplier for juveniles (clamped [0,1]).")
    p.add_argument("--tier4_survival_bins_steps", type=str, default=None,
                   help="Comma-separated step edges for Tier 4 survival histograms (e.g. 0,10,25,50).")

    p.add_argument("--shock_start_prob", type=float, default=None)
    p.add_argument("--shock_duration_min", type=int, default=None)
    p.add_argument("--shock_duration_max", type=int, default=None)
    p.add_argument("--shock_cooldown_min", type=int, default=None)
    p.add_argument("--shock_cooldown_max", type=int, default=None)


    # -----------------------------
    # Checkpointing (Checkpoint Bundle v2)
    # -----------------------------
    p.add_argument(
        "--checkpoint_bundle_v2_period_gen",
        type=int,
        default=None,
        help="Save a Checkpoint Bundle v2 every K generations (0=off). Overrides --checkpoint_period_gen.",
    )
    p.add_argument(
        "--checkpoint_bundle_v2_subdir",
        type=str,
        default=None,
        help="Subdirectory under run_dir to store checkpoint bundles (default: checkpoint_bundles_v2).",
    )
    p.add_argument(
        "--checkpoint_bundle_v2_strict_replay_mode",
        type=int,
        default=None,
        help="Bundle v2 metadata strict mode (1=on, 0=off). Default: 1.",
    )

    # Legacy compatibility flags (v1 writer removed; treated as v2 aliases/no-ops)
    p.add_argument(
        "--checkpoint_subdir",
        type=str,
        default="",
        help="(legacy alias) Subdirectory under run_dir for checkpoint bundles.",
    )
    p.add_argument(
        "--checkpoint_include_banks",
        type=int,
        default=None,
        help="(ignored) Bundle v2 captures required state; this flag is legacy.",
    )

    p.add_argument(
        "--checkpoint_period_gen",
        type=int,
        default=0,
        help="(legacy alias) Save a Checkpoint Bundle v2 every K generations (0=off).",
    )
    p.add_argument(
        "--checkpoint_rotate",
        type=int,
        default=0,
        help="Keep only the newest N checkpoint bundles (0=keep all).",
    )
    p.add_argument("--runs_dir", type=str, default=default_runs_dir())
    p.add_argument("--allow_existing_run_dir", action="store_true", help="Allow using an existing <runs_dir>/<run_id> directory (do not uniquify)")
    p.add_argument("--run_id", type=str, default="", help="Optional explicit run folder name (default: timestamp).")
    p.add_argument("--run_tag", type=str, default="", help="Optional label used in output naming.")
    return p.parse_args(argv)


def build_run_plan_from_argv(argv: Sequence[str] | None = None) -> RunPlan:
    """Resolve CLI-style arguments into a pure single-cell run plan."""

    args = parse_args(argv)
    run_id = str(getattr(args, "run_id", "") or "").strip()
    return RunPlan(
        argv=tuple(str(x) for x in (argv or ())),
        args=args,
        run_id=run_id,
        runs_dir=os.path.abspath(str(getattr(args, "runs_dir", "") or default_runs_dir())),
        seed=int(getattr(args, "seed", 0) or 0),
        device=str(getattr(args, "device", "auto") or "auto"),
        bank_device=str(getattr(args, "bank_device", "auto") or "auto"),
    )


def _resolve_resume_path(path_str: str) -> str:
    """Resolve a resume path robustly across wrapper working-directory changes.

    The evo_run wrapper changes the process working directory into the repo.
    Users often pass paths like 'runs/<run_id>/checkpoint_bundles_v2/<bundle>'.
    If that path doesn't exist relative to CWD, we also try resolving it
    relative to the parent directory of this repo (the usual project root).
    """
    from pathlib import Path

    p = Path(path_str)
    if p.is_absolute():
        return str(p)

    # First: relative to current working directory.
    cand1 = (Path.cwd() / p).resolve()
    if cand1.exists():
        return str(cand1)

    # Second: relative to repo parent (main.py lives inside engine/).
    try:
        repo_dir = Path(__file__).resolve().parent
        repo_parent = repo_dir.parent
        cand2 = (repo_parent / p).resolve()
        if cand2.exists():
            return str(cand2)
    except Exception:
        pass

    # Fallback: return cand1 string (will error later with a clear message).
    return str(cand1)


def _load_json_spec_arg(arg: str) -> dict:
    """Load a JSON MapSpec from a CLI argument.

    Supports:
    - Inline JSON: '{"name":"...", "type":"uniform", "params":{...}}'
    - File reference: '@path/to/spec.json'

    Fails fast on invalid JSON.
    """
    import json
    from pathlib import Path

    if arg is None:
        raise ValueError("MapSpec arg must not be None")
    s = str(arg).strip()
    if not s:
        raise ValueError("MapSpec arg must be non-empty")

    if s.startswith("@"):
        path = Path(s[1:]).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MapSpec file not found: {path}")
        data = path.read_text(encoding="utf-8")
        return json.loads(data)

    return json.loads(s)


def snapshot_global_rng_state() -> dict[str, Any]:
    """Capture process-global RNG state used by legacy compatibility paths."""

    state: dict[str, Any] = {}
    try:
        import random as _random

        state["python_random"] = _random.getstate()
    except Exception as exc:
        state["python_random_error"] = repr(exc)
    try:
        import numpy as _np

        np_state = _np.random.get_state()
        state["numpy_random"] = (
            np_state[0],
            np_state[1].copy(),
            int(np_state[2]),
            int(np_state[3]),
            float(np_state[4]),
        )
    except Exception as exc:
        state["numpy_random_error"] = repr(exc)
    try:
        state["torch_cpu"] = torch.get_rng_state().clone()
    except Exception as exc:
        state["torch_cpu_error"] = repr(exc)
    try:
        if torch.cuda.is_available():
            state["torch_cuda"] = [t.clone() for t in torch.cuda.get_rng_state_all()]
        else:
            state["torch_cuda"] = []
    except Exception as exc:
        state["torch_cuda_error"] = repr(exc)
    return state


def restore_global_rng_state(state: dict[str, Any] | None) -> None:
    """Restore a snapshot captured by snapshot_global_rng_state."""

    if not isinstance(state, dict):
        return
    try:
        import random as _random

        if "python_random" in state:
            _random.setstate(state["python_random"])
    except Exception:
        pass
    try:
        import numpy as _np

        if "numpy_random" in state:
            _np.random.set_state(state["numpy_random"])
    except Exception:
        pass
    try:
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
    except Exception:
        pass
    try:
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
    except Exception:
        pass


def snapshot_torch_global_settings() -> dict[str, Any]:
    """Capture torch process-global settings that construction may alter."""

    settings: dict[str, Any] = {}
    try:
        settings["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
    except Exception as exc:
        settings["deterministic_algorithms_error"] = repr(exc)
    try:
        settings["num_threads"] = int(torch.get_num_threads())
    except Exception as exc:
        settings["num_threads_error"] = repr(exc)
    try:
        settings["num_interop_threads"] = int(torch.get_num_interop_threads())
    except Exception as exc:
        settings["num_interop_threads_error"] = repr(exc)
    try:
        settings["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        settings["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        settings["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception as exc:
        settings["cudnn_error"] = repr(exc)
    try:
        settings["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception as exc:
        settings["cuda_matmul_error"] = repr(exc)
    return settings


def restore_torch_global_settings(settings: dict[str, Any] | None) -> None:
    """Restore torch process-global settings captured before/for a context."""

    if not isinstance(settings, dict):
        return
    try:
        if "deterministic_algorithms" in settings:
            torch.use_deterministic_algorithms(bool(settings["deterministic_algorithms"]))
    except Exception:
        pass
    try:
        if "num_threads" in settings:
            torch.set_num_threads(int(settings["num_threads"]))
    except Exception:
        pass
    try:
        if "num_interop_threads" in settings:
            torch.set_num_interop_threads(int(settings["num_interop_threads"]))
    except Exception:
        pass
    try:
        if "cudnn_deterministic" in settings:
            torch.backends.cudnn.deterministic = bool(settings["cudnn_deterministic"])
        if "cudnn_benchmark" in settings:
            torch.backends.cudnn.benchmark = bool(settings["cudnn_benchmark"])
        if "cudnn_allow_tf32" in settings:
            torch.backends.cudnn.allow_tf32 = bool(settings["cudnn_allow_tf32"])
    except Exception:
        pass
    try:
        if "cuda_matmul_allow_tf32" in settings:
            torch.backends.cuda.matmul.allow_tf32 = bool(settings["cuda_matmul_allow_tf32"])
    except Exception:
        pass


def build_run_context_from_plan(plan: RunPlan) -> RunContext:
    """Build a live per-cell context while preserving caller global state."""

    caller_rng_state = snapshot_global_rng_state()
    caller_torch_settings = snapshot_torch_global_settings()
    try:
        ctx = _build_run_context_from_plan_unpreserved(plan)
        ctx.execution_global_rng_state = snapshot_global_rng_state()
        ctx.execution_torch_global_settings = snapshot_torch_global_settings()
        return ctx
    finally:
        restore_torch_global_settings(caller_torch_settings)
        restore_global_rng_state(caller_rng_state)


def _build_run_context_from_plan_unpreserved(plan: RunPlan) -> RunContext:
    """Build the live per-cell context for one resolved run plan."""

    args = plan.args
    det_strict = bool(int(os.getenv("DEBUG_DETERMINISM", "0")))
    det_info = _apply_torch_strict_determinism()

    # P0-C: bootstrap no longer seeds torch global RNG. Torch randomness must
    # be consumed through RNGManager stream generators in runtime paths.
    torch_seed_info = _seed_torch_rng(int(getattr(args, "seed", 0) or 0))
    py_np_seed_info = _seed_python_numpy_global(int(getattr(args, "seed", 0) or 0))
    try:
        # Persist for manifest.json
        args._torch_seed_u32 = int(torch_seed_info.get("torch_seed_u32") or 0)  # type: ignore[attr-defined]
        args._py_np_seed_u32 = int(py_np_seed_info.get("python_random_seed_u32") or 0)  # type: ignore[attr-defined]
    except Exception:
        pass

    # P0-A runtime guard: after bootstrapping seeds, patch seed APIs to fail
    # fast if called later (accidental refactor drift).
    try:
        from simulation.determinism_enforcer import install_runtime_rng_guards

        install_runtime_rng_guards()
    except Exception:
        pass

    # ------------------------------------------------------------
    # Observability: engine logging (console + JSONL)
    # ------------------------------------------------------------
    try:
        from emergence.obs_logging import setup_logging, get_logger, log_event  # noqa: WPS433 (repo-local import)
        log_dir = os.environ.get("EMERG_LOG_DIR", "").strip()
        if not log_dir:
            base_out = os.environ.get("EMERGENCE_OUTPUTS_DIR", "").strip() or os.getcwd()
            log_dir = os.path.join(base_out, "logs")
        os.makedirs(log_dir, exist_ok=True)
        setup_logging(scope="engine", jsonl_path=os.path.join(log_dir, "engine.jsonl"), console_prefix="[ENGINE] ")
        elog = get_logger("emergence.engine.main")
        log_event(elog, event="ENGINE_START", seed=int(getattr(args, "seed", 0) or 0), argv=" ".join(sys.argv))
        # Determinism mode visibility
        log_event(elog, event="DETERMINISM_MODE", strict=det_strict, torch=det_info, torch_seed=torch_seed_info, py_np_seed=py_np_seed_info)
        print(f"[ENGINE] Determinism strict={det_strict}", flush=True)

    except Exception:
        elog = None  # type: ignore[assignment]


    _t0_wall = time.time()

    # Torch threading
    # --------------
    # In strict determinism mode we force single-thread execution to avoid
    # scheduler-dependent numerical drift in CPU kernels.
    if det_strict:
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            # Must be set before the first inter-op parallel region.
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    else:
        # Allow CLI override, otherwise keep a sensible default.
        if int(getattr(args, "torch_threads", 0) or 0) > 0:
            torch.set_num_threads(int(args.torch_threads))
        else:
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    env_cfg = EnvConfig()
    # Grid scaling override
    if getattr(args, "grid_size", None) is not None:
        gs = int(args.grid_size)
        if gs <= 0:
            raise SystemExit("--grid_size must be a positive integer")
        env_cfg.width = gs
        env_cfg.height = gs
    cxx26_scope_lock_requested = bool(getattr(args, "native_runtime_cxx26_scope_lock_enable", False))
    if cxx26_scope_lock_requested:
        env_cfg.patch_world_enable = False
        env_cfg.patch_world_observation_enable = False
        env_cfg.reproduce_requires_mate = False

    pop_cfg = PopConfig(
        population_size=args.pop,
        parent_fraction=args.parents,
        cull_fraction=args.cull,
    )
    sim_cfg = SimConfig(
        generations=args.generations,
        steps_per_generation=args.steps,
        seed=args.seed,
        inheritance_profile=str(getattr(args, "inheritance_profile", LEGACY_INHERITANCE_PROFILE)),
        policy_tick_stride=int(getattr(args, 'policy_tick_stride', 1) or 1),
    )
    sim_cfg.execution_route_identity = str(
        getattr(args, "execution_route", "legacy_generation_route_v1")
    )
    if sim_cfg.execution_route_identity == "tick_native_execution_route_v1":
        total_ticks = getattr(args, "total_ticks", None)
        if isinstance(total_ticks, bool) or not isinstance(total_ticks, int) or total_ticks <= 0:
            raise SystemExit("--total-ticks is required and must be positive for tick_native_execution_route_v1")
        sim_cfg.total_ticks = int(total_ticks)
        sim_cfg.science_sample_interval_ticks = getattr(args, "science_every_ticks", None)
        sim_cfg.report_every_ticks = int(getattr(args, "report_every_ticks", 0) or 0)
        sim_cfg.checkpoint_every_ticks = int(getattr(args, "checkpoint_every_ticks", 0) or 0)
        runtime_partition_ticks = getattr(args, "runtime_partition_ticks", None)
        sim_cfg.runtime_partition_ticks = (
            int(runtime_partition_ticks)
            if runtime_partition_ticks is not None
            else int(total_ticks)
        )
        if sim_cfg.report_every_ticks < 0 or sim_cfg.checkpoint_every_ticks < 0:
            raise SystemExit("tick-native report/checkpoint cadence must be 0 or a positive integer")
        if sim_cfg.runtime_partition_ticks <= 0:
            raise SystemExit("--runtime-partition-ticks must be a positive integer")

    try:
        if elog is not None:
            from emergence.obs_logging import log_event  # noqa: WPS433
            log_event(
                elog,
                event="ENGINE_CONFIG",
                seed=int(sim_cfg.seed),
                generations=int(sim_cfg.generations),
                steps=int(sim_cfg.steps_per_generation),
                device=str(getattr(sim_cfg, "device", "")),
                bank_device=str(getattr(sim_cfg, "bank_device", "")),
            )
    except Exception:
        pass


    # Execution controls
    sim_cfg.device = str(getattr(args, "device", "auto"))
    sim_cfg.bank_device = str(getattr(args, "bank_device", "auto"))
    sim_cfg.torch_threads = int(getattr(args, "torch_threads", 0) or 0)
    sim_cfg.torch_compile_policy = bool(getattr(args, "torch_compile_policy", False))
    sim_cfg.native_runtime_enable = bool(getattr(args, "native_runtime_enable", False))
    sim_cfg.native_runtime_cxx26_scope_lock_enable = bool(
        getattr(args, "native_runtime_cxx26_scope_lock_enable", False)
    )
    sim_cfg.native_runtime_fixed_policy_family = str(
        getattr(args, "native_runtime_fixed_policy_family", "") or ""
    )
    if bool(sim_cfg.native_runtime_cxx26_scope_lock_enable):
        from native_runtime.safe_default import apply_native_runtime_scope_lock_defaults

        apply_native_runtime_scope_lock_defaults(sim_cfg)
    sim_cfg.native_runtime_allow_runtime_jit_build = bool(
        getattr(args, "native_runtime_allow_runtime_jit_build", False)
    )
    _native_max_births_arg = getattr(args, "native_runtime_max_births_per_generation", None)
    if _native_max_births_arg is not None:
        sim_cfg.native_runtime_max_births_per_generation = int(_native_max_births_arg)
    if bool(getattr(args, "policy_weight_mutation_enable", False)):
        sim_cfg.policy_weight_mutation_enable = True
    if getattr(args, "policy_weight_mutation_mode", None) is not None:
        sim_cfg.policy_weight_mutation_mode = str(args.policy_weight_mutation_mode)
    if getattr(args, "policy_weight_mutation_rate", None) is not None:
        sim_cfg.policy_weight_mutation_rate = float(args.policy_weight_mutation_rate)
    if getattr(args, "policy_weight_mutation_std", None) is not None:
        sim_cfg.policy_weight_mutation_std = float(args.policy_weight_mutation_std)
    sim_cfg.structural_payload_activation_enable = bool(
        getattr(args, "structural_payload_activation_enable", False)
    )
    if getattr(args, "structural_payload_seed_profile", None) is not None:
        sim_cfg.structural_payload_seed_profile = str(args.structural_payload_seed_profile)
    elif bool(sim_cfg.structural_payload_activation_enable):
        sim_cfg.structural_payload_seed_profile = "minimal_canonical_v1"
    if getattr(args, "structural_payload_seed_schema_version", None) is not None:
        sim_cfg.structural_payload_seed_schema_version = str(args.structural_payload_seed_schema_version)
    if getattr(args, "structural_payload_seed_ablation", None) is not None:
        sim_cfg.structural_payload_seed_ablation = str(args.structural_payload_seed_ablation)
    sim_cfg.structural_live_semantics_enable = bool(
        getattr(args, "structural_live_semantics_enable", False)
    )
    if getattr(args, "structural_live_semantics_profile", None) is not None:
        sim_cfg.structural_live_semantics_profile = str(args.structural_live_semantics_profile)
    elif bool(sim_cfg.structural_live_semantics_enable):
        sim_cfg.structural_live_semantics_profile = "connection_field_compiler_v1"
    if getattr(args, "structural_live_semantics_schema_version", None) is not None:
        sim_cfg.structural_live_semantics_schema_version = str(args.structural_live_semantics_schema_version)
    if getattr(args, "structural_live_effect_scale", None) is not None:
        sim_cfg.structural_live_effect_scale = float(args.structural_live_effect_scale)
    sim_cfg.structural_action_opportunity_reporting_enable = bool(
        getattr(args, "structural_action_opportunity_reporting_enable", False)
    )
    sim_cfg.structural_action_margin_reporting_enable = bool(
        getattr(args, "structural_action_margin_reporting_enable", False)
    )
    sim_cfg.structural_selection_funnel_reporting_enable = bool(
        getattr(args, "structural_selection_funnel_reporting_enable", False)
    )
    sim_cfg.structural_tick_reproduction_funnel_reporting_enable = bool(
        getattr(args, "structural_tick_reproduction_funnel_reporting_enable", False)
    )
    validate_structural_live_semantics_config(sim_cfg)

    # -----------------------------
    # CLI overrides
    # Checkpointing controls (v2-only writer)
    #
    # Primary knobs:
    #   --checkpoint_bundle_v2_period_gen
    #   --checkpoint_bundle_v2_subdir
    # Legacy aliases kept for compatibility:
    #   --checkpoint_period_gen (maps to v2 period if v2 flag not set)
    #   --checkpoint_subdir (maps to v2 subdir if v2 flag not set)
    sim_cfg.checkpoint_period_gen = int(getattr(args, "checkpoint_period_gen", 0) or 0)
    sim_cfg.checkpoint_rotate = int(getattr(args, "checkpoint_rotate", 0) or 0)

    _v2_period_arg = getattr(args, "checkpoint_bundle_v2_period_gen", None)
    if _v2_period_arg is None:
        sim_cfg.checkpoint_bundle_v2_period_gen = int(getattr(args, "checkpoint_period_gen", 0) or 0)
    else:
        sim_cfg.checkpoint_bundle_v2_period_gen = int(_v2_period_arg or 0)

    _legacy_subdir_arg = str(getattr(args, "checkpoint_subdir", "") or "").strip()
    if _legacy_subdir_arg:
        # Keep legacy field populated for observability/config dumps.
        sim_cfg.checkpoint_subdir = _legacy_subdir_arg

    _v2_subdir_arg = str(getattr(args, "checkpoint_bundle_v2_subdir", "") or "").strip()
    if _v2_subdir_arg:
        sim_cfg.checkpoint_bundle_v2_subdir = _v2_subdir_arg
    elif _legacy_subdir_arg:
        # Legacy alias: route v2 bundles into the legacy subdir.
        sim_cfg.checkpoint_bundle_v2_subdir = _legacy_subdir_arg

    _strict_arg = getattr(args, "checkpoint_bundle_v2_strict_replay_mode", None)
    if _strict_arg is not None:
        try:
            sim_cfg.checkpoint_bundle_v2_strict_replay_mode = bool(int(_strict_arg))
        except Exception:
            sim_cfg.checkpoint_bundle_v2_strict_replay_mode = bool(_strict_arg)

    # -----------------------------
    if args.enable_profiler:
        sim_cfg.enable_profiler = True

    if args.episode0_pred_err_fitness_weight is not None:
        sim_cfg.episode0_pred_err_fitness_weight = float(
            args.episode0_pred_err_fitness_weight
        )

    if args.devcoupling_enable:
        sim_cfg.devcoupling_enable = True

    if args.late_pred_cost_enable:
        sim_cfg.late_pred_cost_enable = True
    if args.late_pred_cost_weight is not None:
        sim_cfg.late_pred_cost_weight = float(args.late_pred_cost_weight)

    # Step 6: late-life coherence efficiency bonus (life-history tradeoff)
    if getattr(args, "late_coherence_eff_enable", False):
        sim_cfg.late_coherence_eff_enable = True
    if getattr(args, "late_coherence_eff_weight", None) is not None:
        sim_cfg.late_coherence_eff_weight = float(args.late_coherence_eff_weight)
    if getattr(args, "late_coherence_eff_scale", None) is not None:
        sim_cfg.late_coherence_eff_scale = float(args.late_coherence_eff_scale)
    if getattr(args, "late_coherence_eff_min_count", None) is not None:
        sim_cfg.late_coherence_eff_min_count = int(args.late_coherence_eff_min_count)
    if getattr(args, "late_coherence_eff_floor", None) is not None:
        sim_cfg.late_coherence_eff_floor = float(args.late_coherence_eff_floor)

    # Legacy per-tick energy tax (confound) - explicit flags
    if getattr(args, "late_pred_energy_tax_enable", False):
        sim_cfg.late_pred_energy_tax_enable = True
    if getattr(args, "late_pred_energy_tax_weight", None) is not None:
        sim_cfg.late_pred_energy_tax_weight = float(args.late_pred_energy_tax_weight)
    if getattr(args, "late_pred_energy_tax_cap", None) is not None:
        sim_cfg.late_pred_energy_tax_cap = float(args.late_pred_energy_tax_cap)
    if getattr(args, "late_pred_energy_tax_start_step", None) is not None:
        sim_cfg.late_pred_energy_tax_start_step = int(args.late_pred_energy_tax_start_step)

    # Backward-compat CLI aliases:
    # --late_pred_cost_cap / --late_pred_cost_start_step map to legacy energy tax fields.
    if args.late_pred_cost_cap is not None:
        sim_cfg.late_pred_energy_tax_cap = float(args.late_pred_cost_cap)
    if args.late_pred_cost_start_step is not None:
        sim_cfg.late_pred_energy_tax_start_step = int(args.late_pred_cost_start_step)
        sim_cfg.late_pred_cost_start_step = int(args.late_pred_cost_start_step)

    # Part B: adult movement penalty overrides
    if getattr(args, "adult_move_penalty_enable", False):
        sim_cfg.adult_move_penalty_enable = True
    if args.adult_move_penalty_start_step is not None:
        sim_cfg.adult_move_penalty_start_step = int(args.adult_move_penalty_start_step)
    if args.adult_pred_err_ema_alpha is not None:
        sim_cfg.adult_pred_err_ema_alpha = float(args.adult_pred_err_ema_alpha)
    if args.adult_move_penalty_scale is not None:
        sim_cfg.adult_move_penalty_scale = float(args.adult_move_penalty_scale)
    if args.adult_move_penalty_strength is not None:
        sim_cfg.adult_move_penalty_strength = float(args.adult_move_penalty_strength)
    if args.adult_move_penalty_cap is not None:
        sim_cfg.adult_move_penalty_cap = float(args.adult_move_penalty_cap)

    # Phase 4: identity drift overrides
    if getattr(args, "identity_drift_enable", False):
        sim_cfg.identity_drift_enable = True
    if args.identity_probe_every_k is not None:
        sim_cfg.identity_probe_every_k = int(args.identity_probe_every_k)
    if args.identity_drift_start_step is not None:
        sim_cfg.identity_drift_start_step = int(args.identity_drift_start_step)
    if args.identity_drift_scale is not None:
        sim_cfg.identity_drift_scale = float(args.identity_drift_scale)
    if args.identity_drift_strength is not None:
        sim_cfg.identity_drift_strength = float(args.identity_drift_strength)
    if args.identity_drift_cap is not None:
        sim_cfg.identity_drift_cap = float(args.identity_drift_cap)
    if args.identity_drift_ema_alpha is not None:
        sim_cfg.identity_drift_ema_alpha = float(args.identity_drift_ema_alpha)


    if args.injury_per_damage is not None:
        env_cfg.injury_per_damage = float(args.injury_per_damage)
    if args.injury_repro_block is not None:
        env_cfg.injury_repro_block = float(args.injury_repro_block)

    # Tier 1: spatial ecology overrides
    if getattr(args, "spatial_ecology_enable", False):
        env_cfg.spatial_ecology_enable = True
    if getattr(args, "spatial_ecology_seed", None) is not None:
        env_cfg.spatial_ecology_seed = int(args.spatial_ecology_seed)

    # Map specs: fail fast on malformed JSON (no silent fallback)
    try:
        if getattr(args, "resource_regrowth_map_spec", None):
            env_cfg.resource_regrowth_map_spec = _load_json_spec_arg(args.resource_regrowth_map_spec)
        if getattr(args, "resource_spawn_density_map_spec", None):
            env_cfg.resource_spawn_density_map_spec = _load_json_spec_arg(args.resource_spawn_density_map_spec)
        if getattr(args, "hazard_rate_map_spec", None):
            env_cfg.hazard_rate_map_spec = _load_json_spec_arg(args.hazard_rate_map_spec)
    except Exception as e:
        raise SystemExit(f"Invalid spatial ecology MapSpec: {e}")

    if getattr(args, "spatial_ecology_log_formats", None):
        env_cfg.spatial_ecology_log_formats = [s.strip().lower() for s in str(args.spatial_ecology_log_formats).split(",") if s.strip()]
    if getattr(args, "spatial_ecology_log_region_id_map", False):
        env_cfg.spatial_ecology_log_region_id_map = True
    if getattr(args, "spatial_ecology_no_log_stats", False):
        env_cfg.spatial_ecology_log_stats = False


    # Tier 2: spatial droughts overrides
    if getattr(args, "spatial_droughts_enable", False):
        env_cfg.spatial_droughts_enable = True
    try:
        if getattr(args, "spatial_droughts_spec", None):
            env_cfg.spatial_droughts_spec = _load_json_spec_arg(args.spatial_droughts_spec)
    except Exception as e:
        raise SystemExit(f"Invalid spatial droughts spec: {e}")

    if getattr(args, "spatial_droughts_log_timeseries_every", None) is not None:
        env_cfg.spatial_droughts_log_timeseries_every = int(args.spatial_droughts_log_timeseries_every)
    if getattr(args, "spatial_droughts_log_periodic_every", None) is not None:
        env_cfg.spatial_droughts_log_periodic_every = int(args.spatial_droughts_log_periodic_every)

    # === Tier 3 change: migration pressure overrides ===
    if getattr(args, "migration_pressure_enable", False):
        env_cfg.migration_pressure_enable = True

    # Load Tier 3 specs (JSON inline or @file). Fail fast on bad JSON.
    try:
        if getattr(args, "tier3_regions_spec", None):
            env_cfg.tier3_regions_spec = _load_json_spec_arg(args.tier3_regions_spec)
        if getattr(args, "tier3_regional_carrying_capacity", None):
            env_cfg.tier3_regional_carrying_capacity = _load_json_spec_arg(args.tier3_regional_carrying_capacity)
        if getattr(args, "tier3_movement_cost_map", None):
            env_cfg.tier3_movement_cost_map = _load_json_spec_arg(args.tier3_movement_cost_map)
        if getattr(args, "tier3_region_transition_risk", None):
            env_cfg.tier3_region_transition_risk = _load_json_spec_arg(args.tier3_region_transition_risk)
    except Exception as e:
        raise SystemExit(f"Invalid Tier 3 spec: {e}")

    if getattr(args, "tier3_validation_force_random_walk", False):
        env_cfg.tier3_validation_force_random_walk = True
    # === Tier 3 change: end migration pressure overrides ===

    # -----------------------------
    # Tier 4: juvenile spatial vulnerability overrides
    # -----------------------------
    if getattr(args, "tier4_juvenile_vulnerability_enable", False):
        env_cfg.tier4_juvenile_vulnerability_enable = True

    if getattr(args, "tier4_juvenile_steps", None) is not None:
        env_cfg.tier4_juvenile_steps = int(args.tier4_juvenile_steps)
    if getattr(args, "tier4_juvenile_grace_steps", None) is not None:
        env_cfg.tier4_juvenile_grace_steps = int(args.tier4_juvenile_grace_steps)

    if getattr(args, "tier4_juvenile_hazard_multiplier", None) is not None:
        env_cfg.tier4_juvenile_hazard_multiplier = float(args.tier4_juvenile_hazard_multiplier)
    if getattr(args, "tier4_juvenile_movement_penalty", None) is not None:
        env_cfg.tier4_juvenile_movement_penalty = float(args.tier4_juvenile_movement_penalty)
    if getattr(args, "tier4_juvenile_harvest_multiplier", None) is not None:
        env_cfg.tier4_juvenile_harvest_multiplier = float(args.tier4_juvenile_harvest_multiplier)

    # Optional comma-separated edges
    if getattr(args, "tier4_survival_bins_steps", None):
        try:
            edges = [int(s.strip()) for s in str(args.tier4_survival_bins_steps).split(",") if s.strip()]
            if len(edges) >= 2:
                env_cfg.tier4_survival_bins_steps = edges
        except Exception as e:
            raise SystemExit(f"Invalid --tier4_survival_bins_steps: {e}")

    if args.shock_start_prob is not None:
        sim_cfg.shock_start_prob = float(args.shock_start_prob)
    if args.shock_duration_min is not None:
        sim_cfg.shock_duration_min = int(args.shock_duration_min)
    if args.shock_duration_max is not None:
        sim_cfg.shock_duration_max = int(args.shock_duration_max)
    if args.shock_cooldown_min is not None:
        sim_cfg.shock_cooldown_min = int(args.shock_cooldown_min)
    if args.shock_cooldown_max is not None:
        sim_cfg.shock_cooldown_max = int(args.shock_cooldown_max)

    # Run directory
    runs_dir = os.path.abspath(args.runs_dir)

    # -----------------------------
    # Phase 1.5: Generic knob override layer (--knob section.field=value)
    # MUST happen after normal CLI flags, and BEFORE run_dir creation / RNG-seeding / derived config.
    applied_overrides, genome_overrides = apply_knob_overrides(
        knob_pairs=list(getattr(args, "knob", []) or []),
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
    )
    # Audit trail: keep as JSON-serializable fields on sim_cfg (persisted in config.json)
    sim_cfg.knob_overrides = list(applied_overrides)
    sim_cfg.genome_overrides = dict(genome_overrides)
    sim_cfg.plan_freeze_enforced = _plan_freeze_enforced_from_env()

    # Performance profiles are explicit implementation profiles. Apply them
    # after registry/generic knobs so accepted output-cost reductions cannot be
    # accidentally defeated by a registry axis encoded as --knob.
    profile_name = str(getattr(args, "performance_profile", "") or "").strip()
    profile_audit = apply_performance_profile(sim_cfg, profile_name)
    sim_cfg.performance_profile = str(profile_audit.get("profile", profile_name)) if profile_name else ""
    sim_cfg.performance_profile_audit = dict(profile_audit)
    policy_context = str(getattr(sim_cfg, "policy_architecture_context", "") or "").strip()
    if policy_context:
        if policy_context == "clean_control":
            pass
        elif policy_context == "crowding_existing":
            env_cfg.width = 8
            env_cfg.height = 8
        else:
            raise SystemExit(f"Unknown sim_cfg.policy_architecture_context={policy_context!r}")

    try:
        sim_cfg.knob_override_receipts = verify_knob_override_readback(
            applied_overrides=list(applied_overrides),
            env_cfg=env_cfg,
            pop_cfg=pop_cfg,
            sim_cfg=sim_cfg,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    run_id_raw = str(getattr(args, "run_id", "") or "").strip()
    run_id_explicit = bool(run_id_raw)
    run_id = run_id_raw
    if not run_id:
        run_id = _default_run_id(args)

    # NOTE: base_run_dir must be defined regardless of whether --run_id was provided.
    # Otherwise, allow_existing_run_dir / unique dir logic can crash when run_id is set.
    base_run_dir = os.path.join(runs_dir, run_id)
    if bool(getattr(args, 'allow_existing_run_dir', False)):
        run_dir = os.path.abspath(base_run_dir)
        os.makedirs(run_dir, exist_ok=True)
    else:
        # WebUI planned runs pre-create the directory skeleton and expect the engine
        # to write into that exact directory (no _N suffixing).
        if run_id_explicit and os.path.exists(base_run_dir) and _is_planned_skeleton_run_dir(base_run_dir):
            run_dir = os.path.abspath(base_run_dir)
        else:
            run_dir = _unique_run_dir(base_run_dir)
    os.makedirs(run_dir, exist_ok=True)
    _write_manifest_json(run_dir, args)

    # -----------------------------
    # Phase 5B/5D: seed ControlPlane inbox (orchestration-only)
    # -----------------------------
    seed_paths = list(getattr(args, "control_inbox_seed_json", []) or [])
    if seed_paths:
        inbox_dir = os.path.join(run_dir, "inbox")
        os.makedirs(inbox_dir, exist_ok=True)
        for sp in seed_paths:
            sp = str(sp)
            if not sp:
                continue
            src = os.path.abspath(sp)
            if not os.path.exists(src):
                raise SystemExit(f"--control_inbox_seed_json not found: {sp}")
            base = os.path.basename(src)
            dst = os.path.join(inbox_dir, base)
            if os.path.exists(dst):
                i = 1
                while True:
                    cand = os.path.join(inbox_dir, f"{base}.{i}")
                    if not os.path.exists(cand):
                        dst = cand
                        break
                    i += 1
            tmp_dst = dst + ".tmp"
            with open(src, "rb") as fsrc, open(tmp_dst, "wb") as fdst:
                fdst.write(fsrc.read())
            os.replace(tmp_dst, dst)

    # -----------------------------
    # Resume from checkpoint (Checkpoint v1)
    # -----------------------------
    resume_from = str(getattr(args, "resume_from", "") or "").strip()
    payload = None
    start_gen = 0
    if resume_from:
        resume_from = _resolve_resume_path(resume_from)
        payload = checkpoint_load(resume_from)

        # ------------------------------------------------------------
        # Resume dispatch:
        # - Checkpoint v1 provides config_snapshot + meta.json
        # - Checkpoint Bundle v2 provides metadata.json + canonical state files
        # ------------------------------------------------------------
        is_bundle_v2 = bool(isinstance(payload, dict) and (payload.get("_format") == "bundle_v2" or "metadata" in payload))

        if not is_bundle_v2:
            snap = (payload.get("config_snapshot", {}) or {})
            # Rebuild configs from checkpoint snapshot for determinism.
            env_cfg = _dataclass_from_snapshot(EnvConfig, snap.get("env_cfg", {}) or {})
            # If the user provides --grid_size while resuming, it must match the checkpoint,
            # otherwise the world topology changes and the checkpoint is invalid.
            if getattr(args, "grid_size", None) is not None:
                gs = int(args.grid_size)
                if gs <= 0:
                    raise SystemExit("--grid_size must be a positive integer")
                if int(getattr(env_cfg, "width", gs)) != gs or int(getattr(env_cfg, "height", gs)) != gs:
                    raise SystemExit(
                        f"Cannot resume with --grid_size {gs}: checkpoint env width/height is "
                        f"{getattr(env_cfg, 'width', '?')}x{getattr(env_cfg, 'height', '?')}."
                    )

            pop_cfg = _dataclass_from_snapshot(PopConfig, snap.get("pop_cfg", {}) or {})
            sim_cfg = _dataclass_from_snapshot(SimConfig, snap.get("sim_cfg", {}) or {})

            # Treat --gen as "additional generations to run" when resuming.
            sim_cfg.generations = int(getattr(args, "generations", sim_cfg.generations) or sim_cfg.generations)

            # Checkpointing is output policy, so CLI can override safely.
            sim_cfg.checkpoint_period_gen = int(getattr(args, "checkpoint_period_gen", 0) or 0)
            sim_cfg.checkpoint_rotate = int(getattr(args, "checkpoint_rotate", 0) or 0)

            _v2_period_arg = getattr(args, "checkpoint_bundle_v2_period_gen", None)
            if _v2_period_arg is None:
                sim_cfg.checkpoint_bundle_v2_period_gen = int(getattr(args, "checkpoint_period_gen", 0) or 0)
            else:
                sim_cfg.checkpoint_bundle_v2_period_gen = int(_v2_period_arg or 0)

            _legacy_subdir_arg = str(getattr(args, "checkpoint_subdir", "") or "").strip()
            if _legacy_subdir_arg:
                sim_cfg.checkpoint_subdir = _legacy_subdir_arg

            _v2_subdir_arg = str(getattr(args, "checkpoint_bundle_v2_subdir", "") or "").strip()
            if _v2_subdir_arg:
                sim_cfg.checkpoint_bundle_v2_subdir = _v2_subdir_arg
            elif _legacy_subdir_arg:
                sim_cfg.checkpoint_bundle_v2_subdir = _legacy_subdir_arg

            _strict_arg = getattr(args, "checkpoint_bundle_v2_strict_replay_mode", None)
            if _strict_arg is not None:
                try:
                    sim_cfg.checkpoint_bundle_v2_strict_replay_mode = bool(int(_strict_arg))
                except Exception:
                    sim_cfg.checkpoint_bundle_v2_strict_replay_mode = bool(_strict_arg)

            meta = payload.get("meta", {}) or {}
            sim_pos = meta.get("sim_position", {}) or {}
            ckpt_gen = sim_pos.get("generation", None)
            try:
                start_gen = int(ckpt_gen) + 1 if ckpt_gen is not None else 0
            except Exception:
                start_gen = 0
        else:
            # Bundle v2 does not currently include a full config_snapshot; keep the
            # configs built from CLI/defaults, and only restore simulation state.
            md = (payload.get("metadata", {}) or {})
            ckpt_gen = md.get("generation_index", None)
            # Bundle v2 metadata.generation_index is the *next* generation boundary
            # (i.e., the absolute generation index to start running from).
            # Do NOT add +1 here, or resume will skip a generation.
            try:
                start_gen = int(ckpt_gen) if ckpt_gen is not None else 0
            except Exception:
                start_gen = 0

            # Treat --gen as "additional generations to run" when resuming (output policy).
            try:
                sim_cfg.generations = int(getattr(args, "generations", sim_cfg.generations) or sim_cfg.generations)
            except Exception:
                pass

            # Checkpointing knobs are output policy.
            try:
                sim_cfg.checkpoint_period_gen = int(getattr(args, "checkpoint_period_gen", 0) or 0)
                sim_cfg.checkpoint_rotate = int(getattr(args, "checkpoint_rotate", 0) or 0)

                _v2_period_arg = getattr(args, "checkpoint_bundle_v2_period_gen", None)
                if _v2_period_arg is None:
                    sim_cfg.checkpoint_bundle_v2_period_gen = int(getattr(args, "checkpoint_period_gen", 0) or 0)
                else:
                    sim_cfg.checkpoint_bundle_v2_period_gen = int(_v2_period_arg or 0)

                _legacy_subdir_arg = str(getattr(args, "checkpoint_subdir", "") or "").strip()
                if _legacy_subdir_arg:
                    sim_cfg.checkpoint_subdir = _legacy_subdir_arg

                _v2_subdir_arg = str(getattr(args, "checkpoint_bundle_v2_subdir", "") or "").strip()
                if _v2_subdir_arg:
                    sim_cfg.checkpoint_bundle_v2_subdir = _v2_subdir_arg
                elif _legacy_subdir_arg:
                    sim_cfg.checkpoint_bundle_v2_subdir = _legacy_subdir_arg

                _strict_arg = getattr(args, "checkpoint_bundle_v2_strict_replay_mode", None)
                if _strict_arg is not None:
                    try:
                        sim_cfg.checkpoint_bundle_v2_strict_replay_mode = bool(int(_strict_arg))
                    except Exception:
                        sim_cfg.checkpoint_bundle_v2_strict_replay_mode = bool(_strict_arg)
            except Exception:
                pass

# Persist run config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "env_cfg": env_cfg.__dict__,
                "pop_cfg": pop_cfg.__dict__,
                "sim_cfg": sim_cfg.__dict__,
                "genome_cfg": {"overrides": dict(getattr(sim_cfg, "genome_overrides", {}) or {})},
                "resume_from": resume_from,
            },
            f,
            indent=2,
        )

    write_run_config_csv(run_dir, env_cfg, pop_cfg, sim_cfg)

    logger = MetricsLogger(run_dir)
    runner = SimulationRunner(env_cfg, pop_cfg, sim_cfg, logger, run_dir, perf=args.perf)
    ctx = RunContext(
        plan=plan,
        run_dir=str(run_dir),
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
        logger=logger,
        runner=runner,
        payload=payload,
        start_gen=int(start_gen),
        started_wall=float(_t0_wall),
        engine_logger=elog,
    )
    return ctx


def execute_run_context(ctx: RunContext) -> RunExecutionResult:
    """Execute one prebuilt live per-cell run context."""

    restore_torch_global_settings(ctx.execution_torch_global_settings)
    restore_global_rng_state(ctx.execution_global_rng_state)

    if str(getattr(ctx.sim_cfg, "execution_route_identity", "")) == "tick_native_execution_route_v1":
        from simulation.tick_native_route import execute_tick_native_reference_v1

        if ctx.payload is not None:
            from simulation.checkpoint_bundle_v2 import validate_inheritance_checkpoint_compatibility

            validate_inheritance_checkpoint_compatibility(ctx.runner, ctx.payload)
            checkpoint_apply(ctx.runner, ctx.payload, restore_rng=True, restore_banks=True)
            execute_tick_native_reference_v1(
                ctx.runner, resume_bundle_dir=str(getattr(ctx.plan.args, "resume_from", ""))
            )
        else:
            execute_tick_native_reference_v1(ctx.runner)
    elif ctx.payload is not None:
        from simulation.checkpoint_bundle_v2 import validate_inheritance_checkpoint_compatibility

        # This compatibility gate is unconditional: repaired mechanics must
        # never be silently crossed by disabling general strict replay.
        validate_inheritance_checkpoint_compatibility(ctx.runner, ctx.payload)
        # Phase 3 determinism: mark resume mode for optional debug tooling.
        try:
            setattr(ctx.runner, "_is_resume_run", True)
        except Exception:
            pass
        checkpoint_apply(ctx.runner, ctx.payload, restore_rng=True, restore_banks=True)
        ctx.runner.run(start_gen=ctx.start_gen, skip_init=True)
    else:
        try:
            setattr(ctx.runner, "_is_resume_run", False)
        except Exception:
            pass
        ctx.runner.run()

    _write_policy_family_head_runtime_summaries(ctx.run_dir, ctx.runner)

    try:
        if ctx.engine_logger is not None:
            from emergence.obs_logging import log_event  # noqa: WPS433
            log_event(
                ctx.engine_logger,
                event="ENGINE_END",
                status="complete",
                elapsed_s=round(time.time() - ctx.started_wall, 3),
                run_dir=str(ctx.run_dir),
            )
    except Exception:
        pass


    print(f"Run complete. Metrics written to: {os.path.join(ctx.run_dir, 'metrics.csv')}")
    if ctx.sim_cfg.devcoupling_enable:
        print(f"Developmental coupling written to: {os.path.join(ctx.run_dir, 'devcoupling.csv')}")
    return RunExecutionResult(
        status="complete",
        returncode=0,
        run_dir=str(ctx.run_dir),
        metrics_path=os.path.join(ctx.run_dir, "metrics.csv"),
        run_config_path=os.path.join(ctx.run_dir, "run_config.csv"),
        manifest_path=os.path.join(ctx.run_dir, "manifest.json"),
        wall_seconds=round(time.time() - ctx.started_wall, 3),
    )


def _write_policy_family_head_runtime_summaries(run_dir: str, runner: Any) -> None:
    payload: dict[str, Any] = {
        "schema_version": "policy_family_head_runtime_summaries_v1",
        "authority": getattr(runner, "_policy_family_head_residual_authority_total", None),
        "cost": getattr(runner, "_policy_family_head_residual_cost_total", None),
        "choice_margin": getattr(runner, "_policy_family_head_authority_choice_margin_total", None),
        "magnitude_surface": getattr(runner, "_policy_family_head_authority_magnitude_surface_total", None),
        "opportunity_conditioned_candidates": getattr(runner, "_policy_family_head_opportunity_conditioned_candidate_total", None),
    }
    if all(value is None for key, value in payload.items() if key != "schema_version"):
        return
    path = os.path.join(str(run_dir), "policy_family_head_runtime_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def execute_run_plan(plan: RunPlan) -> RunExecutionResult:
    """Build and execute one resolved single-cell run plan."""

    return execute_run_context(build_run_context_from_plan(plan))


def main(argv: Sequence[str] | None = None) -> int:
    if not os.environ.get("EIEL_FLAT_OUTPUT_COST_SIDECAR"):
        plan = build_run_plan_from_argv(argv)
        result = execute_run_plan(plan)
        return int(result.returncode)

    from simulation.output_cost_timing import (  # noqa: WPS433
        begin_flat_output_cost_timing,
        finalize_flat_output_cost_timing,
    )

    timing = begin_flat_output_cost_timing()
    ctx: RunContext | None = None
    run_error: BaseException | None = None
    try:
        plan = build_run_plan_from_argv(argv)
        ctx = build_run_context_from_plan(plan)
        setattr(ctx.runner, "_flat_output_cost_timing", timing)
        result = execute_run_context(ctx)
        return int(result.returncode)
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        finalize_flat_output_cost_timing(
            timing,
            run_dir=(ctx.run_dir if ctx is not None else None),
            run_complete=run_error is None,
            error=run_error,
        )
if __name__ == "__main__":
    raise SystemExit(main())
