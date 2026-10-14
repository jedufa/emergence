"""experiments.harness.run_experiments

Experiment Registry & Automated Validation Harness (v1)

This module provides a boring, reproducible runner for experiment registries.

Key behaviors (v1):
- Reads a declarative YAML registry.
- Expands the full matrix deterministically using registry.matrix_order.
- Executes the simulation via `scripts/utils/run_evo.py`.
- Writes batch-level provenance (registry hash, git commit, environment info).
- Prevents accidental overwrite of batches (unless --overwrite is passed).
- Provides --dry_run to preview commands without executing.
- Performs minimal post-run correctness checks (exit code + required artifacts).

Non-goals (v1):
- No distributed compute.
- No dashboards.
- No agent inspection.
- No adaptive tuning.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None

try:
    from experiments.projects.emergence.experiments.harness.schema import ExperimentRegistry, load_and_validate_registry
    from experiments.projects.emergence.experiments.harness.output_storage import (
        build_storage_plan,
        campaign_output_profile_knobs,
        validate_output_profile,
        write_storage_plan,
        write_storage_summary,
    )
except ModuleNotFoundError:
    from .schema import ExperimentRegistry, load_and_validate_registry
    from .output_storage import (
        build_storage_plan,
        campaign_output_profile_knobs,
        validate_output_profile,
        write_storage_plan,
        write_storage_summary,
    )

from experiments_shared.runner_utils import default_project_runs_root, normalize_project_runs_root

_STE_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "engine"
    / "native_runtime"
    / "scientific_throughput_efficiency_v1.py"
)
_STE_SPEC = importlib.util.spec_from_file_location(
    "eiel_scientific_throughput_efficiency_v1",
    _STE_MODULE_PATH,
)
if _STE_SPEC is None or _STE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to load Scientific Throughput Efficiency module: {_STE_MODULE_PATH}")
_STE_MODULE = importlib.util.module_from_spec(_STE_SPEC)
_STE_SPEC.loader.exec_module(_STE_MODULE)
aggregate_scientific_throughput_efficiency_reports = (
    _STE_MODULE.aggregate_scientific_throughput_efficiency_reports
)
build_cpu_scientific_throughput_efficiency_report = (
    _STE_MODULE.build_cpu_scientific_throughput_efficiency_report
)
collect_run_scientific_throughput_efficiency_reports = (
    _STE_MODULE.collect_run_scientific_throughput_efficiency_reports
)
scientific_throughput_efficiency_enabled = (
    _STE_MODULE.scientific_throughput_efficiency_enabled
)
write_scientific_throughput_efficiency_report_files = (
    _STE_MODULE.write_scientific_throughput_efficiency_report_files
)


def _write_harness_decision_debug(
    *,
    batch_dir: Path,
    verdict: str,
    exit_code: int,
    reasons: List[Dict[str, Any]],
    inputs: Dict[str, Any],
    pointers: Dict[str, Any],
    fail_summary: str = "",
) -> None:
    """Write a decision_debug.json sibling to summary.json (best-effort).

    This is intentionally light-weight: the harness does not understand validator
    semantics; it only records its own decision points.
    """
    try:
        from emergence.decision_debug import load_debug_ctx_from_env, ensure_packet, add_layer_decision, write_json  # noqa: WPS433

        base = load_debug_ctx_from_env()
        pkt = ensure_packet(base)
        add_layer_decision(
            pkt,
            layer="harness",
            verdict=str(verdict),
            exit_code=int(exit_code),
            reasons=list(reasons or []),
            inputs=dict(inputs or {}),
            pointers=dict(pointers or {}),
            fail_summary=str(fail_summary or ""),
        )
        write_json(batch_dir / "decision_debug.json", pkt)
    except Exception:
        return


# -----------------------------
# Helpers
# -----------------------------

HARNESS_VERSION = "0.1.0"
HARNESS_API = 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_emergence_runs_root(raw: str) -> str:
    env_root = str(os.environ.get("EMERGENCE_RUNS_DIR") or "").strip()
    if env_root:
        return env_root
    text = str(raw or "").strip()
    if not text or text == "runs":
        return str(default_project_runs_root("emergence"))
    return str(
        normalize_project_runs_root(
            repo_root=_repo_root(),
            project_root=_project_root(),
            project_id="emergence",
            raw=text,
        )
    )
# -----------------------------
# Console styling (ANSI)
# -----------------------------

class Ansi:
    """ANSI escape codes for terminal coloring.

    Notes:
        - Works on most Unix terminals + WSL.
        - Keep color usage sparse and meaningful (structure, focus, status).
    """
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


def _c(s: str, *codes: str, enable: bool = True) -> str:
    """Colorize a string with ANSI codes.

    Args:
        s: Text to wrap.
        *codes: ANSI codes (e.g., Ansi.CYAN, Ansi.BOLD).
        enable: If False, returns s unchanged.

    Returns:
        Styled string (or original if disabled).
    """
    if not enable or not codes:
        return s
    return "".join(codes) + s + Ansi.RESET


def _use_color() -> bool:
    """Return True if stdout looks like an interactive terminal."""
    try:
        return bool(getattr(sys.stdout, "isatty", lambda: False)())
    except Exception:
        return False


# Default runtime heuristic for wall-clock estimation (seconds per generation).
# This is intentionally blunt. It is for log readability, not scientific measurement.
EST_SECONDS_PER_GEN_DEFAULT = 3.5


def _cli_get_value(cli: List[str], flag_names: Sequence[str]) -> Optional[str]:
    """Get the value following the first matching flag in a CLI token list.

    Args:
        cli: Flat list of CLI tokens.
        flag_names: Candidate flag names (e.g., ["--generations", "--gen"]).

    Returns:
        The string value after the flag, or None if not found or malformed.
    """
    for i, tok in enumerate(cli):
        if tok in flag_names:
            if i + 1 < len(cli):
                return str(cli[i + 1])
            return None
    return None


def _axis_values_display(values: List[str]) -> str:
    """Pretty-print axis values for the harness plan.

    Notes:
        YAML booleans often become Python bools which are then str()-coerced to
        "True"/"False". For human readability, we display these as lowercase
        booleans without quotes.

    Args:
        values: Axis values as strings.

    Returns:
        A compact display string, e.g. "[false, true]" or "[off, on]".
    """
    norm: List[str] = []
    for v in values:
        if v == "True":
            norm.append("true")
        elif v == "False":
            norm.append("false")
        else:
            norm.append(v)
    return "[" + ", ".join(norm) + "]"


def _print_harness_plan(
        *,
        registry_path: Path,
        reg: Any,
        resolved: List[Dict[str, Any]],
        current_run_index: Optional[int] = None,
    ) -> None:
    
    
    """Print a short, human-readable plan for the upcoming harness run.

    Args:
        registry_path: Path to the registry YAML.
        reg: Normalized ExperimentRegistry (schema.py).
        resolved: Fully expanded run list (seed × axes).
    """
    # Derive “defaults” from the first resolved CLI, when available.
    gens = None
    steps = None
    pop = None
    stride = None

    if resolved:
        cli0 = list(resolved[0].get("cli", []))
        gens = _cli_get_value(cli0, ["--generations", "--gen"])
        steps = _cli_get_value(cli0, ["--steps"])
        pop = _cli_get_value(cli0, ["--pop", "--pop_size"])
        stride = _cli_get_value(cli0, ["--policy_tick_stride"])

    # Estimate wall-clock duration using a blunt constant.
    # If gens is unknown, we can’t estimate.
    est_total_seconds = None
    if gens is not None:
        try:
            est_total_seconds = float(gens) * float(EST_SECONDS_PER_GEN_DEFAULT) * float(len(resolved))
        except Exception:
            est_total_seconds = None

    now = datetime.datetime.now()
    eta_str = "(unknown)"
    end_str = "(unknown)"
    if est_total_seconds is not None:
        eta = datetime.timedelta(seconds=int(est_total_seconds))
        end = now + eta
        eta_str = str(eta)
        end_str = end.strftime("%Y-%m-%d %H:%M:%S")

    use_color = _use_color()
    print("\n" + _c("===============================================================================", Ansi.CYAN, Ansi.BOLD, enable=use_color), flush=True)
    print("\n" + _c("=== HARNESS PLAN (v1) ===", Ansi.CYAN, Ansi.BOLD, enable=use_color), flush=True)
    print(f"{_c('Registry:', Ansi.CYAN, Ansi.BOLD, enable=use_color)} {registry_path}", flush=True)
    print(f"{_c('Name:', Ansi.CYAN, Ansi.BOLD, enable=use_color)} {getattr(reg, 'name', '(unknown)')} | {_c('Version:', Ansi.CYAN, Ansi.BOLD, enable=use_color)} {getattr(reg, 'version', '(unknown)')}", flush=True)

    if current_run_index is not None and resolved:
        width = len(str(len(resolved)))
        print(
            f"Current run: [{int(current_run_index):0{width}d}/{len(resolved):0{width}d}]",
            flush=True,
        )

    desc = getattr(reg, "description", "") or ""
    if desc:
        print(f"{_c('Description:', Ansi.CYAN, Ansi.BOLD, enable=use_color)} {desc}", flush=True)

    intent = getattr(reg, "intent", "") or ""
    if intent:
        print(f"{_c('Question:', Ansi.CYAN, Ansi.BOLD, enable=use_color)} {intent}", flush=True)

    unique_configs = 1
    try:
        unique_configs = int(len(resolved) / max(1, len(getattr(reg, "seeds", []))))
    except Exception:
        unique_configs = 1

#    print(f"Planned runs: {len(resolved)} | Unique configs: {unique_configs} | Seeds: {len(getattr(reg, 'seeds', []))}",flush=True,)
    print(f"{_c('Planned runs:', Ansi.CYAN, Ansi.BOLD, enable=use_color)}  " + f"{len(resolved)} | Unique configs: {unique_configs} | Seeds: {len(getattr(reg, 'seeds', []))}",flush=True,)

    # Defaults line
    defaults_bits = []
    if gens is not None:
        defaults_bits.append(f"generations={gens}")
    if steps is not None:
        defaults_bits.append(f"steps={steps}")
    if pop is not None:
        defaults_bits.append(f"pop={pop}")
    if stride is not None:
        defaults_bits.append(f"policy_tick_stride={stride}")

    if defaults_bits:
        print(f"{_c('Defaults:', Ansi.CYAN, Ansi.BOLD, enable=use_color)} " + ", ".join(defaults_bits), flush=True)

    if est_total_seconds is not None:
        print(
            f"{_c('Estimate:', Ansi.CYAN, Ansi.BOLD, enable=use_color)} total≈{eta_str} (assumes {EST_SECONDS_PER_GEN_DEFAULT:.1f}s/gen) | end≈{end_str}",
            flush=True,
        )

    # Axes summary
    print(f"{_c('Axes:', Ansi.CYAN, Ansi.BOLD, enable=use_color)}", flush=True)
    for ax in getattr(reg, "axes", []):
        print(f"  - {ax.name}: {_axis_values_display(list(ax.values))}", flush=True)

    # Run list summary (include gens/steps and seed)
    print(f"{_c('Run list:', Ansi.CYAN, Ansi.BOLD, enable=use_color)}", flush=True)
    width = len(str(len(resolved))) if resolved else 1
    for i, row in enumerate(resolved, start=1):
        seed = row.get("seed")
        axes = row.get("axes", {})
        axes_str = ", ".join([f"{k}={v}" for k, v in axes.items()])
        g = gens
        s = steps
        # If overrides exist per run, detect them from per-run cli.
        cli = list(row.get("cli", []))
        g2 = _cli_get_value(cli, ["--generations", "--gen"])
        s2 = _cli_get_value(cli, ["--steps"])
        if g2 is not None:
            g = g2
        if s2 is not None:
            s = s2
        extras = []
        if g is not None:
            extras.append(f"gen={g}")
        if s is not None:
            extras.append(f"steps={s}")
        extras_str = (" | " + ", ".join(extras)) if extras else ""
        marker = ">>" if (current_run_index is not None and int(current_run_index) == int(i)) else "  "
        print(
            f"{marker} [{i:0{width}d}/{len(resolved):0{width}d}] seed={seed} | {axes_str}{extras_str}",
            flush=True,
        )

    print("\n" + _c("===============================================================================", Ansi.CYAN, Ansi.BOLD, enable=use_color), flush=True)


def _utc_timestamp() -> str:
    """Return UTC timestamp string suitable for folder names."""
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _rand_suffix(n: int = 6) -> str:
    """Generate a short random hex suffix."""
    return "".join(random.choice("0123456789abcdef") for _ in range(n))


def _sha256_bytes(b: bytes) -> str:
    """Compute sha256 hex digest of bytes."""
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _short_token(text: str, n: int = 10) -> str:
    """Return a stable short hex token for a string."""
    raw = str(text or "").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[: max(4, int(n))]


def _compact_runtime_run_id(*, logical_run_id: str, ordinal: int) -> str:
    """Build compact deterministic runtime run_id for filesystem paths."""
    return f"r{int(ordinal):04d}_{_short_token(str(logical_run_id), 8)}"


def _enforce_rel_path_budget(*, base_root: Path, path: Path, label: str, max_rel_chars: int = 180) -> None:
    """Fail fast when a path under base_root exceeds relative-path budget."""
    try:
        rel = str(path.resolve().relative_to(base_root.resolve())).replace("\\", "/")
    except Exception:
        rel = str(path).replace("\\", "/")
    if len(rel) > int(max_rel_chars):
        raise RuntimeError(
            f"RUN_PATH_TOO_LONG: {label} relative path exceeds {int(max_rel_chars)} chars ({len(rel)}): {rel}"
        )


def _run_git(args: List[str]) -> Tuple[int, str]:
    """Run a git command if available.

    Returns (returncode, stdout_str).
    """
    try:
        p = subprocess.run(["git"] + args, check=False, capture_output=True, text=True)
        out = (p.stdout or "").strip()
        if not out:
            out = (p.stderr or "").strip()
        return int(p.returncode), out
    except Exception:
        return 1, ""


def _git_state(repo_root: Path) -> Dict[str, Any]:
    """Capture git commit + dirty info.

    Args:
        repo_root: Repository root.

    Returns:
        Dict[str, Any]: Git metadata.
    """
    cwd = os.getcwd()
    try:
        os.chdir(str(repo_root))
        rc, sha = _run_git(["rev-parse", "HEAD"])
        commit = sha if rc == 0 else ""

        rc2, shortsha = _run_git(["rev-parse", "--short", "HEAD"])
        short_commit = shortsha if rc2 == 0 else ""

        rc3, branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        branch_name = branch if rc3 == 0 else ""

        rc4, status = _run_git(["status", "--porcelain"])
        dirty = bool(status.strip()) if rc4 == 0 else False

        return {
            "commit": commit,
            "short_commit": short_commit,
            "branch": branch_name,
            "dirty": dirty,
        }
    finally:
        os.chdir(cwd)


def _write_git_diff_if_dirty(repo_root: Path, out_path: Path) -> Optional[Dict[str, Any]]:
    """If git repo is dirty, write a patch file and return info.

    Args:
        repo_root: Repository root.
        out_path: Where to write the patch.

    Returns:
        Optional[Dict[str, Any]]: Patch presence + hash, or None if not written.
    """
    cwd = os.getcwd()
    try:
        os.chdir(str(repo_root))
        rc, status = _run_git(["status", "--porcelain"])
        if rc != 0 or not status.strip():
            return None

        # Get a unified diff (includes staged+unstaged).
        # First: unstaged
        p1 = subprocess.run(["git", "diff"], check=False, capture_output=True)
        # Then staged
        p2 = subprocess.run(["git", "diff", "--staged"], check=False, capture_output=True)
        diff_bytes = (p1.stdout or b"") + (b"\n\n# --- STAGED DIFF ---\n\n") + (p2.stdout or b"")

        out_path.write_bytes(diff_bytes)
        return {
            "path": out_path.name,
            "sha256": _sha256_bytes(diff_bytes),
            "bytes": int(len(diff_bytes)),
        }
    except Exception:
        return None
    finally:
        os.chdir(cwd)


def _key_lib_versions() -> Dict[str, str]:
    """Collect versions for libraries that may affect determinism."""
    versions: Dict[str, str] = {}
    for pkg in ["numpy", "torch", "numba", "yaml"]:
        try:
            mod = __import__(pkg)
            v = getattr(mod, "__version__", "")
            if v:
                versions[pkg] = str(v)
        except Exception:
            pass
    return versions


def _thread_env_snapshot() -> Dict[str, str]:
    """Capture thread-related env vars that commonly change determinism/perf."""
    keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
    ]
    out: Dict[str, str] = {}
    for k in keys:
        if k in os.environ:
            out[k] = str(os.environ.get(k, ""))
    return out


def _format_run_id(fmt: str, ctx: Dict[str, Any]) -> str:
    """Format a run_id using a Python format string."""
    try:
        return fmt.format(**ctx)
    except Exception as e:
        raise ValueError(f"Failed to format run_id with format='{fmt}'. ctx={ctx}. error={e}")


def _format_run_tag(fmt: str, ctx: Dict[str, Any]) -> str:
    """Format a run_tag using a Python format string."""
    try:
        return fmt.format(**ctx)
    except Exception as e:
        raise ValueError(f"Failed to format run_tag with format='{fmt}'. ctx={ctx}. error={e}")


def _flatten_cli_defaults(cli_defaults: List[List[str]]) -> List[str]:
    """Flatten list-of-lists into a single arg list."""
    out: List[str] = []
    for row in cli_defaults:
        out.extend([str(x) for x in row])
    return out


def _ensure_empty_dir(path: Path, overwrite: bool) -> None:
    """Create directory, refusing to overwrite unless allowed."""
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to write into existing directory: {path} (use --overwrite to force)")
    path.mkdir(parents=True, exist_ok=True)


def _artifact_check(run_dir: Path, expected: List[str]) -> Tuple[bool, List[str]]:
    """Verify required artifacts exist.

    Returns:
        (ok, missing_list)
    """
    missing: List[str] = []
    for rel in expected:
        p = run_dir / rel
        if not p.exists():
            missing.append(rel)
    return (len(missing) == 0), missing


# -----------------------------
# Matrix expansion
# -----------------------------


def _axis_by_name(reg: ExperimentRegistry) -> Dict[str, Any]:
    """Convenience mapping from axis name to RegistryAxis."""
    return {a.name: a for a in reg.axes}


def expand_matrix(reg: ExperimentRegistry) -> List[Dict[str, Any]]:
    """Expand the full matrix deterministically.

    Each row dict contains:
    - axis values (labels)
    - seed (int)
    """
    axis_map = _axis_by_name(reg)

    # Build ordered list of axes (excluding seed) per matrix_order.
    ordered_axes = [axis_map[name] for name in reg.matrix_order if name != "seed"]

    rows: List[Dict[str, Any]] = []

    def rec(i: int, ctx: Dict[str, Any]) -> None:
        if i >= len(ordered_axes):
            for s in reg.seeds:
                row = dict(ctx)
                row["seed"] = int(s)
                rows.append(row)
            return
        ax = ordered_axes[i]
        for v in ax.values:
            ctx2 = dict(ctx)
            ctx2[ax.name] = v
            rec(i + 1, ctx2)

    rec(0, {})
    return rows


# -----------------------------
# Main runner
# -----------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI args for harness."""
    p = argparse.ArgumentParser(description="Experiment registry harness (v1)")
    p.add_argument("--registry", type=str, required=True, help="Path to registry YAML file")
    p.add_argument(
        "--runs_root",
        type=str,
        default="runs",
        help="Root runs directory (default: normalized project heavy-data runs root)",
    )
    p.add_argument("--batch_tag", type=str, default="", help="Optional extra tag appended to batch folder")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into an existing batch directory")
    p.add_argument("--dry_run", action="store_true", help="Expand and write artifacts, but do not execute")
    p.add_argument("--print_cmds", action="store_true", help="Print resolved CLI commands")
    p.add_argument(
        "--knob",
        action="append",
        default=[],
        help=(
            "Pass-through simulation knob override to all child runs (repeatable). "
            "Format: root.field=value (e.g., sim_cfg.checkpoint_bundle_v2_period_gen=10). "
            "This is appended to each resolved child simulation CLI as: --knob <value>."
        ),
    )
    p.add_argument(
        "--perf",
        action="store_true",
        help="Enable per-generation perf bucket timing (writes perf_buckets.csv under each run_dir). "
             "Also propagates --perf to child simulation runs (equivalent to setting EMERGENCE_PERF=1).",
    )
    p.add_argument(
        "--performance_profile",
        "--performance-profile",
        dest="performance_profile",
        type=str,
        default="",
        help="Apply a named simulation performance profile to every child run after generic --knob overrides.",
    )
    p.add_argument(
        "--campaign_output_profile",
        "--campaign-output-profile",
        dest="campaign_output_profile",
        type=str,
        default="compact_default",
        help=(
            "Apply a named campaign output/storage profile after registry and generic knobs. "
            "Profiles: compact_default, compact_funnel, fast_scan, repeat_parity, debug_event_trace, full_replay."
        ),
    )
    p.add_argument(
        "--allow_large_output",
        "--allow-large-output",
        dest="allow_large_output",
        action="store_true",
        help="Acknowledge large debug/full-replay output profiles during storage preflight.",
    )
    p.add_argument(
        "--allow_generation_checkpointing",
        "--allow-generation-checkpointing",
        dest="allow_generation_checkpointing",
        action="store_true",
        help="Explicitly allow checkpoint_bundle_v2_period_gen=1 when using full_replay.",
    )

    # Selection (optional): run only a subset of expanded rows.
    # This is used by Phase 2.5 manifest expansion tools that want one case per wrapper job.
    p.add_argument(
        "--only_run_id",
        action="append",
        default=[],
        help="Run only the specified run_id (repeatable). If omitted, runs all cases.",
    )
    p.add_argument(
        "--only_seed",
        action="append",
        default=[],
        help="Run only the specified seed(s) (repeatable). If omitted, uses registry seeds.",
    )

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the harness.

    Returns:
        int: process exit code (0 if all runs ok).
    """
    args = _parse_args(argv)

    args.runs_root = _normalize_emergence_runs_root(str(getattr(args, "runs_root", "runs")))

    # ------------------------------------------------------------
    # Observability: harness/test logging (console + JSONL)
    # ------------------------------------------------------------
    try:
        from emergence.obs_logging import setup_logging, get_logger, log_event  # noqa: WPS433 (repo-local import)
        log_dir = os.environ.get("EMERG_LOG_DIR", "").strip()
        if not log_dir:
            # Fallback: colocate under runs_root so local runs still produce logs.
            log_dir = str(Path(args.runs_root).expanduser().resolve() / "logs")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        setup_logging(scope="tests", jsonl_path=str(Path(log_dir) / "tests.jsonl"), console_prefix="[TESTS] ")
        tlog = get_logger("emergence.tests.harness")
        log_event(tlog, event="HARNESS_START", runs_root=str(args.runs_root), argv=" ".join(sys.argv))
    except Exception:
        tlog = None  # type: ignore[assignment]


    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to run the experiment harness. Install with: python -m pip install pyyaml"
        )


    # Repo root is the parent of this file's parent 'experiments' folder.
    repo_root = Path(__file__).resolve().parents[2]

    # Resolve registry path relative to repo_root (NOT cwd).
    registry_arg = Path(args.registry).expanduser()
    if registry_arg.is_absolute():
        registry_path = registry_arg.resolve()
    else:
        registry_path = (repo_root / registry_arg).resolve()

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    raw_bytes = registry_path.read_bytes()
    registry_hash = _sha256_bytes(raw_bytes)[:12]

    doc = yaml.safe_load(raw_bytes)
    if not isinstance(doc, dict):
        raise ValueError("Registry YAML must parse into a mapping/dict.")

    reg = load_and_validate_registry(doc)
    if reg.harness_api != HARNESS_API:
        raise ValueError(f"Registry harness_api={reg.harness_api} but harness supports HARNESS_API={HARNESS_API}.")

    git = _git_state(repo_root)
    git_sha_short = git.get("short_commit", "") or "nogit"

    ts = _utc_timestamp()
    batch_id = f"b{ts}_{_rand_suffix(4)}"
    if str(args.batch_tag).strip():
        batch_id = f"{batch_id}_t{_short_token(str(args.batch_tag).strip(), 6)}"

    runs_root = Path(args.runs_root).expanduser().resolve()
    exp_token = _short_token(f"{reg.name}|{reg.version}|{registry_hash}", 12)

    batch_dir = runs_root / "e" / exp_token / batch_id

    _ensure_empty_dir(batch_dir, overwrite=args.overwrite)
    sim_runs_dir = batch_dir / "r"
    sim_runs_dir.mkdir(parents=True, exist_ok=True)
    _enforce_rel_path_budget(base_root=runs_root, path=batch_dir, label="batch_dir")

    # Write manifest
    manifest_path = batch_dir / "manifest.json"
    matrix_csv_path = batch_dir / "matrix.csv"
    summary_path = batch_dir / "summary.json"
    storage_plan_path = batch_dir / "storage_plan.json"
    storage_summary_path = batch_dir / "storage_summary.json"
    campaign_output_profile = validate_output_profile(getattr(args, "campaign_output_profile", "compact_default"))
    campaign_started = time.perf_counter()

    manifest: Dict[str, Any] = {
        "harness": {
            "version": HARNESS_VERSION,
            "api": HARNESS_API,
        },
        "batch": {
            "id": batch_id,
            "created_at_utc": ts,
            "runs_root": str(runs_root),
            "batch_dir": str(batch_dir),
            "sim_runs_dir": str(sim_runs_dir),
        },
        "registry": {
            "path": str(registry_path),
            "sha256": _sha256_bytes(raw_bytes),
            "sha256_short": registry_hash,
            "name": reg.name,
            "version": reg.version,
            "description": reg.description,
        },
        "git": git,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "lib_versions": _key_lib_versions(),
            "thread_env": _thread_env_snapshot(),
        },
        "argv": {
            "harness_argv": sys.argv,
        },
        "campaign_output": {
            "profile": campaign_output_profile,
            "allow_large_output": bool(getattr(args, "allow_large_output", False)),
            "allow_generation_checkpointing": bool(getattr(args, "allow_generation_checkpointing", False)),
        },
    }

    # If dirty, write a diff patch.
    diff_info = None
    if bool(git.get("dirty", False)):
        diff_info = _write_git_diff_if_dirty(repo_root, batch_dir / "manifest_git_diff.patch")
    if diff_info is not None:
        manifest["git"]["diff_patch"] = diff_info

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # Expand matrix
    rows = expand_matrix(reg)

    # Optional selection filter (used by wrapper jobs that run a single case)
    only_run_ids = {str(x).strip() for x in (getattr(args, "only_run_id", []) or []) if str(x).strip()}
    only_seeds = set()
    try:
        for x in (getattr(args, "only_seed", []) or []):
            xs = str(x).strip()
            if xs:
                only_seeds.add(int(xs))
    except Exception:
        raise ValueError("--only_seed must be an integer (repeatable)")

    # Prepare per-run commands
    base_cli = _flatten_cli_defaults(reg.cli_defaults)
    performance_profile = str(getattr(args, "performance_profile", "") or "").strip()
    profile_passthrough: List[str] = []
    if performance_profile:
        profile_passthrough.extend(["--performance_profile", performance_profile])
    output_profile_knobs = campaign_output_profile_knobs(
        profile=campaign_output_profile,
        allow_large_output=bool(getattr(args, "allow_large_output", False)),
        allow_generation_checkpointing=bool(getattr(args, "allow_generation_checkpointing", False)),
    )
    output_profile_passthrough: List[str] = []
    for knob in output_profile_knobs:
        output_profile_passthrough.extend(["--knob", knob])

    # Optional: append global knob overrides to every child simulation run.
    knob_passthrough: List[str] = []
    try:
        for k in (getattr(args, "knob", []) or []):
            ks = str(k).strip()
            if ks:
                knob_passthrough.extend(["--knob", ks])
    except Exception:
        raise ValueError("--knob must be provided as strings in the form root.field=value (repeatable)")

    matrix_rows_for_csv: List[Dict[str, Any]] = []
    # Requested either via harness CLI (--perf) or via environment (EMERGENCE_PERF=1).
    try:
        perf_env = (os.environ.get("EMERGENCE_PERF") or "").strip()
        perf_on = bool(getattr(args, "perf", False)) or (perf_env in ("1", "true", "True", "yes", "YES"))
        if perf_on and "--perf" not in base_cli:
            base_cli.append("--perf")
    except Exception:
        # Never fail the harness over instrumentation flags.
        pass


    resolved: List[Dict[str, Any]] = []

    axis_map = _axis_by_name(reg)

    for row_idx, r in enumerate(rows, start=1):
        ctx: Dict[str, Any] = {
            "name": reg.name,
            "run_tag_prefix": reg.naming.run_tag_prefix,
        }
        # axis labels
        for ax in reg.axes:
            ctx[ax.name] = r[ax.name]
        seed = int(r["seed"])
        ctx["seed"] = seed

        run_id = _format_run_id(reg.naming.run_id_format, ctx)

        # Selection filters
        if only_seeds and seed not in only_seeds:
            continue
        if only_run_ids and str(run_id) not in only_run_ids:
            continue
        run_tag = _format_run_tag(reg.naming.run_tag_format, ctx)
        runtime_run_id = _compact_runtime_run_id(logical_run_id=str(run_id), ordinal=int(row_idx))

        # Refuse to collide with existing run dirs unless overwrite.
        run_dir = sim_runs_dir / runtime_run_id
        _enforce_rel_path_budget(base_root=runs_root, path=run_dir, label=f"run_dir:{run_id}")
        if run_dir.exists() and not args.overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir} (use --overwrite to force)")

        cli: List[str] = []
        cli.extend(base_cli)
        cli.extend(profile_passthrough)

        # axis flags in registry order (excluding seed)
        for name in reg.matrix_order:
            if name == "seed":
                continue
            ax = axis_map[name]
            v = r[name]
            cli.extend([str(x) for x in ax.flags[v]])

        # Global knob overrides (applied to all runs in this harness batch).
        # Important: place before --seed/--run_id so downstream parsers see a conventional ordering.
        if knob_passthrough:
            cli.extend(list(knob_passthrough))
        if output_profile_passthrough:
            # Campaign output policy is applied last so compact defaults cannot
            # be defeated by old registry axes unless an explicit heavy profile
            # and acknowledgement are selected.
            cli.extend(list(output_profile_passthrough))

        cli.extend(["--seed", str(seed)])
        cli.extend(["--runs_dir", str(sim_runs_dir)])
        cli.extend(["--run_id", str(runtime_run_id)])
        cli.extend(["--run_tag", str(run_tag)])

        resolved_row = {
            "run_id": run_id,
            "runtime_run_id": runtime_run_id,
            "run_tag": run_tag,
            "seed": seed,
            "axes": {ax.name: r[ax.name] for ax in reg.axes},
            "cli": cli,
            "run_dir": str(run_dir),
        }
        resolved.append(resolved_row)

        matrix_row = {
            "run_id": run_id,
            "runtime_run_id": runtime_run_id,
            "run_tag": run_tag,
            "seed": seed,
            **{ax.name: r[ax.name] for ax in reg.axes},
            "runs_dir": str(sim_runs_dir),
            "run_dir": str(run_dir),
            "cli": " ".join(cli),
        }
        matrix_rows_for_csv.append(matrix_row)


    storage_plan = build_storage_plan(
        registry_name=str(reg.name),
        resolved_rows=resolved,
        output_profile=campaign_output_profile,
    )
    write_storage_plan(storage_plan_path, storage_plan)
    manifest["campaign_output"]["storage_plan_json"] = str(storage_plan_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # Print a human-readable plan before executing anything expensive.
    _print_harness_plan(registry_path=registry_path, reg=reg, resolved=resolved)
    print(
        "Storage preflight: "
        f"profile={storage_plan['output_profile']} risk={storage_plan['risk']} "
        f"planned_cells={storage_plan['planned_cells']} ok={storage_plan['ok']} "
        f"plan={storage_plan_path}",
        flush=True,
    )
    for warning in list(storage_plan.get("warnings", []))[:10]:
        print(f"Storage warning: {warning}", flush=True)

    # Write matrix.csv
    with open(matrix_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(matrix_rows_for_csv[0].keys()) if matrix_rows_for_csv else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in matrix_rows_for_csv:
            w.writerow(row)

    if not bool(storage_plan.get("ok", False)):
        summary = {
            "status": "failed_storage_preflight",
            "planned_runs": len(resolved),
            "ok": 0,
            "failed": len(resolved),
            "runs": [],
            "storage_plan_json": str(storage_plan_path),
            "storage_errors": list(storage_plan.get("errors", [])),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        _write_harness_decision_debug(
            batch_dir=batch_dir,
            verdict="FAIL",
            exit_code=2,
            reasons=[{"code": "STORAGE_PREFLIGHT_FAILED", "msg": "campaign output profile is unsafe"}],
            inputs={
                "planned_runs": int(len(resolved)),
                "storage_risk": str(storage_plan.get("risk", "")),
                "storage_errors": list(storage_plan.get("errors", [])),
            },
            pointers={"summary_json": str(summary_path), "storage_plan_json": str(storage_plan_path)},
            fail_summary="FAIL_SUMMARY | layer=harness | code=STORAGE_PREFLIGHT_FAILED",
        )
        return 2

    if args.print_cmds:
        for row in resolved:
            print("[cmd]", "python -m engine.main", *row["cli"], flush=True)

    if args.dry_run:
        summary = {
            "status": "dry_run",
            "planned_runs": len(resolved),
            "ok": 0,
            "failed": 0,
            "runs": [],
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        _write_harness_decision_debug(
            batch_dir=batch_dir,
            verdict="PASS",
            exit_code=0,
            reasons=[{"code": "DRY_RUN", "msg": "harness dry-run (no execution)"}],
            inputs={"planned_runs": int(len(resolved))},
            pointers={"summary_json": str(summary_path)},
            fail_summary="",
        )
        return 0

    # Execute
    try:
        from experiments.projects.emergence.scripts.utils.run_evo import run_evo
    except ModuleNotFoundError:
        from scripts.utils.run_evo import run_evo

    run_summaries: List[Dict[str, Any]] = []
    failed = 0
    interrupted = False

    total_runs = len(resolved)
    width = len(str(total_runs)) if total_runs else 1
    
    use_color = _use_color()
    for idx, row in enumerate(resolved, start=1):
        run_id = row["run_id"]
        runtime_run_id = row.get("runtime_run_id")
        run_dir = Path(row["run_dir"])
        cli = list(row["cli"])
        

        # Run banner + repeated harness plan (with the current run highlighted).
        # This makes it harder to get lost when launching large grids.
        axes = row.get("axes", {})
        axes_str = ", ".join([f"{k}={v}" for k, v in axes.items()])
        print("\n" + _c("--- RUN START ---", Ansi.CYAN, Ansi.BOLD, enable=use_color), flush=True)
        print(
            f"{_c('Run:', Ansi.YELLOW, Ansi.BOLD, enable=use_color)} "
            f"[{idx:0{width}d}/{total_runs:0{width}d}] | run_id={run_id} | runtime_run_id={runtime_run_id} | run_tag={row.get('run_tag')}",
            flush=True,
        )
        print(f"Seed: {row.get('seed')} | {axes_str}", flush=True)
        # Estimated remaining duration for the whole harness (including this run).
        # Uses a blunt constant seconds/gen heuristic for log readability.
        gens0 = _cli_get_value(list(resolved[0].get("cli", [])), ["--generations", "--gen"]) if resolved else None
        run_seconds: Optional[float] = None
        try:
            g_this = _cli_get_value(cli, ["--generations", "--gen"]) or gens0
            if g_this is not None:
                run_seconds = float(g_this) * float(EST_SECONDS_PER_GEN_DEFAULT)
        except Exception:
            run_seconds = None

        remaining_seconds: Optional[float] = None
        try:
            if gens0 is not None:
                # Assume per-run generations are constant unless explicitly overridden.
                per_run = []
                for r in resolved:
                    g_r = _cli_get_value(list(r.get("cli", [])), ["--generations", "--gen"]) or gens0
                    per_run.append(float(g_r) * float(EST_SECONDS_PER_GEN_DEFAULT))
                remaining_seconds = float(sum(per_run[idx - 1 :]))
        except Exception:
            remaining_seconds = None

        if remaining_seconds is not None:
            eta = datetime.timedelta(seconds=int(remaining_seconds))
            end = datetime.datetime.now() + eta
            eta_line = (
                f"Estimated left: ≈{str(eta)} (assumes {EST_SECONDS_PER_GEN_DEFAULT:.1f}s/gen) | "
                f"end≈{end.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(_c(eta_line, Ansi.GRAY, Ansi.DIM, enable=use_color), flush=True)

        print(_c("---------------", Ansi.CYAN, enable=use_color), flush=True)

        # Execute simulation.
        # IMPORTANT: Ctrl-C compatibility
        # - The simulation itself is responsible for saving an interrupt checkpoint
        #   at a safe boundary (end of generation).
        # - scripts.utils.run_evo forwards Ctrl-C (SIGINT) to the child process group
        #   and returns RunResult(interrupted=True) instead of crashing the harness.
        result = run_evo(cli)

        status = "ok"
        missing: List[str] = []

        if getattr(result, "interrupted", False):
            status = "interrupted"
            interrupted = True
        elif result.returncode != 0:
            status = "failed_exitcode"
        else:
            if not run_dir.exists():
                status = "missing_rundir"
            else:
                ok, missing = _artifact_check(run_dir, reg.expected_artifacts)
                if not ok:
                    status = "missing_artifacts"

        if status != "ok" and status != "interrupted":
            failed += 1

        run_summaries.append(
            {
                "run_id": run_id,
                "runtime_run_id": runtime_run_id,
                "run_tag": row["run_tag"],
                "seed": row["seed"],
                "axes": row["axes"],
                "status": status,
                "returncode": int(result.returncode),
                "missing": missing,
                "run_dir": str(run_dir),
            }
        )

        # Write summary incrementally (so partial failures still leave evidence).
        summary = {
            "status": "interrupted" if interrupted else ("complete" if failed == 0 else "incomplete"),
            "planned_runs": len(resolved),
            "ok": int(len(resolved) - failed),
            "failed": int(failed),
            "runs": run_summaries,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        if interrupted:
            # Stop launching additional runs.
            break

    storage_summary = write_storage_summary(
        storage_summary_path,
        batch_dir,
        output_profile=campaign_output_profile,
    )
    try:
        latest_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        if isinstance(latest_summary, dict):
            latest_summary["storage_plan_json"] = str(storage_plan_path)
            latest_summary["storage_summary_json"] = str(storage_summary_path)
            latest_summary["storage_summary"] = {
                "total_bytes": int(storage_summary.get("total_bytes", 0)),
                "total_files": int(storage_summary.get("total_files", 0)),
                "evidence_compatibility": str(storage_summary.get("evidence_compatibility", "")),
            }
            if scientific_throughput_efficiency_enabled():
                run_dirs = [
                    Path(str(row.get("run_dir", "")))
                    for row in run_summaries
                    if row.get("run_dir")
                ]
                reports = collect_run_scientific_throughput_efficiency_reports(run_dirs)
                aggregate = aggregate_scientific_throughput_efficiency_reports(reports)
                if aggregate is None:
                    aggregate = build_cpu_scientific_throughput_efficiency_report(
                        resident_worlds=int(len(resolved)),
                        active_worlds=int(latest_summary.get("ok", 0)),
                        completed_worlds=int(latest_summary.get("ok", 0)),
                        elapsed_seconds=max(0.0, time.perf_counter() - campaign_started),
                        parameter_evaluations=int(len(resolved)),
                        storage_bytes_written=int(storage_summary.get("total_bytes", 0)),
                    )
                latest_summary["scientific_throughput_efficiency"] = aggregate
                latest_summary["scientific_throughput_efficiency_report_paths"] = (
                    write_scientific_throughput_efficiency_report_files(
                        aggregate,
                        batch_dir,
                    )
                )
            summary_path.write_text(json.dumps(latest_summary, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


    try:
        if tlog is not None:
            from emergence.obs_logging import log_event  # noqa: WPS433
            log_event(
                tlog,
                event="HARNESS_END",
                status=("interrupted" if interrupted else ("complete" if failed == 0 else "incomplete")),
                planned_runs=int(len(resolved)),
                failed=int(failed),
                summary_path=str(summary_path),
            )
    except Exception:
        pass

    if interrupted:
        # Match conventional Ctrl-C return code.
        _write_harness_decision_debug(
            batch_dir=batch_dir,
            verdict="INFRA",
            exit_code=130,
            reasons=[{"code": "INTERRUPTED", "msg": "Ctrl-C / SIGINT"}],
            inputs={"planned_runs": int(len(resolved)), "failed": int(failed)},
            pointers={"summary_json": str(summary_path)},
            fail_summary="FAIL_SUMMARY | layer=harness | code=INTERRUPTED | interrupted by user",
        )
        return 130

    if failed == 0:
        _write_harness_decision_debug(
            batch_dir=batch_dir,
            verdict="PASS",
            exit_code=0,
            reasons=[{"code": "OK", "msg": "all runs ok"}],
            inputs={"planned_runs": int(len(resolved)), "failed": 0},
            pointers={"summary_json": str(summary_path)},
            fail_summary="",
        )
        return 0

    # Harness-level failure is always rc=2.
    _write_harness_decision_debug(
        batch_dir=batch_dir,
        verdict="FAIL",
        exit_code=2,
        reasons=[{"code": "HARNESS_RUNS_FAILED", "msg": f"{failed} run(s) failed"}],
        inputs={"planned_runs": int(len(resolved)), "failed": int(failed)},
        pointers={"summary_json": str(summary_path)},
        fail_summary=f"FAIL_SUMMARY | layer=harness | code=HARNESS_RUNS_FAILED | {failed} run(s) failed",
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
