#!/usr/bin/env python3
from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT
DEFAULT_REGISTRY = PROJECT_ROOT / "registries" / "performance" / "eiel_historical_to_current_four_path_benchmark.yaml"
DEFAULT_EVIDENCE_ROOT = Path(public_runtime_path('evidence_raw/eiel_historical_current_four_path_benchmark_2026'))
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
FLAT_OUTPUT_COST_ATTRIBUTION_MODES = ("off", "required")
FLAT_OUTPUT_COST_SIDECAR_ENV = "EIEL_FLAT_OUTPUT_COST_SIDECAR"
FLAT_OUTPUT_COST_SAMPLE_SCHEMA = "eiel_flat_output_cost_sample_v1"
FLAT_OUTPUT_COST_PHASES = (
    "simulation",
    "reporting_projection",
    "reporting_reducer",
    "serialization",
    "write",
    "flush",
)
FLAT_OUTPUT_COST_SAMPLE_KEYS = frozenset(
    {
        "schema_version",
        "exclusive_ns",
        "call_counts",
        "run_total_ns",
        "complete",
        "errors",
    }
)
SHARED_METRIC_KEYS = (
    "final_alive",
    "total_successful_births",
    "total_failed_births",
    "total_deaths",
)
MINIMUM_W1_REPETITIONS = 5
MINIMUM_W6_REPETITIONS = 3
METRICS_SUMMARY_HEADER = (
    "generation",
    "alive",
    "births",
    "deaths",
    "avg_gen_time_s",
)
CURRENT_REFERENCE_NATIVE_PAIR_SPECS = (
    ("current_w6_reference", "current_w6_native"),
    ("current_w1_reference", "current_w1_native"),
)
OUTPUT_BURDEN_NATIVE_TO_REFERENCE_MIN_RATIO = 0.80
OUTPUT_BURDEN_NATIVE_TO_REFERENCE_MAX_RATIO = 1.25
BENCHMARK_WALL_TIME_CV_MAX = 0.05
PROMOTION_CURRENT_W1_SPEEDUP_MIN = 1.50
PROMOTION_CURRENT_W6_SPEEDUP_MIN = 1.15
NATIVE_PEAK_RSS_MAX_REFERENCE_RATIO = 1.25
PROMOTED_MEASURED_ARMS = (
    "current_w1_reference",
    "current_w1_native",
    "current_w6_reference",
    "current_w6_native",
)
NATIVE_WORKER_AUDIT_SCHEMA = "eiel_native_worker_promotion_audit_v1"
CELL_RESULT_SCHEMA = "eiel_four_path_cell_result_v2"
TIMED_ENV_FORCED_OFF_VALUES = {
    "DEBUG_DETERMINISM": "0",
    "EIEL_STRICT_DETERMINISM": "0",
    # This parity diagnostic defaults on when absent.
    "EIEL_MEMBERSHIP_PARITY": "0",
}
TIMED_ENV_CONTROL_MARKERS = ("TRACE", "DEBUG", "PARITY", "PROBE", "DETERMINISM")
TIMED_ENV_DIAGNOSTIC_PREFIXES = (
    "EIEL_BIRTH_DELTA_",
    "EIEL_EVOPY_LEDGER",
    "EIEL_FORENSIC_",
    "EIEL_GUARD_ENV_ALIVE",
    "EIEL_MEMBERSHIP_",
    "EIEL_STATE_DUMP",
)
CURRENT_MATCHED_PUBLIC_OUTPUT_KNOBS = (
    "sim_cfg.typed_lifecycle_public_output_route_neutral_enable=true",
    "sim_cfg.lineage_observability_enable=true",
    "sim_cfg.runnercore1_write_summary=false",
    "sim_cfg.native_runtime_write_summary=false",
    "sim_cfg.developmental_trace_write_enable=false",
    "sim_cfg.phase_e_e1_enable=false",
    "sim_cfg.a3_recovery_taxonomy_enable=false",
    "sim_cfg.lineage_qd_archive_enable=false",
)
CURRENT_ENGINE_ONLY_KNOBS = (
    # The historical engine predates this field and has no equivalent
    # prediction subsystem to disable.
    "sim_cfg.force_prediction_disabled=true",
)
REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES = (
    "metrics.csv",
    "behaviors.csv",
    "metrics/lineage_descriptors.jsonl",
    "metrics/lineage_summary.csv",
    "metrics/typed_lifecycle_events.csv",
    "metrics/typed_lifecycle_events.jsonl",
    "viz/metrics_summary.csv",
)
EXACT_CURRENT_PUBLIC_OUTPUT_FILES = (
    "behaviors.csv",
    "metrics/lineage_descriptors.jsonl",
    "metrics/lineage_summary.csv",
    "metrics/typed_lifecycle_events.csv",
    "metrics/typed_lifecycle_events.jsonl",
)
OUTPUT_PURPOSE_BY_PATH = {
    "metrics.csv": "generation_metrics",
    "behaviors.csv": "agent_behavior_rows",
    "metrics/lineage_descriptors.jsonl": "lineage_descriptors",
    "metrics/lineage_summary.csv": "lineage_summary",
    "metrics/typed_lifecycle_events.csv": "typed_lifecycle_events",
    "metrics/typed_lifecycle_events.jsonl": "typed_lifecycle_events",
    "viz/metrics_summary.csv": "generation_metrics_summary",
}


@dataclass(frozen=True)
class Arm:
    name: str
    project_root: Path
    workers: int
    native: bool
    reference: str = "current"
    optional: bool = False


@dataclass(frozen=True)
class Cell:
    cell_id: str
    seed: int
    generations: int
    steps: int
    pop: int
    grid_size: int
    food_spawn_per_step: int
    max_food: int
    churn: str
    checkpoint_mode: str
    checkpoint_period: int


def _load_registry(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the benchmark registry")
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _object_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        return dict(payload) if isinstance(payload, dict) else {}
    return {}


def _native_worker_probe_main(argv: list[str]) -> int:
    if len(argv) < 6 or "--" not in argv:
        raise SystemExit(
            "native worker probe requires AUDIT_PATH EXTENSION_PATH SHA256 SOURCE_COMMIT "
            "MAIN_PATH -- ARGS"
        )
    separator = argv.index("--")
    if separator != 5:
        raise SystemExit("native worker probe received malformed positional arguments")
    audit_path = Path(argv[0]).expanduser().resolve()
    expected_extension = Path(argv[1]).expanduser().resolve(strict=True)
    expected_sha256 = _normalize_sha256(argv[2])
    expected_source_commit = str(argv[3]).strip()
    if len(expected_source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_source_commit.lower()
    ):
        raise SystemExit("native worker probe requires a full hexadecimal source commit")
    engine_main_path = Path(argv[4]).expanduser().resolve(strict=True)
    simulation_argv = list(argv[separator + 1 :])
    record: dict[str, Any] = {
        "schema_version": NATIVE_WORKER_AUDIT_SCHEMA,
        "audit_path": str(audit_path),
        "pid": os.getpid(),
        "expected_extension_path": str(expected_extension),
        "expected_extension_sha256": expected_sha256,
        "expected_comparator_source_commit": expected_source_commit,
        "extension_path": None,
        "extension_sha256": None,
        "extension_abi_version": None,
        "extension_version": None,
        "loaded_module_path": None,
        "comparator_source_commit": None,
        "comparator_module_path": None,
        "comparator_module_sha256": None,
        "runtime_jit": None,
        "native_ownership_active": None,
        "mixed_ownership": None,
        "stale_extension": None,
        "checks": {},
        "first_failure_cause": None,
        "exception": None,
        "main_returncode": None,
        "ok": False,
    }

    def set_first_failure(cause: str) -> None:
        if record["first_failure_cause"] is None:
            record["first_failure_cause"] = cause

    def add_check(name: str, ok: bool) -> None:
        record["checks"][name] = bool(ok)
        if not ok:
            set_first_failure(name)

    def capture_runtime(runner: Any) -> None:
        decision = _object_payload(getattr(runner, "_native_safe_default_decision", None))
        native_summary = _object_payload(getattr(runner, "_native_runtime_summary", None))
        engine = getattr(runner, "_native_eiel_engine", None)
        counters = _object_payload(getattr(engine, "counters", None))
        capability = _object_payload(getattr(engine, "capability", None))
        compact_result = _object_payload(native_summary.get("compact_result"))
        sim_cfg = getattr(runner, "sim_cfg", None)
        owner = getattr(engine, "_owner", None)
        native_owner = getattr(owner, "_native_owner", None)

        runtime_jit = bool(
            getattr(sim_cfg, "native_runtime_allow_runtime_jit_build", False)
            or int(counters.get("runtime_jit_build_requests", -1)) != 0
            or capability.get("runtime_jit_in_backend") is not False
            or capability.get("production_jit_free") is not True
        )
        mixed_ownership = bool(
            int(counters.get("normal_python_agent_traversals", -1)) != 0
            or int(counters.get("normal_apply_owner_checkpoint_to_runner_calls", -1)) != 0
            or compact_result.get("python_agent_sync") is not False
            or capability.get("supports_python_agent_sync_in_normal_lane") is not False
        )
        ownership_checks = {
            "native_decision_active": decision.get("native_execution_active") is True,
            "native_decision_not_fail_closed": decision.get("fail_closed") is False,
            "native_runtime_executed": native_summary.get("executed") is True,
            "native_runtime_decision_expected": native_summary.get("decision")
            == "EIEL-NATIVE-RUNTIME-PERSISTENT-GENERATION",
            "native_engine_present": engine is not None,
            "native_resident_owner_present": native_owner is not None,
        }
        native_ownership_active = all(ownership_checks.values())
        record.update(
            {
                "native_safe_default_decision": decision,
                "native_runtime": {
                    "schema_version": native_summary.get("schema_version"),
                    "decision": native_summary.get("decision"),
                    "executed": native_summary.get("executed"),
                    "counters": counters,
                    "capability": capability,
                    "compact_python_agent_sync": compact_result.get("python_agent_sync"),
                },
                "runtime_jit": runtime_jit,
                "native_ownership_active": native_ownership_active,
                "mixed_ownership": mixed_ownership,
            }
        )
        for name, ok in ownership_checks.items():
            add_check(name, ok)
        add_check("runtime_jit_false", not runtime_jit)
        add_check("mixed_ownership_false", not mixed_ownership)
        add_check(
            "runtime_extension_abi_matches_worker",
            capability.get("extension_abi_version") == record.get("extension_abi_version"),
        )
        add_check(
            "runtime_extension_version_matches_worker",
            capability.get("extension_version") == record.get("extension_version"),
        )

    caught: BaseException | None = None
    try:
        from native_runtime import four_way_interchange as comparator_module
        from native_runtime.bindings import load_prebuilt_extension, native_extension_status

        env_status = native_extension_status()
        selected_status = native_extension_status(expected_extension)
        loaded_module = load_prebuilt_extension(expected_extension)
        loaded_module_path = Path(str(getattr(loaded_module, "__file__", ""))).resolve()
        observed_sha256 = _sha256(expected_extension)
        comparator_module_path = Path(str(comparator_module.__file__)).resolve()
        expected_comparator_path = (
            PROJECT_ROOT / "engine" / "native_runtime" / "four_way_interchange.py"
        ).resolve()
        comparator_source_commit = _git(["rev-parse", "HEAD"], PROJECT_ROOT)
        record.update(
            {
                "extension_path": str(Path(selected_status.extension_path).resolve()),
                "extension_sha256": observed_sha256,
                "extension_abi_version": selected_status.extension_abi_version,
                "extension_version": selected_status.extension_version,
                "loaded_module_path": str(loaded_module_path),
                "comparator_source_commit": comparator_source_commit,
                "comparator_module_path": str(comparator_module_path),
                "comparator_module_sha256": _sha256(comparator_module_path),
                "pre_status": selected_status.to_dict(),
                "environment_status": env_status.to_dict(),
            }
        )
        preflight_checks = {
            "extension_environment_path_exact": bool(env_status.extension_path)
            and Path(env_status.extension_path).resolve() == expected_extension,
            "extension_available": bool(selected_status.available),
            "extension_path_exact": Path(selected_status.extension_path).resolve()
            == expected_extension,
            "extension_sha256_exact": observed_sha256 == expected_sha256,
            "extension_manifest_sha256_verified": bool(
                selected_status.output_sha256_verified
            ),
            "extension_abi_compatible": bool(selected_status.abi_compatible),
            "loaded_module_path_exact": loaded_module_path == expected_extension,
            "loaded_module_abi_exact": str(
                getattr(loaded_module, "EIEL_NATIVE_RUNTIME_ABI_VERSION", "")
            )
            == selected_status.extension_abi_version,
            "loaded_module_version_exact": str(
                getattr(loaded_module, "EIEL_NATIVE_RUNTIME_EXTENSION_VERSION", "")
            )
            == selected_status.extension_version,
            "extension_runtime_jit_false": selected_status.runtime_jit is False,
            "comparator_source_commit_exact": comparator_source_commit
            == expected_source_commit,
            "comparator_module_path_exact": comparator_module_path
            == expected_comparator_path,
        }
        for name, ok in preflight_checks.items():
            add_check(name, ok)
        if not all(preflight_checks.values()):
            raise RuntimeError("native worker extension preflight failed")

        module_name = f"_eiel_benchmark_engine_main_{os.getpid()}"
        spec = importlib.util.spec_from_file_location(module_name, engine_main_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load benchmark engine main: {engine_main_path}")
        engine_main = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = engine_main
        spec.loader.exec_module(engine_main)
        original_execute = engine_main.execute_run_context

        def audited_execute(ctx: Any) -> Any:
            try:
                return original_execute(ctx)
            finally:
                capture_runtime(ctx.runner)

        engine_main.execute_run_context = audited_execute
        returncode = int(engine_main.main(simulation_argv))
        record["main_returncode"] = returncode
        add_check("simulation_returncode_zero", returncode == 0)
    except BaseException as exc:
        caught = exc
        record["exception"] = f"{type(exc).__name__}: {exc}"
        set_first_failure(f"native_worker_exception:{type(exc).__name__}")
    finally:
        try:
            from native_runtime.bindings import native_extension_status

            post_status = native_extension_status()
            post_sha256 = _sha256(expected_extension) if expected_extension.is_file() else None
            record["post_status"] = post_status.to_dict()
            add_check(
                "post_extension_path_exact",
                bool(post_status.extension_path)
                and Path(post_status.extension_path).resolve() == expected_extension,
            )
            add_check("post_extension_sha256_exact", post_sha256 == expected_sha256)
            add_check(
                "post_extension_abi_exact",
                post_status.extension_abi_version == record.get("extension_abi_version"),
            )
            add_check(
                "post_extension_version_exact",
                post_status.extension_version == record.get("extension_version"),
            )
            add_check("post_extension_runtime_jit_false", post_status.runtime_jit is False)
        except BaseException as exc:
            record["post_status_error"] = f"{type(exc).__name__}: {exc}"
            set_first_failure(f"native_worker_post_status:{type(exc).__name__}")
        stale_checks = (
            "extension_environment_path_exact",
            "extension_path_exact",
            "extension_sha256_exact",
            "loaded_module_path_exact",
            "runtime_extension_abi_matches_worker",
            "runtime_extension_version_matches_worker",
            "post_extension_path_exact",
            "post_extension_sha256_exact",
            "post_extension_abi_exact",
            "post_extension_version_exact",
            "comparator_source_commit_exact",
            "comparator_module_path_exact",
        )
        record["stale_extension"] = not all(
            record["checks"].get(name) is True for name in stale_checks
        )
        add_check("stale_extension_false", record["stale_extension"] is False)
        record["ok"] = bool(record["checks"]) and all(record["checks"].values())
        if not record["ok"] and record["first_failure_cause"] is None:
            set_first_failure("native_worker_audit_failed")
        _write_json(audit_path, record)
    if caught is not None:
        raise caught
    return int(record["main_returncode"] or 0) if record["ok"] else 86


def _native_worker_command(
    command: list[str],
    *,
    audit_path: Path,
    native_extension: Path,
    expected_sha256: str,
    expected_source_commit: str,
) -> list[str]:
    if len(command) < 2:
        raise ValueError("native worker command requires Python and engine main path")
    return [
        command[0],
        str(Path(__file__).resolve()),
        "--native-worker-probe",
        str(audit_path),
        str(native_extension.resolve(strict=True)),
        _normalize_sha256(expected_sha256),
        str(expected_source_commit),
        command[1],
        "--",
        *command[2:],
    ]


def _load_native_worker_audit(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": NATIVE_WORKER_AUDIT_SCHEMA,
            "audit_path": str(path),
            "ok": False,
            "first_failure_cause": "native_worker_audit_missing",
            "checks": {},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "schema_version": NATIVE_WORKER_AUDIT_SCHEMA,
            "audit_path": str(path),
            "ok": False,
            "first_failure_cause": f"native_worker_audit_invalid:{type(exc).__name__}",
            "checks": {},
        }
    if not isinstance(payload, dict) or payload.get("schema_version") != NATIVE_WORKER_AUDIT_SCHEMA:
        return {
            "schema_version": NATIVE_WORKER_AUDIT_SCHEMA,
            "audit_path": str(path),
            "ok": False,
            "first_failure_cause": "native_worker_audit_schema_mismatch",
            "checks": {},
        }
    return dict(payload)


def _python_interpreter_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    # Python uses the invoked venv symlink to discover pyvenv.cfg. Dereferencing
    # it here silently switches the benchmark to the system package set.
    return Path(os.path.abspath(expanded))


def _run_capture(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, timeout: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True, timeout=timeout)
    return {
        "command": cmd,
        "cwd": str(cwd),
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "wall_s": float(time.perf_counter() - started),
    }


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=str(cwd), text=True).strip()


def _environment_record(
    py: Path,
    evidence_root: Path,
    historical_project: Path | None,
) -> dict[str, Any]:
    def py_mod_version(project: Path, module: str) -> str:
        env = _base_env(project)
        result = _run_capture(
            [str(py), "-c", f"import {module}; print(getattr({module}, '__version__', 'unknown'))"],
            cwd=project,
            env=env,
            timeout=60,
        )
        return result["stdout"].strip() if result["returncode"] == 0 else f"unavailable: {result['stderr'].strip()}"

    record: dict[str, Any] = {
        "schema_version": "eiel_four_path_environment_v1",
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python_executable": str(py),
        "python_version": subprocess.check_output([str(py), "--version"], text=True).strip(),
        "current_commit": _git(["rev-parse", "HEAD"], PROJECT_ROOT),
        "current_status_short": _git(["status", "--short", "--branch"], PROJECT_ROOT),
        "storage_path": str(evidence_root),
        "numpy_current": py_mod_version(PROJECT_ROOT, "numpy"),
        "torch_current": py_mod_version(PROJECT_ROOT, "torch"),
    }
    if historical_project is None:
        record["historical_environment"] = {"applicability": "not_applicable"}
    else:
        record.update(
            {
                "historical_commit": _git(["rev-parse", "HEAD"], historical_project),
                "historical_status_short": _git(
                    ["status", "--short", "--branch"], historical_project
                ),
                "numpy_historical_env": py_mod_version(historical_project, "numpy"),
                "torch_historical_env": py_mod_version(historical_project, "torch"),
            }
        )
    for cmd in (["lscpu"], ["free", "-h"], ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]):
        exe = shutil.which(cmd[0])
        if exe:
            result = _run_capture([exe, *cmd[1:]], cwd=PROJECT_ROOT, timeout=30)
            record[" ".join(cmd)] = result["stdout"].strip() or result["stderr"].strip()
    return record


def _base_env(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(project_root), str(project_root / "engine")]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["DEBUG_DETERMINISM"] = "0"
    env["EIEL_STRICT_DETERMINISM"] = "0"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def _is_timed_env_control(key: str) -> bool:
    if key in {
        "DEBUG_DETERMINISM",
        "EMERGENCE_DEBUG_DETERMINISM_DEFAULT",
        FLAT_OUTPUT_COST_SIDECAR_ENV,
    }:
        return True
    return key.startswith(TIMED_ENV_DIAGNOSTIC_PREFIXES) or (
        key.startswith("EIEL_") and any(marker in key for marker in TIMED_ENV_CONTROL_MARKERS)
    )


def _env_control_enabled(value: str) -> bool:
    return str(value).strip().lower() not in {"", "0", "false", "no", "off", "none"}


def _timed_subprocess_env(project_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    env = _base_env(project_root)
    inherited_controls = sorted(key for key in os.environ if _is_timed_env_control(key))
    controls_before_sanitization = sorted(key for key in env if _is_timed_env_control(key))
    for key in controls_before_sanitization:
        env.pop(key, None)
    env.update(TIMED_ENV_FORCED_OFF_VALUES)

    effective_controls = {
        key: value
        for key, value in sorted(env.items())
        if _is_timed_env_control(key)
    }
    enabled_controls = {
        key: value
        for key, value in effective_controls.items()
        if _env_control_enabled(value)
    }
    if enabled_controls:
        raise RuntimeError(
            "timed subprocess environment retained enabled trace/debug/parity controls: "
            f"{sorted(enabled_controls)}"
        )
    record = {
        "schema_version": "eiel_four_path_timed_env_policy_v1",
        "policy": "clear_inherited_trace_debug_parity_probe_controls_then_force_known_defaults_off",
        "selection_rule": (
            "DEBUG_DETERMINISM or EMERGENCE_DEBUG_DETERMINISM_DEFAULT or "
            "known EIEL diagnostic prefixes or EIEL_* containing TRACE, DEBUG, "
            "PARITY, PROBE, or DETERMINISM, plus the cost-sidecar destination"
        ),
        "cleared_inherited_keys": inherited_controls,
        "cleared_effective_keys": controls_before_sanitization,
        "forced_off_values": dict(sorted(TIMED_ENV_FORCED_OFF_VALUES.items())),
        "effective_control_values": effective_controls,
        "enabled_control_values": enabled_controls,
        "ok": not enabled_controls,
    }
    return env, record


def _registry_cell(
    env_cfg: dict[str, Any],
    *,
    seed: Any,
    churn: dict[str, Any],
    workload: dict[str, Any],
    checkpoint: dict[str, Any],
) -> Cell:
    cell_id = (
        f"s{int(seed)}_{churn['name']}_{workload['name']}_"
        f"ckpt{checkpoint['name']}"
    )
    return Cell(
        cell_id=cell_id,
        seed=int(seed),
        generations=int(workload["generations"]),
        steps=int(env_cfg["steps_per_generation"]),
        pop=int(env_cfg["population"]),
        grid_size=int(env_cfg["grid_size"]),
        food_spawn_per_step=int(churn["food_spawn_per_step"]),
        max_food=int(churn["max_food"]),
        churn=str(churn["name"]),
        checkpoint_mode=str(checkpoint["name"]),
        checkpoint_period=int(checkpoint["checkpoint_bundle_v2_period_gen"]),
    )


def _cells_from_registry(registry: dict[str, Any], *, limit_cells: int | None = None) -> list[Cell]:
    env_cfg = registry["benchmark_envelope"]
    matrix = registry["matrix"]
    explicit_cells = matrix.get("cells")
    if explicit_cells is not None:
        if not isinstance(explicit_cells, list):
            raise ValueError("matrix.cells must be a list")
        cells = [
            _registry_cell(
                env_cfg,
                seed=spec["seed"],
                churn=dict(spec["churn"]),
                workload=dict(spec["workload"]),
                checkpoint=dict(spec["checkpoint"]),
            )
            for spec in explicit_cells
        ]
    else:
        cells = [
            _registry_cell(
                env_cfg,
                seed=seed,
                churn=dict(churn),
                workload=dict(workload),
                checkpoint=dict(checkpoint),
            )
            for seed in matrix["seeds"]
            for churn in matrix["churn_levels"]
            for workload in matrix["workload_lengths"]
            for checkpoint in matrix["checkpoint_modes"]
        ]
    cell_ids = [cell.cell_id for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("benchmark matrix generated duplicate deterministic cell IDs")
    if limit_cells is not None:
        cells = cells[: int(limit_cells)]
    return cells


def _arm_specs(registry: dict[str, Any]) -> dict[str, Arm]:
    paths = {"current_project_root": Path(registry["current_project_root"])}
    if registry.get("historical_project_root"):
        paths["historical_project_root"] = Path(registry["historical_project_root"])
    arms = {}
    for name, spec in registry["arms"].items():
        project_ref = str(spec["project_root"])
        arms[name] = Arm(
            name=name,
            project_root=paths.get(project_ref, Path(project_ref)),
            workers=int(spec["workers"]),
            native=bool(spec.get("native", False)),
            reference=str(spec.get("reference", "current")),
            optional=bool(spec.get("optional", False)),
        )
    return arms


def _common_knobs(cell: Cell) -> list[str]:
    return [
        "sim_cfg.maintenance_enable=false",
        "sim_cfg.chemistry_functional_coupling_enable=false",
        "sim_cfg.genome_c1_enable=false",
        "sim_cfg.genome_c2_enable=false",
        "env_cfg.reproduce_requires_mate=false",
        "env_cfg.reproduce_try_prob=1.0",
        f"env_cfg.food_spawn_per_step={cell.food_spawn_per_step}",
        f"env_cfg.max_food={cell.max_food}",
        "env_cfg.view_radius=1",
        "genome.use_gru=false",
        "genome.memory_dim=0",
    ]


def _cell_command(py: Path, arm: Arm, cell: Cell, run_id: str, runs_dir: Path) -> list[str]:
    cmd = [
        str(py),
        str(arm.project_root / "engine" / "main.py"),
        "--seed",
        str(cell.seed),
        "--gen",
        str(cell.generations),
        "--steps",
        str(cell.steps),
        "--pop",
        str(cell.pop),
        "--grid_size",
        str(cell.grid_size),
        "--device",
        "cpu",
        "--bank_device",
        "cpu",
        "--torch_threads",
        "1",
        "--policy_tick_stride",
        "2",
        "--checkpoint_bundle_v2_period_gen",
        str(cell.checkpoint_period),
        "--checkpoint_bundle_v2_strict_replay_mode",
        "1",
        "--run_id",
        run_id,
        "--runs_dir",
        str(runs_dir),
        "--allow_existing_run_dir",
    ]
    for knob in _common_knobs(cell):
        cmd.extend(["--knob", knob])
    if arm.reference == "current":
        for knob in CURRENT_ENGINE_ONLY_KNOBS:
            cmd.extend(["--knob", knob])
        for knob in CURRENT_MATCHED_PUBLIC_OUTPUT_KNOBS:
            cmd.extend(["--knob", knob])
        cmd.extend(
            [
                "--knob",
                f"sim_cfg.native_safe_default_enable={'true' if arm.native else 'false'}",
            ]
        )
        cmd.extend(
            [
                "--native_runtime_cxx26_scope_lock_enable",
                "--native_runtime_fixed_policy_family",
                "mlp",
            ]
        )
    if arm.native:
        cmd.extend(
            [
                "--native_runtime_enable",
                "--native_runtime_max_births_per_generation",
                str(max(1, cell.pop * 8)),
            ]
        )
    return cmd


def _parse_time_v(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "Elapsed (wall clock) time (h:mm:ss or m:ss)":
            out["elapsed_hms"] = value
        elif key == "User time (seconds)":
            out["user_time_s"] = _float_or_none(value)
        elif key == "System time (seconds)":
            out["system_time_s"] = _float_or_none(value)
        elif key == "Percent of CPU this job got":
            out["cpu_percent"] = value
            out["cpu_percent_value"] = _float_or_none(value.rstrip("%"))
        elif key == "Maximum resident set size (kbytes)":
            out["max_rss_kb"] = _int_or_none(value)
    return out


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _validate_flat_output_cost_sample(sample: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(sample, dict):
        return ["sample_not_object"]

    observed_keys = set(sample)
    missing_keys = sorted(FLAT_OUTPUT_COST_SAMPLE_KEYS - observed_keys)
    unexpected_keys = sorted(observed_keys - FLAT_OUTPUT_COST_SAMPLE_KEYS)
    if missing_keys:
        errors.append(f"missing_top_level_keys:{','.join(missing_keys)}")
    if unexpected_keys:
        errors.append(f"unexpected_top_level_keys:{','.join(unexpected_keys)}")
    if sample.get("schema_version") != FLAT_OUTPUT_COST_SAMPLE_SCHEMA:
        errors.append("schema_version_mismatch")

    for field in ("exclusive_ns", "call_counts"):
        values = sample.get(field)
        if not isinstance(values, dict):
            errors.append(f"{field}_not_object")
            continue
        phase_keys = set(values)
        missing_phases = sorted(set(FLAT_OUTPUT_COST_PHASES) - phase_keys)
        unexpected_phases = sorted(phase_keys - set(FLAT_OUTPUT_COST_PHASES))
        if missing_phases:
            errors.append(f"{field}_missing_phases:{','.join(missing_phases)}")
        if unexpected_phases:
            errors.append(f"{field}_unexpected_phases:{','.join(unexpected_phases)}")
        for phase in FLAT_OUTPUT_COST_PHASES:
            value = values.get(phase)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"{field}_{phase}_not_nonnegative_integer")

    run_total_ns = sample.get("run_total_ns")
    if not isinstance(run_total_ns, int) or isinstance(run_total_ns, bool) or run_total_ns < 0:
        errors.append("run_total_ns_not_nonnegative_integer")
    exclusive_ns = sample.get("exclusive_ns")
    if (
        isinstance(run_total_ns, int)
        and not isinstance(run_total_ns, bool)
        and isinstance(exclusive_ns, dict)
        and all(
            isinstance(exclusive_ns.get(phase), int)
            and not isinstance(exclusive_ns.get(phase), bool)
            and exclusive_ns[phase] >= 0
            for phase in FLAT_OUTPUT_COST_PHASES
        )
        and sum(int(exclusive_ns[phase]) for phase in FLAT_OUTPUT_COST_PHASES) > run_total_ns
    ):
        errors.append("exclusive_ns_sum_exceeds_run_total_ns")

    complete = sample.get("complete")
    if not isinstance(complete, bool):
        errors.append("complete_not_boolean")
    sample_errors = sample.get("errors")
    if not isinstance(sample_errors, list) or any(
        not isinstance(error, str) or not error for error in sample_errors
    ):
        errors.append("errors_not_nonempty_string_list")
    elif isinstance(complete, bool) and complete != (not sample_errors):
        errors.append("complete_errors_inconsistent")
    return errors


def _load_flat_output_cost_sample(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "applicability": "applicable",
        "sidecar_path": str(path),
        "loaded": False,
        "valid": False,
        "complete": False,
        "errors": [],
        "sample": None,
    }
    if not path.is_file():
        result["errors"] = ["missing_sidecar"]
        return result
    try:
        sample = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result["errors"] = [f"sidecar_read_error:{type(exc).__name__}"]
        return result

    result["loaded"] = True
    result["sample"] = sample
    validation_errors = _validate_flat_output_cost_sample(sample)
    result["valid"] = not validation_errors
    result["errors"] = validation_errors
    if validation_errors:
        return result

    assert isinstance(sample, dict)
    producer_errors = list(sample["errors"])
    if not bool(sample["complete"]):
        result["errors"] = ["sample_incomplete", *[f"sample_error:{error}" for error in producer_errors]]
        return result
    result["complete"] = True
    return result


def _flat_output_cost_sidecar_path(
    evidence_root: Path,
    rep: str,
    arm: Arm,
    run_id: str,
) -> Path:
    return evidence_root / "flat_output_cost_sidecars" / rep / arm.name / f"{run_id}.json"


def _output_purpose(relative_path: str) -> str:
    if relative_path in OUTPUT_PURPOSE_BY_PATH:
        return OUTPUT_PURPOSE_BY_PATH[relative_path]
    lowered = relative_path.lower()
    if _is_checkpoint_output(lowered):
        return "checkpoint_sidecar" if "sidecar" in lowered else "checkpoint"
    if "event_tape" in lowered:
        return "event_tape"
    if "cache" in lowered and ("trace" in lowered or "event" in lowered):
        return "cache_trace"
    if "summary" in lowered:
        return "summary"
    if any(token in lowered for token in ("debug", "diagnostic", "trace", "forensic", "probe")):
        return "diagnostic"
    return "supporting_output"


def _output_requirement(relative_path: str, purpose: str) -> tuple[str, bool, bool]:
    if relative_path in REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES:
        return "required", True, False
    if purpose in {"diagnostic", "event_tape", "cache_trace"}:
        return "debug_only", False, True
    return "optional", False, False


def _artifact_row_count(path: Path) -> tuple[int | None, str | None]:
    if not path.is_file() or path.suffix.lower() not in {".csv", ".jsonl"}:
        return None, None
    try:
        if path.suffix.lower() == ".jsonl":
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), None
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        return max(0, len(rows) - 1), None
    except (OSError, UnicodeError, csv.Error) as exc:
        return None, f"{type(exc).__name__}:{exc}"


def _output_inventory_entry(run_dir: Path, relative_path: str) -> dict[str, Any]:
    path = run_dir / relative_path
    profile = _public_file_profile(path)
    purpose = _output_purpose(relative_path)
    requirement_class, required, debug_only = _output_requirement(relative_path, purpose)
    row_count, row_count_error = _artifact_row_count(path)
    return {
        "relative_path": relative_path,
        "enabled": bool(profile["exists"]),
        "purpose": purpose,
        "requirement_class": requirement_class,
        "required": required,
        "debug_only": debug_only,
        "count": 1 if profile["exists"] else 0,
        "rows": row_count,
        "row_count_error": row_count_error,
        "bytes": int(profile["bytes"]),
        "sha256": profile["sha256"],
    }


def _csv_semantic_profile(path: Path) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "schema_version": "eiel_full_csv_semantic_profile_v1",
        "path": str(path),
        "exists": path.is_file(),
        "column_count": 0,
        "row_count": 0,
        "header_sha256": None,
        "semantic_sha256": None,
        "raw_bytes": 0,
        "raw_sha256": None,
        "errors": [],
        "valid": False,
    }
    if not path.is_file():
        profile["errors"].append("missing_file")
        return profile
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except OSError as exc:
        profile["errors"].append(f"read_error:{type(exc).__name__}")
        return profile
    except UnicodeDecodeError:
        profile["errors"].append("invalid_utf8")
        return profile
    except csv.Error as exc:
        profile["errors"].append(f"csv_error:{exc}")
        return profile

    profile["raw_bytes"] = len(raw)
    profile["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    if not rows:
        profile["errors"].append("missing_header")
        return profile
    header = rows[0]
    profile["column_count"] = len(header)
    profile["row_count"] = len(rows) - 1
    if not header or any(not name for name in header):
        profile["errors"].append("empty_header_field")
    if len(set(header)) != len(header):
        profile["errors"].append("duplicate_header_field")
    if len(rows) == 1:
        profile["errors"].append("missing_data_rows")
    for row_index, row in enumerate(rows[1:], start=1):
        if len(row) != len(header):
            profile["errors"].append(f"row_{row_index}_column_count")

    header_json = json.dumps(header, ensure_ascii=True, separators=(",", ":"))
    semantic = hashlib.sha256()
    semantic.update(header_json.encode("utf-8"))
    semantic.update(b"\n")
    for row in rows[1:]:
        semantic.update(json.dumps(row, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
        semantic.update(b"\n")
    profile["header_sha256"] = hashlib.sha256(header_json.encode("utf-8")).hexdigest()
    profile["semantic_sha256"] = semantic.hexdigest()
    profile["valid"] = not profile["errors"]
    return profile


def _first_csv_semantic_mismatch(reference_path: Path, native_path: Path) -> dict[str, Any] | None:
    try:
        with reference_path.open(newline="", encoding="utf-8") as stream:
            reference_rows = list(csv.reader(stream))
        with native_path.open(newline="", encoding="utf-8") as stream:
            native_rows = list(csv.reader(stream))
    except (OSError, UnicodeError, csv.Error) as exc:
        return {"surface": "csv_read", "error": f"{type(exc).__name__}:{exc}"}
    reference_header = reference_rows[0] if reference_rows else []
    native_header = native_rows[0] if native_rows else []
    if reference_header != native_header:
        for index in range(max(len(reference_header), len(native_header))):
            reference_name = reference_header[index] if index < len(reference_header) else None
            native_name = native_header[index] if index < len(native_header) else None
            if reference_name != native_name:
                return {
                    "surface": "header",
                    "column_index": index,
                    "reference": reference_name,
                    "native": native_name,
                }
    for row_index in range(1, max(len(reference_rows), len(native_rows))):
        reference_row = reference_rows[row_index] if row_index < len(reference_rows) else None
        native_row = native_rows[row_index] if row_index < len(native_rows) else None
        if reference_row == native_row:
            continue
        if reference_row is None or native_row is None:
            return {
                "surface": "row_count",
                "row_index": row_index - 1,
                "reference_present": reference_row is not None,
                "native_present": native_row is not None,
            }
        for column_index in range(max(len(reference_row), len(native_row))):
            reference_value = reference_row[column_index] if column_index < len(reference_row) else None
            native_value = native_row[column_index] if column_index < len(native_row) else None
            if reference_value != native_value:
                return {
                    "surface": "data",
                    "row_index": row_index - 1,
                    "column_index": column_index,
                    "column": reference_header[column_index] if column_index < len(reference_header) else None,
                    "reference": reference_value,
                    "native": native_value,
                }
    return None


def _metrics_csv_pair_fairness(reference: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    header_equal = reference.get("header_sha256") == native.get("header_sha256")
    semantic_rows_equal = reference.get("semantic_sha256") == native.get("semantic_sha256")
    ok = bool(reference.get("valid")) and bool(native.get("valid")) and header_equal and semantic_rows_equal
    first_mismatch = None
    if not ok and reference.get("exists") and native.get("exists"):
        first_mismatch = _first_csv_semantic_mismatch(Path(str(reference["path"])), Path(str(native["path"])))
    return {
        "schema_version": "eiel_metrics_csv_full_semantic_pair_v1",
        "semantic_scope": "all_emitted_columns_and_rows",
        "reference_valid": bool(reference.get("valid")),
        "native_valid": bool(native.get("valid")),
        "reference_errors": list(reference.get("errors") or []),
        "native_errors": list(native.get("errors") or []),
        "reference_column_count": int(reference.get("column_count") or 0),
        "native_column_count": int(native.get("column_count") or 0),
        "reference_row_count": int(reference.get("row_count") or 0),
        "native_row_count": int(native.get("row_count") or 0),
        "header_equal": header_equal,
        "semantic_rows_equal": semantic_rows_equal,
        "reference_semantic_sha256": reference.get("semantic_sha256"),
        "native_semantic_sha256": native.get("semantic_sha256"),
        "first_mismatch": first_mismatch,
        "ok": ok,
    }


def _metrics_summary_profile(path: Path) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "schema_version": "eiel_metrics_summary_profile_v1",
        "path": str(path),
        "exists": path.is_file(),
        "expected_header": list(METRICS_SUMMARY_HEADER),
        "header": None,
        "header_exact": False,
        "row_count": 0,
        "semantic_rows": [],
        "avg_gen_time_s_values": [],
        "generation_order_valid": False,
        "raw_bytes": 0,
        "raw_sha256": None,
        "errors": [],
        "valid": False,
    }
    if not path.is_file():
        profile["errors"].append("missing_file")
        return profile

    try:
        raw = path.read_bytes()
    except OSError as exc:
        profile["errors"].append(f"read_error:{type(exc).__name__}")
        return profile
    profile["raw_bytes"] = len(raw)
    profile["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        profile["errors"].append("invalid_utf8")
        return profile

    lines = text.splitlines()
    expected_header_line = ",".join(METRICS_SUMMARY_HEADER)
    profile["header_exact"] = bool(lines) and lines[0] == expected_header_line
    try:
        csv_rows = list(csv.reader(io.StringIO(text, newline="")))
    except csv.Error as exc:
        profile["errors"].append(f"csv_error:{exc}")
        return profile
    if not csv_rows:
        profile["errors"].append("missing_header")
        return profile

    profile["header"] = csv_rows[0]
    if csv_rows[0] != list(METRICS_SUMMARY_HEADER) or not profile["header_exact"]:
        profile["errors"].append("header_mismatch")

    data_rows = csv_rows[1:]
    profile["row_count"] = len(data_rows)
    if not data_rows:
        profile["errors"].append("missing_data_rows")

    previous_generation: int | None = None
    order_valid = True
    for row_index, row in enumerate(data_rows, start=1):
        if len(row) != len(METRICS_SUMMARY_HEADER):
            profile["errors"].append(f"row_{row_index}_column_count")
            order_valid = False
            continue
        try:
            generation, alive, births, deaths = (int(value) for value in row[:4])
        except (TypeError, ValueError):
            profile["errors"].append(f"row_{row_index}_semantic_integer")
            order_valid = False
            continue
        if min(generation, alive, births, deaths) < 0:
            profile["errors"].append(f"row_{row_index}_negative_semantic_value")
        if previous_generation is not None and generation <= previous_generation:
            profile["errors"].append(f"row_{row_index}_generation_order")
            order_valid = False
        previous_generation = generation
        profile["semantic_rows"].append(
            {
                "generation": generation,
                "alive": alive,
                "births": births,
                "deaths": deaths,
            }
        )
        try:
            avg_gen_time_s = float(row[4])
        except (TypeError, ValueError):
            profile["errors"].append(f"row_{row_index}_avg_gen_time_s_not_float")
            continue
        if not math.isfinite(avg_gen_time_s) or avg_gen_time_s < 0.0:
            profile["errors"].append(f"row_{row_index}_avg_gen_time_s_not_finite_nonnegative")
            continue
        profile["avg_gen_time_s_values"].append(avg_gen_time_s)

    profile["generation_order_valid"] = order_valid and bool(data_rows)
    profile["valid"] = not profile["errors"]
    return profile


def _metrics_summary_pair_fairness(
    reference: dict[str, Any],
    native: dict[str, Any],
) -> dict[str, Any]:
    reference_rows = list(reference.get("semantic_rows") or [])
    native_rows = list(native.get("semantic_rows") or [])
    semantic_rows_equal = reference_rows == native_rows
    first_semantic_mismatch: dict[str, Any] | None = None
    if not semantic_rows_equal:
        for index in range(max(len(reference_rows), len(native_rows))):
            reference_row = reference_rows[index] if index < len(reference_rows) else None
            native_row = native_rows[index] if index < len(native_rows) else None
            if reference_row == native_row:
                continue
            differences = {
                key: {
                    "reference": reference_row.get(key) if reference_row is not None else None,
                    "native": native_row.get(key) if native_row is not None else None,
                }
                for key in METRICS_SUMMARY_HEADER[:4]
                if reference_row is None
                or native_row is None
                or reference_row.get(key) != native_row.get(key)
            }
            first_semantic_mismatch = {
                "row_index": index,
                "differences": differences,
            }
            break

    raw_bytes_equal = bool(reference.get("exists")) and bool(native.get("exists")) and (
        reference.get("raw_sha256") == native.get("raw_sha256")
    )
    ok = bool(reference.get("valid")) and bool(native.get("valid")) and semantic_rows_equal
    return {
        "schema_version": "eiel_metrics_summary_pair_fairness_v1",
        "semantic_keys": list(METRICS_SUMMARY_HEADER[:4]),
        "timing_key": "avg_gen_time_s",
        "timing_requirement": "finite_nonnegative_only_not_equal",
        "reference_valid": bool(reference.get("valid")),
        "native_valid": bool(native.get("valid")),
        "reference_errors": list(reference.get("errors") or []),
        "native_errors": list(native.get("errors") or []),
        "semantic_rows_equal": semantic_rows_equal,
        "first_semantic_mismatch": first_semantic_mismatch,
        "reference_avg_gen_time_s_values": list(reference.get("avg_gen_time_s_values") or []),
        "native_avg_gen_time_s_values": list(native.get("avg_gen_time_s_values") or []),
        "reference_raw_bytes": reference.get("raw_bytes"),
        "native_raw_bytes": native.get("raw_bytes"),
        "reference_raw_sha256": reference.get("raw_sha256"),
        "native_raw_sha256": native.get("raw_sha256"),
        "raw_bytes_equal": raw_bytes_equal,
        "raw_byte_inequality": not raw_bytes_equal,
        "ok": ok,
    }


def _is_checkpoint_output(relative_path: str) -> bool:
    return "checkpoint" in relative_path or "bundle_gen" in relative_path


def _public_file_profile(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "bytes": 0, "sha256": None}
    try:
        size = path.stat().st_size
        digest = _sha256(path)
    except OSError:
        return {"exists": False, "bytes": 0, "sha256": None}
    return {"exists": True, "bytes": int(size), "sha256": digest}


def _canonical_checkpoint_storage_profile(run_dir: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    checkpoint_root = run_dir / "checkpoint_bundles_v2"
    for metadata_path in sorted(checkpoint_root.glob("bundle_gen*/metadata.json")):
        relative_bundle = metadata_path.parent.relative_to(run_dir).as_posix()
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(
                {
                    "bundle": relative_bundle,
                    "reason": "checkpoint_metadata_read_failed",
                    "detail": f"{type(exc).__name__}:{exc}",
                }
            )
            continue
        digest = str(metadata.get("canonical_checkpoint_storage_digest", "") or "")
        schema_version = str(
            metadata.get("canonical_checkpoint_storage_schema_version", "") or ""
        )
        field_count = metadata.get("canonical_checkpoint_storage_field_count")
        generation_index = metadata.get("generation_index")
        if not digest or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            errors.append(
                {
                    "bundle": relative_bundle,
                    "reason": "canonical_checkpoint_storage_digest_missing_or_invalid",
                }
            )
            continue
        if not schema_version:
            errors.append(
                {
                    "bundle": relative_bundle,
                    "reason": "canonical_checkpoint_storage_schema_version_missing",
                }
            )
            continue
        if isinstance(field_count, bool) or not isinstance(field_count, int) or field_count < 1:
            errors.append(
                {
                    "bundle": relative_bundle,
                    "reason": "canonical_checkpoint_storage_field_count_invalid",
                }
            )
            continue
        if isinstance(generation_index, bool) or not isinstance(generation_index, int):
            errors.append(
                {
                    "bundle": relative_bundle,
                    "reason": "checkpoint_generation_index_invalid",
                }
            )
            continue
        entries.append(
            {
                "bundle": relative_bundle,
                "generation_index": generation_index,
                "schema_version": schema_version,
                "field_count": field_count,
                "digest": digest,
            }
        )
    return {
        "schema_version": "eiel_four_path_canonical_checkpoint_storage_profile_v1",
        "entries": entries,
        "errors": errors,
        "ok": not errors,
    }


def _run_output_profile(run_dir: Path) -> dict[str, Any]:
    relative_files: list[str] = []
    checkpoint_relative_files: list[str] = []
    output_bytes = 0
    checkpoint_bytes = 0
    if run_dir.exists():
        for path in run_dir.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(run_dir).as_posix()
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            relative_files.append(relative_path)
            output_bytes += size
            if _is_checkpoint_output(relative_path):
                checkpoint_relative_files.append(relative_path)
                checkpoint_bytes += size
    inventory_paths = sorted(set(relative_files) | set(REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES))
    return {
        "schema_version": "eiel_four_path_run_output_profile_v2",
        "relative_files": sorted(relative_files),
        "output_bytes": int(output_bytes),
        "output_files": len(relative_files),
        "checkpoint_relative_files": sorted(checkpoint_relative_files),
        "checkpoint_bytes": int(checkpoint_bytes),
        "checkpoint_files": len(checkpoint_relative_files),
        "canonical_checkpoint_storage": _canonical_checkpoint_storage_profile(run_dir),
        "metrics_csv": _csv_semantic_profile(run_dir / "metrics.csv"),
        "metrics_summary": _metrics_summary_profile(run_dir / "viz" / "metrics_summary.csv"),
        "required_public_outputs": {
            relative_path: _public_file_profile(run_dir / relative_path)
            for relative_path in REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES
        },
        "output_inventory": [
            _output_inventory_entry(run_dir, relative_path)
            for relative_path in inventory_paths
        ],
    }


def _last_metrics(run_dir: Path) -> dict[str, Any]:
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return {"metrics_exists": False}
    rows: list[dict[str, str]] = []
    with metrics.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {"metrics_exists": True, "metric_rows": 0}
    last = rows[-1]
    totals: dict[str, float] = {}
    for key in ("successful_births", "failed_births", "deaths", "birth_attempts"):
        vals = [_float_or_none(row.get(key, "")) for row in rows]
        totals[f"total_{key}"] = float(sum(v for v in vals if v is not None))
    return {
        "metrics_exists": True,
        "metric_rows": len(rows),
        "final_generation": _int_or_none(last.get("generation", "")),
        "final_alive": _int_or_none(last.get("alive", "")),
        "final_successful_births": _int_or_none(last.get("successful_births", "")),
        "final_failed_births": _int_or_none(last.get("failed_births", "")),
        "final_deaths": _int_or_none(last.get("deaths", "")),
        **totals,
    }


def _scan_native_status(run_dir: Path, out_path: Path) -> dict[str, Any]:
    text = ""
    if out_path.exists():
        text = out_path.read_text(encoding="utf-8", errors="replace")
    status = {
        "hidden_fallback_seen": "hidden_fallback': True" in text or '"hidden_fallback": true' in text,
        "native_fail_closed_seen": "native_runtime_fail_closed_exception" in text,
        "native_runtime_seen": "native_runtime" in text or "EIEL-NATIVE-RUNTIME" in text,
    }
    for p in run_dir.rglob("*.json"):
        try:
            body = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if '"hidden_fallback": true' in body or "hidden_fallback': True" in body:
            status["hidden_fallback_seen"] = True
        if "native_runtime_fail_closed_exception" in body:
            status["native_fail_closed_seen"] = True
        if "native_runtime" in body or "EIEL-NATIVE-RUNTIME" in body:
            status["native_runtime_seen"] = True
    return status


def _run_finalization_status(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "viz" / "run_status.json"
    if not path.is_file():
        return {"path": str(path), "loaded": False, "status": None, "complete": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "path": str(path),
            "loaded": False,
            "status": None,
            "complete": False,
            "error": f"{type(exc).__name__}:{exc}",
        }
    status = str(payload.get("status", "") or "") if isinstance(payload, dict) else ""
    return {
        "path": str(path),
        "loaded": isinstance(payload, dict),
        "status": status or None,
        "complete": status.lower() in {"complete", "completed", "success", "succeeded"},
    }


def _cell_exclusion_reasons(
    *,
    arm: Arm,
    returncode: int,
    run_dir: Path,
    finalization: dict[str, Any],
    metrics: dict[str, Any],
    output_profile: dict[str, Any],
    cost_attribution: dict[str, Any],
    time_v: dict[str, Any],
    native_status: dict[str, Any],
    native_worker_audit: dict[str, Any] | None,
    generations: int,
) -> list[str]:
    reasons: list[str] = []
    if native_worker_audit is not None and native_worker_audit.get("first_failure_cause"):
        reasons.append(str(native_worker_audit["first_failure_cause"]))
    if returncode != 0:
        reasons.append(f"subprocess_returncode:{returncode}")
    if not run_dir.is_dir():
        reasons.append("run_directory_missing")
    if not bool(finalization.get("complete")):
        reasons.append("run_finalization_incomplete")
    if not bool(metrics.get("metrics_exists")):
        reasons.append("metrics_csv_missing")
    elif int(metrics.get("metric_rows") or 0) < int(generations):
        reasons.append("metrics_generation_rows_incomplete")
    if arm.reference == "current":
        required = dict(output_profile.get("required_public_outputs") or {})
        missing_required = [
            relative_path
            for relative_path in REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES
            if not bool(dict(required.get(relative_path) or {}).get("exists"))
        ]
        if missing_required:
            reasons.append(f"required_public_output_missing:{missing_required[0]}")
    if not bool(cost_attribution.get("complete")):
        errors = list(cost_attribution.get("errors") or [])
        suffix = f":{errors[0]}" if errors else ""
        reasons.append(f"flat_output_cost_attribution_incomplete{suffix}")
    for field in ("user_time_s", "system_time_s", "cpu_percent_value", "max_rss_kb"):
        if time_v.get(field) is None:
            reasons.append(f"resource_metric_missing:{field}")
            break
    if arm.native:
        if bool(native_status.get("hidden_fallback_seen")):
            reasons.append("hidden_native_fallback")
        if bool(native_status.get("native_fail_closed_seen")):
            reasons.append("native_runtime_fail_closed")
        if native_worker_audit is None or not bool(native_worker_audit.get("ok")):
            reasons.append("native_worker_promotion_audit_failed")
    return list(dict.fromkeys(reasons))


def _run_cell(
    py: Path,
    arm: Arm,
    cell: Cell,
    rep: str,
    evidence_root: Path,
    native_extension: Path | None,
    *,
    flat_output_cost_attribution: str = "off",
) -> dict[str, Any]:
    if flat_output_cost_attribution not in FLAT_OUTPUT_COST_ATTRIBUTION_MODES:
        raise ValueError(
            f"unsupported flat output cost attribution mode: {flat_output_cost_attribution}"
        )
    runs_dir = evidence_root / "campaign_results" / rep / arm.name / "runs"
    run_id = f"{rep}_{arm.name}_{cell.cell_id}"
    run_dir = runs_dir / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    log_dir = evidence_root / "logs" / rep / arm.name
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"{run_id}.out"
    time_path = log_dir / f"{run_id}.time"
    simulation_cmd = _cell_command(py, arm, cell, run_id, runs_dir)
    cmd = list(simulation_cmd)
    env, timed_env_record = _timed_subprocess_env(arm.project_root)
    native_worker_audit_path: Path | None = None
    sidecar_path: Path | None = None
    if arm.reference == "historical":
        cost_attribution: dict[str, Any] = {
            "mode": flat_output_cost_attribution,
            "applicability": "not_applicable",
            "reason": "historical_arm_does_not_implement_current_sidecar_contract",
            "sidecar_path": None,
            "loaded": False,
            "valid": True,
            "complete": True,
            "errors": [],
            "sample": None,
        }
    elif flat_output_cost_attribution == "off":
        cost_attribution = {
            "mode": "off",
            "applicability": "disabled",
            "sidecar_path": None,
            "loaded": False,
            "valid": True,
            "complete": True,
            "errors": [],
            "sample": None,
        }
    else:
        sidecar_path = _flat_output_cost_sidecar_path(evidence_root, rep, arm, run_id)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.unlink(missing_ok=True)
        env[FLAT_OUTPUT_COST_SIDECAR_ENV] = str(sidecar_path)
        cost_attribution = {
            "mode": flat_output_cost_attribution,
            "applicability": "applicable",
            "sidecar_path": str(sidecar_path),
            "loaded": False,
            "valid": False,
            "complete": False,
            "errors": ["subprocess_not_finished"],
            "sample": None,
        }
    if arm.native:
        if native_extension is None:
            raise RuntimeError("native arm requested without native extension")
        exact_native_extension = native_extension.resolve(strict=True)
        env["EIEL_NATIVE_RUNTIME_EXTENSION"] = str(exact_native_extension)
        native_worker_audit_path = (
            evidence_root / "native_worker_audits" / rep / arm.name / f"{run_id}.json"
        )
        native_worker_audit_path.parent.mkdir(parents=True, exist_ok=True)
        native_worker_audit_path.unlink(missing_ok=True)
        cmd = _native_worker_command(
            simulation_cmd,
            audit_path=native_worker_audit_path,
            native_extension=exact_native_extension,
            expected_sha256=_sha256(exact_native_extension),
            expected_source_commit=_git(["rev-parse", "HEAD"], arm.project_root),
        )
    timed_cmd = ["/usr/bin/time", "-v", "-o", str(time_path), *cmd] if Path("/usr/bin/time").exists() else cmd
    started = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as out:
        proc = subprocess.run(timed_cmd, cwd=str(arm.project_root), env=env, stdout=out, stderr=subprocess.STDOUT, text=True)
    wall_s = time.perf_counter() - started
    if sidecar_path is not None:
        cost_attribution = {
            "mode": flat_output_cost_attribution,
            **_load_flat_output_cost_sample(sidecar_path),
        }
    output_profile = _run_output_profile(run_dir)
    metrics = _last_metrics(run_dir)
    native_status = _scan_native_status(run_dir, out_path) if arm.native else {}
    finalization = _run_finalization_status(run_dir)
    time_v = _parse_time_v(time_path)
    native_worker_audit = (
        _load_native_worker_audit(native_worker_audit_path)
        if native_worker_audit_path is not None
        else None
    )
    exclusion_reasons = _cell_exclusion_reasons(
        arm=arm,
        returncode=int(proc.returncode),
        run_dir=run_dir,
        finalization=finalization,
        metrics=metrics,
        output_profile=output_profile,
        cost_attribution=cost_attribution,
        time_v=time_v,
        native_status=native_status,
        native_worker_audit=native_worker_audit,
        generations=cell.generations,
    )
    complete = not exclusion_reasons
    result = {
        "schema_version": CELL_RESULT_SCHEMA,
        "rep": rep,
        "arm": arm.name,
        "cell_id": cell.cell_id,
        "seed": cell.seed,
        "generations": cell.generations,
        "steps": cell.steps,
        "population": cell.pop,
        "grid_size": cell.grid_size,
        "churn": cell.churn,
        "checkpoint_mode": cell.checkpoint_mode,
        "checkpoint_period": cell.checkpoint_period,
        "workers_for_arm": arm.workers,
        "native": arm.native,
        "command": cmd,
        "simulation_command": simulation_cmd,
        "cwd": str(arm.project_root),
        "run_dir": str(run_dir),
        "stdout_path": str(out_path),
        "time_path": str(time_path),
        "returncode": int(proc.returncode),
        "wall_s": float(wall_s),
        "wall_time_source": "external_time.perf_counter_around_subprocess",
        "output_bytes": int(output_profile["output_bytes"]),
        "output_files": int(output_profile["output_files"]),
        "checkpoint_bytes": int(output_profile["checkpoint_bytes"]),
        "checkpoint_files": int(output_profile["checkpoint_files"]),
        "output_profile": output_profile,
        "flat_output_cost_attribution": cost_attribution,
        "timed_subprocess_environment": timed_env_record,
        "time_v": time_v,
        "metrics": metrics,
        "run_finalization": finalization,
        "native_status": native_status,
        "native_worker_audit": native_worker_audit,
        "promotion_eligible": bool(complete),
        "exclusion_reasons": exclusion_reasons,
        "first_failure_cause": exclusion_reasons[0] if exclusion_reasons else None,
        "status": "completed" if complete else "excluded_incomplete",
        "complete": bool(complete),
    }
    _write_json(log_dir / f"{run_id}.result.json", result)
    return result


def _run_arm(
    py: Path,
    arm: Arm,
    cells: list[Cell],
    rep: str,
    evidence_root: Path,
    native_extension: Path | None,
    *,
    flat_output_cost_attribution: str = "off",
) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=arm.workers) as pool:
        futures = {
            pool.submit(
                _run_cell,
                py,
                arm,
                cell,
                rep,
                evidence_root,
                native_extension,
                flat_output_cost_attribution=flat_output_cost_attribution,
            ): cell
            for cell in cells
        }
        for future in concurrent.futures.as_completed(futures):
            cell = futures[future]
            try:
                result = future.result()
            except BaseException as exc:
                first_cause = f"benchmark_runner_exception:{type(exc).__name__}"
                result = {
                    "schema_version": CELL_RESULT_SCHEMA,
                    "rep": rep,
                    "arm": arm.name,
                    "cell_id": cell.cell_id,
                    "seed": cell.seed,
                    "generations": cell.generations,
                    "steps": cell.steps,
                    "workers_for_arm": arm.workers,
                    "native": arm.native,
                    "returncode": None,
                    "wall_s": None,
                    "output_bytes": 0,
                    "output_files": 0,
                    "checkpoint_bytes": 0,
                    "checkpoint_files": 0,
                    "output_profile": {},
                    "flat_output_cost_attribution": {},
                    "time_v": {},
                    "metrics": {},
                    "native_status": {},
                    "native_worker_audit": None,
                    "promotion_eligible": False,
                    "exclusion_reasons": [first_cause],
                    "first_failure_cause": first_cause,
                    "status": "excluded_runner_failure",
                    "complete": False,
                    "runner_exception": f"{type(exc).__name__}: {exc}",
                }
                failure_path = (
                    evidence_root
                    / "logs"
                    / rep
                    / arm.name
                    / f"{rep}_{arm.name}_{cell.cell_id}.result.json"
                )
                _write_json(failure_path, result)
            results.append(result)
            print(
                f"[{rep} {arm.name}] {len(results)}/{len(cells)} "
                f"{result['cell_id']} rc={result['returncode']} wall={result['wall_s']}",
                flush=True,
            )
    wall_s = time.perf_counter() - started
    results.sort(key=lambda row: str(row["cell_id"]))
    incomplete = [row for row in results if not bool(row.get("promotion_eligible"))]
    eligible_results = [row for row in results if bool(row.get("promotion_eligible"))]
    time_v_rows = [dict(row.get("time_v") or {}) for row in eligible_results]
    resource_metrics_complete = bool(eligible_results) and all(
        row.get("user_time_s") is not None
        and row.get("system_time_s") is not None
        and row.get("cpu_percent_value") is not None
        and row.get("max_rss_kb") is not None
        for row in time_v_rows
    )
    child_user_time_s = float(sum(float(row.get("user_time_s") or 0.0) for row in time_v_rows))
    child_system_time_s = float(sum(float(row.get("system_time_s") or 0.0) for row in time_v_rows))
    child_cpu_time_s = child_user_time_s + child_system_time_s
    native_audits = [dict(row.get("native_worker_audit") or {}) for row in results] if arm.native else []
    native_identity_rows = [
        (
            audit.get("extension_path"),
            audit.get("extension_sha256"),
            audit.get("extension_abi_version"),
            audit.get("extension_version"),
        )
        for audit in native_audits
    ]
    native_worker_gate_ok = (not arm.native) or (
        bool(native_audits)
        and len(native_audits) == len(results)
        and all(bool(audit.get("ok")) for audit in native_audits)
        and len(set(native_identity_rows)) == 1
    )
    campaign_exclusion_reasons: list[str] = []
    if incomplete:
        campaign_exclusion_reasons.append("failed_or_incomplete_cell")
    if arm.native and not native_worker_gate_ok:
        campaign_exclusion_reasons.append("native_worker_identity_or_runtime_contract_failed")
    summary = {
        "schema_version": "eiel_four_path_arm_campaign_result_v1",
        "rep": rep,
        "arm": arm.name,
        "reference": arm.reference,
        "workers": arm.workers,
        "native": arm.native,
        "wall_s": float(wall_s),
        "cells": len(results),
        "attempted_cells": len(results),
        "complete_cells": len(eligible_results),
        "incomplete_cells": len(incomplete),
        "cell_generations": int(sum(int(row["generations"]) for row in eligible_results)),
        "cell_steps": int(sum(int(row["generations"]) * int(row["steps"]) for row in eligible_results)),
        "cells_per_second": float(len(eligible_results) / max(wall_s, 1.0e-9)),
        "cell_generations_per_second": float(sum(int(row["generations"]) for row in eligible_results) / max(wall_s, 1.0e-9)),
        "steps_per_second": float(sum(int(row["generations"]) * int(row["steps"]) for row in eligible_results) / max(wall_s, 1.0e-9)),
        "output_bytes": int(sum(int(row["output_bytes"]) for row in eligible_results)),
        "output_files": int(sum(int(row["output_files"]) for row in eligible_results)),
        "checkpoint_bytes": int(sum(int(row["checkpoint_bytes"]) for row in eligible_results)),
        "checkpoint_files": int(sum(int(row["checkpoint_files"]) for row in eligible_results)),
        "peak_cell_rss_kb": max(
            [int(row.get("time_v", {}).get("max_rss_kb") or 0) for row in eligible_results] or [0]
        ),
        "sum_cell_peak_rss_kb": int(sum(int(row.get("max_rss_kb") or 0) for row in time_v_rows)),
        "child_user_time_s": child_user_time_s,
        "child_system_time_s": child_system_time_s,
        "child_cpu_time_s": child_cpu_time_s,
        "mean_cell_cpu_percent": (
            statistics.mean(float(row["cpu_percent_value"]) for row in time_v_rows)
            if resource_metrics_complete
            else None
        ),
        "worker_utilization_ratio": child_cpu_time_s / max(wall_s * arm.workers, 1.0e-9),
        "resource_metrics_complete": resource_metrics_complete,
        "flat_output_cost_attribution_mode": flat_output_cost_attribution,
        "flat_output_cost_attribution_complete": bool(eligible_results) and all(
            bool(row.get("flat_output_cost_attribution", {}).get("complete"))
            for row in eligible_results
        ),
        "resource_metric_definitions": {
            "peak_cell_rss_kb": "maximum GNU time -v RSS among cell subprocesses; not summed concurrent RSS",
            "sum_cell_peak_rss_kb": "sum of per-cell peak RSS; workload burden, not simultaneous peak memory",
            "worker_utilization_ratio": "sum child user+system CPU seconds divided by campaign wall seconds and configured workers",
        },
        "any_hidden_fallback": any(bool(row.get("native_status", {}).get("hidden_fallback_seen")) for row in results),
        "any_native_fail_closed": any(bool(row.get("native_status", {}).get("native_fail_closed_seen")) for row in results),
        "any_runtime_jit": any(audit.get("runtime_jit") is not False for audit in native_audits) if arm.native else False,
        "any_mixed_ownership": any(audit.get("mixed_ownership") is not False for audit in native_audits) if arm.native else False,
        "any_stale_extension": any(audit.get("stale_extension") is not False for audit in native_audits) if arm.native else False,
        "all_expected_native_ownership": all(audit.get("native_ownership_active") is True for audit in native_audits) if arm.native and native_audits else not arm.native,
        "native_worker_identity_homogeneous": len(set(native_identity_rows)) == 1 if native_identity_rows else not arm.native,
        "native_worker_gate_ok": native_worker_gate_ok,
        "promotion_eligible": not campaign_exclusion_reasons,
        "exclusion_reasons": campaign_exclusion_reasons,
        "first_failure_cause": (
            incomplete[0].get("first_failure_cause")
            if incomplete
            else (campaign_exclusion_reasons[0] if campaign_exclusion_reasons else None)
        ),
        "results": eligible_results,
        "excluded_results": incomplete,
    }
    if incomplete:
        summary["incomplete_cell_ids"] = [row["cell_id"] for row in incomplete]
    _write_json(evidence_root / "campaign_results" / rep / arm.name / "arm_summary.json", summary)
    return summary


def _stats(values: Iterable[float]) -> dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {"count": 0}
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": mean,
        "median": statistics.median(vals),
        "stdev": stdev,
        "cv": float(stdev / mean) if mean else None,
    }


def _cell_results_by_arm(campaigns: list[dict[str, Any]]) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    indexed: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for campaign in campaigns:
        arm = str(campaign["arm"])
        for row in campaign.get("results", []):
            key = (str(row["rep"]), str(row["cell_id"]))
            indexed.setdefault(arm, {})[key] = row
    return indexed


def _shared_metric_parity(campaigns: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = _cell_results_by_arm(campaigns)
    pair_results: list[dict[str, Any]] = []
    total_pairs = 0
    total_mismatches = 0
    first_mismatch: dict[str, Any] | None = None
    for reference_arm, native_arm in CURRENT_REFERENCE_NATIVE_PAIR_SPECS:
        reference_rows = by_arm.get(reference_arm)
        native_rows = by_arm.get(native_arm)
        if not reference_rows or not native_rows:
            continue
        common_keys = sorted(set(reference_rows) & set(native_rows))
        pair_mismatches = 0
        pair_first: dict[str, Any] | None = None
        for rep, cell_id in common_keys:
            reference_metrics = reference_rows[(rep, cell_id)].get("metrics", {})
            native_metrics = native_rows[(rep, cell_id)].get("metrics", {})
            differences = {
                key: {
                    "reference": reference_metrics.get(key),
                    "native": native_metrics.get(key),
                }
                for key in SHARED_METRIC_KEYS
                if reference_metrics.get(key) != native_metrics.get(key)
            }
            if differences:
                pair_mismatches += 1
                mismatch = {
                    "reference_arm": reference_arm,
                    "native_arm": native_arm,
                    "rep": rep,
                    "cell_id": cell_id,
                    "differences": differences,
                }
                if pair_first is None:
                    pair_first = mismatch
                if first_mismatch is None:
                    first_mismatch = mismatch
        total_pairs += len(common_keys)
        total_mismatches += pair_mismatches
        pair_results.append(
            {
                "reference_arm": reference_arm,
                "native_arm": native_arm,
                "paired_cells": len(common_keys),
                "mismatched_cells": pair_mismatches,
                "ok": pair_mismatches == 0 and len(common_keys) > 0,
                "first_mismatch": pair_first,
            }
        )
    return {
        "schema_version": "eiel_shared_metric_parity_v1",
        "metric_keys": list(SHARED_METRIC_KEYS),
        "pairs": pair_results,
        "paired_cells": total_pairs,
        "mismatched_cells": total_mismatches,
        "ok": bool(pair_results) and total_mismatches == 0 and total_pairs > 0,
        "first_mismatch": first_mismatch,
    }


def _matched_output_checkpoint_profile(campaigns: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = _cell_results_by_arm(campaigns)
    pair_results: list[dict[str, Any]] = []
    total_paired_cells = 0
    total_missing_cells = 0
    total_mismatched_cells = 0
    total_raw_metrics_summary_inequalities = 0
    first_mismatch: dict[str, Any] | None = None
    first_raw_metrics_summary_inequality: dict[str, Any] | None = None

    for reference_arm, native_arm in CURRENT_REFERENCE_NATIVE_PAIR_SPECS:
        reference_rows = by_arm.get(reference_arm)
        native_rows = by_arm.get(native_arm)
        if not reference_rows or not native_rows:
            continue
        reference_keys = set(reference_rows)
        native_keys = set(native_rows)
        common_keys = sorted(reference_keys & native_keys)
        all_keys = sorted(reference_keys | native_keys)
        pair_missing_cells = 0
        pair_mismatched_cells = 0
        pair_raw_inequalities = 0
        pair_first_mismatch: dict[str, Any] | None = None

        for rep, cell_id in all_keys:
            reference_row = reference_rows.get((rep, cell_id))
            native_row = native_rows.get((rep, cell_id))
            if reference_row is None or native_row is None:
                mismatch = {
                    "reference_arm": reference_arm,
                    "native_arm": native_arm,
                    "rep": rep,
                    "cell_id": cell_id,
                    "reasons": ["missing_paired_cell"],
                    "reference_present": reference_row is not None,
                    "native_present": native_row is not None,
                }
                pair_missing_cells += 1
                pair_mismatched_cells += 1
                if pair_first_mismatch is None:
                    pair_first_mismatch = mismatch
                if first_mismatch is None:
                    first_mismatch = mismatch
                continue

            reference_profile = dict(reference_row.get("output_profile") or {})
            native_profile = dict(native_row.get("output_profile") or {})
            reference_files = {
                relative_path
                for relative_path in reference_profile.get("relative_files") or []
                if not _is_checkpoint_output(str(relative_path))
            }
            native_files = {
                relative_path
                for relative_path in native_profile.get("relative_files") or []
                if not _is_checkpoint_output(str(relative_path))
            }
            reference_only_files = sorted(reference_files - native_files)
            native_only_files = sorted(native_files - reference_files)
            output_file_profile_equal = not reference_only_files and not native_only_files
            reference_required = dict(reference_profile.get("required_public_outputs") or {})
            native_required = dict(native_profile.get("required_public_outputs") or {})
            missing_required_public_outputs = sorted(
                relative_path
                for relative_path in REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES
                if not bool(dict(reference_required.get(relative_path) or {}).get("exists"))
                or not bool(dict(native_required.get(relative_path) or {}).get("exists"))
            )
            exact_public_output_mismatches = sorted(
                relative_path
                for relative_path in EXACT_CURRENT_PUBLIC_OUTPUT_FILES
                if relative_path not in missing_required_public_outputs
                and dict(reference_required.get(relative_path) or {}).get("sha256")
                != dict(native_required.get(relative_path) or {}).get("sha256")
            )
            reference_inventory = {
                str(entry["relative_path"]): dict(entry)
                for entry in reference_profile.get("output_inventory") or []
            }
            native_inventory = {
                str(entry["relative_path"]): dict(entry)
                for entry in native_profile.get("output_inventory") or []
            }
            inventory_equivalence: list[dict[str, Any]] = []
            output_row_count_mismatches: list[str] = []
            output_row_count_errors: list[str] = []
            debug_only_outputs_enabled: list[str] = []
            for relative_path in sorted(set(reference_inventory) | set(native_inventory)):
                reference_entry = reference_inventory.get(relative_path, {})
                native_entry = native_inventory.get(relative_path, {})
                if reference_entry.get("row_count_error") or native_entry.get("row_count_error"):
                    output_row_count_errors.append(relative_path)
                if (
                    reference_entry.get("requirement_class") == "debug_only"
                    or native_entry.get("requirement_class") == "debug_only"
                ) and (reference_entry.get("enabled") or native_entry.get("enabled")):
                    debug_only_outputs_enabled.append(relative_path)
                reference_rows_count = reference_entry.get("rows")
                native_rows_count = native_entry.get("rows")
                rows_comparable = (
                    bool(reference_entry.get("enabled"))
                    and bool(native_entry.get("enabled"))
                    and (reference_rows_count is not None or native_rows_count is not None)
                )
                row_count_equal = not rows_comparable or reference_rows_count == native_rows_count
                if not row_count_equal:
                    output_row_count_mismatches.append(relative_path)
                inventory_equivalence.append(
                    {
                        "relative_path": relative_path,
                        "purpose": reference_entry.get("purpose") or native_entry.get("purpose"),
                        "requirement_class": reference_entry.get("requirement_class")
                        or native_entry.get("requirement_class"),
                        "reference_enabled": bool(reference_entry.get("enabled")),
                        "native_enabled": bool(native_entry.get("enabled")),
                        "reference_rows": reference_rows_count,
                        "native_rows": native_rows_count,
                        "row_count_equal": row_count_equal,
                        "reference_bytes": int(reference_entry.get("bytes") or 0),
                        "native_bytes": int(native_entry.get("bytes") or 0),
                        "bytes_equal": reference_entry.get("bytes") == native_entry.get("bytes"),
                        "sha256_equal": reference_entry.get("sha256") == native_entry.get("sha256"),
                    }
                )

            reference_checkpoint_files = list(reference_profile.get("checkpoint_relative_files") or [])
            native_checkpoint_files = list(native_profile.get("checkpoint_relative_files") or [])
            reference_canonical_checkpoint = dict(
                reference_profile.get("canonical_checkpoint_storage") or {}
            )
            native_canonical_checkpoint = dict(
                native_profile.get("canonical_checkpoint_storage") or {}
            )
            checkpoint_configuration_equal = (
                reference_row.get("checkpoint_mode") == native_row.get("checkpoint_mode")
                and reference_row.get("checkpoint_period") == native_row.get("checkpoint_period")
            )
            checkpoint_enabled = int(reference_row.get("checkpoint_period") or 0) > 0
            reference_canonical_entries = list(
                reference_canonical_checkpoint.get("entries") or []
            )
            native_canonical_entries = list(native_canonical_checkpoint.get("entries") or [])
            canonical_checkpoint_storage_equal = (
                bool(reference_canonical_checkpoint.get("ok", False))
                and bool(native_canonical_checkpoint.get("ok", False))
                and reference_canonical_entries == native_canonical_entries
                and (not checkpoint_enabled or bool(reference_canonical_entries))
            )
            checkpoint_profile_equal = (
                checkpoint_configuration_equal
                and canonical_checkpoint_storage_equal
            )
            metrics_summary = _metrics_summary_pair_fairness(
                dict(reference_profile.get("metrics_summary") or {}),
                dict(native_profile.get("metrics_summary") or {}),
            )
            metrics_csv = _metrics_csv_pair_fairness(
                dict(reference_profile.get("metrics_csv") or {}),
                dict(native_profile.get("metrics_csv") or {}),
            )

            reasons: list[str] = []
            if not output_file_profile_equal:
                reasons.append("output_file_profile_mismatch")
            if missing_required_public_outputs:
                reasons.append("required_public_output_missing")
            if exact_public_output_mismatches:
                reasons.append("exact_public_output_byte_mismatch")
            if output_row_count_mismatches:
                reasons.append("output_row_count_mismatch")
            if output_row_count_errors:
                reasons.append("output_row_count_error")
            if debug_only_outputs_enabled:
                reasons.append("debug_only_output_enabled")
            if not checkpoint_profile_equal:
                reasons.append("checkpoint_profile_mismatch")
            if not bool(metrics_csv["ok"]):
                reasons.append("metrics_csv_full_semantic_mismatch")
            if not bool(metrics_summary["ok"]):
                reasons.append("metrics_summary_semantic_mismatch")

            if bool(metrics_summary["raw_byte_inequality"]):
                pair_raw_inequalities += 1
                raw_inequality = {
                    "reference_arm": reference_arm,
                    "native_arm": native_arm,
                    "rep": rep,
                    "cell_id": cell_id,
                    "reference_raw_bytes": metrics_summary["reference_raw_bytes"],
                    "native_raw_bytes": metrics_summary["native_raw_bytes"],
                    "reference_raw_sha256": metrics_summary["reference_raw_sha256"],
                    "native_raw_sha256": metrics_summary["native_raw_sha256"],
                    "reference_avg_gen_time_s_values": metrics_summary["reference_avg_gen_time_s_values"],
                    "native_avg_gen_time_s_values": metrics_summary["native_avg_gen_time_s_values"],
                }
                if first_raw_metrics_summary_inequality is None:
                    first_raw_metrics_summary_inequality = raw_inequality

            if reasons:
                mismatch = {
                    "reference_arm": reference_arm,
                    "native_arm": native_arm,
                    "rep": rep,
                    "cell_id": cell_id,
                    "reasons": reasons,
                    "reference_only_files": reference_only_files,
                    "native_only_files": native_only_files,
                    "missing_required_public_outputs": missing_required_public_outputs,
                    "exact_public_output_mismatches": exact_public_output_mismatches,
                    "output_row_count_mismatches": output_row_count_mismatches,
                    "output_row_count_errors": output_row_count_errors,
                    "debug_only_outputs_enabled": debug_only_outputs_enabled,
                    "output_inventory_equivalence": inventory_equivalence,
                    "reference_required_public_outputs": reference_required,
                    "native_required_public_outputs": native_required,
                    "checkpoint_configuration_equal": checkpoint_configuration_equal,
                    "canonical_checkpoint_storage_equal": canonical_checkpoint_storage_equal,
                    "reference_canonical_checkpoint_storage": reference_canonical_checkpoint,
                    "native_canonical_checkpoint_storage": native_canonical_checkpoint,
                    "reference_checkpoint_files": reference_checkpoint_files,
                    "native_checkpoint_files": native_checkpoint_files,
                    "reference_checkpoint_bytes": reference_row.get("checkpoint_bytes"),
                    "native_checkpoint_bytes": native_row.get("checkpoint_bytes"),
                    "metrics_csv": metrics_csv,
                    "metrics_summary": metrics_summary,
                }
                pair_mismatched_cells += 1
                if pair_first_mismatch is None:
                    pair_first_mismatch = mismatch
                if first_mismatch is None:
                    first_mismatch = mismatch

        total_paired_cells += len(common_keys)
        total_missing_cells += pair_missing_cells
        total_mismatched_cells += pair_mismatched_cells
        total_raw_metrics_summary_inequalities += pair_raw_inequalities
        pair_results.append(
            {
                "reference_arm": reference_arm,
                "native_arm": native_arm,
                "expected_cells": len(all_keys),
                "paired_cells": len(common_keys),
                "missing_cells": pair_missing_cells,
                "mismatched_cells": pair_mismatched_cells,
                "raw_metrics_summary_inequalities": pair_raw_inequalities,
                "first_mismatch": pair_first_mismatch,
                "ok": pair_mismatched_cells == 0 and len(common_keys) > 0,
            }
        )

    return {
        "schema_version": "eiel_matched_output_checkpoint_profile_v1",
        "pairs": pair_results,
        "paired_cells": total_paired_cells,
        "missing_cells": total_missing_cells,
        "mismatched_cells": total_mismatched_cells,
        "raw_metrics_summary_inequalities": total_raw_metrics_summary_inequalities,
        "raw_metrics_summary_inequality_is_diagnostic_only": True,
        "first_mismatch": first_mismatch,
        "first_raw_metrics_summary_inequality": first_raw_metrics_summary_inequality,
        "ok": bool(pair_results) and total_mismatched_cells == 0 and total_paired_cells > 0,
    }


def _output_burden_fairness(
    arms: dict[str, dict[str, Any]],
    campaigns: list[dict[str, Any]] | None = None,
    *,
    min_native_to_reference_ratio: float = OUTPUT_BURDEN_NATIVE_TO_REFERENCE_MIN_RATIO,
    max_native_to_reference_ratio: float = OUTPUT_BURDEN_NATIVE_TO_REFERENCE_MAX_RATIO,
) -> dict[str, Any]:
    by_arm = _cell_results_by_arm(campaigns or [])
    per_cell_checks_required = campaigns is not None
    pairs: list[dict[str, Any]] = []
    total_paired_cells = 0
    total_missing_cells = 0
    total_out_of_bounds_cells = 0
    first_offending_cell: dict[str, Any] | None = None
    for reference_arm, native_arm in CURRENT_REFERENCE_NATIVE_PAIR_SPECS:
        ref_stats = arms.get(reference_arm, {}).get("output_bytes", {})
        native_stats = arms.get(native_arm, {}).get("output_bytes", {})
        if "median" not in ref_stats or "median" not in native_stats:
            continue
        reference_median = float(ref_stats["median"])
        native_median = float(native_stats["median"])
        ratio = native_median / reference_median if reference_median > 0.0 else None
        ok = (
            ratio is not None
            and math.isfinite(ratio)
            and min_native_to_reference_ratio <= ratio <= max_native_to_reference_ratio
        )
        reference_rows = by_arm.get(reference_arm, {})
        native_rows = by_arm.get(native_arm, {})
        reference_keys = set(reference_rows)
        native_keys = set(native_rows)
        common_keys = sorted(reference_keys & native_keys)
        missing_keys = sorted(reference_keys ^ native_keys)
        pair_out_of_bounds_cells = 0
        pair_first_offending_cell: dict[str, Any] | None = None
        cell_ratios: list[float] = []
        for rep, cell_id in common_keys:
            reference_value = reference_rows[(rep, cell_id)].get("output_bytes")
            native_value = native_rows[(rep, cell_id)].get("output_bytes")
            try:
                reference_bytes = float(reference_value)
                native_bytes = float(native_value)
            except (TypeError, ValueError):
                reference_bytes = native_bytes = math.nan

            cell_ratio = (
                native_bytes / reference_bytes
                if math.isfinite(reference_bytes)
                and math.isfinite(native_bytes)
                and reference_bytes > 0.0
                and native_bytes >= 0.0
                else None
            )
            cell_ok = (
                cell_ratio is not None
                and math.isfinite(cell_ratio)
                and min_native_to_reference_ratio
                <= cell_ratio
                <= max_native_to_reference_ratio
            )
            if cell_ratio is not None and math.isfinite(cell_ratio):
                cell_ratios.append(cell_ratio)
            if not cell_ok:
                pair_out_of_bounds_cells += 1
                offender = {
                    "reference_arm": reference_arm,
                    "native_arm": native_arm,
                    "rep": rep,
                    "cell_id": cell_id,
                    "reference_output_bytes": reference_value,
                    "native_output_bytes": native_value,
                    "native_to_reference_ratio": cell_ratio,
                    "min_allowed_ratio": min_native_to_reference_ratio,
                    "max_allowed_ratio": max_native_to_reference_ratio,
                    "reason": (
                        "ratio_out_of_bounds"
                        if cell_ratio is not None and math.isfinite(cell_ratio)
                        else "invalid_output_bytes_for_ratio"
                    ),
                }
                if pair_first_offending_cell is None:
                    pair_first_offending_cell = offender
                if first_offending_cell is None:
                    first_offending_cell = offender

        per_cell_ok = (
            len(common_keys) > 0
            and not missing_keys
            and pair_out_of_bounds_cells == 0
        )
        pair_ok = ok and (per_cell_ok if per_cell_checks_required else True)
        total_paired_cells += len(common_keys)
        total_missing_cells += len(missing_keys)
        total_out_of_bounds_cells += pair_out_of_bounds_cells
        pairs.append(
            {
                "reference_arm": reference_arm,
                "native_arm": native_arm,
                "reference_median_output_bytes": reference_median,
                "native_median_output_bytes": native_median,
                "native_to_reference_ratio": ratio,
                "min_allowed_ratio": min_native_to_reference_ratio,
                "max_allowed_ratio": max_native_to_reference_ratio,
                "aggregate_median_ok": ok,
                "per_cell_checks_required": per_cell_checks_required,
                "paired_cells": len(common_keys),
                "missing_cells": len(missing_keys),
                "first_missing_cell": (
                    {"rep": missing_keys[0][0], "cell_id": missing_keys[0][1]}
                    if missing_keys
                    else None
                ),
                "out_of_bounds_cells": pair_out_of_bounds_cells,
                "cell_native_to_reference_ratio": _stats(cell_ratios),
                "first_offending_cell": pair_first_offending_cell,
                "per_cell_ok": per_cell_ok if per_cell_checks_required else None,
                "ok": pair_ok,
            }
        )
    return {
        "schema_version": "eiel_output_burden_fairness_v1",
        "per_cell_checks_required": per_cell_checks_required,
        "pairs": pairs,
        "paired_cells": total_paired_cells,
        "missing_cells": total_missing_cells,
        "out_of_bounds_cells": total_out_of_bounds_cells,
        "first_offending_cell": first_offending_cell,
        "ok": bool(pairs) and all(bool(pair["ok"]) for pair in pairs),
    }


def _aggregate_output_inventory(campaigns: list[dict[str, Any]]) -> dict[str, Any]:
    accumulators: dict[tuple[str, str], dict[str, Any]] = {}
    expected_runs_by_arm: dict[str, int] = {}
    for campaign in campaigns:
        arm = str(campaign["arm"])
        for result in campaign.get("results", []):
            expected_runs_by_arm[arm] = expected_runs_by_arm.get(arm, 0) + 1
            for entry in dict(result.get("output_profile") or {}).get("output_inventory") or []:
                relative_path = str(entry["relative_path"])
                key = (arm, relative_path)
                accumulator = accumulators.setdefault(
                    key,
                    {
                        "arm": arm,
                        "relative_path": relative_path,
                        "purpose": entry.get("purpose"),
                        "requirement_class": entry.get("requirement_class"),
                        "required": bool(entry.get("required")),
                        "debug_only": bool(entry.get("debug_only")),
                        "enabled_runs": 0,
                        "count": 0,
                        "rows": [],
                        "bytes": [],
                    },
                )
                if entry.get("enabled"):
                    accumulator["enabled_runs"] += 1
                accumulator["count"] += int(entry.get("count") or 0)
                if entry.get("rows") is not None:
                    accumulator["rows"].append(int(entry["rows"]))
                accumulator["bytes"].append(int(entry.get("bytes") or 0))

    entries: list[dict[str, Any]] = []
    for (arm, _), accumulator in sorted(accumulators.items()):
        rows = list(accumulator.pop("rows"))
        byte_values = list(accumulator.pop("bytes"))
        expected_runs = expected_runs_by_arm.get(arm, 0)
        entries.append(
            accumulator
            | {
                "expected_runs": expected_runs,
                "enabled_for_all_runs": accumulator["enabled_runs"] == expected_runs,
                "row_count": _stats(rows),
                "byte_count": _stats(byte_values),
            }
        )
    entries_by_arm_path = {
        (str(entry["arm"]), str(entry["relative_path"])): entry
        for entry in entries
    }
    reference_native_equivalence: list[dict[str, Any]] = []
    for reference_arm, native_arm in CURRENT_REFERENCE_NATIVE_PAIR_SPECS:
        if reference_arm not in expected_runs_by_arm or native_arm not in expected_runs_by_arm:
            continue
        paths = sorted(
            {path for arm, path in entries_by_arm_path if arm == reference_arm}
            | {path for arm, path in entries_by_arm_path if arm == native_arm}
        )
        for relative_path in paths:
            reference = entries_by_arm_path.get((reference_arm, relative_path), {})
            native = entries_by_arm_path.get((native_arm, relative_path), {})
            classification_equal = (
                reference.get("purpose") == native.get("purpose")
                and reference.get("requirement_class") == native.get("requirement_class")
            )
            enabled_count_equal = reference.get("enabled_runs") == native.get("enabled_runs")
            file_count_equal = reference.get("count") == native.get("count")
            row_count_equal = reference.get("row_count") == native.get("row_count")
            reference_bytes = dict(reference.get("byte_count") or {}).get("median")
            native_bytes = dict(native.get("byte_count") or {}).get("median")
            byte_ratio = (
                float(native_bytes) / float(reference_bytes)
                if reference_bytes not in (None, 0) and native_bytes is not None
                else None
            )
            reference_native_equivalence.append(
                {
                    "reference_arm": reference_arm,
                    "native_arm": native_arm,
                    "relative_path": relative_path,
                    "purpose": reference.get("purpose") or native.get("purpose"),
                    "requirement_class": reference.get("requirement_class")
                    or native.get("requirement_class"),
                    "classification_equal": classification_equal,
                    "enabled_count_equal": enabled_count_equal,
                    "file_count_equal": file_count_equal,
                    "row_count_equal": row_count_equal,
                    "reference_median_bytes": reference_bytes,
                    "native_median_bytes": native_bytes,
                    "native_to_reference_byte_ratio": byte_ratio,
                    "equivalent": classification_equal
                    and enabled_count_equal
                    and file_count_equal
                    and row_count_equal,
                }
            )
    return {
        "schema_version": "eiel_flat_output_inventory_v1",
        "entries": entries,
        "reference_native_equivalence": reference_native_equivalence,
        "reference_native_equivalence_ok": bool(reference_native_equivalence)
        and all(bool(row["equivalent"]) for row in reference_native_equivalence),
        "classification_contract": {
            "required": "must be present in every current reference/native timed run",
            "debug_only": "diagnostic artifact excluded from the compact authoritative timed profile",
            "optional": "supporting artifact counted in file and byte burden when enabled",
        },
    }


def _repetition_plan(
    registry: dict[str, Any],
    arms: dict[str, Arm],
    selected_arm_names: list[str],
    *,
    global_override: int | None = None,
    w1_override: int | None = None,
    w6_override: int | None = None,
) -> dict[str, int]:
    configured = dict(registry.get("benchmark_repetitions") or {})
    configured_w1 = int(configured.get("w1", MINIMUM_W1_REPETITIONS))
    configured_w6 = int(configured.get("w6", MINIMUM_W6_REPETITIONS))
    if global_override is not None:
        configured_w1 = configured_w6 = int(global_override)
    if w1_override is not None:
        configured_w1 = int(w1_override)
    if w6_override is not None:
        configured_w6 = int(w6_override)
    if configured_w1 < MINIMUM_W1_REPETITIONS:
        raise ValueError(f"W1 repetitions must be >= {MINIMUM_W1_REPETITIONS}, got {configured_w1}")
    if configured_w6 < MINIMUM_W6_REPETITIONS:
        raise ValueError(f"W6 repetitions must be >= {MINIMUM_W6_REPETITIONS}, got {configured_w6}")
    return {
        name: configured_w1 if int(arms[name].workers) == 1 else configured_w6
        for name in selected_arm_names
    }


def _repetition_gate(
    campaigns: list[dict[str, Any]],
    required_repetitions_by_arm: dict[str, int] | None = None,
) -> dict[str, Any]:
    repetitions_by_arm: dict[str, set[str]] = {}
    for campaign in campaigns:
        repetitions_by_arm.setdefault(str(campaign["arm"]), set()).add(str(campaign["rep"]))
    if required_repetitions_by_arm is None:
        required_repetitions_by_arm = {
            arm: MINIMUM_W1_REPETITIONS if "_w1" in arm else MINIMUM_W6_REPETITIONS
            for arm in repetitions_by_arm
        }
    arms = []
    for arm, required in sorted(required_repetitions_by_arm.items()):
        actual = len(repetitions_by_arm.get(arm, set()))
        arms.append({"arm": arm, "required": int(required), "actual": actual, "ok": actual >= int(required)})
    return {
        "schema_version": "eiel_four_path_repetition_gate_v1",
        "w1_minimum": MINIMUM_W1_REPETITIONS,
        "w6_minimum": MINIMUM_W6_REPETITIONS,
        "arms": arms,
        "ok": bool(arms) and all(bool(row["ok"]) for row in arms),
    }


def _extension_integrity_check(path: Path, expected_sha256: str, phase: str) -> dict[str, Any]:
    exists = path.is_file()
    observed_sha256 = _sha256(path) if exists else None
    return {
        "phase": phase,
        "path": str(path),
        "exists": exists,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "ok": exists and observed_sha256 == expected_sha256,
    }


def _normalize_sha256(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("expected native extension SHA-256 must be exactly 64 hexadecimal characters")
    return normalized


def _promotion_policy_from_registry(registry: dict[str, Any]) -> dict[str, Any]:
    engine_root = PROJECT_ROOT / "engine"
    if str(engine_root) not in sys.path:
        sys.path.insert(0, str(engine_root))
    from native_runtime.promotion_envelope import (  # noqa: WPS433
        load_promotion_envelope,
        promotion_envelope_contract_path,
    )

    contract_path = promotion_envelope_contract_path()
    envelope = load_promotion_envelope(contract_path)
    thresholds = envelope.promotion_thresholds
    contract_values = {
        "current_w1_reference_to_native_min_speedup": float(
            thresholds.primary_performance.minimum_speedup
        ),
        "current_w6_reference_to_native_min_speedup": float(
            thresholds.secondary_performance.minimum_speedup
        ),
        "measured_arm_wall_time_cv_max": float(thresholds.stability.maximum),
        "output_native_to_reference_min_ratio": float(
            thresholds.output_fairness.minimum
        ),
        "output_native_to_reference_max_ratio": float(
            thresholds.output_fairness.maximum
        ),
        "native_peak_rss_to_same_cell_reference_max_ratio": float(
            thresholds.memory.maximum
        ),
    }
    configured = dict(registry.get("promotion_gates") or {})
    if configured:
        for field, expected in contract_values.items():
            if field not in configured:
                raise ValueError(
                    f"promotion_gates.{field} is required as a mirror of the validated "
                    "promotion envelope"
                )
            try:
                observed = float(configured[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"promotion_gates.{field} must be a finite number") from exc
            if not math.isfinite(observed) or observed != expected:
                raise ValueError(
                    f"promotion_gates.{field} must exactly match validated promotion envelope "
                    f"value {expected}, got {observed}"
                )

    code_values = {
        "current_w1_reference_to_native_min_speedup": PROMOTION_CURRENT_W1_SPEEDUP_MIN,
        "current_w6_reference_to_native_min_speedup": PROMOTION_CURRENT_W6_SPEEDUP_MIN,
        "measured_arm_wall_time_cv_max": BENCHMARK_WALL_TIME_CV_MAX,
        "output_native_to_reference_min_ratio": OUTPUT_BURDEN_NATIVE_TO_REFERENCE_MIN_RATIO,
        "output_native_to_reference_max_ratio": OUTPUT_BURDEN_NATIVE_TO_REFERENCE_MAX_RATIO,
        "native_peak_rss_to_same_cell_reference_max_ratio": NATIVE_PEAK_RSS_MAX_REFERENCE_RATIO,
    }
    for field, observed in code_values.items():
        expected = contract_values[field]
        if float(observed) != expected:
            raise ValueError(
                f"benchmark constant {field} drifted from validated promotion envelope: "
                f"expected {expected}, got {observed}"
            )
    measurement = thresholds.measurement
    if (
        int(measurement.w1_repetitions_min) != MINIMUM_W1_REPETITIONS
        or int(measurement.w6_repetitions_min) != MINIMUM_W6_REPETITIONS
    ):
        raise ValueError("benchmark repetition constants drifted from validated promotion envelope")
    configured_repetitions = dict(registry.get("benchmark_repetitions") or {})
    if configured_repetitions and (
        int(configured_repetitions.get("w1", -1)) != int(measurement.w1_repetitions_min)
        or int(configured_repetitions.get("w6", -1)) != int(measurement.w6_repetitions_min)
    ):
        raise ValueError(
            "benchmark_repetitions must exactly match the validated promotion envelope"
        )

    policy = {
        "schema_version": "eiel_four_path_preregistered_promotion_policy_v1",
        "source": {
            "contract_schema_version": envelope.schema_version,
            "contract_id": envelope.contract_id,
            "contract_version": envelope.contract_version,
            "contract_path": str(contract_path),
            "contract_sha256": _sha256(contract_path),
            "registry_mirror_checked": bool(configured),
        },
        **contract_values,
        "w1_repetitions_min": int(measurement.w1_repetitions_min),
        "w6_repetitions_min": int(measurement.w6_repetitions_min),
        "warmup_repetitions": int(measurement.warmup_repetitions),
        "native_peak_rss_exceptions": list(
            configured.get("native_peak_rss_exceptions") or []
        ),
    }

    known_pairs = set(CURRENT_REFERENCE_NATIVE_PAIR_SPECS)
    seen_exception_ids: set[str] = set()
    validated_exceptions: list[dict[str, Any]] = []
    for index, raw_exception in enumerate(policy["native_peak_rss_exceptions"]):
        if not isinstance(raw_exception, dict):
            raise ValueError(f"native_peak_rss_exceptions[{index}] must be an object")
        exception = dict(raw_exception)
        exception_id = str(exception.get("id", "") or "").strip()
        reference_arm = str(exception.get("reference_arm", "") or "")
        native_arm = str(exception.get("native_arm", "") or "")
        rep = str(exception.get("rep", "") or "").strip()
        cell_id = str(exception.get("cell_id", "") or "").strip()
        rationale = str(exception.get("rationale", "") or "").strip()
        approval_record = str(exception.get("approval_record", "") or "").strip()
        try:
            max_ratio = float(exception.get("max_native_to_reference_ratio"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"native_peak_rss_exceptions[{index}] requires a finite max ratio"
            ) from exc
        if not exception_id or exception_id in seen_exception_ids:
            raise ValueError("native peak RSS exception IDs must be non-empty and unique")
        if (reference_arm, native_arm) not in known_pairs:
            raise ValueError(f"native peak RSS exception {exception_id} names an unknown arm pair")
        if not rep or not cell_id:
            raise ValueError(
                f"native peak RSS exception {exception_id} must bind one exact rep and cell_id"
            )
        if not math.isfinite(max_ratio) or max_ratio <= NATIVE_PEAK_RSS_MAX_REFERENCE_RATIO:
            raise ValueError(
                f"native peak RSS exception {exception_id} must have a finite bound above "
                f"{NATIVE_PEAK_RSS_MAX_REFERENCE_RATIO}"
            )
        if not rationale or not approval_record or exception.get(
            "approved_before_authoritative_run"
        ) is not True:
            raise ValueError(
                f"native peak RSS exception {exception_id} requires rationale, approval_record, "
                "and approved_before_authoritative_run=true"
            )
        seen_exception_ids.add(exception_id)
        validated_exceptions.append(
            {
                "id": exception_id,
                "reference_arm": reference_arm,
                "native_arm": native_arm,
                "rep": rep,
                "cell_id": cell_id,
                "max_native_to_reference_ratio": max_ratio,
                "rationale": rationale,
                "approval_record": approval_record,
                "approved_before_authoritative_run": True,
            }
        )
    policy["native_peak_rss_exceptions"] = validated_exceptions
    accepted_exception_ids = set(thresholds.memory.accepted_exceptions)
    observed_exception_ids = {str(row["id"]) for row in validated_exceptions}
    if observed_exception_ids != accepted_exception_ids:
        raise ValueError(
            "promotion_gates.native_peak_rss_exceptions IDs must exactly match the validated "
            "promotion envelope accepted_exceptions"
        )
    return policy


def _campaign_promotion_eligible(campaign: dict[str, Any]) -> bool:
    explicit = campaign.get("promotion_eligible")
    if explicit is not None:
        return bool(explicit)
    return (
        int(campaign.get("incomplete_cells", 0) or 0) == 0
        and not bool(campaign.get("any_hidden_fallback"))
        and not bool(campaign.get("any_native_fail_closed"))
    )


def _cell_exclusion_audit(campaigns: list[dict[str, Any]]) -> dict[str, Any]:
    excluded: list[dict[str, Any]] = []
    promotion_ineligible_campaigns: list[dict[str, Any]] = []
    inventory_excludes_failed_cells = True
    for campaign in campaigns:
        arm = str(campaign.get("arm", ""))
        rep = str(campaign.get("rep", ""))
        included_keys = {
            (str(row.get("rep", rep)), str(row.get("cell_id", "")))
            for row in campaign.get("results", [])
            if bool(row.get("promotion_eligible", row.get("complete", True)))
        }
        excluded_rows = list(campaign.get("excluded_results") or [])
        excluded_rows.extend(
            row
            for row in campaign.get("results", [])
            if not bool(row.get("promotion_eligible", row.get("complete", True)))
        )
        for row in excluded_rows:
            key = (str(row.get("rep", rep)), str(row.get("cell_id", "")))
            if key in included_keys:
                inventory_excludes_failed_cells = False
            reasons = list(row.get("exclusion_reasons") or [])
            first_cause = row.get("first_failure_cause") or (
                reasons[0] if reasons else "failed_cell_missing_first_cause"
            )
            excluded.append(
                {
                    "arm": arm,
                    "rep": key[0],
                    "cell_id": key[1],
                    "first_failure_cause": str(first_cause),
                    "exclusion_reasons": reasons or [str(first_cause)],
                    "result_path": row.get("stdout_path"),
                }
            )
        if not _campaign_promotion_eligible(campaign):
            promotion_ineligible_campaigns.append(
                {
                    "arm": arm,
                    "rep": rep,
                    "first_failure_cause": campaign.get("first_failure_cause")
                    or (excluded[-1]["first_failure_cause"] if excluded else None)
                    or "promotion_ineligible_campaign_missing_first_cause",
                    "exclusion_reasons": list(campaign.get("exclusion_reasons") or []),
                }
            )
    first_cause = excluded[0] if excluded else (
        promotion_ineligible_campaigns[0] if promotion_ineligible_campaigns else None
    )
    return {
        "schema_version": "eiel_four_path_cell_exclusion_audit_v1",
        "excluded_cells": len(excluded),
        "promotion_ineligible_campaigns": len(promotion_ineligible_campaigns),
        "inventory_excludes_failed_cells": inventory_excludes_failed_cells,
        "first_failure": first_cause,
        "cells": excluded,
        "campaigns": promotion_ineligible_campaigns,
        "ok": not excluded
        and not promotion_ineligible_campaigns
        and inventory_excludes_failed_cells,
    }


def _performance_threshold_gate(
    speedups: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for lane, speedup_key, threshold_field in (
        (
            "current_w1_reference_to_native",
            "single_lane_native_b_over_e",
            "current_w1_reference_to_native_min_speedup",
        ),
        (
            "current_w6_reference_to_native",
            "native_contribution_c_over_d",
            "current_w6_reference_to_native_min_speedup",
        ),
    ):
        observed = speedups.get(speedup_key)
        minimum = float(policy[threshold_field])
        ok = (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and math.isfinite(float(observed))
            and float(observed) >= minimum
        )
        rows.append(
            {
                "lane": lane,
                "speedup_key": speedup_key,
                "observed_speedup": observed,
                "minimum_speedup": minimum,
                "ok": ok,
            }
        )
    return {
        "schema_version": "eiel_four_path_performance_threshold_gate_v1",
        "rows": rows,
        "ok": all(bool(row["ok"]) for row in rows),
    }


def _stability_gate(
    arms: dict[str, dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    maximum = float(policy["measured_arm_wall_time_cv_max"])
    rows: list[dict[str, Any]] = []
    for arm in PROMOTED_MEASURED_ARMS:
        cv = dict(arms.get(arm, {}).get("wall_s") or {}).get("cv")
        ok = (
            isinstance(cv, (int, float))
            and not isinstance(cv, bool)
            and math.isfinite(float(cv))
            and float(cv) <= maximum
        )
        rows.append({"arm": arm, "cv": cv, "maximum_cv": maximum, "ok": ok})
    return {
        "schema_version": "eiel_four_path_promoted_arm_stability_gate_v1",
        "promoted_measured_arms": list(PROMOTED_MEASURED_ARMS),
        "maximum_cv": maximum,
        "arms": rows,
        "ok": all(bool(row["ok"]) for row in rows),
    }


def _native_peak_rss_fairness(
    campaigns: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    by_arm = _cell_results_by_arm(campaigns)
    default_max = float(policy["native_peak_rss_to_same_cell_reference_max_ratio"])
    exceptions = {
        (
            str(row["reference_arm"]),
            str(row["native_arm"]),
            str(row["rep"]),
            str(row["cell_id"]),
        ): dict(row)
        for row in policy["native_peak_rss_exceptions"]
    }
    rows: list[dict[str, Any]] = []
    missing_pairs: list[dict[str, Any]] = []
    for reference_arm, native_arm in CURRENT_REFERENCE_NATIVE_PAIR_SPECS:
        reference_rows = by_arm.get(reference_arm, {})
        native_rows = by_arm.get(native_arm, {})
        all_keys = sorted(set(reference_rows) | set(native_rows))
        if not all_keys:
            missing_pairs.append(
                {
                    "reference_arm": reference_arm,
                    "native_arm": native_arm,
                    "reason": "arm_pair_missing",
                }
            )
            continue
        for rep, cell_id in all_keys:
            reference = reference_rows.get((rep, cell_id))
            native = native_rows.get((rep, cell_id))
            if reference is None or native is None:
                missing_pairs.append(
                    {
                        "reference_arm": reference_arm,
                        "native_arm": native_arm,
                        "rep": rep,
                        "cell_id": cell_id,
                        "reason": "same_cell_pair_missing",
                    }
                )
                continue
            reference_rss = dict(reference.get("time_v") or {}).get("max_rss_kb")
            native_rss = dict(native.get("time_v") or {}).get("max_rss_kb")
            ratio = None
            if (
                isinstance(reference_rss, (int, float))
                and not isinstance(reference_rss, bool)
                and isinstance(native_rss, (int, float))
                and not isinstance(native_rss, bool)
                and float(reference_rss) > 0.0
                and float(native_rss) >= 0.0
            ):
                ratio = float(native_rss) / float(reference_rss)
            exception = exceptions.get((reference_arm, native_arm, rep, cell_id))
            allowed_max = (
                float(exception["max_native_to_reference_ratio"])
                if exception is not None
                else default_max
            )
            ok = ratio is not None and math.isfinite(ratio) and ratio <= allowed_max
            rows.append(
                {
                    "reference_arm": reference_arm,
                    "native_arm": native_arm,
                    "rep": rep,
                    "cell_id": cell_id,
                    "reference_peak_rss_kb": reference_rss,
                    "native_peak_rss_kb": native_rss,
                    "native_to_reference_ratio": ratio,
                    "default_max_ratio": default_max,
                    "allowed_max_ratio": allowed_max,
                    "exception_id": exception.get("id") if exception else None,
                    "ok": ok,
                }
            )
    failures = [row for row in rows if not bool(row["ok"])]
    return {
        "schema_version": "eiel_four_path_same_cell_peak_rss_gate_v1",
        "default_max_ratio": default_max,
        "preregistered_exceptions": list(policy["native_peak_rss_exceptions"]),
        "paired_cells": len(rows),
        "missing_pairs": len(missing_pairs),
        "failed_cells": len(failures),
        "first_failure": failures[0] if failures else (missing_pairs[0] if missing_pairs else None),
        "rows": rows,
        "missing_pair_rows": missing_pairs,
        "ok": bool(rows) and not failures and not missing_pairs,
    }


def _native_worker_promotion_gate(
    campaigns: list[dict[str, Any]],
    extension_integrity: dict[str, Any],
) -> dict[str, Any]:
    expected_path = extension_integrity.get("path")
    expected_sha256 = extension_integrity.get("expected_sha256")
    campaign_rows: list[dict[str, Any]] = []
    all_audits: list[dict[str, Any]] = []
    native_arms_seen: set[str] = set()
    for campaign in campaigns:
        if not bool(campaign.get("native")) and not str(campaign.get("arm", "")).endswith(
            "_native"
        ):
            continue
        arm = str(campaign["arm"])
        native_arms_seen.add(arm)
        results = [*list(campaign.get("results") or []), *list(campaign.get("excluded_results") or [])]
        audits = [dict(row.get("native_worker_audit") or {}) for row in results]
        all_audits.extend(audits)
        identity_rows = [
            (
                audit.get("extension_path"),
                audit.get("extension_sha256"),
                audit.get("extension_abi_version"),
                audit.get("extension_version"),
                audit.get("comparator_source_commit"),
                audit.get("comparator_module_path"),
                audit.get("comparator_module_sha256"),
            )
            for audit in audits
        ]
        identity_homogeneous = bool(identity_rows) and len(set(identity_rows)) == 1
        exact_expected = all(
            (expected_path is None or audit.get("extension_path") == expected_path)
            and (expected_sha256 is None or audit.get("extension_sha256") == expected_sha256)
            for audit in audits
        )
        audits_complete = bool(audits) and len(audits) == int(
            campaign.get("attempted_cells", campaign.get("cells", len(results))) or 0
        )
        runtime_jit_false = all(audit.get("runtime_jit") is False for audit in audits)
        ownership_expected = all(
            audit.get("native_ownership_active") is True for audit in audits
        )
        mixed_ownership_false = all(
            audit.get("mixed_ownership") is False for audit in audits
        )
        stale_extension_false = all(
            audit.get("stale_extension") is False for audit in audits
        )
        audit_ok = all(bool(audit.get("ok")) for audit in audits)
        w6_homogeneous = identity_homogeneous if "_w6_" in f"_{arm}_" else True
        row_ok = all(
            (
                audits_complete,
                audit_ok,
                exact_expected,
                w6_homogeneous,
                runtime_jit_false,
                ownership_expected,
                mixed_ownership_false,
                stale_extension_false,
            )
        )
        campaign_rows.append(
            {
                "arm": arm,
                "rep": str(campaign.get("rep", "")),
                "workers": campaign.get("workers"),
                "worker_records": len(audits),
                "audits_complete": audits_complete,
                "audit_ok": audit_ok,
                "extension_matches_expected": exact_expected,
                "identity_homogeneous": identity_homogeneous,
                "w6_identity_homogeneous": w6_homogeneous,
                "runtime_jit_false": runtime_jit_false,
                "expected_native_ownership": ownership_expected,
                "mixed_ownership_false": mixed_ownership_false,
                "stale_extension_false": stale_extension_false,
                "identity": list(identity_rows[0]) if identity_homogeneous else None,
                "ok": row_ok,
            }
        )
    required_native_arms = {"current_w1_native", "current_w6_native"}
    required_arms_present = required_native_arms <= native_arms_seen
    return {
        "schema_version": "eiel_four_path_native_worker_promotion_gate_v1",
        "expected_extension_path": expected_path,
        "expected_extension_sha256": expected_sha256,
        "required_native_arms": sorted(required_native_arms),
        "required_native_arms_present": required_arms_present,
        "campaigns": campaign_rows,
        "runtime_jit_false": bool(all_audits)
        and all(audit.get("runtime_jit") is False for audit in all_audits),
        "expected_native_ownership": bool(all_audits)
        and all(audit.get("native_ownership_active") is True for audit in all_audits),
        "mixed_ownership_false": bool(all_audits)
        and all(audit.get("mixed_ownership") is False for audit in all_audits),
        "stale_extension_false": bool(all_audits)
        and all(audit.get("stale_extension") is False for audit in all_audits),
        "w6_identity_homogeneous": bool(campaign_rows)
        and all(bool(row["w6_identity_homogeneous"]) for row in campaign_rows),
        "ok": required_arms_present
        and bool(campaign_rows)
        and all(bool(row["ok"]) for row in campaign_rows),
    }


def _flat_output_cost_attribution_analysis(
    campaigns: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    if mode not in FLAT_OUTPUT_COST_ATTRIBUTION_MODES:
        raise ValueError(f"unsupported flat output cost attribution mode: {mode}")

    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    reference_by_arm: dict[str, str] = {}
    for campaign in campaigns:
        arm = str(campaign["arm"])
        rows_by_arm.setdefault(arm, []).extend(
            dict(result) for result in campaign.get("results", [])
        )
        if campaign.get("reference"):
            reference_by_arm[arm] = str(campaign["reference"])

    raw_samples: list[dict[str, Any]] = []
    analysis_errors: list[dict[str, Any]] = []
    arms: dict[str, Any] = {}
    current_expected = 0
    current_complete = 0
    for arm, rows in sorted(rows_by_arm.items()):
        historical = reference_by_arm.get(arm) == "historical" or arm.startswith("historical_")
        if historical:
            arms[arm] = {
                "applicability": "not_applicable",
                "reason": "historical_arm_does_not_implement_current_sidecar_contract",
                "expected_samples": 0,
                "complete_samples": 0,
                "run_total_ns": {"count": 0},
                "phases": {
                    phase: {"applicability": "not_applicable"}
                    for phase in FLAT_OUTPUT_COST_PHASES
                },
                "complete": True,
                "errors": [],
            }
            continue
        if mode == "off":
            arms[arm] = {
                "applicability": "disabled",
                "expected_samples": 0,
                "complete_samples": 0,
                "run_total_ns": {"count": 0},
                "phases": {
                    phase: {"applicability": "disabled"}
                    for phase in FLAT_OUTPUT_COST_PHASES
                },
                "complete": True,
                "errors": [],
            }
            continue

        current_expected += len(rows)
        complete_samples: list[dict[str, Any]] = []
        arm_errors: list[dict[str, Any]] = []
        for row in rows:
            attribution = row.get("flat_output_cost_attribution")
            if not isinstance(attribution, dict):
                attribution = {
                    "applicability": "applicable",
                    "sidecar_path": None,
                    "loaded": False,
                    "valid": False,
                    "complete": False,
                    "errors": ["missing_cell_attribution_record"],
                    "sample": None,
                }
            sample = attribution.get("sample")
            validation_errors = _validate_flat_output_cost_sample(sample)
            sample_complete = (
                not validation_errors
                and isinstance(sample, dict)
                and bool(sample.get("complete"))
                and bool(attribution.get("complete"))
            )
            combined_errors = list(attribution.get("errors") or [])
            combined_errors.extend(
                error for error in validation_errors if error not in combined_errors
            )
            raw_row = {
                "arm": arm,
                "rep": row.get("rep"),
                "cell_id": row.get("cell_id"),
                "sidecar_path": attribution.get("sidecar_path"),
                "loaded": bool(attribution.get("loaded")),
                "valid": bool(attribution.get("valid")) and not validation_errors,
                "complete": sample_complete,
                "errors": combined_errors,
                "sample": sample,
            }
            raw_samples.append(raw_row)
            if raw_row["complete"] and isinstance(raw_row["sample"], dict):
                complete_samples.append(raw_row["sample"])
                current_complete += 1
            else:
                error = {
                    "arm": arm,
                    "rep": row.get("rep"),
                    "cell_id": row.get("cell_id"),
                    "errors": raw_row["errors"] or ["incomplete_cost_sample"],
                }
                arm_errors.append(error)
                analysis_errors.append(error)

        phase_stats: dict[str, Any] = {}
        for phase in FLAT_OUTPUT_COST_PHASES:
            fractions = [
                float(sample["exclusive_ns"][phase]) / float(sample["run_total_ns"])
                for sample in complete_samples
                if int(sample["run_total_ns"]) > 0
            ]
            phase_stats[phase] = {
                "applicability": "applicable",
                "exclusive_ns": _stats(
                    sample["exclusive_ns"][phase] for sample in complete_samples
                ),
                "call_counts": _stats(
                    sample["call_counts"][phase] for sample in complete_samples
                ),
                "run_total_fraction": _stats(fractions),
            }
        arms[arm] = {
            "applicability": "applicable",
            "expected_samples": len(rows),
            "complete_samples": len(complete_samples),
            "run_total_ns": _stats(sample["run_total_ns"] for sample in complete_samples),
            "phases": phase_stats,
            "complete": len(complete_samples) == len(rows),
            "errors": arm_errors,
        }

    complete = mode == "off" or current_complete == current_expected
    return {
        "schema_version": "eiel_flat_output_cost_attribution_analysis_v1",
        "mode": mode,
        "sidecar_environment_variable": FLAT_OUTPUT_COST_SIDECAR_ENV,
        "sample_schema_version": FLAT_OUTPUT_COST_SAMPLE_SCHEMA,
        "phase_order": list(FLAT_OUTPUT_COST_PHASES),
        "sidecar_storage": "external_to_run_dir_and_excluded_from_output_inventory",
        "current_samples_expected": current_expected,
        "current_samples_complete": current_complete,
        "raw_samples": raw_samples,
        "arms": arms,
        "complete": complete,
        "errors": analysis_errors,
    }


def _analyze(
    campaigns: list[dict[str, Any]],
    evidence_root: Path,
    *,
    required_repetitions_by_arm: dict[str, int] | None = None,
    extension_integrity: dict[str, Any] | None = None,
    flat_output_cost_attribution: str = "off",
    promotion_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    promotion_policy = promotion_policy or _promotion_policy_from_registry({})
    cell_exclusion_audit = _cell_exclusion_audit(campaigns)
    eligible_campaigns = [row for row in campaigns if _campaign_promotion_eligible(row)]
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in eligible_campaigns:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    arms = {
        arm: {
            "wall_s": _stats(row["wall_s"] for row in rows),
            "cells_per_second": _stats(row["cells_per_second"] for row in rows),
            "steps_per_second": _stats(row["steps_per_second"] for row in rows),
            "output_bytes": _stats(row["output_bytes"] for row in rows),
            "checkpoint_bytes": _stats(row["checkpoint_bytes"] for row in rows),
            "peak_cell_rss_kb": _stats(row.get("peak_cell_rss_kb", 0) for row in rows),
            "sum_cell_peak_rss_kb": _stats(row.get("sum_cell_peak_rss_kb", 0) for row in rows),
            "child_user_time_s": _stats(row.get("child_user_time_s", 0.0) for row in rows),
            "child_system_time_s": _stats(row.get("child_system_time_s", 0.0) for row in rows),
            "child_cpu_time_s": _stats(row.get("child_cpu_time_s", 0.0) for row in rows),
            "worker_utilization_ratio": _stats(row.get("worker_utilization_ratio", 0.0) for row in rows),
            "resource_metrics_complete": all(bool(row.get("resource_metrics_complete")) for row in rows),
            "complete": all(int(row["incomplete_cells"]) == 0 for row in rows),
            "any_hidden_fallback": any(bool(row.get("any_hidden_fallback")) for row in rows),
            "any_native_fail_closed": any(bool(row.get("any_native_fail_closed")) for row in rows),
        }
        for arm, rows in sorted(by_arm.items())
    }

    def med(arm: str) -> float | None:
        stats = arms.get(arm, {}).get("wall_s", {})
        return float(stats["median"]) if "median" in stats else None

    speedups: dict[str, Any] = {}
    pairs = {
        "historical_to_current_max_a_over_d": ("historical_w1", "current_w6_native"),
        "current_serial_codebase_a_over_b": ("historical_w1", "current_w1_reference"),
        "w6_contribution_b_over_c": ("current_w1_reference", "current_w6_reference"),
        "native_contribution_c_over_d": ("current_w6_reference", "current_w6_native"),
        "single_lane_native_b_over_e": ("current_w1_reference", "current_w1_native"),
    }
    for name, (num, den) in pairs.items():
        n = med(num)
        d = med(den)
        speedups[name] = (n / d) if n is not None and d not in (None, 0.0) else None
    parts = [
        speedups.get("current_serial_codebase_a_over_b"),
        speedups.get("w6_contribution_b_over_c"),
        speedups.get("native_contribution_c_over_d"),
    ]
    speedups["composition_product"] = math.prod(float(v) for v in parts if v is not None) if all(v is not None for v in parts) else None
    shared_metric_parity = _shared_metric_parity(eligible_campaigns)
    matched_output_checkpoint_profile = _matched_output_checkpoint_profile(eligible_campaigns)
    output_burden_fairness = _output_burden_fairness(
        arms,
        eligible_campaigns,
        min_native_to_reference_ratio=float(
            promotion_policy["output_native_to_reference_min_ratio"]
        ),
        max_native_to_reference_ratio=float(
            promotion_policy["output_native_to_reference_max_ratio"]
        ),
    )
    output_inventory = _aggregate_output_inventory(eligible_campaigns)
    cost_attribution = _flat_output_cost_attribution_analysis(
        eligible_campaigns,
        mode=flat_output_cost_attribution,
    )
    repetition_gate = _repetition_gate(eligible_campaigns, required_repetitions_by_arm)
    performance_threshold_gate = _performance_threshold_gate(speedups, promotion_policy)
    stability_gate = _stability_gate(arms, promotion_policy)
    native_peak_rss_gate = _native_peak_rss_fairness(
        eligible_campaigns,
        promotion_policy,
    )
    extension_integrity = extension_integrity or {
        "required": False,
        "ok": True,
        "checks": [],
    }
    native_worker_promotion_gate = _native_worker_promotion_gate(
        campaigns,
        extension_integrity,
    )
    resource_metrics = {
        "schema_version": "eiel_four_path_resource_metrics_v1",
        "definitions": {
            "peak_cell_rss_kb": "maximum per-cell GNU time -v RSS; not aggregate concurrent RSS",
            "sum_cell_peak_rss_kb": "sum of per-cell peak RSS values; workload memory burden",
            "worker_utilization_ratio": "sum child CPU seconds divided by campaign wall seconds and configured workers",
        },
        "arms": {
            arm: {
                "peak_cell_rss_kb": stats["peak_cell_rss_kb"],
                "sum_cell_peak_rss_kb": stats["sum_cell_peak_rss_kb"],
                "child_user_time_s": stats["child_user_time_s"],
                "child_system_time_s": stats["child_system_time_s"],
                "child_cpu_time_s": stats["child_cpu_time_s"],
                "worker_utilization_ratio": stats["worker_utilization_ratio"],
                "complete": stats["resource_metrics_complete"],
            }
            for arm, stats in arms.items()
        },
        "same_cell_native_rss_gate": native_peak_rss_gate,
    }
    resource_metrics["ok"] = bool(resource_metrics["arms"]) and all(
        bool(row["complete"]) for row in resource_metrics["arms"].values()
    ) and bool(native_peak_rss_gate["ok"])
    stability_flags = [
        {
            "arm": row["arm"],
            "reason": (
                "cv_gt_0_05" if row["cv"] is not None else "cv_missing_for_promoted_arm"
            ),
            "cv": row["cv"],
        }
        for row in stability_gate["arms"]
        if not bool(row["ok"])
    ]
    fail_reasons: list[str] = []
    if not bool(cell_exclusion_audit["ok"]):
        fail_reasons.append("failed_or_incomplete_cells_excluded")
    if not all(bool(arm.get("complete")) for arm in arms.values()):
        fail_reasons.append("incomplete_arm")
    if not bool(shared_metric_parity["ok"]):
        fail_reasons.append("current_reference_native_shared_metric_mismatch")
    if not bool(matched_output_checkpoint_profile["ok"]):
        fail_reasons.append("current_reference_native_output_checkpoint_profile_mismatch")
    if not bool(output_burden_fairness["ok"]):
        fail_reasons.append("native_reference_output_burden_ratio_out_of_bounds")
    if not bool(repetition_gate["ok"]):
        fail_reasons.append("insufficient_w1_or_w6_repetitions")
    if not bool(performance_threshold_gate["ok"]):
        fail_reasons.append("preregistered_native_speedup_threshold_not_met")
    if not bool(stability_gate["ok"]):
        fail_reasons.append("measured_repetition_stability_cv_exceeded")
    if not bool(resource_metrics["ok"]):
        fail_reasons.append("resource_metrics_incomplete")
    if not bool(native_peak_rss_gate["ok"]):
        fail_reasons.append("native_peak_rss_same_cell_ratio_exceeded")
    if not bool(native_worker_promotion_gate["w6_identity_homogeneous"]):
        fail_reasons.append("w6_native_worker_extension_identity_not_homogeneous")
    if not bool(native_worker_promotion_gate["runtime_jit_false"]):
        fail_reasons.append("runtime_jit_detected_or_not_proven_false")
    if not bool(native_worker_promotion_gate["expected_native_ownership"]):
        fail_reasons.append("expected_native_ownership_not_proven")
    if not bool(native_worker_promotion_gate["mixed_ownership_false"]):
        fail_reasons.append("mixed_native_python_ownership_detected_or_not_proven_false")
    if not bool(native_worker_promotion_gate["stale_extension_false"]):
        fail_reasons.append("stale_native_extension_detected_or_not_proven_false")
    if not bool(native_worker_promotion_gate["ok"]):
        fail_reasons.append("native_worker_promotion_contract_failed")
    if not bool(extension_integrity.get("ok")):
        fail_reasons.append("native_extension_sha256_changed_during_campaign")
    if flat_output_cost_attribution == "required" and not bool(cost_attribution["complete"]):
        fail_reasons.append("current_flat_output_cost_attribution_incomplete")
    analysis = {
        "schema_version": "eiel_four_path_benchmark_analysis_v1",
        "promotion_policy": promotion_policy,
        "eligible_campaigns": len(eligible_campaigns),
        "attempted_campaigns": len(campaigns),
        "cell_exclusion_audit": cell_exclusion_audit,
        "arms": arms,
        "speedups": speedups,
        "performance_threshold_gate": performance_threshold_gate,
        "performance_measurement": {
            "source": "external_time.perf_counter_around_repeated_arm_subprocess_campaigns",
            "metrics_summary_avg_gen_time_s_role": "finite_nonnegative_output_validity_only",
        },
        "shared_metric_parity": shared_metric_parity,
        "matched_output_checkpoint_profile": matched_output_checkpoint_profile,
        "output_burden_fairness": output_burden_fairness,
        "output_inventory": output_inventory,
        "flat_output_cost_attribution": cost_attribution,
        "repetition_gate": repetition_gate,
        "resource_metrics": resource_metrics,
        "native_peak_rss_gate": native_peak_rss_gate,
        "native_worker_promotion_gate": native_worker_promotion_gate,
        "extension_integrity": extension_integrity,
        "fail_closed": bool(fail_reasons),
        "fail_closed_reasons": fail_reasons,
        "stability_cv_max": float(promotion_policy["measured_arm_wall_time_cv_max"]),
        "stability_gate": stability_gate,
        "stability_flags": stability_flags,
    }
    _write_json(evidence_root / "analysis" / "benchmark_analysis.json", analysis)
    _write_json(evidence_root / "analysis" / "flat_output_inventory.json", output_inventory)
    _write_json(
        evidence_root / "analysis" / "flat_output_cost_attribution.json",
        cost_attribution,
    )
    _write_json(
        evidence_root / "analysis" / "flat_output_fairness_report.json",
        {
            "schema_version": "eiel_flat_output_fairness_report_v1",
            "matched_output_checkpoint_profile": matched_output_checkpoint_profile,
            "output_burden_fairness": output_burden_fairness,
            "ok": bool(matched_output_checkpoint_profile["ok"]) and bool(output_burden_fairness["ok"]),
        },
    )
    _write_json(
        evidence_root / "analysis" / "promotion_evidence.json",
        {
            "schema_version": "eiel_four_path_promotion_evidence_v1",
            "promotion_safe": not bool(fail_reasons),
            "fail_closed_reasons": fail_reasons,
            "promotion_policy": promotion_policy,
            "cell_exclusion_audit": cell_exclusion_audit,
            "speedups": speedups,
            "performance_threshold_gate": performance_threshold_gate,
            "repetition_gate": repetition_gate,
            "stability_cv_max": float(promotion_policy["measured_arm_wall_time_cv_max"]),
            "stability_gate": stability_gate,
            "stability_flags": stability_flags,
            "stability_ok": bool(stability_gate["ok"]),
            "resource_metrics": resource_metrics,
            "native_peak_rss_gate": native_peak_rss_gate,
            "native_worker_promotion_gate": native_worker_promotion_gate,
            "extension_integrity": extension_integrity,
            "shared_metric_parity_ok": bool(shared_metric_parity["ok"]),
            "full_public_output_parity_ok": bool(matched_output_checkpoint_profile["ok"]),
            "output_burden_fairness_ok": bool(output_burden_fairness["ok"]),
            "flat_output_cost_attribution": cost_attribution,
            "flat_output_cost_attribution_ok": bool(cost_attribution["complete"]),
        },
    )
    with (evidence_root / "analysis" / "arm_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "arm",
                "repetitions",
                "median_wall_s",
                "mean_wall_s",
                "cv",
                "median_cells_per_second",
                "median_steps_per_second",
                "median_peak_cell_rss_kb",
                "median_worker_utilization_ratio",
                "resource_metrics_complete",
                "complete",
                "any_hidden_fallback",
            ],
        )
        writer.writeheader()
        for arm, stats in arms.items():
            writer.writerow(
                {
                    "arm": arm,
                    "repetitions": stats["wall_s"].get("count"),
                    "median_wall_s": stats["wall_s"].get("median"),
                    "mean_wall_s": stats["wall_s"].get("mean"),
                    "cv": stats["wall_s"].get("cv"),
                    "median_cells_per_second": stats["cells_per_second"].get("median"),
                    "median_steps_per_second": stats["steps_per_second"].get("median"),
                    "median_peak_cell_rss_kb": stats["peak_cell_rss_kb"].get("median"),
                    "median_worker_utilization_ratio": stats["worker_utilization_ratio"].get("median"),
                    "resource_metrics_complete": stats["resource_metrics_complete"],
                    "complete": stats["complete"],
                    "any_hidden_fallback": stats["any_hidden_fallback"],
                }
            )
    return analysis


def _smoke_native_extension(
    py: Path,
    extension: Path,
    evidence_root: Path,
    *,
    artifact_source: str,
) -> dict[str, Any]:
    exact_path = extension.expanduser().resolve(strict=True)
    if not exact_path.is_file():
        raise RuntimeError(f"native extension is not a file: {exact_path}")
    pre_smoke_sha256 = _sha256(exact_path)
    smoke_env = _base_env(PROJECT_ROOT)
    smoke_env["EIEL_NATIVE_RUNTIME_EXTENSION"] = str(exact_path)
    smoke = _run_capture(
        [
            str(py),
            "-c",
            (
                "import json,sys; from pathlib import Path; "
                "from native_runtime.bindings import native_extension_status, load_prebuilt_extension; "
                "p=Path(sys.argv[1]).resolve(strict=True); s=native_extension_status(p); "
                "assert Path(s.extension_path).resolve(strict=True) == p; "
                "print(json.dumps(s.to_dict(), sort_keys=True)); "
                "load_prebuilt_extension(p); print('loaded', p)"
            ),
            str(exact_path),
        ],
        cwd=PROJECT_ROOT,
        env=smoke_env,
        timeout=120,
    )
    post_smoke_sha256 = _sha256(exact_path) if exact_path.is_file() else None
    smoke.update(
        {
            "artifact_source": artifact_source,
            "extension_path": str(exact_path),
            "pre_smoke_sha256": pre_smoke_sha256,
            "post_smoke_sha256": post_smoke_sha256,
            "artifact_unchanged": post_smoke_sha256 == pre_smoke_sha256,
        }
    )
    _write_json(evidence_root / "environment" / "native_load_smoke.json", smoke)
    if smoke["returncode"] != 0:
        raise RuntimeError(f"native extension load smoke failed: {smoke['stderr'][-1000:]}")
    if not smoke["artifact_unchanged"]:
        raise RuntimeError("native extension changed during load smoke")
    return smoke


def _build_native_extension(py: Path, evidence_root: Path, *, skip_build: bool = False) -> Path:
    build_dir = evidence_root / "environment" / "native_build"
    output = evidence_root / "environment" / "eiel_native_runtime_ext_four_path.so"
    if skip_build:
        if not output.is_file():
            raise RuntimeError(
                "--skip-native-build requires an existing reusable evidence-root extension: "
                f"{output}"
            )
        _smoke_native_extension(
            py,
            output,
            evidence_root,
            artifact_source="legacy_evidence_root_reuse",
        )
        return output.resolve(strict=True)
    cmd = [
        str(py),
        str(PROJECT_ROOT / "tools" / "scripts" / "build_eiel_native_runtime_extension.py"),
        "--build-dir",
        str(build_dir),
        "--output",
        str(output),
    ]
    result = _run_capture(cmd, cwd=PROJECT_ROOT, timeout=600)
    result["output_sha256"] = _sha256(output) if output.exists() else ""
    _write_json(evidence_root / "environment" / "native_build_result.json", result)
    if result["returncode"] != 0 or not output.exists():
        raise RuntimeError(f"native extension build failed: {result['stderr'][-1000:]}")
    _smoke_native_extension(
        py,
        output,
        evidence_root,
        artifact_source="benchmark_build",
    )
    return output.resolve(strict=True)


def _prepare_native_extension(
    py: Path,
    evidence_root: Path,
    *,
    external_path: Path | None = None,
    expected_external_sha256: str | None = None,
    skip_build: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if external_path is not None and skip_build:
        raise ValueError("--native-extension and --skip-native-build are mutually exclusive")
    if expected_external_sha256 is not None and external_path is None:
        raise ValueError("--expected-native-extension-sha256 requires --native-extension")

    required_sha256 = (
        _normalize_sha256(expected_external_sha256)
        if expected_external_sha256 is not None
        else None
    )
    if external_path is not None:
        try:
            native_extension = external_path.expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"external native extension does not exist: {external_path}") from exc
        if not native_extension.is_file():
            raise ValueError(f"external native extension is not a file: {native_extension}")
        extension_sha256 = _sha256(native_extension)
        if required_sha256 is not None and extension_sha256 != required_sha256:
            raise ValueError(
                "external native extension SHA-256 mismatch: "
                f"expected {required_sha256}, observed {extension_sha256}"
            )
        artifact_source = "external_in_place"
        _smoke_native_extension(
            py,
            native_extension,
            evidence_root,
            artifact_source=artifact_source,
        )
    else:
        native_extension = _build_native_extension(py, evidence_root, skip_build=skip_build)
        extension_sha256 = _sha256(native_extension)
        artifact_source = "legacy_evidence_root_reuse" if skip_build else "benchmark_build"

    initial_check = _extension_integrity_check(
        native_extension,
        extension_sha256,
        "post_external_load_smoke" if external_path is not None else "post_build_or_reuse",
    )
    extension_integrity = {
        "schema_version": "eiel_immutable_native_extension_manifest_v1",
        "required": True,
        "artifact_source": artifact_source,
        "build_invoked": artifact_source == "benchmark_build",
        "copy_or_rewrite_permitted": artifact_source != "external_in_place",
        "path": str(native_extension),
        "required_sha256": required_sha256,
        "expected_sha256": extension_sha256,
        "initial_sha256": extension_sha256,
        "checks": [initial_check],
        "ok": bool(initial_check["ok"]),
    }
    if not extension_integrity["ok"]:
        raise RuntimeError("native extension changed before immutable manifest creation")
    environment_dir = evidence_root / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    (environment_dir / "eiel_native_runtime_ext_four_path.so.sha256").write_text(
        f"{extension_sha256}  {native_extension}\n",
        encoding="utf-8",
    )
    _write_json(
        environment_dir / "immutable_native_extension_manifest.json",
        extension_integrity,
    )
    return native_extension, extension_integrity


def _write_flat_results(campaigns: list[dict[str, Any]], evidence_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for campaign in campaigns:
        for row in campaign["results"]:
            rows.append(
                {
                    "rep": row["rep"],
                    "arm": row["arm"],
                    "cell_id": row["cell_id"],
                    "returncode": row["returncode"],
                    "complete": row["complete"],
                    "wall_s": row["wall_s"],
                    "max_rss_kb": row.get("time_v", {}).get("max_rss_kb"),
                    "output_bytes": row["output_bytes"],
                    "checkpoint_bytes": row["checkpoint_bytes"],
                    "final_alive": row.get("metrics", {}).get("final_alive"),
                    "total_successful_births": row.get("metrics", {}).get("total_successful_births"),
                    "total_failed_births": row.get("metrics", {}).get("total_failed_births"),
                    "total_deaths": row.get("metrics", {}).get("total_deaths"),
                }
            )
    path = evidence_root / "campaign_results" / "cell_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["rep", "arm", "cell_id"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the EIEL historical/current four-path performance benchmark.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--evidence-root", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--limit-cells", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--w1-repetitions", type=int, default=None)
    parser.add_argument("--w6-repetitions", type=int, default=None)
    parser.add_argument("--include-arm-e", action="store_true")
    parser.add_argument("--warmup-cells", type=int, default=1)
    parser.add_argument(
        "--skip-native-build",
        action="store_true",
        help="Reuse the existing evidence-root native extension; fail if it is absent.",
    )
    parser.add_argument(
        "--native-extension",
        type=Path,
        default=None,
        help="Use and smoke this exact existing native extension in place; never build or copy it.",
    )
    parser.add_argument(
        "--expected-native-extension-sha256",
        default=None,
        help="Required SHA-256 for --native-extension.",
    )
    parser.add_argument("--only-arms", default="")
    parser.add_argument(
        "--flat-output-cost-attribution",
        choices=FLAT_OUTPUT_COST_ATTRIBUTION_MODES,
        default="off",
    )
    args = parser.parse_args(argv)
    if args.native_extension is not None and args.skip_native_build:
        parser.error("--native-extension and --skip-native-build are mutually exclusive")
    if args.expected_native_extension_sha256 is not None and args.native_extension is None:
        parser.error("--expected-native-extension-sha256 requires --native-extension")
    if args.expected_native_extension_sha256 is not None:
        try:
            args.expected_native_extension_sha256 = _normalize_sha256(
                args.expected_native_extension_sha256
            )
        except ValueError as exc:
            parser.error(str(exc))

    registry = _load_registry(args.registry)
    try:
        promotion_policy = _promotion_policy_from_registry(registry)
    except ValueError as exc:
        raise SystemExit(f"invalid preregistered promotion policy: {exc}") from exc
    evidence_root = Path(
        args.evidence_root or registry.get("evidence_root") or DEFAULT_EVIDENCE_ROOT
    ).resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    py = _python_interpreter_path(args.python)
    arms = _arm_specs(registry)
    cells = _cells_from_registry(registry, limit_cells=args.limit_cells)
    if not cells:
        raise SystemExit("no benchmark cells generated")

    selected_arm_names = [name.strip() for name in args.only_arms.split(",") if name.strip()]
    if not selected_arm_names:
        default_arm_order = (
            "historical_w1",
            "current_w1_reference",
            "current_w6_reference",
            "current_w1_native",
            "current_w6_native",
        )
        selected_arm_names = [name for name in default_arm_order if name in arms]
        selected_arm_names.extend(name for name in arms if name not in selected_arm_names)
    unknown_arm_names = sorted(set(selected_arm_names) - set(arms))
    if unknown_arm_names:
        raise SystemExit(f"unknown benchmark arms: {unknown_arm_names}")
    if (
        args.include_arm_e
        and "current_w1_native" in arms
        and "current_w1_native" not in selected_arm_names
    ):
        selected_arm_names.append("current_w1_native")
    try:
        repetitions_by_arm = _repetition_plan(
            registry,
            arms,
            selected_arm_names,
            global_override=args.repetitions,
            w1_override=args.w1_repetitions,
            w6_override=args.w6_repetitions,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid authoritative repetition plan: {exc}") from exc

    _, timed_env_manifest = _timed_subprocess_env(PROJECT_ROOT)
    measured_execution_count = len(cells) * sum(repetitions_by_arm.values())
    warmup_execution_count = min(len(cells), max(0, int(args.warmup_cells))) * len(
        selected_arm_names
    )
    manifest = {
        "schema_version": "eiel_four_path_canonical_manifest_v1",
        "registry": str(args.registry.resolve()),
        "cells": [cell.__dict__ for cell in cells],
        "arms": {name: arms[name].__dict__ | {"project_root": str(arms[name].project_root)} for name in selected_arm_names},
        "repetitions_by_arm": repetitions_by_arm,
        "minimum_repetitions": {
            "w1": MINIMUM_W1_REPETITIONS,
            "w6": MINIMUM_W6_REPETITIONS,
        },
        "warmup_cells": int(args.warmup_cells),
        "execution_count": {
            "measured_cells": measured_execution_count,
            "warmup_cells": warmup_execution_count,
            "total_cells": measured_execution_count + warmup_execution_count,
        },
        "preregistered_promotion_policy": promotion_policy,
        "flat_output_cost_attribution": {
            "mode": args.flat_output_cost_attribution,
            "sidecar_environment_variable": FLAT_OUTPUT_COST_SIDECAR_ENV,
            "sample_schema_version": FLAT_OUTPUT_COST_SAMPLE_SCHEMA,
            "exclusive_phases": list(FLAT_OUTPUT_COST_PHASES),
            "sidecar_storage": "external_to_run_dir_and_excluded_from_output_inventory",
        },
        "performance_measurement": {
            "source": "external_time.perf_counter_around_subprocess",
            "aggregation": "median_of_repeated_arm_campaign_wall_s",
            "metrics_summary_avg_gen_time_s_role": "finite_nonnegative_output_validity_only",
        },
        "timed_subprocess_environment": timed_env_manifest,
        "current_matched_public_output_policy": {
            "required_route_neutral_files": [
                "metrics/typed_lifecycle_events.csv",
                "metrics/typed_lifecycle_events.jsonl",
            ],
            "required_public_files": list(REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES),
            "byte_exact_public_files": list(EXACT_CURRENT_PUBLIC_OUTPUT_FILES),
            "current_only_knobs": list(CURRENT_MATCHED_PUBLIC_OUTPUT_KNOBS),
            "diagnostic_outputs_disabled": [
                "developmental trace JSONL",
                "Phase E E1 metrics",
                "A3 recovery taxonomy metrics",
                "lineage QD archive",
                "native runtime summary",
                "runnercore summary",
            ],
            "comparison_policy": "all emitted relative files and all emitted bytes remain counted",
        },
    }
    _write_json(evidence_root / "canonical_manifest" / "canonical_campaign_manifest.json", manifest)
    historical_arm = arms.get("historical_w1")
    _write_json(
        evidence_root / "environment" / "environment.json",
        _environment_record(
            py,
            evidence_root,
            historical_arm.project_root if historical_arm is not None else None,
        ),
    )

    native_extension = None
    extension_integrity: dict[str, Any] = {"required": False, "ok": True, "checks": []}
    native_arm_selected = any(arms[name].native for name in selected_arm_names)
    if native_arm_selected or args.native_extension is not None:
        try:
            native_extension, extension_integrity = _prepare_native_extension(
                py,
                evidence_root,
                external_path=args.native_extension,
                expected_external_sha256=args.expected_native_extension_sha256,
                skip_build=bool(args.skip_native_build),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"native extension preparation failed: {exc}") from exc
        if not native_arm_selected:
            print(
                "[native-extension] validated external artifact but no native arm is selected",
                flush=True,
            )

    def verify_extension(phase: str) -> None:
        if native_extension is None:
            return
        check = _extension_integrity_check(native_extension, str(extension_integrity["expected_sha256"]), phase)
        extension_integrity["checks"].append(check)
        extension_integrity["ok"] = all(bool(row["ok"]) for row in extension_integrity["checks"])
        extension_integrity["final_sha256"] = check["observed_sha256"]
        _write_json(
            evidence_root / "environment" / "immutable_native_extension_manifest.json",
            extension_integrity,
        )
        if not check["ok"]:
            raise RuntimeError(f"immutable native extension changed at {phase}")

    campaigns: list[dict[str, Any]] = []
    warmup_cells = cells[: max(0, int(args.warmup_cells))]
    if warmup_cells:
        for name in selected_arm_names:
            print(f"[warmup] {name}", flush=True)
            summary = _run_arm(
                py,
                arms[name],
                warmup_cells,
                "warmup",
                evidence_root,
                native_extension,
                flat_output_cost_attribution=args.flat_output_cost_attribution,
            )
            campaigns.append(summary | {"warmup": True})
            try:
                verify_extension(f"post_warmup_{name}")
            except RuntimeError:
                _write_json(evidence_root / "campaign_results" / "campaigns_partial.json", campaigns)
                raise
            if not bool(summary.get("promotion_eligible")):
                failure = {
                    "schema_version": "eiel_four_path_benchmark_failure_v1",
                    "phase": "warmup",
                    "arm": name,
                    "rep": "warmup",
                    "first_failure_cause": summary.get("first_failure_cause")
                    or "warmup_campaign_not_promotion_eligible",
                    "campaign": summary,
                }
                _write_json(evidence_root / "analysis" / "benchmark_failure.json", failure)
                _write_json(evidence_root / "campaign_results" / "campaigns_partial.json", campaigns)
                return 2

    order_template = registry.get("repetition_order") or [selected_arm_names]
    for rep_index in range(max(repetitions_by_arm.values())):
        rep = f"rep{rep_index + 1}"
        active_arm_names = [name for name in selected_arm_names if rep_index < repetitions_by_arm[name]]
        order = [name for name in order_template[rep_index % len(order_template)] if name in active_arm_names]
        for name in active_arm_names:
            if name not in order:
                order.append(name)
        for name in order:
            print(f"[{rep}] starting {name} workers={arms[name].workers} cells={len(cells)}", flush=True)
            summary = _run_arm(
                py,
                arms[name],
                cells,
                rep,
                evidence_root,
                native_extension,
                flat_output_cost_attribution=args.flat_output_cost_attribution,
            )
            campaigns.append(summary)
            try:
                verify_extension(f"post_campaign_{rep}_{name}")
            except RuntimeError:
                _write_json(evidence_root / "campaign_results" / "campaigns_partial.json", campaigns)
                raise
            if not bool(summary.get("promotion_eligible")):
                _write_json(evidence_root / "campaign_results" / "campaigns_partial.json", campaigns)
                measured_partial = [row for row in campaigns if not row.get("warmup")]
                _write_flat_results(measured_partial, evidence_root)
                analysis = _analyze(
                    measured_partial,
                    evidence_root,
                    required_repetitions_by_arm=repetitions_by_arm,
                    extension_integrity=extension_integrity,
                    flat_output_cost_attribution=args.flat_output_cost_attribution,
                    promotion_policy=promotion_policy,
                )
                _write_json(
                    evidence_root / "analysis" / "benchmark_failure.json",
                    {
                        "schema_version": "eiel_four_path_benchmark_failure_v1",
                        "phase": "measured_campaign",
                        "arm": name,
                        "rep": rep,
                        "first_failure_cause": summary.get("first_failure_cause")
                        or "campaign_not_promotion_eligible",
                        "campaign_exclusion_reasons": list(
                            summary.get("exclusion_reasons") or []
                        ),
                        "analysis_fail_closed_reasons": analysis["fail_closed_reasons"],
                    },
                )
                return 2

    measured_campaigns = [row for row in campaigns if not row.get("warmup")]
    try:
        verify_extension("post_all_campaigns")
    except RuntimeError:
        _write_json(evidence_root / "campaign_results" / "campaigns_partial.json", campaigns)
        raise
    _write_json(evidence_root / "campaign_results" / "campaigns.json", measured_campaigns)
    _write_flat_results(measured_campaigns, evidence_root)
    analysis = _analyze(
        measured_campaigns,
        evidence_root,
        required_repetitions_by_arm=repetitions_by_arm,
        extension_integrity=extension_integrity,
        flat_output_cost_attribution=args.flat_output_cost_attribution,
        promotion_policy=promotion_policy,
    )
    print(
        json.dumps(
            {
                "evidence_root": str(evidence_root),
                "analysis": analysis["speedups"],
                "fail_closed": analysis["fail_closed"],
                "fail_closed_reasons": analysis["fail_closed_reasons"],
            },
            sort_keys=True,
        )
    )
    return 2 if bool(analysis["fail_closed"]) else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--native-worker-probe":
        raise SystemExit(_native_worker_probe_main(sys.argv[2:]))
    raise SystemExit(main())
