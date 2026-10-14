#!/usr/bin/env python3
from __future__ import annotations

"""Run the one exact, source-bound Phase E GPU qualification suite.

Importing this module and asking for ``--help`` are deliberately CUDA-blind.
Torch-dependent contracts and the CUDA device are inspected only by a real
``run`` command.  A receipt manifest is written only after all ten tests pass.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, Mapping, Sequence


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ENGINE: Final[Path] = ROOT / "engine"
SCHEMA_VERSION: Final[str] = "eiel_phase_e_gpu_suite_execution_manifest_v1"
SCOPE: Final[str] = "phase_e_exact_ten_node_same_source_gpu_suite_v1"
JUNIT_FILENAME: Final[str] = "phase_e_gpu_suite.junit.xml"
LOG_FILENAME: Final[str] = "phase_e_gpu_suite.log"
BASETEMP_DIRNAME: Final[str] = "pytest-tmp"
MANIFEST_FILENAME: Final[str] = "phase_e_gpu_suite_execution_manifest.json"
PREBUILT_ENV: Final[str] = "EIEL_CUDA_ARCHITECTURE_B_PREBUILT"
DEFAULT_BUILD_MAX_JOBS: Final[int] = 1
MAX_BUILD_MAX_JOBS: Final[int] = 256
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}")
_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "scope",
        "source",
        "command",
        "required_gpu_test_node_ids",
        "runner_environment",
        "cuda_environment",
        "generation_artifact",
        "modular_tick_artifact",
        "pytest",
        "manifest_sha256",
    }
)


class GpuSuiteError(RuntimeError):
    """The controlled GPU suite cannot produce qualification evidence."""


def _load_contract() -> SimpleNamespace:
    """Load Torch-backed source declarations without asking Torch for a GPU."""

    for path in (ROOT, ENGINE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from perf.cuda_architecture_b_generation import (
        CUDA_GENERATION_PREBUILT_CAPABILITY_CONTRACT,
        HASH_SOURCES as GENERATION_HASH_SOURCES,
    )
    from perf.cuda_architecture_b_modular_tick_native import (
        HASH_SOURCES as MODULAR_TICK_HASH_SOURCES,
    )
    from perf.cuda_architecture_b_phase_e_gpu_gate import (
        PHASE_E_NATIVE_SOURCE_FILES,
        PHASE_E_TEST_SOURCE_FILES,
        REQUIRED_GPU_TEST_NODE_IDS,
    )
    from perf.cuda_architecture_b_prebuilt import validate_cuda_prebuilt_manifest
    from tools.scripts.build_eiel_cuda_architecture_b_extension import (
        EXPORTED_FUNCTIONS as GENERATION_EXPORTED_FUNCTIONS,
    )

    return SimpleNamespace(
        capability_contract=dict(CUDA_GENERATION_PREBUILT_CAPABILITY_CONTRACT),
        generation_hash_sources=tuple(
            path.resolve() for path in GENERATION_HASH_SOURCES
        ),
        modular_tick_hash_sources=tuple(
            path.resolve() for path in MODULAR_TICK_HASH_SOURCES
        ),
        native_source_files=tuple(PHASE_E_NATIVE_SOURCE_FILES),
        test_source_files=tuple(PHASE_E_TEST_SOURCE_FILES),
        required_nodes=tuple(REQUIRED_GPU_TEST_NODE_IDS),
        required_exports=tuple(GENERATION_EXPORTED_FUNCTIONS),
        validate_prebuilt=validate_cuda_prebuilt_manifest,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _raw_absolute_path(value: str | Path, *, name: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise GpuSuiteError(f"{name} must be an absolute path")
    if ".." in raw.parts:
        raise GpuSuiteError(f"{name} must not contain parent traversal")
    return Path(os.path.abspath(os.fspath(raw)))


def _reject_symlink_components(path: Path, *, name: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if os.path.lexists(current) and stat.S_ISLNK(os.lstat(current).st_mode):
            raise GpuSuiteError(f"{name} contains a symlink component: {current}")


def _existing_regular_file(value: str | Path, *, name: str) -> Path:
    path = _raw_absolute_path(value, name=name)
    _reject_symlink_components(path, name=name)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as exc:
        raise GpuSuiteError(f"{name} is missing: {path}") from exc
    if not stat.S_ISREG(mode):
        raise GpuSuiteError(f"{name} must be a regular file: {path}")
    return path


def _new_run_directory(value: str | Path) -> Path:
    path = _raw_absolute_path(value, name="run directory")
    if path == Path(path.anchor):
        raise GpuSuiteError("run directory cannot be a filesystem root")
    _reject_symlink_components(path, name="run directory")
    parent = path.parent
    if not parent.is_dir():
        raise GpuSuiteError(f"run directory parent must already exist: {parent}")
    if os.path.lexists(path):
        raise GpuSuiteError(f"run directory must be new: {path}")
    return path


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GpuSuiteError(f"{name} is unreadable JSON") from exc
    if not isinstance(value, dict):
        raise GpuSuiteError(f"{name} must contain one JSON object")
    return value


def _source_rows(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in paths:
        checked = _existing_regular_file(source, name="source inventory file")
        try:
            relative = checked.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise GpuSuiteError(
                f"source inventory file escapes repository root: {checked}"
            ) from exc
        rows.append(
            {
                "path": str(checked),
                "repo_relative_path": relative,
                "sha256": _sha256_file(checked),
            }
        )
    return rows


def _git(command: Sequence[str], *, paths: Sequence[str] = ()) -> str:
    result = subprocess.run(
        ["git", *command, *(paths if paths else ())],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise GpuSuiteError(f"git source identity command failed: {detail}")
    return result.stdout.strip()


def _source_identity(contract: SimpleNamespace) -> dict[str, Any]:
    runner_paths = (
        "tools/scripts/eiel_phase_e_gpu_suite.py",
        "tests/tools/test_eiel_phase_e_gpu_suite.py",
    )
    relevant = tuple(
        dict.fromkeys(
            (*contract.native_source_files, *contract.test_source_files, *runner_paths)
        )
    )
    for relative in relevant:
        path = ROOT / relative
        _existing_regular_file(path, name="relevant source")
    tracked = _git(("ls-files", "--error-unmatch", "--", *relevant))
    if set(tracked.splitlines()) != set(relevant):
        raise GpuSuiteError("relevant Phase E source inventory is not fully tracked")
    dirty = _git(("status", "--porcelain=v1", "--untracked-files=all"))
    if dirty:
        raise GpuSuiteError(
            "the Phase E worktree must be fully committed and clean"
        )
    branch = _git(("rev-parse", "--abbrev-ref", "HEAD"))
    head = _git(("rev-parse", "HEAD"))
    if branch == "HEAD" or not branch:
        raise GpuSuiteError("Phase E GPU evidence requires a named source branch")
    if _COMMIT_RE.fullmatch(head) is None:
        raise GpuSuiteError("Phase E GPU evidence requires a full source commit")
    return {
        "repository_root": str(ROOT),
        "branch": branch,
        "commit": head,
        "test_commit": head,
        "relevant_paths": list(relevant),
        "relevant_source_clean": True,
    }


def _validate_generation_artifact(
    artifact_value: str | Path, contract: SimpleNamespace
) -> dict[str, Any]:
    artifact = _existing_regular_file(
        artifact_value, name="generation artifact"
    )
    sidecar = _existing_regular_file(
        artifact.with_suffix(artifact.suffix + ".manifest.json"),
        name="generation artifact sidecar",
    )
    try:
        identity = contract.validate_prebuilt(
            artifact, required_exports=contract.required_exports
        )
    except RuntimeError as exc:
        raise GpuSuiteError(
            "generation artifact does not satisfy the immutable prebuilt contract"
        ) from exc
    manifest = identity.manifest
    if manifest.get("capability_contract") != contract.capability_contract:
        raise GpuSuiteError("generation artifact capability contract is not exact")
    raw_rows = manifest.get("sources")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(
        contract.generation_hash_sources
    ):
        raise GpuSuiteError("generation artifact source inventory is not exact")
    expected_rows = _source_rows(contract.generation_hash_sources)
    for raw, expected in zip(raw_rows, expected_rows, strict=True):
        if not isinstance(raw, dict):
            raise GpuSuiteError("generation artifact source row is invalid")
        raw_path = _existing_regular_file(
            str(raw.get("path", "")), name="generation manifest source"
        )
        if (
            raw_path != Path(expected["path"])
            or raw.get("sha256") != expected["sha256"]
        ):
            raise GpuSuiteError("generation artifact source identity is not exact")
    return {
        "path": str(artifact),
        "sha256": _sha256_file(artifact),
        "manifest_path": str(sidecar),
        "manifest_sha256": _sha256_file(sidecar),
        "stable_module_name": identity.module_name,
        "source_rows": expected_rows,
    }


def _pytest_command(
    *, python_executable: Path, junit: Path, basetemp: Path, nodes: Sequence[str]
) -> list[str]:
    return [
        str(python_executable),
        "-m",
        "pytest",
        "-q",
        "--strict-markers",
        "-o",
        "addopts=",
        f"--junitxml={junit}",
        f"--basetemp={basetemp}",
        *nodes,
    ]


def _junit_node_id(case: ET.Element) -> str:
    classname = str(case.attrib.get("classname", ""))
    name = str(case.attrib.get("name", ""))
    if not classname or not name:
        raise GpuSuiteError("GPU JUnit testcase identity is incomplete")
    return f"{classname.replace('.', '/')}.py::{name}"


def _validate_junit(path: Path, *, required_nodes: Sequence[str], log: Path) -> None:
    checked = _existing_regular_file(path, name="GPU JUnit")
    try:
        tree = ET.parse(checked)
    except (ET.ParseError, OSError) as exc:
        raise GpuSuiteError("GPU JUnit is unreadable") from exc
    cases = list(tree.getroot().iter("testcase"))
    observed = [_junit_node_id(case) for case in cases]
    if observed != list(required_nodes):
        raise GpuSuiteError("GPU JUnit does not contain the exact ordered ten nodes")
    for case in cases:
        if any(case.find(outcome) is not None for outcome in ("failure", "error", "skipped")):
            raise GpuSuiteError("GPU JUnit contains a non-passing outcome")
    log_text = log.read_text(encoding="utf-8", errors="replace")
    if re.search(r"\b(?:xfailed|xpassed|xfail|xpass)\b", log_text, re.IGNORECASE):
        raise GpuSuiteError("GPU log contains xfail or xpass evidence")


def _locate_modular_artifact(
    basetemp: Path, contract: SimpleNamespace
) -> dict[str, Any]:
    if not basetemp.is_dir():
        raise GpuSuiteError("pytest basetemp is missing")
    _reject_symlink_components(basetemp, name="pytest basetemp")
    candidates: list[Path] = []
    for candidate in basetemp.rglob("*.so"):
        checked = _existing_regular_file(
            candidate, name="standalone modular-tick artifact"
        )
        if not checked.name.startswith("eiel_cuda_architecture_b_modular_tick_"):
            raise GpuSuiteError(
                f"unexpected shared object exists under pytest basetemp: {checked}"
            )
        candidates.append(checked)
    if len(candidates) != 1:
        raise GpuSuiteError(
            "pytest basetemp must contain exactly one standalone modular-tick artifact"
        )
    artifact = candidates[0]
    try:
        artifact.relative_to(basetemp)
    except ValueError as exc:
        raise GpuSuiteError("modular-tick artifact escapes pytest basetemp") from exc
    return {
        "path": str(artifact),
        "sha256": _sha256_file(artifact),
        "source_rows": _source_rows(contract.modular_tick_hash_sources),
    }


def _query_cuda_environment() -> dict[str, Any]:
    """Inspect CUDA only after a real ten-node pytest process succeeded."""

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise GpuSuiteError("CUDA became unavailable after the GPU suite")
    count = int(torch.cuda.device_count())
    if count < 1:
        raise GpuSuiteError("no CUDA device is visible after the GPU suite")
    index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    capability = torch.cuda.get_device_capability(index)
    return {
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": True,
        "visible_device_count": count,
        "current_device_index": index,
        "device_name": str(properties.name),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "device_total_memory_bytes": int(properties.total_memory),
    }


def _query_cuda_environment_with_selection(
    *, cuda_visible_devices: str, prebuilt_artifact: str, max_jobs: int
) -> dict[str, Any]:
    updates = {
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "MAX_JOBS": str(max_jobs),
        PREBUILT_ENV: prebuilt_artifact,
    }
    prior = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        return _query_cuda_environment()
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _atomic_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise GpuSuiteError(f"execution manifest already exists: {path}") from exc
    try:
        payload = _json_bytes(manifest)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_max_jobs(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= MAX_BUILD_MAX_JOBS:
        raise GpuSuiteError(
            f"max_jobs must be an exact integer in [1, {MAX_BUILD_MAX_JOBS}]"
        )
    return value


def _expected_command_from_manifest(manifest: Mapping[str, Any]) -> list[str]:
    pytest = manifest.get("pytest")
    runner = manifest.get("runner_environment")
    nodes = manifest.get("required_gpu_test_node_ids")
    if not isinstance(pytest, dict) or not isinstance(runner, dict) or not isinstance(nodes, list):
        raise GpuSuiteError("execution manifest command bindings are malformed")
    return _pytest_command(
        python_executable=Path(str(runner.get("python_executable", ""))),
        junit=Path(str(pytest.get("junit_path", ""))),
        basetemp=Path(str(pytest.get("basetemp_path", ""))),
        nodes=tuple(str(node) for node in nodes),
    )


def _validate_source_row_bindings(
    rows: Any,
    *,
    expected_paths: Sequence[Path] | None,
    name: str,
) -> None:
    if not isinstance(rows, list):
        raise GpuSuiteError(f"execution manifest {name} rows are malformed")
    if expected_paths is not None and len(rows) != len(expected_paths):
        raise GpuSuiteError(f"execution manifest {name} rows are not exact")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "path",
            "repo_relative_path",
            "sha256",
        }:
            raise GpuSuiteError(f"execution manifest {name} row is malformed")
        source = _existing_regular_file(str(row.get("path", "")), name=name)
        try:
            relative = source.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise GpuSuiteError(f"execution manifest {name} source escapes root") from exc
        if row.get("repo_relative_path") != relative:
            raise GpuSuiteError(f"execution manifest {name} relative path mismatch")
        if row.get("sha256") != _sha256_file(source):
            raise GpuSuiteError(f"execution manifest {name} source hash mismatch")
        if expected_paths is not None and source != expected_paths[index].resolve():
            raise GpuSuiteError(f"execution manifest {name} source order mismatch")


def validate_execution_manifest(
    manifest_path: str | Path,
    *,
    required_nodes: Sequence[str] | None = None,
    contract: SimpleNamespace | None = None,
) -> dict[str, Any]:
    """Validate self-digest, exact invocation, fixed paths, and bound files."""

    if contract is None:
        contract = _load_contract()
    path = _existing_regular_file(manifest_path, name="execution manifest")
    manifest = _load_json(path, name="execution manifest")
    if set(manifest) != _TOP_LEVEL_FIELDS:
        raise GpuSuiteError("execution manifest fields are not exact")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("scope") != SCOPE:
        raise GpuSuiteError("execution manifest schema or scope mismatch")
    if stat.S_IMODE(os.lstat(path).st_mode) & 0o222:
        raise GpuSuiteError("execution manifest is not immutable")
    digest = manifest.get("manifest_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise GpuSuiteError("execution manifest self-digest is invalid")
    if digest != _canonical_digest(manifest):
        raise GpuSuiteError("execution manifest self-digest mismatch")
    source = manifest.get("source")
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "repository_root",
            "branch",
            "commit",
            "test_commit",
            "relevant_paths",
            "relevant_source_clean",
        }
        or source.get("repository_root") != str(ROOT)
        or source.get("relevant_source_clean") is not True
        or _COMMIT_RE.fullmatch(str(source.get("commit", ""))) is None
        or source.get("test_commit") != source.get("commit")
        or not str(source.get("branch", ""))
    ):
        raise GpuSuiteError("execution manifest source identity is invalid")
    nodes = manifest.get("required_gpu_test_node_ids")
    expected_nodes = (
        tuple(required_nodes)
        if required_nodes is not None
        else tuple(contract.required_nodes)
    )
    if expected_nodes != tuple(contract.required_nodes):
        raise GpuSuiteError("requested GPU nodes do not match the Phase E gate")
    if not isinstance(nodes, list) or tuple(nodes) != expected_nodes or len(nodes) != 10:
        raise GpuSuiteError("execution manifest GPU nodes are not exact")
    command = manifest.get("command")
    if command != _expected_command_from_manifest(manifest):
        raise GpuSuiteError("execution manifest pytest command is not exact")
    pytest = manifest.get("pytest")
    if (
        not isinstance(pytest, dict)
        or set(pytest)
        != {
            "exit_code",
            "junit_path",
            "junit_sha256",
            "log_path",
            "log_sha256",
            "basetemp_path",
        }
        or pytest.get("exit_code") != 0
    ):
        raise GpuSuiteError("execution manifest does not bind a successful pytest run")
    run_dir = path.parent
    if path.name != MANIFEST_FILENAME:
        raise GpuSuiteError("execution manifest filename is not fixed")
    fixed = {
        "junit_path": run_dir / JUNIT_FILENAME,
        "log_path": run_dir / LOG_FILENAME,
        "basetemp_path": run_dir / BASETEMP_DIRNAME,
    }
    for field, expected in fixed.items():
        if Path(str(pytest.get(field, ""))) != expected:
            raise GpuSuiteError(f"execution manifest {field} is not fixed")
    for prefix, record in (
        ("generation artifact", manifest.get("generation_artifact")),
        ("modular-tick artifact", manifest.get("modular_tick_artifact")),
    ):
        if not isinstance(record, dict):
            raise GpuSuiteError(f"execution manifest {prefix} binding is malformed")
        bound = _existing_regular_file(str(record.get("path", "")), name=prefix)
        if record.get("sha256") != _sha256_file(bound):
            raise GpuSuiteError(f"execution manifest {prefix} hash mismatch")
    generation = manifest["generation_artifact"]
    if set(generation) != {
        "path",
        "sha256",
        "manifest_path",
        "manifest_sha256",
        "stable_module_name",
        "source_rows",
    }:
        raise GpuSuiteError("execution manifest generation binding fields are not exact")
    sidecar = _existing_regular_file(
        str(generation.get("manifest_path", "")),
        name="generation artifact sidecar",
    )
    if generation.get("manifest_sha256") != _sha256_file(sidecar):
        raise GpuSuiteError("execution manifest generation sidecar hash mismatch")
    runner = manifest.get("runner_environment")
    if not isinstance(runner, dict) or set(runner) != {
        "python_executable",
        "python_version",
        "max_jobs",
        "cuda_visible_devices",
        "prebuilt_environment_variable",
        "prebuilt_environment_value",
        "pytest_addopts",
        "pytest_disable_plugin_autoload",
        "python_dont_write_bytecode",
        "python_hash_seed",
    }:
        raise GpuSuiteError("execution manifest runner environment is malformed")
    runner_python = Path(str(runner.get("python_executable", "")))
    if (
        not runner_python.is_absolute()
        or not runner_python.is_file()
        or not os.access(runner_python, os.X_OK)
        or runner_python.resolve() != Path(sys.executable).resolve()
    ):
        raise GpuSuiteError("execution manifest runner Python is not exact")
    max_jobs = runner.get("max_jobs")
    if (
        type(max_jobs) is not str
        or not max_jobs.isascii()
        or not max_jobs.isdecimal()
        or str(int(max_jobs)) != max_jobs
        or not 1 <= int(max_jobs) <= MAX_BUILD_MAX_JOBS
        or not str(runner.get("cuda_visible_devices", "")).strip()
        or runner.get("prebuilt_environment_variable") != PREBUILT_ENV
        or runner.get("prebuilt_environment_value") != generation.get("path")
        or runner.get("pytest_addopts") != ""
        or runner.get("pytest_disable_plugin_autoload") != "1"
        or runner.get("python_dont_write_bytecode") != "1"
        or runner.get("python_hash_seed") != "0"
    ):
        raise GpuSuiteError("execution manifest runner environment is not exact")
    cuda = manifest.get("cuda_environment")
    if (
        not isinstance(cuda, dict)
        or set(cuda)
        != {
            "python_version",
            "numpy_version",
            "torch_version",
            "torch_cuda_version",
            "cuda_available",
            "visible_device_count",
            "current_device_index",
            "device_name",
            "compute_capability",
            "device_total_memory_bytes",
        }
        or cuda.get("cuda_available") is not True
        or not isinstance(cuda.get("visible_device_count"), int)
        or int(cuda["visible_device_count"]) < 1
        or type(cuda.get("current_device_index")) is not int
        or not 0 <= int(cuda["current_device_index"]) < int(cuda["visible_device_count"])
        or type(cuda.get("compute_capability")) is not list
        or len(cuda["compute_capability"]) != 2
        or any(type(value) is not int or value < 0 for value in cuda["compute_capability"])
        or type(cuda.get("device_total_memory_bytes")) is not int
        or int(cuda["device_total_memory_bytes"]) <= 0
        or any(
            type(cuda.get(name)) is not str or not str(cuda[name])
            for name in (
                "python_version",
                "numpy_version",
                "torch_version",
                "torch_cuda_version",
                "device_name",
            )
        )
    ):
        raise GpuSuiteError("execution manifest CUDA environment is invalid")
    modular = manifest["modular_tick_artifact"]
    if set(modular) != {"path", "sha256", "source_rows"}:
        raise GpuSuiteError("execution manifest modular binding fields are not exact")
    _validate_source_row_bindings(
        generation.get("source_rows"),
        expected_paths=contract.generation_hash_sources,
        name="generation source",
    )
    _validate_source_row_bindings(
        modular.get("source_rows"),
        expected_paths=contract.modular_tick_hash_sources,
        name="modular-tick source",
    )
    for field, hash_field, name in (
        ("junit_path", "junit_sha256", "GPU JUnit"),
        ("log_path", "log_sha256", "GPU log"),
    ):
        bound = _existing_regular_file(str(pytest.get(field, "")), name=name)
        if pytest.get(hash_field) != _sha256_file(bound):
            raise GpuSuiteError(f"execution manifest {name} hash mismatch")
    _validate_junit(
        Path(str(pytest["junit_path"])),
        required_nodes=expected_nodes,
        log=Path(str(pytest["log_path"])),
    )
    modular_path = Path(str(modular.get("path", "")))
    try:
        modular_path.relative_to(Path(str(pytest["basetemp_path"])))
    except ValueError as exc:
        raise GpuSuiteError("execution manifest modular artifact escapes basetemp") from exc
    if _validate_generation_artifact(
        str(generation.get("path", "")), contract
    ) != generation:
        raise GpuSuiteError("execution manifest generation binding is not exact")
    if _locate_modular_artifact(
        Path(str(pytest["basetemp_path"])), contract
    ) != modular:
        raise GpuSuiteError("execution manifest modular binding is not exact")
    return manifest


def run_suite(
    *,
    run_directory: str | Path,
    generation_artifact: str | Path,
    cuda_visible_devices: str,
    python_executable: str | Path = sys.executable,
    max_jobs: int = DEFAULT_BUILD_MAX_JOBS,
) -> Path:
    contract = _load_contract()
    max_jobs = _validate_max_jobs(max_jobs)
    if len(contract.required_nodes) != 10 or len(set(contract.required_nodes)) != 10:
        raise GpuSuiteError("Phase E gate does not define exactly ten unique GPU nodes")
    run_dir = _new_run_directory(run_directory)
    python_path = _raw_absolute_path(python_executable, name="Python executable")
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise GpuSuiteError(f"Python executable is missing or not executable: {python_path}")
    if python_path != _raw_absolute_path(sys.executable, name="runner Python executable"):
        raise GpuSuiteError(
            "GPU pytest must use the same Python executable as the controlled runner"
        )
    if not cuda_visible_devices.strip():
        raise GpuSuiteError("CUDA visible-device selection must be explicit and non-empty")
    if any(character in cuda_visible_devices for character in "\r\n\0"):
        raise GpuSuiteError("CUDA visible-device selection contains unsafe characters")
    source_before = _source_identity(contract)
    generation = _validate_generation_artifact(generation_artifact, contract)
    modular_sources_before = _source_rows(contract.modular_tick_hash_sources)
    junit = run_dir / JUNIT_FILENAME
    log = run_dir / LOG_FILENAME
    basetemp = run_dir / BASETEMP_DIRNAME
    manifest_path = run_dir / MANIFEST_FILENAME
    command = _pytest_command(
        python_executable=python_path,
        junit=junit,
        basetemp=basetemp,
        nodes=contract.required_nodes,
    )
    environment = dict(os.environ)
    environment[PREBUILT_ENV] = generation["path"]
    environment["MAX_JOBS"] = str(max_jobs)
    environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    run_dir.mkdir(mode=0o700)
    try:
        with log.open("xb") as handle:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise GpuSuiteError(
                f"exact Phase E GPU pytest suite failed with exit code {result.returncode}"
            )
        _validate_junit(junit, required_nodes=contract.required_nodes, log=log)
        modular = _locate_modular_artifact(basetemp, contract)
        source_after = _source_identity(contract)
        if source_after != source_before:
            raise GpuSuiteError("Phase E source identity changed during the GPU suite")
        if modular["source_rows"] != modular_sources_before:
            raise GpuSuiteError("modular-tick sources changed during the GPU suite")
        cuda_environment = _query_cuda_environment_with_selection(
            cuda_visible_devices=cuda_visible_devices,
            prebuilt_artifact=generation["path"],
            max_jobs=max_jobs,
        )
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "scope": SCOPE,
            "source": source_before,
            "command": command,
            "required_gpu_test_node_ids": list(contract.required_nodes),
            "runner_environment": {
                "python_executable": str(python_path),
                "python_version": platform.python_version(),
                "max_jobs": str(max_jobs),
                "cuda_visible_devices": cuda_visible_devices,
                "prebuilt_environment_variable": PREBUILT_ENV,
                "prebuilt_environment_value": generation["path"],
                "pytest_addopts": "",
                "pytest_disable_plugin_autoload": "1",
                "python_dont_write_bytecode": "1",
                "python_hash_seed": "0",
            },
            "cuda_environment": cuda_environment,
            "generation_artifact": generation,
            "modular_tick_artifact": modular,
            "pytest": {
                "exit_code": 0,
                "junit_path": str(junit),
                "junit_sha256": _sha256_file(junit),
                "log_path": str(log),
                "log_sha256": _sha256_file(log),
                "basetemp_path": str(basetemp),
            },
        }
        manifest["manifest_sha256"] = _canonical_digest(manifest)
        _atomic_manifest(manifest_path, manifest)
        validate_execution_manifest(
            manifest_path,
            required_nodes=contract.required_nodes,
            contract=contract,
        )
        return manifest_path
    except Exception:
        if manifest_path.exists():
            # A manifest is created last.  Never leave one behind when its own
            # post-write validation fails.
            manifest_path.chmod(0o600)
            manifest_path.unlink()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact ten-node Phase E GPU suite and write one immutable "
            "execution manifest only after complete success."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="execute the controlled GPU suite")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--generation-artifact", required=True)
    run.add_argument("--cuda-visible-devices", required=True)
    run.add_argument("--python", default=sys.executable)
    run.add_argument(
        "--max-jobs",
        type=int,
        default=DEFAULT_BUILD_MAX_JOBS,
        help=(
            "maximum native build workers exposed as MAX_JOBS "
            f"(default: {DEFAULT_BUILD_MAX_JOBS})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = run_suite(
            run_directory=args.run_dir,
            generation_artifact=args.generation_artifact,
            cuda_visible_devices=args.cuda_visible_devices,
            python_executable=args.python,
            max_jobs=args.max_jobs,
        )
    except GpuSuiteError as exc:
        print(f"EIEL_PHASE_E_GPU_SUITE_REJECTED: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
