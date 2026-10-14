"""Strict loader and executor for ``eiel_validation_plan_v1`` plans.

Plans use argument arrays and an explicit environment allowlist. They never
pass text to a shell. Accelerator selection is rejected at validation time and
hidden again in every child environment at execution time. This policy prevents
accidental accelerator use by operator-controlled plans; it is not an OS-level
sandbox for hostile executables.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "eiel_validation_plan_v1"
RECEIPT_VERSION = "eiel_validation_plan_receipt_v1"
_ID_RE = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*\Z")
_ENV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_GPU_ENV_RE = re.compile(r"(?:CUDA|NVIDIA|ROCM|HIP|GPU)", re.IGNORECASE)
_GPU_VALUE_RE = re.compile(r"(?:^|[^a-z0-9])(?:cuda|gpu|rocm)(?::\d+)?(?:$|[^a-z0-9])", re.IGNORECASE)
_GPU_FLAGS = ("--cuda", "--gpu", "--use-gpu", "--require-gpu")
_DEVICE_FLAGS = ("--device", "--accelerator")
_GPU_EXECUTABLES = frozenset({"nvidia-smi", "nvcc", "rocm-smi", "hipcc"})
_SHELL_EXECUTABLES = frozenset(
    {"bash", "cmd", "csh", "dash", "fish", "ksh", "powershell", "pwsh", "sh", "tcsh", "zsh"}
)
_CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "void",
    "ROCR_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
}
_TOP_KEYS = frozenset(
    {"schema_version", "plan_id", "description", "cpu_only", "defaults", "steps"}
)
_DEFAULT_KEYS = frozenset({"cwd", "timeout_seconds", "inherit_env", "env"})
_STEP_KEYS = frozenset({"id", "argv", "cwd", "timeout_seconds", "env"})


class ValidationPlanError(RuntimeError):
    """A plan is invalid or could not be executed safely."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that also rejects duplicate mapping keys."""


def _unique_yaml_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValidationPlanError(f"duplicate YAML field {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_yaml_mapping,
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationPlanError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValidationPlanError(f"non-finite JSON number {value!r} is not allowed")


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValidationPlanError(f"{name} must be an object with string keys")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], allowed: frozenset[str], *, name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValidationPlanError(f"{name} contains unsupported fields: {unknown}")


def _identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValidationPlanError(f"{name} must be a lowercase stable identifier")
    return value


def _timeout(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 86_400:
        raise ValidationPlanError(f"{name} must be an integer from 1 through 86400")
    return value


def _relative_cwd(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValidationPlanError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValidationPlanError(f"{name} must stay beneath the execution root")
    return path.as_posix()


def _environment(value: Any, *, name: str) -> dict[str, str]:
    env = _object(value, name=name)
    for key, item in env.items():
        if _ENV_RE.fullmatch(key) is None:
            raise ValidationPlanError(f"{name} has an invalid variable name: {key!r}")
        if _GPU_ENV_RE.search(key):
            raise ValidationPlanError(f"{name} may not configure GPU variable {key!r}")
        if not isinstance(item, str) or "\x00" in item:
            raise ValidationPlanError(f"{name}.{key} must be a string without NUL")
        if _GPU_VALUE_RE.fullmatch(item.strip()):
            raise ValidationPlanError(f"{name}.{key} may not request a GPU value")
    return dict(sorted(env.items()))


def _argv(value: Any, *, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item and "\x00" not in item for item in value)
    ):
        raise ValidationPlanError(f"{name} must be a non-empty array of non-empty strings")
    executable = Path(value[0]).name.lower()
    if executable in _GPU_EXECUTABLES:
        raise ValidationPlanError(f"{name} invokes a GPU executable")
    if executable in _SHELL_EXECUTABLES:
        raise ValidationPlanError(f"{name} may not invoke a shell interpreter")
    if not Path(value[0]).is_absolute() and "/" not in value[0]:
        raise ValidationPlanError(
            f"{name}[0] must be an absolute or explicitly relative executable path"
        )
    for index, item in enumerate(value):
        lowered = item.lower()
        if lowered in {"cuda", "gpu", "rocm"} or any(
            lowered == flag or lowered.startswith(flag + "=") for flag in _GPU_FLAGS
        ):
            raise ValidationPlanError(f"{name} requests GPU execution at argument {index}")
        for flag in _DEVICE_FLAGS:
            if lowered.startswith(flag + "=") and _GPU_VALUE_RE.search(lowered.split("=", 1)[1]):
                raise ValidationPlanError(f"{name} requests a GPU device at argument {index}")
            if lowered == flag and index + 1 < len(value) and _GPU_VALUE_RE.search(value[index + 1]):
                raise ValidationPlanError(f"{name} requests a GPU device at argument {index + 1}")
        if ("torch.cuda" in lowered or "cupy.cuda" in lowered or "jax.devices('gpu" in lowered):
            raise ValidationPlanError(f"{name} contains an inline GPU request at argument {index}")
    return list(value)


def validate_plan(value: Any) -> dict[str, Any]:
    """Return a canonical plan or raise on every unsupported/unsafe shape."""

    plan = _object(value, name="plan")
    _exact_keys(plan, _TOP_KEYS, name="plan")
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValidationPlanError(f"schema_version must be {SCHEMA_VERSION!r}")
    if plan.get("cpu_only") is not True:
        raise ValidationPlanError("cpu_only must be true")
    plan_id = _identifier(plan.get("plan_id"), name="plan_id")
    description = plan.get("description", "")
    if not isinstance(description, str):
        raise ValidationPlanError("description must be a string")

    defaults = _object(plan.get("defaults", {}), name="defaults")
    _exact_keys(defaults, _DEFAULT_KEYS, name="defaults")
    default_cwd = _relative_cwd(defaults.get("cwd", "."), name="defaults.cwd")
    default_timeout = _timeout(defaults.get("timeout_seconds", 300), name="defaults.timeout_seconds")
    inherit_env = defaults.get("inherit_env", [])
    if not isinstance(inherit_env, list) or not all(isinstance(item, str) for item in inherit_env):
        raise ValidationPlanError("defaults.inherit_env must be an array of variable names")
    if len(set(inherit_env)) != len(inherit_env):
        raise ValidationPlanError("defaults.inherit_env must not contain duplicates")
    for key in inherit_env:
        if _ENV_RE.fullmatch(key) is None or _GPU_ENV_RE.search(key):
            raise ValidationPlanError(f"defaults.inherit_env contains unsafe variable {key!r}")
    inherit_env = sorted(inherit_env)
    default_env = _environment(defaults.get("env", {}), name="defaults.env")

    raw_steps = plan.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValidationPlanError("steps must be a non-empty array")
    steps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_steps):
        step = _object(raw, name=f"steps[{index}]")
        _exact_keys(step, _STEP_KEYS, name=f"steps[{index}]")
        step_id = _identifier(step.get("id"), name=f"steps[{index}].id")
        if step_id in seen:
            raise ValidationPlanError(f"duplicate step id: {step_id!r}")
        seen.add(step_id)
        steps.append(
            {
                "id": step_id,
                "argv": _argv(step.get("argv"), name=f"steps[{index}].argv"),
                "cwd": _relative_cwd(step.get("cwd", default_cwd), name=f"steps[{index}].cwd"),
                "timeout_seconds": _timeout(
                    step.get("timeout_seconds", default_timeout),
                    name=f"steps[{index}].timeout_seconds",
                ),
                "env": _environment(step.get("env", {}), name=f"steps[{index}].env"),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan_id,
        "description": description,
        "cpu_only": True,
        "defaults": {
            "cwd": default_cwd,
            "timeout_seconds": default_timeout,
            "inherit_env": inherit_env,
            "env": default_env,
        },
        "steps": steps,
    }


def load_plan(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationPlanError(f"cannot read plan: {source}") from exc
    try:
        if source.suffix.lower() == ".json":
            value = json.loads(
                text,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        elif source.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.load(text, Loader=_UniqueKeySafeLoader)
        else:
            raise ValidationPlanError("plan filename must end in .json, .yaml, or .yml")
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValidationPlanError(f"plan is not valid {source.suffix.lower()[1:].upper()}") from exc
    return validate_plan(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _captured_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _execution_root(value: str | Path) -> Path:
    root = Path(value)
    if not root.is_absolute():
        raise ValidationPlanError("execution root must be absolute")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ValidationPlanError(f"execution root is unavailable: {root}") from exc
    if not resolved.is_dir():
        raise ValidationPlanError(f"execution root is not a directory: {resolved}")
    return resolved


def _step_cwd(root: Path, raw: str, *, step_id: str) -> Path:
    try:
        cwd = (root / raw).resolve(strict=True)
        cwd.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValidationPlanError(f"step {step_id!r} cwd escapes or is unavailable") from exc
    if not cwd.is_dir():
        raise ValidationPlanError(f"step {step_id!r} cwd is not a directory")
    return cwd


def execute_plan(plan: Mapping[str, Any], *, root: str | Path) -> dict[str, Any]:
    """Execute a validated plan sequentially and return its JSON receipt."""

    canonical = validate_plan(plan)
    checked_root = _execution_root(root)
    defaults = canonical["defaults"]
    base_env: dict[str, str] = {}
    for key in defaults["inherit_env"]:
        if key not in os.environ:
            raise ValidationPlanError(f"required inherited variable is unset: {key}")
        base_env[key] = os.environ[key]
    base_env.update(defaults["env"])
    base_env.update(_CPU_ENV)

    rows: list[dict[str, Any]] = []
    overall = "passed"
    for step in canonical["steps"]:
        cwd = _step_cwd(checked_root, step["cwd"], step_id=step["id"])
        env = dict(base_env)
        env.update(step["env"])
        env.update(_CPU_ENV)
        try:
            completed = subprocess.run(
                step["argv"],
                cwd=cwd,
                env=env,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=step["timeout_seconds"],
            )
            row = {
                "id": step["id"],
                "status": "passed" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "argv": step["argv"],
                "cwd": step["cwd"],
                "environment_keys": sorted(env),
                "cpu_masked_variables": sorted(_CPU_ENV),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except subprocess.TimeoutExpired as exc:
            row = {
                "id": step["id"],
                "status": "timed_out",
                "returncode": None,
                "argv": step["argv"],
                "cwd": step["cwd"],
                "environment_keys": sorted(env),
                "cpu_masked_variables": sorted(_CPU_ENV),
                "stdout": _captured_text(exc.stdout),
                "stderr": _captured_text(exc.stderr),
            }
        except OSError as exc:
            row = {
                "id": step["id"],
                "status": "launch_failed",
                "returncode": None,
                "argv": step["argv"],
                "cwd": step["cwd"],
                "environment_keys": sorted(env),
                "cpu_masked_variables": sorted(_CPU_ENV),
                "stdout": "",
                "stderr": str(exc),
            }
        rows.append(row)
        if row["status"] != "passed":
            overall = "failed"
            break

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_VERSION,
        "plan_schema_version": SCHEMA_VERSION,
        "plan_id": canonical["plan_id"],
        "plan_sha256": hashlib.sha256(_canonical_bytes(canonical)).hexdigest(),
        "cpu_only": True,
        "status": overall,
        "steps": rows,
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    return receipt


def write_receipt(path: str | Path, receipt: Mapping[str, Any]) -> None:
    """Atomically write a receipt as canonical, human-readable JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
