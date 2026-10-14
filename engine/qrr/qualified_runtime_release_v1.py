"""Immutable Qualified Runtime Release V1 materialization and verification."""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import ast
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Final
from uuid import uuid4

import torch

from engine.campaigns.canonical import canonical_sha256, pretty_json
from engine.campaigns.compiler import (
    compile_campaign,
    load_campaign_spec,
    resolve_campaign_profile,
)
from engine.campaigns.providers import (
    ExecutionProviderRegistryV1,
    default_execution_provider_registry,
)
from engine.campaigns.qualification import (
    execution_route_qualification_status_v1,
)
from engine.campaigns.tick_native import TickNativeError, TickNativeExecutionRequestV1
from engine.campaigns.long_horizon_qualification import (
    SCIENCE_ROUTE_QUALIFIED,
)
from engine.qrr.qualified_execution_envelope_v1 import (
    build_qualified_execution_envelope_v1,
    load_qualified_execution_envelope_v1,
)
from engine.qrr.historical_escape_suite_v1 import historical_escape_inventory_v1
from engine.qrr.tick_native_materialization_v1 import canonical_tick_native_plan_v1

QUALIFIED_RUNTIME_RELEASE_SCHEMA_VERSION: Final[str] = (
    "eiel_qualified_runtime_release_v1"
)
QRR_INTEGRITY_SCHEMA_VERSION: Final[str] = "eiel_qrr_integrity_manifest_v1"
DEFAULT_QRR_ROOT: Final[Path] = Path(
    public_runtime_path('qualified_runtime_releases_v1')
)
DEFAULT_QRR_RUN_ROOT: Final[Path] = Path(public_runtime_path('qrr_runs_v1'))
_ROUTE_SOURCE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".json",
        ".py",
        ".toml",
        ".yaml",
        ".yml",
    }
)


class QualifiedRuntimeReleaseError(ValueError):
    """A release cannot be built or verified without weakening authority."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(pretty_json(value), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def runtime_environment_fingerprint_v1() -> dict[str, str | None]:
    """Capture exact-match V1 semantic runtime dependencies without CUDA work."""

    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    libc_name, libc_version = platform.libc_ver()
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "libc": f"{libc_name}-{libc_version}" if libc_name else None,
        "torch_version": torch_version,
        "torch_cuda_version": torch.version.cuda,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


def stable_python_executable_v1(
    python_executable: str | Path | None = None,
) -> Path:
    """Select an explicit durable interpreter without erasing venv identity."""

    executable = Path(python_executable or sys.executable).expanduser().absolute()
    if "emergence-worktrees" in executable.parts:
        raise QualifiedRuntimeReleaseError(
            "QRR runtime Python executable belongs to an ephemeral worktree"
        )
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise QualifiedRuntimeReleaseError(
            "QRR stable Python executable is absent or not executable"
        )
    return executable


def stable_runtime_environment_fingerprint_v1(
    python_executable: str | Path | None = None,
) -> dict[str, str | None]:
    """Fingerprint the exact environment seen through the sealed launcher."""

    executable = stable_python_executable_v1(python_executable)
    probe = """
import importlib.metadata
import json
import os
import platform
import sys

try:
    torch_version = importlib.metadata.version("torch")
except importlib.metadata.PackageNotFoundError:
    torch_version = None
try:
    import torch
except Exception as exc:
    raise SystemExit(f"torch import failed: {type(exc).__name__}: {exc}")
libc_name, libc_version = platform.libc_ver()
print(json.dumps({
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "python_executable": sys.executable,
    "python_prefix": sys.prefix,
    "platform_system": platform.system(),
    "platform_machine": platform.machine(),
    "libc": f"{libc_name}-{libc_version}" if libc_name else None,
    "torch_version": torch_version,
    "torch_cuda_version": torch.version.cuda,
    "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [str(executable), "-I", "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise QualifiedRuntimeReleaseError(
            "QRR stable Python environment probe failed: " + completed.stderr.strip()
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise QualifiedRuntimeReleaseError(
            "QRR stable Python environment probe was malformed"
        ) from exc
    if not isinstance(value, Mapping):
        raise QualifiedRuntimeReleaseError(
            "QRR stable Python environment probe was not an object"
        )
    fingerprint = {str(name): item for name, item in value.items()}
    observed_executable = fingerprint.get("python_executable")
    if (
        not isinstance(observed_executable, str)
        or Path(observed_executable).absolute() != executable
    ):
        raise QualifiedRuntimeReleaseError(
            "QRR stable Python executable did not preserve exact identity"
        )
    return fingerprint


def validate_runtime_environment_v1(expected: Mapping[str, Any]) -> None:
    expected_executable = expected.get("python_executable")
    observed = (
        stable_runtime_environment_fingerprint_v1(expected_executable)
        if isinstance(expected_executable, str)
        and Path(expected_executable).is_absolute()
        and Path(expected_executable).is_file()
        else runtime_environment_fingerprint_v1()
    )
    if dict(expected) != observed:
        differing = sorted(
            name
            for name in set(expected) | set(observed)
            if expected.get(name) != observed.get(name)
        )
        raise QualifiedRuntimeReleaseError(
            "QRR environment exact-match failed: " + ", ".join(differing)
        )


def _module_name_for_path(path: Path, source_root: Path) -> tuple[str, bool]:
    relative = path.relative_to(source_root)
    parts = list(relative.with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _local_module_path(module: str, source_root: Path) -> Path | None:
    parts = tuple(part for part in module.split(".") if part)
    for root in (source_root, source_root / "engine"):
        file_path = root.joinpath(*parts).with_suffix(".py")
        package_path = root.joinpath(*parts) / "__init__.py"
        if file_path.is_file():
            return file_path.resolve()
        if package_path.is_file():
            return package_path.resolve()
    return None


def _imported_local_paths(path: Path, source_root: Path) -> set[Path]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise QualifiedRuntimeReleaseError(
            f"cannot inspect Python source closure: {path}: {exc}"
        ) from exc
    module_name, is_package = _module_name_for_path(path, source_root)
    package_parts = (
        module_name.split(".") if is_package else module_name.split(".")[:-1]
    )
    imported: set[Path] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = list(package_parts)
            if node.level:
                drop = node.level - 1
                prefix = prefix[: len(prefix) - drop] if drop else prefix
            elif node.module:
                prefix = []
            base = ".".join((*prefix, *(node.module or "").split("."))).strip(".")
            if base:
                names.append(base)
            names.extend(
                f"{base}.{alias.name}" if base else alias.name
                for alias in node.names
                if alias.name != "*"
            )
        for name in names:
            candidate = _local_module_path(name, source_root)
            if candidate is not None:
                imported.add(candidate)
    return imported


def discover_python_payload_paths_v1(source_root: Path) -> tuple[Path, ...]:
    """Discover route code/data roots plus their static Python import closure."""

    root = source_root.expanduser().resolve()
    route_roots = tuple(
        root / relative
        for relative in (
            "engine/campaigns",
            "engine/native_runtime",
            "engine/perf",
            "engine/simulation",
        )
    )
    initial: set[Path] = set()
    for route_root in route_roots:
        if not route_root.is_dir():
            raise QualifiedRuntimeReleaseError(
                f"qualified route source root is absent: {route_root}"
            )
        initial.update(
            path.resolve()
            for path in route_root.rglob("*")
            if path.is_file()
            and path.suffix in _ROUTE_SOURCE_SUFFIXES
            and "__pycache__" not in path.parts
        )
    for relative in (
        "engine/__init__.py",
        "tools/__init__.py",
        "tools/scripts/__init__.py",
        "tools/scripts/eiel_campaign.py",
        "tools/scripts/eiel_wrc_science.py",
    ):
        candidate = (root / relative).resolve()
        if candidate.is_file():
            initial.add(candidate)
    initial.update(
        path.resolve() for path in (root / "engine/qrr").glob("*.py") if path.is_file()
    )
    queue: deque[Path] = deque(sorted(path for path in initial if path.suffix == ".py"))
    discovered = set(initial)
    while queue:
        source = queue.popleft()
        for dependency in _imported_local_paths(source, root):
            if dependency not in discovered:
                discovered.add(dependency)
                queue.append(dependency)
            relative = dependency.relative_to(root)
            for parent in relative.parents:
                init = (root / parent / "__init__.py").resolve()
                if init.is_file() and init not in discovered:
                    discovered.add(init)
                    queue.append(init)
    return tuple(sorted(discovered))


def discover_runtime_data_payload_paths_v1(source_root: Path) -> tuple[Path, ...]:
    """Return route-imported declarative data that Python imports cannot reveal."""

    root = source_root.expanduser().resolve()
    declared: list[Path] = []
    for relative in (
        "_spec/authoritative_state/v1",
        "registries/performance",
    ):
        data_root = root / relative
        if data_root.is_dir():
            declared.extend(
                path.resolve()
                for path in data_root.rglob("*")
                if path.is_file() and path.suffix in {".json", ".yaml", ".yml", ".toml"}
            )
    return tuple(sorted(set(declared)))


def _git_source_identity(source_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        head = run("rev-parse", "HEAD")
        status = run("status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise QualifiedRuntimeReleaseError(
            "QRR source is not a readable Git checkout"
        ) from exc
    if status:
        raise QualifiedRuntimeReleaseError(
            "QRR build requires an exact clean qualified source checkout"
        )
    return {
        "git_commit": head,
        "git_commit_sha256": canonical_sha256({"git_commit": head}),
        "working_tree": "clean",
    }


def _verified_artifact_rows(
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise QualifiedRuntimeReleaseError("qualified spec has no artifact map")
    verified: list[tuple[str, Mapping[str, Any], Path, str]] = []
    for artifact_id, raw in sorted(artifacts.items()):
        if not isinstance(raw, Mapping):
            raise QualifiedRuntimeReleaseError(f"artifact {artifact_id!r} is malformed")
        path = Path(str(raw.get("path") or "")).expanduser().resolve()
        digest = raw.get("sha256")
        if not path.is_file() or path.is_symlink():
            raise QualifiedRuntimeReleaseError(
                f"artifact {artifact_id!r} is absent or symlinked"
            )
        if not isinstance(digest, str) or _file_sha256(path) != digest:
            raise QualifiedRuntimeReleaseError(
                f"artifact {artifact_id!r} SHA-256 differs"
            )
        verified.append((str(artifact_id), raw, path, digest))
    if not verified:
        raise QualifiedRuntimeReleaseError("qualified spec has no frozen artifacts")
    common_root = Path(
        os.path.commonpath([str(path.parent) for _, _, path, _ in verified])
    )
    rows: list[dict[str, Any]] = []
    for artifact_id, raw, path, digest in verified:
        role = "native" if path.suffix in {".so", ".dll", ".dylib"} else "contracts"
        rows.append(
            {
                "artifact_id": artifact_id,
                "qualified_path": str(path),
                "payload_path": (
                    Path("payload/artifacts") / path.relative_to(common_root)
                ).as_posix(),
                "sha256": digest,
                "abi": raw.get("abi"),
                "role": role,
            }
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class QualifiedRuntimeReleaseV1:
    """Machine-checkable identity for one immutable executable release."""

    release_id: str
    source_identity: Mapping[str, Any]
    qualified_execution_envelope: Mapping[str, Any]
    environment_contract: Mapping[str, Any]
    qualified_artifacts: tuple[Mapping[str, Any], ...]
    python_payload: tuple[Mapping[str, Any], ...]
    qualification_authority: Mapping[str, Any]
    template_plan_sha256: str
    historical_escape_suite_id: str
    historical_escape_suite_inventory_sha256: str
    launcher_sha256: str
    created_at_utc: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": QUALIFIED_RUNTIME_RELEASE_SCHEMA_VERSION,
            "source_identity": dict(self.source_identity),
            "qualified_execution_envelope": dict(self.qualified_execution_envelope),
            "environment_contract": dict(self.environment_contract),
            "qualified_artifacts": [dict(row) for row in self.qualified_artifacts],
            "python_payload": [dict(row) for row in self.python_payload],
            "qualification_authority": dict(self.qualification_authority),
            "template_plan_sha256": self.template_plan_sha256,
            "historical_escape_suite_id": self.historical_escape_suite_id,
            "historical_escape_suite_inventory_sha256": (
                self.historical_escape_suite_inventory_sha256
            ),
            "launcher_sha256": self.launcher_sha256,
        }

    @property
    def expected_release_id(self) -> str:
        return f"qrr1-{canonical_sha256(self.identity_payload())[:32]}"

    def __post_init__(self) -> None:
        if self.release_id != self.expected_release_id:
            raise QualifiedRuntimeReleaseError("QRR release identity is invalid")

    def payload(self) -> dict[str, Any]:
        payload = {
            **self.identity_payload(),
            "release_id": self.release_id,
            "created_at_utc": self.created_at_utc,
        }
        return {**payload, "manifest_sha256": canonical_sha256(payload)}


def _launcher_text(python_executable: str | Path) -> str:
    executable = Path(python_executable).expanduser()
    if not executable.is_absolute():
        raise QualifiedRuntimeReleaseError(
            "QRR launcher requires an absolute Python executable"
        )
    return f"""#!{executable}
from pathlib import Path
import os
import sys

sys.dont_write_bytecode = True
release_root = Path(__file__).resolve().parents[1]
if os.environ.get("PYTHONPATH"):
    raise SystemExit("QRR refuses external PYTHONPATH")
payload = release_root / "payload" / "python"
sys.path.insert(0, str(payload / "engine"))
sys.path.insert(0, str(payload))
from engine.qrr.cli import main
raise SystemExit(main(release_root=release_root))
"""


def _copy_payload_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _write_integrity_manifest(release_root: Path) -> None:
    rows = []
    for path in sorted(release_root.rglob("*")):
        if not path.is_file() or path.name == "integrity.json":
            continue
        rows.append(
            {
                "path": path.relative_to(release_root).as_posix(),
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {
        "schema_version": QRR_INTEGRITY_SCHEMA_VERSION,
        "files": rows,
    }
    _atomic_json(
        release_root / "integrity.json",
        {**payload, "integrity_sha256": canonical_sha256(payload)},
    )


def _seal_release_permissions(release_root: Path) -> None:
    """Make the materialized tree read-only after its integrity proof."""

    for path in sorted(release_root.rglob("*")):
        if path.is_file():
            path.chmod(0o555 if path.parent.name == "bin" else 0o444)
    for path in sorted(
        (row for row in release_root.rglob("*") if row.is_dir()),
        key=lambda row: len(row.parts),
        reverse=True,
    ):
        path.chmod(0o555)
    release_root.chmod(0o555)


def verify_release_integrity_v1(release_root: Path) -> dict[str, Any]:
    root = release_root.expanduser().resolve()
    try:
        integrity = json.loads((root / "integrity.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualifiedRuntimeReleaseError(
            "QRR integrity manifest is unreadable"
        ) from exc
    if not isinstance(integrity, Mapping):
        raise QualifiedRuntimeReleaseError("QRR integrity manifest is malformed")
    core = {key: value for key, value in integrity.items() if key != "integrity_sha256"}
    if integrity.get("schema_version") != QRR_INTEGRITY_SCHEMA_VERSION or integrity.get(
        "integrity_sha256"
    ) != canonical_sha256(core):
        raise QualifiedRuntimeReleaseError("QRR integrity manifest identity differs")
    expected = {
        str(row["path"]): row
        for row in integrity.get("files", ())
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "integrity.json"
    }
    if actual != set(expected):
        raise QualifiedRuntimeReleaseError("QRR integrity file inventory differs")
    for relative, row in expected.items():
        path = root / relative
        if path.is_symlink() or _file_sha256(path) != row.get("sha256"):
            raise QualifiedRuntimeReleaseError(
                f"QRR payload integrity differs: {relative}"
            )
    return {
        "schema_version": "eiel_qrr_integrity_verification_v1",
        "release_root": str(root),
        "file_count": len(expected),
        "integrity_sha256": integrity["integrity_sha256"],
        "valid": True,
    }


def _contained_release_path(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise QualifiedRuntimeReleaseError(
            "QRR manifest contains an unsafe payload path"
        )
    resolved = (root / raw).resolve()
    if root != resolved and root not in resolved.parents:
        raise QualifiedRuntimeReleaseError("QRR manifest payload escapes release root")
    return resolved


def _validate_release_semantic_bindings_v1(
    root: Path, release: QualifiedRuntimeReleaseV1
) -> None:
    launcher = root / "bin/qrr"
    if not launcher.is_file() or launcher.is_symlink():
        raise QualifiedRuntimeReleaseError("QRR canonical launcher is absent")
    if _file_sha256(launcher) != release.launcher_sha256:
        raise QualifiedRuntimeReleaseError("QRR canonical launcher SHA-256 differs")
    expected_python = release.environment_contract.get("python_executable")
    if (
        not isinstance(expected_python, str)
        or not Path(expected_python).is_absolute()
        or launcher.read_text(encoding="utf-8").splitlines()[0]
        != f"#!{expected_python}"
    ):
        raise QualifiedRuntimeReleaseError(
            "QRR launcher is not bound to its stable environment contract"
        )
    for row in release.qualified_artifacts:
        relative = row.get("payload_path")
        digest = row.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise QualifiedRuntimeReleaseError("QRR artifact binding is malformed")
        artifact = _contained_release_path(root, relative)
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or _file_sha256(artifact) != digest
        ):
            raise QualifiedRuntimeReleaseError(
                f"QRR qualified artifact binding differs: {relative}"
            )
    for row in release.python_payload:
        relative = row.get("path")
        digest = row.get("sha256")
        size = row.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or type(size) is not int
            or size < 0
        ):
            raise QualifiedRuntimeReleaseError(
                "QRR Python payload binding is malformed"
            )
        source = _contained_release_path(root / "payload/python", relative)
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size != size
            or _file_sha256(source) != digest
        ):
            raise QualifiedRuntimeReleaseError(
                f"QRR Python payload binding differs: {relative}"
            )
    expected_python_payload_sha256 = canonical_sha256(
        {"files": [dict(row) for row in release.python_payload]}
    )
    if (
        release.source_identity.get("python_payload_sha256")
        != expected_python_payload_sha256
    ):
        raise QualifiedRuntimeReleaseError("QRR Python source-closure identity differs")
    if (
        release.historical_escape_suite_id != "HistoricalQualificationEscapeSuiteV1"
        or len(release.historical_escape_suite_inventory_sha256) != 64
    ):
        raise QualifiedRuntimeReleaseError(
            "QRR historical escape-suite binding is malformed"
        )
    envelope = load_qualified_execution_envelope_v1(
        release.qualified_execution_envelope
    )
    authority = release.qualification_authority
    legacy_compatibility_manifest = (
        envelope.time_contract.get("authority") == "legacy_generation_route_v1"
        and "qualification_dimensions" not in authority
        and "qualification_evidence_binding" not in authority
        and "execution_lane" not in authority
        and "production_science_authorized" not in authority
    )
    if not legacy_compatibility_manifest and authority.get(
        "qualification_dimensions"
    ) != list(envelope.qualification_dimensions):
        raise QualifiedRuntimeReleaseError(
            "QRR qualification dimensions differ from the frozen envelope"
        )
    if not legacy_compatibility_manifest and authority.get(
        "qualification_evidence_binding"
    ) != dict(envelope.qualification_evidence_binding):
        raise QualifiedRuntimeReleaseError(
            "QRR qualification evidence differs from the frozen envelope"
        )
    science_authorized = SCIENCE_ROUTE_QUALIFIED in envelope.qualification_dimensions
    expected_lane = (
        "production_science" if science_authorized else "qualification_or_qualprep"
    )
    if not legacy_compatibility_manifest and (
        authority.get("execution_lane") != expected_lane
        or authority.get("production_science_authorized") is not science_authorized
    ):
        raise QualifiedRuntimeReleaseError("QRR execution-lane authority is malformed")
    if science_authorized and envelope.qualification_evidence_binding.get(
        "source_sha256"
    ) != release.source_identity.get("git_commit_sha256"):
        raise QualifiedRuntimeReleaseError(
            "science qualification evidence differs from QRR source identity"
        )


def load_qualified_runtime_release_v1(
    release_root: Path, *, verify_integrity: bool = True
) -> QualifiedRuntimeReleaseV1:
    root = release_root.expanduser().resolve()
    if verify_integrity:
        verify_release_integrity_v1(root)
    try:
        payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualifiedRuntimeReleaseError("QRR manifest is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise QualifiedRuntimeReleaseError("QRR manifest is malformed")
    unhashed = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    if payload.get("manifest_sha256") != canonical_sha256(unhashed):
        raise QualifiedRuntimeReleaseError("QRR manifest SHA-256 differs")
    release = QualifiedRuntimeReleaseV1(
        release_id=str(payload.get("release_id") or ""),
        source_identity=dict(payload.get("source_identity") or {}),
        qualified_execution_envelope=dict(
            payload.get("qualified_execution_envelope") or {}
        ),
        environment_contract=dict(payload.get("environment_contract") or {}),
        qualified_artifacts=tuple(
            dict(row) for row in payload.get("qualified_artifacts", ())
        ),
        python_payload=tuple(dict(row) for row in payload.get("python_payload", ())),
        qualification_authority=dict(payload.get("qualification_authority") or {}),
        template_plan_sha256=str(payload.get("template_plan_sha256") or ""),
        historical_escape_suite_id=str(payload.get("historical_escape_suite_id") or ""),
        historical_escape_suite_inventory_sha256=str(
            payload.get("historical_escape_suite_inventory_sha256") or ""
        ),
        launcher_sha256=str(payload.get("launcher_sha256") or ""),
        created_at_utc=str(payload.get("created_at_utc") or ""),
    )
    _validate_release_semantic_bindings_v1(root, release)
    validate_runtime_environment_v1(release.environment_contract)
    return release


def _exact_route_qualified_preflight_v1(
    preflight: Mapping[str, Any], plan_payload: Mapping[str, Any]
) -> bool:
    """Bind release authority to the actually materialized reference plan."""

    route = preflight.get("execution_route")
    return bool(
        preflight.get("schema_version") == "eiel_campaign_preflight_v1"
        and preflight.get("plan_sha256") == plan_payload.get("plan_sha256")
        and preflight.get("ready_for_execution") is True
        and isinstance(route, Mapping)
        and route.get("authority_state") == "route_qualified"
        and route.get("artifact_conformance_exact") is True
    )


def build_qualified_runtime_release_v1(
    *,
    base_spec_path: Path,
    preflight_receipt_path: Path,
    release_root: Path = DEFAULT_QRR_ROOT,
    source_root: Path | None = None,
    provider_registry: ExecutionProviderRegistryV1 | None = None,
    tick_native_execution_path: Path | None = None,
    runtime_python_executable: Path | None = None,
) -> Path:
    """Build a QRR only from an exact, already route-qualified authority."""

    source = (
        Path(__file__).resolve().parents[2]
        if source_root is None
        else source_root.expanduser().resolve()
    )
    source_identity = _git_source_identity(source)
    spec_path = base_spec_path.expanduser().resolve()
    preflight_path = preflight_receipt_path.expanduser().resolve()
    spec = load_campaign_spec(spec_path)
    plan = compile_campaign(spec)
    plan_payload = plan.payload()
    tick_reference: dict[str, Any] | None = None
    if tick_native_execution_path is not None:
        reference_path = tick_native_execution_path.expanduser().resolve()
        try:
            raw_reference = json.loads(reference_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise QualifiedRuntimeReleaseError(
                "tick-native reference execution is unreadable"
            ) from exc
        if not isinstance(raw_reference, Mapping):
            raise QualifiedRuntimeReleaseError(
                "tick-native reference execution must be an object"
            )
        try:
            tick_request = TickNativeExecutionRequestV1(
                total_ticks=raw_reference.get(
                    "total_ticks", raw_reference.get("absolute_ticks")
                ),
                checkpoint_ticks=(
                    tuple(raw_reference["checkpoint_ticks"])
                    if isinstance(raw_reference.get("checkpoint_ticks"), list)
                    else ()
                ),
                checkpoint_every_ticks=raw_reference.get(
                    "checkpoint_every_ticks",
                    raw_reference.get("checkpoint_interval_absolute_ticks"),
                ),
                deliberate_restore_tick=raw_reference.get(
                    "deliberate_restore_tick",
                    raw_reference.get("intentional_restore_tick"),
                ),
                deliberate_restore_ticks=(
                    tuple(raw_reference["deliberate_restore_ticks"])
                    if isinstance(raw_reference.get("deliberate_restore_ticks"), list)
                    else ()
                ),
            )
        except TickNativeError as exc:
            raise QualifiedRuntimeReleaseError(
                f"tick-native reference execution is invalid: {exc}"
            ) from exc
        plan_payload = canonical_tick_native_plan_v1(
            plan_payload, tick_request=tick_request
        )
        tick_reference = tick_request.payload()
    profile = resolve_campaign_profile(spec)
    providers = provider_registry or default_execution_provider_registry()
    provider = providers.get(plan.profile_id)
    if provider is None:
        raise QualifiedRuntimeReleaseError("qualified profile has no provider")
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualifiedRuntimeReleaseError(
            "qualification preflight is unreadable"
        ) from exc
    if not isinstance(preflight, Mapping):
        raise QualifiedRuntimeReleaseError("qualification preflight is malformed")
    route = preflight.get("execution_route")
    if not _exact_route_qualified_preflight_v1(preflight, plan_payload):
        raise QualifiedRuntimeReleaseError(
            "QRR build requires an exact route-qualified execution preflight"
        )
    assert isinstance(route, Mapping)
    if SCIENCE_ROUTE_QUALIFIED in route.get("qualification_dimensions", ()):
        receipt_path_raw = route.get("qualification_receipt_path")
        receipt_sha256 = route.get("qualification_receipt_sha256")
        if not isinstance(receipt_path_raw, str) or not isinstance(receipt_sha256, str):
            raise QualifiedRuntimeReleaseError(
                "science QRR build requires the immutable route evidence receipt"
            )
        receipt_path = Path(receipt_path_raw).expanduser().resolve()
        if (
            not receipt_path.is_file()
            or receipt_path.is_symlink()
            or _file_sha256(receipt_path) != receipt_sha256
        ):
            raise QualifiedRuntimeReleaseError(
                "science route evidence receipt binding differs"
            )
        try:
            route_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise QualifiedRuntimeReleaseError(
                "science route evidence receipt is unreadable"
            ) from exc
        contract_owner = getattr(provider, "execution_route_contract", None)
        if not callable(contract_owner):
            raise QualifiedRuntimeReleaseError(
                "science QRR provider has no execution-route contract"
            )
        contract = contract_owner(plan_payload)
        normalized_route = execution_route_qualification_status_v1(
            contract, route_receipt
        )
        if (
            normalized_route.get("authority_state") != "route_qualified"
            or normalized_route.get("qualification_dimensions")
            != route.get("qualification_dimensions")
            or normalized_route.get("qualification_evidence_binding")
            != route.get("qualification_evidence_binding")
        ):
            raise QualifiedRuntimeReleaseError(
                "science QRR preflight authority is not reproduced by its evidence receipt"
            )
    envelope = build_qualified_execution_envelope_v1(
        plan_payload,
        provider=provider,
        profile=profile,
        qualification_authority=route,
    )
    if route.get("execution_route_sha256") != envelope.execution_route_sha256:
        raise QualifiedRuntimeReleaseError(
            "qualification preflight route differs from current source closure"
        )
    envelope.validate_plan(plan_payload, provider=provider)
    source_paths = discover_python_payload_paths_v1(source)
    tooling_root = Path(__file__).resolve().parents[2]
    payload_sources: dict[Path, tuple[Path, str]] = {
        path.relative_to(source): (path, "qualified_runtime_source")
        for path in source_paths
    }
    payload_sources.update(
        {
            path.relative_to(source): (path, "qualified_runtime_data")
            for path in discover_runtime_data_payload_paths_v1(source)
        }
    )
    for path in sorted((tooling_root / "engine/qrr").glob("*.py")):
        if path.is_file():
            payload_sources[path.relative_to(tooling_root)] = (
                path.resolve(),
                "qrr_release_tooling",
            )
    python_rows = tuple(
        {
            "path": relative.as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
            "source_role": role,
        }
        for relative, (path, role) in sorted(payload_sources.items())
    )
    artifact_rows = _verified_artifact_rows(spec)
    authority = {
        "preflight_schema_version": preflight.get("schema_version"),
        "preflight_receipt_sha256": _file_sha256(preflight_path),
        "plan_sha256": plan.plan_sha256,
        "qualification_key_sha256": plan.qualification_key_sha256,
        "execution_route": dict(route),
        "qualification_reuse": preflight.get("qualification_reuse"),
        "artifact_location_policy": (
            "qualified_absolute_paths_route_bound_with_frozen_payload_copies_v1"
        ),
        "artifact_relocation_requires_delta_qualification": True,
        "time_authority": (
            "absolute_tick_v1"
            if tick_reference is not None
            else "legacy_generation_route_v1"
        ),
        "base_spec_time_role": (
            "internal_compatibility_only"
            if tick_reference is not None
            else "legacy_public_authority"
        ),
    }
    preflight_resources = preflight.get("resources")
    sealed_resource_promotion = (
        preflight_resources.get("allocation_diagnostic_resource_promotion")
        if isinstance(preflight_resources, Mapping)
        else None
    )
    if isinstance(sealed_resource_promotion, Mapping):
        authority["sealed_allocation_resource_promotion_sha256"] = canonical_sha256(
            sealed_resource_promotion
        )
    source_identity_payload = {
        **source_identity,
        "python_payload_sha256": canonical_sha256({"files": list(python_rows)}),
    }
    science_authorized = SCIENCE_ROUTE_QUALIFIED in envelope.qualification_dimensions
    if science_authorized and envelope.qualification_evidence_binding.get(
        "source_sha256"
    ) != source_identity.get("git_commit_sha256"):
        raise QualifiedRuntimeReleaseError(
            "science qualification evidence does not bind the QRR source commit"
        )
    authority.update(
        {
            "qualification_dimensions": list(envelope.qualification_dimensions),
            "qualification_evidence_binding": dict(
                envelope.qualification_evidence_binding
            ),
            "execution_lane": (
                "production_science"
                if science_authorized
                else "qualification_or_qualprep"
            ),
            "production_science_authorized": science_authorized,
        }
    )
    environment_contract = stable_runtime_environment_fingerprint_v1(
        runtime_python_executable
    )
    historical_inventory = historical_escape_inventory_v1()
    historical_suite_id = str(historical_inventory["suite_id"])
    historical_inventory_sha256 = str(historical_inventory["inventory_sha256"])
    launcher_text = _launcher_text(str(environment_contract["python_executable"]))
    launcher_sha256 = hashlib.sha256(launcher_text.encode("utf-8")).hexdigest()
    identity_payload = {
        "schema_version": QUALIFIED_RUNTIME_RELEASE_SCHEMA_VERSION,
        "source_identity": source_identity_payload,
        "qualified_execution_envelope": envelope.payload(),
        "environment_contract": environment_contract,
        "qualified_artifacts": [dict(row) for row in artifact_rows],
        "python_payload": [dict(row) for row in python_rows],
        "qualification_authority": authority,
        "template_plan_sha256": plan.plan_sha256,
        "historical_escape_suite_id": historical_suite_id,
        "historical_escape_suite_inventory_sha256": historical_inventory_sha256,
        "launcher_sha256": launcher_sha256,
    }
    release_id = f"qrr1-{canonical_sha256(identity_payload)[:32]}"
    release = QualifiedRuntimeReleaseV1(
        release_id=release_id,
        source_identity=source_identity_payload,
        qualified_execution_envelope=envelope.payload(),
        environment_contract=environment_contract,
        qualified_artifacts=artifact_rows,
        python_payload=python_rows,
        qualification_authority=authority,
        template_plan_sha256=plan.plan_sha256,
        historical_escape_suite_id=historical_suite_id,
        historical_escape_suite_inventory_sha256=historical_inventory_sha256,
        launcher_sha256=launcher_sha256,
        created_at_utc=_now(),
    )
    destination = release_root.expanduser().resolve() / release_id
    if destination.exists():
        existing = load_qualified_runtime_release_v1(destination)
        if existing.identity_payload() != release.identity_payload():
            raise QualifiedRuntimeReleaseError(
                "existing release ID has different immutable content"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{release_id}.{uuid4().hex}.tmp"
    staging.mkdir(mode=0o700)
    try:
        for relative, (path, _role) in sorted(payload_sources.items()):
            _copy_payload_file(
                path,
                staging / "payload/python" / relative,
            )
        for row in artifact_rows:
            _copy_payload_file(
                Path(str(row["qualified_path"])), staging / str(row["payload_path"])
            )
        _copy_payload_file(spec_path, staging / "payload/spec/base_spec.yaml")
        _atomic_json(staging / "payload/spec/execution_plan.json", plan_payload)
        if tick_reference is not None:
            _atomic_json(
                staging / "payload/spec/tick_native_reference_execution.json",
                tick_reference,
            )
        _copy_payload_file(
            preflight_path, staging / "payload/receipts/preflight_receipt.json"
        )
        _atomic_json(staging / "manifest.json", release.payload())
        _atomic_json(
            staging / "payload/receipts/release_creation_receipt.json",
            {
                "schema_version": "eiel_qrr_creation_receipt_v1",
                "release_id": release.release_id,
                "manifest_sha256": release.payload()["manifest_sha256"],
                "source_identity": dict(release.source_identity),
                "execution_route_sha256": envelope.execution_route_sha256,
                "historical_escape_suite_inventory_sha256": (
                    historical_inventory_sha256
                ),
                "created_at_utc": release.created_at_utc,
            },
        )
        launcher = staging / "bin/qrr"
        launcher.parent.mkdir(parents=True, exist_ok=True)
        launcher.write_text(launcher_text, encoding="utf-8")
        launcher.chmod(0o755)
        _write_integrity_manifest(staging)
        verify_release_integrity_v1(staging)
        _seal_release_permissions(staging)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


__all__ = [
    "DEFAULT_QRR_ROOT",
    "DEFAULT_QRR_RUN_ROOT",
    "QUALIFIED_RUNTIME_RELEASE_SCHEMA_VERSION",
    "QualifiedRuntimeReleaseError",
    "QualifiedRuntimeReleaseV1",
    "build_qualified_runtime_release_v1",
    "discover_python_payload_paths_v1",
    "discover_runtime_data_payload_paths_v1",
    "load_qualified_runtime_release_v1",
    "runtime_environment_fingerprint_v1",
    "stable_python_executable_v1",
    "stable_runtime_environment_fingerprint_v1",
    "validate_runtime_environment_v1",
    "verify_release_integrity_v1",
]
