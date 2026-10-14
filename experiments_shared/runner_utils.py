"""Project-local runner utilities for Emergence adapter entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional


def resolve_registry_path(*, project_root: Path, raw: str, spec_subdir: str = "_spec/registry") -> Path:
    """Resolve a registry path against the project's canonical registry tree."""

    text = str(raw or "").strip()
    if not text:
        raise ValueError("registry path is required")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate.resolve()
    resolved = (Path(project_root).resolve() / spec_subdir / candidate).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"registry not found under {spec_subdir}: {text}")
    return resolved


def default_experiment_data_root() -> Path:
    """Return the shared heavy-data root for experiment projects."""

    raw = str(os.environ.get("CAUSYL_EXPERIMENT_DATA_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / "causyl_experiment_data").expanduser().resolve()


def default_project_data_root(project_id: str) -> Path:
    """Return the heavy-data root for one project."""

    project = str(project_id or "").strip().lower()
    if not project:
        raise ValueError("project_id is required")
    return (default_experiment_data_root() / project).resolve()


def default_project_runs_root(project_id: str, *, external_data_root: Optional[Path] = None) -> Path:
    """Return the default durable runs root for one project."""

    base = external_data_root.resolve() if external_data_root is not None else default_project_data_root(project_id)
    return (base / "runs").resolve()


def normalize_project_runs_root(
    *,
    repo_root: Path,
    project_root: Path,
    project_id: str,
    raw: str,
    external_data_root: Optional[Path] = None,
) -> Path:
    """Normalize absolute, repo-relative, and project-relative run roots."""

    text = str(raw or "").strip()
    if not text:
        return default_project_runs_root(project_id, external_data_root=external_data_root)

    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    repo = Path(repo_root).resolve()
    project = Path(project_root).resolve()
    if text.startswith("experiments/projects/"):
        return (repo / candidate).resolve()
    return (project / candidate).resolve()


def ensure_project_output_env(
    *,
    project_id: str,
    repo_root: Path,
    project_root: Path,
    runs_root: str = "",
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Return normalized project output environment variables.

    This archived-project implementation intentionally avoids contract lookups
    outside the Emergence tree. It preserves the environment names consumed by
    the existing registry harness.
    """

    env_out = dict(os.environ)
    if env is not None:
        env_out.update({str(key): str(value) for key, value in dict(env).items()})

    data_root = default_project_data_root(project_id)
    normalized_runs_root = normalize_project_runs_root(
        repo_root=repo_root,
        project_root=project_root,
        project_id=project_id,
        raw=runs_root,
        external_data_root=data_root,
    )
    env_out.setdefault("EXPERIMENT_OUTPUTS_DIR", str(data_root))
    if str(project_id or "").strip().lower() == "emergence":
        env_out.setdefault("EMERGENCE_OUTPUTS_DIR", str(data_root))
        env_out.setdefault("EMERGENCE_RUNS_DIR", str(normalized_runs_root))
    return env_out
