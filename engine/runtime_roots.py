"""Deterministic, side-effect-free runtime and artifact root resolution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final


RUNTIME_ROOT_ENV: Final[str] = "EIEL_RUNTIME_ROOT"
ARTIFACT_ROOT_ENV: Final[str] = "EIEL_ARTIFACT_ROOT"
DEFAULT_RUNTIME_ROOT: Final[Path] = Path.home() / ".local" / "share" / "emergence"
DEFAULT_ARTIFACT_DIRECTORY: Final[str] = "artifacts"


class RuntimeRootConfigurationError(ValueError):
    """A configured runtime or artifact root is malformed."""


@dataclass(frozen=True, slots=True)
class RuntimeRoots:
    runtime_root: Path
    artifact_root: Path
    runtime_source: str
    artifact_source: str

    def as_dict(self) -> dict[str, str]:
        return {
            "runtime_root": str(self.runtime_root),
            "artifact_root": str(self.artifact_root),
            "runtime_source": self.runtime_source,
            "artifact_source": self.artifact_source,
        }


def _configured_path(value: str | Path, *, field: str) -> Path:
    rendered = os.fspath(value)
    if not rendered.strip():
        raise RuntimeRootConfigurationError(f"{field} must not be empty")
    if any(character in rendered for character in "\0\r\n"):
        raise RuntimeRootConfigurationError(
            f"{field} must not contain NUL or line breaks"
        )
    path = Path(rendered).expanduser()
    if not path.is_absolute():
        raise RuntimeRootConfigurationError(f"{field} must be absolute")
    return path.resolve(strict=False)


def resolve_runtime_roots(
    *,
    runtime_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> RuntimeRoots:
    """Resolve explicit values, then EIEL environment values, then defaults.

    The resolver performs no directory creation and does not require either root
    to exist. When no artifact root is supplied, artifacts live below the
    resolved runtime root rather than below a separate fixed path.
    """

    environment = os.environ if environ is None else environ
    if runtime_root is not None:
        runtime_value = runtime_root
        runtime_source = "explicit"
    elif environment.get(RUNTIME_ROOT_ENV, "").strip():
        runtime_value = environment[RUNTIME_ROOT_ENV]
        runtime_source = f"environment:{RUNTIME_ROOT_ENV}"
    else:
        runtime_value = DEFAULT_RUNTIME_ROOT
        runtime_source = "default"
    resolved_runtime = _configured_path(runtime_value, field="runtime_root")

    if artifact_root is not None:
        artifact_value = artifact_root
        artifact_source = "explicit"
    elif environment.get(ARTIFACT_ROOT_ENV, "").strip():
        artifact_value = environment[ARTIFACT_ROOT_ENV]
        artifact_source = f"environment:{ARTIFACT_ROOT_ENV}"
    else:
        artifact_value = resolved_runtime / DEFAULT_ARTIFACT_DIRECTORY
        artifact_source = "runtime_root_default"

    return RuntimeRoots(
        runtime_root=resolved_runtime,
        artifact_root=_configured_path(artifact_value, field="artifact_root"),
        runtime_source=runtime_source,
        artifact_source=artifact_source,
    )


def public_runtime_path(relative: str = "") -> str:
    """Locate a runtime-owned path without creating it.

    Existing consumers that accept strings use this adapter to the same root
    resolver. Callers must set EIEL_RUNTIME_ROOT before importing consumers
    that cache their defaults. Historical receipts are never synthesized.
    """
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise RuntimeRootConfigurationError("runtime child must be relative and confined")
    root = resolve_runtime_roots().runtime_root
    value = str(root / part)
    return value + "/" if relative.endswith("/") else value


__all__ = [
    "ARTIFACT_ROOT_ENV",
    "DEFAULT_ARTIFACT_DIRECTORY",
    "DEFAULT_RUNTIME_ROOT",
    "RUNTIME_ROOT_ENV",
    "RuntimeRootConfigurationError",
    "RuntimeRoots",
    "resolve_runtime_roots",
    "public_runtime_path",
]
