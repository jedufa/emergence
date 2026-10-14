from __future__ import annotations

from pathlib import Path

import pytest

from engine.runtime_roots import (
    ARTIFACT_ROOT_ENV,
    DEFAULT_RUNTIME_ROOT,
    RUNTIME_ROOT_ENV,
    RuntimeRootConfigurationError,
    resolve_runtime_roots,
)


def test_defaults_keep_artifacts_below_runtime_root() -> None:
    roots = resolve_runtime_roots(environ={})

    assert roots.runtime_root == DEFAULT_RUNTIME_ROOT
    assert roots.artifact_root == DEFAULT_RUNTIME_ROOT / "artifacts"
    assert roots.runtime_source == "default"
    assert roots.artifact_source == "runtime_root_default"


def test_environment_configures_both_roots(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    artifact = tmp_path / "artifact-store"

    roots = resolve_runtime_roots(
        environ={RUNTIME_ROOT_ENV: str(runtime), ARTIFACT_ROOT_ENV: str(artifact)}
    )

    assert roots.runtime_root == runtime
    assert roots.artifact_root == artifact
    assert roots.runtime_source == f"environment:{RUNTIME_ROOT_ENV}"
    assert roots.artifact_source == f"environment:{ARTIFACT_ROOT_ENV}"


def test_explicit_roots_override_environment(tmp_path: Path) -> None:
    explicit_runtime = tmp_path / "explicit-runtime"
    explicit_artifact = tmp_path / "explicit-artifacts"

    roots = resolve_runtime_roots(
        runtime_root=explicit_runtime,
        artifact_root=explicit_artifact,
        environ={
            RUNTIME_ROOT_ENV: str(tmp_path / "env-runtime"),
            ARTIFACT_ROOT_ENV: str(tmp_path / "env-artifacts"),
        },
    )

    assert roots.runtime_root == explicit_runtime
    assert roots.artifact_root == explicit_artifact
    assert roots.runtime_source == "explicit"
    assert roots.artifact_source == "explicit"


def test_artifact_default_follows_configured_runtime(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"

    roots = resolve_runtime_roots(runtime_root=runtime, environ={})

    assert roots.artifact_root == runtime / "artifacts"


@pytest.mark.parametrize(
    "value", ("", "  ", "relative/path", "bad\npath", "bad\0path")
)
def test_explicit_malformed_root_is_rejected(value: str) -> None:
    with pytest.raises(RuntimeRootConfigurationError):
        resolve_runtime_roots(runtime_root=value, environ={})


def test_resolution_has_no_filesystem_side_effect(tmp_path: Path) -> None:
    runtime = tmp_path / "not-created"
    artifact = tmp_path / "also-not-created"

    roots = resolve_runtime_roots(
        runtime_root=runtime, artifact_root=artifact, environ={}
    )

    assert roots.as_dict()["runtime_root"] == str(runtime)
    assert not runtime.exists()
    assert not artifact.exists()
