from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.checkpoint import checkpoint_load
from simulation.checkpoint_trust import (
    UntrustedCheckpointArtifactError,
    checkpoint_trust_policy,
    require_trusted_checkpoint_artifact,
)


def test_checkpoint_trust_policy_is_deterministic(tmp_path: Path) -> None:
    artifact = tmp_path / "checkpoint"
    first = checkpoint_trust_policy(artifact, trusted=True, loader="test-loader", reason="local replay")
    second = checkpoint_trust_policy(artifact, trusted=True, loader="test-loader", reason="local replay")

    assert first == second
    assert first.trusted is True
    assert first.loader == "test-loader"


def test_untrusted_checkpoint_artifact_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(UntrustedCheckpointArtifactError, match="requires trusted=True"):
        require_trusted_checkpoint_artifact(tmp_path, trusted=False, loader="legacy loader")


def test_legacy_v1_checkpoint_requires_explicit_trust_even_with_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = tmp_path / "legacy_v1"
    ckpt.mkdir()
    (ckpt / "meta.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    monkeypatch.setenv("EIEL_ALLOW_CHECKPOINT_V1", "1")

    with pytest.raises(UntrustedCheckpointArtifactError, match="checkpoint v1 legacy loader"):
        checkpoint_load(str(ckpt))
