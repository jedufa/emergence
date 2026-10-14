"""Trust policy helpers for checkpoint and replay artifact loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointTrustPolicy:
    """Resolved trust decision for a checkpoint artifact load."""

    artifact_path: Path
    trusted: bool
    loader: str
    reason: str


class UntrustedCheckpointArtifactError(PermissionError):
    """Raised when unsafe checkpoint deserialization is attempted untrusted."""


def checkpoint_trust_policy(path: str | Path, *, trusted: bool, loader: str, reason: str = "") -> CheckpointTrustPolicy:
    """Return a deterministic trust decision for a checkpoint artifact."""

    return CheckpointTrustPolicy(
        artifact_path=Path(path).expanduser().resolve(),
        trusted=bool(trusted),
        loader=str(loader),
        reason=str(reason),
    )


def require_trusted_checkpoint_artifact(path: str | Path, *, trusted: bool, loader: str, reason: str = "") -> CheckpointTrustPolicy:
    """Require explicit trust before crossing an unsafe deserialization boundary."""

    policy = checkpoint_trust_policy(path, trusted=trusted, loader=loader, reason=reason)
    if not policy.trusted:
        raise UntrustedCheckpointArtifactError(
            f"{policy.loader} requires trusted=True for checkpoint artifact loading: {policy.artifact_path}"
        )
    return policy
