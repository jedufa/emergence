"""Bind a compiled campaign plan to one immutable cross-path candidate chain."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Final

from engine.native_runtime.inheritance_cross_path_qualified_artifact_v1 import (
    CANDIDATE_AUTHORITY_SCHEMA_VERSION,
    UPSTREAM_BINDING_IDS,
    InheritanceCrossPathCandidateArtifactLoaderV1,
)

from .canonical import canonical_bytes, canonical_sha256
from .inheritance_cross_path_candidate_staging_v1 import (
    CANDIDATE_BINDING_MANIFEST_SCHEMA_VERSION,
)
from .production_engine_provider import PROFILE_ID, PROVIDER_ID


CANDIDATE_PLAN_BINDING_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_candidate_plan_binding_v1"
)
_STORAGE_MODES: Final[frozenset[str]] = frozenset(
    {"exact_full", "compact_complete"}
)


class InheritanceCrossPathCandidatePlanError(ValueError):
    """A candidate binding or source plan cannot produce an exact plan."""


def _fail(message: str) -> InheritanceCrossPathCandidatePlanError:
    return InheritanceCrossPathCandidatePlanError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_copy(value: Any, *, field: str) -> Any:
    try:
        return json.loads(canonical_bytes(value).decode("ascii"))
    except ValueError as exc:
        raise _fail(f"{field} must contain canonical JSON data") from exc


def _binding_path(value: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise _fail("candidate binding manifest path must be absolute")
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise _fail("candidate binding manifest path must not contain symlinks")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise _fail(f"candidate binding manifest is unavailable: {exc}") from exc
    if not resolved.is_file():
        raise _fail("candidate binding manifest must be a regular file")
    return resolved


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise _fail(f"candidate binding manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise _fail("candidate binding manifest must be a JSON object")
    return dict(payload)


@dataclass(frozen=True, slots=True)
class InheritanceCrossPathCandidateBindingV1:
    """One validated non-production candidate binding and its byte identity."""

    path: Path
    binding_manifest_sha256: str
    candidate_authority_sha256: str
    native_abi: str
    storage_mode: str
    artifacts: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        if (
            not self.path.is_absolute()
            or len(self.binding_manifest_sha256) != 64
            or len(self.candidate_authority_sha256) != 64
            or not self.native_abi
            or self.storage_mode not in _STORAGE_MODES
            or set(self.artifacts) != set(UPSTREAM_BINDING_IDS)
        ):
            raise _fail("candidate binding value is incomplete")


@dataclass(frozen=True, slots=True)
class InheritanceCrossPathCandidatePlanV1:
    """A four-artifact plan plus identities kept separate from one another."""

    plan: Mapping[str, Any]
    plan_sha256: str
    binding_manifest_sha256: str
    candidate_authority_sha256: str
    storage_mode: str

    def __post_init__(self) -> None:
        if (
            self.plan.get("plan_sha256") != self.plan_sha256
            or len(self.plan_sha256) != 64
            or len(self.binding_manifest_sha256) != 64
            or len(self.candidate_authority_sha256) != 64
            or self.storage_mode not in _STORAGE_MODES
        ):
            raise _fail("candidate plan identity is incomplete")
        unsigned = dict(self.plan)
        unsigned.pop("plan_sha256", None)
        if canonical_sha256(unsigned) != self.plan_sha256:
            raise _fail("candidate plan digest is stale")


def load_inheritance_cross_path_candidate_binding_v1(
    manifest_path: Path,
) -> InheritanceCrossPathCandidateBindingV1:
    """Reject activation of a retired standalone candidate binding."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "candidate binding activation is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    path = _binding_path(manifest_path)
    payload = _read_manifest(path)
    expected_fields = {
        "schema_version",
        "candidate_authority_schema_version",
        "authorization_scope",
        "production_authority",
        "provider_id",
        "profile_id",
        "native_abi",
        "candidate_authority_sha256",
        "inputs",
        "artifacts",
        "validation",
    }
    artifacts = payload.get("artifacts")
    native_abi = payload.get("native_abi")
    if (
        set(payload) != expected_fields
        or payload.get("schema_version")
        != CANDIDATE_BINDING_MANIFEST_SCHEMA_VERSION
        or payload.get("candidate_authority_schema_version")
        != CANDIDATE_AUTHORITY_SCHEMA_VERSION
        or payload.get("authorization_scope") != "mechanic_isolation_gate_only"
        or payload.get("production_authority") is not False
        or payload.get("provider_id") != PROVIDER_ID
        or payload.get("profile_id") != PROFILE_ID
        or not isinstance(native_abi, str)
        or not native_abi
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != set(UPSTREAM_BINDING_IDS)
    ):
        raise _fail("candidate binding manifest identity is invalid")

    loader = InheritanceCrossPathCandidateArtifactLoaderV1(
        artifacts,
        provider_id=PROVIDER_ID,
        profile_id=PROFILE_ID,
        expected_native_abi=native_abi,
    )
    validated = loader.validate_candidate()
    if payload.get("candidate_authority_sha256") != (
        loader.candidate_authority_sha256
    ):
        raise _fail("candidate binding authority digest is stale")

    inputs = payload.get("inputs")
    envelope = (
        inputs.get("execution_envelope")
        if isinstance(inputs, Mapping)
        else None
    )
    capacity = (
        envelope.get("reproduction_exposure_capacity")
        if isinstance(envelope, Mapping)
        else None
    )
    registry_capacity = validated["registry"].get(
        "reproduction_exposure_capacity"
    )
    projection = validated["telemetry_contract"].get("projection_contract")
    projection_envelope = (
        projection.get("envelope")
        if isinstance(projection, Mapping)
        else None
    )
    storage_mode = capacity.get("storage_mode") if isinstance(capacity, Mapping) else None
    if (
        storage_mode not in _STORAGE_MODES
        or not isinstance(registry_capacity, Mapping)
        or dict(registry_capacity) != dict(capacity)
        or not isinstance(projection_envelope, Mapping)
        or projection_envelope.get("storage_mode") != storage_mode
    ):
        raise _fail("candidate binding storage authority is inconsistent")

    rows = {
        artifact_id: {
            "path": str(Path(str(artifacts[artifact_id]["path"])).resolve()),
            "sha256": str(artifacts[artifact_id]["sha256"]),
            "abi": str(artifacts[artifact_id]["abi"]),
        }
        for artifact_id in UPSTREAM_BINDING_IDS
    }
    return InheritanceCrossPathCandidateBindingV1(
        path=path,
        binding_manifest_sha256=_sha256(path),
        candidate_authority_sha256=loader.candidate_authority_sha256,
        native_abi=native_abi,
        storage_mode=storage_mode,
        artifacts=rows,
    )


def compose_inheritance_cross_path_candidate_plan_v1(
    source_plan: Mapping[str, Any],
    binding: InheritanceCrossPathCandidateBindingV1,
) -> InheritanceCrossPathCandidatePlanV1:
    """Reject composition of a retired standalone candidate plan."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "candidate plan composition is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    if not isinstance(source_plan, Mapping):
        raise _fail("source_plan must be a mapping")
    if not isinstance(binding, InheritanceCrossPathCandidateBindingV1):
        raise _fail("binding must be a validated candidate binding v1")
    plan = _canonical_copy(source_plan, field="source_plan")
    declared_plan_sha256 = plan.pop("plan_sha256", None)
    if (
        not isinstance(declared_plan_sha256, str)
        or canonical_sha256(plan) != declared_plan_sha256
    ):
        raise _fail("source plan digest is stale")
    if plan.get("resolved_profile") != PROFILE_ID:
        raise _fail("source plan uses another runtime profile")
    qualification = plan.get("qualification_key")
    if not isinstance(qualification, Mapping):
        raise _fail("source plan lacks a qualification key")
    qualification = dict(qualification)
    if not isinstance(qualification.get("artifact_hashes"), Mapping):
        raise _fail("source plan qualification key lacks artifact hashes")

    artifacts = {
        artifact_id: dict(binding.artifacts[artifact_id])
        for artifact_id in UPSTREAM_BINDING_IDS
    }
    qualification["artifact_hashes"] = {
        artifact_id: artifacts[artifact_id]["sha256"]
        for artifact_id in UPSTREAM_BINDING_IDS
    }
    plan["qualification_key"] = qualification
    plan["qualification_key_sha256"] = canonical_sha256(qualification)
    plan["artifacts"] = artifacts
    plan["candidate_binding"] = {
        "schema_version": CANDIDATE_PLAN_BINDING_SCHEMA_VERSION,
        "authorization_scope": "mechanic_isolation_gate_only",
        "production_authority": False,
        "binding_manifest_path": str(binding.path),
        "binding_manifest_sha256": binding.binding_manifest_sha256,
        "candidate_authority_sha256": binding.candidate_authority_sha256,
        "storage_mode": binding.storage_mode,
    }
    plan_sha256 = canonical_sha256(plan)
    plan["plan_sha256"] = plan_sha256
    return InheritanceCrossPathCandidatePlanV1(
        plan=plan,
        plan_sha256=plan_sha256,
        binding_manifest_sha256=binding.binding_manifest_sha256,
        candidate_authority_sha256=binding.candidate_authority_sha256,
        storage_mode=binding.storage_mode,
    )


def bind_inheritance_cross_path_candidate_plan_v1(
    source_plan: Mapping[str, Any],
    manifest_path: Path,
) -> InheritanceCrossPathCandidatePlanV1:
    """Reject binding of a retired standalone candidate plan."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "candidate plan binding is disabled"
    )


__all__ = [
    "CANDIDATE_PLAN_BINDING_SCHEMA_VERSION",
    "InheritanceCrossPathCandidateBindingV1",
    "InheritanceCrossPathCandidatePlanError",
    "InheritanceCrossPathCandidatePlanV1",
    "bind_inheritance_cross_path_candidate_plan_v1",
    "compose_inheritance_cross_path_candidate_plan_v1",
    "load_inheritance_cross_path_candidate_binding_v1",
]
