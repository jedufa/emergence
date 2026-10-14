from __future__ import annotations

"""Deterministic branch identity for Architecture B checkpoint continuations."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Final


CHECKPOINT_BRANCH_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_checkpoint_branch_v1"
)
BRANCH_IDENTITY_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_branch_identity_v1"
)
RNG_LINEAGE_PROVENANCE_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_rng_lineage_provenance_v1"
)
BRANCH_ID_DOMAIN: Final[bytes] = b"EIEL\0GPU_BRANCH\0V1\0"
BRANCH_ID_PREFIX: Final[str] = "eiel-gpu-b1-"
RNG_INHERITANCE_MODE: Final[str] = "exact_checkpoint_state"

_BRANCH_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[a-z0-9][a-z0-9._-]{0,127}"
)
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_TRAJECTORY_EXCLUDED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "branch_id",
        "branch_identity",
        "branch_identity_sha256",
        "branch_key",
        "branch_provenance",
        "provenance",
        "rng_lineage_provenance",
    }
)
_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "branch_key",
        "branch_id",
        "branch_identity_sha256",
        "branch_identity",
        "trajectory_projection",
        "rng_lineage_provenance",
    }
)
_IDENTITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "branch_key",
        "source_checkpoint",
        "source_lineage",
        "trajectory_projection",
    }
)
_SOURCE_CHECKPOINT_FIELDS: Final[frozenset[str]] = frozenset(
    {"generation", "projection_sha256"}
)
_SOURCE_LINEAGE_FIELDS: Final[frozenset[str]] = frozenset(
    {"branch_id", "rng_state_sha256"}
)
_RNG_PROVENANCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "branch_key",
        "branch_id",
        "source_branch_id",
        "source_checkpoint_projection_sha256",
        "inheritance_mode",
        "source_rng_state_sha256",
        "inherited_rng_state_sha256",
        "reseeded",
        "seed_derivation",
    }
)


class CudaCheckpointBranchError(ValueError):
    """Raised when a branch identity or provenance record is invalid."""


class CudaCheckpointBranchCollisionError(CudaCheckpointBranchError):
    """Raised when one branch ID resolves to non-identical valid records."""


def _json_value(value: Any, *, path: str) -> Any:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise CudaCheckpointBranchError(
                    f"{path} contains a non-string JSON object key"
                )
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise CudaCheckpointBranchError(
        f"{path} contains a non-JSON value of type {type(value).__name__}"
    )


def canonical_branch_json_bytes(value: Any) -> bytes:
    """Encode strict canonical JSON used by branch identity and provenance."""

    prepared = _json_value(value, path="value")
    try:
        encoded = json.dumps(
            prepared,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CudaCheckpointBranchError(
            f"value is not valid canonical branch JSON: {exc}"
        ) from exc
    return encoded.encode("utf-8")


def _canonical_object(value: Any, *, path: str) -> tuple[dict[str, Any], bytes]:
    if not isinstance(value, Mapping):
        raise CudaCheckpointBranchError(f"{path} must be a JSON object")
    raw = canonical_branch_json_bytes(value)
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):  # pragma: no cover - guarded by Mapping
        raise AssertionError("canonical JSON object decoded to a non-object")
    return decoded, raw


def validate_branch_key(value: Any) -> str:
    """Return an unmodified, stable ASCII branch key or fail closed."""

    if type(value) is not str or _BRANCH_KEY_PATTERN.fullmatch(value) is None:
        raise CudaCheckpointBranchError(
            "branch_key must be 1-128 lowercase ASCII letters, digits, '.', '_', "
            "or '-', and must start with a letter or digit"
        )
    return value


def _strict_sha256(value: Any, *, path: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise CudaCheckpointBranchError(
            f"{path} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _generation(value: Any, *, path: str) -> int:
    if type(value) is not int or value < 0:
        raise CudaCheckpointBranchError(f"{path} must be an exact integer >= 0")
    return value


def _source_branch_id(value: Any) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or value.strip() != value
        or not value.isascii()
        or not value.isprintable()
    ):
        raise CudaCheckpointBranchError(
            "source_branch_id must be null or a non-empty stable ASCII identifier"
        )
    return value


def _trajectory_projection(value: Any) -> tuple[dict[str, Any], bytes]:
    projection, _ = _canonical_object(value, path="trajectory_projection")
    projected = {
        key: item
        for key, item in projection.items()
        if key not in _TRAJECTORY_EXCLUDED_FIELDS
    }
    return _canonical_object(projected, path="trajectory_projection")


def checkpoint_branch_identity_sha256(identity: Mapping[str, Any]) -> str:
    """Return the domain-separated digest of a canonical branch identity."""

    return hashlib.sha256(
        BRANCH_ID_DOMAIN + canonical_branch_json_bytes(identity)
    ).hexdigest()


def checkpoint_branch_id(identity: Mapping[str, Any]) -> str:
    """Return the stable public ID for a canonical branch identity."""

    return BRANCH_ID_PREFIX + checkpoint_branch_identity_sha256(identity)


@dataclass(frozen=True)
class CudaCheckpointBranch:
    """Immutable JSON-backed checkpoint branch record."""

    branch_key: str
    branch_id: str
    branch_identity_sha256: str
    _branch_identity_json: bytes = field(repr=False)
    _trajectory_projection_json: bytes = field(repr=False)
    _rng_lineage_provenance_json: bytes = field(repr=False)

    @property
    def branch_identity(self) -> dict[str, Any]:
        return json.loads(self._branch_identity_json)

    @property
    def trajectory_projection(self) -> dict[str, Any]:
        return json.loads(self._trajectory_projection_json)

    @property
    def rng_lineage_provenance(self) -> dict[str, Any]:
        return json.loads(self._rng_lineage_provenance_json)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_BRANCH_SCHEMA_VERSION,
            "branch_key": self.branch_key,
            "branch_id": self.branch_id,
            "branch_identity_sha256": self.branch_identity_sha256,
            "branch_identity": self.branch_identity,
            "trajectory_projection": self.trajectory_projection,
            "rng_lineage_provenance": self.rng_lineage_provenance,
        }


def create_checkpoint_branch(
    *,
    branch_key: str,
    source_checkpoint_projection_sha256: str,
    source_generation: int,
    trajectory_projection: Mapping[str, Any],
    source_rng_state_sha256: str,
    inherited_rng_state_sha256: str | None = None,
    source_branch_id: str | None = None,
) -> CudaCheckpointBranch:
    """Create one deterministic branch with exact checkpoint RNG inheritance."""

    key = validate_branch_key(branch_key)
    checkpoint_digest = _strict_sha256(
        source_checkpoint_projection_sha256,
        path="source_checkpoint_projection_sha256",
    )
    generation = _generation(source_generation, path="source_generation")
    source_rng_digest = _strict_sha256(
        source_rng_state_sha256,
        path="source_rng_state_sha256",
    )
    inherited_rng_digest = _strict_sha256(
        (
            source_rng_digest
            if inherited_rng_state_sha256 is None
            else inherited_rng_state_sha256
        ),
        path="inherited_rng_state_sha256",
    )
    if inherited_rng_digest != source_rng_digest:
        raise CudaCheckpointBranchError(
            "checkpoint branches must inherit the exact RNG state without reseeding"
        )
    parent_id = _source_branch_id(source_branch_id)
    projected, projected_json = _trajectory_projection(trajectory_projection)

    identity = {
        "schema_version": BRANCH_IDENTITY_SCHEMA_VERSION,
        "branch_key": key,
        "source_checkpoint": {
            "generation": generation,
            "projection_sha256": checkpoint_digest,
        },
        "source_lineage": {
            "branch_id": parent_id,
            "rng_state_sha256": source_rng_digest,
        },
        "trajectory_projection": projected,
    }
    identity_json = canonical_branch_json_bytes(identity)
    identity_digest = hashlib.sha256(BRANCH_ID_DOMAIN + identity_json).hexdigest()
    branch_id = BRANCH_ID_PREFIX + identity_digest
    provenance = {
        "schema_version": RNG_LINEAGE_PROVENANCE_SCHEMA_VERSION,
        "branch_key": key,
        "branch_id": branch_id,
        "source_branch_id": parent_id,
        "source_checkpoint_projection_sha256": checkpoint_digest,
        "inheritance_mode": RNG_INHERITANCE_MODE,
        "source_rng_state_sha256": source_rng_digest,
        "inherited_rng_state_sha256": inherited_rng_digest,
        "reseeded": False,
        "seed_derivation": None,
    }
    return CudaCheckpointBranch(
        branch_key=key,
        branch_id=branch_id,
        branch_identity_sha256=identity_digest,
        _branch_identity_json=identity_json,
        _trajectory_projection_json=projected_json,
        _rng_lineage_provenance_json=canonical_branch_json_bytes(provenance),
    )


def _record(value: CudaCheckpointBranch | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, CudaCheckpointBranch):
        return value.to_record()
    record, _ = _canonical_object(value, path="checkpoint branch record")
    return record


def validate_checkpoint_branch(
    value: CudaCheckpointBranch | Mapping[str, Any],
) -> CudaCheckpointBranch:
    """Recompute and validate all identity and exact-inheritance fields."""

    record = _record(value)
    if set(record) != _RECORD_FIELDS:
        raise CudaCheckpointBranchError(
            "checkpoint branch record fields do not match the v1 schema"
        )
    if record["schema_version"] != CHECKPOINT_BRANCH_SCHEMA_VERSION:
        raise CudaCheckpointBranchError("checkpoint branch schema_version is invalid")

    key = validate_branch_key(record["branch_key"])
    identity, identity_json = _canonical_object(
        record["branch_identity"], path="branch_identity"
    )
    projection, projection_json = _trajectory_projection(
        record["trajectory_projection"]
    )
    if set(identity) != _IDENTITY_FIELDS:
        raise CudaCheckpointBranchError(
            "branch_identity fields do not match the v1 schema"
        )
    if identity["schema_version"] != BRANCH_IDENTITY_SCHEMA_VERSION:
        raise CudaCheckpointBranchError("branch identity schema_version is invalid")
    if identity["branch_key"] != key:
        raise CudaCheckpointBranchError("branch identity branch_key does not match")
    if canonical_branch_json_bytes(identity["trajectory_projection"]) != projection_json:
        raise CudaCheckpointBranchError(
            "branch identity trajectory_projection does not match the record"
        )

    source, _ = _canonical_object(
        identity["source_checkpoint"], path="branch_identity.source_checkpoint"
    )
    if set(source) != _SOURCE_CHECKPOINT_FIELDS:
        raise CudaCheckpointBranchError(
            "source checkpoint fields do not match the v1 schema"
        )
    checkpoint_digest = _strict_sha256(
        source["projection_sha256"],
        path="branch_identity.source_checkpoint.projection_sha256",
    )
    _generation(
        source["generation"], path="branch_identity.source_checkpoint.generation"
    )
    source_lineage, _ = _canonical_object(
        identity["source_lineage"], path="branch_identity.source_lineage"
    )
    if set(source_lineage) != _SOURCE_LINEAGE_FIELDS:
        raise CudaCheckpointBranchError(
            "source lineage fields do not match the v1 schema"
        )
    identity_parent_id = _source_branch_id(source_lineage["branch_id"])
    identity_rng_digest = _strict_sha256(
        source_lineage["rng_state_sha256"],
        path="branch_identity.source_lineage.rng_state_sha256",
    )

    identity_digest = _strict_sha256(
        record["branch_identity_sha256"], path="branch_identity_sha256"
    )
    expected_digest = hashlib.sha256(BRANCH_ID_DOMAIN + identity_json).hexdigest()
    if identity_digest != expected_digest:
        raise CudaCheckpointBranchError("branch_identity_sha256 does not match identity")
    expected_id = BRANCH_ID_PREFIX + expected_digest
    if record["branch_id"] != expected_id:
        raise CudaCheckpointBranchError("branch_id does not match canonical identity")

    provenance, provenance_json = _canonical_object(
        record["rng_lineage_provenance"], path="rng_lineage_provenance"
    )
    if set(provenance) != _RNG_PROVENANCE_FIELDS:
        raise CudaCheckpointBranchError(
            "RNG lineage provenance fields do not match the v1 schema"
        )
    if provenance["schema_version"] != RNG_LINEAGE_PROVENANCE_SCHEMA_VERSION:
        raise CudaCheckpointBranchError("RNG lineage schema_version is invalid")
    if provenance["branch_key"] != key or provenance["branch_id"] != expected_id:
        raise CudaCheckpointBranchError(
            "RNG lineage branch identity does not match the record"
        )
    provenance_parent_id = _source_branch_id(provenance["source_branch_id"])
    if provenance_parent_id != identity_parent_id:
        raise CudaCheckpointBranchError(
            "RNG lineage parent branch does not match branch identity"
        )
    if provenance["source_checkpoint_projection_sha256"] != checkpoint_digest:
        raise CudaCheckpointBranchError(
            "RNG lineage source checkpoint does not match branch identity"
        )
    source_rng_digest = _strict_sha256(
        provenance["source_rng_state_sha256"],
        path="rng_lineage_provenance.source_rng_state_sha256",
    )
    inherited_rng_digest = _strict_sha256(
        provenance["inherited_rng_state_sha256"],
        path="rng_lineage_provenance.inherited_rng_state_sha256",
    )
    if (
        source_rng_digest != identity_rng_digest
        or provenance["inheritance_mode"] != RNG_INHERITANCE_MODE
        or provenance["reseeded"] is not False
        or provenance["seed_derivation"] is not None
        or inherited_rng_digest != source_rng_digest
    ):
        raise CudaCheckpointBranchError(
            "RNG lineage must prove exact checkpoint inheritance without reseeding"
        )

    return CudaCheckpointBranch(
        branch_key=key,
        branch_id=expected_id,
        branch_identity_sha256=expected_digest,
        _branch_identity_json=identity_json,
        _trajectory_projection_json=projection_json,
        _rng_lineage_provenance_json=provenance_json,
    )


def checkpoint_branch_trajectory_projection(
    value: CudaCheckpointBranch | Mapping[str, Any],
) -> dict[str, Any]:
    """Return trajectory-only state, excluding branch and RNG provenance."""

    if isinstance(value, CudaCheckpointBranch):
        return value.trajectory_projection
    if "trajectory_projection" in value:
        projection = value["trajectory_projection"]
    else:
        projection = value
    projected, _ = _trajectory_projection(projection)
    return projected


def checkpoint_branch_trajectory_sha256(
    value: CudaCheckpointBranch | Mapping[str, Any],
) -> str:
    projection = checkpoint_branch_trajectory_projection(value)
    return hashlib.sha256(canonical_branch_json_bytes(projection)).hexdigest()


def validate_checkpoint_branch_collision(
    existing: CudaCheckpointBranch | Mapping[str, Any],
    candidate: CudaCheckpointBranch | Mapping[str, Any],
) -> str:
    """Classify distinct/idempotent records and reject same-ID collisions."""

    left = validate_checkpoint_branch(existing)
    right = validate_checkpoint_branch(candidate)
    if left.branch_id != right.branch_id:
        return "distinct"
    if canonical_branch_json_bytes(left.to_record()) == canonical_branch_json_bytes(
        right.to_record()
    ):
        return "idempotent"
    raise CudaCheckpointBranchCollisionError(
        f"branch ID collision with non-idempotent record: {left.branch_id}"
    )


def validate_checkpoint_branch_batch(
    branches: Sequence[CudaCheckpointBranch | Mapping[str, Any]],
) -> tuple[CudaCheckpointBranch, ...]:
    """Validate a branch batch and reject duplicate keys or IDs."""

    if isinstance(branches, (str, bytes, bytearray)):
        raise CudaCheckpointBranchError("checkpoint branch batch must be a sequence")
    validated: list[CudaCheckpointBranch] = []
    keys: set[str] = set()
    ids: set[str] = set()
    for index, value in enumerate(branches):
        branch = validate_checkpoint_branch(value)
        if branch.branch_key in keys:
            raise CudaCheckpointBranchError(
                f"checkpoint branch batch contains duplicate branch_key at index {index}: "
                f"{branch.branch_key}"
            )
        if branch.branch_id in ids:
            raise CudaCheckpointBranchCollisionError(
                f"checkpoint branch batch contains duplicate branch_id at index {index}: "
                f"{branch.branch_id}"
            )
        keys.add(branch.branch_key)
        ids.add(branch.branch_id)
        validated.append(branch)
    return tuple(validated)


def create_checkpoint_branch_batch(
    specifications: Sequence[Mapping[str, Any]],
) -> tuple[CudaCheckpointBranch, ...]:
    """Create and validate a batch of keyword specifications."""

    if isinstance(specifications, (str, bytes, bytearray)):
        raise CudaCheckpointBranchError(
            "checkpoint branch specifications must be a sequence"
        )
    branches: list[CudaCheckpointBranch] = []
    for index, specification in enumerate(specifications):
        if not isinstance(specification, Mapping):
            raise CudaCheckpointBranchError(
                f"checkpoint branch specification at index {index} must be a mapping"
            )
        try:
            branches.append(create_checkpoint_branch(**dict(specification)))
        except TypeError as exc:
            raise CudaCheckpointBranchError(
                f"checkpoint branch specification at index {index} is invalid: {exc}"
            ) from exc
    return validate_checkpoint_branch_batch(branches)


__all__ = [
    "BRANCH_ID_DOMAIN",
    "BRANCH_ID_PREFIX",
    "BRANCH_IDENTITY_SCHEMA_VERSION",
    "CHECKPOINT_BRANCH_SCHEMA_VERSION",
    "RNG_INHERITANCE_MODE",
    "RNG_LINEAGE_PROVENANCE_SCHEMA_VERSION",
    "CudaCheckpointBranch",
    "CudaCheckpointBranchCollisionError",
    "CudaCheckpointBranchError",
    "canonical_branch_json_bytes",
    "checkpoint_branch_id",
    "checkpoint_branch_identity_sha256",
    "checkpoint_branch_trajectory_projection",
    "checkpoint_branch_trajectory_sha256",
    "create_checkpoint_branch",
    "create_checkpoint_branch_batch",
    "validate_branch_key",
    "validate_checkpoint_branch",
    "validate_checkpoint_branch_batch",
    "validate_checkpoint_branch_collision",
]
