"""Fail-closed loader for registry-bound inheritance cross-path CUDA artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Final


MANIFEST_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_flattened_controller_extension_manifest_v1"
)
QUALIFIED_CACHE_WIDTH_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_qualified_cache_width_v1"
)
QUALIFIED_CACHE_WIDTHS: Final[frozenset[int]] = frozenset({1, 2})
QUALIFIED_CACHE_WIDTH_DEFINE_PREFIX: Final[str] = (
    "-DEIEL_CUDA_ARCHITECTURE_B_CACHE_WIDTH_VALID="
)
MECHANIC_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_mechanic_qualification_v2"
)
REPORTING_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_reporting_qualification_v1"
)
REGISTRY_SCHEMA_VERSION: Final[str] = "inheritance_cross_path_registry_v2"
TELEMETRY_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_telemetry_contract_v1"
)
MODULE_NAME: Final[str] = "eiel_cuda_architecture_b_flattened_controller_v1"
CROSS_PATH_ENTRYPOINT: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_cross_path_windows_v1"
)
RUN_CONTROL_ATTRIBUTE: Final[str] = "RunControlV2"
BUILD_PROVENANCE_ENTRYPOINT: Final[str] = (
    "eiel_authoritative_state_build_provenance_v1"
)
BUILD_PROVENANCE_SCHEMA_VERSION: Final[str] = (
    "eiel_authoritative_state_binary_build_provenance_v1"
)
COMMITTED_SOURCE_SNAPSHOT_SCHEMA_VERSION: Final[str] = (
    "eiel_authoritative_state_committed_source_snapshot_v1"
)
ARTIFACT_IDS: Final[tuple[str, ...]] = (
    "cuda_extension",
    "extension_manifest",
    "inheritance_registry",
    "telemetry_contract",
    "qualification_receipt",
    "reporting_receipt",
)
UPSTREAM_BINDING_IDS: Final[tuple[str, ...]] = (
    "cuda_extension",
    "extension_manifest",
    "inheritance_registry",
    "telemetry_contract",
)
CANDIDATE_AUTHORITY_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_mechanic_candidate_authority_v1"
)
CROSS_STORAGE_PARITY_CAPTURE_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_cross_storage_resident_capture_v1"
)
CROSS_STORAGE_PARITY_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_cross_storage_parity_evidence_v1"
)
CROSS_STORAGE_PARITY_PLAN_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_cross_storage_parity_plan_v1"
)
CROSS_STORAGE_PARITY_RESULT_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_cross_storage_parity_result_v1"
)
SAME_STORAGE_QUALIFICATION_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_same_storage_qualification_v1"
)
PARITY_DIMENSION_IDS: Final[tuple[str, ...]] = (
    "cells",
    "cell_order",
    "modes",
    "seeds",
    "generations",
    "checkpoints",
)
PARITY_EXACTNESS_IDS: Final[tuple[str, ...]] = (
    "final_state",
    "rng",
    "checkpoint",
    "attempt_content",
    "outcome_content",
    "selected_parent_content",
    "child_content",
    "commit_content",
    "donor_content",
    "primary_eligibility_aggregate",
    "mate_eligibility_aggregate",
    "reporting_independence",
)


class InheritanceCrossPathQualifiedArtifactError(RuntimeError):
    """The declared cross-path artifact chain is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class CrossPathArtifactBindingV1:
    """One immutable plan artifact after path and digest verification."""

    artifact_id: str
    path: Path
    sha256: str
    abi: str


@dataclass(frozen=True, slots=True)
class QualifiedInheritanceCrossPathArtifactV1:
    """Loaded CUDA module together with its fully verified authority chain."""

    module: ModuleType
    bindings: Mapping[str, CrossPathArtifactBindingV1]
    manifest: Mapping[str, Any]
    registry: Mapping[str, Any]
    telemetry_contract: Mapping[str, Any]
    mechanic_receipt: Mapping[str, Any]
    reporting_receipt: Mapping[str, Any]
    candidate_authority_sha256: str


@dataclass(frozen=True, slots=True)
class CandidateInheritanceCrossPathArtifactV1:
    """Loaded mechanic-gate candidate with no production authorization."""

    module: ModuleType
    bindings: Mapping[str, CrossPathArtifactBindingV1]
    manifest: Mapping[str, Any]
    registry: Mapping[str, Any]
    telemetry_contract: Mapping[str, Any]
    candidate_authority_sha256: str
    production_authority: bool = False


def _fail(message: str) -> InheritanceCrossPathQualifiedArtifactError:
    return InheritanceCrossPathQualifiedArtifactError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _validate_manifest_build_provenance(
    payload: Mapping[str, Any],
) -> dict[str, str]:
    provenance = payload.get("authoritative_state_build_provenance_v1")
    if (
        not isinstance(provenance, Mapping)
        or set(provenance)
        != {
            "schema_version",
            "build_source_commit",
            "compiled_source_snapshot_sha256",
        }
        or provenance.get("schema_version") != BUILD_PROVENANCE_SCHEMA_VERSION
        or not _is_commit(provenance.get("build_source_commit"))
        or not _is_sha256(provenance.get("compiled_source_snapshot_sha256"))
    ):
        raise _fail(
            "cross-path extension manifest build provenance is missing or invalid"
        )
    snapshot = payload.get("committed_source_snapshot")
    if not isinstance(snapshot, Mapping):
        raise _fail("cross-path committed source snapshot is missing")
    snapshot_fields = {
        "schema_version",
        "build_source_commit",
        "sources",
        "compiled_source_snapshot_sha256",
    }
    sources = snapshot.get("sources")
    if (
        set(snapshot) != snapshot_fields
        or snapshot.get("schema_version")
        != COMMITTED_SOURCE_SNAPSHOT_SCHEMA_VERSION
        or snapshot.get("build_source_commit")
        != provenance.get("build_source_commit")
        or not isinstance(sources, Sequence)
        or isinstance(sources, (str, bytes))
        or not sources
    ):
        raise _fail("cross-path committed source snapshot is invalid")
    normalized_sources: list[dict[str, str]] = []
    for index, row in enumerate(sources):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or not isinstance(row.get("path"), str)
            or not row["path"]
            or Path(row["path"]).is_absolute()
            or ".." in Path(row["path"]).parts
            or not _is_sha256(row.get("sha256"))
        ):
            raise _fail(
                f"cross-path committed source snapshot row {index} is invalid"
            )
        normalized_sources.append(
            {"path": str(row["path"]), "sha256": str(row["sha256"])}
        )
    if normalized_sources != sorted(normalized_sources, key=lambda row: row["path"]):
        raise _fail("cross-path committed source snapshot order is noncanonical")
    unsigned_snapshot = {
        "schema_version": snapshot["schema_version"],
        "build_source_commit": snapshot["build_source_commit"],
        "sources": normalized_sources,
    }
    observed_snapshot_sha256 = snapshot.get(
        "compiled_source_snapshot_sha256"
    )
    if (
        not _is_sha256(observed_snapshot_sha256)
        or observed_snapshot_sha256 != _canonical_sha256(unsigned_snapshot)
        or observed_snapshot_sha256
        != provenance.get("compiled_source_snapshot_sha256")
    ):
        raise _fail("cross-path committed source snapshot hash is invalid")
    return {
        "schema_version": str(provenance["schema_version"]),
        "build_source_commit": str(provenance["build_source_commit"]),
        "compiled_source_snapshot_sha256": str(
            provenance["compiled_source_snapshot_sha256"]
        ),
    }


def _validate_binary_build_provenance(
    module: ModuleType,
    *,
    manifest: Mapping[str, Any],
) -> None:
    expected = _validate_manifest_build_provenance(manifest)
    export = getattr(module, BUILD_PROVENANCE_ENTRYPOINT, None)
    if not callable(export):
        raise _fail(
            "loaded extension lacks authoritative-state build provenance"
        )
    try:
        observed = dict(export())
    except (TypeError, ValueError) as exc:
        raise _fail(
            "loaded extension build provenance is not a mapping"
        ) from exc
    if observed != expected:
        raise _fail(
            "cross-path manifest/binary build provenance mismatch"
        )


def _json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise _fail(f"cross-path {name} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise _fail(f"cross-path {name} must be a JSON object")
    return dict(value)


def _default_module_loader(path: Path, module_name: str) -> ModuleType:
    # Loading torch first establishes the libtorch shared-library context.
    import torch  # noqa: F401, PLC0415

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise _fail(f"cannot construct cross-path CUDA loader for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bindings(
    rows: Mapping[str, Any],
    artifact_ids: Sequence[str] = ARTIFACT_IDS,
) -> dict[str, CrossPathArtifactBindingV1]:
    result: dict[str, CrossPathArtifactBindingV1] = {}
    for artifact_id in artifact_ids:
        row = rows.get(artifact_id)
        if not isinstance(row, Mapping):
            raise _fail(f"cross-path artifact is missing: {artifact_id}")
        raw_path = row.get("path")
        sha256 = row.get("sha256")
        abi = row.get("abi")
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or not _is_sha256(sha256)
            or not isinstance(abi, str)
            or not abi
        ):
            raise _fail(f"cross-path artifact binding is invalid: {artifact_id}")
        declared = Path(raw_path).expanduser()
        if not declared.is_absolute() or declared.is_symlink():
            raise _fail(
                f"cross-path artifact must be an absolute non-symlink: {artifact_id}"
            )
        path = declared.resolve()
        if not path.is_file() or _sha256(path) != sha256:
            raise _fail(f"cross-path artifact binding is stale: {artifact_id}")
        result[artifact_id] = CrossPathArtifactBindingV1(
            artifact_id=artifact_id,
            path=path,
            sha256=str(sha256),
            abi=abi,
        )
    return result


def _candidate_authority_sha256(
    bindings: Mapping[str, CrossPathArtifactBindingV1],
    *,
    provider_id: str,
    profile_id: str,
    expected_native_abi: str,
) -> str:
    payload = {
        "schema_version": CANDIDATE_AUTHORITY_SCHEMA_VERSION,
        "authorization_scope": "mechanic_isolation_gate_only",
        "production_authority": False,
        "provider_id": provider_id,
        "profile_id": profile_id,
        "native_abi": expected_native_abi,
        "artifact_sha256s": {
            artifact_id: bindings[artifact_id].sha256
            for artifact_id in UPSTREAM_BINDING_IDS
        },
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _require_identity(
    payload: Mapping[str, Any],
    *,
    schema_version: str,
    provider_id: str,
    profile_id: str,
    name: str,
) -> None:
    if payload.get("schema_version") != schema_version:
        raise _fail(f"cross-path {name} schema mismatch")
    if payload.get("provider_id") != provider_id:
        raise _fail(f"cross-path {name} provider mismatch")
    if payload.get("profile_id") != profile_id:
        raise _fail(f"cross-path {name} profile mismatch")


def _require_hash_bindings(
    value: Any,
    bindings: Mapping[str, CrossPathArtifactBindingV1],
    artifact_ids: Sequence[str],
    *,
    name: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(artifact_ids):
        raise _fail(f"cross-path {name} artifact bindings are incomplete")
    for artifact_id in artifact_ids:
        if value.get(artifact_id) != bindings[artifact_id].sha256:
            raise _fail(
                f"cross-path {name} artifact hash mismatch: {artifact_id}"
            )


def _require_digest_map(
    value: Any,
    artifact_ids: Sequence[str],
    *,
    name: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(artifact_ids):
        raise _fail(f"cross-path {name} artifact bindings are incomplete")
    result: dict[str, str] = {}
    for artifact_id in artifact_ids:
        digest = value.get(artifact_id)
        if not _is_sha256(digest):
            raise _fail(
                f"cross-path {name} artifact hash is invalid: {artifact_id}"
            )
        result[artifact_id] = str(digest)
    return result


def _validate_parity_capture(
    value: Any,
    *,
    storage_mode: str,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"cross-path {name} parity capture is missing")
    required = {
        "schema_version",
        "storage_mode",
        "capture_sha256",
        "ok",
        "preregistered",
        "real_resident_execution",
        "extension_sha256",
        "extension_manifest_sha256",
        "native_abi",
        "registry_sha256",
        "telemetry_contract_sha256",
        "dimension_sha256s",
    }
    dimensions = value.get("dimension_sha256s")
    if (
        set(value) != required
        or value.get("schema_version")
        != CROSS_STORAGE_PARITY_CAPTURE_SCHEMA_VERSION
        or value.get("storage_mode") != storage_mode
        or value.get("ok") is not True
        or value.get("preregistered") is not True
        or value.get("real_resident_execution") is not True
        or not isinstance(value.get("native_abi"), str)
        or not value.get("native_abi")
        or any(
            not _is_sha256(value.get(field))
            for field in (
                "capture_sha256",
                "extension_sha256",
                "extension_manifest_sha256",
                "registry_sha256",
                "telemetry_contract_sha256",
            )
        )
        or not isinstance(dimensions, Mapping)
        or set(dimensions) != set(PARITY_DIMENSION_IDS)
        or any(not _is_sha256(dimensions.get(field)) for field in PARITY_DIMENSION_IDS)
    ):
        raise _fail(f"cross-path {name} parity capture is invalid")
    return dict(value)


def _validate_cross_storage_parity(
    value: Any,
    *,
    qualified_gate_artifacts: Mapping[str, str],
    production_artifacts: Mapping[str, str],
    expected_native_abi: str,
) -> None:
    if not isinstance(value, Mapping):
        raise _fail("cross-path cross-storage parity evidence is missing")
    captures = value.get("captures")
    exactness = value.get("exactness")
    required = {
        "schema_version",
        "ok",
        "parity_plan_sha256",
        "capture_manifest_sha256",
        "parity_result_sha256",
        "captures",
        "exactness",
    }
    if (
        set(value) != required
        or value.get("schema_version")
        != CROSS_STORAGE_PARITY_EVIDENCE_SCHEMA_VERSION
        or value.get("ok") is not True
        or any(
            not _is_sha256(value.get(field))
            for field in (
                "parity_plan_sha256",
                "capture_manifest_sha256",
                "parity_result_sha256",
            )
        )
        or not isinstance(captures, Mapping)
        or set(captures) != {"exact_full", "compact_complete"}
        or not isinstance(exactness, Mapping)
        or set(exactness) != set(PARITY_EXACTNESS_IDS)
        or any(exactness.get(field) is not True for field in PARITY_EXACTNESS_IDS)
    ):
        raise _fail("cross-path cross-storage parity evidence is invalid")
    exact_full = _validate_parity_capture(
        captures["exact_full"],
        storage_mode="exact_full",
        name="exact-full",
    )
    compact = _validate_parity_capture(
        captures["compact_complete"],
        storage_mode="compact_complete",
        name="compact-complete",
    )
    if (
        exact_full["extension_sha256"] != compact["extension_sha256"]
        or exact_full["extension_manifest_sha256"]
        != compact["extension_manifest_sha256"]
        or exact_full["native_abi"] != compact["native_abi"]
        or exact_full["native_abi"] != expected_native_abi
        or exact_full["dimension_sha256s"] != compact["dimension_sha256s"]
    ):
        raise _fail(
            "cross-path parity captures differ outside storage registry/telemetry"
        )
    if (
        exact_full["capture_sha256"] == compact["capture_sha256"]
        or
        exact_full["registry_sha256"] == compact["registry_sha256"]
        or exact_full["telemetry_contract_sha256"]
        == compact["telemetry_contract_sha256"]
    ):
        raise _fail(
            "cross-path cross-storage registry and telemetry hashes are not distinct"
        )
    expected_qualified = {
        "cuda_extension": exact_full["extension_sha256"],
        "extension_manifest": exact_full["extension_manifest_sha256"],
        "inheritance_registry": exact_full["registry_sha256"],
        "telemetry_contract": exact_full["telemetry_contract_sha256"],
    }
    expected_production = {
        "cuda_extension": compact["extension_sha256"],
        "extension_manifest": compact["extension_manifest_sha256"],
        "inheritance_registry": compact["registry_sha256"],
        "telemetry_contract": compact["telemetry_contract_sha256"],
    }
    if (
        dict(qualified_gate_artifacts) != expected_qualified
        or dict(production_artifacts) != expected_production
    ):
        raise _fail(
            "cross-path cross-storage parity evidence artifact hash substitution"
        )


def _validate_cross_storage_plan_result(
    *,
    plan: Any,
    result: Any,
    evidence: Mapping[str, Any],
    qualified_gate_artifacts: Mapping[str, str],
    production_artifacts: Mapping[str, str],
    expected_native_abi: str,
    provider_id: str,
    profile_id: str,
) -> None:
    captures = evidence["captures"]
    exact_capture = captures["exact_full"]
    compact_capture = captures["compact_complete"]
    dimensions = exact_capture["dimension_sha256s"]
    expected_plan = {
        "schema_version": CROSS_STORAGE_PARITY_PLAN_SCHEMA_VERSION,
        "provider_id": provider_id,
        "profile_id": profile_id,
        "execution_route": "real_resident",
        "preregistered_capture_count": 2,
        "native_abi": expected_native_abi,
        "captures": {
            "exact_full": {
                "storage_mode": "exact_full",
                "artifact_hashes": dict(qualified_gate_artifacts),
            },
            "compact_complete": {
                "storage_mode": "compact_complete",
                "artifact_hashes": dict(production_artifacts),
            },
        },
        "dimension_sha256s": dict(dimensions),
        "required_exactness": list(PARITY_EXACTNESS_IDS),
    }
    if (
        not isinstance(plan, Mapping)
        or dict(plan) != expected_plan
        or _canonical_sha256(plan) != evidence["parity_plan_sha256"]
    ):
        raise _fail(
            "cross-path parity evidence does not bind the typed preregistered plan"
        )
    expected_result = {
        "schema_version": CROSS_STORAGE_PARITY_RESULT_SCHEMA_VERSION,
        "verifier_id": (
            "inheritance_cross_path_cross_storage_parity_verifier_v1"
        ),
        "ok": True,
        "plan_sha256": evidence["parity_plan_sha256"],
        "capture_manifest_sha256": evidence["capture_manifest_sha256"],
        "captures": {
            "exact_full": dict(exact_capture),
            "compact_complete": dict(compact_capture),
        },
        "exactness": dict(evidence["exactness"]),
    }
    if (
        not isinstance(result, Mapping)
        or dict(result) != expected_result
        or _canonical_sha256(result) != evidence["parity_result_sha256"]
    ):
        raise _fail(
            "cross-path parity evidence does not bind the typed verifier result"
        )


def _validate_storage_qualification(
    value: Any,
    *,
    qualified_gate_artifacts: Mapping[str, str],
    production_artifacts: Mapping[str, str],
    expected_native_abi: str,
    provider_id: str,
    profile_id: str,
) -> None:
    if not isinstance(value, Mapping):
        raise _fail("cross-path storage qualification is missing")
    mode = value.get("mode")
    if mode == "same_storage":
        if (
            set(value)
            != {
                "schema_version",
                "mode",
                "storage_mode",
                "hashes_identical",
                "qualified_gate_artifacts_sha256",
                "production_artifacts_sha256",
            }
            or value.get("schema_version")
            != SAME_STORAGE_QUALIFICATION_SCHEMA_VERSION
            or value.get("storage_mode")
            not in {"exact_full", "compact_complete"}
            or value.get("hashes_identical") is not True
            or not _is_sha256(value.get("qualified_gate_artifacts_sha256"))
            or not _is_sha256(value.get("production_artifacts_sha256"))
            or dict(qualified_gate_artifacts) != dict(production_artifacts)
        ):
            raise _fail("cross-path same-storage qualification hashes differ")
        canonical = hashlib.sha256(
            json.dumps(
                dict(qualified_gate_artifacts),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        if (
            value.get("qualified_gate_artifacts_sha256") != canonical
            or value.get("production_artifacts_sha256") != canonical
        ):
            raise _fail("cross-path same-storage qualification digest mismatch")
        return
    if mode == "cross_storage_promotion":
        if (
            set(value)
            != {
                "mode",
                "from_storage_mode",
                "to_storage_mode",
                "parity_evidence",
                "parity_plan",
                "parity_result",
            }
            or value.get("from_storage_mode") != "exact_full"
            or value.get("to_storage_mode") != "compact_complete"
        ):
            raise _fail("cross-path cross-storage promotion declaration is invalid")
        _validate_cross_storage_parity(
            value.get("parity_evidence"),
            qualified_gate_artifacts=qualified_gate_artifacts,
            production_artifacts=production_artifacts,
            expected_native_abi=expected_native_abi,
        )
        evidence = value.get("parity_evidence")
        assert isinstance(evidence, Mapping)
        _validate_cross_storage_plan_result(
            plan=value.get("parity_plan"),
            result=value.get("parity_result"),
            evidence=evidence,
            qualified_gate_artifacts=qualified_gate_artifacts,
            production_artifacts=production_artifacts,
            expected_native_abi=expected_native_abi,
            provider_id=provider_id,
            profile_id=profile_id,
        )
        return
    raise _fail("cross-path storage qualification mode is invalid")


def _validate_manifest(
    payload: Mapping[str, Any],
    *,
    extension: CrossPathArtifactBindingV1,
    expected_qualified_cache_width: int | None = None,
    require_authoritative_state_identity: bool = True,
) -> None:
    from engine.authoritative_state.production_identity import (
        validate_identity_against_local_schema_v1,
        validate_production_identity_v1,
    )
    from engine.authoritative_state.schema import load_schema_v1

    entrypoints = payload.get("entrypoints")
    attributes = payload.get("required_attributes")
    identity = payload.get("artifact_identity")
    cuda_images = (
        identity.get("cuda_images") if isinstance(identity, Mapping) else None
    )
    cache_width = payload.get("qualified_cache_width")
    cache_width_value = (
        cache_width.get("value") if isinstance(cache_width, Mapping) else None
    )
    cache_width_definition = (
        cache_width.get("compile_definition")
        if isinstance(cache_width, Mapping)
        else None
    )
    expected_definition = (
        f"{QUALIFIED_CACHE_WIDTH_DEFINE_PREFIX}(cache_width=={cache_width_value})"
        if type(cache_width_value) is int
        else None
    )
    cxx_flags = payload.get("cxx_flags")
    nvcc_flags = payload.get("nvcc_flags")
    _validate_manifest_build_provenance(payload)
    if require_authoritative_state_identity:
        authoritative_identity = validate_production_identity_v1(
            payload.get("authoritative_state_identity_v1"),
            path="extension_manifest.authoritative_state_identity_v1",
        )
        validate_identity_against_local_schema_v1(
            authoritative_identity,
            schema=load_schema_v1(),
            provider_id="eiel-production-provider-v1",
            profile_id="eiel-production-runtime-v1",
        )
    cache_width_contract_valid = (
        isinstance(cache_width, Mapping)
        and set(cache_width)
        == {"schema_version", "value", "compile_definition"}
        and cache_width.get("schema_version")
        == QUALIFIED_CACHE_WIDTH_SCHEMA_VERSION
        and type(cache_width_value) is int
        and cache_width_value in QUALIFIED_CACHE_WIDTHS
        and cache_width_definition == expected_definition
        and (
            expected_qualified_cache_width is None
            or cache_width_value == expected_qualified_cache_width
        )
        and isinstance(cxx_flags, Sequence)
        and not isinstance(cxx_flags, (str, bytes))
        and all(isinstance(flag, str) for flag in cxx_flags)
        and isinstance(nvcc_flags, Sequence)
        and not isinstance(nvcc_flags, (str, bytes))
        and all(isinstance(flag, str) for flag in nvcc_flags)
        and [
            flag
            for flag in cxx_flags
            if isinstance(flag, str)
            and flag.startswith(QUALIFIED_CACHE_WIDTH_DEFINE_PREFIX)
        ]
        == [expected_definition]
        and [
            flag
            for flag in nvcc_flags
            if isinstance(flag, str)
            and flag.startswith(QUALIFIED_CACHE_WIDTH_DEFINE_PREFIX)
        ]
        == [expected_definition]
    )
    if not cache_width_contract_valid:
        raise _fail(
            "cross-path extension manifest qualified cache-width contract "
            "is missing or mismatched"
        )
    try:
        artifact_path = Path(str(payload.get("artifact") or "")).resolve()
    except (OSError, RuntimeError) as exc:
        raise _fail("cross-path extension manifest artifact path is invalid") from exc
    if (
        payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or payload.get("module_name") != MODULE_NAME
        or artifact_path != extension.path
        or payload.get("artifact_sha256") != extension.sha256
        or isinstance(entrypoints, (str, bytes))
        or not isinstance(entrypoints, Sequence)
        or CROSS_PATH_ENTRYPOINT not in entrypoints
        or BUILD_PROVENANCE_ENTRYPOINT not in entrypoints
        or isinstance(attributes, (str, bytes))
        or not isinstance(attributes, Sequence)
        or RUN_CONTROL_ATTRIBUTE not in attributes
        or payload.get("source_snapshot_stable") is not True
        or payload.get("immutable_output") is not True
        or payload.get("prebuilt_outside_measured_execution") is not True
        or payload.get("runtime_jit_required_by_validation_route") is not False
        or payload.get("runtime_jit_compilation") is not False
        or payload.get("sass_only") is not True
        or payload.get("cuda_arch") != "sm_75"
        or not isinstance(cuda_images, Mapping)
        or cuda_images.get("target") != "sm_75"
        or cuda_images.get("sass_only") is not True
        or cuda_images.get("ptx_embedded") is not False
        or cuda_images.get("runtime_jit_required") is not False
    ):
        raise _fail(
            "cross-path extension manifest does not declare the qualified "
            "specialized sm75 SASS-only route"
        )


def _validate_registry(
    payload: Mapping[str, Any],
    *,
    expected_native_abi: str,
) -> None:
    if (
        payload.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or payload.get("native_abi") != expected_native_abi
    ):
        raise _fail("cross-path registry native binding mismatch")


def _validate_telemetry(
    payload: Mapping[str, Any],
    *,
    provider_id: str,
    profile_id: str,
    expected_native_abi: str,
    registry_sha256: str,
) -> None:
    _require_identity(
        payload,
        schema_version=TELEMETRY_CONTRACT_SCHEMA_VERSION,
        provider_id=provider_id,
        profile_id=profile_id,
        name="telemetry contract",
    )
    projection_contract = payload.get("projection_contract")
    if (
        payload.get("native_abi") != expected_native_abi
        or payload.get("registry_sha256") != registry_sha256
        or not isinstance(projection_contract, Mapping)
    ):
        raise _fail("cross-path telemetry contract binding is invalid")
    try:
        from engine.campaigns.inheritance_cross_path_exposure_projection_contract_v1 import (  # noqa: PLC0415
            validate_inheritance_cross_path_exposure_projection_contract_v1,
        )

        validate_inheritance_cross_path_exposure_projection_contract_v1(
            projection_contract
        )
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        raise _fail(
            "cross-path telemetry projection contract is invalid"
        ) from exc


def _validate_mechanic(
    payload: Mapping[str, Any],
    *,
    bindings: Mapping[str, CrossPathArtifactBindingV1],
    provider_id: str,
    profile_id: str,
    expected_native_abi: str,
) -> None:
    _require_identity(
        payload,
        schema_version=MECHANIC_RECEIPT_SCHEMA_VERSION,
        provider_id=provider_id,
        profile_id=profile_id,
        name="mechanic receipt",
    )
    artifact = payload.get("artifact")
    qualified_gate_artifacts = _require_digest_map(
        payload.get("qualified_gate_artifacts"),
        UPSTREAM_BINDING_IDS,
        name="qualified-gate",
    )
    production_artifacts = _require_digest_map(
        payload.get("production_artifacts"),
        UPSTREAM_BINDING_IDS,
        name="production",
    )
    _require_hash_bindings(
        payload.get("artifact_bindings"),
        bindings,
        UPSTREAM_BINDING_IDS,
        name="mechanic receipt production",
    )
    if (
        payload.get("ok") is not True
        or payload.get("registry_sha256")
        != bindings["inheritance_registry"].sha256
        or not isinstance(artifact, Mapping)
        or artifact.get("sha256") != bindings["cuda_extension"].sha256
        or artifact.get("abi") != expected_native_abi
        or not isinstance(payload.get("execution_contract"), Mapping)
    ):
        raise _fail("cross-path mechanic receipt binding is invalid")
    if production_artifacts != {
        artifact_id: bindings[artifact_id].sha256
        for artifact_id in UPSTREAM_BINDING_IDS
    }:
        raise _fail("cross-path mechanic production artifact binding is invalid")
    _validate_storage_qualification(
        payload.get("storage_qualification"),
        qualified_gate_artifacts=qualified_gate_artifacts,
        production_artifacts=production_artifacts,
        expected_native_abi=expected_native_abi,
        provider_id=provider_id,
        profile_id=profile_id,
    )


def _validate_reporting(
    payload: Mapping[str, Any],
    *,
    bindings: Mapping[str, CrossPathArtifactBindingV1],
    provider_id: str,
    profile_id: str,
) -> None:
    _require_identity(
        payload,
        schema_version=REPORTING_RECEIPT_SCHEMA_VERSION,
        provider_id=provider_id,
        profile_id=profile_id,
        name="reporting receipt",
    )
    _require_hash_bindings(
        payload.get("artifact_bindings"),
        bindings,
        (*UPSTREAM_BINDING_IDS, "qualification_receipt"),
        name="reporting receipt",
    )
    parity = payload.get("reporting_parity")
    enabled = parity.get("enabled") if isinstance(parity, Mapping) else None
    disabled = parity.get("disabled") if isinstance(parity, Mapping) else None
    if (
        payload.get("ok") is not True
        or not isinstance(parity, Mapping)
        or set(parity)
        != {
            "authoritative_state_exact",
            "native_reproduction_exposure_exact",
            "enabled",
            "disabled",
        }
        or parity.get("authoritative_state_exact") is not True
        or parity.get("native_reproduction_exposure_exact") is not True
        or not isinstance(enabled, Mapping)
        or not isinstance(disabled, Mapping)
        or set(enabled)
        != {
            "evidence_sha256",
            "authoritative_state_sha256",
            "reporting_rows_present",
        }
        or set(disabled) != set(enabled)
        or not _is_sha256(enabled.get("evidence_sha256"))
        or not _is_sha256(disabled.get("evidence_sha256"))
        or not _is_sha256(enabled.get("authoritative_state_sha256"))
        or disabled.get("authoritative_state_sha256")
        != enabled.get("authoritative_state_sha256")
        or enabled.get("reporting_rows_present") is not True
        or disabled.get("reporting_rows_present") is not False
    ):
        raise _fail("cross-path reporting enabled/disabled parity is invalid")


class InheritanceCrossPathCandidateArtifactLoaderV1:
    """Tombstone for the retired standalone candidate artifact loader."""

    def __init__(
        self,
        artifact_rows: Mapping[str, Any],
        *,
        provider_id: str,
        profile_id: str,
        expected_native_abi: str,
        expected_qualified_cache_width: int | None = None,
        module_loader: Callable[[Path, str], ModuleType] | None = None,
    ) -> None:
        raise _fail(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "cross-path candidate artifact loading is disabled"
        )
        # Historical implementation retained temporarily for evidence reading.
        if not provider_id or not profile_id or not expected_native_abi:
            raise _fail("cross-path candidate-loader identity cannot be empty")
        self._bindings = _bindings(artifact_rows, UPSTREAM_BINDING_IDS)
        self._provider_id = provider_id
        self._profile_id = profile_id
        self._expected_native_abi = expected_native_abi
        if (
            expected_qualified_cache_width is not None
            and (
                type(expected_qualified_cache_width) is not int
                or expected_qualified_cache_width not in QUALIFIED_CACHE_WIDTHS
            )
        ):
            raise _fail("expected qualified cache width must equal 1 or 2")
        self._expected_qualified_cache_width = expected_qualified_cache_width
        self._module_loader = module_loader or _default_module_loader
        self.candidate_authority_sha256 = _candidate_authority_sha256(
            self._bindings,
            provider_id=provider_id,
            profile_id=profile_id,
            expected_native_abi=expected_native_abi,
        )

    @property
    def candidate_bindings(self) -> dict[str, str]:
        return {
            artifact_id: self._bindings[artifact_id].sha256
            for artifact_id in UPSTREAM_BINDING_IDS
        }

    def validate_candidate(self) -> dict[str, Mapping[str, Any]]:
        """Validate candidate hashes and the specialized sm75 SASS-only route."""

        manifest = _json(
            self._bindings["extension_manifest"].path,
            name="extension manifest",
        )
        registry = _json(
            self._bindings["inheritance_registry"].path,
            name="inheritance registry",
        )
        telemetry = _json(
            self._bindings["telemetry_contract"].path,
            name="telemetry contract",
        )
        _validate_manifest(
            manifest,
            extension=self._bindings["cuda_extension"],
            expected_qualified_cache_width=(
                self._expected_qualified_cache_width
            ),
            require_authoritative_state_identity=False,
        )
        _validate_registry(
            registry,
            expected_native_abi=self._expected_native_abi,
        )
        _validate_telemetry(
            telemetry,
            provider_id=self._provider_id,
            profile_id=self._profile_id,
            expected_native_abi=self._expected_native_abi,
            registry_sha256=self._bindings["inheritance_registry"].sha256,
        )
        return {
            "manifest": manifest,
            "registry": registry,
            "telemetry_contract": telemetry,
        }

    def __call__(self) -> CandidateInheritanceCrossPathArtifactV1:
        payloads = self.validate_candidate()
        module = self._module_loader(
            self._bindings["cuda_extension"].path,
            MODULE_NAME,
        )
        if not callable(getattr(module, CROSS_PATH_ENTRYPOINT, None)):
            raise _fail("loaded candidate lacks the cross-path CUDA entrypoint")
        if getattr(module, RUN_CONTROL_ATTRIBUTE, None) is None:
            raise _fail("loaded candidate lacks the native run-control attribute")
        _validate_binary_build_provenance(
            module,
            manifest=payloads["manifest"],
        )
        return CandidateInheritanceCrossPathArtifactV1(
            module=module,
            bindings=dict(self._bindings),
            manifest=payloads["manifest"],
            registry=payloads["registry"],
            telemetry_contract=payloads["telemetry_contract"],
            candidate_authority_sha256=self.candidate_authority_sha256,
        )


class InheritanceCrossPathQualifiedArtifactLoaderV1:
    """Tombstone for the retired standalone qualified artifact loader."""

    def __init__(
        self,
        artifact_rows: Mapping[str, Any],
        *,
        provider_id: str,
        profile_id: str,
        expected_native_abi: str,
        expected_qualified_cache_width: int | None = None,
        module_loader: Callable[[Path, str], ModuleType] | None = None,
    ) -> None:
        raise _fail(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "cross-path qualified artifact loading is disabled"
        )
        # Historical implementation retained temporarily for evidence reading.
        if not provider_id or not profile_id or not expected_native_abi:
            raise _fail("cross-path loader identity cannot be empty")
        self._bindings = _bindings(artifact_rows)
        self._provider_id = provider_id
        self._profile_id = profile_id
        self._expected_native_abi = expected_native_abi
        if (
            expected_qualified_cache_width is not None
            and (
                type(expected_qualified_cache_width) is not int
                or expected_qualified_cache_width not in QUALIFIED_CACHE_WIDTHS
            )
        ):
            raise _fail("expected qualified cache width must equal 1 or 2")
        self._expected_qualified_cache_width = expected_qualified_cache_width
        self._module_loader = module_loader or _default_module_loader

    @property
    def route_bindings(self) -> dict[str, str]:
        """Return every content hash that determines loader authority."""

        return {
            artifact_id: self._bindings[artifact_id].sha256
            for artifact_id in ARTIFACT_IDS
        }

    def validate_chain(self) -> dict[str, Mapping[str, Any]]:
        """Validate the complete JSON authority graph without loading CUDA."""

        manifest = _json(
            self._bindings["extension_manifest"].path,
            name="extension manifest",
        )
        registry = _json(
            self._bindings["inheritance_registry"].path,
            name="inheritance registry",
        )
        telemetry = _json(
            self._bindings["telemetry_contract"].path,
            name="telemetry contract",
        )
        mechanic = _json(
            self._bindings["qualification_receipt"].path,
            name="mechanic receipt",
        )
        reporting = _json(
            self._bindings["reporting_receipt"].path,
            name="reporting receipt",
        )
        _validate_manifest(
            manifest,
            extension=self._bindings["cuda_extension"],
            expected_qualified_cache_width=(
                self._expected_qualified_cache_width
            ),
        )
        _validate_registry(
            registry,
            expected_native_abi=self._expected_native_abi,
        )
        _validate_telemetry(
            telemetry,
            provider_id=self._provider_id,
            profile_id=self._profile_id,
            expected_native_abi=self._expected_native_abi,
            registry_sha256=self._bindings["inheritance_registry"].sha256,
        )
        _validate_mechanic(
            mechanic,
            bindings=self._bindings,
            provider_id=self._provider_id,
            profile_id=self._profile_id,
            expected_native_abi=self._expected_native_abi,
        )
        _validate_reporting(
            reporting,
            bindings=self._bindings,
            provider_id=self._provider_id,
            profile_id=self._profile_id,
        )
        return {
            "manifest": manifest,
            "registry": registry,
            "telemetry_contract": telemetry,
            "mechanic_receipt": mechanic,
            "reporting_receipt": reporting,
        }

    def __call__(
        self,
        artifact_path: str | Path,
        qualification_receipt_path: str | Path,
        reporting_receipt_path: str | Path,
    ) -> QualifiedInheritanceCrossPathArtifactV1:
        supplied = {
            "cuda_extension": Path(artifact_path).expanduser().resolve(),
            "qualification_receipt": Path(
                qualification_receipt_path
            ).expanduser().resolve(),
            "reporting_receipt": Path(reporting_receipt_path).expanduser().resolve(),
        }
        for artifact_id, path in supplied.items():
            if path != self._bindings[artifact_id].path:
                raise _fail(
                    f"cross-path loader path differs from frozen plan: {artifact_id}"
                )
        payloads = self.validate_chain()
        module = self._module_loader(
            self._bindings["cuda_extension"].path,
            MODULE_NAME,
        )
        if not callable(getattr(module, CROSS_PATH_ENTRYPOINT, None)):
            raise _fail("loaded extension lacks the cross-path CUDA entrypoint")
        if getattr(module, RUN_CONTROL_ATTRIBUTE, None) is None:
            raise _fail("loaded extension lacks the native run-control attribute")
        _validate_binary_build_provenance(
            module,
            manifest=payloads["manifest"],
        )
        identity_export = getattr(
            module, "eiel_authoritative_state_identity_v1", None
        )
        if not callable(identity_export):
            raise _fail(
                "loaded extension lacks authoritative-state self-description"
            )
        from engine.authoritative_state.production_identity import (
            validate_admission_identity_v1,
        )

        manifest_identity = payloads["manifest"].get(
            "authoritative_state_identity_v1"
        )
        binary_identity = dict(identity_export())
        try:
            validate_admission_identity_v1(
                request=manifest_identity,
                provider=manifest_identity,
                manifest=manifest_identity,
                binary=binary_identity,
            )
        except ValueError as exc:
            raise _fail(
                f"cross-path manifest/binary state identity mismatch: {exc}"
            ) from exc
        return QualifiedInheritanceCrossPathArtifactV1(
            module=module,
            bindings=dict(self._bindings),
            manifest=payloads["manifest"],
            registry=payloads["registry"],
            telemetry_contract=payloads["telemetry_contract"],
            mechanic_receipt=payloads["mechanic_receipt"],
            reporting_receipt=payloads["reporting_receipt"],
            candidate_authority_sha256=_candidate_authority_sha256(
                self._bindings,
                provider_id=self._provider_id,
                profile_id=self._profile_id,
                expected_native_abi=self._expected_native_abi,
            ),
        )


__all__ = [
    "ARTIFACT_IDS",
    "CANDIDATE_AUTHORITY_SCHEMA_VERSION",
    "CROSS_PATH_ENTRYPOINT",
    "CandidateInheritanceCrossPathArtifactV1",
    "CrossPathArtifactBindingV1",
    "InheritanceCrossPathCandidateArtifactLoaderV1",
    "InheritanceCrossPathQualifiedArtifactError",
    "InheritanceCrossPathQualifiedArtifactLoaderV1",
    "MANIFEST_SCHEMA_VERSION",
    "QUALIFIED_CACHE_WIDTH_SCHEMA_VERSION",
    "QUALIFIED_CACHE_WIDTHS",
    "MECHANIC_RECEIPT_SCHEMA_VERSION",
    "QualifiedInheritanceCrossPathArtifactV1",
    "REGISTRY_SCHEMA_VERSION",
    "REPORTING_RECEIPT_SCHEMA_VERSION",
    "TELEMETRY_CONTRACT_SCHEMA_VERSION",
]
