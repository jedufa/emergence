"""Atomically stage a non-production cross-path native artifact candidate.

The deterministic native build owns the extension and its adjacent manifest.
This module composes the registry and telemetry contracts around those immutable
artifacts, validates the four-artifact candidate chain without loading CUDA,
and publishes a binding manifest as the final completion marker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import ModuleType
from typing import Any, Final

from .canonical import pretty_json
from .compiler import load_campaign_spec
from .production_engine_provider import PROFILE_ID, PROVIDER_ID
from .inheritance_cross_path_exposure_projection_contract_v1 import (
    validate_inheritance_cross_path_exposure_projection_contract_v1,
)
from .inheritance_cross_path_qualification_receipts_v1 import (
    compose_cross_path_telemetry_contract_v1,
)
from .inheritance_cross_path_registry_composer_v1 import (
    AgentIdAllocationCapacityV1,
    InheritanceCrossPathRegistryEnvelopeV1,
    ReproductionExposureCapacityV1,
    compose_inheritance_cross_path_registry_v2,
)
from engine.native_runtime.inheritance_cross_path_qualified_artifact_v1 import (
    CANDIDATE_AUTHORITY_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    QUALIFIED_CACHE_WIDTHS,
    REGISTRY_SCHEMA_VERSION,
    TELEMETRY_CONTRACT_SCHEMA_VERSION,
    InheritanceCrossPathCandidateArtifactLoaderV1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    required_resident_agent_id_capacity_v1,
)


CANDIDATE_BINDING_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_candidate_binding_manifest_v1"
)
REGISTRY_FILENAME: Final[str] = "inheritance-cross-path-registry.json"
TELEMETRY_FILENAME: Final[str] = "inheritance-cross-path-telemetry-contract.json"
BINDING_FILENAME: Final[str] = "inheritance-cross-path-candidate-binding.json"


class InheritanceCrossPathCandidateStagingError(RuntimeError):
    """Candidate inputs cannot be validated or published safely."""


def _fail(message: str) -> InheritanceCrossPathCandidateStagingError:
    return InheritanceCrossPathCandidateStagingError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise _fail(f"{field} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping) or not payload:
        raise _fail(f"{field} must be a non-empty JSON object")
    return dict(payload)


def _reject_symlink_components(path: Path, *, field: str) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise _fail(f"{field} must not contain symlinks")


def _input_file(path: Path, *, field: str, absolute: bool = False) -> Path:
    if absolute and not path.is_absolute():
        raise _fail(f"{field} must be an absolute path")
    _reject_symlink_components(path, field=field)
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise _fail(f"{field} is not a regular file")
    return resolved


def _positive(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise _fail(f"{field} must be a positive integer")
    return value


def _obligations(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if (
        not result
        or len(result) != len(set(result))
        or any(not isinstance(value, str) or not value for value in result)
    ):
        raise _fail("obligations must be unique non-empty names")
    return result


@dataclass(frozen=True, slots=True)
class CandidateArtifactStagingRequestV1:
    """Typed declarative inputs for one immutable candidate publication."""

    spec_path: Path
    extension_path: Path
    projection_contract_path: Path
    output_dir: Path
    max_admitted_batch: int
    max_generations: int
    ticks_per_generation: int
    expected_qualified_cache_width: int
    required_obligations: tuple[str, ...]
    exposure_storage_mode: str
    exposure_attempt_capacity: int
    exposure_snapshot_capacity: int
    exposure_aggregate_capacity: int

    def __post_init__(self) -> None:
        _positive(self.max_admitted_batch, field="max_admitted_batch")
        _positive(self.max_generations, field="max_generations")
        _positive(self.ticks_per_generation, field="ticks_per_generation")
        if (
            type(self.expected_qualified_cache_width) is not int
            or self.expected_qualified_cache_width not in QUALIFIED_CACHE_WIDTHS
        ):
            raise _fail("expected_qualified_cache_width must equal 1 or 2")
        _obligations(self.required_obligations)
        ReproductionExposureCapacityV1(
            storage_mode=self.exposure_storage_mode,
            attempt_capacity=_positive(
                self.exposure_attempt_capacity,
                field="exposure_attempt_capacity",
            ),
            snapshot_capacity=_positive(
                self.exposure_snapshot_capacity,
                field="exposure_snapshot_capacity",
            ),
            aggregate_capacity=_positive(
                self.exposure_aggregate_capacity,
                field="exposure_aggregate_capacity",
            ),
        )


@dataclass(frozen=True, slots=True)
class CandidateArtifactPublicationV1:
    """Paths and immutable identity of one staged candidate chain."""

    output_dir: Path
    registry_path: Path
    telemetry_contract_path: Path
    binding_manifest_path: Path
    candidate_authority_sha256: str
    artifact_rows: Mapping[str, Mapping[str, str]]

    def payload(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "registry_path": str(self.registry_path),
            "telemetry_contract_path": str(self.telemetry_contract_path),
            "binding_manifest_path": str(self.binding_manifest_path),
            "candidate_authority_sha256": self.candidate_authority_sha256,
            "production_authority": False,
        }


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(raw_temporary)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(pretty_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link is an atomic no-replace publication on the same
        # filesystem.  The temporary inode is then unlinked.
        os.link(temporary, path)
    except FileExistsError as exc:
        raise _fail(f"refusing to overwrite candidate artifact: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _artifact_rows(
    *,
    extension: Path,
    manifest: Path,
    registry: Path,
    telemetry: Path,
    native_abi: str,
) -> dict[str, dict[str, str]]:
    paths_and_abis = {
        "cuda_extension": (extension, native_abi),
        "extension_manifest": (manifest, MANIFEST_SCHEMA_VERSION),
        "inheritance_registry": (registry, REGISTRY_SCHEMA_VERSION),
        "telemetry_contract": (
            telemetry,
            TELEMETRY_CONTRACT_SCHEMA_VERSION,
        ),
    }
    return {
        artifact_id: {
            "path": str(path),
            "sha256": _sha256(path),
            "abi": abi,
        }
        for artifact_id, (path, abi) in paths_and_abis.items()
    }


def stage_inheritance_cross_path_candidate_v1(
    request: CandidateArtifactStagingRequestV1,
    *,
    native_api: ModuleType | None = None,
) -> CandidateArtifactPublicationV1:
    """Reject staging of a new retired-route candidate artifact chain."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "candidate artifact staging is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    spec_path = _input_file(request.spec_path, field="spec_path")
    projection_path = _input_file(
        request.projection_contract_path,
        field="projection_contract_path",
    )
    extension = _input_file(
        request.extension_path,
        field="extension_path",
        absolute=True,
    )
    manifest = _input_file(
        Path(f"{extension}.manifest.json"),
        field="adjacent extension manifest",
        absolute=True,
    )
    if manifest.parent != extension.parent:
        raise _fail("extension manifest must be adjacent to the extension")

    output_dir = request.output_dir
    if not output_dir.is_absolute():
        raise _fail("output_dir must be an absolute path")
    _reject_symlink_components(output_dir.parent, field="output_dir")
    if output_dir.exists() or output_dir.is_symlink():
        raise _fail(f"refusing to overwrite output_dir: {output_dir}")
    if not output_dir.parent.is_dir():
        raise _fail("output_dir parent must already exist")

    spec = load_campaign_spec(spec_path)
    projection_contract = _read_object(
        projection_path,
        field="projection_contract_path",
    )
    validate_inheritance_cross_path_exposure_projection_contract_v1(
        projection_contract
    )
    projection_envelope = projection_contract.get("envelope")
    expected_projection_bounds = {
        "max_batch": request.max_admitted_batch,
        "storage_mode": request.exposure_storage_mode,
        "attempt_capacity": request.exposure_attempt_capacity,
        "snapshot_capacity": request.exposure_snapshot_capacity,
        "aggregate_capacity": request.exposure_aggregate_capacity,
    }
    if (
        not isinstance(projection_envelope, Mapping)
        or any(
            projection_envelope.get(field) != value
            for field, value in expected_projection_bounds.items()
        )
    ):
        raise _fail(
            "projection contract envelope differs from candidate staging bounds"
        )
    identity_allocation = AgentIdAllocationCapacityV1.from_payload(
        projection_envelope.get("agent_id_allocation")
    )
    if (
        identity_allocation.allocation_generation_count
        < request.max_generations
    ):
        raise _fail(
            "projection contract identity allocation does not cover "
            "candidate max_generations"
        )
    required_identity_capacity = required_resident_agent_id_capacity_v1(
        identity_allocation.current_capacity,
        identity_allocation.population_next_id,
        identity_allocation.max_births_per_tick,
        identity_allocation.slot_capacity,
        allocation_generation_count=(
            identity_allocation.allocation_generation_count
        ),
        retained_capacity=identity_allocation.retained_capacity,
    )
    if (
        identity_allocation.required_aggregate_capacity
        != required_identity_capacity
    ):
        raise _fail(
            "projection contract required identity allocation K is stale"
        )
    if request.exposure_aggregate_capacity < required_identity_capacity:
        raise _fail(
            "candidate aggregate capacity is below the required production K"
        )
    capacity = ReproductionExposureCapacityV1(
        storage_mode=request.exposure_storage_mode,
        attempt_capacity=request.exposure_attempt_capacity,
        snapshot_capacity=request.exposure_snapshot_capacity,
        aggregate_capacity=request.exposure_aggregate_capacity,
        agent_id_allocation=identity_allocation,
    )
    envelope = InheritanceCrossPathRegistryEnvelopeV1(
        max_admitted_batch=request.max_admitted_batch,
        max_generations=request.max_generations,
        ticks_per_generation=request.ticks_per_generation,
        required_obligations=_obligations(request.required_obligations),
        reproduction_exposure_capacity=capacity,
    )
    registry_payload = compose_inheritance_cross_path_registry_v2(
        spec,
        envelope,
        native_api=native_api,
    )
    native_abi = str(registry_payload["native_abi"])

    created_output = False
    try:
        output_dir.mkdir(mode=0o755)
        created_output = True
        registry_path = output_dir / REGISTRY_FILENAME
        telemetry_path = output_dir / TELEMETRY_FILENAME
        binding_path = output_dir / BINDING_FILENAME

        _write_new(registry_path, registry_payload)
        telemetry_payload = compose_cross_path_telemetry_contract_v1(
            native_abi=native_abi,
            registry_sha256=_sha256(registry_path),
            projection_contract=projection_contract,
        )
        _write_new(telemetry_path, telemetry_payload)
        rows = _artifact_rows(
            extension=extension,
            manifest=manifest,
            registry=registry_path,
            telemetry=telemetry_path,
            native_abi=native_abi,
        )
        loader = InheritanceCrossPathCandidateArtifactLoaderV1(
            rows,
            provider_id=PROVIDER_ID,
            profile_id=PROFILE_ID,
            expected_native_abi=native_abi,
            expected_qualified_cache_width=(
                request.expected_qualified_cache_width
            ),
        )
        validated = loader.validate_candidate()
        qualified_cache_width = dict(
            validated["manifest"]["qualified_cache_width"]
        )

        binding_payload = {
            "schema_version": CANDIDATE_BINDING_MANIFEST_SCHEMA_VERSION,
            "candidate_authority_schema_version": (
                CANDIDATE_AUTHORITY_SCHEMA_VERSION
            ),
            "authorization_scope": "mechanic_isolation_gate_only",
            "production_authority": False,
            "provider_id": PROVIDER_ID,
            "profile_id": PROFILE_ID,
            "native_abi": native_abi,
            "candidate_authority_sha256": (
                loader.candidate_authority_sha256
            ),
            "inputs": {
                "campaign_spec": {
                    "path": str(spec_path),
                    "sha256": _sha256(spec_path),
                },
                "projection_contract": {
                    "path": str(projection_path),
                    "sha256": _sha256(projection_path),
                },
                "execution_envelope": {
                    "max_admitted_batch": request.max_admitted_batch,
                    "max_generations": request.max_generations,
                    "ticks_per_generation": request.ticks_per_generation,
                    "qualified_cache_width": qualified_cache_width,
                    "required_obligations": sorted(
                        request.required_obligations
                    ),
                    "reproduction_exposure_capacity": capacity.payload(),
                },
            },
            "artifacts": rows,
            "validation": {
                "candidate_loader": (
                    "InheritanceCrossPathCandidateArtifactLoaderV1"
                ),
                "cuda_module_imported": False,
                "four_artifact_chain_valid": True,
            },
        }
        _write_new(binding_path, binding_payload)
        directory_descriptor = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        if created_output:
            shutil.rmtree(output_dir)
        raise

    return CandidateArtifactPublicationV1(
        output_dir=output_dir,
        registry_path=registry_path,
        telemetry_contract_path=telemetry_path,
        binding_manifest_path=binding_path,
        candidate_authority_sha256=loader.candidate_authority_sha256,
        artifact_rows=rows,
    )


__all__ = [
    "BINDING_FILENAME",
    "CANDIDATE_BINDING_MANIFEST_SCHEMA_VERSION",
    "CandidateArtifactPublicationV1",
    "CandidateArtifactStagingRequestV1",
    "InheritanceCrossPathCandidateStagingError",
    "REGISTRY_FILENAME",
    "TELEMETRY_FILENAME",
    "stage_inheritance_cross_path_candidate_v1",
]
