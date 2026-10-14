"""Self-resolving front door for ``EIEL_SCIENCE_REQUEST_V1``.

The public surface deliberately accepts only a science request.  Machine and
route inputs are resolved from one sealed current-authority set, while the
existing Science Instrument materializer and Launchpad execution front door
remain the admission, execution, and artifact-lifecycle authorities.

This module does not grant Qualification authority.  It compiles request
semantics, selects exactly one compatible current machine binding, derives the
host context and Baseline, materializes all physical batches, and optionally
executes each batch through the canonical Launchpad path before aggregating the
compact reducer results.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Final

from emergence_contracts import ContentDigest, canonical_json_bytes, canonical_sha256
from emergence_contracts import ScienceWorldBundleV1
from emergence_contracts.science_request_v1 import (
    EielScienceRequestV1,
    validate_eiel_science_request_v1,
)
from engine.campaigns.artifact_lifecycle import (
    ArtifactLifecycleError,
    RunStorageGuard,
    StorageBudgetExceeded,
    build_artifact_disposition_manifest_v1,
    build_run_artifact_failure_receipt_v1,
    build_run_passport_v1,
    build_sealed_artifact_inventory_row_v1,
    build_storage_disposition_receipt_v1,
    ensure_lifecycle_contract_v1,
    register_pending_durable_archive_v1,
    run_artifact_transaction_fence_v1,
    validate_artifact_disposition_manifest_v1,
    validate_run_artifact_lifecycle_contract_v1,
    write_run_artifact_failure_receipt_v1,
    write_run_passport_v1,
    write_storage_disposition_receipt_v1,
)
from engine.campaigns.baseline_v1 import (
    BASELINE_PROFILE_ID,
    BaselineDeclarationV1,
    BaselineDeviationV1,
    load_baseline_profile_v1,
)
from engine.campaigns.cuda_full_evolution_v2_provider import (
    CUDA_FULL_EVOLUTION_V2_BASELINE_DEVIATION_REQUIREMENT_IDS_V1,
)
from engine.perf.cuda_full_evolution_v2.workhorse import (
    WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1,
    validate_workhorse_selected_world_observation_v1,
)
from engine.campaigns.orchestration import atomic_write_json
from engine.campaigns.interesting_checkpoints import reject_disposable_checkpoint_dependencies
from engine.native_runtime.cuda_snapshot_publication_v1 import (
    NativeSnapshotPublicationError,
    native_snapshot_implementation_is_native_v1,
    native_snapshot_implementation_v1,
)
from engine.native_runtime.cuda_snapshot_v1.default_binding import (
    resolve_snapshot_extension_path_v1,
    validate_resolved_snapshot_extension_v1,
)
from engine.native_runtime.cuda_snapshot_v1.native_loader import (
    NativeSnapshotLoadError,
)
import eiel_composition.science_cuda_v2_carrier_v1 as science_carrier
from eiel_composition.science_cuda_v2_carrier_v1 import (
    SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1,
    SCIENCE_CUDA_V2_OBSERVATION_OFF_V1,
    SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1,
    SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1,
    science_cuda_v2_carrier_profiles_for_world_v1,
)
from eiel_composition.science_cuda_v2_generic_host_context_v1 import (
    GenericCudaV2ScienceHostContextError,
    materialize_cuda_v2_explicit_bundle_host_context_v1,
)
from eiel_composition.science_instrument_artifacts_v1 import (
    serialize_science_host_context_v1,
)
from eiel_composition.science_instrument_operator_v1 import (
    SCIENCE_INSTRUMENT_EXECUTION_LINEAGE_SCHEMA_VERSION,
    SCIENCE_INSTRUMENT_OPERATOR_SCHEMA_VERSION,
    ScienceInstrumentOperatorError,
    execute_materialized_science_batch_v1,
    load_science_instrument_materialization_v1,
    materialize_explicit_science_world_bundle_v1,
    resolve_compact_reducer_tail_seconds_v1,
)
from eiel_composition.science_request_v1_compiler import (
    ScienceRequestV1CompilationV1,
    compile_science_request_v1,
)
from eiel_composition.science_workhorse_selected_host_context_v1 import (
    WorkhorseSelectedHostContextError,
    materialize_workhorse_selected_host_context_v1,
)
from emergence_evidence.science_campaign_atlas_v1 import (
    ScienceCampaignAtlasError,
    aggregate_science_campaign_atlas_v1,
)
from emergence_evidence.science_strategy_attribution_v1 import (
    StrategyAttributionError,
    strategy_attribution_required_for_bundle_v1,
)
from emergence_evidence.science_compact_world_atlas_v1 import (
    CompactWorldAtlasReducerError,
    CompactWorldAtlasIncrementalReducerV1,
    NATIVE_REDUCER_ARTIFACT_USE_DEFAULT_TECHNICAL_V1,
    _sealed_terminal_index_v1,
)
from emergence_evidence.native_compact_reducer_v1 import (
    default_native_compact_reducer_binding_v1,
    load_native_compact_reducer_v1,
)


SCIENCE_REQUEST_V1_FRONT_DOOR_SCHEMA_VERSION: Final[str] = (
    "eiel_science_request_v1_front_door_v1"
)
SCIENCE_EXECUTION_AUTHORITY_SCHEMA_VERSION: Final[str] = (
    "eiel_science_execution_authority_v1"
)
SCIENCE_EXECUTION_AUTHORITY_SET_SCHEMA_VERSION: Final[str] = (
    "eiel_science_execution_authority_set_v1"
)
MAX_SCIENCE_EXECUTION_AUTHORITIES_V1: Final[int] = 4096
SCIENCE_REQUEST_V1_PLAN_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_science_request_v1_plan_receipt_v1"
)
SCIENCE_REQUEST_V1_RUN_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_science_request_v1_run_receipt_v1"
)

MISSING_SCIENTIFIC_CAPABILITY: Final[str] = "MISSING_SCIENTIFIC_CAPABILITY"
UNRESOLVED_SCIENCE_AUTHORITY: Final[str] = "UNRESOLVED_SCIENCE_AUTHORITY"
AMBIGUOUS_SCIENCE_AUTHORITY: Final[str] = "AMBIGUOUS_SCIENCE_AUTHORITY"
INVALID_SCIENCE_AUTHORITY: Final[str] = "INVALID_SCIENCE_AUTHORITY"
SCIENCE_MATERIALIZATION_FAILED: Final[str] = "SCIENCE_MATERIALIZATION_FAILED"
SCIENCE_BATCH_EXECUTION_FAILED: Final[str] = "SCIENCE_BATCH_EXECUTION_FAILED"
SCIENCE_AGGREGATION_FAILED: Final[str] = "SCIENCE_AGGREGATION_FAILED"
WAIT_RESOURCE: Final[str] = "WAIT_RESOURCE"
WAIT_STORAGE: Final[str] = "WAIT_STORAGE"
SCIENCE_REQUEST_V1_RUN_JOURNAL_SCHEMA_VERSION: Final[str] = (
    "eiel_science_request_v1_run_journal_v1"
)

SCIENCE_AUTHORITY_ENVIRONMENT_VARIABLE: Final[str] = (
    "EIEL_SCIENCE_EXECUTION_AUTHORITY_V1"
)
def _configured_science_runtime_root_v1() -> Path:
    value = os.environ.get(
        "EIEL_SCIENCE_RUNTIME_ROOT", public_runtime_path('science-request-v1')
    )
    root = Path(value).expanduser()
    if not value.strip() or not root.is_absolute():
        raise ValueError("EIEL_SCIENCE_RUNTIME_ROOT must be a nonempty absolute path")
    return root


DEFAULT_SCIENCE_RUNTIME_ROOT: Final[Path] = _configured_science_runtime_root_v1()
DEFAULT_SCIENCE_AUTHORITY_PATHS: Final[tuple[Path, ...]] = (
    Path(public_runtime_path('authorities/eiel-science-execution-authority-v1.json')),
    Path("/etc/eiel/science-execution-authority-v1.json"),
)
DEFAULT_SCIENCE_AUTHORITY_INSTALL_PATH: Final[Path] = DEFAULT_SCIENCE_AUTHORITY_PATHS[0]
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


class ScienceRequestV1FrontDoorError(ValueError):
    """A stable fail-closed outcome at the public science boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})

    def to_result(self) -> dict[str, Any]:
        return {
            "schema_version": SCIENCE_REQUEST_V1_FRONT_DOOR_SCHEMA_VERSION,
            "ok": False,
            "status": self.code,
            "result_class": self.code,
            "request_sha256": self.details.get("request_sha256"),
            "runtime_root": self.details.get("runtime_root"),
            "error": str(self),
            "details": self.details,
            "execution_started": self.details.get("execution_started", False),
            "qualification_authority_effect": False,
        }


@dataclass(frozen=True, slots=True)
class ScienceExecutionAuthorityV1:
    """One exact, already-sealed machine/route binding selected for a request."""

    authority_id: str
    authority_sha256: str
    authority_source: str
    authority_source_file_sha256: str
    base_campaign_plan_path: Path
    base_campaign_plan_file_sha256: str
    base_campaign_plan_sha256: str
    baseline: Mapping[str, Any]
    baseline_profile_sha256: str
    science_context: Mapping[str, Any]
    available_physical_batch_capacity: int
    physical_batch_capacity: int
    temporal_k: int

    def __post_init__(self) -> None:
        if not _TOKEN.fullmatch(self.authority_id):
            raise ScienceRequestV1FrontDoorError(
                INVALID_SCIENCE_AUTHORITY,
                "science execution authority_id is invalid",
            )
        for field_name in (
            "authority_sha256",
            "authority_source_file_sha256",
            "base_campaign_plan_file_sha256",
            "base_campaign_plan_sha256",
            "baseline_profile_sha256",
        ):
            _sha256(getattr(self, field_name), field=field_name)
        if (
            type(self.available_physical_batch_capacity) is not int
            or self.available_physical_batch_capacity < 1
            or type(self.physical_batch_capacity) is not int
            or self.physical_batch_capacity < 1
            or self.physical_batch_capacity > self.available_physical_batch_capacity
            or type(self.temporal_k) is not int
            or self.temporal_k < 1
        ):
            raise ScienceRequestV1FrontDoorError(
                INVALID_SCIENCE_AUTHORITY,
                "science execution authority has invalid physical B/K",
            )

    def identity_projection(self) -> dict[str, Any]:
        context_digest = self.science_context.get("context_sha256")
        return {
            "authority_id": self.authority_id,
            "authority_sha256": self.authority_sha256,
            "authority_source": self.authority_source,
            "authority_source_file_sha256": self.authority_source_file_sha256,
            "base_campaign_plan": {
                "path": str(self.base_campaign_plan_path),
                "file_sha256": self.base_campaign_plan_file_sha256,
                "plan_sha256": self.base_campaign_plan_sha256,
            },
            "baseline": dict(self.baseline),
            "baseline_profile_sha256": self.baseline_profile_sha256,
            "science_context_sha256": context_digest,
            "available_physical_batch_capacity": (
                self.available_physical_batch_capacity
            ),
            "physical_batch_capacity": self.physical_batch_capacity,
            "temporal_k": self.temporal_k,
        }


ScienceAuthorityResolverV1 = Callable[
    [EielScienceRequestV1, ScienceRequestV1CompilationV1, Path],
    ScienceExecutionAuthorityV1,
]
ScienceBatchExecutorV1 = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class _LoadedScienceRequestV1:
    request: EielScienceRequestV1
    source_path: Path | None
    source_file_sha256: str


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            f"{field} is not a lowercase SHA-256 digest",
        )
    return value


def _sha256_digest_hex(value: object, *, field: str) -> str:
    if isinstance(value, Mapping):
        if value.get("algorithm") != "sha256":
            raise ScienceRequestV1FrontDoorError(
                INVALID_SCIENCE_AUTHORITY,
                f"{field} algorithm is not SHA-256",
            )
        value = value.get("hex")
    return _sha256(value, field=field)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, field: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            f"{field} must not be a symbolic link: {expanded}",
        )
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            f"{field} is not a regular file: {resolved}",
        )
    return resolved


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    resolved = _regular_file(path, field=field)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            f"{field} is unreadable JSON: {resolved}",
        ) from error
    if not isinstance(value, dict):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            f"{field} must contain a JSON object",
        )
    return value


def _load_request(
    request: Mapping[str, Any] | str | Path,
) -> _LoadedScienceRequestV1:
    if isinstance(request, Mapping):
        value = dict(request)
        source_path = None
        source_file_sha256 = canonical_sha256(value).hex
    else:
        try:
            source_path = _regular_file(Path(request), field="science request")
            value = _read_json(source_path, field="science request")
        except ScienceRequestV1FrontDoorError as error:
            raise ScienceRequestV1FrontDoorError(
                "INVALID_SCIENCE_REQUEST", str(error)
            ) from error
        source_file_sha256 = _file_sha256(source_path)
    try:
        reject_disposable_checkpoint_dependencies({"request_path": source_path})
        reject_disposable_checkpoint_dependencies(
            value, base_dir=None if source_path is None else source_path.parent,
        )
        normalized = validate_eiel_science_request_v1(value)
    except (TypeError, ValueError) as error:
        raise ScienceRequestV1FrontDoorError(
            "INVALID_SCIENCE_REQUEST",
            f"science request validation failed: {error}",
        ) from error
    return _LoadedScienceRequestV1(
        request=normalized,
        source_path=source_path,
        source_file_sha256=source_file_sha256,
    )


def _request_world(request: EielScienceRequestV1) -> dict[str, int]:
    value = request.to_dict().get("world")
    if not isinstance(value, Mapping):
        raise ScienceRequestV1FrontDoorError(
            "INVALID_SCIENCE_REQUEST", "normalized request lacks world geometry"
        )
    return {
        field: int(value[field])
        for field in ("width", "height", "population", "capacity")
    }


def _request_run(request: EielScienceRequestV1) -> dict[str, Any]:
    value = request.to_dict().get("run")
    if not isinstance(value, Mapping):
        raise ScienceRequestV1FrontDoorError(
            "INVALID_SCIENCE_REQUEST", "normalized request lacks run control"
        )
    return dict(value)


def _compilation_uses_selected_world_continuation_v1(
    compilation: ScienceRequestV1CompilationV1,
) -> bool:
    flags = tuple(
        isinstance(semantic := world.semantic_inputs.to_value(), Mapping)
        and isinstance(initialization := semantic.get("initialization"), Mapping)
        and initialization.get("schema_version")
        == "eiel_science_selected_world_initialization_v1"
        for world in compilation.world_bundle.worlds
    )
    if any(flags) and (not all(flags) or len(flags) != 1):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "V1 selected-world continuation must own exactly one Science Request world",
        )
    return bool(flags and flags[0])


def _is_selected_world_lowered_batch_v1(lowered: Mapping[str, Any]) -> bool:
    materialization = lowered.get("science_materialization")
    return bool(
        isinstance(materialization, Mapping)
        and materialization.get("schema_version")
        == "eiel_science_selected_world_technical_lowering_v1"
    )


def _materialized_world_ids_v1(science: object) -> list[str] | None:
    if not isinstance(science, Mapping):
        return None
    if (
        science.get("schema_version")
        == "eiel_science_selected_world_technical_lowering_v1"
    ):
        world_id = science.get("semantic_world_id")
        return [world_id] if isinstance(world_id, str) and world_id else None
    raw = science.get("semantic_world_ids")
    if not isinstance(raw, list) or any(
        not isinstance(item, str) or not item for item in raw
    ):
        return None
    return list(raw)


def _selected_world_request_v1(request: EielScienceRequestV1) -> bool:
    return isinstance(request.to_dict().get("continuation"), Mapping)


def _request_carrier_observation_mode_v1(
    request: EielScienceRequestV1,
) -> str:
    return (
        SCIENCE_CUDA_V2_OBSERVATION_OFF_V1
        if _selected_world_request_v1(request)
        else SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1
    )


def _check_scientific_capability(request: EielScienceRequestV1) -> None:
    """Require a catalogued carrier with the request's observation boundary.

    The public request remains generic over world dimensions.  Compatibility
    comes from the canonical carrier catalog (and its provider observation
    boundary), not from a route-local geometry table.  Installed authority
    selection is intentionally left to ``resolve_current_science_authority_v1``.
    """

    payload = request.to_dict()
    reduction = payload.get("reduce")
    configuration = (
        reduction.get("configuration")
        if isinstance(reduction, Mapping)
        else None
    )
    if isinstance(configuration, Mapping) and configuration.get(
        "require_direct_genotype_reproductive_attribution", False
    ):
        raise ScienceRequestV1FrontDoorError(
            "INVALID_SCIENCE_REQUEST",
            "the selected science carrier cannot provide literal locus/allele genotype identity",
            details={
                "request_sha256": request.request_sha256().to_dict(),
                "execution_started": False,
            },
        )
    world = _request_world(request)
    observation_mode = _request_carrier_observation_mode_v1(request)
    profiles = science_cuda_v2_carrier_profiles_for_world_v1(
        world, observation_mode=observation_mode
    )
    if profiles:
        return
    all_profiles = [
        {
            "carrier_profile_id": profile.carrier_profile_id,
            "world": {
                "width": profile.width,
                "height": profile.height,
                "population": profile.initial_population,
                "capacity": profile.capacity,
            },
            "provider_id": profile.provider_id,
            "observation_enabled": bool(
                science_cuda_v2_carrier_profiles_for_world_v1(
                    {
                        "width": profile.width,
                        "height": profile.height,
                        "population": profile.initial_population,
                        "capacity": profile.capacity,
                    },
                    observation_mode=observation_mode,
                )
            ),
        }
        for profile in science_carrier.SCIENCE_CUDA_V2_PROFILES
    ]
    raise ScienceRequestV1FrontDoorError(
        MISSING_SCIENTIFIC_CAPABILITY,
        "no installed observation-enabled science carrier matches the requested world",
        details={
            "request_sha256": request.request_sha256().to_dict(),
            "requested_world": world,
            "available_carrier_profiles": all_profiles,
            "missing_capabilities": [
                (
                    "workhorse_selected_world_host_boundary_observation_authority"
                    if observation_mode == SCIENCE_CUDA_V2_OBSERVATION_OFF_V1
                    else "observation_enabled_heterogeneous_compact_science_execution_authority"
                ),
                "compatible_science_carrier_for_requested_world",
            ],
            "failure_phase": "pre_execution_capability_resolution",
        },
    )


def _preflight_native_snapshot_binding_v1(
    base_plan: Mapping[str, Any],
) -> None:
    """Fail during planning when the selected native snapshot cannot load.

    The provider still performs the authoritative validation at execution
    time.  This earlier check makes the canonical science plan command expose
    a missing or stale source-bound artifact before materialization or GPU
    dispatch, while retaining the strict loader and its no-JIT behavior.
    """

    science_materialization = base_plan.get("science_materialization")
    observation_mode = (
        science_materialization.get("observation_mode")
        if isinstance(science_materialization, Mapping)
        else None
    )
    if observation_mode != SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1:
        return
    try:
        implementation = native_snapshot_implementation_v1()
    except NativeSnapshotPublicationError as error:
        raise ScienceRequestV1FrontDoorError(
            MISSING_SCIENTIFIC_CAPABILITY,
            f"native snapshot implementation is unavailable before execution: {error}",
            details={
                "failure_phase": "pre_execution_capability_resolution",
                "native_snapshot_implementation": None,
            },
        ) from error
    if not native_snapshot_implementation_is_native_v1(implementation):
        return

    try:
        artifact, manifest, _ = validate_resolved_snapshot_extension_v1()
    except (NativeSnapshotLoadError, OSError, ValueError) as initial_error:
        error: Exception = initial_error
        try:
            configured_path = str(resolve_snapshot_extension_path_v1())
        except (NativeSnapshotLoadError, OSError, ValueError) as resolution_error:
            configured_path = None
            error = resolution_error
        raise ScienceRequestV1FrontDoorError(
            MISSING_SCIENTIFIC_CAPABILITY,
            f"native snapshot extension is unavailable before execution: {error}",
            details={
                "failure_phase": "pre_execution_capability_resolution",
                "native_snapshot_implementation": implementation,
                "native_snapshot_extension_path": configured_path,
            },
        ) from error

    # Keep the successful identity available to callers only through the
    # validated artifact itself; no environment mutation or runtime loading is
    # performed by planning.
    if not artifact.is_file() or not manifest.is_file():
        raise ScienceRequestV1FrontDoorError(
            MISSING_SCIENTIFIC_CAPABILITY,
            "native snapshot extension validation returned non-files",
            details={
                "failure_phase": "pre_execution_capability_resolution",
                "native_snapshot_implementation": implementation,
                "native_snapshot_extension_path": str(artifact),
                "native_snapshot_manifest_path": str(manifest),
            },
        )


def _development_baseline(*, acs_checkpoint_slots: int | None = None) -> dict[str, Any]:
    """Build the route-bound development Baseline declaration.

    The ACS2 route conforms to the two-slot checkpoint requirement, so that
    requirement is not a deviation for an ACS2 plan.  Other routes retain the
    historical five-deviation development declaration.  The default keeps
    this helper compatible with synthetic/front-door fixtures that do not bind
    an ACS slot count; live authority resolution always supplies the sealed
    plan's explicit value.
    """

    if acs_checkpoint_slots is not None and (
        type(acs_checkpoint_slots) is not int
        or acs_checkpoint_slots not in {0, 1, 2}
    ):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science authority ACS checkpoint slots are invalid",
        )
    requirement_ids = tuple(
        requirement_id
        for requirement_id in CUDA_FULL_EVOLUTION_V2_BASELINE_DEVIATION_REQUIREMENT_IDS_V1
        if not (
            requirement_id == "checkpoint.acs_two_slot"
            and acs_checkpoint_slots == 2
        )
    )
    declaration = BaselineDeclarationV1(
        profile_id=BASELINE_PROFILE_ID,
        intent="development",
        deviations=tuple(
            BaselineDeviationV1(
                requirement_id=requirement_id,
                reason=(
                    "The current CUDA V2 native science carrier declares this exact "
                    "development-route Baseline delta; no Qualification authority is inferred."
                ),
            )
            for requirement_id in requirement_ids
        ),
    )
    return declaration.payload()


def _default_authority_sources() -> tuple[Path, ...]:
    configured = os.environ.get(SCIENCE_AUTHORITY_ENVIRONMENT_VARIABLE)
    if configured is not None:
        if not configured.strip():
            raise ScienceRequestV1FrontDoorError(
                UNRESOLVED_SCIENCE_AUTHORITY,
                f"{SCIENCE_AUTHORITY_ENVIRONMENT_VARIABLE} is empty",
            )
        return (Path(configured),)
    config_root = os.environ.get("XDG_CONFIG_HOME")
    user_path = (
        Path(config_root).expanduser() / "eiel/science-execution-authority-v1.json"
        if config_root
        else Path.home() / ".config/eiel/science-execution-authority-v1.json"
    )
    existing: list[Path] = []
    for candidate in (*DEFAULT_SCIENCE_AUTHORITY_PATHS, user_path):
        if candidate.exists() or candidate.is_symlink():
            existing.append(candidate)
    if not existing:
        raise ScienceRequestV1FrontDoorError(
            UNRESOLVED_SCIENCE_AUTHORITY,
            "no sealed current science execution authority is installed",
            details={
                "environment_pointer": SCIENCE_AUTHORITY_ENVIRONMENT_VARIABLE,
                "searched_paths": [
                    str(path) for path in (*DEFAULT_SCIENCE_AUTHORITY_PATHS, user_path)
                ],
            },
        )
    return tuple(existing)


def _workhorse_off_host_observation_authority_v1(
    base_plan: Mapping[str, Any],
    *,
    profile: Any,
) -> bool:
    """Recognize only the Workhorse OFF-runtime/host-observation capability."""

    if profile.provider_id != science_carrier.WORKHORSE_PROVIDER_ID:
        return False
    analysis = base_plan.get("analysis")
    extension = analysis.get("extension") if isinstance(analysis, Mapping) else None
    operation = (
        extension.get("contract") if isinstance(extension, Mapping) else None
    )
    execution = base_plan.get("execution")
    materialization = base_plan.get("science_materialization")
    return bool(
        isinstance(operation, Mapping)
        and operation.get("operation") in {"EXECUTION_ROW", "INDEXED_ACS_CONTINUATION"}
        and isinstance(execution, Mapping)
        and execution.get("native_streamed_checkpoint") is True
        and execution.get("acs_checkpoint_slots") == 2
        and (
            not isinstance(materialization, Mapping)
            or materialization.get("observation_mode")
            in {None, SCIENCE_CUDA_V2_OBSERVATION_OFF_V1}
        )
    )


def _science_authority_host_context_v1(
    base_path: Path,
    *,
    logical_world_capacity: int,
    context_scope_id: str,
):
    """Dispatch only the two existing CUDA V2 science authority contexts."""

    base_plan = _read_json(base_path, field="base_campaign_plan")
    workhorse_profiles = [
        profile
        for profile in science_carrier.SCIENCE_CUDA_V2_PROFILES
        if profile.provider_id == science_carrier.WORKHORSE_PROVIDER_ID
        and profile.native_profile_id == base_plan.get("resolved_profile")
    ]
    if len(workhorse_profiles) > 1:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "Workhorse base plan matches multiple carrier profiles",
        )
    if workhorse_profiles:
        try:
            return materialize_workhorse_selected_host_context_v1(
                base_plan,
                profile=workhorse_profiles[0],
                logical_world_capacity=logical_world_capacity,
                context_scope_id=context_scope_id,
            )
        except (WorkhorseSelectedHostContextError, TypeError, ValueError) as error:
            raise ScienceRequestV1FrontDoorError(
                INVALID_SCIENCE_AUTHORITY,
                f"selected-world Workhorse host context is invalid: {error}",
            ) from error
    return materialize_cuda_v2_explicit_bundle_host_context_v1(
        base_path,
        logical_world_capacity=logical_world_capacity,
        context_scope_id=context_scope_id,
    )


def activate_science_execution_authority_v1(
    base_campaign_plan: str | Path,
    *,
    authority_id: str,
    output_path: str | Path = DEFAULT_SCIENCE_AUTHORITY_INSTALL_PATH,
    append_to_set: bool = False,
) -> dict[str, Any]:
    """Atomically install one exact current Science Instrument authority.

    ``append_to_set`` uses the already-public authority-set envelope so a
    finite collection of exact absolute-tick base plans can coexist.  It is a
    bounded machine-administration write, not a new resolver or manager.
    """

    if _TOKEN.fullmatch(authority_id) is None:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY, "science authority_id is invalid"
        )
    base_path = _regular_file(Path(base_campaign_plan), field="base_campaign_plan")
    base_plan = _read_json(base_path, field="base_campaign_plan")
    plan_sha256 = base_plan.get("plan_sha256")
    if (
        base_plan.get("schema_version") != "eiel_campaign_plan_v1"
        or not isinstance(plan_sha256, str)
        or canonical_sha256(
            {key: value for key, value in base_plan.items() if key != "plan_sha256"}
        ).hex
        != plan_sha256
    ):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "base campaign plan is not an exact canonical campaign plan",
        )
    lifecycle = base_plan.get("artifact_lifecycle")
    if (
        not isinstance(lifecycle, Mapping)
        or lifecycle.get("lifecycle_profile") != "SCIENCE_COMPACT"
    ):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "base campaign plan does not bind SCIENCE_COMPACT lifecycle",
        )
    try:
        context = _science_authority_host_context_v1(
            base_path,
            logical_world_capacity=1,
            context_scope_id=(
                "eiel-science-authority-activation-"
                + canonical_sha256(
                    {"authority_id": authority_id, "plan_sha256": plan_sha256}
                ).hex[:24]
            ),
        )
    except (
        GenericCudaV2ScienceHostContextError,
        ScienceRequestV1FrontDoorError,
        TypeError,
        ValueError,
    ) as error:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            f"base campaign plan is not a current executable science route: {error}",
        ) from error
    physical = context.physical_execution_core.to_value()
    carrier_profile_id = physical.get("carrier_profile_id")
    profile = next(
        (
            candidate
            for candidate in science_carrier.SCIENCE_CUDA_V2_PROFILES
            if candidate.carrier_profile_id == carrier_profile_id
        ),
        None,
    )
    if profile is None:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "current science route does not bind a registered carrier profile",
            details={"carrier_profile_id": carrier_profile_id},
        )
    compact_profile = science_cuda_v2_carrier_profiles_for_world_v1(
        {
            "width": profile.width,
            "height": profile.height,
            "population": profile.initial_population,
            "capacity": profile.capacity,
        },
        observation_mode=SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1,
    )
    if not compact_profile and not _workhorse_off_host_observation_authority_v1(
        base_plan,
        profile=profile,
    ):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "current science route lacks a compact sidecar or the scoped "
            "Workhorse OFF-runtime host-observation capability",
            details={"carrier_profile_id": profile.carrier_profile_id},
        )
    core = {
        "schema_version": SCIENCE_EXECUTION_AUTHORITY_SCHEMA_VERSION,
        "authority_id": authority_id,
        "status": "CURRENT",
        "world": {
            "width": profile.width,
            "height": profile.height,
            "population": profile.initial_population,
            "capacity": profile.capacity,
        },
        "base_campaign_plan": {
            "path": str(base_path),
            "file_sha256": _file_sha256(base_path),
            "plan_sha256": plan_sha256,
        },
    }
    authority = {**core, "authority_sha256": canonical_sha256(core).hex}
    target = Path(output_path).expanduser()
    if target.is_symlink():
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            f"science authority install target must not be a symbolic link: {target}",
        )
    payload: dict[str, Any] = authority
    if append_to_set:
        retained: list[dict[str, Any]] = []
        if target.exists():
            source = _regular_file(target, field="science execution authority")
            for existing in _authority_rows(source):
                verified_existing, _existing_plan = _verified_authority_row(
                    existing,
                    source=source,
                )
                retained.append(verified_existing)
        same_id = [row for row in retained if row["authority_id"] == authority_id]
        if same_id and same_id != [authority]:
            raise ScienceRequestV1FrontDoorError(
                INVALID_SCIENCE_AUTHORITY,
                "science authority_id is already bound to another exact plan",
            )
        if not same_id:
            retained.append(authority)
        payload = _science_execution_authority_set_payload_v1(
            retained,
            source=target.absolute(),
        )
    try:
        atomic_write_json(target, payload)
    except (OSError, ValueError) as error:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            f"science authority activation failed: {error}",
        ) from error
    resolved = _regular_file(target, field="science execution authority")
    rows = _authority_rows(resolved)
    matching = [row for row in rows if row.get("authority_id") == authority_id]
    if len(matching) != 1:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "installed science authority verification differs",
        )
    verified, verified_plan = _verified_authority_row(matching[0], source=resolved)
    if verified_plan != base_path or verified != authority:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "installed science authority verification differs",
        )
    return {
        "schema_version": SCIENCE_REQUEST_V1_FRONT_DOOR_SCHEMA_VERSION,
        "ok": True,
        "status": (
            "SCIENCE_AUTHORITY_SET_MEMBER_ACTIVATED"
            if append_to_set
            else "SCIENCE_AUTHORITY_ACTIVATED"
        ),
        "authority_id": authority_id,
        "authority_sha256": authority["authority_sha256"],
        "authority_path": str(resolved),
        "authority_file_sha256": _file_sha256(resolved),
        "base_campaign_plan_sha256": plan_sha256,
        "authority_set_sha256": (
            payload.get("authority_set_sha256") if append_to_set else None
        ),
        "authority_count": len(rows),
        "qualification_authority_effect": False,
    }


def _science_execution_authority_set_payload_v1(
    authorities: Sequence[Mapping[str, Any]],
    *,
    source: Path,
) -> dict[str, Any]:
    """Seal one bounded existing-schema authority set in a single pass."""

    if not authorities or len(authorities) > MAX_SCIENCE_EXECUTION_AUTHORITIES_V1:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority set exceeds its bounded member limit"
            if authorities
            else "science execution authority set must not be empty",
            details={
                "authority_count": len(authorities),
                "maximum_authority_count": MAX_SCIENCE_EXECUTION_AUTHORITIES_V1,
            },
        )
    rows = sorted(
        (dict(row) for row in authorities),
        key=lambda row: str(row.get("authority_id")),
    )
    authority_ids: dict[str, str] = {}
    match_keys: dict[tuple[object, ...], str] = {}
    for row in rows:
        verified, row_plan_path = _verified_authority_row(row, source=source)
        authority_id = str(verified["authority_id"])
        prior_authority = authority_ids.get(authority_id)
        if prior_authority is not None:
            raise ScienceRequestV1FrontDoorError(
                INVALID_SCIENCE_AUTHORITY,
                "science execution authority set contains duplicate authority_id",
                details={"authority_id": authority_id},
            )
        authority_ids[authority_id] = str(verified["authority_sha256"])
        if verified.get("status") != "CURRENT":
            continue
        row_plan = _read_json(row_plan_path, field="base_campaign_plan")
        execution = row_plan.get("execution")
        if not isinstance(execution, Mapping):
            raise ScienceRequestV1FrontDoorError(
                INVALID_SCIENCE_AUTHORITY,
                "science authority base plan lacks execution control",
            )
        resource = row_plan.get("resource_estimate")
        physical_capacity = (
            resource.get("physical_batch_cells")
            if isinstance(resource, Mapping)
            else None
        )
        if type(physical_capacity) is not int or physical_capacity < 1:
            batch = execution.get("batch")
            physical_capacity = (
                batch.get("cells")
                if isinstance(batch, Mapping)
                else None
            )
        if type(physical_capacity) is not int or physical_capacity < 1:
            physical_capacity = None
        key = (
            canonical_sha256(verified["world"]).hex,
            execution.get("total_ticks"),
            canonical_sha256(execution.get("checkpoint_ticks")).hex,
            # Distinct sealed physical shapes are intentionally allowed to
            # coexist so the resolver can choose the largest exact partition
            # for each logical request.  Unknown capacity remains a single
            # ambiguity class and therefore still fails closed.
            physical_capacity,
        )
        prior = match_keys.get(key)
        if prior is not None and prior != verified["authority_sha256"]:
            raise ScienceRequestV1FrontDoorError(
                AMBIGUOUS_SCIENCE_AUTHORITY,
                "authority set would contain two distinct exact matches",
            )
        match_keys[key] = str(verified["authority_sha256"])
    set_core = {
        "schema_version": SCIENCE_EXECUTION_AUTHORITY_SET_SCHEMA_VERSION,
        "authorities": rows,
    }
    return {
        **set_core,
        "authority_set_sha256": canonical_sha256(set_core).hex,
    }


def activate_science_execution_authority_set_v1(
    authority_members: Sequence[tuple[str, str | Path]],
    *,
    output_path: str | Path,
    preserve_from: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Validate and install a finite authority set with one final write.

    Each member is prepared through the singleton activation path in a bounded
    task-local staging directory.  This keeps set construction linear for the
    finite tick inventories used by Expeditions and reuses the existing public
    authority-set schema and resolver.
    """

    members = tuple(authority_members)
    preserved_sources = tuple(preserve_from)
    if (
        (not members and not preserved_sources)
        or len(members) + len(preserved_sources)
        > MAX_SCIENCE_EXECUTION_AUTHORITIES_V1
    ):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority set exceeds its bounded member limit"
            if members or preserved_sources
            else "science execution authority set must not be empty",
            details={
                "authority_count": len(members) + len(preserved_sources),
                "maximum_authority_count": MAX_SCIENCE_EXECUTION_AUTHORITIES_V1,
            },
        )
    target = Path(output_path).expanduser()
    if target.is_symlink():
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            f"science authority install target must not be a symbolic link: {target}",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, Any]] = []
    for preserved_source_raw in preserved_sources:
        preserved_source = _regular_file(
            Path(preserved_source_raw),
            field="preserved science execution authority",
        )
        for row in _authority_rows(preserved_source):
            verified, _plan = _verified_authority_row(row, source=preserved_source)
            prepared.append(verified)
    with tempfile.TemporaryDirectory(
        prefix=".eiel-science-authority-set-",
        dir=target.parent,
    ) as staging_raw:
        staging = Path(staging_raw)
        for index, member in enumerate(members):
            if (
                not isinstance(member, tuple)
                or len(member) != 2
                or not isinstance(member[0], str)
            ):
                raise ScienceRequestV1FrontDoorError(
                    INVALID_SCIENCE_AUTHORITY,
                    "science authority member must be an (authority_id, base_plan) pair",
                )
            authority_id, base_plan = member
            member_path = staging / f"member-{index:04d}.json"
            activate_science_execution_authority_v1(
                base_plan,
                authority_id=authority_id,
                output_path=member_path,
            )
            prepared.extend(_authority_rows(member_path))
        if len(prepared) > MAX_SCIENCE_EXECUTION_AUTHORITIES_V1:
            raise ScienceRequestV1FrontDoorError(
                INVALID_SCIENCE_AUTHORITY,
                "science execution authority set exceeds its bounded member limit",
                details={
                    "authority_count": len(prepared),
                    "maximum_authority_count": MAX_SCIENCE_EXECUTION_AUTHORITIES_V1,
                },
            )
        payload = _science_execution_authority_set_payload_v1(
            prepared,
            source=target.absolute(),
        )
    try:
        atomic_write_json(target, payload)
    except (OSError, ValueError) as error:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            f"science authority activation failed: {error}",
        ) from error
    resolved = _regular_file(target, field="science execution authority")
    installed = _authority_rows(resolved)
    for row in installed:
        _verified_authority_row(row, source=resolved)
    return {
        "schema_version": SCIENCE_REQUEST_V1_FRONT_DOOR_SCHEMA_VERSION,
        "ok": True,
        "status": "SCIENCE_AUTHORITY_SET_ACTIVATED",
        "authority_ids_sha256": canonical_sha256(
            [row["authority_id"] for row in installed]
        ).hex,
        "authority_path": str(resolved),
        "authority_file_sha256": _file_sha256(resolved),
        "authority_set_sha256": payload["authority_set_sha256"],
        "authority_count": len(installed),
        "maximum_authority_count": MAX_SCIENCE_EXECUTION_AUTHORITIES_V1,
        "qualification_authority_effect": False,
    }


def _authority_rows(path: Path) -> tuple[dict[str, Any], ...]:
    payload = _read_json(path, field="science execution authority")
    schema = payload.get("schema_version")
    if schema == SCIENCE_EXECUTION_AUTHORITY_SCHEMA_VERSION:
        return (payload,)
    if schema != SCIENCE_EXECUTION_AUTHORITY_SET_SCHEMA_VERSION:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            f"unsupported science execution authority schema in {path}",
        )
    if set(payload) != {"schema_version", "authorities", "authority_set_sha256"}:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority set has invalid fields",
        )
    expected = _sha256(payload["authority_set_sha256"], field="authority_set_sha256")
    core = {
        key: value for key, value in payload.items() if key != "authority_set_sha256"
    }
    if canonical_sha256(core).hex != expected:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority set digest mismatch",
        )
    rows = payload.get("authorities")
    if (
        not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority set must contain authority objects",
        )
    if len(rows) > MAX_SCIENCE_EXECUTION_AUTHORITIES_V1:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority set exceeds its bounded member limit",
            details={
                "authority_count": len(rows),
                "maximum_authority_count": MAX_SCIENCE_EXECUTION_AUTHORITIES_V1,
            },
        )
    return tuple(dict(row) for row in rows)


def _resolve_reference_path(raw: object, *, source: Path, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY, f"{field}.path must be a non-empty string"
        )
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = source.parent / candidate
    return _regular_file(candidate, field=field)


def _verified_authority_row(
    row: Mapping[str, Any], *, source: Path
) -> tuple[dict[str, Any], Path]:
    value = dict(row)
    required = {
        "schema_version",
        "authority_id",
        "status",
        "world",
        "base_campaign_plan",
        "authority_sha256",
    }
    if set(value) != required or value.get("schema_version") != (
        SCIENCE_EXECUTION_AUTHORITY_SCHEMA_VERSION
    ):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority has invalid fields or schema",
        )
    expected = _sha256(value["authority_sha256"], field="authority_sha256")
    core = {key: item for key, item in value.items() if key != "authority_sha256"}
    if canonical_sha256(core).hex != expected:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority digest mismatch",
        )
    authority_id = value.get("authority_id")
    if not isinstance(authority_id, str) or _TOKEN.fullmatch(authority_id) is None:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority_id is invalid",
        )
    if value.get("status") not in {"CURRENT", "RETIRED"}:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority status is invalid",
        )
    world = value.get("world")
    if (
        not isinstance(world, Mapping)
        or set(world) != {"width", "height", "population", "capacity"}
        or any(type(item) is not int or item < 1 for item in world.values())
    ):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority world shape is invalid",
        )
    base_ref = value.get("base_campaign_plan")
    if not isinstance(base_ref, Mapping) or set(base_ref) != {
        "path",
        "file_sha256",
        "plan_sha256",
    }:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority base plan reference is invalid",
        )
    base_path = _resolve_reference_path(
        base_ref.get("path"), source=source, field="base_campaign_plan"
    )
    file_digest = _sha256(
        base_ref.get("file_sha256"), field="base_campaign_plan.file_sha256"
    )
    if _file_sha256(base_path) != file_digest:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority base plan file digest mismatch",
        )
    plan = _read_json(base_path, field="base_campaign_plan")
    plan_digest = _sha256(
        base_ref.get("plan_sha256"), field="base_campaign_plan.plan_sha256"
    )
    if plan.get("plan_sha256") != plan_digest:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority base plan identity mismatch",
        )
    return value, base_path


def resolve_current_science_authority_v1(
    request: EielScienceRequestV1,
    compilation: ScienceRequestV1CompilationV1,
    repo_root: Path,
) -> ScienceExecutionAuthorityV1:
    """Resolve exactly one compatible sealed machine binding or fail closed."""

    del repo_root
    _check_scientific_capability(request)
    requested_world = _request_world(request)
    observation_mode = _request_carrier_observation_mode_v1(request)
    run = _request_run(request)
    expected_ticks = int(run["ticks"])
    expected_checkpoints = list(run["checkpoint_ticks"])
    batch_plan = compilation.nominal_batches
    requested_batch_capacity = max(len(batch["world_ids"]) for batch in batch_plan)

    candidates: list[tuple[dict[str, Any], Path, Path]] = []
    for source_candidate in _default_authority_sources():
        source = _regular_file(source_candidate, field="science execution authority")
        for row in _authority_rows(source):
            verified, base_path = _verified_authority_row(row, source=source)
            if verified.get("status") != "CURRENT":
                continue
            if verified.get("world") != requested_world:
                continue
            plan = _read_json(base_path, field="base_campaign_plan")
            execution = plan.get("execution")
            if not isinstance(execution, Mapping):
                raise ScienceRequestV1FrontDoorError(
                    INVALID_SCIENCE_AUTHORITY,
                    "science execution base plan lacks execution control",
                )
            if (
                execution.get("total_ticks") != expected_ticks
                or execution.get("checkpoint_ticks") != expected_checkpoints
            ):
                continue
            candidates.append((verified, base_path, source))

    if not candidates:
        raise ScienceRequestV1FrontDoorError(
            UNRESOLVED_SCIENCE_AUTHORITY,
            "no current science execution authority exactly matches the request",
            details={
                "requested_world": requested_world,
                "total_ticks": expected_ticks,
                "checkpoint_ticks": expected_checkpoints,
                "compatible_carrier_profiles": [
                    profile.carrier_profile_id
                    for profile in science_cuda_v2_carrier_profiles_for_world_v1(
                        requested_world,
                        observation_mode=observation_mode,
                    )
                ],
            },
        )

    # A current authority binds an exact physical B.  More than one sealed B
    # may legitimately be installed for the same world/horizon (for example a
    # fast B64 shape and a B16 tail shape).  Select the largest exact shape
    # that partitions this request's logical world set; if equally suitable
    # authorities remain, fail closed rather than guessing between distinct
    # machine identities.  This is capability selection, not scientific
    # policy, and it keeps nominal request batching independent of hardware.
    context_scope_id = "eiel-science-request-v1-" + request.request_sha256().hex[:24]
    probed: list[
        tuple[dict[str, Any], Path, Path, Any, int]
    ] = []
    probe_errors: list[Exception] = []
    for candidate_authority, candidate_base, candidate_source in candidates:
        try:
            candidate_context = _science_authority_host_context_v1(
                candidate_base,
                logical_world_capacity=1,
                context_scope_id=context_scope_id,
            )
            available = candidate_context.route_adapter.maximum_batch_cells
            if type(available) is not int or available < 1:
                raise ScienceRequestV1FrontDoorError(
                    INVALID_SCIENCE_AUTHORITY,
                    "science carrier exposed an invalid physical capacity",
                )
            probed.append(
                (
                    candidate_authority,
                    candidate_base,
                    candidate_source,
                    candidate_context,
                    available,
                )
            )
        except (
            GenericCudaV2ScienceHostContextError,
            ScienceRequestV1FrontDoorError,
            TypeError,
            ValueError,
        ) as error:
            probe_errors.append(error)
    if not probed:
        identities = {row[0]["authority_sha256"] for row in candidates}
        if len(identities) != 1:
            # Preserve the existing fail-closed ambiguity outcome when the
            # candidate rows themselves cannot be probed to compare capacity.
            raise ScienceRequestV1FrontDoorError(
                AMBIGUOUS_SCIENCE_AUTHORITY,
                "multiple distinct current science execution authorities match the request",
                details={"matching_authority_sha256s": sorted(identities)},
            )
        error = probe_errors[0]
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            f"current science authority cannot derive an exact host context: {error}",
        ) from error

    logical_world_count = compilation.world_bundle.world_count
    exact = [
        row for row in probed if logical_world_count % row[4] == 0
    ]
    if not exact:
        raise ScienceRequestV1FrontDoorError(
            MISSING_SCIENTIFIC_CAPABILITY,
            "no current science authority provides an exact physical partition for the request",
            details={
                "logical_world_count": logical_world_count,
                "available_physical_batch_capacities": sorted(
                    {row[4] for row in probed}
                ),
            },
        )
    maximum_capacity = max(row[4] for row in exact)
    matching = [row for row in exact if row[4] == maximum_capacity]
    identities = {row[0]["authority_sha256"] for row in matching}
    if len(identities) != 1:
        raise ScienceRequestV1FrontDoorError(
            AMBIGUOUS_SCIENCE_AUTHORITY,
            "multiple distinct current science execution authorities match the request",
            details={"matching_authority_sha256s": sorted(identities)},
        )
    authority, base_path, source, probe_context, available_physical_batch_capacity = matching[0]
    matched_base_plan = _read_json(base_path, field="base_campaign_plan")
    matched_execution = matched_base_plan.get("execution")
    if not isinstance(matched_execution, Mapping):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution base plan lacks execution control",
        )
    matched_acs_checkpoint_slots = matched_execution.get("acs_checkpoint_slots", 0)
    if type(matched_acs_checkpoint_slots) is not int or matched_acs_checkpoint_slots not in {
        0,
        1,
        2,
    }:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution base plan ACS checkpoint slots are invalid",
        )
    selected_workhorse_authority = any(
        profile.provider_id == science_carrier.WORKHORSE_PROVIDER_ID
        and profile.native_profile_id == matched_base_plan.get("resolved_profile")
        for profile in science_carrier.SCIENCE_CUDA_V2_PROFILES
    )
    if selected_workhorse_authority and (
        requested_batch_capacity != 1
        or observation_mode != SCIENCE_CUDA_V2_OBSERVATION_OFF_V1
        or not _compilation_uses_selected_world_continuation_v1(compilation)
    ):
        raise ScienceRequestV1FrontDoorError(
            MISSING_SCIENTIFIC_CAPABILITY,
            "Workhorse authority is scoped to singleton exact selected-world "
            "continuation with observation OFF",
        )
    try:
        # ``run.batch`` is a nominal ownership/packing preference emitted by
        # the Science Request compiler, not a claim that one physical launch
        # must carry that many worlds.  Resolve the sealed carrier's actual
        # capacity first, then let the existing carrier adapter partition each
        # nominal batch to that capacity.  This keeps physical B an authority
        # property and avoids adding a frozen machine limit to the public
        # Expedition/Science Request contract.
        physical_batch_capacity = available_physical_batch_capacity
        context = (
            probe_context
            if physical_batch_capacity == 1
            else _science_authority_host_context_v1(
                base_path,
                logical_world_capacity=physical_batch_capacity,
                context_scope_id=context_scope_id,
            )
        )
    except (
        GenericCudaV2ScienceHostContextError,
        ScienceRequestV1FrontDoorError,
        TypeError,
        ValueError,
    ) as error:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            f"current science authority cannot derive an exact host context: {error}",
        ) from error
    fixed = context.route_adapter.fixed_flat_controls.to_value()
    temporal_k = fixed.get("cuda_native.temporal_k")
    if type(temporal_k) is not int or temporal_k < 1:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "derived science context lacks an exact native temporal_k",
        )
    base_ref = authority["base_campaign_plan"]
    return ScienceExecutionAuthorityV1(
        authority_id=str(authority["authority_id"]),
        authority_sha256=str(authority["authority_sha256"]),
        authority_source=str(source),
        authority_source_file_sha256=_file_sha256(source),
        base_campaign_plan_path=base_path,
        base_campaign_plan_file_sha256=str(base_ref["file_sha256"]),
        base_campaign_plan_sha256=str(base_ref["plan_sha256"]),
        baseline=_development_baseline(
            acs_checkpoint_slots=matched_acs_checkpoint_slots
        ),
        baseline_profile_sha256=str(load_baseline_profile_v1()["profile_sha256"]),
        science_context=serialize_science_host_context_v1(context),
        available_physical_batch_capacity=available_physical_batch_capacity,
        physical_batch_capacity=physical_batch_capacity,
        temporal_k=temporal_k,
    )


def _revalidate_science_authority_v1(
    authority: ScienceExecutionAuthorityV1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Take one final content-bound snapshot before creating or reusing output."""

    source = _regular_file(
        Path(authority.authority_source), field="science execution authority"
    )
    if _file_sha256(source) != authority.authority_source_file_sha256:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science execution authority source changed after resolution",
        )
    matches: list[tuple[dict[str, Any], Path]] = []
    for row in _authority_rows(source):
        verified, base_path = _verified_authority_row(row, source=source)
        if (
            verified.get("status") == "CURRENT"
            and verified.get("authority_sha256") == authority.authority_sha256
        ):
            matches.append((verified, base_path))
    if len(matches) != 1:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "resolved science authority is no longer exactly current",
        )
    row, base_path = matches[0]
    base_ref = row["base_campaign_plan"]
    if (
        base_path != authority.base_campaign_plan_path
        or base_ref.get("file_sha256") != authority.base_campaign_plan_file_sha256
        or base_ref.get("plan_sha256") != authority.base_campaign_plan_sha256
    ):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "resolved base campaign plan binding changed after resolution",
        )
    base_plan = _read_json(base_path, field="base_campaign_plan")
    canonical_plan_sha256 = canonical_sha256(
        {key: value for key, value in base_plan.items() if key != "plan_sha256"}
    ).hex
    if (
        _file_sha256(base_path) != authority.base_campaign_plan_file_sha256
        or base_plan.get("plan_sha256") != authority.base_campaign_plan_sha256
        or canonical_plan_sha256 != authority.base_campaign_plan_sha256
    ):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "resolved base campaign plan changed or is not canonically sealed",
        )
    baseline_profile = load_baseline_profile_v1()
    if baseline_profile.get("profile_sha256") != authority.baseline_profile_sha256:
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "current Baseline profile changed after authority resolution",
        )
    context = dict(authority.science_context)
    declared_context_sha256 = context.pop("context_sha256", None)
    if declared_context_sha256 != canonical_sha256(context).to_dict():
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "resolved science context is not canonically sealed",
        )
    return base_plan, {**context, "context_sha256": declared_context_sha256}


def _materialization_identifier(
    request: EielScienceRequestV1,
    compilation: ScienceRequestV1CompilationV1,
    authority: ScienceExecutionAuthorityV1,
) -> str:
    return canonical_sha256(
        {
            "schema_version": "eiel_science_request_v1_materialization_identity_v1",
            "request_sha256": request.request_sha256().to_dict(),
            "compilation_receipt_sha256": compilation.compilation_receipt[
                "receipt_sha256"
            ],
            "bundle_content_sha256": compilation.world_bundle.bundle_content_sha256().to_dict(),
            "authority_sha256": authority.authority_sha256,
            "authority_source_file_sha256": authority.authority_source_file_sha256,
            "base_campaign_plan_sha256": authority.base_campaign_plan_sha256,
            "context_sha256": authority.science_context["context_sha256"],
            "baseline": dict(authority.baseline),
            "baseline_profile_sha256": authority.baseline_profile_sha256,
        }
    ).hex


def _write_immutable_json(path: Path, value: Mapping[str, Any], *, field: str) -> Path:
    encoded = canonical_json_bytes(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            f"{field} already exists and will not be overwritten: {path}",
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path


def _read_verified_plan_receipt(path: Path) -> dict[str, Any]:
    value = _read_json(path, field="science plan receipt")
    if value.get("schema_version") != SCIENCE_REQUEST_V1_PLAN_RECEIPT_SCHEMA_VERSION:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "existing science materialization has an unsupported plan receipt",
        )
    declared = value.get("receipt_sha256")
    core = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if declared != canonical_sha256(core).to_dict():
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "existing science plan receipt digest mismatch",
        )
    return value


def _science_coordinator_storage_allowance_v1(
    compilation: ScienceRequestV1CompilationV1,
) -> int:
    """Bound run-local coordinator material independently of child outputs."""

    return max(16 * 1024**2, compilation.world_bundle.world_count * 64 * 1024)


def _aggregate_science_run_lifecycle_v1(
    base_lifecycle: Mapping[str, Any],
    lowered_batches: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    coordinator_durable_allowance_bytes: int,
) -> dict[str, Any]:
    """Cover every retained child while preserving an already-sufficient base."""

    if (
        type(coordinator_durable_allowance_bytes) is not int
        or coordinator_durable_allowance_bytes < 0
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "science coordinator storage allowance is invalid",
        )
    base = validate_run_artifact_lifecycle_contract_v1(base_lifecycle)
    if not lowered_batches:
        return base
    child_contracts: list[dict[str, Any]] = []
    for index, lowered in enumerate(lowered_batches):
        if not isinstance(lowered, Mapping):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_MATERIALIZATION_FAILED,
                f"lowered science batch {index} is invalid",
            )
        try:
            child = ensure_lifecycle_contract_v1(lowered)
        except ArtifactLifecycleError as error:
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_MATERIALIZATION_FAILED,
                f"lowered science batch {index} lacks a valid lifecycle contract",
            ) from error
        if child["lifecycle_profile"] != base["lifecycle_profile"]:
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_MATERIALIZATION_FAILED,
                f"lowered science batch {index} lifecycle profile differs",
            )
        child_contracts.append(child)

    base_storage = base["storage"]
    scratch_required = sum(
        int(child["storage"]["scratch_max_bytes"]) for child in child_contracts
    )
    checkpoint_required = sum(
        int(child["storage"]["checkpoint_max_bytes"]) for child in child_contracts
    )
    durable_required = coordinator_durable_allowance_bytes + sum(
        int(child["storage"]["durable_max_bytes"]) for child in child_contracts
    )
    emergency_required = max(
        int(base_storage["emergency_reserve_bytes"]),
        *(
            int(child["storage"]["emergency_reserve_bytes"])
            for child in child_contracts
        ),
    )
    storage = {
        "scratch_max_bytes": max(
            int(base_storage["scratch_max_bytes"]), scratch_required
        ),
        "checkpoint_max_bytes": max(
            int(base_storage["checkpoint_max_bytes"]), checkpoint_required
        ),
        "durable_max_bytes": max(
            int(base_storage["durable_max_bytes"]), durable_required
        ),
        "emergency_reserve_bytes": emergency_required,
    }
    storage["total_max_bytes"] = max(
        int(base_storage["total_max_bytes"]),
        sum(storage.values()),
    )
    if storage == dict(base_storage):
        return base
    candidate = {**base, "storage": storage}
    candidate.pop("contract_sha256", None)
    return validate_run_artifact_lifecycle_contract_v1(candidate)


def _science_lifecycle_plan_v1(
    *,
    root: Path,
    phase: str,
    loaded: _LoadedScienceRequestV1,
    compilation: ScienceRequestV1CompilationV1,
    authority: ScienceExecutionAuthorityV1,
    base_plan: Mapping[str, Any],
    lowered_batches: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
    coordinator_durable_allowance_bytes: int = 0,
) -> dict[str, Any]:
    lifecycle = base_plan.get("artifact_lifecycle")
    execution = base_plan.get("execution")
    if not isinstance(lifecycle, Mapping) or not isinstance(execution, Mapping):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science base plan lacks execution or lifecycle authority",
        )
    bound_lifecycle = (
        dict(lifecycle)
        if lowered_batches is None
        else _aggregate_science_run_lifecycle_v1(
            lifecycle,
            lowered_batches,
            coordinator_durable_allowance_bytes=coordinator_durable_allowance_bytes,
        )
    )
    bound_execution = dict(execution)
    if bound_lifecycle != dict(lifecycle):
        budgets = bound_execution.get("budgets")
        rebound_budgets = dict(budgets) if isinstance(budgets, Mapping) else {}
        rebound_budgets["disk_bytes"] = max(
            int(rebound_budgets.get("disk_bytes", 0)),
            int(bound_lifecycle["storage"]["total_max_bytes"]),
        )
        bound_execution["budgets"] = rebound_budgets
    core = {
        "schema_version": "eiel_science_request_v1_lifecycle_plan_v1",
        "campaign_id": str(loaded.request.to_dict()["campaign_id"]),
        "phase": phase,
        "resolved_profile": base_plan.get("resolved_profile"),
        "execution": bound_execution,
        "artifact_lifecycle": bound_lifecycle,
        "output": {"runtime_root": str(root)},
        "cells": [],
        "science_request": {
            "request_sha256": loaded.request.request_sha256().to_dict(),
            "compilation_receipt_sha256": compilation.compilation_receipt[
                "receipt_sha256"
            ],
            "bundle_content_sha256": (
                compilation.world_bundle.bundle_content_sha256().to_dict()
            ),
            "authority_sha256": authority.authority_sha256,
            "base_campaign_plan_sha256": authority.base_campaign_plan_sha256,
        },
        "qualification_authority_effect": False,
    }
    plan = {**core, "plan_sha256": canonical_sha256(core).hex}
    ensure_lifecycle_contract_v1(plan)
    return plan


def _science_lifecycle_source_identity_v1(
    *,
    loaded: _LoadedScienceRequestV1,
    compilation: ScienceRequestV1CompilationV1,
    authority: ScienceExecutionAuthorityV1,
) -> dict[str, Any]:
    return {
        "request_sha256": loaded.request.request_sha256().to_dict(),
        "request_source_file_sha256": loaded.source_file_sha256,
        "compilation_receipt_sha256": compilation.compilation_receipt["receipt_sha256"],
        "bundle_content_sha256": (
            compilation.world_bundle.bundle_content_sha256().to_dict()
        ),
        "authority_sha256": authority.authority_sha256,
        "authority_source_file_sha256": authority.authority_source_file_sha256,
        "base_campaign_plan_file_sha256": (authority.base_campaign_plan_file_sha256),
        "base_campaign_plan_sha256": authority.base_campaign_plan_sha256,
        "baseline_profile_sha256": authority.baseline_profile_sha256,
    }


def _write_science_lifecycle_start_v1(
    *,
    root: Path,
    plan: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    provider_id: str,
) -> None:
    lifecycle = ensure_lifecycle_contract_v1(plan)
    _write_immutable_json(
        root / "science_lifecycle_plan.json",
        plan,
        field="science lifecycle plan",
    )
    _write_immutable_json(
        root / "artifact_lifecycle_contract.json",
        lifecycle,
        field="science artifact lifecycle contract",
    )
    passport = build_run_passport_v1(
        plan,
        run_root=root,
        terminal_status="active",
        source_identity=source_identity,
        provider_identity={
            "provider_id": provider_id,
            "profile_id": plan.get("resolved_profile"),
            "execution_route_sha256": None,
        },
        config_identity=canonical_sha256(source_identity).hex,
        seed_identity=canonical_sha256(
            {"request_sha256": source_identity.get("request_sha256")}
        ).hex,
        disposition_manifest_path=str(root / "artifact_disposition_manifest.json"),
        disposition_receipt_path=str(root / "storage_disposition_receipt.json"),
    )
    write_run_passport_v1(root / "run_passport.json", passport)


def _science_inventory_binding_v1(relative: str) -> dict[str, str]:
    if "/reducer-artifacts/" in f"/{relative}":
        return {
            "artifact_family": "science.compact.hereditary_attribution",
            "retention_class": "R4",
            "owner_writer": "science.compact.world_atlas.v1",
            "release_predicate": "science audit hold released",
        }
    if relative == "campaign_atlas_result.json":
        return {
            "artifact_family": "science.compact.campaign_atlas",
            "retention_class": "R4",
            "owner_writer": "eiel-science-request-v1-aggregator",
            "release_predicate": "science audit hold released",
        }
    if relative.startswith("batch_result_"):
        return {
            "artifact_family": "science.compact.batch_result",
            "retention_class": "R4",
            "owner_writer": "eiel-science-request-v1-orchestrator",
            "release_predicate": "campaign aggregate verified and audit hold released",
        }
    if relative in {
        "world_bundle.json",
        "compilation_bundle.json",
        "carrier_execution_plan.json",
    } or relative.startswith("lowered_batch_"):
        return {
            "artifact_family": "science.executable_materialization",
            "retention_class": "R3",
            "owner_writer": "eiel-science-instrument-v1",
            "release_predicate": "all bound executions and recovery obligations closed",
        }
    return {
        "artifact_family": "lifecycle.control",
        "retention_class": "R5",
        "owner_writer": "eiel-science-request-v1-front-door",
        "release_predicate": "terminal science and lifecycle authority retained",
    }


def _exact_child_lifecycle_file_v1(
    attempt_root: Path,
    declared_path: object,
    *,
    filename: str,
    field: str,
) -> Path:
    if not isinstance(declared_path, str) or not declared_path:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"{field} path is absent",
        )
    raw = Path(declared_path).expanduser()
    candidate = raw if raw.is_absolute() else attempt_root / raw
    expected = (attempt_root / filename).resolve()
    if candidate.is_symlink() or candidate.resolve() != expected:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"{field} path differs from its exact attempt-owned location",
        )
    return _regular_file(expected, field=field)


def _authenticated_child_disposition_v1(
    campaign_root: Path,
    journal_row: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    batch_index = journal_row.get("batch_index")
    if type(batch_index) is not int or batch_index < 0:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "completed science child has an invalid batch identity",
        )
    attempt_root = _journal_attempt_root_v1(
        campaign_root,
        journal_row,
        batch_index=batch_index,
    )
    lifecycle = journal_row.get("child_lifecycle")
    if not isinstance(lifecycle, Mapping):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"completed science child {batch_index} lacks lifecycle authority",
        )
    marker_path = _exact_child_lifecycle_file_v1(
        attempt_root,
        lifecycle.get("terminal_lifecycle_commit_path"),
        filename="terminal_lifecycle_commit.json",
        field=f"science child {batch_index} terminal lifecycle commit",
    )
    manifest_path = _exact_child_lifecycle_file_v1(
        attempt_root,
        lifecycle.get("artifact_disposition_manifest_path"),
        filename="artifact_disposition_manifest.json",
        field=f"science child {batch_index} artifact disposition manifest",
    )
    storage_path = _exact_child_lifecycle_file_v1(
        attempt_root,
        lifecycle.get("storage_disposition_receipt_path"),
        filename="storage_disposition_receipt.json",
        field=f"science child {batch_index} storage disposition receipt",
    )
    marker = _read_json(marker_path, field="science child terminal lifecycle commit")
    marker_core = dict(marker)
    marker_sha256 = marker_core.pop("commit_sha256", None)
    manifest_ref = marker.get("manifest")
    storage_ref = marker.get("storage_receipt")
    if (
        marker.get("schema_version") != "eiel_execution_terminal_lifecycle_commit_v1"
        or marker_sha256 != canonical_sha256(marker_core).hex
        or lifecycle.get("terminal_lifecycle_commit_sha256") != marker_sha256
        or lifecycle.get("terminal_lifecycle_commit_file_sha256")
        != _file_sha256(marker_path)
        or not isinstance(manifest_ref, Mapping)
        or not isinstance(storage_ref, Mapping)
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science child {batch_index} terminal lifecycle identity differs",
        )
    try:
        manifest = validate_artifact_disposition_manifest_v1(
            _read_json(
                manifest_path,
                field="science child artifact disposition manifest",
            )
        )
    except ArtifactLifecycleError as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science child {batch_index} disposition inventory identity is invalid",
        ) from error
    if (
        manifest.get("runtime_root") != str(attempt_root)
        or manifest.get("plan_sha256") != marker.get("plan_sha256")
        or manifest.get("terminal_status") != "completed"
        or manifest.get("inventory", {}).get("mode") != "producer_declared"
        or manifest.get("inventory", {}).get("complete") is not True
        or manifest_ref.get("path") != manifest_path.name
        or manifest_ref.get("file_sha256") != _file_sha256(manifest_path)
        or manifest_ref.get("manifest_sha256") != manifest.get("manifest_sha256")
        or lifecycle.get("artifact_disposition_manifest_sha256")
        != manifest.get("manifest_sha256")
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science child {batch_index} disposition inventory is not exact and complete",
        )
    storage = _read_json(
        storage_path, field="science child storage disposition receipt"
    )
    storage_core = dict(storage)
    storage_sha256 = storage_core.pop("receipt_sha256", None)
    lifecycle_storage_sha256 = lifecycle.get("storage_disposition_receipt_sha256")
    marker_storage_sha256 = storage_ref.get("receipt_sha256")
    marker_storage_file_sha256 = storage_ref.get("file_sha256")
    current_storage_file_sha256 = _file_sha256(storage_path)
    successor_after_terminal_commit = lifecycle.get(
        "storage_receipt_successor_after_terminal_commit"
    )
    marker_receipt_is_current = storage_sha256 == marker_storage_sha256
    storage_transition_is_exact = (
        marker_receipt_is_current
        and lifecycle_storage_sha256 == storage_sha256
        and successor_after_terminal_commit is False
        and current_storage_file_sha256 == marker_storage_file_sha256
    ) or (
        not marker_receipt_is_current
        and (
            (
                lifecycle_storage_sha256 == storage_sha256
                and successor_after_terminal_commit is True
            )
            or (
                lifecycle_storage_sha256 == marker_storage_sha256
                and successor_after_terminal_commit is False
            )
        )
    )
    if (
        storage.get("schema_version") != "eiel_storage_disposition_receipt_v1"
        or storage_sha256 != canonical_sha256(storage_core).hex
        or storage.get("manifest_sha256") != manifest.get("manifest_sha256")
        or storage.get("plan_sha256") != manifest.get("plan_sha256")
        or storage.get("campaign_id") != manifest.get("campaign_id")
        or storage.get("run_id") != manifest.get("run_id")
        or storage.get("lifecycle_contract_sha256")
        != manifest.get("lifecycle_contract_sha256")
        or storage_ref.get("path") != storage_path.name
        or not isinstance(marker_storage_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", marker_storage_sha256) is None
        or not isinstance(marker_storage_file_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", marker_storage_file_sha256) is None
        or not storage_transition_is_exact
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science child {batch_index} storage disposition is not authenticated",
        )
    return attempt_root, manifest, storage


def _child_release_category_v1(artifact: Mapping[str, Any]) -> str:
    checkpoint_role = artifact.get("checkpoint_role")
    family = str(artifact.get("artifact_family") or "").lower()
    retention_class = str(artifact.get("retention_class") or "RX")
    if checkpoint_role == "intermediate":
        return "intermediate_checkpoints"
    if "developmental" in family and "trace" in family:
        return "developmental_traces"
    if "event" in family:
        return "event_jsonl"
    if "trace" in family or "debug" in family:
        return "debug_traces"
    if retention_class == "R2":
        return "reproducible_derived"
    return "scratch"


def _exact_child_inventory_path_v1(value: Any, *, index: int) -> str:
    if not isinstance(value, str):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science child inventory row {index} path is malformed",
        )
    normalized = value.replace("\\", "/").strip("/")
    child_path = Path(normalized)
    if (
        not normalized
        or value != normalized
        or child_path.is_absolute()
        or ".." in child_path.parts
        or child_path.as_posix() != normalized
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science child inventory row {index} path is not canonical and relative",
        )
    return normalized


def _authenticated_child_released_paths_v1(
    attempt_root: Path,
    manifest: Mapping[str, Any],
    storage: Mapping[str, Any],
) -> set[str]:
    """Return only child objects proven absent by exact lifecycle release.

    A producer-declared manifest is an inventory of what the child produced,
    not a promise that every release-eligible transient still exists after its
    own terminal lifecycle worker has run.  The outer campaign may therefore
    omit an absent R0--R2 object only when the authenticated child storage
    receipt accounts for the exact manifest row.  Durable, ambiguous, and
    merely missing objects remain fail-closed in the normal inventory walk.
    """

    removed = storage.get("removed")
    if not isinstance(removed, list) or any(
        not isinstance(row, Mapping) for row in removed
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "science child storage disposition removed rows are malformed",
        )
    manifest_rows: dict[str, Mapping[str, Any]] = {}
    for index, raw_artifact in enumerate(manifest.get("artifacts", [])):
        if not isinstance(raw_artifact, Mapping):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science child inventory row {index} is malformed",
            )
        canonical_path = _exact_child_inventory_path_v1(
            raw_artifact.get("path"),
            index=index,
        )
        if canonical_path in manifest_rows:
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science child inventory row {index} duplicates an object path",
            )
        manifest_rows[canonical_path] = raw_artifact
    removed_paths: set[str] = set()
    released_bytes = 0
    for index, raw in enumerate(removed):
        row = dict(raw)
        relative = str(row.get("path") or "").replace("\\", "/").strip("/")
        child_path = Path(relative)
        if (
            not relative
            or child_path.is_absolute()
            or ".." in child_path.parts
            or relative in removed_paths
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science child release row {index} has an invalid or duplicate path",
            )
        artifact = manifest_rows.get(relative)
        if (
            artifact is None
            or artifact.get("exact_object_identity") is not True
            or artifact.get("release_eligible") is not True
            or artifact.get("disposition") != "RELEASE"
            or artifact.get("retention_class") not in {"R0", "R1", "R2"}
            or artifact.get("artifact_type") not in {"file", "sealed_tree"}
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science child release row {index} exceeds manifest authority",
            )
        digest_name = (
            "sha256" if artifact.get("artifact_type") == "file" else "tree_sha256"
        )
        expected = {
            "path": relative,
            "bytes": artifact.get("bytes"),
            "category": _child_release_category_v1(artifact),
            "retention_class": artifact.get("retention_class"),
            "checkpoint_role": artifact.get("checkpoint_role"),
            "release_predicate": artifact.get("release_predicate"),
            "artifact_type": artifact.get("artifact_type"),
            digest_name: artifact.get(digest_name),
        }
        if row != expected:
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science child release row {index} differs from its exact manifest row",
            )
        candidate = attempt_root / child_path
        try:
            candidate.resolve(strict=False).relative_to(attempt_root.resolve())
        except ValueError as error:
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science child release row {index} escapes its attempt root",
            ) from error
        if candidate.exists() or candidate.is_symlink():
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science child release row {index} still resolves after claimed release",
            )
        removed_paths.add(relative)
        released_bytes += int(artifact.get("bytes", 0))

    summary = manifest.get("summary")
    if not isinstance(summary, Mapping):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "science child disposition summary is malformed",
        )
    produced_bytes = int(summary.get("produced_bytes", -1))
    unresolved_bytes = int(summary.get("unresolved_rx_bytes", -1))
    if (
        storage.get("bytes_before") != produced_bytes
        or storage.get("bytes_released") != released_bytes
        or storage.get("bytes_retained") != produced_bytes - released_bytes
        or storage.get("bytes_unresolved_RX") != unresolved_bytes
        or (storage.get("action_failures") not in (None, []))
        or (
            removed_paths
            and storage.get("result")
            not in {
                "applied",
                "release_applied",
                "release_applied_archive_pending",
                "release_applied_archive_pending_retry",
            }
        )
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "science child release receipt byte accounting or result differs",
        )
    return removed_paths


def _prefixed_child_inventory_rows_v1(
    campaign_root: Path,
    attempt_root: Path,
    manifest: Mapping[str, Any],
    storage: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[str], tuple[str, ...]]:
    attempt_relative = str(attempt_root.relative_to(campaign_root)).replace("\\", "/")
    released_paths = _authenticated_child_released_paths_v1(
        attempt_root,
        manifest,
        storage,
    )
    rows: list[dict[str, Any]] = []
    covered_files: set[str] = set()
    covered_trees: list[str] = []
    for index, artifact in enumerate(manifest.get("artifacts", [])):
        if not isinstance(artifact, Mapping):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science child inventory row {index} is malformed",
            )
        child_relative = _exact_child_inventory_path_v1(
            artifact.get("path"),
            index=index,
        )
        child_path = Path(child_relative)
        artifact_type = artifact.get("artifact_type")
        if (
            not child_relative
            or child_path.is_absolute()
            or ".." in child_path.parts
            or artifact.get("exact_object_identity") is not True
            or artifact_type not in {"file", "sealed_tree"}
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science child inventory row {index} is not an exact object",
            )
        candidate = attempt_root / child_path
        try:
            candidate.resolve().relative_to(attempt_root)
        except ValueError as error:
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science child inventory row {index} escapes its attempt root",
            ) from error
        if child_relative in released_paths:
            continue
        if (
            candidate.is_symlink()
            or (artifact_type == "file" and not candidate.is_file())
            or (artifact_type == "sealed_tree" and not candidate.is_dir())
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science child inventory row {index} no longer resolves exactly",
            )
        try:
            observed = build_sealed_artifact_inventory_row_v1(
                run_root=attempt_root,
                path=candidate,
                artifact_family="science.child.outer-closeout-verification",
                retention_class="R5",
                owner_writer="science-request-v1-front-door",
            )
        except (ArtifactLifecycleError, OSError) as error:
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science child inventory row {index} could not be reverified",
            ) from error
        digest_name = "sha256" if artifact_type == "file" else "tree_sha256"
        if (
            observed.get("path") != child_relative
            or observed.get("artifact_type") != artifact_type
            or observed.get("bytes") != artifact.get("bytes")
            or observed.get(digest_name) != artifact.get(digest_name)
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science child inventory row {index} content identity differs",
            )
        relative = f"{attempt_relative}/{child_relative}"
        translated = {
            key: artifact[key]
            for key in (
                "artifact_family",
                "artifact_type",
                "owner_writer",
                "bytes",
                "retention_class",
                "decision_evidence_class",
                "replacement_artifact_identity",
                "live_reference_status",
                "audit_hold_status",
                "qualification_hold_status",
                "release_predicate",
                "reason",
                "checkpoint_role",
                "checkpoint_identity",
                "checkpoint_run_association",
                "evidence_certificate_id",
                "evidence_certificate_sha256",
                "evidence_class",
                "certificate_disposition",
                "downstream_consumers",
                "last_use",
            )
            if key in artifact
        }
        translated["path"] = relative
        if artifact_type == "file":
            translated["sha256"] = artifact.get("sha256")
            covered_files.add(relative)
        else:
            translated["tree_sha256"] = artifact.get("tree_sha256")
            covered_trees.append(relative.rstrip("/") + "/")
        rows.append(translated)
    return rows, covered_files, tuple(sorted(covered_trees))


def _is_child_lifecycle_wrapper_v1(relative: str) -> bool:
    if relative in {
        "artifact_disposition_manifest.json",
        "storage_disposition_receipt.json",
        "terminal_lifecycle_commit.json",
        "execution_result.json",
        "compact_reducer_execution_receipt.json",
        "batch_result.json",
    }:
        return True
    return (
        re.fullmatch(
            r"\.eiel-runtime-archive-release-v1/[0-9a-f]{64}/journal(?:\.json|\.lock)",
            relative,
        )
        is not None
    )


def _collect_science_lifecycle_inventory_v1(root: Path) -> list[dict[str, Any]]:
    excluded = {
        "artifact_disposition_manifest.json",
        "storage_disposition_receipt.json",
    }
    rows: list[dict[str, Any]] = []
    covered_files: set[str] = set()
    covered_trees: tuple[str, ...] = ()
    managed_attempts: dict[str, Path] = {}
    journal_path = root / "run_journal.json"
    if journal_path.is_file() and not journal_path.is_symlink():
        journal = _read_science_run_journal_v1(root)
        require_child_lifecycle = bool(
            journal.get("identity", {}).get("require_launchpad_lifecycle")
        )
        imported_trees: list[str] = []
        for journal_row in journal["batches"]:
            if journal_row.get("state") != "COMPLETED":
                continue
            child_lifecycle = journal_row.get("child_lifecycle")
            if child_lifecycle is None and not require_child_lifecycle:
                continue
            attempt_root, child_manifest, child_storage = (
                _authenticated_child_disposition_v1(root, journal_row)
            )
            imported, child_files, child_trees = _prefixed_child_inventory_rows_v1(
                root,
                attempt_root,
                child_manifest,
                child_storage,
            )
            attempt_relative = str(attempt_root.relative_to(root)).replace("\\", "/")
            managed_attempts[attempt_relative] = attempt_root
            rows.extend(imported)
            covered_files.update(child_files)
            imported_trees.extend(child_trees)
        covered_trees = tuple(sorted(imported_trees))
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        if path.is_symlink():
            raise ArtifactLifecycleError(
                f"science lifecycle inventory refuses symbolic link: {path}"
            )
        if not path.is_file():
            continue
        relative = str(path.relative_to(root)).replace("\\", "/")
        if relative in excluded:
            continue
        if relative in covered_files or any(
            relative.startswith(prefix) for prefix in covered_trees
        ):
            continue
        for attempt_relative in managed_attempts:
            prefix = attempt_relative.rstrip("/") + "/"
            if relative.startswith(prefix):
                child_relative = relative[len(prefix) :]
                if not _is_child_lifecycle_wrapper_v1(child_relative):
                    raise ScienceRequestV1FrontDoorError(
                        SCIENCE_BATCH_EXECUTION_FAILED,
                        "authenticated science child contains an object absent from "
                        f"its complete producer inventory: {relative}",
                    )
                break
        rows.append(
            build_sealed_artifact_inventory_row_v1(
                run_root=root,
                path=path,
                **_science_inventory_binding_v1(relative),
            )
        )
    return rows


def _publish_science_lifecycle_closeout_v1(
    *,
    root: Path,
    plan: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    provider_id: str,
    terminal_status: str,
    inventory_complete: bool,
    register_archive: bool,
) -> dict[str, Any]:
    manifest_path = root / "artifact_disposition_manifest.json"
    storage_path = root / "storage_disposition_receipt.json"
    final_passport = build_run_passport_v1(
        plan,
        run_root=root,
        terminal_status=terminal_status,
        source_identity=source_identity,
        provider_identity={
            "provider_id": provider_id,
            "profile_id": plan.get("resolved_profile"),
            "execution_route_sha256": None,
        },
        config_identity=canonical_sha256(source_identity).hex,
        seed_identity=canonical_sha256(
            {"request_sha256": source_identity.get("request_sha256")}
        ).hex,
        disposition_manifest_path=str(manifest_path),
        disposition_receipt_path=str(storage_path),
    )
    write_run_passport_v1(root / "run_passport.json", final_passport)
    inventory = _collect_science_lifecycle_inventory_v1(root)
    lifecycle = ensure_lifecycle_contract_v1(plan)
    disposition = build_artifact_disposition_manifest_v1(
        plan,
        run_root=root,
        terminal_status=terminal_status,
        live_reference_status=("none" if terminal_status == "completed" else "active"),
        audit_hold_status=(
            "active"
            if lifecycle["retention"]["audit_hold_binding"] == "required"
            else "not_required"
        ),
        qualification_hold_status=(
            "active"
            if lifecycle["retention"]["qualification_hold_binding"] == "required"
            else "not_required"
        ),
        artifact_inventory=inventory,
        inventory_complete=inventory_complete,
        max_objects=max(4096, len(inventory)),
    )
    produced_bytes = int(disposition["summary"]["produced_bytes"])
    storage = build_storage_disposition_receipt_v1(
        manifest=disposition,
        bytes_before=produced_bytes,
        bytes_retained=produced_bytes,
        bytes_released=0,
        bytes_unresolved_rx=int(disposition["summary"]["unresolved_rx_bytes"]),
        authorization_reason="dry_run_only_no_destructive_cleanup_authority",
        retention_references=[str(manifest_path)],
        plan_sha256=str(plan["plan_sha256"]),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        result="dry_run",
    )
    with run_artifact_transaction_fence_v1(root):
        _write_immutable_json(
            manifest_path,
            disposition,
            field="science artifact disposition manifest",
        )
        write_storage_disposition_receipt_v1(storage_path, storage)
        pointer = (
            register_pending_durable_archive_v1(
                manifest_path, _transaction_fence_held=True
            )
            if register_archive and terminal_status == "completed"
            else None
        )
    validated = validate_artifact_disposition_manifest_v1(
        _read_json(manifest_path, field="science artifact disposition manifest")
    )
    return {
        "contract_path": str(root / "artifact_lifecycle_contract.json"),
        "contract_sha256": lifecycle["contract_sha256"],
        "run_passport_path": str(root / "run_passport.json"),
        "run_passport_sha256": final_passport["passport_sha256"],
        "artifact_disposition_manifest_path": str(manifest_path),
        "artifact_disposition_manifest_sha256": validated["manifest_sha256"],
        "storage_disposition_receipt_path": str(storage_path),
        "storage_disposition_receipt_sha256": storage["receipt_sha256"],
        "durable_archive_pointer_path": None if pointer is None else str(pointer),
        "inventory_complete": validated["inventory"]["complete"],
        "terminal_status": validated["terminal_status"],
    }


def _record_science_lifecycle_failure_v1(
    *,
    root: Path,
    plan: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    provider_id: str,
    stage: str,
    failure_type: str,
    message: str,
) -> dict[str, Any]:
    failure = build_run_artifact_failure_receipt_v1(
        plan,
        run_root=root,
        stage=stage,
        failure_type=failure_type,
        message=message[:4096],
        status="failed",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_run_artifact_failure_receipt_v1(
        root / "run_artifact_failure_receipt.json", failure
    )
    return _publish_science_lifecycle_closeout_v1(
        root=root,
        plan=plan,
        source_identity=source_identity,
        provider_id=provider_id,
        terminal_status="failed",
        inventory_complete=True,
        register_archive=False,
    )


def _verify_science_lifecycle_closeout_v1(
    *, root: Path, expected_plan_sha256: str
) -> dict[str, Any]:
    manifest_path = root / "artifact_disposition_manifest.json"
    storage_path = root / "storage_disposition_receipt.json"
    passport_path = root / "run_passport.json"
    manifest = validate_artifact_disposition_manifest_v1(
        _read_json(manifest_path, field="science artifact disposition manifest")
    )
    storage = _read_json(storage_path, field="science storage disposition receipt")
    storage_core = dict(storage)
    supplied_storage_sha256 = storage_core.pop("receipt_sha256", None)
    passport = _read_json(passport_path, field="science run passport")
    passport_core = dict(passport)
    supplied_passport_sha256 = passport_core.pop("passport_sha256", None)
    if (
        manifest.get("runtime_root") != str(root)
        or manifest.get("plan_sha256") != expected_plan_sha256
        or manifest.get("terminal_status") != "completed"
        or manifest.get("inventory", {}).get("complete") is not True
        or storage.get("manifest_sha256") != manifest.get("manifest_sha256")
        or storage.get("plan_sha256") != expected_plan_sha256
        or supplied_storage_sha256 != canonical_sha256(storage_core).hex
        or passport.get("plan_sha256") != expected_plan_sha256
        or passport.get("terminal_status") != "completed"
        or supplied_passport_sha256 != canonical_sha256(passport_core).hex
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "science lifecycle closeout is incomplete or identity-inconsistent",
        )
    return {
        "contract_path": str(root / "artifact_lifecycle_contract.json"),
        "run_passport_path": str(passport_path),
        "run_passport_sha256": supplied_passport_sha256,
        "artifact_disposition_manifest_path": str(manifest_path),
        "artifact_disposition_manifest_sha256": manifest["manifest_sha256"],
        "storage_disposition_receipt_path": str(storage_path),
        "storage_disposition_receipt_sha256": supplied_storage_sha256,
        "inventory_complete": True,
        "terminal_status": "completed",
    }


def _verify_materialized_batch_ownership_v1(
    *,
    compilation: ScienceRequestV1CompilationV1,
    materialization: Mapping[str, Any] | tuple[dict[str, Any], ...],
) -> tuple[tuple[str, ...], ...]:
    raw_lowered = (
        materialization.get("lowered_batches")
        if isinstance(materialization, Mapping)
        else materialization
    )
    expected = tuple(
        tuple(str(world_id) for world_id in row["world_ids"])
        for row in compilation.nominal_batches
    )
    if not isinstance(raw_lowered, (list, tuple)):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "physical materialization does not expose a batch sequence",
        )
    expected_world_ids = [world_id for batch in expected for world_id in batch]
    observed: list[tuple[str, ...]] = []
    observed_world_ids: list[str] = []
    for index, lowered in enumerate(raw_lowered):
        science = (
            lowered.get("science_materialization")
            if isinstance(lowered, Mapping)
            else None
        )
        world_ids = _materialized_world_ids_v1(science)
        if (
            not isinstance(world_ids, list)
            or any(not isinstance(value, str) or not value for value in world_ids)
            or not world_ids
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_MATERIALIZATION_FAILED,
                f"physical batch {index} has invalid world ownership",
            )
        observed_batch = tuple(world_ids)
        observed.append(observed_batch)
        observed_world_ids.extend(observed_batch)
    # Nominal request batches are an ownership projection only.  A sealed
    # carrier may faithfully partition one nominal row into several physical
    # launches (or use a different profile grouping) without changing which
    # semantic worlds the request owns.  Authenticate the exact world set and
    # reject loss, duplication, or injection while leaving physical packing to
    # the carrier authority.
    if sorted(observed_world_ids) != sorted(expected_world_ids):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "physical materialization world ownership differs from the request",
        )
    return tuple(observed)


def validate_science_request_v1(
    request: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Validate and compile request semantics without resolving a runtime route."""

    loaded = _load_request(request)
    try:
        compilation = compile_science_request_v1(loaded.request)
    except (TypeError, ValueError) as error:
        raise ScienceRequestV1FrontDoorError(
            "INVALID_SCIENCE_REQUEST",
            f"science request compilation failed: {error}",
            details={
                "request_sha256": loaded.request.request_sha256().to_dict(),
                "execution_started": False,
            },
        ) from error
    return {
        "schema_version": SCIENCE_REQUEST_V1_FRONT_DOOR_SCHEMA_VERSION,
        "ok": True,
        "status": "VALID",
        "result_class": "SCIENCE_REQUEST_VALIDATION",
        "request_sha256": loaded.request.request_sha256().to_dict(),
        "request_source": {
            "path": None if loaded.source_path is None else str(loaded.source_path),
            "file_sha256": loaded.source_file_sha256,
        },
        "campaign_id": loaded.request.to_dict()["campaign_id"],
        "logical_world_count": compilation.world_bundle.world_count,
        "nominal_batch_plan": compilation.compilation_receipt["nominal_batch_plan"],
        "compilation_receipt": compilation.compilation_receipt,
        "authority_resolution_performed": False,
        "runtime_effects_performed": False,
        "execution_started": False,
        "qualification_authority_effect": False,
    }


def _plan_loaded_request(
    loaded: _LoadedScienceRequestV1,
    compilation: ScienceRequestV1CompilationV1,
    *,
    repo_root: Path,
    materialization_parent: Path,
    authority_resolver: ScienceAuthorityResolverV1,
) -> tuple[dict[str, Any], ScienceExecutionAuthorityV1, Path]:
    _check_scientific_capability(loaded.request)
    authority = authority_resolver(loaded.request, compilation, repo_root)
    base_plan, science_context = _revalidate_science_authority_v1(authority)
    selected_world_continuation = (
        _compilation_uses_selected_world_continuation_v1(compilation)
    )
    # Founder campaigns use the authenticated compact sidecar route and must
    # preflight its native snapshot binding.  Selected-world continuation uses
    # the existing OFF host-observation boundary instead; requiring the sidecar
    # extension there would incorrectly make exact continuation depend on an
    # observation implementation it does not consume.
    if not selected_world_continuation:
        _preflight_native_snapshot_binding_v1(base_plan)
    observation_mode = (
        SCIENCE_CUDA_V2_OBSERVATION_OFF_V1
        if selected_world_continuation
        else SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1
    )
    science_use_class = (
        SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1
        if selected_world_continuation
        else SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1
    )
    identifier = _materialization_identifier(loaded.request, compilation, authority)
    root = materialization_parent.expanduser().resolve() / identifier
    lifecycle_plan = _science_lifecycle_plan_v1(
        root=root,
        phase="materialization",
        loaded=loaded,
        compilation=compilation,
        authority=authority,
        base_plan=base_plan,
    )
    lifecycle_source = _science_lifecycle_source_identity_v1(
        loaded=loaded,
        compilation=compilation,
        authority=authority,
    )
    receipt_path = root / "science_request_plan_receipt.json"
    if root.exists() or root.is_symlink():
        if root.is_symlink() or not root.is_dir():
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_MATERIALIZATION_FAILED,
                f"science materialization target is not a regular directory: {root}",
            )
        receipt = _read_verified_plan_receipt(receipt_path)
        if (
            receipt.get("request_sha256") != loaded.request.request_sha256().to_dict()
            or receipt.get("authority") != authority.identity_projection()
            or receipt.get("lifecycle", {}).get("plan_sha256")
            != lifecycle_plan["plan_sha256"]
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_MATERIALIZATION_FAILED,
                "existing science materialization identity differs",
            )
        _, materialization_receipt, _, lowered_batches = (
            load_science_instrument_materialization_v1(root)
        )
        identities = materialization_receipt.get("identities")
        if (
            not isinstance(identities, Mapping)
            or identities.get("bundle_content_sha256")
            != compilation.world_bundle.bundle_content_sha256().to_dict()
            or identities.get("base_campaign_plan_sha256")
            != authority.base_campaign_plan_sha256
            or receipt.get("compilation_receipt", {}).get("receipt_sha256")
            != compilation.compilation_receipt["receipt_sha256"]
            or materialization_receipt.get("observation_mode") != observation_mode
            or materialization_receipt.get("science_use_class")
            != science_use_class
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_MATERIALIZATION_FAILED,
                "existing science materialization differs from the current compilation",
            )
        _verify_materialized_batch_ownership_v1(
            compilation=compilation,
            materialization=lowered_batches,
        )
        resolved_inputs = receipt.get("resolved_inputs")
        if not isinstance(resolved_inputs, Mapping):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_MATERIALIZATION_FAILED,
                "existing science materialization lacks resolved input snapshots",
            )
        for name, logical_sha256 in (
            ("base_campaign_plan", authority.base_campaign_plan_sha256),
            ("science_context", authority.science_context["context_sha256"]),
        ):
            raw = resolved_inputs.get(name)
            if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
                raise ScienceRequestV1FrontDoorError(
                    SCIENCE_MATERIALIZATION_FAILED,
                    f"existing materialization lacks {name} snapshot identity",
                )
            snapshot = _regular_file(Path(raw["path"]), field=f"resolved {name}")
            if (
                snapshot.parent != root
                or _file_sha256(snapshot) != raw.get("file_sha256")
                or raw.get("logical_sha256") != logical_sha256
            ):
                raise ScienceRequestV1FrontDoorError(
                    SCIENCE_MATERIALIZATION_FAILED,
                    f"existing materialization {name} snapshot changed",
                )
        lifecycle_closeout = _verify_science_lifecycle_closeout_v1(
            root=root, expected_plan_sha256=str(lifecycle_plan["plan_sha256"])
        )
        return (
            {**receipt, "reused": True, "lifecycle_closeout": lifecycle_closeout},
            authority,
            root,
        )

    run = _request_run(loaded.request)
    lifecycle = base_plan.get("artifact_lifecycle")
    if not isinstance(lifecycle, Mapping):
        raise ScienceRequestV1FrontDoorError(
            INVALID_SCIENCE_AUTHORITY,
            "science base plan lacks a lifecycle contract",
        )
    planned_bytes = _science_coordinator_storage_allowance_v1(compilation)
    guard: RunStorageGuard | None = None
    lifecycle_started = False
    try:
        guard = RunStorageGuard(
            lifecycle,
            run_root=root,
            planned_bytes=planned_bytes,
            reservation_window_bytes=planned_bytes,
        )
        storage_receipt = guard.enforce(stage="science_request_v1_materialization")
        _write_science_lifecycle_start_v1(
            root=root,
            plan=lifecycle_plan,
            source_identity=lifecycle_source,
            provider_id="eiel-science-request-v1-materializer",
        )
        lifecycle_started = True
        base_snapshot = _write_immutable_json(
            root / "resolved_base_campaign_plan.json",
            base_plan,
            field="resolved base campaign plan snapshot",
        )
        context_snapshot = _write_immutable_json(
            root / "resolved_science_context.json",
            science_context,
            field="resolved science context snapshot",
        )
        materialization = materialize_explicit_science_world_bundle_v1(
            compilation.world_bundle.to_dict(),
            context_snapshot,
            base_snapshot,
            baseline=authority.baseline,
            output_directory=root,
            maximum_total_ticks=int(run["ticks"]),
            physical_batch_capacity=authority.physical_batch_capacity,
            temporal_k=authority.temporal_k,
            observation_mode=observation_mode,
            science_use_class=science_use_class,
        )
        lineage = materialization.get("lineage")
        base_lineage = (
            lineage.get("base_campaign_plan") if isinstance(lineage, Mapping) else None
        )
        context_lineage = (
            lineage.get("host_context") if isinstance(lineage, Mapping) else None
        )
        if (
            not isinstance(base_lineage, Mapping)
            or base_lineage.get("path") != str(base_snapshot)
            or base_lineage.get("file_sha256") != _file_sha256(base_snapshot)
            or base_lineage.get("plan_sha256") != authority.base_campaign_plan_sha256
            or not isinstance(context_lineage, Mapping)
            or context_lineage.get("path") != str(context_snapshot)
            or context_lineage.get("file_sha256") != _file_sha256(context_snapshot)
            or materialization.get("identities", {}).get("base_campaign_plan_sha256")
            != authority.base_campaign_plan_sha256
            or materialization.get("identities", {}).get("context_sha256")
            != authority.science_context["context_sha256"]
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_MATERIALIZATION_FAILED,
                "science materialization lineage differs from resolved authority",
            )
        _verify_materialized_batch_ownership_v1(
            compilation=compilation,
            materialization=materialization,
        )
    except StorageBudgetExceeded as error:
        if lifecycle_started:
            _record_science_lifecycle_failure_v1(
                root=root,
                plan=lifecycle_plan,
                source_identity=lifecycle_source,
                provider_id="eiel-science-request-v1-materializer",
                stage="materialization-storage",
                failure_type="WAIT_STORAGE",
                message=str(error),
            )
        raise ScienceRequestV1FrontDoorError(
            "WAIT_STORAGE",
            "science materialization is waiting at the canonical storage boundary",
            details={"storage_receipt": error.receipt},
        ) from error
    except (
        ArtifactLifecycleError,
        ScienceInstrumentOperatorError,
        ScienceRequestV1FrontDoorError,
        OSError,
    ) as error:
        if lifecycle_started:
            try:
                _record_science_lifecycle_failure_v1(
                    root=root,
                    plan=lifecycle_plan,
                    source_identity=lifecycle_source,
                    provider_id="eiel-science-request-v1-materializer",
                    stage="materialization",
                    failure_type=type(error).__name__,
                    message=str(error),
                )
            except (ArtifactLifecycleError, OSError, ScienceRequestV1FrontDoorError):
                pass
        if isinstance(error, ScienceRequestV1FrontDoorError):
            raise
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            f"science request materialization failed: {error}",
            details={"materialization_root": str(root)},
        ) from error
    finally:
        if guard is not None:
            try:
                guard.release()
            except ArtifactLifecycleError as error:
                if lifecycle_started:
                    try:
                        _record_science_lifecycle_failure_v1(
                            root=root,
                            plan=lifecycle_plan,
                            source_identity=lifecycle_source,
                            provider_id="eiel-science-request-v1-materializer",
                            stage="materialization-storage-release",
                            failure_type=type(error).__name__,
                            message=str(error),
                        )
                    except (
                        ArtifactLifecycleError,
                        OSError,
                        ScienceRequestV1FrontDoorError,
                    ):
                        pass
                raise ScienceRequestV1FrontDoorError(
                    SCIENCE_MATERIALIZATION_FAILED,
                    "science materialization storage reservation could not be released",
                    details={"materialization_root": str(root)},
                ) from error

    core: dict[str, Any] = {
        "schema_version": SCIENCE_REQUEST_V1_PLAN_RECEIPT_SCHEMA_VERSION,
        "front_door_schema_version": SCIENCE_REQUEST_V1_FRONT_DOOR_SCHEMA_VERSION,
        "ok": True,
        "status": "ADMITTED_EXECUTABLE_MATERIALIZATION",
        "result_class": "SCIENCE_EXECUTABLE_MATERIALIZATION",
        "campaign_id": loaded.request.to_dict()["campaign_id"],
        "request_sha256": loaded.request.request_sha256().to_dict(),
        "request_source": {
            "path": None if loaded.source_path is None else str(loaded.source_path),
            "file_sha256": loaded.source_file_sha256,
        },
        "authority": authority.identity_projection(),
        "resolved_inputs": {
            "base_campaign_plan": {
                "path": str(base_snapshot),
                "file_sha256": _file_sha256(base_snapshot),
                "logical_sha256": authority.base_campaign_plan_sha256,
                "authority_source_file_sha256": (
                    authority.base_campaign_plan_file_sha256
                ),
            },
            "science_context": {
                "path": str(context_snapshot),
                "file_sha256": _file_sha256(context_snapshot),
                "logical_sha256": authority.science_context["context_sha256"],
            },
            "baseline_profile_sha256": authority.baseline_profile_sha256,
        },
        "compilation_receipt": compilation.compilation_receipt,
        "materialization": {
            "root": str(root),
            "receipt_path": materialization["receipt_path"],
            "receipt_file_sha256": materialization["receipt_file_sha256"],
            "receipt_sha256": materialization["receipt_sha256"],
            "bundle_admission": {
                "outcome": materialization["bundle_admission"]["outcome"],
                "artifact": materialization["artifacts"]["bundle_admission"],
                "bundle_admission_sha256": materialization["identities"][
                    "bundle_admission_sha256"
                ],
            },
            "carrier_plan": {
                "artifact": materialization["artifacts"]["carrier_plan"],
                "carrier_plan_sha256": materialization["identities"][
                    "carrier_plan_sha256"
                ],
                "logical_bundle_size": materialization["carrier_plan"][
                    "logical_bundle_size"
                ],
            },
            "batch_count": len(materialization["lowered_batches"]),
            "selected_physical_batch_capacity": (authority.physical_batch_capacity),
            "available_physical_batch_capacity": (
                authority.available_physical_batch_capacity
            ),
        },
        "lifecycle": {
            "plan_path": str(root / "science_lifecycle_plan.json"),
            "plan_sha256": lifecycle_plan["plan_sha256"],
            "contract_path": str(root / "artifact_lifecycle_contract.json"),
            "contract_sha256": lifecycle["contract_sha256"],
            "run_passport_path": str(root / "run_passport.json"),
            "artifact_disposition_manifest_path": str(
                root / "artifact_disposition_manifest.json"
            ),
            "storage_disposition_receipt_path": str(
                root / "storage_disposition_receipt.json"
            ),
            "completion_condition": (
                "authoritative only when the terminal passport, complete exact "
                "disposition manifest, and bound storage receipt all validate"
            ),
        },
        "storage_admission": storage_receipt,
        "runtime_root": None,
        "runtime_effects_performed": False,
        "execution_started": False,
        "prequal_started": False,
        "qualification_started": False,
        "qualification_authority_effect": False,
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core).to_dict()}
    _write_immutable_json(receipt_path, receipt, field="science plan receipt")
    try:
        lifecycle_closeout = _publish_science_lifecycle_closeout_v1(
            root=root,
            plan=lifecycle_plan,
            source_identity=lifecycle_source,
            provider_id="eiel-science-request-v1-materializer",
            terminal_status="completed",
            inventory_complete=True,
            register_archive=True,
        )
    except (ArtifactLifecycleError, OSError, ScienceRequestV1FrontDoorError) as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            f"science materialization lifecycle closeout failed: {error}",
            details={"materialization_root": str(root)},
        ) from error
    return (
        {
            **receipt,
            "reused": False,
            "lifecycle_closeout": lifecycle_closeout,
        },
        authority,
        root,
    )


def plan_science_request_v1(
    request: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path | None = None,
    materialization_parent: str | Path | None = None,
    authority_resolver: ScienceAuthorityResolverV1 | None = None,
) -> dict[str, Any]:
    """Resolve, compile, admit, and materialize without executing a batch."""

    loaded = _load_request(request)
    try:
        compilation = compile_science_request_v1(loaded.request)
    except (TypeError, ValueError) as error:
        raise ScienceRequestV1FrontDoorError(
            "INVALID_SCIENCE_REQUEST",
            f"science request compilation failed: {error}",
            details={
                "request_sha256": loaded.request.request_sha256().to_dict(),
                "execution_started": False,
            },
        ) from error
    repository = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    parent = Path(
        materialization_parent or DEFAULT_SCIENCE_RUNTIME_ROOT / "materializations"
    )
    result, _, _ = _plan_loaded_request(
        loaded,
        compilation,
        repo_root=repository,
        materialization_parent=parent,
        authority_resolver=authority_resolver or resolve_current_science_authority_v1,
    )
    return result


def _safe_campaign_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.").lower()
    if not result:
        raise ScienceRequestV1FrontDoorError(
            "INVALID_SCIENCE_REQUEST", "campaign_id has no safe runtime projection"
        )
    return result[:96]


def _fresh_runtime_root(parent: Path, campaign_id: str) -> Path:
    parent = parent.expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science runtime parent is invalid: {parent}",
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(
        tempfile.mkdtemp(
            prefix=f"{_safe_campaign_id(campaign_id)}--{timestamp}--",
            dir=parent,
        )
    ).resolve()


def _write_run_failure(
    root: Path,
    *,
    request_sha256: Mapping[str, Any],
    stage: str,
    code: str,
    message: str,
) -> Path:
    core = {
        "schema_version": "eiel_science_request_v1_run_failure_receipt_v1",
        "request_sha256": dict(request_sha256),
        "runtime_root": str(root),
        "status": code,
        "stage": stage,
        "message": message[:4096],
        "automatic_prune_allowed": False,
        "qualification_authority_effect": False,
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core).to_dict()}
    return _write_immutable_json(
        root / "science_campaign_failure_receipt.json",
        receipt,
        field="science campaign failure receipt",
    )


def _verified_child_path_v1(root: Path, raw: object, *, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED, f"{field} path is absent"
        )
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED, f"{field} path escapes its batch root"
        )
    path = _regular_file(root / relative, field=field)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED, f"{field} path escapes its batch root"
        ) from error
    return path


def _verify_child_launchpad_lifecycle_v1(
    *, batch_root: Path, front_door_result: Mapping[str, Any], bound_plan_sha256: str
) -> dict[str, Any]:
    marker_path = _regular_file(
        batch_root / "terminal_lifecycle_commit.json",
        field="batch terminal lifecycle commit",
    )
    marker = _read_json(marker_path, field="batch terminal lifecycle commit")
    marker_core = dict(marker)
    supplied_marker_sha256 = marker_core.pop("commit_sha256", None)
    if (
        marker.get("schema_version") != "eiel_execution_terminal_lifecycle_commit_v1"
        or marker.get("plan_sha256") != bound_plan_sha256
        or supplied_marker_sha256 != canonical_sha256(marker_core).hex
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "batch terminal lifecycle commit identity differs",
        )
    result_ref = marker.get("result")
    manifest_ref = marker.get("manifest")
    storage_ref = marker.get("storage_receipt")
    if not all(
        isinstance(value, Mapping) for value in (result_ref, manifest_ref, storage_ref)
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "batch terminal lifecycle references are incomplete",
        )
    result_path = _verified_child_path_v1(
        batch_root, result_ref.get("path"), field="batch execution result"
    )
    manifest_path = _verified_child_path_v1(
        batch_root,
        manifest_ref.get("path"),
        field="batch artifact disposition manifest",
    )
    storage_path = _verified_child_path_v1(
        batch_root,
        storage_ref.get("path"),
        field="batch storage disposition receipt",
    )
    persisted_result = _read_json(result_path, field="batch execution result")
    persisted_result_core = dict(persisted_result)
    supplied_result_sha256 = persisted_result_core.pop("result_sha256", None)
    manifest = validate_artifact_disposition_manifest_v1(
        _read_json(manifest_path, field="batch artifact disposition manifest")
    )
    storage = _read_json(storage_path, field="batch storage disposition receipt")
    storage_core = dict(storage)
    supplied_storage_sha256 = storage_core.pop("receipt_sha256", None)
    storage_receipt_semantically_bound = (
        storage.get("schema_version") == "eiel_storage_disposition_receipt_v1"
        and supplied_storage_sha256 == canonical_sha256(storage_core).hex
        and storage.get("manifest_sha256") == manifest.get("manifest_sha256")
        and storage.get("plan_sha256") == bound_plan_sha256
        and storage.get("campaign_id") == manifest.get("campaign_id")
        and storage.get("run_id") == manifest.get("run_id")
        and storage.get("lifecycle_contract_sha256")
        == manifest.get("lifecycle_contract_sha256")
    )
    storage_reference_current = (
        storage_ref.get("file_sha256") == _file_sha256(storage_path)
        and storage_ref.get("receipt_sha256") == supplied_storage_sha256
    )
    # The fast lifecycle release actuator is allowed to update the mutable
    # storage disposition receipt after the terminal lifecycle commit has
    # been sealed.  The commit still proves the exact path and the sealed
    # manifest/plan binding; accept the authorized semantic successor so a
    # completed child is not made unrecoverable merely because R0-R2 release
    # ran between provider closeout and science-batch authentication.
    storage_reference_successor = (
        storage_ref.get("path") == "storage_disposition_receipt.json"
        and not storage_reference_current
        and storage_receipt_semantically_bound
    )
    if (
        persisted_result != dict(front_door_result)
        or supplied_result_sha256 != canonical_sha256(persisted_result_core).hex
        or result_ref.get("file_sha256") != _file_sha256(result_path)
        or result_ref.get("result_sha256") != supplied_result_sha256
        or manifest.get("runtime_root") != str(batch_root)
        or manifest.get("plan_sha256") != bound_plan_sha256
        or manifest.get("terminal_status") != "completed"
        or manifest.get("inventory", {}).get("complete") is not True
        or manifest_ref.get("file_sha256") != _file_sha256(manifest_path)
        or manifest_ref.get("manifest_sha256") != manifest.get("manifest_sha256")
        or not storage_receipt_semantically_bound
        or not (storage_reference_current or storage_reference_successor)
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "batch terminal lifecycle material is not exact or complete",
        )
    return {
        "terminal_lifecycle_commit_path": str(marker_path),
        "terminal_lifecycle_commit_file_sha256": _file_sha256(marker_path),
        "terminal_lifecycle_commit_sha256": supplied_marker_sha256,
        "artifact_disposition_manifest_path": str(manifest_path),
        "artifact_disposition_manifest_sha256": manifest["manifest_sha256"],
        "storage_disposition_receipt_path": str(storage_path),
        "storage_disposition_receipt_sha256": supplied_storage_sha256,
        "storage_receipt_successor_after_terminal_commit": (
            storage_reference_successor
        ),
    }


def _science_batch_launchpad_failure_diagnostics_v1(
    *, result: Mapping[str, Any], front_door_result: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Keep bounded Launchpad failure evidence on the fail-closed path.

    A failed canonical execution result must never be accepted as a science
    batch.  It must, however, remain diagnosable: replacing the provider's
    status and warning with only the terminal-lifecycle authentication error
    makes the next repair unnecessarily speculative.
    """

    candidates = (result, front_door_result)
    diagnostics: dict[str, Any] = {}
    for field, output_name in (
        ("status", "launchpad_status"),
        ("result_class", "launchpad_result_class"),
        ("error", "launchpad_error"),
    ):
        for candidate in candidates:
            value = candidate.get(field)
            if isinstance(value, str) and value:
                diagnostics[output_name] = value[:1024]
                break
    warnings: list[str] = []
    for candidate in candidates:
        raw_warnings = candidate.get("warnings")
        if not isinstance(raw_warnings, list):
            continue
        for warning in raw_warnings:
            if isinstance(warning, str) and warning and warning not in warnings:
                warnings.append(warning[:2048])
            if len(warnings) == 8:
                break
        if len(warnings) == 8:
            break
    if warnings:
        diagnostics["launchpad_warnings"] = warnings
    summary_parts = [
        f"{key}={diagnostics[key]}"
        for key in ("launchpad_status", "launchpad_result_class", "launchpad_error")
        if key in diagnostics
    ]
    summary_parts.extend(f"launchpad_warning={warning}" for warning in warnings)
    summary = (
        "; ".join(summary_parts) or "Launchpad returned no bounded failure diagnostics"
    )
    return summary[:4096], diagnostics


def _verified_selected_world_host_observation_v1(
    *,
    result: Mapping[str, Any],
    batch_root: Path,
    expected_world_ids: list[str],
) -> dict[str, Any]:
    binding = result.get("selected_world_host_observation")
    front_door_result = result.get("front_door_result")
    terminal = (
        front_door_result.get("workhorse_terminalized_indexed_continuation")
        if isinstance(front_door_result, Mapping)
        else None
    )
    terminal_binding = (
        terminal.get("host_boundary_observation")
        if isinstance(terminal, Mapping)
        else None
    )
    expected_fields = {
        "observation_mode",
        "path",
        "file_sha256",
        "content_sha256",
        "file_bytes",
        "checkpoint_occurrence_sha256",
        "checkpoint_occurrence_projection_sha256",
        "terminally_reopened",
        "runtime_observation_mode",
        "host_adapter_observation_only",
    }
    if (
        len(expected_world_ids) != 1
        or not isinstance(front_door_result, Mapping)
        or type(binding) is not dict
        or set(binding) != expected_fields
        or terminal_binding != binding
        or front_door_result.get("workhorse_terminalized_operation") != terminal
        or binding.get("observation_mode")
        != WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1
        or binding.get("runtime_observation_mode") != "OFF"
        or binding.get("terminally_reopened") is not True
        or binding.get("host_adapter_observation_only") is not True
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "selected-world Science Request child observation binding is inexact",
        )
    try:
        observation = validate_workhorse_selected_world_observation_v1(
            binding["path"],
            expected_checkpoint_occurrence_sha256=binding[
                "checkpoint_occurrence_sha256"
            ],
            expected_checkpoint_occurrence_projection_sha256=binding[
                "checkpoint_occurrence_projection_sha256"
            ],
        )
        observation_path = Path(str(binding["path"])).resolve(strict=True)
        observation_path.relative_to(batch_root.resolve())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"selected-world Science Request child observation is invalid: {error}",
        ) from error
    integrity = observation.get("integrity")
    event_coverage = observation.get("event_observation_coverage")
    if (
        observation.get("cell_id") != expected_world_ids[0]
        or binding["path"] != observation["observation_path"]
        or binding["file_sha256"] != observation["observation_file_sha256"]
        or binding["content_sha256"]
        != observation["observation_content_sha256"]
        or binding["file_bytes"] != observation["observation_file_bytes"]
        or observation.get("execution_authority_granted") is not False
        or observation.get("qualification_authority_granted") is not False
        or observation.get("named_phenotype_reward_authority_granted") is not False
        or not isinstance(integrity, Mapping)
        or integrity.get("runtime_observation_mode_remains_off") is not True
        or not isinstance(event_coverage, Mapping)
        or event_coverage.get("event_kind_rows_available") is not False
        or event_coverage.get("birth_death_causal_rows_claimed") is not False
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "selected-world Science Request child observation changed authority or identity",
        )
    return dict(binding)


def _verify_science_batch_result_v1(
    *,
    result: Mapping[str, Any],
    batch_index: int,
    batch_root: Path,
    lowered: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
    require_launchpad_lifecycle: bool,
) -> dict[str, Any] | None:
    science = lowered.get("science_materialization")
    expected_world_ids = _materialized_world_ids_v1(science)
    identities = materialization_receipt.get("identities")
    source = materialization_receipt.get("lineage")
    science_lineage = result.get("science_lineage")
    source_lineage = result.get("source_lineage")
    selected_world_host_observation = _is_selected_world_lowered_batch_v1(lowered)
    published_identity_fields = (
        "request_sha256",
        "campaign_spec_sha256",
        "compilation_bundle_sha256",
        "bundle_content_sha256",
        "bundle_admission_sha256",
        "carrier_plan_sha256",
        "base_campaign_plan_sha256",
    )
    if (
        not isinstance(expected_world_ids, list)
        or not isinstance(identities, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(science_lineage, Mapping)
        or not isinstance(source_lineage, Mapping)
        or result.get("batch_index") != batch_index
        or result.get("runtime_root") != str(batch_root)
        or result.get("scientific_execution")
        is not (not selected_world_host_observation)
        or (
            selected_world_host_observation
            and result.get("engineering_only") is not True
        )
        or science_lineage.get("schema_version")
        != SCIENCE_INSTRUMENT_EXECUTION_LINEAGE_SCHEMA_VERSION
        or science_lineage.get("instrument_contract") != "EIEL_SCIENCE_INSTRUMENT_V1"
        or science_lineage.get("science_use_class") != science.get("science_use_class")
        or science_lineage.get("observation_mode") != science.get("observation_mode")
        or science_lineage.get("batch_index") != batch_index
        or science_lineage.get("world_ids") != expected_world_ids
        or any(
            field not in identities
            or science_lineage.get(field) != identities.get(field)
            for field in published_identity_fields
        )
        or science_lineage.get("lowered_plan_sha256") != lowered.get("plan_sha256")
        or source_lineage.get("compiled_plan_sha256") != lowered.get("plan_sha256")
        or any(source_lineage.get(field) != value for field, value in source.items())
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} result lineage differs from its materialization",
        )
    bound_plan_sha256 = source_lineage.get("bound_plan_sha256")
    if (
        not isinstance(bound_plan_sha256, str)
        or len(bound_plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in bound_plan_sha256)
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} lacks a bound Launchpad plan identity",
        )
    if selected_world_host_observation and not require_launchpad_lifecycle:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "selected-world host observation requires canonical Launchpad lifecycle",
        )
    if not require_launchpad_lifecycle:
        return None
    front_door_result = result.get("front_door_result")
    if not isinstance(front_door_result, Mapping):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} lacks a canonical Launchpad result",
        )
    if result.get("ok") is not True:
        failure_summary, failure_diagnostics = (
            _science_batch_launchpad_failure_diagnostics_v1(
                result=result,
                front_door_result=front_door_result,
            )
        )
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} has no authenticated accepted Launchpad "
            f"terminal lifecycle; {failure_summary}",
            details={
                "batch_index": batch_index,
                "runtime_root": str(batch_root),
                "execution_started": True,
                **failure_diagnostics,
            },
        )
    if selected_world_host_observation:
        _verified_selected_world_host_observation_v1(
            result=result,
            batch_root=batch_root,
            expected_world_ids=[str(value) for value in expected_world_ids],
        )
    return _verify_child_launchpad_lifecycle_v1(
        batch_root=batch_root,
        front_door_result=front_door_result,
        bound_plan_sha256=bound_plan_sha256,
    )


def _existing_runtime_root_v1(value: str | Path) -> Path:
    """Resolve an explicit resume root without accepting links or files."""

    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science resume root must not be a symbolic link: {candidate}",
        )
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science resume root is not a regular directory: {resolved}",
        )
    return resolved


def _science_run_identity_v1(
    *,
    campaign_id: str,
    request_digest: Mapping[str, Any],
    materialization_root: Path,
    materialization_receipt_sha256: object,
    authority: ScienceExecutionAuthorityV1,
    lifecycle_plan_sha256: object,
    compilation: ScienceRequestV1CompilationV1,
    lowered_batches: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    require_launchpad_lifecycle: bool,
) -> dict[str, Any]:
    world_ids: list[list[str]] = []
    for index, lowered in enumerate(lowered_batches):
        science = lowered.get("science_materialization")
        ids = _materialized_world_ids_v1(science)
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_MATERIALIZATION_FAILED,
                f"lowered batch {index} lacks stable semantic world ownership",
            )
        world_ids.append(list(ids))
    return {
        "campaign_id": campaign_id,
        "request_sha256": dict(request_digest),
        "compilation_receipt_sha256": compilation.compilation_receipt["receipt_sha256"],
        "materialization_root": str(materialization_root),
        "materialization_plan_receipt_sha256": materialization_receipt_sha256,
        "authority_sha256": authority.authority_sha256,
        "base_campaign_plan_sha256": authority.base_campaign_plan_sha256,
        "science_context_sha256": authority.science_context["context_sha256"],
        "lifecycle_plan_sha256": lifecycle_plan_sha256,
        "batch_count": len(world_ids),
        "batch_world_ids": world_ids,
        "require_launchpad_lifecycle": require_launchpad_lifecycle,
    }


def _new_science_run_journal_v1(
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCIENCE_REQUEST_V1_RUN_JOURNAL_SCHEMA_VERSION,
        "identity": dict(identity),
        "status": "RUNNING",
        "batches": [
            {
                "batch_index": index,
                "world_ids": list(world_ids),
                "state": "PENDING",
                "attempt": 0,
                "attempt_root": None,
                "result": None,
                "child_lifecycle": None,
                "wait": None,
            }
            for index, world_ids in enumerate(identity["batch_world_ids"])
        ],
        "aggregate": None,
        "last_event": "created",
    }


def _write_science_run_journal_v1(root: Path, journal: Mapping[str, Any]) -> Path:
    core = dict(journal)
    core.pop("journal_sha256", None)
    if core.get("schema_version") != SCIENCE_REQUEST_V1_RUN_JOURNAL_SCHEMA_VERSION:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "science run journal schema is invalid",
        )
    sealed = {**core, "journal_sha256": canonical_sha256(core).to_dict()}
    path = root / "run_journal.json"
    try:
        atomic_write_json(path, sealed)
    except (OSError, TypeError, ValueError) as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science run journal could not be written: {error}",
            details={"runtime_root": str(root)},
        ) from error
    return path


def _read_science_run_journal_v1(root: Path) -> dict[str, Any]:
    path = _regular_file(root / "run_journal.json", field="science run journal")
    value = _read_json(path, field="science run journal")
    declared = value.get("journal_sha256")
    core = {key: item for key, item in value.items() if key != "journal_sha256"}
    if (
        value.get("schema_version") != SCIENCE_REQUEST_V1_RUN_JOURNAL_SCHEMA_VERSION
        or declared != canonical_sha256(core).to_dict()
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "science run journal digest or schema is invalid",
            details={"runtime_root": str(root)},
        )
    batches = value.get("batches")
    if not isinstance(batches, list) or any(
        not isinstance(row, Mapping) for row in batches
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "science run journal batches are invalid",
            details={"runtime_root": str(root)},
        )
    return value


def _validate_science_run_journal_identity_v1(
    journal: Mapping[str, Any], *, expected: Mapping[str, Any]
) -> None:
    if journal.get("identity") != dict(expected):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "science resume root is bound to another request or materialization",
            details={
                "resume_identity": journal.get("identity"),
                "expected_identity": dict(expected),
            },
        )
    batches = journal.get("batches")
    expected_world_ids = expected.get("batch_world_ids")
    if (
        not isinstance(batches, list)
        or not isinstance(expected_world_ids, list)
        or len(batches) != len(expected_world_ids)
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "science resume journal batch set differs from the materialization",
        )
    for index, row in enumerate(batches):
        if (
            not isinstance(row, Mapping)
            or row.get("batch_index") != index
            or row.get("world_ids") != expected_world_ids[index]
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science resume journal batch {index} ownership differs",
            )


def _journal_attempt_root_v1(
    root: Path, row: Mapping[str, Any], *, batch_index: int
) -> Path:
    raw = row.get("attempt_root")
    if not isinstance(raw, str) or not raw:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} lacks an attempt root",
        )
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} attempt root escapes the campaign root",
        )
    unresolved_candidate = root / relative
    if unresolved_candidate.is_symlink():
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} attempt root is a symbolic link",
        )
    candidate = unresolved_candidate.resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} attempt root escapes the campaign root",
        ) from error
    return candidate


_RETRYABLE_ATTEMPT_ADMISSION_RESULT_CLASS_V1: Final[str] = (
    "ATTEMPT_ADMISSION_UNAVAILABLE"
)
_RETRYABLE_ATTEMPT_ADMISSION_WARNING_PREFIX_V1: Final[str] = (
    "CRCA_SCHEDULER_ATTEMPT_REJECTED:"
)
_RETRYABLE_ATTEMPT_ADMISSION_RESOURCE_MARKERS_V1: Final[tuple[str, ...]] = (
    "single-worker execution lock",
    "scheduler process",
    "exclusive_gpu_conflict",
    "resource manager",
    "resource admission",
    "device capacity",
    "start lease",
)
_NON_RETRYABLE_ATTEMPT_ADMISSION_MARKERS_V1: Final[tuple[str, ...]] = (
    "canonical continuation lineage",
    "published terminal",
    "request identity",
    "semantic change",
)
_RETRYABLE_BATCH_STATES_V1: Final[frozenset[str]] = frozenset(
    {WAIT_RESOURCE, WAIT_STORAGE, "RESOURCE_WAIT", "STORAGE_WAIT"}
)


def _retryable_admission_warning_v1(value: object) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return (
        _RETRYABLE_ATTEMPT_ADMISSION_WARNING_PREFIX_V1 in value
        and not any(marker in lowered for marker in _NON_RETRYABLE_ATTEMPT_ADMISSION_MARKERS_V1)
        and any(marker in lowered for marker in _RETRYABLE_ATTEMPT_ADMISSION_RESOURCE_MARKERS_V1)
    )


def _retryable_admission_result_v1(
    *, root: Path, row: Mapping[str, Any], batch_index: int
) -> dict[str, Any] | None:
    """Authenticate the narrow class of failures safe to retry in-place.

    A scheduler admission rejection has not launched a child and therefore may
    be retried as a new attempt of the *same* physical batch.  This helper is
    deliberately stricter than the generic wait mapping: a semantic,
    authentication, provider, or terminal failure must remain terminal.
    """

    try:
        attempt_root = _journal_attempt_root_v1(
            root, row, batch_index=batch_index
        )
    except ScienceRequestV1FrontDoorError:
        return None
    result_path = attempt_root / "execution_result.json"
    if result_path.is_symlink() or not result_path.is_file():
        return None
    try:
        result = _read_json(
            result_path, field=f"science batch {batch_index} admission result"
        )
    except ScienceRequestV1FrontDoorError:
        return None
    warnings = result.get("warnings")
    authority = result.get("crca_attempt_authority")
    reason_codes = authority.get("reason_codes") if isinstance(authority, Mapping) else None
    if (
        result.get("schema_version") != "eiel_execution_result_v2"
        or result.get("ok") is not False
        or result.get("status") != "precheck_failed"
        or result.get("result_class")
        != _RETRYABLE_ATTEMPT_ADMISSION_RESULT_CLASS_V1
        or not isinstance(warnings, list)
        or not any(
            _retryable_admission_warning_v1(value)
            for value in warnings
        )
        or not isinstance(authority, Mapping)
        or authority.get("execution_authority_granted") is not False
        or not isinstance(reason_codes, list)
        or not any(
            _retryable_admission_warning_v1(value)
            for value in reason_codes
        )
    ):
        return None
    failure = row.get("failure")
    failure_message = failure.get("message") if isinstance(failure, Mapping) else None
    if not isinstance(failure_message, str):
        failure_message = ""
    if _RETRYABLE_ATTEMPT_ADMISSION_RESULT_CLASS_V1 not in failure_message:
        # A failed result may have been sealed before the outer journal was
        # updated.  The authenticated execution result is sufficient in that
        # narrow case; no other result class is accepted here.
        failure_message = str(result.get("error") or "")
    if (
        _RETRYABLE_ATTEMPT_ADMISSION_RESULT_CLASS_V1 not in failure_message
        and not any(_retryable_admission_warning_v1(value) for value in warnings)
    ):
        return None
    return result


def _advance_retryable_science_batch_attempt_v1(
    *, root: Path, row: Mapping[str, Any], batch_index: int, reason: str
) -> dict[str, Any]:
    """Move one wait/admission-failed batch to a fresh immutable attempt.

    The previous attempt remains untouched and is indexed in the journal.  A
    retry never reuses an attempt directory and never replays completed
    batches.
    """

    updated = dict(row)
    old_attempt = updated.get("attempt")
    if type(old_attempt) is not int or old_attempt <= 0:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} retry attempt identity is invalid",
        )
    old_root = _journal_attempt_root_v1(root, updated, batch_index=batch_index)
    new_attempt = old_attempt + 1
    new_relative = (
        Path("batches")
        / f"batch-{batch_index:04d}"
        / f"attempt-{new_attempt:04d}"
    )
    new_root = (root / new_relative).resolve()
    if new_root.exists() or new_root.is_symlink():
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} retry attempt root already exists",
            details={"runtime_root": str(root), "batch_index": batch_index},
        )
    history = updated.get("retry_history")
    if history is None:
        history = []
    if not isinstance(history, list) or any(
        not isinstance(value, Mapping) for value in history
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} retry history is invalid",
        )
    history = [dict(value) for value in history]
    history.append(
        {
            "attempt": old_attempt,
            "attempt_root": str(old_root.relative_to(root.resolve())),
            "state": updated.get("state"),
            "reason": reason,
            "failure": updated.get("failure"),
            "wait": updated.get("wait"),
            "result": updated.get("result"),
        }
    )
    updated["retry_history"] = history
    updated["attempt"] = new_attempt
    updated["attempt_root"] = str(new_relative)
    updated["state"] = "PENDING"
    updated["wait"] = None
    updated["failure"] = None
    updated["result"] = None
    updated["child_lifecycle"] = None
    return updated


def _failed_batch_retry_index_v1(root: Path, journal: Mapping[str, Any]) -> int | None:
    """Return the sole failed batch only when its admission rejection is safe."""

    failed = [
        (index, row)
        for index, row in enumerate(journal.get("batches", []))
        if isinstance(row, Mapping) and row.get("state") == "FAILED"
    ]
    if len(failed) != 1:
        return None
    batch_index, row = failed[0]
    current = journal.get("current_batch_index")
    if current != batch_index or _retryable_admission_result_v1(
        root=root, row=row, batch_index=batch_index
    ) is None:
        return None
    # Completed rows before the failed batch are immutable and all later rows
    # must still be pending; otherwise this is not a single resumable frontier.
    rows = journal.get("batches")
    if not isinstance(rows, list):
        return None
    for index, candidate in enumerate(rows):
        if not isinstance(candidate, Mapping):
            return None
        if index < batch_index and candidate.get("state") != "COMPLETED":
            return None
        if index > batch_index and candidate.get("state") not in {
            "PENDING",
            *(_RETRYABLE_BATCH_STATES_V1),
        }:
            return None
    return batch_index


def _seal_science_batch_result_v1(result: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(result)
    declared = core.pop("result_sha256", None)
    digest = canonical_sha256(core).hex
    if declared is not None and declared not in {
        digest,
        canonical_sha256(core).to_dict(),
    }:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "science batch result digest is invalid",
        )
    return {**core, "result_sha256": digest}


def _write_or_verify_science_batch_result_v1(
    *,
    path: Path,
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    sealed = _seal_science_batch_result_v1(result)
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science batch result is a symbolic link: {path}",
            )
        existing = _read_json(path, field="science batch result")
        if existing != sealed:
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science batch result already exists with different content: {path}",
            )
    else:
        _write_immutable_json(path, sealed, field="science batch result")
    return sealed, _file_sha256(path)


def _write_or_verify_json_v1(
    *,
    path: Path,
    value: Mapping[str, Any],
    field: str,
) -> Path:
    """Write an immutable artifact once, or prove an identical existing copy."""

    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"{field} is a symbolic link: {path}",
            )
        if _read_json(path, field=field) != dict(value):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"{field} already exists with different content: {path}",
            )
        return path
    return _write_immutable_json(path, value, field=field)


def _reconcile_precloseout_run_receipt_v1(
    *,
    path: Path,
    candidate: Mapping[str, Any],
    journal: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse an immutable receipt across one proven child-lifecycle successor.

    A child storage disposition receipt may advance after its terminal commit
    when the canonical release actuator applies exact R0-R2 dispositions.  A
    process interrupted before the outer lifecycle closeout must not rewrite
    its already sealed campaign receipt, but that lifecycle-only successor
    must not make the completed campaign unrecoverable either.
    """

    receipt_path = _regular_file(path, field="science campaign run receipt")
    existing = _read_json(receipt_path, field="science campaign run receipt")
    existing_core = {
        key: value for key, value in existing.items() if key != "receipt_sha256"
    }
    if existing.get("receipt_sha256") != canonical_sha256(existing_core).to_dict():
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "existing science campaign run receipt digest mismatch",
        )
    if existing == dict(candidate):
        return existing

    candidate_core = {
        key: value for key, value in candidate.items() if key != "receipt_sha256"
    }
    existing_batches = existing_core.pop("completed_batches", None)
    candidate_batches = candidate_core.pop("completed_batches", None)
    journal_batches = journal.get("batches")
    if (
        existing_core != candidate_core
        or not isinstance(existing_batches, list)
        or not isinstance(candidate_batches, list)
        or not isinstance(journal_batches, list)
        or len(existing_batches) != len(candidate_batches)
        or len(existing_batches) != len(journal_batches)
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "existing science campaign run receipt differs outside child lifecycle authority",
        )

    for batch_index, (existing_raw, candidate_raw, journal_raw) in enumerate(
        zip(existing_batches, candidate_batches, journal_batches, strict=True)
    ):
        if not all(
            isinstance(value, Mapping)
            for value in (existing_raw, candidate_raw, journal_raw)
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science campaign receipt batch {batch_index} is malformed",
            )
        existing_batch = dict(existing_raw)
        candidate_batch = dict(candidate_raw)
        existing_lifecycle = existing_batch.pop("child_lifecycle", None)
        candidate_lifecycle = candidate_batch.pop("child_lifecycle", None)
        journal_lifecycle = journal_raw.get("child_lifecycle")
        if (
            existing_batch != candidate_batch
            or not isinstance(existing_lifecycle, Mapping)
            or not isinstance(candidate_lifecycle, Mapping)
            or dict(existing_lifecycle) != journal_lifecycle
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science campaign receipt batch {batch_index} differs from its journal",
            )
        if dict(existing_lifecycle) == dict(candidate_lifecycle):
            continue

        existing_stable = dict(existing_lifecycle)
        candidate_stable = dict(candidate_lifecycle)
        existing_storage_sha256 = existing_stable.pop(
            "storage_disposition_receipt_sha256", None
        )
        candidate_storage_sha256 = candidate_stable.pop(
            "storage_disposition_receipt_sha256", None
        )
        existing_successor = existing_stable.pop(
            "storage_receipt_successor_after_terminal_commit", None
        )
        candidate_successor = candidate_stable.pop(
            "storage_receipt_successor_after_terminal_commit", None
        )
        if (
            existing_stable != candidate_stable
            or existing_successor is not False
            or candidate_successor is not True
            or not isinstance(existing_storage_sha256, str)
            or not isinstance(candidate_storage_sha256, str)
            or existing_storage_sha256 == candidate_storage_sha256
            or re.fullmatch(r"[0-9a-f]{64}", existing_storage_sha256) is None
            or re.fullmatch(r"[0-9a-f]{64}", candidate_storage_sha256) is None
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                "existing science campaign receipt differs by more than an "
                f"authenticated storage lifecycle successor for batch {batch_index}",
            )
    return existing


def _load_completed_science_batch_v1(
    *,
    root: Path,
    row: Mapping[str, Any],
    batch_index: int,
    lowered: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
    require_launchpad_lifecycle: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempt_root = _journal_attempt_root_v1(root, row, batch_index=batch_index)
    result_ref = row.get("result")
    if not isinstance(result_ref, Mapping):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} completed state lacks a result reference",
        )
    raw_path = result_ref.get("path")
    if (
        not isinstance(raw_path, str)
        or Path(raw_path).is_absolute()
        or ".." in Path(raw_path).parts
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} result reference is invalid",
        )
    unresolved_result_path = root / Path(raw_path)
    if unresolved_result_path.is_symlink():
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} result is a symbolic link",
        )
    result_path = unresolved_result_path.resolve()
    try:
        result_path.relative_to(root.resolve())
    except ValueError as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} result reference escapes the campaign root",
        ) from error
    if result_path.parent != attempt_root or not result_path.is_file():
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} result is not a regular file owned by its attempt",
        )
    result = _read_json(result_path, field=f"science batch {batch_index} result")
    supplied = result.get("result_sha256")
    core = {key: item for key, item in result.items() if key != "result_sha256"}
    if (
        supplied != canonical_sha256(core).hex
        or result_ref.get("result_sha256") != supplied
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} completed result digest mismatch",
        )
    if result_ref.get("file_sha256") != _file_sha256(result_path):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} completed result file changed",
        )
    campaign_result_path = root / f"batch_result_{batch_index:04d}.json"
    if campaign_result_path.is_symlink() or not campaign_result_path.is_file():
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} campaign result is not a regular file",
        )
    campaign_result = _read_json(
        campaign_result_path,
        field=f"science batch {batch_index} campaign result",
    )
    if campaign_result != result:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} campaign result differs from its attempt result",
        )
    child_lifecycle = _verify_science_batch_result_v1(
        result=result,
        batch_index=batch_index,
        batch_root=attempt_root,
        lowered=lowered,
        materialization_receipt=materialization_receipt,
        require_launchpad_lifecycle=require_launchpad_lifecycle,
    )
    canonical_result_path = root / f"batch_result_{batch_index:04d}.json"
    if canonical_result_path.is_symlink() or not canonical_result_path.is_file():
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} canonical result is not a regular file",
        )
    if (
        _read_json(
            canonical_result_path,
            field=f"science batch {batch_index} canonical result",
        )
        != result
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science batch {batch_index} canonical result differs from its attempt",
        )
    receipt = {
        "batch_index": batch_index,
        "path": str(canonical_result_path),
        "file_sha256": _file_sha256(canonical_result_path),
        "result_sha256": supplied,
        "ok": result.get("ok"),
        "status": result.get("status"),
        "result_class": result.get("result_class"),
        "runtime_root": str(attempt_root),
        "materialized_world_ids": _materialized_world_ids_v1(
            lowered["science_materialization"]
        ),
        "child_lifecycle": child_lifecycle,
        **(
            {
                "selected_world_host_observation": dict(
                    result["selected_world_host_observation"]
                )
            }
            if _is_selected_world_lowered_batch_v1(lowered)
            else {}
        ),
    }
    return result, receipt


def _science_batch_result_envelope_v1(
    *,
    result: Mapping[str, Any],
    physical_batch_index: int,
    request_sha256: ContentDigest,
    compilation: ScienceRequestV1CompilationV1,
) -> dict[str, Any]:
    """Project one completed child with every required scientific payload."""

    reducer = result.get("compact_reducer")
    atlas = reducer.get("atlas_result") if isinstance(reducer, Mapping) else None
    if not isinstance(atlas, Mapping):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            f"science batch {physical_batch_index} lacks a completed compact atlas result",
            details={"execution_started": True},
        )
    try:
        strategy_required = strategy_attribution_required_for_bundle_v1(
            compilation.world_bundle
        )
    except StrategyAttributionError as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            f"science batch {physical_batch_index} strategy contract is invalid: {error}",
            details={"execution_started": True},
        ) from error
    strategy = reducer.get("strategy_attribution_result")
    if strategy_required and not isinstance(strategy, Mapping):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            f"science batch {physical_batch_index} lacks the required direct strategy attribution result",
            details={
                "execution_started": True,
                "required_payload": "strategy_attribution_result",
            },
        )
    if not strategy_required and strategy is not None:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            f"science batch {physical_batch_index} returned an unrequested strategy attribution result",
            details={"execution_started": True},
        )
    envelope = {
        "physical_batch_index": physical_batch_index,
        "request_sha256": request_sha256.hex,
        "atlas_result": dict(atlas),
    }
    if isinstance(strategy, Mapping):
        envelope["strategy_attribution_result"] = dict(strategy)
    return envelope


def _science_batch_result_envelope_for_batch_v1(
    *,
    result: Mapping[str, Any],
    lowered: Mapping[str, Any],
    physical_batch_index: int,
    request_sha256: ContentDigest,
    compilation: ScienceRequestV1CompilationV1,
) -> dict[str, Any] | None:
    """Build a reducer envelope only for batches that produce one.

    Selected-world continuations deliberately use the authenticated host
    observation boundary with runtime observation disabled. They therefore
    have no compact atlas result to project. Keeping this decision beside the
    ordinary envelope constructor prevents a fresh execution from
    authenticating the child correctly and then incorrectly demanding a
    reducer payload.
    """

    if _is_selected_world_lowered_batch_v1(lowered):
        return None
    return _science_batch_result_envelope_v1(
        result=result,
        physical_batch_index=physical_batch_index,
        request_sha256=request_sha256,
        compilation=compilation,
    )


def _reducer_recovery_source_attempt_root_v1(
    source_root: str | Path, *, batch_index: int
) -> Path:
    """Resolve one explicitly named sealed child; never search a run family."""

    candidate = _existing_runtime_root_v1(source_root)
    if (candidate / "execution_result.json").is_file():
        return candidate
    journal = _read_science_run_journal_v1(candidate)
    batches = journal.get("batches")
    if (
        not isinstance(batches, list)
        or not 0 <= batch_index < len(batches)
        or not isinstance(batches[batch_index], Mapping)
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"reducer recovery source has no batch {batch_index}",
            details={"reducer_recovery_source_root": str(candidate)},
        )
    return _journal_attempt_root_v1(
        candidate, batches[batch_index], batch_index=batch_index
    )


def _reducer_recovery_crashed_provider_projection_v1(
    *,
    source_attempt: Path,
    batch_index: int,
    batch_id: str,
    world_ids: list[str],
    science: Mapping[str, Any],
    total_ticks: int,
    lowered: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a sealed provider batch when the parent crashed late.

    The provider batch manifest and terminal compact capture are the only
    source of authority here.  This projection never fabricates a persisted
    execution receipt and never turns a partial parent control plane into a
    normal successful child.  The ordinary reducer-only path consumes the
    resulting in-memory projections and writes fresh recovery receipts in its
    own root.
    """

    batch_manifest_path = _regular_file(
        source_attempt / "batches" / batch_id / "manifest.json",
        field="reducer recovery source provider batch manifest",
    )
    batch_manifest = _read_json(
        batch_manifest_path,
        field="reducer recovery source provider batch manifest",
    )
    if (
        batch_manifest.get("schema_version") != "eiel_campaign_batch_manifest_v1"
        or batch_manifest.get("status") != "completed"
        or batch_manifest.get("batch_id") != batch_id
        or batch_manifest.get("campaign_id") != lowered.get("campaign_id")
        or batch_manifest.get("profile_id") != lowered.get("resolved_profile")
        or batch_manifest.get("provider_id")
        != "eiel-full-evolution-cuda-native-provider-v1"
        or batch_manifest.get("cell_ids") != world_ids
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "reducer recovery source provider batch manifest is not bound to the current batch",
            details={"reducer_recovery_source_root": str(source_attempt)},
        )
    source_plan_sha256 = batch_manifest.get("plan_sha256")
    if not isinstance(source_plan_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", source_plan_sha256
    ) is None:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "reducer recovery source provider batch manifest lacks its plan digest",
        )

    source_result = batch_manifest.get("result")
    if not isinstance(source_result, Mapping):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "reducer recovery source provider batch result is absent",
        )
    capture = source_result.get("compact_science_capture_index")
    if not isinstance(capture, Mapping):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery source lacks its terminal compact capture index",
        )
    raw_capture_path = capture.get("path")
    if not isinstance(raw_capture_path, str) or not raw_capture_path:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery source compact capture index path is absent",
        )
    capture_candidate = Path(raw_capture_path).expanduser()
    if not capture_candidate.is_absolute():
        capture_candidate = source_attempt / capture_candidate
    capture_index = _regular_file(
        capture_candidate,
        field="reducer recovery source terminal compact capture index",
    )
    try:
        capture_index.relative_to(source_attempt.resolve())
    except ValueError as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery source capture index escapes its child root",
        ) from error
    try:
        if capture_index.stat().st_mode & 0o222:
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_AGGREGATION_FAILED,
                "reducer recovery source capture index is not sealed read-only",
            )
    except OSError as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery source capture index cannot be inspected",
        ) from error
    capture_index_payload = _read_json(
        capture_index,
        field="reducer recovery source terminal compact capture index",
    )
    capture_body = {
        key: value
        for key, value in capture_index_payload.items()
        if key not in {"index_sha256", "path"}
    }
    source_index_sha256 = capture_index_payload.get("index_sha256")
    if (
        not isinstance(source_index_sha256, str)
        or source_index_sha256 != canonical_sha256(capture_body).hex
        or capture.get("index_sha256") != source_index_sha256
        or capture_index_payload.get("schema_version")
        != "eiel_cuda_v2_compact_science_capture_index_v1"
        or capture_index_payload.get("batch_id") != batch_id
        or capture_index_payload.get("ordered_cell_ids") != world_ids
        or capture_index_payload.get("observation_abi")
        != science.get("observation_abi")
        or capture_index_payload.get("terminal_next_absolute_tick") != total_ticks
        or capture_index_payload.get("dropped_record_count") != 0
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery source capture index is not an authenticated terminal batch",
            details={"reducer_recovery_source_root": str(source_attempt)},
        )
    expected_control_set = _sha256_digest_hex(
        science.get("native_world_control_set_sha256"),
        field="reducer recovery expected materialization control-set digest",
    )
    if capture_index_payload.get("native_world_control_set_sha256") != expected_control_set:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery source capture control set differs from the current batch",
        )
    dispatches = capture_index_payload.get("dispatch_artifacts")
    snapshots = capture_index_payload.get("snapshot_artifacts")
    if (
        not isinstance(dispatches, list)
        or not isinstance(snapshots, list)
        or not dispatches
        or not snapshots
        or any(not isinstance(pointer, Mapping) for pointer in (*dispatches, *snapshots))
        or source_result.get("compact_science_dispatch_artifact_count")
        != len(dispatches)
        or source_result.get("compact_science_snapshot_artifact_count")
        != len(snapshots)
        or source_result.get("world_count") != len(world_ids)
        or source_result.get("next_absolute_tick") != total_ticks
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery source capture ownership counts are invalid",
        )

    relative_capture = capture_index.relative_to(source_attempt.resolve()).as_posix()
    inventory = batch_manifest.get("artifact_inventory")
    inventory_rows = (
        [row for row in inventory if isinstance(row, Mapping)]
        if isinstance(inventory, list)
        else []
    )
    matching_inventory = [
        row
        for row in inventory_rows
        if row.get("path") == relative_capture
        and row.get("sha256") == _file_sha256(capture_index)
    ]
    if len(matching_inventory) != 1:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery source capture index is absent from provider inventory",
        )
    for pointer in (*dispatches, *snapshots):
        raw_path = pointer.get("path")
        artifact_sha256 = pointer.get("artifact_sha256")
        if (
            not isinstance(raw_path, str)
            or not isinstance(artifact_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_AGGREGATION_FAILED,
                "reducer recovery source artifact pointer is invalid",
            )
        artifact_path = Path(raw_path).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = source_attempt / artifact_path
        try:
            artifact_path.resolve().relative_to(source_attempt.resolve())
        except ValueError as error:
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_AGGREGATION_FAILED,
                "reducer recovery source artifact escapes its child root",
            ) from error
        _regular_file(
            artifact_path / "manifest.json",
            field="reducer recovery source artifact manifest",
        )

    source_execution_core = {
        "schema_version": "eiel_science_instrument_execution_result_v1",
        "operator_schema_version": SCIENCE_INSTRUMENT_OPERATOR_SCHEMA_VERSION,
        "engineering_only": False,
        "scientific_execution": True,
        "prequal_started": False,
        "qualification_started": False,
        "batch_index": batch_index,
        "runtime_root": str(source_attempt),
        "campaign_id": batch_manifest["campaign_id"],
        "profile_id": batch_manifest["profile_id"],
        "plan_sha256": source_plan_sha256,
        "purpose": "simulation",
        "ok": True,
        "status": "completed",
        "execution_status": "completed",
        "result_class": "SUCCESS",
        "source_projection": {
            "schema_version": "eiel_science_reducer_recovery_source_projection_v1",
            "mode": "PRODUCER_BATCH_MANIFEST_RECOVERY",
            "source_batch_manifest_path": str(batch_manifest_path),
            "source_batch_manifest_file_sha256": _file_sha256(batch_manifest_path),
            "source_capture_index_path": str(capture_index),
            "source_capture_index_file_sha256": _file_sha256(capture_index),
            "source_capture_index_sha256": source_index_sha256,
            "provider_batch_status": "completed",
            "parent_crashed_after_producer": True,
        },
    }
    source_execution = {
        **source_execution_core,
        "result_sha256": canonical_sha256(source_execution_core).hex,
    }
    source_execution_sha256 = source_execution["result_sha256"]
    source_reducer_core = {
        "status": "CRASHED_AFTER_PRODUCER",
        "batch_id": batch_id,
        "science_compact_root": str(capture_index.parent),
        "execution_route": "native_cpp_v1",
        "producer_completed_before_tail_wait": True,
        "recovery_only": True,
        "source_projection": dict(source_execution_core["source_projection"]),
    }
    source_reducer_receipt_core = {
        "schema_version": "eiel_native_compact_reducer_execution_receipt_v1",
        "technical_only": True,
        "qualification_started": False,
        "producer": {
            "ok": True,
            "status": "completed",
            "result_class": "SUCCESS",
        },
        "reducer": source_reducer_core,
        "source_projection": dict(source_execution_core["source_projection"]),
    }
    source_reducer_receipt = {
        **source_reducer_receipt_core,
        "receipt_sha256": canonical_sha256(source_reducer_receipt_core).hex,
    }
    source_lifecycle = {
        "schema_version": "eiel_science_reducer_recovery_source_projection_v1",
        "mode": "PRODUCER_BATCH_MANIFEST_RECOVERY",
        "terminal_lifecycle_commit_available": False,
        "lifecycle_closeout_unavailable": True,
        "source_batch_manifest_path": str(batch_manifest_path),
        "source_batch_manifest_file_sha256": _file_sha256(batch_manifest_path),
        "source_capture_index_path": str(capture_index),
        "source_capture_index_file_sha256": _file_sha256(capture_index),
        "source_capture_index_sha256": source_index_sha256,
        "source_batch_manifest_status": "completed",
    }
    return {
        "source_execution": source_execution,
        "source_execution_sha256": source_execution_sha256,
        "source_reducer": source_reducer_core,
        "source_reducer_receipt": source_reducer_receipt,
        "source_reducer_sha256": source_reducer_receipt["receipt_sha256"],
        "source_lifecycle": source_lifecycle,
        "source_execution_path": batch_manifest_path,
        "source_reducer_path": batch_manifest_path,
        "capture_index": capture_index,
    }


def _rebind_reducer_results_to_current_bundle_v1(
    atlas: Mapping[str, Any],
    strategy: Mapping[str, Any] | None,
    *,
    source_bundle: ScienceWorldBundleV1,
    current_bundle: ScienceWorldBundleV1,
    world_ids: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Rebind a sealed reducer result across an identity-only bundle repair.

    A reducer replay may need the source-bound bundle to validate the sealed
    capture.  If that bundle differs only in acquisition/reducer-contract
    identity, the scientific world semantics are still checked against the
    current materialization and the small result is re-authenticated under
    that current identity.  The bulk capture remains source-bound and is
    named in the recovery provenance by the caller.
    """

    if source_bundle.bundle_content_sha256() == current_bundle.bundle_content_sha256():
        return dict(atlas), None if strategy is None else dict(strategy)

    source_worlds = {world.world_id: world for world in source_bundle.worlds}
    current_worlds = {world.world_id: world for world in current_bundle.worlds}
    if set(source_worlds) != set(current_worlds):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "reducer recovery source/current bundle ownership differs",
        )
    for world_id in world_ids:
        source_world = source_worlds.get(world_id)
        current_world = current_worlds.get(world_id)
        if (
            source_world is None
            or current_world is None
            or source_world.semantic_world_sha256
            != current_world.semantic_world_sha256
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_MATERIALIZATION_FAILED,
                "reducer recovery source/current bundle semantics differ",
            )

    def project_result(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
        projected = deepcopy(dict(value))
        rows = projected.get("world_results")
        if not isinstance(rows, list) or [
            row.get("world_id") for row in rows if isinstance(row, Mapping)
        ] != list(world_ids):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_AGGREGATION_FAILED,
                f"reducer recovery {label} world order differs",
            )
        rebound_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ScienceRequestV1FrontDoorError(
                    SCIENCE_AGGREGATION_FAILED,
                    f"reducer recovery {label} world row is invalid",
                )
            world_id = row.get("world_id")
            source_world = source_worlds[world_id]
            current_world = current_worlds[world_id]
            if (
                row.get("semantic_world_sha256")
                != source_world.semantic_world_sha256.to_dict()
                or row.get("acquisition_sha256")
                != source_world.acquisition_sha256.to_dict()
                or row.get("science") != source_world.science.to_value()
                or row.get("dimension_bindings")
                != source_world.semantic_inputs.to_value()["dimension_bindings"]
            ):
                raise ScienceRequestV1FrontDoorError(
                    SCIENCE_AGGREGATION_FAILED,
                    f"reducer recovery {label} source row identity differs",
                )
            rebound = dict(row)
            rebound["semantic_world_sha256"] = (
                current_world.semantic_world_sha256.to_dict()
            )
            rebound["acquisition_sha256"] = current_world.acquisition_sha256.to_dict()
            rebound.pop("result_sha256", None)
            rebound["result_sha256"] = canonical_sha256(rebound).to_dict()
            rebound_rows.append(rebound)
        projected["bundle_content_sha256"] = current_bundle.bundle_content_sha256().to_dict()
        projected["world_results"] = rebound_rows
        projected.pop("result_sha256", None)
        projected["result_sha256"] = canonical_sha256(projected).to_dict()
        return projected

    rebound_atlas = project_result(atlas, label="atlas result")
    rebound_strategy = (
        None if strategy is None else project_result(strategy, label="strategy result")
    )
    return rebound_atlas, rebound_strategy


def _reducer_recovery_batch_executor_v1(
    materialization_root: Path,
    *,
    batch_index: int,
    runtime_root: Path,
    source_root: Path,
    compilation: ScienceRequestV1CompilationV1,
    materialization_receipt: Mapping[str, Any],
    lowered_batches: tuple[dict[str, Any], ...],
    compact_reducer_workers: int,
    **_: Any,
) -> dict[str, Any]:
    """Replay one sealed reducer input after a completed provider execution.

    This is deliberately an executor-shaped callback so the normal science
    front door still owns storage, journaling, authentication, aggregation,
    and closeout.  It never invokes Launchpad or a provider and never copies
    the sealed producer payload into the recovery root.
    """

    if not materialization_root.is_dir():
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            f"reducer recovery materialization is absent: {materialization_root}",
        )
    if type(compact_reducer_workers) is not int or compact_reducer_workers < 1:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery requires at least one compact reducer worker",
        )
    if not 0 <= batch_index < len(lowered_batches):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            f"reducer recovery batch index is outside the materialization: {batch_index}",
        )
    lowered = lowered_batches[batch_index]
    science = lowered.get("science_materialization")
    if not isinstance(science, Mapping):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            f"reducer recovery batch {batch_index} lacks science materialization metadata",
        )
    world_ids = science.get("semantic_world_ids")
    if (
        not isinstance(world_ids, list)
        or not world_ids
        or any(not isinstance(value, str) or not value for value in world_ids)
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            f"reducer recovery batch {batch_index} has invalid world ownership",
        )
    physical_batches = lowered.get("physical_batches")
    if not isinstance(physical_batches, list) or len(physical_batches) != 1:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "reducer recovery requires one physical batch per child",
        )
    physical_batch = physical_batches[0]
    batch_id = (
        physical_batch.get("batch_id") if isinstance(physical_batch, Mapping) else None
    )
    if not isinstance(batch_id, str) or not batch_id or Path(batch_id).name != batch_id:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "reducer recovery physical batch identity is invalid",
        )
    execution = lowered.get("execution")
    total_ticks = execution.get("total_ticks") if isinstance(execution, Mapping) else None
    if type(total_ticks) is not int or total_ticks < 1:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "reducer recovery execution total_ticks is invalid",
        )
    recovery_tail_budget_seconds, recovery_tail_budget_source = (
        resolve_compact_reducer_tail_seconds_v1(
            None,
            world_count=len(world_ids),
            total_ticks=total_ticks,
        )
    )

    source_attempt = _reducer_recovery_source_attempt_root_v1(
        source_root, batch_index=batch_index
    )
    source_execution_file = source_attempt / "execution_result.json"
    source_reducer_file = source_attempt / "compact_reducer_execution_receipt.json"
    source_lifecycle_file = source_attempt / "terminal_lifecycle_commit.json"
    complete_recovery_control = all(
        path.is_file() and not path.is_symlink()
        for path in (source_execution_file, source_reducer_file, source_lifecycle_file)
    )
    if complete_recovery_control:
        source_execution_path = _regular_file(
            source_execution_file,
            field="reducer recovery source execution result",
        )
        source_execution = _read_json(
            source_execution_path, field="reducer recovery source execution result"
        )
        source_execution_core = dict(source_execution)
        source_execution_sha256 = source_execution_core.pop("result_sha256", None)
        if (
            not isinstance(source_execution_sha256, str)
            or source_execution_sha256 != canonical_sha256(source_execution_core).hex
            or source_execution.get("ok") is not True
            or source_execution.get("status") != "completed"
            or source_execution.get("execution_status") != "completed"
            or source_execution.get("result_class") != "SUCCESS"
            or source_execution.get("run_root") != str(source_attempt)
            or source_execution.get("campaign_id") != lowered.get("campaign_id")
            or source_execution.get("profile_id") != lowered.get("resolved_profile")
            or source_execution.get("purpose") != "simulation"
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                "reducer recovery source is not an authenticated completed provider result",
                details={"reducer_recovery_source_root": str(source_attempt)},
            )
        source_plan_sha256 = source_execution.get("plan_sha256")
        if not isinstance(source_plan_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", source_plan_sha256
        ) is None:
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                "reducer recovery source lacks its bound execution plan digest",
            )
        source_lifecycle = _verify_child_launchpad_lifecycle_v1(
            batch_root=source_attempt,
            front_door_result=source_execution,
            bound_plan_sha256=source_plan_sha256,
        )
        source_reducer_path = _regular_file(
            source_reducer_file,
            field="reducer recovery source reducer receipt",
        )
        source_reducer_receipt = _read_json(
            source_reducer_path, field="reducer recovery source reducer receipt"
        )
        source_reducer_core = dict(source_reducer_receipt)
        source_reducer_sha256 = source_reducer_core.pop("receipt_sha256", None)
        if (
            not isinstance(source_reducer_sha256, str)
            or source_reducer_sha256 != canonical_sha256(source_reducer_core).hex
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                "reducer recovery source reducer receipt digest is invalid",
            )
        source_reducer = source_reducer_receipt.get("reducer")
        producer = source_reducer_receipt.get("producer")
        if (
            not isinstance(source_reducer, Mapping)
            or not isinstance(producer, Mapping)
            or producer.get("ok") is not True
            or producer.get("status") != "completed"
            or producer.get("result_class") != "SUCCESS"
            or source_reducer.get("status") not in {"FAILED", "TAIL_TIMEOUT"}
            or source_reducer.get("execution_route") != "native_cpp_v1"
            or source_reducer.get("batch_id") != batch_id
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                "reducer recovery source is not a completed-provider reducer failure",
                details={"reducer_recovery_source_root": str(source_attempt)},
            )
    else:
        projection = _reducer_recovery_crashed_provider_projection_v1(
            source_attempt=source_attempt,
            batch_index=batch_index,
            batch_id=batch_id,
            world_ids=world_ids,
            science=science,
            total_ticks=total_ticks,
            lowered=lowered,
        )
        source_execution = projection["source_execution"]
        source_execution_sha256 = projection["source_execution_sha256"]
        source_reducer = projection["source_reducer"]
        source_reducer_receipt = projection["source_reducer_receipt"]
        source_reducer_sha256 = projection["source_reducer_sha256"]
        source_lifecycle = projection["source_lifecycle"]
        source_execution_path = projection["source_execution_path"]
        source_reducer_path = projection["source_reducer_path"]
    raw_compact_root = source_reducer.get("science_compact_root")
    if not isinstance(raw_compact_root, str) or not raw_compact_root:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery source lacks its compact root",
        )
    compact_root_candidate = Path(raw_compact_root).expanduser()
    if not compact_root_candidate.is_absolute():
        compact_root_candidate = source_attempt / compact_root_candidate
    if compact_root_candidate.is_symlink():
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery source compact root is a symbolic link",
        )
    compact_root = compact_root_candidate.resolve()
    try:
        compact_root.relative_to(source_attempt.resolve())
    except ValueError as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery source compact root escapes its child root",
        ) from error
    capture_index = _regular_file(
        compact_root / "capture-index-terminal.json",
        field="reducer recovery terminal capture index",
    )
    capture_index_payload = _read_json(
        capture_index, field="reducer recovery terminal capture index"
    )
    expected_observation_abi = science.get("observation_abi")
    expected_native_control_set = science.get("native_world_control_set_sha256")
    source_native_control_set = capture_index_payload.get(
        "native_world_control_set_sha256"
    )
    expected_native_control_set = _sha256_digest_hex(
        expected_native_control_set,
        field="reducer recovery materialization control-set digest",
    )
    source_native_control_set = _sha256_digest_hex(
        source_native_control_set,
        field="reducer recovery capture-index control-set digest",
    )
    if (
        capture_index_payload.get("batch_id") != batch_id
        or capture_index_payload.get("ordered_cell_ids") != world_ids
        or capture_index_payload.get("observation_abi") != expected_observation_abi
        or source_native_control_set != expected_native_control_set
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery capture index is not bound to the current science batch",
            details={"reducer_recovery_source_root": str(source_attempt)},
        )

    current_bundle = compilation.world_bundle
    bundle = current_bundle
    source_bound_bundle_path: Path | None = None
    source_bound_bundle: ScienceWorldBundleV1 | None = None
    current_bundle_content_sha256 = current_bundle.bundle_content_sha256().hex
    source_passport_candidates = [
        source_root / "run_passport.json",
        source_attempt / "run_passport.json",
    ]
    if source_attempt.parents:
        source_passport_candidates.append(source_attempt.parents[2] / "run_passport.json")
    for passport_path in source_passport_candidates:
        if not passport_path.is_file() or passport_path.is_symlink():
            continue
        passport = _read_json(
            passport_path, field="reducer recovery source run passport"
        )
        source_identity = passport.get("source_identity")
        if not isinstance(source_identity, Mapping):
            continue
        source_bundle_identity = source_identity.get("bundle_content_sha256")
        source_materialization = source_identity.get("materialization_root")
        if (
            not isinstance(source_materialization, str)
            or not source_materialization
            or not isinstance(source_bundle_identity, Mapping)
        ):
            continue
        candidate_path = Path(source_materialization).expanduser()
        try:
            candidate_path = candidate_path.resolve()
        except OSError:
            continue
        candidate_bundle_path = candidate_path / "world_bundle.json"
        if not candidate_bundle_path.is_file() or candidate_bundle_path.is_symlink():
            continue
        try:
            candidate_bundle = ScienceWorldBundleV1.from_dict(
                _read_json(
                    candidate_bundle_path,
                    field="reducer recovery source-bound world bundle",
                )
            )
        except (OSError, TypeError, ValueError):
            continue
        if candidate_bundle.bundle_content_sha256().to_dict() != dict(
            source_bundle_identity
        ):
            continue
        if candidate_bundle.bundle_content_sha256().hex == current_bundle_content_sha256:
            break
        current_worlds = {world.world_id: world for world in current_bundle.worlds}
        source_worlds = {world.world_id: world for world in candidate_bundle.worlds}
        if set(current_worlds) != set(source_worlds):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_MATERIALIZATION_FAILED,
                "reducer recovery source-bound bundle world ownership differs",
            )
        if any(
            current_worlds[world_id].semantic_world_sha256
            != source_worlds[world_id].semantic_world_sha256
            for world_id in current_worlds
        ):
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_MATERIALIZATION_FAILED,
                "reducer recovery source-bound bundle semantics differ",
            )
        source_bound_bundle = candidate_bundle
        source_bound_bundle_path = candidate_bundle_path
        bundle = candidate_bundle
        break

    bundle_worlds = {world.world_id: world for world in bundle.worlds}
    if any(world_id not in bundle_worlds for world_id in world_ids):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "reducer recovery source names a world outside the current materialization",
        )
    selected_worlds = tuple(bundle_worlds[world_id] for world_id in world_ids)
    if tuple(world.world_id for world in bundle.worlds) == tuple(world_ids):
        batch_bundle = bundle
    else:
        batch_bundle = ScienceWorldBundleV1(
            bundle_id=f"{bundle.bundle_id}--{batch_id}",
            compact_source_specs=bundle.compact_source_specs,
            compiler_component_sha256=bundle.compiler_component_sha256,
            worlds=selected_worlds,
            instrument_contract=bundle.instrument_contract,
            schema=bundle.schema,
            schema_version=bundle.schema_version,
        )

    try:
        sealed_capture_index = _sealed_terminal_index_v1(capture_index, bundle)
    except CompactWorldAtlasReducerError as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            f"reducer recovery source compact capture authentication failed: {error}",
            details={"reducer_recovery_source_root": str(source_attempt)},
        ) from error
    if sealed_capture_index is None:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery source compact capture is not fully sealed",
            details={"reducer_recovery_source_root": str(source_attempt)},
        )

    try:
        native_artifact, native_manifest, native_manifest_sha256 = (
            default_native_compact_reducer_binding_v1()
        )
        native_binding = load_native_compact_reducer_v1(
            native_artifact,
            native_manifest,
            require_production_qualified=False,
            expected_manifest_sha256=native_manifest_sha256,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            f"reducer recovery native compact reducer is unavailable: {error}",
        ) from error

    reduction_started = time.monotonic()
    reducer: CompactWorldAtlasIncrementalReducerV1 | None = None
    try:
        reducer = CompactWorldAtlasIncrementalReducerV1(
            batch_bundle,
            max_workers=compact_reducer_workers,
            native_artifact=native_binding,
            output_root=runtime_root,
        )
        atlas = reducer.finalize(capture_index)
        strategy_attribution_result = reducer.strategy_attribution_result
    except (CompactWorldAtlasReducerError, OSError, TypeError, ValueError) as error:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            f"reducer recovery compact atlas reduction failed: {error}",
            details={"reducer_recovery_source_root": str(source_attempt)},
        ) from error
    finally:
        if reducer is not None:
            reducer.close()
    reduction_seconds = time.monotonic() - reduction_started
    if not isinstance(atlas, Mapping):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "reducer recovery compact atlas returned a non-object result",
        )
    if source_bound_bundle is not None:
        atlas, strategy_attribution_result = (
            _rebind_reducer_results_to_current_bundle_v1(
                atlas,
                strategy_attribution_result,
                source_bundle=source_bound_bundle,
                current_bundle=current_bundle,
                world_ids=tuple(world_ids),
            )
        )

    identities = materialization_receipt.get("identities")
    source = materialization_receipt.get("lineage")
    if not isinstance(identities, Mapping) or not isinstance(source, Mapping):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "reducer recovery materialization identities are incomplete",
        )
    science_use_class = lowered.get("science_use_class")
    observation_mode = science.get("observation_mode")
    lowered_plan_sha256 = lowered.get("plan_sha256")
    if (
        not isinstance(science_use_class, str)
        or not isinstance(observation_mode, str)
        or not isinstance(lowered_plan_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", lowered_plan_sha256) is None
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "reducer recovery materialization execution identity is incomplete",
        )
    source_lineage = {
        **dict(source),
        "compiled_plan_sha256": lowered_plan_sha256,
        "bound_plan_sha256": canonical_sha256(
            {
                "reducer_recovery_source_execution_result_sha256": source_execution_sha256,
                "runtime_root": str(runtime_root),
            }
        ).hex,
    }
    science_lineage = {
        "schema_version": SCIENCE_INSTRUMENT_EXECUTION_LINEAGE_SCHEMA_VERSION,
        "instrument_contract": "EIEL_SCIENCE_INSTRUMENT_V1",
        "science_use_class": science_use_class,
        "observation_mode": observation_mode,
        "request_sha256": identities.get("request_sha256"),
        "campaign_spec_sha256": identities.get("campaign_spec_sha256"),
        "compilation_bundle_sha256": identities.get("compilation_bundle_sha256"),
        "bundle_content_sha256": identities.get("bundle_content_sha256"),
        "bundle_admission_sha256": identities.get("bundle_admission_sha256"),
        "carrier_plan_sha256": identities.get("carrier_plan_sha256"),
        "base_campaign_plan_sha256": identities.get("base_campaign_plan_sha256"),
        "lowered_plan_sha256": lowered_plan_sha256,
        "batch_index": batch_index,
        "world_ids": list(world_ids),
    }
    recovery_provenance = {
        "schema_version": "eiel_science_reducer_recovery_provenance_v1",
        "mode": "REDUCER_ONLY",
        "simulation_rerun": False,
        "provider_rerun": False,
        "source_attempt_root": str(source_attempt),
        "source_execution_result_sha256": source_execution_sha256,
        "source_reducer_receipt_path": str(source_reducer_path),
        "source_reducer_receipt_sha256": source_reducer_sha256,
        "source_capture_index_path": str(capture_index),
        "source_capture_index_file_sha256": _file_sha256(capture_index),
        "source_bound_materialization_root": (
            None
            if source_bound_bundle_path is None
            else str(source_bound_bundle_path.parent)
        ),
        "source_bound_bundle_content_sha256": (
            None
            if source_bound_bundle is None
            else source_bound_bundle.bundle_content_sha256().to_dict()
        ),
        "current_compilation_bundle_content_sha256": (
            current_bundle.bundle_content_sha256().to_dict()
        ),
        "reducer_identity_migration": (
            None
            if source_bound_bundle is None
            else "legacy_exact_genome_state_binding_to_exact_tracked_genotype_state_v1"
        ),
        "source_terminal_lifecycle": source_lifecycle,
        "source_reducer_failure_status": source_reducer.get("status"),
        "source_campaign_id": source_execution.get("campaign_id"),
        "source_profile_id": source_execution.get("profile_id"),
        "source_batch_id": capture_index_payload.get("batch_id"),
        "source_observation_abi": capture_index_payload.get("observation_abi"),
        "source_native_world_control_set_sha256": capture_index_payload.get(
            "native_world_control_set_sha256"
        ),
        "native_reducer_manifest_sha256": native_manifest_sha256,
        "native_artifact_use_class": NATIVE_REDUCER_ARTIFACT_USE_DEFAULT_TECHNICAL_V1,
        "reduction_seconds": reduction_seconds,
        "tail_budget_seconds": recovery_tail_budget_seconds,
        "tail_budget_source": recovery_tail_budget_source,
        "tail_slo_met": reduction_seconds <= recovery_tail_budget_seconds,
    }
    compact_reducer = {
        "schema_version": "eiel_compact_world_atlas_recovery_execution_v1",
        "enabled": True,
        "asynchronous": False,
        "producer_notification_required": False,
        "producer_backpressure_path": False,
        "producer_waits_for_reducer_progress": False,
        "tail_budget_seconds": recovery_tail_budget_seconds,
        "tail_budget_source": recovery_tail_budget_source,
        "status": "COMPLETED",
        "batch_id": batch_id,
        "science_compact_root": str(compact_root),
        "max_workers": compact_reducer_workers,
        "execution_route": "native_cpp_v1",
        "native_binding": "checked_in_default",
        "native_manifest_sha256": native_manifest_sha256,
        "native_artifact_use_class": NATIVE_REDUCER_ARTIFACT_USE_DEFAULT_TECHNICAL_V1,
        "tail_seconds": reduction_seconds,
        "tail_slo_met": reduction_seconds <= recovery_tail_budget_seconds,
        "producer_completed_before_tail_wait": True,
        "recovery_only": True,
        "recovery_consumer_boundary": "post_terminal_offline_reducer_replay",
        "atlas_result": dict(atlas),
        "strategy_attribution_result": (
            None
            if strategy_attribution_result is None
            else dict(strategy_attribution_result)
        ),
        "second_python_attribution_pass": False,
        "recovery_provenance": recovery_provenance,
    }
    atlas_identity = {
        key: atlas.get(key)
        for key in (
            "schema_version",
            "reducer_id",
            "reducer_contract_version",
            "result_sha256",
            "capture_index_sha256",
            "logical_world_count",
            "physical_batch_count",
            "physical_composition_used",
        )
        if key in atlas
    }
    reducer_receipt_core = {
        "schema_version": "eiel_native_compact_reducer_execution_receipt_v1",
        "technical_only": True,
        "qualification_started": False,
        "science_use_class": science_use_class,
        "batch_index": batch_index,
        "runtime_root": str(runtime_root),
        "producer": {
            "ok": source_execution.get("ok"),
            "status": source_execution.get("status"),
            "result_class": source_execution.get("result_class"),
        },
        "reducer": {
            key: value
            for key, value in compact_reducer.items()
            if key not in {"atlas_result", "strategy_attribution_result"}
        },
        "atlas_identity": atlas_identity,
        "strategy_identity": (
            None
            if strategy_attribution_result is None
            else {
                "schema_version": strategy_attribution_result.get("schema_version"),
                "result_sha256": strategy_attribution_result.get("result_sha256"),
                "canonical_sha256": canonical_sha256(
                    dict(strategy_attribution_result)
                ).hex,
            }
        ),
        "recovery_provenance": recovery_provenance,
    }
    reducer_receipt = {
        **reducer_receipt_core,
        "receipt_sha256": canonical_sha256(reducer_receipt_core).hex,
    }
    _write_immutable_json(
        runtime_root / "compact_reducer_execution_receipt.json",
        reducer_receipt,
        field="reducer recovery execution receipt",
    )
    return {
        "schema_version": "eiel_science_instrument_execution_result_v1",
        "operator_schema_version": SCIENCE_INSTRUMENT_OPERATOR_SCHEMA_VERSION,
        "engineering_only": False,
        "scientific_execution": True,
        "prequal_started": False,
        "qualification_started": False,
        "batch_index": batch_index,
        "runtime_root": str(runtime_root),
        "source_lineage": source_lineage,
        "science_lineage": science_lineage,
        "ok": True,
        "status": "completed",
        "result_class": "SUCCESS",
        "compact_reducer": compact_reducer,
        "recovery_provenance": recovery_provenance,
    }


def _wait_status_from_result_v1(value: object) -> str | None:
    if isinstance(value, StorageBudgetExceeded):
        return WAIT_STORAGE
    if isinstance(value, ScienceRequestV1FrontDoorError):
        if value.code in {WAIT_RESOURCE, WAIT_STORAGE}:
            return value.code
        details = value.details
        launchpad_class = details.get("launchpad_result_class")
        launchpad_warnings = details.get("launchpad_warnings")
        if (
            launchpad_class == _RETRYABLE_ATTEMPT_ADMISSION_RESULT_CLASS_V1
            and isinstance(launchpad_warnings, list)
            and any(
                _retryable_admission_warning_v1(item)
                for item in launchpad_warnings
            )
        ):
            return WAIT_RESOURCE
        message = str(value)
        if (
            _RETRYABLE_ATTEMPT_ADMISSION_RESULT_CLASS_V1 in message
            and any(
                _retryable_admission_warning_v1(line)
                for line in message.splitlines()
            )
        ):
            return WAIT_RESOURCE
        return None
    if not isinstance(value, Mapping):
        return None
    raw_status = value.get("status")
    raw_class = value.get("result_class")
    normalized = {str(raw_status).upper(), str(raw_class).upper()}
    if (
        str(raw_status).upper() == "PRECHECK_FAILED"
        and str(raw_class).upper()
        == _RETRYABLE_ATTEMPT_ADMISSION_RESULT_CLASS_V1
    ):
        return WAIT_RESOURCE
    if normalized & {"WAIT_RESOURCE", "RESOURCE_WAIT"}:
        return WAIT_RESOURCE
    if normalized & {"WAIT_STORAGE", "STORAGE_WAIT"}:
        return WAIT_STORAGE
    return None


def run_science_request_v1(
    request: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path | None = None,
    materialization_parent: str | Path | None = None,
    runtime_parent: str | Path | None = None,
    resume_root: str | Path | None = None,
    reducer_recovery_source_root: str | Path | None = None,
    authority_resolver: ScienceAuthorityResolverV1 | None = None,
    batch_executor: ScienceBatchExecutorV1 | None = None,
    execution_front_door: Callable[..., Mapping[str, Any]] | None = None,
    crca_adapter: object | None = None,
    state_root: str | Path | None = None,
    compact_reducer_workers: int = 4,
) -> dict[str, Any]:
    """Execute every admitted batch, aggregate its reducer, and close out."""

    loaded = _load_request(request)
    try:
        compilation = compile_science_request_v1(loaded.request)
    except (TypeError, ValueError) as error:
        raise ScienceRequestV1FrontDoorError(
            "INVALID_SCIENCE_REQUEST",
            f"science request compilation failed: {error}",
            details={
                "request_sha256": loaded.request.request_sha256().to_dict(),
                "execution_started": False,
            },
        ) from error
    if reducer_recovery_source_root is not None and any(
        value is not None
        for value in (resume_root, batch_executor, execution_front_door, crca_adapter)
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "reducer-only recovery cannot be combined with resume or execution overrides",
        )
    recovery_source_root = (
        None
        if reducer_recovery_source_root is None
        else _existing_runtime_root_v1(reducer_recovery_source_root)
    )
    repository = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    plan_result, authority, materialization_root = _plan_loaded_request(
        loaded,
        compilation,
        repo_root=repository,
        materialization_parent=Path(
            materialization_parent or DEFAULT_SCIENCE_RUNTIME_ROOT / "materializations"
        ),
        authority_resolver=authority_resolver or resolve_current_science_authority_v1,
    )
    campaign_id = str(loaded.request.to_dict()["campaign_id"])
    runtime_root = (
        _existing_runtime_root_v1(resume_root)
        if resume_root is not None
        else _fresh_runtime_root(
            Path(runtime_parent or DEFAULT_SCIENCE_RUNTIME_ROOT / "runs"),
            campaign_id
            + ("-reducer-recovery" if recovery_source_root is not None else ""),
        )
    )
    request_identity = loaded.request.request_sha256()
    request_digest = request_identity.to_dict()
    base_snapshot_ref = plan_result.get("resolved_inputs", {}).get("base_campaign_plan")
    if not isinstance(base_snapshot_ref, Mapping) or not isinstance(
        base_snapshot_ref.get("path"), str
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "admitted materialization lacks its resolved base-plan snapshot",
        )
    base_snapshot = _regular_file(
        Path(base_snapshot_ref["path"]), field="resolved base campaign plan"
    )
    if base_snapshot.parent != materialization_root or _file_sha256(
        base_snapshot
    ) != base_snapshot_ref.get("file_sha256"):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "resolved base campaign plan changed before execution",
        )
    base_plan = _read_json(base_snapshot, field="resolved base campaign plan")
    (
        _,
        materialization_receipt,
        _,
        lowered_batches,
    ) = load_science_instrument_materialization_v1(materialization_root)
    _verify_materialized_batch_ownership_v1(
        compilation=compilation,
        materialization=lowered_batches,
    )
    selected_world_flags = tuple(
        _is_selected_world_lowered_batch_v1(lowered)
        for lowered in lowered_batches
    )
    if any(selected_world_flags) and (
        not all(selected_world_flags) or len(selected_world_flags) != 1
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_MATERIALIZATION_FAILED,
            "V1 selected-world host-observation closeout requires exactly one batch",
        )
    selected_world_host_closeout = bool(
        selected_world_flags and selected_world_flags[0]
    )
    if selected_world_host_closeout and recovery_source_root is not None:
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            "selected-world host observation has no compact reducer recovery phase",
        )
    planned_bytes = _science_coordinator_storage_allowance_v1(compilation)
    lifecycle_plan = _science_lifecycle_plan_v1(
        root=runtime_root,
        phase="run",
        loaded=loaded,
        compilation=compilation,
        authority=authority,
        base_plan=base_plan,
        lowered_batches=lowered_batches,
        coordinator_durable_allowance_bytes=planned_bytes,
    )
    lifecycle_source = {
        **_science_lifecycle_source_identity_v1(
            loaded=loaded,
            compilation=compilation,
            authority=authority,
        ),
        "materialization_plan_receipt_sha256": plan_result["receipt_sha256"],
        "materialization_root": str(materialization_root),
    }
    if recovery_source_root is not None:
        lifecycle_source["reducer_recovery_source_root"] = str(recovery_source_root)
    lifecycle = ensure_lifecycle_contract_v1(lifecycle_plan)
    guard = RunStorageGuard(
        lifecycle,
        run_root=runtime_root,
        planned_bytes=planned_bytes,
        reservation_window_bytes=planned_bytes,
    )
    batch_count = int(plan_result["materialization"]["batch_count"])
    batch_runtime_parent = runtime_root / "batches"
    batch_results: list[dict[str, Any]] = []
    host_observation_results: list[dict[str, Any]] = []
    batch_receipts: list[dict[str, Any]] = []
    if recovery_source_root is not None:

        def _reducer_recovery_executor(
            materialization: str | Path, **kwargs: Any
        ) -> dict[str, Any]:
            return _reducer_recovery_batch_executor_v1(
                materialization,
                source_root=recovery_source_root,
                compilation=compilation,
                materialization_receipt=materialization_receipt,
                lowered_batches=lowered_batches,
                compact_reducer_workers=compact_reducer_workers,
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key != "compact_reducer_workers"
                },
            )

        executor: ScienceBatchExecutorV1 = _reducer_recovery_executor
    else:
        executor = batch_executor or execute_materialized_science_batch_v1
    require_launchpad_lifecycle = (
        batch_executor is None and recovery_source_root is None
    )
    identity = _science_run_identity_v1(
        campaign_id=campaign_id,
        request_digest=request_digest,
        materialization_root=materialization_root,
        materialization_receipt_sha256=plan_result["receipt_sha256"],
        authority=authority,
        lifecycle_plan_sha256=lifecycle_plan["plan_sha256"],
        compilation=compilation,
        lowered_batches=lowered_batches,
        require_launchpad_lifecycle=require_launchpad_lifecycle,
    )
    start_path = runtime_root / "science_campaign_run_start.json"

    def _write_or_verify_run_start(storage_admission: Mapping[str, Any]) -> Path:
        core = {
            "schema_version": "eiel_science_request_v1_run_start_v1",
            "campaign_id": campaign_id,
            "request_sha256": request_digest,
            "runtime_root": str(runtime_root),
            "materialization_plan_receipt_sha256": plan_result["receipt_sha256"],
            "authority_sha256": authority.authority_sha256,
            "batch_count": batch_count,
            "batch_execution_authority": (
                "sealed_completed_child_reducer_recovery"
                if recovery_source_root is not None
                else "canonical_launchpad_per_batch"
            ),
            "batch_runtime_parent": str(batch_runtime_parent),
            "lifecycle_plan_sha256": lifecycle_plan["plan_sha256"],
            "storage_admission": dict(storage_admission),
            "qualification_authority_effect": False,
        }
        return _write_or_verify_json_v1(
            path=start_path,
            value={**core, "receipt_sha256": canonical_sha256(core).to_dict()},
            field="science campaign run start",
        )

    journal_path = runtime_root / "run_journal.json"
    lifecycle_started = False
    guard_released = False
    sealed_completed_receipt: dict[str, Any] | None = None
    if resume_root is not None:
        existing_plan = _read_json(
            runtime_root / "science_lifecycle_plan.json",
            field="science lifecycle plan",
        )
        existing_contract = _read_json(
            runtime_root / "artifact_lifecycle_contract.json",
            field="science artifact lifecycle contract",
        )
        if existing_plan != lifecycle_plan or existing_contract != lifecycle:
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                "science resume lifecycle binding differs from the current request",
                details={"runtime_root": str(runtime_root)},
            )
        lifecycle_started = True
        journal = _read_science_run_journal_v1(runtime_root)
        _validate_science_run_journal_identity_v1(journal, expected=identity)
        if journal.get("status") == "FAILED":
            retry_batch_index = _failed_batch_retry_index_v1(runtime_root, journal)
            if retry_batch_index is None:
                raise ScienceRequestV1FrontDoorError(
                    SCIENCE_BATCH_EXECUTION_FAILED,
                    "science resume root contains a terminal failure; automatic retry is forbidden",
                    details={"runtime_root": str(runtime_root)},
                )
            row = _advance_retryable_science_batch_attempt_v1(
                root=runtime_root,
                row=journal["batches"][retry_batch_index],
                batch_index=retry_batch_index,
                reason="RETRYABLE_ATTEMPT_ADMISSION_FAILURE",
            )
            journal["batches"][retry_batch_index] = row
            journal["status"] = "RUNNING"
            journal["last_event"] = (
                f"batch-{retry_batch_index:04d}-retryable-admission-resume"
            )
            _write_science_run_journal_v1(runtime_root, journal)
        if journal.get("status") == "COMPLETED":
            # A completed campaign is reusable only when every child receipt
            # still authenticates against the bound attempt, materialization,
            # and canonical child lifecycle.  Do this before returning the
            # cached campaign receipt so a tampered child can never be skipped.
            for batch_index, journal_row in enumerate(journal["batches"]):
                if journal_row.get("state") != "COMPLETED":
                    raise ScienceRequestV1FrontDoorError(
                        SCIENCE_BATCH_EXECUTION_FAILED,
                        "completed science run journal contains a non-completed batch",
                        details={
                            "runtime_root": str(runtime_root),
                            "batch_index": batch_index,
                        },
                    )
                _load_completed_science_batch_v1(
                    root=runtime_root,
                    row=journal_row,
                    batch_index=batch_index,
                    lowered=lowered_batches[batch_index],
                    materialization_receipt=materialization_receipt,
                    require_launchpad_lifecycle=require_launchpad_lifecycle,
                )
            receipt_path = _regular_file(
                runtime_root / "science_campaign_run_receipt.json",
                field="science campaign run receipt",
            )
            receipt = _read_json(receipt_path, field="science campaign run receipt")
            declared = receipt.get("receipt_sha256")
            receipt_core = {
                key: item for key, item in receipt.items() if key != "receipt_sha256"
            }
            if declared != canonical_sha256(receipt_core).to_dict():
                raise ScienceRequestV1FrontDoorError(
                    SCIENCE_BATCH_EXECUTION_FAILED,
                    "completed science run receipt digest mismatch",
                )
            if (
                receipt.get("campaign_id") != campaign_id
                or receipt.get("request_sha256") != request_digest
                or receipt.get("runtime_root") != str(runtime_root)
                or receipt.get("status") != "COMPLETED"
                or receipt.get("result_class") != "SCIENCE_CAMPAIGN_COMPLETED"
            ):
                raise ScienceRequestV1FrontDoorError(
                    SCIENCE_BATCH_EXECUTION_FAILED,
                    "completed science run receipt identity differs",
                )
            sealed_completed_receipt = receipt
            try:
                closeout = _verify_science_lifecycle_closeout_v1(
                    root=runtime_root,
                    expected_plan_sha256=str(lifecycle_plan["plan_sha256"]),
                )
            except ScienceRequestV1FrontDoorError:
                # A process may have terminated after sealing the completed
                # journal but before lifecycle closeout.  Continue below and
                # let the canonical closeout repair the same bound root.
                journal["status"] = "RUNNING"
                journal["last_event"] = "resume_after_unfinished_closeout"
                _write_science_run_journal_v1(runtime_root, journal)
            else:
                return {**receipt, "receipt_path": str(receipt_path), "lifecycle_closeout": closeout}
        if sealed_completed_receipt is None and all(
            row.get("state") == "COMPLETED" for row in journal["batches"]
        ):
            receipt_path = runtime_root / "science_campaign_run_receipt.json"
            if receipt_path.is_file() and not receipt_path.is_symlink():
                receipt = _read_json(
                    receipt_path, field="science campaign run receipt"
                )
                declared = receipt.get("receipt_sha256")
                receipt_core = {
                    key: item
                    for key, item in receipt.items()
                    if key != "receipt_sha256"
                }
                if declared != canonical_sha256(receipt_core).to_dict():
                    raise ScienceRequestV1FrontDoorError(
                        SCIENCE_BATCH_EXECUTION_FAILED,
                        "completed science run receipt digest mismatch",
                    )
                if (
                    receipt.get("campaign_id") != campaign_id
                    or receipt.get("request_sha256") != request_digest
                    or receipt.get("runtime_root") != str(runtime_root)
                    or receipt.get("status") != "COMPLETED"
                    or receipt.get("result_class")
                    != "SCIENCE_CAMPAIGN_COMPLETED"
                ):
                    raise ScienceRequestV1FrontDoorError(
                        SCIENCE_BATCH_EXECUTION_FAILED,
                        "completed science run receipt identity differs",
                    )
                sealed_completed_receipt = receipt
    else:
        _write_science_lifecycle_start_v1(
            root=runtime_root,
            plan=lifecycle_plan,
            source_identity=lifecycle_source,
            provider_id="eiel-science-request-v1-orchestrator",
        )
        lifecycle_started = True
        journal = _new_science_run_journal_v1(identity=identity)
        _write_science_run_journal_v1(runtime_root, journal)

    def _return_wait(
        status: str,
        *,
        batch_index: int | None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal guard_released
        if batch_index is not None:
            row = dict(journal["batches"][batch_index])
            row["state"] = status
            row["wait"] = {
                "status": status,
                "details": dict(details or {}),
            }
            journal["batches"][batch_index] = row
        journal["status"] = status
        journal["current_batch_index"] = batch_index
        journal["last_event"] = status
        _write_science_run_journal_v1(runtime_root, journal)
        if not start_path.exists():
            _write_or_verify_run_start(dict(details or {}))
        if not guard_released:
            try:
                guard.release()
            except ArtifactLifecycleError:
                pass
            guard_released = True
        core = {
            "schema_version": SCIENCE_REQUEST_V1_RUN_RECEIPT_SCHEMA_VERSION,
            "ok": False,
            "status": status,
            "result_class": "RESOURCE_WAIT"
            if status == WAIT_RESOURCE
            else "STORAGE_WAIT",
            "campaign_id": campaign_id,
            "request_sha256": request_digest,
            "runtime_root": str(runtime_root),
            "resume_token": str(runtime_root),
            "materialization_root": str(materialization_root),
            "materialization_plan_receipt_sha256": plan_result["receipt_sha256"],
            "authority_sha256": authority.authority_sha256,
            "completed_batches": batch_receipts,
            "pending_batches": [
                index
                for index, row in enumerate(journal["batches"])
                if row.get("state") != "COMPLETED"
            ],
            "journal_path": str(journal_path),
            "details": dict(details or {}),
            "closeout_completed": False,
            "execution_started": batch_index is not None,
            "qualification_authority_effect": False,
        }
        receipt = {**core, "receipt_sha256": canonical_sha256(core).to_dict()}
        wait_path = runtime_root / "science_campaign_wait_receipt.json"
        atomic_write_json(wait_path, receipt)
        return {**receipt, "receipt_path": str(wait_path)}

    try:
        storage_admission = guard.enforce(stage="before_science_campaign")
    except StorageBudgetExceeded as error:
        return _return_wait(
            WAIT_STORAGE,
            batch_index=None,
            details={
                "runtime_root": str(runtime_root),
                "storage_receipt": error.receipt,
            },
        )
    except (
        ArtifactLifecycleError,
        OSError,
        ScienceInstrumentOperatorError,
        ScienceRequestV1FrontDoorError,
    ) as error:
        try:
            guard.release()
        except ArtifactLifecycleError:
            pass
        if lifecycle_started:
            _record_science_lifecycle_failure_v1(
                root=runtime_root,
                plan=lifecycle_plan,
                source_identity=lifecycle_source,
                provider_id="eiel-science-request-v1-orchestrator",
                stage="run-initialization",
                failure_type=type(error).__name__,
                message=str(error),
            )
        if isinstance(error, ScienceRequestV1FrontDoorError):
            raise
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science campaign initialization failed: {error}",
            details={"runtime_root": str(runtime_root)},
        ) from error
    if resume_root is not None:
        start_existing = _read_json(start_path, field="science campaign run start")
        storage_admission = start_existing.get("storage_admission")
        if not isinstance(storage_admission, Mapping):
            storage_admission = {"status": "RESUMED"}
    _write_or_verify_run_start(storage_admission)
    for batch_index in range(batch_count):
        journal_row = dict(journal["batches"][batch_index])
        journal_state = journal_row.get("state")
        if journal_state == "COMPLETED":
            try:
                batch_result, batch_receipt = _load_completed_science_batch_v1(
                    root=runtime_root,
                    row=journal_row,
                    batch_index=batch_index,
                    lowered=lowered_batches[batch_index],
                    materialization_receipt=materialization_receipt,
                    require_launchpad_lifecycle=require_launchpad_lifecycle,
                )
            except ScienceRequestV1FrontDoorError:
                # An allegedly completed child that cannot be authenticated is
                # never silently rerun or skipped.
                raise
            batch_receipts.append(batch_receipt)
            if selected_world_host_closeout:
                binding = batch_result.get("selected_world_host_observation")
                if not isinstance(binding, Mapping):
                    raise ScienceRequestV1FrontDoorError(
                        SCIENCE_AGGREGATION_FAILED,
                        "selected-world completed child lacks its host observation",
                    )
                host_observation_results.append(
                    {
                        "physical_batch_index": batch_index,
                        "world_id": batch_receipt["materialized_world_ids"][0],
                        "host_boundary_observation": dict(binding),
                    }
                )
                continue
            reducer = batch_result.get("compact_reducer")
            atlas = (
                reducer.get("atlas_result") if isinstance(reducer, Mapping) else None
            )
            if not isinstance(atlas, Mapping):
                raise ScienceRequestV1FrontDoorError(
                    SCIENCE_AGGREGATION_FAILED,
                    f"science batch {batch_index} lacks a completed compact atlas result",
                    details={
                        "runtime_root": str(runtime_root),
                        "execution_started": True,
                    },
                )
            batch_results.append(
                _science_batch_result_envelope_v1(
                    result=batch_result,
                    physical_batch_index=batch_index,
                    request_sha256=request_identity,
                    compilation=compilation,
                )
            )
            continue
        if journal_state == "FAILED":
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science batch {batch_index} has a terminal failure; automatic retry is forbidden",
                details={"runtime_root": str(runtime_root), "batch_index": batch_index},
            )
        if journal_state in _RETRYABLE_BATCH_STATES_V1:
            wait_value = journal_row.get("wait")
            wait_details = (
                wait_value.get("details")
                if isinstance(wait_value, Mapping)
                else None
            )
            admission_wait = (
                isinstance(wait_details, Mapping)
                and wait_details.get("result_class")
                == _RETRYABLE_ATTEMPT_ADMISSION_RESULT_CLASS_V1
            ) or (
                _retryable_admission_result_v1(
                    root=runtime_root,
                    row=journal_row,
                    batch_index=batch_index,
                )
                is not None
            )
            if admission_wait:
                journal_row = _advance_retryable_science_batch_attempt_v1(
                    root=runtime_root,
                    row=journal_row,
                    batch_index=batch_index,
                    reason=f"RESUME_{str(journal_state).upper()}",
                )
                journal["batches"][batch_index] = journal_row
        attempt = journal_row.get("attempt", 0)
        if type(attempt) is not int or attempt < 0:
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science batch {batch_index} attempt identity is invalid",
            )
        if attempt == 0:
            attempt = 1
            journal_row["attempt"] = attempt
            journal_row["attempt_root"] = str(
                Path("batches") / f"batch-{batch_index:04d}" / f"attempt-{attempt:04d}"
            )
        batch_root = _journal_attempt_root_v1(
            runtime_root, journal_row, batch_index=batch_index
        )
        journal_row["state"] = "RUNNING"
        journal_row["wait"] = None
        journal["batches"][batch_index] = journal_row
        journal["status"] = "RUNNING"
        journal["current_batch_index"] = batch_index
        journal["last_event"] = f"batch-{batch_index:04d}-attempt-{attempt:04d}-started"
        _write_science_run_journal_v1(runtime_root, journal)
        kwargs: dict[str, Any] = {
            "batch_index": batch_index,
            "runtime_root": batch_root,
            "repo_root": repository,
            "compact_reducer_workers": compact_reducer_workers,
        }
        if require_launchpad_lifecycle:
            kwargs["nested_storage_parent_run_root"] = runtime_root
        if execution_front_door is not None:
            kwargs["execution_front_door"] = execution_front_door
        if crca_adapter is not None:
            kwargs["crca_adapter"] = crca_adapter
        if state_root is not None:
            kwargs["state_root"] = state_root
        try:
            guard.enforce(stage=f"before_science_batch:{batch_index:04d}")
            result = executor(materialization_root, **kwargs)
        except StorageBudgetExceeded as error:
            return _return_wait(
                WAIT_STORAGE,
                batch_index=batch_index,
                details={"storage_receipt": error.receipt},
            )
        except Exception as error:
            wait_status = _wait_status_from_result_v1(error)
            if wait_status is not None:
                return _return_wait(
                    wait_status,
                    batch_index=batch_index,
                    details=getattr(error, "details", None),
                )
            journal_row["state"] = "FAILED"
            journal_row["failure"] = {
                "type": type(error).__name__,
                "message": str(error)[:4096],
            }
            journal["batches"][batch_index] = journal_row
            journal["status"] = "FAILED"
            journal["last_event"] = f"batch-{batch_index:04d}-failed"
            _write_science_run_journal_v1(runtime_root, journal)
            _write_run_failure(
                runtime_root,
                request_sha256=request_digest,
                stage=f"batch-{batch_index:04d}",
                code=SCIENCE_BATCH_EXECUTION_FAILED,
                message=str(error),
            )
            try:
                guard.release()
                guard_released = True
            except ArtifactLifecycleError:
                pass
            if not (runtime_root / "artifact_disposition_manifest.json").exists():
                _record_science_lifecycle_failure_v1(
                    root=runtime_root,
                    plan=lifecycle_plan,
                    source_identity=lifecycle_source,
                    provider_id="eiel-science-request-v1-orchestrator",
                    stage=f"batch-{batch_index:04d}",
                    failure_type=type(error).__name__,
                    message=str(error),
                )
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science batch {batch_index} raised before closeout: {error}",
                details={
                    "runtime_root": str(runtime_root),
                    "batch_index": batch_index,
                    "execution_started": True,
                },
            ) from error
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            # Operator/process termination is not a fabricated batch failure.
            # Release only the reservation; the RUNNING journal state remains
            # resumable on the same attempt root.
            try:
                guard.release()
                guard_released = True
            except ArtifactLifecycleError:
                pass
            raise
        if not isinstance(result, Mapping):
            journal_row["state"] = "FAILED"
            journal_row["failure"] = {
                "type": "INVALID_BATCH_RESULT",
                "message": "batch executor returned a non-object result",
            }
            journal["batches"][batch_index] = journal_row
            journal["status"] = "FAILED"
            journal["last_event"] = f"batch-{batch_index:04d}-failed"
            _write_science_run_journal_v1(runtime_root, journal)
            _write_run_failure(
                runtime_root,
                request_sha256=request_digest,
                stage=f"batch-{batch_index:04d}",
                code=SCIENCE_BATCH_EXECUTION_FAILED,
                message="batch executor returned a non-object result",
            )
            try:
                guard.release()
                guard_released = True
            except ArtifactLifecycleError:
                pass
            _record_science_lifecycle_failure_v1(
                root=runtime_root,
                plan=lifecycle_plan,
                source_identity=lifecycle_source,
                provider_id="eiel-science-request-v1-orchestrator",
                stage=f"batch-{batch_index:04d}",
                failure_type="INVALID_BATCH_RESULT",
                message="batch executor returned a non-object result",
            )
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_BATCH_EXECUTION_FAILED,
                f"science batch {batch_index} returned a non-object result",
                details={"runtime_root": str(runtime_root), "execution_started": True},
            )
        batch_result = dict(result)
        wait_status = _wait_status_from_result_v1(batch_result)
        if wait_status is not None:
            return _return_wait(
                wait_status,
                batch_index=batch_index,
                details={
                    "status": batch_result.get("status"),
                    "result_class": batch_result.get("result_class"),
                },
            )
        batch_envelope: dict[str, Any] | None = None
        try:
            child_lifecycle = _verify_science_batch_result_v1(
                result=batch_result,
                batch_index=batch_index,
                batch_root=batch_root,
                lowered=lowered_batches[batch_index],
                materialization_receipt=materialization_receipt,
                require_launchpad_lifecycle=require_launchpad_lifecycle,
            )
            if recovery_source_root is not None:
                recovery_provenance = batch_result.get("recovery_provenance")
                if not isinstance(recovery_provenance, Mapping):
                    raise ScienceRequestV1FrontDoorError(
                        SCIENCE_BATCH_EXECUTION_FAILED,
                        "reducer recovery batch lacks source provenance",
                    )
                child_lifecycle = dict(recovery_provenance)
            batch_envelope = _science_batch_result_envelope_for_batch_v1(
                result=batch_result,
                lowered=lowered_batches[batch_index],
                physical_batch_index=batch_index,
                request_sha256=request_identity,
                compilation=compilation,
            )
        except ScienceRequestV1FrontDoorError as error:
            journal_row["state"] = "FAILED"
            journal_row["failure"] = {
                "type": error.code,
                "message": str(error)[:4096],
            }
            journal["batches"][batch_index] = journal_row
            journal["status"] = "FAILED"
            journal["last_event"] = f"batch-{batch_index:04d}-authentication-failed"
            _write_science_run_journal_v1(runtime_root, journal)
            _write_run_failure(
                runtime_root,
                request_sha256=request_digest,
                stage=f"batch-{batch_index:04d}-authentication",
                code=SCIENCE_BATCH_EXECUTION_FAILED,
                message=str(error),
            )
            try:
                guard.release()
                guard_released = True
            except ArtifactLifecycleError:
                pass
            _record_science_lifecycle_failure_v1(
                root=runtime_root,
                plan=lifecycle_plan,
                source_identity=lifecycle_source,
                provider_id="eiel-science-request-v1-orchestrator",
                stage=f"batch-{batch_index:04d}-authentication",
                failure_type=error.code,
                message=str(error),
            )
            raise
        attempt_result_path = batch_root / "batch_result.json"
        sealed_batch_result, attempt_result_file_sha256 = (
            _write_or_verify_science_batch_result_v1(
                path=attempt_result_path,
                result=batch_result,
            )
        )
        batch_result = sealed_batch_result
        batch_result_path = runtime_root / f"batch_result_{batch_index:04d}.json"
        _write_or_verify_json_v1(
            path=batch_result_path,
            value=batch_result,
            field=f"science batch {batch_index} result",
        )
        journal_row["state"] = "COMPLETED"
        journal_row["result"] = {
            "path": str(attempt_result_path.relative_to(runtime_root)),
            "file_sha256": attempt_result_file_sha256,
            "result_sha256": batch_result["result_sha256"],
        }
        journal_row["child_lifecycle"] = child_lifecycle
        journal["batches"][batch_index] = journal_row
        journal["last_event"] = f"batch-{batch_index:04d}-completed"
        _write_science_run_journal_v1(runtime_root, journal)
        batch_receipts.append(
            {
                "batch_index": batch_index,
                "path": str(batch_result_path),
                "file_sha256": _file_sha256(batch_result_path),
                "result_sha256": batch_result["result_sha256"],
                "ok": batch_result.get("ok"),
                "status": batch_result.get("status"),
                "result_class": batch_result.get("result_class"),
                "runtime_root": str(batch_root),
                "materialized_world_ids": _materialized_world_ids_v1(
                    lowered_batches[batch_index]["science_materialization"]
                ),
                "child_lifecycle": child_lifecycle,
                **(
                    {
                        "selected_world_host_observation": dict(
                            batch_result["selected_world_host_observation"]
                        )
                    }
                    if _is_selected_world_lowered_batch_v1(
                        lowered_batches[batch_index]
                    )
                    else {}
                ),
            }
        )
        if batch_result.get("ok") is not True:
            journal_row["state"] = "FAILED"
            journal_row["failure"] = {
                "type": str(batch_result.get("result_class") or "BATCH_TERMINATED"),
                "message": str(batch_result.get("status") or "batch terminated"),
            }
            journal["batches"][batch_index] = journal_row
            journal["status"] = "FAILED"
            journal["last_event"] = f"batch-{batch_index:04d}-terminal-failure"
            _write_science_run_journal_v1(runtime_root, journal)
            core = {
                "schema_version": SCIENCE_REQUEST_V1_RUN_RECEIPT_SCHEMA_VERSION,
                "ok": False,
                "status": batch_result.get("status") or "BATCH_TERMINATED",
                "result_class": batch_result.get("result_class"),
                "campaign_id": campaign_id,
                "request_sha256": request_digest,
                "runtime_root": str(runtime_root),
                "materialization_root": str(materialization_root),
                "completed_batches": batch_receipts,
                "aggregate": None,
                "lifecycle": {
                    "plan_sha256": lifecycle_plan["plan_sha256"],
                    "run_passport_path": str(runtime_root / "run_passport.json"),
                    "artifact_disposition_manifest_path": str(
                        runtime_root / "artifact_disposition_manifest.json"
                    ),
                    "storage_disposition_receipt_path": str(
                        runtime_root / "storage_disposition_receipt.json"
                    ),
                },
                "closeout_completed": True,
                "closeout_completion_condition": (
                    "authoritative only when the terminal lifecycle artifacts validate"
                ),
                "execution_started": True,
                "qualification_authority_effect": False,
            }
            receipt = {**core, "receipt_sha256": canonical_sha256(core).to_dict()}
            closeout_path = _write_immutable_json(
                runtime_root / "science_campaign_run_receipt.json",
                receipt,
                field="science campaign run receipt",
            )
            guard.enforce(stage="science_campaign_failed_closeout")
            guard.release()
            guard_released = True
            lifecycle_closeout = _publish_science_lifecycle_closeout_v1(
                root=runtime_root,
                plan=lifecycle_plan,
                source_identity=lifecycle_source,
                provider_id="eiel-science-request-v1-orchestrator",
                terminal_status="failed",
                inventory_complete=True,
                register_archive=False,
            )
            return {
                **receipt,
                "receipt_path": str(closeout_path),
                "lifecycle_closeout": lifecycle_closeout,
            }
        if selected_world_host_closeout:
            binding = batch_result.get("selected_world_host_observation")
            if not isinstance(binding, Mapping):
                raise ScienceRequestV1FrontDoorError(
                    SCIENCE_AGGREGATION_FAILED,
                    "selected-world completed child lacks its host observation",
                )
            host_observation_results.append(
                {
                    "physical_batch_index": batch_index,
                    "world_id": batch_receipts[-1]["materialized_world_ids"][0],
                    "host_boundary_observation": dict(binding),
                }
            )
            continue
        reducer = batch_result.get("compact_reducer")
        atlas = reducer.get("atlas_result") if isinstance(reducer, Mapping) else None
        if not isinstance(atlas, Mapping):
            journal_row["state"] = "FAILED"
            journal_row["failure"] = {
                "type": SCIENCE_AGGREGATION_FAILED,
                "message": "completed batch lacks a compact atlas result",
            }
            journal["batches"][batch_index] = journal_row
            journal["status"] = "FAILED"
            journal["last_event"] = f"batch-{batch_index:04d}-reduction-failed"
            _write_science_run_journal_v1(runtime_root, journal)
            _write_run_failure(
                runtime_root,
                request_sha256=request_digest,
                stage=f"batch-{batch_index:04d}-reduction",
                code=SCIENCE_AGGREGATION_FAILED,
                message="completed batch lacks a compact atlas result",
            )
            try:
                guard.release()
                guard_released = True
            except ArtifactLifecycleError:
                pass
            _record_science_lifecycle_failure_v1(
                root=runtime_root,
                plan=lifecycle_plan,
                source_identity=lifecycle_source,
                provider_id="eiel-science-request-v1-orchestrator",
                stage=f"batch-{batch_index:04d}-reduction",
                failure_type=SCIENCE_AGGREGATION_FAILED,
                message="completed batch lacks a compact atlas result",
            )
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_AGGREGATION_FAILED,
                f"science batch {batch_index} lacks a completed compact atlas result",
                details={"runtime_root": str(runtime_root), "execution_started": True},
            )
        if batch_envelope is None:
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_AGGREGATION_FAILED,
                "non-selected science batch lacks its compact atlas envelope",
                details={"runtime_root": str(runtime_root), "execution_started": True},
            )
        batch_results.append(
            batch_envelope
        )

    if (
        (
            selected_world_host_closeout
            and (
                batch_results
                or len(host_observation_results) != 1
                or host_observation_results[0].get("physical_batch_index") != 0
            )
        )
        or (
            not selected_world_host_closeout
            and (
                host_observation_results
                or len(batch_results) != batch_count
                or [row.get("physical_batch_index") for row in batch_results]
                != list(range(batch_count))
            )
        )
        or [row.get("state") for row in journal["batches"]]
        != ["COMPLETED"] * batch_count
    ):
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_AGGREGATION_FAILED,
            "science campaign closeout requires every authenticated batch observation",
            details={"runtime_root": str(runtime_root)},
        )
    aggregate_paths: dict[str, dict[str, Any]] = {}
    aggregate: dict[str, Any] | None = None
    host_observation_closeout: dict[str, Any] | None = None
    if selected_world_host_closeout:
        host_observation_closeout = {
            "mode": "AUTHENTICATED_SELECTED_WORLD_HOST_BOUNDARY_V1",
            "world_count": 1,
            "physical_batch_count": 1,
            "observations": host_observation_results,
            "runtime_observation_mode": "OFF",
            "host_adapter_observation_only": True,
            "atlas_generated": False,
            "causal_event_rows_available": False,
            "individual_lifetime_histories_available": False,
        }
    else:
        try:
            aggregate = aggregate_science_campaign_atlas_v1(
                request_sha256=request_identity,
                world_bundle=compilation.world_bundle,
                batch_results=batch_results,
            )
        except (ScienceCampaignAtlasError, TypeError, ValueError) as error:
            journal["status"] = "FAILED"
            journal["last_event"] = "campaign-aggregate-failed"
            _write_science_run_journal_v1(runtime_root, journal)
            _write_run_failure(
                runtime_root,
                request_sha256=request_digest,
                stage="campaign-aggregate",
                code=SCIENCE_AGGREGATION_FAILED,
                message=str(error),
            )
            try:
                guard.release()
                guard_released = True
            except ArtifactLifecycleError:
                pass
            _record_science_lifecycle_failure_v1(
                root=runtime_root,
                plan=lifecycle_plan,
                source_identity=lifecycle_source,
                provider_id="eiel-science-request-v1-orchestrator",
                stage="campaign-aggregate",
                failure_type=SCIENCE_AGGREGATION_FAILED,
                message=str(error),
            )
            raise ScienceRequestV1FrontDoorError(
                SCIENCE_AGGREGATION_FAILED,
                f"science campaign aggregation failed: {error}",
                details={"runtime_root": str(runtime_root), "execution_started": True},
            ) from error
        for name in ("result", "verification_receipt"):
            path = _write_or_verify_json_v1(
                path=runtime_root / f"campaign_atlas_{name}.json",
                value=aggregate[name],
                field=f"science campaign aggregate {name}",
            )
            aggregate_paths[name] = {
                "path": str(path),
                "file_sha256": _file_sha256(path),
            }
    core = {
        "schema_version": SCIENCE_REQUEST_V1_RUN_RECEIPT_SCHEMA_VERSION,
        "ok": True,
        "status": "COMPLETED",
        "result_class": "SCIENCE_CAMPAIGN_COMPLETED",
        "campaign_id": campaign_id,
        "request_sha256": request_digest,
        "runtime_root": str(runtime_root),
        "materialization_root": str(materialization_root),
        "materialization_plan_receipt_sha256": plan_result["receipt_sha256"],
        "authority_sha256": authority.authority_sha256,
        "completed_batches": batch_receipts,
        "aggregate": (
            None
            if aggregate is None
            else {
                **aggregate_paths,
                "result_sha256": aggregate["result"]["result_sha256"],
                "verification_receipt_sha256": aggregate["verification_receipt"][
                    "receipt_sha256"
                ],
            }
        ),
        **(
            {"host_observation_closeout": host_observation_closeout}
            if host_observation_closeout is not None
            else {}
        ),
        "batch_execution_authority": (
            "sealed_completed_child_reducer_recovery"
            if recovery_source_root is not None
            else "canonical_launchpad_per_batch"
        ),
        **(
            {
                "reducer_recovery": {
                    "mode": "REDUCER_ONLY",
                    "simulation_rerun": False,
                    "provider_rerun": False,
                    "source_root": str(recovery_source_root),
                }
            }
            if recovery_source_root is not None
            else {}
        ),
        "lifecycle": {
            "plan_sha256": lifecycle_plan["plan_sha256"],
            "run_passport_path": str(runtime_root / "run_passport.json"),
            "artifact_disposition_manifest_path": str(
                runtime_root / "artifact_disposition_manifest.json"
            ),
            "storage_disposition_receipt_path": str(
                runtime_root / "storage_disposition_receipt.json"
            ),
        },
        "reduction_completed": not selected_world_host_closeout,
        "aggregation_completed": not selected_world_host_closeout,
        **(
            {"observation_closeout_completed": True}
            if selected_world_host_closeout
            else {}
        ),
        "closeout_completed": True,
        "closeout_completion_condition": (
            "authoritative only when the terminal lifecycle artifacts validate"
        ),
        "execution_started": True,
        "prequal_started": False,
        "qualification_started": False,
        "qualification_authority_effect": False,
    }
    candidate_receipt = {**core, "receipt_sha256": canonical_sha256(core).to_dict()}
    closeout_path = runtime_root / "science_campaign_run_receipt.json"
    if closeout_path.exists() or closeout_path.is_symlink():
        receipt = _reconcile_precloseout_run_receipt_v1(
            path=closeout_path,
            candidate=candidate_receipt,
            journal=journal,
        )
    else:
        receipt = candidate_receipt
        closeout_path = _write_or_verify_json_v1(
            path=closeout_path,
            value=receipt,
            field="science campaign run receipt",
        )
    journal["status"] = "COMPLETED"
    journal["aggregate"] = (
        host_observation_closeout
        if aggregate is None
        else {
            **aggregate_paths,
            "result_sha256": aggregate["result"]["result_sha256"],
            "verification_receipt_sha256": aggregate["verification_receipt"][
                "receipt_sha256"
            ],
        }
    )
    journal["last_event"] = "completed_receipt_sealed"
    _write_science_run_journal_v1(runtime_root, journal)
    try:
        guard.enforce(stage="science_campaign_terminal_closeout")
        guard.release()
        guard_released = True
        lifecycle_closeout = _publish_science_lifecycle_closeout_v1(
            root=runtime_root,
            plan=lifecycle_plan,
            source_identity=lifecycle_source,
            provider_id="eiel-science-request-v1-orchestrator",
            terminal_status="completed",
            inventory_complete=True,
            register_archive=True,
        )
    except (
        ArtifactLifecycleError,
        OSError,
        ScienceRequestV1FrontDoorError,
        StorageBudgetExceeded,
    ) as error:
        if not guard_released:
            try:
                guard.release()
            except ArtifactLifecycleError:
                pass
        raise ScienceRequestV1FrontDoorError(
            SCIENCE_BATCH_EXECUTION_FAILED,
            f"science campaign lifecycle closeout failed: {error}",
            details={"runtime_root": str(runtime_root), "execution_started": True},
        ) from error
    return {
        **receipt,
        "receipt_path": str(closeout_path),
        "lifecycle_closeout": lifecycle_closeout,
    }


__all__ = (
    "AMBIGUOUS_SCIENCE_AUTHORITY",
    "DEFAULT_SCIENCE_AUTHORITY_INSTALL_PATH",
    "DEFAULT_SCIENCE_RUNTIME_ROOT",
    "INVALID_SCIENCE_AUTHORITY",
    "MAX_SCIENCE_EXECUTION_AUTHORITIES_V1",
    "MISSING_SCIENTIFIC_CAPABILITY",
    "SCIENCE_AUTHORITY_ENVIRONMENT_VARIABLE",
    "SCIENCE_BATCH_EXECUTION_FAILED",
    "SCIENCE_EXECUTION_AUTHORITY_SCHEMA_VERSION",
    "SCIENCE_EXECUTION_AUTHORITY_SET_SCHEMA_VERSION",
    "SCIENCE_MATERIALIZATION_FAILED",
    "SCIENCE_REQUEST_V1_FRONT_DOOR_SCHEMA_VERSION",
    "SCIENCE_REQUEST_V1_RUN_JOURNAL_SCHEMA_VERSION",
    "ScienceExecutionAuthorityV1",
    "ScienceRequestV1FrontDoorError",
    "UNRESOLVED_SCIENCE_AUTHORITY",
    "WAIT_RESOURCE",
    "WAIT_STORAGE",
    "activate_science_execution_authority_v1",
    "activate_science_execution_authority_set_v1",
    "plan_science_request_v1",
    "resolve_current_science_authority_v1",
    "run_science_request_v1",
    "validate_science_request_v1",
)
