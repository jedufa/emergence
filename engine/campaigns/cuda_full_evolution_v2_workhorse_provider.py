"""Single-operation canonical Launchpad provider for Workhorse V2.

One compiled campaign contains exactly one physical batch and that batch is
exactly one Workhorse operation.  An ``EXECUTION_ROW`` binds one exact
shape/backend/B/K contract.  A ``DEVICE_ENVELOPE_CALIBRATION`` binds one exact
shape/backend artifact and every compiled K, then performs the physical
calibration once under its own Attempt.  Matrix-wide tuning and selection are
deliberately offline consumers; they never share an outer Attempt.

Scheduler authority is not interpreted in this engine module.  The canonical
composition adapter is injected by Launchpad and must reopen and heartbeat the
actual Scheduler pointer before returning the non-authoritative parent context
passed to ``workhorse.canonical_rows``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final, Protocol
from uuid import uuid4

from . import cuda_full_evolution_v2_perf_surface as _perf_surface
from .acs_native_streamed_checkpoint_v1 import (
    DEFAULT_CHUNK_BYTES as ACS2_CAPTURE_CHUNK_BYTES_V1,
)

from .contracts import (
    CampaignPlanV1,
    ExecutionRouteContractV1,
    MetricFamilyV1,
    ParameterContractV1,
    RuntimeProfileV1,
    UnsupportedReasonV1,
)
from .providers import (
    ACS_NATIVE_RUNTIME_ABI_V1,
    ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1,
    ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1,
    ACS_NATIVE_RUNTIME_EXTENSION_KIND_V1,
    ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1,
    ACS_NATIVE_RUNTIME_MANIFEST_KIND_V1,
    CampaignProviderError,
    ExecutionProviderRegistryV1,
    ResidentBatchRequestV1,
    _acs_native_runtime_artifact_binding_v1,
    _acs_native_runtime_route_component_v1,
    default_execution_provider_registry,
)
from .qualification import EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION
from .registry import ProfileRegistryV1
from .cuda_full_evolution_v2_workhorse_resources import (
    WORKHORSE_SIGNED_PLANNING_ACCOUNTING_SCHEMA_V2,
    workhorse_checkpoint_storage_accounting_v2,
)
from .science_cuda_v2_native_world_v1 import (
    native_world_control_set_sha256_v1,
    project_science_cuda_v2_native_world_v1,
)


_benchmark = _perf_surface.benchmark_v2
_workhorse = _perf_surface.workhorse_v2


WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2 = (
    _workhorse.WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2
)
WORKHORSE_MINIMUM_HEADROOM_NUMERATOR_V2 = (
    _workhorse.WORKHORSE_MINIMUM_HEADROOM_NUMERATOR_V2
)
BACKEND_FUSED_CTA_TEMPORAL_V2 = _workhorse.BACKEND_FUSED_CTA_TEMPORAL_V2
BACKEND_PHASE_GRAPH_V2 = _workhorse.BACKEND_PHASE_GRAPH_V2
BACKEND_REFERENCE_PHASED_V2 = _workhorse.BACKEND_REFERENCE_PHASED_V2
FULL_EVOLUTION_CAPABILITIES_V2 = _workhorse.FULL_EVOLUTION_CAPABILITIES_V2
SUPPORTED_WORKHORSE_SHAPES_V2 = _workhorse.SUPPORTED_WORKHORSE_SHAPES_V2
WORKHORSE_SEMANTIC_CASE_OPERATION_V2 = _workhorse.WORKHORSE_SEMANTIC_CASE_OPERATION_V2
FullEvolutionCudaWorkhorseContractV2 = _workhorse.FullEvolutionCudaWorkhorseContractV2
WorkhorseCompileShapeV2 = _workhorse.WorkhorseCompileShapeV2
canonical_sha256 = _workhorse.canonical_sha256
cuda_device_identity_v2 = _benchmark.cuda_device_identity_v2
run_and_seal_workhorse_device_envelope_calibration_v2 = (
    _workhorse.run_and_seal_workhorse_device_envelope_calibration_v2
)
run_workhorse_canonical_admitted_row_v2 = (
    _workhorse.run_workhorse_canonical_admitted_row_v2
)
run_workhorse_indexed_continuation_v2 = (
    _workhorse.run_workhorse_indexed_continuation_v2
)
run_workhorse_semantic_case_v2 = _workhorse.run_workhorse_semantic_case_v2
reopen_workhorse_checkpoint_bundle_v2 = (
    _workhorse.reopen_workhorse_checkpoint_bundle_v2
)
inspect_workhorse_checkpoint_bundle_v2 = (
    _workhorse.inspect_workhorse_checkpoint_bundle_v2
)
write_workhorse_indexed_checkpoint_set_v1 = (
    _workhorse.write_workhorse_indexed_checkpoint_set_v1
)
inspect_workhorse_indexed_checkpoint_set_v1 = (
    _workhorse.inspect_workhorse_indexed_checkpoint_set_v1
)
compile_workhorse_absolute_dispatch_plan_v2 = (
    _workhorse.compile_workhorse_absolute_dispatch_plan_v2
)
workhorse_repack_binding_v2 = _workhorse.workhorse_repack_binding_v2
workhorse_world_identity_binding_v2 = (
    _workhorse.workhorse_world_identity_binding_v2
)
validate_workhorse_world_identity_binding_v2 = (
    _workhorse.validate_workhorse_world_identity_binding_v2
)
workhorse_indexed_continuation_contract_v2 = (
    _workhorse.workhorse_indexed_continuation_contract_v2
)
seal_workhorse_terminalized_device_envelope_calibration_receipt_v2 = (
    _workhorse.seal_workhorse_terminalized_device_envelope_calibration_receipt_v2
)
seal_workhorse_terminalized_semantic_case_receipt_v2 = (
    _workhorse.seal_workhorse_terminalized_semantic_case_receipt_v2
)
seal_workhorse_technical_row_receipt_v2 = (
    _workhorse.seal_workhorse_technical_row_receipt_v2
)
validate_external_checkpoint_source_binding_v2 = (
    _workhorse.validate_external_checkpoint_source_binding_v2
)
validate_workhorse_device_envelope_calibration_receipt_v2 = (
    _workhorse.validate_workhorse_device_envelope_calibration_receipt_v2
)
validate_workhorse_live_device_under_semantic_parent_v2 = (
    _workhorse.validate_workhorse_live_device_under_semantic_parent_v2
)
run_and_seal_workhorse_science_compatibility_demo_v1 = (
    _workhorse.run_and_seal_workhorse_science_compatibility_demo_v1
)
seal_workhorse_terminalized_science_demo_receipt_v1 = (
    _workhorse.seal_workhorse_terminalized_science_demo_receipt_v1
)
seal_workhorse_terminalized_indexed_continuation_receipt_v2 = (
    _workhorse.seal_workhorse_terminalized_indexed_continuation_receipt_v2
)
validate_workhorse_science_demo_contract_v1 = (
    _workhorse.validate_workhorse_science_demo_contract_v1
)
validate_workhorse_science_demo_result_v1 = (
    _workhorse.validate_workhorse_science_demo_result_v1
)
validate_workhorse_science_semantic_inheritance_binding_v1 = (
    _workhorse.validate_workhorse_science_semantic_inheritance_binding_v1
)
validate_workhorse_terminalized_science_demo_receipt_v1 = (
    _workhorse.validate_workhorse_terminalized_science_demo_receipt_v1
)
validate_workhorse_terminalized_indexed_continuation_receipt_v2 = (
    _workhorse.validate_workhorse_terminalized_indexed_continuation_receipt_v2
)
validate_workhorse_science_checkpoint_compatibility_binding_v2 = (
    _workhorse.validate_workhorse_science_checkpoint_compatibility_binding_v2
)
validate_workhorse_terminalized_device_envelope_calibration_receipt_v2 = (
    _workhorse.validate_workhorse_terminalized_device_envelope_calibration_receipt_v2
)
validate_workhorse_terminalized_semantic_case_receipt_v2 = (
    _workhorse.validate_workhorse_terminalized_semantic_case_receipt_v2
)
validate_workhorse_technical_row_receipt_v2 = (
    _workhorse.validate_workhorse_technical_row_receipt_v2
)
reopen_workhorse_semantic_case_fragment_v2 = (
    _workhorse.reopen_workhorse_semantic_case_fragment_v2
)
workhorse_artifact_binding_v2 = _workhorse.workhorse_artifact_binding_v2
workhorse_backend_temporal_k_candidates_v2 = (
    _workhorse.workhorse_backend_temporal_k_candidates_v2
)
validate_workhorse_fixed_tick_measurement_schedule_v3 = (
    _workhorse.validate_workhorse_fixed_tick_measurement_schedule_v3
)
validate_workhorse_indexed_continuation_contract_v2 = (
    _workhorse.validate_workhorse_indexed_continuation_contract_v2
)
validate_workhorse_indexed_checkpoint_publication_seals_v2 = (
    _workhorse.validate_workhorse_indexed_checkpoint_publication_seals_v2
)
WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1 = (
    _workhorse.WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1
)
seal_workhorse_selected_world_observation_v1 = (
    _workhorse.seal_workhorse_selected_world_observation_v1
)
validate_workhorse_selected_world_observation_v1 = (
    _workhorse.validate_workhorse_selected_world_observation_v1
)
validate_workhorse_indexed_continuation_result_v2 = (
    _workhorse.validate_workhorse_indexed_continuation_result_v2
)
workhorse_semantic_case_law_v2 = _workhorse.workhorse_semantic_case_law_v2
workhorse_semantic_case_operation_contract_v2 = (
    _workhorse.workhorse_semantic_case_operation_contract_v2
)
validate_authoritative_micro_denominator_v2 = (
    _workhorse.validate_authoritative_micro_denominator_v2
)
validate_workhorse_semantic_gate_receipt_v2 = (
    _workhorse.validate_workhorse_semantic_gate_receipt_v2
)
workhorse_fixed_tick_measurement_schedule_v3 = (
    _workhorse.workhorse_fixed_tick_measurement_schedule_v3
)
workhorse_row_checkpoint_binding_v2 = (
    _workhorse.workhorse_row_checkpoint_binding_v2
)
workhorse_science_checkpoint_compatibility_binding_v2 = (
    _workhorse.workhorse_science_checkpoint_compatibility_binding_v2
)
workhorse_science_demo_contract_v1 = _workhorse.workhorse_science_demo_contract_v1
workhorse_science_semantic_inheritance_binding_v1 = (
    _workhorse.workhorse_science_semantic_inheritance_binding_v1
)
write_workhorse_semantic_case_aggregate_gate_receipt_v2 = (
    _workhorse.write_workhorse_semantic_case_aggregate_gate_receipt_v2
)


WORLD_BASE: Final[str] = "tensor_world_cuda_full_evolution_workhorse_v2"
PROVIDER_ID: Final[str] = "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
OPERATION_EXTENSION_ID: Final[str] = (
    "cuda_full_evolution_workhorse_v2_single_operation_v1"
)
OPERATION_SCHEMA_V2: Final[str] = "eiel_workhorse_launchpad_operation_v3"
OPERATION_SCHEMA_V3: Final[str] = "eiel_workhorse_launchpad_operation_v4"
EXACT_FOUNDER_BOOTSTRAP_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_exact_founder_bootstrap_v1"
)
EXACT_FOUNDER_MATERIALIZATION_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_exact_founder_materialization_v1"
)
EXACT_FOUNDER_WORLD_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_exact_founder_world_v1"
)
EXACT_FOUNDER_INITIALIZATION_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_exact_founder_initialization_v1"
)
# These values are the regular-row initializer arguments already owned by
# ``run_workhorse_canonical_admitted_row_v2``.  Exact-founder plans bind and
# pass the complete vector explicitly instead of relying on that fallback.
EXACT_FOUNDER_WORLD_UID_BASE_V1: Final[int] = 210_000
EXACT_FOUNDER_RNG_KEY_V1: Final[int] = 83
EXACT_FOUNDER_RESOURCE_MULTIPLIER_V1: Final[float] = 1.0
EXACT_FOUNDER_MUTATION_RATE_V1: Final[float] = 0.125
EXECUTION_ROW_OPERATION_V2: Final[str] = "EXECUTION_ROW"
DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2: Final[str] = "DEVICE_ENVELOPE_CALIBRATION"
SEMANTIC_EVIDENCE_CASE_OPERATION_V2: Final[str] = WORKHORSE_SEMANTIC_CASE_OPERATION_V2
SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1: Final[str] = (
    _workhorse.WORKHORSE_SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1
)
INDEXED_ACS_CONTINUATION_OPERATION_V2: Final[str] = (
    _workhorse.WORKHORSE_INDEXED_CONTINUATION_OPERATION_V2
)
INDEXED_ACS_CONTINUATION_AGGREGATE_WTPS_METRIC_V2: Final[str] = (
    _workhorse.WORKHORSE_INDEXED_CONTINUATION_AGGREGATE_WTPS_METRIC_V2
)
SUPPORTED_OPERATION_V2: Final[str] = EXECUTION_ROW_OPERATION_V2
SUPPORTED_OPERATIONS_V2: Final[frozenset[str]] = frozenset(
    {
        EXECUTION_ROW_OPERATION_V2,
        DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2,
        SEMANTIC_EVIDENCE_CASE_OPERATION_V2,
        SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
        INDEXED_ACS_CONTINUATION_OPERATION_V2,
    }
)
ATTEMPT_AUTHORITY_NAME: Final[str] = "control/crca/attempt-authority.json"
RUNTIME_HEADROOM_REJECTION_CODE_V2: Final[str] = "WORKHORSE_RUNTIME_HEADROOM_REJECTED"


class WorkhorseLaunchpadOperationAdapterV2(Protocol):
    """Composition-only authority seam injected into the engine provider."""

    def bind_workhorse_parent_context_v2(
        self,
        *,
        attempt_authority_path: Path,
        target: Path,
        campaign_plan: Mapping[str, Any],
        bound_execution_request: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def validate_workhorse_external_checkpoint_source_v2(
        self,
        binding: Mapping[str, Any],
        *,
        expected_identity_sha256: str,
    ) -> Mapping[str, Any]: ...

    def materialize_workhorse_semantic_checkpoint_occurrence_v2(
        self,
        *,
        canonical_parent_authority_binding: Mapping[str, Any],
        case_fragment_path: Path,
        science_checkpoint_compatibility_path: Path,
        occurrence_root: Path,
    ) -> Mapping[str, Any]: ...

    def materialize_workhorse_science_checkpoint_occurrence_v1(
        self,
        *,
        canonical_parent_authority_binding: Mapping[str, Any],
        checkpoint_manifest_path: Path,
        absolute_bundle_root: Path,
        occurrence_root: Path,
        parent_checkpoint_digest: Mapping[str, str] | None,
    ) -> Mapping[str, Any]: ...

    def materialize_workhorse_row_checkpoint_occurrence_v2(
        self,
        *,
        canonical_parent_authority_binding: Mapping[str, Any],
        checkpoint_manifest_path: Path,
        absolute_bundle_root: Path,
        occurrence_root: Path,
        parent_checkpoint_digest: Mapping[str, str] | None = None,
        expected_operation: str = EXECUTION_ROW_OPERATION_V2,
    ) -> Mapping[str, Any]: ...

    def restore_workhorse_science_checkpoint_occurrence_v1(
        self,
        *,
        canonical_parent_authority_binding: Mapping[str, Any],
        checkpoint_manifest_path: Path,
        projection: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def workhorse_exact_founder_bootstrap_contract_v1() -> dict[str, Any]:
    """Declare the only regular-row shape that may publish a branch parent.

    This is an internal provider operation contract, not a new science-policy
    surface.  The values describe the existing regular-row initializer and
    compiled kernel defaults.  Callers cannot supply alternate identities or
    controls through the exact-founder bootstrap route.
    """

    body = {
        "schema_version": EXACT_FOUNDER_BOOTSTRAP_SCHEMA_V1,
        "mode": "REGULAR_ROW_ACS2_TWO_OCCURRENCE_V1",
        "time_authority": "absolute_tick_v1",
        "runtime_observation_mode": "OFF",
        "claim_class": "TECHNICAL_ONLY",
        "science_authority_granted": False,
        "qualification_authority_granted": False,
        "world_identity_authority": "WORKHORSE_REGULAR_ROW_INITIALIZER_V1",
        "world_uid_base": EXACT_FOUNDER_WORLD_UID_BASE_V1,
        "rng_key": EXACT_FOUNDER_RNG_KEY_V1,
        "founder_control_authority": "WORKHORSE_NATIVE_COMPILED_DEFAULTS_V1",
        "founder_controls": {
            "environment.resource.multiplier": (
                EXACT_FOUNDER_RESOURCE_MULTIPLIER_V1
            ),
            "mutation.rate": EXACT_FOUNDER_MUTATION_RATE_V1,
        },
        "unsupported_control_policy": "FAIL_CLOSED",
    }
    return {**body, "contract_sha256": canonical_sha256(body)}


def _payload(plan: CampaignPlanV1 | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, CampaignPlanV1):
        return plan.payload()
    if not isinstance(plan, Mapping):
        raise CampaignProviderError("Workhorse plan must be a mapping")
    return dict(plan)


def _sha(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CampaignProviderError(f"{label} must be a lowercase SHA-256")
    return value


def _sealed_binding(value: object, *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != {
        "path",
        "file_sha256",
        "content_sha256",
    }:
        raise CampaignProviderError(f"{label} must contain exact sealed bindings")
    path = value["path"]
    if type(path) is not str or not path:
        raise CampaignProviderError(f"{label} path is absent")
    return {
        "path": path,
        "file_sha256": _sha(value["file_sha256"], label=f"{label} file digest"),
        "content_sha256": _sha(
            value["content_sha256"], label=f"{label} content digest"
        ),
    }


def _device_identity(value: object) -> dict[str, Any]:
    fields = {
        "index",
        "name",
        "uuid",
        "pci_domain_id",
        "pci_bus_id",
        "pci_device_id",
        "total_memory_bytes",
        "multiprocessor_count",
        "sm",
        "cuda_runtime",
    }
    if type(value) is not dict or set(value) != fields:
        raise CampaignProviderError("calibration device identity is inexact")
    device = dict(value)
    for field in ("index", "pci_domain_id", "pci_bus_id", "pci_device_id"):
        if type(device[field]) is not int or device[field] < 0:
            raise CampaignProviderError(f"calibration device {field} is invalid")
    for field in ("total_memory_bytes", "multiprocessor_count"):
        if type(device[field]) is not int or device[field] <= 0:
            raise CampaignProviderError(f"calibration device {field} is invalid")
    for field in ("name", "uuid", "sm", "cuda_runtime"):
        if (
            type(device[field]) is not str
            or not device[field]
            or device[field] != device[field].strip()
        ):
            raise CampaignProviderError(f"calibration device {field} is absent")
    return device


def _reserve_policy(value: object) -> dict[str, Any]:
    expected = {
        "mode",
        "reserve_bytes_requested",
        "minimum_fraction_numerator",
        "minimum_fraction_denominator",
        "transient_free_bytes_is_plan_authority",
    }
    if type(value) is not dict or set(value) != expected:
        raise CampaignProviderError("calibration reserve policy is inexact")
    policy = dict(value)
    if (
        policy["mode"] != "STABLE_TOTAL_MINUS_EFFECTIVE_RESERVE_V2"
        or type(policy["reserve_bytes_requested"]) is not int
        or policy["reserve_bytes_requested"] < 0
        or policy["minimum_fraction_numerator"]
        != WORKHORSE_MINIMUM_HEADROOM_NUMERATOR_V2
        or policy["minimum_fraction_denominator"]
        != WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2
        or policy["transient_free_bytes_is_plan_authority"] is not False
    ):
        raise CampaignProviderError("calibration reserve policy changed")
    return policy


def _absolute_tick_span(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "time_authority",
        "start_next_tick",
        "target_next_tick",
        "total_ticks",
    }:
        raise CampaignProviderError("calibration absolute-tick span is inexact")
    span = dict(value)
    start = span["start_next_tick"]
    target = span["target_next_tick"]
    total = span["total_ticks"]
    if (
        span["time_authority"] != "absolute_tick_v1"
        or type(start) is not int
        or start < 0
        or type(target) is not int
        or type(total) is not int
        or total <= 0
        or target != start + total
    ):
        raise CampaignProviderError("calibration absolute-tick span changed")
    return span


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publication_digest_hex(value: object, label: str) -> str:
    """Normalize an occurrence digest descriptor for compact publication."""

    if isinstance(value, Mapping):
        if value.get("algorithm") != "sha256":
            raise CampaignProviderError(f"{label} algorithm is not SHA-256")
        value = value.get("hex")
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CampaignProviderError(f"{label} is not a SHA-256")
    return value


def _quarantine_workhorse_publication_v1(path: str | Path) -> Path | None:
    """Move one exact uncommitted provider publication out of canonical view."""

    publication = Path(path).expanduser()
    if publication.is_symlink():
        raise CampaignProviderError(
            "Workhorse failed publication path is a symbolic link"
        )
    if not publication.exists():
        return None
    publication = publication.resolve()
    failed = publication.with_name(
        f"{publication.name}.publication.failed.{uuid4().hex}"
    )
    while failed.exists() or failed.is_symlink():
        failed = publication.with_name(
            f"{publication.name}.publication.failed.{uuid4().hex}"
        )
    os.replace(publication, failed)
    descriptor = os.open(
        publication.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return failed


def _quarantine_workhorse_commit_paths_v1(
    paths: Sequence[str | Path], failure: BaseException
) -> None:
    for path in paths:
        try:
            failed = _quarantine_workhorse_publication_v1(path)
        except BaseException as cleanup_error:
            failure.add_note(
                "Workhorse checkpoint quarantine also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        else:
            if failed is not None:
                failure.add_note(f"Workhorse checkpoint quarantined at {failed}")


def _commit_workhorse_science_acs_checkpoint_v1(
    *,
    context: Any,
    coordinator: Any,
    token: Any,
    checkpoint_manifest_path: Path,
    occurrence_projection: Mapping[str, Any],
    event_sink_root: Path,
    checkpoint_root: Path,
    boundary_root: Path,
) -> None:
    """Inventory one science checkpoint and ACK only after exact CRCA commit."""

    quarantine_paths: list[Path] = []
    try:
        if not checkpoint_manifest_path.is_file():
            raise CampaignProviderError(
                "Workhorse science ACS checkpoint manifest is absent"
            )
        for field in ("occurrence_root", "absolute_tick_bundle_path"):
            raw_path = occurrence_projection.get(field)
            if type(raw_path) is not str:
                raise CampaignProviderError(
                    "Workhorse science ACS occurrence projection is incomplete"
                )
            committed_path = Path(raw_path).resolve()
            try:
                committed_path.relative_to(checkpoint_root.resolve())
            except ValueError as exc:
                raise CampaignProviderError(
                    "Workhorse science ACS occurrence escaped its root"
                ) from exc
            quarantine_paths.append(committed_path)
        for root, family in (
            (
                event_sink_root,
                "workhorse_science_demo_event_ack_v1",
            ),
            (
                checkpoint_root,
                "workhorse_science_demo_checkpoint_v1",
            ),
            (
                boundary_root,
                "workhorse_science_demo_boundary_record_v1",
            ),
        ):
            for artifact_path in sorted(root.rglob("*")):
                if artifact_path.is_symlink():
                    raise CampaignProviderError(
                        "science checkpoint commit contains a symbolic link"
                    )
                if artifact_path.is_file():
                    context.record_artifact(
                        artifact_path,
                        artifact_family=family,
                        retention_class="R3",
                    )
        coordinator.acknowledge_checkpoint(token)
    except BaseException as exc:
        _quarantine_workhorse_commit_paths_v1(
            quarantine_paths,
            exc,
        )
        raise


def _commit_workhorse_row_acs_checkpoint_v2(
    *,
    context: Any,
    coordinator: Any,
    token: Any,
    operation_adapter: Any,
    canonical_parent_authority_binding: Mapping[str, Any],
    contract: FullEvolutionCudaWorkhorseContractV2,
    expected_checkpoint_tick: int,
    parent_checkpoint_digest: Mapping[str, str] | None = None,
    expected_operation: str = EXECUTION_ROW_OPERATION_V2,
) -> dict[str, Any]:
    """Commit one row checkpoint occurrence and lifecycle inventory before ACK."""

    receipt = dict(coordinator.settle_checkpoint(token))
    reopened, state_path = inspect_workhorse_checkpoint_bundle_v2(
        receipt["manifest_path"]
    )
    if (
        type(expected_checkpoint_tick) is not int
        or expected_checkpoint_tick <= contract.start_next_tick
        or expected_checkpoint_tick
        > contract.start_next_tick + contract.total_ticks
        or receipt["checkpoint_at_tick"] != expected_checkpoint_tick
        or reopened["checkpoint_at_tick"] != expected_checkpoint_tick
        or reopened["canonical_state"]["sha256"]
        != receipt["canonical_state_sha256"]
        or state_path.stat().st_size != receipt["canonical_state_bytes"]
    ):
        raise CampaignProviderError(
            "Workhorse row ACS fresh reopen changed canonical state"
        )
    absolute_bundle_root = (
        Path(context.batch_root)
        / "workhorse-absolute-checkpoints"
        / f"checkpoint-tick-{expected_checkpoint_tick:012d}"
    )
    occurrence_root = (
        Path(context.batch_root)
        / "workhorse-checkpoint-occurrences"
        / f"checkpoint-tick-{expected_checkpoint_tick:012d}"
    )
    indexed_root = (
        Path(context.batch_root)
        / "workhorse-indexed-checkpoints"
        / f"checkpoint-tick-{expected_checkpoint_tick:012d}"
    )
    commit_paths = [indexed_root, occurrence_root, absolute_bundle_root]
    try:
        indexed = write_workhorse_indexed_checkpoint_set_v1(
            receipt["manifest_path"],
            indexed_root,
        )
        indexed_identity = indexed.get("world_state_set_identity")
        indexed_storage = indexed.get("storage_model")
        if (
            not isinstance(indexed_identity, Mapping)
            or indexed_identity.get("checkpoint_at_tick")
            != expected_checkpoint_tick
            or indexed_identity.get("shape_id") != contract.shape_id
            or indexed_identity.get("world_count") != contract.B
            or indexed_identity.get("semantic_transition_sha256")
            != contract.semantic_transition_sha256
            or indexed_storage
            != {
                "canonical_state_payload_copies": 1,
                "indexed_world_payload_copies": 0,
                "world_rows_are_range_references": True,
                "source_payload_must_remain_live": True,
            }
        ):
            raise CampaignProviderError(
                "Workhorse row ACS indexed checkpoint changed state identity"
            )
        occurrence_kwargs = {
            "canonical_parent_authority_binding": canonical_parent_authority_binding,
            "checkpoint_manifest_path": Path(receipt["manifest_path"]),
            "absolute_bundle_root": absolute_bundle_root,
            "occurrence_root": occurrence_root,
            "parent_checkpoint_digest": parent_checkpoint_digest,
        }
        if expected_operation == INDEXED_ACS_CONTINUATION_OPERATION_V2:
            occurrence_kwargs["expected_operation"] = expected_operation
        materialized = operation_adapter.materialize_workhorse_row_checkpoint_occurrence_v2(
            **occurrence_kwargs
        )
        if type(materialized) is not dict or set(materialized) != {
            "canonical_state",
            "canonical_state_nbytes",
            "canonical_state_sha256",
            "projection",
        }:
            raise CampaignProviderError(
                "Workhorse row ACS occurrence result is inexact"
            )
        projection = materialized["projection"]
        if (
            materialized["canonical_state"] is not None
            or materialized["canonical_state_nbytes"]
            != receipt["canonical_state_bytes"]
            or materialized["canonical_state_sha256"]
            != receipt["canonical_state_sha256"]
            or not isinstance(projection, Mapping)
            or projection.get("schema_version")
            != (
                "eiel_workhorse_indexed_continuation_checkpoint_occurrence_projection_v2"
                if expected_operation == INDEXED_ACS_CONTINUATION_OPERATION_V2
                else "eiel_workhorse_row_checkpoint_occurrence_projection_v2"
            )
            or projection.get("checkpoint_at_tick") != expected_checkpoint_tick
            or projection.get("canonical_state_sha256")
            != receipt["canonical_state_sha256"]
            or projection.get("canonical_visible") is not True
            or projection.get("restore_admitted_exact") is not False
            or projection.get("parent_checkpoint_digest")
            != (
                None
                if parent_checkpoint_digest is None
                else dict(parent_checkpoint_digest)
            )
        ):
            raise CampaignProviderError(
                "Workhorse row ACS occurrence changed checkpoint identity"
            )
        lifecycle = getattr(context, "lifecycle_contract", None)
        checkpoint_role: Mapping[str, Any] | None = None
        if isinstance(lifecycle, Mapping):
            checkpoints = lifecycle.get("checkpoints")
            roles = (
                checkpoints.get("roles")
                if isinstance(checkpoints, Mapping)
                else None
            )
            if not isinstance(roles, list):
                raise CampaignProviderError(
                    "Workhorse ACS lifecycle lacks absolute checkpoint roles"
                )
            matches = [
                row
                for row in roles
                if isinstance(row, Mapping)
                and row.get("checkpoint_tick", row.get("tick"))
                == expected_checkpoint_tick
            ]
            if len(matches) != 1:
                raise CampaignProviderError(
                    "Workhorse ACS checkpoint lacks one exact lifecycle role"
                )
            checkpoint_role = matches[0]
        payload_retention = (
            str(checkpoint_role.get("retention_class"))
            if checkpoint_role is not None
            else "R2"
        )
        if payload_retention not in {"R0", "R1", "R2", "R3", "R4", "R5"}:
            raise CampaignProviderError(
                "Workhorse ACS checkpoint lifecycle retention is unsupported"
            )
        checkpoint_metadata = {
            "checkpoint_role": (
                checkpoint_role.get("role") if checkpoint_role is not None else None
            ),
            "checkpoint_identity": (
                checkpoint_role.get("identity")
                if checkpoint_role is not None
                else None
            ),
            "checkpoint_at_tick": expected_checkpoint_tick,
        }
        for root, family, retention_class in (
            (
                Path(receipt["path"]),
                "workhorse_v2_acs_checkpoint",
                payload_retention,
            ),
            (
                indexed_root,
                "workhorse_v2_acs_indexed_checkpoint",
                payload_retention,
            ),
            (
                absolute_bundle_root,
                "workhorse_v2_acs_absolute_tick_bundle",
                payload_retention,
            ),
            (
                occurrence_root,
                "workhorse_v2_acs_checkpoint_occurrence",
                "R3",
            ),
        ):
            for artifact_path in sorted(root.rglob("*")):
                if artifact_path.is_symlink():
                    raise CampaignProviderError(
                        "Workhorse row ACS commit contains a symbolic link"
                    )
                if artifact_path.is_file():
                    context.record_artifact(
                        artifact_path,
                        artifact_family=family,
                        retention_class=retention_class,
                        **checkpoint_metadata,
                    )
        receipt["checkpoint_occurrence_projection"] = dict(projection)
        receipt["indexed_checkpoint"] = {
            "manifest_path": indexed["manifest_path"],
            "manifest_file_sha256": indexed["manifest_file_sha256"],
            "manifest_body_sha256": indexed["manifest_body_sha256"],
            "world_state_set_identity": dict(indexed_identity),
            "storage_model": dict(indexed_storage),
        }
        coordinator.acknowledge_checkpoint(token)
        return receipt
    except BaseException as exc:
        _quarantine_workhorse_commit_paths_v1(commit_paths, exc)
        raise


def _semantic_provisional_result(
    value: object,
    *,
    operation: Mapping[str, Any],
    canonical_parent: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fields = {
        "schema_version",
        "claim_class",
        "qualification_authority_granted",
        "science_authority_granted",
        "case_id",
        "expected_attempt_terminal_status",
        "attempt_terminal_contract",
        "provisional_result_class",
        "provider_terminalization_action",
        "case_fragment_path",
        "case_fragment_file_sha256",
        "case_fragment_body_sha256",
        "source_checkpoint_identity_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise CampaignProviderError("semantic provisional result fields are inexact")
    result = dict(value)
    semantic_case = operation["semantic_case_contract"]
    terminal = semantic_case["attempt_terminal_contract"]
    case_id = semantic_case["case_law"]["case_id"]
    expected_status = terminal["attempt_terminal_status"]
    expected_class = (
        "SUCCESS_PENDING_COMPLETED_TERMINAL"
        if expected_status == "COMPLETED"
        else "EXPECTED_NEGATIVE_PENDING_FAILED_TERMINAL"
    )
    expected_action = (
        "RETURN_FOR_COMPLETED_TERMINALIZATION"
        if expected_status == "COMPLETED"
        else "RETURN_FOR_FAILED_ATTEMPT_TERMINALIZATION"
    )
    fragment_path = Path(str(result["case_fragment_path"]))
    try:
        fragment, _ = reopen_workhorse_semantic_case_fragment_v2(fragment_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CampaignProviderError(
            f"semantic raw fragment cannot reopen exactly: {exc}"
        ) from exc
    if (
        result["schema_version"] != "eiel_workhorse_semantic_case_provisional_result_v2"
        or result["claim_class"] != "TECHNICAL_ONLY_PROVISIONAL"
        or result["qualification_authority_granted"] is not False
        or result["science_authority_granted"] is not False
        or result["case_id"] != case_id
        or result["expected_attempt_terminal_status"] != expected_status
        or result["attempt_terminal_contract"] != terminal
        or result["provisional_result_class"] != expected_class
        or result["provider_terminalization_action"] != expected_action
        or result["source_checkpoint_identity_sha256"]
        != operation["source_checkpoint_identity_sha256"]
        or fragment_path.is_symlink()
        or not fragment_path.is_file()
        or result["case_fragment_file_sha256"] != _file_sha256(fragment_path)
        or result["case_fragment_body_sha256"] != fragment.get("fragment_body_sha256")
        or fragment.get("case_id") != case_id
        or fragment.get("semantic_temporal_k") != semantic_case["semantic_temporal_k"]
        or fragment.get("shape_id") != operation["contract"]["shape_id"]
        or fragment.get("artifact_sha256") != operation["artifact"]["sha256"]
        or fragment.get("canonical_parent_binding") != canonical_parent
        or fragment.get("external_source_checkpoint_binding")
        != operation["external_source_checkpoint_binding"]
    ):
        raise CampaignProviderError(
            "semantic provisional result differs from its exact operation"
        )
    return result, fragment


def _finalize_science_demo_provider_result_v1(
    *,
    provider_result: Mapping[str, Any],
    operation: Mapping[str, Any],
    manifest_path: Path,
    canonical_terminal_binding: Mapping[str, Any],
    operation_adapter: WorkhorseLaunchpadOperationAdapterV2,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "operation",
        "shape_id",
        "backend_id",
        "B",
        "temporal_k",
        "artifact_sha256",
        "sealed_artifact_identity_sha256",
        "science_demo_result_path",
        "science_demo_result_file_sha256",
        "science_demo_result_content_sha256",
        "absolute_frontiers",
        "boundary_next_ticks",
        "checkpoint_ticks",
        "continuation_restore_ticks",
        "terminal_restore_validation_tick",
        "result_class",
    }
    if type(provider_result) is not dict or set(provider_result) not in {
        frozenset(expected_fields),
        frozenset({*expected_fields, "acs_checkpoint_staging"}),
    }:
        raise CampaignProviderError(
            "Workhorse science demo provider result fields are inexact"
        )
    contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(operation["contract"])
    try:
        result = validate_workhorse_science_demo_result_v1(
            provider_result["science_demo_result_path"]
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CampaignProviderError(
            f"Workhorse science demo result cannot reopen: {exc}"
        ) from exc
    acs_staging = provider_result.get("acs_checkpoint_staging")
    result_has_acs = result.get("acs_checkpoint_transport") is not None
    if (acs_staging is not None) is not result_has_acs:
        raise CampaignProviderError(
            "Workhorse science demo ACS transport/result binding differs"
        )
    if acs_staging is not None and (
        not isinstance(acs_staging, Mapping)
        or acs_staging.get("closed") is not True
        or acs_staging.get("writer_alive") is not False
        or acs_staging.get("failure_latched") is not False
        or len(acs_staging.get("checkpoint_receipts") or ()) != 3
        or [
            row.get("checkpoint_at_tick")
            for row in acs_staging.get("checkpoint_receipts", ())
            if isinstance(row, Mapping)
        ]
        != [16, 32, 40]
    ):
        raise CampaignProviderError(
            "Workhorse science demo ACS staging receipt is invalid"
        )
    if (
        provider_result["schema_version"]
        != "eiel_workhorse_science_demo_provider_result_v1"
        or provider_result["operation"] != operation["operation"]
        or provider_result["shape_id"] != contract.shape_id
        or provider_result["backend_id"] != contract.backend_id
        or provider_result["B"] != contract.B
        or provider_result["temporal_k"] != contract.temporal_k
        or provider_result["artifact_sha256"] != operation["artifact"]["sha256"]
        or provider_result["sealed_artifact_identity_sha256"]
        != operation["artifact"]["sealed_artifact_identity_sha256"]
        or provider_result["science_demo_result_path"] != result["result_path"]
        or provider_result["science_demo_result_file_sha256"]
        != result["result_file_sha256"]
        or provider_result["science_demo_result_content_sha256"]
        != result["result_content_sha256"]
        or provider_result["absolute_frontiers"]
        != {
            "start_next_tick": 0,
            "target_next_tick": 40,
            "committed_next_tick": 40,
        }
        or provider_result["boundary_next_ticks"] != [0, 8, 16, 24, 32, 40]
        or provider_result["checkpoint_ticks"] != [16, 32, 40]
        or provider_result["continuation_restore_ticks"] != [16, 32]
        or provider_result["terminal_restore_validation_tick"] != 40
        or provider_result["result_class"] != "TECHNICAL_COMPATIBILITY_DEMO_PASSED"
    ):
        raise CampaignProviderError(
            "Workhorse science demo provider result is not exact"
        )
    for checkpoint, projection in zip(
        result["checkpoint_bundles"],
        result["checkpoint_occurrences"],
        strict=True,
    ):
        try:
            restored = (
                operation_adapter.restore_workhorse_science_checkpoint_occurrence_v1(
                    canonical_parent_authority_binding=result[
                        "canonical_parent_authority_binding"
                    ],
                    checkpoint_manifest_path=Path(checkpoint["manifest_path"]),
                    projection=projection,
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise CampaignProviderError(
                f"Workhorse science checkpoint occurrence cannot reopen: {exc}"
            ) from exc
        if restored.get("canonical_state_sha256") != checkpoint["state_file_sha256"]:
            raise CampaignProviderError(
                "Workhorse science occurrence restored a different checkpoint state"
            )
    output = manifest_path.parent / "workhorse-science-demo-receipt.json"
    receipt = (
        validate_workhorse_terminalized_science_demo_receipt_v1(output)
        if output.exists() and not output.is_symlink()
        else seal_workhorse_terminalized_science_demo_receipt_v1(
            output,
            science_demo_result_path=result["result_path"],
            canonical_terminal_binding=canonical_terminal_binding,
        )
    )
    return {
        "schema_version": "eiel_workhorse_terminalized_operation_binding_v2",
        "operation": operation["operation"],
        "result_class": "SUCCESS",
        "demo_result_class": "TECHNICAL_COMPATIBILITY_DEMO_PASSED",
        "final_receipt_path": receipt["receipt_path"],
        "final_receipt_file_sha256": receipt["receipt_file_sha256"],
        "final_receipt_content_sha256": receipt["receipt_content_sha256"],
        "science_demo_receipt_path": receipt["receipt_path"],
        "science_demo_receipt_file_sha256": receipt["receipt_file_sha256"],
        "science_demo_receipt_content_sha256": receipt["receipt_content_sha256"],
        "attempt_id": canonical_terminal_binding.get("attempt_id"),
        "lifecycle_binding": {
            "artifact_family": "science.workhorse_compatibility_demo_v1",
            "retention_class": "R3",
            "owner_writer": ("cuda_full_evolution_workhorse_v2_single_operation_owner"),
            "release_predicate": (
                "technical science-compatibility evidence retained; "
                "no science authority"
            ),
        },
    }


def _indexed_continuation_publication_seals_v2(
    *,
    provider_result: Mapping[str, Any],
    contract: FullEvolutionCudaWorkhorseContractV2,
    checkpoint_ticks: Sequence[int],
) -> list[dict[str, Any]]:
    """Reopen every published checkpoint before sealing payload-free evidence."""

    receipts = provider_result.get("indexed_checkpoint_receipts")
    telemetry = provider_result.get("acs_checkpoint_staging")
    if (
        type(receipts) is not list
        or len(receipts) != len(checkpoint_ticks)
        or type(telemetry) is not dict
        or telemetry.get("closed") is not True
        or telemetry.get("writer_alive") is not False
        or telemetry.get("failure_latched") is not False
    ):
        raise CampaignProviderError(
            "indexed continuation ACS2 publication coverage is incomplete"
        )
    telemetry_receipts = telemetry.get("checkpoint_receipts")
    if type(telemetry_receipts) is not list or len(telemetry_receipts) != len(receipts):
        raise CampaignProviderError(
            "indexed continuation ACS2 telemetry differs from committed checkpoints"
        )
    publications: list[dict[str, Any]] = []
    for receipt, telemetry_receipt, tick in zip(
        receipts, telemetry_receipts, checkpoint_ticks, strict=True
    ):
        if type(receipt) is not dict or type(telemetry_receipt) is not dict:
            raise CampaignProviderError(
                "indexed continuation checkpoint receipt is malformed"
            )
        detached_receipt = {
            key: item
            for key, item in receipt.items()
            if key not in {"checkpoint_occurrence_projection", "indexed_checkpoint"}
        }
        if telemetry_receipt != detached_receipt:
            raise CampaignProviderError(
                "indexed continuation ACS2 telemetry changed its detached receipt"
            )
        reopened, state_path = inspect_workhorse_checkpoint_bundle_v2(
            receipt.get("manifest_path")
        )
        indexed_binding = receipt.get("indexed_checkpoint")
        projection = receipt.get("checkpoint_occurrence_projection")
        if type(indexed_binding) is not dict or type(projection) is not dict:
            raise CampaignProviderError(
                "indexed continuation checkpoint lacks index or occurrence"
            )
        indexed, indexed_state_path = inspect_workhorse_indexed_checkpoint_set_v1(
            indexed_binding.get("manifest_path")
        )
        if (
            receipt.get("checkpoint_at_tick") != tick
            or reopened.get("checkpoint_at_tick") != tick
            or reopened.get("contract") != contract.to_dict()
            or state_path != indexed_state_path
            or receipt.get("canonical_state_bytes") != state_path.stat().st_size
            or receipt.get("canonical_state_sha256")
            != reopened["canonical_state"]["sha256"]
            or receipt.get("bundle_body_sha256") != reopened["bundle_body_sha256"]
            or indexed_binding.get("manifest_file_sha256")
            != _file_sha256(indexed_binding["manifest_path"])
            or indexed_binding.get("manifest_body_sha256")
            != indexed["manifest_body_sha256"]
            or indexed_binding.get("world_state_set_identity")
            != indexed["world_state_set_identity"]
            or projection.get("checkpoint_at_tick") != tick
            or projection.get("canonical_state_sha256")
            != receipt.get("canonical_state_sha256")
        ):
            raise CampaignProviderError(
                "indexed continuation checkpoint publication changed identity"
            )
        body = {
            "schema_version": (
                _workhorse.WORKHORSE_INDEXED_CHECKPOINT_PUBLICATION_SCHEMA_V2
            ),
            "checkpoint_at_tick": tick,
            "canonical_state_bytes": receipt["canonical_state_bytes"],
            "canonical_state_sha256": receipt["canonical_state_sha256"],
            "checkpoint_bundle_body_sha256": receipt["bundle_body_sha256"],
            "indexed_manifest_file_sha256": indexed_binding[
                "manifest_file_sha256"
            ],
            "indexed_manifest_body_sha256": indexed_binding[
                "manifest_body_sha256"
            ],
            "world_state_set_identity": dict(
                indexed_binding["world_state_set_identity"]
            ),
            "checkpoint_occurrence_projection_content_sha256": projection[
                "projection_content_sha256"
            ],
            "absolute_tick_bundle_digest": _publication_digest_hex(
                projection["absolute_tick_bundle_digest"],
                "absolute tick bundle digest",
            ),
            "occurrence_digest": _publication_digest_hex(
                projection["occurrence_digest"], "occurrence digest"
            ),
        }
        publications.append(
            {**body, "publication_seal_sha256": canonical_sha256(body)}
        )
    return validate_workhorse_indexed_checkpoint_publication_seals_v2(
        publications,
        contract=contract,
        checkpoint_ticks=checkpoint_ticks,
    )


def _finalize_indexed_continuation_provider_result_v2(
    *,
    provider_result: Mapping[str, Any],
    operation: Mapping[str, Any],
    manifest_path: Path,
    canonical_terminal_binding: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "operation",
        "shape_id",
        "backend_id",
        "B",
        "preferred_temporal_k",
        "source_world_state_set_sha256",
        "science_continuation_identity_sha256",
        "physical_continuation_plan_sha256",
        "execution_result_path",
        "execution_result_file_sha256",
        "execution_result_content_sha256",
        "absolute_frontiers",
        "checkpoint_ticks",
        "aggregate_world_ticks_per_second",
        "indexed_checkpoint_receipts",
        "acs_checkpoint_staging",
    }
    logical_continuation = operation["continuation_contract"][
        "science_continuation_identity"
    ]
    selected_world_observation_required = (
        "source_semantic_world_sha256" in logical_continuation
    )
    if selected_world_observation_required:
        expected_fields.add("host_boundary_observation")
    if type(provider_result) is not dict or set(provider_result) != expected_fields:
        raise CampaignProviderError(
            "indexed continuation provider result fields are inexact"
        )
    contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
        operation["contract"]
    )
    continuation = operation["continuation_contract"]
    result_path = Path(str(provider_result["execution_result_path"]))
    result = validate_workhorse_indexed_continuation_result_v2(result_path)
    if (
        provider_result["schema_version"]
        != "eiel_workhorse_indexed_continuation_provider_result_v2"
        or provider_result["shape_id"] != contract.shape_id
        or provider_result["backend_id"] != contract.backend_id
        or provider_result["B"] != contract.B
        or provider_result["preferred_temporal_k"] != contract.temporal_k
        or provider_result["source_world_state_set_sha256"]
        != operation["source_checkpoint_identity_sha256"]
        or provider_result["science_continuation_identity_sha256"]
        != continuation["science_continuation_identity"][
            "science_continuation_identity_sha256"
        ]
        or provider_result["physical_continuation_plan_sha256"]
        != continuation["physical_continuation_plan"][
            "physical_continuation_plan_sha256"
        ]
        or result_path.resolve()
        != manifest_path.parent / "workhorse-indexed-continuation-result.json"
        or result_path.is_symlink()
        or not result_path.is_file()
        or provider_result["execution_result_file_sha256"]
        != result["receipt_file_sha256"]
        or provider_result["execution_result_content_sha256"]
        != result["receipt_content_sha256"]
        or provider_result["absolute_frontiers"] != result["absolute_frontiers"]
        or provider_result["checkpoint_ticks"] != continuation["checkpoint_ticks"]
        or provider_result["aggregate_world_ticks_per_second"]
        != result["aggregate_world_ticks_per_second"]
    ):
        raise CampaignProviderError(
            "indexed continuation provider result changed identity"
        )
    publications = _indexed_continuation_publication_seals_v2(
        provider_result=provider_result,
        contract=contract,
        checkpoint_ticks=continuation["checkpoint_ticks"],
    )
    host_observation_binding: dict[str, Any] | None = None
    if selected_world_observation_required:
        pointer = provider_result.get("host_boundary_observation")
        publication = publications[-1]
        if type(pointer) is not dict or set(pointer) != {
            "observation_mode",
            "path",
            "file_sha256",
            "content_sha256",
            "file_bytes",
            "checkpoint_occurrence_sha256",
            "checkpoint_occurrence_projection_sha256",
        }:
            raise CampaignProviderError(
                "selected-world continuation observation pointer is inexact"
            )
        try:
            observation = validate_workhorse_selected_world_observation_v1(
                pointer["path"],
                expected_checkpoint_occurrence_sha256=publication[
                    "occurrence_digest"
                ],
                expected_checkpoint_occurrence_projection_sha256=publication[
                    "checkpoint_occurrence_projection_content_sha256"
                ],
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise CampaignProviderError(
                f"selected-world host observation cannot reopen: {exc}"
            ) from exc
        expected_observation_path = (
            manifest_path.parent
            / "workhorse-selected-world-host-observation.json"
        ).resolve()
        if (
            contract.B != 1
            or pointer["observation_mode"]
            != WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1
            or Path(str(pointer["path"])).resolve() != expected_observation_path
            or pointer["file_sha256"]
            != observation["observation_file_sha256"]
            or pointer["content_sha256"]
            != observation["observation_content_sha256"]
            or pointer["file_bytes"] != observation["observation_file_bytes"]
            or pointer["checkpoint_occurrence_sha256"]
            != publication["occurrence_digest"]
            or pointer["checkpoint_occurrence_projection_sha256"]
            != publication[
                "checkpoint_occurrence_projection_content_sha256"
            ]
        ):
            raise CampaignProviderError(
                "selected-world host observation changed terminal identity"
            )
        host_observation_binding = {
            **dict(pointer),
            "terminally_reopened": True,
            "runtime_observation_mode": "OFF",
            "host_adapter_observation_only": True,
        }
    output = (
        manifest_path.parent
        / "workhorse-terminalized-indexed-continuation-receipt.json"
    )
    receipt = (
        validate_workhorse_terminalized_indexed_continuation_receipt_v2(output)
        if output.exists() and not output.is_symlink()
        else seal_workhorse_terminalized_indexed_continuation_receipt_v2(
            output,
            execution_result_path=result_path,
            canonical_terminal_binding=canonical_terminal_binding,
            indexed_checkpoint_publications=publications,
        )
    )
    if receipt.get("indexed_checkpoint_publications") != publications:
        raise CampaignProviderError(
            "terminalized indexed continuation changed checkpoint publications"
        )
    return {
        "schema_version": "eiel_workhorse_terminalized_operation_binding_v2",
        "operation": operation["operation"],
        "final_receipt_path": receipt["receipt_path"],
        "final_receipt_file_sha256": receipt["receipt_file_sha256"],
        "final_receipt_content_sha256": receipt["receipt_content_sha256"],
        "indexed_continuation_receipt_path": receipt["receipt_path"],
        **(
            {"host_boundary_observation": host_observation_binding}
            if host_observation_binding is not None
            else {}
        ),
        "attempt_id": canonical_terminal_binding.get("attempt_id"),
        "lifecycle_binding": {
            "artifact_family": "science.workhorse_indexed_acs_continuation_v2",
            "retention_class": "R3",
            "owner_writer": (
                "cuda_full_evolution_workhorse_v2_single_operation_owner"
            ),
            "release_predicate": (
                "technical continuation retained; no science authority"
            ),
        },
    }


def _operation_contract(plan: CampaignPlanV1 | Mapping[str, Any]) -> dict[str, Any]:
    value = _payload(plan)
    analysis = value.get("analysis")
    extension = analysis.get("extension") if isinstance(analysis, Mapping) else None
    if (
        not isinstance(extension, Mapping)
        or extension.get("id") != OPERATION_EXTENSION_ID
        or not isinstance(extension.get("contract"), Mapping)
    ):
        raise CampaignProviderError(
            "Workhorse plan lacks its exact operation extension"
        )
    raw = dict(extension["contract"])
    row_fields = {
        "schema_version",
        "operation",
        "contract",
        "artifact",
        "envelope",
        "authoritative_micro_receipt",
        "semantic_gate_receipt",
        "science_continuation_contract_sha256",
        "source_checkpoint_identity_sha256",
        "measurement_schedule",
    }
    exact_founder_row_fields = row_fields | {"exact_founder_bootstrap"}
    calibration_fields = {
        "schema_version",
        "operation",
        "shape_id",
        "backend_id",
        "artifact",
        "compiled_temporal_k",
        "device_identity",
        "reserve_policy",
        "absolute_tick_span",
        "observation_mode",
        "science_continuation_claimed",
        "science_authority_granted",
        "qualification_authority_granted",
    }
    semantic_fields = {
        "schema_version",
        "operation",
        "semantic_case_contract",
        "contract",
        "artifact",
        "envelope",
        "science_checkpoint_compatibility_binding",
        "science_continuation_contract_sha256",
        "source_checkpoint_identity_sha256",
        "external_source_checkpoint_binding",
    }
    science_fields = {
        "schema_version",
        "operation",
        "contract",
        "artifact",
        "envelope",
        "semantic_inheritance_binding",
        "science_checkpoint_compatibility_binding",
        "science_demo_contract",
        "science_continuation_contract_sha256",
        "source_checkpoint_identity_sha256",
    }
    continuation_fields = {
        "schema_version",
        "operation",
        "contract",
        "artifact",
        "envelope",
        "authoritative_micro_receipt",
        "semantic_gate_receipt",
        "science_continuation_contract_sha256",
        "source_checkpoint_identity_sha256",
        "continuation_contract",
    }
    if raw.get("schema_version") not in {OPERATION_SCHEMA_V2, OPERATION_SCHEMA_V3}:
        raise CampaignProviderError("Workhorse operation schema changed")
    operation_name = raw.get("operation")
    if operation_name not in SUPPORTED_OPERATIONS_V2:
        raise CampaignProviderError("canonical provider operation is unsupported")
    if operation_name == SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1:
        if (
            raw.get("schema_version") != OPERATION_SCHEMA_V3
            or set(raw) != science_fields
        ):
            raise CampaignProviderError(
                "Workhorse science demo operation fields are inexact"
            )
        try:
            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                raw["contract"]
            )
            artifact_full = workhorse_artifact_binding_v2(raw["artifact"]["path"])
            envelope = _sealed_binding(raw["envelope"], label="device envelope")
            semantic = validate_workhorse_science_semantic_inheritance_binding_v1(
                raw["semantic_inheritance_binding"]
            )
            science = validate_workhorse_science_checkpoint_compatibility_binding_v2(
                raw["science_checkpoint_compatibility_binding"]
            )
            demo = validate_workhorse_science_demo_contract_v1(
                raw["science_demo_contract"],
                artifact_binding=artifact_full,
                envelope_binding=envelope,
                semantic_inheritance_binding=semantic,
                science_checkpoint_compatibility_binding=science,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise CampaignProviderError(
                f"Workhorse science demo contract is invalid: {exc}"
            ) from exc
        artifact = raw["artifact"]
        if type(artifact) is not dict or set(artifact) != {
            "path",
            "sha256",
            "sealed_artifact_identity_sha256",
        }:
            raise CampaignProviderError("Workhorse science artifact binding is inexact")
        normalized_artifact = {
            "path": str(artifact["path"]),
            "sha256": _sha(artifact["sha256"], label="artifact digest"),
            "sealed_artifact_identity_sha256": _sha(
                artifact["sealed_artifact_identity_sha256"],
                label="sealed artifact identity",
            ),
        }
        if normalized_artifact != {
            "path": artifact_full["path"],
            "sha256": artifact_full["sha256"],
            "sealed_artifact_identity_sha256": artifact_full[
                "sealed_artifact_identity_sha256"
            ],
        }:
            raise CampaignProviderError("Workhorse science artifact changed")
        science_digest = _sha(
            raw["science_continuation_contract_sha256"],
            label="science continuation contract",
        )
        if (
            contract.shape_id != demo["shape_id"]
            or contract.backend_id != demo["backend_id"]
            or contract.B != demo["B"]
            or contract.temporal_k != demo["temporal_k"]
            or contract.start_next_tick != demo["start_next_tick"]
            or contract.total_ticks != demo["total_ticks"]
            or science_digest != science["science_consumer_extension_contract_sha256"]
            or raw["source_checkpoint_identity_sha256"] is not None
        ):
            raise CampaignProviderError(
                "Workhorse science execution differs from its closed demo"
            )
        execution = value.get("execution")
        if (
            not isinstance(execution, Mapping)
            or execution.get("start_next_tick", 0) != 0
            or execution.get("total_ticks") != 40
            or execution.get("checkpoint_ticks") != [16, 32, 40]
        ):
            raise CampaignProviderError(
                "science demo plan differs from its absolute-tick schedule"
            )
        if extension.get("contract_sha256") != canonical_sha256(raw):
            raise CampaignProviderError("Workhorse operation extension digest changed")
        return {
            **raw,
            "contract": contract.to_dict(),
            "artifact": normalized_artifact,
            "envelope": envelope,
            "semantic_inheritance_binding": semantic,
            "science_checkpoint_compatibility_binding": science,
            "science_demo_contract": demo,
            "science_continuation_contract_sha256": science_digest,
            "operation_contract_sha256": canonical_sha256(raw),
        }
    if operation_name == INDEXED_ACS_CONTINUATION_OPERATION_V2:
        if raw.get("schema_version") != OPERATION_SCHEMA_V3 or set(raw) != continuation_fields:
            raise CampaignProviderError(
                "Workhorse indexed continuation operation fields are inexact"
            )
        try:
            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                raw["contract"]
            )
            continuation = validate_workhorse_indexed_continuation_contract_v2(
                raw["continuation_contract"]
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise CampaignProviderError(
                f"Workhorse indexed continuation contract is invalid: {exc}"
            ) from exc
        artifact = raw["artifact"]
        if type(artifact) is not dict or set(artifact) != {
            "path",
            "sha256",
            "sealed_artifact_identity_sha256",
        }:
            raise CampaignProviderError(
                "Workhorse continuation artifact binding is inexact"
            )
        artifact = {
            "path": str(artifact["path"]),
            "sha256": _sha(artifact["sha256"], label="artifact digest"),
            "sealed_artifact_identity_sha256": _sha(
                artifact["sealed_artifact_identity_sha256"],
                label="sealed artifact identity",
            ),
        }
        envelope = _sealed_binding(raw["envelope"], label="device envelope")
        micro = _sealed_binding(
            raw["authoritative_micro_receipt"],
            label="authoritative micro receipt",
        )
        semantic = _sealed_binding(
            raw["semantic_gate_receipt"], label="semantic gate"
        )
        science = _sha(
            raw["science_continuation_contract_sha256"],
            label="science continuation contract",
        )
        source_checkpoint = _sha(
            raw["source_checkpoint_identity_sha256"],
            label="source world-state set identity",
        )
        logical = continuation["science_continuation_identity"]
        physical = continuation["physical_continuation_plan"]
        execution = value.get("execution")
        target = contract.start_next_tick + contract.total_ticks
        if (
            physical["target_contract"] != contract.to_dict()
            or logical["science_continuation_contract_sha256"] != science
            or logical["source_world_state_set_sha256"] != source_checkpoint
            or continuation["dispatch_plan"]["preferred_temporal_k"]
            != contract.temporal_k
            or not isinstance(execution, Mapping)
            or execution.get("start_next_tick") != contract.start_next_tick
            or execution.get("total_ticks") != target
            or execution.get("checkpoint_ticks")
            != continuation["checkpoint_ticks"]
        ):
            raise CampaignProviderError(
                "Workhorse indexed continuation plan changed its carrier, "
                "packing, or absolute-tick schedule"
            )
        if extension.get("contract_sha256") != canonical_sha256(raw):
            raise CampaignProviderError(
                "Workhorse operation extension digest changed"
            )
        return {
            **raw,
            "contract": contract.to_dict(),
            "artifact": artifact,
            "envelope": envelope,
            "authoritative_micro_receipt": micro,
            "semantic_gate_receipt": semantic,
            "science_continuation_contract_sha256": science,
            "source_checkpoint_identity_sha256": source_checkpoint,
            "continuation_contract": continuation,
            "operation_contract_sha256": canonical_sha256(raw),
        }
    if operation_name == SEMANTIC_EVIDENCE_CASE_OPERATION_V2:
        if set(raw) != semantic_fields:
            raise CampaignProviderError(
                "Workhorse semantic operation fields are inexact"
            )
        semantic_case = raw["semantic_case_contract"]
        if not isinstance(semantic_case, Mapping):
            raise CampaignProviderError("Workhorse semantic case contract is absent")
        case_law = semantic_case.get("case_law")
        case_id = case_law.get("case_id") if isinstance(case_law, Mapping) else None
        semantic_temporal_k = semantic_case.get("semantic_temporal_k")
        shape_id = semantic_case.get("shape_id")
        if (
            type(case_id) is not str
            or not case_id
            or type(semantic_temporal_k) is not int
            or type(shape_id) is not str
        ):
            raise CampaignProviderError(
                "Workhorse semantic case identity is incomplete"
            )
        try:
            expected_semantic_case = workhorse_semantic_case_operation_contract_v2(
                case_id=case_id,
                shape_id=shape_id,
                semantic_temporal_k=semantic_temporal_k,
            )
            law = workhorse_semantic_case_law_v2(
                case_id,
                semantic_temporal_k=semantic_temporal_k,
            )
            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                raw["contract"]
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CampaignProviderError(
                f"Workhorse semantic case contract is invalid: {exc}"
            ) from exc
        if dict(semantic_case) != expected_semantic_case:
            raise CampaignProviderError(
                "Workhorse semantic case contract differs from the closed case table"
            )
        backend_role_valid = (
            contract.backend_id == BACKEND_REFERENCE_PHASED_V2
            if law.artifact_role == "REFERENCE"
            else contract.backend_id
            in {BACKEND_PHASE_GRAPH_V2, BACKEND_FUSED_CTA_TEMPORAL_V2}
        )
        if (
            contract.shape_id != shape_id
            or contract.B != law.B
            or contract.temporal_k != law.dispatch_temporal_k
            or contract.total_ticks != law.total_ticks
            or contract.start_next_tick != law.start_next_absolute_tick
            or not backend_role_valid
        ):
            raise CampaignProviderError(
                "Workhorse execution contract differs from its semantic case law"
            )
        artifact = raw["artifact"]
        if type(artifact) is not dict or set(artifact) != {
            "path",
            "sha256",
            "sealed_artifact_identity_sha256",
        }:
            raise CampaignProviderError("Workhorse artifact binding is inexact")
        artifact = {
            "path": str(artifact["path"]),
            "sha256": _sha(artifact["sha256"], label="artifact digest"),
            "sealed_artifact_identity_sha256": _sha(
                artifact["sealed_artifact_identity_sha256"],
                label="sealed artifact identity",
            ),
        }
        envelope = _sealed_binding(raw["envelope"], label="device envelope")
        try:
            science_binding = (
                validate_workhorse_science_checkpoint_compatibility_binding_v2(
                    raw["science_checkpoint_compatibility_binding"]
                )
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CampaignProviderError(
                f"Workhorse science compatibility binding is invalid: {exc}"
            ) from exc
        science_digest = _sha(
            raw["science_continuation_contract_sha256"],
            label="science continuation contract",
        )
        if (
            science_digest
            != science_binding["science_consumer_extension_contract_sha256"]
        ):
            raise CampaignProviderError(
                "Workhorse semantic science continuation identity changed"
            )
        source_checkpoint = raw["source_checkpoint_identity_sha256"]
        external_source = raw["external_source_checkpoint_binding"]
        if law.requires_source_checkpoint:
            source_checkpoint = _sha(
                source_checkpoint,
                label="semantic source checkpoint identity",
            )
            if not isinstance(external_source, Mapping):
                raise CampaignProviderError(
                    "restore semantic case lacks its external checkpoint binding"
                )
            try:
                external_source = validate_external_checkpoint_source_binding_v2(
                    external_source,
                    expected_identity_sha256=source_checkpoint,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise CampaignProviderError(
                    f"semantic external checkpoint binding is invalid: {exc}"
                ) from exc
        elif source_checkpoint is not None or external_source is not None:
            raise CampaignProviderError(
                "semantic case gained an unauthorized source checkpoint"
            )
        execution = value.get("execution")
        target = contract.start_next_tick + contract.total_ticks
        if (
            not isinstance(execution, Mapping)
            or execution.get("start_next_tick", 0) != contract.start_next_tick
            or execution.get("total_ticks") != target
            or execution.get("checkpoint_ticks") != [target]
        ):
            raise CampaignProviderError(
                "semantic plan differs from its absolute-tick case law"
            )
        if extension.get("contract_sha256") != canonical_sha256(raw):
            raise CampaignProviderError("Workhorse operation extension digest changed")
        return {
            **raw,
            "semantic_case_contract": expected_semantic_case,
            "contract": contract.to_dict(),
            "artifact": artifact,
            "envelope": envelope,
            "science_checkpoint_compatibility_binding": science_binding,
            "science_continuation_contract_sha256": science_digest,
            "source_checkpoint_identity_sha256": source_checkpoint,
            "external_source_checkpoint_binding": external_source,
            "launchpad_operation_contract_sha256": canonical_sha256(raw),
            "operation_contract_sha256": expected_semantic_case[
                "operation_contract_sha256"
            ],
        }
    if operation_name == DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2:
        if set(raw) != calibration_fields:
            raise CampaignProviderError(
                "Workhorse calibration operation fields are inexact"
            )
        shape = next(
            (
                item
                for item in SUPPORTED_WORKHORSE_SHAPES_V2
                if item.shape_id == raw["shape_id"]
            ),
            None,
        )
        if shape is None:
            raise CampaignProviderError("calibration shape is unsupported")
        backend_id = raw["backend_id"]
        if type(backend_id) is not str or not backend_id:
            raise CampaignProviderError("calibration backend is absent")
        compiled_k = raw["compiled_temporal_k"]
        if (
            type(compiled_k) is not list
            or compiled_k
            != list(workhorse_backend_temporal_k_candidates_v2(shape, backend_id))
            or len(set(compiled_k)) != len(compiled_k)
        ):
            raise CampaignProviderError(
                "calibration must bind every compiled K exactly"
            )
        artifact = raw["artifact"]
        if type(artifact) is not dict or set(artifact) != {
            "path",
            "sha256",
            "sealed_artifact_identity_sha256",
        }:
            raise CampaignProviderError("Workhorse artifact binding is inexact")
        artifact = {
            "path": str(artifact["path"]),
            "sha256": _sha(artifact["sha256"], label="artifact digest"),
            "sealed_artifact_identity_sha256": _sha(
                artifact["sealed_artifact_identity_sha256"],
                label="sealed artifact identity",
            ),
        }
        if (
            raw["observation_mode"] != "OFF"
            or raw["science_continuation_claimed"] is not False
            or raw["science_authority_granted"] is not False
            or raw["qualification_authority_granted"] is not False
        ):
            raise CampaignProviderError(
                "calibration must remain technical-only with observation OFF"
            )
        if extension.get("contract_sha256") != canonical_sha256(raw):
            raise CampaignProviderError("Workhorse operation extension digest changed")
        span = _absolute_tick_span(raw["absolute_tick_span"])
        execution = value.get("execution")
        if (
            not isinstance(execution, Mapping)
            or execution.get("start_next_tick", 0) != span["start_next_tick"]
            or execution.get("total_ticks") != span["total_ticks"]
            or execution.get("checkpoint_ticks") != [span["target_next_tick"]]
        ):
            raise CampaignProviderError(
                "calibration plan differs from its absolute-tick span"
            )
        return {
            **raw,
            "artifact": artifact,
            "device_identity": _device_identity(raw["device_identity"]),
            "reserve_policy": _reserve_policy(raw["reserve_policy"]),
            "absolute_tick_span": span,
            "operation_contract_sha256": canonical_sha256(raw),
        }
    exact_founder = raw.get("exact_founder_bootstrap")
    ordinary_row = raw.get("schema_version") == OPERATION_SCHEMA_V2 and (
        set(raw) == row_fields
    )
    exact_founder_row = raw.get("schema_version") == OPERATION_SCHEMA_V3 and (
        set(raw) == exact_founder_row_fields
    )
    if not ordinary_row and not exact_founder_row:
        raise CampaignProviderError("Workhorse row operation fields are inexact")
    if exact_founder_row and exact_founder != (
        workhorse_exact_founder_bootstrap_contract_v1()
    ):
        raise CampaignProviderError(
            "Workhorse exact-founder bootstrap contract changed"
        )
    contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(raw["contract"])
    artifact = raw["artifact"]
    if type(artifact) is not dict or set(artifact) != {
        "path",
        "sha256",
        "sealed_artifact_identity_sha256",
    }:
        raise CampaignProviderError("Workhorse artifact binding is inexact")
    artifact = {
        "path": str(artifact["path"]),
        "sha256": _sha(artifact["sha256"], label="artifact digest"),
        "sealed_artifact_identity_sha256": _sha(
            artifact["sealed_artifact_identity_sha256"],
            label="sealed artifact identity",
        ),
    }
    envelope = _sealed_binding(raw["envelope"], label="device envelope")
    micro = _sealed_binding(
        raw["authoritative_micro_receipt"], label="authoritative micro receipt"
    )
    semantic = _sealed_binding(raw["semantic_gate_receipt"], label="semantic gate")
    science = _sha(
        raw["science_continuation_contract_sha256"],
        label="science continuation contract",
    )
    source_checkpoint = raw["source_checkpoint_identity_sha256"]
    if source_checkpoint is not None:
        _sha(source_checkpoint, label="source checkpoint identity")
    try:
        schedule = validate_workhorse_fixed_tick_measurement_schedule_v3(
            raw["measurement_schedule"], temporal_k=contract.temporal_k
        )
    except Exception as exc:
        raise CampaignProviderError(
            f"Workhorse row measurement schedule is invalid: {exc}"
        ) from exc
    if contract.total_ticks != schedule["total_ticks"]:
        raise CampaignProviderError(
            "Workhorse row horizon must exactly cover the fixed-tick schedule"
        )
    execution = value.get("execution")
    checkpoints = value.get("checkpoints")
    acs_checkpoint_slots = (
        execution.get("acs_checkpoint_slots")
        if isinstance(execution, Mapping)
        and "acs_checkpoint_slots" in execution
        else (
            checkpoints.get("acs_slots")
            if isinstance(checkpoints, Mapping)
            else None
        )
    )
    lifecycle = value.get("artifact_lifecycle")
    roles = (
        lifecycle.get("checkpoints", {}).get("roles")
        if isinstance(lifecycle, Mapping)
        and isinstance(lifecycle.get("checkpoints"), Mapping)
        else None
    )
    target_tick = contract.start_next_tick + contract.total_ticks
    expected_ticks = [contract.temporal_k, target_tick]
    if exact_founder_row and (
        contract.start_next_tick != 0
        or not isinstance(execution, Mapping)
        or acs_checkpoint_slots != 2
        or execution.get("checkpoint_ticks") != expected_ticks
        or not isinstance(roles, list)
        or [
            {
                "checkpoint_tick": row.get("checkpoint_tick"),
                "role": row.get("role"),
                "retention_class": row.get("retention_class"),
            }
            for row in roles
            if isinstance(row, Mapping)
        ]
        != [
            {
                "checkpoint_tick": contract.temporal_k,
                "role": "intermediate",
                "retention_class": "R4",
            },
            {
                "checkpoint_tick": target_tick,
                "role": "terminal",
                "retention_class": "R4",
            },
        ]
    ):
        raise CampaignProviderError(
            "Workhorse exact-founder row requires tick-zero ACS2, first-dispatch "
            "genesis, and terminal R4"
        )
    if extension.get("contract_sha256") != canonical_sha256(raw):
        raise CampaignProviderError("Workhorse operation extension digest changed")
    return {
        **raw,
        "contract": contract.to_dict(),
        "artifact": artifact,
        "envelope": envelope,
        "authoritative_micro_receipt": micro,
        "semantic_gate_receipt": semantic,
        "science_continuation_contract_sha256": science,
        "measurement_schedule": schedule,
        "operation_contract_sha256": canonical_sha256(raw),
    }


def _digest_object_v1(value: object) -> dict[str, str]:
    return {"algorithm": "sha256", "hex": canonical_sha256(value)}


def _exact_founder_plan_projection_v1(
    value: Mapping[str, Any],
    *,
    operation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive the producer-owned semantic and runtime founder projections."""

    contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
        operation["contract"]
    )
    execution = value.get("execution")
    cells = value.get("cells")
    batches = value.get("physical_batches")
    if (
        not isinstance(execution, Mapping)
        or not isinstance(cells, list)
        or len(cells) != 1
        or not isinstance(cells[0], Mapping)
        or not isinstance(batches, list)
        or len(batches) != 1
        or contract.start_next_tick != 0
        or execution.get("acs_checkpoint_slots") != 2
        or execution.get("checkpoint_ticks")
        != [contract.temporal_k, contract.total_ticks]
    ):
        raise CampaignProviderError(
            "Workhorse exact-founder projection requires one tick-zero ACS2 row"
        )
    marker = operation.get("exact_founder_bootstrap")
    if marker != workhorse_exact_founder_bootstrap_contract_v1():
        raise CampaignProviderError(
            "Workhorse exact-founder projection lacks its producer contract"
        )
    identity = workhorse_world_identity_binding_v2(
        world_uids=tuple(
            EXACT_FOUNDER_WORLD_UID_BASE_V1 + ordinal
            for ordinal in range(contract.B)
        ),
        rng_keys=(EXACT_FOUNDER_RNG_KEY_V1,) * contract.B,
    )
    shape = contract.shape
    world = {
        "schema_version": EXACT_FOUNDER_WORLD_SCHEMA_V1,
        "base": WORLD_BASE,
        "shape": {
            "shape_id": shape.shape_id,
            "width": shape.width,
            "height": shape.height,
            "initial_population": shape.initial_population,
            "capacity": shape.capacity,
        },
    }
    initialization = {
        "schema_version": EXACT_FOUNDER_INITIALIZATION_SCHEMA_V1,
        "initializer": "WORKHORSE_INITIALIZE_EXACT_V1",
        "population_fixture": False,
    }
    seed_bindings = dict(cells[0].get("seeds") or {})
    if (
        set(seed_bindings) != {"ecology", "task"}
        or any(type(seed) is not int or seed < 0 for seed in seed_bindings.values())
    ):
        raise CampaignProviderError(
            "Workhorse exact-founder semantic seeds are unavailable"
        )
    controls = [
        {
            "control_id": control_id,
            "resolved_value": resolved_value,
            "schedule": {"kind": "static"},
            "descriptor_sha256": _digest_object_v1(
                {
                    "schema_version": (
                        "eiel_workhorse_native_default_control_descriptor_v1"
                    ),
                    "control_id": control_id,
                    "resolved_value": resolved_value,
                    "authority": "WORKHORSE_NATIVE_COMPILED_DEFAULTS_V1",
                    "unsupported_control_policy": "FAIL_CLOSED",
                }
            ),
        }
        for control_id, resolved_value in sorted(
            marker["founder_controls"].items()
        )
    ]
    initial_state_identity = _digest_object_v1(
        {
            "world": world,
            "initialization": initialization,
            "seed_bindings": seed_bindings,
        }
    )
    semantic_inputs = {
        "world": world,
        "initialization": initialization,
        "dimension_bindings": dict(cells[0].get("dimensions") or {}),
        "control_bindings": controls,
        "seed_bindings": seed_bindings,
        "time": {
            "authority": "absolute_tick_v1",
            "start_next_tick": 0,
            "total_ticks": contract.total_ticks,
            "checkpoint_ticks": [contract.temporal_k, contract.total_ticks],
        },
        "initial_state_identity": initial_state_identity,
    }
    science = {
        "test_id": "WORKHORSE_EXACT_FOUNDER_BOOTSTRAP",
        "test_version": "1",
        "study_id": str(value.get("campaign_id")),
        "arm_id": "NATIVE_DEFAULTS",
        "replicate": 0,
    }
    native_world = project_science_cuda_v2_native_world_v1(
        {"semantic_inputs": semantic_inputs, "science": science}
    )
    semantic_world_sha256 = _digest_object_v1(
        {
            "schema_version": "eiel_science_world_record_v1",
            "science": science,
            "semantic_inputs": semantic_inputs,
        }
    )
    cell_projection = {
        "native_world_control": native_world.to_dict(),
        "science_world_sha256": semantic_world_sha256,
    }
    materialization_body = {
        "schema_version": EXACT_FOUNDER_MATERIALIZATION_SCHEMA_V1,
        "claim_class": "TECHNICAL_ONLY",
        "science_authority_granted": False,
        "qualification_authority_granted": False,
        "producer_id": PROVIDER_ID,
        "runtime_observation_mode": "OFF",
        "exact_founder_bootstrap_contract_sha256": marker["contract_sha256"],
        "world_identity_binding": identity,
        "semantic_world_sha256": semantic_world_sha256,
        "native_world_control_set_sha256": _digest_object_v1(
            {
                "schema_version": (
                    "eiel_science_cuda_v2_native_world_control_set_v1"
                ),
                "ordered_control_digests": [native_world.control_digest],
            }
        ),
        "founder_control_authority": marker["founder_control_authority"],
        "founder_controls": dict(marker["founder_controls"]),
        "unsupported_control_policy": "FAIL_CLOSED",
    }
    expected_control_set = native_world_control_set_sha256_v1([native_world])
    if materialization_body["native_world_control_set_sha256"]["hex"] != (
        expected_control_set
    ):
        raise CampaignProviderError(
            "Workhorse exact-founder native control-set identity changed"
        )
    materialization = {
        **materialization_body,
        "materialization_sha256": _digest_object_v1(materialization_body),
    }
    return cell_projection, materialization


def materialize_workhorse_exact_founder_plan_v1(
    plan: CampaignPlanV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Attach exact founder identity only to the opt-in regular-row plan.

    Ordinary Workhorse plans are returned byte-for-byte.  Exact bootstrap
    identities are derived entirely from the authenticated operation and the
    canonical regular-row initializer; pre-existing mismatched projections
    fail closed instead of being repaired at execution time.
    """

    value = _payload(plan)
    operation = _operation_contract(value)
    marker = operation.get("exact_founder_bootstrap")
    cells = value.get("cells")
    existing_cell = (
        cells[0]
        if isinstance(cells, list)
        and len(cells) == 1
        and isinstance(cells[0], Mapping)
        else {}
    )
    if marker is None:
        if isinstance(value.get("science_materialization"), Mapping) and value[
            "science_materialization"
        ].get("schema_version") == EXACT_FOUNDER_MATERIALIZATION_SCHEMA_V1:
            raise CampaignProviderError(
                "ordinary Workhorse row gained exact-founder identity"
            )
        return value
    cell_projection, materialization = _exact_founder_plan_projection_v1(
        value, operation=operation
    )
    for name, expected in cell_projection.items():
        if name in existing_cell and existing_cell[name] != expected:
            raise CampaignProviderError(
                f"Workhorse exact-founder cell {name} changed"
            )
    existing_materialization = value.get("science_materialization")
    if (
        existing_materialization is not None
        and existing_materialization != materialization
    ):
        raise CampaignProviderError(
            "Workhorse exact-founder science materialization changed"
        )
    bound = dict(value)
    bound_cells = [dict(existing_cell)]
    bound_cells[0].update(cell_projection)
    bound["cells"] = bound_cells
    bound["science_materialization"] = materialization
    bound.pop("plan_sha256", None)
    bound["plan_sha256"] = canonical_sha256(bound)
    return bound


def _exact_founder_runtime_world_identity_v1(
    plan: CampaignPlanV1 | Mapping[str, Any],
) -> dict[str, Any] | None:
    value = materialize_workhorse_exact_founder_plan_v1(plan)
    operation = _operation_contract(value)
    if operation.get("exact_founder_bootstrap") is None:
        return None
    materialization = value.get("science_materialization")
    if not isinstance(materialization, Mapping):
        raise CampaignProviderError(
            "Workhorse exact-founder runtime identity is absent"
        )
    try:
        return validate_workhorse_world_identity_binding_v2(
            materialization["world_identity_binding"],
            expected_world_count=FullEvolutionCudaWorkhorseContractV2.from_mapping(
                operation["contract"]
            ).B,
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise CampaignProviderError(
            f"Workhorse exact-founder runtime identity is invalid: {exc}"
        ) from exc


def _execution_row_checkpoint_ticks_v2(
    plan: CampaignPlanV1 | Mapping[str, Any],
    *,
    operation: Mapping[str, Any],
    contract: FullEvolutionCudaWorkhorseContractV2,
) -> tuple[int, ...]:
    """Reopen the plan-bound row schedule without extending the operation schema."""

    value = _payload(plan)
    execution = value.get("execution")
    if not isinstance(execution, Mapping):
        raise CampaignProviderError("Workhorse row execution control is absent")
    raw_ticks = execution.get("checkpoint_ticks")
    try:
        binding = workhorse_row_checkpoint_binding_v2(
            contract,
            science_continuation_contract_sha256=str(
                operation["science_continuation_contract_sha256"]
            ),
            source_checkpoint_identity_sha256=operation.get(
                "source_checkpoint_identity_sha256"
            ),
            checkpoint_ticks=raw_ticks,
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise CampaignProviderError(
            f"Workhorse row checkpoint schedule is invalid: {exc}"
        ) from exc
    return tuple(binding["checkpoint_ticks"])


def _validate_science_founder_default_controls_v1(
    plan: CampaignPlanV1 | Mapping[str, Any],
    *,
    checkpoint_ticks: Sequence[int],
    acs_checkpoint_slots: int,
) -> None:
    """Fail closed where the regular row still uses native default controls.

    The selected-world continuation route owns the versioned mutable control
    ABI.  A regular Science Request founder row has not yet acquired that
    applicator, so its two Expedition controls must remain the native defaults.
    The two-checkpoint ACS2 bootstrap additionally requires both defaults to be
    explicit and static in the authenticated science-world projection.
    """

    value = _payload(plan)
    if not isinstance(value.get("science_materialization"), Mapping):
        return
    cells = value.get("cells")
    if (
        not isinstance(cells, list)
        or len(cells) != 1
        or not isinstance(cells[0], Mapping)
    ):
        raise CampaignProviderError(
            "Workhorse science founder requires one exact semantic cell"
        )
    native_world = cells[0].get("native_world_control")
    rows = (
        native_world.get("control_bindings")
        if isinstance(native_world, Mapping)
        else None
    )
    if not isinstance(rows, list):
        raise CampaignProviderError(
            "Workhorse science founder lacks its native control projection"
        )
    expected = {
        "environment.resource.multiplier": (
            EXACT_FOUNDER_RESOURCE_MULTIPLIER_V1
        ),
        "mutation.rate": EXACT_FOUNDER_MUTATION_RATE_V1,
    }
    observed: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise CampaignProviderError(
                "Workhorse science founder control binding is invalid"
            )
        control_id = raw.get("control_id")
        if control_id in expected:
            if control_id in observed:
                raise CampaignProviderError(
                    "Workhorse science founder duplicates an Expedition control"
                )
            observed[str(control_id)] = raw
    for control_id, raw in observed.items():
        if (
            raw.get("resolved_value") != expected[control_id]
            or raw.get("schedule") != {"kind": "static"}
        ):
            raise CampaignProviderError(
                "regular Workhorse founder controls must remain the declared "
                "native defaults resource=1.0 and mutation=0.125"
            )
    bootstrap = len(tuple(checkpoint_ticks)) == 2
    if bootstrap and (
        acs_checkpoint_slots != 2
        or set(observed) != set(expected)
    ):
        raise CampaignProviderError(
            "Workhorse founder exact-state bootstrap requires ACS2 and explicit "
            "static native defaults resource=1.0 and mutation=0.125"
        )


def is_cuda_full_evolution_workhorse_v2_spec_v1(spec: Mapping[str, Any]) -> bool:
    world = spec.get("world")
    return isinstance(world, Mapping) and world.get("base") == WORLD_BASE


def is_cuda_full_evolution_workhorse_v2_profile_v1(profile_id: object) -> bool:
    return type(profile_id) is str and any(
        profile_id == shape.profile_id for shape in SUPPORTED_WORKHORSE_SHAPES_V2
    )


def cuda_full_evolution_workhorse_v2_signed_planning_resource_accounting_v2(
    *,
    spec: Mapping[str, Any],
    logical_cells: int,
    physical_batch_cells: int,
    compiled_batch_count: int,
    comparison_branch_count: int,
) -> dict[str, Any] | None:
    """Bind Workhorse disk admission to the sealed physical operation contract."""

    if not is_cuda_full_evolution_workhorse_v2_spec_v1(spec):
        raise CampaignProviderError(
            "signed Workhorse planning accounting requires its exact world"
        )
    operation = _operation_contract(spec)
    if operation["operation"] == DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2:
        return None
    if (
        logical_cells != 1
        or physical_batch_cells != 1
        or compiled_batch_count != 1
        or comparison_branch_count != 0
    ):
        raise CampaignProviderError(
            "signed Workhorse planning accounting requires one physical operation"
        )
    contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
        operation["contract"]
    )
    execution = spec.get("execution")
    checkpoints = spec.get("checkpoints")
    if not isinstance(execution, Mapping) or not isinstance(checkpoints, Mapping):
        raise CampaignProviderError(
            "signed Workhorse planning accounting lacks execution/checkpoints"
        )
    checkpoint_ticks = tuple(execution.get("checkpoint_ticks") or ())
    acs_slots = checkpoints.get("acs_slots", 0)
    if operation["operation"] == EXECUTION_ROW_OPERATION_V2:
        schedule = operation["measurement_schedule"]
        dispatch_count = (
            schedule["warmup_dispatches"] + schedule["measured_dispatches"]
        )
    elif operation["operation"] == INDEXED_ACS_CONTINUATION_OPERATION_V2:
        dispatch_count = operation["continuation_contract"]["dispatch_plan"][
            "dispatch_count"
        ]
    else:
        dispatch_count = (
            contract.total_ticks + contract.temporal_k - 1
        ) // contract.temporal_k
    launchpad_operation_contract_sha256 = operation.get(
        "launchpad_operation_contract_sha256",
        operation["operation_contract_sha256"],
    )
    storage = workhorse_checkpoint_storage_accounting_v2(
        contract=contract,
        checkpoint_ticks=checkpoint_ticks,
        acs_checkpoint_slots=acs_slots,
        dispatch_count=dispatch_count,
    )
    checkpoint_schedule_sha256 = canonical_sha256(
        {
            "time_authority": "absolute_tick_v1",
            "checkpoint_ticks": list(checkpoint_ticks),
        }
    )
    binding = {
        "schema_version": WORKHORSE_SIGNED_PLANNING_ACCOUNTING_SCHEMA_V2,
        "profile_id": contract.profile_id,
        "operation": operation["operation"],
        "operation_contract_sha256": launchpad_operation_contract_sha256,
        "logical_cells": logical_cells,
        "physical_batch_cells": physical_batch_cells,
        "compiled_batch_count": compiled_batch_count,
        "shape_id": contract.shape_id,
        "backend_id": contract.backend_id,
        "physical_B": contract.B,
        "temporal_k": contract.temporal_k,
        "total_ticks": contract.total_ticks,
        "checkpoint_schedule_sha256": checkpoint_schedule_sha256,
        "storage_accounting_sha256": canonical_sha256(storage),
    }
    return {
        **binding,
        "authority": "signed_provider_operation_contract_v2",
        "resource_accounting_binding_sha256": canonical_sha256(binding),
        "storage_accounting": storage,
    }


def _profile(
    shape: WorkhorseCompileShapeV2,
    *,
    operation: str = EXECUTION_ROW_OPERATION_V2,
) -> RuntimeProfileV1:
    if operation not in SUPPORTED_OPERATIONS_V2:
        raise CampaignProviderError(f"unsupported Workhorse operation: {operation!r}")
    metrics = MetricFamilyV1(
        family_id="workhorse_v2_single_operation_technical_metrics",
        metrics=(
            "workhorse.p10_world_ticks_per_second",
            INDEXED_ACS_CONTINUATION_AGGREGATE_WTPS_METRIC_V2,
            "workhorse.device_envelope_calibration_completed",
            "workhorse.semantic_case_provisional_completed",
            "workhorse.science_compatibility_demo_completed",
        ),
        required_state_components=FULL_EVOLUTION_CAPABILITIES_V2,
        null_semantics="null means the exact technical operation did not measure it",
    )
    common_parameters = {
        "workhorse.operation": ParameterContractV1(
            "workhorse.operation",
            str,
            lambda value, expected=operation: value == expected,
            "one canonical Workhorse operation",
            per_cell=False,
        ),
        "workhorse.shape_id": ParameterContractV1(
            "workhorse.shape_id",
            str,
            lambda value, expected=shape.shape_id: value == expected,
            "exact compile-time Workhorse shape",
            per_cell=False,
        ),
        "workhorse.backend_id": ParameterContractV1(
            "workhorse.backend_id",
            str,
            lambda value: bool(value),
            "exact Workhorse backend identity",
            per_cell=False,
        ),
    }
    if operation in {
        EXECUTION_ROW_OPERATION_V2,
        INDEXED_ACS_CONTINUATION_OPERATION_V2,
    }:
        operation_parameters = {
            "workhorse.B": ParameterContractV1(
                "workhorse.B",
                int,
                lambda value: 1 <= value <= 65_535,
                "immutable grid.y world batch",
                per_cell=False,
            ),
            "workhorse.temporal_k": ParameterContractV1(
                "workhorse.temporal_k",
                int,
                lambda value, allowed=shape.temporal_k_candidates: value in allowed,
                "immutable absolute-tick dispatch span",
                per_cell=False,
            ),
        }
    elif operation == DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2:
        expected_k_digests = {
            canonical_sha256(
                list(workhorse_backend_temporal_k_candidates_v2(shape, backend_id))
            )
            for backend_id in (
                BACKEND_REFERENCE_PHASED_V2,
                BACKEND_PHASE_GRAPH_V2,
                BACKEND_FUSED_CTA_TEMPORAL_V2,
            )
        }
        operation_parameters = {
            "workhorse.compiled_temporal_k_sha256": ParameterContractV1(
                "workhorse.compiled_temporal_k_sha256",
                str,
                lambda value, expected=expected_k_digests: value in expected,
                "digest of the complete ordered backend-supported K list",
                per_cell=False,
            ),
            "workhorse.reserve_bytes": ParameterContractV1(
                "workhorse.reserve_bytes",
                int,
                lambda value: value >= 0,
                "explicit stable-total calibration reserve",
                per_cell=False,
            ),
        }
    elif operation == SEMANTIC_EVIDENCE_CASE_OPERATION_V2:
        operation_parameters = {
            "workhorse.semantic_case_id": ParameterContractV1(
                "workhorse.semantic_case_id",
                str,
                lambda value: bool(value) and value == value.strip(),
                "closed Workhorse semantic case identity",
                per_cell=False,
            ),
            "workhorse.B": ParameterContractV1(
                "workhorse.B",
                int,
                lambda value: 1 <= value <= 2,
                "closed semantic-case world batch",
                per_cell=False,
            ),
            "workhorse.temporal_k": ParameterContractV1(
                "workhorse.temporal_k",
                int,
                lambda value, allowed=shape.temporal_k_candidates: value in allowed,
                "closed semantic-case dispatch span",
                per_cell=False,
            ),
            "workhorse.semantic_temporal_k": ParameterContractV1(
                "workhorse.semantic_temporal_k",
                int,
                lambda value, allowed=shape.temporal_k_candidates: (
                    value in allowed and value > 1
                ),
                "compiled K governing the closed 24-case table",
                per_cell=False,
            ),
            "workhorse.attempt_terminal_status": ParameterContractV1(
                "workhorse.attempt_terminal_status",
                str,
                lambda value: value in {"COMPLETED", "FAILED"},
                "closed semantic-case Scheduler terminal status",
                per_cell=False,
            ),
        }
    else:
        operation_parameters = {
            "workhorse.B": ParameterContractV1(
                "workhorse.B",
                int,
                lambda value: value == 10,
                "closed science-demo world batch",
                per_cell=False,
            ),
            "workhorse.temporal_k": ParameterContractV1(
                "workhorse.temporal_k",
                int,
                lambda value: value == 1,
                "closed one-tick science-demo dispatch",
                per_cell=False,
            ),
        }
    parameter_contracts = {**common_parameters, **operation_parameters}
    return RuntimeProfileV1(
        profile_id=shape.profile_id,
        profile_version=2,
        backend="cuda_resident",
        world_base=WORLD_BASE,
        world_overlays=frozenset(),
        policy_family="fixed_mlp",
        selection_modes=frozenset({"standard"}),
        observation_contract="OFF",
        state_components=FULL_EVOLUTION_CAPABILITIES_V2,
        checkpoint_components=(
            "absolute_tick_frontiers",
            "source_checkpoint_identity",
            "science_continuation_contract_identity",
        ),
        parameter_contracts=parameter_contracts,
        uniform_batch_parameters=tuple(parameter_contracts),
        metric_families=(metrics,),
        execution_owner="cuda_full_evolution_workhorse_v2_single_row_owner",
        arbitrary_positive_generations=True,
        continuation_supported=(operation == INDEXED_ACS_CONTINUATION_OPERATION_V2),
        cancellation_supported=False,
        batch_min=1,
        batch_max=1,
        qualified_batch_min=1,
        qualified_batch_max=1,
        generation_counter_bits=64,
        qualification_invariants={
            "route_id": shape.route_id,
            "shape_id": shape.shape_id,
            "time_authority": "absolute_tick_v1",
            "operation_cardinality": "one_exact_operation_per_attempt",
            "operation": operation,
            "formal_qualification_authority": "none",
        },
        qualification_status="experimental",
        limitations=(
            "TECHNICAL_ONLY; execution grants no Qualification or science authority.",
            "Only one exact Workhorse operation is supported per ExecutionRequestV2.",
            "Offline aggregation and winner selection are separate non-authoritative steps.",
        ),
    )


def cuda_full_evolution_workhorse_v2_profile_registry_v1(
    *,
    shape_id: str | None = None,
    operation: str = EXECUTION_ROW_OPERATION_V2,
) -> ProfileRegistryV1:
    registry = ProfileRegistryV1()
    for shape in SUPPORTED_WORKHORSE_SHAPES_V2:
        if shape_id is not None and shape.shape_id != shape_id:
            continue
        registry.register(_profile(shape, operation=operation))
    if not registry.profiles():
        raise CampaignProviderError(
            f"unsupported Workhorse profile shape: {shape_id!r}"
        )
    return registry


class CudaFullEvolutionWorkhorseV2ExecutionProviderV1:
    """Own exactly one pre-concretized Workhorse operation."""

    provider_id = PROVIDER_ID
    resident_segment_generation_min = 1
    reporting_parity_supported = False

    def __init__(
        self,
        *,
        shape: WorkhorseCompileShapeV2,
        operation_adapter: WorkhorseLaunchpadOperationAdapterV2 | None = None,
        acs_require_native_pool: bool = True,
    ) -> None:
        self.shape = shape
        self.profile_id = shape.profile_id
        self.operation_adapter = operation_adapter
        if type(acs_require_native_pool) is not bool:
            raise CampaignProviderError(
                "Workhorse ACS native-pool policy must be an exact bool"
            )
        self._acs_require_native_pool = acs_require_native_pool

    @staticmethod
    def _single_batch(
        plan: Mapping[str, Any], batch_id: str | None = None
    ) -> Mapping[str, Any]:
        batches = plan.get("physical_batches")
        if not isinstance(batches, list) or len(batches) != 1:
            raise CampaignProviderError(
                "Workhorse ExecutionRequestV2 must compile exactly one physical batch"
            )
        batch = batches[0]
        if not isinstance(batch, Mapping) or (
            batch_id is not None and batch.get("batch_id") != batch_id
        ):
            raise CampaignProviderError("Workhorse physical batch identity differs")
        cells = batch.get("cell_ids")
        if (
            not isinstance(cells, Sequence)
            or isinstance(cells, (str, bytes))
            or len(cells) != 1
        ):
            raise CampaignProviderError(
                "one Workhorse operation must contain exactly one campaign cell"
            )
        return batch

    def translate_batch(
        self, plan: CampaignPlanV1 | Mapping[str, Any], batch_id: str
    ) -> ResidentBatchRequestV1:
        value = _payload(plan)
        operation = _operation_contract(value)
        batch = self._single_batch(value, batch_id)
        execution = value.get("execution")
        acs_checkpoint_slots = (
            execution.get("acs_checkpoint_slots", 0)
            if isinstance(execution, Mapping)
            else 0
        )
        if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {0, 1, 2}:
            raise CampaignProviderError(
                "Workhorse ACS checkpoint slots must be zero, one, or two"
            )
        if operation["operation"] in {
            EXECUTION_ROW_OPERATION_V2,
            INDEXED_ACS_CONTINUATION_OPERATION_V2,
        }:
            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                operation["contract"]
            )
            if contract.shape_id != self.shape.shape_id:
                raise CampaignProviderError("Workhorse provider shape differs from row")
            start = contract.start_next_tick
            total = contract.total_ticks
            target = start + total
            dispatch = contract.temporal_k
            event_capacity = 2 * contract.shape.capacity
            parameter_values = (
                ("workhorse.operation", operation["operation"]),
                ("workhorse.shape_id", contract.shape_id),
                ("workhorse.backend_id", contract.backend_id),
                ("workhorse.B", contract.B),
                ("workhorse.temporal_k", contract.temporal_k),
            )
            if operation["operation"] == INDEXED_ACS_CONTINUATION_OPERATION_V2:
                if acs_checkpoint_slots != 2:
                    raise CampaignProviderError(
                        "indexed Workhorse continuation requires the ACS2 two-slot "
                        "pipeline"
                    )
                continuation = operation["continuation_contract"]
                tick_control_schema = (
                    "eiel_workhorse_indexed_continuation_tick_control_v2"
                )
                tick_control_extra = {
                    "dispatch_plan": dict(continuation["dispatch_plan"]),
                    "science_continuation_identity": dict(
                        continuation["science_continuation_identity"]
                    ),
                    "physical_continuation_plan": dict(
                        continuation["physical_continuation_plan"]
                    ),
                }
                checkpoint_ticks = tuple(continuation["checkpoint_ticks"])
            else:
                tick_control_schema = "eiel_workhorse_single_row_tick_control_v2"
                tick_control_extra = {}
                checkpoint_ticks = _execution_row_checkpoint_ticks_v2(
                    value,
                    operation=operation,
                    contract=contract,
                )
                if len(checkpoint_ticks) > max(1, acs_checkpoint_slots):
                    raise CampaignProviderError(
                        "Workhorse row checkpoint schedule exceeds its ACS slots"
                    )
                _validate_science_founder_default_controls_v1(
                    value,
                    checkpoint_ticks=checkpoint_ticks,
                    acs_checkpoint_slots=acs_checkpoint_slots,
                )
        elif operation["operation"] == DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2:
            if operation["shape_id"] != self.shape.shape_id:
                raise CampaignProviderError(
                    "Workhorse provider shape differs from calibration"
                )
            span = operation["absolute_tick_span"]
            start = span["start_next_tick"]
            total = span["total_ticks"]
            target = span["target_next_tick"]
            dispatch = total
            event_capacity = 1
            compiled_k = list(operation["compiled_temporal_k"])
            parameter_values = (
                ("workhorse.operation", operation["operation"]),
                ("workhorse.shape_id", operation["shape_id"]),
                ("workhorse.backend_id", operation["backend_id"]),
                (
                    "workhorse.compiled_temporal_k_sha256",
                    canonical_sha256(compiled_k),
                ),
                (
                    "workhorse.reserve_bytes",
                    operation["reserve_policy"]["reserve_bytes_requested"],
                ),
            )
            tick_control_schema = (
                "eiel_workhorse_device_envelope_calibration_tick_control_v2"
            )
            tick_control_extra = {
                "compiled_temporal_k": compiled_k,
                "compiled_temporal_k_sha256": canonical_sha256(compiled_k),
                "reserve_policy": dict(operation["reserve_policy"]),
            }
            checkpoint_ticks = (target,)
        elif operation["operation"] == SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1:
            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                operation["contract"]
            )
            if contract.shape_id != self.shape.shape_id:
                raise CampaignProviderError(
                    "Workhorse provider shape differs from science demo"
                )
            start = contract.start_next_tick
            total = contract.total_ticks
            target = start + total
            dispatch = contract.temporal_k
            event_capacity = 2 * contract.shape.capacity
            checkpoint_ticks = tuple(
                operation["science_demo_contract"]["checkpoint_ticks"]
            )
            parameter_values = (
                ("workhorse.operation", operation["operation"]),
                ("workhorse.shape_id", contract.shape_id),
                ("workhorse.backend_id", contract.backend_id),
                ("workhorse.B", contract.B),
                ("workhorse.temporal_k", contract.temporal_k),
            )
            tick_control_schema = "eiel_workhorse_science_demo_tick_control_v1"
            tick_control_extra = {
                "boundary_next_ticks": list(
                    operation["science_demo_contract"]["boundary_next_ticks"]
                ),
                "continuation_restore_ticks": list(
                    operation["science_demo_contract"]["continuation_restore_ticks"]
                ),
                "terminal_restore_validation_tick": operation["science_demo_contract"][
                    "terminal_restore_validation_tick"
                ],
            }
        else:
            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                operation["contract"]
            )
            semantic_case = operation["semantic_case_contract"]
            case_law = semantic_case["case_law"]
            if contract.shape_id != self.shape.shape_id:
                raise CampaignProviderError(
                    "Workhorse provider shape differs from semantic case"
                )
            start = contract.start_next_tick
            total = contract.total_ticks
            target = start + total
            dispatch = contract.temporal_k
            event_capacity = 2 * contract.shape.capacity
            parameter_values = (
                ("workhorse.operation", operation["operation"]),
                ("workhorse.shape_id", contract.shape_id),
                ("workhorse.backend_id", contract.backend_id),
                ("workhorse.semantic_case_id", case_law["case_id"]),
                ("workhorse.B", contract.B),
                ("workhorse.temporal_k", contract.temporal_k),
                (
                    "workhorse.semantic_temporal_k",
                    semantic_case["semantic_temporal_k"],
                ),
                (
                    "workhorse.attempt_terminal_status",
                    semantic_case["attempt_terminal_contract"][
                        "attempt_terminal_status"
                    ],
                ),
            )
            tick_control_schema = "eiel_workhorse_semantic_case_tick_control_v2"
            tick_control_extra = {
                "case_id": case_law["case_id"],
                "semantic_temporal_k": semantic_case["semantic_temporal_k"],
                "attempt_terminal_contract": dict(
                    semantic_case["attempt_terminal_contract"]
                ),
            }
            checkpoint_ticks = (target,)
        request = ResidentBatchRequestV1(
            profile_id=self.profile_id,
            batch_id=batch_id,
            cell_ids=tuple(str(cell) for cell in batch["cell_ids"]),
            checkpoint_milestones=(),
            parameter_arrays=tuple(
                (name, (parameter_value,)) for name, parameter_value in parameter_values
            ),
            seed_arrays=(),
            total_ticks=target,
            start_next_tick=start,
            checkpoint_ticks=checkpoint_ticks,
            checkpoint_storage_mode="disk",
            reporting_enabled=True,
            acs_checkpoint_slots=acs_checkpoint_slots,
            acs_capture_instrumentation=bool(acs_checkpoint_slots),
            native_streamed_checkpoint=True,
            resident_owner_span_ticks=total,
            native_dispatch_window_ticks=dispatch,
            event_window_capacity=event_capacity,
            source_window_ticks=dispatch,
            tick_native_control={
                "schema_version": tick_control_schema,
                "time_authority": "absolute_tick_v1",
                "operation": operation["operation"],
                "total_ticks": target,
                "start_next_tick": start,
                "checkpoint_ticks": list(checkpoint_ticks),
                "resident_owner_span_ticks": total,
                "native_dispatch_window_ticks": dispatch,
                "source_window_ticks": dispatch,
                "event_window_capacity": event_capacity,
                **tick_control_extra,
            },
        )
        route = self.execution_route_contract(value)
        return replace(
            request,
            execution_route_sha256=route.execution_route_sha256,
        )

    def readiness_reasons(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
    ) -> tuple[UnsupportedReasonV1, ...]:
        try:
            value = _payload(plan)
            operation = _operation_contract(value)
            self._single_batch(value, request.batch_id)
            execution = value.get("execution")
            plan_acs_checkpoint_slots = (
                execution.get("acs_checkpoint_slots", 0)
                if isinstance(execution, Mapping)
                else 0
            )
            if (
                type(plan_acs_checkpoint_slots) is not int
                or plan_acs_checkpoint_slots not in {0, 1, 2}
                or request.acs_checkpoint_slots != plan_acs_checkpoint_slots
                or request.native_streamed_checkpoint is not True
                or request.acs_capture_instrumentation
                is not bool(plan_acs_checkpoint_slots)
            ):
                raise CampaignProviderError(
                    "Workhorse ACS checkpoint route differs from the compiled plan"
                )
            if operation["operation"] == EXECUTION_ROW_OPERATION_V2:
                contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                    operation["contract"]
                )
                checkpoint_ticks = _execution_row_checkpoint_ticks_v2(
                    value,
                    operation=operation,
                    contract=contract,
                )
                if (
                    tuple(request.checkpoint_ticks) != checkpoint_ticks
                    or len(checkpoint_ticks) > max(1, plan_acs_checkpoint_slots)
                ):
                    raise CampaignProviderError(
                        "Workhorse row request changed its exact checkpoint schedule"
                    )
                _validate_science_founder_default_controls_v1(
                    value,
                    checkpoint_ticks=checkpoint_ticks,
                    acs_checkpoint_slots=plan_acs_checkpoint_slots,
                )
                _exact_founder_runtime_world_identity_v1(value)
            if plan_acs_checkpoint_slots and operation["operation"] not in {
                EXECUTION_ROW_OPERATION_V2,
                SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
                INDEXED_ACS_CONTINUATION_OPERATION_V2,
            }:
                raise CampaignProviderError(
                    "Workhorse operation has no physical ACS checkpoint seam"
                )
            artifact = workhorse_artifact_binding_v2(operation["artifact"]["path"])
            shape_id = (
                operation["contract"]["shape_id"]
                if operation["operation"]
                in {
                    EXECUTION_ROW_OPERATION_V2,
                    SEMANTIC_EVIDENCE_CASE_OPERATION_V2,
                    SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
                    INDEXED_ACS_CONTINUATION_OPERATION_V2,
                }
                else operation["shape_id"]
            )
            backend_id = (
                operation["contract"]["backend_id"]
                if operation["operation"]
                in {
                    EXECUTION_ROW_OPERATION_V2,
                    SEMANTIC_EVIDENCE_CASE_OPERATION_V2,
                    SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
                    INDEXED_ACS_CONTINUATION_OPERATION_V2,
                }
                else operation["backend_id"]
            )
            if (
                artifact["sha256"] != operation["artifact"]["sha256"]
                or artifact["sealed_artifact_identity_sha256"]
                != operation["artifact"]["sealed_artifact_identity_sha256"]
                or artifact["identity"]["shape_id"] != shape_id
                or artifact["identity"]["backend_id"] != backend_id
            ):
                raise CampaignProviderError(
                    "Workhorse artifact differs from exact operation"
                )
            if operation["operation"] in {
                EXECUTION_ROW_OPERATION_V2,
                SEMANTIC_EVIDENCE_CASE_OPERATION_V2,
                SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
                INDEXED_ACS_CONTINUATION_OPERATION_V2,
            }:
                contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                    operation["contract"]
                )
                envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
                    operation["envelope"]["path"]
                )
                if (
                    envelope["receipt_file_sha256"]
                    != operation["envelope"]["file_sha256"]
                    or envelope["receipt_content_sha256"]
                    != operation["envelope"]["content_sha256"]
                    or envelope["artifact"] != artifact
                    or envelope["shape_id"] != contract.shape_id
                    or envelope["backend_id"] != contract.backend_id
                ):
                    raise CampaignProviderError(
                        "Workhorse envelope differs from exact execution"
                    )
                row = next(
                    (
                        item
                        for item in envelope["expected_rows"]
                        if item["B"] == contract.B
                        and item["temporal_k"] == contract.temporal_k
                    ),
                    None,
                )
                if row is None or row["planned_admission"].get("admitted") is not True:
                    raise CampaignProviderError(
                        "Workhorse operation is not envelope-admitted"
                    )
                if (
                    operation["operation"] == SEMANTIC_EVIDENCE_CASE_OPERATION_V2
                    and operation["external_source_checkpoint_binding"] is not None
                ):
                    if self.operation_adapter is None:
                        raise CampaignProviderError(
                            "canonical Workhorse Scheduler adapter is not injected"
                        )
                    validated_source = self.operation_adapter.validate_workhorse_external_checkpoint_source_v2(
                        operation["external_source_checkpoint_binding"],
                        expected_identity_sha256=operation[
                            "source_checkpoint_identity_sha256"
                        ],
                    )
                    if (
                        not isinstance(validated_source, Mapping)
                        or dict(validated_source)
                        != operation["external_source_checkpoint_binding"]
                    ):
                        raise CampaignProviderError(
                            "canonical external checkpoint source projection changed"
                        )
            elif (
                operation["device_identity"]["total_memory_bytes"]
                <= operation["reserve_policy"]["reserve_bytes_requested"]
            ):
                raise CampaignProviderError(
                    "calibration reserve consumes the declared device"
                )
            if self.operation_adapter is None:
                raise CampaignProviderError(
                    "canonical Workhorse Scheduler adapter is not injected"
                )
            if (
                request.execution_route_sha256
                != self.execution_route_contract(value).execution_route_sha256
            ):
                raise CampaignProviderError("Workhorse route digest differs")
        except (
            CampaignProviderError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return (
                UnsupportedReasonV1(
                    "workhorse_v2_single_operation_invalid",
                    "execution",
                    str(exc),
                    self.profile_id,
                ),
            )
        return ()

    def execution_route_contract(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1 | None = None,
    ) -> ExecutionRouteContractV1:
        value = _payload(plan)
        operation = _operation_contract(value)
        if operation["operation"] in {
            EXECUTION_ROW_OPERATION_V2,
            INDEXED_ACS_CONTINUATION_OPERATION_V2,
        }:
            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                operation["contract"]
            )
            if contract.shape_id != self.shape.shape_id:
                raise CampaignProviderError("Workhorse provider shape differs")
            envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
                operation["envelope"]["path"]
            )
            shape_id = contract.shape_id
            backend_id = contract.backend_id
            start = contract.start_next_tick
            target = start + contract.total_ticks
            operation_identity: dict[str, Any] = {
                "contract": contract.to_dict(),
                "compiled_temporal_k": contract.temporal_k,
                "reserve_policy": None,
                "absolute_tick_span": {
                    "time_authority": "absolute_tick_v1",
                    "start_next_tick": start,
                    "target_next_tick": target,
                    "total_ticks": contract.total_ticks,
                },
                "offline_matrix_aggregation_only": (
                    operation["operation"] == EXECUTION_ROW_OPERATION_V2
                ),
            }
            if operation["operation"] == INDEXED_ACS_CONTINUATION_OPERATION_V2:
                continuation = operation["continuation_contract"]
                operation_identity.update(
                    {
                        "compiled_temporal_k": list(
                            continuation["dispatch_plan"]["admitted_temporal_k"]
                        ),
                        "indexed_continuation_contract": dict(continuation),
                        "science_continuation_identity": dict(
                            continuation["science_continuation_identity"]
                        ),
                        "physical_continuation_plan": dict(
                            continuation["physical_continuation_plan"]
                        ),
                    }
                )
            selection = {
                "device_identity": envelope["device_identity"],
                "B": contract.B,
                "temporal_k": contract.temporal_k,
                "compiled_temporal_k": None,
                "execution_mode": contract.backend_id,
            }
            if operation["operation"] == INDEXED_ACS_CONTINUATION_OPERATION_V2:
                selection["compiled_temporal_k"] = list(
                    operation["continuation_contract"]["dispatch_plan"][
                        "admitted_temporal_k"
                    ]
                )
            checkpoint_identity = {
                "source_checkpoint_identity_sha256": operation[
                    "source_checkpoint_identity_sha256"
                ],
                "science_continuation_contract_sha256": operation[
                    "science_continuation_contract_sha256"
                ],
            }
            if operation["operation"] == INDEXED_ACS_CONTINUATION_OPERATION_V2:
                checkpoint_identity.update(
                    {
                        "source_world_state_set_sha256": operation[
                            "source_checkpoint_identity_sha256"
                        ],
                        "checkpoint_ticks": list(
                            operation["continuation_contract"]["checkpoint_ticks"]
                        ),
                        "repack_binding_sha256": operation[
                            "continuation_contract"
                        ]["repack_binding"]["repack_binding_sha256"],
                        "dispatch_plan_sha256": operation[
                            "continuation_contract"
                        ]["dispatch_plan"]["dispatch_plan_sha256"],
                    }
                )
        elif operation["operation"] == SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1:
            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                operation["contract"]
            )
            if contract.shape_id != self.shape.shape_id:
                raise CampaignProviderError(
                    "Workhorse provider science-demo shape differs"
                )
            envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
                operation["envelope"]["path"]
            )
            shape_id = contract.shape_id
            backend_id = contract.backend_id
            start = contract.start_next_tick
            target = start + contract.total_ticks
            operation_identity = {
                "contract": contract.to_dict(),
                "science_demo_contract": dict(operation["science_demo_contract"]),
                "semantic_inheritance_binding": dict(
                    operation["semantic_inheritance_binding"]
                ),
                "compiled_temporal_k": contract.temporal_k,
                "reserve_policy": None,
                "absolute_tick_span": {
                    "time_authority": "absolute_tick_v1",
                    "start_next_tick": start,
                    "target_next_tick": target,
                    "total_ticks": contract.total_ticks,
                },
                "offline_matrix_aggregation_only": False,
            }
            selection = {
                "device_identity": envelope["device_identity"],
                "B": contract.B,
                "temporal_k": contract.temporal_k,
                "compiled_temporal_k": None,
                "execution_mode": contract.backend_id,
            }
            checkpoint_identity = {
                "source_checkpoint_identity_sha256": None,
                "science_continuation_contract_sha256": operation[
                    "science_continuation_contract_sha256"
                ],
                "science_checkpoint_compatibility_binding": dict(
                    operation["science_checkpoint_compatibility_binding"]
                ),
                "checkpoint_ticks": list(
                    operation["science_demo_contract"]["checkpoint_ticks"]
                ),
                "continuation_restore_ticks": list(
                    operation["science_demo_contract"]["continuation_restore_ticks"]
                ),
            }
        elif operation["operation"] == SEMANTIC_EVIDENCE_CASE_OPERATION_V2:
            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                operation["contract"]
            )
            if contract.shape_id != self.shape.shape_id:
                raise CampaignProviderError(
                    "Workhorse provider semantic-case shape differs"
                )
            envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
                operation["envelope"]["path"]
            )
            semantic_case = operation["semantic_case_contract"]
            shape_id = contract.shape_id
            backend_id = contract.backend_id
            start = contract.start_next_tick
            target = start + contract.total_ticks
            operation_identity = {
                "contract": contract.to_dict(),
                "semantic_case_contract": dict(semantic_case),
                "semantic_case_id": semantic_case["case_law"]["case_id"],
                "semantic_temporal_k": semantic_case["semantic_temporal_k"],
                "artifact_role": semantic_case["case_law"]["artifact_role"],
                "attempt_terminal_contract": dict(
                    semantic_case["attempt_terminal_contract"]
                ),
                "compiled_temporal_k": contract.temporal_k,
                "launchpad_operation_contract_sha256": operation[
                    "launchpad_operation_contract_sha256"
                ],
                "reserve_policy": None,
                "absolute_tick_span": {
                    "time_authority": "absolute_tick_v1",
                    "start_next_tick": start,
                    "target_next_tick": target,
                    "total_ticks": contract.total_ticks,
                },
                "offline_matrix_aggregation_only": False,
            }
            selection = {
                "device_identity": envelope["device_identity"],
                "B": contract.B,
                "temporal_k": contract.temporal_k,
                "compiled_temporal_k": None,
                "semantic_temporal_k": semantic_case["semantic_temporal_k"],
                "execution_mode": contract.backend_id,
            }
            checkpoint_identity = {
                "source_checkpoint_identity_sha256": operation[
                    "source_checkpoint_identity_sha256"
                ],
                "science_continuation_contract_sha256": operation[
                    "science_continuation_contract_sha256"
                ],
                "science_checkpoint_compatibility_binding": dict(
                    operation["science_checkpoint_compatibility_binding"]
                ),
                "external_source_checkpoint_binding": (
                    None
                    if operation["external_source_checkpoint_binding"] is None
                    else dict(operation["external_source_checkpoint_binding"])
                ),
            }
        else:
            if operation["shape_id"] != self.shape.shape_id:
                raise CampaignProviderError(
                    "Workhorse provider calibration shape differs"
                )
            shape_id = operation["shape_id"]
            backend_id = operation["backend_id"]
            span = operation["absolute_tick_span"]
            start = span["start_next_tick"]
            target = span["target_next_tick"]
            operation_identity = {
                "contract": None,
                "compiled_temporal_k": list(operation["compiled_temporal_k"]),
                "reserve_policy": dict(operation["reserve_policy"]),
                "absolute_tick_span": dict(span),
                "offline_matrix_aggregation_only": False,
            }
            selection = {
                "device_identity": dict(operation["device_identity"]),
                "B": 1,
                "temporal_k": None,
                "compiled_temporal_k": list(operation["compiled_temporal_k"]),
                "execution_mode": operation["backend_id"],
            }
            checkpoint_identity = {
                "source_checkpoint_identity_sha256": None,
                "science_continuation_contract_sha256": None,
            }
        execution = value.get("execution")
        acs_checkpoint_slots = (
            execution.get("acs_checkpoint_slots", 0)
            if isinstance(execution, Mapping)
            else 0
        )
        if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {0, 1, 2}:
            raise CampaignProviderError(
                "Workhorse route ACS checkpoint slots must be zero, one, or two"
            )
        native_acs_runtime_component = _acs_native_runtime_route_component_v1(
            value,
            required=bool(acs_checkpoint_slots),
        )
        route = ExecutionRouteContractV1(
            schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            components=(
                {
                    "component_id": "absolute_tick_time_contract",
                    "time_authority": "absolute_tick_v1",
                    "start_next_tick": start,
                    "target_next_tick": target,
                },
                {
                    "component_id": "cuda_full_evolution_workhorse_v2_operation",
                    "operation": operation["operation"],
                    "operation_contract_sha256": operation["operation_contract_sha256"],
                    "shape_id": shape_id,
                    "backend_id": backend_id,
                    "route_id": self.shape.route_id,
                    **operation_identity,
                    "one_operation_per_attempt": True,
                },
                {
                    "component_id": "cuda_full_evolution_workhorse_v2_artifact_unit",
                    "artifact_path": operation["artifact"]["path"],
                    "artifact_sha256": operation["artifact"]["sha256"],
                    "sealed_artifact_identity_sha256": operation["artifact"][
                        "sealed_artifact_identity_sha256"
                    ],
                },
                {
                    "component_id": "cuda_full_evolution_workhorse_v2_device_selection",
                    **selection,
                },
                {
                    "component_id": "cuda_full_evolution_workhorse_v2_checkpoint_science",
                    "checkpoint_schema": (
                        "eiel_full_evolution_cuda_workhorse_checkpoint_v2"
                    ),
                    **checkpoint_identity,
                    "observation_mode": "OFF",
                    "science_continuation_claimed": False,
                    "checkpoint_occurrence_authority": (
                        "outer_canonical_campaign_provider"
                    ),
                    "terminal_next_tick": target,
                },
                *(
                    (
                        {
                            "component_id": "cuda_full_evolution_workhorse_v2_acs2_checkpoint_transport",
                            "acs_checkpoint_slots": acs_checkpoint_slots,
                            "capture_mode": "direct_resident_cuda_state_v1",
                            "worker_boundary": "path_and_scalar_metadata_only",
                            "worker_accepts_cuda_objects": False,
                            "worker_accepts_callbacks": False,
                            "worker_full_state_bytes_materialized": 0,
                            "capture_pool": "one_shared_two_slab_pool_not_per_slot",
                            "pinned_slab_count": 2,
                            "producer_full_state_host_bytes_materialized": 0,
                            "producer_compacted_cuda_bytes_materialized": 0,
                            "publication": "strict_noreplace_post_fsync_readback_v1",
                        },
                    )
                    if acs_checkpoint_slots
                    else ()
                ),
                *(
                    (native_acs_runtime_component,)
                    if native_acs_runtime_component is not None
                    else ()
                ),
            ),
            required_obligations=(
                "one_exact_scheduler_attempt",
                "live_exclusive_device_heartbeat_before_cuda",
                "absolute_tick_terminal_frontiers",
                "scheduler_reopened_terminal_before_final_receipt",
                *(
                    (
                        "async_checkpoint_backpressure",
                        "async_checkpoint_failure_latch",
                        "acs2_host_only_capture",
                    )
                    if acs_checkpoint_slots
                    else ()
                ),
            ),
        )
        if request is not None and request.execution_route_sha256 not in {
            None,
            route.execution_route_sha256,
        }:
            raise CampaignProviderError("Workhorse bound route differs")
        return route

    def estimate_resources(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        requests: Sequence[ResidentBatchRequestV1],
    ) -> dict[str, Any]:
        if len(requests) != 1:
            raise CampaignProviderError(
                "Workhorse provider accepts exactly one request"
            )
        acs_checkpoint_slots = requests[0].acs_checkpoint_slots
        if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {0, 1, 2}:
            raise CampaignProviderError(
                "Workhorse ACS checkpoint slots must be zero, one, or two"
            )
        operation = _operation_contract(plan)
        if operation["operation"] == DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2:
            if acs_checkpoint_slots:
                raise CampaignProviderError(
                    "Workhorse device calibration does not support ACS checkpoint slots"
                )
            total_bytes = operation["device_identity"]["total_memory_bytes"]
            reserve = operation["reserve_policy"]["reserve_bytes_requested"]
            minimum = (
                total_bytes * WORKHORSE_MINIMUM_HEADROOM_NUMERATOR_V2
                + WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2
                - 1
            ) // WORKHORSE_MINIMUM_HEADROOM_DENOMINATOR_V2
            return {
                "resource_estimator_id": (
                    "workhorse_v2_device_envelope_calibration_v1"
                ),
                "vram_bytes_upper_bound": max(0, total_bytes - max(reserve, minimum)),
                "wall_seconds_upper_bound": max(
                    300, 120 * len(operation["compiled_temporal_k"])
                ),
                "disk_bytes_upper_bound": 128 * 1024**2,
                "physical_batch_cells": 1,
                "workhorse_calibration_owner_B": 1,
                "workhorse_compiled_temporal_k": list(operation["compiled_temporal_k"]),
                "exclusive_device_required": True,
                "acs_checkpoint_slots": 0,
            }
        contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            operation["contract"]
        )
        envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
            operation["envelope"]["path"]
        )
        row = next(
            item
            for item in envelope["expected_rows"]
            if item["B"] == contract.B and item["temporal_k"] == contract.temporal_k
        )
        admission = row["planned_admission"]
        if operation["operation"] == EXECUTION_ROW_OPERATION_V2:
            schedule = operation["measurement_schedule"]
            dispatch_count = (
                schedule["warmup_dispatches"] + schedule["measured_dispatches"]
            )
        elif operation["operation"] == INDEXED_ACS_CONTINUATION_OPERATION_V2:
            dispatch_count = operation["continuation_contract"]["dispatch_plan"][
                "dispatch_count"
            ]
        else:
            dispatch_count = (
                contract.total_ticks + contract.temporal_k - 1
            ) // contract.temporal_k
        storage = workhorse_checkpoint_storage_accounting_v2(
            contract=contract,
            checkpoint_ticks=requests[0].checkpoint_ticks,
            acs_checkpoint_slots=acs_checkpoint_slots,
            dispatch_count=dispatch_count,
        )
        return {
            "resource_estimator_id": (
                "workhorse_v2_exact_envelope_semantic_case_v1"
                if operation["operation"] == SEMANTIC_EVIDENCE_CASE_OPERATION_V2
                else (
                    "workhorse_v2_science_compatibility_demo_v1"
                    if operation["operation"] == SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1
                    else (
                        "workhorse_v2_indexed_acs_continuation_v2"
                        if operation["operation"]
                        == INDEXED_ACS_CONTINUATION_OPERATION_V2
                        else "workhorse_v2_exact_envelope_row_v1"
                    )
                )
            ),
            "vram_bytes_upper_bound": admission["required_bytes"],
            "wall_seconds_upper_bound": max(300, contract.total_ticks * 60),
            **storage,
            "physical_batch_cells": 1,
            "workhorse_world_batch_B": contract.B,
            "workhorse_temporal_k": contract.temporal_k,
            "acs_checkpoint_slots": acs_checkpoint_slots,
            "checkpoint_capture_mode": (
                "direct_resident_cuda_state_v1"
                if acs_checkpoint_slots
                else "synchronous_v1"
            ),
            "acs_capture_pool_count": 1 if acs_checkpoint_slots else 0,
            "acs_pinned_slab_count": 2 if acs_checkpoint_slots else 0,
            "acs_pinned_host_bytes_upper_bound": (
                2 * ACS2_CAPTURE_CHUNK_BYTES_V1 if acs_checkpoint_slots else 0
            ),
            "acs_producer_full_state_host_bytes": 0,
            "acs_producer_compacted_cuda_bytes": 0,
            "acs_worker_full_state_bytes": 0,
            "acs_worker_cuda_references": 0,
        }

    def crca_source_identity_v1(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        launchpad_source_identity: Mapping[str, Any],
        preflight: Mapping[str, Any],
    ) -> dict[str, Any]:
        operation = _operation_contract(plan)
        git = launchpad_source_identity.get("git")
        if not isinstance(git, Mapping):
            raise CampaignProviderError("Workhorse CRCA source identity lacks Git")
        return {
            "source_commit": git.get("commit"),
            "source_tree_identity": dict(launchpad_source_identity),
            "dirty_patch_sha256": (
                None
                if git.get("working_tree_clean") is True
                else git.get("working_tree_status_sha256")
            ),
            "toolchain": {
                "artifact_sha256": operation["artifact"]["sha256"],
                "sealed_artifact_identity_sha256": operation["artifact"][
                    "sealed_artifact_identity_sha256"
                ],
            },
            "environment": {
                "preflight_runtime_environment": dict(
                    preflight.get("runtime_environment") or {}
                )
            },
            "device_conformance_envelope": (
                validate_workhorse_device_envelope_calibration_receipt_v2(
                    operation["envelope"]["path"]
                )["device_identity"]
                if operation["operation"]
                in {
                    EXECUTION_ROW_OPERATION_V2,
                    INDEXED_ACS_CONTINUATION_OPERATION_V2,
                    SEMANTIC_EVIDENCE_CASE_OPERATION_V2,
                    SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
                }
                else dict(operation["device_identity"])
            ),
        }

    def crca_assurance_v1(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        preflight: Mapping[str, Any],
    ) -> dict[str, Any]:
        operation = _operation_contract(plan)
        return {
            "intended_claim_id": {
                EXECUTION_ROW_OPERATION_V2: "TECHNICAL_ONLY_WORKHORSE_V2_SINGLE_ROW",
                DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2: (
                    "TECHNICAL_ONLY_WORKHORSE_V2_DEVICE_ENVELOPE_CALIBRATION"
                ),
                SEMANTIC_EVIDENCE_CASE_OPERATION_V2: (
                    "TECHNICAL_ONLY_WORKHORSE_V2_SEMANTIC_EVIDENCE_CASE"
                ),
                SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1: (
                    "TECHNICAL_ONLY_WORKHORSE_V2_SCIENCE_COMPATIBILITY_DEMO"
                ),
                INDEXED_ACS_CONTINUATION_OPERATION_V2: (
                    "TECHNICAL_ONLY_WORKHORSE_V2_INDEXED_ACS_CONTINUATION"
                ),
            }[operation["operation"]],
            "qualified_envelope": None,
            "obligations": {
                "preflight": {
                    "ready_for_execution": preflight.get("ready_for_execution")
                },
                "execution_route": self.execution_route_contract(plan).payload(),
            },
            "authority_references": {},
            "review_references": {},
            "initial_holds": {
                "formal_qualification": {
                    "status": "HOLD_PREQUAL_ONLY",
                    "reason": "new Workhorse route has no qualification inheritance",
                }
            },
        }

    def execute_batch(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
        context: Any,
    ) -> dict[str, Any]:
        """Execute with the exact plan-bound native ACS runtime, when selected."""

        value = _payload(plan)
        binding = _acs_native_runtime_artifact_binding_v1(
            value,
            required=bool(request.acs_checkpoint_slots),
        )
        prior_extension = os.environ.get(
            ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1
        )
        if binding is not None:
            os.environ[ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1] = str(
                binding["extension_path"]
            )
        try:
            return self._execute_batch_with_bound_acs_runtime(
                value,
                request,
                context,
            )
        finally:
            if binding is not None:
                if prior_extension is None:
                    os.environ.pop(
                        ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1,
                        None,
                    )
                else:
                    os.environ[ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1] = (
                        prior_extension
                    )

    def _execute_batch_with_bound_acs_runtime(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
        context: Any,
    ) -> dict[str, Any]:
        if self.operation_adapter is None:
            raise CampaignProviderError("canonical Workhorse adapter is unavailable")
        value = _payload(plan)
        operation = _operation_contract(value)
        if request.batch_id != self._single_batch(value)["batch_id"]:
            raise CampaignProviderError("Workhorse execution request changed batch")
        # Use the existing campaign-owned facade so this provider does not
        # create another campaigns -> QRR architectural edge.
        from .preflight import build_execution_request_v1

        bound_request = build_execution_request_v1(
            self, value, request.batch_id
        ).payload()
        parent_path = Path(context.batch_root) / "workhorse-parent-context.json"
        parent = dict(
            self.operation_adapter.bind_workhorse_parent_context_v2(
                attempt_authority_path=Path(context.run_root) / ATTEMPT_AUTHORITY_NAME,
                target=parent_path,
                campaign_plan=value,
                bound_execution_request=bound_request,
            )
        )
        context.record_artifact(
            parent_path,
            artifact_family="workhorse_parent_context_v2",
            retention_class="R3",
        )
        if operation["operation"] == DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2:
            calibration_path = (
                Path(context.batch_root)
                / "workhorse-device-envelope-calibration-measurement.json"
            )
            calibration = run_and_seal_workhorse_device_envelope_calibration_v2(
                operation["artifact"]["path"],
                output_path=calibration_path,
                source_identity={
                    "schema_version": ("eiel_workhorse_calibration_source_identity_v2"),
                    "campaign_plan_sha256": value["plan_sha256"],
                    "operation_contract_sha256": operation["operation_contract_sha256"],
                    "artifact_sha256": operation["artifact"]["sha256"],
                    "sealed_artifact_identity_sha256": operation["artifact"][
                        "sealed_artifact_identity_sha256"
                    ],
                },
                canonical_parent_authority_binding=parent,
                reserve_bytes=operation["reserve_policy"]["reserve_bytes_requested"],
            )
            if (
                calibration.get("shape_id") != operation["shape_id"]
                or calibration.get("backend_id") != operation["backend_id"]
                or calibration.get("device_identity") != operation["device_identity"]
                or sorted(
                    int(value) for value in calibration.get("owner_components_by_k", {})
                )
                != operation["compiled_temporal_k"]
                or calibration.get("capacity_basis", {}).get("reserve_bytes_requested")
                != operation["reserve_policy"]["reserve_bytes_requested"]
            ):
                raise CampaignProviderError(
                    "physical calibration result differs from bound operation"
                )
            context.record_artifact(
                calibration_path,
                artifact_family=(
                    "workhorse_device_envelope_calibration_measurement_v2"
                ),
                retention_class="R3",
            )
            span = operation["absolute_tick_span"]
            target = span["target_next_tick"]
            checkpoint = {
                "schema_version": "eiel_workhorse_operation_checkpoint_v2",
                "time_authority": "absolute_tick_v1",
                "operation": operation["operation"],
                "next_absolute_tick": target,
                "source_checkpoint_identity_sha256": None,
                "science_continuation_contract_sha256": None,
                "observation_mode": "OFF",
                "science_continuation_claimed": False,
                "terminal_cell_digests": {
                    request.cell_ids[0]: calibration["receipt_content_sha256"]
                },
            }
            return {
                "status": "completed",
                "checkpoint": checkpoint,
                "metrics_rows": [
                    {
                        "cell_id": request.cell_ids[0],
                        "metrics": {
                            "workhorse.device_envelope_calibration_completed": 1
                        },
                    }
                ],
                "result": {
                    "schema_version": (
                        "eiel_workhorse_device_envelope_calibration_provider_result_v2"
                    ),
                    "operation": operation["operation"],
                    "shape_id": operation["shape_id"],
                    "backend_id": operation["backend_id"],
                    "artifact_sha256": operation["artifact"]["sha256"],
                    "sealed_artifact_identity_sha256": operation["artifact"][
                        "sealed_artifact_identity_sha256"
                    ],
                    "compiled_temporal_k": list(operation["compiled_temporal_k"]),
                    "device_identity": dict(operation["device_identity"]),
                    "reserve_policy": dict(operation["reserve_policy"]),
                    "calibration_receipt_path": str(calibration_path.resolve()),
                    "calibration_receipt_file_sha256": calibration[
                        "receipt_file_sha256"
                    ],
                    "calibration_receipt_content_sha256": calibration[
                        "receipt_content_sha256"
                    ],
                    "absolute_frontiers": {
                        "start_next_tick": span["start_next_tick"],
                        "target_next_tick": target,
                        "committed_next_tick": target,
                    },
                },
            }
        if operation["operation"] == SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1:
            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                operation["contract"]
            )
            batch_root = Path(context.batch_root)
            result_path = batch_root / "workhorse-science-demo-result.json"
            event_sink_root = batch_root / "workhorse-science-event-sinks"
            checkpoint_root = batch_root / "workhorse-science-checkpoints"
            boundary_root = batch_root / "workhorse-science-boundaries"
            restore_root = batch_root / "workhorse-science-restores"
            acs_coordinator: Any | None = None
            checkpoint_commit_callback: Any | None = None
            if request.acs_checkpoint_slots:
                from .cuda_full_evolution_v2_acs_adapter import (
                    CudaFullEvolutionV2WorkhorseACSCoordinatorV1,
                )

                acs_coordinator = CudaFullEvolutionV2WorkhorseACSCoordinatorV1(
                    slots=request.acs_checkpoint_slots,
                    checkpoint_root=checkpoint_root,
                    artifact_path=operation["artifact"]["path"],
                    attempt_authority_binding=parent,
                    require_native_pool=self._acs_require_native_pool,
                )

                def checkpoint_commit_callback(
                    token: Any,
                    checkpoint_manifest_path: Path,
                    occurrence_projection: Mapping[str, Any],
                ) -> None:
                    _commit_workhorse_science_acs_checkpoint_v1(
                        context=context,
                        coordinator=acs_coordinator,
                        token=token,
                        checkpoint_manifest_path=checkpoint_manifest_path,
                        occurrence_projection=occurrence_projection,
                        event_sink_root=event_sink_root,
                        checkpoint_root=checkpoint_root,
                        boundary_root=boundary_root,
                    )
            try:
                result = run_and_seal_workhorse_science_compatibility_demo_v1(
                    operation["artifact"]["path"],
                    envelope_receipt_path=operation["envelope"]["path"],
                    semantic_inheritance_binding=operation["semantic_inheritance_binding"],
                    science_checkpoint_compatibility_binding=operation[
                        "science_checkpoint_compatibility_binding"
                    ],
                    demo_contract=operation["science_demo_contract"],
                    canonical_parent_authority_binding=parent,
                    operation_contract_binding=operation,
                    checkpoint_occurrence_adapter=self.operation_adapter,
                    event_sink_root=event_sink_root,
                    checkpoint_root=checkpoint_root,
                    boundary_record_root=boundary_root,
                    restore_receipt_root=restore_root,
                    result_path=result_path,
                    checkpoint_transport=acs_coordinator,
                    checkpoint_commit_callback=checkpoint_commit_callback,
                )
                if acs_coordinator is not None:
                    acs_coordinator.close()
            except BaseException:
                if acs_coordinator is not None:
                    acs_coordinator.abort()
                raise
            acs_staging = (
                acs_coordinator.telemetry()
                if acs_coordinator is not None
                else None
            )
            context.record_artifact(
                result_path,
                artifact_family="workhorse_science_demo_result_v1",
                retention_class="R3",
            )
            for root, family, retention_class in (
                (
                    event_sink_root,
                    "workhorse_science_demo_event_ack_v1",
                    "R3",
                ),
                (
                    checkpoint_root,
                    "workhorse_science_demo_checkpoint_v1",
                    "R3",
                ),
                (
                    boundary_root,
                    "workhorse_science_demo_boundary_record_v1",
                    "R3",
                ),
                (
                    restore_root,
                    "workhorse_science_demo_restore_receipt_v1",
                    "R3",
                ),
            ):
                for artifact_path in sorted(root.rglob("*")):
                    if artifact_path.is_symlink():
                        raise CampaignProviderError(
                            "science demo evidence contains a symbolic link"
                        )
                    if artifact_path.is_file():
                        context.record_artifact(
                            artifact_path,
                            artifact_family=family,
                            retention_class=retention_class,
                        )
            target = contract.start_next_tick + contract.total_ticks
            checkpoint = {
                "schema_version": "eiel_workhorse_operation_checkpoint_v2",
                "time_authority": "absolute_tick_v1",
                "operation": operation["operation"],
                "next_absolute_tick": target,
                "checkpoint_ticks": list(
                    operation["science_demo_contract"]["checkpoint_ticks"]
                ),
                "source_checkpoint_identity_sha256": None,
                "science_continuation_contract_sha256": operation[
                    "science_continuation_contract_sha256"
                ],
                "science_checkpoint_sidecars": result["checkpoint_sidecars"],
                "observation_mode": "OFF",
                "science_continuation_claimed": False,
                "terminal_cell_digests": {
                    request.cell_ids[0]: result["result_content_sha256"]
                },
                **(
                    {"acs_checkpoint_staging": acs_staging}
                    if acs_staging is not None
                    else {}
                ),
            }
            return {
                "status": "completed",
                "checkpoint": checkpoint,
                "metrics_rows": [
                    {
                        "cell_id": request.cell_ids[0],
                        "metrics": {
                            "workhorse.science_compatibility_demo_completed": 1
                        },
                    }
                ],
                "result": {
                    "schema_version": (
                        "eiel_workhorse_science_demo_provider_result_v1"
                    ),
                    "operation": operation["operation"],
                    "shape_id": contract.shape_id,
                    "backend_id": contract.backend_id,
                    "B": contract.B,
                    "temporal_k": contract.temporal_k,
                    "artifact_sha256": operation["artifact"]["sha256"],
                    "sealed_artifact_identity_sha256": operation["artifact"][
                        "sealed_artifact_identity_sha256"
                    ],
                    "science_demo_result_path": result["result_path"],
                    "science_demo_result_file_sha256": result["result_file_sha256"],
                    "science_demo_result_content_sha256": result[
                        "result_content_sha256"
                    ],
                    "absolute_frontiers": result["absolute_frontiers"],
                    "boundary_next_ticks": list(
                        operation["science_demo_contract"]["boundary_next_ticks"]
                    ),
                    "checkpoint_ticks": list(
                        operation["science_demo_contract"]["checkpoint_ticks"]
                    ),
                    "continuation_restore_ticks": list(
                        operation["science_demo_contract"]["continuation_restore_ticks"]
                    ),
                    "terminal_restore_validation_tick": operation[
                        "science_demo_contract"
                    ]["terminal_restore_validation_tick"],
                    "result_class": result["result_class"],
                    **(
                        {"acs_checkpoint_staging": acs_staging}
                        if acs_staging is not None
                        else {}
                    ),
                },
            }
        if operation["operation"] == INDEXED_ACS_CONTINUATION_OPERATION_V2:
            if request.acs_checkpoint_slots != 2:
                raise CampaignProviderError(
                    "indexed continuation execution requires the ACS2 two-slot pipeline"
                )
            from .cuda_full_evolution_v2_acs_adapter import (
                CudaFullEvolutionV2WorkhorseACSCoordinatorV1,
            )

            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                operation["contract"]
            )
            batch_root = Path(context.batch_root)
            result_path = batch_root / "workhorse-indexed-continuation-result.json"
            event_sink_root = batch_root / "workhorse-indexed-continuation-event-sinks"
            coordinator = CudaFullEvolutionV2WorkhorseACSCoordinatorV1(
                slots=2,
                checkpoint_root=(
                    batch_root / "workhorse-indexed-continuation-checkpoints"
                ),
                artifact_path=operation["artifact"]["path"],
                attempt_authority_binding=parent,
                require_native_pool=self._acs_require_native_pool,
            )
            committed_checkpoints: list[dict[str, Any]] = []

            def commit_fifo_head() -> dict[str, Any]:
                pending = coordinator.pending_tokens()
                if not pending:
                    raise CampaignProviderError(
                        "indexed continuation ACS2 FIFO unexpectedly became empty"
                    )
                source_occurrence = operation["continuation_contract"][
                    "science_continuation_identity"
                ].get("source_checkpoint_occurrence_sha256")
                occurrence_parent = (
                    committed_checkpoints[-1]["checkpoint_occurrence_projection"][
                        "occurrence_digest"
                    ]
                    if committed_checkpoints
                    else (
                        None
                        if source_occurrence is None
                        else {"algorithm": "sha256", "hex": source_occurrence}
                    )
                )
                committed = _commit_workhorse_row_acs_checkpoint_v2(
                    context=context,
                    coordinator=coordinator,
                    token=pending[0],
                    operation_adapter=self.operation_adapter,
                    canonical_parent_authority_binding=parent,
                    contract=contract,
                    expected_checkpoint_tick=pending[0].compute_frontier,
                    parent_checkpoint_digest=occurrence_parent,
                    expected_operation=INDEXED_ACS_CONTINUATION_OPERATION_V2,
                )
                committed_checkpoints.append(committed)
                return committed

            def checkpoint_boundary_handler(
                *,
                state: Any,
                contract: FullEvolutionCudaWorkhorseContractV2,
                checkpoint_at_tick: int,
                event_sink_receipt_path: str | Path,
            ) -> None:
                while len(coordinator.pending_tokens()) >= 2:
                    commit_fifo_head()
                coordinator.capture_workhorse_checkpoint(
                    state,
                    contract=contract,
                    checkpoint_at_tick=checkpoint_at_tick,
                    event_sink_receipt_path=event_sink_receipt_path,
                )

            try:
                result = run_workhorse_indexed_continuation_v2(
                    operation["artifact"]["path"],
                    envelope_receipt_path=operation["envelope"]["path"],
                    canonical_parent_authority_binding=parent,
                    authoritative_micro_receipt_path=operation[
                        "authoritative_micro_receipt"
                    ]["path"],
                    semantic_gate_receipt_path=operation["semantic_gate_receipt"][
                        "path"
                    ],
                    continuation_contract=operation["continuation_contract"],
                    event_sink_root=event_sink_root,
                    execution_result_path=result_path,
                    checkpoint_boundary_handler=checkpoint_boundary_handler,
                )
                if result.get("outcome") == "RUNTIME_HEADROOM_REJECTED":
                    if coordinator.pending_tokens():
                        raise CampaignProviderError(
                            "headroom-rejected continuation captured a checkpoint"
                        )
                else:
                    while coordinator.pending_tokens():
                        commit_fifo_head()
                coordinator.close()
            except BaseException:
                coordinator.abort()
                raise
            if result.get("outcome") == "RUNTIME_HEADROOM_REJECTED":
                raise CampaignProviderError(
                    f"{RUNTIME_HEADROOM_REJECTION_CODE_V2}: indexed continuation "
                    "failed live VRAM admission before fresh-owner restore"
                )
            expected_ticks = operation["continuation_contract"]["checkpoint_ticks"]
            if (
                result["checkpoint_capture_ticks"] != expected_ticks
                or [row["checkpoint_at_tick"] for row in committed_checkpoints]
                != expected_ticks
            ):
                raise CampaignProviderError(
                    "indexed continuation ACS2 commit schedule changed"
                )
            acs_staging = coordinator.telemetry()
            context.record_artifact(
                result_path,
                artifact_family="workhorse_indexed_continuation_result_v2",
                retention_class="R3",
            )
            for artifact_path in sorted(event_sink_root.rglob("*")):
                if artifact_path.is_symlink():
                    raise CampaignProviderError(
                        "indexed continuation event evidence contains a symbolic link"
                    )
                if artifact_path.is_file():
                    context.record_artifact(
                        artifact_path,
                        artifact_family="workhorse_indexed_continuation_event_sink_v2",
                        retention_class="R3",
                    )
            terminal = committed_checkpoints[-1]
            terminal_index = terminal["indexed_checkpoint"]
            target = contract.start_next_tick + contract.total_ticks
            logical_continuation = operation["continuation_contract"][
                "science_continuation_identity"
            ]
            host_boundary_observation: dict[str, Any] | None = None
            if "source_semantic_world_sha256" in logical_continuation:
                if contract.B != 1 or len(request.cell_ids) != 1:
                    raise CampaignProviderError(
                        "selected-world host observation requires exactly one logical cell"
                    )
                occurrence = terminal["checkpoint_occurrence_projection"]
                observation = seal_workhorse_selected_world_observation_v1(
                    Path(context.batch_root)
                    / "workhorse-selected-world-host-observation.json",
                    execution_result_path=result_path,
                    terminal_indexed_checkpoint_manifest_path=terminal_index[
                        "manifest_path"
                    ],
                    source_checkpoint_occurrence_sha256=(
                        _publication_digest_hex(
                            occurrence["occurrence_digest"],
                            "selected-world terminal occurrence digest",
                        )
                    ),
                    source_checkpoint_occurrence_projection_sha256=occurrence[
                        "projection_content_sha256"
                    ],
                    cell_id=request.cell_ids[0],
                )
                context.record_artifact(
                    observation["observation_path"],
                    artifact_family=(
                        "science.workhorse_selected_world_host_observation_v1"
                    ),
                    retention_class="R4",
                )
                host_boundary_observation = {
                    "observation_mode": observation["observation_mode"],
                    "path": observation["observation_path"],
                    "file_sha256": observation["observation_file_sha256"],
                    "content_sha256": observation["observation_content_sha256"],
                    "file_bytes": observation["observation_file_bytes"],
                    "checkpoint_occurrence_sha256": occurrence[
                        "occurrence_digest"
                    ]["hex"],
                    "checkpoint_occurrence_projection_sha256": occurrence[
                        "projection_content_sha256"
                    ],
                }
            checkpoint = {
                "schema_version": "eiel_workhorse_operation_checkpoint_v2",
                "time_authority": "absolute_tick_v1",
                "operation": operation["operation"],
                "next_absolute_tick": target,
                "checkpoint_ticks": list(expected_ticks),
                "source_checkpoint_identity_sha256": operation[
                    "source_checkpoint_identity_sha256"
                ],
                "science_continuation_contract_sha256": operation[
                    "science_continuation_contract_sha256"
                ],
                "science_continuation_identity": dict(
                    operation["continuation_contract"][
                        "science_continuation_identity"
                    ]
                ),
                "terminal_indexed_world_state_set_identity": dict(
                    terminal_index["world_state_set_identity"]
                ),
                "indexed_checkpoint_receipts": committed_checkpoints,
                "observation_mode": "OFF",
                "science_continuation_claimed": False,
                **(
                    {
                        "host_boundary_observation": dict(
                            host_boundary_observation
                        )
                    }
                    if host_boundary_observation is not None
                    else {}
                ),
                "acs_checkpoint_staging": acs_staging,
                "terminal_cell_digests": {
                    request.cell_ids[0]: result["receipt_content_sha256"]
                },
            }
            return {
                "status": "completed",
                "checkpoint": checkpoint,
                "metrics_rows": [
                    {
                        "cell_id": request.cell_ids[0],
                        "metrics": {
                            INDEXED_ACS_CONTINUATION_AGGREGATE_WTPS_METRIC_V2: result[
                                "aggregate_world_ticks_per_second"
                            ]
                        },
                    }
                ],
                "result": {
                    "schema_version": (
                        "eiel_workhorse_indexed_continuation_provider_result_v2"
                    ),
                    "operation": operation["operation"],
                    "shape_id": contract.shape_id,
                    "backend_id": contract.backend_id,
                    "B": contract.B,
                    "preferred_temporal_k": contract.temporal_k,
                    "source_world_state_set_sha256": operation[
                        "source_checkpoint_identity_sha256"
                    ],
                    "science_continuation_identity_sha256": operation[
                        "continuation_contract"
                    ]["science_continuation_identity"][
                        "science_continuation_identity_sha256"
                    ],
                    "physical_continuation_plan_sha256": operation[
                        "continuation_contract"
                    ]["physical_continuation_plan"][
                        "physical_continuation_plan_sha256"
                    ],
                    "execution_result_path": str(result_path.resolve()),
                    "execution_result_file_sha256": result["receipt_file_sha256"],
                    "execution_result_content_sha256": result[
                        "receipt_content_sha256"
                    ],
                    "absolute_frontiers": result["absolute_frontiers"],
                    "checkpoint_ticks": list(expected_ticks),
                    "aggregate_world_ticks_per_second": result[
                        "aggregate_world_ticks_per_second"
                    ],
                    "indexed_checkpoint_receipts": committed_checkpoints,
                    "acs_checkpoint_staging": acs_staging,
                    **(
                        {
                            "host_boundary_observation": dict(
                                host_boundary_observation
                            )
                        }
                        if host_boundary_observation is not None
                        else {}
                    ),
                },
            }
        if operation["operation"] == SEMANTIC_EVIDENCE_CASE_OPERATION_V2:
            semantic_case = operation["semantic_case_contract"]
            case_law = semantic_case["case_law"]
            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                operation["contract"]
            )
            # This is deliberately after the adapter's Scheduler reopen and
            # heartbeat, and before the single physical semantic-case call.
            validate_workhorse_live_device_under_semantic_parent_v2(parent)
            fragment_root = Path(context.batch_root) / "workhorse-semantic-fragment"
            event_sink_root = Path(context.batch_root) / "workhorse-semantic-event-sink"
            provisional_raw = run_workhorse_semantic_case_v2(
                operation["artifact"]["path"],
                case_id=case_law["case_id"],
                semantic_temporal_k=semantic_case["semantic_temporal_k"],
                event_sink_root=event_sink_root,
                fragment_output_root=fragment_root,
                canonical_parent_authority_binding=parent,
                external_source_checkpoint_binding=operation[
                    "external_source_checkpoint_binding"
                ],
                science_checkpoint_compatibility_binding=operation[
                    "science_checkpoint_compatibility_binding"
                ],
            )
            provisional, fragment = _semantic_provisional_result(
                provisional_raw,
                operation=operation,
                canonical_parent=parent,
            )
            fragment_path = Path(provisional["case_fragment_path"])
            if fragment_path.parent.resolve() != fragment_root.resolve():
                raise CampaignProviderError(
                    "semantic raw fragment escaped its exact output root"
                )
            fragment_artifact_paths: list[str] = []
            for evidence_root, artifact_family in (
                (fragment_root, "workhorse_semantic_case_raw_fragment_v2"),
                (event_sink_root, "workhorse_semantic_case_event_sink_v2"),
            ):
                for artifact_path in sorted(evidence_root.rglob("*")):
                    if artifact_path.is_symlink():
                        raise CampaignProviderError(
                            "semantic raw evidence contains a symbolic link"
                        )
                    if artifact_path.is_file():
                        context.record_artifact(
                            artifact_path,
                            artifact_family=artifact_family,
                            retention_class="R3",
                        )
                        fragment_artifact_paths.append(str(artifact_path.resolve()))

            produced_source: dict[str, Any] | None = None
            occurrence_artifact_paths: list[str] = []
            if case_law["produces_base_checkpoint"] is True:
                base_checkpoint = fragment.get("base_checkpoint")
                if not isinstance(base_checkpoint, Mapping) or not isinstance(
                    base_checkpoint.get("identity"), Mapping
                ):
                    raise CampaignProviderError(
                        "checkpoint semantic case did not produce its base checkpoint"
                    )
                materialized = self.operation_adapter.materialize_workhorse_semantic_checkpoint_occurrence_v2(
                    canonical_parent_authority_binding=parent,
                    case_fragment_path=fragment_path,
                    science_checkpoint_compatibility_path=(
                        Path(__file__).resolve().parents[2]
                        / operation["science_checkpoint_compatibility_binding"][
                            "science_consumer_extension_contract_path"
                        ]
                    ),
                    occurrence_root=Path(context.batch_root)
                    / "workhorse-semantic-checkpoint-occurrence",
                )
                if type(materialized) is not dict or set(materialized) != {
                    "external_source_checkpoint_binding",
                    "artifact_paths",
                }:
                    raise CampaignProviderError(
                        "canonical semantic checkpoint occurrence result is inexact"
                    )
                produced_source = validate_external_checkpoint_source_binding_v2(
                    materialized["external_source_checkpoint_binding"],
                    expected_identity_sha256=base_checkpoint["identity"][
                        "base_checkpoint_identity_sha256"
                    ],
                )
                raw_occurrence_paths = materialized["artifact_paths"]
                if (
                    not isinstance(raw_occurrence_paths, Sequence)
                    or isinstance(raw_occurrence_paths, (str, bytes))
                    or not raw_occurrence_paths
                ):
                    raise CampaignProviderError(
                        "semantic checkpoint occurrence inventory is absent"
                    )
                for raw_path in raw_occurrence_paths:
                    artifact_path = Path(str(raw_path))
                    if (
                        not artifact_path.is_absolute()
                        or artifact_path.is_symlink()
                        or not artifact_path.is_file()
                    ):
                        raise CampaignProviderError(
                            "semantic checkpoint occurrence artifact is unsafe"
                        )
                    resolved = str(artifact_path.resolve())
                    if resolved not in fragment_artifact_paths:
                        context.record_artifact(
                            artifact_path,
                            artifact_family=(
                                "workhorse_semantic_checkpoint_occurrence_v2"
                            ),
                            retention_class="R3",
                        )
                    occurrence_artifact_paths.append(resolved)
                if len(set(occurrence_artifact_paths)) != len(
                    occurrence_artifact_paths
                ):
                    raise CampaignProviderError(
                        "semantic checkpoint occurrence inventory contains duplicates"
                    )
            elif fragment.get("base_checkpoint") is not None:
                raise CampaignProviderError(
                    "non-checkpoint semantic case produced checkpoint authority"
                )

            terminal_contract = semantic_case["attempt_terminal_contract"]
            final_frontiers = terminal_contract["final_frontiers"]
            start = contract.start_next_tick
            target = start + contract.total_ticks
            committed = final_frontiers["published_artifact_next_tick"]
            absolute_frontiers = {
                "start_next_tick": start,
                "target_next_tick": target,
                "committed_next_tick": committed,
            }
            checkpoint = {
                "schema_version": "eiel_workhorse_operation_checkpoint_v2",
                "time_authority": "absolute_tick_v1",
                "operation": operation["operation"],
                "next_absolute_tick": committed,
                "source_checkpoint_identity_sha256": operation[
                    "source_checkpoint_identity_sha256"
                ],
                "science_continuation_contract_sha256": operation[
                    "science_continuation_contract_sha256"
                ],
                "produced_external_source_checkpoint_binding": produced_source,
                "observation_mode": "OFF",
                "science_continuation_claimed": False,
                "attempt_terminal_contract": dict(terminal_contract),
                "terminal_cell_digests": {
                    request.cell_ids[0]: provisional["case_fragment_body_sha256"]
                },
            }
            return {
                "status": "completed",
                "checkpoint": checkpoint,
                "metrics_rows": [
                    {
                        "cell_id": request.cell_ids[0],
                        "metrics": {"workhorse.semantic_case_provisional_completed": 1},
                    }
                ],
                "result": {
                    "schema_version": (
                        "eiel_workhorse_semantic_case_provider_result_v2"
                    ),
                    "operation": operation["operation"],
                    "case_id": case_law["case_id"],
                    "shape_id": contract.shape_id,
                    "backend_id": contract.backend_id,
                    "B": contract.B,
                    "temporal_k": contract.temporal_k,
                    "semantic_temporal_k": semantic_case["semantic_temporal_k"],
                    "artifact_sha256": operation["artifact"]["sha256"],
                    "sealed_artifact_identity_sha256": operation["artifact"][
                        "sealed_artifact_identity_sha256"
                    ],
                    "expected_attempt_terminal_status": provisional[
                        "expected_attempt_terminal_status"
                    ],
                    "attempt_terminal_contract": dict(terminal_contract),
                    "provisional_result_class": provisional["provisional_result_class"],
                    "provider_terminalization_action": provisional[
                        "provider_terminalization_action"
                    ],
                    "case_fragment_path": provisional["case_fragment_path"],
                    "case_fragment_file_sha256": provisional[
                        "case_fragment_file_sha256"
                    ],
                    "case_fragment_body_sha256": provisional[
                        "case_fragment_body_sha256"
                    ],
                    "source_checkpoint_identity_sha256": operation[
                        "source_checkpoint_identity_sha256"
                    ],
                    "external_source_checkpoint_binding": operation[
                        "external_source_checkpoint_binding"
                    ],
                    "science_checkpoint_compatibility_binding": operation[
                        "science_checkpoint_compatibility_binding"
                    ],
                    "produced_external_source_checkpoint_binding": produced_source,
                    "checkpoint_occurrence_artifact_paths": (occurrence_artifact_paths),
                    "absolute_frontiers": absolute_frontiers,
                },
            }
        contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            operation["contract"]
        )
        execution_result_path = (
            Path(context.batch_root) / "workhorse-row-execution-result.json"
        )
        row_event_sink_root = Path(context.batch_root) / "event-sink"
        acs_coordinator: Any | None = None
        acs_checkpoint_receipt: dict[str, Any] | None = None
        if request.acs_checkpoint_slots:
            from .cuda_full_evolution_v2_acs_adapter import (
                CudaFullEvolutionV2WorkhorseACSCoordinatorV1,
            )

            acs_coordinator = CudaFullEvolutionV2WorkhorseACSCoordinatorV1(
                slots=request.acs_checkpoint_slots,
                checkpoint_root=Path(context.batch_root) / "workhorse-checkpoints",
                artifact_path=operation["artifact"]["path"],
                attempt_authority_binding=parent,
                require_native_pool=self._acs_require_native_pool,
            )
        try:
            checkpoint_ticks = _execution_row_checkpoint_ticks_v2(
                value,
                operation=operation,
                contract=contract,
            )
            _validate_science_founder_default_controls_v1(
                value,
                checkpoint_ticks=checkpoint_ticks,
                acs_checkpoint_slots=request.acs_checkpoint_slots,
            )
            exact_founder_world_identity = (
                _exact_founder_runtime_world_identity_v1(value)
            )
            result = run_workhorse_canonical_admitted_row_v2(
                operation["artifact"]["path"],
                envelope_receipt_path=operation["envelope"]["path"],
                canonical_parent_authority_binding=parent,
                authoritative_micro_receipt_path=operation["authoritative_micro_receipt"][
                    "path"
                ],
                semantic_gate_receipt_path=operation["semantic_gate_receipt"]["path"],
                science_continuation_contract_sha256=operation[
                    "science_continuation_contract_sha256"
                ],
                event_sink_root=row_event_sink_root,
                execution_result_path=execution_result_path,
                B=contract.B,
                temporal_k=contract.temporal_k,
                world_identity_binding=exact_founder_world_identity,
                warmup_ticks=operation["measurement_schedule"]["warmup_ticks"],
                measurement_window_ticks=operation["measurement_schedule"][
                    "measurement_window_ticks"
                ],
                measured_windows=operation["measurement_schedule"]["measured_windows"],
                checkpoint_transport=acs_coordinator,
                checkpoint_ticks=checkpoint_ticks,
            )
            if acs_coordinator is not None:
                pending_tokens = acs_coordinator.pending_tokens()
                if result.get("outcome") == "RUNTIME_HEADROOM_REJECTED":
                    if pending_tokens:
                        raise CampaignProviderError(
                            "Workhorse rejected row unexpectedly captured an ACS "
                            "checkpoint"
                        )
                else:
                    if len(pending_tokens) != len(checkpoint_ticks):
                        raise CampaignProviderError(
                            "Workhorse row ACS checkpoint token schedule differs"
                        )
                    parent_occurrence_digest: Mapping[str, str] | None = None
                    committed_receipts: list[dict[str, Any]] = []
                    for checkpoint_index, (token, checkpoint_tick) in enumerate(
                        zip(pending_tokens, checkpoint_ticks, strict=True)
                    ):
                        committed = _commit_workhorse_row_acs_checkpoint_v2(
                            context=context,
                            coordinator=acs_coordinator,
                            token=token,
                            operation_adapter=self.operation_adapter,
                            canonical_parent_authority_binding=parent,
                            contract=contract,
                            expected_checkpoint_tick=checkpoint_tick,
                            parent_checkpoint_digest=parent_occurrence_digest,
                        )
                        projection = committed.get(
                            "checkpoint_occurrence_projection"
                        )
                        occurrence_digest = (
                            projection.get("occurrence_digest")
                            if isinstance(projection, Mapping)
                            else None
                        )
                        if (
                            checkpoint_index + 1 < len(checkpoint_ticks)
                            and not isinstance(occurrence_digest, Mapping)
                        ):
                            raise CampaignProviderError(
                                "Workhorse row ACS occurrence digest is absent"
                            )
                        if isinstance(occurrence_digest, Mapping):
                            parent_occurrence_digest = dict(occurrence_digest)
                        committed_receipts.append(committed)
                    acs_checkpoint_receipt = committed_receipts[-1]
                acs_coordinator.close()
        except BaseException:
            if acs_coordinator is not None:
                acs_coordinator.abort()
            raise
        acs_staging = (
            acs_coordinator.telemetry() if acs_coordinator is not None else None
        )
        context.record_artifact(
            execution_result_path,
            artifact_family="workhorse_row_execution_result_v2",
            retention_class="R3",
        )
        if row_event_sink_root.exists():
            for artifact_path in sorted(row_event_sink_root.rglob("*")):
                if artifact_path.is_symlink():
                    raise CampaignProviderError(
                        "row event-sink evidence contains a symbolic link"
                    )
                if artifact_path.is_file():
                    context.record_artifact(
                        artifact_path,
                        artifact_family="workhorse_row_event_sink_v2",
                        retention_class="R3",
                    )
        target = contract.start_next_tick + contract.total_ticks
        if result.get("outcome") == "RUNTIME_HEADROOM_REJECTED":
            expected_frontiers = {
                "start_next_tick": contract.start_next_tick,
                "target_next_tick": target,
                "committed_next_tick": contract.start_next_tick,
            }
            if result.get("absolute_frontiers") != expected_frontiers:
                raise CampaignProviderError(
                    f"{RUNTIME_HEADROOM_REJECTION_CODE_V2}: canonical rejection "
                    "changed its exact start frontier"
                )
            raise CampaignProviderError(
                f"{RUNTIME_HEADROOM_REJECTION_CODE_V2}: runtime VRAM admission "
                "failed before dispatch; the current Scheduler Attempt must "
                "terminalize FAILED at the start frontier and retry requires a "
                "fresh ExecutionRequestV2"
            )
        checkpoint = {
            "schema_version": "eiel_workhorse_operation_checkpoint_v2",
            "time_authority": "absolute_tick_v1",
            "next_absolute_tick": target,
            "source_checkpoint_identity_sha256": operation[
                "source_checkpoint_identity_sha256"
            ],
            "science_continuation_contract_sha256": operation[
                "science_continuation_contract_sha256"
            ],
            "row_checkpoint_binding": result["checkpoint_binding"],
            **(
                {"acs_checkpoint_receipt": acs_checkpoint_receipt}
                if acs_staging is not None
                else {}
            ),
            "terminal_cell_digests": {
                request.cell_ids[0]: result["receipt_content_sha256"]
            },
        }
        return {
            "status": "completed",
            "checkpoint": checkpoint,
            "metrics_rows": [
                {
                    "cell_id": request.cell_ids[0],
                    "metrics": {
                        "workhorse.p10_world_ticks_per_second": result[
                            "p10_world_ticks_per_second"
                        ]
                    },
                }
            ],
            "result": {
                "schema_version": "eiel_workhorse_single_row_provider_result_v2",
                "operation": SUPPORTED_OPERATION_V2,
                "shape_id": contract.shape_id,
                "backend_id": contract.backend_id,
                "B": contract.B,
                "temporal_k": contract.temporal_k,
                "execution_result_path": str(execution_result_path.resolve()),
                "execution_result_file_sha256": result["receipt_file_sha256"],
                "execution_result_content_sha256": result["receipt_content_sha256"],
                "absolute_frontiers": result["absolute_frontiers"],
                "checkpoint_binding": result["checkpoint_binding"],
                **(
                    {"acs_checkpoint_staging": acs_staging}
                    if acs_staging is not None
                    else {}
                ),
            },
        }

    def finalize_attempt_terminal_v2(
        self,
        *,
        plan: CampaignPlanV1 | Mapping[str, Any],
        run_root: str | Path,
        canonical_terminal_binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = _payload(plan)
        operation = _operation_contract(value)
        batch = self._single_batch(value)
        manifest_path = (
            Path(run_root).resolve()
            / "batches"
            / str(batch["batch_id"])
            / "manifest.json"
        )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CampaignProviderError(
                f"cannot reopen Workhorse batch result: {exc}"
            ) from exc
        provider_result = (
            manifest.get("result") if isinstance(manifest, Mapping) else None
        )
        if (
            not isinstance(provider_result, Mapping)
            or provider_result.get("operation") != operation["operation"]
        ):
            raise CampaignProviderError("Workhorse batch result is not exact")
        if operation["operation"] == SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1:
            return _finalize_science_demo_provider_result_v1(
                provider_result=provider_result,
                operation=operation,
                manifest_path=manifest_path,
                canonical_terminal_binding=canonical_terminal_binding,
                operation_adapter=self.operation_adapter,
            )
        if operation["operation"] == INDEXED_ACS_CONTINUATION_OPERATION_V2:
            return _finalize_indexed_continuation_provider_result_v2(
                provider_result=provider_result,
                operation=operation,
                manifest_path=manifest_path,
                canonical_terminal_binding=canonical_terminal_binding,
            )
        if operation["operation"] == DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2:
            if (
                provider_result.get("schema_version")
                != "eiel_workhorse_device_envelope_calibration_provider_result_v2"
                or provider_result.get("shape_id") != operation["shape_id"]
                or provider_result.get("backend_id") != operation["backend_id"]
                or provider_result.get("artifact_sha256")
                != operation["artifact"]["sha256"]
                or provider_result.get("sealed_artifact_identity_sha256")
                != operation["artifact"]["sealed_artifact_identity_sha256"]
                or provider_result.get("compiled_temporal_k")
                != operation["compiled_temporal_k"]
                or provider_result.get("device_identity")
                != operation["device_identity"]
                or provider_result.get("reserve_policy") != operation["reserve_policy"]
            ):
                raise CampaignProviderError(
                    "Workhorse calibration batch result is not exact"
                )
            output = (
                manifest_path.parent
                / "workhorse-device-envelope-calibration-receipt.json"
            )
            receipt = (
                validate_workhorse_terminalized_device_envelope_calibration_receipt_v2(
                    output
                )
                if output.exists() and not output.is_symlink()
                else seal_workhorse_terminalized_device_envelope_calibration_receipt_v2(
                    output,
                    calibration_receipt_path=str(
                        provider_result["calibration_receipt_path"]
                    ),
                    canonical_terminal_binding=canonical_terminal_binding,
                )
            )
            return {
                "schema_version": ("eiel_workhorse_terminalized_operation_binding_v2"),
                "operation": operation["operation"],
                "final_receipt_path": receipt["receipt_path"],
                "final_receipt_file_sha256": receipt["receipt_file_sha256"],
                "final_receipt_content_sha256": receipt["receipt_content_sha256"],
                "calibration_receipt_path": receipt["receipt_path"],
                "attempt_id": canonical_terminal_binding.get("attempt_id"),
                "lifecycle_binding": {
                    "artifact_family": (
                        "qualification.workhorse_device_envelope_calibration_v2"
                    ),
                    "retention_class": "R3",
                    "owner_writer": (
                        "cuda_full_evolution_workhorse_v2_single_operation_owner"
                    ),
                    "release_predicate": (
                        "technical calibration retained for exact row admission"
                    ),
                },
            }
        if operation["operation"] == SEMANTIC_EVIDENCE_CASE_OPERATION_V2:
            expected_fields = {
                "schema_version",
                "operation",
                "case_id",
                "shape_id",
                "backend_id",
                "B",
                "temporal_k",
                "semantic_temporal_k",
                "artifact_sha256",
                "sealed_artifact_identity_sha256",
                "expected_attempt_terminal_status",
                "attempt_terminal_contract",
                "provisional_result_class",
                "provider_terminalization_action",
                "case_fragment_path",
                "case_fragment_file_sha256",
                "case_fragment_body_sha256",
                "source_checkpoint_identity_sha256",
                "external_source_checkpoint_binding",
                "science_checkpoint_compatibility_binding",
                "produced_external_source_checkpoint_binding",
                "checkpoint_occurrence_artifact_paths",
                "absolute_frontiers",
            }
            if (
                type(provider_result) is not dict
                or set(provider_result) != expected_fields
            ):
                raise CampaignProviderError(
                    "Workhorse semantic provider result fields are inexact"
                )
            semantic_case = operation["semantic_case_contract"]
            case_law = semantic_case["case_law"]
            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                operation["contract"]
            )
            terminal_contract = semantic_case["attempt_terminal_contract"]
            expected_status = terminal_contract["attempt_terminal_status"]
            expected_result_class = (
                "SUCCESS" if expected_status == "COMPLETED" else "VALID_NEGATIVE"
            )
            expected_provisional_class = (
                "SUCCESS_PENDING_COMPLETED_TERMINAL"
                if expected_status == "COMPLETED"
                else "EXPECTED_NEGATIVE_PENDING_FAILED_TERMINAL"
            )
            expected_action = (
                "RETURN_FOR_COMPLETED_TERMINALIZATION"
                if expected_status == "COMPLETED"
                else "RETURN_FOR_FAILED_ATTEMPT_TERMINALIZATION"
            )
            expected_absolute_frontiers = {
                "start_next_tick": contract.start_next_tick,
                "target_next_tick": (contract.start_next_tick + contract.total_ticks),
                "committed_next_tick": terminal_contract["final_frontiers"][
                    "published_artifact_next_tick"
                ],
            }
            fragment_path = Path(str(provider_result["case_fragment_path"]))
            try:
                fragment, _ = reopen_workhorse_semantic_case_fragment_v2(fragment_path)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise CampaignProviderError(
                    f"Workhorse semantic fragment cannot reopen: {exc}"
                ) from exc
            if (
                provider_result["schema_version"]
                != "eiel_workhorse_semantic_case_provider_result_v2"
                or provider_result["case_id"] != case_law["case_id"]
                or provider_result["shape_id"] != contract.shape_id
                or provider_result["backend_id"] != contract.backend_id
                or provider_result["B"] != contract.B
                or provider_result["temporal_k"] != contract.temporal_k
                or provider_result["semantic_temporal_k"]
                != semantic_case["semantic_temporal_k"]
                or provider_result["artifact_sha256"] != operation["artifact"]["sha256"]
                or provider_result["sealed_artifact_identity_sha256"]
                != operation["artifact"]["sealed_artifact_identity_sha256"]
                or provider_result["expected_attempt_terminal_status"]
                != expected_status
                or provider_result["attempt_terminal_contract"] != terminal_contract
                or provider_result["provisional_result_class"]
                != expected_provisional_class
                or provider_result["provider_terminalization_action"] != expected_action
                or provider_result["source_checkpoint_identity_sha256"]
                != operation["source_checkpoint_identity_sha256"]
                or provider_result["external_source_checkpoint_binding"]
                != operation["external_source_checkpoint_binding"]
                or provider_result["science_checkpoint_compatibility_binding"]
                != operation["science_checkpoint_compatibility_binding"]
                or provider_result["absolute_frontiers"] != expected_absolute_frontiers
                or fragment_path.is_symlink()
                or not fragment_path.is_file()
                or provider_result["case_fragment_file_sha256"]
                != _file_sha256(fragment_path)
                or provider_result["case_fragment_body_sha256"]
                != fragment.get("fragment_body_sha256")
                or fragment.get("case_id") != case_law["case_id"]
                or fragment.get("canonical_parent_binding", {}).get("attempt_id")
                != canonical_terminal_binding.get("attempt_id")
            ):
                raise CampaignProviderError(
                    "Workhorse semantic provider result is not exact"
                )
            produced_source = provider_result[
                "produced_external_source_checkpoint_binding"
            ]
            occurrence_paths = provider_result["checkpoint_occurrence_artifact_paths"]
            if case_law["produces_base_checkpoint"] is True:
                base_checkpoint = fragment.get("base_checkpoint")
                if (
                    not isinstance(base_checkpoint, Mapping)
                    or not isinstance(base_checkpoint.get("identity"), Mapping)
                    or not isinstance(produced_source, Mapping)
                    or validate_external_checkpoint_source_binding_v2(
                        produced_source,
                        expected_identity_sha256=base_checkpoint["identity"][
                            "base_checkpoint_identity_sha256"
                        ],
                    )
                    != produced_source
                    or not isinstance(occurrence_paths, list)
                    or not occurrence_paths
                    or any(
                        type(path) is not str
                        or not Path(path).is_absolute()
                        or Path(path).is_symlink()
                        or not Path(path).is_file()
                        for path in occurrence_paths
                    )
                ):
                    raise CampaignProviderError(
                        "Workhorse semantic checkpoint occurrence result changed"
                    )
            elif produced_source is not None or occurrence_paths != []:
                raise CampaignProviderError(
                    "Workhorse semantic case gained checkpoint occurrence output"
                )
            output = manifest_path.parent / "workhorse-semantic-case-receipt.json"
            receipt = (
                validate_workhorse_terminalized_semantic_case_receipt_v2(output)
                if output.exists() and not output.is_symlink()
                else seal_workhorse_terminalized_semantic_case_receipt_v2(
                    output,
                    case_fragment_path=fragment_path,
                    canonical_terminal_binding=canonical_terminal_binding,
                    produced_external_source_checkpoint_binding=produced_source,
                )
            )
            reopened_receipt = validate_workhorse_terminalized_semantic_case_receipt_v2(
                output
            )
            if (
                receipt != reopened_receipt
                or receipt.get("case_id") != case_law["case_id"]
                or receipt.get("shape_id") != contract.shape_id
                or receipt.get("artifact_sha256") != operation["artifact"]["sha256"]
                or receipt.get("case_fragment_path") != str(fragment_path.resolve())
                or receipt.get("case_fragment_file_sha256")
                != provider_result["case_fragment_file_sha256"]
                or receipt.get("case_fragment_body_sha256")
                != provider_result["case_fragment_body_sha256"]
                or receipt.get("result_class") != expected_result_class
            ):
                raise CampaignProviderError(
                    "Workhorse terminal semantic receipt differs from the operation"
                )
            return {
                "schema_version": ("eiel_workhorse_terminalized_operation_binding_v2"),
                "operation": operation["operation"],
                "result_class": expected_result_class,
                "final_receipt_path": receipt["receipt_path"],
                "final_receipt_file_sha256": receipt["receipt_file_sha256"],
                "final_receipt_content_sha256": receipt["receipt_content_sha256"],
                "semantic_case_receipt_path": receipt["receipt_path"],
                "semantic_case_receipt_file_sha256": receipt["receipt_file_sha256"],
                "semantic_case_receipt_content_sha256": receipt[
                    "receipt_content_sha256"
                ],
                "attempt_id": canonical_terminal_binding.get("attempt_id"),
                "lifecycle_binding": {
                    "artifact_family": "qualification.workhorse_semantic_case_v2",
                    "retention_class": "R3",
                    "owner_writer": (
                        "cuda_full_evolution_workhorse_v2_single_operation_owner"
                    ),
                    "release_predicate": (
                        "technical semantic case retained for exact GPU-free aggregation"
                    ),
                },
            }
        if (
            provider_result.get("schema_version")
            != "eiel_workhorse_single_row_provider_result_v2"
        ):
            raise CampaignProviderError("Workhorse row batch result is not exact")
        output = manifest_path.parent / "workhorse-technical-row-receipt.json"
        receipt = (
            validate_workhorse_technical_row_receipt_v2(output)
            if output.exists() and not output.is_symlink()
            else seal_workhorse_technical_row_receipt_v2(
                output,
                execution_result_path=str(provider_result["execution_result_path"]),
                canonical_terminal_binding=canonical_terminal_binding,
            )
        )
        return {
            "schema_version": "eiel_workhorse_terminalized_operation_binding_v2",
            "operation": operation["operation"],
            "final_receipt_path": receipt["receipt_path"],
            "final_receipt_file_sha256": receipt["receipt_file_sha256"],
            "final_receipt_content_sha256": receipt["receipt_content_sha256"],
            "row_receipt_path": receipt["receipt_path"],
            "row_receipt_file_sha256": receipt["receipt_file_sha256"],
            "row_receipt_content_sha256": receipt["receipt_content_sha256"],
            "attempt_id": canonical_terminal_binding.get("attempt_id"),
            "lifecycle_binding": {
                "artifact_family": "qualification.workhorse_technical_row_v2",
                "retention_class": "R3",
                "owner_writer": ("cuda_full_evolution_workhorse_v2_single_row_owner"),
                "release_predicate": (
                    "technical row retained for offline matrix aggregation"
                ),
            },
        }


def cuda_full_evolution_workhorse_v2_provider_registry_v1(
    *, operation_adapter: WorkhorseLaunchpadOperationAdapterV2 | None = None
) -> ExecutionProviderRegistryV1:
    registry = default_execution_provider_registry()
    for shape in SUPPORTED_WORKHORSE_SHAPES_V2:
        registry.register(
            CudaFullEvolutionWorkhorseV2ExecutionProviderV1(
                shape=shape,
                operation_adapter=operation_adapter,
            )
        )
    return registry


__all__ = [
    "ATTEMPT_AUTHORITY_NAME",
    "CudaFullEvolutionWorkhorseV2ExecutionProviderV1",
    "DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2",
    "EXECUTION_ROW_OPERATION_V2",
    "EXACT_FOUNDER_BOOTSTRAP_SCHEMA_V1",
    "EXACT_FOUNDER_MATERIALIZATION_SCHEMA_V1",
    "INDEXED_ACS_CONTINUATION_AGGREGATE_WTPS_METRIC_V2",
    "INDEXED_ACS_CONTINUATION_OPERATION_V2",
    "OPERATION_EXTENSION_ID",
    "OPERATION_SCHEMA_V2",
    "OPERATION_SCHEMA_V3",
    "PROVIDER_ID",
    "SEMANTIC_EVIDENCE_CASE_OPERATION_V2",
    "SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1",
    "SUPPORTED_OPERATION_V2",
    "SUPPORTED_OPERATIONS_V2",
    "WORLD_BASE",
    "cuda_full_evolution_workhorse_v2_profile_registry_v1",
    "cuda_full_evolution_workhorse_v2_provider_registry_v1",
    "cuda_full_evolution_workhorse_v2_signed_planning_resource_accounting_v2",
    "compile_workhorse_absolute_dispatch_plan_v2",
    "inspect_workhorse_indexed_checkpoint_set_v1",
    "is_cuda_full_evolution_workhorse_v2_profile_v1",
    "is_cuda_full_evolution_workhorse_v2_spec_v1",
    "materialize_workhorse_exact_founder_plan_v1",
    "workhorse_exact_founder_bootstrap_contract_v1",
    "workhorse_indexed_continuation_contract_v2",
    "workhorse_repack_binding_v2",
]
