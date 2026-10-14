"""Deterministic W32 CUDA V2 science host-context and MUTSUP authoring.

This module is deliberately host-only.  It reads the already sealed
engineering route summary, binds its native artifact/device identities, and
builds the smallest science-instrument context needed to author an actual
paired-seed MUTSUP request.  It never builds, reserves, launches, or touches
CUDA.

The semantic campaign is kept separate from acquisition and physical identity:
the campaign names W32 inputs, controls, seeds, observations, and ownership;
the host context names the selected native route and observation adapter; and
the physical execution core carries the sealed artifact/device receipt.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final

import yaml

from emergence_contracts import (
    PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION,
    CanonicalPayloadV1,
    ContentDigest,
    EEDReleaseStatusV1,
    ProfileResourceEnvelopeV1,
    SchemaVersion,
    ScienceControlDescriptorV1,
    ScienceInstrumentManifestV1,
    ScienceObservationProfileDescriptorV1,
    ScienceObservationProfileV1,
    ScienceRecordKindV1,
    ScienceReducerDescriptorV1,
    SCIENCE_GENOTYPE_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1,
    ScienceRouteBindingV1,
    canonical_json_bytes,
    canonical_sha256,
    science_observation_profile_catalog_sha256_v1,
)
from engine.campaigns.baseline_v1 import BASELINE_PROFILE_ID
from engine.campaigns.artifact_lifecycle import lifecycle_profile_contract_v1
from engine.campaigns.compiler import compile_campaign, load_campaign_spec
from engine.campaigns.cuda_full_evolution_v2_provider import (
    CUDA_FULL_EVOLUTION_V2_BASELINE_DEVIATION_REQUIREMENT_IDS_V1,
    CudaFullEvolutionV2ExecutionProviderV1,
    cuda_full_evolution_v2_profile_registry_v1,
    cuda_full_evolution_v2_wall_seconds_upper_bound_v1,
)
from engine.native_runtime.cuda_v2_science_capture_v1 import (
    science_capture_producer_phase_registry_v1,
)
from engine.perf.cuda_full_evolution_v2.contract import (
    CUDA_NATIVE_ABI_V1,
    CUDA_NATIVE_PROFILE_ID_V2,
    CUDA_NATIVE_ROUTE_V2,
    FULL_EVOLUTION_CAPABILITIES_V2,
    INITIAL_POPULATION_V2,
    RESIDENT_CAPACITY_C_V2,
    WORLD_WIDTH_V2,
)

from emergence_evidence.science_mutsup_direct_retention_v1 import (
    MUTSUP_DIRECT_RETENTION_CONFIGURATION_SCHEMA_VERSION,
    MUTSUP_DIRECT_RETENTION_REDUCER_ID,
    MUTSUP_DIRECT_RETENTION_OUTPUT_CONTRACT_ID,
    mutsup_direct_retention_reducer_descriptor_v1,
)

from .science_instrument import (
    SHADOW_PROJECTION_AUTHORITY,
    ScienceCRCASemanticBindingV1,
    compile_science_campaign_v1,
    science_control_catalog_sha256,
)
from .science_instrument_artifacts_v1 import serialize_science_host_context_v1
from .science_instrument_host_front_door_v1 import ScienceInstrumentHostContextV1
from .science_qrr_projection_v1 import ScienceQRRRouteAdapterV1
from .science_qualification_envelope_projection_v1 import (
    ScienceQualifiedEnvelopeProjectionV1,
)
from .science_request_v2 import normalize_rich_production_science_request_v2
from .science_world_bundle_v1 import materialize_science_world_bundle_v1


SCIENCE_CUDA_V2_HOST_CONTEXT_SCHEMA_VERSION: Final[str] = (
    "eiel_science_cuda_v2_host_context_v1"
)
SCIENCE_CUDA_V2_MUTSUP_AUTHORING_SCHEMA_VERSION: Final[str] = (
    "eiel_science_cuda_v2_mutsup_authoring_v1"
)

_SCIENCE_OBSERVATION_ADAPTER_DEPENDENCIES: Final[tuple[str, ...]] = (
    "engine/native_runtime/cuda_v2_science_capture_v1.py",
    "engine/native_runtime/science_observation_projection_v1.py",
    "engine/perf/cuda_full_evolution_v2/direct_sparse_runtime.py",
    "engine/perf/cuda_full_evolution_v2/science_extension.py",
)
_SCIENCE_OBSERVATION_ABI_PATH: Final[str] = (
    "_spec/science_instrument/v1/science_observation_abi_v1.json"
)

SCIENCE_CUDA_V2_W32_CARRIER_PROFILE_ID: Final[str] = "w32x32p256c512_gpuv2"
SCIENCE_CUDA_V2_W32_WORLD_PROFILE_ID: Final[str] = "mutsup-cuda-v2"
SCIENCE_CUDA_V2_W32_FOUNDER_PROFILE_IDS: Final[tuple[str, ...]] = (
    "portable-demo-founder-a-v1",
    "portable-demo-founder-b-v1",
)
SCIENCE_CUDA_V2_W32_PLACEMENT_PROFILE_ID: Final[str] = "portable-demo-placement-v1"
SCIENCE_CUDA_V2_W32_VARIATION_PROFILE_ID: Final[str] = "portable-demo-variation-v1"
SCIENCE_CUDA_V2_W32_POLICY_PROFILE_ID: Final[str] = "cuda-v2-fixed-mlp-policy-v1"
SCIENCE_CUDA_V2_W32_MUTATION_OPERATOR_ID: Final[str] = "genome_v2_mutation"

SCIENCE_CUDA_V2_MUTSUP_REQUEST_ID: Final[str] = "mutsup-w32-paired-seed-10k-v1"
SCIENCE_CUDA_V2_MUTSUP_CAMPAIGN_ID: Final[str] = "mutsup-w32-paired-seed-10k-v1"
SCIENCE_CUDA_V2_MUTSUP_TEST_ID: Final[str] = "MUTSUP"
SCIENCE_CUDA_V2_MUTSUP_TEST_VERSION: Final[str] = "1"
SCIENCE_CUDA_V2_MUTSUP_STUDY_ID: Final[str] = "MUTSUP-W32-PAIRED-SEED-10K"
SCIENCE_CUDA_V2_MUTSUP_TASK_SEEDS: Final[tuple[int, ...]] = (
    100001,
    100002,
    100003,
)
SCIENCE_CUDA_V2_MUTSUP_OBSERVATION_TICKS: Final[tuple[int, ...]] = (
    0,
    80,
    160,
    400,
    800,
    1600,
    5000,
    10000,
)
SCIENCE_CUDA_V2_MUTSUP_CHECKPOINT_TICKS: Final[tuple[int, ...]] = (
    5000,
    10000,
)
SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS: Final[int] = 10_000

_DEFAULT_PACKET_RELATIVE = Path(
    "_artifacts/eiel-science-instrument-v1-gpuv2-20260827/"
    "engineering-complete/engineering_complete_packet_v1.json"
)
_DEFAULT_DEVICE_RECEIPT_RELATIVE = Path(
    "_artifacts/eiel-science-instrument-v1-gpuv2-20260827/"
    "engineering-demo-inputs/mixed-b136/sealed.device-selection.json"
)
_DEFAULT_AUTHORING_SPEC_RELATIVE = Path(
    "_spec/campaigns/v1/science_instrument/mutsup_w32_paired_seed_10k_v1.yaml"
)
_DEFAULT_BASE_PLAN_SPEC_RELATIVE = Path(
    "_spec/campaigns/v1/science_instrument/mutsup_w32_direct_retention_10k_base_v1.yaml"
)

# This is the sealed route digest named by the engineering-complete packet.
# The route is a physical input identity, so a summary that substitutes a
# syntactically valid but different digest must fail closed.
_RETAINED_ENGINEERING_PACKET_ROUTE_PROFILE_SHA256 = (
    "8ca159dfa9795ce68de6572b9351a9d5f03a72feda4f472dd0ac694a1b98b83e"
)

_OBSERVATION_CAPABILITY_IDS: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            "birth",
            "death",
            "mutation",
            "birth_parentage",
            "mutation_provenance",
            "genome_identity",
            "lineage_ancestry",
            "birth_count",
            "mutation_count",
            "population_count",
            "causal_lifecycle",
            "control_readback",
            *SCIENCE_GENOTYPE_ATTRIBUTION_CAPABILITIES_V1,
            *SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1,
        }
    )
)


class ScienceCudaV2HostContextError(ValueError):
    """The sealed W32 route or MUTSUP authoring contract is not exact."""


def _payload(schema: str, value: object) -> CanonicalPayloadV1:
    return CanonicalPayloadV1.from_value(SchemaVersion(schema), value)


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScienceCudaV2HostContextError(f"{field} must be an object")
    return dict(value)


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ScienceCudaV2HostContextError(f"{field} must be a non-empty exact string")
    return value


def _sha256(value: object, *, field: str) -> ContentDigest:
    if isinstance(value, ContentDigest):
        return value
    if not isinstance(value, str):
        raise ScienceCudaV2HostContextError(f"{field} must be a SHA-256 hex string")
    try:
        return ContentDigest("sha256", value)
    except (TypeError, ValueError) as error:
        raise ScienceCudaV2HostContextError(
            f"{field} is not a SHA-256 digest"
        ) from error


def _digest(value: object, *, field: str) -> ContentDigest:
    if isinstance(value, Mapping):
        if set(value) != {"algorithm", "hex"}:
            raise ScienceCudaV2HostContextError(f"{field} is not an exact digest")
        try:
            return ContentDigest(value["algorithm"], value["hex"])
        except (TypeError, ValueError) as error:
            raise ScienceCudaV2HostContextError(
                f"{field} is not a SHA-256 digest"
            ) from error
    return _sha256(value, field=field)


def _default_repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ScienceCudaV2HostContextError(f"{field} is not a regular file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScienceCudaV2HostContextError(f"{field} is unreadable: {path}") from error
    return _mapping(raw, field=field)


def _resolve_path(path: str | Path, *, repository_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    return candidate.resolve()


def _load_summary(value: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if value is None:
        return _read_json(
            _default_repository_root() / _DEFAULT_PACKET_RELATIVE,
            field="engineering route summary",
        )
    if isinstance(value, Mapping):
        return dict(value)
    return _read_json(
        Path(value).expanduser().resolve(), field="engineering route summary"
    )


def _load_device_receipt(
    value: Mapping[str, Any] | str | Path | None,
    *,
    repository_root: Path,
) -> dict[str, Any] | None:
    if value is None:
        default = repository_root / _DEFAULT_DEVICE_RECEIPT_RELATIVE
        return (
            _read_json(default, field="sealed device selection receipt")
            if default.is_file()
            else None
        )
    if isinstance(value, Mapping):
        return dict(value)
    return _read_json(
        _resolve_path(value, repository_root=repository_root),
        field="sealed device selection receipt",
    )


def _route_identity(
    summary: Mapping[str, Any],
    *,
    device_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Extract and cross-check exact route/artifact/device identity."""

    physical = _mapping(summary.get("physical_route"), field="summary.physical_route")
    component = _mapping(
        summary.get("component_identities"), field="summary.component_identities"
    )
    artifact = _mapping(summary.get("native_artifact"), field="summary.native_artifact")
    summary_device = _mapping(
        artifact.get("device"), field="summary.native_artifact.device"
    )
    receipt = {} if device_receipt is None else dict(device_receipt)
    receipt_device = (
        _mapping(receipt.get("device_identity"), field="device_receipt.device_identity")
        if receipt
        else {}
    )

    def choose(
        field: str,
        *values: object,
        required: bool = True,
    ) -> object:
        selected = next((value for value in values if value is not None), None)
        if required and selected is None:
            raise ScienceCudaV2HostContextError(f"route summary lacks {field}")
        for alternate in values:
            if alternate is not None and selected is not None and alternate != selected:
                raise ScienceCudaV2HostContextError(
                    f"route summary identity mismatch for {field}"
                )
        return selected

    route_id = _text(
        choose(
            "route_id",
            physical.get("native_route_id"),
            physical.get("route_id"),
            component.get("native_route"),
            receipt.get("route_id"),
        ),
        field="route_id",
    )
    profile_id = _text(
        choose(
            "profile_id",
            physical.get("profile_id"),
            component.get("native_profile"),
            receipt.get("profile_id"),
        ),
        field="profile_id",
    )
    carrier_profile_id = _text(
        choose("carrier_profile_id", physical.get("carrier_profile_id")),
        field="carrier_profile_id",
    )
    provider_id = _text(
        choose("provider_id", physical.get("provider_id"), receipt.get("provider_id")),
        field="provider_id",
    )
    provider_family = _text(
        choose("provider_family", physical.get("provider_family"), required=False)
        or "cuda_full_evolution_v2",
        field="provider_family",
    )
    route_sha = _sha256(
        choose(
            "execution_route_sha256",
            physical.get("execution_route_sha256"),
            physical.get("route_sha256"),
            receipt.get("execution_route_sha256"),
            receipt.get("route_sha256"),
        ),
        field="execution_route_sha256",
    )
    binary_sha = _sha256(
        choose(
            "native_artifact.binary_sha256",
            artifact.get("binary_sha256"),
            _mapping(receipt.get("artifact"), field="device_receipt.artifact").get(
                "binary_sha256"
            )
            if receipt
            else None,
        ),
        field="binary_sha256",
    )
    manifest_sha = _sha256(
        choose(
            "native_artifact.manifest_sha256",
            artifact.get("manifest_sha256"),
            _mapping(receipt.get("artifact"), field="device_receipt.artifact").get(
                "manifest_sha256"
            )
            if receipt
            else None,
        ),
        field="manifest_sha256",
    )
    manifest_file_sha = _sha256(
        choose(
            "native_artifact.manifest_file_sha256",
            artifact.get("manifest_file_sha256"),
            _mapping(receipt.get("artifact"), field="device_receipt.artifact").get(
                "manifest_file_sha256"
            )
            if receipt
            else None,
        ),
        field="manifest_file_sha256",
    )
    artifact_path = _text(
        choose("native_artifact.path", artifact.get("path"), required=False)
        or "unresolved-native-artifact-path",
        field="native_artifact.path",
    )
    device: dict[str, Any] = {}
    for key in sorted(set(summary_device) | set(receipt_device)):
        value = choose(
            f"device.{key}",
            summary_device.get(key),
            receipt_device.get(key),
            required=False,
        )
        if value is not None:
            device[key] = value
    if not device:
        raise ScienceCudaV2HostContextError(
            "route summary lacks native device identity"
        )
    receipt_sha = receipt.get("receipt_sha256")
    return {
        "route_id": route_id,
        "profile_id": profile_id,
        "carrier_profile_id": carrier_profile_id,
        "provider_id": provider_id,
        "provider_family": provider_family,
        "execution_route_sha256": route_sha,
        "binary_sha256": binary_sha,
        "manifest_sha256": manifest_sha,
        "manifest_file_sha256": manifest_file_sha,
        "artifact_path": artifact_path,
        "device": device,
        "device_selection_receipt_sha256": (
            None
            if receipt_sha is None
            else _digest(receipt_sha, field="receipt_sha256")
        ),
        "native_abi": _text(
            choose("native_abi", receipt.get("native_abi"), required=False)
            or CUDA_NATIVE_ABI_V1,
            field="native_abi",
        ),
    }


def _control_descriptor() -> ScienceControlDescriptorV1:
    return ScienceControlDescriptorV1(
        control_id="mutation.enabled",
        value_contract=_payload(
            "eiel_science_boolean_control_v1",
            {"kind": "boolean", "values": [False, True]},
        ),
        unit="boolean",
        activation_phases=("mutation",),
        heterogeneity_modes=("global",),
        schedule_kinds=("static",),
        readback_capability_ids=("control_readback",),
        interaction_ids=("genome_v2_mutation",),
        component_id="cuda_full_evolution_v2.mutation",
        qualified_envelope_subjects=("controls.mutation.enabled",),
    )


def _observation_profile_descriptor() -> ScienceObservationProfileDescriptorV1:
    registry = science_capture_producer_phase_registry_v1()
    return ScienceObservationProfileDescriptorV1(
        profile=ScienceObservationProfileV1.STANDARD,
        observation_abi_version=SchemaVersion("eiel_science_observation_abi_v1"),
        record_kinds=tuple(ScienceRecordKindV1),
        required_nonempty_record_kinds=(
            ScienceRecordKindV1.CONTROL_BINDING,
            ScienceRecordKindV1.WORLD_COMPLETION,
        ),
        capability_ids=_OBSERVATION_CAPABILITY_IDS,
        producer_phase_registry_ordinals=tuple(sorted(registry.items())),
        default_partition_span_ticks=1000,
    )


def _resource_envelope(*, logical_world_count: int) -> ProfileResourceEnvelopeV1:
    kinds = tuple(sorted(ScienceRecordKindV1, key=lambda item: item.value))
    max_records = 512
    max_record_bytes = 65_536
    chunk_max_records = logical_world_count * max_records * len(kinds)
    chunk_max_bytes = 4 * 1024 * 1024
    max_in_flight_chunks = 2
    fixed_buffer_bytes = 1024
    per_world_cursor_bytes = 256
    per_agent_bytes = 1
    resident_bytes = (
        fixed_buffer_bytes
        + max_in_flight_chunks * chunk_max_bytes
        + logical_world_count * per_world_cursor_bytes
        + logical_world_count * RESIDENT_CAPACITY_C_V2 * per_agent_bytes
    )
    return ProfileResourceEnvelopeV1(
        profile=ScienceObservationProfileV1.STANDARD,
        max_records_per_tick_per_world_by_kind=tuple(
            (kind, max_records) for kind in kinds
        ),
        max_record_bytes_by_kind=tuple((kind, max_record_bytes) for kind in kinds),
        chunk_max_records=chunk_max_records,
        chunk_max_bytes=chunk_max_bytes,
        max_in_flight_chunks=max_in_flight_chunks,
        fixed_buffer_bytes=fixed_buffer_bytes,
        per_world_cursor_bytes=per_world_cursor_bytes,
        per_resident_agent_observation_state_bytes=per_agent_bytes,
        logical_world_count=logical_world_count,
        resident_capacity=RESIDENT_CAPACITY_C_V2,
        max_resident_observation_bytes=resident_bytes,
    )


def _route_adapter() -> ScienceQRRRouteAdapterV1:
    return ScienceQRRRouteAdapterV1(
        route_id=CUDA_NATIVE_ROUTE_V2,
        machine_profile_id=BASELINE_PROFILE_ID,
        world_profile_id=SCIENCE_CUDA_V2_W32_WORLD_PROFILE_ID,
        world_width=WORLD_WIDTH_V2,
        maximum_batch_cells=136,
        task_seed_owner="task_environment",
        control_path_by_id=(("mutation.enabled", "cuda_native.mutation_enabled"),),
        seed_path_by_owner=tuple(
            (owner, f"cuda_native.seed.{owner}")
            for owner in sorted(
                (
                    "ecology",
                    "founder",
                    "policy",
                    "mutation",
                    "initialization",
                    "intervention",
                )
            )
        ),
        fixed_flat_controls=_payload(
            "eiel_science_qrr_fixed_controls_v1",
            {
                "cuda_native.event_transport": "direct_sparse_v3",
                "cuda_native.execution_mode": "cuda_graph_monolith",
                "cuda_native.science_observation_mode": "TYPED_SIDECAR_V1",
                "cuda_native.temporal_k": 1,
            },
        ),
        acs_capture_instrumentation=False,
        acs_checkpoint_slots=0,
        living_lineage_canopy=False,
        lifecycle_total_max_bytes=8 * 1024 * 1024 * 1024,
    )


def _manifest(
    control: ScienceControlDescriptorV1,
    reducer: ScienceReducerDescriptorV1,
    profile: ScienceObservationProfileDescriptorV1,
    *,
    observation_abi_sha256: ContentDigest,
) -> ScienceInstrumentManifestV1:
    world_profiles = {
        "world_profile_id": SCIENCE_CUDA_V2_W32_WORLD_PROFILE_ID,
        "carrier_profile_id": SCIENCE_CUDA_V2_W32_CARRIER_PROFILE_ID,
        "native_profile_id": CUDA_NATIVE_PROFILE_ID_V2,
        "geometry": {"width": WORLD_WIDTH_V2, "height": WORLD_WIDTH_V2},
        "initial_population": INITIAL_POPULATION_V2,
        "resident_capacity": RESIDENT_CAPACITY_C_V2,
    }
    component_graph = {
        "instrument": "EIEL_SCIENCE_INSTRUMENT_V1",
        "world_profile": world_profiles,
        "founder_profiles": list(SCIENCE_CUDA_V2_W32_FOUNDER_PROFILE_IDS),
        "placement_profile": SCIENCE_CUDA_V2_W32_PLACEMENT_PROFILE_ID,
        "variation_profile": SCIENCE_CUDA_V2_W32_VARIATION_PROFILE_ID,
        "policy_profile": SCIENCE_CUDA_V2_W32_POLICY_PROFILE_ID,
        "mutation_operator": SCIENCE_CUDA_V2_W32_MUTATION_OPERATOR_ID,
        "control_descriptor": control.descriptor_sha256().to_dict(),
        "reducer_descriptor": reducer.descriptor_sha256().to_dict(),
        "observation_profile_descriptor": profile.descriptor_sha256().to_dict(),
        "native_capabilities": list(FULL_EVOLUTION_CAPABILITIES_V2),
    }
    return ScienceInstrumentManifestV1(
        component_graph_sha256=canonical_sha256(component_graph),
        control_catalog_sha256=science_control_catalog_sha256((control,)),
        world_profile_catalog_sha256=canonical_sha256(
            [world_profiles, SCIENCE_CUDA_V2_W32_CARRIER_PROFILE_ID]
        ),
        founder_profile_catalog_sha256=canonical_sha256(
            list(SCIENCE_CUDA_V2_W32_FOUNDER_PROFILE_IDS)
        ),
        policy_profile_catalog_sha256=canonical_sha256(
            [SCIENCE_CUDA_V2_W32_POLICY_PROFILE_ID]
        ),
        mutation_operator_catalog_sha256=canonical_sha256(
            [SCIENCE_CUDA_V2_W32_MUTATION_OPERATOR_ID]
        ),
        observation_abi_sha256=observation_abi_sha256,
        observation_profile_catalog_sha256=(
            science_observation_profile_catalog_sha256_v1((profile,))
        ),
        reducer_protocol_version=SchemaVersion("eiel_science_reducer_protocol_v1"),
        route_binding_protocol_version=SchemaVersion("eiel_science_route_binding_v1"),
        eed_release_status=EEDReleaseStatusV1.EXCLUDED,
    )


def _no_qualification_authority(route_id: str) -> ContentDigest:
    return canonical_sha256(
        {
            "schema_version": "eiel_science_no_qualification_authority_v1",
            "route_id": route_id,
            "status": "NO_QUALIFICATION_AUTHORITY",
            "scope": "engineering_and_exploratory_science_only",
        }
    )


def _crca_binding() -> ScienceCRCASemanticBindingV1:
    return _crca_binding_for_schedule(
        total_ticks=SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS,
        checkpoint_ticks=SCIENCE_CUDA_V2_MUTSUP_CHECKPOINT_TICKS,
        observation_ticks=SCIENCE_CUDA_V2_MUTSUP_OBSERVATION_TICKS,
        study_id=SCIENCE_CUDA_V2_MUTSUP_STUDY_ID,
    )


def _crca_binding_for_schedule(
    *,
    total_ticks: int,
    checkpoint_ticks: tuple[int, ...],
    observation_ticks: tuple[int, ...],
    study_id: str,
) -> ScienceCRCASemanticBindingV1:
    semantic_core = {
        "schema_version": "eiel_crca_mutsup_semantic_binding_v1",
        "test_id": SCIENCE_CUDA_V2_MUTSUP_TEST_ID,
        "test_version": SCIENCE_CUDA_V2_MUTSUP_TEST_VERSION,
        "study_id": study_id,
        "absolute_start_next_tick": 0,
        "absolute_target_next_tick": total_ticks,
        "trajectory_checkpoint_ticks": [0, *checkpoint_ticks],
        "observation_ticks": list(observation_ticks),
        "authority": SHADOW_PROJECTION_AUTHORITY,
    }
    semantic_digest = canonical_sha256(semantic_core)
    projection_core = {
        "schema_version": "eiel_crca_shadow_mutsup_projection_v1",
        "authority": SHADOW_PROJECTION_AUTHORITY,
        "layers": {
            "semantic": {
                "payload": semantic_core,
                "digest": semantic_digest.to_dict(),
            }
        },
    }
    return ScienceCRCASemanticBindingV1(
        semantic_run_digest=semantic_digest,
        projection_sha256=canonical_sha256(projection_core),
        absolute_start_next_tick=0,
        absolute_target_next_tick=total_ticks,
        trajectory_checkpoint_ticks=(0, *checkpoint_ticks),
    )


def _file_content_digest(path: Path, *, field: str) -> ContentDigest:
    if path.is_symlink() or not path.is_file():
        raise ScienceCudaV2HostContextError(f"{field} is not a regular file: {path}")
    return ContentDigest("sha256", hashlib.sha256(path.read_bytes()).hexdigest())


def _observation_adapter_digest(repository_root: Path) -> ContentDigest:
    dependencies = [
        {
            "path": relative,
            "sha256": _file_content_digest(
                repository_root / relative,
                field=f"observation adapter dependency {relative}",
            ).to_dict(),
        }
        for relative in _SCIENCE_OBSERVATION_ADAPTER_DEPENDENCIES
    ]
    return canonical_sha256(
        {
            "schema_version": "eiel_science_observation_adapter_dependency_closure_v1",
            "component": "eiel_cuda_full_evolution_science_capture_adapter_v1",
            "dependencies": dependencies,
            "producer_phase_registry": science_capture_producer_phase_registry_v1(),
        }
    )


def _observation_abi_digest(repository_root: Path) -> ContentDigest:
    return _file_content_digest(
        repository_root / _SCIENCE_OBSERVATION_ABI_PATH,
        field="science observation ABI",
    )


def _physical_core(
    identity: Mapping[str, Any],
    *,
    temporal_k: int = 1,
    logical_bundle_size: int = 1024,
    target_physical_batch: int = 1024,
) -> CanonicalPayloadV1:
    device = dict(identity["device"])
    payload: dict[str, Any] = {
        "schema_version": str(PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION),
        "route_id": identity["route_id"],
        "provider_id": identity["provider_id"],
        "provider_family": identity["provider_family"],
        "native_profile_id": identity["profile_id"],
        "carrier_profile_id": identity["carrier_profile_id"],
        "native_abi": identity["native_abi"],
        "execution_backend": "direct_sparse_v3",
        "execution_mode": "cuda_graph_monolith",
        "temporal_k": temporal_k,
        "geometry": {
            "width": WORLD_WIDTH_V2,
            "height": WORLD_WIDTH_V2,
            "initial_population": INITIAL_POPULATION_V2,
            "resident_capacity": RESIDENT_CAPACITY_C_V2,
        },
        "target_execution_quantum": {
            "logical_bundle_size": logical_bundle_size,
            "target_physical_batch": target_physical_batch,
        },
        "route_profile_sha256": identity["execution_route_sha256"].to_dict(),
        "route_lineage": {
            "current_host_execution_route_sha256": identity[
                "execution_route_sha256"
            ].to_dict(),
            **(
                {
                    "retained_engineering_packet_route_sha256": identity[
                        "retained_engineering_packet_route_sha256"
                    ].to_dict(),
                    "identity_rule": (
                        "current route binds live host execution sources while the retained "
                        "packet route preserves the accepted pre-integration provenance"
                    ),
                }
                if identity.get("retained_engineering_packet_route_sha256")
                is not None
                else {
                    "identity_rule": (
                        "current route binds the compiled plan, native artifact, sealed "
                        "device selection, and live host execution closure"
                    )
                }
            ),
        },
        "native_artifact": {
            "path": identity["artifact_path"],
            "binary_sha256": identity["binary_sha256"].to_dict(),
            "manifest_sha256": identity["manifest_sha256"].to_dict(),
            "manifest_file_sha256": identity["manifest_file_sha256"].to_dict(),
        },
        "device_identity": device,
    }
    if identity.get("device_selection_receipt_sha256") is not None:
        payload["device_selection_receipt_sha256"] = identity[
            "device_selection_receipt_sha256"
        ].to_dict()
    return _payload(str(PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION), payload)


def _compile_w32_cuda_v2_base_plan_payload_v1(
    *,
    total_ticks: int,
    checkpoint_ticks: tuple[int, ...],
) -> dict[str, Any]:
    """Compile the canonical base plan without depending on host-context identity."""

    if (
        type(total_ticks) is not int
        or total_ticks < 1
        or not checkpoint_ticks
        or tuple(checkpoint_ticks) != tuple(sorted(set(checkpoint_ticks)))
        or any(
            type(tick) is not int or tick <= 0 or tick > total_ticks
            for tick in checkpoint_ticks
        )
        or checkpoint_ticks[-1] != total_ticks
    ):
        raise ScienceCudaV2HostContextError(
            "base plan timing must use sorted positive absolute ticks ending at total_ticks"
        )

    source = _default_repository_root() / _DEFAULT_BASE_PLAN_SPEC_RELATIVE
    try:
        spec = load_campaign_spec(source)
    except Exception as error:
        raise ScienceCudaV2HostContextError(
            f"checked-in native base campaign spec is not loadable: {source}"
        ) from error

    source_total = int(spec["execution"]["total_ticks"])
    source_checkpoints = tuple(spec["execution"]["checkpoint_ticks"])
    if total_ticks != source_total or tuple(checkpoint_ticks) != source_checkpoints:
        spec = deepcopy(spec)
        execution = spec["execution"]
        execution["total_ticks"] = total_ticks
        execution["checkpoint_ticks"] = list(checkpoint_ticks)
        technical_gate = execution["technical_gate"]
        technical_gate["total_ticks"] = total_ticks
        technical_gate["checkpoint_ticks"] = list(checkpoint_ticks)
        tick_control = execution["tick_native_control"]
        tick_control["total_ticks"] = total_ticks
        tick_control["checkpoint_ticks"] = list(checkpoint_ticks)
        schedules = tick_control["schedules"]
        schedules["checkpoint_ticks"] = list(checkpoint_ticks)
        for schedule_name in (
            "report_ticks",
            "science_sample_ticks",
            "reducer_ticks",
            "archive_ticks",
        ):
            schedules[schedule_name] = [total_ticks]

    # Always project the maintained lifecycle profile rather than trusting a
    # hand-authored copy in the campaign input. This preserves the repository's
    # exact R0-R2 release policy when timing matches the checked-in 10k default.
    execution = spec["execution"]
    budgets = execution["budgets"]
    budgets["wall_seconds"] = max(
        int(budgets["wall_seconds"]),
        cuda_full_evolution_v2_wall_seconds_upper_bound_v1(total_ticks),
    )
    spec["artifact_lifecycle"] = lifecycle_profile_contract_v1(
        "SCIENCE_COMPACT",
        total_max_bytes=execution["budgets"]["disk_bytes"],
        checkpoint_schedule=checkpoint_ticks,
        checkpoint_authority="absolute_tick",
    )

    try:
        plan = compile_campaign(
            spec,
            registry=cuda_full_evolution_v2_profile_registry_v1(),
        )
    except Exception as error:
        raise ScienceCudaV2HostContextError(
            "canonical CUDA V2 base campaign compilation failed"
        ) from error
    if plan.profile_id != CUDA_NATIVE_PROFILE_ID_V2:
        raise ScienceCudaV2HostContextError(
            "compiled base plan resolved to a different native profile"
        )
    return plan.payload()


def _bind_current_execution_route_v1(
    identity: Mapping[str, Any],
    *,
    total_ticks: int,
    checkpoint_ticks: tuple[int, ...],
) -> dict[str, Any]:
    """Replace the retained packet route with the exact live host route identity."""

    plan = _compile_w32_cuda_v2_base_plan_payload_v1(
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoint_ticks,
    )
    try:
        route = CudaFullEvolutionV2ExecutionProviderV1().execution_route_contract(plan)
    except Exception as error:
        raise ScienceCudaV2HostContextError(
            "current CUDA V2 execution-route identity is unavailable"
        ) from error
    components = {
        str(row.get("component_id")): dict(row)
        for row in route.components
        if isinstance(row, Mapping)
    }
    artifact = components.get("cuda_full_evolution_v2_artifact_unit")
    device = components.get("cuda_full_evolution_v2_device_selection")
    if artifact is None or device is None:
        raise ScienceCudaV2HostContextError(
            "current CUDA V2 route lacks artifact or device identity"
        )
    expected_artifact = {
        "binary_sha256": identity["binary_sha256"].hex,
        "manifest_sha256": identity["manifest_sha256"].hex,
        "manifest_file_sha256": identity["manifest_file_sha256"].hex,
    }
    if any(artifact.get(key) != value for key, value in expected_artifact.items()):
        raise ScienceCudaV2HostContextError(
            "current CUDA V2 route artifact differs from the retained engineering packet"
        )
    receipt = identity.get("device_selection_receipt_sha256")
    if receipt is not None and device.get("selection_receipt_sha256") != receipt.hex:
        raise ScienceCudaV2HostContextError(
            "current CUDA V2 route device receipt differs from the sealed input"
        )
    result = dict(identity)
    result["retained_engineering_packet_route_sha256"] = identity[
        "execution_route_sha256"
    ]
    result["execution_route_sha256"] = ContentDigest(
        "sha256", route.execution_route_sha256
    )
    return result


def materialize_w32_cuda_v2_science_host_context_v1(
    summary: Mapping[str, Any] | str | Path | None = None,
    *,
    device_receipt: Mapping[str, Any] | str | Path | None = None,
    logical_world_count: int = 6,
    total_ticks: int = SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS,
    checkpoint_ticks: tuple[int, ...] | None = None,
    observation_ticks: tuple[int, ...] | None = None,
) -> ScienceInstrumentHostContextV1:
    """Build the exact reusable W32/SCIENCE_STANDARD host context."""

    if type(logical_world_count) is not int or logical_world_count != 6:
        raise ScienceCudaV2HostContextError(
            "the MUTSUP host context is fixed to six paired worlds"
        )
    if type(total_ticks) is not int or total_ticks < 1:
        raise ScienceCudaV2HostContextError(
            "total_ticks must be a positive absolute-tick horizon"
        )
    if checkpoint_ticks is None:
        checkpoint_ticks = (
            SCIENCE_CUDA_V2_MUTSUP_CHECKPOINT_TICKS
            if total_ticks == SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS
            else (total_ticks,)
        )
    checkpoint_ticks = tuple(checkpoint_ticks)
    if (
        not checkpoint_ticks
        or checkpoint_ticks != tuple(sorted(set(checkpoint_ticks)))
        or any(
            type(tick) is not int or tick <= 0 or tick > total_ticks
            for tick in checkpoint_ticks
        )
        or checkpoint_ticks[-1] != total_ticks
    ):
        raise ScienceCudaV2HostContextError(
            "checkpoint_ticks must be sorted positive absolute ticks ending at total_ticks"
        )
    if observation_ticks is None:
        observation_ticks = tuple(
            tick
            for tick in SCIENCE_CUDA_V2_MUTSUP_OBSERVATION_TICKS
            if tick <= total_ticks
        )
        if not observation_ticks or observation_ticks[-1] != total_ticks:
            observation_ticks = (*observation_ticks, total_ticks)
    observation_ticks = tuple(observation_ticks)
    if (
        observation_ticks != tuple(sorted(set(observation_ticks)))
        or any(
            type(tick) is not int or tick < 0 or tick > total_ticks
            for tick in observation_ticks
        )
        or not observation_ticks
        or observation_ticks[0] != 0
        or observation_ticks[-1] != total_ticks
    ):
        raise ScienceCudaV2HostContextError(
            "observation_ticks must be sorted absolute ticks from zero through total_ticks"
        )
    study_id = (
        SCIENCE_CUDA_V2_MUTSUP_STUDY_ID
        if total_ticks == SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS
        else f"MUTSUP-W32-PAIRED-SEED-{total_ticks}-CANARY"
    )
    loaded_summary = _load_summary(summary)
    repository_root = _default_repository_root()
    loaded_receipt = _load_device_receipt(
        device_receipt,
        repository_root=repository_root,
    )
    identity = _route_identity(loaded_summary, device_receipt=loaded_receipt)
    expected = {
        "route_id": CUDA_NATIVE_ROUTE_V2,
        "profile_id": CUDA_NATIVE_PROFILE_ID_V2,
        "carrier_profile_id": SCIENCE_CUDA_V2_W32_CARRIER_PROFILE_ID,
        "native_abi": CUDA_NATIVE_ABI_V1,
    }
    for field, expected_value in expected.items():
        if identity[field] != expected_value:
            raise ScienceCudaV2HostContextError(
                f"sealed route is not the canonical W32 CUDA V2 {field}: "
                f"{identity[field]!r}"
            )
    if identity["execution_route_sha256"].hex != (
        _RETAINED_ENGINEERING_PACKET_ROUTE_PROFILE_SHA256
    ):
        raise ScienceCudaV2HostContextError(
            "route summary identity mismatch for execution_route_sha256"
        )
    identity = _bind_current_execution_route_v1(
        identity,
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoint_ticks,
    )
    control = _control_descriptor()
    reducer = mutsup_direct_retention_reducer_descriptor_v1()
    profile = _observation_profile_descriptor()
    resource = _resource_envelope(logical_world_count=logical_world_count)
    adapter = _route_adapter()
    observation_adapter_sha256 = _observation_adapter_digest(repository_root)
    observation_abi_sha256 = _observation_abi_digest(repository_root)
    manifest = _manifest(
        control,
        reducer,
        profile,
        observation_abi_sha256=observation_abi_sha256,
    )
    no_authority = _no_qualification_authority(identity["route_id"])
    qualified = ScienceQualifiedEnvelopeProjectionV1(
        instrument_manifest_sha256=manifest.manifest_sha256(),
        route_id=identity["route_id"],
        machine_profile_id=BASELINE_PROFILE_ID,
        qualification_authority_sha256=no_authority,
        rows=(),
    )
    capability_projection = canonical_sha256(
        {
            "native_profile_id": identity["profile_id"],
            "native_route_id": identity["route_id"],
            "native_abi": identity["native_abi"],
            "carrier_profile_id": identity["carrier_profile_id"],
            "capabilities": list(FULL_EVOLUTION_CAPABILITIES_V2),
            "geometry": {
                "width": WORLD_WIDTH_V2,
                "height": WORLD_WIDTH_V2,
                "initial_population": INITIAL_POPULATION_V2,
                "resident_capacity": RESIDENT_CAPACITY_C_V2,
            },
        }
    )
    checkpoint_adapter = canonical_sha256(
        {
            "contract": "eiel_cuda_full_evolution_science_checkpoint_extension_v1",
            "frontier": "ScienceObservationPrefixCursorV1",
            "time_authority": "absolute_tick_v1",
            "checkpoint_ticks": list(checkpoint_ticks),
        }
    )
    semantic_conformance = canonical_sha256(
        {
            "native_profile_id": identity["profile_id"],
            "route_id": identity["route_id"],
            "time_authority": "absolute_tick_v1",
            "start_next_tick": 0,
            "total_ticks": total_ticks,
            "checkpoint_ticks": list(checkpoint_ticks),
            "status": "ENGINEERING_AND_EXPLORATORY_SCIENCE_ONLY",
        }
    )
    route_binding = ScienceRouteBindingV1(
        instrument_manifest_sha256=manifest.manifest_sha256(),
        route_id=identity["route_id"],
        route_profile_sha256=identity["execution_route_sha256"],
        machine_profile_id=BASELINE_PROFILE_ID,
        capability_projection_sha256=capability_projection,
        native_build_closure_sha256=identity["manifest_sha256"],
        execution_adapter_sha256=adapter.execution_adapter_sha256(),
        control_adapter_sha256=adapter.control_adapter_sha256(),
        observation_adapter_sha256=observation_adapter_sha256,
        checkpoint_adapter_sha256=checkpoint_adapter,
        semantic_conformance_sha256=semantic_conformance,
        resource_envelope_sha256=canonical_sha256(resource.to_dict()),
        qualified_envelope_sha256=qualified.projection_sha256(),
        qualification_authority_sha256=no_authority,
    )
    compiler_sha256 = canonical_sha256(
        {
            "component": "eiel_composition.compile_science_campaign_v1",
            "campaign_schema": "eiel_science_campaign_v1",
            "version": 1,
        }
    )
    machine_profile_sha256 = canonical_sha256(
        {
            "profile_id": BASELINE_PROFILE_ID,
            "identity_scope": "semantic_machine_profile",
        }
    )
    admission_kernel_sha256 = canonical_sha256(
        {
            "schema_version": "eiel_attempt_admission_kernel_v1",
            "authority": "canonical_crca_attempt_admission",
            "status": "HOST_CONTEXT_BINDING_ONLY",
            "route_id": identity["route_id"],
        }
    )
    scientific_capabilities = tuple(
        sorted(
            {
                f"world_profile:{SCIENCE_CUDA_V2_W32_WORLD_PROFILE_ID}",
                *(
                    f"founder_profile:{item}"
                    for item in SCIENCE_CUDA_V2_W32_FOUNDER_PROFILE_IDS
                ),
                f"placement_profile:{SCIENCE_CUDA_V2_W32_PLACEMENT_PROFILE_ID}",
                f"variation_profile:{SCIENCE_CUDA_V2_W32_VARIATION_PROFILE_ID}",
                "control:mutation.enabled",
                "schedule:static",
            }
        )
    )
    observation_capabilities = tuple(sorted(set(_OBSERVATION_CAPABILITY_IDS)))
    return ScienceInstrumentHostContextV1(
        instrument_manifest=manifest,
        route_binding=route_binding,
        route_adapter=adapter,
        control_descriptors=(control,),
        reducer_descriptors=(reducer,),
        observation_profile_descriptors=(profile,),
        crca_semantic_binding=_crca_binding_for_schedule(
            total_ticks=total_ticks,
            checkpoint_ticks=checkpoint_ticks,
            observation_ticks=observation_ticks,
            study_id=study_id,
        ),
        physical_execution_core=_physical_core(identity),
        resource_envelope=resource,
        compiler_component_sha256=compiler_sha256,
        qualified_envelope=qualified,
        available_scientific_capability_ids=scientific_capabilities,
        available_observation_capability_ids=observation_capabilities,
        machine_profile_sha256=machine_profile_sha256,
        admission_kernel_receipt_sha256=admission_kernel_sha256,
    )


def load_mutsup_w32_authoring_spec_v1(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the compact, hash-free MUTSUP authoring intent."""

    source = (
        _default_repository_root() / _DEFAULT_AUTHORING_SPEC_RELATIVE
        if path is None
        else Path(path).expanduser().resolve()
    )
    if source.is_symlink() or not source.is_file():
        raise ScienceCudaV2HostContextError(
            f"authoring spec is not a regular file: {source}"
        )
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ScienceCudaV2HostContextError(
            f"authoring spec is unreadable: {source}"
        ) from error
    spec = _mapping(value, field="authoring_spec")
    if spec.get("schema_version") != SCIENCE_CUDA_V2_MUTSUP_AUTHORING_SCHEMA_VERSION:
        raise ScienceCudaV2HostContextError("authoring spec schema is unsupported")
    required = {
        "schema_version",
        "request_id",
        "campaign_id",
        "question",
        "task_seeds",
        "mutation_values",
        "total_ticks",
        "checkpoint_ticks",
        "observation_milestones",
        "artifact_lifecycle_profile",
        "qualification_claims",
    }
    if set(spec) != required:
        raise ScienceCudaV2HostContextError(
            f"authoring spec fields are not exact: {sorted(set(spec) ^ required)}"
        )
    if spec["request_id"] != SCIENCE_CUDA_V2_MUTSUP_REQUEST_ID:
        raise ScienceCudaV2HostContextError("authoring spec request identity differs")
    if spec["campaign_id"] != SCIENCE_CUDA_V2_MUTSUP_CAMPAIGN_ID:
        raise ScienceCudaV2HostContextError("authoring spec campaign identity differs")
    if tuple(spec["task_seeds"]) != SCIENCE_CUDA_V2_MUTSUP_TASK_SEEDS:
        raise ScienceCudaV2HostContextError("authoring spec task seeds differ")
    if spec["mutation_values"] != [False, True]:
        raise ScienceCudaV2HostContextError(
            "authoring spec mutation values must be [false, true]"
        )
    if spec["total_ticks"] != SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS:
        raise ScienceCudaV2HostContextError("authoring spec total_ticks differs")
    if tuple(spec["checkpoint_ticks"]) != SCIENCE_CUDA_V2_MUTSUP_CHECKPOINT_TICKS:
        raise ScienceCudaV2HostContextError("authoring spec checkpoint ticks differ")
    if (
        tuple(spec["observation_milestones"])
        != SCIENCE_CUDA_V2_MUTSUP_OBSERVATION_TICKS
    ):
        raise ScienceCudaV2HostContextError(
            "authoring spec observation milestones differ"
        )
    if spec["artifact_lifecycle_profile"] != "SCIENCE_COMPACT":
        raise ScienceCudaV2HostContextError("MUTSUP authoring requires SCIENCE_COMPACT")
    if spec["qualification_claims"] is not False:
        raise ScienceCudaV2HostContextError(
            "MUTSUP authoring must make no qualification claims"
        )
    _text(spec["question"], field="authoring_spec.question")
    return spec


def _mutsup_campaign_document(
    manifest: ScienceInstrumentManifestV1,
    reducer: ScienceReducerDescriptorV1,
    *,
    reducer_configuration: Mapping[str, Any],
    task_seeds: tuple[int, ...],
    campaign_id: str,
    study_id: str,
    total_ticks: int,
    checkpoint_ticks: tuple[int, ...],
    observation_ticks: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "schema_version": "eiel_science_campaign_v1",
        "product_id": "EIEL_SCIENCE_INSTRUMENT_V1",
        "campaign_id": campaign_id,
        "instrument": {
            "product_id": "EIEL_SCIENCE_INSTRUMENT_V1",
            "instrument_manifest_sha256": manifest.manifest_sha256().to_dict(),
        },
        "question": {
            "question_id": "Q-MUTSUP-W32-PAIRED-SEED-10K",
            "hypothesis_ids": ["H-MUTATION-GENOME-RETENTION-REPRODUCTION"],
            "labels": [
                "MUTSUP",
                "paired-seed",
                "mutation-control",
                "exploratory-science",
            ],
            "description": (
                "Under the fixed native mutation law, do mutation-enabled worlds "
                "produce attributable genome changes whose carriers later reproduce "
                "versus paired mutation-off controls?"
            ),
            "preregistration_ref": None,
            "science_ownership": {
                "test_id": SCIENCE_CUDA_V2_MUTSUP_TEST_ID,
                "test_version": SCIENCE_CUDA_V2_MUTSUP_TEST_VERSION,
                "study_id": study_id,
                "arm_dimension_ids": ["mutation_enabled"],
                "replicate_dimension_id": "task_seed",
            },
            "no_claims": True,
        },
        "design": {
            "dimensions": [
                {
                    "dimension_id": "mutation_enabled",
                    "binding_path": "controls.mutation.enabled",
                    "canonical_values": [False, True],
                    "ordering": "declared",
                },
                {
                    "dimension_id": "task_seed",
                    "binding_path": "seeds.task_environment",
                    "canonical_values": list(task_seeds),
                    "ordering": "declared",
                },
            ],
            "pairing_groups": ["task_seed"],
            "repeat_groups": ["mutation_enabled"],
            "expected_cell_count": len(task_seeds) * 2,
        },
        "world": {
            "world_profile_id": SCIENCE_CUDA_V2_W32_WORLD_PROFILE_ID,
            "overlay_profile_ids": [],
            "geometry": {"width": WORLD_WIDTH_V2, "height": WORLD_WIDTH_V2},
            "resident_capacity": RESIDENT_CAPACITY_C_V2,
        },
        "initialization": {
            "initial_population": INITIAL_POPULATION_V2,
            "initial_live_count": INITIAL_POPULATION_V2,
            "founder_profiles": list(SCIENCE_CUDA_V2_W32_FOUNDER_PROFILE_IDS),
            "exact_founder_composition": [
                {
                    "founder_profile_id": SCIENCE_CUDA_V2_W32_FOUNDER_PROFILE_IDS[0],
                    "count": INITIAL_POPULATION_V2 // 2,
                },
                {
                    "founder_profile_id": SCIENCE_CUDA_V2_W32_FOUNDER_PROFILE_IDS[1],
                    "count": INITIAL_POPULATION_V2 // 2,
                },
            ],
            "spatial_placement_profile": SCIENCE_CUDA_V2_W32_PLACEMENT_PROFILE_ID,
            "standing_variation_profile": SCIENCE_CUDA_V2_W32_VARIATION_PROFILE_ID,
        },
        "controls": [
            {
                "control_id": "mutation.enabled",
                "dimension_binding": "mutation_enabled",
                "schedule": {"kind": "static"},
            }
        ],
        "seeds": [
            {"owner": "ecology", "binding_kind": "literal", "literal": 7001},
            {"owner": "founder", "binding_kind": "literal", "literal": 7002},
            {"owner": "policy", "binding_kind": "literal", "literal": 7003},
            {"owner": "mutation", "binding_kind": "literal", "literal": 8001},
            {
                "owner": "task_environment",
                "binding_kind": "dimension_bound",
                "dimension_id": "task_seed",
                "held_equal_in_pairing_groups": ["mutation_enabled"],
            },
        ],
        "time": {
            "authority": "absolute_tick_v1",
            "start_next_tick": 0,
            "total_ticks": total_ticks,
            "observation_ticks": list(observation_ticks),
            "checkpoint_ticks": list(checkpoint_ticks),
            "branches": [],
        },
        "observations": {
            "profile_id": "SCIENCE_STANDARD_V1",
            "required_capability_ids": list(_OBSERVATION_CAPABILITY_IDS),
            "event_observations": ["birth", "death", "mutation"],
            "state_observations": [
                "birth_parentage",
                "mutation_provenance",
                "genome_identity",
                "lineage_ancestry",
            ],
            "runtime_counters": ["birth_count", "mutation_count", "population_count"],
            "milestone_ticks": list(observation_ticks),
        },
        "execution": {
            "machine_profile_id": BASELINE_PROFILE_ID,
            "logical_world_count": len(task_seeds) * 2,
            "batching_policy": "authority_selected",
            "maximum_physical_batch_worlds": 136,
            "checkpoint_mode": "absolute_tick",
        },
        "reducers": [
            {
                "reducer_id": reducer.reducer_id,
                "contract_version": str(reducer.contract_version),
                "descriptor_sha256": reducer.descriptor_sha256().to_dict(),
                "configuration": dict(reducer_configuration),
                "required_observation_capability_ids": list(
                    reducer.required_observation_capability_ids
                ),
                "output_contract_id": reducer.output_contract_id,
            }
        ],
        "artifact_lifecycle": {
            "profile_id": "SCIENCE_COMPACT",
            "contract_ref": "eiel_run_artifact_lifecycle_contract_v1",
            "explicit_intent_allowed_override": None,
        },
        "decision": {
            "offline_decision_rule_ids": ["mutsup-direct-retention-descriptive-v1"],
            "expected_outputs": [MUTSUP_DIRECT_RETENTION_OUTPUT_CONTRACT_ID],
            "presentation_metadata": {
                "science_status": "EXPLORATORY_SCIENCE",
                "qualification_claims": False,
                "claim_boundary": "descriptive_direct_retention_only",
            },
        },
    }


def _world_ownership_configuration(
    worlds: tuple[Any, ...],
    *,
    milestone_ticks: tuple[int, ...],
) -> dict[str, Any]:
    retention_offsets = [tick for tick in milestone_ticks if tick > 0]
    if not retention_offsets:
        raise ScienceCudaV2HostContextError(
            "MUTSUP reduction requires at least one positive retention offset"
        )
    ownership: list[dict[str, Any]] = []
    for world in worlds:
        science = world.science.to_value()
        ownership.append(
            {
                "world_uid": world.semantic_world_sha256.to_dict(),
                "study_id": science["study_id"],
                "arm_id": science["arm_id"],
                "replicate": science["replicate"],
                "pair_id": f"task-seed-{science['replicate']}",
            }
        )
    ownership.sort(key=lambda row: row["world_uid"]["hex"])
    return {
        "schema_version": str(MUTSUP_DIRECT_RETENTION_CONFIGURATION_SCHEMA_VERSION),
        "mutation_observation_semantics": "applied_outcomes_only_complete",
        "world_ownership": ownership,
        "pairing": {
            "off_arm_ids": ["mutation_enabled=False"],
            "on_arm_ids": ["mutation_enabled=True"],
        },
        "milestone_offsets_ticks": retention_offsets,
    }


def build_mutsup_w32_paired_seed_request_v1(
    context: ScienceInstrumentHostContextV1 | None = None,
    *,
    task_seeds: tuple[int, ...] = SCIENCE_CUDA_V2_MUTSUP_TASK_SEEDS,
    total_ticks: int = SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS,
    checkpoint_ticks: tuple[int, ...] | None = None,
    observation_ticks: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Build one normalized-ready, six-world MUTSUP request with no hashes guessed."""

    if tuple(task_seeds) != SCIENCE_CUDA_V2_MUTSUP_TASK_SEEDS:
        raise ScienceCudaV2HostContextError(
            "MUTSUP request is fixed to the three paired task seeds"
        )
    if checkpoint_ticks is None:
        checkpoint_ticks = (
            SCIENCE_CUDA_V2_MUTSUP_CHECKPOINT_TICKS
            if total_ticks == SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS
            else (total_ticks,)
        )
    checkpoint_ticks = tuple(checkpoint_ticks)
    if observation_ticks is None:
        observation_ticks = tuple(
            tick
            for tick in SCIENCE_CUDA_V2_MUTSUP_OBSERVATION_TICKS
            if tick <= total_ticks
        )
        if not observation_ticks or observation_ticks[-1] != total_ticks:
            observation_ticks = (*observation_ticks, total_ticks)
    observation_ticks = tuple(observation_ticks)
    canary = total_ticks != SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS
    request_id = (
        SCIENCE_CUDA_V2_MUTSUP_REQUEST_ID
        if not canary
        else f"mutsup-w32-paired-seed-{total_ticks}-canary-v1"
    )
    campaign_id = request_id
    study_id = (
        SCIENCE_CUDA_V2_MUTSUP_STUDY_ID
        if not canary
        else f"MUTSUP-W32-PAIRED-SEED-{total_ticks}-CANARY"
    )
    host_context = context or materialize_w32_cuda_v2_science_host_context_v1(
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoint_ticks,
        observation_ticks=observation_ticks,
    )
    if context is not None and (
        host_context.crca_semantic_binding.absolute_target_next_tick != total_ticks
        or host_context.crca_semantic_binding.trajectory_checkpoint_ticks
        != (0, *checkpoint_ticks)
    ):
        raise ScienceCudaV2HostContextError(
            "supplied host context CRCA binding does not match request timing"
        )
    if host_context.crca_semantic_binding.absolute_target_next_tick != (
        host_context.crca_semantic_binding.absolute_start_next_tick + total_ticks
    ):
        raise ScienceCudaV2HostContextError(
            "host CRCA binding does not match the 10,000-tick request"
        )
    reducer = host_context.reducer_descriptors[0]
    if reducer.reducer_id != MUTSUP_DIRECT_RETENTION_REDUCER_ID:
        raise ScienceCudaV2HostContextError(
            "host context does not expose direct-retention MUTSUP"
        )
    # The world IDs are deterministic semantic identities.  Materialize once
    # with a schema-complete configuration, then replace that configuration
    # with explicit owner rows before returning the request.
    provisional = _mutsup_campaign_document(
        host_context.instrument_manifest,
        reducer,
        reducer_configuration={
            "schema_version": str(MUTSUP_DIRECT_RETENTION_CONFIGURATION_SCHEMA_VERSION),
            "mutation_observation_semantics": "applied_outcomes_only_complete",
            "world_ownership": [],
            "pairing": {
                "off_arm_ids": ["mutation_enabled=False"],
                "on_arm_ids": ["mutation_enabled=True"],
            },
            "milestone_offsets_ticks": list(observation_ticks),
        },
        task_seeds=tuple(task_seeds),
        campaign_id=campaign_id,
        study_id=study_id,
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoint_ticks,
        observation_ticks=observation_ticks,
    )
    baseline_declaration = {
        "schema_version": "eiel_baseline_declaration_v1",
        "profile_id": BASELINE_PROFILE_ID,
        "intent": "development",
        "deviations": [
            {
                "requirement_id": requirement_id,
                "reason": reason,
            }
            for requirement_id, reason in (
                (
                    "evolution.full_evolution",
                    "CUDA Native V2 preserves the seven Full Evolution capabilities "
                    "under a new unqualified transition route.",
                ),
                (
                    "runtime.production_cuda",
                    "The CUDA Native V2 engineering provider is not the qualified "
                    "production provider.",
                ),
                (
                    "science.wrc_variability",
                    "This exploratory W32 study has no admitted WRC science envelope.",
                ),
                (
                    "checkpoint.acs_two_slot",
                    "This branch uses the native ACS0 absolute-tick checkpoint route; "
                    "ACS2 integration is a separate future branch delta.",
                ),
                (
                    "authority.device_fallback",
                    "A sealed exact-device CUDA V2 receipt and FORBIDDEN instrument "
                    "fallback policy replace generic fallback authority.",
                ),
            )
        ],
    }
    if tuple(
        row["requirement_id"] for row in baseline_declaration["deviations"]
    ) != CUDA_FULL_EVOLUTION_V2_BASELINE_DEVIATION_REQUIREMENT_IDS_V1:
        raise ScienceCudaV2HostContextError(
            "CUDA V2 engineering Baseline declaration is internally inconsistent"
        )
    normalized = normalize_rich_production_science_request_v2(
        {
            "schema_version": "eiel_production_science_request_v2",
            "request_id": request_id,
            "baseline": baseline_declaration,
            "science_campaign": provisional,
        }
    )
    compilation = compile_science_campaign_v1(
        normalized.campaign,
        instrument_manifest=host_context.instrument_manifest,
        route_binding=host_context.route_binding,
        control_descriptors=host_context.control_descriptors,
        reducer_descriptors=host_context.reducer_descriptors,
        crca_semantic_binding=host_context.crca_semantic_binding,
        physical_execution_core=host_context.physical_execution_core,
        compiler_component_sha256=host_context.compiler_component_sha256,
    )
    bundle = materialize_science_world_bundle_v1(
        ((normalized.campaign, compilation),),
        bundle_id="MUTSUP-W32-PAIRED-SEED-10K-AUTHORING",
    )
    configuration = _world_ownership_configuration(
        bundle.worlds,
        milestone_ticks=observation_ticks,
    )
    campaign = _mutsup_campaign_document(
        host_context.instrument_manifest,
        reducer,
        reducer_configuration=configuration,
        task_seeds=tuple(task_seeds),
        campaign_id=campaign_id,
        study_id=study_id,
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoint_ticks,
        observation_ticks=observation_ticks,
    )
    request = {
        "schema_version": "eiel_production_science_request_v2",
        "request_id": request_id,
        "baseline": baseline_declaration,
        "science_campaign": campaign,
    }
    # Ensure the final explicit ownership configuration did not accidentally
    # alter semantic world identities and that every field is normalizable.
    final = normalize_rich_production_science_request_v2(request)
    if (
        final.campaign.campaign_spec_sha256()
        == normalized.campaign.campaign_spec_sha256()
    ):
        raise ScienceCudaV2HostContextError(
            "MUTSUP reducer ownership was not materialized"
        )
    return request


def write_w32_cuda_v2_mutsup_artifacts_v1(
    context_path: str | Path,
    request_path: str | Path,
    *,
    summary: Mapping[str, Any] | str | Path | None = None,
    device_receipt: Mapping[str, Any] | str | Path | None = None,
    total_ticks: int = SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS,
    checkpoint_ticks: tuple[int, ...] | None = None,
    observation_ticks: tuple[int, ...] | None = None,
) -> tuple[Path, Path]:
    """Write canonical host-context and request JSON files deterministically."""

    context = materialize_w32_cuda_v2_science_host_context_v1(
        summary,
        device_receipt=device_receipt,
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoint_ticks,
        observation_ticks=observation_ticks,
    )
    request = build_mutsup_w32_paired_seed_request_v1(
        context,
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoint_ticks,
        observation_ticks=observation_ticks,
    )
    context_value = serialize_science_host_context_v1(context)
    normalized = normalize_rich_production_science_request_v2(request)
    request_value = normalized.normalized_request
    context_target = Path(context_path).expanduser().resolve()
    request_target = Path(request_path).expanduser().resolve()
    context_target.parent.mkdir(parents=True, exist_ok=True)
    request_target.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable_canonical_json(context_target, context_value)
    _write_immutable_canonical_json(request_target, request_value)
    return context_target, request_target


def build_w32_cuda_v2_mutsup_base_plan_v1(
    context: ScienceInstrumentHostContextV1,
    *,
    total_ticks: int,
    checkpoint_ticks: tuple[int, ...],
) -> dict[str, Any]:
    """Compile the checked-in native base spec through the campaign compiler.

    The science request is lowered separately.  This plan remains the
    canonical compiler product for the native route, while the context keeps
    the richer physical artifact/device identity.  A canary uses a
    deterministic in-memory timing adjustment to the same checked-in spec.
    """

    plan = _compile_w32_cuda_v2_base_plan_payload_v1(
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoint_ticks,
    )
    physical = _mapping(
        context.physical_execution_core.to_value(),
        field="physical_execution_core",
    )
    native_artifact = _mapping(
        physical.get("native_artifact"), field="physical_execution_core.native_artifact"
    )
    expected_artifacts = {
        "cuda_extension": _mapping(
            native_artifact.get("binary_sha256"), field="binary_sha256"
        )["hex"],
        "extension_manifest": _mapping(
            native_artifact.get("manifest_file_sha256"), field="manifest_file_sha256"
        )["hex"],
    }
    for name, expected_sha in expected_artifacts.items():
        row = _mapping(plan.get("artifacts"), field="base plan artifacts").get(name)
        if not isinstance(row, Mapping) or row.get("sha256") != expected_sha:
            raise ScienceCudaV2HostContextError(
                f"compiled base plan artifact identity mismatch for {name}"
            )
    current_route = CudaFullEvolutionV2ExecutionProviderV1().execution_route_contract(
        plan
    )
    bound_route = _mapping(
        physical.get("route_profile_sha256"), field="route_profile_sha256"
    ).get("hex")
    if current_route.execution_route_sha256 != bound_route:
        raise ScienceCudaV2HostContextError(
            "compiled base plan route differs from the live host-context binding"
        )
    return plan


def _write_immutable_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create or verify one canonical immutable JSON input without overwriting it."""

    encoded = canonical_json_bytes(value)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise ScienceCudaV2HostContextError(
                f"immutable output is already bound to different bytes: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
    except (OSError, ValueError) as error:
        try:
            path.unlink()
        except OSError:
            pass
        raise ScienceCudaV2HostContextError(
            f"cannot write immutable output: {path}"
        ) from error


def write_w32_cuda_v2_mutsup_inputs_v1(
    context_path: str | Path,
    request_path: str | Path,
    base_plan_path: str | Path,
    *,
    summary: Mapping[str, Any] | str | Path | None = None,
    device_receipt: Mapping[str, Any] | str | Path | None = None,
    total_ticks: int = SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS,
    checkpoint_ticks: tuple[int, ...] | None = None,
    observation_ticks: tuple[int, ...] | None = None,
) -> tuple[Path, Path, Path]:
    """Write the complete immutable context/request/base-plan input triplet."""

    context = materialize_w32_cuda_v2_science_host_context_v1(
        summary,
        device_receipt=device_receipt,
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoint_ticks,
        observation_ticks=observation_ticks,
    )
    request = build_mutsup_w32_paired_seed_request_v1(
        context,
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoint_ticks,
        observation_ticks=observation_ticks,
    )
    normalized = normalize_rich_production_science_request_v2(request)
    actual_checkpoints = tuple(normalized.campaign.time.to_value()["checkpoint_ticks"])
    context_target = Path(context_path).expanduser().resolve()
    request_target = Path(request_path).expanduser().resolve()
    base_target = Path(base_plan_path).expanduser().resolve()
    _write_immutable_canonical_json(
        context_target,
        serialize_science_host_context_v1(context),
    )
    _write_immutable_canonical_json(request_target, normalized.normalized_request)
    _write_immutable_canonical_json(
        base_target,
        build_w32_cuda_v2_mutsup_base_plan_v1(
            context,
            total_ticks=total_ticks,
            checkpoint_ticks=actual_checkpoints,
        ),
    )
    return context_target, request_target, base_target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--device-receipt", type=Path, default=None)
    parser.add_argument(
        "--context-output",
        type=Path,
        required=True,
        help="fresh output path for the canonical host-context artifact",
    )
    parser.add_argument(
        "--request-output",
        type=Path,
        required=True,
        help="fresh output path for the normalized MUTSUP request",
    )
    parser.add_argument(
        "--base-plan-output",
        type=Path,
        default=None,
        help="optional output path for the immutable canonical-lowering base plan",
    )
    parser.add_argument(
        "--total-ticks",
        type=int,
        default=SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS,
        help="absolute tick horizon (default: 10000; use 160 for the canary)",
    )
    parser.add_argument(
        "--checkpoint-ticks",
        type=int,
        nargs="+",
        default=None,
        help="explicit positive absolute checkpoint ticks ending at --total-ticks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kwargs = {
        "summary": args.summary,
        "device_receipt": args.device_receipt,
        "total_ticks": args.total_ticks,
        "checkpoint_ticks": (
            None if args.checkpoint_ticks is None else tuple(args.checkpoint_ticks)
        ),
    }
    if args.base_plan_output is None:
        context_path, request_path = write_w32_cuda_v2_mutsup_artifacts_v1(
            args.context_output,
            args.request_output,
            **kwargs,
        )
        output = {"context": str(context_path), "request": str(request_path)}
    else:
        context_path, request_path, base_plan_path = write_w32_cuda_v2_mutsup_inputs_v1(
            args.context_output,
            args.request_output,
            args.base_plan_output,
            **kwargs,
        )
        output = {
            "base_plan": str(base_plan_path),
            "context": str(context_path),
            "request": str(request_path),
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = (
    "SCIENCE_CUDA_V2_HOST_CONTEXT_SCHEMA_VERSION",
    "SCIENCE_CUDA_V2_MUTSUP_AUTHORING_SCHEMA_VERSION",
    "SCIENCE_CUDA_V2_MUTSUP_CHECKPOINT_TICKS",
    "SCIENCE_CUDA_V2_MUTSUP_OBSERVATION_TICKS",
    "SCIENCE_CUDA_V2_MUTSUP_REQUEST_ID",
    "SCIENCE_CUDA_V2_MUTSUP_TASK_SEEDS",
    "SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS",
    "SCIENCE_CUDA_V2_W32_CARRIER_PROFILE_ID",
    "SCIENCE_CUDA_V2_W32_WORLD_PROFILE_ID",
    "ScienceCudaV2HostContextError",
    "build_mutsup_w32_paired_seed_request_v1",
    "build_w32_cuda_v2_mutsup_base_plan_v1",
    "load_mutsup_w32_authoring_spec_v1",
    "main",
    "materialize_w32_cuda_v2_science_host_context_v1",
    "write_w32_cuda_v2_mutsup_artifacts_v1",
    "write_w32_cuda_v2_mutsup_inputs_v1",
)
