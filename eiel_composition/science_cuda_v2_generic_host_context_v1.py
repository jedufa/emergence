"""Generic host context for explicit CUDA V2 Science Instrument bundles."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any, Final

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    EEDReleaseStatusV1,
    SchemaVersion,
    ScienceControlDescriptorV1,
    ScienceInstrumentManifestV1,
    ScienceObservationProfileDescriptorV1,
    ScienceObservationProfileV1,
    ScienceRecordKindV1,
    ScienceRouteBindingV1,
    SCIENCE_GENOTYPE_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_HEREDITARY_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1,
    canonical_sha256,
    science_observation_profile_catalog_sha256_v1,
)
from emergence_evidence.science_compact_world_atlas_v1 import (
    COMPACT_WORLD_ATLAS_REQUIRED_CAPABILITIES,
    compact_world_atlas_reducer_descriptor_v1,
)
from engine.campaigns.baseline_v1 import BASELINE_PROFILE_ID
from engine.campaigns.cuda_full_evolution_v2_provider import (
    CudaFullEvolutionV2ExecutionProviderV1,
)
from engine.campaigns.science_cuda_v2_native_world_v1 import (
    SCIENCE_CUDA_V2_SUPPORTED_NATIVE_CONTROL_IDS_V1,
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

from .science_cuda_v2_host_context_v1 import (
    SCIENCE_CUDA_V2_W32_CARRIER_PROFILE_ID,
    _no_qualification_authority,
    _payload,
    _physical_core,
    _resource_envelope,
)
from .exact_natural_source_state_v1 import (
    EXACT_NATURAL_SOURCE_STATE_CAPABILITY_V1,
)
from .science_instrument import (
    SHADOW_PROJECTION_AUTHORITY,
    ScienceCRCASemanticBindingV1,
    science_control_catalog_sha256,
)
from .science_instrument_host_front_door_v1 import ScienceInstrumentHostContextV1
from .science_qrr_projection_v1 import ScienceQRRRouteAdapterV1
from .science_qualification_envelope_projection_v1 import (
    ScienceQualifiedEnvelopeProjectionV1,
)


GENERIC_CUDA_V2_SCIENCE_CONTEXT_ID: Final[str] = (
    "eiel_science_cuda_v2_explicit_bundle_host_context_v1"
)
GENERIC_CUDA_V2_SCIENCE_WORLD_PROFILE_ID: Final[str] = (
    "eiel-w32-native-science-v1"
)


class GenericCudaV2ScienceHostContextError(ValueError):
    """The compiled native route cannot form an exact generic host context."""


def _read_plan(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value).expanduser()
    if path.is_symlink() or not path.resolve().is_file():
        raise GenericCudaV2ScienceHostContextError(
            f"compiled base plan is not a regular file: {path}"
        )
    try:
        result = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenericCudaV2ScienceHostContextError(
            f"compiled base plan is unreadable: {path}"
        ) from exc
    if not isinstance(result, dict):
        raise GenericCudaV2ScienceHostContextError("compiled base plan must be an object")
    return result


def _sha256_file(path: Path) -> ContentDigest:
    if path.is_symlink() or not path.is_file():
        raise GenericCudaV2ScienceHostContextError(
            f"instrument dependency is not a regular file: {path}"
        )
    return ContentDigest("sha256", hashlib.sha256(path.read_bytes()).hexdigest())


def _plan_identity(plan: Mapping[str, Any]) -> tuple[dict[str, Any], int, int]:
    if (
        plan.get("schema_version") != "eiel_campaign_plan_v1"
        or plan.get("resolved_profile") != CUDA_NATIVE_PROFILE_ID_V2
    ):
        raise GenericCudaV2ScienceHostContextError(
            "compiled base plan is not the CUDA V2 native profile"
        )
    try:
        route = CudaFullEvolutionV2ExecutionProviderV1().execution_route_contract(plan)
    except Exception as exc:
        raise GenericCudaV2ScienceHostContextError(
            f"compiled native execution route is invalid: {exc}"
        ) from exc
    components = {
        str(row.get("component_id")): dict(row)
        for row in route.components
        if isinstance(row, Mapping)
    }
    artifact = components.get("cuda_full_evolution_v2_artifact_unit")
    device = components.get("cuda_full_evolution_v2_device_selection")
    artifacts = plan.get("artifacts")
    cuda_artifact = (
        artifacts.get("cuda_extension") if isinstance(artifacts, Mapping) else None
    )
    if not all(isinstance(row, Mapping) for row in (artifact, device, cuda_artifact)):
        raise GenericCudaV2ScienceHostContextError(
            "compiled native route lacks artifact or device identity"
        )
    batch_size = device.get("B")
    temporal_k = device.get("temporal_k")
    if type(batch_size) is not int or batch_size < 1 or type(temporal_k) is not int:
        raise GenericCudaV2ScienceHostContextError("sealed B/K identity is invalid")
    identity = {
        "route_id": CUDA_NATIVE_ROUTE_V2,
        "profile_id": route.profile_id,
        "carrier_profile_id": SCIENCE_CUDA_V2_W32_CARRIER_PROFILE_ID,
        "provider_id": route.provider_id,
        "provider_family": "cuda_full_evolution_v2",
        "execution_route_sha256": ContentDigest(
            "sha256", route.execution_route_sha256
        ),
        "binary_sha256": ContentDigest("sha256", artifact["binary_sha256"]),
        "manifest_sha256": ContentDigest("sha256", artifact["manifest_sha256"]),
        "manifest_file_sha256": ContentDigest(
            "sha256", artifact["manifest_file_sha256"]
        ),
        "artifact_path": str(cuda_artifact["path"]),
        "device": dict(device["device_identity"]),
        "device_selection_receipt_sha256": ContentDigest(
            "sha256", device["selection_receipt_sha256"]
        ),
        "native_abi": CUDA_NATIVE_ABI_V1,
    }
    return identity, batch_size, temporal_k


def generic_cuda_v2_science_control_descriptors_v1() -> tuple[
    ScienceControlDescriptorV1, ...
]:
    boolean_ids = {
        control_id
        for control_id in SCIENCE_CUDA_V2_SUPPORTED_NATIVE_CONTROL_IDS_V1
        if control_id.endswith(".enabled")
    }
    return tuple(
        ScienceControlDescriptorV1(
            control_id=control_id,
            value_contract=_payload(
                "eiel_science_native_resolved_control_value_v1",
                {
                    "kind": "boolean" if control_id in boolean_ids else "numeric",
                    "native_control_id": control_id,
                },
            ),
            unit="boolean" if control_id in boolean_ids else "native_unit",
            activation_phases=("native_world_control",),
            heterogeneity_modes=("per_world",),
            schedule_kinds=(
                "piecewise_constant_absolute_ticks",
                "pulse_absolute_ticks",
                "static",
                "switch_absolute_tick",
            ),
            readback_capability_ids=("control_readback",),
            interaction_ids=("native_world_control",),
            component_id="cuda_full_evolution_v2.native_world_control",
            qualified_envelope_subjects=(f"controls.{control_id}",),
        )
        for control_id in SCIENCE_CUDA_V2_SUPPORTED_NATIVE_CONTROL_IDS_V1
    )


def _observation_descriptor() -> ScienceObservationProfileDescriptorV1:
    capabilities = tuple(
        sorted(
            {
                *COMPACT_WORLD_ATLAS_REQUIRED_CAPABILITIES,
                *SCIENCE_GENOTYPE_ATTRIBUTION_CAPABILITIES_V1,
                *SCIENCE_HEREDITARY_ATTRIBUTION_CAPABILITIES_V1,
                *SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1,
                "birth",
                "birth_count",
                "birth_parentage",
                "causal_lifecycle",
                "control_readback",
                "death",
                EXACT_NATURAL_SOURCE_STATE_CAPABILITY_V1,
                "genome_identity",
                "lineage_ancestry",
                "mutation",
                "mutation_count",
                "mutation_provenance",
                "population_count",
            }
        )
    )
    return ScienceObservationProfileDescriptorV1(
        profile=ScienceObservationProfileV1.STANDARD,
        observation_abi_version=SchemaVersion(
            "eiel_cuda_v2_compact_science_capture_v1"
        ),
        record_kinds=tuple(ScienceRecordKindV1),
        required_nonempty_record_kinds=(
            ScienceRecordKindV1.CONTROL_BINDING,
            ScienceRecordKindV1.WORLD_COMPLETION,
        ),
        capability_ids=capabilities,
        producer_phase_registry_ordinals=(
            ("cuda_full_evolution_v2.compact_science_capture", 0),
        ),
        default_partition_span_ticks=64,
    )


def _crca_binding(
    *, total_ticks: int, checkpoint_ticks: tuple[int, ...], scope_id: str
) -> ScienceCRCASemanticBindingV1:
    core = {
        "schema_version": "eiel_crca_explicit_science_bundle_binding_v1",
        "scope_id": scope_id,
        "absolute_start_next_tick": 0,
        "absolute_target_next_tick": total_ticks,
        "trajectory_checkpoint_ticks": [0, *checkpoint_ticks],
        "authority": SHADOW_PROJECTION_AUTHORITY,
    }
    semantic = canonical_sha256(core)
    return ScienceCRCASemanticBindingV1(
        semantic_run_digest=semantic,
        projection_sha256=canonical_sha256(
            {
                "schema_version": "eiel_crca_shadow_explicit_science_projection_v1",
                "authority": SHADOW_PROJECTION_AUTHORITY,
                "semantic": {"payload": core, "digest": semantic.to_dict()},
            }
        ),
        absolute_start_next_tick=0,
        absolute_target_next_tick=total_ticks,
        trajectory_checkpoint_ticks=(0, *checkpoint_ticks),
    )


def materialize_cuda_v2_explicit_bundle_host_context_v1(
    base_campaign_plan: Mapping[str, Any] | str | Path,
    *,
    logical_world_capacity: int = 1024,
    context_scope_id: str = GENERIC_CUDA_V2_SCIENCE_CONTEXT_ID,
) -> ScienceInstrumentHostContextV1:
    """Bind a generic explicit-world Science Instrument context to one sealed route."""

    if type(logical_world_capacity) is not int or logical_world_capacity < 1:
        raise GenericCudaV2ScienceHostContextError(
            "logical world capacity must be positive"
        )
    plan = _read_plan(base_campaign_plan)
    identity, batch_size, temporal_k = _plan_identity(plan)
    if batch_size < logical_world_capacity:
        raise GenericCudaV2ScienceHostContextError(
            "sealed physical B is smaller than the requested logical capacity"
        )
    execution = plan.get("execution")
    lifecycle = plan.get("artifact_lifecycle")
    storage = lifecycle.get("storage") if isinstance(lifecycle, Mapping) else None
    if not isinstance(execution, Mapping) or not isinstance(storage, Mapping):
        raise GenericCudaV2ScienceHostContextError(
            "compiled plan lacks execution or lifecycle contracts"
        )
    total_ticks = execution.get("total_ticks")
    checkpoint_ticks = tuple(execution.get("checkpoint_ticks", ()))
    if (
        type(total_ticks) is not int
        or total_ticks < 1
        or not checkpoint_ticks
        or checkpoint_ticks[-1] != total_ticks
    ):
        raise GenericCudaV2ScienceHostContextError(
            "compiled plan absolute-time contract is invalid"
        )
    controls = generic_cuda_v2_science_control_descriptors_v1()
    reducer = compact_world_atlas_reducer_descriptor_v1()
    observation = _observation_descriptor()
    repository_root = Path(__file__).resolve().parents[1]
    founder_registry = repository_root / (
        "_spec/science_instrument/v1/native_founder_profile_registry_v1.json"
    )
    founder_value = json.loads(founder_registry.read_text(encoding="utf-8"))
    founder_ids = tuple(
        sorted(str(row["profile_id"]) for row in founder_value["profiles"])
    )
    observation_dependencies = (
        repository_root
        / "emergence_contracts/science_compact_capture.py",
        repository_root
        / "engine/native_runtime/cuda_v2_science_capture_v1.py",
        repository_root
        / "engine/native_runtime/cuda_v2_compact_science_capture_v1.py",
        repository_root
        / "engine/perf/cuda_full_evolution_v2/science_native_world_runtime_v1.py",
        Path(__file__).resolve(),
    )
    observation_abi = canonical_sha256(
        [
            {"path": str(path.relative_to(repository_root)), "sha256": _sha256_file(path).to_dict()}
            for path in observation_dependencies
        ]
    )
    world_profile = {
        "world_profile_id": GENERIC_CUDA_V2_SCIENCE_WORLD_PROFILE_ID,
        "carrier_profile_id": SCIENCE_CUDA_V2_W32_CARRIER_PROFILE_ID,
        "native_profile_id": CUDA_NATIVE_PROFILE_ID_V2,
        "geometry": {"width": WORLD_WIDTH_V2, "height": WORLD_WIDTH_V2},
        "initial_population_max": INITIAL_POPULATION_V2,
        "resident_capacity": RESIDENT_CAPACITY_C_V2,
    }
    component_graph = {
        "instrument": "EIEL_SCIENCE_INSTRUMENT_V1",
        "context_scope_id": context_scope_id,
        "world_profile": world_profile,
        "control_descriptors": [row.descriptor_sha256().to_dict() for row in controls],
        "reducer_descriptor": reducer.descriptor_sha256().to_dict(),
        "native_artifact_manifest_sha256": identity["manifest_sha256"].to_dict(),
    }
    manifest = ScienceInstrumentManifestV1(
        component_graph_sha256=canonical_sha256(component_graph),
        control_catalog_sha256=science_control_catalog_sha256(controls),
        world_profile_catalog_sha256=canonical_sha256([world_profile]),
        founder_profile_catalog_sha256=canonical_sha256(
            {"registry_sha256": _sha256_file(founder_registry).to_dict(), "profile_ids": founder_ids}
        ),
        policy_profile_catalog_sha256=canonical_sha256(
            ["cuda-v2-native-diploid-policy-genome-v1"]
        ),
        mutation_operator_catalog_sha256=canonical_sha256(
            ["cuda-v2-native-point-mutation-and-recombination-v1"]
        ),
        observation_abi_sha256=observation_abi,
        observation_profile_catalog_sha256=(
            science_observation_profile_catalog_sha256_v1((observation,))
        ),
        reducer_protocol_version=SchemaVersion("eiel_science_reducer_protocol_v1"),
        route_binding_protocol_version=SchemaVersion("eiel_science_route_binding_v1"),
        eed_release_status=EEDReleaseStatusV1.EXCLUDED,
    )
    lifecycle_bytes = int(storage["total_max_bytes"])
    adapter = ScienceQRRRouteAdapterV1(
        route_id=CUDA_NATIVE_ROUTE_V2,
        machine_profile_id=BASELINE_PROFILE_ID,
        world_profile_id=GENERIC_CUDA_V2_SCIENCE_WORLD_PROFILE_ID,
        world_width=WORLD_WIDTH_V2,
        maximum_batch_cells=batch_size,
        task_seed_owner="task_environment",
        control_path_by_id=tuple(
            (control_id, f"science_world.controls.{control_id}")
            for control_id in SCIENCE_CUDA_V2_SUPPORTED_NATIVE_CONTROL_IDS_V1
        ),
        seed_path_by_owner=tuple(
            (owner, f"science_world.seeds.{owner}")
            for owner in (
                "ecology",
                "founder",
                "initialization",
                "intervention",
                "mutation",
                "policy",
            )
        ),
        fixed_flat_controls=_payload(
            "eiel_science_cuda_v2_explicit_bundle_route_controls_v1",
            {
                "cuda_native.event_transport": "direct_sparse_v3",
                "cuda_native.execution_mode": "cuda_graph_monolith",
                "cuda_native.science_observation_mode": "COMPACT_SIDECAR_V1",
                "cuda_native.temporal_k": temporal_k,
            },
        ),
        acs_capture_instrumentation=False,
        acs_checkpoint_slots=0,
        living_lineage_canopy=True,
        lifecycle_total_max_bytes=lifecycle_bytes,
    )
    resource = _resource_envelope(logical_world_count=logical_world_capacity)
    no_authority = _no_qualification_authority(CUDA_NATIVE_ROUTE_V2)
    qualified = ScienceQualifiedEnvelopeProjectionV1(
        instrument_manifest_sha256=manifest.manifest_sha256(),
        route_id=CUDA_NATIVE_ROUTE_V2,
        machine_profile_id=BASELINE_PROFILE_ID,
        qualification_authority_sha256=no_authority,
        rows=(),
    )
    capability_projection = canonical_sha256(
        {
            "native_profile_id": CUDA_NATIVE_PROFILE_ID_V2,
            "native_route_id": CUDA_NATIVE_ROUTE_V2,
            "carrier_profile_id": SCIENCE_CUDA_V2_W32_CARRIER_PROFILE_ID,
            "capabilities": list(FULL_EVOLUTION_CAPABILITIES_V2),
            "generic_world_controls": list(
                SCIENCE_CUDA_V2_SUPPORTED_NATIVE_CONTROL_IDS_V1
            ),
            "founder_profile_ids": list(founder_ids),
        }
    )
    physical = _physical_core(
        identity,
        temporal_k=temporal_k,
        logical_bundle_size=logical_world_capacity,
        target_physical_batch=batch_size,
    )
    route_binding = ScienceRouteBindingV1(
        instrument_manifest_sha256=manifest.manifest_sha256(),
        route_id=CUDA_NATIVE_ROUTE_V2,
        route_profile_sha256=identity["execution_route_sha256"],
        machine_profile_id=BASELINE_PROFILE_ID,
        capability_projection_sha256=capability_projection,
        native_build_closure_sha256=identity["manifest_sha256"],
        execution_adapter_sha256=adapter.execution_adapter_sha256(),
        control_adapter_sha256=adapter.control_adapter_sha256(),
        observation_adapter_sha256=observation_abi,
        checkpoint_adapter_sha256=canonical_sha256(
            {
                "schema_version": "eiel_science_absolute_tick_checkpoint_adapter_v1",
                "checkpoint_ticks": list(checkpoint_ticks),
                "compact_frontier": "eiel_cuda_v2_compact_science_checkpoint_frontier_v1",
            }
        ),
        semantic_conformance_sha256=canonical_sha256(
            {
                "time_authority": "absolute_tick_v1",
                "total_ticks": total_ticks,
                "checkpoint_ticks": list(checkpoint_ticks),
                "status": "SCIENTIFIC_EXECUTION_NO_QUALIFICATION_AUTHORITY",
            }
        ),
        resource_envelope_sha256=canonical_sha256(resource.to_dict()),
        qualified_envelope_sha256=qualified.projection_sha256(),
        qualification_authority_sha256=no_authority,
    )
    scientific_capabilities = tuple(
        sorted(
            {
                f"world_profile:{GENERIC_CUDA_V2_SCIENCE_WORLD_PROFILE_ID}",
                *(f"founder_profile:{item}" for item in founder_ids),
                *(
                    f"control:{item}"
                    for item in SCIENCE_CUDA_V2_SUPPORTED_NATIVE_CONTROL_IDS_V1
                ),
                "schedule:piecewise_constant_absolute_ticks",
                "schedule:pulse_absolute_ticks",
                "schedule:static",
                "schedule:switch_absolute_tick",
                "exact_natural_source_state_materialization_v1",
            }
        )
    )
    return ScienceInstrumentHostContextV1(
        instrument_manifest=manifest,
        route_binding=route_binding,
        route_adapter=adapter,
        control_descriptors=controls,
        reducer_descriptors=(reducer,),
        observation_profile_descriptors=(observation,),
        crca_semantic_binding=_crca_binding(
            total_ticks=total_ticks,
            checkpoint_ticks=checkpoint_ticks,
            scope_id=context_scope_id,
        ),
        physical_execution_core=physical,
        resource_envelope=resource,
        compiler_component_sha256=canonical_sha256(
            {"component": GENERIC_CUDA_V2_SCIENCE_CONTEXT_ID, "version": 1}
        ),
        qualified_envelope=qualified,
        available_scientific_capability_ids=scientific_capabilities,
        available_observation_capability_ids=observation.capability_ids,
        machine_profile_sha256=canonical_sha256(
            {"profile_id": BASELINE_PROFILE_ID, "identity_scope": "semantic_machine_profile"}
        ),
        admission_kernel_receipt_sha256=canonical_sha256(
            {
                "schema_version": "eiel_attempt_admission_kernel_v1",
                "authority": "canonical_crca_attempt_admission",
                "status": "HOST_CONTEXT_BINDING_ONLY",
                "route_id": CUDA_NATIVE_ROUTE_V2,
            }
        ),
    )


__all__ = (
    "GENERIC_CUDA_V2_SCIENCE_CONTEXT_ID",
    "GENERIC_CUDA_V2_SCIENCE_WORLD_PROFILE_ID",
    "GenericCudaV2ScienceHostContextError",
    "generic_cuda_v2_science_control_descriptors_v1",
    "materialize_cuda_v2_explicit_bundle_host_context_v1",
)
