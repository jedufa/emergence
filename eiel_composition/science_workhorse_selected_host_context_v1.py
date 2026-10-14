"""Narrow host context for exact selected-world Workhorse continuations.

This adapter grants no general Workhorse science authority.  It exists only
for singleton, ACS2, observation-OFF selected-world children whose compact
observation is authenticated later at the existing host boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any, Final

from emergence_contracts import (
    ContentDigest,
    EEDReleaseStatusV1,
    PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION,
    ProfileResourceEnvelopeV1,
    SchemaVersion,
    ScienceCudaV2CarrierProfileV1,
    ScienceInstrumentManifestV1,
    ScienceObservationProfileDescriptorV1,
    ScienceObservationProfileV1,
    ScienceRecordKindV1,
    ScienceRouteBindingV1,
    canonical_sha256,
    science_observation_profile_catalog_sha256_v1,
)
from emergence_evidence.science_compact_world_atlas_v1 import (
    compact_world_atlas_reducer_descriptor_v1,
)
from engine.campaigns.baseline_v1 import BASELINE_PROFILE_ID
from engine.campaigns.cuda_full_evolution_v2_workhorse_provider import (
    EXECUTION_ROW_OPERATION_V2,
    PROVIDER_ID as WORKHORSE_PROVIDER_ID,
    CudaFullEvolutionWorkhorseV2ExecutionProviderV1,
)
from engine.perf.cuda_full_evolution_v2.workhorse.contract import (
    WORKHORSE_ABI_V2,
    workhorse_shape_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.selected_world_observation import (
    WORKHORSE_SELECTED_WORLD_OBSERVATION_SCHEMA_V1,
)

from .science_cuda_v2_generic_host_context_v1 import (
    _crca_binding,
    generic_cuda_v2_science_control_descriptors_v1,
)
from .science_cuda_v2_host_context_v1 import _no_qualification_authority, _payload
from .science_instrument import (
    SHADOW_PROJECTION_AUTHORITY,
    science_control_catalog_sha256,
)
from .science_instrument_host_front_door_v1 import ScienceInstrumentHostContextV1
from .science_qrr_projection_v1 import ScienceQRRRouteAdapterV1
from .science_qualification_envelope_projection_v1 import (
    ScienceQualifiedEnvelopeProjectionV1,
)


WORKHORSE_SELECTED_HOST_CONTEXT_ID_V1: Final[str] = (
    "eiel_science_workhorse_selected_host_context_v1"
)
_AUTHORIZED_CONTROL_IDS: Final[tuple[str, str]] = (
    "environment.resource.multiplier",
    "mutation.rate",
)
_AUTHORIZED_SCHEDULE_KINDS: Final[tuple[str, str]] = (
    "piecewise_constant_absolute_ticks",
    "static",
)


class WorkhorseSelectedHostContextError(ValueError):
    """The base plan is outside the exact selected-world Workhorse boundary."""


def _resource_envelope_v1(
    *,
    logical_world_count: int,
    resident_capacity: int,
    record_kinds: tuple[ScienceRecordKindV1, ...],
) -> ProfileResourceEnvelopeV1:
    kinds = tuple(sorted(record_kinds, key=lambda item: item.value))
    max_records = 512
    max_record_bytes = 65_536
    chunk_max_bytes = 4 * 1024 * 1024
    max_in_flight_chunks = 2
    fixed_buffer_bytes = 1024
    per_world_cursor_bytes = 256
    per_agent_bytes = 1
    resident_bytes = (
        fixed_buffer_bytes
        + max_in_flight_chunks * chunk_max_bytes
        + logical_world_count * per_world_cursor_bytes
        + logical_world_count * resident_capacity * per_agent_bytes
    )
    return ProfileResourceEnvelopeV1(
        profile=ScienceObservationProfileV1.STANDARD,
        max_records_per_tick_per_world_by_kind=tuple(
            (kind, max_records) for kind in kinds
        ),
        max_record_bytes_by_kind=tuple(
            (kind, max_record_bytes) for kind in kinds
        ),
        chunk_max_records=logical_world_count * max_records * len(kinds),
        chunk_max_bytes=chunk_max_bytes,
        max_in_flight_chunks=max_in_flight_chunks,
        fixed_buffer_bytes=fixed_buffer_bytes,
        per_world_cursor_bytes=per_world_cursor_bytes,
        per_resident_agent_observation_state_bytes=per_agent_bytes,
        logical_world_count=logical_world_count,
        resident_capacity=resident_capacity,
        max_resident_observation_bytes=resident_bytes,
    )


def _selected_observation_descriptor_v1() -> ScienceObservationProfileDescriptorV1:
    """Describe only facts authenticated by the OFF host-boundary adapter."""

    return ScienceObservationProfileDescriptorV1(
        profile=ScienceObservationProfileV1.STANDARD,
        observation_abi_version=SchemaVersion(
            WORKHORSE_SELECTED_WORLD_OBSERVATION_SCHEMA_V1
        ),
        record_kinds=(
            ScienceRecordKindV1.CONTROL_BINDING,
            ScienceRecordKindV1.WORLD_STATE_SNAPSHOT,
            ScienceRecordKindV1.AGENT_STATE_SNAPSHOT,
            ScienceRecordKindV1.WORLD_COMPLETION,
            ScienceRecordKindV1.CHECKPOINT_MARKER,
        ),
        required_nonempty_record_kinds=(
            ScienceRecordKindV1.CONTROL_BINDING,
            ScienceRecordKindV1.WORLD_STATE_SNAPSHOT,
            ScienceRecordKindV1.WORLD_COMPLETION,
            ScienceRecordKindV1.CHECKPOINT_MARKER,
        ),
        capability_ids=tuple(
            sorted(
                {
                    "birth_count",
                    "control_readback",
                    "death_count",
                    "genome_identity",
                    "mutation_count",
                    "population_count",
                    "total_event_count",
                }
            )
        ),
        producer_phase_registry_ordinals=(
            (
                "engine.perf.cuda_full_evolution_v2.workhorse."
                "selected_world_observation",
                0,
            ),
        ),
        default_partition_span_ticks=1,
    )


def _source_closure_sha256_v1() -> ContentDigest:
    repository_root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        repository_root / "eiel_composition/science_cuda_v2_carrier_v1.py",
        repository_root / "eiel_composition/science_instrument_operator_v1.py",
        repository_root
        / "engine/campaigns/cuda_full_evolution_v2_workhorse_provider.py",
        repository_root
        / "engine/perf/cuda_full_evolution_v2/workhorse/selected_world_observation.py",
    )
    return canonical_sha256(
        [
            {
                "path": str(path.relative_to(repository_root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ]
    )


def materialize_workhorse_selected_host_context_v1(
    base_plan: Mapping[str, Any],
    *,
    profile: ScienceCudaV2CarrierProfileV1,
    logical_world_capacity: int,
    context_scope_id: str,
) -> ScienceInstrumentHostContextV1:
    """Bind one exact B1 Workhorse authority for selected-state children only."""

    if type(logical_world_capacity) is not int or logical_world_capacity != 1:
        raise WorkhorseSelectedHostContextError(
            "selected-world Workhorse host context requires singleton capacity"
        )
    if profile.provider_id != WORKHORSE_PROVIDER_ID:
        raise WorkhorseSelectedHostContextError("carrier profile is not Workhorse")
    execution = base_plan.get("execution")
    lifecycle = base_plan.get("artifact_lifecycle")
    analysis = base_plan.get("analysis")
    extension = analysis.get("extension") if isinstance(analysis, Mapping) else None
    operation = extension.get("contract") if isinstance(extension, Mapping) else None
    native_contract = (
        operation.get("contract") if isinstance(operation, Mapping) else None
    )
    if (
        base_plan.get("schema_version") != "eiel_campaign_plan_v1"
        or base_plan.get("resolved_profile") != profile.native_profile_id
        or not isinstance(execution, Mapping)
        or not isinstance(lifecycle, Mapping)
        or lifecycle.get("lifecycle_profile") != "SCIENCE_COMPACT"
        or execution.get("start_next_tick") != 0
        or execution.get("acs_checkpoint_slots") != 2
        or execution.get("native_streamed_checkpoint") is not True
        or not isinstance(operation, Mapping)
        or operation.get("operation") != EXECUTION_ROW_OPERATION_V2
        or not isinstance(native_contract, Mapping)
        or native_contract.get("B") != 1
        or native_contract.get("shape_id") != profile.native_shape_id
        or native_contract.get("profile_id") != profile.native_profile_id
        or native_contract.get("route_id") != profile.native_route_id
        or native_contract.get("native_abi") != WORKHORSE_ABI_V2
        or native_contract.get("backend_id") not in profile.backend_ids
    ):
        raise WorkhorseSelectedHostContextError(
            "authority must be a B1/ACS2 SCIENCE_COMPACT Workhorse execution row"
        )
    total_ticks = execution.get("total_ticks")
    checkpoint_ticks = tuple(execution.get("checkpoint_ticks", ()))
    if (
        type(total_ticks) is not int
        or total_ticks < 1
        or checkpoint_ticks != (total_ticks,)
        or native_contract.get("start_next_tick") != 0
        or native_contract.get("total_ticks") != total_ticks
    ):
        raise WorkhorseSelectedHostContextError(
            "selected-world Workhorse authority requires one terminal absolute checkpoint"
        )
    shape = workhorse_shape_v2(profile.native_shape_id)
    try:
        route = CudaFullEvolutionWorkhorseV2ExecutionProviderV1(
            shape=shape
        ).execution_route_contract(base_plan)
    except Exception as error:
        raise WorkhorseSelectedHostContextError(
            f"Workhorse execution route is invalid: {error}"
        ) from error
    components = {
        str(row.get("component_id")): row
        for row in route.components
        if isinstance(row, Mapping)
    }
    artifact = components.get("cuda_full_evolution_workhorse_v2_artifact_unit")
    device = components.get("cuda_full_evolution_workhorse_v2_device_selection")
    checkpoint = components.get("cuda_full_evolution_workhorse_v2_checkpoint_science")
    absolute_time = components.get("absolute_tick_time_contract")
    route_operation = components.get("cuda_full_evolution_workhorse_v2_operation")
    acs_transport = components.get(
        "cuda_full_evolution_workhorse_v2_acs2_checkpoint_transport"
    )
    acs_runtime = components.get("acs_native_runtime_binding")
    if (
        not all(
            isinstance(row, Mapping)
            for row in (
                artifact,
                device,
                checkpoint,
                absolute_time,
                route_operation,
                acs_transport,
                acs_runtime,
            )
        )
        or absolute_time.get("time_authority") != "absolute_tick_v1"
        or absolute_time.get("start_next_tick") != 0
        or absolute_time.get("target_next_tick") != total_ticks
        or route_operation.get("operation") != EXECUTION_ROW_OPERATION_V2
        or route_operation.get("contract") != native_contract
        or route_operation.get("absolute_tick_span")
        != {
            "time_authority": "absolute_tick_v1",
            "start_next_tick": 0,
            "target_next_tick": total_ticks,
            "total_ticks": total_ticks,
        }
        or checkpoint.get("observation_mode") != "OFF"
        or checkpoint.get("terminal_next_tick") != total_ticks
        or device.get("B") != 1
        or device.get("temporal_k") != native_contract["temporal_k"]
        or device.get("execution_mode") not in profile.backend_ids
        or acs_transport.get("acs_checkpoint_slots") != 2
    ):
        raise WorkhorseSelectedHostContextError(
            "Workhorse route lost its singleton OFF/ACS2 identity"
        )
    storage = lifecycle.get("storage")
    if not isinstance(storage, Mapping) or type(storage.get("total_max_bytes")) is not int:
        raise WorkhorseSelectedHostContextError(
            "Workhorse authority lifecycle storage is invalid"
        )

    controls = tuple(
        sorted(
            (
                replace(
                    descriptor,
                    schedule_kinds=_AUTHORIZED_SCHEDULE_KINDS,
                )
                for descriptor in generic_cuda_v2_science_control_descriptors_v1()
                if descriptor.control_id in _AUTHORIZED_CONTROL_IDS
            ),
            key=lambda descriptor: descriptor.control_id,
        )
    )
    if tuple(descriptor.control_id for descriptor in controls) != _AUTHORIZED_CONTROL_IDS:
        raise WorkhorseSelectedHostContextError(
            "selected-world Workhorse control descriptors drifted"
        )
    reducer = compact_world_atlas_reducer_descriptor_v1()
    observation = _selected_observation_descriptor_v1()
    world_profile_id = (
        f"eiel-science-world-shape-w{profile.width}x{profile.height}-"
        f"p{profile.initial_population}-c{profile.capacity}-v1"
    )
    world_profile = {
        "world_profile_id": world_profile_id,
        "carrier_profile_id": profile.carrier_profile_id,
        "native_profile_id": profile.native_profile_id,
        "geometry": {"width": profile.width, "height": profile.height},
        "initial_population_max": profile.initial_population,
        "resident_capacity": profile.capacity,
        "authority_scope": "selected_world_exact_off_host_observation_only_v1",
    }
    observation_abi = _source_closure_sha256_v1()
    component_graph = {
        "instrument": "EIEL_SCIENCE_INSTRUMENT_V1",
        "context_scope_id": context_scope_id,
        "world_profile": world_profile,
        "control_descriptors": [row.descriptor_sha256().to_dict() for row in controls],
        "reducer_descriptor": reducer.descriptor_sha256().to_dict(),
        "route_profile_sha256": route.execution_route_sha256,
        "observation_authority": "authenticated_workhorse_host_boundary_v1",
    }
    manifest = ScienceInstrumentManifestV1(
        component_graph_sha256=canonical_sha256(component_graph),
        control_catalog_sha256=science_control_catalog_sha256(controls),
        world_profile_catalog_sha256=canonical_sha256([world_profile]),
        founder_profile_catalog_sha256=canonical_sha256([]),
        policy_profile_catalog_sha256=canonical_sha256([]),
        mutation_operator_catalog_sha256=canonical_sha256(
            ["workhorse-selected-parent-preserved-rng-v1"]
        ),
        observation_abi_sha256=observation_abi,
        observation_profile_catalog_sha256=(
            science_observation_profile_catalog_sha256_v1((observation,))
        ),
        reducer_protocol_version=SchemaVersion("eiel_science_reducer_protocol_v1"),
        route_binding_protocol_version=SchemaVersion(
            "eiel_science_route_binding_v1"
        ),
        eed_release_status=EEDReleaseStatusV1.EXCLUDED,
    )
    adapter = ScienceQRRRouteAdapterV1(
        route_id=profile.native_route_id,
        machine_profile_id=BASELINE_PROFILE_ID,
        world_profile_id=world_profile_id,
        world_width=profile.width,
        maximum_batch_cells=1,
        task_seed_owner="task_environment",
        control_path_by_id=tuple(
            (control_id, f"science_world.controls.{control_id}")
            for control_id in _AUTHORIZED_CONTROL_IDS
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
            "eiel_science_workhorse_selected_route_controls_v1",
            {
                "cuda_native.science_observation_mode": "OFF",
                "cuda_native.temporal_k": native_contract["temporal_k"],
                "workhorse.operation": EXECUTION_ROW_OPERATION_V2,
                "workhorse.temporal_k": native_contract["temporal_k"],
            },
        ),
        acs_capture_instrumentation=True,
        acs_checkpoint_slots=2,
        living_lineage_canopy=True,
        lifecycle_total_max_bytes=int(storage["total_max_bytes"]),
    )
    resource = _resource_envelope_v1(
        logical_world_count=1,
        resident_capacity=profile.capacity,
        record_kinds=observation.record_kinds,
    )
    no_authority = _no_qualification_authority(profile.native_route_id)
    qualified = ScienceQualifiedEnvelopeProjectionV1(
        instrument_manifest_sha256=manifest.manifest_sha256(),
        route_id=profile.native_route_id,
        machine_profile_id=BASELINE_PROFILE_ID,
        qualification_authority_sha256=no_authority,
        rows=(),
    )
    route_digest = ContentDigest("sha256", route.execution_route_sha256)
    native_closure = canonical_sha256(
        {
            "artifact": dict(artifact),
            "acs_native_runtime": dict(acs_runtime),
        }
    )
    physical = _payload(
        str(PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION),
        {
            "schema_version": str(PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION),
            "route_id": profile.native_route_id,
            "provider_id": profile.provider_id,
            "provider_family": profile.provider_family,
            "native_profile_id": profile.native_profile_id,
            "carrier_profile_id": profile.carrier_profile_id,
            "native_abi": native_contract["native_abi"],
            "execution_backend": profile.backend_ids[0],
            "execution_mode": profile.backend_ids[0],
            "temporal_k": native_contract["temporal_k"],
            "geometry": {
                "width": profile.width,
                "height": profile.height,
                "initial_population": profile.initial_population,
                "resident_capacity": profile.capacity,
            },
            "target_execution_quantum": {
                "logical_bundle_size": 1,
                "target_physical_batch": 1,
            },
            "route_profile_sha256": route_digest.to_dict(),
            "route_lineage": {
                "current_host_execution_route_sha256": route_digest.to_dict(),
                "identity_rule": (
                    "exact Workhorse route, artifact, ACS2 runtime, and selected-world "
                    "host-observation boundary"
                ),
            },
            "native_artifact": {
                "path": artifact["artifact_path"],
                "binary_sha256": {
                    "algorithm": "sha256",
                    "hex": artifact["artifact_sha256"],
                },
                "sealed_artifact_identity_sha256": {
                    "algorithm": "sha256",
                    "hex": artifact["sealed_artifact_identity_sha256"],
                },
            },
            "device_identity": dict(device["device_identity"]),
            "selected_world_only": True,
            "observation_mode": "OFF_AUTHENTICATED_HOST_BOUNDARY_V1",
            "acs_checkpoint_slots": 2,
        }
    )
    capability_projection = canonical_sha256(
        {
            "native_profile_id": profile.native_profile_id,
            "native_route_id": profile.native_route_id,
            "carrier_profile_id": profile.carrier_profile_id,
            "controls": list(_AUTHORIZED_CONTROL_IDS),
            "schedule_kinds": list(_AUTHORIZED_SCHEDULE_KINDS),
            "selected_world_only": True,
            "observation_mode": "OFF_AUTHENTICATED_HOST_BOUNDARY_V1",
        }
    )
    route_binding = ScienceRouteBindingV1(
        instrument_manifest_sha256=manifest.manifest_sha256(),
        route_id=profile.native_route_id,
        route_profile_sha256=route_digest,
        machine_profile_id=BASELINE_PROFILE_ID,
        capability_projection_sha256=capability_projection,
        native_build_closure_sha256=native_closure,
        execution_adapter_sha256=adapter.execution_adapter_sha256(),
        control_adapter_sha256=adapter.control_adapter_sha256(),
        observation_adapter_sha256=observation_abi,
        checkpoint_adapter_sha256=canonical_sha256(
            {
                "schema_version": "eiel_workhorse_selected_checkpoint_adapter_v1",
                "checkpoint_ticks": list(checkpoint_ticks),
                "acs_checkpoint_slots": 2,
                "host_observation": True,
            }
        ),
        semantic_conformance_sha256=canonical_sha256(
            {
                "time_authority": "absolute_tick_v1",
                "total_ticks": total_ticks,
                "checkpoint_ticks": list(checkpoint_ticks),
                "selected_world_only": True,
                "authority": SHADOW_PROJECTION_AUTHORITY,
            }
        ),
        resource_envelope_sha256=canonical_sha256(resource.to_dict()),
        qualified_envelope_sha256=qualified.projection_sha256(),
        qualification_authority_sha256=no_authority,
    )
    available_science = tuple(
        sorted(
            {
                f"world_profile:{world_profile_id}",
                *(f"control:{control_id}" for control_id in _AUTHORIZED_CONTROL_IDS),
                *(f"schedule:{kind}" for kind in _AUTHORIZED_SCHEDULE_KINDS),
                "absolute_tick_restore_v1",
                "scientific_checkpoint_fork_v1",
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
            {"component": WORKHORSE_SELECTED_HOST_CONTEXT_ID_V1, "version": 1}
        ),
        qualified_envelope=qualified,
        available_scientific_capability_ids=available_science,
        available_observation_capability_ids=observation.capability_ids,
        machine_profile_sha256=canonical_sha256(
            {
                "profile_id": BASELINE_PROFILE_ID,
                "identity_scope": "semantic_machine_profile",
            }
        ),
        admission_kernel_receipt_sha256=canonical_sha256(
            {
                "schema_version": "eiel_attempt_admission_kernel_v1",
                "authority": "canonical_crca_attempt_admission",
                "status": "HOST_CONTEXT_BINDING_ONLY",
                "route_id": profile.native_route_id,
            }
        ),
    )


__all__ = (
    "WORKHORSE_SELECTED_HOST_CONTEXT_ID_V1",
    "WorkhorseSelectedHostContextError",
    "materialize_workhorse_selected_host_context_v1",
)
