"""One parameterized Science Instrument carrier family over canonical CUDA V2."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Final

from emergence_contracts import (
    ContentDigest,
    ScienceAdmissionOutcomeV1,
    ScienceCudaV2CarrierProfileV1,
    ScienceCudaV2ExecutionPlanV1,
    ScienceCudaV2PhysicalBatchV1,
    ScienceCudaV2WorldMappingV1,
    ScienceWorldBundleV1,
    canonical_sha256,
)
from emergence_contracts.adaptive_science_expedition_v1 import StateRefV1
from emergence_contracts.artifacts import RunArtifactRefV1
from engine.campaigns.canonical import canonical_sha256 as campaign_sha256
from engine.campaigns.artifact_lifecycle import (
    CHECKPOINT_TRANSIT,
    SCIENCE_COMPACT,
    lifecycle_profile_contract_v1,
    validate_run_artifact_lifecycle_contract_v1,
)
from engine.campaigns.cuda_full_evolution_v2_provider import (
    PROFILE_ID as MICRO_PROFILE_ID,
    PROVIDER_ID as MICRO_PROVIDER_ID,
    cuda_full_evolution_v2_materialized_resource_estimate_v1,
)
from engine.campaigns.cuda_full_evolution_v2_workhorse_provider import (
    EXECUTION_ROW_OPERATION_V2,
    INDEXED_ACS_CONTINUATION_AGGREGATE_WTPS_METRIC_V2,
    INDEXED_ACS_CONTINUATION_OPERATION_V2,
    OPERATION_EXTENSION_ID,
    PROVIDER_ID as WORKHORSE_PROVIDER_ID,
    FullEvolutionCudaWorkhorseContractV2,
    compile_workhorse_absolute_dispatch_plan_v2,
    validate_authoritative_micro_denominator_v2,
    validate_workhorse_device_envelope_calibration_receipt_v2,
    validate_workhorse_semantic_gate_receipt_v2,
    workhorse_artifact_binding_v2,
    workhorse_indexed_continuation_contract_v2,
    workhorse_repack_binding_v2,
)
from engine.campaigns.cuda_full_evolution_v2_workhorse_resources import (
    workhorse_checkpoint_storage_accounting_v2,
)
from engine.campaigns.providers import (
    ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1,
    ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1,
)
from engine.perf.cuda_full_evolution_v2.contract import (
    AGENT_CAPACITY_V2,
    CUDA_NATIVE_PROFILE_ID_V2,
    CUDA_NATIVE_ROUTE_V2,
    INITIAL_POPULATION_V2,
    TEMPORAL_K_MAX_V2,
    TEMPORAL_K_MIN_V2,
    WORLD_WIDTH_V2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.contract import (
    BACKEND_PHASE_GRAPH_V2,
    WORKHORSE_TEMPORAL_K_CANDIDATES_V2,
    workhorse_shape_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.continuation import (
    workhorse_scientific_control_contract_v1,
)

from .science_world_bundle_v1 import (
    ScienceWorldBundleAdmissionV1,
    ScienceWorldBundleError,
)
from .science_cuda_v2_native_control_v1 import (
    ScienceCudaV2NativeControlError,
    derive_science_cuda_v2_native_rng_identity_v1,
    project_science_cuda_v2_native_control_v1,
)
from engine.campaigns.science_cuda_v2_native_world_v1 import (
    ScienceCudaV2NativeWorldError,
    native_world_control_set_sha256_v1,
    project_science_cuda_v2_native_world_v1,
)

from .adaptive_expedition.continuation_adapter import (
    SelectedWorldBranchError,
    resolve_workhorse_selected_world_continuation_v1,
)
from .workhorse_v2_materialize import (
    _apply_workhorse_storage_accounting_v2,
    _build_indexed_continuation_operation_v2,
    _workhorse_wall_seconds_upper_bound_v2,
)


SCIENCE_CUDA_V2_ADAPTER_ID: Final[str] = (
    "eiel_science_cuda_full_evolution_v2_carrier_adapter_v1"
)
SCIENCE_CUDA_V2_MICRO_BACKEND_ID: Final[str] = "direct_sparse_v3"
SCIENCE_CUDA_V2_TARGET_BATCH: Final[int] = 1024
SCIENCE_CUDA_V2_OBSERVATION_OFF_V1: Final[str] = "OFF"
SCIENCE_CUDA_V2_OBSERVATION_TYPED_SIDECAR_V1: Final[str] = "TYPED_SIDECAR_V1"
SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1: Final[str] = (
    "COMPACT_SIDECAR_V1"
)
SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1: Final[str] = (
    "ENGINEERING_DEMONSTRATION"
)
SCIENCE_CUDA_V2_USE_CLASS_EXPLORATORY_SCIENCE_V1: Final[str] = "EXPLORATORY_SCIENCE"
SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1: Final[str] = (
    "SCIENTIFIC_EXECUTION"
)
SCIENCE_CUDA_V2_USE_CLASSES_V1: Final[tuple[str, str, str]] = (
    SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1,
    SCIENCE_CUDA_V2_USE_CLASS_EXPLORATORY_SCIENCE_V1,
    SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1,
)
SCIENCE_CUDA_V2_CANONICAL_PARAMETER_IDS_V1: Final[frozenset[str]] = frozenset(
    {
        "cuda_native.temporal_k",
        "cuda_native.execution_mode",
        "cuda_native.event_transport",
        "cuda_native.science_observation_mode",
    }
)


def _bind_executable_materialized_resource_envelope_v1(
    plan: dict[str, object],
) -> None:
    """Bind post-lowering acquisition costs to budgets and lifecycle storage."""

    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping) or not all(
        isinstance(artifacts.get(name), Mapping)
        for name in (
            "cuda_extension",
            "extension_manifest",
            "device_selection_receipt",
        )
    ):
        # Host-only carrier projections deliberately omit native artifacts.
        return
    estimate = cuda_full_evolution_v2_materialized_resource_estimate_v1(plan)
    required = {
        name: estimate.get(name)
        for name in (
            "vram_bytes_upper_bound",
            "wall_seconds_upper_bound",
            "disk_bytes_upper_bound",
            "event_bytes_upper_bound",
            "checkpoint_bytes_upper_bound",
            "fixed_output_bytes_upper_bound",
        )
    }
    if any(type(value) is not int or value < 0 for value in required.values()):
        raise ScienceWorldBundleError(
            "provider returned an incomplete materialized resource estimate"
        )

    execution = plan.get("execution")
    if not isinstance(execution, dict):
        raise ScienceWorldBundleError("lowered execution contract is invalid")
    budgets = execution.get("budgets")
    if not isinstance(budgets, Mapping):
        raise ScienceWorldBundleError("lowered execution budgets are invalid")
    rebound_budgets = dict(budgets)
    rebound_budgets["vram_bytes"] = max(
        int(rebound_budgets.get("vram_bytes", 0)),
        int(required["vram_bytes_upper_bound"]),
    )
    rebound_budgets["disk_bytes"] = int(required["disk_bytes_upper_bound"])
    rebound_budgets["wall_seconds"] = max(
        int(rebound_budgets.get("wall_seconds", 0)),
        int(required["wall_seconds_upper_bound"]),
    )
    execution["budgets"] = rebound_budgets

    resource = dict(plan.get("resource_estimate", {}))
    resource.update(estimate)
    durable_noncheckpoint = (
        int(required["disk_bytes_upper_bound"])
        - int(required["event_bytes_upper_bound"])
        - int(required["checkpoint_bytes_upper_bound"])
    )
    if durable_noncheckpoint < int(required["fixed_output_bytes_upper_bound"]):
        raise ScienceWorldBundleError(
            "provider materialized disk accounting is internally inconsistent"
        )
    resource.update(
        {
            "authority": "provider_bound_post_science_lowering_v1",
            "durable_noncheckpoint_bytes_upper_bound": durable_noncheckpoint,
            "declared_vram_bytes": rebound_budgets["vram_bytes"],
            "declared_disk_bytes": rebound_budgets["disk_bytes"],
            "declared_wall_seconds": rebound_budgets["wall_seconds"],
        }
    )

    # Events are run-local scratch; checkpoints and compact evidence have
    # distinct owners. Include a bounded control-file allowance, then solve
    # the SCIENCE_COMPACT five-percent emergency reserve exactly.
    scratch_required = int(required["event_bytes_upper_bound"]) + 64 * 1024**2
    checkpoint_required = int(required["checkpoint_bytes_upper_bound"])
    committed = scratch_required + checkpoint_required + durable_noncheckpoint
    total = max(4, (committed * 20 + 18) // 19)
    while total - max(1, total // 20) < committed:
        total += 1
    lifecycle = lifecycle_profile_contract_v1(
        SCIENCE_COMPACT,
        total_max_bytes=total,
        checkpoint_schedule=tuple(execution.get("checkpoint_ticks", ())),
        checkpoint_authority="absolute_tick",
    )
    emergency = int(lifecycle["storage"]["emergency_reserve_bytes"])
    lifecycle["storage"] = {
        "total_max_bytes": total,
        "scratch_max_bytes": scratch_required,
        "checkpoint_max_bytes": checkpoint_required,
        "durable_max_bytes": (
            total - scratch_required - checkpoint_required - emergency
        ),
        "emergency_reserve_bytes": emergency,
    }
    lifecycle.pop("contract_sha256", None)
    rebound_lifecycle = validate_run_artifact_lifecycle_contract_v1(lifecycle)
    # The execution disk budget is the complete bounded lifecycle envelope;
    # the resource estimate remains the provider's planned produced bytes.
    rebound_budgets["disk_bytes"] = total
    resource["declared_disk_bytes"] = total
    plan["artifact_lifecycle"] = rebound_lifecycle
    resource.update(
        {
            "lifecycle_profile": SCIENCE_COMPACT,
            "lifecycle_contract_sha256": rebound_lifecycle["contract_sha256"],
            "lifecycle_total_max_bytes": total,
            "lifecycle_scratch_max_bytes": scratch_required,
            "lifecycle_checkpoint_max_bytes": checkpoint_required,
            "lifecycle_durable_max_bytes": rebound_lifecycle["storage"][
                "durable_max_bytes"
            ],
            "lifecycle_emergency_reserve_bytes": emergency,
        }
    )
    plan["resource_estimate"] = resource
    materialization = plan.get("science_materialization")
    if not isinstance(materialization, dict):
        raise ScienceWorldBundleError(
            "materialized resource binding lacks science materialization metadata"
        )
    binding = {
        "schema_version": "eiel_science_materialized_resource_binding_v1",
        "resource_estimator_id": estimate.get("resource_estimator_id"),
        "physical_batch_cells": estimate.get("physical_batch_cells"),
        "vram_bytes_upper_bound": required["vram_bytes_upper_bound"],
        "disk_bytes_upper_bound": required["disk_bytes_upper_bound"],
        "wall_seconds_upper_bound": required["wall_seconds_upper_bound"],
        "lifecycle_contract_sha256": rebound_lifecycle["contract_sha256"],
    }
    materialization["materialized_resource_binding"] = {
        **binding,
        "binding_sha256": canonical_sha256(binding).to_dict(),
    }


def _profile_catalog_v1() -> tuple[ScienceCudaV2CarrierProfileV1, ...]:
    if MICRO_PROFILE_ID != CUDA_NATIVE_PROFILE_ID_V2:
        raise ScienceWorldBundleError("CUDA V2 micro provider profile identity drifted")
    w64 = workhorse_shape_v2("w64h64p1024c2048")
    w128 = workhorse_shape_v2("w128h80p2560c5120")
    profiles = (
        ScienceCudaV2CarrierProfileV1(
            carrier_profile_id="w32x32p256c512_gpuv2",
            native_shape_id="w32h32p256c512",
            provider_family="cuda_full_evolution_v2",
            provider_id=MICRO_PROVIDER_ID,
            native_profile_id=CUDA_NATIVE_PROFILE_ID_V2,
            native_route_id=CUDA_NATIVE_ROUTE_V2,
            width=WORLD_WIDTH_V2,
            height=WORLD_WIDTH_V2,
            initial_population=INITIAL_POPULATION_V2,
            capacity=AGENT_CAPACITY_V2,
            backend_ids=(SCIENCE_CUDA_V2_MICRO_BACKEND_ID,),
            temporal_k_values=tuple(
                range(TEMPORAL_K_MIN_V2, TEMPORAL_K_MAX_V2 + 1)
            ),
            target_physical_batch=SCIENCE_CUDA_V2_TARGET_BATCH,
        ),
        ScienceCudaV2CarrierProfileV1(
            carrier_profile_id="w64x64p1024c2048_gpuv2",
            native_shape_id=w64.shape_id,
            provider_family="cuda_full_evolution_v2_workhorse",
            provider_id=WORKHORSE_PROVIDER_ID,
            native_profile_id=w64.profile_id,
            native_route_id=w64.route_id,
            width=w64.width,
            height=w64.height,
            initial_population=w64.initial_population,
            capacity=w64.capacity,
            backend_ids=(BACKEND_PHASE_GRAPH_V2,),
            temporal_k_values=WORKHORSE_TEMPORAL_K_CANDIDATES_V2,
            target_physical_batch=SCIENCE_CUDA_V2_TARGET_BATCH,
        ),
        ScienceCudaV2CarrierProfileV1(
            carrier_profile_id="w128x80p2560c5120_gpuv2",
            native_shape_id=w128.shape_id,
            provider_family="cuda_full_evolution_v2_workhorse",
            provider_id=WORKHORSE_PROVIDER_ID,
            native_profile_id=w128.profile_id,
            native_route_id=w128.route_id,
            width=w128.width,
            height=w128.height,
            initial_population=w128.initial_population,
            capacity=w128.capacity,
            backend_ids=(BACKEND_PHASE_GRAPH_V2,),
            temporal_k_values=WORKHORSE_TEMPORAL_K_CANDIDATES_V2,
            target_physical_batch=SCIENCE_CUDA_V2_TARGET_BATCH,
        ),
    )
    if len({profile.carrier_profile_id for profile in profiles}) != len(profiles):
        raise ScienceWorldBundleError("CUDA V2 carrier profile catalog duplicates identity")
    return profiles


SCIENCE_CUDA_V2_PROFILES: Final[tuple[ScienceCudaV2CarrierProfileV1, ...]] = (
    _profile_catalog_v1()
)
SCIENCE_CUDA_V2_PROFILES_BY_ID: Final[dict[str, ScienceCudaV2CarrierProfileV1]] = {
    profile.carrier_profile_id: profile for profile in SCIENCE_CUDA_V2_PROFILES
}


def science_cuda_v2_carrier_profiles_for_world_v1(
    world: Mapping[str, object],
    *,
    observation_mode: str = SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1,
) -> tuple[ScienceCudaV2CarrierProfileV1, ...]:
    """Return catalog profiles that can carry one requested science world.

    Geometry and population are data in the canonical carrier profile catalog;
    this helper deliberately contains no campaign-specific shape table.  The
    observation-mode filter mirrors the existing lowering boundary: typed and
    compact sidecars are currently available only on the installed micro
    provider, while ``OFF`` remains a physical-carrier-only projection.
    """

    if not isinstance(world, Mapping):
        return ()
    required = ("width", "height", "population", "capacity")
    if any(type(world.get(field)) is not int for field in required):
        return ()
    requested = tuple(int(world[field]) for field in required)
    matches: list[ScienceCudaV2CarrierProfileV1] = []
    for profile in SCIENCE_CUDA_V2_PROFILES:
        profile_world = (
            profile.width,
            profile.height,
            profile.initial_population,
            profile.capacity,
        )
        population_admitted = requested[2] == profile.initial_population
        if profile.provider_id == MICRO_PROVIDER_ID:
            population_admitted = 1 <= requested[2] <= profile.initial_population
        if requested[:2] != profile_world[:2] or requested[3] != profile.capacity:
            continue
        if not population_admitted:
            continue
        if observation_mode in {
            SCIENCE_CUDA_V2_OBSERVATION_TYPED_SIDECAR_V1,
            SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1,
        } and profile.provider_id != MICRO_PROVIDER_ID:
            continue
        matches.append(profile)
    return tuple(matches)


def science_cuda_v2_profile_catalog_sha256_v1():
    return canonical_sha256(
        {
            "adapter_id": SCIENCE_CUDA_V2_ADAPTER_ID,
            "profiles": [profile.to_dict() for profile in SCIENCE_CUDA_V2_PROFILES],
        }
    )


def _world_profile_v1(world) -> ScienceCudaV2CarrierProfileV1:
    semantic = world.semantic_inputs.to_value()
    world_input = semantic.get("world")
    initialization = semantic.get("initialization")
    if not isinstance(world_input, dict) or not isinstance(initialization, dict):
        raise ScienceWorldBundleError(f"{world.world_id}: world inputs are not objects")
    geometry = world_input.get("geometry")
    if not isinstance(geometry, dict):
        raise ScienceWorldBundleError(f"{world.world_id}: geometry is absent")
    selected_world = initialization.get("schema_version") == (
        "eiel_science_selected_world_initialization_v1"
    )
    population = initialization.get(
        "expected_initial_population" if selected_world else "initial_population"
    )
    key = (
        geometry.get("width"),
        geometry.get("height"),
        population,
        world_input.get("resident_capacity"),
    )
    matches = []
    for profile in SCIENCE_CUDA_V2_PROFILES:
        exact_shape = (
            geometry.get("width") == profile.width
            and geometry.get("height") == profile.height
            and world_input.get("resident_capacity") == profile.capacity
        )
        population_admitted = population == profile.initial_population
        if profile.provider_id == MICRO_PROVIDER_ID:
            population_admitted = (
                type(population) is int
                and 1 <= population <= profile.initial_population
            )
        if exact_shape and population_admitted:
            matches.append(profile)
    if len(matches) != 1:
        raise ScienceWorldBundleError(
            f"{world.world_id}: no exact CUDA V2 science carrier profile for {key}"
        )
    return matches[0]


def _requested_observations_v1(world) -> tuple[str, ...]:
    acquisition = world.acquisition.to_value()
    requested = set(acquisition["event_observations"])
    requested.update(acquisition["state_observations"])
    requested.update(acquisition["runtime_counters"])
    for reducer in world.reducers.to_value():
        requested.update(reducer["required_observations"])
    return tuple(sorted(requested))


def _world_mapping_v1(world) -> ScienceCudaV2WorldMappingV1:
    # The native V2 ABI uses a positive numeric world subject in its RNG
    # address.  Native-bound worlds derive it from the explicit initial-state
    # and science pairing identity; arm-specific semantic controls therefore
    # do not perturb paired OFF/ON RNG addressing.  The resolved seed bindings
    # are still required so a world cannot reach the native route without an
    # explicit deterministic seed contract.
    semantic_inputs = world.semantic_inputs.to_value()
    initialization = semantic_inputs.get("initialization")
    selected_world = (
        isinstance(initialization, Mapping)
        and initialization.get("schema_version")
        == "eiel_science_selected_world_initialization_v1"
    )
    seed_bindings = semantic_inputs.get("seed_bindings")
    if not isinstance(seed_bindings, dict):
        raise ScienceWorldBundleError(
            f"{world.world_id}: resolved seed bindings are absent"
        )
    task_seed = seed_bindings.get("task")
    if task_seed is None:
        task_seed = seed_bindings.get("task_environment")
    if not selected_world and (type(task_seed) is not int or task_seed < 0):
        raise ScienceWorldBundleError(
            f"{world.world_id}: resolved task seed must be a nonnegative integer"
        )
    state_ref: StateRefV1 | None = None
    if selected_world:
        try:
            state_ref = StateRefV1.from_dict(initialization["state_ref"])
        except Exception as exc:
            raise ScienceWorldBundleError(
                f"{world.world_id}: selected StateRef is invalid: {exc}"
            ) from exc
        semantic_world_uid = state_ref.logical_world_id
    else:
        semantic_world_uid = int(world.semantic_world_sha256.hex[:16], 16) & (
            (1 << 63) - 1
        )
    if semantic_world_uid == 0:
        raise ScienceWorldBundleError(
            f"{world.world_id}: semantic digest produced reserved world UID zero"
        )
    try:
        native_world = project_science_cuda_v2_native_world_v1(world)
    except ScienceCudaV2NativeWorldError as exc:
        raise ScienceWorldBundleError(
            f"{world.world_id}: native world projection is invalid: {exc}"
        ) from exc
    # Preserve the old scalar mutation projection when this world lies inside
    # that compatibility subset.  The generic per-world payload below is the
    # execution authority for heterogeneous native batches.
    try:
        native_control = project_science_cuda_v2_native_control_v1(world)
    except ScienceCudaV2NativeControlError:
        native_control = None
    native_uid: int | None = None
    native_rng_identity: ContentDigest | None = None
    if not selected_world:
        try:
            native_uid, native_rng_identity_hex = (
                derive_science_cuda_v2_native_rng_identity_v1(world)
            )
            native_rng_identity = ContentDigest(
                algorithm="sha256", hex=native_rng_identity_hex
            )
        except ScienceCudaV2NativeControlError as exc:
            raise ScienceWorldBundleError(
                f"{world.world_id}: native RNG identity is invalid: {exc}"
            ) from exc
    world_uid = semantic_world_uid
    return ScienceCudaV2WorldMappingV1(
        world_id=world.world_id,
        world_uid=world_uid,
        semantic_world_sha256=world.semantic_world_sha256,
        acquisition_sha256=world.acquisition_sha256,
        requested_observations=_requested_observations_v1(world),
        native_rng_world_uid=native_uid,
        native_rng_identity_sha256=native_rng_identity,
        native_control_vector=(
            None if native_control is None else native_control.control_vector
        ),
        native_control_digest=(
            None
            if native_control is None
            else ContentDigest(algorithm="sha256", hex=native_control.control_digest)
        ),
        native_capability_mask=(
            None if native_control is None else native_control.capability_mask
        ),
        native_world_control=native_world.to_dict(),
    )


def plan_science_cuda_v2_execution_v1(
    bundle: ScienceWorldBundleV1,
    admission: ScienceWorldBundleAdmissionV1,
    *,
    physical_batch_capacity: int | Mapping[str, int] = SCIENCE_CUDA_V2_TARGET_BATCH,
    temporal_k: int | Mapping[str, int] = 1,
    world_order: Iterable[str] | None = None,
    execution_quantum_id: str | None = None,
) -> ScienceCudaV2ExecutionPlanV1:
    """Partition admitted worlds by closed native profile without semantic mutation."""

    if admission.bundle_content_sha256 != bundle.bundle_content_sha256():
        raise ScienceWorldBundleError("CUDA V2 planning admission binds another bundle")
    if admission.outcome is not ScienceAdmissionOutcomeV1.ADMITTED:
        raise ScienceWorldBundleError("CUDA V2 planning requires ADMITTED worlds")
    world_by_id = {world.world_id: world for world in bundle.worlds}
    admitted = list(admission.admitted_world_ids)
    if world_order is None:
        ordered = admitted
    else:
        ordered = list(world_order)
        if sorted(ordered) != sorted(admitted):
            raise ScienceWorldBundleError("CUDA V2 world order is not an exact permutation")

    grouped: dict[tuple[str, str], list[ScienceCudaV2WorldMappingV1]] = defaultdict(list)
    seen_uids: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    for world_id in ordered:
        world = world_by_id[world_id]
        profile = _world_profile_v1(world)
        mapping = _world_mapping_v1(world)
        semantic_inputs = world.semantic_inputs.to_value()
        time = semantic_inputs.get("time")
        if not isinstance(time, dict):
            raise ScienceWorldBundleError(f"{world_id}: resolved time is absent")
        compatibility = canonical_sha256(
            {
                "profile_id": profile.carrier_profile_id,
                "time_authority": time.get("authority"),
                "start_next_tick": time.get("start_next_tick", 0),
                "total_ticks": time.get("total_ticks"),
            }
        ).hex
        group_key = (profile.carrier_profile_id, compatibility)
        prior = seen_uids[group_key].get(mapping.world_uid)
        if prior is not None and prior != world_id:
            raise ScienceWorldBundleError(
                f"world UID collision between {prior} and {world_id}"
            )
        seen_uids[group_key][mapping.world_uid] = world_id
        grouped[group_key].append(mapping)

    def selected(value: int | Mapping[str, int], profile_id: str, label: str) -> int:
        candidate = value.get(profile_id) if isinstance(value, Mapping) else value
        if type(candidate) is not int or candidate <= 0:
            raise ScienceWorldBundleError(f"{label} for {profile_id} must be positive")
        return candidate

    batches: list[ScienceCudaV2PhysicalBatchV1] = []
    for profile in SCIENCE_CUDA_V2_PROFILES:
        profile_groups = [
            (compatibility, mappings)
            for (profile_id, compatibility), mappings in grouped.items()
            if profile_id == profile.carrier_profile_id
        ]
        for _compatibility, mappings in sorted(profile_groups, key=lambda item: item[0]):
            if not mappings:
                continue
            capacity = selected(
                physical_batch_capacity, profile.carrier_profile_id, "physical capacity"
            )
            if capacity > profile.target_physical_batch:
                raise ScienceWorldBundleError(
                    f"physical capacity {capacity} exceeds current target for "
                    f"{profile.carrier_profile_id}"
                )
            selected_k = selected(temporal_k, profile.carrier_profile_id, "temporal K")
            if selected_k not in profile.temporal_k_values:
                raise ScienceWorldBundleError(
                    f"temporal K {selected_k} is unsupported for {profile.carrier_profile_id}"
                )
            for offset in range(0, len(mappings), capacity):
                batch_worlds = mappings[offset : offset + capacity]
                legacy_controls = {
                    (
                        tuple(sorted(world.native_control_vector.items()))
                        if world.native_control_vector is not None
                        else None,
                        world.native_control_digest,
                        world.native_capability_mask,
                    )
                    for world in batch_worlds
                }
                uniform_legacy_control = (
                    next(iter(legacy_controls))
                    if len(legacy_controls) == 1
                    else (None, None, None)
                )
                batches.append(
                    ScienceCudaV2PhysicalBatchV1(
                        physical_batch_index=len(batches),
                        carrier_profile=profile,
                        backend_id=profile.backend_ids[0],
                        temporal_k=selected_k,
                        world_slots=tuple(enumerate(batch_worlds)),
                        native_observation_union=tuple(
                            sorted(
                                {
                                    observation
                                    for world in batch_worlds
                                    for observation in world.requested_observations
                                }
                            )
                        ),
                        native_control_vector=(
                            dict(uniform_legacy_control[0])
                            if uniform_legacy_control[0] is not None
                            else None
                        ),
                        native_control_digest=uniform_legacy_control[1],
                        native_capability_mask=uniform_legacy_control[2],
                    )
                )
    return ScienceCudaV2ExecutionPlanV1(
        bundle_content_sha256=bundle.bundle_content_sha256(),
        bundle_admission_sha256=canonical_sha256(admission.to_dict()),
        carrier_profile_catalog_sha256=science_cuda_v2_profile_catalog_sha256_v1(),
        logical_bundle_size=len(admitted),
        batches=tuple(batches),
        execution_quantum_id=execution_quantum_id,
    )


def _selected_world_initialization_v1(world: object) -> dict[str, Any] | None:
    semantic = world.semantic_inputs.to_value()
    initialization = semantic.get("initialization")
    if not isinstance(initialization, Mapping) or initialization.get(
        "schema_version"
    ) != "eiel_science_selected_world_initialization_v1":
        return None
    return dict(initialization)


def _regular_file_sha256_v1(path: object, *, field: str) -> tuple[Path, str]:
    if not isinstance(path, str) or not path:
        raise ScienceWorldBundleError(f"{field} path is absent")
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ScienceWorldBundleError(f"{field} must not be a symbolic link")
    resolved = requested.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ScienceWorldBundleError(f"{field} is not a regular file")
    return resolved, hashlib.sha256(resolved.read_bytes()).hexdigest()


def _workhorse_template_bindings_v1(
    base_plan: Mapping[str, object],
    *,
    profile: ScienceCudaV2CarrierProfileV1,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reopen the exact Workhorse authority material used by a child plan."""

    analysis = base_plan.get("analysis")
    extension = analysis.get("extension") if isinstance(analysis, Mapping) else None
    raw = extension.get("contract") if isinstance(extension, Mapping) else None
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
    continuation_fields = (row_fields - {"measurement_schedule"}) | {
        "continuation_contract"
    }
    if (
        not isinstance(extension, Mapping)
        or extension.get("id") != OPERATION_EXTENSION_ID
        or not isinstance(raw, Mapping)
        or raw.get("operation")
        not in {EXECUTION_ROW_OPERATION_V2, INDEXED_ACS_CONTINUATION_OPERATION_V2}
        or set(raw)
        != (
            row_fields
            if raw.get("operation") == EXECUTION_ROW_OPERATION_V2
            else continuation_fields
        )
        or extension.get("contract_sha256") != campaign_sha256(dict(raw))
    ):
        raise ScienceWorldBundleError(
            "selected-world lowering requires a sealed Workhorse row or "
            "indexed-continuation authority plan"
        )
    try:
        raw_artifact = raw["artifact"]
        artifact = workhorse_artifact_binding_v2(raw_artifact["path"])
        expected_artifact = {
            "path": artifact["path"],
            "sha256": artifact["sha256"],
            "sealed_artifact_identity_sha256": artifact[
                "sealed_artifact_identity_sha256"
            ],
        }
        envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
            raw["envelope"]["path"]
        )
        semantic = validate_workhorse_semantic_gate_receipt_v2(
            raw["semantic_gate_receipt"]["path"],
            expected_shape_id=profile.native_shape_id,
            expected_candidate_artifact_sha256=artifact["sha256"],
            expected_candidate_backend_id=str(artifact["identity"]["backend_id"]),
        )
        micro = validate_authoritative_micro_denominator_v2(
            raw["authoritative_micro_receipt"]["path"],
            expected_device_identity=envelope["device_identity"],
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ScienceWorldBundleError(
            f"selected-world Workhorse authority cannot reopen exactly: {error}"
        ) from error
    if (
        dict(raw_artifact) != expected_artifact
        or envelope.get("artifact") != artifact
        or artifact.get("identity", {}).get("shape_id") != profile.native_shape_id
        or artifact.get("identity", {}).get("backend_id")
        not in profile.backend_ids
        or raw["envelope"]
        != {
            "path": envelope["receipt_path"],
            "file_sha256": envelope["receipt_file_sha256"],
            "content_sha256": envelope["receipt_content_sha256"],
        }
        or raw["authoritative_micro_receipt"]
        != {
            "path": micro["receipt_path"],
            "file_sha256": micro["receipt_file_sha256"],
            "content_sha256": micro["receipt_content_sha256"],
        }
        or raw["semantic_gate_receipt"]
        != {
            "path": semantic["receipt_path"],
            "file_sha256": semantic["receipt_file_sha256"],
            "content_sha256": semantic["semantic_gate_receipt_sha256"],
        }
    ):
        raise ScienceWorldBundleError(
            "selected-world Workhorse authority bindings changed"
        )
    artifacts = base_plan.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ScienceWorldBundleError(
            "selected-world Workhorse authority lacks ACS2 native artifacts"
        )
    for artifact_id in (
        ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1,
        ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1,
    ):
        binding = artifacts.get(artifact_id)
        if not isinstance(binding, Mapping):
            raise ScienceWorldBundleError(
                "selected-world Workhorse authority lacks ACS2 native artifacts"
            )
        _path, file_sha256 = _regular_file_sha256_v1(
            binding.get("path"), field=artifact_id
        )
        if binding.get("sha256") != file_sha256:
            raise ScienceWorldBundleError(
                f"selected-world Workhorse {artifact_id} changed"
            )
    return artifact, envelope, micro, semantic


def _selected_workhorse_control_values_v1(
    native_world_control: Mapping[str, object], *, branch_tick: int
) -> tuple[float, float]:
    rows = native_world_control.get("control_bindings")
    if not isinstance(rows, list):
        raise ScienceWorldBundleError(
            "selected-world controls are absent from the native projection"
        )
    values: dict[str, float] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ScienceWorldBundleError("selected-world control binding is invalid")
        control_id = raw.get("control_id")
        if control_id not in {
            "environment.resource.multiplier",
            "mutation.rate",
        }:
            raise ScienceWorldBundleError(
                f"selected-world Workhorse control {control_id!r} is unsupported"
            )
        schedule = raw.get("schedule")
        resolved = raw.get("resolved_value")
        if not isinstance(schedule, Mapping):
            raise ScienceWorldBundleError("selected-world control schedule is absent")
        if schedule.get("kind") == "static" and set(schedule) == {"kind"}:
            pass
        elif schedule.get("kind") == "piecewise_constant_absolute_ticks":
            segments = schedule.get("segments")
            if (
                not isinstance(segments, list)
                or len(segments) != 1
                or not isinstance(segments[0], Mapping)
                or set(segments[0]) != {"start_absolute_tick", "value"}
                or segments[0].get("start_absolute_tick") != branch_tick
                or segments[0].get("value") != resolved
            ):
                raise ScienceWorldBundleError(
                    "selected-world controls may change only once at the branch tick"
                )
        else:
            raise ScienceWorldBundleError(
                "selected-world controls require a static or branch-tick-only schedule"
            )
        if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
            raise ScienceWorldBundleError("selected-world control value is not numeric")
        values[str(control_id)] = float(resolved)
    if set(values) != {
        "environment.resource.multiplier",
        "mutation.rate",
    }:
        raise ScienceWorldBundleError(
            "selected-world continuation requires exactly resource multiplier and mutation rate"
        )
    return (
        values["environment.resource.multiplier"],
        values["mutation.rate"],
    )


def _lower_selected_workhorse_batch_v1(
    base_plan: Mapping[str, object],
    bundle: ScienceWorldBundleV1,
    admission: ScienceWorldBundleAdmissionV1,
    carrier_plan: ScienceCudaV2ExecutionPlanV1,
    *,
    physical_batch_index: int,
    observation_mode: str,
    science_use_class: str,
) -> dict[str, object]:
    """Lower one exact retained Workhorse row as a technical child campaign."""

    batch = carrier_plan.batches[physical_batch_index]
    profile = batch.carrier_profile
    if (
        profile.provider_id != WORKHORSE_PROVIDER_ID
        or batch.physical_batch_size != 1
        or len(carrier_plan.batches) != 1
        or bundle.world_count != 1
    ):
        raise ScienceWorldBundleError(
            "V1 selected-world continuation requires one Workhorse parent per child request"
        )
    if (
        observation_mode != SCIENCE_CUDA_V2_OBSERVATION_OFF_V1
        or science_use_class
        != SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1
    ):
        raise ScienceWorldBundleError(
            "selected-world Workhorse continuation is technical-only until its "
            "authenticated observation adapter exists"
        )
    ordinal, mapping = batch.world_slots[0]
    world_by_id = {world.world_id: world for world in bundle.worlds}
    world = world_by_id[mapping.world_id]
    initialization = _selected_world_initialization_v1(world)
    if initialization is None:
        raise ScienceWorldBundleError("selected-world initialization is absent")
    try:
        state_ref = StateRefV1.from_dict(initialization["state_ref"])
        authority_ref = RunArtifactRefV1.from_dict(
            initialization["authority_receipt_ref"]
        )
        resolved = resolve_workhorse_selected_world_continuation_v1(
            state_ref=state_ref,
            authority_receipt_ref=authority_ref,
        )
    except (KeyError, SelectedWorldBranchError, TypeError, ValueError) as error:
        raise ScienceWorldBundleError(
            f"selected-world continuation authority is invalid: {error}"
        ) from error
    semantic = world.semantic_inputs.to_value()
    time = semantic.get("time")
    if not isinstance(time, Mapping):
        raise ScienceWorldBundleError("selected-world absolute-time contract is absent")
    start_next_tick = time.get("start_next_tick")
    total_ticks = time.get("total_ticks")
    checkpoint_ticks = time.get("checkpoint_ticks")
    if (
        start_next_tick != state_ref.next_absolute_tick
        or mapping.world_uid != state_ref.logical_world_id
        or type(total_ticks) is not int
        or total_ticks <= 0
        or not isinstance(checkpoint_ticks, list)
        or not checkpoint_ticks
        or any(type(tick) is not int for tick in checkpoint_ticks)
    ):
        raise ScienceWorldBundleError(
            "selected-world carrier differs from its StateRef frontier"
        )
    target_next_tick = start_next_tick + total_ticks
    if (
        checkpoint_ticks != sorted(set(checkpoint_ticks))
        or checkpoint_ticks[-1] != target_next_tick
        or any(
            tick <= start_next_tick or tick > target_next_tick
            for tick in checkpoint_ticks
        )
        or time.get("deliberate_restore_ticks") != [start_next_tick]
    ):
        raise ScienceWorldBundleError(
            "selected-world checkpoints do not exactly cover the child horizon"
        )
    if not isinstance(mapping.native_world_control, Mapping):
        raise ScienceWorldBundleError(
            "selected-world native control projection is absent"
        )
    resource_multiplier, mutation_rate = _selected_workhorse_control_values_v1(
        mapping.native_world_control,
        branch_tick=start_next_tick,
    )
    artifact, envelope, micro, semantic_gate = _workhorse_template_bindings_v1(
        base_plan,
        profile=profile,
    )
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=profile.native_shape_id,
        backend_id=batch.backend_id,
        B=1,
        temporal_k=batch.temporal_k,
        total_ticks=total_ticks,
        start_next_tick=start_next_tick,
    )
    admitted_temporal_k = sorted(
        {
            row["temporal_k"]
            for row in envelope["expected_rows"]
            if row.get("B") == 1
            and type(row.get("temporal_k")) is int
            and row["temporal_k"] <= contract.temporal_k
            and isinstance(row.get("planned_admission"), Mapping)
            and row["planned_admission"].get("admitted") is True
        }
    )
    if 1 not in admitted_temporal_k or contract.temporal_k not in admitted_temporal_k:
        raise ScienceWorldBundleError(
            "selected-world Workhorse envelope must admit B1 at K1 and preferred K"
        )
    try:
        repack = workhorse_repack_binding_v2(
            [resolved.manifest_path],
            ordered_world_uids=[state_ref.logical_world_id],
            target_contract=contract,
        )
        dispatch = compile_workhorse_absolute_dispatch_plan_v2(
            start_next_tick=start_next_tick,
            target_next_tick=target_next_tick,
            preferred_temporal_k=contract.temporal_k,
            admitted_temporal_k=admitted_temporal_k,
            boundary_next_ticks=checkpoint_ticks,
        )
        scientific_control = workhorse_scientific_control_contract_v1(
            effective_next_absolute_tick=start_next_tick,
            resource_multiplier=resource_multiplier,
            mutation_rate=mutation_rate,
        )
        continuation = workhorse_indexed_continuation_contract_v2(
            science_continuation_contract_sha256=mapping.semantic_world_sha256.hex,
            repack_binding=repack,
            dispatch_plan=dispatch,
            checkpoint_ticks=checkpoint_ticks,
            scientific_control_contract=scientific_control,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise ScienceWorldBundleError(
            f"selected-world Workhorse continuation cannot be represented: {error}"
        ) from error
    operation = _build_indexed_continuation_operation_v2(
        contract=contract,
        artifact=artifact,
        envelope=envelope,
        authoritative_micro=micro,
        semantic_gate=semantic_gate,
        science_continuation_sha256=mapping.semantic_world_sha256.hex,
        continuation=continuation,
    )
    parameters = {
        "workhorse.operation": INDEXED_ACS_CONTINUATION_OPERATION_V2,
        "workhorse.shape_id": contract.shape_id,
        "workhorse.backend_id": contract.backend_id,
        "workhorse.B": 1,
        "workhorse.temporal_k": contract.temporal_k,
    }
    cell = {
        "ordinal": ordinal,
        "cell_id": mapping.world_id,
        "branch_id": f"selected-child-{mapping.world_id}",
        "rng_lineage_id": f"retained-rng-{state_ref.world_trajectory_id}",
        "dimensions": dict(semantic.get("dimension_bindings") or {}),
        "parameters": parameters,
        "seeds": {},
        "science_seed_bindings": {},
        "science_world_id": mapping.world_id,
        "science_logical_world_id": state_ref.logical_world_id,
        "native_rng_world_uid": None,
        "native_rng_identity_sha256": None,
        "native_world_control": dict(mapping.native_world_control),
        "science_world_sha256": mapping.semantic_world_sha256.to_dict(),
        "science_acquisition_sha256": mapping.acquisition_sha256.to_dict(),
        "science_requested_observations": list(mapping.requested_observations),
        "science_ownership": world.science.to_value(),
        "science_semantic_inputs": semantic,
        "science_acquisition": world.acquisition.to_value(),
        "science_reducers": world.reducers.to_value(),
    }
    bound = deepcopy(dict(base_plan))
    bound["campaign_id"] = (
        f"eiel-science-{bundle.bundle_id.lower()}-selected-batch-"
        f"{physical_batch_index:04d}"
    )
    bound["description"] = (
        "Technical exact selected-world Workhorse child through the canonical "
        "ExecutionRequestV2 provider route."
    )
    bound["no_claims"] = (
        "TECHNICAL_ONLY. Exact branch engineering evidence; no scientific, "
        "production, QualPrep, or Qualification authority."
    )
    bound["resolved_profile"] = profile.native_profile_id
    bound["cells"] = [cell]
    bound["physical_batches"] = [
        {
            "ordinal": 0,
            "batch_id": f"science-selected-batch-{physical_batch_index:04d}",
            "cell_ids": [mapping.world_id],
        }
    ]
    # A selected child has no generated comparison branches, but the canonical
    # execution/CRCA contracts still require the campaign-plan comparison
    # projection to be an object.  Preserve the route-template reporting
    # parity declaration while clearing any authoring-time branches so this
    # fresh child remains a single canonical run.
    base_comparisons = base_plan.get("comparisons", {})
    if isinstance(base_comparisons, Mapping):
        bound["comparisons"] = {**dict(base_comparisons), "branches": []}
    elif isinstance(base_comparisons, list) and not base_comparisons:
        # Older in-memory test/template plans used an empty list.  Normalize
        # that inert legacy projection at this private lowering boundary.
        bound["comparisons"] = {"branches": []}
    else:
        raise ScienceWorldBundleError(
            "base campaign plan comparisons must be an object or an empty legacy list"
        )
    execution = deepcopy(dict(base_plan.get("execution") or {}))
    budgets = deepcopy(dict(execution.get("budgets") or {}))
    capacity = envelope.get("capacity_basis")
    if (
        not isinstance(capacity, Mapping)
        or type(capacity.get("planning_available_bytes")) is not int
        or type(capacity.get("effective_reserve_bytes")) is not int
    ):
        raise ScienceWorldBundleError(
            "selected-world Workhorse envelope lacks bounded device capacity"
        )
    budgets.update(
        {
            "vram_bytes": capacity["planning_available_bytes"],
            "minimum_free_gpu_bytes": capacity["effective_reserve_bytes"],
            "wall_seconds": max(
                int(budgets.get("wall_seconds", 0)),
                _workhorse_wall_seconds_upper_bound_v2(total_ticks),
            ),
        }
    )
    execution.update(
        {
            "execution_route_identity": "tick_native_execution_route_v1",
            "time_authority": "absolute_tick_v1",
            "total_ticks": target_next_tick,
            "start_next_tick": start_next_tick,
            "checkpoint_ticks": list(checkpoint_ticks),
            "deliberate_restore_tick": start_next_tick,
            "deliberate_restore_ticks": [start_next_tick],
            "resident_owner_span_ticks": total_ticks,
            "native_dispatch_window_ticks": contract.temporal_k,
            "source_window_ticks": contract.temporal_k,
            "event_window_capacity": 2 * contract.shape.capacity,
            "checkpoint_storage_mode": "disk",
            "acs_checkpoint_slots": 2,
            "native_streamed_checkpoint": True,
            "budgets": budgets,
            "tick_native_control": {
                "schema_version": (
                    "eiel_workhorse_indexed_continuation_tick_control_v2"
                ),
                "time_authority": "absolute_tick_v1",
                "operation": INDEXED_ACS_CONTINUATION_OPERATION_V2,
                "total_ticks": target_next_tick,
                "start_next_tick": start_next_tick,
                "checkpoint_ticks": list(checkpoint_ticks),
                "resident_owner_span_ticks": total_ticks,
                "native_dispatch_window_ticks": contract.temporal_k,
                "source_window_ticks": contract.temporal_k,
                "event_window_capacity": 2 * contract.shape.capacity,
                "dispatch_plan": dict(dispatch),
                "science_continuation_identity": dict(
                    continuation["science_continuation_identity"]
                ),
                "physical_continuation_plan": dict(
                    continuation["physical_continuation_plan"]
                ),
            },
        }
    )
    technical_gate = execution.get("technical_gate")
    if isinstance(technical_gate, Mapping):
        execution["technical_gate"] = {
            **dict(technical_gate),
            "total_ticks": target_next_tick,
            "checkpoint_ticks": list(checkpoint_ticks),
        }
    bound["execution"] = execution
    lifecycle = base_plan.get("artifact_lifecycle")
    base_storage = lifecycle.get("storage") if isinstance(lifecycle, Mapping) else None
    lifecycle_profile = (
        lifecycle.get("lifecycle_profile")
        if isinstance(lifecycle, Mapping)
        else CHECKPOINT_TRANSIT
    )
    if not isinstance(base_storage, Mapping) or type(
        base_storage.get("total_max_bytes")
    ) is not int:
        raise ScienceWorldBundleError(
            "selected-world Workhorse authority lacks bounded lifecycle storage"
        )
    rebound_lifecycle = lifecycle_profile_contract_v1(
        str(lifecycle_profile),
        total_max_bytes=base_storage["total_max_bytes"],
        checkpoint_schedule=checkpoint_ticks,
        checkpoint_authority="absolute_tick",
    )
    rebound_lifecycle["checkpoints"]["roles"] = [
        {
            "identity": f"checkpoint-tick-{tick:012d}",
            "checkpoint_tick": tick,
            "role": "terminal" if tick == target_next_tick else "intermediate",
            "retention_class": "R4",
        }
        for tick in checkpoint_ticks
    ]
    rebound_lifecycle.pop("contract_sha256", None)
    bound["artifact_lifecycle"] = validate_run_artifact_lifecycle_contract_v1(
        rebound_lifecycle
    )
    storage = workhorse_checkpoint_storage_accounting_v2(
        contract=contract,
        checkpoint_ticks=checkpoint_ticks,
        acs_checkpoint_slots=2,
        dispatch_count=dispatch["dispatch_count"],
    )
    _apply_workhorse_storage_accounting_v2(bound, storage)
    bound["artifact_lifecycle"].pop("contract_sha256", None)
    bound["artifact_lifecycle"] = validate_run_artifact_lifecycle_contract_v1(
        bound["artifact_lifecycle"]
    )
    admitted_row = next(
        row
        for row in envelope["expected_rows"]
        if row.get("B") == 1 and row.get("temporal_k") == contract.temporal_k
    )
    bound["resource_estimate"] = {
        "resource_estimator_id": "workhorse_v2_indexed_acs_continuation_v2",
        "vram_bytes_upper_bound": admitted_row["planned_admission"][
            "required_bytes"
        ],
        "wall_seconds_upper_bound": _workhorse_wall_seconds_upper_bound_v2(
            total_ticks
        ),
        **storage,
        "physical_batch_cells": 1,
        "workhorse_world_batch_B": 1,
        "workhorse_temporal_k": contract.temporal_k,
        "acs_checkpoint_slots": 2,
    }
    bound["analysis"] = {
        "required_metrics": [
            INDEXED_ACS_CONTINUATION_AGGREGATE_WTPS_METRIC_V2
        ],
        "optional_metrics": [],
        "rules": [],
        "extension": {
            "id": OPERATION_EXTENSION_ID,
            "contract": deepcopy(operation),
            "contract_sha256": campaign_sha256(operation),
        },
    }
    bound["science_materialization"] = {
        "schema_version": "eiel_science_selected_world_technical_lowering_v1",
        "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
        "bundle_admission_sha256": canonical_sha256(admission.to_dict()).to_dict(),
        "physical_execution_plan_sha256": carrier_plan.plan_sha256().to_dict(),
        "physical_batch_index": physical_batch_index,
        "carrier_profile_id": profile.carrier_profile_id,
        "semantic_world_id": mapping.world_id,
        "observation_mode": SCIENCE_CUDA_V2_OBSERVATION_OFF_V1,
        "science_use_class": science_use_class,
        "science_authority_granted": False,
        "qualification_authority_granted": False,
        "state_ref": state_ref.to_dict(),
        "authority_receipt_ref": authority_ref.to_dict(),
        "parent_run_id": str(resolved.parent_run_id),
        "source_checkpoint_occurrence_sha256": (
            resolved.checkpoint_occurrence_sha256.to_dict()
        ),
        "source_parent_occurrence_sha256": (
            resolved.parent_occurrence_sha256.to_dict()
        ),
        "first_permitted_divergence_tick": start_next_tick,
        "scientific_control_contract_sha256": {
            "algorithm": "sha256",
            "hex": scientific_control["control_contract_sha256"],
        },
        "parent_lifecycle_requirement": (
            "RETAIN_R4_UNTIL_CHILD_TERMINAL_AND_NO_LIVE_CONTINUATION"
        ),
        "child_identity_rule": "FRESH_RUN_AND_ATTEMPT_NEVER_PARENT_RETRY",
        "repacking_semantics": "LOGICAL_IDENTITY_EXCLUDES_PHYSICAL_SLOT",
        "fallback_policy": "FORBIDDEN",
    }
    bound["science_use_class"] = science_use_class
    bound.pop("plan_sha256", None)
    bound["plan_sha256"] = campaign_sha256(bound)
    return bound


def lower_science_cuda_v2_batch_to_campaign_plan_v1(
    base_plan: Mapping[str, object],
    bundle: ScienceWorldBundleV1,
    admission: ScienceWorldBundleAdmissionV1,
    carrier_plan: ScienceCudaV2ExecutionPlanV1,
    *,
    physical_batch_index: int,
    execution_mode: str | None = None,
    event_transport: str | None = None,
    observation_mode: str = "OFF",
    science_use_class: str = SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1,
    instrument_manifest_sha256: ContentDigest | None = None,
    crca_semantic_run_digest: ContentDigest | None = None,
    _precomputed_bundle_content_sha256: ContentDigest | None = None,
    _precomputed_carrier_plan_sha256: ContentDigest | None = None,
) -> dict[str, object]:
    """Lower one explicit-world batch to the canonical V2 provider plan.

    The existing campaign compiler remains the authoring compiler.  This
    narrow materialization step only replaces its logical cells and physical
    batch with the already-admitted explicit worlds; it never asks the native
    provider to interpret factorial/study logic.  A separate call is required
    only for each physically incompatible profile/horizon batch.  Controls,
    seeds, initialization, acquisition ownership, and reducers remain
    per-world and do not force a partition.
    """

    if not isinstance(base_plan, Mapping):
        raise ScienceWorldBundleError("base campaign plan must be an object")
    if science_use_class not in SCIENCE_CUDA_V2_USE_CLASSES_V1:
        raise ScienceWorldBundleError(
            "unsupported CUDA V2 science use class"
        )
    if science_use_class == SCIENCE_CUDA_V2_USE_CLASS_EXPLORATORY_SCIENCE_V1:
        if not isinstance(instrument_manifest_sha256, ContentDigest):
            raise ScienceWorldBundleError(
                "EXPLORATORY_SCIENCE lowering requires the exact instrument manifest digest"
            )
        if not isinstance(crca_semantic_run_digest, ContentDigest):
            raise ScienceWorldBundleError(
                "EXPLORATORY_SCIENCE lowering requires the exact CRCA semantic run digest"
            )
    bundle_content_sha256 = (
        bundle.bundle_content_sha256()
        if _precomputed_bundle_content_sha256 is None
        else _precomputed_bundle_content_sha256
    )
    carrier_plan_sha256 = (
        carrier_plan.plan_sha256()
        if _precomputed_carrier_plan_sha256 is None
        else _precomputed_carrier_plan_sha256
    )
    if (
        not isinstance(bundle_content_sha256, ContentDigest)
        or not isinstance(carrier_plan_sha256, ContentDigest)
        or admission.bundle_content_sha256 != bundle_content_sha256
    ):
        raise ScienceWorldBundleError("lowering admission binds another bundle")
    if admission.outcome is not ScienceAdmissionOutcomeV1.ADMITTED:
        raise ScienceWorldBundleError("lowering requires an admitted bundle")
    if not isinstance(physical_batch_index, int) or not 0 <= physical_batch_index < len(
        carrier_plan.batches
    ):
        raise ScienceWorldBundleError("physical batch index is outside the carrier plan")
    batch = carrier_plan.batches[physical_batch_index]
    profile = batch.carrier_profile
    expected_provider = profile.provider_id
    if base_plan.get("resolved_profile") != profile.native_profile_id:
        raise ScienceWorldBundleError(
            "base campaign plan profile does not match the selected carrier profile"
        )
    base_cells = [
        row for row in base_plan.get("cells", ()) if isinstance(row, Mapping)
    ]
    template_parameters: dict[str, object] = {}
    if base_cells and isinstance(base_cells[0].get("parameters"), Mapping):
        # The base plan is a route template, not a source of science
        # semantics. Keep only the canonical provider parameter surface so
        # campaign-specific controls from an authority artifact cannot leak
        # into a newly materialized WorldBundle.
        template_parameters = {
            key: value
            for key, value in base_cells[0]["parameters"].items()
            if key in SCIENCE_CUDA_V2_CANONICAL_PARAMETER_IDS_V1
        }
    execution = base_plan.get("execution")
    if not isinstance(execution, Mapping):
        raise ScienceWorldBundleError("base campaign plan execution is absent")
    acs_checkpoint_slots = execution.get("acs_checkpoint_slots", 0)
    if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {
        0,
        1,
        2,
    }:
        raise ScienceWorldBundleError(
            "base campaign plan acs_checkpoint_slots must be 0, 1, or 2"
        )
    native_execution_mode = execution_mode or str(
        template_parameters.get("cuda_native.execution_mode", "eager_monolith")
    )
    native_event_transport = event_transport or str(
        template_parameters.get("cuda_native.event_transport", "direct_sparse_v3")
    )
    if observation_mode not in {
        SCIENCE_CUDA_V2_OBSERVATION_OFF_V1,
        SCIENCE_CUDA_V2_OBSERVATION_TYPED_SIDECAR_V1,
        SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1,
    }:
        raise ScienceWorldBundleError(
            "unsupported CUDA V2 science observation mode"
        )
    if (
        observation_mode
        in {
            SCIENCE_CUDA_V2_OBSERVATION_TYPED_SIDECAR_V1,
            SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1,
        }
        and expected_provider != MICRO_PROVIDER_ID
    ):
        capture_label = (
            "typed"
            if observation_mode == SCIENCE_CUDA_V2_OBSERVATION_TYPED_SIDECAR_V1
            else "compact"
        )
        raise ScienceWorldBundleError(
            f"{capture_label} CUDA V2 science capture is not implemented for the selected "
            f"carrier provider {expected_provider!r}; use observation OFF or a "
            "provider with the canonical observation ABI"
        )
    world_by_id = {world.world_id: world for world in bundle.worlds}
    selected_worlds = [
        world
        for _, mapping in batch.world_slots
        if (world := world_by_id.get(mapping.world_id)) is not None
        and _selected_world_initialization_v1(world) is not None
    ]
    if selected_worlds:
        if len(selected_worlds) != len(batch.world_slots):
            raise ScienceWorldBundleError(
                "selected-world and founder initialization cannot share a physical batch"
            )
        return _lower_selected_workhorse_batch_v1(
            base_plan,
            bundle,
            admission,
            carrier_plan,
            physical_batch_index=physical_batch_index,
            observation_mode=observation_mode,
            science_use_class=science_use_class,
        )
    native_control = None
    if batch.native_control_vector is not None:
        native_control = {
            "schema_version": "eiel_science_cuda_v2_native_control_v1",
            "capability_mask": batch.native_capability_mask,
            "control_vector": dict(batch.native_control_vector),
            "control_digest": batch.native_control_digest.to_dict(),
        }
    first_time: Mapping[str, object] | None = None
    checkpoint_union: set[int] = set()
    observation_tick_union: set[int] = set()
    native_world_controls = []
    cells: list[dict[str, object]] = []
    for ordinal, mapping in batch.world_slots:
        world = world_by_id.get(mapping.world_id)
        if world is None:
            raise ScienceWorldBundleError(
                f"carrier batch names unknown world {mapping.world_id}"
            )
        semantic = world.semantic_inputs.to_value()
        seeds = semantic.get("seed_bindings")
        time = semantic.get("time")
        if not isinstance(seeds, dict) or not isinstance(time, dict):
            raise ScienceWorldBundleError(
                f"{world.world_id}: resolved seeds/time are incomplete"
            )
        ecology = seeds.get("ecology")
        if type(ecology) is not int or ecology < 0:
            raise ScienceWorldBundleError(
                f"{world.world_id}: ecology seed must be a nonnegative integer"
            )
        if first_time is None:
            first_time = time
        elif (
            first_time.get("authority") != time.get("authority")
            or first_time.get("start_next_tick", 0) != time.get("start_next_tick", 0)
            or first_time.get("total_ticks") != time.get("total_ticks")
        ):
            raise ScienceWorldBundleError(
                "carrier batch contains incompatible absolute-time contracts"
            )
        raw_checkpoints = time.get("checkpoint_ticks", [])
        if not isinstance(raw_checkpoints, (list, tuple)) or any(
            type(value) is not int for value in raw_checkpoints
        ):
            raise ScienceWorldBundleError(
                f"{world.world_id}: checkpoint_ticks must be an integer array"
            )
        checkpoint_union.update(raw_checkpoints)
        acquisition = world.acquisition.to_value()
        raw_snapshot_ticks = (
            acquisition.get("snapshot_ticks", [])
            if isinstance(acquisition, Mapping)
            else []
        )
        if not isinstance(raw_snapshot_ticks, (list, tuple)) or any(
            type(value) is not int for value in raw_snapshot_ticks
        ):
            raise ScienceWorldBundleError(
                f"{world.world_id}: snapshot_ticks must be an integer array"
            )
        observation_tick_union.update(raw_snapshot_ticks)
        if mapping.native_world_control is None:
            raise ScienceWorldBundleError(
                f"{world.world_id}: native world control is absent"
            )
        native_world_controls.append(
            project_science_cuda_v2_native_world_v1(
                {
                    "semantic_inputs": semantic,
                    "science": world.science.to_value(),
                }
            )
        )
        parameters = dict(template_parameters)
        parameters.update(
            {
                "cuda_native.temporal_k": batch.temporal_k,
                "cuda_native.execution_mode": native_execution_mode,
                "cuda_native.event_transport": native_event_transport,
                "cuda_native.science_observation_mode": observation_mode,
            }
        )
        cells.append(
            {
                "ordinal": ordinal,
                "cell_id": mapping.world_id,
                "branch_id": f"science-branch-{mapping.world_id}",
                "rng_lineage_id": f"science-rng-{mapping.world_id}",
                "dimensions": {
                    "science_world_id": mapping.world_id,
                    "test_id": world.science.to_value()["test_id"],
                    "study_id": world.science.to_value()["study_id"],
                },
                "parameters": parameters,
                # The native V2 provider's task seed is a numeric world
                # subject.  The adapter's mapping is derived from the full
                # semantic digest and is retained separately in the bundle.
                "seeds": {
                    "ecology": ecology,
                    "task": (
                        mapping.native_rng_world_uid
                        if mapping.native_rng_world_uid is not None
                        else mapping.world_uid
                    ),
                },
                "science_seed_bindings": seeds,
                "science_world_id": mapping.world_id,
                # The semantic/logical observation subject is distinct from
                # the paired native RNG subject in ``seeds.task``.  Preserve
                # it explicitly so OFF/ON or arm-paired worlds do not merge
                # their canonical records merely because they intentionally
                # share native RNG addressing.
                "science_logical_world_id": mapping.world_uid,
                "native_rng_world_uid": mapping.native_rng_world_uid,
                "native_rng_identity_sha256": (
                    None
                    if mapping.native_rng_identity_sha256 is None
                    else mapping.native_rng_identity_sha256.to_dict()
                ),
                "native_control": (
                    None
                    if mapping.native_control_vector is None
                    else {
                        "schema_version": "eiel_science_cuda_v2_native_control_v1",
                        "capability_mask": mapping.native_capability_mask,
                        "control_vector": dict(mapping.native_control_vector),
                        "control_digest": mapping.native_control_digest.to_dict(),
                    }
                ),
                "native_world_control": dict(mapping.native_world_control),
                "science_world_sha256": mapping.semantic_world_sha256.to_dict(),
                "science_acquisition_sha256": mapping.acquisition_sha256.to_dict(),
                "science_requested_observations": list(mapping.requested_observations),
                "science_ownership": world.science.to_value(),
                "science_semantic_inputs": semantic,
                "science_acquisition": world.acquisition.to_value(),
                "science_reducers": world.reducers.to_value(),
                # These bindings are acquisition metadata.  They deliberately
                # remain separate from the physical carrier identity and are
                # consumed only by the host-side canonical observation adapter.
                "science_instrument_manifest_sha256": (
                    instrument_manifest_sha256.to_dict()
                    if instrument_manifest_sha256 is not None
                    else canonical_sha256(
                        {
                            "instrument_contract": "EIEL_SCIENCE_INSTRUMENT_V1",
                            "carrier_profile_id": profile.carrier_profile_id,
                        }
                    ).to_dict()
                ),
                "science_crca_semantic_run_digest": (
                    crca_semantic_run_digest.to_dict()
                    if crca_semantic_run_digest is not None
                    else canonical_sha256(
                        {
                            "bundle_content_sha256": bundle_content_sha256.to_dict(),
                            "world_id": world.world_id,
                            "semantic_world_sha256": mapping.semantic_world_sha256.to_dict(),
                        }
                    ).to_dict()
                ),
                "science_semantic_world_key": {
                    "world_id": world.world_id,
                    "semantic_world_sha256": mapping.semantic_world_sha256.to_dict(),
                },
            }
        )
    if first_time is None:
        raise ScienceWorldBundleError("carrier batch has no worlds")
    total_ticks = first_time.get("total_ticks")
    start_next_tick = first_time.get("start_next_tick", 0)
    if type(total_ticks) is not int or total_ticks <= 0:
        raise ScienceWorldBundleError("world time requires positive total_ticks")
    if type(start_next_tick) is not int or start_next_tick < 0:
        raise ScienceWorldBundleError("world time start_next_tick is invalid")
    target = start_next_tick + total_ticks
    checkpoints = tuple(sorted(checkpoint_union))
    if not checkpoints or checkpoints[-1] != target:
        checkpoints = (*checkpoints, target)
    if any(value <= start_next_tick or value > target for value in checkpoints):
        raise ScienceWorldBundleError("world checkpoint ticks are outside the horizon")
    bound = dict(base_plan)
    bound["campaign_id"] = (
        f"eiel-science-{bundle.bundle_id.lower()}-batch-{physical_batch_index:04d}"
    )
    if science_use_class == SCIENCE_CUDA_V2_USE_CLASS_EXPLORATORY_SCIENCE_V1:
        bound["description"] = (
            "Bounded exploratory science batch through the "
            f"{expected_provider} canonical V2 provider; execution remains "
            "engineering-only and non-authoritative."
        )
        bound["no_claims"] = (
            "EXPLORATORY_SCIENCE. Exact-route bounded pre-qualification "
            "exploration with non-authoritative results and engineering-only "
            "execution; no production, PREQUAL, or Qualification authority."
        )
    elif science_use_class == SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1:
        bound["description"] = (
            "Engineering-only lowered explicit science-world batch through the "
            f"{expected_provider} canonical V2 provider."
        )
        bound["no_claims"] = (
            "TECHNICAL_ONLY. Engineering demonstration only; no PREQUAL, "
            "Qualification, production, or scientific authority."
        )
    else:
        bound["description"] = (
            "Real scientific execution of explicit EIEL Science Instrument worlds "
            f"through the {expected_provider} native CUDA V2 provider."
        )
        bound["no_claims"] = (
            "SCIENTIFIC_EXECUTION. Scientific results are permitted subject to "
            "the frozen study and evidence contracts; no PREQUAL, QualPrep, "
            "Qualification, or production-qualification authority is claimed."
        )
    bound["resolved_profile"] = profile.native_profile_id
    bound["cells"] = cells
    bound["physical_batches"] = [
        {
            "ordinal": 0,
            "batch_id": f"science-batch-{physical_batch_index:04d}",
            "cell_ids": [mapping.world_id for _, mapping in batch.world_slots],
            "native_world_control_set_sha256": {
                "algorithm": "sha256",
                "hex": native_world_control_set_sha256_v1(native_world_controls),
            },
        }
    ]
    raw_tick_control = execution.get("tick_native_control", {})
    if not isinstance(raw_tick_control, Mapping):
        raise ScienceWorldBundleError("base campaign plan tick_native_control is invalid")
    tick_control = {
        **dict(raw_tick_control),
        "total_ticks": total_ticks,
        "start_next_tick": start_next_tick,
        "checkpoint_ticks": list(checkpoints),
        "resident_owner_span_ticks": batch.temporal_k,
        "native_dispatch_window_ticks": batch.temporal_k,
        "source_window_ticks": batch.temporal_k,
        "event_window_capacity": int(
            raw_tick_control.get(
                "event_window_capacity", 1024 * batch.temporal_k
            )
        ),
    }
    schedules = tick_control.get("schedules")
    if isinstance(schedules, Mapping):
        observation_schedule = sorted(
            tick
            for tick in observation_tick_union
            if start_next_tick < tick <= target
        )
        tick_control["schedules"] = {
            **dict(schedules),
            "checkpoint_ticks": list(checkpoints),
            "report_ticks": observation_schedule,
            "science_sample_ticks": observation_schedule,
            "reducer_ticks": observation_schedule,
            "archive_ticks": list(checkpoints),
        }
    bound_execution = {
        **dict(execution),
        "execution_route_identity": "tick_native_execution_route_v1",
        "total_ticks": total_ticks,
        "start_next_tick": start_next_tick,
        "checkpoint_ticks": list(checkpoints),
        "deliberate_restore_ticks": list(
            first_time.get("deliberate_restore_ticks", [])
        ),
        "resident_owner_span_ticks": batch.temporal_k,
        "native_dispatch_window_ticks": batch.temporal_k,
        "source_window_ticks": batch.temporal_k,
        "checkpoint_storage_mode": "disk",
        # ACS is a producer-owned persistence path.  Preserve the already
        # compiled base-plan selection so science observation/reduction cannot
        # silently disable ACS2 or become part of its progress boundary.
        "acs_checkpoint_slots": acs_checkpoint_slots,
        "native_streamed_checkpoint": True,
        "tick_native_control": tick_control,
    }
    technical_gate = bound_execution.get("technical_gate")
    if isinstance(technical_gate, Mapping):
        bound_execution["technical_gate"] = {
            **dict(technical_gate),
            "total_ticks": total_ticks,
            "checkpoint_ticks": list(checkpoints),
        }
    bound["execution"] = bound_execution
    bound["execution"]["native_world_control_set_sha256"] = {
        "algorithm": "sha256",
        "hex": native_world_control_set_sha256_v1(native_world_controls),
    }
    if native_control is not None:
        bound["physical_batches"][0]["native_control"] = native_control
        bound["execution"]["native_control"] = native_control
    lifecycle = base_plan.get("artifact_lifecycle")
    storage = lifecycle.get("storage") if isinstance(lifecycle, Mapping) else None
    total_max_bytes = (
        storage.get("total_max_bytes") if isinstance(storage, Mapping) else None
    )
    if lifecycle is not None:
        if type(total_max_bytes) is not int or total_max_bytes < 4:
            raise ScienceWorldBundleError(
                "base campaign plan lifecycle lacks a bounded storage envelope"
            )
        rebound_lifecycle = lifecycle_profile_contract_v1(
            SCIENCE_COMPACT,
            total_max_bytes=total_max_bytes,
            checkpoint_schedule=checkpoints,
            checkpoint_authority="absolute_tick",
        )
        rebound_lifecycle["storage"] = {
            **dict(rebound_lifecycle["storage"]),
            **dict(storage),
        }
        rebound_lifecycle.pop("contract_sha256", None)
        bound["artifact_lifecycle"] = validate_run_artifact_lifecycle_contract_v1(
            rebound_lifecycle
        )
    bound["science_materialization"] = {
        "schema_version": "eiel_science_v2_lowered_explicit_batch_v1",
        "bundle_content_sha256": bundle_content_sha256.to_dict(),
        "bundle_admission_sha256": canonical_sha256(admission.to_dict()).to_dict(),
        "physical_execution_plan_sha256": carrier_plan_sha256.to_dict(),
        "physical_batch_index": physical_batch_index,
        "carrier_profile_id": profile.carrier_profile_id,
        "physical_batch_size": batch.physical_batch_size,
        "semantic_world_ids": [mapping.world_id for _, mapping in batch.world_slots],
        "observation_mode": observation_mode,
        "engineering_capture": (
            observation_mode
            in {
                SCIENCE_CUDA_V2_OBSERVATION_TYPED_SIDECAR_V1,
                SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1,
            }
        ),
        "observation_abi": (
            "eiel_science_observation_record_v1"
            if observation_mode == SCIENCE_CUDA_V2_OBSERVATION_TYPED_SIDECAR_V1
            else (
                "eiel_cuda_v2_compact_science_capture_v1"
                if observation_mode
                == SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1
                else None
            )
        ),
        "fallback_policy": "FORBIDDEN",
        "science_use_class": science_use_class,
        "native_control": native_control,
        "native_world_control_set_sha256": {
            "algorithm": "sha256",
            "hex": native_world_control_set_sha256_v1(native_world_controls),
        },
        "ordered_native_world_control_digests": [
            value.control_digest for value in native_world_controls
        ],
    }
    bound["science_use_class"] = science_use_class
    _bind_executable_materialized_resource_envelope_v1(bound)
    bound.pop("plan_sha256", None)
    bound["plan_sha256"] = campaign_sha256(bound)
    return bound


@dataclass(frozen=True, slots=True)
class CudaV2ScienceCarrierFamilyAdapterV1:
    """The one host adapter family over the canonical CUDA V2 provider boundary.

    Campaign/study expansion is completed before this adapter is called.  The
    adapter only plans and lowers already explicit, admitted worlds; each
    lowered result remains an ordinary campaign plan for the maintained
    preflight/execute front door.
    """

    adapter_id: str = SCIENCE_CUDA_V2_ADAPTER_ID

    @property
    def profiles(self) -> tuple[ScienceCudaV2CarrierProfileV1, ...]:
        return SCIENCE_CUDA_V2_PROFILES

    def plan(
        self,
        bundle: ScienceWorldBundleV1,
        admission: ScienceWorldBundleAdmissionV1,
        **options,
    ) -> ScienceCudaV2ExecutionPlanV1:
        return plan_science_cuda_v2_execution_v1(bundle, admission, **options)

    def lower_batch(
        self,
        base_plan: Mapping[str, object],
        bundle: ScienceWorldBundleV1,
        admission: ScienceWorldBundleAdmissionV1,
        carrier_plan: ScienceCudaV2ExecutionPlanV1,
        *,
        physical_batch_index: int,
        execution_mode: str | None = None,
        event_transport: str | None = None,
        observation_mode: str = SCIENCE_CUDA_V2_OBSERVATION_OFF_V1,
        science_use_class: str = SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1,
        instrument_manifest_sha256: ContentDigest | None = None,
        crca_semantic_run_digest: ContentDigest | None = None,
        _precomputed_bundle_content_sha256: ContentDigest | None = None,
        _precomputed_carrier_plan_sha256: ContentDigest | None = None,
    ) -> dict[str, object]:
        """Lower one physical carrier batch to an ordinary provider plan."""

        return lower_science_cuda_v2_batch_to_campaign_plan_v1(
            base_plan,
            bundle,
            admission,
            carrier_plan,
            physical_batch_index=physical_batch_index,
            execution_mode=execution_mode,
            event_transport=event_transport,
            observation_mode=observation_mode,
            science_use_class=science_use_class,
            instrument_manifest_sha256=instrument_manifest_sha256,
            crca_semantic_run_digest=crca_semantic_run_digest,
            _precomputed_bundle_content_sha256=_precomputed_bundle_content_sha256,
            _precomputed_carrier_plan_sha256=_precomputed_carrier_plan_sha256,
        )

    def lower_all(
        self,
        base_plan: Mapping[str, object],
        bundle: ScienceWorldBundleV1,
        admission: ScienceWorldBundleAdmissionV1,
        carrier_plan: ScienceCudaV2ExecutionPlanV1,
        *,
        execution_mode: str | None = None,
        event_transport: str | None = None,
        observation_mode: str = SCIENCE_CUDA_V2_OBSERVATION_OFF_V1,
        science_use_class: str = SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1,
        instrument_manifest_sha256: ContentDigest | None = None,
        crca_semantic_run_digest: ContentDigest | None = None,
    ) -> tuple[dict[str, object], ...]:
        """Lower every planned batch in stable physical order."""

        # The bundle is immutable for this materialization.  Authenticate its
        # content once and reuse that digest across every physical lowering;
        # recomputing the full 10k-world canonical payload per batch adds no
        # provenance assurance and dominates CPU materialization time.
        bundle_content_sha256 = bundle.bundle_content_sha256()
        carrier_plan_sha256 = carrier_plan.plan_sha256()
        return tuple(
            self.lower_batch(
                base_plan,
                bundle,
                admission,
                carrier_plan,
                physical_batch_index=index,
                execution_mode=execution_mode,
                event_transport=event_transport,
                observation_mode=observation_mode,
                science_use_class=science_use_class,
                instrument_manifest_sha256=instrument_manifest_sha256,
                crca_semantic_run_digest=crca_semantic_run_digest,
                _precomputed_bundle_content_sha256=bundle_content_sha256,
                _precomputed_carrier_plan_sha256=carrier_plan_sha256,
            )
            for index in range(len(carrier_plan.batches))
        )


__all__ = (
    "SCIENCE_CUDA_V2_ADAPTER_ID",
    "SCIENCE_CUDA_V2_MICRO_BACKEND_ID",
    "SCIENCE_CUDA_V2_PROFILES",
    "SCIENCE_CUDA_V2_PROFILES_BY_ID",
    "SCIENCE_CUDA_V2_TARGET_BATCH",
    "SCIENCE_CUDA_V2_OBSERVATION_OFF_V1",
    "SCIENCE_CUDA_V2_CANONICAL_PARAMETER_IDS_V1",
    "SCIENCE_CUDA_V2_OBSERVATION_TYPED_SIDECAR_V1",
    "SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1",
    "SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1",
    "SCIENCE_CUDA_V2_USE_CLASS_EXPLORATORY_SCIENCE_V1",
    "SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1",
    "SCIENCE_CUDA_V2_USE_CLASSES_V1",
    "CudaV2ScienceCarrierFamilyAdapterV1",
    "lower_science_cuda_v2_batch_to_campaign_plan_v1",
    "plan_science_cuda_v2_execution_v1",
    "science_cuda_v2_carrier_profiles_for_world_v1",
    "science_cuda_v2_profile_catalog_sha256_v1",
)
