from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import jsonschema
import pytest

from eiel_composition import science_cuda_v2_carrier_v1 as carrier_module
from emergence_contracts import (
    PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION,
    CanonicalPayloadV1,
    SchemaVersion,
    ScienceAdmissionOutcomeV1,
    canonical_sha256,
)
from engine.campaigns.artifact_lifecycle import (
    SCIENCE_COMPACT,
    lifecycle_profile_contract_v1,
)
from eiel_composition.science_cuda_v2_carrier_v1 import (
    SCIENCE_CUDA_V2_ADAPTER_ID,
    SCIENCE_CUDA_V2_PROFILES,
    CudaV2ScienceCarrierFamilyAdapterV1,
    lower_science_cuda_v2_batch_to_campaign_plan_v1,
)
from eiel_composition.science_instrument import (
    ScienceCRCASemanticBindingV1,
    compile_science_campaign_v1,
)
from eiel_composition.science_instrument_demo_v1 import (
    _control_descriptors,
    _instrument_manifest,
    _profile_descriptor,
    _reducer_descriptor,
    _request,
    _resource_envelope,
    _route_and_empty_envelope,
)
from eiel_composition.science_request_v2 import (
    normalize_rich_production_science_request_v2,
)
from eiel_composition.science_world_bundle_v1 import (
    ScienceWorldBundleError,
    admit_science_world_bundle_v1,
    materialize_science_world_bundle_v1,
)


PROFILE_CASES = (
    ("W32", 32, 32, 256, 512, "w32x32p256c512_gpuv2"),
    ("W64", 64, 64, 1024, 2048, "w64x64p1024c2048_gpuv2"),
    ("W128X80", 128, 80, 2560, 5120, "w128x80p2560c5120_gpuv2"),
)


def _payload(schema: str, value: object) -> CanonicalPayloadV1:
    return CanonicalPayloadV1.from_value(SchemaVersion(schema), value)


def _compiled_profile(
    family: str,
    width: int,
    height: int,
    population: int,
    capacity: int,
    *,
    seed_count: int = 3,
    selection_pressure: float = 0.25,
    founder_profile_ids: tuple[str, str] = (
        "portable-demo-founder-a-v1",
        "portable-demo-founder-b-v1",
    ),
):
    controls = _control_descriptors()
    reducer = _reducer_descriptor()
    profile = _profile_descriptor()
    manifest = _instrument_manifest(controls, reducer, profile)
    route, _ = _route_and_empty_envelope(
        manifest, _resource_envelope(), route_id="cuda-v2-carrier-host-test"
    )
    request = deepcopy(_request(manifest, reducer))
    campaign = request["science_campaign"]
    campaign["campaign_id"] = f"cuda-v2-{family.lower()}-{seed_count}"
    ownership = campaign["question"]["science_ownership"]
    ownership["test_id"] = family
    ownership["study_id"] = f"{family}-CUDA-V2-CARRIER-HOST"
    world = campaign["world"]
    world["world_profile_id"] = f"{family.lower()}-cuda-v2"
    world["geometry"] = {"width": width, "height": height}
    world["resident_capacity"] = capacity
    initialization = campaign["initialization"]
    initialization["initial_population"] = population
    initialization["initial_live_count"] = population
    initialization["founder_profiles"] = list(founder_profile_ids)
    initialization["exact_founder_composition"] = [
        {"founder_profile_id": founder_profile_ids[0], "count": population // 2},
        {
            "founder_profile_id": founder_profile_ids[1],
            "count": population - population // 2,
        },
    ]
    if seed_count != 3:
        ownership.pop("arm_dimension_ids", None)
        ownership["fixed_arm_id"] = "fixed_controls"
        campaign["design"] = {
            "dimensions": [
                {
                    "dimension_id": "task_seed",
                    "binding_path": "seeds.task_environment",
                    "canonical_values": list(range(10_000, 10_000 + seed_count)),
                    "ordering": "declared",
                }
            ],
            "pairing_groups": ["task_seed"],
            "repeat_groups": [],
            "expected_cell_count": seed_count,
        }
        campaign["controls"][0].pop("dimension_binding", None)
        campaign["controls"][0]["value"] = selection_pressure
        campaign["seeds"] = [
            {"owner": "ecology", "binding_kind": "literal", "literal": 7001},
            {"owner": "mutation", "binding_kind": "literal", "literal": 8001},
            {
                "owner": "task_environment",
                "binding_kind": "dimension_bound",
                "dimension_id": "task_seed",
            },
        ]
        campaign["execution"]["logical_world_count"] = seed_count
        campaign["execution"]["maximum_physical_batch_worlds"] = seed_count
    else:
        campaign["design"]["expected_cell_count"] = 12
        campaign["execution"]["logical_world_count"] = 12
        campaign["execution"]["maximum_physical_batch_worlds"] = 12
    normalized = normalize_rich_production_science_request_v2(request).campaign
    crca = ScienceCRCASemanticBindingV1(
        semantic_run_digest=canonical_sha256({"run": family, "seeds": seed_count}),
        projection_sha256=canonical_sha256({"projection": family}),
        absolute_start_next_tick=0,
        absolute_target_next_tick=20,
        trajectory_checkpoint_ticks=(0, 10, 20),
    )
    compilation = compile_science_campaign_v1(
        normalized,
        instrument_manifest=manifest,
        route_binding=route,
        control_descriptors=controls,
        reducer_descriptors=(reducer,),
        crca_semantic_binding=crca,
        physical_execution_core=_payload(
            str(PHYSICAL_EXECUTION_CORE_SCHEMA_VERSION),
            {"route_id": route.route_id, "logical_world_partitions": [list(range(seed_count if seed_count != 3 else 12))]},
        ),
        compiler_component_sha256=canonical_sha256(
            {"component": "compile_science_campaign_v1", "version": 1}
        ),
    )
    return normalized, compilation


def _admit(bundle):
    return admit_science_world_bundle_v1(
        bundle,
        supported_observations=(
            "birth",
            "death",
            "genome_identity",
            "lineage_ancestry",
            "birth_count",
            "population_count",
            "causal_lifecycle",
        ),
        supported_profiles=("SCIENCE_STANDARD_V1",),
        maximum_total_ticks=5000,
    )


def test_one_parameterized_family_has_three_b_free_profiles() -> None:
    adapter = CudaV2ScienceCarrierFamilyAdapterV1()
    assert adapter.adapter_id == SCIENCE_CUDA_V2_ADAPTER_ID
    assert adapter.profiles == SCIENCE_CUDA_V2_PROFILES
    assert [profile.carrier_profile_id for profile in adapter.profiles] == [
        case[-1] for case in PROFILE_CASES
    ]
    assert all("b1024" not in profile.carrier_profile_id for profile in adapter.profiles)
    assert all(profile.target_physical_batch == 1024 for profile in adapter.profiles)
    assert {profile.provider_family for profile in adapter.profiles} == {
        "cuda_full_evolution_v2",
        "cuda_full_evolution_v2_workhorse",
    }


def test_mixed_profiles_partition_without_changing_science_identity() -> None:
    compiled = [
        _compiled_profile(family, width, height, population, capacity)
        for family, width, height, population, capacity, _ in PROFILE_CASES
    ]
    bundle = materialize_science_world_bundle_v1(
        compiled, bundle_id="CUDA-V2-MIXED-PROFILE-BUNDLE"
    )
    admission = _admit(bundle)
    assert admission.outcome is ScienceAdmissionOutcomeV1.ADMITTED
    adapter = CudaV2ScienceCarrierFamilyAdapterV1()
    first = adapter.plan(
        bundle,
        admission,
        physical_batch_capacity={case[-1]: 7 for case in PROFILE_CASES},
        temporal_k=1,
        execution_quantum_id="ORG-QUANTUM-1024",
    )
    alternate = adapter.plan(
        bundle,
        admission,
        physical_batch_capacity={case[-1]: 12 for case in PROFILE_CASES},
        temporal_k=2,
        world_order=reversed(admission.admitted_world_ids),
    )
    assert first.logical_bundle_size == bundle.world_count == 36
    assert len(first.batches) == 6
    assert len(alternate.batches) == 3
    assert first.plan_sha256() != alternate.plan_sha256()
    assert first.execution_quantum_id == "ORG-QUANTUM-1024"
    assert first.fallback_policy == "FORBIDDEN"

    def mappings(plan):
        return {
            world.world_id: (
                world.world_uid,
                world.semantic_world_sha256,
                world.acquisition_sha256,
                world.requested_observations,
            )
            for batch in plan.batches
            for _, world in batch.world_slots
        }

    assert mappings(first) == mappings(alternate)
    bundle_identity = {
        world.world_id: (world.semantic_world_sha256, world.acquisition_sha256)
        for world in bundle.worlds
    }
    assert {
        world_id: (row[1], row[2]) for world_id, row in mappings(first).items()
    } == bundle_identity
    assert {
        batch.carrier_profile.carrier_profile_id for batch in first.batches
    } == {case[-1] for case in PROFILE_CASES}
    schema = json.loads(
        Path(
            "_spec/science_instrument/v1/"
            "science_cuda_v2_execution_plan.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(first.to_dict())
    for batch in first.batches:
        assert batch.physical_batch_size <= 7
        assert batch.native_observation_union
        assert all(
            set(world.requested_observations).issubset(batch.native_observation_union)
            for _, world in batch.world_slots
        )


def test_exact_1024_world_bundle_is_one_physical_w32_batch() -> None:
    campaign = _compiled_profile("MUTSUP", 32, 32, 256, 512, seed_count=1024)
    bundle = materialize_science_world_bundle_v1(
        (campaign,), bundle_id="CUDA-V2-B1024-EXPLICIT-WORLDS"
    )
    admission = _admit(bundle)
    plan = CudaV2ScienceCarrierFamilyAdapterV1().plan(bundle, admission)
    assert bundle.world_count == 1024
    assert len(plan.batches) == 1
    assert plan.batches[0].physical_batch_size == 1024
    assert plan.batches[0].carrier_profile.carrier_profile_id == (
        "w32x32p256c512_gpuv2"
    )
    assert len({world.world_uid for _, world in plan.batches[0].world_slots}) == 1024


def test_lowering_materializes_explicit_worlds_without_physical_identity_leak() -> None:
    normalized, compilation = _compiled_profile("MUTSUP", 32, 32, 256, 512, seed_count=3)
    bundle = materialize_science_world_bundle_v1(
        ((normalized, compilation),), bundle_id="CUDA-V2-LOWERING"
    )
    admission = _admit(bundle)
    carrier_plan = CudaV2ScienceCarrierFamilyAdapterV1().plan(
        bundle, admission, physical_batch_capacity=3, temporal_k=1
    )
    base_plan = {
        "resolved_profile": "eiel_full_evolution_cuda_native_v2",
        "cells": [
            {
                "parameters": {
                    "cuda_native.execution_mode": "eager_monolith",
                    "cuda_native.event_transport": "direct_sparse_v3",
                    "selection.pressure": 999,
                }
            }
        ],
        "execution": {
            "total_ticks": 20,
            "start_next_tick": 0,
            "checkpoint_ticks": [10, 20],
            "tick_native_control": {"event_window_capacity": 4096},
        },
        "artifacts": {},
        "resource_estimate": {},
    }
    lowered = lower_science_cuda_v2_batch_to_campaign_plan_v1(
        base_plan, bundle, admission, carrier_plan, physical_batch_index=0
    )
    assert lowered["resolved_profile"] == "eiel_full_evolution_cuda_native_v2"
    assert lowered["execution"]["checkpoint_ticks"] == [10, 20]
    assert lowered["execution"]["execution_route_identity"] == "tick_native_execution_route_v1"
    assert len(lowered["cells"]) == 3
    assert lowered["physical_batches"][0]["ordinal"] == 0
    assert lowered["physical_batches"][0]["batch_id"] == "science-batch-0000"
    assert lowered["physical_batches"][0]["cell_ids"] == [
        cell["cell_id"] for cell in lowered["cells"]
    ]
    assert lowered["physical_batches"][0][
        "native_world_control_set_sha256"
    ] == lowered["execution"]["native_world_control_set_sha256"]
    assert all("native_world_control" in cell for cell in lowered["cells"])
    assert all(cell["seeds"]["task"] > 0 for cell in lowered["cells"])
    assert all("physical_slot" not in cell for cell in lowered["cells"])
    assert all("selection.pressure" not in cell["parameters"] for cell in lowered["cells"])

    acs2_base_plan = deepcopy(base_plan)
    acs2_base_plan["execution"]["acs_checkpoint_slots"] = 2
    acs2_lowered = lower_science_cuda_v2_batch_to_campaign_plan_v1(
        acs2_base_plan,
        bundle,
        admission,
        carrier_plan,
        physical_batch_index=0,
    )
    assert lowered["execution"]["acs_checkpoint_slots"] == 0
    assert acs2_lowered["execution"]["acs_checkpoint_slots"] == 2
    assert acs2_lowered["plan_sha256"] != lowered["plan_sha256"]

def test_executable_lowering_rebinds_post_acquisition_resource_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized, compilation = _compiled_profile(
        "GENERIC-RESOURCE", 32, 32, 256, 512, seed_count=3
    )
    bundle = materialize_science_world_bundle_v1(
        ((normalized, compilation),), bundle_id="CUDA-V2-RESOURCE-REBIND"
    )
    admission = _admit(bundle)
    carrier_plan = CudaV2ScienceCarrierFamilyAdapterV1().plan(
        bundle, admission, physical_batch_capacity=3, temporal_k=1
    )
    lifecycle = lifecycle_profile_contract_v1(
        SCIENCE_COMPACT,
        total_max_bytes=1_000_000_000,
        checkpoint_schedule=(10, 20),
        checkpoint_authority="absolute_tick",
    )
    base_plan = {
        "resolved_profile": "eiel_full_evolution_cuda_native_v2",
        "cells": [
            {
                "parameters": {
                    "cuda_native.execution_mode": "eager_monolith",
                    "cuda_native.event_transport": "direct_sparse_v3",
                }
            }
        ],
        "execution": {
            "total_ticks": 20,
            "start_next_tick": 0,
            "checkpoint_ticks": [10, 20],
            "tick_native_control": {"event_window_capacity": 4096},
            "budgets": {
                "vram_bytes": 500,
                "disk_bytes": 1_000_000_000,
                "wall_seconds": 100,
            },
        },
        "artifact_lifecycle": lifecycle,
        "artifacts": {
            "cuda_extension": {},
            "extension_manifest": {},
            "device_selection_receipt": {},
        },
        "resource_estimate": {"disk_bytes_upper_bound": 999_000_000},
    }
    monkeypatch.setattr(
        carrier_module,
        "cuda_full_evolution_v2_materialized_resource_estimate_v1",
        lambda _plan: {
            "resource_estimator_id": "generic-post-lowering-test-v1",
            "physical_batch_cells": 3,
            "vram_bytes_upper_bound": 1_000,
            "wall_seconds_upper_bound": 300,
            "disk_bytes_upper_bound": 10_000,
            "event_bytes_upper_bound": 2_000,
            "checkpoint_bytes_upper_bound": 3_000,
            "fixed_output_bytes_upper_bound": 1_000,
            "storage_reservation_mode": "bounded_incremental_guarded_v1",
            "storage_reservation_guard_scope": (
                "provider_dispatch_and_checkpoint_boundary_v1"
            ),
            "storage_reservation_window_bytes_upper_bound": 4_000,
            "storage_reservation_control_bytes_upper_bound": 1_000,
            "storage_reservation_dispatch_increment_bytes_upper_bound": 1_000,
            "storage_reservation_boundary_increment_bytes_upper_bound": 2_000,
            "storage_reservation_emergency_checkpoint_bytes_upper_bound": 1_500,
            "storage_reservation_total_envelope_bytes_upper_bound": 10_000,
        },
    )

    lowered = lower_science_cuda_v2_batch_to_campaign_plan_v1(
        base_plan,
        bundle,
        admission,
        carrier_plan,
        physical_batch_index=0,
        observation_mode="COMPACT_SIDECAR_V1",
    )

    assert lowered["resource_estimate"]["disk_bytes_upper_bound"] == 10_000
    assert lowered["resource_estimate"]["authority"] == (
        "provider_bound_post_science_lowering_v1"
    )
    assert lowered["resource_estimate"]["storage_reservation_mode"] == (
        "bounded_incremental_guarded_v1"
    )
    assert (
        lowered["resource_estimate"][
            "storage_reservation_window_bytes_upper_bound"
        ]
        == 4_000
    )
    assert lowered["execution"]["budgets"]["vram_bytes"] == 1_000
    assert lowered["execution"]["budgets"]["disk_bytes"] == lowered[
        "artifact_lifecycle"
    ]["storage"]["total_max_bytes"]
    assert lowered["execution"]["budgets"]["disk_bytes"] > 10_000
    assert lowered["execution"]["budgets"]["wall_seconds"] == 300
    assert lowered["artifact_lifecycle"]["storage"]["total_max_bytes"] < (
        lifecycle["storage"]["total_max_bytes"]
    )
    binding = lowered["science_materialization"][
        "materialized_resource_binding"
    ]
    assert binding["physical_batch_cells"] == 3
    assert binding["disk_bytes_upper_bound"] == 10_000


def test_heterogeneous_b1024_world_controls_remain_one_native_batch() -> None:
    first = _compiled_profile(
        "MIXED-A",
        32,
        32,
        256,
        512,
        seed_count=512,
        selection_pressure=0.25,
        founder_profile_ids=("founder-neutral-v1", "founder-resource-v1"),
    )
    second = _compiled_profile(
        "MIXED-B",
        32,
        32,
        256,
        512,
        seed_count=512,
        selection_pressure=1.5,
        founder_profile_ids=("founder-movement-v1", "founder-risk-v1"),
    )
    bundle = materialize_science_world_bundle_v1(
        (first, second), bundle_id="CUDA-V2-GENERIC-MIXED-B1024"
    )
    admission = _admit(bundle)
    plan = CudaV2ScienceCarrierFamilyAdapterV1().plan(bundle, admission)

    assert bundle.world_count == 1024
    assert len(plan.batches) == 1
    batch = plan.batches[0]
    assert batch.physical_batch_size == 1024
    assert batch.native_control_vector is None
    controls = [
        world.native_world_control for _, world in batch.world_slots
    ]
    assert all(control is not None for control in controls)
    assert len({control["control_digest"]["hex"] for control in controls}) == 1024
    assert {
        control["initialization"]["founder_profiles"][0]
        for control in controls
    } == {"founder-neutral-v1", "founder-movement-v1"}


def test_compact_capture_lowering_is_declarative_and_not_k1_bound() -> None:
    campaign = _compiled_profile(
        "GENERIC-COMPACT",
        32,
        32,
        32,
        512,
        seed_count=2,
        founder_profile_ids=("founder-neutral-v1", "founder-resource-v1"),
    )
    bundle = materialize_science_world_bundle_v1(
        (campaign,), bundle_id="CUDA-V2-GENERIC-COMPACT"
    )
    admission = _admit(bundle)
    carrier_plan = CudaV2ScienceCarrierFamilyAdapterV1().plan(
        bundle, admission, physical_batch_capacity=2, temporal_k=8
    )
    base_plan = {
        "resolved_profile": "eiel_full_evolution_cuda_native_v2",
        "cells": [
            {
                "parameters": {
                    "cuda_native.execution_mode": "eager_monolith",
                    "cuda_native.event_transport": "direct_sparse_v3",
                }
            }
        ],
        "execution": {
            "total_ticks": 20,
            "start_next_tick": 0,
            "checkpoint_ticks": [10, 20],
            "tick_native_control": {"event_window_capacity": 4096},
        },
        "artifacts": {},
        "resource_estimate": {},
    }
    lowered = lower_science_cuda_v2_batch_to_campaign_plan_v1(
        base_plan,
        bundle,
        admission,
        carrier_plan,
        physical_batch_index=0,
        observation_mode="COMPACT_SIDECAR_V1",
    )
    assert lowered["science_materialization"]["observation_abi"] == (
        "eiel_cuda_v2_compact_science_capture_v1"
    )
    assert lowered["science_materialization"]["physical_batch_size"] == 2
    assert lowered["execution"]["resident_owner_span_ticks"] == 8
    assert {cell["science_ownership"]["test_id"] for cell in lowered["cells"]} == {
        "GENERIC-COMPACT"
    }


def test_unsupported_geometry_fails_closed_before_physical_plan() -> None:
    campaign = _compiled_profile("MUTSUP", 48, 48, 256, 512)
    bundle = materialize_science_world_bundle_v1(
        (campaign,), bundle_id="CUDA-V2-UNSUPPORTED-GEOMETRY"
    )
    with pytest.raises(ScienceWorldBundleError, match="no exact CUDA V2"):
        CudaV2ScienceCarrierFamilyAdapterV1().plan(bundle, _admit(bundle))


def test_typed_capture_fails_closed_for_workhorse_profiles_until_their_abi_exists() -> None:
    normalized, compilation = _compiled_profile("MUTSUP", 64, 64, 1024, 2048, seed_count=1)
    bundle = materialize_science_world_bundle_v1(
        ((normalized, compilation),), bundle_id="CUDA-V2-WORKHORSE-TYPED-CAPTURE"
    )
    admission = _admit(bundle)
    carrier_plan = CudaV2ScienceCarrierFamilyAdapterV1().plan(
        bundle, admission, physical_batch_capacity=1, temporal_k=1
    )
    base_plan = {
        "resolved_profile": carrier_plan.batches[0].carrier_profile.native_profile_id,
        "cells": [{"parameters": {"cuda_native.event_transport": "direct_sparse_v3"}}],
        "execution": {"tick_native_control": {"event_window_capacity": 4096}},
    }
    with pytest.raises(ScienceWorldBundleError, match="typed CUDA V2 science capture"):
        lower_science_cuda_v2_batch_to_campaign_plan_v1(
            base_plan,
            bundle,
            admission,
            carrier_plan,
            physical_batch_index=0,
            observation_mode="TYPED_SIDECAR_V1",
        )
