from __future__ import annotations

from types import SimpleNamespace

import pytest

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    ContractError,
    SchemaVersion,
    ScienceCudaV2ExecutionPlanV1,
    ScienceCudaV2PhysicalBatchV1,
    ScienceCudaV2WorldMappingV1,
    canonical_sha256,
)
from eiel_composition.science_cuda_v2_native_control_v1 import (
    SCIENCE_CUDA_V2_FULL_CAPABILITY_MASK,
    SCIENCE_CUDA_V2_MUTATION_OFF_CAPABILITY_MASK,
    ScienceCudaV2NativeControlError,
    ScienceCudaV2NativeControlV1,
    derive_science_cuda_v2_native_rng_identity_v1,
    project_science_cuda_v2_native_control_v1,
)
from eiel_composition.science_cuda_v2_carrier_v1 import SCIENCE_CUDA_V2_PROFILES


def _payload(schema: str, value: object) -> CanonicalPayloadV1:
    return CanonicalPayloadV1.from_value(SchemaVersion(schema), value)


def _world(
    *, arm_id: str, mutation_enabled: object, schedule: dict[str, object] | None = None
):
    world = {"geometry": {"width": 32, "height": 32}, "resident_capacity": 512}
    initialization = {"initial_population": 256, "founders": ["a", "b"]}
    seed_bindings = {"ecology": 7001, "task": 101}
    initial_state_identity = canonical_sha256(
        {
            "world": world,
            "initialization": initialization,
            "seed_bindings": seed_bindings,
        }
    ).to_dict()
    schedule = {"kind": "static"} if schedule is None else schedule
    descriptor = canonical_sha256({"control": "mutation.enabled"}).to_dict()
    semantic = {
        "world": world,
        "initialization": initialization,
        "dimension_bindings": {"arm": arm_id},
        "control_bindings": [
            {
                "control_id": "mutation.enabled",
                "resolved_value": mutation_enabled,
                "schedule": schedule,
                "descriptor_sha256": descriptor,
            }
        ],
        "seed_bindings": seed_bindings,
        "time": {"authority": "absolute_tick_v1", "total_ticks": 20},
        "initial_state_identity": initial_state_identity,
    }
    science = {
        "test_id": "MUTATION",
        "test_version": "1",
        "study_id": "PAIRING",
        "arm_id": arm_id,
        "replicate": 1,
    }
    return SimpleNamespace(
        semantic_inputs=_payload("eiel_science_resolved_inputs_v1", semantic),
        science=_payload("eiel_science_ownership_v1", science),
    )


def test_mutation_mask_and_receipt_are_exact() -> None:
    off = project_science_cuda_v2_native_control_v1(
        _world(arm_id="mutation=off", mutation_enabled=False)
    )
    on = project_science_cuda_v2_native_control_v1(
        _world(arm_id="mutation=on", mutation_enabled=True)
    )
    assert off.capability_mask == SCIENCE_CUDA_V2_MUTATION_OFF_CAPABILITY_MASK
    assert on.capability_mask == SCIENCE_CUDA_V2_FULL_CAPABILITY_MASK
    assert off.control_digest != on.control_digest
    assert ScienceCudaV2NativeControlV1.from_dict(off.to_dict()) == off
    tampered = {
        **off.to_dict(),
        "capability_mask": SCIENCE_CUDA_V2_FULL_CAPABILITY_MASK,
    }
    with pytest.raises(ScienceCudaV2NativeControlError, match="mask"):
        ScienceCudaV2NativeControlV1.from_dict(tampered)
    wrong_algorithm = off.to_dict()
    wrong_algorithm["control_digest"] = {
        "algorithm": "not-sha256",
        "hex": off.control_digest,
    }
    with pytest.raises(ScienceCudaV2NativeControlError, match="digest"):
        ScienceCudaV2NativeControlV1.from_dict(wrong_algorithm)


def test_native_rng_identity_excludes_arm_but_binds_study_and_replicate() -> None:
    off = _world(arm_id="mutation=off", mutation_enabled=False)
    on = _world(arm_id="mutation=on", mutation_enabled=True)
    assert derive_science_cuda_v2_native_rng_identity_v1(off) == (
        derive_science_cuda_v2_native_rng_identity_v1(on)
    )
    other = _world(arm_id="mutation=off", mutation_enabled=False)
    other.science = _payload(
        "eiel_science_ownership_v1",
        {
            **other.science.to_value(),
            "study_id": "OTHER-STUDY",
        },
    )
    assert derive_science_cuda_v2_native_rng_identity_v1(off) != (
        derive_science_cuda_v2_native_rng_identity_v1(other)
    )


def test_paired_masks_partition_batches_but_retain_one_native_rng_subject() -> None:
    off_control = project_science_cuda_v2_native_control_v1(
        _world(arm_id="off", mutation_enabled=False)
    )
    on_control = project_science_cuda_v2_native_control_v1(
        _world(arm_id="on", mutation_enabled=True)
    )
    identity = canonical_sha256({"pair": "same"})

    def mapping(world_id: str, control: ScienceCudaV2NativeControlV1):
        return ScienceCudaV2WorldMappingV1(
            world_id=world_id,
            world_uid=17 if world_id == "off" else 18,
            semantic_world_sha256=canonical_sha256({"world": world_id}),
            acquisition_sha256=canonical_sha256({"acquisition": world_id}),
            requested_observations=("population_count",),
            native_rng_world_uid=77,
            native_rng_identity_sha256=identity,
            native_control_vector=control.control_vector,
            native_control_digest=ContentDigest(
                algorithm="sha256", hex=control.control_digest
            ),
            native_capability_mask=control.capability_mask,
        )

    off = mapping("off", off_control)
    on = mapping("on", on_control)
    profile = SCIENCE_CUDA_V2_PROFILES[0]
    batches = tuple(
        ScienceCudaV2PhysicalBatchV1(
            physical_batch_index=index,
            carrier_profile=profile,
            backend_id=profile.backend_ids[0],
            temporal_k=1,
            world_slots=((0, world),),
            native_observation_union=("population_count",),
            native_control_vector=control.control_vector,
            native_control_digest=ContentDigest(
                algorithm="sha256", hex=control.control_digest
            ),
            native_capability_mask=control.capability_mask,
        )
        for index, (world, control) in enumerate(((off, off_control), (on, on_control)))
    )
    plan = ScienceCudaV2ExecutionPlanV1(
        bundle_content_sha256=canonical_sha256({"bundle": 1}),
        bundle_admission_sha256=canonical_sha256({"admission": 1}),
        carrier_profile_catalog_sha256=canonical_sha256({"catalog": 1}),
        logical_bundle_size=2,
        batches=batches,
    )
    assert [
        world.world_uid for batch in plan.batches for _, world in batch.world_slots
    ] == [17, 18]
    assert [
        world.native_rng_world_uid
        for batch in plan.batches
        for _, world in batch.world_slots
    ] == [77, 77]
    assert [batch.native_capability_mask for batch in plan.batches] == [95, 127]
    heterogeneous = ScienceCudaV2PhysicalBatchV1(
        physical_batch_index=0,
        carrier_profile=profile,
        backend_id=profile.backend_ids[0],
        temporal_k=1,
        world_slots=((0, off), (1, on)),
        native_observation_union=("population_count",),
    )
    assert heterogeneous.native_control_vector is None
    assert [
        world.native_capability_mask for _, world in heterogeneous.world_slots
    ] == [95, 127]


@pytest.mark.parametrize(
    "row",
    [
        {
            "control_id": "selection.pressure",
            "resolved_value": 0.5,
            "schedule": {"kind": "static"},
            "descriptor_sha256": canonical_sha256({"control": "selection"}).to_dict(),
        },
        {
            "control_id": "mutation.enabled",
            "resolved_value": 0.25,
            "schedule": {"kind": "static"},
            "descriptor_sha256": canonical_sha256({"control": "mutation"}).to_dict(),
        },
        {
            "control_id": "mutation.enabled",
            "resolved_value": False,
            "schedule": {"kind": "pulse_absolute_ticks"},
            "descriptor_sha256": canonical_sha256({"control": "mutation"}).to_dict(),
        },
    ],
)
def test_unrepresentable_controls_fail_closed(row: dict[str, object]) -> None:
    world = _world(arm_id="unsupported", mutation_enabled=False)
    semantic = world.semantic_inputs.to_value()
    semantic["control_bindings"] = [row]
    with pytest.raises(
        ScienceCudaV2NativeControlError, match="unsupported|requires|must"
    ):
        project_science_cuda_v2_native_control_v1(semantic)
