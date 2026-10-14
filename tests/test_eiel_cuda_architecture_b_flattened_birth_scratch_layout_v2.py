from __future__ import annotations

import pytest

from engine.perf.cuda_architecture_b_flattened_birth_scratch_layout_v2 import (
    BIRTH_SCRATCH_SEGMENTS_V2,
    INT64_MAX_V2,
    LEGACY_SCRATCH_PREFIX_BYTES_V2,
    SCRATCH_ARENA_ALIGNMENT_V2,
    SCRATCH_SEGMENT_ALIGNMENT_V2,
    BirthScratchLayoutArithmeticError,
    BirthScratchLayoutStatusV2,
    checked_align_up_v2,
    checked_multiply_nonnegative_v2,
    make_flattened_birth_scratch_layout_v2,
    make_legacy_tail_birth_scratch_layout_v2,
)


EXPECTED_B64_OFFSETS = {
    "mini_slot_rows": 0,
    "mini_alive_rows": 1_536,
    "mini_genome_int32_planes": 1_728,
    "mini_policy_memory_dimension_rows": 7_872,
    "mini_policy_use_gru_rows": 9_408,
    "mini_genome_float_planes": 9_600,
    "mini_energy_rows": 28_032,
    "parent_plan_rows": 29_568,
    "mate_plan_rows": 30_080,
    "child_plan_rows": 30_592,
    "identity_parent_slots": 31_104,
    "identity_child_slots": 31_616,
    "identity_child_ids": 32_128,
    "identity_child_energy": 32_640,
    "identity_count": 33_152,
    "queue_head": 33_168,
    "queue_count": 33_184,
    "parent_agent_ids": 33_200,
    "child_lineage_founder_ids": 33_712,
    "child_lineage_depths": 33_968,
    "genome_int32_planes": 34_224,
    "policy_memory_dimension_rows": 36_272,
    "policy_use_gru_rows": 36_784,
    "genome_float_planes": 36_848,
    "drive_rows": 42_992,
    "chemistry_rows": 44_016,
    "birth_reset_shock_rows": 45_552,
    "policy_mlp_rows": 45_808,
    "hmem_rows": 570_096,
    "python_random_post_words": 578_288,
    "python_random_trial_words": 580_784,
    "python_random_snapshots": 583_280,
    "planner_pcg64_snapshot": 583_360,
    "runner_pcg64_snapshot": 583_408,
    "population_pcg64_snapshot": 583_456,
    "torch_state": 583_504,
    "mutation_flags": 588_560,
    "mutation_deltas": 588_624,
    "linked_donor_ordinals": 588_880,
    "diploid_staged_children": 589_328,
    "genome_upgrade_v2_staged_children": 605_712,
    "event_staging_rows": 668_176,
    "genome_upgrade_v2_qualification_fault_stage": 681_488,
    "scalar_statuses": 681_504,
}

EXPECTED_B64_BYTES = {
    "mini_slot_rows": 1_536,
    "mini_alive_rows": 192,
    "mini_genome_int32_planes": 6_144,
    "mini_policy_memory_dimension_rows": 1_536,
    "mini_policy_use_gru_rows": 192,
    "mini_genome_float_planes": 18_432,
    "mini_energy_rows": 1_536,
    "parent_plan_rows": 512,
    "mate_plan_rows": 512,
    "child_plan_rows": 512,
    "identity_parent_slots": 512,
    "identity_child_slots": 512,
    "identity_child_ids": 512,
    "identity_child_energy": 512,
    "identity_count": 8,
    "queue_head": 8,
    "queue_count": 8,
    "parent_agent_ids": 512,
    "child_lineage_founder_ids": 256,
    "child_lineage_depths": 256,
    "genome_int32_planes": 2_048,
    "policy_memory_dimension_rows": 512,
    "policy_use_gru_rows": 64,
    "genome_float_planes": 6_144,
    "drive_rows": 1_024,
    "chemistry_rows": 1_536,
    "birth_reset_shock_rows": 256,
    "policy_mlp_rows": 524_288,
    "hmem_rows": 8_192,
    "python_random_post_words": 2_496,
    "python_random_trial_words": 2_496,
    "python_random_snapshots": 80,
    "planner_pcg64_snapshot": 48,
    "runner_pcg64_snapshot": 48,
    "population_pcg64_snapshot": 48,
    "torch_state": 5_056,
    "mutation_flags": 64,
    "mutation_deltas": 256,
    "linked_donor_ordinals": 448,
    "diploid_staged_children": 16_384,
    "genome_upgrade_v2_staged_children": 62_464,
    "event_staging_rows": 13_312,
    "genome_upgrade_v2_qualification_fault_stage": 4,
    "scalar_statuses": 176,
}


def test_layout_matches_header_segment_order_offsets_and_bytes() -> None:
    layout = make_flattened_birth_scratch_layout_v2(64, 64, 2_048, 32, 5_056)

    assert layout.ok()
    assert layout.status is BirthScratchLayoutStatusV2.OK
    assert layout.mini_state_rows == 192
    assert [segment.value for segment in BIRTH_SCRATCH_SEGMENTS_V2] == list(
        EXPECTED_B64_OFFSETS
    )
    assert dict(layout.offsets) == EXPECTED_B64_OFFSETS
    assert dict(layout.relative_offsets) == EXPECTED_B64_OFFSETS
    assert dict(layout.segment_bytes) == EXPECTED_B64_BYTES
    assert layout.required_bytes == 681_728
    assert layout.bytes_per_cell == 681_728

    previous_end = 0
    for segment in BIRTH_SCRATCH_SEGMENTS_V2:
        child = layout.segment(segment)
        assert child.offset % SCRATCH_SEGMENT_ALIGNMENT_V2 == 0
        assert child.offset >= previous_end
        previous_end = child.end_offset
    assert previous_end <= layout.required_bytes
    assert layout.required_bytes % SCRATCH_ARENA_ALIGNMENT_V2 == 0


def test_required_bytes_are_monotonic_for_header_contract_examples() -> None:
    layouts = [
        make_flattened_birth_scratch_layout_v2(
            births, 64, 2_048, 32, 5_056
        )
        for births in range(65)
    ]

    assert all(layout.ok() for layout in layouts)
    assert [layout.required_bytes for layout in layouts] == sorted(
        layout.required_bytes for layout in layouts
    )
    assert layouts[0].required_bytes > 0


def test_legacy_prefix_places_unchanged_layout_in_tail() -> None:
    relative = make_flattened_birth_scratch_layout_v2(2, 64, 2_048, 32, 5_056)
    tail = make_legacy_tail_birth_scratch_layout_v2(
        2, 64, 2_048, 32, 5_056
    )

    assert tail.ok()
    assert tail.prefix_bytes == LEGACY_SCRATCH_PREFIX_BYTES_V2
    assert tail.tail_offset == LEGACY_SCRATCH_PREFIX_BYTES_V2
    assert tail.required_bytes == relative.required_bytes
    assert tail.bytes_per_cell == (
        LEGACY_SCRATCH_PREFIX_BYTES_V2 + relative.required_bytes
    )
    assert dict(tail.relative_offsets) == dict(relative.offsets)
    assert dict(tail.offsets) == {
        name: LEGACY_SCRATCH_PREFIX_BYTES_V2 + offset
        for name, offset in relative.offsets.items()
    }


@pytest.mark.parametrize(
    ("arguments", "status"),
    [
        ((-1, 64, 1, 1, 1), BirthScratchLayoutStatusV2.NEGATIVE_CONFIGURATION),
        (
            (65, 64, 1, 1, 1),
            BirthScratchLayoutStatusV2.BIRTH_COUNT_EXCEEDS_CAPACITY,
        ),
        (
            (INT64_MAX_V2, INT64_MAX_V2, 0, 0, 0),
            BirthScratchLayoutStatusV2.INT64_OVERFLOW,
        ),
        (
            (1, 1, INT64_MAX_V2, 0, 0),
            BirthScratchLayoutStatusV2.INT64_OVERFLOW,
        ),
        (
            (0, 0, 0, 0, INT64_MAX_V2),
            BirthScratchLayoutStatusV2.INT64_OVERFLOW,
        ),
    ],
)
def test_invalid_header_configurations_fail_closed(
    arguments: tuple[int, int, int, int, int],
    status: BirthScratchLayoutStatusV2,
) -> None:
    layout = make_flattened_birth_scratch_layout_v2(*arguments)

    assert not layout.ok()
    assert layout.status is status
    assert not layout.segments
    assert layout.required_bytes == 0
    assert layout.bytes_per_cell == 0


def test_late_overflow_preserves_header_derived_mini_state_rows() -> None:
    layout = make_flattened_birth_scratch_layout_v2(
        1, 1, INT64_MAX_V2, 0, 0
    )

    assert layout.status is BirthScratchLayoutStatusV2.INT64_OVERFLOW
    assert layout.mini_state_rows == 3


def test_prefix_and_public_arithmetic_helpers_are_checked() -> None:
    aligned = make_flattened_birth_scratch_layout_v2(
        0, 0, 0, 0, 0, prefix_bytes=LEGACY_SCRATCH_PREFIX_BYTES_V2 + 1
    )
    overflow = make_flattened_birth_scratch_layout_v2(
        0, 0, 0, 0, 0, prefix_bytes=INT64_MAX_V2
    )

    assert aligned.tail_offset == LEGACY_SCRATCH_PREFIX_BYTES_V2 + 256
    assert aligned.tail_offset % SCRATCH_ARENA_ALIGNMENT_V2 == 0
    assert overflow.status is BirthScratchLayoutStatusV2.INT64_OVERFLOW
    assert checked_align_up_v2(17, 16) == 32
    assert checked_multiply_nonnegative_v2(7, 9) == 63
    with pytest.raises(BirthScratchLayoutArithmeticError):
        checked_align_up_v2(INT64_MAX_V2, 16)
    with pytest.raises(BirthScratchLayoutArithmeticError):
        checked_align_up_v2(0, INT64_MAX_V2 + 1)
    with pytest.raises(BirthScratchLayoutArithmeticError):
        checked_multiply_nonnegative_v2(INT64_MAX_V2, 2)
    with pytest.raises(BirthScratchLayoutArithmeticError):
        checked_multiply_nonnegative_v2(INT64_MAX_V2 + 1, 0)
    with pytest.raises(TypeError):
        make_flattened_birth_scratch_layout_v2(True, 1, 1, 1, 1)
