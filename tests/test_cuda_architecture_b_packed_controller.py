from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from perf.cuda_architecture_b_packed_controller import (
    allocate_flat_packed_controller_envelope,
    build_flat_packed_controller_envelope_layout,
    packed_controller_workspace_views,
    validate_flat_packed_controller_envelope_layout,
)
from perf.cuda_architecture_b_packed_state import (
    PACKED_STATE_ALIGNMENT_BYTES,
    PackedStateLayoutError,
    build_packed_state_layout,
)


SHAPE_BINDINGS = {
    "cell": 2,
    "capacity": 3,
    "active_population_count": 2,
    "resident_free_queue_count": 1,
    "policy_tick_stride": 2,
    "agent_id_capacity": 8,
    "height": 2,
    "width": 3,
    "drive_channels": 4,
    "chemistry_channels": 6,
    "gru_pack_width": 5,
    "gru_hidden": 2,
    "mlp_pack_width": 7,
    "mutation_rng_state_bytes": 16,
    "event_capacity": 4,
    "tick_capacity": 3,
    "ordinary_birth_capacity": 2,
    "turnover_capacity": 3,
    "birth_agent_id_capacity": 9,
}


@pytest.fixture(scope="module")
def state_layout():
    return build_packed_state_layout(SHAPE_BINDINGS)


def test_exactness_envelope_is_deterministic_aligned_and_keeps_state_prefix(
    state_layout,
) -> None:
    first = build_flat_packed_controller_envelope_layout(
        state_layout,
        max_ticks=2,
        observation_size=11,
        action_count=7,
    )
    repeated = build_flat_packed_controller_envelope_layout(
        state_layout,
        max_ticks=2,
        observation_size=11,
        action_count=7,
    )

    assert first == repeated
    assert first.manifest_sha256 == repeated.manifest_sha256
    assert first.state_bytes == state_layout.total_bytes
    assert first.envelope_bytes > first.state_bytes
    assert first.envelope_bytes % PACKED_STATE_ALIGNMENT_BYTES == 0
    assert first.workspace_fields[0].offset_bytes >= state_layout.total_bytes
    assert all(
        field.offset_bytes % PACKED_STATE_ALIGNMENT_BYTES == 0
        for field in first.workspace_fields
    )
    assert first.workspace_fields_by_name["observations"].shape == (2, 2, 3, 11)
    assert first.workspace_fields_by_name["logits"].shape == (2, 2, 3, 7)
    assert first.workspace_fields_by_name["active_mechanics_counts"].shape == (
        2,
        2,
        5,
    )
    assert first.workspace_fields_by_name["residual_death_agent_ids"].shape == (
        2,
        2,
        3,
    )


def test_production_capture_omits_diagnostic_history_and_is_smaller(state_layout) -> None:
    exactness = build_flat_packed_controller_envelope_layout(
        state_layout,
        max_ticks=8,
        observation_size=11,
        action_count=7,
        capture_mode="exactness",
    )
    production = build_flat_packed_controller_envelope_layout(
        state_layout,
        max_ticks=8,
        observation_size=11,
        action_count=7,
        capture_mode="production",
    )

    assert production.envelope_bytes < exactness.envelope_bytes
    assert "observations" not in production.workspace_fields_by_name
    assert "logits" not in production.workspace_fields_by_name
    assert "chemistry_draws" in production.workspace_fields_by_name
    assert "residual_death_count" in production.workspace_fields_by_name


def test_workspace_views_are_zero_copy_and_do_not_overlap_state(state_layout) -> None:
    controller_layout = build_flat_packed_controller_envelope_layout(
        state_layout,
        max_ticks=2,
        observation_size=11,
        action_count=7,
    )
    envelope = torch.zeros(controller_layout.envelope_bytes, dtype=torch.uint8)
    views = packed_controller_workspace_views(
        state_layout,
        controller_layout,
        envelope,
    )

    views["status"].fill_(9)
    status = controller_layout.workspace_fields_by_name["status"]
    assert torch.equal(
        envelope.narrow(0, status.offset_bytes, status.nbytes),
        torch.tensor([9, 0, 0, 0, 9, 0, 0, 0], dtype=torch.uint8),
    )
    assert not torch.count_nonzero(envelope[: state_layout.total_bytes])


def test_controller_layout_fails_closed_on_identity_or_geometry_changes(
    state_layout,
) -> None:
    layout = build_flat_packed_controller_envelope_layout(
        state_layout,
        max_ticks=2,
        observation_size=11,
        action_count=7,
    )
    with pytest.raises(PackedStateLayoutError, match="manifest hash mismatch"):
        validate_flat_packed_controller_envelope_layout(
            state_layout,
            replace(layout, max_ticks=3),
        )
    with pytest.raises(PackedStateLayoutError, match="capture mode"):
        build_flat_packed_controller_envelope_layout(
            state_layout,
            max_ticks=2,
            observation_size=11,
            action_count=7,
            capture_mode="debug",
        )
    with pytest.raises(PackedStateLayoutError, match="action_count"):
        build_flat_packed_controller_envelope_layout(
            state_layout,
            max_ticks=2,
            observation_size=11,
            action_count=6,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA envelope required")
def test_cuda_allocation_preserves_state_prefix_and_zeroes_workspace(state_layout) -> None:
    controller_layout = build_flat_packed_controller_envelope_layout(
        state_layout,
        max_ticks=2,
        observation_size=11,
        action_count=7,
    )
    state = torch.arange(
        state_layout.total_bytes,
        dtype=torch.int64,
        device="cuda",
    ).remainder_(251).to(torch.uint8)
    envelope = allocate_flat_packed_controller_envelope(
        state_layout,
        state,
        controller_layout,
    )

    assert torch.equal(envelope[: state_layout.total_bytes], state)
    assert not torch.count_nonzero(envelope[state_layout.total_bytes :])
