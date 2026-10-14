"""Strict typed continuation contract for the composed R11 owner."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_event_ring_r11 import (
    TensorWorldC4EventRingRowR11V1,
    make_c4_event_ring_r11_v1,
)
from engine.tensor_world.c5_determinism import (
    C5_CONTINUATION_AUTHORITY_DECISION_R11_V1,
    C5_CONTINUATION_AUTHORITY_SCHEMA_VERSION_R11_V1,
    C5_TYPED_CONTINUATION_SCHEMA_VERSION_R11_V1,
    TensorWorldC5ReplayContextV1,
    c5_continuation_authority_source_receipt_r11_v1,
    canonical_tensor_world_digest_v1,
    capture_c5_checkpoint_v1,
    make_c5_pcg64_context_v1,
    restore_c5_checkpoint_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]
BACKEND = "tensor-world-r11-test"
SOURCE = "r11-test-source"
PROVIDER = "r11-test-provider"


def _event_row(*, sequence: int) -> TensorWorldC4EventRingRowR11V1:
    return TensorWorldC4EventRingRowR11V1(
        event_type=1,
        generation=7,
        tick=40 + sequence,
        actor_id=100 + sequence,
        target_id=-1,
        value=float(sequence),
    )


def _tensor_snapshot(value: object, path: str = "") -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor):
        snapshot[path] = value.clone()
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            snapshot.update(_tensor_snapshot(item, f"{path}[{index}]"))
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            snapshot.update(
                _tensor_snapshot(getattr(value, field.name), f"{path}.{field.name}")
            )
    return snapshot


def _assert_snapshot_exact(
    expected: dict[str, torch.Tensor], *values: tuple[str, object]
) -> None:
    actual: dict[str, torch.Tensor] = {}
    for name, value in values:
        actual.update(_tensor_snapshot(value, name))
    assert actual.keys() == expected.keys()
    for name, tensor in actual.items():
        assert tensor.equal(expected[name]), name


def test_r11_typed_continuation_has_distinct_roles_and_counters() -> None:
    bundle, _ = make_c4_fixture_v1(batch=2, capacity=8, active=3, width=8)
    context = make_c5_pcg64_context_v1(bundle, streams=2, seed=17, typed=True)
    context.validate_typed_continuation_for_bundle(bundle)
    assert context.typed_continuation_schema == C5_TYPED_CONTINUATION_SCHEMA_VERSION_R11_V1
    assert context.typed_role_indices == (0, 1, 2, 3, 4)
    assert tuple(context.raw64_transition_counts.shape) == (2, 5)
    assert tuple(context.output32_return_counts.shape) == (2, 5)


def test_r11_continuation_authority_receipt_preserves_legacy_contract() -> None:
    receipt = c5_continuation_authority_source_receipt_r11_v1()
    assert receipt["schema_version"] == (
        C5_CONTINUATION_AUTHORITY_SCHEMA_VERSION_R11_V1
    )
    assert receipt["decision"] == C5_CONTINUATION_AUTHORITY_DECISION_R11_V1
    assert receipt["legacy_without_event_ring"] == "bundle.events.active_count"
    assert receipt["typed_r11_with_event_ring"] == (
        "bundle.event_ring.producer_cursor"
    )
    assert receipt["event_sequence_mismatch"] == (
        "fail_closed_before_checkpoint_mutation"
    )
    assert receipt["gate_claim"] is False
    assert receipt["qualification_impact"] == {
        "legacy_event_cursor_contract": "PRESERVED_COMPATIBLE_INTERFACE",
        "r11_typed_event_continuation": "DELTA_QUALIFICATION_REQUIRED",
        "r11_gaussian_window_continuation": "DELTA_QUALIFICATION_REQUIRED",
        "broad_requalification_required": False,
    }
    assert all(
        len(source["sha256"]) == 64 and (ROOT / source["path"]).is_file()
        for source in receipt["sources"].values()
    )


def test_r11_strict_validation_rejects_legacy_anonymous_context() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=4, active=2, width=4)
    legacy = TensorWorldC5ReplayContextV1(
        rng_state_words=torch.zeros((1, 2, 4), dtype=torch.int64),
        rng_cursors=torch.zeros((1, 2), dtype=torch.int64),
        event_sequence=torch.zeros(1, dtype=torch.int64),
        event_order_keys=torch.zeros((1, 4), dtype=torch.int64),
        rng_buffer_words=torch.zeros((1, 2, 2), dtype=torch.int64),
        semantic_draw_counters=torch.zeros((1, 2), dtype=torch.int64),
    )
    legacy.rng_state_words[:, :, 2] = 1
    with pytest.raises(TensorWorldContractError, match="typed continuation"):
        legacy.validate_typed_continuation_for_bundle(bundle)


def test_r11_typed_checkpoint_restores_event_ring_at_control_boundary() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=4, active=2, width=4)
    bundle.event_ring = make_c4_event_ring_r11_v1(
        batch=1, capacity=4, event_capacity=2
    )
    context = make_c5_pcg64_context_v1(bundle, streams=2, seed=23, typed=True)
    checkpoint = capture_c5_checkpoint_v1(
        bundle,
        context,
        backend_id=BACKEND,
        source_commit=SOURCE,
        provider_contract_id=PROVIDER,
    )
    bundle.event_ring.append(
        world=0,
        rows=(TensorWorldC4EventRingRowR11V1(
            event_type=1,
            generation=1,
            tick=1,
            actor_id=0,
            target_id=-1,
            value=0.0,
        ),),
    )
    context.event_sequence.fill_(1)
    bundle.base.base.core.agents.health[0, 0] = -1.0
    digest = restore_c5_checkpoint_v1(
        checkpoint,
        bundle,
        context,
        expected_backend_id=BACKEND,
        expected_source_commit=SOURCE,
        expected_provider_contract_id=PROVIDER,
    )
    assert digest == checkpoint.digest
    assert int(bundle.event_ring.producer_cursor[0]) == 0
    assert float(bundle.base.base.core.agents.health[0, 0]) >= 0.0


def test_r11_post_event_checkpoint_uses_ring_producer_and_restores_exactly() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=4, active=2, width=4)
    bundle.event_ring = make_c4_event_ring_r11_v1(
        batch=1, capacity=4, event_capacity=2
    )
    context = make_c5_pcg64_context_v1(bundle, streams=2, seed=29, typed=True)
    bundle.event_ring.append(world=0, rows=(_event_row(sequence=0),))
    context.event_sequence.fill_(1)
    assert bundle.events is not None
    assert bundle.events.active_count.tolist() == [0]

    checkpoint = capture_c5_checkpoint_v1(
        bundle,
        context,
        backend_id=BACKEND,
        source_commit=SOURCE,
        provider_contract_id=PROVIDER,
        metadata={"boundary": "after_resident_event_commit"},
    )
    bundle.event_ring.append(world=0, rows=(_event_row(sequence=1),))
    context.event_sequence.fill_(2)
    bundle.base.base.core.agents.health[0, 0] = -1.0

    digest = restore_c5_checkpoint_v1(
        checkpoint,
        bundle,
        context,
        expected_backend_id=BACKEND,
        expected_source_commit=SOURCE,
        expected_provider_contract_id=PROVIDER,
    )
    assert digest == checkpoint.digest
    assert canonical_tensor_world_digest_v1(bundle, context) == checkpoint.digest
    assert bundle.events.active_count.tolist() == [0]
    assert bundle.event_ring.producer_cursor.tolist() == [1]
    assert context.event_sequence.tolist() == [1]
    assert bundle.event_ring.banks[0].typed_sequence[0, 0].item() == 0
    assert bundle.event_ring.banks[0].typed_generation[0, 0].item() == 7
    assert float(bundle.base.base.core.agents.health[0, 0]) >= 0.0


def test_r11_ring_cursor_mismatch_fails_closed_with_exact_rollback() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=4, active=2, width=4)
    bundle.event_ring = make_c4_event_ring_r11_v1(
        batch=1, capacity=4, event_capacity=2
    )
    context = make_c5_pcg64_context_v1(bundle, streams=2, seed=31, typed=True)
    bundle.event_ring.append(world=0, rows=(_event_row(sequence=0),))
    before_capture = _tensor_snapshot(bundle, "bundle") | _tensor_snapshot(
        context, "context"
    )
    with pytest.raises(TensorWorldContractError, match="event_ring.producer_cursor"):
        capture_c5_checkpoint_v1(
            bundle,
            context,
            backend_id=BACKEND,
            source_commit=SOURCE,
            provider_contract_id=PROVIDER,
        )
    _assert_snapshot_exact(before_capture, ("bundle", bundle), ("context", context))

    context.event_sequence.fill_(1)
    checkpoint = capture_c5_checkpoint_v1(
        bundle,
        context,
        backend_id=BACKEND,
        source_commit=SOURCE,
        provider_contract_id=PROVIDER,
    )
    bundle.base.base.core.agents.health[0, 0] = 3.25
    destination_before = _tensor_snapshot(bundle, "bundle") | _tensor_snapshot(
        context, "context"
    )
    corrupted = copy.deepcopy(checkpoint)
    corrupted.replay_context.event_sequence.zero_()
    with pytest.raises(TensorWorldContractError, match="event_ring.producer_cursor"):
        restore_c5_checkpoint_v1(
            corrupted,
            bundle,
            context,
            expected_backend_id=BACKEND,
            expected_source_commit=SOURCE,
            expected_provider_contract_id=PROVIDER,
        )
    _assert_snapshot_exact(
        destination_before,
        ("bundle", bundle),
        ("context", context),
    )


def test_legacy_cursor_still_uses_event_tape_when_ring_is_absent() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=4, active=2, width=4)
    assert bundle.event_ring is None and bundle.events is not None
    context = make_c5_pcg64_context_v1(bundle, streams=2, seed=37, typed=True)
    bundle.events.active_count.fill_(1)
    with pytest.raises(TensorWorldContractError, match="resident C4 event cursor"):
        context.validate_for_bundle(bundle)
