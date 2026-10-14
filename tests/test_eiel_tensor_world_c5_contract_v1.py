"""Preparation-bound C5 digest/checkpoint contract tests."""

from __future__ import annotations

import copy

import pytest
import torch

from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_events import append_c4_death_events_v1
from engine.tensor_world.c5_determinism import (
    C5_CHECKPOINT_SCHEMA_VERSION_V1,
    TensorWorldC5ReplayContextV1,
    canonical_tensor_world_digest_v1,
    c5_pcg64_bounded_uint32_v1,
    c5_pcg64_uniform_float32_v1,
    capture_c5_checkpoint_v1,
    make_c5_pcg64_context_v1,
    run_c5_replay_resume_parity_v1,
    restore_c5_checkpoint_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


def _context(bundle, *, streams: int = 3, words: int = 4):
    return TensorWorldC5ReplayContextV1(
        rng_state_words=torch.arange(
            bundle.batch * streams * words, dtype=torch.int64
        ).reshape(bundle.batch, streams, words),
        rng_cursors=torch.zeros((bundle.batch, streams), dtype=torch.int64),
        event_sequence=torch.zeros(bundle.batch, dtype=torch.int64),
        event_order_keys=torch.arange(
            bundle.batch * bundle.capacity, dtype=torch.int64
        ).reshape(bundle.batch, bundle.capacity),
    )


def test_c5_digest_requires_explicit_rng_and_event_context() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=4, width=8)
    with pytest.raises(TensorWorldContractError, match="explicit RNG/event"):
        canonical_tensor_world_digest_v1(bundle)


def test_c5_digest_is_repeatable_and_changes_with_trajectory_state() -> None:
    first, _ = make_c4_fixture_v1(batch=1, capacity=8, active=4, width=8)
    second, _ = make_c4_fixture_v1(batch=1, capacity=8, active=4, width=8)
    first_context = _context(first)
    second_context = _context(second)
    assert canonical_tensor_world_digest_v1(first, first_context) == canonical_tensor_world_digest_v1(
        second, second_context
    )

    first.base.base.core.agents.energy[0, 0] += 1.0
    assert canonical_tensor_world_digest_v1(first, first_context) != canonical_tensor_world_digest_v1(
        second, second_context
    )


def test_c5_capture_restore_preserves_state_and_rng_cursors() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    context = _context(bundle)
    checkpoint = capture_c5_checkpoint_v1(
        bundle,
        context,
        backend_id="tensor_world_c5_preparation_v1",
        source_commit="source-current-test",
        provider_contract_id="tensor_world_provider_preparation_v1",
        metadata={"safe_boundary": "tick_0", "checkpoint_cadence": 8},
    )
    assert checkpoint.schema_version == C5_CHECKPOINT_SCHEMA_VERSION_V1
    assert checkpoint.state.device.type == "cpu"
    original_context = copy.deepcopy(context.rng_state_words)

    bundle.base.base.core.agents.energy[0, 0] = 0.0
    context.rng_state_words.add_(100)
    context.rng_cursors.add_(7)
    restored = restore_c5_checkpoint_v1(
        checkpoint,
        bundle,
        context,
        expected_backend_id="tensor_world_c5_preparation_v1",
        expected_source_commit="source-current-test",
        expected_provider_contract_id="tensor_world_provider_preparation_v1",
    )
    assert restored == checkpoint.digest
    assert torch.equal(context.rng_state_words, original_context)
    assert torch.equal(
        context.rng_cursors,
        torch.zeros_like(context.rng_cursors),
    )
    assert canonical_tensor_world_digest_v1(bundle, context) == checkpoint.digest


def test_c5_restore_rejects_identity_mismatch_without_mutation() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=4, width=8)
    context = _context(bundle)
    checkpoint = capture_c5_checkpoint_v1(
        bundle,
        context,
        backend_id="tensor_world_c5_preparation_v1",
        source_commit="source-current-test",
        provider_contract_id="tensor_world_provider_preparation_v1",
    )
    before = bundle.base.base.core.agents.energy.clone()
    with pytest.raises(TensorWorldContractError, match="backend identity"):
        restore_c5_checkpoint_v1(
            checkpoint,
            bundle,
            context,
            expected_backend_id="wrong-backend",
            expected_source_commit="source-current-test",
            expected_provider_contract_id="tensor_world_provider_preparation_v1",
        )
    assert torch.equal(bundle.base.base.core.agents.energy, before)


def test_c5_checkpoint_detects_corrupted_snapshot() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=4, width=8)
    context = _context(bundle)
    checkpoint = capture_c5_checkpoint_v1(
        bundle,
        context,
        backend_id="tensor_world_c5_preparation_v1",
        source_commit="source-current-test",
        provider_contract_id="tensor_world_provider_preparation_v1",
    )
    checkpoint.state.base.base.core.agents.energy[0, 0] += 1.0
    with pytest.raises(TensorWorldContractError, match="checkpoint digest is corrupted"):
        restore_c5_checkpoint_v1(
            checkpoint,
            bundle,
            context,
            expected_backend_id="tensor_world_c5_preparation_v1",
            expected_source_commit="source-current-test",
            expected_provider_contract_id="tensor_world_provider_preparation_v1",
        )


def test_c5_context_rejects_event_cursor_divergence() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    context = _context(bundle)
    assert bundle.events is not None
    bundle.events.active_count[0] = 1
    with pytest.raises(TensorWorldContractError, match="event_sequence"):
        canonical_tensor_world_digest_v1(bundle, context)


def test_c5_restore_stages_optional_state_before_live_mutation() -> None:
    source, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    source.scalar_rng = None
    source_context = _context(source)
    checkpoint = capture_c5_checkpoint_v1(
        source,
        source_context,
        backend_id="tensor_world_c5_preparation_v1",
        source_commit="source-current-test",
        provider_contract_id="tensor_world_provider_preparation_v1",
    )

    destination, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    destination_context = _context(destination)
    before_energy = destination.base.base.core.agents.energy.clone()
    before_ids = destination.base.base.core.agents.agent_id.clone()
    with pytest.raises(TensorWorldContractError, match="destination type"):
        restore_c5_checkpoint_v1(
            checkpoint,
            destination,
            destination_context,
            expected_backend_id="tensor_world_c5_preparation_v1",
            expected_source_commit="source-current-test",
            expected_provider_contract_id="tensor_world_provider_preparation_v1",
        )
    assert torch.equal(destination.base.base.core.agents.energy, before_energy)
    assert torch.equal(destination.base.base.core.agents.agent_id, before_ids)


def test_c5_replay_resume_harness_matches_multiple_checkpoint_boundaries() -> None:
    bundle, _ = make_c4_fixture_v1(batch=2, capacity=8, active=3, width=8)
    context = make_c5_pcg64_context_v1(bundle, streams=2, seed=19)

    def advance(state, replay_context) -> None:
        for world in range(state.batch):
            values = c5_pcg64_uniform_float32_v1(
                replay_context, state, world=world, stream=0, count=2
            )
            bounded = c5_pcg64_bounded_uint32_v1(
                replay_context, state, world=world, stream=1, max_value=31
            )
            state.base.base.core.agents.energy[world, 0] += values[0].item() + bounded * 0.001
        state.validate()

    result = run_c5_replay_resume_parity_v1(
        bundle,
        context,
        ticks=6,
        checkpoint_ticks=(1, 3, 5),
        advance=advance,
        source_commit="source-current-test",
    )
    assert result["all_equal"] is True
    assert result["comparisons"] == {"1": True, "3": True, "5": True}


def test_c5_replay_ladder_preserves_multiworld_event_order_and_fresh_repeat() -> None:
    """Exercise event tape ordering and checkpoint resume on a CPU oracle.

    C5's digest must include both the C4 event tape and the explicit event
    cursor.  The first tick publishes one source-ordered death row per world;
    later ticks advance only agent/RNG state.  Repeating the entire run from
    the same fixture and seed must reproduce both fresh and resumed digests.
    """

    def run_once() -> dict[str, object]:
        bundle, _ = make_c4_fixture_v1(
            batch=2, capacity=8, active=3, width=8, seed=23
        )
        context = make_c5_pcg64_context_v1(bundle, streams=2, seed=71)

        def advance(state, replay_context) -> None:
            # Publish the event tape exactly once, at the first safe boundary.
            # ``event_sequence`` and the tape cursor are kept in lockstep;
            # replay from a later checkpoint therefore never republishes rows.
            if bool((replay_context.event_sequence == 0).all().item()):
                population = state.population
                for world in range(state.batch):
                    population.death_count[world] = 1
                    population.death_slots[world, 0] = 2
                    population.death_agent_ids[world, 0] = int(
                        state.base.base.core.agents.agent_id[world, 2]
                    )
                    population.death_reason_codes[world, 0] = 2
                append_c4_death_events_v1(
                    state,
                    generation=0,
                    tick=0,
                    event_sequence=replay_context.event_sequence,
                )

            for world in range(state.batch):
                values = c5_pcg64_uniform_float32_v1(
                    replay_context, state, world=world, stream=0, count=2
                )
                bounded = c5_pcg64_bounded_uint32_v1(
                    replay_context, state, world=world, stream=1, max_value=31
                )
                state.base.base.core.agents.energy[world, 0] += (
                    values[0].item() + bounded * 0.001
                )
            state.validate()

        return run_c5_replay_resume_parity_v1(
            bundle,
            context,
            ticks=4,
            checkpoint_ticks=(1, 2, 3, 4),
            advance=advance,
            source_commit="source-current-c5-cpu-test",
        )

    first = run_once()
    second = run_once()
    assert first["all_equal"] is True
    assert second["all_equal"] is True
    assert first["comparisons"] == {"1": True, "2": True, "3": True, "4": True}
    assert second["comparisons"] == first["comparisons"]
    assert first["fresh_digests"] == second["fresh_digests"]
    assert first["resumed_digests"] == second["resumed_digests"]
