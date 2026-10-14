"""CPU preparation for C4 birth transactions across C5 replay boundaries."""

from __future__ import annotations

from engine.tensor_world.c4_fixture import (
    make_c4_adversarial_fixture_v1,
    make_c4_fixture_v1,
)
from engine.tensor_world.c4_population import (
    TensorWorldC4BirthConfigV1,
    commit_c4_ordinary_births_oracle_v1,
    plan_c4_ordinary_births_oracle_v1,
)
from engine.tensor_world.c4_transaction import execute_c4_population_transaction_oracle_v1
from engine.tensor_world.c5_determinism import (
    make_c5_pcg64_context_v1,
    run_c5_replay_resume_parity_v1,
)


def test_c5_replay_resume_preserves_c4_birth_transaction_once() -> None:
    """A birth transaction is visible once and is not replayed after resume."""

    def run_once() -> dict[str, object]:
        bundle, _ = make_c4_fixture_v1(
            batch=1, capacity=8, active=3, width=8, seed=31
        )
        context = make_c5_pcg64_context_v1(bundle, streams=1, seed=17)
        birth_config = TensorWorldC4BirthConfigV1(max_births=2)

        def advance(state, replay_context) -> None:
            # The event cursor is the safe-boundary ownership signal.  A
            # resumed run starts with a nonzero cursor and must not publish or
            # commit the already-settled birth transaction again.
            if bool((replay_context.event_sequence == 0).all().item()):
                plan_c4_ordinary_births_oracle_v1(
                    state,
                    birth_config,
                    rng_context=replay_context,
                    rng_stream=0,
                )
                commit_c4_ordinary_births_oracle_v1(
                    state,
                    birth_config,
                    replay_context=replay_context,
                )
            state.validate()

        result = run_c5_replay_resume_parity_v1(
            bundle,
            context,
            ticks=3,
            checkpoint_ticks=(1, 2, 3),
            advance=advance,
            source_commit="source-current-c5-c4-birth-cpu-test",
        )
        assert result["all_equal"] is True
        assert result["comparisons"] == {"1": True, "2": True, "3": True}
        return result

    first = run_once()
    second = run_once()
    assert first["fresh_digests"] == second["fresh_digests"]
    assert first["resumed_digests"] == second["resumed_digests"]
    # Tick zero is the pre-transaction digest; the first post-tick digest
    # includes two births and their typed event rows.  Later digests retain
    # those rows and identities without appending another birth transaction.
    assert first["fresh_digests"][1] != first["fresh_digests"][0]


def test_c5_replay_resume_binds_mixed_world_c4_death_events_once() -> None:
    """C4 death publication advances the C5 event cursor per world."""

    def run_once() -> dict[str, object]:
        bundle, _ = make_c4_adversarial_fixture_v1(
            "mixed_world_transaction_counts", seed=43
        )
        context = make_c5_pcg64_context_v1(bundle, streams=2, seed=29)
        receipts: list[dict[str, object]] = []

        def advance(state, replay_context) -> None:
            if bool((replay_context.event_sequence == 0).all().item()):
                receipt = execute_c4_population_transaction_oracle_v1(
                    state,
                    rng_context=replay_context,
                )
                assert receipt.phase_status_map["typed_lifecycle_event_commit"] == "PASS"
                assert receipt.phase_counts["typed_lifecycle_event_commit"][
                    "event_active_count"
                ] == (1, 2)
                receipts.append(receipt.as_dict())
            state.validate()

        result = run_c5_replay_resume_parity_v1(
            bundle,
            context,
            ticks=3,
            checkpoint_ticks=(1, 2, 3),
            advance=advance,
            source_commit="source-current-c5-c4-mixed-death-test",
        )
        assert len(receipts) == 1
        assert result["all_equal"] is True
        assert result["comparisons"] == {"1": True, "2": True, "3": True}
        return result

    first = run_once()
    second = run_once()
    assert first["fresh_digests"] == second["fresh_digests"]
    assert first["resumed_digests"] == second["resumed_digests"]
