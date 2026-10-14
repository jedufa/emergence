from __future__ import annotations

from tools.scripts import b48_r2_seed1708_terminal_closure_v1 as reducer


def _row(
    index: int,
    *,
    seed: int,
    positive: bool,
    family_state: str | None = None,
) -> dict[str, object]:
    state = family_state or f"{index:064x}"
    row = {
        "cell_id": f"cell-{seed}",
        "paired_off_cell_id": f"off-{seed}",
        "seed": seed,
        "founder_profile": "neutral",
        "mutation_regime": "low_q10",
        "mutation_rate_q10": 16,
        "mutation_step_q10": 16,
        "identity_type": "preregistered_package_identity",
        "state_hash": state,
        "event_transaction_id": f"tx-{index}",
        "event_generation": index,
        "event_child_stable_id": index + 100,
        "terminal_descendant_carrier_count": 1 if positive else 0,
        "terminal_descendant_carriers": [
            {
                "carrier_stable_id": 3591 if family_state else index,
                "carrier_slot": 14 if family_state else index,
                "exact_values": {
                    "exact_hereditary_digest": "d" * 64 if family_state else f"{index:064x}"
                },
            }
        ],
        "positive_mutation_persistence": positive,
    }
    row["event_identity_key"] = reducer._event_identity_key(row)
    row["event_family_key"] = reducer.event_family_key(row)
    return row


def test_event_family_key_collapses_repeated_terminal_family() -> None:
    rows = [
        _row(1, seed=3062, positive=True, family_state="a" * 64),
        _row(2, seed=3062, positive=True, family_state="a" * 64),
    ]

    families = reducer._family_rows(rows)

    assert len(families) == 1
    assert families[0]["row_count"] == 2
    assert len(families[0]["event_rows"]) == 2


def test_summary_reports_rows_families_and_seed_rates_separately() -> None:
    primary = [
        _row(1, seed=3062, positive=True, family_state="a" * 64),
        _row(2, seed=3062, positive=True, family_state="a" * 64),
        _row(3, seed=8933, positive=False),
    ]
    secondary = [_row(4, seed=3062, positive=False)]

    summary = reducer.summarize_classified(
        primary_rows=primary,
        secondary_rows=secondary,
        seeds=[3062, 8933],
    )

    assert summary["primary_positive_event_row_count"] == 2
    assert summary["primary_positive_event_family_count"] == 1
    assert summary["primary_positive_seed_count"] == 1
    assert summary["primary_positive_rows_by_seed"] == {"3062": 2, "8933": 0}
    assert summary["primary_positive_families_by_seed"] == {"3062": 1, "8933": 0}
    assert summary["mutation_origin_claim_status"] == (
        "not_claimed_without_full_ordinary_birth_chain_reconstruction"
    )


def test_r2_pre_audit_decision_family() -> None:
    no_positive = {"primary_positive_event_family_count": 0}
    rare = {"primary_positive_event_family_count": 1, "primary_positive_seed_count": 1}
    replicated = {"primary_positive_event_family_count": 2, "primary_positive_seed_count": 2}

    assert reducer._decision(no_positive, comparisons_ok=True)["decision"] == "GV2-B48-R2-C"
    assert reducer._decision(rare, comparisons_ok=True)["decision"] == "GV2-B48-R2-B"
    assert reducer._decision(replicated, comparisons_ok=True)["decision"] == "GV2-B48-R2-A"
    assert reducer._decision(replicated, comparisons_ok=False)["decision"] == "GV2-B48-R2-E"
