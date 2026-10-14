from __future__ import annotations

from tools.scripts import b48_r2_seed1708_localization_prereg_v1 as prereg


def _positive_row(child: int, generation: int) -> dict[str, object]:
    return {
        "cell_id": "cell-72d2e7a8ed9bf6e53a08",
        "paired_off_cell_id": "cell-1e1241233d3bdeb5832c",
        "seed": 1708,
        "founder_profile": "neutral",
        "mutation_rate_q10": 16,
        "mutation_step_q10": 16,
        "identity_type": "preregistered_package_identity",
        "state_hash": "a" * 64,
        "positive_mutation_persistence": True,
        "event_child_stable_id": child,
        "event_generation": generation,
        "event_transaction_id": f"tx-{child}",
        "terminal_descendant_carriers": [
            {
                "carrier_slot": 14,
                "carrier_stable_id": 3591,
                "exact_values": {"exact_hereditary_digest": "b" * 64},
            }
        ],
    }


def test_heldout_seed_selection_excludes_all_r1_seeds() -> None:
    seeds = prereg.deterministic_heldout_seeds()

    assert len(seeds) == 12
    assert len(set(seeds)) == 12
    assert not (set(seeds) & set(prereg.R1_SEEDS))
    assert seeds == prereg.deterministic_heldout_seeds()


def test_positive_rows_collapse_to_single_seed1708_family() -> None:
    result = prereg.classify_positive_family(
        [
            _positive_row(218, 1),
            _positive_row(355, 2),
            _positive_row(2801, 23),
            _positive_row(3129, 25),
            {
                **_positive_row(1, 1),
                "identity_type": "allele_genotype_state_identity",
            },
        ]
    )

    assert result["positive_row_count"] == 4
    assert result["positive_event_family_count"] == 1
    assert (
        result["interpretation"]
        == "one_package_identity_event_family_counted_across_four_event_rows"
    )
    family = result["families"][0]
    assert family["seed"] == 1708
    assert family["terminal_carrier_stable_id"] == 3591
    assert [row["event_generation"] for row in family["event_rows"]] == [1, 2, 23, 25]


def test_include_rows_pairs_off_and_low_for_each_seed() -> None:
    rows = prereg._include_rows([2222, 3333])

    assert rows == [
        {
            "ecology_seed": 2222,
            "task_seed": 0,
            "founder_profile": "neutral",
            "mutation_rate_q10": 0,
            "mutation_step_q10": 0,
        },
        {
            "ecology_seed": 2222,
            "task_seed": 0,
            "founder_profile": "neutral",
            "mutation_rate_q10": 16,
            "mutation_step_q10": 16,
        },
        {
            "ecology_seed": 3333,
            "task_seed": 0,
            "founder_profile": "neutral",
            "mutation_rate_q10": 0,
            "mutation_step_q10": 0,
        },
        {
            "ecology_seed": 3333,
            "task_seed": 0,
            "founder_profile": "neutral",
            "mutation_rate_q10": 16,
            "mutation_step_q10": 16,
        },
    ]


def test_reducer_contract_separates_rows_families_and_origins() -> None:
    contract = prereg.reducer_audit_contract()

    assert contract["primary_scientific_count"] == "positive_event_family_count"
    assert contract["raw_positive_row_count_is_not_independent_origin_count"] is True
    assert "event_transaction_id" in contract["event_family_key"]["excluded_from_family_key"]
    assert (
        contract["mutation_origin_key"]["status"]
        == "claim_only_if_full_ordinary_birth_chain_reconstructed"
    )
    assert "positive_event_families" in contract["required_reducer_fields"]
    assert "unique_mutation_origins_where_reconstructable" in contract["required_reducer_fields"]
