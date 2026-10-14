from __future__ import annotations

import random

import pytest

from evolution.recomb72_inheritance import (
    AUDITED_LOCI,
    BOUNDED_INTERPOLATION,
    CURRENT_AUTHORITATIVE,
    FOUNDER_PATTERNS,
    INHERITANCE_MODES,
    LOCUS_BOUNDS,
    MODE_DRAW_BUDGETS,
    MUTATION_OFF_CONTRACT,
    PARENT_COPY_PER_LOCUS,
    WHOLE_PACKAGE_SINGLE_PARENT,
    Recomb72InheritanceError,
    canonical_package,
    deterministic_parent_pairs,
    founder_packages,
    inherit_package,
    merge_ancestry,
    mode_metadata,
    mode_sha256,
    registry_metadata,
    registry_sha256,
)


class CountingRandom(random.Random):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.draw_count = 0

    def random(self) -> float:
        self.draw_count += 1
        return super().random()


def test_frozen_registry_and_founder_patterns_are_complete_and_bounded() -> None:
    assert len(AUDITED_LOCI) == 6
    assert set(FOUNDER_PATTERNS) == {
        "single_extreme_scalar",
        "resource_specialization_extremes",
        "correlated_multi_trait_packages",
    }
    for pattern in FOUNDER_PATTERNS:
        a, b = founder_packages(pattern)
        assert tuple(a) == AUDITED_LOCI
        assert tuple(b) == AUDITED_LOCI
        assert canonical_package(a) == a
        assert canonical_package(b) == b
    a, b = founder_packages("single_extreme_scalar")
    assert a["exploration_bias"] == -1.0
    assert b["exploration_bias"] == 1.0
    a, b = founder_packages("resource_specialization_extremes")
    assert a["resource_handling_axis"] == -1.0
    assert b["resource_handling_axis"] == 1.0
    a, b = founder_packages("correlated_multi_trait_packages")
    assert a == {locus: LOCUS_BOUNDS[locus][0] for locus in AUDITED_LOCI}
    assert b == {locus: LOCUS_BOUNDS[locus][1] for locus in AUDITED_LOCI}


def test_current_authoritative_is_exact_midpoint_with_zero_draws() -> None:
    a, b = founder_packages("correlated_multi_trait_packages")
    rng = CountingRandom(11)
    result = inherit_package(a, b, mode=CURRENT_AUTHORITATIVE, rng=rng)
    assert rng.draw_count == result.draw_count == 0
    assert result.child == {
        locus: 0.5 * (a[locus] + b[locus]) for locus in AUDITED_LOCI
    }
    assert result.donor_by_locus == {}
    assert result.package_donor is None


def test_parent_copy_is_exact_per_locus_and_uses_fixed_draw_budget() -> None:
    a, b = founder_packages("correlated_multi_trait_packages")
    rng = CountingRandom(22)
    result = inherit_package(a, b, mode=PARENT_COPY_PER_LOCUS, rng=rng)
    assert rng.draw_count == result.draw_count == len(AUDITED_LOCI)
    assert result.draw_count == MODE_DRAW_BUDGETS[PARENT_COPY_PER_LOCUS]
    assert set(result.donor_by_locus) == set(AUDITED_LOCI)
    for locus in AUDITED_LOCI:
        assert result.child[locus] in {a[locus], b[locus]}
        expected = a[locus] if result.donor_by_locus[locus] == "a" else b[locus]
        assert result.child[locus] == expected


def test_bounded_interpolation_stays_in_parental_and_legal_intervals() -> None:
    a, b = founder_packages("correlated_multi_trait_packages")
    rng = CountingRandom(33)
    result = inherit_package(a, b, mode=BOUNDED_INTERPOLATION, rng=rng)
    assert rng.draw_count == result.draw_count == len(AUDITED_LOCI)
    for locus in AUDITED_LOCI:
        assert min(a[locus], b[locus]) <= result.child[locus] <= max(
            a[locus], b[locus]
        )
        lo, hi = LOCUS_BOUNDS[locus]
        assert lo <= result.child[locus] <= hi


def test_whole_package_donor_preserves_package_integrity_with_one_draw() -> None:
    a, b = founder_packages("correlated_multi_trait_packages")
    for seed in range(20):
        rng = CountingRandom(seed)
        result = inherit_package(
            a, b, mode=WHOLE_PACKAGE_SINGLE_PARENT, rng=rng
        )
        assert rng.draw_count == result.draw_count == 1
        assert result.child in (a, b)
        assert set(result.donor_by_locus.values()) == {result.package_donor}


def test_modes_are_deterministic_and_hash_bound() -> None:
    a, b = founder_packages("correlated_multi_trait_packages")
    assert len(registry_sha256()) == 64
    assert registry_metadata()["mutation_off_contract"] == MUTATION_OFF_CONTRACT
    assert MUTATION_OFF_CONTRACT["mutation_events_required"] == 0
    assert MUTATION_OFF_CONTRACT["mutation_rng_draws_required"] == 0
    hashes = set()
    for mode in INHERITANCE_MODES:
        first = inherit_package(a, b, mode=mode, rng=random.Random(44))
        second = inherit_package(a, b, mode=mode, rng=random.Random(44))
        assert first == second
        assert first.mode_sha256 == mode_sha256(mode)
        assert mode_metadata(mode)["draw_budget"]["inheritance.py"] == MODE_DRAW_BUDGETS[mode]
        hashes.add(first.mode_sha256)
    assert len(hashes) == len(INHERITANCE_MODES)


def test_registry_metadata_returns_detached_contract_data() -> None:
    metadata = registry_metadata()
    metadata["neutral_package"]["exploration_bias"] = 99.0
    metadata["founder_patterns"]["single_extreme_scalar"]["a"][
        "exploration_bias"
    ] = 99.0
    metadata["mutation_off_contract"]["scalar_mutation"] = True
    fresh = registry_metadata()
    assert fresh["neutral_package"]["exploration_bias"] == 0.0
    assert (
        fresh["founder_patterns"]["single_extreme_scalar"]["a"][
            "exploration_bias"
        ]
        == -1.0
    )
    assert fresh["mutation_off_contract"]["scalar_mutation"] is False


def test_parent_pair_schedule_is_order_invariant_genotype_blind_and_distinct() -> None:
    expected = deterministic_parent_pairs(
        [9, 4, 7, 2], seed=101, generation=5, child_count=32
    )
    assert expected == deterministic_parent_pairs(
        [2, 7, 4, 9], seed=101, generation=5, child_count=32
    )
    assert len(expected) == 32
    assert all(a != b for a, b in expected)
    assert set(value for pair in expected for value in pair) <= {2, 4, 7, 9}
    assert expected != deterministic_parent_pairs(
        [2, 4, 7, 9], seed=101, generation=6, child_count=32
    )


def test_ancestry_merge_is_equal_normalized_and_canonical() -> None:
    assert merge_ancestry({"founder-a": 2.0}, {"founder-b": 3.0}) == {
        "founder-a": 0.5,
        "founder-b": 0.5,
    }
    assert merge_ancestry(
        {"z": 1.0, "a": 1.0},
        {"a": 1.0, "b": 1.0},
    ) == {"a": 0.5, "b": 0.25, "z": 0.25}


@pytest.mark.parametrize(
    "call",
    [
        lambda: canonical_package({}),
        lambda: canonical_package(
            {
                **founder_packages("single_extreme_scalar")[0],
                "exploration_bias": 2.0,
            }
        ),
        lambda: inherit_package(
            *founder_packages("single_extreme_scalar"),
            mode="unknown",
            rng=random.Random(1),
        ),
        lambda: deterministic_parent_pairs(
            [1], seed=1, generation=0, child_count=1
        ),
        lambda: deterministic_parent_pairs(
            [1, 1], seed=1, generation=0, child_count=1
        ),
        lambda: merge_ancestry({}, {"b": 1.0}),
    ],
)
def test_contract_fails_closed(call) -> None:
    with pytest.raises(Recomb72InheritanceError):
        call()
