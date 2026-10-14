from __future__ import annotations

import numpy as np

from engine.perf.hapb_v2.exactness_contract_v1 import (
    evaluate_fixed_mlp_policy_logits_from_pack_exact_v1,
    logits_sha256_v1,
)
from engine.perf.hapb_v2.fixtures_v1 import all_policy_eval_fixtures_v1


def test_fixtures_are_deterministic_and_match_the_contract() -> None:
    first = {fixture.name: fixture for fixture in all_policy_eval_fixtures_v1()}
    second = {fixture.name: fixture for fixture in all_policy_eval_fixtures_v1()}
    assert set(first) == {
        "tiny_single_row",
        "small_multi_row",
        "b48_like",
        "duplicate_content",
        "mixed_content",
    }
    for name, fixture in first.items():
        assert fixture.fixture_sha256 == second[name].fixture_sha256
        for row in range(fixture.rows):
            recomputed = evaluate_fixed_mlp_policy_logits_from_pack_exact_v1(
                fixture.packs[row],
                fixture.observations[row],
                slot=int(fixture.logical_slots[row]),
                drives=fixture.drives,
                attack_bias=fixture.attack_bias,
                action_temperature=fixture.action_temperature,
                drive_logit_gain=fixture.drive_logit_gain,
            )
            assert np.array_equal(recomputed, fixture.expected_logits[row])
            assert logits_sha256_v1(recomputed) == fixture.expected_logits_sha256[row]


def test_duplicate_and_mixed_content_groupings() -> None:
    fixtures = {fixture.name: fixture for fixture in all_policy_eval_fixtures_v1()}
    duplicate = fixtures["duplicate_content"]
    assert len(set(int(value) for value in duplicate.content_ids)) == 1
    assert all(
        np.array_equal(duplicate.packs[0], duplicate.packs[row])
        for row in range(duplicate.rows)
    )
    mixed = fixtures["mixed_content"]
    assert len(set(int(value) for value in mixed.content_ids)) == 3
