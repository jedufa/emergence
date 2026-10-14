from __future__ import annotations

from dataclasses import replace

import pytest

from agents.genome import Genome
from evolution.inheritance_linkage_digest_table_v1 import (
    InheritanceLinkageDigestTableError,
    LinkageDigestTableV1,
    build_linkage_digest_table_v1,
    founder_codes_v1,
)
from evolution.inheritance_registry_v1 import (
    LINKAGE_BLOCKS,
    canonical_hereditary_payload_sha256,
    canonical_parent_order,
)


def _founder(marker: int) -> Genome:
    low = marker < 0
    return Genome.from_dict(
        {
            "schema_version": "modular_genome_v1",
            "hidden_size": 16 if low else 32,
            "exploration_bias": -0.8 if low else 0.8,
            "aggression_bias": -1.2 if low else 1.2,
            "reproduction_threshold": 20.0 if low else 70.0,
            "mutation_rate": 0.01 if low else 0.02,
            "action_temperature": 0.5 if low else 2.0,
            "use_gru": False,
            "memory_dim": 0,
            "drive_inertia": 0.2 if low else 0.9,
            "drive_noise": 0.01 if low else 0.08,
            "drive_logit_gain": 0.4 if low else 2.4,
            "pred_lr": 0.1 if low else 0.8,
            "drive_sp_hunger": 0.2 if low else 0.8,
            "drive_sp_pain": 0.1 if low else 0.7,
            "drive_sp_fatigue": 0.3 if low else 0.9,
            "drive_sp_fear": 0.15 if low else 0.65,
            "exp_prior_strength": 0.2 if low else 0.7,
            "exp_prior_noise": 0.02 if low else 0.3,
            "drive_birth_noise": 0.01 if low else 0.15,
            "resource_handling_axis": -0.75 if low else 0.75,
            "family_head_residuals": (
                {"movement": 0.3, "gather": -0.3, "rest": 0.0, "attack": 0.0}
                if low
                else {"movement": 0.0, "gather": 0.0, "rest": -0.3, "attack": 0.3}
            ),
            "developmental_program": {
                "schema_version": "developmental_program_v1",
                "rules": [],
            },
            "modules": [],
            "connections": [],
        }
    )


def _ordered_founders() -> tuple[Genome, Genome]:
    parents, _digests = canonical_parent_order(_founder(-1), _founder(1))
    return parents


def test_table_covers_every_reachable_code_and_exact_digest_order() -> None:
    parent_0, parent_1 = _ordered_founders()
    table = build_linkage_digest_table_v1(parent_0, parent_1)

    assert table.block_ids == tuple(block_id for block_id, _ in LINKAGE_BLOCKS)
    assert table.genotype_count == 128
    assert founder_codes_v1(table) == (0, 127)
    assert table.digest_by_code[0] == canonical_hereditary_payload_sha256(parent_0)
    assert table.digest_by_code[127] == canonical_hereditary_payload_sha256(parent_1)
    for left in range(table.genotype_count):
        for right in range(table.genotype_count):
            observed = table.canonical_pair(left, right)
            expected = (
                (left, right)
                if table.digest_by_code[left] <= table.digest_by_code[right]
                else (right, left)
            )
            assert observed == expected


def test_child_code_copies_one_allele_per_block_from_canonical_parents() -> None:
    table = build_linkage_digest_table_v1(*_ordered_founders())
    parent_0 = 0b1010101
    parent_1 = 0b0101010
    donors = (0, 1, 1, 0, 1, 0, 0)
    child = table.child_code(parent_0, parent_1, donors)
    for block, donor in enumerate(donors):
        expected = (
            (parent_0 >> block) & 1
            if donor == 0
            else (parent_1 >> block) & 1
        )
        assert (child >> block) & 1 == expected


def test_table_payload_is_hash_bound_and_round_trips() -> None:
    table = build_linkage_digest_table_v1(*_ordered_founders())
    payload = table.payload()
    restored = LinkageDigestTableV1(
        schema_version=payload["schema_version"],
        registry_version=payload["registry_version"],
        parent_order_version=payload["parent_order_version"],
        genotype_code_version=payload["genotype_code_version"],
        block_ids=tuple(payload["block_ids"]),
        digest_by_code=tuple(payload["digest_by_code"]),
        rank_by_code=tuple(payload["rank_by_code"]),
        table_sha256=payload["table_sha256"],
    )
    assert restored == table
    with pytest.raises(
        InheritanceLinkageDigestTableError,
        match="hash mismatch",
    ):
        replace(table, table_sha256="0" * 64)


def test_equal_founder_alleles_share_rank_and_preserve_argument_tie_order() -> None:
    left, right = _ordered_founders()
    # Only one declared block differs; codes differing solely in the other six
    # blocks intentionally collapse to the same authoritative digest.
    for _block_id, fields in LINKAGE_BLOCKS[1:]:
        for field in fields:
            if hasattr(right, field) and hasattr(left, field):
                setattr(right, field, getattr(left, field))
    right.refresh_canonical_payload_cache()
    ordered, _ = canonical_parent_order(left, right)
    table = build_linkage_digest_table_v1(*ordered)
    assert len(set(table.digest_by_code)) == 2
    equal_codes = [
        code
        for code, digest in enumerate(table.digest_by_code)
        if digest == table.digest_by_code[0]
    ]
    assert len(equal_codes) == 64
    assert table.canonical_pair(equal_codes[1], equal_codes[0]) == (
        equal_codes[1],
        equal_codes[0],
    )


def test_founder_difference_outside_registry_fails_closed() -> None:
    parent_0, parent_1 = _ordered_founders()
    parent_1.sensor_radius = parent_0.sensor_radius + 1
    parent_1.refresh_canonical_payload_cache()
    ordered, _ = canonical_parent_order(parent_0, parent_1)
    with pytest.raises(
        InheritanceLinkageDigestTableError,
        match="outside the linkage registry",
    ):
        build_linkage_digest_table_v1(*ordered)

