from __future__ import annotations

import pytest

from engine.tw_rngv2.address import (
    DOMAIN_BY_ID,
    AddressError,
    EventAddress,
    assign_accepted_child_uids,
    encode_counter,
    encode_key,
    pair_event_rank,
    validate_fields,
)
from engine.tw_rngv2.contract import CONTRACT_HASH, MASK48, SEMANTIC_EPOCH
from engine.tw_rngv2.status import Status


def test_counter_layout_roundtrip_golden() -> None:
    from engine.tw_rngv2.address import decode_counter

    ctr = encode_counter(
        subject_uid=0x123456789ABC,
        absolute_tick=0xFEDCBA987654,
        operation_id=0x1111,
        draw_block=0x2222,
    )
    assert ctr == (0x56789ABC, 0x76541234, 0xFEDCBA98, 0x22221111)
    decoded = decode_counter(ctr)
    assert decoded["subject_uid"] == 0x123456789ABC
    assert decoded["absolute_tick"] == 0xFEDCBA987654
    assert decoded["operation_id"] == 0x1111
    assert decoded["draw_block"] == 0x2222


def test_key_and_counter_are_structurally_injective() -> None:
    assert CONTRACT_HASH
    keys = set()
    counters = set()
    for world in (0, 1, 56, 0xFFFFFFFF):
        for domain in DOMAIN_BY_ID:
            key = encode_key(
                rng_namespace_key64=0x1111,
                world_uid=world,
                domain_id=domain,
                semantic_epoch=SEMANTIC_EPOCH,
            )
            assert key not in keys
            keys.add(key)
    for subject in (0, 1, MASK48):
        for tick in (0, 1, MASK48):
            for op in (0, 1, 0xFFFF):
                for block in (0, 1, 0xFFFF):
                    ctr = encode_counter(
                        subject_uid=subject,
                        absolute_tick=tick,
                        operation_id=op,
                        draw_block=block,
                    )
                    assert ctr not in counters
                    counters.add(ctr)
    assert len(keys) == 4 * len(DOMAIN_BY_ID)
    assert len(counters) == 3 * 3 * 3 * 3


def test_unknown_and_overflow_fail_closed() -> None:
    assert validate_fields(
        world_uid=0,
        domain_id=0x00FF,
        semantic_epoch=1,
        subject_uid=0,
        absolute_tick=0,
        operation_id=0,
        draw_block=0,
    ) is Status.RNG_UNKNOWN_DOMAIN
    assert validate_fields(
        world_uid=0,
        domain_id=0x0100,
        semantic_epoch=1,
        subject_uid=1 << 48,
        absolute_tick=0,
        operation_id=0,
        draw_block=0,
    ) is Status.RNG_FIELD_OVERFLOW
    with pytest.raises(AddressError):
        EventAddress(0, 0x0100, 0, 0, 0).counter_words(1 << 16)


def test_layout_and_slot_reuse_do_not_alias() -> None:
    first = EventAddress(0, 0x0310, subject_uid=7, absolute_tick=3, operation_id=1)
    reused_slot_same_entity = EventAddress(
        0, 0x0310, subject_uid=7, absolute_tick=9, operation_id=1
    )
    new_entity_same_slot = EventAddress(
        0, 0x0310, subject_uid=8, absolute_tick=9, operation_id=1
    )
    assert first.identity_tuple(0, 0) != reused_slot_same_entity.identity_tuple(0, 0)
    assert first.identity_tuple(0, 0) != new_entity_same_slot.identity_tuple(0, 0)
    assigned, nxt = assign_accepted_child_uids(
        next_entity_uid=10, accepted_in_canonical_order=(True, False, True)
    )
    assert assigned == [10, None, 11]
    assert nxt == 12


def test_pair_parent_order_is_canonical() -> None:
    peers = [((5, 2), 0), ((9, 1), 0), ((2, 5), 1)]
    rank_a = pair_event_rank((2, 5), 0, peers)
    rank_b = pair_event_rank((5, 2), 0, peers)
    assert rank_a == rank_b
    assert pair_event_rank((5, 2), 1, peers) != rank_a
