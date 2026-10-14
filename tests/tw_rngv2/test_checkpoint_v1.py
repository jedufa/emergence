from __future__ import annotations

from engine.tw_rngv2.address import EventAddress
from engine.tw_rngv2.checkpoint import capture_checkpoint, restore_checkpoint
from engine.tw_rngv2.contract import CONTRACT_HASH
from engine.tw_rngv2.oracle import PrototypeOracle
from engine.tw_rngv2.transaction import Transaction


def test_checkpoint_contains_only_semantic_fields() -> None:
    snap = capture_checkpoint(
        world_uids=(0, 1),
        absolute_tick=12,
        next_entity_uid=(4, 9),
    )
    payload = snap.to_dict()
    assert payload["contract_hash"] == CONTRACT_HASH
    assert payload["pending_work"] == "none"
    forbidden = {"mt", "tape", "gauss", "cursor", "philox_state"}
    blob = str(payload).lower()
    assert not any(word in blob for word in forbidden)
    restored = restore_checkpoint(payload)
    assert restored.digest() == snap.digest()


def test_late_failure_is_atomic_and_replay_matches() -> None:
    oracle = PrototypeOracle()
    tx = Transaction(oracle, world_uids=(0,), absolute_tick=3, next_entity_uid=(2,))
    addr = EventAddress(0, 0x0331, 2, 3, 1)
    first = tx.draw(addr)
    assert first.status.name == "RNG_OK"
    frontier = tx.abort()
    assert tx.published is None
    assert frontier.absolute_tick == 3
    again = oracle.generate(addr)
    assert again.words == first.words
    assert again.z0_bits == first.z0_bits
    assert again.accepted_ordinal == first.accepted_ordinal
