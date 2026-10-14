from __future__ import annotations

import pytest
import torch

from engine.perf.cuda_architecture_b_modular_state_abi import (
    ModularStorageError,
    pack_modular_storage,
    rebind_modular_lifecycle_storage,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_sequence_replay_v1 import (
    replay_modular_recurrent_mutation_sequence_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_sequence_v1 import (
    build_modular_recurrent_mutation_sequence_v1,
)
from tests.perf.test_eiel_modular_recurrent_mutation_sequence_replay_v1 import (
    PROFILE,
    _chain_fixture,
)


def _case():
    original_sequence, initial_records, identity_table, _ = _chain_fixture()
    authority = pack_modular_storage(
        initial_records,
        agent_capacity=len(initial_records),
        profile_id=PROFILE,
    )
    assert authority.identity_table == identity_table
    entries = [
        {
            key: value
            for key, value in entry.to_dict().items()
            if key != "receipt_sha256"
        }
        for entry in original_sequence.entries[: original_sequence.count]
        if entry is not None
    ]
    sequence = build_modular_recurrent_mutation_sequence_v1(
        profile_id=PROFILE,
        source_packed_sha256=authority.packed_sha256,
        entries=entries,
    )
    replay = replay_modular_recurrent_mutation_sequence_v1(
        sequence,
        authority.records_sidecar,
        identity_table=authority.identity_table,
    )
    expected = pack_modular_storage(
        replay.final_records,
        agent_capacity=authority.layout.agent_capacity,
        profile_id=PROFILE,
    )
    alive = torch.tensor(
        [record is not None for record in replay.final_records], dtype=torch.bool
    )
    ids = torch.tensor(
        [
            int(record["agent_id"]) if record is not None else -1
            for record in replay.final_records
        ],
        dtype=torch.int64,
    )
    tokens = torch.tensor(
        [
            int(record["agent_slot_token"]) if record is not None else 0
            for record in replay.final_records
        ],
        dtype=torch.int64,
    )
    return authority, sequence, expected, alive, ids, tokens


def test_sequence_checkpoint_rebind_replays_child_chain_exactly() -> None:
    authority, sequence, expected, alive, ids, tokens = _case()

    rebound = rebind_modular_lifecycle_storage(
        authority,
        profile_id=PROFILE,
        terminal_alive_mask=alive,
        terminal_slot_to_agent_id=ids,
        terminal_slot_tokens=tokens,
        selected_parent_slot_by_child_slot={1: 0, 2: 0},
        live_tensors=expected.tensors,
        mutation_receipt=sequence,
    )

    assert rebound.canonical_bytes() == expected.canonical_bytes()
    assert [record["agent_id"] for record in rebound.records_sidecar if record] == [
        100,
        400,
        300,
    ]


def test_sequence_checkpoint_rebind_prunes_independent_terminal_deaths() -> None:
    authority, sequence, expected, alive, ids, tokens = _case()
    records = list(expected.records_sidecar)
    records[0] = None
    expected = pack_modular_storage(
        records,
        agent_capacity=authority.layout.agent_capacity,
        profile_id=PROFILE,
    )
    alive[0] = False
    ids[0] = -1
    tokens[0] = 0

    rebound = rebind_modular_lifecycle_storage(
        authority,
        profile_id=PROFILE,
        terminal_alive_mask=alive,
        terminal_slot_to_agent_id=ids,
        terminal_slot_tokens=tokens,
        selected_parent_slot_by_child_slot={1: 0, 2: 0},
        live_tensors=expected.tensors,
        mutation_receipt=sequence,
    )

    assert rebound.canonical_bytes() == expected.canonical_bytes()
    assert rebound.records_sidecar[0] is None


def test_sequence_checkpoint_rebind_rejects_terminal_owner_mismatch() -> None:
    authority, sequence, expected, alive, ids, tokens = _case()
    ids[2] = 999

    with pytest.raises(ModularStorageError, match="terminal owner mismatch"):
        rebind_modular_lifecycle_storage(
            authority,
            profile_id=PROFILE,
            terminal_alive_mask=alive,
            terminal_slot_to_agent_id=ids,
            terminal_slot_tokens=tokens,
            selected_parent_slot_by_child_slot={1: 0, 2: 0},
            live_tensors=expected.tensors,
            mutation_receipt=sequence,
        )
