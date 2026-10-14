from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from agents.structural_identity_v1 import new_native_identity
from perf.cuda_architecture_b_modular_state_abi import (
    ModularStorageError,
    pack_modular_storage,
    rebind_modular_lifecycle_storage,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    build_modular_recurrent_mutation_receipt_v1,
    float64_to_bits,
    mutation_ancestry_sha256_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_replay_v1 import (
    replay_modular_recurrent_mutation_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_modular_v1 import (
    PreparedDirectResidentModularV1,
    derive_modular_lifecycle_structural_origins_v2,
)
from tests.test_eiel_cuda_architecture_b_direct_resident_modular_v1 import _patch
from tests.perf.test_eiel_cuda_architecture_b_modular_state_abi import (
    _module,
    _record,
)


PROFILE_ID = "b2-12x32-v1"
RUNTIME_MAP_NAMES = (
    "structural_live_module_state",
    "structural_live_module_outputs",
    "structural_development_latch_by_instance_id",
    "structural_development_effective_gain_by_instance_id",
)


def _authority():
    return pack_modular_storage(
        [
            _record(
                [_module(0), _module(1, gated=True)],
                with_runtime=True,
                agent_id=10,
                agent_slot_token=100,
            ),
            _record(
                [_module(2)],
                with_runtime=True,
                agent_id=11,
                agent_slot_token=110,
            ),
            _record(
                [_module(3)],
                with_runtime=True,
                agent_id=12,
                agent_slot_token=120,
            ),
            None,
        ],
        agent_capacity=4,
        profile_id=PROFILE_ID,
    )


def _terminal():
    return {
        "terminal_alive_mask": torch.tensor(
            [True, True, False, True], dtype=torch.bool
        ),
        "terminal_slot_to_agent_id": torch.tensor([20, 11, -1, 21], dtype=torch.int64),
        "terminal_slot_tokens": torch.tensor([200, 110, 120, 210], dtype=torch.int64),
    }


def _mutation_case(transaction_kind: str = "turnover"):
    source = _module(0)
    duplicate = deepcopy(source)
    duplicate["module_id"] = "m001"
    duplicate["params"]["live_output_gain"] = 0.0
    duplicate.update(
        new_native_identity(
            context={
                "campaign_id": "modular-lifecycle-rebind-test",
                "branch_id": "mutation",
                "generation": 0,
                "tick": 0,
                "child_agent_id": 20,
                "parent_agent_ids": [10],
                "lineage_token": "mutation-child",
            },
            operator_id="duplicate_with_decay",
            attempt_index=0,
            operator_role="neutral_duplicate",
            parent_module=source,
            inherited_homology=False,
        )
    )
    parent = _record(
        [source, duplicate],
        with_runtime=True,
        agent_id=10,
        agent_slot_token=100,
    )
    authority = pack_modular_storage(
        [parent, None, None, None],
        agent_capacity=4,
        profile_id=PROFILE_ID,
    )
    selected = authority.records_sidecar[0]["genome"]["modules"][1]
    transaction = {
        "transaction_kind": transaction_kind,
        "transaction_ordinal": 0,
        "parent_agent_id": 10,
        "parent_slot": 0,
        "parent_slot_token": 100,
        "child_agent_id": 20,
        "child_slot": 3,
        "child_slot_token": 200,
        "attempt_ordinal": 0,
        "draw0_float64_bits": float64_to_bits(0.6),
        "draw1_float64_bits": float64_to_bits(0.6796875),
        "operation": "refine_descendant_parameter",
        "status": "accepted",
        "reason": "proposal_valid",
        "selected_instance_id_index": authority.identity_table.index(
            selected["module_instance_id"]
        ),
        "selected_homology_id_index": authority.identity_table.index(
            selected["homology_id"]
        ),
        "changed_dimension": "live_input_gain",
        "changed": True,
    }
    receipt = build_modular_recurrent_mutation_receipt_v1(
        profile_id=PROFILE_ID,
        source_packed_sha256=authority.packed_sha256,
        ancestry_tape_sha256=mutation_ancestry_sha256_v1(
            transaction_kind=transaction_kind,
            parent_slot=0,
            child_slot=3,
            parent_agent_id=10,
            child_agent_id=20,
            child_slot_token=200,
        ),
        transactions=[transaction],
    )
    child = deepcopy(authority.records_sidecar[0])
    child["genome"] = replay_modular_recurrent_mutation_v1(
        child["genome"], receipt, identity_table=authority.identity_table
    )
    child["agent_id"] = 20
    child["agent_slot_token"] = 200
    for name in RUNTIME_MAP_NAMES:
        child[name] = {}
    expected = pack_modular_storage(
        [authority.records_sidecar[0], None, None, child],
        agent_capacity=4,
        profile_id=PROFILE_ID,
    )
    terminal = {
        "terminal_alive_mask": torch.tensor(
            [True, False, False, True], dtype=torch.bool
        ),
        "terminal_slot_to_agent_id": torch.tensor(
            [10, -1, -1, 20], dtype=torch.int64
        ),
        "terminal_slot_tokens": torch.tensor(
            [100, 0, 0, 200], dtype=torch.int64
        ),
    }
    return authority, terminal, receipt, expected


def test_lifecycle_rebind_is_deterministic_for_same_slot_reuse() -> None:
    authority = _authority()
    terminal = _terminal()

    first = rebind_modular_lifecycle_storage(
        authority,
        profile_id=PROFILE_ID,
        selected_parent_slot_by_child_slot={0: 0, 3: 1},
        **terminal,
    )
    second = rebind_modular_lifecycle_storage(
        authority,
        profile_id=PROFILE_ID,
        selected_parent_slot_by_child_slot={3: 1, 0: 0},
        **terminal,
    )

    assert len(first.layout.fields) == 57
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.records_sidecar[1] == authority.records_sidecar[1]
    assert first.records_sidecar[2] is None
    for child_slot, parent_slot in ((0, 0), (3, 1)):
        child = first.records_sidecar[child_slot]
        parent = authority.records_sidecar[parent_slot]
        assert child is not None and parent is not None
        assert child["genome"] == parent["genome"]
        assert child["gated_recurrent_enabled"] == parent["gated_recurrent_enabled"]
        assert child["agent_id"] == int(
            terminal["terminal_slot_to_agent_id"][child_slot].item()
        )
        assert child["agent_slot_token"] == int(
            terminal["terminal_slot_tokens"][child_slot].item()
        )
        for name in RUNTIME_MAP_NAMES:
            assert child[name] == {}


def test_lifecycle_rebind_replays_exact_mutation_before_live_plane_check() -> None:
    authority, terminal, receipt, expected = _mutation_case()

    rebound = rebind_modular_lifecycle_storage(
        authority,
        profile_id=PROFILE_ID,
        selected_parent_slot_by_child_slot={3: 0},
        live_tensors=expected.tensors,
        mutation_receipt=receipt,
        **terminal,
    )

    assert rebound.canonical_bytes() == expected.canonical_bytes()
    assert rebound.records_sidecar[3]["genome"]["modules"][1]["params"][
        "live_input_gain"
    ] == 0.875


def test_lifecycle_rebind_accepts_typed_ordinary_birth_mutation() -> None:
    authority, terminal, receipt, expected = _mutation_case("ordinary_birth")

    rebound = rebind_modular_lifecycle_storage(
        authority,
        profile_id=PROFILE_ID,
        selected_parent_slot_by_child_slot={3: 0},
        live_tensors=expected.tensors,
        mutation_receipt=receipt,
        **terminal,
    )

    assert receipt.transactions[0].transaction_kind == "ordinary_birth"
    assert rebound.canonical_bytes() == expected.canonical_bytes()


def test_lifecycle_rebind_rejects_mutation_receipt_provenance_mismatch() -> None:
    authority, terminal, receipt, expected = _mutation_case()
    transaction = receipt.transactions[0]
    wrong_source = build_modular_recurrent_mutation_receipt_v1(
        profile_id=PROFILE_ID,
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256=receipt.ancestry_tape_sha256,
        transactions=[transaction],
    )
    wrong_ancestry = build_modular_recurrent_mutation_receipt_v1(
        profile_id=PROFILE_ID,
        source_packed_sha256=authority.packed_sha256,
        ancestry_tape_sha256="22" * 32,
        transactions=[transaction],
    )

    for malformed, message in (
        (wrong_source, "source authority mismatch"),
        (wrong_ancestry, "ancestry mismatch"),
    ):
        with pytest.raises(ModularStorageError, match=message):
            rebind_modular_lifecycle_storage(
                authority,
                profile_id=PROFILE_ID,
                selected_parent_slot_by_child_slot={3: 0},
                live_tensors=expected.tensors,
                mutation_receipt=malformed,
                **terminal,
            )


@pytest.mark.parametrize(
    ("mapping", "message"),
    (
        ({3: 1}, "does not exactly cover"),
        ({0: 0, 1: 1, 3: 1}, "does not exactly cover"),
        ({0: 0, 3: 3}, "has no cold authority"),
        ({0: 0, 3: 4}, "out of range"),
        ({0: 0, True: 1}, "exact integers"),
    ),
)
def test_lifecycle_rebind_rejects_invalid_parent_mapping(
    mapping: dict[int, int],
    message: str,
) -> None:
    with pytest.raises(ModularStorageError, match=message):
        rebind_modular_lifecycle_storage(
            _authority(),
            profile_id=PROFILE_ID,
            selected_parent_slot_by_child_slot=mapping,
            **_terminal(),
        )


def test_lifecycle_rebind_rejects_profile_shape_and_duplicate_owner_mismatch() -> None:
    authority = _authority()
    terminal = _terminal()
    mapping = {0: 0, 3: 1}

    with pytest.raises(ModularStorageError, match="profile mismatch"):
        rebind_modular_lifecycle_storage(
            authority,
            profile_id="b2-16x48-v1",
            selected_parent_slot_by_child_slot=mapping,
            **terminal,
        )

    malformed = deepcopy(terminal)
    malformed["terminal_alive_mask"] = torch.ones(3, dtype=torch.bool)
    with pytest.raises(ModularStorageError, match="terminal_alive_mask"):
        rebind_modular_lifecycle_storage(
            authority,
            profile_id=PROFILE_ID,
            selected_parent_slot_by_child_slot=mapping,
            **malformed,
        )

    duplicate = deepcopy(terminal)
    duplicate["terminal_slot_to_agent_id"][3] = 11
    with pytest.raises(ModularStorageError, match="duplicate occupied terminal owner"):
        rebind_modular_lifecycle_storage(
            authority,
            profile_id=PROFILE_ID,
            selected_parent_slot_by_child_slot=mapping,
            **duplicate,
        )


def test_resident_cold_origin_disambiguates_tensor_identical_genomes() -> None:
    first = _record(
        [_module(0)],
        with_runtime=True,
        agent_id=10,
        agent_slot_token=100,
    )
    second = deepcopy(first)
    second["agent_id"] = 11
    second["agent_slot_token"] = 110
    first["genome"]["genes"]["cold_marker"] = "first"
    second["genome"]["genes"]["cold_marker"] = "second"
    authority = pack_modular_storage(
        [first, second, None, None],
        agent_capacity=4,
        profile_id=PROFILE_ID,
    )
    terminal = {
        "terminal_alive_mask": torch.tensor(
            [True, True, False, True], dtype=torch.bool
        ),
        "terminal_slot_to_agent_id": torch.tensor(
            [10, 11, -1, 20], dtype=torch.int64
        ),
        "terminal_slot_tokens": torch.tensor(
            [100, 110, 0, 200], dtype=torch.int64
        ),
    }
    expected = rebind_modular_lifecycle_storage(
        authority,
        profile_id=PROFILE_ID,
        selected_parent_slot_by_child_slot={3: 1},
        **terminal,
    )
    patch = _patch()
    patch = type(patch)(
        source=patch.source,
        arguments=patch.arguments,
        physical_generation_summary=patch.physical_generation_summary,
        physical_runtime_bookkeeping=patch.physical_runtime_bookkeeping,
        physical_patch_stats=patch.physical_patch_stats,
        physical_full_tick_output_tensors=patch.physical_full_tick_output_tensors,
        live_banks={
            "alive_mask": terminal["terminal_alive_mask"].unsqueeze(0),
            "slot_to_agent_id": terminal["terminal_slot_to_agent_id"].unsqueeze(0),
            "slot_tokens": terminal["terminal_slot_tokens"].unsqueeze(0),
        },
        checkpoint_projection_inputs=patch.checkpoint_projection_inputs,
    )
    prepared = PreparedDirectResidentModularV1(
        patch=patch,
        modular_tensors=tuple(
            expected.tensors[field.name].unsqueeze(0).clone()
            for field in expected.layout.fields
        ),
        profile_id=PROFILE_ID,
        cold_origin_slots=torch.tensor([[0, 1, -1, 1]], dtype=torch.int64),
        modular_authorities=(authority,),
    )

    mapping = derive_modular_lifecycle_structural_origins_v2(prepared)
    rebound = rebind_modular_lifecycle_storage(
        authority,
        profile_id=PROFILE_ID,
        selected_parent_slot_by_child_slot=mapping,
        **terminal,
    )

    assert mapping == {3: 1}
    assert rebound.records_sidecar[3]["genome"]["genes"]["cold_marker"] == "second"
