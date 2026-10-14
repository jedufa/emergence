from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from eiel_composition.adaptive_expedition._legacy_atlas import (
    AdaptiveExpeditionAtlasError,
    add_atlas_entry_v1,
    atlas_entries_for_niche_v1,
    new_expedition_atlas_v1,
    validate_expedition_atlas_v1,
)
from eiel_composition.adaptive_expedition.ledger import (
    ACTION_DISPATCH_BOUND,
    ACTION_SETTLED,
    ACTION_WAIT,
    DECISION_PREPARED,
    EXPEDITION_TERMINAL,
    FRONTIER_OBSERVED,
    STOP_REQUESTED,
    LedgerIntegrityError,
    LedgerTransitionError,
    append_ledger_event_v1,
    append_ledger_store_event_v1,
    initialize_ledger_store_v1,
    new_expedition_ledger_v1,
    project_ledger_to_public_journal_v1,
    read_ledger_store_v1,
    reconstruct_expedition_frontier_v1,
    validate_expedition_ledger_v1,
)
from emergence_contracts.adaptive_science_expedition_v1 import JournalRecordV1
from eiel_composition.adaptive_expedition._legacy_parent_bank import (
    ParentBankCapacityError,
    ParentBankIntegrityError,
    add_parent_hold_v1,
    drop_parent_reference_v1,
    new_retained_parent_bank_v1,
    release_parent_hold_v1,
    retain_parent_v1,
    validate_retained_parent_bank_v1,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _ledger() -> dict[str, object]:
    return new_expedition_ledger_v1(
        expedition_id="expedition-test",
        manifest_sha256=_digest("manifest"),
        policy_sha256=_digest("policy"),
    )


def _observe(ledger: dict[str, object], ordinal: int) -> dict[str, object]:
    return append_ledger_event_v1(
        ledger,
        event_type=FRONTIER_OBSERVED,
        frontier_ordinal=ordinal,
        payload={
            "frontier_id": f"frontier-{ordinal}",
            "observation_sha256": _digest(f"observation-{ordinal}"),
        },
    )


def _prepare(
    ledger: dict[str, object], ordinal: int, action_ids: list[str]
) -> dict[str, object]:
    return append_ledger_event_v1(
        ledger,
        event_type=DECISION_PREPARED,
        frontier_ordinal=ordinal,
        payload={
            "decision_id": f"decision-{ordinal}",
            "decision_sha256": _digest(f"decision-{ordinal}"),
            "action_ids": action_ids,
            "rng_cursor_before_sha256": _digest(f"rng-before-{ordinal}"),
            "rng_cursor_after_sha256": _digest(f"rng-after-{ordinal}"),
        },
    )


def _bind(
    ledger: dict[str, object],
    ordinal: int,
    action_id: str,
    *,
    child_label: str | None = None,
    next_absolute_tick: int = 0,
) -> dict[str, object]:
    child = child_label or action_id
    return append_ledger_event_v1(
        ledger,
        event_type=ACTION_DISPATCH_BOUND,
        frontier_ordinal=ordinal,
        payload={
            "decision_id": f"decision-{ordinal}",
            "action_id": action_id,
            "child_request_sha256": _digest(f"request-{child}"),
            "child_binding_sha256": _digest(f"binding-{child}"),
            # Time belongs to the child binding, never to the outer frontier.
            "child_next_absolute_tick": next_absolute_tick,
        },
    )


def _settle(
    ledger: dict[str, object],
    ordinal: int,
    action_id: str,
    *,
    child_label: str | None = None,
) -> dict[str, object]:
    child = child_label or action_id
    return append_ledger_event_v1(
        ledger,
        event_type=ACTION_SETTLED,
        frontier_ordinal=ordinal,
        payload={
            "action_id": action_id,
            "child_binding_sha256": _digest(f"binding-{child}"),
            "authenticated_child_request_sha256": _digest(f"request-{child}"),
            "outcome": "COMPLETED",
            "child_result_sha256": _digest(f"result-{child}"),
        },
    )


def test_ledger_reconstructs_mixed_trajectory_frontier_without_shared_tick() -> None:
    ledger = _prepare(_observe(_ledger(), 0), 0, ["action-a", "action-b"])
    ledger = _bind(ledger, 0, "action-a", next_absolute_tick=0)
    ledger = append_ledger_event_v1(
        ledger,
        event_type=ACTION_WAIT,
        frontier_ordinal=0,
        payload={
            "action_id": "action-a",
            "child_binding_sha256": _digest("binding-action-a"),
            "wait_kind": "WAIT_STORAGE",
            "wait_receipt_sha256": _digest("wait-action-a"),
        },
    )
    ledger = _settle(ledger, 0, "action-a")
    ledger = _bind(ledger, 0, "action-b", next_absolute_tick=900)
    ledger = _settle(ledger, 0, "action-b")
    ledger = _prepare(_observe(ledger, 1), 1, [])
    ledger = append_ledger_event_v1(
        ledger,
        event_type=EXPEDITION_TERMINAL,
        frontier_ordinal=1,
        payload={"terminal_status": "COMPLETED"},
    )

    projection = reconstruct_expedition_frontier_v1(ledger)
    assert projection["current_frontier_ordinal"] == 1
    assert projection["pending_action_ids"] == []
    assert projection["terminal"]["terminal_status"] == "COMPLETED"
    assert "next_absolute_tick" not in projection["frontiers"][0]
    bindings = projection["frontiers"][0]["actions"]
    assert bindings["action-a"]["binding"]["child_next_absolute_tick"] == 0
    assert bindings["action-b"]["binding"]["child_next_absolute_tick"] == 900


def test_private_ledger_folds_into_single_public_journal_contract() -> None:
    ledger = _prepare(_observe(_ledger(), 0), 0, ["action-a", "action-b"])
    ledger = _bind(ledger, 0, "action-a")
    ledger = append_ledger_event_v1(
        ledger,
        event_type=ACTION_WAIT,
        frontier_ordinal=0,
        payload={
            "action_id": "action-a",
            "child_binding_sha256": _digest("binding-action-a"),
            "wait_kind": "WAIT_RESOURCE",
            "wait_receipt_sha256": _digest("wait-action-a"),
        },
    )
    ledger = _settle(ledger, 0, "action-a")
    ledger = _bind(ledger, 0, "action-b")
    ledger = _settle(ledger, 0, "action-b")

    mappings = project_ledger_to_public_journal_v1(ledger)
    records = [JournalRecordV1.from_dict(item) for item in mappings]
    assert [record.record_kind.value for record in records] == [
        "OBSERVATION_ACCEPTED",
        "DECISION_COMMITTED",
        "CHILD_REQUEST_GENERATED",
        "CHILD_REQUEST_GENERATED",
        "CHILD_REQUEST_GENERATED",
        "CHILD_REQUEST_GENERATED",
        "FRONTIER_SETTLED",
    ]
    assert records[0].previous_record_sha256 is None
    for prior, current in zip(records[:-1], records[1:], strict=True):
        assert current.previous_record_sha256 == prior.record_sha256()
    private_wait = records[3].payload.to_value()["private_event"]
    assert private_wait["event_type"] == ACTION_WAIT
    assert private_wait["payload"]["wait_kind"] == "WAIT_RESOURCE"


def test_ledger_requires_prepared_decision_before_dispatch() -> None:
    ledger = _observe(_ledger(), 0)
    with pytest.raises(LedgerTransitionError, match="prepared decision"):
        _bind(ledger, 0, "action-a")


def test_ledger_rejects_duplicate_action_and_child_bindings() -> None:
    ledger = _prepare(_observe(_ledger(), 0), 0, ["action-a", "action-b"])
    ledger = _bind(ledger, 0, "action-a")
    with pytest.raises(LedgerTransitionError, match="exactly once"):
        _bind(ledger, 0, "action-a")
    with pytest.raises(LedgerTransitionError, match="child binding identity"):
        _bind(ledger, 0, "action-b", child_label="action-a")
    with pytest.raises(LedgerTransitionError, match="child request identity"):
        append_ledger_event_v1(
            ledger,
            event_type=ACTION_DISPATCH_BOUND,
            frontier_ordinal=0,
            payload={
                "decision_id": "decision-0",
                "action_id": "action-b",
                "child_request_sha256": _digest("request-action-a"),
                "child_binding_sha256": _digest("binding-action-b"),
            },
        )


def test_ledger_rejects_settlement_from_wrong_authenticated_child_request() -> None:
    ledger = _prepare(_observe(_ledger(), 0), 0, ["action-a"])
    ledger = _bind(ledger, 0, "action-a")

    with pytest.raises(LedgerTransitionError, match="differs from its dispatch"):
        append_ledger_event_v1(
            ledger,
            event_type=ACTION_SETTLED,
            frontier_ordinal=0,
            payload={
                "action_id": "action-a",
                "child_binding_sha256": _digest("binding-action-a"),
                "authenticated_child_request_sha256": _digest(
                    "request-authenticated-wrong-child"
                ),
                "outcome": "COMPLETED",
                "child_result_sha256": _digest("result-action-a"),
            },
        )


def test_ledger_detects_tampering_anywhere_in_chain() -> None:
    ledger = _prepare(_observe(_ledger(), 0), 0, [])
    tampered = deepcopy(ledger)
    tampered["events"][0]["payload"]["frontier_id"] = "hostile"
    with pytest.raises(LedgerIntegrityError, match="digest changed"):
        validate_expedition_ledger_v1(tampered)


def test_stop_blocks_new_dispatch_but_allows_bound_child_to_settle() -> None:
    ledger = _prepare(_observe(_ledger(), 0), 0, ["action-a", "action-b"])
    ledger = _bind(ledger, 0, "action-a")
    ledger = append_ledger_event_v1(
        ledger,
        event_type=STOP_REQUESTED,
        frontier_ordinal=0,
        payload={"reason": "operator stop"},
    )
    with pytest.raises(LedgerTransitionError, match="cannot dispatch"):
        _bind(ledger, 0, "action-b")
    ledger = _settle(ledger, 0, "action-a")
    ledger = append_ledger_event_v1(
        ledger,
        event_type=EXPEDITION_TERMINAL,
        frontier_ordinal=0,
        payload={"terminal_status": "STOPPED"},
    )
    assert (
        reconstruct_expedition_frontier_v1(ledger)["terminal"]["terminal_status"]
        == "STOPPED"
    )
    records = [
        JournalRecordV1.from_dict(item)
        for item in project_ledger_to_public_journal_v1(ledger)
    ]
    assert records[-3].record_kind.value == "DECISION_COMMITTED"
    assert records[-3].payload.to_value()["private_event"]["event_type"] == (
        STOP_REQUESTED
    )
    assert records[-1].record_kind.value == "EXPEDITION_TERMINATED"


def test_ledger_store_reconstructs_without_mutable_current_file(tmp_path: Path) -> None:
    root = initialize_ledger_store_v1(tmp_path / "ledger", _ledger())
    stored = append_ledger_store_event_v1(
        root,
        event_type=FRONTIER_OBSERVED,
        frontier_ordinal=0,
        payload={
            "frontier_id": "frontier-0",
            "observation_sha256": _digest("observation-0"),
        },
    )
    assert read_ledger_store_v1(root) == stored
    assert not (root / "current.json").exists()

    event_path = root / "events" / "00000000.json"
    event_path.write_bytes(
        event_path.read_bytes().replace(b"frontier-0", b"frontier-x")
    )
    with pytest.raises(LedgerIntegrityError, match="digest changed"):
        read_ledger_store_v1(root)


def _atlas_entry(label: str, *, niche_id: str = "niche-a") -> dict[str, object]:
    return {
        "entry_id": f"entry-{label}",
        "candidate_id": f"candidate-{label}",
        "frontier_ordinal": 0,
        "niche_id": niche_id,
        "observation_frame_sha256": _digest(f"frame-{label}"),
        "semantic_world_sha256": _digest(f"semantic-{label}"),
        "acquisition_sha256": _digest(f"acquisition-{label}"),
        "reducer_result_sha256": _digest(f"reducer-{label}"),
        "descriptor_sha256": _digest(f"descriptor-{label}"),
        "source_settlement_sha256": _digest(f"settlement-{label}"),
        "epistemic": {
            "coordinates": {"novelty": 2, "viability": 0.75},
            "classification": "interesting",
        },
    }


def _atlas() -> dict[str, object]:
    return new_expedition_atlas_v1(
        expedition_id="expedition-test",
        manifest_sha256=_digest("manifest"),
        descriptor_set_sha256=_digest("descriptors"),
    )


def test_atlas_is_compact_derived_and_allows_same_niche_entries() -> None:
    atlas = add_atlas_entry_v1(_atlas(), _atlas_entry("a"))
    atlas = add_atlas_entry_v1(atlas, _atlas_entry("b"))
    entries = atlas_entries_for_niche_v1(atlas, niche_id="niche-a")
    assert [entry["candidate_id"] for entry in entries] == [
        "candidate-a",
        "candidate-b",
    ]
    assert atlas["state_retention_authority"] is False
    assert atlas["authority_effect"] == "none"


@pytest.mark.parametrize(
    "forbidden",
    [
        {"checkpoint_path": "/tmp/emergence-test/storage/checkpoint"},
        {"nested": {"state_ref": {"sha256": _digest("state")}}},
        {"lifecycle_ref": _digest("lifecycle")},
        {"raw_payload": "payload"},
    ],
)
def test_atlas_rejects_state_or_lifecycle_material(forbidden: object) -> None:
    entry = _atlas_entry("unsafe")
    entry["epistemic"] = forbidden
    with pytest.raises(AdaptiveExpeditionAtlasError, match="cannot retain state"):
        add_atlas_entry_v1(_atlas(), entry)


def test_atlas_rejects_duplicate_and_tampered_entries() -> None:
    atlas = add_atlas_entry_v1(_atlas(), _atlas_entry("a"))
    with pytest.raises(AdaptiveExpeditionAtlasError, match="duplicated"):
        add_atlas_entry_v1(atlas, _atlas_entry("a"))
    tampered = deepcopy(atlas)
    tampered["entries"][0]["epistemic"]["classification"] = "rewritten"
    with pytest.raises(AdaptiveExpeditionAtlasError, match="digest changed"):
        validate_expedition_atlas_v1(tampered)


def _caps(*, max_entries: int = 4, max_total_bytes: int = 400) -> dict[str, int]:
    return {
        "max_entries": max_entries,
        "max_total_bytes": max_total_bytes,
    }


def _bank(**caps: int) -> dict[str, object]:
    return new_retained_parent_bank_v1(
        expedition_id="expedition-test",
        manifest_sha256=_digest("manifest"),
        caps=_caps(**caps),
    )


def _hold(label: str) -> dict[str, str]:
    return {
        "hold_id": f"hold-{label}",
        "hold_kind": "ACTIVE_CHILD",
        "reference_sha256": _digest(f"hold-reference-{label}"),
    }


def _parent(
    label: str,
    *,
    niche_id: str = "niche-a",
    kind: str = "BRANCHABLE_EXACT",
    state_bytes: int = 60,
    priority: int = 0,
    holds: list[dict[str, str]] | None = None,
    next_absolute_tick: int = 0,
) -> dict[str, object]:
    return {
        "parent_id": f"parent-{label}",
        "candidate_id": f"candidate-{label}",
        "niche_id": niche_id,
        "frontier_ordinal": 0,
        "retention_priority": priority,
        "state_ref": {
            "kind": kind,
            "state_ref_sha256": _digest(f"state-{label}"),
            "state_bytes": state_bytes,
            "lifecycle_manifest_sha256": _digest(f"lifecycle-{label}"),
            "lifecycle_object_id": f"object-{label}",
            "next_absolute_tick": next_absolute_tick,
            "semantic_world_sha256": _digest(f"semantic-{label}"),
        },
        "holds": holds or [],
    }


def test_parent_bank_keeps_divergent_exact_and_replay_states_in_same_niche() -> None:
    first = retain_parent_v1(_bank(), _parent("a"))
    second = retain_parent_v1(
        first["bank"],
        _parent("b", kind="REPLAY_DERIVABLE", next_absolute_tick=120),
    )
    third = retain_parent_v1(second["bank"], _parent("c"))
    fourth = retain_parent_v1(third["bank"], _parent("d"))
    bank = fourth["bank"]
    assert bank["entry_count"] == 4
    assert {entry["state_ref"]["kind"] for entry in bank["entries"]} == {
        "BRANCHABLE_EXACT",
        "REPLAY_DERIVABLE",
    }
    assert {entry["niche_id"] for entry in bank["entries"]} == {"niche-a"}


def test_parent_bank_eviction_is_deterministic_and_diversity_aware() -> None:
    bank = retain_parent_v1(
        _bank(max_entries=2, max_total_bytes=200),
        _parent("a", niche_id="niche-a", priority=1),
    )["bank"]
    bank = retain_parent_v1(bank, _parent("b", niche_id="niche-a", priority=5))["bank"]
    update = retain_parent_v1(bank, _parent("c", niche_id="niche-b", priority=0))
    assert [entry["parent_id"] for entry in update["bank"]["entries"]] == [
        "parent-b",
        "parent-c",
    ]
    assert update["removed_references"][0]["parent_id"] == "parent-a"
    assert update["removed_references"][0]["effect"] == ("DROP_BANK_REFERENCE_ONLY")
    assert update["deletion_authority"] is False


def test_protected_hold_blocks_eviction_until_explicit_release() -> None:
    bank = retain_parent_v1(
        _bank(max_entries=1, max_total_bytes=100),
        _parent("a", holds=[_hold("active")]),
    )["bank"]
    with pytest.raises(ParentBankCapacityError, match="held"):
        retain_parent_v1(bank, _parent("b"))

    bank = release_parent_hold_v1(bank, parent_id="parent-a", hold_id="hold-active")
    update = retain_parent_v1(bank, _parent("b"))
    assert [entry["parent_id"] for entry in update["bank"]["entries"]] == ["parent-b"]
    assert update["lifecycle_release_authority"] is False


def test_parent_hold_updates_never_grant_deletion_authority() -> None:
    bank = retain_parent_v1(_bank(), _parent("a"))["bank"]
    bank = add_parent_hold_v1(bank, parent_id="parent-a", hold=_hold("consumer"))
    assert bank["entries"][0]["holds"][0]["hold_id"] == "hold-consumer"
    with pytest.raises(ParentBankCapacityError, match="protected"):
        drop_parent_reference_v1(bank, parent_id="parent-a", reason="settled")
    bank = release_parent_hold_v1(bank, parent_id="parent-a", hold_id="hold-consumer")
    update = drop_parent_reference_v1(
        bank, parent_id="parent-a", reason="POLICY_RELEASE"
    )
    assert update["bank"]["entry_count"] == 0
    assert update["removed_references"][0]["deletion_authority"] is False


def test_parent_bank_requires_exact_bytes_and_lifecycle_reference() -> None:
    missing = _parent("missing")
    del missing["state_ref"]["lifecycle_manifest_sha256"]
    with pytest.raises(ParentBankIntegrityError, match="fields differ"):
        retain_parent_v1(_bank(), missing)

    invalid_bytes = _parent("bytes", state_bytes=0)
    with pytest.raises(ParentBankIntegrityError, match="exact positive"):
        retain_parent_v1(_bank(), invalid_bytes)

    legacy_kind = _parent("legacy", kind="EXACT_STATE")
    with pytest.raises(ParentBankIntegrityError, match="kind is unsupported"):
        retain_parent_v1(_bank(), legacy_kind)


def test_parent_bank_rejects_duplicate_state_and_digest_tampering() -> None:
    bank = retain_parent_v1(_bank(), _parent("a"))["bank"]
    duplicate = _parent("b")
    duplicate["state_ref"]["state_ref_sha256"] = _digest("state-a")
    with pytest.raises(ParentBankIntegrityError, match="state identity"):
        retain_parent_v1(bank, duplicate)

    tampered = deepcopy(bank)
    tampered["entries"][0]["retention_priority"] = 99
    with pytest.raises(ParentBankIntegrityError, match="entry digest changed"):
        validate_retained_parent_bank_v1(tampered)
