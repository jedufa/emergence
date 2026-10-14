from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.actors.common import ActorError, new_record, revise_record, verify_record
from tools.actors.store import ActorRecordStore

FIXED = "2026-08-03T12:00:00Z"


def test_actor_store_preserves_revision_history_and_integrity(tmp_path: Path) -> None:
    store = ActorRecordStore(tmp_path, "test-records")
    first = new_record(
        actor="testActor",
        record_type="test_record",
        record_id="TST-ABC123",
        status="DRAFT",
        payload={"value": 1},
        generated_at=FIXED,
    )
    store.put(first)
    second = revise_record(
        first,
        payload={"value": 2},
        status="READY",
        action="update",
        generated_at="2026-08-03T12:01:00Z",
    )
    store.put(second, allow_update=True)

    assert store.get("TST-ABC123")["payload"] == {"value": 2}
    assert [row["revision"] for row in store.history("TST-ABC123")] == [1, 2]
    assert json.loads(store.index_path.read_text(encoding="utf-8"))["record_count"] == 1


def test_actor_store_is_semantically_idempotent(tmp_path: Path) -> None:
    store = ActorRecordStore(tmp_path, "test-records")
    first = new_record(
        actor="testActor",
        record_type="test_record",
        record_id="TST-IDEMPOTENT",
        status="READY",
        payload={"same": True},
        generated_at=FIXED,
    )
    second = new_record(
        actor="testActor",
        record_type="test_record",
        record_id="TST-IDEMPOTENT",
        status="READY",
        payload={"same": True},
        generated_at="2026-08-03T12:05:00Z",
    )
    store.put(first)
    store.put(second)
    assert store.get("TST-IDEMPOTENT")["created_at_utc"] == FIXED


def test_actor_record_tampering_is_detected(tmp_path: Path) -> None:
    store = ActorRecordStore(tmp_path, "test-records")
    record = new_record(
        actor="testActor",
        record_type="test_record",
        record_id="TST-TAMPER",
        status="READY",
        payload={"value": 1},
        generated_at=FIXED,
    )
    path = store.put(record)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["payload"]["value"] = 999
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert verify_record(tampered)
    with pytest.raises(ActorError):
        store.get("TST-TAMPER")
