"""Small file-backed record store shared by EIEL CLI actors."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.actors.canonical import canonical_sha256
from tools.actors.common import (
    ActorError,
    advisory_lock,
    atomic_write_json,
    load_actor_record,
    read_json,
    require_explicit_schema_version,
    validate_record_id,
    verify_record,
)
from tools.actors.contract_versions import WORKER_RESULT_SCHEMA_VERSION

INDEX_SCHEMA_VERSION = "eiel_actor_record_index_v1"
INDEX_MANIFEST_SCHEMA_VERSION = "eiel_actor_record_index_manifest_v2"
INDEX_ROW_SCHEMA_VERSION = "eiel_actor_record_index_row_v1"
HISTORY_HEAD_SCHEMA_VERSION = "eiel_actor_history_head_v1"
INDEX_SHARD_COUNT = 256

STORE_AUDIT_VALID = "VALID"
STORE_AUDIT_RECOVERABLE_STALE = "RECOVERABLE_STALE"
STORE_AUDIT_CORRUPT = "CORRUPT"



@dataclass(frozen=True)
class NamespaceWritePolicy:
    """Fail-closed identity contract for one authoritative record namespace."""

    actor: str
    record_types: frozenset[str]
    payload_schema_versions: frozenset[str]


# These namespaces carry lifecycle or repository-authoritative truth.  Reads
# remain available through ordinary ActorRecordStore instances, while semantic
# writes require a manager-scoped adapter and an envelope matching this policy.
NAMESPACE_WRITE_POLICIES: dict[str, NamespaceWritePolicy] = {
    "work": NamespaceWritePolicy(
        actor="workManager",
        record_types=frozenset({"work_item"}),
        payload_schema_versions=frozenset({"eiel_work_item_v1"}),
    ),
    "sessions": NamespaceWritePolicy(
        actor="sessionManager",
        record_types=frozenset({"codex_session"}),
        payload_schema_versions=frozenset({"eiel_codex_session_v1"}),
    ),
    "bootstrap": NamespaceWritePolicy(
        actor="bootstrap",
        record_types=frozenset({"repository_bootstrap"}),
        payload_schema_versions=frozenset({"eiel_repository_bootstrap_v1"}),
    ),
    "dispatch-requests": NamespaceWritePolicy(
        actor="assignmentLinker",
        record_types=frozenset({"dispatch_request"}),
        payload_schema_versions=frozenset({"eiel_dispatch_request_v1"}),
    ),
    "assignments": NamespaceWritePolicy(
        actor="assignmentLinker",
        record_types=frozenset({"assignment"}),
        payload_schema_versions=frozenset({"eiel_worker_assignment_v1"}),
    ),
    "leases": NamespaceWritePolicy(
        actor="scheduler",
        record_types=frozenset({"assignment_lease"}),
        payload_schema_versions=frozenset({"eiel_assignment_lease_v1"}),
    ),
    "worker-results": NamespaceWritePolicy(
        actor="scheduler",
        record_types=frozenset({"worker_result"}),
        payload_schema_versions=frozenset({WORKER_RESULT_SCHEMA_VERSION}),
    ),
}


@dataclass(frozen=True)
class ActorWriteScope:
    """Explicit semantic-write capability bound to one manager namespace."""

    namespace: str
    actor: str
    record_types: frozenset[str]


class ActorRecordStore:
    def __init__(
        self,
        root: str | Path,
        namespace: str,
        *,
        write_scope: ActorWriteScope | None = None,
        reference_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.reference_root = (
            Path(reference_root).expanduser().resolve()
            if reference_root is not None
            else self.root
        )
        self.namespace = str(namespace).strip().lower()
        if not self.namespace or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for ch in self.namespace):
            raise ActorError(f"invalid actor store namespace: {namespace!r}")
        self.write_scope = write_scope
        if write_scope is not None:
            if write_scope.namespace != self.namespace:
                raise ActorError(
                    "actor store write scope namespace mismatch",
                    code="actor_store_write_scope_invalid",
                )
            policy = NAMESPACE_WRITE_POLICIES.get(self.namespace)
            if policy is None:
                raise ActorError(
                    f"namespace {self.namespace} has no declared semantic writer policy",
                    code="actor_store_write_scope_invalid",
                )
            if write_scope.actor != policy.actor or not write_scope.record_types or not write_scope.record_types.issubset(policy.record_types):
                raise ActorError(
                    f"write scope is not authorized for namespace {self.namespace}",
                    code="actor_store_write_scope_invalid",
                )
        self.base = self.root / self.namespace
        self.records_dir = self.base / "records"
        self.history_dir = self.base / "history"
        self.index_rows_dir = self.base / "index-rows"
        self.index_path = self.base / "index.json"
        self.lock_path = self.base / ".lock"

    def _validate_namespace_identity(
        self,
        record: Mapping[str, Any],
        *,
        require_write_scope: bool,
    ) -> None:
        policy = NAMESPACE_WRITE_POLICIES.get(self.namespace)
        if policy is None:
            return
        actor = str(record.get("actor") or "")
        record_type = str(record.get("record_type") or "")
        payload = record.get("payload")
        payload_schema = str(dict(payload).get("schema_version") or "") if isinstance(payload, Mapping) else ""
        problems: list[str] = []
        if actor != policy.actor:
            problems.append(f"actor must be {policy.actor!r}, observed {actor!r}")
        if record_type not in policy.record_types:
            problems.append(
                "record_type must be one of " + ", ".join(sorted(policy.record_types)) + f"; observed {record_type!r}"
            )
        if payload_schema not in policy.payload_schema_versions:
            problems.append(
                "payload schema_version must be one of "
                + ", ".join(sorted(policy.payload_schema_versions))
                + f"; observed {payload_schema!r}"
            )
        if problems:
            raise ActorError(
                f"record identity is not authorized for {self.namespace} namespace",
                code="actor_store_namespace_identity_mismatch",
                problems=problems,
                related_records=[str(record.get("record_id") or "")],
            )
        if require_write_scope:
            scope = self.write_scope
            if scope is None:
                raise ActorError(
                    f"semantic writes to {self.namespace} require a manager-scoped store adapter",
                    code="actor_store_write_scope_required",
                    related_records=[str(record.get("record_id") or "")],
                )
            if actor != scope.actor or record_type not in scope.record_types:
                raise ActorError(
                    f"record is outside the active write scope for {self.namespace}",
                    code="actor_store_write_scope_mismatch",
                    related_records=[str(record.get("record_id") or "")],
                )

    def record_path(self, record_id: str) -> Path:
        return self.records_dir / f"{validate_record_id(record_id)}.json"

    def history_path(self, record_id: str, revision: int) -> Path:
        return self.history_dir / validate_record_id(record_id) / f"{int(revision):06d}.json"

    def history_head_path(self, record_id: str) -> Path:
        return self.history_dir / validate_record_id(record_id) / "HEAD.json"

    def index_row_path(self, record_id: str) -> Path:
        normalized = validate_record_id(record_id)
        shard = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:2]
        return self.index_rows_dir / shard / f"{normalized}.json"

    def exists(self, record_id: str) -> bool:
        record_id = validate_record_id(record_id)
        return self.record_path(record_id).is_file() or (self.history_dir / record_id).is_dir()

    @staticmethod
    def _semantically_same(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        """Return whether two retry records describe the same accepted transition.

        Generated timestamps and the resulting digest may differ on retry.  The
        durable transition identity is the actor/type/status/payload/source/action
        content.  This lets recovery promote an orphan history revision and makes
        an exact semantic retry idempotent instead of manufacturing an immutable
        history conflict.
        """

        keys = ("actor", "record_type", "record_id", "revision", "status", "payload", "source_refs", "action", "note")
        return all(left.get(key) == right.get(key) for key in keys)

    def _read_current_unlocked(self, record_id: str) -> dict[str, Any] | None:
        path = self.record_path(record_id)
        if not path.is_file():
            return None
        record = read_json(path, name=f"{self.namespace} record")
        try:
            loaded = load_actor_record(record)
            self._validate_namespace_identity(loaded, require_write_scope=False)
            return loaded
        except ActorError as exc:
            raise ActorError(
                f"invalid {self.namespace} record {record_id}: {exc}",
                code=exc.code,
                problems=exc.problems,
            ) from exc

    def _write_history_head_locked(self, record: Mapping[str, Any]) -> Path:
        record_id = validate_record_id(str(record.get("record_id") or ""))
        revision = int(record.get("revision") or 0)
        if revision < 1:
            raise ActorError("history head revision must be at least 1", code="actor_store_history_invalid")
        payload: dict[str, Any] = {
            "schema_version": HISTORY_HEAD_SCHEMA_VERSION,
            "record_id": record_id,
            "revision": revision,
            "record_sha256": str(record.get("record_sha256") or ""),
            "snapshot": self.history_path(record_id, revision).name,
        }
        payload["head_sha256"] = canonical_sha256(payload)
        return atomic_write_json(self.history_head_path(record_id), payload)

    def _read_history_head_unlocked(self, record_id: str) -> dict[str, Any] | None:
        normalized = validate_record_id(record_id)
        path = self.history_head_path(normalized)
        if not path.is_file():
            return None
        head = read_json(path, name="record history head")
        require_explicit_schema_version(
            head,
            expected=HISTORY_HEAD_SCHEMA_VERSION,
            label="actor history head",
        )
        expected = head.get("head_sha256")
        material = dict(head)
        material.pop("head_sha256", None)
        if not isinstance(expected, str) or expected != canonical_sha256(material):
            raise ActorError("actor history head digest mismatch", code="actor_store_history_head_invalid")
        revision = int(head.get("revision") or 0)
        if (
            str(head.get("record_id") or "") != normalized
            or revision < 1
            or str(head.get("snapshot") or "") != self.history_path(normalized, revision).name
        ):
            raise ActorError("actor history head identity mismatch", code="actor_store_history_head_invalid")
        return dict(head)

    def _load_history_snapshot_unlocked(self, record_id: str, revision: int) -> dict[str, Any]:
        normalized = validate_record_id(record_id)
        path = self.history_path(normalized, revision)
        if not path.is_file():
            raise ActorError(
                f"immutable history revision is missing for {normalized}@{revision}",
                code="actor_store_history_gap",
            )
        raw = read_json(path, name="record history snapshot")
        try:
            record = load_actor_record(raw)
            self._validate_namespace_identity(record, require_write_scope=False)
        except ActorError as exc:
            raise ActorError(
                f"invalid history for {normalized}@{revision}: {exc}",
                code=exc.code,
                problems=exc.problems,
            ) from exc
        if (
            str(record.get("record_id") or "") != normalized
            or int(record.get("revision") or 0) != int(revision)
            or path.stem != f"{int(revision):06d}"
        ):
            raise ActorError(
                f"history identity mismatch for {normalized}@{revision}",
                code="actor_store_history_invalid",
            )
        return record

    def _latest_history_record_unlocked(self, record_id: str) -> dict[str, Any] | None:
        """Read one latest immutable snapshot, not every historical payload.

        New stores use a constant-size HEAD pointer.  Legacy stores are
        supported by a filename-only scan followed by one snapshot read; the
        next successful write installs HEAD so this migration cost is paid once.
        """

        normalized = validate_record_id(record_id)
        head = self._read_history_head_unlocked(normalized)
        if head is not None:
            latest = self._load_history_snapshot_unlocked(normalized, int(head["revision"]))
            if latest.get("record_sha256") != head.get("record_sha256"):
                raise ActorError(
                    f"history head conflicts with immutable snapshot for {normalized}",
                    code="actor_store_history_head_invalid",
                )
            return latest
        directory = self.history_dir / normalized
        if not directory.is_dir():
            return None
        revisions = [path for path in directory.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json")]
        if not revisions:
            return None
        latest_path = max(revisions, key=lambda path: path.name)
        return self._load_history_snapshot_unlocked(normalized, int(latest_path.stem))

    def _history_records_unlocked(self, record_id: str) -> list[dict[str, Any]]:
        directory = self.history_dir / validate_record_id(record_id)
        if not directory.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        expected = 1
        for path in sorted(directory.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json")):
            raw = read_json(path, name="record history snapshot")
            try:
                record = load_actor_record(raw)
                self._validate_namespace_identity(record, require_write_scope=False)
            except ActorError as exc:
                raise ActorError(
                    f"invalid history for {record_id}: {exc}",
                    code=exc.code,
                    problems=exc.problems,
                ) from exc
            revision = int(record.get("revision") or 0)
            if revision != expected or path.stem != f"{revision:06d}":
                raise ActorError(
                    f"history revision gap or filename mismatch for {record_id}: expected {expected}, observed {revision}",
                    code="actor_store_history_gap",
                )
            if str(record.get("record_id") or "") != record_id:
                raise ActorError(f"history record ID mismatch for {record_id}", code="actor_store_history_invalid")
            if rows:
                predecessor = record.get("previous_record_sha256")
                if predecessor is not None and predecessor != rows[-1].get("record_sha256"):
                    raise ActorError(
                        f"history predecessor link mismatch for {record_id}@{revision}",
                        code="actor_store_history_predecessor_mismatch",
                    )
            rows.append(record)
            expected += 1
        return rows

    def _recover_record_locked(self, record_id: str, *, rebuild_index: bool = False) -> dict[str, Any] | None:
        """Promote contiguous immutable history to the current record.

        History is the write-ahead source.  A crash after the history write but
        before the current/index writes therefore converges on the next read or
        write.  Gaps and conflicts remain fail-closed.
        """

        record_id = validate_record_id(record_id)
        history = self._history_records_unlocked(record_id)
        current = self._read_current_unlocked(record_id)
        if not history:
            if current is not None:
                raise ActorError(
                    f"current record has no immutable history authority for {record_id}",
                    code="actor_store_unproven_current",
                )
            return None
        if current is not None:
            current_revision = int(current.get("revision") or 0)
            if current_revision < 1 or current_revision > len(history):
                raise ActorError(f"current revision is outside history for {record_id}", code="actor_store_history_conflict")
            historical_current = history[current_revision - 1]
            if historical_current.get("record_sha256") != current.get("record_sha256"):
                raise ActorError(f"current record conflicts with immutable history for {record_id}", code="actor_store_history_conflict")
        latest = history[-1]
        if current is None or int(current.get("revision") or 0) < int(latest.get("revision") or 0):
            atomic_write_json(self.record_path(record_id), latest)
            current = latest
        if rebuild_index and current is not None and self._index_record_mismatch_unlocked(current):
            self.rebuild_index_locked(recover=False)
        return current

    def _recover_record_bounded_locked(
        self,
        record_id: str,
        *,
        repair_index_row: bool = False,
    ) -> dict[str, Any] | None:
        """Recover one torn current record from a constant-size history head.

        Full continuity validation remains the explicit audit/recovery surface.
        Routine reads compare only current state with the latest immutable
        snapshot selected by HEAD (or one legacy filename scan during upgrade).
        """

        normalized = validate_record_id(record_id)
        current = self._read_current_unlocked(normalized)
        latest = self._latest_history_record_unlocked(normalized)
        if latest is None:
            if current is not None:
                raise ActorError(
                    f"current record has no immutable history authority for {normalized}",
                    code="actor_store_unproven_current",
                )
            return None
        latest_revision = int(latest.get("revision") or 0)
        if current is not None:
            current_revision = int(current.get("revision") or 0)
            if current_revision > latest_revision:
                raise ActorError(
                    f"current revision is ahead of immutable history for {normalized}",
                    code="actor_store_history_conflict",
                )
            current_snapshot = self._load_history_snapshot_unlocked(normalized, current_revision)
            if current_snapshot.get("record_sha256") != current.get("record_sha256"):
                raise ActorError(
                    f"current record conflicts with immutable history for {normalized}",
                    code="actor_store_history_conflict",
                )
        if current is None or int(current.get("revision") or 0) < latest_revision:
            atomic_write_json(self.record_path(normalized), latest)
            current = latest
        # Install/refresh the bounded head during a mutating recovery/write.
        self._write_history_head_locked(latest)
        if repair_index_row and current is not None:
            self._upsert_index_record_locked(current)
        return current

    def _index_record_mismatch_unlocked(self, record: Mapping[str, Any]) -> bool:
        """Return whether the durable index omits or misstates one current record."""

        if not self.index_path.is_file():
            return True
        index = self._read_index_unlocked()
        record_id = str(record.get("record_id") or "")
        for row in list(index.get("records") or []):
            if str(dict(row or {}).get("record_id") or "") != record_id:
                continue
            return not (
                int(dict(row or {}).get("revision") or 0) == int(record.get("revision") or 0)
                and str(dict(row or {}).get("record_sha256") or "") == str(record.get("record_sha256") or "")
            )
        return True

    def _read_legacy_index_unlocked(self, index: Mapping[str, Any]) -> dict[str, Any]:
        require_explicit_schema_version(
            index,
            expected=INDEX_SCHEMA_VERSION,
            label="actor record index",
        )
        if str(index.get("namespace") or "") != self.namespace:
            raise ActorError(
                f"actor record index namespace mismatch: expected {self.namespace}",
                code="actor_index_malformed",
            )
        records = index.get("records")
        if not isinstance(records, list) or any(not isinstance(row, Mapping) for row in records):
            raise ActorError(
                "actor record index records must be an array of objects",
                code="actor_index_malformed",
            )
        if int(index.get("record_count", -1)) != len(records):
            raise ActorError(
                "actor record index record_count mismatch",
                code="actor_index_malformed",
            )
        expected = index.get("index_sha256")
        material = dict(index)
        material.pop("index_sha256", None)
        if not isinstance(expected, str) or expected != canonical_sha256(material):
            raise ActorError(
                "actor record index digest mismatch",
                code="actor_index_digest_mismatch",
            )
        return dict(index)

    def _read_index_manifest_unlocked(self, index: Mapping[str, Any]) -> dict[str, Any]:
        require_explicit_schema_version(
            index,
            expected=INDEX_MANIFEST_SCHEMA_VERSION,
            label="actor record index manifest",
        )
        if set(index) != {
            "schema_version", "namespace", "record_count", "shard_count",
            "row_schema_version", "layout", "index_sha256",
        }:
            raise ActorError("actor record index manifest is malformed", code="actor_index_malformed")
        if (
            str(index.get("namespace") or "") != self.namespace
            or int(index.get("shard_count") or 0) != INDEX_SHARD_COUNT
            or str(index.get("row_schema_version") or "") != INDEX_ROW_SCHEMA_VERSION
            or str(index.get("layout") or "") != "sha256-prefix-2"
            or int(index.get("record_count") or 0) < 0
        ):
            raise ActorError("actor record index manifest is malformed", code="actor_index_malformed")
        expected = index.get("index_sha256")
        material = dict(index)
        material.pop("index_sha256", None)
        if not isinstance(expected, str) or expected != canonical_sha256(material):
            raise ActorError("actor record index digest mismatch", code="actor_index_digest_mismatch")
        return dict(index)

    def _read_index_row_unlocked(self, path: Path) -> dict[str, Any]:
        row = read_json(path, name="actor record index row")
        require_explicit_schema_version(
            row,
            expected=INDEX_ROW_SCHEMA_VERSION,
            label="actor record index row",
        )
        expected = row.get("row_sha256")
        material = dict(row)
        material.pop("row_sha256", None)
        if not isinstance(expected, str) or expected != canonical_sha256(material):
            raise ActorError("actor record index row digest mismatch", code="actor_index_digest_mismatch")
        record_id = validate_record_id(str(row.get("record_id") or ""))
        if path.resolve() != self.index_row_path(record_id).resolve():
            raise ActorError("actor record index row path mismatch", code="actor_index_malformed")
        return {key: value for key, value in row.items() if key not in {"schema_version", "row_sha256"}}

    def _logical_index_from_manifest_unlocked(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        validated = self._read_index_manifest_unlocked(manifest)
        paths = sorted(self.index_rows_dir.glob("*/*.json")) if self.index_rows_dir.exists() else []
        rows = [self._read_index_row_unlocked(path) for path in paths]
        rows.sort(key=lambda row: str(row.get("record_id") or ""))
        if int(validated.get("record_count") or 0) != len(rows):
            raise ActorError("actor record index manifest count mismatch", code="actor_index_malformed")
        payload: dict[str, Any] = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "namespace": self.namespace,
            "record_count": len(rows),
            "records": rows,
        }
        payload["index_sha256"] = canonical_sha256(payload)
        return payload

    def _read_index_storage_unlocked(self) -> tuple[str, dict[str, Any]]:
        if not self.index_path.is_file():
            raise ActorError(f"{self.namespace} index is missing", code="actor_index_missing")
        try:
            raw = read_json(self.index_path, name=f"{self.namespace} index")
        except ActorError as exc:
            raise ActorError(
                f"malformed {self.namespace} index: {exc}",
                code="actor_index_malformed",
            ) from exc
        observed = require_explicit_schema_version(
            raw,
            expected=INDEX_MANIFEST_SCHEMA_VERSION,
            supported=(INDEX_SCHEMA_VERSION,),
            label="actor record index",
        )
        if observed == INDEX_SCHEMA_VERSION:
            return observed, self._read_legacy_index_unlocked(raw)
        return observed, self._read_index_manifest_unlocked(raw)

    def _read_index_unlocked(self) -> dict[str, Any]:
        """Return the logical v1 index view over legacy or sharded storage.

        This is an explicit audit/export operation and may enumerate all index
        rows.  Ordinary record writes update exactly one shard row plus the
        constant-size manifest and never build this aggregate view.
        """

        kind, value = self._read_index_storage_unlocked()
        if kind == INDEX_SCHEMA_VERSION:
            return value
        return self._logical_index_from_manifest_unlocked(value)

    def load_index_read_only(self) -> dict[str, Any]:
        """Public strict loader used by audits and deterministic tests."""

        return self._read_index_unlocked()

    def get_revision_read_only(self, record_id: str, revision: int) -> dict[str, Any]:
        """Load one exact immutable revision without recovery or index writes."""

        normalized = validate_record_id(record_id)
        try:
            normalized_revision = int(revision)
        except (TypeError, ValueError) as exc:
            raise ActorError(
                f"invalid history revision for {normalized}: {revision!r}",
                code="actor_store_history_invalid",
            ) from exc
        if normalized_revision < 1:
            raise ActorError(
                f"history revision must be at least 1 for {normalized}",
                code="actor_store_history_invalid",
            )
        path = self.history_path(normalized, normalized_revision)
        if not path.is_file():
            raise ActorError(
                f"unknown {self.namespace} history revision: {normalized}@{normalized_revision}",
                code="actor_store_history_missing",
                related_records=[normalized],
            )
        raw = read_json(path, name="record history snapshot")
        try:
            record = load_actor_record(raw)
            self._validate_namespace_identity(record, require_write_scope=False)
        except ActorError as exc:
            raise ActorError(
                f"invalid history for {normalized}@{normalized_revision}: {exc}",
                code=exc.code,
                problems=exc.problems,
                related_records=[normalized],
            ) from exc
        if (
            str(record.get("record_id") or "") != normalized
            or int(record.get("revision") or 0) != normalized_revision
            or path.stem != f"{normalized_revision:06d}"
        ):
            raise ActorError(
                f"history identity mismatch for {normalized}@{normalized_revision}",
                code="actor_store_history_invalid",
                related_records=[normalized],
            )
        return record

    def audit_index_read_only(self) -> dict[str, Any]:
        """Exhaustively compare derived index, current records, and history.

        The method is deliberately non-mutating. Purely derived/index or torn
        current-state defects are classified as ``RECOVERABLE_STALE`` and carry
        exact repair actions. Immutable-history gaps or current/history conflicts
        are ``CORRUPT`` and are never laundered into a rebuild recommendation.
        """

        corrupt: list[str] = []
        recoverable: list[str] = []
        repair_actions: list[dict[str, Any]] = []
        current_records: dict[str, dict[str, Any]] = {}
        latest_history: dict[str, dict[str, Any]] = {}

        current_ids = (
            {path.stem for path in self.records_dir.glob("*.json")}
            if self.records_dir.exists()
            else set()
        )
        history_ids = (
            {path.name for path in self.history_dir.iterdir() if path.is_dir()}
            if self.history_dir.exists()
            else set()
        )
        record_ids = sorted(current_ids | history_ids)

        for record_id in record_ids:
            try:
                current = self._read_current_unlocked(record_id)
            except ActorError as exc:
                corrupt.append(f"current record {record_id} is invalid: {exc}")
                current = None
            try:
                history = self._history_records_unlocked(record_id)
            except ActorError as exc:
                corrupt.append(f"immutable history {record_id} is invalid: {exc}")
                history = []

            if current is not None:
                current_records[record_id] = current
            if history:
                latest_history[record_id] = history[-1]

            if current is None and history:
                recoverable.append(
                    f"current record {record_id} is missing while contiguous immutable history remains"
                )
                repair_actions.append({
                    "action": "promote_latest_history",
                    "record_id": record_id,
                    "description": "Promote the latest contiguous immutable revision to current state, then rebuild the index.",
                })
                continue
            if current is not None and not history:
                corrupt.append(f"current record {record_id} has no immutable history")
                continue
            if current is None or not history:
                continue

            current_revision = int(current.get("revision") or 0)
            if current_revision > len(history) or current_revision < 1:
                corrupt.append(f"current revision for {record_id} is outside immutable history")
                continue
            historical_current = history[current_revision - 1]
            if historical_current.get("record_sha256") != current.get("record_sha256"):
                corrupt.append(f"current record {record_id} conflicts with immutable history")
                continue
            latest = history[-1]
            if current_revision < int(latest.get("revision") or 0):
                recoverable.append(
                    f"current record {record_id} is behind latest immutable history revision"
                )
                repair_actions.append({
                    "action": "promote_latest_history",
                    "record_id": record_id,
                    "description": "Promote the latest contiguous immutable revision to current state, then rebuild the index.",
                })

        index: dict[str, Any] | None = None
        if self.index_path.is_file():
            try:
                index = self._read_index_unlocked()
            except ActorError as exc:
                recoverable.append(f"derived index is invalid: {exc}")
                repair_actions.append({
                    "action": "rebuild_index",
                    "namespace": self.namespace,
                    "description": "Explicitly rebuild the derived index from validated current records after resolving any corrupt record/history findings.",
                })
        elif record_ids:
            recoverable.append("derived index is missing")
            repair_actions.append({
                "action": "rebuild_index",
                "namespace": self.namespace,
                "description": "Explicitly rebuild the derived index from validated current records.",
            })

        if index is not None:
            rows = [dict(row) for row in list(index.get("records") or [])]
            indexed_ids: list[str] = []
            for position, row in enumerate(rows):
                record_id = str(row.get("record_id") or "")
                if not record_id:
                    recoverable.append(f"derived index row {position} has no record_id")
                    continue
                indexed_ids.append(record_id)
                expected_path = self.record_path(record_id)
                raw_path = str(row.get("path") or "")
                try:
                    observed_path = Path(raw_path).expanduser().resolve()
                except (OSError, RuntimeError, ValueError):
                    observed_path = Path("/")
                try:
                    observed_path.relative_to(self.records_dir.resolve())
                    confined = True
                except ValueError:
                    confined = False
                if not confined or observed_path != expected_path.resolve():
                    recoverable.append(
                        f"derived index row {record_id} has a non-canonical or unconfined path"
                    )
                current = current_records.get(record_id)
                if current is None:
                    recoverable.append(f"derived index row {record_id} has no matching current record")
                    continue
                expected_row = self._index_row(current, path=expected_path)
                for field in (
                    "record_id", "record_type", "actor", "status", "revision",
                    "updated_at_utc", "record_sha256", "path", "errors",
                ):
                    if row.get(field) != expected_row.get(field):
                        recoverable.append(
                            f"derived index row {record_id} field {field} does not match current record"
                        )
            duplicates = sorted({value for value in indexed_ids if indexed_ids.count(value) > 1})
            for record_id in duplicates:
                recoverable.append(f"derived index contains duplicate record_id {record_id}")
            indexed_set = set(indexed_ids)
            current_set = set(current_records)
            for record_id in sorted(current_set - indexed_set):
                recoverable.append(f"derived index omits current record {record_id}")
            for record_id in sorted(indexed_set - current_set):
                # Already reported above, but retain one concise set-level edge.
                recoverable.append(f"derived index contains ghost record {record_id}")
            if recoverable and not any(
                action.get("action") == "rebuild_index" for action in repair_actions
            ):
                repair_actions.append({
                    "action": "rebuild_index",
                    "namespace": self.namespace,
                    "description": "Explicitly rebuild the derived index from validated current records.",
                })

        classification = (
            STORE_AUDIT_CORRUPT
            if corrupt
            else STORE_AUDIT_RECOVERABLE_STALE
            if recoverable
            else STORE_AUDIT_VALID
        )
        return {
            "schema_version": "eiel_actor_store_integrity_audit_v1",
            "namespace": self.namespace,
            "classification": classification,
            "corrupt_findings": list(dict.fromkeys(corrupt)),
            "recoverable_findings": list(dict.fromkeys(recoverable)),
            "repair_actions": [
                dict(value)
                for value in {
                    canonical_sha256(value): value for value in repair_actions
                }.values()
            ],
            "record_count": len(record_ids),
            "current_record_count": len(current_records),
            "history_record_count": len(latest_history),
            "index_present": self.index_path.is_file(),
            "index_record_count": len(list(index.get("records") or [])) if index else 0,
        }


    @staticmethod
    def _index_row(record: Mapping[str, Any], *, path: Path) -> dict[str, Any]:
        errors = verify_record(record)
        return {
            "record_id": record.get("record_id"),
            "record_type": record.get("record_type"),
            "actor": record.get("actor"),
            "status": "INVALID" if errors else record.get("status"),
            "revision": record.get("revision"),
            "updated_at_utc": record.get("updated_at_utc"),
            "record_sha256": record.get("record_sha256"),
            "path": str(path),
            "errors": errors,
        }

    def _write_index_row_locked(self, row: Mapping[str, Any]) -> Path:
        record_id = validate_record_id(str(row.get("record_id") or ""))
        payload: dict[str, Any] = {"schema_version": INDEX_ROW_SCHEMA_VERSION, **dict(row)}
        payload["row_sha256"] = canonical_sha256(payload)
        return atomic_write_json(self.index_row_path(record_id), payload)

    def _write_index_manifest_locked(self, record_count: int) -> Path:
        payload: dict[str, Any] = {
            "schema_version": INDEX_MANIFEST_SCHEMA_VERSION,
            "namespace": self.namespace,
            "record_count": int(record_count),
            "shard_count": INDEX_SHARD_COUNT,
            "row_schema_version": INDEX_ROW_SCHEMA_VERSION,
            "layout": "sha256-prefix-2",
        }
        payload["index_sha256"] = canonical_sha256(payload)
        return atomic_write_json(self.index_path, payload)

    def _replace_sharded_index_locked(self, rows: Sequence[Mapping[str, Any]]) -> Path:
        if self.index_rows_dir.exists():
            shutil.rmtree(self.index_rows_dir)
        for row in rows:
            self._write_index_row_locked(row)
        return self._write_index_manifest_locked(len(rows))

    def _ensure_sharded_index_locked(self) -> dict[str, Any]:
        if not self.index_path.is_file():
            rows = self._list_unlocked()
            self._replace_sharded_index_locked(rows)
            return self._read_index_manifest_unlocked(read_json(self.index_path, name=f"{self.namespace} index"))
        kind, value = self._read_index_storage_unlocked()
        if kind == INDEX_SCHEMA_VERSION:
            rows = [dict(row) for row in list(value.get("records") or [])]
            self._replace_sharded_index_locked(rows)
            return self._read_index_manifest_unlocked(read_json(self.index_path, name=f"{self.namespace} index"))
        return value

    def _upsert_index_record_locked(self, record: Mapping[str, Any]) -> Path:
        """Update one sharded row and one constant-size manifest."""

        manifest = self._ensure_sharded_index_locked()
        record_id = validate_record_id(str(record.get("record_id") or ""))
        row_path = self.index_row_path(record_id)
        existed = row_path.is_file()
        self._write_index_row_locked(self._index_row(record, path=self.record_path(record_id)))
        count = int(manifest.get("record_count") or 0) + (0 if existed else 1)
        return self._write_index_manifest_locked(count)

    def get(self, record_id: str) -> dict[str, Any]:
        """Read one current record without mutating or repairing storage.

        Recovery is an explicit operation.  A read must never rewrite current
        state, a history head, or an index merely because a crash left them
        stale.  Call ``recover_record`` when mutation is intentionally desired.
        """

        return self.get_read_only(record_id)

    def get_recovering(self, record_id: str) -> dict[str, Any]:
        """Compatibility seam for explicit bounded single-record recovery."""

        record_id = validate_record_id(record_id)
        with advisory_lock(self.lock_path):
            record = self._recover_record_bounded_locked(record_id, repair_index_row=False)
            if record is None:
                raise ActorError(f"unknown {self.namespace} record: {record_id}")
            return record

    def get_current(self, record_id: str) -> dict[str, Any]:
        """Strict current-state read with no recovery or index mutation."""

        return self.get_read_only(record_id)

    def recover_record(self, record_id: str) -> dict[str, Any]:
        """Explicitly run exhaustive history recovery for one record."""

        normalized = validate_record_id(record_id)
        with advisory_lock(self.lock_path):
            record = self._recover_record_locked(normalized, rebuild_index=False)
            if record is None:
                raise ActorError(f"unknown {self.namespace} record: {normalized}")
            self._write_history_head_locked(record)
            self._upsert_index_record_locked(record)
            return record

    def get_read_only(self, record_id: str) -> dict[str, Any]:
        """Read one authority-bearing current record without mutating storage.

        A current-only JSON file is not authority.  It may be useful forensic
        material after a broken copy or interrupted import, but ordinary
        projections and decisions must be backed by an exact immutable history
        snapshot.
        """

        record_id = validate_record_id(record_id)
        record = self._read_current_unlocked(record_id)
        if record is None:
            raise ActorError(f"unknown {self.namespace} record: {record_id}")
        latest = self._latest_history_record_unlocked(record_id)
        if latest is None:
            raise ActorError(
                f"current record has no immutable history authority for {record_id}",
                code="actor_store_unproven_current",
            )
        revision = int(record.get("revision") or 0)
        historical = self._load_history_snapshot_unlocked(record_id, revision)
        if historical.get("record_sha256") != record.get("record_sha256"):
            raise ActorError(
                f"current record conflicts with immutable history for {record_id}",
                code="actor_store_history_conflict",
            )
        return record

    def get_unproven_current_read_only(self, record_id: str) -> dict[str, Any]:
        """Forensic-only current read that never grants lifecycle authority."""

        record_id = validate_record_id(record_id)
        record = self._read_current_unlocked(record_id)
        if record is None:
            raise ActorError(f"unknown {self.namespace} record: {record_id}")
        return record

    def put(self, record: Mapping[str, Any], *, allow_update: bool = False) -> Path:
        if self.namespace in NAMESPACE_WRITE_POLICIES:
            self._validate_namespace_identity(record, require_write_scope=True)
        errors = verify_record(record)
        if errors:
            raise ActorError("refusing to store invalid actor record: " + "; ".join(errors))
        from tools.actors.reference_integrity import validate_record_source_references

        source_validation = validate_record_source_references(
            record,
            state_root=self.reference_root,
        )
        if not source_validation["valid"]:
            raise ActorError(
                "refusing to store record with unresolved or contradictory source references",
                code="actor_source_reference_invalid",
                problems=list(source_validation["errors"]),
                related_records=[str(record.get("record_id") or "")],
            )
        record_id = validate_record_id(str(record["record_id"]))
        revision = int(record.get("revision") or 0)
        if revision < 1:
            raise ActorError("record revision must be at least 1")
        with advisory_lock(self.lock_path):
            current_path = self.record_path(record_id)
            existing = self._recover_record_bounded_locked(record_id, repair_index_row=False)
            if existing is not None and revision == int(existing.get("revision") or 0):
                if existing.get("record_sha256") == record.get("record_sha256") or self._semantically_same(existing, record):
                    self._upsert_index_record_locked(existing)
                    return current_path
            if existing is not None and not allow_update:
                if existing.get("record_sha256") == record.get("record_sha256"):
                    return current_path
                if (
                    existing.get("record_type") == record.get("record_type")
                    and existing.get("status") == record.get("status")
                    and existing.get("payload") == record.get("payload")
                    and existing.get("source_refs") == record.get("source_refs")
                ):
                    return current_path
                raise ActorError(f"record already exists: {record_id}")
            if existing is not None and allow_update:
                expected_revision = int(existing.get("revision") or 0) + 1
                if revision != expected_revision:
                    raise ActorError(f"revision mismatch for {record_id}: expected {expected_revision}, got {revision}")
            elif existing is None and revision != 1:
                raise ActorError(f"revision mismatch for new {record_id}: expected 1, got {revision}")

            history_path = self.history_path(record_id, revision)
            if history_path.exists():
                existing_history = read_json(history_path, name="record history snapshot")
                if existing_history.get("record_sha256") != record.get("record_sha256"):
                    if self._semantically_same(existing_history, record):
                        self._write_history_head_locked(existing_history)
                        atomic_write_json(current_path, existing_history)
                        self._upsert_index_record_locked(existing_history)
                        return current_path
                    raise ActorError(f"immutable history conflict: {history_path}")
            else:
                atomic_write_json(history_path, record)
            # History snapshot and bounded HEAD are the write-ahead authority.
            # Current state and the derived index may be repaired from them.
            self._write_history_head_locked(record)
            atomic_write_json(current_path, record)
            self._upsert_index_record_locked(record)
            return current_path

    def _list_unlocked(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.records_dir.exists():
            return rows
        for path in sorted(self.records_dir.glob("*.json")):
            try:
                record = read_json(path, name=f"{self.namespace} record")
                errors = verify_record(record)
            except ActorError as exc:
                rows.append({"record_id": path.stem, "status": "INVALID", "error": str(exc), "path": str(path)})
                continue
            rows.append({
                "record_id": record.get("record_id"),
                "record_type": record.get("record_type"),
                "actor": record.get("actor"),
                "status": "INVALID" if errors else record.get("status"),
                "revision": record.get("revision"),
                "updated_at_utc": record.get("updated_at_utc"),
                "record_sha256": record.get("record_sha256"),
                "path": str(path),
                "errors": errors,
            })
        return rows

    def list(self) -> list[dict[str, Any]]:
        """List current records without recovering history or rewriting indexes."""

        with advisory_lock(self.lock_path):
            if self.index_path.is_file():
                # Validate the storage discriminator before returning state.  A
                # future or malformed derived index is not silently replaced by
                # a read operation.
                self._read_index_storage_unlocked()
            return self._list_unlocked()

    def list_read_only(self) -> list[dict[str, Any]]:
        """List current records with bounded torn-state diagnostics.

        At most the current snapshot, latest immutable snapshot, and immutable
        snapshot matching the current revision are parsed per record. Exhaustive
        continuity/index validation remains ``audit_index_read_only``.
        """

        record_ids = {path.stem for path in self.records_dir.glob("*.json")} if self.records_dir.exists() else set()
        if self.history_dir.exists():
            record_ids.update(path.name for path in self.history_dir.iterdir() if path.is_dir())
        rows: list[dict[str, Any]] = []
        for record_id in sorted(record_ids):
            try:
                record = self._read_current_unlocked(record_id)
                latest = self._latest_history_record_unlocked(record_id)
            except ActorError as exc:
                rows.append({
                    "record_id": record_id,
                    "status": "INVALID",
                    "integrity_classification": STORE_AUDIT_CORRUPT,
                    "error": str(exc),
                    "path": str(self.record_path(record_id)),
                })
                continue
            if record is None:
                if latest is not None:
                    rows.append({
                        "record_id": record_id,
                        "record_type": latest.get("record_type"),
                        "actor": latest.get("actor"),
                        "status": STORE_AUDIT_RECOVERABLE_STALE,
                        "integrity_classification": STORE_AUDIT_RECOVERABLE_STALE,
                        "revision": latest.get("revision"),
                        "record_sha256": latest.get("record_sha256"),
                        "error": "current record missing while immutable history remains",
                        "repair_action": "promote_latest_history",
                        "path": str(self.record_path(record_id)),
                    })
                else:
                    rows.append({
                        "record_id": record_id,
                        "status": "INVALID",
                        "integrity_classification": STORE_AUDIT_CORRUPT,
                        "error": "record has neither current state nor immutable history",
                        "path": str(self.record_path(record_id)),
                    })
                continue
            if latest is None:
                rows.append({
                    "record_id": record_id,
                    "status": "INVALID",
                    "integrity_classification": STORE_AUDIT_CORRUPT,
                    "error": "current record has no immutable history",
                    "path": str(self.record_path(record_id)),
                })
                continue
            current_revision = int(record.get("revision") or 0)
            latest_revision = int(latest.get("revision") or 0)
            if current_revision < 1 or current_revision > latest_revision:
                rows.append({
                    "record_id": record_id,
                    "status": "INVALID",
                    "integrity_classification": STORE_AUDIT_CORRUPT,
                    "error": "current revision is outside immutable history",
                    "path": str(self.record_path(record_id)),
                })
                continue
            try:
                historical_current = self._load_history_snapshot_unlocked(record_id, current_revision)
            except ActorError as exc:
                rows.append({
                    "record_id": record_id,
                    "status": "INVALID",
                    "integrity_classification": STORE_AUDIT_CORRUPT,
                    "error": str(exc),
                    "path": str(self.record_path(record_id)),
                })
                continue
            if historical_current.get("record_sha256") != record.get("record_sha256"):
                rows.append({
                    "record_id": record_id,
                    "status": "INVALID",
                    "integrity_classification": STORE_AUDIT_CORRUPT,
                    "error": "current record conflicts with immutable history",
                    "path": str(self.record_path(record_id)),
                })
                continue
            behind = current_revision < latest_revision
            rows.append({
                "record_id": record.get("record_id"),
                "record_type": record.get("record_type"),
                "actor": record.get("actor"),
                "status": STORE_AUDIT_RECOVERABLE_STALE if behind else record.get("status"),
                "record_status": record.get("status"),
                "integrity_classification": STORE_AUDIT_RECOVERABLE_STALE if behind else STORE_AUDIT_VALID,
                "revision": record.get("revision"),
                "latest_history_revision": latest.get("revision"),
                "updated_at_utc": record.get("updated_at_utc"),
                "record_sha256": record.get("record_sha256"),
                "path": str(self.record_path(record_id)),
                "errors": [],
                **({
                    "error": "current record is behind latest immutable history",
                    "repair_action": "promote_latest_history",
                } if behind else {}),
            })
        return rows

    def _history_summary_unlocked(self, record_id: str) -> list[dict[str, Any]]:
        directory = self.history_dir / record_id
        if not directory.exists():
            if self.record_path(record_id).is_file():
                raise ActorError(f"history missing for existing record: {record_id}")
            raise ActorError(f"unknown {self.namespace} record: {record_id}")
        records = self._history_records_unlocked(record_id)
        return [{
            "revision": record.get("revision"),
            "status": record.get("status"),
            "updated_at_utc": record.get("updated_at_utc"),
            "record_sha256": record.get("record_sha256"),
            "path": str(self.history_path(record_id, int(record.get("revision") or 0))),
            "errors": [],
        } for record in records]

    def history(self, record_id: str) -> list[dict[str, Any]]:
        record_id = validate_record_id(record_id)
        with advisory_lock(self.lock_path):
            return self._history_summary_unlocked(record_id)

    def history_read_only(self, record_id: str) -> list[dict[str, Any]]:
        """Read immutable history without creating a lock or repairing state."""

        return self._history_summary_unlocked(validate_record_id(record_id))

    def _write_index_locked(self, rows: list[dict[str, Any]]) -> Path:
        return self._replace_sharded_index_locked(rows)

    def rebuild_index_locked(self, *, recover: bool = True) -> Path:
        if recover:
            record_ids = {path.stem for path in self.records_dir.glob("*.json")} if self.records_dir.exists() else set()
            if self.history_dir.exists():
                record_ids.update(path.name for path in self.history_dir.iterdir() if path.is_dir())
            for record_id in sorted(record_ids):
                recovered = self._recover_record_locked(record_id, rebuild_index=False)
                if recovered is not None:
                    self._write_history_head_locked(recovered)
        return self._write_index_locked(self._list_unlocked())

    def rebuild_index(self) -> Path:
        with advisory_lock(self.lock_path):
            return self.rebuild_index_locked()

    def files_for_record(self, record_id: str) -> Iterable[tuple[str, Path]]:
        self.get(record_id)
        current = self.record_path(record_id)
        yield f"{self.namespace}/{current.name}", current
        for row in self.history(record_id):
            path = Path(str(row["path"]))
            yield f"{self.namespace}/history/{path.name}", path
        if self.index_path.is_file():
            yield f"{self.namespace}/index.json", self.index_path
        row = self.index_row_path(record_id)
        if row.is_file():
            yield f"{self.namespace}/index-rows/{row.parent.name}/{row.name}", row


def scoped_actor_store(
    root: str | Path,
    namespace: str,
    *,
    actor: str,
    record_types: Sequence[str],
    reference_root: str | Path | None = None,
) -> ActorRecordStore:
    """Return an explicit semantic-writer adapter for one declared manager."""

    normalized_namespace = str(namespace).strip().lower()
    scope = ActorWriteScope(
        namespace=normalized_namespace,
        actor=str(actor).strip(),
        record_types=frozenset(str(value).strip() for value in record_types if str(value).strip()),
    )
    return ActorRecordStore(
        root,
        normalized_namespace,
        write_scope=scope,
        reference_root=reference_root,
    )
