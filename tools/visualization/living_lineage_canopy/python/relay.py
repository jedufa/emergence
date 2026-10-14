from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .artifact_readers.checkpoint_bundle_v2 import (
    choose_cell,
    projection_paths_for_cell,
    snapshot_boundary,
    snapshot_from_projection,
    snapshot_order_key,
)
from .production_exporter import cross_check_gv2, export_gv2_checkpoint_run
from .provenance import (
    NO_CLAIMS_STATEMENT,
    canonical_json_bytes,
    exporter_git_commit,
    output_hash,
    sha256_file,
)
from .schema import peak_living_count, validate_playback


RELAY_SCHEMA_VERSION = "llc_near_live_playback_relay_v0_manifest"
RELAY_STATUS_SCHEMA_VERSION = "llc_near_live_playback_relay_v0_status"
RELAY_RECEIPT_SCHEMA_VERSION = "llc_near_live_playback_relay_v0_receipt"
STAGED_SOURCE_SCHEMA_VERSION = "llc_near_live_staged_playback_source_v0"
RELAY_VERSION = "LLC-NEAR-LIVE-PLAYBACK-RELAY-V0"
DEFAULT_POLL_SECONDS = 2.0

RelayState = Literal[
    "initializing",
    "watching",
    "updating",
    "paused",
    "source_stale",
    "source_complete",
    "source_interrupted",
    "integrity_error",
    "stopped",
]


@dataclass(frozen=True)
class FileFingerprint:
    path: Path
    size: int
    mtime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> "FileFingerprint":
        stat = path.stat()
        return cls(path=path, size=stat.st_size, mtime_ns=stat.st_mtime_ns)

    def stable_with(self, other: "FileFingerprint") -> bool:
        return self.size == other.size and self.mtime_ns == other.mtime_ns


@dataclass(frozen=True)
class SourceFile:
    path: Path
    sha256: str

    def to_json(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class SourceUpdate:
    source_kind: str
    source_run_id: str
    source_root: Path
    boundary: int
    source_state: RelayState
    source_classification: str
    semantics_mode: str
    branch_semantics: str
    playback: dict[str, Any]
    provenance: dict[str, Any]
    cross_check: dict[str, Any]
    source_files: tuple[SourceFile, ...]
    warnings: tuple[str, ...] = ()


class RelayIntegrityError(RuntimeError):
    pass


class SourceNotReady(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        source_state: RelayState | None = None,
    ) -> None:
        super().__init__(message)
        self.source_state = source_state


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, data: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(data)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        tmp = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def _source_files(paths: list[Path] | tuple[Path, ...]) -> tuple[SourceFile, ...]:
    return tuple(
        SourceFile(path=path.resolve(), sha256=sha256_file(path))
        for path in sorted(paths, key=lambda item: str(item))
    )


def _copy_playback_through_boundary(
    playback: dict[str, Any], boundary: int
) -> dict[str, Any]:
    metadata = dict(playback["metadata"])
    metadata["last_tick"] = min(boundary, int(metadata["last_tick"]))
    agents = []
    for agent in playback["agents"]:
        if int(agent["birth_tick"]) > boundary:
            continue
        row = dict(agent)
        death_tick = row.get("death_tick")
        if death_tick is not None and int(death_tick) > boundary:
            row["death_tick"] = None
            row["generation_died"] = None
            row.pop("death_interval", None)
            row.pop("death_tick_source", None)
            row.pop("source_death_event_sequence", None)
        agents.append(row)
    lineages = []
    living_lineage_ids = {agent["lineage_id"] for agent in agents}
    for lineage in playback["lineages"]:
        row = dict(lineage)
        if (
            row["lineage_id"] in living_lineage_ids
            or int(row["creation_tick"]) <= boundary
        ):
            extinct = row.get("extinction_tick")
            if extinct is not None and int(extinct) > boundary:
                row["extinction_tick"] = None
            lineages.append(row)
    branches = []
    living_branch_ids = {agent["branch_id"] for agent in agents}
    for branch in playback["branches"]:
        row = dict(branch)
        if (
            row["branch_id"] in living_branch_ids
            or int(row["creation_tick"]) <= boundary
        ):
            extinct = row.get("extinction_tick")
            if extinct is not None and int(extinct) > boundary:
                row["extinction_tick"] = None
            branches.append(row)
    sliced = {
        "metadata": metadata,
        "lineages": lineages,
        "branches": branches,
        "agents": agents,
    }
    validate_playback(sliced)
    return sliced


def build_staged_playback_source(
    *,
    playback_path: Path,
    provenance_path: Path,
    out: Path,
    boundaries: list[int],
    source_state: RelayState = "source_interrupted",
) -> dict[str, Any]:
    playback = load_json(playback_path)
    provenance = load_json(provenance_path)
    validate_playback(playback)
    if not boundaries:
        raise ValueError("at least one boundary is required")
    unique_boundaries = sorted({int(boundary) for boundary in boundaries})
    if unique_boundaries[-1] > int(playback["metadata"]["last_tick"]):
        raise ValueError("stage boundary exceeds playback last_tick")
    out.mkdir(parents=True, exist_ok=True)
    stages = []
    previous_boundary: int | None = None
    for index, boundary in enumerate(unique_boundaries, start=1):
        stage_dir = out / "stages" / f"stage_{index:06d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        sliced = _copy_playback_through_boundary(playback, boundary)
        stage_provenance = dict(provenance)
        stage_provenance["source_root"] = str(out.resolve())
        stage_provenance["source_root_supplied"] = str(out)
        stage_provenance["source_root_resolved"] = str(out.resolve())
        stage_provenance["source_files"] = []
        stage_provenance["output_sha256"] = output_hash(sliced)
        stage_provenance["generated_at"] = utc_now()
        stage_provenance["command"] = "llc near-live staged playback fixture"
        playback_file = stage_dir / "playback_snapshot.json"
        provenance_file = stage_dir / "provenance.json"
        cross_check_file = stage_dir / "cross_check.json"
        playback_hash = write_json_atomic(playback_file, sliced)
        stage_provenance["output_sha256"] = playback_hash
        provenance_hash = write_json_atomic(provenance_file, stage_provenance)
        living = sum(
            1
            for agent in sliced["agents"]
            if agent.get("death_tick") is None or int(agent["death_tick"]) > boundary
        )
        cross_check = {
            "schema_version": "llc_staged_playback_cross_check_v0",
            "ok": True,
            "boundary": boundary,
            "agent_count": len(sliced["agents"]),
            "living_count": living,
            "lineage_count": len(sliced["lineages"]),
            "playback_sha256": playback_hash,
            "provenance_sha256": provenance_hash,
            "source_mode": "staged_normalized_playback_fixture",
            "raw_source_available": False,
        }
        cross_check_hash = write_json_atomic(cross_check_file, cross_check)
        source_files = _source_files([playback_file, provenance_file, cross_check_file])
        stages.append(
            {
                "index": index,
                "boundary": boundary,
                "expected_previous_boundary": previous_boundary,
                "playback": str(playback_file.relative_to(out)),
                "playback_sha256": playback_hash,
                "provenance": str(provenance_file.relative_to(out)),
                "provenance_sha256": provenance_hash,
                "cross_check": str(cross_check_file.relative_to(out)),
                "cross_check_sha256": cross_check_hash,
                "source_files": [item.to_json() for item in source_files],
            }
        )
        previous_boundary = boundary
    manifest = {
        "schema_version": STAGED_SOURCE_SCHEMA_VERSION,
        "source_run_id": playback["metadata"].get(
            "dataset_id", "staged_playback_source"
        ),
        "source_state": source_state,
        "source_classification": playback["metadata"].get("source_classification"),
        "semantics_mode": playback["metadata"].get("semantics_mode"),
        "branch_semantics": playback["metadata"].get("branch_semantics"),
        "final_boundary": unique_boundaries[-1],
        "stage_count": len(stages),
        "created_at": utc_now(),
        "stages": stages,
        "warnings": [
            "Staged normalized-playback fixture reconstructed from an accepted compact export because the raw G88 run tree was intentionally removed after review acceptance.",
            NO_CLAIMS_STATEMENT,
        ],
    }
    write_json_atomic(out / "staged_manifest.json", manifest)
    return manifest


class StagedPlaybackSource:
    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root.resolve()
        self._last_seen: dict[Path, FileFingerprint] = {}

    def scan(self, accepted_boundary: int | None) -> SourceUpdate:
        manifest_path = self.source_root / "staged_manifest.json"
        if not manifest_path.is_file():
            raise SourceNotReady(f"missing staged manifest: {manifest_path}")
        manifest = load_json(manifest_path)
        if manifest.get("schema_version") != STAGED_SOURCE_SCHEMA_VERSION:
            raise RelayIntegrityError("unsupported staged source schema")
        candidates = []
        for stage in manifest.get("stages") or []:
            boundary = int(stage["boundary"])
            if accepted_boundary is not None and boundary <= accepted_boundary:
                continue
            paths = [
                self.source_root / stage["playback"],
                self.source_root / stage["provenance"],
                self.source_root / stage["cross_check"],
            ]
            if not all(path.is_file() for path in paths):
                continue
            if not self._stable(paths):
                continue
            candidates.append((boundary, stage, paths))
        if not candidates:
            raise SourceNotReady("no new stable staged playback boundary")
        boundary, stage, paths = sorted(candidates, key=lambda item: item[0])[0]
        expected_previous = stage.get("expected_previous_boundary")
        if expected_previous != accepted_boundary:
            raise RelayIntegrityError(
                f"staged boundary gap: expected previous {expected_previous}, accepted {accepted_boundary}"
            )
        playback = load_json(paths[0])
        provenance = load_json(paths[1])
        cross_check = load_json(paths[2])
        validate_playback(playback)
        if output_hash(playback) != stage["playback_sha256"]:
            raise RelayIntegrityError(
                f"staged playback hash mismatch at boundary {boundary}"
            )
        if sha256_file(paths[1]) != stage["provenance_sha256"]:
            raise RelayIntegrityError(
                f"staged provenance hash mismatch at boundary {boundary}"
            )
        if sha256_file(paths[2]) != stage["cross_check_sha256"]:
            raise RelayIntegrityError(
                f"staged cross-check hash mismatch at boundary {boundary}"
            )
        if not cross_check.get("ok"):
            raise RelayIntegrityError(
                f"staged cross-check failed at boundary {boundary}"
            )
        return SourceUpdate(
            source_kind="staged_normalized_playback",
            source_run_id=str(manifest["source_run_id"]),
            source_root=self.source_root,
            boundary=boundary,
            source_state=manifest.get("source_state", "source_interrupted"),
            source_classification=str(
                manifest.get("source_classification")
                or playback["metadata"].get("source_classification")
            ),
            semantics_mode=str(
                manifest.get("semantics_mode")
                or playback["metadata"].get("semantics_mode")
            ),
            branch_semantics=str(
                manifest.get("branch_semantics")
                or playback["metadata"].get("branch_semantics")
            ),
            playback=playback,
            provenance=provenance,
            cross_check=cross_check,
            source_files=_source_files(paths),
            warnings=tuple(manifest.get("warnings") or ()),
        )

    def _stable(self, paths: list[Path]) -> bool:
        stable = True
        for path in paths:
            current = FileFingerprint.from_path(path)
            previous = self._last_seen.get(path)
            if previous is None or not current.stable_with(previous):
                stable = False
            self._last_seen[path] = current
        return stable


class Gv2CheckpointSource:
    def __init__(
        self, source_root: Path, *, cell_id: str | None, work_root: Path
    ) -> None:
        self.source_root = source_root.resolve()
        self.cell_id = cell_id
        self.work_root = work_root
        self._last_seen: dict[Path, FileFingerprint] = {}
        self._published_stable_paths: tuple[Path, ...] = ()
        self._published_boundary: int | None = None

    def scan(self, accepted_boundary: int | None) -> SourceUpdate:
        selected_cell = self.cell_id or choose_cell(self.source_root)
        paths = projection_paths_for_cell(self.source_root, selected_cell)
        if not paths:
            terminal_state = self.source_state()
            raise SourceNotReady(
                "no checkpoint projection files found",
                source_state=(
                    terminal_state
                    if terminal_state in {"source_complete", "source_interrupted"}
                    else None
                ),
            )
        source_state = self.source_state()
        stable_paths = [path for path in paths if self._stable_path(path)]
        if (
            source_state in {"source_complete", "source_interrupted"}
            and len(stable_paths) != len(paths)
        ):
            raise SourceNotReady(
                "terminal source has checkpoint projections awaiting stability",
                source_state="watching",
            )
        if len(stable_paths) < 1:
            raise SourceNotReady("checkpoint projections not stable yet")
        stable_path_tuple = tuple(stable_paths)
        if (
            accepted_boundary is not None
            and source_state not in {"source_complete", "source_interrupted"}
            and self._published_boundary == accepted_boundary
            and self._published_stable_paths == stable_path_tuple
        ):
            raise SourceNotReady("no new stable checkpoint boundary")
        snapshots = [snapshot_from_projection(path) for path in stable_paths]
        pairs = sorted(
            zip(snapshots, stable_paths, strict=True),
            key=lambda item: snapshot_order_key(item[0]),
        )
        boundaries = [
            (
                0
                if index == 0 and snapshot.absolute_tick is None
                else snapshot_boundary(snapshot)
            )
            for index, (snapshot, _) in enumerate(pairs)
        ]
        if boundaries != sorted(set(boundaries)):
            raise RelayIntegrityError(
                f"duplicate or regressed checkpoint sequence: {boundaries}"
            )
        boundary = boundaries[-1]
        terminal_state_transition = (
            accepted_boundary is not None
            and boundary == accepted_boundary
            and source_state in {"source_complete", "source_interrupted"}
        )
        if accepted_boundary is not None and accepted_boundary not in boundaries:
            raise RelayIntegrityError(
                f"accepted boundary {accepted_boundary} missing from current source history"
            )
        if (
            accepted_boundary is not None
            and boundary <= accepted_boundary
            and not terminal_state_transition
        ):
            if boundary == accepted_boundary:
                self._published_stable_paths = stable_path_tuple
                self._published_boundary = boundary
            raise SourceNotReady("no new stable checkpoint boundary")
        view_root = self._build_source_view(tuple(path for _, path in pairs))
        playback, provenance, summary = export_gv2_checkpoint_run(
            view_root,
            cell_id=selected_cell,
            generated_at=utc_now(),
            command=f"llc near-live relay export source={self.source_root}",
            snapshots=tuple(snapshot for snapshot, _ in pairs),
        )
        cross_check = cross_check_gv2(
            view_root,
            playback,
            cell_id=selected_cell,
            snapshots=tuple(snapshot for snapshot, _ in pairs),
        )
        if not cross_check.get("ok"):
            raise RelayIntegrityError(
                "GV2 cross-check failed: "
                + "; ".join(cross_check.get("mismatches") or [])
            )
        self._published_stable_paths = stable_path_tuple
        self._published_boundary = boundary
        return SourceUpdate(
            source_kind="gv2_checkpoint_bundle_v2",
            source_run_id=self.source_root.name,
            source_root=self.source_root,
            boundary=boundary,
            source_state=source_state,
            source_classification=summary["classification"],
            semantics_mode=summary["semantics_mode"],
            branch_semantics=playback["metadata"].get(
                "branch_semantics", "lineage_root_projection"
            ),
            playback=playback,
            provenance=provenance,
            cross_check=cross_check,
            source_files=_source_files(tuple(path for _, path in pairs)),
            warnings=tuple(playback["metadata"].get("warnings") or ()),
        )

    def source_state(self) -> RelayState:
        completed = False
        for name, schema_version in (
            ("qrr_run_receipt.json", "eiel_qrr_run_receipt_v1"),
            ("campaign_manifest.json", "eiel_campaign_execution_manifest_v1"),
        ):
            path = self.source_root / name
            if not path.is_file():
                continue
            receipt = load_json(path)
            if receipt.get("schema_version") != schema_version:
                raise RelayIntegrityError(f"unsupported terminal source schema: {path}")
            completed = completed or receipt.get("status") == "completed"
        terminal_path = self.source_root / "canopy_source_terminal.json"
        if terminal_path.is_file():
            terminal = load_json(terminal_path)
            terminal_core = {
                key: value for key, value in terminal.items() if key != "record_sha256"
            }
            terminal_digest = hashlib.sha256(
                json.dumps(
                    terminal_core,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
            ).hexdigest()
            if (
                terminal.get("schema_version") != "eiel_canopy_source_terminal_v1"
                or terminal.get("state") != "source_interrupted"
                or terminal.get("record_sha256") != terminal_digest
            ):
                raise RelayIntegrityError(
                    "Canopy source terminal marker is malformed or digest-mismatched"
                )
            if completed:
                raise RelayIntegrityError(
                    "Canopy source terminal marker contradicts a completed source receipt"
                )
            return "source_interrupted"
        if completed:
            return "source_complete"
        return "watching"

    def _stable_path(self, path: Path) -> bool:
        current = FileFingerprint.from_path(path)
        previous = self._last_seen.get(path)
        self._last_seen[path] = current
        return previous is not None and current.stable_with(previous)

    def _build_source_view(self, stable_paths: tuple[Path, ...]) -> Path:
        view_root = self.work_root / "source_view"
        if view_root.exists():
            shutil.rmtree(view_root)
        for path in stable_paths:
            relative = path.relative_to(self.source_root)
            target = view_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path)
        return view_root


class LineagePlaybackRelay:
    def __init__(
        self,
        *,
        source_root: Path,
        output_root: Path,
        source_kind: str = "auto",
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        cell_id: str | None = None,
        relay_id: str | None = None,
    ) -> None:
        if poll_seconds < 1.0:
            raise ValueError("poll interval must be at least one second")
        self.source_root = source_root.resolve()
        self.output_root = output_root.resolve()
        self.poll_seconds = poll_seconds
        self.relay_id = relay_id
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.receipts_root = self.output_root / "receipts"
        self.work_root = self.output_root / "_work"
        self.receipts_root.mkdir(parents=True, exist_ok=True)
        self.work_root.mkdir(parents=True, exist_ok=True)
        detected_kind = self._detect_source_kind(source_kind)
        self.source_kind = detected_kind
        if detected_kind == "staged_normalized_playback":
            self.source = StagedPlaybackSource(self.source_root)
        elif detected_kind == "gv2_checkpoint_bundle_v2":
            self.source = Gv2CheckpointSource(
                self.source_root, cell_id=cell_id, work_root=self.work_root
            )
        else:
            raise ValueError(f"unsupported source_kind {source_kind!r}")

    def update_once(self) -> dict[str, Any]:
        started = time.perf_counter()
        manifest = self._load_manifest_if_present()
        accepted_boundary = (
            manifest.get("latest_exported_playback_boundary") if manifest else None
        )
        sequence = int(manifest.get("sequence_number", 0)) if manifest else 0
        relay_id = self.relay_id or (
            manifest.get("relay_id")
            if manifest
            else f"llc-relay-{uuid.uuid4().hex[:12]}"
        )
        try:
            self._verify_accepted_source_hashes()
            update = self._scan_source(accepted_boundary)
            terminal_state_transition = (
                accepted_boundary is not None
                and update.boundary == int(accepted_boundary)
                and update.source_state in {"source_complete", "source_interrupted"}
                and manifest is not None
                and manifest.get("source_state") != update.source_state
            )
            if (
                accepted_boundary is not None
                and update.boundary <= int(accepted_boundary)
                and not terminal_state_transition
            ):
                idle_state: RelayState = (
                    manifest["source_state"]
                    if manifest
                    and manifest.get("source_state")
                    in {"source_complete", "source_interrupted"}
                    else "watching"
                )
                status = self._status(
                    relay_id=relay_id,
                    state=idle_state,
                    source_boundary=accepted_boundary,
                    playback_boundary=accepted_boundary,
                    last_error=None,
                    warnings=["No new completed source boundary."],
                    sequence_number=sequence,
                    source_classification=manifest.get("current_source_classification")
                    if manifest
                    else None,
                    semantics_mode=manifest.get("semantics_mode") if manifest else None,
                    branch_semantics=manifest.get("branch_semantics")
                    if manifest
                    else None,
                )
                write_json_atomic(self.output_root / "relay_status.json", status)
                return {
                    "updated": False,
                    "status": status,
                    "duration_ms": (time.perf_counter() - started) * 1000,
                }
            result = self._publish(
                update, relay_id=relay_id, sequence_number=sequence + 1, started=started
            )
            return result
        except SourceNotReady as error:
            state: RelayState = "initializing" if not manifest else "source_stale"
            if error.source_state is not None:
                state = error.source_state
            elif manifest and manifest.get("source_state") in {
                "source_interrupted",
                "source_complete",
            }:
                state = manifest["source_state"]
            status = self._status(
                relay_id=relay_id,
                state=state,
                source_boundary=accepted_boundary,
                playback_boundary=accepted_boundary,
                last_error=None,
                warnings=[str(error)],
                sequence_number=sequence,
                source_classification=manifest.get("current_source_classification")
                if manifest
                else None,
                semantics_mode=manifest.get("semantics_mode") if manifest else None,
                branch_semantics=manifest.get("branch_semantics") if manifest else None,
            )
            write_json_atomic(self.output_root / "relay_status.json", status)
            return {
                "updated": False,
                "status": status,
                "duration_ms": (time.perf_counter() - started) * 1000,
            }
        except Exception as error:
            status = self._status(
                relay_id=relay_id,
                state="integrity_error",
                source_boundary=accepted_boundary,
                playback_boundary=accepted_boundary,
                last_error=str(error),
                warnings=[],
                sequence_number=sequence,
                source_classification=manifest.get("current_source_classification")
                if manifest
                else None,
                semantics_mode=manifest.get("semantics_mode") if manifest else None,
                branch_semantics=manifest.get("branch_semantics") if manifest else None,
            )
            write_json_atomic(self.output_root / "relay_status.json", status)
            return {
                "updated": False,
                "status": status,
                "error": str(error),
                "duration_ms": (time.perf_counter() - started) * 1000,
            }

    def _scan_source(self, accepted_boundary: int | None) -> SourceUpdate:
        """Bound heavy checkpoint decoding to one relay per run group."""

        lock_path = self.output_root.parent / ".checkpoint_scan.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                return self.source.scan(accepted_boundary)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def watch(self, *, max_updates: int | None = None) -> None:
        updates = 0
        while True:
            result = self.update_once()
            if result.get("updated"):
                updates += 1
                print(
                    json.dumps(
                        {
                            "updated": True,
                            "sequence": result["manifest"]["sequence_number"],
                            "boundary": result["manifest"][
                                "latest_exported_playback_boundary"
                            ],
                        },
                        sort_keys=True,
                    )
                )
            else:
                print(
                    json.dumps(
                        {
                            "updated": False,
                            "state": result["status"]["state"],
                            "warning_count": len(
                                result["status"].get("incomplete_artifact_warnings")
                                or []
                            ),
                        },
                        sort_keys=True,
                    )
                )
            if max_updates is not None and updates >= max_updates:
                break
            if result["status"]["state"] in {
                "source_complete",
                "source_interrupted",
                "integrity_error",
            }:
                break
            time.sleep(self.poll_seconds)

    def _detect_source_kind(self, source_kind: str) -> str:
        if source_kind != "auto":
            return source_kind
        if (self.source_root / "staged_manifest.json").is_file():
            return "staged_normalized_playback"
        return "gv2_checkpoint_bundle_v2"

    def _load_manifest_if_present(self) -> dict[str, Any] | None:
        path = self.output_root / "relay_manifest.json"
        if not path.is_file():
            return None
        manifest = load_json(path)
        validate_relay_manifest(path)
        self.relay_id = str(manifest["relay_id"])
        return manifest

    def _verify_accepted_source_hashes(self) -> None:
        for receipt_path in sorted(self.receipts_root.glob("update_*.json")):
            receipt = load_json(receipt_path)
            if receipt.get("schema_version") != RELAY_RECEIPT_SCHEMA_VERSION:
                raise RelayIntegrityError(f"unsupported receipt schema: {receipt_path}")
            for row in receipt.get("source_files") or []:
                path = Path(row["path"])
                if not path.is_file():
                    raise RelayIntegrityError(f"accepted source file missing: {path}")
                actual = sha256_file(path)
                if actual != row["sha256"]:
                    raise RelayIntegrityError(f"accepted source hash changed: {path}")

    def _publish(
        self,
        update: SourceUpdate,
        *,
        relay_id: str,
        sequence_number: int,
        started: float,
    ) -> dict[str, Any]:
        validation_start = time.perf_counter()
        validate_playback(update.playback)
        playback_hash = output_hash(update.playback)
        provenance = dict(update.provenance)
        provenance["output_sha256"] = playback_hash
        provenance["source_root"] = str(update.source_root)
        provenance["source_root_resolved"] = str(update.source_root.resolve())
        provenance["source_files"] = [item.to_json() for item in update.source_files]
        provenance["relay_version"] = RELAY_VERSION
        provenance["relay_sequence_number"] = sequence_number
        provenance_hash = output_hash(provenance)
        cross_check_hash = output_hash(update.cross_check)
        validation_ms = (time.perf_counter() - validation_start) * 1000

        playback_path = self.output_root / "playback_snapshot.json"
        provenance_path = self.output_root / "provenance.json"
        cross_check_path = self.output_root / "cross_check.json"
        receipt_path = self.receipts_root / f"update_{sequence_number:06d}.json"
        scan_ms = validation_ms
        written_at = utc_now()
        receipt = {
            "schema_version": RELAY_RECEIPT_SCHEMA_VERSION,
            "relay_id": relay_id,
            "sequence_number": sequence_number,
            "source_kind": update.source_kind,
            "source_run_id": update.source_run_id,
            "source_root": str(update.source_root),
            "source_boundary": update.boundary,
            "playback_boundary": update.boundary,
            "playback_sha256": playback_hash,
            "provenance_sha256": provenance_hash,
            "cross_check_sha256": cross_check_hash,
            "source_files": [item.to_json() for item in update.source_files],
            "created_at": written_at,
            "cross_check_ok": bool(update.cross_check.get("ok")),
        }
        manifest = {
            "schema_version": RELAY_SCHEMA_VERSION,
            "relay_id": relay_id,
            "source_run_id": update.source_run_id,
            "source_root": str(update.source_root),
            "source_state": update.source_state,
            "current_source_classification": update.source_classification,
            "semantics_mode": update.semantics_mode,
            "branch_semantics": update.branch_semantics,
            "latest_completed_source_boundary": update.boundary,
            "latest_exported_playback_boundary": update.boundary,
            "sequence_number": sequence_number,
            "playback_file_name": playback_path.name,
            "playback_sha256": playback_hash,
            "provenance_file_name": provenance_path.name,
            "provenance_sha256": provenance_hash,
            "status_file_name": "relay_status.json",
            "cross_check_file_name": cross_check_path.name,
            "cross_check_sha256": cross_check_hash,
            "chunks": [],
            "source_lag": 0,
            "last_successful_update": written_at,
            "last_source_scan": written_at,
            "relay_version": RELAY_VERSION,
            "relay_commit": exporter_git_commit(),
            "poll_seconds": self.poll_seconds,
            "warnings": list(update.warnings),
            "scientific_boundary": NO_CLAIMS_STATEMENT,
        }
        status = self._status(
            relay_id=relay_id,
            state=update.source_state,
            source_boundary=update.boundary,
            playback_boundary=update.boundary,
            last_error=None,
            warnings=list(update.warnings),
            sequence_number=sequence_number,
            source_classification=update.source_classification,
            semantics_mode=update.semantics_mode,
            branch_semantics=update.branch_semantics,
            last_successful_update=written_at,
        )
        performance = {
            "schema_version": "llc_near_live_playback_relay_v0_performance",
            "sequence_number": sequence_number,
            "source_scan_ms": scan_ms,
            "export_or_copy_ms": 0.0,
            "validation_ms": validation_ms,
            "hash_ms": 0.0,
            "total_update_ms": (time.perf_counter() - started) * 1000,
            "playback_bytes": len(canonical_json_bytes(update.playback)),
            "provenance_bytes": len(canonical_json_bytes(provenance)),
            "peak_living_count": peak_living_count(update.playback),
            "agent_count": len(update.playback["agents"]),
        }
        write_json_atomic(playback_path, update.playback)
        write_json_atomic(provenance_path, provenance)
        write_json_atomic(cross_check_path, update.cross_check)
        write_json_atomic(receipt_path, receipt)
        write_json_atomic(self.output_root / "relay_performance.json", performance)
        write_json_atomic(self.output_root / "relay_manifest.json", manifest)
        write_json_atomic(self.output_root / "relay_status.json", status)
        return {
            "updated": True,
            "manifest": manifest,
            "status": status,
            "performance": performance,
        }

    def _status(
        self,
        *,
        relay_id: str,
        state: RelayState,
        source_boundary: int | None,
        playback_boundary: int | None,
        last_error: str | None,
        warnings: list[str],
        sequence_number: int,
        source_classification: str | None,
        semantics_mode: str | None,
        branch_semantics: str | None,
        last_successful_update: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        return {
            "schema_version": RELAY_STATUS_SCHEMA_VERSION,
            "relay_id": relay_id,
            "state": state,
            "source_boundary": source_boundary,
            "playback_boundary": playback_boundary,
            "source_lag": None
            if source_boundary is None or playback_boundary is None
            else source_boundary - playback_boundary,
            "last_error": last_error,
            "last_successful_update": last_successful_update,
            "last_source_scan": now,
            "incomplete_artifact_warnings": warnings,
            "source_classification": source_classification,
            "semantics_mode": semantics_mode,
            "branch_semantics": branch_semantics,
            "exact_projected_timing_semantics": "exact births where supplied; checkpoint/projected deaths where documented by playback metadata",
            "relay_sequence": sequence_number,
            "relay_version": RELAY_VERSION,
        }


def validate_relay_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("schema_version") != RELAY_SCHEMA_VERSION:
        raise RelayIntegrityError("unsupported relay manifest schema")
    root = path.parent
    playback_path = root / manifest["playback_file_name"]
    provenance_path = root / manifest["provenance_file_name"]
    status_path = root / manifest["status_file_name"]
    if not playback_path.is_file():
        raise RelayIntegrityError("manifest playback file is missing")
    if not provenance_path.is_file():
        raise RelayIntegrityError("manifest provenance file is missing")
    if not status_path.is_file():
        raise RelayIntegrityError("manifest status file is missing")
    if sha256_file(playback_path) != manifest.get("playback_sha256"):
        raise RelayIntegrityError("manifest playback hash mismatch")
    if sha256_file(provenance_path) != manifest.get("provenance_sha256"):
        raise RelayIntegrityError("manifest provenance hash mismatch")
    sequence = manifest.get("sequence_number")
    if not isinstance(sequence, int) or sequence < 1:
        raise RelayIntegrityError("manifest sequence_number must be a positive integer")
    return manifest
