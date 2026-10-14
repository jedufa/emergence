from __future__ import annotations

"""Read-only HTTP data boundary for the Scientific Chronicle Explorer."""

from collections import Counter
from http import HTTPStatus
import hashlib
import heapq
import io
import itertools
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import sys
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import zipfile


EVENT_SCHEMA = "eiel_formal_event_v1"
ENTRY_SCHEMA = "eiel_scientific_chronicle_entry_v1"
EPISODE_SCHEMA = "eiel_scientific_episode_v1"
CONTEXT_SCHEMA = "eiel_event_context_window_v1"
MANIFEST_SCHEMA = "eiel_scientific_chronicle_manifest_v1"
COMPARISON_SCHEMA = "eiel_scientific_chronicle_comparison_v1"
GEN1_COMPANION_SCHEMA = "eiel_gen1_chronicle_companion_index_v1"
OBSERVER_MANIFEST_SCHEMA = "eiel_observer_manifest_v1"
OBSERVER_WORLD_SCHEMA = "eiel_observer_world_summary_v1"
OBSERVER_INDEX_SCHEMA = "eiel_observer_index_v1"
OBSERVER_SCHEMAS = {
    "feed": "eiel_observer_feed_item_v1",
    "episode": "eiel_observer_episode_story_v1",
    "agent": "eiel_observer_agent_card_v1",
    "lineage": "eiel_observer_lineage_card_v1",
    "region": "eiel_observer_region_card_v1",
}
OBSERVER_PROFILES = ("public_plain", "field_observer", "dry_humour", "documentary")
OBSERVER_ARTIFACTS = {
    "feed": "observer_feed.jsonl",
    "episode": "observer_episodes.jsonl",
    "agent": "observer_agents.jsonl",
    "lineage": "observer_lineages.jsonl",
    "region": "observer_regions.jsonl",
    "world": "observer_world_summary.json",
    "index": "observer_index.json",
    "manifest": "observer_manifest.json",
}
OBSERVER_HASHED_ARTIFACTS = frozenset(
    {
        "observer_feed.jsonl",
        "observer_episodes.jsonl",
        "observer_agents.jsonl",
        "observer_lineages.jsonl",
        "observer_regions.jsonl",
        "observer_world_summary.json",
        "observer_world_history.md",
        "observer_audio_script.txt",
        "observer_index.json",
        "observer_template_inventory.json",
        "observer_storage_preflight.json",
    }
)
LIVE_MANIFEST_SCHEMA = "eiel_chronicle_live_manifest_v1"
LIVE_CURSOR_SCHEMA = "eiel_chronicle_live_cursor_v1"
LIVE_ITEM_SCHEMA = "eiel_chronicle_live_observer_item_v1"
LIVE_EPISODE_SCHEMA = "eiel_chronicle_live_observer_episode_v1"
LIVE_ARTIFACTS = {
    "feed": "live_observer_feed.jsonl",
    "episodes": "live_observer_episodes.jsonl",
    "cursor": "live_observer_cursor.json",
    "status": "live_observer_status.json",
    "health": "live_observer_health.json",
    "index": "live_observer_index.json",
    "manifest": "live_observer_manifest.json",
}
LIVE_HASHED_ARTIFACTS = frozenset(
    {LIVE_ARTIFACTS[name] for name in ("feed", "episodes", "cursor", "index")}
)
DOCUMENTARY_MANIFEST_SCHEMA = "eiel_two_host_documentary_manifest_v1"
DOCUMENTARY_SCRIPT_SCHEMA = "eiel_two_host_documentary_script_v1"
DOCUMENTARY_INDEX_SCHEMA = "eiel_two_host_documentary_index_v1"
DOCUMENTARY_ARTIFACTS = {
    "manifest": "documentary_manifest.json",
    "plan": "documentary_episode_plan.json",
    "script": "documentary_script.json",
    "transcript": "documentary_transcript.md",
    "transcript_html": "documentary_transcript.html",
    "chapters": "documentary_chapters.json",
    "citations": "documentary_citations.json",
    "verifier": "documentary_verifier_report.json",
    "index": "documentary_index.json",
}

ARTIFACTS = {
    "events": "formal_events.jsonl",
    "entries": "scientific_chronicle.jsonl",
    "episodes": "episodes.json",
    "contexts": "context_windows.jsonl",
    "summary": "scientific_chronicle_summary.json",
    "manifest": "chronicle_manifest.json",
    "storage": "chronicle_storage_summary.json",
    "index": "index.json",
}

_RUN_KEY_RE = re.compile(r"^root(?P<index>[0-9]+)/(?P<relative>[A-Za-z0-9._\-/]+)$")
_ARCHIVE_RUN_KEY_RE = re.compile(
    r"^archive/(?P<campaign>[A-Za-z0-9._-]+)/(?P<run>[A-Za-z0-9._-]+)/(?P<world>[A-Za-z0-9._-]+)$"
)
_ARCHIVE_KEY_RE = re.compile(r"^archive/(?P<campaign>[A-Za-z0-9._-]+)/(?P<run>[A-Za-z0-9._-]+)$")
ARCHIVE_MANIFEST = "chronicle_archive_manifest.json"
ARCHIVE_SCHEMA = "eiel_chronicle_archive_manifest_v1"
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_JSONL_BYTES = 512 * 1024 * 1024
_MAX_JSONL_LINE_BYTES = 4 * 1024 * 1024
_MAX_SCAN_DIRS = 25_000
_MAX_PAGE = 200
_MAX_OFFSET = 100_000
_MAX_DETAIL_EVENTS = 2_000
_OBSERVER_PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "background": 4}
_MAX_ARCHIVE_MEMBERS = 100_000
_MAX_ARCHIVE_MEMBER_BYTES = _MAX_JSONL_BYTES
_MAX_ARCHIVE_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
_archive_verification_cache: dict[tuple[str, int, int, int], str] = {}
DIALOGUE_AUDIENCES = (
    "dry_humour",
    "public_plain",
    "field_observer",
    "documentary",
)


class ChronicleReadError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_chronicle", status: int = HTTPStatus.UNPROCESSABLE_ENTITY):
        super().__init__(message)
        self.code = code
        self.status = int(status)


class _OwnedZipStream:
    """File-like ZIP member stream that closes its owning ZipFile."""

    def __init__(self, archive: zipfile.ZipFile, stream: Any):
        self._archive = archive
        self._stream = stream

    def __enter__(self) -> "_OwnedZipStream":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[Any]:
        return iter(self._stream)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            self._archive.close()


class _ArchivePath:
    """Small pathlib-compatible view over one verified ZIP, never extracted."""

    def __init__(self, archive_path: Path, member: str = ""):
        self.archive_path = archive_path
        clean = PurePosixPath(str(member).replace("\\", "/"))
        self.member = "" if str(clean) == "." else clean.as_posix().strip("/")

    def __truediv__(self, value: object) -> "_ArchivePath":
        relative = _safe_relative(str(value))
        member = (PurePosixPath(self.member) / PurePosixPath(relative.as_posix())).as_posix()
        return _ArchivePath(self.archive_path, member)

    @property
    def name(self) -> str:
        return PurePosixPath(self.member).name if self.member else self.archive_path.name

    @property
    def parent(self) -> "_ArchivePath":
        if not self.member:
            return self
        parent = PurePosixPath(self.member).parent
        return _ArchivePath(self.archive_path, "" if str(parent) == "." else parent.as_posix())

    def resolve(self) -> "_ArchivePath":
        return self

    def relative_to(self, other: object) -> PurePosixPath:
        if not isinstance(other, _ArchivePath) or self.archive_path.resolve() != other.archive_path.resolve():
            raise ValueError("archive paths have different authority")
        return PurePosixPath(self.member).relative_to(PurePosixPath(other.member))

    def as_posix(self) -> str:
        return self.member

    def is_symlink(self) -> bool:
        return False

    def _info(self) -> zipfile.ZipInfo | None:
        if not self.member:
            return None
        with zipfile.ZipFile(self.archive_path, "r") as archive:
            try:
                return archive.getinfo(self.member)
            except KeyError:
                return None

    def is_file(self) -> bool:
        info = self._info()
        return bool(info is not None and not info.is_dir())

    def is_dir(self) -> bool:
        if not self.member:
            return True
        prefix = self.member.rstrip("/") + "/"
        with zipfile.ZipFile(self.archive_path, "r") as archive:
            return any(name.startswith(prefix) for name in archive.namelist())

    def stat(self) -> Any:
        info = self._info()
        if info is None:
            raise FileNotFoundError(self.member)
        return SimpleNamespace(st_size=info.file_size)

    def open(self, mode: str = "r", encoding: str | None = None) -> _OwnedZipStream:
        if mode not in {"r", "rb"}:
            raise ValueError("Chronicle archives are read-only")
        info = self._info()
        if info is None or info.is_dir():
            raise FileNotFoundError(self.member)
        if info.file_size > _MAX_ARCHIVE_MEMBER_BYTES:
            raise ChronicleReadError(
                f"archive member exceeds reader bound: {self.name}",
                code="artifact_too_large",
            )
        archive = zipfile.ZipFile(self.archive_path, "r")
        try:
            raw = archive.open(self.member, "r")
        except Exception:
            archive.close()
            raise
        stream = raw if mode == "rb" else io.TextIOWrapper(raw, encoding=encoding or "utf-8")
        return _OwnedZipStream(archive, stream)

    def read_text(self, encoding: str = "utf-8") -> str:
        with self.open("r", encoding=encoding) as handle:
            return handle.read()

    def __str__(self) -> str:
        return f"{self.archive_path}!/{self.member}"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _approved_roots(cfg: Any) -> tuple[Path, ...]:
    configured = tuple(Path(value).expanduser().resolve() for value in (getattr(cfg, "chronicle_roots", ()) or ()))
    roots = configured or (Path(cfg.runs_root).expanduser().resolve(),)
    return tuple(dict.fromkeys(roots))


def _approved_live_roots(cfg: Any) -> tuple[Path, ...]:
    roots = tuple(
        Path(value).expanduser().resolve()
        for value in (getattr(cfg, "chronicle_live_roots", ()) or ())
    )
    return tuple(dict.fromkeys(roots))


def _archive_root(cfg: Any) -> Path:
    roots = _archive_roots(cfg)
    if not roots:
        # Direct library consumers predating archive configuration remain
        # isolated. Production WebUIConfig always supplies the default root.
        return Path("/__chronicle_archive_disabled__")
    return roots[0]


def _archive_roots(cfg: Any) -> tuple[Path, ...]:
    configured = tuple(
        Path(value).expanduser().resolve()
        for value in (getattr(cfg, "chronicle_archive_roots", ()) or ())
    )
    if configured:
        return tuple(dict.fromkeys(configured))
    legacy = getattr(cfg, "chronicle_archive_root", None)
    if legacy is None:
        return ()
    return (Path(legacy).expanduser().resolve(),)


def _archive_path(root: Path, campaign_id: str, run_id: str) -> Path:
    campaign = _safe_component(campaign_id, "campaign")
    run = _safe_component(run_id, "run")
    path = (root / campaign / run / "chronicles.zip").resolve()
    if not _inside(path, root) or path.is_symlink():
        raise ChronicleReadError(
            "archive path escapes approved root",
            code="path_traversal",
            status=HTTPStatus.BAD_REQUEST,
        )
    return path


def _safe_component(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", text) or text in {".", ".."}:
        raise ChronicleReadError(
            f"invalid archive {label} identity",
            code="invalid_archive_key",
            status=HTTPStatus.BAD_REQUEST,
        )
    return text


def _read_archive_checksum(path: Path) -> str:
    sidecar = Path(f"{path}.sha256")
    if sidecar.is_symlink() or not sidecar.is_file() or sidecar.stat().st_size > 4096:
        raise ChronicleReadError("archive checksum sidecar is missing", code="archive_unverified")
    try:
        line = sidecar.read_text(encoding="utf-8").strip().splitlines()[0]
    except (OSError, UnicodeError, IndexError) as exc:
        raise ChronicleReadError("archive checksum sidecar is unreadable", code="archive_unverified") from exc
    expected = line.split()[0].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ChronicleReadError("archive checksum sidecar is invalid", code="archive_unverified")
    return expected


def _validated_archive_at_root(
    root: Path,
    campaign_id: str,
    run_id: str,
) -> tuple[Path, str, dict[str, Any], tuple[str, ...]]:
    path = _archive_path(root, campaign_id, run_id)
    if not path.is_file():
        raise ChronicleReadError("Chronicle archive not found", code="not_found", status=HTTPStatus.NOT_FOUND)
    expected = _read_archive_checksum(path)
    stat = path.stat()
    signature = (
        str(path),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
        int(stat.st_size),
    )
    observed = _archive_verification_cache.get(signature)
    if observed != expected:
        observed = _sha256(path)
        if observed != expected:
            raise ChronicleReadError("Chronicle archive checksum mismatch", code="archive_checksum_mismatch")
        _archive_verification_cache.clear()
        _archive_verification_cache[signature] = observed
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if len(infos) > _MAX_ARCHIVE_MEMBERS:
                raise ChronicleReadError("Chronicle archive has too many members", code="archive_too_large")
            names: list[str] = []
            seen: set[str] = set()
            total_uncompressed = 0
            for info in infos:
                raw_name = info.filename.replace("\\", "/")
                pure = PurePosixPath(raw_name)
                if (
                    not raw_name
                    or raw_name.startswith("/")
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or pure.as_posix() != raw_name.rstrip("/")
                ):
                    raise ChronicleReadError("Chronicle archive contains an unsafe member", code="path_traversal")
                name = pure.as_posix()
                if name in seen:
                    raise ChronicleReadError("Chronicle archive contains duplicate members", code="archive_invalid")
                seen.add(name)
                names.append(name)
                unix_mode = (info.external_attr >> 16) & 0o170000
                if unix_mode == 0o120000:
                    raise ChronicleReadError("Chronicle archive contains a symlink", code="path_traversal")
                if not info.is_dir() and info.file_size > _MAX_ARCHIVE_MEMBER_BYTES:
                    raise ChronicleReadError("Chronicle archive member exceeds reader bound", code="archive_too_large")
                total_uncompressed += int(info.file_size)
                if total_uncompressed > _MAX_ARCHIVE_TOTAL_BYTES:
                    raise ChronicleReadError("Chronicle archive exceeds total reader bound", code="archive_too_large")
            if ARCHIVE_MANIFEST not in seen:
                raise ChronicleReadError("Chronicle archive manifest is missing", code="missing_artifact")
            manifest_info = archive.getinfo(ARCHIVE_MANIFEST)
            if manifest_info.file_size > _MAX_JSON_BYTES:
                raise ChronicleReadError("Chronicle archive manifest is too large", code="archive_too_large")
            try:
                manifest = json.loads(archive.read(ARCHIVE_MANIFEST).decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ChronicleReadError("Chronicle archive manifest is corrupt", code="corrupt_artifact") from exc
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ChronicleReadError(f"Chronicle archive is corrupt: {exc}", code="corrupt_archive") from exc
    if not isinstance(manifest, dict):
        raise ChronicleReadError("Chronicle archive manifest has an invalid shape", code="schema_mismatch")
    observed_schema = str(manifest.get("schema_version") or "")
    if observed_schema and observed_schema != ARCHIVE_SCHEMA:
        raise ChronicleReadError("Chronicle archive manifest schema is unsupported", code="schema_mismatch")
    if str(manifest.get("campaign_id") or campaign_id) != campaign_id:
        raise ChronicleReadError("Chronicle archive campaign identity mismatch", code="identity_mismatch")
    if str(manifest.get("run_id") or run_id) != run_id:
        raise ChronicleReadError("Chronicle archive run identity mismatch", code="identity_mismatch")
    return path, expected, manifest, tuple(names)


def _validated_archive(
    cfg: Any,
    campaign_id: str,
    run_id: str,
) -> tuple[Path, str, dict[str, Any], tuple[str, ...]]:
    for root in _archive_roots(cfg):
        if _archive_path(root, campaign_id, run_id).is_file():
            return _validated_archive_at_root(root, campaign_id, run_id)
    raise ChronicleReadError(
        "Chronicle archive not found",
        code="not_found",
        status=HTTPStatus.NOT_FOUND,
    )


def _safe_relative(value: str) -> Path:
    text = str(value or "").strip().replace("\\", "/")
    parts = text.split("/")
    if not text or text.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ChronicleReadError("unsafe relative path", code="path_traversal", status=HTTPStatus.BAD_REQUEST)
    return Path(*parts)


def _parse_root_key(key: str, roots: Sequence[Path]) -> tuple[Path, Path]:
    match = _RUN_KEY_RE.fullmatch(str(key or ""))
    if not match:
        raise ChronicleReadError("invalid run key", code="invalid_run_key", status=HTTPStatus.BAD_REQUEST)
    index = int(match.group("index"))
    if index < 0 or index >= len(roots):
        raise ChronicleReadError("run root is not approved", code="unapproved_root", status=HTTPStatus.BAD_REQUEST)
    root = roots[index]
    relative_text = match.group("relative")
    candidate = root if relative_text == "__root__" else root / _safe_relative(relative_text)
    candidate = candidate.resolve()
    if not _inside(candidate, root) or candidate.is_symlink():
        raise ChronicleReadError("run path escapes approved root", code="path_traversal", status=HTTPStatus.BAD_REQUEST)
    return root, candidate


def _chronicle_root(run_root: Path) -> Path:
    nested = run_root / "chronicle"
    if (nested / ARTIFACTS["manifest"]).is_file() or (nested / ARTIFACTS["events"]).is_file():
        return nested
    return run_root


def _load_json(path: Path, *, expected_type: type | tuple[type, ...] = dict) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ChronicleReadError(f"missing required artifact: {path.name}", code="missing_artifact")
    size = path.stat().st_size
    if size > _MAX_JSON_BYTES:
        raise ChronicleReadError(f"artifact exceeds bounded JSON reader: {path.name}", code="artifact_too_large")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ChronicleReadError(f"corrupt JSON artifact {path.name}: {exc}", code="corrupt_artifact") from exc
    if not isinstance(value, expected_type):
        raise ChronicleReadError(f"unexpected JSON shape in {path.name}", code="schema_mismatch")
    return value


def _require_schema(record: Mapping[str, Any], expected: str, artifact: str) -> None:
    observed = str(record.get("schema_version", ""))
    if observed != expected:
        raise ChronicleReadError(
            f"unsupported schema in {artifact}: expected {expected}, observed {observed or 'missing'}",
            code="schema_mismatch",
        )


def _validate_event(record: Mapping[str, Any], artifact: str) -> None:
    _require_schema(record, EVENT_SCHEMA, artifact)
    required = {
        "event_id", "run_id", "generation", "step", "phase_order", "intra_step_sequence",
        "event_type", "event_category", "severity", "live_priority", "scientific_importance",
        "historical_importance", "public_interest", "visual_interest", "confidence_class",
        "causality_class", "source_component", "source_record_refs",
    }
    missing = sorted(required.difference(record))
    if missing or not str(record.get("event_id", "")).startswith("evt_"):
        raise ChronicleReadError(f"invalid formal event fields in {artifact}: {', '.join(missing) or 'event_id'}", code="schema_mismatch")


def _validate_entry(record: Mapping[str, Any], artifact: str) -> None:
    _require_schema(record, ENTRY_SCHEMA, artifact)
    if not str(record.get("chronicle_entry_id", "")).startswith("chr_") or not isinstance(record.get("source_event_ids"), list):
        raise ChronicleReadError(f"invalid Chronicle entry in {artifact}", code="schema_mismatch")


def _validate_episode(record: Mapping[str, Any], artifact: str) -> None:
    _require_schema(record, EPISODE_SCHEMA, artifact)
    if not str(record.get("episode_id", "")).startswith("epi_") or not isinstance(record.get("source_event_ids"), list):
        raise ChronicleReadError(f"invalid episode in {artifact}", code="schema_mismatch")


def _validate_context(record: Mapping[str, Any], artifact: str) -> None:
    _require_schema(record, CONTEXT_SCHEMA, artifact)
    if not str(record.get("context_window_id", "")).startswith("ctx_"):
        raise ChronicleReadError(f"invalid context window in {artifact}", code="schema_mismatch")


def _validate_observer(kind: str) -> Callable[[Mapping[str, Any], str], None]:
    expected = OBSERVER_SCHEMAS[kind]

    def validate(record: Mapping[str, Any], artifact: str) -> None:
        _require_schema(record, expected, artifact)
        if record.get("presentation_only") is not True or str(record.get("audience_profile")) not in OBSERVER_PROFILES:
            raise ChronicleReadError(f"invalid Observer presentation record in {artifact}", code="schema_mismatch")
        if kind in {"feed", "episode"} and not isinstance(record.get("source_links"), dict):
            raise ChronicleReadError(f"Observer record lacks source links in {artifact}", code="traceability_mismatch")
        if kind in {"agent", "lineage", "region"} and not isinstance(record.get("source_event_ids"), list):
            raise ChronicleReadError(f"Observer card lacks source event links in {artifact}", code="traceability_mismatch")

    return validate


def _iter_jsonl(path: Path, validator: Callable[[Mapping[str, Any], str], None]) -> Iterator[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ChronicleReadError(f"missing required artifact: {path.name}", code="missing_artifact")
    if path.stat().st_size > _MAX_JSONL_BYTES:
        raise ChronicleReadError(f"artifact exceeds bounded JSONL reader: {path.name}", code="artifact_too_large")
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                if len(raw.encode("utf-8")) > _MAX_JSONL_LINE_BYTES:
                    raise ChronicleReadError(f"oversized JSONL record in {path.name}:{line_number}", code="artifact_too_large")
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ChronicleReadError(f"corrupt JSONL artifact {path.name}:{line_number}: {exc}", code="corrupt_artifact") from exc
                if not isinstance(record, dict):
                    raise ChronicleReadError(f"non-object JSONL record in {path.name}:{line_number}", code="schema_mismatch")
                validator(record, f"{path.name}:{line_number}")
                yield record
    except UnicodeError as exc:
        raise ChronicleReadError(f"invalid UTF-8 in {path.name}: {exc}", code="corrupt_artifact") from exc


def _load_episodes(root: Path) -> list[dict[str, Any]]:
    payload = _load_json(root / ARTIFACTS["episodes"], expected_type=list)
    episodes: list[dict[str, Any]] = []
    for index, value in enumerate(payload):
        if not isinstance(value, dict):
            raise ChronicleReadError(f"non-object episode at index {index}", code="schema_mismatch")
        _validate_episode(value, f"{ARTIFACTS['episodes']}:{index}")
        episodes.append(value)
    return episodes


def _relative_key(index: int, root: Path, run_root: Path) -> str:
    relative = run_root.resolve().relative_to(root.resolve()).as_posix()
    return f"root{index}/{relative if relative != '.' else '__root__'}"


def _walk_for(root: Path, filename: str, *, max_depth: int) -> Iterator[Path]:
    if not root.is_dir() or root.is_symlink():
        return
    visited = 0
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        visited += 1
        if visited > _MAX_SCAN_DIRS:
            break
        current_path = Path(current)
        try:
            depth = len(current_path.relative_to(root).parts)
        except ValueError:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            name for name in dirnames
            if not name.startswith(".") and not (current_path / name).is_symlink()
        )
        if depth >= int(max_depth):
            dirnames[:] = []
        if filename in filenames:
            candidate = current_path / filename
            if _inside(candidate, root) and not candidate.is_symlink():
                yield candidate


def discover_runs(cfg: Any) -> list[dict[str, Any]]:
    roots = _approved_roots(cfg)
    depth = max(1, min(32, int(getattr(cfg, "chronicle_scan_depth", 8) or 8)))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for index, root in enumerate(roots):
        for manifest_path in _walk_for(root, ARTIFACTS["manifest"], max_depth=depth):
            chronicle_root = manifest_path.parent
            if not (chronicle_root / ARTIFACTS["events"]).is_file():
                continue
            run_root = chronicle_root.parent if chronicle_root.name == "chronicle" else chronicle_root
            signature = (index, str(run_root.resolve()))
            if signature in seen:
                continue
            seen.add(signature)
            run_key = _relative_key(index, root, run_root)
            try:
                manifest = _load_json(manifest_path)
                _require_schema(manifest, MANIFEST_SCHEMA, ARTIFACTS["manifest"])
                summary = _load_json(chronicle_root / ARTIFACTS["summary"])
                storage_summary_available = True
                try:
                    storage = _load_json(chronicle_root / ARTIFACTS["storage"])
                except ChronicleReadError as exc:
                    if exc.code != "missing_artifact":
                        raise
                    # Dialogue reads the formal event stream and its summary;
                    # storage accounting is useful metadata, not a dialogue
                    # prerequisite. Keep older compact runs discoverable.
                    storage = {}
                    storage_summary_available = False
                row = {
                    "run_key": run_key,
                    "run_path": str(run_root),
                    "status": "ready",
                    "run_id": str(manifest.get("run_id") or summary.get("run_id") or run_root.name),
                    "campaign_id": manifest.get("campaign_id"),
                    "branch_id": manifest.get("branch_id"),
                    "source_commit": str(manifest.get("source_commit") or "unknown"),
                    "profile": str(manifest.get("profile") or summary.get("profile") or "unknown"),
                    "source_completeness": str(manifest.get("source_completeness") or summary.get("source_completeness") or "unknown"),
                    "event_count": int(manifest.get("event_count", summary.get("event_count", 0)) or 0),
                    "episode_count": int(manifest.get("episode_count", summary.get("episode_count", 0)) or 0),
                    "generation_end": summary.get("last_generation"),
                    "storage_bytes": int(storage.get("total_bytes", manifest.get("storage_bytes", 0)) or 0),
                    "storage_summary_available": storage_summary_available,
                    "finalized": bool(manifest.get("finalized", summary.get("finalized", False))),
                    "retrospective": str(manifest.get("profile", "")) == "retrospective",
                    "parent_checkpoint_id": manifest.get("parent_checkpoint_id"),
                    "schema_versions": {"manifest": str(manifest.get("schema_version", ""))},
                    "observer_profiles": _observer_profiles(run_root),
                    "documentary_available": _documentary_available(run_root),
                    "modified_at_ns": int(manifest_path.stat().st_mtime_ns),
                    "limitations": list(summary.get("limitations", []) or []) + list(manifest.get("limitations", []) or []),
                }
            except ChronicleReadError as exc:
                row = {
                    "run_key": run_key,
                    "status": "invalid",
                    "error_code": exc.code,
                    "error": str(exc),
                    "run_id": run_root.name,
                }
            rows.append(row)
    rows.extend(discover_archived_runs(cfg))
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("campaign_id", "")),
            str(row.get("run_id", "")),
            str(row.get("world_id", "")),
            str(row["run_key"]),
        ),
    )


def discover_archived_runs(cfg: Any) -> list[dict[str, Any]]:
    """Discover verified completed worlds without extracting their run ZIPs."""

    rows: list[dict[str, Any]] = []
    seen_runs: set[tuple[str, str]] = set()
    for root in _archive_roots(cfg):
        if not root.is_dir() or root.is_symlink():
            continue
        for campaign_dir in sorted(root.iterdir(), key=lambda path: path.name):
            if not campaign_dir.is_dir() or campaign_dir.is_symlink():
                continue
            try:
                campaign_id = _safe_component(campaign_dir.name, "campaign")
            except ChronicleReadError:
                continue
            for run_dir in sorted(campaign_dir.iterdir(), key=lambda path: path.name):
                if not run_dir.is_dir() or run_dir.is_symlink():
                    continue
                identity = (campaign_id, run_dir.name)
                if identity in seen_runs:
                    continue
                try:
                    run_id = _safe_component(run_dir.name, "run")
                    archive_path, checksum, archive_manifest, names = (
                        _validated_archive_at_root(root, campaign_id, run_id)
                    )
                    world_ids = sorted(
                        {
                            parts[1]
                            for name in names
                            for parts in [PurePosixPath(name).parts]
                            if (
                                len(parts) == 4
                                and parts[0] == "worlds"
                                and parts[2] == "chronicle"
                                and parts[3] == ARTIFACTS["manifest"]
                            )
                        }
                    )
                    if not world_ids:
                        raise ChronicleReadError("Chronicle archive contains no worlds", code="missing_artifact")
                    for world_id in world_ids:
                        _safe_component(world_id, "world")
                        run_root = _ArchivePath(archive_path, f"worlds/{world_id}")
                        chronicle_root = run_root / "chronicle"
                        manifest = _load_json(chronicle_root / ARTIFACTS["manifest"])
                        _require_schema(manifest, MANIFEST_SCHEMA, ARTIFACTS["manifest"])
                        summary = _load_json(chronicle_root / ARTIFACTS["summary"])
                        storage = _load_json(chronicle_root / ARTIFACTS["storage"])
                        rows.append(
                            {
                                "run_key": f"archive/{campaign_id}/{run_id}/{world_id}",
                                "run_path": f"{archive_path}!/worlds/{world_id}",
                                "archive_key": f"archive/{campaign_id}/{run_id}",
                                "archive_root": str(root),
                                "status": "ready",
                                "storage_kind": "verified_zip",
                                "run_id": str(manifest.get("run_id") or summary.get("run_id") or world_id),
                                "world_id": world_id,
                                "campaign_id": str(manifest.get("campaign_id") or campaign_id),
                                "archive_run_id": run_id,
                                "branch_id": manifest.get("branch_id"),
                                "source_commit": str(manifest.get("source_commit") or "unknown"),
                                "profile": str(manifest.get("profile") or summary.get("profile") or "unknown"),
                                "source_completeness": str(
                                    manifest.get("source_completeness")
                                    or summary.get("source_completeness")
                                    or "unknown"
                                ),
                                "event_count": int(manifest.get("event_count", summary.get("event_count", 0)) or 0),
                                "episode_count": int(manifest.get("episode_count", summary.get("episode_count", 0)) or 0),
                                "generation_end": summary.get("last_generation"),
                                "storage_bytes": int(storage.get("total_bytes", 0) or 0),
                                "archive_bytes": int(archive_path.stat().st_size),
                                "archive_sha256": checksum,
                                "finalized": bool(manifest.get("finalized", summary.get("finalized", False))),
                                "retrospective": str(manifest.get("profile", "")) == "retrospective",
                                "parent_checkpoint_id": manifest.get("parent_checkpoint_id"),
                                "schema_versions": {
                                    "manifest": str(manifest.get("schema_version", "")),
                                    "archive": str(archive_manifest.get("schema_version") or ARCHIVE_SCHEMA),
                                },
                                "observer_profiles": _observer_profiles(run_root),
                                "documentary_available": _documentary_available(run_root),
                                "modified_at_ns": int(archive_path.stat().st_mtime_ns),
                                "limitations": list(summary.get("limitations", []) or [])
                                + list(manifest.get("limitations", []) or []),
                            }
                        )
                    seen_runs.add(identity)
                except ChronicleReadError as exc:
                    rows.append(
                        {
                            "archive_key": f"archive/{campaign_dir.name}/{run_dir.name}",
                            "run_key": f"archive/{campaign_dir.name}/{run_dir.name}/__invalid__",
                            "archive_root": str(root),
                            "status": "invalid",
                            "storage_kind": "zip",
                            "campaign_id": campaign_dir.name,
                            "archive_run_id": run_dir.name,
                            "run_id": run_dir.name,
                            "error_code": exc.code,
                            "error": str(exc),
                        }
                    )
    return rows


def _dialogue_render_observer_event() -> Callable[..., dict[str, Any]]:
    """Load the existing deterministic Observer renderer without a second copy."""

    engine_root = Path(__file__).resolve().parents[2] / "engine"
    if str(engine_root) not in sys.path:
        sys.path.insert(0, str(engine_root))
    from simulation.chronicle.observer import render_observer_event

    return render_observer_event


def dialogue(
    cfg: Any,
    run_key: str,
    audience: str,
    qs: Mapping[str, list[str]],
) -> dict[str, Any]:
    """Return a read-only Creator/Scientist presentation of Chronicle events."""

    if audience not in DIALOGUE_AUDIENCES:
        raise ChronicleReadError(
            "invalid dialogue audience",
            code="invalid_query",
            status=HTTPStatus.BAD_REQUEST,
        )
    _run_root, chronicle_root, manifest, summary = _resolve_run(cfg, run_key)
    events = list(
        _iter_jsonl(chronicle_root / ARTIFACTS["events"], _validate_event)
    )
    try:
        entries = list(
            _iter_jsonl(chronicle_root / ARTIFACTS["entries"], _validate_entry)
        )
    except ChronicleReadError as exc:
        if exc.code != "missing_artifact":
            raise
        # The dialogue renderer can safely derive a bounded presentation from
        # the formal events when an older compact run omitted rendered entries.
        entries = []
    entry_by_event: dict[str, dict[str, Any]] = {}
    for entry in entries:
        for event_id in entry.get("source_event_ids", []) or []:
            entry_by_event.setdefault(str(event_id), entry)
    limit = _int_param(qs, "limit")
    if limit is None:
        limit = 500
    if limit < 1 or limit > 1000:
        raise ChronicleReadError(
            "dialogue limit must be between 1 and 1000",
            code="invalid_query",
            status=HTTPStatus.BAD_REQUEST,
        )
    renderer = _dialogue_render_observer_event()
    renderer_profile = "cosmic_creator" if audience == "dry_humour" else audience
    source_completeness = str(
        manifest.get("source_completeness")
        or summary.get("source_completeness")
        or "unknown"
    )
    messages: list[dict[str, Any]] = []
    selected = events[:limit]
    for event in selected:
        event_id = str(event["event_id"])
        entry = entry_by_event.get(event_id)
        if entry is None:
            entry = {
                "chronicle_entry_id": f"derived_{event_id}",
                "source_event_ids": [event_id],
                "subject_references": {},
                "limitations": [
                    "No matching rendered Chronicle entry was available."
                ],
            }
        rendered = renderer(
            event,
            entry,
            audience_profile=renderer_profile,
            source_completeness=source_completeness,
        )
        generation = int(event.get("generation", 0))
        common = {
            "event_id": event_id,
            "generation": generation,
            "headline": rendered.get("headline"),
            "confidence_class": rendered.get("confidence_class"),
            "source_completeness": source_completeness,
        }
        messages.append(
            {
                **common,
                "message_id": f"{event_id}:scientist",
                "speaker": "scientist",
                "speaker_label": "Scientist",
                "text": str(rendered.get("plain_factual_text") or ""),
            }
        )
        creator_text = str(
            rendered.get("humour_line")
            or rendered.get("headline")
            or "The record has added another bounded observation."
        )
        messages.append(
            {
                **common,
                "message_id": f"{event_id}:creator",
                "speaker": "creator",
                "speaker_label": "Creator",
                "text": creator_text,
            }
        )
    return {
        "schema_version": "eiel_chronicle_dialogue_v1",
        "run_key": run_key,
        "run_id": manifest.get("run_id") or summary.get("run_id"),
        "campaign_id": manifest.get("campaign_id"),
        "audience": audience,
        "style": "creator_vs_scientist",
        "presentation_only": True,
        "authoritative_source": "Scientific Chronicle",
        "source_completeness": source_completeness,
        "event_count": len(events),
        "selected_event_count": len(selected),
        "truncated": len(selected) < len(events),
        "messages": messages,
    }


def archive_library(cfg: Any) -> dict[str, Any]:
    rows = discover_archived_runs(cfg)
    campaigns: dict[str, dict[str, Any]] = {}
    for row in rows:
        campaign_id = str(row.get("campaign_id") or "unknown")
        run_id = str(row.get("archive_run_id") or row.get("run_id") or "unknown")
        campaign = campaigns.setdefault(
            campaign_id,
            {"campaign_id": campaign_id, "runs": {}},
        )
        run = campaign["runs"].setdefault(
            run_id,
            {
                "run_id": run_id,
                "archive_key": row.get("archive_key"),
                "status": row.get("status"),
                "archive_sha256": row.get("archive_sha256"),
                "archive_bytes": row.get("archive_bytes"),
                "source_commits": set(),
                "profiles": set(),
                "worlds": [],
            },
        )
        if row.get("status") != "ready":
            run["status"] = "invalid"
            run["error_code"] = row.get("error_code")
            run["error"] = row.get("error")
            continue
        run["source_commits"].add(str(row.get("source_commit") or "unknown"))
        run["profiles"].add(str(row.get("profile") or "unknown"))
        run["worlds"].append(
            {
                key: row.get(key)
                for key in (
                    "world_id", "run_id", "run_key", "profile", "source_completeness",
                    "source_commit", "event_count", "episode_count", "generation_end",
                    "storage_bytes", "finalized", "observer_profiles", "documentary_available",
                )
            }
        )
    result: list[dict[str, Any]] = []
    for campaign in campaigns.values():
        run_rows: list[dict[str, Any]] = []
        for run in campaign["runs"].values():
            run["source_commits"] = sorted(run["source_commits"])
            run["profiles"] = sorted(run["profiles"])
            run["worlds"] = sorted(run["worlds"], key=lambda row: str(row.get("world_id") or ""))
            run["world_count"] = len(run["worlds"])
            run_rows.append(run)
        result.append(
            {
                "campaign_id": campaign["campaign_id"],
                "runs": sorted(run_rows, key=lambda row: str(row["run_id"])),
            }
        )
    return {
        "schema_version": "eiel_chronicle_archive_library_v1",
        "archive_root": str(_archive_root(cfg)),
        "archive_roots": [str(root) for root in _archive_roots(cfg)],
        "campaigns": sorted(result, key=lambda row: str(row["campaign_id"])),
    }


def resolve_archive_download(cfg: Any, archive_key: str) -> tuple[Path, str]:
    """Resolve and re-verify one approved ZIP for the HTTP download boundary."""

    match = _ARCHIVE_KEY_RE.fullmatch(str(archive_key or ""))
    if not match:
        raise ChronicleReadError(
            "invalid archive key",
            code="invalid_archive_key",
            status=HTTPStatus.BAD_REQUEST,
        )
    path, checksum, _manifest, _names = _validated_archive(
        cfg,
        match.group("campaign"),
        match.group("run"),
    )
    return path, checksum


def _resolve_run(cfg: Any, run_key: str) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    archive_match = _ARCHIVE_RUN_KEY_RE.fullmatch(str(run_key or ""))
    if archive_match:
        campaign_id = archive_match.group("campaign")
        run_id = archive_match.group("run")
        world_id = archive_match.group("world")
        archive_path, _checksum, _archive_manifest, names = _validated_archive(cfg, campaign_id, run_id)
        prefix = f"worlds/{world_id}/chronicle/"
        if not any(name.startswith(prefix) for name in names):
            raise ChronicleReadError("Chronicle world not found", code="not_found", status=HTTPStatus.NOT_FOUND)
        run_root = _ArchivePath(archive_path, f"worlds/{world_id}")
        chronicle_root = run_root / "chronicle"
        manifest = _load_json(chronicle_root / ARTIFACTS["manifest"])
        _require_schema(manifest, MANIFEST_SCHEMA, ARTIFACTS["manifest"])
        summary = _load_json(chronicle_root / ARTIFACTS["summary"])
        return run_root, chronicle_root, manifest, summary
    roots = _approved_roots(cfg)
    _root, run_root = _parse_root_key(run_key, roots)
    chronicle_root = _chronicle_root(run_root)
    if not _inside(chronicle_root, _root) or chronicle_root.is_symlink():
        raise ChronicleReadError("Chronicle path escapes approved root", code="path_traversal", status=HTTPStatus.BAD_REQUEST)
    manifest = _load_json(chronicle_root / ARTIFACTS["manifest"])
    _require_schema(manifest, MANIFEST_SCHEMA, ARTIFACTS["manifest"])
    summary = _load_json(chronicle_root / ARTIFACTS["summary"])
    return run_root, chronicle_root, manifest, summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _observer_profiles(run_root: Path) -> list[str]:
    candidates = (run_root / "observer", run_root / "chronicle" / "observer")
    profiles: list[str] = []
    for base in candidates:
        if base.is_symlink() or not base.is_dir() or not _inside(base, run_root):
            continue
        for profile in OBSERVER_PROFILES:
            manifest = base / profile / OBSERVER_ARTIFACTS["manifest"]
            if manifest.is_file() and not manifest.is_symlink() and _inside(manifest, run_root):
                profiles.append(profile)
    return sorted(set(profiles), key=OBSERVER_PROFILES.index)


def _documentary_candidates(run_root: Path) -> tuple[Path, ...]:
    return (
        run_root / "documentary",
        run_root / "chronicle" / "documentary",
        run_root / "chronicle" / "pod1",
        run_root / "pod1",
    )


def _documentary_available(run_root: Path) -> bool:
    return any(
        candidate.is_dir()
        and not candidate.is_symlink()
        and _inside(candidate, run_root)
        and (candidate / DOCUMENTARY_ARTIFACTS["manifest"]).is_file()
        for candidate in _documentary_candidates(run_root)
    )


def _resolve_documentary(cfg: Any, run_key: str) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_root, chronicle_root, scientific_manifest, scientific_summary = _resolve_run(cfg, run_key)
    documentary_root = next(
        (
            candidate.resolve()
            for candidate in _documentary_candidates(run_root)
            if candidate.is_dir()
            and not candidate.is_symlink()
            and _inside(candidate, run_root)
            and (candidate / DOCUMENTARY_ARTIFACTS["manifest"]).is_file()
        ),
        None,
    )
    if documentary_root is None:
        raise ChronicleReadError("POD1 documentary is not available for this run", code="not_found", status=HTTPStatus.NOT_FOUND)
    manifest = _load_json(documentary_root / DOCUMENTARY_ARTIFACTS["manifest"])
    _require_schema(manifest, DOCUMENTARY_MANIFEST_SCHEMA, DOCUMENTARY_ARTIFACTS["manifest"])
    if manifest.get("presentation_only") is not True or manifest.get("source_immutable") is not True:
        raise ChronicleReadError("POD1 documentary manifest does not declare read-only presentation boundary", code="schema_mismatch")
    if str(manifest.get("run_id")) != str(scientific_manifest.get("run_id")):
        raise ChronicleReadError("POD1 documentary and Scientific Chronicle run identities differ", code="traceability_mismatch")
    for name, expected in dict(manifest.get("source_hashes_after", {}) or {}).items():
        source_path = chronicle_root / _safe_relative(str(name))
        if source_path.is_symlink() or not _inside(source_path.resolve(), chronicle_root) or not source_path.is_file():
            raise ChronicleReadError("POD1 source hash references unsafe or missing Chronicle artifact", code="traceability_mismatch")
        if _sha256(source_path) != str(expected):
            raise ChronicleReadError("POD1 source hash no longer matches the Scientific Chronicle", code="source_hash_mismatch")
    for artifact in list(manifest.get("artifacts", []) or []):
        if not isinstance(artifact, dict):
            raise ChronicleReadError("POD1 artifact inventory is invalid", code="schema_mismatch")
        if str(artifact.get("relative_path", "")) == DOCUMENTARY_ARTIFACTS["manifest"]:
            continue
        relative = _safe_relative(str(artifact.get("relative_path", "")))
        raw_path = documentary_root / relative
        path = raw_path.resolve()
        if raw_path.is_symlink() or not _inside(path, documentary_root) or not path.is_file():
            raise ChronicleReadError("POD1 artifact is unsafe or missing", code="traceability_mismatch")
        if _sha256(path) != str(artifact.get("sha256", "")):
            raise ChronicleReadError("POD1 artifact hash does not match its manifest", code="documentary_hash_mismatch")
    index = _load_json(documentary_root / DOCUMENTARY_ARTIFACTS["index"])
    _require_schema(index, DOCUMENTARY_INDEX_SCHEMA, DOCUMENTARY_ARTIFACTS["index"])
    return run_root, chronicle_root, documentary_root, manifest, index, scientific_summary


def _resolve_observer(
    cfg: Any,
    run_key: str,
    profile: str,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if profile not in OBSERVER_PROFILES:
        raise ChronicleReadError("invalid Observer audience profile", code="invalid_query", status=HTTPStatus.BAD_REQUEST)
    run_root, chronicle_root, scientific_manifest, scientific_summary = _resolve_run(cfg, run_key)
    candidates = (
        run_root / "observer" / profile,
        run_root / "chronicle" / "observer" / profile,
    )
    observer_root = next(
        (
            candidate.resolve()
            for candidate in candidates
            if candidate.is_dir()
            and not candidate.is_symlink()
            and (candidate / OBSERVER_ARTIFACTS["manifest"]).is_file()
            and _inside(candidate, run_root)
        ),
        None,
    )
    if observer_root is None:
        raise ChronicleReadError("Observer profile is not available for this run", code="not_found", status=HTTPStatus.NOT_FOUND)
    manifest = _load_json(observer_root / OBSERVER_ARTIFACTS["manifest"])
    _require_schema(manifest, OBSERVER_MANIFEST_SCHEMA, OBSERVER_ARTIFACTS["manifest"])
    if manifest.get("presentation_only") is not True or str(manifest.get("audience_profile")) != profile:
        raise ChronicleReadError("Observer manifest does not match the selected profile", code="schema_mismatch")
    if str(manifest.get("run_id")) != str(scientific_manifest.get("run_id")):
        raise ChronicleReadError("Observer and Scientific Chronicle run identities differ", code="traceability_mismatch")
    source_inventory = list(manifest.get("source_chronicle_artifacts", []) or [])
    if {
        str(artifact.get("relative_path", ""))
        for artifact in source_inventory
        if isinstance(artifact, dict)
    } != set(ARTIFACTS.values()):
        raise ChronicleReadError("Observer source inventory is incomplete", code="traceability_mismatch")
    for artifact in source_inventory:
        if not isinstance(artifact, dict):
            raise ChronicleReadError("Observer source inventory is invalid", code="schema_mismatch")
        relative = _safe_relative(str(artifact.get("relative_path", "")))
        raw_source_path = chronicle_root / relative
        source_path = raw_source_path.resolve()
        if raw_source_path.is_symlink() or not _inside(source_path, chronicle_root) or not source_path.is_file():
            raise ChronicleReadError("Observer source link is unsafe or missing", code="traceability_mismatch")
        if _sha256(source_path) != str(artifact.get("sha256", "")):
            raise ChronicleReadError("Observer source hash no longer matches the Scientific Chronicle", code="source_hash_mismatch")
    observer_inventory = list(manifest.get("artifacts", []) or [])
    if {
        str(artifact.get("relative_path", ""))
        for artifact in observer_inventory
        if isinstance(artifact, dict)
    } != set(OBSERVER_HASHED_ARTIFACTS):
        raise ChronicleReadError("Observer artifact inventory is incomplete", code="traceability_mismatch")
    for artifact in observer_inventory:
        if not isinstance(artifact, dict):
            raise ChronicleReadError("Observer artifact inventory is invalid", code="schema_mismatch")
        relative = _safe_relative(str(artifact.get("relative_path", "")))
        raw_observer_path = observer_root / relative
        observer_path = raw_observer_path.resolve()
        if raw_observer_path.is_symlink() or not _inside(observer_path, observer_root) or not observer_path.is_file():
            raise ChronicleReadError("Observer artifact is unsafe or missing", code="traceability_mismatch")
        if _sha256(observer_path) != str(artifact.get("sha256", "")):
            raise ChronicleReadError("Observer artifact hash does not match its manifest", code="observer_hash_mismatch")
    return run_root, chronicle_root, observer_root, manifest, scientific_manifest, scientific_summary


def observer_overview(cfg: Any, run_key: str, profile: str) -> dict[str, Any]:
    run_root, _chronicle_root_value, observer_root, manifest, _scientific_manifest, scientific_summary = _resolve_observer(
        cfg, run_key, profile
    )
    world = _load_json(observer_root / OBSERVER_ARTIFACTS["world"])
    _require_schema(world, OBSERVER_WORLD_SCHEMA, OBSERVER_ARTIFACTS["world"])
    index = _load_json(observer_root / OBSERVER_ARTIFACTS["index"])
    _require_schema(index, OBSERVER_INDEX_SCHEMA, OBSERVER_ARTIFACTS["index"])
    return {
        "schema_version": "eiel_observer_explorer_overview_v1",
        "run_key": run_key,
        "audience_profile": profile,
        "available_profiles": _observer_profiles(run_root),
        "presentation_only": True,
        "authoritative_source": "Scientific Chronicle",
        "manifest": manifest,
        "world_summary": world,
        "index": index,
        "source_evidence": {
            "run_id": manifest.get("run_id"),
            "source_commit": manifest.get("source_commit"),
            "source_completeness": manifest.get("source_completeness"),
            "scientific_source_hashes_verified": True,
        },
        "limitations": list(world.get("limitations", []) or []) + list(scientific_summary.get("limitations", []) or []),
    }


def _observer_feed_matches(row: Mapping[str, Any], qs: Mapping[str, list[str]]) -> bool:
    generation = int(row.get("generation", 0))
    generation_start = _int_param(qs, "generation_start")
    generation_end = _int_param(qs, "generation_end")
    if generation_start is not None and generation < generation_start:
        return False
    if generation_end is not None and generation > generation_end:
        return False
    exact = {
        "category": str(row.get("event_category", "")),
        "event_type": str(row.get("event_type", "")),
    }
    links = dict(row.get("source_links", {}) or {})
    family_values = {str(value) for value in links.get("episode_ids", []) or []}
    for name, observed in exact.items():
        accepted = _values(qs, name)
        if accepted and observed not in accepted:
            return False
    accepted_families = _values(qs, "episode_family")
    if accepted_families and str(row.get("episode_family") or "") not in accepted_families:
        return False
    dimensions = dict(row.get("presentation_dimensions", {}) or {})
    minimum_public = _int_param(qs, "public_min")
    minimum_visual = _int_param(qs, "visual_min")
    if minimum_public is not None and int(dimensions.get("public_interest", 0)) < minimum_public:
        return False
    if minimum_visual is not None and int(dimensions.get("visual_interest", 0)) < minimum_visual:
        return False
    priority = _one(qs, "minimum_live_priority")
    if priority:
        if priority not in _OBSERVER_PRIORITY_RANK:
            raise ChronicleReadError("invalid minimum live priority", code="invalid_query", status=HTTPStatus.BAD_REQUEST)
        if _OBSERVER_PRIORITY_RANK.get(str(dimensions.get("live_priority", "background")), 4) > _OBSERVER_PRIORITY_RANK[priority]:
            return False
    references = dict(row.get("subject_references", {}) or {})
    agent_id = _one(qs, "agent_id")
    if agent_id and agent_id not in {str(value) for value in references.get("agent_ids", []) or []}:
        return False
    lineage_id = _one(qs, "lineage_id")
    if lineage_id and lineage_id not in {str(value) for value in references.get("lineage_ids", []) or []}:
        return False
    region_id = _one(qs, "region_id")
    if region_id and str(references.get("region_id") or "") != region_id:
        return False
    episode_id = _one(qs, "source_episode_id")
    if episode_id and episode_id not in family_values:
        return False
    return True


def _scientific_id_sets(chronicle_root: Path) -> dict[str, set[str]]:
    return {
        "event": {
            str(row["event_id"])
            for row in _iter_jsonl(chronicle_root / ARTIFACTS["events"], _validate_event)
        },
        "entry": {
            str(row["chronicle_entry_id"])
            for row in _iter_jsonl(chronicle_root / ARTIFACTS["entries"], _validate_entry)
        },
        "episode": {str(row["episode_id"]) for row in _load_episodes(chronicle_root)},
    }


def _validate_observer_source_links(
    kind: str,
    row: Mapping[str, Any],
    source_ids: Mapping[str, set[str]],
) -> None:
    if kind in {"feed", "episode"}:
        links = dict(row.get("source_links", {}) or {})
        event_ids = {str(value) for value in links.get("event_ids", []) or []}
        entry_ids = {str(value) for value in links.get("entry_ids", []) or []}
        episode_ids = {str(value) for value in links.get("episode_ids", []) or []}
        if links.get("episode_id"):
            episode_ids.add(str(links["episode_id"]))
    else:
        event_ids = {str(value) for value in row.get("source_event_ids", []) or []}
        entry_ids = set()
        episode_ids = {str(value) for value in row.get("source_episode_ids", []) or []}
    if not event_ids or not event_ids.issubset(source_ids["event"]):
        raise ChronicleReadError("Observer record references missing scientific events", code="traceability_mismatch")
    if entry_ids and not entry_ids.issubset(source_ids["entry"]):
        raise ChronicleReadError("Observer record references missing scientific entries", code="traceability_mismatch")
    if episode_ids and not episode_ids.issubset(source_ids["episode"]):
        raise ChronicleReadError("Observer record references missing scientific episodes", code="traceability_mismatch")


def observer_feed(cfg: Any, run_key: str, profile: str, qs: Mapping[str, list[str]]) -> dict[str, Any]:
    _run_root_value, chronicle_root, observer_root, manifest, _scientific_manifest, scientific_summary = _resolve_observer(
        cfg, run_key, profile
    )
    offset = _int_param(qs, "offset", 0) or 0
    limit = _int_param(qs, "limit", 50) or 50
    if offset < 0 or offset > _MAX_OFFSET or limit < 1 or limit > _MAX_PAGE:
        raise ChronicleReadError("paging bounds exceeded", code="invalid_paging", status=HTTPStatus.BAD_REQUEST)
    source_ids = _scientific_id_sets(chronicle_root)
    matches: list[dict[str, Any]] = []
    for row in _iter_jsonl(observer_root / OBSERVER_ARTIFACTS["feed"], _validate_observer("feed")):
        _validate_observer_source_links("feed", row, source_ids)
        if _observer_feed_matches(row, qs):
            matches.append(row)
    return {
        "schema_version": "eiel_observer_explorer_feed_v1",
        "run_key": run_key,
        "audience_profile": profile,
        "presentation_only": True,
        "authoritative_source": "Scientific Chronicle",
        "offset": offset,
        "limit": limit,
        "total": len(matches),
        "items": matches[offset : offset + limit],
        "source_completeness": manifest.get("source_completeness"),
        "limitations": list(scientific_summary.get("limitations", []) or []),
    }


def observer_story(cfg: Any, run_key: str, profile: str, kind: str, identifier: str) -> dict[str, Any]:
    if kind not in OBSERVER_SCHEMAS:
        raise ChronicleReadError("invalid Observer story kind", code="invalid_query", status=HTTPStatus.BAD_REQUEST)
    _run_root_value, chronicle_root, observer_root, manifest, _scientific_manifest, scientific_summary = _resolve_observer(
        cfg, run_key, profile
    )
    rows = list(_iter_jsonl(observer_root / OBSERVER_ARTIFACTS[kind], _validate_observer(kind)))
    source_ids = _scientific_id_sets(chronicle_root)
    for row in rows:
        _validate_observer_source_links(kind, row, source_ids)
    id_fields = {
        "feed": "observer_item_id",
        "episode": "observer_episode_id",
        "agent": "agent_id",
        "lineage": "lineage_id",
        "region": "region_id",
    }
    id_field = id_fields[kind]
    story = next((row for row in rows if str(row.get(id_field)) == identifier), None)
    if story is None and kind == "episode":
        story = next(
            (
                row
                for row in rows
                if str(dict(row.get("source_links", {}) or {}).get("episode_id", "")) == identifier
            ),
            None,
        )
    if story is None:
        raise ChronicleReadError("Observer story was not found", code="not_found", status=HTTPStatus.NOT_FOUND)
    return {
        "schema_version": "eiel_observer_explorer_story_v1",
        "run_key": run_key,
        "audience_profile": profile,
        "kind": kind,
        "presentation_only": True,
        "authoritative_source": "Scientific Chronicle",
        "story": story,
        "source_evidence": {
            "source_event_ids": list(story.get("source_event_ids", []) or [])
            if kind in {"agent", "lineage", "region"}
            else list(dict(story.get("source_links", {}) or {}).get("event_ids", []) or []),
            "source_episode_ids": list(story.get("source_episode_ids", []) or [])
            if kind in {"agent", "lineage", "region"}
            else [dict(story.get("source_links", {}) or {}).get("episode_id")]
            if kind == "episode"
            else [],
            "source_commit": manifest.get("source_commit"),
            "source_completeness": manifest.get("source_completeness"),
        },
        "limitations": list(story.get("limitations", []) or []) + list(scientific_summary.get("limitations", []) or []),
    }


def observer_stories(
    cfg: Any,
    run_key: str,
    profile: str,
    kind: str,
    qs: Mapping[str, list[str]],
) -> dict[str, Any]:
    if kind not in {"episode", "agent", "lineage", "region"}:
        raise ChronicleReadError("invalid Observer story kind", code="invalid_query", status=HTTPStatus.BAD_REQUEST)
    _run_root_value, chronicle_root, observer_root, manifest, _scientific_manifest, scientific_summary = _resolve_observer(
        cfg, run_key, profile
    )
    offset = _int_param(qs, "offset", 0) or 0
    limit = _int_param(qs, "limit", 50) or 50
    if offset < 0 or offset > _MAX_OFFSET or limit < 1 or limit > _MAX_PAGE:
        raise ChronicleReadError("paging bounds exceeded", code="invalid_paging", status=HTTPStatus.BAD_REQUEST)
    rows = list(_iter_jsonl(observer_root / OBSERVER_ARTIFACTS[kind], _validate_observer(kind)))
    source_ids = _scientific_id_sets(chronicle_root)
    for row in rows:
        _validate_observer_source_links(kind, row, source_ids)
    return {
        "schema_version": "eiel_observer_explorer_story_list_v1",
        "run_key": run_key,
        "audience_profile": profile,
        "kind": kind,
        "presentation_only": True,
        "authoritative_source": "Scientific Chronicle",
        "offset": offset,
        "limit": limit,
        "total": len(rows),
        "items": rows[offset : offset + limit],
        "source_completeness": manifest.get("source_completeness"),
        "limitations": list(scientific_summary.get("limitations", []) or []),
    }


def run_overview(cfg: Any, run_key: str) -> dict[str, Any]:
    run_root, root, manifest, summary = _resolve_run(cfg, run_key)
    storage = _load_json(root / ARTIFACTS["storage"])
    episodes = _load_episodes(root)
    category_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    important: list[dict[str, Any]] = []
    first_coordinate: dict[str, int] | None = None
    last_coordinate: dict[str, int] | None = None
    latest_population: dict[str, Any] = {}
    event_schemas: set[str] = set()
    for event in _iter_jsonl(root / ARTIFACTS["events"], _validate_event):
        event_schemas.add(str(event["schema_version"]))
        category_counts[str(event.get("event_category", "unknown"))] += 1
        priority_counts[str(event.get("live_priority", "unknown"))] += 1
        coordinate = {"generation": int(event.get("generation", 0)), "step": int(event.get("step", 0))}
        first_coordinate = first_coordinate or coordinate
        last_coordinate = coordinate
        if str(event.get("event_type")) == "generation_summary":
            latest_population = dict(event.get("measurements", {}) or {})
        if str(event.get("live_priority")) in {"critical", "high"} and len(important) < 50:
            important.append({
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "generation": event.get("generation"),
                "step": event.get("step"),
                "severity": event.get("severity"),
                "live_priority": event.get("live_priority"),
                "subject_id": event.get("subject_id"),
            })
    episode_counts = Counter(str(episode.get("episode_type", "unknown")) for episode in episodes)
    return {
        "schema_version": "eiel_chronicle_explorer_overview_v1",
        "run_key": run_key,
        "identity": {
            "run_id": manifest.get("run_id"),
            "campaign_id": manifest.get("campaign_id"),
            "branch_id": manifest.get("branch_id"),
            "parent_checkpoint_id": manifest.get("parent_checkpoint_id"),
            "source_commit": manifest.get("source_commit"),
        },
        "profile": manifest.get("profile", summary.get("profile")),
        "source_completeness": manifest.get("source_completeness", summary.get("source_completeness", "unknown")),
        "finalized": bool(manifest.get("finalized", summary.get("finalized", False))),
        "retrospective": str(manifest.get("profile", "")) == "retrospective",
        "observer_profiles": _observer_profiles(run_root),
        "documentary_available": _documentary_available(run_root),
        "schema_versions": {
            "manifest": manifest.get("schema_version"),
            "formal_events": sorted(event_schemas),
            "episodes": sorted({str(episode.get("schema_version", "")) for episode in episodes}),
        },
        "generation_range": {"first": first_coordinate, "last": last_coordinate},
        "population_summary": latest_population,
        "event_count": int(summary.get("event_count", 0) or 0),
        "event_counts_by_category": dict(sorted(category_counts.items())),
        "event_counts_by_type": dict(sorted(dict(summary.get("event_counts_by_type", {}) or {}).items())),
        "event_counts_by_priority": dict(sorted(priority_counts.items())),
        "episode_count": len(episodes),
        "episode_counts_by_family": dict(sorted(episode_counts.items())),
        "open_episode_count": sum(1 for episode in episodes if str(episode.get("status")) == "open"),
        "final_episode_count": sum(1 for episode in episodes if str(episode.get("status")) == "closed"),
        "open_episodes": [episode for episode in episodes if str(episode.get("status")) == "open"][-50:],
        "final_episodes": [episode for episode in episodes if str(episode.get("status")) == "closed"][-50:],
        "important_events": important,
        "suppressed_counts": dict(summary.get("suppressed_counts", {}) or {}),
        "storage": storage,
        "limitations": list(summary.get("limitations", []) or []) + list(manifest.get("limitations", []) or []),
    }


def documentary_overview(cfg: Any, run_key: str) -> dict[str, Any]:
    _run_root, _chronicle_root_value, documentary_root, manifest, index, scientific_summary = _resolve_documentary(cfg, run_key)
    script = _load_json(documentary_root / DOCUMENTARY_ARTIFACTS["script"])
    _require_schema(script, DOCUMENTARY_SCRIPT_SCHEMA, DOCUMENTARY_ARTIFACTS["script"])
    chapters = _load_json(documentary_root / DOCUMENTARY_ARTIFACTS["chapters"])
    citations = _load_json(documentary_root / DOCUMENTARY_ARTIFACTS["citations"])
    verifier = _load_json(documentary_root / DOCUMENTARY_ARTIFACTS["verifier"])
    plan = _load_json(documentary_root / DOCUMENTARY_ARTIFACTS["plan"])
    transcript_path = documentary_root / DOCUMENTARY_ARTIFACTS["transcript"]
    if transcript_path.is_symlink() or not _inside(transcript_path.resolve(), documentary_root) or not transcript_path.is_file():
        raise ChronicleReadError("POD1 transcript is unsafe or missing", code="traceability_mismatch")
    return {
        "schema_version": "eiel_two_host_documentary_explorer_v1",
        "run_key": run_key,
        "run_id": manifest.get("run_id"),
        "presentation_only": True,
        "authoritative_source": "Scientific Chronicle",
        "documentary_root": str(documentary_root),
        "manifest": manifest,
        "index": index,
        "plan": plan,
        "script_summary": {
            "generation_mode": script.get("generation_mode"),
            "word_count": script.get("estimated_word_count"),
            "estimated_duration_seconds": script.get("estimated_duration_seconds"),
            "utterance_count": len(script.get("utterances", []) or []),
            "speakers": script.get("speakers"),
        },
        "chapters": chapters,
        "citations": citations,
        "verifier": verifier,
        "utterances": list(script.get("utterances", []) or [])[:_MAX_PAGE],
        "transcript": transcript_path.read_text(encoding="utf-8"),
        "host_metrics": index.get("host_metrics", manifest.get("host_metrics", {})),
        "limitations": [
            "POD1 is a derived public documentary artifact; the Scientific Chronicle remains authoritative.",
            "NotebookLM output, if used manually, is not scientific evidence and must pass verification before acceptance.",
            *list(scientific_summary.get("limitations", []) or []),
        ],
    }


def _one(qs: Mapping[str, list[str]], name: str, default: str = "") -> str:
    return str((qs.get(name) or [default])[0] or default).strip()


def _int_param(qs: Mapping[str, list[str]], name: str, default: int | None = None) -> int | None:
    value = _one(qs, name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ChronicleReadError(f"{name} must be an integer", code="invalid_query", status=HTTPStatus.BAD_REQUEST) from exc


def _values(qs: Mapping[str, list[str]], name: str) -> set[str]:
    return {item.strip() for raw in qs.get(name, []) for item in str(raw).split(",") if item.strip()}


def _patch_value(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{int(value[0])},{int(value[1])}"
    return ""


def _event_matches(
    event: Mapping[str, Any],
    entry: Mapping[str, Any],
    episode: Mapping[str, Any] | None,
    qs: Mapping[str, list[str]],
    source_completeness: str,
) -> bool:
    if _one(qs, "item_kind") == "episode":
        return False
    generation = int(event.get("generation", 0))
    step = int(event.get("step", 0))
    scalar_bounds = {
        "scientific": int(event.get("scientific_importance", 0)),
        "historical": int(event.get("historical_importance", 0)),
        "public": int(event.get("public_interest", 0)),
        "visual": int(event.get("visual_interest", 0)),
    }
    for name, observed in (("generation", generation), ("step", step)):
        low = _int_param(qs, f"{name}_start")
        high = _int_param(qs, f"{name}_end")
        if low is not None and observed < low:
            return False
        if high is not None and observed > high:
            return False
    exact_fields = {
        "category": str(event.get("event_category", "")),
        "event_type": str(event.get("event_type", "")),
        "severity": str(event.get("severity", "")),
        "priority": str(event.get("live_priority", "")),
        "confidence": str(event.get("confidence_class", "")),
        "causality": str(event.get("causality_class", "")),
        "source_completeness": source_completeness,
        "region_id": str(event.get("region_id") or ""),
        "patch": _patch_value(event.get("patch_coordinates")),
    }
    for name, observed in exact_fields.items():
        accepted = _values(qs, name)
        if accepted and observed not in accepted:
            return False
    for name, observed in scalar_bounds.items():
        low = _int_param(qs, f"{name}_min")
        high = _int_param(qs, f"{name}_max")
        if low is not None and observed < low:
            return False
        if high is not None and observed > high:
            return False
    agent_id = _int_param(qs, "agent_id")
    if agent_id is not None and agent_id not in {int(value) for value in event.get("agent_ids", []) or []}:
        return False
    lineage = _one(qs, "lineage_id")
    if lineage and lineage not in {str(value) for value in event.get("lineage_ids", []) or []}:
        return False
    tag = _one(qs, "tag")
    if tag and tag not in {str(value) for value in event.get("tags", []) or []}:
        return False
    family = _one(qs, "episode_family")
    if family and (episode is None or str(episode.get("episode_type")) != family):
        return False
    episode_status = _one(qs, "episode_status")
    if episode_status and (episode is None or str(episode.get("status")) != episode_status):
        return False
    return True


def _timeline_entry_items(
    root: Path,
    episodes_by_id: Mapping[str, Mapping[str, Any]],
    qs: Mapping[str, list[str]],
    source_completeness: str,
) -> Iterator[tuple[tuple[Any, ...], dict[str, Any]]]:
    event_iter = _iter_jsonl(root / ARTIFACTS["events"], _validate_event)
    entry_iter = _iter_jsonl(root / ARTIFACTS["entries"], _validate_entry)
    sentinel = object()
    for event, entry in itertools.zip_longest(event_iter, entry_iter, fillvalue=sentinel):
        if event is sentinel or entry is sentinel:
            raise ChronicleReadError("formal event and Chronicle entry feeds have different lengths", code="traceability_mismatch")
        assert isinstance(event, dict) and isinstance(entry, dict)
        source_ids = [str(value) for value in entry.get("source_event_ids", []) or []]
        if str(event.get("event_id")) not in source_ids:
            raise ChronicleReadError("Chronicle entry is not traceable to its paired formal event", code="traceability_mismatch")
        episode = episodes_by_id.get(str(entry.get("episode_id") or ""))
        if not _event_matches(event, entry, episode, qs, source_completeness):
            continue
        important = bool(
            str(event.get("live_priority")) in {"critical", "high"}
            or str(event.get("severity")) in {"critical", "major"}
            or int(event.get("scientific_importance", 0)) >= 4
            or int(event.get("historical_importance", 0)) >= 4
        )
        item = {
            "kind": "generation_summary" if str(event.get("event_type")) == "generation_summary" else "chronicle_entry",
            "entry": entry,
            "formal_event": event,
            "episode": ({
                "episode_id": episode.get("episode_id"),
                "episode_type": episode.get("episode_type"),
                "status": episode.get("status"),
            } if episode else None),
            "important_formal_event": important,
        }
        key = (
            int(event.get("generation", 0)), int(event.get("step", 0)),
            int(event.get("phase_order", 0)), int(event.get("intra_step_sequence", 0)),
            1, str(event.get("event_id", "")),
        )
        yield key, item


def _episode_matches(episode: Mapping[str, Any], qs: Mapping[str, list[str]], source_completeness: str) -> bool:
    if _one(qs, "item_kind") in {"entry", "event"}:
        return False
    if _values(qs, "source_completeness") and source_completeness not in _values(qs, "source_completeness"):
        return False
    family = _one(qs, "episode_family")
    if family and str(episode.get("episode_type")) != family:
        return False
    status = _one(qs, "episode_status")
    if status and str(episode.get("status")) != status:
        return False
    generation_start = _int_param(qs, "generation_start")
    generation_end = _int_param(qs, "generation_end")
    if generation_start is not None and int(episode.get("end_generation", 0)) < generation_start:
        return False
    if generation_end is not None and int(episode.get("start_generation", 0)) > generation_end:
        return False
    agent_id = _int_param(qs, "agent_id")
    if agent_id is not None and agent_id not in {int(value) for value in episode.get("agent_ids", []) or []}:
        return False
    lineage = _one(qs, "lineage_id")
    if lineage and lineage not in {str(value) for value in episode.get("lineage_ids", []) or []}:
        return False
    region = _one(qs, "region_id")
    if region and str(episode.get("affected_region") or "") != region:
        return False
    event_only_filters = {
        "category", "event_type", "severity", "priority", "confidence", "causality", "patch", "tag",
        "scientific_min", "scientific_max", "historical_min", "historical_max",
        "public_min", "public_max", "visual_min", "visual_max", "step_start", "step_end",
    }
    return not any(_one(qs, name) for name in event_only_filters)


def timeline(cfg: Any, run_key: str, qs: Mapping[str, list[str]]) -> dict[str, Any]:
    _run_root, root, manifest, summary = _resolve_run(cfg, run_key)
    episodes = _load_episodes(root)
    episodes_by_id = {str(episode["episode_id"]): episode for episode in episodes}
    completeness = str(manifest.get("source_completeness", summary.get("source_completeness", "unknown")))
    episode_items = (
        (
            (
                int(episode.get("start_generation", 0)), int(episode.get("start_step", 0)),
                -1, -1, 0, str(episode.get("episode_id", "")),
            ),
            {"kind": "episode", "episode": episode},
        )
        for episode in episodes
        if _episode_matches(episode, qs, completeness)
    )
    merged = heapq.merge(
        _timeline_entry_items(root, episodes_by_id, qs, completeness),
        episode_items,
        key=lambda pair: pair[0],
    )
    offset = _int_param(qs, "offset", 0) or 0
    limit = _int_param(qs, "limit", 50) or 50
    if offset < 0 or offset > _MAX_OFFSET or limit < 1 or limit > _MAX_PAGE:
        raise ChronicleReadError("paging bounds exceeded", code="invalid_paging", status=HTTPStatus.BAD_REQUEST)
    items: list[dict[str, Any]] = []
    total = 0
    for _key, item in merged:
        if offset <= total < offset + limit:
            items.append(item)
        total += 1
    return {
        "schema_version": "eiel_chronicle_explorer_timeline_v1",
        "run_key": run_key,
        "source_completeness": completeness,
        "offset": offset,
        "limit": limit,
        "total": total,
        "items": items,
        "limitations": list(summary.get("limitations", []) or []),
    }


def _events_by_ids(root: Path, wanted: set[str], *, maximum: int = _MAX_DETAIL_EVENTS) -> list[dict[str, Any]]:
    if len(wanted) > maximum:
        raise ChronicleReadError("detail request exceeds source event bound", code="detail_too_large", status=HTTPStatus.BAD_REQUEST)
    found: dict[str, dict[str, Any]] = {}
    for event in _iter_jsonl(root / ARTIFACTS["events"], _validate_event):
        event_id = str(event.get("event_id"))
        if event_id in wanted:
            found[event_id] = event
    missing = sorted(wanted.difference(found))
    if missing:
        raise ChronicleReadError(f"source event references are missing: {', '.join(missing[:5])}", code="traceability_mismatch")
    return [found[event_id] for event_id in wanted if event_id in found]


def _load_contexts(root: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(root / ARTIFACTS["contexts"], _validate_context))


def episode_detail(cfg: Any, run_key: str, episode_id: str) -> dict[str, Any]:
    _run_root, root, manifest, summary = _resolve_run(cfg, run_key)
    episode = next((value for value in _load_episodes(root) if str(value.get("episode_id")) == episode_id), None)
    if episode is None:
        raise ChronicleReadError("episode not found", code="not_found", status=HTTPStatus.NOT_FOUND)
    source_ids = [str(value) for value in episode.get("source_event_ids", []) or []]
    events = _events_by_ids(root, set(source_ids))
    by_id = {str(event["event_id"]): event for event in events}
    ordered_events = [by_id[event_id] for event_id in source_ids]
    summary_ids = [str(event["event_id"]) for event in ordered_events if str(event.get("event_type")) == "generation_summary"]
    contexts = [
        context for context in _load_contexts(root)
        if set(str(value) for value in context.get("source_event_ids", []) or []).intersection(source_ids)
    ]
    return {
        "schema_version": "eiel_chronicle_explorer_episode_detail_v1",
        "run_key": run_key,
        "source_completeness": manifest.get("source_completeness", summary.get("source_completeness", "unknown")),
        "episode": episode,
        "source_events": ordered_events,
        "supporting_generation_summary_ids": summary_ids,
        "supporting_summary_semantics": "references_to_formal_events_not_duplicated_evidence",
        "context_windows": contexts,
        "limitations": list(episode.get("limitations", []) or []) + list(summary.get("limitations", []) or []),
    }


def _subject_event_match(event: Mapping[str, Any], kind: str, identifier: str) -> bool:
    if kind == "agent":
        try:
            wanted = int(identifier)
        except ValueError:
            return False
        if wanted in {int(value) for value in event.get("agent_ids", []) or []}:
            return True
        measurements = dict(event.get("measurements", {}) or {})
        return wanted in {
            int(value) for key, value in measurements.items()
            if key in {"parent_agent_id", "mate_agent_id", "child_agent_id"} and str(value).lstrip("-").isdigit()
        }
    if kind == "lineage":
        return identifier in {str(value) for value in event.get("lineage_ids", []) or []} or (
            str(event.get("subject_type")) == "lineage" and str(event.get("subject_id")) == identifier
        )
    if kind == "region":
        return str(event.get("region_id") or "") == identifier
    if kind == "patch":
        return _patch_value(event.get("patch_coordinates")) == identifier
    return False


def _genome_links(run_root: Path, kind: str, identifier: str) -> list[dict[str, Any]]:
    candidates = (
        run_root / "chronicle" / "genome_attribution" / "mutation_events.jsonl",
        run_root / "genome_attribution" / "mutation_events.jsonl",
    )
    path = next(
        (
            value for value in candidates
            if value.is_file() and not value.is_symlink() and _inside(value, run_root)
        ),
        None,
    )
    if path is None or path.stat().st_size > _MAX_JSONL_BYTES:
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict) or str(row.get("schema_version")) != "eiel_mutation_event_v1":
                    continue
                match = False
                if kind == "agent":
                    match = str(row.get("child_agent_id")) == identifier or identifier in {str(value) for value in row.get("parent_agent_ids", []) or []}
                elif kind == "lineage":
                    match = str(row.get("lineage_id")) == identifier
                if match:
                    rows.append(row)
                    if len(rows) >= 200:
                        break
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    return rows


def _gen1_companion(run_root: Path, run_id: str) -> tuple[Path | None, dict[str, Any]]:
    candidates = (
        run_root / "chronicle" / "gen1_readiness" / "chronicle_companion_index.json",
        run_root / "gen1_readiness" / "chronicle_companion_index.json",
        run_root / "chronicle_companion_index.json",
    )
    path = next((value for value in candidates if value.is_file() and not value.is_symlink()), None)
    if path is None:
        return None, {}
    try:
        payload = _load_json(path)
    except ChronicleReadError:
        return None, {}
    if str(payload.get("schema_version")) != GEN1_COMPANION_SCHEMA:
        return None, {}
    run = dict(dict(payload.get("runs", {}) or {}).get(str(run_id), {}) or {})
    return path, run


def _gen1_subject_links(run_root: Path, run_id: str, kind: str, identifier: str) -> dict[str, Any]:
    path, run = _gen1_companion(run_root, run_id)
    if path is None:
        return {"status": "unavailable", "reason": "external GEN1 readiness companion not present"}
    mutation_key = "agent_mutations" if kind == "agent" else "lineage_mutations" if kind == "lineage" else ""
    profile_key = "agent_profiles" if kind == "agent" else "lineage_profiles" if kind == "lineage" else ""
    mutation_ids = list(dict(run.get(mutation_key, {}) or {}).get(identifier, []) or []) if mutation_key else []
    profile_ids = list(dict(run.get(profile_key, {}) or {}).get(identifier, []) or []) if profile_key else []
    return {
        "status": "available",
        "attachment_mode": "external_read_only_companion",
        "companion_path": str(path),
        "mutation_event_ids": mutation_ids[:_MAX_DETAIL_EVENTS],
        "success_profile_ids": profile_ids[:_MAX_DETAIL_EVENTS],
        "truncated": len(mutation_ids) > _MAX_DETAIL_EVENTS or len(profile_ids) > _MAX_DETAIL_EVENTS,
    }


def _gen1_event_links(run_root: Path, run_id: str, event_id: str) -> dict[str, Any]:
    path, run = _gen1_companion(run_root, run_id)
    if path is None:
        return {"status": "unavailable", "reason": "external GEN1 readiness companion not present"}
    by_event = dict(run.get("chronicle_event_mutations", {}) or {})
    by_birth = dict(run.get("birth_event_mutations", {}) or {})
    mutation_ids = sorted({str(value) for value in [*by_event.get(event_id, []), *by_birth.get(event_id, [])]})
    return {
        "status": "available",
        "attachment_mode": "external_read_only_companion",
        "companion_path": str(path),
        "mutation_event_ids": mutation_ids[:_MAX_DETAIL_EVENTS],
        "truncated": len(mutation_ids) > _MAX_DETAIL_EVENTS,
    }


def subject_detail(cfg: Any, run_key: str, kind: str, identifier: str) -> dict[str, Any]:
    if kind not in {"agent", "lineage", "region", "patch"}:
        raise ChronicleReadError("invalid subject kind", code="invalid_query", status=HTTPStatus.BAD_REQUEST)
    run_root, root, manifest, summary = _resolve_run(cfg, run_key)
    events = [
        event for event in _iter_jsonl(root / ARTIFACTS["events"], _validate_event)
        if _subject_event_match(event, kind, identifier)
    ]
    if len(events) > _MAX_DETAIL_EVENTS:
        events = events[:_MAX_DETAIL_EVENTS]
        truncated = True
    else:
        truncated = False
    event_ids = {str(event["event_id"]) for event in events}
    episodes = [
        episode for episode in _load_episodes(root)
        if event_ids.intersection(str(value) for value in episode.get("source_event_ids", []) or [])
        or (kind == "agent" and identifier in {str(value) for value in episode.get("agent_ids", []) or []})
        or (kind == "lineage" and identifier in {str(value) for value in episode.get("lineage_ids", []) or []})
        or (kind in {"region", "patch"} and str(episode.get("affected_region") or "") == identifier)
    ]
    contexts = [
        context for context in _load_contexts(root)
        if event_ids.intersection(str(value) for value in context.get("source_event_ids", []) or [])
    ]
    return {
        "schema_version": "eiel_chronicle_explorer_subject_detail_v1",
        "run_key": run_key,
        "subject": {"kind": kind, "id": identifier},
        "source_completeness": manifest.get("source_completeness", summary.get("source_completeness", "unknown")),
        "events": events,
        "episodes": episodes,
        "context_windows": contexts,
        "genome_links": _genome_links(run_root, kind, identifier),
        "gen1_readiness_links": _gen1_subject_links(
            run_root, str(manifest.get("run_id") or "unknown"), kind, identifier
        ),
        "partial_coverage": True,
        "truncated": truncated,
        "limitations": [
            "This page contains only subjects referenced by bounded Chronicle records; it is not a complete biography or world trace.",
            *list(summary.get("limitations", []) or []),
        ],
    }


def context_detail(cfg: Any, run_key: str, context_id: str = "", event_id: str = "") -> dict[str, Any]:
    _run_root, root, manifest, summary = _resolve_run(cfg, run_key)
    contexts = _load_contexts(root)
    context = next((value for value in contexts if context_id and str(value.get("context_window_id")) == context_id), None)
    if context is None and event_id:
        context = next(
            (value for value in contexts if event_id in {str(item) for item in value.get("source_event_ids", []) or []}),
            None,
        )
    if context is None:
        raise ChronicleReadError("context window not found", code="not_found", status=HTTPStatus.NOT_FOUND)
    source_ids = [str(value) for value in context.get("source_event_ids", []) or []]
    events = _events_by_ids(root, set(source_ids))
    by_id = {str(event["event_id"]): event for event in events}
    trigger_ids = {str(value) for value in context.get("trigger_event_ids", []) or []}
    trigger_indexes = [index for index, value in enumerate(source_ids) if value in trigger_ids]
    first_trigger = min(trigger_indexes, default=0)
    last_trigger = max(trigger_indexes, default=first_trigger)
    ordered = [
        {
            "position": (
                "trigger" if event_id_value in trigger_ids
                else "before" if index < first_trigger
                else "after" if index > last_trigger
                else "between_triggers"
            ),
            "event": by_id[event_id_value],
        }
        for index, event_id_value in enumerate(source_ids)
    ]
    return {
        "schema_version": "eiel_chronicle_explorer_context_detail_v1",
        "run_key": run_key,
        "label": "formal-event context window",
        "context_window": context,
        "events": ordered,
        "source_completeness": manifest.get("source_completeness", summary.get("source_completeness", "unknown")),
        "limitations": list(context.get("limitations", []) or []) + [
            "This is a bounded formal-event context window, not a full step/action Simulation Trace."
        ],
    }


def event_detail(cfg: Any, run_key: str, event_id: str) -> dict[str, Any]:
    run_root, root, manifest, summary = _resolve_run(cfg, run_key)
    events = _events_by_ids(root, {event_id}, maximum=1)
    event = events[0]
    contexts = [
        context for context in _load_contexts(root)
        if event_id in {str(value) for value in context.get("source_event_ids", []) or []}
    ]
    return {
        "schema_version": "eiel_chronicle_explorer_formal_event_detail_v1",
        "run_key": run_key,
        "formal_event": event,
        "context_windows": contexts,
        "gen1_readiness_links": _gen1_event_links(
            run_root, str(manifest.get("run_id") or "unknown"), event_id
        ),
        "source_completeness": manifest.get("source_completeness", summary.get("source_completeness", "unknown")),
        "limitations": list(summary.get("limitations", []) or []),
    }


def _comparison_key(index: int, root: Path, directory: Path) -> str:
    relative = directory.resolve().relative_to(root.resolve()).as_posix()
    return f"root{index}/{relative if relative != '.' else '__root__'}"


def discover_comparisons(cfg: Any) -> list[dict[str, Any]]:
    roots = _approved_roots(cfg)
    depth = max(1, min(32, int(getattr(cfg, "chronicle_scan_depth", 8) or 8)))
    rows: list[dict[str, Any]] = []
    for index, root in enumerate(roots):
        for path in _walk_for(root, "comparison_summary.json", max_depth=depth):
            key = _comparison_key(index, root, path.parent)
            try:
                summary = _load_json(path)
                _require_schema(summary, COMPARISON_SCHEMA, path.name)
                rows.append({"comparison_key": key, "status": "ready", "summary": summary})
            except ChronicleReadError as exc:
                rows.append({"comparison_key": key, "status": "invalid", "error_code": exc.code, "error": str(exc)})
    return sorted(rows, key=lambda row: str(row["comparison_key"]))


def _resolve_comparison(cfg: Any, comparison_key: str) -> Path:
    root, directory = _parse_root_key(comparison_key, _approved_roots(cfg))
    if not _inside(directory, root) or directory.is_symlink():
        raise ChronicleReadError("comparison path escapes approved root", code="path_traversal", status=HTTPStatus.BAD_REQUEST)
    return directory


def _first_raw_difference(cfg: Any, summary: Mapping[str, Any]) -> dict[str, Any]:
    runs = [row for row in discover_runs(cfg) if row.get("status") == "ready"]
    by_id: dict[str, list[str]] = {}
    for row in runs:
        by_id.setdefault(str(row.get("run_id")), []).append(str(row["run_key"]))
    keys_a = by_id.get(str(summary.get("run_a_id")), [])
    keys_b = by_id.get(str(summary.get("run_b_id")), [])
    base = {
        "schema_version": "eiel_chronicle_first_raw_difference_v1",
        "status": "unavailable",
        "disclaimer": "A raw difference may be scientifically immaterial and never replaces meaningful divergence.",
    }
    if len(keys_a) != 1 or len(keys_b) != 1:
        return {**base, "reason": "run IDs do not resolve uniquely within approved roots"}
    _ra, root_a, _ma, _sa = _resolve_run(cfg, keys_a[0])
    _rb, root_b, _mb, _sb = _resolve_run(cfg, keys_b[0])
    events_a = _iter_jsonl(root_a / ARTIFACTS["events"], _validate_event)
    events_b = _iter_jsonl(root_b / ARTIFACTS["events"], _validate_event)
    sentinel = object()
    for index, (event_a, event_b) in enumerate(itertools.zip_longest(events_a, events_b, fillvalue=sentinel)):
        if event_a is sentinel or event_b is sentinel:
            return {**base, "status": "different", "index": index, "event_a": None if event_a is sentinel else event_a, "event_b": None if event_b is sentinel else event_b}
        assert isinstance(event_a, dict) and isinstance(event_b, dict)
        if json.dumps(event_a, sort_keys=True, separators=(",", ":"), ensure_ascii=True) != json.dumps(event_b, sort_keys=True, separators=(",", ":"), ensure_ascii=True):
            return {**base, "status": "different", "index": index, "event_a": event_a, "event_b": event_b}
    return {**base, "status": "identical", "reason": "formal event feeds are byte-equivalent after canonical JSON parsing"}


def comparison_detail(cfg: Any, comparison_key: str, *, include_raw: bool) -> dict[str, Any]:
    root = _resolve_comparison(cfg, comparison_key)
    summary = _load_json(root / "comparison_summary.json")
    _require_schema(summary, COMPARISON_SCHEMA, "comparison_summary.json")
    matched = _load_json(root / "matched_episodes.json", expected_type=list)
    unmatched = _load_json(root / "unmatched_episodes.json")
    meaningful = _load_json(root / "first_meaningful_divergence.json")
    metrics = _load_json(root / "key_metric_context.json")
    discovered = [row for row in discover_runs(cfg) if row.get("status") == "ready"]
    provenance = {
        "run_a": [row for row in discovered if str(row.get("run_id")) == str(summary.get("run_a_id"))],
        "run_b": [row for row in discovered if str(row.get("run_id")) == str(summary.get("run_b_id"))],
    }
    return {
        "schema_version": "eiel_chronicle_explorer_comparison_detail_v1",
        "comparison_key": comparison_key,
        "default_view": "meaningful_divergence",
        "summary": summary,
        "run_provenance": provenance,
        "matched_episodes": matched,
        "unmatched_episodes": unmatched,
        "first_meaningful_divergence": meaningful,
        "key_metric_context": metrics,
        "first_raw_difference": _first_raw_difference(cfg, summary) if include_raw else None,
        "raw_difference_disclaimer": "The optional raw-difference view may show immaterial differences and does not replace the scientific default.",
        "causal_status": "not_assessed_beyond_source_event_classifications",
        "limitations": list(summary.get("limitations", []) or []),
    }


def _validate_live_item(record: Mapping[str, Any], artifact: str) -> None:
    _require_schema(record, LIVE_ITEM_SCHEMA, artifact)
    if (
        not str(record.get("observer_item_id", "")).startswith("obs_")
        or str(record.get("audience_profile")) not in OBSERVER_PROFILES
        or record.get("presentation_only") is not True
    ):
        raise ChronicleReadError(f"invalid LIVE1 Observer item in {artifact}", code="schema_mismatch")
    links = dict(record.get("source_links", {}) or {})
    if not links.get("event_ids") or not links.get("entry_ids"):
        raise ChronicleReadError(f"LIVE1 item lacks source links in {artifact}", code="traceability_mismatch")


def _validate_live_episode(record: Mapping[str, Any], artifact: str) -> None:
    _require_schema(record, LIVE_EPISODE_SCHEMA, artifact)
    links = dict(record.get("source_links", {}) or {})
    if record.get("presentation_only") is not True or not links.get("event_ids"):
        raise ChronicleReadError(f"LIVE1 episode lacks source links in {artifact}", code="traceability_mismatch")


def discover_live_runs(cfg: Any) -> list[dict[str, Any]]:
    roots = _approved_live_roots(cfg)
    depth = max(1, min(32, int(getattr(cfg, "chronicle_scan_depth", 8) or 8)))
    rows: list[dict[str, Any]] = []
    for index, root in enumerate(roots):
        for manifest_path in _walk_for(root, LIVE_ARTIFACTS["manifest"], max_depth=depth):
            live_root = manifest_path.parent
            live_key = _relative_key(index, root, live_root)
            try:
                manifest = _load_json(manifest_path)
                _require_schema(manifest, LIVE_MANIFEST_SCHEMA, LIVE_ARTIFACTS["manifest"])
                status = _load_json(live_root / LIVE_ARTIFACTS["status"])
                rows.append(
                    {
                        "live_key": live_key,
                        "status": str(status.get("status", "unknown")),
                        "run_id": str(manifest.get("run_id") or live_root.name),
                        "source_identity": dict(manifest.get("source_identity", {}) or {}),
                        "source_event_count": int(manifest.get("source_event_count", 0) or 0),
                        "source_episode_count": int(manifest.get("source_episode_count", 0) or 0),
                        "audience_profiles": list(manifest.get("audience_profiles", []) or []),
                        "read_only_external_consumer": manifest.get("read_only_external_consumer") is True,
                    }
                )
            except ChronicleReadError as exc:
                rows.append({"live_key": live_key, "status": "invalid", "error_code": exc.code, "error": str(exc)})
    return sorted(rows, key=lambda row: (str(row.get("run_id", "")), str(row["live_key"])))


def _resolve_live(
    cfg: Any,
    live_key: str,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    roots = _approved_live_roots(cfg)
    if not roots:
        raise ChronicleReadError("no LIVE1 companion roots are configured", code="not_found", status=HTTPStatus.NOT_FOUND)
    approved, live_root = _parse_root_key(live_key, roots)
    if not _inside(live_root, approved) or live_root.is_symlink():
        raise ChronicleReadError("LIVE1 path escapes approved root", code="path_traversal", status=HTTPStatus.BAD_REQUEST)
    manifest = _load_json(live_root / LIVE_ARTIFACTS["manifest"])
    _require_schema(manifest, LIVE_MANIFEST_SCHEMA, LIVE_ARTIFACTS["manifest"])
    if manifest.get("read_only_external_consumer") is not True or manifest.get("source_writes_attempted") is not False:
        raise ChronicleReadError("LIVE1 read-only boundary is not declared", code="schema_mismatch")
    inventory = list(manifest.get("artifacts", []) or [])
    observed_paths = {
        str(row.get("relative_path", "")) for row in inventory if isinstance(row, dict)
    }
    if observed_paths != LIVE_HASHED_ARTIFACTS:
        raise ChronicleReadError("LIVE1 artifact inventory is incomplete", code="traceability_mismatch")
    for artifact in inventory:
        if not isinstance(artifact, dict):
            raise ChronicleReadError("LIVE1 artifact inventory is invalid", code="schema_mismatch")
        relative = _safe_relative(str(artifact.get("relative_path", "")))
        raw_path = live_root / relative
        path = raw_path.resolve()
        if raw_path.is_symlink() or not _inside(path, live_root) or not path.is_file():
            raise ChronicleReadError("LIVE1 artifact is unsafe or missing", code="traceability_mismatch")
        if _sha256(path) != str(artifact.get("sha256", "")):
            raise ChronicleReadError("LIVE1 artifact hash does not match its manifest", code="live_hash_mismatch")
    cursor = _load_json(live_root / LIVE_ARTIFACTS["cursor"])
    _require_schema(cursor, LIVE_CURSOR_SCHEMA, LIVE_ARTIFACTS["cursor"])
    index = _load_json(live_root / LIVE_ARTIFACTS["index"])
    status = _load_json(live_root / LIVE_ARTIFACTS["status"])
    feed = list(_iter_jsonl(live_root / LIVE_ARTIFACTS["feed"], _validate_live_item))
    episodes = list(_iter_jsonl(live_root / LIVE_ARTIFACTS["episodes"], _validate_live_episode))
    emitted = [str(value) for value in cursor.get("emitted_observer_ids", []) or []]
    observed = [str(row.get("observer_item_id", "")) for row in feed]
    if observed != emitted or len(observed) != len(set(observed)):
        raise ChronicleReadError("LIVE1 cursor and feed identities differ", code="traceability_mismatch")
    return live_root, manifest, cursor, {**index, "status": status}, feed, episodes


def live_overview(cfg: Any, live_key: str) -> dict[str, Any]:
    _root, manifest, cursor, index, feed, episodes = _resolve_live(cfg, live_key)
    return {
        "schema_version": "eiel_chronicle_live_explorer_overview_v1",
        "live_key": live_key,
        "run_id": manifest.get("run_id"),
        "source_identity": manifest.get("source_identity"),
        "source_event_count": cursor.get("source_event_count"),
        "source_episode_count": cursor.get("source_episode_count"),
        "item_count": len(feed),
        "episode_presentation_count": len(episodes),
        "audience_profiles": manifest.get("audience_profiles"),
        "read_only_external_consumer": True,
        "simulation_control_available": False,
        "replay_not_simulation_replay": True,
        "index": index,
        "gen1_boundary": manifest.get("gen1_boundary"),
    }


def _live_matches(row: Mapping[str, Any], qs: Mapping[str, list[str]]) -> bool:
    profile = _one(qs, "profile", "public_plain")
    if profile not in OBSERVER_PROFILES:
        raise ChronicleReadError("invalid LIVE1 audience profile", code="invalid_query", status=HTTPStatus.BAD_REQUEST)
    if str(row.get("audience_profile")) != profile:
        return False
    if _values(qs, "category") and str(row.get("event_category")) not in _values(qs, "category"):
        return False
    if _values(qs, "event_type") and str(row.get("event_type")) not in _values(qs, "event_type"):
        return False
    generation = int(row.get("generation", 0) or 0)
    start = _int_param(qs, "generation_start")
    end = _int_param(qs, "generation_end")
    if start is not None and generation < start or end is not None and generation > end:
        return False
    references = dict(row.get("subject_references", {}) or {})
    subject = _one(qs, "subject")
    if subject and subject not in {
        *(str(value) for value in references.get("agent_ids", []) or []),
        *(str(value) for value in references.get("lineage_ids", []) or []),
        str(references.get("region_id") or ""),
    }:
        return False
    return True


def live_feed(cfg: Any, live_key: str, qs: Mapping[str, list[str]]) -> dict[str, Any]:
    _root, manifest, cursor, index, feed, _episodes = _resolve_live(cfg, live_key)
    after = _int_param(qs, "after", 0) or 0
    requested = _int_param(qs, "limit", 50) or 50
    maximum = min(_MAX_PAGE, int(index.get("maximum_transport_page", _MAX_PAGE) or _MAX_PAGE))
    if after < 0 or after > len(feed) or requested < 1 or requested > maximum:
        raise ChronicleReadError("LIVE1 paging bounds exceeded", code="invalid_paging", status=HTTPStatus.BAD_REQUEST)
    matches = [row for row in feed if _live_matches(row, qs)]
    items = matches[after : after + requested]
    next_cursor = after + len(items)
    return {
        "schema_version": "eiel_chronicle_live_transport_page_v1",
        "live_key": live_key,
        "run_id": manifest.get("run_id"),
        "profile": _one(qs, "profile", "public_plain"),
        "after": after,
        "next_cursor": next_cursor,
        "total": len(matches),
        "unread": max(0, len(matches) - next_cursor),
        "has_more": next_cursor < len(matches),
        "items": items,
        "source_cursor": {
            "source_event_count": cursor.get("source_event_count"),
            "last_generation": cursor.get("last_generation"),
            "health_status": cursor.get("health_status"),
        },
    }


def live_replay(cfg: Any, live_key: str, qs: Mapping[str, list[str]]) -> dict[str, Any]:
    page_qs = dict(qs)
    page_qs["after"] = ["0"]
    page_qs["limit"] = [str(min(_MAX_PAGE, _int_param(qs, "limit", _MAX_PAGE) or _MAX_PAGE))]
    page = live_feed(cfg, live_key, page_qs)
    _root, manifest, cursor, index, _feed, episodes = _resolve_live(cfg, live_key)
    profile = page["profile"]
    return {
        "schema_version": "eiel_chronicle_live_replay_v1",
        "label": "Chronicle replay, not simulation replay.",
        "live_key": live_key,
        "run_id": manifest.get("run_id"),
        "profile": profile,
        "items": page["items"],
        "total": page["total"],
        "episodes": [row for row in episodes if str(row.get("audience_profile")) == profile],
        "source_cursor": cursor,
        "replay_controls": {
            "event_step": True,
            "generation_step": True,
            "generation_scrub": True,
            "episode_jump": True,
            "subject_follow": True,
            "speech_is_optional_browser_presentation": True,
        },
        "maximum_simultaneous_bubbles": index.get("maximum_simultaneous_bubbles"),
    }


def live_branch_replay(cfg: Any, qs: Mapping[str, list[str]]) -> dict[str, Any]:
    key_a = _one(qs, "run_a_live_key")
    key_b = _one(qs, "run_b_live_key")
    generation = _int_param(qs, "generation")
    _ra, ma, _ca, _ia, feed_a, _ea = _resolve_live(cfg, key_a)
    _rb, mb, _cb, _ib, feed_b, _eb = _resolve_live(cfg, key_b)
    profile = _one(qs, "profile", "public_plain")
    select = lambda rows: [
        row for row in rows
        if str(row.get("audience_profile")) == profile
        and (generation is None or int(row.get("generation", 0) or 0) == generation)
    ][:_MAX_PAGE]
    comparison_key = _one(qs, "comparison_key")
    comparison = comparison_detail(cfg, comparison_key, include_raw=False) if comparison_key else None
    return {
        "schema_version": "eiel_chronicle_live_branch_replay_v1",
        "label": "Chronicle branch comparison replay, not simulation replay.",
        "run_a": {"live_key": key_a, "run_id": ma.get("run_id"), "items": select(feed_a)},
        "run_b": {"live_key": key_b, "run_id": mb.get("run_id"), "items": select(feed_b)},
        "generation": generation,
        "profile": profile,
        "first_meaningful_divergence": comparison.get("first_meaningful_divergence") if comparison else None,
        "comparison": comparison,
    }


def handle_get_request(*, cfg: Any, path: str, qs: Mapping[str, list[str]]) -> tuple[int, dict[str, Any]]:
    """Dispatch bounded GET-only Explorer calls without writing run artifacts."""

    try:
        if path == "/api/chronicle/runs":
            return HTTPStatus.OK, {
                "schema_version": "eiel_chronicle_explorer_run_list_v1",
                "approved_roots": [str(root) for root in _approved_roots(cfg)],
                "archive_root": str(_archive_root(cfg)),
                "archive_roots": [str(root) for root in _archive_roots(cfg)],
                "runs": discover_runs(cfg),
            }
        if path == "/api/chronicle/library":
            return HTTPStatus.OK, archive_library(cfg)
        if path == "/api/chronicle/run":
            return HTTPStatus.OK, run_overview(cfg, _one(qs, "run_key"))
        if path == "/api/chronicle/dialogue":
            return HTTPStatus.OK, dialogue(
                cfg,
                _one(qs, "run_key"),
                _one(qs, "audience", "dry_humour"),
                qs,
            )
        if path == "/api/chronicle/timeline":
            return HTTPStatus.OK, timeline(cfg, _one(qs, "run_key"), qs)
        if path == "/api/chronicle/episode":
            return HTTPStatus.OK, episode_detail(cfg, _one(qs, "run_key"), _one(qs, "episode_id"))
        if path == "/api/chronicle/subject":
            return HTTPStatus.OK, subject_detail(cfg, _one(qs, "run_key"), _one(qs, "kind"), _one(qs, "id"))
        if path == "/api/chronicle/context":
            return HTTPStatus.OK, context_detail(
                cfg, _one(qs, "run_key"), context_id=_one(qs, "context_id"), event_id=_one(qs, "event_id")
            )
        if path == "/api/chronicle/event":
            return HTTPStatus.OK, event_detail(cfg, _one(qs, "run_key"), _one(qs, "event_id"))
        if path == "/api/chronicle/observer":
            return HTTPStatus.OK, observer_overview(
                cfg, _one(qs, "run_key"), _one(qs, "profile", "public_plain")
            )
        if path == "/api/chronicle/observer/feed":
            return HTTPStatus.OK, observer_feed(
                cfg, _one(qs, "run_key"), _one(qs, "profile", "public_plain"), qs
            )
        if path == "/api/chronicle/observer/story":
            return HTTPStatus.OK, observer_story(
                cfg,
                _one(qs, "run_key"),
                _one(qs, "profile", "public_plain"),
                _one(qs, "kind"),
                _one(qs, "id"),
            )
        if path == "/api/chronicle/observer/stories":
            return HTTPStatus.OK, observer_stories(
                cfg,
                _one(qs, "run_key"),
                _one(qs, "profile", "public_plain"),
                _one(qs, "kind"),
                qs,
            )
        if path == "/api/chronicle/documentary":
            return HTTPStatus.OK, documentary_overview(cfg, _one(qs, "run_key"))
        if path == "/api/chronicle/live/runs":
            return HTTPStatus.OK, {
                "schema_version": "eiel_chronicle_live_explorer_run_list_v1",
                "approved_roots": [str(root) for root in _approved_live_roots(cfg)],
                "runs": discover_live_runs(cfg),
            }
        if path == "/api/chronicle/live":
            return HTTPStatus.OK, live_overview(cfg, _one(qs, "live_key"))
        if path == "/api/chronicle/live/feed":
            return HTTPStatus.OK, live_feed(cfg, _one(qs, "live_key"), qs)
        if path == "/api/chronicle/live/replay":
            return HTTPStatus.OK, live_replay(cfg, _one(qs, "live_key"), qs)
        if path == "/api/chronicle/live/branch":
            return HTTPStatus.OK, live_branch_replay(cfg, qs)
        if path == "/api/chronicle/comparisons":
            return HTTPStatus.OK, {
                "schema_version": "eiel_chronicle_explorer_comparison_list_v1",
                "comparisons": discover_comparisons(cfg),
            }
        if path == "/api/chronicle/comparison":
            include_raw = _one(qs, "include_raw").lower() in {"1", "true", "yes"}
            return HTTPStatus.OK, comparison_detail(cfg, _one(qs, "comparison_key"), include_raw=include_raw)
        raise ChronicleReadError("Chronicle endpoint not found", code="not_found", status=HTTPStatus.NOT_FOUND)
    except ChronicleReadError as exc:
        return exc.status, {
            "schema_version": "eiel_chronicle_explorer_error_v1",
            "ok": False,
            "error_code": exc.code,
            "error": str(exc),
        }
    except Exception as exc:
        return HTTPStatus.INTERNAL_SERVER_ERROR, {
            "schema_version": "eiel_chronicle_explorer_error_v1",
            "ok": False,
            "error_code": "isolated_reader_failure",
            "error": f"{type(exc).__name__}: {exc}",
        }


__all__ = [
    "ChronicleReadError",
    "archive_library",
    "comparison_detail",
    "context_detail",
    "discover_comparisons",
    "discover_live_runs",
    "discover_runs",
    "episode_detail",
    "event_detail",
    "handle_get_request",
    "live_branch_replay",
    "live_feed",
    "live_overview",
    "live_replay",
    "observer_feed",
    "observer_overview",
    "observer_story",
    "observer_stories",
    "run_overview",
    "resolve_archive_download",
    "subject_detail",
    "timeline",
]
