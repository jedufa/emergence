"""Shared deterministic contracts and I/O helpers for EIEL CLI actors."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import pwd
import shlex
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from tools.actors.canonical import canonical_sha256, pretty_json

ACTOR_RECORD_SCHEMA_VERSION = "eiel_actor_record_envelope_v1"

ACTOR_SOURCE_REF_KINDS = frozenset({
    "work", "session", "bootstrap", "mission", "technical_qualification",
    "qualification", "resource_observation", "resource_assessment",
    "evidence_assessment", "dispatch_request", "assignment", "lease",
    "worker_result",
})
EXTERNAL_SOURCE_REF_KINDS = frozenset({
    "repository", "librarian_preflight", "knowledge_librarian_preflight",
    "input", "work_package", "campaign_evidence_manifest", "qf_catalog",
    "authority_register", "qualification_catalog",
    "qualification_authority_register", "manifest", "evidence",
    "resource_demand",
})
EXTERNAL_UNVERIFIABLE_SOURCE_KIND = "external_unverifiable"


def validate_source_ref_shape(raw: Any) -> list[str]:
    """Validate one typed source-reference envelope without dereferencing it."""

    if not isinstance(raw, Mapping):
        return ["source reference must be an object"]
    ref = dict(raw)
    kind = str(ref.get("kind") or "").strip()
    if not kind:
        return ["source reference kind is required"]
    errors: list[str] = []
    if kind in ACTOR_SOURCE_REF_KINDS:
        record_id = str(ref.get("record_id") or "").strip()
        if not record_id:
            errors.append(f"actor source reference {kind!r} requires record_id")
        else:
            try:
                validate_record_id(record_id)
            except ActorError as exc:
                errors.append(str(exc))
        revision = ref.get("revision")
        if revision is not None and (not isinstance(revision, int) or isinstance(revision, bool) or revision < 1):
            errors.append("actor source reference revision must be an integer >= 1")
        digest = ref.get("record_sha256", ref.get("sha256"))
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest.lower())
        ):
            errors.append("actor source reference digest must be a 64-character SHA-256 hex string")
        schema = ref.get("schema_version")
        if schema is not None and schema != "eiel_actor_record_ref_v1":
            errors.append("actor source reference schema_version is unsupported")
    elif kind in EXTERNAL_SOURCE_REF_KINDS:
        locator = ref.get("uri") if ref.get("uri") is not None else ref.get("path")
        if not isinstance(locator, str) or not locator.strip():
            errors.append(f"external source reference {kind!r} requires uri or path")
        digest = ref.get("sha256")
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest.lower())
        ):
            errors.append("external source reference sha256 must be a 64-character SHA-256 hex string")
    elif kind == EXTERNAL_UNVERIFIABLE_SOURCE_KIND:
        if not isinstance(ref.get("uri"), str) or not str(ref.get("uri") or "").strip():
            errors.append("external_unverifiable source reference requires uri")
        if not isinstance(ref.get("reason"), str) or not str(ref.get("reason") or "").strip():
            errors.append("external_unverifiable source reference requires reason")
    else:
        errors.append(f"unknown source reference kind: {kind}")
    return errors


def validate_source_refs_shape(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return ["record source_refs must be a list"]
    errors: list[str] = []
    for index, ref in enumerate(raw):
        errors.extend(f"source_refs[{index}]: {error}" for error in validate_source_ref_shape(ref))
    return errors
DEFAULT_ACTOR_STATE_ROOT = (
    Path(os.environ.get("XDG_STATE_HOME", "~/.local/state")).expanduser()
    / "eiel"
    / "actors"
)
FIXED_TIME_ENV = "EIEL_FIXED_TIME_UTC"
STATE_ROOT_ENV = "EIEL_ACTOR_STATE_ROOT"
EIEL_LAUNCHER = "python -m tools.eiel"
NORMALIZED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ID_RE = re.compile(r"^[A-Z][A-Z0-9-]{2,63}$")


class ActorError(RuntimeError):
    """Fail-closed error raised by actor managers with Codex guidance."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "actor_error",
        problems: Sequence[str] | None = None,
        warnings: Sequence[str] | None = None,
        next_actions: Sequence[Mapping[str, Any] | str] | None = None,
        related_records: Sequence[Mapping[str, Any] | str] | None = None,
    ) -> None:
        super().__init__(str(message))
        self.code = str(code)
        self.problems = [str(item) for item in (problems or (str(message),))]
        self.warnings = [str(item) for item in (warnings or ())]
        self.next_actions = [normalize_action(item) for item in (next_actions or ())]
        self.related_records = [normalize_related_record(item) for item in (related_records or ())]



CODEX_GUIDANCE_SCHEMA_VERSION = "eiel_codex_guidance_v1"

# Authority-changing actions must be self-describing wherever they are
# projected.  A consumer should not have to infer the next authority from a
# command string or from whichever manager happened to render the action.
ACTION_AUTHORITY_REQUIREMENTS = {
    "start_independent_review": ("REVIEW_AUTHORITY", "review"),
    "start_repair_session": ("IMPLEMENTATION_AUTHORITY", "repair"),
    "repair_implementation": ("IMPLEMENTATION_AUTHORITY", "repair"),
    "return_for_repair": ("IMPLEMENTATION_AUTHORITY", "repair"),
    "return_to_engineering_authority": ("ENGINEERING_AUTHORITY", "engineering"),
}


def normalize_action(value: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        action = dict(value)
        action.setdefault("description", str(action.get("command") or action.get("action") or "next action"))
        requirement = ACTION_AUTHORITY_REQUIREMENTS.get(str(action.get("action") or ""))
        if requirement:
            action.setdefault("required_authority_role", requirement[0])
            action.setdefault("authority_phase", requirement[1])
        return action
    text = str(value)
    return {"description": text}


def normalize_related_record(value: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {"record_id": str(value)}


def codex_guidance(
    *,
    status: str,
    reason_codes: Sequence[str] = (),
    problems: Sequence[str] = (),
    warnings: Sequence[str] = (),
    next_actions: Sequence[Mapping[str, Any] | str] = (),
    related_records: Sequence[Mapping[str, Any] | str] = (),
) -> dict[str, Any]:
    """Return the stable recovery surface consumed by Codex."""

    return {
        "guidance_schema_version": CODEX_GUIDANCE_SCHEMA_VERSION,
        "status": str(status),
        "reason_codes": [str(item) for item in reason_codes],
        "problems": [str(item) for item in problems],
        "warnings": [str(item) for item in warnings],
        "next_actions": [normalize_action(item) for item in next_actions],
        "related_records": [normalize_related_record(item) for item in related_records],
    }


def with_guidance(
    payload: Mapping[str, Any],
    *,
    status: str,
    reason_codes: Sequence[str] = (),
    problems: Sequence[str] = (),
    warnings: Sequence[str] = (),
    next_actions: Sequence[Mapping[str, Any] | str] = (),
    related_records: Sequence[Mapping[str, Any] | str] = (),
) -> dict[str, Any]:
    out = dict(payload)
    out.update(
        codex_guidance(
            status=status,
            reason_codes=reason_codes,
            problems=problems,
            warnings=warnings,
            next_actions=next_actions,
            related_records=related_records,
        )
    )
    return out


def actor_error_payload(exc: ActorError, *, actor: str, command: str) -> dict[str, Any]:
    return with_guidance(
        {
            "ok": False,
            "actor": actor,
            "command": command,
            "error": str(exc),
        },
        status="BLOCKED",
        reason_codes=[exc.code],
        problems=exc.problems,
        warnings=exc.warnings,
        next_actions=exc.next_actions,
        related_records=exc.related_records,
    )

class ActorArgumentParser(argparse.ArgumentParser):
    """Argument parser that returns actionable, machine-readable CLI errors."""

    def error(self, message: str) -> None:  # pragma: no cover - exercised through subprocess tests
        prog = self.prog
        raise ActorError(
            f"invalid command usage: {message}",
            code="cli_usage_error",
            problems=[message],
            next_actions=[
                {
                    "action": "show_help",
                    "description": f"Read the command help for {prog}.",
                    "command": f"{prog} --help",
                }
            ],
        )


def forbidden_operation_error(actor_command: str, operation: str, allowed_alternative: str) -> ActorError:
    return ActorError(
        f"{actor_command} does not own {operation}",
        code="forbidden_actor_operation",
        problems=[f"The requested operation '{operation}' is outside this actor's authority."],
        next_actions=[
            {
                "action": "use_allowed_surface",
                "description": allowed_alternative,
            }
        ],
    )


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def actor_state_root(value: str | Path | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    configured = os.environ.get(STATE_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    xdg_state_home = Path(os.environ.get("XDG_STATE_HOME", "~/.local/state")).expanduser()
    return (xdg_state_home / "eiel" / "actors").resolve()


def canonical_host_actor_state_root() -> Path:
    """Return the host-canonical actor root without environment overrides."""

    account_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    return account_home / ".local" / "state" / "eiel" / "actors"


def gpu_uuid_equal_v1(left: object, right: object) -> bool:
    """Compare physical GPU UUIDs across NVML and CUDA presentation forms."""

    if not isinstance(left, str) or not isinstance(right, str):
        return False
    normalized: list[str] = []
    for value in (left, right):
        item = value.strip().lower()
        if item.startswith("gpu-"):
            item = item[4:]
        if not item:
            return False
        normalized.append(item)
    return normalized[0] == normalized[1]


def render_eiel_command(*arguments: object) -> str:
    """Render one repository-first CLI command with shell-safe arguments.

    Guidance emitted by actor managers must use the same launcher contract as
    bootstrap.  The helper keeps that contract in one place while still
    allowing callers to pass preformatted placeholders when they deliberately
    need them.
    """

    rendered = [EIEL_LAUNCHER]
    for value in arguments:
        text = str(value)
        if text.startswith("<") and text.endswith(">"):
            rendered.append(text)
        else:
            rendered.append(shlex.quote(text))
    return " ".join(rendered)


def utc_now(value: str | None = None) -> str:
    raw = value or os.environ.get(FIXED_TIME_ENV)
    if raw:
        return validate_timestamp(raw)
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def validate_timestamp(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ActorError("timestamp must be non-empty")
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ActorError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ActorError("timestamp must include a timezone")
    return parsed.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def require_test_fixture_context() -> None:
    """Allow raw owner fixture persistence only from an active pytest case."""

    if not os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("EIEL_ALLOW_TEST_FIXTURE_WRITES") != "1":
        raise ActorError(
            "raw authoritative-record fixture persistence is test-only",
            code="test_fixture_write_forbidden",
        )


def require_known_fields(
    raw: Mapping[str, Any],
    *,
    allowed: Iterable[str],
    label: str,
) -> None:
    """Reject undeclared durable top-level fields for closed authoritative ABIs."""

    allowed_set = {str(value) for value in allowed}
    unknown = sorted(str(key) for key in raw if str(key) not in allowed_set)
    if unknown:
        raise ActorError(
            f"{label} contains unsupported top-level fields",
            code="schema_unknown_fields",
            problems=unknown,
        )


def require_explicit_schema_version(
    value: Mapping[str, Any],
    *,
    expected: str,
    label: str,
    supported: Sequence[str] | None = None,
) -> str:
    """Validate one persisted mapping's explicit schema discriminator.

    Creation-time normalizers may add the current schema version to a new
    in-memory object.  Loaders must not do that to already-persisted data: a
    missing discriminator is ambiguous, while an unknown discriminator may be
    a future format that this code must not rewrite or reinterpret.  Keeping
    these failure classes distinct is the small, reusable contract needed by
    the operator tooling without introducing a migration framework.
    """

    if not isinstance(value, Mapping):
        raise ActorError(
            f"{label} must be an object",
            code="schema_record_malformed",
        )
    if "schema_version" not in value:
        raise ActorError(
            f"{label} schema_version is missing",
            code="schema_version_missing",
        )
    observed = value.get("schema_version")
    if not isinstance(observed, str) or not observed.strip():
        raise ActorError(
            f"{label} schema_version is malformed",
            code="schema_version_malformed",
        )
    observed = observed.strip()
    accepted = tuple(dict.fromkeys([expected, *(supported or ())]))
    if observed not in accepted:
        raise ActorError(
            f"unsupported {label} schema_version: {observed}",
            code="schema_version_unsupported",
            problems=[f"Supported schema versions: {', '.join(accepted)}"],
        )
    return observed


def timestamp_age_hours(value: str, *, now: str | None = None) -> float:
    """Return the non-negative age of an ISO-8601 timestamp in hours.

    Actor receipts are frequently reused after a Codex interruption.  The age
    is deliberately calculated in one shared place so every recovery surface
    interprets stale bootstrap evidence the same way.
    """

    then = dt.datetime.fromisoformat(validate_timestamp(value).replace("Z", "+00:00"))
    current = dt.datetime.fromisoformat(utc_now(now).replace("Z", "+00:00"))
    return max(0.0, (current - then).total_seconds() / 3600.0)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path, *, name: str = "JSON document") -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActorError(f"cannot read {name}: {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ActorError(f"{name} must be a JSON object: {source}")
    return dict(value)


def read_structured(path: str | Path, *, name: str = "document") -> dict[str, Any]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        return read_json(source, name=name)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency is present in EIEL
            raise ActorError("PyYAML is required to read YAML documents") from exc
        try:
            value = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ActorError(f"cannot read {name}: {source}: {exc}") from exc
        if not isinstance(value, dict):
            raise ActorError(f"{name} must be an object: {source}")
        return dict(value)
    raise ActorError(f"unsupported document type for {name}: {source.suffix or '<none>'}")


def _fsync_directory(path: Path) -> None:
    """Durably commit a directory entry after an atomic rename.

    ``os.replace`` protects readers from partial files, but without syncing the
    parent directory a successful command can still disappear after power loss.
    EIEL record writes are recovery authority, so durability is fail-closed.
    """

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(target: Path, text: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        _fsync_directory(target.parent)
        return target
    except OSError as exc:
        raise ActorError(
            f"durable atomic write failed: {target}: {exc}",
            code="actor_store_durable_write_failed",
            problems=[str(target), str(exc)],
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        _fsync_directory(target.parent)
        return target
    except OSError as exc:
        raise ActorError(
            f"durable atomic write failed: {target}: {exc}",
            code="actor_store_durable_write_failed",
            problems=[str(target), str(exc)],
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    return _atomic_write_text(Path(path), pretty_json(dict(value)))


def atomic_write_text(path: str | Path, text: str) -> Path:
    return _atomic_write_text(Path(path), text)


def stable_id(prefix: str, value: Any, *, length: int = 12) -> str:
    normalized = str(prefix).strip().upper().replace("_", "-")
    if not _ID_RE.match(normalized):
        raise ActorError(f"invalid record ID prefix: {prefix!r}")
    digest = canonical_sha256(value).upper()
    return f"{normalized}-{digest[:length]}"


def validate_record_id(value: str) -> str:
    text = str(value).strip().upper()
    if not _ID_RE.match(text):
        raise ActorError(f"invalid record ID: {value!r}")
    return text


def _record_without_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload.pop("record_sha256", None)
    return payload


def seal_record(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = _record_without_hash(record)
    payload["record_sha256"] = canonical_sha256(payload)
    return payload


def load_actor_record(raw: Any) -> dict[str, Any]:
    """Strictly load one durable actor record envelope.

    Version discrimination happens before digest or shape validation so an
    unknown future format is never misreported as ordinary corruption.  This
    function deliberately does not normalize the record.
    """

    if not isinstance(raw, Mapping):
        raise ActorError("actor record must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        raw,
        expected=ACTOR_RECORD_SCHEMA_VERSION,
        label="actor record envelope",
    )
    errors = verify_record(raw)
    if not errors:
        return dict(raw)
    if errors == ["record_sha256 mismatch"]:
        raise ActorError(
            "actor record digest mismatch",
            code="actor_record_digest_mismatch",
            problems=errors,
        )
    raise ActorError(
        "actor record is malformed",
        code="actor_record_malformed",
        problems=errors,
    )


def verify_record(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if "schema_version" not in record:
        errors.append("actor record schema_version is missing")
    elif not isinstance(record.get("schema_version"), str) or not str(record.get("schema_version") or "").strip():
        errors.append("actor record schema_version is malformed")
    elif record.get("schema_version") != ACTOR_RECORD_SCHEMA_VERSION:
        errors.append("unsupported actor record schema_version")
    expected = record.get("record_sha256")
    actual = canonical_sha256(_record_without_hash(record))
    if not isinstance(expected, str) or expected != actual:
        errors.append("record_sha256 mismatch")
    try:
        validate_record_id(str(record.get("record_id") or ""))
    except ActorError as exc:
        errors.append(str(exc))
    if not isinstance(record.get("payload"), Mapping):
        errors.append("record payload must be an object")
    if not isinstance(record.get("history"), list):
        errors.append("record history must be a list")
    revision = int(record.get("revision") or 0)
    previous = record.get("previous_record_sha256")
    if revision <= 1:
        if previous not in {None, ""}:
            errors.append("revision 1 must not declare previous_record_sha256")
    elif previous is not None and (
        not isinstance(previous, str) or not re.fullmatch(r"[0-9a-f]{64}", previous)
    ):
        errors.append("previous_record_sha256 is malformed")
    errors.extend(validate_source_refs_shape(record.get("source_refs")))
    return errors


def new_record(
    *,
    actor: str,
    record_type: str,
    record_id: str,
    status: str,
    payload: Mapping[str, Any],
    generated_at: str,
    source_refs: Sequence[Mapping[str, Any]] = (),
    action: str = "create",
    note: str | None = None,
) -> dict[str, Any]:
    timestamp = validate_timestamp(generated_at)
    event: dict[str, Any] = {
        "revision": 1,
        "action": action,
        "at_utc": timestamp,
        "status": status,
    }
    if note:
        event["note"] = str(note)
    return seal_record(
        {
            "schema_version": ACTOR_RECORD_SCHEMA_VERSION,
            "record_type": str(record_type),
            "record_id": validate_record_id(record_id),
            "actor": str(actor),
            "status": str(status),
            "revision": 1,
            "created_at_utc": timestamp,
            "updated_at_utc": timestamp,
            "source_refs": [dict(row) for row in source_refs],
            "payload": dict(payload),
            "history": [event],
        }
    )


def revise_record(
    record: Mapping[str, Any],
    *,
    payload: Mapping[str, Any] | None = None,
    status: str | None = None,
    action: str,
    generated_at: str,
    note: str | None = None,
    source_refs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    errors = verify_record(record)
    if errors:
        raise ActorError("cannot revise invalid actor record: " + "; ".join(errors))
    timestamp = validate_timestamp(generated_at)
    revision = int(record.get("revision") or 0) + 1
    event: dict[str, Any] = {
        "revision": revision,
        "action": str(action),
        "at_utc": timestamp,
        "status": str(status or record.get("status") or "UNKNOWN"),
    }
    if note:
        event["note"] = str(note)
    updated = _record_without_hash(record)
    updated.update(
        {
            "status": str(status or record.get("status") or "UNKNOWN"),
            "revision": revision,
            "updated_at_utc": timestamp,
            "payload": dict(payload if payload is not None else record.get("payload") or {}),
            # New revisions bind the exact predecessor bytes.  Historical V1
            # revisions without this optional link remain readable, while any
            # revision produced by current code is substitution-evident.
            "previous_record_sha256": str(record.get("record_sha256") or ""),
            # Immutable revision files already preserve the complete event
            # sequence.  Re-embedding every prior event in every new snapshot
            # made total storage quadratic in revision count.  The envelope now
            # carries only the event that produced this revision; historical
            # v1 snapshots with cumulative arrays remain valid and readable.
            "history": [event],
        }
    )
    if source_refs is not None:
        updated["source_refs"] = [dict(row) for row in source_refs]
    return seal_record(updated)


@contextlib.contextmanager
def advisory_lock(path: str | Path) -> Iterator[None]:
    """Acquire an exclusive local file lock or fail without mutation.

    Lifecycle correctness depends on serialization. Silently continuing when
    ``fcntl`` or ``flock`` is unavailable converts an invariant into wishful
    thinking, a traditional human concurrency strategy that EIEL declines.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - supported production host is POSIX
        raise ActorError(
            "exclusive actor locking is unavailable on this platform",
            code="actor_lock_unavailable",
            problems=[str(target)],
        ) from exc
    try:
        handle = target.open("a+", encoding="utf-8")
    except OSError as exc:
        raise ActorError(
            f"cannot open actor lock: {target}: {exc}",
            code="actor_lock_unavailable",
            problems=[str(target), str(exc)],
        ) from exc
    locked = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
        except OSError as exc:
            raise ActorError(
                f"cannot acquire actor lock: {target}: {exc}",
                code="actor_lock_unavailable",
                problems=[str(target), str(exc)],
            ) from exc
        yield
    finally:
        if locked:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError as exc:
                handle.close()
                raise ActorError(
                    f"cannot release actor lock: {target}: {exc}",
                    code="actor_lock_release_failed",
                    problems=[str(target), str(exc)],
                ) from exc
        handle.close()


def deterministic_zip(
    output: str | Path,
    files: Iterable[tuple[str, str | Path]],
) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted((str(name).replace("\\", "/"), Path(path)) for name, path in files)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, source in entries:
            if not source.is_file():
                raise ActorError(f"cannot export missing file: {source}")
            info = zipfile.ZipInfo(name, date_time=NORMALIZED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
    return target


def copy_file(source: str | Path, destination: str | Path) -> Path:
    src = Path(source)
    dst = Path(destination)
    if not src.is_file():
        raise ActorError(f"source file does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def parse_set_values(values: Sequence[str]) -> dict[str, Any]:
    """Parse repeated ``key=JSON-value`` command-line patches."""

    out: dict[str, Any] = {}
    for item in values:
        key, sep, raw = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            raise ActorError(f"--set requires key=value, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        out[key] = value
    return out


def apply_dotted_patch(payload: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(payload)))
    for dotted_key, value in patch.items():
        parts = [part for part in str(dotted_key).split(".") if part]
        if not parts:
            raise ActorError("patch keys must be non-empty")
        cursor: dict[str, Any] = result
        for part in parts[:-1]:
            current = cursor.get(part)
            if current is None:
                current = {}
                cursor[part] = current
            if not isinstance(current, dict):
                raise ActorError(f"cannot patch through non-object field: {dotted_key}")
            cursor = current
        cursor[parts[-1]] = value
    return result


def compact_pretty_json(value: Any, *, indent: int = 0, width: int = 280) -> str:
    """Render deterministic JSON with small objects and scalar arrays inline.

    This is intentionally a display format, not a canonical serialization.
    It keeps the structured compact Workspace opening readable while honoring
    package display-line budgets better than ordinary fully-expanded JSON.
    """

    prefix = " " * indent
    if value is None or isinstance(value, (bool, int, float, str)):
        return prefix + json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, Mapping):
        obj = {str(key): item for key, item in value.items()}
        one_line = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(", ", ": "))
        simple = all(
            item is None
            or isinstance(item, (bool, int, float, str))
            or (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes, bytearray))
                and all(sub is None or isinstance(sub, (bool, int, float, str)) for sub in item)
            )
            for item in obj.values()
        )
        if simple and len(prefix) + len(one_line) <= width:
            return prefix + one_line
        if not obj:
            return prefix + "{}"
        lines = [prefix + "{"]
        items = sorted(obj.items(), key=lambda pair: pair[0])
        for index, (key, item) in enumerate(items):
            rendered = compact_pretty_json(item, indent=indent + 2, width=width)
            rendered_lines = rendered.splitlines() or [" " * (indent + 2)]
            key_prefix = " " * (indent + 2) + json.dumps(key, ensure_ascii=False) + ": "
            first = rendered_lines[0].lstrip()
            lines.append(key_prefix + first)
            if len(rendered_lines) > 1:
                lines.extend(rendered_lines[1:])
            if index < len(items) - 1:
                lines[-1] += ","
        lines.append(prefix + "}")
        return "\n".join(lines)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = list(value)
        one_line = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(", ", ": "))
        scalar = all(item is None or isinstance(item, (bool, int, float, str)) for item in rows)
        shallow = all(
            item is None
            or isinstance(item, (bool, int, float, str))
            or (
                isinstance(item, Mapping)
                and all(sub is None or isinstance(sub, (bool, int, float, str)) for sub in item.values())
            )
            for item in rows
        )
        if (scalar or shallow) and len(prefix) + len(one_line) <= width:
            return prefix + one_line
        if not rows:
            return prefix + "[]"
        lines = [prefix + "["]
        for index, item in enumerate(rows):
            rendered = compact_pretty_json(item, indent=indent + 2, width=width)
            item_lines = rendered.splitlines()
            lines.extend(item_lines)
            if index < len(rows) - 1:
                lines[-1] += ","
        lines.append(prefix + "]")
        return "\n".join(lines)
    return prefix + json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def bounded_compact_json(value: Any, *, maximum_lines: int = 120) -> str:
    """Render compact JSON within an advisory line budget when possible."""

    limit = max(1, int(maximum_lines))
    for width in (160, 220, 280, 360, 480, 720):
        rendered = compact_pretty_json(value, width=width)
        if len(rendered.splitlines()) <= limit:
            return rendered + "\n"
    # Extremely large candidate/warning sets still remain valid structured JSON
    # rather than violating the Work Package display budget.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def emit_payload(
    payload: Mapping[str, Any],
    *,
    output: str | Path | None = None,
    human_text: str | None = None,
    json_output: bool = True,
    display_payload: Mapping[str, Any] | None = None,
    compact_json: bool = False,
    maximum_display_lines: int = 120,
) -> None:
    """Persist the full payload while optionally displaying a derived view.

    ``display_payload`` is presentation-only.  It lets compact Workspace output
    avoid duplicating full records without changing the deterministic object
    written through ``--output``.  Existing callers retain their prior behavior.
    """

    if output:
        atomic_write_json(output, payload)
    shown = dict(display_payload) if display_payload is not None else dict(payload)
    if json_output or human_text is None:
        rendered = bounded_compact_json(shown, maximum_lines=maximum_display_lines) if compact_json else pretty_json(shown)
        print(rendered, end="")
    else:
        print(human_text.rstrip() + "\n")


def cli_output_preferences(
    raw_args: Sequence[str],
    *,
    parsed_args: argparse.Namespace | None = None,
) -> tuple[bool, str | Path | None]:
    """Recover presentation flags even when argument parsing fails.

    Actor parsers deliberately raise :class:`ActorError` instead of exiting.
    A malformed command can therefore fail before an ``argparse.Namespace``
    exists.  Human callers still deserve the human error surface they asked
    for, and ``--output`` should retain the same deterministic error payload as
    successful commands.  This helper reads only the two presentation flags;
    it does not reinterpret command semantics.
    """

    human = bool(getattr(parsed_args, "human", False)) if parsed_args is not None else False
    output: str | Path | None = (
        getattr(parsed_args, "output", None) if parsed_args is not None else None
    )
    values = [str(value) for value in raw_args]
    if "--human" in values:
        human = True
    if output is None:
        for index, value in enumerate(values):
            if value == "--output" and index + 1 < len(values):
                candidate = values[index + 1]
                if candidate and not candidate.startswith("-"):
                    output = candidate
                break
            if value.startswith("--output="):
                candidate = value.split("=", 1)[1]
                if candidate:
                    output = candidate
                break
    return human, output


def priority_ordered_findings(values: Sequence[Any]) -> list[Any]:
    """Keep integrity-critical findings visible in bounded human output."""

    def priority(value: Any) -> tuple[int, str]:
        if isinstance(value, Mapping):
            text = " ".join(str(value.get(key) or "") for key in ("severity", "code", "category", "summary", "message"))
        else:
            text = str(value)
        upper = text.upper()
        if any(token in upper for token in ("CRITICAL", "CORRUPT", "INTEGRITY", "FATAL", "SECURITY")):
            rank = 0
        elif any(token in upper for token in ("ERROR", "INVALID", "MISSING", "CONFLICT", "STALE")):
            rank = 1
        elif any(token in upper for token in ("WARNING", "CAUTION", "PARTIAL")):
            rank = 2
        else:
            rank = 3
        return rank, upper

    return sorted(list(values), key=priority)


def human_error_text(payload: Mapping[str, Any]) -> str:
    """Render one concise but actionable human error summary."""

    lines = [f"ERROR: {payload.get('error') or 'command blocked'}"]
    for problem in priority_ordered_findings(payload.get("problems") or [])[:5]:
        lines.append(f"Problem: {problem}")
    warnings = priority_ordered_findings(payload.get("warnings") or [])
    if warnings:
        lines.append(f"Warnings: {len(warnings)}")
        for warning in warnings[:3]:
            lines.append(f"Warning: {warning}")
    actions = list(payload.get("next_actions") or [])
    if actions:
        action = dict(actions[0])
        lines.append(f"Next: {action.get('description') or 'inspect the structured error'}")
        if action.get("command"):
            lines.append(f"Command: {action['command']}")
    return "\n".join(lines) + "\n"


def emit_actor_error(
    exc: ActorError,
    *,
    actor: str,
    command: str,
    raw_args: Sequence[str] = (),
    parsed_args: argparse.Namespace | None = None,
) -> None:
    """Emit an actor failure using the caller's requested presentation mode."""

    payload = actor_error_payload(exc, actor=actor, command=command)
    human, output = cli_output_preferences(raw_args, parsed_args=parsed_args)
    emit_payload(
        payload,
        output=output,
        human_text=human_error_text(payload),
        json_output=not human,
    )
