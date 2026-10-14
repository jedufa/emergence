"""Report-only adapter to the canonical exact-root retirement actuator.

Call only at the owning run's terminal boundary after all consumers close.
Markdown must contain nonempty Inputs, Settings, Results, Limitations and
Reproduction headings. The caller supplies self-contained scientific content;
the mechanical cleanup review grants no scientific or qualification authority.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import tempfile
from typing import Sequence

from . import canopy_supervision_v1 as canopy_supervision
from . import root_retirement as retirement

_MARKER = "\n<!-- EIEL_REPORT_ONLY_CLEANUP_V1\n"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cleanup_summary(receipt: dict) -> str:
    return ("\n## Cleanup\n\nVerified complete: retired " + str(receipt["root_count"])
            + " exact run roots; released " + str(receipt["bytes_released"])
            + " bytes. No archival was requested.\n")


def _safe(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise retirement.RootRetirementError("report-only paths cannot contain symlinks")
    if path.exists() and path.is_file() and path.stat().st_nlink != 1:
        raise retirement.RootRetirementError("report-only files must have one link")


def _write(path: Path, text: str) -> None:
    _safe(path)
    fd, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path: Path) -> dict:
    _safe(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise retirement.RootRetirementError("invalid report-only transaction")
    return value


def read_report_markdown_v1(path: Path) -> str:
    """Read bounded original Markdown, validating any appended cleanup record.

    Completed report replay uses the original report body, not its generated
    cleanup appendix. Root/stage absence is still verified by closeout itself.
    """
    path = Path(path)
    _safe(path)
    if path.suffix != ".md":
        raise retirement.RootRetirementError("report must be a Markdown file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        observed = os.fstat(fd)
        limit = 1024 * 1024
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1 or observed.st_size > limit:
            raise retirement.RootRetirementError("report must be a bounded single-link regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            encoded = stream.read(limit + 1)
        if len(encoded) > limit:
            raise retirement.RootRetirementError("report exceeds reading bound")
        text = encoded.decode("utf-8")
    finally:
        os.close(fd)
    if _MARKER not in text:
        return text
    try:
        prefix, suffix = text.split(_MARKER, 1)
        if not suffix.endswith("\n-->\n"):
            raise ValueError("incomplete marker")
        completion = json.loads(suffix[:-5])
        checksum = completion.pop("completion_sha256", None)
        if checksum != retirement.canonical_sha256(completion):
            raise ValueError("completion checksum mismatch")
        receipt = completion["receipt"]
        core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if receipt.get("receipt_sha256") != retirement.canonical_sha256(core) or receipt.get("result") != "applied":
            raise ValueError("cleanup receipt mismatch")
        summary = _cleanup_summary(receipt)
        if not prefix.endswith(summary):
            raise ValueError("cleanup summary mismatch")
        original = prefix[:-len(summary)]
        if _digest(original) != completion["identity"]["report_sha256"]:
            raise ValueError("original report checksum mismatch")
        return original
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise retirement.RootRetirementError(f"invalid report cleanup record: {error}") from error


def _remove_sidecars(directory: Path) -> None:
    if not directory.exists():
        return
    _safe(directory)
    allowed = {"transaction.json", "progress.json"}
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        members = os.listdir(fd)
        if any(name not in allowed for name in members):
            raise retirement.RootRetirementError("unexpected report-only transaction member")
        identities = {}
        for name in members:
            observed = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise retirement.RootRetirementError("unexpected linked transaction member")
            identities[name] = (observed.st_dev, observed.st_ino)
        for name in members:
            observed = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if identities[name] != (observed.st_dev, observed.st_ino):
                raise retirement.RootRetirementError("transaction member changed before cleanup")
            os.unlink(name, dir_fd=fd)
        os.fsync(fd)
        named = directory.stat(follow_symlinks=False)
        opened = os.fstat(fd)
        if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
            raise retirement.RootRetirementError("transaction directory changed before cleanup")
        directory.rmdir()
        parent_fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        os.close(fd)


def _read_regular_json(path: Path, *, limit: int = 16 * 1024 * 1024) -> tuple[dict, bytes]:
    """Read one bounded JSON file without following a final symlink."""
    _safe(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise retirement.RootRetirementError(f"Canopy closeout input is not a bounded single-link file: {path}")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(fd)
        named = os.stat(path, follow_symlinks=False)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if len(payload) > limit or identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or identity[:2] != (named.st_dev, named.st_ino):
            raise retirement.RootRetirementError(f"Canopy closeout input changed while reading: {path}")
    finally:
        os.close(fd)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise retirement.RootRetirementError(f"Canopy closeout input is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise retirement.RootRetirementError(f"Canopy closeout input is not an object: {path}")
    return value, payload


def _canonical_runtime_path(path: Path) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    _safe(path)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise retirement.RootRetirementError(f"Canopy closeout path is unavailable: {path}") from exc
    if resolved != path:
        raise retirement.RootRetirementError(f"Canopy closeout path is not canonical: {path}")
    return path


def _terminal_canopy_scratch_receipt_locked(
    relay_root: Path,
    roots: Sequence[Path],
    *,
    apply: bool = False,
    prior_plan: dict | None = None,
) -> dict | None:
    """Validate or remove only proven terminal relay `_work/source_view` links.

    The relay scratch links are disposable aliases made by the read-only Canopy
    projector. This preparer is intentionally limited to one exact cell root
    under the canonical relay namespace. It never opens or follows a target.
    """
    base = Path(canopy_supervision.DEFAULT_CANOPY_RELAY_ROOT)
    relay_root = Path(os.path.abspath(os.fspath(relay_root)))
    if not relay_root.is_relative_to(base):
        return None
    _safe(base)
    relay_root = _canonical_runtime_path(relay_root)
    base = _canonical_runtime_path(base)
    try:
        parts = relay_root.relative_to(base).parts
    except ValueError as exc:
        raise retirement.RootRetirementError("Canopy relay root escaped its canonical namespace") from exc
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise retirement.RootRetirementError("Canopy closeout requires one exact relay group/cell root")
    group_id, cell_id = parts

    root_fd = os.open(relay_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            work_fd = os.open("_work", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise retirement.RootRetirementError("Canopy relay work root is not a real directory") from exc
        try:
            try:
                view_fd = os.open("source_view", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=work_fd)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise retirement.RootRetirementError("Canopy source view is not a real directory") from exc
            try:
                view_named = os.stat("source_view", dir_fd=work_fd, follow_symlinks=False)
                view_opened = os.fstat(view_fd)
                if (view_named.st_dev, view_named.st_ino) != (view_opened.st_dev, view_opened.st_ino):
                    raise retirement.RootRetirementError("Canopy source view changed while opening")
                if view_opened.st_dev != os.fstat(root_fd).st_dev:
                    raise retirement.RootRetirementError("Canopy source view crosses a filesystem boundary")
            finally:
                os.close(view_fd)
        finally:
            os.close(work_fd)
    finally:
        os.close(root_fd)

    # Bind this exact output root to a source run explicitly included in this
    # closeout. The manifest and provenance are the relay authority; the
    # supervisor receipt is deliberately not required because a manually
    # attached terminal relay may be absent from that receipt.
    relay_manifest, manifest_payload = _read_regular_json(relay_root / "relay_manifest.json")
    source_text = relay_manifest.get("source_root")
    if not isinstance(source_text, str):
        raise retirement.RootRetirementError("Canopy relay manifest has no source root")
    source_root = _canonical_runtime_path(Path(source_text))
    known_roots = {_canonical_runtime_path(Path(root)) for root in roots if Path(root).exists()}
    if source_root not in known_roots:
        raise retirement.RootRetirementError("Canopy relay source root is not part of this exact closeout")
    if relay_manifest.get("source_run_id") != source_root.name:
        raise retirement.RootRetirementError("Canopy relay source run identity mismatch")
    relay_id = relay_manifest.get("relay_id")
    if not isinstance(relay_id, str) or not relay_id:
        raise retirement.RootRetirementError("Canopy relay identity is missing")
    if relay_manifest.get("schema_version") != "llc_near_live_playback_relay_v0_manifest":
        raise retirement.RootRetirementError("unsupported Canopy relay manifest schema")
    if relay_manifest.get("source_state") not in {"source_complete", "source_interrupted"}:
        raise retirement.RootRetirementError("Canopy relay is not terminal")
    if relay_manifest.get("playback_file_name") != "playback_snapshot.json" or relay_manifest.get("provenance_file_name") != "provenance.json" or relay_manifest.get("cross_check_file_name") != "cross_check.json" or relay_manifest.get("status_file_name") != "relay_status.json":
        raise retirement.RootRetirementError("Canopy relay manifest names unexpected source artifacts")

    manifest_hashes = {
        "playback_snapshot.json": relay_manifest.get("playback_sha256"),
        "provenance.json": relay_manifest.get("provenance_sha256"),
        "cross_check.json": relay_manifest.get("cross_check_sha256"),
    }
    payloads: dict[str, dict] = {}
    payload_hashes: dict[str, str] = {}
    for name, expected_hash in manifest_hashes.items():
        value, payload = _read_regular_json(relay_root / name)
        if not isinstance(expected_hash, str) or hashlib.sha256(payload).hexdigest() != expected_hash:
            raise retirement.RootRetirementError(f"Canopy relay manifest hash mismatch: {name}")
        payloads[name] = value
        payload_hashes[name] = expected_hash
    status, _ = _read_regular_json(relay_root / "relay_status.json")
    if status.get("schema_version") != "llc_near_live_playback_relay_v0_status" or status.get("relay_id") != relay_id or status.get("state") != relay_manifest.get("source_state") or status.get("relay_sequence") != relay_manifest.get("sequence_number"):
        raise retirement.RootRetirementError("Canopy relay terminal status does not match its manifest")
    provenance = payloads["provenance.json"]
    if provenance.get("source_root") != str(source_root) or provenance.get("source_root_resolved") != str(source_root) or provenance.get("source_cell_id") != cell_id:
        raise retirement.RootRetirementError("Canopy relay provenance does not bind the source run/cell")
    source_files = provenance.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise retirement.RootRetirementError("Canopy relay provenance has no source file inventory")
    expected_links: dict[str, str] = {}
    for row in source_files:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64:
            raise retirement.RootRetirementError("Canopy relay source file provenance is malformed")
        source_file = Path(row["path"])
        if not source_file.is_absolute() or ".." in source_file.parts or str(source_file) != row["path"]:
            raise retirement.RootRetirementError("Canopy relay source file path is not canonical")
        try:
            relative = source_file.relative_to(source_root).as_posix()
        except ValueError as exc:
            raise retirement.RootRetirementError("Canopy relay provenance points outside its source root") from exc
        parsed = PurePosixPath(relative)
        if not relative or parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts) or relative in expected_links:
            raise retirement.RootRetirementError("Canopy relay source file path is unsafe or duplicated")
        expected_links[relative] = str(source_file)

    # Inspect liveness immediately before opening our own directory handles:
    # the canonical probe intentionally treats every process FD under the
    # target as a live reference, including this process's FDs.
    if retirement.RunStorageGuard.run_root_is_reserved(relay_root):
        raise retirement.RootRetirementError("active storage reservation blocks Canopy scratch cleanup")
    references, errors = retirement._process_references(relay_root)
    if errors or references:
        raise retirement.RootRetirementError("Canopy relay is not quiescent for scratch cleanup")

    root_fd = os.open(relay_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        work_fd = os.open("_work", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            view_fd = os.open("source_view", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=work_fd)
            try:
                links: list[dict] = []

                def inspect(fd: int, prefix: str = "") -> None:
                    for name in sorted(os.listdir(fd), key=os.fsencode):
                        if name in {".", ".."} or "/" in name:
                            raise retirement.RootRetirementError("unsafe Canopy source-view member")
                        relative = f"{prefix}/{name}" if prefix else name
                        observed = os.stat(name, dir_fd=fd, follow_symlinks=False)
                        if stat.S_ISDIR(observed.st_mode):
                            if not any(path.startswith(relative + "/") for path in expected_links):
                                raise retirement.RootRetirementError("unexpected Canopy source-view directory")
                            child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                            try:
                                if os.fstat(child_fd).st_dev != os.fstat(root_fd).st_dev:
                                    raise retirement.RootRetirementError("Canopy source view crosses a filesystem boundary")
                                inspect(child_fd, relative)
                            finally:
                                os.close(child_fd)
                        elif stat.S_ISLNK(observed.st_mode):
                            expected_target = expected_links.get(relative)
                            if expected_target is None:
                                raise retirement.RootRetirementError("unexpected Canopy source-view symlink")
                            target = os.readlink(name, dir_fd=fd)
                            if target != expected_target:
                                raise retirement.RootRetirementError("Canopy source-view symlink target differs from provenance")
                            links.append({"path": relative, "device": int(observed.st_dev), "inode": int(observed.st_ino), "target": target})
                        else:
                            raise retirement.RootRetirementError("unexpected Canopy source-view member type")

                inspect(view_fd)
                if not links and prior_plan is None:
                    return None
                root_stat = os.fstat(root_fd)
                source_stat = source_root.stat(follow_symlinks=False)
                plan = {
                    "relay_root": str(relay_root),
                    "source_root": str(source_root),
                    "cell_id": cell_id,
                    "relay_id": relay_id,
                    "relay_root_identity": [int(root_stat.st_dev), int(root_stat.st_ino)],
                    "source_root_identity": [int(source_stat.st_dev), int(source_stat.st_ino)],
                    "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                    "provenance_sha256": payload_hashes["provenance.json"],
                    "source_files_sha256": retirement.canonical_sha256(source_files),
                    "links": links,
                }
                if prior_plan is not None:
                    identity_keys = (
                        "relay_root", "source_root", "cell_id", "relay_id",
                        "relay_root_identity", "source_root_identity", "manifest_sha256",
                        "provenance_sha256", "source_files_sha256",
                    )
                    if any(prior_plan.get(key) != plan.get(key) for key in identity_keys):
                        raise retirement.RootRetirementError("terminal Canopy scratch preparation identity changed")
                    prior_links = {row.get("path"): row for row in prior_plan.get("links", []) if isinstance(row, dict)}
                    if any(prior_links.get(row["path"]) != row for row in links):
                        raise retirement.RootRetirementError("terminal Canopy scratch link identity changed")
                if not apply:
                    return plan
                # Recheck each exact inode and target before unlink. Unknown
                # or retargeted entries were rejected by the full traversal
                # above before any scratch changes.
                for row in links:
                    components = PurePosixPath(row["path"]).parts
                    parent_fd = os.dup(view_fd)
                    try:
                        for component in components[:-1]:
                            child_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                            os.close(parent_fd)
                            parent_fd = child_fd
                        current = os.stat(components[-1], dir_fd=parent_fd, follow_symlinks=False)
                        target = os.readlink(components[-1], dir_fd=parent_fd)
                        if not stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino) != (row["device"], row["inode"]) or target != row["target"]:
                            raise retirement.RootRetirementError("Canopy source-view symlink changed before unlink")
                        os.unlink(components[-1], dir_fd=parent_fd)
                        os.fsync(parent_fd)
                    finally:
                        os.close(parent_fd)
            finally:
                os.close(view_fd)
        finally:
            os.close(work_fd)
    finally:
        os.close(root_fd)
    return {"relay_root": str(relay_root), "source_root": str(source_root), "cell_id": cell_id, "relay_id": relay_id, "removed_links": links, "plan": plan}


def _terminal_canopy_scratch_receipt(
    relay_root: Path,
    roots: Sequence[Path],
    *,
    apply: bool = False,
    prior_plan: dict | None = None,
) -> dict | None:
    base = Path(canopy_supervision.DEFAULT_CANOPY_RELAY_ROOT)
    candidate = Path(os.path.abspath(os.fspath(relay_root)))
    if not candidate.is_relative_to(base):
        return None
    relative_parts = candidate.relative_to(base).parts
    if len(relative_parts) != 2 or any(part in {"", ".", ".."} for part in relative_parts):
        return None
    from .artifact_lifecycle import run_artifact_transaction_fence_v1

    with run_artifact_transaction_fence_v1(candidate):
        return _terminal_canopy_scratch_receipt_locked(
            candidate, roots, apply=apply, prior_plan=prior_plan
        )


def closeout_report_only_run_v1(
    *, report_path: Path, report_markdown: str, roots: Sequence[Path],
    source_commit: str, run_id: str, apply: bool = False,
) -> dict:
    """Publish a report, seal exact roots, and optionally retire them safely.

    A pending transaction is resumable only with identical inputs. Changes to
    sealed roots fail in the existing actuator, never trigger a new inventory.
    """
    report_path = Path(report_path).expanduser().absolute()
    _safe(report_path)
    if report_path.suffix != ".md" or _MARKER in report_markdown:
        raise retirement.RootRetirementError("a plain Markdown report is required")
    sections = re.split(r"(?m)^#{1,6}\s+", report_markdown)
    found = {part.split("\n", 1)[0].strip().lower(): part.split("\n", 1)[1].strip()
             for part in sections[1:] if "\n" in part}
    if any(not found.get(name) for name in ("inputs", "settings", "results", "limitations", "reproduction")):
        raise retirement.RootRetirementError("report requires nonempty Inputs, Settings, Results, Limitations, Reproduction sections")
    root_paths = [Path(root).expanduser().absolute() for root in roots]
    for root in root_paths:
        _safe(root)
        if any(str(root) in target for target in re.findall(r"\]\(([^)]+)\)", report_markdown)):
            raise retirement.RootRetirementError("report links cannot depend on retired roots")
    identity = {"report_sha256": _digest(report_markdown), "roots": [str(p) for p in root_paths],
                "source_commit": source_commit, "run_id": run_id}
    identity_sha = retirement.canonical_sha256(identity)
    transaction = report_path.with_name("." + report_path.name + ".closeout")
    _safe(transaction)
    reference = {"path": str(report_path), "sha256": _digest(report_markdown)}
    request = retirement.validate_root_retirement_request_v1({
        "schema_version": retirement.REQUEST_SCHEMA_VERSION,
        "retirement_id": identity_sha, "programme_id": run_id, "work_id": run_id,
        "sealed_at_utc": retirement._now(),
        "source": {"repository": "emergence", "commit_sha": source_commit},
        "allowed_runtime_roots": [str(p) for p in sorted(retirement.CANONICAL_RUNTIME_ROOTS)],
        "compact_authority": reference, "terminal_authorities": [reference],
        "validation": {"predicates": {key: True for key in retirement.REQUIRED_PREDICATES},
                       "semantic_review": {"status": "accepted_with_limitations", "reviewer": "report-only-terminal-owner"}},
        "roots": [{"root_id": str(i), "path": str(p), "kind": "run_artifact", "terminal_status": "completed",
                   "unresolved_rx_bytes": 0, "live_reference_count": 0, "authority_receipts": [reference]}
                  for i, p in enumerate(root_paths)],
        "claim_limits": ["Caller owns terminal and last-consumer decisions; mechanical review is not independent or scientific review."],
    })
    report_path.parent.mkdir(parents=True, exist_ok=True)
    parent_fd = os.open(report_path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        if report_path.exists():
            existing = report_path.read_text(encoding="utf-8")
            if _MARKER in existing:
                prefix, suffix = existing.split(_MARKER, 1)
                if not suffix.endswith("\n-->\n"):
                    raise retirement.RootRetirementError("invalid cleanup completion record")
                completion = json.loads(suffix[:-5])
                checksum = completion.pop("completion_sha256", None)
                if checksum != retirement.canonical_sha256(completion) or completion.get("identity") != identity:
                    raise retirement.RootRetirementError("cleanup completion identity mismatch")
                receipt = completion["receipt"]
                receipt_core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
                if receipt.get("receipt_sha256") != retirement.canonical_sha256(receipt_core) or receipt.get("result") != "applied":
                    raise retirement.RootRetirementError("invalid applied cleanup receipt")
                if prefix != report_markdown + _cleanup_summary(receipt):
                    raise retirement.RootRetirementError("completed report content differs")
                for row in request["roots"]:
                    root = Path(row["path"])
                    stage = root.parent / retirement._stage_name(identity_sha, row["root_id"])
                    if root.exists() or root.is_symlink() or stage.exists() or stage.is_symlink():
                        raise retirement.RootRetirementError("completed cleanup root or stage has reappeared")
                _remove_sidecars(transaction)
                return {"status": "COMPLETED", "report_path": str(report_path), "cleanup": receipt}
            if existing != report_markdown:
                raise retirement.RootRetirementError("existing report differs from requested report")
        else:
            _write(report_path, report_markdown)
        transaction.mkdir(exist_ok=True)
        bundle_path = transaction / "transaction.json"
        def build_bundle(scratch_receipts: list[dict]) -> dict:
            inventory = retirement.build_root_retirement_inventory_v1(request)
            intent = retirement.build_root_retirement_intent_v1(request=request, inventory=inventory, status="PREPARED")
            review_core = {"schema_version": retirement.REVIEW_SCHEMA_VERSION, "retirement_id": identity_sha,
                           "request_sha256": retirement.canonical_sha256(request), "inventory_sha256": inventory["inventory_sha256"],
                           "status": "accepted_with_limitations", "reviewer": "report-only-terminal-owner",
                           "reviewed_at_utc": retirement._now(), "findings": ["Exact roots sealed; report published before cleanup."],
                           "limitations": ["REVIEWER_INDEPENDENCE_UNAVAILABLE", "Mechanical inventory review; no scientific or qualification claim."]}
            return {"identity": identity, "request": request, "inventory": inventory, "intent": intent,
                    "canopy_terminal_scratch_cleanup": scratch_receipts,
                    "review": {**review_core, "review_sha256": retirement.canonical_sha256(review_core)}}

        if bundle_path.exists():
            bundle = _read(bundle_path)
            if bundle.get("identity") != identity:
                raise retirement.RootRetirementError("pending cleanup identity differs")
            original = retirement.validate_root_retirement_request_v1(bundle["request"])
            if {k: v for k, v in original.items() if k != "sealed_at_utc"} != {k: v for k, v in request.items() if k != "sealed_at_utc"}:
                raise retirement.RootRetirementError("pending cleanup request differs")
            request = bundle["request"]
            if bundle.get("phase") == "CANOPY_SCRATCH_PENDING":
                preparation_sha = bundle.get("preparation_sha256")
                preparation_core = {key: value for key, value in bundle.items() if key != "preparation_sha256"}
                if preparation_sha != retirement.canonical_sha256(preparation_core):
                    raise retirement.RootRetirementError("Canopy scratch preparation seal mismatch")
                plans = bundle.get("canopy_scratch_plans")
                if not isinstance(plans, list) or not plans:
                    raise retirement.RootRetirementError("Canopy scratch preparation journal is malformed")
                validated = []
                for plan in plans:
                    if not isinstance(plan, dict):
                        raise retirement.RootRetirementError("Canopy scratch preparation plan is malformed")
                    current = _terminal_canopy_scratch_receipt(
                        Path(str(plan.get("relay_root"))), root_paths, apply=False, prior_plan=plan
                    )
                    if current is None:
                        raise retirement.RootRetirementError("prepared Canopy scratch root disappeared")
                    validated.append(current)
                if not apply:
                    return {"status": "CLEANUP_PENDING", "report_path": str(report_path),
                            "transaction_path": str(transaction), "phase": "CANOPY_SCRATCH_PENDING"}
                scratch_receipts = [
                    _terminal_canopy_scratch_receipt(
                        Path(str(plan["relay_root"])), root_paths, apply=True, prior_plan=plan
                    )
                    for plan in plans
                ]
                bundle = build_bundle([receipt for receipt in scratch_receipts if receipt is not None])
                _write(bundle_path, json.dumps(bundle, sort_keys=True))
            elif not all(key in bundle for key in ("inventory", "intent", "review")):
                raise retirement.RootRetirementError("pending cleanup transaction is incomplete; cannot reseal")
        else:
            if any(transaction.iterdir()):
                raise retirement.RootRetirementError("incomplete transaction requires recovery; cannot reseal")
            canopy_plans = [
                receipt
                for root in root_paths
                if (receipt := _terminal_canopy_scratch_receipt(root, root_paths, apply=False)) is not None
            ]
            if canopy_plans:
                prep_core = {"identity": identity, "request": request, "phase": "CANOPY_SCRATCH_PENDING",
                             "canopy_scratch_plans": canopy_plans}
                bundle = {**prep_core, "preparation_sha256": retirement.canonical_sha256(prep_core)}
                _write(bundle_path, json.dumps(bundle, sort_keys=True))
                if not apply:
                    return {"status": "CLEANUP_PENDING", "report_path": str(report_path),
                            "transaction_path": str(transaction), "phase": "CANOPY_SCRATCH_PENDING"}
                for plan in canopy_plans:
                    current = _terminal_canopy_scratch_receipt(
                        Path(str(plan["relay_root"])), root_paths, apply=False, prior_plan=plan
                    )
                    if current is None:
                        raise retirement.RootRetirementError("prepared Canopy scratch root disappeared")
                scratch_receipts = [
                    _terminal_canopy_scratch_receipt(
                        Path(str(plan["relay_root"])), root_paths, apply=True, prior_plan=plan
                    )
                    for plan in canopy_plans
                ]
                bundle = build_bundle([receipt for receipt in scratch_receipts if receipt is not None])
            else:
                bundle = build_bundle([])
            _write(bundle_path, json.dumps(bundle, sort_keys=True))
        if not apply:
            return {"status": "CLEANUP_PENDING", "report_path": str(report_path), "transaction_path": str(transaction)}
        progress = transaction / "progress.json"
        intent = _read(progress) if progress.exists() else bundle["intent"]
        receipt = retirement.apply_root_retirement_v1(request=request, inventory=bundle["inventory"], intent=intent,
                                                     review=bundle["review"], progress_path=progress)
        completion = {"identity": identity, "receipt": receipt}
        completion["completion_sha256"] = retirement.canonical_sha256(completion)
        _write(report_path, report_markdown + _cleanup_summary(receipt) + _MARKER + json.dumps(completion, sort_keys=True) + "\n-->\n")
        _remove_sidecars(transaction)
        return {"status": "COMPLETED", "report_path": str(report_path), "cleanup": receipt}
    finally:
        os.close(parent_fd)
