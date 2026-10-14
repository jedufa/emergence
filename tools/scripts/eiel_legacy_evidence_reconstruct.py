#!/usr/bin/env python3
"""Bounded reconstruction of lifecycle authority for pre-lifecycle roots.

The normal lifecycle actuator never infers purpose from an old file.  This
tool is the explicit one-time bridge for roots that predate manifests:

* inspect one exact root (never an aggregate storage parent);
* seal bounded directory/file segments with hashes;
* record evidence references and conservative candidate dispositions;
* emit a manifest marked ``REQUIRES_REVIEW``;
* optionally promote it with a complete, hash-bound review receipt.

No deletion or archive is performed here.  Once promoted, the existing
manifest-bound prune/archive machinery is the only actuator.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from engine.campaigns.artifact_lifecycle import (
    ArtifactLifecycleError,
    RETENTION_CLASSES,
    apply_legacy_evidence_review_v1,
    build_artifact_disposition_manifest_v1,
    build_sealed_artifact_inventory_row_v1,
    resolve_run_artifact_lifecycle_v1,
    validate_artifact_disposition_manifest_v1,
    validate_legacy_evidence_review_receipt_v1,
)
from engine.campaigns.canonical import canonical_sha256, pretty_json


MANIFEST_NAME = "artifact_disposition_manifest.json"
DEFAULT_MAX_OBJECTS = 4096
DEFAULT_MAX_BYTES = 512 * 1024 * 1024 * 1024

# These are candidate rules only.  They never authorize release by
# themselves; the generated manifest remains review-gated until every row is
# explicitly decided by an evidence review receipt.
RELEASE_DIRECTORY_NAMES = frozenset(
    {
        "_source_cache_v2",
        "cache",
        "scratch",
        "tmp",
        "staging",
        "intermediate",
        "objects",
        "source_projections_v2",
    }
)
DURABLE_DIRECTORY_NAMES = frozenset(
    {
        "checkpoints",
        "checkpoint_bundles_v2",
        "cells",
        "durable",
        "evidence",
        "qualification",
        "qualprep",
        "closeout",
        "contracts",
    }
)
DIAGNOSTIC_DIRECTORY_NAMES = frozenset(
    {"debug", "diagnostic", "diagnostics", "forensic", "traces", "trace", "raw"}
)
CONTROL_FILE_TOKENS = (
    "manifest",
    "receipt",
    "passport",
    "report",
    "summary",
    "metrics",
    "contract",
    "qualification",
    "closeout",
    "plan",
)
DIAGNOSTIC_FILE_TOKENS = ("debug", "diagnostic", "forensic", "trace", "failure")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_root(root: Path) -> Path:
    if root.expanduser().is_symlink():
        raise ArtifactLifecycleError("legacy reconstruction root cannot be a symlink")
    resolved = root.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ArtifactLifecycleError("legacy reconstruction root must be a directory")
    forbidden = {
        Path("/"),
        Path(public_runtime_path()),
        Path(public_runtime_path('')),
        Path(public_runtime_path('evidence_raw')),
        Path(public_runtime_path('campaigns')),
        Path(public_runtime_path('live')),
        Path(public_runtime_path('builds')),
    }
    if resolved in forbidden:
        raise ArtifactLifecycleError(
            "legacy reconstruction requires one exact historical root, not an aggregate storage parent"
        )
    return resolved


def _path_tokens(path: Path) -> set[str]:
    return {part.lower() for part in path.parts}


def _file_kind(path: Path, *, root: Path) -> tuple[str, str]:
    lower = path.name.lower()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ArtifactLifecycleError(f"legacy file escapes reconstruction root: {path}") from exc
    # Classify only directories inside the selected root.  Using absolute path
    # parts would let an unrelated ancestor named ``cache``/``scratch`` turn a
    # direct historical file into a release candidate.
    tokens = {part.lower() for part in relative.parts[:-1]}
    if any(token in lower for token in DIAGNOSTIC_FILE_TOKENS):
        return "diagnostic", "R3"
    if any(token in lower for token in CONTROL_FILE_TOKENS):
        return "control", "R5"
    if "checkpoint" in lower or "bundle_gen" in lower:
        return "checkpoint", "R4"
    # A file inside one of the explicitly reproducible directories is handled
    # as part of that sealed tree, but keep this branch for direct root files.
    if tokens & RELEASE_DIRECTORY_NAMES:
        return "reproducible", "R2"
    return "unresolved", "RX"


def _directory_kind(path: Path) -> tuple[str, str] | None:
    name = path.name.lower()
    if name in RELEASE_DIRECTORY_NAMES:
        return "reproducible", "R2"
    if name in DURABLE_DIRECTORY_NAMES:
        return "checkpoint_or_durable", "R4"
    if name in DIAGNOSTIC_DIRECTORY_NAMES:
        return "diagnostic", "R3"
    return None


def _collect_segments(root: Path, *, max_objects: int) -> list[tuple[Path, str, str]]:
    segments: list[tuple[Path, str, str]] = []

    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if child.is_symlink():
                raise ArtifactLifecycleError(f"legacy root contains a symlink: {child}")
            if child.is_dir():
                classification = _directory_kind(child)
                if classification is not None:
                    family, retention_class = classification
                    segments.append((child, family, retention_class))
                else:
                    visit(child)
            elif child.is_file():
                family, retention_class = _file_kind(child, root=root)
                segments.append((child, family, retention_class))
            else:
                raise ArtifactLifecycleError(f"legacy root contains unsupported object: {child}")
            if len(segments) > max_objects:
                raise ArtifactLifecycleError(
                    "legacy reconstruction object bound exceeded; rerun with a larger --max-objects or a narrower root"
                )

    visit(root)
    return segments


def _segment_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for member in path.rglob("*"):
        if member.is_symlink():
            raise ArtifactLifecycleError(f"legacy segment contains a symlink: {member}")
        if member.is_file():
            total += int(member.stat().st_size)
    return total


def _reference_row(path: Path, *, root: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ArtifactLifecycleError(f"evidence reference is not a regular file: {path}")
    try:
        display = str(resolved.relative_to(root)).replace("\\", "/")
    except ValueError:
        display = str(resolved)
    return {"kind": "evidence_reference", "path": display, "sha256": _sha256_file(resolved)}


def _discover_local_references(root: Path, *, max_count: int = 256) -> list[dict[str, str]]:
    candidates: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        if not path.is_file() or path.is_symlink() or path.name == MANIFEST_NAME:
            continue
        lower = path.name.lower()
        if any(token in lower for token in CONTROL_FILE_TOKENS):
            candidates.append(path)
        if len(candidates) >= max_count:
            break
    return [_reference_row(path, root=root) for path in candidates]


def _refresh_summary(manifest: dict[str, Any]) -> None:
    rows = [row for row in manifest.get("artifacts", []) if isinstance(row, Mapping)]
    manifest["summary"] = {
        "produced_bytes": sum(int(row.get("bytes", 0)) for row in rows),
        "release_eligible_bytes": sum(
            int(row.get("bytes", 0)) for row in rows if row.get("release_eligible") is True
        ),
        "unresolved_rx_bytes": sum(
            int(row.get("bytes", 0)) for row in rows if row.get("retention_class") == "RX"
        ),
        "by_retention_class": {
            retention_class: {
                "objects": sum(
                    1 for row in rows if row.get("retention_class") == retention_class
                ),
                "bytes": sum(
                    int(row.get("bytes", 0))
                    for row in rows
                    if row.get("retention_class") == retention_class
                ),
            }
            for retention_class in RETENTION_CLASSES
        },
        "lifecycle_violations": sum(
            1
            for row in rows
            if "violates the lifecycle observability policy" in str(row.get("reason"))
        ),
    }


def _rebind_hash(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest.pop("manifest_sha256", None)
    return {**manifest, "manifest_sha256": canonical_sha256(manifest)}


def reconstruct_root(
    root: Path,
    *,
    max_objects: int = DEFAULT_MAX_OBJECTS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    evidence_references: Iterable[Path] = (),
) -> dict[str, Any]:
    """Create a review-gated reconstructed disposition for one root."""

    runtime_root = _safe_root(root)
    if (runtime_root / MANIFEST_NAME).exists():
        raise ArtifactLifecycleError(
            "root already has a disposition manifest; review or inspect the existing authority"
        )
    if max_objects < 1 or max_bytes < 1:
        raise ArtifactLifecycleError("reconstruction bounds must be positive")
    segments = _collect_segments(runtime_root, max_objects=max_objects)
    segment_bytes = sum(_segment_bytes(path) for path, _, _ in segments)
    if segment_bytes > max_bytes:
        raise ArtifactLifecycleError(
            f"legacy reconstruction byte bound exceeded: {segment_bytes} > {max_bytes}"
        )
    references = _discover_local_references(runtime_root)
    seen_refs = {(row["path"], row["sha256"]) for row in references}
    for path in evidence_references:
        row = _reference_row(path, root=runtime_root)
        if (row["path"], row["sha256"]) not in seen_refs:
            references.append(row)
            seen_refs.add((row["path"], row["sha256"]))
    inventory: list[dict[str, Any]] = []
    for path, family, retention_class in segments:
        inventory.append(
            build_sealed_artifact_inventory_row_v1(
                run_root=runtime_root,
                path=path,
                artifact_family=f"legacy_{family}",
                retention_class=retention_class,
                owner_writer="legacy-reconstruction",
                candidate_basis="bounded_path_and_control_evidence_review_required",
            )
        )
    total_budget = max(4, segment_bytes + 1)
    contract = resolve_run_artifact_lifecycle_v1(
        None,
        total_max_bytes=total_budget,
        checkpoint_schedule=[],
        legacy=True,
    )
    plan = {
        "campaign_id": runtime_root.name,
        "plan_sha256": canonical_sha256(
            {
                "runtime_root": str(runtime_root),
                "segment_bytes": segment_bytes,
                "evidence_basis": references,
            }
        ),
        "artifact_lifecycle": contract,
        "execution": {"generations": 0, "checkpoints": []},
    }
    metadata = {
        "schema_version": "eiel_legacy_evidence_reconstruction_v1",
        "status": "REQUIRES_REVIEW",
        "source_manifest_status": "ABSENT",
        "method": "bounded_legacy_evidence_reconstruction_v1",
        "created_at_utc": _now(),
        "scope": {
            "max_objects": int(max_objects),
            "max_bytes": int(max_bytes),
            "inventory_complete": True,
            "candidate_object_count": len(inventory),
            "candidate_bytes": segment_bytes,
        },
        "evidence_basis": references,
        "review": {
            "required": True,
            "status": "required",
            "receipt_sha256": None,
            "reviewer": None,
            "accepted_at_utc": None,
            "allow_automatic_release": False,
        },
    }
    manifest = build_artifact_disposition_manifest_v1(
        plan,
        run_root=runtime_root,
        terminal_status="unknown",
        live_reference_status="unknown",
        audit_hold_status="not_required",
        qualification_hold_status="not_required",
        artifact_inventory=inventory,
        inventory_complete=True,
        legacy_reconstruction=metadata,
    )
    # Keep candidate intent visible while refusing to authorize it.  The
    # review receipt must turn each row into a final semantic disposition.
    for row in manifest["artifacts"]:
        retention_class = row.get("retention_class")
        if retention_class in {"R0", "R1", "R2"}:
            row["disposition"] = "RELEASE"
            row["release_eligible"] = False
            row["release_predicate"] = "candidate release; explicit evidence review required"
            row["reason"] = "candidate release; explicit evidence review required"
        elif retention_class in {"R3", "R4", "R5"}:
            row["disposition"] = "RETAIN_DURABLE"
            row["release_eligible"] = False
            row["release_predicate"] = "candidate durable material; explicit evidence review required"
            row["reason"] = "candidate durable material; explicit evidence review required"
        else:
            row["disposition"] = "RX"
            row["release_eligible"] = False
            row["reason"] = "purpose unresolved; explicit evidence review required"
    _refresh_summary(manifest)
    return _rebind_hash(manifest)


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactLifecycleError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(pretty_json(dict(value)), encoding="utf-8")
    os.replace(temporary, target)


def reconstruct_or_review(
    root: Path,
    *,
    review_receipt: Path | None,
    max_objects: int,
    max_bytes: int,
    evidence_references: Iterable[Path],
) -> dict[str, Any]:
    runtime_root = _safe_root(root)
    manifest_path = runtime_root / MANIFEST_NAME
    if review_receipt is None:
        return reconstruct_root(
            runtime_root,
            max_objects=max_objects,
            max_bytes=max_bytes,
            evidence_references=evidence_references,
        )
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ArtifactLifecycleError(
            "--review-receipt requires the review-gated candidate manifest in the root"
        )
    candidate = validate_artifact_disposition_manifest_v1(load_json_object(manifest_path))
    receipt = validate_legacy_evidence_review_receipt_v1(load_json_object(review_receipt))
    return apply_legacy_evidence_review_v1(candidate, receipt)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded legacy evidence reconstruction (review-gated; no deletion)"
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--max-objects", type=int, default=DEFAULT_MAX_OBJECTS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--evidence-reference", action="append", type=Path, default=[])
    parser.add_argument("--review-receipt", type=Path)
    parser.add_argument("--write", action="store_true", help="write the candidate/accepted manifest into the root")
    parser.add_argument("--out", type=Path, help="write the JSON result to this path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = reconstruct_or_review(
            args.root,
            review_receipt=args.review_receipt,
            max_objects=args.max_objects,
            max_bytes=args.max_bytes,
            evidence_references=args.evidence_reference,
        )
        if args.write:
            _write_json(args.root.expanduser().resolve() / MANIFEST_NAME, result)
        output = {
            "ok": True,
            "mode": "accepted_review" if args.review_receipt is not None else "reconstruction_preview",
            "manifest_path": str(args.root.expanduser().resolve() / MANIFEST_NAME),
            "manifest": result,
            "written": bool(args.write),
        }
    except (OSError, UnicodeError, json.JSONDecodeError, ArtifactLifecycleError) as exc:
        output = {
            "ok": False,
            "mode": "blocked",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        text = json.dumps(output, indent=2, sort_keys=True) + "\n"
        if args.out is not None:
            args.out.expanduser().resolve().write_text(text, encoding="utf-8")
        print(text, end="")
        return 1
    text = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.expanduser().resolve().write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
