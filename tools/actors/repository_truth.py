"""Durable repository-delivery, adoption, and integration truth for EIEL.

Git is an execution substrate, not a permanent evidence archive.  This module
captures the exact commit graph, file blobs, and binary patch needed to verify a
delivery after live Git objects disappear.  It also records explicit adoption
of pre-existing changes and content-equivalent integration (merge, rebase, or
cherry-pick) without pretending those operations were performed by the original
Session.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.actors.common import (
    ActorError,
    actor_state_root,
    atomic_write_bytes,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    read_json,
    require_explicit_schema_version,
    stable_id,
    utc_now,
)

DELIVERY_PROOF_SCHEMA_VERSION = "eiel_repository_delivery_proof_v1"
ADOPTION_RECEIPT_SCHEMA_VERSION = "eiel_repository_adoption_receipt_v1"
INTEGRATION_RECEIPT_SCHEMA_VERSION = "eiel_repository_integration_receipt_v1"
STORAGE_MERGE_CLOSEOUT_RECEIPT_SCHEMA_VERSION = "eiel_storage_merge_closeout_v1"
STATE_NAMESPACE_SCHEMA_VERSION = "eiel_repository_state_namespace_v1"
RELOCATION_RECEIPT_SCHEMA_VERSION = "eiel_repository_relocation_receipt_v1"
ARTIFACT_REFERENCE_SCHEMA_VERSION = "eiel_repository_truth_artifact_ref_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40,64}$")


def _git(repo_root: str | Path, *args: str, binary: bool = False) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", "-C", str(Path(repo_root).expanduser().resolve()), *args],
        text=not binary,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _resolve_commit(repo_root: str | Path, value: str, *, label: str) -> str:
    requested = str(value or "").strip()
    result = _git(repo_root, "rev-parse", "--verify", f"{requested}^{{commit}}")
    if result.returncode != 0 or not str(result.stdout or "").strip():
        raise ActorError(
            f"{label} is not available in the repository",
            code="repository_commit_invalid",
            problems=[str(result.stderr or "").strip() or requested],
        )
    return str(result.stdout).strip().lower()


def _git_text(repo_root: str | Path, *args: str, label: str) -> str:
    result = _git(repo_root, *args)
    if result.returncode != 0:
        raise ActorError(
            f"cannot inspect {label}",
            code="repository_truth_capture_failed",
            problems=[str(result.stderr or "").strip() or "git command failed"],
        )
    return str(result.stdout or "").strip()


def _blob_id(repo_root: str | Path, commit: str, path: str) -> str | None:
    result = _git(repo_root, "rev-parse", "--verify", f"{commit}:{path}")
    if result.returncode != 0:
        return None
    value = str(result.stdout or "").strip().lower()
    return value if _GIT_OBJECT_RE.fullmatch(value) else None


def _git_common_dir(repo_root: str | Path) -> Path:
    result = _git(repo_root, "rev-parse", "--git-common-dir")
    if result.returncode != 0 or not str(result.stdout or "").strip():
        raise ActorError("repository has no resolvable Git common directory", code="repository_git_common_dir_missing")
    raw = Path(str(result.stdout).strip())
    return raw.resolve() if raw.is_absolute() else (Path(repo_root).expanduser().resolve() / raw).resolve()


def ensure_repository_state_namespace(
    *,
    state_root: str | Path | None,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Bind one Git repository family to one authoritative actor-state namespace."""

    state = actor_state_root(state_root)
    state.mkdir(parents=True, exist_ok=True)
    common = _git_common_dir(repo_root)
    namespace_path = state / "repository-state-namespace.json"
    git_binding_path = common / "eiel-actor-state-namespace.json"
    if namespace_path.is_file():
        namespace = read_json(namespace_path, name="repository state namespace")
        require_explicit_schema_version(namespace, expected=STATE_NAMESPACE_SCHEMA_VERSION, label="repository state namespace")
    else:
        namespace = {
            "schema_version": STATE_NAMESPACE_SCHEMA_VERSION,
            "namespace_id": stable_id("STATE", {"state_root": str(state), "git_common_dir": str(common)}),
            "state_root_path_at_binding": str(state),
            "git_common_dir_at_binding": str(common),
            "bound_at_utc": utc_now(None),
        }
        namespace["namespace_sha256"] = canonical_sha256(namespace)
        atomic_write_json(namespace_path, namespace)
    namespace_id = str(namespace.get("namespace_id") or "")
    if not namespace_id:
        raise ActorError("repository state namespace has no namespace_id", code="repository_state_namespace_invalid")
    if git_binding_path.is_file():
        binding = read_json(git_binding_path, name="Git repository state namespace binding")
        if str(binding.get("namespace_id") or "") != namespace_id:
            raise ActorError(
                "Git repository is already bound to another actor-state namespace",
                code="repository_state_namespace_conflict",
                problems=[
                    f"repository namespace: {binding.get('namespace_id')}",
                    f"requested namespace: {namespace_id}",
                    "Create an explicit state fork or reconcile the existing namespace; do not claim the same delivery independently.",
                ],
            )
    else:
        atomic_write_json(git_binding_path, {
            "schema_version": STATE_NAMESPACE_SCHEMA_VERSION,
            "namespace_id": namespace_id,
            "namespace_sha256": namespace.get("namespace_sha256"),
            "state_root_path_at_binding": str(state),
            "bound_at_utc": utc_now(None),
        })
    if Path(str(namespace.get("git_common_dir_at_binding") or common)).resolve() != common:
        raise ActorError(
            "actor-state namespace is already bound to another Git repository family",
            code="repository_state_namespace_conflict",
            problems=[str(namespace.get("git_common_dir_at_binding")), str(common)],
        )
    return dict(namespace)


def _root(state_root: str | Path | None) -> Path:
    return actor_state_root(state_root) / "repository-truth"


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> Path:
    expected = _canonical_bytes(value)
    if path.exists():
        observed = path.read_bytes()
        if observed != expected:
            raise ActorError(
                f"immutable repository-truth artifact conflict: {path}",
                code="repository_truth_artifact_conflict",
                problems=[str(path)],
            )
        return path
    return atomic_write_json(path, value)


def _write_immutable_bytes(path: Path, data: bytes) -> Path:
    if path.exists():
        if path.read_bytes() != data:
            raise ActorError(
                f"immutable repository-truth artifact conflict: {path}",
                code="repository_truth_artifact_conflict",
                problems=[str(path)],
            )
        return path
    return atomic_write_bytes(path, data)


def _reference(path: Path, value: Mapping[str, Any], *, state_root: str | Path | None) -> dict[str, Any]:
    root = actor_state_root(state_root)
    return {
        "schema_version": ARTIFACT_REFERENCE_SCHEMA_VERSION,
        "artifact_id": value.get("artifact_id"),
        "artifact_schema_version": value.get("schema_version"),
        "path": str(path),
        "state_root_locator": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "canonical_sha256": canonical_sha256(value),
    }


def load_repository_truth_artifact(
    reference: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    expected_schema_version: str,
) -> dict[str, Any]:
    require_explicit_schema_version(
        reference,
        expected=ARTIFACT_REFERENCE_SCHEMA_VERSION,
        label="repository-truth artifact reference",
    )
    locator = str(reference.get("state_root_locator") or "").strip()
    candidates = [Path(str(reference.get("path") or ""))]
    if locator:
        candidates.append(actor_state_root(state_root) / locator)
    path = next((candidate for candidate in candidates if str(candidate) and candidate.is_file()), None)
    if path is None:
        raise ActorError(
            "repository-truth artifact is missing",
            code="repository_truth_artifact_missing",
            problems=[str(candidate) for candidate in candidates],
        )
    if file_sha256(path) != str(reference.get("sha256") or ""):
        raise ActorError(
            "repository-truth artifact file digest mismatch",
            code="repository_truth_artifact_digest_mismatch",
        )
    value = read_json(path, name="repository-truth artifact")
    require_explicit_schema_version(
        value,
        expected=expected_schema_version,
        label="repository-truth artifact",
    )
    embedded = str(value.get("artifact_sha256") or "")
    material = dict(value)
    material.pop("artifact_sha256", None)
    if not _SHA256_RE.fullmatch(embedded) or embedded != canonical_sha256(material):
        raise ActorError(
            "repository-truth artifact canonical digest mismatch",
            code="repository_truth_artifact_canonical_digest_mismatch",
        )
    if str(reference.get("artifact_id") or "") != str(value.get("artifact_id") or ""):
        raise ActorError(
            "repository-truth artifact identity mismatch",
            code="repository_truth_artifact_identity_mismatch",
        )
    return value


def _changed_paths(repo_root: str | Path, base_commit: str, delivery_commit: str) -> list[str]:
    result = _git(repo_root, "diff", "--name-only", "--no-renames", "-z", f"{base_commit}..{delivery_commit}")
    if result.returncode != 0:
        raise ActorError(
            "cannot calculate delivery changed paths",
            code="repository_truth_capture_failed",
            problems=[str(result.stderr or "").strip() or "git diff failed"],
        )
    return sorted(item for item in str(result.stdout or "").split("\x00") if item)


def capture_delivery_proof(
    *,
    state_root: str | Path | None,
    repo_root: str | Path,
    work_id: str,
    session_id: str,
    package_sha256: str,
    base_commit: str,
    delivery_commit: str,
    changed_files: Sequence[str] | None = None,
    delivery_kind: str = "MANAGED_SESSION_DELIVERY",
    adopted_from: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = Path(repo_root).expanduser().resolve()
    state_namespace = ensure_repository_state_namespace(state_root=state_root, repo_root=repo)
    base = _resolve_commit(repo, base_commit, label="delivery base commit")
    delivery = _resolve_commit(repo, delivery_commit, label="delivery commit")
    paths = sorted(set(str(path) for path in (changed_files or _changed_paths(repo, base, delivery)) if str(path)))
    tree = _git_text(repo, "show", "-s", "--format=%T", delivery, label="delivery tree").lower()
    parents = [item.lower() for item in _git_text(repo, "show", "-s", "--format=%P", delivery, label="delivery parents").split() if item]
    ancestors = [item.lower() for item in _git_text(repo, "rev-list", delivery, label="delivery ancestry").splitlines() if item]
    file_blobs = [
        {
            "path": path,
            "base_blob": _blob_id(repo, base, path),
            "delivery_blob": _blob_id(repo, delivery, path),
        }
        for path in paths
    ]
    patch_result = _git(repo, "diff", "--binary", "--no-renames", base, delivery, "--", *paths, binary=True)
    if patch_result.returncode != 0:
        raise ActorError(
            "cannot capture binary delivery patch",
            code="repository_truth_capture_failed",
            problems=[bytes(patch_result.stderr or b"").decode("utf-8", "replace") or "git diff --binary failed"],
        )
    patch_bytes = bytes(patch_result.stdout or b"")
    patch_sha = __import__("hashlib").sha256(patch_bytes).hexdigest()
    patch_path = _root(state_root) / "delivery-patches" / f"{patch_sha}.patch"
    _write_immutable_bytes(patch_path, patch_bytes)
    body: dict[str, Any] = {
        "schema_version": DELIVERY_PROOF_SCHEMA_VERSION,
        "artifact_id": None,
        "work_id": work_id,
        "session_id": session_id,
        "package_sha256": package_sha256,
        "delivery_kind": delivery_kind,
        "state_namespace_id": state_namespace.get("namespace_id"),
        "repo_root_at_capture": str(repo),
        "base_commit": base,
        "delivery_commit": delivery,
        "delivery_tree": tree,
        "delivery_parents": parents,
        "ancestor_commits": ancestors,
        "changed_files": paths,
        "file_blobs": file_blobs,
        "binary_patch": {
            "sha256": patch_sha,
            "path": str(patch_path),
            "state_root_locator": patch_path.relative_to(actor_state_root(state_root)).as_posix(),
            "size_bytes": len(patch_bytes),
        },
        "adopted_from": dict(adopted_from or {}) or None,
        "captured_at_utc": utc_now(generated_at),
    }
    body["artifact_id"] = stable_id("DELPROOF", {key: value for key, value in body.items() if key != "artifact_id"})
    material = dict(body)
    body["artifact_sha256"] = canonical_sha256(material)
    path = _root(state_root) / "delivery-proofs" / f"{body['artifact_id']}.json"
    _write_immutable_json(path, body)
    return body, _reference(path, body, state_root=state_root)


def capture_adoption_receipt(
    *,
    state_root: str | Path | None,
    repo_root: str | Path,
    work_id: str,
    session_id: str,
    adopted_commit: str,
    reason: str,
    external_producer: str | None,
    generated_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not str(reason or "").strip():
        raise ActorError("external delivery adoption requires a reason", code="repository_adoption_reason_missing")
    commit = _resolve_commit(repo_root, adopted_commit, label="adopted commit")
    body: dict[str, Any] = {
        "schema_version": ADOPTION_RECEIPT_SCHEMA_VERSION,
        "artifact_id": None,
        "work_id": work_id,
        "adopting_session_id": session_id,
        "adopted_commit": commit,
        "external_producer": str(external_producer or "").strip() or None,
        "reason": str(reason).strip(),
        "performed_by_adopting_session": False,
        "authority_effect": "EXPLICIT_RECONCILIATION_ONLY",
        "recorded_at_utc": utc_now(generated_at),
    }
    body["artifact_id"] = stable_id("ADOPT", {key: value for key, value in body.items() if key != "artifact_id"})
    material = dict(body)
    body["artifact_sha256"] = canonical_sha256(material)
    path = _root(state_root) / "adoptions" / f"{body['artifact_id']}.json"
    _write_immutable_json(path, body)
    return body, _reference(path, body, state_root=state_root)


def capture_integration_receipt(
    *,
    state_root: str | Path | None,
    repo_root: str | Path,
    work_id: str,
    review_session_id: str,
    delivery_proof_ref: Mapping[str, Any],
    integration_commit: str,
    note: str | None = None,
    generated_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    proof = load_repository_truth_artifact(
        delivery_proof_ref,
        state_root=state_root,
        expected_schema_version=DELIVERY_PROOF_SCHEMA_VERSION,
    )
    commit = _resolve_commit(repo_root, integration_commit, label="integration commit")
    mismatches: list[str] = []
    for row in proof.get("file_blobs") or []:
        if not isinstance(row, Mapping):
            continue
        path = str(row.get("path") or "")
        expected = row.get("delivery_blob")
        observed = _blob_id(repo_root, commit, path)
        if observed != expected:
            mismatches.append(f"{path}: reviewed blob {expected}, integration blob {observed}")
    if mismatches:
        raise ActorError(
            "integration commit is not content-equivalent to the accepted delivery on reviewed paths",
            code="repository_integration_content_mismatch",
            problems=mismatches,
        )
    reviewed_commit = str(proof.get("delivery_commit") or "")
    relation = "EXACT_COMMIT"
    if commit != reviewed_commit:
        ancestor = _git(repo_root, "merge-base", "--is-ancestor", reviewed_commit, commit)
        relation = "DESCENDANT_INTEGRATION" if ancestor.returncode == 0 else "CONTENT_EQUIVALENT_REWRITE"
    body: dict[str, Any] = {
        "schema_version": INTEGRATION_RECEIPT_SCHEMA_VERSION,
        "artifact_id": None,
        "work_id": work_id,
        "review_session_id": review_session_id,
        "reviewed_delivery_proof": dict(delivery_proof_ref),
        "reviewed_commit": reviewed_commit,
        "integration_commit": commit,
        "relation": relation,
        "verified_paths": [str(dict(row).get("path") or "") for row in proof.get("file_blobs") or []],
        "note": str(note or "").strip() or None,
        "recorded_at_utc": utc_now(generated_at),
    }
    body["artifact_id"] = stable_id("INTEGRATE", {key: value for key, value in body.items() if key != "artifact_id"})
    material = dict(body)
    body["artifact_sha256"] = canonical_sha256(material)
    path = _root(state_root) / "integrations" / f"{body['artifact_id']}.json"
    _write_immutable_json(path, body)
    return body, _reference(path, body, state_root=state_root)


def capture_storage_merge_closeout_receipt(
    *,
    state_root: str | Path | None,
    work_id: str,
    source_branch: str | None,
    source_worktree: str | None,
    delivery_sha: str,
    target_branch: str | None,
    integration_sha: str,
    integration_relation: str,
    acceptance_state: Mapping[str, Any],
    certificate_report_reproduction: Mapping[str, Any],
    evidence_roots: Sequence[Mapping[str, Any]],
    bytes_before: int,
    bytes_reclaimed: int,
    bytes_retained: int,
    unresolved_unknown_policy_bytes: int,
    status: str,
    retained_reasons: Sequence[str] = (),
    generated_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist one immutable receipt for managed merge storage closeout."""

    counters = {
        "bytes_before": bytes_before,
        "bytes_reclaimed": bytes_reclaimed,
        "bytes_retained": bytes_retained,
        "unresolved_unknown_policy_bytes": unresolved_unknown_policy_bytes,
    }
    if any(type(value) is not int or value < 0 for value in counters.values()):
        raise ActorError(
            "storage merge closeout byte counters are invalid",
            code="storage_merge_closeout_accounting_invalid",
        )
    body: dict[str, Any] = {
        "schema_version": STORAGE_MERGE_CLOSEOUT_RECEIPT_SCHEMA_VERSION,
        "artifact_id": None,
        "work_id": str(work_id),
        "source_branch": str(source_branch or "").strip() or None,
        "source_worktree": str(source_worktree or "").strip() or None,
        "delivery_sha": str(delivery_sha).strip().lower(),
        "target_branch": str(target_branch or "").strip() or None,
        "integration_sha": str(integration_sha).strip().lower(),
        "integration_relation": str(integration_relation or "").strip(),
        "acceptance_state": dict(acceptance_state),
        "certificate_report_reproduction": dict(certificate_report_reproduction),
        "evidence_roots": [dict(row) for row in evidence_roots],
        **counters,
        "status": str(status or "").strip().upper(),
        "retained_reasons": [str(reason) for reason in retained_reasons],
        "ordinary_transient_policy": "RELEASE_AFTER_DURABLE_CLOSEOUT",
        "archive_policy": "DIRECT_HOT_RELEASE_ONLY_NO_HDD",
        "recorded_at_utc": utc_now(generated_at),
    }
    if not body["delivery_sha"] or not body["integration_sha"]:
        raise ActorError(
            "storage merge closeout requires delivery and integration SHA",
            code="storage_merge_closeout_identity_missing",
        )
    body["artifact_id"] = stable_id(
        "STORAGE-CLOSEOUT",
        {key: value for key, value in body.items() if key != "artifact_id"},
    )
    material = dict(body)
    body["artifact_sha256"] = canonical_sha256(material)
    path = _root(state_root) / "storage-closeouts" / f"{body['artifact_id']}.json"
    _write_immutable_json(path, body)
    return body, _reference(path, body, state_root=state_root)


def capture_repository_relocation_receipt(
    *,
    state_root: str | Path | None,
    old_repo_root: str | Path,
    new_repo_root: str | Path,
    material_sha256: str,
    authority: str,
    reason: str,
    generated_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authorize a content-equivalent repository/worktree relocation explicitly."""

    old_root = Path(old_repo_root).expanduser().resolve()
    new_root = Path(new_repo_root).expanduser().resolve()
    if not str(authority or "").strip() or not str(reason or "").strip():
        raise ActorError("repository relocation requires authority and reason", code="repository_relocation_authority_missing")
    old_head = _resolve_commit(old_root, "HEAD", label="old repository HEAD")
    new_head = _resolve_commit(new_root, "HEAD", label="new repository HEAD")
    old_tree = _git_text(old_root, "show", "-s", "--format=%T", old_head, label="old repository tree").lower()
    new_tree = _git_text(new_root, "show", "-s", "--format=%T", new_head, label="new repository tree").lower()
    if old_tree != new_tree:
        raise ActorError(
            "repository relocation is not content-equivalent",
            code="repository_relocation_content_mismatch",
            problems=[f"old tree: {old_tree}", f"new tree: {new_tree}"],
        )
    namespace = ensure_repository_state_namespace(state_root=state_root, repo_root=new_root)
    body: dict[str, Any] = {
        "schema_version": RELOCATION_RECEIPT_SCHEMA_VERSION,
        "artifact_id": None,
        "state_namespace_id": namespace.get("namespace_id"),
        "old_repo_root": str(old_root),
        "new_repo_root": str(new_root),
        "old_head": old_head,
        "new_head": new_head,
        "content_tree": new_tree,
        "material_sha256": str(material_sha256 or "").strip().lower(),
        "authority": str(authority).strip(),
        "reason": str(reason).strip(),
        "recorded_at_utc": utc_now(generated_at),
    }
    body["artifact_id"] = stable_id("RELOC", {key: value for key, value in body.items() if key != "artifact_id"})
    material = dict(body)
    body["artifact_sha256"] = canonical_sha256(material)
    path = _root(state_root) / "relocations" / f"{body['artifact_id']}.json"
    _write_immutable_json(path, body)
    return body, _reference(path, body, state_root=state_root)


def persisted_commit_relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    """Resolve delivery relation from retained proof material without live Git."""

    left_commit = str(left.get("delivery_commit") or "")
    right_commit = str(right.get("delivery_commit") or "")
    if not left_commit or not right_commit:
        return "unknown"
    if left_commit == right_commit:
        return "same"
    if left_commit in set(str(value) for value in right.get("ancestor_commits") or []):
        return "left_ancestor"
    if right_commit in set(str(value) for value in left.get("ancestor_commits") or []):
        return "right_ancestor"
    return "divergent"


__all__ = [
    "ADOPTION_RECEIPT_SCHEMA_VERSION",
    "ARTIFACT_REFERENCE_SCHEMA_VERSION",
    "DELIVERY_PROOF_SCHEMA_VERSION",
    "INTEGRATION_RECEIPT_SCHEMA_VERSION",
    "STORAGE_MERGE_CLOSEOUT_RECEIPT_SCHEMA_VERSION",
    "STATE_NAMESPACE_SCHEMA_VERSION",
    "RELOCATION_RECEIPT_SCHEMA_VERSION",
    "capture_adoption_receipt",
    "capture_delivery_proof",
    "capture_integration_receipt",
    "capture_storage_merge_closeout_receipt",
    "capture_repository_relocation_receipt",
    "ensure_repository_state_namespace",
    "load_repository_truth_artifact",
    "persisted_commit_relation",
]
