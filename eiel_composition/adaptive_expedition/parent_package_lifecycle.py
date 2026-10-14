"""Physical lifecycle accounting for exact Retained Parent Bank packages.

The public Parent Bank records scientific retention choices and exact
``StateRefV1`` identities.  This internal adapter accounts for the complete
run-local package that backs each generated exact state.  A package is staged
once at a stable locator as R1 recovery material; moving it after publication
would change the public StateRef identity.  The next deterministic Decision
therefore promotes that same sealed locator to R4, or leaves it as releasable R1,
by a receipt folded into the existing Expedition ledger.  No manager,
database, cleanup actuator, or authority is introduced here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Final

from emergence_contracts import canonical_json_bytes
from emergence_contracts.adaptive_science_expedition_v1 import (
    ExpeditionManifestV1,
    ObservationFrameV1,
    RetainedParentBankV1,
    StateClassV1,
)
from emergence_contracts.canonical import canonical_sha256

from .ledger import read_ledger_store_v1
from .policy_protocol import canonical_plain_value_v1


PARENT_CANDIDATE_RECEIPT_SCHEMA_V1: Final[str] = (
    "eiel_adaptive_parent_candidate_package_receipt_internal_v1"
)
PARENT_PACKAGE_DECISION_SCHEMA_V1: Final[str] = (
    "eiel_adaptive_parent_package_decision_internal_v1"
)
PARENT_PACKAGE_TERMINAL_SCHEMA_V1: Final[str] = (
    "eiel_adaptive_parent_package_terminal_internal_v1"
)
PARENT_PACKAGE_ACTIVE_RELEASE_SCHEMA_V1: Final[str] = (
    "eiel_adaptive_parent_package_active_release_v1"
)
PARENT_PACKAGE_ACTIVE_RELEASE_ROOT_V1: Final[str] = (
    "control/parent-package-releases"
)
PARENT_PACKAGE_LEDGER_FIELD_V1: Final[str] = "retained_parent_package_lifecycle"
PARENT_CANDIDATE_SETTLEMENT_FIELD_V1: Final[str] = (
    "retained_parent_candidate_package"
)
PARENT_CANDIDATE_RELATIVE_ROOT_V1: Final[str] = (
    "scratch/retained-parent-candidates"
)
LEGACY_PARENT_PACKAGE_RELATIVE_ROOT_V1: Final[str] = (
    "durable/retained-parents"
)
MAXIMUM_PARENT_PACKAGE_FILES_V1: Final[int] = 128
_OBSERVATION_FRAME_NAME_V1: Final[str] = "observation-frame.json"


class ParentPackageLifecycleError(ValueError):
    """An exact parent package cannot be classified without weakening safety."""


def expedition_root_for_ledger_v1(ledger_root: str | Path) -> Path:
    """Resolve the compact outer root without requiring another public field."""

    ledger = Path(ledger_root).expanduser().absolute()
    if ledger.name == "ledger" and ledger.parent.name == "durable":
        return ledger.parent.parent
    return ledger.parent


def parent_candidate_root_for_ledger_v1(ledger_root: str | Path) -> Path:
    return expedition_root_for_ledger_v1(ledger_root) / Path(
        PARENT_CANDIDATE_RELATIVE_ROOT_V1
    )


def _plain_mapping(value: object, *, field: str) -> dict[str, Any]:
    plain = canonical_plain_value_v1(value)
    if type(plain) is not dict:
        raise ParentPackageLifecycleError(f"{field} must be a canonical object")
    return plain


def _plain_sequence(value: object, *, field: str) -> list[Any]:
    plain = canonical_plain_value_v1(value)
    if type(plain) is not list:
        raise ParentPackageLifecycleError(f"{field} must be a canonical array")
    return plain


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ParentPackageLifecycleError(
            f"parent package file cannot be hashed: {path}"
        ) from error
    return digest.hexdigest()


def _bounded_relative_path(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ParentPackageLifecycleError(f"{field} must be a non-empty path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ParentPackageLifecycleError(f"{field} must be a bounded relative path")
    return path.as_posix()


def _within(path: Path, root: Path, *, field: str) -> Path:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ParentPackageLifecycleError(f"{field} must not be a symbolic link")
    try:
        resolved = requested.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ParentPackageLifecycleError(f"{field} escapes its sealed root") from error
    return resolved


def _package_inventory_v1(package_root: Path) -> dict[str, Any]:
    if package_root.is_symlink() or not package_root.is_dir():
        raise ParentPackageLifecycleError("parent package root is not a regular directory")
    rows: list[dict[str, object]] = []
    for path in sorted(package_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ParentPackageLifecycleError(
                "parent package inventory refuses symbolic links"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ParentPackageLifecycleError(
                "parent package contains a non-regular object"
            )
        rows.append(
            {
                "path": path.relative_to(package_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
        if len(rows) > MAXIMUM_PARENT_PACKAGE_FILES_V1:
            raise ParentPackageLifecycleError(
                "parent package exceeds the bounded file-count limit"
            )
    if not rows:
        raise ParentPackageLifecycleError("parent package is empty")
    return {
        "file_count": len(rows),
        "full_package_bytes": sum(int(row["size_bytes"]) for row in rows),
        "package_inventory_sha256": canonical_sha256(rows).hex,
    }


def _read_exact_frame_v1(path: Path) -> ObservationFrameV1:
    if path.is_symlink() or not path.is_file():
        raise ParentPackageLifecycleError(
            "parent package lacks its exact ObservationFrame"
        )
    try:
        payload = path.read_bytes()
        frame = ObservationFrameV1.from_dict(json.loads(payload.decode("ascii")))
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ParentPackageLifecycleError(
            "parent package ObservationFrame is invalid"
        ) from error
    if payload != canonical_json_bytes(frame.to_dict()):
        raise ParentPackageLifecycleError(
            "parent package ObservationFrame is not canonical JSON"
        )
    if frame.source_state_ref.state_class is not StateClassV1.BRANCHABLE_EXACT:
        raise ParentPackageLifecycleError(
            "parent package ObservationFrame is not branchable exact"
        )
    return frame


def _allowed_package_parent_v1(package_root: Path, run_root: Path) -> bool:
    relative = package_root.relative_to(run_root).as_posix()
    return package_root.parent in {
        run_root / PARENT_CANDIDATE_RELATIVE_ROOT_V1,
        run_root / LEGACY_PARENT_PACKAGE_RELATIVE_ROOT_V1,
    } and relative.startswith(
        (f"{PARENT_CANDIDATE_RELATIVE_ROOT_V1}/", f"{LEGACY_PARENT_PACKAGE_RELATIVE_ROOT_V1}/")
    )


def build_staged_parent_candidate_receipt_v1(
    *,
    manifest: ExpeditionManifestV1,
    observation_frame: ObservationFrameV1,
    run_root: str | Path,
    raw_child_runtime_root: str | Path | None,
) -> dict[str, Any]:
    """Seal one generated exact package as run-local R1 recovery material."""

    root = Path(run_root).expanduser().absolute()
    state = observation_frame.source_state_ref
    if (
        observation_frame.expedition_id != manifest.expedition_id
        or state.state_class is not StateClassV1.BRANCHABLE_EXACT
    ):
        raise ParentPackageLifecycleError(
            "candidate frame differs from the frozen Expedition or is not exact"
        )
    artifact = _within(
        Path(state.source_artifact_ref.locator),
        root,
        field="candidate StateRef source artifact",
    )
    package_root = artifact.parent.parent
    if not _allowed_package_parent_v1(package_root, root):
        raise ParentPackageLifecycleError(
            "candidate package is outside the bounded Expedition staging root"
        )
    retained_frame = _read_exact_frame_v1(
        package_root / _OBSERVATION_FRAME_NAME_V1
    )
    if retained_frame != observation_frame:
        raise ParentPackageLifecycleError(
            "candidate package ObservationFrame differs from settlement"
        )
    selected_world_bytes = sum(
        path.stat().st_size
        for path in artifact.parent.iterdir()
        if path.is_file() and not path.is_symlink()
    )
    if selected_world_bytes != state.source_artifact_ref.size_bytes:
        raise ParentPackageLifecycleError(
            "StateRef byte accounting differs from the selected-world package"
        )
    inventory = _package_inventory_v1(package_root)
    relative = package_root.relative_to(root).as_posix()
    raw_root = (
        None
        if raw_child_runtime_root is None
        else str(Path(raw_child_runtime_root).expanduser().absolute())
    )
    core = {
        "schema_version": PARENT_CANDIDATE_RECEIPT_SCHEMA_V1,
        "expedition_id": manifest.expedition_id,
        "manifest_sha256": manifest.manifest_sha256().hex,
        "state_ref_id": f"SR-{state.state_ref_sha256().hex}",
        "observation_frame_sha256": observation_frame.observation_frame_sha256().hex,
        "package_root": relative,
        "package_file_count": inventory["file_count"],
        "full_package_bytes": inventory["full_package_bytes"],
        "package_inventory_sha256": inventory["package_inventory_sha256"],
        "state_artifact_bytes": state.source_artifact_ref.size_bytes,
        "raw_child_runtime_root": raw_root,
        "staging_retention_class": "R1_ACTIVE_RECOVERY",
        "promotion_authority": "NEXT_DETERMINISTIC_PARENT_BANK_DECISION",
        "deletion_authority": False,
        "raw_child_root_mutation_authority": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core).hex}


def _validate_parent_candidate_receipt_identity_v1(
    receipt: Mapping[str, object],
    *,
    manifest: ExpeditionManifestV1,
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "expedition_id",
        "manifest_sha256",
        "state_ref_id",
        "observation_frame_sha256",
        "package_root",
        "package_file_count",
        "full_package_bytes",
        "package_inventory_sha256",
        "state_artifact_bytes",
        "raw_child_runtime_root",
        "staging_retention_class",
        "promotion_authority",
        "deletion_authority",
        "raw_child_root_mutation_authority",
        "receipt_sha256",
    }
    core = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if (
        set(receipt) != expected
        or receipt.get("schema_version") != PARENT_CANDIDATE_RECEIPT_SCHEMA_V1
        or receipt.get("expedition_id") != manifest.expedition_id
        or receipt.get("manifest_sha256") != manifest.manifest_sha256().hex
        or receipt.get("staging_retention_class") != "R1_ACTIVE_RECOVERY"
        or receipt.get("promotion_authority")
        != "NEXT_DETERMINISTIC_PARENT_BANK_DECISION"
        or receipt.get("deletion_authority") is not False
        or receipt.get("raw_child_root_mutation_authority") is not False
        or receipt.get("receipt_sha256") != canonical_sha256(core).hex
    ):
        raise ParentPackageLifecycleError("parent candidate receipt identity changed")
    return dict(receipt)


def validate_staged_parent_candidate_receipt_v1(
    value: Mapping[str, object],
    *,
    manifest: ExpeditionManifestV1,
    run_root: str | Path,
) -> dict[str, Any]:
    receipt = _plain_mapping(value, field="parent candidate receipt")
    _validate_parent_candidate_receipt_identity_v1(receipt, manifest=manifest)
    relative = _bounded_relative_path(
        receipt.get("package_root"), field="candidate package_root"
    )
    root = Path(run_root).expanduser().absolute()
    package_root = _within(
        root / relative,
        root,
        field="candidate package root",
    )
    if not package_root.is_dir() or not _allowed_package_parent_v1(package_root, root):
        raise ParentPackageLifecycleError("candidate package root is not bounded")
    frame = _read_exact_frame_v1(package_root / _OBSERVATION_FRAME_NAME_V1)
    state = frame.source_state_ref
    artifact = _within(
        Path(state.source_artifact_ref.locator),
        package_root,
        field="candidate StateRef source artifact",
    )
    selected_world_bytes = sum(
        path.stat().st_size
        for path in artifact.parent.iterdir()
        if path.is_file() and not path.is_symlink()
    )
    inventory = _package_inventory_v1(package_root)
    if (
        receipt.get("state_ref_id") != f"SR-{state.state_ref_sha256().hex}"
        or receipt.get("observation_frame_sha256")
        != frame.observation_frame_sha256().hex
        or receipt.get("state_artifact_bytes")
        != state.source_artifact_ref.size_bytes
        or selected_world_bytes != state.source_artifact_ref.size_bytes
        or receipt.get("package_file_count") != inventory["file_count"]
        or receipt.get("full_package_bytes") != inventory["full_package_bytes"]
        or receipt.get("package_inventory_sha256")
        != inventory["package_inventory_sha256"]
    ):
        raise ParentPackageLifecycleError(
            "parent candidate package differs from its sealed receipt"
        )
    return receipt


def _active_parent_release_proofs_v1(
    *,
    ledger: Mapping[str, object],
    manifest: ExpeditionManifestV1,
    run_root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read exact active-release authority, including interrupted boundaries."""

    directory = run_root / PARENT_PACKAGE_ACTIVE_RELEASE_ROOT_V1
    if not directory.exists():
        return {}
    if directory.is_symlink() or not directory.is_dir():
        raise ParentPackageLifecycleError(
            "active parent-release control root is not a regular directory"
        )
    decision_bindings: set[tuple[str, str]] = set()
    for raw_event in _plain_sequence(ledger.get("events"), field="ledger events"):
        event = _plain_mapping(raw_event, field="ledger event")
        if event.get("event_type") != "DECISION_PREPARED":
            continue
        payload = _plain_mapping(event.get("payload"), field="Decision payload")
        raw_receipt = payload.get(PARENT_PACKAGE_LEDGER_FIELD_V1)
        if isinstance(raw_receipt, Mapping):
            receipt = _plain_mapping(raw_receipt, field="parent package Decision receipt")
            decision_bindings.add(
                (str(event.get("event_sha256")), canonical_sha256(receipt).hex)
            )
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for authority_path in sorted(directory.glob("*.authority.json")):
        authority = _plain_mapping(
            json.loads(authority_path.read_bytes()),
            field="active parent-release authority",
        )
        expected_fields = {
            "schema_version",
            "expedition_id",
            "manifest_sha256",
            "lifecycle_contract_sha256",
            "runtime_root",
            "boundary_frontier_ordinal",
            "boundary_decision_event_sha256",
            "boundary_decision_receipt_sha256",
            "selected_parent_state_ref_ids",
            "artifact_rows",
        }
        if (
            set(authority) != expected_fields
            or authority.get("schema_version")
            != PARENT_PACKAGE_ACTIVE_RELEASE_SCHEMA_V1
            or authority.get("expedition_id") != manifest.expedition_id
            or authority.get("manifest_sha256") != manifest.manifest_sha256().hex
            or authority.get("runtime_root") != str(run_root)
            or (str(authority.get("boundary_decision_event_sha256")), str(authority.get("boundary_decision_receipt_sha256")))
            not in decision_bindings
            or type(authority.get("boundary_frontier_ordinal")) is not int
            or authority.get("boundary_frontier_ordinal") < 0
        ):
            raise ParentPackageLifecycleError(
                "active parent-release authority identity changed"
            )
        selected = _plain_sequence(
            authority.get("selected_parent_state_ref_ids"),
            field="active parent-release selected parents",
        )
        if any(type(value) is not str for value in selected) or selected != sorted(set(selected)):
            raise ParentPackageLifecycleError(
                "active parent-release selected parents are not canonical"
            )
        raw_rows = _plain_sequence(
            authority.get("artifact_rows"), field="active parent-release artifact rows"
        )
        if not raw_rows:
            raise ParentPackageLifecycleError(
                "active parent-release authority has no artifact rows"
            )
        authority_rows: list[dict[str, Any]] = []
        for raw_row in raw_rows:
            row = _plain_mapping(raw_row, field="active parent-release artifact row")
            path = _bounded_relative_path(row.get("path"), field="active package path")
            if not path.startswith(f"{PARENT_CANDIDATE_RELATIVE_ROOT_V1}/"):
                raise ParentPackageLifecycleError(
                    "active parent-release artifact escapes candidate staging"
                )
            if (
                row.get("artifact_type") != "sealed_tree"
                or row.get("retention_class") != "R1"
                or row.get("release_eligible") is not None
                or type(row.get("bytes")) is not int
                or type(row.get("full_package_bytes")) is not int
                or row.get("bytes") != row.get("full_package_bytes")
                or not isinstance(row.get("tree_sha256"), str)
                or len(str(row.get("tree_sha256"))) != 64
                or not isinstance(row.get("state_ref_id"), str)
                or not isinstance(row.get("package_inventory_sha256"), str)
                or len(str(row.get("package_inventory_sha256"))) != 64
                or not isinstance(row.get("source_decision_event_sha256"), str)
                or not isinstance(row.get("source_decision_receipt_sha256"), str)
                or (
                    str(row.get("source_decision_event_sha256")),
                    str(row.get("source_decision_receipt_sha256")),
                )
                not in decision_bindings
            ):
                raise ParentPackageLifecycleError(
                    "active parent-release artifact authority is malformed"
                )
            authority_rows.append(row)
        if authority_rows != sorted(
            authority_rows,
            key=lambda row: (str(row["state_ref_id"]), str(row["path"])),
        ):
            raise ParentPackageLifecycleError(
                "active parent-release artifact rows are not sorted"
            )
        receipt_path = authority_path.with_name(
            authority_path.name.removesuffix(".authority.json") + ".receipt.json"
        )
        proof_status = "PENDING"
        receipt: dict[str, Any] | None = None
        if receipt_path.exists() or receipt_path.is_symlink():
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise ParentPackageLifecycleError(
                    "active parent-release receipt is not a regular file"
                )
            receipt = _plain_mapping(
                json.loads(receipt_path.read_bytes()),
                field="active parent-release receipt",
            )
            receipt_core = {
                key: item for key, item in receipt.items() if key != "receipt_sha256"
            }
            expected_paths = sorted(str(row["path"]) for row in authority_rows)
            if (
                set(receipt)
                != {
                    "schema_version",
                    "authority_sha256",
                    "authority_path",
                    "status",
                    "released_paths",
                    "released_bytes",
                    "release_journal",
                    "release_journal_sha256",
                    "release_journal_status",
                    "receipt_sha256",
                }
                or receipt.get("schema_version") != PARENT_PACKAGE_ACTIVE_RELEASE_SCHEMA_V1
                or receipt.get("authority_sha256") != canonical_sha256(authority).hex
                or receipt.get("authority_path")
                != authority_path.relative_to(run_root).as_posix()
                or receipt.get("status") != "RELEASED"
                or receipt.get("released_paths") != expected_paths
                or receipt.get("released_bytes")
                != sum(int(row["bytes"]) for row in authority_rows)
                or receipt.get("release_journal_status") != "COMPLETE"
                or receipt.get("receipt_sha256") != canonical_sha256(receipt_core).hex
            ):
                raise ParentPackageLifecycleError(
                    "active parent-release receipt identity changed"
                )
            journal_relative = _bounded_relative_path(
                receipt.get("release_journal"), field="active release journal"
            )
            journal_path = run_root / journal_relative
            if journal_path.is_symlink() or not journal_path.is_file():
                raise ParentPackageLifecycleError(
                    "active parent-release journal is not present"
                )
            journal = _plain_mapping(
                json.loads(journal_path.read_bytes()),
                field="active parent-release journal",
            )
            selected_journal_rows = {
                str(row.get("path")): row
                for row in journal.get("selected", [])
                if isinstance(row, Mapping)
            }
            if (
                receipt.get("release_journal_sha256")
                != canonical_sha256(journal).hex
                or journal.get("status") != "COMPLETE"
                or set(selected_journal_rows) != set(expected_paths)
                or any(
                    selected_journal_rows[path].get("bytes")
                    != next(
                        row["bytes"]
                        for row in authority_rows
                        if row["path"] == path
                    )
                    for path in expected_paths
                )
            ):
                raise ParentPackageLifecycleError(
                    "active parent-release journal does not prove the exact release"
                )
            proof_status = "RELEASED"
        for row in authority_rows:
            key = (str(row["state_ref_id"]), str(row["path"]))
            proof = {
                "status": proof_status,
                "authority": authority,
                "authority_row": row,
                "receipt": receipt,
            }
            if key in result and result[key] != proof:
                raise ParentPackageLifecycleError(
                    "active parent-release proofs conflict"
                )
            result[key] = proof
    return result


def _candidate_receipts_from_ledger_v1(
    ledger: Mapping[str, object],
    *,
    manifest: ExpeditionManifestV1,
    run_root: Path,
) -> dict[str, dict[str, Any]]:
    by_state: dict[str, dict[str, Any]] = {}
    by_package: dict[str, str] = {}
    active_proofs = _active_parent_release_proofs_v1(
        ledger=ledger, manifest=manifest, run_root=run_root
    )
    for raw_event in _plain_sequence(ledger.get("events"), field="ledger events"):
        event = _plain_mapping(raw_event, field="ledger event")
        if event.get("event_type") != "ACTION_SETTLED":
            continue
        payload = _plain_mapping(event.get("payload"), field="settlement payload")
        raw = payload.get(PARENT_CANDIDATE_SETTLEMENT_FIELD_V1)
        if raw is None:
            continue
        receipt = _plain_mapping(raw, field="settled parent candidate")
        _validate_parent_candidate_receipt_identity_v1(receipt, manifest=manifest)
        relative_package = _bounded_relative_path(
            receipt.get("package_root"), field="candidate package_root"
        )
        package_path = run_root / relative_package
        if package_path.exists() or package_path.is_symlink():
            receipt = validate_staged_parent_candidate_receipt_v1(
                receipt,
                manifest=manifest,
                run_root=run_root,
            )
        else:
            proof = active_proofs.get((str(receipt["state_ref_id"]), relative_package))
            if proof is None:
                raise ParentPackageLifecycleError(
                    "candidate package disappeared without active release authority"
                )
            authority_row = proof["authority_row"]
            if (
                authority_row.get("full_package_bytes") != receipt["full_package_bytes"]
                or authority_row.get("package_inventory_sha256")
                != receipt["package_inventory_sha256"]
            ):
                raise ParentPackageLifecycleError(
                    "active release authority differs from the candidate receipt"
                )
        state_id = str(receipt["state_ref_id"])
        package_root = str(receipt["package_root"])
        if state_id in by_state and by_state[state_id] != receipt:
            raise ParentPackageLifecycleError(
                "one candidate StateRef has conflicting physical packages"
            )
        prior_state = by_package.get(package_root)
        if prior_state is not None and prior_state != state_id:
            raise ParentPackageLifecycleError(
                "one physical package claims multiple StateRef identities"
            )
        by_state[state_id] = receipt
        by_package[package_root] = state_id
    return by_state


def _policy_reason_map_v1(outcomes: Mapping[str, object]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for raw in _plain_sequence(
        outcomes.get("policy_not_retained"), field="policy not-retained rows"
    ):
        row = _plain_mapping(raw, field="policy not-retained row")
        state_id = row.get("state_ref_id")
        reason = row.get("reason_code")
        if type(state_id) is not str or not state_id or type(reason) is not str or not reason:
            raise ParentPackageLifecycleError(
                "policy not-retained row lacks a StateRef and reason"
            )
        if state_id in reasons and reasons[state_id] != reason:
            raise ParentPackageLifecycleError(
                "policy supplied conflicting discard reasons for one StateRef"
            )
        reasons[state_id] = reason
    return reasons


def _bank_entries_v1(bank: RetainedParentBankV1) -> dict[str, Any]:
    return {
        f"SR-{entry.state_ref.state_ref_sha256().hex}": entry
        for entry in bank.entries
    }


def _full_bank_accounting_v1(
    *,
    manifest: ExpeditionManifestV1,
    bank: RetainedParentBankV1,
    candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    bank.validate_against_manifest(manifest)
    rows: list[dict[str, object]] = []
    for state_id, entry in sorted(_bank_entries_v1(bank).items()):
        candidate = candidates.get(state_id)
        if candidate is None:
            package_bytes = entry.retained_bytes
            accounting = "STATE_REF_EXACT_PACKAGE_BYTES"
            package_root = None
            package_sha256 = None
        else:
            package_bytes = int(candidate["full_package_bytes"])
            accounting = "RUN_LOCAL_FULL_PACKAGE_BYTES"
            package_root = candidate["package_root"]
            package_sha256 = candidate["package_inventory_sha256"]
        rows.append(
            {
                "state_ref_id": state_id,
                "package_accounting": accounting,
                "package_root": package_root,
                "package_inventory_sha256": package_sha256,
                "full_package_bytes": package_bytes,
            }
        )
    total = sum(int(row["full_package_bytes"]) for row in rows)
    if len(rows) > manifest.maximum_retained_parent_entries:
        raise ParentPackageLifecycleError(
            "physical Parent Bank exceeds the manifest entry cap"
        )
    if total > manifest.maximum_retained_parent_bytes:
        raise ParentPackageLifecycleError(
            "physical Parent Bank full packages exceed the manifest byte cap"
        )
    return {
        "entry_count": len(rows),
        "full_package_bytes": total,
        "maximum_entries": manifest.maximum_retained_parent_entries,
        "maximum_full_package_bytes": manifest.maximum_retained_parent_bytes,
        "entries": rows,
        "accounting_sha256": canonical_sha256(rows).hex,
    }


def _prior_package_outcomes_v1(
    ledger: Mapping[str, object],
    *,
    before_frontier_ordinal: int | None = None,
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for raw_event in _plain_sequence(ledger.get("events"), field="ledger events"):
        event = _plain_mapping(raw_event, field="ledger event")
        if event.get("event_type") != "DECISION_PREPARED":
            continue
        event_frontier = event.get("frontier_ordinal")
        if type(event_frontier) is not int or event_frontier < 0:
            raise ParentPackageLifecycleError(
                "parent package Decision event frontier is invalid"
            )
        if (
            before_frontier_ordinal is not None
            and event_frontier >= before_frontier_ordinal
        ):
            # Replaying the current committed Decision must rebuild its same
            # receipt from the preceding history, not treat itself as history.
            continue
        payload = _plain_mapping(event.get("payload"), field="Decision payload")
        raw_receipt = payload.get(PARENT_PACKAGE_LEDGER_FIELD_V1)
        if raw_receipt is None:
            continue
        receipt = _plain_mapping(raw_receipt, field="parent package Decision receipt")
        if receipt.get("schema_version") != PARENT_PACKAGE_DECISION_SCHEMA_V1:
            raise ParentPackageLifecycleError(
                "parent package Decision receipt schema changed"
            )
        for raw_row in _plain_sequence(
            receipt.get("package_outcomes"), field="parent package outcomes"
        ):
            row = _plain_mapping(raw_row, field="parent package outcome")
            latest[str(row.get("state_ref_id"))] = row
    return latest


def build_parent_package_decision_receipt_v1(
    *,
    manifest: ExpeditionManifestV1,
    ledger: Mapping[str, object],
    frontier_ordinal: int,
    observation_frames: Sequence[ObservationFrameV1],
    retained_parent_bank: RetainedParentBankV1,
    retention_outcomes: Mapping[str, object],
    run_root: str | Path,
) -> dict[str, Any]:
    """Resolve staged candidates at the next deterministic policy Decision."""

    root = Path(run_root).expanduser().absolute()
    outcomes = _plain_mapping(retention_outcomes, field="retention outcomes")
    candidates = _candidate_receipts_from_ledger_v1(
        ledger, manifest=manifest, run_root=root
    )
    prior = _prior_package_outcomes_v1(
        ledger,
        before_frontier_ordinal=frontier_ordinal,
    )
    current_state_ids = {
        f"SR-{frame.source_state_ref.state_ref_sha256().hex}"
        for frame in observation_frames
        if frame.source_state_ref.state_class is StateClassV1.BRANCHABLE_EXACT
    }
    current_candidates = {
        state_id: receipt
        for state_id, receipt in candidates.items()
        if state_id in current_state_ids and state_id not in prior
    }
    reason_by_state = _policy_reason_map_v1(outcomes)
    released = {
        str(value)
        for value in _plain_sequence(
            outcomes.get("released_state_ref_ids"), field="released StateRef IDs"
        )
    }
    bank_entries = _bank_entries_v1(retained_parent_bank)
    rows: list[dict[str, Any]] = []
    for state_id, candidate in sorted(current_candidates.items()):
        if state_id in bank_entries:
            if state_id in reason_by_state:
                raise ParentPackageLifecycleError(
                    "one candidate is both retained and explicitly discarded"
                )
            status = "RETAINED_R4"
            retention_class = "R4"
            reasons = list(bank_entries[state_id].reason_codes)
        else:
            reason = reason_by_state.get(state_id)
            if reason is None:
                raise ParentPackageLifecycleError(
                    "policy omitted the discard reason for an exact parent candidate"
                )
            status = "DISCARDED_RELEASE_R1"
            retention_class = "R1"
            reasons = [reason]
        rows.append(
            {
                "state_ref_id": state_id,
                "package_root": candidate["package_root"],
                "previous_status": "STAGED_R1_ACTIVE_RECOVERY",
                "status": status,
                "retention_class": retention_class,
                "reason_codes": reasons,
                "resolved_frontier_ordinal": frontier_ordinal,
                "full_package_bytes": candidate["full_package_bytes"],
                "package_inventory_sha256": candidate[
                    "package_inventory_sha256"
                ],
                "raw_child_runtime_root": candidate["raw_child_runtime_root"],
                "active_or_recovery_protection": (
                    "UNTIL_BOUND_ACTION_SETTLES_OR_EXPEDITION_TERMINAL"
                ),
                "deletion_authority": False,
                "raw_child_root_mutation_authority": False,
            }
        )
    for state_id in sorted(released):
        candidate = candidates.get(state_id)
        if candidate is None:
            continue
        reason = reason_by_state.get(state_id)
        if reason is None:
            raise ParentPackageLifecycleError(
                "policy released a generated exact parent without a discard reason"
            )
        rows.append(
            {
                "state_ref_id": state_id,
                "package_root": candidate["package_root"],
                "previous_status": "RETAINED_R4",
                "status": "DISCARDED_RELEASE_R1",
                "retention_class": "R1",
                "reason_codes": [reason],
                "resolved_frontier_ordinal": frontier_ordinal,
                "full_package_bytes": candidate["full_package_bytes"],
                "package_inventory_sha256": candidate[
                    "package_inventory_sha256"
                ],
                "raw_child_runtime_root": candidate["raw_child_runtime_root"],
                "active_or_recovery_protection": (
                    "UNTIL_BOUND_ACTION_SETTLES_OR_EXPEDITION_TERMINAL"
                ),
                "deletion_authority": False,
                "raw_child_root_mutation_authority": False,
            }
        )
    accounting = _full_bank_accounting_v1(
        manifest=manifest,
        bank=retained_parent_bank,
        candidates=candidates,
    )
    core = {
        "schema_version": PARENT_PACKAGE_DECISION_SCHEMA_V1,
        "expedition_id": manifest.expedition_id,
        "manifest_sha256": manifest.manifest_sha256().hex,
        "frontier_ordinal": frontier_ordinal,
        "package_outcomes": sorted(
            rows,
            key=lambda row: (
                str(row["state_ref_id"]),
                str(row["previous_status"]),
            ),
        ),
        "physical_bank_accounting": accounting,
        "classification_authority": "RECORDED_POLICY_PARENT_BANK_DECISION_ONLY",
        "stable_locator_no_move": True,
        "deletion_authority": False,
        "raw_child_root_mutation_authority": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core).hex}


def build_parent_package_terminal_receipt_v1(
    *,
    manifest: ExpeditionManifestV1,
    ledger: Mapping[str, object],
    frontier_ordinal: int,
    retained_parent_bank: RetainedParentBankV1,
    terminal_status: str,
    run_root: str | Path,
) -> dict[str, Any]:
    """Seal the final R4/R1 classification used by outer lifecycle inventory."""

    root = Path(run_root).expanduser().absolute()
    candidates = _candidate_receipts_from_ledger_v1(
        ledger, manifest=manifest, run_root=root
    )
    latest = _prior_package_outcomes_v1(ledger)
    bank_entries = _bank_entries_v1(retained_parent_bank)
    rows: list[dict[str, Any]] = []
    for state_id, candidate in sorted(candidates.items()):
        bank_entry = bank_entries.get(state_id)
        prior = latest.get(state_id)
        if bank_entry is not None:
            status = "RETAINED_R4"
            retention_class = "R4"
            reasons = list(bank_entry.reason_codes)
        elif prior is None:
            status = "DISCARDED_RELEASE_R1"
            retention_class = "R1"
            reasons = ["NO_SUBSEQUENT_POLICY_RETENTION_DECISION_BEFORE_TERMINAL"]
        elif prior.get("status") == "DISCARDED_RELEASE_R1":
            status = "DISCARDED_RELEASE_R1"
            retention_class = "R1"
            reasons = list(prior.get("reason_codes", ()))
        else:
            raise ParentPackageLifecycleError(
                "a previously retained package left the final Bank without a release reason"
            )
        rows.append(
            {
                "state_ref_id": state_id,
                "package_root": candidate["package_root"],
                "status": status,
                "retention_class": retention_class,
                "reason_codes": reasons,
                "full_package_bytes": candidate["full_package_bytes"],
                "package_inventory_sha256": candidate[
                    "package_inventory_sha256"
                ],
                "raw_child_runtime_root": candidate["raw_child_runtime_root"],
                "release_predicate": (
                    "Retained Parent Bank release decision and all child recovery obligations closed"
                    if retention_class == "R4"
                    else "explicit discard recorded, child settled, and raw child lifecycle remains independent"
                ),
                "deletion_authority": False,
                "raw_child_root_mutation_authority": False,
            }
        )
    accounting = _full_bank_accounting_v1(
        manifest=manifest,
        bank=retained_parent_bank,
        candidates=candidates,
    )
    retained_bytes = sum(
        int(row["full_package_bytes"])
        for row in rows
        if row["retention_class"] == "R4"
    )
    discarded_bytes = sum(
        int(row["full_package_bytes"])
        for row in rows
        if row["retention_class"] == "R1"
    )
    core = {
        "schema_version": PARENT_PACKAGE_TERMINAL_SCHEMA_V1,
        "expedition_id": manifest.expedition_id,
        "manifest_sha256": manifest.manifest_sha256().hex,
        "frontier_ordinal": frontier_ordinal,
        "terminal_status": terminal_status,
        "package_outcomes": rows,
        "physical_bank_accounting": accounting,
        "retained_r4_full_package_bytes": retained_bytes,
        "discarded_release_r1_full_package_bytes": discarded_bytes,
        "unresolved_package_count": 0,
        "classification_authority": "FINAL_RECORDED_PARENT_BANK_AND_POLICY_HISTORY",
        "stable_locator_no_move": True,
        "deletion_authority": False,
        "raw_child_root_mutation_authority": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core).hex}


def validate_parent_package_terminal_receipt_v1(
    value: Mapping[str, object],
    *,
    manifest: ExpeditionManifestV1,
) -> dict[str, Any]:
    receipt = _plain_mapping(value, field="terminal parent package receipt")
    expected = {
        "schema_version",
        "expedition_id",
        "manifest_sha256",
        "frontier_ordinal",
        "terminal_status",
        "package_outcomes",
        "physical_bank_accounting",
        "retained_r4_full_package_bytes",
        "discarded_release_r1_full_package_bytes",
        "unresolved_package_count",
        "classification_authority",
        "stable_locator_no_move",
        "deletion_authority",
        "raw_child_root_mutation_authority",
        "receipt_sha256",
    }
    core = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if (
        set(receipt) != expected
        or receipt.get("schema_version") != PARENT_PACKAGE_TERMINAL_SCHEMA_V1
        or receipt.get("expedition_id") != manifest.expedition_id
        or receipt.get("manifest_sha256") != manifest.manifest_sha256().hex
        or receipt.get("unresolved_package_count") != 0
        or receipt.get("classification_authority")
        != "FINAL_RECORDED_PARENT_BANK_AND_POLICY_HISTORY"
        or receipt.get("stable_locator_no_move") is not True
        or receipt.get("deletion_authority") is not False
        or receipt.get("raw_child_root_mutation_authority") is not False
        or receipt.get("receipt_sha256") != canonical_sha256(core).hex
    ):
        raise ParentPackageLifecycleError(
            "terminal parent package receipt identity changed"
        )
    return receipt


def terminal_parent_package_inventory_bindings_v1(
    *,
    manifest: ExpeditionManifestV1,
    run_root: str | Path,
) -> dict[str, dict[str, str]]:
    """Reopen final ledger classification for the canonical outer inventory."""

    root = Path(run_root).expanduser().absolute()
    ledger_root = root / "durable" / "ledger"
    if not ledger_root.exists():
        return {}
    ledger = read_ledger_store_v1(ledger_root)
    terminal = next(
        (
            event
            for event in reversed(ledger["events"])
            if event["event_type"] == "EXPEDITION_TERMINAL"
        ),
        None,
    )
    candidates = _candidate_receipts_from_ledger_v1(
        ledger, manifest=manifest, run_root=root
    )
    if terminal is None:
        if candidates:
            raise ParentPackageLifecycleError(
                "parent packages cannot enter terminal inventory before terminal ledger closeout"
            )
        return {}
    raw_receipt = terminal["payload"].get(PARENT_PACKAGE_LEDGER_FIELD_V1)
    if raw_receipt is None:
        if candidates:
            raise ParentPackageLifecycleError(
                "terminal ledger lacks physical Parent Bank classification"
            )
        return {}
    receipt = validate_parent_package_terminal_receipt_v1(
        _plain_mapping(raw_receipt, field="terminal parent package receipt"),
        manifest=manifest,
    )
    rows = {
        str(row["state_ref_id"]): _plain_mapping(
            row, field="terminal parent package outcome"
        )
        for row in _plain_sequence(
            receipt["package_outcomes"], field="terminal package outcomes"
        )
    }
    if set(rows) != set(candidates):
        raise ParentPackageLifecycleError(
            "terminal package classification does not cover every staged candidate"
        )
    active_proofs = _active_parent_release_proofs_v1(
        ledger=ledger, manifest=manifest, run_root=root
    )
    bindings: dict[str, dict[str, str]] = {}
    for state_id, candidate in sorted(candidates.items()):
        row = rows[state_id]
        if (
            row.get("package_root") != candidate["package_root"]
            or row.get("full_package_bytes") != candidate["full_package_bytes"]
            or row.get("package_inventory_sha256")
            != candidate["package_inventory_sha256"]
        ):
            raise ParentPackageLifecycleError(
                "terminal package outcome differs from the sealed candidate"
            )
        package_path = root / str(candidate["package_root"])
        if not package_path.exists() and not package_path.is_symlink():
            proof = active_proofs.get(
                (state_id, str(candidate["package_root"]))
            )
            if (
                proof is None
                or proof.get("status") != "RELEASED"
                or row.get("retention_class") != "R1"
            ):
                raise ParentPackageLifecycleError(
                    "terminal parent package is absent without a completed release proof"
                )
            # The exact sealed tree was already released by the existing
            # journaled actuator.  It must not be re-inventoried as a final
            # survivor, but its candidate receipt remains in the ledger.
            continue
        retention_class = row.get("retention_class")
        if retention_class == "R4" and row.get("status") == "RETAINED_R4":
            binding = {
                "artifact_family": "adaptive_expedition.retained_exact_parent",
                "retention_class": "R4",
                "owner_writer": "eiel-adaptive-expedition-coordinator-v1",
                "release_predicate": str(row["release_predicate"]),
            }
        elif retention_class == "R1" and row.get("status") == "DISCARDED_RELEASE_R1":
            binding = {
                "artifact_family": (
                    "adaptive_expedition.discarded_exact_parent_candidate"
                ),
                "retention_class": "R1",
                "owner_writer": "eiel-adaptive-expedition-coordinator-v1",
                "release_predicate": str(row["release_predicate"]),
            }
        else:
            raise ParentPackageLifecycleError(
                "terminal parent package classification is unsupported"
            )
        bindings[str(candidate["package_root"])] = binding
    return bindings


__all__ = (
    "LEGACY_PARENT_PACKAGE_RELATIVE_ROOT_V1",
    "PARENT_CANDIDATE_RELATIVE_ROOT_V1",
    "PARENT_CANDIDATE_SETTLEMENT_FIELD_V1",
    "PARENT_PACKAGE_LEDGER_FIELD_V1",
    "PARENT_PACKAGE_ACTIVE_RELEASE_ROOT_V1",
    "PARENT_PACKAGE_ACTIVE_RELEASE_SCHEMA_V1",
    "ParentPackageLifecycleError",
    "build_parent_package_decision_receipt_v1",
    "build_parent_package_terminal_receipt_v1",
    "build_staged_parent_candidate_receipt_v1",
    "expedition_root_for_ledger_v1",
    "parent_candidate_root_for_ledger_v1",
    "terminal_parent_package_inventory_bindings_v1",
    "validate_parent_package_terminal_receipt_v1",
    "validate_staged_parent_candidate_receipt_v1",
)
