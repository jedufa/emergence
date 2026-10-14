"""Canonical artifact-lifecycle binding for the Expedition coordinator root.

The Expedition Machine owns only compact coordination state.  Its generated
Science Request children retain their existing, independent lifecycle roots
and authority.  This module binds the compact outer root to the repository's
existing lifecycle contract, storage admission, inventory, disposition, and
background archive queue; it is not a new storage manager.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Final

from emergence_contracts import canonical_json_bytes
from emergence_contracts.adaptive_science_expedition_v1 import ExpeditionManifestV1
from emergence_contracts.canonical import canonical_sha256
from engine.campaigns.artifact_lifecycle import (
    ArtifactLifecycleError,
    RunStorageGuard,
    SCIENCE_COMPACT,
    build_artifact_disposition_manifest_v1,
    build_active_artifact_disposition_manifest_v1,
    build_run_passport_v1,
    build_sealed_artifact_inventory_row_v1,
    build_storage_disposition_receipt_v1,
    ensure_lifecycle_contract_v1,
    lifecycle_profile_contract_v1,
    register_pending_durable_archive_v1,
    run_artifact_transaction_fence_v1,
    validate_artifact_disposition_manifest_v1,
    validate_run_artifact_lifecycle_contract_v1,
    write_run_passport_v1,
)

from .parent_package_lifecycle import (
    LEGACY_PARENT_PACKAGE_RELATIVE_ROOT_V1,
    PARENT_CANDIDATE_RELATIVE_ROOT_V1,
    PARENT_PACKAGE_ACTIVE_RELEASE_ROOT_V1,
    PARENT_PACKAGE_ACTIVE_RELEASE_SCHEMA_V1,
    PARENT_PACKAGE_DECISION_SCHEMA_V1,
    PARENT_PACKAGE_LEDGER_FIELD_V1,
    ParentPackageLifecycleError,
    terminal_parent_package_inventory_bindings_v1,
)

EXPEDITION_LIFECYCLE_POLICY_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_adaptive_expedition_lifecycle_policy_v1"
)
EXPEDITION_LIFECYCLE_PLAN_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_adaptive_expedition_lifecycle_plan_v1"
)
EXPEDITION_LIFECYCLE_PROVIDER_ID_V1: Final[str] = (
    "eiel-adaptive-expedition-coordinator-v1"
)
_SUPPORTED_CHILD_PROFILES_V1: Final[frozenset[str]] = frozenset(
    {"CHECKPOINT_TRANSIT", "SCIENCE_COMPACT", "DEBUG_FORENSIC"}
)
_DISPOSITION_NAME: Final[str] = "artifact_disposition_manifest.json"
_STORAGE_RECEIPT_NAME: Final[str] = "storage_disposition_receipt.json"
_PASSPORT_NAME: Final[str] = "run_passport.json"
_ACTUATOR_CONTROL_PREFIXES_V1: Final[tuple[str, ...]] = (
    ".eiel-runtime-archive-release-v1/",
    ".eiel-runtime-hot-release-v1/",
)
_ACTIVE_PARENT_RELEASE_SCHEMA_V1: Final[str] = PARENT_PACKAGE_ACTIVE_RELEASE_SCHEMA_V1
_ACTIVE_PARENT_RELEASE_ROOT_V1: Final[str] = PARENT_PACKAGE_ACTIVE_RELEASE_ROOT_V1
DEFAULT_OUTER_LIFECYCLE_MAX_BYTES_V1: Final[int] = 512 * 1024 * 1024
MAX_OUTER_LIFECYCLE_OBJECTS_V1: Final[int] = 4096
_OUTER_MANIFEST_OBJECT_RESERVE_V1: Final[int] = 16
_SEALED_COORDINATION_ROOTS_V1: Final[tuple[str, ...]] = (
    "closeout",
    "control",
    "durable/founder-basis",
    "durable/ledger",
)


class ExpeditionLifecycleError(ValueError):
    """The outer coordinator root lacks canonical lifecycle authority."""


@dataclass(frozen=True, slots=True)
class ExpeditionLifecyclePolicyV1:
    child_contract_class: str
    outer_contract: Mapping[str, Any]
    lifecycle_authority: str
    retention_authority_created: bool


@dataclass(slots=True)
class ExpeditionLifecycleContextV1:
    run_root: Path
    manifest: ExpeditionManifestV1
    plan: Mapping[str, Any]
    contract: Mapping[str, Any]
    source_identity: Mapping[str, Any]
    guard: RunStorageGuard
    admitted: bool


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExpeditionLifecycleError(f"{field} must be an object")
    return dict(value)


def _read_mapping(path: Path, *, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExpeditionLifecycleError(f"{field} is not a regular file: {path}")
    try:
        return _mapping(json.loads(path.read_bytes()), field=field)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExpeditionLifecycleError(f"{field} is unreadable JSON: {path}") from error


def _publish_canonical_once(path: Path, value: object, *, field: str) -> Path:
    """Create immutable canonical bytes or authenticate an identical survivor."""

    if path.is_symlink():
        raise ExpeditionLifecycleError(f"{field} is a symbolic link: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ExpeditionLifecycleError(f"{field} parent is invalid: {path.parent}")
    expected = canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
                raise ExpeditionLifecycleError(
                    f"{field} differs from the immutable recovered run: {path}"
                )
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def resolve_expedition_lifecycle_policy_v1(
    manifest: ExpeditionManifestV1,
) -> ExpeditionLifecyclePolicyV1:
    envelope = manifest.artifact_lifecycle_policy
    if str(envelope.schema_version) != EXPEDITION_LIFECYCLE_POLICY_SCHEMA_VERSION_V1:
        raise ExpeditionLifecycleError("unsupported Expedition lifecycle policy schema")
    payload = _mapping(envelope.to_value(), field="artifact_lifecycle_policy")
    required = {
        "child_contract_class",
        "lifecycle_authority",
        "outer_contract",
        "retention_authority_created",
    }
    if set(payload) != required:
        raise ExpeditionLifecycleError(
            "Expedition lifecycle policy has unknown or missing fields"
        )
    child_profile = payload["child_contract_class"]
    if child_profile not in _SUPPORTED_CHILD_PROFILES_V1:
        raise ExpeditionLifecycleError("unsupported child lifecycle contract class")
    if payload["lifecycle_authority"] != "canonical_run_artifact_lifecycle":
        raise ExpeditionLifecycleError("canonical artifact lifecycle is required")
    if payload["retention_authority_created"] is not False:
        raise ExpeditionLifecycleError(
            "Expedition policy cannot create retention authority"
        )
    try:
        outer = validate_run_artifact_lifecycle_contract_v1(
            _mapping(payload["outer_contract"], field="outer lifecycle contract")
        )
    except ArtifactLifecycleError as error:
        raise ExpeditionLifecycleError(
            f"outer lifecycle contract is invalid: {error}"
        ) from error
    if (
        outer["lifecycle_profile"] != "SCIENCE_COMPACT"
        or outer["checkpoints"].get("authority") != "absolute_tick"
        or outer["checkpoints"].get("checkpoint_ticks") != []
        or outer["retention"].get("qualification_hold_binding") != "not_required"
        or outer["observability"].get("raw_tick_trace_policy") != "forbidden"
        or outer["observability"].get("debug_event_trace_policy") != "forbidden"
    ):
        raise ExpeditionLifecycleError(
            "outer coordinator lifecycle must be compact, tick-native, no-qualification, and trace-free"
        )
    return ExpeditionLifecyclePolicyV1(
        child_contract_class=str(child_profile),
        outer_contract=outer,
        lifecycle_authority=str(payload["lifecycle_authority"]),
        retention_authority_created=False,
    )


def expected_child_lifecycle_profile_v1(manifest: ExpeditionManifestV1) -> str:
    return resolve_expedition_lifecycle_policy_v1(manifest).child_contract_class


def _preflight_expedition_outer_object_budget_v1(
    manifest: ExpeditionManifestV1,
) -> dict[str, int]:
    """Prove the declared Expedition can fit the fixed outer object cap.

    Control, ledger, closeout, and founder-basis members are inventoried as
    exact sealed trees.  Exact parent candidates retain independent lifecycle
    rows because their final R1/R4 classifications may differ.  One action can
    stage at most one such package; the retained-entry term is an additional
    conservative recovery allowance for legacy stable package roots.
    """

    if not isinstance(manifest, ExpeditionManifestV1):
        raise ExpeditionLifecycleError("Expedition manifest is invalid")
    maximum_candidate_packages = (
        manifest.maximum_frontiers * manifest.maximum_actions_per_frontier
    )
    maximum_parent_package_objects = (
        maximum_candidate_packages + manifest.maximum_retained_parent_entries
    )
    maximum_outer_objects = (
        _OUTER_MANIFEST_OBJECT_RESERVE_V1 + maximum_parent_package_objects
    )
    if maximum_outer_objects > MAX_OUTER_LIFECYCLE_OBJECTS_V1:
        raise ExpeditionLifecycleError(
            "declared Expedition exceeds the fixed outer lifecycle object cap "
            "before runtime creation"
        )
    return {
        "fixed_coordination_object_reserve": _OUTER_MANIFEST_OBJECT_RESERVE_V1,
        "maximum_candidate_package_objects": maximum_candidate_packages,
        "maximum_recovery_parent_package_objects": (
            manifest.maximum_retained_parent_entries
        ),
        "maximum_outer_objects": maximum_outer_objects,
        "outer_object_cap": MAX_OUTER_LIFECYCLE_OBJECTS_V1,
    }


def default_expedition_lifecycle_policy_value_v1() -> dict[str, Any]:
    """Return the canonical compact outer/child policy for new manifests."""

    return {
        "child_contract_class": SCIENCE_COMPACT,
        "lifecycle_authority": "canonical_run_artifact_lifecycle",
        "outer_contract": lifecycle_profile_contract_v1(
            SCIENCE_COMPACT,
            total_max_bytes=DEFAULT_OUTER_LIFECYCLE_MAX_BYTES_V1,
            checkpoint_schedule=(),
            checkpoint_authority="absolute_tick",
        ),
        "retention_authority_created": False,
    }


def expedition_lifecycle_policy_value_v1(
    *, total_max_bytes: int = DEFAULT_OUTER_LIFECYCLE_MAX_BYTES_V1
) -> dict[str, Any]:
    """Build the canonical outer policy with an explicitly bound envelope.

    The default remains the checked-in 512 MiB compact envelope.  A larger
    value is an explicit manifest-bound execution choice, never a global
    change to the lifecycle policy.
    """

    if type(total_max_bytes) is not int or total_max_bytes <= 0:
        raise ExpeditionLifecycleError(
            "outer lifecycle total_max_bytes must be a positive integer"
        )
    return {
        "child_contract_class": SCIENCE_COMPACT,
        "lifecycle_authority": "canonical_run_artifact_lifecycle",
        "outer_contract": lifecycle_profile_contract_v1(
            SCIENCE_COMPACT,
            total_max_bytes=total_max_bytes,
            checkpoint_schedule=(),
            checkpoint_authority="absolute_tick",
        ),
        "retention_authority_created": False,
    }


def build_expedition_lifecycle_plan_v1(
    *, manifest: ExpeditionManifestV1, run_root: Path
) -> dict[str, Any]:
    policy = resolve_expedition_lifecycle_policy_v1(manifest)
    manifest_sha256 = manifest.manifest_sha256().hex
    core = {
        "schema_version": EXPEDITION_LIFECYCLE_PLAN_SCHEMA_VERSION_V1,
        "campaign_id": f"adaptive-expedition-{manifest_sha256[:24]}",
        "phase": "operator",
        "resolved_profile": "EIEL_ADAPTIVE_SCIENCE_EXPEDITION_OPERATOR_V1",
        "execution": {
            "execution_route_identity": "adaptive_expedition_coordination_v1",
            "time_authority": "absolute_tick_v1",
            "total_ticks": manifest.maximum_absolute_tick,
            "checkpoint_ticks": [],
            "budgets": {
                "disk_bytes": policy.outer_contract["storage"]["total_max_bytes"]
            },
        },
        "artifact_lifecycle": dict(policy.outer_contract),
        "output": {"runtime_root": str(run_root)},
        "cells": [],
        "expedition": {
            "expedition_id": manifest.expedition_id,
            "manifest_sha256": manifest_sha256,
            "founder_basis_sha256": manifest.founder_basis_sha256().hex,
            "policy_binding_sha256": canonical_sha256(
                manifest.policy_binding.to_dict()
            ).hex,
            "outer_search_seed": manifest.outer_search_seed,
            "child_contract_class": policy.child_contract_class,
        },
        "qualification_authority_effect": False,
    }
    plan = {**core, "plan_sha256": canonical_sha256(core).hex}
    ensure_lifecycle_contract_v1(plan)
    return plan


def _source_identity_v1(manifest: ExpeditionManifestV1) -> dict[str, Any]:
    return {
        "expedition_id": manifest.expedition_id,
        "manifest_sha256": manifest.manifest_sha256().hex,
        "founder_basis_sha256": manifest.founder_basis_sha256().hex,
        "policy_binding_sha256": canonical_sha256(
            manifest.policy_binding.to_dict()
        ).hex,
    }


def _passport_v1(
    *,
    context: ExpeditionLifecycleContextV1,
    terminal_status: str,
) -> dict[str, Any]:
    return build_run_passport_v1(
        context.plan,
        run_root=context.run_root,
        terminal_status=terminal_status,
        source_identity=context.source_identity,
        provider_identity={
            "provider_id": EXPEDITION_LIFECYCLE_PROVIDER_ID_V1,
            "profile_id": "EIEL_ADAPTIVE_SCIENCE_EXPEDITION_OPERATOR_V1",
            "execution_route_sha256": None,
        },
        config_identity=canonical_sha256(context.source_identity).hex,
        seed_identity=canonical_sha256(
            {
                "outer_search_seed": context.plan["expedition"]["outer_search_seed"],
                "manifest_sha256": context.source_identity["manifest_sha256"],
            }
        ).hex,
        disposition_manifest_path=str(context.run_root / _DISPOSITION_NAME),
        disposition_receipt_path=str(context.run_root / _STORAGE_RECEIPT_NAME),
    )


def open_expedition_lifecycle_v1(
    *, manifest: ExpeditionManifestV1, run_root: Path
) -> ExpeditionLifecycleContextV1:
    """Validate/admit the compact outer root before any Expedition dispatch."""

    root = run_root.expanduser().absolute()
    plan = build_expedition_lifecycle_plan_v1(manifest=manifest, run_root=root)
    contract = ensure_lifecycle_contract_v1(plan)
    source_identity = _source_identity_v1(manifest)
    guard = RunStorageGuard(
        contract,
        run_root=root,
        planned_bytes=int(contract["storage"]["total_max_bytes"]),
        reservation_window_bytes=int(contract["storage"]["total_max_bytes"]),
    )
    disposition_path = root / _DISPOSITION_NAME
    storage_path = root / _STORAGE_RECEIPT_NAME
    closeout_started = disposition_path.exists() or storage_path.exists()
    already_closed = closeout_started
    if already_closed and not (disposition_path.is_file() and storage_path.is_file()):
        # A manifest can be sealed before its receipt.  Admit that exact partial
        # closeout so the same bound process can finish it; every other partial
        # shape remains fail-closed.
        if not disposition_path.is_file() or storage_path.exists():
            raise ExpeditionLifecycleError(
                "outer lifecycle closeout is incomplete in an unsupported shape"
            )
        already_closed = False
    admitted = False
    if not already_closed:
        guard.enforce(stage="before_expedition_coordination")
        admitted = True
    context = ExpeditionLifecycleContextV1(
        run_root=root,
        manifest=manifest,
        plan=plan,
        contract=contract,
        source_identity=source_identity,
        guard=guard,
        admitted=admitted,
    )
    try:
        _publish_canonical_once(
            root / "expedition_lifecycle_plan.json",
            plan,
            field="Expedition lifecycle plan",
        )
        _publish_canonical_once(
            root / "artifact_lifecycle_contract.json",
            contract,
            field="Expedition artifact lifecycle contract",
        )
        if not closeout_started:
            write_run_passport_v1(
                root / _PASSPORT_NAME,
                _passport_v1(context=context, terminal_status="active"),
            )
    except Exception:
        if context.admitted:
            try:
                context.guard.release()
            finally:
                context.admitted = False
        raise
    return context


def enforce_expedition_storage_v1(
    context: ExpeditionLifecycleContextV1, *, stage: str
) -> Mapping[str, Any]:
    if not context.admitted:
        return {"status": "ALREADY_CLOSED", "ok": True, "stage": stage}
    return context.guard.enforce(stage=stage)


def _active_parent_release_authority_paths_v1(
    root: Path, decision_event_sha256: str
) -> tuple[Path, Path]:
    directory = root / _ACTIVE_PARENT_RELEASE_ROOT_V1
    stem = f"decision-{decision_event_sha256}"
    return directory / f"{stem}.authority.json", directory / f"{stem}.receipt.json"


def _active_parent_release_json_v1(path: Path, *, field: str) -> dict[str, Any]:
    return _read_mapping(path, field=field)


def _active_parent_release_authority_rows_v1(
    root: Path,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    frozenset[tuple[str, str]],
]:
    """Load authenticated prior active-release authorities for crash recovery."""

    directory = root / _ACTIVE_PARENT_RELEASE_ROOT_V1
    if not directory.exists():
        return {}, frozenset()
    if directory.is_symlink() or not directory.is_dir():
        raise ExpeditionLifecycleError(
            "active parent-release control root is not a regular directory"
        )
    result: dict[tuple[str, str], dict[str, Any]] = {}
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(directory.glob("*.authority.json")):
        value = _active_parent_release_json_v1(
            path, field="active parent-release authority"
        )
        if value.get("schema_version") != _ACTIVE_PARENT_RELEASE_SCHEMA_V1:
            raise ExpeditionLifecycleError(
                "active parent-release authority schema differs"
            )
        rows = value.get("artifact_rows")
        if not isinstance(rows, list):
            raise ExpeditionLifecycleError(
                "active parent-release authority rows are malformed"
            )
        receipt_path = path.with_name(
            path.name.removesuffix(".authority.json") + ".receipt.json"
        )
        if receipt_path.exists() or receipt_path.is_symlink():
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise ExpeditionLifecycleError(
                    "active parent-release receipt is not a regular file"
                )
            _validate_active_parent_release_receipt_v1(
                _active_parent_release_json_v1(
                    receipt_path, field="active parent-release receipt"
                ),
                authority=value,
                root=root,
            )

        for raw_row in rows:
            if not isinstance(raw_row, Mapping):
                raise ExpeditionLifecycleError(
                    "active parent-release authority row is malformed"
                )
            row = dict(raw_row)
            key = (str(row.get("state_ref_id")), str(row.get("path")))
            if key in result and result[key] != row:
                raise ExpeditionLifecycleError(
                    "active parent-release authorities conflict"
                )
            if receipt_path.exists() or receipt_path.is_symlink():
                if key in completed and completed[key] != row:
                    raise ExpeditionLifecycleError(
                        "active parent-release authorities conflict"
                    )
                completed[key] = row
            else:
                result[key] = row
    return result, frozenset(completed)


def _validate_active_parent_release_authority_v1(
    value: Mapping[str, Any],
    *,
    context: ExpeditionLifecycleContextV1,
    frontier_ordinal: int,
    decision_event_sha256: str,
    decision_receipt_sha256: str,
    selected_parent_state_ref_ids: Sequence[str],
    artifact_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    authority = _mapping(value, field="active parent-release authority")
    expected = {
        "schema_version": _ACTIVE_PARENT_RELEASE_SCHEMA_V1,
        "expedition_id": context.manifest.expedition_id,
        "manifest_sha256": context.manifest.manifest_sha256().hex,
        "lifecycle_contract_sha256": context.contract["contract_sha256"],
        "runtime_root": str(context.run_root),
        "boundary_frontier_ordinal": frontier_ordinal,
        "boundary_decision_event_sha256": decision_event_sha256,
        "boundary_decision_receipt_sha256": decision_receipt_sha256,
        "selected_parent_state_ref_ids": sorted(selected_parent_state_ref_ids),
        "artifact_rows": sorted(
            [dict(row) for row in artifact_rows],
            key=lambda row: (str(row.get("state_ref_id")), str(row.get("path"))),
        ),
    }
    if authority != expected:
        raise ExpeditionLifecycleError(
            "active parent-release authority differs from the committed Decision"
        )
    return authority


def _validate_active_parent_release_receipt_v1(
    value: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    receipt = _mapping(value, field="active parent-release receipt")
    core = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    expected_fields = {
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
    if (
        set(receipt) != expected_fields
        or receipt.get("schema_version") != _ACTIVE_PARENT_RELEASE_SCHEMA_V1
    ):
        raise ExpeditionLifecycleError("active parent-release receipt fields differ")
    authority_sha256 = canonical_sha256(authority).hex
    if (
        receipt.get("authority_sha256") != authority_sha256
        or receipt.get("status") != "RELEASED"
        or receipt.get("receipt_sha256") != canonical_sha256(core).hex
    ):
        raise ExpeditionLifecycleError("active parent-release receipt identity changed")
    authority_path = root / str(receipt.get("authority_path"))
    if authority_path.resolve() != (
        root / _ACTIVE_PARENT_RELEASE_ROOT_V1 / str(authority_path.name)
    ).resolve():
        raise ExpeditionLifecycleError("active parent-release authority path differs")
    expected_paths = sorted(str(row["path"]) for row in authority["artifact_rows"])
    if receipt.get("released_paths") != expected_paths:
        raise ExpeditionLifecycleError("active parent-release receipt paths differ")
    if receipt.get("released_bytes") != sum(
        int(row["bytes"]) for row in authority["artifact_rows"]
    ):
        raise ExpeditionLifecycleError("active parent-release receipt bytes differ")
    journal = root / str(receipt.get("release_journal"))
    try:
        journal.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ExpeditionLifecycleError(
            "active parent-release journal escapes the runtime root"
        ) from exc
    journal_value = _active_parent_release_json_v1(
        journal, field="active parent-release journal"
    )
    if (
        receipt.get("release_journal_status") != "COMPLETE"
        or journal_value.get("status") != "COMPLETE"
        or receipt.get("release_journal_sha256")
        != canonical_sha256(journal_value).hex
    ):
        raise ExpeditionLifecycleError("active parent-release journal is incomplete")
    for path in expected_paths:
        target = root / path
        if target.exists() or target.is_symlink():
            raise ExpeditionLifecycleError(
                "released parent package reappeared at its stable locator"
            )
    return receipt


def release_active_parent_packages_v1(
    context: ExpeditionLifecycleContextV1,
    *,
    ledger: Mapping[str, object],
    frontier_ordinal: int,
    decision_payload: Mapping[str, object],
    selected_parent_state_ref_ids: Sequence[str],
) -> Mapping[str, Any]:
    """Release discarded exact packages at a committed safe boundary.

    The function composes the existing exact prune actuator.  The outer
    lifecycle reservation remains live; only the owning guard capability can
    authorize this narrow R1 operation.
    """

    if not context.admitted:
        raise ExpeditionLifecycleError(
            "active parent-package release requires a live Expedition guard"
        )
    raw_events = ledger.get("events")
    if not isinstance(raw_events, list):
        raise ExpeditionLifecycleError("active parent-release ledger is malformed")
    selected_parents = {str(value) for value in selected_parent_state_ref_ids}
    decision_receipt = decision_payload.get(PARENT_PACKAGE_LEDGER_FIELD_V1)
    if not isinstance(decision_receipt, Mapping):
        return {
            "schema_version": _ACTIVE_PARENT_RELEASE_SCHEMA_V1,
            "status": "NOOP",
            "released_paths": [],
            "released_bytes": 0,
        }
    decision_receipt = _mapping(
        decision_receipt, field="active parent-release Decision receipt"
    )
    if decision_receipt.get("schema_version") != PARENT_PACKAGE_DECISION_SCHEMA_V1:
        raise ExpeditionLifecycleError("active parent-release Decision schema differs")
    decision_receipt_sha256 = canonical_sha256(decision_receipt).hex
    decision_events = []
    latest_rows: dict[str, tuple[dict[str, Any], str, str]] = {}
    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            raise ExpeditionLifecycleError("active parent-release ledger event is malformed")
        if (
            raw_event.get("event_type") != "DECISION_PREPARED"
            or raw_event.get("frontier_ordinal") is None
        ):
            continue
        payload = raw_event.get("payload")
        if not isinstance(payload, Mapping):
            raise ExpeditionLifecycleError("active parent-release Decision payload is malformed")
        raw_receipt = payload.get(PARENT_PACKAGE_LEDGER_FIELD_V1)
        if not isinstance(raw_receipt, Mapping):
            continue
        receipt = _mapping(raw_receipt, field="parent package Decision receipt")
        if receipt.get("schema_version") != PARENT_PACKAGE_DECISION_SCHEMA_V1:
            raise ExpeditionLifecycleError("parent package Decision receipt schema differs")
        event_sha256 = str(raw_event.get("event_sha256") or "")
        if len(event_sha256) != 64:
            raise ExpeditionLifecycleError("active parent-release Decision event identity is invalid")
        for raw_row in receipt.get("package_outcomes", []):
            if not isinstance(raw_row, Mapping):
                raise ExpeditionLifecycleError("parent package outcome is malformed")
            row = dict(raw_row)
            state_ref_id = str(row.get("state_ref_id") or "")
            if state_ref_id:
                latest_rows[state_ref_id] = (
                    row,
                    event_sha256,
                    canonical_sha256(receipt).hex,
                )
        if (
            int(raw_event.get("frontier_ordinal")) == frontier_ordinal
            and receipt == decision_receipt
        ):
            decision_events.append(raw_event)
    if len(decision_events) != 1:
        raise ExpeditionLifecycleError(
            "active parent-release boundary is not bound to exactly one Decision event"
        )
    decision_event_sha256 = str(decision_events[0]["event_sha256"])
    authority_path, receipt_path = _active_parent_release_authority_paths_v1(
        context.run_root, decision_event_sha256
    )
    if receipt_path.is_file():
        if not authority_path.is_file():
            raise ExpeditionLifecycleError(
                "active parent-release receipt lacks its authority"
            )
        authority = _active_parent_release_json_v1(
            authority_path, field="active parent-release authority"
        )
        authority_rows = authority.get("artifact_rows")
        if not isinstance(authority_rows, list):
            raise ExpeditionLifecycleError(
                "active parent-release authority rows are malformed"
            )
        _validate_active_parent_release_authority_v1(
            authority,
            context=context,
            frontier_ordinal=frontier_ordinal,
            decision_event_sha256=decision_event_sha256,
            decision_receipt_sha256=decision_receipt_sha256,
            selected_parent_state_ref_ids=sorted(selected_parents),
            artifact_rows=authority_rows,
        )
        receipt = _validate_active_parent_release_receipt_v1(
            _active_parent_release_json_v1(
                receipt_path, field="active parent-release receipt"
            ),
            authority=authority,
            root=context.run_root,
        )
        return receipt

    prior_authority_rows, completed_authority_keys = (
        _active_parent_release_authority_rows_v1(context.run_root)
    )
    artifact_rows: list[dict[str, Any]] = []
    for state_ref_id, (outcome, source_event_sha256, source_receipt_sha256) in sorted(
        latest_rows.items()
    ):
        if (
            outcome.get("status") != "DISCARDED_RELEASE_R1"
            or outcome.get("retention_class") != "R1"
            or state_ref_id in selected_parents
        ):
            continue
        package_root = str(outcome.get("package_root") or "")
        if not package_root or Path(package_root).is_absolute() or ".." in Path(package_root).parts:
            raise ExpeditionLifecycleError("active parent-release package path is invalid")
        package_path = context.run_root / package_root
        prior = prior_authority_rows.get((state_ref_id, package_root))
        if package_path.exists() or package_path.is_symlink():
            if package_path.is_symlink() or not package_path.is_dir():
                raise ExpeditionLifecycleError("active parent-release package is not a directory")
            inventory_row = build_sealed_artifact_inventory_row_v1(
                run_root=context.run_root,
                path=package_path,
                artifact_family="adaptive_expedition.discarded_exact_parent_candidate",
                retention_class="R1",
                owner_writer=EXPEDITION_LIFECYCLE_PROVIDER_ID_V1,
                release_predicate=(
                    "active exact release boundary, decision settled, and no bound "
                    "child references this parent package"
                ),
            )
            if inventory_row.get("bytes") != outcome.get("full_package_bytes"):
                raise ExpeditionLifecycleError("active parent-release package bytes differ")
            row = {
                **inventory_row,
                "state_ref_id": state_ref_id,
                "full_package_bytes": outcome["full_package_bytes"],
                "package_inventory_sha256": outcome["package_inventory_sha256"],
                "source_decision_event_sha256": source_event_sha256,
                "source_decision_receipt_sha256": source_receipt_sha256,
            }
        elif (state_ref_id, package_root) in completed_authority_keys:
            # The historical Decision remains in the ledger, but its exact
            # release already has a verified terminal receipt.  Do not feed
            # that old outcome back into the next boundary's selection.
            continue
        elif prior is not None:
            row = dict(prior)
        else:
            raise ExpeditionLifecycleError(
                "active parent-release package disappeared without exact authority"
            )
        artifact_rows.append(row)

    if not artifact_rows:
        return {
            "schema_version": _ACTIVE_PARENT_RELEASE_SCHEMA_V1,
            "status": "NOOP",
            "released_paths": [],
            "released_bytes": 0,
        }
    authority = _validate_active_parent_release_authority_v1(
        {
            "schema_version": _ACTIVE_PARENT_RELEASE_SCHEMA_V1,
            "expedition_id": context.manifest.expedition_id,
            "manifest_sha256": context.manifest.manifest_sha256().hex,
            "lifecycle_contract_sha256": context.contract["contract_sha256"],
            "runtime_root": str(context.run_root),
            "boundary_frontier_ordinal": frontier_ordinal,
            "boundary_decision_event_sha256": decision_event_sha256,
            "boundary_decision_receipt_sha256": decision_receipt_sha256,
            "selected_parent_state_ref_ids": sorted(selected_parents),
            "artifact_rows": artifact_rows,
        },
        context=context,
        frontier_ordinal=frontier_ordinal,
        decision_event_sha256=decision_event_sha256,
        decision_receipt_sha256=decision_receipt_sha256,
        selected_parent_state_ref_ids=sorted(selected_parents),
        artifact_rows=artifact_rows,
    )
    _publish_canonical_once(
        authority_path,
        authority,
        field="Active parent-release authority",
    )
    from tools.scripts.eiel_runtime_storage_prune import apply_prune_plan

    disposition = build_active_artifact_disposition_manifest_v1(
        context.plan,
        run_root=context.run_root,
        artifact_inventory=authority["artifact_rows"],
        active_release_authority=authority,
        max_objects=MAX_OUTER_LIFECYCLE_OBJECTS_V1,
    )
    release_plan = {
        "schema_version": "eiel_runtime_storage_prune_plan_v1",
        "root": str(context.run_root),
        "apply_required": True,
        "selected_count": len(disposition["artifacts"]),
        "selected_bytes": sum(
            int(row["bytes"]) for row in disposition["artifacts"]
        ),
        "selected": [dict(row) for row in disposition["artifacts"]],
    }
    result = apply_prune_plan(
        context.run_root,
        release_plan,
        disposition_manifest=disposition,
        _active_run_guard=context.guard,
    )
    selected_paths = sorted(str(row["path"]) for row in disposition["artifacts"])
    if (
        result.get("failures")
        or sorted(str(path) for path in result.get("deleted", [])) != selected_paths
        or result.get("release_journal_status") != "COMPLETE"
    ):
        raise ExpeditionLifecycleError(
            "active parent-release actuator did not complete the exact selection"
        )
    enforce_expedition_storage_v1(
        context,
        stage=f"after_parent_package_release_{frontier_ordinal:04d}",
    )
    journal_path = Path(str(result["release_journal"])).expanduser().resolve()
    try:
        journal_relative = journal_path.relative_to(context.run_root).as_posix()
    except ValueError as exc:
        raise ExpeditionLifecycleError(
            "active parent-release actuator journal escapes the runtime root"
        ) from exc
    journal_value = _active_parent_release_json_v1(
        journal_path, field="active parent-release journal"
    )
    receipt_core = {
        "schema_version": _ACTIVE_PARENT_RELEASE_SCHEMA_V1,
        "authority_sha256": canonical_sha256(authority).hex,
        "authority_path": authority_path.relative_to(context.run_root).as_posix(),
        "status": "RELEASED",
        "released_paths": selected_paths,
        "released_bytes": sum(
            int(row["bytes"]) for row in disposition["artifacts"]
        ),
        "release_journal": journal_relative,
        "release_journal_sha256": canonical_sha256(journal_value).hex,
        "release_journal_status": result["release_journal_status"],
    }
    receipt = {**receipt_core, "receipt_sha256": canonical_sha256(receipt_core).hex}
    _publish_canonical_once(
        receipt_path,
        receipt,
        field="Active parent-release receipt",
    )
    return receipt


def _inventory_binding(relative: str) -> dict[str, str]:
    if relative.startswith("closeout/"):
        return {
            "artifact_family": "adaptive_expedition.compact_closeout",
            "retention_class": "R5",
            "owner_writer": EXPEDITION_LIFECYCLE_PROVIDER_ID_V1,
            "release_predicate": "human review and successor Expedition obligations closed",
        }
    if relative.startswith("durable/ledger/"):
        return {
            "artifact_family": "adaptive_expedition.history_ledger",
            "retention_class": "R4",
            "owner_writer": EXPEDITION_LIFECYCLE_PROVIDER_ID_V1,
            "release_predicate": "Expedition audit and replay obligations closed",
        }
    if relative.startswith("durable/founder-basis/"):
        return {
            "artifact_family": "adaptive_expedition.initial_exact_founder",
            "retention_class": "R4",
            "owner_writer": EXPEDITION_LIFECYCLE_PROVIDER_ID_V1,
            "release_predicate": (
                "Expedition founder and all exact continuation obligations closed"
            ),
        }
    if relative.startswith("durable/retained-parents/"):
        return {
            "artifact_family": "adaptive_expedition.retained_exact_parent",
            "retention_class": "R4",
            "owner_writer": EXPEDITION_LIFECYCLE_PROVIDER_ID_V1,
            "release_predicate": (
                "Retained Parent Bank release decision and all child recovery "
                "obligations closed"
            ),
        }
    if relative.startswith("scratch/retained-parent-candidates/"):
        return {
            "artifact_family": "adaptive_expedition.exact_parent_candidate_active",
            "retention_class": "R1",
            "owner_writer": EXPEDITION_LIFECYCLE_PROVIDER_ID_V1,
            "release_predicate": (
                "next deterministic Parent Bank decision recorded and all active "
                "child recovery obligations closed"
            ),
        }
    if relative.startswith("materialized/"):
        return {
            "artifact_family": "adaptive_expedition.child_materialization",
            "retention_class": "R3",
            "owner_writer": EXPEDITION_LIFECYCLE_PROVIDER_ID_V1,
            "release_predicate": "all child execution and recovery obligations closed",
        }
    return {
        "artifact_family": "adaptive_expedition.lifecycle_control",
        "retention_class": "R5",
        "owner_writer": EXPEDITION_LIFECYCLE_PROVIDER_ID_V1,
        "release_predicate": "terminal Expedition authority retained",
    }


def _collect_inventory(context: ExpeditionLifecycleContextV1) -> list[dict[str, Any]]:
    excluded = {_DISPOSITION_NAME, _STORAGE_RECEIPT_NAME}
    try:
        parent_package_bindings = terminal_parent_package_inventory_bindings_v1(
            manifest=context.manifest,
            run_root=context.run_root,
        )
    except ParentPackageLifecycleError as error:
        raise ExpeditionLifecycleError(
            f"physical Parent Bank inventory is invalid: {error}"
        ) from error
    rows: list[dict[str, Any]] = []
    package_roots = tuple(sorted(parent_package_bindings))
    for package_root in package_roots:
        if len(rows) >= MAX_OUTER_LIFECYCLE_OBJECTS_V1:
            raise ExpeditionLifecycleError(
                "outer lifecycle inventory exceeds its fixed object cap"
            )
        package_path = context.run_root / package_root
        if package_path.is_symlink() or not package_path.is_dir():
            raise ExpeditionLifecycleError(
                "exact parent package root is not a regular directory: "
                f"{package_path}"
            )
        rows.append(
            build_sealed_artifact_inventory_row_v1(
                run_root=context.run_root,
                path=package_path,
                **parent_package_bindings[package_root],
            )
        )
    sealed_coordination_roots: list[str] = []
    for relative in _SEALED_COORDINATION_ROOTS_V1:
        path = context.run_root / relative
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_dir():
            raise ExpeditionLifecycleError(
                f"sealed coordination root is invalid: {path}"
            )
        if len(rows) >= MAX_OUTER_LIFECYCLE_OBJECTS_V1:
            raise ExpeditionLifecycleError(
                "outer lifecycle inventory exceeds its fixed object cap"
            )
        rows.append(
            build_sealed_artifact_inventory_row_v1(
                run_root=context.run_root,
                path=path,
                **_inventory_binding(f"{relative}/"),
            )
        )
        sealed_coordination_roots.append(relative)
    for path in sorted(context.run_root.rglob("*"), key=lambda item: str(item)):
        if path.is_symlink():
            raise ExpeditionLifecycleError(
                f"outer lifecycle inventory refuses symbolic link: {path}"
            )
        if not path.is_file():
            continue
        relative = str(path.relative_to(context.run_root)).replace("\\", "/")
        if relative in excluded:
            continue
        if any(
            relative.startswith(f"{sealed_root}/")
            for sealed_root in sealed_coordination_roots
        ):
            continue
        parent_binding = next(
            (
                parent_package_bindings[package_root]
                for package_root in package_roots
                if relative.startswith(f"{package_root}/")
            ),
            None,
        )
        is_parent_package = relative.startswith(
            (
                f"{PARENT_CANDIDATE_RELATIVE_ROOT_V1}/",
                f"{LEGACY_PARENT_PACKAGE_RELATIVE_ROOT_V1}/",
            )
        )
        if is_parent_package and parent_binding is None:
            raise ExpeditionLifecycleError(
                "exact parent package lacks a terminal Bank retain/discard receipt"
            )
        # The package is one exact sealed-tree object.  This makes canonical
        # release atomic at the package root and avoids mutating read-only
        # checkpoint subdirectories while retaining full member identity.
        if parent_binding is not None:
            continue
        if len(rows) >= MAX_OUTER_LIFECYCLE_OBJECTS_V1:
            raise ExpeditionLifecycleError(
                "outer lifecycle inventory exceeds its fixed object cap"
            )
        rows.append(
            build_sealed_artifact_inventory_row_v1(
                run_root=context.run_root,
                path=path,
                **(
                    parent_binding
                    if parent_binding is not None
                    else _inventory_binding(relative)
                ),
            )
        )
    return sorted(rows, key=lambda row: str(row["path"]))


def _verify_closeout(
    context: ExpeditionLifecycleContextV1,
) -> dict[str, Any]:
    manifest_path = context.run_root / _DISPOSITION_NAME
    storage_path = context.run_root / _STORAGE_RECEIPT_NAME
    passport_path = context.run_root / _PASSPORT_NAME
    manifest = validate_artifact_disposition_manifest_v1(
        _read_mapping(manifest_path, field="Expedition disposition manifest")
    )
    storage = _read_mapping(storage_path, field="Expedition storage receipt")
    passport = _read_mapping(passport_path, field="Expedition run passport")
    storage_core = {
        key: item for key, item in storage.items() if key != "receipt_sha256"
    }
    passport_core = {
        key: item for key, item in passport.items() if key != "passport_sha256"
    }
    manifest_inventory = manifest.get("artifacts")
    removed_rows = storage.get("removed", [])
    inventory_matches = False
    if isinstance(manifest_inventory, list) and isinstance(removed_rows, list):
        manifest_by_path = {
            str(row.get("path")): row
            for row in manifest_inventory
            if isinstance(row, Mapping)
        }
        removed_by_path = {
            str(row.get("path")): row
            for row in removed_rows
            if isinstance(row, Mapping)
        }
        removed_valid = len(removed_by_path) == len(removed_rows) and all(
            path in manifest_by_path
            and manifest_by_path[path].get("release_eligible") is True
            and manifest_by_path[path].get("retention_class") in {"R0", "R1", "R2"}
            and all(
                removed.get(key) == manifest_by_path[path].get(key)
                for key in ("bytes", "sha256", "tree_sha256")
                if key in manifest_by_path[path]
            )
            and not (context.run_root / path).exists()
            and not (context.run_root / path).is_symlink()
            for path, removed in removed_by_path.items()
        )
        if removed_valid and removed_by_path:
            excluded = {_DISPOSITION_NAME, _STORAGE_RECEIPT_NAME}
            expected_paths = sorted(set(manifest_by_path) - set(removed_by_path))
            survivor_tree_roots: list[str] = []
            survivors_match = True
            for relative in expected_paths:
                manifest_row = manifest_by_path[relative]
                target = context.run_root / relative
                if target.is_symlink() or not target.exists():
                    survivors_match = False
                    break
                try:
                    manifest_row = manifest_by_path[relative]
                    current_row = build_sealed_artifact_inventory_row_v1(
                        run_root=context.run_root,
                        path=target,
                        artifact_family=str(manifest_row["artifact_family"]),
                        retention_class=str(manifest_row["retention_class"]),
                        owner_writer=str(manifest_row["owner_writer"]),
                        release_predicate=str(manifest_row["release_predicate"]),
                    )
                except (ArtifactLifecycleError, OSError):
                    survivors_match = False
                    break
                if not all(
                    manifest_row.get(key) == item for key, item in current_row.items()
                ):
                    survivors_match = False
                    break
                if current_row.get("artifact_type") == "sealed_tree":
                    survivor_tree_roots.append(relative)
            if survivors_match:
                expected_file_paths = {
                    relative
                    for relative in expected_paths
                    if manifest_by_path[relative].get("artifact_type") == "file"
                }
                for path in context.run_root.rglob("*"):
                    if path.is_symlink():
                        survivors_match = False
                        break
                    if not path.is_file():
                        continue
                    relative = str(path.relative_to(context.run_root)).replace(
                        "\\", "/"
                    )
                    if relative in excluded:
                        continue
                    if relative.startswith(_ACTUATOR_CONTROL_PREFIXES_V1):
                        continue
                    if relative in expected_file_paths or any(
                        relative.startswith(f"{tree_root}/")
                        for tree_root in survivor_tree_roots
                    ):
                        continue
                    survivors_match = False
                    break
            inventory_matches = survivors_match
        elif removed_valid:
            current_inventory = _collect_inventory(context)
            inventory_matches = (
                len(manifest_inventory) == len(current_inventory)
                and [row.get("path") for row in manifest_inventory]
                == [row["path"] for row in current_inventory]
                and all(
                    all(
                        manifest_row.get(key) == item
                        for key, item in current_row.items()
                    )
                    and manifest_row.get("exact_object_identity") is True
                    and manifest_row.get("release_eligible")
                    is (current_row.get("retention_class") in {"R0", "R1"})
                    and manifest_row.get("disposition")
                    == (
                        "RELEASE"
                        if current_row.get("retention_class") in {"R0", "R1"}
                        else (
                            "RETAIN_DURABLE"
                            if current_row.get("retention_class") in {"R3", "R4", "R5"}
                            else "KEEP_ACTIVE"
                        )
                    )
                    for manifest_row, current_row in zip(
                        manifest_inventory, current_inventory, strict=True
                    )
                )
            )
    if (
        manifest.get("runtime_root") != str(context.run_root)
        or manifest.get("plan_sha256") != context.plan["plan_sha256"]
        or manifest.get("terminal_status") != "completed"
        or manifest.get("lifecycle_contract_sha256")
        != context.contract["contract_sha256"]
        or manifest.get("inventory", {}).get("complete") is not True
        or manifest.get("summary", {}).get("unresolved_rx_bytes") != 0
        or not inventory_matches
        or storage.get("manifest_sha256") != manifest.get("manifest_sha256")
        or storage.get("plan_sha256") != context.plan["plan_sha256"]
        or storage.get("receipt_sha256") != canonical_sha256(storage_core).hex
        or passport.get("plan_sha256") != context.plan["plan_sha256"]
        or passport.get("terminal_status") != "completed"
        or passport.get("passport_sha256") != canonical_sha256(passport_core).hex
    ):
        raise ExpeditionLifecycleError(
            "outer lifecycle closeout is identity-inconsistent"
        )
    return {
        "contract_path": str(context.run_root / "artifact_lifecycle_contract.json"),
        "contract_sha256": context.contract["contract_sha256"],
        "run_passport_path": str(passport_path),
        "run_passport_sha256": passport["passport_sha256"],
        "artifact_disposition_manifest_path": str(manifest_path),
        "artifact_disposition_manifest_sha256": manifest["manifest_sha256"],
        "storage_disposition_receipt_path": str(storage_path),
        "storage_disposition_receipt_sha256": storage["receipt_sha256"],
        "inventory_complete": True,
        "terminal_status": "completed",
    }


def close_expedition_lifecycle_v1(
    context: ExpeditionLifecycleContextV1,
) -> dict[str, Any]:
    """Seal exact compact inventory and enqueue only canonical background work."""

    manifest_path = context.run_root / _DISPOSITION_NAME
    storage_path = context.run_root / _STORAGE_RECEIPT_NAME
    if manifest_path.is_file() and storage_path.is_file():
        with run_artifact_transaction_fence_v1(context.run_root):
            verified = _verify_closeout(context)
            register_pending_durable_archive_v1(
                manifest_path, _transaction_fence_held=True
            )
        return verified
    if context.admitted:
        context.guard.enforce(stage="expedition_terminal_closeout")
    write_run_passport_v1(
        context.run_root / _PASSPORT_NAME,
        _passport_v1(context=context, terminal_status="completed"),
    )
    inventory = _collect_inventory(context)
    disposition = build_artifact_disposition_manifest_v1(
        context.plan,
        run_root=context.run_root,
        terminal_status="completed",
        live_reference_status="none",
        audit_hold_status="active",
        qualification_hold_status="not_required",
        artifact_inventory=inventory,
        inventory_complete=True,
        max_objects=MAX_OUTER_LIFECYCLE_OBJECTS_V1,
    )
    produced_bytes = int(disposition["summary"]["produced_bytes"])
    storage = build_storage_disposition_receipt_v1(
        manifest=disposition,
        bytes_before=produced_bytes,
        bytes_retained=produced_bytes,
        bytes_released=0,
        bytes_unresolved_rx=int(disposition["summary"]["unresolved_rx_bytes"]),
        authorization_reason="dry_run_only_no_destructive_cleanup_authority",
        retention_references=[str(manifest_path)],
        plan_sha256=str(context.plan["plan_sha256"]),
        result="dry_run",
    )
    if context.admitted:
        context.guard.release()
        context.admitted = False
    with run_artifact_transaction_fence_v1(context.run_root):
        _publish_canonical_once(
            manifest_path,
            disposition,
            field="Expedition disposition manifest",
        )
        _publish_canonical_once(
            storage_path,
            storage,
            field="Expedition storage disposition receipt",
        )
        verified = _verify_closeout(context)
        register_pending_durable_archive_v1(manifest_path, _transaction_fence_held=True)
    return verified


def release_expedition_storage_v1(context: ExpeditionLifecycleContextV1) -> None:
    if context.admitted:
        context.guard.release()
        context.admitted = False


__all__ = (
    "EXPEDITION_LIFECYCLE_PLAN_SCHEMA_VERSION_V1",
    "EXPEDITION_LIFECYCLE_POLICY_SCHEMA_VERSION_V1",
    "DEFAULT_OUTER_LIFECYCLE_MAX_BYTES_V1",
    "MAX_OUTER_LIFECYCLE_OBJECTS_V1",
    "ExpeditionLifecycleContextV1",
    "ExpeditionLifecycleError",
    "ExpeditionLifecyclePolicyV1",
    "build_expedition_lifecycle_plan_v1",
    "close_expedition_lifecycle_v1",
    "default_expedition_lifecycle_policy_value_v1",
    "expedition_lifecycle_policy_value_v1",
    "enforce_expedition_storage_v1",
    "expected_child_lifecycle_profile_v1",
    "open_expedition_lifecycle_v1",
    "release_expedition_storage_v1",
    "release_active_parent_packages_v1",
    "resolve_expedition_lifecycle_policy_v1",
)
