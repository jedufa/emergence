"""Compose standalone planners into a hash-bound planning-only receipt.

This module has no execution, CUDA discovery, artifact loading, or routing
authority. Caller-supplied inventory is data. The only artifact operation is
an explicit streaming SHA-256 verification of the selected regular file.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from engine.artifact_catalog import (
    ArtifactCatalogV2,
    ArtifactCatalogError,
    verify_artifact_sha256,
)
from engine.capacity_profiles import CapacityProfileError, validate_profile
from engine.device_selection import DeviceSelectionError, plan_device_selection
from engine.dispatch_planning import (
    DispatchPlanningError,
    LogicalDispatchRequest,
    QualifiedChunkProfile,
    plan_dispatch,
)
from engine.qualification_plans import QualificationPlanError, validate_plan


PLANNING_REQUEST_SCHEMA_V1 = "eiel_planning_request_v1"
PLANNING_RECEIPT_SCHEMA_V1 = "eiel_planning_receipt_v1"
MAX_RECEIPT_PHYSICAL_DISPATCHES = 100_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}\Z")
_ID_RE = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*\Z")
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "logical_request",
        "dispatch",
        "artifact_constraints",
        "qualification_plan_ids",
    }
)
_DISPATCH_FIELDS = frozenset(
    {"profile_id", "qualified_chunk_sizes", "max_physical_dispatches", "tail_policy"}
)
_ARTIFACT_FIELDS = frozenset(
    {"kind", "abi", "mechanics_envelope_id", "required_capabilities"}
)
_INVENTORY_FIELDS = frozenset(
    {
        "index",
        "uuid",
        "compute_capability",
        "available_device_memory_bytes",
        "memory_headroom_bytes",
        "estimated_latency_ms",
        "estimated_throughput",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "plan_id",
        "plan_sha256",
        "receipt_sha256",
        "mechanics_envelope_id",
        "qualified_chunk_sizes",
        "capabilities",
        "status",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "authority",
        "offline_verification_scope",
        "inputs",
        "resolved_profile",
        "qualification_bindings",
        "selected_device",
        "selected_artifact",
        "dispatch_plan",
        "receipt_sha256",
    }
)
_INPUT_FIELDS = frozenset(
    {
        "logical_request",
        "planning_request_sha256",
        "caller_inventory_sha256",
        "capacity_profile_sha256",
        "artifact_catalog_schema_version",
        "artifact_catalog_sha256",
        "qualification_plans",
    }
)
_SELECTED_DEVICE_FIELDS = _INVENTORY_FIELDS | frozenset(
    {
        "selection_basis",
        "caller_inventory_count",
        "artifact_compatible_device_count",
        "policy_eligible_device_count",
    }
)
_SELECTED_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "kind",
        "abi",
        "architecture",
        "build_identity",
        "status",
        "relative_path",
        "sha256",
        "requirements",
        "qualification",
        "hash_verification",
    }
)


class PlanningReceiptError(RuntimeError):
    """Planning inputs, composition, or receipt validation failed."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PlanningReceiptError("value is not canonical JSON data") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _object(value: Any, *, fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PlanningReceiptError(f"{name} must be an object with string keys")
    missing = sorted(fields - set(value))
    extra = sorted(set(value) - fields)
    if missing or extra:
        raise PlanningReceiptError(
            f"{name} fields are not exact; missing={missing!r} extra={extra!r}"
        )
    return dict(value)


def _token(value: Any, *, name: str, identifier: bool = False) -> str:
    pattern = _ID_RE if identifier else _TOKEN_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise PlanningReceiptError(f"{name} is not a valid stable token")
    return value


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PlanningReceiptError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanningReceiptError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise PlanningReceiptError(f"{name} must be a positive finite number")
    return result


def _sorted_tokens(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list):
        raise PlanningReceiptError(f"{name} must be an array")
    result = [_token(item, name=f"{name}[{index}]") for index, item in enumerate(value)]
    if result != sorted(set(result)):
        raise PlanningReceiptError(f"{name} must be sorted unique tokens")
    return result


def _chunk_sizes(value: Any, *, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise PlanningReceiptError(f"{name} must be a nonempty array")
    result = [_integer(item, name=f"{name}[{index}]", minimum=1) for index, item in enumerate(value)]
    if result != sorted(set(result), reverse=True):
        raise PlanningReceiptError(f"{name} must be unique and strictly descending")
    for larger, smaller in zip(result, result[1:]):
        if larger % smaller:
            raise PlanningReceiptError(f"{name} must form a divisibility chain")
    return result


def _max_physical_dispatches(value: Any) -> int:
    result = _integer(value, name="dispatch.max_physical_dispatches", minimum=1)
    if result > MAX_RECEIPT_PHYSICAL_DISPATCHES:
        raise PlanningReceiptError(
            "dispatch.max_physical_dispatches exceeds the planning receipt maximum "
            f"of {MAX_RECEIPT_PHYSICAL_DISPATCHES}"
        )
    return result


def _canonical_request(value: Any) -> dict[str, Any]:
    request = _object(value, fields=_REQUEST_FIELDS, name="planning_request")
    if request["schema_version"] != PLANNING_REQUEST_SCHEMA_V1:
        raise PlanningReceiptError("unsupported planning request schema_version")
    logical = LogicalDispatchRequest.parse(request["logical_request"])
    dispatch = _object(request["dispatch"], fields=_DISPATCH_FIELDS, name="dispatch")
    tail_policy = dispatch["tail_policy"]
    if tail_policy not in {"allow", "reject"}:
        raise PlanningReceiptError("dispatch.tail_policy must be 'allow' or 'reject'")
    artifact = _object(
        request["artifact_constraints"],
        fields=_ARTIFACT_FIELDS,
        name="artifact_constraints",
    )
    plan_ids = _sorted_tokens(request["qualification_plan_ids"], name="qualification_plan_ids")
    if not plan_ids or any(_ID_RE.fullmatch(item) is None for item in plan_ids):
        raise PlanningReceiptError("qualification_plan_ids must be nonempty stable identifiers")
    return {
        "schema_version": PLANNING_REQUEST_SCHEMA_V1,
        "logical_request": logical.logical_id,
        "dispatch": {
            "profile_id": _token(dispatch["profile_id"], name="dispatch.profile_id", identifier=True),
            "qualified_chunk_sizes": _chunk_sizes(
                dispatch["qualified_chunk_sizes"], name="dispatch.qualified_chunk_sizes"
            ),
            "max_physical_dispatches": _max_physical_dispatches(
                dispatch["max_physical_dispatches"]
            ),
            "tail_policy": tail_policy,
        },
        "artifact_constraints": {
            "kind": _token(artifact["kind"], name="artifact_constraints.kind"),
            "abi": _token(artifact["abi"], name="artifact_constraints.abi"),
            "mechanics_envelope_id": _token(
                artifact["mechanics_envelope_id"],
                name="artifact_constraints.mechanics_envelope_id",
            ),
            "required_capabilities": _sorted_tokens(
                artifact["required_capabilities"],
                name="artifact_constraints.required_capabilities",
            ),
        },
        "qualification_plan_ids": plan_ids,
    }


def _canonical_inventory(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PlanningReceiptError("caller_inventory must be an array")
    rows: list[dict[str, Any]] = []
    for position, raw in enumerate(value):
        row = _object(raw, fields=_INVENTORY_FIELDS, name=f"caller_inventory[{position}]")
        rows.append(
            {
                "index": _integer(row["index"], name=f"caller_inventory[{position}].index"),
                "uuid": _token(row["uuid"], name=f"caller_inventory[{position}].uuid"),
                "compute_capability": _token(
                    row["compute_capability"],
                    name=f"caller_inventory[{position}].compute_capability",
                ),
                "available_device_memory_bytes": _integer(
                    row["available_device_memory_bytes"],
                    name=f"caller_inventory[{position}].available_device_memory_bytes",
                ),
                "memory_headroom_bytes": _integer(
                    row["memory_headroom_bytes"],
                    name=f"caller_inventory[{position}].memory_headroom_bytes",
                ),
                "estimated_latency_ms": _positive_number(
                    row["estimated_latency_ms"],
                    name=f"caller_inventory[{position}].estimated_latency_ms",
                ),
                "estimated_throughput": _positive_number(
                    row["estimated_throughput"],
                    name=f"caller_inventory[{position}].estimated_throughput",
                ),
            }
        )
    rows.sort(key=lambda row: (row["index"], row["uuid"]))
    if len({row["index"] for row in rows}) != len(rows):
        raise PlanningReceiptError("caller inventory indexes must be unique")
    if len({row["uuid"] for row in rows}) != len(rows):
        raise PlanningReceiptError("caller inventory UUIDs must be unique")
    return rows


def _canonical_plans(
    plans: Any, expected_ids: list[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(plans, Sequence) or isinstance(plans, (str, bytes)):
        raise PlanningReceiptError("qualification_plans must be an array")
    checked: dict[str, dict[str, Any]] = {}
    try:
        for raw in plans:
            plan = validate_plan(raw)
            if plan["plan_id"] in checked:
                raise PlanningReceiptError("qualification plan IDs must be unique")
            checked[plan["plan_id"]] = plan
    except QualificationPlanError as exc:
        raise PlanningReceiptError(f"qualification plan is invalid: {exc}") from exc
    if sorted(checked) != expected_ids:
        raise PlanningReceiptError("qualification plan IDs do not match planning request")
    summaries = [
        {"plan_id": plan_id, "plan_sha256": _digest(checked[plan_id])}
        for plan_id in sorted(checked)
    ]
    return summaries, checked


def _canonical_bindings(
    value: Any,
    *,
    plans: Mapping[str, Mapping[str, Any]],
    request: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PlanningReceiptError("qualification_bindings must be an array")
    rows: list[dict[str, Any]] = []
    for position, raw in enumerate(value):
        binding = _object(
            raw,
            fields=_BINDING_FIELDS,
            name=f"qualification_bindings[{position}]",
        )
        plan_id = _token(
            binding["plan_id"],
            name=f"qualification_bindings[{position}].plan_id",
            identifier=True,
        )
        if plan_id not in plans:
            raise PlanningReceiptError(f"qualification binding references unknown plan {plan_id!r}")
        plan_sha = binding["plan_sha256"]
        receipt_sha = binding["receipt_sha256"]
        if not isinstance(plan_sha, str) or _SHA256_RE.fullmatch(plan_sha) is None:
            raise PlanningReceiptError("qualification binding plan_sha256 is invalid")
        if not isinstance(receipt_sha, str) or _SHA256_RE.fullmatch(receipt_sha) is None:
            raise PlanningReceiptError("qualification binding receipt_sha256 is invalid")
        if plan_sha != _digest(plans[plan_id]):
            raise PlanningReceiptError(f"qualification binding plan digest mismatch for {plan_id!r}")
        envelope = _token(
            binding["mechanics_envelope_id"],
            name=f"qualification_bindings[{position}].mechanics_envelope_id",
        )
        if envelope != request["artifact_constraints"]["mechanics_envelope_id"]:
            raise PlanningReceiptError("qualification binding mechanics envelope mismatch")
        if binding["status"] != "qualified":
            raise PlanningReceiptError("qualification binding status must be 'qualified'")
        capabilities = _sorted_tokens(
            binding["capabilities"],
            name=f"qualification_bindings[{position}].capabilities",
        )
        plan_capabilities = set(plans[plan_id]["capability_requirements"])
        for case in plans[plan_id]["cases"]:
            plan_capabilities.update(case["capability_requirements"])
        if not plan_capabilities.issubset(capabilities):
            raise PlanningReceiptError(
                f"qualification binding capabilities do not cover plan {plan_id!r}"
            )
        rows.append(
            {
                "plan_id": plan_id,
                "plan_sha256": plan_sha,
                "receipt_sha256": receipt_sha,
                "mechanics_envelope_id": envelope,
                "qualified_chunk_sizes": _chunk_sizes(
                    binding["qualified_chunk_sizes"],
                    name=f"qualification_bindings[{position}].qualified_chunk_sizes",
                ),
                "capabilities": capabilities,
                "status": "qualified",
            }
        )
    rows.sort(key=lambda row: row["plan_id"])
    if [row["plan_id"] for row in rows] != request["qualification_plan_ids"]:
        raise PlanningReceiptError("qualification binding IDs do not match planning request")
    if len({row["receipt_sha256"] for row in rows}) != 1:
        raise PlanningReceiptError(
            "qualification bindings must share one aggregate receipt digest"
        )
    qualified_chunks = {
        size for row in rows for size in row["qualified_chunk_sizes"]
    }
    if not set(request["dispatch"]["qualified_chunk_sizes"]).issubset(qualified_chunks):
        raise PlanningReceiptError("dispatch chunks are not covered by qualification bindings")
    bound_capabilities = {capability for row in rows for capability in row["capabilities"]}
    if not set(request["artifact_constraints"]["required_capabilities"]).issubset(
        bound_capabilities
    ):
        raise PlanningReceiptError("required capabilities are not covered by qualification bindings")
    return rows


def _catalog_mapping(catalog: ArtifactCatalogV2) -> dict[str, Any]:
    return {
        "schema_version": catalog.schema_version,
        "artifacts": [
            {
                "artifact_id": row.artifact_id,
                "kind": row.kind,
                "abi": row.abi,
                "architecture": row.architecture,
                "build_identity": row.build_identity,
                "status": row.status,
                "relative_path": row.relative_path.as_posix(),
                "sha256": row.sha256,
                "requirements": {
                    "compute_capability": row.requirements.compute_capability,
                    "minimum_device_memory_bytes": row.requirements.minimum_device_memory_bytes,
                },
                "qualification": {
                    "schema_version": row.qualification.schema_version,
                    "envelope_id": row.qualification.envelope_id,
                    "receipt_sha256": row.qualification.receipt_sha256,
                    "capabilities": list(row.qualification.capabilities),
                },
            }
            for row in catalog.artifacts
        ],
    }


def _device_policy(profile: Mapping[str, Any]) -> dict[str, Any]:
    benchmark = profile["benchmark"]
    selector = benchmark["device_selector"]
    if selector == "cpu":
        raise PlanningReceiptError("GPU planning receipt rejects the cpu device selector")
    if benchmark["objective"] == "correctness":
        raise PlanningReceiptError("GPU planning receipt requires latency or throughput objective")
    device_index: int | None = None
    device_uuid: str | None = None
    if selector.startswith("cuda:"):
        device_index = int(selector.removeprefix("cuda:"))
    elif selector.startswith("uuid:"):
        device_uuid = selector.removeprefix("uuid:")
    elif selector != "auto":
        raise PlanningReceiptError("unsupported benchmark device selector")
    return {
        "schema_version": "eiel_device_selection_policy_v1",
        "device_index": device_index,
        "device_uuid": device_uuid,
        "minimum_memory_headroom_bytes": benchmark["memory_headroom_bytes"],
        "objective": benchmark["objective"],
    }


def create_planning_receipt(
    planning_request: Any,
    *,
    caller_inventory: Any,
    capacity_profile: Any,
    artifact_catalog: ArtifactCatalogV2,
    qualification_plans: Any,
    qualification_bindings: Any,
) -> dict[str, Any]:
    """Return a verified deterministic receipt without executing a workload."""

    try:
        request = _canonical_request(planning_request)
        inventory = _canonical_inventory(caller_inventory)
        profile = validate_profile(capacity_profile)
        if not isinstance(artifact_catalog, ArtifactCatalogV2):
            raise PlanningReceiptError("artifact_catalog must be an ArtifactCatalogV2")
        if Path(profile["paths"]["artifact_root"]).resolve(strict=False) != artifact_catalog.artifact_root:
            raise PlanningReceiptError("capacity profile artifact root does not match catalog root")
        plan_summaries, plans = _canonical_plans(
            qualification_plans, request["qualification_plan_ids"]
        )
        bindings = _canonical_bindings(
            qualification_bindings, plans=plans, request=request
        )
        bound_receipt_sha256 = bindings[0]["receipt_sha256"]
        constraints = request["artifact_constraints"]

        viable: list[dict[str, Any]] = []
        for device in inventory:
            matches = tuple(
                artifact
                for artifact in artifact_catalog.filter_qualified(
                    abi=constraints["abi"],
                    compute_capability=device["compute_capability"],
                    available_device_memory_bytes=device["available_device_memory_bytes"],
                    mechanics_envelope_id=constraints["mechanics_envelope_id"],
                    kind=constraints["kind"],
                    required_capabilities=constraints["required_capabilities"],
                )
                if artifact.qualification.receipt_sha256 == bound_receipt_sha256
            )
            if len(matches) > 1:
                raise PlanningReceiptError(
                    f"artifact selection is ambiguous for device {device['uuid']!r}"
                )
            if matches:
                viable.append(device)
        if not viable:
            raise PlanningReceiptError(
                "no artifact-compatible device binds the requested qualification receipt"
            )
        projected = [
            {
                key: row[key]
                for key in (
                    "index",
                    "uuid",
                    "memory_headroom_bytes",
                    "estimated_latency_ms",
                    "estimated_throughput",
                )
            }
            for row in viable
        ]
        device_plan = plan_device_selection(projected, _device_policy(profile))
        selected_device = next(row for row in viable if row["uuid"] == device_plan.selected.uuid)
        selected_matches = tuple(
            artifact
            for artifact in artifact_catalog.filter_qualified(
                abi=constraints["abi"],
                compute_capability=selected_device["compute_capability"],
                available_device_memory_bytes=selected_device["available_device_memory_bytes"],
                mechanics_envelope_id=constraints["mechanics_envelope_id"],
                kind=constraints["kind"],
                required_capabilities=constraints["required_capabilities"],
            )
            if artifact.qualification.receipt_sha256 == bound_receipt_sha256
        )
        if len(selected_matches) != 1:
            raise PlanningReceiptError("selected device artifact provenance is not unique")
        artifact = selected_matches[0]
        verify_artifact_sha256(artifact, artifact_root=artifact_catalog.artifact_root)
        if artifact.qualification.receipt_sha256 != bound_receipt_sha256:
            raise PlanningReceiptError(
                "selected artifact qualification receipt does not bind every requested plan"
            )
        dispatch = request["dispatch"]
        logical = LogicalDispatchRequest.parse(request["logical_request"])
        dispatch_plan = plan_dispatch(
            logical,
            profile=QualifiedChunkProfile(
                profile_id=dispatch["profile_id"],
                request_prefix=logical.prefix,
                qualified_chunk_sizes=tuple(dispatch["qualified_chunk_sizes"]),
            ),
            max_physical_dispatches=dispatch["max_physical_dispatches"],
        )
        if dispatch["tail_policy"] == "reject" and dispatch_plan.tail.units:
            raise PlanningReceiptError("dispatch tail is rejected by policy")
    except PlanningReceiptError:
        raise
    except (
        ArtifactCatalogError,
        CapacityProfileError,
        DeviceSelectionError,
        DispatchPlanningError,
    ) as exc:
        raise PlanningReceiptError(str(exc)) from exc

    catalog_mapping = _catalog_mapping(artifact_catalog)
    resolved_profile = {
        "capacity_profile": profile,
        "dispatch": dispatch,
        "artifact_constraints": constraints,
        "qualification_plan_ids": request["qualification_plan_ids"],
    }
    receipt: dict[str, Any] = {
        "schema_version": PLANNING_RECEIPT_SCHEMA_V1,
        "status": "planned_verified",
        "authority": "planning_only",
        "offline_verification_scope": "self_consistency_only",
        "inputs": {
            "logical_request": request["logical_request"],
            "planning_request_sha256": _digest(request),
            "caller_inventory_sha256": _digest(inventory),
            "capacity_profile_sha256": _digest(profile),
            "artifact_catalog_schema_version": artifact_catalog.schema_version,
            "artifact_catalog_sha256": _digest(catalog_mapping),
            "qualification_plans": plan_summaries,
        },
        "resolved_profile": resolved_profile,
        "qualification_bindings": bindings,
        "selected_device": {
            **selected_device,
            "selection_basis": device_plan.selection_basis,
            "caller_inventory_count": len(inventory),
            "artifact_compatible_device_count": len(viable),
            "policy_eligible_device_count": device_plan.eligible_device_count,
        },
        "selected_artifact": {
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind,
            "abi": artifact.abi,
            "architecture": artifact.architecture,
            "build_identity": artifact.build_identity,
            "status": artifact.status,
            "relative_path": artifact.relative_path.as_posix(),
            "sha256": artifact.sha256,
            "requirements": {
                "compute_capability": artifact.requirements.compute_capability,
                "minimum_device_memory_bytes": artifact.requirements.minimum_device_memory_bytes,
            },
            "qualification": {
                "schema_version": artifact.qualification.schema_version,
                "envelope_id": artifact.qualification.envelope_id,
                "receipt_sha256": artifact.qualification.receipt_sha256,
                "capabilities": list(artifact.qualification.capabilities),
            },
            "hash_verification": {
                "algorithm": "sha256",
                "expected_sha256": artifact.sha256,
                "actual_sha256": artifact.sha256,
                "verified": True,
            },
        },
        "dispatch_plan": dispatch_plan.to_mapping(),
    }
    receipt["receipt_sha256"] = _digest(receipt)
    verify_planning_receipt(receipt)
    return receipt


def verify_planning_receipt(value: Any) -> None:
    """Verify self-digest and internal consistency without re-reading sources.

    Source catalogs, qualification plans, caller inventory, and artifact bytes
    are verified while creating the receipt. Offline verification cannot
    re-establish those observations without the original inputs and therefore
    grants no execution or routing authority.
    """

    try:
        _verify_planning_receipt(value)
    except PlanningReceiptError:
        raise
    except (CapacityProfileError, DispatchPlanningError) as exc:
        raise PlanningReceiptError(f"receipt structure is invalid: {exc}") from exc


def _verify_planning_receipt(value: Any) -> None:
    """Fail closed unless a planning receipt is structurally and cryptographically valid."""

    receipt = _object(value, fields=_RECEIPT_FIELDS, name="receipt")
    if receipt["schema_version"] != PLANNING_RECEIPT_SCHEMA_V1:
        raise PlanningReceiptError("unsupported planning receipt schema_version")
    if receipt["status"] != "planned_verified" or receipt["authority"] != "planning_only":
        raise PlanningReceiptError("receipt status or authority is invalid")
    if receipt["offline_verification_scope"] != "self_consistency_only":
        raise PlanningReceiptError("receipt offline verification scope is invalid")
    observed = receipt["receipt_sha256"]
    if not isinstance(observed, str) or _SHA256_RE.fullmatch(observed) is None:
        raise PlanningReceiptError("receipt_sha256 is invalid")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    if _digest(unsigned) != observed:
        raise PlanningReceiptError("receipt self digest mismatch")

    inputs = receipt["inputs"]
    inputs = _object(inputs, fields=_INPUT_FIELDS, name="receipt.inputs")
    if inputs["artifact_catalog_schema_version"] != "eiel_artifact_catalog_v2":
        raise PlanningReceiptError("receipt artifact catalog schema is invalid")
    for field in (
        "planning_request_sha256",
        "caller_inventory_sha256",
        "capacity_profile_sha256",
        "artifact_catalog_sha256",
    ):
        if not isinstance(inputs[field], str) or _SHA256_RE.fullmatch(inputs[field]) is None:
            raise PlanningReceiptError(f"receipt.inputs.{field} is invalid")
    resolved = _object(
        receipt["resolved_profile"],
        fields=frozenset(
            {"capacity_profile", "dispatch", "artifact_constraints", "qualification_plan_ids"}
        ),
        name="receipt.resolved_profile",
    )
    try:
        canonical_profile = validate_profile(resolved["capacity_profile"])
    except CapacityProfileError as exc:
        raise PlanningReceiptError("receipt contains an invalid capacity profile") from exc
    if _digest(canonical_profile) != inputs["capacity_profile_sha256"]:
        raise PlanningReceiptError("receipt capacity profile digest mismatch")
    request = _canonical_request(
        {
            "schema_version": PLANNING_REQUEST_SCHEMA_V1,
            "logical_request": inputs["logical_request"],
            "dispatch": resolved["dispatch"],
            "artifact_constraints": resolved["artifact_constraints"],
            "qualification_plan_ids": resolved["qualification_plan_ids"],
        }
    )
    if _digest(request) != inputs["planning_request_sha256"]:
        raise PlanningReceiptError("receipt planning request digest mismatch")
    plan_summaries = inputs["qualification_plans"]
    if (
        not isinstance(plan_summaries, list)
        or [row.get("plan_id") for row in plan_summaries if isinstance(row, dict)]
        != resolved["qualification_plan_ids"]
        or any(
            not isinstance(row, dict)
            or set(row) != {"plan_id", "plan_sha256"}
            or not isinstance(row["plan_sha256"], str)
            or _SHA256_RE.fullmatch(row["plan_sha256"]) is None
            for row in plan_summaries
        )
    ):
        raise PlanningReceiptError("receipt qualification plan summaries are invalid")
    device = _object(
        receipt["selected_device"],
        fields=_SELECTED_DEVICE_FIELDS,
        name="receipt.selected_device",
    )
    canonical_device = _canonical_inventory(
        [{field: device[field] for field in _INVENTORY_FIELDS}]
    )[0]
    if any(canonical_device[field] != device[field] for field in _INVENTORY_FIELDS):
        raise PlanningReceiptError("receipt selected device values are invalid")
    for field in (
        "caller_inventory_count",
        "artifact_compatible_device_count",
        "policy_eligible_device_count",
    ):
        _integer(device[field], name=f"receipt.selected_device.{field}", minimum=1)
    if not (
        device["policy_eligible_device_count"]
        <= device["artifact_compatible_device_count"]
        <= device["caller_inventory_count"]
    ):
        raise PlanningReceiptError("receipt selected device counts are inconsistent")
    if device["selection_basis"] not in {"explicit_device", "latency", "throughput"}:
        raise PlanningReceiptError("receipt device selection basis is invalid")
    artifact = _object(
        receipt["selected_artifact"],
        fields=_SELECTED_ARTIFACT_FIELDS,
        name="receipt.selected_artifact",
    )
    for field in ("artifact_id", "kind", "abi", "architecture", "build_identity"):
        _token(artifact[field], name=f"receipt.selected_artifact.{field}")
    if artifact["status"] != "qualified":
        raise PlanningReceiptError("receipt selected artifact is not qualified")
    relative_path = artifact["relative_path"]
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise PlanningReceiptError("receipt artifact relative path is invalid")
    pure_path = PurePosixPath(relative_path)
    if (
        pure_path.is_absolute()
        or pure_path.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise PlanningReceiptError("receipt artifact relative path is invalid")
    constraints = resolved["artifact_constraints"]
    if (
        artifact["kind"] != constraints["kind"]
        or artifact["abi"] != constraints["abi"]
        or artifact["architecture"] != device["compute_capability"]
    ):
        raise PlanningReceiptError("receipt artifact identity violates resolved constraints")
    if not isinstance(artifact["sha256"], str) or _SHA256_RE.fullmatch(artifact["sha256"]) is None:
        raise PlanningReceiptError("receipt selected artifact sha256 is invalid")
    requirements = _object(
        artifact["requirements"],
        fields=frozenset({"compute_capability", "minimum_device_memory_bytes"}),
        name="receipt.selected_artifact.requirements",
    )
    if (
        _token(requirements["compute_capability"], name="artifact compute capability")
        != device["compute_capability"]
        or _integer(
            requirements["minimum_device_memory_bytes"],
            name="artifact minimum device memory",
        )
        > device["available_device_memory_bytes"]
    ):
        raise PlanningReceiptError("receipt artifact and selected device are incompatible")
    qualification = _object(
        artifact["qualification"],
        fields=frozenset(
            {"schema_version", "envelope_id", "receipt_sha256", "capabilities"}
        ),
        name="receipt.selected_artifact.qualification",
    )
    _token(qualification["schema_version"], name="artifact qualification schema")
    if (
        not isinstance(qualification["receipt_sha256"], str)
        or _SHA256_RE.fullmatch(qualification["receipt_sha256"]) is None
    ):
        raise PlanningReceiptError("artifact qualification receipt digest is invalid")
    _sorted_tokens(qualification["capabilities"], name="artifact qualification capabilities")
    if qualification["envelope_id"] != constraints["mechanics_envelope_id"]:
        raise PlanningReceiptError("receipt artifact mechanics envelope mismatch")
    if not set(constraints["required_capabilities"]).issubset(
        qualification["capabilities"]
    ):
        raise PlanningReceiptError("receipt artifact capabilities violate resolved constraints")
    verification = artifact.get("hash_verification")
    if not isinstance(verification, dict) or set(verification) != {
        "algorithm",
        "expected_sha256",
        "actual_sha256",
        "verified",
    }:
        raise PlanningReceiptError("artifact hash verification fields are invalid")
    if (
        verification["algorithm"] != "sha256"
        or verification["verified"] is not True
        or verification["expected_sha256"] != artifact.get("sha256")
        or verification["actual_sha256"] != artifact.get("sha256")
    ):
        raise PlanningReceiptError("artifact hash verification binding is invalid")
    bindings = receipt["qualification_bindings"]
    if (
        not isinstance(bindings, list)
        or not bindings
        or any(not isinstance(row, dict) for row in bindings)
    ):
        raise PlanningReceiptError("receipt qualification bindings are invalid")
    bindings = [
        _object(
            row,
            fields=_BINDING_FIELDS,
            name=f"receipt.qualification_bindings[{position}]",
        )
        for position, row in enumerate(bindings)
    ]
    if [row["plan_id"] for row in bindings] != resolved["qualification_plan_ids"]:
        raise PlanningReceiptError("receipt qualification binding IDs mismatch")
    summaries_by_id = {row["plan_id"]: row["plan_sha256"] for row in plan_summaries}
    bound_chunks: set[int] = set()
    bound_capabilities: set[str] = set()
    for position, row in enumerate(bindings):
        if (
            row.get("status") != "qualified"
            or summaries_by_id.get(row.get("plan_id")) != row.get("plan_sha256")
            or row.get("receipt_sha256") != qualification["receipt_sha256"]
            or row.get("mechanics_envelope_id") != constraints["mechanics_envelope_id"]
        ):
            raise PlanningReceiptError("receipt qualification binding is invalid")
        chunks = _chunk_sizes(
            row["qualified_chunk_sizes"],
            name=f"receipt.qualification_bindings[{position}].qualified_chunk_sizes",
        )
        capabilities = _sorted_tokens(
            row["capabilities"],
            name=f"receipt.qualification_bindings[{position}].capabilities",
        )
        bound_chunks.update(chunks)
        bound_capabilities.update(capabilities)
    if not set(resolved["dispatch"]["qualified_chunk_sizes"]).issubset(bound_chunks):
        raise PlanningReceiptError("receipt dispatch chunks lack qualification binding")
    if not set(constraints["required_capabilities"]).issubset(bound_capabilities):
        raise PlanningReceiptError("receipt capabilities lack qualification binding")
    dispatch_plan = receipt["dispatch_plan"]
    if not isinstance(dispatch_plan, dict) or set(dispatch_plan) != {
        "schema_version",
        "profile_id",
        "logical_request",
        "qualified_units",
        "physical_dispatches",
        "tail",
    }:
        raise PlanningReceiptError("receipt dispatch plan fields are invalid")
    if dispatch_plan["logical_request"] != inputs["logical_request"]:
        raise PlanningReceiptError("receipt dispatch logical request mismatch")
    if (
        dispatch_plan["profile_id"] != resolved["dispatch"]["profile_id"]
        or dispatch_plan["schema_version"] != "eiel_dispatch_plan_v1"
    ):
        raise PlanningReceiptError("receipt dispatch profile binding is invalid")
    logical = LogicalDispatchRequest.parse(dispatch_plan["logical_request"])
    expected_dispatch_plan = plan_dispatch(
        logical,
        profile=QualifiedChunkProfile(
            profile_id=resolved["dispatch"]["profile_id"],
            request_prefix=logical.prefix,
            qualified_chunk_sizes=tuple(
                resolved["dispatch"]["qualified_chunk_sizes"]
            ),
        ),
        max_physical_dispatches=resolved["dispatch"]["max_physical_dispatches"],
    ).to_mapping()
    if dispatch_plan != expected_dispatch_plan:
        raise PlanningReceiptError("receipt dispatch plan is not canonical")
    physical = dispatch_plan["physical_dispatches"]
    tail = dispatch_plan["tail"]
    if not isinstance(physical, list) or not isinstance(tail, dict):
        raise PlanningReceiptError("receipt dispatch rows are invalid")
    offset = 0
    allowed_chunks = set(resolved["dispatch"]["qualified_chunk_sizes"])
    if len(physical) > resolved["dispatch"]["max_physical_dispatches"]:
        raise PlanningReceiptError("receipt physical dispatch count exceeds resolved maximum")
    for ordinal, row in enumerate(physical):
        if (
            not isinstance(row, dict)
            or set(row) != {"ordinal", "logical_offset", "units"}
            or row["ordinal"] != ordinal
            or row["logical_offset"] != offset
            or isinstance(row["units"], bool)
            or not isinstance(row["units"], int)
            or row["units"] <= 0
            or row["units"] not in allowed_chunks
        ):
            raise PlanningReceiptError("receipt physical dispatch sequence is invalid")
        offset += row["units"]
    if (
        set(tail) != {"logical_offset", "units"}
        or tail["logical_offset"] != offset
        or isinstance(tail["units"], bool)
        or not isinstance(tail["units"], int)
        or tail["units"] < 0
        or offset + tail["units"] != logical.units
        or dispatch_plan["qualified_units"] != offset
    ):
        raise PlanningReceiptError("receipt dispatch totals are invalid")
    if resolved["dispatch"]["tail_policy"] == "reject" and tail["units"]:
        raise PlanningReceiptError("receipt contains a tail rejected by resolved policy")
