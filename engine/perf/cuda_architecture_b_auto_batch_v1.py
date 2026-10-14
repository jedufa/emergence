from __future__ import annotations

"""Fail-closed automatic batch selection for CUDA Architecture B."""

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from typing import Any, Final

from engine.perf.cuda_architecture_b_memory_inventory_v1 import (
    AUTHORITY_BINDING_FIELDS_V1,
    MAX_ACCOUNTED_BYTES_V1,
    MemoryInventoryContractError,
    batch_sizing_bytes_from_memory_inventory_v1,
)


SCHEMA_VERSION: Final[str] = "eiel_cuda_architecture_b_auto_batch_v1"
RECEIPT_SCHEMA_VERSION: Final[str] = SCHEMA_VERSION

_MATCH_FIELDS: Final[tuple[str, ...]] = AUTHORITY_BINDING_FIELDS_V1
_STRING_FIELDS: Final[frozenset[str]] = frozenset(
    set(_MATCH_FIELDS) - {"generation_count", "ticks_per_generation"}
)
_SHA256_FIELDS: Final[frozenset[str]] = frozenset(
    {"extension_sha256", "state_layout_sha256"}
)


def _is_int(value: object) -> bool:
    return type(value) is int


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _positive_int(
    values: Mapping[str, Any],
    field: str,
    *,
    scope: str,
    errors: list[str],
) -> int | None:
    value = values.get(field)
    if (
        not _is_int(value)
        or value <= 0
        or value > MAX_ACCOUNTED_BYTES_V1
    ):
        errors.append(
            f"{scope}.{field}: expected an integer in "
            f"[1, {MAX_ACCOUNTED_BYTES_V1}]"
        )
        return None
    return value


def _nonnegative_int(
    values: Mapping[str, Any],
    field: str,
    *,
    scope: str,
    errors: list[str],
) -> int | None:
    value = values.get(field)
    if (
        not _is_int(value)
        or value < 0
        or value > MAX_ACCOUNTED_BYTES_V1
    ):
        errors.append(
            f"{scope}.{field}: expected an integer in "
            f"[0, {MAX_ACCOUNTED_BYTES_V1}]"
        )
        return None
    return value


def _parse_workload(
    value: object,
    *,
    scope: str,
) -> tuple[dict[str, Any], list[str]]:
    parsed = {field: None for field in _MATCH_FIELDS}
    if not isinstance(value, Mapping):
        return parsed, [f"{scope}: expected a mapping"]

    errors: list[str] = []
    for field in _MATCH_FIELDS:
        field_value = value.get(field)
        if field in _STRING_FIELDS:
            if (
                not isinstance(field_value, str)
                or not field_value
                or field_value != field_value.strip()
                or len(field_value) > 256
            ):
                errors.append(
                    f"{scope}.{field}: expected a trimmed nonempty string "
                    "<= 256 chars"
                )
                continue
            if field in _SHA256_FIELDS and (
                len(field_value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in field_value
                )
            ):
                errors.append(
                    f"{scope}.{field}: expected exactly 64 lowercase hex characters"
                )
                continue
            if field == "active_policy_family" and field_value not in {
                "fixed_mlp",
                "gru",
            }:
                errors.append(
                    f"{scope}.{field}: expected exactly 'fixed_mlp' or 'gru'"
                )
                continue
        elif (
            not _is_int(field_value)
            or field_value <= 0
            or field_value > MAX_ACCOUNTED_BYTES_V1
        ):
            errors.append(
                f"{scope}.{field}: expected an integer in "
                f"[1, {MAX_ACCOUNTED_BYTES_V1}]"
            )
            continue
        parsed[field] = field_value
    return parsed, errors


def _parse_resources(
    *,
    device_resources: object,
    memory_policy: object,
    kernel_resources: object,
    mandatory_kernels: object,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, int | None],
    list[str],
]:
    device = {"total_bytes": None, "free_bytes": None, "sm_count": None}
    memory = {
        "reserve_floor_bytes": None,
        "reserve_fraction": None,
        "shared_bytes": None,
        "per_cell_bytes": None,
    }
    calculations: dict[str, int | None] = {
        "reserve_fraction_bytes": None,
        "reserve_bytes": None,
        "memory_numerator_bytes": None,
        "memory_max_cells": None,
        "minimum_blocks_per_sm": None,
        "occupancy_wave_cells": None,
        "mechanical_max_cells": None,
    }
    errors: list[str] = []

    if not isinstance(device_resources, Mapping):
        errors.append("device_resources: expected a mapping")
    else:
        device["total_bytes"] = _positive_int(
            device_resources,
            "total_bytes",
            scope="device_resources",
            errors=errors,
        )
        device["free_bytes"] = _nonnegative_int(
            device_resources,
            "free_bytes",
            scope="device_resources",
            errors=errors,
        )
        device["sm_count"] = _positive_int(
            device_resources,
            "sm_count",
            scope="device_resources",
            errors=errors,
        )
        if (
            device["total_bytes"] is not None
            and device["free_bytes"] is not None
            and device["free_bytes"] > device["total_bytes"]
        ):
            errors.append(
                "device_resources.free_bytes: must not exceed total_bytes"
            )

    if not isinstance(memory_policy, Mapping):
        errors.append("memory_policy: expected a mapping")
    else:
        memory["reserve_floor_bytes"] = _nonnegative_int(
            memory_policy,
            "reserve_floor_bytes",
            scope="memory_policy",
            errors=errors,
        )
        memory["shared_bytes"] = _nonnegative_int(
            memory_policy,
            "shared_bytes",
            scope="memory_policy",
            errors=errors,
        )
        memory["per_cell_bytes"] = _positive_int(
            memory_policy,
            "per_cell_bytes",
            scope="memory_policy",
            errors=errors,
        )
        reserve_fraction = memory_policy.get("reserve_fraction")
        if type(reserve_fraction) not in (int, float):
            errors.append(
                "memory_policy.reserve_fraction: expected a finite number in [0, 1]"
            )
        else:
            try:
                normalized_fraction = float(reserve_fraction)
            except OverflowError:
                normalized_fraction = math.inf
            if (
                not math.isfinite(normalized_fraction)
                or not 0.0 <= normalized_fraction <= 1.0
            ):
                errors.append(
                    "memory_policy.reserve_fraction: expected a finite number "
                    "in [0, 1]"
                )
            else:
                memory["reserve_fraction"] = normalized_fraction

    mandatory_names: list[str] = []
    if not _is_sequence(mandatory_kernels):
        errors.append("mandatory_kernels: expected a nonempty sequence")
    else:
        for index, name in enumerate(mandatory_kernels):
            if not isinstance(name, str) or not name.strip():
                errors.append(
                    f"mandatory_kernels[{index}]: expected a nonempty string"
                )
            else:
                mandatory_names.append(name)
        if not mandatory_names:
            errors.append("mandatory_kernels: expected at least one kernel")
        duplicates = sorted(
            name
            for name, count in Counter(mandatory_names).items()
            if count > 1
        )
        if duplicates:
            errors.append(
                "mandatory_kernels: duplicate names: " + ", ".join(duplicates)
            )

    kernel_receipts: list[dict[str, Any]] = []
    if not isinstance(kernel_resources, Mapping):
        errors.append("kernel_resources: expected a mapping")
    else:
        for name in sorted(set(mandatory_names)):
            resource = kernel_resources.get(name)
            blocks = (
                resource.get("max_blocks_per_multiprocessor")
                if isinstance(resource, Mapping)
                else None
            )
            valid = (
                _is_int(blocks)
                and blocks > 0
                and blocks <= MAX_ACCOUNTED_BYTES_V1
            )
            kernel_receipts.append(
                {
                    "name": name,
                    "valid": valid,
                    "max_blocks_per_multiprocessor": blocks if valid else None,
                    "rejection_reasons": (
                        [] if valid else ["invalid_or_missing_blocks_per_sm"]
                    ),
                }
            )
            if not valid:
                errors.append(
                    f"kernel_resources.{name}.max_blocks_per_multiprocessor: "
                    "expected a positive integer"
                )

    if errors:
        return device, memory, kernel_receipts, calculations, errors

    reserve_fraction_bytes = math.floor(
        device["total_bytes"] * memory["reserve_fraction"]
    )
    reserve_bytes = max(
        memory["reserve_floor_bytes"], reserve_fraction_bytes
    )
    memory_numerator_bytes = (
        device["free_bytes"] - reserve_bytes - memory["shared_bytes"]
    )
    memory_max_cells = memory_numerator_bytes // memory["per_cell_bytes"]
    minimum_blocks_per_sm = min(
        receipt["max_blocks_per_multiprocessor"]
        for receipt in kernel_receipts
    )
    occupancy_wave_cells = device["sm_count"] * minimum_blocks_per_sm
    calculations = {
        "reserve_fraction_bytes": reserve_fraction_bytes,
        "reserve_bytes": reserve_bytes,
        "memory_numerator_bytes": memory_numerator_bytes,
        "memory_max_cells": memory_max_cells,
        "minimum_blocks_per_sm": minimum_blocks_per_sm,
        "occupancy_wave_cells": occupancy_wave_cells,
        "mechanical_max_cells": min(memory_max_cells, occupancy_wave_cells),
    }
    return device, memory, kernel_receipts, calculations, []


def _parse_envelopes(
    values: object,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not _is_sequence(values):
        return [], ["qualified_envelopes: expected a sequence"]

    receipts: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, value in enumerate(values):
        envelope, row_errors = _parse_workload(
            value, scope=f"qualified_envelopes[{index}]"
        )
        if isinstance(value, Mapping):
            batch_cells = value.get("batch_cells")
            qualified = value.get("qualified")
            envelope_id = value.get("envelope_id")
            if (
                not _is_int(batch_cells)
                or batch_cells <= 0
                or batch_cells > MAX_ACCOUNTED_BYTES_V1
            ):
                row_errors.append(
                    f"qualified_envelopes[{index}].batch_cells: "
                    f"expected an integer in [1, {MAX_ACCOUNTED_BYTES_V1}]"
                )
                batch_cells = None
            if type(qualified) is not bool:
                row_errors.append(
                    f"qualified_envelopes[{index}].qualified: expected a boolean"
                )
                qualified = None
            if "envelope_id" in value and (
                not isinstance(envelope_id, str) or not envelope_id.strip()
            ):
                row_errors.append(
                    f"qualified_envelopes[{index}].envelope_id: "
                    "expected a nonempty string"
                )
                envelope_id = None
        else:
            batch_cells = None
            qualified = None
            envelope_id = None
        envelope.update(
            {
                "batch_cells": batch_cells,
                "qualified": qualified,
                "envelope_id": envelope_id,
            }
        )
        receipts.append(
            {
                "envelope": envelope,
                "valid": not row_errors,
                "matches_request": None,
                "within_memory_limit": None,
                "within_occupancy_wave": None,
                "eligible": False,
                "rejection_reasons": [
                    error.split(": ", maxsplit=1)[-1]
                    for error in row_errors
                ],
            }
        )
        errors.extend(row_errors)

    if not values:
        errors.append("qualified_envelopes: expected at least one envelope")

    identity_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for receipt in receipts:
        if not receipt["valid"]:
            continue
        identity = tuple(
            receipt["envelope"][field] for field in _MATCH_FIELDS
        ) + (receipt["envelope"]["batch_cells"],)
        identity_rows.setdefault(identity, []).append(receipt)
    ambiguous_identities = {
        identity
        for identity, rows in identity_rows.items()
        if len(rows) > 1
        and (
            any(row["envelope"]["envelope_id"] is None for row in rows)
            or len({row["envelope"]["envelope_id"] for row in rows})
            != len(rows)
        )
    }
    if ambiguous_identities:
        errors.append(
            "qualified_envelopes: duplicate execution envelope and batch shape"
        )
        for receipt in receipts:
            identity = tuple(
                receipt["envelope"][field] for field in _MATCH_FIELDS
            ) + (receipt["envelope"]["batch_cells"],)
            if receipt["valid"] and identity in ambiguous_identities:
                receipt["valid"] = False
                receipt["rejection_reasons"].append("duplicate_envelope")
    return receipts, errors


def _envelope_sort_key(receipt: Mapping[str, Any]) -> tuple[Any, ...]:
    envelope = receipt["envelope"]
    return (
        envelope["world"] or "",
        envelope["profile"] or "",
        envelope["generation_count"] or -1,
        envelope["ticks_per_generation"] or -1,
        envelope["summary_mode"] or "",
        envelope["tick_output_storage"] or "",
        envelope["batch_cells"] or -1,
        envelope["envelope_id"] or "",
        str(envelope["qualified"]),
    )


def _select_split_resources_v1(
    *,
    request: Mapping[str, Any],
    qualified_envelopes: Sequence[Mapping[str, Any]],
    device_resources: Mapping[str, Any],
    memory_policy: Mapping[str, Any],
    kernel_resources: Mapping[str, Mapping[str, Any]],
    mandatory_kernels: Sequence[str],
) -> dict[str, Any]:
    """Select the largest explicitly qualified shape within both ceilings.

    The selector performs no CUDA calls. Malformed or ambiguous snapshots
    return a complete rejection receipt instead of raising or inferring a batch
    shape from the mechanical maximum.
    """

    parsed_request, request_errors = _parse_workload(request, scope="request")
    (
        device,
        memory,
        kernel_receipts,
        calculations,
        resource_errors,
    ) = _parse_resources(
        device_resources=device_resources,
        memory_policy=memory_policy,
        kernel_resources=kernel_resources,
        mandatory_kernels=mandatory_kernels,
    )
    envelope_receipts, envelope_errors = _parse_envelopes(
        qualified_envelopes
    )
    input_errors = request_errors + resource_errors + envelope_errors

    qualified_matching_shapes: set[int] = set()
    feasible_qualified_shapes: set[int] = set()
    request_valid = not request_errors
    for receipt in envelope_receipts:
        envelope = receipt["envelope"]
        if not receipt["valid"] or not request_valid:
            continue
        mismatches = [
            field
            for field in _MATCH_FIELDS
            if envelope[field] != parsed_request[field]
        ]
        receipt["matches_request"] = not mismatches
        receipt["rejection_reasons"].extend(
            f"request_mismatch:{field}" for field in mismatches
        )
        if mismatches:
            continue
        if envelope["qualified"] is not True:
            receipt["rejection_reasons"].append(
                "not_explicitly_qualified"
            )
            continue

        batch_cells = envelope["batch_cells"]
        qualified_matching_shapes.add(batch_cells)
        memory_max = calculations["memory_max_cells"]
        occupancy_wave = calculations["occupancy_wave_cells"]
        if memory_max is not None:
            receipt["within_memory_limit"] = batch_cells <= memory_max
            if not receipt["within_memory_limit"]:
                receipt["rejection_reasons"].append("exceeds_memory_max")
        if occupancy_wave is not None:
            receipt["within_occupancy_wave"] = batch_cells <= occupancy_wave
            if not receipt["within_occupancy_wave"]:
                receipt["rejection_reasons"].append(
                    "exceeds_occupancy_wave"
                )
        receipt["eligible"] = bool(
            receipt["within_memory_limit"]
            and receipt["within_occupancy_wave"]
        )
        if receipt["eligible"]:
            feasible_qualified_shapes.add(batch_cells)

    envelope_receipts.sort(key=_envelope_sort_key)
    selected_batch_cells = (
        max(feasible_qualified_shapes)
        if feasible_qualified_shapes and not input_errors
        else None
    )
    selected_envelope = next(
        (
            dict(receipt["envelope"])
            for receipt in envelope_receipts
            if selected_batch_cells is not None
            and receipt["eligible"]
            and receipt["envelope"]["batch_cells"]
            == selected_batch_cells
        ),
        None,
    )

    if input_errors:
        decision = "rejected_fail_closed_malformed_input"
        rejection_reasons = ["malformed_or_ambiguous_input"]
    elif not qualified_matching_shapes:
        decision = "rejected_no_qualified_matching_shape"
        rejection_reasons = ["no_explicitly_qualified_matching_shape"]
    elif not feasible_qualified_shapes:
        decision = "rejected_no_feasible_qualified_shape"
        rejection_reasons = ["no_feasible_explicitly_qualified_shape"]
    else:
        decision = "selected_explicitly_qualified_shape"
        rejection_reasons = []

    return {
        "schema_version": SCHEMA_VERSION,
        "ok": selected_batch_cells is not None,
        "status": "selected" if selected_batch_cells is not None else "rejected",
        "decision": decision,
        "selected_batch_cells": selected_batch_cells,
        "selected_envelope": selected_envelope,
        "request": parsed_request,
        "selection_policy": {
            "qualification_source": "explicit_matching_envelopes_only",
            "mechanical_limit_rule": (
                "min(memory_max_cells, occupancy_wave_cells)"
            ),
            "ranking": "largest_feasible_qualified_batch_cells",
            "mechanical_max_implies_qualification": False,
        },
        "resource_inputs": {
            "device": device,
            "memory_policy": memory,
            "mandatory_kernels": kernel_receipts,
        },
        "calculations": calculations,
        "qualified_matching_shapes": sorted(qualified_matching_shapes),
        "feasible_qualified_shapes": sorted(feasible_qualified_shapes),
        "envelope_receipts": envelope_receipts,
        "input_errors": input_errors,
        "rejection_reasons": rejection_reasons,
    }


def _combined_workload(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    return {
        "world": value.get("world"),
        "profile": value.get("profile"),
        "build_identity": value.get("build_identity"),
        "extension_sha256": value.get("extension_sha256"),
        "state_layout_sha256": value.get("state_layout_sha256"),
        "active_policy_family": value.get("active_policy_family"),
        "generation_count": value.get("generations"),
        "ticks_per_generation": value.get("ticks"),
        "summary_mode": value.get("summary_mode"),
        "tick_output_storage": value.get("tick_output_storage"),
    }


def _combined_envelopes(values: object) -> object:
    if not _is_sequence(values):
        return values
    converted: list[object] = []
    for value in values:
        if not isinstance(value, Mapping):
            converted.append(value)
            continue
        workload = _combined_workload(value)
        converted.append(
            {
                **workload,
                "batch_cells": value.get("batch_size"),
                "qualified": value.get("qualified"),
                "envelope_id": value.get("envelope_id"),
            }
        )
    return converted


def _combined_resource_inputs(
    resources: object,
) -> tuple[object, object, object, object]:
    if not isinstance(resources, Mapping):
        return resources, resources, resources, resources
    blocks = resources.get("mandatory_kernel_blocks_per_sm")
    if isinstance(blocks, Mapping):
        kernels = {
            name: {"max_blocks_per_multiprocessor": value}
            for name, value in blocks.items()
        }
        mandatory = list(blocks)
    else:
        kernels = blocks
        mandatory = blocks
    return (
        {
            "total_bytes": resources.get("total_memory_bytes"),
            "free_bytes": resources.get("free_memory_bytes"),
            "sm_count": resources.get("sm_count"),
        },
        {
            "reserve_floor_bytes": resources.get("reserve_floor_bytes"),
            "reserve_fraction": resources.get("reserve_fraction"),
            "shared_bytes": resources.get("shared_bytes"),
            "per_cell_bytes": resources.get("per_cell_bytes"),
        },
        kernels,
        mandatory,
    )


def _combined_rejection(code: str, details: Sequence[str]) -> dict[str, Any]:
    return {"code": code, "details": list(details)}


def _combined_reason(reason: str) -> str:
    mismatch_fields = {
        "world": "world",
        "profile": "profile",
        "build_identity": "build_identity",
        "extension_sha256": "extension_sha256",
        "state_layout_sha256": "state_layout_sha256",
        "active_policy_family": "active_policy_family",
        "generation_count": "generations",
        "ticks_per_generation": "ticks",
        "summary_mode": "summary",
        "tick_output_storage": "tick_output_storage",
    }
    if reason.startswith("request_mismatch:"):
        field = reason.partition(":")[2]
        return f"{mismatch_fields[field]}_mismatch"
    return {
        "exceeds_memory_max": "above_memory_max",
        "exceeds_occupancy_wave": "above_occupancy_wave",
    }.get(reason, reason)


def _select_combined_resources_v1(
    *,
    request: object,
    resources: object,
    qualified_envelopes: object,
) -> dict[str, Any]:
    device, memory, kernels, mandatory = _combined_resource_inputs(resources)
    split = _select_split_resources_v1(
        request=_combined_workload(request),
        qualified_envelopes=_combined_envelopes(qualified_envelopes),
        device_resources=device,
        memory_policy=memory,
        kernel_resources=kernels,
        mandatory_kernels=mandatory,
    )
    calculations = split["calculations"]
    combined_calculations = {
        "reserve_fraction_bytes": calculations["reserve_fraction_bytes"],
        "reserve_bytes": calculations["reserve_bytes"],
        "available_batch_bytes": calculations["memory_numerator_bytes"],
        "memory_max": calculations["memory_max_cells"],
        "minimum_mandatory_blocks_per_sm": calculations[
            "minimum_blocks_per_sm"
        ],
        "occupancy_wave": calculations["occupancy_wave_cells"],
        "mechanical_max": calculations["mechanical_max_cells"],
    }

    malformed = split["decision"] == "rejected_fail_closed_malformed_input"
    empty_registry = _is_sequence(qualified_envelopes) and not qualified_envelopes
    candidates: list[dict[str, Any]] = []
    if not malformed:
        selected_envelope = split["selected_envelope"] or {}
        selected_id = selected_envelope.get("envelope_id")
        selected_size = split["selected_batch_cells"]
        for row in split["envelope_receipts"]:
            envelope = row["envelope"]
            envelope_id = envelope["envelope_id"]
            reasons = [
                _combined_reason(reason)
                for reason in row["rejection_reasons"]
            ]
            selected = (
                selected_id is not None
                and envelope_id == selected_id
                and envelope["batch_cells"] == selected_size
            )
            tied_out = (
                selected_size is not None
                and row["eligible"]
                and envelope["batch_cells"] == selected_size
                and not selected
            )
            if tied_out:
                reasons.append("deterministic_envelope_id_tiebreak")
            candidates.append(
                {
                    "envelope_id": envelope_id,
                    "qualified": envelope["qualified"],
                    "world": envelope["world"],
                    "profile": envelope["profile"],
                    "build_identity": envelope["build_identity"],
                    "extension_sha256": envelope["extension_sha256"],
                    "state_layout_sha256": envelope["state_layout_sha256"],
                    "active_policy_family": envelope["active_policy_family"],
                    "generations": envelope["generation_count"],
                    "ticks": envelope["ticks_per_generation"],
                    "summary_mode": envelope["summary_mode"],
                    "tick_output_storage": envelope["tick_output_storage"],
                    "batch_size": envelope["batch_cells"],
                    "matches_request": row["matches_request"],
                    "within_memory_max": row["within_memory_limit"],
                    "within_occupancy_wave": row["within_occupancy_wave"],
                    "eligible": row["eligible"] and not tied_out,
                    "selected": selected,
                    "rejection_reasons": reasons,
                }
            )
        candidates.sort(key=lambda row: (row["batch_size"], row["envelope_id"]))

    if malformed and not empty_registry:
        errors = split["input_errors"]
        if any(error.startswith("request") for error in errors):
            rejection_code = "malformed_request"
        elif any(
            error.startswith(
                (
                    "device_resources",
                    "memory_policy",
                    "kernel_resources",
                    "mandatory_kernels",
                )
            )
            for error in errors
        ):
            rejection_code = "malformed_resources"
        else:
            rejection_code = "malformed_qualified_envelopes"
        rejection_reasons = [_combined_rejection(rejection_code, errors)]
    elif empty_registry or split["decision"] == "rejected_no_qualified_matching_shape":
        rejection_reasons = [
            _combined_rejection(
                "no_qualified_matching_envelope",
                ["no explicitly qualified envelope matched the request"],
            )
        ]
    elif split["decision"] == "rejected_no_feasible_qualified_shape":
        rejection_reasons = [
            _combined_rejection(
                "no_qualified_shape_within_limits",
                ["all qualified matching shapes exceed a mechanical ceiling"],
            )
        ]
    else:
        rejection_reasons = []

    selected = split["status"] == "selected"
    selected_envelope = split["selected_envelope"] or {}
    normalized_request = split["request"]
    normalized_inputs = split["resource_inputs"]
    normalized_device = normalized_inputs["device"]
    normalized_memory = normalized_inputs["memory_policy"]
    normalized_resources = {
        "total_memory_bytes": normalized_device["total_bytes"],
        "free_memory_bytes": normalized_device["free_bytes"],
        "reserve_floor_bytes": normalized_memory["reserve_floor_bytes"],
        "reserve_fraction": normalized_memory["reserve_fraction"],
        "shared_bytes": normalized_memory["shared_bytes"],
        "per_cell_bytes": normalized_memory["per_cell_bytes"],
        "sm_count": normalized_device["sm_count"],
        "mandatory_kernel_blocks_per_sm": {
            row["name"]: row["max_blocks_per_multiprocessor"]
            for row in normalized_inputs["mandatory_kernels"]
        },
    }
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "decision": "selected" if selected else "rejected",
        "selected_batch_size": split["selected_batch_cells"],
        "selected_envelope_id": selected_envelope.get("envelope_id"),
        "request": {
            "world": normalized_request["world"],
            "profile": normalized_request["profile"],
            "build_identity": normalized_request["build_identity"],
            "extension_sha256": normalized_request["extension_sha256"],
            "state_layout_sha256": normalized_request["state_layout_sha256"],
            "active_policy_family": normalized_request["active_policy_family"],
            "generations": normalized_request["generation_count"],
            "ticks": normalized_request["ticks_per_generation"],
            "summary_mode": normalized_request["summary_mode"],
            "tick_output_storage": normalized_request["tick_output_storage"],
        },
        "resources": normalized_resources,
        "computations": combined_calculations,
        "selection_policy": {
            "qualification": "explicit_matching_envelopes_only",
            "capacity": "min(memory_max, occupancy_wave)",
            "ranking": "largest_batch_then_lexicographic_envelope_id",
            "mechanical_max_implies_qualification": False,
        },
        "candidates": candidates,
        "rejection_reasons": rejection_reasons,
    }


def select_auto_batch_v1(
    *,
    request: Mapping[str, Any],
    qualified_envelopes: Sequence[Mapping[str, Any]],
    resources: Mapping[str, Any] | None = None,
    device_resources: Mapping[str, Any] | None = None,
    memory_policy: Mapping[str, Any] | None = None,
    kernel_resources: Mapping[str, Mapping[str, Any]] | None = None,
    mandatory_kernels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Select from explicit qualifications using either resource receipt form."""

    if resources is not None:
        return _select_combined_resources_v1(
            request=request,
            resources=resources,
            qualified_envelopes=qualified_envelopes,
        )
    return _select_split_resources_v1(
        request=request,
        qualified_envelopes=qualified_envelopes,
        device_resources=device_resources,
        memory_policy=memory_policy,
        kernel_resources=kernel_resources,
        mandatory_kernels=mandatory_kernels,
    )


def select_auto_batch_from_memory_inventory_v1(
    *,
    request: object,
    qualified_envelopes: Sequence[Mapping[str, Any]],
    resources: object,
    memory_inventory: object,
) -> dict[str, Any]:
    """Fail closed until native CUDA supplies trusted runtime inventory authority."""

    resource_mapping = isinstance(resources, Mapping)
    normalized_resources: object = (
        dict(resources) if resource_mapping else resources
    )
    expected_binding, request_errors = _parse_workload(
        _combined_workload(request),
        scope="request",
    )

    def inventory_summary() -> dict[str, Any]:
        if not isinstance(memory_inventory, Mapping):
            return {
                "schema_version": None,
                "status": None,
                "ok": False,
            }
        return {
            "schema_version": memory_inventory.get("schema_version"),
            "status": memory_inventory.get("status"),
            "ok": memory_inventory.get("ok"),
            "authority_binding_sha256": memory_inventory.get(
                "authority_binding_sha256"
            ),
        }

    if request_errors:
        receipt = select_auto_batch_v1(
            request=request,
            resources=resources,
            qualified_envelopes=qualified_envelopes,
        )
        receipt["memory_inventory"] = inventory_summary()
        return receipt

    try:
        sizing = batch_sizing_bytes_from_memory_inventory_v1(
            memory_inventory,
            expected_authority_binding=expected_binding,
        )
    except MemoryInventoryContractError as exc:
        if isinstance(normalized_resources, dict):
            normalized_resources["per_cell_bytes"] = None
            normalized_resources["shared_bytes"] = None
        receipt = select_auto_batch_v1(
            request=request,
            resources=normalized_resources,
            qualified_envelopes=qualified_envelopes,
        )
        receipt["rejection_reasons"] = [
            {
                "code": "memory_inventory_not_usable",
                "details": [str(exc)],
            }
        ]
        receipt["memory_inventory"] = inventory_summary()
        return receipt

    if not isinstance(normalized_resources, dict):
        receipt = select_auto_batch_v1(
            request=request,
            resources=normalized_resources,
            qualified_envelopes=qualified_envelopes,
        )
        receipt["memory_inventory"] = inventory_summary()
        return receipt

    conflicts = [
        field
        for field, value in sizing.items()
        if field in normalized_resources
        and normalized_resources[field] != value
    ]
    if conflicts:
        normalized_resources["per_cell_bytes"] = None
        normalized_resources["shared_bytes"] = None
        receipt = select_auto_batch_v1(
            request=request,
            resources=normalized_resources,
            qualified_envelopes=qualified_envelopes,
        )
        receipt["rejection_reasons"] = [
            {
                "code": "memory_inventory_resource_conflict",
                "details": [
                    f"resources.{field} disagrees with memory inventory"
                    for field in sorted(conflicts)
                ],
            }
        ]
        receipt["memory_inventory"] = {
            "schema_version": memory_inventory.get("schema_version"),
            "status": memory_inventory.get("status"),
            "ok": memory_inventory.get("ok"),
        }
        return receipt

    normalized_resources.update(sizing)
    receipt = select_auto_batch_v1(
        request=request,
        resources=normalized_resources,
        qualified_envelopes=qualified_envelopes,
    )
    assert isinstance(memory_inventory, Mapping)
    receipt["memory_inventory"] = {
        **inventory_summary(),
        "active_policy_family": memory_inventory.get("active_policy_family"),
        "cell_count": memory_inventory.get("cell_count"),
        "physical_storage_count": memory_inventory.get(
            "physical_storage_count"
        ),
        "per_cell_bytes": sizing["per_cell_bytes"],
        "shared_bytes": sizing["shared_bytes"],
    }
    return receipt


__all__ = [
    "RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "select_auto_batch_from_memory_inventory_v1",
    "select_auto_batch_v1",
]
