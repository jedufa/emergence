"""Pure helpers for bounded resident CUDA generation chunk execution."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final


SUPPORTED_CHUNK_GENERATIONS_V1: Final[frozenset[int]] = frozenset(
    {1, 2, 8, 32, 64, 128}
)

_REQUIRED_ZERO_COUNTERS: Final[tuple[str, ...]] = (
    "python_tick_iterations",
    "python_generation_iterations",
)
_HOST_POLICY_COUNTERS: Final[tuple[str, ...]] = (
    "host_synchronizations_inside_loop",
    "intermediate_device_to_host_operations",
    "intermediate_host_to_device_operations",
    "intermediate_host_synchronizations",
    "aten_operations_inside_tick_loop",
    "exact_cpu_policy_service_calls_inside_tick_loop",
)
_HOST_POLICY_ATEN_LOWER_BOUND_SEMANTICS: Final[str] = (
    "lower_bound_from_explicit_native_host_transfers"
)
_HOST_POLICY_ATEN_ZERO_SEMANTICS: Final[str] = "exact_zero"
_AGGREGATE_OWNED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "runtime_jit",
        "hidden_fallback",
        "native_controller_invocations",
        "native_dispatches",
        "native_chunk_submissions",
        "python_chunk_control_boundaries",
        "resident_chunk_generation_start",
        "resident_chunk_generation_end",
        "initial_generation",
        "terminal_next_generation",
        "native_tick_iterations",
        "requested_generations",
        "completed_generations",
        "completed_generations_by_cell",
        "native_host_policy_service_used",
        "aten_operations_inside_tick_loop_count_semantics",
        *_HOST_POLICY_COUNTERS,
        *_REQUIRED_ZERO_COUNTERS,
    }
)


def _require_exact_int(value: object, *, path: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path} must be an exact integer >= {minimum}")
    return value


def _require_sequence(value: object, *, path: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ValueError(f"{path} must be a nonempty sequence")
    if not value:
        raise ValueError(f"{path} must be a nonempty sequence")
    return value


def validate_chunk_sequence(
    total_generations: int,
    chunks: Sequence[int],
) -> tuple[int, ...]:
    """Return an exact supported decomposition of ``total_generations``."""

    total = _require_exact_int(
        total_generations, path="total_generations", minimum=1
    )
    sequence = _require_sequence(chunks, path="chunks")
    normalized: list[int] = []
    for index, value in enumerate(sequence):
        chunk = _require_exact_int(value, path=f"chunks[{index}]", minimum=1)
        if chunk not in SUPPORTED_CHUNK_GENERATIONS_V1:
            allowed = ", ".join(
                str(item) for item in sorted(SUPPORTED_CHUNK_GENERATIONS_V1)
            )
            raise ValueError(
                f"chunks[{index}] must be one of {{{allowed}}}"
            )
        normalized.append(chunk)
    if sum(normalized) != total:
        raise ValueError("chunks must sum exactly to total_generations")
    return tuple(normalized)


def generation_slice(
    total_generations: int,
    generation_offset: int,
    chunk_generations: int,
) -> slice:
    """Return the bounded generation-row slice for one chunk."""

    total = _require_exact_int(
        total_generations, path="total_generations", minimum=1
    )
    start = _require_exact_int(
        generation_offset, path="generation_offset", minimum=0
    )
    chunk = _require_exact_int(
        chunk_generations, path="chunk_generations", minimum=1
    )
    stop = start + chunk
    if start >= total or stop > total:
        raise ValueError(
            "generation slice must satisfy 0 <= start < stop <= total_generations"
        )
    return slice(start, stop)


def tick_slice(
    total_generations: int,
    ticks_per_generation: int,
    generation_offset: int,
    chunk_generations: int,
) -> slice:
    """Return the flattened generation-major tick-row slice for one chunk."""

    ticks = _require_exact_int(
        ticks_per_generation, path="ticks_per_generation", minimum=1
    )
    generations = generation_slice(
        total_generations,
        generation_offset,
        chunk_generations,
    )
    return slice(generations.start * ticks, generations.stop * ticks)


def _json_safe_copy(value: object, *, path: str) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} keys must be strings")
            copied[key] = _json_safe_copy(item, path=f"{path}.{key}")
        return copied
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _json_safe_copy(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} contains a non-JSON-safe value")


def _require_counter(
    receipt: Mapping[str, object],
    field: str,
    *,
    receipt_index: int,
) -> int:
    if field not in receipt:
        raise ValueError(f"receipts[{receipt_index}].{field} is required")
    return _require_exact_int(
        receipt[field],
        path=f"receipts[{receipt_index}].{field}",
        minimum=0,
    )


def _is_additive_counter(field: str) -> bool:
    if field in _AGGREGATE_OWNED_FIELDS or "_per_" in field:
        return False
    if field in {
        "launch_count",
        "dispatch_count",
        "tick_count",
        "transfer_count",
        "synchronization_count",
    }:
        return True
    return field.endswith(
        (
            "_launches",
            "_dispatches",
            "_iterations",
            "_operations",
            "_synchronizations",
            "_transfers",
            "_ticks",
            "_launch_count",
            "_dispatch_count",
            "_tick_count",
            "_transfer_count",
            "_synchronization_count",
        )
    ) or field.startswith("ticks_")


def aggregate_native_chunk_receipts(
    receipts: Sequence[Mapping[str, object]],
    batch_cells: int,
    total_generations: int,
    ticks_per_generation: int,
    initial_generation: int = 0,
) -> dict[str, Any]:
    """Validate and aggregate exact native controller chunk receipts."""

    batch = _require_exact_int(batch_cells, path="batch_cells", minimum=1)
    total = _require_exact_int(
        total_generations, path="total_generations", minimum=1
    )
    ticks = _require_exact_int(
        ticks_per_generation, path="ticks_per_generation", minimum=1
    )
    initial = _require_exact_int(
        initial_generation, path="initial_generation", minimum=0
    )
    sequence = _require_sequence(receipts, path="receipts")

    normalized_receipts: list[dict[str, Any]] = []
    chunk_generations: list[int] = []
    host_policy_modes: list[bool] = []
    host_policy_totals = {field: 0 for field in _HOST_POLICY_COUNTERS}
    generation_cursor = initial
    for index, value in enumerate(sequence):
        if not isinstance(value, Mapping):
            raise ValueError(f"receipts[{index}] must be a mapping")
        receipt = value
        for field in ("runtime_jit", "hidden_fallback"):
            if receipt.get(field) is not False:
                raise ValueError(f"receipts[{index}].{field} must be false")
        for field in _REQUIRED_ZERO_COUNTERS:
            if _require_counter(receipt, field, receipt_index=index) != 0:
                raise ValueError(f"receipts[{index}].{field} must equal 0")

        host_policy_marker = receipt.get("native_host_policy_service_used", False)
        if type(host_policy_marker) is not bool:
            raise ValueError(
                f"receipts[{index}].native_host_policy_service_used "
                "must be a boolean"
            )
        host_policy_used = bool(host_policy_marker)
        host_policy_modes.append(host_policy_used)
        host_policy_counters: dict[str, int] = {}
        for field in _HOST_POLICY_COUNTERS:
            if (
                host_policy_used
                or field in receipt
                or field == "host_synchronizations_inside_loop"
            ):
                host_policy_counters[field] = _require_counter(
                    receipt, field, receipt_index=index
                )
            else:
                host_policy_counters[field] = 0

        service_calls = host_policy_counters[
            "exact_cpu_policy_service_calls_inside_tick_loop"
        ]
        host_syncs = host_policy_counters["host_synchronizations_inside_loop"]
        intermediate_syncs = host_policy_counters[
            "intermediate_host_synchronizations"
        ]
        aten_semantics = receipt.get(
            "aten_operations_inside_tick_loop_count_semantics",
            _HOST_POLICY_ATEN_ZERO_SEMANTICS,
        )
        if host_policy_used:
            if service_calls < 1:
                raise ValueError(
                    f"receipts[{index}] declared native host policy service "
                    "without any service calls"
                )
            if host_syncs != service_calls * 2:
                raise ValueError(
                    f"receipts[{index}].host_synchronizations_inside_loop "
                    "must equal two per exact CPU policy service call"
                )
            if intermediate_syncs != host_syncs:
                raise ValueError(
                    f"receipts[{index}].intermediate_host_synchronizations "
                    "must equal host_synchronizations_inside_loop"
                )
            transfer_fields = (
                "intermediate_device_to_host_operations",
                "intermediate_host_to_device_operations",
            )
            for field in transfer_fields:
                if host_policy_counters[field] < 1:
                    raise ValueError(
                        f"receipts[{index}].{field} must be positive when "
                        "native host policy service is used"
                    )
            expected_aten_lower_bound = sum(
                host_policy_counters[field] for field in transfer_fields
            )
            if (
                host_policy_counters["aten_operations_inside_tick_loop"]
                != expected_aten_lower_bound
            ):
                raise ValueError(
                    f"receipts[{index}].aten_operations_inside_tick_loop "
                    "must equal the explicit native host-transfer lower bound"
                )
            if aten_semantics != _HOST_POLICY_ATEN_LOWER_BOUND_SEMANTICS:
                raise ValueError(
                    f"receipts[{index}]."
                    "aten_operations_inside_tick_loop_count_semantics "
                    "must identify the explicit native host-transfer lower bound"
                )
        else:
            nonzero_host_policy_fields = [
                field
                for field, counter in host_policy_counters.items()
                if counter != 0
            ]
            if nonzero_host_policy_fields:
                raise ValueError(
                    f"receipts[{index}] reports host policy work without "
                    "declaring native_host_policy_service_used: "
                    + ", ".join(nonzero_host_policy_fields)
                )
            if aten_semantics != _HOST_POLICY_ATEN_ZERO_SEMANTICS:
                raise ValueError(
                    f"receipts[{index}]."
                    "aten_operations_inside_tick_loop_count_semantics "
                    "must equal exact_zero when host policy service is unused"
                )
        for field, counter in host_policy_counters.items():
            host_policy_totals[field] += counter

        if _require_counter(receipt, "native_dispatches", receipt_index=index) != 1:
            raise ValueError(
                f"receipts[{index}].native_dispatches must equal 1"
            )
        if "native_controller_invocations" in receipt and _require_counter(
            receipt,
            "native_controller_invocations",
            receipt_index=index,
        ) != 1:
            raise ValueError(
                f"receipts[{index}].native_controller_invocations must equal 1"
            )

        requested = _require_counter(
            receipt, "requested_generations", receipt_index=index
        )
        completed = _require_counter(
            receipt, "completed_generations", receipt_index=index
        )
        if requested < 1 or completed != requested:
            raise ValueError(
                f"receipts[{index}] must report exact positive requested and "
                "completed generations"
            )
        chunk_start = _require_counter(
            receipt, "resident_chunk_generation_start", receipt_index=index
        )
        chunk_end = _require_counter(
            receipt, "resident_chunk_generation_end", receipt_index=index
        )
        expected_end = generation_cursor + requested
        if chunk_start != generation_cursor or chunk_end != expected_end:
            raise ValueError(
                f"receipts[{index}] generation interval must equal "
                f"[{generation_cursor}, {expected_end}]"
            )
        completed_by_cell = receipt.get("completed_generations_by_cell")
        if not isinstance(completed_by_cell, Sequence) or isinstance(
            completed_by_cell, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"receipts[{index}].completed_generations_by_cell must be a sequence"
            )
        if len(completed_by_cell) != batch or any(
            type(value) is not int or value != expected_end
            for value in completed_by_cell
        ):
            raise ValueError(
                f"receipts[{index}].completed_generations_by_cell must contain "
                f"{batch} exact absolute next-generation values"
            )
        native_ticks = _require_counter(
            receipt, "native_tick_iterations", receipt_index=index
        )
        if native_ticks != requested * ticks:
            raise ValueError(
                f"receipts[{index}].native_tick_iterations contradicts its "
                "generation count"
            )
        if host_policy_used and service_calls != native_ticks:
            raise ValueError(
                f"receipts[{index}]."
                "exact_cpu_policy_service_calls_inside_tick_loop "
                "must equal native_tick_iterations"
            )

        chunk_generations.append(requested)
        generation_cursor = expected_end
        normalized_receipts.append(
            _json_safe_copy(receipt, path=f"receipts[{index}]")
        )

    validate_chunk_sequence(total, chunk_generations)
    if any(mode != host_policy_modes[0] for mode in host_policy_modes[1:]):
        raise ValueError(
            "receipts must not change native host policy service mode "
            "between resident chunks"
        )

    all_fields = set().union(*(receipt.keys() for receipt in sequence))
    additive: dict[str, int] = {}
    for field in sorted(all_fields):
        if type(field) is not str:
            raise ValueError("receipt keys must be strings")
        if not _is_additive_counter(field):
            continue
        additive[field] = sum(
            _require_counter(receipt, field, receipt_index=index)
            for index, receipt in enumerate(sequence)
        )

    chunk_count = len(sequence)
    aggregate: dict[str, Any] = {
        "initial_generation": initial,
        "terminal_next_generation": initial + total,
        "native_controller_invocations": chunk_count,
        "native_dispatches": chunk_count,
        "native_chunk_submissions": chunk_count,
        "python_chunk_control_boundaries": max(0, chunk_count - 1),
        "native_tick_iterations": total * ticks,
        "requested_generations": total,
        "completed_generations": total,
        "completed_generations_by_cell": [initial + total] * batch,
        "python_generation_iterations": 0,
        "python_tick_iterations": 0,
        "native_host_policy_service_used": host_policy_modes[0],
        "aten_operations_inside_tick_loop_count_semantics": (
            _HOST_POLICY_ATEN_LOWER_BOUND_SEMANTICS
            if host_policy_modes[0]
            else _HOST_POLICY_ATEN_ZERO_SEMANTICS
        ),
        "runtime_jit": False,
        "hidden_fallback": False,
        "resident_state_rebuilt_between_chunks": False,
        "full_state_transfers_between_chunks": 0,
        "production_native_chunk_loop": False,
    }
    aggregate.update(host_policy_totals)
    aggregate.update(additive)
    aggregate["chunk_receipts"] = normalized_receipts
    return aggregate


def pointer_identity_receipt(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, Any]:
    """Report exact pointer-map equality with deterministic mismatch order."""

    inventories: list[dict[str, int]] = []
    for label, value in (("before", before), ("after", after)):
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be a mapping")
        inventory: dict[str, int] = {}
        for name, pointer in value.items():
            if type(name) is not str:
                raise ValueError(f"{label} keys must be strings")
            inventory[name] = _require_exact_int(
                pointer, path=f"{label}.{name}", minimum=0
            )
        inventories.append(inventory)

    before_inventory, after_inventory = inventories
    mismatches = [
        {
            "surface": name,
            "before": before_inventory.get(name),
            "after": after_inventory.get(name),
        }
        for name in sorted(before_inventory.keys() | after_inventory.keys())
        if before_inventory.get(name) != after_inventory.get(name)
    ]
    return {
        "all_exact": not mismatches,
        "pointer_count": len(before_inventory),
        "first_mismatch": mismatches[0] if mismatches else None,
        "mismatches": mismatches,
    }
