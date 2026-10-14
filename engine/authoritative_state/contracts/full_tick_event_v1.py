"""Exact contracts for the retained full-tick and event-state ABI families."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import torch

from engine.authoritative_state.errors import fail
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    PreparedDirectResidentTrajectoryV1,
    validate_prepared_direct_resident_trajectory_v1,
)


FULL_TICK_EVENT_CONTRACT_ERROR: Final[str] = "E_STATE_FULL_TICK_EVENT_CONTRACT"
PROFILE: Final[str] = "patch_fixed_mlp_direct_resident_v1"
BINDING_SOURCE: Final[str] = (
    "engine/perf/cuda_architecture_b_flattened_controller_v1_binding.cpp"
)
HEADER_SOURCE: Final[str] = (
    "engine/perf/cuda_architecture_b_flattened_controller_v1.hpp"
)
ALLOCATOR_SOURCE: Final[str] = (
    "engine/perf/cuda_architecture_b_direct_resident_source_allocator_v1.py"
)
CHECKPOINT_SOURCE: Final[str] = (
    "engine/perf/cuda_architecture_b_checkpoint.py"
)
CHRONICLE_SOURCE: Final[str] = "engine/native_runtime/lifecycle_events.py"

_FULL_TICK: Final[tuple[tuple[str, str, str, str, str, str], ...]] = (
    ("age", "int32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_state"),
    ("gru_upkeep_cost", "float32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_state"),
    ("cost_charge_scalar", "float32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_state"),
    ("cost_charge_maturation_scalar", "float32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_state"),
    ("drive_inertia", "float32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_state"),
    ("drive_noise", "float32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_state"),
    ("prev_drive_delta", "float32", "[B,S,D]", "sum(active_count_by_cell)*D", "live_slots", "resident_state"),
    ("drive_volatility", "float32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_state"),
    ("drive_jitter", "float32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_state"),
    ("episode0_drive_jitter_sum", "float32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_state"),
    ("last_energy_delta", "float32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_state"),
    ("last_health_delta", "float32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_state"),
    ("last_injury_delta", "float32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_state"),
    ("selfharm_events", "int32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_state"),
    ("episode0_selfharm_events", "int32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_state"),
    ("chemistry", "float32", "[B,S,CHEM]", "sum(active_count_by_cell)*CHEM", "live_slots", "resident_state"),
    ("runner_pcg64_state_words", "uint64", "[B,4]", "B*4", "full", "runner_rng_state"),
    ("runner_pcg64_buffer_words", "int64", "[B,2]", "B*2", "full", "runner_rng_state"),
    ("pcg64_state_words", "uint64", "[B,4]", "B*4", "full", "policy_rng_state"),
    ("pcg64_buffer_words", "int64", "[B,2]", "B*2", "full", "policy_rng_state"),
    (
        "drive_noise_samples_by_tick_slot",
        "float32",
        "[B,N,S,D]",
        "count_true(drive_noise_samples_valid_by_tick_slot)*D",
        "noise_values",
        "drive_noise_tape",
    ),
    (
        "drive_noise_samples_valid_by_tick_slot",
        "bool",
        "[B,N,S]",
        "count_true(drive_noise_samples_valid_by_tick_slot)",
        "noise_mask",
        "drive_noise_tape",
    ),
    ("slot_free_queue", "int64", "[B,S]", "sum(slot_free_queue_count_by_cell)", "queue", "resident_identity_queue"),
    ("slot_free_queue_count", "int64", "[B]", "B", "full", "resident_identity_queue"),
    ("residual_ids", "int64", "[B,S]", "sum(residual_count_by_cell)", "residual", "residual_scratch"),
    ("residual_slots", "int64", "[B,S]", "sum(residual_count_by_cell)", "residual", "residual_scratch"),
    ("residual_count", "int64", "[B]", "B", "full", "residual_scratch"),
    ("policy_draws", "int64", "[B]", "B", "full", "policy_rng_state"),
    ("metrics_draws", "int64", "[B]", "B", "full", "metrics_rng_state"),
)

_EVENT_NAMES: Final[tuple[str, ...]] = (
    "event_tape_type",
    "event_tape_generation",
    "event_tape_tick",
    "event_tape_actor_id",
    "event_tape_target_id",
    "event_tape_value",
    "typed_lifecycle_event_event_type",
    "typed_lifecycle_event_generation",
    "typed_lifecycle_event_tick",
    "typed_lifecycle_event_sequence",
    "typed_lifecycle_event_actor_agent_id",
    "typed_lifecycle_event_target_agent_id",
    "typed_lifecycle_event_actor_slot",
    "typed_lifecycle_event_target_slot",
    "typed_lifecycle_event_parent_agent_id",
    "typed_lifecycle_event_child_agent_id",
    "typed_lifecycle_event_lineage_founder_id",
    "typed_lifecycle_event_lineage_depth",
    "typed_lifecycle_event_subtype_code",
    "typed_lifecycle_event_reason_code",
    "typed_lifecycle_event_success",
    "typed_lifecycle_event_value",
)
_EVENT_DTYPES: Final[tuple[str, ...]] = (
    "int32", "int64", "int64", "int64", "int64", "float32",
    "int32", "int64", "int64", "int64", "int64", "int64", "int64", "int64",
    "int64", "int64", "int64", "int32", "int32", "int32", "int8", "float32",
)


def _capacity_formula(shape: str) -> str:
    return "*".join(shape.removeprefix("[").removesuffix("]").split(","))


def _stale_tail(kind: str) -> str:
    return {
        "live_slots": "non_live_physical_slots_are_not_authoritative",
        "full": "none_full_storage_authoritative",
        "noise_values": "mask_false_samples_are_stale",
        "noise_mask": "false_marks_a_stale_sample",
        "queue": "entries_after_each_cell_queue_count_are_stale",
        "residual": "entries_after_each_cell_residual_count_are_stale",
        "events": "entries_after_each_cell_event_active_count_are_stale",
        "initial_rows": "int64_max_entries_are_unused_reporting_rows",
        "reporting": "rows_without_an_initial_agent_id_are_stale",
    }[kind]


def _checkpoint(kind: str, name: str) -> str:
    if kind == "residual_scratch":
        return "excluded_ephemeral_scratch"
    if kind == "drive_noise_tape":
        return "canonical_initial_value_preserved_not_terminal_runtime"
    if kind in {"runner_rng_state", "policy_rng_state", "metrics_rng_state"}:
        return "rng_state_or_cursor_rebuilt_by_settlement"
    if name == "event_active_count":
        return "event_cursor_metadata"
    if name in _EVENT_NAMES:
        return "active_prefix_tensor"
    if name in {"event_status", "event_status_by_tick"}:
        return "validated_not_persisted"
    if name == "initial_agent_ids_by_row":
        return "recomputed_identity_mapping"
    if name in {"terminal_survival", "terminal_rows"}:
        return "reporting_output_not_checkpoint_state"
    return "terminal_resident_tensor"


def _scalar_domain(dtype: str, name: str) -> dict[str, Any]:
    domains: dict[str, tuple[str, int, str]] = {
        "bool": ("unsigned", 8, "false..true"),
        "int8": ("signed", 8, "-128..127"),
        "int32": ("signed", 32, "-2147483648..2147483647"),
        "int64": (
            "signed",
            64,
            "-9223372036854775808..9223372036854775807",
        ),
        "uint64": ("unsigned", 64, "0..18446744073709551615"),
        "float32": ("signed_ieee754", 32, "all_ieee754_binary32_bit_patterns"),
    }
    signedness, bit_width, valid_range = domains[dtype]
    sentinels: tuple[dict[str, Any], ...] = ()
    if name == "initial_agent_ids_by_row":
        sentinels = (
            {
                "value": 9223372036854775807,
                "meaning": "unused_reporting_row",
            },
        )
    elif name == "slot_free_queue":
        sentinels = ({"value": -1, "meaning": "unused_queue_tail"},)
    elif name == "residual_ids":
        sentinels = ({"value": -1, "meaning": "unused_residual_id_tail"},)
    elif name == "residual_slots":
        sentinels = ({"value": -1, "meaning": "unused_residual_slot_tail"},)
    return {
        "signedness": signedness,
        "bit_width": bit_width,
        "valid_range": valid_range,
        "sentinels": sentinels,
    }


def _row(
    *,
    ordinal: int,
    name: str,
    dtype: str,
    shape: str,
    valid: str,
    mutability: str,
    stale_kind: str,
    owner: str,
    checkpoint: str,
    chronicle: str,
) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "name": name,
        "kind": "tensor",
        "dtype": dtype,
        **_scalar_domain(dtype, name),
        "rank": shape.count(",") + 1,
        "shape_formula": shape,
        "capacity_formula": _capacity_formula(shape),
        "valid_row_count_formula": valid,
        "mutability": mutability,
        "native_pointer_constness": (
            "const_pointer" if mutability == "read_only" else "mutable_pointer"
        ),
        "stale_tail_semantics": _stale_tail(stale_kind),
        "checkpoint_participation": checkpoint,
        "chronicle_participation": chronicle,
        "provider_participation": "none_direct_route",
        "analysis_participation": "none_direct_route",
        "profile_availability": PROFILE,
        "semantic_owner": owner,
        "source": ALLOCATOR_SOURCE,
        "source_binding": BINDING_SOURCE,
        "source_pointer_declaration": HEADER_SOURCE,
        "checkpoint_source": CHECKPOINT_SOURCE,
        "chronicle_source": CHRONICLE_SOURCE,
        "boundary": "pre_dispatch_native_abi",
        "metadata_status": "supported_exact",
    }


def _full_tick_contract() -> tuple[dict[str, Any], ...]:
    rows = []
    for ordinal, (name, dtype, shape, valid, stale_kind, owner) in enumerate(_FULL_TICK):
        rows.append(
            _row(
                ordinal=ordinal,
                name=name,
                dtype=dtype,
                shape=shape,
                valid=valid,
                mutability="read_only" if ordinal in {2, 3} else "read_write",
                stale_kind=stale_kind,
                owner=owner,
                checkpoint=_checkpoint(owner, name),
                chronicle="not_directly_serialized",
            )
        )
    return tuple(rows)


def _event_contract() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for ordinal, (name, dtype) in enumerate(zip(_EVENT_NAMES, _EVENT_DTYPES, strict=True)):
        rows.append(
            _row(
                ordinal=ordinal,
                name=name,
                dtype=dtype,
                shape="[B,E]",
                valid="sum(event_active_count_by_cell)",
                mutability="read_write",
                stale_kind="events",
                owner="native_lifecycle_event_tape",
                checkpoint=_checkpoint("event_tape", name),
                chronicle=(
                    "typed_direct_route"
                    if ordinal >= 6
                    else "legacy_fallback_only"
                ),
            )
        )
    suffix = (
        ("event_active_count", "int64", "[B]", "B", "full", "native_lifecycle_event_cursor"),
        ("event_status", "int32", "[B]", "B", "full", "native_lifecycle_event_status"),
        ("lineage_founder_id", "int32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_lineage_state"),
        ("lineage_depth", "int32", "[B,S]", "sum(active_count_by_cell)", "live_slots", "resident_lineage_state"),
        ("initial_agent_ids_by_row", "int64", "[B,R]", "count_below_sentinel(initial_agent_ids_by_row)", "initial_rows", "terminal_reporting_identity"),
        ("terminal_survival", "int32", "[B,R]", "count_below_sentinel(initial_agent_ids_by_row)", "reporting", "terminal_reporting"),
        ("terminal_rows", "int32", "[B,R]", "count_below_sentinel(initial_agent_ids_by_row)", "reporting", "terminal_reporting"),
        ("event_status_by_tick", "int32", "[B,OUT]", "B*OUT", "full", "native_lifecycle_event_status"),
        ("event_sequence_base", "int64", "[B]", "B", "full", "native_lifecycle_event_cursor"),
    )
    for ordinal, (name, dtype, shape, valid, stale_kind, owner) in enumerate(
        suffix, start=len(_EVENT_NAMES)
    ):
        rows.append(
            _row(
                ordinal=ordinal,
                name=name,
                dtype=dtype,
                shape=shape,
                valid=valid,
                mutability=(
                    "read_only"
                    if name == "initial_agent_ids_by_row"
                    else "read_write"
                ),
                stale_kind=stale_kind,
                owner=owner,
                checkpoint=_checkpoint(owner, name),
                chronicle=(
                    "typed_direct_route_cursor"
                    if name == "event_active_count"
                    else "not_directly_serialized"
                ),
            )
        )
    return tuple(rows)


def build_full_tick_event_contracts_v1() -> dict[str, tuple[dict[str, Any], ...]]:
    """Return exact ABI-order metadata for both retained state families."""

    contracts = {
        "full_tick_state": _full_tick_contract(),
        "event_state": _event_contract(),
    }
    assert len(contracts["full_tick_state"]) == 29
    assert len(contracts["event_state"]) == 31
    return contracts


def unresolved_full_tick_event_entries_v1() -> tuple[str, ...]:
    """Report exact-metadata gaps; the closed contract has none."""

    return tuple(
        f"{family}.{row['name']}"
        for family, rows in build_full_tick_event_contracts_v1().items()
        for row in rows
        if row["metadata_status"] != "supported_exact"
    )


def _int_counts(
    tensor: Any, *, path: str, batch: int, capacity: int
) -> list[int]:
    if (
        not torch.is_tensor(tensor)
        or tensor.dtype != torch.int64
        or list(tensor.shape) != [batch]
    ):
        raise fail(
            FULL_TICK_EVENT_CONTRACT_ERROR,
            path,
            f"must be int64[{batch}]",
        )
    counts = [int(value) for value in tensor.detach().cpu().tolist()]
    if any(value < 0 or value > capacity for value in counts):
        raise fail(
            FULL_TICK_EVENT_CONTRACT_ERROR,
            path,
            f"values must be within [0,{capacity}]",
        )
    return counts


def _expected_shapes(
    prepared: PreparedDirectResidentTrajectoryV1,
) -> tuple[dict[str, int], list[int], list[int], int]:
    source = prepared.source
    full = source.full_tick_state_tensors
    event = source.event_state_tensors
    energy = source.live_state["energy"]
    drives = source.live_state["drives"]
    if not torch.is_tensor(energy) or energy.ndim != 2:
        raise fail(FULL_TICK_EVENT_CONTRACT_ERROR, "live_state.energy", "must be rank 2")
    if not torch.is_tensor(drives) or drives.ndim != 3:
        raise fail(FULL_TICK_EVENT_CONTRACT_ERROR, "live_state.drives", "must be rank 3")
    batch, slots = (int(value) for value in energy.shape)
    symbols = {
        "B": batch,
        "S": slots,
        "D": int(drives.shape[2]),
        "CHEM": int(full[15].shape[2]) if torch.is_tensor(full[15]) and full[15].ndim == 3 else -1,
        "N": int(full[20].shape[1]) if torch.is_tensor(full[20]) and full[20].ndim == 4 else -1,
        "E": int(event[0].shape[1]) if torch.is_tensor(event[0]) and event[0].ndim == 2 else -1,
        "I": int(event[26].shape[1]) if torch.is_tensor(event[26]) and event[26].ndim == 2 else -1,
        "R": int(event[27].shape[1]) if torch.is_tensor(event[27]) and event[27].ndim == 2 else -1,
        "OUT": int(event[29].shape[1]) if torch.is_tensor(event[29]) and event[29].ndim == 2 else -1,
    }
    active = _int_counts(
        source.active_count_state,
        path="active_count_state",
        batch=batch,
        capacity=slots,
    )
    event_counts = _int_counts(
        event[22],
        path="event_state_tensors.event_active_count",
        batch=batch,
        capacity=symbols["E"],
    )
    initial_ids = event[26]
    if not torch.is_tensor(initial_ids) or initial_ids.dtype != torch.int64:
        raise fail(
            FULL_TICK_EVENT_CONTRACT_ERROR,
            "event_state_tensors.initial_agent_ids_by_row",
            "must be an int64 tensor",
        )
    reporting_rows = int(
        (initial_ids.detach() < torch.iinfo(torch.int64).max).sum().item()
    )
    return symbols, active, event_counts, reporting_rows


def _shape_from_formula(formula: str, symbols: Mapping[str, int]) -> list[int]:
    return [
        int(part) if part.isdecimal() else symbols[part]
        for part in formula[1:-1].split(",")
    ]


def reconcile_full_tick_event_receipt_v1(
    receipt: Mapping[str, Any],
    *,
    prepared: PreparedDirectResidentTrajectoryV1,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Validate a prepared owner and replace observer heuristics with exact metadata."""

    validate_prepared_direct_resident_trajectory_v1(prepared)
    tensors = receipt.get("tensors")
    if not isinstance(tensors, list):
        raise fail(FULL_TICK_EVENT_CONTRACT_ERROR, "receipt.tensors", "must be a list")
    by_path = {
        str(row.get("path")): row
        for row in tensors
        if isinstance(row, Mapping)
    }
    contracts = build_full_tick_event_contracts_v1()
    symbols, active, event_counts, reporting_rows = _expected_shapes(prepared)
    residual_counts = _int_counts(
        prepared.source.full_tick_state_tensors[26],
        path="full_tick_state_tensors.residual_count",
        batch=symbols["B"],
        capacity=symbols["S"],
    )
    queue_counts = _int_counts(
        prepared.source.full_tick_state_tensors[23],
        path="full_tick_state_tensors.slot_free_queue_count",
        batch=symbols["B"],
        capacity=symbols["S"],
    )
    noise_mask = prepared.source.full_tick_state_tensors[21]
    if (
        not torch.is_tensor(noise_mask)
        or noise_mask.dtype != torch.bool
        or list(noise_mask.shape)
        != [symbols["B"], symbols["N"], symbols["S"]]
    ):
        raise fail(
            FULL_TICK_EVENT_CONTRACT_ERROR,
            "full_tick_state_tensors.drive_noise_samples_valid_by_tick_slot",
            "must be bool[B,N,S] and match the noise-value tape",
        )
    valid_noise = int(noise_mask.detach().sum().item())
    valid_by_formula = {
        "sum(active_count_by_cell)": sum(active),
        "sum(active_count_by_cell)*D": sum(active) * symbols["D"],
        "sum(active_count_by_cell)*CHEM": sum(active) * symbols["CHEM"],
        "sum(event_active_count_by_cell)": sum(event_counts),
        "sum(slot_free_queue_count_by_cell)": sum(queue_counts),
        "sum(residual_count_by_cell)": sum(residual_counts),
        "count_true(drive_noise_samples_valid_by_tick_slot)": valid_noise,
        "count_true(drive_noise_samples_valid_by_tick_slot)*D": valid_noise
        * symbols["D"],
        "count_below_sentinel(initial_agent_ids_by_row)": reporting_rows,
    }
    reconciled: dict[str, tuple[dict[str, Any], ...]] = {}
    for family, prefix in (
        ("full_tick_state", "full_tick_state_tensors"),
        ("event_state", "event_state_tensors"),
    ):
        rows: list[dict[str, Any]] = []
        for contract in contracts[family]:
            path = f"{prefix}.{contract['name']}"
            observed = by_path.get(path)
            if observed is None:
                raise fail(FULL_TICK_EVENT_CONTRACT_ERROR, path, "is absent from receipt")
            expected_shape = _shape_from_formula(contract["shape_formula"], symbols)
            if observed.get("dtype") != contract["dtype"]:
                raise fail(
                    FULL_TICK_EVENT_CONTRACT_ERROR,
                    path,
                    f"dtype must be {contract['dtype']}, got {observed.get('dtype')}",
                )
            if observed.get("rank") != contract["rank"] or observed.get("shape") != expected_shape:
                raise fail(
                    FULL_TICK_EVENT_CONTRACT_ERROR,
                    path,
                    f"shape must be {expected_shape}, got {observed.get('shape')}",
                )
            formula = contract["valid_row_count_formula"]
            if formula in valid_by_formula:
                valid_count = valid_by_formula[formula]
            else:
                valid_count = 1
                for dimension in expected_shape:
                    valid_count *= dimension
            exact = dict(observed)
            exact.update(
                mutability=contract["mutability"],
                valid_count=valid_count,
                valid_count_semantics=formula,
                stale_tail_semantics=contract["stale_tail_semantics"],
                metadata_status=contract["metadata_status"],
                contract_ordinal=contract["ordinal"],
            )
            rows.append(exact)
        reconciled[family] = tuple(rows)
    return reconciled


__all__ = [
    "FULL_TICK_EVENT_CONTRACT_ERROR",
    "build_full_tick_event_contracts_v1",
    "reconcile_full_tick_event_receipt_v1",
    "unresolved_full_tick_event_entries_v1",
]
