#!/usr/bin/env python3
from __future__ import annotations

"""Qualify retained bounded Flat/b1 compact rows without promoting capability."""

import argparse
import copy
import dataclasses
import hashlib
import json
import struct
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ENGINE: Final[Path] = ROOT / "engine"
for _path in (ROOT, ENGINE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from engine.native_runtime import compact_summary_planes as abi  # noqa: E402
from engine.native_runtime import compact_summary_projection as projection  # noqa: E402
from engine.native_runtime.checkpoint_digests import (  # noqa: E402
    DigestLayer,
    checkpoint_digest_manifest,
)
from engine.native_runtime.compact_summary_authority_adapter import (  # noqa: E402
    CompactSummaryAuthorityInputs,
    assemble_compact_summary_source,
)
from engine.native_runtime.compact_summary_cpu import (  # noqa: E402
    build_compact_summary_row,
)
from engine.native_runtime.public_metrics_overlay import (  # noqa: E402
    NATIVE_LINEAGE_METRICS_INPUT_KEY,
)
from engine.native_runtime.four_way_interchange import (  # noqa: E402
    _canonical_checkpoint_storage_verdict,
    _projection_digest_first_difference,
)
from tools.scripts.eiel_cuda_architecture_b_compact_summary_validator import (  # noqa: E402
    compare_exact_rows,
)
from simulation.metrics_pkg.lineage import (  # noqa: E402
    LineageMetricsState,
    compute_lineage_snapshot,
    native_lineage_agents_from_input,
)


SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_compact_summary_flat_g2_gate_v1"
)
GATE_ID: Final[str] = "CUDA-ARCH-B-STAGE3-FLAT-G2-COMPACT-ABI1"
GENERATION_COUNT: Final[int] = 2
SUPPORTED_GENERATION_COUNTS: Final[frozenset[int]] = frozenset(
    {1, 2, 8, 32, 64, 128}
)
BATCH_CELLS: Final[int] = 1
TICKS_PER_GENERATION: Final[int] = 80
PLANE_NAMES: Final[tuple[str, ...]] = (
    "validity",
    "defined_mask",
    "status_codes",
    "integer_values",
    "float_values",
    "digest_values",
    "text_values",
)
CAPABILITY_FIELDS: Final[tuple[str, str]] = (
    "resident_generation_summaries_supported",
    "eiel_compact_generation_summary_v1_supported",
)
PENDING_PRODUCER_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "producer_pending",
        "enabled_but_producer_pending",
        "all_invalid_noop",
        "not_allocated",
        "disabled",
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class FlatG2RetainedRun:
    """One bounded native invocation retained at the final host boundary."""

    compact_summary_planes: Any | None
    final_checkpoint_projection: Mapping[str, Any] | None
    receipt: Mapping[str, Any] | None
    summary_state_before: Any | None
    summary_state_after: Any | None
    rng_state_before: Any | None
    rng_state_after: Any | None
    process_identity: str | int | None


@dataclasses.dataclass(frozen=True, slots=True)
class FlatG2CpuOracleRun:
    """Canonical rows and checkpoint from one bounded independent CPU run."""

    canonical_rows: tuple[Mapping[str, Any], ...]
    final_checkpoint_projection: Mapping[str, Any]
    lineage_state_after: LineageMetricsState
    provenance: Mapping[str, Any]


NativeProducer = Callable[[], FlatG2RetainedRun | None]
CpuOracleProducer = Callable[[], FlatG2CpuOracleRun | None]
AuthorityAdapter = Callable[[Any], Any]
RowBuilder = Callable[[Any], Mapping[str, Any]]
CpuAuthorityInputBuilder = Callable[[Mapping[str, Any], int, LineageMetricsState], Any]
LineageStateAdvance = Callable[[Any], LineageMetricsState]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError("output must be a .json file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_generation_count(generation_count: Any) -> int:
    if type(generation_count) is not int or generation_count not in (
        SUPPORTED_GENERATION_COUNTS
    ):
        raise ValueError(
            "generation_count must be one of "
            f"{sorted(SUPPORTED_GENERATION_COUNTS)}; got {generation_count!r}"
        )
    return generation_count


def _generation_label(generation_count: Any) -> str:
    return f"g{generation_count}" if type(generation_count) is int else "g_invalid"


def _schema_version(generation_count: int) -> str:
    if generation_count == GENERATION_COUNT:
        return SCHEMA_VERSION
    return f"eiel_cuda_architecture_b_compact_summary_flat_g{generation_count}_gate_v1"


def _gate_id(generation_count: int) -> str:
    if generation_count == GENERATION_COUNT:
        return GATE_ID
    return f"CUDA-ARCH-B-STAGE3-FLAT-G{generation_count}-COMPACT-ABI1"


def _claims(
    *,
    generation_count: int = GENERATION_COUNT,
    execution_performed: bool = False,
    exact: bool = False,
) -> dict[str, Any]:
    generation_label = _generation_label(generation_count)
    return {
        "execution_performed": execution_performed,
        f"flat_{generation_label}_b1_rows_exact": exact,
        "final_checkpoint_exact": exact,
        "same_process_repeat_exact": exact,
        "summary_read_only": exact,
        "summary_rng_non_consuming": exact,
        "canonical_native_capability": False,
        "capability_promoted": False,
    }


def producer_pending_result(
    *,
    missing: Sequence[str],
    reason: str = "live controller compact retention/rows are unavailable",
    producer_calls: int = 0,
    cpu_oracle_calls: int = 0,
    generation_count: int = GENERATION_COUNT,
) -> dict[str, Any]:
    """Return the precise non-error state used until the native producer exists."""

    return {
        "schema_version": _schema_version(generation_count),
        "gate_id": _gate_id(generation_count),
        "mode": "live",
        "status": "producer_pending",
        "ok": False,
        "decision": "producer_pending",
        "reason": reason,
        "missing_integration_inputs": list(dict.fromkeys(missing)),
        "producer_calls": producer_calls,
        "cpu_oracle_calls": cpu_oracle_calls,
        "native_dispatches_attempted": producer_calls,
        "canonical_native_capability": False,
        "capability_promoted": False,
        "claims": _claims(generation_count=generation_count),
    }


def _rejected_result(
    *,
    decision: str,
    reason: str,
    first_mismatch: Any = None,
    details: Mapping[str, Any] | None = None,
    producer_calls: int = 2,
    cpu_oracle_calls: int = 0,
    generation_count: int = GENERATION_COUNT,
) -> dict[str, Any]:
    payload = {
        "schema_version": _schema_version(generation_count),
        "gate_id": _gate_id(generation_count),
        "mode": "live",
        "status": "rejected",
        "ok": False,
        "decision": decision,
        "reason": reason,
        "first_mismatch": first_mismatch,
        "producer_calls": producer_calls,
        "cpu_oracle_calls": cpu_oracle_calls,
        "canonical_native_capability": False,
        "capability_promoted": False,
        "claims": _claims(
            generation_count=generation_count,
            execution_performed=producer_calls > 0 or cpu_oracle_calls > 0
        ),
    }
    if details is not None:
        payload["details"] = dict(details)
        if isinstance(details.get("checks"), Mapping):
            payload["checks"] = dict(details["checks"])
        if isinstance(details.get("failed_checks"), Sequence):
            payload["failed_checks"] = list(details["failed_checks"])
    return payload


def _fingerprint(value: Any) -> str:
    digest = hashlib.sha256()
    active: set[int] = set()

    def add(item: Any) -> None:
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"numpy\0")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(array.view(np.uint8).tobytes())
            return
        if hasattr(item, "detach") and hasattr(item, "to") and hasattr(item, "dtype"):
            tensor = item.detach().to("cpu").contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().view(np.uint8).tobytes())
            return
        if item is None or type(item) in {bool, int, str, bytes}:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b"\0")
            digest.update(item if isinstance(item, bytes) else repr(item).encode("utf-8"))
            return
        if type(item) is float:
            digest.update(b"float\0")
            digest.update(struct.pack(">d", item))
            return
        if isinstance(item, np.generic):
            add(item.item())
            return

        identity = id(item)
        if identity in active:
            raise ValueError("cannot fingerprint cyclic retained evidence")
        active.add(identity)
        try:
            if isinstance(item, Mapping):
                digest.update(b"mapping\0")
                for key in sorted(item, key=lambda candidate: repr(candidate)):
                    add(key)
                    add(item[key])
                return
            if isinstance(item, (list, tuple)):
                digest.update(type(item).__name__.encode("ascii") + b"\0")
                for child in item:
                    add(child)
                return
            if isinstance(item, (set, frozenset)):
                digest.update(type(item).__name__.encode("ascii") + b"\0")
                for child in sorted(item, key=_fingerprint):
                    add(child)
                return
            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                digest.update(type(item).__qualname__.encode("utf-8") + b"\0")
                for field in dataclasses.fields(item):
                    digest.update(field.name.encode("utf-8") + b"\0")
                    add(getattr(item, field.name))
                return
            to_dict = getattr(item, "to_dict", None)
            if callable(to_dict):
                digest.update(type(item).__qualname__.encode("utf-8") + b"\0")
                add(to_dict())
                return
            attributes = getattr(item, "__dict__", None)
            if isinstance(attributes, Mapping):
                digest.update(type(item).__qualname__.encode("utf-8") + b"\0")
                add(attributes)
                return
            digest.update(type(item).__qualname__.encode("utf-8") + b"\0")
            digest.update(repr(item).encode("utf-8"))
        finally:
            active.remove(identity)

    add(value)
    return digest.hexdigest()


def _snapshot_verdict(before: Any, after: Any) -> dict[str, Any]:
    before_digest = _fingerprint(before)
    after_digest = _fingerprint(after)
    return {
        "exact": before_digest == after_digest,
        "before_sha256": before_digest,
        "after_sha256": after_digest,
    }


def _as_host_planes(value: Any) -> abi.CompactSummaryPlanes:
    if isinstance(value, abi.CompactSummaryPlanes):
        abi.validate_compact_summary_plane_layout(value)
        return value
    mapping = value.as_mapping() if hasattr(value, "as_mapping") else value
    if not isinstance(mapping, Mapping) or set(mapping) != set(PLANE_NAMES):
        raise TypeError("compact_summary_planes must expose the seven ABI planes")
    validity = mapping["validity"]
    if isinstance(validity, np.ndarray):
        planes = abi.CompactSummaryPlanes(
            **{name: mapping[name] for name in PLANE_NAMES}
        )
        abi.validate_compact_summary_plane_layout(planes)
        return planes
    from engine.perf import cuda_architecture_b_compact_summary as torch_abi

    return torch_abi.to_numpy_compact_summary_planes(mapping)


def _receipt_value(receipt: Mapping[str, Any], field: str) -> Any:
    if field in receipt:
        return receipt[field]
    native = receipt.get("native_controller_receipt")
    if isinstance(native, Mapping):
        return native.get(field)
    return None


def _receipt_verdict(
    receipt: Mapping[str, Any],
    *,
    generation_count: int = GENERATION_COUNT,
) -> dict[str, Any]:
    generation_count = _validate_generation_count(generation_count)
    exact_expected = {
        "native_dispatches": 1,
        "native_tick_iterations": generation_count * TICKS_PER_GENERATION,
        "requested_generations": generation_count,
        "completed_generations": generation_count,
        "completed_generations_by_cell": [generation_count],
        "python_generation_iterations": 0,
        "python_tick_iterations": 0,
        "aten_operations_inside_tick_loop": 0,
        "aten_dispatches_in_extension_call": 0,
        "aten_allocations_in_extension_call": 0,
        "outputs_preallocated_before_timing": True,
        "scratch_preallocated_before_timing": True,
        "torch_allocated_delta_bytes": 0,
        "torch_peak_allocated_delta_bytes": 0,
        "runtime_jit": False,
        "hidden_fallback": False,
        "compact_summary_planes_enabled": True,
        "summary_projection_mutates_simulation_state": False,
        "summary_projection_consumes_rng": False,
    }
    checks = {
        name: {
            "expected": expected,
            "actual": _receipt_value(receipt, name),
            "exact": _receipt_value(receipt, name) == expected,
        }
        for name, expected in exact_expected.items()
    }
    row_status = _receipt_value(receipt, "compact_summary_rows_status")
    row_status_exact = (
        type(row_status) is str
        and row_status not in PENDING_PRODUCER_STATUSES
        and bool(row_status)
    )
    checks["compact_summary_rows_status"] = {
        "expected": "non-pending retained canonical rows",
        "actual": row_status,
        "exact": row_status_exact,
    }
    pointer_stability = receipt.get("pointer_stability_receipt")
    pointer_stability_exact = (
        isinstance(pointer_stability, Mapping)
        and pointer_stability.get("all_exact") is True
        and pointer_stability.get("first_mismatch") is None
        and type(pointer_stability.get("pointer_count")) is int
        and pointer_stability["pointer_count"] > 0
    )
    checks["pointer_stability_receipt"] = {
        "expected": "all preallocated input/output pointers stable",
        "actual": pointer_stability,
        "exact": pointer_stability_exact,
    }
    failed = [name for name, check in checks.items() if not check["exact"]]
    return {"ok": not failed, "failed": failed, "checks": checks}


def _tail_is_bit_zero(
    planes: abi.CompactSummaryPlanes,
    *,
    generation_count: int = GENERATION_COUNT,
) -> bool:
    generation_count = _validate_generation_count(generation_count)
    values = planes.as_mapping()
    return all(
        not bool(np.any(array[generation_count:].view(np.uint8)))
        for array in values.values()
    )


def _callable_provenance(value: Any) -> str:
    module = str(getattr(value, "__module__", type(value).__module__))
    name = str(
        getattr(value, "__qualname__", getattr(value, "__name__", type(value).__qualname__))
    )
    return f"{module}.{name}"


def _clone_lineage_state(state: Any) -> LineageMetricsState:
    if not isinstance(state, LineageMetricsState):
        raise TypeError("CPU compact authority requires LineageMetricsState")
    return LineageMetricsState.from_dict(copy.deepcopy(state.to_dict()))


def _lineage_state_after_authority_input(inputs: Any) -> LineageMetricsState:
    if not isinstance(inputs, CompactSummaryAuthorityInputs):
        raise TypeError(
            "default CPU lineage advance requires CompactSummaryAuthorityInputs"
        )
    boundaries = inputs.run_result.get("boundaries")
    if not isinstance(boundaries, Sequence) or isinstance(
        boundaries, (str, bytes, bytearray)
    ):
        raise ValueError("CPU run_many result boundaries are unavailable")
    if inputs.boundary_index >= len(boundaries):
        raise ValueError("CPU compact authority boundary index is out of range")
    boundary = boundaries[inputs.boundary_index]
    if not isinstance(boundary, Mapping):
        raise TypeError("CPU compact authority boundary must be a mapping")
    generation = boundary.get("completed_generation")
    if type(generation) is not int:
        raise TypeError("CPU compact authority generation must be an int")
    overlay = inputs.public_metrics_overlay
    values = overlay.get("values") if isinstance(overlay, Mapping) else None
    lineage_payload = (
        values.get(NATIVE_LINEAGE_METRICS_INPUT_KEY)
        if isinstance(values, Mapping)
        else None
    )
    if not isinstance(lineage_payload, Mapping):
        raise ValueError("CPU public overlay lacks lineage metrics input authority")
    agents = native_lineage_agents_from_input(
        lineage_payload,
        expected_generation=generation,
    )
    _summary, _rows, _events, state_after = compute_lineage_snapshot(
        generation=generation,
        agents=agents,
        sim_cfg=inputs.sim_cfg,
        state=_clone_lineage_state(inputs.lineage_state_before),
    )
    return _clone_lineage_state(state_after)


def _validated_cpu_run_many_result(
    value: Any,
    *,
    generation_count: int = GENERATION_COUNT,
) -> Mapping[str, Any]:
    generation_count = _validate_generation_count(generation_count)
    execution_call = f"run_many({generation_count})"
    if not isinstance(value, Mapping):
        raise TypeError(f"CPU {execution_call} must return a mapping")
    exact_scalars = {
        "schema_version": "eiel_native_architecture_b_run_result_v1",
        "terminal_status": "run_complete",
        "executed": True,
        "native_owner": True,
        "hidden_fallback": False,
        "runtime_jit": False,
        "start_generation": 0,
        "requested_generations": generation_count,
        "completed_generations": generation_count,
        "controller_invocations": 1,
        "python_generation_dispatches": 0,
        "native_generation_transactions": generation_count,
        "turnover_commits": generation_count,
    }
    for field, expected in exact_scalars.items():
        if value.get(field) != expected:
            raise ValueError(
                f"CPU {execution_call} {field} must be {expected!r}; "
                f"got {value.get(field)!r}"
            )
    boundaries = value.get("boundaries")
    if not isinstance(boundaries, Sequence) or isinstance(
        boundaries, (str, bytes, bytearray)
    ) or len(boundaries) != generation_count:
        raise ValueError(
            f"CPU {execution_call} must expose exactly {generation_count} boundaries"
        )
    for generation, boundary in enumerate(boundaries):
        if not isinstance(boundary, Mapping):
            raise TypeError(f"CPU boundary {generation} must be a mapping")
        generation_result = boundary.get("generation_result")
        if not isinstance(generation_result, Mapping):
            raise TypeError(
                f"CPU boundary {generation} generation_result must be a mapping"
            )
        turnover_result = boundary.get("turnover_result")
        if not isinstance(turnover_result, Mapping):
            raise TypeError(
                f"CPU boundary {generation} turnover_result must be a mapping"
            )
        exact_boundary = {
            "completed_generation": generation,
            "next_generation": generation + 1,
        }
        for field, expected in exact_boundary.items():
            if boundary.get(field) != expected:
                raise ValueError(
                    f"CPU boundary {generation} {field} must be {expected}"
                )
        if generation_result.get("generation_index") != generation:
            raise ValueError(
                f"CPU boundary {generation} generation_result index is not exact"
            )
        for label, receipt in (
            ("generation_result", generation_result),
            ("turnover_result", turnover_result),
        ):
            for field, expected in {
                "executed": True,
                "hidden_fallback": False,
                "native_owner": True,
            }.items():
                if receipt.get(field) is not expected:
                    raise ValueError(
                        f"CPU boundary {generation} {label}.{field} "
                        f"must be {expected}"
                    )
    final_checkpoint = value.get("final_checkpoint_projection")
    if not isinstance(final_checkpoint, Mapping):
        raise TypeError(
            f"CPU {execution_call} final checkpoint projection is unavailable"
        )
    final_boundary_checkpoint = boundaries[-1].get(
        "post_turnover_checkpoint_projection"
    )
    if not isinstance(final_boundary_checkpoint, Mapping):
        raise TypeError(
            f"CPU generation {generation_count - 1} post-turnover checkpoint "
            "is unavailable"
        )
    if _fingerprint(final_checkpoint) != _fingerprint(final_boundary_checkpoint):
        raise ValueError(
            "CPU final checkpoint contradicts generation "
            f"{generation_count - 1} post-turnover boundary"
        )
    return value


def _execute_cpu_oracle_once(
    cpu_runner: Any,
    *,
    authority_input_builder: CpuAuthorityInputBuilder,
    initial_lineage_state: LineageMetricsState,
    authority_adapter: AuthorityAdapter = assemble_compact_summary_source,
    row_builder: RowBuilder = build_compact_summary_row,
    lineage_state_advance: LineageStateAdvance = _lineage_state_after_authority_input,
    generation_count: int = GENERATION_COUNT,
) -> FlatG2CpuOracleRun:
    """Execute one independent accepted bounded CPU compact producer."""

    generation_count = _validate_generation_count(generation_count)
    run_many = getattr(cpu_runner, "run_many", None)
    if not callable(run_many):
        raise TypeError("CPU oracle runner must expose run_many")
    run_result = _validated_cpu_run_many_result(
        run_many(generation_count),
        generation_count=generation_count,
    )
    run_result_before = _fingerprint(run_result)
    state = _clone_lineage_state(initial_lineage_state)
    initial_state_sha256 = _fingerprint(state)
    rows: list[dict[str, Any]] = []
    lineage_flow: list[dict[str, Any]] = []
    for generation in range(generation_count):
        state_before = _clone_lineage_state(state)
        authority_input = authority_input_builder(
            run_result,
            generation,
            state_before,
        )
        retained_state_before = getattr(
            authority_input, "lineage_state_before", None
        )
        if not isinstance(retained_state_before, LineageMetricsState):
            raise TypeError(
                f"CPU authority input {generation} must retain LineageMetricsState"
            )
        if _fingerprint(retained_state_before) != _fingerprint(state_before):
            raise ValueError(
                f"CPU authority input {generation} did not retain the supplied lineage state"
            )
        authority_before = _fingerprint(authority_input)
        row = dict(row_builder(authority_adapter(authority_input)))
        authority_after = _fingerprint(authority_input)
        if authority_before != authority_after:
            raise ValueError(
                f"CPU compact authority adapter mutated generation {generation} inputs"
            )
        row_generation = (
            row.get("generation_end", {}).get("generation")
            if isinstance(row.get("generation_end"), Mapping)
            else None
        )
        if row_generation != generation:
            raise ValueError(
                "CPU compact rows must cover exactly generations "
                f"{list(range(generation_count))}; "
                f"generation {generation} produced {row_generation!r}"
            )
        if row.get("world") != "flat":
            raise ValueError("CPU compact oracle is restricted to Flat rows")
        projection.canonical_compact_summary_bytes(row)
        state = lineage_state_advance(authority_input)
        if not isinstance(state, LineageMetricsState):
            raise TypeError("CPU lineage advance must return LineageMetricsState")
        lineage_flow.append(
            {
                "generation": generation,
                "state_before_sha256": _fingerprint(state_before),
                "state_after_sha256": _fingerprint(state),
            }
        )
        rows.append(row)
    run_result_after = _fingerprint(run_result)
    if run_result_before != run_result_after:
        raise ValueError(
            f"CPU compact projection mutated run_many({generation_count}) authority"
        )
    canonical_sha256 = [
        hashlib.sha256(projection.canonical_compact_summary_bytes(row)).hexdigest()
        for row in rows
    ]
    return FlatG2CpuOracleRun(
        canonical_rows=tuple(rows),
        final_checkpoint_projection=run_result["final_checkpoint_projection"],
        lineage_state_after=_clone_lineage_state(state),
        provenance={
            "authority_kind": "independent_cpu_oracle",
            "execution_api": "accepted_cpu_run_many",
            "execution_call": f"run_many({generation_count})",
            "generation_sequence": list(range(generation_count)),
            "candidate_rows_consumed": False,
            "candidate_planes_consumed": False,
            "authority_input_builder": _callable_provenance(
                authority_input_builder
            ),
            "authority_adapter": _callable_provenance(authority_adapter),
            "row_builder": _callable_provenance(row_builder),
            "lineage_state_advance": _callable_provenance(
                lineage_state_advance
            ),
            "lineage_initial_state_sha256": initial_state_sha256,
            "lineage_state_flow": lineage_flow,
            "canonical_row_sha256": canonical_sha256,
            "run_many_result_before_sha256": run_result_before,
            "run_many_result_after_sha256": run_result_after,
            "run_many_result_read_only": run_result_before == run_result_after,
            "final_checkpoint_source": (
                "cpu_run_many_result.final_checkpoint_projection"
            ),
        },
    )


def _expected_rows(
    inputs: Sequence[Any],
    *,
    authority_adapter: AuthorityAdapter,
    row_builder: RowBuilder,
    generation_count: int = GENERATION_COUNT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generation_count = _validate_generation_count(generation_count)
    if isinstance(inputs, (str, bytes, bytearray)) or len(inputs) != generation_count:
        raise ValueError(
            f"CPU authority inputs must contain exactly {generation_count} rows"
        )
    before = _fingerprint(inputs)
    rows = [dict(row_builder(authority_adapter(item))) for item in inputs]
    after = _fingerprint(inputs)
    generations = [row["generation_end"]["generation"] for row in rows]
    sequence_exact = generations == list(range(generation_count))
    worlds = [row.get("world") for row in rows]
    return rows, {
        "read_only": before == after,
        "before_sha256": before,
        "after_sha256": after,
        "generation_sequence": generations,
        "generation_sequence_exact": sequence_exact,
        "worlds": worlds,
        "flat_exact": worlds == ["flat"] * generation_count,
    }


def _plane_and_row_verdict(
    actual: abi.CompactSummaryPlanes,
    expected_rows: Sequence[Mapping[str, Any]],
    *,
    generation_count: int = GENERATION_COUNT,
) -> dict[str, Any]:
    generation_count = _validate_generation_count(generation_count)
    if isinstance(expected_rows, (str, bytes, bytearray)) or len(
        expected_rows
    ) != generation_count:
        raise ValueError(
            f"canonical_rows must contain exactly {generation_count} rows"
        )
    generations, cells = abi.validate_compact_summary_planes(actual)
    if generations < generation_count or cells != BATCH_CELLS:
        raise ValueError(
            "native compact planes must be "
            f"[at least {generation_count} generations, exactly 1 cell]"
        )
    expected = projection.encode_compact_summary_matrix(
        [[expected_rows[index]] for index in range(generation_count)]
    )
    actual_values = actual.as_mapping()
    expected_values = expected.as_mapping()
    plane_checks = {}
    for name in PLANE_NAMES:
        actual_prefix = np.ascontiguousarray(actual_values[name][:generation_count])
        expected_prefix = expected_values[name]
        plane_checks[name] = {
            "exact": (
                actual_prefix.dtype == expected_prefix.dtype
                and actual_prefix.shape == expected_prefix.shape
                and actual_prefix.tobytes(order="C")
                == expected_prefix.tobytes(order="C")
            ),
            "dtype": str(actual_prefix.dtype),
            "shape": list(actual_prefix.shape),
        }

    projected = [
        projection.project_compact_summary_row(actual, generation=index, cell=0)
        for index in range(generation_count)
    ]
    semantic_checks = [
        compare_exact_rows(expected_rows[index], projected[index])
        for index in range(generation_count)
    ]
    canonical_checks = [
        projection.canonical_compact_summary_bytes(expected_rows[index])
        == projection.canonical_compact_summary_bytes(projected[index])
        for index in range(generation_count)
    ]
    defined_masks = []
    digest_checks = []
    for generation in range(generation_count):
        expected_mask = int(expected.defined_mask[generation, 0, 0])
        actual_mask = int(actual.defined_mask[generation, 0, 0])
        defined_masks.append(
            {
                "generation_index": generation,
                "expected": expected_mask,
                "actual": actual_mask,
                "exact": expected_mask == actual_mask,
            }
        )
        digest_checks.append(
            {
                field: bool(
                    np.array_equal(
                        expected.digest_values[
                            generation, 0, abi.DIGEST_FIELD_INDEX[field]
                        ],
                        actual.digest_values[
                            generation, 0, abi.DIGEST_FIELD_INDEX[field]
                        ],
                    )
                )
                for field in abi.DIGEST_FIELDS
            }
        )

    validity = [int(value) for value in actual.validity[:, 0].tolist()]
    validity_exact = validity[:generation_count] == [1] * generation_count and all(
        value == 0 for value in validity[generation_count:]
    )
    tail_exact = _tail_is_bit_zero(actual, generation_count=generation_count)
    checks = {
        "all_plane_prefix_bytes_exact": all(
            check["exact"] for check in plane_checks.values()
        ),
        "all_semantic_rows_exact": all(
            bool(check["exact"]) for check in semantic_checks
        ),
        "all_canonical_row_bytes_exact": all(canonical_checks),
        "all_defined_masks_exact": all(check["exact"] for check in defined_masks),
        "all_digest_fields_exact": all(
            all(fields.values()) for fields in digest_checks
        ),
        "validity_exact": validity_exact,
        "tail_bit_zero_noop": tail_exact,
    }
    first_mismatch = next(
        (
            check["first_mismatch"]
            for check in semantic_checks
            if not bool(check["exact"])
        ),
        None,
    )
    if first_mismatch is None and not all(checks.values()):
        first_mismatch = next(name for name, exact in checks.items() if not exact)
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "first_mismatch": first_mismatch,
        "shape": {"generation_capacity": generations, "cells": cells},
        "validity": validity,
        "planes": plane_checks,
        "semantic_rows": semantic_checks,
        "canonical_bytes_exact": canonical_checks,
        "defined_masks": defined_masks,
        "digest_fields": digest_checks,
        "digest_field_names": list(abi.DIGEST_FIELDS),
        "abi_field_counts": {
            "status": len(abi.STATUS_FIELDS),
            "integer": len(abi.INTEGER_FIELDS),
            "float": len(abi.FLOAT_FIELDS),
            "digest": len(abi.DIGEST_FIELDS),
            "text": len(abi.TEXT_FIELDS),
            "defined_mask_bits": len(abi.DEFINED_FIELDS),
        },
        "projected_rows": projected,
    }


def _checkpoint_verdict(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, Any]:
    expected_semantic = checkpoint_digest_manifest(
        expected, layer=DigestLayer.TRAJECTORY_STATE
    )
    actual_semantic = checkpoint_digest_manifest(
        actual, layer=DigestLayer.TRAJECTORY_STATE
    )
    semantic_exact = expected_semantic["digest"] == actual_semantic["digest"]
    semantic_mismatch = (
        None
        if semantic_exact
        else _projection_digest_first_difference(expected, actual)
    )
    storage = _canonical_checkpoint_storage_verdict(expected, actual)
    return {
        "ok": semantic_exact and bool(storage["ok"]),
        "first_mismatch": semantic_mismatch or storage.get("first_mismatch"),
        "authoritative_semantic": {
            "exact": semantic_exact,
            "expected_sha256": expected_semantic["digest"],
            "actual_sha256": actual_semantic["digest"],
            "field_count": expected_semantic["field_count"],
        },
        "canonical_checkpoint_storage": storage,
    }


def _cpu_oracle_provenance_verdict(
    run: FlatG2CpuOracleRun,
    *,
    generation_count: int = GENERATION_COUNT,
) -> dict[str, Any]:
    generation_count = _validate_generation_count(generation_count)
    expected_generations = list(range(generation_count))
    provenance = run.provenance
    flow = provenance.get("lineage_state_flow")
    flow_exact = (
        isinstance(flow, Sequence)
        and not isinstance(flow, (str, bytes, bytearray))
        and len(flow) == generation_count
        and all(isinstance(item, Mapping) for item in flow)
    )
    lineage_flow_exact = False
    if flow_exact:
        lineage_flow_exact = (
            [item.get("generation") for item in flow] == expected_generations
            and provenance.get("lineage_initial_state_sha256")
            == flow[0].get("state_before_sha256")
            and all(
                flow[index - 1].get("state_after_sha256")
                == flow[index].get("state_before_sha256")
                for index in range(1, generation_count)
            )
            and flow[-1].get("state_after_sha256")
            == _fingerprint(run.lineage_state_after)
        )
    row_generations = []
    row_worlds = []
    for row in run.canonical_rows:
        generation_end = row.get("generation_end")
        row_generations.append(
            generation_end.get("generation")
            if isinstance(generation_end, Mapping)
            else None
        )
        row_worlds.append(row.get("world"))
    canonical_sha256 = [
        hashlib.sha256(projection.canonical_compact_summary_bytes(row)).hexdigest()
        for row in run.canonical_rows
    ]
    callable_fields = (
        "authority_input_builder",
        "authority_adapter",
        "row_builder",
        "lineage_state_advance",
    )
    streaming = provenance.get("authority_kind") == (
        "independent_streaming_cpu_oracle"
    )
    checks = {
        "authority_kind": provenance.get("authority_kind")
        in {"independent_cpu_oracle", "independent_streaming_cpu_oracle"},
        "execution_api": provenance.get("execution_api")
        == (
            "accepted_cpu_run_with_persistence"
            if streaming
            else "accepted_cpu_run_many"
        ),
        "execution_call": provenance.get("execution_call")
        == (
            f"run_with_persistence({generation_count})"
            if streaming
            else f"run_many({generation_count})"
        ),
        "candidate_independent": (
            provenance.get("candidate_rows_consumed") is False
            and provenance.get("candidate_planes_consumed") is False
        ),
        "generation_sequence": (
            len(run.canonical_rows) == generation_count
            and provenance.get("generation_sequence") == expected_generations
            and row_generations == expected_generations
            and row_worlds == ["flat"] * generation_count
        ),
        "lineage_state_flow": lineage_flow_exact,
        "canonical_row_sha256": provenance.get("canonical_row_sha256")
        == canonical_sha256,
        "run_many_result_read_only": (
            provenance.get("streamed_authority_inputs_read_only") is True
            if streaming
            else (
                provenance.get("run_many_result_read_only") is True
                and provenance.get("run_many_result_before_sha256")
                == provenance.get("run_many_result_after_sha256")
            )
        ),
        "final_checkpoint_source": provenance.get("final_checkpoint_source")
        == (
            "cpu_terminal_only_result.final_checkpoint_projection"
            if streaming
            else "cpu_run_many_result.final_checkpoint_projection"
        ),
        "retention_contract": (
            provenance.get("result_retention") == "terminal_only"
            and provenance.get("emitted_boundary_count") == generation_count
            and provenance.get("retained_boundary_count") == 1
            if streaming
            else True
        ),
        "exact_code_sources": all(
            type(provenance.get(field)) is str and bool(provenance.get(field))
            for field in callable_fields
        ),
    }
    failed = [name for name, exact in checks.items() if not exact]
    return {
        "ok": not failed,
        "checks": checks,
        "failed": failed,
        "first_mismatch": (
            None
            if not failed
            else {
                "authority": "cpu_oracle",
                "path": f"$.provenance.{failed[0]}",
                "reason": "source_provenance_mismatch",
            }
        ),
    }


def _cpu_oracle_repeat_verdict(
    first: FlatG2CpuOracleRun,
    repeat: FlatG2CpuOracleRun,
    *,
    generation_count: int = GENERATION_COUNT,
) -> dict[str, Any]:
    generation_count = _validate_generation_count(generation_count)
    first_provenance = _cpu_oracle_provenance_verdict(
        first,
        generation_count=generation_count,
    )
    repeat_provenance = _cpu_oracle_provenance_verdict(
        repeat,
        generation_count=generation_count,
    )
    row_count_exact = (
        len(first.canonical_rows) == generation_count
        and len(repeat.canonical_rows) == generation_count
    )
    row_checks = (
        [
            compare_exact_rows(
                first.canonical_rows[index], repeat.canonical_rows[index]
            )
            for index in range(generation_count)
        ]
        if row_count_exact
        else []
    )
    canonical_checks = (
        [
            projection.canonical_compact_summary_bytes(first.canonical_rows[index])
            == projection.canonical_compact_summary_bytes(
                repeat.canonical_rows[index]
            )
            for index in range(generation_count)
        ]
        if row_count_exact
        else []
    )
    checkpoint = _checkpoint_verdict(
        first.final_checkpoint_projection,
        repeat.final_checkpoint_projection,
    )
    lineage_state = _snapshot_verdict(
        first.lineage_state_after,
        repeat.lineage_state_after,
    )
    provenance_checks = {
        "first_exact": first_provenance["ok"],
        "repeat_exact": repeat_provenance["ok"],
    }
    checks = {
        "canonical_row_count_exact": row_count_exact,
        "all_semantic_rows_exact": all(
            bool(check["exact"]) for check in row_checks
        ) and row_count_exact,
        "all_canonical_row_bytes_exact": all(canonical_checks) and row_count_exact,
        "final_checkpoint_exact": checkpoint["ok"],
        "lineage_state_after_exact": lineage_state["exact"],
        "source_provenance_exact": all(provenance_checks.values()),
    }
    first_mismatch: Any = (
        None
        if row_count_exact
        else {
            "authority": "cpu_oracle_repeat",
            "path": "$.canonical_rows",
            "reason": "row_count_mismatch",
            "expected": generation_count,
            "actual": {
                "first": len(first.canonical_rows),
                "repeat": len(repeat.canonical_rows),
            },
        }
    )
    for generation, check in enumerate(row_checks):
        if not bool(check["exact"]):
            first_mismatch = {
                "authority": "cpu_oracle_repeat",
                "generation_index": generation,
                **(check.get("first_mismatch") or {
                    "path": "$",
                    "reason": "canonical_bytes_mismatch",
                }),
            }
            break
    if first_mismatch is None and not checkpoint["ok"]:
        first_mismatch = checkpoint.get("first_mismatch") or {
            "authority": "cpu_oracle_repeat",
            "path": "$.final_checkpoint_projection",
            "reason": "checkpoint_mismatch",
        }
    if first_mismatch is None and not lineage_state["exact"]:
        first_mismatch = {
            "authority": "cpu_oracle_repeat",
            "path": "$.lineage_state_after",
            "reason": "value_mismatch",
        }
    if first_mismatch is None and not all(provenance_checks.values()):
        first_mismatch = (
            first_provenance.get("first_mismatch")
            or repeat_provenance.get("first_mismatch")
        )
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "first_mismatch": first_mismatch,
        "semantic_rows": row_checks,
        "canonical_bytes_exact": canonical_checks,
        "final_checkpoint": checkpoint,
        "lineage_state_after": lineage_state,
        "provenance_checks": provenance_checks,
        "first_provenance": first_provenance,
        "repeat_provenance": repeat_provenance,
    }


def _receipt_first_mismatch(
    verdict: Mapping[str, Any], *, label: str
) -> dict[str, Any] | None:
    failed = verdict.get("failed")
    checks = verdict.get("checks")
    if not isinstance(failed, Sequence) or not failed or not isinstance(checks, Mapping):
        return None
    field = failed[0]
    check = checks.get(field)
    return {
        "path": f"$.{label}.receipt.{field}",
        "reason": "receipt_contract_mismatch",
        "expected": check.get("expected") if isinstance(check, Mapping) else None,
        "actual": check.get("actual") if isinstance(check, Mapping) else None,
    }


def _repeat_plane_verdict(
    first: abi.CompactSummaryPlanes, repeat: abi.CompactSummaryPlanes
) -> dict[str, Any]:
    first_values = first.as_mapping()
    repeat_values = repeat.as_mapping()
    checks = {}
    for name in PLANE_NAMES:
        left = first_values[name]
        right = repeat_values[name]
        checks[name] = (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C")
        )
    return {"ok": all(checks.values()), "planes": checks}


def _missing_run_inputs(run: FlatG2RetainedRun | None, *, label: str) -> list[str]:
    if run is None:
        return [f"{label}.retained_run"]
    if not isinstance(run, FlatG2RetainedRun):
        return [f"{label}.retained_run_type"]
    missing = []
    for field in dataclasses.fields(run):
        if getattr(run, field.name) is None:
            missing.append(f"{label}.{field.name}")
    return missing


def qualify_retained_runs(
    first: FlatG2RetainedRun,
    repeat: FlatG2RetainedRun,
    *,
    cpu_oracle: FlatG2CpuOracleRun,
    cpu_oracle_repeat: FlatG2CpuOracleRun,
    expected_final_checkpoint: Mapping[str, Any] | None = None,
    generation_count: int = GENERATION_COUNT,
) -> dict[str, Any]:
    """Evaluate native rows only against separately executed CPU authority."""

    try:
        generation_count = _validate_generation_count(generation_count)
    except Exception as exc:
        return _rejected_result(
            decision="compact_generation_count_unsupported",
            reason=f"{type(exc).__name__}: {exc}",
            first_mismatch={"type": type(exc).__name__, "message": str(exc)},
            producer_calls=0,
            generation_count=(
                generation_count
                if type(generation_count) is int
                else GENERATION_COUNT
            ),
        )
    generation_label = _generation_label(generation_count)
    expected_generations = list(range(generation_count))
    missing = [
        *_missing_run_inputs(first, label="first"),
        *_missing_run_inputs(repeat, label="repeat"),
    ]
    if not isinstance(cpu_oracle, FlatG2CpuOracleRun):
        missing.append("cpu_oracle.first")
    if not isinstance(cpu_oracle_repeat, FlatG2CpuOracleRun):
        missing.append("cpu_oracle.repeat")
    if missing:
        return producer_pending_result(
            missing=missing,
            producer_calls=2,
            cpu_oracle_calls=2,
            generation_count=generation_count,
        )
    try:
        first_planes = _as_host_planes(first.compact_summary_planes)
        repeat_planes = _as_host_planes(repeat.compact_summary_planes)
        expected_rows = cpu_oracle.canonical_rows
        cpu_repeat = _cpu_oracle_repeat_verdict(
            cpu_oracle,
            cpu_oracle_repeat,
            generation_count=generation_count,
        )
        first_rows_verdict = _plane_and_row_verdict(
            first_planes,
            expected_rows,
            generation_count=generation_count,
        )
        repeat_rows_verdict = _plane_and_row_verdict(
            repeat_planes,
            expected_rows,
            generation_count=generation_count,
        )
        first_checkpoint = _checkpoint_verdict(
            cpu_oracle.final_checkpoint_projection,
            first.final_checkpoint_projection,
        )
        repeat_checkpoint = _checkpoint_verdict(
            cpu_oracle.final_checkpoint_projection,
            repeat.final_checkpoint_projection,
        )
        stored_checkpoint = (
            {"ok": True, "first_mismatch": None, "not_requested": True}
            if expected_final_checkpoint is None
            else _checkpoint_verdict(
                expected_final_checkpoint,
                cpu_oracle.final_checkpoint_projection,
            )
        )
        checkpoint_repeat = _checkpoint_verdict(
            first.final_checkpoint_projection, repeat.final_checkpoint_projection
        )
        plane_repeat = _repeat_plane_verdict(first_planes, repeat_planes)
        first_receipt = _receipt_verdict(
            first.receipt,
            generation_count=generation_count,
        )
        repeat_receipt = _receipt_verdict(
            repeat.receipt,
            generation_count=generation_count,
        )
        first_state = _snapshot_verdict(
            first.summary_state_before, first.summary_state_after
        )
        repeat_state = _snapshot_verdict(
            repeat.summary_state_before, repeat.summary_state_after
        )
        first_rng = _snapshot_verdict(first.rng_state_before, first.rng_state_after)
        repeat_rng = _snapshot_verdict(
            repeat.rng_state_before, repeat.rng_state_after
        )
        row_repeat = [
            projection.canonical_compact_summary_bytes(
                first_rows_verdict["projected_rows"][index]
            )
            == projection.canonical_compact_summary_bytes(
                repeat_rows_verdict["projected_rows"][index]
            )
            for index in range(generation_count)
        ]
    except Exception as exc:
        return _rejected_result(
            decision=f"flat_{generation_label}_compact_evidence_invalid",
            reason=f"{type(exc).__name__}: {exc}",
            first_mismatch={"type": type(exc).__name__, "message": str(exc)},
            cpu_oracle_calls=2,
            generation_count=generation_count,
        )

    checks = {
        "independent_cpu_oracle_repeat_exact": cpu_repeat["ok"],
        "independent_cpu_source_provenance": (
            cpu_repeat["first_provenance"]["ok"]
            and cpu_oracle.provenance.get("candidate_rows_consumed") is False
            and cpu_oracle.provenance.get("candidate_planes_consumed") is False
        ),
        "cpu_generation_sequence_exact": (
            cpu_oracle.provenance.get("generation_sequence")
            == expected_generations
        ),
        "cpu_authority_read_only": (
            cpu_oracle.provenance.get("streamed_authority_inputs_read_only") is True
            if cpu_oracle.provenance.get("authority_kind")
            == "independent_streaming_cpu_oracle"
            else cpu_oracle.provenance.get("run_many_result_read_only") is True
        ),
        "stored_cpu_checkpoint_exact": stored_checkpoint["ok"],
        "first_rows_exact": first_rows_verdict["ok"],
        "repeat_rows_exact": repeat_rows_verdict["ok"],
        "first_final_checkpoint_exact": first_checkpoint["ok"],
        "repeat_final_checkpoint_exact": repeat_checkpoint["ok"],
        "same_process_checkpoint_repeat": checkpoint_repeat["ok"],
        "same_process_plane_repeat": plane_repeat["ok"],
        "same_process_row_repeat": all(row_repeat),
        "same_process_identity": (
            first.process_identity is not None
            and first.process_identity == repeat.process_identity
        ),
        "first_summary_state_read_only": first_state["exact"],
        "repeat_summary_state_read_only": repeat_state["exact"],
        "first_summary_rng_non_consuming": first_rng["exact"],
        "repeat_summary_rng_non_consuming": repeat_rng["exact"],
        "first_execution_contract": first_receipt["ok"],
        "repeat_execution_contract": repeat_receipt["ok"],
    }
    failed = [name for name, exact in checks.items() if not exact]
    first_mismatch = (
        cpu_repeat.get("first_mismatch")
        or stored_checkpoint.get("first_mismatch")
        or first_rows_verdict.get("first_mismatch")
        or repeat_rows_verdict.get("first_mismatch")
        or first_checkpoint.get("first_mismatch")
        or repeat_checkpoint.get("first_mismatch")
        or checkpoint_repeat.get("first_mismatch")
        or _receipt_first_mismatch(first_receipt, label="first_run")
        or _receipt_first_mismatch(repeat_receipt, label="repeat_run")
        or (failed[0] if failed else None)
    )
    details = {
        "checks": checks,
        "failed_checks": failed,
        "expected_authority": {
            "source": "independent_cpu_run_many",
            "candidate_derived": False,
            "provenance": dict(cpu_oracle.provenance),
            "repeat": cpu_repeat,
            "stored_checkpoint": stored_checkpoint,
        },
        "first_run": {
            "rows": {
                key: value
                for key, value in first_rows_verdict.items()
                if key != "projected_rows"
            },
            "authority": {
                "expected_source": "independent_cpu_run_many",
                "candidate_authority_inputs_consumed": False,
            },
            "receipt": first_receipt,
            "summary_state": first_state,
            "summary_rng": first_rng,
            "final_checkpoint": first_checkpoint,
        },
        "repeat_run": {
            "rows": {
                key: value
                for key, value in repeat_rows_verdict.items()
                if key != "projected_rows"
            },
            "authority": {
                "expected_source": "independent_cpu_run_many",
                "candidate_authority_inputs_consumed": False,
            },
            "receipt": repeat_receipt,
            "summary_state": repeat_state,
            "summary_rng": repeat_rng,
            "final_checkpoint": repeat_checkpoint,
        },
        "same_process_repeat": {
            "process_identity": first.process_identity,
            "planes": plane_repeat,
            "rows": row_repeat,
            "final_checkpoint": checkpoint_repeat,
        },
    }
    if failed:
        return _rejected_result(
            decision=f"flat_{generation_label}_compact_abi1_rejected",
            reason="one or more exactness or execution-contract checks failed",
            first_mismatch=first_mismatch,
            details=details,
            cpu_oracle_calls=2,
            generation_count=generation_count,
        )
    return {
        "schema_version": _schema_version(generation_count),
        "gate_id": _gate_id(generation_count),
        "mode": "live",
        "status": (
            f"exact_flat_{generation_label}_b1_compact_rows_capability_false"
        ),
        "ok": True,
        "decision": (
            f"flat_{generation_label}_compact_abi1_gate_closed_capability_false"
        ),
        "bounded_shape": {
            "world": "flat",
            "generations": generation_count,
            "batch_cells": BATCH_CELLS,
            "ticks_per_generation": TICKS_PER_GENERATION,
        },
        "producer_calls": 2,
        "cpu_oracle_calls": 2,
        "expected_authority": {
            "source": "independent_cpu_run_many",
            "candidate_derived": False,
            "generation_sequence": expected_generations,
        },
        "checks": checks,
        "failed_checks": [],
        "first_mismatch": None,
        "details": details,
        "canonical_native_capability": False,
        "capability_promoted": False,
        "claims": _claims(
            generation_count=generation_count,
            execution_performed=True,
            exact=True,
        ),
    }


def execute_live_gate(
    *,
    capability: Mapping[str, Any] | None = None,
    producer: NativeProducer | None = None,
    cpu_oracle_producer: CpuOracleProducer | None = None,
    expected_final_checkpoint: Mapping[str, Any] | None = None,
    generation_count: int = GENERATION_COUNT,
) -> dict[str, Any]:
    """Repeat independent CPU authority and native candidate in one process."""

    try:
        generation_count = _validate_generation_count(generation_count)
    except Exception as exc:
        return _rejected_result(
            decision="compact_generation_count_unsupported",
            reason=f"{type(exc).__name__}: {exc}",
            first_mismatch={"type": type(exc).__name__, "message": str(exc)},
            producer_calls=0,
            generation_count=(
                generation_count
                if type(generation_count) is int
                else GENERATION_COUNT
            ),
        )
    generation_label = _generation_label(generation_count)
    missing = []
    if not isinstance(capability, Mapping):
        missing.append("native_capability")
    if producer is None:
        missing.append("same_process_native_producer")
    if cpu_oracle_producer is None:
        missing.append("independent_cpu_run_many_oracle")
    producer_status = (
        capability.get("compact_summary_native_producer_status")
        if isinstance(capability, Mapping)
        else None
    )
    if (
        type(producer_status) is not str
        or not producer_status
        or producer_status in PENDING_PRODUCER_STATUSES
    ):
        missing.append("retained_compact_summary_rows")
    if missing:
        return producer_pending_result(
            missing=missing,
            generation_count=generation_count,
        )

    capability_violations = [
        field for field in CAPABILITY_FIELDS if capability.get(field) is not False
    ]
    if capability_violations:
        return _rejected_result(
            decision="compact_capability_boundary_violation",
            reason=(
                "the gate requires capability to remain explicitly false: "
                + ", ".join(capability_violations)
            ),
            producer_calls=0,
            generation_count=generation_count,
        )
    cpu_oracle_calls = 0
    try:
        cpu_oracle_calls += 1
        cpu_oracle = cpu_oracle_producer()
        if not isinstance(cpu_oracle, FlatG2CpuOracleRun):
            raise TypeError(
                "first CPU oracle producer call did not return FlatG2CpuOracleRun"
            )
        cpu_oracle_calls += 1
        cpu_oracle_repeat = cpu_oracle_producer()
        if not isinstance(cpu_oracle_repeat, FlatG2CpuOracleRun):
            raise TypeError(
                "repeat CPU oracle producer call did not return FlatG2CpuOracleRun"
            )
        cpu_repeat = _cpu_oracle_repeat_verdict(
            cpu_oracle,
            cpu_oracle_repeat,
            generation_count=generation_count,
        )
    except Exception as exc:
        return _rejected_result(
            decision=f"flat_{generation_label}_cpu_oracle_error",
            reason=f"{type(exc).__name__}: {exc}",
            first_mismatch={"type": type(exc).__name__, "message": str(exc)},
            producer_calls=0,
            cpu_oracle_calls=cpu_oracle_calls,
            generation_count=generation_count,
        )
    if not cpu_repeat["ok"]:
        return _rejected_result(
            decision=f"flat_{generation_label}_cpu_oracle_repeat_rejected",
            reason="independent CPU compact producer did not repeat byte-exactly",
            first_mismatch=cpu_repeat["first_mismatch"],
            details={
                "checks": {
                    "independent_cpu_oracle_repeat_exact": False,
                },
                "failed_checks": ["independent_cpu_oracle_repeat_exact"],
                "expected_authority": {
                    "source": "independent_cpu_run_many",
                    "candidate_derived": False,
                    "first_provenance": dict(cpu_oracle.provenance),
                    "repeat_provenance": dict(cpu_oracle_repeat.provenance),
                    "repeat": cpu_repeat,
                },
            },
            producer_calls=0,
            cpu_oracle_calls=cpu_oracle_calls,
            generation_count=generation_count,
        )
    producer_calls = 0
    try:
        producer_calls += 1
        first = producer()
        first_missing = _missing_run_inputs(first, label="first")
        if first_missing:
            return producer_pending_result(
                missing=first_missing,
                producer_calls=1,
                cpu_oracle_calls=cpu_oracle_calls,
                reason="first native dispatch did not retain complete compact sources/rows",
                generation_count=generation_count,
            )
        producer_calls += 1
        repeat = producer()
        repeat_missing = _missing_run_inputs(repeat, label="repeat")
        if repeat_missing:
            return producer_pending_result(
                missing=repeat_missing,
                producer_calls=2,
                cpu_oracle_calls=cpu_oracle_calls,
                reason="repeat native dispatch did not retain complete compact sources/rows",
                generation_count=generation_count,
            )
    except Exception as exc:
        return _rejected_result(
            decision=f"flat_{generation_label}_compact_producer_error",
            reason=f"{type(exc).__name__}: {exc}",
            first_mismatch={"type": type(exc).__name__, "message": str(exc)},
            producer_calls=producer_calls,
            cpu_oracle_calls=cpu_oracle_calls,
            generation_count=generation_count,
        )
    return qualify_retained_runs(
        first,
        repeat,
        cpu_oracle=cpu_oracle,
        cpu_oracle_repeat=cpu_oracle_repeat,
        expected_final_checkpoint=expected_final_checkpoint,
        generation_count=generation_count,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--generation-count",
        type=int,
        choices=sorted(SUPPORTED_GENERATION_COUNTS),
        default=GENERATION_COUNT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = execute_live_gate(generation_count=args.generation_count)
    if args.output is not None:
        _write_json(args.output.expanduser().resolve(), payload)
    print(
        json.dumps(
            {
                "ok": payload["ok"],
                "decision": payload["decision"],
                "reason": payload.get("reason"),
                "output": str(args.output.expanduser().resolve()) if args.output else None,
            },
            sort_keys=True,
        )
    )
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
