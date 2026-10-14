#!/usr/bin/env python3
"""Byte-exact probe for the native resident event-window drain boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import torch


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "engine"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from engine.perf.cuda_architecture_b_boundary import (  # noqa: E402
    EVENT_COLUMN_DTYPES,
    EVENT_COLUMN_NAMES,
)
from engine.perf.cuda_architecture_b_resident_window_runtime_v1 import (  # noqa: E402
    drain_event_window,
)


SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_resident_event_archive_v1_probe_receipt_v1"
)
MODULE_NAME: Final[str] = "eiel_cuda_architecture_b_flattened_controller_v1"
ENTRYPOINT: Final[str] = "eiel_cuda_architecture_b_drain_event_window_once_v1"
ACTIVE_COUNTS: Final[tuple[int, ...]] = (2, 4)
# The isolated entrypoint owns a fresh archive. Nonzero continuation bases are
# exercised by the multi-window controller, which retains the archive between
# drains.
SEQUENCE_BASES: Final[tuple[int, ...]] = (0, 0)
PHYSICAL_CAPACITY: Final[int] = 6
FAIL_CLOSED_CASES: Final[tuple[str, ...]] = (
    "bad_sequence",
    "nonzero_status",
)


@dataclass(frozen=True)
class Options:
    extension: Path
    output: Path


@dataclass
class EventBank:
    columns: dict[str, torch.Tensor]
    active_count: torch.Tensor
    sequence_base: torch.Tensor
    status: torch.Tensor
    archives: list[list[dict[str, torch.Tensor]]]


@dataclass(frozen=True)
class TensorSnapshot:
    dtype: torch.dtype
    shape: tuple[int, ...]
    payload: bytes
    pointer: int


class ProbeMismatch(RuntimeError):
    """Raised when the native drain differs from the Python boundary."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the native resident event-window drain byte-for-byte "
            "with the Python drain_event_window boundary."
        )
    )
    parser.add_argument(
        "--extension",
        type=Path,
        required=True,
        help="Prebuilt immutable flattened-controller extension.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON receipt path.",
    )
    return parser


def _options_from_namespace(namespace: argparse.Namespace) -> Options:
    extension = namespace.extension.expanduser().resolve()
    output = namespace.output.expanduser().resolve()
    if extension.suffix != ".so" or not extension.is_file():
        raise ValueError("--extension must name an existing .so file")
    if output.suffix.lower() != ".json":
        raise ValueError("--output must end in .json")
    if output == extension:
        raise ValueError("--output must differ from --extension")
    return Options(extension=extension, output=output)


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError("receipt path must end in .json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_extension(extension: Path) -> ModuleType:
    if MODULE_NAME in sys.modules:
        raise RuntimeError(f"canonical extension name is already loaded: {MODULE_NAME}")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, extension)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct extension loader for {extension}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.__name__ != MODULE_NAME:
        raise RuntimeError("flattened controller was not loaded canonically")
    if not callable(getattr(module, ENTRYPOINT, None)):
        raise RuntimeError(f"extension is missing entrypoint {ENTRYPOINT}")
    return module


def _deterministic_values(
    dtype: torch.dtype,
    shape: tuple[int, ...],
    *,
    salt: int,
) -> torch.Tensor:
    count = 1
    for dimension in shape:
        count *= dimension
    ordinal = torch.arange(count, dtype=torch.int64).reshape(shape)
    if dtype.is_floating_point:
        return (ordinal.to(torch.float64) / 8.0 + salt + 0.25).to(dtype)
    if dtype == torch.int8:
        return ((ordinal + salt) % 101 - 50).to(dtype)
    return (ordinal + salt * 1000).to(dtype)


def _archive_prefixes(
    bases: Sequence[int],
) -> list[list[dict[str, torch.Tensor]]]:
    archives: list[list[dict[str, torch.Tensor]]] = []
    for cell, base in enumerate(bases):
        chunk = {
            name: _deterministic_values(
                EVENT_COLUMN_DTYPES[name],
                (base,),
                salt=200 + 17 * column_index + cell,
            )
            for column_index, name in enumerate(EVENT_COLUMN_NAMES)
        }
        chunk["typed_lifecycle_event_sequence"] = torch.arange(
            base,
            dtype=torch.int64,
        )
        archives.append([chunk] if base else [])
    return archives


def _build_synthetic_bank(
    device: str,
    *,
    fault: str | None = None,
) -> EventBank:
    if fault not in {None, *FAIL_CLOSED_CASES}:
        raise ValueError(f"unsupported synthetic fault: {fault}")
    cells = len(ACTIVE_COUNTS)
    columns = {
        name: _deterministic_values(
            EVENT_COLUMN_DTYPES[name],
            (cells, PHYSICAL_CAPACITY),
            salt=11 + 19 * column_index,
        )
        .to(device)
        .contiguous()
        for column_index, name in enumerate(EVENT_COLUMN_NAMES)
    }
    columns["typed_lifecycle_event_sequence"] = torch.stack(
        [
            torch.arange(
                base,
                base + PHYSICAL_CAPACITY,
                dtype=torch.int64,
            )
            for base in SEQUENCE_BASES
        ]
    ).to(device)
    active_count = torch.tensor(ACTIVE_COUNTS, dtype=torch.int64, device=device)
    sequence_base = torch.tensor(
        SEQUENCE_BASES,
        dtype=torch.int64,
        device=device,
    )
    status = torch.zeros(cells, dtype=torch.int32, device=device)
    if fault == "bad_sequence":
        columns["typed_lifecycle_event_sequence"][1, 1].add_(37)
    elif fault == "nonzero_status":
        status[1] = 9
    return EventBank(
        columns=columns,
        active_count=active_count,
        sequence_base=sequence_base,
        status=status,
        archives=_archive_prefixes(SEQUENCE_BASES),
    )


def _state_tensors(bank: EventBank) -> dict[str, torch.Tensor]:
    return {
        **bank.columns,
        "active_count": bank.active_count,
        "event_sequence_base": bank.sequence_base,
        "event_status": bank.status,
    }


def _tensor_bytes(tensor: torch.Tensor, *, label: str) -> bytes:
    if not isinstance(tensor, torch.Tensor):
        raise ProbeMismatch(f"{label} is not a Torch tensor")
    value = tensor.detach().to(device="cpu").contiguous()
    if value.requires_grad:
        raise ProbeMismatch(f"{label} unexpectedly requires gradients")
    return value.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")


def _snapshot(tensors: Mapping[str, torch.Tensor]) -> dict[str, TensorSnapshot]:
    return {
        name: TensorSnapshot(
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            payload=_tensor_bytes(tensor, label=name),
            pointer=int(tensor.data_ptr()),
        )
        for name, tensor in sorted(tensors.items())
    }


def _assert_snapshot_unchanged(
    before: Mapping[str, TensorSnapshot],
    tensors: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> int:
    if set(before) != set(tensors):
        raise ProbeMismatch(f"{label} tensor surface changed")
    for name, prior in before.items():
        tensor = tensors[name]
        if tensor.dtype != prior.dtype or tuple(tensor.shape) != prior.shape:
            raise ProbeMismatch(f"{label}.{name} metadata changed")
        if _tensor_bytes(tensor, label=f"{label}.{name}") != prior.payload:
            raise ProbeMismatch(f"{label}.{name} bytes changed")
        if int(tensor.data_ptr()) != prior.pointer:
            raise ProbeMismatch(f"{label}.{name} pointer changed")
    return len(before)


def _pointer_identity(
    before: Mapping[str, TensorSnapshot],
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, bool]:
    return {
        name: before[name].pointer == int(tensors[name].data_ptr())
        for name in sorted(before)
    }


def _update_bytes(digest: Any, label: str, payload: bytes) -> None:
    encoded_label = label.encode("utf-8")
    digest.update(len(encoded_label).to_bytes(8, "big"))
    digest.update(encoded_label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _update_tensor_digest(digest: Any, label: str, tensor: torch.Tensor) -> None:
    metadata = json.dumps(
        {
            "dtype": str(tensor.dtype),
            "label": label,
            "shape": list(tensor.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _update_bytes(digest, f"{label}.metadata", metadata)
    _update_bytes(
        digest,
        f"{label}.bytes",
        _tensor_bytes(tensor, label=label),
    )


def _record_tensor_pair(
    expected_digest: Any,
    actual_digest: Any,
    *,
    label: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    if expected.dtype != actual.dtype:
        raise ProbeMismatch(
            f"{label} dtype mismatch: expected {expected.dtype}, got {actual.dtype}"
        )
    if tuple(expected.shape) != tuple(actual.shape):
        raise ProbeMismatch(
            f"{label} shape mismatch: expected {list(expected.shape)}, "
            f"got {list(actual.shape)}"
        )
    if _tensor_bytes(expected, label=f"{label}.python") != _tensor_bytes(
        actual,
        label=f"{label}.native",
    ):
        raise ProbeMismatch(f"{label} tensor bytes differ")
    _update_tensor_digest(expected_digest, label, expected)
    _update_tensor_digest(actual_digest, label, actual)


def _record_value_pair(
    expected_digest: Any,
    actual_digest: Any,
    *,
    label: str,
    expected: Any,
    actual: Any,
) -> None:
    if expected != actual:
        raise ProbeMismatch(f"{label} mismatch: expected {expected!r}, got {actual!r}")
    expected_bytes = json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    actual_bytes = json.dumps(
        actual,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _update_bytes(expected_digest, label, expected_bytes)
    _update_bytes(actual_digest, label, actual_bytes)


def _sequence(value: object, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise ProbeMismatch(f"{label} must be a sequence")
    return value


def _native_chunks(
    result: Mapping[str, Any],
) -> tuple[tuple[torch.Tensor, ...], ...]:
    cells = _sequence(result.get("event_chunks"), label="event_chunks")
    if len(cells) != len(ACTIVE_COUNTS):
        raise ProbeMismatch("native event_chunks cell count differs")
    output: list[tuple[torch.Tensor, ...]] = []
    for cell, (cell_value, count) in enumerate(zip(cells, ACTIVE_COUNTS, strict=True)):
        chunks = _sequence(cell_value, label=f"event_chunks[{cell}]")
        expected_chunk_count = int(count > 0)
        if len(chunks) != expected_chunk_count:
            raise ProbeMismatch(
                f"event_chunks[{cell}] must contain {expected_chunk_count} chunk"
            )
        if not chunks:
            output.append(())
            continue
        columns = _sequence(chunks[0], label=f"event_chunks[{cell}][0]")
        if len(columns) != len(EVENT_COLUMN_NAMES):
            raise ProbeMismatch(f"event_chunks[{cell}][0] must contain 22 columns")
        typed_columns: list[torch.Tensor] = []
        for column_index, column in enumerate(columns):
            if not isinstance(column, torch.Tensor):
                raise ProbeMismatch(
                    f"event_chunks[{cell}][0][{column_index}] is not a tensor"
                )
            if column.device.type != "cpu" or not column.is_contiguous():
                raise ProbeMismatch(
                    f"event_chunks[{cell}][0][{column_index}] is not contiguous CPU"
                )
            typed_columns.append(column)
        output.append(tuple(typed_columns))
    return tuple(output)


def _python_pointer_identity(receipt: Mapping[str, Any]) -> dict[str, bool]:
    pointer_receipt = receipt.get("pointer_identity")
    if not isinstance(pointer_receipt, Mapping):
        raise ProbeMismatch("Python drain omitted pointer_identity")
    rows = _sequence(pointer_receipt.get("rows"), label="pointer_identity.rows")
    identity: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProbeMismatch("pointer_identity row is not a mapping")
        surface = row.get("surface")
        exact = row.get("exact")
        if not isinstance(surface, str) or type(exact) is not bool:
            raise ProbeMismatch("pointer_identity row is malformed")
        identity[surface] = exact
    if pointer_receipt.get("all_exact") is not True or not all(identity.values()):
        raise ProbeMismatch("Python drain changed tensor pointers")
    return identity


def _run_success_case(
    module: ModuleType,
    device: str,
    expected_digest: Any,
    actual_digest: Any,
) -> dict[str, int]:
    expected = _build_synthetic_bank(device)
    actual = _build_synthetic_bank(device)
    actual_state = _state_tensors(actual)
    actual_before = _snapshot(actual_state)
    expected_source_before = _snapshot(
        {**expected.columns, "event_status": expected.status}
    )
    actual_source_before = _snapshot({**actual.columns, "event_status": actual.status})
    prefix_lengths = [len(archive) for archive in expected.archives]

    python_receipt = drain_event_window(
        expected.columns,
        active_count=expected.active_count,
        event_sequence_base=expected.sequence_base,
        event_status=expected.status,
        archives=expected.archives,
    )
    raw_native = getattr(module, ENTRYPOINT)(
        [actual.columns[name] for name in EVENT_COLUMN_NAMES],
        actual.active_count,
        actual.sequence_base,
        actual.status,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    if not isinstance(raw_native, Mapping):
        raise ProbeMismatch(f"{ENTRYPOINT} did not return a mapping")
    native_chunks = _native_chunks(raw_native)

    tensor_comparisons = 0
    for cell, count in enumerate(ACTIVE_COUNTS):
        if count == 0:
            continue
        expected_chunk = expected.archives[cell][prefix_lengths[cell]]
        for column_index, name in enumerate(EVENT_COLUMN_NAMES):
            _record_tensor_pair(
                expected_digest,
                actual_digest,
                label=f"{device}.drained.cell={cell}.{name}",
                expected=expected_chunk[name],
                actual=native_chunks[cell][column_index],
            )
            tensor_comparisons += 1

    for name in EVENT_COLUMN_NAMES:
        _record_tensor_pair(
            expected_digest,
            actual_digest,
            label=f"{device}.source_after.{name}",
            expected=expected.columns[name],
            actual=actual.columns[name],
        )
        tensor_comparisons += 1
    for label, expected_tensor, actual_tensor in (
        ("active_count_after", expected.active_count, actual.active_count),
        ("sequence_base_after", expected.sequence_base, actual.sequence_base),
        ("status_after", expected.status, actual.status),
    ):
        _record_tensor_pair(
            expected_digest,
            actual_digest,
            label=f"{device}.{label}",
            expected=expected_tensor,
            actual=actual_tensor,
        )
        tensor_comparisons += 1

    source_preservation_checks = _assert_snapshot_unchanged(
        expected_source_before,
        {**expected.columns, "event_status": expected.status},
        label=f"{device}.python_source",
    )
    source_preservation_checks += _assert_snapshot_unchanged(
        actual_source_before,
        {**actual.columns, "event_status": actual.status},
        label=f"{device}.native_source",
    )

    python_pointers = _python_pointer_identity(python_receipt)
    native_pointers = _pointer_identity(actual_before, actual_state)
    if not all(native_pointers.values()):
        raise ProbeMismatch("native drain changed tensor pointers")
    _record_value_pair(
        expected_digest,
        actual_digest,
        label=f"{device}.pointer_identity",
        expected=python_pointers,
        actual=native_pointers,
    )

    structured_pairs = (
        (
            "active_count_before",
            python_receipt["active_count_before"],
            raw_native.get("active_count_before"),
        ),
        (
            "sequence_base_before",
            python_receipt["event_sequence_base_before"],
            raw_native.get("event_sequence_base_before"),
        ),
        (
            "sequence_base_after",
            python_receipt["event_sequence_base_after"],
            raw_native.get("event_sequence_base_after"),
        ),
        (
            "archive_bytes",
            python_receipt["host_archive_bytes_appended"],
            raw_native.get("host_archive_bytes_appended"),
        ),
        ("source_columns_preserved", True, raw_native.get("source_columns_preserved")),
    )
    for label, expected_value, actual_value in structured_pairs:
        _record_value_pair(
            expected_digest,
            actual_digest,
            label=f"{device}.{label}",
            expected=expected_value,
            actual=actual_value,
        )

    return {
        "pointer_identity_checks": len(python_pointers),
        "source_preservation_checks": source_preservation_checks,
        "structured_comparisons": len(structured_pairs) + 1,
        "tensor_comparisons": tensor_comparisons,
    }


def _caught_error(call: Any, *, label: str, required_text: str) -> str:
    try:
        call()
    except Exception as error:
        message = str(error)
        if required_text not in message.lower():
            raise ProbeMismatch(
                f"{label} rejected for the wrong reason: {message}"
            ) from error
        return f"{type(error).__name__}: {message}"
    raise ProbeMismatch(f"{label} did not fail closed")


def _run_fail_closed_case(
    module: ModuleType,
    device: str,
    fault: str,
    expected_digest: Any,
    actual_digest: Any,
) -> dict[str, Any]:
    expected = _build_synthetic_bank(device, fault=fault)
    actual = _build_synthetic_bank(device, fault=fault)
    expected_state = _state_tensors(expected)
    actual_state = _state_tensors(actual)
    expected_before = _snapshot(expected_state)
    actual_before = _snapshot(actual_state)
    required_text = "sequence" if fault == "bad_sequence" else "status"

    python_error = _caught_error(
        lambda: drain_event_window(
            expected.columns,
            active_count=expected.active_count,
            event_sequence_base=expected.sequence_base,
            event_status=expected.status,
            archives=expected.archives,
        ),
        label=f"{device}.{fault}.python",
        required_text=required_text,
    )
    native_error = _caught_error(
        lambda: getattr(module, ENTRYPOINT)(
            [actual.columns[name] for name in EVENT_COLUMN_NAMES],
            actual.active_count,
            actual.sequence_base,
            actual.status,
        ),
        label=f"{device}.{fault}.native",
        required_text=required_text,
    )
    if device == "cuda":
        torch.cuda.synchronize()

    no_mutation_checks = _assert_snapshot_unchanged(
        expected_before,
        expected_state,
        label=f"{device}.{fault}.python",
    )
    no_mutation_checks += _assert_snapshot_unchanged(
        actual_before,
        actual_state,
        label=f"{device}.{fault}.native",
    )
    tensor_comparisons = 0
    for name in sorted(expected_state):
        _record_tensor_pair(
            expected_digest,
            actual_digest,
            label=f"{device}.{fault}.unchanged.{name}",
            expected=expected_state[name],
            actual=actual_state[name],
        )
        tensor_comparisons += 1
    _record_value_pair(
        expected_digest,
        actual_digest,
        label=f"{device}.{fault}.rejected_without_mutation",
        expected=True,
        actual=True,
    )
    return {
        "case": fault,
        "native_error": native_error,
        "no_mutation_checks": no_mutation_checks,
        "python_error": python_error,
        "tensor_comparisons": tensor_comparisons,
        "verdict": "pass",
    }


def _run_device_comparison(module: ModuleType, device: str) -> dict[str, Any]:
    expected_digest = hashlib.sha256()
    actual_digest = hashlib.sha256()
    success = _run_success_case(
        module,
        device,
        expected_digest,
        actual_digest,
    )
    fail_closed = [
        _run_fail_closed_case(
            module,
            device,
            fault,
            expected_digest,
            actual_digest,
        )
        for fault in FAIL_CLOSED_CASES
    ]
    expected_sha256 = expected_digest.hexdigest()
    actual_sha256 = actual_digest.hexdigest()
    if expected_sha256 != actual_sha256:
        raise ProbeMismatch(f"{device} aggregate hashes differ")
    return {
        "active_counts": list(ACTIVE_COUNTS),
        "actual_sha256": actual_sha256,
        "byte_exact": True,
        "device": device,
        "expected_sha256": expected_sha256,
        "fail_closed": fail_closed,
        "fail_closed_case_count": len(fail_closed),
        "no_mutation_checks": sum(case["no_mutation_checks"] for case in fail_closed),
        "physical_capacity": PHYSICAL_CAPACITY,
        "pointer_identity_checks": success["pointer_identity_checks"],
        "sequence_bases": list(SEQUENCE_BASES),
        "source_preservation_checks": success["source_preservation_checks"],
        "structured_comparisons": (
            success["structured_comparisons"] + len(fail_closed)
        ),
        "tensor_comparisons": success["tensor_comparisons"]
        + sum(case["tensor_comparisons"] for case in fail_closed),
        "verdict": "pass",
    }


def _combined_hash(
    comparisons: Sequence[Mapping[str, Any]],
    field: str,
) -> str:
    digest = hashlib.sha256()
    for comparison in comparisons:
        _update_bytes(
            digest,
            str(comparison["device"]),
            bytes.fromhex(str(comparison[field])),
        )
    return digest.hexdigest()


def run_probe(extension: Path) -> dict[str, Any]:
    extension_sha256 = _sha256(extension)
    module = _load_extension(extension)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    device_comparisons = [_run_device_comparison(module, device) for device in devices]
    expected_combined = _combined_hash(device_comparisons, "expected_sha256")
    actual_combined = _combined_hash(device_comparisons, "actual_sha256")
    if expected_combined != actual_combined:
        raise ProbeMismatch("combined aggregate hashes differ")
    if _sha256(extension) != extension_sha256:
        raise RuntimeError("extension changed while the probe was running")
    return {
        "byte_exact": True,
        "comparisons": {
            "by_device": {
                row["device"]: {
                    key: value
                    for key, value in row.items()
                    if key not in {"actual_sha256", "expected_sha256"}
                }
                for row in device_comparisons
            },
            "device_count": len(device_comparisons),
            "fail_closed_case_count": sum(
                row["fail_closed_case_count"] for row in device_comparisons
            ),
            "no_mutation_checks": sum(
                row["no_mutation_checks"] for row in device_comparisons
            ),
            "pointer_identity_checks": sum(
                row["pointer_identity_checks"] for row in device_comparisons
            ),
            "source_preservation_checks": sum(
                row["source_preservation_checks"] for row in device_comparisons
            ),
            "structured_comparisons": sum(
                row["structured_comparisons"] for row in device_comparisons
            ),
            "tensor_comparisons": sum(
                row["tensor_comparisons"] for row in device_comparisons
            ),
        },
        "devices": devices,
        "entrypoint": ENTRYPOINT,
        "extension": str(extension),
        "extension_sha256": extension_sha256,
        "hashes": {
            "actual_by_device_sha256": {
                row["device"]: row["actual_sha256"] for row in device_comparisons
            },
            "actual_combined_sha256": actual_combined,
            "expected_by_device_sha256": {
                row["device"]: row["expected_sha256"] for row in device_comparisons
            },
            "expected_combined_sha256": expected_combined,
        },
        "module_name": MODULE_NAME,
        "schema_version": SCHEMA_VERSION,
        "verdict": "pass",
    }


def _failure_receipt(
    options: Options,
    *,
    extension_sha256: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "byte_exact": False,
        "comparisons": None,
        "devices": None,
        "entrypoint": ENTRYPOINT,
        "error": {
            "message": str(error),
            "type": type(error).__name__,
        },
        "extension": str(options.extension),
        "extension_sha256": extension_sha256,
        "hashes": {
            "actual_by_device_sha256": None,
            "actual_combined_sha256": None,
            "expected_by_device_sha256": None,
            "expected_combined_sha256": None,
        },
        "module_name": MODULE_NAME,
        "schema_version": SCHEMA_VERSION,
        "verdict": "fail",
    }


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    try:
        options = _options_from_namespace(namespace)
    except ValueError as error:
        build_parser().error(str(error))

    extension_sha256 = _sha256(options.extension)
    try:
        receipt = run_probe(options.extension)
    except Exception as error:  # A failed probe must still leave a JSON verdict.
        receipt = _failure_receipt(
            options,
            extension_sha256=extension_sha256,
            error=error,
        )
        _write_receipt(options.output, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 1
    _write_receipt(options.output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
