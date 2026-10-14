#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "eiel_option_d_m17_runner_drive_noise_receipt_v1"
OPERATOR_ID = "runner.drive_noise.standard_normal"
STREAM_ID = "numpy.runner"
ALGORITHM_ID = "numpy.pcg64.xsl_rr_128_64"

ZERO_COUNTER_FIELDS = (
    "exact_cpu_policy_service_calls_inside_tick_loop",
    "host_synchronizations_inside_loop",
    "intermediate_host_synchronizations",
    "intermediate_device_to_host_operations",
    "intermediate_host_to_device_operations",
    "aten_operations_inside_tick_loop",
)


class M17ReceiptError(RuntimeError):
    pass


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_projection(path: Path) -> Mapping[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise M17ReceiptError(f"{path} did not contain a JSON object")
    projection = payload.get("projection")
    if not isinstance(projection, Mapping):
        raise M17ReceiptError(f"{path} omitted projection")
    metadata = projection.get("metadata")
    if not isinstance(metadata, Mapping):
        raise M17ReceiptError(f"{path} omitted projection.metadata")
    return payload


def _canonical_digest(value: Any) -> str:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _strict_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise M17ReceiptError(f"{field} must be an integer")
    return int(value)


def _nested_mapping(payload: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any]:
    current: Any = payload
    dotted = ".".join(path)
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise M17ReceiptError(f"gate receipt omitted {dotted}")
        current = current[key]
    if not isinstance(current, Mapping):
        raise M17ReceiptError(f"gate receipt {dotted} must be an object")
    return current


def _gate_decision(gate: Mapping[str, Any]) -> Any:
    semantic = gate.get("semantic_slice")
    if isinstance(semantic, Mapping) and "decision" in semantic:
        return semantic.get("decision")
    return gate.get("decision")


def _chunk_receipts(gate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    dispatch = _nested_mapping(gate, ("dispatch",))
    chunks = dispatch.get("chunk_receipts")
    if not isinstance(chunks, list) or not chunks:
        raise M17ReceiptError("gate receipt omitted dispatch.chunk_receipts")
    out: list[Mapping[str, Any]] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            raise M17ReceiptError(f"dispatch.chunk_receipts[{index}] must be an object")
        out.append(chunk)
    return out


def _source_checks(*, kernels: Path, direct_source: Path, binding: Path) -> dict[str, Any]:
    kernels_text = kernels.read_text(encoding="utf-8")
    direct_text = direct_source.read_text(encoding="utf-8")
    binding_text = binding.read_text(encoding="utf-8")
    checks = {
        "flattened_kernel_materializes_missing_drive_noise": (
            "materialize_missing_drive_noise_v1(" in kernels_text
            and "pcg64_numpy_standard_normal" in kernels_text
            and "full_tick_outputs.drive_noise_draws" in kernels_text
            and "metrics_np_semantic_draws" in kernels_text
        ),
        "direct_resident_path_does_not_call_host_refill": (
            "refill_drive_noise_window_v1" not in direct_text
        ),
        "host_refill_compatibility_symbol_still_bound": (
            "eiel_cuda_architecture_b_refill_drive_noise_window_v1" in binding_text
        ),
        "source_files": {
            str(kernels): _file_sha256(kernels),
            str(direct_source): _file_sha256(direct_source),
            str(binding): _file_sha256(binding),
        },
    }
    checks["device_source_binding_passed"] = bool(
        checks["flattened_kernel_materializes_missing_drive_noise"]
        and checks["direct_resident_path_does_not_call_host_refill"]
    )
    return checks


def _projection_summary(path: Path, projection_payload: Mapping[str, Any]) -> dict[str, Any]:
    projection = projection_payload["projection"]
    metadata = projection["metadata"]
    runner_state = metadata.get("native_runtime_numpy_runner_state")
    stream_indices = metadata.get("native_runtime_rng_manager_stream_indices")
    if not isinstance(runner_state, Mapping):
        raise M17ReceiptError(f"{path} omitted native_runtime_numpy_runner_state")
    if not isinstance(stream_indices, Mapping):
        raise M17ReceiptError(
            f"{path} omitted native_runtime_rng_manager_stream_indices"
        )
    metrics = stream_indices.get("metrics")
    if not isinstance(metrics, Mapping):
        raise M17ReceiptError(f"{path} omitted metrics stream indices")
    metrics_np = _strict_int(metrics.get("np"), field=f"{path}: metrics.np")
    tensors = projection.get("tensors")
    if not isinstance(tensors, Mapping):
        raise M17ReceiptError(f"{path} omitted projection.tensors")
    drive_samples = tensors.get("drive_noise_samples_by_tick_slot")
    drive_valid = tensors.get("drive_noise_samples_valid_by_tick_slot")
    if not isinstance(drive_samples, Mapping) or not isinstance(drive_valid, Mapping):
        raise M17ReceiptError(f"{path} omitted drive-noise compatibility tensors")
    return {
        "path": str(path),
        "source_generation": projection_payload.get("source_generation"),
        "source_tick": projection_payload.get("source_tick"),
        "source_rng_sha256": projection_payload.get("source_rng_sha256"),
        "source_projection_sha256": projection_payload.get("source_projection_sha256"),
        "runner_pcg64_state_digest": _canonical_digest(runner_state),
        "metrics_np_semantic_draw_index": metrics_np,
        "drive_noise_samples_digest": {
            "dtype": drive_samples.get("dtype"),
            "shape": drive_samples.get("shape"),
            "sha256": drive_samples.get("sha256"),
        },
        "drive_noise_samples_valid_digest": {
            "dtype": drive_valid.get("dtype"),
            "shape": drive_valid.get("shape"),
            "sha256": drive_valid.get("sha256"),
        },
    }


def build_receipt(
    *,
    gate_path: Path,
    checkpoint_before_path: Path,
    checkpoint_after_path: Path,
    kernels_source: Path,
    direct_source: Path,
    binding_source: Path,
) -> dict[str, Any]:
    gate = _read_json(gate_path)
    if not isinstance(gate, Mapping):
        raise M17ReceiptError("gate receipt must be a JSON object")
    chunks = _chunk_receipts(gate)
    native = _nested_mapping(gate, ("dispatch", "native_controller_receipt"))
    counter_totals = {
        field: sum(_strict_int(chunk.get(field, 0), field=field) for chunk in chunks)
        for field in ZERO_COUNTER_FIELDS
    }
    native_host_policy_service_used = any(
        bool(chunk.get("native_host_policy_service_used")) for chunk in chunks
    ) or bool(native.get("native_host_policy_service_used"))
    zero_inside_loop_boundary = (
        all(value == 0 for value in counter_totals.values())
        and not native_host_policy_service_used
    )
    before_payload = _read_projection(checkpoint_before_path)
    after_payload = _read_projection(checkpoint_after_path)
    before = _projection_summary(checkpoint_before_path, before_payload)
    after = _projection_summary(checkpoint_after_path, after_payload)
    metrics_np_delta = (
        after["metrics_np_semantic_draw_index"]
        - before["metrics_np_semantic_draw_index"]
    )
    source = _source_checks(
        kernels=kernels_source,
        direct_source=direct_source,
        binding=binding_source,
    )
    blockers = [
        "isolated_drive_noise_semantic_draw_delta_not_exported_by_gate",
        "raw_pcg64_word_delta_not_exported_by_gate",
        "drive_noise_samples_checkpoint_compatibility_surface_still_present",
    ]
    passed = bool(
        gate.get("ok") is True
        and zero_inside_loop_boundary
        and source["device_source_binding_passed"]
        and metrics_np_delta >= 0
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "operator_id": OPERATOR_ID,
        "stream_id": STREAM_ID,
        "algorithm_id": ALGORITHM_ID,
        "decision": (
            "M17-A-ROUTE-BOUND-RUNNER-DRIVE-NOISE-DEVICE-OWNERSHIP-RECEIPT"
            if passed
            else "M17-REJECTED-RUNNER-DRIVE-NOISE-RECEIPT"
        ),
        "passed": passed,
        "gate": {
            "path": str(gate_path),
            "sha256": _file_sha256(gate_path),
            "ok": gate.get("ok"),
            "status": gate.get("status"),
            "decision": _gate_decision(gate),
        },
        "source": source,
        "ownership": {
            "device_materialization_source_bound": bool(
                source["device_source_binding_passed"]
            ),
            "direct_resident_host_refill_called": not bool(
                source["direct_resident_path_does_not_call_host_refill"]
            ),
            "host_refill_symbol_retained_for_compatibility": bool(
                source["host_refill_compatibility_symbol_still_bound"]
            ),
            "native_host_policy_service_used": native_host_policy_service_used,
            "zero_inside_loop_cpu_native_boundary": zero_inside_loop_boundary,
            "zero_counter_totals": counter_totals,
        },
        "checkpoint_boundary": {
            "before": before,
            "after": after,
            "runner_pcg64_state_changed": (
                before["runner_pcg64_state_digest"]
                != after["runner_pcg64_state_digest"]
            ),
            "source_rng_sha256_changed": (
                before["source_rng_sha256"] != after["source_rng_sha256"]
            ),
            "metrics_np_semantic_draw_index_delta": metrics_np_delta,
            "metrics_np_delta_contains_multiple_runner_operators": True,
        },
        "draw_accounting": {
            "isolated_drive_noise_semantic_draw_delta_available": False,
            "raw_pcg64_delta_available": False,
            "promotion_qualified": False,
            "blockers": blockers,
        },
    }
    receipt["receipt_sha256"] = _canonical_digest(receipt)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--checkpoint-before", type=Path, required=True)
    parser.add_argument("--checkpoint-after", type=Path, required=True)
    parser.add_argument(
        "--kernels-source",
        type=Path,
        default=Path("engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"),
    )
    parser.add_argument(
        "--direct-source",
        type=Path,
        default=Path("engine/perf/cuda_architecture_b_direct_resident_source_allocator_v1.py"),
    )
    parser.add_argument(
        "--binding-source",
        type=Path,
        default=Path("engine/perf/cuda_architecture_b_flattened_controller_v1_binding.cpp"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    receipt = build_receipt(
        gate_path=args.gate,
        checkpoint_before_path=args.checkpoint_before,
        checkpoint_after_path=args.checkpoint_after,
        kernels_source=args.kernels_source,
        direct_source=args.direct_source,
        binding_source=args.binding_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
