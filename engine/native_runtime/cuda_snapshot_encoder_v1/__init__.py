"""Coarse Python control boundary for exact native V1 snapshot encoding.

The proof is intentionally host-only and synchronous.  Python validates and
binds one semantic snapshot plan, then each boundary crosses into native code
once with two contiguous CPU tensors.  No snapshot row objects cross back.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from time import perf_counter_ns
from typing import Any

from engine.native_runtime.cuda_snapshot_plan_v1 import (
    CompiledCudaSnapshotPlanV1,
    NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1,
    NATIVE_SNAPSHOT_STATE_CAPTURE_FULL_RESIDENT_V1,
    NATIVE_SNAPSHOT_STATE_CAPTURE_TRANSPORT_SCHEMA_V1,
    validate_cuda_snapshot_plan_v1,
)

from .contract import (
    NATIVE_SNAPSHOT_ENCODER_ABI_V1,
    NATIVE_SNAPSHOT_ENCODER_EXTENSION_V1,
    NATIVE_SNAPSHOT_ENCODER_RECEIPT_SCHEMA_V1,
    NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1,
    NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1,
)
from .native_loader import (
    LoadedNativeSnapshotEncoderV1,
    load_native_snapshot_encoder_v1,
)


def _validate_state_capture_descriptor_v1(result: Mapping[str, Any]) -> None:
    if (
        result.get("state_capture_transport_schema_version")
        != NATIVE_SNAPSHOT_STATE_CAPTURE_TRANSPORT_SCHEMA_V1
        or result.get("state_capture_preferred_mode")
        != NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1
    ):
        raise RuntimeError("native snapshot state capture transport differs")
    mode = result.get("state_capture_mode")
    if mode not in {
        NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1,
        NATIVE_SNAPSHOT_STATE_CAPTURE_FULL_RESIDENT_V1,
    }:
        raise RuntimeError("native snapshot state capture mode differs")
    mode_is_preferred = result.get("state_capture_mode_is_preferred")
    if type(mode_is_preferred) is not bool or mode_is_preferred != (
        mode == NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1
    ):
        raise RuntimeError("native snapshot preferred capture-mode receipt differs")
    resident_row_bytes = result.get("resident_state_row_bytes")
    input_row_bytes = result.get("state_input_row_bytes")
    input_bytes = result.get("state_input_bytes")
    selected_bytes = result.get("canonical_state_bytes_selected")
    if (
        type(resident_row_bytes) is not int
        or resident_row_bytes <= 0
        or type(input_row_bytes) is not int
        or input_row_bytes <= 0
        or type(input_bytes) is not int
        or input_bytes <= 0
        or input_bytes % input_row_bytes
        or type(selected_bytes) is not int
        or not 0 < selected_bytes <= input_bytes
        or (
            mode == NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1
            and input_row_bytes >= resident_row_bytes
        )
        or (
            mode == NATIVE_SNAPSHOT_STATE_CAPTURE_FULL_RESIDENT_V1
            and input_row_bytes != resident_row_bytes
        )
    ):
        raise RuntimeError("native snapshot state capture byte receipt differs")


def compile_native_snapshot_encoder_plan_v1(
    plan: CompiledCudaSnapshotPlanV1 | bytes,
    *,
    extension_path: str | os.PathLike[str] | None = None,
    loaded_extension: LoadedNativeSnapshotEncoderV1 | None = None,
) -> Any:
    """Bind one validated semantic plan outside the snapshot hot path."""

    validated = validate_cuda_snapshot_plan_v1(plan)
    loaded = loaded_extension or load_native_snapshot_encoder_v1(extension_path)
    native_plan = loaded.module.compile_snapshot_encoder_plan_v1(
        validated.canonical_plan_bytes,
        validated.to_mapping(),
    )
    if native_plan.plan_sha256 != validated.plan_sha256:
        raise RuntimeError("native encoder plan identity differs")
    return native_plan


def encode_native_snapshot_v1(
    native_plan: Any,
    *,
    state: Any,
    active_metrics_projection: Mapping[str, Any],
    absolute_tick: int,
) -> dict[str, Any]:
    """Encode one boundary natively and return only payloads plus a descriptor."""

    if type(absolute_tick) is not int or absolute_tick < 0:
        raise ValueError("snapshot absolute tick must be a nonnegative exact integer")
    if not isinstance(active_metrics_projection, Mapping):
        raise TypeError("active metrics projection must be a mapping")
    external_started_ns = perf_counter_ns()
    result = dict(
        native_plan.encode(
            state,
            dict(active_metrics_projection),
            absolute_tick,
        )
    )
    external_full_call_elapsed_ns = perf_counter_ns() - external_started_ns
    if result.get("schema_version") != NATIVE_SNAPSHOT_ENCODER_RECEIPT_SCHEMA_V1:
        raise RuntimeError("native snapshot encoder receipt schema differs")
    _validate_state_capture_descriptor_v1(result)
    timing = dict(result.get("timing", {}))
    if "core_encode_compress_hash_ns" not in timing:
        raise RuntimeError("native snapshot encoder core timing is absent")
    timing.update(
        {
            "external_boundary_clock": "time.perf_counter_ns",
            "external_full_call_elapsed_ns": external_full_call_elapsed_ns,
        }
    )
    result["timing"] = timing
    return result


def encode_native_snapshot_to_stage_file_v1(
    native_plan: Any,
    *,
    state: Any,
    active_metrics_projection: Mapping[str, Any],
    absolute_tick: int,
    stage_path: str | os.PathLike[str],
    encoding_policy: str = NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1,
    _test_fail_after_compressed_bytes: int | None = None,
) -> dict[str, Any]:
    """Encode, write, and fsync one exclusive stage file entirely natively.

    This synchronous Gate 4A proof intentionally retains caller ownership of
    the input tensors for the full call.  The returned descriptor contains no
    snapshot payload bytes.  A raised error never denotes completion; if the
    exclusive create succeeded, the explicit stage path remains for caller
    cleanup and forensic inspection.
    """

    if type(absolute_tick) is not int or absolute_tick < 0:
        raise ValueError("snapshot absolute tick must be a nonnegative exact integer")
    if not isinstance(active_metrics_projection, Mapping):
        raise TypeError("active metrics projection must be a mapping")
    raw_stage_path = os.fspath(stage_path)
    if not isinstance(raw_stage_path, str) or not os.path.isabs(raw_stage_path):
        raise ValueError("snapshot stage path must be one absolute filesystem path")
    if encoding_policy != NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1:
        raise ValueError("snapshot stage encoding policy is unsupported")
    if _test_fail_after_compressed_bytes is not None and (
        type(_test_fail_after_compressed_bytes) is not int
        or _test_fail_after_compressed_bytes < 0
    ):
        raise ValueError("snapshot stage fault injection byte count is invalid")
    external_started_ns = perf_counter_ns()
    result = dict(
        native_plan.encode_to_stage_file(
            state,
            dict(active_metrics_projection),
            absolute_tick,
            raw_stage_path,
            encoding_policy,
            _test_fail_after_compressed_bytes,
        )
    )
    external_full_call_elapsed_ns = perf_counter_ns() - external_started_ns
    if result.get("schema_version") != NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1:
        raise RuntimeError("native snapshot stage-file receipt schema differs")
    _validate_state_capture_descriptor_v1(result)
    if "canonical_jsonl" in result or "deterministic_gzip_level_9" in result:
        raise RuntimeError("native snapshot stage-file API returned payload bytes")
    timing = dict(result.get("timing", {}))
    if "native_encode_write_fsync_close_ns" not in timing:
        raise RuntimeError("native snapshot stage-file timing is absent")
    timing.update(
        {
            "external_boundary_clock": "time.perf_counter_ns",
            "external_full_call_elapsed_ns": external_full_call_elapsed_ns,
        }
    )
    result["timing"] = timing
    return result


def encode_owned_native_snapshot_to_stage_file_v1(
    native_plan: Any,
    *,
    state: Any,
    metrics: Any,
    absolute_tick: int,
    stage_path: str | os.PathLike[str],
    encoding_policy: str,
    _test_fail_after_compressed_bytes: int | None = None,
) -> dict[str, Any]:
    """Encode one exclusively leased resident tensor slot to a stage file.

    This internal service boundary constructs no Python metrics projection and
    computes its content digest in native code.  Native encode, compression,
    write, and file fsync run without the GIL.  The caller must hold an
    exclusive lease on both tensors until this call returns; native tensor
    version guards fail closed if ordinary Torch mutation violates that lease.
    """

    if type(absolute_tick) is not int or absolute_tick < 0:
        raise ValueError("snapshot absolute tick must be a nonnegative exact integer")
    raw_stage_path = os.fspath(stage_path)
    if not isinstance(raw_stage_path, str) or not os.path.isabs(raw_stage_path):
        raise ValueError("snapshot stage path must be one absolute filesystem path")
    if encoding_policy != NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1:
        raise ValueError("snapshot stage encoding policy is unsupported")
    if _test_fail_after_compressed_bytes is not None and (
        type(_test_fail_after_compressed_bytes) is not int
        or _test_fail_after_compressed_bytes < 0
    ):
        raise ValueError("snapshot stage fault injection byte count is invalid")
    external_started_ns = perf_counter_ns()
    result = dict(
        native_plan.encode_owned_tensors_to_stage_file(
            state,
            metrics,
            absolute_tick,
            raw_stage_path,
            encoding_policy,
            _test_fail_after_compressed_bytes,
        )
    )
    external_full_call_elapsed_ns = perf_counter_ns() - external_started_ns
    if result.get("schema_version") != NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1:
        raise RuntimeError("native snapshot stage-file receipt schema differs")
    _validate_state_capture_descriptor_v1(result)
    if "canonical_jsonl" in result or "deterministic_gzip_level_9" in result:
        raise RuntimeError("native snapshot stage-file API returned payload bytes")
    if (
        result.get("input_ownership_contract")
        != "native_service_exclusive_tensor_lease_until_completion_v1"
        or result.get("gil_held_for_full_synchronous_call") is not False
    ):
        raise RuntimeError("native owned snapshot stage-file contract differs")
    metrics_sha256 = result.get("active_metrics_content_sha256")
    if (
        not isinstance(metrics_sha256, str)
        or len(metrics_sha256) != 64
        or any(child not in "0123456789abcdef" for child in metrics_sha256)
    ):
        raise RuntimeError("native snapshot metrics content digest is invalid")
    timing = dict(result.get("timing", {}))
    if "native_encode_write_fsync_close_ns" not in timing:
        raise RuntimeError("native snapshot stage-file timing is absent")
    timing.update(
        {
            "external_boundary_clock": "time.perf_counter_ns",
            "external_full_call_elapsed_ns": external_full_call_elapsed_ns,
        }
    )
    result["timing"] = timing
    return result


__all__ = [
    "LoadedNativeSnapshotEncoderV1",
    "NATIVE_SNAPSHOT_ENCODER_ABI_V1",
    "NATIVE_SNAPSHOT_ENCODER_EXTENSION_V1",
    "NATIVE_SNAPSHOT_ENCODER_RECEIPT_SCHEMA_V1",
    "NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1",
    "NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1",
    "compile_native_snapshot_encoder_plan_v1",
    "encode_native_snapshot_to_stage_file_v1",
    "encode_native_snapshot_v1",
    "encode_owned_native_snapshot_to_stage_file_v1",
    "load_native_snapshot_encoder_v1",
]
