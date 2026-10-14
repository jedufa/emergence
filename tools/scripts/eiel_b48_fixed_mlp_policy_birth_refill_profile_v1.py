#!/usr/bin/env python3
"""Measure one exact physical-B48 fixed-MLP policy-birth refill surface."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import canonical_sha256  # noqa: E402
from engine.campaigns.orchestration import atomic_write_json  # noqa: E402


CELLS = 48
EPISODES = 124
POLICY_WIDTH = 9_991
PACKED_ROW_BYTES = (POLICY_WIDTH + 7) // 8
REQUIRED_FUTURE_EPISODES = 3
REQUIRED_REPEATS = 5
BENCHMARK_ARMS = ("legacy", "m7_physical_refill", "m8_suffix_reuse")
M7_BUILD_SCHEMA = "eiel_option_d_m7_batched_cpu_policy_refill_build_identity_v1"
M8_BUILD_SCHEMA = "eiel_option_d_m8_resident_policy_suffix_reuse_build_identity_v1"
SPLIT_FIELDS = (
    "torch_random_generation_time_ns",
    "torch_compare_time_ns",
    "crossover_pack_time_ns",
    "torch_compare_pack_time_ns",
    "torch_randn_generation_time_ns",
    "policy_allocation_stack_copy_time_ns",
    "device_to_host_time_ns",
    "host_to_device_time_ns",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hardware_identity(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or device.index not in (None, 0):
        raise RuntimeError("component profiling requires CUDA device 0")
    properties = torch.cuda.get_device_properties(0)
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if completed.returncode != 0 or len(fields) != 2 or not all(fields):
        raise RuntimeError("exact GPU hardware identity is unavailable")
    identity = {
        "device_index": 0,
        "device_count": int(torch.cuda.device_count()),
        "device_name": str(properties.name),
        "device_capability": [int(properties.major), int(properties.minor)],
        "device_total_memory_bytes": int(properties.total_memory),
        "device_multiprocessor_count": int(properties.multi_processor_count),
        "gpu_uuid": fields[0],
        "driver_version": fields[1],
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime_version": str(torch.version.cuda),
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def _validated_build_identity(
    value: Any,
    *,
    schema_version: str,
    enabled_field: str,
    expected_enabled: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"manifest lacks {schema_version} build identity")
    identity = dict(value)
    observed_digest = identity.pop("identity_sha256", None)
    macro = {
        "option_d_m7_batched_cpu_policy_refill_enabled": (
            "EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED"
        ),
        "option_d_m8_resident_policy_suffix_reuse_enabled": (
            "EIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED"
        ),
    }[enabled_field]
    expected_authority = {
        "option_d_m7_batched_cpu_policy_refill_enabled": (
            "candidate_batched_exact_cpu_lsb0_policy_refill"
        ),
        "option_d_m8_resident_policy_suffix_reuse_enabled": (
            "candidate_resident_policy_suffix_reuse"
        ),
    }[enabled_field]
    runtime_sha256 = identity.get("runtime_resource_receipt_sha256")
    if observed_digest != canonical_sha256(identity):
        raise RuntimeError(f"{schema_version} digest mismatch")
    if (
        identity.get("schema_version") != schema_version
        or identity.get(enabled_field) is not expected_enabled
        or identity.get("compile_time_macro") != macro
        or identity.get("receipt_field") != enabled_field
        or identity.get("default_enabled") is not False
        or identity.get("compile_definition")
        != f"-D{macro}={1 if expected_enabled else 0}"
        or identity.get("authority")
        != (expected_authority if expected_enabled else "disabled")
        or identity.get("identity_authority") != "compiled_binary_resource_receipt"
        or identity.get("runtime_resource_receipt_confirmed") is not True
        or not isinstance(runtime_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", runtime_sha256) is None
    ):
        raise RuntimeError(f"manifest has the wrong {schema_version} arm identity")
    identity["identity_sha256"] = observed_digest
    return identity


def _validated_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    artifact: Path,
    expected_arm: str,
) -> dict[str, Any]:
    if expected_arm not in BENCHMARK_ARMS:
        raise RuntimeError(f"unknown component benchmark arm: {expected_arm}")
    expected_m7 = expected_arm != "legacy"
    expected_m8 = expected_arm == "m8_suffix_reuse"
    artifact_sha256 = _sha256(artifact)
    if (
        manifest.get("module_name") != "eiel_cuda_architecture_b_wdl_jungle_eed_v1"
        or manifest.get("artifact_sha256") != artifact_sha256
        or not isinstance(manifest.get("artifact_identity"), Mapping)
        or manifest["artifact_identity"].get("sha256") != artifact_sha256
    ):
        raise RuntimeError("artifact is not exactly hash-bound by its manifest")
    snapshot = manifest.get("committed_source_snapshot")
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("manifest lacks committed source snapshot")
    snapshot_payload = dict(snapshot)
    snapshot_digest = snapshot_payload.pop("compiled_source_snapshot_sha256", None)
    source_commit = snapshot_payload.get("build_source_commit")
    if (
        snapshot_payload.get("schema_version")
        != "eiel_authoritative_state_committed_source_snapshot_v1"
        or not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or snapshot_digest != canonical_sha256(snapshot_payload)
    ):
        raise RuntimeError("committed source snapshot identity is invalid")
    m7 = _validated_build_identity(
        manifest.get("option_d_m7_batched_cpu_policy_refill_build"),
        schema_version=M7_BUILD_SCHEMA,
        enabled_field="option_d_m7_batched_cpu_policy_refill_enabled",
        expected_enabled=expected_m7,
    )
    m8 = _validated_build_identity(
        manifest.get("option_d_m8_resident_policy_suffix_reuse_build"),
        schema_version=M8_BUILD_SCHEMA,
        enabled_field="option_d_m8_resident_policy_suffix_reuse_enabled",
        expected_enabled=expected_m8,
    )
    return {
        "artifact_sha256": artifact_sha256,
        "artifact_source_commit": source_commit,
        "compiled_source_snapshot_sha256": snapshot_digest,
        "m7": m7,
        "m8": m8,
    }


def _validated_run_contract(
    *, benchmark_arm: str, repeats: int, expected_output_digest: str | None
) -> tuple[bool, bool]:
    if benchmark_arm not in BENCHMARK_ARMS:
        raise ValueError(f"unknown component benchmark arm: {benchmark_arm}")
    if repeats != REQUIRED_REPEATS:
        raise ValueError(f"exactly {REQUIRED_REPEATS} repeats are required")
    expected_m7_enabled = benchmark_arm != "legacy"
    expected_m8_enabled = benchmark_arm == "m8_suffix_reuse"
    if expected_m7_enabled and (
        not isinstance(expected_output_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_output_digest) is None
    ):
        raise ValueError("candidate arms require the exact control output digest")
    return expected_m7_enabled, expected_m8_enabled


def _summary(values: list[int]) -> dict[str, float | int]:
    samples = sorted(values)
    if not samples:
        raise ValueError("at least one timing sample is required")
    return {
        "count": len(samples),
        "min_ns": samples[0],
        "median_ns": statistics.median(samples),
        "mean_ns": statistics.fmean(samples),
        "p95_ns": samples[math.ceil(0.95 * len(samples)) - 1],
        "max_ns": samples[-1],
    }


def _load(path: Path, module_name: str) -> Any:
    if not module_name.startswith("eiel_cuda_architecture_b_"):
        raise RuntimeError("manifest contains an unexpected extension module name")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("flattened-controller artifact cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_states() -> torch.Tensor:
    rows = []
    for cell in range(CELLS):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0xE1E1_0000 + cell)
        rows.append(generator.get_state())
    state = torch.stack(rows).contiguous()
    if state.dtype != torch.uint8 or tuple(state.shape) != (CELLS, 5_056):
        raise RuntimeError(
            "Torch CPU generator state ABI differs from the selected route"
        )
    return state


def _fixture(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "source_state": _source_states().to(device),
        "delta": torch.empty((CELLS, EPISODES, 1), dtype=torch.float32, device=device),
        "mutation_mask": torch.empty(
            (CELLS, EPISODES, 1), dtype=torch.bool, device=device
        ),
        "crossover": torch.empty(
            (CELLS, EPISODES, PACKED_ROW_BYTES), dtype=torch.uint8, device=device
        ),
        "post_state": torch.empty(
            (CELLS, EPISODES, 0), dtype=torch.uint8, device=device
        ),
        "semantic": torch.empty((CELLS, EPISODES), dtype=torch.int64, device=device),
        "absolute_cursor": torch.zeros((CELLS,), dtype=torch.int64, device=device),
        "semantic_counter": torch.zeros((CELLS,), dtype=torch.int64, device=device),
        "window_base": torch.empty((CELLS,), dtype=torch.int64, device=device),
    }


def _reset(fixture: Mapping[str, torch.Tensor]) -> None:
    fixture["delta"].fill_(7.0)
    fixture["mutation_mask"].fill_(True)
    fixture["crossover"].fill_(0xA5)
    fixture["semantic"].fill_(-1)
    fixture["window_base"].zero_()


def _output_digest(fixture: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in (
        "source_state",
        "delta",
        "mutation_mask",
        "crossover",
        "post_state",
        "semantic",
        "absolute_cursor",
        "semantic_counter",
        "window_base",
    ):
        value = fixture[name].detach().cpu().contiguous()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _call(module: Any, fixture: Mapping[str, torch.Tensor]) -> Mapping[str, Any]:
    return module.eiel_cuda_architecture_b_refill_fixed_mlp_policy_window_v1(
        fixture["source_state"],
        fixture["delta"],
        fixture["mutation_mask"],
        fixture["crossover"],
        fixture["post_state"],
        fixture["semantic"],
        fixture["absolute_cursor"],
        fixture["semantic_counter"],
        fixture["window_base"],
        POLICY_WIDTH,
        False,
        "dense",
        0.0,
        0.0,
        "",
    )


def _call_reuse(module: Any, fixture: Mapping[str, torch.Tensor]) -> Mapping[str, Any]:
    return module.eiel_cuda_architecture_b_refill_or_reuse_fixed_mlp_policy_window_v1(
        fixture["source_state"],
        fixture["delta"],
        fixture["mutation_mask"],
        fixture["crossover"],
        fixture["post_state"],
        fixture["semantic"],
        fixture["absolute_cursor"],
        fixture["semantic_counter"],
        fixture["window_base"],
        POLICY_WIDTH,
        False,
        "dense",
        0.0,
        0.0,
        REQUIRED_FUTURE_EPISODES,
        "",
    )


def _timed_call(
    callback: Any, fixture: Mapping[str, torch.Tensor], device: torch.device
) -> tuple[Mapping[str, Any], dict[str, int]]:
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    start_ns = time.perf_counter_ns()
    result = callback(fixture)
    end_event.record()
    end_event.synchronize()
    return result, {
        "host_wall_time_ns": time.perf_counter_ns() - start_ns,
        "cuda_stream_time_ns": int(
            round(start_event.elapsed_time(end_event) * 1_000_000)
        ),
    }


def _validate_m8_reuse_receipt(
    result: Mapping[str, Any],
    fixture: Mapping[str, torch.Tensor],
    *,
    pre_callback_digest: str,
) -> None:
    if (
        result.get("operation") != "reuse_fixed_mlp_policy_window_v1"
        or result.get("physical_refill_performed") is not False
        or result.get("resident_window_reused") is not True
        or result.get("option_d_m8_resident_policy_suffix_reuse_enabled") is not True
        or result.get("option_d_m8_resident_policy_suffix_reuse_used") is not True
        or result.get("option_d_m7_batched_cpu_policy_refill_used") is not False
        or result.get("required_future_episodes") != REQUIRED_FUTURE_EPISODES
        or result.get("host_to_device_time_ns") != 0
        or result.get("host_to_device_bytes_lower_bound") != 0
        or result.get("torch_random_generation_time_ns") != 0
        or result.get("torch_compare_pack_time_ns") != 0
        or result.get("torch_randn_generation_time_ns") != 0
        or result.get("policy_allocation_stack_copy_time_ns") != 0
        or (
            "host_to_device_operations_lower_bound" in result
            and result.get("host_to_device_operations_lower_bound") != 0
        )
        or pre_callback_digest != _output_digest(fixture)
    ):
        raise RuntimeError("M8 callback did not prove exact zero-mutation reuse")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--expected-output-digest")
    parser.add_argument("--benchmark-arm", choices=BENCHMARK_ARMS, required=True)
    parser.add_argument("--require-split-telemetry", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite immutable policy-refill profile")
    expected_m7_enabled, expected_m8_enabled = _validated_run_contract(
        benchmark_arm=args.benchmark_arm,
        repeats=args.repeats,
        expected_output_digest=args.expected_output_digest,
    )
    manifest = json.loads(args.manifest.read_text())
    manifest_identity = _validated_manifest_identity(
        manifest, artifact=args.artifact, expected_arm=args.benchmark_arm
    )
    module = _load(args.artifact, str(manifest["module_name"]))
    device = torch.device("cuda")
    hardware_identity = _hardware_identity(device)
    fixture = _fixture(device)
    _reset(fixture)
    _call(module, fixture)
    torch.cuda.synchronize(device)

    samples: dict[str, list[int]] = {
        "host_wall_time_ns": [],
        "cuda_stream_time_ns": [],
    }
    output_digests: list[str] = []
    result_identity: dict[str, Any] | None = None
    invariant_state = {
        name: fixture[name].clone()
        for name in ("source_state", "absolute_cursor", "semantic_counter")
    }
    for _ in range(args.repeats):
        _reset(fixture)
        expected_operation = "refill_fixed_mlp_policy_window_v1"
        callback = functools.partial(_call, module)
        pre_callback_digest: str | None = None
        if expected_m8_enabled:
            prefill = _call(module, fixture)
            if (
                prefill.get("operation") != "refill_fixed_mlp_policy_window_v1"
                or prefill.get("option_d_m7_batched_cpu_policy_refill_used") is not True
            ):
                raise RuntimeError("M8 prefill did not use exact M7 physical fallback")
            torch.cuda.synchronize(device)
            pre_callback_digest = _output_digest(fixture)
            expected_operation = "reuse_fixed_mlp_policy_window_v1"
            callback = functools.partial(_call_reuse, module)
        result, timing = _timed_call(callback, fixture, device)
        if result.get("operation") != expected_operation:
            raise RuntimeError("artifact returned the wrong callback operation")
        if (
            result.get("batch_cells") != CELLS
            or result.get("physical_capacity") != EPISODES
        ):
            raise RuntimeError("artifact returned the wrong physical refill surface")
        for name, expected in invariant_state.items():
            if not torch.equal(fixture[name], expected):
                raise RuntimeError(f"refill changed preserved state: {name}")
        if result.get("source_rng_state_preserved") is not True:
            raise RuntimeError("refill did not attest source-state preservation")
        if result.get("absolute_cursor_preserved") is not True:
            raise RuntimeError("refill did not attest absolute-cursor preservation")
        if result.get("semantic_draws_preserved") is not True:
            raise RuntimeError("refill did not attest semantic-counter preservation")
        if expected_m8_enabled:
            assert pre_callback_digest is not None
            _validate_m8_reuse_receipt(
                result, fixture, pre_callback_digest=pre_callback_digest
            )
        if args.require_split_telemetry:
            missing = [field for field in SPLIT_FIELDS if field not in result]
            if missing:
                raise RuntimeError(
                    f"M0 artifact omitted split timing fields: {missing}"
                )
            if int(result["torch_compare_pack_time_ns"]) != (
                int(result["torch_compare_time_ns"])
                + int(result["crossover_pack_time_ns"])
            ):
                raise RuntimeError("compare plus pack timing does not close")
        for name, value in timing.items():
            samples[name].append(value)
        for field in SPLIT_FIELDS:
            if field in result:
                samples.setdefault(field, []).append(int(result[field]))
        if "host_wall_time_ns" in result:
            native_host_wall_time_ns = int(result["host_wall_time_ns"])
            samples.setdefault("native_host_wall_time_ns", []).append(
                native_host_wall_time_ns
            )
            native_accounted_time_ns = sum(
                int(result.get(field, 0))
                for field in (
                    "device_to_host_time_ns",
                    "torch_random_generation_time_ns",
                    "torch_compare_time_ns",
                    "crossover_pack_time_ns",
                    "torch_randn_generation_time_ns",
                    "policy_allocation_stack_copy_time_ns",
                )
            )
            samples.setdefault("native_accounted_time_ns", []).append(
                native_accounted_time_ns
            )
            samples.setdefault("native_residual_time_ns", []).append(
                native_host_wall_time_ns - native_accounted_time_ns
            )
            samples.setdefault(
                "outer_synchronization_and_binding_residual_time_ns", []
            ).append(timing["host_wall_time_ns"] - native_host_wall_time_ns)
        output_digests.append(_output_digest(fixture))
        identity = {
            "option_d_m0_telemetry_enabled": result.get(
                "option_d_m0_telemetry_enabled"
            ),
            "option_d_m4_device_policy_refill_enabled": result.get(
                "option_d_m4_device_policy_refill_enabled"
            ),
            "option_d_m4_device_policy_refill_authority": result.get(
                "option_d_m4_device_policy_refill_authority"
            ),
            "generated_rows": result.get(
                "option_d_m4_device_policy_refill_generated_rows"
            ),
            "generated_bytes": result.get(
                "option_d_m4_device_policy_refill_generated_bytes"
            ),
            "option_d_m7_batched_cpu_policy_refill_enabled": result.get(
                "option_d_m7_batched_cpu_policy_refill_enabled"
            ),
            "option_d_m7_batched_cpu_policy_refill_used": result.get(
                "option_d_m7_batched_cpu_policy_refill_used"
            ),
            "option_d_m7_batched_cpu_policy_refill_authority": result.get(
                "option_d_m7_batched_cpu_policy_refill_authority"
            ),
            "option_d_m8_resident_policy_suffix_reuse_enabled": result.get(
                "option_d_m8_resident_policy_suffix_reuse_enabled"
            ),
            "option_d_m8_resident_policy_suffix_reuse_used": result.get(
                "option_d_m8_resident_policy_suffix_reuse_used"
            ),
            "option_d_m8_resident_policy_suffix_reuse_authority": result.get(
                "option_d_m8_resident_policy_suffix_reuse_authority"
            ),
            "physical_refill_performed": result.get("physical_refill_performed"),
            "resident_window_reused": result.get("resident_window_reused"),
        }
        if (
            identity["option_d_m7_batched_cpu_policy_refill_enabled"]
            is not expected_m7_enabled
        ):
            raise RuntimeError("runtime receipt has the wrong M7 arm identity")
        expected_m7_used = expected_m7_enabled and not expected_m8_enabled
        if (
            identity["option_d_m7_batched_cpu_policy_refill_used"]
            is not expected_m7_used
        ):
            raise RuntimeError("runtime refill did not use the expected M7 arm")
        if expected_m8_enabled and (
            identity["option_d_m8_resident_policy_suffix_reuse_enabled"] is not True
            or identity["option_d_m8_resident_policy_suffix_reuse_used"] is not True
        ):
            raise RuntimeError("runtime callback did not use the expected M8 arm")
        if result_identity is not None and result_identity != identity:
            raise RuntimeError("refill authority identity changed between repeats")
        result_identity = identity
    if len(set(output_digests)) != 1:
        raise RuntimeError("policy-refill output changed between identical repeats")
    output_digest = output_digests[0]
    if args.expected_output_digest and output_digest != args.expected_output_digest:
        raise RuntimeError(
            "policy-refill output differs from the expected control digest"
        )

    receipt = {
        "schema_version": "eiel_b48_fixed_mlp_policy_birth_refill_profile_v1",
        "authority": "fixed_mlp_callback_component_profile_v1",
        "status": "success",
        "purpose": "same-surface fixed-MLP callback component gate input",
        "benchmark_arm": args.benchmark_arm,
        "callback_operation": (
            "resident_policy_suffix_reuse"
            if expected_m8_enabled
            else "physical_policy_refill"
        ),
        "required_future_episodes": (
            REQUIRED_FUTURE_EPISODES if expected_m8_enabled else None
        ),
        "required_repeats": REQUIRED_REPEATS,
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": manifest_identity["artifact_sha256"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "artifact_source_commit": manifest_identity["artifact_source_commit"],
        "compiled_source_snapshot_sha256": manifest_identity[
            "compiled_source_snapshot_sha256"
        ],
        "host_profiler_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "build_identity": {
            "m0": manifest["option_d_m0_telemetry_build"],
            "m4": manifest["option_d_m4_device_policy_refill_build"],
            "m7": manifest_identity["m7"],
            "m8": manifest_identity["m8"],
        },
        "hardware_identity": hardware_identity,
        "fixture": {
            "cells": CELLS,
            "episodes": EPISODES,
            "policy_width": POLICY_WIDTH,
            "packed_row_bytes": PACKED_ROW_BYTES,
            "mutation_enable": False,
            "mutation_mode": "dense",
            "compact_mutation_width": 1,
        },
        "result_identity": result_identity,
        "timing_ns": {
            name: _summary(values) for name, values in sorted(samples.items())
        },
        "output_digest": output_digest,
        "expected_output_digest": args.expected_output_digest,
        "exact_control_digest_match": (
            output_digest == args.expected_output_digest
            if args.expected_output_digest
            else None
        ),
        "scope_limit": "isolated B48 compact disabled-mutation refill; not end-to-end, checkpoint, or qualification evidence",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    args.output_root.mkdir(parents=True)
    path = args.output_root / "m0_policy_birth_refill_profile_receipt.json"
    atomic_write_json(path, receipt)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
