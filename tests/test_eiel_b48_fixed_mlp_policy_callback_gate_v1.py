from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from engine.campaigns.canonical import canonical_sha256
from tools.scripts.eiel_b48_fixed_mlp_policy_callback_gate_v1 import (
    EXACT_FIXTURE,
    ComponentGateError,
    reduce_component_profiles,
)
from tools.scripts.eiel_b48_fixed_mlp_policy_birth_refill_profile_v1 import (
    M7_BUILD_SCHEMA,
    M8_BUILD_SCHEMA,
)


def _identity(schema: str, field: str, enabled: bool, salt: str) -> dict[str, Any]:
    macro = {
        "option_d_m7_batched_cpu_policy_refill_enabled": (
            "EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED"
        ),
        "option_d_m8_resident_policy_suffix_reuse_enabled": (
            "EIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED"
        ),
    }[field]
    authority = {
        "option_d_m7_batched_cpu_policy_refill_enabled": (
            "candidate_batched_exact_cpu_lsb0_policy_refill"
        ),
        "option_d_m8_resident_policy_suffix_reuse_enabled": (
            "candidate_resident_policy_suffix_reuse"
        ),
    }[field]
    payload = {
        "schema_version": schema,
        field: enabled,
        "compile_time_macro": macro,
        "receipt_field": field,
        "default_enabled": False,
        "compile_definition": f"-D{macro}={1 if enabled else 0}",
        "authority": authority if enabled else "disabled",
        "identity_authority": "compiled_binary_resource_receipt",
        "runtime_resource_receipt_confirmed": True,
        "runtime_resource_receipt_sha256": salt * 64,
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def _hardware() -> dict[str, Any]:
    payload = {
        "device_index": 0,
        "device_count": 1,
        "device_name": "test GPU",
        "device_capability": [7, 5],
        "device_total_memory_bytes": 6_000_000_000,
        "device_multiprocessor_count": 24,
        "gpu_uuid": "GPU-test",
        "driver_version": "999.1",
        "torch_version": "2.test",
        "torch_cuda_runtime_version": "12.test",
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def _profile(path: Path, *, arm: str, median_ns: int) -> Path:
    m7 = arm != "legacy"
    m8 = arm == "m8_suffix_reuse"
    output_digest = "d" * 64
    payload: dict[str, Any] = {
        "schema_version": "eiel_b48_fixed_mlp_policy_birth_refill_profile_v1",
        "authority": "fixed_mlp_callback_component_profile_v1",
        "status": "success",
        "benchmark_arm": arm,
        "callback_operation": (
            "resident_policy_suffix_reuse" if m8 else "physical_policy_refill"
        ),
        "required_future_episodes": 3 if m8 else None,
        "required_repeats": 5,
        "artifact_sha256": ("1" if arm == "legacy" else "2" if not m8 else "3") * 64,
        "manifest_sha256": ("4" if arm == "legacy" else "5" if not m8 else "6") * 64,
        "artifact_source_commit": "a" * 40,
        "compiled_source_snapshot_sha256": "b" * 64,
        "host_profiler_source_commit": "a" * 40,
        "build_identity": {
            "m7": _identity(
                M7_BUILD_SCHEMA,
                "option_d_m7_batched_cpu_policy_refill_enabled",
                m7,
                "7" if arm == "legacy" else "8" if not m8 else "9",
            ),
            "m8": _identity(
                M8_BUILD_SCHEMA,
                "option_d_m8_resident_policy_suffix_reuse_enabled",
                m8,
                "a" if arm == "legacy" else "b" if not m8 else "c",
            ),
        },
        "hardware_identity": _hardware(),
        "fixture": dict(EXACT_FIXTURE),
        "result_identity": {
            "option_d_m7_batched_cpu_policy_refill_enabled": m7,
            "option_d_m7_batched_cpu_policy_refill_used": m7 and not m8,
            "option_d_m8_resident_policy_suffix_reuse_enabled": m8,
            "option_d_m8_resident_policy_suffix_reuse_used": m8,
            "physical_refill_performed": False if m8 else None,
            "resident_window_reused": True if m8 else None,
        },
        "timing_ns": {
            "host_wall_time_ns": {
                "count": 5,
                "min_ns": median_ns - 2,
                "median_ns": median_ns,
                "mean_ns": float(median_ns),
                "p95_ns": median_ns + 2,
                "max_ns": median_ns + 2,
            }
        },
        "output_digest": output_digest,
        "expected_output_digest": output_digest if m7 else None,
        "exact_control_digest_match": True if m7 else None,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _profiles(tmp_path: Path, *, m7_ns: int = 100, m8_ns: int = 90) -> dict[str, Path]:
    return {
        "legacy": _profile(tmp_path / "legacy.json", arm="legacy", median_ns=110),
        "m7": _profile(tmp_path / "m7.json", arm="m7_physical_refill", median_ns=m7_ns),
        "m8": _profile(tmp_path / "m8.json", arm="m8_suffix_reuse", median_ns=m8_ns),
    }


def _reduce(paths: dict[str, Path]) -> dict[str, Any]:
    return reduce_component_profiles(
        legacy_path=paths["legacy"], m7_path=paths["m7"], m8_path=paths["m8"]
    )


def _mutate(path: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("receipt_sha256")
    mutation(payload)
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_component_gate_is_digest_bound_and_uses_both_candidate_medians(
    tmp_path: Path,
) -> None:
    result = _reduce(_profiles(tmp_path))

    unsigned = dict(result)
    assert unsigned.pop("receipt_sha256") == canonical_sha256(unsigned)
    assert result["callback_median_gate_passed"] is True
    assert result["median_speedup_over_legacy"]["m7_physical_refill"] == 1.1
    assert result["m8_build_binds_m7_exact_physical_refill_fallback"] is True
    assert result["required_repeats_per_arm"] == 5


def test_component_gate_fails_when_either_candidate_is_below_1_10(
    tmp_path: Path,
) -> None:
    result = _reduce(_profiles(tmp_path, m7_ns=101))

    assert result["callback_median_gate_passed"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(required_repeats=4), "exact frozen"),
        (lambda row: row.update(artifact_source_commit="e" * 40), "source commit"),
        (
            lambda row: row["hardware_identity"].update(device_name="other"),
            "hardware identity digest",
        ),
        (lambda row: row.update(expected_output_digest=None), "control digest"),
        (
            lambda row: row["build_identity"]["m8"].update(
                option_d_m8_resident_policy_suffix_reuse_enabled=False
            ),
            "identity digest",
        ),
    ],
)
def test_component_gate_fails_closed_on_contract_drift(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    paths = _profiles(tmp_path)
    _mutate(paths["m8"], mutation)

    with pytest.raises(ComponentGateError, match=message):
        _reduce(paths)


def test_component_gate_rejects_profile_receipt_tampering(tmp_path: Path) -> None:
    paths = _profiles(tmp_path)
    payload = json.loads(paths["m7"].read_text(encoding="utf-8"))
    payload["timing_ns"]["host_wall_time_ns"]["median_ns"] = 1
    paths["m7"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ComponentGateError, match="profile digest mismatch"):
        _reduce(paths)
