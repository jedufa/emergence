from __future__ import annotations

import hashlib

import pytest
import torch

from engine.campaigns.canonical import canonical_sha256
from tools.scripts.eiel_b48_fixed_mlp_policy_birth_refill_profile_v1 import (
    CELLS,
    M7_BUILD_SCHEMA,
    M8_BUILD_SCHEMA,
    PACKED_ROW_BYTES,
    POLICY_WIDTH,
    _fixture,
    _output_digest,
    _reset,
    _source_states,
    _summary,
    _validate_m8_reuse_receipt,
    _validated_manifest_identity,
    _validated_run_contract,
)


def test_policy_birth_refill_profile_uses_the_physical_compact_surface() -> None:
    assert CELLS == 48
    assert POLICY_WIDTH == 9_991
    assert PACKED_ROW_BYTES == 1_249
    states = _source_states()
    assert states.dtype == torch.uint8
    assert tuple(states.shape) == (48, 5_056)
    assert not torch.equal(states[0], states[1])


def test_policy_birth_refill_output_digest_binds_names_shapes_and_bytes() -> None:
    fixture = {
        "source_state": torch.zeros((1,), dtype=torch.uint8),
        "delta": torch.zeros((1,), dtype=torch.float32),
        "mutation_mask": torch.zeros((1,), dtype=torch.bool),
        "crossover": torch.zeros((1,), dtype=torch.uint8),
        "post_state": torch.empty((0,), dtype=torch.uint8),
        "semantic": torch.zeros((1,), dtype=torch.int64),
        "absolute_cursor": torch.zeros((1,), dtype=torch.int64),
        "semantic_counter": torch.zeros((1,), dtype=torch.int64),
        "window_base": torch.zeros((1,), dtype=torch.int64),
    }
    before = _output_digest(fixture)
    fixture["crossover"][0] = 1
    assert _output_digest(fixture) != before
    assert _summary([30, 10, 20])["median_ns"] == 20
    assert _summary(list(range(1, 21)))["p95_ns"] == 19


def test_policy_birth_refill_reset_satisfies_initial_window_cursor_contract() -> None:
    fixture = _fixture(torch.device("cpu"))
    _reset(fixture)
    assert torch.equal(fixture["window_base"], fixture["absolute_cursor"])
    assert bool(torch.all(fixture["window_base"] >= 0))


def _build_identity(schema: str, field: str, enabled: bool) -> dict[str, object]:
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
    payload: dict[str, object] = {
        "schema_version": schema,
        field: enabled,
        "compile_time_macro": macro,
        "receipt_field": field,
        "default_enabled": False,
        "compile_definition": f"-D{macro}={1 if enabled else 0}",
        "authority": authority if enabled else "disabled",
        "identity_authority": "compiled_binary_resource_receipt",
        "runtime_resource_receipt_confirmed": True,
        "runtime_resource_receipt_sha256": "a" * 64,
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def test_component_profile_contract_requires_exact_five_and_control_digest() -> None:
    assert _validated_run_contract(
        benchmark_arm="legacy", repeats=5, expected_output_digest=None
    ) == (False, False)
    assert _validated_run_contract(
        benchmark_arm="m8_suffix_reuse",
        repeats=5,
        expected_output_digest="a" * 64,
    ) == (True, True)
    with pytest.raises(ValueError, match="exactly 5"):
        _validated_run_contract(
            benchmark_arm="legacy", repeats=4, expected_output_digest=None
        )
    with pytest.raises(ValueError, match="control output digest"):
        _validated_run_contract(
            benchmark_arm="m7_physical_refill",
            repeats=5,
            expected_output_digest=None,
        )


def test_component_profile_validates_artifact_source_and_m7_m8_build_identity(
    tmp_path,
) -> None:
    artifact = tmp_path / "candidate.so"
    artifact.write_bytes(b"exact candidate")
    source_payload = {
        "schema_version": "eiel_authoritative_state_committed_source_snapshot_v1",
        "build_source_commit": "b" * 40,
        "sources": [{"path": "engine/candidate.cpp", "sha256": "c" * 64}],
    }
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "module_name": "eiel_cuda_architecture_b_wdl_jungle_eed_v1",
        "artifact_sha256": artifact_sha256,
        "artifact_identity": {"sha256": artifact_sha256},
        "committed_source_snapshot": {
            **source_payload,
            "compiled_source_snapshot_sha256": canonical_sha256(source_payload),
        },
        "option_d_m7_batched_cpu_policy_refill_build": _build_identity(
            M7_BUILD_SCHEMA,
            "option_d_m7_batched_cpu_policy_refill_enabled",
            True,
        ),
        "option_d_m8_resident_policy_suffix_reuse_build": _build_identity(
            M8_BUILD_SCHEMA,
            "option_d_m8_resident_policy_suffix_reuse_enabled",
            True,
        ),
    }

    identity = _validated_manifest_identity(
        manifest, artifact=artifact, expected_arm="m8_suffix_reuse"
    )

    assert identity["artifact_source_commit"] == "b" * 40
    assert identity["m7"]["option_d_m7_batched_cpu_policy_refill_enabled"] is True
    assert identity["m8"]["option_d_m8_resident_policy_suffix_reuse_enabled"] is True

    manifest["option_d_m8_resident_policy_suffix_reuse_build"] = _build_identity(
        M8_BUILD_SCHEMA,
        "option_d_m8_resident_policy_suffix_reuse_enabled",
        False,
    )
    with pytest.raises(RuntimeError, match="wrong.*m8"):
        _validated_manifest_identity(
            manifest, artifact=artifact, expected_arm="m8_suffix_reuse"
        )


def test_component_profile_m8_callback_requires_exact_zero_mutation_reuse() -> None:
    fixture = _fixture(torch.device("cpu"))
    _reset(fixture)
    before = _output_digest(fixture)
    receipt = {
        "operation": "reuse_fixed_mlp_policy_window_v1",
        "physical_refill_performed": False,
        "resident_window_reused": True,
        "option_d_m8_resident_policy_suffix_reuse_enabled": True,
        "option_d_m8_resident_policy_suffix_reuse_used": True,
        "option_d_m7_batched_cpu_policy_refill_used": False,
        "required_future_episodes": 3,
        "host_to_device_time_ns": 0,
        "host_to_device_bytes_lower_bound": 0,
        "host_to_device_operations_lower_bound": 0,
        "torch_random_generation_time_ns": 0,
        "torch_compare_pack_time_ns": 0,
        "torch_randn_generation_time_ns": 0,
        "policy_allocation_stack_copy_time_ns": 0,
    }

    _validate_m8_reuse_receipt(receipt, fixture, pre_callback_digest=before)

    receipt["host_to_device_bytes_lower_bound"] = 1
    with pytest.raises(RuntimeError, match="zero-mutation reuse"):
        _validate_m8_reuse_receipt(receipt, fixture, pre_callback_digest=before)
