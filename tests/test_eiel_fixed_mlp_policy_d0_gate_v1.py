from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from engine.campaigns.canonical import canonical_sha256
from tools.scripts import eiel_fixed_mlp_policy_d0_gate_v1 as gate


def _seal(payload: dict[str, object]) -> dict[str, object]:
    result = deepcopy(payload)
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def _component(arm: str, *, digest: str = "a" * 64) -> dict[str, object]:
    return _seal(
        {
            "schema_version": f"{gate.SCHEMA_VERSION}_component_worker_v1",
            "authority": gate.AUTHORITY,
            "status": "passed",
            "worker_kind": "component",
            "arm": arm,
            "pid": 101,
            "artifact_sha256": {
                "legacy": "1",
                "m7_physical_refill": "2",
                "m8_suffix_reuse": "3",
            }[arm]
            * 64,
            "manifest_sha256": {
                "legacy": "4",
                "m7_physical_refill": "5",
                "m8_suffix_reuse": "6",
            }[arm]
            * 64,
            "cells": gate.EXPECTED_CELLS,
            "ticks_per_generation": gate.TICKS_PER_GENERATION,
            "physical_refill": {
                "output_digest_sha256": digest,
                "tensor_digest_sha256s": {
                    "source_state": "b" * 64,
                    "crossover": "c" * 64,
                    "absolute_cursor": "d" * 64,
                },
            },
            "heterogeneous_suffix_reuse": (
                {
                    "changed_tensor_names": [],
                    "pre_and_post_tensor_digest_sha256": "e" * 64,
                }
                if arm == "m8_suffix_reuse"
                else None
            ),
            "qualification_claim": False,
            "production_admission": False,
        }
    )


def _trajectory(
    arm: str,
    action: str,
    *,
    projection: str = "f" * 64,
    checkpoint_sha: str | None = None,
) -> dict[str, object]:
    coordinates = {
        "terminal": (0, 1),
        "prefix": (0, gate.CHECKPOINT_GENERATION),
        "resume": (gate.CHECKPOINT_GENERATION, gate.TERMINAL_GENERATION),
        "uninterrupted": (0, gate.TERMINAL_GENERATION),
    }
    start_generation, terminal_generation = coordinates[action]
    projection_digests = {
        "cell-000": {
            "trajectory_state_sha256": projection,
            "canonical_checkpoint_storage_sha256": projection,
        }
    }
    return _seal(
        {
            "schema_version": f"{gate.SCHEMA_VERSION}_trajectory_worker_v1",
            "authority": gate.AUTHORITY,
            "status": "passed",
            "worker_kind": "trajectory",
            "action": action,
            "arm": arm,
            "pid": 202,
            "start_generation": start_generation,
            "terminal_generation": terminal_generation,
            "terminal_tick": 0,
            "artifact_sha256": {
                "legacy": "1",
                "m7_physical_refill": "2",
                "m8_suffix_reuse": "3",
            }[arm]
            * 64,
            "manifest_sha256": {
                "legacy": "4",
                "m7_physical_refill": "5",
                "m8_suffix_reuse": "6",
            }[arm]
            * 64,
            "diagnostic_plan_sha256": "7" * 64,
            "request_payload_sha256": "8" * 64,
            "request_generations": (
                1 if action == "terminal" else gate.TERMINAL_GENERATION
            ),
            "request_checkpoint_milestones": (
                [1]
                if action == "terminal"
                else [gate.CHECKPOINT_GENERATION, gate.TERMINAL_GENERATION]
            ),
            "request_agent_id_allocation_terminal_generation": (
                1 if action == "terminal" else gate.TERMINAL_GENERATION
            ),
            "segment_agent_id_allocation_generations": (
                2
                if action == "terminal"
                else min(gate.TERMINAL_GENERATION - start_generation, 128)
            ),
            "cells": gate.EXPECTED_CELLS,
            "ticks_per_generation": gate.TICKS_PER_GENERATION,
            "projection_digests": projection_digests,
            "serialized_checkpoint_sha256": (
                checkpoint_sha if action == "prefix" else None
            ),
            "source_checkpoint_sha256": (
                checkpoint_sha if action == "resume" else None
            ),
            "serialized_checkpoint_projection_digest_set_sha256": (
                canonical_sha256(projection_digests) if action == "prefix" else None
            ),
            "source_checkpoint_projection_digest_set_sha256": (
                canonical_sha256(projection_digests) if action == "resume" else None
            ),
            "fresh_owner_restore": action == "resume",
            "future_native_consumption": action == "resume",
            "turnover_and_control_receipts_valid": True,
            "qualification_claim": False,
            "production_admission": False,
        }
    )


def _artifacts() -> dict[str, dict[str, str]]:
    return {
        arm: {
            "artifact_sha256": {
                "legacy": "1",
                "m7_physical_refill": "2",
                "m8_suffix_reuse": "3",
            }[arm]
            * 64,
            "manifest_sha256": {
                "legacy": "4",
                "m7_physical_refill": "5",
                "m8_suffix_reuse": "6",
            }[arm]
            * 64,
        }
        for arm in gate.ARMS
    }


def _valid_reducer_inputs() -> dict[str, object]:
    checkpoint_sha = "9" * 64
    return {
        "artifact_identities": _artifacts(),
        "components": {arm: _component(arm) for arm in gate.ARMS},
        "arm_terminals": {arm: _trajectory(arm, "terminal") for arm in gate.ARMS},
        "prefix": _trajectory(
            "m8_suffix_reuse", "prefix", checkpoint_sha=checkpoint_sha
        ),
        "uninterrupted": _trajectory("m8_suffix_reuse", "uninterrupted"),
        "restores": [
            _trajectory("m8_suffix_reuse", "resume", checkpoint_sha=checkpoint_sha),
            _trajectory("m8_suffix_reuse", "resume", checkpoint_sha=checkpoint_sha),
        ],
        "producer_exit_confirmed": True,
    }


def test_reducer_closes_exact_component_checkpoint_and_replay_contract() -> None:
    decision = gate._reduce_receipts(**_valid_reducer_inputs())
    assert decision["d0_exact"] is True
    assert decision["forced_m8_to_m7_refill_exact"] is True
    assert decision["checkpoint_resume_parity"] is True
    assert decision["second_replay_parity"] is True
    assert decision["partial_t80_checkpoint_provable"] is False
    assert decision["faithful_checkpoint_boundary"] == "generation:3:tick:0"


def test_reducer_fails_closed_on_one_physical_tensor_mismatch() -> None:
    inputs = _valid_reducer_inputs()
    components = deepcopy(inputs["components"])
    components["m7_physical_refill"] = _component("m7_physical_refill", digest="0" * 64)
    inputs["components"] = components
    with pytest.raises(gate.D0GateError, match="physical-refill tensors"):
        gate._reduce_receipts(**inputs)


def test_reducer_fails_closed_without_producer_process_release() -> None:
    inputs = _valid_reducer_inputs()
    inputs["producer_exit_confirmed"] = False
    with pytest.raises(gate.D0GateError, match="producer process release"):
        gate._reduce_receipts(**inputs)


def test_reducer_fails_closed_on_replay_trajectory_drift() -> None:
    inputs = _valid_reducer_inputs()
    checkpoint_sha = "9" * 64
    inputs["restores"] = [
        _trajectory("m8_suffix_reuse", "resume", checkpoint_sha=checkpoint_sha),
        _trajectory(
            "m8_suffix_reuse",
            "resume",
            checkpoint_sha=checkpoint_sha,
            projection="0" * 64,
        ),
    ]
    with pytest.raises(gate.D0GateError, match="fresh restore 2"):
        gate._reduce_receipts(**inputs)


def test_reducer_binds_every_checkpoint_worker_to_frozen_m8_artifact() -> None:
    inputs = _valid_reducer_inputs()
    prefix = deepcopy(inputs["prefix"])
    prefix.pop("receipt_sha256")
    prefix["artifact_sha256"] = "0" * 64
    inputs["prefix"] = _seal(prefix)
    with pytest.raises(gate.D0GateError, match="frozen M8 artifact"):
        gate._reduce_receipts(**inputs)


def test_worker_receipt_digest_detects_post_write_tamper() -> None:
    receipt = _component("legacy")
    receipt["pid"] = 999
    with pytest.raises(gate.D0GateError, match="worker receipt is invalid"):
        gate._validate_worker_receipt(receipt, kind="component", arm="legacy")


def test_tensor_digest_binds_name_shape_dtype_and_bytes() -> None:
    tensors = {
        "crossover": torch.zeros((2, 3), dtype=torch.uint8),
        "cursor": torch.zeros((2,), dtype=torch.int64),
    }
    before = gate._tensor_digests(tensors)
    tensors["crossover"][1, 2] = 1
    after = gate._tensor_digests(tensors)
    assert before["crossover"] != after["crossover"]
    assert before["cursor"] == after["cursor"]


def test_worker_command_binds_exact_m8_artifact_and_checkpoint(tmp_path: Path) -> None:
    args = Namespace(
        plan=tmp_path / "plan.json",
        legacy_artifact=tmp_path / "legacy.so",
        legacy_manifest=tmp_path / "legacy.json",
        m7_artifact=tmp_path / "m7.so",
        m7_manifest=tmp_path / "m7.json",
        m8_artifact=tmp_path / "m8.so",
        m8_manifest=tmp_path / "m8.json",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    command = gate._worker_command(
        args,
        kind="trajectory",
        arm="m8_suffix_reuse",
        action="resume",
        output_root=tmp_path / "out",
        checkpoint_path=checkpoint,
    )
    assert command[command.index("--artifact") + 1] == str(args.m8_artifact)
    assert command[command.index("--manifest") + 1] == str(args.m8_manifest)
    assert command[command.index("--checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--action") + 1] == "resume"
    assert command[:3] == [
        gate.sys.executable,
        "-m",
        "tools.scripts.eiel_fixed_mlp_policy_d0_gate_v1",
    ]


def test_trajectory_execution_shapes_respect_native_executor_minimum() -> None:
    assert gate._trajectory_execution_shape("terminal") == (0, 1, 160)
    for action in ("prefix", "resume", "uninterrupted"):
        start, end, absolute_tick_span = gate._trajectory_execution_shape(action)
        assert 3 <= end - start <= 128
        assert absolute_tick_span is None


def test_checkpoint_derivative_declares_absolute_g6_horizon_and_g3_g6_milestones() -> (
    None
):
    source = {
        "execution": {
            "generations": 1,
            "ticks_per_generation": gate.TICKS_PER_GENERATION,
            "checkpoints": [1],
        },
        "plan_sha256": "0" * 64,
    }
    derivative = gate._checkpoint_derivative_plan(source)
    assert source["execution"]["generations"] == 1
    assert derivative["execution"]["generations"] == gate.TERMINAL_GENERATION
    assert derivative["execution"]["checkpoints"] == [
        gate.CHECKPOINT_GENERATION,
        gate.TERMINAL_GENERATION,
    ]
    unsigned = deepcopy(derivative)
    observed = unsigned.pop("plan_sha256")
    assert observed == canonical_sha256(unsigned)


def test_reducer_rejects_checkpoint_request_with_segment_not_terminal_horizon() -> None:
    inputs = _valid_reducer_inputs()
    restored = deepcopy(inputs["restores"][0])
    restored.pop("receipt_sha256")
    restored["request_generations"] = (
        gate.TERMINAL_GENERATION - gate.CHECKPOINT_GENERATION
    )
    inputs["restores"][0] = _seal(restored)
    with pytest.raises(gate.D0GateError, match="admitted request binding"):
        gate._reduce_receipts(**inputs)


def test_static_evidence_reuse_is_exact_contract_and_receipt_digest_bound(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "prior"
    contract: dict[str, object] = {
        "schema_version": f"{gate.SCHEMA_VERSION}_execution_contract_v1",
        "authority": gate.AUTHORITY,
        "repository_head": "a" * 40,
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    gate.atomic_write_json(source_root / "execution_contract.json", contract)
    for arm in gate.ARMS:
        gate.atomic_write_json(
            source_root / "component" / arm / gate.WORKER_RECEIPT,
            _component(arm),
        )
        gate.atomic_write_json(
            source_root / "terminal_b48_g1" / arm / gate.WORKER_RECEIPT,
            _trajectory(arm, "terminal"),
        )

    components, terminals, binding = gate._load_reusable_static_evidence(
        source_root=source_root,
        expected_contract=contract,
        artifact_identities=_artifacts(),
    )
    assert tuple(components) == gate.ARMS
    assert tuple(terminals) == gate.ARMS
    assert len(binding["receipts"]) == 6
    assert binding["source_execution_contract_sha256"] == contract["contract_sha256"]

    changed_contract = dict(contract)
    changed_contract["repository_head"] = "b" * 40
    with pytest.raises(gate.D0GateError, match="contract is not exact"):
        gate._load_reusable_static_evidence(
            source_root=source_root,
            expected_contract=changed_contract,
            artifact_identities=_artifacts(),
        )


def test_multigeneration_m8_allows_exact_fallback_but_t160_does_not() -> None:
    gate._validate_callback_counts(
        expected_m8=True,
        expected_callback_count=477,
        reused=300,
        physical=177,
        require_all_reuse=False,
    )
    with pytest.raises(gate.D0GateError, match="did not reuse every"):
        gate._validate_callback_counts(
            expected_m8=True,
            expected_callback_count=159,
            reused=158,
            physical=1,
            require_all_reuse=True,
        )


def test_native_owner_contract_distinguishes_tick_and_generation_receipts() -> None:
    tick = gate._native_owner_contract(generations=1, tick_span=True)
    assert tick["native_tick_iterations"] == 160
    assert tick["terminal_generation_boundary"] is True
    assert "logical_generation_count" not in tick

    generation = gate._native_owner_contract(generations=3, tick_span=False)
    assert generation["logical_generation_count"] == 3
    assert "native_tick_iterations" not in generation
    assert "terminal_generation_boundary" not in generation


def _runtime_callback(arm: str, *, reused: bool = False) -> dict[str, object]:
    m7_enabled = arm != "legacy"
    m8_enabled = arm == "m8_suffix_reuse"
    physical = not reused
    return {
        "option_d_m7_batched_cpu_policy_refill_enabled": m7_enabled,
        "option_d_m7_batched_cpu_policy_refill_used": m7_enabled and physical,
        "option_d_m7_batched_cpu_policy_refill_authority": (
            "candidate_batched_exact_cpu_compact_refill"
            if m7_enabled and physical
            else (
                "compiled_candidate_not_used_for_surface" if m7_enabled else "disabled"
            )
        ),
        "option_d_m7_batched_cpu_policy_refill_schema_version": (
            "eiel_option_d_m7_batched_cpu_policy_refill_v1"
            if m7_enabled and physical
            else None
        ),
        "option_d_m8_resident_policy_suffix_reuse_enabled": m8_enabled,
        "option_d_m8_resident_policy_suffix_reuse_used": reused,
        "option_d_m8_resident_policy_suffix_reuse_authority": (
            "candidate_resident_policy_suffix_reuse"
            if reused
            else (
                "compiled_candidate_physical_refill_fallback"
                if m8_enabled
                else "disabled"
            )
        ),
        "option_d_m8_resident_policy_suffix_reuse_schema_version": (
            "eiel_option_d_m8_resident_policy_suffix_reuse_v1" if m8_enabled else None
        ),
        "resident_window_reused": reused,
        "physical_refill_performed": physical,
        "turnover_boundary": False,
    }


def _outer_refill(arm: str) -> dict[str, object]:
    result = _runtime_callback(arm)
    for name in (
        "option_d_m8_resident_policy_suffix_reuse_enabled",
        "option_d_m8_resident_policy_suffix_reuse_used",
        "option_d_m8_resident_policy_suffix_reuse_authority",
        "option_d_m8_resident_policy_suffix_reuse_schema_version",
        "resident_window_reused",
        "physical_refill_performed",
        "turnover_boundary",
    ):
        result.pop(name)
    result.update(
        {
            "operation": "refill_fixed_mlp_policy_window_v1",
            "absolute_cursor_preserved": True,
            "semantic_draws_preserved": True,
            "source_rng_state_preserved": True,
        }
    )
    return result


def _generation_result(
    arm: str,
    *,
    action: str = "prefix",
    generations: int = 3,
    start_generation: int | None = None,
    physical_window_generations: int = 2,
) -> dict[str, object]:
    if start_generation is None:
        start_generation = gate.CHECKPOINT_GENERATION if action == "resume" else 0
    first = min(
        physical_window_generations - start_generation % physical_window_generations,
        generations,
    )
    child_generation_counts = [first]
    while sum(child_generation_counts) < generations:
        child_generation_counts.append(
            min(
                physical_window_generations,
                generations - sum(child_generation_counts),
            )
        )
    windows = []
    for child_generations in child_generation_counts:
        callbacks = [
            _runtime_callback(arm, reused=arm == "m8_suffix_reuse")
            for _ in range(child_generations * (gate.TICKS_PER_GENERATION - 1))
        ]
        windows.append(
            {
                "requested_generations": child_generations,
                "completed_generations": child_generations,
                "policy_birth_segment_refill_receipts": callbacks,
                "turnover_shadow_plan_launches": 0,
                "turnover_cull_refill_plan_launches": 0,
                "turnover_commit_launches": 0,
                "turnover_publish_launches": 0,
                "compact_lifecycle_digest_launches": 0,
            }
        )
    result: dict[str, object] = {
        **gate._native_owner_contract(generations=generations, tick_span=False),
        "schema_version": (
            "eiel_cuda_architecture_b_native_resident_window_controller_v1"
        ),
        "physical_window_generations": physical_window_generations,
        "window_count": len(windows),
        "first_window_generations": first,
        "settled_window_count": len(windows),
        "physical_window_alignment": "absolute_generation_lattice_v1",
        "window_receipts": windows,
        "policy_refill_receipts": [_outer_refill(arm) for _ in range(len(windows) - 1)],
        "checkpointed_gaussian_policy_suffix_preserved": (
            start_generation > 0 and start_generation % physical_window_generations != 0
        ),
        "resume_boundary_refill": (
            start_generation > 0 and start_generation % physical_window_generations == 0
        ),
        "resume_policy_refill_receipt": (
            _outer_refill(arm)
            if start_generation > 0
            and start_generation % physical_window_generations == 0
            else {}
        ),
    }
    return result


def test_generation_controller_flattens_child_callbacks_and_validates_boundaries() -> (
    None
):
    summary = gate._policy_runtime_summary(
        _generation_result("m8_suffix_reuse"),
        arm="m8_suffix_reuse",
        generations=3,
        require_all_reuse=False,
        generation_action="prefix",
        start_generation=0,
    )
    assert summary["settled_window_count"] == 2
    assert summary["callback_receipt_count"] == 3 * 159
    assert summary["resident_suffix_reuse_count"] == 3 * 159
    assert summary["outer_physical_refill_count"] == 1
    assert summary["resume_boundary_refill"] is False


def test_generation_controller_requires_generation_bound_callbacks_per_child() -> None:
    result = _generation_result("m8_suffix_reuse")
    result["window_receipts"][1]["policy_birth_segment_refill_receipts"].pop()
    with pytest.raises(gate.D0GateError, match="generation-bound callback"):
        gate._policy_runtime_summary(
            result,
            arm="m8_suffix_reuse",
            generations=3,
            require_all_reuse=False,
            generation_action="prefix",
            start_generation=0,
        )


def test_generation_controller_requires_g_minus_one_exact_outer_refills() -> None:
    result = _generation_result("m8_suffix_reuse")
    result["policy_refill_receipts"].pop()
    with pytest.raises(gate.D0GateError, match="outer policy-refill count"):
        gate._policy_runtime_summary(
            result,
            arm="m8_suffix_reuse",
            generations=3,
            require_all_reuse=False,
            generation_action="prefix",
            start_generation=0,
        )


def test_odd_generation_resume_preserves_reconstructed_suffix_without_refill() -> None:
    valid = _generation_result("m8_suffix_reuse", action="resume")
    summary = gate._policy_runtime_summary(
        valid,
        arm="m8_suffix_reuse",
        generations=3,
        require_all_reuse=False,
        generation_action="resume",
        start_generation=gate.CHECKPOINT_GENERATION,
    )
    assert summary["settled_window_count"] == 2
    assert summary["resume_boundary_refill"] is False

    invalid = _generation_result("m8_suffix_reuse", action="resume")
    invalid["resume_boundary_refill"] = True
    invalid["resume_policy_refill_receipt"] = _outer_refill("m8_suffix_reuse")
    with pytest.raises(gate.D0GateError, match="impossible resume-boundary"):
        gate._policy_runtime_summary(
            invalid,
            arm="m8_suffix_reuse",
            generations=3,
            require_all_reuse=False,
            generation_action="resume",
            start_generation=gate.CHECKPOINT_GENERATION,
        )


def test_aligned_generation_resume_requires_exact_m7_boundary_refill() -> None:
    result = _generation_result("m8_suffix_reuse", action="resume", start_generation=4)
    summary = gate._policy_runtime_summary(
        result,
        arm="m8_suffix_reuse",
        generations=3,
        require_all_reuse=False,
        generation_action="resume",
        start_generation=4,
    )
    assert summary["resume_boundary_refill"] is True

    result["resume_policy_refill_receipt"] = {}
    with pytest.raises(gate.D0GateError, match="omitted its exact boundary"):
        gate._policy_runtime_summary(
            result,
            arm="m8_suffix_reuse",
            generations=3,
            require_all_reuse=False,
            generation_action="resume",
            start_generation=4,
        )
