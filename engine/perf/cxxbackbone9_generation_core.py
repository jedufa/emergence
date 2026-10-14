from __future__ import annotations

"""CXXBACKBONE-9 monolithic generation boundary.

This module intentionally fails closed for the live runner until the native
generation core owns policy execution and resident state mutation in one call.
It records the concrete call graph and integration blockers so production code
cannot mistake the CXXBACKBONE8R one-call envelope for a monolithic loop.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import time

import torch


SCHEMA_VERSION = "cxxbackbone9_generation_core_v1"
CALL_GRAPH_SCHEMA_VERSION = "cxxbackbone9_generation_call_graph_v1"
RESIDENT_STATE_SCHEMA_VERSION = "resident_runtime_state_v9_audit_v1"
DECISION_POLICY_INTEGRATION_BLOCKED = "CXXBACKBONE9-D/POLICY-INTEGRATION-BLOCKED-AFTER-TWO-APPROACHES"

MANDATORY_BORDER_COUNTERS: tuple[str, ...] = (
    "python_generation_calls",
    "python_tick_iterations_inside_lane",
    "python_policy_callbacks",
    "python_observation_callbacks",
    "python_grouping_calls",
    "python_row_map_builds",
    "python_action_applications",
    "python_environment_steps",
    "python_per_agent_traversals",
    "torch_tensor_constructions",
    "numpy_allocations",
    "h2d_bytes",
    "d2h_bytes",
    "action_copies",
    "observation_copies",
    "cuda_launches",
    "synchronizations",
    "checkpoint_pack_adapters",
    "report_projection_adapters",
)


@dataclass(frozen=True)
class ResidentRuntimeStateV9Audit:
    schema_version: str
    authoritative_owner: str
    execution: tuple[str, ...]
    population: tuple[str, ...]
    policy: tuple[str, ...]
    observation: tuple[str, ...]
    action_environment: tuple[str, ...]
    patch: tuple[str, ...]
    rng: tuple[str, ...]
    checkpoint: tuple[str, ...]
    cuda: tuple[str, ...]
    current_blockers: tuple[str, ...]


def resident_runtime_state_v9_audit_payload() -> dict[str, Any]:
    return asdict(
        ResidentRuntimeStateV9Audit(
            schema_version=RESIDENT_STATE_SCHEMA_VERSION,
            authoritative_owner="blocked_not_yet_native_authoritative",
            execution=("generation", "tick", "seed_cell_id", "device", "capability_mask", "status_error"),
            population=(
                "alive_mask",
                "stable_ids",
                "slots",
                "slot_tokens",
                "positions",
                "occupancy",
                "alive_row_order",
                "id_to_slot_map",
            ),
            policy=(
                "policy_family",
                "mlp_policy_bank",
                "gru_policy_bank",
                "group_rows",
                "group_offsets",
                "row_maps",
                "hmem",
                "action_cache",
                "cache_cursor_validity",
                "membership_epoch",
            ),
            observation=("flat_frame", "patch_frame", "normalization_metadata", "capacity_metadata"),
            action_environment=(
                "actions",
                "movement_proposals",
                "conflict_buffers",
                "positions",
                "occupancy",
                "energy",
                "health_injury_age_where_active",
                "action_costs",
                "flat_patch_action_outcomes",
            ),
            patch=(
                "food",
                "fertility",
                "cover",
                "shelter",
                "movement_cost",
                "scent",
                "waste",
                "density",
                "regrowth_recovery_state",
                "accumulators",
            ),
            rng=("policy_rng", "environment_rng", "draw_indices"),
            checkpoint=("state_manifest", "hashes", "rebuild_metadata", "event_metric_cursor", "capability_state"),
            cuda=("persistent_device_tensors", "capacity", "launch_counters", "transfer_counters", "sync_counters"),
            current_blockers=(
                "live_runner_policy_banks_are_exposed_as_runner_torch_members_not_a_single_resident_state_argument",
                "checkpoint_bundle_v2_materializes_python_runner_state_not_resident_runtime_state_v9",
                "production_tick_loop_still_owns_spawn_context_behavior_logging_and_generation_summary",
            ),
        )
    )


def border_counter_zero_payload(*, python_generation_calls: int = 0) -> dict[str, float]:
    payload = {name: 0.0 for name in MANDATORY_BORDER_COUNTERS}
    payload["python_generation_calls"] = float(python_generation_calls)
    return payload


def generation_call_graph_payload() -> dict[str, Any]:
    rows = [
        {
            "operation": "tick loop",
            "python_function": "simulation.runnercore.tick_loop._run_ticks",
            "native_function": "",
            "owner": "python",
            "tick_frequency": "every_tick",
            "python_callback_count": "steps_per_generation",
            "tensor_allocation_count": "indirect_policy_refill_dependent",
            "h2d_d2h": "policy tensors and action result when device is CUDA",
            "synchronization": "policy result materialization",
            "checkpoint_dependency": "runner/checkpoint_bundle_v2 state after generation",
            "reporting_dependency": "runner summary and behavior logger",
            "intended_cpp_owner": "cxxbackbone9 execute_generation_cpu/cuda",
        },
        {
            "operation": "cache schedule",
            "python_function": "simulation.runnercore.action_cache.runnercore1_actions_strided",
            "native_function": "",
            "owner": "python",
            "tick_frequency": "every_tick",
            "python_callback_count": "cache hit/miss path",
            "tensor_allocation_count": "miss_path_only",
            "h2d_d2h": "none on cache hit; refill uses policy tensors",
            "synchronization": "policy refill dependent",
            "checkpoint_dependency": "policy_action_cache_v1",
            "reporting_dependency": "perf buckets",
            "intended_cpp_owner": "native action cache cursor and writeback",
        },
        {
            "operation": "native observation helper",
            "python_function": "simulation.runnercore.action_cache._compute_actions_k",
            "native_function": "cxxbackbone8r_flat/patch_observation_frame_cpu_v1",
            "owner": "mixed_python_dispatch_native_compute",
            "tick_frequency": "cache_refill",
            "python_callback_count": "one Python dispatch per refill",
            "tensor_allocation_count": "observation frame and torch conversion",
            "h2d_d2h": "observation H2D when CUDA policy",
            "synchronization": "policy dependent",
            "checkpoint_dependency": "none directly",
            "reporting_dependency": "cxxbackbone5 counters",
            "intended_cpp_owner": "native generation observation stage",
        },
        {
            "operation": "grouping and row maps",
            "python_function": "simulation.runnercore.action_cache._compute_actions_k",
            "native_function": "cxxbackbone8r_group_rows_cpu_v1",
            "owner": "mixed_python_dispatch_native_compute",
            "tick_frequency": "cache_refill",
            "python_callback_count": "one Python dispatch per refill",
            "tensor_allocation_count": "slots/mlp_idx/gru_idx tensors",
            "h2d_d2h": "row maps copied to torch device by Python",
            "synchronization": "policy dependent",
            "checkpoint_dependency": "slot/token maps",
            "reporting_dependency": "cxxbackbone5 counters",
            "intended_cpp_owner": "native generation grouping stage",
        },
        {
            "operation": "MLP/GRU/Hmem policy",
            "python_function": "simulation.runnercore.action_cache._compute_actions_k",
            "native_function": "cxxbackbone3_gru_policy_aten_v1 and cxxbackbone4_mlp_policy_aten_v1 when flags enabled",
            "owner": "mixed_python_grouping_native_aten_policy",
            "tick_frequency": "cache_refill",
            "python_callback_count": "one Python policy callback per refill",
            "tensor_allocation_count": "bank selects, action output, Hmem writeback",
            "h2d_d2h": "actions D2H at refill; Hmem bank transfer depends on device/bank_device",
            "synchronization": "action result copied to CPU",
            "checkpoint_dependency": "Hmem and action cache",
            "reporting_dependency": "policy buckets and memory telemetry",
            "intended_cpp_owner": "shared native policy callable inside generation core",
        },
        {
            "operation": "action application and Patch state",
            "python_function": "simulation.runnercore.tick_loop._apply_env_tick_arrays",
            "native_function": "cxxbackbone8r_flat/patch_apply_actions_cpu_v1",
            "owner": "mixed_python_dispatch_native_compute",
            "tick_frequency": "every_tick",
            "python_callback_count": "one Python dispatch per tick",
            "tensor_allocation_count": "input/output conversion per tick",
            "h2d_d2h": "CPU lane only in CXX8R",
            "synchronization": "none for CPU lane",
            "checkpoint_dependency": "world arrays and runner resident state",
            "reporting_dependency": "movement/energy/Patch counters",
            "intended_cpp_owner": "native generation action/environment stage",
        },
        {
            "operation": "generation counters and checkpoint projection",
            "python_function": "simulation.runnercore.tick_loop._run_ticks and checkpoint_bundle_v2",
            "native_function": "",
            "owner": "python",
            "tick_frequency": "generation_boundary",
            "python_callback_count": "one or more generation-boundary adapters",
            "tensor_allocation_count": "checkpoint/report dependent",
            "h2d_d2h": "checkpoint materialization dependent",
            "synchronization": "generation boundary",
            "checkpoint_dependency": "full",
            "reporting_dependency": "full",
            "intended_cpp_owner": "native compact output projection with Python serialization only",
        },
    ]
    return {
        "schema_version": CALL_GRAPH_SCHEMA_VERSION,
        "rows": rows,
        "summary": {
            "python_owned_operations": sum(1 for row in rows if row["owner"] == "python"),
            "mixed_dispatch_operations": sum(1 for row in rows if str(row["owner"]).startswith("mixed")),
            "native_owned_operations": 0,
            "completion_artifact": False,
        },
    }


def native_generation_support_blockers(resident_state: Any, capability_mask: Any) -> tuple[str, ...]:
    blockers: list[str] = []
    if not isinstance(resident_state, dict):
        blockers.append("resident_state_v9_dict_required")
    else:
        if resident_state.get("schema_version") != RESIDENT_STATE_SCHEMA_VERSION:
            blockers.append("resident_runtime_state_v9_not_authoritative")
        if resident_state.get("authoritative_owner") != "native":
            blockers.append("resident_state_owner_not_native")
        if "policy_banks" not in resident_state:
            blockers.append("policy_banks_missing_from_resident_state")
        if "world_arrays" not in resident_state:
            blockers.append("world_arrays_missing_from_resident_state")
        if "checkpoint_projection" not in resident_state:
            blockers.append("checkpoint_projection_missing_from_resident_state")
    if not isinstance(capability_mask, dict):
        blockers.append("capability_mask_dict_required")
    elif bool(capability_mask.get("churn", False)):
        blockers.append("controlled_churn_not_owned_by_native_generation")
    return tuple(blockers)


def execute_generation(
    resident_state: Any,
    capability_mask: Any,
    generation_index: int,
    steps_per_generation: int,
) -> dict[str, Any]:
    """Fail-closed CXXBACKBONE9 generation entrypoint.

    Returning a blocked payload instead of falling back preserves the invariant
    that a successful result means Python did not iterate ticks or mutate state.
    """

    started = time.perf_counter()
    blockers = native_generation_support_blockers(resident_state, capability_mask)
    counters = border_counter_zero_payload(python_generation_calls=1)
    return {
        "schema_version": SCHEMA_VERSION,
        "executed": False,
        "hidden_fallback": False,
        "decision": DECISION_POLICY_INTEGRATION_BLOCKED,
        "generation_index": int(generation_index),
        "steps_per_generation": int(steps_per_generation),
        "native_tick_loop_count": 0,
        "support_blockers": list(blockers),
        "border_counters": counters,
        "monolithic_cpp_loop": False,
        "cuda_resident_generation_loop": False,
        "elapsed_s": float(time.perf_counter() - started),
    }


def attempt_aten_policy_generation_integration(*, build_directory: str | Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "approach": "A/shared_pybind_aten_policy_inside_generation_extension",
        "attempted": True,
        "ok": False,
        "compiled_existing_policy_extension": False,
        "compiled_existing_generation_extension": False,
        "policy_callable_available": False,
        "generation_loop_integrated": False,
        "blockers": [],
    }
    try:
        from perf.cxxbackbone3_policy_aten import load_policy_aten_extension

        load_policy_aten_extension(build_directory=build_directory)
        result["compiled_existing_policy_extension"] = True
        result["policy_callable_available"] = True
    except Exception as exc:  # pragma: no cover - evidence path records exact text.
        result["blockers"].append(f"policy_aten_extension_compile_failed:{type(exc).__name__}:{exc}")
    try:
        from perf.cxxbackbone8r_generation_island import load_generation_island_extension

        load_generation_island_extension(build_directory=build_directory)
        result["compiled_existing_generation_extension"] = True
    except Exception as exc:  # pragma: no cover - evidence path records exact text.
        result["blockers"].append(f"generation_island_extension_compile_failed:{type(exc).__name__}:{exc}")
    if result["compiled_existing_policy_extension"] and result["compiled_existing_generation_extension"]:
        result["blockers"].extend(
            [
                "existing_extensions_compile_but_are_separate_python_dispatched_pybind_modules",
                "live_runner_policy_inputs_are_constructed_by_python_compute_actions_k_before_aten_call",
                "hmem_bank_update_and_action_cache_writeback_remain_python_orchestrated",
                "joining_these_without_resident_state_v9_would_create_an_invalid_python_envelope",
            ]
        )
    result["elapsed_s"] = float(time.perf_counter() - started)
    return result


def attempt_object_free_policy_authority_integration(*, build_directory: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "approach": "B/object_free_cxxbackbone1_policy_authority_kernel",
        "attempted": True,
        "ok": False,
        "compiled_policy_authority_kernel": False,
        "fixture_exact_parity": False,
        "generation_loop_integrated": False,
        "blockers": [],
    }
    try:
        from perf.cxxbackbone1_policy_authority import (
            compare_policy_authority_results,
            compile_policy_authority_kernel,
            load_policy_authority_backend,
            make_policy_authority_fixture,
            result_digest,
            run_policy_authority_python_reference,
        )

        compile_info = compile_policy_authority_kernel(Path(build_directory) / "policy_authority")
        result["compile_info"] = compile_info
        result["compiled_policy_authority_kernel"] = bool(compile_info.get("ok"))
        if not bool(compile_info.get("ok")):
            result["blockers"].append("object_free_policy_authority_compile_failed")
        else:
            backend = load_policy_authority_backend(str(compile_info["library_path"]))
            fixture = make_policy_authority_fixture(rows=9, slots=13, seed=909)
            expected = run_policy_authority_python_reference(fixture)
            observed = backend(fixture)
            parity = compare_policy_authority_results(expected, observed, atol=0.0)
            result["fixture_exact_parity"] = bool(parity.get("ok"))
            result["fixture_digest"] = result_digest(observed)
            result["fixture_parity"] = parity
    except Exception as exc:  # pragma: no cover - evidence path records exact text.
        result["blockers"].append(f"object_free_policy_authority_attempt_failed:{type(exc).__name__}:{exc}")
    if result["fixture_exact_parity"]:
        result["blockers"].extend(
            [
                "object_free_policy_authority_is_ctypes_fixture_authority_not_live_runner_bank_authority",
                "kernel_does_not_own_flat_patch_observation_construction_or_environment_mutation",
                "kernel_does_not_integrate_checkpoint_bundle_v2_or_resident_runtime_state_v9",
                "using_it_from_python_per_refill_would_not_satisfy_one_native_generation_call",
            ]
        )
    result["elapsed_s"] = float(time.perf_counter() - started)
    return result


def cuda_residency_attempt_payload() -> dict[str, Any]:
    return {
        "attempted": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "status": "blocked_before_mutation",
        "blockers": [
            "cuda_patch_observation_helper_exists_but_generation_state_is_not_cuda_resident",
            "policy_action_cache_and_hmem_are_not_owned_by_one_cuda_generation_call",
            "action_application_and_patch_update_cuda_kernels_are_not_integrated",
        ],
    }


def controlled_membership_attempt_payload() -> dict[str, Any]:
    return {
        "attempted": True,
        "status": "blocked_before_mutation",
        "blockers": [
            "resident_runtime_state_v9_membership_epoch_is_schema_only",
            "checkpoint_safe_birth_death_slot_reuse_not_owned_by_native_generation",
            "implementing_membership_without_monolithic_policy_state_would_risk_stale_hmem_cache",
        ],
    }


__all__ = [
    "CALL_GRAPH_SCHEMA_VERSION",
    "DECISION_POLICY_INTEGRATION_BLOCKED",
    "RESIDENT_STATE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "attempt_aten_policy_generation_integration",
    "attempt_object_free_policy_authority_integration",
    "border_counter_zero_payload",
    "controlled_membership_attempt_payload",
    "cuda_residency_attempt_payload",
    "execute_generation",
    "generation_call_graph_payload",
    "native_generation_support_blockers",
    "resident_runtime_state_v9_audit_payload",
]
