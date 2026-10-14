from __future__ import annotations

"""Fail-closed Phase E GPU qualification and bounded Stage1 unlock receipts.

The qualification receipt is evidence that the exact ten source-bound GPU
parity cases passed.  It authorizes only the dedicated two-run Stage1 probe.
The final receipt additionally binds two semantically identical real Stage1
runs.  Only the final receipt may authorize the normal bounded Stage1 caller.
"""

from copy import deepcopy
from dataclasses import dataclass, field, fields
from hashlib import sha256
import json
import math
import re
from typing import Any, Final, Mapping

from perf.cuda_architecture_b_phase_e_formal_gate import (
    FORMAL_GATE_NUMERIC_RULE_ID,
    NAMED_FORMAL_TOLERANCE_FIELDS,
    REQUIRED_PHASE_E_ARMS,
    SUPPORTED_LOGICAL_PROFILES,
)


PHASE_E_GPU_GATE_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_phase_e_gpu_gate_v1"
)
PHASE_E_GPU_QUALIFICATION_SCOPE: Final[str] = (
    "phase_e_same_source_gpu_parity_qualification_v1"
)
PHASE_E_GPU_FINAL_SCOPE: Final[str] = (
    "phase_e_bounded_stage1_two_run_unlock_v1"
)
PHASE_E_GPU_GATE_ERROR: Final[str] = "EIEL_PHASE_E_GPU_GATE_REJECTED"
PHASE_E_GENERATION_LANE_OWNER: Final[str] = (
    "single_cell_native_generation_phase_e_v1"
)
PHASE_E_MODULAR_TICK_LANE_OWNER: Final[str] = (
    "native_aten_device_resident_modular_tick_v1"
)
PHASE_E_DEVELOPMENT_MATURE_START_STEP: Final[int] = 20
PHASE_E_DEVELOPMENT_LATE_START_STEP: Final[int] = 40
PHASE_E_STAGE1_PROBE_RUN_IDS: Final[tuple[str, str]] = (
    "flat-one-repeat-1",
    "flat-one-repeat-2",
)
PHASE_E_RUNTIME_SOURCE_ROOTS: Final[tuple[str, ...]] = (
    "engine",
    "tools",
    "tests",
)
REQUIRED_GPU_TEST_NODE_IDS: Final[tuple[str, ...]] = (
    "tests/perf/test_eiel_cuda_architecture_b_modular_generation.py::"
    "test_cuda_default_disabled_path_is_bit_exact_to_generation_v3",
    "tests/perf/test_eiel_cuda_architecture_b_modular_generation.py::"
    "test_cuda_enabled_neutral_modular_bank_is_bit_exact_to_generation_v3",
    "tests/perf/test_eiel_cuda_architecture_b_modular_generation.py::"
    "test_cuda_enabled_structured_bank_changes_state_output_and_action_evidence",
    "tests/perf/test_eiel_phase_e_12_arm_cuda_parity.py::"
    "test_cuda_native_development_dependency_and_dormancy_reactivation_match_cpu",
    "tests/perf/test_eiel_phase_e_12_arm_cuda_parity.py::"
    "test_cuda_native_gated_projection_and_routing_match_cpu_without_rng_drift",
    "tests/perf/test_eiel_phase_e_12_arm_cuda_parity.py::"
    "test_cuda_native_structural_cost_matches_cpu_and_remains_reporting_only",
    "tests/perf/test_eiel_phase_e_12_arm_cuda_parity.py::"
    "test_cuda_prune_and_slot_reuse_matches_cpu_exactly_without_rng_drift",
    "tests/perf/test_eiel_phase_e_12_arm_cuda_parity.py::"
    "test_cuda_capacity_migration_first_continuation_matches_cpu_without_rng_drift",
    "tests/perf/test_eiel_phase_e_active_modular_checkpoint_cuda.py::"
    "test_enabled_modular_generation_checkpoint_resume_is_exact[b2-12x32-v1]",
    "tests/perf/test_eiel_phase_e_active_modular_checkpoint_cuda.py::"
    "test_enabled_modular_generation_checkpoint_resume_is_exact[b2-16x48-v1]",
)
PHASE_E_NATIVE_SOURCE_FILES: Final[tuple[str, ...]] = (
    "engine/agents/batched_policy.py",
    "engine/agents/modular_genome_v1.py",
    "engine/evolution/structural_development_v1.py",
    "engine/evolution/structural_gated_recurrent_v1.py",
    "engine/evolution/structural_observation_routing_v1.py",
    "engine/evolution/structural_output_routing_v1.py",
    "engine/evolution/structural_sustainable_growth_v1.py",
    "engine/native_runtime/architecture_b_reporting.py",
    "engine/native_runtime/checkpoint_digests.py",
    "engine/native_runtime/engine.py",
    "engine/native_runtime/flat_population_reporting.py",
    "engine/native_runtime/four_way_interchange.py",
    "engine/native_runtime/interchange_projection.py",
    "engine/native_runtime/lifecycle_events.py",
    "engine/native_runtime/public_metrics_overlay.py",
    "engine/perf/cuda_architecture_b_birth_transaction.py",
    "engine/perf/cuda_architecture_b_boundary.py",
    "engine/perf/cuda_architecture_b_generation.py",
    "engine/perf/cuda_architecture_b_generation_birth_state.py",
    "engine/perf/cuda_architecture_b_lifecycle.py",
    "engine/perf/cuda_architecture_b_modular_generation_contract.py",
    "engine/perf/cuda_architecture_b_modular_policy_step.py",
    "engine/perf/cuda_architecture_b_modular_state_bank.py",
    "engine/perf/cuda_architecture_b_modular_state_abi.py",
    "engine/perf/cuda_architecture_b_modular_tick.py",
    "engine/perf/cuda_architecture_b_modular_tick_native.py",
    "engine/perf/cuda_architecture_b_patch_context.py",
    "engine/perf/cuda_architecture_b_pcg64.py",
    "engine/perf/cuda_architecture_b_prebuilt.py",
    "engine/perf/cuda_architecture_b_reporting.py",
    "engine/perf/cuda_architecture_b_stage1b.py",
    "engine/perf/cuda_architecture_b_checkpoint.py",
    "engine/perf/cuda_architecture_b_child_policy.py",
    "engine/perf/cuda_architecture_b_torch_rng_state_pack_v1.cuh",
    "engine/perf/cuda_architecture_b_torch_rng_state_pack_v1.py",
    "engine/perf/cuda_architecture_b_phase_e_formal_composition.py",
    "engine/perf/cuda_architecture_b_phase_e_formal_gate.py",
    "engine/perf/cuda_architecture_b_phase_e_gpu_gate.py",
    "engine/perf/cxxbackbone10_resident_state_owner.py",
    "engine/perf/native/cuda_architecture_b_active_mechanics.cu",
    "engine/perf/native/cuda_architecture_b_asexual_birth_plan.cu",
    "engine/perf/native/cuda_architecture_b_birth_lifecycle_events.cu",
    "engine/perf/native/cuda_architecture_b_birth_microcycle.cpp",
    "engine/perf/native/cuda_architecture_b_birth_transaction.cpp",
    "engine/perf/native/cuda_architecture_b_child_drives.cu",
    "engine/perf/native/cuda_architecture_b_child_policy.cu",
    "engine/perf/native/cuda_architecture_b_death_class_concat.cu",
    "engine/perf/native/cuda_architecture_b_death_lifecycle_events.cu",
    "engine/perf/native/cuda_architecture_b_fixed_mlp.cu",
    "engine/perf/native/cuda_architecture_b_flat_food_spawn.cu",
    "engine/perf/native/cuda_architecture_b_generation.cpp",
    "engine/perf/native/cuda_architecture_b_generation_context.hpp",
    "engine/perf/native/cuda_architecture_b_generation_multicell.cpp",
    "engine/perf/native/cuda_architecture_b_lifecycle.cu",
    "engine/perf/native/cuda_architecture_b_modular_tick.cpp",
    "engine/perf/native/cuda_architecture_b_modular_tick_bindings.cpp",
    "engine/perf/native/cuda_architecture_b_modular_tick.hpp",
    "engine/perf/native/cuda_architecture_b_patch_context.cu",
    "engine/perf/native/cuda_architecture_b_pcg64.cuh",
    "engine/perf/native/cuda_architecture_b_policy_cache.cu",
    "engine/perf/native/cuda_architecture_b_turnover.cu",
    "engine/perf/native/cxxbackbone10_resident_state_owner.cpp",
    "engine/perf/native/cxxbackbone8r_generation_island_cuda.cu",
    "tools/scripts/build_eiel_cuda_architecture_b_extension.py",
    "tools/scripts/eiel_architecture_b_run_many_parity.py",
    "tools/scripts/eiel_cxxbackbone8r_checkpoint_continuation.py",
    "tools/scripts/eiel_cxxbackbone8r_first_milestone.py",
    "tools/scripts/eiel_cuda_architecture_b_stage1_timing.py",
    "tools/scripts/eiel_flat_generation_four_way_parity.py",
    "tools/scripts/eiel_flat_patch_cpu_oracle_gate.py",
    "tools/scripts/eiel_phase_e_gpu_receipt.py",
    "tools/scripts/eiel_phase_e_gpu_suite.py",
    "tools/scripts/eiel_phase_e_stage1_probe.py",
)
PHASE_E_TEST_SOURCE_FILES: Final[tuple[str, ...]] = (
    "tests/perf/test_eiel_cuda_architecture_b_generation.py",
    "tests/perf/test_eiel_cuda_architecture_b_phase_e_formal_composition.py",
    "tests/perf/test_eiel_cuda_architecture_b_phase_e_formal_gate.py",
    "tests/perf/test_eiel_cuda_architecture_b_phase_e_gpu_gate.py",
    "tests/perf/test_eiel_phase_e_formal_composition_checkpoint.py",
    "tests/perf/test_eiel_phase_e_formal_composition_interactions.py",
    "tests/perf/test_eiel_cuda_architecture_b_modular_generation.py",
    "tests/perf/test_eiel_cuda_architecture_b_modular_generation_contract.py",
    "tests/perf/test_eiel_cuda_architecture_b_modular_policy_step.py",
    "tests/perf/test_eiel_cuda_architecture_b_modular_state_abi.py",
    "tests/perf/test_eiel_cuda_architecture_b_modular_state_bank.py",
    "tests/perf/test_eiel_cuda_architecture_b_modular_checkpoint_owner.py",
    "tests/perf/test_eiel_phase_e_12_arm_cuda_parity.py",
    "tests/perf/test_eiel_phase_e_active_modular_checkpoint.py",
    "tests/perf/test_eiel_phase_e_active_modular_checkpoint_cuda.py",
    "tests/perf/test_eiel_phase_e_modular_checkpoint_interchange.py",
    "tests/perf/test_eiel_phase_e_modular_checkpoint_resume.py",
    "tests/perf/test_eiel_phase_e_modular_tick_fixture_v1.py",
    "tests/perf/test_eiel_cuda_architecture_b_modular_tick.py",
    "tests/perf/test_eiel_cuda_architecture_b_modular_lifecycle.py",
    "tests/perf/test_eiel_cuda_architecture_b_modular_capacity_epoch.py",
    "tests/perf/test_eiel_cuda_architecture_b_reporting.py",
    "tests/tools/test_eiel_cuda_architecture_b_stage1_timing.py",
    "tests/tools/test_eiel_phase_e_gpu_receipt.py",
    "tests/tools/test_eiel_phase_e_gpu_suite.py",
    "tests/tools/test_eiel_phase_e_receipt_bundle.py",
    "tests/tools/test_eiel_phase_e_stage1_probe.py",
)
PHASE_E_FORMAL_CPU_ALLOWED_SKIP_NODE_IDS: Final[tuple[str, ...]] = (
    "tests/perf/test_eiel_cuda_architecture_b_generation.py::"
    "test_deterministic_generation_matches_cpu_action_boundary[False]",
    "tests/perf/test_eiel_cuda_architecture_b_generation.py::"
    "test_deterministic_generation_matches_cpu_action_boundary[True]",
    "tests/perf/test_eiel_cuda_architecture_b_generation.py::"
    "test_active_mechanics_is_composed_before_lifecycle",
    "tests/perf/test_eiel_cuda_architecture_b_generation.py::"
    "test_patch_context_composition_matches_standalone_device_oracle",
    "tests/perf/test_eiel_cuda_architecture_b_generation.py::"
    "test_patch_context_precedes_action_and_defers_new_emission",
    "tests/perf/test_eiel_cuda_architecture_b_generation.py::"
    "test_stochastic_multistep_actions_and_rng_state_match_numpy_pcg64[False]",
    "tests/perf/test_eiel_cuda_architecture_b_generation.py::"
    "test_stochastic_multistep_actions_and_rng_state_match_numpy_pcg64[True]",
    "tests/perf/test_eiel_cuda_architecture_b_generation.py::"
    "test_stochastic_rng_and_state_continue_across_generation_calls",
    "tests/perf/test_eiel_cuda_architecture_b_generation.py::"
    "test_action_time_death_cull_matches_composed_device_oracle",
    "tests/perf/test_eiel_cuda_architecture_b_generation.py::"
    "test_death_event_capacity_failure_is_atomic_and_fail_closed",
    "tests/perf/test_eiel_cuda_architecture_b_generation.py::"
    "test_generation_extension_name_is_bound_to_all_source_hashes",
    "tests/perf/test_eiel_cuda_architecture_b_modular_generation.py::"
    "test_cuda_default_disabled_path_is_bit_exact_to_generation_v3",
    "tests/perf/test_eiel_cuda_architecture_b_modular_generation.py::"
    "test_cuda_enabled_neutral_modular_bank_is_bit_exact_to_generation_v3",
    "tests/perf/test_eiel_cuda_architecture_b_modular_generation.py::"
    "test_cuda_enabled_structured_bank_changes_state_output_and_action_evidence",
    "tests/perf/test_eiel_phase_e_12_arm_cuda_parity.py::"
    "test_cuda_native_development_dependency_and_dormancy_reactivation_match_cpu",
    "tests/perf/test_eiel_phase_e_12_arm_cuda_parity.py::"
    "test_cuda_native_gated_projection_and_routing_match_cpu_without_rng_drift",
    "tests/perf/test_eiel_phase_e_12_arm_cuda_parity.py::"
    "test_cuda_native_structural_cost_matches_cpu_and_remains_reporting_only",
    "tests/perf/test_eiel_phase_e_12_arm_cuda_parity.py::"
    "test_cuda_prune_and_slot_reuse_matches_cpu_exactly_without_rng_drift",
    "tests/perf/test_eiel_phase_e_12_arm_cuda_parity.py::"
    "test_cuda_capacity_migration_first_continuation_matches_cpu_without_rng_drift",
    "tests/perf/test_eiel_phase_e_active_modular_checkpoint_cuda.py::"
    "test_enabled_modular_generation_checkpoint_resume_is_exact[b2-12x32-v1]",
    "tests/perf/test_eiel_phase_e_active_modular_checkpoint_cuda.py::"
    "test_enabled_modular_generation_checkpoint_resume_is_exact[b2-16x48-v1]",
    "tests/perf/test_eiel_cuda_architecture_b_modular_tick.py::"
    "test_tiny_cuda_tick_matches_cpu_tensor_results",
    "tests/perf/test_eiel_cuda_architecture_b_modular_lifecycle.py::"
    "test_tiny_cuda_prune_and_reuse_preserves_resident_order",
    "tests/perf/test_eiel_cuda_architecture_b_modular_capacity_epoch.py::"
    "test_optional_tiny_cuda_validation_not_run",
    "tests/perf/test_eiel_cuda_architecture_b_reporting.py::"
    "test_live_reporting_matches_cpu_owner_public_generation_outputs[flat]",
    "tests/perf/test_eiel_cuda_architecture_b_reporting.py::"
    "test_live_reporting_matches_cpu_owner_public_generation_outputs[patch-p2]",
)

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}")
_BRANCH_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
_AUTHORIZATION_MARKER: Final[object] = object()

_QUALIFICATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "receipt_kind",
        "scope",
        "source_branch",
        "source_commit",
        "test_commit",
        "formal_receipt_sha256",
        "formal_raw_composition_sha256",
        "native_source_bundle_sha256",
        "test_source_bundle_sha256",
        "generation_artifact_sha256",
        "generation_artifact_manifest_sha256",
        "modular_tick_artifact_sha256",
        "gpu_execution_manifest_sha256",
        "junit_sha256",
        "log_sha256",
        "environment",
        "logical_profiles",
        "test_results",
        "test_counts",
        "arms",
        "lane_owners",
        "hidden_fallback_count",
        "modular_rng_draw_count",
        "numeric_rule",
        "claims",
        "receipt_sha256",
    }
)
_FINAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "receipt_kind",
        "scope",
        "source_branch",
        "source_commit",
        "test_commit",
        "formal_receipt_sha256",
        "formal_raw_composition_sha256",
        "native_source_bundle_sha256",
        "test_source_bundle_sha256",
        "generation_artifact_sha256",
        "generation_artifact_manifest_sha256",
        "modular_tick_artifact_sha256",
        "gpu_execution_manifest_sha256",
        "qualification_receipt_sha256",
        "logical_profiles",
        "stage1_runs",
        "claims",
        "receipt_sha256",
    }
)
_TEST_RESULT_FIELDS: Final[frozenset[str]] = frozenset(
    {"node_id", "status", "evidence_sha256"}
)
_TEST_COUNT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "collected",
        "passed",
        "failures",
        "errors",
        "skipped",
        "xfailed",
        "xpassed",
    }
)
_ARM_FIELDS: Final[frozenset[str]] = frozenset(
    {"status", "evidence_sha256", "first_mismatch"}
)
_ENVIRONMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "python_version",
        "numpy_version",
        "torch_version",
        "torch_cuda_version",
        "device_name",
        "compute_capability",
        "device_total_memory_bytes",
    }
)
_LANE_FIELDS: Final[frozenset[str]] = frozenset(
    {"generation", "modular_tick", "hidden_fallback"}
)
_NUMERIC_RULE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "rule_id",
        "review_status",
        "atol",
        "rtol",
        "named_float_tolerance_fields",
        "all_other_fields_exact",
    }
)
_QUALIFICATION_CLAIMS: Final[dict[str, bool]] = {
    "gpu_parity_present": True,
    "bounded_gpu_formal_composition_qualified": True,
    "stage1_probe_authorized": True,
    "bounded_stage1_owner_qualified": False,
    "caller_locked": True,
    "full_phase_e_qualified": False,
    "phase_f_authorized": False,
    "campaign_qualified": False,
    "speed_claimed": False,
}
_FINAL_CLAIMS: Final[dict[str, bool]] = {
    "gpu_parity_present": True,
    "bounded_gpu_formal_composition_qualified": True,
    "stage1_probe_authorized": False,
    "bounded_stage1_owner_qualified": True,
    "caller_locked": False,
    "full_phase_e_qualified": True,
    "phase_f_authorized": True,
    "campaign_qualified": False,
    "speed_claimed": False,
}
_STAGE1_RUN_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "run_id",
        "mode",
        "profile_id",
        "status",
        "output_sha256",
        "semantic_sha256",
        "checkpoint_sha256",
        "public_output_sha256",
        "action_sha256",
        "rng_sha256",
        "identity_sha256",
        "slot_map_sha256",
        "hidden_fallback_count",
        "lane_owner",
    }
)


class PhaseEGpuGateError(RuntimeError):
    """The GPU qualification or final Stage1 receipt is invalid."""


@dataclass(frozen=True)
class PhaseEGpuGateBindings:
    source_branch: str
    source_commit: str
    test_commit: str
    formal_receipt_sha256: str
    formal_raw_composition_sha256: str
    native_source_bundle_sha256: str
    test_source_bundle_sha256: str
    generation_artifact_sha256: str
    generation_artifact_manifest_sha256: str
    modular_tick_artifact_sha256: str
    gpu_execution_manifest_sha256: str
    junit_sha256: str
    log_sha256: str


@dataclass(frozen=True)
class PhaseEGpuAuthorization:
    kind: str
    receipt_sha256: str
    source_commit: str
    generation_artifact_sha256: str
    qualification_receipt_sha256: str | None
    logical_profiles: tuple[str, ...]
    bounded_stage1_owner_qualified: bool
    phase_f_authorized: bool
    _marker: object = field(repr=False, compare=False)


_MINTED_AUTHORIZATIONS: Final[
    dict[int, tuple[PhaseEGpuAuthorization, str]]
] = {}


def _fail(message: str) -> PhaseEGpuGateError:
    return PhaseEGpuGateError(f"{PHASE_E_GPU_GATE_ERROR}: {message}")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _fail("receipt must be canonical JSON data") from exc


def phase_e_gpu_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    payload = deepcopy(dict(receipt))
    payload.pop("receipt_sha256", None)
    return sha256(_canonical_bytes(payload)).hexdigest()


def _authorization_fingerprint(authorization: PhaseEGpuAuthorization) -> str:
    payload = {
        item.name: getattr(authorization, item.name)
        for item in fields(authorization)
        if item.name != "_marker"
    }
    return sha256(_canonical_bytes(payload)).hexdigest()


def _require_fields(
    value: Any,
    expected: frozenset[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{name} must be a mapping")
    observed = set(value)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        extra = sorted(observed - set(expected))
        raise _fail(f"{name} fields mismatch missing={missing} extra={extra}")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _fail(f"{name} must be a lowercase SHA-256")
    return value


def _require_commit(value: Any, *, name: str) -> str:
    if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
        raise _fail(f"{name} must be a full lowercase Git commit")
    return value


def _require_branch(value: Any, *, name: str) -> str:
    if type(value) is not str or _BRANCH_RE.fullmatch(value) is None:
        raise _fail(f"{name} must be a branch name")
    return value


def _require_exact_int(value: Any, expected: int, *, name: str) -> None:
    if type(value) is not int or value != expected:
        raise _fail(f"{name} must equal {expected}")


def _require_exact_bool(value: Any, expected: bool, *, name: str) -> None:
    if type(value) is not bool or value is not expected:
        raise _fail(f"{name} must be {expected}")


def _validate_common_bindings(
    receipt: Mapping[str, Any],
    bindings: PhaseEGpuGateBindings,
) -> None:
    if not isinstance(bindings, PhaseEGpuGateBindings):
        raise TypeError("bindings must be PhaseEGpuGateBindings")
    _require_branch(receipt["source_branch"], name="source_branch")
    _require_commit(receipt["source_commit"], name="source_commit")
    _require_commit(receipt["test_commit"], name="test_commit")
    exact_values = {
        "source_branch": bindings.source_branch,
        "source_commit": bindings.source_commit,
        "test_commit": bindings.test_commit,
        "formal_receipt_sha256": bindings.formal_receipt_sha256,
        "formal_raw_composition_sha256": bindings.formal_raw_composition_sha256,
        "native_source_bundle_sha256": bindings.native_source_bundle_sha256,
        "test_source_bundle_sha256": bindings.test_source_bundle_sha256,
        "generation_artifact_sha256": bindings.generation_artifact_sha256,
        "generation_artifact_manifest_sha256": (
            bindings.generation_artifact_manifest_sha256
        ),
        "modular_tick_artifact_sha256": bindings.modular_tick_artifact_sha256,
        "gpu_execution_manifest_sha256": bindings.gpu_execution_manifest_sha256,
    }
    for name, expected in exact_values.items():
        if name.endswith("sha256"):
            _require_sha256(receipt[name], name=name)
        if receipt[name] != expected:
            raise _fail(f"{name} does not match the external binding")


def _validate_profiles(value: Any) -> tuple[str, ...]:
    if type(value) is not list or tuple(value) != SUPPORTED_LOGICAL_PROFILES:
        raise _fail("logical_profiles must be the exact supported profile order")
    return tuple(value)


def _validate_numeric_rule(value: Any) -> None:
    numeric = _require_fields(value, _NUMERIC_RULE_FIELDS, name="numeric_rule")
    if numeric["rule_id"] != FORMAL_GATE_NUMERIC_RULE_ID:
        raise _fail("numeric rule id mismatch")
    if numeric["review_status"] != "formally_accepted":
        raise _fail("numeric rule is not formally accepted")
    atol = numeric["atol"]
    rtol = numeric["rtol"]
    if type(atol) is not float or not math.isfinite(atol) or atol != 2.0e-7:
        raise _fail("numeric atol must equal 2e-7")
    if type(rtol) is not float or not math.isfinite(rtol) or rtol != 0.0:
        raise _fail("numeric rtol must equal 0")
    if (
        type(numeric["named_float_tolerance_fields"]) is not list
        or tuple(numeric["named_float_tolerance_fields"])
        != NAMED_FORMAL_TOLERANCE_FIELDS
    ):
        raise _fail("named float tolerance fields mismatch")
    _require_exact_bool(
        numeric["all_other_fields_exact"],
        True,
        name="numeric_rule.all_other_fields_exact",
    )


def _validate_claims(value: Any, expected: Mapping[str, bool]) -> None:
    claims = _require_fields(value, frozenset(expected), name="claims")
    for name, required in expected.items():
        _require_exact_bool(claims[name], required, name=f"claims.{name}")


def _validate_self_digest(receipt: Mapping[str, Any]) -> str:
    observed = _require_sha256(receipt["receipt_sha256"], name="receipt_sha256")
    expected = phase_e_gpu_receipt_sha256(receipt)
    if observed != expected:
        raise _fail("receipt self digest mismatch")
    return observed


def _authorization(
    *,
    kind: str,
    receipt_sha256: str,
    source_commit: str,
    generation_artifact_sha256: str,
    qualification_receipt_sha256: str | None,
    logical_profiles: tuple[str, ...],
    bounded_stage1_owner_qualified: bool,
    phase_f_authorized: bool,
) -> PhaseEGpuAuthorization:
    authorization = PhaseEGpuAuthorization(
        kind=kind,
        receipt_sha256=receipt_sha256,
        source_commit=source_commit,
        generation_artifact_sha256=generation_artifact_sha256,
        qualification_receipt_sha256=qualification_receipt_sha256,
        logical_profiles=logical_profiles,
        bounded_stage1_owner_qualified=bounded_stage1_owner_qualified,
        phase_f_authorized=phase_f_authorized,
        _marker=_AUTHORIZATION_MARKER,
    )
    _MINTED_AUTHORIZATIONS[id(authorization)] = (
        authorization,
        _authorization_fingerprint(authorization),
    )
    return authorization


def require_phase_e_gpu_authorization(
    authorization: Any,
    *,
    final: bool,
    profile_id: str,
) -> PhaseEGpuAuthorization:
    """Reject fabricated Booleans, mappings, dataclasses, and the wrong receipt tier."""

    minted = _MINTED_AUTHORIZATIONS.get(id(authorization))
    if (
        type(authorization) is not PhaseEGpuAuthorization
        or authorization._marker is not _AUTHORIZATION_MARKER
        or minted is None
        or minted[0] is not authorization
        or minted[1] != _authorization_fingerprint(authorization)
    ):
        raise _fail("authorization was not minted by the GPU receipt validator")
    expected_kind = "final" if final else "qualification"
    if authorization.kind != expected_kind:
        raise _fail(f"{expected_kind} authorization is required")
    if profile_id not in authorization.logical_profiles:
        raise _fail("profile is outside the receipt envelope")
    if final and not authorization.bounded_stage1_owner_qualified:
        raise _fail("final receipt does not qualify the bounded Stage1 owner")
    if not final and authorization.bounded_stage1_owner_qualified:
        raise _fail("qualification receipt overclaims normal Stage1 ownership")
    return authorization


def validate_phase_e_gpu_qualification_receipt(
    receipt: Mapping[str, Any],
    *,
    bindings: PhaseEGpuGateBindings,
) -> PhaseEGpuAuthorization:
    receipt = _require_fields(
        receipt,
        _QUALIFICATION_FIELDS,
        name="qualification receipt",
    )
    if receipt["schema_version"] != PHASE_E_GPU_GATE_SCHEMA_VERSION:
        raise _fail("qualification schema version mismatch")
    if receipt["receipt_kind"] != "qualification":
        raise _fail("receipt kind must be qualification")
    if receipt["scope"] != PHASE_E_GPU_QUALIFICATION_SCOPE:
        raise _fail("qualification scope mismatch")
    _validate_common_bindings(receipt, bindings)
    if receipt["junit_sha256"] != bindings.junit_sha256:
        raise _fail("JUnit digest does not match the external binding")
    if receipt["log_sha256"] != bindings.log_sha256:
        raise _fail("log digest does not match the external binding")
    _require_sha256(receipt["junit_sha256"], name="junit_sha256")
    _require_sha256(receipt["log_sha256"], name="log_sha256")
    profiles = _validate_profiles(receipt["logical_profiles"])

    environment = _require_fields(
        receipt["environment"],
        _ENVIRONMENT_FIELDS,
        name="environment",
    )
    for name in (
        "python_version",
        "numpy_version",
        "torch_version",
        "torch_cuda_version",
        "device_name",
        "compute_capability",
    ):
        if type(environment[name]) is not str or not environment[name]:
            raise _fail(f"environment.{name} must be a nonempty string")
    if (
        type(environment["device_total_memory_bytes"]) is not int
        or environment["device_total_memory_bytes"] <= 0
    ):
        raise _fail("environment.device_total_memory_bytes must be positive")

    counts = _require_fields(
        receipt["test_counts"],
        _TEST_COUNT_FIELDS,
        name="test_counts",
    )
    expected_counts = {
        "collected": len(REQUIRED_GPU_TEST_NODE_IDS),
        "passed": len(REQUIRED_GPU_TEST_NODE_IDS),
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    for name, expected in expected_counts.items():
        _require_exact_int(counts[name], expected, name=f"test_counts.{name}")

    results = receipt["test_results"]
    if type(results) is not list or len(results) != len(REQUIRED_GPU_TEST_NODE_IDS):
        raise _fail("test_results must contain exactly ten results")
    node_ids: list[str] = []
    for index, result in enumerate(results):
        result = _require_fields(
            result,
            _TEST_RESULT_FIELDS,
            name=f"test_results[{index}]",
        )
        if type(result["node_id"]) is not str:
            raise _fail(f"test_results[{index}].node_id must be a string")
        node_ids.append(result["node_id"])
        if result["status"] != "passed":
            raise _fail(f"test result is not passed: {result['node_id']}")
        if result["evidence_sha256"] != bindings.junit_sha256:
            raise _fail("test result evidence digest mismatch")
    if tuple(node_ids) != REQUIRED_GPU_TEST_NODE_IDS:
        raise _fail("test node IDs or order do not match the exact qualification set")
    if len(set(node_ids)) != len(node_ids):
        raise _fail("duplicate GPU test node")

    arms = receipt["arms"]
    if not isinstance(arms, Mapping) or set(arms) != set(REQUIRED_PHASE_E_ARMS):
        raise _fail("Phase E arm set mismatch")
    for name in REQUIRED_PHASE_E_ARMS:
        arm = _require_fields(arms[name], _ARM_FIELDS, name=f"arms.{name}")
        if arm["status"] != "gpu_parity_passed":
            raise _fail(f"Phase E arm did not pass: {name}")
        if arm["evidence_sha256"] != bindings.junit_sha256:
            raise _fail(f"Phase E arm evidence mismatch: {name}")
        if arm["first_mismatch"] is not None:
            raise _fail(f"Phase E arm records a mismatch: {name}")

    lanes = _require_fields(receipt["lane_owners"], _LANE_FIELDS, name="lane_owners")
    if lanes["generation"] != PHASE_E_GENERATION_LANE_OWNER:
        raise _fail("generation lane owner mismatch")
    if lanes["modular_tick"] != PHASE_E_MODULAR_TICK_LANE_OWNER:
        raise _fail("modular tick lane owner mismatch")
    _require_exact_bool(
        lanes["hidden_fallback"], False, name="lane_owners.hidden_fallback"
    )
    _require_exact_int(
        receipt["hidden_fallback_count"], 0, name="hidden_fallback_count"
    )
    _require_exact_int(
        receipt["modular_rng_draw_count"], 0, name="modular_rng_draw_count"
    )
    _validate_numeric_rule(receipt["numeric_rule"])
    _validate_claims(receipt["claims"], _QUALIFICATION_CLAIMS)
    digest = _validate_self_digest(receipt)
    return _authorization(
        kind="qualification",
        receipt_sha256=digest,
        source_commit=bindings.source_commit,
        generation_artifact_sha256=bindings.generation_artifact_sha256,
        qualification_receipt_sha256=None,
        logical_profiles=profiles,
        bounded_stage1_owner_qualified=False,
        phase_f_authorized=False,
    )


def validate_phase_e_gpu_final_receipt(
    receipt: Mapping[str, Any],
    *,
    bindings: PhaseEGpuGateBindings,
    qualification_receipt: Mapping[str, Any],
) -> PhaseEGpuAuthorization:
    qualification = validate_phase_e_gpu_qualification_receipt(
        qualification_receipt,
        bindings=bindings,
    )
    receipt = _require_fields(receipt, _FINAL_FIELDS, name="final receipt")
    if receipt["schema_version"] != PHASE_E_GPU_GATE_SCHEMA_VERSION:
        raise _fail("final schema version mismatch")
    if receipt["receipt_kind"] != "final":
        raise _fail("receipt kind must be final")
    if receipt["scope"] != PHASE_E_GPU_FINAL_SCOPE:
        raise _fail("final scope mismatch")
    _validate_common_bindings(receipt, bindings)
    profiles = _validate_profiles(receipt["logical_profiles"])
    if receipt["qualification_receipt_sha256"] != qualification.receipt_sha256:
        raise _fail("final receipt does not bind the accepted qualification receipt")

    runs = receipt["stage1_runs"]
    if type(runs) is not list or len(runs) != len(PHASE_E_STAGE1_PROBE_RUN_IDS):
        raise _fail("final receipt requires exactly two Stage1 runs")
    semantic_fields = (
        "semantic_sha256",
        "checkpoint_sha256",
        "public_output_sha256",
        "action_sha256",
        "rng_sha256",
        "identity_sha256",
        "slot_map_sha256",
    )
    first: Mapping[str, Any] | None = None
    for index, run in enumerate(runs):
        run = _require_fields(run, _STAGE1_RUN_FIELDS, name=f"stage1_runs[{index}]")
        if run["run_id"] != PHASE_E_STAGE1_PROBE_RUN_IDS[index]:
            raise _fail("Stage1 run ID or order mismatch")
        if run["mode"] != "flat-one":
            raise _fail("only flat-one is qualified by the final receipt")
        if run["profile_id"] not in SUPPORTED_LOGICAL_PROFILES:
            raise _fail("Stage1 run profile is outside the supported envelope")
        if run["status"] != "passed":
            raise _fail("Stage1 probe run did not pass")
        for name in ("output_sha256", *semantic_fields):
            _require_sha256(run[name], name=f"stage1_runs[{index}].{name}")
        _require_exact_int(
            run["hidden_fallback_count"],
            0,
            name=f"stage1_runs[{index}].hidden_fallback_count",
        )
        if run["lane_owner"] != PHASE_E_GENERATION_LANE_OWNER:
            raise _fail("Stage1 run lane owner mismatch")
        if first is None:
            first = run
        else:
            if run["profile_id"] != first["profile_id"]:
                raise _fail("Stage1 repeat profile mismatch")
            if run["output_sha256"] == first["output_sha256"]:
                raise _fail("Stage1 repeats must have distinct output identities")
            for name in semantic_fields:
                if run[name] != first[name]:
                    raise _fail(f"Stage1 repeat semantic mismatch: {name}")

    _validate_claims(receipt["claims"], _FINAL_CLAIMS)
    digest = _validate_self_digest(receipt)
    return _authorization(
        kind="final",
        receipt_sha256=digest,
        source_commit=bindings.source_commit,
        generation_artifact_sha256=bindings.generation_artifact_sha256,
        qualification_receipt_sha256=qualification.receipt_sha256,
        logical_profiles=(str(first["profile_id"]),),
        bounded_stage1_owner_qualified=True,
        phase_f_authorized=True,
    )


__all__ = [
    "PHASE_E_GENERATION_LANE_OWNER",
    "PHASE_E_GPU_FINAL_SCOPE",
    "PHASE_E_GPU_GATE_ERROR",
    "PHASE_E_GPU_GATE_SCHEMA_VERSION",
    "PHASE_E_GPU_QUALIFICATION_SCOPE",
    "PHASE_E_MODULAR_TICK_LANE_OWNER",
    "PHASE_E_NATIVE_SOURCE_FILES",
    "PHASE_E_RUNTIME_SOURCE_ROOTS",
    "PHASE_E_STAGE1_PROBE_RUN_IDS",
    "PHASE_E_TEST_SOURCE_FILES",
    "REQUIRED_GPU_TEST_NODE_IDS",
    "REQUIRED_PHASE_E_ARMS",
    "SUPPORTED_LOGICAL_PROFILES",
    "PhaseEGpuAuthorization",
    "PhaseEGpuGateBindings",
    "PhaseEGpuGateError",
    "phase_e_gpu_receipt_sha256",
    "require_phase_e_gpu_authorization",
    "validate_phase_e_gpu_final_receipt",
    "validate_phase_e_gpu_qualification_receipt",
]
