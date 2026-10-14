from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from perf.cuda_architecture_b_phase_e_gpu_gate import (
    PHASE_E_GENERATION_LANE_OWNER,
    PHASE_E_GPU_FINAL_SCOPE,
    PHASE_E_GPU_GATE_SCHEMA_VERSION,
    PHASE_E_GPU_QUALIFICATION_SCOPE,
    PHASE_E_MODULAR_TICK_LANE_OWNER,
    PHASE_E_STAGE1_PROBE_RUN_IDS,
    REQUIRED_GPU_TEST_NODE_IDS,
    REQUIRED_PHASE_E_ARMS,
    SUPPORTED_LOGICAL_PROFILES,
    PhaseEGpuGateError,
    PhaseEGpuGateBindings,
    phase_e_gpu_receipt_sha256,
    require_phase_e_gpu_authorization,
    validate_phase_e_gpu_final_receipt,
    validate_phase_e_gpu_qualification_receipt,
)
from perf.cuda_architecture_b_phase_e_formal_gate import (
    FORMAL_GATE_NUMERIC_RULE_ID,
    NAMED_FORMAL_TOLERANCE_FIELDS,
)


SOURCE_BRANCH = "integration/evolutionary-capacity-phase-e"
SOURCE_COMMIT = "1" * 40
TEST_COMMIT = "2" * 40
FORMAL_RECEIPT_SHA256 = "3" * 64
FORMAL_RAW_COMPOSITION_SHA256 = "4" * 64
NATIVE_SOURCE_BUNDLE_SHA256 = "5" * 64
TEST_SOURCE_BUNDLE_SHA256 = "6" * 64
GENERATION_ARTIFACT_SHA256 = "7" * 64
GENERATION_ARTIFACT_MANIFEST_SHA256 = "8" * 64
MODULAR_TICK_ARTIFACT_SHA256 = "9" * 64
GPU_EXECUTION_MANIFEST_SHA256 = "d" * 64
JUNIT_SHA256 = "a" * 64
LOG_SHA256 = "b" * 64
FINAL_LANE = "flat-one"


def _bindings() -> PhaseEGpuGateBindings:
    return PhaseEGpuGateBindings(
        source_branch=SOURCE_BRANCH,
        source_commit=SOURCE_COMMIT,
        test_commit=TEST_COMMIT,
        formal_receipt_sha256=FORMAL_RECEIPT_SHA256,
        formal_raw_composition_sha256=FORMAL_RAW_COMPOSITION_SHA256,
        native_source_bundle_sha256=NATIVE_SOURCE_BUNDLE_SHA256,
        test_source_bundle_sha256=TEST_SOURCE_BUNDLE_SHA256,
        generation_artifact_sha256=GENERATION_ARTIFACT_SHA256,
        generation_artifact_manifest_sha256=GENERATION_ARTIFACT_MANIFEST_SHA256,
        modular_tick_artifact_sha256=MODULAR_TICK_ARTIFACT_SHA256,
        gpu_execution_manifest_sha256=GPU_EXECUTION_MANIFEST_SHA256,
        junit_sha256=JUNIT_SHA256,
        log_sha256=LOG_SHA256,
    )


def _bound_fields() -> dict[str, str]:
    bindings = _bindings()
    return {
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
        "junit_sha256": bindings.junit_sha256,
        "log_sha256": bindings.log_sha256,
    }


def _qualification_receipt() -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema_version": PHASE_E_GPU_GATE_SCHEMA_VERSION,
        "receipt_kind": "qualification",
        "scope": PHASE_E_GPU_QUALIFICATION_SCOPE,
        **_bound_fields(),
        "environment": {
            "python_version": "3.12.3",
            "numpy_version": "2.1.3",
            "torch_version": "2.7.1",
            "torch_cuda_version": "12.8",
            "device_name": "test-device",
            "compute_capability": "7.5",
            "device_total_memory_bytes": 6_442_450_944,
        },
        "logical_profiles": list(SUPPORTED_LOGICAL_PROFILES),
        "test_results": [
            {
                "node_id": node_id,
                "status": "passed",
                "evidence_sha256": JUNIT_SHA256,
            }
            for node_id in REQUIRED_GPU_TEST_NODE_IDS
        ],
        "test_counts": {
            "collected": 10,
            "passed": 10,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
        },
        "arms": {
            arm: {
                "status": "gpu_parity_passed",
                "evidence_sha256": JUNIT_SHA256,
                "first_mismatch": None,
            }
            for arm in REQUIRED_PHASE_E_ARMS
        },
        "lane_owners": {
            "generation": PHASE_E_GENERATION_LANE_OWNER,
            "modular_tick": PHASE_E_MODULAR_TICK_LANE_OWNER,
            "hidden_fallback": False,
        },
        "numeric_rule": {
            "rule_id": FORMAL_GATE_NUMERIC_RULE_ID,
            "review_status": "formally_accepted",
            "atol": 2.0e-7,
            "rtol": 0.0,
            "named_float_tolerance_fields": list(NAMED_FORMAL_TOLERANCE_FIELDS),
            "all_other_fields_exact": True,
        },
        "hidden_fallback_count": 0,
        "modular_rng_draw_count": 0,
        "claims": {
            "gpu_parity_present": True,
            "bounded_gpu_formal_composition_qualified": True,
            "stage1_probe_authorized": True,
            "bounded_stage1_owner_qualified": False,
            "caller_locked": True,
            "full_phase_e_qualified": False,
            "phase_f_authorized": False,
            "campaign_qualified": False,
            "speed_claimed": False,
        },
    }
    receipt["receipt_sha256"] = phase_e_gpu_receipt_sha256(receipt)
    return receipt


def _stage1_run(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": "passed",
        "mode": FINAL_LANE,
        "profile_id": SUPPORTED_LOGICAL_PROFILES[0],
        "output_sha256": (
            "2" * 64
            if run_id == PHASE_E_STAGE1_PROBE_RUN_IDS[0]
            else "5" * 64
        ),
        "semantic_sha256": "c" * 64,
        "checkpoint_sha256": "e" * 64,
        "public_output_sha256": "f" * 64,
        "action_sha256": "0" * 64,
        "rng_sha256": "1" * 64,
        "identity_sha256": "3" * 64,
        "slot_map_sha256": "4" * 64,
        "hidden_fallback_count": 0,
        "lane_owner": PHASE_E_GENERATION_LANE_OWNER,
    }


def _final_receipt(
    qualification_receipt: dict[str, object],
) -> dict[str, object]:
    final_bound_fields = {
        name: value
        for name, value in _bound_fields().items()
        if name not in {"junit_sha256", "log_sha256"}
    }
    receipt: dict[str, object] = {
        "schema_version": PHASE_E_GPU_GATE_SCHEMA_VERSION,
        "receipt_kind": "final",
        "scope": PHASE_E_GPU_FINAL_SCOPE,
        **final_bound_fields,
        "qualification_receipt_sha256": qualification_receipt["receipt_sha256"],
        "logical_profiles": list(SUPPORTED_LOGICAL_PROFILES),
        "stage1_runs": [
            _stage1_run(PHASE_E_STAGE1_PROBE_RUN_IDS[0]),
            _stage1_run(PHASE_E_STAGE1_PROBE_RUN_IDS[1]),
        ],
        "claims": {
            "gpu_parity_present": True,
            "bounded_gpu_formal_composition_qualified": True,
            "stage1_probe_authorized": False,
            "bounded_stage1_owner_qualified": True,
            "caller_locked": False,
            "full_phase_e_qualified": True,
            "phase_f_authorized": True,
            "campaign_qualified": False,
            "speed_claimed": False,
        },
    }
    receipt["receipt_sha256"] = phase_e_gpu_receipt_sha256(receipt)
    return receipt


def _rehash(receipt: dict[str, object]) -> None:
    receipt["receipt_sha256"] = phase_e_gpu_receipt_sha256(receipt)


def test_qualification_accepts_exact_ten_passed_same_source_nodes() -> None:
    receipt = _qualification_receipt()

    authorization = validate_phase_e_gpu_qualification_receipt(
        receipt,
        bindings=_bindings(),
    )

    assert authorization.kind == "qualification"
    assert authorization.receipt_sha256 == receipt["receipt_sha256"]
    assert authorization.source_commit == SOURCE_COMMIT
    assert authorization.generation_artifact_sha256 == GENERATION_ARTIFACT_SHA256
    assert authorization.qualification_receipt_sha256 is None
    assert authorization.logical_profiles == SUPPORTED_LOGICAL_PROFILES
    assert authorization.bounded_stage1_owner_qualified is False
    assert authorization.phase_f_authorized is False


def test_validator_minted_authorization_cannot_be_promoted_with_dataclass_replace() -> None:
    authorization = validate_phase_e_gpu_qualification_receipt(
        _qualification_receipt(),
        bindings=_bindings(),
    )
    promoted = replace(
        authorization,
        kind="final",
        bounded_stage1_owner_qualified=True,
        phase_f_authorized=True,
    )

    with pytest.raises(PhaseEGpuGateError, match="was not minted"):
        require_phase_e_gpu_authorization(
            promoted,
            final=True,
            profile_id=SUPPORTED_LOGICAL_PROFILES[0],
        )


def test_validator_minted_authorization_detects_low_level_mutation() -> None:
    authorization = validate_phase_e_gpu_qualification_receipt(
        _qualification_receipt(),
        bindings=_bindings(),
    )
    object.__setattr__(authorization, "kind", "final")
    object.__setattr__(authorization, "bounded_stage1_owner_qualified", True)
    object.__setattr__(authorization, "phase_f_authorized", True)

    with pytest.raises(PhaseEGpuGateError, match="was not minted"):
        require_phase_e_gpu_authorization(
            authorization,
            final=True,
            profile_id=SUPPORTED_LOGICAL_PROFILES[0],
        )


def test_required_gpu_suite_is_exactly_ten_named_nodes() -> None:
    assert REQUIRED_GPU_TEST_NODE_IDS == (
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
    assert len(REQUIRED_GPU_TEST_NODE_IDS) == 10
    assert len(set(REQUIRED_GPU_TEST_NODE_IDS)) == 10


@pytest.mark.parametrize(
    "variant",
    ("missing", "extra", "duplicate", "equivalent_spelling"),
)
def test_qualification_rejects_nonexact_node_set(variant: str) -> None:
    receipt = _qualification_receipt()
    results = receipt["test_results"]
    assert isinstance(results, list)
    if variant == "missing":
        results.pop()
    elif variant == "extra":
        results.append(
            {
                "node_id": "tests/perf/test_unapproved.py::test_extra",
                "status": "passed",
                "evidence_sha256": JUNIT_SHA256,
            }
        )
    elif variant == "duplicate":
        results[-1] = deepcopy(results[0])
    else:
        results[0]["node_id"] = f"./{results[0]['node_id']}"
    _rehash(receipt)

    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_qualification_receipt(receipt, bindings=_bindings())


@pytest.mark.parametrize(
    "outcome",
    ("skipped", "failed", "error", "xfailed", "xpassed"),
)
def test_qualification_rejects_every_nonpass_outcome(outcome: str) -> None:
    receipt = _qualification_receipt()
    receipt["test_results"][4]["status"] = outcome
    _rehash(receipt)

    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_qualification_receipt(receipt, bindings=_bindings())


@pytest.mark.parametrize(
    "count_field",
    ("failures", "errors", "skipped", "xfailed", "xpassed"),
)
def test_qualification_rejects_nonzero_nonpass_count(count_field: str) -> None:
    receipt = _qualification_receipt()
    receipt["test_counts"]["passed"] = 9
    receipt["test_counts"][count_field] = 1
    _rehash(receipt)

    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_qualification_receipt(receipt, bindings=_bindings())


@pytest.mark.parametrize(
    "field",
    (
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
    ),
)
def test_qualification_rejects_every_binding_mismatch(field: str) -> None:
    receipt = _qualification_receipt()
    bindings = _bindings()
    replacement = "different/source" if field == "source_branch" else "f" * len(
        str(receipt[field])
    )
    bindings = replace(bindings, **{field: replacement})

    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_qualification_receipt(receipt, bindings=bindings)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("logical_profiles", [SUPPORTED_LOGICAL_PROFILES[0]]),
        ("hidden_fallback_count", 1),
        ("modular_rng_draw_count", 1),
    ),
)
def test_qualification_rejects_wrong_contract(
    field: str,
    replacement: object,
) -> None:
    receipt = _qualification_receipt()
    receipt[field] = replacement
    _rehash(receipt)

    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_qualification_receipt(receipt, bindings=_bindings())


@pytest.mark.parametrize(
    ("lane_field", "replacement"),
    (
        ("generation", "python_shadow"),
        ("modular_tick", "python_shadow"),
        ("hidden_fallback", True),
    ),
)
def test_qualification_rejects_wrong_lane_owner(
    lane_field: str,
    replacement: object,
) -> None:
    receipt = _qualification_receipt()
    receipt["lane_owners"][lane_field] = replacement
    _rehash(receipt)

    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_qualification_receipt(receipt, bindings=_bindings())


def test_qualification_rejects_missing_or_failed_arm() -> None:
    receipt = _qualification_receipt()
    del receipt["arms"][REQUIRED_PHASE_E_ARMS[-1]]
    _rehash(receipt)
    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_qualification_receipt(receipt, bindings=_bindings())

    receipt = _qualification_receipt()
    receipt["arms"][REQUIRED_PHASE_E_ARMS[0]]["status"] = "failed"
    _rehash(receipt)
    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_qualification_receipt(receipt, bindings=_bindings())


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("atol", 2.1e-7), ("rtol", 1.0e-12)),
)
def test_qualification_rejects_wrong_tolerance(
    field: str,
    replacement: float,
) -> None:
    receipt = _qualification_receipt()
    receipt["numeric_rule"][field] = replacement
    _rehash(receipt)

    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_qualification_receipt(receipt, bindings=_bindings())


@pytest.mark.parametrize(
    ("claim", "replacement"),
    (
        ("gpu_parity_present", False),
        ("bounded_gpu_formal_composition_qualified", False),
        ("stage1_probe_authorized", False),
        ("bounded_stage1_owner_qualified", True),
        ("caller_locked", False),
        ("full_phase_e_qualified", True),
        ("phase_f_authorized", True),
        ("campaign_qualified", True),
        ("speed_claimed", True),
    ),
)
def test_qualification_rejects_wrong_claim(
    claim: str,
    replacement: bool,
) -> None:
    receipt = _qualification_receipt()
    receipt["claims"][claim] = replacement
    _rehash(receipt)

    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_qualification_receipt(receipt, bindings=_bindings())


def test_qualification_rejects_self_hash_tamper_and_extra_field() -> None:
    receipt = _qualification_receipt()
    receipt["receipt_sha256"] = "f" * 64
    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_qualification_receipt(receipt, bindings=_bindings())

    receipt = _qualification_receipt()
    receipt["unreviewed"] = True
    _rehash(receipt)
    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_qualification_receipt(receipt, bindings=_bindings())


def test_final_accepts_exactly_two_matching_passed_flat_one_runs() -> None:
    qualification = _qualification_receipt()
    receipt = _final_receipt(qualification)

    authorization = validate_phase_e_gpu_final_receipt(
        receipt,
        bindings=_bindings(),
        qualification_receipt=qualification,
    )

    assert authorization.kind == "final"
    assert authorization.receipt_sha256 == receipt["receipt_sha256"]
    assert authorization.qualification_receipt_sha256 == qualification["receipt_sha256"]
    assert authorization.source_commit == SOURCE_COMMIT
    assert authorization.generation_artifact_sha256 == GENERATION_ARTIFACT_SHA256
    assert authorization.logical_profiles == (SUPPORTED_LOGICAL_PROFILES[0],)
    assert authorization.bounded_stage1_owner_qualified is True
    assert authorization.phase_f_authorized is True


@pytest.mark.parametrize(
    "variant",
    ("one", "three", "failed", "mode", "lane_owner"),
)
def test_final_rejects_wrong_run_count_status_or_lane(variant: str) -> None:
    qualification = _qualification_receipt()
    receipt = _final_receipt(qualification)
    runs = receipt["stage1_runs"]
    assert isinstance(runs, list)
    if variant == "one":
        runs.pop()
    elif variant == "three":
        runs.append(_stage1_run("flat-one-repeat-3"))
    elif variant == "failed":
        runs[1]["status"] = "failed"
    elif variant == "mode":
        runs[1]["mode"] = "patch-one"
    else:
        runs[1]["lane_owner"] = "python_shadow"
    _rehash(receipt)

    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_final_receipt(
            receipt,
            bindings=_bindings(),
            qualification_receipt=qualification,
        )


@pytest.mark.parametrize(
    "field",
    (
        "semantic_sha256",
        "checkpoint_sha256",
        "public_output_sha256",
        "action_sha256",
        "rng_sha256",
        "identity_sha256",
        "slot_map_sha256",
        "profile_id",
    ),
)
def test_final_rejects_any_semantic_or_exact_digest_mismatch(field: str) -> None:
    qualification = _qualification_receipt()
    receipt = _final_receipt(qualification)
    receipt["stage1_runs"][1][field] = (
        SUPPORTED_LOGICAL_PROFILES[1] if field == "profile_id" else "2" * 64
    )
    _rehash(receipt)

    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_final_receipt(
            receipt,
            bindings=_bindings(),
            qualification_receipt=qualification,
        )


def test_final_rejects_fallback_activity() -> None:
    qualification = _qualification_receipt()
    receipt = _final_receipt(qualification)
    receipt["stage1_runs"][1]["hidden_fallback_count"] = 1
    _rehash(receipt)

    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_final_receipt(
            receipt,
            bindings=_bindings(),
            qualification_receipt=qualification,
        )


def test_final_rejects_duplicate_stage1_output_identity() -> None:
    qualification = _qualification_receipt()
    receipt = _final_receipt(qualification)
    receipt["stage1_runs"][1]["output_sha256"] = receipt["stage1_runs"][0][
        "output_sha256"
    ]
    _rehash(receipt)

    with pytest.raises(RuntimeError, match="distinct output identities"):
        validate_phase_e_gpu_final_receipt(
            receipt,
            bindings=_bindings(),
            qualification_receipt=qualification,
        )


@pytest.mark.parametrize(
    ("claim", "replacement"),
    (
        ("gpu_parity_present", False),
        ("bounded_gpu_formal_composition_qualified", False),
        ("stage1_probe_authorized", True),
        ("bounded_stage1_owner_qualified", False),
        ("caller_locked", True),
        ("full_phase_e_qualified", False),
        ("phase_f_authorized", False),
        ("campaign_qualified", True),
        ("speed_claimed", True),
    ),
)
def test_final_rejects_wrong_claim(claim: str, replacement: bool) -> None:
    qualification = _qualification_receipt()
    receipt = _final_receipt(qualification)
    receipt["claims"][claim] = replacement
    _rehash(receipt)

    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_final_receipt(
            receipt,
            bindings=_bindings(),
            qualification_receipt=qualification,
        )


def test_receipt_scopes_cannot_be_substituted() -> None:
    qualification = _qualification_receipt()
    final = _final_receipt(qualification)

    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_final_receipt(
            qualification,
            bindings=_bindings(),
            qualification_receipt=qualification,
        )
    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_qualification_receipt(final, bindings=_bindings())


def test_final_rejects_wrong_qualification_receipt_and_self_hash_tamper() -> None:
    qualification = _qualification_receipt()
    receipt = _final_receipt(qualification)

    wrong_qualification = deepcopy(qualification)
    wrong_qualification["log_sha256"] = "e" * 64
    _rehash(wrong_qualification)
    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_final_receipt(
            receipt,
            bindings=_bindings(),
            qualification_receipt=wrong_qualification,
        )

    receipt["receipt_sha256"] = "e" * 64
    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_final_receipt(
            receipt,
            bindings=_bindings(),
            qualification_receipt=qualification,
        )


def test_final_rejects_extra_top_level_field() -> None:
    qualification = _qualification_receipt()
    receipt = _final_receipt(qualification)
    receipt["unreviewed"] = True
    _rehash(receipt)

    with pytest.raises(RuntimeError):
        validate_phase_e_gpu_final_receipt(
            receipt,
            bindings=_bindings(),
            qualification_receipt=qualification,
        )
