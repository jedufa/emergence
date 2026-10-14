from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from perf.cuda_architecture_b_modular_state_abi import build_modular_storage_layout
from perf.cuda_architecture_b_phase_e_formal_composition import (
    PhaseEFormalCompositionRaw,
    execute_phase_e_cpu_formal_composition,
)

from perf.cuda_architecture_b_phase_e_formal_gate import (
    EXACT_REQUIREMENTS,
    FORMAL_GATE_ACTION_VOCABULARY_VERSION,
    FORMAL_GATE_ACTION_WIDTH,
    FORMAL_GATE_CATEGORICAL_RULE_ID,
    FORMAL_GATE_CHECKPOINT_SCHEMA_VERSION,
    FORMAL_GATE_COMPOSITION_ORDER_VERSION,
    FORMAL_GATE_LANE_OWNER,
    FORMAL_GATE_NUMERIC_RULE_ID,
    FORMAL_GATE_OBSERVATION_SCHEMA_VERSION,
    FORMAL_GATE_OBSERVATION_WIDTH,
    FORMAL_GATE_PLANE_COUNT,
    FORMAL_GATE_PLANE_SCHEMA_VERSION,
    FORMAL_GATE_POLICY_PROFILE,
    FORMAL_GATE_RNG_RULE_ID,
    FORMAL_GATE_SCHEMA_VERSION,
    FORMAL_GATE_SCOPE,
    NAMED_FORMAL_TOLERANCE_FIELDS,
    REQUIRED_GATE_IDS,
    REQUIRED_PHASE_E_ARMS,
    SUPPORTED_LOGICAL_PROFILES,
    PhaseEFormalGateBindings,
    PhaseEFormalGateError,
    phase_e_formal_composition_raw_sha256,
    phase_e_formal_receipt_sha256,
    validate_phase_e_formal_composition_receipt,
)


SOURCE_BRANCH = "integration/evolutionary-capacity-phase-e"
SOURCE_COMMIT = "1" * 40
TEST_COMMIT = "2" * 40
NATIVE_SOURCE_BUNDLE_SHA256 = "2" * 64
CAPABILITY_SHA256 = "3" * 64
FIXTURE_SOURCE_SHA256 = "4" * 64
TEST_SOURCE_SHA256 = "5" * 64
VALIDATOR_IMPLEMENTATION_SHA256 = "6" * 64
COMPONENT_IMPLEMENTATION_SHA256 = {
    "modular_tick": "7" * 64,
    "policy_composition": "8" * 64,
}


def _composition_observation() -> PhaseEFormalCompositionRaw:
    layout = build_modular_storage_layout(1)
    planes = {
        field.name: torch.full(
            field.shape,
            field.fill_value,
            dtype=field.torch_dtype,
        ).contiguous()
        for field in layout.fields
    }
    planes["agent_occupied"][0] = True
    planes["agent_id"][0] = 41
    planes["agent_slot_token"][0] = 9001
    planes["module_count"][0] = 1
    planes["module_execution_order"][0, 0] = 0
    planes["dependency_topological_order"][0, 0] = 0
    planes["module_occupied"][0, 0] = True
    planes["module_local_id_index"][0, 0] = 100
    planes["module_instance_id_index"][0, 0] = 200
    planes["module_homology_id_index"][0, 0] = 300
    planes["module_type_index"][0, 0] = 400
    planes["module_type_code"][0, 0] = 1
    planes["identity_origin_code"][0, 0] = 2
    planes["operator_code"][0, 0] = 1
    planes["integrator_parameters"][0, 0] = torch.tensor(
        [0.75, 0.5, 0.75, 0.5], dtype=torch.float32
    )
    planes["module_state_present"][0, 0] = True
    planes["module_state"][0, 0, 0] = 0.2
    planes["module_output_present"][0, 0] = True
    planes["module_previous_output"][0, 0, 0] = 0.1
    planes["input_binding_count"][0, 0] = 1
    planes["input_semantic_code"][0, 0, 0] = 0
    planes["input_semantic_id_index"][0, 0, 0] = 500
    planes["input_gain"][0, 0, 0] = 1.0
    planes["input_enabled"][0, 0, 0] = True
    planes["output_binding_count"][0, 0] = 1
    planes["output_semantic_code"][0, 0, 0] = 0
    planes["output_semantic_id_index"][0, 0, 0] = 600
    planes["output_action_index"][0, 0, 0] = 5
    planes["output_gain"][0, 0, 0] = 1.0
    planes["output_enabled"][0, 0, 0] = True
    planes["development_explicit"][0, 0] = True
    planes["development_stage_gains"][0, 0] = 1.0

    modular_observations = torch.zeros((1, 27), dtype=torch.float32)
    modular_observations[0, 25] = 0.8
    modular_observations[0, 26] = 0.6
    hidden_size = 3
    return execute_phase_e_cpu_formal_composition(
        planes=planes,
        modular_observations=modular_observations,
        fatigue=torch.tensor([0.1], dtype=torch.float32),
        stage_indices=torch.tensor([0], dtype=torch.int64),
        policy_observations=torch.linspace(
            -0.5, 0.5, 27, dtype=torch.float32
        ).reshape(1, 27),
        w1=torch.zeros((1, hidden_size, 27), dtype=torch.float32),
        b1=torch.zeros((1, hidden_size), dtype=torch.float32),
        w2=torch.zeros((1, hidden_size, hidden_size), dtype=torch.float32),
        b2=torch.zeros((1, hidden_size), dtype=torch.float32),
        w3=torch.zeros((1, FORMAL_GATE_ACTION_WIDTH, hidden_size), dtype=torch.float32),
        b3=torch.tensor(
            [[0.2, 0.05, 0.04, 0.03, 0.02, 0.0, 0.5]], dtype=torch.float32
        ),
        attack_bias=torch.zeros((1,), dtype=torch.float32),
        action_temperature=torch.ones((1,), dtype=torch.float32),
        validity_mask=torch.tensor(
            [[True, True, True, True, True, True, False]], dtype=torch.bool
        ),
        authority_mask=torch.tensor(
            [[False, True, True, True, True, True, True]], dtype=torch.bool
        ),
        rng_stream_indices={"environment": 23, "policy": 17},
    )


def _receipt(
    observed: PhaseEFormalCompositionRaw | None = None,
) -> dict[str, object]:
    if observed is None:
        observed = _composition_observation()
    receipt: dict[str, object] = {
        "schema_version": FORMAL_GATE_SCHEMA_VERSION,
        "scope": FORMAL_GATE_SCOPE,
        "lane_owner": FORMAL_GATE_LANE_OWNER,
        "source_branch": SOURCE_BRANCH,
        "source_commit": SOURCE_COMMIT,
        "test_commit": TEST_COMMIT,
        "native_source_bundle_sha256": NATIVE_SOURCE_BUNDLE_SHA256,
        "gpu_artifact_present": False,
        "native_artifact_sha256": None,
        "capability_sha256": CAPABILITY_SHA256,
        "fixture_source_sha256": FIXTURE_SOURCE_SHA256,
        "test_source_sha256": TEST_SOURCE_SHA256,
        "plane_schema_version": FORMAL_GATE_PLANE_SCHEMA_VERSION,
        "plane_count": FORMAL_GATE_PLANE_COUNT,
        "action_vocabulary_version": FORMAL_GATE_ACTION_VOCABULARY_VERSION,
        "action_width": FORMAL_GATE_ACTION_WIDTH,
        "observation_schema_version": FORMAL_GATE_OBSERVATION_SCHEMA_VERSION,
        "observation_width": FORMAL_GATE_OBSERVATION_WIDTH,
        "policy_profile": FORMAL_GATE_POLICY_PROFILE,
        "checkpoint_schema_version": FORMAL_GATE_CHECKPOINT_SCHEMA_VERSION,
        "component_implementation_sha256": dict(COMPONENT_IMPLEMENTATION_SHA256),
        "validator_implementation_sha256": VALIDATOR_IMPLEMENTATION_SHA256,
        "raw_composition_sha256": phase_e_formal_composition_raw_sha256(observed),
        "categorical_rule_id": FORMAL_GATE_CATEGORICAL_RULE_ID,
        "rng_rule_id": FORMAL_GATE_RNG_RULE_ID,
        "composition_order_version": FORMAL_GATE_COMPOSITION_ORDER_VERSION,
        "logical_profiles": list(SUPPORTED_LOGICAL_PROFILES),
        "arms": {
            name: {
                "status": "cpu_reference_passed_gpu_pending",
                "comparison_basis": "cpu_reference_component_and_static_preparation",
                "evidence_sha256": f"{index + 10:064x}",
                "first_mismatch": None,
            }
            for index, name in enumerate(REQUIRED_PHASE_E_ARMS)
        },
        "excluded_arms": ["reproduction"],
        "gates": {
            gate_id: {
                "status": "cpu_reference_or_static_ready_gpu_pending",
                "evidence_sha256": f"{index + 100:064x}",
                "first_mismatch": None,
            }
            for index, gate_id in enumerate(REQUIRED_GATE_IDS)
        },
        "hidden_fallback_count": 0,
        "modular_rng_draw_count": 0,
        "exact_requirements": {name: True for name in EXACT_REQUIREMENTS},
        "numeric_rule": {
            "rule_id": FORMAL_GATE_NUMERIC_RULE_ID,
            "review_status": "formally_accepted",
            "atol": 2.0e-7,
            "rtol": 0.0,
            "named_float_tolerance_fields": list(NAMED_FORMAL_TOLERANCE_FIELDS),
            "all_other_fields_exact": True,
        },
        "claims": {
            "stage1_gpu_ready_but_locked": True,
            "caller_locked": True,
            "formal_composition_receipt_accepted": True,
            "bounded_stage1_owner_qualified": False,
            "bounded_gpu_formal_composition_qualified": False,
            "gpu_parity_present": False,
            "whole_generation_cpu_oracle_present": False,
            "full_phase_e_qualified": False,
            "phase_f_authorized": False,
            "campaign_qualified": False,
            "speed_claimed": False,
        },
    }
    receipt["receipt_sha256"] = phase_e_formal_receipt_sha256(receipt)
    return receipt


def _bindings(receipt: dict[str, object]) -> PhaseEFormalGateBindings:
    return PhaseEFormalGateBindings(
        source_branch=SOURCE_BRANCH,
        source_commit=SOURCE_COMMIT,
        test_commit=TEST_COMMIT,
        native_source_bundle_sha256=NATIVE_SOURCE_BUNDLE_SHA256,
        gpu_artifact_present=False,
        native_artifact_sha256=None,
        capability_sha256=CAPABILITY_SHA256,
        fixture_source_sha256=FIXTURE_SOURCE_SHA256,
        test_source_sha256=TEST_SOURCE_SHA256,
        plane_schema_version=FORMAL_GATE_PLANE_SCHEMA_VERSION,
        plane_count=FORMAL_GATE_PLANE_COUNT,
        action_vocabulary_version=FORMAL_GATE_ACTION_VOCABULARY_VERSION,
        action_width=FORMAL_GATE_ACTION_WIDTH,
        observation_schema_version=FORMAL_GATE_OBSERVATION_SCHEMA_VERSION,
        observation_width=FORMAL_GATE_OBSERVATION_WIDTH,
        policy_profile=FORMAL_GATE_POLICY_PROFILE,
        checkpoint_schema_version=FORMAL_GATE_CHECKPOINT_SCHEMA_VERSION,
        component_implementation_sha256=dict(COMPONENT_IMPLEMENTATION_SHA256),
        validator_implementation_sha256=VALIDATOR_IMPLEMENTATION_SHA256,
        raw_composition_sha256=str(receipt["raw_composition_sha256"]),
        categorical_rule_id=FORMAL_GATE_CATEGORICAL_RULE_ID,
        rng_rule_id=FORMAL_GATE_RNG_RULE_ID,
        composition_order_version=FORMAL_GATE_COMPOSITION_ORDER_VERSION,
        receipt_sha256=str(receipt["receipt_sha256"]),
    )


def _rehash(receipt: dict[str, object]) -> None:
    receipt["receipt_sha256"] = phase_e_formal_receipt_sha256(receipt)


def _validate(
    receipt: dict[str, object],
    *,
    bindings: PhaseEFormalGateBindings | None = None,
    observed: PhaseEFormalCompositionRaw | None = None,
):
    if bindings is None:
        bindings = _bindings(receipt)
    if observed is None:
        observed = _composition_observation()
    return validate_phase_e_formal_composition_receipt(
        receipt,
        bindings=bindings,
        observed_composition=observed,
    )


def test_valid_receipt_returns_only_locked_cpu_preparation() -> None:
    observed = _composition_observation()
    receipt = _receipt(observed)

    authorization = _validate(receipt, observed=observed)

    assert authorization.scope == FORMAL_GATE_SCOPE
    assert authorization.lane_owner == FORMAL_GATE_LANE_OWNER
    assert authorization.logical_profiles == SUPPORTED_LOGICAL_PROFILES
    assert authorization.arms == REQUIRED_PHASE_E_ARMS
    assert authorization.gate_ids == REQUIRED_GATE_IDS
    assert authorization.receipt_sha256 == receipt["receipt_sha256"]
    assert authorization.to_dict()["numeric_rule_id"] == FORMAL_GATE_NUMERIC_RULE_ID
    assert authorization.raw_composition_sha256 == receipt["raw_composition_sha256"]
    assert authorization.caller_locked is True
    assert authorization.gpu_parity_present is False
    assert authorization.gpu_artifact_present is False
    assert authorization.native_artifact_sha256 is None
    assert authorization.stage1_gpu_ready_but_locked is True
    assert authorization.formal_composition_receipt_accepted is True
    assert authorization.bounded_stage1_owner_qualified is False
    assert authorization.bounded_gpu_formal_composition_qualified is False
    assert authorization.whole_generation_cpu_oracle_present is False
    assert authorization.full_phase_e_qualified is False
    assert authorization.phase_f_authorized is False
    assert authorization.campaign_qualified is False
    assert authorization.speed_claimed is False


def test_digest_rejects_tampering_and_unapproved_rehash() -> None:
    receipt = _receipt()
    bindings = _bindings(receipt)
    receipt["hidden_fallback_count"] = 1

    with pytest.raises(PhaseEFormalGateError, match="canonical receipt digest mismatch"):
        _validate(receipt, bindings=bindings)

    _rehash(receipt)
    with pytest.raises(PhaseEFormalGateError, match="not the approved digest"):
        _validate(receipt, bindings=bindings)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("schema_version", "v2", "schema mismatch"),
        ("scope", "full_phase_e", "scope mismatch"),
        ("lane_owner", "python_shadow", "lane owner mismatch"),
        ("logical_profiles", ["b2-12x32-v1"], "both exact logical B2 profiles"),
        ("excluded_arms", [], "exclude only the reproduction arm"),
        ("hidden_fallback_count", 1, "hidden_fallback_count"),
        ("modular_rng_draw_count", 1, "modular_rng_draw_count"),
    ),
)
def test_top_level_contract_fails_closed(
    field: str,
    replacement: object,
    message: str,
) -> None:
    receipt = _receipt()
    receipt[field] = replacement
    _rehash(receipt)

    with pytest.raises(PhaseEFormalGateError, match=message):
        _validate(receipt)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("source_commit", "approved binding"),
        ("native_artifact_sha256", "must not bind a native artifact"),
        ("capability_sha256", "approved binding"),
        ("fixture_source_sha256", "approved binding"),
        ("test_source_sha256", "approved binding"),
    ),
)
def test_external_identity_bindings_cannot_be_self_attested(
    field: str,
    message: str,
) -> None:
    receipt = _receipt()
    bindings = _bindings(receipt)
    replacement = "a" * (40 if field == "source_commit" else 64)
    receipt[field] = replacement
    _rehash(receipt)

    with pytest.raises(PhaseEFormalGateError, match=message):
        _validate(receipt, bindings=bindings)


def test_exact_twelve_arms_forbid_missing_extra_or_reproduction_arm() -> None:
    for mutation in ("missing", "extra", "reproduction"):
        receipt = _receipt()
        arms = receipt["arms"]
        assert isinstance(arms, dict)
        if mutation == "missing":
            arms.pop("integrator")
        else:
            arms[mutation] = deepcopy(arms["integrator"])
        _rehash(receipt)

        with pytest.raises(PhaseEFormalGateError, match="exact 12-arm"):
            _validate(receipt)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("status", "passed", "locked pending GPU parity"),
        (
            "comparison_basis",
            "cpu_reference_to_native_accelerated",
            "comparison basis is invalid",
        ),
    ),
)
def test_arm_status_cannot_claim_gpu_or_native_parity(
    field: str,
    replacement: str,
    message: str,
) -> None:
    receipt = _receipt()
    arms = receipt["arms"]
    assert isinstance(arms, dict)
    arm = arms[REQUIRED_PHASE_E_ARMS[0]]
    assert isinstance(arm, dict)
    arm[field] = replacement
    _rehash(receipt)

    with pytest.raises(PhaseEFormalGateError, match=message):
        _validate(receipt)


@pytest.mark.parametrize("damage", ("missing", "failed", "mismatch", "bad_hash"))
def test_every_required_gate_must_have_clean_evidence(damage: str) -> None:
    receipt = _receipt()
    gates = receipt["gates"]
    assert isinstance(gates, dict)
    gate_id = REQUIRED_GATE_IDS[0]
    gate = gates[gate_id]
    assert isinstance(gate, dict)
    if damage == "missing":
        gates.pop(gate_id)
        message = "gate IDs"
    elif damage == "failed":
        gate["status"] = "passed"
        message = "GPU-pending"
    elif damage == "mismatch":
        gate["first_mismatch"] = "planes.module_state[0,0]"
        message = "reports a mismatch"
    else:
        gate["evidence_sha256"] = "invalid"
        message = "lowercase SHA-256"
    _rehash(receipt)

    with pytest.raises(PhaseEFormalGateError, match=message):
        _validate(receipt)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("atol", 2.1e-7, "atol=2e-7"),
        ("rtol", 1.0e-8, "rtol=0"),
        ("review_status", "proposed", "not formally accepted"),
        ("all_other_fields_exact", False, "must remain exact"),
        (
            "named_float_tolerance_fields",
            ["unexposed_reset_gate"],
            "accepted float fields",
        ),
    ),
)
def test_numeric_rule_cannot_expand_beyond_prepared_float_surfaces(
    field: str,
    replacement: object,
    message: str,
) -> None:
    receipt = _receipt()
    rule = receipt["numeric_rule"]
    assert isinstance(rule, dict)
    rule[field] = replacement
    _rehash(receipt)

    with pytest.raises(PhaseEFormalGateError, match=message):
        _validate(receipt)


def test_float_tolerance_and_exact_surfaces_match_prepared_evidence() -> None:
    assert "module_state" in NAMED_FORMAL_TOLERANCE_FIELDS
    assert "module_previous_output" in NAMED_FORMAL_TOLERANCE_FIELDS
    assert "state_before" in NAMED_FORMAL_TOLERANCE_FIELDS
    assert "state_after" in NAMED_FORMAL_TOLERANCE_FIELDS
    assert "module_output" in NAMED_FORMAL_TOLERANCE_FIELDS
    assert "action_logits" in NAMED_FORMAL_TOLERANCE_FIELDS
    assert "reset_gate" not in NAMED_FORMAL_TOLERANCE_FIELDS
    assert "update_gate" not in NAMED_FORMAL_TOLERANCE_FIELDS
    assert "candidate_state_intermediate" not in NAMED_FORMAL_TOLERANCE_FIELDS

    assert "module_state" not in EXACT_REQUIREMENTS
    assert "action_logits" not in EXACT_REQUIREMENTS
    assert "module_state_presence" in EXACT_REQUIREMENTS
    assert "module_output_presence" in EXACT_REQUIREMENTS
    assert "routing_masks" in EXACT_REQUIREMENTS
    assert "dormancy_masks" in EXACT_REQUIREMENTS
    assert "chosen_actions" in EXACT_REQUIREMENTS
    assert "inactive_module_state_canonical" in EXACT_REQUIREMENTS
    assert "inactive_module_output_canonical" in EXACT_REQUIREMENTS
    assert "checkpoint_projection" in EXACT_REQUIREMENTS
    assert "checkpoint_resume" in EXACT_REQUIREMENTS


def test_exact_requirements_and_claims_cannot_be_weakened() -> None:
    receipt = _receipt()
    requirements = receipt["exact_requirements"]
    assert isinstance(requirements, dict)
    requirements["checkpoint_projection"] = False
    _rehash(receipt)
    with pytest.raises(PhaseEFormalGateError, match="weakened"):
        _validate(receipt)

    receipt = _receipt()
    claims = receipt["claims"]
    assert isinstance(claims, dict)
    claims["formal_composition_receipt_accepted"] = False
    _rehash(receipt)
    with pytest.raises(PhaseEFormalGateError, match="missing or overbroad"):
        _validate(receipt)

    receipt = _receipt()
    claims = receipt["claims"]
    assert isinstance(claims, dict)
    claims["full_phase_e_qualified"] = True
    _rehash(receipt)
    with pytest.raises(PhaseEFormalGateError, match="overbroad"):
        _validate(receipt)

    for claim in (
        "bounded_stage1_owner_qualified",
        "bounded_gpu_formal_composition_qualified",
        "gpu_parity_present",
        "whole_generation_cpu_oracle_present",
        "phase_f_authorized",
        "campaign_qualified",
        "speed_claimed",
    ):
        receipt = _receipt()
        claims = receipt["claims"]
        assert isinstance(claims, dict)
        claims[claim] = True
        _rehash(receipt)
        with pytest.raises(PhaseEFormalGateError, match="overbroad"):
            _validate(receipt)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("source_branch", "integration/not-the-approved-branch"),
        ("test_commit", "a" * 40),
        ("native_source_bundle_sha256", "a" * 64),
        ("raw_composition_sha256", "a" * 64),
        ("plane_schema_version", "future_plane_schema"),
        ("plane_count", 58),
        ("action_vocabulary_version", "future_actions"),
        ("action_width", 8),
        ("observation_schema_version", "future_observations"),
        ("observation_width", 28),
        ("policy_profile", "learned_policy"),
        ("checkpoint_schema_version", "future_checkpoint"),
        ("validator_implementation_sha256", "a" * 64),
        ("categorical_rule_id", "future_categorical"),
        ("rng_rule_id", "future_rng"),
        ("composition_order_version", "future_composition"),
    ),
)
def test_every_bridge_binding_mismatch_fails_closed(
    field: str,
    replacement: object,
) -> None:
    receipt = _receipt()
    bindings = _bindings(receipt)
    receipt[field] = replacement
    _rehash(receipt)

    with pytest.raises(PhaseEFormalGateError):
        _validate(receipt, bindings=bindings)


def test_component_hash_map_and_gpu_artifact_state_fail_closed() -> None:
    receipt = _receipt()
    bindings = _bindings(receipt)
    components = receipt["component_implementation_sha256"]
    assert isinstance(components, dict)
    components["modular_tick"] = "a" * 64
    _rehash(receipt)
    with pytest.raises(PhaseEFormalGateError, match="component implementation"):
        _validate(receipt, bindings=bindings)

    receipt = _receipt()
    receipt["gpu_artifact_present"] = True
    receipt["native_artifact_sha256"] = "a" * 64
    _rehash(receipt)
    with pytest.raises(PhaseEFormalGateError, match="GPU artifact"):
        _validate(receipt)


def _validate_corrupt_observation(observed: PhaseEFormalCompositionRaw) -> None:
    receipt = _receipt(observed)
    _validate(receipt, observed=observed)


def test_raw_evidence_digest_is_deterministic_and_externally_bound() -> None:
    observed = _composition_observation()
    assert phase_e_formal_composition_raw_sha256(observed) == (
        phase_e_formal_composition_raw_sha256(deepcopy(observed))
    )
    receipt = _receipt(observed)
    bindings = _bindings(receipt)
    observed.composed_logits[0, 0] += 1.0

    with pytest.raises(PhaseEFormalGateError, match="not the approved evidence"):
        _validate(receipt, bindings=bindings, observed=observed)


def test_oracle_recomputation_rejects_double_residual() -> None:
    observed = _composition_observation()
    observed.composed_logits.add_(observed.modular_action_residuals)

    with pytest.raises(PhaseEFormalGateError, match="composed_logits"):
        _validate_corrupt_observation(observed)


def test_oracle_recomputation_rejects_mask_or_authority_change() -> None:
    observed = _composition_observation()
    observed.authority_mask_after[0, 1] = ~observed.authority_mask_after[0, 1]

    with pytest.raises(PhaseEFormalGateError, match="changed the authority mask"):
        _validate_corrupt_observation(observed)


def test_oracle_recomputation_rejects_wrong_selected_action() -> None:
    observed = _composition_observation()
    observed.deterministic_actions[0] = (
        int(observed.deterministic_actions[0]) + 1
    ) % FORMAL_GATE_ACTION_WIDTH

    with pytest.raises(PhaseEFormalGateError, match="deterministic_actions"):
        _validate_corrupt_observation(observed)


def test_oracle_recomputation_rejects_rng_index_drift() -> None:
    observed = _composition_observation()
    observed.rng_stream_indices_after["policy"] += 1

    with pytest.raises(PhaseEFormalGateError, match="RNG stream index"):
        _validate_corrupt_observation(observed)


def test_oracle_recomputation_rejects_unowned_plane_change() -> None:
    observed = _composition_observation()
    observed.planes_after["agent_id"][0] += 1

    with pytest.raises(PhaseEFormalGateError, match="unowned plane agent_id"):
        _validate_corrupt_observation(observed)


def test_oracle_float_rule_accepts_bounded_delta_and_rejects_excess() -> None:
    observed = _composition_observation()
    observed.planes_after["module_state"][0, 0, 0] += 1.0e-7
    _validate_corrupt_observation(observed)

    observed = _composition_observation()
    observed.planes_after["module_state"][0, 0, 0] += 3.0e-7
    with pytest.raises(PhaseEFormalGateError, match="module_state"):
        _validate_corrupt_observation(observed)


def test_oracle_float_rule_rejects_nonfinite_output() -> None:
    observed = _composition_observation()
    observed.composed_logits[0, 0] = float("nan")

    with pytest.raises(PhaseEFormalGateError, match="nonfinite"):
        _validate_corrupt_observation(observed)


def test_unknown_fields_and_noncanonical_json_fail_closed() -> None:
    receipt = _receipt()
    receipt["future_claim"] = True
    _rehash(receipt)
    with pytest.raises(PhaseEFormalGateError, match="fields mismatch"):
        _validate(receipt)

    receipt = _receipt()
    receipt["numeric_rule"] = {"atol": float("nan")}
    with pytest.raises(PhaseEFormalGateError, match="canonical JSON"):
        phase_e_formal_receipt_sha256(receipt)
