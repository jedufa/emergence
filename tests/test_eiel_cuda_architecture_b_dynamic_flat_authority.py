from __future__ import annotations

import json

import pytest

from engine.perf.cuda_architecture_b_dynamic_flat_authority import (
    AUTHORITY_POLICY_ID,
    AuthorityLabel,
    AuthoritySource,
    ComparatorInputOrigin,
    DynamicFlatPromotionEvidence,
    classify_dynamic_flat_authority,
    emit_authority_policy_evidence,
    require_authority_label,
    validate_dynamic_flat_promotion,
)


def _complete_evidence(**changes: bool) -> DynamicFlatPromotionEvidence:
    values = {
        "cpu_oracle_exact": True,
        "cpu_repeat_exact": True,
        "cpu_checkpoint_continuation_exact": True,
        "candidate_same_device_repeat_exact": True,
        "candidate_checkpoint_continuation_exact": True,
        "invariants_exact": True,
        "sanitizer_clean": True,
        "stage2_bounded_prefix_exact": True,
        "candidate_derived_comparator_inputs": False,
    }
    values.update(changes)
    return DynamicFlatPromotionEvidence(**values)


def test_cpp_cpu_is_the_sole_independent_semantic_authority() -> None:
    classification = classify_dynamic_flat_authority(AuthoritySource.CPP_CPU)
    assert classification.authority_label == (
        AuthorityLabel.INDEPENDENT_SEMANTIC_AUTHORITY.value
    )
    assert classification.independent_eligible is True
    require_authority_label(
        classification, AuthorityLabel.INDEPENDENT_SEMANTIC_AUTHORITY
    )


def test_stage2_is_independent_before_its_capability_boundary() -> None:
    classification = classify_dynamic_flat_authority(
        AuthoritySource.IMMUTABLE_STAGE2,
        stage2_status=0,
        future_live_slots_present=False,
    )
    assert classification.authority_label == (
        AuthorityLabel.INDEPENDENT_BOUNDED_COMPARATOR.value
    )
    assert classification.capability_boundary is None
    require_authority_label(
        classification, AuthorityLabel.INDEPENDENT_BOUNDED_COMPARATOR
    )


@pytest.mark.parametrize("status", [4, "STATUS_MISSING_DRIVE_NOISE"])
def test_stage2_is_rejected_as_independent_after_missing_noise(status: int | str) -> None:
    classification = classify_dynamic_flat_authority(
        AuthoritySource.IMMUTABLE_STAGE2,
        stage2_status=status,
    )
    assert classification.authority_label is None
    assert classification.independent_eligible is False
    assert classification.capability_boundary is not None
    assert classification.capability_boundary.status == "STATUS_MISSING_DRIVE_NOISE"
    with pytest.raises(ValueError, match="STATUS_MISSING_DRIVE_NOISE"):
        require_authority_label(
            classification, AuthorityLabel.INDEPENDENT_BOUNDED_COMPARATOR
        )


def test_future_newborn_slot_emits_explicit_capability_boundary() -> None:
    classification = classify_dynamic_flat_authority(
        AuthoritySource.IMMUTABLE_STAGE2,
        future_live_slots_present=True,
    )
    receipt = classification.to_dict()["capability_boundary"]
    assert receipt == {
        "policy_id": AUTHORITY_POLICY_ID,
        "source": AuthoritySource.IMMUTABLE_STAGE2.value,
        "capability": "independent_dynamic_stochastic_rows",
        "supported": False,
        "status": "FUTURE_LIVE_SLOTS_PRESENT",
        "reason": (
            "immutable Stage2 stochastic rows cover only the initial "
            "live-slot set"
        ),
    }


def test_cpu_derived_stage2_tape_is_only_derived_corroboration() -> None:
    classification = classify_dynamic_flat_authority(
        AuthoritySource.IMMUTABLE_STAGE2,
        input_origin=ComparatorInputOrigin.CPU_LIVE_HISTORY,
        future_live_slots_present=True,
    )
    assert classification.authority_label == AuthorityLabel.DERIVED_CORROBORATION.value
    assert classification.independent_eligible is False
    with pytest.raises(ValueError, match="invalid"):
        require_authority_label(
            classification, AuthorityLabel.INDEPENDENT_BOUNDED_COMPARATOR
        )


def test_candidate_derived_stage2_tape_is_rejected_as_circular() -> None:
    with pytest.raises(ValueError, match="circular"):
        classify_dynamic_flat_authority(
            AuthoritySource.IMMUTABLE_STAGE2,
            input_origin=ComparatorInputOrigin.CANDIDATE_LIVE_HISTORY,
        )


def test_candidate_evidence_is_self_check_not_independent_authority() -> None:
    classification = classify_dynamic_flat_authority(
        AuthoritySource.CUDA_CANDIDATE
    )
    assert classification.authority_label == AuthorityLabel.CANDIDATE_SELF_CHECK.value
    assert classification.independent_eligible is False


@pytest.mark.parametrize(
    "missing",
    [
        "cpu_oracle_exact",
        "cpu_repeat_exact",
        "cpu_checkpoint_continuation_exact",
        "candidate_same_device_repeat_exact",
        "candidate_checkpoint_continuation_exact",
        "invariants_exact",
        "sanitizer_clean",
        "stage2_bounded_prefix_exact",
    ],
)
def test_promotion_rejects_each_missing_required_check(missing: str) -> None:
    with pytest.raises(ValueError, match=missing):
        validate_dynamic_flat_promotion(_complete_evidence(**{missing: False}))


def test_promotion_rejects_candidate_derived_comparator_inputs() -> None:
    with pytest.raises(ValueError, match="non_circular_comparator_inputs"):
        validate_dynamic_flat_promotion(
            _complete_evidence(candidate_derived_comparator_inputs=True)
        )


def test_complete_promotion_stack_is_eligible() -> None:
    receipt = validate_dynamic_flat_promotion(_complete_evidence())
    assert receipt["policy_id"] == AUTHORITY_POLICY_ID
    assert receipt["promotion_eligible"] is True
    assert receipt["independent_semantic_authority"] == "cpp_cpu"


def test_policy_evidence_emits_all_required_receipts(tmp_path) -> None:
    paths = emit_authority_policy_evidence(tmp_path)
    assert set(paths) == {
        "dynamic_flat_authority_policy.md",
        "dynamic_flat_authority_policy.json",
        "authority_capability_matrix.json",
        "authority_comparator_labels.json",
    }
    policy = json.loads((tmp_path / "dynamic_flat_authority_policy.json").read_text())
    assert policy["policy_id"] == AUTHORITY_POLICY_ID
    assert policy["sole_independent_semantic_authority"] == "cpp_cpu"
