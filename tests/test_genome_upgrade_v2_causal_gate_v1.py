from __future__ import annotations

from dataclasses import fields, replace

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.evolution.genome_upgrade_v2_c1_executor_v1 import (
    BackendCaseOutputV1,
    execute_c1_v1,
    reference_dimensions_for_fixture_v1,
)
from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (
    DECISION_A,
    DECISION_B,
    DECISION_C,
    DECISION_D,
    MANDATORY_FAMILIES,
    PREREGISTRATION_SHA256,
    REQUIRED_BACKENDS,
    REQUIRED_DIMENSIONS,
    REQUIRED_REPEAT_INDICES,
    AuthorityBindingV1,
    BackendCaseReceiptV1,
    CheckpointReceiptV1,
    MatchedPairEvidenceV1,
    evaluate_c1_v1,
    fixture_by_family_v1,
    preregistration_payload_v1,
)


HASH = "1" * 64
COMMIT = "2" * 40


def _authority() -> AuthorityBindingV1:
    return AuthorityBindingV1(
        source_commit=COMMIT,
        package_id_sha256=HASH,
        package_receipt_sha256=HASH,
        artifact_sha256=HASH,
        execution_route_sha256=HASH,
        checkpoint_sha256=HASH,
    )


def _checkpoint(family: str) -> CheckpointReceiptV1:
    fixture = fixture_by_family_v1(family)
    return CheckpointReceiptV1(
        family_id=family,
        pair_id=fixture["pair_id"],
        control_case_id=f"{fixture['pair_id']}-control",
        treatment_case_id=f"{fixture['pair_id']}-treatment",
        authority=_authority(),
        restored_checkpoint_sha256s=(HASH, HASH),
        execution_receipt_sha256=content_sha256(
            {"checkpoint": family}
        ),
        exact_restore=True,
    )


def _outputs() -> list[BackendCaseOutputV1]:
    outputs = []
    for family in MANDATORY_FAMILIES:
        for arm in ("control", "treatment"):
            dimensions = reference_dimensions_for_fixture_v1(family, arm)
            for backend in REQUIRED_BACKENDS:
                for repeat in REQUIRED_REPEAT_INDICES:
                    outputs.append(
                        BackendCaseOutputV1(
                            family_id=family,
                            arm=arm,
                            backend=backend,
                            repeat_index=repeat,
                            dimensions=dimensions,
                            execution_receipt_sha256=content_sha256(
                                {
                                    "family": family,
                                    "arm": arm,
                                    "backend": backend,
                                    "repeat": repeat,
                                }
                            ),
                            no_fallback=True,
                            no_jit=True,
                        )
                    )
    return outputs


def _pairs() -> tuple[MatchedPairEvidenceV1, ...]:
    result = execute_c1_v1(
        authorities={family: _authority() for family in MANDATORY_FAMILIES},
        backend_outputs=_outputs(),
        checkpoint_receipts={
            family: _checkpoint(family) for family in MANDATORY_FAMILIES
        },
    )
    return result.pairs


def _replace_pair(
    pairs: tuple[MatchedPairEvidenceV1, ...],
    index: int,
    pair: MatchedPairEvidenceV1,
) -> tuple[MatchedPairEvidenceV1, ...]:
    if index < 0:
        index += len(pairs)
    return (*pairs[:index], pair, *pairs[index + 1 :])


def test_preregistration_freezes_seven_exact_fixture_contracts() -> None:
    payload = preregistration_payload_v1()
    assert content_sha256(payload) == PREREGISTRATION_SHA256
    assert tuple(row["family_id"] for row in payload["fixtures"]) == (
        MANDATORY_FAMILIES
    )
    assert all(
        tuple(row["required_dimensions"]) == REQUIRED_DIMENSIONS
        and tuple(row["required_backends"]) == REQUIRED_BACKENDS
        and tuple(row["required_repeat_indices"])
        == REQUIRED_REPEAT_INDICES
        and row["control"]["context_q10"] is not None
        and row["treatment"]["context_q10"] is not None
        and row["fixture_sha256"]
        for row in payload["fixtures"]
    )
    assert payload["identity_contract"]["package_id_required"] is True
    assert payload["identity_contract"]["package_receipt_required"] is True


def test_all_computed_family_predicates_decide_a_deterministically() -> None:
    pairs = _pairs()
    first = evaluate_c1_v1(pairs)
    second = evaluate_c1_v1(tuple(reversed(pairs)))
    assert first["decision"] == DECISION_A
    assert second == first
    assert all(row["supported"] for row in first["family_support"])
    unsigned = {
        key: value for key, value in first.items() if key != "decision_sha256"
    }
    assert first["decision_sha256"] == content_sha256(unsigned)


def test_support_is_computed_not_supplied_by_caller() -> None:
    assert "supported" not in {
        field.name for field in fields(MatchedPairEvidenceV1)
    }
    assert "link_supported" not in {
        field.name for field in fields(BackendCaseReceiptV1)
    }
    pairs = _pairs()
    target = pairs[-1]
    control_cost = target.control.receipts[0].dimensions["cost"]
    changed_receipts = tuple(
        replace(
            receipt,
            dimensions={
                **receipt.dimensions,
                "cost": control_cost,
            },
        )
        for receipt in target.treatment.receipts
    )
    result = evaluate_c1_v1(
        _replace_pair(
            pairs,
            -1,
            replace(
                target,
                treatment=replace(
                    target.treatment,
                    receipts=changed_receipts,
                ),
            ),
        )
    )
    assert result["decision"] == DECISION_B
    assert result["reasons"] == [
        "negative_family_predicates:pleiotropic_tradeoffs"
    ]


def test_missing_or_ambiguous_family_decides_c() -> None:
    pairs = _pairs()
    assert evaluate_c1_v1(pairs[:-1])["decision"] == DECISION_C
    duplicated = (*pairs[:-1], pairs[0])
    assert evaluate_c1_v1(duplicated)["decision"] == DECISION_C


@pytest.mark.parametrize(
    "defect",
    [
        "package_identity",
        "missing_repeat",
        "backend_parity",
        "fallback",
        "frozen_context",
        "checkpoint",
        "fixture",
        "malformed_dimensions",
        "cross_family_authority",
    ],
)
def test_audit_defects_fail_closed_as_d(defect: str) -> None:
    pairs = _pairs()
    pair = pairs[0]
    control = pair.control
    receipts = list(control.receipts)
    if defect == "package_identity":
        receipts[0] = replace(
            receipts[0],
            authority=replace(
                receipts[0].authority,
                package_id_sha256="0" * 63,
            ),
        )
    elif defect == "missing_repeat":
        receipts.pop()
    elif defect == "backend_parity":
        dimensions = dict(receipts[-1].dimensions)
        phenotype = list(dimensions["phenotype"])
        phenotype[0] += 1
        dimensions["phenotype"] = tuple(phenotype)
        receipts[-1] = replace(receipts[-1], dimensions=dimensions)
    elif defect == "fallback":
        receipts[0] = replace(receipts[0], no_fallback=False)
    elif defect == "frozen_context":
        for index, receipt in enumerate(receipts):
            dimensions = dict(receipt.dimensions)
            context = list(dimensions["context"])
            context[0] = 1
            dimensions["context"] = tuple(context)
            receipts[index] = replace(receipt, dimensions=dimensions)
    elif defect == "checkpoint":
        pair = replace(
            pair,
            checkpoint=replace(pair.checkpoint, exact_restore=False),
        )
    elif defect == "fixture":
        pair = replace(pair, fixture_sha256="0" * 64)
    elif defect == "malformed_dimensions":
        receipts[0] = replace(receipts[0], dimensions={"cost": (1, 2)})
    elif defect == "cross_family_authority":
        changed_authority = replace(
            receipts[0].authority,
            execution_route_sha256="3" * 64,
        )
        receipts = [
            replace(receipt, authority=changed_authority)
            for receipt in receipts
        ]
        treatment_receipts = tuple(
            replace(receipt, authority=changed_authority)
            for receipt in pair.treatment.receipts
        )
        pair = replace(
            pair,
            treatment=replace(
                pair.treatment,
                receipts=treatment_receipts,
            ),
            checkpoint=replace(
                pair.checkpoint,
                authority=changed_authority,
            ),
        )
    if defect in {
        "package_identity",
        "missing_repeat",
        "backend_parity",
        "fallback",
        "frozen_context",
        "malformed_dimensions",
        "cross_family_authority",
    }:
        pair = replace(
            pair,
            control=replace(control, receipts=tuple(receipts)),
        )
    result = evaluate_c1_v1(_replace_pair(pairs, 0, pair))
    assert result["decision"] == DECISION_D
    assert result["reasons"]


def test_invalid_pair_type_fails_closed_as_d() -> None:
    result = evaluate_c1_v1([object()])  # type: ignore[list-item]
    assert result["decision"] == DECISION_D
