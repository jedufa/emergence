from __future__ import annotations

import copy

import pytest

from eiel_composition.adaptive_expedition.closeout import (
    ENGINEERING_G5_TERMINAL_STATUS_V1,
    REVIEW_SECTION_KEYS_V1,
    ExpeditionCloseoutError,
    build_expedition_closeout_v1,
    validate_expedition_closeout_v1,
)
from eiel_composition.adaptive_expedition.review_projection import (
    REVIEW_SECTION_SCHEMA_V1,
    ExpeditionReviewProjectionError,
    validate_human_review_projection_v1,
)


def _digest(character: str) -> dict[str, str]:
    return {"algorithm": "sha256", "hex": character * 64}


def _closeout() -> dict[str, object]:
    return build_expedition_closeout_v1(
        expedition_id="EXPEDITION-TEST-V1",
        terminal_status=ENGINEERING_G5_TERMINAL_STATUS_V1,
        programme_stage="G5_ENGINEERING",
        source_commit="a" * 40,
        manifest_sha256=_digest("1"),
        policy_sha256=_digest("2"),
        ledger_sha256=_digest("3"),
        atlas_sha256=_digest("4"),
        retained_parent_bank_sha256=_digest("5"),
        review_projection={
            key: {
                "schema_version": REVIEW_SECTION_SCHEMA_V1,
                "question_ordinal": ordinal,
                "section_key": key,
                "status_code": "NO_ROWS_RECORDED",
                "rows": [],
                "evidence_refs": [],
            }
            for ordinal, key in enumerate(REVIEW_SECTION_KEYS_V1, start=1)
        },
        performance_summary={"measurements": []},
        lifecycle_summary={"unresolved_rx": 0},
        validation_summary={"gates": ["G0", "G1"]},
    )


def test_closeout_is_deterministic_and_answers_exactly_thirteen_questions() -> None:
    first = _closeout()
    second = _closeout()
    assert first == second
    assert len(first["review"]) == 13
    assert validate_expedition_closeout_v1(first) == first
    assert first["qualification_status"] == "NOT_REQUESTED"
    assert first["programme_stage"] == "G5_ENGINEERING"


def test_closeout_rejects_missing_review_section_and_non_g5_terminal() -> None:
    review = {
        key: {
            "schema_version": REVIEW_SECTION_SCHEMA_V1,
            "question_ordinal": ordinal,
            "section_key": key,
            "status_code": "NO_ROWS_RECORDED",
            "rows": [],
            "evidence_refs": [],
        }
        for ordinal, key in enumerate(REVIEW_SECTION_KEYS_V1[:-1], start=1)
    }
    with pytest.raises(ExpeditionCloseoutError, match="thirteen"):
        build_expedition_closeout_v1(
            expedition_id="EXPEDITION-TEST-V1",
            terminal_status=ENGINEERING_G5_TERMINAL_STATUS_V1,
            programme_stage="G5_ENGINEERING",
            source_commit="a" * 40,
            manifest_sha256=_digest("1"),
            policy_sha256=_digest("2"),
            ledger_sha256=_digest("3"),
            atlas_sha256=_digest("4"),
            retained_parent_bank_sha256=_digest("5"),
            review_projection=review,
            performance_summary={},
            lifecycle_summary={},
            validation_summary={},
        )
    with pytest.raises(ExpeditionCloseoutError, match="terminal"):
        build_expedition_closeout_v1(
            expedition_id="EXPEDITION-TEST-V1",
            terminal_status="WEEK_COMPLETE",
            programme_stage="REAL_SCIENCE_USE",
            source_commit="a" * 40,
            manifest_sha256=_digest("1"),
            policy_sha256=_digest("2"),
            ledger_sha256=_digest("3"),
            atlas_sha256=_digest("4"),
            retained_parent_bank_sha256=_digest("5"),
            review_projection=_closeout()["review"],
            performance_summary={},
            lifecycle_summary={},
            validation_summary={},
        )


def test_closeout_tampering_fails_digest_recomputation() -> None:
    value = _closeout()
    tampered = copy.deepcopy(value)
    tampered["review"]["unresolved_scientific_questions"] = {"rows": ["changed"]}
    with pytest.raises(ExpeditionCloseoutError):
        validate_expedition_closeout_v1(tampered)


@pytest.mark.parametrize(
    "forbidden_field",
    ["scientific_claim", "Qualification_status", "execution_authority"],
)
def test_review_recursively_rejects_claim_qualification_and_authority_fields(
    forbidden_field: str,
) -> None:
    review = copy.deepcopy(_closeout()["review"])
    section = review["human_scientific_choices"]
    section["status_code"] = "RECORDED_INDEX_ROWS_NO_CLAIMS"
    section["rows"] = [
        {
            "row_kind": "POLICY_PRESENTATION_HUMAN_CHOICE",
            "choice_id": "HUMAN_DECISION",
            "choice": "A human selects the next bounded question.",
            "must_remain_human": True,
            "metadata": {forbidden_field: "FORBIDDEN"},
        }
    ]
    with pytest.raises(ExpeditionReviewProjectionError, match="forbidden"):
        validate_human_review_projection_v1(review)


def test_review_requires_exact_row_and_direct_evidence_shapes() -> None:
    review = copy.deepcopy(_closeout()["review"])
    section = review["human_scientific_choices"]
    section["status_code"] = "RECORDED_INDEX_ROWS_NO_CLAIMS"
    section["rows"] = [
        {
            "row_kind": "POLICY_PRESENTATION_HUMAN_CHOICE",
            "choice_id": "HUMAN_DECISION",
            "choice": "A human selects the next bounded question.",
            "must_remain_human": True,
            "notes": [],
        }
    ]
    with pytest.raises(ExpeditionReviewProjectionError, match="non-exact"):
        validate_human_review_projection_v1(review)

    evidence_review = copy.deepcopy(_closeout()["review"])
    evidence_review["authorized_search_space"]["evidence_refs"] = [
        {
            "artifact_id": "EXPEDITION_MANIFEST",
            "relative_path": "../../unbound.json",
            "content_sha256": _digest("a"),
        }
    ]
    with pytest.raises(ExpeditionReviewProjectionError, match="direct artifact path"):
        validate_human_review_projection_v1(evidence_review)


def test_discarded_parent_frontier_index_is_bounded() -> None:
    review = copy.deepcopy(_closeout()["review"])
    section = review["discarded_parent_candidates"]
    section["status_code"] = "RECORDED_INDEX_ROWS_NO_CLAIMS"
    section["rows"] = [
        {
            "row_kind": "POLICY_PARENT_DISCARD_REASON_INDEX",
            "reason_code": "BOUNDED_BANK_BUDGET_LOWER_PRIORITY",
            "distinct_state_count": 1,
            "state_ref_id_samples": ["SR-1"],
            "state_ref_ids_omitted": 0,
            "frontier_ordinal_samples": list(range(25)),
            "frontier_ordinals_omitted": 0,
            "decision_sha256_samples": ["a" * 64],
            "decision_sha256s_omitted": 0,
        }
    ]
    with pytest.raises(ExpeditionReviewProjectionError, match="ordinal-array"):
        validate_human_review_projection_v1(review)
