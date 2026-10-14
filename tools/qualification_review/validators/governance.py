"""Authority-transition preconditions and semantic risk triggers."""

from __future__ import annotations

from ..io import InputStore, ReviewInputError
from ..model import CheckResult, SemanticQuestion
from .assertions import validate_inline_assertion


def _question(
    spec: dict,
    trigger_type: str,
    template: str,
    evidence: list[str],
) -> SemanticQuestion:
    return SemanticQuestion(
        trigger_id=f"{spec['id']}:{trigger_type}",
        trigger_type=trigger_type,
        question=template.format(transition_id=spec["id"]),
        evidence_pointers=tuple(sorted(set(evidence))),
    )


def validate_authority_transition(
    spec: dict,
    store: InputStore,
    rules: dict,
) -> tuple[CheckResult, list[SemanticQuestion]]:
    result = CheckResult(spec["id"], "authority_transition")
    questions: list[SemanticQuestion] = []
    evidence = list(spec.get("evidence_pointers", []))
    try:
        for precondition in spec.get("preconditions", []):
            document = store.read_json(precondition["path"])
            error = validate_inline_assertion(document, precondition)
            if error:
                result.fail(f"precondition {precondition['id']!r}: {error}")
        templates = rules["question_templates"]
        if spec["current_authority"] != spec["proposed_authority"]:
            questions.append(
                _question(spec, "authority_change", templates["authority_change"], evidence)
            )
        if spec.get("changed_runtime_dependencies"):
            questions.append(
                _question(
                    spec,
                    "changed_runtime_dependency",
                    templates["changed_runtime_dependency"],
                    evidence + spec["changed_runtime_dependencies"],
                )
            )
        changed_boundaries = [
            row["evidence_pointer"]
            for row in spec.get("inherited_evidence_boundaries", [])
            if row["dependency_changed"]
        ]
        if changed_boundaries:
            questions.append(
                _question(
                    spec,
                    "inheritance_boundary_change",
                    templates["inheritance_boundary_change"],
                    evidence + changed_boundaries,
                )
            )
        conditions = spec.get("evidence_conditions", [])
        risky_conditions = [
            row["evidence_pointer"]
            for row in conditions
            if row["status"] in rules["judgment_evidence_statuses"]
        ]
        if risky_conditions:
            questions.append(
                _question(
                    spec,
                    "evidence_condition",
                    templates["evidence_condition"],
                    evidence + risky_conditions,
                )
            )
        interpretation = spec.get("limitations", []) + spec.get("no_claims", [])
        if interpretation:
            questions.append(
                _question(
                    spec,
                    "limitations_or_no_claims",
                    templates["limitations_or_no_claims"],
                    evidence + interpretation,
                )
            )
        decision = spec.get("decision", {})
        if decision.get("irreversible", False):
            questions.append(
                _question(
                    spec,
                    "irreversible_decision",
                    templates["irreversible_decision"],
                    evidence + decision.get("evidence_pointers", []),
                )
            )
        result.details = {
            "current_authority": spec["current_authority"],
            "proposed_authority": spec["proposed_authority"],
            "precondition_count": len(spec.get("preconditions", [])),
            "risk_trigger_count": len(questions),
        }
    except ReviewInputError as exc:
        result.fail(str(exc))
    return result, questions
