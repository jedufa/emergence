"""Exact cross-backend reducer for the Genome V2 environment-switch gate."""

from __future__ import annotations

from typing import Any, Final, Mapping, Sequence

from engine.authoritative_state.schema import content_sha256
from .genome_upgrade_v2_hereditary_capacity_gate_v1 import (
    AUTHORITY_FIELDS,
    PROVIDER_ID,
    RUNTIME_PROFILE_ID,
    validate_authority_v1,
)


SCHEMA_VERSION: Final[str] = "genome_v2_p10_environment_switch_gate_v1"
EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "genome_v2_p10_environment_switch_evidence_v1"
)
SNAPSHOT_SCHEMA_VERSION: Final[str] = (
    "genome_v2_p10_environment_switch_snapshot_v1"
)
CONDITIONS: Final[tuple[str, ...]] = (
    "stable_a",
    "stable_b",
    "a_to_b",
    "b_to_a",
)
BACKENDS: Final[tuple[str, ...]] = ("reference", "native", "cuda")
REPEAT_INDICES: Final[tuple[int, ...]] = (0, 1)
SNAPSHOT_FIELDS: Final[tuple[str, ...]] = (
    "context_q10",
    "products_q10",
    "regulatory_q10",
    "phenotypes_q10",
    "runtime_consequences_q10",
    "expression_cost_q10",
    "regulatory_cost_q10",
    "runtime_cost_q10",
)
DECISION_A: Final[str] = "ENVIRONMENT-SWITCH-A"
DECISION_B: Final[str] = "ENVIRONMENT-SWITCH-B"
DECISION_C: Final[str] = "ENVIRONMENT-SWITCH-C"
DECISION_D: Final[str] = "ENVIRONMENT-SWITCH-D"


def preregistration_payload_v1() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "programme": "GENOME-UPGRADE-V2/P1.0",
        "gate": "environment_switch",
        "provider_id": PROVIDER_ID,
        "runtime_profile_id": RUNTIME_PROFILE_ID,
        "conditions": list(CONDITIONS),
        "backends": list(BACKENDS),
        "repeat_indices": list(REPEAT_INDICES),
        "snapshot_fields": list(SNAPSHOT_FIELDS),
        "trajectory_contract": {
            "stable_a": ["A", "A"],
            "stable_b": ["B", "B"],
            "a_to_b": ["A", "B"],
            "b_to_a": ["B", "A"],
        },
        "exactness": {
            "same_genotype_all_conditions_backends_repeats": True,
            "reference_native_cuda_outputs_and_costs": "exact",
            "repeat_outputs_and_costs": "exact",
            "switch_endpoints_equal_matching_stable_environment": True,
            "environment_a_and_b_outputs_must_differ": True,
            "no_fallback_or_jit": True,
        },
        "decision_contract": {
            DECISION_A: "complete_exact_invariant_genotype_switch_evidence",
            DECISION_B: "complete_valid_evidence_without_environment_response",
            DECISION_C: "incomplete_matrix_or_snapshot",
            DECISION_D: "identity_genotype_parity_or_integrity_failure",
        },
        "claims_boundary": (
            "bounded environment-dependent expression only; no general "
            "adaptation, cognition, intelligence, or open-ended evolution claim"
        ),
    }


PREREGISTRATION_SHA256: Final[str] = content_sha256(
    preregistration_payload_v1()
)


def _snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        *SNAPSHOT_FIELDS,
    }:
        raise ValueError("environment snapshot fields differ from contract")
    if value.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("environment snapshot schema is invalid")
    result = dict(value)
    for name in SNAPSHOT_FIELDS:
        item = result[name]
        if name.endswith("_cost_q10"):
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        elif (
            isinstance(item, (str, bytes))
            or not isinstance(item, Sequence)
            or not item
            or any(isinstance(x, bool) or not isinstance(x, int) for x in item)
        ):
            raise ValueError(f"{name} must be a nonempty integer sequence")
        else:
            result[name] = list(item)
    return result


def _decision(
    decision: str,
    *,
    authority: Mapping[str, str] | None,
    reasons: Sequence[str],
    genotype_sha256: str | None,
    endpoint_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "authority": None if authority is None else dict(authority),
        "decision": decision,
        "reasons": list(reasons),
        "genotype_sha256": genotype_sha256,
        "endpoint_sha256s": dict(sorted(endpoint_sha256s.items())),
        "matrix": {
            "conditions": len(CONDITIONS),
            "backends": len(BACKENDS),
            "repeats": len(REPEAT_INDICES),
            "expected_rows": len(CONDITIONS) * len(BACKENDS) * len(REPEAT_INDICES),
        },
        "claims": {
            "environmental_expression_supported": decision == DECISION_A,
            "general_adaptation": False,
        },
    }
    return {**value, "decision_sha256": content_sha256(value)}


def evaluate_environment_switch_v1(
    evidence: Sequence[Mapping[str, Any]],
    *,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact 4-condition x 3-backend x 2-repeat switch evidence."""

    try:
        canonical_authority = validate_authority_v1(authority)
    except ValueError as exc:
        return _decision(
            DECISION_D,
            authority=None,
            reasons=[str(exc)],
            genotype_sha256=None,
            endpoint_sha256s={},
        )
    expected = {
        (condition, backend, repeat)
        for condition in CONDITIONS
        for backend in BACKENDS
        for repeat in REPEAT_INDICES
    }
    indexed: dict[tuple[str, str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    genotype_hashes: set[str] = set()
    integrity: list[str] = []
    incomplete: list[str] = []
    for raw in evidence:
        if not isinstance(raw, Mapping):
            integrity.append("evidence_row_is_not_an_object")
            continue
        expected_fields = {
            "schema_version",
            "preregistration_sha256",
            "authority",
            "condition",
            "backend",
            "repeat_index",
            "genotype_sha256",
            "initial",
            "terminal",
            "no_fallback",
            "no_jit",
            "evidence_sha256",
        }
        unsigned = dict(raw)
        supplied_hash = unsigned.pop("evidence_sha256", None)
        key = (raw.get("condition"), raw.get("backend"), raw.get("repeat_index"))
        if (
            set(raw) != expected_fields
            or raw.get("schema_version") != EVIDENCE_SCHEMA_VERSION
            or raw.get("preregistration_sha256") != PREREGISTRATION_SHA256
            or supplied_hash != content_sha256(unsigned)
            or key in indexed
        ):
            integrity.append(f"{key!r}:schema_hash_or_duplicate")
            continue
        if key not in expected:
            integrity.append(f"{key!r}:unexpected_cell")
            continue
        try:
            row_authority = validate_authority_v1(raw.get("authority"))
            initial = _snapshot(raw.get("initial"))
            terminal = _snapshot(raw.get("terminal"))
        except ValueError as exc:
            incomplete.append(f"{key!r}:{exc}")
            continue
        if row_authority != canonical_authority:
            integrity.append(f"{key!r}:mixed_authority")
        genotype = raw.get("genotype_sha256")
        if (
            not isinstance(genotype, str)
            or len(genotype) != 64
            or any(character not in "0123456789abcdef" for character in genotype)
        ):
            integrity.append(f"{key!r}:invalid_genotype_sha256")
        else:
            genotype_hashes.add(genotype)
        if type(raw.get("no_fallback")) is not bool or type(raw.get("no_jit")) is not bool:
            incomplete.append(f"{key!r}:execution_flags")
        elif not raw["no_fallback"] or not raw["no_jit"]:
            integrity.append(f"{key!r}:fallback_or_jit")
        indexed[key] = (initial, terminal)
    missing = sorted(expected - set(indexed))
    if len(genotype_hashes) > 1:
        integrity.append("genotype_changed_across_environment_matrix")
    if integrity:
        return _decision(
            DECISION_D,
            authority=canonical_authority,
            reasons=integrity,
            genotype_sha256=next(iter(genotype_hashes), None),
            endpoint_sha256s={},
        )
    if missing or incomplete:
        incomplete_reasons = list(incomplete)
        if missing:
            incomplete_reasons.insert(0, f"missing_cells:{missing}")
        return _decision(
            DECISION_C,
            authority=canonical_authority,
            reasons=incomplete_reasons,
            genotype_sha256=next(iter(genotype_hashes), None),
            endpoint_sha256s={},
        )
    reference: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    parity_failures: list[str] = []
    for condition in CONDITIONS:
        baseline = indexed[(condition, "reference", 0)]
        reference[condition] = baseline
        for backend in BACKENDS:
            for repeat in REPEAT_INDICES:
                if indexed[(condition, backend, repeat)] != baseline:
                    parity_failures.append(
                        f"{condition}:{backend}:{repeat}:output_or_cost_mismatch"
                    )
    if parity_failures:
        return _decision(
            DECISION_D,
            authority=canonical_authority,
            reasons=parity_failures,
            genotype_sha256=next(iter(genotype_hashes), None),
            endpoint_sha256s={},
        )
    stable_a = reference["stable_a"]
    stable_b = reference["stable_b"]
    a_to_b = reference["a_to_b"]
    b_to_a = reference["b_to_a"]
    endpoint_hashes = {
        condition: content_sha256({"initial": initial, "terminal": terminal})
        for condition, (initial, terminal) in reference.items()
    }
    stable = stable_a[0] == stable_a[1] and stable_b[0] == stable_b[1]
    endpoint_exact = (
        a_to_b[0] == stable_a[1]
        and a_to_b[1] == stable_b[1]
        and b_to_a[0] == stable_b[1]
        and b_to_a[1] == stable_a[1]
    )
    response = stable_a[1] != stable_b[1]
    promoted = stable and endpoint_exact and response
    return _decision(
        DECISION_A if promoted else DECISION_B,
        authority=canonical_authority,
        reasons=(
            []
            if promoted
            else [
                *([] if stable else ["stable_environment_not_stable"]),
                *([] if endpoint_exact else ["switch_endpoints_not_exact"]),
                *([] if response else ["environment_outputs_do_not_differ"]),
            ]
        ),
        genotype_sha256=next(iter(genotype_hashes), None),
        endpoint_sha256s=endpoint_hashes,
    )


__all__ = [
    "AUTHORITY_FIELDS",
    "BACKENDS",
    "CONDITIONS",
    "DECISION_A",
    "DECISION_B",
    "DECISION_C",
    "DECISION_D",
    "EVIDENCE_SCHEMA_VERSION",
    "PREREGISTRATION_SHA256",
    "REPEAT_INDICES",
    "SCHEMA_VERSION",
    "SNAPSHOT_FIELDS",
    "SNAPSHOT_SCHEMA_VERSION",
    "evaluate_environment_switch_v1",
    "preregistration_payload_v1",
]
