"""Deterministic terminal reducer for Genome V2 hereditary capacity evidence."""

from __future__ import annotations

from typing import Any, Final, Mapping, Sequence

from engine.authoritative_state.schema import content_sha256


SCHEMA_VERSION: Final[str] = "genome_v2_p10_hereditary_capacity_gate_v1"
EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "genome_v2_p10_hereditary_capacity_evidence_v1"
)
PROVIDER_ID: Final[str] = "eiel-production-provider-v1"
RUNTIME_PROFILE_ID: Final[str] = "eiel-production-runtime-v1"
REPEAT_INDICES: Final[tuple[int, ...]] = (0, 1)
DECISION_A: Final[str] = "HEREDITARY-CAPACITY-A"
DECISION_B: Final[str] = "HEREDITARY-CAPACITY-B"
DECISION_C: Final[str] = "HEREDITARY-CAPACITY-C"
DECISION_D: Final[str] = "HEREDITARY-CAPACITY-D"
AUTHORITY_FIELDS: Final[tuple[str, ...]] = (
    "provider_id",
    "runtime_profile_id",
    "source_commit",
    "package_id_sha256",
    "package_receipt_sha256",
    "artifact_sha256",
    "execution_route_sha256",
)
METRIC_FIELDS: Final[tuple[str, ...]] = (
    "allele_count",
    "genotype_count",
    "heterozygosity_q10",
    "hidden_allele_count",
    "retained_hidden_allele_count",
    "fixed_allele_count",
    "lost_allele_count",
    "retained_package_count",
    "linkage_disequilibrium_abs_q10",
    "homogenization_generation",
    "expression_cost_q10",
    "runtime_cost_q10",
)


def preregistration_payload_v1() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "programme": "GENOME-UPGRADE-V2/P1.0",
        "gate": "expanded_hereditary_capacity",
        "provider_id": PROVIDER_ID,
        "runtime_profile_id": RUNTIME_PROFILE_ID,
        "required_repeat_indices": list(REPEAT_INDICES),
        "campaign_design": {
            "world": "patch",
            "overlays": [],
            "policy_family": "fixed_mlp",
            "selection_mode": "standard",
            "grid_size": 16,
            "initial_population": 64,
            "resident_capacity": 256,
            "founder_placement": "balanced",
            "genome_profile": "genome_upgrade_v2_fixed_q10_v1",
            "founder_profile": "pleiotropic_tradeoffs",
            "mutation_rate_q10": 20,
            "mutation_step_q10": 64,
            "ecology_seeds": [101, 202],
            "task_seeds": [3003],
            "backend": "cuda_resident",
            "logical_cells": 2,
            "physical_batch_maximum": 2,
            "generations": 3,
            "ticks_per_generation": 80,
            "checkpoint_generations": [0, 3],
            "digest_every_generation": True,
            "checkpoint_storage_mode": "disk",
            "fresh_repeats": 0,
            "continuation_cells": 0,
        },
        "required_metrics": list(METRIC_FIELDS),
        "metric_contract": {
            "heterozygosity_q10": [0, 1024],
            "linkage_disequilibrium_abs_q10": [0, 1024],
            "homogenization_generation": (
                "exact_nonnegative_integer_only_with_contiguous_observations_or_"
                "null_when_terminal_remains_diverse"
            ),
            "counts_and_costs": "nonnegative_integer",
        },
        "promotion_predicate": {
            "allele_count_minimum": 2,
            "genotype_count_minimum": 2,
            "heterozygosity_strictly_positive": True,
            "hidden_allele_retention_strictly_positive": True,
            "package_retention_strictly_positive": True,
            "expression_or_runtime_cost_strictly_positive": True,
            "causal_effect_required": True,
            "cost_tradeoff_required": True,
            "mutation_reachability_required": True,
            "immediate_cap_saturation_forbidden": True,
            "no_fallback_or_jit": True,
        },
        "decision_contract": {
            DECISION_A: "complete_exact_repeated_evidence_meets_predicate",
            DECISION_B: "complete_exact_repeated_evidence_does_not_meet_predicate",
            DECISION_C: "incomplete_repeat_or_metric_evidence",
            DECISION_D: "identity_integrity_or_schema_failure",
        },
        "claims_boundary": (
            "bounded hereditary capacity and cost evidence only; no ecological "
            "usefulness, cognition, intelligence, or open-ended evolution claim"
        ),
    }


PREREGISTRATION_SHA256: Final[str] = content_sha256(
    preregistration_payload_v1()
)


def _hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_authority_v1(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(AUTHORITY_FIELDS):
        raise ValueError("hereditary-capacity authority fields differ from contract")
    authority = {name: value[name] for name in AUTHORITY_FIELDS}
    if (
        authority["provider_id"] != PROVIDER_ID
        or authority["runtime_profile_id"] != RUNTIME_PROFILE_ID
        or not _commit(authority["source_commit"])
        or any(
            not _hash(authority[name])
            for name in AUTHORITY_FIELDS
            if name.endswith("_sha256")
        )
    ):
        raise ValueError("hereditary-capacity authority is invalid")
    return authority


def _decision(
    decision: str,
    *,
    authority: Mapping[str, str] | None,
    reasons: Sequence[str],
    repeat_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "authority": None if authority is None else dict(authority),
        "decision": decision,
        "reasons": list(reasons),
        "repeat_summaries": [dict(row) for row in repeat_summaries],
        "claims": {
            "hereditary_capacity_supported": decision == DECISION_A,
            "ecological_usefulness": False,
        },
    }
    return {**value, "decision_sha256": content_sha256(value)}


def evaluate_hereditary_capacity_v1(
    evidence: Sequence[Mapping[str, Any]],
    *,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce exactly two content- and authority-bound hereditary observations."""

    try:
        canonical_authority = validate_authority_v1(authority)
    except ValueError as exc:
        return _decision(
            DECISION_D,
            authority=None,
            reasons=[str(exc)],
            repeat_summaries=[],
        )
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        return _decision(
            DECISION_D,
            authority=canonical_authority,
            reasons=["evidence_must_be_a_sequence"],
            repeat_summaries=[],
        )
    indexed: dict[int, Mapping[str, Any]] = {}
    integrity: list[str] = []
    incomplete: list[str] = []
    summaries: list[dict[str, Any]] = []
    for raw in evidence:
        if not isinstance(raw, Mapping):
            integrity.append("evidence_row_is_not_an_object")
            continue
        repeat = raw.get("repeat_index")
        if type(repeat) is not int or repeat in indexed:
            integrity.append("invalid_or_duplicate_repeat_index")
            continue
        indexed[repeat] = raw
        expected_fields = {
            "schema_version",
            "preregistration_sha256",
            "authority",
            "repeat_index",
            "metrics",
            "causal_effect_observed",
            "cost_tradeoff_observed",
            "mutation_reachable",
            "immediate_cap_saturation",
            "no_fallback",
            "no_jit",
            "evidence_sha256",
        }
        unsigned = dict(raw)
        supplied_hash = unsigned.pop("evidence_sha256", None)
        if (
            set(raw) != expected_fields
            or raw.get("schema_version") != EVIDENCE_SCHEMA_VERSION
            or raw.get("preregistration_sha256") != PREREGISTRATION_SHA256
            or supplied_hash != content_sha256(unsigned)
        ):
            integrity.append(f"repeat_{repeat}:schema_or_hash_mismatch")
            continue
        try:
            row_authority = validate_authority_v1(raw.get("authority"))
        except ValueError:
            integrity.append(f"repeat_{repeat}:invalid_authority")
            continue
        if row_authority != canonical_authority:
            integrity.append(f"repeat_{repeat}:mixed_authority")
        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_FIELDS):
            incomplete.append(f"repeat_{repeat}:metric_fields")
            continue
        for name, value in metrics.items():
            if name == "homogenization_generation" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                incomplete.append(f"repeat_{repeat}:invalid_{name}")
        for name in ("heterozygosity_q10", "linkage_disequilibrium_abs_q10"):
            value = metrics.get(name)
            if type(value) is int and value > 1024:
                incomplete.append(f"repeat_{repeat}:{name}_out_of_bounds")
        if (
            type(metrics.get("retained_hidden_allele_count")) is int
            and type(metrics.get("hidden_allele_count")) is int
            and metrics["retained_hidden_allele_count"]
            > metrics["hidden_allele_count"]
        ):
            integrity.append(f"repeat_{repeat}:retention_exceeds_hidden_count")
        boolean_fields = (
            "causal_effect_observed",
            "cost_tradeoff_observed",
            "mutation_reachable",
            "immediate_cap_saturation",
            "no_fallback",
            "no_jit",
        )
        if any(type(raw.get(name)) is not bool for name in boolean_fields):
            incomplete.append(f"repeat_{repeat}:boolean_fields")
            continue
        summaries.append(
            {
                "repeat_index": repeat,
                "metrics": dict(metrics),
                **{name: raw[name] for name in boolean_fields},
                "evidence_sha256": supplied_hash,
            }
        )
    missing = sorted(set(REPEAT_INDICES) - set(indexed))
    unexpected = sorted(set(indexed) - set(REPEAT_INDICES))
    if unexpected:
        integrity.append(f"unexpected_repeats:{unexpected}")
    if integrity:
        return _decision(
            DECISION_D,
            authority=canonical_authority,
            reasons=integrity,
            repeat_summaries=summaries,
        )
    if missing or incomplete:
        incomplete_reasons = list(incomplete)
        if missing:
            incomplete_reasons.insert(0, f"missing_repeats:{missing}")
        return _decision(
            DECISION_C,
            authority=canonical_authority,
            reasons=incomplete_reasons,
            repeat_summaries=summaries,
        )
    promoted = all(
        row["metrics"]["allele_count"] >= 2
        and row["metrics"]["genotype_count"] >= 2
        and row["metrics"]["heterozygosity_q10"] > 0
        and row["metrics"]["retained_hidden_allele_count"] > 0
        and row["metrics"]["retained_package_count"] > 0
        and (
            row["metrics"]["expression_cost_q10"] > 0
            or row["metrics"]["runtime_cost_q10"] > 0
        )
        and row["causal_effect_observed"]
        and row["cost_tradeoff_observed"]
        and row["mutation_reachable"]
        and not row["immediate_cap_saturation"]
        and row["no_fallback"]
        and row["no_jit"]
        for row in summaries
    )
    return _decision(
        DECISION_A if promoted else DECISION_B,
        authority=canonical_authority,
        reasons=[] if promoted else ["promotion_predicate_not_met"],
        repeat_summaries=sorted(summaries, key=lambda row: row["repeat_index"]),
    )


__all__ = [
    "AUTHORITY_FIELDS",
    "DECISION_A",
    "DECISION_B",
    "DECISION_C",
    "DECISION_D",
    "EVIDENCE_SCHEMA_VERSION",
    "METRIC_FIELDS",
    "PREREGISTRATION_SHA256",
    "REPEAT_INDICES",
    "SCHEMA_VERSION",
    "evaluate_hereditary_capacity_v1",
    "preregistration_payload_v1",
    "validate_authority_v1",
]
