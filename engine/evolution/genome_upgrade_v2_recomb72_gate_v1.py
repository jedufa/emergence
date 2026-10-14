"""Preregistered reducer for the bounded Genome V2 RECOMB-72 campaign."""

from __future__ import annotations

from typing import Any, Final, Mapping, Sequence

from engine.authoritative_state.schema import content_sha256

from .genome_upgrade_v2_recomb72_founders_v1 import (
    FOUNDER_PATTERN_IDS,
    FOUNDER_PROFILE_IDS,
    FOUNDER_STATE_IDS,
    resolve_recomb72_gv2_founder_profile_v1,
)


SCHEMA_VERSION: Final[str] = "genome_v2_recomb72_gate_v1"
CELL_SCHEMA_VERSION: Final[str] = "genome_v2_recomb72_cell_summary_v1"
SEED_AUDIT_SCHEMA_VERSION: Final[str] = (
    "genome_v2_recomb72_seed_authority_v1"
)
PROVIDER_ID: Final[str] = "eiel-production-provider-v1"
RUNTIME_PROFILE_ID: Final[str] = "eiel-production-runtime-v1"
DESIGN_LABEL: Final[str] = (
    "Ecology-blocked deterministic design with independent profile-specific "
    "selection tapes."
)
ECOLOGY_SEEDS: Final[tuple[int, ...]] = (101, 202, 303, 404, 505, 606)
TASK_SEED: Final[int] = 0
MILESTONES: Final[tuple[int, ...]] = (0, 5, 10, 20, 30)
POST_FOUNDER_MILESTONES: Final[tuple[int, ...]] = (5, 10, 20, 30)
MIXED_STATE_IDS: Final[tuple[str, ...]] = (
    "all_heterozygous_ab",
    "balanced_homozygous_aa_bb",
)
CONTROL_STATE_IDS: Final[tuple[str, ...]] = (
    "homozygous_aa_control",
    "homozygous_bb_control",
)
HIDDEN_PATTERN_ID: Final[str] = "context_conditional_hidden_pair"
CORRELATED_PATTERN_ID: Final[str] = (
    "correlated_multi_product_linkage_packages"
)
PACKAGE_LOCKING_THRESHOLD_Q10: Final[int] = 896
MINIMUM_RECOMBINANT_BLOCKS: Final[int] = 4
MINIMUM_RETENTION_PATTERNS: Final[int] = 2
DECISION_A: Final[str] = "RECOMB72-GV2-A"
DECISION_B: Final[str] = "RECOMB72-GV2-B"
DECISION_C: Final[str] = "RECOMB72-GV2-C"
DECISION_D: Final[str] = "RECOMB72-GV2-D"
AUTHORITY_FIELDS: Final[tuple[str, ...]] = (
    "provider_id",
    "runtime_profile_id",
    "source_commit",
    "package_id_sha256",
    "package_receipt_sha256",
    "artifact_sha256",
    "execution_route_sha256",
)
MILESTONE_FIELDS: Final[tuple[str, ...]] = (
    "generation",
    "population_alive",
    "endpoint_a_copies",
    "endpoint_b_copies",
    "hidden_genotype_count",
    "intact_package_a_copies",
    "intact_package_b_copies",
    "recombinant_or_partial_copies",
    "linkage_disequilibrium_abs_q10",
)
CELL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "preregistration_sha256",
        "authority",
        "seed_audit_sha256",
        "profile_id",
        "founder_population_sha256",
        "pattern_id",
        "state_id",
        "ecology_seed",
        "task_seed",
        "backend",
        "mutation_rate_q10",
        "mutation_step_q10",
        "ordinary_births_enabled",
        "no_fallback",
        "no_jit",
        "checkpoint_exact",
        "reporting_exact",
        "chronicle_exact",
        "cancellation_clean",
        "cross_cell_isolation_exact",
        "paired_delta_claimed",
        "shared_selection_noise_claimed",
        "milestones",
        "cell_sha256",
    }
)
OUTCOME_IDS: Final[tuple[str, ...]] = (
    "allele_retention",
    "hidden_genotype_retention",
    "intact_package_retention",
    "recombinant_partial_appearance_and_g30_coexistence",
    "linkage_decay_descriptive",
    "control_integrity",
)


def preregistration_payload_v1() -> dict[str, Any]:
    """Return the complete frozen RECOMB-72-GV2 decision contract."""

    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "recomb72-gv2-hereditary-capacity-audit",
        "design_label": DESIGN_LABEL,
        "provider_id": PROVIDER_ID,
        "runtime_profile_id": RUNTIME_PROFILE_ID,
        "campaign_design": {
            "world": "patch",
            "overlays": [],
            "policy_family": "fixed_mlp",
            "campaign_selection_mode": "genotype_blind_neutral",
            "turnover_selection_mode": "genotype_blind_neutral",
            "grid_size": 16,
            "initial_population": 64,
            "resident_capacity": 256,
            "founder_placement": "balanced",
            "genome_profile": "genome_upgrade_v2_fixed_q10_v1",
            "founder_profile_ids": list(FOUNDER_PROFILE_IDS),
            "founder_pattern_ids": list(FOUNDER_PATTERN_IDS),
            "founder_state_ids": list(FOUNDER_STATE_IDS),
            "ecology_seeds": list(ECOLOGY_SEEDS),
            "task_seed": TASK_SEED,
            "logical_cells": 72,
            "physical_batch": "b72",
            "physical_batch_maximum": 72,
            "generations": 30,
            "ticks_per_generation": 80,
            "run_label": "b72g30t80",
            "milestone_generations": list(MILESTONES),
            "digest_every_generation": True,
            "checkpoint_storage_mode": "disk",
            "reporting_profile": "compact",
            "chronicle_profile": "scientific_full",
            "chronicle_archive_enabled": True,
            "evidence_closeout_enabled": False,
            "mutation_rate_q10": 0,
            "mutation_step_q10": 0,
            "ordinary_births_enabled": False,
            "turnover_enabled": True,
            "turnover_cull_fraction": 0.5,
            "turnover_parent_fraction": 1.0,
            "fresh_repeats": 0,
            "continuation_cells": 0,
            "cancellation": True,
            "backend": "cuda_resident",
            "no_fallback": True,
            "no_jit": True,
        },
        "seed_and_claim_contract": {
            "ecology_blocking": "six blocks of twelve founder profiles",
            "selection_tapes": "independent_profile_specific",
            "fully_paired_trajectory_claims_allowed": False,
            "forbidden_claims": [
                "paired_delta",
                "shared_selection_noise",
            ],
            "aggregation_rule": (
                "count exact cell-level predicates within ecology blocks; "
                "never subtract profile trajectories as paired observations"
            ),
        },
        "cell_input": {
            "schema_version": CELL_SCHEMA_VERSION,
            "milestone_fields": list(MILESTONE_FIELDS),
            "exact_rows": 72,
            "one_row_per_founder_profile_and_ecology_seed": True,
            "content_addressed": True,
            "authority_bound": True,
            "seed_audit_bound": True,
        },
        "aggregation_scopes": {
            "ecology_block": (
                "one ecology seed across twelve independently selected profiles"
            ),
            "mixed_state": list(MIXED_STATE_IDS),
            "control_state": list(CONTROL_STATE_IDS),
            "pattern": list(FOUNDER_PATTERN_IDS),
            "cross_profile_pairing": "forbidden",
        },
        "promotion_predicate": {
            "control_integrity": (
                "both absorbing controls exact at every milestone in every "
                "pattern and all six ecology blocks"
            ),
            "allele_retention_g30": (
                "both endpoints positive at G30 in both mixed states and all "
                "six ecology blocks for at least two patterns"
            ),
            "endpoint_persistence": (
                "both endpoints positive at G5,G10,G20,G30 in both mixed "
                "states and all six ecology blocks for at least two patterns"
            ),
            "hidden_genotype_retention": (
                "hidden genotype count positive at G30 in both mixed states "
                "for all six hidden-pattern ecology blocks"
            ),
            "intact_package_retention": (
                "both intact endpoint package-copy counts positive at G30 in "
                "both mixed states for all six correlated-pattern blocks"
            ),
            "recombinant_or_partial": (
                "appears by G30 and coexists at G30 with an intact package in "
                "at least four of six ecology blocks for at least one "
                "correlated mixed state"
            ),
            "minimum_retention_patterns": MINIMUM_RETENTION_PATTERNS,
            "minimum_recombinant_blocks": MINIMUM_RECOMBINANT_BLOCKS,
        },
        "package_locking_risk": {
            "threshold_q10": PACKAGE_LOCKING_THRESHOLD_Q10,
            "threshold_rationale": (
                "frozen seven-eighths concentration boundary relative to the "
                "exact 50/50 G0 mixed-state endpoint geometry"
            ),
            "denominator": (
                "intact_a_plus_intact_b_plus_recombinant_or_partial_copies"
            ),
            "predicate": (
                "max_intact_package_copies_times_1024_gte_threshold_q10_times_"
                "denominator"
            ),
            "minimum_supporting_blocks": 4,
            "scope": "at_least_one_correlated_mixed_state",
            "classification": (
                "B only when all retention predicates pass, recombinant "
                "coexistence A predicate fails, and concentration risk passes"
            ),
        },
        "linkage_decay": {
            "scope": "each correlated mixed-state ecology block",
            "measure": "G0_minus_G30_linkage_disequilibrium_abs_q10",
            "interpretation": "descriptive_only",
            "must_be_separately_evaluable": True,
            "may_mask_retention": False,
            "minimum_supporting_blocks": 12,
        },
        "outcomes": list(OUTCOME_IDS),
        "decision_contract": {
            DECISION_D: "integrity_failure_or_not_evaluable",
            DECISION_A: "all_frozen_retention_and_recombination_predicates_pass",
            DECISION_B: "a_like_retention_with_frozen_package_locking_risk",
            DECISION_C: "complete_interpretable_failure",
        },
        "claims_boundary": (
            "bounded deterministic hereditary retention and recombination "
            "description only; no paired-delta, shared-selection-noise, "
            "fitness, ecological advantage, evolvability, cognition, "
            "intelligence, or emergence claim"
        ),
    }


PREREGISTRATION_SHA256: Final[str] = content_sha256(
    preregistration_payload_v1()
)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_authority(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(AUTHORITY_FIELDS):
        raise ValueError("RECOMB-72 authority fields differ from contract")
    result = {name: value[name] for name in AUTHORITY_FIELDS}
    if (
        result["provider_id"] != PROVIDER_ID
        or result["runtime_profile_id"] != RUNTIME_PROFILE_ID
        or not _valid_commit(result["source_commit"])
        or any(
            not _valid_sha256(result[name])
            for name in AUTHORITY_FIELDS
            if name.endswith("_sha256")
        )
    ):
        raise ValueError("RECOMB-72 authority is invalid")
    return result


def _validate_seed_audit(
    value: Any,
    *,
    authority: Mapping[str, str],
) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("seed audit must be an object")
    supplied_hash = value.get("audit_sha256")
    unsigned = dict(value)
    unsigned.pop("audit_sha256", None)
    checks = value.get("checks")
    matrix = value.get("matrix_contract")
    audit_authority = value.get("authority")
    if (
        value.get("schema_version") != SEED_AUDIT_SCHEMA_VERSION
        or not _valid_sha256(supplied_hash)
        or supplied_hash != content_sha256(unsigned)
        or value.get("design_label") != DESIGN_LABEL
        or value.get("fully_paired_claims_allowed") is not False
        or not isinstance(checks, Mapping)
        or not checks
        or any(item is not True for item in checks.values())
        or not isinstance(matrix, Mapping)
        or matrix.get("cell_count") != 72
        or matrix.get("founder_profile_ids") != list(FOUNDER_PROFILE_IDS)
        or matrix.get("ecology_seeds") != list(ECOLOGY_SEEDS)
        or matrix.get("task_seed") != TASK_SEED
        or matrix.get("physical_batch_shape") != "b72"
        or not isinstance(audit_authority, Mapping)
        or audit_authority.get("package_id_sha256")
        != authority["package_id_sha256"]
        or audit_authority.get("execution_route_sha256")
        != authority["execution_route_sha256"]
        or audit_authority.get("world_identity") != "patch"
        or not _valid_sha256(audit_authority.get("configuration_sha256"))
    ):
        raise ValueError("seed audit is absent, stale, or contradicts the design")
    return supplied_hash


def _profile_parts(profile_id: str) -> tuple[str, str]:
    prefix, pattern_id, state_id = profile_id.split(".", maxsplit=2)
    if (
        prefix != "recomb72_gv2_v1"
        or pattern_id not in FOUNDER_PATTERN_IDS
        or state_id not in FOUNDER_STATE_IDS
    ):
        raise ValueError("founder profile identity is invalid")
    return pattern_id, state_id


def _validate_milestones(value: Any) -> dict[int, dict[str, int]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("milestones must be a sequence")
    indexed: dict[int, dict[str, int]] = {}
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != set(MILESTONE_FIELDS):
            raise ValueError("milestone fields differ from contract")
        generation = raw.get("generation")
        if type(generation) is not int or generation in indexed:
            raise ValueError("milestone generation is invalid or duplicate")
        row: dict[str, int] = {}
        for name in MILESTONE_FIELDS:
            item = raw.get(name)
            if type(item) is not int or item < 0:
                raise ValueError(f"milestone {name} must be a nonnegative integer")
            row[name] = item
        if row["population_alive"] <= 0:
            raise ValueError("milestone population must remain positive")
        if (
            row["endpoint_a_copies"] + row["endpoint_b_copies"]
            != 2 * row["population_alive"]
        ):
            raise ValueError("endpoint copies differ from diploid population")
        if row["hidden_genotype_count"] > row["population_alive"]:
            raise ValueError("hidden genotype count exceeds population")
        package_total = (
            row["intact_package_a_copies"]
            + row["intact_package_b_copies"]
            + row["recombinant_or_partial_copies"]
        )
        if package_total > 2 * row["population_alive"]:
            raise ValueError("package-copy counts exceed diploid population")
        if row["linkage_disequilibrium_abs_q10"] > 1024:
            raise ValueError("linkage disequilibrium is outside Q10")
        indexed[generation] = row
    if set(indexed) != set(MILESTONES):
        raise ValueError("milestone generations differ from G0/5/10/20/30")
    return indexed


def _empty_outcomes() -> dict[str, dict[str, Any]]:
    return {
        outcome_id: {
            "status": "not_evaluable",
            "supporting_blocks": 0,
        }
        for outcome_id in OUTCOME_IDS
    }


def _decision(
    decision: str,
    *,
    authority: Mapping[str, str] | None,
    seed_audit_sha256: str | None,
    reasons: Sequence[str],
    outcomes: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, bool],
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "authority": None if authority is None else dict(authority),
        "seed_audit_sha256": seed_audit_sha256,
        "decision": decision,
        "not_evaluable": decision == DECISION_D,
        "reasons": list(reasons),
        "gates": dict(gates),
        "outcomes": {
            name: dict(outcomes[name])
            for name in OUTCOME_IDS
        },
        "claims": {
            "paired_delta": False,
            "shared_selection_noise": False,
            "bounded_recomb72_support": decision == DECISION_A,
        },
    }
    return {**payload, "decision_sha256": content_sha256(payload)}


def _control_failures(
    rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> list[str]:
    failures: list[str] = []
    for pattern_id in FOUNDER_PATTERN_IDS:
        for state_id in CONTROL_STATE_IDS:
            for ecology_seed in ECOLOGY_SEEDS:
                row = rows[(pattern_id, state_id, ecology_seed)]
                for generation in MILESTONES:
                    metric = row["milestones"][generation]
                    copies = 2 * metric["population_alive"]
                    aa = state_id == "homozygous_aa_control"
                    expected = {
                        "endpoint_a_copies": copies if aa else 0,
                        "endpoint_b_copies": 0 if aa else copies,
                        "hidden_genotype_count": 0,
                        "intact_package_a_copies": copies if aa else 0,
                        "intact_package_b_copies": 0 if aa else copies,
                        "recombinant_or_partial_copies": 0,
                    }
                    if any(metric[name] != value for name, value in expected.items()):
                        failures.append(
                            f"{pattern_id}:{state_id}:{ecology_seed}:"
                            f"G{generation}:control_not_absorbing"
                        )
    return failures


def _scientific_outcomes(
    rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], bool, bool]:
    retained_patterns: list[str] = []
    persistent_patterns: list[str] = []
    for pattern_id in FOUNDER_PATTERN_IDS:
        retained = all(
            rows[(pattern_id, state_id, seed)]["milestones"][30][
                "endpoint_a_copies"
            ]
            > 0
            and rows[(pattern_id, state_id, seed)]["milestones"][30][
                "endpoint_b_copies"
            ]
            > 0
            for state_id in MIXED_STATE_IDS
            for seed in ECOLOGY_SEEDS
        )
        persistent = all(
            rows[(pattern_id, state_id, seed)]["milestones"][generation][
                "endpoint_a_copies"
            ]
            > 0
            and rows[(pattern_id, state_id, seed)]["milestones"][generation][
                "endpoint_b_copies"
            ]
            > 0
            for state_id in MIXED_STATE_IDS
            for seed in ECOLOGY_SEEDS
            for generation in POST_FOUNDER_MILESTONES
        )
        if retained:
            retained_patterns.append(pattern_id)
        if persistent:
            persistent_patterns.append(pattern_id)

    hidden_blocks = sum(
        rows[(HIDDEN_PATTERN_ID, state_id, seed)]["milestones"][30][
            "hidden_genotype_count"
        ]
        > 0
        for state_id in MIXED_STATE_IDS
        for seed in ECOLOGY_SEEDS
    )
    hidden_pass = hidden_blocks == len(MIXED_STATE_IDS) * len(ECOLOGY_SEEDS)

    intact_blocks = sum(
        rows[(CORRELATED_PATTERN_ID, state_id, seed)]["milestones"][30][
            "intact_package_a_copies"
        ]
        > 0
        and rows[(CORRELATED_PATTERN_ID, state_id, seed)]["milestones"][30][
            "intact_package_b_copies"
        ]
        > 0
        for state_id in MIXED_STATE_IDS
        for seed in ECOLOGY_SEEDS
    )
    intact_pass = intact_blocks == len(MIXED_STATE_IDS) * len(ECOLOGY_SEEDS)

    recombination_by_state: dict[str, int] = {}
    concentration_by_state: dict[str, int] = {}
    linkage_rows: list[dict[str, Any]] = []
    for state_id in MIXED_STATE_IDS:
        recombinant_support = 0
        concentration_support = 0
        for seed in ECOLOGY_SEEDS:
            row = rows[(CORRELATED_PATTERN_ID, state_id, seed)]
            milestone_rows = row["milestones"]
            g30 = milestone_rows[30]
            appeared = any(
                milestone_rows[generation]["recombinant_or_partial_copies"] > 0
                for generation in POST_FOUNDER_MILESTONES
            )
            coexists = (
                g30["recombinant_or_partial_copies"] > 0
                and (
                    g30["intact_package_a_copies"]
                    + g30["intact_package_b_copies"]
                )
                > 0
            )
            if appeared and coexists:
                recombinant_support += 1
            denominator = (
                g30["intact_package_a_copies"]
                + g30["intact_package_b_copies"]
                + g30["recombinant_or_partial_copies"]
            )
            concentrated = (
                denominator > 0
                and max(
                    g30["intact_package_a_copies"],
                    g30["intact_package_b_copies"],
                )
                * 1024
                >= PACKAGE_LOCKING_THRESHOLD_Q10 * denominator
            )
            if concentrated:
                concentration_support += 1
            g0_linkage = milestone_rows[0]["linkage_disequilibrium_abs_q10"]
            g30_linkage = g30["linkage_disequilibrium_abs_q10"]
            linkage_rows.append(
                {
                    "state_id": state_id,
                    "ecology_seed": seed,
                    "g0_q10": g0_linkage,
                    "g30_q10": g30_linkage,
                    "g0_minus_g30_q10": g0_linkage - g30_linkage,
                }
            )
        recombination_by_state[state_id] = recombinant_support
        concentration_by_state[state_id] = concentration_support

    allele_pass = len(retained_patterns) >= MINIMUM_RETENTION_PATTERNS
    persistence_pass = len(persistent_patterns) >= MINIMUM_RETENTION_PATTERNS
    recombination_pass = (
        max(recombination_by_state.values(), default=0)
        >= MINIMUM_RECOMBINANT_BLOCKS
    )
    concentration_risk = (
        max(concentration_by_state.values(), default=0)
        >= MINIMUM_RECOMBINANT_BLOCKS
    )
    retention_core = (
        allele_pass and persistence_pass and hidden_pass and intact_pass
    )
    outcomes = {
        "allele_retention": {
            "status": "supported" if allele_pass and persistence_pass else "failed",
            "supporting_blocks": len(retained_patterns) * len(ECOLOGY_SEEDS),
            "minimum_supporting_patterns": MINIMUM_RETENTION_PATTERNS,
            "g30_patterns": retained_patterns,
            "persistent_patterns": persistent_patterns,
        },
        "hidden_genotype_retention": {
            "status": "supported" if hidden_pass else "failed",
            "supporting_blocks": hidden_blocks,
            "minimum_supporting_blocks": 12,
            "scope": "two_hidden_mixed_states_x_six_ecology_blocks",
        },
        "intact_package_retention": {
            "status": "supported" if intact_pass else "failed",
            "supporting_blocks": intact_blocks,
            "minimum_supporting_blocks": 12,
            "scope": "two_correlated_mixed_states_x_six_ecology_blocks",
        },
        "recombinant_partial_appearance_and_g30_coexistence": {
            "status": "supported" if recombination_pass else "failed",
            "supporting_blocks": max(recombination_by_state.values(), default=0),
            "minimum_supporting_blocks": MINIMUM_RECOMBINANT_BLOCKS,
            "by_correlated_mixed_state": recombination_by_state,
        },
        "linkage_decay_descriptive": {
            "status": "descriptive",
            "supporting_blocks": len(linkage_rows),
            "minimum_supporting_blocks": 12,
            "decayed_blocks": sum(
                row["g0_minus_g30_q10"] > 0 for row in linkage_rows
            ),
            "unchanged_blocks": sum(
                row["g0_minus_g30_q10"] == 0 for row in linkage_rows
            ),
            "increased_blocks": sum(
                row["g0_minus_g30_q10"] < 0 for row in linkage_rows
            ),
            "rows": linkage_rows,
            "affects_retention_decision": False,
        },
        "control_integrity": {
            "status": "supported",
            "supporting_blocks": len(FOUNDER_PATTERN_IDS) * len(ECOLOGY_SEEDS),
            "minimum_supporting_blocks": 18,
            "milestones_per_control": len(MILESTONES),
        },
    }
    package_risk = {
        "present": concentration_risk,
        "threshold_q10": PACKAGE_LOCKING_THRESHOLD_Q10,
        "by_correlated_mixed_state": concentration_by_state,
    }
    outcomes["intact_package_retention"]["package_locking_risk"] = package_risk
    return outcomes, retention_core and recombination_pass, (
        retention_core and not recombination_pass and concentration_risk
    )


def evaluate_recomb72_v1(
    evidence: Sequence[Mapping[str, Any]],
    *,
    authority: Mapping[str, Any],
    seed_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce the exact 72-cell milestone matrix without paired deltas."""

    gates = {
        "authority": False,
        "seed_audit": False,
        "integrity": False,
        "controls": False,
    }
    try:
        canonical_authority = _validate_authority(authority)
    except ValueError as exc:
        return _decision(
            DECISION_D,
            authority=None,
            seed_audit_sha256=None,
            reasons=[str(exc)],
            outcomes=_empty_outcomes(),
            gates=gates,
        )
    gates["authority"] = True
    try:
        seed_audit_sha256 = _validate_seed_audit(
            seed_audit,
            authority=canonical_authority,
        )
    except ValueError as exc:
        return _decision(
            DECISION_D,
            authority=canonical_authority,
            seed_audit_sha256=None,
            reasons=[str(exc)],
            outcomes=_empty_outcomes(),
            gates=gates,
        )
    gates["seed_audit"] = True
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        return _decision(
            DECISION_D,
            authority=canonical_authority,
            seed_audit_sha256=seed_audit_sha256,
            reasons=["evidence_must_be_a_sequence"],
            outcomes=_empty_outcomes(),
            gates=gates,
        )

    expected = {
        (pattern_id, state_id, ecology_seed)
        for pattern_id in FOUNDER_PATTERN_IDS
        for state_id in FOUNDER_STATE_IDS
        for ecology_seed in ECOLOGY_SEEDS
    }
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    integrity: list[str] = []
    for raw in evidence:
        if not isinstance(raw, Mapping):
            integrity.append("cell_row_is_not_an_object")
            continue
        profile_id = raw.get("profile_id")
        ecology_seed = raw.get("ecology_seed")
        try:
            pattern_id, state_id = _profile_parts(profile_id)
        except (AttributeError, ValueError):
            integrity.append("cell_founder_profile_invalid")
            continue
        key = (pattern_id, state_id, ecology_seed)
        unsigned = dict(raw)
        supplied_hash = unsigned.pop("cell_sha256", None)
        if (
            set(raw) != CELL_FIELDS
            or raw.get("schema_version") != CELL_SCHEMA_VERSION
            or raw.get("preregistration_sha256") != PREREGISTRATION_SHA256
            or not _valid_sha256(supplied_hash)
            or supplied_hash != content_sha256(unsigned)
            or key in indexed
            or key not in expected
        ):
            integrity.append(f"{key!r}:schema_hash_duplicate_or_unexpected")
            continue
        if (
            raw.get("pattern_id") != pattern_id
            or raw.get("state_id") != state_id
            or raw.get("task_seed") != TASK_SEED
            or raw.get("seed_audit_sha256") != seed_audit_sha256
            or raw.get("founder_population_sha256")
            != resolve_recomb72_gv2_founder_profile_v1(profile_id)[
                "payload_sha256"
            ]
        ):
            integrity.append(f"{key!r}:profile_seed_or_audit_binding")
            continue
        try:
            row_authority = _validate_authority(raw.get("authority"))
            milestones = _validate_milestones(raw.get("milestones"))
        except ValueError as exc:
            integrity.append(f"{key!r}:{exc}")
            continue
        if row_authority != canonical_authority:
            integrity.append(f"{key!r}:mixed_authority")
        exact_flags = {
            "backend": "cuda_resident",
            "mutation_rate_q10": 0,
            "mutation_step_q10": 0,
            "ordinary_births_enabled": False,
            "no_fallback": True,
            "no_jit": True,
            "checkpoint_exact": True,
            "reporting_exact": True,
            "chronicle_exact": True,
            "cancellation_clean": True,
            "cross_cell_isolation_exact": True,
            "paired_delta_claimed": False,
            "shared_selection_noise_claimed": False,
        }
        for name, expected_value in exact_flags.items():
            if raw.get(name) != expected_value or type(raw.get(name)) is not type(
                expected_value
            ):
                integrity.append(f"{key!r}:{name}_contract_failure")
        indexed[key] = {
            "profile_id": profile_id,
            "pattern_id": pattern_id,
            "state_id": state_id,
            "ecology_seed": ecology_seed,
            "milestones": milestones,
        }
    missing = sorted(expected - set(indexed))
    if missing:
        integrity.append(f"missing_cells:{missing}")
    if len(indexed) != 72:
        integrity.append(f"exact_cell_count:{len(indexed)}")
    if integrity:
        return _decision(
            DECISION_D,
            authority=canonical_authority,
            seed_audit_sha256=seed_audit_sha256,
            reasons=integrity,
            outcomes=_empty_outcomes(),
            gates=gates,
        )
    gates["integrity"] = True

    control_failures = _control_failures(indexed)
    if control_failures:
        outcomes = _empty_outcomes()
        outcomes["control_integrity"] = {
            "status": "failed",
            "supporting_blocks": 0,
            "minimum_supporting_blocks": 18,
            "failures": control_failures,
        }
        return _decision(
            DECISION_D,
            authority=canonical_authority,
            seed_audit_sha256=seed_audit_sha256,
            reasons=control_failures,
            outcomes=outcomes,
            gates=gates,
        )
    gates["controls"] = True

    outcomes, promoted, package_locking = _scientific_outcomes(indexed)
    if promoted:
        decision = DECISION_A
        reasons: list[str] = []
    elif package_locking:
        decision = DECISION_B
        reasons = ["a_like_retention_with_package_locking_risk"]
    else:
        decision = DECISION_C
        reasons = [
            name
            for name in OUTCOME_IDS
            if outcomes[name]["status"] == "failed"
        ]
    return _decision(
        decision,
        authority=canonical_authority,
        seed_audit_sha256=seed_audit_sha256,
        reasons=reasons,
        outcomes=outcomes,
        gates=gates,
    )


__all__ = [
    "AUTHORITY_FIELDS",
    "CELL_SCHEMA_VERSION",
    "CORRELATED_PATTERN_ID",
    "DECISION_A",
    "DECISION_B",
    "DECISION_C",
    "DECISION_D",
    "DESIGN_LABEL",
    "ECOLOGY_SEEDS",
    "HIDDEN_PATTERN_ID",
    "MILESTONES",
    "MILESTONE_FIELDS",
    "OUTCOME_IDS",
    "PACKAGE_LOCKING_THRESHOLD_Q10",
    "PREREGISTRATION_SHA256",
    "PROVIDER_ID",
    "RUNTIME_PROFILE_ID",
    "SCHEMA_VERSION",
    "TASK_SEED",
    "evaluate_recomb72_v1",
    "preregistration_payload_v1",
]
