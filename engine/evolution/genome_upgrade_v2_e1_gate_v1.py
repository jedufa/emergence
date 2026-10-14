"""Preregistered Genome V2 P1.0 bounded ecological usefulness gate."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Final, Mapping, Sequence

from engine.authoritative_state.schema import content_sha256
from .genome_upgrade_v2_causal_gate_v1 import (
    DECISION_A as C1_DECISION_A,
    PREREGISTRATION_SHA256 as C1_PREREGISTRATION_SHA256,
    PROVIDER_ID,
    RUNTIME_PROFILE_ID,
    fixture_input_dimensions_v1,
)
from .genome_upgrade_v2 import GenomeUpgradeV2V1
from .genome_upgrade_v2_package_v1 import (
    validate_checkpoint_authority_binding_v1,
)


SCHEMA_VERSION: Final[str] = "genome_v2_p10_e1_ecological_gate_v1"
CELL_SCHEMA_VERSION: Final[str] = "genome_v2_p10_e1_cell_evidence_v1"
DECISION_A: Final[str] = "E1-A"
DECISION_B: Final[str] = "E1-B"
DECISION_C: Final[str] = "E1-C"
DECISION_D: Final[str] = "E1-D"
SEEDS: Final[tuple[int, ...]] = (9101, 9202, 9303, 9404, 9505, 9606)
TASK_SEED: Final[int] = 27101
REPRESENTATIVE_SEED: Final[int] = 9101
ARMS: Final[tuple[str, str]] = ("control", "treatment")
CAMPAIGNS: Final[tuple[str, ...]] = (
    "genome-v2-e1-neutral-no-influence-g32",
    "genome-v2-e1-patch-pleiotropic-g32",
    "genome-v2-e1-wdl-hazard-regulatory-g32",
)
NO_INFLUENCE_CAMPAIGN: Final[str] = CAMPAIGNS[0]
TREATMENT_CAMPAIGNS: Final[tuple[str, str]] = CAMPAIGNS[1:]
PRIMARY_METRICS: Final[tuple[str, ...]] = (
    "population.terminal_alive",
    "population.successful_births_cumulative",
)
OBSERVATION_COUNTS: Final[tuple[str, ...]] = (
    "genome_v2.expression_observation_count",
    "genome_v2.context_observation_count",
    "genome_v2.cost_observation_count",
    "genome_v2.cap_observation_count",
)


def _founder_payload_v1(profile_id: str) -> dict[str, Any]:
    if profile_id == "neutral":
        genome = GenomeUpgradeV2V1.neutral()
        genotype = tuple(
            value
            for homolog in genome.alleles
            for product in homolog
            for value in product
        ) + tuple(
            value for homolog in genome.modifiers for value in homolog
        )
        source = {"kind": "canonical_neutral", "arm": "neutral"}
    else:
        genotype, _context = fixture_input_dimensions_v1(
            profile_id, "treatment"
        )
        source = {
            "kind": "genome_upgrade_v2_c1_fixture_v2",
            "family_id": profile_id,
            "arm": "treatment",
        }
    return {
        "schema_version": "genome_upgrade_v2_founder_profile_v1",
        "profile_id": profile_id,
        "source": source,
        "genotype_q10": list(genotype),
    }


FOUNDER_PAYLOADS: Final[dict[str, dict[str, Any]]] = {
    profile_id: _founder_payload_v1(profile_id)
    for profile_id in (
        "neutral",
        "pleiotropic_tradeoffs",
        "regulatory_activation_inhibition",
    )
}
CAMPAIGN_FOUNDER_PAYLOADS: Final[dict[str, dict[str, str]]] = {
    CAMPAIGNS[0]: {
        "control": "neutral",
        "treatment": "neutral",
    },
    CAMPAIGNS[1]: {
        "control": "neutral",
        "treatment": "pleiotropic_tradeoffs",
    },
    CAMPAIGNS[2]: {
        "control": "neutral",
        "treatment": "regulatory_activation_inhibition",
    },
}


def founder_payload_sha256s_v1() -> dict[str, str]:
    return {
        payload_id: content_sha256(payload)
        for payload_id, payload in FOUNDER_PAYLOADS.items()
    }


def cell_input_schema_v1() -> dict[str, Any]:
    """Describe the exact stable telemetry projection expected per E1 cell."""

    return {
        "schema_version": CELL_SCHEMA_VERSION,
        "source_reporting_schema_version": "genome_v2_reporting_v2",
        "source_compact_summary_schema_version": (
            "genome_upgrade_v2_compact_generation_summary_v2"
        ),
        "primary_metric_sources": {
            "population.terminal_alive": "native_metric_projection@G32",
            "population.successful_births_cumulative": (
                "native_metric_projection@G1:G32_sum"
            ),
        },
        "observation_count_sources": {
            "genome_v2.expression_observation_count": (
                "compact_generation_summary.products_q10"
            ),
            "genome_v2.context_observation_count": (
                "compact_generation_summary.contexts"
            ),
            "genome_v2.cost_observation_count": (
                "compact_generation_summary.expression_cost_q10"
            ),
            "genome_v2.cap_observation_count": (
                "compact_generation_summary.consequences.cap_binding"
            ),
        },
        "metric_value_contract": "nonnegative_integer_complete_exact_fields",
        "trajectory_sha256": (
            "bounded_generation_metric_sequence_plus_terminal_mechanics_"
            "checkpoint_sha256"
        ),
        "checkpoint_sha256": "terminal_G32_checkpoint_sha256",
        "representative_parity": (
            "fresh_repeat_exact_and_G16_disk_continuation_exact"
        ),
    }


def preregistration_payload_v1() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "programme": "GENOME-UPGRADE-V2/P1.0",
        "gate": "E1",
        "provider_id": PROVIDER_ID,
        "runtime_profile_id": RUNTIME_PROFILE_ID,
        "admission": {
            "required_c1_decision": C1_DECISION_A,
            "c1_preregistration_sha256": C1_PREREGISTRATION_SHA256,
        },
        "world": {
            "side": 16,
            "initial_population": 64,
            "capacity": 256,
        },
        "extent": {
            "generations": 32,
            "ticks_per_generation": 80,
            "paired_seeds": list(SEEDS),
            "task_seed": TASK_SEED,
            "logical_cells": 36,
            "mutation_rate_q10": 0,
        },
        "campaigns": [
            {
                "campaign_id": campaign_id,
                "arms": dict(CAMPAIGN_FOUNDER_PAYLOADS[campaign_id]),
            }
            for campaign_id in CAMPAIGNS
        ],
        "founder_payloads": [
            {
                "payload_id": payload_id,
                "payload": FOUNDER_PAYLOADS[payload_id],
                "payload_sha256": founder_payload_sha256s_v1()[payload_id],
            }
            for payload_id in sorted(FOUNDER_PAYLOADS)
        ],
        "representative_parity": {
            "paired_seed": REPRESENTATIVE_SEED,
            "fresh_repeat": "exact",
            "continuation_split_generation": 16,
            "required_cells": 6,
        },
        "required_primary_metrics": list(PRIMARY_METRICS),
        "required_observation_counts": list(OBSERVATION_COUNTS),
        "cell_input_schema": cell_input_schema_v1(),
        "decision_contract": {
            DECISION_D: "identity_integrity_or_parity_failure",
            DECISION_C: "incomplete_matrix_metrics_or_fewer_than_six_pairs",
            DECISION_A: (
                "neutral_no_influence_exact_and_each_treatment_campaign_has_"
                "at_least_one_primary_outcome_favorable_in_four_of_six_with_"
                "strict_positive_median_no_negative_primary_median_and_all_"
                "required_observation_surfaces"
            ),
            DECISION_B: "complete_valid_evidence_not_meeting_e1_a",
        },
        "claims_boundary": (
            "bounded ecological usefulness only; no intelligence, general "
            "cognition, open-ended evolution, consciousness, speciation, or "
            "universal biological realism claim"
        ),
    }


PREREGISTRATION_SHA256: Final[str] = content_sha256(
    preregistration_payload_v1()
)


def _valid_hash(value: Any) -> bool:
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


@dataclass(frozen=True, slots=True)
class E1AuthorityBindingV1:
    source_commit: str
    package_id_sha256: str
    package_receipt_sha256: str
    artifact_sha256: str
    execution_route_sha256s: Mapping[str, str]
    checkpoint_authority_bindings: Mapping[str, Mapping[str, Any]]
    c1_preregistration_sha256: str
    c1_decision_sha256: str
    c1_decision: str
    campaign_plan_sha256s: Mapping[str, str]
    founder_payload_sha256s: Mapping[str, str]
    checkpoint_sha256s: Mapping[str, str]
    parity_receipt_sha256s: Mapping[str, str]
    reporting_parity_receipt_sha256s: Mapping[str, str]
    e1_preregistration_sha256: str = PREREGISTRATION_SHA256
    provider_id: str = PROVIDER_ID
    runtime_profile_id: str = RUNTIME_PROFILE_ID

    def payload(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "runtime_profile_id": self.runtime_profile_id,
            "source_commit": self.source_commit,
            "package_id_sha256": self.package_id_sha256,
            "package_receipt_sha256": self.package_receipt_sha256,
            "artifact_sha256": self.artifact_sha256,
            "execution_route_sha256s": dict(
                self.execution_route_sha256s
            ),
            "checkpoint_authority_bindings": {
                campaign: dict(binding)
                for campaign, binding in (
                    self.checkpoint_authority_bindings.items()
                )
            },
            "c1_preregistration_sha256": self.c1_preregistration_sha256,
            "c1_decision_sha256": self.c1_decision_sha256,
            "c1_decision": self.c1_decision,
            "campaign_plan_sha256s": dict(self.campaign_plan_sha256s),
            "founder_payload_sha256s": dict(self.founder_payload_sha256s),
            "checkpoint_sha256s": dict(self.checkpoint_sha256s),
            "parity_receipt_sha256s": dict(
                self.parity_receipt_sha256s
            ),
            "reporting_parity_receipt_sha256s": dict(
                self.reporting_parity_receipt_sha256s
            ),
            "e1_preregistration_sha256": self.e1_preregistration_sha256,
        }

    def authority_sha256(self) -> str:
        return content_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class E1CellEvidenceV1:
    campaign_id: str
    arm: str
    ecology_seed: int
    task_seed: int
    authority_sha256: str
    campaign_plan_sha256: str
    execution_route_sha256: str
    founder_payload_id: str
    founder_payload_sha256: str
    checkpoint_sha256: str
    execution_receipt_sha256: str
    trajectory_sha256: str
    route_parity_receipt_sha256: str
    fresh_repeat_exact: bool
    continuation_g16_exact: bool
    no_fallback: bool
    no_jit: bool
    metrics: Mapping[str, int]

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.campaign_id, self.arm, self.ecology_seed)

    def payload(self) -> dict[str, Any]:
        value = {
            "schema_version": CELL_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "arm": self.arm,
            "ecology_seed": self.ecology_seed,
            "task_seed": self.task_seed,
            "authority_sha256": self.authority_sha256,
            "campaign_plan_sha256": self.campaign_plan_sha256,
            "execution_route_sha256": self.execution_route_sha256,
            "founder_payload_id": self.founder_payload_id,
            "founder_payload_sha256": self.founder_payload_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "execution_receipt_sha256": self.execution_receipt_sha256,
            "trajectory_sha256": self.trajectory_sha256,
            "route_parity_receipt_sha256": (
                self.route_parity_receipt_sha256
            ),
            "fresh_repeat_exact": self.fresh_repeat_exact,
            "continuation_g16_exact": self.continuation_g16_exact,
            "no_fallback": self.no_fallback,
            "no_jit": self.no_jit,
            "metrics": dict(self.metrics),
        }
        return {**value, "cell_sha256": content_sha256(value)}


def _expected_keys() -> set[tuple[str, str, int]]:
    return {
        (campaign, arm, seed)
        for campaign in CAMPAIGNS
        for arm in ARMS
        for seed in SEEDS
    }


def _checkpoint_key(campaign: str, arm: str, seed: int) -> str:
    return f"{campaign}:{arm}:{seed}"


def _authority_defects(authority: Any) -> list[str]:
    if not isinstance(authority, E1AuthorityBindingV1):
        return ["invalid_authority_type"]
    defects: list[str] = []
    if (
        set(authority.execution_route_sha256s) != set(CAMPAIGNS)
        or set(authority.checkpoint_authority_bindings) != set(CAMPAIGNS)
    ):
        return ["campaign_route_authority_set_mismatch"]
    route_bindings: dict[str, dict[str, str]] = {}
    for campaign in CAMPAIGNS:
        try:
            route_bindings[campaign] = (
                validate_checkpoint_authority_binding_v1(
                    authority.checkpoint_authority_bindings[campaign]
                )
            )
        except ValueError:
            return [
                f"{campaign}:invalid_17_field_checkpoint_authority_binding"
            ]
    if authority.provider_id != PROVIDER_ID:
        defects.append("provider_identity_mismatch")
    if authority.runtime_profile_id != RUNTIME_PROFILE_ID:
        defects.append("runtime_profile_identity_mismatch")
    if not _valid_commit(authority.source_commit):
        defects.append("invalid_source_commit")
    for name in (
        "package_id_sha256",
        "package_receipt_sha256",
        "artifact_sha256",
        "c1_preregistration_sha256",
        "c1_decision_sha256",
        "e1_preregistration_sha256",
    ):
        if not _valid_hash(getattr(authority, name)):
            defects.append(f"invalid_{name}")
    common_joins = {
        "provider_id": authority.provider_id,
        "runtime_profile_id": authority.runtime_profile_id,
        "source_commit": authority.source_commit,
        "package_id_sha256": authority.package_id_sha256,
        "package_receipt_sha256": authority.package_receipt_sha256,
        "artifact_sha256": authority.artifact_sha256,
    }
    if any(
        not _valid_hash(authority.execution_route_sha256s[campaign])
        or any(
            route_bindings[campaign][name] != expected
            for name, expected in common_joins.items()
        )
        or route_bindings[campaign]["execution_route_sha256"]
        != authority.execution_route_sha256s[campaign]
        for campaign in CAMPAIGNS
    ):
        defects.append("17_field_checkpoint_authority_join_mismatch")
    if (
        authority.c1_preregistration_sha256
        != C1_PREREGISTRATION_SHA256
        or authority.c1_decision != C1_DECISION_A
        or authority.e1_preregistration_sha256
        != PREREGISTRATION_SHA256
    ):
        defects.append("preregistration_or_c1_admission_mismatch")
    if (
        set(authority.campaign_plan_sha256s) != set(CAMPAIGNS)
        or any(
            not _valid_hash(value)
            for value in authority.campaign_plan_sha256s.values()
        )
    ):
        defects.append("campaign_plan_identity_set_mismatch")
    if authority.founder_payload_sha256s != founder_payload_sha256s_v1():
        defects.append("founder_payload_identity_set_mismatch")
    expected_checkpoint_keys = {
        _checkpoint_key(*key) for key in _expected_keys()
    }
    if (
        set(authority.checkpoint_sha256s) != expected_checkpoint_keys
        or any(
            not _valid_hash(value)
            for value in authority.checkpoint_sha256s.values()
        )
    ):
        defects.append("checkpoint_identity_set_mismatch")
    expected_parity_keys = {
        _checkpoint_key(campaign, arm, REPRESENTATIVE_SEED)
        for campaign in CAMPAIGNS
        for arm in ARMS
    }
    if (
        set(authority.parity_receipt_sha256s) != expected_parity_keys
        or any(
            not _valid_hash(value)
            for value in authority.parity_receipt_sha256s.values()
        )
    ):
        defects.append("representative_parity_identity_set_mismatch")
    if (
        set(authority.reporting_parity_receipt_sha256s) != set(CAMPAIGNS)
        or any(
            not _valid_hash(value)
            for value in authority.reporting_parity_receipt_sha256s.values()
        )
    ):
        defects.append("reporting_parity_identity_set_mismatch")
    return defects


def evaluate_e1_v1(
    cells: Sequence[E1CellEvidenceV1],
    *,
    authority: E1AuthorityBindingV1,
) -> dict[str, Any]:
    """Reduce one exact post-run E1 matrix using the frozen A/B/C/D rules."""

    authority_defects = _authority_defects(authority)
    if authority_defects:
        return _decision(DECISION_D, [], authority_defects)
    indexed: dict[tuple[str, str, int], E1CellEvidenceV1] = {}
    integrity: list[str] = []
    for cell in cells:
        if not isinstance(cell, E1CellEvidenceV1):
            integrity.append("invalid_cell_evidence_type")
            continue
        if cell.key in indexed:
            integrity.append(f"duplicate_cell:{cell.key!r}")
        indexed[cell.key] = cell
    missing = sorted(_expected_keys() - set(indexed))
    unexpected = sorted(set(indexed) - _expected_keys())
    if unexpected:
        integrity.append(f"unexpected_cells:{unexpected!r}")
    if integrity:
        return _decision(DECISION_D, [], integrity)
    if missing:
        return _decision(
            DECISION_C,
            [],
            [f"incomplete_36_cell_matrix:{missing!r}"],
        )

    incomplete: list[str] = []
    for key, cell in sorted(indexed.items()):
        expected_payload_id = CAMPAIGN_FOUNDER_PAYLOADS[cell.campaign_id][
            cell.arm
        ]
        checkpoint_key = _checkpoint_key(*key)
        for hash_name in (
            "authority_sha256",
            "campaign_plan_sha256",
            "execution_route_sha256",
            "founder_payload_sha256",
            "checkpoint_sha256",
            "execution_receipt_sha256",
            "trajectory_sha256",
            "route_parity_receipt_sha256",
        ):
            if not _valid_hash(getattr(cell, hash_name)):
                integrity.append(f"{key!r}:invalid_{hash_name}")
        if (
            cell.task_seed != TASK_SEED
            or cell.authority_sha256 != authority.authority_sha256()
            or cell.campaign_plan_sha256
            != authority.campaign_plan_sha256s[cell.campaign_id]
            or cell.execution_route_sha256
            != authority.execution_route_sha256s[cell.campaign_id]
            or cell.founder_payload_id != expected_payload_id
            or cell.founder_payload_sha256
            != authority.founder_payload_sha256s[expected_payload_id]
            or cell.checkpoint_sha256
            != authority.checkpoint_sha256s[checkpoint_key]
        ):
            integrity.append(f"{key!r}:identity_join_mismatch")
        if not cell.no_fallback or not cell.no_jit:
            integrity.append(f"{key!r}:fallback_or_jit_present")
        if cell.ecology_seed == REPRESENTATIVE_SEED and (
            not cell.fresh_repeat_exact or not cell.continuation_g16_exact
            or cell.route_parity_receipt_sha256
            != authority.parity_receipt_sha256s[checkpoint_key]
        ):
            integrity.append(f"{key!r}:representative_parity_failure")
        if (
            not isinstance(cell.metrics, Mapping)
            or set(cell.metrics) != set((*PRIMARY_METRICS, *OBSERVATION_COUNTS))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in cell.metrics.values()
            )
        ):
            incomplete.append(f"{key!r}:incomplete_or_invalid_metrics")
    if integrity:
        return _decision(DECISION_D, [], integrity)
    if incomplete:
        return _decision(DECISION_C, [], incomplete)

    deltas: list[dict[str, Any]] = []
    for campaign in CAMPAIGNS:
        for seed in SEEDS:
            control = indexed[(campaign, "control", seed)]
            treatment = indexed[(campaign, "treatment", seed)]
            row: dict[str, Any] = {
                "campaign_id": campaign,
                "ecology_seed": seed,
                "control_cell_sha256": control.payload()["cell_sha256"],
                "treatment_cell_sha256": treatment.payload()["cell_sha256"],
                "trajectory_exact": (
                    control.trajectory_sha256 == treatment.trajectory_sha256
                ),
            }
            for metric in PRIMARY_METRICS:
                row[f"control_{metric}"] = control.metrics[metric]
                row[f"treatment_{metric}"] = treatment.metrics[metric]
                row[f"delta_{metric}"] = (
                    treatment.metrics[metric] - control.metrics[metric]
                )
            deltas.append(row)

    no_influence_rows = [
        row
        for row in deltas
        if row["campaign_id"] == NO_INFLUENCE_CAMPAIGN
    ]
    no_influence_exact = all(
        row["trajectory_exact"]
        and all(row[f"delta_{metric}"] == 0 for metric in PRIMARY_METRICS)
        for row in no_influence_rows
    )
    summaries: list[dict[str, Any]] = []
    treatment_supported = True
    no_adverse_median = True
    for campaign in TREATMENT_CAMPAIGNS:
        rows = [row for row in deltas if row["campaign_id"] == campaign]
        metric_rows = []
        for metric in PRIMARY_METRICS:
            values = [int(row[f"delta_{metric}"]) for row in rows]
            metric_median = median(values)
            favorable_count = sum(value > 0 for value in values)
            metric_rows.append(
                {
                    "metric": metric,
                    "paired_deltas": values,
                    "favorable_seed_count": favorable_count,
                    "paired_median": metric_median,
                    "supports": favorable_count >= 4 and metric_median > 0,
                }
            )
            no_adverse_median = no_adverse_median and metric_median >= 0
        supports = any(row["supports"] for row in metric_rows)
        treatment_supported = treatment_supported and supports
        summaries.append(
            {
                "campaign_id": campaign,
                "metrics": metric_rows,
                "supports": supports,
            }
        )
    observations_complete = all(
        cell.metrics[name] > 0
        for cell in indexed.values()
        for name in OBSERVATION_COUNTS
    )
    a_supported = (
        no_influence_exact
        and treatment_supported
        and no_adverse_median
        and observations_complete
    )
    reasons = []
    if not no_influence_exact:
        reasons.append("neutral_no_influence_not_exact")
    if not treatment_supported:
        reasons.append("treatment_campaign_direction_rule_not_met")
    if not no_adverse_median:
        reasons.append("strictly_adverse_primary_median")
    if not observations_complete:
        reasons.append("required_observation_surface_missing")
    decision = _decision(
        DECISION_A if a_supported else DECISION_B,
        deltas,
        reasons,
    )
    decision["no_influence_exact"] = no_influence_exact
    decision["observations_complete"] = observations_complete
    decision["treatment_summaries"] = summaries
    unsigned = {
        key: value for key, value in decision.items()
        if key != "decision_sha256"
    }
    decision["decision_sha256"] = content_sha256(unsigned)
    return decision


def _decision(
    decision: str,
    deltas: Sequence[Mapping[str, Any]],
    reasons: Sequence[str],
) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "decision": decision,
        "cell_count": 36 if len(deltas) == 18 else 0,
        "pair_count": len(deltas),
        "paired_deltas": [dict(row) for row in deltas],
        "reasons": sorted(set(reasons)),
        "claims_boundary": preregistration_payload_v1()["claims_boundary"],
    }
    return {**value, "decision_sha256": content_sha256(value)}


__all__ = [
    "ARMS",
    "CAMPAIGNS",
    "CAMPAIGN_FOUNDER_PAYLOADS",
    "DECISION_A",
    "DECISION_B",
    "DECISION_C",
    "DECISION_D",
    "E1AuthorityBindingV1",
    "E1CellEvidenceV1",
    "FOUNDER_PAYLOADS",
    "OBSERVATION_COUNTS",
    "PREREGISTRATION_SHA256",
    "PRIMARY_METRICS",
    "REPRESENTATIVE_SEED",
    "SEEDS",
    "TASK_SEED",
    "evaluate_e1_v1",
    "cell_input_schema_v1",
    "founder_payload_sha256s_v1",
    "preregistration_payload_v1",
]
