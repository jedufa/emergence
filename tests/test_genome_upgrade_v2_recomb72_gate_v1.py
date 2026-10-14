from __future__ import annotations

from engine.authoritative_state.schema import content_sha256
from engine.evolution.genome_upgrade_v2_recomb72_founders_v1 import (
    FOUNDER_PATTERN_IDS,
    FOUNDER_PROFILE_IDS,
    FOUNDER_STATE_IDS,
    resolve_recomb72_gv2_founder_profile_v1,
)
from engine.evolution.genome_upgrade_v2_recomb72_gate_v1 import (
    CELL_SCHEMA_VERSION,
    CORRELATED_PATTERN_ID,
    DECISION_A,
    DECISION_B,
    DECISION_C,
    DECISION_D,
    DESIGN_LABEL,
    ECOLOGY_SEEDS,
    HIDDEN_PATTERN_ID,
    MILESTONES,
    OUTCOME_IDS,
    PREREGISTRATION_SHA256,
    evaluate_recomb72_v1,
    preregistration_payload_v1,
)


AUTHORITY = {
    "provider_id": "eiel-production-provider-v1",
    "runtime_profile_id": "eiel-production-runtime-v1",
    "source_commit": "a" * 40,
    "package_id_sha256": "b" * 64,
    "package_receipt_sha256": "c" * 64,
    "artifact_sha256": "d" * 64,
    "execution_route_sha256": "e" * 64,
}


def _seed_audit() -> dict:
    value = {
        "schema_version": "genome_v2_recomb72_seed_authority_v1",
        "design_label": DESIGN_LABEL,
        "fully_paired_claims_allowed": False,
        "authority": {
            "package_id_sha256": AUTHORITY["package_id_sha256"],
            "execution_route_sha256": AUTHORITY["execution_route_sha256"],
            "world_identity": "patch",
            "configuration_sha256": "f" * 64,
        },
        "matrix_contract": {
            "cell_count": 72,
            "founder_profile_ids": list(FOUNDER_PROFILE_IDS),
            "ecology_seeds": list(ECOLOGY_SEEDS),
            "task_seed": 0,
            "physical_batch_shape": "b72",
        },
        "checks": {
            "exact_72_cell_matrix": True,
            "ecology_blocked_physical_order": True,
            "profile_specific_selection_tapes": True,
        },
    }
    return {**value, "audit_sha256": content_sha256(value)}


def _profile_parts(profile_id: str) -> tuple[str, str]:
    _prefix, pattern_id, state_id = profile_id.split(".", maxsplit=2)
    return pattern_id, state_id


def _milestone(
    *,
    pattern_id: str,
    state_id: str,
    ecology_seed: int,
    generation: int,
    recombinant: bool,
    concentrated: bool,
) -> dict[str, int]:
    population = 64
    copies = 2 * population
    if state_id == "homozygous_aa_control":
        a_copies, b_copies = copies, 0
    elif state_id == "homozygous_bb_control":
        a_copies, b_copies = 0, copies
    else:
        a_copies, b_copies = 64, 64
    control = state_id in {
        "homozygous_aa_control",
        "homozygous_bb_control",
    }
    if control:
        intact_a, intact_b, recombinant_copies = a_copies, b_copies, 0
    elif pattern_id == CORRELATED_PATTERN_ID:
        if concentrated:
            intact_a, intact_b, recombinant_copies = 120, 1, 0
        else:
            intact_a, intact_b = 48, 48
            recombinant_copies = (
                16
                if recombinant
                and ecology_seed in ECOLOGY_SEEDS[:4]
                and generation == 30
                else 0
            )
    else:
        intact_a, intact_b, recombinant_copies = 32, 32, 0
    return {
        "generation": generation,
        "population_alive": population,
        "endpoint_a_copies": a_copies,
        "endpoint_b_copies": b_copies,
        "hidden_genotype_count": (
            8
            if pattern_id == HIDDEN_PATTERN_ID and not control
            else 0
        ),
        "intact_package_a_copies": intact_a,
        "intact_package_b_copies": intact_b,
        "recombinant_or_partial_copies": recombinant_copies,
        "linkage_disequilibrium_abs_q10": max(0, 800 - 10 * generation),
    }


def _evidence(
    *,
    recombinant: bool = True,
    concentrated: bool = False,
) -> list[dict]:
    audit = _seed_audit()
    rows: list[dict] = []
    for profile_id in FOUNDER_PROFILE_IDS:
        pattern_id, state_id = _profile_parts(profile_id)
        for ecology_seed in ECOLOGY_SEEDS:
            value = {
                "schema_version": CELL_SCHEMA_VERSION,
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "authority": dict(AUTHORITY),
                "seed_audit_sha256": audit["audit_sha256"],
                "profile_id": profile_id,
                "founder_population_sha256": (
                    resolve_recomb72_gv2_founder_profile_v1(profile_id)[
                        "payload_sha256"
                    ]
                ),
                "pattern_id": pattern_id,
                "state_id": state_id,
                "ecology_seed": ecology_seed,
                "task_seed": 0,
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
                "milestones": [
                    _milestone(
                        pattern_id=pattern_id,
                        state_id=state_id,
                        ecology_seed=ecology_seed,
                        generation=generation,
                        recombinant=recombinant,
                        concentrated=concentrated,
                    )
                    for generation in MILESTONES
                ],
            }
            rows.append({**value, "cell_sha256": content_sha256(value)})
    return rows


def _rehash(row: dict) -> None:
    unsigned = dict(row)
    unsigned.pop("cell_sha256", None)
    row["cell_sha256"] = content_sha256(unsigned)


def _evaluate(rows: list[dict]) -> dict:
    return evaluate_recomb72_v1(
        rows,
        authority=AUTHORITY,
        seed_audit=_seed_audit(),
    )


def test_preregistration_freezes_exact_independent_tape_design() -> None:
    payload = preregistration_payload_v1()

    assert payload["design_label"] == (
        "Ecology-blocked deterministic design with independent "
        "profile-specific selection tapes."
    )
    assert payload["campaign_design"]["run_label"] == "b72g30t80"
    assert payload["campaign_design"]["milestone_generations"] == [0, 5, 10, 20, 30]
    assert payload["campaign_design"]["mutation_rate_q10"] == 0
    assert payload["campaign_design"]["ordinary_births_enabled"] is False
    assert payload["campaign_design"]["backend"] == "cuda_resident"
    assert payload["seed_and_claim_contract"]["forbidden_claims"] == [
        "paired_delta",
        "shared_selection_noise",
    ]
    assert payload["seed_and_claim_contract"][
        "fully_paired_trajectory_claims_allowed"
    ] is False
    assert len(payload["campaign_design"]["founder_profile_ids"]) == 12
    assert len(FOUNDER_PATTERN_IDS) == 3
    assert len(FOUNDER_STATE_IDS) == 4
    assert content_sha256(payload) == PREREGISTRATION_SHA256


def test_exact_a_reports_six_separate_outcomes() -> None:
    result = _evaluate(_evidence())

    assert result["decision"] == DECISION_A
    assert result["not_evaluable"] is False
    assert tuple(result["outcomes"]) == OUTCOME_IDS
    assert all(result["gates"].values())
    assert result["outcomes"]["allele_retention"]["g30_patterns"] == list(
        FOUNDER_PATTERN_IDS
    )
    assert result["outcomes"]["hidden_genotype_retention"]["supporting_blocks"] == 12
    assert result["outcomes"]["intact_package_retention"]["supporting_blocks"] == 12
    assert result["outcomes"][
        "recombinant_partial_appearance_and_g30_coexistence"
    ]["supporting_blocks"] == 4
    assert result["outcomes"]["linkage_decay_descriptive"][
        "affects_retention_decision"
    ] is False
    unsigned = dict(result)
    supplied = unsigned.pop("decision_sha256")
    assert content_sha256(unsigned) == supplied


def test_b_is_frozen_a_like_retention_with_package_locking_risk() -> None:
    result = _evaluate(_evidence(recombinant=False, concentrated=True))

    assert result["decision"] == DECISION_B
    assert result["reasons"] == ["a_like_retention_with_package_locking_risk"]
    assert result["outcomes"]["intact_package_retention"][
        "package_locking_risk"
    ]["present"] is True
    assert result["outcomes"][
        "recombinant_partial_appearance_and_g30_coexistence"
    ]["status"] == "failed"


def test_c_is_complete_interpretable_retention_failure() -> None:
    rows = _evidence(recombinant=False)
    for row in rows:
        if row["state_id"] in {
            "all_heterozygous_ab",
            "balanced_homozygous_aa_bb",
        }:
            for milestone in row["milestones"]:
                if milestone["generation"] > 0:
                    milestone["endpoint_a_copies"] = 0
                    milestone["endpoint_b_copies"] = 128
            _rehash(row)

    result = _evaluate(rows)

    assert result["decision"] == DECISION_C
    assert result["not_evaluable"] is False
    assert result["outcomes"]["allele_retention"]["status"] == "failed"


def test_missing_cell_is_d_and_not_evaluable() -> None:
    result = _evaluate(_evidence()[:-1])

    assert result["decision"] == DECISION_D
    assert result["not_evaluable"] is True
    assert any(reason.startswith("missing_cells:") for reason in result["reasons"])
    assert all(
        outcome["status"] == "not_evaluable"
        for outcome in result["outcomes"].values()
    )


def test_control_failure_is_d_before_scientific_classification() -> None:
    rows = _evidence()
    control = next(
        row for row in rows if row["state_id"] == "homozygous_aa_control"
    )
    control["milestones"][2]["endpoint_a_copies"] = 127
    control["milestones"][2]["endpoint_b_copies"] = 1
    _rehash(control)

    result = _evaluate(rows)

    assert result["decision"] == DECISION_D
    assert result["gates"]["integrity"] is True
    assert result["gates"]["controls"] is False
    assert result["outcomes"]["control_integrity"]["status"] == "failed"


def test_paired_delta_and_shared_selection_noise_claims_are_rejected() -> None:
    for claim in ("paired_delta_claimed", "shared_selection_noise_claimed"):
        rows = _evidence()
        rows[0][claim] = True
        _rehash(rows[0])

        result = _evaluate(rows)

        assert result["decision"] == DECISION_D
        assert result["not_evaluable"] is True
        assert any(claim in reason for reason in result["reasons"])
        assert result["claims"]["paired_delta"] is False
        assert result["claims"]["shared_selection_noise"] is False


def test_integrity_gates_reject_execution_or_seed_audit_drift() -> None:
    rows = _evidence()
    rows[0]["mutation_rate_q10"] = 1
    _rehash(rows[0])
    assert _evaluate(rows)["decision"] == DECISION_D

    audit = _seed_audit()
    audit["design_label"] = DESIGN_LABEL.removesuffix(".")
    unsigned = dict(audit)
    unsigned.pop("audit_sha256")
    audit["audit_sha256"] = content_sha256(unsigned)
    result = evaluate_recomb72_v1(
        _evidence(),
        authority=AUTHORITY,
        seed_audit=audit,
    )
    assert result["decision"] == DECISION_D
    assert result["gates"]["seed_audit"] is False
