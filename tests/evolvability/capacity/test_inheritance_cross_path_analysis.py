from __future__ import annotations

import copy

import pytest

from engine.evolution.inheritance_cross_path_analysis import (
    InheritanceCrossPathAnalysisError,
    InheritanceCrossPathContract,
    analyze_inheritance_cross_path,
)


def _contract(
    *,
    arms: tuple[str, ...] = ("base", "repair"),
    contrasts: tuple[tuple[str, str, str], ...] = (
        ("repair_vs_base", "base", "repair"),
    ),
    paths: tuple[str, ...] = ("refill", "live"),
    patterns: tuple[str, ...] = ("scalar", "package"),
    pairs: tuple[int | str, ...] = (11, 22, 33),
    generations: tuple[int, ...] = tuple(range(9)),
    milestones: tuple[int, ...] = (2, 4, 8),
) -> dict:
    labels = {
        "PASS": "all paths",
        "LOCK": "locking",
        "REFILL": "refill only",
        "LIVE": "live only",
        "FAIL": "failed",
    }
    rules = [
        {"when": "integrity_failure", "decision": "FAIL"},
        {"when": "all_paths_confirmed_and_locking", "decision": "LOCK"},
        {"when": "all_paths_confirmed", "decision": "PASS"},
    ]
    if "refill" in paths:
        rules.append(
            {"when": "confirmed_paths_exact", "paths": ["refill"], "decision": "REFILL"}
        )
    if "live" in paths:
        rules.append(
            {"when": "confirmed_paths_exact", "paths": ["live"], "decision": "LIVE"}
        )
    rules.append({"when": "otherwise", "decision": "FAIL"})
    return {
        "schema_version": "test_contract_v1",
        "identity": {
            "arms": list(arms),
            "paths": list(paths),
            "patterns": list(patterns),
            "paired_values": list(pairs),
            "paired_field": "seed",
        },
        "contrasts": [
            {
                "name": name,
                "baseline_arm": baseline,
                "treatment_arm": treatment,
            }
            for name, baseline, treatment in contrasts
        ],
        "generations": list(generations),
        "milestones": list(milestones),
        "loci": ["x", "y"],
        "pattern_contracts": {
            pattern: {
                "primary_locus": "x" if pattern == patterns[0] else "y",
                "benefit_metrics": (
                    ["variance", "extreme", "package"]
                    if pattern == "package"
                    else ["variance", "extreme"]
                ),
                "homogenization_loci": ["x" if pattern == patterns[0] else "y"],
            }
            for pattern in patterns
        },
        "metric_sources": {
            "variance": {
                "path": ["loci", "$primary_locus", "retention"],
            },
            "extreme": {
                "path": ["loci", "$primary_locus", "extreme"],
            },
            "package": {"path": ["packages", "intact"]},
        },
        "homogenization": {
            "variance_path": ["loci", "$locus", "variance"],
            "threshold_ratio_exclusive": 0.20,
            "consecutive_generations": 2,
        },
        "benefit": {
            "strict_positive_delta": True,
            "treatment_value_positive": True,
            "milestones_required": True,
            "terminal_generation": generations[-1],
            "homogenization_later_required": True,
            "path_reversal_invalidates": True,
            "confirmation_contrasts": [name for name, _baseline, _treatment in contrasts],
        },
        "required_gates": ["parity", "provenance"],
        "row_gates": [
            {
                "name": "mutation_zero",
                "path": ["population", "mutations"],
                "operator": "equals",
                "expected": 0,
            },
            {
                "name": "no_fallback",
                "path": ["integrity", "no_fallback"],
                "operator": "is_true",
            },
            {
                "name": "treatment_event_integrity",
                "path": ["inheritance", "exact"],
                "operator": "is_true",
                "modes": ["repair"],
                "generation_scope": "non_initial",
            },
        ],
        "locking": {
            "contrast": contrasts[0][0],
            "pattern": "package",
            "generation": generations[-1],
            "effective_path": ["ancestry", "effective"],
            "concentration_path": ["ancestry", "top1"],
            "retention_metric": "package",
            "effective_treatment_lte_baseline_multiplier": 0.5,
            "concentration_treatment_gte_baseline_multiplier": 2.0,
            "treatment_retention_gte": 0.9,
        },
        "contribution_reporting": {
            "paths": [paths[-1]],
            "hhi_path": ["reproduction", "hhi"],
            "effective_path": ["reproduction", "effective"],
            "opportunities_path": ["reproduction", "opportunities"],
            "successes_path": ["reproduction", "successes"],
        },
        "decisions": {"rules": rules, "labels": labels},
    }


def _rows(
    contract: dict,
    *,
    confirmed_paths: set[str] | None = None,
    locking_path: str | None = None,
) -> list[dict]:
    identity = contract["identity"]
    arms = tuple(identity.get("arms", ()))
    if not arms:
        arms = (
            identity["baseline_mode"],
            identity["treatment_mode"],
        )
    baseline_arms = {
        contrast["baseline_arm"] for contrast in contract.get("contrasts", ())
    }
    if not baseline_arms:
        baseline_arms = {identity["baseline_mode"]}
    paths = tuple(identity["paths"])
    confirmed = set(paths if confirmed_paths is None else confirmed_paths)
    generations = tuple(contract["generations"])
    rows = []
    for mode in arms:
        for pattern in identity["patterns"]:
            for path in paths:
                for seed in identity["paired_values"]:
                    for generation in generations:
                        initial = generation == generations[0]
                        treated = mode not in baseline_arms
                        wins = path in confirmed
                        if initial:
                            variance = retention = extreme = package = 1.0
                        elif treated and wins:
                            variance = retention = extreme = 0.8
                            package = 0.95
                        else:
                            variance = retention = 0.1
                            extreme = package = 0.0
                        effective = 20.0
                        top1 = 0.1
                        if (
                            treated
                            and path == locking_path
                            and pattern == "package"
                            and generation == generations[-1]
                        ):
                            effective = 8.0
                            top1 = 0.25
                        rows.append(
                            {
                                "mode": mode,
                                "pattern": pattern,
                                "reproduction_path": path,
                                "seed": seed,
                                "generation": generation,
                                "population": {"mutations": 0},
                                "integrity": {"no_fallback": True},
                                "inheritance": {"exact": True},
                                "loci": {
                                    locus: {
                                        "variance": variance,
                                        "retention": retention,
                                        "extreme": extreme,
                                    }
                                    for locus in contract["loci"]
                                },
                                "packages": {"intact": package},
                                "ancestry": {
                                    "effective": effective,
                                    "top1": top1,
                                },
                                "reproduction": {
                                    "hhi": 0.05,
                                    "effective": 20.0,
                                    "opportunities": 10,
                                    "successes": 5,
                                },
                            }
                        )
    return rows


def _analyze(
    contract: dict,
    rows: list[dict],
    gates: dict[str, bool] | None = None,
) -> dict:
    return analyze_inheritance_cross_path(
        rows,
        analysis_contract=contract,
        external_gates={"parity": True, "provenance": True}
        if gates is None
        else gates,
    )


def test_contract_derives_matrix_size_and_stable_hash() -> None:
    contract = InheritanceCrossPathContract.from_mapping(_contract())
    assert contract.expected_cell_count == 2 * 2 * 2 * 3
    assert contract.expected_row_count == contract.expected_cell_count * 9
    assert len(contract.sha256) == 64
    assert contract.sha256 == InheritanceCrossPathContract.from_mapping(
        _contract()
    ).sha256


def test_declarative_nested_identity_paths_accept_chronicle_row_shape() -> None:
    contract = _contract()
    identity = contract["identity"]
    identity.pop("paired_field")
    identity.update(
        {
            "mode_path": ["identity", "mode"],
            "pattern_path": ["identity", "pattern"],
            "path_path": ["identity", "reproduction_path"],
            "paired_path": ["identity", "paired_seed"],
            "generation_path": ["generation"],
        }
    )
    rows = _rows(contract)
    for row in rows:
        row["identity"] = {
            "mode": row.pop("mode"),
            "pattern": row.pop("pattern"),
            "reproduction_path": row.pop("reproduction_path"),
            "paired_seed": row.pop("seed"),
        }

    result = _analyze(contract, rows)

    assert result["decision"] == "PASS"
    assert result["expected_generation_rows"] == len(rows)
    assert result["fail_closed_reasons"] == []


def test_three_arms_and_multiple_declared_contrasts_are_analyzed() -> None:
    contract = _contract(
        arms=("base", "repair", "candidate"),
        contrasts=(
            ("repair_vs_base", "base", "repair"),
            ("candidate_vs_base", "base", "candidate"),
        ),
    )
    contract.pop("locking")
    result = _analyze(contract, _rows(contract))

    assert result["decision"] == "PASS"
    assert result["arms"] == ["base", "repair", "candidate"]
    assert result["contrasts"] == [
        {
            "name": "repair_vs_base",
            "baseline_arm": "base",
            "treatment_arm": "repair",
        },
        {
            "name": "candidate_vs_base",
            "baseline_arm": "base",
            "treatment_arm": "candidate",
        },
    ]
    assert result["expected_cells"] == 3 * 2 * 2 * 3
    assert {
        summary["contrast"]
        for summary in result["paired_milestone_summaries"]
    } == {"repair_vs_base", "candidate_vs_base"}
    assert all(
        set(path_result["contrasts"])
        == {"repair_vs_base", "candidate_vs_base"}
        for path_result in result["path_confirmations"].values()
    )
    assert all(
        path_result["required_contrasts"]
        == ["repair_vs_base", "candidate_vs_base"]
        for path_result in result["path_confirmations"].values()
    )


def test_legacy_two_arm_configuration_preserves_existing_behavior() -> None:
    contract = _contract()
    contract["identity"]["baseline_mode"] = contract["identity"].pop("arms")[0]
    contract["identity"]["treatment_mode"] = "repair"
    contract.pop("contrasts")
    contract["benefit"].pop("confirmation_contrasts")
    contract["locking"].pop("contrast")

    parsed = InheritanceCrossPathContract.from_mapping(contract)
    result = _analyze(contract, _rows(contract))

    assert parsed.modes == ("base", "repair")
    assert tuple(contrast.name for contrast in parsed.contrasts) == (
        "treatment_vs_baseline",
    )
    assert result["decision"] == "PASS"
    assert {
        summary["contrast"]
        for summary in result["paired_milestone_summaries"]
    } == {"treatment_vs_baseline"}


def test_configuration_selects_which_contrasts_gate_the_decision() -> None:
    contract = _contract(
        arms=("base", "repair", "candidate"),
        contrasts=(
            ("repair_vs_base", "base", "repair"),
            ("candidate_vs_base", "base", "candidate"),
        ),
    )
    contract["benefit"]["confirmation_contrasts"] = ["repair_vs_base"]
    contract.pop("locking")
    rows = _rows(contract)
    for row in rows:
        if row["mode"] == "candidate" and row["generation"] != 0:
            for locus in row["loci"].values():
                locus["variance"] = 0.05
                locus["retention"] = 0.05
                locus["extreme"] = 0.0
            row["packages"]["intact"] = 0.0

    result = _analyze(contract, rows)

    assert result["decision"] == "PASS"
    assert result["path_confirmations"]["refill"]["required_contrasts"] == [
        "repair_vs_base"
    ]
    assert (
        result["path_confirmations"]["refill"]["contrasts"][
            "candidate_vs_base"
        ]["confirmed"]
        is False
    )


def test_second_matrix_shape_and_schedule_are_not_engine_constants() -> None:
    contract = _contract(
        paths=("alpha", "beta", "gamma"),
        patterns=("only",),
        pairs=("north", "south"),
        generations=(0, 1, 2, 3, 4),
        milestones=(1, 4),
    )
    # The local locking fixture is optional for this shape.
    contract.pop("locking")
    result = _analyze(contract, _rows(contract))
    assert result["decision"] == "PASS"
    assert result["expected_cells"] == 2 * 3 * 1 * 2
    assert result["expected_generation_rows"] == 2 * 3 * 1 * 2 * 5
    assert {
        row["generation"] for row in result["paired_milestone_summaries"]
    } == {1, 4}


@pytest.mark.parametrize(
    ("confirmed", "expected"),
    [
        ({"refill", "live"}, "PASS"),
        ({"refill"}, "REFILL"),
        ({"live"}, "LIVE"),
        (set(), "FAIL"),
    ],
)
def test_decisions_are_selected_by_contract_rules(
    confirmed: set[str], expected: str
) -> None:
    contract = _contract()
    result = _analyze(contract, _rows(contract, confirmed_paths=confirmed))
    assert result["decision"] == expected


def test_locking_rule_is_entirely_contract_supplied() -> None:
    contract = _contract()
    result = _analyze(contract, _rows(contract, locking_path="live"))
    assert result["decision"] == "LOCK"
    assert result["locking"]["paths"] == ["live"]


def test_external_and_row_gates_precede_scientific_decisions() -> None:
    contract = _contract()
    result = _analyze(
        contract,
        _rows(contract),
        {"parity": False, "provenance": True},
    )
    assert result["decision"] == "FAIL"
    assert result["fail_closed_reasons"] == ["external_gate_failed:parity"]

    rows = _rows(contract)
    rows[0]["population"]["mutations"] = 1
    result = _analyze(contract, rows)
    assert result["decision"] == "FAIL"
    assert "row_gate_failed:mutation_zero" in result["fail_closed_reasons"][0]


def test_missing_and_duplicate_rows_fail_closed() -> None:
    contract = _contract()
    missing = _rows(contract)
    missing.pop()
    assert _analyze(contract, missing)["decision"] == "FAIL"
    duplicate = _rows(contract)
    duplicate.append(copy.deepcopy(duplicate[-1]))
    assert _analyze(contract, duplicate)["decision"] == "FAIL"


def test_path_reversal_fails_closed() -> None:
    contract = _contract()
    rows = _rows(contract)
    target = next(
        row
        for row in rows
        if row["mode"] == "repair"
        and row["reproduction_path"] == "live"
        and row["pattern"] == "scalar"
        and row["seed"] == 11
        and row["generation"] == 4
    )
    target["loci"]["x"]["retention"] = 0.01
    result = _analyze(contract, rows)
    assert result["decision"] == "FAIL"
    assert result["fail_closed_reasons"] == ["path_reversal"]


def test_all_pairs_are_required_and_contribution_report_is_observational() -> None:
    contract = _contract()
    rows = _rows(contract)
    for row in rows:
        if (
            row["mode"] == "repair"
            and row["reproduction_path"] == "live"
            and row["pattern"] == "scalar"
            and row["seed"] == 33
            and row["generation"] != 0
        ):
            row["loci"]["x"]["retention"] = 0.1
            row["loci"]["x"]["extreme"] = 0.0
    result = _analyze(contract, rows)
    assert result["decision"] == "REFILL"
    assert result["path_confirmations"]["live"]["confirmed"] is False
    assert result["contribution_inequality"]["observational_only"] is True


def test_invalid_contract_is_rejected_before_observations() -> None:
    contract = _contract()
    contract["milestones"] = [999]
    with pytest.raises(InheritanceCrossPathAnalysisError):
        InheritanceCrossPathContract.from_mapping(contract)


@pytest.mark.parametrize(
    "malformation",
    [
        "duplicate_arm",
        "non_string_arm",
        "empty_contrasts",
        "unknown_arm",
        "same_arm",
        "duplicate_name",
        "duplicate_pair",
        "mixed_identity_forms",
        "unknown_confirmation_contrast",
        "missing_locking_contrast",
    ],
)
def test_malformed_arm_and_contrast_declarations_fail_closed(
    malformation: str,
) -> None:
    contract = _contract(
        arms=("base", "repair", "candidate"),
        contrasts=(
            ("repair_vs_base", "base", "repair"),
            ("candidate_vs_base", "base", "candidate"),
        ),
    )
    if malformation == "duplicate_arm":
        contract["identity"]["arms"] = ["base", "repair", "repair"]
    elif malformation == "non_string_arm":
        contract["identity"]["arms"] = ["base", "repair", 3]
    elif malformation == "empty_contrasts":
        contract["contrasts"] = []
    elif malformation == "unknown_arm":
        contract["contrasts"][0]["treatment_arm"] = "absent"
    elif malformation == "same_arm":
        contract["contrasts"][0]["treatment_arm"] = "base"
    elif malformation == "duplicate_name":
        contract["contrasts"][1]["name"] = contract["contrasts"][0]["name"]
    elif malformation == "duplicate_pair":
        contract["contrasts"][1]["baseline_arm"] = "base"
        contract["contrasts"][1]["treatment_arm"] = "repair"
    elif malformation == "mixed_identity_forms":
        contract["identity"]["baseline_mode"] = "base"
    elif malformation == "unknown_confirmation_contrast":
        contract["benefit"]["confirmation_contrasts"] = ["absent"]
    elif malformation == "missing_locking_contrast":
        contract["locking"].pop("contrast")
    else:  # pragma: no cover - parameter list and branches must stay aligned.
        raise AssertionError(malformation)

    with pytest.raises(InheritanceCrossPathAnalysisError):
        InheritanceCrossPathContract.from_mapping(contract)
