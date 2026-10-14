"""Profile decision logic for campaign evidence."""

from __future__ import annotations

from statistics import mean
from typing import Any, Mapping

from tools.evidence.paired import ComparisonSpec, compare_paired
from tools.evidence.profiles import (
    COG_P1_ECOLOGIES,
    COG_P1_INIT_ARMS,
    COG_P1_SEEDS,
    ProfileContract,
    get_profile,
)
from tools.evidence.reward_audit import audit_reward_leakage
from tools.evidence.schema import parse_number


COG_P1_PAIR_METRICS = (
    "alive",
    "mean_offspring_total",
    "behavior_patch_interaction_proxy",
    "behavior_risk_exposure_proxy",
    "perception_cost_charged_mean",
)

QUAL_DELTA_REQUIRED_FLAGS = (
    "qf_route_receipt_present_flag",
    "qf_route_receipt_sha256_match_flag",
    "qf_route_binding_exact_flag",
    "qf_source_binding_exact_flag",
    "qf_coverage_complete_flag",
    "qf_repeat_parity_pass_flag",
    "qf_checkpoint_resume_pass_flag",
    "qf_finite_outputs_flag",
    "qf_no_integrity_failure_flag",
    "qf_no_posthoc_widening_flag",
)


def _flatten_run_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = {k: v for k, v in row.items() if k != "final_metrics"}
        metrics = row.get("final_metrics")
        if isinstance(metrics, Mapping):
            item.update(metrics)
        out.append(item)
    return out


def _summary_index(comparison: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(row.get("ecology_context") or ""), str(row.get("init_arm") or "")): row
        for row in comparison.get("summary", [])
    }


def _metric_mean(summary_row: Mapping[str, Any], metric: str) -> float | None:
    payload = summary_row.get(metric)
    if isinstance(payload, Mapping):
        return parse_number(payload.get("mean_delta"))
    return None


def _mean_present(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return mean(present) if present else None


def analyze_cog_p1(schema_status: Mapping[str, Any], profile: ProfileContract | None = None) -> dict[str, Any]:
    profile = profile or get_profile("COG-P1")
    rows = list(schema_status.get("run_status") or [])
    flat_rows = _flatten_run_rows(rows)
    reward_audit = audit_reward_leakage(rows, profile.reward_leakage_columns)

    normal_vs_ablated = compare_paired(
        flat_rows,
        ComparisonSpec(
            name="normal_vs_ablated",
            metric_names=COG_P1_PAIR_METRICS,
            group_by_dimensions=("ecology_context", "init_arm"),
            pair_by_dimensions=("seed",),
            left_filter={"perception_condition": "perception_normal"},
            right_filter={"perception_condition": "perception_ablated"},
            delta_label="normal_minus_ablated",
        ),
    )
    normal_index = _summary_index(normal_vs_ablated)

    flat_alive = _mean_present([_metric_mean(normal_index.get((COG_P1_ECOLOGIES[0], init), {}), "alive") for init in COG_P1_INIT_ARMS])
    flat_births = _mean_present(
        [_metric_mean(normal_index.get((COG_P1_ECOLOGIES[0], init), {}), "mean_offspring_total") for init in COG_P1_INIT_ARMS]
    )
    pressure_specificity: list[dict[str, Any]] = []
    pressure_alive_deltas: list[float] = []
    pressure_birth_deltas: list[float] = []
    for ecology in COG_P1_ECOLOGIES[1:]:
        alive = _mean_present([_metric_mean(normal_index.get((ecology, init), {}), "alive") for init in COG_P1_INIT_ARMS])
        births = _mean_present([_metric_mean(normal_index.get((ecology, init), {}), "mean_offspring_total") for init in COG_P1_INIT_ARMS])
        alive_minus_flat = None if alive is None or flat_alive is None else alive - flat_alive
        births_minus_flat = None if births is None or flat_births is None else births - flat_births
        if alive_minus_flat is not None:
            pressure_alive_deltas.append(alive_minus_flat)
        if births_minus_flat is not None:
            pressure_birth_deltas.append(births_minus_flat)
        pressure_specificity.append(
            {
                "ecology_context": ecology,
                "pressure_mean_delta_alive": alive,
                "flat_mean_delta_alive": flat_alive,
                "pressure_minus_flat_delta_alive": alive_minus_flat,
                "pressure_mean_delta_mean_offspring_total": births,
                "flat_mean_delta_mean_offspring_total": flat_births,
                "pressure_minus_flat_delta_mean_offspring_total": births_minus_flat,
            }
        )

    founder_summary: list[dict[str, Any]] = []
    founder_contradictions = 0
    for ecology in COG_P1_ECOLOGIES:
        legacy_alive = _metric_mean(normal_index.get((ecology, "init_legacy"), {}), "alive")
        balanced_alive = _metric_mean(normal_index.get((ecology, "init_founder_balanced"), {}), "alive")
        delta = None if legacy_alive is None or balanced_alive is None else balanced_alive - legacy_alive
        if legacy_alive is not None and balanced_alive is not None and legacy_alive > 0 and balanced_alive <= 0:
            founder_contradictions += 1
        founder_summary.append(
            {
                "ecology_context": ecology,
                "legacy_mean_delta_alive": legacy_alive,
                "founder_balanced_mean_delta_alive": balanced_alive,
                "founder_balanced_minus_legacy_delta_alive": delta,
            }
        )

    cost_values = [
        parse_number(row.get("perception_cost_charged_mean"))
        for row in flat_rows
        if row.get("perception_cost_charged_mean") not in (None, "")
    ]
    cost_summary = {
        "observed_count": len([value for value in cost_values if value is not None]),
        "mean_perception_cost_charged": _mean_present(cost_values),
        "min_perception_cost_charged": min([value for value in cost_values if value is not None], default=None),
        "max_perception_cost_charged": max([value for value in cost_values if value is not None], default=None),
    }

    paired_coverage = sum(int(row.get("paired_count") or 0) for row in normal_vs_ablated["summary"])
    expected_pairs = len(COG_P1_ECOLOGIES) * len(COG_P1_INIT_ARMS) * len(COG_P1_SEEDS)
    coverage_ok = paired_coverage >= expected_pairs
    pressure_signal = bool(pressure_alive_deltas and pressure_birth_deltas and min(pressure_alive_deltas) > 0 and min(pressure_birth_deltas) > 0)

    errors = list(schema_status.get("errors") or [])
    decision = "B"
    rationale: list[str] = []
    if reward_audit["severity"] == "fatal":
        decision = "D"
        rationale.append("Reward leakage or direct reward/pressure flag was active.")
    elif not schema_status.get("ok") or not coverage_ok or reward_audit["severity"] == "ambiguous":
        decision = "C"
        if not schema_status.get("ok"):
            rationale.append("Schema, completeness, or control issues prevent interpretation.")
        if not coverage_ok:
            rationale.append(f"Paired coverage {paired_coverage}/{expected_pairs} is incomplete.")
        if reward_audit["severity"] == "ambiguous":
            rationale.append("Reward audit is ambiguous because expected or suspicious reward fields need review.")
    elif founder_contradictions:
        decision = "C"
        rationale.append("Founder-balanced arm contradicts the primary signal severely.")
    elif pressure_signal:
        decision = "A"
        rationale.append("Pressure-context normal-minus-ablated deltas exceed flat-control deltas with complete paired coverage.")
    else:
        decision = "B"
        rationale.append("Campaign is valid, but the expected pressure-specific perception signal is not supported.")

    claims_allowed = [
        "The report can describe bounded COG-P1 matrix coverage and procedural evidence status.",
        "The report can compare descriptive paired deltas within the supplied campaign matrix.",
    ]
    if decision == "A":
        claims_allowed.append("The supplied outputs support the bounded COG-P1 go criterion for pressure-specific perception signal.")
    elif decision == "B":
        claims_allowed.append("The supplied outputs support a valid no-go profile decision for the bounded COG-P1 criterion.")

    claims_not_allowed = [
        "Do not claim cognition emergence.",
        "Do not claim open-ended evolution.",
        "Do not claim ecological generality beyond the COG-P1 campaign matrix.",
        "Do not use p-values, significance language, or proof language.",
    ]
    next_action = {
        "A": "Prepare a review dossier and decide whether to run the next bounded profile.",
        "B": "Review the bounded no-go result and decide whether a new hypothesis or profile is warranted.",
        "C": "Repair reporting, completeness, schema, or control issues before interpretation.",
        "D": "Invalidate this campaign for decision purposes and remove the fatal confound before any rerun.",
    }[decision]

    return {
        "profile": "COG-P1",
        "decision": decision,
        "rationale": rationale or errors,
        "claims_allowed": claims_allowed,
        "claims_not_allowed": claims_not_allowed,
        "recommended_next_action": next_action,
        "normal_vs_ablated": normal_vs_ablated,
        "pressure_specificity": pressure_specificity,
        "founder_summary": founder_summary,
        "cost_summary": cost_summary,
        "reward_audit": reward_audit,
        "paired_coverage": {"observed": paired_coverage, "expected": expected_pairs, "ok": coverage_ok},
        "pressure_signal_supported": pressure_signal,
    }


def analyze_qual_delta_1(schema_status: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(schema_status.get("run_status") or [])
    flat_rows = _flatten_run_rows(rows)
    complete_rows = [row for row in flat_rows if row.get("status") == "complete"]
    errors = list(schema_status.get("errors") or [])
    required_failures: list[str] = []
    region_supported = False
    integrity_failure = False

    for row in complete_rows:
        for flag in QUAL_DELTA_REQUIRED_FLAGS:
            value = parse_number(row.get(flag))
            if value is None or value <= 0:
                required_failures.append(
                    f"{row.get('run_id') or row.get('scenario_id')}: {flag} is not true"
                )
        no_integrity = parse_number(row.get("qf_no_integrity_failure_flag"))
        if no_integrity is not None and no_integrity <= 0:
            integrity_failure = True
        supported = parse_number(row.get("qf_region_supported_flag"))
        if supported is not None and supported > 0:
            region_supported = True

    decision = "B"
    rationale: list[str] = []
    if integrity_failure:
        decision = "D"
        rationale.append("Integrity, deterministic, route, or fatal output failure invalidates the qualification evidence.")
    elif not schema_status.get("ok") or not complete_rows:
        decision = "C"
        rationale.append("Schema, reporting, completeness, or control issue prevents interpretation.")
    elif required_failures:
        decision = "C"
        rationale.append("One or more mandatory qualification gates is incomplete or false.")
    elif region_supported:
        decision = "A"
        rationale.append("Predeclared region is supported by complete route/source-bound qualification evidence.")
    else:
        decision = "B"
        rationale.append("Campaign is valid, but interpolation or equivalence is not supported.")

    return {
        "profile": "QUAL-DELTA-1",
        "decision": decision,
        "rationale": rationale or errors,
        "claims_allowed": [
            "The report may state the bounded procedural qualification decision.",
            "An A decision may accept only the predeclared region or a narrower named region.",
            "B, C, and D decisions do not expand any qualification envelope.",
        ],
        "claims_not_allowed": [
            "Do not claim intelligence, emergence, cognition, or ecological generality.",
            "Do not infer campaign-specific science from engineering qualification.",
            "Do not widen an envelope after seeing favorable outputs.",
            "Do not bypass exact route or source evidence binding.",
        ],
        "recommended_next_action": {
            "A": "Bind the accepted region and exclusions into a QF-1 catalog record for external review.",
            "B": "Preserve exact/slice/path evidence and revise the candidate envelope before another campaign.",
            "C": "Repair reporting, completeness, schema, or controls before interpretation.",
            "D": "Invalidate this evidence for qualification and repair the integrity failure before rerun.",
        }[decision],
        "reward_audit": {"severity": "not_applicable", "checked_columns": []},
        "paired_coverage": {
            "observed": len(complete_rows),
            "expected": schema_status.get("run_count_expected", 0),
            "ok": bool(complete_rows) and len(required_failures) == 0,
        },
        "qualification_gate_failures": sorted(set(required_failures)),
        "region_supported": region_supported,
        "integrity_failure": integrity_failure,
    }


def analyze_profile(schema_status: Mapping[str, Any], profile_name: str) -> dict[str, Any]:
    profile = get_profile(profile_name)
    if profile.name == "COG-P1":
        return analyze_cog_p1(schema_status, profile)
    if profile.name == "QUAL-DELTA-1":
        return analyze_qual_delta_1(schema_status)
    raise NotImplementedError(f"{profile.name} decision reporting is reserved but not implemented")
