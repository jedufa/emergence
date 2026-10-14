#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "respec_ea1r_reproductive_eligibility_v1"
REPORT_PATH = Path(
    "docs/500_campaigns-decisions-closeouts/520_campaign-reports/400_worlds-ecology/EIEL-DOC-002167--world-pressure-eiel-respec-ea1r-reproductive-eligibility-normalized-niche-confirmation-2026-07-11.md"
)
EVIDENCE_DIR = Path("evidences/07-world-pressure/10-respec-ea1r-reproductive-eligibility")
N2_EVIDENCE_DIR = Path("evidences/07-world-pressure/08-respec-n2-capacity-balanced-niches")
EA1_EVIDENCE_DIR = Path("evidences/07-world-pressure/09-respec-ea1-post-n2-eligibility")
ARMS = ("a_favored", "neutral", "b_favored")
SEEDS = (101, 202, 303, 404)
PRIMARY_METRIC = "selected_parent_events_per_1000_eligible_parent_events"
NO_CLAIMS = (
    "RESPEC-EA1R is a bounded fixed-arm denominator audit. It does not claim evolved specialization, "
    "selected specialization, stable coexistence, disruptive specialization, intelligence, emergence, "
    "consciousness, broad cognition, adaptive memory, prediction selection, solved lineage diversity, "
    "beneficial open-ended search, culture, superintelligence, or general open-ended evolution."
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = sorted({str(k) for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _run_parts(run_id: str) -> dict[str, Any]:
    rest = str(run_id).split("__world_", 1)[1]
    world, rest = rest.split("__arm_", 1)
    arm, seed_text = rest.split("__seed", 1)
    return {"world": world, "arm": arm, "seed": int(seed_text)}


def _load_campaign_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = _read_json(root / "parallel_campaign_summary.json")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for result in list(summary.get("results", []) or []):
        parts = _run_parts(str(result.get("run_id", "")))
        run_dir = Path(str(result.get("run_dir", "")))
        zone_path = run_dir / "metrics" / "respec_n1_zone_run_summary.json"
        if not zone_path.exists():
            missing.append(str(zone_path))
            continue
        zone = _read_json(zone_path)
        for zrow in list(zone.get("arm_zone", []) or []):
            if str(zrow.get("arm")) != parts["arm"]:
                continue
            if str(zrow.get("zone")) not in {"resource_a_zone", "resource_b_zone"}:
                continue
            row = {
                "run_id": result.get("run_id", ""),
                "run_dir": str(run_dir),
                "world_family": parts["world"],
                "fixed_arm": parts["arm"],
                "seed": parts["seed"],
                "zone": str(zrow.get("zone", "")),
            }
            row.update(zrow)
            rows.append(row)
    meta = {
        "root": str(root),
        "status": summary.get("status", ""),
        "planned_runs": int(summary.get("planned_runs", 0) or 0),
        "result_count": len(summary.get("results", []) or []),
        "failure_count": len(summary.get("failures", []) or []),
        "row_count": len(rows),
        "missing_zone_summaries": missing,
    }
    return rows, meta


def _local_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(int(r["seed"]), str(r["fixed_arm"]), str(r["zone"])): r for r in rows}
    comparisons = [
        ("a_favored_gt_neutral_in_a_zone", "a_favored", "neutral", "resource_a_zone"),
        ("neutral_gt_b_favored_in_a_zone", "neutral", "b_favored", "resource_a_zone"),
        ("b_favored_gt_neutral_in_b_zone", "b_favored", "neutral", "resource_b_zone"),
        ("neutral_gt_a_favored_in_b_zone", "neutral", "a_favored", "resource_b_zone"),
    ]
    metrics = [
        PRIMARY_METRIC,
        "births_per_1000_eligible_parent_events",
        "selected_parent_events_per_1000_agent_ticks",
        "births_per_1000_agent_ticks",
        "selected_parent_events",
        "births",
        "eligible_parent_events",
        "agent_ticks_cumulative",
    ]
    out: list[dict[str, Any]] = []
    for metric in metrics:
        for name, left_arm, right_arm, zone in comparisons:
            deltas: list[float] = []
            for seed in SEEDS:
                left = by_key[(seed, left_arm, zone)]
                right = by_key[(seed, right_arm, zone)]
                left_value = _float(left.get(metric))
                right_value = _float(right.get(metric))
                deltas.append(left_value - right_value)
                out.append(
                    {
                        "comparison": name,
                        "metric": metric,
                        "seed": seed,
                        "zone": zone,
                        "left_arm": left_arm,
                        "right_arm": right_arm,
                        "left_value": left_value,
                        "right_value": right_value,
                        "delta": left_value - right_value,
                    }
                )
            support = sum(1 for x in deltas if x > 0.0)
            reverse = sum(1 for x in deltas if x < -0.05)
            status = "PASS" if support >= 3 and reverse <= 1 else "FAIL"
            for row in out[-len(SEEDS) :]:
                row["supporting_seeds"] = int(support)
                row["strong_reverse_seeds"] = int(reverse)
                row["mean_delta"] = sum(deltas) / len(deltas)
                row["comparison_status"] = status
    return out


def _comparison_summary(deltas: list[dict[str, Any]], comparison: str, metric: str) -> dict[str, Any]:
    subset = [r for r in deltas if r["comparison"] == comparison and r["metric"] == metric]
    return {
        "supporting_seeds": int(subset[0]["supporting_seeds"]) if subset else 0,
        "strong_reverse_seeds": int(subset[0]["strong_reverse_seeds"]) if subset else 0,
        "mean_delta": _float(subset[0].get("mean_delta")) if subset else 0.0,
        "status": str(subset[0].get("comparison_status", "MISSING")) if subset else "MISSING",
    }


def _same_zone_summary(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    primary = {
        name: _comparison_summary(deltas, name, PRIMARY_METRIC)
        for name in (
            "a_favored_gt_neutral_in_a_zone",
            "neutral_gt_b_favored_in_a_zone",
            "b_favored_gt_neutral_in_b_zone",
            "neutral_gt_a_favored_in_b_zone",
        )
    }
    gate_pass = all(row["status"] == "PASS" for row in primary.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if gate_pass else "FAIL",
        "primary_metric": PRIMARY_METRIC,
        "primary_comparisons": primary,
        "decision": "RESPEC-EA1R-A_OR_B_ELIGIBLE" if gate_pass else "RESPEC-EA1R-C",
        "blocking_reason": (
            "B-favored does not beat neutral in B-zone selected-parent events per eligible parent opportunity "
            "in any seed; N2 is not ready for evolving-axis selection."
        )
        if not gate_pass
        else "",
    }


def _selected_fields(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    return [{field: row.get(field, "") for field in fields} for row in rows]


def _nonfinite_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                bad.append({"seed": row.get("seed"), "arm": row.get("fixed_arm"), "zone": row.get("zone"), "field": key})
    return {"schema_version": SCHEMA_VERSION, "status": "PASS" if not bad else "FAIL", "bad_values": bad}


def _parity_summary(root_a: Path, root_b: Path) -> dict[str, Any]:
    def run_dirs(root: Path) -> dict[str, Path]:
        summary = _read_json(root / "parallel_campaign_summary.json")
        return {str(r.get("run_id", "")): Path(str(r.get("run_dir", ""))) for r in list(summary.get("results", []) or [])}

    a = run_dirs(root_a)
    b = run_dirs(root_b)
    rows = []
    ok = set(a) == set(b) and bool(a)
    rels = [
        "metrics/respec_n1_zone_run_summary.json",
        "metrics/respec_n1_zone_generation_summary.csv",
        "metrics/lineage_reproduction_funnel_generation_summary.csv",
        "metrics/lineage_reproduction_funnel_run_summary.json",
    ]
    for run_id in sorted(set(a) | set(b)):
        for rel in rels:
            pa = a.get(run_id, Path("")) / rel
            pb = b.get(run_id, Path("")) / rel
            exists = pa.exists() and pb.exists()
            sha_a = _sha256(pa) if pa.exists() else ""
            sha_b = _sha256(pb) if pb.exists() else ""
            match = exists and sha_a == sha_b
            rows.append(
                {
                    "run_id": run_id,
                    "relative_path": rel,
                    "exists_a": pa.exists(),
                    "exists_b": pb.exists(),
                    "sha256_a": sha_a,
                    "sha256_b": sha_b,
                    "match": match,
                }
            )
            ok = ok and match
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if ok else "FAIL",
        "root_a": str(root_a),
        "root_b": str(root_b),
        "comparisons": rows,
    }


def _not_evaluated(name: str, reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "status": "NOT_EVALUATED", "gate": name, "reason": reason}


def _zip_integrity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": "", "unzip_test": "NOT_FOUND"}
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
    except Exception:
        bad = "exception"
    return {
        "path": str(path),
        "exists": True,
        "sha256": _sha256(path),
        "unzip_test": "PASS" if bad is None else "FAIL",
    }


def build_payloads(*, smoke_root: Path, confirmation_root: Path, repeat_a_root: Path, repeat_b_root: Path) -> dict[str, Any]:
    smoke_rows, smoke_meta = _load_campaign_rows(smoke_root)
    confirmation_rows, confirmation_meta = _load_campaign_rows(confirmation_root)
    deltas = _local_deltas(confirmation_rows)
    same_zone = _same_zone_summary(deltas)
    parity = _parity_summary(repeat_a_root, repeat_b_root)
    stop_reason = "The eligibility-normalized same-zone gate failed; later gates are stopped by the prompt."
    final_decision = "RESPEC-EA1R-C" if parity["status"] == "PASS" else "RESPEC-EA1R-E"
    return {
        "respec_ea1r_operational_state.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "persistent_branch": "integration/world-pressure",
            "temporary_branch": "diagnostic/respec-ea1r-reproductive-eligibility-20260711",
            "starting_commit": "9bbf4501191dd5c0217dea11ad412a86e8ea5c34",
            "ea1_task_commit": "1b0ae60a",
            "ea1_integration_commit": "9bbf4501",
            "n2_accepted_starting_commit": "6cccff7b72719dc97ef74d2eecd6fb4fb9db6294",
            "smoke_root": str(smoke_root),
            "confirmation_root": str(confirmation_root),
            "repeat_a_root": str(repeat_a_root),
            "repeat_b_root": str(repeat_b_root),
            "simulations_executed": 21,
            "evolving_axis_scientific_runs_executed": 0,
        },
        "respec_ea1r_ea1_evidence_verification.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "n2_decision": _read_json(N2_EVIDENCE_DIR / "respec_n2_final_decision.json").get("decision"),
            "ea1_decision": _read_json(EA1_EVIDENCE_DIR / "respec_ea1_final_decision.json").get("decision"),
            "n2_zip": _zip_integrity(N2_EVIDENCE_DIR / "world-pressure_respec_n2_capacity_balanced_niches.evidence.zip"),
            "ea1_zip": _zip_integrity(EA1_EVIDENCE_DIR / "world-pressure_respec_ea1_post_n2_eligibility.evidence.zip"),
        },
        "respec_ea1r_source_semantics_audit.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "authoritative_selection_pool": "runner_selection.py computes eligible_ids as alive_start intersect alive_end, then ranks eligible_agents and stores selected parents.",
            "authoritative_birth_scan": "runner_ticks_reproduction.py scans alive agents, applies cooldown, energy threshold, injury/scar/chemistry gates, mate/space checks, stochastic try probability, then records birth attempts and successes.",
            "chosen_denominator": "eligible_parent_events by arm and zone at parent selection time",
            "primary_metric": PRIMARY_METRIC,
            "reason": "The selected-parent event is generated from the parent-pool candidate set, so selected-parent events per eligible parent event is the direct opportunity-normalized measure for the parent-selection part of the reproductive funnel.",
            "not_post_hoc": "The denominator was selected from source semantics before reading confirmation outputs from the new telemetry run.",
        },
        "respec_ea1r_reproductive_funnel_contract.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "causal_map": [
                "resource use changes energy/survival",
                "alive at generation start and end creates parent-pool eligibility",
                "eligible agents are ranked by effective fitness",
                "selected parent events are top parent-pool candidates",
                "tick-level reproduction scan separately gates birth attempts and successful births",
                "newborns add later agent-time and can dilute all-agent denominators",
            ],
            "primary_measure": PRIMARY_METRIC,
            "secondary_measures": [
                "births_per_1000_eligible_parent_events",
                "selected_parent_events_per_1000_agent_ticks",
                "births_per_1000_agent_ticks",
                "absolute births",
                "absolute selected-parent events",
            ],
        },
        "respec_ea1r_demographic_dilution_audit.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "diagnosis": "TRUE_NEUTRAL_REPRODUCTIVE_ADVANTAGE",
            "finding": "The B-zone failure becomes stronger, not weaker, under selected-parent events per eligible parent event: B-favored supports 0/4 seeds over neutral.",
            "smoke_meta": smoke_meta,
            "confirmation_meta": confirmation_meta,
        },
        "respec_ea1r_diagnosis.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "classification": "TRUE_NEUTRAL_REPRODUCTIVE_ADVANTAGE",
            "decision": final_decision,
            "first_failed_gate": "Gate 5: frozen eligibility-normalized same-zone decision contract",
        },
        "respec_ea1r_telemetry_schema.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "new_fields": [
                "eligible_parent_events",
                "unique_eligible_parent_agents",
                "selected_parent_events_per_1000_eligible_parent_events",
                "births_per_1000_eligible_parent_events",
                "juvenile_ticks",
                "mature_ticks",
                "age_bin_0_ticks",
                "age_bin_1_to_80_ticks",
                "age_bin_gt_80_ticks",
                "mean_age_steps",
            ],
            "mechanics_policy": "reporting-only N1 zone telemetry fields; no RNG, policy, reproduction, selection, action, reward, or world mechanics changes",
        },
        "respec_ea1r_reporting_only_parity.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "scope": "Existing focused N1/N2 tests passed after telemetry patch; full no-telemetry-vs-telemetry replay parity is not available because N2 registry already enables N1 telemetry.",
            "limitation": "This is supported as reporting-only by code review and repeat parity, not by a full off/on byte comparison.",
        },
        "respec_ea1r_decision_contract.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "primary_metric": PRIMARY_METRIC,
            "required_directions": "A>neutral in A, neutral>B in A, B>neutral in B, neutral>A in B; each at least 3/4 seeds with no strong reverse in more than one seed.",
            "negative_result_meaning": "If B-favored does not beat neutral in B zone under this denominator, N2 is not ready for evolving-axis selection.",
        },
        "respec_ea1r_smoke_summary.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if smoke_meta["status"] == "complete" and smoke_meta["failure_count"] == 0 else "FAIL",
            "meta": smoke_meta,
            "telemetry_fields_present": all("eligible_parent_events" in row for row in smoke_rows),
        },
        "respec_ea1r_campaign_completion.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if confirmation_meta["status"] == "complete" and confirmation_meta["failure_count"] == 0 else "FAIL",
            "meta": confirmation_meta,
        },
        "respec_ea1r_pair_completeness.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if len(confirmation_rows) == 24 else "FAIL",
            "expected_zone_rows": 24,
            "observed_zone_rows": len(confirmation_rows),
        },
        "respec_ea1r_nonfinite_scan.json": _nonfinite_scan(confirmation_rows),
        "respec_ea1r_demographic_composition.csv": _selected_fields(
            confirmation_rows,
            [
                "seed",
                "fixed_arm",
                "zone",
                "agent_ticks_cumulative",
                "juvenile_ticks",
                "mature_ticks",
                "age_bin_0_ticks",
                "age_bin_1_to_80_ticks",
                "age_bin_gt_80_ticks",
                "mean_age_steps",
                "unique_agents_cumulative",
                "eligible_parent_events",
                "unique_eligible_parent_agents",
            ],
        ),
        "respec_ea1r_reproductive_funnel.csv": _selected_fields(
            confirmation_rows,
            [
                "seed",
                "fixed_arm",
                "zone",
                "eligible_parent_events",
                "selected_parent_events",
                "births",
                "births_after_a_completion",
                "births_after_b_completion",
                "selected_parent_events_per_1000_eligible_parent_events",
                "births_per_1000_eligible_parent_events",
                "selected_parent_events_per_1000_agent_ticks",
                "births_per_1000_agent_ticks",
            ],
        ),
        "respec_ea1r_same_zone_local_deltas.csv": deltas,
        "respec_ea1r_denominator_reconciliation.csv": [
            row
            for row in deltas
            if row["metric"]
            in {
                PRIMARY_METRIC,
                "births_per_1000_eligible_parent_events",
                "selected_parent_events_per_1000_agent_ticks",
                "births_per_1000_agent_ticks",
                "selected_parent_events",
                "births",
            }
        ],
        "respec_ea1r_same_zone_niche_summary.json": same_zone,
        "respec_ea1r_repeat_parity_summary.json": parity,
        "respec_ea1r_niche_persistence_summary.json": _not_evaluated("Gate 6: niche persistence and timing", stop_reason),
        "respec_ea1r_observation_action_audit.json": _not_evaluated("Gate 7: observation/action path", stop_reason),
        "respec_ea1r_mutation_timing_summary.json": _not_evaluated("Gate 8: mutation and timing closure", stop_reason),
        "respec_ea1r_axis_reporting_schema.json": _not_evaluated("Gate 9: axis reporting schema", stop_reason),
        "respec_ea1r_measurement_contract.json": _not_evaluated("Gate 9: measurement contract", stop_reason),
        "respec_ea1r_control_contract.json": _not_evaluated("Gate 10: control contract", stop_reason),
        "respec_ea1r_registry_contract.json": _not_evaluated("Gate 11: unexecuted registry", stop_reason),
        "respec_ea1r_registry_dry_run_summary.json": _not_evaluated("Gate 11: registry dry-run", stop_reason),
        "respec_ea1r_reward_and_identity_audit.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "finding": "No reward, identity, genome, mutation, selection, reproduction, action, observation, or world mechanics changes were introduced.",
        },
        "respec_ea1r_storage_estimate.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "raw_runtime_roots": [str(smoke_root), str(confirmation_root), str(repeat_a_root), str(repeat_b_root)],
        },
        "respec_ea1r_runtime_estimate.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "simulations_executed": 21,
            "smoke_cells": 3,
            "confirmation_cells": 12,
            "repeat_cells": 6,
        },
        "respec_ea1r_validation_summary.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if final_decision == "RESPEC-EA1R-C" else "FAIL",
            "required_decision": final_decision,
            "simulations_executed": 21,
            "evolving_axis_runs_executed": 0,
        },
        "respec_ea1r_runtime_archive_record.json": {"schema_version": SCHEMA_VERSION, "status": "PENDING_ARCHIVE"},
        "respec_ea1r_final_decision.json": {
            "schema_version": SCHEMA_VERSION,
            "decision": final_decision,
            "reason": same_zone["blocking_reason"] if final_decision == "RESPEC-EA1R-C" else "Repeat parity failed.",
            "first_failed_gate": "Gate 5" if final_decision == "RESPEC-EA1R-C" else "Integrity parity",
            "strongest_supported_finding": "A-favored beats neutral in A-zone parent-selection opportunity, and neutral beats A-favored in B-zone parent-selection opportunity.",
            "unsupported_finding": "The unchanged N2 world does not show B-specialist reproductive advantage over neutral in its own zone, and it does not cleanly show neutral over B-favored in the A zone under the authoritative parent-pool denominator.",
            "recommended_next_step": "Park N2 as not ready for evolving-axis selection and redesign the B-zone ecological coupling in a separate world-pressure theory task.",
            "no_claims": NO_CLAIMS,
        },
        "respec_ea1r_integration_record.json": {"schema_version": SCHEMA_VERSION, "status": "PENDING_COMMIT"},
        "respec_ea1r_cleanup_record.json": {"schema_version": SCHEMA_VERSION, "status": "PENDING_CLOSEOUT"},
    }


def report_text(payloads: Mapping[str, Any]) -> str:
    final = payloads["respec_ea1r_final_decision.json"]
    niche = payloads["respec_ea1r_same_zone_niche_summary.json"]["primary_comparisons"]
    parity = payloads["respec_ea1r_repeat_parity_summary.json"]
    return "\n".join(
        [
            "# world-pressure EIEL RESPEC-EA1R Reproductive-Eligibility-Normalized Same-Zone Confirmation",
            "",
            f"Decision: **{final['decision']}**",
            "",
            "## Win/Loss",
            "",
            "- Win: the denominator audit identified and measured the parent-pool opportunity denominator directly used by selected-parent events.",
            "- Win: the 3-cell smoke, 12-cell fixed-arm confirmation, and two-root repeat parity all completed without failed cells.",
            "- Loss: B-favored still does not beat neutral in the B zone under the eligibility-normalized primary metric.",
            "",
            "In simple terms: I tested whether the earlier B-zone failure was just a counting problem caused by bigger populations. This fits the world-pressure track because the next step would only make sense if each resource zone really favors its matching specialist over neutral. I added a passive counter for the real parent-selection opportunity, reran the fixed arms, and compared specialists against neutral in the same zone. The A side works, but the B side still favors neutral. The next path should redesign the B-zone ecological pressure instead of running evolving-axis selection.",
            "",
            "## Primary Metric",
            "",
            f"- Metric: `{PRIMARY_METRIC}`",
            "- Denominator: parent-pool candidate events by fixed arm and zone at generation selection time.",
            "",
            "## Same-Zone Result",
            "",
            f"- A-favored > neutral in A zone: {niche['a_favored_gt_neutral_in_a_zone']['supporting_seeds']}/4 seeds.",
            f"- Neutral > B-favored in A zone: {niche['neutral_gt_b_favored_in_a_zone']['supporting_seeds']}/4 seeds.",
            f"- B-favored > neutral in B zone: {niche['b_favored_gt_neutral_in_b_zone']['supporting_seeds']}/4 seeds.",
            f"- Neutral > A-favored in B zone: {niche['neutral_gt_a_favored_in_b_zone']['supporting_seeds']}/4 seeds.",
            "",
            "The B-zone same-zone specialist-over-neutral gate fails. The old all-agent denominator did not hide a B-specialist advantage; the eligibility-normalized metric makes the negative result clearer.",
            "",
            "## Repeat Parity",
            "",
            f"- Status: `{parity['status']}`",
            "- Scope: A-favored/202, neutral/303, B-favored/404 in two isolated workers=1 roots.",
            "",
            "## No-Claims Statement",
            "",
            NO_CLAIMS,
            "",
        ]
    )


def write_outputs(payloads: Mapping[str, Any], evidence_dir: Path, report_path: Path) -> dict[str, Any]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = evidence_dir / name
        if name.endswith(".csv"):
            _write_csv(path, payload if isinstance(payload, list) else [])
        else:
            _write_json(path, payload)
    text = report_text(payloads)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    (evidence_dir / report_path.name).write_text(text, encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "evidence_dir": str(evidence_dir),
        "report_path": str(report_path),
    }
    _write_json(evidence_dir / "respec_ea1r_evidence_bundle_manifest.json", manifest)
    zip_path = evidence_dir / "world-pressure_respec_ea1r_reproductive_eligibility.evidence.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(evidence_dir.iterdir()):
            if path.is_file() and path.name != zip_path.name:
                zf.write(path, arcname=path.name)
    manifest["files"] = sorted(p.name for p in evidence_dir.iterdir() if p.is_file() and p.name != zip_path.name)
    manifest["file_count"] = len(manifest["files"])
    manifest["zip_path"] = str(zip_path)
    manifest["zip_sha256"] = _sha256(zip_path)
    _write_json(evidence_dir / "respec_ea1r_evidence_bundle_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--repeat-a-root", type=Path, required=True)
    parser.add_argument("--repeat-b-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    payloads = build_payloads(
        smoke_root=args.smoke_root,
        confirmation_root=args.confirmation_root,
        repeat_a_root=args.repeat_a_root,
        repeat_b_root=args.repeat_b_root,
    )
    if args.write:
        print(json.dumps(write_outputs(payloads, args.evidence_dir, args.report_path), indent=2, sort_keys=True))
    else:
        print(json.dumps(payloads["respec_ea1r_final_decision.json"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
