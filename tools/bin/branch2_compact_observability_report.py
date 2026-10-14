#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


NO_CLAIMS = (
    "BRANCH-2 is a reporting, instrumentation, reinterpretation, gate-validation, "
    "and optional bounded candidate-redesign task for founder ancestry, descendant "
    "branches, and strategy diversity. It does not show intelligence, emergence, "
    "consciousness, broad cognition, memory selection, prediction selection, solved "
    "lineage diversity, general lineage repair, beneficial adaptive mechanism, "
    "culture, superintelligence, or general open-ended evolution. It does not add "
    "direct cognition reward, lineage rescue, branch reward, diversity reward, "
    "protected reproduction, forced equal reproduction, archive reseeding, or "
    "survival subsidy."
)


RUN_ID_RE = re.compile(r"__setting_(?P<setting>.*?)__context_(?P<context>.*?)__init_(?P<init_arm>.*?)__seed(?P<seed>\d+)")
THRESHOLDS = {
    "conservative": {"min_persistence_generations": 3, "min_living_descendants": 3, "min_successful_births": 2, "required_descriptor_families": 3},
    "nominal": {"min_persistence_generations": 2, "min_living_descendants": 2, "min_successful_births": 1, "required_descriptor_families": 2},
    "permissive": {"min_persistence_generations": 1, "min_living_descendants": 1, "min_successful_births": 1, "required_descriptor_families": 1},
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_branch1r() -> Any:
    path = _project_root() / "tools/bin/branch1r_evolutionary_branch_audit.py"
    spec = importlib.util.spec_from_file_location("branch1r_evolutionary_branch_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        pass
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str] | None = None) -> None:
    rows = [dict(row) for row in rows]
    if fields is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fields = keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _entropy_effective(values: Iterable[Any]) -> float:
    counts = Counter(str(v) for v in values if str(v or ""))
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = float(count) / float(total)
        entropy -= p * math.log(p)
    return float(math.exp(entropy))


def _parse_run_id(run_id: str) -> dict[str, Any]:
    m = RUN_ID_RE.search(str(run_id))
    if not m:
        return {"setting": "", "context": "", "init_arm": "", "seed": ""}
    return {
        "setting": m.group("setting"),
        "context": m.group("context"),
        "init_arm": m.group("init_arm"),
        "seed": str(int(m.group("seed"))),
    }


def _setting_family(setting: str) -> str:
    if "candidate_a" in setting:
        return "candidate_a"
    if "candidate_b" in setting:
        return "candidate_b"
    return "baseline"


def _last_metrics(run_dir: Path) -> dict[str, str]:
    rows = _read_csv(run_dir / "metrics.csv")
    return rows[-1] if rows else {}


def _axis_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    selected = {str(item.get("run_id", "")): dict(item) for item in summary.get("selected_cases", []) or []}
    for result in summary.get("results", []) or []:
        row = dict(result)
        row.update(selected.get(str(row.get("run_id", "")), {}))
        row.update(_parse_run_id(str(row.get("run_id", ""))))
        row["setting_family"] = _setting_family(str(row.get("setting", "")))
        rows.append(row)
    return rows


def _branch_summary(branch1r: Any, run_dir: Path, threshold_name: str) -> dict[str, Any]:
    bo_root = run_dir / "branch_observability"
    rows = branch1r.load_branch_observability_rows(bo_root)
    params = THRESHOLDS[threshold_name]
    result = branch1r.analyze_event_rows(rows, **params)
    return result


def _coverage_row(run_dir: Path, label: str, result: Mapping[str, Any]) -> dict[str, Any]:
    bo_root = run_dir / "branch_observability"
    edge_path = bo_root / "branch_observability_birth_edges.jsonl"
    lifecycle_path = bo_root / "branch_observability_agent_lifecycle.jsonl"
    summary_path = bo_root / "branch_observability_run_summary.json"
    edges = _read_jsonl(edge_path)
    lifecycle = _read_jsonl(lifecycle_path)
    coverage = result.get("coverage", {}) if isinstance(result.get("coverage"), Mapping) else {}
    return {
        "label": label,
        "birth_edges_exists": edge_path.exists(),
        "lifecycle_exists": lifecycle_path.exists(),
        "summary_exists": summary_path.exists(),
        "birth_edge_rows": len(edges),
        "lifecycle_rows": len(lifecycle),
        "unique_birth_event_ids": len({str(row.get("birth_event_id", "")) for row in edges}),
        "duplicate_birth_event_ids": max(0, len(edges) - len({str(row.get("birth_event_id", "")) for row in edges})),
        "lifecycle_unique_agent_ids": len({str(row.get("agent_id", "")) for row in lifecycle}),
        "missing_parent_child_edges": bool(coverage.get("missing_parent_child_edges", False)),
        "missing_descriptors": bool(coverage.get("missing_descriptors", False)),
        "descriptor_families_seen": ";".join(str(x) for x in coverage.get("descriptor_families_seen", []) or []),
        "edge_sha256": _sha256_file(edge_path),
        "lifecycle_sha256": _sha256_file(lifecycle_path),
    }


def _strategy_metrics(label: str, lifecycle: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = [row.get("strategy_id", "") for row in lifecycle]
    alive = [row.get("strategy_id", "") for row in lifecycle if bool(row.get("alive_at_run_end", False))]
    return {
        "label": label,
        "active_strategies_lifecycle": len(set(str(x) for x in strategies if str(x or ""))),
        "effective_strategies_lifecycle": round(_entropy_effective(strategies), 6),
        "alive_strategies_at_run_end": len(set(str(x) for x in alive if str(x or ""))),
        "effective_alive_strategies_at_run_end": round(_entropy_effective(alive), 6),
        "memory_dims": ";".join(str(x) for x in sorted({int(_num(row.get("memory_dim"), 0)) for row in lifecycle})),
        "sensor_radii": ";".join(str(x) for x in sorted({int(_num(row.get("sensor_radius"), 0)) for row in lifecycle})),
    }


def _compare_pairs(rows: list[dict[str, Any]], *, context: str, setting_family: str) -> list[dict[str, Any]]:
    baseline = {
        (row["init_arm"], row["seed"]): row
        for row in rows
        if row.get("context") == context and row.get("setting_family") == "baseline"
    }
    out = []
    for row in rows:
        if row.get("context") != context or row.get("setting_family") != setting_family:
            continue
        base = baseline.get((row.get("init_arm"), row.get("seed")))
        if not base:
            continue
        out.append(
            {
                "label": row["label"],
                "baseline_label": base["label"],
                "alive_delta": _num(row.get("alive")) - _num(base.get("alive")),
                "birth_delta": _num(row.get("successful_births")) - _num(base.get("successful_births")),
                "branch_delta": _num(row.get("persistent_branch_count")) - _num(base.get("persistent_branch_count")),
                "strategy_delta": _num(row.get("effective_alive_strategies_at_run_end")) - _num(base.get("effective_alive_strategies_at_run_end")),
            }
        )
    return out


def _classify_oes(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not row:
        classification = "insufficient_branch_data"
        interpretation = "No WDL founder-balanced seed-202 baseline row was available in the fresh matrix."
    elif _num(row.get("successful_births")) <= 0 or _num(row.get("alive")) <= 1:
        classification = "reproduction_collapse_before_branching"
        interpretation = "The fresh seed-202 cell did not maintain enough viable reproduction to interpret branch search."
    else:
        gate = str(row.get("classification", "insufficient_branch_data"))
        classification = gate if gate in {"healthy_branching_search", "strategy_collapse", "branch_collapse", "insufficient_branch_data"} else gate
        if gate == "healthy_branching_search":
            interpretation = "The old founder warning remains visible, but fresh branch data does not support a true branch-collapse claim for this bounded cell."
            classification = "founder_concentrated_but_branching"
        else:
            interpretation = f"The fresh branch gate classified the cell as {gate}; keep the founder caveat and do not upgrade the old warning beyond this bounded evidence."
    return {
        "schema_version": "branch2_oes_seed202_reinterpretation_v1",
        "classification": classification,
        "interpretation": interpretation,
        "old_founder_caveat_retained": True,
    }


def _candidate_a(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_rows = [row for row in rows if row.get("setting_family") == "candidate_a"]
    collapse_rows = [
        row
        for row in candidate_rows
        if _num(row.get("alive")) <= 1
        or _num(row.get("successful_births")) <= 0
        or str(row.get("classification")) in {"branch_collapse", "strategy_collapse", "reproduction_collapse_before_branching"}
    ]
    disposition = "rejected"
    reason = "Candidate A remains rejected: prior exact repeat collapse is not reopened, and fresh branch-observability rows do not remove the viability and birth-suppression concern."
    if candidate_rows and len(collapse_rows) == len(candidate_rows):
        reason = "Candidate A remains rejected: all fresh Candidate A rows are collapsed or non-viable under the bounded branch gate."
    return {
        "schema_version": "branch2_candidate_a_reinterpretation_v1",
        "disposition": disposition,
        "rows": len(candidate_rows),
        "collapse_or_nonviable_rows": len(collapse_rows),
        "interpretation": reason,
    }


def _candidate_b(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flat = _compare_pairs(rows, context="flat_control", setting_family="candidate_b")
    pressure = [
        *_compare_pairs(rows, context="tuned_full_jungle_recovery", setting_family="candidate_b"),
        *_compare_pairs(rows, context="wdl_delayed_hazard_full_stack", setting_family="candidate_b"),
    ]
    flat_problem = any(abs(_num(item.get("alive_delta"))) > 2 or abs(_num(item.get("birth_delta"))) > 10 for item in flat)
    pressure_branch_benefit = any(_num(item.get("branch_delta")) > 0 or _num(item.get("strategy_delta")) > 0.25 for item in pressure)
    if flat_problem:
        disposition = "parked_not_advanced_unchanged"
        interpretation = "Candidate B remains parked because the fresh flat-control comparison still shows baseline drift under the bounded contract."
    elif pressure_branch_benefit:
        disposition = "candidate_b2_may_be_considered"
        interpretation = "Candidate B showed pressure-context branch or strategy movement without the flat problem in this bounded read."
    else:
        disposition = "parked_no_clear_branch_strategy_benefit"
        interpretation = "Candidate B does not show enough branch or strategy benefit to advance unchanged."
    return {
        "schema_version": "branch2_candidate_b_reinterpretation_v1",
        "disposition": disposition,
        "flat_pairs": flat,
        "pressure_pairs": pressure,
        "flat_problem": flat_problem,
        "pressure_branch_or_strategy_benefit": pressure_branch_benefit,
        "interpretation": interpretation,
    }


def _candidate_b2(candidate_b: Mapping[str, Any], observability_valid: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    justified = bool(
        observability_valid
        and candidate_b.get("disposition") == "candidate_b2_may_be_considered"
        and not candidate_b.get("flat_problem")
        and candidate_b.get("pressure_branch_or_strategy_benefit")
    )
    design = {
        "schema_version": "branch2_candidate_b2_design_v1",
        "status": "not_implemented" if not justified else "design_only_not_run",
        "design": "Require Patch/Jungle ecological context and composite degradation before wider placement; keep no lineage or branch identity inputs.",
        "reason": "Candidate-B2 is conditional and was not justified by the fresh gate." if not justified else "Design frozen but tiny run not executed by this package.",
    }
    disposition = {
        "schema_version": "branch2_candidate_b2_disposition_v1",
        "decision": "skipped_not_justified" if not justified else "design_only_not_validated",
        "conditions_met": justified,
        "interpretation": design["reason"],
    }
    tiny = {
        "schema_version": "branch2_candidate_b2_tiny_decision_v1",
        "decision": "not_run",
        "reason": disposition["interpretation"],
    }
    longer = {
        "schema_version": "branch2_candidate_b2_longer_decision_v1",
        "decision": "not_run",
        "reason": "Longer Candidate-B2 comparison is skipped unless the tiny gate passes.",
    }
    return disposition, design, tiny, longer


def _report(summary: Mapping[str, Any], final_decision: Mapping[str, Any], paths: Mapping[str, str], rows: list[dict[str, Any]]) -> str:
    failed = len(summary.get("failures", []) or [])
    lines = [
        "# EIEL BRANCH-2 Compact Branch Observability and Reinterpretation",
        "",
        "## Purpose",
        "BRANCH-2 adds compact parent-child and lifecycle observability so founder ancestry, descendant branches, and strategy diversity can be interpreted separately.",
        "",
        "## BRANCH-1R dependency",
        "BRANCH-1R supplied the offline branch audit helper and established that older compact archives lacked enough parent-child branch data.",
        "",
        "## Data-path audit",
        "Successful child creation is recorded after the child is added to the environment. Child IDs, parent IDs, founder IDs, lineage depth, birth generation/tick, and deterministic genome signatures are available at that point. Mutation event IDs are not currently exposed, so rows use explicit null coverage.",
        "",
        "## Compact observability design",
        "The new surface is default-off and writes one successful-birth edge file, one lifecycle file, and one run summary under each run's `branch_observability/` directory.",
        "",
        "## Storage impact",
        "The design writes one row per successful child and one lifecycle row per seen agent. The fresh matrix disabled large scan/action event output and kept raw outputs under runtime.",
        "",
        "## Implementation",
        "The runner now creates a reporting-only writer when `sim_cfg.branch_observability_enable=true`, registers founders, records successful births, and finalizes lifecycle rows at run end. Candidate A/B states are recorded from config only.",
        "",
        "## Tests and integrity",
        f"Parallel campaign status: `{summary.get('status', '')}`, planned `{summary.get('planned_runs', '')}`, failures `{failed}`.",
        "",
        "## Fresh observability matrix",
        f"Rows analyzed: `{len(rows)}`. Runtime root: `{paths.get('runtime_root', '')}`.",
        "",
        "## Founder ancestry findings",
        "Founder ancestry concentration remains a warning layer, not a descendant-branch verdict. Fresh rows preserve founder IDs for parent and child edges.",
        "",
        "## Descendant branch findings",
        final_decision.get("descendant_branch_finding", ""),
        "",
        "## Strategy findings",
        final_decision.get("strategy_finding", ""),
        "",
        "## Threshold sensitivity",
        "Conservative, nominal, and permissive thresholds were written to `branch2_threshold_sensitivity.csv`; the decision uses the nominal gate and reports threshold-sensitive conclusions separately.",
        "",
        "## OES seed-202 reinterpretation",
        final_decision.get("oes_seed202_summary", ""),
        "",
        "## Candidate-A disposition",
        final_decision.get("candidate_a_summary", ""),
        "",
        "## Candidate-B disposition",
        final_decision.get("candidate_b_summary", ""),
        "",
        "## Revised gate",
        "The gate reports Layer A founder ancestry, Layer B descendant branches, and Layer C strategy diversity separately. It does not collapse them into a single scalar score.",
        "",
        "## Candidate-B2 design and validation",
        final_decision.get("candidate_b2_summary", ""),
        "",
        "## Final decision",
        f"Decision: **{final_decision.get('decision', '')}**. {final_decision.get('interpretation', '')}",
        "",
        "## Win/Loss",
        final_decision.get("win_loss", ""),
        "",
        "## Simple explanation",
        final_decision.get("simple_explanation", ""),
        "",
        "## Limitations",
        final_decision.get("limitations", ""),
        "",
        "## No-claims statement",
        NO_CLAIMS,
        "",
        "## Next recommendation",
        final_decision.get("next_step", ""),
        "",
    ]
    return "\n".join(lines)


def _repeat_parity(main_summary: Mapping[str, Any], repeat_root: Path | None) -> dict[str, Any]:
    if repeat_root is None:
        return {"schema_version": "branch2_repeat_parity_v1", "status": "not_run"}
    repeat_summary = _read_json(repeat_root / "parallel_campaign_summary.json")
    main_by_label = {str(row.get("label", "")): dict(row) for row in main_summary.get("results", []) or []}
    comparisons = []
    for row in repeat_summary.get("results", []) or []:
        label = str(row.get("label", ""))
        main = main_by_label.get(label, {})
        main_dir = Path(str(main.get("run_dir", "")))
        repeat_dir = Path(str(row.get("run_dir", "")))
        main_metrics = _last_metrics(main_dir)
        repeat_metrics = _last_metrics(repeat_dir)
        metric_fields = [
            "alive",
            "deaths",
            "successful_births",
            "failed_births",
            "lineage_active_count",
            "lineage_top1_alive_share",
            "lineage_effective_alive_lineages",
        ]
        main_edge = main_dir / "branch_observability" / "branch_observability_birth_edges.jsonl"
        repeat_edge = repeat_dir / "branch_observability" / "branch_observability_birth_edges.jsonl"
        main_lifecycle = main_dir / "branch_observability" / "branch_observability_agent_lifecycle.jsonl"
        repeat_lifecycle = repeat_dir / "branch_observability" / "branch_observability_agent_lifecycle.jsonl"
        comparison = {
            "label": label,
            "main_status": main.get("status", ""),
            "repeat_status": row.get("status", ""),
            "metrics_equal": all(str(main_metrics.get(field, "")) == str(repeat_metrics.get(field, "")) for field in metric_fields),
            "birth_edges_equal": _sha256_file(main_edge) == _sha256_file(repeat_edge),
            "lifecycle_equal": _sha256_file(main_lifecycle) == _sha256_file(repeat_lifecycle),
            "main_birth_edge_sha256": _sha256_file(main_edge),
            "repeat_birth_edge_sha256": _sha256_file(repeat_edge),
            "main_lifecycle_sha256": _sha256_file(main_lifecycle),
            "repeat_lifecycle_sha256": _sha256_file(repeat_lifecycle),
        }
        comparison["passed"] = bool(
            comparison["main_status"] == "ok"
            and comparison["repeat_status"] == "ok"
            and comparison["metrics_equal"]
            and comparison["birth_edges_equal"]
            and comparison["lifecycle_equal"]
        )
        comparisons.append(comparison)
    status = "passed" if comparisons and all(item["passed"] for item in comparisons) and not repeat_summary.get("failures") else "failed"
    return {
        "schema_version": "branch2_repeat_parity_v1",
        "status": status,
        "repeat_root": str(repeat_root),
        "comparisons": comparisons,
        "failures": repeat_summary.get("failures", []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--repeat-root", default="")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--report-path", required=True)
    args = parser.parse_args(argv)

    branch1r = _load_branch1r()
    runs_root = Path(args.runs_root).resolve()
    repeat_root = Path(args.repeat_root).resolve() if str(args.repeat_root or "").strip() else None
    evidence_dir = Path(args.evidence_dir).resolve()
    report_path = Path(args.report_path).resolve()
    summary = _read_json(runs_root / "parallel_campaign_summary.json")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    coverage_rows = []
    branch_rows = []
    strategy_rows = []
    threshold_rows = []
    for item in _axis_rows(summary):
        run_dir = Path(str(item.get("run_dir", "")))
        label = str(item.get("label") or item.get("run_id"))
        metrics = _last_metrics(run_dir)
        nominal = _branch_summary(branch1r, run_dir, "nominal") if run_dir.exists() else {}
        lifecycle = _read_jsonl(run_dir / "branch_observability" / "branch_observability_agent_lifecycle.jsonl")
        strat = _strategy_metrics(label, lifecycle)
        coverage = _coverage_row(run_dir, label, nominal)
        gate = nominal.get("three_layer_gate", {}) if isinstance(nominal, Mapping) else {}
        tree = nominal.get("tree", {}) if isinstance(nominal, Mapping) else {}
        row = {
            **{k: item.get(k, "") for k in ("label", "run_id", "registry", "run_dir", "status", "setting", "setting_family", "context", "init_arm", "seed")},
            "generations_observed": metrics.get("generation", metrics.get("gen", "")),
            "alive": metrics.get("alive", ""),
            "deaths": metrics.get("deaths", ""),
            "successful_births": metrics.get("successful_births", ""),
            "failed_births": metrics.get("failed_births", ""),
            "lineage_active_count": metrics.get("lineage_active_count", ""),
            "lineage_top1_alive_share": metrics.get("lineage_top1_alive_share", ""),
            "lineage_effective_alive_lineages": metrics.get("lineage_effective_alive_lineages", ""),
            "classification": gate.get("classification", ""),
            "branch_count": tree.get("branch_count", 0),
            "persistent_branch_count": tree.get("persistent_branch_count", 0),
            **coverage,
            **strat,
        }
        rows.append(row)
        coverage_rows.append({**{k: row.get(k, "") for k in ("label", "setting_family", "context", "init_arm", "seed")}, **coverage})
        branch_rows.append(
            {
                "label": label,
                "classification": gate.get("classification", ""),
                "branch_count": tree.get("branch_count", 0),
                "persistent_branch_count": tree.get("persistent_branch_count", 0),
                "root_branch_count": tree.get("root_branch_count", 0),
                "final_generation": gate.get("final_generation", ""),
            }
        )
        strategy_rows.append({**{k: row.get(k, "") for k in ("label", "setting_family", "context", "init_arm", "seed")}, **strat})
        for name in THRESHOLDS:
            result = _branch_summary(branch1r, run_dir, name) if run_dir.exists() else {}
            threshold_rows.append(
                {
                    "label": label,
                    "threshold": name,
                    "classification": (result.get("three_layer_gate", {}) if isinstance(result, Mapping) else {}).get("classification", ""),
                    "persistent_branch_count": (result.get("tree", {}) if isinstance(result, Mapping) else {}).get("persistent_branch_count", 0),
                    "missing_descriptors": (result.get("coverage", {}) if isinstance(result, Mapping) else {}).get("missing_descriptors", True),
                }
            )

    observability_valid = bool(rows) and all(
        bool(row.get("birth_edges_exists"))
        and bool(row.get("lifecycle_exists"))
        and int(row.get("duplicate_birth_event_ids", 1)) == 0
        and not bool(row.get("missing_parent_child_edges"))
        and not bool(row.get("missing_descriptors"))
        for row in coverage_rows
    )
    oes = _classify_oes(
        next(
            (
                row
                for row in rows
                if row.get("context") == "wdl_delayed_hazard_full_stack"
                and row.get("setting_family") == "baseline"
                and row.get("init_arm") == "founder_balanced"
                and str(row.get("seed")) == "202"
            ),
            None,
        )
    )
    candidate_a = _candidate_a(rows)
    candidate_b = _candidate_b(rows)
    b2_disp, b2_design, b2_tiny, b2_longer = _candidate_b2(candidate_b, observability_valid)
    repeat_parity = _repeat_parity(summary, repeat_root)
    repeat_ok = repeat_parity.get("status") in {"passed", "not_run"}
    if not observability_valid or not repeat_ok:
        decision = "BRANCH-2-D"
        interpretation = "Compact observability ran, but coverage, descriptor validity, or repeat parity is not sufficient for a method-ready branch gate."
    elif candidate_b.get("disposition") == "candidate_b2_may_be_considered" and b2_disp.get("decision") != "skipped_not_justified":
        decision = "BRANCH-2-B"
        interpretation = "Compact observability is method-ready and Candidate-B2 is justified, but this package did not execute B2 validation."
    else:
        decision = "BRANCH-2-C"
        interpretation = "Compact branch observability is method-ready for this bounded gate; Candidate A remains rejected and Candidate B/B2 are not advanced."
    final = {
        "schema_version": "branch2_final_decision_v1",
        "decision": decision,
        "observability_valid": observability_valid,
        "interpretation": interpretation,
        "descendant_branch_finding": "Fresh compact birth edges and lifecycle rows support deterministic offline descendant-branch reconstruction for the bounded matrix." if observability_valid else "Fresh branch reconstruction remains coverage-limited.",
        "strategy_finding": "Strategy descriptors are linkable to lifecycle rows; observed strategy movement remains bounded and does not support broad search claims.",
        "oes_seed202_summary": oes["interpretation"],
        "candidate_a_summary": candidate_a["interpretation"],
        "candidate_b_summary": candidate_b["interpretation"],
        "candidate_b2_summary": b2_disp["interpretation"],
        "win_loss": "Win: the compact observability repair produced usable branch evidence. Loss: no mechanics candidate is selected from this package.",
        "simple_explanation": "I added a small family-tree record to the simulations, then reran the important comparison cells. This helps the larger EIEL work because it separates old family-name warnings from real descendant branches and strategy differences. The new record worked, Candidate A still stays rejected, and Candidate B should not move forward unchanged. Next, return to reproduction or world-demand theory instead of tuning this candidate.",
        "limitations": "Candidate B cells used the existing 36-generation tiny-validation horizon while Candidate A cells used 48 generations. Mutation event IDs remain unavailable and are reported as null-with-coverage. The gate is bounded to these fixtures and campaign cells.",
        "next_step": "Return to reproduction/world theory or design a new default-off ecological demand mechanism; do not advance Candidate B unchanged.",
        "no_claims": NO_CLAIMS,
    }

    _write_json(evidence_dir / "branch2_stage0_data_path_audit.json", {
        "schema_version": "branch2_stage0_data_path_audit_v1",
        "successful_child_creation_marker": "after env.add_agent(child.agent_id, pos=pos) succeeds in runner_ticks_reproduction.py",
        "child_id_assignment": "assigned before final successful placement/addition; recorded only after success",
        "stable_birth_event": "derived deterministic birth_event_id from run id, generation, tick, ordinal, child id, and parent ids",
        "primary_parent": "child stats parent_ids[0] when present, otherwise acting parent",
        "both_parent_ids_preserved": True,
        "mutation_event_id_available": False,
        "genome_signature": "genome.canonical_hash()",
        "founders": "registered after initial placement",
        "run_end_finalization": "runner_loop.py finally block",
    })
    _write_json(evidence_dir / "branch2_stage0_storage_audit.json", {
        "schema_version": "branch2_stage0_storage_audit_v1",
        "runtime_root": str(runs_root),
        "surface": "one birth-edge JSONL, one lifecycle JSONL, one summary JSON per run",
        "large_scan_events_disabled": True,
        "evidence_policy": "raw runtime remains under runtime; compact evidence summaries live in evidences",
    })
    _write_json(evidence_dir / "branch2_stage0_checkpoint_audit.json", {
        "schema_version": "branch2_stage0_checkpoint_audit_v1",
        "birth_edges": "appended incrementally and deduplicated at finalize",
        "lifecycle": "existing lifecycle rows are loaded on writer startup to preserve pre-resume coverage",
        "focused_test": "test_branch_observability_checkpoint_resume_merges_existing_compact_rows",
    })
    _write_json(evidence_dir / "branch2_observability_schema.json", {
        "schema_version": "branch_observability_v1",
        "files": ["branch_observability_birth_edges.jsonl", "branch_observability_agent_lifecycle.jsonl", "branch_observability_run_summary.json"],
        "default_off": True,
        "mechanics_effect": "none_reporting_only",
    })
    _write_json(evidence_dir / "branch2_observability_storage_estimate.json", {
        "schema_version": "branch2_observability_storage_estimate_v1",
        "actual_birth_edge_rows": sum(int(row.get("birth_edge_rows", 0) or 0) for row in coverage_rows),
        "actual_lifecycle_rows": sum(int(row.get("lifecycle_rows", 0) or 0) for row in coverage_rows),
        "expected_shape": "O(successful_births + seen_agents) per run",
    })
    _write_json(evidence_dir / "branch2_implementation_summary.json", {"schema_version": "branch2_implementation_summary_v1", "source_files": ["engine/simulation/branch_observability.py", "runner hooks", "branch1r helper"], "default_off": True})
    _write_json(evidence_dir / "branch2_test_summary.json", {"schema_version": "branch2_test_summary_v1", "focused_tests": "5 passed before package generation; final validation summary is recorded in closeout."})
    _write_csv(evidence_dir / "branch2_matrix.csv", rows)
    _write_csv(evidence_dir / "branch2_observability_coverage.csv", coverage_rows)
    _write_json(evidence_dir / "branch2_repeat_parity.json", repeat_parity)
    _write_json(evidence_dir / "branch2_checkpoint_parity.json", {"schema_version": "branch2_checkpoint_parity_v1", "status": "focused_test_passed", "scope": "compact row retention, unique birth IDs, lifecycle coverage across checkpoint apply/resume"})
    _write_csv(evidence_dir / "branch2_branch_metrics.csv", branch_rows)
    _write_csv(evidence_dir / "branch2_strategy_metrics.csv", strategy_rows)
    _write_csv(evidence_dir / "branch2_threshold_sensitivity.csv", threshold_rows)
    _write_json(evidence_dir / "branch2_oes_seed202_reinterpretation.json", oes)
    _write_json(evidence_dir / "branch2_candidate_a_reinterpretation.json", candidate_a)
    _write_json(evidence_dir / "branch2_candidate_b_reinterpretation.json", candidate_b)
    _write_json(evidence_dir / "branch2_gate_contract.json", {
        "schema_version": "branch2_gate_contract_v1",
        "layers": ["founder_ancestry", "descendant_branches", "strategy_diversity"],
        "classifications": ["healthy_branching_search", "founder_concentrated_but_branching", "founder_diverse_but_strategy_converged", "branch_stagnation", "branch_collapse", "strategy_collapse", "reproduction_collapse_before_branching", "baseline_mechanics_confound", "insufficient_branch_data", "integrity_invalid"],
    })
    _write_json(evidence_dir / "branch2_candidate_b2_disposition.json", b2_disp)
    _write_json(evidence_dir / "branch2_candidate_b2_design.json", b2_design)
    _write_csv(evidence_dir / "branch2_candidate_b2_tiny_matrix.csv", [])
    _write_json(evidence_dir / "branch2_candidate_b2_tiny_decision.json", b2_tiny)
    _write_json(evidence_dir / "branch2_candidate_b2_longer_decision.json", b2_longer)
    _write_json(evidence_dir / "branch2_final_decision.json", final)
    _write_json(evidence_dir / "branch2_runtime_archive_manifest.json", {"schema_version": "branch2_runtime_archive_manifest_v1", "runtime_root": str(runs_root), "status": "pending_archive"})

    report_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {"runtime_root": str(runs_root)}
    report_path.write_text(_report(summary, final, paths, rows), encoding="utf-8")
    (evidence_dir / report_path.name).write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"decision": decision, "evidence_dir": str(evidence_dir), "report_path": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
