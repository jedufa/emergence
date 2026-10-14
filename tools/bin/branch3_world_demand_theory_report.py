#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


NO_CLAIMS = (
    "BRANCH-3 is a bounded theory, reporting, and design-gate package for "
    "descendant branching, strategy descriptors, and lineage-blind world demand. "
    "It does not show intelligence, emergence, consciousness, broad cognition, "
    "memory selection, prediction selection, solved lineage diversity, general "
    "lineage repair, beneficial adaptive mechanism, culture, superintelligence, "
    "or general open-ended evolution. It does not add direct cognition reward, "
    "lineage rescue, branch reward, diversity reward, protected reproduction, "
    "forced equal reproduction, archive reseeding, or survival subsidy."
)


COMPRESSION_PROFILES: dict[str, dict[str, tuple[float, ...]]] = {
    "coarse": {
        "lifetime": (0.0, 64.0, 512.0),
        "energy_rate": (0.0, 0.01, 0.05),
        "injury": (0.0, 0.05),
    },
    "nominal": {
        "lifetime": (0.0, 32.0, 128.0, 512.0),
        "energy_rate": (0.0, 0.005, 0.02, 0.10),
        "injury": (0.0, 0.02, 0.10),
    },
    "fine": {
        "lifetime": (0.0, 16.0, 64.0, 128.0, 256.0, 512.0),
        "energy_rate": (0.0, 0.0025, 0.01, 0.025, 0.05, 0.10),
        "injury": (0.0, 0.01, 0.025, 0.05, 0.10, 0.20),
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str] | None = None) -> None:
    rows = [dict(row) for row in rows]
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _bin(value: float, thresholds: Iterable[float]) -> int:
    out = 0
    for threshold in thresholds:
        if value > float(threshold):
            out += 1
    return out


def _effective_count(values: Iterable[Any]) -> float:
    counts = Counter(str(v) for v in values if str(v or ""))
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = float(count) / float(total)
        entropy -= p * math.log(p)
    return float(math.exp(entropy))


def memory_bin(memory_dim: Any) -> str:
    dim = int(_num(memory_dim, 0))
    if dim <= 0:
        return "memory_none"
    if dim <= 8:
        return "memory_low"
    if dim <= 32:
        return "memory_medium"
    return "memory_high"


def reproduction_bin(successful_children: Any) -> str:
    children = int(_num(successful_children, 0))
    if children <= 0:
        return "children_0"
    if children == 1:
        return "children_1"
    if children == 2:
        return "children_2"
    return "children_3plus"


def compressed_strategy_signature(row: Mapping[str, Any], *, profile: str = "nominal") -> tuple[str, ...]:
    thresholds = COMPRESSION_PROFILES[profile]
    lifetime = max(0.0, _num(row.get("lifetime_ticks"), 0.0))
    energy_collected = _num((row.get("food_use_summary") or {}).get("energy_collected") if isinstance(row.get("food_use_summary"), Mapping) else 0.0)
    injury = _num((row.get("damage_summary") or {}).get("injury") if isinstance(row.get("damage_summary"), Mapping) else 0.0)
    energy_rate = energy_collected / max(1.0, lifetime)
    patch_summary = row.get("patch_use_summary")
    patch_coverage = "patch_missing"
    if isinstance(patch_summary, Mapping) and str(patch_summary.get("coverage", "")) != "not_available":
        patch_coverage = "patch_available"
    return (
        memory_bin(row.get("memory_dim")),
        f"sensor_{int(_num(row.get('sensor_radius'), 0))}",
        reproduction_bin(row.get("successful_children_count", row.get("successful_births", 0))),
        f"lifetime_bin_{_bin(lifetime, thresholds['lifetime'])}",
        f"energy_rate_bin_{_bin(energy_rate, thresholds['energy_rate'])}",
        f"injury_bin_{_bin(injury, thresholds['injury'])}",
        patch_coverage,
        "alive" if _bool(row.get("alive", row.get("alive_at_run_end", False))) else "dead",
    )


def strategy_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "schema_version": "branch3_strategy_validation_v1",
            "status": "failed_closed",
            "reason": "No lifecycle rows were available for validation.",
            "strategy_layer_scientifically_discriminating": False,
        }
    raw_ids = [str(row.get("strategy_id", "")) for row in rows if str(row.get("strategy_id", ""))]
    alive = [row for row in rows if _bool(row.get("alive", row.get("alive_at_run_end", False)))]
    alive_raw_ids = [str(row.get("strategy_id", "")) for row in alive if str(row.get("strategy_id", ""))]
    missing_patch = sum(
        1
        for row in rows
        if not isinstance(row.get("patch_use_summary"), Mapping)
        or str((row.get("patch_use_summary") or {}).get("coverage", "")) == "not_available"
    )
    raw_uniqueness_ratio = float(len(set(raw_ids)) / max(1, len(raw_ids)))
    alive_uniqueness_ratio = float(len(set(alive_raw_ids)) / max(1, len(alive_raw_ids)))
    compressed_nominal = [compressed_strategy_signature(row, profile="nominal") for row in rows]
    blocker_reasons = []
    if raw_uniqueness_ratio >= 0.75:
        blocker_reasons.append("raw strategy ids are near per-agent labels")
    if alive and alive_uniqueness_ratio >= 0.75:
        blocker_reasons.append("alive strategy ids are near per-agent labels")
    if missing_patch / max(1, len(rows)) > 0.50:
        blocker_reasons.append("patch-use and niche exposure coverage is missing")
    blocker_reasons.append("available lifecycle rows mix capacity and outcome fields without branch-persistent ecological exposure")
    return {
        "schema_version": "branch3_strategy_validation_v1",
        "status": "failed_closed",
        "strategy_layer_scientifically_discriminating": False,
        "reporting_only_compression_attempted": True,
        "reporting_only_compression_reduces_noise": len(set(compressed_nominal)) < len(set(raw_ids)),
        "raw_strategy_ids": len(set(raw_ids)),
        "raw_lifecycle_rows": len(rows),
        "raw_uniqueness_ratio": round(raw_uniqueness_ratio, 6),
        "alive_rows": len(alive),
        "alive_raw_strategy_ids": len(set(alive_raw_ids)),
        "alive_uniqueness_ratio": round(alive_uniqueness_ratio, 6),
        "nominal_compressed_strategy_ids": len(set(compressed_nominal)),
        "nominal_effective_compressed_strategy_ids": round(_effective_count(compressed_nominal), 6),
        "missing_patch_use_rows": missing_patch,
        "blocker_reasons": blocker_reasons,
        "conclusion": (
            "A deterministic compression can reduce individual noise, but current rows do not expose enough "
            "interpretable ecological behavior to select or validate a mechanics candidate."
        ),
    }


def _label_from_zip_member(name: str) -> str:
    parts = Path(name).parts
    for part in parts:
        if part.startswith("000") and "_" in part:
            return "_".join(part.split("_")[1:])
    return ""


def load_lifecycle_rows_from_archives(archives: Iterable[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    archive_summaries = []
    for archive in archives:
        path = Path(str(archive))
        summary = {"archive": str(path), "exists": path.exists(), "lifecycle_files": 0, "rows": 0}
        if not path.exists():
            archive_summaries.append(summary)
            continue
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith("branch_observability/branch_observability_agent_lifecycle.jsonl"):
                    continue
                summary["lifecycle_files"] += 1
                label = _label_from_zip_member(name)
                with zf.open(name) as handle:
                    for raw in handle:
                        if not raw.strip():
                            continue
                        row = json.loads(raw.decode("utf-8"))
                        if isinstance(row, dict):
                            row["source_archive"] = str(path)
                            row["source_label"] = label
                            rows.append(row)
                            summary["rows"] += 1
        archive_summaries.append(summary)
    return rows, {"archives": archive_summaries, "total_rows": len(rows)}


def compression_sensitivity(lifecycle_rows: list[dict[str, Any]], matrix_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in lifecycle_rows:
        by_label[str(row.get("source_label", ""))].append(row)
    output = []
    labels = [str(row.get("label", "")) for row in matrix_rows] or sorted(by_label)
    for label in labels:
        rows = by_label.get(label, [])
        raw = [str(row.get("strategy_id", "")) for row in rows if str(row.get("strategy_id", ""))]
        alive = [row for row in rows if _bool(row.get("alive", row.get("alive_at_run_end", False)))]
        for profile in ("coarse", "nominal", "fine"):
            compressed = [compressed_strategy_signature(row, profile=profile) for row in rows]
            alive_compressed = [compressed_strategy_signature(row, profile=profile) for row in alive]
            output.append(
                {
                    "label": label,
                    "profile": profile,
                    "lifecycle_rows": len(rows),
                    "raw_strategy_ids": len(set(raw)),
                    "raw_uniqueness_ratio": round(len(set(raw)) / max(1, len(raw)), 6),
                    "alive_rows": len(alive),
                    "alive_raw_strategy_ids": len({str(row.get("strategy_id", "")) for row in alive if str(row.get("strategy_id", ""))}),
                    "compressed_strategy_ids": len(set(compressed)),
                    "effective_compressed_strategy_ids": round(_effective_count(compressed), 6),
                    "alive_compressed_strategy_ids": len(set(alive_compressed)),
                    "effective_alive_compressed_strategy_ids": round(_effective_count(alive_compressed), 6),
                    "patch_use_coverage": "missing" if rows else "not_loaded",
                    "scientific_use": "noise_audit_only",
                }
            )
    return output


def _threshold_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_threshold: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_threshold[str(row.get("threshold", ""))].append(row)
    out = {}
    for threshold, items in sorted(by_threshold.items()):
        out[threshold] = {
            "rows": len(items),
            "classifications": dict(Counter(str(item.get("classification", "")) for item in items)),
            "persistent_branch_count_min": min((_num(item.get("persistent_branch_count")) for item in items), default=0.0),
            "persistent_branch_count_max": max((_num(item.get("persistent_branch_count")) for item in items), default=0.0),
        }
    return out


def _matrix_strategy_uniqueness(rows: list[dict[str, str]]) -> dict[str, Any]:
    ratios = []
    alive_ratios = []
    exact_alive_unique_rows = 0
    for row in rows:
        lifecycle = _num(row.get("lifecycle_rows"), 0.0)
        active = _num(row.get("active_strategies_lifecycle"), 0.0)
        alive = _num(row.get("alive"), 0.0)
        alive_strategies = _num(row.get("alive_strategies_at_run_end"), 0.0)
        if lifecycle > 0:
            ratios.append(active / lifecycle)
        if alive > 0:
            alive_ratio = alive_strategies / alive
            alive_ratios.append(alive_ratio)
            if abs(alive_strategies - alive) < 1e-9:
                exact_alive_unique_rows += 1
    return {
        "rows": len(rows),
        "max_active_strategy_per_lifecycle_ratio": round(max(ratios, default=0.0), 6),
        "min_active_strategy_per_lifecycle_ratio": round(min(ratios, default=0.0), 6),
        "max_alive_strategy_per_alive_agent_ratio": round(max(alive_ratios, default=0.0), 6),
        "min_alive_strategy_per_alive_agent_ratio": round(min(alive_ratios, default=0.0), 6),
        "rows_with_alive_agents_all_unique": exact_alive_unique_rows,
        "treats_nearly_every_agent_unique": bool(max(ratios, default=0.0) >= 0.90 or exact_alive_unique_rows > 0),
    }


def _candidate_family_rows() -> list[dict[str, Any]]:
    return [
        {
            "family": "A_patch_derived_ecological_niche_mosaic",
            "score": 86,
            "causal_clarity": 8,
            "lineage_blindness": 10,
            "strategy_incompatibility": 9,
            "likely_branch_persistence": 8,
            "flat_control_neutrality": 8,
            "observability": 9,
            "implementation_scope": 7,
            "determinism_risk": 8,
            "performance_cost": 8,
            "overlap_with_existing_code": 10,
            "prior_evidence_support": 8,
            "population_size_risk": 5,
            "globally_best_strategy_risk": 5,
            "interpretation": "Strongest design direction, especially as a Path2/ROZ-style patch-derived zone scaffold, but not ready for mechanics until strategy reporting is repaired.",
        },
        {
            "family": "C_temporal_demand_heterogeneity",
            "score": 74,
            "causal_clarity": 8,
            "lineage_blindness": 10,
            "strategy_incompatibility": 8,
            "likely_branch_persistence": 6,
            "flat_control_neutrality": 8,
            "observability": 7,
            "implementation_scope": 5,
            "determinism_risk": 7,
            "performance_cost": 8,
            "overlap_with_existing_code": 8,
            "prior_evidence_support": 7,
            "population_size_risk": 6,
            "globally_best_strategy_risk": 5,
            "interpretation": "Promising through WDL/DRH0-style delayed demand, but prior memory work shows behavior coupling must be explicit and controls strict.",
        },
        {
            "family": "D_predator_resource_trade_off",
            "score": 70,
            "causal_clarity": 7,
            "lineage_blindness": 10,
            "strategy_incompatibility": 8,
            "likely_branch_persistence": 7,
            "flat_control_neutrality": 7,
            "observability": 8,
            "implementation_scope": 6,
            "determinism_risk": 7,
            "performance_cost": 7,
            "overlap_with_existing_code": 8,
            "prior_evidence_support": 6,
            "population_size_risk": 5,
            "globally_best_strategy_risk": 5,
            "interpretation": "Good demand shape, but easily confounded by survival/carrying-capacity shifts and one dominant cautious strategy.",
        },
        {
            "family": "B_resource_specialization_with_handling_tradeoffs",
            "score": 66,
            "causal_clarity": 8,
            "lineage_blindness": 10,
            "strategy_incompatibility": 9,
            "likely_branch_persistence": 7,
            "flat_control_neutrality": 7,
            "observability": 5,
            "implementation_scope": 3,
            "determinism_risk": 5,
            "performance_cost": 6,
            "overlap_with_existing_code": 4,
            "prior_evidence_support": 4,
            "population_size_risk": 5,
            "globally_best_strategy_risk": 6,
            "interpretation": "Scientifically attractive, but too large for the current bounded slice because resource classes and handling costs need new mechanics and reporting.",
        },
        {
            "family": "E_local_niche_construction_with_costs",
            "score": 54,
            "causal_clarity": 6,
            "lineage_blindness": 10,
            "strategy_incompatibility": 7,
            "likely_branch_persistence": 5,
            "flat_control_neutrality": 5,
            "observability": 5,
            "implementation_scope": 3,
            "determinism_risk": 4,
            "performance_cost": 4,
            "overlap_with_existing_code": 5,
            "prior_evidence_support": 3,
            "population_size_risk": 4,
            "globally_best_strategy_risk": 4,
            "interpretation": "Could create roles, but past paths are fragile and it risks hidden subsidies or state explosion.",
        },
        {
            "family": "F_reproduction_timing_or_life_history_tradeoff",
            "score": 49,
            "causal_clarity": 5,
            "lineage_blindness": 10,
            "strategy_incompatibility": 6,
            "likely_branch_persistence": 5,
            "flat_control_neutrality": 4,
            "observability": 6,
            "implementation_scope": 6,
            "determinism_risk": 7,
            "performance_cost": 8,
            "overlap_with_existing_code": 7,
            "prior_evidence_support": 3,
            "population_size_risk": 2,
            "globally_best_strategy_risk": 3,
            "interpretation": "Highest risk of repeating the reproduction-supply mistake; not selected.",
        },
    ]


def _world_inventory_rows() -> list[dict[str, Any]]:
    return [
        {
            "feature": "flat_control",
            "demand": "none or minimal spatial demand",
            "answer_trait_or_behavior": "generic survival and reproduction",
            "cost": "none",
            "coexisting_incompatible_answers": "no",
            "spatial_or_temporal": "neither",
            "observable_enough": "yes",
            "acts_before_founder_concentration": "no",
            "branch_metrics_detect": "baseline only",
            "existing_but_weak": "not_applicable",
            "carrying_capacity_risk": "high if changed",
        },
        {
            "feature": "Patch World food/cover/shelter/waste/density/movement_cost/scent fields",
            "demand": "spatially different resource and safety conditions",
            "answer_trait_or_behavior": "movement, local occupancy, exploitation versus shelter use",
            "cost": "possible movement and opportunity costs",
            "coexisting_incompatible_answers": "yes if zones are balanced as trade-offs",
            "spatial_or_temporal": "spatial",
            "observable_enough": "partly",
            "acts_before_founder_concentration": "yes if active at initialization",
            "branch_metrics_detect": "yes after reporting repair",
            "existing_but_weak": "exists but not currently exposed in branch lifecycle strategy rows",
            "carrying_capacity_risk": "medium",
        },
        {
            "feature": "tuned Jungle recovery and predator surfaces",
            "demand": "risk versus recovery/safety trade-off",
            "answer_trait_or_behavior": "risk exposure, shelter/cover use, movement",
            "cost": "damage and lost resource opportunity",
            "coexisting_incompatible_answers": "possible",
            "spatial_or_temporal": "spatial and episodic",
            "observable_enough": "partly",
            "acts_before_founder_concentration": "possibly",
            "branch_metrics_detect": "yes after reporting repair",
            "existing_but_weak": "exists, but can collapse into survival/carrying-capacity effects",
            "carrying_capacity_risk": "medium_high",
        },
        {
            "feature": "WDL delayed hazard",
            "demand": "cue-hidden-consequence timing",
            "answer_trait_or_behavior": "pre-consequence movement and avoidance",
            "cost": "memory/sensing capacity and opportunity cost only if world exposes a choice",
            "coexisting_incompatible_answers": "limited without two target/action semantics",
            "spatial_or_temporal": "temporal",
            "observable_enough": "partly",
            "acts_before_founder_concentration": "depends on schedule",
            "branch_metrics_detect": "yes after reporting repair",
            "existing_but_weak": "existing surfaces are stronger as memory fixtures than branch strategy mechanics",
            "carrying_capacity_risk": "medium",
        },
        {
            "feature": "Path2 ROZ patch-derived zones",
            "demand": "reproduction/opportunity events intersect patch-derived ecological conditions",
            "answer_trait_or_behavior": "placement, movement, local niche occupancy",
            "cost": "none yet in reporting-only design",
            "coexisting_incompatible_answers": "design supports future trade-offs but mechanics not active",
            "spatial_or_temporal": "spatial, optionally scheduled",
            "observable_enough": "strong for reporting",
            "acts_before_founder_concentration": "can, if enabled at run start",
            "branch_metrics_detect": "yes after strategy reporting repair",
            "existing_but_weak": "design exists, implementation/reporting integration is the right next slice",
            "carrying_capacity_risk": "low while reporting-only",
        },
        {
            "feature": "resource depletion and regrowth maps",
            "demand": "local exploitation versus movement/recovery",
            "answer_trait_or_behavior": "movement and foraging timing",
            "cost": "movement and depleted local food",
            "coexisting_incompatible_answers": "possible",
            "spatial_or_temporal": "both",
            "observable_enough": "partly",
            "acts_before_founder_concentration": "yes",
            "branch_metrics_detect": "not yet in branch lifecycle rows",
            "existing_but_weak": "exists",
            "carrying_capacity_risk": "medium",
        },
    ]


def _report(final_decision: Mapping[str, Any], validation: Mapping[str, Any], matrix_uniqueness: Mapping[str, Any], paths: Mapping[str, str]) -> str:
    return "\n".join(
        [
            "# EIEL BRANCH-3 World-Demand Theory and Strategy Divergence",
            "",
            "## Purpose",
            "BRANCH-3 audits whether the post-BRANCH-2 branch/strategy layer can select a lineage-blind world-demand mechanism, then ranks candidate ecological-demand families under strict controls.",
            "",
            "## BRANCH-2 dependency",
            "Dependency commit: `df8c4d46311969305959d880c1e67eb7a45755c3`. BRANCH-2-C is accepted: compact branch observability is method-ready, Candidate A is rejected, Candidate B is parked, and Candidate-B2 was not justified.",
            "",
            "## Why reproduction-supply candidates failed",
            "Candidate A reduced reproduction enough to collapse or block viable branching. Candidate B widened placement opportunity and increased births, living population, and descendant branches, but also shifted flat controls and produced no paired strategy-diversity gain. That is a carrying-capacity/reproduction-supply effect, not evidence for new ecological strategies.",
            "",
            "## Branch metric audit",
            "Raw branch counts are high because every successful birth edge can create a provisional descendant split before persistence filtering. Conservative and nominal thresholds leave small persistent counts; permissive thresholds accept nearly every birth edge. Persistent branch counts are therefore useful as a method signal, but still threshold-sensitive and not by themselves a biological strategy claim.",
            "",
            "## Strategy metric audit",
            f"Raw lifecycle strategy ids are near per-agent labels in the decision cells: max active-strategy/lifecycle ratio `{matrix_uniqueness.get('max_active_strategy_per_lifecycle_ratio')}`, rows with every alive agent carrying a unique alive strategy id `{matrix_uniqueness.get('rows_with_alive_agents_all_unique')}`. Across the reused archive, repeated cells reduce the global ratio to `{validation.get('raw_uniqueness_ratio')}`, but the per-cell view is the gate-relevant signal. Current raw signatures include genome/capacity fields and outcome fields such as energy, health, injury, offspring count, first reproduction step, and lifetime.",
            "",
            "## Strategy descriptor contract",
            "A valid strategy layer must separate raw individual signatures, compressed behavioral strategies, and persistent ecological strategies. It must normalize behavior by exposure, bin deterministically, avoid genome identity as the strategy identity, and require branch-persistent ecological use before mechanics selection.",
            "",
            "## World-demand inventory",
            "Existing Patch/Jungle/ROZ/WDL surfaces can express spatial and temporal demand, but branch lifecycle rows do not yet carry enough niche exposure and behavior-by-niche information to evaluate those demands as persistent strategy divergence.",
            "",
            "## Causal theory",
            "The missing causal step is incompatible ecological demand: environmental heterogeneity must create at least two viable roles with costs outside their niche. More births can create more descendant branch labels, but branches only become scientifically meaningful when they map to persistent, reproducing, ecologically different strategies.",
            "",
            "## Candidate-family ranking",
            "The strongest family is `A_patch_derived_ecological_niche_mosaic`, preferably integrated with Path2/ROZ patch-derived zone design rather than duplicating machinery. Temporal demand heterogeneity and predator-resource trade-offs are secondary candidates. Resource specialization is attractive but larger. Niche construction and life-history trade-offs are deferred because they risk hidden subsidies or repeating reproduction-supply tuning.",
            "",
            "## Selected design",
            "Selected design direction: Path2/ROZ-style static patch-derived ecological niche mosaic. It should derive zones from food, shelter, cover, waste, density, movement cost, recovery, and possibly predator-risk surfaces. It is design-selected only; mechanics are not implemented in this package because the current strategy layer is not discriminating enough to judge readiness.",
            "",
            "## Reporting scaffold",
            "The next reporting repair must add compact branch-linked niche exposure, movement/action in niche, resource gain, damage/survival, reproduction by niche, movement between niches, compressed strategy membership, and branch-specific strategy persistence. It must pass flags-off parity, repeat parity, compact storage, and checkpoint compatibility if persistent state is touched.",
            "",
            "## Implementation, If Run",
            "No mechanics implementation was run. This package stops before mechanics under BRANCH-3-D because selecting or validating a world-demand candidate with the current strategy descriptors would be scientifically unsafe.",
            "",
            "## Readiness Campaign, If Run",
            "No campaign was run. The required readiness campaign remains blocked until the reporting repair can distinguish ecological strategy divergence from birth/population effects.",
            "",
            "## Founder Ancestry Findings",
            "Founder ancestry remains a diagnostic constraint, not a target. No new founder-specific mechanics were added.",
            "",
            "## Descendant Branch Findings",
            "BRANCH-2 compact birth edges are valid for reconstructing descendant branches, but branch counts alone cannot select a world-demand mechanism because Candidate B showed branch-positive, strategy-neutral population drift.",
            "",
            "## Strategy Findings",
            f"Decision-critical strategy validation failed closed: {', '.join(str(x) for x in validation.get('blocker_reasons', []))}. Deterministic compression is useful as a reporting repair direction, not as current evidence of ecological strategies.",
            "",
            "## World-Demand Findings",
            "Patch-derived ecological niche mosaics are the leading theory direction because they can create local roles and trade-offs without lineage labels or direct rewards. They still need compact branch-linked niche reporting before mechanics can be judged.",
            "",
            "## Final Decision",
            f"Decision: **{final_decision.get('decision')}**. Secondary decision: **{final_decision.get('secondary_decision')}**. {final_decision.get('interpretation')}",
            "",
            "## Win/Loss",
            "Win: BRANCH-3 identifies why the current strategy layer cannot support mechanics selection and freezes the strongest next design direction. Loss: no world-demand mechanics candidate is scientifically ready for a readiness campaign yet.",
            "",
            "## Simple Explanation",
            "I checked whether the new family-tree records can tell us that branches are living in meaningfully different ways. They can show parent-child branches, but the current strategy labels are too noisy and mostly unique to individual agents. This matters because the larger topic is not just making more offspring; it is making different useful ways of living appear and persist. The best next world idea is a patch-based niche mosaic, but first we need better compact reporting that links branches to actual patch use, risk, food, damage, and reproduction.",
            "",
            "## Limitations",
            "This package reuses BRANCH-2 compact evidence and archived lifecycle rows. It does not run a new simulation campaign, does not prove any world-demand mechanism, and does not validate a live patch-derived niche implementation. Compression bins were not tuned for a preferred conclusion.",
            "",
            "## No-Claims Statement",
            NO_CLAIMS,
            "",
            "## Recommended Next Step",
            "Implement the reporting repair first: compact branch-linked ecological exposure and compressed strategy membership for patch-derived niche surfaces, with disabled-control leakage, flags-off parity, repeat parity, and small fixture evidence. Then reconsider a default-off Path2/ROZ patch-derived niche readiness test.",
            "",
            "## Paths",
            f"Evidence directory: `{paths.get('evidence_dir')}`",
            f"Runtime archive manifest: `{paths.get('runtime_archive_manifest')}`",
        ]
    )


def generate(evidence_dir: Path, report_path: Path, branch2_evidence_dir: Path) -> dict[str, Any]:
    matrix_rows = _read_csv(branch2_evidence_dir / "branch2_matrix.csv")
    threshold_rows = _read_csv(branch2_evidence_dir / "branch2_threshold_sensitivity.csv")
    candidate_b = _read_json(branch2_evidence_dir / "branch2_candidate_b_reinterpretation.json")
    final2 = _read_json(branch2_evidence_dir / "branch2_final_decision.json")
    runtime_manifest = _read_json(branch2_evidence_dir / "branch2_runtime_archive_manifest.json")
    matrix_uniqueness = _matrix_strategy_uniqueness(matrix_rows)
    archive_paths = [str(item.get("archive", "")) for item in runtime_manifest.get("archives", []) or [] if item.get("archive")]
    lifecycle_rows, lifecycle_manifest = load_lifecycle_rows_from_archives(archive_paths[:1])
    validation = strategy_validation(lifecycle_rows)
    sensitivity_rows = compression_sensitivity(lifecycle_rows, matrix_rows)
    family_rows = _candidate_family_rows()
    world_rows = _world_inventory_rows()

    evidence_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    _write_json(
        evidence_dir / "branch3_branch2_metric_audit.json",
        {
            "schema_version": "branch3_branch2_metric_audit_v1",
            "branch2_decision": final2.get("decision"),
            "matrix_rows": len(matrix_rows),
            "threshold_summary": _threshold_summary(threshold_rows),
            "candidate_b_flat_pairs": candidate_b.get("flat_pairs", []),
            "candidate_b_pressure_pairs": candidate_b.get("pressure_pairs", []),
            "matrix_strategy_uniqueness": matrix_uniqueness,
            "answers": {
                "raw_branch_counts_high": "Every successful birth edge can create a provisional descendant split before persistence filtering.",
                "one_birth_edge_one_provisional_branch": True,
                "persistent_branch_meaning": "Method-useful but threshold-sensitive; not enough as a biological strategy claim.",
                "strategy_counts_high": "Lifecycle strategy ids include genome/capacity and outcome-sensitive fields, making them nearly per-agent labels.",
                "current_strategy_treats_agents_unique": matrix_uniqueness["treats_nearly_every_agent_unique"],
                "stable_strategy_conclusion": "Candidate B has no paired strategy gain; raw strategy counts are not scientifically usable for mechanism selection.",
                "candidate_b_effect": "Branch-positive primarily through more births/living agents, with unacceptable flat-control drift.",
                "suitable_for_mechanics_selection": False,
            },
        },
    )
    _write_json(
        evidence_dir / "branch3_branch_semantics_review.json",
        {
            "schema_version": "branch3_branch_semantics_review_v1",
            "branch_observability_valid": bool(final2.get("observability_valid")),
            "provisional_branch_semantics": "birth_edge_split",
            "persistent_branch_semantics": "threshold_filtered_descendant_persistence",
            "threshold_sensitive": True,
            "scientific_use": "separate founder ancestry from descendant branch evidence; do not infer strategy divergence alone",
        },
    )
    _write_json(
        evidence_dir / "branch3_strategy_semantics_review.json",
        {
            "schema_version": "branch3_strategy_semantics_review_v1",
            "continuous_or_sensitive_fields": [
                "genome_signature",
                "behavior_signature",
                "ecology_signature",
                "reproduction_signature",
                "cognition_signature",
                "energy_collected",
                "health",
                "injury",
                "offspring_count",
                "first_repro_step",
                "lifetime_ticks",
            ],
            "constant_or_missing_fields": ["patch_use_summary.coverage=not_available", "sensor_radius constant in BRANCH-2 matrix rows"],
            "valid_for_mechanics_selection": False,
            "reason": "The layer is deterministic and linkable but not scientifically discriminating as ecological strategy evidence.",
        },
    )
    _write_json(
        evidence_dir / "branch3_strategy_descriptor_contract.json",
        {
            "schema_version": "branch3_strategy_descriptor_contract_v1",
            "required_layers": ["raw_individual_signature", "compressed_behavioral_strategy", "persistent_ecological_strategy"],
            "must_include": [
                "branch_id_or_descendant_branch_membership",
                "niche_exposure",
                "movement_or_action_in_niche",
                "resource_gain_by_niche",
                "damage_or_survival_by_niche",
                "reproduction_by_niche",
                "generation_persistence",
                "disabled_control_leakage",
            ],
            "must_avoid": ["genome_identity_as_strategy", "unbinned continuous outcomes", "lifespan-only descriptors", "lineage_or_branch_identity_in_mechanics"],
            "binning_policy": "deterministic, predeclared, sensitivity reported, no tuning to obtain a preferred result",
        },
    )
    _write_csv(evidence_dir / "branch3_strategy_compression_sensitivity.csv", sensitivity_rows)
    _write_json(evidence_dir / "branch3_strategy_validation.json", validation)
    _write_csv(evidence_dir / "branch3_world_demand_inventory.csv", world_rows)
    _write_json(
        evidence_dir / "branch3_existing_mechanism_gap_map.json",
        {
            "schema_version": "branch3_existing_mechanism_gap_map_v1",
            "main_gap": "World surfaces exist, but compact branch lifecycle strategy rows do not yet encode branch-linked niche exposure and behavior.",
            "reporting_gap": "Need compact branch-linked niche exposure, action, resource, damage, movement, and reproduction summaries.",
            "mechanics_gap": "No mechanics gap should be attacked before reporting can validate strategy divergence.",
        },
    )
    _write_json(
        evidence_dir / "branch3_prior_decision_index.json",
        {
            "schema_version": "branch3_prior_decision_index_v1",
            "accepted": ["LIT-1-A", "OES-LITGATE-CAVEAT-A", "LRO-3-D", "LRO-4-C", "BRANCH-1R-D", "BRANCH-2-C"],
            "candidate_a": "rejected",
            "candidate_b": "parked_not_advanced_unchanged",
            "candidate_b2": "skipped_not_justified",
            "path2_roz2": "design-selected as relevant patch-derived zone scaffold; mechanics not implemented here",
            "path1_drh0": "useful delayed-demand fixture lineage; not selected as immediate branch strategy mechanic",
        },
    )
    _write_json(
        evidence_dir / "branch3_causal_failure_map.json",
        {
            "schema_version": "branch3_causal_failure_map_v1",
            "candidate_a": ["local crowding penalty", "lower reproduction", "reduced births and viability", "stronger concentration", "collapse before useful branching"],
            "candidate_b": ["wider placement", "fewer no-space failures", "more births and living agents", "more descendant branches", "no paired strategy gain", "unacceptable flat-control drift"],
            "missing_step": "incompatible ecological demand that maps branches to persistent, reproducing, different strategies",
        },
    )
    _write_json(
        evidence_dir / "branch3_required_world_properties.json",
        {
            "schema_version": "branch3_required_world_properties_v1",
            "required": [
                "at least two viable ecological roles",
                "cost outside each role's niche",
                "no universally best role by construction",
                "demand before or during branching window",
                "lineage-blind and branch-blind mechanics",
                "flat-control neutrality",
                "compact branch-linked strategy observability",
            ],
        },
    )
    _write_text(
        evidence_dir / "branch3_strategy_divergence_theory.md",
        "A branch is scientifically useful only when it maps to a persistent way of living. The desired chain is environmental heterogeneity to incompatible local or temporal demands to trait-dependent success to persistent strategy differences to several reproducing descendant branches. The rejected chain is generic population increase to more births to more branch labels without ecological strategy divergence.",
    )
    _write_csv(evidence_dir / "branch3_candidate_family_matrix.csv", family_rows)
    _write_json(
        evidence_dir / "branch3_candidate_ranking.json",
        {
            "schema_version": "branch3_candidate_ranking_v1",
            "ranking": [row["family"] for row in sorted(family_rows, key=lambda item: int(item["score"]), reverse=True)],
            "selected_design_direction": "A_patch_derived_ecological_niche_mosaic",
            "selection_blocked_for_mechanics": True,
            "reason": "Strategy descriptors are not currently discriminating enough to judge readiness.",
        },
    )
    _write_json(
        evidence_dir / "branch3_selection_decision.json",
        {
            "schema_version": "branch3_selection_decision_v1",
            "selected_family": "A_patch_derived_ecological_niche_mosaic",
            "selected_candidate": "Path2/ROZ-style static patch-derived ecological niche mosaic",
            "mechanics_selected_for_implementation": False,
            "decision": "BRANCH-3-DESIGN-ONLY",
            "reason": "Design direction is strongest, but mechanics selection must wait for strategy reporting repair.",
        },
    )
    _write_json(
        evidence_dir / "branch3_selected_candidate_design.json",
        {
            "schema_version": "branch3_selected_candidate_design_v1",
            "purpose": "create local ecological roles from existing patch surfaces without lineage labels or direct rewards",
            "candidate": "Path2/ROZ-style static patch-derived ecological niche mosaic",
            "activation_boundary": "default-off; reporting repair first",
            "surfaces": ["food", "shelter", "cover", "waste", "density", "movement_cost", "recovery", "predator_risk_deferred"],
            "tradeoffs": ["safe-low-food versus risky-high-food", "low-waste-low-density versus high-resource-crowded", "shelter occupancy versus movement/resource opportunity"],
            "neutral_controls": ["flat_control", "patch-disabled negative", "reporting disabled leakage"],
        },
    )
    _write_json(
        evidence_dir / "branch3_selected_candidate_risk_register.json",
        {
            "schema_version": "branch3_selected_candidate_risk_register_v1",
            "risks": [
                {"risk": "creates good-zone versus bad-zone gradient", "mitigation": "require trade-off scoring and no globally best zone"},
                {"risk": "changes carrying capacity", "mitigation": "flat-control and population/birth neutrality gates"},
                {"risk": "strategy labels remain noisy", "mitigation": "repair compact strategy reporting before mechanics readiness"},
                {"risk": "reporting leaks into mechanics", "mitigation": "flags-off parity and disabled-control leakage tests"},
            ],
        },
    )
    _write_json(
        evidence_dir / "branch3_selected_candidate_gate.json",
        {
            "schema_version": "branch3_selected_candidate_gate_v1",
            "must_pass_before_campaign": [
                "strategy reporting repair",
                "disabled-control leakage zero",
                "flags-off trajectory parity",
                "repeat parity",
                "branch-linked niche exposure coverage",
                "flat-control neutrality",
            ],
        },
    )
    _write_json(
        evidence_dir / "branch3_reporting_summary.json",
        {
            "schema_version": "branch3_reporting_summary_v1",
            "status": "reporting_repair_required",
            "new_reporting_added": False,
            "required_next_fields": [
                "branch_niche_exposure",
                "branch_niche_action",
                "branch_niche_resource_gain",
                "branch_niche_damage",
                "branch_niche_reproduction",
                "compressed_strategy_id",
                "strategy_confidence",
                "disabled_leakage",
            ],
        },
    )
    skipped = {
        "status": "not_run",
        "reason": "BRANCH-3-D stops before mechanics/campaign because the current strategy descriptor layer is not scientifically discriminating.",
    }
    _write_json(evidence_dir / "branch3_implementation_summary.json", {"schema_version": "branch3_implementation_summary_v1", **skipped, "source_changes": "reporting_script_and_tests_only"})
    _write_csv(evidence_dir / "branch3_campaign_matrix.csv", [{"cell": "not_run", **skipped}])
    _write_json(evidence_dir / "branch3_repeat_parity.json", {"schema_version": "branch3_repeat_parity_v1", **skipped})
    _write_csv(evidence_dir / "branch3_founder_metrics.csv", [{"cell": "not_run", "status": "blocked_before_campaign", "reason": skipped["reason"]}])
    _write_csv(evidence_dir / "branch3_branch_metrics.csv", [{"source": "branch2_reuse", "branch_observability_valid": bool(final2.get("observability_valid")), "interpretation": "branch layer valid as method signal, insufficient as strategy claim"}])
    _write_csv(evidence_dir / "branch3_strategy_metrics.csv", [{"source": "branch2_archive_reuse", **{k: validation.get(k) for k in ("raw_lifecycle_rows", "raw_strategy_ids", "raw_uniqueness_ratio", "nominal_compressed_strategy_ids")}}])
    _write_csv(evidence_dir / "branch3_world_demand_metrics.csv", [{"cell": "not_run", "status": "blocked_before_campaign", "selected_family": "A_patch_derived_ecological_niche_mosaic"}])
    _write_json(
        evidence_dir / "branch3_readiness_decision.json",
        {
            "schema_version": "branch3_readiness_decision_v1",
            "readiness_campaign_run": False,
            "readiness_pass": False,
            "decision": "not_ready_for_campaign",
            "reason": skipped["reason"],
        },
    )
    _write_json(
        evidence_dir / "branch3_runtime_archive_manifest.json",
        {
            "schema_version": "branch3_runtime_archive_manifest_v1",
            "status": "no_new_runtime",
            "new_runtime_roots": [],
            "new_runtime_archives": [],
            "reused_branch2_archives": archive_paths,
            "lifecycle_load_manifest": lifecycle_manifest,
        },
    )
    final_decision = {
        "schema_version": "branch3_final_decision_v1",
        "decision": "BRANCH-3-D",
        "secondary_decision": "BRANCH-3-DESIGN-ONLY",
        "interpretation": "Current strategy descriptors are deterministic but not scientifically discriminating enough to select or validate a world-demand mechanics candidate.",
        "selected_design_direction": "Path2/ROZ-style static patch-derived ecological niche mosaic",
        "implementation_run": False,
        "campaign_run": False,
        "strategy_reporting_repair_required": True,
        "no_claims": NO_CLAIMS,
        "next_step": "Repair compact branch-linked ecological strategy reporting before a default-off patch-derived niche readiness test.",
    }
    _write_json(evidence_dir / "branch3_final_decision.json", final_decision)
    paths = {
        "evidence_dir": str(evidence_dir),
        "runtime_archive_manifest": str(evidence_dir / "branch3_runtime_archive_manifest.json"),
    }
    _write_text(report_path, _report(final_decision, validation, matrix_uniqueness, paths))
    _write_text(evidence_dir / report_path.name, _report(final_decision, validation, matrix_uniqueness, paths))
    return {
        "decision": final_decision["decision"],
        "secondary_decision": final_decision["secondary_decision"],
        "evidence_dir": str(evidence_dir),
        "report_path": str(report_path),
        "files_written": len(list(evidence_dir.iterdir())),
        "validation": validation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--branch2-evidence-dir", default="evidences/06-open-ended-search/11-branch2-compact-observability")
    args = parser.parse_args(argv)
    summary = generate(Path(args.evidence_dir).resolve(), Path(args.report_path).resolve(), Path(args.branch2_evidence_dir).resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
