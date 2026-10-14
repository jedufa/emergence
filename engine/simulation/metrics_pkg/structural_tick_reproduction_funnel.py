from __future__ import annotations

"""Default-off tick-loop reproductive selection reporting for REP-2H.

The reducer consumes existing lineage reproduction scan/event contexts and
writes compact summaries for the active live tick-loop birth path. It does not
change reproduction eligibility, parent choice, child creation, culling, or RNG.
"""

import json
import os
from collections import Counter
from hashlib import sha256
from typing import Any

from evolution.structural_variant_identity import agent_identity as _variant_agent_identity
from evolution.structural_variant_identity import enabled as _variant_identity_enabled


SCHEMA_VERSION = "structural_tick_reproduction_funnel_v1"
ACTION_REACHABLE_FIELDS = ("aggression_bias", "action_temperature", "drive_logit_gain", "drive_inertia")
BASELINE_EFFECT = {
    "aggression_bias": 0.0,
    "action_temperature": 1.0,
    "drive_logit_gain": 1.0,
    "drive_inertia": 0.0,
}
STAGE_ORDER = (
    "live_effect_alive",
    "reproduction_opportunity",
    "reproduction_attempt",
    "successful_tick_reproducer",
    "child_inheritance",
    "short_retention_next_generation",
    "short_retention_two_generations",
)


def enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "structural_tick_reproduction_funnel_reporting_enable", False)) if sim_cfg is not None else False


def write_generation(
    *,
    runner: Any,
    run_dir: str,
    generation: int,
    alive_after_agents: list[Any],
    scan_action_event_contexts: list[dict[str, Any]] | None = None,
    birth_event_contexts: list[dict[str, Any]] | None = None,
    birth_contexts: list[dict[str, Any]] | None = None,
    post_refill_agents: list[Any] | None = None,
) -> dict[str, Any]:
    if not enabled(getattr(runner, "sim_cfg", None)):
        return {}

    alive_agents = list(alive_after_agents or [])
    post_agents = list(post_refill_agents or alive_agents)
    alive_by_id = {int(getattr(a, "agent_id", -1)): a for a in alive_agents}
    post_by_id = {int(getattr(a, "agent_id", -1)): a for a in post_agents}
    alive_snaps = {aid: _agent_snapshot(runner, agent, stage="alive_end") for aid, agent in alive_by_id.items()}
    post_snaps = {aid: _agent_snapshot(runner, agent, stage="post_refill_alive") for aid, agent in post_by_id.items()}

    scans = [dict(row) for row in list(scan_action_event_contexts or [])]
    events = [dict(row) for row in list(birth_event_contexts or [])]
    births = [dict(row) for row in list(birth_contexts or [])]
    if not births:
        births = [row for row in events if str(row.get("event_type", "")) == "birth_success"]

    opportunity_ids: set[int] = set()
    opportunity_event_count = 0
    no_attempt_count = 0
    ineligible_reasons: Counter[str] = Counter()
    exposure_ids: set[int] = set()
    for row in scans:
        parent_id = _int(row.get("parent_id", row.get("agent_id", -1)), -1)
        event_type = str(row.get("event_type", row.get("event", "")) or "")
        if event_type == "reproduction_scan_exposure":
            if parent_id >= 0:
                exposure_ids.add(parent_id)
        elif event_type == "reproduction_scan_ineligible":
            reason = str(row.get("scan_ineligible_reason_detail", row.get("failure_reason", "")) or "unknown")
            ineligible_reasons[reason] += 1
        elif event_type in {"reproduction_scan_eligible", "reproduction_scan_no_attempt", "reproduction_scan_attempt_recorded"}:
            opportunity_event_count += 1
            if parent_id >= 0:
                opportunity_ids.add(parent_id)
            if event_type == "reproduction_scan_no_attempt":
                no_attempt_count += 1

    attempt_ids: set[int] = set()
    failed_attempt_ids: set[int] = set()
    failed_attempt_count = 0
    failed_pre_count = 0
    failure_reasons: Counter[str] = Counter()
    for row in events:
        if _int(row.get("birth_attempted", 0), 0) <= 0:
            if str(row.get("event_type", "")) not in {"birth_success", "successful_birth"}:
                failed_pre_count += int(str(row.get("event_type", "")).startswith("birth_fail"))
            continue
        pa = _int(row.get("parent_id", -1), -1)
        pb = _int(row.get("mate_id", -1), -1)
        if pa >= 0:
            attempt_ids.add(pa)
            opportunity_ids.add(pa)
        if pb >= 0:
            attempt_ids.add(pb)
            opportunity_ids.add(pb)
        if _int(row.get("birth_success", 0), 0) <= 0 and str(row.get("event_type", "")) != "birth_success":
            failed_attempt_count += 1
            if pa >= 0:
                failed_attempt_ids.add(pa)
            if pb >= 0:
                failed_attempt_ids.add(pb)
            reason = str(row.get("failure_reason", row.get("event_type", "")) or "unknown")
            failure_reasons[reason] += 1

    successful_reproducer_ids: set[int] = set()
    child_links: list[dict[str, Any]] = []
    child_ids: set[int] = set()
    parent_contribution_count = 0
    for row in births:
        if str(row.get("event_type", row.get("event", "")) or "") not in {"birth_success", "successful_birth"}:
            continue
        pa = _int(row.get("parent_id", -1), -1)
        pb = _int(row.get("mate_id", -1), -1)
        cid = _int(row.get("child_id", -1), -1)
        if pa >= 0:
            successful_reproducer_ids.add(pa)
            attempt_ids.add(pa)
            opportunity_ids.add(pa)
            parent_contribution_count += 1
        if pb >= 0:
            successful_reproducer_ids.add(pb)
            attempt_ids.add(pb)
            opportunity_ids.add(pb)
            parent_contribution_count += 1
        if cid >= 0:
            child_ids.add(cid)
        child_snapshot = dict(alive_snaps.get(cid) or post_snaps.get(cid) or {})
        link = {
            "generation": int(generation),
            "child_id": cid,
            "parent_a_id": pa,
            "parent_b_id": pb,
            "child_alive_generation_end": int(cid in alive_by_id),
            "child_alive_post_refill": int(cid in post_by_id),
            "child_live_effect_present": int(bool(child_snapshot.get("live_effect_present", False))),
            "child_live_graph_valid": int(bool(child_snapshot.get("live_graph_valid", False))),
            "child_effect_bin": str(child_snapshot.get("effect_bin", "")),
            "child_effect_signature": str(child_snapshot.get("effect_signature", "")),
        }
        for key in (
            "graph_identity_digest",
            "graph_semantic_digest",
            "canonical_seed_exact",
            "canonical_seed_semantic_match",
            "variant_class",
            "operator_origin_available",
            "operator_origin_set",
            "mutation_operator_origin_set",
            "c1_attempt_count",
            "c1_noop_count",
        ):
            if key in child_snapshot:
                link[f"child_{key}"] = child_snapshot.get(key)
        child_links.append(link)

    previous_links = list(getattr(runner, "_structural_tick_reproduction_recent_child_links", []) or [])
    two_gen_links = list(getattr(runner, "_structural_tick_reproduction_previous_child_links", []) or [])
    retained_next = _retained_links(runner, previous_links, post_by_id, stage="short_retention_next_generation")
    retained_two = _retained_links(runner, two_gen_links, post_by_id, stage="short_retention_two_generations")

    stage_rows = [
        _stage_row("live_effect_alive", list(alive_snaps.values())),
        _stage_row("reproduction_opportunity", [alive_snaps.get(aid, post_snaps.get(aid, {})) for aid in sorted(opportunity_ids)]),
        _stage_row("reproduction_attempt", [alive_snaps.get(aid, post_snaps.get(aid, {})) for aid in sorted(attempt_ids)]),
        _stage_row("successful_tick_reproducer", [alive_snaps.get(aid, post_snaps.get(aid, {})) for aid in sorted(successful_reproducer_ids)]),
        _stage_row("child_inheritance", [alive_snaps.get(cid, post_snaps.get(cid, {})) for cid in sorted(child_ids)]),
        _stage_row("short_retention_next_generation", retained_next),
        _stage_row("short_retention_two_generations", retained_two),
    ]
    first_loss = _first_loss_stage(stage_rows)
    behavior_rates = _opportunity_conditioned_rates(getattr(runner, "_structural_action_opportunity_total", {}) or {})
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generation": int(generation),
        "enabled": True,
        "reporting_only": True,
        "active_reproduction_path": "live_tick_loop_births",
        "inactive_generation_end_refill_excluded": True,
        "stage_summary": stage_rows,
        "first_loss_stage": first_loss,
        "alive_agent_count": len(alive_snaps),
        "live_effect_alive_count": sum(int(bool(row.get("live_effect_present"))) for row in alive_snaps.values()),
        "reproduction_scan_exposure_agent_count": len(exposure_ids),
        "reproduction_opportunity_event_count": int(opportunity_event_count),
        "reproduction_opportunity_agent_count": len(opportunity_ids),
        "no_attempt_count": int(no_attempt_count),
        "reproduction_attempt_agent_count": len(attempt_ids),
        "reproduction_attempt_event_count": sum(1 for row in events if _int(row.get("birth_attempted", 0), 0) > 0),
        "failed_pre_attempt_count": int(failed_pre_count),
        "failed_attempt_count": int(failed_attempt_count),
        "failed_attempt_agent_count": len(failed_attempt_ids),
        "successful_reproducer_count": len(successful_reproducer_ids),
        "successful_birth_count": len(child_links),
        "parent_contribution_count": int(parent_contribution_count),
        "child_count": len(child_ids),
        "child_live_effect_present_count": sum(int(row.get("child_live_effect_present", 0) or 0) for row in child_links),
        "child_live_graph_valid_count": sum(int(row.get("child_live_graph_valid", 0) or 0) for row in child_links),
        "child_next_generation_alive_count": len(retained_next),
        "child_two_generation_alive_count": len(retained_two),
        "live_effect_success_rate_per_opportunity_agent": _rate(
            _live_effect_count(successful_reproducer_ids, alive_snaps, post_snaps),
            _live_effect_count(opportunity_ids, alive_snaps, post_snaps),
        ),
        "success_rate_per_opportunity_agent": _rate(len(successful_reproducer_ids), len(opportunity_ids)),
        "attempt_rate_per_opportunity_agent": _rate(len(attempt_ids), len(opportunity_ids)),
        "child_live_effect_rate_per_successful_birth": _rate(
            sum(int(row.get("child_live_effect_present", 0) or 0) for row in child_links),
            len(child_links),
        ),
        "next_generation_retention_rate_per_child": _rate(len(retained_next), len(child_links)),
        "two_generation_retention_rate_per_child": _rate(len(retained_two), len(child_links)),
        "ineligible_reason_counts": dict(sorted(ineligible_reasons.items())),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "behavior_rates": behavior_rates,
        "child_links": child_links[:512],
    }
    _record_total(runner, summary)
    _write_files(run_dir, summary, getattr(runner, "_structural_tick_reproduction_funnel_total", {}) or {})
    runner._structural_tick_reproduction_previous_child_links = list(previous_links)
    runner._structural_tick_reproduction_recent_child_links = list(child_links)
    return summary


def _agent_snapshot(runner: Any, agent: Any, *, stage: str) -> dict[str, Any]:
    if agent is None:
        return {"stage": stage, "agent_id": -1, "live_graph_valid": False, "live_effect_present": False, "effect_bin": "unknown"}
    aid = int(getattr(agent, "agent_id", -1))
    changed: list[str] = []
    graph_digest = ""
    effect: dict[str, Any] = {}
    stage_id = ""
    live_graph_valid = False
    try:
        expr = runner.resolve_developmental_expression_for_agent(agent)
        if isinstance(expr, dict):
            stage_id = str(expr.get("stage_id", "") or "")
            effect = dict(expr.get("effective", {}) or {})
            summary = dict(expr.get("structural_live_semantics", {}) or {})
            changed = sorted(str(v) for v in list(summary.get("changed_effective_fields", []) or []))
            graph_digest = str(summary.get("graph_digest", "") or "")
            live_graph_valid = bool(graph_digest and not str(graph_digest).startswith("unsupported"))
    except Exception:
        pass
    action_changed = [field for field in changed if field in ACTION_REACHABLE_FIELDS]
    effect_score = 0.0
    for field in ACTION_REACHABLE_FIELDS:
        if field in effect:
            try:
                effect_score += abs(float(effect.get(field)) - float(BASELINE_EFFECT.get(field, 0.0)))
            except Exception:
                pass
    signature_payload = {
        "graph_digest": graph_digest,
        "changed": action_changed,
        "effect": {k: effect.get(k) for k in ACTION_REACHABLE_FIELDS if k in effect},
    }
    out = {
        "stage": stage,
        "agent_id": aid,
        "live_graph_valid": bool(live_graph_valid),
        "live_effect_present": bool(action_changed),
        "changed_effective_fields": action_changed,
        "effect_score": float(effect_score),
        "effect_bin": _effect_bin(effect_score, bool(action_changed)),
        "effect_signature": _stable_hash(signature_payload) if action_changed else "",
        "graph_digest": graph_digest,
        "developmental_stage_id": stage_id,
        "alive": int(bool(getattr(agent, "alive", True))),
    }
    if _variant_identity_enabled(getattr(runner, "sim_cfg", None)):
        try:
            canonical = str(getattr(runner, "_structural_variant_identity_canonical_semantic_digest", "") or "")
            ident = _variant_agent_identity(
                agent,
                canonical_semantic_digest=canonical,
                live_effect_present=bool(action_changed),
            )
            out.update(
                {
                    "graph_identity_digest": ident.get("graph_identity_digest", ""),
                    "graph_semantic_digest": ident.get("graph_semantic_digest", ""),
                    "canonical_seed_exact": int(bool(ident.get("canonical_seed_exact", False))),
                    "canonical_seed_semantic_match": int(bool(ident.get("canonical_seed_semantic_match", False))),
                    "variant_class": str(ident.get("variant_class", "not_evaluable")),
                    "operator_origin_available": int(bool(ident.get("operator_origin_available", False))),
                    "operator_origin_set": ",".join(str(item) for item in list(ident.get("operator_origin_set", []) or [])),
                    "mutation_operator_origin_set": ",".join(
                        str(item) for item in list(ident.get("mutation_operator_origin_set", []) or [])
                    ),
                    "c1_attempt_count": int(ident.get("c1_attempt_count", 0) or 0),
                    "c1_noop_count": int(ident.get("c1_noop_count", 0) or 0),
                }
            )
        except Exception as exc:
            out["variant_identity_error"] = str(exc)
    return out


def _retained_links(runner: Any, links: list[dict[str, Any]], post_by_id: dict[int, Any], *, stage: str) -> list[dict[str, Any]]:
    out = []
    for row in links:
        cid = _int(row.get("child_id", -1), -1)
        agent = post_by_id.get(cid)
        if agent is None:
            continue
        snap = _agent_snapshot(runner, agent, stage=stage)
        snap["child_id"] = cid
        snap["parent_a_id"] = _int(row.get("parent_a_id", -1), -1)
        snap["parent_b_id"] = _int(row.get("parent_b_id", -1), -1)
        out.append(snap)
    return out


def _stage_row(stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    group = [dict(row) for row in rows if isinstance(row, dict) and int(row.get("agent_id", row.get("child_id", -1)) or -1) >= 0]
    live = [row for row in group if bool(row.get("live_effect_present", False))]
    graphs = [row for row in group if bool(row.get("live_graph_valid", False))]
    fields: Counter[str] = Counter()
    bins: Counter[str] = Counter()
    variant_classes: Counter[str] = Counter()
    semantic_digests: Counter[str] = Counter()
    mutation_origins: Counter[str] = Counter()
    canonical_exact = 0
    operator_origin_available = 0
    scores = []
    for row in live:
        fields.update(str(v) for v in list(row.get("changed_effective_fields", []) or []))
        bins[str(row.get("effect_bin", "unknown") or "unknown")] += 1
        scores.append(float(row.get("effect_score", 0.0) or 0.0))
    for row in group:
        variant_class = str(row.get("variant_class", "") or "")
        if variant_class:
            variant_classes[variant_class] += 1
        digest = str(row.get("graph_semantic_digest", "") or "")
        if digest:
            semantic_digests[digest] += 1
        if int(row.get("canonical_seed_exact", 0) or 0) > 0:
            canonical_exact += 1
        if int(row.get("operator_origin_available", 0) or 0) > 0:
            operator_origin_available += 1
        for origin in str(row.get("mutation_operator_origin_set", "") or "").split(","):
            origin = origin.strip()
            if origin:
                mutation_origins[origin] += 1
    return {
        "stage": stage,
        "agent_count": len(group),
        "live_graph_valid_count": len(graphs),
        "live_effect_present_count": len(live),
        "live_effect_fraction": (float(len(live)) / float(len(group))) if group else None,
        "mean_effect_score_live": (sum(scores) / len(scores)) if scores else None,
        "effect_bin_counts": dict(sorted(bins.items())),
        "changed_effective_field_counts": dict(sorted(fields.items())),
        "variant_class_counts": dict(sorted(variant_classes.items())),
        "semantic_digest_count": len(semantic_digests),
        "semantic_digest_top_counts": dict(semantic_digests.most_common(8)),
        "canonical_seed_exact_count": int(canonical_exact),
        "operator_origin_available_count": int(operator_origin_available),
        "mutation_operator_origin_counts": dict(sorted(mutation_origins.items())),
    }


def _first_loss_stage(stage_rows: list[dict[str, Any]]) -> str | None:
    by = {str(row.get("stage")): row for row in stage_rows}
    seen_positive = False
    for stage in STAGE_ORDER:
        row = by.get(stage)
        if row is None:
            continue
        count = int(row.get("live_effect_present_count", 0) or 0)
        if count > 0:
            seen_positive = True
        elif seen_positive:
            return stage
    return None


def _record_total(runner: Any, summary: dict[str, Any]) -> None:
    total = getattr(runner, "_structural_tick_reproduction_funnel_total", None)
    if not isinstance(total, dict):
        total = {"schema_version": SCHEMA_VERSION, "enabled": True, "generation_count": 0, "stage_totals": {}}
    total["generation_count"] = int(total.get("generation_count", 0) or 0) + 1
    total["last_generation"] = int(summary.get("generation", -1))
    total["last_first_loss_stage"] = summary.get("first_loss_stage")
    for key in (
        "alive_agent_count",
        "live_effect_alive_count",
        "reproduction_opportunity_event_count",
        "reproduction_opportunity_agent_count",
        "reproduction_attempt_event_count",
        "reproduction_attempt_agent_count",
        "failed_pre_attempt_count",
        "failed_attempt_count",
        "successful_reproducer_count",
        "successful_birth_count",
        "parent_contribution_count",
        "child_count",
        "child_live_effect_present_count",
        "child_next_generation_alive_count",
        "child_two_generation_alive_count",
    ):
        total[key] = int(total.get(key, 0) or 0) + int(summary.get(key, 0) or 0)
    stage_totals = dict(total.get("stage_totals", {}) or {})
    for row in list(summary.get("stage_summary", []) or []):
        stage = str(row.get("stage"))
        cur = dict(stage_totals.get(stage, {}) or {})
        for key in ("agent_count", "live_effect_present_count", "live_graph_valid_count"):
            cur[key] = int(cur.get(key, 0) or 0) + int(row.get(key, 0) or 0)
        cur["canonical_seed_exact_count"] = int(cur.get("canonical_seed_exact_count", 0) or 0) + int(
            row.get("canonical_seed_exact_count", 0) or 0
        )
        cur["operator_origin_available_count"] = int(cur.get("operator_origin_available_count", 0) or 0) + int(
            row.get("operator_origin_available_count", 0) or 0
        )
        cur["semantic_digest_count_last"] = int(row.get("semantic_digest_count", 0) or 0)
        for dict_key in ("variant_class_counts", "mutation_operator_origin_counts"):
            counter = Counter(cur.get(dict_key, {}) or {})
            counter.update(row.get(dict_key, {}) or {})
            cur[dict_key] = dict(sorted(counter.items()))
        stage_totals[stage] = cur
    total["stage_totals"] = dict(sorted(stage_totals.items()))
    total["success_rate_per_opportunity_agent"] = _rate(
        total.get("successful_reproducer_count"),
        total.get("reproduction_opportunity_agent_count"),
    )
    total["attempt_rate_per_opportunity_agent"] = _rate(
        total.get("reproduction_attempt_agent_count"),
        total.get("reproduction_opportunity_agent_count"),
    )
    total["child_live_effect_rate_per_successful_birth"] = _rate(
        total.get("child_live_effect_present_count"),
        total.get("successful_birth_count"),
    )
    total["next_generation_retention_rate_per_child"] = _rate(
        total.get("child_next_generation_alive_count"),
        total.get("successful_birth_count"),
    )
    total["two_generation_retention_rate_per_child"] = _rate(
        total.get("child_two_generation_alive_count"),
        total.get("successful_birth_count"),
    )
    runner._structural_tick_reproduction_funnel_total = total


def _write_files(run_dir: str, summary: dict[str, Any], total: dict[str, Any]) -> None:
    if not run_dir:
        return
    metrics_dir = os.path.join(str(run_dir), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    jsonl_path = os.path.join(metrics_dir, "structural_tick_reproduction_funnel_generation_summary.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
    summary_path = os.path.join(metrics_dir, "structural_tick_reproduction_funnel_run_summary.json")
    tmp = summary_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(total, f, sort_keys=True, indent=2, ensure_ascii=True)
        f.write("\n")
    os.replace(tmp, summary_path)


def _opportunity_conditioned_rates(opp: dict[str, Any]) -> dict[str, Any]:
    evaluated = max(1, int(opp.get("evaluated_decisions", 0) or 0))
    potential = max(1, int(opp.get("potential_linked_opportunity_count", 0) or 0))
    chosen_total = (
        int(opp.get("attack_chosen_count", 0) or 0)
        + int(opp.get("gather_chosen_count", 0) or 0)
        + int(opp.get("rest_chosen_count", 0) or 0)
        + int(opp.get("move_chosen_count", 0) or 0)
    )
    return {
        "evaluated_decisions": int(opp.get("evaluated_decisions", 0) or 0),
        "potential_linked_opportunity_count": int(opp.get("potential_linked_opportunity_count", 0) or 0),
        "chosen_action_rate_per_decision": float(chosen_total) / float(evaluated),
        "chosen_action_rate_per_potential_linked_opportunity": float(chosen_total) / float(potential),
    }


def _live_effect_count(ids: set[int], alive_snaps: dict[int, dict[str, Any]], post_snaps: dict[int, dict[str, Any]]) -> int:
    return sum(int(bool((alive_snaps.get(aid) or post_snaps.get(aid) or {}).get("live_effect_present", False))) for aid in ids)


def _rate(num: Any, denom: Any) -> float | None:
    d = float(denom or 0.0)
    if d <= 0.0:
        return None
    return float(num or 0.0) / d


def _effect_bin(score: float, live: bool) -> str:
    if not live:
        return "none"
    if score < 0.25:
        return "low"
    if score < 1.0:
        return "medium"
    return "high"


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(data.encode("utf-8")).hexdigest()
