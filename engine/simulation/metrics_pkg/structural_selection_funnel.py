from __future__ import annotations

"""Default-off structural selection funnel reporting for REP-2G.

This module records compact, read-only summaries around the existing selection
and reproduction timing. It does not change ranking, culling, parent draws,
child creation, or RNG use.
"""

import json
import os
from collections import Counter
from hashlib import sha256
from typing import Any


SCHEMA_VERSION = "structural_selection_funnel_v1"
ACTION_REACHABLE_FIELDS = ("aggression_bias", "action_temperature", "drive_logit_gain", "drive_inertia")
BASELINE_EFFECT = {
    "aggression_bias": 0.0,
    "action_temperature": 1.0,
    "drive_logit_gain": 1.0,
    "drive_inertia": 0.0,
}


def enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "structural_selection_funnel_reporting_enable", False)) if sim_cfg is not None else False


def capture_pre_turnover_snapshot(
    runner: Any,
    *,
    generation: int,
    alive_start_agents: list[Any],
    alive_after_agents: list[Any],
    eligible_ids: set[int],
    selected_parent_agents: list[Any],
    birth_contexts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not enabled(getattr(runner, "sim_cfg", None)):
        return {}
    selected_ids = {int(getattr(a, "agent_id", -1)) for a in list(selected_parent_agents or [])}
    alive_after_ids = {int(getattr(a, "agent_id", -1)) for a in list(alive_after_agents or [])}
    alive_after_by_id = {int(getattr(a, "agent_id", -1)): a for a in list(alive_after_agents or [])}
    recent = list(getattr(runner, "_structural_selection_funnel_recent_child_links", []) or [])
    retained_children = []
    for row in recent:
        child_id = int(row.get("child_id", -1))
        if child_id not in alive_after_ids:
            continue
        snap = _agent_snapshot(runner, alive_after_by_id.get(child_id), stage="short_retention_later_living_descendant")
        snap["child_id"] = child_id
        snap["parent_a_id"] = int(row.get("parent_a_id", -1))
        snap["parent_b_id"] = int(row.get("parent_b_id", -1))
        retained_children.append(snap)
    return {
        "schema_version": SCHEMA_VERSION,
        "generation": int(generation),
        "alive_start": [_agent_snapshot(runner, a, stage="alive_start") for a in list(alive_start_agents or [])],
        "alive_end": [_agent_snapshot(runner, a, stage="alive_end") for a in list(alive_after_agents or [])],
        "eligible_ids": sorted(int(v) for v in set(eligible_ids or set())),
        "selected_parent_ids_metric_pool": sorted(selected_ids),
        "selected_parent_metric": [
            _agent_snapshot(runner, a, stage="selected_parent_metric_pool")
            for a in list(selected_parent_agents or [])
        ],
        "birth_contexts": [dict(row) for row in list(birth_contexts or [])],
        "prior_child_retention_rows": retained_children,
    }


def write_generation(
    *,
    runner: Any,
    run_dir: str,
    generation: int,
    pre_turnover_snapshot: dict[str, Any],
    post_refill_agents: list[Any],
    turnover_snapshot: dict[str, Any],
) -> dict[str, Any]:
    if not enabled(getattr(runner, "sim_cfg", None)):
        return {}
    pre = dict(pre_turnover_snapshot or {})
    turnover = dict(turnover_snapshot or {})
    post_agents = list(post_refill_agents or [])
    post_by_id = {int(getattr(a, "agent_id", -1)): a for a in post_agents}

    metric_selected_ids = set(int(v) for v in pre.get("selected_parent_ids_metric_pool", []) or [])
    eligible_ids = set(int(v) for v in pre.get("eligible_ids", []) or [])
    actual_parent_ids = set(int(v) for v in turnover.get("actual_refill_parent_pool_ids", []) or [])
    sampled_ids = set()
    for event in list(turnover.get("parent_draw_events", []) or []):
        sampled_ids.add(int(event.get("parent_a_id", -1)))
        sampled_ids.add(int(event.get("parent_b_id", -1)))
    reproduced_ids = set()
    child_links = []
    child_snapshots = []
    alive_end_by_id = {
        int(row.get("agent_id", -1)): dict(row)
        for row in list(pre.get("alive_end", []) or [])
    }
    for event in list(pre.get("birth_contexts", []) or []):
        pa = int(event.get("parent_id", -1))
        pb = int(event.get("mate_id", -1))
        cid = int(event.get("child_id", -1))
        if pa >= 0:
            reproduced_ids.add(pa)
        if pb >= 0:
            reproduced_ids.add(pb)
        child_snapshot = dict(alive_end_by_id.get(cid, {}))
        if child_snapshot:
            child_snapshot["stage"] = "child_inheritance"
            child_snapshots.append(child_snapshot)
        child_links.append(
            {
                "child_id": cid,
                "parent_a_id": pa,
                "parent_b_id": pb,
                "child_alive_at_generation_end": int(cid in alive_end_by_id),
                "child_live_effect_present": int(bool(child_snapshot.get("live_effect_present", False))),
                "child_effect_signature": str(child_snapshot.get("effect_signature", "")),
                "child_changed_effective_fields": list(child_snapshot.get("changed_effective_fields", []) or []),
                "source": "tick_loop_successful_birth",
            }
        )
    for event in list(turnover.get("successful_reproduction_events", []) or []):
        pa = int(event.get("parent_a_id", -1))
        pb = int(event.get("parent_b_id", -1))
        cid = int(event.get("child_id", -1))
        reproduced_ids.update([pa, pb])
        child = post_by_id.get(cid)
        child_snapshot = _agent_snapshot(runner, child, stage="child_inheritance") if child is not None else {}
        if child_snapshot:
            child_snapshots.append(child_snapshot)
        child_links.append(
            {
                "child_id": cid,
                "parent_a_id": pa,
                "parent_b_id": pb,
                "child_alive_post_refill": int(cid in post_by_id),
                "child_live_effect_present": int(bool(child_snapshot.get("live_effect_present", False))),
                "child_effect_signature": str(child_snapshot.get("effect_signature", "")),
                "child_changed_effective_fields": list(child_snapshot.get("changed_effective_fields", []) or []),
                "source": "generation_end_refill",
            }
        )

    stage_rows = []
    stage_rows.extend(_stage_rows_from_snapshots(pre.get("alive_start", []), "alive_start"))
    stage_rows.extend(_stage_rows_from_snapshots(pre.get("alive_end", []), "alive_end"))
    stage_rows.extend(_stage_rows_for_ids(pre.get("alive_end", []), eligible_ids, "eligible_parent"))
    stage_rows.extend(_stage_rows_for_ids(pre.get("alive_end", []), metric_selected_ids, "selected_parent_metric_pool"))
    stage_rows.extend(_stage_rows_for_ids(pre.get("alive_end", []), actual_parent_ids, "actual_refill_parent_pool"))
    stage_rows.extend(_stage_rows_for_ids(pre.get("alive_end", []), sampled_ids, "sampled_parent_draw"))
    stage_rows.extend(_stage_rows_for_ids(pre.get("alive_end", []), reproduced_ids, "successful_reproducer"))
    stage_rows.extend(_stage_rows_from_snapshots(child_snapshots, "child_inheritance"))
    stage_rows.extend(
        _stage_rows_from_snapshots(
            [_agent_snapshot(runner, a, stage="post_refill_alive") for a in post_agents],
            "post_refill_alive",
        )
    )
    retained = list(pre.get("prior_child_retention_rows", []) or [])
    for row in retained:
        row = dict(row)
        row["stage"] = "short_retention_later_living_descendant"
        row["agent_id"] = int(row.get("child_id", row.get("agent_id", -1)))
        row.setdefault("live_effect_present", bool(row.get("child_live_effect_present", False)))
        stage_rows.append(row)

    stage_summary = _summarize_stage_rows(stage_rows)
    behavior_rates = _opportunity_conditioned_rates(getattr(runner, "_structural_action_opportunity_total", {}) or {})
    first_loss = _first_loss_stage(stage_summary)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generation": int(generation),
        "enabled": True,
        "stage_summary": stage_summary,
        "first_loss_stage": first_loss,
        "behavior_rates": behavior_rates,
        "metric_selected_parent_count": len(metric_selected_ids),
        "actual_refill_parent_pool_count": len(actual_parent_ids),
        "sampled_parent_count": len(sampled_ids),
        "successful_reproducer_count": len(reproduced_ids),
        "child_link_count": len(child_links),
        "child_live_effect_present_count": sum(int(r.get("child_live_effect_present", 0) or 0) for r in child_links),
        "tick_loop_successful_birth_count": sum(1 for r in child_links if r.get("source") == "tick_loop_successful_birth"),
        "generation_end_refill_child_count": sum(1 for r in child_links if r.get("source") == "generation_end_refill"),
        "prior_child_retained_count": len(retained),
        "turnover_status": str(turnover.get("status", "")),
        "child_links": child_links[:256],
        "reporting_only": True,
    }
    _record_total(runner, summary)
    _write_files(run_dir, summary, getattr(runner, "_structural_selection_funnel_total", {}) or {})
    runner._structural_selection_funnel_recent_child_links = list(child_links)
    return summary


def _agent_snapshot(runner: Any, agent: Any, *, stage: str) -> dict[str, Any]:
    if agent is None:
        return {"stage": stage, "agent_id": -1, "live_graph_valid": False, "live_effect_present": False}
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
    action_changed = [f for f in changed if f in ACTION_REACHABLE_FIELDS]
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
    return {
        "stage": stage,
        "agent_id": aid,
        "live_graph_valid": bool(live_graph_valid),
        "live_effect_present": bool(action_changed),
        "changed_effective_fields": action_changed,
        "effect_score": float(effect_score),
        "effect_signature": _stable_hash(signature_payload) if action_changed else "",
        "graph_digest": graph_digest,
        "developmental_stage_id": stage_id,
        "alive": int(bool(getattr(agent, "alive", True))),
    }


def _stage_rows_from_snapshots(rows: Any, stage: str) -> list[dict[str, Any]]:
    out = []
    for row in list(rows or []):
        item = dict(row)
        item["stage"] = stage
        out.append(item)
    return out


def _stage_rows_for_ids(rows: Any, ids: set[int], stage: str) -> list[dict[str, Any]]:
    out = []
    for row in list(rows or []):
        item = dict(row)
        aid = int(item.get("agent_id", -1))
        if aid in ids:
            item["stage"] = stage
            out.append(item)
    return out


def _summarize_stage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by.setdefault(str(row.get("stage", "")), []).append(row)
    out = []
    for stage, group in sorted(by.items()):
        live = [r for r in group if bool(r.get("live_effect_present", False))]
        graphs = [r for r in group if bool(r.get("live_graph_valid", False))]
        scores = [float(r.get("effect_score", 0.0) or 0.0) for r in live]
        fields = Counter()
        for row in live:
            fields.update(str(v) for v in list(row.get("changed_effective_fields", []) or []))
        out.append(
            {
                "stage": stage,
                "agent_count": len(group),
                "live_graph_valid_count": len(graphs),
                "live_effect_present_count": len(live),
                "live_effect_fraction": (float(len(live)) / float(len(group))) if group else None,
                "mean_effect_score_live": (sum(scores) / len(scores)) if scores else None,
                "changed_effective_field_counts": dict(sorted(fields.items())),
            }
        )
    return out


def _first_loss_stage(stage_summary: list[dict[str, Any]]) -> str | None:
    order = [
        "alive_start",
        "alive_end",
        "eligible_parent",
        "selected_parent_metric_pool",
        "actual_refill_parent_pool",
        "sampled_parent_draw",
        "successful_reproducer",
        "child_inheritance",
        "post_refill_alive",
        "short_retention_later_living_descendant",
    ]
    by = {str(row.get("stage")): row for row in stage_summary}
    seen_positive = False
    for stage in order:
        row = by.get(stage)
        if row is None:
            continue
        count = int(row.get("live_effect_present_count", 0) or 0)
        if count > 0:
            seen_positive = True
        elif seen_positive:
            return stage
    return None


def _opportunity_conditioned_rates(opp: dict[str, Any]) -> dict[str, Any]:
    evaluated = max(1, int(opp.get("evaluated_decisions", 0) or 0))
    potential = max(1, int(opp.get("potential_linked_opportunity_count", 0) or 0))
    attack_exec = max(1, int(opp.get("attack_executable_count", 0) or 0))
    gather_exec = max(1, int(opp.get("gather_executable_count", 0) or 0))
    rest_need = max(1, int(opp.get("rest_need_present_count", 0) or 0))
    move_multi = max(1, int(opp.get("decisions_with_multiple_moves", 0) or 0))
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
        "attack_chosen_per_attack_executable": float(int(opp.get("attack_chosen_count", 0) or 0)) / float(attack_exec),
        "gather_chosen_per_gather_executable": float(int(opp.get("gather_chosen_count", 0) or 0)) / float(gather_exec),
        "rest_chosen_per_rest_need": float(int(opp.get("rest_chosen_count", 0) or 0)) / float(rest_need),
        "move_chosen_per_multi_move_decision": float(int(opp.get("move_chosen_count", 0) or 0)) / float(move_multi),
    }


def _record_total(runner: Any, summary: dict[str, Any]) -> None:
    total = getattr(runner, "_structural_selection_funnel_total", None)
    if not isinstance(total, dict):
        total = {"schema_version": SCHEMA_VERSION, "enabled": True, "generation_count": 0, "stage_totals": {}}
    total["generation_count"] = int(total.get("generation_count", 0) or 0) + 1
    total["last_generation"] = int(summary.get("generation", -1))
    total["last_first_loss_stage"] = summary.get("first_loss_stage")
    stage_totals = dict(total.get("stage_totals", {}) or {})
    for row in list(summary.get("stage_summary", []) or []):
        stage = str(row.get("stage"))
        cur = dict(stage_totals.get(stage, {}) or {})
        cur["agent_count"] = int(cur.get("agent_count", 0) or 0) + int(row.get("agent_count", 0) or 0)
        cur["live_effect_present_count"] = int(cur.get("live_effect_present_count", 0) or 0) + int(row.get("live_effect_present_count", 0) or 0)
        cur["live_graph_valid_count"] = int(cur.get("live_graph_valid_count", 0) or 0) + int(row.get("live_graph_valid_count", 0) or 0)
        stage_totals[stage] = cur
    total["stage_totals"] = dict(sorted(stage_totals.items()))
    runner._structural_selection_funnel_total = total


def _write_files(run_dir: str, summary: dict[str, Any], total: dict[str, Any]) -> None:
    if not run_dir:
        return
    metrics_dir = os.path.join(str(run_dir), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    jsonl_path = os.path.join(metrics_dir, "structural_selection_funnel_generation_summary.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
    summary_path = os.path.join(metrics_dir, "structural_selection_funnel_run_summary.json")
    tmp = summary_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(total, f, sort_keys=True, indent=2, ensure_ascii=True)
        f.write("\n")
    os.replace(tmp, summary_path)


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(data.encode("utf-8")).hexdigest()
