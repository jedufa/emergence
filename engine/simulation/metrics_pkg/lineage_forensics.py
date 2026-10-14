from __future__ import annotations

"""Report-only lineage collapse forensics."""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from agents.agent import format_lineage_id, normalize_lineage_stats


SCHEMA_VERSION = 1


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def empty_lineage_forensics_metrics() -> Dict[str, Any]:
    empty_curve = _stable_json({"schema_version": SCHEMA_VERSION, "generations": []})
    empty_counts = _stable_json({"schema_version": SCHEMA_VERSION, "lineages": []})
    empty_events = _stable_json({"schema_version": SCHEMA_VERSION, "events": []})
    empty_context = _stable_json({"schema_version": SCHEMA_VERSION, "contexts": []})
    return {
        "lineage_forensics_enabled": 0,
        "lineage_forensics_schema_version": SCHEMA_VERSION,
        "first_effective_lineage_collapse_generation": -1,
        "first_top1_dominance_generation": -1,
        "first_single_alive_lineage_generation": -1,
        "first_top1_birth_dominance_generation": -1,
        "final_effective_alive_lineages": 0.0,
        "final_alive_lineage_entropy": 0.0,
        "final_top1_alive_share": 0.0,
        "collapse_generation_observed": 0,
        "minority_lineage_birth_window": -1,
        "minority_lineage_death_window": -1,
        "minority_lineage_patch_occupancy_window": -1,
        "final_lineages_with_births": 0,
        "top_lineage_patch_occupancy_share": 0.0,
        "lineages_with_distinct_patch_contexts": 0,
        "branch_candidates_before_collapse": 0,
        "branch_candidates_after_collapse": 0,
        "descriptor_lineages_before_collapse": 0,
        "descriptor_lineages_after_collapse": 0,
        "founder_lineage_survival_curve_json": empty_counts,
        "lineage_alive_counts_by_generation_json": empty_curve,
        "lineage_effective_alive_by_generation_json": empty_curve,
        "lineage_top1_share_by_generation_json": empty_curve,
        "lineage_entropy_by_generation_json": empty_curve,
        "lineage_births_by_generation_json": empty_curve,
        "lineage_top1_birth_share_by_generation_json": empty_curve,
        "lineages_with_births_by_generation_json": empty_curve,
        "lineage_failed_births_by_generation_json": empty_curve,
        "lineage_birth_failure_context_json": empty_context,
        "lineage_deaths_by_generation_json": empty_curve,
        "lineage_top1_death_share_by_generation_json": empty_curve,
        "lineage_death_context_json": empty_context,
        "predator_deaths_by_lineage_json": empty_counts,
        "nonpredator_deaths_by_lineage_json": empty_counts,
        "lineage_patch_occupancy_context_json": empty_context,
        "lineage_bottleneck_events_json": empty_events,
    }


@dataclass
class LineageCollapseForensicsState:
    first_effective_collapse_gen: int = -1
    first_top1_dominance_gen: int = -1
    first_single_alive_gen: int = -1
    first_top1_birth_dominance_gen: int = -1
    last_multi_alive_gen: int = -1
    last_multi_birth_gen: int = -1
    minority_birth_first_gen: int = -1
    minority_birth_last_gen: int = -1
    minority_death_first_gen: int = -1
    minority_death_last_gen: int = -1
    minority_patch_first_gen: int = -1
    minority_patch_last_gen: int = -1
    alive_curve: List[Dict[str, Any]] = field(default_factory=list)
    effective_curve: List[Dict[str, Any]] = field(default_factory=list)
    top1_curve: List[Dict[str, Any]] = field(default_factory=list)
    entropy_curve: List[Dict[str, Any]] = field(default_factory=list)
    birth_curve: List[Dict[str, Any]] = field(default_factory=list)
    birth_top1_curve: List[Dict[str, Any]] = field(default_factory=list)
    birth_lineages_curve: List[Dict[str, Any]] = field(default_factory=list)
    death_curve: List[Dict[str, Any]] = field(default_factory=list)
    death_top1_curve: List[Dict[str, Any]] = field(default_factory=list)
    failed_birth_curve: List[Dict[str, Any]] = field(default_factory=list)
    founder_survival: Dict[str, List[Dict[str, int]]] = field(default_factory=dict)
    predator_deaths_by_lineage: Dict[str, int] = field(default_factory=dict)
    nonpredator_deaths_by_lineage: Dict[str, int] = field(default_factory=dict)
    birth_failure_context: Dict[str, int] = field(default_factory=dict)
    death_context: Dict[str, int] = field(default_factory=dict)
    patch_contexts: List[Dict[str, Any]] = field(default_factory=list)
    bottleneck_events: List[Dict[str, Any]] = field(default_factory=list)
    branch_candidates_before_collapse: int = 0
    branch_candidates_after_collapse: int = 0
    descriptor_lineages_before_collapse: int = 0
    descriptor_lineages_after_collapse: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "lineage_collapse_forensics_state_v1",
            "first_effective_collapse_gen": self.first_effective_collapse_gen,
            "first_top1_dominance_gen": self.first_top1_dominance_gen,
            "first_single_alive_gen": self.first_single_alive_gen,
            "first_top1_birth_dominance_gen": self.first_top1_birth_dominance_gen,
            "last_multi_alive_gen": self.last_multi_alive_gen,
            "last_multi_birth_gen": self.last_multi_birth_gen,
            "minority_birth_first_gen": self.minority_birth_first_gen,
            "minority_birth_last_gen": self.minority_birth_last_gen,
            "minority_death_first_gen": self.minority_death_first_gen,
            "minority_death_last_gen": self.minority_death_last_gen,
            "minority_patch_first_gen": self.minority_patch_first_gen,
            "minority_patch_last_gen": self.minority_patch_last_gen,
            "alive_curve": list(self.alive_curve),
            "effective_curve": list(self.effective_curve),
            "top1_curve": list(self.top1_curve),
            "entropy_curve": list(self.entropy_curve),
            "birth_curve": list(self.birth_curve),
            "birth_top1_curve": list(self.birth_top1_curve),
            "birth_lineages_curve": list(self.birth_lineages_curve),
            "death_curve": list(self.death_curve),
            "death_top1_curve": list(self.death_top1_curve),
            "failed_birth_curve": list(self.failed_birth_curve),
            "founder_survival": {str(k): list(v) for k, v in sorted(self.founder_survival.items())},
            "predator_deaths_by_lineage": dict(sorted(self.predator_deaths_by_lineage.items())),
            "nonpredator_deaths_by_lineage": dict(sorted(self.nonpredator_deaths_by_lineage.items())),
            "birth_failure_context": dict(sorted(self.birth_failure_context.items())),
            "death_context": dict(sorted(self.death_context.items())),
            "patch_contexts": list(self.patch_contexts),
            "bottleneck_events": list(self.bottleneck_events),
            "branch_candidates_before_collapse": self.branch_candidates_before_collapse,
            "branch_candidates_after_collapse": self.branch_candidates_after_collapse,
            "descriptor_lineages_before_collapse": self.descriptor_lineages_before_collapse,
            "descriptor_lineages_after_collapse": self.descriptor_lineages_after_collapse,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LineageCollapseForensicsState":
        state = cls()
        if not isinstance(payload, Mapping):
            return state
        for name in (
            "first_effective_collapse_gen",
            "first_top1_dominance_gen",
            "first_single_alive_gen",
            "first_top1_birth_dominance_gen",
            "last_multi_alive_gen",
            "last_multi_birth_gen",
            "minority_birth_first_gen",
            "minority_birth_last_gen",
            "minority_death_first_gen",
            "minority_death_last_gen",
            "minority_patch_first_gen",
            "minority_patch_last_gen",
            "branch_candidates_before_collapse",
            "branch_candidates_after_collapse",
            "descriptor_lineages_before_collapse",
            "descriptor_lineages_after_collapse",
        ):
            setattr(state, name, _safe_int(payload.get(name, getattr(state, name)), getattr(state, name)))
        for name in (
            "alive_curve",
            "effective_curve",
            "top1_curve",
            "entropy_curve",
            "birth_curve",
            "birth_top1_curve",
            "birth_lineages_curve",
            "death_curve",
            "death_top1_curve",
            "failed_birth_curve",
            "patch_contexts",
            "bottleneck_events",
        ):
            raw = payload.get(name, [])
            setattr(state, name, [dict(x) for x in list(raw or []) if isinstance(x, Mapping)])
        raw_survival = payload.get("founder_survival", {})
        if isinstance(raw_survival, Mapping):
            state.founder_survival = {
                str(k): [dict(x) for x in list(v or []) if isinstance(x, Mapping)]
                for k, v in raw_survival.items()
            }
        for name in ("predator_deaths_by_lineage", "nonpredator_deaths_by_lineage", "birth_failure_context", "death_context"):
            raw = payload.get(name, {})
            if isinstance(raw, Mapping):
                setattr(state, name, {str(k): _safe_int(v, 0) for k, v in raw.items()})
        return state


def _enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "lineage_collapse_forensics_enable", False)) if sim_cfg is not None else False


def validate_lineage_collapse_forensics_config(sim_cfg: Any) -> None:
    if not _enabled(sim_cfg):
        return
    version = _safe_int(getattr(sim_cfg, "lineage_collapse_forensics_schema_version", SCHEMA_VERSION), SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(f"lineage_collapse_forensics_schema_version must be {SCHEMA_VERSION}, got {version}")
    if _safe_float(getattr(sim_cfg, "lineage_collapse_effective_threshold", 1.5), 1.5) <= 0.0:
        raise ValueError("lineage_collapse_effective_threshold must be positive")
    top1 = _safe_float(getattr(sim_cfg, "lineage_collapse_top1_share_threshold", 0.95), 0.95)
    if top1 <= 0.0 or top1 > 1.0:
        raise ValueError("lineage_collapse_top1_share_threshold must be in (0, 1]")
    minority = _safe_float(getattr(sim_cfg, "lineage_collapse_minority_share_threshold", 0.05), 0.05)
    if minority < 0.0 or minority > 1.0:
        raise ValueError("lineage_collapse_minority_share_threshold must be in [0, 1]")
    for name in ("lineage_collapse_patch_context_top_n", "lineage_collapse_json_top_n"):
        if _safe_int(getattr(sim_cfg, name, 8), 8) <= 0:
            raise ValueError(f"{name} must be positive")


def _lineage_id_for_agent(agent: Any) -> str:
    aid = _safe_int(getattr(agent, "agent_id", 0), 0)
    stats = normalize_lineage_stats(getattr(agent, "stats", None), aid, generation=0, tick=0)
    if stats is None:
        return format_lineage_id(aid)
    return str(getattr(stats, "lineage_id", "") or format_lineage_id(aid))


def _lineage_id_from_context(row: Mapping[str, Any]) -> str:
    raw = row.get("lineage_id", "")
    if isinstance(raw, str) and raw.startswith("L"):
        return raw
    lineage_int = _safe_int(_safe_float(raw, 0.0), 0)
    return format_lineage_id(lineage_int) if lineage_int > 0 else ""


def _counts_from_agents(agents: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for agent in list(agents or []):
        if not bool(getattr(agent, "alive", True)):
            continue
        lid = _lineage_id_for_agent(agent)
        if lid:
            counts[lid] = counts.get(lid, 0) + 1
    return dict(sorted(counts.items()))


def _counts_from_contexts(contexts: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in list(contexts or []):
        if not isinstance(row, Mapping):
            continue
        lid = _lineage_id_from_context(row)
        if lid:
            counts[lid] = counts.get(lid, 0) + 1
    return dict(sorted(counts.items()))


def _shares(counts: Mapping[str, int]) -> Dict[str, float]:
    total = float(sum(max(0, int(v)) for v in counts.values()))
    if total <= 0.0:
        return {"entropy": 0.0, "effective": 0.0, "top1": 0.0}
    vals = [float(v) / total for _, v in sorted(counts.items()) if int(v) > 0]
    entropy = -sum(p * math.log(p) for p in vals if p > 0.0)
    return {
        "entropy": float(entropy),
        "effective": float(math.exp(entropy)) if entropy > 0.0 else 1.0,
        "top1": float(max(vals)) if vals else 0.0,
    }


def _top_counts(counts: Mapping[str, int], top_n: int) -> List[Dict[str, Any]]:
    rows = [{"lineage_id": str(k), "count": int(v)} for k, v in sorted(counts.items()) if int(v) > 0]
    rows.sort(key=lambda item: (-int(item["count"]), str(item["lineage_id"])))
    return rows[: max(1, int(top_n))]


def _append_curve(curve: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
    gen = _safe_int(row.get("generation", -1), -1)
    curve[:] = [x for x in curve if _safe_int(x.get("generation", -1), -1) != gen]
    curve.append(dict(row))
    curve.sort(key=lambda x: _safe_int(x.get("generation", 0), 0))


def _decode_json_list(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [dict(x) for x in raw if isinstance(x, Mapping)]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        decoded = json.loads(raw)
    except Exception:
        return []
    if isinstance(decoded, list):
        return [dict(x) for x in decoded if isinstance(x, Mapping)]
    return []


def _add_event(state: LineageCollapseForensicsState, generation: int, event: str, **payload: Any) -> None:
    row = {"event": str(event), "generation": int(generation), **payload}
    key = (int(generation), str(event), str(payload.get("lineage_id", "")))
    for existing in state.bottleneck_events:
        existing_key = (
            _safe_int(existing.get("generation", 0), 0),
            str(existing.get("event", "")),
            str(existing.get("lineage_id", "")),
        )
        if existing_key == key:
            return
    state.bottleneck_events.append(row)
    state.bottleneck_events.sort(
        key=lambda x: (_safe_int(x.get("generation", 0), 0), str(x.get("event", "")), str(x.get("lineage_id", "")))
    )


def _window(first: int, last: int) -> int:
    if int(first) < 0 or int(last) < 0:
        return -1
    return int(last) - int(first) + 1


def _curve_payload(curve: List[Dict[str, Any]]) -> str:
    return _stable_json({"schema_version": SCHEMA_VERSION, "generations": list(curve)})


def _counts_payload(counts: Mapping[str, int]) -> str:
    return _stable_json(
        {
            "schema_version": SCHEMA_VERSION,
            "lineages": [
                {"lineage_id": str(k), "count": int(v)}
                for k, v in sorted(counts.items())
                if int(v) > 0
            ],
        }
    )


def _survival_payload(state: LineageCollapseForensicsState) -> str:
    return _stable_json(
        {
            "schema_version": SCHEMA_VERSION,
            "lineages": [
                {"lineage_id": lid, "survival": list(state.founder_survival[lid])}
                for lid in sorted(state.founder_survival.keys())
            ],
        }
    )


def _context_payload(rows: List[Dict[str, Any]]) -> str:
    return _stable_json({"schema_version": SCHEMA_VERSION, "contexts": list(rows)})


def _context_counts_payload(counts: Mapping[str, int]) -> str:
    return _stable_json(
        {
            "schema_version": SCHEMA_VERSION,
            "contexts": [
                {"context": str(k), "count": int(v)}
                for k, v in sorted(counts.items())
                if int(v) > 0
            ],
        }
    )


def compute_lineage_collapse_forensics(
    *,
    generation: int,
    agents: Iterable[Any],
    sim_cfg: Any = None,
    state: Optional[LineageCollapseForensicsState] = None,
    birth_contexts: Iterable[Mapping[str, Any]] = (),
    death_contexts: Iterable[Mapping[str, Any]] = (),
    lineage_strategy_descriptors_json: Any = "[]",
    lineage_branch_candidates_json: Any = "[]",
    failed_births: int = 0,
    birth_failure_counts: Optional[Mapping[str, int]] = None,
) -> tuple[Dict[str, Any], LineageCollapseForensicsState]:
    st = state if state is not None else LineageCollapseForensicsState()
    if not _enabled(sim_cfg):
        return empty_lineage_forensics_metrics(), st
    validate_lineage_collapse_forensics_config(sim_cfg)

    gen = _safe_int(generation, 0)
    top_n = max(1, _safe_int(getattr(sim_cfg, "lineage_collapse_json_top_n", 8), 8))
    patch_top_n = max(1, _safe_int(getattr(sim_cfg, "lineage_collapse_patch_context_top_n", 8), 8))
    effective_threshold = _safe_float(getattr(sim_cfg, "lineage_collapse_effective_threshold", 1.5), 1.5)
    top1_threshold = _safe_float(getattr(sim_cfg, "lineage_collapse_top1_share_threshold", 0.95), 0.95)
    minority_threshold = _safe_float(getattr(sim_cfg, "lineage_collapse_minority_share_threshold", 0.05), 0.05)

    alive_counts = _counts_from_agents(agents)
    alive_stats = _shares(alive_counts)
    active_count = sum(1 for v in alive_counts.values() if int(v) > 0)
    if active_count > 1:
        st.last_multi_alive_gen = gen
    if alive_stats["effective"] > 0.0 and alive_stats["effective"] <= effective_threshold and st.first_effective_collapse_gen < 0:
        st.first_effective_collapse_gen = gen
        _add_event(st, gen, "first_effective_lineage_collapse", effective_alive_lineages=alive_stats["effective"])
    if alive_stats["top1"] >= top1_threshold and active_count > 0 and st.first_top1_dominance_gen < 0:
        st.first_top1_dominance_gen = gen
        _add_event(st, gen, "first_top1_dominance", top1_alive_share=alive_stats["top1"])
    if active_count == 1 and st.first_single_alive_gen < 0:
        st.first_single_alive_gen = gen
        _add_event(st, gen, "first_single_alive_lineage")

    _append_curve(st.alive_curve, {"generation": gen, "lineages": _top_counts(alive_counts, top_n)})
    _append_curve(st.effective_curve, {"generation": gen, "value": alive_stats["effective"]})
    _append_curve(st.top1_curve, {"generation": gen, "value": alive_stats["top1"]})
    _append_curve(st.entropy_curve, {"generation": gen, "value": alive_stats["entropy"]})
    for lid, count in sorted(alive_counts.items()):
        st.founder_survival.setdefault(lid, [])
        st.founder_survival[lid] = [x for x in st.founder_survival[lid] if _safe_int(x.get("generation", -1), -1) != gen]
        st.founder_survival[lid].append({"generation": gen, "alive": int(count)})
        st.founder_survival[lid].sort(key=lambda x: _safe_int(x.get("generation", 0), 0))

    birth_counts = _counts_from_contexts(birth_contexts)
    birth_stats = _shares(birth_counts)
    birth_lineage_count = sum(1 for v in birth_counts.values() if int(v) > 0)
    if birth_lineage_count > 1:
        st.last_multi_birth_gen = gen
        if st.minority_birth_first_gen < 0:
            st.minority_birth_first_gen = gen
        st.minority_birth_last_gen = gen
    if birth_stats["top1"] >= top1_threshold and birth_lineage_count > 0 and st.first_top1_birth_dominance_gen < 0:
        st.first_top1_birth_dominance_gen = gen
        _add_event(st, gen, "first_top1_birth_dominance", top1_birth_share=birth_stats["top1"])
    _append_curve(st.birth_curve, {"generation": gen, "lineages": _top_counts(birth_counts, top_n)})
    _append_curve(st.birth_top1_curve, {"generation": gen, "value": birth_stats["top1"]})
    _append_curve(st.birth_lineages_curve, {"generation": gen, "value": birth_lineage_count})

    death_counts = _counts_from_contexts(death_contexts)
    death_stats = _shares(death_counts)
    if sum(1 for v in death_counts.values() if int(v) > 0) > 1:
        if st.minority_death_first_gen < 0:
            st.minority_death_first_gen = gen
        st.minority_death_last_gen = gen
    _append_curve(st.death_curve, {"generation": gen, "lineages": _top_counts(death_counts, top_n)})
    _append_curve(st.death_top1_curve, {"generation": gen, "value": death_stats["top1"]})

    death_context_counts: Dict[str, int] = {}
    for row in list(death_contexts or []):
        if not isinstance(row, Mapping):
            continue
        lid = _lineage_id_from_context(row)
        if not lid:
            continue
        cause = str(row.get("death_cause", "unknown") or "unknown")
        death_context_counts[cause] = death_context_counts.get(cause, 0) + 1
        if cause == "predator":
            st.predator_deaths_by_lineage[lid] = int(st.predator_deaths_by_lineage.get(lid, 0)) + 1
        else:
            st.nonpredator_deaths_by_lineage[lid] = int(st.nonpredator_deaths_by_lineage.get(lid, 0)) + 1
    for key, value in death_context_counts.items():
        st.death_context[key] = int(st.death_context.get(key, 0)) + int(value)

    failure_counts: Dict[str, int] = {}
    for key, value in dict(birth_failure_counts or {}).items():
        count = _safe_int(value, 0)
        if count > 0:
            failure_counts[str(key)] = int(count)
    for key, value in sorted(failure_counts.items()):
        st.birth_failure_context[key] = int(st.birth_failure_context.get(key, 0)) + int(value)
    failed_total = sum(failure_counts.values()) if failure_counts else _safe_int(failed_births, 0)
    _append_curve(st.failed_birth_curve, {"generation": gen, "failed_births": int(max(0, failed_total))})

    descriptors = _decode_json_list(lineage_strategy_descriptors_json)
    descriptor_lineages = sum(1 for row in descriptors if _safe_float(row.get("lineage_alive_count", 0.0), 0.0) > 0.0)
    before_collapse = st.first_effective_collapse_gen < 0 or gen <= st.first_effective_collapse_gen
    if before_collapse:
        st.descriptor_lineages_before_collapse += int(descriptor_lineages)
    else:
        st.descriptor_lineages_after_collapse += int(descriptor_lineages)

    patch_rows: List[Dict[str, Any]] = []
    for row in descriptors:
        lineage_id = _lineage_id_from_context({"lineage_id": row.get("lineage_id", "")})
        alive = _safe_float(row.get("lineage_alive_count", 0.0), 0.0)
        if not lineage_id or alive <= 0.0:
            continue
        patch_rows.append(
            {
                "generation": gen,
                "lineage_id": lineage_id,
                "alive": alive,
                "food": _safe_float(row.get("lineage_patch_food_amount_mean", 0.0), 0.0),
                "waste": _safe_float(row.get("lineage_patch_waste_mean", 0.0), 0.0),
                "density": _safe_float(row.get("lineage_patch_density_mean", 0.0), 0.0),
                "shelter": _safe_float(row.get("lineage_patch_shelter_mean", 0.0), 0.0),
                "recovery": _safe_float(row.get("lineage_patch_recovery_activity_mean", 0.0), 0.0),
            }
        )
    patch_rows.sort(key=lambda item: (-float(item["alive"]), str(item["lineage_id"])))
    st.patch_contexts = [x for x in st.patch_contexts if _safe_int(x.get("generation", -1), -1) != gen]
    st.patch_contexts.extend(patch_rows[:patch_top_n])
    st.patch_contexts.sort(key=lambda x: (_safe_int(x.get("generation", 0), 0), str(x.get("lineage_id", ""))))
    if len(patch_rows) > 1:
        if st.minority_patch_first_gen < 0:
            st.minority_patch_first_gen = gen
        st.minority_patch_last_gen = gen
    total_patch_alive = sum(float(row["alive"]) for row in patch_rows)
    top_patch_alive = max((float(row["alive"]) for row in patch_rows), default=0.0)
    top_patch_share = float(top_patch_alive / total_patch_alive) if total_patch_alive > 0.0 else 0.0
    distinct_patch_contexts = sum(1 for row in patch_rows if float(row["alive"]) / max(1.0, total_patch_alive) >= minority_threshold)

    candidates = _decode_json_list(lineage_branch_candidates_json)
    if before_collapse:
        st.branch_candidates_before_collapse += len(candidates)
    else:
        st.branch_candidates_after_collapse += len(candidates)

    if st.last_multi_alive_gen >= 0:
        _add_event(st, st.last_multi_alive_gen, "last_generation_with_more_than_one_lineage_alive")
    if st.last_multi_birth_gen >= 0:
        _add_event(st, st.last_multi_birth_gen, "last_generation_with_more_than_one_lineage_reproducing")

    return {
        "lineage_forensics_enabled": 1,
        "lineage_forensics_schema_version": SCHEMA_VERSION,
        "first_effective_lineage_collapse_generation": int(st.first_effective_collapse_gen),
        "first_top1_dominance_generation": int(st.first_top1_dominance_gen),
        "first_single_alive_lineage_generation": int(st.first_single_alive_gen),
        "first_top1_birth_dominance_generation": int(st.first_top1_birth_dominance_gen),
        "final_effective_alive_lineages": float(alive_stats["effective"]),
        "final_alive_lineage_entropy": float(alive_stats["entropy"]),
        "final_top1_alive_share": float(alive_stats["top1"]),
        "collapse_generation_observed": int(st.first_effective_collapse_gen >= 0 or st.first_top1_dominance_gen >= 0 or st.first_single_alive_gen >= 0),
        "minority_lineage_birth_window": _window(st.minority_birth_first_gen, st.minority_birth_last_gen),
        "minority_lineage_death_window": _window(st.minority_death_first_gen, st.minority_death_last_gen),
        "minority_lineage_patch_occupancy_window": _window(st.minority_patch_first_gen, st.minority_patch_last_gen),
        "final_lineages_with_births": int(birth_lineage_count),
        "top_lineage_patch_occupancy_share": float(top_patch_share),
        "lineages_with_distinct_patch_contexts": int(distinct_patch_contexts),
        "branch_candidates_before_collapse": int(st.branch_candidates_before_collapse),
        "branch_candidates_after_collapse": int(st.branch_candidates_after_collapse),
        "descriptor_lineages_before_collapse": int(st.descriptor_lineages_before_collapse),
        "descriptor_lineages_after_collapse": int(st.descriptor_lineages_after_collapse),
        "founder_lineage_survival_curve_json": _survival_payload(st),
        "lineage_alive_counts_by_generation_json": _curve_payload(st.alive_curve),
        "lineage_effective_alive_by_generation_json": _curve_payload(st.effective_curve),
        "lineage_top1_share_by_generation_json": _curve_payload(st.top1_curve),
        "lineage_entropy_by_generation_json": _curve_payload(st.entropy_curve),
        "lineage_births_by_generation_json": _curve_payload(st.birth_curve),
        "lineage_top1_birth_share_by_generation_json": _curve_payload(st.birth_top1_curve),
        "lineages_with_births_by_generation_json": _curve_payload(st.birth_lineages_curve),
        "lineage_failed_births_by_generation_json": _curve_payload(st.failed_birth_curve),
        "lineage_birth_failure_context_json": _context_counts_payload(st.birth_failure_context),
        "lineage_deaths_by_generation_json": _curve_payload(st.death_curve),
        "lineage_top1_death_share_by_generation_json": _curve_payload(st.death_top1_curve),
        "lineage_death_context_json": _context_counts_payload(st.death_context),
        "predator_deaths_by_lineage_json": _counts_payload(st.predator_deaths_by_lineage),
        "nonpredator_deaths_by_lineage_json": _counts_payload(st.nonpredator_deaths_by_lineage),
        "lineage_patch_occupancy_context_json": _context_payload(st.patch_contexts),
        "lineage_bottleneck_events_json": _stable_json({"schema_version": SCHEMA_VERSION, "events": st.bottleneck_events}),
    }, st
