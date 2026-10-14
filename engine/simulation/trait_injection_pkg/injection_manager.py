from __future__ import annotations

"""simulation/trait_injection_pkg/injection_manager.py

Phase 5B: Trait injection runtime application.

High-level behavior
-------------------
- Trait injection is driven ONLY by ControlPlane command events.
- Injection applies ONLY at SafeBoundary-approved phases:
    * post-reproduction-pre-selection
    * post-selection-pre-mutation
- Idempotency is keyed by (generation_index, apply_phase, command_uuid).

Implementation notes
--------------------
- To avoid checkpoint format changes, we derive active injections by scanning the
  ControlPlane event log (run_dir/events/command_events.jsonl).
- Ordering of multiple injections is deterministic using the event record's
  ordering fields: (target_generation, submission_sequence_number, command_uuid).

No RNG. No wall clock. No changes to selection logic.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os

from simulation.control_plane_pkg.event_log import events_path, iter_events
from simulation.trait_injection_pkg.cost_model import trait_injection_cost_model
from simulation.trait_injection_pkg.schema_v2 import (
    ALLOWED_APPLY_PHASES,
    normalize_duration_generations,
    validate_trait_injection_v2,
)


@dataclass(frozen=True)
class TraitInjectionEvent:
    """Normalized representation of an applied injection command."""

    command_uuid: str
    target_generation: int
    submission_sequence_number: int
    apply_phase: str
    duration_generations: int
    locus_tags: Tuple[str, ...]
    allele_values: Dict[str, Any]
    target_agent_ids: Optional[Tuple[int, ...]]

    def is_active_for(self, generation_index: int) -> bool:
        gi = int(generation_index)
        start = int(self.target_generation)
        dur = int(self.duration_generations)
        return (gi >= start) and (gi <= (start + dur - 1))


def _coerce_value(existing: Any, new_value: Any) -> Any:
    """Coerce new_value to the type of existing when possible."""
    try:
        if isinstance(existing, bool):
            return bool(new_value)
        if isinstance(existing, int) and not isinstance(existing, bool):
            return int(new_value)
        if isinstance(existing, float):
            return float(new_value)
    except Exception:
        return new_value
    return new_value


def _resolve_frozen_apply_phase(sim_cfg: Any) -> Optional[str]:
    """Return campaign-frozen apply_phase from sim_cfg, if available."""
    try:
        ap = str(getattr(sim_cfg, "trait_injection_apply_phase", "") or "")
        return ap or None
    except Exception:
        return None


def _applied_cache(sim: Any) -> set:
    """Get or create the in-memory idempotency cache on sim."""
    try:
        cache = getattr(sim, "_trait_injection_applied_cache", None)
        if isinstance(cache, set):
            return cache
    except Exception:
        pass
    try:
        cache = set()
        setattr(sim, "_trait_injection_applied_cache", cache)
        return cache
    except Exception:
        return set()


def _normalize_event_to_injection(ev: Dict[str, Any], *, frozen_apply_phase: Optional[str]) -> Optional[TraitInjectionEvent]:
    """Convert a ControlPlane event record into a TraitInjectionEvent if applicable."""
    try:
        payload = ev.get("command_payload", {}) or {}
        if not isinstance(payload, dict):
            return None
        if str(payload.get("command_type", "") or "") != "TraitInjectionCommand":
            return None

        ti = payload.get("trait_injection", {})
        if not isinstance(ti, dict):
            return None

        ok, _reason = validate_trait_injection_v2(ti, frozen_apply_phase=frozen_apply_phase)
        if not ok:
            return None

        apply_phase = str(ti.get("apply_phase", "") or "")
        dur = normalize_duration_generations(ti)

        locus_tags_raw = ti.get("locus_tags", []) or []
        locus_tags = tuple(str(x) for x in locus_tags_raw if str(x))

        allele_values = ti.get("allele_values", {}) or {}
        if not isinstance(allele_values, dict):
            allele_values = {}

        target_ids: Optional[Tuple[int, ...]] = None
        if "target_agent_ids" in ti and ti.get("target_agent_ids") is not None:
            ids = ti.get("target_agent_ids")
            if isinstance(ids, list):
                target_ids = tuple(sorted(int(x) for x in ids))

        cu = str(payload.get("command_uuid", "") or ev.get("command_uuid", "") or "")
        tg = int(ev.get("target_generation", payload.get("target_generation", 0) or 0))
        ssn = int(ev.get("submission_sequence_number", 0) or 0)

        return TraitInjectionEvent(
            command_uuid=cu,
            target_generation=tg,
            submission_sequence_number=ssn,
            apply_phase=apply_phase,
            duration_generations=int(dur),
            locus_tags=locus_tags,
            allele_values=dict(allele_values),
            target_agent_ids=target_ids,
        )
    except Exception:
        return None


def _iter_injection_events(run_dir: str, *, frozen_apply_phase: Optional[str]) -> List[TraitInjectionEvent]:
    """Load and normalize all applied injection events from the ControlPlane event log."""
    p = events_path(run_dir)
    if not os.path.exists(p):
        return []

    out: List[TraitInjectionEvent] = []
    for ev in iter_events(run_dir):
        inj = _normalize_event_to_injection(ev, frozen_apply_phase=frozen_apply_phase)
        if inj is not None and inj.command_uuid:
            out.append(inj)

    out.sort(key=lambda e: (int(e.target_generation), int(e.submission_sequence_number), str(e.command_uuid)))
    return out


def _apply_injection_to_population(pop: Any, inj: TraitInjectionEvent) -> None:
    """Apply one injection event to the population (genome patching only)."""
    targets: List[Any] = []

    if inj.target_agent_ids is not None and len(inj.target_agent_ids) > 0:
        for aid in inj.target_agent_ids:
            try:
                a = pop.agents.get(int(aid))
            except Exception:
                a = None
            if a is None:
                continue
            try:
                if hasattr(a, "alive") and not bool(getattr(a, "alive")):
                    continue
            except Exception:
                pass
            targets.append(a)
    else:
        if hasattr(pop, "alive_agents"):
            try:
                targets = list(pop.alive_agents())
            except Exception:
                targets = []
        else:
            try:
                targets = [a for a in (pop.agents or {}).values()]
            except Exception:
                targets = []

    try:
        targets.sort(key=lambda a: int(getattr(a, "agent_id", 0)))
    except Exception:
        pass

    for a in targets:
        g = getattr(a, "genome", None)
        if g is None:
            continue
        for locus in inj.locus_tags:
            if locus not in inj.allele_values:
                continue
            new_val = inj.allele_values.get(locus)
            try:
                existing = getattr(g, locus)
                coerced = _coerce_value(existing, new_val)
                setattr(g, locus, coerced)
            except Exception:
                continue


def apply_trait_injections(*, sim: Any, generation_index: int, apply_phase: str) -> List[str]:
    """Apply all active trait injections for a given (generation, apply_phase).

    Parameters
    ----------
    sim:
        Simulation-like object. Must provide:
          - run_dir: str
          - pop: population with .agents dict and preferably .alive_agents()
          - sim_cfg: config object containing campaign-frozen apply_phase
    generation_index:
        Current generation index.
    apply_phase:
        One of the allowed phases.

    Returns
    -------
    list[str]
        command_uuids applied in this call (deterministic order).
    """
    ap = str(apply_phase or "")
    if ap not in ALLOWED_APPLY_PHASES:
        return []

    run_dir = str(getattr(sim, "run_dir", "") or "")
    if not run_dir:
        return []

    if not os.path.exists(events_path(run_dir)):
        return []

    pop = getattr(sim, "pop", None)
    if pop is None:
        return []

    sim_cfg = getattr(sim, "sim_cfg", None)
    frozen = _resolve_frozen_apply_phase(sim_cfg)

    injections = _iter_injection_events(run_dir, frozen_apply_phase=frozen)
    if not injections:
        return []

    gi = int(generation_index)
    cache = _applied_cache(sim)

    applied: List[str] = []

    for inj in injections:
        if inj.apply_phase != ap:
            continue
        if not inj.is_active_for(gi):
            continue

        key = (int(gi), str(ap), str(inj.command_uuid))
        if key in cache:
            continue

        _apply_injection_to_population(pop, inj)

        try:
            cache.add(key)
        except Exception:
            pass

        applied.append(str(inj.command_uuid))

        # In-memory audit hook (no disk writes).
        try:
            cm = trait_injection_cost_model(inj.duration_generations, inj.apply_phase)
            last = getattr(sim, "_trait_injection_last_cost_models", None)
            if not isinstance(last, dict):
                last = {}
            last[str(inj.command_uuid)] = cm
            setattr(sim, "_trait_injection_last_cost_models", last)
        except Exception:
            pass

    return applied
