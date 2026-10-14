"""Read-only generation-boundary recorder for Scientific Chronicle V1."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

from simulation.chronicle.config import ChronicleConfig, storage_preflight
from simulation.chronicle.episodes import detect_episodes
from simulation.chronicle.render import entries_from_events, terminal_lines
from simulation.chronicle.schema import (
    CONTEXT_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    assert_unique_events,
    canonical_payload,
    event_order_key,
    make_event,
    stable_hash,
)
from simulation.chronicle.storage import (
    ARTIFACT_NAMES,
    artifact_inventory,
    write_json,
    write_jsonl,
)


STATE_SCHEMA_VERSION = "eiel_scientific_chronicle_state_v1"
RECORDER_VERSION = "scientific_chronicle_v1"

_PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "background": 4}
_CROSSING_TYPES = frozenset(
    {
        "lineage_appearance",
        "lineage_dominance_crossing",
        "lineage_leadership_change",
        "lineage_rapid_share_increase",
        "lineage_rapid_share_decline",
        "lineage_near_extinction",
        "lineage_extinction",
        "regional_food_depletion",
        "regional_food_recovery",
        "crowding_hotspot",
        "waste_hotspot",
        "population_bottleneck",
    }
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    return result if math.isfinite(result) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _source_commit() -> str:
    for candidate in Path(__file__).resolve().parents:
        if not (candidate / ".git").exists():
            continue
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=candidate,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            break
        commit = result.stdout.strip()
        if commit:
            return commit
    return "unknown"


def _lineage_id(agent: Any) -> str:
    agent_id = _safe_int(getattr(agent, "agent_id", 0), 0)
    stats = getattr(agent, "stats", None)
    raw = getattr(stats, "lineage_id", "") if stats is not None else ""
    if raw:
        return str(raw)
    founder = _safe_int(
        getattr(stats, "founder_id", agent_id) if stats is not None else agent_id,
        agent_id,
    )
    return f"L{founder:08d}"


def _context_lineage(context: Mapping[str, Any]) -> str:
    raw = context.get("biological_lineage_id") or context.get("lineage_id")
    if isinstance(raw, str) and raw:
        return raw
    numeric = _safe_int(raw, 0)
    return f"L{numeric:08d}" if numeric else ""


def _context_coordinates(context: Mapping[str, Any]) -> tuple[int, int] | None:
    if "y" not in context or "x" not in context:
        return None
    return (_safe_int(context.get("y"), 0), _safe_int(context.get("x"), 0))


def _json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        decoded = json.loads(value)
    except Exception:
        return []
    return [dict(item) for item in decoded if isinstance(item, Mapping)] if isinstance(decoded, list) else []


def _context_windows(
    events: list[Mapping[str, Any]],
    *,
    before: int,
    after: int,
    max_per_generation: int,
) -> list[dict[str, Any]]:
    if not events or max_per_generation <= 0:
        return []
    triggers = [
        index
        for index, event in enumerate(events)
        if str(event.get("live_priority")) in {"critical", "high"} or int(event.get("scientific_importance", 0)) >= 4
    ]
    windows: list[dict[str, Any]] = []
    generation_counts: Counter[int] = Counter()
    last_end = -1
    for index in triggers:
        generation = int(events[index].get("generation", 0))
        if generation_counts[generation] >= int(max_per_generation):
            continue
        start = max(0, index - int(before))
        end = min(len(events) - 1, index + int(after))
        if windows and start <= last_end:
            window = windows[-1]
            prior_ids = list(window["source_event_ids"])
            window["end_generation"] = int(events[end].get("generation", 0))
            window["end_step"] = int(events[end].get("step", 0))
            current_ids = [str(event["event_id"]) for event in events[max(0, index - int(before)) : min(len(events), index + int(after) + 1)]]
            window["source_event_ids"] = sorted(
                set(prior_ids + current_ids),
                key=lambda event_id: next(i for i, event in enumerate(events) if str(event["event_id"]) == event_id),
            )
            window["trigger_event_ids"] = sorted(set(window["trigger_event_ids"] + [str(events[index]["event_id"])]))
            last_end = max(last_end, end)
            continue
        source_ids = [str(event["event_id"]) for event in events[start : end + 1]]
        trigger_id = str(events[index]["event_id"])
        window_id = "ctx_" + stable_hash(
            "eiel-event-context-window-v1",
            {"trigger_event_id": trigger_id, "before": before, "after": after},
        )
        windows.append(
            {
                "schema_version": CONTEXT_SCHEMA_VERSION,
                "context_window_id": window_id,
                "run_id": str(events[index].get("run_id", "")),
                "start_generation": int(events[start].get("generation", 0)),
                "start_step": int(events[start].get("step", 0)),
                "end_generation": int(events[end].get("generation", 0)),
                "end_step": int(events[end].get("step", 0)),
                "trigger_event_ids": [trigger_id],
                "source_event_ids": source_ids,
                "coverage": "compact_formal_event_context",
                "limitations": ["This window contains formal events, not full policy tensors or world state."],
            }
        )
        generation_counts[generation] += 1
        last_end = end
    return canonical_payload(windows)


class ScientificChronicleRecorder:
    """A passive observer. No recorder method is permitted to consume simulation RNG."""

    def __init__(self, sim: Any) -> None:
        self.config = ChronicleConfig.from_sim_cfg(getattr(sim, "sim_cfg", None))
        if not self.config.enabled:
            raise ValueError("ScientificChronicleRecorder requires chronicle_enable=true and a non-off profile")
        sim_cfg = getattr(sim, "sim_cfg", None)
        run_dir = Path(str(getattr(sim, "run_dir", "")))
        configured_run_id = str(getattr(sim_cfg, "chronicle_run_id", "") or "").strip()
        self.run_id = configured_run_id or run_dir.name or "run"
        self.campaign_id = str(getattr(sim_cfg, "campaign_id", "") or "") or None
        self.branch_id = str(getattr(sim_cfg, "branch_id", "") or "") or None
        self.root = run_dir / "chronicle"
        self.events: list[dict[str, Any]] = []
        self._seen_event_ids: set[str] = set()
        self._previous_metrics: dict[str, Any] = {}
        self._previous_lineage_counts: dict[str, int] = {}
        self._cooldown_last_generation: dict[str, int] = {}
        self._condition_streaks: dict[str, int] = {}
        self._food_depletion_observed = False
        self.suppressed_counts: Counter[str] = Counter()
        self.write_failures = 0
        self.disabled_after_write_failure = False
        self.last_emitted_generation = -1
        self.finalized = False
        self._source_completeness = "live_generation_boundary_complete"
        self.preflight = storage_preflight(
            self.config,
            generations=(
                _safe_int(
                    getattr(sim_cfg, "chronicle_planned_generations", 0),
                    0,
                )
                or _safe_int(getattr(sim_cfg, "generations", 0), 0)
            ),
        )
        self._source_commit = str(getattr(sim_cfg, "code_commit_hash", "") or os.environ.get("EIEL_GIT_COMMIT", "") or _source_commit())
        self.parent_checkpoint_id: str | None = None
        self._persist(finalize=False)

    @classmethod
    def maybe_create(cls, sim: Any) -> "ScientificChronicleRecorder | None":
        config = ChronicleConfig.from_sim_cfg(getattr(sim, "sim_cfg", None))
        return cls(sim) if config.enabled else None

    def _candidate(self, **values: Any) -> dict[str, Any]:
        defaults = {
            "event_time_scope": "generation",
            "subject_type": "run",
            "subject_id": self.run_id,
            "severity": "informational",
            "live_priority": "background",
            "scientific_importance": 1,
            "historical_importance": 1,
            "public_interest": 0,
            "visual_interest": 0,
            "novelty_class": "routine",
            "confidence_class": "measured",
            "causality_class": "not_assessed",
            "source_component": "runner.generation_summary",
            "source_record_refs": (),
            "agent_ids": (),
            "lineage_ids": (),
            "patch_coordinates": None,
            "region_id": None,
            "episode_hint": None,
            "measurements": {},
            "state_before_summary": None,
            "state_after_summary": None,
            "factual_summary_fields": {},
            "tags": (),
        }
        defaults.update(values)
        return defaults

    def _lineage_counts(self, agents: Iterable[Any]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for agent in sorted(
            list(agents or []),
            key=lambda item: _safe_int(getattr(item, "agent_id", 0), 0),
        ):
            counts[_lineage_id(agent)] += 1
        return dict(sorted(counts.items()))

    def _generation_candidates(
        self,
        *,
        generation: int,
        alive_agents: Iterable[Any],
        deaths: int,
        births: int,
        failed_births: int,
        cohort_size: int | None,
        cohort_survival_fraction: float | None,
        patch_metrics: Mapping[str, Any],
        birth_contexts: Iterable[Mapping[str, Any]],
        death_contexts: Iterable[Mapping[str, Any]],
        predator_contexts: Iterable[Mapping[str, Any]],
        wdl_contexts: Iterable[Mapping[str, Any]],
        native_lifecycle_rows: Iterable[Mapping[str, Any]],
        native_lifecycle_row_count: int | None,
        native_public_output_counts: Mapping[str, Any] | None,
        native_lineage_counts: Mapping[str, Any] | None,
        generation_source_component: str | None,
        generation_source_record_ref: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        metrics = dict(patch_metrics or {})
        alive_list = list(alive_agents or [])
        native_counts = dict(native_public_output_counts or {})
        native_authority = bool(native_counts)
        alive_count = _safe_int(native_counts.get("alive"), len(alive_list)) if native_authority else len(alive_list)
        lineage_detail_available = native_lineage_counts is not None or not native_authority
        lineage_counts = (
            {
                str(lineage): max(0, _safe_int(count, 0))
                for lineage, count in dict(native_lineage_counts or {}).items()
            }
            if native_lineage_counts is not None
            else self._lineage_counts(alive_list)
        )
        total_lineages = sum(lineage_counts.values())
        shares = {lineage: float(count) / float(total_lineages) if total_lineages else 0.0 for lineage, count in lineage_counts.items()}
        previous_total = sum(self._previous_lineage_counts.values())
        previous_shares = {lineage: float(count) / float(previous_total) if previous_total else 0.0 for lineage, count in self._previous_lineage_counts.items()}
        top_lineage = max(shares, key=lambda lineage: (shares[lineage], lineage), default="")
        previous_top = max(
            previous_shares,
            key=lambda lineage: (previous_shares[lineage], lineage),
            default="",
        )
        source_ref = generation_source_record_ref or f"metrics.csv:generation={int(generation)}"
        cohort_size_value = int(cohort_size) if cohort_size is not None else None
        survival_fraction = (
            _safe_float(cohort_survival_fraction, 1.0)
            if cohort_survival_fraction is not None
            else (float(alive_count) / float(cohort_size_value) if cohort_size_value is not None and cohort_size_value > 0 else None)
        )
        generation_measurements = {
            "alive": alive_count,
            "births": int(births),
            "deaths": int(deaths),
            "failed_births": int(failed_births),
            "cohort_size": cohort_size_value,
            "cohort_survival_fraction": survival_fraction,
            "population_change": (int(alive_count) - cohort_size_value if cohort_size_value is not None else None),
            "active_lineages": len(lineage_counts) if lineage_detail_available else None,
            "effective_lineages": (
                _safe_float(metrics.get("lineage_effective_alive_lineages"), 0.0)
                if lineage_detail_available
                else None
            ),
            "top_lineage_share": (
                max(shares.values(), default=0.0) if lineage_detail_available else None
            ),
            "lineage_entropy": (
                _safe_float(metrics.get("lineage_alive_share_entropy"), 0.0)
                if lineage_detail_available
                else None
            ),
            "food_mean": _safe_float(metrics.get("patch_food_amount_mean"), 0.0),
            "food_depleted_fraction": _safe_float(metrics.get("patch_food_depleted_frac"), 0.0),
            "occupied_waste_mean": _safe_float(metrics.get("occupied_patch_waste_mean"), 0.0),
            "crowding_hotspot_fraction": _safe_float(metrics.get("patch_density_hotspot_frac"), 0.0),
            "regrowth_suppressed_total": _safe_float(metrics.get("patch_regrowth_suppressed_total"), 0.0),
            "predator_events": _safe_int(metrics.get("jungle_predator_events"), 0),
            "predator_damage": _safe_float(metrics.get("jungle_predator_damage_total"), 0.0),
            "delayed_hazard_events": _safe_int(metrics.get("wdl_delayed_hazard_event_count"), 0),
            "recovery_positive_total": _safe_float(metrics.get("patch_recovery_positive_total"), 0.0),
            "native_lifecycle_authority": native_authority,
            "native_typed_lifecycle_rows": (
                int(native_lifecycle_row_count) if native_lifecycle_row_count is not None else len(list(native_lifecycle_rows or []))
            ),
        }
        if native_authority:
            for source_name, measurement_name in (
                ("birth_attempts", "native_birth_attempts"),
                ("capacity", "native_capacity"),
                ("free_slots", "native_free_slots"),
            ):
                if source_name in native_counts:
                    generation_measurements[measurement_name] = _safe_int(native_counts[source_name], 0)
        candidates = [
            self._candidate(
                phase_order=900,
                event_type="generation_summary",
                event_category="summary",
                measurements=generation_measurements,
                source_component=(generation_source_component or "runner.generation_summary"),
                source_record_refs=(source_ref,),
                scientific_importance=2,
                historical_importance=2,
                tags=("generation_summary", "website_ready"),
            )
        ]
        if int(generation) == 0 and str(self.preflight.get("risk_classification")) == "large":
            candidates.append(
                self._candidate(
                    phase_order=920,
                    event_type="chronicle_storage_budget",
                    event_category="integrity",
                    subject_type="run",
                    subject_id=self.run_id,
                    severity="minor",
                    live_priority="low",
                    scientific_importance=2,
                    historical_importance=2,
                    confidence_class="measured",
                    measurements={
                        "estimated_events": int(self.preflight.get("estimated_events", 0)),
                        "estimated_bytes": int(self.preflight.get("estimated_bytes", 0)),
                        "large_output_acknowledged": bool(self.config.allow_large_output),
                    },
                    source_component="chronicle.storage_preflight",
                    source_record_refs=("chronicle_manifest.json:preflight",),
                    tags=("integrity", "storage", "preflight"),
                )
            )

        if self.last_emitted_generation >= 0 and lineage_detail_available:
            for lineage in sorted(set(lineage_counts).difference(self._previous_lineage_counts)):
                candidates.append(
                    self._candidate(
                        phase_order=400,
                        event_type="lineage_appearance",
                        event_category="lineage",
                        subject_type="lineage",
                        subject_id=lineage,
                        lineage_ids=(lineage,),
                        severity="moderate",
                        live_priority="medium",
                        scientific_importance=3,
                        historical_importance=4,
                        novelty_class="new_lineage",
                        measurements={
                            "alive_count": lineage_counts[lineage],
                            "alive_share": shares[lineage],
                        },
                        source_record_refs=(source_ref,),
                        episode_hint="lineage_expansion",
                        tags=("lineage", "appearance"),
                    )
                )
            for lineage in sorted(set(self._previous_lineage_counts).difference(lineage_counts)):
                candidates.append(
                    self._candidate(
                        phase_order=430,
                        event_type="lineage_extinction",
                        event_category="lineage",
                        subject_type="lineage",
                        subject_id=lineage,
                        lineage_ids=(lineage,),
                        severity="critical",
                        live_priority="critical",
                        scientific_importance=5,
                        historical_importance=5,
                        public_interest=3,
                        visual_interest=3,
                        novelty_class="terminal_lineage_event",
                        measurements={
                            "previous_alive_count": self._previous_lineage_counts[lineage],
                            "alive_count": 0,
                        },
                        source_record_refs=(source_ref,),
                        episode_hint="lineage_extinction_sequence",
                        tags=("lineage", "extinction"),
                    )
                )
            for lineage in sorted(set(lineage_counts).intersection(self._previous_lineage_counts)):
                before_share = previous_shares.get(lineage, 0.0)
                after_share = shares.get(lineage, 0.0)
                delta = after_share - before_share
                if before_share < self.config.dominance_threshold <= after_share:
                    candidates.append(
                        self._candidate(
                            phase_order=410,
                            event_type="lineage_dominance_crossing",
                            event_category="lineage",
                            subject_type="lineage",
                            subject_id=lineage,
                            lineage_ids=(lineage,),
                            severity="major",
                            live_priority="high",
                            scientific_importance=4,
                            historical_importance=4,
                            measurements={
                                "share_before": before_share,
                                "share_after": after_share,
                                "threshold": self.config.dominance_threshold,
                            },
                            source_record_refs=(source_ref,),
                            episode_hint="lineage_expansion",
                            tags=("lineage", "dominance_threshold"),
                        )
                    )
                if delta >= self.config.rapid_lineage_share_delta:
                    candidates.append(
                        self._candidate(
                            phase_order=415,
                            event_type="lineage_rapid_share_increase",
                            event_category="lineage",
                            subject_type="lineage",
                            subject_id=lineage,
                            lineage_ids=(lineage,),
                            severity="moderate",
                            live_priority="medium",
                            scientific_importance=3,
                            historical_importance=3,
                            measurements={
                                "share_before": before_share,
                                "share_after": after_share,
                                "share_delta": delta,
                            },
                            source_record_refs=(source_ref,),
                            episode_hint="lineage_expansion",
                            tags=("lineage", "rapid_change"),
                        )
                    )
                if delta <= -self.config.rapid_lineage_share_delta:
                    candidates.append(
                        self._candidate(
                            phase_order=420,
                            event_type="lineage_rapid_share_decline",
                            event_category="lineage",
                            subject_type="lineage",
                            subject_id=lineage,
                            lineage_ids=(lineage,),
                            severity="major",
                            live_priority="high",
                            scientific_importance=4,
                            historical_importance=4,
                            measurements={
                                "share_before": before_share,
                                "share_after": after_share,
                                "share_delta": delta,
                            },
                            source_record_refs=(source_ref,),
                            episode_hint="lineage_decline",
                            tags=("lineage", "rapid_change"),
                        )
                    )
                before_count = self._previous_lineage_counts[lineage]
                after_count = lineage_counts[lineage]
                if before_count > self.config.near_extinction_count and 0 < after_count <= self.config.near_extinction_count:
                    candidates.append(
                        self._candidate(
                            phase_order=425,
                            event_type="lineage_near_extinction",
                            event_category="lineage",
                            subject_type="lineage",
                            subject_id=lineage,
                            lineage_ids=(lineage,),
                            severity="major",
                            live_priority="high",
                            scientific_importance=4,
                            historical_importance=4,
                            measurements={
                                "alive_before": before_count,
                                "alive_after": after_count,
                            },
                            source_record_refs=(source_ref,),
                            episode_hint="lineage_decline",
                            tags=("lineage", "near_extinction"),
                        )
                    )
            if top_lineage and previous_top and top_lineage != previous_top:
                candidates.append(
                    self._candidate(
                        phase_order=405,
                        event_type="lineage_leadership_change",
                        event_category="lineage",
                        subject_type="lineage_pair",
                        subject_id=f"{previous_top}->{top_lineage}",
                        lineage_ids=(previous_top, top_lineage),
                        severity="moderate",
                        live_priority="medium",
                        scientific_importance=3,
                        historical_importance=3,
                        measurements={
                            "previous_leader_share": previous_shares.get(previous_top, 0.0),
                            "new_leader_share": shares.get(top_lineage, 0.0),
                        },
                        source_record_refs=(source_ref,),
                        tags=("lineage", "leadership"),
                    )
                )

        heavy_lifecycle = self.config.profile in {"scientific_full", "debug_trace"}
        for context_index, context in enumerate(list(birth_contexts or [])):
            lineage = _context_lineage(context)
            rare = bool(lineage and lineage_counts.get(lineage, 0) <= self.config.near_extinction_count + 1)
            new = bool(lineage and lineage not in self._previous_lineage_counts and self.last_emitted_generation >= 0)
            notable = rare or new
            if not notable and not heavy_lifecycle:
                continue
            agent_id = _safe_int(context.get("agent_id"), -1)
            coordinates = _context_coordinates(context)
            candidates.append(
                self._candidate(
                    phase_order=200,
                    event_type="notable_birth" if notable else "birth_observation",
                    event_category="lifecycle",
                    subject_type="agent" if agent_id >= 0 else "lineage",
                    subject_id=str(agent_id) if agent_id >= 0 else lineage,
                    agent_ids=(agent_id,) if agent_id >= 0 else (),
                    lineage_ids=(lineage,) if lineage else (),
                    patch_coordinates=coordinates,
                    region_id=f"patch:{coordinates[0]},{coordinates[1]}" if coordinates else None,
                    severity="moderate" if notable else "informational",
                    live_priority="medium" if notable else "background",
                    scientific_importance=3 if notable else 1,
                    historical_importance=(3 if new else 2) if notable else 1,
                    novelty_class=("new_lineage" if new else "rare_lineage_birth") if notable else "routine",
                    measurements={"lineage_alive_count": lineage_counts.get(lineage, 0)},
                    source_record_refs=(f"runner.patch_birth_contexts:g{generation}:i{context_index}",),
                    tags=("birth", "notable", "new_lineage" if new else "rare_lineage") if notable else ("birth", "full_profile"),
                )
            )

        for context_index, context in enumerate(list(death_contexts or [])):
            lineage = _context_lineage(context)
            extinct = bool(lineage and lineage in self._previous_lineage_counts and lineage not in lineage_counts)
            rare = bool(lineage and lineage_counts.get(lineage, 0) <= self.config.near_extinction_count)
            notable = extinct or rare
            if not notable and not heavy_lifecycle:
                continue
            agent_id = _safe_int(context.get("agent_id"), -1)
            coordinates = _context_coordinates(context)
            candidates.append(
                self._candidate(
                    phase_order=300,
                    event_type="notable_death" if notable else "death_observation",
                    event_category="lifecycle",
                    subject_type="agent" if agent_id >= 0 else "lineage",
                    subject_id=str(agent_id) if agent_id >= 0 else lineage,
                    agent_ids=(agent_id,) if agent_id >= 0 else (),
                    lineage_ids=(lineage,) if lineage else (),
                    patch_coordinates=coordinates,
                    region_id=f"patch:{coordinates[0]},{coordinates[1]}" if coordinates else None,
                    severity="critical" if extinct else "major" if rare else "informational",
                    live_priority="critical" if extinct else "high" if rare else "background",
                    scientific_importance=5 if extinct else 4 if rare else 1,
                    historical_importance=5 if extinct else 3 if rare else 1,
                    novelty_class="final_lineage_death" if extinct else "rare_lineage_death" if rare else "routine",
                    measurements={"lineage_alive_after": lineage_counts.get(lineage, 0)},
                    source_record_refs=(f"runner.patch_death_contexts:g{generation}:i{context_index}",),
                    tags=(
                        "death",
                        "notable",
                        "lineage_extinction" if extinct else "rare_lineage",
                    )
                    if notable
                    else ("death", "full_profile"),
                )
            )

        for row_index, row in enumerate(list(native_lifecycle_rows or [])):
            kind = _safe_int(row.get("event_type"), 0)
            if kind not in {1, 2, 3, 4}:
                continue
            founder_id = _safe_int(row.get("lineage_founder_id"), -1)
            lineage = f"L{founder_id:08d}" if founder_id >= 0 else ""
            if kind == 1:
                agent_id = _safe_int(row.get("child_agent_id"), -1)
                rare = bool(
                    lineage_detail_available
                    and lineage
                    and lineage_counts.get(lineage, 0)
                    <= self.config.near_extinction_count + 1
                )
                new = bool(
                    lineage_detail_available
                    and lineage
                    and lineage not in self._previous_lineage_counts
                    and self.last_emitted_generation >= 0
                )
                notable = rare or new
                event_type = "notable_birth" if notable else "birth_observation"
                category = "lifecycle"
            elif kind == 4:
                agent_id = _safe_int(row.get("target_agent_id"), _safe_int(row.get("actor_agent_id"), -1))
                extinct = bool(
                    lineage_detail_available
                    and lineage
                    and lineage in self._previous_lineage_counts
                    and lineage not in lineage_counts
                )
                rare = bool(
                    lineage_detail_available
                    and lineage
                    and lineage_counts.get(lineage, 0)
                    <= self.config.near_extinction_count
                )
                notable = extinct or rare
                event_type = "notable_death" if notable else "death_observation"
                category = "lifecycle"
            elif kind == 2:
                agent_id = _safe_int(row.get("child_agent_id"), -1)
                notable = False
                event_type = "mutation_observation"
                category = "inheritance"
            else:
                agent_id = _safe_int(row.get("actor_agent_id"), -1)
                notable = False
                event_type = "birth_failure_observation"
                category = "lifecycle"
            if not notable and not heavy_lifecycle:
                continue
            native_sequence = _safe_int(row.get("sequence"), row_index)
            native_tick = _safe_int(row.get("tick"), 0)
            candidates.append(
                self._candidate(
                    phase_order=200 if kind == 1 else 300 if kind == 4 else 250,
                    event_time_scope="step",
                    event_type=event_type,
                    event_category=category,
                    subject_type="agent" if agent_id >= 0 else "population",
                    subject_id=str(agent_id) if agent_id >= 0 else "global",
                    agent_ids=(agent_id,) if agent_id >= 0 else (),
                    lineage_ids=(lineage,) if lineage else (),
                    severity="moderate" if notable else "informational",
                    live_priority="medium" if notable else "background",
                    scientific_importance=3 if notable else 1,
                    historical_importance=3 if notable else 1,
                    novelty_class="typed_native_lifecycle_event" if notable else "routine",
                    measurements={
                        "native_event_type_code": kind,
                        "native_tick": native_tick,
                        "native_sequence": native_sequence,
                        "reason_code": _safe_int(row.get("reason_code"), -1),
                        "value": _safe_float(row.get("value"), 0.0),
                    },
                    source_component="native_runtime.typed_lifecycle_events",
                    source_record_refs=(f"metrics/native_typed_lifecycle_events.jsonl:g{generation}:tick{native_tick}:sequence{native_sequence}",),
                    tags=(
                        "native",
                        "typed_lifecycle",
                        "notable" if notable else "full_profile",
                    ),
                )
            )

        previous_depleted = _safe_float(self._previous_metrics.get("patch_food_depleted_frac"), 0.0)
        depleted = _safe_float(metrics.get("patch_food_depleted_frac"), 0.0)
        depletion_active = depleted >= self.config.food_depleted_fraction_threshold
        depletion_ready = self._condition_ready("regional_food_depletion", depletion_active)
        if depletion_active:
            self._food_depletion_observed = True
        if depletion_ready:
            candidates.append(
                self._candidate(
                    phase_order=500,
                    event_type="regional_food_depletion",
                    event_category="ecology",
                    subject_type="world",
                    subject_id="global",
                    severity="major",
                    live_priority="high",
                    scientific_importance=4,
                    historical_importance=4,
                    public_interest=2,
                    visual_interest=3,
                    measurements={
                        "depleted_fraction_before": previous_depleted,
                        "depleted_fraction_after": depleted,
                        "threshold": self.config.food_depleted_fraction_threshold,
                        "recovery_threshold": self.config.food_recovery_fraction_threshold,
                        "persistence_generations": self.config.minimum_persistence_generations,
                    },
                    source_record_refs=(source_ref,),
                    episode_hint="regional_food_depletion",
                    tags=("ecology", "food", "threshold_crossing"),
                )
            )
        recovery_active = self._food_depletion_observed and depleted <= self.config.food_recovery_fraction_threshold
        recovery_ready = self._condition_ready("regional_food_recovery", recovery_active)
        if recovery_ready:
            candidates.append(
                self._candidate(
                    phase_order=505,
                    event_type="regional_food_recovery",
                    event_category="ecology",
                    subject_type="world",
                    subject_id="global",
                    severity="moderate",
                    live_priority="medium",
                    scientific_importance=3,
                    historical_importance=3,
                    measurements={
                        "depleted_fraction_before": previous_depleted,
                        "depleted_fraction_after": depleted,
                        "threshold": self.config.food_recovery_fraction_threshold,
                        "persistence_generations": self.config.minimum_persistence_generations,
                    },
                    source_record_refs=(source_ref,),
                    episode_hint="ecological_recovery",
                    tags=("ecology", "food", "recovery"),
                )
            )
            self._food_depletion_observed = False

        crowding = _safe_float(metrics.get("patch_density_hotspot_frac"), 0.0)
        previous_crowding = _safe_float(self._previous_metrics.get("patch_density_hotspot_frac"), 0.0)
        if self._condition_ready(
            "crowding_hotspot",
            crowding >= self.config.crowding_hotspot_fraction_threshold,
        ):
            candidates.append(
                self._candidate(
                    phase_order=510,
                    event_type="crowding_hotspot",
                    event_category="ecology",
                    subject_type="world",
                    subject_id="global",
                    severity="major",
                    live_priority="high",
                    scientific_importance=4,
                    historical_importance=3,
                    measurements={
                        "hotspot_fraction_before": previous_crowding,
                        "hotspot_fraction_after": crowding,
                        "threshold": self.config.crowding_hotspot_fraction_threshold,
                        "persistence_generations": self.config.minimum_persistence_generations,
                    },
                    source_record_refs=(source_ref,),
                    episode_hint="crowding_waste_crisis",
                    tags=("ecology", "crowding"),
                )
            )
        waste = _safe_float(metrics.get("occupied_patch_waste_mean"), 0.0)
        previous_waste = _safe_float(self._previous_metrics.get("occupied_patch_waste_mean"), 0.0)
        if self._condition_ready("waste_hotspot", waste >= self.config.waste_hotspot_threshold):
            candidates.append(
                self._candidate(
                    phase_order=515,
                    event_type="waste_hotspot",
                    event_category="ecology",
                    subject_type="world",
                    subject_id="global",
                    severity="major",
                    live_priority="high",
                    scientific_importance=4,
                    historical_importance=3,
                    measurements={
                        "occupied_waste_before": previous_waste,
                        "occupied_waste_after": waste,
                        "threshold": self.config.waste_hotspot_threshold,
                        "persistence_generations": self.config.minimum_persistence_generations,
                    },
                    source_record_refs=(source_ref,),
                    episode_hint="crowding_waste_crisis",
                    tags=("ecology", "waste"),
                )
            )
        suppressed_regrowth = _safe_float(metrics.get("patch_regrowth_suppressed_total"), 0.0)
        if self._condition_ready("regrowth_suppression", suppressed_regrowth > self.config.min_magnitude):
            candidates.append(
                self._candidate(
                    phase_order=520,
                    event_type="regrowth_suppression",
                    event_category="ecology",
                    subject_type="world",
                    subject_id="global",
                    severity="moderate",
                    live_priority="medium",
                    scientific_importance=3,
                    historical_importance=2,
                    measurements={
                        "regrowth_suppressed_total": suppressed_regrowth,
                        "persistence_generations": self.config.minimum_persistence_generations,
                    },
                    source_record_refs=(source_ref,),
                    episode_hint="crowding_waste_crisis",
                    causality_class="direct_mechanism",
                    tags=("ecology", "regrowth"),
                )
            )

        predator_events = _safe_int(metrics.get("jungle_predator_events"), len(list(predator_contexts or [])))
        predator_threshold = max(1, int(math.ceil(max(1, alive_count) * 0.05)))
        if self._condition_ready("predator_pressure_spike", predator_events >= predator_threshold):
            candidates.append(
                self._candidate(
                    phase_order=530,
                    event_type="predator_pressure_spike",
                    event_category="ecology",
                    subject_type="population",
                    subject_id="global",
                    severity="major",
                    live_priority="high",
                    scientific_importance=4,
                    historical_importance=3,
                    public_interest=3,
                    visual_interest=4,
                    measurements={
                        "predator_events": predator_events,
                        "predator_damage": _safe_float(metrics.get("jungle_predator_damage_total"), 0.0),
                        "predator_deaths": _safe_int(metrics.get("jungle_predator_deaths"), 0),
                        "threshold": predator_threshold,
                        "persistence_generations": self.config.minimum_persistence_generations,
                    },
                    source_record_refs=(
                        source_ref,
                        "runner.jungle_predator_event_contexts",
                    ),
                    episode_hint="predator_pressure_wave",
                    causality_class="direct_mechanism",
                    tags=("ecology", "predator"),
                )
            )

        wdl_count = _safe_int(metrics.get("wdl_delayed_hazard_event_count"), len(list(wdl_contexts or [])))
        if self._condition_ready("delayed_hazard_cycle", wdl_count > 0):
            candidates.append(
                self._candidate(
                    phase_order=540,
                    event_type="delayed_hazard_cycle",
                    event_category="hazard",
                    subject_type="population",
                    subject_id="global",
                    severity="major" if _safe_float(metrics.get("wdl_delayed_hazard_damage_total"), 0.0) > 0 else "moderate",
                    live_priority="high",
                    scientific_importance=4,
                    historical_importance=3,
                    measurements={
                        "event_count": wdl_count,
                        "damage_total": _safe_float(metrics.get("wdl_delayed_hazard_damage_total"), 0.0),
                        "agents_damaged": _safe_int(metrics.get("wdl_delayed_hazard_agents_damaged"), 0),
                        "persistence_generations": self.config.minimum_persistence_generations,
                    },
                    source_record_refs=(
                        source_ref,
                        "runner.wdl_delayed_hazard_event_contexts",
                    ),
                    episode_hint="delayed_hazard_cycle",
                    causality_class="direct_mechanism",
                    tags=("hazard", "delayed_hazard"),
                )
            )

        positive_recovery = _safe_float(metrics.get("patch_recovery_positive_total"), 0.0)
        previous_recovery = _safe_float(self._previous_metrics.get("patch_recovery_positive_total"), 0.0)
        if self._condition_ready("ecological_recovery", positive_recovery > self.config.min_magnitude):
            candidates.append(
                self._candidate(
                    phase_order=550,
                    event_type="ecological_recovery",
                    event_category="ecology",
                    subject_type="world",
                    subject_id="global",
                    severity="moderate",
                    live_priority="medium",
                    scientific_importance=3,
                    historical_importance=3,
                    measurements={
                        "positive_recovery_total": positive_recovery,
                        "previous_total": previous_recovery,
                        "persistence_generations": self.config.minimum_persistence_generations,
                    },
                    source_record_refs=(source_ref,),
                    episode_hint="ecological_recovery",
                    causality_class="direct_mechanism",
                    tags=("ecology", "recovery"),
                )
            )

        if survival_fraction is not None and self._condition_ready(
            "population_bottleneck",
            survival_fraction <= self.config.bottleneck_survival_fraction,
        ):
            candidates.append(
                self._candidate(
                    phase_order=560,
                    event_type="population_bottleneck",
                    event_category="population",
                    subject_type="population",
                    subject_id="global",
                    severity="critical",
                    live_priority="critical",
                    scientific_importance=5,
                    historical_importance=5,
                    public_interest=3,
                    visual_interest=3,
                    measurements={
                        "cohort_size": cohort_size,
                        "alive": alive_count,
                        "survival_fraction": survival_fraction,
                        "threshold": self.config.bottleneck_survival_fraction,
                        "persistence_generations": self.config.minimum_persistence_generations,
                    },
                    source_record_refs=(source_ref,),
                    episode_hint="population_bottleneck",
                    tags=("population", "bottleneck"),
                )
            )

        for candidate_index, candidate in enumerate(_json_list(metrics.get("lineage_branch_candidates_json"))[:3]):
            lineage = str(candidate.get("lineage_id") or candidate.get("id") or "")
            candidates.append(
                self._candidate(
                    phase_order=440,
                    event_type="branch_candidate",
                    event_category="lineage",
                    subject_type="lineage",
                    subject_id=lineage or f"candidate-{candidate_index}",
                    lineage_ids=(lineage,) if lineage else (),
                    severity="moderate",
                    live_priority="medium",
                    scientific_importance=3,
                    historical_importance=4,
                    measurements={key: value for key, value in candidate.items() if isinstance(value, (str, int, float, bool))},
                    source_record_refs=(
                        source_ref,
                        f"lineage_branch_candidates_json:index={candidate_index}",
                    ),
                    tags=("lineage", "branch_candidate"),
                )
            )

        metrics["chronicle_survival_fraction"] = survival_fraction
        self._previous_metrics = canonical_payload(metrics)
        return candidates, lineage_counts

    def _signature(self, candidate: Mapping[str, Any]) -> str:
        return "|".join(
            (
                str(candidate.get("event_type", "")),
                str(candidate.get("subject_id") or ""),
                str(candidate.get("region_id") or ""),
            )
        )

    def _condition_ready(self, key: str, active: bool) -> bool:
        if not active:
            self._condition_streaks[str(key)] = 0
            return False
        streak = int(self._condition_streaks.get(str(key), 0)) + 1
        self._condition_streaks[str(key)] = streak
        return streak == int(self.config.minimum_persistence_generations)

    def _select_candidates(self, generation: int, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        category_counts: Counter[str] = Counter()
        ordered = sorted(
            candidates,
            key=lambda item: (
                0 if str(item.get("event_type")) == "generation_summary" else 1,
                _PRIORITY_RANK.get(str(item.get("live_priority", "background")), 4),
                -int(item.get("scientific_importance", 0)),
                int(item.get("phase_order", 0)),
                str(item.get("subject_id") or ""),
                str(item.get("event_type") or ""),
            ),
        )
        for candidate in ordered:
            event_type = str(candidate["event_type"])
            priority = str(candidate.get("live_priority", "background"))
            if len(selected) >= self.config.max_stored_events_per_generation:
                self.suppressed_counts["generation_budget"] += 1
                continue
            if self.config.profile == "summary" and event_type != "generation_summary" and priority != "critical":
                self.suppressed_counts["profile"] += 1
                continue
            if self.config.profile == "notable" and int(candidate.get("scientific_importance", 0)) < 2:
                self.suppressed_counts["profile"] += 1
                continue
            category = str(candidate.get("event_category", "unknown"))
            if category_counts[category] >= self.config.max_events_per_category_per_generation:
                self.suppressed_counts[f"category:{category}"] += 1
                continue
            signature = self._signature(candidate)
            cooldown = self.config.region_cooldown_generations if candidate.get("region_id") else self.config.subject_cooldown_generations
            cooldown = max(int(cooldown), int(self.config.aggregation_window_generations))
            last_generation = self._cooldown_last_generation.get(signature)
            if (
                event_type != "generation_summary"
                and event_type not in _CROSSING_TYPES
                and last_generation is not None
                and int(generation) - int(last_generation) <= int(cooldown)
            ):
                self.suppressed_counts["cooldown"] += 1
                continue
            selected.append(candidate)
            category_counts[category] += 1
            self._cooldown_last_generation[signature] = int(generation)
        return sorted(
            selected,
            key=lambda item: (
                int(item.get("phase_order", 0)),
                str(item.get("event_type", "")),
                str(item.get("subject_id") or ""),
            ),
        )

    def observe_generation(
        self,
        *,
        generation: int,
        step: int,
        alive_agents: Iterable[Any],
        deaths: int,
        births: int,
        failed_births: int,
        cohort_size: int | None,
        cohort_survival_fraction: float | None = None,
        patch_metrics: Mapping[str, Any],
        birth_contexts: Iterable[Mapping[str, Any]] = (),
        death_contexts: Iterable[Mapping[str, Any]] = (),
        predator_contexts: Iterable[Mapping[str, Any]] = (),
        wdl_contexts: Iterable[Mapping[str, Any]] = (),
        native_lifecycle_rows: Iterable[Mapping[str, Any]] = (),
        native_lifecycle_row_count: int | None = None,
        native_public_output_counts: Mapping[str, Any] | None = None,
        native_lineage_counts: Mapping[str, Any] | None = None,
        generation_source_component: str | None = None,
        generation_source_record_ref: str | None = None,
        defer_persistence: bool = False,
    ) -> list[dict[str, Any]]:
        if self.disabled_after_write_failure or int(generation) <= int(self.last_emitted_generation):
            return []
        suppressed_before = Counter(self.suppressed_counts)
        native_rows = list(native_lifecycle_rows or [])
        if native_public_output_counts:
            self._source_completeness = (
                "live_generation_boundary_native_typed_lifecycle" if native_rows else "live_generation_boundary_native_aggregate_lifecycle"
            )
        candidates, lineage_counts = self._generation_candidates(
            generation=int(generation),
            alive_agents=alive_agents,
            deaths=int(deaths),
            births=int(births),
            failed_births=int(failed_births),
            cohort_size=(int(cohort_size) if cohort_size is not None else None),
            cohort_survival_fraction=cohort_survival_fraction,
            patch_metrics=patch_metrics,
            birth_contexts=birth_contexts,
            death_contexts=death_contexts,
            predator_contexts=predator_contexts,
            wdl_contexts=wdl_contexts,
            native_lifecycle_rows=native_rows,
            native_lifecycle_row_count=native_lifecycle_row_count,
            native_public_output_counts=native_public_output_counts,
            native_lineage_counts=native_lineage_counts,
            generation_source_component=generation_source_component,
            generation_source_record_ref=generation_source_record_ref,
        )
        selected = self._select_candidates(int(generation), candidates)
        suppressed_delta = {
            key: int(value) - int(suppressed_before.get(key, 0))
            for key, value in self.suppressed_counts.items()
            if int(value) - int(suppressed_before.get(key, 0)) > 0
        }
        if suppressed_delta and len(selected) < self.config.max_stored_events_per_generation:
            selected.append(
                self._candidate(
                    phase_order=940,
                    event_type="chronicle_events_suppressed",
                    event_category="integrity",
                    subject_type="run",
                    subject_id=self.run_id,
                    severity="minor",
                    live_priority="low",
                    scientific_importance=2,
                    historical_importance=2,
                    measurements={
                        "suppressed_count": sum(suppressed_delta.values()),
                        "suppressed_by_reason": suppressed_delta,
                    },
                    source_component="chronicle.event_budget",
                    source_record_refs=(f"chronicle:selection:generation={int(generation)}",),
                    tags=("integrity", "suppression", "bounded_output"),
                )
            )
            selected.sort(
                key=lambda item: (
                    int(item.get("phase_order", 0)),
                    str(item.get("event_type", "")),
                    str(item.get("subject_id") or ""),
                )
            )
        emitted: list[dict[str, Any]] = []
        summary_candidates: list[tuple[int, dict[str, Any]]] = []
        for sequence, candidate in enumerate(selected):
            if str(candidate.get("event_type")) == "generation_summary":
                summary_candidates.append((sequence, candidate))
                continue
            event = make_event(
                run_id=self.run_id,
                campaign_id=self.campaign_id,
                branch_id=self.branch_id,
                generation=int(generation),
                step=int(step),
                intra_step_sequence=int(sequence),
                **candidate,
            )
            if str(event["event_id"]) in self._seen_event_ids:
                self.suppressed_counts["duplicate"] += 1
                continue
            emitted.append(event)
            self._seen_event_ids.add(str(event["event_id"]))
        active_episode_count = 0
        if self.config.episode_enabled:
            active_episode_count = sum(
                1
                for episode in detect_episodes(
                    [*self.events, *emitted],
                    gap_generations=self.config.episode_gap_generations,
                    finalize=False,
                )
                if str(episode.get("status")) == "open"
            )
        for sequence, candidate in summary_candidates:
            measurements = dict(candidate.get("measurements") or {})
            measurements["chronicle_event_count"] = len(emitted) + len(summary_candidates)
            measurements["active_episode_count"] = active_episode_count
            measurements["suppressed_event_count"] = sum(suppressed_delta.values())
            candidate = dict(candidate)
            candidate["measurements"] = measurements
            event = make_event(
                run_id=self.run_id,
                campaign_id=self.campaign_id,
                branch_id=self.branch_id,
                generation=int(generation),
                step=int(step),
                intra_step_sequence=int(sequence),
                **candidate,
            )
            if str(event["event_id"]) not in self._seen_event_ids:
                emitted.append(event)
                self._seen_event_ids.add(str(event["event_id"]))
        self.events.extend(emitted)
        self.events.sort(key=event_order_key)
        self._previous_lineage_counts = dict(lineage_counts)
        self.last_emitted_generation = int(generation)
        if not defer_persistence:
            try:
                self._persist(finalize=False)
            except Exception:
                self.write_failures += 1
                self.disabled_after_write_failure = True
                return emitted
        if self.config.terminal_enabled:
            live_episodes = detect_episodes(
                self.events,
                gap_generations=self.config.episode_gap_generations,
                finalize=False,
            )
            new_entries = entries_from_events(emitted, live_episodes)
            for line in terminal_lines(new_entries, maximum=self.config.max_live_messages_per_generation):
                print(line, flush=True)
        return emitted

    def observe_native_terminal_summaries(
        self,
        summaries: Iterable[Mapping[str, Any]],
        *,
        typed_lifecycle_rows: Iterable[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        """Ingest native-owned generation results without generation-time writes.

        The fully resident owner calls this only after terminal settlement. Compact
        summaries remain authoritative for population counts. Richer lineage and
        ecology facts stay unavailable until their native bounded projections are
        part of the terminal contract; Chronicle must not reconstruct them from
        stale Python objects.
        """

        ordered = [dict(row) for row in summaries]
        if not ordered:
            raise ValueError("native terminal Chronicle requires generation summaries")
        generations = [_safe_int(row.get("generation"), -1) for row in ordered]
        if any(generation < 0 for generation in generations):
            raise ValueError("native terminal Chronicle summary generation is invalid")
        if generations != sorted(set(generations)):
            raise ValueError("native terminal Chronicle summaries must be unique and ordered")

        lifecycle_by_generation: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for raw in typed_lifecycle_rows:
            row = dict(raw)
            generation = _safe_int(row.get("generation"), -1)
            if generation >= 0:
                lifecycle_by_generation[generation].append(row)

        emitted: list[dict[str, Any]] = []
        previous_event_cursor = 0
        rich_projection_count = 0
        for row, generation in zip(ordered, generations, strict=True):
            alive = _safe_int(row.get("alive"), 0)
            native_counts = {"alive": alive}
            for field in ("birth_attempts", "capacity", "free_slots"):
                if field in row:
                    native_counts[field] = _safe_int(row[field], 0)
            event_cursor = _safe_int(row.get("event_cursor"), previous_event_cursor)
            if event_cursor < previous_event_cursor:
                raise ValueError("native terminal Chronicle lifecycle cursor regressed")
            lifecycle_row_count = event_cursor - previous_event_cursor
            previous_event_cursor = event_cursor
            patch_metrics = row.get("patch_metrics", {})
            if not isinstance(patch_metrics, Mapping):
                raise ValueError("native terminal Chronicle patch metrics must be a mapping")
            lineage_counts = row.get("native_lineage_counts")
            if lineage_counts is not None and not isinstance(lineage_counts, Mapping):
                raise ValueError("native terminal Chronicle lineage counts must be a mapping")
            if patch_metrics or lineage_counts is not None:
                rich_projection_count += 1
            cohort_size = (
                _safe_int(row.get("cohort_size"), 0)
                if "cohort_size" in row
                else None
            )
            cohort_survival_fraction = (
                _safe_float(row.get("cohort_survival_fraction"), 0.0)
                if "cohort_survival_fraction" in row
                else None
            )
            emitted.extend(
                self.observe_generation(
                    generation=generation,
                    step=-1,
                    alive_agents=(),
                    deaths=_safe_int(row.get("deaths"), 0),
                    births=_safe_int(row.get("births"), 0),
                    failed_births=_safe_int(row.get("failed_births"), 0),
                    cohort_size=cohort_size,
                    cohort_survival_fraction=cohort_survival_fraction,
                    patch_metrics=patch_metrics,
                    native_lifecycle_rows=lifecycle_by_generation.get(generation, ()),
                    native_lifecycle_row_count=lifecycle_row_count,
                    native_public_output_counts=native_counts,
                    native_lineage_counts=lineage_counts,
                    generation_source_component=str(
                        row.get(
                            "chronicle_source_component",
                            "native_runtime.cuda_architecture_b_runner_owner_v1",
                        )
                    ),
                    generation_source_record_ref=str(
                        row.get(
                            "chronicle_source_record_ref",
                            f"compact_generation_summaries.jsonl:generation={generation}",
                        )
                    ),
                    defer_persistence=True,
                )
            )
        self._source_completeness = (
            "native_terminal_generation_boundary_projections"
            if rich_projection_count == len(ordered)
            else "native_terminal_compact_generation_summaries"
        )
        return emitted

    def _persist(self, *, finalize: bool) -> None:
        assert_unique_events(self.events)
        episodes = (
            detect_episodes(
                self.events,
                gap_generations=self.config.episode_gap_generations,
                finalize=finalize,
            )
            if self.config.episode_enabled
            else []
        )
        entries = entries_from_events(self.events, episodes)
        contexts = (
            _context_windows(
                self.events,
                before=self.config.context_events_before,
                after=self.config.context_events_after,
                max_per_generation=self.config.max_context_windows_per_generation,
            )
            if self.config.context_window_enabled
            else []
        )
        write_jsonl(self.root / ARTIFACT_NAMES["events"], self.events)
        write_jsonl(self.root / ARTIFACT_NAMES["entries"], entries)
        write_json(self.root / ARTIFACT_NAMES["episodes"], episodes)
        write_jsonl(self.root / ARTIFACT_NAMES["contexts"], contexts)
        event_counts = dict(sorted(Counter(str(event["event_type"]) for event in self.events).items()))
        episode_counts = dict(sorted(Counter(str(episode["episode_type"]) for episode in episodes).items()))
        limitations = [
            "Chronicle V1 observes bounded generation and selected lifecycle facts.",
            "It does not contain an unrestricted action or tensor trace.",
        ]
        if self._source_completeness == "live_generation_boundary_native_aggregate_lifecycle":
            limitations.append(
                "The native owner exposed aggregate lifecycle counts but no typed lifecycle rows; exact native agent and lineage lifecycle events are unavailable."
            )
        if self._source_completeness == "native_terminal_compact_generation_summaries":
            limitations.append(
                "The fully resident native owner supplied terminal compact generation summaries; per-generation lineage, ecology, hazard, and cohort-survival projections were unavailable."
            )
        summary = {
            "schema_version": "eiel_scientific_chronicle_summary_v1",
            "run_id": self.run_id,
            "campaign_id": self.campaign_id,
            "branch_id": self.branch_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "profile": self.config.profile,
            "finalized": bool(finalize),
            "last_generation": self.last_emitted_generation,
            "event_count": len(self.events),
            "chronicle_entry_count": len(entries),
            "episode_count": len(episodes),
            "open_episode_count": sum(1 for episode in episodes if episode["status"] == "open"),
            "context_window_count": len(contexts),
            "event_counts_by_type": event_counts,
            "episode_counts_by_type": episode_counts,
            "suppressed_counts": dict(sorted(self.suppressed_counts.items())),
            "write_failures": int(self.write_failures),
            "source_completeness": self._source_completeness,
            "limitations": limitations,
        }
        write_json(self.root / ARTIFACT_NAMES["summary"], summary)
        write_json(
            self.root / ARTIFACT_NAMES["index"],
            {
                "schema_version": "eiel_scientific_chronicle_index_v1",
                "run_id": self.run_id,
                "generation": self.last_emitted_generation,
                "event_count": len(self.events),
                "episode_count": len(episodes),
                "feeds": {
                    "formal_events": ARTIFACT_NAMES["events"],
                    "scientific_chronicle": ARTIFACT_NAMES["entries"],
                    "episodes": ARTIFACT_NAMES["episodes"],
                    "manifest": ARTIFACT_NAMES["manifest"],
                },
                "filter_fields": [
                    "event_type",
                    "event_category",
                    "generation",
                    "step",
                    "live_priority",
                    "subject_type",
                    "subject_id",
                    "lineage_ids",
                    "region_id",
                ],
            },
        )
        inventory = artifact_inventory(
            self.root,
            logical_names=(
                "events",
                "entries",
                "episodes",
                "contexts",
                "summary",
                "index",
            ),
        )
        storage = {
            "schema_version": "eiel_scientific_chronicle_storage_summary_v1",
            "profile": self.config.profile,
            "preflight": self.preflight,
            "artifacts": inventory,
            "total_bytes": sum(int(row["bytes"]) for row in inventory),
            "event_count": len(self.events),
            "suppressed_count": sum(self.suppressed_counts.values()),
        }
        write_json(self.root / ARTIFACT_NAMES["storage"], storage)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "recorder_version": RECORDER_VERSION,
            "run_id": self.run_id,
            "campaign_id": self.campaign_id,
            "branch_id": self.branch_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "source_commit": self._source_commit,
            "config_hash": self.config.config_hash,
            "config": self.config.to_dict(),
            "profile": self.config.profile,
            "thresholds_frozen": True,
            "deterministic_generation_mode": "canonical_generation_boundary_v1",
            "event_count": len(self.events),
            "suppressed_counts": dict(sorted(self.suppressed_counts.items())),
            "episode_count": len(episodes),
            "storage_bytes": int(storage["total_bytes"]),
            "source_completeness": summary["source_completeness"],
            "artifact_names": dict(sorted(ARTIFACT_NAMES.items())),
            "finalized": bool(finalize),
        }
        write_json(self.root / ARTIFACT_NAMES["manifest"], manifest)

    def finalize(self, *, status: str = "complete") -> None:
        if self.finalized or self.disabled_after_write_failure:
            return
        try:
            self._persist(finalize=True)
            self.finalized = True
        except Exception:
            self.write_failures += 1
            self.disabled_after_write_failure = True

    def export_state(self) -> dict[str, Any]:
        """Return versioned observer-only state for checkpoint bundle v2."""
        return canonical_payload(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "recorder_version": RECORDER_VERSION,
                "run_id": self.run_id,
                "campaign_id": self.campaign_id,
                "branch_id": self.branch_id,
                "parent_checkpoint_id": self.parent_checkpoint_id,
                "config_hash": self.config.config_hash,
                "events": self.events,
                "previous_metrics": self._previous_metrics,
                "previous_lineage_counts": self._previous_lineage_counts,
                "cooldown_last_generation": self._cooldown_last_generation,
                "condition_streaks": self._condition_streaks,
                "food_depletion_observed": self._food_depletion_observed,
                "suppressed_counts": dict(self.suppressed_counts),
                "write_failures": self.write_failures,
                "last_emitted_generation": self.last_emitted_generation,
                "source_completeness": self._source_completeness,
            }
        )

    def import_state(self, state: Mapping[str, Any]) -> None:
        if str(state.get("schema_version")) != STATE_SCHEMA_VERSION:
            self._record_checkpoint_mismatch(
                reason="state_schema_version_mismatch",
                expected=STATE_SCHEMA_VERSION,
                observed=str(state.get("schema_version")),
            )
            raise ValueError(f"unsupported Chronicle checkpoint state: {state.get('schema_version')}")
        if str(state.get("config_hash")) != self.config.config_hash:
            self._record_checkpoint_mismatch(
                reason="config_hash_mismatch",
                expected=self.config.config_hash,
                observed=str(state.get("config_hash")),
            )
            raise ValueError("Chronicle checkpoint config hash mismatch")
        events = [dict(event) for event in state.get("events", []) or []]
        events.sort(key=event_order_key)
        assert_unique_events(events)
        self.events = events
        self._seen_event_ids = {str(event["event_id"]) for event in events}
        self._previous_metrics = dict(state.get("previous_metrics", {}) or {})
        self._previous_lineage_counts = {str(key): _safe_int(value, 0) for key, value in dict(state.get("previous_lineage_counts", {}) or {}).items()}
        self._cooldown_last_generation = {str(key): _safe_int(value, -1) for key, value in dict(state.get("cooldown_last_generation", {}) or {}).items()}
        self._condition_streaks = {str(key): _safe_int(value, 0) for key, value in dict(state.get("condition_streaks", {}) or {}).items()}
        self._food_depletion_observed = bool(state.get("food_depletion_observed", False))
        self.suppressed_counts = Counter({str(key): _safe_int(value, 0) for key, value in dict(state.get("suppressed_counts", {}) or {}).items()})
        self.write_failures = _safe_int(state.get("write_failures"), 0)
        self.last_emitted_generation = _safe_int(state.get("last_emitted_generation"), -1)
        self._source_completeness = str(state.get("source_completeness") or "live_generation_boundary_complete")
        self.parent_checkpoint_id = str(state.get("parent_checkpoint_id") or "") or None
        self._persist(finalize=False)

    def _record_checkpoint_mismatch(self, *, reason: str, expected: str, observed: str) -> None:
        event = make_event(
            run_id=self.run_id,
            campaign_id=self.campaign_id,
            branch_id=self.branch_id,
            generation=max(0, int(self.last_emitted_generation)),
            step=0,
            phase_order=980,
            intra_step_sequence=0,
            event_time_scope="run",
            event_type="checkpoint_chronicle_metadata_mismatch",
            event_category="integrity",
            subject_type="run",
            subject_id=self.run_id,
            severity="critical",
            live_priority="critical",
            scientific_importance=5,
            historical_importance=5,
            confidence_class="measured",
            source_component="checkpoint_bundle_v2.runner_state",
            source_record_refs=("runner_state.scientific_chronicle_state",),
            measurements={"reason": reason, "expected": expected, "observed": observed},
            tags=("integrity", "checkpoint", "mismatch"),
        )
        if str(event["event_id"]) not in self._seen_event_ids:
            self.events.append(event)
            self.events.sort(key=event_order_key)
            self._seen_event_ids.add(str(event["event_id"]))
            self._persist(finalize=False)

    def set_parent_checkpoint(self, checkpoint_id: str | None) -> None:
        self.parent_checkpoint_id = str(checkpoint_id or "") or None
        self._persist(finalize=False)
