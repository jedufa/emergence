from __future__ import annotations

"""Theme 5 biological lineage observability.

This module is diagnostic-only. It must not call RNG or change simulation state
except for deterministic fallback normalization of missing lineage metadata.
"""

import csv
import hashlib
import io
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from agents.agent import format_lineage_id, normalize_lineage_stats
from evolution.population_pkg.founder_spatial_separation import founder_spatial_survival_record
from simulation.output_cost_timing import current_flat_output_cost_timing


DESCRIPTOR_SCHEMA_VERSION = "theme5_lineage_descriptors_v1"
CANDIDATE_SCHEMA_VERSION = "theme5_branch_candidates_v1"
ARCHIVE_SCHEMA_VERSION = "theme5_lineage_qd_archive_v2"
ARCHIVE_ENTRY_SCHEMA_VERSION = "theme5_lineage_qd_archive_entry_v2"
ARCHIVE_LEGACY_ENTRY_SCHEMA_VERSION = "theme5_lineage_qd_archive_entry_v1"
ARCHIVE_POLICY_VERSION = "theme5_qd_archive_policy_v2"
DEFAULT_ECOLOGICAL_ROLE = "general_agent"
DESCRIPTOR_VECTOR_FIELDS = [
    "mean_survival_steps",
    "mean_energy",
    "mean_health",
    "mean_injury",
    "mean_offspring_count",
    "mean_first_reproduction_step",
    "mean_hunger_drive",
    "mean_fear_drive",
    "mean_aggression_bias",
    "mean_exploration_bias",
    "mean_prediction_error",
    "mean_identity_drift",
    "mean_drive_jitter",
    "mean_memory_dim",
    "gru_fraction",
    "mean_cognition_complexity_proxy",
]
CANDIDATE_RARE_SHARE_MAX = 0.25
CANDIDATE_RARE_MIN_PERSISTENCE_STEPS = 2
CANDIDATE_DOMINANT_SHARE_MIN = 0.50
CANDIDATE_DIVERGENCE_MIN = 0.15
ARCHIVE_DEFAULT_TOP_N = 8
ARCHIVE_DEFAULT_MIN_SCORE = 0.25
NATIVE_LINEAGE_INPUT_SCHEMA_VERSION = "eiel_native_lineage_metrics_input_v1"
NATIVE_LINEAGE_PATCH_PROFILES = frozenset(("P0", "P1", "P2"))
NATIVE_LINEAGE_PATCH_PROFILE_SCOPE = "full_profile"
NATIVE_LINEAGE_PATCH_CONTRACT_SCHEMA_VERSION = "eiel_patch_native_import_contract_v1"

LINEAGE_SUMMARY_COLUMNS = [
    "generation",
    "active_lineages",
    "dominant_lineage_id",
    "dominant_lineage_population_share",
    "top3_lineage_population_share",
    "lineage_population_entropy",
    "lineage_reproduction_entropy",
    "dominant_lineage_reproduction_share",
    "rare_persistent_lineages",
    "lineage_collapse_warning",
    "lineage_collapse_streak",
]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _agent_id(agent: Any) -> int:
    return _safe_int(getattr(agent, "agent_id", 0), 0)


def _sorted_agents(agents: Iterable[Any]) -> List[Any]:
    return sorted(list(agents or []), key=lambda a: _agent_id(a))


def lineage_entropy(counts: Mapping[str, int]) -> float:
    total = float(sum(max(0, int(v)) for v in counts.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for raw in counts.values():
        count = max(0, int(raw))
        if count <= 0:
            continue
        p = float(count) / total
        entropy -= p * math.log(p)
    return float(entropy)


def top_n_share(counts: Mapping[str, int], n: int) -> float:
    total = float(sum(max(0, int(v)) for v in counts.values()))
    if total <= 0.0 or int(n) <= 0:
        return 0.0
    top = sorted((max(0, int(v)) for v in counts.values()), reverse=True)[: int(n)]
    return float(sum(top)) / total


@dataclass
class LineageMetricsState:
    collapse_streak: int = 0
    population_dominance_streak: int = 0
    reproduction_dominance_streak: int = 0
    last_dominant_lineage_id: str = ""
    seen_lineages: set[str] = field(default_factory=set)
    lineage_presence_counts: Dict[str, int] = field(default_factory=dict)
    lineage_first_seen_generation: Dict[str, int] = field(default_factory=dict)
    archived_entry_ids: set[str] = field(default_factory=set)
    archive_last_generation_by_key: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "theme5_lineage_metrics_state_v1",
            "collapse_streak": int(self.collapse_streak),
            "population_dominance_streak": int(self.population_dominance_streak),
            "reproduction_dominance_streak": int(self.reproduction_dominance_streak),
            "last_dominant_lineage_id": str(self.last_dominant_lineage_id),
            "seen_lineages": sorted(str(x) for x in self.seen_lineages),
            "lineage_presence_counts": {str(k): int(v) for k, v in sorted(self.lineage_presence_counts.items())},
            "lineage_first_seen_generation": {
                str(k): int(v) for k, v in sorted(self.lineage_first_seen_generation.items())
            },
            "archived_entry_ids": sorted(str(x) for x in self.archived_entry_ids),
            "archive_last_generation_by_key": {
                str(k): int(v) for k, v in sorted(self.archive_last_generation_by_key.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LineageMetricsState":
        state = cls()
        if not isinstance(data, Mapping):
            return state
        state.collapse_streak = _safe_int(data.get("collapse_streak", 0), 0)
        state.population_dominance_streak = _safe_int(data.get("population_dominance_streak", 0), 0)
        state.reproduction_dominance_streak = _safe_int(data.get("reproduction_dominance_streak", 0), 0)
        state.last_dominant_lineage_id = str(data.get("last_dominant_lineage_id", "") or "")
        raw_seen = data.get("seen_lineages", [])
        try:
            state.seen_lineages = {str(x) for x in list(raw_seen or [])}
        except Exception:
            state.seen_lineages = set()
        raw_counts = data.get("lineage_presence_counts", {})
        if isinstance(raw_counts, Mapping):
            state.lineage_presence_counts = {str(k): _safe_int(v, 0) for k, v in raw_counts.items()}
        raw_first = data.get("lineage_first_seen_generation", {})
        if isinstance(raw_first, Mapping):
            state.lineage_first_seen_generation = {str(k): _safe_int(v, 0) for k, v in raw_first.items()}
        raw_archived = data.get("archived_entry_ids", [])
        try:
            state.archived_entry_ids = {str(x) for x in list(raw_archived or [])}
        except Exception:
            state.archived_entry_ids = set()
        raw_archive_last = data.get("archive_last_generation_by_key", {})
        if isinstance(raw_archive_last, Mapping):
            state.archive_last_generation_by_key = {str(k): _safe_int(v, 0) for k, v in raw_archive_last.items()}
        return state


@dataclass(frozen=True)
class NativeLineageGenomeRecord:
    """Immutable metrics-only genome view; never attached to the population."""

    hidden_size: int
    memory_dim: int
    use_gru: bool
    aggression_bias: float
    exploration_bias: float


@dataclass(frozen=True)
class NativeLineageStatsRecord:
    """Immutable subset consumed by the existing Theme 5 lineage algorithms."""

    lineage_id: str
    founder_id: int
    parent_ids: tuple[int, ...]
    parent_lineage_ids: tuple[str, ...]
    birth_generation: int
    birth_tick: int
    lineage_depth: int
    offspring_count: int
    survival_steps: float
    energy_collected: float
    first_repro_step: Optional[int]
    pred_err_count: int
    pred_err_sum: float
    episode0_pred_err_mean: float
    identity_drift_ema: float
    drive_jitter: float
    ecological_role: str
    role_source: str
    role_confidence: Optional[float]


@dataclass(frozen=True)
class NativeLineageAgentRecord:
    """Immutable metrics DTO built from native public output and one projection."""

    agent_id: int
    energy: float
    health: float
    injury: float
    drives: tuple[float, float, float, float]
    genome: NativeLineageGenomeRecord
    stats: NativeLineageStatsRecord
    y: int = 0
    x: int = 0


def _native_required(mapping: Mapping[str, Any], name: str, *, context: str) -> Any:
    if name not in mapping:
        raise ValueError(f"{context} is missing required field {name}")
    return mapping[name]


def _native_finite_float(value: Any, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context} is not finite")
    return result


def _native_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} is not an integer")
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} is not an integer") from exc
    if not math.isfinite(numeric) or numeric != float(result):
        raise ValueError(f"{context} is not an integer")
    return result


def _native_nonnegative_int(value: Any, *, context: str) -> int:
    result = _native_int(value, context=context)
    if result < 0:
        raise ValueError(f"{context} must be non-negative")
    return result


def _native_int_tuple(value: Any, *, context: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{context} must be a sequence")
    result = tuple(_native_int(item, context=f"{context}[{index}]") for index, item in enumerate(value))
    if any(item < 0 for item in result):
        raise ValueError(f"{context} contains a negative agent ID")
    return result


def _native_str_tuple(value: Any, *, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{context} must be a sequence")
    return tuple(str(item) for item in value)


def native_lineage_agents_from_input(
    payload: Mapping[str, Any],
    *,
    expected_generation: int,
) -> tuple[NativeLineageAgentRecord, ...]:
    """Validate and materialize a metrics-only native lineage input.

    This creates immutable diagnostic records. It does not construct simulation
    ``Agent`` objects and cannot synchronize or mutate the Python population.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("native lineage metrics input must be a mapping")
    schema = str(payload.get("schema_version", "") or "")
    if schema != NATIVE_LINEAGE_INPUT_SCHEMA_VERSION:
        raise ValueError(f"unsupported native lineage metrics input schema: {schema or 'missing'}")
    world = str(payload.get("world", "") or "").strip().lower()
    if world == "patch":
        if payload.get("patch_profile") not in NATIVE_LINEAGE_PATCH_PROFILES:
            raise ValueError("native Patch lineage metrics input requires a finite P0/P1/P2 profile")
        if payload.get("patch_profile_scope") != NATIVE_LINEAGE_PATCH_PROFILE_SCOPE:
            raise ValueError("native Patch lineage metrics input requires full_profile scope")
        if (
            payload.get("patch_contract_schema_version")
            != NATIVE_LINEAGE_PATCH_CONTRACT_SCHEMA_VERSION
        ):
            raise ValueError("native Patch lineage metrics input has an unsupported contract schema")
        if payload.get("promotion_accepted") is not False:
            raise ValueError("native Patch lineage metrics input requires promotion_accepted=false")
    elif world != "flat":
        raise ValueError("native lineage metrics input is only supported for Flat or admitted Patch")
    generation = _native_nonnegative_int(
        _native_required(payload, "generation", context="native lineage metrics input"),
        context="native lineage metrics input.generation",
    )
    if generation != int(expected_generation):
        raise ValueError(
            "native lineage metrics input generation mismatch: "
            f"expected={int(expected_generation)} actual={generation}"
        )
    raw_records = _native_required(payload, "records", context="native lineage metrics input")
    if isinstance(raw_records, (str, bytes)) or not isinstance(raw_records, Sequence):
        raise ValueError("native lineage metrics input records must be a sequence")

    records: list[NativeLineageAgentRecord] = []
    seen_ids: set[int] = set()
    for index, raw_record in enumerate(raw_records):
        context = f"native lineage metrics input records[{index}]"
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"{context} must be a mapping")
        agent_id = _native_nonnegative_int(
            _native_required(raw_record, "agent_id", context=context),
            context=f"{context}.agent_id",
        )
        if agent_id in seen_ids:
            raise ValueError(f"native lineage metrics input has duplicate agent_id {agent_id}")
        seen_ids.add(agent_id)

        raw_stats = _native_required(raw_record, "stats", context=context)
        raw_genome = _native_required(raw_record, "genome", context=context)
        if not isinstance(raw_stats, Mapping) or not isinstance(raw_genome, Mapping):
            raise ValueError(f"{context} stats and genome must be mappings")
        founder_id = _native_nonnegative_int(
            _native_required(raw_stats, "founder_id", context=f"{context}.stats"),
            context=f"{context}.stats.founder_id",
        )
        lineage_id = str(_native_required(raw_stats, "lineage_id", context=f"{context}.stats") or "")
        if founder_id < 0 or lineage_id != format_lineage_id(founder_id):
            raise ValueError(
                f"{context} has inconsistent Flat lineage identity: "
                f"founder_id={founder_id} lineage_id={lineage_id!r}"
            )
        first_repro_raw = _native_required(raw_stats, "first_repro_step", context=f"{context}.stats")
        first_repro = (
            None
            if first_repro_raw is None
            else _native_nonnegative_int(first_repro_raw, context=f"{context}.stats.first_repro_step")
        )
        confidence_raw = _native_required(raw_stats, "role_confidence", context=f"{context}.stats")
        role_confidence = (
            None
            if confidence_raw is None
            else _native_finite_float(confidence_raw, context=f"{context}.stats.role_confidence")
        )
        stats = NativeLineageStatsRecord(
            lineage_id=lineage_id,
            founder_id=founder_id,
            parent_ids=_native_int_tuple(
                _native_required(raw_stats, "parent_ids", context=f"{context}.stats"),
                context=f"{context}.stats.parent_ids",
            ),
            parent_lineage_ids=_native_str_tuple(
                _native_required(raw_stats, "parent_lineage_ids", context=f"{context}.stats"),
                context=f"{context}.stats.parent_lineage_ids",
            ),
            birth_generation=_native_nonnegative_int(
                _native_required(raw_stats, "birth_generation", context=f"{context}.stats"),
                context=f"{context}.stats.birth_generation",
            ),
            birth_tick=_native_nonnegative_int(
                _native_required(raw_stats, "birth_tick", context=f"{context}.stats"),
                context=f"{context}.stats.birth_tick",
            ),
            lineage_depth=_native_nonnegative_int(
                _native_required(raw_stats, "lineage_depth", context=f"{context}.stats"),
                context=f"{context}.stats.lineage_depth",
            ),
            offspring_count=_native_nonnegative_int(
                _native_required(raw_stats, "offspring_count", context=f"{context}.stats"),
                context=f"{context}.stats.offspring_count",
            ),
            survival_steps=_native_finite_float(
                _native_required(raw_stats, "survival_steps", context=f"{context}.stats"),
                context=f"{context}.stats.survival_steps",
            ),
            energy_collected=_native_finite_float(
                _native_required(raw_stats, "energy_collected", context=f"{context}.stats"),
                context=f"{context}.stats.energy_collected",
            ),
            first_repro_step=first_repro,
            pred_err_count=_native_nonnegative_int(
                _native_required(raw_stats, "pred_err_count", context=f"{context}.stats"),
                context=f"{context}.stats.pred_err_count",
            ),
            pred_err_sum=_native_finite_float(
                _native_required(raw_stats, "pred_err_sum", context=f"{context}.stats"),
                context=f"{context}.stats.pred_err_sum",
            ),
            episode0_pred_err_mean=_native_finite_float(
                _native_required(raw_stats, "episode0_pred_err_mean", context=f"{context}.stats"),
                context=f"{context}.stats.episode0_pred_err_mean",
            ),
            identity_drift_ema=_native_finite_float(
                _native_required(raw_stats, "identity_drift_ema", context=f"{context}.stats"),
                context=f"{context}.stats.identity_drift_ema",
            ),
            drive_jitter=_native_finite_float(
                _native_required(raw_stats, "drive_jitter", context=f"{context}.stats"),
                context=f"{context}.stats.drive_jitter",
            ),
            ecological_role=str(
                _native_required(raw_stats, "ecological_role", context=f"{context}.stats")
                or DEFAULT_ECOLOGICAL_ROLE
            ),
            role_source=str(_native_required(raw_stats, "role_source", context=f"{context}.stats") or "default"),
            role_confidence=role_confidence,
        )
        genome = NativeLineageGenomeRecord(
            hidden_size=_native_nonnegative_int(
                _native_required(raw_genome, "hidden_size", context=f"{context}.genome"),
                context=f"{context}.genome.hidden_size",
            ),
            memory_dim=_native_nonnegative_int(
                _native_required(raw_genome, "memory_dim", context=f"{context}.genome"),
                context=f"{context}.genome.memory_dim",
            ),
            use_gru=bool(_native_required(raw_genome, "use_gru", context=f"{context}.genome")),
            aggression_bias=_native_finite_float(
                _native_required(raw_genome, "aggression_bias", context=f"{context}.genome"),
                context=f"{context}.genome.aggression_bias",
            ),
            exploration_bias=_native_finite_float(
                _native_required(raw_genome, "exploration_bias", context=f"{context}.genome"),
                context=f"{context}.genome.exploration_bias",
            ),
        )
        raw_drives = _native_required(raw_record, "drives", context=context)
        if isinstance(raw_drives, (str, bytes)) or not isinstance(raw_drives, Sequence) or len(raw_drives) != 4:
            raise ValueError(f"{context}.drives must contain exactly four values")
        drives = tuple(
            _native_finite_float(value, context=f"{context}.drives[{drive_index}]")
            for drive_index, value in enumerate(raw_drives)
        )
        records.append(
            NativeLineageAgentRecord(
                agent_id=agent_id,
                energy=_native_finite_float(
                    _native_required(raw_record, "energy", context=context),
                    context=f"{context}.energy",
                ),
                health=_native_finite_float(
                    _native_required(raw_record, "health", context=context),
                    context=f"{context}.health",
                ),
                injury=_native_finite_float(
                    _native_required(raw_record, "injury", context=context),
                    context=f"{context}.injury",
                ),
                drives=drives,  # type: ignore[arg-type]
                genome=genome,
                stats=stats,
                y=_native_nonnegative_int(
                    _native_required(raw_record, "y", context=context),
                    context=f"{context}.y",
                ),
                x=_native_nonnegative_int(
                    _native_required(raw_record, "x", context=context),
                    context=f"{context}.x",
                ),
            )
        )

    ids = [record.agent_id for record in records]
    if ids != sorted(ids):
        raise ValueError("native lineage metrics input records are not in canonical agent_id order")
    expected_count = _native_nonnegative_int(
        _native_required(payload, "record_count", context="native lineage metrics input"),
        context="native lineage metrics input.record_count",
    )
    if expected_count != len(records):
        raise ValueError(
            "native lineage metrics input record_count mismatch: "
            f"expected={expected_count} actual={len(records)}"
        )
    declared_ids = _native_int_tuple(
        _native_required(payload, "agent_ids", context="native lineage metrics input"),
        context="native lineage metrics input.agent_ids",
    )
    if declared_ids != tuple(ids):
        raise ValueError("native lineage metrics input agent_ids disagree with records")
    return tuple(records)


def _lineage_stats_for_agent(agent: Any, *, generation: int = 0, tick: int = 0) -> Any:
    if isinstance(agent, NativeLineageAgentRecord):
        return agent.stats
    aid = _agent_id(agent)
    return normalize_lineage_stats(getattr(agent, "stats", None), aid, generation=generation, tick=tick)


def _lineage_id_for_agent(agent: Any) -> str:
    aid = _agent_id(agent)
    stats = _lineage_stats_for_agent(agent, generation=0, tick=0)
    if stats is None:
        return format_lineage_id(aid)
    return str(getattr(stats, "lineage_id", "") or format_lineage_id(aid))


def _agent_genome(agent: Any) -> Any:
    return getattr(agent, "genome", None)


def _stat(agent: Any) -> Any:
    return getattr(agent, "stats", None)


def _optional_attr(obj: Any, name: str) -> Optional[float]:
    if obj is None or not hasattr(obj, name):
        return None
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _drive_value(agent: Any, index: int) -> Optional[float]:
    drives = getattr(agent, "drives", None)
    if drives is None:
        return None
    try:
        if len(drives) <= int(index):
            return None
        return float(drives[int(index)])
    except Exception:
        return None


def _prediction_error_value(stats: Any) -> Optional[float]:
    if stats is None:
        return None
    count = _safe_int(getattr(stats, "pred_err_count", 0), 0)
    if count > 0:
        return _safe_float(getattr(stats, "pred_err_sum", 0.0), 0.0) / float(count)
    if hasattr(stats, "episode0_pred_err_mean"):
        return _optional_attr(stats, "episode0_pred_err_mean")
    return None


def _mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals)) / float(len(vals))


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _round_score(value: Any) -> float:
    try:
        return round(float(value), 6)
    except Exception:
        return 0.0


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _stable_hash(payload: Mapping[str, Any], *, prefix: str, length: int = 16) -> str:
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return f"{prefix}_{digest[: int(length)]}"


def _sha256_path(path: str) -> Optional[str]:
    if not path:
        return None
    try:
        if os.path.isfile(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return "sha256:" + h.hexdigest()
        if os.path.isdir(path):
            h = hashlib.sha256()
            for root, _, files in os.walk(path):
                for name in sorted(files):
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, path).replace("\\", "/")
                    h.update(rel.encode("utf-8"))
                    with open(full, "rb") as f:
                        for chunk in iter(lambda: f.read(1024 * 1024), b""):
                            h.update(chunk)
            return "sha256:" + h.hexdigest()
    except Exception:
        return None
    return None


def _scenario_config_hash(sim_cfg: Any) -> Optional[str]:
    if sim_cfg is None:
        return None
    try:
        if hasattr(sim_cfg, "__dataclass_fields__"):
            payload = {
                str(k): getattr(sim_cfg, k)
                for k in sorted(getattr(sim_cfg, "__dataclass_fields__", {}).keys())
                if not str(k).startswith("_")
            }
        elif isinstance(sim_cfg, Mapping):
            payload = {str(k): v for k, v in sorted(sim_cfg.items()) if not str(k).startswith("_")}
        else:
            payload = {
                str(k): getattr(sim_cfg, k)
                for k in sorted(dir(sim_cfg))
                if not str(k).startswith("_") and not callable(getattr(sim_cfg, k, None))
            }
        return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    except Exception:
        return None


def _git_commit_hash(sim_cfg: Any) -> Optional[str]:
    for value in (
        getattr(sim_cfg, "git_commit_hash", None) if sim_cfg is not None else None,
        os.environ.get("EIEL_GIT_COMMIT"),
        os.environ.get("GIT_COMMIT"),
    ):
        if value:
            return str(value)
    return None


def _role_payload(role: Optional[str] = None, *, source: str = "default", confidence: Optional[float] = None) -> Dict[str, Any]:
    return {
        "ecological_role": str(role or DEFAULT_ECOLOGICAL_ROLE),
        "lineage_role": str(role or DEFAULT_ECOLOGICAL_ROLE),
        "role_source": str(source or "default"),
        "role_confidence": confidence,
    }


def _agent_role(agent: Any) -> Dict[str, Any]:
    stats = getattr(agent, "stats", None)
    role = getattr(stats, "ecological_role", None) if stats is not None else None
    source = getattr(stats, "role_source", "default") if stats is not None else "default"
    confidence = getattr(stats, "role_confidence", None) if stats is not None else None
    return _role_payload(role, source=str(source or "default"), confidence=confidence)


def _role_summary(members: Iterable[Any]) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    for agent in members:
        counts[str(_agent_role(agent).get("ecological_role", DEFAULT_ECOLOGICAL_ROLE))] += 1
    if not counts:
        return _role_payload()
    dominant_role, dominant_count = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[0]
    total = sum(counts.values())
    payload = _role_payload(dominant_role)
    payload["role_counts"] = {str(k): int(v) for k, v in sorted(counts.items())}
    payload["dominant_role_share"] = float(dominant_count) / float(max(1, total))
    return payload


def _safe_genome_hash(agent: Any) -> Optional[str]:
    genome = _agent_genome(agent)
    if genome is None:
        return None
    for method in ("canonical_hash", "stable_hash"):
        fn = getattr(genome, method, None)
        if callable(fn):
            try:
                return str(fn())
            except Exception:
                continue
    try:
        if hasattr(genome, "to_dict"):
            return "sha256:" + hashlib.sha256(_canonical_json_bytes(genome.to_dict())).hexdigest()
    except Exception:
        return None
    return None


def _representatives_for_lineage(agents: Iterable[Any], lineage_id: str, *, limit: int = 3) -> Dict[str, Any]:
    reps = [a for a in _sorted_agents(agents) if _lineage_id_for_agent(a) == str(lineage_id)]
    reps = reps[: max(0, int(limit))]
    hashes = [_safe_genome_hash(a) for a in reps]
    return {
        "representative_agent_ids": [_agent_id(a) for a in reps],
        "representative_genome_hashes": [str(h) for h in hashes if h],
    }


def _lineage_groups(agents: Iterable[Any]) -> Dict[str, List[Any]]:
    groups: Dict[str, List[Any]] = {}
    for agent in _sorted_agents(agents):
        lid = _lineage_id_for_agent(agent)
        groups.setdefault(lid, []).append(agent)
    return groups


def _descriptor_for_lineage(lineage_id: str, members: List[Any], total_population: int) -> Dict[str, Any]:
    pop_count = int(len(members))
    pop_share = float(pop_count) / float(max(1, int(total_population)))
    role = _role_summary(members)
    stats = [_stat(a) for a in members]
    genomes = [_agent_genome(a) for a in members]
    memory_dims = [_optional_attr(g, "memory_dim") for g in genomes]
    hidden_sizes = [_optional_attr(g, "hidden_size") for g in genomes]
    gru_values = [1.0 if bool(getattr(g, "use_gru", False)) else 0.0 for g in genomes if g is not None]
    cognition_proxy_values = []
    for idx, g in enumerate(genomes):
        if g is None:
            continue
        mem = memory_dims[idx] if idx < len(memory_dims) else None
        hidden = hidden_sizes[idx] if idx < len(hidden_sizes) else None
        proxy = 0.0
        seen = False
        if mem is not None:
            proxy += float(mem) / 64.0
            seen = True
        if hidden is not None:
            proxy += float(hidden) / 128.0
            seen = True
        if bool(getattr(g, "use_gru", False)):
            proxy += 1.0
            seen = True
        if seen:
            cognition_proxy_values.append(proxy)

    first_repro_values = [
        _optional_attr(s, "first_repro_step")
        for s in stats
        if _optional_attr(s, "first_repro_step") is not None
    ]
    row: Dict[str, Any] = {
        "lineage_id": str(lineage_id),
        "population_count": pop_count,
        "population_share": pop_share,
        **role,
        "strategy_descriptor_schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "survival_strategy": {
            "mean_survival_steps": _mean_optional(_optional_attr(s, "survival_steps") for s in stats),
            "mean_first_reproduction_step": _mean_optional(first_repro_values),
        },
        "reproduction_strategy": {
            "mean_offspring_count": _mean_optional(_optional_attr(s, "offspring_count") for s in stats),
            "mean_first_reproduction_step": _mean_optional(first_repro_values),
        },
        "energy_strategy": {
            "mean_energy": _mean_optional(_optional_attr(a, "energy") for a in members),
            "mean_energy_collected": _mean_optional(_optional_attr(s, "energy_collected") for s in stats),
        },
        "risk_stress_profile": {
            "mean_health": _mean_optional(_optional_attr(a, "health") for a in members),
            "mean_injury": _mean_optional(_optional_attr(a, "injury") for a in members),
            "mean_hunger_drive": _mean_optional(_drive_value(a, 0) for a in members),
            "mean_fear_drive": _mean_optional(_drive_value(a, 3) for a in members),
        },
        "cognitive_profile": {
            "mean_memory_dim": _mean_optional(memory_dims),
            "gru_fraction": _mean_optional(gru_values),
            "mean_prediction_error": _mean_optional(_prediction_error_value(s) for s in stats),
            "mean_identity_drift": _mean_optional(_optional_attr(s, "identity_drift_ema") for s in stats),
            "mean_drive_jitter": _mean_optional(_optional_attr(s, "drive_jitter") for s in stats),
            "mean_cognition_complexity_proxy": _mean_optional(cognition_proxy_values),
        },
        "genome_behavior_profile": {
            "mean_aggression_bias": _mean_optional(_optional_attr(g, "aggression_bias") for g in genomes),
            "mean_exploration_bias": _mean_optional(_optional_attr(g, "exploration_bias") for g in genomes),
        },
        "mobility_action_profile": {
            "movement_rate": None,
            "gather_rate": None,
            "rest_rate": None,
            "attack_rate": None,
            "reproduction_action_rate": None,
            "exploration_rate": None,
        },
        "ecology_patch_profile": {
            "patch_use_profile": None,
            "patch_habitat_profile": None,
            "shelter_use_rate": None,
            "predator_exposure_rate": None,
            "predator_death_rate": None,
            "waste_tolerance_proxy": None,
            "disease_survival_proxy": None,
        },
    }
    for section in (
        "survival_strategy",
        "reproduction_strategy",
        "energy_strategy",
        "risk_stress_profile",
        "cognitive_profile",
        "genome_behavior_profile",
    ):
        for key, value in row[section].items():
            row[key] = value
    return row


def _descriptor_numeric_map(descriptor: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for field_name in DESCRIPTOR_VECTOR_FIELDS:
        value = descriptor.get(field_name)
        if value is None:
            continue
        try:
            out[field_name] = float(value)
        except Exception:
            continue
    return out


def descriptor_vector(descriptor: Mapping[str, Any]) -> List[float]:
    numeric = _descriptor_numeric_map(descriptor)
    return [float(numeric[k]) for k in DESCRIPTOR_VECTOR_FIELDS if k in numeric]


def descriptor_vector_from_descriptor(descriptor: Mapping[str, Any]) -> Dict[str, float]:
    numeric = _descriptor_numeric_map(descriptor)
    return {str(k): float(numeric[k]) for k in DESCRIPTOR_VECTOR_FIELDS if k in numeric}


def descriptor_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    amap = _descriptor_numeric_map(a)
    bmap = _descriptor_numeric_map(b)
    keys = [k for k in DESCRIPTOR_VECTOR_FIELDS if k in amap and k in bmap]
    if not keys:
        return 0.0
    acc = 0.0
    for key in keys:
        av = float(amap[key])
        bv = float(bmap[key])
        scale = abs(av) + abs(bv) + 1.0
        acc += ((av - bv) / scale) ** 2
    return float(math.sqrt(acc / float(len(keys))))


def _population_median_descriptor(descriptors: List[Mapping[str, Any]]) -> Dict[str, Any]:
    med: Dict[str, Any] = {"lineage_id": "__population_median__"}
    for field_name in DESCRIPTOR_VECTOR_FIELDS:
        med[field_name] = _median(d.get(field_name) for d in descriptors)
    return med


def compute_lineage_descriptors(
    *,
    generation: int,
    agents: Iterable[Any],
) -> Dict[str, Any]:
    groups = _lineage_groups(agents)
    total = sum(len(v) for v in groups.values())
    lineages = [
        _descriptor_for_lineage(lineage_id, groups[lineage_id], total)
        for lineage_id in sorted(groups.keys())
    ]
    available_fields = sorted(
        field_name
        for field_name in DESCRIPTOR_VECTOR_FIELDS
        if any(desc.get(field_name) is not None for desc in lineages)
    )
    unavailable_fields = sorted(set(DESCRIPTOR_VECTOR_FIELDS) - set(available_fields))
    return {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "generation": _safe_int(generation, 0),
        "lineages": lineages,
        "global_descriptor_summary": {
            "lineage_count": int(len(lineages)),
            "available_numeric_fields": available_fields,
            "unavailable_numeric_fields": unavailable_fields,
        },
    }


def _lineage_row_map(lineage_rows: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for row in lineage_rows or []:
        lid = str(row.get("lineage_id", "") or "")
        if lid:
            out[lid] = row
    return out


def _candidate(
    *,
    generation: int,
    lineage_id: str,
    candidate_type: str,
    score: float,
    reasons: List[str],
    descriptor: Mapping[str, Any],
    lineage_row: Mapping[str, Any],
    supporting_metrics: Mapping[str, Any],
    warning_flags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "generation": _safe_int(generation, 0),
        "lineage_id": str(lineage_id),
        "candidate_type": str(candidate_type),
        "candidate_score": _round_score(score),
        "reasons": list(reasons),
        "population_share": _safe_float(descriptor.get("population_share", 0.0), 0.0),
        "reproduction_share": lineage_row.get("reproduction_share"),
        "persistence_steps": _safe_int(supporting_metrics.get("persistence_steps", 0), 0),
        "ecological_role": str(descriptor.get("ecological_role", DEFAULT_ECOLOGICAL_ROLE) or DEFAULT_ECOLOGICAL_ROLE),
        "lineage_role": str(descriptor.get("lineage_role", DEFAULT_ECOLOGICAL_ROLE) or DEFAULT_ECOLOGICAL_ROLE),
        "role_source": str(descriptor.get("role_source", "default") or "default"),
        "role_confidence": descriptor.get("role_confidence"),
        "behavioral_descriptor_summary": {
            "mean_survival_steps": descriptor.get("mean_survival_steps"),
            "mean_offspring_count": descriptor.get("mean_offspring_count"),
            "mean_memory_dim": descriptor.get("mean_memory_dim"),
            "gru_fraction": descriptor.get("gru_fraction"),
            "mean_injury": descriptor.get("mean_injury"),
            "mean_prediction_error": descriptor.get("mean_prediction_error"),
        },
        "supporting_metrics": dict(supporting_metrics),
        "warning_flags": sorted(str(x) for x in (warning_flags or [])),
    }


def compute_branch_candidates(
    *,
    generation: int,
    summary: Mapping[str, Any],
    lineage_rows: Iterable[Mapping[str, Any]],
    descriptor_record: Mapping[str, Any],
    state: LineageMetricsState,
) -> Dict[str, Any]:
    gen = _safe_int(generation, 0)
    descriptors = list(descriptor_record.get("lineages", []) if isinstance(descriptor_record, Mapping) else [])
    row_by_lineage = _lineage_row_map(lineage_rows)
    dominant_id = str(summary.get("dominant_lineage_id", "") or "")
    dominant_descriptor = next((d for d in descriptors if str(d.get("lineage_id", "")) == dominant_id), None)
    median_descriptor = _population_median_descriptor(descriptors)

    survival_median = _median(d.get("mean_survival_steps") for d in descriptors)
    offspring_median = _median(d.get("mean_offspring_count") for d in descriptors)
    memory_median = _median(d.get("mean_memory_dim") for d in descriptors)
    injury_median = _median(d.get("mean_injury") for d in descriptors)
    cognition_values = [
        _safe_float(d.get("mean_cognition_complexity_proxy"), 0.0)
        for d in descriptors
        if d.get("mean_cognition_complexity_proxy") is not None
    ]
    cognition_median = _median(cognition_values)

    candidates: List[Dict[str, Any]] = []
    for descriptor in sorted(descriptors, key=lambda d: str(d.get("lineage_id", ""))):
        lid = str(descriptor.get("lineage_id", "") or "")
        if not lid:
            continue
        lineage_row = row_by_lineage.get(lid, {})
        pop_share = _safe_float(descriptor.get("population_share", 0.0), 0.0)
        repro_share = lineage_row.get("reproduction_share")
        repro_share_f = _safe_float(repro_share, 0.0) if repro_share is not None else None
        persistence = _safe_int(state.lineage_presence_counts.get(lid, 0), 0)
        divergence_from_dominant = descriptor_distance(descriptor, dominant_descriptor or median_descriptor)
        divergence_from_median = descriptor_distance(descriptor, median_descriptor)
        survival = descriptor.get("mean_survival_steps")
        offspring = descriptor.get("mean_offspring_count")
        memory = descriptor.get("mean_memory_dim")
        gru_fraction = descriptor.get("gru_fraction")
        cognition_proxy = descriptor.get("mean_cognition_complexity_proxy")
        injury = descriptor.get("mean_injury")
        health = descriptor.get("mean_health")

        base_metrics = {
            "persistence_steps": persistence,
            "population_share": pop_share,
            "reproduction_share": repro_share,
            "descriptor_distance_from_dominant": _round_score(divergence_from_dominant),
            "descriptor_distance_from_population_median": _round_score(divergence_from_median),
            "mean_survival_steps": survival,
            "mean_offspring_count": offspring,
            "mean_memory_dim": memory,
            "gru_fraction": gru_fraction,
            "mean_cognition_complexity_proxy": cognition_proxy,
            "mean_injury": injury,
            "mean_health": health,
        }

        if pop_share <= CANDIDATE_RARE_SHARE_MAX and persistence >= CANDIDATE_RARE_MIN_PERSISTENCE_STEPS:
            rarity_score = max(0.0, (CANDIDATE_RARE_SHARE_MAX - pop_share) / CANDIDATE_RARE_SHARE_MAX)
            score = 0.60 * rarity_score + 0.40 * min(1.0, persistence / 5.0)
            candidates.append(
                _candidate(
                    generation=gen,
                    lineage_id=lid,
                    candidate_type="rare_persistent",
                    score=score,
                    reasons=[f"population_share below rare threshold while present for {persistence} steps"],
                    descriptor=descriptor,
                    lineage_row=lineage_row,
                    supporting_metrics={**base_metrics, "rarity_score": _round_score(rarity_score)},
                )
            )

        if pop_share >= CANDIDATE_DOMINANT_SHARE_MIN or (repro_share_f is not None and repro_share_f >= CANDIDATE_DOMINANT_SHARE_MIN):
            dominance = max(pop_share, repro_share_f if repro_share_f is not None else 0.0)
            candidates.append(
                _candidate(
                    generation=gen,
                    lineage_id=lid,
                    candidate_type="dominant",
                    score=dominance,
                    reasons=["population or reproduction share exceeds dominance tracking threshold"],
                    descriptor=descriptor,
                    lineage_row=lineage_row,
                    supporting_metrics={**base_metrics, "dominance_score": _round_score(dominance)},
                    warning_flags=["dominance_is_not_quality"],
                )
            )

        if divergence_from_dominant >= CANDIDATE_DIVERGENCE_MIN or divergence_from_median >= CANDIDATE_DIVERGENCE_MIN:
            div_score = max(divergence_from_dominant, divergence_from_median)
            candidates.append(
                _candidate(
                    generation=gen,
                    lineage_id=lid,
                    candidate_type="behaviorally_divergent",
                    score=div_score,
                    reasons=["descriptor distance from dominant lineage or population median exceeds threshold"],
                    descriptor=descriptor,
                    lineage_row=lineage_row,
                    supporting_metrics={**base_metrics, "behavioral_divergence_score": _round_score(div_score)},
                )
            )

        cog = _safe_float(cognition_proxy, 0.0) if cognition_proxy is not None else None
        cog_med = _safe_float(cognition_median, 0.0) if cognition_median is not None else None
        if cog is not None and cog_med is not None and cog > cog_med and pop_share > 0.0:
            score = min(1.0, (cog - cog_med) / (abs(cog) + abs(cog_med) + 1.0) + pop_share)
            candidates.append(
                _candidate(
                    generation=gen,
                    lineage_id=lid,
                    candidate_type="high_cognition_viable",
                    score=score,
                    reasons=["cognition or memory complexity proxy is above population median while lineage remains viable"],
                    descriptor=descriptor,
                    lineage_row=lineage_row,
                    supporting_metrics={**base_metrics, "cognition_interest_score": _round_score(score)},
                )
            )

        mem = _safe_float(memory, 0.0) if memory is not None else None
        mem_med = _safe_float(memory_median, 0.0) if memory_median is not None else None
        off = _safe_float(offspring, 0.0) if offspring is not None else None
        off_med = _safe_float(offspring_median, 0.0) if offspring_median is not None else None
        surv = _safe_float(survival, 0.0) if survival is not None else None
        surv_med = _safe_float(survival_median, 0.0) if survival_median is not None else None
        if mem is not None and mem_med is not None and mem <= mem_med and (
            (off is not None and off_med is not None and off >= off_med and off > 0.0)
            or (surv is not None and surv_med is not None and surv >= surv_med and surv > 0.0)
        ):
            score = min(1.0, pop_share + 0.25 + (0.25 if off is not None and off_med is not None and off >= off_med else 0.0))
            candidates.append(
                _candidate(
                    generation=gen,
                    lineage_id=lid,
                    candidate_type="low_cognition_strong_baseline",
                    score=score,
                    reasons=["low cognitive cost with survival or reproduction at/above population median"],
                    descriptor=descriptor,
                    lineage_row=lineage_row,
                    supporting_metrics={**base_metrics, "baseline_strength_score": _round_score(score)},
                )
            )

        inj = _safe_float(injury, 0.0) if injury is not None else None
        inj_med = _safe_float(injury_median, 0.0) if injury_median is not None else None
        if inj is not None and inj_med is not None and surv is not None and surv_med is not None and inj <= inj_med and surv >= surv_med:
            score = min(1.0, 0.5 + (surv - surv_med) / (abs(surv) + abs(surv_med) + 1.0))
            candidates.append(
                _candidate(
                    generation=gen,
                    lineage_id=lid,
                    candidate_type="stress_resistant",
                    score=score,
                    reasons=["survival proxy is at/above median while injury/stress proxy is at/below median"],
                    descriptor=descriptor,
                    lineage_row=lineage_row,
                    supporting_metrics={**base_metrics, "stress_resistance_score": _round_score(score)},
                )
            )

        if (repro_share_f is not None and repro_share_f > 0.0 and repro_share_f >= top_n_share({lid: 1}, 1) * 0.25) or (
            off is not None and off_med is not None and off > off_med and off > 0.0
        ):
            score = max(repro_share_f if repro_share_f is not None else 0.0, min(1.0, (off or 0.0) / ((off_med or 0.0) + (off or 0.0) + 1.0)))
            candidates.append(
                _candidate(
                    generation=gen,
                    lineage_id=lid,
                    candidate_type="reproductive_specialist",
                    score=score,
                    reasons=["offspring contribution is above population median or reproduction share is notable"],
                    descriptor=descriptor,
                    lineage_row=lineage_row,
                    supporting_metrics={**base_metrics, "reproduction_score": _round_score(score)},
                )
            )

        if surv is not None and surv_med is not None and surv > surv_med:
            score = min(1.0, (surv - surv_med) / (abs(surv) + abs(surv_med) + 1.0) + 0.5)
            candidates.append(
                _candidate(
                    generation=gen,
                    lineage_id=lid,
                    candidate_type="survival_specialist",
                    score=score,
                    reasons=["survival or age proxy is above population median"],
                    descriptor=descriptor,
                    lineage_row=lineage_row,
                    supporting_metrics={**base_metrics, "survival_score": _round_score(score)},
                )
            )

    candidates = sorted(
        candidates,
        key=lambda c: (-_safe_float(c.get("candidate_score", 0.0), 0.0), str(c.get("candidate_type", "")), str(c.get("lineage_id", ""))),
    )
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "generation": gen,
        "candidates": candidates,
        "global_summary": {
            "candidate_count": len(candidates),
            "candidate_types": sorted({str(c.get("candidate_type", "")) for c in candidates}),
            "dominant_lineage_id": dominant_id,
            "scoring_note": "diagnostic heuristic only; no live evolution changes",
        },
    }


def _descriptor_map(descriptor_record: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    if not isinstance(descriptor_record, Mapping):
        return out
    for desc in descriptor_record.get("lineages", []) or []:
        if isinstance(desc, Mapping):
            lid = str(desc.get("lineage_id", "") or "")
            if lid:
                out[lid] = desc
    return out


def _checkpoint_reference_for_generation(run_dir: str, generation: int) -> Optional[str]:
    if not run_dir:
        return None
    rel = os.path.join("checkpoint_bundles_v2", f"bundle_gen{int(generation):06d}")
    path = os.path.join(str(run_dir), rel)
    if os.path.exists(path):
        return rel.replace("\\", "/")
    return None


def _checkpoint_hash_for_reference(run_dir: str, checkpoint_reference: Optional[str]) -> Optional[str]:
    if not run_dir or not checkpoint_reference:
        return None
    return _sha256_path(os.path.join(str(run_dir), str(checkpoint_reference)))


def _branch_replay_eligibility(checkpoint_reference: Optional[str], checkpoint_hash: Optional[str]) -> Dict[str, Any]:
    eligible = bool(checkpoint_reference and checkpoint_hash)
    return {
        "eligible": eligible,
        "mode": "checkpoint_backed" if eligible else "metadata_only",
        "reason": "trusted checkpoint reference and hash present" if eligible else "no trusted checkpoint reference/hash",
    }


def upgrade_qd_archive_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a v2-compatible archive entry without mutating the input."""
    out = dict(entry or {})
    if str(out.get("schema_version", "")) == ARCHIVE_ENTRY_SCHEMA_VERSION:
        return out
    checkpoint_reference = out.get("checkpoint_reference")
    checkpoint_hash = out.get("checkpoint_hash")
    representative_hashes = out.get("genome_hashes", out.get("representative_genome_hashes", []))
    supporting_metrics = dict(out.get("supporting_metrics", {}) or {})
    out.update(
        {
            "schema_version": ARCHIVE_ENTRY_SCHEMA_VERSION,
            "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
            "checkpoint_hash": checkpoint_hash,
            "scenario_config_hash": out.get("scenario_config_hash", out.get("config_hash")),
            "git_commit_hash": out.get("git_commit_hash"),
            "seed": out.get("seed"),
            "tick": out.get("tick"),
            "founder_id": out.get("founder_id"),
            "founder_group_id": out.get("founder_group_id", out.get("founder_id")),
            "genome_hashes": list(representative_hashes or []),
            "descriptor_vector": out.get("descriptor_vector", []),
            "capture_reasons": list(out.get("capture_reasons", out.get("candidate_reasons", [])) or []),
            "supporting_metrics": supporting_metrics,
            "live_population_context": out.get(
                "live_population_context",
                {
                    "lineage_share_at_capture": out.get("lineage_share_at_capture", out.get("population_share")),
                    "reproduction_share_at_capture": out.get("reproduction_share"),
                },
            ),
            "lineage_share_at_capture": out.get("lineage_share_at_capture", out.get("population_share")),
            "branch_replay_eligibility": out.get(
                "branch_replay_eligibility",
                _branch_replay_eligibility(
                    str(checkpoint_reference) if checkpoint_reference else None,
                    str(checkpoint_hash) if checkpoint_hash else None,
                ),
            ),
            "artifact_hash": out.get("artifact_hash"),
            "trusted_artifact_policy": out.get("trusted_artifact_policy", "checkpoint_backed_required_for_replay"),
            "claims_allowed": list(out.get("claims_allowed", ["metadata-only lineage archive entry"]) or []),
            "claims_not_allowed": list(
                out.get(
                    "claims_not_allowed",
                    [
                        "archive diversity is not live diversity",
                        "archive entries must not influence live selection",
                    ],
                )
                or []
            ),
            "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
        }
    )
    return out


def validate_qd_archive_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    upgraded = upgrade_qd_archive_entry(entry)
    errors: list[dict[str, str]] = []
    if not upgraded.get("archive_entry_id"):
        errors.append({"field": "archive_entry_id", "error": "required"})
    if not upgraded.get("lineage_id"):
        errors.append({"field": "lineage_id", "error": "required"})
    if "claims_allowed" not in upgraded:
        errors.append({"field": "claims_allowed", "error": "required"})
    if "claims_not_allowed" not in upgraded:
        errors.append({"field": "claims_not_allowed", "error": "required"})
    eligibility = upgraded.get("branch_replay_eligibility", {})
    replay_eligible = bool(eligibility.get("eligible")) if isinstance(eligibility, Mapping) else False
    if replay_eligible:
        for field_name in ("checkpoint_reference", "checkpoint_hash", "scenario_config_hash", "generation", "seed"):
            if upgraded.get(field_name) in (None, ""):
                errors.append({"field": field_name, "error": "required_for_replay_eligible_archive_entry"})
    if not replay_eligible and not upgraded.get("checkpoint_reference"):
        upgraded["branch_replay_eligibility"] = _branch_replay_eligibility(None, None)
    return {
        "ok": not errors,
        "schema_version": "theme5_lineage_qd_archive_entry_validation_v1",
        "errors": errors,
        "entry": upgraded,
    }


def _archive_entry_id(payload: Mapping[str, Any]) -> str:
    base = {
        "generation": _safe_int(payload.get("generation", 0), 0),
        "lineage_id": str(payload.get("lineage_id", "") or ""),
        "candidate_type": str(payload.get("candidate_type", "") or ""),
        "candidate_score": _round_score(payload.get("candidate_score", 0.0)),
        "supporting_metrics": payload.get("supporting_metrics", {}),
    }
    return _stable_hash(base, prefix="t5qd", length=20)


def compute_qd_archive_entries(
    *,
    generation: int,
    candidate_record: Mapping[str, Any],
    descriptor_record: Mapping[str, Any],
    agents: Iterable[Any],
    state: LineageMetricsState,
    run_dir: str = "",
    sim_cfg: Any = None,
) -> List[Dict[str, Any]]:
    """Select deterministic metadata-only archive entries from candidates."""
    if not bool(getattr(sim_cfg, "lineage_qd_archive_enable", True)):
        return []
    gen = _safe_int(generation, 0)
    top_n = max(0, _safe_int(getattr(sim_cfg, "lineage_qd_archive_top_n", ARCHIVE_DEFAULT_TOP_N), ARCHIVE_DEFAULT_TOP_N))
    min_score = _safe_float(getattr(sim_cfg, "lineage_qd_archive_min_score", ARCHIVE_DEFAULT_MIN_SCORE), ARCHIVE_DEFAULT_MIN_SCORE)
    cooldown = max(0, _safe_int(getattr(sim_cfg, "lineage_qd_archive_cooldown_generations", 0), 0))
    descriptors = _descriptor_map(descriptor_record)
    candidates = [
        dict(c)
        for c in (candidate_record.get("candidates", []) if isinstance(candidate_record, Mapping) else [])
        if isinstance(c, Mapping) and _safe_float(c.get("candidate_score", 0.0), 0.0) >= min_score
    ]
    candidates = sorted(
        candidates,
        key=lambda c: (
            -_safe_float(c.get("candidate_score", 0.0), 0.0),
            str(c.get("candidate_type", "")),
            str(c.get("lineage_id", "")),
        ),
    )[:top_n]

    entries: List[Dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        lid = str(candidate.get("lineage_id", "") or "")
        ctype = str(candidate.get("candidate_type", "") or "")
        if not lid or not ctype:
            continue
        dedupe_key = f"{lid}:{ctype}"
        last_gen = state.archive_last_generation_by_key.get(dedupe_key)
        if last_gen is not None and cooldown > 0 and (gen - int(last_gen)) < cooldown:
            continue
        desc = descriptors.get(lid, {})
        reps = _representatives_for_lineage(agents, lid)
        checkpoint_reference = _checkpoint_reference_for_generation(str(run_dir), gen)
        checkpoint_hash = _checkpoint_hash_for_reference(str(run_dir), checkpoint_reference)
        descriptor_vector = descriptor_vector_from_descriptor(desc) if desc else []
        entry = {
            "schema_version": ARCHIVE_ENTRY_SCHEMA_VERSION,
            "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
            "archive_policy_version": ARCHIVE_POLICY_VERSION,
            "archive_entry_id": "",
            "generation": gen,
            "tick": None,
            "lineage_id": lid,
            "founder_id": desc.get("founder_id") if isinstance(desc, Mapping) else None,
            "founder_group_id": desc.get("founder_group_id", desc.get("founder_id")) if isinstance(desc, Mapping) else None,
            "candidate_type": ctype,
            "candidate_types": [ctype],
            "candidate_score": _round_score(candidate.get("candidate_score", 0.0)),
            "candidate_rank": int(rank),
            "candidate_reasons": list(candidate.get("reasons", []) or []),
            "capture_reasons": list(candidate.get("reasons", []) or []),
            "population_share": candidate.get("population_share"),
            "reproduction_share": candidate.get("reproduction_share"),
            "lineage_share_at_capture": candidate.get("population_share"),
            "descriptor_summary": candidate.get("behavioral_descriptor_summary") or {
                "mean_survival_steps": desc.get("mean_survival_steps"),
                "mean_offspring_count": desc.get("mean_offspring_count"),
                "mean_memory_dim": desc.get("mean_memory_dim"),
                "gru_fraction": desc.get("gru_fraction"),
                "mean_injury": desc.get("mean_injury"),
            },
            "descriptor_vector": descriptor_vector,
            "supporting_metrics": dict(candidate.get("supporting_metrics", {}) or {}),
            "representative_agent_ids": reps["representative_agent_ids"],
            "representative_genome_hashes": reps["representative_genome_hashes"],
            "genome_hashes": reps["representative_genome_hashes"],
            "checkpoint_reference": checkpoint_reference,
            "checkpoint_hash": checkpoint_hash,
            "source_run_dir": str(run_dir or ""),
            "config_hash": _scenario_config_hash(sim_cfg),
            "scenario_config_hash": _scenario_config_hash(sim_cfg),
            "git_commit_hash": _git_commit_hash(sim_cfg),
            "seed": getattr(sim_cfg, "seed", None) if sim_cfg is not None else None,
            "world_context_summary": None,
            "live_population_context": {
                "lineage_share_at_capture": candidate.get("population_share"),
                "reproduction_share_at_capture": candidate.get("reproduction_share"),
                "live_diversity_not_archive_diversity": True,
            },
            "branch_replay_eligibility": _branch_replay_eligibility(checkpoint_reference, checkpoint_hash),
            "artifact_hash": None,
            "trusted_artifact_policy": "checkpoint_backed_required_for_replay",
            "claims_allowed": [
                "checkpoint-backed replay preparation only when trusted checkpoint provenance is present",
                "metadata-only archive coverage and candidate reporting",
            ],
            "claims_not_allowed": [
                "live diversity is not archive diversity",
                "archive entries do not influence live selection, survival, reproduction, or reseeding",
            ],
            "created_by": "theme5_quality_diversity_archive",
            **_role_payload(
                str(candidate.get("ecological_role", DEFAULT_ECOLOGICAL_ROLE) or DEFAULT_ECOLOGICAL_ROLE),
                source=str(candidate.get("role_source", "default") or "default"),
                confidence=candidate.get("role_confidence"),
            ),
        }
        entry["artifact_hash"] = "sha256:" + hashlib.sha256(_canonical_json_bytes(entry)).hexdigest()
        entry["archive_entry_id"] = _archive_entry_id(entry)
        validation = validate_qd_archive_entry(entry)
        if not bool(validation.get("ok")):
            continue
        entry = dict(validation["entry"])
        if entry["archive_entry_id"] in state.archived_entry_ids:
            continue
        state.archived_entry_ids.add(entry["archive_entry_id"])
        state.archive_last_generation_by_key[dedupe_key] = gen
        entries.append(entry)
    return entries


def write_qd_archive_entries(run_dir: str, entries: Iterable[Mapping[str, Any]]) -> None:
    entries = list(entries or [])
    if not entries:
        return
    archive_dir = os.path.join(str(run_dir), "archives", "lineage_qd")
    entries_dir = os.path.join(archive_dir, "entries")
    os.makedirs(entries_dir, exist_ok=True)
    index_path = os.path.join(archive_dir, "archive_index.jsonl")
    with open(index_path, "a", encoding="utf-8") as index:
        for entry in sorted(
            entries,
            key=lambda e: (
                _safe_int(e.get("generation", 0), 0),
                -_safe_float(e.get("candidate_score", 0.0), 0.0),
                str(e.get("candidate_type", "")),
                str(e.get("lineage_id", "")),
                str(e.get("archive_entry_id", "")),
            ),
        ):
            entry_id = str(entry.get("archive_entry_id", "") or "")
            if not entry_id:
                continue
            entry_path = os.path.join(entries_dir, f"{entry_id}.json")
            upgraded = upgrade_qd_archive_entry(entry)
            with open(entry_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(upgraded, indent=2, sort_keys=True) + "\n")
            index.write(json.dumps(upgraded, sort_keys=True, separators=(",", ":")) + "\n")


def compute_lineage_snapshot(
    *,
    generation: int,
    agents: Iterable[Any],
    sim_cfg: Any = None,
    state: Optional[LineageMetricsState] = None,
    genome_diversity: Optional[float] = None,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], LineageMetricsState]:
    """Compute deterministic Theme 5 lineage summary, rows, events, and state."""
    st = state if state is not None else LineageMetricsState()
    gen = _safe_int(generation, 0)
    sorted_agents = _sorted_agents(agents)

    counts: Counter[str] = Counter()
    offspring_counts: Counter[str] = Counter()
    by_lineage: Dict[str, List[Any]] = {}
    founder_ids: Dict[str, int] = {}
    for agent in sorted_agents:
        aid = _agent_id(agent)
        ast = _lineage_stats_for_agent(agent, generation=0, tick=0)
        lid = str(getattr(ast, "lineage_id", format_lineage_id(aid)) if ast is not None else format_lineage_id(aid))
        counts[lid] += 1
        by_lineage.setdefault(lid, []).append(agent)
        offspring_counts[lid] += max(0, _safe_int(getattr(ast, "offspring_count", 0) if ast is not None else 0, 0))
        if lid not in founder_ids:
            founder_ids[lid] = _safe_int(getattr(ast, "founder_id", aid) if ast is not None else aid, aid)

    active = int(len(counts))
    total = float(sum(counts.values()))
    dominant_lineage_id = ""
    dominant_population_share = 0.0
    if counts:
        dominant_lineage_id, dominant_count = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[0]
        dominant_population_share = float(dominant_count) / float(max(1.0, total))
    top3_population_share = top_n_share(counts, 3)
    population_entropy = lineage_entropy(counts)

    reproduction_total = float(sum(max(0, int(v)) for v in offspring_counts.values()))
    reproduction_entropy: Optional[float] = None
    dominant_reproduction_share: Optional[float] = None
    if reproduction_total > 0.0:
        reproduction_entropy = lineage_entropy(offspring_counts)
        dominant_reproduction_share = top_n_share(offspring_counts, 1)

    rare_persistent_lineages = 0
    per_lineage_rows: List[Dict[str, Any]] = []
    for lid in sorted(by_lineage.keys()):
        members = by_lineage[lid]
        pop_count = int(len(members))
        pop_share = float(pop_count) / float(max(1.0, total))
        stat_objs = [getattr(a, "stats", None) for a in members]
        depths = [_safe_int(getattr(s, "lineage_depth", 0), 0) for s in stat_objs if s is not None]
        max_depth = max(depths or [0])
        if pop_share <= 0.10 and max_depth >= 2:
            rare_persistent_lineages += 1
        off = int(offspring_counts.get(lid, 0))
        row = {
            "schema_version": "theme5_lineage_metrics_v1",
            "generation": gen,
            "lineage_id": lid,
            "founder_id": int(founder_ids.get(lid, 0)),
            "population_count": pop_count,
            "population_share": pop_share,
            **_role_summary(members),
            "offspring_count": off,
            "reproduction_share": (float(off) / reproduction_total) if reproduction_total > 0.0 else None,
            "mean_survival_steps": _mean(getattr(s, "survival_steps", 0.0) for s in stat_objs if s is not None),
            "mean_energy_collected": _mean(getattr(s, "energy_collected", 0.0) for s in stat_objs if s is not None),
            "mean_lineage_depth": _mean(depths),
            "max_lineage_depth": int(max_depth),
        }
        per_lineage_rows.append(row)

    pop_threshold = _safe_float(getattr(sim_cfg, "lineage_dominance_threshold", 0.80), 0.80)
    pop_streak_required = max(1, _safe_int(getattr(sim_cfg, "lineage_dominance_streak_required", 5), 5))
    min_active = max(0, _safe_int(getattr(sim_cfg, "lineage_min_active_lineages", 2), 2))
    entropy_min = _safe_float(getattr(sim_cfg, "lineage_entropy_minimum", 0.10), 0.10)
    repro_threshold = _safe_float(getattr(sim_cfg, "lineage_reproduction_dominance_threshold", 0.80), 0.80)
    repro_streak_required = max(1, _safe_int(getattr(sim_cfg, "lineage_reproduction_dominance_streak_required", 5), 5))

    if dominant_population_share >= pop_threshold and active > 0:
        st.population_dominance_streak += 1
    else:
        st.population_dominance_streak = 0

    if dominant_reproduction_share is not None and dominant_reproduction_share >= repro_threshold:
        st.reproduction_dominance_streak += 1
    else:
        st.reproduction_dominance_streak = 0

    low_active = active < min_active
    low_entropy = active > 0 and population_entropy < entropy_min
    pop_dominant = st.population_dominance_streak >= pop_streak_required
    repro_dominant = st.reproduction_dominance_streak >= repro_streak_required
    collapse_warning = bool(low_active or low_entropy or pop_dominant or repro_dominant)
    if collapse_warning:
        st.collapse_streak += 1
    else:
        st.collapse_streak = 0

    events: List[Dict[str, Any]] = []
    if dominant_lineage_id and st.last_dominant_lineage_id and dominant_lineage_id != st.last_dominant_lineage_id:
        events.append(
            {
                "schema_version": "theme5_lineage_event_v1",
                "generation": gen,
                "event": "dominant_lineage_changed",
                "previous_lineage_id": st.last_dominant_lineage_id,
                "lineage_id": dominant_lineage_id,
            }
        )
    if collapse_warning:
        payload = {
            "schema_version": "theme5_lineage_event_v1",
            "generation": gen,
            "event": "lineage_collapse_warning_triggered",
            "lineage_id": dominant_lineage_id,
            "active_lineages": active,
            "dominant_lineage_population_share": dominant_population_share,
            "lineage_population_entropy": population_entropy,
            "lineage_collapse_streak": int(st.collapse_streak),
            "low_active_lineage_count": bool(low_active),
            "low_lineage_entropy": bool(low_entropy),
            "population_dominance_streak": int(st.population_dominance_streak),
            "reproduction_dominance_streak": int(st.reproduction_dominance_streak),
        }
        if genome_diversity is not None and _safe_float(genome_diversity, 0.0) > 0.0 and low_entropy:
            payload["genome_diversity_nonzero_with_lineage_entropy_collapse"] = True
        events.append(payload)

    current_lineages = set(counts.keys())
    for lid in sorted(current_lineages):
        if lid not in st.lineage_first_seen_generation:
            st.lineage_first_seen_generation[lid] = gen
        st.lineage_presence_counts[lid] = int(st.lineage_presence_counts.get(lid, 0)) + 1
    for extinct in sorted(st.seen_lineages - current_lineages):
        events.append(
            {
                "schema_version": "theme5_lineage_event_v1",
                "generation": gen,
                "event": "lineage_extinct",
                "lineage_id": extinct,
            }
        )
    for row in per_lineage_rows:
        if float(row.get("population_share", 0.0) or 0.0) <= 0.10 and int(row.get("max_lineage_depth", 0) or 0) >= 2:
            events.append(
                {
                    "schema_version": "theme5_lineage_event_v1",
                    "generation": gen,
                    "event": "rare_persistent_lineage_seen",
                    "lineage_id": row["lineage_id"],
                    "population_share": row["population_share"],
                    "max_lineage_depth": row["max_lineage_depth"],
                }
            )

    st.seen_lineages.update(current_lineages)
    if dominant_lineage_id:
        st.last_dominant_lineage_id = dominant_lineage_id

    summary = {
        "generation": gen,
        "active_lineages": active,
        "dominant_lineage_id": dominant_lineage_id,
        "dominant_lineage_population_share": dominant_population_share,
        "top3_lineage_population_share": top3_population_share,
        "lineage_population_entropy": population_entropy,
        "lineage_reproduction_entropy": reproduction_entropy,
        "dominant_lineage_reproduction_share": dominant_reproduction_share,
        "rare_persistent_lineages": rare_persistent_lineages,
        "lineage_collapse_warning": int(collapse_warning),
        "lineage_collapse_streak": int(st.collapse_streak),
    }
    return summary, per_lineage_rows, events, st


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = [_safe_float(v, 0.0) for v in values]
    if not vals:
        return None
    return float(sum(vals)) / float(len(vals))


class LineageMetricsWriter:
    def __init__(self, run_dir: str):
        self.run_dir = str(run_dir)
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.summary_path = os.path.join(self.metrics_dir, "lineage_summary.csv")
        self.metrics_path = os.path.join(self.metrics_dir, "lineage_metrics.jsonl")
        self.events_path = os.path.join(self.metrics_dir, "lineage_events.jsonl")
        self.descriptors_path = os.path.join(self.metrics_dir, "lineage_descriptors.jsonl")
        self.candidates_path = os.path.join(self.metrics_dir, "branch_candidates.jsonl")
        self.founder_spatial_path = os.path.join(self.metrics_dir, "lineage_founder_spatial_survival.jsonl")
        self.state = LineageMetricsState()
        self._last_written_gen = self._read_last_summary_gen()
        self._ensure_files()

    def _ensure_files(self) -> None:
        if not os.path.exists(self.summary_path) or os.path.getsize(self.summary_path) == 0:
            timing = current_flat_output_cost_timing()
            if timing is None:
                with open(self.summary_path, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=LINEAGE_SUMMARY_COLUMNS).writeheader()
            else:
                with timing.serialization_phase():
                    buffer = io.StringIO(newline="")
                    csv.DictWriter(buffer, fieldnames=LINEAGE_SUMMARY_COLUMNS).writeheader()
                    encoded = buffer.getvalue()
                with timing.write_phase():
                    with open(self.summary_path, "w", newline="", encoding="utf-8") as f:
                        f.write(encoded)
        for path in (
            self.metrics_path,
            self.events_path,
            self.descriptors_path,
            self.candidates_path,
            self.founder_spatial_path,
        ):
            if not os.path.exists(path):
                open(path, "a", encoding="utf-8").close()

    def _read_last_summary_gen(self) -> int:
        try:
            if not os.path.exists(self.summary_path):
                return -1
            with open(self.summary_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return -1
            return _safe_int(rows[-1].get("generation", -1), -1)
        except Exception:
            return -1

    def export_state(self) -> Dict[str, Any]:
        return self.state.to_dict()

    def restore_state(self, payload: Mapping[str, Any]) -> None:
        self.state = LineageMetricsState.from_dict(payload if isinstance(payload, Mapping) else {})

    def log_generation(
        self,
        *,
        generation: int,
        agents: Iterable[Any],
        sim_cfg: Any = None,
        genome_diversity: Optional[float] = None,
        native_lineage_input: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not bool(getattr(sim_cfg, "lineage_observability_enable", True)):
            return
        gen = _safe_int(generation, 0)
        if gen <= int(getattr(self, "_last_written_gen", -1)):
            return
        lineage_agents = agents
        if native_lineage_input is not None:
            lineage_agents = native_lineage_agents_from_input(
                native_lineage_input,
                expected_generation=gen,
            )
        summary, lineage_rows, events, self.state = compute_lineage_snapshot(
            generation=gen,
            agents=lineage_agents,
            sim_cfg=sim_cfg,
            state=self.state,
            genome_diversity=genome_diversity,
        )
        descriptor_record = compute_lineage_descriptors(generation=gen, agents=lineage_agents)
        candidate_record = compute_branch_candidates(
            generation=gen,
            summary=summary,
            lineage_rows=lineage_rows,
            descriptor_record=descriptor_record,
            state=self.state,
        )
        archive_entries = compute_qd_archive_entries(
            generation=gen,
            candidate_record=candidate_record,
            descriptor_record=descriptor_record,
            agents=lineage_agents,
            state=self.state,
            run_dir=self.run_dir,
            sim_cfg=sim_cfg,
        )

        row = dict(summary)
        for key in ("lineage_reproduction_entropy", "dominant_lineage_reproduction_share"):
            if row.get(key) is None:
                row[key] = ""
        timing = current_flat_output_cost_timing()
        if timing is None:
            with open(self.summary_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LINEAGE_SUMMARY_COLUMNS)
                writer.writerow({k: row.get(k, "") for k in LINEAGE_SUMMARY_COLUMNS})
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                for item in lineage_rows:
                    f.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
            with open(self.descriptors_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(descriptor_record, sort_keys=True, separators=(",", ":")) + "\n")
            with open(self.candidates_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(candidate_record, sort_keys=True, separators=(",", ":")) + "\n")
        else:
            with timing.serialization_phase():
                summary_buffer = io.StringIO(newline="")
                csv.DictWriter(summary_buffer, fieldnames=LINEAGE_SUMMARY_COLUMNS).writerow(
                    {k: row.get(k, "") for k in LINEAGE_SUMMARY_COLUMNS}
                )
                summary_text = summary_buffer.getvalue()
                metrics_text = "".join(
                    json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                    for item in lineage_rows
                )
                descriptor_text = json.dumps(
                    descriptor_record,
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n"
                candidate_text = json.dumps(
                    candidate_record,
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n"
            for path, text in (
                (self.summary_path, summary_text),
                (self.metrics_path, metrics_text),
                (self.descriptors_path, descriptor_text),
                (self.candidates_path, candidate_text),
            ):
                with timing.write_phase():
                    with open(path, "a", newline="" if path == self.summary_path else None, encoding="utf-8") as f:
                        f.write(text)
        founder_spatial_record = founder_spatial_survival_record(
            generation=gen,
            agents=lineage_agents,
            initial_plan=getattr(sim_cfg, "_lineage_founder_spatial_initial_metrics", None),
        )
        if timing is None:
            with open(self.founder_spatial_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(founder_spatial_record, sort_keys=True, separators=(",", ":")) + "\n")
        else:
            with timing.serialization_phase():
                founder_text = json.dumps(
                    founder_spatial_record,
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n"
            with timing.write_phase():
                with open(self.founder_spatial_path, "a", encoding="utf-8") as f:
                    f.write(founder_text)
        write_qd_archive_entries(self.run_dir, archive_entries)
        if events:
            if timing is None:
                with open(self.events_path, "a", encoding="utf-8") as f:
                    for event in events:
                        f.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            else:
                with timing.serialization_phase():
                    event_text = "".join(
                        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                        for event in events
                    )
                with timing.write_phase():
                    with open(self.events_path, "a", encoding="utf-8") as f:
                        f.write(event_text)
        self._last_written_gen = gen
