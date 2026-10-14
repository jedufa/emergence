"""simulation/metrics_pkg/phase_e_metrics_engine.py

Phase E1 deterministic evolvability metrics engine.

Hard constraints:
- No mutation of main simulation state.
- Stable ordering for all aggregation paths.
- Optional strict RNG budget contract enforcement (expected zero draws).
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from simulation.control_plane_pkg.util_json import canonical_json_bytes

E1_SCHEMA_VERSION = "phase_e_e1_metrics_v1"
E1_SENTINEL_SCHEMA_VERSION = "phase_e_e1_sentinel_v1"
E2_SCHEMA_VERSION = "phase_e_e2_taxonomy_v1"
E2_ANNOTATION_SCHEMA_VERSION = "phase_e_e2_annotation_v1"


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    data = canonical_json_bytes(payload)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _append_canonical_jsonl(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = canonical_json_bytes(payload).decode("utf-8") + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _structure_hash(agent: Any, hash_cache: Optional[Dict[int, str]] = None) -> str:
    cache_key = id(agent)
    if hash_cache is not None and cache_key in hash_cache:
        return hash_cache[cache_key]

    g = getattr(agent, "genome", None)
    if g is None:
        if hash_cache is not None:
            hash_cache[cache_key] = ""
        return ""
    if hasattr(g, "canonical_hash"):
        try:
            value = str(g.canonical_hash())
            if hash_cache is not None:
                hash_cache[cache_key] = value
            return value
        except Exception:
            if hash_cache is not None:
                hash_cache[cache_key] = ""
            return ""
    try:
        payload = getattr(g, "to_dict", lambda: {})()
        value = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if hash_cache is not None:
            hash_cache[cache_key] = value
        return value
    except Exception:
        if hash_cache is not None:
            hash_cache[cache_key] = ""
        return ""


def _module_ids_from_agent(agent: Any) -> List[str]:
    g = getattr(agent, "genome", None)
    if g is None:
        return []
    try:
        d = g.to_dict() if hasattr(g, "to_dict") else dict(g)
    except Exception:
        d = {}
    mods = d.get("modules", []) if isinstance(d, dict) else []
    out: List[str] = []
    if isinstance(mods, list):
        for m in mods:
            if isinstance(m, dict):
                out.append(str(m.get("module_id", "") or ""))
    out = sorted([m for m in out if m])
    return out


def _agent_function_score(agent: Any, *, knockout_module_id: Optional[str] = None) -> float:
    g = getattr(agent, "genome", None)
    if g is None:
        return 0.0
    try:
        d = g.to_dict() if hasattr(g, "to_dict") else dict(g)
    except Exception:
        return 0.0
    if not isinstance(d, dict):
        return 0.0

    modules = d.get("modules", [])
    connections = d.get("connections", [])
    if not isinstance(modules, list):
        modules = []
    if not isinstance(connections, list):
        connections = []

    if knockout_module_id is not None:
        modules = [m for m in modules if str((m or {}).get("module_id", "")) != str(knockout_module_id)]
        alive_mod_ids = {str((m or {}).get("module_id", "")) for m in modules if isinstance(m, dict)}
        connections = [
            c
            for c in connections
            if isinstance(c, dict)
            and str(c.get("src_module_id", "")) in alive_mod_ids
            and str(c.get("dst_module_id", "")) in alive_mod_ids
        ]

    module_n = float(len(modules))
    conn_n = float(len(connections))
    locality_cost = _safe_float(d.get("b3_locality_cost_total", 0.0), 0.0)

    # Deterministic structural proxy used only for E1 measurement.
    return max(0.0, module_n + 0.25 * conn_n - 0.10 * locality_cost)


def _population_structure_hash(sim: Any, generation_index: int, hash_cache: Optional[Dict[int, str]] = None) -> str:
    pop = getattr(sim, "pop", None)
    if pop is None:
        return ""
    rows: List[Dict[str, Any]] = []
    try:
        agents = list(pop.alive_agents())
    except Exception:
        agents = []
    agents = sorted(agents, key=lambda a: int(getattr(a, "agent_id", 0)))
    for a in agents:
        rows.append(
            {
                "agent_id": int(getattr(a, "agent_id", 0)),
                "hash": _structure_hash(a, hash_cache=hash_cache),
                "alive": bool(getattr(a, "alive", False)),
            }
        )
    payload = {
        "schema_version": "phase_e_population_hash_v1",
        "generation": int(generation_index),
        "rows": rows,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _phase_e_hash_reuse_stats(sim_cfg: Any) -> Dict[str, int]:
    stats = getattr(sim_cfg, "_phase_e_changed_agent_hash_reuse_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "schema_version": "phase_e_changed_agent_hash_reuse_stats_v1",
            "phasee_after_rows_total": 0,
            "phasee_after_hash_reused": 0,
            "phasee_after_hash_recomputed": 0,
            "phasee_after_new_agents": 0,
            "phasee_after_removed_agents": 0,
            "phasee_version_unknown_fallbacks": 0,
            "phasee_hash_mismatches": 0,
        }
        try:
            setattr(sim_cfg, "_phase_e_changed_agent_hash_reuse_stats", stats)
        except Exception:
            return stats
    return stats


def _full_rng_indices(rm: Any, generation_index: int) -> Dict[str, int]:
    """Return a deterministic full index map (includes explicit zero-valued keys)."""
    out: Dict[str, int] = {}
    if rm is None:
        return out
    try:
        snap = rm.capture_snapshot(generation_index=int(generation_index))
        streams = snap.get("streams", {}) if isinstance(snap, dict) else {}
        if not isinstance(streams, dict):
            streams = {}
        for sname in sorted(streams.keys(), key=str):
            payload = streams.get(sname, {})
            if not isinstance(payload, dict):
                continue
            idx = payload.get("indices", {})
            if not isinstance(idx, dict):
                continue
            out[f"{sname}.py"] = int(idx.get("py", 0) or 0)
            out[f"{sname}.np"] = int(idx.get("np", 0) or 0)
            out[f"{sname}.torch_cpu"] = int(idx.get("torch_cpu", 0) or 0)
            tc = idx.get("torch_cuda", {})
            if isinstance(tc, dict):
                for dev in sorted(tc.keys(), key=str):
                    out[f"{sname}.torch_cuda.{dev}"] = int(tc.get(dev, 0) or 0)
        return out
    except Exception:
        try:
            return {str(k): int(v) for k, v in dict(rm.indices_snapshot()).items()}
        except Exception:
            return {}


@dataclass
class PhaseEMetricsEngine:
    """Deterministic E1 metrics engine (cadence-based, checkpointable)."""

    run_dir: str
    retention_k: int = 3
    neutrality_window: int = 8

    _seen_first_gen: Dict[str, int] = field(default_factory=dict)
    _neutrality_history: List[float] = field(default_factory=list)
    _taxonomy_counts: Dict[str, int] = field(default_factory=lambda: {"return": 0, "shift": 0, "collapse": 0})

    last_record: Optional[Dict[str, Any]] = None
    last_sentinel: Optional[Dict[str, Any]] = None

    def export_state(self) -> Dict[str, Any]:
        return {
            "schema_version": "phase_e_e1_state_v1",
            "retention_k": int(self.retention_k),
            "neutrality_window": int(self.neutrality_window),
            "seen_first_gen": {str(k): int(v) for k, v in sorted(self._seen_first_gen.items(), key=lambda kv: str(kv[0]))},
            "neutrality_history": [float(x) for x in self._neutrality_history],
            "taxonomy_counts": {
                "return": int(self._taxonomy_counts.get("return", 0)),
                "shift": int(self._taxonomy_counts.get("shift", 0)),
                "collapse": int(self._taxonomy_counts.get("collapse", 0)),
            },
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        try:
            self.retention_k = int(state.get("retention_k", self.retention_k) or self.retention_k)
        except Exception:
            pass
        try:
            self.neutrality_window = int(state.get("neutrality_window", self.neutrality_window) or self.neutrality_window)
        except Exception:
            pass

        seen = state.get("seen_first_gen", {})
        if isinstance(seen, dict):
            self._seen_first_gen = {str(k): int(v) for k, v in seen.items()}

        nh = state.get("neutrality_history", [])
        if isinstance(nh, list):
            self._neutrality_history = [float(x) for x in nh]

        tc = state.get("taxonomy_counts", {})
        if isinstance(tc, dict):
            self._taxonomy_counts = {
                "return": int(tc.get("return", 0) or 0),
                "shift": int(tc.get("shift", 0) or 0),
                "collapse": int(tc.get("collapse", 0) or 0),
            }

    def _build_sentinel(
        self,
        sim: Any,
        generation_index: int,
        before_idx: Dict[str, int],
        before_hash: str,
        hash_cache: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        rm = getattr(sim, "rng_manager", None)
        after_idx = _full_rng_indices(rm, generation_index=int(generation_index)) if rm is not None else {}
        sim_cfg = getattr(sim, "sim_cfg", None)
        reuse_enabled = bool(
            sim_cfg is not None and getattr(sim_cfg, "phase_e_changed_agent_hash_reuse_enable", False)
        )
        after_hash_cache = hash_cache if reuse_enabled else None
        after_hash = _population_structure_hash(sim, generation_index, hash_cache=after_hash_cache)
        if reuse_enabled and sim_cfg is not None:
            stats = _phase_e_hash_reuse_stats(sim_cfg)
            row_count = 0
            try:
                pop = getattr(sim, "pop", None)
                row_count = len(list(pop.alive_agents())) if pop is not None else 0
            except Exception:
                row_count = 0
            stats["phasee_after_rows_total"] = int(stats.get("phasee_after_rows_total", 0)) + int(row_count)
            stats["phasee_after_hash_reused"] = int(stats.get("phasee_after_hash_reused", 0)) + int(
                len(hash_cache) if isinstance(hash_cache, dict) else 0
            )
            stats["phasee_after_hash_recomputed"] = int(stats.get("phasee_after_hash_recomputed", 0)) + max(
                0,
                int(row_count) - int(len(hash_cache) if isinstance(hash_cache, dict) else 0),
            )

        keys = sorted(set(before_idx.keys()) | set(after_idx.keys()))
        idx_delta = {k: int(after_idx.get(k, 0)) - int(before_idx.get(k, 0)) for k in keys}

        return {
            "schema_version": E1_SENTINEL_SCHEMA_VERSION,
            "generation": int(generation_index),
            "rng_indices_before": {k: int(before_idx.get(k, 0)) for k in sorted(before_idx.keys())},
            "rng_indices_after": {k: int(after_idx.get(k, 0)) for k in sorted(after_idx.keys())},
            "rng_indices_delta": idx_delta,
            "population_hash_before": str(before_hash),
            "population_hash_after": str(after_hash),
            "population_hash_equal": bool(before_hash == after_hash),
        }

    def _compute_neutrality(self, agents: List[Any]) -> Tuple[float, float]:
        vals: List[float] = []
        for a in agents:
            g = getattr(a, "genome", None)
            noop = _safe_float(getattr(g, "c1_operator_noop_count", 0), 0.0)
            op_sum = 0.0
            for key in (
                "c1_op_count__bounded_rewire",
                "c1_op_count__duplicate_with_decay",
                "c1_op_count__edge_split_insertion",
                "c1_op_count__fusion_merge_compression",
                "c1_op_count__gate_drift",
                "c1_op_count__motif_duplication",
                "c1_op_count__promoter_expression_mutation",
            ):
                op_sum += _safe_float(getattr(g, key, 0), 0.0)
            den = max(1.0, noop + op_sum)
            vals.append(float(noop / den))

        cur = float(sum(vals) / float(len(vals))) if vals else 0.0
        self._neutrality_history.append(cur)
        if len(self._neutrality_history) > max(2, int(self.neutrality_window)):
            self._neutrality_history = self._neutrality_history[-int(self.neutrality_window) :]

        trend = 0.0
        if len(self._neutrality_history) >= 2:
            trend = float(self._neutrality_history[-1] - self._neutrality_history[0])
        return cur, trend

    def _compute_functional_density(
        self,
        agents: List[Any],
        *,
        max_agents: int,
        max_modules_per_agent: int,
        hash_cache: Optional[Dict[int, str]] = None,
    ) -> Tuple[float, float]:
        sorted_agents = sorted(
            agents,
            key=lambda a: (
                _structure_hash(a, hash_cache=hash_cache),
                int(getattr(a, "agent_id", 0)),
            ),
        )
        chosen = sorted_agents[: max(0, int(max_agents))]

        density_parts: List[float] = []
        all_drops: List[float] = []

        for a in chosen:
            base = _agent_function_score(a)
            mids = _module_ids_from_agent(a)[: max(0, int(max_modules_per_agent))]
            if base <= 0.0 or not mids:
                continue
            drops: List[float] = []
            for mid in mids:
                kscore = _agent_function_score(a, knockout_module_id=mid)
                drop = max(0.0, base - kscore)
                drops.append(drop)
                all_drops.append(drop)
            if drops:
                density_parts.append(float(sum(drops) / (len(drops) * max(base, 1e-9))))

        functional_density = float(sum(density_parts) / len(density_parts)) if density_parts else 0.0

        # Redundancy index: share of near-zero knockout deltas.
        zeroish = 0
        for d in all_drops:
            if d <= 1e-9:
                zeroish += 1
        redundancy = float(zeroish / float(len(all_drops))) if all_drops else 1.0

        return functional_density, redundancy

    def _update_recovery_taxonomy(self, sim: Any) -> Dict[str, Any]:
        rec = getattr(sim, "_a3_recovery_taxonomy_last_gen", None)
        last = ""
        if isinstance(rec, dict):
            last = str(rec.get("classification", "") or "")
            if last in self._taxonomy_counts:
                self._taxonomy_counts[last] = int(self._taxonomy_counts.get(last, 0)) + 1
        total = sum(int(v) for v in self._taxonomy_counts.values())
        return {
            "last": last,
            "counts": {
                "return": int(self._taxonomy_counts.get("return", 0)),
                "shift": int(self._taxonomy_counts.get("shift", 0)),
                "collapse": int(self._taxonomy_counts.get("collapse", 0)),
            },
            "ratios": {
                "return": float(self._taxonomy_counts.get("return", 0) / float(total)) if total > 0 else 0.0,
                "shift": float(self._taxonomy_counts.get("shift", 0) / float(total)) if total > 0 else 0.0,
                "collapse": float(self._taxonomy_counts.get("collapse", 0) / float(total)) if total > 0 else 0.0,
            },
        }

    def compute_post_boundary(self, sim: Any, generation_index: int) -> None:
        sim_cfg = getattr(sim, "sim_cfg", None)
        if sim_cfg is None or not bool(getattr(sim_cfg, "phase_e_e1_enable", False)):
            return

        cadence = max(1, int(getattr(sim_cfg, "phase_e_e1_cadence_gen", 1) or 1))
        if int(generation_index) % cadence != 0:
            return

        rm = getattr(sim, "rng_manager", None)
        scope = nullcontext()
        if rm is not None:
            expected = rm.expected_delta_for_boundary("metrics.phase_e.e1.compute", ctx={})
            scope = rm.budget_scope("metrics.phase_e.e1.compute", expected_delta=expected)

        with scope:
            pop = getattr(sim, "pop", None)
            agents = list(pop.alive_agents()) if pop is not None else []
            agents = sorted(agents, key=lambda a: int(getattr(a, "agent_id", 0)))
            structure_hash_cache: Dict[int, str] = {}

            before_idx = _full_rng_indices(rm, generation_index=int(generation_index)) if rm is not None else {}
            before_hash = _population_structure_hash(sim, generation_index, hash_cache=structure_hash_cache)

            hashes = [_structure_hash(a, hash_cache=structure_hash_cache) for a in agents]
            unique_hashes = sorted({h for h in hashes if h})

            introduced_this_gen = 0
            retained_ge_k = 0
            for h in unique_hashes:
                if h not in self._seen_first_gen:
                    self._seen_first_gen[h] = int(generation_index)
                    introduced_this_gen += 1
                first = int(self._seen_first_gen.get(h, generation_index))
                if int(generation_index) - first >= int(self.retention_k):
                    retained_ge_k += 1

            sir = float(retained_ge_k / float(max(1, introduced_this_gen)))
            neutrality_mean, neutrality_trend = self._compute_neutrality(agents)

            knockout_every = max(1, int(getattr(sim_cfg, "phase_e_e1_knockout_every_gen", 2) or 2))
            functional_density = 0.0
            redundancy_index = 1.0
            if int(generation_index) % knockout_every == 0:
                functional_density, redundancy_index = self._compute_functional_density(
                    agents,
                    max_agents=max(1, int(getattr(sim_cfg, "phase_e_e1_knockout_max_agents", 16) or 16)),
                    max_modules_per_agent=max(1, int(getattr(sim_cfg, "phase_e_e1_knockout_max_modules_per_agent", 4) or 4)),
                    hash_cache=structure_hash_cache,
                )

            taxonomy = self._update_recovery_taxonomy(sim)

            record: Dict[str, Any] = {
                "schema_version": E1_SCHEMA_VERSION,
                "generation": int(generation_index),
                "cadence_gen": int(cadence),
                "retention_k_gen": int(self.retention_k),
                "population_alive": int(len(agents)),
                "unique_structures": int(len(unique_hashes)),
                "introduced_this_gen": int(introduced_this_gen),
                "retained_ge_k": int(retained_ge_k),
                "sir": float(sir),
                "neutrality_mean": float(neutrality_mean),
                "neutrality_trend": float(neutrality_trend),
                "functional_density": float(functional_density),
                "redundancy_index": float(redundancy_index),
                "recovery_taxonomy": taxonomy,
                "knockout_cadence_gen": int(knockout_every),
            }

            sentinel = self._build_sentinel(
                sim,
                generation_index,
                before_idx=before_idx,
                before_hash=before_hash,
                hash_cache=structure_hash_cache,
            )
            record["determinism_sentinel"] = {
                "rng_delta_nonzero_keys": sorted([k for k, v in sentinel.get("rng_indices_delta", {}).items() if int(v) != 0]),
                "population_hash_equal": bool(sentinel.get("population_hash_equal", False)),
            }

            out_metrics_dir = os.path.join(self.run_dir, "metrics")
            _append_canonical_jsonl(os.path.join(out_metrics_dir, "phase_e_e1_metrics.jsonl"), record)
            _append_canonical_jsonl(os.path.join(out_metrics_dir, "phase_e_e1_sentinel.jsonl"), sentinel)
            _atomic_write_json(os.path.join(self.run_dir, "phase_e_e1_latest.json"), record)

            self.last_record = dict(record)
            self.last_sentinel = dict(sentinel)


@dataclass
class PhaseEFailureTaxonomyEngine:
    """Deterministic E2 failure taxonomy engine (cadence-based, checkpointable)."""

    run_dir: str

    _stagnation_streak: int = 0
    _drift_collapse_streak: int = 0
    _category_counts: Dict[str, int] = field(
        default_factory=lambda: {
            "drift_collapse": 0,
            "maintenance_domination": 0,
            "structural_stagnation": 0,
            "bloat_ceiling_hit": 0,
            "mutation_explosion": 0,
            "identity_mimicry": 0,
            "rng_desync": 0,
            "hidden_scalar_reward_channel": 0,
        }
    )
    last_record: Optional[Dict[str, Any]] = None

    def export_state(self) -> Dict[str, Any]:
        return {
            "schema_version": "phase_e_e2_state_v1",
            "stagnation_streak": int(self._stagnation_streak),
            "drift_collapse_streak": int(self._drift_collapse_streak),
            "category_counts": {str(k): int(v) for k, v in sorted(self._category_counts.items(), key=lambda kv: str(kv[0]))},
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._stagnation_streak = _safe_int(state.get("stagnation_streak", 0), 0)
        self._drift_collapse_streak = _safe_int(state.get("drift_collapse_streak", 0), 0)
        cc = state.get("category_counts", {})
        if isinstance(cc, dict):
            for k in self._category_counts.keys():
                self._category_counts[k] = _safe_int(cc.get(k, self._category_counts[k]), self._category_counts[k])

    def _detect_categories(self, sim: Any, generation_index: int, e1_record: Dict[str, Any], e1_sentinel: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        sim_cfg = getattr(sim, "sim_cfg", None)
        details: Dict[str, Any] = {}
        categories: List[str] = []

        pop_alive = _safe_int(e1_record.get("population_alive", 0), 0)
        unique_structures = _safe_int(e1_record.get("unique_structures", 0), 0)
        introduced = _safe_int(e1_record.get("introduced_this_gen", 0), 0)

        # 1) drift_collapse
        rec = e1_record.get("recovery_taxonomy", {}) if isinstance(e1_record.get("recovery_taxonomy", {}), dict) else {}
        rec_last = str(rec.get("last", "") or "")
        collapse_ratio = _safe_float((rec.get("ratios", {}) or {}).get("collapse", 0.0), 0.0)
        if rec_last == "collapse":
            self._drift_collapse_streak += 1
        else:
            self._drift_collapse_streak = 0
        drift_min_streak = max(1, int(getattr(sim_cfg, "phase_e_e2_drift_collapse_min_streak", 2) or 2))
        drift_min_ratio = _safe_float(getattr(sim_cfg, "phase_e_e2_drift_collapse_min_ratio", 0.50), 0.50)
        if self._drift_collapse_streak >= drift_min_streak or collapse_ratio >= drift_min_ratio:
            categories.append("drift_collapse")
            details["drift_collapse"] = {
                "collapse_last": rec_last == "collapse",
                "collapse_ratio": float(collapse_ratio),
                "collapse_streak": int(self._drift_collapse_streak),
                "min_streak": int(drift_min_streak),
                "min_ratio": float(drift_min_ratio),
            }

        # 2) maintenance_domination
        maint = getattr(sim, "_maintenance_domination_annotation_last_gen", None)
        if isinstance(maint, dict) and bool(maint.get("triggered", False)):
            categories.append("maintenance_domination")
            details["maintenance_domination"] = {
                "triggered": True,
                "ratio": _safe_float(maint.get("ratio", 0.0), 0.0),
                "threshold": _safe_float(maint.get("threshold", 0.0), 0.0),
                "requests": _safe_int(maint.get("requests", 0), 0),
                "opportunities": _safe_int(maint.get("opportunities", 0), 0),
            }

        # 3) structural_stagnation
        if introduced <= 0:
            self._stagnation_streak += 1
        else:
            self._stagnation_streak = 0
        stagnation_min = max(1, int(getattr(sim_cfg, "phase_e_e2_structural_stagnation_min_streak", 5) or 5))
        if self._stagnation_streak >= stagnation_min:
            categories.append("structural_stagnation")
            details["structural_stagnation"] = {
                "introduced_this_gen": int(introduced),
                "streak": int(self._stagnation_streak),
                "min_streak": int(stagnation_min),
            }

        # 4) bloat_ceiling_hit
        mod_cap = max(1, int(getattr(sim_cfg, "genome_b2_max_modules_per_agent", 12) or 12))
        pop = getattr(sim, "pop", None)
        agents = sorted(list(pop.alive_agents()) if pop is not None else [], key=lambda a: int(getattr(a, "agent_id", 0)))
        mod_counts: List[int] = []
        for a in agents:
            g = getattr(a, "genome", None)
            d = g.to_dict() if g is not None and hasattr(g, "to_dict") else {}
            mods = d.get("modules", []) if isinstance(d, dict) else []
            mod_counts.append(len(mods) if isinstance(mods, list) else 0)
        avg_mod = float(sum(mod_counts) / float(len(mod_counts))) if mod_counts else 0.0
        max_mod = max(mod_counts) if mod_counts else 0
        bloat_ratio = _safe_float(getattr(sim_cfg, "phase_e_e2_bloat_cap_ratio", 0.95), 0.95)
        if avg_mod >= float(mod_cap) * bloat_ratio or max_mod >= mod_cap:
            categories.append("bloat_ceiling_hit")
            details["bloat_ceiling_hit"] = {
                "avg_modules": float(avg_mod),
                "max_modules": int(max_mod),
                "cap_modules": int(mod_cap),
                "cap_ratio": float(bloat_ratio),
            }

        # 5) mutation_explosion
        explosion_ratio = float(introduced) / float(max(1, pop_alive))
        explosion_min = _safe_float(getattr(sim_cfg, "phase_e_e2_mutation_explosion_ratio", 0.75), 0.75)
        if explosion_ratio >= explosion_min and introduced >= 2:
            categories.append("mutation_explosion")
            details["mutation_explosion"] = {
                "introduced_this_gen": int(introduced),
                "population_alive": int(pop_alive),
                "ratio": float(explosion_ratio),
                "threshold_ratio": float(explosion_min),
            }

        # 6) identity_mimicry
        mimic_ratio = float(unique_structures) / float(max(1, pop_alive))
        mimic_max = _safe_float(getattr(sim_cfg, "phase_e_e2_identity_mimicry_max_ratio", 0.15), 0.15)
        if pop_alive > 0 and mimic_ratio <= mimic_max:
            categories.append("identity_mimicry")
            details["identity_mimicry"] = {
                "unique_structures": int(unique_structures),
                "population_alive": int(pop_alive),
                "ratio": float(mimic_ratio),
                "max_ratio": float(mimic_max),
            }

        # 7) rng_desync
        nonzero_rng_keys = list(e1_sentinel.get("rng_delta_nonzero_keys", [])) if isinstance(e1_sentinel, dict) else []
        if nonzero_rng_keys or not bool(e1_sentinel.get("population_hash_equal", True)):
            categories.append("rng_desync")
            details["rng_desync"] = {
                "rng_delta_nonzero_keys": sorted([str(k) for k in nonzero_rng_keys]),
                "population_hash_equal": bool(e1_sentinel.get("population_hash_equal", True)),
            }

        # 8) hidden_scalar_reward_channel
        if not bool(getattr(sim_cfg, "selection_forbid_chemistry_scalar", True)):
            categories.append("hidden_scalar_reward_channel")
            details["hidden_scalar_reward_channel"] = {
                "selection_forbid_chemistry_scalar": bool(getattr(sim_cfg, "selection_forbid_chemistry_scalar", True)),
                "reason": "chemistry_scalar_guard_disabled",
            }

        categories = sorted(set(categories))
        return categories, details

    def compute_post_boundary(self, sim: Any, generation_index: int) -> None:
        sim_cfg = getattr(sim, "sim_cfg", None)
        if sim_cfg is None or not bool(getattr(sim_cfg, "phase_e_e2_enable", False)):
            return

        cadence = max(1, int(getattr(sim_cfg, "phase_e_e2_cadence_gen", 1) or 1))
        if int(generation_index) % cadence != 0:
            return

        e1 = getattr(sim, "_phase_e_metrics_engine", None)
        e1_record = dict(getattr(e1, "last_record", {}) or {})
        e1_sentinel = dict(getattr(e1, "last_sentinel", {}) or {})
        if not e1_record:
            return

        rm = getattr(sim, "rng_manager", None)
        scope = nullcontext()
        if rm is not None:
            expected = rm.expected_delta_for_boundary("metrics.phase_e.e2.compute", ctx={})
            scope = rm.budget_scope("metrics.phase_e.e2.compute", expected_delta=expected)

        with scope:
            categories, details = self._detect_categories(
                sim,
                generation_index=int(generation_index),
                e1_record=e1_record,
                e1_sentinel=e1_sentinel,
            )
            for c in categories:
                self._category_counts[c] = int(self._category_counts.get(c, 0)) + 1

            record = {
                "schema_version": E2_SCHEMA_VERSION,
                "generation": int(generation_index),
                "cadence_gen": int(cadence),
                "categories": list(categories),
                "category_count": int(len(categories)),
                "category_counts_cumulative": {k: int(v) for k, v in sorted(self._category_counts.items(), key=lambda kv: str(kv[0]))},
                "details": details,
                "linked_e1_generation": int(e1_record.get("generation", generation_index)),
                "linked_e1_schema_version": str(e1_record.get("schema_version", "")),
            }

            out_metrics_dir = os.path.join(self.run_dir, "metrics")
            _append_canonical_jsonl(os.path.join(out_metrics_dir, "phase_e_e2_taxonomy.jsonl"), record)
            for cat in categories:
                ann = {
                    "schema_version": E2_ANNOTATION_SCHEMA_VERSION,
                    "generation": int(generation_index),
                    "category": str(cat),
                    "detail": dict(details.get(cat, {})),
                }
                _append_canonical_jsonl(os.path.join(out_metrics_dir, "phase_e_e2_annotations.jsonl"), ann)
            _atomic_write_json(os.path.join(self.run_dir, "phase_e_e2_latest.json"), record)

            self.last_record = dict(record)
