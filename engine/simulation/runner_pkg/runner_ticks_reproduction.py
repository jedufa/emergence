from __future__ import annotations

"""
simulation/runner_pkg/runner_ticks_reproduction.py

In-world reproduction handling for the tick loop.

Responsibilities:
- Apply reproduction cooldown
- Evaluate reproduction eligibility and mate finding (if required)
- Find / reserve spawn tiles to avoid collisions within a tick
- Apply energy budgeting + conception/gestation failure rules
- Spawn newborns into env + sync policy banks
- Instrumentation: first reproduction timing (analysis-only)

Note:
Logic is moved from the former runner_pkg/steps.py without semantic changes.
"""

from typing import Dict, Set, Tuple
import time
import numpy as np

from agents.chemistry import CHEM_SLOT_CHRONIC_DAMAGE, chemistry_from_values
from agents.chemistry_coupling import reproduction_viability_scale_from_chem_row
from agents.developmental_program_v1 import default_program, module_structure_signature
from agents.agent import assign_child_lineage, format_lineage_id, normalize_lineage_stats
from env.gridworld_pkg.corridor_cp0 import corridor_cp0_agent_home_eligible, corridor_cp0_central_place_eligibility_enabled
from simulation.metrics_pkg.metrics_logger_genome import parent_child_genome_distance_summary

# ============================================================
# PHASE 3 – Birth RNG Trace Probe
# Purpose:
#   Detect RNG divergence inside _handle_births.
#   Logs policy stream index + digest before and after births.
# ============================================================

import os
import hashlib
import json


LRO1_REPRODUCTIVE_OPPORTUNITY_MODE = "local_patch_carrying_capacity_v1"
LRO1_REPRODUCTIVE_OPPORTUNITY_APPLY_STAGE = "pre_attempt_try_prob"
LRO1_REPRODUCTIVE_OPPORTUNITY_DEFAULT_WEIGHTS = {
    "density": -0.35,
    "waste": -0.25,
    "scent": -0.10,
    "shelter_cover": 0.20,
    "food": 0.10,
    "hazard": -0.20,
}


def _lro1_clamp(value: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(value))))


def parse_lro1_reproductive_opportunity_weights(raw: object) -> dict[str, float]:
    if raw in ("", None):
        parsed = {}
    elif isinstance(raw, str):
        parsed = json.loads(raw)
    elif isinstance(raw, dict):
        parsed = dict(raw)
    else:
        raise ValueError("lro1_reproductive_opportunity_weights_json must be a JSON object")
    if not isinstance(parsed, dict):
        raise ValueError("lro1_reproductive_opportunity_weights_json must be a JSON object")
    weights = dict(LRO1_REPRODUCTIVE_OPPORTUNITY_DEFAULT_WEIGHTS)
    for key, value in parsed.items():
        if key not in weights:
            raise ValueError(f"unsupported LRO1 reproductive opportunity weight: {key}")
        weights[str(key)] = float(value)
    return weights


def compute_lro1_reproductive_opportunity(
    context: dict[str, object],
    *,
    enabled: bool,
    mode: str = LRO1_REPRODUCTIVE_OPPORTUNITY_MODE,
    min_multiplier: float = 0.35,
    max_multiplier: float = 1.0,
    weights: dict[str, float] | None = None,
    report_fields: bool = True,
    base_try_prob: float | None = None,
) -> dict[str, object]:
    mode = str(mode or LRO1_REPRODUCTIVE_OPPORTUNITY_MODE)
    min_multiplier = float(min_multiplier)
    max_multiplier = float(max_multiplier)
    if min_multiplier < 0.0 or max_multiplier < min_multiplier:
        raise ValueError("lro1 reproductive opportunity multiplier bounds are invalid")
    if mode != LRO1_REPRODUCTIVE_OPPORTUNITY_MODE:
        raise ValueError(f"unsupported lro1_reproductive_opportunity_mode: {mode}")

    disabled = {
        "lro1_reproductive_opportunity_enable": int(bool(enabled)),
        "lro1_reproductive_opportunity_mode": mode if bool(enabled) else "",
        "lro1_opportunity_score_parent_site": 0.0,
        "lro1_opportunity_multiplier": 1.0,
        "lro1_opportunity_context_density": 0.0,
        "lro1_opportunity_context_waste": 0.0,
        "lro1_opportunity_context_scent": 0.0,
        "lro1_opportunity_context_shelter_cover": 0.0,
        "lro1_opportunity_context_food": 0.0,
        "lro1_opportunity_context_hazard": 0.0,
        "lro1_opportunity_invalid_context": 0,
        "lro1_opportunity_neutral_context": 1,
        "lro1_opportunity_effective_try_prob": "" if base_try_prob is None else float(base_try_prob),
    }
    if not enabled:
        return disabled if report_fields else {}

    def _finite_float(name: str) -> tuple[float, bool]:
        try:
            value = float(context.get(name, 0.0))
        except Exception:
            return 0.0, False
        if not np.isfinite(value):
            return 0.0, False
        return value, True

    density, density_ok = _finite_float("density")
    waste, waste_ok = _finite_float("waste")
    scent, scent_ok = _finite_float("scent")
    shelter_cover, shelter_ok = _finite_float("shelter_cover")
    food, food_ok = _finite_float("food")
    hazard, hazard_ok = _finite_float("hazard")
    invalid_context = int(not all((density_ok, waste_ok, scent_ok, shelter_ok, food_ok, hazard_ok)))
    neutral_context = int(
        density == 0.0
        and waste == 0.0
        and scent == 0.0
        and shelter_cover == 0.0
        and food == 0.0
        and hazard == 0.0
    )
    normalized = {
        "density": _lro1_clamp(density / 9.0, 0.0, 1.0),
        "waste": _lro1_clamp(waste, 0.0, 1.0),
        "scent": _lro1_clamp(scent, 0.0, 1.0),
        "shelter_cover": _lro1_clamp(shelter_cover, 0.0, 1.0),
        "food": _lro1_clamp(food, 0.0, 1.0),
        "hazard": _lro1_clamp(hazard, 0.0, 1.0),
    }
    use_weights = weights if weights is not None else LRO1_REPRODUCTIVE_OPPORTUNITY_DEFAULT_WEIGHTS
    score = float(sum(float(use_weights.get(key, 0.0)) * normalized[key] for key in normalized))
    multiplier = _lro1_clamp(1.0 + score, min_multiplier, max_multiplier)
    effective_try_prob: object = ""
    if base_try_prob is not None:
        effective_try_prob = _lro1_clamp(float(base_try_prob) * multiplier, 0.0, 1.0)
    return {
        "lro1_reproductive_opportunity_enable": 1,
        "lro1_reproductive_opportunity_mode": mode,
        "lro1_opportunity_score_parent_site": score,
        "lro1_opportunity_multiplier": multiplier,
        "lro1_opportunity_context_density": density,
        "lro1_opportunity_context_waste": waste,
        "lro1_opportunity_context_scent": scent,
        "lro1_opportunity_context_shelter_cover": shelter_cover,
        "lro1_opportunity_context_food": food,
        "lro1_opportunity_context_hazard": hazard,
        "lro1_opportunity_invalid_context": invalid_context,
        "lro1_opportunity_neutral_context": neutral_context,
        "lro1_opportunity_effective_try_prob": effective_try_prob,
    }



class TickReproductionMixin:
    def _handle_births(self, alive, successful_births: int, failed_births: int, buckets: Dict[str, float]):
        """
        Determinism probe (Phase 3): birth-internal trace

        Enabled with:
          - EIEL_TRACE_BIRTH_INTERNAL=1

        Emits JSON lines describing ordering and RNG digests without advancing RNG.
        Instrumentation only (no semantic changes).
        """

        if bool(getattr(getattr(self, "sim_cfg", None), "recomb72_audit_enable", False)):
            if not bool(
                getattr(
                    self.sim_cfg,
                    "recomb72_disable_ordinary_tick_births",
                    False,
                )
            ):
                raise RuntimeError(
                    "RECOMB72_AUDIT_CONTRACT: ordinary tick births must be disabled"
                )
            # This return intentionally precedes every trace, eligibility, mate,
            # placement, and RNG surface. Audit reproduction occurs only at the
            # generation boundary.
            return successful_births, failed_births

        trace_internal = os.environ.get("EIEL_TRACE_BIRTH_INTERNAL", "0") == "1"
        trace_callsite = os.environ.get("EIEL_TRACE_BIRTH_CALLSITE", "0") == "1"
        trace_rng = os.environ.get("EIEL_TRACE_BIRTH_RNG", "0") == "1"

        # Always define a safe JSONL emitter so tracing can never crash the simulation.
        # This is intentionally a no-op unless tracing is enabled below.
        def _emit_jsonl(filename: str, prefix_tag: str, payload: dict) -> None:
            return

        # Default: only run the heavy tracing machinery when explicitly enabled.
        trace_on = bool(trace_internal)
        trace_lines = 0
        trace_max = int(os.environ.get("EIEL_TRACE_BIRTH_INTERNAL_MAX", "200") or "200")
        trace_tick_min = int(os.environ.get("EIEL_TRACE_BIRTH_TICK_MIN", "-1") or "-1")
        trace_tick_max = int(os.environ.get("EIEL_TRACE_BIRTH_TICK_MAX", "-1") or "-1")

        # Convenience: if callsite/RNG tracing is enabled but internal isn't, auto-enable
        # internal trace with defaults aimed at the historic divergence window.
        if (not trace_on) and (trace_callsite or trace_rng):
            trace_on = True
            trace_lines = 0
            trace_max = int(os.environ.get("EIEL_TRACE_BIRTH_LINES", "400") or "400")
            trace_tick_min = int(os.environ.get("EIEL_TRACE_BIRTH_TICK_MIN", "54") or "54")
            trace_tick_max = int(os.environ.get("EIEL_TRACE_BIRTH_TICK_MAX", "61") or "61")

        # Determine current tick/gen index (best-effort) for windowed tracing.
        # Prefer the loop's explicit tick if present.
        tick = int(getattr(self, "_tick", getattr(self, "_pred_tick", 0)))
        try:
            gen = int(getattr(self, "gen"))
        except Exception:
            try:
                gen = int(getattr(getattr(self, "pop", None), "_eiel_current_generation_index", -1))
            except Exception:
                gen = -1
        try:
            setattr(getattr(self, "pop", None), "_eiel_current_generation_index", int(gen))
        except Exception:
            pass
        try:
            setattr(self.sim_cfg, "_mutation_weather_generation", int(gen))
        except Exception:
            pass

        # Apply optional tick window.
        if trace_on and trace_tick_min >= 0 and tick < trace_tick_min:
            trace_on = False
        if trace_on and trace_tick_max >= 0 and tick > trace_tick_max:
            trace_on = False

        runner_np_sha_pre = ""
        env_rng_sha_pre = ""

        if trace_on:

            def _emit_jsonl(filename: str, prefix_tag: str, payload: dict) -> None:
                try:
                    run_dir = getattr(self, "run_dir", None)
                    if not run_dir:
                        return
                    dbg_dir = os.path.join(run_dir, "debug")
                    os.makedirs(dbg_dir, exist_ok=True)
                    path = os.path.join(dbg_dir, filename)
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(prefix_tag + " " + json.dumps(payload, sort_keys=True) + "\n")
                except Exception:
                    # tracing must never break the simulation
                    return

            def _sha256_json(obj: object) -> str:
                try:
                    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    return hashlib.sha256(blob).hexdigest()
                except Exception:
                    return "err"

            def _rng_digest(rng_obj: object) -> str:
                try:
                    st = getattr(getattr(rng_obj, "bit_generator", None), "state", None)
                    if st is None:
                        return "none"
                    return _sha256_json(st)
                except Exception:
                    return "err"

            # IMPORTANT (perf): these digests are expensive, so they are computed ONLY when tracing is ON.
            runner_np_sha_pre = _rng_digest(getattr(self, "np_rng", None))
            env_rng_sha_pre = _rng_digest(getattr(getattr(self, "env", None), "rng", None))

        parent_order = []
        if trace_on:
            try:
                parent_order = [int(a.agent_id) for a in alive if getattr(a, "alive", False)]
            except Exception:
                parent_order = []
            payload = {
                "stage": "birth_internal_entry",
                "gen": gen,
                "tick": tick,
                "alive_n_in": int(len(list(alive))),
                "parent_order": parent_order,
                "runner_np_rng_sha": runner_np_sha_pre,
                "env_rng_sha": env_rng_sha_pre,
            }
            _emit_jsonl("birth_internal_trace.jsonl", "[BIRTH_INTERNAL_TRACE]", payload)
            trace_lines += 1

        # ------------------------------------------------------------
        # RNG TRACE – PRE BIRTH
        # ------------------------------------------------------------
        if trace_on and trace_rng:
            try:
                rng = getattr(self, "rng_manager", None)
                if rng is not None:
                    snap = rng.debug_snapshot()  # must already exist
                    payload = {
                        "stage": "pre_birth",
                        "generation": gen,
                        "tick": tick,
                        "policy_index": snap.get("policy_index"),
                        "policy_digest": snap.get("policy_digest"),
                    }
                    _emit_jsonl("birth_rng_trace.jsonl", "[BIRTH_RNG_TRACE]", payload)
            except Exception as _e:
                _emit_jsonl(
                    "birth_rng_trace_errors.jsonl",
                    "[BIRTH_RNG_TRACE_ERROR_PRE]",
                    {
                        "stage": "rng_trace_error_pre",
                        "gen": gen,
                        "tick": tick,
                        "error": str(_e),
                    },
                )





        # Phase 0 perf: total reproduction time (values are seconds).
        t_repro0 = time.perf_counter()
        _t_birth_scan0 = time.perf_counter()
        _t_make_child_sum = 0.0
        births = []
        mated_this_tick: Set[int] = set()
        reserved_tiles: Set[Tuple[int, int]] = set()

        injury_gate = None
        if getattr(self.env_cfg, "enable_injury", False):
            ig = getattr(self.env_cfg, "injury_repro_block", None)
            if ig is not None:
                injury_gate = float(ig)

        scar_gate = None
        if getattr(self.env_cfg, "enable_scar", False):
            sg = getattr(self.env_cfg, "scar_repro_block", None)
            if sg is not None:
                scar_gate = float(sg)

        k_inj = float(getattr(self.env_cfg, "injury_repro_k_injury", 0.0))
        k_scar = float(getattr(self.env_cfg, "injury_repro_k_scar", 0.0))
        p_fail_scar = float(getattr(self.env_cfg, "injury_birth_fail_scar", 0.0))
        p_fail_inj = float(getattr(self.env_cfg, "injury_birth_fail_injury", 0.0))
        chem_repro_enable = bool(getattr(self.sim_cfg, "chemistry_functional_coupling_enable", False)) and bool(
            getattr(self.sim_cfg, "chemistry_repro_viability_enable", False)
        )
        chem_floor = float(getattr(self.sim_cfg, "chemistry_repro_viability_floor", 0.2))
        chem_strength = float(getattr(self.sim_cfg, "chemistry_repro_viability_strength", 0.8))
        chem_chronic_block = float(getattr(self.sim_cfg, "chemistry_repro_chronic_block_threshold", 0.95))

        share = float(self.env_cfg.reproduce_parent_energy_share)
        buffer = float(self.env_cfg.reproduce_min_energy_buffer)
        cd = int(self.env_cfg.reproduce_cooldown_steps)

        try_prob = float(getattr(self.env_cfg, "reproduce_try_prob", 1.0))
        try_prob = max(0.0, min(1.0, try_prob))

        runnercore25_birth_row_spec_bridge = bool(
            getattr(self, "_runnercore25_full_birth_row_spec_bridge_active", False)
            and getattr(self.sim_cfg, "runnercore25_full_birth_resident_newborn_attach_enable", False)
            and hasattr(getattr(self, "pop", None), "make_child_row_spec")
        )
        runnercore25_successful_ids = []

        def _runnercore25_stat_add(key: str, value: int | float = 1) -> None:
            stats = getattr(self.sim_cfg, "_runnercore6_resident_child_row_stats", None)
            if not isinstance(stats, dict):
                return
            old = stats.get(str(key), 0)
            if isinstance(value, float):
                stats[str(key)] = float(old or 0.0) + float(value)
            else:
                stats[str(key)] = int(old or 0) + int(value)

        def _runnercore25_make_child(p1, p2, child_energy: float, buckets: Dict[str, float]):
            if not runnercore25_birth_row_spec_bridge:
                return self.pop.make_child(p1, p2, child_energy, buckets=buckets)
            child_spec = self.pop.make_child_row_spec(p1, p2, child_energy, buckets=buckets)
            if child_spec is None:
                return None
            self.pop.agents[int(child_spec.agent_id)] = child_spec
            _runnercore25_stat_add("runnercore25_full_birth_row_specs_created", 1)
            _runnercore25_stat_add("runnercore25_pre_attach_agent_materializations_avoided", 1)
            return child_spec

        runnercore31_static_dev_threshold = bool(
            getattr(self.sim_cfg, "runnercore31_full_birth_static_dev_threshold_enable", False)
        )
        _runnercore31_default_program = default_program() if runnercore31_static_dev_threshold else {}
        _runnercore31_empty_module_sig = module_structure_signature([]) if runnercore31_static_dev_threshold else ""

        def _runnercore31_stat_add(key: str, value: int = 1) -> None:
            stats = getattr(self.sim_cfg, "_runnercore6_resident_child_row_stats", None)
            if not isinstance(stats, dict):
                stats = {}
                try:
                    setattr(self.sim_cfg, "_runnercore6_resident_child_row_stats", stats)
                except Exception:
                    return
            stats[str(key)] = int(stats.get(str(key), 0) or 0) + int(value)

        def _runnercore31_diag_active() -> bool:
            diag = getattr(self, "env_hotpath_expression_diagnostics", None)
            if diag is None:
                return False
            return bool(
                getattr(diag, "developmental_cache_equality_enabled", False)
                or getattr(diag, "developmental_effective_projection_enabled", False)
                or getattr(diag, "developmental_residual_enabled", False)
            )

        def _runnercore31_effective_scalars_from_expr(agent: object, expr: dict) -> tuple[float, ...] | None:
            fn = getattr(self, "_dev_effective_scalars_from_expression", None)
            if callable(fn):
                try:
                    return fn(agent, expr)
                except Exception:
                    return None
            return None

        def _runnercore31_static_reproduction_threshold(agent: object) -> float | None:
            if not runnercore31_static_dev_threshold or _runnercore31_diag_active():
                return None
            key_fn = getattr(self, "_dev_expression_cache_key_for_agent", None)
            cache = getattr(self, "_dev_expr_cache", None)
            if not callable(key_fn) or not isinstance(cache, dict):
                return None
            try:
                key = key_fn(agent)
            except Exception:
                return None
            cached = cache.get(key)
            genome = getattr(agent, "genome", None)
            if isinstance(cached, dict):
                if bool(getattr(self, "_dev_effective_scalar_cache_enabled", lambda: False)()):
                    scalar_cache = getattr(self, "_dev_effective_scalar_cache", None)
                    if not isinstance(scalar_cache, dict):
                        scalar_cache = {}
                        self._dev_effective_scalar_cache = scalar_cache
                    if key not in scalar_cache:
                        scalars = _runnercore31_effective_scalars_from_expr(agent, cached)
                        if scalars is not None:
                            scalar_cache[key] = scalars
                eff = cached.get("effective", {}) if isinstance(cached.get("effective", {}), dict) else {}
                _runnercore31_stat_add("runnercore31_full_birth_static_dev_threshold_cache_hits", 1)
                return float(eff.get("reproduction_threshold", getattr(genome, "reproduction_threshold", 50.0)))

            if genome is None:
                return None
            modules = list(getattr(genome, "modules", []) or [])
            if modules:
                _runnercore31_stat_add("runnercore31_full_birth_static_dev_threshold_module_fallbacks", 1)
                return None
            if getattr(genome, "developmental_program", None) != _runnercore31_default_program:
                _runnercore31_stat_add("runnercore31_full_birth_static_dev_threshold_program_fallbacks", 1)
                return None
            st = getattr(agent, "stats", None)
            if st is not None:
                baseline_sig = str(getattr(st, "_dev_module_structure_signature", "") or "")
                if baseline_sig and baseline_sig != _runnercore31_empty_module_sig:
                    _runnercore31_stat_add("runnercore31_full_birth_static_dev_threshold_signature_fallbacks", 1)
                    return None

            try:
                aid, gidx, tidx, age = key
                eff = {
                    "aggression_bias": float(getattr(genome, "aggression_bias", 0.0)),
                    "action_temperature": float(getattr(genome, "action_temperature", 1.0)),
                    "drive_logit_gain": float(getattr(genome, "drive_logit_gain", 1.0)),
                    "drive_inertia": float(getattr(genome, "drive_inertia", 0.85)),
                    "drive_noise": float(getattr(genome, "drive_noise", 0.02)),
                    "pred_lr": float(getattr(genome, "pred_lr", 0.08)),
                    "reproduction_threshold": float(getattr(genome, "reproduction_threshold", 50.0)),
                }
                expr = {
                    "stage_id": "default",
                    "active_module_count": 0,
                    "module_count": 0,
                    "active_by_module_id": {},
                    "inactive_reason_by_module_id": {},
                    "module_structure_signature": str(_runnercore31_empty_module_sig),
                    "effective": eff,
                    "structural_change_ignored": False,
                    "structural_change_reason": "",
                    "lock_state_by_module_id": {},
                    "lock_events": [],
                    "lock_event_count": 0,
                }
                lock_state_json = json.dumps({}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                lock_state_sha = hashlib.sha256(lock_state_json.encode("utf-8")).hexdigest()
                if st is not None:
                    if not str(getattr(st, "_dev_module_structure_signature", "") or ""):
                        setattr(st, "_dev_module_structure_signature", str(_runnercore31_empty_module_sig))
                    prev_stage = str(getattr(st, "_dev_stage_id", "") or "")
                    prev_active = self._dev_previous_active_module_count(st)
                    if (str(expr["stage_id"]) != prev_stage) or (int(expr["active_module_count"]) != prev_active):
                        queue_trace = getattr(self, "_dev_queue_trace", None)
                        reason_counts = getattr(self, "_dev_reason_counts", None)
                        if callable(queue_trace):
                            queue_trace(
                                {
                                    "schema_version": "developmental_stage_trace_v1",
                                    "generation": int(gidx),
                                    "tick": int(tidx),
                                    "agent_id": int(aid),
                                    "age_steps": int(age),
                                    "stage_id": "default",
                                    "active_module_count": 0,
                                    "module_count": 0,
                                    "locked_module_count": 0,
                                    "lock_state_sha256": str(lock_state_sha),
                                    "structural_change_ignored": False,
                                    "structural_change_reason": "",
                                    "inactive_reason_counts": reason_counts(expr) if callable(reason_counts) else {},
                                }
                            )
                        self._dev_stage_transition_count_gen = int(
                            getattr(self, "_dev_stage_transition_count_gen", 0) or 0
                        ) + 1
                        counts = getattr(self, "_dev_stage_eval_stage_counts_gen", None)
                        if not isinstance(counts, dict):
                            counts = {}
                        counts["default"] = int(counts.get("default", 0) or 0) + 1
                        self._dev_stage_eval_stage_counts_gen = counts
                    setattr(st, "_dev_stage_id", "default")
                    setattr(st, "_dev_active_module_count", 0)
                    setattr(st, "_dev_last_effective", dict(eff))
                    setattr(st, "_dev_d2_lock_state_by_module_id", {})
                    setattr(st, "_dev_d2_lock_state_sha256", str(lock_state_sha))
                    setattr(
                        st,
                        "_dev_last_expr_snapshot",
                        {
                            "stage_id": "default",
                            "active_module_count": 0,
                            "module_count": 0,
                            "active_by_module_id": {},
                            "inactive_reason_by_module_id": {},
                            "effective": dict(eff),
                            "lock_state_by_module_id": {},
                        },
                    )
                    setattr(st, "_dev_last_trace_gen", int(gidx))
                    setattr(st, "_dev_last_trace_tick", int(tidx))
                cache[key] = expr
                stage_store = getattr(self, "_dev_stage_cache_store", None)
                if callable(stage_store):
                    stage_store(agent, genome, expr, age_steps=int(age))
                if bool(getattr(self, "_dev_effective_scalar_cache_enabled", lambda: False)()):
                    scalars = _runnercore31_effective_scalars_from_expr(agent, expr)
                    if scalars is not None:
                        scalar_cache = getattr(self, "_dev_effective_scalar_cache", None)
                        if not isinstance(scalar_cache, dict):
                            scalar_cache = {}
                            self._dev_effective_scalar_cache = scalar_cache
                        scalar_cache[key] = scalars
                _runnercore31_stat_add("runnercore31_full_birth_static_dev_threshold_synthetic_lookups", 1)
                return float(eff["reproduction_threshold"])
            except Exception:
                _runnercore31_stat_add("runnercore31_full_birth_static_dev_threshold_exception_fallbacks", 1)
                return None

        def _runnercore31_reproduction_threshold_or_none(agent: object) -> float | None:
            threshold = _runnercore31_static_reproduction_threshold(agent)
            if threshold is not None:
                _runnercore31_stat_add("runnercore31_full_birth_static_dev_threshold_hits", 1)
                return float(threshold)
            return None


        self._birth_pre_no_energy_gen = int(getattr(self, "_birth_pre_no_energy_gen", 0))
        self._birth_pre_no_space_gen = int(getattr(self, "_birth_pre_no_space_gen", 0))

        self._birth_attempt_conception_fail_gen = int(getattr(self, "_birth_attempt_conception_fail_gen", 0))
        self._birth_make_child_fail_gen = int(getattr(self, "_birth_make_child_fail_gen", 0))

        self._injury_blocked_births_gen = int(getattr(self, "_injury_blocked_births_gen", 0))
        self._birth_pre_injury_gate_gen = int(getattr(self, "_birth_pre_injury_gate_gen", 0))
        self._birth_fail_injury_gate_gen = int(getattr(self, "_birth_fail_injury_gate_gen", 0))
        self._birth_fail_scar_gate_gen = int(getattr(self, "_birth_fail_scar_gate_gen", 0))
        self._birth_fail_conception_gen = int(getattr(self, "_birth_fail_conception_gen", 0))
        self._birth_precheck_no_space_gen = int(getattr(self, "_birth_precheck_no_space_gen", 0))
        self._birth_precheck_no_mate_gen = int(getattr(self, "_birth_precheck_no_mate_gen", 0))
        self._birth_precheck_scar_gate_gen = int(getattr(self, "_birth_precheck_scar_gate_gen", 0))
        self._birth_precheck_injury_gate_gen = int(getattr(self, "_birth_precheck_injury_gate_gen", 0))

        # Array-state accessors (replaces removed env.agent_pos)
        env_alive = getattr(self.env, "agent_alive", None)
        pos_y = getattr(self.env, "agent_pos_y", None)
        pos_x = getattr(self.env, "agent_pos_x", None)
        if env_alive is None or pos_y is None or pos_x is None:
            raise AttributeError("env missing agent_alive/agent_pos_y/agent_pos_x; reproduction requires array-based env state")
        cap = int(env_alive.shape[0])
        agent_grid = getattr(self.env, "agent_grid", None)
        runnercore32_inline_grid_scan = bool(
            getattr(self.sim_cfg, "runnercore32_full_birth_inline_grid_scan_enable", False)
        )
        grid_h = int(getattr(self.env_cfg, "height", 0) or 0)
        grid_w = int(getattr(self.env_cfg, "width", 0) or 0)
        if runnercore32_inline_grid_scan and agent_grid is not None:
            try:
                grid_h = int(np.asarray(agent_grid).shape[0])
                grid_w = int(np.asarray(agent_grid).shape[1])
            except Exception:
                runnercore32_inline_grid_scan = False
        checkpoint_overrides = getattr(self, "_cxxbackbone10_checkpoint_tensor_overrides", None)
        checkpoint_event_tape_capture_on = bool(
            isinstance(checkpoint_overrides, dict)
            and all(
                hasattr(checkpoint_overrides.get(name), "detach")
                for name in (
                    "event_tape_type",
                    "event_tape_actor_id",
                    "event_tape_target_id",
                    "event_tape_value",
                )
            )
        )
        funnel_on = bool(getattr(getattr(self, "sim_cfg", None), "lineage_reproduction_funnel_reporting_enable", False))
        funnel_events_on = bool(
            funnel_on and getattr(getattr(self, "sim_cfg", None), "lineage_reproduction_funnel_write_events", False)
        ) or checkpoint_event_tape_capture_on
        funnel_event_schema_version = int(
            getattr(getattr(self, "sim_cfg", None), "lineage_reproduction_funnel_event_schema_version", 3) or 3
        )
        funnel_scan_action_events_on = bool(
            funnel_events_on
            and funnel_event_schema_version >= 4
            and getattr(getattr(self, "sim_cfg", None), "lineage_reproduction_funnel_write_scan_action_events", False)
        )
        path2_roz_reporting_on = bool(
            funnel_scan_action_events_on and getattr(getattr(self, "sim_cfg", None), "path2_roz_reporting_enable", False)
        )
        path2_roz_schema_version = int(getattr(getattr(self, "sim_cfg", None), "path2_roz_schema_version", 1) or 1)
        path2_roz_zone_mode = str(
            getattr(getattr(self, "sim_cfg", None), "path2_roz_zone_mode", "static_center_manhattan")
            or "static_center_manhattan"
        )
        path2_roz_window_period_steps = max(
            1,
            int(getattr(getattr(self, "sim_cfg", None), "path2_roz_window_period_steps", 40) or 40),
        )
        path2_roz_window_duration_steps = max(
            0,
            int(getattr(getattr(self, "sim_cfg", None), "path2_roz_window_duration_steps", 8) or 8),
        )
        path2_roz_zone_radius = max(
            0,
            int(getattr(getattr(self, "sim_cfg", None), "path2_roz_zone_radius", 0) or 0),
        )
        path2_roz_schedule_mode = str(
            getattr(getattr(self, "sim_cfg", None), "path2_roz_schedule_mode", "static_periodic_compat")
            or "static_periodic_compat"
        )
        path2_roz_schedule_id = str(
            getattr(getattr(self, "sim_cfg", None), "path2_roz_schedule_id", "") or path2_roz_schedule_mode
        )
        path2_roz_schedule_phase_offset_steps = max(
            0,
            int(getattr(getattr(self, "sim_cfg", None), "path2_roz_schedule_phase_offset_steps", 0) or 0),
        )
        path2_roz_schedule_zones_json = str(
            getattr(getattr(self, "sim_cfg", None), "path2_roz_schedule_zones_json", "") or ""
        ).strip()
        lro1_reproductive_opportunity_enabled = bool(
            getattr(getattr(self, "sim_cfg", None), "lro1_reproductive_opportunity_enable", False)
        )
        lro1_reproductive_opportunity_report = bool(
            lro1_reproductive_opportunity_enabled
            or getattr(getattr(self, "sim_cfg", None), "lro1_reproductive_opportunity_report_fields_enable", True)
        )
        lro1_reproductive_opportunity_mode = str(
            getattr(
                getattr(self, "sim_cfg", None),
                "lro1_reproductive_opportunity_mode",
                LRO1_REPRODUCTIVE_OPPORTUNITY_MODE,
            )
            or LRO1_REPRODUCTIVE_OPPORTUNITY_MODE
        )
        lro1_reproductive_opportunity_apply_stage = str(
            getattr(
                getattr(self, "sim_cfg", None),
                "lro1_reproductive_opportunity_apply_stage",
                LRO1_REPRODUCTIVE_OPPORTUNITY_APPLY_STAGE,
            )
            or LRO1_REPRODUCTIVE_OPPORTUNITY_APPLY_STAGE
        )
        if lro1_reproductive_opportunity_enabled:
            if lro1_reproductive_opportunity_apply_stage != LRO1_REPRODUCTIVE_OPPORTUNITY_APPLY_STAGE:
                raise ValueError(
                    "unsupported lro1_reproductive_opportunity_apply_stage: "
                    f"{lro1_reproductive_opportunity_apply_stage}"
                )
            if lro1_reproductive_opportunity_mode != LRO1_REPRODUCTIVE_OPPORTUNITY_MODE:
                raise ValueError(f"unsupported lro1_reproductive_opportunity_mode: {lro1_reproductive_opportunity_mode}")
        lro1_reproductive_opportunity_min_multiplier = float(
            getattr(getattr(self, "sim_cfg", None), "lro1_reproductive_opportunity_min_multiplier", 0.35) or 0.35
        )
        lro1_reproductive_opportunity_max_multiplier = float(
            getattr(getattr(self, "sim_cfg", None), "lro1_reproductive_opportunity_max_multiplier", 1.0) or 1.0
        )
        lro1_reproductive_opportunity_weights = (
            parse_lro1_reproductive_opportunity_weights(
                getattr(getattr(self, "sim_cfg", None), "lro1_reproductive_opportunity_weights_json", "")
            )
            if lro1_reproductive_opportunity_enabled
            else dict(LRO1_REPRODUCTIVE_OPPORTUNITY_DEFAULT_WEIGHTS)
        )
        lro4_offspring_dispersal_enabled = bool(
            getattr(getattr(self, "sim_cfg", None), "lro4_offspring_dispersal_enable", False)
        )
        lro4_offspring_dispersal_report = bool(
            lro4_offspring_dispersal_enabled
            or getattr(getattr(self, "sim_cfg", None), "lro4_offspring_dispersal_report_fields_enable", True)
        )
        lro4_offspring_dispersal_mode = str(
            getattr(
                getattr(self, "sim_cfg", None),
                "lro4_offspring_dispersal_mode",
                "crowding_conditioned_radius_v1",
            )
            or "crowding_conditioned_radius_v1"
        )
        if lro4_offspring_dispersal_enabled and lro4_offspring_dispersal_mode != "crowding_conditioned_radius_v1":
            raise ValueError(f"unsupported lro4_offspring_dispersal_mode: {lro4_offspring_dispersal_mode}")
        lro4_offspring_dispersal_base_radius = max(
            1,
            int(getattr(getattr(self, "sim_cfg", None), "lro4_offspring_dispersal_base_radius", 1) or 1),
        )
        lro4_offspring_dispersal_max_radius = max(
            lro4_offspring_dispersal_base_radius,
            int(getattr(getattr(self, "sim_cfg", None), "lro4_offspring_dispersal_max_radius", 3) or 3),
        )
        if lro4_offspring_dispersal_enabled and lro4_offspring_dispersal_max_radius > 8:
            raise ValueError("lro4_offspring_dispersal_max_radius must be <= 8")
        lro4_offspring_dispersal_density_threshold = float(
            getattr(getattr(self, "sim_cfg", None), "lro4_offspring_dispersal_density_threshold", 4.0) or 4.0
        )
        lro4_offspring_dispersal_waste_threshold = float(
            getattr(getattr(self, "sim_cfg", None), "lro4_offspring_dispersal_waste_threshold", 0.25) or 0.25
        )
        lro4_offspring_dispersal_no_space_trigger = bool(
            getattr(getattr(self, "sim_cfg", None), "lro4_offspring_dispersal_no_space_trigger_enable", True)
        )
        if lro4_offspring_dispersal_enabled and not bool(
            getattr(getattr(self, "sim_cfg", None), "lro4_offspring_dispersal_deterministic_ring_order", True)
        ):
            raise ValueError("lro4_offspring_dispersal requires deterministic_ring_order=true")

        def _path2_roz_default_zone() -> dict:
            return {
                "id": "static_center",
                "center_y": max(0, int(grid_h) // 2),
                "center_x": max(0, int(grid_w) // 2),
                "radius": int(path2_roz_zone_radius),
            }

        def _path2_roz_schedule_zones() -> list[dict]:
            if path2_roz_schedule_mode == "patch_derived_static":
                cached = getattr(self, "_path2_roz_patch_derived_static_zone", None)
                if isinstance(cached, dict):
                    return [dict(cached)]
                if not bool(getattr(self.env, "patch_world_enable", False)):
                    raise ValueError("path2_roz patch_derived_static requires patch_world_enable=true")
                arrays = {
                    "food": getattr(self.env, "patch_food_amount", None),
                    "cover": getattr(self.env, "patch_cover", None),
                    "shelter": getattr(self.env, "patch_shelter", None),
                    "waste": getattr(self.env, "patch_waste", None),
                    "movement_cost": getattr(self.env, "patch_movement_cost", None),
                    "recovery": getattr(self.env, "patch_recovery_activity", None),
                }
                if arrays["food"] is None:
                    raise ValueError("path2_roz patch_derived_static requires Patch World food arrays")
                best: tuple[float, int, int] | None = None
                for yy in range(int(grid_h)):
                    for xx in range(int(grid_w)):
                        food = float(arrays["food"][yy, xx]) if arrays["food"] is not None else 0.0
                        cover = float(arrays["cover"][yy, xx]) if arrays["cover"] is not None else 0.0
                        shelter = float(arrays["shelter"][yy, xx]) if arrays["shelter"] is not None else 0.0
                        waste = float(arrays["waste"][yy, xx]) if arrays["waste"] is not None else 0.0
                        movement_cost = float(arrays["movement_cost"][yy, xx]) if arrays["movement_cost"] is not None else 1.0
                        recovery = float(arrays["recovery"][yy, xx]) if arrays["recovery"] is not None else 0.0
                        score = (
                            food
                            + max(cover, shelter)
                            + recovery
                            - waste
                            - max(0.0, movement_cost - 1.0) * 0.10
                        )
                        candidate = (float(score), -int(yy), -int(xx))
                        if best is None or candidate > best:
                            best = candidate
                if best is None:
                    raise ValueError("path2_roz patch_derived_static found no Patch World cells")
                _, neg_y, neg_x = best
                cy = -int(neg_y)
                cx = -int(neg_x)
                zone = {
                    "id": f"patch_static_y{cy}_x{cx}",
                    "center_y": cy,
                    "center_x": cx,
                    "radius": int(path2_roz_zone_radius),
                }
                setattr(self, "_path2_roz_patch_derived_static_zone", dict(zone))
                return [zone]
            if not path2_roz_schedule_zones_json or path2_roz_schedule_mode == "static_periodic_compat":
                return [_path2_roz_default_zone()]
            parsed = json.loads(path2_roz_schedule_zones_json)
            if not isinstance(parsed, list) or not parsed:
                raise ValueError("path2_roz_schedule_zones_json must be a non-empty JSON list")
            if path2_roz_schedule_mode == "periodic_single_zone" and len(parsed) != 1:
                raise ValueError("periodic_single_zone requires exactly one schedule zone")
            zones = []
            for idx, raw in enumerate(parsed):
                if not isinstance(raw, dict):
                    raise ValueError("path2_roz_schedule_zones_json entries must be objects")
                cy = int(raw.get("center_y", -1))
                cx = int(raw.get("center_x", -1))
                radius = int(raw.get("radius", -1))
                if cy < 0 or cx < 0 or radius < 0 or cy >= int(grid_h) or cx >= int(grid_w):
                    raise ValueError("path2_roz_schedule_zones_json zone coordinates must be in bounds and radius non-negative")
                zones.append(
                    {
                        "id": str(raw.get("id") or f"zone_{idx:02d}"),
                        "center_y": cy,
                        "center_x": cx,
                        "radius": radius,
                    }
                )
            return zones

        if path2_roz_reporting_on and path2_roz_schedule_mode not in {
            "static_periodic_compat",
            "periodic_single_zone",
            "multi_zone_round_robin",
            "patch_derived_static",
        }:
            raise ValueError(f"path2_roz_schedule_mode unsupported: {path2_roz_schedule_mode}")
        path2_roz_schedule_zones = _path2_roz_schedule_zones() if path2_roz_reporting_on else [_path2_roz_default_zone()]
        path2_roz_schedule_phase_step = int(
            (int(tick) + int(path2_roz_schedule_phase_offset_steps)) % int(path2_roz_window_period_steps)
        )
        path2_roz_window_index = int(
            (int(tick) + int(path2_roz_schedule_phase_offset_steps)) // int(path2_roz_window_period_steps)
        )
        path2_roz_zone_index = (
            int(path2_roz_window_index) % len(path2_roz_schedule_zones)
            if path2_roz_schedule_mode == "multi_zone_round_robin" and path2_roz_schedule_zones
            else 0
        )
        path2_roz_active_zone = path2_roz_schedule_zones[int(path2_roz_zone_index)]
        path2_roz_center_y = int(path2_roz_active_zone["center_y"])
        path2_roz_center_x = int(path2_roz_active_zone["center_x"])
        path2_roz_zone_radius = int(path2_roz_active_zone["radius"])
        path2_roz_zone_id = str(path2_roz_active_zone["id"])
        path2_roz_window_id = int(path2_roz_window_index)
        path2_roz_window_active = bool(
            path2_roz_reporting_on
            and int(grid_h) > 0
            and int(grid_w) > 0
            and int(path2_roz_window_duration_steps) > 0
            and int(path2_roz_schedule_phase_step) < int(path2_roz_window_duration_steps)
        )

        def _path2_roz_distance(y: int, x: int) -> int:
            if not path2_roz_reporting_on or y < 0 or x < 0:
                return -1
            return int(abs(int(y) - int(path2_roz_center_y)) + abs(int(x) - int(path2_roz_center_x)))

        def _path2_roz_in_zone(y: int, x: int) -> int:
            distance = _path2_roz_distance(y, x)
            return int(bool(path2_roz_window_active and distance >= 0 and distance <= int(path2_roz_zone_radius)))

        def _path2_roz_active_cell_count() -> int:
            if not path2_roz_window_active:
                return 0
            total = 0
            radius = int(path2_roz_zone_radius)
            cy = int(path2_roz_center_y)
            cx = int(path2_roz_center_x)
            for yy in range(max(0, cy - radius), min(int(grid_h), cy + radius + 1)):
                remaining = radius - abs(int(yy) - cy)
                x0 = max(0, cx - remaining)
                x1 = min(int(grid_w), cx + remaining + 1)
                total += max(0, x1 - x0)
            return int(total)

        path2_roz_active_cell_count = _path2_roz_active_cell_count()

        def _path2_roz_context(
            *,
            parent_y: int,
            parent_x: int,
            mate_y: int = -1,
            mate_x: int = -1,
            child_y: int = -1,
            child_x: int = -1,
        ) -> dict:
            if not path2_roz_reporting_on:
                return {}
            parent_distance = _path2_roz_distance(parent_y, parent_x)
            mate_distance = _path2_roz_distance(mate_y, mate_x)
            child_distance = _path2_roz_distance(child_y, child_x)
            parent_in_zone = _path2_roz_in_zone(parent_y, parent_x)
            mate_in_zone = _path2_roz_in_zone(mate_y, mate_x)
            child_in_zone = _path2_roz_in_zone(child_y, child_x)
            return {
                "path2_roz_reporting_enable": 1,
                "path2_roz_schema_version": int(path2_roz_schema_version),
                "path2_roz_zone_mode": path2_roz_zone_mode,
                "path2_roz_window_period_steps": int(path2_roz_window_period_steps),
                "path2_roz_window_duration_steps": int(path2_roz_window_duration_steps),
                "path2_roz_zone_radius": int(path2_roz_zone_radius),
                "path2_roz_center_y": int(path2_roz_center_y),
                "path2_roz_center_x": int(path2_roz_center_x),
                "path2_roz_window_active": int(path2_roz_window_active),
                "path2_roz_window_id": int(path2_roz_window_id),
                "path2_roz_active_cell_count": int(path2_roz_active_cell_count),
                "path2_roz_parent_distance_to_zone": int(parent_distance),
                "path2_roz_mate_distance_to_zone": int(mate_distance),
                "path2_roz_child_distance_to_zone": int(child_distance),
                "path2_roz_parent_in_zone": int(parent_in_zone),
                "path2_roz_mate_in_zone": int(mate_in_zone),
                "path2_roz_child_in_zone": int(child_in_zone),
                "path2_roz_birth_in_zone": int(bool(parent_in_zone or child_in_zone)),
                "path2_roz_selected_parent_scan_in_zone": 0,
                "path2_roz_schedule_mode": path2_roz_schedule_mode,
                "path2_roz_schedule_id": path2_roz_schedule_id,
                "path2_roz_zone_id": path2_roz_zone_id,
                "path2_roz_zone_index": int(path2_roz_zone_index),
                "path2_roz_zone_count": int(len(path2_roz_schedule_zones)),
                "path2_roz_window_index": int(path2_roz_window_index),
                "path2_roz_schedule_phase_step": int(path2_roz_schedule_phase_step),
            }

        def _birth_in_bounds(y: int, x: int) -> bool:
            if runnercore32_inline_grid_scan:
                return 0 <= int(y) < int(grid_h) and 0 <= int(x) < int(grid_w)
            return bool(self.env.in_bounds(int(y), int(x)))

        def _birth_adjacent_agent_ids(agent_id: int) -> list[int]:
            aid = int(agent_id)
            if not runnercore32_inline_grid_scan or agent_grid is None:
                return self.env.adjacent_agent_ids(aid)
            y = int(pos_y[aid])
            x = int(pos_x[aid])
            adj: list[int] = []
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= int(ny) < int(grid_h) and 0 <= int(nx) < int(grid_w):
                    other = int(agent_grid[int(ny), int(nx)])
                    if other != -1 and other != aid:
                        adj.append(other)
            return adj

        corridor_cp0_birth_gate_on = corridor_cp0_central_place_eligibility_enabled(self.env_cfg, self.sim_cfg)

        def _lineage_id(agent_obj: object) -> str:
            try:
                aid = int(getattr(agent_obj, "agent_id", 0))
            except Exception:
                aid = 0
            stats = normalize_lineage_stats(getattr(agent_obj, "stats", None), aid, generation=0, tick=0)
            if stats is None:
                return format_lineage_id(aid)
            return str(getattr(stats, "lineage_id", "") or format_lineage_id(aid))

        def _pos_for(agent_obj: object) -> tuple[int, int]:
            try:
                aid = int(getattr(agent_obj, "agent_id", -1))
                if 0 <= aid < cap and bool(env_alive[aid]):
                    return int(pos_y[aid]), int(pos_x[aid])
            except Exception:
                pass
            return -1, -1

        def _grid_float(name: str, y: int, x: int) -> float:
            try:
                arr = getattr(self.env, name, None)
                if arr is None or y < 0 or x < 0:
                    return 0.0
                return float(np.asarray(arr)[int(y), int(x)])
            except Exception:
                return 0.0

        def _free_neighbors(y: int, x: int) -> int:
            if y < 0 or x < 0:
                return 0
            total = 0
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                try:
                    if self.env.in_bounds(int(yy), int(xx)) and int(self.env.agent_grid[int(yy), int(xx)]) == -1:
                        total += 1
                except Exception:
                    continue
            return int(total)

        def _parent_condition_context(parent: object) -> dict:
            stats = getattr(parent, "stats", None)
            return {
                "parent_energy": float(getattr(parent, "energy", 0.0)),
                "parent_health": float(getattr(parent, "health", 0.0)),
                "parent_age": int(getattr(stats, "survival_steps", 0) if stats is not None else 0),
                "parent_effective_fitness": float(getattr(stats, "effective_fitness", 0.0) if stats is not None else 0.0),
                "parent_reproduction_cooldown": int(getattr(parent, "repro_cooldown", 0)),
                "parent_recent_birth_attempt_count": int(getattr(stats, "offspring_count", 0) if stats is not None else 0),
            }

        def _lro1_parent_context(y: int, x: int) -> dict[str, object]:
            return {
                "density": _grid_float("local_density_map", y, x),
                "waste": _grid_float("patch_waste", y, x),
                "scent": _grid_float("patch_scent", y, x),
                "shelter_cover": max(_grid_float("patch_shelter", y, x), _grid_float("patch_cover", y, x)),
                "food": _grid_float("patch_food_amount", y, x),
                "hazard": max(_grid_float("patch_delayed_hazard_active", y, x), _grid_float("patch_delayed_hazard_cue", y, x)),
            }

        def _lro1_parent_fields(parent: object, *, base_try_prob: float | None = None) -> dict[str, object]:
            py, px = _pos_for(parent)
            return compute_lro1_reproductive_opportunity(
                _lro1_parent_context(py, px),
                enabled=lro1_reproductive_opportunity_enabled,
                mode=lro1_reproductive_opportunity_mode,
                min_multiplier=lro1_reproductive_opportunity_min_multiplier,
                max_multiplier=lro1_reproductive_opportunity_max_multiplier,
                weights=lro1_reproductive_opportunity_weights,
                report_fields=lro1_reproductive_opportunity_report,
                base_try_prob=base_try_prob,
            )

        def _lro4_base_fields(
            *,
            parent_y: int,
            parent_x: int,
            child_y: int = -1,
            child_x: int = -1,
            base_candidate_count: int = 0,
            expanded_candidate_count: int = 0,
            triggered: bool = False,
            trigger_reason: str = "",
            selected_from_expanded: bool = False,
        ) -> dict[str, object]:
            if not lro4_offspring_dispersal_report:
                return {}
            selected_radius = (
                abs(int(parent_y) - int(child_y)) + abs(int(parent_x) - int(child_x))
                if int(parent_y) >= 0 and int(parent_x) >= 0 and int(child_y) >= 0 and int(child_x) >= 0
                else -1
            )
            active_leak = int(
                (not lro4_offspring_dispersal_enabled)
                and (bool(triggered) or int(expanded_candidate_count) > 0 or bool(selected_from_expanded))
            )
            return {
                "lro4_offspring_dispersal_enable": int(lro4_offspring_dispersal_enabled),
                "lro4_offspring_dispersal_mode": lro4_offspring_dispersal_mode if lro4_offspring_dispersal_enabled else "",
                "lro4_offspring_dispersal_triggered": int(bool(triggered)),
                "lro4_offspring_dispersal_trigger_reason": str(trigger_reason or ""),
                "lro4_offspring_dispersal_base_radius": int(lro4_offspring_dispersal_base_radius),
                "lro4_offspring_dispersal_max_radius": int(lro4_offspring_dispersal_max_radius),
                "lro4_offspring_dispersal_effective_radius": int(max(lro4_offspring_dispersal_base_radius, selected_radius)),
                "lro4_offspring_dispersal_base_candidate_count": int(base_candidate_count),
                "lro4_offspring_dispersal_expanded_candidate_count": int(expanded_candidate_count),
                "lro4_offspring_dispersal_selected_radius": int(selected_radius),
                "lro4_offspring_dispersal_selected_from_expanded": int(bool(selected_from_expanded)),
                "lro4_offspring_dispersal_parent_density": _grid_float("local_density_map", parent_y, parent_x),
                "lro4_offspring_dispersal_parent_waste": _grid_float("patch_waste", parent_y, parent_x),
                "lro4_offspring_dispersal_child_density": _grid_float("local_density_map", child_y, child_x),
                "lro4_offspring_dispersal_child_waste": _grid_float("patch_waste", child_y, child_x),
                "lro4_offspring_dispersal_disabled_leakage_marker": active_leak,
            }

        def _lro4_ring_candidates(parent: object, mate_obj: object | None, reserved: set[tuple[int, int]]) -> list[tuple[int, int]]:
            centers = []
            py, px = _pos_for(parent)
            if py >= 0 and px >= 0:
                centers.append((int(py), int(px)))
            if mate_obj is not None:
                my, mx = _pos_for(mate_obj)
                if my >= 0 and mx >= 0 and (int(my), int(mx)) not in centers:
                    centers.append((int(my), int(mx)))
            out: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for radius in range(int(lro4_offspring_dispersal_base_radius) + 1, int(lro4_offspring_dispersal_max_radius) + 1):
                for cy, cx in centers:
                    for dy in range(-radius, radius + 1):
                        remaining = radius - abs(int(dy))
                        dx_values = (-remaining, remaining) if remaining else (0,)
                        for dx in dx_values:
                            nx = int(cx) + int(dx)
                            ny = int(cy) + int(dy)
                            pos = (ny, nx)
                            if pos in seen or pos in reserved:
                                continue
                            seen.add(pos)
                            if not _birth_in_bounds(ny, nx):
                                continue
                            if int(agent_grid[ny, nx]) != -1:
                                continue
                            out.append(pos)
            return out

        def _lro4_apply_offspring_dispersal(
            parent: object,
            mate_obj: object | None,
            candidates: list[tuple[int, int]],
            reserved: set[tuple[int, int]],
        ) -> tuple[list[tuple[int, int]], dict[str, object]]:
            py, px = _pos_for(parent)
            if not lro4_offspring_dispersal_enabled:
                return candidates, _lro4_base_fields(
                    parent_y=py,
                    parent_x=px,
                    base_candidate_count=len(candidates),
                )
            density = _grid_float("local_density_map", py, px)
            waste = _grid_float("patch_waste", py, px)
            reasons = []
            if bool(lro4_offspring_dispersal_no_space_trigger) and not candidates:
                reasons.append("no_base_space")
            if density >= float(lro4_offspring_dispersal_density_threshold):
                reasons.append("density")
            if waste >= float(lro4_offspring_dispersal_waste_threshold):
                reasons.append("waste")
            if not reasons:
                return candidates, _lro4_base_fields(
                    parent_y=py,
                    parent_x=px,
                    base_candidate_count=len(candidates),
                )
            expanded = _lro4_ring_candidates(parent, mate_obj, reserved)
            expanded_set = set(expanded)
            merged = list(expanded) + [pos for pos in candidates if pos not in expanded_set]
            selected = merged[0] if merged else (-1, -1)
            return merged, _lro4_base_fields(
                parent_y=py,
                parent_x=px,
                child_y=selected[0],
                child_x=selected[1],
                base_candidate_count=len(candidates),
                expanded_candidate_count=len(expanded),
                triggered=True,
                trigger_reason="+".join(reasons),
                selected_from_expanded=bool(expanded and merged and merged[0] == expanded[0]),
            )

        def _funnel_birth_context(
            parent: object,
            mate_obj: object | None,
            child_obj: object,
            child_pos: tuple[int, int],
            placement_fields: dict[str, object] | None = None,
        ) -> dict:
            py, px = _pos_for(parent)
            my, mx = _pos_for(mate_obj) if mate_obj is not None else (-1, -1)
            cy, cx = int(child_pos[0]), int(child_pos[1])
            ctx = {
                "schema_version": 2,
                "event_schema_version": 2,
                "generation": int(gen),
                "tick": int(tick),
                "event": "successful_birth",
                "event_type": "birth_success",
                "scan_id": str(scan_id or ""),
                "lineage_id": _lineage_id(child_obj),
                "child_lineage_id": _lineage_id(child_obj),
                "parent_lineage_id": _lineage_id(parent),
                "mate_lineage_id": _lineage_id(mate_obj) if mate_obj is not None else "",
                "parent_id": int(getattr(parent, "agent_id", -1)),
                "mate_id": int(getattr(mate_obj, "agent_id", -1)) if mate_obj is not None else -1,
                "child_id": int(getattr(child_obj, "agent_id", -1)),
                "value": float(getattr(child_obj, "energy", 0.0)),
                "child_energy": float(getattr(child_obj, "energy", 0.0)),
                "parent_y": int(py),
                "parent_x": int(px),
                "mate_y": int(my),
                "mate_x": int(mx),
                "child_y": int(cy),
                "child_x": int(cx),
                "local_free_neighbor_count": _free_neighbors(py, px),
                "local_density_parent_site": _grid_float("local_density_map", py, px),
                "local_density_child_site": _grid_float("local_density_map", cy, cx),
                "parent_child_distance": float(abs(int(py) - int(cy)) + abs(int(px) - int(cx))) if py >= 0 and px >= 0 else 0.0,
                "birth_attempt_count": 1,
                "birth_attempted": 1,
                "birth_success": 1,
                "birth_failed": 0,
                "food_parent_site": _grid_float("patch_food_amount", py, px),
                "food_child_site": _grid_float("patch_food_amount", cy, cx),
                "waste_parent_site": _grid_float("patch_waste", py, px),
                "waste_child_site": _grid_float("patch_waste", cy, cx),
                "scent_parent_site": _grid_float("patch_scent", py, px),
                "scent_child_site": _grid_float("patch_scent", cy, cx),
                "shelter_cover_parent_site": max(_grid_float("patch_shelter", py, px), _grid_float("patch_cover", py, px)),
                "shelter_cover_child_site": max(_grid_float("patch_shelter", cy, cx), _grid_float("patch_cover", cy, cx)),
                "predator_hazard_exposure_parent_site": max(_grid_float("patch_delayed_hazard_active", py, px), _grid_float("patch_delayed_hazard_cue", py, px)),
                "predator_hazard_exposure_child_site": max(_grid_float("patch_delayed_hazard_active", cy, cx), _grid_float("patch_delayed_hazard_cue", cy, cx)),
            }
            ctx.update(parent_child_genome_distance_summary(parent, mate_obj, child_obj))
            child_genome = getattr(child_obj, "genome", None)
            ctx.update(
                {
                    "mutation_weather_enabled": int(getattr(child_genome, "mutation_weather_enabled", 0) or 0),
                    "mutation_weather_profile": str(getattr(child_genome, "mutation_weather_profile", "") or ""),
                    "mutation_weather_active": int(getattr(child_genome, "mutation_weather_active", 0) or 0),
                    "mutation_weather_event_id": str(getattr(child_genome, "mutation_weather_event_id", "") or ""),
                    "mutation_weather_generation": int(getattr(child_genome, "mutation_weather_generation", int(gen)) or int(gen)),
                    "mutation_weather_severity": str(getattr(child_genome, "mutation_weather_severity", "off") or "off"),
                    "mutation_weather_scalar_multiplier": float(
                        getattr(child_genome, "mutation_weather_scalar_multiplier", 1.0) or 1.0
                    ),
                    "mutation_weather_count_multiplier": float(
                        getattr(child_genome, "mutation_weather_count_multiplier", 1.0) or 1.0
                    ),
                    "mutation_weather_structural_enabled": int(
                        getattr(child_genome, "mutation_weather_structural_enabled", 0) or 0
                    ),
                    "mutation_weather_affected_offspring": int(
                        getattr(child_genome, "mutation_weather_affected_offspring", 0) or 0
                    ),
                    "child_mutation_weather_active": int(
                        getattr(child_genome, "child_mutation_weather_active", 0) or 0
                    ),
                    "child_mutation_weather_event_id": str(
                        getattr(child_genome, "child_mutation_weather_event_id", "") or ""
                    ),
                    "child_mutation_weather_scalar_multiplier": float(
                        getattr(child_genome, "child_mutation_weather_scalar_multiplier", 1.0) or 1.0
                    ),
                }
            )
            ctx.update(_parent_condition_context(parent))
            ctx.update(_lro1_parent_fields(parent))
            if placement_fields:
                ctx.update(placement_fields)
            if bool(getattr(getattr(self, "sim_cfg", None), "structural_variant_identity_reporting_enable", False)):
                try:
                    from evolution.structural_variant_identity import birth_identity_context

                    ctx.update(
                        birth_identity_context(
                            parent,
                            mate_obj,
                            child_obj,
                            canonical_semantic_digest=str(
                                getattr(self, "_structural_variant_identity_canonical_semantic_digest", "") or ""
                            ),
                        )
                    )
                except Exception as exc:
                    ctx["structural_variant_identity_error"] = str(exc)
            ctx.update(
                _path2_roz_context(
                    parent_y=py,
                    parent_x=px,
                    mate_y=my,
                    mate_x=mx,
                    child_y=cy,
                    child_x=cx,
                )
            )
            return ctx

        def _append_funnel_event_context(ctx: dict) -> None:
            if not funnel_events_on:
                return
            try:
                if not hasattr(self, "_lineage_reproduction_funnel_event_contexts_gen"):
                    self._lineage_reproduction_funnel_event_contexts_gen = []
                self._lineage_reproduction_funnel_event_contexts_gen.append(dict(ctx))
            except Exception:
                pass

        def _append_funnel_scan_action_context(ctx: dict) -> None:
            if not funnel_scan_action_events_on:
                return
            try:
                if not hasattr(self, "_lineage_reproduction_funnel_scan_action_contexts_gen"):
                    self._lineage_reproduction_funnel_scan_action_contexts_gen = []
                self._lineage_reproduction_funnel_scan_action_contexts_gen.append(dict(ctx))
            except Exception:
                pass

        def _scan_context(
            parent: object,
            mate_obj: object | None = None,
            *,
            event_type: str,
            scan_reason_group: str = "",
            scan_reason_detail: str = "",
            free_spawn_cell_count: int = 0,
            placement_fields: dict[str, object] | None = None,
        ) -> dict:
            py, px = _pos_for(parent)
            my, mx = _pos_for(mate_obj) if mate_obj is not None else (-1, -1)
            parent_lineage = _lineage_id(parent)
            mate_lineage = _lineage_id(mate_obj) if mate_obj is not None else ""
            ctx = {
                "schema_version": 4,
                "event_schema_version": 4,
                "generation": int(gen),
                "tick": int(tick),
                "event": str(event_type),
                "event_type": str(event_type),
                "event_id": f"{scan_id}:{str(event_type).replace('reproduction_', '')}",
                "scan_id": str(scan_id or ""),
                "scan_phase": "live_tick_reproduction_scan",
                "scan_sequence": int(scan_sequence),
                "scan_candidate": 1,
                "scan_eligible": int(event_type == "reproduction_scan_eligible"),
                "scan_ineligible": int(event_type == "reproduction_scan_ineligible"),
                "scan_ineligible_reason_group": str(scan_reason_group or ""),
                "scan_ineligible_reason_detail": str(scan_reason_detail or ""),
                "action_not_taken": int(event_type == "reproduction_scan_no_attempt"),
                "action_not_taken_reason_group": str(scan_reason_group or ""),
                "action_not_taken_reason_detail": str(scan_reason_detail or ""),
                "failure_reason": str(scan_reason_detail or ""),
                "failure_reason_group": str(scan_reason_group or ""),
                "lineage_id": parent_lineage,
                "biological_lineage_id": parent_lineage,
                "parent_lineage_id": parent_lineage,
                "parent_biological_lineage_id": parent_lineage,
                "mate_lineage_id": mate_lineage,
                "mate_biological_lineage_id": mate_lineage,
                "parent_id": int(getattr(parent, "agent_id", -1)),
                "parent_agent_id": int(getattr(parent, "agent_id", -1)),
                "agent_id": int(getattr(parent, "agent_id", -1)),
                "mate_id": int(getattr(mate_obj, "agent_id", -1)) if mate_obj is not None else -1,
                "mate_agent_id": int(getattr(mate_obj, "agent_id", -1)) if mate_obj is not None else -1,
                "child_id": -1,
                "child_agent_id": -1,
                "parent_y": int(py),
                "parent_x": int(px),
                "mate_y": int(my),
                "mate_x": int(mx),
                "child_y": -1,
                "child_x": -1,
                "local_free_neighbor_count": _free_neighbors(py, px),
                "parent_free_neighbor_count": _free_neighbors(py, px),
                "candidate_free_spawn_cell_count": int(free_spawn_cell_count),
                "parent_candidate_free_spawn_cells": int(free_spawn_cell_count),
                "birth_attempted": 0,
                "birth_attempt_recorded": 0,
                "birth_success": 0,
                "birth_failed": 0,
                "reproduction_scan_seen": 1,
                "mate_required": int(bool(getattr(self.env_cfg, "reproduce_requires_mate", False))),
                "mate_available": int(mate_obj is not None),
                "mate_eligible": int(mate_obj is not None),
                "mate_block_reason": str(scan_reason_detail or "") if str(scan_reason_group or "") == "mate" else "",
                "local_density_parent_site": _grid_float("local_density_map", py, px),
                "agent_local_density": _grid_float("local_density_map", py, px),
                "food_parent_site": _grid_float("patch_food_amount", py, px),
                "waste_parent_site": _grid_float("patch_waste", py, px),
                "scent_parent_site": _grid_float("patch_scent", py, px),
                "shelter_cover_parent_site": max(_grid_float("patch_shelter", py, px), _grid_float("patch_cover", py, px)),
                "parent_patch_shelter": _grid_float("patch_shelter", py, px),
                "parent_patch_cover": _grid_float("patch_cover", py, px),
            }
            ctx.update(_parent_condition_context(parent))
            ctx.update(_lro1_parent_fields(parent))
            if placement_fields:
                ctx.update(placement_fields)
            ctx.update(_path2_roz_context(parent_y=py, parent_x=px, mate_y=my, mate_x=mx))
            ctx["agent_energy"] = ctx.get("parent_energy", 0.0)
            ctx["agent_health"] = ctx.get("parent_health", 0.0)
            ctx["agent_age"] = ctx.get("parent_age", 0)
            ctx["agent_effective_fitness"] = ctx.get("parent_effective_fitness", 0.0)
            ctx["agent_reproduction_cooldown"] = ctx.get("parent_reproduction_cooldown", 0)
            ctx["agent_free_neighbor_count"] = ctx.get("parent_free_neighbor_count", 0)
            ctx["agent_candidate_free_spawn_cells"] = int(free_spawn_cell_count)
            ctx["agent_patch_food"] = ctx.get("food_parent_site", 0.0)
            ctx["agent_patch_waste"] = ctx.get("waste_parent_site", 0.0)
            ctx["agent_patch_scent"] = ctx.get("scent_parent_site", 0.0)
            ctx["agent_patch_shelter"] = ctx.get("parent_patch_shelter", 0.0)
            ctx["agent_patch_cover"] = ctx.get("parent_patch_cover", 0.0)
            return ctx

        def _funnel_failed_birth_context(
            parent: object,
            mate_obj: object | None,
            *,
            failure_reason: str,
            free_spawn_cell_count: int,
            birth_attempted: bool = True,
            placement_fields: dict[str, object] | None = None,
        ) -> dict:
            if not funnel_events_on:
                return {}
            py, px = _pos_for(parent)
            my, mx = _pos_for(mate_obj) if mate_obj is not None else (-1, -1)
            is_no_space = str(failure_reason) == "birth_fail_no_space"
            ctx = {
                "schema_version": 2,
                "event_schema_version": 2,
                "generation": int(gen),
                "tick": int(tick),
                "event": "failed_birth",
                "event_type": "birth_fail_no_space" if is_no_space else "birth_fail_non_space",
                "scan_id": str(scan_id or ""),
                "failure_reason": str(failure_reason),
                "lineage_id": _lineage_id(parent),
                "child_lineage_id": "",
                "parent_lineage_id": _lineage_id(parent),
                "mate_lineage_id": _lineage_id(mate_obj) if mate_obj is not None else "",
                "parent_id": int(getattr(parent, "agent_id", -1)),
                "mate_id": int(getattr(mate_obj, "agent_id", -1)) if mate_obj is not None else -1,
                "child_id": -1,
                "parent_y": int(py),
                "parent_x": int(px),
                "mate_y": int(my),
                "mate_x": int(mx),
                "child_y": -1,
                "child_x": -1,
                "local_free_neighbor_count": _free_neighbors(py, px),
                "candidate_free_spawn_cell_count": int(free_spawn_cell_count),
                "local_density_parent_site": _grid_float("local_density_map", py, px),
                "local_density_child_site": 0.0,
                "parent_child_distance": -1.0,
                "birth_attempt_count": int(bool(birth_attempted)),
                "failed_birth_attempt_count": int(bool(birth_attempted)),
                "birth_attempted": int(bool(birth_attempted)),
                "birth_success": 0,
                "birth_failed": 1,
                "no_local_space": int(is_no_space),
                "food_parent_site": _grid_float("patch_food_amount", py, px),
                "food_child_site": 0.0,
                "waste_parent_site": _grid_float("patch_waste", py, px),
                "waste_child_site": 0.0,
                "scent_parent_site": _grid_float("patch_scent", py, px),
                "scent_child_site": 0.0,
                "shelter_cover_parent_site": max(_grid_float("patch_shelter", py, px), _grid_float("patch_cover", py, px)),
                "shelter_cover_child_site": 0.0,
                "predator_hazard_exposure_parent_site": max(
                    _grid_float("patch_delayed_hazard_active", py, px),
                    _grid_float("patch_delayed_hazard_cue", py, px),
                ),
                "predator_hazard_exposure_child_site": 0.0,
            }
            ctx.update(_parent_condition_context(parent))
            ctx.update(_lro1_parent_fields(parent))
            if placement_fields:
                ctx.update(placement_fields)
            ctx.update(_path2_roz_context(parent_y=py, parent_x=px, mate_y=my, mate_x=mx))
            return ctx

        scan_sequence = 0
        scan_id = ""
        for agent in alive:
            if not agent.alive:
                continue
            scan_sequence += 1
            scan_id = f"g{int(gen):06d}:t{int(tick):06d}:scan:{int(getattr(agent, 'agent_id', -1)):08d}:{int(scan_sequence):06d}"
            if funnel_scan_action_events_on:
                _append_funnel_scan_action_context(
                    _scan_context(
                        agent,
                        None,
                        event_type="reproduction_scan_exposure",
                    )
                )

            if getattr(agent, "repro_cooldown", 0) > 0:
                agent.repro_cooldown -= 1
            if agent.repro_cooldown != 0:
                self._birth_pre_cooldown_gen += 1
                if funnel_scan_action_events_on:
                    _append_funnel_scan_action_context(
                        _scan_context(
                            agent,
                            None,
                            event_type="reproduction_scan_ineligible",
                            scan_reason_group="cooldown_age",
                            scan_reason_detail="birth_pre_cooldown",
                        )
                    )
                if funnel_events_on:
                    _append_funnel_event_context(
                        _funnel_failed_birth_context(
                            agent,
                            None,
                            failure_reason="birth_pre_cooldown",
                            free_spawn_cell_count=0,
                            birth_attempted=False,
                        )
                    )
                continue

            if agent.agent_id in mated_this_tick:
                if funnel_scan_action_events_on:
                    _append_funnel_scan_action_context(
                        _scan_context(
                            agent,
                            None,
                            event_type="reproduction_scan_ineligible",
                            scan_reason_group="mate",
                            scan_reason_detail="already_mated_this_tick",
                        )
                    )
                continue

            if corridor_cp0_birth_gate_on and not corridor_cp0_agent_home_eligible(self.env, agent):
                if funnel_scan_action_events_on:
                    _append_funnel_scan_action_context(
                        _scan_context(
                            agent,
                            None,
                            event_type="reproduction_scan_ineligible",
                            scan_reason_group="central_place",
                            scan_reason_detail="corridor_cp0_parent_not_home_local",
                        )
                    )
                continue

            _repro_thr_self_fast = _runnercore31_reproduction_threshold_or_none(agent)
            if _repro_thr_self_fast is not None:
                _repro_thr_self = float(_repro_thr_self_fast)
            else:
                _scalars_self = (
                    self.resolve_developmental_effective_scalars_for_agent(agent)
                    if hasattr(self, "resolve_developmental_effective_scalars_for_agent")
                    else None
                )
                if _scalars_self is not None:
                    _repro_thr_self = float(_scalars_self[6])
                else:
                    _dev_self = self.resolve_developmental_expression_for_agent(agent) if hasattr(self, "resolve_developmental_expression_for_agent") else {}
                    _eff_self = _dev_self.get("effective", {}) if isinstance(_dev_self, dict) else {}
                    _repro_thr_self = float(_eff_self.get("reproduction_threshold", agent.genome.reproduction_threshold))
            if agent.energy < _repro_thr_self:
                self._birth_pre_no_energy_gen += 1
                if funnel_scan_action_events_on:
                    _append_funnel_scan_action_context(
                        _scan_context(
                            agent,
                            None,
                            event_type="reproduction_scan_ineligible",
                            scan_reason_group="energy",
                            scan_reason_detail="birth_pre_no_energy",
                        )
                    )
                if funnel_events_on:
                    _append_funnel_event_context(
                        _funnel_failed_birth_context(
                            agent,
                            None,
                            failure_reason="birth_pre_no_energy",
                            free_spawn_cell_count=0,
                            birth_attempted=False,
                        )
                    )
                continue

            inj = float(getattr(agent, "injury", 0.0))
            scar = float(getattr(agent, "scar", 0.0))

            if injury_gate is not None and inj > injury_gate:
                self._injury_blocked_births_gen += 1
                self._birth_pre_injury_gate_gen += 1
                self._birth_precheck_injury_gate_gen += 1
                self._birth_fail_injury_gate_gen += 1
                if funnel_scan_action_events_on:
                    _append_funnel_scan_action_context(
                        _scan_context(
                            agent,
                            None,
                            event_type="reproduction_scan_ineligible",
                            scan_reason_group="health",
                            scan_reason_detail="birth_fail_injury_gate",
                        )
                    )
                if funnel_events_on:
                    _append_funnel_event_context(
                        _funnel_failed_birth_context(
                            agent,
                            None,
                            failure_reason="birth_fail_injury_gate",
                            free_spawn_cell_count=0,
                            birth_attempted=False,
                        )
                    )
                continue
            if scar_gate is not None and scar > scar_gate:
                self._birth_precheck_scar_gate_gen += 1
                self._birth_fail_scar_gate_gen += 1
                if funnel_scan_action_events_on:
                    _append_funnel_scan_action_context(
                        _scan_context(
                            agent,
                            None,
                            event_type="reproduction_scan_ineligible",
                            scan_reason_group="health",
                            scan_reason_detail="birth_fail_scar_gate",
                        )
                    )
                if funnel_events_on:
                    _append_funnel_event_context(
                        _funnel_failed_birth_context(
                            agent,
                            None,
                            failure_reason="birth_fail_scar_gate",
                            free_spawn_cell_count=0,
                            birth_attempted=False,
                        )
                    )
                continue

            viability_self = 1.0
            if chem_repro_enable:
                cself_raw = getattr(agent, "chemistry", None)
                cself = chemistry_from_values(cself_raw if cself_raw is not None else (0.0,) * 6)
                viability_self = reproduction_viability_scale_from_chem_row(cself, floor=chem_floor, strength=chem_strength)
                if float(cself[CHEM_SLOT_CHRONIC_DAMAGE]) >= chem_chronic_block:
                    self._birth_fail_injury_gate_gen += 1
                    if funnel_scan_action_events_on:
                        _append_funnel_scan_action_context(
                            _scan_context(
                                agent,
                                None,
                                event_type="reproduction_scan_ineligible",
                                scan_reason_group="health",
                                scan_reason_detail="birth_fail_injury_gate",
                            )
                        )
                    if funnel_events_on:
                        _append_funnel_event_context(
                            _funnel_failed_birth_context(
                                agent,
                                None,
                                failure_reason="birth_fail_injury_gate",
                                free_spawn_cell_count=0,
                                birth_attempted=False,
                            )
                        )
                    continue
            try:
                agent.stats.chemistry_repro_viability = float(viability_self)
            except Exception:
                pass

            mate = None
            mate_inj = 0.0
            mate_scar = 0.0
            viability_mate = 1.0

            if self.env_cfg.reproduce_requires_mate:
                adj = _birth_adjacent_agent_ids(agent.agent_id)
                if trace_on and trace_lines < trace_max:
                    payload = {"stage": "mate_adj_pre", "gen": gen, "tick": tick, "parent": int(agent.agent_id), "adj": list(map(int, adj))}
                    _emit_jsonl("birth_internal_trace.jsonl", "[BIRTH_INTERNAL_TRACE]", payload)
                    trace_lines += 1
                if not adj:
                    self._birth_pre_no_mate_gen += 1
                    self._birth_precheck_no_mate_gen += 1
                    self._birth_fail_no_mate_gen += 1
                    if funnel_scan_action_events_on:
                        _append_funnel_scan_action_context(
                            _scan_context(
                                agent,
                                None,
                                event_type="reproduction_scan_ineligible",
                                scan_reason_group="mate",
                                scan_reason_detail="birth_fail_no_mate",
                            )
                        )
                    if funnel_events_on:
                        _append_funnel_event_context(
                            _funnel_failed_birth_context(
                                agent,
                                None,
                                failure_reason="birth_fail_no_mate",
                                free_spawn_cell_count=0,
                                birth_attempted=False,
                            )
                        )
                    continue

                self.np_rng.shuffle(adj)
                if trace_on and trace_lines < trace_max:
                    payload = {"stage": "mate_adj_post", "gen": gen, "tick": tick, "parent": int(agent.agent_id), "adj": list(map(int, adj)), "runner_np_rng_sha": _rng_digest(getattr(self, "np_rng", None))}
                    _emit_jsonl("birth_internal_trace.jsonl", "[BIRTH_INTERNAL_TRACE]", payload)
                    trace_lines += 1
                for cand_id in adj:
                    cand = self.pop.agents.get(int(cand_id))
                    if cand is None or (not cand.alive):
                        continue
                    if getattr(cand, "repro_cooldown", 0) != 0:
                        continue
                    if cand.agent_id in mated_this_tick:
                        continue
                    if corridor_cp0_birth_gate_on and not corridor_cp0_agent_home_eligible(self.env, cand):
                        continue
                    _repro_thr_cand_fast = _runnercore31_reproduction_threshold_or_none(cand)
                    if _repro_thr_cand_fast is not None:
                        _repro_thr_cand = float(_repro_thr_cand_fast)
                    else:
                        _scalars_cand = (
                            self.resolve_developmental_effective_scalars_for_agent(cand)
                            if hasattr(self, "resolve_developmental_effective_scalars_for_agent")
                            else None
                        )
                        if _scalars_cand is not None:
                            _repro_thr_cand = float(_scalars_cand[6])
                        else:
                            _dev_cand = self.resolve_developmental_expression_for_agent(cand) if hasattr(self, "resolve_developmental_expression_for_agent") else {}
                            _eff_cand = _dev_cand.get("effective", {}) if isinstance(_dev_cand, dict) else {}
                            _repro_thr_cand = float(_eff_cand.get("reproduction_threshold", cand.genome.reproduction_threshold))
                    if cand.energy < _repro_thr_cand:
                        continue

                    cand_inj = float(getattr(cand, "injury", 0.0))
                    cand_scar = float(getattr(cand, "scar", 0.0))

                    if injury_gate is not None and cand_inj > injury_gate:
                        continue
                    if scar_gate is not None and cand_scar > scar_gate:
                        continue
                    if chem_repro_enable:
                        ccand_raw = getattr(cand, "chemistry", None)
                        ccand = chemistry_from_values(ccand_raw if ccand_raw is not None else (0.0,) * 6)
                        viability_mate = reproduction_viability_scale_from_chem_row(ccand, floor=chem_floor, strength=chem_strength)
                        if float(ccand[CHEM_SLOT_CHRONIC_DAMAGE]) >= chem_chronic_block:
                            continue
                        try:
                            cand.stats.chemistry_repro_viability = float(viability_mate)
                        except Exception:
                            pass

                    mate = cand
                    if trace_on and trace_lines < trace_max:
                        payload = {"stage": "mate_selected", "gen": gen, "tick": tick, "parent": int(agent.agent_id), "mate": int(cand.agent_id)}
                        _emit_jsonl("birth_internal_trace.jsonl", "[BIRTH_INTERNAL_TRACE]", payload)
                        trace_lines += 1
                    mate_inj = cand_inj
                    mate_scar = cand_scar
                    break

                if mate is None:
                    self._birth_pre_no_mate_gen += 1
                    self._birth_precheck_no_mate_gen += 1
                    self._birth_fail_no_mate_gen += 1
                    if funnel_scan_action_events_on:
                        _append_funnel_scan_action_context(
                            _scan_context(
                                agent,
                                None,
                                event_type="reproduction_scan_ineligible",
                                scan_reason_group="mate",
                                scan_reason_detail="birth_fail_no_mate",
                            )
                        )
                    if funnel_events_on:
                        _append_funnel_event_context(
                            _funnel_failed_birth_context(
                                agent,
                                None,
                                failure_reason="birth_fail_no_mate",
                                free_spawn_cell_count=0,
                                birth_attempted=False,
                            )
                        )
                    continue

            candidates: list[Tuple[int, int]] = []
            if mate is None:
                pos = self.env.find_adjacent_empty(agent.agent_id)
                if pos is not None and pos not in reserved_tiles:
                    candidates = [pos]
            else:
                aid = int(agent.agent_id)
                mid = int(mate.agent_id)

                # Ensure both are actually in env (defensive)
                if not (0 <= aid < cap and 0 <= mid < cap and bool(env_alive[aid]) and bool(env_alive[mid])):
                    self._birth_pre_no_mate_gen += 1
                    self._birth_precheck_no_mate_gen += 1
                    self._birth_fail_no_mate_gen += 1
                    if funnel_scan_action_events_on:
                        _append_funnel_scan_action_context(
                            _scan_context(
                                agent,
                                mate,
                                event_type="reproduction_scan_ineligible",
                                scan_reason_group="mate",
                                scan_reason_detail="birth_fail_no_mate",
                            )
                        )
                    if funnel_events_on:
                        _append_funnel_event_context(
                            _funnel_failed_birth_context(
                                agent,
                                mate,
                                failure_reason="birth_fail_no_mate",
                                free_spawn_cell_count=0,
                                birth_attempted=False,
                            )
                        )
                    continue

                ay = int(pos_y[aid]); ax = int(pos_x[aid])
                my = int(pos_y[mid]); mx = int(pos_x[mid])

                cand = [
                    (ay - 1, ax), (ay + 1, ax), (ay, ax - 1), (ay, ax + 1),
                    (my - 1, mx), (my + 1, mx), (my, mx - 1), (my, mx + 1),
                ]
                if trace_on and trace_lines < trace_max:
                    payload = {"stage": "spawn_cand_pre_shuffle", "gen": gen, "tick": tick, "parent": int(agent.agent_id), "mate": (int(mate.agent_id) if mate is not None else None), "cand": [[int(p[0]), int(p[1])] for p in cand], "env_rng_sha": _rng_digest(getattr(getattr(self, "env", None), "rng", None))}
                    _emit_jsonl("birth_internal_trace.jsonl", "[BIRTH_INTERNAL_TRACE]", payload)
                    trace_lines += 1
                self.env.rng.shuffle(cand)
                if trace_on and trace_lines < trace_max:
                    payload = {"stage": "spawn_cand_post_shuffle", "gen": gen, "tick": tick, "parent": int(agent.agent_id), "mate": (int(mate.agent_id) if mate is not None else None), "cand": [[int(p[0]), int(p[1])] for p in cand], "env_rng_sha": _rng_digest(getattr(getattr(self, "env", None), "rng", None))}
                    _emit_jsonl("birth_internal_trace.jsonl", "[BIRTH_INTERNAL_TRACE]", payload)
                    trace_lines += 1
                for ny, nx in cand:
                    if not _birth_in_bounds(ny, nx):
                        continue
                    if agent_grid[ny, nx] != -1:
                        continue
                    if (ny, nx) in reserved_tiles:
                        continue
                    candidates.append((ny, nx))
                if candidates:
                    seen = set()
                    uniq = []
                    for p in candidates:
                        if p not in seen:
                            uniq.append(p)
                            seen.add(p)
                    candidates = uniq

            candidates, lro4_placement_fields = _lro4_apply_offspring_dispersal(agent, mate, candidates, reserved_tiles)
            spawn_pos = None
            for p in candidates:
                if p not in reserved_tiles:
                    spawn_pos = p
                    break
            if spawn_pos is not None:
                py, px = _pos_for(agent)
                lro4_placement_fields = _lro4_base_fields(
                    parent_y=py,
                    parent_x=px,
                    child_y=int(spawn_pos[0]),
                    child_x=int(spawn_pos[1]),
                    base_candidate_count=int(
                        lro4_placement_fields.get("lro4_offspring_dispersal_base_candidate_count", len(candidates))
                    ),
                    expanded_candidate_count=int(
                        lro4_placement_fields.get("lro4_offspring_dispersal_expanded_candidate_count", 0)
                    ),
                    triggered=bool(int(lro4_placement_fields.get("lro4_offspring_dispersal_triggered", 0) or 0)),
                    trigger_reason=str(lro4_placement_fields.get("lro4_offspring_dispersal_trigger_reason", "")),
                    selected_from_expanded=bool(
                        int(lro4_placement_fields.get("lro4_offspring_dispersal_selected_from_expanded", 0) or 0)
                    ),
                )

            if spawn_pos is None:
                self._birth_pre_no_space_gen += 1
                self._birth_precheck_no_space_gen += 1
                self._birth_fail_no_space_gen += 1
                if funnel_scan_action_events_on:
                    _append_funnel_scan_action_context(
                        _scan_context(
                            agent,
                            mate,
                            event_type="reproduction_scan_ineligible",
                            scan_reason_group="no_space",
                            scan_reason_detail="birth_fail_no_space",
                            free_spawn_cell_count=len(candidates),
                            placement_fields=lro4_placement_fields,
                        )
                    )
                if funnel_events_on:
                    _append_funnel_event_context(
                        _funnel_failed_birth_context(
                            agent,
                            mate,
                            failure_reason="birth_fail_no_space",
                            free_spawn_cell_count=len(candidates),
                            placement_fields=lro4_placement_fields,
                        )
                    )
                continue

            lro1_try_fields = _lro1_parent_fields(agent, base_try_prob=float(try_prob))
            effective_try_prob = float(lro1_try_fields.get("lro1_opportunity_effective_try_prob", try_prob) or try_prob)

            if effective_try_prob < 1.0 and float(self.np_rng.random()) > effective_try_prob:
                if funnel_scan_action_events_on:
                    ctx = _scan_context(
                        agent,
                        mate,
                        event_type="reproduction_scan_no_attempt",
                        scan_reason_group="stochastic_try_probability",
                        scan_reason_detail="try_probability_not_taken",
                        free_spawn_cell_count=len(candidates),
                        placement_fields=lro4_placement_fields,
                    )
                    ctx.update(lro1_try_fields)
                    _append_funnel_scan_action_context(ctx)
                continue

            if funnel_scan_action_events_on:
                ctx = _scan_context(
                    agent,
                    mate,
                    event_type="reproduction_scan_eligible",
                    free_spawn_cell_count=len(candidates),
                    placement_fields=lro4_placement_fields,
                )
                ctx.update(lro1_try_fields)
                _append_funnel_scan_action_context(ctx)

            if mate is None:
                give1 = agent.energy * share
                if agent.energy - give1 < buffer:
                    failed_births += 1
                    self._birth_fail_buffer_gen += 1
                    if funnel_events_on:
                        _append_funnel_event_context(
                            _funnel_failed_birth_context(
                                agent,
                                None,
                                failure_reason="birth_fail_buffer",
                                free_spawn_cell_count=len(candidates),
                                placement_fields=lro4_placement_fields,
                            )
                        )
                    continue
            else:
                give1 = agent.energy * share
                give2 = mate.energy * share
                if (agent.energy - give1 < buffer) or (mate.energy - give2 < buffer):
                    failed_births += 1
                    self._birth_fail_buffer_gen += 1
                    if funnel_events_on:
                        _append_funnel_event_context(
                            _funnel_failed_birth_context(
                                agent,
                                mate,
                                failure_reason="birth_fail_buffer",
                                free_spawn_cell_count=len(candidates),
                                placement_fields=lro4_placement_fields,
                            )
                        )
                    continue

            self._birth_attempts_gen += 1

            if (k_inj > 0.0) or (k_scar > 0.0):
                p1 = float(np.exp(-(k_inj * inj + k_scar * scar)))
                p1 *= float(viability_self)
                if float(self.np_rng.random()) > p1:
                    failed_births += 1
                    self._birth_attempt_conception_fail_gen += 1
                    self._birth_fail_conception_gen += 1
                    if funnel_events_on:
                        _append_funnel_event_context(
                            _funnel_failed_birth_context(
                                agent,
                                mate,
                                failure_reason="birth_fail_conception",
                                free_spawn_cell_count=len(candidates),
                                placement_fields=lro4_placement_fields,
                            )
                        )
                    continue

                if mate is not None:
                    p2 = float(np.exp(-(k_inj * mate_inj + k_scar * mate_scar)))
                    p2 *= float(viability_mate)
                    if float(self.np_rng.random()) > p2:
                        failed_births += 1
                        self._birth_attempt_conception_fail_gen += 1
                        self._birth_fail_conception_gen += 1
                        if funnel_events_on:
                            _append_funnel_event_context(
                                _funnel_failed_birth_context(
                                    agent,
                                    mate,
                                    failure_reason="birth_fail_conception",
                                    free_spawn_cell_count=len(candidates),
                                    placement_fields=lro4_placement_fields,
                                )
                            )
                        continue

            reserved_tiles.add(spawn_pos)

            if mate is None:
                agent.energy -= give1
                child_energy = give1

                if (p_fail_scar > 0.0) or (p_fail_inj > 0.0):
                    p_fail = min(0.95, p_fail_scar * scar + p_fail_inj * inj)
                    if float(self.np_rng.random()) < p_fail:
                        agent.energy += give1
                        failed_births += 1
                        self._birth_gestation_fail_gen += 1
                        self._birth_fail_gestation_gen += 1
                        reserved_tiles.discard(spawn_pos)
                        if funnel_events_on:
                            _append_funnel_event_context(
                                _funnel_failed_birth_context(
                                    agent,
                                    None,
                                    failure_reason="birth_fail_gestation",
                                    free_spawn_cell_count=len(candidates),
                                    placement_fields=lro4_placement_fields,
                                )
                            )
                        continue

                _t_mc = time.perf_counter()
                child = _runnercore25_make_child(agent, agent, child_energy, buckets)
                _t_make_child_sum += time.perf_counter() - _t_mc
                if child is None:
                    agent.energy += give1
                    failed_births += 1
                    self._birth_make_child_fail_gen += 1
                    reserved_tiles.discard(spawn_pos)
                    if funnel_events_on:
                        _append_funnel_event_context(
                            _funnel_failed_birth_context(
                                agent,
                                None,
                                failure_reason="birth_make_child_fail",
                                free_spawn_cell_count=len(candidates),
                                placement_fields=lro4_placement_fields,
                            )
                        )
                    continue

                child.injury = 0.0
                child.scar = 0.0
                child.rest_streak = 0
                assign_child_lineage(child, agent, agent, generation=int(gen), tick=int(tick))
                if funnel_events_on:
                    ctx = _funnel_birth_context(
                        agent,
                        None,
                        child,
                        spawn_pos,
                        lro4_placement_fields,
                    )
                    child._lineage_reproduction_funnel_birth_context = ctx
                    if checkpoint_event_tape_capture_on:
                        _append_funnel_event_context(ctx)
                        child._lineage_reproduction_funnel_event_context_emitted = True
                births.append((child, spawn_pos))

                agent.stats.offspring_count += 1

                # Analysis-only: first reproduction timing
                try:
                    st = agent.stats
                    if float(getattr(st, "offspring_count", 0.0)) > 0.0 and getattr(st, "first_repro_step", None) is None:
                        st.first_repro_step = int(getattr(st, "survival_steps", 0))
                except Exception:
                    pass

                agent.repro_cooldown = cd
                mated_this_tick.add(agent.agent_id)

            else:
                agent.energy -= give1
                mate.energy -= give2
                child_energy = give1 + give2

                if (p_fail_scar > 0.0) or (p_fail_inj > 0.0):
                    p_fail = min(
                        0.95,
                        p_fail_scar * (scar + mate_scar) + p_fail_inj * (inj + mate_inj),
                    )
                    if float(self.np_rng.random()) < p_fail:
                        agent.energy += give1
                        mate.energy += give2
                        failed_births += 1
                        self._birth_gestation_fail_gen += 1
                        self._birth_fail_gestation_gen += 1
                        reserved_tiles.discard(spawn_pos)
                        if funnel_events_on:
                            _append_funnel_event_context(
                                _funnel_failed_birth_context(
                                    agent,
                                    mate,
                                    failure_reason="birth_fail_gestation",
                                    free_spawn_cell_count=len(candidates),
                                    placement_fields=lro4_placement_fields,
                                )
                            )
                        continue

                _t_mc0 = time.perf_counter()
                child = _runnercore25_make_child(agent, mate, child_energy, buckets)
                _t_make_child_sum += (time.perf_counter() - _t_mc0)
                if child is None:
                    agent.energy += give1
                    mate.energy += give2
                    failed_births += 1
                    self._birth_make_child_fail_gen += 1
                    reserved_tiles.discard(spawn_pos)
                    if funnel_events_on:
                        _append_funnel_event_context(
                            _funnel_failed_birth_context(
                                agent,
                                mate,
                                failure_reason="birth_make_child_fail",
                                free_spawn_cell_count=len(candidates),
                                placement_fields=lro4_placement_fields,
                            )
                        )
                    continue

                child.injury = 0.0
                child.scar = 0.0
                child.rest_streak = 0
                assign_child_lineage(child, agent, mate, generation=int(gen), tick=int(tick))
                if funnel_events_on:
                    ctx = _funnel_birth_context(
                        agent,
                        mate,
                        child,
                        spawn_pos,
                        lro4_placement_fields,
                    )
                    child._lineage_reproduction_funnel_birth_context = ctx
                    if checkpoint_event_tape_capture_on:
                        _append_funnel_event_context(ctx)
                        child._lineage_reproduction_funnel_event_context_emitted = True
                births.append((child, spawn_pos))

                agent.stats.offspring_count += 1
                mate.stats.offspring_count += 1

                # Analysis-only: first reproduction timing (both parents)
                try:
                    st1 = agent.stats
                    if float(getattr(st1, "offspring_count", 0.0)) > 0.0 and getattr(st1, "first_repro_step", None) is None:
                        st1.first_repro_step = int(getattr(st1, "survival_steps", 0))
                except Exception:
                    pass
                try:
                    st2 = mate.stats
                    if float(getattr(st2, "offspring_count", 0.0)) > 0.0 and getattr(st2, "first_repro_step", None) is None:
                        st2.first_repro_step = int(getattr(st2, "survival_steps", 0))
                except Exception:
                    pass

                agent.repro_cooldown = cd
                mate.repro_cooldown = cd
                mated_this_tick.add(agent.agent_id)
                mated_this_tick.add(mate.agent_id)

        if isinstance(buckets, dict):
            _loop_dt = time.perf_counter() - _t_birth_scan0
            buckets["birth_scan_ms"] = buckets.get("birth_scan_ms", 0.0) + _loop_dt
            buckets["birth_make_child_ms"] = buckets.get("birth_make_child_ms", 0.0) + _t_make_child_sum
            buckets["birth_scan_excl_make_child_ms"] = buckets.get("birth_scan_excl_make_child_ms", 0.0) + max(0.0, _loop_dt - _t_make_child_sum)

        t_birth0 = time.perf_counter()
        newborns = []
        for child, pos in births:
            try:
                self.env.add_agent(child.agent_id, pos=pos)
                successful_births += 1
                newborns.append(child)
                if runnercore25_birth_row_spec_bridge:
                    try:
                        runnercore25_successful_ids.append(int(child.agent_id))
                    except Exception:
                        pass
                if funnel_events_on:
                    try:
                        if not hasattr(self, "_lineage_reproduction_funnel_birth_contexts_gen"):
                            self._lineage_reproduction_funnel_birth_contexts_gen = []
                        ctx = dict(getattr(child, "_lineage_reproduction_funnel_birth_context", {}) or {})
                        if ctx:
                            self._lineage_reproduction_funnel_birth_contexts_gen.append(ctx)
                            if not bool(getattr(child, "_lineage_reproduction_funnel_event_context_emitted", False)):
                                _append_funnel_event_context(ctx)
                    except Exception:
                        pass
                if bool(getattr(self.env, "patch_world_enable", False)):
                    try:
                        from env.gridworld_pkg.patch_world import sample_patch_context  # noqa: WPS433

                        if not hasattr(self, "_patch_birth_contexts_gen"):
                            self._patch_birth_contexts_gen = []
                        self._patch_birth_contexts_gen.append(
                            sample_patch_context(
                                self.env,
                                int(pos[0]),
                                int(pos[1]),
                                agent=child,
                            )
                        )
                    except Exception:
                        pass
                    try:
                        from env.gridworld_pkg.jungle import queue_jungle_birth_emission  # noqa: WPS433

                        scent_emit, waste_emit = queue_jungle_birth_emission(
                            self.env,
                            int(pos[0]),
                            int(pos[1]),
                        )
                        if (scent_emit > 0.0) or (waste_emit > 0.0):
                            self._jungle_birth_emission_count_gen = int(
                                getattr(self, "_jungle_birth_emission_count_gen", 0)
                            ) + 1
                            self._jungle_birth_scent_emitted_total_gen = float(
                                getattr(self, "_jungle_birth_scent_emitted_total_gen", 0.0)
                            ) + float(scent_emit)
                            self._jungle_birth_waste_emitted_total_gen = float(
                                getattr(self, "_jungle_birth_waste_emitted_total_gen", 0.0)
                            ) + float(waste_emit)
                            if not hasattr(self, "_jungle_birth_emission_contexts_gen"):
                                self._jungle_birth_emission_contexts_gen = []
                            self._jungle_birth_emission_contexts_gen.append(
                                sample_patch_context(
                                    self.env,
                                    int(pos[0]),
                                    int(pos[1]),
                                    agent=child,
                                )
                            )
                    except Exception:
                        pass

                try:
                    bo = getattr(self, "branch_observability", None)
                    if bo is not None:
                        parent_ids = tuple(getattr(getattr(child, "stats", None), "parent_ids", ()) or ())
                        primary_parent = agent
                        secondary_parent = mate
                        if parent_ids:
                            primary_parent = self.pop.agents.get(int(parent_ids[0]), agent)
                            if len(parent_ids) > 1:
                                secondary_parent = self.pop.agents.get(int(parent_ids[1]), mate)
                        bo.record_successful_birth(
                            generation=int(gen),
                            tick=int(tick),
                            child=child,
                            primary_parent=primary_parent,
                            secondary_parent=secondary_parent,
                            child_pos=(int(pos[0]), int(pos[1])),
                            placement_fields=lro4_placement_fields,
                            env=self.env,
                        )
                except Exception:
                    pass

                self.behavior_logger.register_agent(
                    child.agent_id,
                    cohort="born",
                    injuries={int(child.agent_id): float(getattr(child, "injury", 0.0))},
                )
            except ValueError:
                failed_births += 1
                self._birth_fail_no_space_gen += 1
                child.alive = False
                self.pop.agents.pop(child.agent_id, None)

        if newborns:
            self._sync_bank_for_agents(newborns, buckets=buckets)
            if hasattr(self, "_sync_gru_bank_for_agents"):
                self._sync_gru_bank_for_agents(newborns, buckets=buckets)

        if runnercore25_birth_row_spec_bridge:
            try:
                self._runnercore25_full_birth_row_spec_newborn_ids = sorted(
                    int(aid) for aid in runnercore25_successful_ids
                )
            except Exception:
                self._runnercore25_full_birth_row_spec_newborn_ids = []
            _runnercore25_stat_add("runnercore25_full_birth_row_specs_applied", len(runnercore25_successful_ids))

        buckets["births_apply"] = buckets.get("births_apply", 0.0) + (time.perf_counter() - t_birth0)
        # Phase 0 perf: full reproduction time (values are seconds).
        buckets["reproduction_ms"] = buckets.get("reproduction_ms", 0.0) + (time.perf_counter() - t_repro0)


        # ------------------------------------------------------------
        # RNG TRACE – POST BIRTH
        # ------------------------------------------------------------
        if trace_on and trace_rng:
            try:
                rng = getattr(self, "rng_manager", None)
                if rng is not None:
                    snap = rng.debug_snapshot()
                    payload = {
                        "stage": "post_birth",
                        "generation": gen,
                        "tick": tick,
                        "policy_index": snap.get("policy_index"),
                        "policy_digest": snap.get("policy_digest"),
                        "alive_n_after": len(list(self.pop.alive_agents())),
                    }
                    _emit_jsonl("birth_rng_trace.jsonl", "[BIRTH_RNG_TRACE]", payload)
            except Exception as _e:
                _emit_jsonl(
                    "birth_rng_trace_errors.jsonl",
                    "[BIRTH_RNG_TRACE_ERROR_POST]",
                    {
                        "stage": "rng_trace_error_post",
                        "gen": gen,
                        "tick": tick,
                        "error": str(_e),
                    },
                )



        # ------------------------------------------------------------
        # Birth-internal trace (exit)
        # ------------------------------------------------------------
        if trace_on and trace_lines < trace_max:
            try:
                alive_after = list(self.pop.alive_agents())
                alive_ids_after = [int(a.agent_id) for a in alive_after]
                parent_set = set(parent_order)
                newborn_ids = sorted([aid for aid in alive_ids_after if aid not in parent_set])
            except Exception:
                newborn_ids = []

            payload = {
                "stage": "birth_internal_exit",
                "gen": gen,
                "tick": tick,
                "successful_births_out": int(successful_births),
                "failed_births_out": int(failed_births),
                "births_len": int(len(births)) if 'births' in locals() else -1,
                "newborn_ids": newborn_ids,
                "runner_np_rng_sha": _rng_digest(getattr(self, "np_rng", None)),
                "env_rng_sha": _rng_digest(getattr(getattr(self, "env", None), "rng", None)),
            }
            _emit_jsonl("birth_internal_trace.jsonl", "[BIRTH_INTERNAL_TRACE]", payload)

        return successful_births, failed_births
