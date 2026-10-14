from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = "env_hotpath_expression_summary_v1"
DECOMPOSITION_SCHEMA_VERSION = "env_apply_decomposition_summary_v1"
PHENOTYPE_DECOMPOSITION_SCHEMA_VERSION = "envhot3_phenotype_decomposition_summary_v1"
DEVELOPMENTAL_RESIDUAL_SCHEMA_VERSION = "envhot7_developmental_residual_summary_v1"
DEVELOPMENTAL_CACHE_EQUALITY_SCHEMA_VERSION = "envhot8_developmental_cache_equality_summary_v1"
DEVELOPMENTAL_EFFECTIVE_PROJECTION_SCHEMA_VERSION = "envhot9_developmental_effective_projection_summary_v1"

DEVELOPMENTAL_EFFECTIVE_PROJECTION_FIELDS = (
    "aggression_bias",
    "action_temperature",
    "drive_logit_gain",
    "drive_inertia",
    "drive_noise",
    "pred_lr",
    "reproduction_threshold",
)
RUNNER_ONLY_EXPRESSION_METADATA_FIELDS = (
    "lock_event_count",
    "lock_events",
    "lock_state_by_module_id",
    "structural_change_ignored",
    "structural_change_reason",
)

_CURRENT_DIAGNOSTICS: ContextVar["EnvHotpathExpressionDiagnostics | None"] = ContextVar(
    "env_hotpath_expression_diagnostics",
    default=None,
)


def _stable_key(*parts: Any) -> str:
    return "|".join(str(part) for part in parts)


def current_diagnostics() -> "EnvHotpathExpressionDiagnostics | None":
    return _CURRENT_DIAGNOSTICS.get()


def set_current_diagnostics(diag: "EnvHotpathExpressionDiagnostics | None"):
    return _CURRENT_DIAGNOSTICS.set(diag)


def reset_current_diagnostics(token: Any) -> None:
    _CURRENT_DIAGNOSTICS.reset(token)


def stable_payload_key(raw: Any) -> tuple[str, int]:
    if not isinstance(raw, dict):
        return f"non_dict:{type(raw).__name__}", 0
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return encoded, len(encoded)


@dataclass
class EnvHotpathExpressionDiagnostics:
    run_dir: str
    schema_version: int = 1
    decomposition_schema_version: int = 1
    phenotype_decomposition_schema_version: int = 1
    developmental_residual_schema_version: int = 1
    developmental_cache_equality_schema_version: int = 1
    developmental_effective_projection_schema_version: int = 1
    decomposition_enabled: bool = False
    phenotype_decomposition_enabled: bool = False
    developmental_residual_enabled: bool = False
    developmental_cache_equality_enabled: bool = False
    developmental_effective_projection_enabled: bool = False
    write_events: bool = False
    started_at_monotonic: float = field(default_factory=time.perf_counter)
    dev_expression_requests: int = 0
    dev_expression_cache_hits: int = 0
    dev_expression_resolves: int = 0
    dev_expression_no_genome: int = 0
    dev_expression_key_counts: Counter[str] = field(default_factory=Counter)
    dev_expression_tick_key_counts: Counter[str] = field(default_factory=Counter)
    dev_expression_generation_key_counts: Counter[str] = field(default_factory=Counter)
    dev_expression_age_key_counts: Counter[str] = field(default_factory=Counter)
    dev_cache_clears: int = 0
    dev_cache_clear_nonempty: int = 0
    dev_cache_clear_entries_total: int = 0
    phenotype_calls: int = 0
    phenotype_key_counts: Counter[str] = field(default_factory=Counter)
    phenotype_generation_key_counts: Counter[str] = field(default_factory=Counter)
    phenotype_source_counts: Counter[str] = field(default_factory=Counter)
    decomp_seconds: Counter[str] = field(default_factory=Counter)
    decomp_counts: Counter[str] = field(default_factory=Counter)
    native_budget_cost_backend_counts: Counter[str] = field(default_factory=Counter)
    native_budget_cost_hit_count: int = 0
    native_budget_cost_fallback_count: int = 0
    native_budget_cost_mismatch_count: int = 0
    native_budget_cost_pack_rows: int = 0
    phenotype_seconds: Counter[str] = field(default_factory=Counter)
    phenotype_counts: Counter[str] = field(default_factory=Counter)
    phenotype_inactive_counts: Counter[str] = field(default_factory=Counter)
    phenotype_inactive_default_counts: Counter[str] = field(default_factory=Counter)
    direct_cost_scalar_hit_count: int = 0
    direct_cost_scalar_ledger_fallback_count: int = 0
    direct_cost_scalar_guard_check_count: int = 0
    direct_cost_scalar_guard_mismatch_count: int = 0
    direct_cost_scalar_max_abs_diff: float = 0.0
    direct_cost_scalar_skipped_reason_counts: Counter[str] = field(default_factory=Counter)
    cost_charge_surface_hit_count: int = 0
    cost_charge_surface_fallback_count: int = 0
    cost_charge_surface_guard_check_count: int = 0
    cost_charge_surface_guard_mismatch_count: int = 0
    cost_charge_surface_remaining_full_phenotype_count: int = 0
    cost_charge_surface_max_abs_diff: float = 0.0
    cost_charge_surface_skipped_reason_counts: Counter[str] = field(default_factory=Counter)
    canonicalize_payload_counts: Counter[str] = field(default_factory=Counter)
    canonicalize_payload_bytes: Counter[str] = field(default_factory=Counter)
    developmental_residual_seconds: Counter[str] = field(default_factory=Counter)
    developmental_residual_counts: Counter[str] = field(default_factory=Counter)
    developmental_residual_source_counts: Counter[str] = field(default_factory=Counter)
    developmental_residual_program_keys: Counter[str] = field(default_factory=Counter)
    developmental_residual_program_age_keys: Counter[str] = field(default_factory=Counter)
    developmental_residual_genome_age_keys: Counter[str] = field(default_factory=Counter)
    developmental_residual_module_keys: Counter[str] = field(default_factory=Counter)
    developmental_residual_stage_counts: Counter[str] = field(default_factory=Counter)
    developmental_cache_equality_counts: Counter[str] = field(default_factory=Counter)
    developmental_cache_equality_seconds: Counter[str] = field(default_factory=Counter)
    developmental_cache_equality_fallback_reasons: Counter[str] = field(default_factory=Counter)
    developmental_cache_equality_mismatch_fields: Counter[str] = field(default_factory=Counter)
    developmental_cache_equality_lock_signatures: Counter[str] = field(default_factory=Counter)
    developmental_cache_equality_max_numeric_diff: float = 0.0
    developmental_effective_projection_counts: Counter[str] = field(default_factory=Counter)
    developmental_effective_projection_seconds: Counter[str] = field(default_factory=Counter)
    developmental_effective_projection_fallback_reasons: Counter[str] = field(default_factory=Counter)
    developmental_effective_projection_mismatch_fields: Counter[str] = field(default_factory=Counter)
    developmental_effective_projection_runner_metadata_fields: Counter[str] = field(default_factory=Counter)
    developmental_effective_projection_lock_signatures: Counter[str] = field(default_factory=Counter)
    developmental_effective_projection_program_signatures: Counter[str] = field(default_factory=Counter)
    developmental_effective_projection_max_numeric_diff: float = 0.0
    event_count: int = 0

    @classmethod
    def maybe_create(cls, runner: Any) -> "EnvHotpathExpressionDiagnostics | None":
        sim_cfg = getattr(runner, "sim_cfg", None)
        expression_enabled = bool(getattr(sim_cfg, "env_hotpath_expression_diagnostics_enable", False))
        decomposition_enabled = bool(getattr(sim_cfg, "env_hotpath_apply_decomposition_enable", False))
        phenotype_decomposition_enabled = bool(getattr(sim_cfg, "env_hotpath_phenotype_decomposition_enable", False))
        developmental_residual_enabled = bool(
            getattr(sim_cfg, "env_hotpath_developmental_residual_diagnostics_enable", False)
        )
        developmental_cache_equality_enabled = bool(
            getattr(sim_cfg, "env_hotpath_developmental_cache_equality_diagnostics_enable", False)
        )
        developmental_effective_projection_enabled = bool(
            getattr(sim_cfg, "env_hotpath_developmental_effective_projection_diagnostics_enable", False)
        )
        if not (
            expression_enabled
            or decomposition_enabled
            or phenotype_decomposition_enabled
            or developmental_residual_enabled
            or developmental_cache_equality_enabled
            or developmental_effective_projection_enabled
        ):
            return None
        return cls(
            run_dir=str(getattr(runner, "run_dir", "")),
            schema_version=int(getattr(sim_cfg, "env_hotpath_expression_diagnostics_schema_version", 1) or 1),
            decomposition_schema_version=int(getattr(sim_cfg, "env_hotpath_apply_decomposition_schema_version", 1) or 1),
            phenotype_decomposition_schema_version=int(
                getattr(sim_cfg, "env_hotpath_phenotype_decomposition_schema_version", 1) or 1
            ),
            developmental_residual_schema_version=int(
                getattr(sim_cfg, "env_hotpath_developmental_residual_diagnostics_schema_version", 1) or 1
            ),
            developmental_cache_equality_schema_version=int(
                getattr(sim_cfg, "env_hotpath_developmental_cache_equality_schema_version", 1) or 1
            ),
            developmental_effective_projection_schema_version=int(
                getattr(sim_cfg, "env_hotpath_developmental_effective_projection_schema_version", 1) or 1
            ),
            decomposition_enabled=bool(decomposition_enabled),
            phenotype_decomposition_enabled=bool(phenotype_decomposition_enabled),
            developmental_residual_enabled=bool(developmental_residual_enabled),
            developmental_cache_equality_enabled=bool(developmental_cache_equality_enabled),
            developmental_effective_projection_enabled=bool(developmental_effective_projection_enabled),
            write_events=bool(getattr(sim_cfg, "env_hotpath_expression_diagnostics_write_events", False)),
        )

    def record_dev_expression_request(
        self,
        *,
        agent_id: int,
        generation: int,
        tick: int,
        age_steps: int,
        cache_hit: bool,
        has_genome: bool,
    ) -> None:
        self.dev_expression_requests += 1
        if bool(cache_hit):
            self.dev_expression_cache_hits += 1
        else:
            self.dev_expression_resolves += 1
        if not bool(has_genome):
            self.dev_expression_no_genome += 1
        key = _stable_key(agent_id, generation, tick, age_steps)
        self.dev_expression_key_counts[key] += 1
        self.dev_expression_tick_key_counts[_stable_key(agent_id, generation, tick)] += 1
        self.dev_expression_generation_key_counts[_stable_key(agent_id, generation)] += 1
        self.dev_expression_age_key_counts[_stable_key(agent_id, age_steps)] += 1
        self._maybe_record_event(
            {
                "event": "dev_expression_request",
                "agent_id": int(agent_id),
                "generation": int(generation),
                "tick": int(tick),
                "age_steps": int(age_steps),
                "cache_hit": bool(cache_hit),
                "has_genome": bool(has_genome),
            }
        )

    def record_dev_cache_clear(self, *, generation: int, tick: int, entries: int) -> None:
        self.dev_cache_clears += 1
        n = max(0, int(entries))
        self.dev_cache_clear_entries_total += n
        if n > 0:
            self.dev_cache_clear_nonempty += 1
        self._maybe_record_event(
            {
                "event": "dev_cache_clear",
                "generation": int(generation),
                "tick": int(tick),
                "entries": int(n),
            }
        )

    def record_phenotype_call(
        self,
        *,
        agent_id: int,
        generation: int,
        tick: int,
        age_steps: int,
        source: str,
    ) -> None:
        self.phenotype_calls += 1
        self.phenotype_key_counts[_stable_key(agent_id, generation, tick, age_steps)] += 1
        self.phenotype_generation_key_counts[_stable_key(agent_id, generation)] += 1
        self.phenotype_source_counts[str(source or "unknown")] += 1
        self._maybe_record_event(
            {
                "event": "phenotype_call",
                "agent_id": int(agent_id),
                "generation": int(generation),
                "tick": int(tick),
                "age_steps": int(age_steps),
                "source": str(source or "unknown"),
            }
        )

    def record_decomposition_bucket(self, name: str, seconds: float, *, count: int = 1) -> None:
        if not bool(self.decomposition_enabled):
            return
        key = str(name or "unknown")
        self.decomp_seconds[key] += max(0.0, float(seconds))
        self.decomp_counts[key] += max(0, int(count))

    def record_phenotype_decomposition_bucket(self, name: str, seconds: float, *, count: int = 1) -> None:
        if not bool(self.phenotype_decomposition_enabled):
            return
        key = str(name or "unknown")
        self.phenotype_seconds[key] += max(0.0, float(seconds))
        self.phenotype_counts[key] += max(0, int(count))

    def record_phenotype_inactive_path(self, name: str, *, count: int = 1) -> None:
        if not bool(self.phenotype_decomposition_enabled):
            return
        key = str(name or "unknown")
        self.phenotype_inactive_counts[key] += max(0, int(count))

    def record_native_budget_cost_aggregation(self, info: dict[str, Any]) -> None:
        if not bool(self.decomposition_enabled):
            return
        backend = str(info.get("backend", "unknown") or "unknown")
        rows = int(info.get("pack_rows", 0) or 0)
        self.native_budget_cost_backend_counts[backend] += max(0, rows)
        self.native_budget_cost_hit_count += max(0, int(info.get("hit_count", 0) or 0))
        self.native_budget_cost_fallback_count += max(0, int(info.get("fallback_count", 0) or 0))
        self.native_budget_cost_mismatch_count += max(0, int(info.get("mismatch_count", 0) or 0))
        self.native_budget_cost_pack_rows += max(0, rows)
        self.record_decomposition_bucket(
            "env_apply_native_budget_cost_input_packing_total",
            float(info.get("input_packing_s", 0.0) or 0.0),
        )
        self.record_decomposition_bucket(
            "env_apply_native_budget_cost_kernel_total",
            float(info.get("native_kernel_s", 0.0) or 0.0),
        )

    def record_phenotype_inactive_default(self, name: str, *, count: int = 1) -> None:
        if not bool(self.phenotype_decomposition_enabled):
            return
        key = str(name or "unknown")
        self.phenotype_inactive_default_counts[key] += max(0, int(count))

    def record_direct_cost_scalar(self, info: dict[str, Any]) -> None:
        if not bool(self.decomposition_enabled):
            return
        self.direct_cost_scalar_hit_count += max(0, int(info.get("hit_count", 0) or 0))
        self.direct_cost_scalar_ledger_fallback_count += max(0, int(info.get("ledger_fallback_count", 0) or 0))
        self.direct_cost_scalar_guard_check_count += max(0, int(info.get("guard_check_count", 0) or 0))
        self.direct_cost_scalar_guard_mismatch_count += max(0, int(info.get("guard_mismatch_count", 0) or 0))
        self.direct_cost_scalar_max_abs_diff = max(
            float(self.direct_cost_scalar_max_abs_diff),
            float(info.get("max_abs_diff", 0.0) or 0.0),
        )
        for reason, count in dict(info.get("skipped_reason_counts", {}) or {}).items():
            self.direct_cost_scalar_skipped_reason_counts[str(reason)] += max(0, int(count or 0))
        self.record_decomposition_bucket(
            "env_apply_direct_cost_scalar_total",
            float(info.get("direct_scalar_s", 0.0) or 0.0),
            count=max(0, int(info.get("hit_count", 0) or 0)),
        )
        self.record_decomposition_bucket(
            "env_apply_direct_cost_scalar_guard_ledger_total",
            float(info.get("guard_ledger_s", 0.0) or 0.0),
            count=max(0, int(info.get("guard_check_count", 0) or 0)),
        )
        self.record_decomposition_bucket(
            "env_apply_direct_cost_scalar_fallback_ledger_total",
            float(info.get("fallback_ledger_s", 0.0) or 0.0),
            count=max(0, int(info.get("ledger_fallback_count", 0) or 0)),
        )

    def record_cost_charge_surface(self, info: dict[str, Any]) -> None:
        if not bool(self.decomposition_enabled):
            return
        self.cost_charge_surface_hit_count += max(0, int(info.get("hit_count", 0) or 0))
        self.cost_charge_surface_fallback_count += max(0, int(info.get("fallback_count", 0) or 0))
        self.cost_charge_surface_guard_check_count += max(0, int(info.get("guard_check_count", 0) or 0))
        self.cost_charge_surface_guard_mismatch_count += max(0, int(info.get("guard_mismatch_count", 0) or 0))
        self.cost_charge_surface_remaining_full_phenotype_count += max(
            0,
            int(info.get("remaining_full_phenotype_count", 0) or 0),
        )
        self.cost_charge_surface_max_abs_diff = max(
            float(self.cost_charge_surface_max_abs_diff),
            float(info.get("max_abs_diff", 0.0) or 0.0),
        )
        for reason, count in dict(info.get("skipped_reason_counts", {}) or {}).items():
            self.cost_charge_surface_skipped_reason_counts[str(reason)] += max(0, int(count or 0))
        self.record_decomposition_bucket(
            "env_apply_cost_charge_surface_total",
            float(info.get("surface_s", 0.0) or 0.0),
            count=max(0, int(info.get("hit_count", 0) or 0)),
        )
        self.record_decomposition_bucket(
            "env_apply_cost_charge_surface_scalar_total",
            float(info.get("scalar_s", 0.0) or 0.0),
            count=max(0, int(info.get("hit_count", 0) or 0)),
        )
        self.record_decomposition_bucket(
            "env_apply_cost_charge_surface_guard_reference_total",
            float(info.get("guard_reference_s", 0.0) or 0.0),
            count=max(0, int(info.get("guard_check_count", 0) or 0)),
        )
        self.record_decomposition_bucket(
            "env_apply_cost_charge_surface_fallback_total",
            float(info.get("fallback_s", 0.0) or 0.0),
            count=max(0, int(info.get("fallback_count", 0) or 0)),
        )

    def record_canonicalize_call(self, *, payload_key: str, payload_bytes: int, seconds: float) -> None:
        if not (bool(self.decomposition_enabled) or bool(self.developmental_residual_enabled)):
            return
        key = str(payload_key)
        self.canonicalize_payload_counts[key] += 1
        self.canonicalize_payload_bytes[key] = max(int(self.canonicalize_payload_bytes.get(key, 0)), int(payload_bytes))
        if bool(self.decomposition_enabled):
            self.record_decomposition_bucket("canonicalize", float(seconds), count=1)

    def record_developmental_residual_bucket(self, name: str, seconds: float = 0.0, *, count: int = 1) -> None:
        if not bool(self.developmental_residual_enabled):
            return
        key = str(name or "unknown")
        self.developmental_residual_seconds[key] += max(0.0, float(seconds))
        self.developmental_residual_counts[key] += max(0, int(count))

    def record_developmental_effective_lookup(
        self,
        *,
        source: str,
        agent_id: int,
        age_steps: int,
        seconds: float,
    ) -> None:
        if not bool(self.developmental_residual_enabled):
            return
        src = str(source or "unknown")
        self.developmental_residual_source_counts[src] += 1
        self.record_developmental_residual_bucket("developmental_effective_lookup_total", seconds, count=1)
        self.developmental_residual_genome_age_keys[_stable_key("agent", int(agent_id), int(age_steps))] += 1

    def record_developmental_stage_resolution(
        self,
        *,
        program_key: str,
        age_steps: int,
        stage_id: str,
        seconds: float,
    ) -> None:
        if not bool(self.developmental_residual_enabled):
            return
        pkey = self._hash_key(program_key)
        self.record_developmental_residual_bucket("stage_resolution_total", seconds, count=1)
        self.developmental_residual_program_keys[pkey] += 1
        self.developmental_residual_program_age_keys[_stable_key(pkey, int(age_steps))] += 1
        self.developmental_residual_stage_counts[str(stage_id or "default")] += 1

    def record_developmental_module_active(
        self,
        *,
        program_key: str,
        age_steps: int,
        module_id: str,
        stage_id: str,
        seconds: float,
    ) -> None:
        if not bool(self.developmental_residual_enabled):
            return
        pkey = self._hash_key(program_key)
        mid = str(module_id or "")
        self.record_developmental_residual_bucket("module_active_total", seconds, count=1)
        self.developmental_residual_program_keys[pkey] += 1
        self.developmental_residual_program_age_keys[_stable_key(pkey, int(age_steps))] += 1
        self.developmental_residual_module_keys[_stable_key(pkey, int(age_steps), mid)] += 1
        self.developmental_residual_stage_counts[str(stage_id or "default")] += 1

    def record_developmental_lock_application(
        self,
        *,
        program_key: str,
        stage_id: str,
        prev_stage_id: str,
        module_count: int,
        seconds: float,
    ) -> None:
        if not bool(self.developmental_residual_enabled):
            return
        pkey = self._hash_key(program_key)
        self.record_developmental_residual_bucket("lock_application_total", seconds, count=1)
        self.developmental_residual_program_keys[pkey] += 1
        self.developmental_residual_stage_counts[str(stage_id or "default")] += 1
        self.developmental_residual_counts[_stable_key("lock_modules", int(max(0, module_count)))] += 1
        if str(stage_id or "default") != str(prev_stage_id or "default"):
            self.developmental_residual_counts["lock_stage_transition_inputs"] += 1

    def record_developmental_genome_resolve(
        self,
        *,
        genome_key: str,
        program_key: str,
        age_steps: int,
        stage_id: str,
        module_count: int,
        seconds: float,
    ) -> None:
        if not bool(self.developmental_residual_enabled):
            return
        gkey = self._hash_key(genome_key)
        pkey = self._hash_key(program_key)
        self.record_developmental_residual_bucket("genome_resolve_total", seconds, count=1)
        self.developmental_residual_program_keys[pkey] += 1
        self.developmental_residual_program_age_keys[_stable_key(pkey, int(age_steps))] += 1
        self.developmental_residual_genome_age_keys[_stable_key(gkey, int(age_steps))] += 1
        self.developmental_residual_stage_counts[str(stage_id or "default")] += 1
        self.developmental_residual_counts[_stable_key("genome_module_count", int(max(0, module_count)))] += 1

    def record_developmental_cache_candidate(
        self,
        *,
        agent_id: int,
        generation: int,
        tick: int,
        age_steps: int,
        lock_state_sha256: str,
    ) -> None:
        if not (
            bool(self.developmental_cache_equality_enabled)
            or bool(self.developmental_effective_projection_enabled)
        ):
            return
        sig = str(lock_state_sha256 or "empty")
        if bool(self.developmental_cache_equality_enabled):
            self.developmental_cache_equality_counts["runner_cache_candidates_stored"] += 1
            self.developmental_cache_equality_lock_signatures[sig] += 1
        if bool(self.developmental_effective_projection_enabled):
            self.developmental_effective_projection_counts["runner_cache_candidates_stored"] += 1
            self.developmental_effective_projection_lock_signatures[sig] += 1
        self._maybe_record_event(
            {
                "event": "dev_cache_equality_candidate",
                "agent_id": int(agent_id),
                "generation": int(generation),
                "tick": int(tick),
                "age_steps": int(age_steps),
                "lock_state_sha256": sig,
            }
        )

    def record_developmental_cache_equality_check(
        self,
        *,
        candidate: Any,
        direct: dict[str, Any],
        agent_id: int,
        age_steps: int,
        seconds: float,
    ) -> None:
        if not bool(self.developmental_cache_equality_enabled):
            return
        self.developmental_cache_equality_counts["candidate_reuse_checks"] += 1
        self.developmental_cache_equality_seconds["equality_check_total"] += max(0.0, float(seconds))
        if not isinstance(candidate, dict):
            self.developmental_cache_equality_fallback_reasons["candidate_missing"] += 1
            return
        expr = candidate.get("expression")
        if not isinstance(expr, dict):
            self.developmental_cache_equality_fallback_reasons["candidate_expression_missing"] += 1
            return
        cand_agent = int(candidate.get("agent_id", -1) or -1)
        cand_age = int(candidate.get("age_steps", -1) or -1)
        if cand_agent != int(agent_id):
            self.developmental_cache_equality_fallback_reasons["agent_id_mismatch"] += 1
            return
        if cand_age != int(age_steps):
            self.developmental_cache_equality_fallback_reasons["age_steps_mismatch"] += 1
            return

        self.developmental_cache_equality_counts["candidate_available"] += 1
        self.developmental_cache_equality_counts["same_agent_age_hits"] += 1
        if "generation" in candidate and "tick" in candidate:
            self.developmental_cache_equality_counts["same_agent_generation_tick_age_metadata_hits"] += 1
        sig = str(candidate.get("lock_state_sha256", "") or "empty")
        self.developmental_cache_equality_lock_signatures[sig] += 1

        full_match, full_fields, full_diff = self._compare_jsonish(expr, direct)
        if full_match:
            self.developmental_cache_equality_counts["full_exact_matches"] += 1
        else:
            self.developmental_cache_equality_counts["full_mismatches"] += 1
            for field_name in full_fields:
                self.developmental_cache_equality_mismatch_fields[f"full:{field_name}"] += 1
        self.developmental_cache_equality_max_numeric_diff = max(
            float(self.developmental_cache_equality_max_numeric_diff),
            float(full_diff),
        )

        cand_effective = expr.get("effective", {})
        direct_effective = direct.get("effective", {})
        projection_match, projection_fields, projection_diff = self._compare_jsonish(cand_effective, direct_effective)
        if projection_match:
            self.developmental_cache_equality_counts["effective_projection_matches"] += 1
        else:
            self.developmental_cache_equality_counts["effective_projection_mismatches"] += 1
            for field_name in projection_fields:
                self.developmental_cache_equality_mismatch_fields[f"effective:{field_name}"] += 1
        self.developmental_cache_equality_max_numeric_diff = max(
            float(self.developmental_cache_equality_max_numeric_diff),
            float(projection_diff),
        )

    def record_developmental_effective_projection_check(
        self,
        *,
        candidate: Any,
        direct: dict[str, Any],
        direct_projection: dict[str, Any],
        agent_id: int,
        age_steps: int,
        seconds: float,
        projection_seconds: float = 0.0,
    ) -> None:
        if not bool(self.developmental_effective_projection_enabled):
            return
        self.developmental_effective_projection_counts["candidate_reuse_checks"] += 1
        self.developmental_effective_projection_seconds["projection_equality_check_total"] += max(0.0, float(seconds))
        self.developmental_effective_projection_seconds["projection_construction_total"] += max(
            0.0,
            float(projection_seconds),
        )
        if not isinstance(candidate, dict):
            self.developmental_effective_projection_fallback_reasons["candidate_missing"] += 1
            return
        expr = candidate.get("expression")
        if not isinstance(expr, dict):
            self.developmental_effective_projection_fallback_reasons["candidate_expression_missing"] += 1
            return
        cand_agent = int(candidate.get("agent_id", -1) or -1)
        cand_age = int(candidate.get("age_steps", -1) or -1)
        if cand_agent != int(agent_id):
            self.developmental_effective_projection_fallback_reasons["agent_id_mismatch"] += 1
            return
        if cand_age != int(age_steps):
            self.developmental_effective_projection_fallback_reasons["age_steps_mismatch"] += 1
            return

        self.developmental_effective_projection_counts["candidate_available"] += 1
        self.developmental_effective_projection_counts["same_agent_age_hits"] += 1
        if "generation" in candidate and "tick" in candidate:
            self.developmental_effective_projection_counts["same_agent_generation_tick_age_metadata_hits"] += 1
        sig = str(candidate.get("lock_state_sha256", "") or "empty")
        self.developmental_effective_projection_lock_signatures[sig] += 1
        program_sig = str(candidate.get("program_sha256", "") or "")
        if program_sig:
            self.developmental_effective_projection_program_signatures[program_sig] += 1
        for field_name in RUNNER_ONLY_EXPRESSION_METADATA_FIELDS:
            if field_name in expr and field_name not in direct:
                self.developmental_effective_projection_runner_metadata_fields[str(field_name)] += 1

        candidate_projection = self._developmental_effective_projection(expr)
        expected_projection = self._developmental_effective_projection(direct_projection)
        projection_match, projection_fields, projection_diff = self._compare_jsonish(
            candidate_projection,
            expected_projection,
        )
        if projection_match:
            self.developmental_effective_projection_counts["projection_exact_matches"] += 1
        else:
            self.developmental_effective_projection_counts["projection_mismatches"] += 1
            for field_name in projection_fields:
                self.developmental_effective_projection_mismatch_fields[str(field_name)] += 1
        self.developmental_effective_projection_max_numeric_diff = max(
            float(self.developmental_effective_projection_max_numeric_diff),
            float(projection_diff),
        )

    def write_summary(self) -> str:
        metrics_dir = os.path.join(str(self.run_dir), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        out_path = os.path.join(metrics_dir, "env_hotpath_expression_summary.json")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "config_schema_version": int(self.schema_version),
            "wall_s": float(time.perf_counter() - self.started_at_monotonic),
            "dev_expression": {
                "requests": int(self.dev_expression_requests),
                "cache_hits": int(self.dev_expression_cache_hits),
                "resolves": int(self.dev_expression_resolves),
                "no_genome": int(self.dev_expression_no_genome),
                "cache_hit_fraction": (
                    float(self.dev_expression_cache_hits) / float(self.dev_expression_requests)
                    if self.dev_expression_requests
                    else 0.0
                ),
                "repeated_same_agent_tick_age_keys": int(sum(1 for count in self.dev_expression_key_counts.values() if count > 1)),
                "repeated_same_agent_tick_keys": int(sum(1 for count in self.dev_expression_tick_key_counts.values() if count > 1)),
                "repeated_same_agent_generation_keys": int(sum(1 for count in self.dev_expression_generation_key_counts.values() if count > 1)),
                "top_repeated_tick_age_keys": self._top_repeated(self.dev_expression_key_counts),
                "top_repeated_tick_keys": self._top_repeated(self.dev_expression_tick_key_counts),
                "top_repeated_generation_keys": self._top_repeated(self.dev_expression_generation_key_counts),
            },
            "phenotype": {
                "calls": int(self.phenotype_calls),
                "repeated_same_agent_tick_age_keys": int(sum(1 for count in self.phenotype_key_counts.values() if count > 1)),
                "repeated_same_agent_generation_keys": int(sum(1 for count in self.phenotype_generation_key_counts.values() if count > 1)),
                "source_counts": dict(sorted(self.phenotype_source_counts.items())),
                "top_repeated_tick_age_keys": self._top_repeated(self.phenotype_key_counts),
                "top_repeated_generation_keys": self._top_repeated(self.phenotype_generation_key_counts),
            },
            "invalidation": {
                "dev_cache_clears": int(self.dev_cache_clears),
                "dev_cache_clear_nonempty": int(self.dev_cache_clear_nonempty),
                "dev_cache_clear_entries_total": int(self.dev_cache_clear_entries_total),
                "mean_entries_per_clear": (
                    float(self.dev_cache_clear_entries_total) / float(self.dev_cache_clears)
                    if self.dev_cache_clears
                    else 0.0
                ),
            },
            "events_written": int(self.event_count if self.write_events else 0),
            "candidate_cache_boundaries": {
                "tick_cache": "agent_id,generation,tick,age_steps",
                "generation_cache": "agent_id,generation",
                "age_cache": "agent_id,age_steps",
            },
        }
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, out_path)
        return out_path

    def write_decomposition_summary(self) -> str:
        metrics_dir = os.path.join(str(self.run_dir), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        out_path = os.path.join(metrics_dir, "env_apply_decomposition_summary.json")
        seconds = dict(sorted((str(k), float(v)) for k, v in self.decomp_seconds.items()))
        counts = dict(sorted((str(k), int(v)) for k, v in self.decomp_counts.items()))
        env_apply_total = float(seconds.get("env_apply_total", 0.0))
        known_non_other = 0.0
        for key, value in seconds.items():
            if key not in {"env_apply_total", "env_apply_other_total"}:
                known_non_other += float(value)
        other_total = max(0.0, env_apply_total - known_non_other) if env_apply_total > 0.0 else 0.0
        canonical_total = int(sum(self.canonicalize_payload_counts.values()))
        canonical_unique = int(len(self.canonicalize_payload_counts))
        payload = {
            "schema_version": DECOMPOSITION_SCHEMA_VERSION,
            "config_schema_version": int(self.decomposition_schema_version),
            "wall_s": float(time.perf_counter() - self.started_at_monotonic),
            "seconds": seconds,
            "counts": counts,
            "derived": {
                "env_apply_other_total": float(other_total),
                "canonicalize_total_calls": int(canonical_total),
                "canonicalize_unique_payloads": int(canonical_unique),
                "canonicalize_repeat_fraction": (
                    float(canonical_total - canonical_unique) / float(canonical_total)
                    if canonical_total > 0
                    else 0.0
                ),
                "canonicalize_seconds": float(seconds.get("canonicalize", 0.0)),
            },
            "canonicalize": {
                "total_calls": int(canonical_total),
                "unique_payloads": int(canonical_unique),
                "top_payloads": self._top_payloads(self.canonicalize_payload_counts, self.canonicalize_payload_bytes),
            },
            "native_cognitive_budget_cost_aggregation": {
                "backend_counts": dict(sorted(self.native_budget_cost_backend_counts.items())),
                "hit_count": int(self.native_budget_cost_hit_count),
                "fallback_count": int(self.native_budget_cost_fallback_count),
                "mismatch_count": int(self.native_budget_cost_mismatch_count),
                "pack_rows": int(self.native_budget_cost_pack_rows),
            },
            "direct_cost_scalar": {
                "hit_count": int(self.direct_cost_scalar_hit_count),
                "ledger_fallback_count": int(self.direct_cost_scalar_ledger_fallback_count),
                "guard_check_count": int(self.direct_cost_scalar_guard_check_count),
                "guard_mismatch_count": int(self.direct_cost_scalar_guard_mismatch_count),
                "max_abs_diff": float(self.direct_cost_scalar_max_abs_diff),
                "skipped_reason_counts": dict(sorted(self.direct_cost_scalar_skipped_reason_counts.items())),
            },
            "cost_charge_surface": {
                "hit_count": int(self.cost_charge_surface_hit_count),
                "fallback_count": int(self.cost_charge_surface_fallback_count),
                "guard_check_count": int(self.cost_charge_surface_guard_check_count),
                "guard_mismatch_count": int(self.cost_charge_surface_guard_mismatch_count),
                "remaining_full_phenotype_count": int(self.cost_charge_surface_remaining_full_phenotype_count),
                "max_abs_diff": float(self.cost_charge_surface_max_abs_diff),
                "skipped_reason_counts": dict(sorted(self.cost_charge_surface_skipped_reason_counts.items())),
            },
        }
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, out_path)
        return out_path

    def write_phenotype_decomposition_summary(self) -> str:
        metrics_dir = os.path.join(str(self.run_dir), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        out_path = os.path.join(metrics_dir, "envhot3_phenotype_decomposition_summary.json")
        seconds = dict(sorted((str(k), float(v)) for k, v in self.phenotype_seconds.items()))
        counts = dict(sorted((str(k), int(v)) for k, v in self.phenotype_counts.items()))
        phenotype_total = float(seconds.get("phenotype_total", 0.0))
        known_non_other = 0.0
        for key, value in seconds.items():
            if key not in {"phenotype_total", "phenotype_other_total"}:
                known_non_other += float(value)
        other_total = max(0.0, phenotype_total - known_non_other) if phenotype_total > 0.0 else 0.0
        payload = {
            "schema_version": PHENOTYPE_DECOMPOSITION_SCHEMA_VERSION,
            "config_schema_version": int(self.phenotype_decomposition_schema_version),
            "wall_s": float(time.perf_counter() - self.started_at_monotonic),
            "seconds": seconds,
            "counts": counts,
            "inactive_paths": dict(sorted((str(k), int(v)) for k, v in self.phenotype_inactive_counts.items())),
            "inactive_default_strategy": dict(
                sorted((str(k), int(v)) for k, v in self.phenotype_inactive_default_counts.items())
            ),
            "derived": {
                "phenotype_other_total": float(other_total),
            },
            "notes": {
                "chemistry_object_total": "tracked in env_apply_decomposition_summary.json as env_apply_chemistry_object_total",
                "maintenance_object_total": "not constructed inside express_cognitive_phenotype; ledger maintenance fields are scalar assembly",
            },
        }
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, out_path)
        return out_path

    def write_developmental_residual_summary(self) -> str:
        metrics_dir = os.path.join(str(self.run_dir), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        out_path = os.path.join(metrics_dir, "envhot7_developmental_residual_summary.json")
        seconds = dict(sorted((str(k), float(v)) for k, v in self.developmental_residual_seconds.items()))
        counts = dict(sorted((str(k), int(v)) for k, v in self.developmental_residual_counts.items()))
        canonical_total = int(sum(self.canonicalize_payload_counts.values()))
        canonical_unique = int(len(self.canonicalize_payload_counts))
        genome_keys = self.developmental_residual_genome_age_keys
        program_age_keys = self.developmental_residual_program_age_keys
        module_keys = self.developmental_residual_module_keys
        payload = {
            "schema_version": DEVELOPMENTAL_RESIDUAL_SCHEMA_VERSION,
            "config_schema_version": int(self.developmental_residual_schema_version),
            "wall_s": float(time.perf_counter() - self.started_at_monotonic),
            "seconds": seconds,
            "counts": counts,
            "source_counts": dict(sorted((str(k), int(v)) for k, v in self.developmental_residual_source_counts.items())),
            "stage_counts": dict(sorted((str(k), int(v)) for k, v in self.developmental_residual_stage_counts.items())),
            "canonicalize": {
                "total_calls": int(canonical_total),
                "unique_payloads": int(canonical_unique),
                "repeat_fraction": (
                    float(canonical_total - canonical_unique) / float(canonical_total)
                    if canonical_total > 0
                    else 0.0
                ),
                "top_payloads": self._top_payloads(self.canonicalize_payload_counts, self.canonicalize_payload_bytes),
            },
            "derived": {
                "unique_program_payload_keys": int(len(self.developmental_residual_program_keys)),
                "unique_program_age_keys": int(len(program_age_keys)),
                "unique_genome_age_keys": int(len(genome_keys)),
                "unique_module_keys": int(len(module_keys)),
                "repeated_program_age_keys": int(sum(1 for count in program_age_keys.values() if int(count) > 1)),
                "repeated_genome_age_keys": int(sum(1 for count in genome_keys.values() if int(count) > 1)),
                "repeated_module_keys": int(sum(1 for count in module_keys.values() if int(count) > 1)),
                "canonicalize_seconds": float(self.decomp_seconds.get("canonicalize", 0.0)),
            },
            "top_repeated": {
                "program_age_keys": self._top_repeated(program_age_keys),
                "genome_age_keys": self._top_repeated(genome_keys),
                "module_keys": self._top_repeated(module_keys),
            },
            "cache_boundaries_under_audit": {
                "runner_cache": "agent_id,generation,tick,age_steps",
                "candidate_genome_age": "genome_payload_or_object_identity,age_steps",
                "candidate_program_age": "developmental_program_payload,age_steps",
                "candidate_module": "developmental_program_payload,age_steps,module_id",
            },
        }
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, out_path)
        return out_path

    def write_developmental_cache_equality_summary(self) -> str:
        metrics_dir = os.path.join(str(self.run_dir), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        out_path = os.path.join(metrics_dir, "envhot8_developmental_cache_equality_summary.json")
        checks = int(self.developmental_cache_equality_counts.get("candidate_reuse_checks", 0))
        available = int(self.developmental_cache_equality_counts.get("candidate_available", 0))
        full_matches = int(self.developmental_cache_equality_counts.get("full_exact_matches", 0))
        effective_matches = int(self.developmental_cache_equality_counts.get("effective_projection_matches", 0))
        payload = {
            "schema_version": DEVELOPMENTAL_CACHE_EQUALITY_SCHEMA_VERSION,
            "config_schema_version": int(self.developmental_cache_equality_schema_version),
            "wall_s": float(time.perf_counter() - self.started_at_monotonic),
            "counts": dict(sorted((str(k), int(v)) for k, v in self.developmental_cache_equality_counts.items())),
            "seconds": dict(sorted((str(k), float(v)) for k, v in self.developmental_cache_equality_seconds.items())),
            "fallback_reasons": dict(
                sorted((str(k), int(v)) for k, v in self.developmental_cache_equality_fallback_reasons.items())
            ),
            "mismatch_fields": dict(
                sorted((str(k), int(v)) for k, v in self.developmental_cache_equality_mismatch_fields.items())
            ),
            "lock_signatures": {
                "unique": int(len(self.developmental_cache_equality_lock_signatures)),
                "top": self._top_repeated(self.developmental_cache_equality_lock_signatures),
            },
            "derived": {
                "candidate_availability_fraction": (float(available) / float(checks) if checks else 0.0),
                "full_exact_match_fraction": (float(full_matches) / float(available) if available else 0.0),
                "effective_projection_match_fraction": (
                    float(effective_matches) / float(available) if available else 0.0
                ),
                "max_numeric_diff": float(self.developmental_cache_equality_max_numeric_diff),
            },
            "contract_notes": {
                "runner_candidate_shape": "full lock-aware developmental expression plus metadata",
                "phenotype_current_shape": "direct Genome.resolve_developmental_expression output",
                "effective_projection": "nested effective dictionaries compared for analysis only",
            },
        }
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, out_path)
        return out_path

    def write_developmental_effective_projection_summary(self) -> str:
        metrics_dir = os.path.join(str(self.run_dir), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        out_path = os.path.join(metrics_dir, "envhot9_developmental_effective_projection_summary.json")
        checks = int(self.developmental_effective_projection_counts.get("candidate_reuse_checks", 0))
        available = int(self.developmental_effective_projection_counts.get("candidate_available", 0))
        matches = int(self.developmental_effective_projection_counts.get("projection_exact_matches", 0))
        payload = {
            "schema_version": DEVELOPMENTAL_EFFECTIVE_PROJECTION_SCHEMA_VERSION,
            "config_schema_version": int(self.developmental_effective_projection_schema_version),
            "wall_s": float(time.perf_counter() - self.started_at_monotonic),
            "projection_fields": list(DEVELOPMENTAL_EFFECTIVE_PROJECTION_FIELDS),
            "counts": dict(
                sorted((str(k), int(v)) for k, v in self.developmental_effective_projection_counts.items())
            ),
            "seconds": dict(
                sorted((str(k), float(v)) for k, v in self.developmental_effective_projection_seconds.items())
            ),
            "fallback_reasons": dict(
                sorted((str(k), int(v)) for k, v in self.developmental_effective_projection_fallback_reasons.items())
            ),
            "mismatch_fields": dict(
                sorted((str(k), int(v)) for k, v in self.developmental_effective_projection_mismatch_fields.items())
            ),
            "runner_only_metadata_fields": dict(
                sorted((str(k), int(v)) for k, v in self.developmental_effective_projection_runner_metadata_fields.items())
            ),
            "lock_signatures": {
                "unique": int(len(self.developmental_effective_projection_lock_signatures)),
                "top": self._top_repeated(self.developmental_effective_projection_lock_signatures),
            },
            "program_signatures": {
                "unique": int(len(self.developmental_effective_projection_program_signatures)),
                "top": self._top_repeated(self.developmental_effective_projection_program_signatures),
            },
            "derived": {
                "candidate_availability_fraction": (float(available) / float(checks) if checks else 0.0),
                "projection_exact_match_fraction": (float(matches) / float(available) if available else 0.0),
                "max_numeric_diff": float(self.developmental_effective_projection_max_numeric_diff),
            },
            "contract_notes": {
                "candidate_shape": "runner full lock-aware developmental expression plus metadata",
                "projection_shape": "whitelisted scalar effective values only",
                "runner_metadata": "tracked but excluded from projection equality",
                "behavior": "diagnostic only; no reuse or phenotype return-shape change",
            },
        }
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, out_path)
        return out_path

    @staticmethod
    def _top_repeated(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
        return [
            {"key": str(key), "count": int(count)}
            for key, count in counter.most_common(int(limit))
            if int(count) > 1
        ]

    @staticmethod
    def _top_payloads(counter: Counter[str], byte_counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
        out = []
        for key, count in counter.most_common(int(limit)):
            out.append(
                {
                    "payload_sha256": hashlib.sha256(str(key).encode("utf-8")).hexdigest(),
                    "count": int(count),
                    "payload_bytes": int(byte_counter.get(key, 0) or 0),
                }
            )
        return out

    @staticmethod
    def _hash_key(key: Any) -> str:
        return hashlib.sha256(str(key).encode("utf-8")).hexdigest()

    @staticmethod
    def _developmental_effective_projection(expression: Any) -> dict[str, Any]:
        if not isinstance(expression, dict):
            return {}
        raw = expression.get("effective", expression)
        if not isinstance(raw, dict):
            return {}
        return {
            str(key): raw[key]
            for key in DEVELOPMENTAL_EFFECTIVE_PROJECTION_FIELDS
            if key in raw
        }

    @classmethod
    def _compare_jsonish(cls, left: Any, right: Any) -> tuple[bool, list[str], float]:
        mismatches: list[str] = []
        max_diff = 0.0

        def _walk(a: Any, b: Any, path: str) -> None:
            nonlocal max_diff
            if isinstance(a, dict) and isinstance(b, dict):
                keys = sorted(set(a.keys()) | set(b.keys()), key=lambda key: str(key))
                for key in keys:
                    child = f"{path}.{key}" if path else str(key)
                    if key not in a:
                        mismatches.append(f"{child}:missing_left")
                    elif key not in b:
                        mismatches.append(f"{child}:missing_right")
                    else:
                        _walk(a.get(key), b.get(key), child)
                return
            if isinstance(a, list) and isinstance(b, list):
                if len(a) != len(b):
                    mismatches.append(f"{path}:len")
                    return
                for idx, (av, bv) in enumerate(zip(a, b)):
                    _walk(av, bv, f"{path}[{idx}]")
                return
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
                diff = abs(float(a) - float(b))
                max_diff = max(float(max_diff), float(diff))
                if diff > 0.0:
                    mismatches.append(path or "value")
                return
            if a != b:
                mismatches.append(path or "value")

        _walk(left, right, "")
        return (not mismatches), mismatches[:50], float(max_diff)

    def _maybe_record_event(self, payload: dict[str, Any]) -> None:
        if not bool(self.write_events):
            return
        metrics_dir = os.path.join(str(self.run_dir), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        path = os.path.join(metrics_dir, "env_hotpath_expression_events.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        self.event_count += 1
