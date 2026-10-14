from __future__ import annotations

"""Report-only lineage reproduction funnel diagnostics.

The module writes compact per-generation lineage rows after the normal
generation transition has already run. It does not call RNG and must not change
selection, reproduction, cull/refill, ecology, rewards, or archive behavior.
"""

import csv
import io
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from agents.agent import format_lineage_id, normalize_lineage_stats
from engine.campaigns.artifact_lifecycle import (
    validate_run_artifact_lifecycle_contract_v1,
)
from engine.campaigns.canonical import canonical_sha256
from evolution.structural_variant_identity import global_event_id


SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 3
SCAN_ACTION_EVENT_SCHEMA_VERSION = 4
SCHEMA_NAME = "lineage_reproduction_funnel_v1"
PATH2_ROZ_SCHEDULE_MODES = {
    "static_periodic_compat",
    "periodic_single_zone",
    "multi_zone_round_robin",
    "patch_derived_static",
}
EVENT_OUTPUT_PROFILES = {"full", "path2_roz_compact_v1", "path2_roz_summary_only_v1"}
LRO1_REPRODUCTIVE_OPPORTUNITY_MODE = "local_patch_carrying_capacity_v1"
LRO1_REPRODUCTIVE_OPPORTUNITY_APPLY_STAGE = "pre_attempt_try_prob"
LRO4_OFFSPRING_DISPERSAL_MODE = "crowding_conditioned_radius_v1"
_FUNNEL_SUMMARY_CACHE_BY_EVENTS_PATH: dict[str, dict[str, Any]] = {}
_LIFECYCLE_CONTRACT_CACHE: dict[str, dict[str, Any] | None] = {}
_LIFECYCLE_EVENT_BYTES: dict[str, int] = {}
_LIFECYCLE_SUMMARY_BYTES: dict[str, int] = {}


def _lifecycle_contract_for_run(run_dir: str) -> dict[str, Any] | None:
    root = str(Path(run_dir).expanduser().resolve())
    if root in _LIFECYCLE_CONTRACT_CACHE:
        return _LIFECYCLE_CONTRACT_CACHE[root]
    path = Path(root) / "artifact_lifecycle_contract.json"
    if not path.is_file():
        _LIFECYCLE_CONTRACT_CACHE[root] = None
        return None
    try:
        import json as _json

        raw = _json.loads(path.read_text(encoding="utf-8"))
        contract = validate_run_artifact_lifecycle_contract_v1(raw)
    except Exception as exc:
        raise RuntimeError(
            f"invalid bound artifact lifecycle contract for funnel writer: {path}: {exc}"
        ) from exc
    _LIFECYCLE_CONTRACT_CACHE[root] = contract
    return contract


def _write_lifecycle_violation(
    *,
    run_dir: str,
    contract: Mapping[str, Any],
    artifact_family: str,
    owner_writer: str,
    artifact_path: str,
    code: str,
    attempted_bytes: int,
    limit_bytes: int | None,
) -> None:
    root = Path(run_dir).expanduser().resolve()
    core = {
        "schema_version": "eiel_lifecycle_storage_violation_receipt_v1",
        "run_root": str(root),
        "lifecycle_profile": contract["lifecycle_profile"],
        "lifecycle_contract_sha256": contract["contract_sha256"],
        "artifact_family": artifact_family,
        "owner_writer": owner_writer,
        "artifact_path": artifact_path,
        "code": code,
        "attempted_bytes": int(attempted_bytes),
        "limit_bytes": None if limit_bytes is None else int(limit_bytes),
        "automatic_prune_allowed": False,
        "status": "blocked",
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    path = root / "lifecycle_storage_violation_receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

CSV_FIELDS = [
    "schema_version",
    "generation",
    "lineage_id",
    "alive_count_before_selection",
    "alive_share_before_selection",
    "eligible_parent_count",
    "eligible_parent_share",
    "ineligible_alive_count",
    "ineligibility_reason_counts_json",
    "selected_parent_count",
    "selected_parent_share",
    "unique_selected_parent_count",
    "selected_parent_lineage_entropy",
    "selected_parent_effective_lineage_count",
    "top1_selected_parent_share",
    "top3_selected_parent_share",
    "birth_attempts",
    "successful_births",
    "failed_births",
    "failed_births_by_reason_json",
    "birth_failure_due_to_no_local_space",
    "offspring_produced_count",
    "offspring_share",
    "successful_birth_lineage_entropy",
    "top1_offspring_share",
    "top3_offspring_share",
    "deaths_before_refill",
    "deaths_after_refill",
    "culled_count",
    "culled_share",
    "pre_cull_alive_share",
    "post_cull_alive_share",
    "post_refill_alive_count",
    "post_refill_alive_share",
    "refill_offspring_lineage_source",
    "refill_amplification_ratio",
    "local_free_neighbor_count_mean",
    "local_density_parent_site_mean",
    "local_density_child_site_mean",
    "parent_child_distance_mean",
    "parent_child_genome_distance_mean",
    "parent_child_genome_distance_max",
    "parent_child_genome_scalar_distance_mean",
    "parent_child_genome_memory_field_delta_mean",
    "parent_child_genome_prediction_field_delta_mean",
    "parent_child_genome_module_gate_delta_mean",
    "child_patch_id_or_coordinates",
    "parent_patch_id_or_coordinates",
    "child_basin_id",
    "parent_basin_id",
    "child_same_basin_as_parent_fraction",
    "lineage_birth_clustering_index",
    "lineage_patch_basin_reuse_rate",
    "food_parent_site_mean",
    "food_child_site_mean",
    "waste_parent_site_mean",
    "waste_child_site_mean",
    "scent_parent_site_mean",
    "scent_child_site_mean",
    "shelter_cover_parent_site_mean",
    "shelter_cover_child_site_mean",
    "predator_hazard_exposure_parent_site_mean",
    "predator_hazard_exposure_child_site_mean",
    "lro4_offspring_dispersal_event_rows",
    "lro4_offspring_dispersal_enabled_rows",
    "lro4_offspring_dispersal_disabled_rows",
    "lro4_offspring_dispersal_disabled_leakage_rows",
    "lro4_offspring_dispersal_triggered_rows",
    "lro4_offspring_dispersal_trigger_reason_counts_json",
    "lro4_offspring_dispersal_expanded_candidate_count_sum",
    "lro4_offspring_dispersal_expanded_candidate_count_mean",
    "lro4_offspring_dispersal_selected_from_expanded_rows",
    "lro4_offspring_dispersal_selected_radius_mean",
    "lro4_offspring_dispersal_parent_density_mean",
    "lro4_offspring_dispersal_parent_waste_mean",
    "lro4_offspring_dispersal_child_density_mean",
    "lro4_offspring_dispersal_child_waste_mean",
    "first_generation_ecological_payoff_metrics_activate",
    "pressure_exposure_before_lineage_collapse",
    "first_generation_zero_selected_parents",
    "first_generation_zero_successful_births",
    "last_generation_with_selected_parents",
    "last_generation_with_successful_births",
    "rolling_nonzero_birth_lineage_count",
    "rolling_nonzero_selected_parent_lineage_count",
    "eligible_parent_share_over_alive_share",
    "selected_parent_share_over_eligible_parent_share",
    "offspring_share_over_selected_parent_share",
    "post_refill_share_over_pre_refill_alive_share",
    "post_refill_share_over_offspring_share",
    "birth_success_share_over_birth_attempt_share",
    "selected_parent_share_over_alive_share",
    "offspring_share_over_alive_share",
    "eligible_amplification_class",
    "selected_parent_amplification_class",
    "offspring_amplification_class",
    "refill_amplification_class",
    "refill_over_offspring_amplification_class",
    "birth_success_amplification_class",
    "selected_over_alive_amplification_class",
    "offspring_over_alive_amplification_class",
]

EVENT_FIELDS = [
    "schema_version",
    "event_schema_version",
    "run_id",
    "generation",
    "tick",
    "event_id",
    "local_birth_event_id",
    "global_birth_event_id",
    "event",
    "event_type",
    "selected_exposure_id",
    "failure_reason",
    "failure_reason_detail",
    "failure_reason_group",
    "linked_attempt_event_id",
    "linked_success_event_id",
    "linked_failure_event_id",
    "linked_scan_event_id",
    "accounting_window_start",
    "accounting_window_end",
    "accounting_window_rule",
    "exposure_phase",
    "lineage_id",
    "child_lineage_id",
    "parent_lineage_id",
    "mate_lineage_id",
    "parent_id",
    "mate_id",
    "child_id",
    "agent_id",
    "biological_lineage_id",
    "parent_agent_id",
    "mate_agent_id",
    "child_agent_id",
    "parent_biological_lineage_id",
    "mate_biological_lineage_id",
    "child_biological_lineage_id",
    "parent_was_selected_parent_this_generation",
    "parent_selected_rank",
    "parent_selected_rank_bucket",
    "selected_parent_pool_size",
    "parent_fraction",
    "scan_id",
    "scan_phase",
    "scan_sequence",
    "scan_candidate",
    "scan_eligible",
    "scan_ineligible",
    "scan_ineligible_reason_group",
    "scan_ineligible_reason_detail",
    "action_offered",
    "action_taken",
    "action_not_taken",
    "action_not_taken_reason_group",
    "action_not_taken_reason_detail",
    "parent_alive_at_exposure",
    "parent_alive_at_attempt_window",
    "birth_attempted",
    "birth_attempt_recorded",
    "birth_success",
    "birth_failed",
    "no_attempt_marker",
    "no_local_space_marker",
    "no_space_marker",
    "non_space_marker",
    "unknown_failure_marker",
    "local_free_neighbor_count",
    "parent_free_neighbor_count",
    "candidate_free_spawn_cell_count",
    "parent_candidate_free_spawn_cells",
    "candidate_free_spawn_cells",
    "parent_energy",
    "reproduction_threshold",
    "energy_threshold_margin",
    "parent_health",
    "parent_age",
    "parent_effective_fitness",
    "parent_reproduction_eligible",
    "parent_ineligible_reason_group",
    "parent_ineligible_reason_detail",
    "cooldown_blocked",
    "age_blocked",
    "energy_blocked",
    "health_blocked",
    "mate_blocked",
    "action_exposure_blocked",
    "unknown_gate_blocked",
    "parent_reproduction_cooldown",
    "parent_recent_birth_attempt_count",
    "agent_energy",
    "agent_health",
    "agent_age",
    "agent_effective_fitness",
    "agent_reproduction_cooldown",
    "agent_local_density",
    "agent_free_neighbor_count",
    "agent_candidate_free_spawn_cells",
    "agent_patch_food",
    "agent_patch_waste",
    "agent_patch_scent",
    "agent_patch_cover",
    "agent_patch_shelter",
    "agent_basin_id",
    "reproduction_scan_seen",
    "mate_required",
    "mate_available",
    "mate_eligible",
    "mate_distance",
    "mate_block_reason",
    "parent_y",
    "parent_x",
    "mate_y",
    "mate_x",
    "child_y",
    "child_x",
    "path2_roz_reporting_enable",
    "path2_roz_schema_version",
    "path2_roz_zone_mode",
    "path2_roz_window_period_steps",
    "path2_roz_window_duration_steps",
    "path2_roz_zone_radius",
    "path2_roz_center_y",
    "path2_roz_center_x",
    "path2_roz_window_active",
    "path2_roz_window_id",
    "path2_roz_active_cell_count",
    "path2_roz_parent_distance_to_zone",
    "path2_roz_mate_distance_to_zone",
    "path2_roz_child_distance_to_zone",
    "path2_roz_parent_in_zone",
    "path2_roz_mate_in_zone",
    "path2_roz_child_in_zone",
    "path2_roz_birth_in_zone",
    "path2_roz_selected_parent_scan_in_zone",
    "path2_roz_schedule_mode",
    "path2_roz_schedule_id",
    "path2_roz_zone_id",
    "path2_roz_zone_index",
    "path2_roz_zone_count",
    "path2_roz_window_index",
    "path2_roz_schedule_phase_step",
    "lro1_reproductive_opportunity_enable",
    "lro1_reproductive_opportunity_mode",
    "lro1_opportunity_score_parent_site",
    "lro1_opportunity_multiplier",
    "lro1_opportunity_context_density",
    "lro1_opportunity_context_waste",
    "lro1_opportunity_context_scent",
    "lro1_opportunity_context_shelter_cover",
    "lro1_opportunity_context_food",
    "lro1_opportunity_context_hazard",
    "lro1_opportunity_invalid_context",
    "lro1_opportunity_neutral_context",
    "lro1_opportunity_effective_try_prob",
    "lro4_offspring_dispersal_enable",
    "lro4_offspring_dispersal_mode",
    "lro4_offspring_dispersal_triggered",
    "lro4_offspring_dispersal_trigger_reason",
    "lro4_offspring_dispersal_base_radius",
    "lro4_offspring_dispersal_max_radius",
    "lro4_offspring_dispersal_effective_radius",
    "lro4_offspring_dispersal_base_candidate_count",
    "lro4_offspring_dispersal_expanded_candidate_count",
    "lro4_offspring_dispersal_selected_radius",
    "lro4_offspring_dispersal_selected_from_expanded",
    "lro4_offspring_dispersal_parent_density",
    "lro4_offspring_dispersal_parent_waste",
    "lro4_offspring_dispersal_child_density",
    "lro4_offspring_dispersal_child_waste",
    "lro4_offspring_dispersal_disabled_leakage_marker",
    "parent_child_distance",
    "parent_child_genome_distance_mean",
    "parent_child_genome_distance_max",
    "parent_child_genome_scalar_distance_mean",
    "parent_child_genome_memory_field_delta",
    "parent_child_genome_prediction_field_delta",
    "parent_child_genome_module_gate_delta",
    "mutation_weather_enabled",
    "mutation_weather_profile",
    "mutation_weather_active",
    "mutation_weather_event_id",
    "mutation_weather_generation",
    "mutation_weather_severity",
    "mutation_weather_scalar_multiplier",
    "mutation_weather_count_multiplier",
    "mutation_weather_structural_enabled",
    "mutation_weather_affected_offspring",
    "child_mutation_weather_active",
    "child_mutation_weather_event_id",
    "child_mutation_weather_scalar_multiplier",
    "local_density_parent_site",
    "parent_local_density",
    "local_density_child_site",
    "food_parent_site",
    "parent_patch_food",
    "food_child_site",
    "waste_parent_site",
    "parent_patch_waste",
    "waste_child_site",
    "scent_parent_site",
    "parent_patch_scent",
    "scent_child_site",
    "shelter_cover_parent_site",
    "parent_patch_shelter",
    "shelter_cover_child_site",
    "parent_patch_cover",
    "predator_hazard_exposure_parent_site",
    "predator_hazard_exposure_child_site",
    "parent_basin_id",
    "child_basin_id",
    "birth_attempt_count",
    "failed_birth_attempt_count",
    "no_local_space",
    "structural_parent_child_variant_event_schema_version",
    "structural_variant_history_schema_version",
    "structural_variant_event_id",
    "structural_variant_event_kind",
    "parent_graph_identity_digest",
    "parent_graph_semantic_digest",
    "parent_graph_supported",
    "parent_operator_origin_set",
    "mate_graph_identity_digest",
    "mate_graph_semantic_digest",
    "mate_graph_supported",
    "mate_operator_origin_set",
    "child_post_recombination_graph_identity_digest",
    "child_post_recombination_graph_semantic_digest",
    "child_post_recombination_graph_supported",
    "child_graph_identity_digest",
    "child_graph_semantic_digest",
    "child_final_graph_identity_digest",
    "child_final_graph_semantic_digest",
    "attached_newborn_graph_identity_digest",
    "attached_newborn_graph_semantic_digest",
    "attached_newborn_matches_final_child",
    "child_variant_class",
    "child_canonical_seed_exact",
    "child_canonical_seed_semantic_match",
    "child_operator_origin_available",
    "child_operator_origin_set",
    "child_mutation_operator_origin_set",
    "child_recombination_origin_set",
    "child_c1_attempt_count",
    "child_c1_noop_count",
    "child_graph_changed_vs_parent",
    "child_semantic_graph_changed_vs_parent",
    "child_graph_supported",
    "mutation_delta_event_count",
    "mutation_delta_event_ids",
    "mutation_delta_statuses",
    "mutation_delta_events_json",
    "mutation_delta_semantic_change_count",
    "mutation_delta_historical_change_count",
    "mutation_delta_valid_noop_count",
    "mutation_delta_rejected_count",
]

_PATH2_ROZ_COMPACT_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event_schema_version",
        "run_id",
        "generation",
        "event_id",
        "event",
        "event_type",
        "failure_reason_group",
        "linked_attempt_event_id",
        "accounting_window_rule",
        "lineage_id",
        "child_lineage_id",
        "parent_lineage_id",
        "mate_lineage_id",
        "parent_id",
        "mate_id",
        "child_id",
        "agent_id",
        "parent_agent_id",
        "mate_agent_id",
        "child_agent_id",
        "parent_biological_lineage_id",
        "mate_biological_lineage_id",
        "child_biological_lineage_id",
        "parent_selected_rank",
        "scan_id",
        "birth_attempted",
        "birth_attempt_recorded",
        "birth_success",
        "birth_failed",
        "parent_energy",
        "parent_ineligible_reason_group",
        "mate_available",
        "mate_block_reason",
        "path2_roz_reporting_enable",
        "path2_roz_schema_version",
        "path2_roz_zone_mode",
        "path2_roz_window_period_steps",
        "path2_roz_window_duration_steps",
        "path2_roz_zone_radius",
        "path2_roz_center_y",
        "path2_roz_center_x",
        "path2_roz_window_active",
        "path2_roz_window_id",
        "path2_roz_active_cell_count",
        "path2_roz_parent_distance_to_zone",
        "path2_roz_mate_distance_to_zone",
        "path2_roz_child_distance_to_zone",
        "path2_roz_parent_in_zone",
        "path2_roz_mate_in_zone",
        "path2_roz_child_in_zone",
        "path2_roz_birth_in_zone",
        "path2_roz_selected_parent_scan_in_zone",
        "path2_roz_schedule_mode",
        "path2_roz_schedule_id",
        "path2_roz_zone_id",
        "path2_roz_zone_index",
        "path2_roz_zone_count",
        "path2_roz_window_index",
        "path2_roz_schedule_phase_step",
        "lro1_reproductive_opportunity_enable",
        "lro1_reproductive_opportunity_mode",
        "lro1_opportunity_score_parent_site",
        "lro1_opportunity_multiplier",
        "lro1_opportunity_invalid_context",
        "lro1_opportunity_neutral_context",
        "lro4_offspring_dispersal_enable",
        "lro4_offspring_dispersal_mode",
        "lro4_offspring_dispersal_triggered",
        "lro4_offspring_dispersal_trigger_reason",
        "lro4_offspring_dispersal_expanded_candidate_count",
        "lro4_offspring_dispersal_selected_from_expanded",
        "lro4_offspring_dispersal_disabled_leakage_marker",
        "mutation_weather_active",
        "mutation_weather_event_id",
        "mutation_weather_generation",
        "parent_child_genome_distance_mean",
        "parent_child_genome_distance_max",
        "parent_child_genome_memory_field_delta",
        "parent_child_genome_prediction_field_delta",
        "parent_child_genome_module_gate_delta",
        "child_mutation_weather_active",
        "child_mutation_weather_event_id",
    }
)


def enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "lineage_reproduction_funnel_reporting_enable", False)) if sim_cfg is not None else False


def validate_config(sim_cfg: Any) -> None:
    if (
        not enabled(sim_cfg)
        and not lro1_reproductive_opportunity_enabled(sim_cfg)
        and not bool(getattr(sim_cfg, "lro4_offspring_dispersal_enable", False))
    ):
        return
    if enabled(sim_cfg):
        version = _safe_int(getattr(sim_cfg, "lineage_reproduction_funnel_schema_version", SCHEMA_VERSION), SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"lineage_reproduction_funnel_schema_version must be {SCHEMA_VERSION}, got {version}")
        event_version = configured_event_schema_version(sim_cfg)
        if event_version not in {EVENT_SCHEMA_VERSION, SCAN_ACTION_EVENT_SCHEMA_VERSION}:
            raise ValueError(
                "lineage_reproduction_funnel_event_schema_version must be "
                f"{EVENT_SCHEMA_VERSION} or {SCAN_ACTION_EVENT_SCHEMA_VERSION}, got {event_version}"
            )
        if scan_action_events_enabled(sim_cfg) and event_version < SCAN_ACTION_EVENT_SCHEMA_VERSION:
            raise ValueError(
                "lineage_reproduction_funnel_write_scan_action_events requires "
                f"event schema version {SCAN_ACTION_EVENT_SCHEMA_VERSION}"
            )
        event_output_profile = str(getattr(sim_cfg, "lineage_reproduction_funnel_event_output_profile", "full") or "full")
        if event_output_profile not in EVENT_OUTPUT_PROFILES:
            raise ValueError(
                "lineage_reproduction_funnel_event_output_profile must be one of "
                f"{sorted(EVENT_OUTPUT_PROFILES)}, got {event_output_profile!r}"
            )
        if event_output_profile == "path2_roz_summary_only_v1" and not bool(
            getattr(sim_cfg, "lineage_reproduction_funnel_incremental_summary_cache_enable", False)
        ):
            raise ValueError(
                "lineage_reproduction_funnel_event_output_profile=path2_roz_summary_only_v1 "
                "requires lineage_reproduction_funnel_incremental_summary_cache_enable=true"
            )
        if path2_roz_reporting_enabled(sim_cfg) and not scan_action_events_enabled(sim_cfg):
            raise ValueError(
                "path2_roz_reporting_enable requires lineage_reproduction_funnel_write_events=true, "
                "lineage_reproduction_funnel_write_scan_action_events=true, and event schema version 4"
            )
    if path2_roz_reporting_enabled(sim_cfg):
        roz_version = _safe_int(getattr(sim_cfg, "path2_roz_schema_version", 1), 1)
        if roz_version != 1:
            raise ValueError(f"path2_roz_schema_version must be 1, got {roz_version}")
        period = _safe_int(getattr(sim_cfg, "path2_roz_window_period_steps", 0), 0)
        duration = _safe_int(getattr(sim_cfg, "path2_roz_window_duration_steps", 0), 0)
        radius = _safe_int(getattr(sim_cfg, "path2_roz_zone_radius", 0), 0)
        if period <= 0:
            raise ValueError("path2_roz_window_period_steps must be positive when path2_roz_reporting_enable=true")
        if duration < 0 or duration > period:
            raise ValueError(
                "path2_roz_window_duration_steps must be between 0 and path2_roz_window_period_steps "
                "when path2_roz_reporting_enable=true"
            )
        if radius < 0:
            raise ValueError("path2_roz_zone_radius must be non-negative when path2_roz_reporting_enable=true")
        schedule_mode = str(getattr(sim_cfg, "path2_roz_schedule_mode", "static_periodic_compat") or "static_periodic_compat")
        if schedule_mode not in PATH2_ROZ_SCHEDULE_MODES:
            raise ValueError(f"path2_roz_schedule_mode unsupported: {schedule_mode}")
        phase_offset = _safe_int(getattr(sim_cfg, "path2_roz_schedule_phase_offset_steps", 0), 0)
        if phase_offset < 0:
            raise ValueError("path2_roz_schedule_phase_offset_steps must be non-negative when path2_roz_reporting_enable=true")
        zones_json = str(getattr(sim_cfg, "path2_roz_schedule_zones_json", "") or "").strip()
        if schedule_mode == "patch_derived_static" and zones_json:
            raise ValueError("patch_derived_static derives zones from Patch World arrays and does not accept zones_json")
        if schedule_mode in {"periodic_single_zone", "multi_zone_round_robin"} and zones_json:
            try:
                zones = json.loads(zones_json)
            except Exception as exc:
                raise ValueError("path2_roz_schedule_zones_json must be valid JSON") from exc
            if not isinstance(zones, list) or not zones:
                raise ValueError("path2_roz_schedule_zones_json must be a non-empty JSON list")
            if schedule_mode == "periodic_single_zone" and len(zones) != 1:
                raise ValueError("periodic_single_zone requires exactly one schedule zone when zones_json is provided")
            for idx, zone in enumerate(zones):
                if not isinstance(zone, dict):
                    raise ValueError("path2_roz_schedule_zones_json entries must be objects")
                for key in ("center_y", "center_x", "radius"):
                    if key not in zone:
                        raise ValueError(f"path2_roz_schedule_zones_json zone {idx} missing {key}")
                    value = _safe_int(zone.get(key), -1)
                    if value < 0:
                        raise ValueError(f"path2_roz_schedule_zones_json zone {idx} {key} must be non-negative")
    if lro1_reproductive_opportunity_enabled(sim_cfg):
        mode = str(
            getattr(sim_cfg, "lro1_reproductive_opportunity_mode", LRO1_REPRODUCTIVE_OPPORTUNITY_MODE)
            or LRO1_REPRODUCTIVE_OPPORTUNITY_MODE
        )
        if mode != LRO1_REPRODUCTIVE_OPPORTUNITY_MODE:
            raise ValueError(f"lro1_reproductive_opportunity_mode unsupported: {mode}")
        stage = str(
            getattr(sim_cfg, "lro1_reproductive_opportunity_apply_stage", LRO1_REPRODUCTIVE_OPPORTUNITY_APPLY_STAGE)
            or LRO1_REPRODUCTIVE_OPPORTUNITY_APPLY_STAGE
        )
        if stage != LRO1_REPRODUCTIVE_OPPORTUNITY_APPLY_STAGE:
            raise ValueError(f"lro1_reproductive_opportunity_apply_stage unsupported: {stage}")
        min_multiplier = _safe_float(getattr(sim_cfg, "lro1_reproductive_opportunity_min_multiplier", 0.35), 0.35)
        max_multiplier = _safe_float(getattr(sim_cfg, "lro1_reproductive_opportunity_max_multiplier", 1.0), 1.0)
        if min_multiplier < 0.0 or max_multiplier < min_multiplier or max_multiplier > 1.0:
            raise ValueError("lro1 reproductive opportunity multipliers must satisfy 0 <= min <= max <= 1")
        raw_weights = getattr(sim_cfg, "lro1_reproductive_opportunity_weights_json", "") or ""
        if isinstance(raw_weights, str):
            try:
                weights = json.loads(raw_weights) if raw_weights.strip() else {}
            except Exception as exc:
                raise ValueError("lro1_reproductive_opportunity_weights_json must be valid JSON") from exc
        elif isinstance(raw_weights, dict):
            weights = raw_weights
        else:
            raise ValueError("lro1_reproductive_opportunity_weights_json must be a JSON object")
        if not isinstance(weights, dict):
            raise ValueError("lro1_reproductive_opportunity_weights_json must be a JSON object")
        allowed = {"density", "waste", "scent", "shelter_cover", "food", "hazard"}
        for key, value in weights.items():
            if key not in allowed:
                raise ValueError(f"unsupported LRO1 reproductive opportunity weight: {key}")
            _safe_float(value, 0.0)
    if bool(getattr(sim_cfg, "lro4_offspring_dispersal_enable", False)):
        mode = str(getattr(sim_cfg, "lro4_offspring_dispersal_mode", LRO4_OFFSPRING_DISPERSAL_MODE) or "")
        if mode != LRO4_OFFSPRING_DISPERSAL_MODE:
            raise ValueError(f"lro4_offspring_dispersal_mode unsupported: {mode}")
        schema_version = _safe_int(getattr(sim_cfg, "lro4_offspring_dispersal_schema_version", 1), 1)
        if schema_version != 1:
            raise ValueError(f"lro4_offspring_dispersal_schema_version must be 1, got {schema_version}")
        base_radius = _safe_int(getattr(sim_cfg, "lro4_offspring_dispersal_base_radius", 1), 1)
        max_radius = _safe_int(getattr(sim_cfg, "lro4_offspring_dispersal_max_radius", 3), 3)
        if base_radius < 1 or max_radius < base_radius or max_radius > 8:
            raise ValueError("lro4_offspring_dispersal radii must satisfy 1 <= base <= max <= 8")
        density_threshold = _safe_float(getattr(sim_cfg, "lro4_offspring_dispersal_density_threshold", 4.0), 4.0)
        waste_threshold = _safe_float(getattr(sim_cfg, "lro4_offspring_dispersal_waste_threshold", 0.25), 0.25)
        if density_threshold < 0.0 or waste_threshold < 0.0:
            raise ValueError("lro4_offspring_dispersal thresholds must be non-negative")
        if not bool(getattr(sim_cfg, "lro4_offspring_dispersal_deterministic_ring_order", True)):
            raise ValueError("lro4_offspring_dispersal requires deterministic_ring_order=true")


def configured_event_schema_version(sim_cfg: Any) -> int:
    return _safe_int(
        getattr(sim_cfg, "lineage_reproduction_funnel_event_schema_version", EVENT_SCHEMA_VERSION),
        EVENT_SCHEMA_VERSION,
    )


def scan_action_events_enabled(sim_cfg: Any) -> bool:
    if sim_cfg is None:
        return False
    return bool(
        enabled(sim_cfg)
        and getattr(sim_cfg, "lineage_reproduction_funnel_write_events", False)
        and getattr(sim_cfg, "lineage_reproduction_funnel_write_scan_action_events", False)
        and configured_event_schema_version(sim_cfg) >= SCAN_ACTION_EVENT_SCHEMA_VERSION
    )


def path2_roz_reporting_enabled(sim_cfg: Any) -> bool:
    if sim_cfg is None:
        return False
    return bool(enabled(sim_cfg) and getattr(sim_cfg, "path2_roz_reporting_enable", False))


def lro1_reproductive_opportunity_enabled(sim_cfg: Any) -> bool:
    if sim_cfg is None:
        return False
    return bool(getattr(sim_cfg, "lro1_reproductive_opportunity_enable", False))


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
    return float(out) if math.isfinite(out) else float(default)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_jsonl_payload(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(_stable_json(row) + "\n" for row in rows)


def _compact_event_row(row: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in sorted(fields):
        value = row.get(field, "")
        if value in ("", None):
            continue
        out[field] = value
    return out


def _agent_id(agent: Any) -> int:
    return _safe_int(getattr(agent, "agent_id", 0), 0)


def _lineage_id_for_agent(agent: Any) -> str:
    aid = _agent_id(agent)
    stats = normalize_lineage_stats(getattr(agent, "stats", None), aid, generation=0, tick=0)
    if stats is None:
        return format_lineage_id(aid)
    return str(getattr(stats, "lineage_id", "") or format_lineage_id(aid))


def lineage_id_from_context(row: Mapping[str, Any]) -> str:
    raw = str(row.get("lineage_id", "") or row.get("child_lineage_id", "") or "")
    if raw.startswith("L"):
        return raw
    raw_i = _safe_int(_safe_float(raw, 0.0), 0)
    return format_lineage_id(raw_i) if raw_i > 0 else ""


def _agent_condition_fields(agent: Any) -> dict[str, Any]:
    if agent is None:
        return {}
    stats = getattr(agent, "stats", None)
    genome = getattr(agent, "genome", None)
    threshold_raw = getattr(genome, "reproduction_threshold", getattr(agent, "reproduction_threshold", ""))
    threshold = "" if threshold_raw in ("", None) else _safe_float(threshold_raw, 0.0)
    energy = _safe_float(getattr(agent, "energy", 0.0), 0.0)
    cooldown = _safe_int(
        getattr(agent, "repro_cooldown", getattr(agent, "reproduction_cooldown", 0)),
        0,
    )
    return {
        "parent_energy": energy,
        "reproduction_threshold": threshold,
        "energy_threshold_margin": "" if threshold == "" else float(energy - _safe_float(threshold, 0.0)),
        "parent_health": _safe_float(getattr(agent, "health", 0.0), 0.0),
        "parent_age": _safe_int(getattr(stats, "survival_steps", 0) if stats is not None else 0, 0),
        "parent_effective_fitness": _safe_float(getattr(stats, "effective_fitness", 0.0) if stats is not None else 0.0, 0.0),
        "parent_reproduction_cooldown": cooldown,
        "parent_recent_birth_attempt_count": _safe_int(
            getattr(stats, "offspring_count", 0) if stats is not None else 0,
            0,
        ),
    }


def _selected_exposure_id(generation: int, rank: int) -> str:
    return f"g{int(generation):06d}:selected:{int(rank):06d}" if int(rank) > 0 else ""


def _accounting_window_payload(generation: int) -> dict[str, Any]:
    gen = int(generation)
    return {
        "accounting_window_start": gen,
        "accounting_window_end": gen,
        "accounting_window_rule": "same_generation_parent_id_linkage_v3",
        "exposure_phase": "post_selection_pre_turnover_snapshot",
    }


def _parent_ineligibility(payload: Mapping[str, Any]) -> tuple[int, str, str]:
    cooldown = _safe_int(payload.get("parent_reproduction_cooldown", 0), 0)
    health = _safe_float(payload.get("parent_health", 0.0), 0.0)
    threshold = payload.get("reproduction_threshold", "")
    energy = _safe_float(payload.get("parent_energy", 0.0), 0.0)
    if cooldown > 0:
        return 0, "cooldown_age", "parent_reproduction_cooldown_positive"
    if health <= 0.0:
        return 0, "health", "parent_health_nonpositive"
    if threshold not in ("", None) and energy < _safe_float(threshold, 0.0):
        return 0, "energy", "parent_energy_below_reproduction_threshold"
    return 1, "", ""


def _apply_v3_event_defaults(
    payload: dict[str, Any],
    *,
    generation: int,
    event_type: str,
    event_schema_version: int = EVENT_SCHEMA_VERSION,
) -> dict[str, Any]:
    version = _safe_int(event_schema_version, EVENT_SCHEMA_VERSION)
    payload["schema_version"] = version
    payload["event_schema_version"] = version
    window = _accounting_window_payload(generation)
    if str(event_type).startswith("reproduction_scan") or str(event_type).startswith("reproduction_action"):
        window = {
            "accounting_window_start": int(generation),
            "accounting_window_end": int(generation),
            "accounting_window_rule": "same_tick_live_reproduction_scan_linkage_v4",
            "exposure_phase": "live_tick_reproduction_scan",
        }
    payload.update({k: payload.get(k, v) for k, v in window.items()})
    rank = _safe_int(payload.get("parent_selected_rank", 0), 0)
    exposure_id = str(payload.get("selected_exposure_id") or _selected_exposure_id(generation, rank))
    payload["selected_exposure_id"] = exposure_id
    payload["failure_reason_group"] = str(payload.get("failure_reason_group") or _failure_reason_group(payload.get("failure_reason", "")))
    payload["failure_reason_detail"] = str(payload.get("failure_reason_detail") or payload.get("failure_reason") or "")
    parent_alive = _safe_int(payload.get("parent_alive_at_exposure", payload.get("parent_reproduction_eligible", 0)), 0)
    payload["parent_alive_at_exposure"] = parent_alive
    payload["parent_alive_at_attempt_window"] = _safe_int(payload.get("parent_alive_at_attempt_window", parent_alive), 0)
    payload["birth_attempt_recorded"] = _safe_int(payload.get("birth_attempt_recorded", int(event_type == "birth_attempt_recorded")), 0)
    payload["no_attempt_marker"] = _safe_int(
        payload.get("no_attempt_marker", int(event_type in {"selected_parent_no_attempt", "reproduction_scan_no_attempt"})),
        0,
    )
    payload["no_space_marker"] = _safe_int(payload.get("no_space_marker", int(event_type == "birth_fail_no_space")), 0)
    payload["non_space_marker"] = _safe_int(payload.get("non_space_marker", int(event_type == "birth_fail_non_space")), 0)
    payload["unknown_failure_marker"] = _safe_int(payload.get("unknown_failure_marker", int(event_type == "birth_fail_unknown")), 0)
    payload["local_free_neighbor_count"] = _safe_int(payload.get("local_free_neighbor_count", payload.get("parent_free_neighbor_count", 0)), 0)
    payload["parent_free_neighbor_count"] = _safe_int(payload.get("parent_free_neighbor_count", payload.get("local_free_neighbor_count", 0)), 0)
    payload["candidate_free_spawn_cell_count"] = _safe_int(
        payload.get("candidate_free_spawn_cell_count", payload.get("parent_candidate_free_spawn_cells", payload.get("candidate_free_spawn_cells", 0))),
        0,
    )
    payload["parent_candidate_free_spawn_cells"] = _safe_int(
        payload.get("parent_candidate_free_spawn_cells", payload.get("candidate_free_spawn_cell_count", payload.get("candidate_free_spawn_cells", 0))),
        0,
    )
    payload["candidate_free_spawn_cells"] = _safe_int(payload.get("candidate_free_spawn_cells", payload.get("candidate_free_spawn_cell_count", 0)), 0)
    parent_id = _safe_int(payload.get("parent_id", payload.get("parent_agent_id", payload.get("agent_id", -1))), -1)
    parent_lineage = str(
        payload.get("parent_biological_lineage_id")
        or payload.get("parent_lineage_id")
        or payload.get("biological_lineage_id")
        or ""
    )
    payload["agent_id"] = _safe_int(payload.get("agent_id", parent_id), parent_id)
    payload["biological_lineage_id"] = str(payload.get("biological_lineage_id") or parent_lineage)
    payload["scan_id"] = str(payload.get("scan_id") or "")
    payload["scan_phase"] = str(
        payload.get("scan_phase")
        or ("live_tick_reproduction_scan" if str(event_type).startswith("reproduction_scan") else "")
    )
    payload["scan_sequence"] = _safe_int(payload.get("scan_sequence", 0), 0)
    payload["scan_candidate"] = _safe_int(payload.get("scan_candidate", int(str(event_type).startswith("reproduction_scan"))), 0)
    payload["scan_eligible"] = _safe_int(payload.get("scan_eligible", int(event_type == "reproduction_scan_eligible")), 0)
    payload["scan_ineligible"] = _safe_int(payload.get("scan_ineligible", int(event_type == "reproduction_scan_ineligible")), 0)
    payload["scan_ineligible_reason_group"] = str(
        payload.get("scan_ineligible_reason_group") or payload.get("failure_reason_group") or ""
    )
    payload["scan_ineligible_reason_detail"] = str(
        payload.get("scan_ineligible_reason_detail") or payload.get("failure_reason_detail") or ""
    )
    payload["action_offered"] = _safe_int(payload.get("action_offered", int(event_type == "reproduction_action_offered")), 0)
    payload["action_taken"] = _safe_int(payload.get("action_taken", int(event_type == "reproduction_action_taken")), 0)
    payload["action_not_taken"] = _safe_int(
        payload.get("action_not_taken", int(event_type == "reproduction_action_not_taken")),
        0,
    )
    payload["action_not_taken_reason_group"] = str(payload.get("action_not_taken_reason_group") or "")
    payload["action_not_taken_reason_detail"] = str(payload.get("action_not_taken_reason_detail") or "")
    payload["agent_energy"] = _safe_float(payload.get("agent_energy", payload.get("parent_energy", 0.0)), 0.0)
    payload["agent_health"] = _safe_float(payload.get("agent_health", payload.get("parent_health", 0.0)), 0.0)
    payload["agent_age"] = _safe_int(payload.get("agent_age", payload.get("parent_age", 0)), 0)
    payload["agent_effective_fitness"] = _safe_float(
        payload.get("agent_effective_fitness", payload.get("parent_effective_fitness", 0.0)),
        0.0,
    )
    payload["agent_reproduction_cooldown"] = _safe_int(
        payload.get("agent_reproduction_cooldown", payload.get("parent_reproduction_cooldown", 0)),
        0,
    )
    payload["agent_local_density"] = _safe_float(payload.get("agent_local_density", payload.get("local_density_parent_site", 0.0)), 0.0)
    payload["agent_free_neighbor_count"] = _safe_int(
        payload.get("agent_free_neighbor_count", payload.get("parent_free_neighbor_count", 0)),
        0,
    )
    payload["agent_candidate_free_spawn_cells"] = _safe_int(
        payload.get("agent_candidate_free_spawn_cells", payload.get("candidate_free_spawn_cell_count", 0)),
        0,
    )
    payload["agent_patch_food"] = _safe_float(payload.get("agent_patch_food", payload.get("food_parent_site", 0.0)), 0.0)
    payload["agent_patch_waste"] = _safe_float(payload.get("agent_patch_waste", payload.get("waste_parent_site", 0.0)), 0.0)
    payload["agent_patch_scent"] = _safe_float(payload.get("agent_patch_scent", payload.get("scent_parent_site", 0.0)), 0.0)
    payload["agent_patch_cover"] = _safe_float(payload.get("agent_patch_cover", payload.get("parent_patch_cover", 0.0)), 0.0)
    payload["agent_patch_shelter"] = _safe_float(payload.get("agent_patch_shelter", payload.get("parent_patch_shelter", 0.0)), 0.0)
    payload["agent_basin_id"] = str(payload.get("agent_basin_id") or payload.get("parent_basin_id") or "")
    payload["cooldown_blocked"] = int(str(payload.get("parent_ineligible_reason_group", "")) == "cooldown_age")
    payload["energy_blocked"] = int(str(payload.get("parent_ineligible_reason_group", "")) == "energy")
    payload["health_blocked"] = int(str(payload.get("parent_ineligible_reason_group", "")) == "health")
    payload["mate_blocked"] = int(str(payload.get("parent_ineligible_reason_group", "")) == "mate")
    payload["age_blocked"] = int(str(payload.get("parent_ineligible_reason_group", "")) == "age")
    payload["action_exposure_blocked"] = int(str(payload.get("parent_ineligible_reason_group", "")) == "action_exposure")
    payload["unknown_gate_blocked"] = int(str(payload.get("parent_ineligible_reason_group", "")) == "unknown")
    payload["reproduction_scan_seen"] = _safe_int(payload.get("reproduction_scan_seen", 1), 1)
    payload["mate_required"] = _safe_int(payload.get("mate_required", 0), 0)
    payload["mate_available"] = _safe_int(payload.get("mate_available", int(_safe_int(payload.get("mate_id", -1), -1) >= 0)), 0)
    payload["mate_eligible"] = _safe_int(payload.get("mate_eligible", payload.get("mate_available", 0)), 0)
    payload["mate_distance"] = payload.get("mate_distance", "")
    payload["mate_block_reason"] = str(payload.get("mate_block_reason") or "")
    payload["linked_attempt_event_id"] = str(payload.get("linked_attempt_event_id") or "")
    payload["linked_success_event_id"] = str(payload.get("linked_success_event_id") or "")
    payload["linked_failure_event_id"] = str(payload.get("linked_failure_event_id") or "")
    payload["linked_scan_event_id"] = str(payload.get("linked_scan_event_id") or "")
    return payload


def _selected_rank_bucket(rank: int) -> str:
    r = _safe_int(rank, 0)
    if r <= 0:
        return ""
    if r == 1:
        return "top1_selected_parent"
    if r <= 3:
        return "top3_selected_parent"
    return "selected_parent_tail"


def _selected_parent_metadata(selected_agents: Iterable[Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    selected = list(selected_agents or [])
    pool_size = len(selected)
    for idx, agent in enumerate(selected, start=1):
        aid = _agent_id(agent)
        out[aid] = {
            "parent_was_selected_parent_this_generation": 1,
            "parent_selected_rank": idx,
            "parent_selected_rank_bucket": _selected_rank_bucket(idx),
            "selected_parent_pool_size": pool_size,
        }
    return out


def _failure_reason_group(reason: Any) -> str:
    r = str(reason or "").strip()
    if not r:
        return ""
    if r in {"birth_fail_no_space", "birth_pre_no_space"}:
        return "no_space"
    if r in {"birth_pre_no_energy"}:
        return "energy"
    if r in {"birth_fail_injury_gate", "birth_fail_scar_gate", "birth_pre_injury_gate"}:
        return "health"
    if r in {"birth_fail_no_mate", "birth_pre_no_mate"}:
        return "mate"
    if r in {"birth_pre_cooldown"}:
        return "cooldown_age"
    if r in {"birth_fail_buffer", "birth_make_child_fail"}:
        return "occupancy_collision"
    if r in {"birth_fail_conception", "birth_fail_gestation"}:
        return "other"
    return "unknown"


def _event_type_for(row: Mapping[str, Any]) -> str:
    existing = str(row.get("event_type", "") or "")
    if existing:
        return existing
    event = str(row.get("event", "") or "")
    reason = str(row.get("failure_reason", "") or "")
    if event == "selected_parent_exposure":
        return "selected_parent_exposure"
    if event == "successful_birth":
        return "birth_success"
    if event == "failed_birth":
        if reason == "birth_fail_no_space" or _safe_int(row.get("no_local_space", 0), 0) > 0:
            return "birth_fail_no_space"
        if reason:
            return "birth_fail_non_space"
        return "birth_fail_unknown"
    return event or "birth_fail_unknown"


def _legacy_event_for(event_type: str) -> str:
    if event_type == "selected_parent_exposure":
        return "selected_parent_exposure"
    if event_type == "birth_success":
        return "successful_birth"
    if event_type.startswith("birth_fail"):
        return "failed_birth"
    return str(event_type or "")


def _normalize_event_row(
    row: Mapping[str, Any],
    *,
    generation: int,
    event_index: int,
    run_id: str,
    selected_meta: Mapping[int, Mapping[str, Any]],
    agent_by_id: Mapping[int, Any],
    parent_fraction: float | None,
    event_schema_version: int = EVENT_SCHEMA_VERSION,
) -> dict[str, Any]:
    payload = dict(row)
    event_type = _event_type_for(payload)
    parent_id = _safe_int(payload.get("parent_agent_id", payload.get("parent_id", -1)), -1)
    mate_id = _safe_int(payload.get("mate_agent_id", payload.get("mate_id", -1)), -1)
    child_id = _safe_int(payload.get("child_agent_id", payload.get("child_id", -1)), -1)
    parent_lineage = str(
        payload.get("parent_biological_lineage_id")
        or payload.get("parent_lineage_id")
        or payload.get("lineage_id")
        or ""
    )
    child_lineage = str(payload.get("child_biological_lineage_id") or payload.get("child_lineage_id") or "")
    mate_lineage = str(payload.get("mate_biological_lineage_id") or payload.get("mate_lineage_id") or "")
    reason = str(payload.get("failure_reason", "") or "")
    selected = dict(selected_meta.get(parent_id, {}) or {})
    parent_agent = agent_by_id.get(parent_id)
    condition = _agent_condition_fields(parent_agent)
    if condition:
        payload.update({k: payload.get(k, v) for k, v in condition.items()})
    payload.update(
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_schema_version": event_schema_version,
            "run_id": str(run_id or ""),
            "generation": _safe_int(payload.get("generation", generation), generation),
            "tick": _safe_int(payload.get("tick", -1), -1),
            "event_id": str(payload.get("event_id") or f"g{int(generation):06d}:e{int(event_index):08d}"),
            "event": _legacy_event_for(event_type),
            "event_type": event_type,
            "failure_reason": reason,
            "failure_reason_group": _failure_reason_group(reason),
            "parent_id": parent_id,
            "mate_id": mate_id,
            "child_id": child_id,
            "parent_agent_id": parent_id,
            "mate_agent_id": mate_id,
            "child_agent_id": child_id,
            "parent_lineage_id": parent_lineage,
            "mate_lineage_id": mate_lineage,
            "child_lineage_id": child_lineage,
            "parent_biological_lineage_id": parent_lineage,
            "mate_biological_lineage_id": mate_lineage,
            "child_biological_lineage_id": child_lineage,
            "lineage_id": str(payload.get("lineage_id") or child_lineage or parent_lineage),
            "birth_attempted": _safe_int(
                payload.get(
                    "birth_attempted",
                    int(event_type in {"birth_success", "birth_fail_no_space", "birth_fail_non_space", "birth_fail_unknown"}),
                ),
                0,
            ),
            "birth_success": _safe_int(payload.get("birth_success", int(event_type == "birth_success")), 0),
            "birth_failed": _safe_int(payload.get("birth_failed", int(event_type.startswith("birth_fail"))), 0),
            "no_local_space_marker": int(event_type == "birth_fail_no_space"),
            "no_local_space": int(event_type == "birth_fail_no_space"),
            "candidate_free_spawn_cells": _safe_int(
                payload.get("candidate_free_spawn_cells", payload.get("candidate_free_spawn_cell_count", 0)),
                0,
            ),
            "candidate_free_spawn_cell_count": _safe_int(
                payload.get("candidate_free_spawn_cell_count", payload.get("candidate_free_spawn_cells", 0)),
                0,
            ),
            "parent_local_density": _safe_float(
                payload.get("parent_local_density", payload.get("local_density_parent_site", 0.0)),
                0.0,
            ),
            "parent_patch_food": _safe_float(payload.get("parent_patch_food", payload.get("food_parent_site", 0.0)), 0.0),
            "parent_patch_waste": _safe_float(payload.get("parent_patch_waste", payload.get("waste_parent_site", 0.0)), 0.0),
            "parent_patch_scent": _safe_float(payload.get("parent_patch_scent", payload.get("scent_parent_site", 0.0)), 0.0),
            "parent_patch_shelter": _safe_float(
                payload.get("parent_patch_shelter", payload.get("shelter_cover_parent_site", 0.0)),
                0.0,
            ),
            "parent_patch_cover": _safe_float(payload.get("parent_patch_cover", payload.get("shelter_cover_parent_site", 0.0)), 0.0),
        }
    )
    payload["parent_was_selected_parent_this_generation"] = _safe_int(
        payload.get("parent_was_selected_parent_this_generation", selected.get("parent_was_selected_parent_this_generation", 0)),
        0,
    )
    if event_type == "birth_success":
        local_event_id = str(payload.get("event_id") or "")
        payload["local_birth_event_id"] = local_event_id
        payload["global_birth_event_id"] = str(
            payload.get("global_birth_event_id")
            or global_event_id(
                run_id=str(run_id or ""),
                local_event_id=local_event_id,
                event_kind="birth_success",
                generation=_safe_int(payload.get("generation", generation), generation),
                tick=_safe_int(payload.get("tick", -1), -1),
                event_sequence=event_index,
            )
        )
        count = _safe_int(payload.get("mutation_delta_event_count", 0), 0)
        existing_ids = [item.strip() for item in str(payload.get("mutation_delta_event_ids") or "").split(",")]
        ids_incomplete = len(existing_ids) != count or any(not item for item in existing_ids)
        if count > 0 and ids_incomplete:
            payload["mutation_delta_event_ids"] = ",".join(
                global_event_id(
                    run_id=str(run_id or ""),
                    local_event_id=f"{local_event_id}:c1:{idx:02d}",
                    event_kind="mutation_attempt",
                    generation=_safe_int(payload.get("generation", generation), generation),
                    tick=_safe_int(payload.get("tick", -1), -1),
                    event_sequence=idx,
                )
                for idx in range(count)
            )
    payload["parent_selected_rank"] = _safe_int(
        payload.get("parent_selected_rank", selected.get("parent_selected_rank", 0)),
        0,
    )
    payload["parent_selected_rank_bucket"] = str(
        payload.get("parent_selected_rank_bucket", selected.get("parent_selected_rank_bucket", "")) or ""
    )
    payload["selected_parent_pool_size"] = _safe_int(
        payload.get("selected_parent_pool_size", selected.get("selected_parent_pool_size", 0)),
        0,
    )
    payload["parent_fraction"] = "" if parent_fraction is None else _safe_float(parent_fraction, 0.0)
    payload["parent_reproduction_eligible"] = int(
        payload["parent_was_selected_parent_this_generation"] > 0
        or payload["parent_selected_rank"] > 0
        or bool(parent_agent is not None and getattr(parent_agent, "alive", False))
    )
    _apply_v3_event_defaults(
        payload,
        generation=int(generation),
        event_type=event_type,
        event_schema_version=event_schema_version,
    )
    return {field: payload.get(field, "") for field in EVENT_FIELDS}


def _selected_parent_event_rows(
    *,
    generation: int,
    selected_agents: Iterable[Any],
    run_id: str,
    parent_fraction: float | None,
) -> list[dict[str, Any]]:
    selected = list(selected_agents or [])
    rows: list[dict[str, Any]] = []
    pool_size = len(selected)
    for idx, agent in enumerate(selected, start=1):
        aid = _agent_id(agent)
        condition = _agent_condition_fields(agent)
        lineage_id = _lineage_id_for_agent(agent)
        row: dict[str, Any] = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "run_id": str(run_id or ""),
            "generation": int(generation),
            "tick": -1,
            "event_id": f"g{int(generation):06d}:selected:{idx:06d}",
            "event": "selected_parent_exposure",
            "event_type": "selected_parent_exposure",
            "selected_exposure_id": _selected_exposure_id(int(generation), idx),
            "lineage_id": lineage_id,
            "parent_lineage_id": lineage_id,
            "parent_biological_lineage_id": lineage_id,
            "parent_id": aid,
            "parent_agent_id": aid,
            "mate_id": -1,
            "mate_agent_id": -1,
            "child_id": -1,
            "child_agent_id": -1,
            "parent_was_selected_parent_this_generation": 1,
            "parent_selected_rank": idx,
            "parent_selected_rank_bucket": _selected_rank_bucket(idx),
            "selected_parent_pool_size": pool_size,
            "parent_fraction": "" if parent_fraction is None else _safe_float(parent_fraction, 0.0),
            "birth_attempted": 0,
            "birth_attempt_recorded": 0,
            "birth_success": 0,
            "birth_failed": 0,
            "parent_alive_at_exposure": int(bool(getattr(agent, "alive", True))),
            "parent_alive_at_attempt_window": int(bool(getattr(agent, "alive", True))),
        }
        row.update(condition)
        eligible, reason_group, reason_detail = _parent_ineligibility(row)
        row["parent_reproduction_eligible"] = eligible
        row["parent_ineligible_reason_group"] = reason_group
        row["parent_ineligible_reason_detail"] = reason_detail
        _apply_v3_event_defaults(row, generation=int(generation), event_type="selected_parent_exposure")
        rows.append({field: row.get(field, "") for field in EVENT_FIELDS})
    return rows


def _derived_selected_parent_rows(
    *,
    generation: int,
    selected_events: Iterable[Mapping[str, Any]],
    birth_events: Iterable[Mapping[str, Any]],
    event_schema_version: int = EVENT_SCHEMA_VERSION,
) -> list[dict[str, Any]]:
    first_attempt_by_parent: dict[int, dict[str, Any]] = {}
    for event in birth_events:
        parent_id = _safe_int(event.get("parent_id", event.get("parent_agent_id", -1)), -1)
        if parent_id < 0 or parent_id in first_attempt_by_parent:
            continue
        if _safe_int(event.get("birth_attempted", 0), 0) > 0:
            first_attempt_by_parent[parent_id] = dict(event)

    rows: list[dict[str, Any]] = []
    for selected in selected_events:
        payload = dict(selected)
        parent_id = _safe_int(payload.get("parent_id", payload.get("parent_agent_id", -1)), -1)
        exposure_id = str(payload.get("selected_exposure_id") or payload.get("event_id") or "")
        eligible = _safe_int(payload.get("parent_reproduction_eligible", 0), 0)
        attempt = first_attempt_by_parent.get(parent_id)
        status_type = "selected_parent_eligible" if eligible > 0 else "selected_parent_ineligible"
        status = dict(payload)
        status.update(
            {
                "event_id": f"{exposure_id}:eligible" if eligible > 0 else f"{exposure_id}:ineligible",
                "event": status_type,
                "event_type": status_type,
                "birth_attempted": 0,
                "birth_attempt_recorded": 0,
                "birth_success": 0,
                "birth_failed": 0,
                "linked_attempt_event_id": str(attempt.get("event_id", "") if attempt else ""),
                "linked_success_event_id": str(attempt.get("event_id", "") if attempt and attempt.get("event_type") == "birth_success" else ""),
                "linked_failure_event_id": str(attempt.get("event_id", "") if attempt and str(attempt.get("event_type", "")).startswith("birth_fail") else ""),
            }
        )
        _apply_v3_event_defaults(
            status,
            generation=int(generation),
            event_type=status_type,
            event_schema_version=event_schema_version,
        )
        rows.append({field: status.get(field, "") for field in EVENT_FIELDS})

        if attempt is None:
            no_attempt = dict(payload)
            no_attempt.update(
                {
                    "event_id": f"{exposure_id}:no_attempt",
                    "event": "selected_parent_no_attempt",
                    "event_type": "selected_parent_no_attempt",
                    "birth_attempted": 0,
                    "birth_attempt_recorded": 0,
                    "birth_success": 0,
                    "birth_failed": 0,
                    "no_attempt_marker": 1,
                }
            )
            _apply_v3_event_defaults(
                no_attempt,
                generation=int(generation),
                event_type="selected_parent_no_attempt",
                event_schema_version=event_schema_version,
            )
            rows.append({field: no_attempt.get(field, "") for field in EVENT_FIELDS})
    return rows


def _birth_attempt_recorded_rows(
    *,
    generation: int,
    birth_events: Iterable[Mapping[str, Any]],
    event_schema_version: int = EVENT_SCHEMA_VERSION,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in birth_events:
        if _safe_int(event.get("birth_attempted", 0), 0) <= 0:
            continue
        event_type = str(event.get("event_type", "") or "")
        attempt = dict(event)
        attempt_id = str(event.get("event_id") or "")
        attempt.update(
            {
                "event_id": f"{attempt_id}:attempt" if attempt_id else "",
                "event": "birth_attempt_recorded",
                "event_type": "birth_attempt_recorded",
                "birth_attempt_recorded": 1,
                "birth_success": 0,
                "birth_failed": 0,
                "linked_attempt_event_id": attempt_id,
                "linked_success_event_id": attempt_id if event_type == "birth_success" else "",
                "linked_failure_event_id": attempt_id if event_type.startswith("birth_fail") else "",
            }
        )
        _apply_v3_event_defaults(
            attempt,
            generation=int(generation),
            event_type="birth_attempt_recorded",
            event_schema_version=event_schema_version,
        )
        rows.append({field: attempt.get(field, "") for field in EVENT_FIELDS})
    return rows


def _scan_action_rows(
    *,
    generation: int,
    scan_events: Iterable[Mapping[str, Any]],
    birth_events: Iterable[Mapping[str, Any]],
    selected_meta: Mapping[int, Mapping[str, Any]],
    event_schema_version: int,
) -> list[dict[str, Any]]:
    if _safe_int(event_schema_version, EVENT_SCHEMA_VERSION) < SCAN_ACTION_EVENT_SCHEMA_VERSION:
        return []
    birth_by_scan_id: dict[str, dict[str, Any]] = {}
    for event in birth_events:
        scan_id = str(event.get("scan_id") or "")
        if scan_id and scan_id not in birth_by_scan_id and _safe_int(event.get("birth_attempted", 0), 0) > 0:
            birth_by_scan_id[scan_id] = dict(event)

    rows: list[dict[str, Any]] = []
    exposure_by_scan_id: dict[str, dict[str, Any]] = {}
    for raw in list(scan_events or []):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        event_type = str(row.get("event_type") or row.get("event") or "")
        if not event_type.startswith("reproduction_scan") and not event_type.startswith("reproduction_action"):
            continue
        parent_id = _safe_int(row.get("parent_id", row.get("parent_agent_id", -1)), -1)
        selected = dict(selected_meta.get(parent_id, {}) or {})
        if selected:
            row["parent_was_selected_parent_this_generation"] = _safe_int(
                row.get("parent_was_selected_parent_this_generation", selected.get("parent_was_selected_parent_this_generation", 0)),
                0,
            )
            row["parent_selected_rank"] = _safe_int(row.get("parent_selected_rank", selected.get("parent_selected_rank", 0)), 0)
            row["parent_selected_rank_bucket"] = str(
                row.get("parent_selected_rank_bucket", selected.get("parent_selected_rank_bucket", "")) or ""
            )
            row["selected_parent_pool_size"] = _safe_int(
                row.get("selected_parent_pool_size", selected.get("selected_parent_pool_size", 0)),
                0,
            )
        if _safe_int(row.get("path2_roz_reporting_enable", 0), 0) > 0:
            row["path2_roz_selected_parent_scan_in_zone"] = int(
                bool(
                    _safe_int(row.get("parent_selected_rank", 0), 0) > 0
                    and _safe_int(row.get("path2_roz_parent_in_zone", 0), 0) > 0
                )
            )
        scan_id = str(row.get("scan_id") or "")
        birth = birth_by_scan_id.get(scan_id)
        if birth is not None:
            birth_event_id = str(birth.get("event_id") or "")
            row["linked_attempt_event_id"] = str(row.get("linked_attempt_event_id") or birth_event_id)
            if str(birth.get("event_type") or "") == "birth_success":
                row["linked_success_event_id"] = str(row.get("linked_success_event_id") or birth_event_id)
            if str(birth.get("event_type") or "").startswith("birth_fail"):
                row["linked_failure_event_id"] = str(row.get("linked_failure_event_id") or birth_event_id)
        _apply_v3_event_defaults(
            row,
            generation=int(generation),
            event_type=event_type,
            event_schema_version=event_schema_version,
        )
        clean = {field: row.get(field, "") for field in EVENT_FIELDS}
        rows.append(clean)
        if event_type == "reproduction_scan_exposure" and scan_id:
            exposure_by_scan_id[scan_id] = clean

    for scan_id, birth in sorted(birth_by_scan_id.items()):
        exposure = exposure_by_scan_id.get(scan_id, birth)
        birth_event_id = str(birth.get("event_id") or "")
        attempt = dict(exposure)
        attempt.update(
            {
                "event_id": f"{scan_id}:scan_attempt_recorded",
                "event": "reproduction_scan_attempt_recorded",
                "event_type": "reproduction_scan_attempt_recorded",
                "birth_attempted": 0,
                "birth_attempt_recorded": 1,
                "birth_success": 0,
                "birth_failed": 0,
                "linked_scan_event_id": str(exposure.get("event_id") or ""),
                "linked_attempt_event_id": birth_event_id,
                "linked_success_event_id": birth_event_id if str(birth.get("event_type") or "") == "birth_success" else "",
                "linked_failure_event_id": birth_event_id if str(birth.get("event_type") or "").startswith("birth_fail") else "",
            }
        )
        _apply_v3_event_defaults(
            attempt,
            generation=int(generation),
            event_type="reproduction_scan_attempt_recorded",
            event_schema_version=event_schema_version,
        )
        rows.append({field: attempt.get(field, "") for field in EVENT_FIELDS})
    return rows


def _counts_from_agents(agents: Iterable[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for agent in list(agents or []):
        if not bool(getattr(agent, "alive", True)):
            continue
        lid = _lineage_id_for_agent(agent)
        if lid:
            counts[lid] += 1
    return dict(sorted(counts.items()))


def _counts_from_contexts(contexts: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in list(contexts or []):
        if not isinstance(item, Mapping):
            continue
        lid = lineage_id_from_context(item)
        if lid:
            counts[lid] += 1
    return dict(sorted(counts.items()))


def _sum_counts(counts: Mapping[str, int]) -> int:
    return int(sum(max(0, int(v)) for v in counts.values()))


def _share(count: int, total: int) -> float:
    return float(count) / float(total) if int(total) > 0 else 0.0


def _entropy(counts: Mapping[str, int]) -> float:
    total = float(_sum_counts(counts))
    if total <= 0.0:
        return 0.0
    ent = 0.0
    for raw in counts.values():
        count = max(0, int(raw))
        if count <= 0:
            continue
        p = float(count) / total
        ent -= p * math.log(p)
    return float(ent)


def _effective(counts: Mapping[str, int]) -> float:
    ent = _entropy(counts)
    return float(math.exp(ent)) if ent > 0.0 else (1.0 if _sum_counts(counts) > 0 else 0.0)


def _top_share(counts: Mapping[str, int], n: int) -> float:
    total = _sum_counts(counts)
    if total <= 0:
        return 0.0
    top = sorted((max(0, int(v)) for v in counts.values()), reverse=True)[: max(1, int(n))]
    return float(sum(top)) / float(total)


def _ratio(num: float, den: float) -> float | None:
    n = _safe_float(num, 0.0)
    d = _safe_float(den, 0.0)
    if d == 0.0:
        return None
    return float(n / d)


def _ratio_value(num: float, den: float) -> float:
    value = _ratio(num, den)
    return -1.0 if value is None else float(value)


def classify_amplification(value: float | None) -> str:
    if value is None:
        return "denominator unavailable"
    v = _safe_float(value, 0.0)
    if v >= 2.0:
        return "collapse-stage amplification"
    if v >= 1.5:
        return "strong amplification"
    if v >= 1.1:
        return "mild amplification"
    return "no amplification"


def _mean_context(contexts: list[Mapping[str, Any]], key: str) -> float:
    vals = [_safe_float(row.get(key), 0.0) for row in contexts if key in row]
    return float(sum(vals) / float(len(vals))) if vals else 0.0


def _max_context(contexts: list[Mapping[str, Any]], key: str) -> float:
    vals = [_safe_float(row.get(key), 0.0) for row in contexts if key in row]
    return float(max(vals)) if vals else 0.0


def _first_coord(contexts: list[Mapping[str, Any]], y_key: str, x_key: str) -> str:
    for row in contexts:
        if y_key in row and x_key in row:
            return f"{_safe_int(row.get(y_key), 0)},{_safe_int(row.get(x_key), 0)}"
    return ""


def _first_string(contexts: list[Mapping[str, Any]], key: str) -> str:
    for row in contexts:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def capture_pre_turnover_snapshot(
    *,
    generation: int,
    alive_start_agents: Iterable[Any],
    alive_after_agents: Iterable[Any],
    eligible_ids: Iterable[int],
    selected_parent_agents: Iterable[Any],
    successful_births: int,
    failed_births: int,
    birth_failure_counts: Mapping[str, int],
    birth_contexts: Iterable[Mapping[str, Any]],
    sim_cfg: Any,
    env_cfg: Any = None,
    birth_event_contexts: Iterable[Mapping[str, Any]] | None = None,
    scan_action_event_contexts: Iterable[Mapping[str, Any]] | None = None,
    parent_fraction: float | None = None,
) -> dict[str, Any]:
    if not enabled(sim_cfg):
        return {}
    validate_config(sim_cfg)
    after_agents = list(alive_after_agents or [])
    eligible_set = {int(x) for x in set(eligible_ids or [])}
    eligible_agents = [agent for agent in after_agents if _agent_id(agent) in eligible_set]
    selected_agents = list(selected_parent_agents or [])
    birth_rows = [dict(x) for x in list(birth_contexts or []) if isinstance(x, Mapping)]
    event_rows = [dict(x) for x in list(birth_event_contexts or birth_rows) if isinstance(x, Mapping)]
    scan_action_rows = [dict(x) for x in list(scan_action_event_contexts or []) if isinstance(x, Mapping)]
    return {
        "schema_version": SCHEMA_VERSION,
        "event_schema_version": configured_event_schema_version(sim_cfg),
        "generation": int(generation),
        "generation_start_alive_counts": _counts_from_agents(alive_start_agents),
        "alive_before_selection_counts": _counts_from_agents(after_agents),
        "eligible_parent_counts": _counts_from_agents(eligible_agents),
        "selected_parent_counts": _counts_from_agents(selected_agents),
        "selected_parent_agent_ids": [_agent_id(a) for a in selected_agents],
        "unique_selected_parent_count": int(len({_agent_id(a) for a in selected_agents})),
        "selected_parent_event_rows": _selected_parent_event_rows(
            generation=int(generation),
            selected_agents=selected_agents,
            run_id="",
            parent_fraction=parent_fraction,
        ),
        "selected_parent_metadata": _selected_parent_metadata(selected_agents),
        "parent_condition_by_agent_id": {
            str(_agent_id(a)): _agent_condition_fields(a)
            for a in after_agents
            if _agent_id(a) >= 0
        },
        "birth_contexts": birth_rows,
        "birth_event_contexts": event_rows,
        "scan_action_event_contexts": scan_action_rows,
        "successful_birth_counts": _counts_from_contexts(birth_rows),
        "successful_births_total": int(successful_births),
        "failed_births_total": int(failed_births),
        "birth_failure_counts": {str(k): int(v) for k, v in sorted(dict(birth_failure_counts or {}).items())},
        "ecology_payoff_generation": _ecology_payoff_generation(generation, env_cfg),
        "parent_fraction": None if parent_fraction is None else _safe_float(parent_fraction, 0.0),
    }


def _ecology_payoff_generation(generation: int, env_cfg: Any) -> int:
    if env_cfg is None:
        return -1
    if bool(getattr(env_cfg, "wdl_delayed_hazard_enable", False)):
        return int(generation)
    if bool(getattr(env_cfg, "patch_world_waste_consequence_enable", False)):
        return int(generation)
    if bool(getattr(env_cfg, "jungle_predator_pressure_enable", False)):
        return int(generation)
    if bool(getattr(env_cfg, "jungle_ecological_recovery_enable", False)):
        return int(generation)
    return -1


def write_generation(
    *,
    run_dir: str,
    sim_cfg: Any,
    generation: int,
    pre_turnover_snapshot: Mapping[str, Any],
    post_refill_agents: Iterable[Any],
    turnover_snapshot: Mapping[str, Any] | None = None,
) -> None:
    if not enabled(sim_cfg):
        return
    validate_config(sim_cfg)
    if not isinstance(pre_turnover_snapshot, Mapping) or not pre_turnover_snapshot:
        return
    writer = LineageReproductionFunnelWriter(
        run_dir,
        incremental_summary_cache_enable=bool(
            getattr(sim_cfg, "lineage_reproduction_funnel_incremental_summary_cache_enable", False)
        ),
    )
    writer.write_generation(
        generation=int(generation),
        pre_turnover_snapshot=dict(pre_turnover_snapshot),
        post_refill_agents=list(post_refill_agents or []),
        turnover_snapshot=dict(turnover_snapshot or {}),
        write_events=bool(getattr(sim_cfg, "lineage_reproduction_funnel_write_events", False)),
        event_schema_version=configured_event_schema_version(sim_cfg),
        write_scan_action_events=scan_action_events_enabled(sim_cfg),
        event_output_profile=str(getattr(sim_cfg, "lineage_reproduction_funnel_event_output_profile", "full") or "full"),
    )


class LineageReproductionFunnelWriter:
    def __init__(self, run_dir: str, *, incremental_summary_cache_enable: bool = False):
        self.run_dir = str(run_dir)
        self._lifecycle_contract = _lifecycle_contract_for_run(self.run_dir)
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.csv_path = os.path.join(self.metrics_dir, "lineage_reproduction_funnel_generation_summary.csv")
        self.jsonl_path = os.path.join(self.metrics_dir, "lineage_reproduction_funnel_generation_summary.jsonl")
        self.run_summary_path = os.path.join(self.metrics_dir, "lineage_reproduction_funnel_run_summary.json")
        self.schema_path = os.path.join(self.metrics_dir, "lineage_reproduction_funnel_schema.json")
        self.events_path = os.path.join(self.metrics_dir, "lineage_reproduction_funnel_events.jsonl")
        self.summary_cache_path = os.path.join(self.metrics_dir, "lineage_reproduction_funnel_summary_cache.json")
        self.incremental_summary_cache_enable = bool(incremental_summary_cache_enable)
        self._ensure_schema()
        self._lifecycle_event_path_key = str(Path(self.events_path).resolve())
        if self._lifecycle_event_path_key not in _LIFECYCLE_EVENT_BYTES:
            try:
                _LIFECYCLE_EVENT_BYTES[self._lifecycle_event_path_key] = int(
                    Path(self.events_path).stat().st_size
                )
            except OSError:
                _LIFECYCLE_EVENT_BYTES[self._lifecycle_event_path_key] = 0
        self._lifecycle_summary_path_key = "|".join(
            str(Path(path).resolve()) for path in (self.csv_path, self.jsonl_path)
        )
        if self._lifecycle_summary_path_key not in _LIFECYCLE_SUMMARY_BYTES:
            _LIFECYCLE_SUMMARY_BYTES[self._lifecycle_summary_path_key] = sum(
                int(Path(path).stat().st_size)
                for path in (self.csv_path, self.jsonl_path)
                if Path(path).is_file()
            )
        self._event_writer_diagnostics = {
            "event_writer_rows": 0,
            "event_writer_batches": 0,
            "event_writer_file_opens": 0,
            "event_writer_writes": 0,
            "event_writer_json_encodes": 0,
            "event_writer_bytes": 0,
            "event_writer_fallbacks": 0,
            "event_writer_errors": 0,
        }

    def _ensure_schema(self) -> None:
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
        if not os.path.exists(self.schema_path):
            schema = {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "event_schema_version": EVENT_SCHEMA_VERSION,
                "supported_event_schema_versions": [EVENT_SCHEMA_VERSION, SCAN_ACTION_EVENT_SCHEMA_VERSION],
                "supported_event_output_profiles": sorted(EVENT_OUTPUT_PROFILES),
                "csv_fields": list(CSV_FIELDS),
                "event_file": "lineage_reproduction_funnel_events.jsonl",
                "event_fields": list(EVENT_FIELDS),
                "scan_action_event_classes_v4": [
                    "reproduction_scan_exposure",
                    "reproduction_scan_eligible",
                    "reproduction_scan_ineligible",
                    "reproduction_scan_no_attempt",
                    "reproduction_scan_attempt_recorded",
                ],
                "scan_action_limitations_v4": [
                    "reproduction_action_offered and reproduction_action_taken are reserved but not emitted until live reproduction is modelled as an explicit policy action.",
                    "v4 scan/action rows require lineage_reproduction_funnel_write_scan_action_events=true and lineage_reproduction_funnel_event_schema_version=4.",
                ],
                "path2_roz_reporting_v1": {
                    "enable_flag": "path2_roz_reporting_enable",
                    "requires": [
                        "lineage_reproduction_funnel_write_events=true",
                        "lineage_reproduction_funnel_write_scan_action_events=true",
                        "lineage_reproduction_funnel_event_schema_version=4",
                    ],
                    "zone_mode": "static_center_manhattan",
                    "mechanics_policy": "reporting-only coordinate overlay; no gating, bonus, reproduction, lineage rescue, or action semantics change",
                },
                "path2_roz_schedule_reporting_v1": {
                    "schedule_modes": sorted(PATH2_ROZ_SCHEDULE_MODES),
                    "mechanics_policy": "reporting-only deterministic schedule metadata; no reproduction mechanics or action semantics change",
                },
                "zero_denominator_policy": "ratio fields use -1.0 in CSV and denominator unavailable classification",
                "mechanics_policy": "reporting-only; no selection, reproduction, cull/refill, ecology, reward, archive, founder-default changes",
            }
            if self._lifecycle_contract is not None:
                schema["lifecycle_binding"] = {
                    "artifact_family": "lineage_reproduction_funnel.events",
                    "event_zone": "scratch",
                    "summary_zone": "durable",
                    "owner_writer": "LineageReproductionFunnelWriter",
                    "lifecycle_profile": self._lifecycle_contract["lifecycle_profile"],
                    "lifecycle_contract_sha256": self._lifecycle_contract[
                        "contract_sha256"
                    ],
                    "output_profile": self._lifecycle_contract["observability"][
                        "output_profile"
                    ],
                }
            with open(self.schema_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(schema, indent=2, sort_keys=True) + "\n")

    def _check_lifecycle_event_write(self, payload: str) -> None:
        contract = self._lifecycle_contract
        if contract is None:
            return
        observability = contract["observability"]
        event_policy = str(observability["debug_event_trace_policy"])
        payload_bytes = len(payload.encode("utf-8"))
        if event_policy == "forbidden":
            _write_lifecycle_violation(
                run_dir=self.run_dir,
                contract=contract,
                artifact_family="lineage_reproduction_funnel.events",
                owner_writer="LineageReproductionFunnelWriter",
                artifact_path="metrics/lineage_reproduction_funnel_events.jsonl",
                code="undeclared_event_trace",
                attempted_bytes=payload_bytes,
                limit_bytes=0,
            )
            raise RuntimeError(
                "lifecycle contract forbids lineage reproduction funnel event JSONL"
            )
        limit = int(contract["storage"]["scratch_max_bytes"])
        ring = observability["ring_buffer"]
        if event_policy == "bounded" and ring["enabled"]:
            limit = min(limit, int(ring["max_bytes"]))
        current = int(_LIFECYCLE_EVENT_BYTES[self._lifecycle_event_path_key])
        if current + payload_bytes > limit:
            _write_lifecycle_violation(
                run_dir=self.run_dir,
                contract=contract,
                artifact_family="lineage_reproduction_funnel.events",
                owner_writer="LineageReproductionFunnelWriter",
                artifact_path="metrics/lineage_reproduction_funnel_events.jsonl",
                code="scratch_quota_exceeded",
                attempted_bytes=current + payload_bytes,
                limit_bytes=limit,
            )
            raise RuntimeError(
                "lifecycle contract scratch quota blocks lineage reproduction funnel event JSONL"
            )
        _LIFECYCLE_EVENT_BYTES[self._lifecycle_event_path_key] = current + payload_bytes

    def _check_lifecycle_summary_write(self, payload_bytes: int) -> None:
        contract = self._lifecycle_contract
        if contract is None:
            return
        limit = int(contract["storage"]["durable_max_bytes"])
        current = int(_LIFECYCLE_SUMMARY_BYTES[self._lifecycle_summary_path_key])
        attempted = current + int(payload_bytes)
        if attempted > limit:
            _write_lifecycle_violation(
                run_dir=self.run_dir,
                contract=contract,
                artifact_family="lineage_reproduction_funnel.generation_summary",
                owner_writer="LineageReproductionFunnelWriter",
                artifact_path="metrics/lineage_reproduction_funnel_generation_summary.*",
                code="durable_quota_exceeded",
                attempted_bytes=attempted,
                limit_bytes=limit,
            )
            raise RuntimeError(
                "lifecycle contract durable quota blocks lineage reproduction funnel summaries"
            )
        _LIFECYCLE_SUMMARY_BYTES[self._lifecycle_summary_path_key] = attempted

    def write_generation(
        self,
        *,
        generation: int,
        pre_turnover_snapshot: Mapping[str, Any],
        post_refill_agents: Iterable[Any],
        turnover_snapshot: Mapping[str, Any],
        write_events: bool,
        event_schema_version: int = EVENT_SCHEMA_VERSION,
        write_scan_action_events: bool = False,
        event_output_profile: str = "full",
    ) -> None:
        existing = self._existing_generations()
        if int(generation) in existing:
            return
        rows = build_generation_rows(
            generation=int(generation),
            pre_turnover_snapshot=pre_turnover_snapshot,
            post_refill_agents=post_refill_agents,
            turnover_snapshot=turnover_snapshot,
        )
        summary_csv_payload: str | None = None
        summary_json_payload: str | None = None
        if self._lifecycle_contract is not None:
            clean_rows = [
                {field: row.get(field, "") for field in CSV_FIELDS} for row in rows
            ]
            csv_buffer = io.StringIO(newline="")
            csv.DictWriter(csv_buffer, fieldnames=CSV_FIELDS).writerows(clean_rows)
            summary_csv_payload = csv_buffer.getvalue()
            summary_json_payload = _stable_jsonl_payload(clean_rows)
            self._check_lifecycle_summary_write(
                len(summary_csv_payload.encode("utf-8"))
                + len(summary_json_payload.encode("utf-8"))
            )
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f_csv, open(self.jsonl_path, "a", encoding="utf-8") as f_jsonl:
            if summary_csv_payload is not None and summary_json_payload is not None:
                f_csv.write(summary_csv_payload)
                f_jsonl.write(summary_json_payload)
            else:
                writer = csv.DictWriter(f_csv, fieldnames=CSV_FIELDS)
                for row in rows:
                    clean = {field: row.get(field, "") for field in CSV_FIELDS}
                    writer.writerow(clean)
                    f_jsonl.write(_stable_json(clean) + "\n")
        cache_ready_for_append = False
        appended_events: list[dict[str, Any]] = []
        if write_events:
            event_output_profile = str(event_output_profile or "full")
            if event_output_profile not in EVENT_OUTPUT_PROFILES:
                raise ValueError(
                    "lineage_reproduction_funnel_event_output_profile must be one of "
                    f"{sorted(EVENT_OUTPUT_PROFILES)}, got {event_output_profile!r}"
                )
            if event_output_profile == "path2_roz_summary_only_v1" and not self.incremental_summary_cache_enable:
                raise ValueError(
                    "lineage_reproduction_funnel_event_output_profile=path2_roz_summary_only_v1 "
                    "requires incremental_summary_cache_enable=true"
                )
            cache_ready_for_append = self._prepare_event_summary_cache_for_append()
            event_schema_version = _safe_int(event_schema_version, EVENT_SCHEMA_VERSION)
            run_id = os.path.basename(os.path.abspath(self.run_dir))
            selected_events = [
                dict(row)
                for row in list(pre_turnover_snapshot.get("selected_parent_event_rows", []) or [])
                if isinstance(row, Mapping)
            ]
            selected_meta = {
                _safe_int(k, -1): dict(v)
                for k, v in dict(pre_turnover_snapshot.get("selected_parent_metadata", {}) or {}).items()
                if isinstance(v, Mapping)
            }
            parent_condition_by_id = {
                _safe_int(k, -1): dict(v)
                for k, v in dict(pre_turnover_snapshot.get("parent_condition_by_agent_id", {}) or {}).items()
                if isinstance(v, Mapping)
            }
            condition_agents = {
                aid: type("_ConditionAgent", (), {"agent_id": aid, "alive": True, "stats": type("_Stats", (), {})()})()
                for aid in parent_condition_by_id
            }
            for aid, fields in parent_condition_by_id.items():
                agent = condition_agents[aid]
                for key, value in fields.items():
                    if key.startswith("parent_"):
                        setattr(agent, key.removeprefix("parent_"), value)
                    else:
                        setattr(agent, key, value)
                try:
                    setattr(agent.stats, "survival_steps", _safe_int(fields.get("parent_age", 0), 0))
                    setattr(agent.stats, "effective_fitness", _safe_float(fields.get("parent_effective_fitness", 0.0), 0.0))
                    setattr(agent.stats, "offspring_count", _safe_int(fields.get("parent_recent_birth_attempt_count", 0), 0))
                except Exception:
                    pass
            raw_birth_events = list(
                pre_turnover_snapshot.get(
                    "birth_event_contexts",
                    pre_turnover_snapshot.get("birth_contexts", []),
                )
                or []
            )
            parent_fraction = pre_turnover_snapshot.get("parent_fraction", None)
            selected_by_parent: dict[int, dict[str, Any]] = {}
            for idx, event in enumerate(selected_events, start=1):
                payload = dict(event)
                payload["run_id"] = run_id
                payload["event_id"] = str(payload.get("event_id") or _selected_exposure_id(int(generation), idx))
                payload["selected_exposure_id"] = str(payload.get("selected_exposure_id") or payload["event_id"])
                _apply_v3_event_defaults(
                    payload,
                    generation=int(generation),
                    event_type=str(payload.get("event_type") or "selected_parent_exposure"),
                    event_schema_version=event_schema_version,
                )
                clean = {field: payload.get(field, "") for field in EVENT_FIELDS}
                selected_events[idx - 1] = clean
                selected_by_parent[_safe_int(clean.get("parent_id", clean.get("parent_agent_id", -1)), -1)] = clean
            normalized_birth_events: list[dict[str, Any]] = []
            offset = len(selected_events)
            for idx, event in enumerate(raw_birth_events, start=1):
                if not isinstance(event, Mapping):
                    continue
                normalized = _normalize_event_row(
                    event,
                    generation=int(generation),
                    event_index=offset + idx,
                    run_id=run_id,
                    selected_meta=selected_meta,
                    agent_by_id=condition_agents,
                    parent_fraction=parent_fraction,
                    event_schema_version=event_schema_version,
                )
                parent_id = _safe_int(normalized.get("parent_id", normalized.get("parent_agent_id", -1)), -1)
                selected = selected_by_parent.get(parent_id)
                if selected is not None:
                    normalized["selected_exposure_id"] = str(selected.get("selected_exposure_id") or selected.get("event_id") or "")
                    _apply_v3_event_defaults(
                        normalized,
                        generation=int(generation),
                        event_type=str(normalized.get("event_type") or ""),
                        event_schema_version=event_schema_version,
                    )
                normalized_birth_events.append({field: normalized.get(field, "") for field in EVENT_FIELDS})
            attempts_by_parent: dict[int, dict[str, Any]] = {}
            for event in normalized_birth_events:
                parent_id = _safe_int(event.get("parent_id", event.get("parent_agent_id", -1)), -1)
                if parent_id >= 0 and parent_id not in attempts_by_parent and _safe_int(event.get("birth_attempted", 0), 0) > 0:
                    attempts_by_parent[parent_id] = event
            for idx, event in enumerate(selected_events):
                parent_id = _safe_int(event.get("parent_id", event.get("parent_agent_id", -1)), -1)
                attempt = attempts_by_parent.get(parent_id)
                if attempt is None:
                    continue
                updated = dict(event)
                updated["linked_attempt_event_id"] = str(attempt.get("event_id") or "")
                if str(attempt.get("event_type", "")) == "birth_success":
                    updated["linked_success_event_id"] = str(attempt.get("event_id") or "")
                if str(attempt.get("event_type", "")).startswith("birth_fail"):
                    updated["linked_failure_event_id"] = str(attempt.get("event_id") or "")
                selected_events[idx] = {field: updated.get(field, "") for field in EVENT_FIELDS}
            events = []
            events.extend(selected_events)
            events.extend(
                _derived_selected_parent_rows(
                    generation=int(generation),
                    selected_events=selected_events,
                    birth_events=normalized_birth_events,
                    event_schema_version=event_schema_version,
                )
            )
            events.extend(
                _birth_attempt_recorded_rows(
                    generation=int(generation),
                    birth_events=normalized_birth_events,
                    event_schema_version=event_schema_version,
                )
            )
            if write_scan_action_events:
                events.extend(
                    _scan_action_rows(
                        generation=int(generation),
                        scan_events=[
                            dict(row)
                            for row in list(pre_turnover_snapshot.get("scan_action_event_contexts", []) or [])
                            if isinstance(row, Mapping)
                        ],
                        birth_events=normalized_birth_events,
                        selected_meta=selected_meta,
                        event_schema_version=event_schema_version,
                    )
                )
            events.extend(normalized_birth_events)
            if events:
                if event_output_profile in {"path2_roz_compact_v1", "path2_roz_summary_only_v1"}:
                    events = [_compact_event_row(event, _PATH2_ROZ_COMPACT_EVENT_FIELDS) for event in events]
                if event_output_profile != "path2_roz_summary_only_v1":
                    payload = _stable_jsonl_payload(events)
                    self._check_lifecycle_event_write(payload)
                    self._event_writer_diagnostics["event_writer_rows"] += len(events)
                    self._event_writer_diagnostics["event_writer_batches"] += 1
                    self._event_writer_diagnostics["event_writer_file_opens"] += 1
                    self._event_writer_diagnostics["event_writer_writes"] += 1
                    self._event_writer_diagnostics["event_writer_json_encodes"] += len(events)
                    self._event_writer_diagnostics["event_writer_bytes"] += len(payload)
                    if self.incremental_summary_cache_enable:
                        cache = _FUNNEL_SUMMARY_CACHE_BY_EVENTS_PATH.get(self.events_path)
                        if cache is not None:
                            diagnostics = cache.setdefault("diagnostics", {})
                            diagnostics["event_writer_rows"] = int(diagnostics.get("event_writer_rows", 0)) + len(events)
                            diagnostics["event_writer_batches"] = int(diagnostics.get("event_writer_batches", 0)) + 1
                            diagnostics["event_writer_file_opens"] = int(diagnostics.get("event_writer_file_opens", 0)) + 1
                            diagnostics["event_writer_writes"] = int(diagnostics.get("event_writer_writes", 0)) + 1
                            diagnostics["event_writer_json_encodes"] = int(diagnostics.get("event_writer_json_encodes", 0)) + len(events)
                            diagnostics["event_writer_bytes"] = int(diagnostics.get("event_writer_bytes", 0)) + len(payload)
                    with open(self.events_path, "a", encoding="utf-8") as f:
                        f.write(payload)
                appended_events = [dict(event) for event in events if isinstance(event, Mapping)]
            self._apply_event_summary_cache_events(
                appended_events,
                cache_ready_for_append=cache_ready_for_append,
                persisted_event_file=event_output_profile != "path2_roz_summary_only_v1",
            )
        self._write_run_summary()

    def _existing_generations(self) -> set[int]:
        if not os.path.exists(self.jsonl_path):
            return set()
        out: set[int] = set()
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    out.add(int(row.get("generation", -1)))
                except Exception:
                    continue
        return out

    def _read_rows(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.jsonl_path):
            return []
        rows = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        rows.sort(key=lambda r: (_safe_int(r.get("generation", 0), 0), str(r.get("lineage_id", ""))))
        return rows

    def _write_run_summary(self) -> None:
        rows = self._read_rows()
        summary = classify_run(rows)
        summary.update(self._cached_event_summary())
        with open(self.run_summary_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def _cached_event_summary(self) -> dict[str, Any]:
        if not self.incremental_summary_cache_enable:
            return self._event_summary()
        cache = self._ensure_event_summary_cache()
        if not self._event_summary_cache_valid(cache):
            diagnostics = dict(cache.get("diagnostics", {}) if isinstance(cache, Mapping) else {})
            diagnostics["invalidations"] = int(diagnostics.get("invalidations", 0)) + 1
            diagnostics["full_scan_fallbacks"] = int(diagnostics.get("full_scan_fallbacks", 0)) + 1
            cache = self._refresh_event_summary_cache(diagnostics=diagnostics)
        diagnostics = cache.setdefault("diagnostics", {})
        diagnostics["hits"] = int(diagnostics.get("hits", 0)) + 1
        diagnostics["run_summary_writes"] = int(diagnostics.get("run_summary_writes", 0)) + 1
        diagnostics["summary_hits"] = int(diagnostics.get("summary_hits", 0)) + 1
        diagnostics["summary_writes"] = int(diagnostics.get("summary_writes", 0)) + 1
        diagnostics["json_decodes_avoided"] = int(diagnostics.get("json_decodes_avoided", 0)) + int(
            cache.get("line_count", 0)
        )
        self._write_summary_cache_sidecar(cache)
        return dict(cache.get("summary", {}))

    def _event_summary_cache_valid(self, cache: Mapping[str, Any]) -> bool:
        stat = _event_file_stat(self.events_path)
        return (
            bool(cache.get("exists", False)) == bool(stat["exists"])
            and int(cache.get("size", 0)) == int(stat["size"])
            and int(cache.get("mtime_ns", 0)) == int(stat["mtime_ns"])
        )

    def _ensure_event_summary_cache(self) -> dict[str, Any]:
        cache = _FUNNEL_SUMMARY_CACHE_BY_EVENTS_PATH.get(self.events_path)
        if cache is not None and self._event_summary_cache_valid(cache):
            return cache
        diagnostics = dict(cache.get("diagnostics", {}) if isinstance(cache, Mapping) else {})
        if cache is not None:
            diagnostics["invalidations"] = int(diagnostics.get("invalidations", 0)) + 1
        return self._refresh_event_summary_cache(diagnostics=diagnostics)

    def _refresh_event_summary_cache(self, *, diagnostics: Mapping[str, Any] | None = None) -> dict[str, Any]:
        summary = self._event_summary()
        stat = _event_file_stat(self.events_path)
        line_stats = _event_file_line_stats(self.events_path)
        accumulators = self._event_summary_accumulators_from_file(self.events_path)
        merged_diagnostics = dict(diagnostics or {})
        if bool(stat["exists"]) and int(stat["size"]) > 0:
            merged_diagnostics["resume_initializations"] = int(merged_diagnostics.get("resume_initializations", 0)) + 1
        merged_diagnostics["enabled"] = True
        merged_diagnostics["initial_scan_lines"] = int(line_stats.get("line_count", 0))
        merged_diagnostics["blank_lines"] = int(line_stats.get("blank_lines", 0))
        merged_diagnostics["malformed_lines"] = int(line_stats.get("malformed_lines", 0))
        merged_diagnostics.setdefault("events_applied", 0)
        merged_diagnostics.setdefault("hits", 0)
        merged_diagnostics.setdefault("invalidations", 0)
        merged_diagnostics.setdefault("full_scan_fallbacks", 0)
        merged_diagnostics.setdefault("run_summary_writes", 0)
        merged_diagnostics.setdefault("summary_hits", 0)
        merged_diagnostics.setdefault("summary_writes", 0)
        merged_diagnostics.setdefault("json_decodes_avoided", 0)
        merged_diagnostics.setdefault("json_encodes_avoided", 0)
        cache = {
            "exists": bool(stat["exists"]),
            "size": int(stat["size"]),
            "mtime_ns": int(stat["mtime_ns"]),
            "line_count": int(line_stats.get("line_count", 0)),
            "summary": summary,
            "accumulators": accumulators,
            "diagnostics": merged_diagnostics,
        }
        _FUNNEL_SUMMARY_CACHE_BY_EVENTS_PATH[self.events_path] = cache
        self._write_summary_cache_sidecar(cache)
        return cache

    def _prepare_event_summary_cache_for_append(self) -> bool:
        if not self.incremental_summary_cache_enable:
            return False
        cache = self._ensure_event_summary_cache()
        return self._event_summary_cache_valid(cache)

    def _apply_event_summary_cache_events(
        self,
        events: Iterable[Mapping[str, Any]],
        *,
        cache_ready_for_append: bool,
        persisted_event_file: bool = True,
    ) -> None:
        if not self.incremental_summary_cache_enable:
            return
        event_rows = [dict(event) for event in list(events or []) if isinstance(event, Mapping)]
        diagnostics = dict(
            _FUNNEL_SUMMARY_CACHE_BY_EVENTS_PATH.get(self.events_path, {}).get("diagnostics", {})
            if isinstance(_FUNNEL_SUMMARY_CACHE_BY_EVENTS_PATH.get(self.events_path), Mapping)
            else {}
        )
        if not cache_ready_for_append:
            diagnostics["full_scan_fallbacks"] = int(diagnostics.get("full_scan_fallbacks", 0)) + 1
        diagnostics["events_applied"] = int(diagnostics.get("events_applied", 0)) + len(event_rows)
        diagnostics["json_decodes_avoided"] = int(diagnostics.get("json_decodes_avoided", 0)) + len(event_rows)
        cache = _FUNNEL_SUMMARY_CACHE_BY_EVENTS_PATH.get(self.events_path)
        if cache_ready_for_append and isinstance(cache, Mapping) and self._merge_event_summary_cache_events(cache, event_rows, diagnostics=diagnostics):
            return
        self._refresh_event_summary_cache(diagnostics=diagnostics)

    def _event_summary_for_rows(self, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        event_rows = [dict(row) for row in list(rows or []) if isinstance(row, Mapping)]
        if not event_rows:
            old_path = self.events_path
            try:
                self.events_path = os.path.join(tempfile.mkdtemp(prefix="lineage-empty-events-"), "events.jsonl")
                return self._event_summary()
            finally:
                self.events_path = old_path
        old_path = self.events_path
        with tempfile.TemporaryDirectory(prefix="lineage-event-delta-") as tmp:
            tmp_path = os.path.join(tmp, "events.jsonl")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(_stable_jsonl_payload(event_rows))
            try:
                self.events_path = tmp_path
                return self._event_summary()
            finally:
                self.events_path = old_path

    def _event_summary_accumulators_from_file(self, path: str) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        row = json.loads(text)
                    except Exception:
                        continue
                    if isinstance(row, Mapping):
                        rows.append(dict(row))
        return self._event_summary_accumulators_from_rows(rows)

    def _event_summary_accumulators_from_rows(self, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        schedule_zone_ids: set[str] = set()
        first_active_schedule_zone_id = ""
        last_active_schedule_zone_id = ""
        weather_active_generations: set[int] = set()
        weather_event_ids: set[str] = set()
        weather_on_distance_sum = 0.0
        weather_on_distance_count = 0
        weather_off_distance_sum = 0.0
        weather_off_distance_count = 0
        weather_on_distance_max = 0.0
        weather_off_distance_max = 0.0
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            event_type = str(row.get("event_type", "") or _event_type_for(row))
            if _safe_int(row.get("path2_roz_reporting_enable", 0), 0) > 0 and _safe_int(
                row.get("path2_roz_window_active", 0), 0
            ) > 0:
                zone_id = str(row.get("path2_roz_zone_id") or "")
                if zone_id:
                    schedule_zone_ids.add(zone_id)
                    if not first_active_schedule_zone_id:
                        first_active_schedule_zone_id = zone_id
                    last_active_schedule_zone_id = zone_id
            if event_type == "birth_success":
                weather_active = _safe_int(
                    row.get("child_mutation_weather_active", row.get("mutation_weather_active", 0)),
                    0,
                ) > 0
                distance_mean = _safe_float(row.get("parent_child_genome_distance_mean", 0.0), 0.0)
                distance_max = _safe_float(row.get("parent_child_genome_distance_max", 0.0), 0.0)
                if weather_active:
                    weather_on_distance_sum += distance_mean
                    weather_on_distance_count += 1
                    weather_on_distance_max = max(weather_on_distance_max, distance_max)
                    gen = _safe_int(row.get("mutation_weather_generation", row.get("generation", -1)), -1)
                    if gen >= 0:
                        weather_active_generations.add(gen)
                    event_id = str(row.get("child_mutation_weather_event_id", row.get("mutation_weather_event_id", "")) or "")
                    if event_id:
                        weather_event_ids.add(event_id)
                else:
                    weather_off_distance_sum += distance_mean
                    weather_off_distance_count += 1
                    weather_off_distance_max = max(weather_off_distance_max, distance_max)
        return {
            "schedule_zone_ids": sorted(schedule_zone_ids),
            "first_active_schedule_zone_id": first_active_schedule_zone_id,
            "last_active_schedule_zone_id": last_active_schedule_zone_id,
            "weather_active_generations": sorted(weather_active_generations),
            "weather_event_ids": sorted(weather_event_ids),
            "weather_on_distance_sum": float(weather_on_distance_sum),
            "weather_on_distance_count": int(weather_on_distance_count),
            "weather_off_distance_sum": float(weather_off_distance_sum),
            "weather_off_distance_count": int(weather_off_distance_count),
            "weather_on_distance_max": float(weather_on_distance_max),
            "weather_off_distance_max": float(weather_off_distance_max),
        }

    def _merge_event_summary_cache_events(
        self,
        cache: Mapping[str, Any],
        rows: Iterable[Mapping[str, Any]],
        *,
        diagnostics: Mapping[str, Any],
    ) -> bool:
        event_rows = [dict(row) for row in list(rows or []) if isinstance(row, Mapping)]
        if not event_rows:
            return False
        try:
            delta = self._event_summary_for_rows(event_rows)
            delta_acc = self._event_summary_accumulators_from_rows(event_rows)
            summary = dict(cache.get("summary", {}) if isinstance(cache, Mapping) else {})
            base_acc = dict(cache.get("accumulators", {}) if isinstance(cache.get("accumulators"), Mapping) else {})
            merged_acc = self._merge_event_summary_accumulators(base_acc, delta_acc)
            skip = {
                "event_schema_version",
                "event_type_counts",
                "failure_reason_group_counts",
                "lro4_offspring_dispersal_trigger_reason_counts",
                "path2_roz_schedule_distinct_zone_count",
                "weather_active_generation_count",
                "weather_event_count",
                "parent_child_genome_distance_mean_weather_off",
                "parent_child_genome_distance_mean_weather_on",
                "parent_child_genome_distance_max_weather_off",
                "parent_child_genome_distance_max_weather_on",
            }
            for key, value in delta.items():
                if key in skip:
                    continue
                if isinstance(value, (int, float)) and isinstance(summary.get(key, 0), (int, float)):
                    summary[key] = summary.get(key, 0) + value
            summary["event_schema_version"] = max(
                _safe_int(summary.get("event_schema_version", EVENT_SCHEMA_VERSION), EVENT_SCHEMA_VERSION),
                _safe_int(delta.get("event_schema_version", EVENT_SCHEMA_VERSION), EVENT_SCHEMA_VERSION),
            )
            for key in ("event_type_counts", "failure_reason_group_counts", "lro4_offspring_dispersal_trigger_reason_counts"):
                counts = Counter(dict(summary.get(key, {}) or {}))
                counts.update(dict(delta.get(key, {}) or {}))
                summary[key] = dict(sorted(counts.items()))
            base_last_zone = str(base_acc.get("last_active_schedule_zone_id", "") or "")
            delta_first_zone = str(delta_acc.get("first_active_schedule_zone_id", "") or "")
            if base_last_zone and delta_first_zone and base_last_zone != delta_first_zone:
                summary["path2_roz_schedule_zone_switch_count"] = int(
                    summary.get("path2_roz_schedule_zone_switch_count", 0)
                ) + 1
            summary["path2_roz_schedule_distinct_zone_count"] = len(merged_acc.get("schedule_zone_ids", []) or [])
            summary["weather_active_generation_count"] = len(merged_acc.get("weather_active_generations", []) or [])
            summary["weather_event_count"] = len(merged_acc.get("weather_event_ids", []) or [])
            on_count = int(merged_acc.get("weather_on_distance_count", 0) or 0)
            off_count = int(merged_acc.get("weather_off_distance_count", 0) or 0)
            summary["parent_child_genome_distance_mean_weather_on"] = (
                float(merged_acc.get("weather_on_distance_sum", 0.0)) / float(on_count) if on_count else 0.0
            )
            summary["parent_child_genome_distance_mean_weather_off"] = (
                float(merged_acc.get("weather_off_distance_sum", 0.0)) / float(off_count) if off_count else 0.0
            )
            summary["parent_child_genome_distance_max_weather_on"] = float(merged_acc.get("weather_on_distance_max", 0.0))
            summary["parent_child_genome_distance_max_weather_off"] = float(merged_acc.get("weather_off_distance_max", 0.0))
            stat = _event_file_stat(self.events_path)
            line_count = int(cache.get("line_count", 0)) + len(event_rows)
            merged_cache = {
                "exists": bool(stat["exists"]),
                "size": int(stat["size"]),
                "mtime_ns": int(stat["mtime_ns"]),
                "line_count": int(line_count),
                "summary": summary,
                "accumulators": merged_acc,
                "diagnostics": dict(diagnostics),
            }
            _FUNNEL_SUMMARY_CACHE_BY_EVENTS_PATH[self.events_path] = merged_cache
            self._write_summary_cache_sidecar(merged_cache)
            return True
        except Exception:
            return False

    def _merge_event_summary_accumulators(
        self,
        base: Mapping[str, Any],
        delta: Mapping[str, Any],
    ) -> dict[str, Any]:
        schedule_zone_ids = set(str(x) for x in list(base.get("schedule_zone_ids", []) or []))
        schedule_zone_ids.update(str(x) for x in list(delta.get("schedule_zone_ids", []) or []))
        weather_active_generations = set(int(x) for x in list(base.get("weather_active_generations", []) or []))
        weather_active_generations.update(int(x) for x in list(delta.get("weather_active_generations", []) or []))
        weather_event_ids = set(str(x) for x in list(base.get("weather_event_ids", []) or []))
        weather_event_ids.update(str(x) for x in list(delta.get("weather_event_ids", []) or []))
        return {
            "schedule_zone_ids": sorted(schedule_zone_ids),
            "first_active_schedule_zone_id": str(
                base.get("first_active_schedule_zone_id", "") or delta.get("first_active_schedule_zone_id", "") or ""
            ),
            "last_active_schedule_zone_id": str(
                delta.get("last_active_schedule_zone_id", "") or base.get("last_active_schedule_zone_id", "") or ""
            ),
            "weather_active_generations": sorted(weather_active_generations),
            "weather_event_ids": sorted(weather_event_ids),
            "weather_on_distance_sum": float(base.get("weather_on_distance_sum", 0.0) or 0.0)
            + float(delta.get("weather_on_distance_sum", 0.0) or 0.0),
            "weather_on_distance_count": int(base.get("weather_on_distance_count", 0) or 0)
            + int(delta.get("weather_on_distance_count", 0) or 0),
            "weather_off_distance_sum": float(base.get("weather_off_distance_sum", 0.0) or 0.0)
            + float(delta.get("weather_off_distance_sum", 0.0) or 0.0),
            "weather_off_distance_count": int(base.get("weather_off_distance_count", 0) or 0)
            + int(delta.get("weather_off_distance_count", 0) or 0),
            "weather_on_distance_max": max(
                float(base.get("weather_on_distance_max", 0.0) or 0.0),
                float(delta.get("weather_on_distance_max", 0.0) or 0.0),
            ),
            "weather_off_distance_max": max(
                float(base.get("weather_off_distance_max", 0.0) or 0.0),
                float(delta.get("weather_off_distance_max", 0.0) or 0.0),
            ),
        }

    def _write_summary_cache_sidecar(self, cache: Mapping[str, Any]) -> None:
        if not self.incremental_summary_cache_enable:
            return
        diagnostics = dict(cache.get("diagnostics", {}) if isinstance(cache, Mapping) else {})
        sidecar = {
            "enabled": True,
            "events_file": os.path.basename(self.events_path),
            "events_file_exists": bool(cache.get("exists", False)),
            "events_file_size_bytes": int(cache.get("size", 0)),
            "events_file_mtime_ns": int(cache.get("mtime_ns", 0)),
            "line_count": int(cache.get("line_count", 0)),
            "diagnostics": diagnostics,
            "event_writer_diagnostics": dict(self._event_writer_diagnostics),
            "summary": dict(cache.get("summary", {}) if isinstance(cache, Mapping) else {}),
            "mechanics_policy": "summary cache only; no selection, reproduction, cull/refill, ecology, reward, archive, or event byte-order changes",
        }
        with open(self.summary_cache_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")

    def _event_summary(self) -> dict[str, Any]:
        out = {
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "selected_parent_exposure_rows": 0,
            "selected_parent_eligible_rows": 0,
            "selected_parent_ineligible_rows": 0,
            "selected_parent_no_attempt_rows": 0,
            "birth_attempt_rows": 0,
            "birth_attempt_recorded_rows": 0,
            "birth_success_rows": 0,
            "birth_fail_rows": 0,
            "birth_fail_no_space_rows": 0,
            "birth_fail_non_space_rows": 0,
            "birth_fail_unknown_rows": 0,
            "reproduction_scan_exposure_rows": 0,
            "reproduction_scan_eligible_rows": 0,
            "reproduction_scan_ineligible_rows": 0,
            "reproduction_scan_no_attempt_rows": 0,
            "reproduction_scan_attempt_recorded_rows": 0,
            "reproduction_action_offered_rows": 0,
            "reproduction_action_taken_rows": 0,
            "reproduction_action_not_taken_rows": 0,
            "rows_with_parent_condition": 0,
            "rows_with_scan_id": 0,
            "rows_with_eligibility_reason": 0,
            "rows_with_mate_context": 0,
            "rows_with_linked_attempt_event_id": 0,
            "rows_with_accounting_window": 0,
            "rows_with_parent_lineage": 0,
            "rows_with_mate_lineage": 0,
            "rows_with_selected_parent_rank": 0,
            "path2_roz_event_rows": 0,
            "path2_roz_window_active_rows": 0,
            "path2_roz_scan_exposure_rows": 0,
            "path2_roz_scan_eligible_rows": 0,
            "path2_roz_scan_ineligible_rows": 0,
            "path2_roz_scan_no_attempt_rows": 0,
            "path2_roz_scan_attempt_recorded_rows": 0,
            "path2_roz_birth_success_rows": 0,
            "path2_roz_birth_fail_rows": 0,
            "path2_roz_parent_in_zone_rows": 0,
            "path2_roz_mate_in_zone_rows": 0,
            "path2_roz_child_in_zone_rows": 0,
            "path2_roz_birth_in_zone_rows": 0,
            "path2_roz_selected_parent_scan_in_zone_rows": 0,
            "path2_roz_schedule_window_active_rows": 0,
            "path2_roz_schedule_zone_switch_count": 0,
            "path2_roz_schedule_distinct_zone_count": 0,
            "path2_roz_schedule_parent_in_zone_rows": 0,
            "path2_roz_schedule_selected_parent_scan_in_zone_rows": 0,
            "lro1_event_rows": 0,
            "lro1_enabled_rows": 0,
            "lro1_disabled_rows": 0,
            "lro1_disabled_leakage_rows": 0,
            "lro1_multiplier_lt1_rows": 0,
            "lro1_multiplier_eq1_rows": 0,
            "lro1_invalid_context_rows": 0,
            "lro1_neutral_context_rows": 0,
            "lro1_scan_exposure_rows": 0,
            "lro1_scan_eligible_rows": 0,
            "lro1_scan_no_attempt_rows": 0,
            "lro1_birth_success_rows": 0,
            "lro1_birth_fail_rows": 0,
            "lro4_offspring_dispersal_event_rows": 0,
            "lro4_offspring_dispersal_enabled_rows": 0,
            "lro4_offspring_dispersal_disabled_rows": 0,
            "lro4_offspring_dispersal_disabled_leakage_rows": 0,
            "lro4_offspring_dispersal_triggered_rows": 0,
            "lro4_offspring_dispersal_expanded_candidate_count_sum": 0,
            "lro4_offspring_dispersal_selected_from_expanded_rows": 0,
            "lro4_offspring_dispersal_trigger_reason_counts": {},
            "weather_active_generation_count": 0,
            "weather_event_count": 0,
            "offspring_affected_by_weather_count": 0,
            "parent_child_genome_distance_mean_weather_off": 0.0,
            "parent_child_genome_distance_mean_weather_on": 0.0,
            "parent_child_genome_distance_max_weather_off": 0.0,
            "parent_child_genome_distance_max_weather_on": 0.0,
            "memory_field_delta_events_weather_on": 0,
            "prediction_field_delta_events_weather_on": 0,
            "module_gate_delta_events_weather_on": 0,
            "event_type_counts": {},
            "failure_reason_group_counts": {},
        }
        if not os.path.exists(self.events_path):
            return out
        event_counts: Counter[str] = Counter()
        group_counts: Counter[str] = Counter()
        schedule_zone_ids: set[str] = set()
        weather_active_generations: set[int] = set()
        weather_event_ids: set[str] = set()
        weather_on_distances: list[float] = []
        weather_off_distances: list[float] = []
        weather_on_max_distances: list[float] = []
        weather_off_max_distances: list[float] = []
        last_active_schedule_zone_id = ""
        max_event_schema_version = EVENT_SCHEMA_VERSION
        with open(self.events_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                event_type = str(row.get("event_type", "") or _event_type_for(row))
                max_event_schema_version = max(
                    max_event_schema_version,
                    _safe_int(row.get("event_schema_version", EVENT_SCHEMA_VERSION), EVENT_SCHEMA_VERSION),
                )
                group = str(row.get("failure_reason_group", "") or "")
                event_counts[event_type] += 1
                if group:
                    group_counts[group] += 1
                if event_type == "selected_parent_exposure":
                    out["selected_parent_exposure_rows"] += 1
                if event_type == "selected_parent_eligible":
                    out["selected_parent_eligible_rows"] += 1
                if event_type == "selected_parent_ineligible":
                    out["selected_parent_ineligible_rows"] += 1
                if event_type == "selected_parent_no_attempt":
                    out["selected_parent_no_attempt_rows"] += 1
                if _safe_int(row.get("birth_attempted", 0), 0) > 0:
                    out["birth_attempt_rows"] += 1
                if event_type == "birth_attempt_recorded":
                    out["birth_attempt_recorded_rows"] += 1
                if event_type == "birth_success":
                    out["birth_success_rows"] += 1
                    weather_active = _safe_int(row.get("child_mutation_weather_active", row.get("mutation_weather_active", 0)), 0) > 0
                    distance_mean = _safe_float(row.get("parent_child_genome_distance_mean", 0.0), 0.0)
                    distance_max = _safe_float(row.get("parent_child_genome_distance_max", 0.0), 0.0)
                    if weather_active:
                        out["offspring_affected_by_weather_count"] += 1
                        weather_on_distances.append(distance_mean)
                        weather_on_max_distances.append(distance_max)
                        gen = _safe_int(row.get("mutation_weather_generation", row.get("generation", -1)), -1)
                        if gen >= 0:
                            weather_active_generations.add(gen)
                        event_id = str(row.get("child_mutation_weather_event_id", row.get("mutation_weather_event_id", "")) or "")
                        if event_id:
                            weather_event_ids.add(event_id)
                        if _safe_float(row.get("parent_child_genome_memory_field_delta", 0.0), 0.0) > 0.0:
                            out["memory_field_delta_events_weather_on"] += 1
                        if _safe_float(row.get("parent_child_genome_prediction_field_delta", 0.0), 0.0) > 0.0:
                            out["prediction_field_delta_events_weather_on"] += 1
                        if _safe_float(row.get("parent_child_genome_module_gate_delta", 0.0), 0.0) > 0.0:
                            out["module_gate_delta_events_weather_on"] += 1
                    else:
                        weather_off_distances.append(distance_mean)
                        weather_off_max_distances.append(distance_max)
                if event_type.startswith("birth_fail"):
                    out["birth_fail_rows"] += 1
                if event_type == "birth_fail_no_space":
                    out["birth_fail_no_space_rows"] += 1
                if event_type == "birth_fail_non_space":
                    out["birth_fail_non_space_rows"] += 1
                if event_type == "birth_fail_unknown" or group == "unknown":
                    out["birth_fail_unknown_rows"] += 1
                if event_type == "reproduction_scan_exposure":
                    out["reproduction_scan_exposure_rows"] += 1
                if event_type == "reproduction_scan_eligible":
                    out["reproduction_scan_eligible_rows"] += 1
                if event_type == "reproduction_scan_ineligible":
                    out["reproduction_scan_ineligible_rows"] += 1
                if event_type == "reproduction_scan_no_attempt":
                    out["reproduction_scan_no_attempt_rows"] += 1
                if event_type == "reproduction_scan_attempt_recorded":
                    out["reproduction_scan_attempt_recorded_rows"] += 1
                if event_type == "reproduction_action_offered":
                    out["reproduction_action_offered_rows"] += 1
                if event_type == "reproduction_action_taken":
                    out["reproduction_action_taken_rows"] += 1
                if event_type == "reproduction_action_not_taken":
                    out["reproduction_action_not_taken_rows"] += 1
                if row.get("parent_energy", "") not in ("", None):
                    out["rows_with_parent_condition"] += 1
                if str(row.get("scan_id", "") or ""):
                    out["rows_with_scan_id"] += 1
                if str(row.get("parent_ineligible_reason_group", "") or ""):
                    out["rows_with_eligibility_reason"] += 1
                if _safe_int(row.get("mate_available", 0), 0) > 0 or str(row.get("mate_block_reason", "") or ""):
                    out["rows_with_mate_context"] += 1
                if str(row.get("linked_attempt_event_id", "") or ""):
                    out["rows_with_linked_attempt_event_id"] += 1
                if str(row.get("accounting_window_rule", "") or ""):
                    out["rows_with_accounting_window"] += 1
                if str(row.get("parent_biological_lineage_id", "") or row.get("parent_lineage_id", "")):
                    out["rows_with_parent_lineage"] += 1
                if str(row.get("mate_biological_lineage_id", "") or row.get("mate_lineage_id", "")):
                    out["rows_with_mate_lineage"] += 1
                if _safe_int(row.get("parent_selected_rank", 0), 0) > 0:
                    out["rows_with_selected_parent_rank"] += 1
                if row.get("lro1_reproductive_opportunity_enable", "") not in ("", None):
                    out["lro1_event_rows"] += 1
                    lro1_enabled = _safe_int(row.get("lro1_reproductive_opportunity_enable", 0), 0) > 0
                    lro1_multiplier = _safe_float(row.get("lro1_opportunity_multiplier", 1.0), 1.0)
                    lro1_score = _safe_float(row.get("lro1_opportunity_score_parent_site", 0.0), 0.0)
                    if lro1_enabled:
                        out["lro1_enabled_rows"] += 1
                    else:
                        out["lro1_disabled_rows"] += 1
                        if abs(lro1_multiplier - 1.0) > 1e-12 or abs(lro1_score) > 1e-12:
                            out["lro1_disabled_leakage_rows"] += 1
                    if lro1_multiplier < 1.0:
                        out["lro1_multiplier_lt1_rows"] += 1
                    if abs(lro1_multiplier - 1.0) <= 1e-12:
                        out["lro1_multiplier_eq1_rows"] += 1
                    if _safe_int(row.get("lro1_opportunity_invalid_context", 0), 0) > 0:
                        out["lro1_invalid_context_rows"] += 1
                    if _safe_int(row.get("lro1_opportunity_neutral_context", 0), 0) > 0:
                        out["lro1_neutral_context_rows"] += 1
                    if event_type == "reproduction_scan_exposure":
                        out["lro1_scan_exposure_rows"] += 1
                    if event_type == "reproduction_scan_eligible":
                        out["lro1_scan_eligible_rows"] += 1
                    if event_type == "reproduction_scan_no_attempt":
                        out["lro1_scan_no_attempt_rows"] += 1
                    if event_type == "birth_success":
                        out["lro1_birth_success_rows"] += 1
                    if event_type.startswith("birth_fail"):
                        out["lro1_birth_fail_rows"] += 1
                if row.get("lro4_offspring_dispersal_enable", "") not in ("", None):
                    out["lro4_offspring_dispersal_event_rows"] += 1
                    lro4_enabled = _safe_int(row.get("lro4_offspring_dispersal_enable", 0), 0) > 0
                    if lro4_enabled:
                        out["lro4_offspring_dispersal_enabled_rows"] += 1
                    else:
                        out["lro4_offspring_dispersal_disabled_rows"] += 1
                    if _safe_int(row.get("lro4_offspring_dispersal_disabled_leakage_marker", 0), 0) > 0:
                        out["lro4_offspring_dispersal_disabled_leakage_rows"] += 1
                    if _safe_int(row.get("lro4_offspring_dispersal_triggered", 0), 0) > 0:
                        out["lro4_offspring_dispersal_triggered_rows"] += 1
                        reason = str(row.get("lro4_offspring_dispersal_trigger_reason", "") or "unknown")
                        out["lro4_offspring_dispersal_trigger_reason_counts"][reason] = (
                            int(out["lro4_offspring_dispersal_trigger_reason_counts"].get(reason, 0)) + 1
                        )
                    out["lro4_offspring_dispersal_expanded_candidate_count_sum"] += _safe_int(
                        row.get("lro4_offspring_dispersal_expanded_candidate_count", 0),
                        0,
                    )
                    if _safe_int(row.get("lro4_offspring_dispersal_selected_from_expanded", 0), 0) > 0:
                        out["lro4_offspring_dispersal_selected_from_expanded_rows"] += 1
                if _safe_int(row.get("path2_roz_reporting_enable", 0), 0) > 0:
                    out["path2_roz_event_rows"] += 1
                    if _safe_int(row.get("path2_roz_window_active", 0), 0) > 0:
                        out["path2_roz_window_active_rows"] += 1
                        out["path2_roz_schedule_window_active_rows"] += 1
                        zone_id = str(row.get("path2_roz_zone_id") or "")
                        if zone_id:
                            schedule_zone_ids.add(zone_id)
                            if last_active_schedule_zone_id and zone_id != last_active_schedule_zone_id:
                                out["path2_roz_schedule_zone_switch_count"] += 1
                            last_active_schedule_zone_id = zone_id
                    if event_type == "reproduction_scan_exposure":
                        out["path2_roz_scan_exposure_rows"] += 1
                    if event_type == "reproduction_scan_eligible":
                        out["path2_roz_scan_eligible_rows"] += 1
                    if event_type == "reproduction_scan_ineligible":
                        out["path2_roz_scan_ineligible_rows"] += 1
                    if event_type == "reproduction_scan_no_attempt":
                        out["path2_roz_scan_no_attempt_rows"] += 1
                    if event_type == "reproduction_scan_attempt_recorded":
                        out["path2_roz_scan_attempt_recorded_rows"] += 1
                    if event_type == "birth_success":
                        out["path2_roz_birth_success_rows"] += 1
                    if event_type.startswith("birth_fail"):
                        out["path2_roz_birth_fail_rows"] += 1
                    if _safe_int(row.get("path2_roz_parent_in_zone", 0), 0) > 0:
                        out["path2_roz_parent_in_zone_rows"] += 1
                        out["path2_roz_schedule_parent_in_zone_rows"] += 1
                    if _safe_int(row.get("path2_roz_mate_in_zone", 0), 0) > 0:
                        out["path2_roz_mate_in_zone_rows"] += 1
                    if _safe_int(row.get("path2_roz_child_in_zone", 0), 0) > 0:
                        out["path2_roz_child_in_zone_rows"] += 1
                    if _safe_int(row.get("path2_roz_birth_in_zone", 0), 0) > 0:
                        out["path2_roz_birth_in_zone_rows"] += 1
                    if _safe_int(row.get("path2_roz_selected_parent_scan_in_zone", 0), 0) > 0:
                        out["path2_roz_selected_parent_scan_in_zone_rows"] += 1
                        out["path2_roz_schedule_selected_parent_scan_in_zone_rows"] += 1
        out["event_type_counts"] = dict(sorted(event_counts.items()))
        out["failure_reason_group_counts"] = dict(sorted(group_counts.items()))
        out["lro4_offspring_dispersal_trigger_reason_counts"] = dict(
            sorted(out["lro4_offspring_dispersal_trigger_reason_counts"].items())
        )
        out["event_schema_version"] = max_event_schema_version
        out["path2_roz_schedule_distinct_zone_count"] = len(schedule_zone_ids)
        out["weather_active_generation_count"] = len(weather_active_generations)
        out["weather_event_count"] = len(weather_event_ids)
        out["parent_child_genome_distance_mean_weather_off"] = (
            sum(weather_off_distances) / float(len(weather_off_distances)) if weather_off_distances else 0.0
        )
        out["parent_child_genome_distance_mean_weather_on"] = (
            sum(weather_on_distances) / float(len(weather_on_distances)) if weather_on_distances else 0.0
        )
        out["parent_child_genome_distance_max_weather_off"] = max(weather_off_max_distances) if weather_off_max_distances else 0.0
        out["parent_child_genome_distance_max_weather_on"] = max(weather_on_max_distances) if weather_on_max_distances else 0.0
        return out


def _reset_lineage_reproduction_funnel_summary_caches_for_tests() -> None:
    _FUNNEL_SUMMARY_CACHE_BY_EVENTS_PATH.clear()


def _event_file_stat(path: str) -> dict[str, Any]:
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return {"exists": False, "size": 0, "mtime_ns": 0}
    return {"exists": True, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _event_file_line_stats(path: str) -> dict[str, int]:
    out = {"line_count": 0, "blank_lines": 0, "malformed_lines": 0}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                out["blank_lines"] += 1
                continue
            try:
                json.loads(text)
            except Exception:
                out["malformed_lines"] += 1
                continue
            out["line_count"] += 1
    return out


def build_generation_rows(
    *,
    generation: int,
    pre_turnover_snapshot: Mapping[str, Any],
    post_refill_agents: Iterable[Any],
    turnover_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    alive_counts = dict(pre_turnover_snapshot.get("alive_before_selection_counts", {}) or {})
    eligible_counts = dict(pre_turnover_snapshot.get("eligible_parent_counts", {}) or {})
    selected_counts = dict(pre_turnover_snapshot.get("selected_parent_counts", {}) or {})
    offspring_counts = dict(pre_turnover_snapshot.get("successful_birth_counts", {}) or {})
    post_refill_counts = _counts_from_agents(post_refill_agents)
    pre_cull_counts = dict(turnover_snapshot.get("pre_cull_alive_counts", alive_counts) or alive_counts)
    post_cull_counts = dict(turnover_snapshot.get("post_cull_alive_counts", {}) or {})
    culled_counts = dict(turnover_snapshot.get("culled_counts", {}) or {})
    refill_counts = dict(turnover_snapshot.get("refill_offspring_counts", {}) or {})
    refill_parent_source_counts = dict(turnover_snapshot.get("refill_parent_source_counts", {}) or {})
    birth_failures = dict(pre_turnover_snapshot.get("birth_failure_counts", {}) or {})
    birth_contexts = [dict(x) for x in list(pre_turnover_snapshot.get("birth_contexts", []) or []) if isinstance(x, Mapping)]

    total_alive = _sum_counts(alive_counts)
    total_eligible = _sum_counts(eligible_counts)
    total_selected = _sum_counts(selected_counts)
    total_offspring = _sum_counts(offspring_counts)
    total_pre_cull = _sum_counts(pre_cull_counts)
    total_post_cull = _sum_counts(post_cull_counts)
    total_post_refill = _sum_counts(post_refill_counts)
    total_culled = _sum_counts(culled_counts)
    lineages = sorted(
        set(alive_counts)
        | set(eligible_counts)
        | set(selected_counts)
        | set(offspring_counts)
        | set(post_refill_counts)
        | set(culled_counts)
        | set(refill_counts)
    )
    selected_entropy = _entropy(selected_counts)
    offspring_entropy = _entropy(offspring_counts)
    rows: list[dict[str, Any]] = []
    rolling_nonzero_birth = int(sum(1 for v in offspring_counts.values() if int(v) > 0))
    rolling_nonzero_selected = int(sum(1 for v in selected_counts.values() if int(v) > 0))
    for lineage_id in lineages:
        lineage_contexts = [row for row in birth_contexts if lineage_id_from_context(row) == lineage_id]
        alive_count = int(alive_counts.get(lineage_id, 0))
        eligible_count = int(eligible_counts.get(lineage_id, 0))
        selected_count = int(selected_counts.get(lineage_id, 0))
        offspring_count = int(offspring_counts.get(lineage_id, 0))
        post_refill_count = int(post_refill_counts.get(lineage_id, 0))
        pre_cull_count = int(pre_cull_counts.get(lineage_id, 0))
        post_cull_count = int(post_cull_counts.get(lineage_id, 0))
        culled_count = int(culled_counts.get(lineage_id, 0))
        alive_share = _share(alive_count, total_alive)
        eligible_share = _share(eligible_count, total_eligible)
        selected_share = _share(selected_count, total_selected)
        offspring_share = _share(offspring_count, total_offspring)
        pre_cull_share = _share(pre_cull_count, total_pre_cull)
        post_cull_share = _share(post_cull_count, total_post_cull)
        post_refill_share = _share(post_refill_count, total_post_refill)
        birth_attempts = int(sum(_safe_int(row.get("birth_attempt_count", 0), 0) for row in lineage_contexts))
        success_total = int(pre_turnover_snapshot.get("successful_births_total", 0) or 0)
        failed_total = int(pre_turnover_snapshot.get("failed_births_total", 0) or 0)
        lro4_contexts = [row for row in lineage_contexts if row.get("lro4_offspring_dispersal_enable", "") not in ("", None)]
        lro4_enabled_rows = sum(1 for row in lro4_contexts if _safe_int(row.get("lro4_offspring_dispersal_enable", 0), 0) > 0)
        lro4_triggered_rows = sum(
            1 for row in lro4_contexts if _safe_int(row.get("lro4_offspring_dispersal_triggered", 0), 0) > 0
        )
        lro4_selected_from_expanded_rows = sum(
            1 for row in lro4_contexts if _safe_int(row.get("lro4_offspring_dispersal_selected_from_expanded", 0), 0) > 0
        )
        lro4_trigger_reasons = Counter(
            str(row.get("lro4_offspring_dispersal_trigger_reason", "") or "unknown")
            for row in lro4_contexts
            if _safe_int(row.get("lro4_offspring_dispersal_triggered", 0), 0) > 0
        )
        lro4_expanded_sum = sum(
            _safe_int(row.get("lro4_offspring_dispersal_expanded_candidate_count", 0), 0) for row in lro4_contexts
        )
        eligible_ratio = _ratio(eligible_share, alive_share)
        selected_ratio = _ratio(selected_share, eligible_share)
        offspring_selected_ratio = _ratio(offspring_share, selected_share)
        refill_precull_ratio = _ratio(post_refill_share, pre_cull_share)
        refill_offspring_ratio = _ratio(post_refill_share, offspring_share)
        birth_success_ratio = _ratio(float(offspring_count), float(birth_attempts))
        selected_alive_ratio = _ratio(selected_share, alive_share)
        offspring_alive_ratio = _ratio(offspring_share, alive_share)
        row = {
            "schema_version": SCHEMA_VERSION,
            "generation": int(generation),
            "lineage_id": lineage_id,
            "alive_count_before_selection": alive_count,
            "alive_share_before_selection": alive_share,
            "eligible_parent_count": eligible_count,
            "eligible_parent_share": eligible_share,
            "ineligible_alive_count": max(0, alive_count - eligible_count),
            "ineligibility_reason_counts_json": _stable_json({"not_alive_at_selection_boundary": max(0, alive_count - eligible_count)}),
            "selected_parent_count": selected_count,
            "selected_parent_share": selected_share,
            "unique_selected_parent_count": selected_count,
            "selected_parent_lineage_entropy": selected_entropy,
            "selected_parent_effective_lineage_count": _effective(selected_counts),
            "top1_selected_parent_share": _top_share(selected_counts, 1),
            "top3_selected_parent_share": _top_share(selected_counts, 3),
            "birth_attempts": birth_attempts,
            "successful_births": offspring_count,
            "failed_births": failed_total if len(lineages) <= 1 else 0,
            "failed_births_by_reason_json": _stable_json(birth_failures),
            "birth_failure_due_to_no_local_space": int(birth_failures.get("birth_fail_no_space", 0)),
            "offspring_produced_count": offspring_count,
            "offspring_share": offspring_share,
            "successful_birth_lineage_entropy": offspring_entropy,
            "top1_offspring_share": _top_share(offspring_counts, 1),
            "top3_offspring_share": _top_share(offspring_counts, 3),
            "deaths_before_refill": int(turnover_snapshot.get("deaths_before_refill", 0) or total_culled),
            "deaths_after_refill": int(turnover_snapshot.get("deaths_after_refill", 0) or 0),
            "culled_count": culled_count,
            "culled_share": _share(culled_count, max(1, total_culled)),
            "pre_cull_alive_share": pre_cull_share,
            "post_cull_alive_share": post_cull_share,
            "post_refill_alive_count": post_refill_count,
            "post_refill_alive_share": post_refill_share,
            "refill_offspring_lineage_source": int(refill_parent_source_counts.get(lineage_id, refill_counts.get(lineage_id, 0))),
            "refill_amplification_ratio": _ratio_value(post_refill_share, pre_cull_share),
            "local_free_neighbor_count_mean": _mean_context(lineage_contexts, "local_free_neighbor_count"),
            "local_density_parent_site_mean": _mean_context(lineage_contexts, "local_density_parent_site"),
            "local_density_child_site_mean": _mean_context(lineage_contexts, "local_density_child_site"),
            "parent_child_distance_mean": _mean_context(lineage_contexts, "parent_child_distance"),
            "parent_child_genome_distance_mean": _mean_context(lineage_contexts, "parent_child_genome_distance_mean"),
            "parent_child_genome_distance_max": _max_context(lineage_contexts, "parent_child_genome_distance_max"),
            "parent_child_genome_scalar_distance_mean": _mean_context(lineage_contexts, "parent_child_genome_scalar_distance_mean"),
            "parent_child_genome_memory_field_delta_mean": _mean_context(lineage_contexts, "parent_child_genome_memory_field_delta"),
            "parent_child_genome_prediction_field_delta_mean": _mean_context(lineage_contexts, "parent_child_genome_prediction_field_delta"),
            "parent_child_genome_module_gate_delta_mean": _mean_context(lineage_contexts, "parent_child_genome_module_gate_delta"),
            "child_patch_id_or_coordinates": _first_coord(lineage_contexts, "child_y", "child_x"),
            "parent_patch_id_or_coordinates": _first_coord(lineage_contexts, "parent_y", "parent_x"),
            "child_basin_id": _first_string(lineage_contexts, "child_basin_id"),
            "parent_basin_id": _first_string(lineage_contexts, "parent_basin_id"),
            "child_same_basin_as_parent_fraction": _mean_context(lineage_contexts, "child_same_basin_as_parent"),
            "lineage_birth_clustering_index": _mean_context(lineage_contexts, "lineage_birth_clustering_index"),
            "lineage_patch_basin_reuse_rate": _mean_context(lineage_contexts, "lineage_patch_basin_reuse_rate"),
            "food_parent_site_mean": _mean_context(lineage_contexts, "food_parent_site"),
            "food_child_site_mean": _mean_context(lineage_contexts, "food_child_site"),
            "waste_parent_site_mean": _mean_context(lineage_contexts, "waste_parent_site"),
            "waste_child_site_mean": _mean_context(lineage_contexts, "waste_child_site"),
            "scent_parent_site_mean": _mean_context(lineage_contexts, "scent_parent_site"),
            "scent_child_site_mean": _mean_context(lineage_contexts, "scent_child_site"),
            "shelter_cover_parent_site_mean": _mean_context(lineage_contexts, "shelter_cover_parent_site"),
            "shelter_cover_child_site_mean": _mean_context(lineage_contexts, "shelter_cover_child_site"),
            "predator_hazard_exposure_parent_site_mean": _mean_context(lineage_contexts, "predator_hazard_exposure_parent_site"),
            "predator_hazard_exposure_child_site_mean": _mean_context(lineage_contexts, "predator_hazard_exposure_child_site"),
            "lro4_offspring_dispersal_event_rows": len(lro4_contexts),
            "lro4_offspring_dispersal_enabled_rows": lro4_enabled_rows,
            "lro4_offspring_dispersal_disabled_rows": max(0, len(lro4_contexts) - lro4_enabled_rows),
            "lro4_offspring_dispersal_disabled_leakage_rows": sum(
                1 for row in lro4_contexts if _safe_int(row.get("lro4_offspring_dispersal_disabled_leakage_marker", 0), 0) > 0
            ),
            "lro4_offspring_dispersal_triggered_rows": lro4_triggered_rows,
            "lro4_offspring_dispersal_trigger_reason_counts_json": _stable_json(dict(sorted(lro4_trigger_reasons.items()))),
            "lro4_offspring_dispersal_expanded_candidate_count_sum": lro4_expanded_sum,
            "lro4_offspring_dispersal_expanded_candidate_count_mean": _ratio_value(
                float(lro4_expanded_sum),
                float(len(lro4_contexts)),
            ),
            "lro4_offspring_dispersal_selected_from_expanded_rows": lro4_selected_from_expanded_rows,
            "lro4_offspring_dispersal_selected_radius_mean": _mean_context(
                lro4_contexts,
                "lro4_offspring_dispersal_selected_radius",
            ),
            "lro4_offspring_dispersal_parent_density_mean": _mean_context(
                lro4_contexts,
                "lro4_offspring_dispersal_parent_density",
            ),
            "lro4_offspring_dispersal_parent_waste_mean": _mean_context(
                lro4_contexts,
                "lro4_offspring_dispersal_parent_waste",
            ),
            "lro4_offspring_dispersal_child_density_mean": _mean_context(
                lro4_contexts,
                "lro4_offspring_dispersal_child_density",
            ),
            "lro4_offspring_dispersal_child_waste_mean": _mean_context(
                lro4_contexts,
                "lro4_offspring_dispersal_child_waste",
            ),
            "first_generation_ecological_payoff_metrics_activate": int(pre_turnover_snapshot.get("ecology_payoff_generation", -1) or -1),
            "pressure_exposure_before_lineage_collapse": 0.0,
            "first_generation_zero_selected_parents": int(generation) if selected_count <= 0 else -1,
            "first_generation_zero_successful_births": int(generation) if offspring_count <= 0 else -1,
            "last_generation_with_selected_parents": int(generation) if selected_count > 0 else -1,
            "last_generation_with_successful_births": int(generation) if offspring_count > 0 else -1,
            "rolling_nonzero_birth_lineage_count": rolling_nonzero_birth,
            "rolling_nonzero_selected_parent_lineage_count": rolling_nonzero_selected,
            "eligible_parent_share_over_alive_share": _ratio_value(eligible_share, alive_share),
            "selected_parent_share_over_eligible_parent_share": _ratio_value(selected_share, eligible_share),
            "offspring_share_over_selected_parent_share": _ratio_value(offspring_share, selected_share),
            "post_refill_share_over_pre_refill_alive_share": _ratio_value(post_refill_share, pre_cull_share),
            "post_refill_share_over_offspring_share": _ratio_value(post_refill_share, offspring_share),
            "birth_success_share_over_birth_attempt_share": _ratio_value(float(offspring_count), float(birth_attempts)),
            "selected_parent_share_over_alive_share": _ratio_value(selected_share, alive_share),
            "offspring_share_over_alive_share": _ratio_value(offspring_share, alive_share),
            "eligible_amplification_class": classify_amplification(eligible_ratio),
            "selected_parent_amplification_class": classify_amplification(selected_ratio),
            "offspring_amplification_class": classify_amplification(offspring_selected_ratio),
            "refill_amplification_class": classify_amplification(refill_precull_ratio),
            "refill_over_offspring_amplification_class": classify_amplification(refill_offspring_ratio),
            "birth_success_amplification_class": classify_amplification(birth_success_ratio),
            "selected_over_alive_amplification_class": classify_amplification(selected_alive_ratio),
            "offspring_over_alive_amplification_class": classify_amplification(offspring_alive_ratio),
        }
        rows.append(row)
    return rows


def classify_run(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    material = [dict(r) for r in list(rows or [])]
    generations = sorted({_safe_int(row.get("generation", 0), 0) for row in material})
    final_rows = [row for row in material if _safe_int(row.get("generation", -1), -1) == (generations[-1] if generations else -1)]
    final_top = None
    if final_rows:
        final_top = max(final_rows, key=lambda r: (_safe_float(r.get("post_refill_alive_share", 0.0), 0.0), str(r.get("lineage_id", ""))))

    stage_counts: Counter[str] = Counter()
    for row in material:
        for field, stage in (
            ("eligible_amplification_class", "eligibility/survival filtering"),
            ("selected_parent_amplification_class", "selected-parent compression"),
            ("offspring_amplification_class", "reproductive success / birth success skew"),
            ("refill_amplification_class", "cull/refill amplification"),
            ("refill_over_offspring_amplification_class", "cull/refill amplification"),
        ):
            if str(row.get(field, "")) in {"strong amplification", "collapse-stage amplification"}:
                stage_counts[stage] += 1
    if not material:
        classification = "insufficient telemetry"
        decision = "Decision 7 - reporting insufficient; deeper engine tracing required"
    elif len(stage_counts) <= 0:
        classification = "insufficient telemetry"
        decision = "Decision 1 - funnel instrumentation ready, mechanism not yet classified from existing data"
    elif len(stage_counts) == 1:
        classification = stage_counts.most_common(1)[0][0]
        decision = _decision_for_classification(classification)
    else:
        classification = "mixed"
        decision = "Decision 6 - mixed funnel cascade supported"
    return {
        "schema_version": SCHEMA_VERSION,
        "generation_count": len(generations),
        "row_count": len(material),
        "future_winning_lineage_id": str(final_top.get("lineage_id", "")) if final_top else "",
        "future_winning_lineage_final_post_refill_share": _safe_float(final_top.get("post_refill_alive_share", 0.0), 0.0) if final_top else 0.0,
        "stage_amplification_counts": dict(sorted(stage_counts.items())),
        "collapse_stage_classification": classification,
        "decision": decision,
        "limitations": [
            "Tiny fixture runs validate reporting only; mechanism classification requires funnel-enabled campaign data.",
            "Optional patch, basin, and pressure fields are populated only when the active substrate exposes those contexts.",
        ],
    }


def _decision_for_classification(classification: str) -> str:
    if classification == "selected-parent compression":
        return "Decision 2 - selected-parent compression supported"
    if classification == "cull/refill amplification":
        return "Decision 3 - cull/refill amplification supported"
    if classification == "offspring locality / local basin reinforcement":
        return "Decision 4 - offspring locality/local basin reinforcement supported"
    if classification == "pressure payoff too late":
        return "Decision 5 - pressure payoff too late supported"
    if classification == "mixed":
        return "Decision 6 - mixed funnel cascade supported"
    return "Decision 1 - funnel instrumentation ready, mechanism not yet classified from existing data"
