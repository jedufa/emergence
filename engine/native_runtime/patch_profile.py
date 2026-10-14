from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


PATCH_PROFILE_CLASSIFICATION_SCHEMA_VERSION = (
    "eiel_native_patch_profile_classification_v1"
)
PATCH_PROFILE_P0 = "P0"
PATCH_PROFILE_P1 = "P1"
PATCH_PROFILE_P2 = "P2"
SUPPORTED_PATCH_PROFILES = (PATCH_PROFILE_P0, PATCH_PROFILE_P1, PATCH_PROFILE_P2)

_JUNGLE_SWITCHES = (
    "jungle_predator_pressure_enable",
    "jungle_reproduction_emissions_enable",
    "jungle_juvenile_predator_vulnerability_enable",
    "jungle_terrain_protection_enable",
    "jungle_ecological_recovery_enable",
    "jungle_recovery_fertility_enable",
    "jungle_resource_spread_enable",
)

_DELAYED_WORLD_SWITCHES = (
    "wdl_delayed_hazard_enable",
    "wdl_delayed_hazard_observation_enable",
    "prediction_demand_enable",
    "prediction_demand_observation_enable",
    "delayed_resource_depletion_enable",
    "delayed_resource_depletion_observation_enable",
    "delayed_crowding_waste_enable",
    "delayed_crowding_waste_observation_enable",
    "path1_drh0_reporting_enable",
    "path1_drh0_refuge_reachability_enable",
    "path1_drh0_observation_enable",
    "path1_drh0_consequence_enable",
    "path1_drh0_refuge_damage_reduction_enable",
    "path1_drh0_static_refuge_control_enable",
    "mem_max_delayed_choice_enable",
    "mem_max_delayed_choice_observation_enable",
    "mem_max_delayed_choice_global_cue_observation_enable",
    "mem_max_delayed_choice_navigation_observation_enable",
)

_ENV_UNSUPPORTED_SWITCH_GROUPS = (
    (
        "unsupported_respec",
        (
            "respec0b_resource_specialization_enable",
            "respec0b_resource_equalized_sham_enable",
            "respec_n1_zone_telemetry_enable",
            "respec_n2_capacity_balanced_mixed_enable",
        ),
    ),
    ("unsupported_mosaic", ("mosaic0a_depletion_recovery_enable",)),
    ("unsupported_predrisk", ("predrisk_fixed1_enable",)),
    (
        "unsupported_corridor",
        ("corridor_cp0_enable", "corridor_hc1_homing_cue_enable"),
    ),
    (
        "unsupported_habacclim",
        ("habacclim_enable", "habacclim_observation_enable"),
    ),
    ("unsupported_roz2a", ("roz2a_patch_niche_enable",)),
    ("unsupported_spatial_ecology", ("spatial_ecology_enable",)),
    ("unsupported_spatial_droughts", ("spatial_droughts_enable",)),
)

_SIM_UNSUPPORTED_SWITCH_GROUPS = (
    (
        "unsupported_mosaic",
        ("mosaic_fixed1_probe_enable", "mosaic_fixed1r_telemetry_enable"),
    ),
    (
        "unsupported_predrisk",
        ("predrisk_fixed1_probe_enable", "predrisk_fixed1_telemetry_enable"),
    ),
    (
        "unsupported_corridor",
        (
            "corridor_cp0_central_place_eligibility_enable",
            "corridor_cp0_probe_enable",
            "corridor_cp0_telemetry_enable",
        ),
    ),
    (
        "unsupported_habacclim",
        ("habacclim1_probe_enable", "habacclim1_telemetry_enable"),
    ),
)

_PATCH_LIFECYCLE_EMISSIONS = (
    "patch_scent_birth_emit",
    "patch_scent_death_emit",
    "patch_waste_birth_emit",
    "patch_waste_death_emit",
)

_MISSING = object()

_EXPLICIT_CONFIG_SOURCE_ALIASES = {
    "sim_cfg": frozenset(
        (
            "shock_pulsed",
            "memory_action_window_instrumentation_enable",
            "memory_action_window_emit_jsonl",
            "memory_action_window_include_lineage",
            "memory_action_window_include_memory_state",
            "memory_action_window_max_events_per_run",
            "native_flat_reporting_schema_version",
            "native_runtime_use_precomputed_drive_noise",
            "ordinary_births_enabled",
            "generation_turnover_enabled",
            "lifecycle_mode",
            "population_size",
            "cull_fraction",
            "parent_fraction",
            "turnover_selection_mode",
            "turnover_selection_seed",
            "rng_semantics_version",
            "hapb_rng_semantics_version",
            "world_uid",
            "rng_namespace_key64",
        )
    ),
    "env_cfg": frozenset(("world", "world_mode", "patch_waste_move_emit")),
}


@dataclass(frozen=True)
class PatchProfileClassification:
    """Closed Patch Architecture B profile decision."""

    schema_version: str = PATCH_PROFILE_CLASSIFICATION_SCHEMA_VERSION
    supported: bool = False
    profile: str | None = None
    unsupported_reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "supported": self.supported,
            "profile": self.profile,
            "unsupported_reasons": list(self.unsupported_reasons),
        }


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _snapshot(value: Any, *, source: str, reasons: list[str]) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    try:
        payload = vars(value)
    except TypeError:
        _add_reason(reasons, f"malformed_inspection:{source}")
        return None
    if not isinstance(payload, Mapping):
        _add_reason(reasons, f"malformed_inspection:{source}")
        return None
    return payload


def _reject_undeclared_config_fields(
    value: Any,
    payload: Mapping[str, Any],
    *,
    source: str,
    reasons: list[str],
) -> None:
    declared_fields = getattr(type(value), "__dataclass_fields__", None)
    if not isinstance(declared_fields, Mapping):
        _add_reason(reasons, f"configuration_schema_unavailable:{source}")
        return
    admitted = set(declared_fields)
    admitted.update(_EXPLICIT_CONFIG_SOURCE_ALIASES.get(source, ()))
    for name in sorted(
        payload_field
        for payload_field in set(payload).difference(admitted)
        if not str(payload_field).startswith("_")
    ):
        _add_reason(reasons, f"unknown_configuration_field:{source}.{name}")


def _raw_field(
    payload: Mapping[str, Any] | None,
    name: str,
    *,
    source: str,
    reasons: list[str],
    required: bool = True,
) -> Any:
    if payload is None:
        return _MISSING
    value = payload.get(name, _MISSING)
    if value is _MISSING and required:
        _add_reason(reasons, f"missing_inspection_field:{source}.{name}")
    return value


def _strict_bool(
    payload: Mapping[str, Any] | None,
    name: str,
    *,
    source: str,
    reasons: list[str],
    required: bool = True,
) -> bool | None:
    value = _raw_field(
        payload,
        name,
        source=source,
        reasons=reasons,
        required=required,
    )
    if value is _MISSING:
        return None
    if type(value) is not bool:
        _add_reason(reasons, f"malformed_inspection_field:{source}.{name}")
        return None
    return value


def _strict_int(
    payload: Mapping[str, Any] | None,
    name: str,
    *,
    source: str,
    reasons: list[str],
) -> int | None:
    value = _raw_field(payload, name, source=source, reasons=reasons)
    if value is _MISSING:
        return None
    if type(value) is not int:
        _add_reason(reasons, f"malformed_inspection_field:{source}.{name}")
        return None
    return value


def _strict_string(
    payload: Mapping[str, Any] | None,
    name: str,
    *,
    source: str,
    reasons: list[str],
    required: bool = True,
) -> str | None:
    value = _raw_field(
        payload,
        name,
        source=source,
        reasons=reasons,
        required=required,
    )
    if value is _MISSING:
        return None
    if type(value) is not str or not value.strip():
        _add_reason(reasons, f"malformed_inspection_field:{source}.{name}")
        return None
    return value.strip().lower()


def _strict_number(
    payload: Mapping[str, Any] | None,
    name: str,
    *,
    source: str,
    reasons: list[str],
    required: bool = True,
) -> float | None:
    value = _raw_field(
        payload,
        name,
        source=source,
        reasons=reasons,
        required=required,
    )
    if value is _MISSING:
        return None
    if type(value) not in (int, float):
        _add_reason(reasons, f"malformed_inspection_field:{source}.{name}")
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        _add_reason(reasons, f"malformed_inspection_field:{source}.{name}")
        return None
    return numeric


def _inspect_switch_group(
    payload: Mapping[str, Any] | None,
    names: Sequence[str],
    *,
    source: str,
    group_reason: str,
    reasons: list[str],
) -> None:
    for name in names:
        enabled = _strict_bool(
            payload,
            name,
            source=source,
            reasons=reasons,
        )
        if enabled is True:
            _add_reason(reasons, group_reason)
            _add_reason(reasons, f"unsupported_switch:{source}.{name}")


def _inspect_tw_rngv2_fields(
    payload: Mapping[str, Any] | None,
    *,
    reasons: list[str],
) -> None:
    if payload is None:
        return
    version = _raw_field(
        payload,
        "rng_semantics_version",
        source="sim_cfg",
        reasons=reasons,
        required=False,
    )
    if version is not _MISSING:
        if type(version) is not int or version not in (1, 2):
            _add_reason(
                reasons,
                "malformed_inspection_field:sim_cfg.rng_semantics_version",
            )
    world_uid = _raw_field(
        payload,
        "world_uid",
        source="sim_cfg",
        reasons=reasons,
        required=False,
    )
    if world_uid is not _MISSING:
        if type(world_uid) is not int or world_uid < 0:
            _add_reason(reasons, "malformed_inspection_field:sim_cfg.world_uid")
    namespace_key64 = _raw_field(
        payload,
        "rng_namespace_key64",
        source="sim_cfg",
        reasons=reasons,
        required=False,
    )
    if namespace_key64 is not _MISSING:
        if (
            type(namespace_key64) is not int
            or namespace_key64 != 0xCA05117E450A51C0
        ):
            _add_reason(
                reasons,
                "malformed_inspection_field:sim_cfg.rng_namespace_key64",
            )


def _inspect_runtime_policy_rows(
    runner: Any,
    *,
    allow_recurrent: bool = False,
    reasons: list[str],
) -> None:
    if runner is None:
        return
    payload = _snapshot(runner, source="runner", reasons=reasons)
    if payload is None:
        return
    rows = payload.get("_policy_use_gru", payload.get("policy_use_gru", _MISSING))
    if rows is _MISSING:
        state = payload.get("_runnercore_state", payload.get("runnercore_state", _MISSING))
        if state is not _MISSING and state is not None:
            state_payload = _snapshot(state, source="runner.runnercore_state", reasons=reasons)
            if state_payload is None:
                return
            rows = state_payload.get("policy_use_gru", _MISSING)
    if rows is _MISSING:
        population = payload.get("pop", _MISSING)
        alive_agents = getattr(population, "alive_agents", None)
        if callable(alive_agents):
            try:
                agents = list(alive_agents())
            except Exception:
                _add_reason(reasons, "policy_family_detection_failed")
                return
            genome_rows: list[bool] = []
            for agent in agents:
                use_gru = getattr(getattr(agent, "genome", None), "use_gru", _MISSING)
                if type(use_gru) is not bool:
                    _add_reason(reasons, "policy_family_detection_failed")
                    return
                genome_rows.append(use_gru)
            rows = genome_rows
    if rows is _MISSING:
        _add_reason(reasons, "policy_family_detection_failed")
        return
    if not isinstance(rows, (bool, list, tuple)) and callable(
        getattr(rows, "tolist", None)
    ):
        rows = rows.tolist()
    if type(rows) is bool:
        recurrent = rows
    elif isinstance(rows, (list, tuple)):
        if any(type(value) is not bool for value in rows):
            _add_reason(reasons, "policy_family_detection_failed")
            return
        recurrent = any(rows)
    else:
        _add_reason(reasons, "policy_family_detection_failed")
        return
    if recurrent and not allow_recurrent:
        _add_reason(reasons, "unsupported_gru_or_recurrent_policy")
        _add_reason(reasons, "fixed_mlp_policy_family_gru_rows_not_supported")

    for name in ("_shock_active_rem", "_shock_cooldown_rem"):
        value = payload.get(name, _MISSING)
        if value is _MISSING:
            continue
        if type(value) is not int:
            _add_reason(reasons, f"malformed_inspection_field:runner.{name}")
        elif value > 0:
            _add_reason(reasons, "native_shock_schedule_not_supported")
    shock_last = payload.get("_shock_last_active", _MISSING)
    if shock_last is not _MISSING:
        if type(shock_last) is not bool:
            _add_reason(
                reasons,
                "malformed_inspection_field:runner._shock_last_active",
            )
        elif shock_last:
            _add_reason(reasons, "native_shock_schedule_not_supported")


def _classification(
    profile: str | None,
    reasons: list[str],
) -> PatchProfileClassification:
    supported = profile in SUPPORTED_PATCH_PROFILES and not reasons
    return PatchProfileClassification(
        supported=supported,
        profile=profile if supported else None,
        unsupported_reasons=tuple(reasons),
    )


def classify_patch_profile(
    sim_cfg: Any,
    env_cfg: Any,
    runner: Any | None = None,
    *,
    world: Any | None = None,
    accepted_policy_families: Sequence[str] = ("mlp",),
) -> PatchProfileClassification:
    """Classify one config into the finite Patch Architecture B envelope.

    The function only reads the supplied objects. Missing fields, wrong types,
    unsupported feature switches, and unknown runtime policy rows all produce
    an unsupported decision rather than an inferred baseline profile.
    """

    reasons: list[str] = []
    normalized_policy_families = frozenset(
        str(value).strip().lower() for value in accepted_policy_families
    )
    if not normalized_policy_families or not normalized_policy_families.issubset(
        {"mlp", "gru", "mixed"}
    ):
        raise ValueError(
            "accepted_policy_families must contain only mlp, gru, and/or mixed"
        )
    sim = _snapshot(sim_cfg, source="sim_cfg", reasons=reasons)
    env = _snapshot(env_cfg, source="env_cfg", reasons=reasons)
    if sim is None or env is None:
        return _classification(None, reasons)
    _reject_undeclared_config_fields(
        sim_cfg,
        sim,
        source="sim_cfg",
        reasons=reasons,
    )
    _reject_undeclared_config_fields(
        env_cfg,
        env,
        source="env_cfg",
        reasons=reasons,
    )
    _inspect_tw_rngv2_fields(sim, reasons=reasons)

    patch_enabled = _strict_bool(
        env,
        "patch_world_enable",
        source="env_cfg",
        reasons=reasons,
    )
    if world is None:
        explicit_world = env.get("world", env.get("world_mode", _MISSING))
        if explicit_world is _MISSING:
            normal_world = "patch" if patch_enabled is True else "flat"
        elif type(explicit_world) is str and explicit_world.strip():
            normal_world = explicit_world.strip().lower()
        else:
            normal_world = None
            _add_reason(reasons, "malformed_inspection_field:world")
    elif type(world) is str and world.strip():
        normal_world = world.strip().lower()
    else:
        normal_world = None
        _add_reason(reasons, "malformed_inspection_field:world")
    if normal_world != "patch":
        _add_reason(reasons, "unsupported_world_not_patch")
    if patch_enabled is not True:
        _add_reason(reasons, "patch_world_gate_required")

    patch_schema = _strict_int(
        env,
        "patch_world_schema_version",
        source="env_cfg",
        reasons=reasons,
    )
    if patch_schema is not None and patch_schema != 1:
        _add_reason(reasons, "patch_world_schema_1_required")

    device = _strict_string(
        sim,
        "device",
        source="sim_cfg",
        reasons=reasons,
    )
    bank_device = _strict_string(
        sim,
        "bank_device",
        source="sim_cfg",
        reasons=reasons,
    )
    if device is not None and device != "cpu":
        _add_reason(reasons, "cuda_or_non_cpu_tensor_runtime_not_supported")
    if bank_device is not None and bank_device != "cpu":
        _add_reason(reasons, "cuda_or_non_cpu_tensor_runtime_not_supported")

    policy_family = _strict_string(
        sim,
        "native_runtime_fixed_policy_family",
        source="sim_cfg",
        reasons=reasons,
    )
    if policy_family is not None and policy_family not in normalized_policy_families:
        _add_reason(reasons, "unsupported_gru_or_recurrent_policy")
        _add_reason(reasons, "fixed_mlp_policy_family_required")
    runtime_jit = _strict_bool(
        sim,
        "native_runtime_allow_runtime_jit_build",
        source="sim_cfg",
        reasons=reasons,
    )
    if runtime_jit is True:
        _add_reason(reasons, "runtime_jit_not_allowed")

    reporting_schema = _strict_string(
        sim,
        "native_flat_reporting_schema_version",
        source="sim_cfg",
        reasons=reasons,
        required=False,
    )
    if (
        reporting_schema is not None
        and reporting_schema != "eiel_flat_native_reporting_v1"
    ):
        _add_reason(reasons, "reporting_schema_version_mismatch")

    for name in (
        "memory_action_window_instrumentation_enable",
        "memory_action_window_emit_jsonl",
        "memory_action_window_include_lineage",
        "memory_action_window_include_memory_state",
    ):
        enabled = _strict_bool(
            sim,
            name,
            source="sim_cfg",
            reasons=reasons,
            required=False,
        )
        if enabled is True:
            _add_reason(reasons, "memory_action_window_not_supported")
            _add_reason(reasons, f"unsupported_switch:sim_cfg.{name}")
    memory_event_limit = _raw_field(
        sim,
        "memory_action_window_max_events_per_run",
        source="sim_cfg",
        reasons=reasons,
        required=False,
    )
    if memory_event_limit is not _MISSING:
        if type(memory_event_limit) is not int:
            _add_reason(
                reasons,
                "malformed_inspection_field:sim_cfg.memory_action_window_max_events_per_run",
            )
        elif memory_event_limit != 0:
            _add_reason(reasons, "memory_action_window_not_supported")
            _add_reason(
                reasons,
                "unsupported_nonzero_field:sim_cfg.memory_action_window_max_events_per_run",
            )

    lifecycle_fields = {
        "ordinary_births_enabled",
        "generation_turnover_enabled",
        "population_size",
        "cull_fraction",
        "parent_fraction",
        "lifecycle_mode",
        "turnover_selection_mode",
        "turnover_selection_seed",
    }
    if any(name in sim for name in lifecycle_fields):
        ordinary_births = _strict_bool(
            sim,
            "ordinary_births_enabled",
            source="sim_cfg",
            reasons=reasons,
        )
        generation_turnover = _strict_bool(
            sim,
            "generation_turnover_enabled",
            source="sim_cfg",
            reasons=reasons,
        )
        population_size = _strict_int(
            sim,
            "population_size",
            source="sim_cfg",
            reasons=reasons,
        )
        cull_fraction = _strict_number(
            sim,
            "cull_fraction",
            source="sim_cfg",
            reasons=reasons,
        )
        parent_fraction = _strict_number(
            sim,
            "parent_fraction",
            source="sim_cfg",
            reasons=reasons,
        )
        turnover_selection_mode = _strict_int(
            sim,
            "turnover_selection_mode",
            source="sim_cfg",
            reasons=reasons,
        )
        lifecycle_mode = _raw_field(
            sim,
            "lifecycle_mode",
            source="sim_cfg",
            reasons=reasons,
            required=False,
        )
        if lifecycle_mode is not _MISSING and type(lifecycle_mode) is not int:
            _add_reason(reasons, "malformed_lifecycle_mode")
        turnover_selection_seed = _raw_field(
            sim,
            "turnover_selection_seed",
            source="sim_cfg",
            reasons=reasons,
        )
        legacy_lifecycle = (
            type(ordinary_births) is bool
            and generation_turnover is True
            and population_size in {64, 128, 256, 512}
            and cull_fraction == 0.5
            and parent_fraction == 1.0
            and lifecycle_mode in {_MISSING, 0}
            and turnover_selection_mode == 1
            and type(turnover_selection_seed) is int
            and 0 <= turnover_selection_seed < (1 << 64)
        )
        hybrid_lifecycle = (
            type(ordinary_births) is bool
            and generation_turnover is False
            and population_size in {64, 128, 192, 256, 512}
            and cull_fraction == 0.0
            and parent_fraction == 1.0
            and lifecycle_mode == 1
            and turnover_selection_mode == 1
            and type(turnover_selection_seed) is int
            and 0 <= turnover_selection_seed < (1 << 64)
        )
        if not (legacy_lifecycle or hybrid_lifecycle):
            _add_reason(reasons, "production_lifecycle_frozen_profile_required")
            if turnover_selection_seed is not _MISSING and (
                type(turnover_selection_seed) is not int
                or turnover_selection_seed < 0
                or turnover_selection_seed >= (1 << 64)
            ):
                _add_reason(
                    reasons,
                    "malformed_inspection_field:sim_cfg.turnover_selection_seed",
                )

    _inspect_runtime_policy_rows(
        runner,
        allow_recurrent=(
            policy_family in {"gru", "mixed"}
            and policy_family in normalized_policy_families
        ),
        reasons=reasons,
    )
    _inspect_switch_group(
        env,
        _JUNGLE_SWITCHES,
        source="env_cfg",
        group_reason="unsupported_active_jungle",
        reasons=reasons,
    )
    _inspect_switch_group(
        env,
        _DELAYED_WORLD_SWITCHES,
        source="env_cfg",
        group_reason="unsupported_wdl_or_delayed_world",
        reasons=reasons,
    )
    for group_reason, names in _ENV_UNSUPPORTED_SWITCH_GROUPS:
        _inspect_switch_group(
            env,
            names,
            source="env_cfg",
            group_reason=group_reason,
            reasons=reasons,
        )
    for group_reason, names in _SIM_UNSUPPORTED_SWITCH_GROUPS:
        _inspect_switch_group(
            sim,
            names,
            source="sim_cfg",
            group_reason=group_reason,
            reasons=reasons,
        )

    mate_required = _strict_bool(
        env,
        "reproduce_requires_mate",
        source="env_cfg",
        reasons=reasons,
    )
    inheritance_profile = _strict_string(
        sim,
        "inheritance_profile",
        source="sim_cfg",
        reasons=reasons,
        required=False,
    )
    diploid_inheritance = inheritance_profile == "diploid_linkage_v1"
    if mate_required is True and not diploid_inheritance:
        _add_reason(reasons, "unsupported_reproduction_mode_mate_required")
        _add_reason(
            reasons,
            "mate_required_exact_recombination_rng_not_supported",
        )
    if diploid_inheritance and mate_required is not True:
        _add_reason(reasons, "diploid_inheritance_requires_mate")

    tier4_enabled = _strict_bool(
        env,
        "tier4_juvenile_vulnerability_enable",
        source="env_cfg",
        reasons=reasons,
    )
    tier4_steps = _strict_int(
        env,
        "tier4_juvenile_steps",
        source="env_cfg",
        reasons=reasons,
    )
    if tier4_enabled is True or (tier4_steps is not None and tier4_steps > 0):
        _add_reason(reasons, "tier4_juvenile_vulnerability_not_supported")

    shock_active = False
    shock_pulsed = _strict_bool(
        sim,
        "shock_pulsed",
        source="sim_cfg",
        reasons=reasons,
        required=False,
    )
    shock_autopilot = _strict_bool(
        sim,
        "pressure_autopilot_allow_shock_enable",
        source="sim_cfg",
        reasons=reasons,
    )
    shock_active = shock_pulsed is True or shock_autopilot is True
    for name in ("shock_start_prob", "shock_period", "shock_duration"):
        value = _strict_number(
            sim,
            name,
            source="sim_cfg",
            reasons=reasons,
        )
        if value is not None and value > 0.0:
            shock_active = True
    if shock_active:
        _add_reason(reasons, "native_shock_schedule_not_supported")

    for name in _PATCH_LIFECYCLE_EMISSIONS:
        value = _strict_number(
            env,
            name,
            source="env_cfg",
            reasons=reasons,
        )
        if value is not None and value != 0.0:
            _add_reason(reasons, "unsupported_patch_lifecycle_emissions")
            _add_reason(reasons, f"unsupported_nonzero_field:env_cfg.{name}")
    density_emit = _strict_number(
        env,
        "patch_waste_density_emit_scale",
        source="env_cfg",
        reasons=reasons,
    )
    if density_emit is not None and density_emit != 0.0:
        _add_reason(reasons, "unsupported_patch_density_waste_emission")
        _add_reason(
            reasons,
            "unsupported_nonzero_field:env_cfg.patch_waste_density_emit_scale",
        )
    move_waste_emit = _strict_number(
        env,
        "patch_waste_move_emit",
        source="env_cfg",
        reasons=reasons,
        required=False,
    )
    if move_waste_emit is not None and move_waste_emit != 0.0:
        _add_reason(reasons, "unsupported_patch_lifecycle_emissions")
        _add_reason(
            reasons,
            "unsupported_nonzero_field:env_cfg.patch_waste_move_emit",
        )
    fertility_effect = _strict_bool(
        env,
        "patch_waste_fertility_effect_enable",
        source="env_cfg",
        reasons=reasons,
    )

    density_radius = _strict_int(
        env,
        "patch_density_radius",
        source="env_cfg",
        reasons=reasons,
    )
    visibility_from_cover = _strict_bool(
        env,
        "patch_visibility_from_cover_enable",
        source="env_cfg",
        reasons=reasons,
    )
    visibility_default = _strict_number(
        env,
        "patch_visibility_default",
        source="env_cfg",
        reasons=reasons,
    )
    visibility_min = _strict_number(
        env,
        "patch_visibility_min",
        source="env_cfg",
        reasons=reasons,
    )
    visibility_max = _strict_number(
        env,
        "patch_visibility_max",
        source="env_cfg",
        reasons=reasons,
    )
    if density_radius is not None and density_radius < 0:
        _add_reason(reasons, "invalid_patch_density_radius")
    if (
        visibility_min is not None
        and visibility_max is not None
        and visibility_min > visibility_max
    ):
        _add_reason(reasons, "invalid_patch_visibility_bounds")
    if (
        visibility_from_cover is False
        and visibility_default is not None
        and visibility_min is not None
        and visibility_max is not None
        and not visibility_min <= visibility_default <= visibility_max
    ):
        _add_reason(reasons, "invalid_patch_visibility_default")

    observation = _strict_bool(
        env,
        "patch_world_observation_enable",
        source="env_cfg",
        reasons=reasons,
    )
    observation_schema = _strict_int(
        env,
        "patch_world_observation_schema_version",
        source="env_cfg",
        reasons=reasons,
    )
    stacked = _strict_bool(
        env,
        "patch_world_observation_stacked_gather_enable",
        source="env_cfg",
        reasons=reasons,
    )
    numba = _strict_bool(
        env,
        "patch_world_observation_numba_gather_enable",
        source="env_cfg",
        reasons=reasons,
    )
    diagnostics = _strict_bool(
        env,
        "patch_world_diagnostics_enable",
        source="env_cfg",
        reasons=reasons,
    )
    lineage_descriptors = _strict_bool(
        env,
        "patch_world_lineage_descriptors_enable",
        source="env_cfg",
        reasons=reasons,
    )
    descriptor_schema = _strict_int(
        env,
        "patch_world_descriptor_schema_version",
        source="env_cfg",
        reasons=reasons,
    )
    waste_consequence = _strict_bool(
        env,
        "patch_world_waste_consequence_enable",
        source="env_cfg",
        reasons=reasons,
    )
    movement_apply = _strict_bool(
        env,
        "patch_movement_cost_apply_enable",
        source="env_cfg",
        reasons=reasons,
    )

    if stacked is True:
        _add_reason(reasons, "patch_observation_stacked_gather_not_supported")
    if numba is True:
        _add_reason(reasons, "patch_observation_numba_gather_not_supported")
    if (diagnostics is True or lineage_descriptors is True) and (
        descriptor_schema is not None and descriptor_schema != 1
    ):
        _add_reason(reasons, "patch_descriptor_schema_1_required")

    profile: str | None = None
    profile_values_known = all(
        value is not None
        for value in (observation, waste_consequence, movement_apply)
    )
    if profile_values_known and observation is False:
        if waste_consequence is False and movement_apply is False:
            profile = PATCH_PROFILE_P0
            if fertility_effect is True:
                _add_reason(reasons, "p2_fertility_effect_requires_p2_profile")
        else:
            _add_reason(reasons, "unsupported_patch_profile_combination")
    elif profile_values_known and observation is True:
        if observation_schema != 1:
            _add_reason(reasons, "patch_observation_schema_1_required")
        if diagnostics is not True:
            _add_reason(reasons, "patch_world_diagnostics_required")
        if lineage_descriptors is not True:
            _add_reason(reasons, "patch_world_lineage_descriptors_required")
        if descriptor_schema != 1:
            _add_reason(reasons, "patch_descriptor_schema_1_required")
        if waste_consequence is False and movement_apply is False:
            profile = PATCH_PROFILE_P1
            if fertility_effect is True:
                _add_reason(reasons, "p2_fertility_effect_requires_p2_profile")
        elif waste_consequence is True and movement_apply is True:
            profile = PATCH_PROFILE_P2
        else:
            _add_reason(reasons, "unsupported_patch_profile_combination")

    return _classification(profile, reasons)


def classify_patch_architecture_b_profile(
    sim_cfg: Any,
    env_cfg: Any,
    runner: Any | None = None,
    *,
    world: Any | None = None,
) -> PatchProfileClassification:
    """Explicitly named alias for callers integrating Architecture B."""

    return classify_patch_profile(sim_cfg, env_cfg, runner, world=world)


__all__ = (
    "PATCH_PROFILE_CLASSIFICATION_SCHEMA_VERSION",
    "PATCH_PROFILE_P0",
    "PATCH_PROFILE_P1",
    "PATCH_PROFILE_P2",
    "SUPPORTED_PATCH_PROFILES",
    "PatchProfileClassification",
    "classify_patch_architecture_b_profile",
    "classify_patch_profile",
)
