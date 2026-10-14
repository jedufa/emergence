from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from env.env_config import EnvConfig
import native_runtime.safe_default as safe_default_module
from native_runtime.engine import NativeEielEngine, _runtime_capability
from native_runtime.promotion_envelope import PromotionEnvelopeValidationError
from native_runtime.safe_default import (
    NATIVE_SAFE_DEFAULT_PROFILE,
    NativeSafeDefaultDecision,
    apply_native_runtime_scope_lock_defaults,
    resolve_native_safe_default,
)
from native_runtime.bindings.extension import NativeExtensionStatus
from native_runtime.types import NativeRuntimeCapability
from simulation.runner_pkg.runner_config import SimConfig


def _cfg() -> SimConfig:
    cfg = SimConfig(
        seed=123,
        generations=1,
        steps_per_generation=4,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=True,
        policy_deterministic_actions=True,
    )
    cfg.native_runtime_cxx26_scope_lock_enable = True
    apply_native_runtime_scope_lock_defaults(cfg)
    cfg.native_runtime_fixed_policy_family = "mlp"
    cfg.native_runtime_allow_runtime_jit_build = False
    cfg.genome_c1_enable = False
    cfg.genome_c2_enable = False
    cfg.chemistry_functional_coupling_enable = False
    cfg.maintenance_enable = False
    cfg.force_prediction_disabled = True
    cfg.shock_start_prob = 0.0
    cfg.shock_period = 0
    cfg.shock_duration = 0
    return cfg


def _env() -> EnvConfig:
    cfg = EnvConfig(width=6, height=6, view_radius=1)
    cfg.patch_world_enable = False
    cfg.jungle_predator_pressure_enable = False
    cfg.wdl_delayed_hazard_enable = False
    cfg.reproduce_requires_mate = False
    return cfg


def _set_prebuilt(monkeypatch, tmp_path: Path) -> Path:
    extension = tmp_path / "eiel_native_runtime_ext.so"
    extension.write_bytes(b"not loaded by resolver tests")
    manifest = {
        "schema_version": "eiel_native_runtime_build_manifest_v1",
        "extension_abi_version": "eiel_native_runtime_abi_v1",
        "extension_version": "eiel_native_runtime_flat_v1",
        "output_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
    }
    extension.with_suffix(extension.suffix + ".manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EIEL_NATIVE_RUNTIME_EXTENSION", str(extension))
    monkeypatch.setattr(
        safe_default_module,
        "load_prebuilt_extension",
        lambda path: SimpleNamespace(),
    )
    monkeypatch.setattr(safe_default_module, "NATIVE_SAFE_DEFAULT_PROMOTION_ACCEPTED", False)
    monkeypatch.setattr(safe_default_module, "FLAT_NATIVE_PROMOTION_CERTIFICATE_PATH", "")
    monkeypatch.setattr(safe_default_module, "FLAT_NATIVE_PROMOTION_CERTIFICATE_SHA256", "")
    return extension


def test_native_safe_default_pre_promotion_fixture_keeps_supported_flat_envelope_on_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    decision = resolve_native_safe_default(_cfg(), _env())

    assert decision.profile == NATIVE_SAFE_DEFAULT_PROFILE
    assert decision.prebuilt_extension_available is True
    assert decision.native_envelope_supported is True
    assert decision.native_supported is True
    assert safe_default_module.NATIVE_SAFE_DEFAULT_PROMOTION_ACCEPTED is False
    assert decision.promotion_accepted is False
    assert decision.native_execution_active is False
    assert decision.reference_required is True
    assert decision.lifecycle_default_active is False
    assert decision.known_multigeneration_parity_block is False
    assert decision.selected_native_blocks == ()
    assert "known_multigeneration_parity_block" not in decision.unsupported_reasons
    assert decision.fallback_policy == "reference_required"


def test_native_safe_default_pre_promotion_explicit_diagnostic_request_runs_native(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    cfg = _cfg()
    cfg.native_runtime_diagnostic_enable = True
    decision = resolve_native_safe_default(cfg, _env())

    assert decision.native_envelope_supported is True
    assert decision.native_supported is True
    assert decision.promotion_accepted is False
    assert decision.native_execution_active is True
    assert decision.reference_required is False
    assert decision.lifecycle_default_active is False
    assert "native_action_cache" in decision.selected_native_blocks
    assert "native_lifecycle_default" not in decision.selected_native_blocks
    assert decision.fallback_policy == "explicit_native_request"


def test_native_safe_default_active_prediction_routes_normal_config_to_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    cfg = _cfg()
    cfg.force_prediction_disabled = False
    decision = resolve_native_safe_default(cfg, _env())

    assert decision.native_supported is False
    assert decision.reference_required is True
    assert decision.fail_closed is False
    assert decision.fallback_policy == "reference_required"
    assert "prediction_bookkeeping_not_native_owned" in decision.unsupported_reasons


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("injury_max", 0.0, "invalid_injury_max"),
        ("injury_max", float("nan"), "invalid_injury_max"),
        ("scar_max", -1.0, "invalid_scar_max"),
        ("scar_max", float("inf"), "invalid_scar_max"),
    ],
)
def test_invalid_injury_scar_max_routes_auto_and_fails_explicit_native(
    tmp_path: Path,
    monkeypatch,
    field: str,
    value: float,
    reason: str,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    env_cfg = _env()
    setattr(env_cfg, field, value)

    auto = resolve_native_safe_default(_cfg(), env_cfg)
    explicit_cfg = _cfg()
    explicit_cfg.native_runtime_enable = True
    explicit = resolve_native_safe_default(explicit_cfg, env_cfg)

    assert reason in auto.unsupported_reasons
    assert auto.reference_required is True
    assert auto.fail_closed is False
    assert reason in explicit.unsupported_reasons
    assert explicit.reference_required is True
    assert explicit.fail_closed is True


def test_native_safe_default_active_prediction_explicit_native_request_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    cfg = _cfg()
    cfg.force_prediction_disabled = False
    cfg.native_runtime_enable = True
    decision = resolve_native_safe_default(cfg, _env())

    assert decision.native_supported is False
    assert decision.reference_required is True
    assert decision.fail_closed is True
    assert decision.fallback_policy == "fail_closed"
    assert "prediction_bookkeeping_not_native_owned" in decision.unsupported_reasons


def test_native_safe_default_unsupported_normal_configs_reference_without_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    env = _env()
    env.patch_world_enable = True
    env.jungle_predator_pressure_enable = True
    env.wdl_delayed_hazard_enable = True
    cfg = _cfg()
    cfg.device = "cuda"
    cfg.genome_c1_enable = True
    cfg.genome_c2_enable = True
    cfg.chemistry_functional_coupling_enable = True
    cfg.maintenance_enable = True
    decision = resolve_native_safe_default(cfg, env)

    assert decision.reference_required is True
    assert decision.fail_closed is False
    assert "unsupported_world_patch" in decision.unsupported_reasons
    assert "unsupported_world_jungle" in decision.unsupported_reasons
    assert "unsupported_world_wdl" in decision.unsupported_reasons
    assert "unsupported_device" in decision.unsupported_reasons
    assert "unsupported_structural_operator_c1" in decision.unsupported_reasons
    assert "unsupported_structural_operator_c2" in decision.unsupported_reasons
    assert "unsupported_chemistry_or_maintenance" in decision.unsupported_reasons


def test_native_safe_default_explicit_unsupported_native_request_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    env = _env()
    env.patch_world_enable = True
    cfg = _cfg()
    cfg.native_runtime_enable = True
    cfg.native_exact_multigen_required = False
    cfg.native_runtime_diagnostic_ack_multigen_parity_open = True
    decision = resolve_native_safe_default(cfg, env)

    assert decision.fail_closed is True
    assert decision.native_supported is False
    assert "patch_churn_birth_commit_not_supported" in decision.unsupported_reasons


def test_native_safe_default_missing_prebuilt_and_runtime_jit_are_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EIEL_NATIVE_RUNTIME_EXTENSION", raising=False)
    cfg = _cfg()
    cfg.native_runtime_enable = True
    cfg.native_exact_multigen_required = False
    cfg.native_runtime_diagnostic_ack_multigen_parity_open = True
    missing = resolve_native_safe_default(cfg, _env())

    assert missing.fail_closed is True
    assert "prebuilt_native_runtime_extension_required" in missing.unsupported_reasons

    _set_prebuilt(monkeypatch, tmp_path)
    cfg.native_runtime_allow_runtime_jit_build = True
    jit = resolve_native_safe_default(cfg, _env())

    assert jit.fail_closed is True
    assert "runtime_jit_not_allowed" in jit.unsupported_reasons


def test_native_safe_default_gru_and_mate_required_remain_gated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    env = _env()
    env.reproduce_requires_mate = True
    runner = SimpleNamespace(
        _policy_use_gru=[True],
        sim_cfg=_cfg(),
        env_cfg=env,
    )
    runner.sim_cfg.native_runtime_enable = True
    runner.sim_cfg.native_exact_multigen_required = False
    runner.sim_cfg.native_runtime_diagnostic_ack_multigen_parity_open = True
    decision = resolve_native_safe_default(runner.sim_cfg, runner.env_cfg, runner)

    assert decision.fail_closed is True
    assert "fixed_mlp_policy_family_gru_rows_not_supported" in decision.unsupported_reasons
    assert "mate_required_exact_recombination_rng_not_supported" in decision.unsupported_reasons


def test_native_safe_default_config_defaults_keep_legacy_configs_loadable() -> None:
    cfg = SimConfig()

    assert cfg.native_execution_profile == NATIVE_SAFE_DEFAULT_PROFILE
    assert cfg.native_safe_default_enable is True
    assert cfg.native_exact_multigen_required is True
    assert cfg.native_runtime_diagnostic_enable is False


def test_native_scope_lock_sets_inactive_shock_defaults_before_overrides() -> None:
    cfg = SimConfig()

    apply_native_runtime_scope_lock_defaults(cfg)

    assert cfg.native_runtime_fixed_policy_family == "mlp"
    assert cfg.shock_start_prob == 0.0
    assert cfg.shock_period == 0
    assert cfg.shock_duration == 0


def test_active_stochastic_shock_routes_normal_request_to_reference(tmp_path: Path, monkeypatch) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    cfg = _cfg()
    cfg.shock_start_prob = 0.01

    decision = resolve_native_safe_default(cfg, _env())

    assert decision.native_supported is False
    assert decision.reference_required is True
    assert decision.fail_closed is False
    assert "native_shock_schedule_not_supported" in decision.unsupported_reasons


def test_active_periodic_and_pulsed_shock_are_rejected(tmp_path: Path, monkeypatch) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    periodic = _cfg()
    periodic.shock_period = 10
    periodic.shock_duration = 3
    pulsed = _cfg()
    pulsed.shock_pulsed = True

    periodic_decision = resolve_native_safe_default(periodic, _env())
    pulsed_decision = resolve_native_safe_default(pulsed, _env())

    assert "native_shock_schedule_not_supported" in periodic_decision.unsupported_reasons
    assert "native_shock_schedule_not_supported" in pulsed_decision.unsupported_reasons


def test_checkpointed_shock_state_is_rejected_even_with_inactive_config(tmp_path: Path, monkeypatch) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    cfg = _cfg()
    runner = SimpleNamespace(
        sim_cfg=cfg,
        env_cfg=_env(),
        _policy_use_gru=[False],
        _shock_active_rem=0,
        _shock_cooldown_rem=2,
    )

    decision = resolve_native_safe_default(cfg, runner.env_cfg, runner)

    assert decision.native_supported is False
    assert "native_shock_schedule_not_supported" in decision.unsupported_reasons


def test_explicit_native_shock_request_fails_closed(tmp_path: Path, monkeypatch) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    cfg = _cfg()
    cfg.native_runtime_enable = True
    cfg.shock_start_prob = 0.5

    decision = resolve_native_safe_default(cfg, _env())

    assert decision.fail_closed is True
    assert decision.fallback_policy == "fail_closed"
    assert "native_shock_schedule_not_supported" in decision.unsupported_reasons


def test_policy_family_inspection_failure_routes_reference_and_explicit_native_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class BrokenRows:
        def any(self):
            raise RuntimeError("resident family inspection failed")

        def __iter__(self):
            raise RuntimeError("resident family iteration failed")

    _set_prebuilt(monkeypatch, tmp_path)
    cfg = _cfg()
    runner = SimpleNamespace(sim_cfg=cfg, env_cfg=_env(), _policy_use_gru=BrokenRows())

    normal = resolve_native_safe_default(cfg, runner.env_cfg, runner)
    cfg.native_runtime_enable = True
    explicit = resolve_native_safe_default(cfg, runner.env_cfg, runner)

    assert normal.reference_required is True
    assert normal.fail_closed is False
    assert "policy_family_detection_failed" in normal.unsupported_reasons
    assert explicit.fail_closed is True
    assert "policy_family_detection_failed" in explicit.unsupported_reasons


def test_capability_defaults_are_non_claiming_and_runtime_population_is_explicit() -> None:
    defaults = NativeRuntimeCapability().to_dict()
    boolean_defaults = {key: value for key, value in defaults.items() if isinstance(value, bool)}

    assert boolean_defaults
    assert not any(boolean_defaults.values())

    accepted = {
        "extension_abi_version": "eiel_native_runtime_abi_v1",
        "extension_version": "eiel_native_runtime_flat_v1",
        "policy_family": "mlp",
        "device": "cpu",
        "lifecycle_branch": "flat_generation_birth_commit",
        "checkpoint_mode": "off",
        "reporting_supported": True,
        "benchmark_qualified": True,
        "accepted_default_native_envelope": True,
    }
    flat = _runtime_capability(world="flat", prebuilt_available=True, **accepted)
    patch = _runtime_capability(world="patch", prebuilt_available=True, **accepted)
    jit_flat = _runtime_capability(world="flat", prebuilt_available=False, **accepted)
    assert flat.cpu is True
    assert flat.persistent_engine is True
    assert flat.production_jit_free is True
    assert flat.supports_flat_generation is True
    assert flat.supports_patch_generation is False
    assert flat.supports_cxx19_formal_benchmark_gate is True
    assert patch.supports_flat_generation is False
    assert patch.supports_patch_generation is False
    assert patch.supports_exact_public_output_serialization is False
    assert jit_flat.runtime_jit_in_backend is True
    assert jit_flat.production_jit_free is False
    assert jit_flat.supports_cxx19_formal_benchmark_gate is False

    unqualified = _runtime_capability(
        world="flat",
        prebuilt_available=True,
        **{**accepted, "benchmark_qualified": False},
    )
    assert unqualified.supports_exact_public_output_serialization is True
    assert unqualified.supports_cxx19_formal_benchmark_gate is False


def test_restored_engine_claims_flat_capabilities_only_after_qualified_runner_bind() -> None:
    engine = NativeEielEngine.__new__(NativeEielEngine)
    engine.runner = None
    engine.world = "flat"
    engine.capability = NativeRuntimeCapability()
    runner = SimpleNamespace(sim_cfg=_cfg(), env_cfg=_env())
    decision = NativeSafeDefaultDecision(
        native_supported=True,
        native_execution_active=True,
        prebuilt_extension_available=True,
        extension_loaded=True,
        extension_abi_compatible=True,
        extension_hash_verified=True,
        extension_runtime_jit=False,
        extension_abi_version="eiel_native_runtime_abi_v1",
        extension_version="eiel_native_runtime_flat_v1",
        reporting_supported=True,
        reporting_qualification="route_neutral_exact",
        checkpoint_mode="generation_boundary_v2",
    )

    assert engine.capability.supports_flat_generation is False
    engine.bind_qualified_runner(runner, decision)

    assert engine.runner is runner
    assert engine.capability.supports_flat_generation is True
    assert engine.capability.reporting_support == "route_neutral_exact"
    assert engine.capability.accepted_default_native_envelope is False
    assert engine.capability.supports_cxx19_formal_benchmark_gate is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("behavior_log_enable", False),
        ("lineage_observability_enable", False),
        ("typed_lifecycle_public_output_route_neutral_enable", False),
        ("runnercore1_write_summary", True),
        ("native_runtime_write_summary", True),
        ("developmental_trace_write_enable", True),
        ("a3_recovery_taxonomy_enable", True),
        ("phase_e_e1_enable", True),
        ("phase_e_e2_enable", True),
        ("lineage_qd_archive_enable", True),
        ("campaign_output_profile", "full"),
    ],
)
def test_reporting_profile_override_routes_auto_and_fails_explicit_native(
    tmp_path: Path,
    monkeypatch,
    field: str,
    value: object,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    cfg = _cfg()
    setattr(cfg, field, value)
    auto = resolve_native_safe_default(cfg, _env())
    cfg.native_runtime_enable = True
    explicit = resolve_native_safe_default(cfg, _env())

    reason = f"reporting_profile_mismatch:{field}"
    assert auto.reference_required is True and auto.fail_closed is False
    assert reason in auto.reporting_reasons
    assert explicit.fail_closed is True
    assert reason in explicit.unsupported_reasons


def test_scope_lock_installs_exact_public_reporting_profile() -> None:
    cfg = SimConfig()
    apply_native_runtime_scope_lock_defaults(cfg)

    assert cfg.shock_pulsed is False
    assert cfg.pressure_autopilot_allow_shock_enable is False
    assert cfg.behavior_log_enable is True
    assert cfg.lineage_observability_enable is True
    assert cfg.typed_lifecycle_public_output_route_neutral_enable is True
    assert cfg.campaign_output_profile == "compact_default"
    for field in (
        "runnercore1_write_summary",
        "native_runtime_write_summary",
        "developmental_trace_write_enable",
        "a3_recovery_taxonomy_enable",
        "phase_e_e1_enable",
        "phase_e_e2_enable",
        "lineage_qd_archive_enable",
    ):
        assert getattr(cfg, field) is False


@pytest.mark.parametrize("period", [1, 2, 7])
def test_strict_generation_boundary_v2_accepts_any_positive_cadence(
    tmp_path: Path,
    monkeypatch,
    period: int,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    cfg = _cfg()
    cfg.checkpoint_bundle_v2_period_gen = period
    decision = resolve_native_safe_default(cfg, _env())

    assert decision.checkpoint_supported is True
    assert decision.checkpoint_mode == "generation_boundary_v2"
    assert decision.checkpoint_effective_period_gen == period
    assert "native_save_resume" in decision.checkpoint_supported_routes


def test_checkpoint_legacy_alias_and_equal_dual_period_are_consistent(tmp_path: Path, monkeypatch) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    legacy = _cfg()
    legacy.checkpoint_period_gen = 3
    legacy_decision = resolve_native_safe_default(legacy, _env())
    equal = _cfg()
    equal.checkpoint_period_gen = 4
    equal.checkpoint_bundle_v2_period_gen = 4
    equal_decision = resolve_native_safe_default(equal, _env())

    assert legacy_decision.checkpoint_effective_period_gen == 3
    assert "checkpoint_period_gen" in legacy_decision.deprecated_aliases_used
    assert equal_decision.checkpoint_supported is True
    assert equal_decision.checkpoint_effective_period_gen == 4


@pytest.mark.parametrize(
    ("v2", "legacy", "strict", "reason"),
    [
        (-1, 0, True, "negative_checkpoint_period:checkpoint_bundle_v2_period_gen"),
        (0, -1, True, "negative_checkpoint_period:checkpoint_period_gen"),
        (2, 3, True, "conflicting_checkpoint_periods"),
        (0, 0, False, "checkpoint_non_strict_mode_not_supported"),
    ],
)
def test_checkpoint_invalid_modes_route_auto_and_fail_explicit(
    tmp_path: Path,
    monkeypatch,
    v2: int,
    legacy: int,
    strict: bool,
    reason: str,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    cfg = _cfg()
    cfg.checkpoint_bundle_v2_period_gen = v2
    cfg.checkpoint_period_gen = legacy
    cfg.checkpoint_bundle_v2_strict_replay_mode = strict
    auto = resolve_native_safe_default(cfg, _env())
    cfg.native_runtime_enable = True
    explicit = resolve_native_safe_default(cfg, _env())

    assert reason in auto.checkpoint_reasons
    assert auto.reference_required is True and auto.fail_closed is False
    assert explicit.fail_closed is True


def test_partial_generation_resume_is_never_native_qualified(tmp_path: Path, monkeypatch) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    cfg = _cfg()
    runner = SimpleNamespace(
        sim_cfg=cfg,
        env_cfg=_env(),
        _policy_use_gru=[False],
        _eiel_resume_start_tick=1,
    )
    decision = resolve_native_safe_default(cfg, runner.env_cfg, runner)

    assert decision.checkpoint_partial_generation is True
    assert "partial_generation_checkpoint_not_supported" in decision.checkpoint_reasons
    assert decision.reference_required is True


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("patch_world_enable", "unsupported_world_patch"),
        ("patch_world_observation_enable", "patch_world_observation_not_supported"),
        ("patch_world_observation_stacked_gather_enable", "unsupported_world_patch"),
        ("patch_world_observation_numba_gather_enable", "unsupported_world_patch"),
        ("patch_movement_cost_apply_enable", "unsupported_world_patch"),
        ("roz2a_patch_niche_enable", "unsupported_world_patch"),
        ("patch_world_waste_consequence_enable", "unsupported_world_patch"),
        ("patch_waste_fertility_effect_enable", "unsupported_world_patch"),
        ("jungle_predator_pressure_enable", "unsupported_world_jungle"),
        ("jungle_reproduction_emissions_enable", "unsupported_world_jungle"),
        ("jungle_juvenile_predator_vulnerability_enable", "unsupported_world_jungle"),
        ("jungle_terrain_protection_enable", "unsupported_world_jungle"),
        ("jungle_ecological_recovery_enable", "unsupported_world_jungle"),
        ("jungle_recovery_fertility_enable", "unsupported_world_jungle"),
        ("jungle_resource_spread_enable", "unsupported_world_jungle"),
        ("wdl_delayed_hazard_enable", "unsupported_world_wdl"),
        ("wdl_delayed_hazard_observation_enable", "unsupported_world_wdl"),
        ("prediction_demand_enable", "unsupported_world_wdl"),
        ("prediction_demand_observation_enable", "unsupported_world_wdl"),
        ("delayed_resource_depletion_enable", "unsupported_world_wdl"),
        ("delayed_resource_depletion_observation_enable", "unsupported_world_wdl"),
        ("delayed_crowding_waste_enable", "unsupported_world_wdl"),
        ("delayed_crowding_waste_observation_enable", "unsupported_world_wdl"),
        ("path1_drh0_reporting_enable", "unsupported_world_wdl"),
        ("path1_drh0_refuge_reachability_enable", "unsupported_world_wdl"),
        ("path1_drh0_observation_enable", "unsupported_world_wdl"),
        ("path1_drh0_consequence_enable", "unsupported_world_wdl"),
        ("path1_drh0_refuge_damage_reduction_enable", "unsupported_world_wdl"),
        ("path1_drh0_static_refuge_control_enable", "unsupported_world_wdl"),
        ("mem_max_delayed_choice_enable", "unsupported_world_wdl"),
        ("mem_max_delayed_choice_observation_enable", "unsupported_world_wdl"),
        ("mem_max_delayed_choice_global_cue_observation_enable", "unsupported_world_wdl"),
        ("mem_max_delayed_choice_navigation_observation_enable", "unsupported_world_wdl"),
        ("reproduce_requires_mate", "unsupported_reproduction_mode_mate_required"),
    ],
)
def test_each_independent_unsupported_env_switch_is_rejected(
    tmp_path: Path,
    monkeypatch,
    field: str,
    reason: str,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    env = _env()
    setattr(env, field, True)
    decision = resolve_native_safe_default(_cfg(), env)

    assert reason in decision.unsupported_reasons
    assert decision.reference_required is True


def test_lifecycle_profile_world_alias_and_shock_alias_are_independently_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    lifecycle = _cfg()
    lifecycle.native_runtime_birth_commit_enable = False
    world = _env()
    world.world_mode = "flat_alias"
    shock = _cfg()
    shock.pressure_autopilot_allow_shock_enable = True

    assert "unsupported_lifecycle_branch" in resolve_native_safe_default(lifecycle, _env()).unsupported_reasons
    assert "unsupported_world_alias" in resolve_native_safe_default(_cfg(), world).unsupported_reasons
    assert "native_shock_schedule_not_supported" in resolve_native_safe_default(shock, _env()).unsupported_reasons


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("device", "cuda", "unsupported_device"),
        ("bank_device", "cuda", "unsupported_device"),
        ("maintenance_enable", True, "unsupported_chemistry_or_maintenance"),
        ("chemistry_functional_coupling_enable", True, "unsupported_chemistry_or_maintenance"),
        ("genome_c1_enable", True, "unsupported_structural_operator_c1"),
        ("genome_c2_enable", True, "unsupported_structural_operator_c2"),
        ("native_runtime_fixed_policy_family", "gru", "unsupported_policy_family"),
        ("native_execution_profile", "unsupported_diagnostic", "unsupported_execution_profile"),
    ],
)
def test_each_independent_unsupported_sim_surface_routes_reference(
    tmp_path: Path,
    monkeypatch,
    field: str,
    value: object,
    reason: str,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    cfg = _cfg()
    setattr(cfg, field, value)
    decision = resolve_native_safe_default(cfg, _env())

    assert reason in decision.unsupported_reasons
    assert decision.reference_required is True
    assert decision.fail_closed is False


def test_resident_gru_rows_are_independently_rejected(tmp_path: Path, monkeypatch) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    cfg = _cfg()
    runner = SimpleNamespace(sim_cfg=cfg, env_cfg=_env(), _policy_use_gru=[True])
    decision = resolve_native_safe_default(cfg, runner.env_cfg, runner)

    assert "unsupported_gru_rows" in decision.unsupported_reasons
    assert decision.reference_required is True


def test_explicit_reference_remains_reference_even_when_extension_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("EIEL_NATIVE_RUNTIME_EXTENSION", raising=False)
    cfg = _cfg()
    cfg.native_execution_profile = "reference"
    decision = resolve_native_safe_default(cfg, _env())

    assert decision.requested_execution_mode == "explicit_reference"
    assert decision.reference_required is True
    assert decision.fail_closed is False
    assert decision.fallback_policy == "explicit_reference"


@pytest.mark.parametrize(
    ("loader_error", "reason"),
    [
        (ImportError("import failed"), "native_extension_import_failed"),
        (RuntimeError("load failed"), "native_extension_load_failed"),
    ],
)
def test_extension_import_and_load_failures_are_distinct(
    tmp_path: Path,
    monkeypatch,
    loader_error: Exception,
    reason: str,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)

    def fail_loader(path):
        raise loader_error

    monkeypatch.setattr(safe_default_module, "load_prebuilt_extension", fail_loader)
    auto = resolve_native_safe_default(_cfg(), _env())
    cfg = _cfg()
    cfg.native_runtime_enable = True
    explicit = resolve_native_safe_default(cfg, _env())

    assert reason in auto.unsupported_reasons
    assert auto.fail_closed is False
    assert explicit.fail_closed is True


@pytest.mark.parametrize(
    ("abi", "version", "reason"),
    [
        ("wrong-abi", "eiel_native_runtime_flat_v1", "native_extension_abi_mismatch"),
        ("eiel_native_runtime_abi_v1", "wrong-version", "native_extension_version_mismatch"),
    ],
)
def test_extension_abi_and_version_mismatch_are_distinct(
    tmp_path: Path,
    monkeypatch,
    abi: str,
    version: str,
    reason: str,
) -> None:
    extension = _set_prebuilt(monkeypatch, tmp_path)
    monkeypatch.setattr(
        safe_default_module,
        "native_extension_status",
        lambda: NativeExtensionStatus(
            extension_path=str(extension),
            extension_abi_version=abi,
            extension_version=version,
            abi_compatible=False,
            output_sha256_verified=True,
            available=False,
            error="mismatch",
        ),
    )

    decision = resolve_native_safe_default(_cfg(), _env())
    assert reason in decision.unsupported_reasons


def test_malformed_extension_status_fails_qualification(tmp_path: Path, monkeypatch) -> None:
    extension = _set_prebuilt(monkeypatch, tmp_path)
    monkeypatch.setattr(
        safe_default_module,
        "native_extension_status",
        lambda: SimpleNamespace(
            extension_path=str(extension),
            extension_abi_version="eiel_native_runtime_abi_v1",
            extension_version="eiel_native_runtime_flat_v1",
            abi_compatible=True,
            output_sha256_verified=True,
            available="yes",
            runtime_jit=False,
            error="",
        ),
    )

    decision = resolve_native_safe_default(_cfg(), _env())
    assert "extension_status_malformed" in decision.unsupported_reasons


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("contract contents are wrong", "promotion_envelope_invalid"),
        ("cannot read promotion envelope /missing", "promotion_envelope_unavailable"),
    ],
)
def test_promotion_envelope_failure_routes_auto_and_fails_explicit(
    monkeypatch,
    message: str,
    reason: str,
) -> None:
    auto_cfg = _cfg()
    explicit_cfg = _cfg()
    explicit_cfg.native_runtime_enable = True

    def fail_contract():
        raise PromotionEnvelopeValidationError(message)

    monkeypatch.setattr(safe_default_module, "load_promotion_envelope", fail_contract)
    auto = resolve_native_safe_default(auto_cfg, _env())
    explicit = resolve_native_safe_default(explicit_cfg, _env())

    assert reason in auto.unsupported_reasons
    assert auto.reference_required is True and auto.fail_closed is False
    assert explicit.fail_closed is True


def test_reporting_and_benchmark_qualification_derive_from_promotion_decision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    extension = _set_prebuilt(monkeypatch, tmp_path)
    prepromotion = resolve_native_safe_default(_cfg(), _env())
    assert prepromotion.reporting_supported is True
    assert prepromotion.benchmark_qualified is False

    monkeypatch.setattr(safe_default_module, "NATIVE_SAFE_DEFAULT_PROMOTION_ACCEPTED", True)
    flag_only = resolve_native_safe_default(_cfg(), _env())
    assert flag_only.promotion_accepted is False
    assert flag_only.lifecycle_default_active is False
    assert flag_only.benchmark_qualified is False
    assert "promotion_certificate_missing" in flag_only.unsupported_reasons
    explicit_cfg = _cfg()
    explicit_cfg.native_runtime_enable = True
    explicit_without_certificate = resolve_native_safe_default(explicit_cfg, _env())
    assert explicit_without_certificate.fail_closed is True
    assert explicit_without_certificate.native_execution_active is False

    extension_sha256 = hashlib.sha256(extension.read_bytes()).hexdigest()
    monkeypatch.setattr(
        safe_default_module,
        "_qualify_promotion_certificate",
        lambda _extension: safe_default_module.NativePromotionCertificateQualification(
            accepted=True,
            path=str(tmp_path / "flat_native_promotion_certificate_v1.json"),
            sha256="a" * 64,
            candidate_commit_sha="b" * 40,
        ),
    )
    promoted = resolve_native_safe_default(_cfg(), _env())
    assert promoted.lifecycle_default_active is True
    assert promoted.promotion_accepted is True
    assert promoted.promotion_certificate_sha256 == "a" * 64
    assert promoted.extension_sha256 == extension_sha256
    assert promoted.reporting_supported is True
    assert promoted.benchmark_qualified is True
