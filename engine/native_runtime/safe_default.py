from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .bindings import load_prebuilt_extension, native_extension_status
from .promotion_certificate import (
    PromotionCertificateValidationError,
    load_promotion_certificate,
    require_promotable_certificate,
)
from .promotion_envelope import (
    CONTRACT_SCHEMA_VERSION,
    FlatNativePromotionEnvelope,
    PromotionEnvelopeValidationError,
    load_promotion_envelope,
)


NATIVE_SAFE_DEFAULT_PROFILE = "native_safe_default"
SAFE_DEFAULT_SCHEMA_VERSION = "eiel_native_safe_default_decision_v1"
PARITY_EVIDENCE_VERSION = "eiel-flat-native-authoritative-promotion-20260716"
NATIVE_SAFE_DEFAULT_PROMOTION_ACCEPTED = True
FLAT_NATIVE_PROMOTION_CERTIFICATE_PATH = (
    "registries/performance/flat_native_promotion_certificate_v1.json"
)
FLAT_NATIVE_PROMOTION_CERTIFICATE_SHA256 = (
    "fb9b46101c1cf4c56d615b38379844fe633eb9578b747fbdea29610910930a9a"
)
FLAT_NATIVE_PROMOTION_ENVELOPE_VERSION = CONTRACT_SCHEMA_VERSION
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_REPORTING_PROFILE_FIXED_VALUES = {
    "behavior_log_enable": True,
    "lineage_observability_enable": True,
    "typed_lifecycle_public_output_route_neutral_enable": True,
    "runnercore1_write_summary": False,
    "native_runtime_write_summary": False,
    "developmental_trace_write_enable": False,
    "a3_recovery_taxonomy_enable": False,
    "phase_e_e1_enable": False,
    "phase_e_e2_enable": False,
    "lineage_qd_archive_enable": False,
}

_JUNGLE_SWITCHES = (
    "jungle_predator_pressure_enable",
    "jungle_reproduction_emissions_enable",
    "jungle_juvenile_predator_vulnerability_enable",
    "jungle_terrain_protection_enable",
    "jungle_ecological_recovery_enable",
    "jungle_recovery_fertility_enable",
    "jungle_resource_spread_enable",
)

_PATCH_SWITCHES = (
    "patch_world_enable",
    "patch_world_observation_enable",
    "patch_world_observation_stacked_gather_enable",
    "patch_world_observation_numba_gather_enable",
    "patch_movement_cost_apply_enable",
    "roz2a_patch_niche_enable",
    "patch_world_waste_consequence_enable",
    "patch_waste_fertility_effect_enable",
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

SAFE_INFRASTRUCTURE_BLOCKS = (
    "prebuilt_extension_loading",
    "runtime_jit_prohibited",
    "persistent_native_engine",
    "initial_import_projection",
    "native_action_cache",
    "fixed_mlp_policy_execution",
    "native_rng_state_projection",
    "native_checkpoint_sidecar",
    "native_batch_entrypoint",
    "zero_normal_agent_sync",
    "native_authority_counters",
    "fail_closed_support_resolver",
    "native_lifecycle_default",
)

UNSUPPORTED_BLOCKS = (
    "patch_native_lifecycle",
    "jungle_native_lifecycle",
    "wdl_native_lifecycle",
    "cuda_generation",
    "gru_native_memory",
    "c1_structural_mutation",
    "c2_structural_recombination",
    "policy_family_switching",
    "mate_required_reproduction",
    "chemistry",
    "maintenance",
    "prediction_bookkeeping",
)


@dataclass(frozen=True)
class NativeSafeDefaultDecision:
    schema_version: str = SAFE_DEFAULT_SCHEMA_VERSION
    profile: str = NATIVE_SAFE_DEFAULT_PROFILE
    requested_execution_mode: str = "auto"
    native_safe_default_enabled: bool = True
    native_requested: bool = False
    native_diagnostic_requested: bool = False
    native_envelope_supported: bool = False
    native_supported: bool = False
    promotion_accepted: bool = NATIVE_SAFE_DEFAULT_PROMOTION_ACCEPTED
    promotion_certificate_path: str = FLAT_NATIVE_PROMOTION_CERTIFICATE_PATH
    promotion_certificate_sha256: str = FLAT_NATIVE_PROMOTION_CERTIFICATE_SHA256
    promotion_certificate_candidate_commit_sha: str = ""
    promotion_certificate_error: str = ""
    native_execution_active: bool = False
    reference_required: bool = True
    fail_closed: bool = False
    prebuilt_extension_required: bool = True
    runtime_jit_allowed: bool = False
    runtime_jit_prohibited: bool = True
    lifecycle_default_active: bool = False
    known_multigeneration_parity_block: bool = False
    selected_native_blocks: tuple[str, ...] = field(default_factory=tuple)
    unsupported_reasons: tuple[str, ...] = field(default_factory=tuple)
    fallback_policy: str = "reference_required"
    parity_evidence_version: str = PARITY_EVIDENCE_VERSION
    prebuilt_extension_available: bool = False
    prebuilt_extension_error: str = ""
    extension_path: str = ""
    extension_sha256: str = ""
    extension_abi_version: str = ""
    extension_version: str = ""
    extension_abi_compatible: bool = False
    extension_hash_verified: bool = False
    extension_loaded: bool = False
    extension_runtime_jit: bool = False
    reporting_supported: bool = False
    reporting_profile: str = ""
    reporting_qualification: str = ""
    reporting_schema_version: str = ""
    reporting_reasons: tuple[str, ...] = field(default_factory=tuple)
    checkpoint_supported: bool = False
    checkpoint_mode: str = "off"
    checkpoint_effective_period_gen: int = 0
    checkpoint_strict_replay: bool = True
    checkpoint_partial_generation: bool = False
    checkpoint_reasons: tuple[str, ...] = field(default_factory=tuple)
    checkpoint_supported_routes: tuple[str, ...] = field(default_factory=tuple)
    benchmark_qualified: bool = False
    promotion_envelope_version: str = FLAT_NATIVE_PROMOTION_ENVELOPE_VERSION
    promotion_envelope_error: str = ""
    deprecated_aliases_used: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_native_blocks"] = list(self.selected_native_blocks)
        payload["unsupported_reasons"] = list(self.unsupported_reasons)
        payload["reporting_reasons"] = list(self.reporting_reasons)
        payload["checkpoint_reasons"] = list(self.checkpoint_reasons)
        payload["checkpoint_supported_routes"] = list(self.checkpoint_supported_routes)
        payload["deprecated_aliases_used"] = list(self.deprecated_aliases_used)
        return payload


@dataclass(frozen=True)
class NativeCheckpointQualification:
    supported: bool
    mode: str
    effective_period_gen: int
    strict_replay: bool
    partial_generation: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    supported_routes: tuple[str, ...] = field(default_factory=tuple)
    deprecated_aliases_used: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NativeReportingQualification:
    supported: bool
    profile: str = ""
    qualification: str = ""
    schema_version: str = ""
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NativeExtensionQualification:
    supported: bool
    available: bool = False
    path: str = ""
    sha256: str = ""
    abi_version: str = ""
    extension_version: str = ""
    abi_compatible: bool = False
    hash_verified: bool = False
    loaded: bool = False
    runtime_jit: bool = False
    error: str = ""
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NativePromotionCertificateQualification:
    accepted: bool = False
    path: str = ""
    sha256: str = ""
    candidate_commit_sha: str = ""
    reasons: tuple[str, ...] = field(default_factory=tuple)
    error: str = ""


def _bool_cfg(obj: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(obj, name, default))


def _str_cfg(obj: Any, name: str, default: str = "") -> str:
    return str(getattr(obj, name, default) or default).strip()


def _normal_device(value: Any) -> str:
    return str(value or "cpu").strip().lower()


def _promotion_envelope() -> tuple[FlatNativePromotionEnvelope | None, str, str]:
    try:
        return load_promotion_envelope(), "", ""
    except PromotionEnvelopeValidationError as exc:
        error = str(exc)
        reason = "promotion_envelope_unavailable" if "cannot read promotion envelope" in error else "promotion_envelope_invalid"
        return None, reason, error
    except (ImportError, OSError, TypeError, ValueError) as exc:
        return None, "promotion_envelope_unavailable", repr(exc)
    except Exception as exc:
        return None, "promotion_envelope_invalid", repr(exc)


def _qualify_promotion_certificate(
    extension: NativeExtensionQualification,
) -> NativePromotionCertificateQualification:
    if not NATIVE_SAFE_DEFAULT_PROMOTION_ACCEPTED:
        return NativePromotionCertificateQualification()

    configured_path = str(FLAT_NATIVE_PROMOTION_CERTIFICATE_PATH or "").strip()
    configured_sha256 = str(FLAT_NATIVE_PROMOTION_CERTIFICATE_SHA256 or "").strip()
    if not configured_path:
        return NativePromotionCertificateQualification(
            reasons=("promotion_certificate_missing",),
        )
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        return NativePromotionCertificateQualification(
            path=str(path),
            sha256=configured_sha256,
            reasons=("promotion_certificate_missing",),
        )
    if not configured_sha256:
        return NativePromotionCertificateQualification(
            path=str(path),
            reasons=("promotion_certificate_hash_missing",),
        )

    try:
        certificate = load_promotion_certificate(
            path,
            expected_sha256=configured_sha256,
        )
        certificate = require_promotable_certificate(
            certificate,
            expected_sha256=configured_sha256,
        )
    except PromotionCertificateValidationError as exc:
        return NativePromotionCertificateQualification(
            path=str(path),
            sha256=configured_sha256,
            reasons=("promotion_certificate_invalid",),
            error=str(exc),
        )
    except (OSError, TypeError, ValueError) as exc:
        return NativePromotionCertificateQualification(
            path=str(path),
            sha256=configured_sha256,
            reasons=("promotion_certificate_unavailable",),
            error=repr(exc),
        )

    reasons: list[str] = []
    if certificate["extension_sha256"] != extension.sha256:
        reasons.append("promotion_certificate_extension_hash_mismatch")
    if certificate["extension_abi_version"] != extension.abi_version:
        reasons.append("promotion_certificate_extension_abi_mismatch")
    if certificate["extension_version"] != extension.extension_version:
        reasons.append("promotion_certificate_extension_version_mismatch")
    if certificate["parity_evidence_version"] != PARITY_EVIDENCE_VERSION:
        reasons.append("promotion_certificate_parity_evidence_mismatch")
    return NativePromotionCertificateQualification(
        accepted=not reasons,
        path=str(path),
        sha256=configured_sha256,
        candidate_commit_sha=str(certificate["candidate_release_commit_sha"]),
        reasons=tuple(sorted(set(reasons))),
    )


def _reporting_profile_values(envelope: FlatNativePromotionEnvelope) -> dict[str, Any]:
    return {
        "campaign_output_profile": envelope.supported_profile.output.campaign_profile,
        **_REPORTING_PROFILE_FIXED_VALUES,
    }


def apply_native_runtime_scope_lock_defaults(sim_cfg: Any) -> None:
    """Apply the fixed Flat envelope before explicit CLI/knob overrides."""

    envelope, reason, error = _promotion_envelope()
    if envelope is None:
        setattr(sim_cfg, "_native_promotion_envelope_bootstrap_reason", reason)
        setattr(sim_cfg, "_native_promotion_envelope_bootstrap_error", error)
        return
    profile = envelope.supported_profile
    if not _str_cfg(sim_cfg, "native_runtime_fixed_policy_family"):
        sim_cfg.native_runtime_fixed_policy_family = profile.policy.family
    sim_cfg.chemistry_functional_coupling_enable = profile.chemistry.functional_coupling_allowed
    sim_cfg.maintenance_enable = profile.chemistry.maintenance_allowed
    sim_cfg.genome_c1_enable = profile.structural_operators.c1_mode != "disabled"
    sim_cfg.genome_c2_enable = profile.structural_operators.c2_mode != "disabled"
    sim_cfg.enable_evolvable_sensor_channel_mask = False
    sim_cfg.enable_evolvable_sensor_radius = False
    sim_cfg.enable_evolvable_introspection_channels = False
    sim_cfg.enable_evolvable_self_regulation_controls = False
    sim_cfg.enable_evolvable_self_modification_targets = False
    sim_cfg.enable_evolvable_directed_inheritance_channels = False
    sim_cfg.shock_pulsed = False
    sim_cfg.shock_start_prob = 0.0
    sim_cfg.shock_period = 0
    sim_cfg.shock_duration = 0
    sim_cfg.shock_duration_min = 0
    sim_cfg.shock_duration_max = 0
    sim_cfg.pressure_autopilot_allow_shock_enable = False
    sim_cfg.native_flat_reporting_schema_version = profile.reporting.schema_version
    for name, value in _reporting_profile_values(envelope).items():
        setattr(sim_cfg, name, value)


def _integer_period(value: Any, *, name: str, reasons: list[str]) -> int:
    if isinstance(value, bool):
        reasons.append(f"invalid_checkpoint_period:{name}")
        return 0
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError, OverflowError):
        reasons.append(f"invalid_checkpoint_period:{name}")
        return 0
    if isinstance(value, float) and not value.is_integer():
        reasons.append(f"invalid_checkpoint_period:{name}")
        return 0
    if parsed < 0:
        reasons.append(f"negative_checkpoint_period:{name}")
        return 0
    return parsed


def qualify_native_checkpoint_mode(
    sim_cfg: Any,
    runner: Any | None = None,
    *,
    envelope: FlatNativePromotionEnvelope | None = None,
) -> NativeCheckpointQualification:
    reasons: list[str] = []
    if envelope is None:
        envelope, envelope_reason, _ = _promotion_envelope()
        if envelope is None:
            reasons.append(envelope_reason)
    v2_period = _integer_period(
        getattr(sim_cfg, "checkpoint_bundle_v2_period_gen", 0),
        name="checkpoint_bundle_v2_period_gen",
        reasons=reasons,
    )
    legacy_period = _integer_period(
        getattr(sim_cfg, "checkpoint_period_gen", 0),
        name="checkpoint_period_gen",
        reasons=reasons,
    )
    if v2_period > 0 and legacy_period > 0 and v2_period != legacy_period:
        reasons.append("conflicting_checkpoint_periods")
    effective_period = v2_period if v2_period > 0 else legacy_period
    strict_replay = _bool_cfg(sim_cfg, "checkpoint_bundle_v2_strict_replay_mode", True)
    if not strict_replay:
        reasons.append("checkpoint_non_strict_mode_not_supported")

    partial_generation = False
    if runner is not None:
        try:
            partial_generation = int(getattr(runner, "_eiel_resume_start_tick", 0) or 0) > 0
        except (TypeError, ValueError, OverflowError):
            partial_generation = True
        metadata = getattr(runner, "_cxxbackbone10_checkpoint_metadata_overrides", None)
        if isinstance(metadata, Mapping):
            try:
                partial_generation = partial_generation or int(metadata.get("tick", 0) or 0) > 0
            except (TypeError, ValueError, OverflowError):
                partial_generation = True
            phase = str(metadata.get("agent_grid_runner_phase", "") or "").strip().lower()
            if phase and phase != "generation_boundary":
                partial_generation = True
    if partial_generation:
        reasons.append("partial_generation_checkpoint_not_supported")

    if envelope is not None:
        checkpoint_contract = envelope.supported_profile.checkpoint
        required_modes = (
            ("partial_generation", partial_generation),
            ("off", effective_period == 0),
            ("complete_generation_write", effective_period > 0),
            ("native_save_resume", effective_period > 0),
        )
        for mode_name, requested in required_modes:
            if not requested:
                continue
            try:
                supported = checkpoint_contract.mode(mode_name).supported
            except KeyError:
                reasons.append(f"checkpoint_contract_mode_missing:{mode_name}")
            else:
                if mode_name == "partial_generation":
                    if supported:
                        reasons.append("promotion_envelope_invalid")
                elif not supported:
                    reasons.append(f"checkpoint_mode_not_supported:{mode_name}")

    aliases = ("checkpoint_period_gen",) if legacy_period > 0 else ()
    mode = "generation_boundary_v2" if effective_period > 0 else "off"
    return NativeCheckpointQualification(
        supported=not reasons,
        mode=mode,
        effective_period_gen=int(effective_period),
        strict_replay=bool(strict_replay),
        partial_generation=bool(partial_generation),
        reasons=tuple(sorted(set(reasons))),
        supported_routes=(
            tuple(mode.name for mode in envelope.supported_profile.checkpoint.modes if mode.supported)
            if envelope is not None
            else ()
        ),
        deprecated_aliases_used=aliases,
    )


def qualified_checkpoint_period_gen(sim_cfg: Any, runner: Any | None = None) -> int:
    qualification = qualify_native_checkpoint_mode(sim_cfg, runner)
    if not qualification.supported:
        joined = ",".join(qualification.reasons)
        raise RuntimeError(f"EIEL-NATIVE-CHECKPOINT-UNSUPPORTED: {joined}")
    return int(qualification.effective_period_gen)


def qualify_native_reporting(
    sim_cfg: Any,
    runner: Any | None = None,
    *,
    envelope: FlatNativePromotionEnvelope | None = None,
) -> NativeReportingQualification:
    reasons: list[str] = []
    if envelope is None:
        envelope, envelope_reason, _ = _promotion_envelope()
        if envelope is None:
            return NativeReportingQualification(
                supported=False,
                reasons=(envelope_reason,),
            )
    reporting_contract = envelope.supported_profile.reporting
    configured_schema = _str_cfg(
        sim_cfg,
        "native_flat_reporting_schema_version",
        reporting_contract.schema_version,
    )
    if configured_schema != reporting_contract.schema_version:
        reasons.append("reporting_schema_version_mismatch")
    for name, expected in _reporting_profile_values(envelope).items():
        actual = getattr(sim_cfg, name, expected)
        if actual != expected:
            reasons.append(f"reporting_profile_mismatch:{name}")
    if not all(
        (
            reporting_contract.metrics_csv_exact,
            reporting_contract.behaviors_exact,
            reporting_contract.lineage_exact,
            reporting_contract.typed_lifecycle_exact,
            reporting_contract.native_owner_provenance_required,
            reporting_contract.finalization_required,
            not reporting_contract.stale_python_agent_population_source_allowed,
            not reporting_contract.missing_resident_field_allowed,
            envelope.supported_profile.output.complete_flush_required,
            not envelope.supported_profile.output.route_only_diagnostics_allowed,
            not envelope.supported_profile.output.partial_outputs_promotion_eligible,
        )
    ):
        reasons.append("promotion_envelope_invalid")

    try:
        from .flat_population_reporting import native_flat_population_reporting_records
        from simulation.metrics_pkg.population_reporting import (
            FLAT_POPULATION_REPORTING_SCHEMA_VERSION,
        )

        from .interchange_projection import SCHEMA_VERSION as INTERCHANGE_SCHEMA_VERSION
        from .lifecycle_events import FIELD_DTYPES, SCHEMA_VERSION as LIFECYCLE_SCHEMA_VERSION
        from .public_metrics_overlay import (
            BEHAVIOR_ROW_FIELDS,
            NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS,
            NATIVE_LINEAGE_METRICS_INPUT_SCHEMA_VERSION,
        )

        if not callable(native_flat_population_reporting_records):
            reasons.append("resident_reporting_adapter_not_callable")
        if len(BEHAVIOR_ROW_FIELDS) != 21 or len(set(BEHAVIOR_ROW_FIELDS)) != 21:
            reasons.append("behavior_reporting_contract_malformed")
        if len(NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS) != 66 or len(
            set(NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS)
        ) != 66:
            reasons.append("resident_reporting_required_fields_incomplete")
        if not FIELD_DTYPES or len(FIELD_DTYPES) != len(set(FIELD_DTYPES)):
            reasons.append("typed_lifecycle_reporting_contract_malformed")
        schema_pairs = (
            (reporting_contract.schemas.population, FLAT_POPULATION_REPORTING_SCHEMA_VERSION),
            (reporting_contract.schemas.lineage, NATIVE_LINEAGE_METRICS_INPUT_SCHEMA_VERSION),
            (reporting_contract.schemas.typed_lifecycle, LIFECYCLE_SCHEMA_VERSION),
            (reporting_contract.schemas.interchange, INTERCHANGE_SCHEMA_VERSION),
        )
        if any(expected != actual for expected, actual in schema_pairs):
            reasons.append("reporting_schema_version_mismatch")
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        reasons.append("resident_reporting_contract_unavailable")

    if runner is not None:
        overlay = getattr(runner, "_native_runtime_public_metrics_overlay", None)
        if isinstance(overlay, Mapping):
            source = str(overlay.get("source", "") or "")
            if source and source != "native_owner_checkpoint_projection":
                reasons.append("stale_python_agent_population_source")
    return NativeReportingQualification(
        supported=not reasons,
        profile=reporting_contract.profile,
        qualification=reporting_contract.qualification,
        schema_version=reporting_contract.schema_version,
        reasons=tuple(sorted(set(reasons))),
    )


def qualify_native_extension(
    *,
    envelope: FlatNativePromotionEnvelope | None = None,
) -> NativeExtensionQualification:
    reasons: list[str] = []
    if envelope is None:
        envelope, envelope_reason, envelope_error = _promotion_envelope()
        if envelope is None:
            return NativeExtensionQualification(
                supported=False,
                error=envelope_error,
                reasons=(envelope_reason,),
            )
    binary_contract = envelope.binary_contract
    try:
        status = native_extension_status()
        for boolean_field in (
            "available",
            "abi_compatible",
            "output_sha256_verified",
            "runtime_jit",
        ):
            if type(getattr(status, boolean_field)) is not bool:
                raise TypeError(f"native extension status {boolean_field} must be bool")
        for string_field in (
            "extension_path",
            "extension_abi_version",
            "extension_version",
            "error",
        ):
            if type(getattr(status, string_field)) is not str:
                raise TypeError(f"native extension status {string_field} must be str")
        path = str(status.extension_path or "")
        abi_version = str(status.extension_abi_version or "")
        extension_version = str(status.extension_version or "")
        available = bool(status.available)
        abi_compatible = bool(status.abi_compatible)
        hash_verified = bool(status.output_sha256_verified)
        runtime_jit = bool(status.runtime_jit)
        error = str(status.error or "")
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        return NativeExtensionQualification(
            supported=False,
            error=repr(exc),
            reasons=("extension_status_malformed",),
        )
    except Exception as exc:
        return NativeExtensionQualification(
            supported=False,
            error=repr(exc),
            reasons=("extension_status_query_failed",),
        )

    if binary_contract.prebuilt_extension_required and (not path or not Path(path).is_file()):
        reasons.extend(("prebuilt_extension_missing", "prebuilt_native_runtime_extension_required"))
    if abi_version != binary_contract.extension_abi_version:
        reasons.append("native_extension_abi_mismatch")
    if extension_version != binary_contract.extension_version:
        reasons.append("native_extension_version_mismatch")
    if not abi_compatible and abi_version == binary_contract.extension_abi_version and extension_version == binary_contract.extension_version:
        reasons.append("native_extension_capability_malformed")
    if binary_contract.extension_sha256_required and not hash_verified:
        reasons.append("native_extension_hash_mismatch")
    if runtime_jit and not binary_contract.runtime_jit_allowed:
        reasons.append("runtime_jit_not_allowed")
    if not available and not reasons:
        reasons.append("prebuilt_extension_unavailable")

    digest = ""
    if path and Path(path).is_file():
        try:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            reasons.append("native_extension_hash_read_failed")

    loaded = False
    if not reasons:
        try:
            load_prebuilt_extension(path)
            loaded = True
        except ImportError as exc:
            error = repr(exc)
            reasons.append("native_extension_import_failed")
        except Exception as exc:
            error = repr(exc)
            message = str(exc)
            if "loaded native extension ABI/version mismatch" in message:
                if f"abi={binary_contract.extension_abi_version}" not in message:
                    reasons.append("native_extension_abi_mismatch")
                if f"version={binary_contract.extension_version}" not in message:
                    reasons.append("native_extension_version_mismatch")
                if not reasons:
                    reasons.append("native_extension_load_failed")
            else:
                reasons.append("native_extension_load_failed")
    return NativeExtensionQualification(
        supported=not reasons,
        available=available,
        path=path,
        sha256=digest,
        abi_version=abi_version,
        extension_version=extension_version,
        abi_compatible=abi_compatible,
        hash_verified=hash_verified,
        loaded=loaded,
        runtime_jit=runtime_jit,
        error=error,
        reasons=tuple(sorted(set(reasons))),
    )


def _has_resident_gru_policy_rows(runner: Any | None) -> bool:
    if runner is None:
        return False
    rows = getattr(runner, "_policy_use_gru", None)
    if rows is None:
        pop = getattr(runner, "pop", None)
        if pop is None:
            raise RuntimeError("resident policy-family population is unavailable")
        alive_agents = getattr(pop, "alive_agents", None)
        if not callable(alive_agents):
            raise RuntimeError("resident policy-family rows are unavailable")
        for agent in alive_agents():
            genome = getattr(agent, "genome", None)
            if genome is None:
                raise RuntimeError("resident policy-family genome is unavailable")
            if bool(getattr(genome, "use_gru", False)) or int(getattr(genome, "memory_dim", 0) or 0) > 0:
                return True
        return False
    try:
        any_value = rows.any()
        if hasattr(any_value, "item"):
            return bool(any_value.item())
        return bool(any_value)
    except Exception:
        return any(bool(value) for value in rows)


def _shock_schedule_active(sim_cfg: Any, runner: Any | None) -> bool:
    try:
        configured = (
            _bool_cfg(sim_cfg, "shock_pulsed")
            or _bool_cfg(sim_cfg, "pressure_autopilot_allow_shock_enable")
            or float(getattr(sim_cfg, "shock_start_prob", 0.0) or 0.0) > 0.0
            or int(getattr(sim_cfg, "shock_period", 0) or 0) > 0
            or int(getattr(sim_cfg, "shock_duration", 0) or 0) > 0
        )
        if configured:
            return True
        if runner is None:
            return False
        return (
            int(getattr(runner, "_shock_active_rem", 0) or 0) > 0
            or int(getattr(runner, "_shock_cooldown_rem", 0) or 0) > 0
            or bool(getattr(runner, "_shock_last_active", False))
        )
    except Exception:
        return True


def _prediction_bookkeeping_active(sim_cfg: Any) -> bool:
    return not (
        _bool_cfg(sim_cfg, "ablate_prediction")
        or _bool_cfg(sim_cfg, "force_prediction_disabled")
    )


def _unsupported_reasons(
    sim_cfg: Any,
    env_cfg: Any | None,
    runner: Any | None,
    envelope: FlatNativePromotionEnvelope | None,
) -> list[str]:
    reasons: list[str] = []
    if envelope is None:
        return reasons
    profile = envelope.supported_profile
    device = _normal_device(getattr(sim_cfg, "device", "cpu"))
    bank_device = _normal_device(getattr(sim_cfg, "bank_device", "cpu"))
    if device != profile.device.simulation or bank_device != profile.device.tensor_bank:
        reasons.append("unsupported_device")
        reasons.append("cuda_or_non_cpu_tensor_runtime_not_supported")

    fixed_family = _str_cfg(sim_cfg, "native_runtime_fixed_policy_family").lower()
    if fixed_family != profile.policy.family:
        reasons.append("unsupported_policy_family")
        reasons.append("fixed_policy_family_not_supported")
    if fixed_family == profile.policy.family and not _bool_cfg(sim_cfg, "native_runtime_cxx26_scope_lock_enable"):
        reasons.append("cxx26_scope_lock_not_enabled")
    if profile.lifecycle.branch == "flat_generation_birth_commit" and not _bool_cfg(
        sim_cfg, "native_runtime_birth_commit_enable", True
    ):
        reasons.append("unsupported_lifecycle_branch")
        reasons.append("flat_generation_birth_commit_required")
    if fixed_family == profile.policy.family:
        try:
            has_gru_rows = _has_resident_gru_policy_rows(runner)
        except Exception:
            reasons.append("policy_family_detection_failed")
        else:
            if has_gru_rows:
                reasons.append("unsupported_gru_rows")
                reasons.append("fixed_mlp_policy_family_gru_rows_not_supported")

    if _shock_schedule_active(sim_cfg, runner):
        reasons.append("native_shock_schedule_not_supported")

    if profile.structural_operators.c1_mode == "disabled" and _bool_cfg(sim_cfg, "genome_c1_enable"):
        reasons.append("unsupported_structural_operator_c1")
        reasons.append("c1_structural_mutation_not_supported")
    if profile.structural_operators.c2_mode == "disabled" and _bool_cfg(sim_cfg, "genome_c2_enable"):
        reasons.append("unsupported_structural_operator_c2")
        reasons.append("c2_structural_recombination_not_supported")
    for name, reason in (
        ("enable_evolvable_sensor_channel_mask", "evolvable_sensor_channel_mask_not_supported"),
        ("enable_evolvable_sensor_radius", "evolvable_sensor_radius_not_supported"),
        ("enable_evolvable_introspection_channels", "evolvable_introspection_channels_not_supported"),
        ("enable_evolvable_self_regulation_controls", "evolvable_self_regulation_controls_not_supported"),
        ("enable_evolvable_self_modification_targets", "evolvable_self_modification_targets_not_supported"),
        ("enable_evolvable_directed_inheritance_channels", "evolvable_directed_inheritance_channels_not_supported"),
    ):
        if _bool_cfg(sim_cfg, name):
            reasons.append(reason)

    if env_cfg is not None:
        world_alias = _str_cfg(
            env_cfg,
            "world",
            _str_cfg(env_cfg, "world_mode", profile.world),
        ).lower()
        allowed_worlds = {profile.world, *profile.world_aliases_allowed}
        if world_alias not in allowed_worlds:
            reasons.append("unsupported_world_alias")
        active_patch = [name for name in _PATCH_SWITCHES if _bool_cfg(env_cfg, name)]
        if active_patch:
            reasons.append("unsupported_world_patch")
            reasons.extend(f"unsupported_env_switch:{name}" for name in active_patch)
            if "patch_world_enable" in active_patch and _bool_cfg(
                sim_cfg, "native_runtime_birth_commit_enable", True
            ):
                reasons.append("patch_churn_birth_commit_not_supported")
        if "patch_world_observation_enable" in active_patch:
            reasons.append("patch_world_observation_not_supported")
        active_jungle = [name for name in _JUNGLE_SWITCHES if _bool_cfg(env_cfg, name)]
        if active_jungle:
            reasons.append("unsupported_world_jungle")
            reasons.extend(f"unsupported_env_switch:{name}" for name in active_jungle)
        active_delayed = [name for name in _DELAYED_WORLD_SWITCHES if _bool_cfg(env_cfg, name)]
        if active_delayed:
            reasons.append("unsupported_world_wdl")
            reasons.extend(f"unsupported_env_switch:{name}" for name in active_delayed)
        if _bool_cfg(env_cfg, "reproduce_requires_mate"):
            reasons.append("unsupported_reproduction_mode_mate_required")
            reasons.append("mate_required_exact_recombination_rng_not_supported")
        for enabled_name, maximum_name, reason in (
            ("enable_injury", "injury_max", "invalid_injury_max"),
            ("enable_scar", "scar_max", "invalid_scar_max"),
        ):
            if not _bool_cfg(env_cfg, enabled_name):
                continue
            try:
                maximum = float(getattr(env_cfg, maximum_name))
            except (AttributeError, TypeError, ValueError, OverflowError):
                reasons.append(reason)
            else:
                if not math.isfinite(maximum) or maximum <= 0.0:
                    reasons.append(reason)

    if not profile.chemistry.functional_coupling_allowed and _bool_cfg(
        sim_cfg, "chemistry_functional_coupling_enable"
    ):
        reasons.append("unsupported_chemistry_or_maintenance")
    if not profile.chemistry.maintenance_allowed and _bool_cfg(sim_cfg, "maintenance_enable"):
        reasons.append("unsupported_chemistry_or_maintenance")
    if _prediction_bookkeeping_active(sim_cfg):
        reasons.append("prediction_bookkeeping_not_native_owned")

    return sorted(set(reasons))


def resolve_native_safe_default(
    sim_cfg: Any,
    env_cfg: Any | None = None,
    runner: Any | None = None,
) -> NativeSafeDefaultDecision:
    """Resolve the canonical native safe-default decision for one run.

    Envelope qualification and automatic default promotion are independent.
    Before promotion, supported explicit requests run native, normal requests
    stay on reference, and unsupported explicit requests fail closed.
    """

    profile = _str_cfg(sim_cfg, "native_execution_profile", NATIVE_SAFE_DEFAULT_PROFILE) or NATIVE_SAFE_DEFAULT_PROFILE
    enabled = _bool_cfg(sim_cfg, "native_safe_default_enable", True)
    legacy_request = _bool_cfg(sim_cfg, "native_runtime_enable")
    diagnostic_request = _bool_cfg(sim_cfg, "native_runtime_diagnostic_enable")
    explicit_reference = profile in {"none", "reference"}
    native_requested = bool((legacy_request or diagnostic_request) and not explicit_reference)
    envelope, envelope_reason, envelope_error = _promotion_envelope()
    if envelope is None:
        reporting = NativeReportingQualification(
            supported=False,
            reasons=(envelope_reason,),
        )
        checkpoint = NativeCheckpointQualification(
            supported=False,
            mode="off",
            effective_period_gen=0,
            strict_replay=False,
            partial_generation=False,
            reasons=(envelope_reason,),
            supported_routes=(),
        )
        extension = NativeExtensionQualification(
            supported=False,
            error=envelope_error,
            reasons=(envelope_reason,),
        )
    else:
        reporting = qualify_native_reporting(sim_cfg, runner, envelope=envelope)
        checkpoint = qualify_native_checkpoint_mode(sim_cfg, runner, envelope=envelope)
        extension = qualify_native_extension(envelope=envelope)

    reasons: list[str] = []
    if profile not in {NATIVE_SAFE_DEFAULT_PROFILE, "none", "reference"}:
        reasons.append("unsupported_execution_profile")
    if not enabled:
        reasons.append("native_safe_default_disabled")
    if envelope is None:
        reasons.append(envelope_reason)
    reasons.extend(_unsupported_reasons(sim_cfg, env_cfg, runner, envelope))
    reasons.extend(reporting.reasons)
    reasons.extend(checkpoint.reasons)
    reasons.extend(extension.reasons)
    if _bool_cfg(sim_cfg, "native_runtime_allow_runtime_jit_build"):
        reasons.append("runtime_jit_not_allowed")

    certificate = _qualify_promotion_certificate(extension)
    if NATIVE_SAFE_DEFAULT_PROMOTION_ACCEPTED:
        reasons.extend(certificate.reasons)

    envelope_supported = (
        enabled
        and profile == NATIVE_SAFE_DEFAULT_PROFILE
        and not reasons
    )
    promotion_accepted = bool(
        NATIVE_SAFE_DEFAULT_PROMOTION_ACCEPTED and certificate.accepted
    )
    lifecycle_default_active = bool(envelope_supported and promotion_accepted and not explicit_reference)
    native_execution_active = bool(envelope_supported and (native_requested or lifecycle_default_active))
    native_supported = bool(envelope_supported)
    fail_closed = bool(native_requested and not native_supported)
    reference_required = not native_execution_active
    if lifecycle_default_active:
        selected = SAFE_INFRASTRUCTURE_BLOCKS
    elif native_execution_active:
        selected = tuple(block for block in SAFE_INFRASTRUCTURE_BLOCKS if block != "native_lifecycle_default")
    else:
        selected = ()
    deprecated = list(checkpoint.deprecated_aliases_used)
    if legacy_request:
        deprecated.append("native_runtime_enable")
    fallback_policy = (
        "fail_closed"
        if fail_closed
        else (
            "explicit_reference"
            if explicit_reference
            else (
                "native_lifecycle_default"
                if lifecycle_default_active
                else ("explicit_native_request" if native_execution_active else "reference_required")
            )
        )
    )
    return NativeSafeDefaultDecision(
        requested_execution_mode=(
            "explicit_reference"
            if explicit_reference
            else ("diagnostic_native" if diagnostic_request else ("legacy_native" if legacy_request else "auto"))
        ),
        native_safe_default_enabled=enabled,
        native_requested=native_requested,
        native_diagnostic_requested=diagnostic_request,
        native_envelope_supported=envelope_supported,
        native_supported=native_supported,
        promotion_accepted=promotion_accepted,
        promotion_certificate_path=certificate.path,
        promotion_certificate_sha256=certificate.sha256,
        promotion_certificate_candidate_commit_sha=certificate.candidate_commit_sha,
        promotion_certificate_error=certificate.error,
        native_execution_active=native_execution_active,
        reference_required=reference_required,
        fail_closed=fail_closed,
        prebuilt_extension_required=(
            envelope.binary_contract.prebuilt_extension_required if envelope is not None else True
        ),
        runtime_jit_allowed=(
            envelope.binary_contract.runtime_jit_allowed if envelope is not None else False
        ),
        runtime_jit_prohibited=(
            not envelope.binary_contract.runtime_jit_allowed if envelope is not None else True
        ),
        lifecycle_default_active=lifecycle_default_active,
        known_multigeneration_parity_block=False,
        selected_native_blocks=tuple(selected),
        unsupported_reasons=tuple(sorted(set(reasons))),
        fallback_policy=fallback_policy,
        prebuilt_extension_available=extension.available,
        prebuilt_extension_error=extension.error,
        extension_path=extension.path,
        extension_sha256=extension.sha256,
        extension_abi_version=extension.abi_version,
        extension_version=extension.extension_version,
        extension_abi_compatible=extension.abi_compatible,
        extension_hash_verified=extension.hash_verified,
        extension_loaded=extension.loaded,
        extension_runtime_jit=extension.runtime_jit,
        reporting_supported=reporting.supported,
        reporting_profile=reporting.profile,
        reporting_qualification=reporting.qualification,
        reporting_schema_version=reporting.schema_version,
        reporting_reasons=reporting.reasons,
        checkpoint_supported=checkpoint.supported,
        checkpoint_mode=checkpoint.mode,
        checkpoint_effective_period_gen=checkpoint.effective_period_gen,
        checkpoint_strict_replay=checkpoint.strict_replay,
        checkpoint_partial_generation=checkpoint.partial_generation,
        checkpoint_reasons=checkpoint.reasons,
        checkpoint_supported_routes=checkpoint.supported_routes,
        benchmark_qualified=bool(envelope_supported and lifecycle_default_active),
        promotion_envelope_version=(
            envelope.schema_version if envelope is not None else FLAT_NATIVE_PROMOTION_ENVELOPE_VERSION
        ),
        promotion_envelope_error=envelope_error,
        deprecated_aliases_used=tuple(sorted(set(deprecated))),
    )


def native_generation_requested(runner: Any) -> bool:
    sim_cfg = getattr(runner, "sim_cfg", None)
    env_cfg = getattr(runner, "env_cfg", None)
    if sim_cfg is None:
        return False
    decision = resolve_native_safe_default(sim_cfg, env_cfg, runner)
    try:
        setattr(runner, "_native_safe_default_decision", decision.to_dict())
    except Exception:
        pass
    if decision.fail_closed:
        joined = ",".join(decision.unsupported_reasons)
        raise RuntimeError(f"EIEL-NATIVE-RUNTIME-UNSUPPORTED: {joined}")
    return bool(decision.native_execution_active)
