"""Strict loader for the versioned Flat native promotion envelope."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import ABI_VERSION, EXTENSION_VERSION, SCHEMA_VERSION as RUNTIME_SCHEMA_VERSION


CONTRACT_SCHEMA_VERSION = "eiel_flat_native_promotion_envelope_v1"
CONTRACT_ID = "flat_native_promotion_envelope_v1"
CONTRACT_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_RELATIVE_PATH = Path(
    "registries/performance/flat_native_promotion_envelope_v1.json"
)

_CHECKPOINT_MODES = (
    ("off", True, "exact_no_checkpoint"),
    ("complete_generation_write", True, "exact_atomic_generation_boundary_v2"),
    ("native_save_resume", True, "exact_generation_boundary_v2"),
    ("reference_to_native", True, "exact_four_way_interchange"),
    ("native_to_reference", True, "exact_four_way_interchange"),
    ("native_to_reference_to_native", True, "exact_four_way_interchange"),
    ("partial_generation", False, "unsupported"),
)

_REQUIRED_OUTPUT_FILES = (
    "metrics.csv",
    "behaviors.csv",
    "metrics/lineage_descriptors.jsonl",
    "metrics/lineage_summary.csv",
    "metrics/typed_lifecycle_events.csv",
    "metrics/typed_lifecycle_events.jsonl",
    "viz/metrics_summary.csv",
)

_BYTE_EXACT_FILES = (
    "behaviors.csv",
    "metrics/lineage_descriptors.jsonl",
    "metrics/lineage_summary.csv",
    "metrics/typed_lifecycle_events.csv",
    "metrics/typed_lifecycle_events.jsonl",
)

_SEMANTIC_EXACT_FILES = (
    "metrics.csv",
    "viz/metrics_summary.csv",
)

_DISALLOWED_FEATURES = (
    "patch_world",
    "jungle_predator_mechanics",
    "wdl_delayed_world_mechanics",
    "gru_resident_hidden_memory",
    "prediction_or_coherence_mode",
    "cuda_or_non_cpu_device",
    "maintenance",
    "non_disabled_chemistry",
    "non_disabled_c1",
    "non_disabled_c2",
    "general_lifecycle_branch",
    "unsupported_diagnostics_or_output_profile",
    "partial_generation_checkpoint",
    "unsupported_policy_family",
    "world_alias",
    "mate_required_reproduction",
    "stochastic_shock",
    "periodic_shock",
    "pulsed_shock",
    "restored_active_shock_state",
    "shock_alias_or_override_activation",
    "evolvable_cognition_surfaces",
    "runtime_jit",
)

_WORKER_IDENTITY_FIELDS = (
    "extension_path",
    "extension_sha256",
    "extension_abi_version",
    "extension_version",
    "runtime_jit",
    "native_owner_active",
    "comparator_source_commit",
)

_REQUIRED_ARTIFACTS = (
    "candidate_release_commit",
    "flat_owned_clean_source_manifest",
    "final_extension_build_manifest",
    "full_regression",
    "representative_g2_parity",
    "representative_g4_parity",
    "representative_g8_parity",
    "production_checkpoint_four_way",
    "long_horizon_a",
    "long_horizon_b",
    "long_horizon_c",
    "long_horizon_d",
    "four_cell_output_fairness",
    "authoritative_five_arm_benchmark",
    "promotion_certificate",
    "post_promotion_routing_matrix",
)

_REQUIRED_PROVENANCE_FIELDS = (
    "candidate_release_commit_sha",
    "flat_owned_source_clean",
    "extension_sha256",
    "extension_abi_version",
    "extension_version",
    "cpp_source_sha256",
    "binding_source_sha256",
    "build_command",
    "compiler_environment",
    "regression_log_sha256",
    "parity_result_sha256",
    "fairness_result_sha256",
    "benchmark_manifest_sha256",
    "benchmark_result_sha256",
    "long_horizon_result_sha256",
)

_CERTIFICATE_REQUIRED_FIELDS = (
    "supported_envelope",
    "candidate_release_commit_sha",
    "extension_sha256",
    "extension_abi_version",
    "extension_version",
    "reporting_schema_version",
    "checkpoint_qualification",
    "parity_evidence_version",
    "g2_result_sha256",
    "g4_result_sha256",
    "g8_result_sha256",
    "checkpoint_result_sha256",
    "long_horizon_a_result_sha256",
    "long_horizon_b_result_sha256",
    "long_horizon_c_result_sha256",
    "long_horizon_d_result_sha256",
    "full_regression_sha256",
    "output_fairness_result_sha256",
    "output_fairness_ratio",
    "benchmark_result_sha256",
    "current_reference_w1_median_wall_s",
    "current_native_w1_median_wall_s",
    "w1_native_speedup",
    "current_reference_w6_median_wall_s",
    "current_native_w6_median_wall_s",
    "w6_native_speedup",
    "measured_arm_coefficients_of_variation",
    "peak_memory_result",
    "hidden_fallback_false",
    "mixed_ownership_false",
    "runtime_jit_false",
    "decision_timestamp",
)

_ROUTING_RULES = (
    "unsupported_or_unqualified_request",
    "missing_extension",
    "abi_mismatch",
    "version_mismatch",
    "extension_import_failure",
    "extension_load_failure",
    "native_runtime_failure",
)

_NATIVE_RESULT_REJECTIONS = (
    "ok_false",
    "status_not_passed",
    "malformed_capability_response",
    "incomplete_generation",
    "missing_finalization",
    "missing_required_output",
    "inconsistent_owner_identity",
    "hidden_fallback",
    "runtime_jit",
    "extension_hash_mismatch",
    "partial_checkpoint",
    "output_flush_failure",
    "mixed_ownership",
    "worker_identity_mismatch",
    "stale_comparator",
    "reporting_unqualified",
    "benchmark_unqualified",
)

_REJECTION_ENFORCEMENT_SCOPES = (
    "runner",
    "campaign_analyzer",
    "benchmark_analyzer",
    "promotion_gate",
)

_CHECKPOINT_WRITE_FAILURES = (
    "serialization_failure",
    "sidecar_write_failure",
    "atomic_rename_failure",
    "fsync_or_flush_failure",
    "insufficient_space",
    "permission_failure",
    "malformed_projection",
    "digest_failure",
)

_FAIRNESS_REQUIREMENTS = (
    "exact_scientific_semantics",
    "exact_public_reporting",
    "same_filenames",
    "same_checkpoint_cadence",
    "complete_output_flush",
    "failed_or_incomplete_cells_excluded",
    "w6_worker_extension_identity_homogeneous",
    "no_hidden_fallback",
    "no_mixed_ownership",
    "runtime_jit_false",
)

_POST_PROMOTION_CASES = (
    "supported_flat_cpu_fixed_mlp_auto_selects_native",
    "explicit_reference_remains_reference",
    "supported_explicit_native_selects_native",
    "unsupported_auto_routes_reference_with_structured_reason",
    "unsupported_explicit_native_fails_closed",
    "missing_extension_auto_routes_reference_with_structured_reason",
    "missing_extension_explicit_native_fails_closed",
    "abi_or_version_mismatch_distinguishes_auto_and_explicit_native",
    "failed_native_result_rejected",
    "checkpoint_write_error_rejected",
    "all_workers_use_one_extension_identity",
    "default_native_scientific_smoke_exact",
)

_POST_PROMOTION_REQUIREMENTS = (
    "default_native_provenance",
    "extension_sha256_match",
    "runtime_jit_false",
    "hidden_fallback_false",
    "mixed_ownership_false",
    "exact_scientific_outputs",
    "exact_public_outputs",
    "complete_output_flush",
)


class PromotionEnvelopeValidationError(ValueError):
    """Raised when the promotion contract cannot be trusted."""


@dataclass(frozen=True, slots=True)
class PolicyProfile:
    family: str
    selection: str
    resident_gru_allowed: bool


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    simulation: str
    tensor_bank: str
    cuda_allowed: bool


@dataclass(frozen=True, slots=True)
class LifecycleProfile:
    branch: str
    reproduction: str
    general_lifecycle_allowed: bool


@dataclass(frozen=True, slots=True)
class ChemistryProfile:
    mode: str
    functional_coupling_allowed: bool
    maintenance_allowed: bool


@dataclass(frozen=True, slots=True)
class StructuralOperatorProfile:
    c1_mode: str
    c2_mode: str


@dataclass(frozen=True, slots=True)
class CheckpointMode:
    name: str
    supported: bool
    qualification: str


@dataclass(frozen=True, slots=True)
class CheckpointProfile:
    modes: tuple[CheckpointMode, ...]

    def mode(self, name: str) -> CheckpointMode:
        for mode in self.modes:
            if mode.name == name:
                return mode
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class ReportingSchemas:
    population: str
    lineage: str
    typed_lifecycle: str
    interchange: str


@dataclass(frozen=True, slots=True)
class ReportingProfile:
    schema_version: str
    profile: str
    qualification: str
    schemas: ReportingSchemas
    metrics_csv_exact: bool
    behaviors_exact: bool
    lineage_exact: bool
    typed_lifecycle_exact: bool
    stale_python_agent_population_source_allowed: bool
    missing_resident_field_allowed: bool
    native_owner_provenance_required: bool
    finalization_required: bool


@dataclass(frozen=True, slots=True)
class OutputProfile:
    campaign_profile: str
    profile_id: str
    required_files: tuple[str, ...]
    byte_exact_files: tuple[str, ...]
    semantic_exact_files: tuple[str, ...]
    same_filenames_required: bool
    same_checkpoint_cadence_required: bool
    complete_flush_required: bool
    route_only_diagnostics_allowed: bool
    partial_outputs_promotion_eligible: bool


@dataclass(frozen=True, slots=True)
class SupportedProfile:
    world: str
    world_aliases_allowed: tuple[str, ...]
    policy: PolicyProfile
    device: DeviceProfile
    lifecycle: LifecycleProfile
    chemistry: ChemistryProfile
    structural_operators: StructuralOperatorProfile
    checkpoint: CheckpointProfile
    reporting: ReportingProfile
    output: OutputProfile


@dataclass(frozen=True, slots=True)
class BinaryContract:
    runtime_schema_version: str
    extension_abi_version: str
    extension_version: str
    prebuilt_extension_required: bool
    runtime_jit_allowed: bool
    extension_sha256_required: bool
    worker_homogeneity_required: bool
    worker_identity_fields: tuple[str, ...]
    stale_comparator_allowed: bool
    mixed_ownership_allowed: bool


@dataclass(frozen=True, slots=True)
class EvidenceRequirements:
    required_artifacts: tuple[str, ...]
    required_provenance_fields: tuple[str, ...]
    certificate_required_fields: tuple[str, ...]
    all_results_from_committed_source: bool
    uncommitted_implementation_allowed: bool
    hash_algorithm: str
    certificate_schema_version: str
    certificate_sha256_required: bool
    promotion_constants_must_reference_certificate: bool


@dataclass(frozen=True, slots=True)
class RoutingRule:
    trigger: str
    phase: str
    auto_behavior: str
    explicit_native_behavior: str


@dataclass(frozen=True, slots=True)
class FailureAtomicity:
    success_marker_allowed: bool
    promotion_eligible_result_allowed: bool
    partial_metrics_accepted: bool
    partial_checkpoint_valid: bool
    durable_failure_status_required: bool
    durable_first_cause_required: bool
    mid_cell_fallback_allowed: bool
    partial_reference_continuation_allowed: bool
    partial_output_handling: str
    benchmark_inventory_excludes_failed_or_incomplete: bool


@dataclass(frozen=True, slots=True)
class FailureSemantics:
    routing_rules: tuple[RoutingRule, ...]
    native_result_rejections: tuple[str, ...]
    rejection_enforcement_scopes: tuple[str, ...]
    checkpoint_write_failures: tuple[str, ...]
    checkpoint_write_failure_behavior: str
    failure_atomicity: FailureAtomicity


@dataclass(frozen=True, slots=True)
class MeasurementPlan:
    primary_statistic: str
    w1_repetitions_min: int
    w6_repetitions_min: int
    warmup_repetitions: int


@dataclass(frozen=True, slots=True)
class SpeedupThreshold:
    comparison: str
    minimum_speedup: float
    inclusive: bool


@dataclass(frozen=True, slots=True)
class HistoricalPerformance:
    comparison: str
    required_for_promotion: bool
    incomparable_contract_fails_promotion: bool


@dataclass(frozen=True, slots=True)
class StabilityThreshold:
    statistic: str
    maximum: float
    inclusive: bool
    scope: str


@dataclass(frozen=True, slots=True)
class OutputFairnessThreshold:
    statistic: str
    minimum: float
    maximum: float
    inclusive: bool
    scope: str


@dataclass(frozen=True, slots=True)
class MemoryThreshold:
    statistic: str
    maximum: float
    inclusive: bool
    exception_policy: str
    accepted_exceptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    measurement: MeasurementPlan
    primary_performance: SpeedupThreshold
    secondary_performance: SpeedupThreshold
    historical_performance: HistoricalPerformance
    stability: StabilityThreshold
    output_fairness: OutputFairnessThreshold
    memory: MemoryThreshold
    fairness_requirements: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExpectedPromotionState:
    native_safe_default_promotion_accepted: bool
    parity_evidence_version_source: str
    certificate_pointer_required: bool


@dataclass(frozen=True, slots=True)
class PostPromotionProof:
    required: bool
    after_atomic_promotion_commit: bool
    expected_promotion_state: ExpectedPromotionState
    cases: tuple[str, ...]
    requirements: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FlatNativePromotionEnvelope:
    schema_version: str
    contract_id: str
    contract_version: int
    supported_profile: SupportedProfile
    disallowed_features: tuple[str, ...]
    binary_contract: BinaryContract
    evidence_requirements: EvidenceRequirements
    failure_semantics: FailureSemantics
    promotion_thresholds: PromotionThresholds
    post_promotion_proof: PostPromotionProof


def promotion_envelope_contract_path(path: str | Path | None = None) -> Path:
    """Resolve a contract path from the repository, independent of cwd."""

    if path is None:
        return (PROJECT_ROOT / DEFAULT_CONTRACT_RELATIVE_PATH).resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _object(value: object, path: str, fields: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionEnvelopeValidationError(f"{path} must be an object")
    keys = set(value.keys())
    if not all(isinstance(key, str) for key in keys):
        raise PromotionEnvelopeValidationError(f"{path} contains a non-string field name")
    expected = set(fields)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing:
        raise PromotionEnvelopeValidationError(f"{path} missing required fields: {missing}")
    if unknown:
        raise PromotionEnvelopeValidationError(f"{path} contains unknown fields: {unknown}")
    return value


def _array(value: object, path: str) -> list[Any]:
    if type(value) is not list:
        raise PromotionEnvelopeValidationError(f"{path} must be an array")
    return value


def _string(value: object, path: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise PromotionEnvelopeValidationError(f"{path} must be a non-empty trimmed string")
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise PromotionEnvelopeValidationError(f"{path} must be a boolean")
    return value


def _integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise PromotionEnvelopeValidationError(f"{path} must be an integer")
    return value


def _number(value: object, path: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise PromotionEnvelopeValidationError(f"{path} must be a finite number")
    return float(value)


def _literal_string(value: object, path: str, expected: str) -> str:
    actual = _string(value, path)
    if actual != expected:
        raise PromotionEnvelopeValidationError(
            f"{path} must be {expected!r} for contract v{CONTRACT_VERSION}, got {actual!r}"
        )
    return actual


def _literal_boolean(value: object, path: str, expected: bool) -> bool:
    actual = _boolean(value, path)
    if actual is not expected:
        raise PromotionEnvelopeValidationError(
            f"{path} must be {expected!r} for contract v{CONTRACT_VERSION}"
        )
    return actual


def _literal_integer(value: object, path: str, expected: int) -> int:
    actual = _integer(value, path)
    if actual != expected:
        raise PromotionEnvelopeValidationError(
            f"{path} must be {expected} for contract v{CONTRACT_VERSION}, got {actual}"
        )
    return actual


def _literal_number(value: object, path: str, expected: float) -> float:
    actual = _number(value, path)
    if actual != expected:
        raise PromotionEnvelopeValidationError(
            f"{path} must be {expected} for contract v{CONTRACT_VERSION}, got {actual}"
        )
    return actual


def _string_sequence(value: object, path: str, expected: tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(_array(value, path)))
    if len(values) != len(set(values)):
        raise PromotionEnvelopeValidationError(f"{path} contains duplicate values")
    if values != expected:
        raise PromotionEnvelopeValidationError(
            f"{path} must exactly match the contract v{CONTRACT_VERSION} preregistration"
        )
    return values


def _parse_policy(value: object) -> PolicyProfile:
    path = "$.supported_profile.policy"
    row = _object(value, path, ("family", "selection", "resident_gru_allowed"))
    return PolicyProfile(
        family=_literal_string(row["family"], f"{path}.family", "mlp"),
        selection=_literal_string(row["selection"], f"{path}.selection", "fixed"),
        resident_gru_allowed=_literal_boolean(
            row["resident_gru_allowed"], f"{path}.resident_gru_allowed", False
        ),
    )


def _parse_device(value: object) -> DeviceProfile:
    path = "$.supported_profile.device"
    row = _object(value, path, ("simulation", "tensor_bank", "cuda_allowed"))
    return DeviceProfile(
        simulation=_literal_string(row["simulation"], f"{path}.simulation", "cpu"),
        tensor_bank=_literal_string(row["tensor_bank"], f"{path}.tensor_bank", "cpu"),
        cuda_allowed=_literal_boolean(row["cuda_allowed"], f"{path}.cuda_allowed", False),
    )


def _parse_lifecycle(value: object) -> LifecycleProfile:
    path = "$.supported_profile.lifecycle"
    row = _object(value, path, ("branch", "reproduction", "general_lifecycle_allowed"))
    return LifecycleProfile(
        branch=_literal_string(
            row["branch"], f"{path}.branch", "flat_generation_birth_commit"
        ),
        reproduction=_literal_string(
            row["reproduction"], f"{path}.reproduction", "asexual_resident_birth_commit"
        ),
        general_lifecycle_allowed=_literal_boolean(
            row["general_lifecycle_allowed"], f"{path}.general_lifecycle_allowed", False
        ),
    )


def _parse_chemistry(value: object) -> ChemistryProfile:
    path = "$.supported_profile.chemistry"
    row = _object(
        value,
        path,
        ("mode", "functional_coupling_allowed", "maintenance_allowed"),
    )
    return ChemistryProfile(
        mode=_literal_string(row["mode"], f"{path}.mode", "disabled"),
        functional_coupling_allowed=_literal_boolean(
            row["functional_coupling_allowed"],
            f"{path}.functional_coupling_allowed",
            False,
        ),
        maintenance_allowed=_literal_boolean(
            row["maintenance_allowed"], f"{path}.maintenance_allowed", False
        ),
    )


def _parse_structural_operators(value: object) -> StructuralOperatorProfile:
    path = "$.supported_profile.structural_operators"
    row = _object(value, path, ("c1_mode", "c2_mode"))
    return StructuralOperatorProfile(
        c1_mode=_literal_string(row["c1_mode"], f"{path}.c1_mode", "disabled"),
        c2_mode=_literal_string(row["c2_mode"], f"{path}.c2_mode", "disabled"),
    )


def _parse_checkpoint(value: object) -> CheckpointProfile:
    path = "$.supported_profile.checkpoint"
    row = _object(value, path, ("modes",))
    raw_modes = _array(row["modes"], f"{path}.modes")
    modes: list[CheckpointMode] = []
    for index, raw_mode in enumerate(raw_modes):
        mode_path = f"{path}.modes[{index}]"
        mode = _object(raw_mode, mode_path, ("name", "supported", "qualification"))
        modes.append(
            CheckpointMode(
                name=_string(mode["name"], f"{mode_path}.name"),
                supported=_boolean(mode["supported"], f"{mode_path}.supported"),
                qualification=_string(mode["qualification"], f"{mode_path}.qualification"),
            )
        )
    actual = tuple((mode.name, mode.supported, mode.qualification) for mode in modes)
    if actual != _CHECKPOINT_MODES:
        raise PromotionEnvelopeValidationError(
            f"{path}.modes must exactly match the contract v{CONTRACT_VERSION} checkpoint matrix"
        )
    return CheckpointProfile(modes=tuple(modes))


def _parse_reporting(value: object) -> ReportingProfile:
    path = "$.supported_profile.reporting"
    row = _object(
        value,
        path,
        (
            "schema_version",
            "profile",
            "qualification",
            "schemas",
            "metrics_csv_exact",
            "behaviors_exact",
            "lineage_exact",
            "typed_lifecycle_exact",
            "stale_python_agent_population_source_allowed",
            "missing_resident_field_allowed",
            "native_owner_provenance_required",
            "finalization_required",
        ),
    )
    schemas_path = f"{path}.schemas"
    raw_schemas = _object(
        row["schemas"],
        schemas_path,
        ("population", "lineage", "typed_lifecycle", "interchange"),
    )
    schemas = ReportingSchemas(
        population=_literal_string(
            raw_schemas["population"], schemas_path + ".population", "flat_population_reporting_v1"
        ),
        lineage=_literal_string(
            raw_schemas["lineage"], schemas_path + ".lineage", "eiel_native_lineage_metrics_input_v1"
        ),
        typed_lifecycle=_literal_string(
            raw_schemas["typed_lifecycle"],
            schemas_path + ".typed_lifecycle",
            "eiel_typed_lifecycle_event_v1",
        ),
        interchange=_literal_string(
            raw_schemas["interchange"],
            schemas_path + ".interchange",
            "eiel_reference_native_interchange_projection_v1",
        ),
    )
    return ReportingProfile(
        schema_version=_literal_string(
            row["schema_version"],
            f"{path}.schema_version",
            "eiel_flat_native_reporting_v1",
        ),
        profile=_literal_string(
            row["profile"], f"{path}.profile", "compact_route_neutral_exact"
        ),
        qualification=_literal_string(
            row["qualification"], f"{path}.qualification", "route_neutral_exact"
        ),
        schemas=schemas,
        metrics_csv_exact=_literal_boolean(
            row["metrics_csv_exact"], f"{path}.metrics_csv_exact", True
        ),
        behaviors_exact=_literal_boolean(
            row["behaviors_exact"], f"{path}.behaviors_exact", True
        ),
        lineage_exact=_literal_boolean(row["lineage_exact"], f"{path}.lineage_exact", True),
        typed_lifecycle_exact=_literal_boolean(
            row["typed_lifecycle_exact"], f"{path}.typed_lifecycle_exact", True
        ),
        stale_python_agent_population_source_allowed=_literal_boolean(
            row["stale_python_agent_population_source_allowed"],
            f"{path}.stale_python_agent_population_source_allowed",
            False,
        ),
        missing_resident_field_allowed=_literal_boolean(
            row["missing_resident_field_allowed"],
            f"{path}.missing_resident_field_allowed",
            False,
        ),
        native_owner_provenance_required=_literal_boolean(
            row["native_owner_provenance_required"],
            f"{path}.native_owner_provenance_required",
            True,
        ),
        finalization_required=_literal_boolean(
            row["finalization_required"], f"{path}.finalization_required", True
        ),
    )


def _parse_output(value: object) -> OutputProfile:
    path = "$.supported_profile.output"
    row = _object(
        value,
        path,
        (
            "campaign_profile",
            "profile_id",
            "required_files",
            "byte_exact_files",
            "semantic_exact_files",
            "same_filenames_required",
            "same_checkpoint_cadence_required",
            "complete_flush_required",
            "route_only_diagnostics_allowed",
            "partial_outputs_promotion_eligible",
        ),
    )
    return OutputProfile(
        campaign_profile=_literal_string(
            row["campaign_profile"], f"{path}.campaign_profile", "compact_default"
        ),
        profile_id=_literal_string(
            row["profile_id"], f"{path}.profile_id", "compact_route_neutral_exact"
        ),
        required_files=_string_sequence(
            row["required_files"], f"{path}.required_files", _REQUIRED_OUTPUT_FILES
        ),
        byte_exact_files=_string_sequence(
            row["byte_exact_files"], f"{path}.byte_exact_files", _BYTE_EXACT_FILES
        ),
        semantic_exact_files=_string_sequence(
            row["semantic_exact_files"], f"{path}.semantic_exact_files", _SEMANTIC_EXACT_FILES
        ),
        same_filenames_required=_literal_boolean(
            row["same_filenames_required"], f"{path}.same_filenames_required", True
        ),
        same_checkpoint_cadence_required=_literal_boolean(
            row["same_checkpoint_cadence_required"],
            f"{path}.same_checkpoint_cadence_required",
            True,
        ),
        complete_flush_required=_literal_boolean(
            row["complete_flush_required"], f"{path}.complete_flush_required", True
        ),
        route_only_diagnostics_allowed=_literal_boolean(
            row["route_only_diagnostics_allowed"],
            f"{path}.route_only_diagnostics_allowed",
            False,
        ),
        partial_outputs_promotion_eligible=_literal_boolean(
            row["partial_outputs_promotion_eligible"],
            f"{path}.partial_outputs_promotion_eligible",
            False,
        ),
    )


def _parse_supported_profile(value: object) -> SupportedProfile:
    path = "$.supported_profile"
    row = _object(
        value,
        path,
        (
            "world",
            "world_aliases_allowed",
            "policy",
            "device",
            "lifecycle",
            "chemistry",
            "structural_operators",
            "checkpoint",
            "reporting",
            "output",
        ),
    )
    return SupportedProfile(
        world=_literal_string(row["world"], f"{path}.world", "flat"),
        world_aliases_allowed=_string_sequence(
            row["world_aliases_allowed"], f"{path}.world_aliases_allowed", ()
        ),
        policy=_parse_policy(row["policy"]),
        device=_parse_device(row["device"]),
        lifecycle=_parse_lifecycle(row["lifecycle"]),
        chemistry=_parse_chemistry(row["chemistry"]),
        structural_operators=_parse_structural_operators(row["structural_operators"]),
        checkpoint=_parse_checkpoint(row["checkpoint"]),
        reporting=_parse_reporting(row["reporting"]),
        output=_parse_output(row["output"]),
    )


def _parse_binary_contract(value: object) -> BinaryContract:
    path = "$.binary_contract"
    row = _object(
        value,
        path,
        (
            "runtime_schema_version",
            "extension_abi_version",
            "extension_version",
            "prebuilt_extension_required",
            "runtime_jit_allowed",
            "extension_sha256_required",
            "worker_homogeneity_required",
            "worker_identity_fields",
            "stale_comparator_allowed",
            "mixed_ownership_allowed",
        ),
    )
    return BinaryContract(
        runtime_schema_version=_literal_string(
            row["runtime_schema_version"], f"{path}.runtime_schema_version", RUNTIME_SCHEMA_VERSION
        ),
        extension_abi_version=_literal_string(
            row["extension_abi_version"], f"{path}.extension_abi_version", ABI_VERSION
        ),
        extension_version=_literal_string(
            row["extension_version"], f"{path}.extension_version", EXTENSION_VERSION
        ),
        prebuilt_extension_required=_literal_boolean(
            row["prebuilt_extension_required"], f"{path}.prebuilt_extension_required", True
        ),
        runtime_jit_allowed=_literal_boolean(
            row["runtime_jit_allowed"], f"{path}.runtime_jit_allowed", False
        ),
        extension_sha256_required=_literal_boolean(
            row["extension_sha256_required"], f"{path}.extension_sha256_required", True
        ),
        worker_homogeneity_required=_literal_boolean(
            row["worker_homogeneity_required"], f"{path}.worker_homogeneity_required", True
        ),
        worker_identity_fields=_string_sequence(
            row["worker_identity_fields"],
            f"{path}.worker_identity_fields",
            _WORKER_IDENTITY_FIELDS,
        ),
        stale_comparator_allowed=_literal_boolean(
            row["stale_comparator_allowed"], f"{path}.stale_comparator_allowed", False
        ),
        mixed_ownership_allowed=_literal_boolean(
            row["mixed_ownership_allowed"], f"{path}.mixed_ownership_allowed", False
        ),
    )


def _parse_evidence_requirements(value: object) -> EvidenceRequirements:
    path = "$.evidence_requirements"
    row = _object(
        value,
        path,
        (
            "required_artifacts",
            "required_provenance_fields",
            "certificate_required_fields",
            "all_results_from_committed_source",
            "uncommitted_implementation_allowed",
            "hash_algorithm",
            "certificate_schema_version",
            "certificate_sha256_required",
            "promotion_constants_must_reference_certificate",
        ),
    )
    return EvidenceRequirements(
        required_artifacts=_string_sequence(
            row["required_artifacts"], f"{path}.required_artifacts", _REQUIRED_ARTIFACTS
        ),
        required_provenance_fields=_string_sequence(
            row["required_provenance_fields"],
            f"{path}.required_provenance_fields",
            _REQUIRED_PROVENANCE_FIELDS,
        ),
        certificate_required_fields=_string_sequence(
            row["certificate_required_fields"],
            f"{path}.certificate_required_fields",
            _CERTIFICATE_REQUIRED_FIELDS,
        ),
        all_results_from_committed_source=_literal_boolean(
            row["all_results_from_committed_source"],
            f"{path}.all_results_from_committed_source",
            True,
        ),
        uncommitted_implementation_allowed=_literal_boolean(
            row["uncommitted_implementation_allowed"],
            f"{path}.uncommitted_implementation_allowed",
            False,
        ),
        hash_algorithm=_literal_string(row["hash_algorithm"], f"{path}.hash_algorithm", "sha256"),
        certificate_schema_version=_literal_string(
            row["certificate_schema_version"],
            f"{path}.certificate_schema_version",
            "flat_native_promotion_certificate_v1",
        ),
        certificate_sha256_required=_literal_boolean(
            row["certificate_sha256_required"], f"{path}.certificate_sha256_required", True
        ),
        promotion_constants_must_reference_certificate=_literal_boolean(
            row["promotion_constants_must_reference_certificate"],
            f"{path}.promotion_constants_must_reference_certificate",
            True,
        ),
    )


def _parse_routing_rules(value: object) -> tuple[RoutingRule, ...]:
    path = "$.failure_semantics.routing_rules"
    raw_rules = _array(value, path)
    rules: list[RoutingRule] = []
    for index, raw_rule in enumerate(raw_rules):
        rule_path = f"{path}[{index}]"
        row = _object(
            raw_rule,
            rule_path,
            ("trigger", "phase", "auto_behavior", "explicit_native_behavior"),
        )
        rules.append(
            RoutingRule(
                trigger=_string(row["trigger"], f"{rule_path}.trigger"),
                phase=_string(row["phase"], f"{rule_path}.phase"),
                auto_behavior=_string(row["auto_behavior"], f"{rule_path}.auto_behavior"),
                explicit_native_behavior=_string(
                    row["explicit_native_behavior"], f"{rule_path}.explicit_native_behavior"
                ),
            )
        )
    if tuple(rule.trigger for rule in rules) != _ROUTING_RULES:
        raise PromotionEnvelopeValidationError(
            f"{path} must exactly match the contract v{CONTRACT_VERSION} trigger matrix"
        )
    for rule in rules[:-1]:
        expected = (
            "before_native_selection",
            "route_reference_with_structured_reason",
            "fail_closed_no_reference_execution",
        )
        actual = (rule.phase, rule.auto_behavior, rule.explicit_native_behavior)
        if actual != expected:
            raise PromotionEnvelopeValidationError(
                f"{path} trigger {rule.trigger!r} contradicts preselection failure semantics"
            )
    terminal = rules[-1]
    if (
        terminal.phase,
        terminal.auto_behavior,
        terminal.explicit_native_behavior,
    ) != ("after_native_selection", "fail_cell_no_fallback", "fail_cell_no_fallback"):
        raise PromotionEnvelopeValidationError(
            f"{path} trigger 'native_runtime_failure' contradicts post-selection failure semantics"
        )
    return tuple(rules)


def _parse_failure_atomicity(value: object) -> FailureAtomicity:
    path = "$.failure_semantics.failure_atomicity"
    row = _object(
        value,
        path,
        (
            "success_marker_allowed",
            "promotion_eligible_result_allowed",
            "partial_metrics_accepted",
            "partial_checkpoint_valid",
            "durable_failure_status_required",
            "durable_first_cause_required",
            "mid_cell_fallback_allowed",
            "partial_reference_continuation_allowed",
            "partial_output_handling",
            "benchmark_inventory_excludes_failed_or_incomplete",
        ),
    )
    return FailureAtomicity(
        success_marker_allowed=_literal_boolean(
            row["success_marker_allowed"], f"{path}.success_marker_allowed", False
        ),
        promotion_eligible_result_allowed=_literal_boolean(
            row["promotion_eligible_result_allowed"],
            f"{path}.promotion_eligible_result_allowed",
            False,
        ),
        partial_metrics_accepted=_literal_boolean(
            row["partial_metrics_accepted"], f"{path}.partial_metrics_accepted", False
        ),
        partial_checkpoint_valid=_literal_boolean(
            row["partial_checkpoint_valid"], f"{path}.partial_checkpoint_valid", False
        ),
        durable_failure_status_required=_literal_boolean(
            row["durable_failure_status_required"],
            f"{path}.durable_failure_status_required",
            True,
        ),
        durable_first_cause_required=_literal_boolean(
            row["durable_first_cause_required"],
            f"{path}.durable_first_cause_required",
            True,
        ),
        mid_cell_fallback_allowed=_literal_boolean(
            row["mid_cell_fallback_allowed"], f"{path}.mid_cell_fallback_allowed", False
        ),
        partial_reference_continuation_allowed=_literal_boolean(
            row["partial_reference_continuation_allowed"],
            f"{path}.partial_reference_continuation_allowed",
            False,
        ),
        partial_output_handling=_literal_string(
            row["partial_output_handling"],
            f"{path}.partial_output_handling",
            "quarantine_or_mark_incomplete",
        ),
        benchmark_inventory_excludes_failed_or_incomplete=_literal_boolean(
            row["benchmark_inventory_excludes_failed_or_incomplete"],
            f"{path}.benchmark_inventory_excludes_failed_or_incomplete",
            True,
        ),
    )


def _parse_failure_semantics(value: object) -> FailureSemantics:
    path = "$.failure_semantics"
    row = _object(
        value,
        path,
        (
            "routing_rules",
            "native_result_rejections",
            "rejection_enforcement_scopes",
            "checkpoint_write_failures",
            "checkpoint_write_failure_behavior",
            "failure_atomicity",
        ),
    )
    return FailureSemantics(
        routing_rules=_parse_routing_rules(row["routing_rules"]),
        native_result_rejections=_string_sequence(
            row["native_result_rejections"],
            f"{path}.native_result_rejections",
            _NATIVE_RESULT_REJECTIONS,
        ),
        rejection_enforcement_scopes=_string_sequence(
            row["rejection_enforcement_scopes"],
            f"{path}.rejection_enforcement_scopes",
            _REJECTION_ENFORCEMENT_SCOPES,
        ),
        checkpoint_write_failures=_string_sequence(
            row["checkpoint_write_failures"],
            f"{path}.checkpoint_write_failures",
            _CHECKPOINT_WRITE_FAILURES,
        ),
        checkpoint_write_failure_behavior=_literal_string(
            row["checkpoint_write_failure_behavior"],
            f"{path}.checkpoint_write_failure_behavior",
            "fail_cell_no_success_result",
        ),
        failure_atomicity=_parse_failure_atomicity(row["failure_atomicity"]),
    )


def _parse_measurement(value: object) -> MeasurementPlan:
    path = "$.promotion_thresholds.measurement"
    row = _object(
        value,
        path,
        ("primary_statistic", "w1_repetitions_min", "w6_repetitions_min", "warmup_repetitions"),
    )
    return MeasurementPlan(
        primary_statistic=_literal_string(
            row["primary_statistic"], f"{path}.primary_statistic", "median_campaign_wall_s"
        ),
        w1_repetitions_min=_literal_integer(
            row["w1_repetitions_min"], f"{path}.w1_repetitions_min", 5
        ),
        w6_repetitions_min=_literal_integer(
            row["w6_repetitions_min"], f"{path}.w6_repetitions_min", 3
        ),
        warmup_repetitions=_literal_integer(
            row["warmup_repetitions"], f"{path}.warmup_repetitions", 1
        ),
    )


def _parse_speedup(value: object, path: str, comparison: str, minimum: float) -> SpeedupThreshold:
    row = _object(value, path, ("comparison", "minimum_speedup", "inclusive"))
    return SpeedupThreshold(
        comparison=_literal_string(row["comparison"], f"{path}.comparison", comparison),
        minimum_speedup=_literal_number(
            row["minimum_speedup"], f"{path}.minimum_speedup", minimum
        ),
        inclusive=_literal_boolean(row["inclusive"], f"{path}.inclusive", True),
    )


def _parse_historical(value: object) -> HistoricalPerformance:
    path = "$.promotion_thresholds.historical_performance"
    row = _object(
        value,
        path,
        ("comparison", "required_for_promotion", "incomparable_contract_fails_promotion"),
    )
    return HistoricalPerformance(
        comparison=_literal_string(
            row["comparison"], f"{path}.comparison", "historical_reference_to_current_native"
        ),
        required_for_promotion=_literal_boolean(
            row["required_for_promotion"], f"{path}.required_for_promotion", False
        ),
        incomparable_contract_fails_promotion=_literal_boolean(
            row["incomparable_contract_fails_promotion"],
            f"{path}.incomparable_contract_fails_promotion",
            False,
        ),
    )


def _parse_stability(value: object) -> StabilityThreshold:
    path = "$.promotion_thresholds.stability"
    row = _object(value, path, ("statistic", "maximum", "inclusive", "scope"))
    return StabilityThreshold(
        statistic=_literal_string(
            row["statistic"], f"{path}.statistic", "coefficient_of_variation"
        ),
        maximum=_literal_number(row["maximum"], f"{path}.maximum", 0.05),
        inclusive=_literal_boolean(row["inclusive"], f"{path}.inclusive", True),
        scope=_literal_string(
            row["scope"], f"{path}.scope", "every_promoted_measured_arm"
        ),
    )


def _parse_output_fairness(value: object) -> OutputFairnessThreshold:
    path = "$.promotion_thresholds.output_fairness"
    row = _object(value, path, ("statistic", "minimum", "maximum", "inclusive", "scope"))
    return OutputFairnessThreshold(
        statistic=_literal_string(
            row["statistic"], f"{path}.statistic", "native_to_reference_output_bytes_ratio"
        ),
        minimum=_literal_number(row["minimum"], f"{path}.minimum", 0.8),
        maximum=_literal_number(row["maximum"], f"{path}.maximum", 1.25),
        inclusive=_literal_boolean(row["inclusive"], f"{path}.inclusive", True),
        scope=_literal_string(
            row["scope"], f"{path}.scope", "every_measured_pair_and_aggregate"
        ),
    )


def _parse_memory(value: object) -> MemoryThreshold:
    path = "$.promotion_thresholds.memory"
    row = _object(
        value,
        path,
        ("statistic", "maximum", "inclusive", "exception_policy", "accepted_exceptions"),
    )
    return MemoryThreshold(
        statistic=_literal_string(
            row["statistic"],
            f"{path}.statistic",
            "native_peak_rss_over_current_reference_peak_rss_same_cell",
        ),
        maximum=_literal_number(row["maximum"], f"{path}.maximum", 1.25),
        inclusive=_literal_boolean(row["inclusive"], f"{path}.inclusive", True),
        exception_policy=_literal_string(
            row["exception_policy"],
            f"{path}.exception_policy",
            "predeclared_documented_bounded_exception_only",
        ),
        accepted_exceptions=_string_sequence(
            row["accepted_exceptions"], f"{path}.accepted_exceptions", ()
        ),
    )


def _parse_promotion_thresholds(value: object) -> PromotionThresholds:
    path = "$.promotion_thresholds"
    row = _object(
        value,
        path,
        (
            "measurement",
            "primary_performance",
            "secondary_performance",
            "historical_performance",
            "stability",
            "output_fairness",
            "memory",
            "fairness_requirements",
        ),
    )
    return PromotionThresholds(
        measurement=_parse_measurement(row["measurement"]),
        primary_performance=_parse_speedup(
            row["primary_performance"],
            f"{path}.primary_performance",
            "current_reference_w1_median_over_current_native_w1_median",
            1.5,
        ),
        secondary_performance=_parse_speedup(
            row["secondary_performance"],
            f"{path}.secondary_performance",
            "current_reference_w6_median_over_current_native_w6_median",
            1.15,
        ),
        historical_performance=_parse_historical(row["historical_performance"]),
        stability=_parse_stability(row["stability"]),
        output_fairness=_parse_output_fairness(row["output_fairness"]),
        memory=_parse_memory(row["memory"]),
        fairness_requirements=_string_sequence(
            row["fairness_requirements"],
            f"{path}.fairness_requirements",
            _FAIRNESS_REQUIREMENTS,
        ),
    )


def _parse_post_promotion_proof(value: object) -> PostPromotionProof:
    path = "$.post_promotion_proof"
    row = _object(
        value,
        path,
        ("required", "after_atomic_promotion_commit", "expected_promotion_state", "cases", "requirements"),
    )
    state_path = f"{path}.expected_promotion_state"
    raw_state = _object(
        row["expected_promotion_state"],
        state_path,
        (
            "native_safe_default_promotion_accepted",
            "parity_evidence_version_source",
            "certificate_pointer_required",
        ),
    )
    state = ExpectedPromotionState(
        native_safe_default_promotion_accepted=_literal_boolean(
            raw_state["native_safe_default_promotion_accepted"],
            f"{state_path}.native_safe_default_promotion_accepted",
            True,
        ),
        parity_evidence_version_source=_literal_string(
            raw_state["parity_evidence_version_source"],
            f"{state_path}.parity_evidence_version_source",
            "promotion_certificate",
        ),
        certificate_pointer_required=_literal_boolean(
            raw_state["certificate_pointer_required"],
            f"{state_path}.certificate_pointer_required",
            True,
        ),
    )
    return PostPromotionProof(
        required=_literal_boolean(row["required"], f"{path}.required", True),
        after_atomic_promotion_commit=_literal_boolean(
            row["after_atomic_promotion_commit"], f"{path}.after_atomic_promotion_commit", True
        ),
        expected_promotion_state=state,
        cases=_string_sequence(row["cases"], f"{path}.cases", _POST_PROMOTION_CASES),
        requirements=_string_sequence(
            row["requirements"], f"{path}.requirements", _POST_PROMOTION_REQUIREMENTS
        ),
    )


def _validate_invariants(envelope: FlatNativePromotionEnvelope) -> None:
    profile = envelope.supported_profile
    if profile.world_aliases_allowed or "world_alias" not in envelope.disallowed_features:
        raise PromotionEnvelopeValidationError("world alias policy is contradictory")
    if profile.checkpoint.mode("partial_generation").supported:
        raise PromotionEnvelopeValidationError("partial-generation checkpoint cannot be supported")
    if set(profile.output.byte_exact_files) & set(profile.output.semantic_exact_files):
        raise PromotionEnvelopeValidationError("output exactness classes overlap")
    if set(profile.output.byte_exact_files) | set(profile.output.semantic_exact_files) != set(
        profile.output.required_files
    ):
        raise PromotionEnvelopeValidationError("every required output must have one exactness class")
    if envelope.binary_contract.runtime_jit_allowed:
        raise PromotionEnvelopeValidationError("runtime JIT cannot be allowed by the promotion contract")
    if not {"runtime_jit", "hidden_fallback", "mixed_ownership"}.issubset(
        envelope.failure_semantics.native_result_rejections
    ):
        raise PromotionEnvelopeValidationError("native-result rejection semantics are incomplete")
    thresholds = envelope.promotion_thresholds
    if not thresholds.output_fairness.minimum <= 1.0 <= thresholds.output_fairness.maximum:
        raise PromotionEnvelopeValidationError("output fairness interval must contain parity")
    if thresholds.primary_performance.minimum_speedup < thresholds.secondary_performance.minimum_speedup:
        raise PromotionEnvelopeValidationError("primary performance gate cannot be weaker than secondary")
    if thresholds.stability.maximum <= 0.0 or thresholds.memory.maximum < 1.0:
        raise PromotionEnvelopeValidationError("stability and memory thresholds are contradictory")
    if len(envelope.post_promotion_proof.cases) != 12:
        raise PromotionEnvelopeValidationError("post-promotion proof must contain exactly 12 cases")


def validate_promotion_envelope(data: object) -> FlatNativePromotionEnvelope:
    """Validate untrusted contract data and return an immutable typed value."""

    path = "$"
    row = _object(
        data,
        path,
        (
            "schema_version",
            "contract_id",
            "contract_version",
            "supported_profile",
            "disallowed_features",
            "binary_contract",
            "evidence_requirements",
            "failure_semantics",
            "promotion_thresholds",
            "post_promotion_proof",
        ),
    )
    envelope = FlatNativePromotionEnvelope(
        schema_version=_literal_string(
            row["schema_version"], "$.schema_version", CONTRACT_SCHEMA_VERSION
        ),
        contract_id=_literal_string(row["contract_id"], "$.contract_id", CONTRACT_ID),
        contract_version=_literal_integer(
            row["contract_version"], "$.contract_version", CONTRACT_VERSION
        ),
        supported_profile=_parse_supported_profile(row["supported_profile"]),
        disallowed_features=_string_sequence(
            row["disallowed_features"], "$.disallowed_features", _DISALLOWED_FEATURES
        ),
        binary_contract=_parse_binary_contract(row["binary_contract"]),
        evidence_requirements=_parse_evidence_requirements(row["evidence_requirements"]),
        failure_semantics=_parse_failure_semantics(row["failure_semantics"]),
        promotion_thresholds=_parse_promotion_thresholds(row["promotion_thresholds"]),
        post_promotion_proof=_parse_post_promotion_proof(row["post_promotion_proof"]),
    )
    _validate_invariants(envelope)
    return envelope


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PromotionEnvelopeValidationError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> None:
    raise PromotionEnvelopeValidationError(f"non-finite JSON number is not allowed: {value}")


def load_promotion_envelope(path: str | Path | None = None) -> FlatNativePromotionEnvelope:
    """Load and validate the Flat contract, resolving relative paths from the repo."""

    contract_path = promotion_envelope_contract_path(path)
    try:
        raw = contract_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PromotionEnvelopeValidationError(
            f"cannot read promotion envelope {contract_path}: {exc}"
        ) from exc
    try:
        data = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonfinite_json_number,
        )
    except PromotionEnvelopeValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise PromotionEnvelopeValidationError(
            f"malformed promotion envelope JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    return validate_promotion_envelope(data)


__all__ = [
    "CONTRACT_ID",
    "CONTRACT_SCHEMA_VERSION",
    "CONTRACT_VERSION",
    "DEFAULT_CONTRACT_RELATIVE_PATH",
    "BinaryContract",
    "ChemistryProfile",
    "CheckpointMode",
    "CheckpointProfile",
    "DeviceProfile",
    "EvidenceRequirements",
    "ExpectedPromotionState",
    "FailureAtomicity",
    "FailureSemantics",
    "FlatNativePromotionEnvelope",
    "HistoricalPerformance",
    "LifecycleProfile",
    "MeasurementPlan",
    "MemoryThreshold",
    "OutputFairnessThreshold",
    "OutputProfile",
    "PolicyProfile",
    "PostPromotionProof",
    "PromotionEnvelopeValidationError",
    "PromotionThresholds",
    "ReportingProfile",
    "ReportingSchemas",
    "RoutingRule",
    "SpeedupThreshold",
    "StabilityThreshold",
    "StructuralOperatorProfile",
    "SupportedProfile",
    "load_promotion_envelope",
    "promotion_envelope_contract_path",
    "validate_promotion_envelope",
]
