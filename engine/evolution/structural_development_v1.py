from __future__ import annotations

"""ESCAP-6 bounded hereditary developmental expression and dependencies.

The hereditary dependency authority is ``homology_id``.  Runtime expression
and lifetime commitments are resolved to ``module_instance_id``.  The module
owns no RNG, changes no tensor shape, and preserves an absent module-local
record exactly.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from agents.developmental_program_v1 import resolve_stage_id
from agents.structural_identity_v1 import module_is_native_homology
from evolution.structural_live_modules_v1 import LIVE_PROFILE, live_modules_enabled


DEVELOPMENTAL_EXPRESSION_BLOCK_KEY = "developmental_expression"
DEVELOPMENTAL_MODULE_SCHEMA_VERSION = "escap6_module_development_v1"
DEVELOPMENTAL_STAGE_PROFILE_VERSION = "escap6_three_stage_v1"
DEVELOPMENTAL_STAGE_IDS: tuple[str, ...] = ("early", "mature", "late")
DEVELOPMENTAL_FEATURE_PROFILE = "escap6_bounded_module_development_v1"

# The first boundary reuses the existing authoritative cognition maturity
# default.  The second is the next fixed maturity horizon; neither is mutable
# by the ESCAP-6 v1 hereditary operator.
MATURE_START_STEP = 20
LATE_START_STEP = 40

MIN_STAGE_GAIN = 0.0
MAX_STAGE_GAIN = 1.0
MAX_PREREQUISITES_PER_MODULE = 2
MAX_DEPENDENCY_EDGES_PER_AGENT = 8
DEPENDENCY_IDENTITY_AUTHORITY = "homology_id"
CYCLE_POLICY = "reject_deterministic_homology_topological_order_v1"
INACTIVE_STATE_POLICY = "freeze_existing_state_v1"
INACTIVE_OUTPUT_POLICY = "publish_exact_zero_no_stale_output_v1"
LATCH_VERSION = "escap6_module_instance_dependency_readiness_latch_v1"
LATCH_SEMANTICS = "first_ready_opens_for_organism_lifetime_v1"
DEVELOPMENT_OPERATOR_PROFILE = "escap6_developmental_expression_operator_v1"
DEVELOPMENT_OPERATOR_ATTEMPTS = 1
DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT = 2
MIGRATION_RNG_DRAWS = 0
SUPPORTED_LANE = "python_reference_cpu_policy_stride_1"


def structural_development_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "structural_development_enable", False))


def canonical_stage_program(
    *,
    mature_start_step: int = MATURE_START_STEP,
    late_start_step: int = LATE_START_STEP,
) -> dict[str, Any]:
    mature = int(mature_start_step)
    late = int(late_start_step)
    if not 0 < mature < late:
        raise ValueError(
            "ESCAP6_STAGE_BOUNDARIES_INVALID "
            f"mature_start_step={mature};late_start_step={late}"
        )
    return {
        "schema_version": "developmental_program_v1",
        "stages": [
            {"stage_id": "early", "start_step": 0, "end_step": mature - 1},
            {"stage_id": "mature", "start_step": mature, "end_step": late - 1},
            {"stage_id": "late", "start_step": late, "end_step": -1},
        ],
        "module_windows": [],
        "lock_commitments": [],
        "lock_dependencies": [],
    }


def validate_stage_program(
    program: Mapping[str, Any] | None,
    *,
    mature_start_step: int = MATURE_START_STEP,
    late_start_step: int = LATE_START_STEP,
) -> dict[str, Any]:
    """Require the exact bounded three-stage ESCAP-6 programme."""

    if not isinstance(program, Mapping):
        raise ValueError("ESCAP6_STAGE_PROGRAM_MISSING")
    expected = canonical_stage_program(
        mature_start_step=int(mature_start_step),
        late_start_step=int(late_start_step),
    )
    stages = program.get("stages")
    if not isinstance(stages, list) or stages != expected["stages"]:
        raise ValueError(
            "ESCAP6_STAGE_PROFILE_MISMATCH "
            f"expected={expected['stages']!r};observed={stages!r}"
        )
    schema = str(program.get("schema_version", "") or "")
    if schema != expected["schema_version"]:
        raise ValueError(
            "ESCAP6_STAGE_SCHEMA_MISMATCH "
            f"expected={expected['schema_version']};observed={schema}"
        )
    for legacy_key in ("module_windows", "lock_commitments", "lock_dependencies"):
        rows = program.get(legacy_key, [])
        if not isinstance(rows, list) or rows:
            raise ValueError(f"ESCAP6_UNBOUNDED_LEGACY_STAGE_ROWS key={legacy_key}")
    # Exercise the existing authoritative resolver at every boundary rather
    # than implementing a parallel stage interpretation.
    boundary_checks = (
        (0, "early"),
        (int(mature_start_step) - 1, "early"),
        (int(mature_start_step), "mature"),
        (int(late_start_step) - 1, "mature"),
        (int(late_start_step), "late"),
    )
    for age, expected_stage in boundary_checks:
        observed = resolve_stage_id(dict(expected), age_steps=int(age))
        if observed != expected_stage:
            raise ValueError(
                "ESCAP6_EXISTING_STAGE_RESOLVER_MISMATCH "
                f"age={age};expected={expected_stage};observed={observed}"
            )
    return expected


def resolve_structural_stage(age_steps: int, sim_cfg: Any) -> tuple[str, int]:
    mature = int(getattr(sim_cfg, "structural_development_mature_start_step", MATURE_START_STEP))
    late = int(getattr(sim_cfg, "structural_development_late_start_step", LATE_START_STEP))
    program = canonical_stage_program(mature_start_step=mature, late_start_step=late)
    stage_id = resolve_stage_id(program, age_steps=max(0, int(age_steps)))
    return stage_id, DEVELOPMENTAL_STAGE_IDS.index(stage_id)


def validate_structural_development_config(
    sim_cfg: Any,
    *,
    device: Any = "cpu",
    policy_stride: int = 1,
) -> None:
    if not structural_development_enabled(sim_cfg):
        return
    if not live_modules_enabled(sim_cfg):
        raise RuntimeError("ESCAP6_REQUIRES_ESCAP3_LIVE_MODULES")
    if str(getattr(sim_cfg, "structural_live_modules_profile", "none") or "none") != LIVE_PROFILE:
        raise RuntimeError("ESCAP6_ESCAP3_PROFILE_MISMATCH")

    expected = (
        DEVELOPMENTAL_FEATURE_PROFILE,
        DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
        DEVELOPMENTAL_STAGE_PROFILE_VERSION,
        DEVELOPMENTAL_STAGE_IDS,
        MATURE_START_STEP,
        LATE_START_STEP,
        MIN_STAGE_GAIN,
        MAX_STAGE_GAIN,
        MAX_PREREQUISITES_PER_MODULE,
        MAX_DEPENDENCY_EDGES_PER_AGENT,
        DEPENDENCY_IDENTITY_AUTHORITY,
        CYCLE_POLICY,
        INACTIVE_STATE_POLICY,
        INACTIVE_OUTPUT_POLICY,
        LATCH_VERSION,
        LATCH_SEMANTICS,
        DEVELOPMENT_OPERATOR_PROFILE,
        DEVELOPMENT_OPERATOR_ATTEMPTS,
        DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT,
        MIGRATION_RNG_DRAWS,
        SUPPORTED_LANE,
    )
    actual = (
        str(getattr(sim_cfg, "structural_development_profile", "none") or "none"),
        str(getattr(sim_cfg, "structural_development_module_schema_version", DEVELOPMENTAL_MODULE_SCHEMA_VERSION) or DEVELOPMENTAL_MODULE_SCHEMA_VERSION),
        str(getattr(sim_cfg, "structural_development_stage_profile_version", DEVELOPMENTAL_STAGE_PROFILE_VERSION) or DEVELOPMENTAL_STAGE_PROFILE_VERSION),
        tuple(getattr(sim_cfg, "structural_development_stage_ids", DEVELOPMENTAL_STAGE_IDS) or ()),
        int(getattr(sim_cfg, "structural_development_mature_start_step", MATURE_START_STEP)),
        int(getattr(sim_cfg, "structural_development_late_start_step", LATE_START_STEP)),
        float(getattr(sim_cfg, "structural_development_min_stage_gain", MIN_STAGE_GAIN)),
        float(getattr(sim_cfg, "structural_development_max_stage_gain", MAX_STAGE_GAIN)),
        int(getattr(sim_cfg, "structural_development_max_prerequisites_per_module", MAX_PREREQUISITES_PER_MODULE)),
        int(getattr(sim_cfg, "structural_development_max_dependency_edges_per_agent", MAX_DEPENDENCY_EDGES_PER_AGENT)),
        str(getattr(sim_cfg, "structural_development_dependency_identity_authority", DEPENDENCY_IDENTITY_AUTHORITY) or DEPENDENCY_IDENTITY_AUTHORITY),
        str(getattr(sim_cfg, "structural_development_cycle_policy", CYCLE_POLICY) or CYCLE_POLICY),
        str(getattr(sim_cfg, "structural_development_inactive_state_policy", INACTIVE_STATE_POLICY) or INACTIVE_STATE_POLICY),
        str(getattr(sim_cfg, "structural_development_inactive_output_policy", INACTIVE_OUTPUT_POLICY) or INACTIVE_OUTPUT_POLICY),
        str(getattr(sim_cfg, "structural_development_latch_version", LATCH_VERSION) or LATCH_VERSION),
        str(getattr(sim_cfg, "structural_development_latch_semantics", LATCH_SEMANTICS) or LATCH_SEMANTICS),
        str(getattr(sim_cfg, "structural_development_operator_profile", DEVELOPMENT_OPERATOR_PROFILE) or DEVELOPMENT_OPERATOR_PROFILE),
        int(getattr(sim_cfg, "structural_development_operator_attempts", DEVELOPMENT_OPERATOR_ATTEMPTS) or 0),
        int(getattr(sim_cfg, "structural_development_operator_draws_per_attempt", DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT) or 0),
        int(getattr(sim_cfg, "structural_development_migration_rng_draws", MIGRATION_RNG_DRAWS) or 0),
        str(getattr(sim_cfg, "structural_development_supported_lane", SUPPORTED_LANE) or SUPPORTED_LANE),
    )
    if actual != expected:
        raise RuntimeError(f"ESCAP6_CONFIG_CONTRACT_MISMATCH expected={expected!r};actual={actual!r}")
    if str(device) != "cpu":
        raise RuntimeError(f"ESCAP6_CPU_REFERENCE_ONLY device={device}")
    if int(policy_stride) != 1:
        raise RuntimeError(f"ESCAP6_POLICY_STRIDE_UNSUPPORTED stride={int(policy_stride)}")
    prohibited = (
        "native_runtime_enable",
        "runnercore_native_enable",
        "cxxrewrite_native_tick_authoritative_enable",
        "cxxrewrite_resident_generation_runner_enable",
        "cxxwave1g_full_cpp_backbone_enable",
        "cxxbackbone8_resident_flat_patch_generation_island_enable",
        "cuda_enable",
        "gpu_enable",
    )
    enabled_prohibited = [name for name in prohibited if bool(getattr(sim_cfg, name, False))]
    if enabled_prohibited:
        raise RuntimeError(f"ESCAP6_UNSUPPORTED_EXECUTION_LANE flags={enabled_prohibited!r}")


def _finite_gain(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"ESCAP6_GAIN_INVALID field={field};value={value!r}")
    try:
        gain = float(value)
    except Exception as exc:
        raise ValueError(f"ESCAP6_GAIN_INVALID field={field};value={value!r}") from exc
    if not math.isfinite(gain) or gain < MIN_STAGE_GAIN or gain > MAX_STAGE_GAIN:
        raise ValueError(f"ESCAP6_GAIN_OUT_OF_BOUNDS field={field};value={gain!r}")
    return float(gain)


def neutral_developmental_expression_record() -> dict[str, Any]:
    return {
        "schema_version": DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
        "stage_profile_version": DEVELOPMENTAL_STAGE_PROFILE_VERSION,
        "stage_gains": [1.0, 1.0, 1.0],
        "prerequisite_homology_ids": [],
        "latch_dependency_readiness": False,
    }


def normalize_developmental_expression_record(
    raw: Any,
    *,
    module_homology_id: str = "",
) -> dict[str, Any] | None:
    """Canonicalize an explicit record; ``None`` means exactly absent."""

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("ESCAP6_DEVELOPMENT_RECORD_NOT_MAPPING")
    allowed = {
        "schema_version",
        "stage_profile_version",
        "stage_gains",
        "prerequisite_homology_ids",
        "latch_dependency_readiness",
    }
    unknown = sorted(set(str(key) for key in raw) - allowed)
    if unknown:
        raise ValueError(f"ESCAP6_DEVELOPMENT_RECORD_UNKNOWN_FIELDS fields={unknown!r}")
    schema = str(raw.get("schema_version", "") or "")
    profile = str(raw.get("stage_profile_version", "") or "")
    if schema != DEVELOPMENTAL_MODULE_SCHEMA_VERSION:
        raise ValueError(f"ESCAP6_DEVELOPMENT_SCHEMA_MISMATCH observed={schema}")
    if profile != DEVELOPMENTAL_STAGE_PROFILE_VERSION:
        raise ValueError(f"ESCAP6_DEVELOPMENT_STAGE_PROFILE_MISMATCH observed={profile}")
    gains_raw = raw.get("stage_gains")
    if not isinstance(gains_raw, (list, tuple)) or len(gains_raw) != len(DEVELOPMENTAL_STAGE_IDS):
        raise ValueError("ESCAP6_STAGE_GAIN_VECTOR_INVALID")
    gains = [_finite_gain(value, field=f"stage_gains[{index}]") for index, value in enumerate(gains_raw)]
    prereq_raw = raw.get("prerequisite_homology_ids")
    if not isinstance(prereq_raw, (list, tuple)):
        raise ValueError("ESCAP6_PREREQUISITES_NOT_LIST")
    prereqs = [str(value or "").strip() for value in prereq_raw]
    if any(not value for value in prereqs):
        raise ValueError("ESCAP6_PREREQUISITE_EMPTY")
    if len(prereqs) != len(set(prereqs)):
        raise ValueError("ESCAP6_PREREQUISITE_DUPLICATE")
    if len(prereqs) > MAX_PREREQUISITES_PER_MODULE:
        raise ValueError("ESCAP6_PREREQUISITE_CAP_EXCEEDED")
    prereqs = sorted(prereqs)
    owner = str(module_homology_id or "").strip()
    if owner and owner in prereqs:
        raise ValueError(f"ESCAP6_PREREQUISITE_SELF_EDGE homology_id={owner}")
    latch = raw.get("latch_dependency_readiness")
    if not isinstance(latch, bool):
        raise ValueError("ESCAP6_LATCH_FIELD_NOT_BOOL")
    return {
        "schema_version": DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
        "stage_profile_version": DEVELOPMENTAL_STAGE_PROFILE_VERSION,
        "stage_gains": gains,
        "prerequisite_homology_ids": prereqs,
        "latch_dependency_readiness": bool(latch),
    }


def effective_developmental_expression_record(raw: Any) -> dict[str, Any]:
    normalized = normalize_developmental_expression_record(raw)
    return neutral_developmental_expression_record() if normalized is None else normalized


def _module_sort_key(module: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(module.get("homology_id", "")),
        str(module.get("module_instance_id", "")),
        str(module.get("module_id", "")),
    )


def agent_has_developmental_records(modules: Iterable[Mapping[str, Any]]) -> bool:
    for module in modules:
        params = module.get("params", {})
        if isinstance(params, Mapping) and DEVELOPMENTAL_EXPRESSION_BLOCK_KEY in params:
            return True
    return False


def normalize_agent_developmental_modules(modules: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Canonicalize only explicit records, preserving absent bytes."""

    out: list[dict[str, Any]] = []
    saw_explicit_record = False
    for module in modules:
        record = deepcopy(dict(module))
        params = record.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("ESCAP6_MODULE_PARAMS_NOT_MAPPING")
        if DEVELOPMENTAL_EXPRESSION_BLOCK_KEY in params:
            saw_explicit_record = True
            if str(record.get("module_type", "")) != "generic":
                raise ValueError(
                    "ESCAP6_DEVELOPMENT_RECORD_REQUIRES_GENERIC_MODULE "
                    f"module_id={record.get('module_id', '')}"
                )
            if not module_is_native_homology(record):
                raise ValueError(
                    "ESCAP6_DEVELOPMENT_RECORD_REQUIRES_NATIVE_HOMOLOGY "
                    f"module_id={record.get('module_id', '')}"
                )
            normalized = normalize_developmental_expression_record(
                params.get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY),
                module_homology_id=str(record.get("homology_id", "")),
            )
            if normalized is None:
                raise ValueError("ESCAP6_EXPLICIT_DEVELOPMENT_RECORD_NULL")
            next_params = deepcopy(params)
            next_params[DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = normalized
            record["params"] = next_params
        out.append(record)
    # An all-absent payload is the legacy byte-preserving path.  Strict graph
    # validation is required as soon as ESCAP-6 representation is allocated;
    # enabled runtime also invokes the public validator explicitly.
    if saw_explicit_record:
        validate_agent_developmental_graph(out)
    return out


def validate_agent_developmental_graph(
    modules: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate endpoints/caps/DAG and return homology-ordered topology."""

    records = [dict(module) for module in modules]
    homology_to_module: dict[str, dict[str, Any]] = {}
    instance_ids: set[str] = set()
    for module in sorted(records, key=_module_sort_key):
        homology_id = str(module.get("homology_id", "") or "")
        instance_id = str(module.get("module_instance_id", "") or "")
        if homology_id:
            if homology_id in homology_to_module:
                raise ValueError(f"ESCAP6_AMBIGUOUS_HOMOLOGY homology_id={homology_id}")
            homology_to_module[homology_id] = module
        if instance_id:
            if instance_id in instance_ids:
                raise ValueError(f"ESCAP6_DUPLICATE_MODULE_INSTANCE instance_id={instance_id}")
            instance_ids.add(instance_id)

    prerequisites: dict[str, tuple[str, ...]] = {}
    total_edges = 0
    owners: set[str] = set()
    for module in sorted(records, key=_module_sort_key):
        params = module.get("params", {})
        if not isinstance(params, Mapping):
            raise ValueError("ESCAP6_MODULE_PARAMS_NOT_MAPPING")
        if DEVELOPMENTAL_EXPRESSION_BLOCK_KEY not in params:
            continue
        if not module_is_native_homology(module):
            raise ValueError("ESCAP6_UNRESOLVED_LEGACY_DEVELOPMENT_OWNER")
        owner = str(module.get("homology_id", ""))
        normalized = normalize_developmental_expression_record(
            params.get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY),
            module_homology_id=owner,
        )
        assert normalized is not None
        owners.add(owner)
        deps = tuple(str(value) for value in normalized["prerequisite_homology_ids"])
        for target in deps:
            target_module = homology_to_module.get(target)
            if target_module is None:
                raise ValueError(
                    f"ESCAP6_MISSING_PREREQUISITE_HOMOLOGY owner={owner};target={target}"
                )
            if not module_is_native_homology(target_module):
                raise ValueError(
                    f"ESCAP6_UNRESOLVED_LEGACY_PREREQUISITE owner={owner};target={target}"
                )
            if str(target_module.get("module_type", "")) != "generic":
                raise ValueError(
                    f"ESCAP6_PREREQUISITE_NOT_GENERIC owner={owner};target={target}"
                )
        prerequisites[owner] = deps
        total_edges += len(deps)
    if total_edges > MAX_DEPENDENCY_EDGES_PER_AGENT:
        raise ValueError(
            "ESCAP6_DEPENDENCY_EDGE_CAP_EXCEEDED "
            f"observed={total_edges};max={MAX_DEPENDENCY_EDGES_PER_AGENT}"
        )

    # Include every native module in the topological result so a dependency
    # target without its own explicit record still resolves before consumers.
    nodes = {
        homology_id
        for homology_id, module in homology_to_module.items()
        if str(module.get("module_type", "")) == "generic"
    }
    indegree = {node: 0 for node in nodes}
    consumers: dict[str, list[str]] = {node: [] for node in nodes}
    for owner, deps in prerequisites.items():
        for target in deps:
            indegree[owner] += 1
            consumers[target].append(owner)
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    ordered_ids: list[str] = []
    while ready:
        node = ready.pop(0)
        ordered_ids.append(node)
        for consumer in sorted(consumers.get(node, [])):
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                ready.append(consumer)
                ready.sort()
    if len(ordered_ids) != len(nodes):
        cyclic = sorted(node for node, degree in indegree.items() if degree > 0)
        raise ValueError(f"ESCAP6_DEPENDENCY_CYCLE homology_ids={cyclic!r}")
    return [homology_to_module[homology_id] for homology_id in ordered_ids]


@dataclass(frozen=True)
class DevelopmentalResolution:
    stage_id: str
    stage_index: int
    rows_by_module_instance_id: dict[str, dict[str, Any]]
    latch_state_by_module_instance_id: dict[str, bool]
    topological_homology_ids: tuple[str, ...]


def resolve_developmental_expression(
    modules: Iterable[Mapping[str, Any]],
    *,
    stage_id: str,
    latch_state_by_module_instance_id: Mapping[str, Any] | None = None,
) -> DevelopmentalResolution:
    """Resolve requested/readiness/effective gains without execution RNG."""

    sid = str(stage_id or "")
    if sid not in DEVELOPMENTAL_STAGE_IDS:
        raise ValueError(f"ESCAP6_STAGE_ID_UNSUPPORTED stage_id={sid}")
    stage_index = DEVELOPMENTAL_STAGE_IDS.index(sid)
    normalized_modules = normalize_agent_developmental_modules(modules)
    ordered = validate_agent_developmental_graph(normalized_modules)
    input_latches = {
        str(key): bool(value)
        for key, value in dict(latch_state_by_module_instance_id or {}).items()
        if bool(value)
    }
    valid_instances = {str(module.get("module_instance_id", "")) for module in ordered}
    next_latches = {
        key: True for key in sorted(input_latches) if key in valid_instances
    }
    effective_by_homology: dict[str, float] = {}
    rows: dict[str, dict[str, Any]] = {}
    for module in ordered:
        homology_id = str(module.get("homology_id", ""))
        instance_id = str(module.get("module_instance_id", ""))
        params = module.get("params", {}) if isinstance(module.get("params", {}), Mapping) else {}
        record = effective_developmental_expression_record(
            params.get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY)
            if DEVELOPMENTAL_EXPRESSION_BLOCK_KEY in params
            else None
        )
        requested = float(record["stage_gains"][stage_index])
        deps = tuple(str(value) for value in record["prerequisite_homology_ids"])
        currently_ready = all(effective_by_homology.get(target, 0.0) > 0.0 for target in deps)
        latch_enabled = bool(record["latch_dependency_readiness"])
        latched_before = bool(next_latches.get(instance_id, False))
        gate_ready = bool(currently_ready or (latch_enabled and latched_before))
        if latch_enabled and currently_ready:
            next_latches[instance_id] = True
        effective = requested if gate_ready else 0.0
        effective_by_homology[homology_id] = float(effective)
        rows[instance_id] = {
            "homology_id": homology_id,
            "module_instance_id": instance_id,
            "stage_id": sid,
            "stage_index": int(stage_index),
            "requested_stage_gain": requested,
            "prerequisite_homology_ids": list(deps),
            "dependency_ready_now": bool(currently_ready),
            "latch_enabled": latch_enabled,
            "latched_before": latched_before,
            "latched_after": bool(next_latches.get(instance_id, False)),
            "effective_gain": float(effective),
            "active": bool(effective > 0.0),
        }
    return DevelopmentalResolution(
        stage_id=sid,
        stage_index=int(stage_index),
        rows_by_module_instance_id=rows,
        latch_state_by_module_instance_id=next_latches,
        topological_homology_ids=tuple(str(module.get("homology_id", "")) for module in ordered),
    )
