"""Strict loader and deterministic compiler for ``eiel_campaign_v1``."""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from dataclasses import replace
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from engine.qualification_plans import (
    MATRIX_SCHEMA_VERSION,
    QualificationPlanError,
    compile_matrix,
    load_document,
)

from .acs_checkpoint_staging import ACS_CHECKPOINT_SLOTS_DEFAULT_V1
from .canonical import canonical_bytes, canonical_sha256
from .contracts import (
    CampaignPlanV1,
    CampaignRequirementsV1,
    ComparisonBranchV1,
    LogicalCellV1,
    PhysicalBatchV1,
    QualificationKeyV1,
    RuntimeProfileV1,
    Scalar,
)
from .artifact_lifecycle import (
    ArtifactLifecycleError,
    resolve_run_artifact_lifecycle_v1,
)
from .registry import ProfileRegistryV1, default_profile_registry
from .tick_native import (
    AbsoluteTickScheduleV2,
    TICK_NATIVE_EXECUTION_ROUTE_V1,
    TickNativeError,
    TickNativeExecutionRequestV1,
    TickScheduleV1,
)
from .tick_native_control import (
    TickNativeControlError,
    TickNativeControlV1,
    assert_tick_native_plan_payload_v1,
    physical_b136_event_window_capacity_v1,
)
from .wrc_variability import (
    INVARIANT_NAME as WRC_VARIABILITY_INVARIANT_NAME,
    WRCVariabilityEnvelopeError,
    build_wrc_parameter_variability_envelope_v1,
)
from .resources import (
    CHECKPOINT_BYTES_PER_TRAJECTORY_BOUNDARY,
    RESIDENT_COMPACT_CHECKPOINT_BYTES_PER_TRAJECTORY_BOUNDARY,
    planning_acs_native_fixed_mlp_b136_checkpoint_upper_bound_v1,
    planning_disk_upper_bound_absolute_ticks_v1,
    planning_disk_upper_bound_v1,
)

SCHEMA_VERSION = "eiel_campaign_v1"
PLAN_SCHEMA_VERSION = "eiel_campaign_plan_v1"
QUALIFICATION_KEY_SCHEMA_VERSION = "eiel_runtime_qualification_key_v1"
_ID_RE = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*\Z")
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
SIGNED_PHYSICAL_CHECKPOINT_ACCOUNTING_SCHEMA_V1 = (
    "eiel_signed_physical_checkpoint_planning_accounting_v1"
)
PlanningResourceAccountingV1 = Callable[..., Mapping[str, Any] | None]
_SEMANTICALLY_NEUTRAL_CELL_ID_PARAMETERS_V1 = frozenset(
    {
        "hapb_frag.identity_reuse",
        "hapb_frag.force_physical_refill",
    }
)
_TOP_KEYS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "description",
        "no_claims",
        "run_contract",
        "world",
        "policy",
        "selection",
        "parameters",
        "matrix",
        "execution",
        "checkpoints",
        "comparisons",
        "artifacts",
        "analysis",
        "output",
        "artifact_lifecycle",
        "_legacy_lifecycle_derived",
    }
)


class CampaignError(RuntimeError):
    """A campaign specification cannot be compiled safely."""


def _object(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CampaignError(f"{path} must be an object with string keys")
    return dict(value)


def _exact(value: Mapping[str, Any], allowed: frozenset[str], *, path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise CampaignError(f"{path} contains unsupported fields: {unknown}")


def _identifier(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise CampaignError(f"{path} must be a lowercase stable identifier")
    return value


def _text(value: Any, *, path: str, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise CampaignError(
            f"{path} must be {'a nonempty ' if nonempty else 'a '}string"
        )
    return value


def _integer(value: Any, *, path: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CampaignError(f"{path} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, *, path: str) -> bool:
    if type(value) is not bool:
        raise CampaignError(f"{path} must be boolean")
    return value


def _scalar(value: Any, *, path: str) -> Scalar:
    if not isinstance(value, (bool, int, float, str)) or (
        isinstance(value, float) and not math.isfinite(value)
    ):
        raise CampaignError(f"{path} must be a finite scalar")
    return value


def _string_list(value: Any, *, path: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CampaignError(f"{path} must be an array of strings")
    if len(value) != len(set(value)):
        raise CampaignError(f"{path} must not contain duplicates")
    return list(value)


def _canonical_artifacts(value: Any) -> dict[str, dict[str, Scalar]]:
    artifacts = _object(value, path="artifacts")
    if not artifacts:
        raise CampaignError("artifacts must not be empty")
    result: dict[str, dict[str, Scalar]] = {}
    for name, raw in sorted(artifacts.items()):
        _identifier(name, path=f"artifacts.{name}")
        row = _object(raw, path=f"artifacts.{name}")
        _exact(
            row,
            frozenset({"path", "sha256", "kind", "abi", "authority"}),
            path=f"artifacts.{name}",
        )
        path_value = _text(
            row.get("path"), path=f"artifacts.{name}.path", nonempty=True
        )
        digest = row.get("sha256")
        if not isinstance(digest, str) or _SHA_RE.fullmatch(digest) is None:
            raise CampaignError(f"artifacts.{name}.sha256 must be lowercase SHA-256")
        result[name] = {
            "path": path_value,
            "sha256": digest,
            "kind": _text(
                row.get("kind", name), path=f"artifacts.{name}.kind", nonempty=True
            ),
            "abi": _text(
                row.get("abi", "unspecified"),
                path=f"artifacts.{name}.abi",
                nonempty=True,
            ),
        }
        if "authority" in row:
            result[name]["authority"] = _text(
                row.get("authority"),
                path=f"artifacts.{name}.authority",
                nonempty=True,
            )
    return result


def _analysis(value: Any) -> dict[str, Any]:
    analysis = _object(value, path="analysis")
    _exact(
        analysis,
        frozenset({"required_metrics", "optional_metrics", "rules", "extension"}),
        path="analysis",
    )
    required = sorted(
        _string_list(
            analysis.get("required_metrics", []), path="analysis.required_metrics"
        )
    )
    optional = sorted(
        _string_list(
            analysis.get("optional_metrics", []), path="analysis.optional_metrics"
        )
    )
    if set(required) & set(optional):
        raise CampaignError("analysis required_metrics and optional_metrics overlap")
    rules = analysis.get("rules", [])
    if not isinstance(rules, list):
        raise CampaignError("analysis.rules must be an array")
    canonical_rules: list[dict[str, Any]] = []
    allowed_rule = frozenset(
        {
            "id",
            "metric",
            "aggregate",
            "group_by",
            "pair_by",
            "operator",
            "threshold",
            "minimum_coverage",
        }
    )
    for index, raw in enumerate(rules):
        rule = _object(raw, path=f"analysis.rules[{index}]")
        _exact(rule, allowed_rule, path=f"analysis.rules[{index}]")
        rule_id = _identifier(rule.get("id"), path=f"analysis.rules[{index}].id")
        metric = _text(
            rule.get("metric"), path=f"analysis.rules[{index}].metric", nonempty=True
        )
        aggregate = rule.get("aggregate", "mean")
        if aggregate not in {"sum", "mean", "count", "paired_delta"}:
            raise CampaignError(f"analysis.rules[{index}].aggregate is unsupported")
        operator = rule.get("operator", ">=")
        if operator not in {">", ">=", "<", "<=", "==", "!="}:
            raise CampaignError(f"analysis.rules[{index}].operator is unsupported")
        threshold = _scalar(
            rule.get("threshold"), path=f"analysis.rules[{index}].threshold"
        )
        coverage = rule.get("minimum_coverage", 1.0)
        if (
            isinstance(coverage, bool)
            or not isinstance(coverage, (int, float))
            or not 0 <= coverage <= 1
        ):
            raise CampaignError(
                f"analysis.rules[{index}].minimum_coverage must be in [0,1]"
            )
        canonical_rules.append(
            {
                "id": rule_id,
                "metric": metric,
                "aggregate": aggregate,
                "group_by": sorted(
                    _string_list(
                        rule.get("group_by", []),
                        path=f"analysis.rules[{index}].group_by",
                    )
                ),
                "pair_by": sorted(
                    _string_list(
                        rule.get("pair_by", []), path=f"analysis.rules[{index}].pair_by"
                    )
                ),
                "operator": operator,
                "threshold": threshold,
                "minimum_coverage": float(coverage),
            }
        )
    if len({rule["id"] for rule in canonical_rules}) != len(canonical_rules):
        raise CampaignError("analysis.rules contains duplicate ids")
    result: dict[str, Any] = {
        "required_metrics": required,
        "optional_metrics": optional,
        "rules": canonical_rules,
    }
    raw_extension = analysis.get("extension")
    if raw_extension is not None:
        extension = _object(raw_extension, path="analysis.extension")
        _exact(
            extension,
            frozenset({"id", "contract", "contract_sha256"}),
            path="analysis.extension",
        )
        extension_id = _identifier(extension.get("id"), path="analysis.extension.id")
        contract = _object(
            extension.get("contract"), path="analysis.extension.contract"
        )
        digest = extension.get("contract_sha256")
        if not isinstance(digest, str) or _SHA_RE.fullmatch(digest) is None:
            raise CampaignError(
                "analysis.extension.contract_sha256 must be lowercase SHA-256"
            )
        try:
            canonical_contract = json.loads(canonical_bytes(contract))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CampaignError(
                "analysis.extension.contract must be canonical JSON"
            ) from exc
        if canonical_sha256(canonical_contract) != digest:
            raise CampaignError("analysis.extension contract SHA-256 mismatch")
        result["extension"] = {
            "id": extension_id,
            "contract": canonical_contract,
            "contract_sha256": digest,
        }
    return result


def validate_campaign_spec(value: Any) -> dict[str, Any]:
    spec = _object(value, path="campaign")
    _exact(spec, _TOP_KEYS, path="campaign")
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise CampaignError(f"schema_version must be {SCHEMA_VERSION!r}")
    campaign_id = _identifier(spec.get("campaign_id"), path="campaign_id")
    description = _text(spec.get("description", ""), path="description")
    no_claims = _text(spec.get("no_claims"), path="no_claims", nonempty=True)

    run_contract_raw = spec.get("run_contract")
    run_contract: dict[str, str] | None = None
    if run_contract_raw is not None:
        requested = _object(run_contract_raw, path="run_contract")
        _exact(requested, frozenset({"mode", "profile_id"}), path="run_contract")
        mode = _identifier(requested.get("mode"), path="run_contract.mode")
        if mode not in {"optional", "strict"}:
            raise CampaignError("run_contract.mode must be optional or strict")
        run_contract = {
            "mode": mode,
            "profile_id": _identifier(
                requested.get("profile_id"), path="run_contract.profile_id"
            ),
        }

    world = _object(spec.get("world"), path="world")
    _exact(world, frozenset({"base", "overlays"}), path="world")
    world_base = _identifier(world.get("base"), path="world.base")
    overlays = sorted(_string_list(world.get("overlays", []), path="world.overlays"))
    for index, overlay in enumerate(overlays):
        _identifier(overlay, path=f"world.overlays[{index}]")

    policy = _object(spec.get("policy"), path="policy")
    _exact(policy, frozenset({"family"}), path="policy")
    policy_family = _identifier(policy.get("family"), path="policy.family")
    selection = _object(spec.get("selection", {"mode": "standard"}), path="selection")
    _exact(selection, frozenset({"mode"}), path="selection")
    selection_mode = _identifier(
        selection.get("mode", "standard"), path="selection.mode"
    )

    parameters = _object(spec.get("parameters", {}), path="parameters")
    canonical_parameters = {
        name: _scalar(item, path=f"parameters.{name}")
        for name, item in sorted(parameters.items())
    }

    matrix = _object(spec.get("matrix"), path="matrix")
    _exact(
        matrix,
        frozenset(
            {
                "dimensions",
                "parameter_bindings",
                "seed_bindings",
                "include",
                "exclude",
                "expected_cells",
                "max_cells",
            }
        ),
        path="matrix",
    )
    dimensions = matrix.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise CampaignError("matrix.dimensions must be a nonempty array")
    parameter_bindings = _object(
        matrix.get("parameter_bindings", {}), path="matrix.parameter_bindings"
    )
    seed_bindings = _object(
        matrix.get("seed_bindings", {}), path="matrix.seed_bindings"
    )
    for dimension, parameter in parameter_bindings.items():
        _identifier(dimension, path=f"matrix.parameter_bindings.{dimension}")
        if not isinstance(parameter, str) or not parameter:
            raise CampaignError(
                f"matrix.parameter_bindings.{dimension} must name a typed parameter"
            )
    allowed_seed_owners = {"ecology", "task", "policy", "founder"}
    for dimension, owner in seed_bindings.items():
        _identifier(dimension, path=f"matrix.seed_bindings.{dimension}")
        if owner not in allowed_seed_owners:
            raise CampaignError(
                f"matrix.seed_bindings.{dimension} has unsupported owner {owner!r}"
            )
    duplicate_seed_owners = sorted(
        owner
        for owner in set(seed_bindings.values())
        if list(seed_bindings.values()).count(owner) > 1
    )
    if duplicate_seed_owners:
        raise CampaignError(
            f"matrix.seed_bindings has ambiguous duplicate owners: {duplicate_seed_owners}"
        )
    overlap = set(parameter_bindings) & set(seed_bindings)
    if overlap:
        raise CampaignError(
            f"matrix dimensions have ambiguous ownership: {sorted(overlap)}"
        )

    execution = _object(spec.get("execution"), path="execution")
    route_identity = execution.get("execution_route_identity")
    tick_native = route_identity == TICK_NATIVE_EXECUTION_ROUTE_V1
    tick_native_execution_keys = frozenset(
        {
            "backend",
            "execution_route_identity",
            "total_ticks",
            "start_next_tick",
            "checkpoint_ticks",
            "checkpoint_every_ticks",
            "science_sample_interval_ticks",
            "report_every_ticks",
            "runtime_partition_ticks",
            "resident_owner_span_ticks",
            "native_dispatch_window_ticks",
            "reducer_every_ticks",
            "archive_every_ticks",
            "event_window_capacity",
            "source_window_ticks",
            "tick_native_control",
            "deliberate_restore_tick",
            "deliberate_restore_ticks",
            "batch",
            "budgets",
            "cancellation",
            "technical_gate",
            "gpu_resource_policy",
        }
    )
    legacy_execution_keys = frozenset(
        {
            "backend",
            "execution_route_identity",
            "generations",
            "ticks_per_generation",
            "batch",
            "budgets",
            "cancellation",
            "technical_gate",
            "physical_window_generations",
        }
    )
    checkpoint_ticks: tuple[int, ...] = ()
    tick_request: TickNativeExecutionRequestV1 | None = None
    tick_control: TickNativeControlV1 | None = None
    if tick_native:
        _exact(execution, tick_native_execution_keys, path="execution")
        try:
            tick_request = TickNativeExecutionRequestV1.from_execution(dict(execution))
        except TickNativeError as exc:
            raise CampaignError(
                f"execution tick-native horizon is invalid: {exc}"
            ) from exc
        try:
            tick_control = TickNativeControlV1.from_execution(execution)
        except TickNativeControlError as exc:
            raise CampaignError(
                f"execution tick-native control is invalid: {exc}"
            ) from exc
        # Keep generation-only validation branches inert.  The active request
        # has no private G/T projection.
        generations = 0
        ticks = 0
        checkpoint_ticks = tick_control.resolved_checkpoint_ticks
        backend = _identifier(execution.get("backend"), path="execution.backend")
        try:
            AbsoluteTickScheduleV2(
                total_ticks=tick_request.total_ticks,
                science_sample_interval_ticks=tick_control.science_sample_interval_ticks,
            )
        except TickNativeError as exc:
            raise CampaignError(
                f"execution tick-native horizon is invalid: {exc}"
            ) from exc
    else:
        if route_identity not in (None, "legacy_generation_route_v1"):
            raise CampaignError("execution.execution_route_identity is unsupported")
        _exact(execution, legacy_execution_keys, path="execution")
        backend = _identifier(execution.get("backend"), path="execution.backend")
        generations = _integer(
            execution.get("generations"), path="execution.generations"
        )
        ticks = _integer(
            execution.get("ticks_per_generation"), path="execution.ticks_per_generation"
        )
        try:
            TickScheduleV1.from_legacy(generations, ticks)
        except TickNativeError as exc:
            raise CampaignError(f"execution G/T horizon is invalid: {exc}") from exc
    gpu_resource_policy_selector = execution.get("gpu_resource_policy")
    if gpu_resource_policy_selector is not None:
        if not tick_native:
            raise CampaignError(
                "execution.gpu_resource_policy requires tick-native execution"
            )
        gpu_resource_policy_selector = _identifier(
            gpu_resource_policy_selector,
            path="execution.gpu_resource_policy",
        )
        try:
            from .cuda_full_evolution_v2_shared_route import (
                resolve_shared_w32_gpu_policy_v1,
            )

            resolve_shared_w32_gpu_policy_v1(gpu_resource_policy_selector)
        except ValueError as exc:
            raise CampaignError(str(exc)) from exc
    physical_window_generations = execution.get("physical_window_generations")
    if physical_window_generations is not None:
        physical_window_generations = _integer(
            physical_window_generations,
            path="execution.physical_window_generations",
        )
        if physical_window_generations not in {1, 2}:
            raise CampaignError("execution.physical_window_generations must be 1 or 2")
    max_internal_lifecycle_segments_per_dispatch = execution.get(
        "max_internal_lifecycle_segments_per_dispatch"
    )
    if max_internal_lifecycle_segments_per_dispatch is not None:
        if not tick_native:
            raise CampaignError(
                "execution.max_internal_lifecycle_segments_per_dispatch is "
                "valid only for tick-native campaigns"
            )
        max_internal_lifecycle_segments_per_dispatch = _integer(
            max_internal_lifecycle_segments_per_dispatch,
            path="execution.max_internal_lifecycle_segments_per_dispatch",
        )
        if max_internal_lifecycle_segments_per_dispatch > 128:
            raise CampaignError(
                "execution.max_internal_lifecycle_segments_per_dispatch exceeds 128"
            )
    if tick_native:
        # A generation-indexed checkpoint list is never a valid native input.
        # It is retained in the schema solely for legacy campaigns.
        raw_generation_milestones = (spec.get("checkpoints") or {}).get(
            "milestones", []
        )
        if raw_generation_milestones:
            raise CampaignError(
                "tick-native campaigns must use execution.checkpoint_ticks, not "
                "checkpoints.milestones"
            )
    batch = _object(execution.get("batch"), path="execution.batch")
    _exact(batch, frozenset({"mode", "cells", "maximum"}), path="execution.batch")
    mode = batch.get("mode", "automatic")
    if mode not in {"automatic", "explicit", "bounded"}:
        raise CampaignError("execution.batch.mode is unsupported")
    cells = batch.get("cells")
    maximum = batch.get("maximum")
    if mode == "explicit":
        cells = _integer(cells, path="execution.batch.cells")
    elif cells is not None:
        raise CampaignError("execution.batch.cells is valid only in explicit mode")
    if mode in {"automatic", "bounded"}:
        maximum = _integer(maximum, path="execution.batch.maximum")
    elif maximum is not None:
        raise CampaignError("execution.batch.maximum is not valid in explicit mode")
    budgets = _object(execution.get("budgets"), path="execution.budgets")
    _exact(
        budgets,
        frozenset(
            {
                "vram_bytes",
                "host_bytes",
                "minimum_free_gpu_bytes",
                "maximum_gpu_memory_slope_bytes_per_window",
                "maximum_host_memory_slope_bytes_per_window",
                "maximum_event_ring_occupancy_slope_per_window",
                "disk_bytes",
                "wall_seconds",
            }
        ),
        path="execution.budgets",
    )
    canonical_budgets = {
        name: _integer(budgets.get(name), path=f"execution.budgets.{name}")
        for name in (
            "vram_bytes",
            "host_bytes",
            "minimum_free_gpu_bytes",
            "maximum_gpu_memory_slope_bytes_per_window",
            "maximum_host_memory_slope_bytes_per_window",
            "maximum_event_ring_occupancy_slope_per_window",
            "disk_bytes",
            "wall_seconds",
        )
        if name in budgets
    }
    technical_gate = _object(
        execution.get("technical_gate", {}), path="execution.technical_gate"
    )
    _exact(
        technical_gate,
        frozenset(
            {
                "selectors",
                "partition",
                "generations",
                "checkpoint_milestones",
                "total_ticks",
                "checkpoint_ticks",
                "profile",
            }
        ),
        path="execution.technical_gate",
    )
    technical_gate_profile = technical_gate.get("profile")
    if technical_gate_profile is not None and (
        not isinstance(technical_gate_profile, str)
        or not technical_gate_profile
        or technical_gate_profile.strip() != technical_gate_profile
    ):
        raise CampaignError("execution.technical_gate.profile must be a stable string")
    technical_gate_selectors_raw = technical_gate.get("selectors", [])
    if not isinstance(technical_gate_selectors_raw, list):
        raise CampaignError("execution.technical_gate.selectors must be an array")
    technical_gate_partition = technical_gate.get("partition", True)
    if type(technical_gate_partition) is not bool:
        raise CampaignError("execution.technical_gate.partition must be boolean")
    technical_gate_generations_raw = technical_gate.get("generations")
    technical_gate_milestones_raw = technical_gate.get("checkpoint_milestones")
    technical_gate_total_ticks_raw = technical_gate.get("total_ticks")
    technical_gate_checkpoint_ticks_raw = technical_gate.get("checkpoint_ticks")
    if tick_native:
        if (
            technical_gate_generations_raw is not None
            or technical_gate_milestones_raw is not None
        ):
            raise CampaignError(
                "tick-native technical gates must use absolute ticks, not "
                "generation fields"
            )
        technical_gate_total_ticks = (
            tick_request.total_ticks
            if technical_gate_total_ticks_raw is None
            else _integer(
                technical_gate_total_ticks_raw,
                path="execution.technical_gate.total_ticks",
            )
        )
        if technical_gate_total_ticks > tick_request.total_ticks:
            raise CampaignError(
                "execution.technical_gate.total_ticks exceeds execution.total_ticks"
            )
        if technical_gate_checkpoint_ticks_raw is None:
            technical_gate_checkpoint_ticks = []
        else:
            if not isinstance(technical_gate_checkpoint_ticks_raw, list):
                raise CampaignError(
                    "execution.technical_gate.checkpoint_ticks must be an array"
                )
            technical_gate_checkpoint_ticks = sorted(
                {
                    _integer(
                        item,
                        path=("execution.technical_gate.checkpoint_ticks" f"[{index}]"),
                        minimum=0,
                    )
                    for index, item in enumerate(technical_gate_checkpoint_ticks_raw)
                }
            )
            if technical_gate_checkpoint_ticks != technical_gate_checkpoint_ticks_raw:
                raise CampaignError(
                    "execution.technical_gate.checkpoint_ticks must be sorted unique"
                )
            if any(
                item > technical_gate_total_ticks
                for item in technical_gate_checkpoint_ticks
            ):
                raise CampaignError(
                    "technical-gate checkpoint tick exceeds technical-gate total_ticks"
                )
        technical_gate_generations = 0
        technical_gate_milestones = []
    else:
        if (
            technical_gate_total_ticks_raw is not None
            or technical_gate_checkpoint_ticks_raw is not None
        ):
            raise CampaignError("legacy technical gates must use generation fields")
        technical_gate_generations = (
            min(5, generations)
            if technical_gate_generations_raw is None
            else _integer(
                technical_gate_generations_raw,
                path="execution.technical_gate.generations",
            )
        )
        if technical_gate_generations > generations:
            raise CampaignError(
                "execution.technical_gate.generations exceeds execution.generations"
            )
        if technical_gate_milestones_raw is None:
            technical_gate_milestones = [
                value for value in (0, 5) if value <= generations
            ]
        else:
            if not isinstance(technical_gate_milestones_raw, list):
                raise CampaignError(
                    "execution.technical_gate.checkpoint_milestones must be an array"
                )
            technical_gate_milestones = sorted(
                {
                    _integer(
                        item,
                        path=(
                            "execution.technical_gate.checkpoint_milestones"
                            f"[{index}]"
                        ),
                        minimum=0,
                    )
                    for index, item in enumerate(technical_gate_milestones_raw)
                }
            )
            if technical_gate_milestones != technical_gate_milestones_raw:
                raise CampaignError(
                    "execution.technical_gate.checkpoint_milestones "
                    "must be sorted unique"
                )
            if any(
                item > technical_gate_generations for item in technical_gate_milestones
            ):
                raise CampaignError(
                    "technical-gate checkpoint milestone exceeds "
                    "execution.technical_gate.generations"
                )

    checkpoints = _object(spec.get("checkpoints"), path="checkpoints")
    _exact(
        checkpoints,
        frozenset(
            {
                "milestones",
                "digest_every_generation",
                "storage_mode",
                "acs_slots",
                "acs_capture_instrumentation",
                "native_streamed_checkpoint",
                "world_sharded_checkpoint",
            }
        ),
        path="checkpoints",
    )
    milestones = checkpoints.get("milestones", [])
    if not isinstance(milestones, list):
        raise CampaignError("checkpoints.milestones must be an array")
    canonical_milestones = sorted(
        {
            _integer(
                item,
                path=f"checkpoints.milestones[{index}]",
                minimum=0,
            )
            for index, item in enumerate(milestones)
        }
    )
    if canonical_milestones != milestones:
        raise CampaignError("checkpoints.milestones must be sorted unique")
    if any(item > generations for item in canonical_milestones):
        raise CampaignError("checkpoint milestone exceeds execution.generations")
    digest_every = checkpoints.get("digest_every_generation", not tick_native)
    if type(digest_every) is not bool:
        raise CampaignError("checkpoints.digest_every_generation must be boolean")
    checkpoint_storage_mode = checkpoints.get("storage_mode", "disk")
    if checkpoint_storage_mode not in {"disk", "disk_and_memory"}:
        raise CampaignError(
            "checkpoints.storage_mode must be 'disk' or 'disk_and_memory'"
        )
    acs_checkpoint_slots = _integer(
        checkpoints.get("acs_slots", ACS_CHECKPOINT_SLOTS_DEFAULT_V1),
        path="checkpoints.acs_slots",
        minimum=0,
    )
    if acs_checkpoint_slots not in {0, 1, 2, 3}:
        raise CampaignError("checkpoints.acs_slots must be 0, 1, 2, or 3")
    acs_capture_instrumentation = _boolean(
        checkpoints.get("acs_capture_instrumentation", False),
        path="checkpoints.acs_capture_instrumentation",
    )
    native_streamed_checkpoint = _boolean(
        checkpoints.get("native_streamed_checkpoint", False),
        path="checkpoints.native_streamed_checkpoint",
    )
    world_sharded_checkpoint = _boolean(
        checkpoints.get("world_sharded_checkpoint", False),
        path="checkpoints.world_sharded_checkpoint",
    )
    if world_sharded_checkpoint and not native_streamed_checkpoint:
        raise CampaignError(
            "checkpoints.world_sharded_checkpoint requires native_streamed_checkpoint"
        )
    if world_sharded_checkpoint and not tick_native:
        raise CampaignError(
            "checkpoints.world_sharded_checkpoint requires absolute-tick execution"
        )

    comparisons = _object(spec.get("comparisons", {}), path="comparisons")
    _exact(
        comparisons,
        frozenset(
            {
                "fresh_repeats",
                "fresh_selectors",
                "continuation_splits",
                "continuation_cells",
                "continuation_baseline",
                "continuation_selectors",
                "reporting_parity",
                "reporting_selector",
            }
        ),
        path="comparisons",
    )
    fresh_repeats = _integer(
        comparisons.get("fresh_repeats", 0), path="comparisons.fresh_repeats", minimum=0
    )
    continuation_cells = _integer(
        comparisons.get("continuation_cells", fresh_repeats),
        path="comparisons.continuation_cells",
        minimum=0,
    )
    continuation_baseline = comparisons.get("continuation_baseline", "fresh_branch")
    if continuation_baseline not in {"fresh_branch", "campaign_primary"}:
        raise CampaignError(
            "comparisons.continuation_baseline must be fresh_branch or campaign_primary"
        )
    reporting_parity = _boolean(
        comparisons.get("reporting_parity", True),
        path="comparisons.reporting_parity",
    )
    splits = comparisons.get("continuation_splits", [])
    if not isinstance(splits, list):
        raise CampaignError("comparisons.continuation_splits must be an array")
    canonical_splits = sorted(
        {
            _integer(item, path=f"comparisons.continuation_splits[{index}]")
            for index, item in enumerate(splits)
        }
    )
    if canonical_splits != splits or any(
        item >= generations for item in canonical_splits
    ):
        raise CampaignError(
            "continuation splits must be sorted unique and less than generations"
        )
    missing_split_milestones = sorted(set(canonical_splits) - set(canonical_milestones))
    if missing_split_milestones:
        raise CampaignError(
            f"continuation splits require checkpoint milestones: {missing_split_milestones}"
        )
    dimension_ids = {row["id"] for row in dimensions}

    def selectors(name: str) -> list[dict[str, Scalar]]:
        raw = comparisons.get(name, [])
        if not isinstance(raw, list):
            raise CampaignError(f"comparisons.{name} must be an array")
        result: list[dict[str, Scalar]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping) or not item:
                raise CampaignError(
                    f"comparisons.{name}[{index}] must be a non-empty object"
                )
            unknown = sorted(set(item) - dimension_ids)
            if unknown:
                raise CampaignError(
                    f"comparisons.{name}[{index}] has unknown dimensions: {unknown}"
                )
            if any(
                isinstance(value, (dict, list, tuple, set)) or value is None
                for value in item.values()
            ):
                raise CampaignError(
                    f"comparisons.{name}[{index}] values must be scalar"
                )
            result.append(dict(sorted(item.items())))
        if len({canonical_bytes(row) for row in result}) != len(result):
            raise CampaignError(f"comparisons.{name} must not contain duplicates")
        return result

    fresh_selectors = selectors("fresh_selectors")
    continuation_selectors = selectors("continuation_selectors")
    reporting_selector_raw = comparisons.get("reporting_selector")
    reporting_selector: dict[str, Scalar] | None = None
    if reporting_selector_raw is not None:
        if (
            not isinstance(reporting_selector_raw, Mapping)
            or not reporting_selector_raw
        ):
            raise CampaignError(
                "comparisons.reporting_selector must be a non-empty object"
            )
        unknown = sorted(set(reporting_selector_raw) - dimension_ids)
        if unknown:
            raise CampaignError(
                f"comparisons.reporting_selector has unknown dimensions: {unknown}"
            )
        if any(
            isinstance(value, (dict, list, tuple, set)) or value is None
            for value in reporting_selector_raw.values()
        ):
            raise CampaignError("comparisons.reporting_selector values must be scalar")
        reporting_selector = dict(sorted(reporting_selector_raw.items()))
    if fresh_selectors and fresh_repeats:
        raise CampaignError(
            "comparisons must use fresh_repeats or fresh_selectors, not both"
        )
    if continuation_selectors and continuation_cells:
        raise CampaignError(
            "comparisons must use continuation_cells or continuation_selectors, not both"
        )
    if tick_native and (
        fresh_repeats
        or fresh_selectors
        or continuation_cells
        or continuation_selectors
        or canonical_splits
        or reporting_selector is not None
    ):
        raise CampaignError(
            "tick-native campaigns do not support legacy comparison branches"
        )

    output = _object(spec.get("output"), path="output")
    _exact(
        output,
        frozenset(
            {
                "runtime_root",
                "reporting_profile",
                "chronicle_profile",
                "chronicle_history_root",
                "chronicle_archive_enabled",
                "evidence_closeout_enabled",
            }
        ),
        path="output",
    )
    runtime_root = Path(
        _text(output.get("runtime_root"), path="output.runtime_root", nonempty=True)
    )
    if not runtime_root.is_absolute():
        raise CampaignError("output.runtime_root must be absolute")
    reporting_profile = _identifier(
        output.get("reporting_profile", "compact"), path="output.reporting_profile"
    )
    chronicle_profile = _identifier(
        output.get("chronicle_profile", "notable"),
        path="output.chronicle_profile",
    )
    if chronicle_profile not in {"summary", "notable", "scientific_full"}:
        raise CampaignError(
            "output.chronicle_profile must be 'summary', 'notable', or "
            "'scientific_full'"
        )
    chronicle_history_root = Path(
        _text(
            output.get(
                "chronicle_history_root", public_runtime_path('chronicle_history')
            ),
            path="output.chronicle_history_root",
            nonempty=True,
        )
    )
    if not chronicle_history_root.is_absolute():
        raise CampaignError("output.chronicle_history_root must be absolute")
    chronicle_archive_enabled = output.get("chronicle_archive_enabled", True)
    if type(chronicle_archive_enabled) is not bool:
        raise CampaignError("output.chronicle_archive_enabled must be boolean")
    evidence_closeout_enabled = output.get("evidence_closeout_enabled", False)
    if type(evidence_closeout_enabled) is not bool:
        raise CampaignError("output.evidence_closeout_enabled must be boolean")

    raw_artifact_lifecycle = spec.get("artifact_lifecycle")
    raw_lifecycle_checkpoints = (
        raw_artifact_lifecycle.get("checkpoints")
        if isinstance(raw_artifact_lifecycle, Mapping)
        else None
    )
    tick_native_legacy_lifecycle = bool(
        tick_native
        and isinstance(raw_lifecycle_checkpoints, Mapping)
        and raw_lifecycle_checkpoints.get("authority", "legacy_generation")
        == "legacy_generation"
    )
    try:
        artifact_lifecycle = resolve_run_artifact_lifecycle_v1(
            raw_artifact_lifecycle,
            total_max_bytes=canonical_budgets["disk_bytes"],
            checkpoint_schedule=(
                checkpoint_ticks if tick_native else canonical_milestones
            ),
            terminal_generation=None if tick_native else generations,
            terminal_tick=(
                tick_request.total_ticks if tick_native and tick_request else None
            ),
            checkpoint_authority=(
                "absolute_tick" if tick_native else "legacy_generation"
            ),
            legacy=raw_artifact_lifecycle is None
            or spec.get("_legacy_lifecycle_derived") is True
            or tick_native_legacy_lifecycle,
        )
    except ArtifactLifecycleError as exc:
        raise CampaignError(str(exc)) from exc

    cancellation = execution.get("cancellation", True)
    if type(cancellation) is not bool:
        raise CampaignError("execution.cancellation must be boolean")

    canonical_batch: dict[str, Any] = {"mode": mode}
    if cells is not None:
        canonical_batch["cells"] = cells
    if maximum is not None:
        canonical_batch["maximum"] = maximum

    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "description": description,
        "no_claims": no_claims,
        **({"run_contract": run_contract} if run_contract is not None else {}),
        "world": {"base": world_base, "overlays": overlays},
        "policy": {"family": policy_family},
        "selection": {"mode": selection_mode},
        "parameters": canonical_parameters,
        "matrix": {
            "dimensions": dimensions,
            "parameter_bindings": dict(sorted(parameter_bindings.items())),
            "seed_bindings": dict(sorted(seed_bindings.items())),
            "include": matrix.get("include", []),
            "exclude": matrix.get("exclude", []),
            "expected_cells": matrix.get("expected_cells"),
            "max_cells": _integer(
                matrix.get("max_cells", 10_000), path="matrix.max_cells"
            ),
        },
        "execution": {
            "backend": backend,
            **(
                {
                    "execution_route_identity": TICK_NATIVE_EXECUTION_ROUTE_V1,
                    "total_ticks": tick_request.total_ticks,
                    "start_next_tick": tick_request.start_next_tick,
                    "checkpoint_ticks": list(checkpoint_ticks),
                    "deliberate_restore_tick": tick_request.deliberate_restore_tick,
                    "deliberate_restore_ticks": list(
                        tick_request.canonical_deliberate_restore_ticks
                    ),
                    "tick_native_control": (
                        tick_control.payload() if tick_control is not None else {}
                    ),
                    **(
                        {"checkpoint_every_ticks": tick_request.checkpoint_every_ticks}
                        if tick_request.checkpoint_every_ticks is not None
                        else {}
                    ),
                }
                if tick_native and tick_request is not None
                else {
                    "generations": generations,
                    "ticks_per_generation": ticks,
                }
            ),
            **(
                (
                    {"physical_window_generations": physical_window_generations}
                    if physical_window_generations is not None
                    else {}
                )
                if not tick_native
                else {}
            ),
            **(
                (
                    {
                        "max_internal_lifecycle_segments_per_dispatch": (
                            max_internal_lifecycle_segments_per_dispatch
                        )
                    }
                    if max_internal_lifecycle_segments_per_dispatch is not None
                    else {}
                )
                if not tick_native
                else {}
            ),
            **(
                {"gpu_resource_policy": gpu_resource_policy_selector}
                if gpu_resource_policy_selector is not None
                else {}
            ),
            "batch": canonical_batch,
            "budgets": canonical_budgets,
            "cancellation": cancellation,
            "technical_gate": {
                "selectors": technical_gate_selectors_raw,
                "partition": technical_gate_partition,
                **(
                    {"profile": technical_gate_profile}
                    if technical_gate_profile is not None
                    else {}
                ),
                **(
                    {"generations": technical_gate_generations}
                    if not tick_native and technical_gate_generations_raw is not None
                    else {}
                ),
                **(
                    {"checkpoint_milestones": technical_gate_milestones}
                    if not tick_native and technical_gate_milestones_raw is not None
                    else {}
                ),
                **(
                    {"total_ticks": technical_gate_total_ticks}
                    if tick_native and technical_gate_total_ticks_raw is not None
                    else {}
                ),
                **(
                    {"checkpoint_ticks": technical_gate_checkpoint_ticks}
                    if tick_native and technical_gate_checkpoint_ticks_raw is not None
                    else {}
                ),
            },
        },
        "checkpoints": {
            "milestones": canonical_milestones,
            "digest_every_generation": digest_every,
            "storage_mode": checkpoint_storage_mode,
            "acs_slots": acs_checkpoint_slots,
            **(
                {"native_streamed_checkpoint": True}
                if native_streamed_checkpoint
                else {}
            ),
            **(
                {"world_sharded_checkpoint": True}
                if world_sharded_checkpoint
                else {}
            ),
            **(
                {"acs_capture_instrumentation": True}
                if acs_capture_instrumentation
                else {}
            ),
        },
        "comparisons": {
            "fresh_repeats": fresh_repeats,
            "fresh_selectors": fresh_selectors,
            "continuation_splits": canonical_splits,
            "continuation_cells": continuation_cells,
            "continuation_baseline": continuation_baseline,
            "continuation_selectors": continuation_selectors,
            "reporting_parity": reporting_parity,
            "reporting_selector": reporting_selector,
        },
        "artifacts": _canonical_artifacts(spec.get("artifacts")),
        "analysis": _analysis(spec.get("analysis", {})),
        "output": {
            "runtime_root": str(runtime_root),
            "reporting_profile": reporting_profile,
            "chronicle_profile": chronicle_profile,
            "chronicle_history_root": str(chronicle_history_root),
            "chronicle_archive_enabled": chronicle_archive_enabled,
            "evidence_closeout_enabled": evidence_closeout_enabled,
        },
        "artifact_lifecycle": artifact_lifecycle,
        **(
            {"_legacy_lifecycle_derived": True}
            if spec.get("artifact_lifecycle") is None
            or spec.get("_legacy_lifecycle_derived") is True
            else {}
        ),
    }


def load_campaign_spec(path: str | Path) -> dict[str, Any]:
    try:
        return validate_campaign_spec(load_document(path))
    except QualificationPlanError as exc:
        raise CampaignError(str(exc)) from exc


def _compile_matrix(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    matrix = spec["matrix"]
    try:
        compiled = compile_matrix(
            {
                "schema_version": MATRIX_SCHEMA_VERSION,
                "plan_id": spec["campaign_id"],
                "description": spec["description"],
                "dimensions": matrix["dimensions"],
                "capability_requirements": [],
                "case_id_prefix": "cell",
                "include": matrix["include"],
                "exclude": matrix["exclude"],
                "templates": [],
                "max_cases": matrix["max_cells"],
            }
        )
    except QualificationPlanError as exc:
        raise CampaignError(str(exc)) from exc
    cases = list(compiled["cases"])
    expected = matrix["expected_cells"]
    if expected is not None and _integer(expected, path="matrix.expected_cells") != len(
        cases
    ):
        raise CampaignError(
            f"matrix expected_cells={expected} but compiled {len(cases)}"
        )
    dimension_ids = {dimension["id"] for dimension in compiled["dimensions"]}
    bound = set(matrix["parameter_bindings"]) | set(matrix["seed_bindings"])
    unbound = sorted(dimension_ids - bound)
    unknown = sorted(bound - dimension_ids)
    if unbound or unknown:
        raise CampaignError(
            f"matrix bindings are incomplete; unbound={unbound} unknown={unknown}"
        )
    return cases


def _requirements(spec: Mapping[str, Any]) -> CampaignRequirementsV1:
    parameter_names = set(spec["parameters"]) | set(
        spec["matrix"]["parameter_bindings"].values()
    )
    required_metrics = set(spec["analysis"]["required_metrics"])
    required_metrics.update(rule["metric"] for rule in spec["analysis"]["rules"])
    return CampaignRequirementsV1(
        backend=spec["execution"]["backend"],
        world_base=spec["world"]["base"],
        world_overlays=frozenset(spec["world"]["overlays"]),
        policy_family=spec["policy"]["family"],
        selection_mode=spec["selection"]["mode"],
        parameter_names=frozenset(parameter_names),
        required_metrics=frozenset(required_metrics),
        require_continuation=bool(spec["comparisons"]["continuation_splits"]),
        require_cancellation=bool(spec["execution"]["cancellation"]),
    )


def _batch_size(
    spec: Mapping[str, Any],
    *,
    profile_batch_max: int,
    qualified_batch_max: int,
    cells: int,
) -> int:
    batch = spec["execution"]["batch"]
    if batch["mode"] == "explicit":
        requested = int(batch["cells"])
    else:
        # Automatic and bounded modes choose a safe admitted physical extent.
        # The YAML maximum remains a ceiling, not a request to form an
        # execution-ineligible batch and fail later in preflight.
        requested = min(int(batch["maximum"]), qualified_batch_max)
    if requested > profile_batch_max:
        raise CampaignError(
            f"requested batch {requested} exceeds profile admitted maximum {profile_batch_max}"
        )
    return min(requested, cells)


def _partition_physical_batches(
    cells: list[LogicalCellV1],
    *,
    maximum_cells: int,
    uniform_parameters: tuple[str, ...],
) -> tuple[PhysicalBatchV1, ...]:
    """Partition cells by profile-declared native scalar coherence boundaries."""

    groups: dict[bytes, list[LogicalCellV1]] = {}
    for cell in cells:
        missing = [name for name in uniform_parameters if name not in cell.parameters]
        if missing:
            raise CampaignError(
                f"profile uniform batch parameters are absent from {cell.cell_id}: {missing}"
            )
        coherence = [[name, cell.parameters[name]] for name in uniform_parameters]
        groups.setdefault(canonical_bytes(coherence), []).append(cell)

    batches: list[PhysicalBatchV1] = []
    for group in groups.values():
        for index in range(0, len(group), maximum_cells):
            chunk = group[index : index + maximum_cells]
            ordinal = len(batches)
            batches.append(
                PhysicalBatchV1(
                    ordinal=ordinal,
                    batch_id=f"batch-{ordinal:04d}",
                    cell_ids=tuple(cell.cell_id for cell in chunk),
                )
            )
    return tuple(batches)


def _uses_acs100x_fixed_mlp_b136_compact_accounting(
    spec: Mapping[str, Any],
    *,
    cells: int,
    batch_size: int,
    comparison_trajectories: int,
) -> bool:
    """Select the source-bound compact accounting fence for physical B136."""

    parameters = spec["parameters"]
    execution = spec["execution"]
    control = execution.get("tick_native_control")
    return (
        execution.get("execution_route_identity") == TICK_NATIVE_EXECUTION_ROUTE_V1
        and bool(spec["checkpoints"].get("native_streamed_checkpoint", False))
        and spec["policy"]["family"] == "fixed_mlp"
        and spec["world"]["base"] == "patch"
        and cells == 136
        and batch_size == 136
        and comparison_trajectories == 0
        and isinstance(control, Mapping)
        and control.get("resident_owner_span_ticks") == 1_120
        and control.get("native_dispatch_window_ticks") == 1_120
        and control.get("source_window_ticks") == 160
        and control.get("event_window_capacity")
        == physical_b136_event_window_capacity_v1()
        and parameters.get("patch.grid_size") == 32
        and parameters.get("patch.initial_population") == 256
        and parameters.get("patch.capacity") == 512
        and parameters.get("eed.phenotype_fixed") is False
        and parameters.get("cuda_architecture_b.cpu_request_staging_enable") is False
        and parameters.get("policy_weight_mutation_enable", False) is False
    )


def _estimate(
    spec: Mapping[str, Any], *, cells: int, batch_size: int
) -> dict[str, int | float | str]:
    tick_native = (
        spec["execution"].get("execution_route_identity")
        == TICK_NATIVE_EXECUTION_ROUTE_V1
    )
    if tick_native:
        total_ticks = int(spec["execution"]["total_ticks"])
        control = spec["execution"].get("tick_native_control")
        if not isinstance(control, Mapping):
            raise CampaignError("tick-native estimate is missing tick_native_control")
        owner_span = int(control["resident_owner_span_ticks"])
        dispatch_window = int(control["native_dispatch_window_ticks"])
        checkpoint_schedule = tuple(spec["execution"].get("checkpoint_ticks") or ())
        checkpoints = len(checkpoint_schedule)
        # Resident estimates use active windows only.  Historical output and
        # checkpoint bytes may scale with the declared schedule, but the
        # source/event resident terms are deliberately horizon-independent.
        source_window = int(control["source_window_ticks"])
        active_source_windows = max(
            1, (dispatch_window + source_window - 1) // source_window
        )
        generations = active_source_windows
    else:
        total_ticks = int(spec["execution"]["generations"]) * int(
            spec["execution"]["ticks_per_generation"]
        )
        generations = int(spec["execution"]["generations"])
        checkpoints = len(spec["checkpoints"]["milestones"]) + 1
    fresh_count = len(spec["comparisons"]["fresh_selectors"]) or int(
        spec["comparisons"]["fresh_repeats"]
    )
    continuation_count = len(spec["comparisons"]["continuation_selectors"]) or int(
        spec["comparisons"]["continuation_cells"]
    )
    comparison_trajectories = (
        fresh_count
        + (continuation_count * len(spec["comparisons"]["continuation_splits"]))
        + int(spec["comparisons"]["reporting_parity"])
    )
    trajectories = cells + comparison_trajectories
    # Conservative profile-independent planning estimates. Profile preflight replaces
    # these with audited tensor/resource-provider values before execution authority.
    if tick_native:
        comparison_trajectories = 0
    else:
        comparison_trajectories = (
            fresh_count
            + (continuation_count * len(spec["comparisons"]["continuation_splits"]))
            + int(spec["comparisons"]["reporting_parity"])
        )
    trajectories = cells + comparison_trajectories
    # Conservative profile-independent planning estimates. Profile preflight replaces
    # these with audited tensor/resource-provider values before execution authority.
    summary_bytes = (
        trajectories * total_ticks * 32
        if tick_native
        else trajectories * generations * 4096
    )
    checkpoint_bytes = trajectories * checkpoints * 64 * 1024 * 1024
    native_aggregate = bool(
        spec["checkpoints"].get("native_streamed_checkpoint", False)
        and batch_size > 1
    )
    physical_batch_count = (cells + batch_size - 1) // batch_size
    checkpoint_physical_object_count = checkpoints * (
        physical_batch_count + comparison_trajectories
        if native_aggregate
        else trajectories
    )
    execution = spec["execution"]
    batch = execution["batch"]
    lifecycle = spec.get("artifact_lifecycle")
    compact_resident_batch = (
        execution.get("backend") == "cuda_resident"
        and batch.get("mode") in {"automatic", "bounded"}
        and batch_size < cells
        and isinstance(lifecycle, Mapping)
        and lifecycle.get("lifecycle_profile") == "SCIENCE_COMPACT"
    )
    checkpoint_bytes_per_trajectory_boundary = (
        RESIDENT_COMPACT_CHECKPOINT_BYTES_PER_TRAJECTORY_BOUNDARY
        if compact_resident_batch
        else CHECKPOINT_BYTES_PER_TRAJECTORY_BOUNDARY
    )
    checkpoint_trajectories = (
        batch_size + comparison_trajectories
        if compact_resident_batch
        else trajectories
    )
    checkpoint_accounting_basis = "legacy_logical_trajectory_v1"
    checkpoint_bytes_per_boundary = checkpoint_bytes // checkpoints
    checkpoint_physical_payload_bytes = 0
    checkpoint_physical_metadata_bytes = 0
    compact_b136 = _uses_acs100x_fixed_mlp_b136_compact_accounting(
        spec,
        cells=cells,
        batch_size=batch_size,
        comparison_trajectories=comparison_trajectories,
    )
    if compact_b136:
        compact = planning_acs_native_fixed_mlp_b136_checkpoint_upper_bound_v1(
            logical_cells=cells,
            batch_cells=batch_size,
            checkpoint_boundaries=checkpoints,
            comparison_trajectories=comparison_trajectories,
        )
        checkpoint_accounting_basis = str(compact["checkpoint_accounting_basis"])
        checkpoint_bytes = int(compact["checkpoint_bytes_upper_bound"])
        checkpoint_bytes_per_boundary = int(
            compact["checkpoint_bytes_per_boundary_upper_bound"]
        )
        checkpoint_physical_payload_bytes = int(
            compact["checkpoint_physical_payload_bytes_per_boundary"]
        )
        checkpoint_physical_metadata_bytes = int(
            compact["checkpoint_physical_metadata_bytes_per_boundary"]
        )
        checkpoint_physical_object_count = int(
            compact["checkpoint_physical_object_count_upper_bound"]
        )
    if compact_resident_batch and not compact_b136:
        checkpoint_bytes = (
            checkpoint_trajectories
            * checkpoints
            * checkpoint_bytes_per_trajectory_boundary
        )
        checkpoint_bytes_per_boundary = checkpoint_bytes // checkpoints
        checkpoint_physical_object_count = checkpoints * (
            physical_batch_count + comparison_trajectories
        )
        checkpoint_accounting_basis = "resident_compact_batch_v1"
    disk_upper_bound = (
        summary_bytes + checkpoint_bytes
        if compact_b136
        else (
            planning_disk_upper_bound_absolute_ticks_v1(
                logical_trajectories=trajectories,
                total_ticks=total_ticks,
                checkpoint_boundaries=checkpoints,
            )
            if tick_native
            else planning_disk_upper_bound_v1(
                logical_trajectories=trajectories,
                logical_generations=generations,
                checkpoint_boundaries=checkpoints,
                checkpoint_bytes_per_trajectory_boundary=(
                    checkpoint_bytes_per_trajectory_boundary
                ),
            )
        )
    )
    if compact_b136 or compact_resident_batch:
        disk_upper_bound = summary_bytes + checkpoint_bytes
    elif tick_native:
        disk_upper_bound = planning_disk_upper_bound_absolute_ticks_v1(
            logical_trajectories=trajectories,
            total_ticks=total_ticks,
            checkpoint_boundaries=checkpoints,
        )
    else:
        disk_upper_bound = planning_disk_upper_bound_v1(
            logical_trajectories=trajectories,
            logical_generations=generations,
            checkpoint_boundaries=checkpoints,
            checkpoint_bytes_per_trajectory_boundary=(
                checkpoint_bytes_per_trajectory_boundary
            ),
        )
    return {
        "authority": "planning_upper_bound_not_execution_admission",
        "logical_cells": cells,
        "logical_trajectories": trajectories,
        "comparison_trajectories": comparison_trajectories,
        "physical_batch_cells": batch_size,
        "summary_bytes_upper_bound": summary_bytes,
        "checkpoint_bytes_upper_bound": checkpoint_bytes,
        "checkpoint_trajectories_upper_bound": checkpoint_trajectories,
        "checkpoint_bytes_per_trajectory_boundary": (
            checkpoint_bytes_per_trajectory_boundary
        ),
        "checkpoint_bytes_per_boundary_upper_bound": checkpoint_bytes_per_boundary,
        "checkpoint_accounting_basis": checkpoint_accounting_basis,
        "checkpoint_physical_payload_bytes_per_boundary": (
            checkpoint_physical_payload_bytes
        ),
        "checkpoint_physical_metadata_bytes_per_boundary": (
            checkpoint_physical_metadata_bytes
        ),
        "checkpoint_boundaries": checkpoints,
        "checkpoint_representation": (
            "aggregate_physical_batch_v1"
            if native_aggregate
            else "per_trajectory_bundle_v1"
        ),
        "checkpoint_physical_object_count_upper_bound": (
            checkpoint_physical_object_count
        ),
        "disk_bytes_upper_bound": disk_upper_bound,
        "peak_resident_gpu_owner_event_bytes": (
            "unavailable_until_execution_provider_admission"
        ),
        "peak_host_checkpoint_import_scratch_bytes": (
            CHECKPOINT_BYTES_PER_TRAJECTORY_BOUNDARY
        ),
        "total_durable_output_checkpoint_bytes_upper_bound": disk_upper_bound,
        "resident_gpu_owner_event_bytes_horizon_invariant": True,
        "host_checkpoint_import_scratch_bytes_horizon_invariant": True,
        "resource_dimensions_schema_version": (
            "eiel_three_domain_resource_dimensions_v1"
        ),
        "vram_bytes_upper_bound": "unavailable_until_execution_provider_admission",
        "wall_seconds_upper_bound": "unavailable_until_execution_provider_admission",
        "batch_admission": "planning_ceiling_not_resource_selected",
        "logical_agent_ticks": trajectories * total_ticks,
        **(
            {
                "resident_owner_span_ticks": owner_span,
                "native_dispatch_window_ticks": dispatch_window,
                "resident_event_capacity": int(control["event_window_capacity"]),
                "resident_event_bytes_horizon_invariant": True,
                "resident_source_bytes_horizon_invariant": True,
                "requested_total_ticks": total_ticks,
            }
            if tick_native
            else {}
        ),
        "declared_vram_bytes": int(spec["execution"]["budgets"]["vram_bytes"]),
        "declared_disk_bytes": int(spec["execution"]["budgets"]["disk_bytes"]),
        "declared_wall_seconds": int(spec["execution"]["budgets"]["wall_seconds"]),
    }


def _apply_signed_workhorse_storage_accounting_v2(
    estimate: dict[str, int | float | str],
    accounting: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    profile_id: str,
    logical_cells: int,
    physical_batch_cells: int,
    compiled_batch_count: int,
    comparison_branch_count: int,
) -> None:
    expected_fields = {
        "schema_version",
        "profile_id",
        "operation",
        "operation_contract_sha256",
        "logical_cells",
        "physical_batch_cells",
        "compiled_batch_count",
        "shape_id",
        "backend_id",
        "physical_B",
        "temporal_k",
        "total_ticks",
        "checkpoint_schedule_sha256",
        "storage_accounting_sha256",
        "authority",
        "resource_accounting_binding_sha256",
        "storage_accounting",
    }
    if not isinstance(accounting, Mapping) or set(accounting) != expected_fields:
        raise CampaignError("signed Workhorse storage accounting shape is invalid")
    analysis = spec.get("analysis")
    extension = analysis.get("extension") if isinstance(analysis, Mapping) else None
    operation = extension.get("contract") if isinstance(extension, Mapping) else None
    execution = spec.get("execution")
    checkpoints = spec.get("checkpoints")
    if (
        not isinstance(operation, Mapping)
        or not isinstance(execution, Mapping)
        or not isinstance(checkpoints, Mapping)
        or logical_cells != 1
        or physical_batch_cells != 1
        or compiled_batch_count != 1
        or comparison_branch_count != 0
    ):
        raise CampaignError("signed Workhorse storage accounting route is invalid")
    contract = operation.get("contract")
    if not isinstance(contract, Mapping):
        raise CampaignError("signed Workhorse storage accounting lacks contract")
    checkpoint_ticks = tuple(execution.get("checkpoint_ticks") or ())
    schedule_sha256 = canonical_sha256(
        {
            "time_authority": "absolute_tick_v1",
            "checkpoint_ticks": list(checkpoint_ticks),
        }
    )
    binding = {
        key: accounting[key]
        for key in (
            "schema_version",
            "profile_id",
            "operation",
            "operation_contract_sha256",
            "logical_cells",
            "physical_batch_cells",
            "compiled_batch_count",
            "shape_id",
            "backend_id",
            "physical_B",
            "temporal_k",
            "total_ticks",
            "checkpoint_schedule_sha256",
            "storage_accounting_sha256",
        )
    }
    if (
        accounting["schema_version"]
        != "eiel_workhorse_signed_planning_storage_accounting_v2"
        or accounting["authority"] != "signed_provider_operation_contract_v2"
        or accounting["profile_id"] != profile_id
        or accounting["operation"] != operation.get("operation")
        or accounting["operation_contract_sha256"]
        != extension.get("contract_sha256")
        or accounting["logical_cells"] != logical_cells
        or accounting["physical_batch_cells"] != physical_batch_cells
        or accounting["compiled_batch_count"] != compiled_batch_count
        or accounting["shape_id"] != contract.get("shape_id")
        or accounting["backend_id"] != contract.get("backend_id")
        or accounting["physical_B"] != contract.get("B")
        or accounting["temporal_k"] != contract.get("temporal_k")
        or accounting["total_ticks"] != contract.get("total_ticks")
        or accounting["checkpoint_schedule_sha256"] != schedule_sha256
        or accounting["resource_accounting_binding_sha256"]
        != canonical_sha256(binding)
    ):
        raise CampaignError(
            "signed Workhorse storage accounting differs from compiled operation"
        )
    storage = accounting["storage_accounting"]
    storage_fields = {
        "schema_version",
        "time_authority",
        "shape_id",
        "physical_B",
        "state_bytes_per_world",
        "checkpoint_payload_bytes",
        "checkpoint_ticks",
        "checkpoint_count",
        "physical_checkpoint_payload_copies",
        "absolute_tick_projection_payload_copies",
        "indexed_world_payload_copies",
        "checkpoint_durable_payload_bytes_upper_bound",
        "checkpoint_index_and_occurrence_metadata_bytes_upper_bound",
        "checkpoint_bytes_upper_bound",
        "acs_checkpoint_slots",
        "acs_pending_capture_files_upper_bound",
        "acs_pending_capture_file_bytes_upper_bound",
        "event_dispatch_count_upper_bound",
        "event_bytes_upper_bound",
        "durable_noncheckpoint_bytes_upper_bound",
        "fixed_output_bytes_upper_bound",
        "compact_output_bytes_upper_bound",
        "durable_noncheckpoint_bytes_upper_bound",
        "disk_bytes_upper_bound",
        "horizon_dependent_resident_bytes",
        "storage_model",
    }
    if not isinstance(storage, Mapping) or set(storage) != storage_fields:
        raise CampaignError("signed Workhorse storage payload shape is invalid")
    slots = checkpoints.get("acs_slots", 0)
    checkpoint_count = len(checkpoint_ticks) if slots else 0
    integer_fields = storage_fields - {
        "schema_version",
        "time_authority",
        "shape_id",
        "checkpoint_ticks",
        "storage_model",
    }
    if any(type(storage[key]) is not int or storage[key] < 0 for key in integer_fields):
        raise CampaignError("signed Workhorse storage bytes are invalid")
    if (
        storage["schema_version"] != "eiel_workhorse_checkpoint_storage_accounting_v2"
        or storage["time_authority"] != "absolute_tick_v1"
        or storage["shape_id"] != accounting["shape_id"]
        or storage["physical_B"] != accounting["physical_B"]
        or storage["checkpoint_ticks"] != list(checkpoint_ticks)
        or storage["checkpoint_count"] != checkpoint_count
        or storage["acs_checkpoint_slots"] != slots
        or storage["physical_checkpoint_payload_copies"] != checkpoint_count
        or storage["absolute_tick_projection_payload_copies"] != checkpoint_count
        or storage["indexed_world_payload_copies"] != 0
        or storage["checkpoint_payload_bytes"]
        != storage["physical_B"] * storage["state_bytes_per_world"]
        or storage["checkpoint_durable_payload_bytes_upper_bound"]
        != storage["checkpoint_payload_bytes"] * 2 * checkpoint_count
        or storage["checkpoint_bytes_upper_bound"]
        != storage["checkpoint_durable_payload_bytes_upper_bound"]
        + storage["checkpoint_index_and_occurrence_metadata_bytes_upper_bound"]
        or storage["acs_pending_capture_files_upper_bound"]
        != min(slots, checkpoint_count)
        or storage["acs_pending_capture_file_bytes_upper_bound"]
        != storage["acs_pending_capture_files_upper_bound"]
        * storage["checkpoint_payload_bytes"]
        or storage["durable_noncheckpoint_bytes_upper_bound"]
        != storage["event_bytes_upper_bound"]
        + storage["fixed_output_bytes_upper_bound"]
        + storage["compact_output_bytes_upper_bound"]
        or storage["disk_bytes_upper_bound"]
        != storage["checkpoint_bytes_upper_bound"]
        + storage["acs_pending_capture_file_bytes_upper_bound"]
        + storage["durable_noncheckpoint_bytes_upper_bound"]
        or storage["horizon_dependent_resident_bytes"] != 0
        or accounting["storage_accounting_sha256"] != canonical_sha256(storage)
    ):
        raise CampaignError("signed Workhorse storage arithmetic is invalid")
    estimate.update(
        {
            "authority": "planning_upper_bound_with_signed_workhorse_storage_v2",
            "resource_estimator_id": "workhorse_v2_exact_checkpoint_storage_v2",
            "resource_accounting_binding_sha256": accounting[
                "resource_accounting_binding_sha256"
            ],
            "checkpoint_accounting_basis": storage["storage_model"],
            "checkpoint_representation": "aggregate_plus_indexed_ranges_v2",
            "checkpoint_bytes_upper_bound": storage[
                "checkpoint_bytes_upper_bound"
            ],
            "checkpoint_physical_payload_bytes_per_boundary": storage[
                "checkpoint_payload_bytes"
            ],
            "checkpoint_physical_object_count_upper_bound": 2 * checkpoint_count,
            "acs_pending_capture_file_bytes_upper_bound": storage[
                "acs_pending_capture_file_bytes_upper_bound"
            ],
            "event_bytes_upper_bound": storage["event_bytes_upper_bound"],
            "fixed_output_bytes_upper_bound": storage[
                "fixed_output_bytes_upper_bound"
            ],
            "durable_noncheckpoint_bytes_upper_bound": storage[
                "durable_noncheckpoint_bytes_upper_bound"
            ],
            "disk_bytes_upper_bound": storage["disk_bytes_upper_bound"],
            "total_durable_output_checkpoint_bytes_upper_bound": storage[
                "disk_bytes_upper_bound"
            ],
            "horizon_dependent_resident_bytes": 0,
        }
    )


def _apply_signed_physical_checkpoint_accounting_v1(
    estimate: dict[str, int | float | str],
    accounting: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    profile_id: str,
    logical_cells: int,
    physical_batch_cells: int,
    compiled_batch_count: int,
    comparison_branch_count: int,
) -> None:
    """Apply one provider-authenticated aggregate checkpoint planning bound."""

    execution = spec["execution"]
    shared_policy = execution.get("gpu_resource_policy") == (
        "w32x32p256c512_gpuv2_shared"
    )
    expected_keys = {
        "schema_version",
        "authority",
        "profile_id",
        "execution_route_identity",
        "resource_estimator_id",
        "resource_accounting_binding_sha256",
        "artifact_binary_sha256",
        "artifact_manifest_sha256",
        "device_selection_receipt_sha256",
        "device_identity_sha256",
        "event_transport",
        "logical_cells",
        "physical_batch_cells",
        "temporal_k",
        "total_ticks",
        "checkpoint_schedule_sha256",
        "checkpoint_boundaries",
        "checkpoint_representation",
        "checkpoint_accounting_basis",
        "checkpoint_bytes_per_trajectory_boundary",
        "checkpoint_physical_payload_bytes_per_boundary",
        "checkpoint_physical_metadata_bytes_per_boundary",
        "checkpoint_bytes_per_boundary_upper_bound",
        "checkpoint_bytes_upper_bound",
        "checkpoint_physical_object_count_upper_bound",
        "event_bytes_upper_bound",
        "durable_noncheckpoint_bytes_upper_bound",
        "fixed_output_bytes_upper_bound",
        "disk_bytes_upper_bound",
        "horizon_dependent_resident_bytes",
    }
    if shared_policy:
        expected_keys.update(
            {
                "compiled_batch_count",
                "gpu_resource_policy_id",
                "vram_bytes_upper_bound",
                "wall_seconds_upper_bound",
                "serial_physical_batch_execution",
            }
        )
    if not isinstance(accounting, Mapping) or set(accounting) != expected_keys:
        raise CampaignError("signed physical checkpoint accounting shape is invalid")
    parameters = spec["parameters"]
    checkpoint_ticks = tuple(execution.get("checkpoint_ticks") or ())
    expected_schedule_sha256 = canonical_sha256(
        {
            "time_authority": "absolute_tick_v1",
            "checkpoint_ticks": list(checkpoint_ticks),
        }
    )
    identity = {
        "schema_version": SIGNED_PHYSICAL_CHECKPOINT_ACCOUNTING_SCHEMA_V1,
        "authority": "signed_provider_artifact_device_selection_v1",
        "profile_id": profile_id,
        "execution_route_identity": TICK_NATIVE_EXECUTION_ROUTE_V1,
        "resource_estimator_id": (
            "cuda_full_evolution_v2_direct_sparse_sealed_bk_v3"
        ),
        "event_transport": "direct_sparse_v3",
        "logical_cells": logical_cells,
        "physical_batch_cells": physical_batch_cells,
        "temporal_k": parameters.get("cuda_native.temporal_k"),
        "total_ticks": execution.get("total_ticks"),
        "checkpoint_schedule_sha256": expected_schedule_sha256,
        "checkpoint_boundaries": len(checkpoint_ticks),
        "checkpoint_representation": "aggregate_physical_batch_v1",
        "checkpoint_accounting_basis": (
            "cuda_full_evolution_v2_signed_native_aggregate_v1"
        ),
        "horizon_dependent_resident_bytes": 0,
        **(
            {
                "compiled_batch_count": compiled_batch_count,
                "gpu_resource_policy_id": (
                    "w32x32p256c512_gpuv2_shared"
                ),
                "serial_physical_batch_execution": True,
            }
            if shared_policy
            else {}
        ),
    }
    if any(accounting.get(key) != value for key, value in identity.items()):
        raise CampaignError(
            "signed physical checkpoint accounting differs from the compiled route"
        )
    if (
        comparison_branch_count != 0
        or (
            shared_policy
            and logical_cells != physical_batch_cells * compiled_batch_count
        )
        or (not shared_policy and compiled_batch_count != 1)
    ):
        raise CampaignError(
            "signed physical checkpoint accounting physical partition is invalid"
        )
    for key in (
        "resource_accounting_binding_sha256",
        "artifact_binary_sha256",
        "artifact_manifest_sha256",
        "device_selection_receipt_sha256",
        "device_identity_sha256",
    ):
        if (
            not isinstance(accounting[key], str)
            or _SHA_RE.fullmatch(accounting[key]) is None
        ):
            raise CampaignError(
                f"signed physical checkpoint accounting {key} is invalid"
            )
    binding_identity = {
        "schema_version": accounting["schema_version"],
        "profile_id": accounting["profile_id"],
        "artifact_binary_sha256": accounting["artifact_binary_sha256"],
        "artifact_manifest_sha256": accounting["artifact_manifest_sha256"],
        "device_selection_receipt_sha256": accounting[
            "device_selection_receipt_sha256"
        ],
        "device_identity_sha256": accounting["device_identity_sha256"],
        "event_transport": accounting["event_transport"],
        "logical_cells": accounting["logical_cells"],
        "physical_batch_cells": accounting["physical_batch_cells"],
        "temporal_k": accounting["temporal_k"],
        "total_ticks": accounting["total_ticks"],
        "checkpoint_schedule_sha256": accounting["checkpoint_schedule_sha256"],
        **(
            {
                "compiled_batch_count": accounting["compiled_batch_count"],
                "gpu_resource_policy_id": accounting["gpu_resource_policy_id"],
            }
            if shared_policy
            else {}
        ),
    }
    expected_binding_sha256 = canonical_sha256(binding_identity)
    if accounting["resource_accounting_binding_sha256"] != expected_binding_sha256:
        raise CampaignError(
            "signed physical checkpoint accounting identity binding is invalid"
        )
    positive_integer_keys = (
        "logical_cells",
        "physical_batch_cells",
        "temporal_k",
        "total_ticks",
        "checkpoint_boundaries",
        "checkpoint_bytes_per_trajectory_boundary",
        "checkpoint_physical_payload_bytes_per_boundary",
        "checkpoint_bytes_per_boundary_upper_bound",
        "checkpoint_bytes_upper_bound",
        "checkpoint_physical_object_count_upper_bound",
        "disk_bytes_upper_bound",
        *(
            (
                "compiled_batch_count",
                "vram_bytes_upper_bound",
                "wall_seconds_upper_bound",
            )
            if shared_policy
            else ()
        ),
    )
    nonnegative_integer_keys = (
        "checkpoint_physical_metadata_bytes_per_boundary",
        "event_bytes_upper_bound",
        "durable_noncheckpoint_bytes_upper_bound",
        "fixed_output_bytes_upper_bound",
        "horizon_dependent_resident_bytes",
    )
    if any(
        type(accounting[key]) is not int or accounting[key] < 1
        for key in positive_integer_keys
    ) or any(
        type(accounting[key]) is not int or accounting[key] < 0
        for key in nonnegative_integer_keys
    ):
        raise CampaignError("signed physical checkpoint accounting bytes are invalid")
    if (
        accounting["checkpoint_bytes_per_boundary_upper_bound"]
        != accounting["checkpoint_physical_payload_bytes_per_boundary"]
        + accounting["checkpoint_physical_metadata_bytes_per_boundary"]
        or accounting["checkpoint_bytes_upper_bound"]
        != accounting["checkpoint_boundaries"]
        * (compiled_batch_count if shared_policy else 1)
        * accounting["checkpoint_bytes_per_boundary_upper_bound"]
        or accounting["checkpoint_physical_object_count_upper_bound"]
        != accounting["checkpoint_boundaries"]
        * (compiled_batch_count if shared_policy else 1)
        or accounting["disk_bytes_upper_bound"]
        != accounting["checkpoint_bytes_upper_bound"]
        + accounting["event_bytes_upper_bound"]
        + accounting["durable_noncheckpoint_bytes_upper_bound"]
        + accounting["fixed_output_bytes_upper_bound"]
    ):
        raise CampaignError(
            "signed physical checkpoint accounting arithmetic is invalid"
        )
    estimate.update(
        {
            "authority": (
                "planning_upper_bound_with_signed_provider_physical_accounting_v1"
            ),
            "checkpoint_bytes_upper_bound": int(
                accounting["checkpoint_bytes_upper_bound"]
            ),
            "checkpoint_bytes_per_trajectory_boundary": int(
                accounting["checkpoint_bytes_per_trajectory_boundary"]
            ),
            "checkpoint_bytes_per_boundary_upper_bound": int(
                accounting["checkpoint_bytes_per_boundary_upper_bound"]
            ),
            "checkpoint_accounting_basis": str(
                accounting["checkpoint_accounting_basis"]
            ),
            "checkpoint_physical_payload_bytes_per_boundary": int(
                accounting["checkpoint_physical_payload_bytes_per_boundary"]
            ),
            "checkpoint_physical_metadata_bytes_per_boundary": int(
                accounting["checkpoint_physical_metadata_bytes_per_boundary"]
            ),
            "checkpoint_representation": str(accounting["checkpoint_representation"]),
            "checkpoint_physical_object_count_upper_bound": int(
                accounting["checkpoint_physical_object_count_upper_bound"]
            ),
            "event_bytes_upper_bound": int(accounting["event_bytes_upper_bound"]),
            "durable_noncheckpoint_bytes_upper_bound": int(
                accounting["durable_noncheckpoint_bytes_upper_bound"]
            ),
            "fixed_output_bytes_upper_bound": int(
                accounting["fixed_output_bytes_upper_bound"]
            ),
            "disk_bytes_upper_bound": int(accounting["disk_bytes_upper_bound"]),
            "total_durable_output_checkpoint_bytes_upper_bound": int(
                accounting["disk_bytes_upper_bound"]
            ),
            "resource_estimator_id": str(accounting["resource_estimator_id"]),
            "resource_accounting_binding_sha256": str(
                accounting["resource_accounting_binding_sha256"]
            ),
            "artifact_binary_sha256": str(accounting["artifact_binary_sha256"]),
            "artifact_manifest_sha256": str(accounting["artifact_manifest_sha256"]),
            "device_selection_receipt_sha256": str(
                accounting["device_selection_receipt_sha256"]
            ),
            "device_identity_sha256": str(accounting["device_identity_sha256"]),
            "event_transport": str(accounting["event_transport"]),
            "horizon_dependent_resident_bytes": 0,
            **(
                {
                    "gpu_resource_policy_id": accounting[
                        "gpu_resource_policy_id"
                    ],
                    "vram_bytes_upper_bound": int(
                        accounting["vram_bytes_upper_bound"]
                    ),
                    "wall_seconds_upper_bound": int(
                        accounting["wall_seconds_upper_bound"]
                    ),
                    "serial_physical_batch_execution": True,
                }
                if shared_policy
                else {}
            ),
        }
    )


def _resolve_profile(
    spec: Mapping[str, Any], registry: ProfileRegistryV1
) -> RuntimeProfileV1:
    profile, reasons = registry.resolve(_requirements(spec))
    if profile is None:
        compact = [
            {"code": row.code, "path": row.path, "profile_id": row.profile_id}
            for row in reasons
        ]
        raise CampaignError(
            f"no runtime profile satisfies campaign capabilities: {compact}"
        )
    return profile


def resolve_campaign_profile(
    value: Mapping[str, Any], *, registry: ProfileRegistryV1 | None = None
) -> RuntimeProfileV1:
    """Resolve and return the campaign-specialized runtime profile."""

    spec = validate_campaign_spec(value)
    resolved_registry = default_profile_registry() if registry is None else registry
    return _resolve_profile(spec, resolved_registry)


def compile_campaign(
    value: Mapping[str, Any],
    *,
    registry: ProfileRegistryV1 | None = None,
    planning_resource_accounting: PlanningResourceAccountingV1 | None = None,
) -> CampaignPlanV1:
    spec = validate_campaign_spec(value)
    registry = default_profile_registry() if registry is None else registry
    profile = _resolve_profile(spec, registry)
    tick_native = (
        spec["execution"].get("execution_route_identity")
        == TICK_NATIVE_EXECUTION_ROUTE_V1
    )
    if not tick_native and not profile.arbitrary_positive_generations:
        raise CampaignError(
            f"profile {profile.profile_id} lacks arbitrary positive generation support"
        )
    if not tick_native and spec["execution"]["generations"] >= 2 ** (
        profile.generation_counter_bits - 1
    ):
        raise CampaignError(
            "execution.generations exceeds the profile generation counter"
        )
    for dimension, parameter in spec["matrix"]["parameter_bindings"].items():
        contract = profile.parameter_contracts[parameter]
        if not contract.per_cell:
            raise CampaignError(
                f"matrix dimension {dimension!r} targets scalar-only parameter {parameter!r}"
            )

    cases = _compile_matrix(spec)
    parameter_bindings = spec["matrix"]["parameter_bindings"]
    seed_bindings = spec["matrix"]["seed_bindings"]
    cells: list[LogicalCellV1] = []
    for ordinal, case in enumerate(cases):
        dimensions = dict(case["assignments"])
        parameters = dict(spec["parameters"])
        seeds: dict[str, int] = {}
        for dimension, parameter in parameter_bindings.items():
            parameters[str(parameter)] = dimensions[dimension]
        for dimension, owner in seed_bindings.items():
            raw_seed = dimensions[dimension]
            if (
                isinstance(raw_seed, bool)
                or not isinstance(raw_seed, int)
                or raw_seed < 0
            ):
                raise CampaignError(
                    f"matrix seed dimension {dimension!r} must contain nonnegative integers"
                )
            seeds[str(owner)] = raw_seed
        missing_seed_owners = sorted({"ecology", "task"} - set(seeds))
        if missing_seed_owners:
            raise CampaignError(
                f"matrix is missing seed ownership: {missing_seed_owners}"
            )
        for name, item in parameters.items():
            contract = profile.parameter_contracts.get(name)
            if contract is None or not contract.accepts(item):
                raise CampaignError(
                    f"cell {ordinal} parameter {name!r} violates its typed contract"
                )
        identity = {
            "campaign_id": spec["campaign_id"],
            "dimensions": dimensions,
            "parameters": {
                name: value
                for name, value in parameters.items()
                if name not in _SEMANTICALLY_NEUTRAL_CELL_ID_PARAMETERS_V1
            },
            "seeds": seeds,
        }
        digest = canonical_sha256(identity)
        cells.append(
            LogicalCellV1(
                ordinal=ordinal,
                cell_id=f"cell-{digest[:20]}",
                branch_id=f"branch-{digest[20:40]}",
                rng_lineage_id=f"rng-{digest[40:60]}",
                dimensions=dimensions,
                parameters=dict(sorted(parameters.items())),
                seeds=dict(sorted(seeds.items())),
            )
        )
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise CampaignError("compiled cell IDs collide")

    gate_selectors = spec["execution"]["technical_gate"]["selectors"]
    gate_cells: list[LogicalCellV1] = []
    for index, raw_selector in enumerate(gate_selectors):
        if not isinstance(raw_selector, Mapping) or not raw_selector:
            raise CampaignError(
                f"execution.technical_gate.selectors[{index}] must be a non-empty object"
            )
        unknown = sorted(set(raw_selector) - set(cells[0].dimensions))
        if unknown:
            raise CampaignError(
                "execution.technical_gate.selectors"
                f"[{index}] has unknown dimensions: {unknown}"
            )
        matches = [
            cell
            for cell in cells
            if all(
                cell.dimensions.get(name) == value
                for name, value in raw_selector.items()
            )
        ]
        if len(matches) != 1:
            raise CampaignError(
                "execution.technical_gate.selectors"
                f"[{index}] must match exactly one logical cell; matched {len(matches)}"
            )
        gate_cells.append(matches[0])
    if len({cell.cell_id for cell in gate_cells}) != len(gate_cells):
        raise CampaignError(
            "execution.technical_gate.selectors resolve duplicate logical cells"
        )

    batch_size = _batch_size(
        spec,
        profile_batch_max=profile.batch_max,
        qualified_batch_max=profile.qualified_batch_max,
        cells=len(cells),
    )
    gate_partition = bool(spec["execution"]["technical_gate"]["partition"])
    if gate_cells and gate_partition:
        gate_ids = {cell.cell_id for cell in gate_cells}
        staged = [
            *_partition_physical_batches(
                gate_cells,
                maximum_cells=batch_size,
                uniform_parameters=profile.uniform_batch_parameters,
            ),
            *_partition_physical_batches(
                [cell for cell in cells if cell.cell_id not in gate_ids],
                maximum_cells=batch_size,
                uniform_parameters=profile.uniform_batch_parameters,
            ),
        ]
        batches = tuple(
            replace(
                batch,
                ordinal=ordinal,
                batch_id=f"batch-{ordinal:04d}",
            )
            for ordinal, batch in enumerate(staged)
        )
    else:
        batches = _partition_physical_batches(
            cells,
            maximum_cells=batch_size,
            uniform_parameters=profile.uniform_batch_parameters,
        )
    # An explicit full-matrix request is a physical-shape contract, not a
    # ceiling.  Never silently split it because a uniform-parameter boundary
    # or technical-gate partition was introduced downstream.
    requested_batch = spec["execution"]["batch"]
    if (
        requested_batch["mode"] == "explicit"
        and requested_batch["cells"] == len(cells)
        and gate_partition is False
        and (
            len(batches) != 1
            or len(batches[0].cell_ids) != len(cells)
            or tuple(batches[0].cell_ids)
            != tuple(cell.cell_id for cell in cells)
        )
    ):
        raise CampaignError(
            "explicit full-matrix batch was partitioned; physical shape is not "
            "the requested single batch"
        )
    gpu_resource_policy: dict[str, Any] = {}
    gpu_resource_policy_selector = spec["execution"].get("gpu_resource_policy")
    if gpu_resource_policy_selector is not None:
        from .cuda_full_evolution_v2_shared_route import (
            resolve_shared_w32_gpu_policy_v1,
            validate_shared_w32_plan_v1,
        )

        try:
            gpu_resource_policy = resolve_shared_w32_gpu_policy_v1(
                gpu_resource_policy_selector
            )
            temporal_k_values = {
                cell.parameters.get("cuda_native.temporal_k") for cell in cells
            }
            if (
                len(temporal_k_values) != 1
                or type(next(iter(temporal_k_values))) is not int
            ):
                raise CampaignError(
                    "shared W32 campaign requires one exact integer temporal K"
                )
            validate_shared_w32_plan_v1(
                {
                    "resolved_profile": profile.profile_id,
                    "no_claims": spec["no_claims"],
                    "execution": {
                        "acs_checkpoint_slots": spec["checkpoints"].get(
                            "acs_slots", ACS_CHECKPOINT_SLOTS_DEFAULT_V1
                        ),
                        "gpu_resource_policy": gpu_resource_policy,
                    },
                    "physical_batches": [
                        {
                            "batch_id": batch.batch_id,
                            "cell_ids": list(batch.cell_ids),
                        }
                        for batch in batches
                    ],
                },
                temporal_k=int(next(iter(temporal_k_values))),
            )
        except ValueError as exc:
            raise CampaignError(str(exc)) from exc
    gate_execution_mode = (
        "staged_primary_batches"
        if gate_cells and gate_partition
        else "pre_primary_canary" if gate_cells else "disabled"
    )
    if gate_execution_mode == "pre_primary_canary":
        if (
            spec["execution"].get("execution_route_identity")
            == TICK_NATIVE_EXECUTION_ROUTE_V1
        ):
            native_gate = spec["execution"]["technical_gate"]
            gate_canary = {
                "batch_id": "technical-gate-canary",
                "restore_batch_id": "technical-gate-fresh-owner-restore",
                "time_authority": "absolute_tick_v1",
                "total_ticks": native_gate.get(
                    "total_ticks", spec["execution"]["total_ticks"]
                ),
                "checkpoint_ticks": native_gate.get("checkpoint_ticks", []),
                "cell_ids": [cell.cell_id for cell in gate_cells],
            }
        else:
            gate_canary = {
                "batch_id": "technical-gate-canary",
                "restore_batch_id": "technical-gate-fresh-owner-restore",
                "generations": spec["execution"]["technical_gate"].get(
                    "generations",
                    min(5, spec["execution"]["generations"]),
                ),
                "checkpoint_milestones": spec["execution"]["technical_gate"].get(
                    "checkpoint_milestones",
                    [
                        value
                        for value in (0, 5)
                        if value <= spec["execution"]["generations"]
                    ],
                ),
                "cell_ids": [cell.cell_id for cell in gate_cells],
            }
    else:
        gate_canary = None
    artifacts = spec["artifacts"]
    qualification_parameters = {
        name: value
        for name, value in spec["parameters"].items()
        if profile.parameter_contracts[name].qualification_affecting
    }
    for name, contract in sorted(profile.parameter_contracts.items()):
        if not contract.qualification_affecting or name in qualification_parameters:
            continue
        per_cell_values = {
            cell.parameters[name] for cell in cells if name in cell.parameters
        }
        if not per_cell_values:
            continue
        if len(per_cell_values) != 1:
            raise CampaignError(
                "qualification-affecting per-cell parameter "
                f"{name!r} varies across compiled cells; split the campaign into "
                "separately qualified route partitions"
            )
        qualification_parameters[name] = next(iter(per_cell_values))
    requested_wrc_parameters = {
        name for cell in cells for name in cell.parameters
    } | set(spec["parameters"])
    try:
        wrc_variability_envelope = build_wrc_parameter_variability_envelope_v1(
            profile,
            world_base=spec["world"]["base"],
            world_overlays=tuple(spec["world"]["overlays"]),
            requested_parameter_names=requested_wrc_parameters,
        )
    except WRCVariabilityEnvelopeError as exc:
        raise CampaignError(str(exc)) from exc
    wrc_variability_invariants = (
        {WRC_VARIABILITY_INVARIANT_NAME: wrc_variability_envelope.envelope_sha256}
        if wrc_variability_envelope is not None
        else {}
    )

    qualification = QualificationKeyV1(
        schema_version=QUALIFICATION_KEY_SCHEMA_VERSION,
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        backend=profile.backend,
        artifact_hashes=tuple(
            (name, str(row["sha256"])) for name, row in sorted(artifacts.items())
        ),
        observation_contract=profile.observation_contract,
        state_components=profile.state_components,
        checkpoint_components=profile.checkpoint_components,
        fixed_invariants=tuple(
            sorted(
                {
                    **profile.qualification_invariants,
                    **wrc_variability_invariants,
                    **{
                        f"parameter.{name}": value
                        for name, value in qualification_parameters.items()
                    },
                }.items()
            )
        ),
        admitted_batch_range=(
            profile.qualified_batch_min,
            profile.qualified_batch_max,
        ),
        variability_envelope=(
            wrc_variability_envelope.payload()
            if wrc_variability_envelope is not None
            else {}
        ),
    )
    qualification_payload = qualification.payload()
    fresh_repeats = int(spec["comparisons"]["fresh_repeats"])
    continuation_cells = int(spec["comparisons"]["continuation_cells"])
    continuation_baseline = str(spec["comparisons"]["continuation_baseline"])
    if fresh_repeats > len(cells) or continuation_cells > len(cells):
        raise CampaignError(
            "comparison cell counts must not exceed compiled logical cells"
        )

    def selected_cells(
        selector_rows: Sequence[Mapping[str, Scalar]],
        fallback_count: int,
        *,
        path: str,
    ) -> list[LogicalCellV1]:
        if not selector_rows:
            return cells[:fallback_count]
        selected: list[LogicalCellV1] = []
        for index, selector in enumerate(selector_rows):
            matches = [
                cell
                for cell in cells
                if all(
                    cell.dimensions.get(name) == value
                    for name, value in selector.items()
                )
            ]
            if len(matches) != 1:
                raise CampaignError(
                    f"{path}[{index}] must match exactly one logical cell; matched {len(matches)}"
                )
            selected.append(matches[0])
        if len({cell.cell_id for cell in selected}) != len(selected):
            raise CampaignError(f"{path} resolves duplicate logical cells")
        return selected

    fresh_cells = selected_cells(
        spec["comparisons"]["fresh_selectors"],
        fresh_repeats,
        path="comparisons.fresh_selectors",
    )
    continuation_source_cells = selected_cells(
        spec["comparisons"]["continuation_selectors"],
        continuation_cells,
        path="comparisons.continuation_selectors",
    )
    comparison_branches: list[ComparisonBranchV1] = []
    for cell in fresh_cells:
        identity = {
            "kind": "fresh_repeat",
            "source_cell_id": cell.cell_id,
            "repeat_ordinal": 1,
        }
        comparison_branches.append(
            ComparisonBranchV1(
                ordinal=len(comparison_branches),
                branch_id=f"repeat-{canonical_sha256(identity)[:20]}",
                kind="fresh_repeat",
                source_cell_id=cell.cell_id,
                repeat_ordinal=1,
                split_generation=None,
                baseline_mode=None,
            )
        )
    for split in spec["comparisons"]["continuation_splits"]:
        for cell in continuation_source_cells:
            identity = {
                "kind": "checkpoint_continuation",
                "source_cell_id": cell.cell_id,
                "split_generation": split,
            }
            comparison_branches.append(
                ComparisonBranchV1(
                    ordinal=len(comparison_branches),
                    branch_id=f"resume-{canonical_sha256(identity)[:20]}",
                    kind="checkpoint_continuation",
                    source_cell_id=cell.cell_id,
                    repeat_ordinal=None,
                    split_generation=int(split),
                    baseline_mode=continuation_baseline,
                )
            )
    if spec["comparisons"]["reporting_parity"] and not tick_native:
        reporting_cells = selected_cells(
            (
                [spec["comparisons"]["reporting_selector"]]
                if spec["comparisons"]["reporting_selector"] is not None
                else []
            ),
            1,
            path="comparisons.reporting_selector",
        )
        reporting_cell = reporting_cells[0]
        reporting_identity = {
            "kind": "reporting_parity",
            "source_cell_id": reporting_cell.cell_id,
            "reporting_enabled": False,
        }
        comparison_branches.append(
            ComparisonBranchV1(
                ordinal=len(comparison_branches),
                branch_id=f"reporting-{canonical_sha256(reporting_identity)[:20]}",
                kind="reporting_parity",
                source_cell_id=reporting_cell.cell_id,
                repeat_ordinal=None,
                split_generation=None,
                baseline_mode=None,
            )
        )
    estimate = _estimate(
        spec,
        cells=len(cells),
        batch_size=max(len(batch.cell_ids) for batch in batches),
    )
    if planning_resource_accounting is not None:
        if not callable(planning_resource_accounting):
            raise CampaignError("planning_resource_accounting must be callable")
        try:
            signed_accounting = planning_resource_accounting(
                spec=spec,
                logical_cells=len(cells),
                physical_batch_cells=max(len(batch.cell_ids) for batch in batches),
                compiled_batch_count=len(batches),
                comparison_branch_count=len(comparison_branches),
            )
        except Exception as exc:
            raise CampaignError(
                "signed provider planning resource accounting failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if signed_accounting is not None:
            apply_accounting = (
                _apply_signed_workhorse_storage_accounting_v2
                if signed_accounting.get("schema_version")
                == "eiel_workhorse_signed_planning_storage_accounting_v2"
                else _apply_signed_physical_checkpoint_accounting_v1
            )
            apply_accounting(
                estimate,
                signed_accounting,
                spec=spec,
                profile_id=profile.profile_id,
                logical_cells=len(cells),
                physical_batch_cells=max(len(batch.cell_ids) for batch in batches),
                compiled_batch_count=len(batches),
                comparison_branch_count=len(comparison_branches),
            )
    estimate.update(
        {
            "compiled_logical_cells": len(cells),
            "compiled_batch_count": len(batches),
            "compiled_physical_batch_cells_min": min(
                len(batch.cell_ids) for batch in batches
            ),
            "compiled_physical_batch_cells_max": max(
                len(batch.cell_ids) for batch in batches
            ),
            "compiled_physical_batch_cells_total": sum(
                len(batch.cell_ids) for batch in batches
            ),
            "compiled_partition": len(batches) != 1,
        }
    )
    if (
        int(estimate["disk_bytes_upper_bound"])
        > spec["execution"]["budgets"]["disk_bytes"]
    ):
        raise CampaignError(
            "planned disk upper bound exceeds execution.budgets.disk_bytes"
        )
    lifecycle = spec["artifact_lifecycle"]
    lifecycle_storage = lifecycle["storage"]
    if int(estimate["disk_bytes_upper_bound"]) > int(
        lifecycle_storage["total_max_bytes"]
    ):
        raise CampaignError(
            "planned disk upper bound exceeds artifact_lifecycle.storage.total_max_bytes"
        )
    if int(estimate["checkpoint_bytes_upper_bound"]) > int(
        lifecycle_storage["checkpoint_max_bytes"]
    ):
        raise CampaignError(
            "planned checkpoint upper bound exceeds artifact_lifecycle.storage.checkpoint_max_bytes"
        )
    if int(estimate.get("acs_pending_capture_file_bytes_upper_bound", 0)) > int(
        lifecycle_storage["scratch_max_bytes"]
    ):
        raise CampaignError(
            "planned ACS scratch upper bound exceeds artifact_lifecycle.storage.scratch_max_bytes"
        )
    if int(estimate.get("durable_noncheckpoint_bytes_upper_bound", 0)) > int(
        lifecycle_storage["durable_max_bytes"]
    ):
        raise CampaignError(
            "planned durable output upper bound exceeds artifact_lifecycle.storage.durable_max_bytes"
        )
    estimate.update(
        {
            "lifecycle_profile": lifecycle["lifecycle_profile"],
            "lifecycle_contract_sha256": lifecycle["contract_sha256"],
            "lifecycle_total_max_bytes": int(lifecycle_storage["total_max_bytes"]),
            "lifecycle_scratch_max_bytes": int(lifecycle_storage["scratch_max_bytes"]),
            "lifecycle_checkpoint_max_bytes": int(
                lifecycle_storage["checkpoint_max_bytes"]
            ),
            "lifecycle_durable_max_bytes": int(lifecycle_storage["durable_max_bytes"]),
            "lifecycle_emergency_reserve_bytes": int(
                lifecycle_storage["emergency_reserve_bytes"]
            ),
        }
    )
    if tick_native:
        private_adapter_ticks = None
        private_adapter_generations = None
    else:
        private_adapter_generations = int(spec["execution"]["generations"])
        private_adapter_ticks = int(spec["execution"]["ticks_per_generation"])
    plan = CampaignPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        campaign_id=spec["campaign_id"],
        description=spec["description"],
        no_claims=spec["no_claims"],
        profile_id=profile.profile_id,
        qualification_key=qualification_payload,
        qualification_key_sha256=canonical_sha256(qualification_payload),
        # Tick-native requests carry no generation-shaped shadow authority.
        generations=private_adapter_generations,
        ticks_per_generation=private_adapter_ticks,
        checkpoints=tuple(spec["checkpoints"]["milestones"]),
        digest_every_generation=spec["checkpoints"]["digest_every_generation"],
        checkpoint_storage_mode=spec["checkpoints"]["storage_mode"],
        acs_checkpoint_slots=spec["checkpoints"].get(
            "acs_slots", ACS_CHECKPOINT_SLOTS_DEFAULT_V1
        ),
        acs_capture_instrumentation=(
            spec["checkpoints"].get("acs_capture_instrumentation", False)
            or bool(
                spec["checkpoints"].get("acs_slots", ACS_CHECKPOINT_SLOTS_DEFAULT_V1)
            )
        ),
        native_streamed_checkpoint=spec["checkpoints"].get(
            "native_streamed_checkpoint", False
        ),
        world_sharded_checkpoint=spec["checkpoints"].get(
            "world_sharded_checkpoint", False
        ),
        budgets=spec["execution"]["budgets"],
        physical_window_generations=spec["execution"].get(
            "physical_window_generations"
        ),
        cells=tuple(cells),
        batches=batches,
        comparison_branches=tuple(comparison_branches),
        analysis=spec["analysis"],
        artifacts=artifacts,
        output=spec["output"],
        resource_estimate=estimate,
        artifact_lifecycle=lifecycle,
        tick_schedule=(
            AbsoluteTickScheduleV2(
                total_ticks=int(spec["execution"]["total_ticks"]),
                science_sample_interval_ticks=(
                    int(
                        spec["execution"]["tick_native_control"][
                            "science_sample_interval_ticks"
                        ]
                    )
                    if spec["execution"]["tick_native_control"].get(
                        "science_sample_interval_ticks"
                    )
                    is not None
                    else None
                ),
            ).payload()
            if spec["execution"].get("execution_route_identity")
            == TICK_NATIVE_EXECUTION_ROUTE_V1
            else TickScheduleV1.from_legacy(
                spec["execution"]["generations"],
                spec["execution"]["ticks_per_generation"],
            ).payload()
        ),
        execution_route_identity=spec["execution"].get("execution_route_identity"),
        total_ticks=spec["execution"].get("total_ticks"),
        start_next_tick=int(spec["execution"].get("start_next_tick") or 0),
        checkpoint_ticks=tuple(spec["execution"].get("checkpoint_ticks") or ()),
        deliberate_restore_tick=spec["execution"].get("deliberate_restore_tick"),
        deliberate_restore_ticks=tuple(
            spec["execution"].get("deliberate_restore_ticks") or ()
        ),
        internal_lifecycle_interval_ticks=None,
        max_internal_lifecycle_segments_per_dispatch=None,
        tick_native_control=(
            dict(spec["execution"].get("tick_native_control") or {})
            if spec["execution"].get("execution_route_identity")
            == TICK_NATIVE_EXECUTION_ROUTE_V1
            else {}
        ),
        gpu_resource_policy=gpu_resource_policy,
        technical_gate={
            "required": bool(gate_cells),
            "partition": gate_partition,
            "execution_mode": gate_execution_mode,
            "selectors": [dict(row) for row in gate_selectors],
            **(
                {"profile": spec["execution"]["technical_gate"]["profile"]}
                if "profile" in spec["execution"]["technical_gate"]
                else {}
            ),
            "cell_ids": [cell.cell_id for cell in gate_cells],
            "batch_ids": [
                batch.batch_id
                for batch in batches
                if any(
                    cell_id in {cell.cell_id for cell in gate_cells}
                    for cell_id in batch.cell_ids
                )
            ],
            "canary": gate_canary,
        },
    )
    requested_run_contract = spec.get("run_contract")
    if requested_run_contract is not None:
        from .run_contract_v1 import resolve_run_contract_v1

        plan = replace(
            plan,
            run_contract=resolve_run_contract_v1(
                plan.payload(include_hash=False),
                requested_run_contract,
            ),
        )
    if plan.execution_route_identity == TICK_NATIVE_EXECUTION_ROUTE_V1:
        try:
            assert_tick_native_plan_payload_v1(plan.payload(include_hash=False))
        except TickNativeControlError as exc:
            raise CampaignError(str(exc)) from exc
    return replace(plan, plan_sha256=canonical_sha256(plan.payload(include_hash=False)))


__all__ = [
    "CampaignError",
    "PLAN_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "compile_campaign",
    "load_campaign_spec",
    "resolve_campaign_profile",
    "validate_campaign_spec",
]
