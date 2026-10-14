"""Strict data model for versioned qualification matrices.

Qualification plans describe cases and their relationships. They contain no
commands, executable paths, environment variables, or production-routing
decisions, and this module deliberately provides no executor.
"""

from __future__ import annotations

import hashlib
import heapq
from itertools import product
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

SCHEMA_VERSION = "eiel_qualification_plan_v1"
MATRIX_SCHEMA_VERSION = "eiel_qualification_matrix_v1"
DEFAULT_MAX_MATRIX_CASES = 10_000
EXECUTION_ROUTE_CANARY_SCHEMA_VERSION = "eiel_execution_route_canary_plan_v1"
EXECUTION_ROUTE_OBLIGATIONS = frozenset(
    {
        "cold_start",
        "first_generation",
        "first_tick",
        "intra_generation",
        "generation_boundary",
        "first_post_boundary_tick",
        "second_post_boundary_tick",
        "post_boundary_policy_cache_refill",
        "ordinary_birth",
        "ordinary_birth_disabled",
        "turnover_refill",
        "multicell_execution",
        "odd_physical_tail",
        "checkpoint_export",
        "disk_checkpoint_reload",
        "fresh_owner_restore",
        "first_post_restore_tick",
        "first_post_restore_refill",
        "terminal_settlement",
        "reporting_parity",
        "alternate_batch_partition",
        "non_generation_divisible_terminal_tick",
        "non_generation_divisible_checkpoint_tick",
        "non_lattice_divisible_terminal_tick",
        "non_lattice_divisible_checkpoint_tick",
        "exact_absolute_tick_checkpoint_export",
        "fresh_owner_partial_generation_restore",
        "fresh_owner_partial_tick_restore",
        "first_committed_tick_after_partial_restore",
        "acs_capture_barrier_measurement",
        "async_checkpoint_backpressure",
        "async_checkpoint_failure_latch",
        "acs2_host_only_capture",
        "durable_frontier_publication",
        "async_compact_science_backpressure",
        "async_compact_science_failure_latch",
        "compact_science_host_only_capture",
        "compact_science_checkpoint_prefix_drain",
        "one_exact_scheduler_attempt",
        "live_exclusive_device_heartbeat_before_cuda",
        "absolute_tick_terminal_frontiers",
        "scheduler_reopened_terminal_before_final_receipt",
    }
)
_BOUNDARY_OBLIGATIONS = frozenset(
    {
        "generation_boundary",
        "first_post_boundary_tick",
        "second_post_boundary_tick",
        "post_boundary_policy_cache_refill",
        "turnover_refill",
    }
)
_CONTINUATION_OBLIGATIONS = frozenset(
    {
        "checkpoint_export",
        "disk_checkpoint_reload",
        "fresh_owner_restore",
        "first_post_restore_tick",
        "first_post_restore_refill",
    }
)
_LEGACY_TICK_NATIVE_OBLIGATION_ALIASES = frozenset(
    {
        "non_generation_divisible_terminal_tick",
        "non_generation_divisible_checkpoint_tick",
        "fresh_owner_partial_generation_restore",
    }
)
_CANONICAL_TICK_NATIVE_OBLIGATIONS = frozenset(
    {
        "non_lattice_divisible_terminal_tick",
        "non_lattice_divisible_checkpoint_tick",
        "exact_absolute_tick_checkpoint_export",
        "fresh_owner_partial_tick_restore",
        "first_committed_tick_after_partial_restore",
    }
)
_TICK_NATIVE_OBLIGATIONS = (
    _LEGACY_TICK_NATIVE_OBLIGATION_ALIASES | _CANONICAL_TICK_NATIVE_OBLIGATIONS
)
_ID_RE = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*\Z")
_TOP_KEYS = frozenset(
    {
        "schema_version",
        "plan_id",
        "description",
        "dimensions",
        "capability_requirements",
        "cases",
    }
)
_DIMENSION_KEYS = frozenset({"id", "description", "values"})
_CASE_KEYS = frozenset(
    {"id", "description", "assignments", "prerequisites", "capability_requirements"}
)
_MATRIX_KEYS = frozenset(
    {
        "schema_version",
        "plan_id",
        "description",
        "dimensions",
        "capability_requirements",
        "case_id_prefix",
        "include",
        "exclude",
        "templates",
        "max_cases",
    }
)
_TEMPLATE_KEYS = frozenset(
    {"selector", "description", "prerequisites", "capability_requirements"}
)


class QualificationPlanError(RuntimeError):
    """A qualification plan does not satisfy the v1 data contract."""


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QualificationPlanError(f"{name} must be a positive integer")
    return value


def _route_canary_settings(route_contract: Mapping[str, Any]) -> dict[str, Any]:
    raw = route_contract.get("canary_requirements", {})
    if not isinstance(raw, Mapping):
        raise QualificationPlanError(
            "execution route canary_requirements must be an object"
        )
    if raw.get("time_authority") == "absolute_tick_v1":
        total_ticks = _positive_integer(raw.get("total_ticks"), name="total_ticks")
        checkpoints = raw.get("checkpoint_ticks", [])
        if not isinstance(checkpoints, (list, tuple)) or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value > total_ticks
            for value in checkpoints
        ):
            raise QualificationPlanError(
                "checkpoint_ticks must contain positive ticks inside total_ticks"
            )
        raw_batches = raw.get(
            "admitted_physical_batch_sizes",
            route_contract.get("admitted_physical_batch_sizes", [1]),
        )
        if (
            not isinstance(raw_batches, (list, tuple))
            or not raw_batches
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in raw_batches
            )
        ):
            raise QualificationPlanError(
                "admitted_physical_batch_sizes must contain positive integers"
            )
        return {
            "tick_native": True,
            "total_ticks": total_ticks,
            "checkpoint_ticks": sorted(set(checkpoints) | {total_ticks}),
            "resident_owner_span_ticks": _positive_integer(
                (
                    1_120
                    if raw.get("resident_owner_span_ticks") is None
                    else raw.get("resident_owner_span_ticks")
                ),
                name="resident_owner_span_ticks",
            ),
            "native_dispatch_window_ticks": _positive_integer(
                (
                    1_120
                    if raw.get("native_dispatch_window_ticks") is None
                    else raw.get("native_dispatch_window_ticks")
                ),
                name="native_dispatch_window_ticks",
            ),
            "event_window_capacity": _positive_integer(
                (
                    65_536
                    if raw.get("event_window_capacity") is None
                    else raw.get("event_window_capacity")
                ),
                name="event_window_capacity",
            ),
            "admitted_physical_batch_sizes": sorted(set(raw_batches)),
        }
    ticks = raw.get(
        "qualified_ticks_per_generation",
        route_contract.get("qualified_ticks_per_generation"),
    )
    ticks = _positive_integer(ticks, name="qualified_ticks_per_generation")
    raw_batches = raw.get(
        "admitted_physical_batch_sizes",
        route_contract.get("admitted_physical_batch_sizes", [1]),
    )
    if (
        not isinstance(raw_batches, (list, tuple))
        or not raw_batches
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in raw_batches
        )
    ):
        raise QualificationPlanError(
            "admitted_physical_batch_sizes must contain positive integers"
        )
    batches = sorted(set(raw_batches))
    boundary_generations = _positive_integer(
        raw.get("boundary_generations", 2), name="boundary_generations"
    )
    return {
        "tick_native": False,
        "ticks_per_generation": ticks,
        "admitted_physical_batch_sizes": batches,
        "boundary_generations": boundary_generations,
        "resident_segment_generation_min": _positive_integer(
            raw.get("resident_segment_generation_min", 1),
            name="resident_segment_generation_min",
        ),
        "odd_tail_generations": _positive_integer(
            raw.get("odd_tail_generations", 3), name="odd_tail_generations"
        ),
    }


def _generate_tick_native_execution_route_canary_v1(
    *,
    route_sha256: str,
    route_contract_payload: Mapping[str, Any],
    required_set: set[str],
    selected: set[str],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an absolute-tick canary without generation-shaped assignments."""

    batches = list(settings["admitted_physical_batch_sizes"])
    needs_b2 = bool(selected & {"multicell_execution", "alternate_batch_partition"})
    if needs_b2 and 2 not in batches:
        raise QualificationPlanError(
            "route obligations require physical B2, but B2 is not admitted"
        )
    physical_batch_cells = 2 if needs_b2 else min(batches)
    rows: list[dict[str, Any]] = []

    def add_case(
        case_id: str,
        *,
        mode: str,
        repeat: int = 1,
        total_ticks: int = 161,
        checkpoint_ticks: str = "17,160,161",
        deliberate_restore_tick: int | str = "none",
        reporting_enabled: bool = False,
        report_every_ticks: int | str = "none",
        partition: str = "default",
        prerequisites: list[str] | None = None,
        obligations: set[str] | None = None,
    ) -> None:
        rows.append(
            {
                "id": case_id,
                "description": f"Absolute-tick execution-route canary {mode} branch.",
                "assignments": {
                    "checkpoint_ticks": checkpoint_ticks,
                    "deliberate_restore_tick": deliberate_restore_tick,
                    "mode": mode,
                    "partition": partition,
                    "physical_batch_cells": physical_batch_cells,
                    "repeat": repeat,
                    "report_every_ticks": report_every_ticks,
                    "reporting_enabled": reporting_enabled,
                    "time_authority": "absolute_tick_v1",
                    "total_ticks": total_ticks,
                },
                "prerequisites": prerequisites or [],
                "capability_requirements": sorted(obligations or set()),
            }
        )

    specialized = (
        _CONTINUATION_OBLIGATIONS
        | _TICK_NATIVE_OBLIGATIONS
        | {"reporting_parity", "alternate_batch_partition"}
    )
    base_obligations = selected - specialized
    if base_obligations:
        add_case("fresh-repeat-a", mode="fresh", obligations=base_obligations)
        add_case(
            "fresh-repeat-b",
            mode="fresh",
            repeat=2,
            prerequisites=["fresh-repeat-a"],
            obligations=base_obligations,
        )
    continuation = selected & _CONTINUATION_OBLIGATIONS
    if continuation:
        add_case(
            "uninterrupted",
            mode="uninterrupted",
            total_ticks=1121,
            checkpoint_ticks="500,1121",
        )
        add_case(
            "disk-continuation",
            mode="disk-continuation",
            total_ticks=1121,
            checkpoint_ticks="500,1121",
            deliberate_restore_tick=500,
            prerequisites=["uninterrupted"],
            obligations=continuation,
        )
    if "reporting_parity" in selected:
        add_case("reporting-disabled", mode="reporting")
        add_case(
            "reporting-enabled",
            mode="reporting",
            reporting_enabled=True,
            report_every_ticks=100,
            prerequisites=["reporting-disabled"],
            obligations={"reporting_parity"},
        )
    if "alternate_batch_partition" in selected:
        add_case("partition-default", mode="partition")
        add_case(
            "partition-alternate",
            mode="partition",
            partition="alternate",
            prerequisites=["partition-default"],
            obligations={"alternate_batch_partition"},
        )
    tick_obligations = selected & _TICK_NATIVE_OBLIGATIONS
    if tick_obligations:
        add_case(
            "tick-native-partial-restore",
            mode="tick-native-continuation",
            deliberate_restore_tick=17,
            obligations=tick_obligations,
        )
    if not rows:
        add_case("tick-native-fresh", mode="fresh", obligations=selected)

    dimensions = []
    for dimension_id in sorted(rows[0]["assignments"]):
        values_by_key = {
            _value_key(row["assignments"][dimension_id]): row["assignments"][
                dimension_id
            ]
            for row in rows
        }
        dimensions.append(
            {
                "id": dimension_id,
                "description": "",
                "values": [values_by_key[key] for key in sorted(values_by_key)],
            }
        )
    plan = validate_plan(
        {
            "schema_version": SCHEMA_VERSION,
            "plan_id": f"route-canary-{route_sha256[:16]}",
            "description": f"Absolute-tick canary for execution route {route_sha256}.",
            "dimensions": dimensions,
            "capability_requirements": [],
            "cases": rows,
        }
    )
    case_ticks = {row["id"]: row["assignments"]["total_ticks"] for row in rows}
    return {
        "schema_version": EXECUTION_ROUTE_CANARY_SCHEMA_VERSION,
        "execution_route_sha256": route_sha256,
        "route_contract": dict(route_contract_payload),
        "required_obligations": sorted(required_set),
        "missing_obligations": sorted(selected),
        "canary_requirements": {
            "time_authority": "absolute_tick_v1",
            "admitted_physical_batch_sizes": batches,
            "resident_owner_span_ticks": settings["resident_owner_span_ticks"],
            "native_dispatch_window_ticks": settings["native_dispatch_window_ticks"],
            "event_window_capacity": settings["event_window_capacity"],
        },
        "shape": {
            "time_authority": "absolute_tick_v1",
            "physical_batch_cells": physical_batch_cells,
            "total_ticks": max(case_ticks.values()),
            "max_physical_batch_cells": physical_batch_cells,
            "max_total_ticks": max(case_ticks.values()),
            "case_total_ticks": case_ticks,
        },
        "qualification_plan": plan,
    }


def generate_execution_route_canary_plan(
    route_contract: Mapping[str, Any],
    *,
    missing_obligations: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build the smallest declarative canary plan for missing route evidence.

    The returned qualification plan contains no executable path or command.
    Providers remain responsible for translating its cases into authoritative
    campaign requests.
    """

    if not isinstance(route_contract, Mapping):
        raise QualificationPlanError("execution route contract must be an object")
    route_sha256 = route_contract.get(
        "execution_route_sha256", route_contract.get("route_sha256")
    )
    if (
        not isinstance(route_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", route_sha256) is None
    ):
        raise QualificationPlanError(
            "execution route contract must contain a lowercase SHA-256 route hash"
        )
    required = route_contract.get("required_obligations")
    if not isinstance(required, (list, tuple)) or not required:
        raise QualificationPlanError("required_obligations must be a non-empty array")
    required_set = set(required)
    if not all(isinstance(item, str) for item in required):
        raise QualificationPlanError("required_obligations must contain strings")
    unknown = sorted(required_set - EXECUTION_ROUTE_OBLIGATIONS)
    if unknown:
        raise QualificationPlanError(f"unsupported route obligations: {unknown}")
    selected = required_set if missing_obligations is None else set(missing_obligations)
    if not all(isinstance(item, str) for item in selected):
        raise QualificationPlanError("missing_obligations must contain strings")
    unknown_missing = sorted(selected - required_set)
    if unknown_missing:
        raise QualificationPlanError(
            f"missing obligations are not required by the route: {unknown_missing}"
        )
    if not selected:
        raise QualificationPlanError("no missing execution-route obligations")
    tick_native_selected = selected & _TICK_NATIVE_OBLIGATIONS
    if tick_native_selected:
        components = route_contract.get("components")
        if not isinstance(components, (list, tuple)) or not any(
            isinstance(component, Mapping)
            and component.get("component_id") == "absolute_tick_time_contract"
            for component in components
        ):
            raise QualificationPlanError(
                "tick-native obligations require the absolute-tick route component"
            )

    route_contract_payload = json.loads(
        json.dumps(
            route_contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    settings = _route_canary_settings(route_contract)
    if settings["tick_native"]:
        return _generate_tick_native_execution_route_canary_v1(
            route_sha256=route_sha256,
            route_contract_payload=route_contract_payload,
            required_set=required_set,
            selected=selected,
            settings=settings,
        )
    generations = settings["resident_segment_generation_min"]
    if selected & (_BOUNDARY_OBLIGATIONS | _CONTINUATION_OBLIGATIONS):
        generations = max(generations, settings["boundary_generations"])
    if "odd_physical_tail" in selected:
        generations = max(generations, settings["odd_tail_generations"])
        if generations % 2 == 0:
            generations += 1
    batches = settings["admitted_physical_batch_sizes"]
    needs_b2 = bool(selected & {"multicell_execution", "alternate_batch_partition"})
    if needs_b2 and 2 not in batches:
        raise QualificationPlanError(
            "route obligations require physical B2, but B2 is not admitted"
        )
    physical_batch_cells = 2 if needs_b2 else min(batches)
    ticks = settings["ticks_per_generation"]
    if "second_post_boundary_tick" in selected and ticks < 2:
        raise QualificationPlanError(
            "second_post_boundary_tick requires at least two qualified ticks"
        )

    rows: list[dict[str, Any]] = []

    def add_case(
        case_id: str,
        *,
        mode: str,
        repeat: int,
        case_generations: int | None = None,
        continuation_split_generation: int | str = "none",
        reporting: bool = False,
        partition: str = "default",
        prerequisites: list[str] | None = None,
        obligations: set[str] | None = None,
        tick_native: bool = False,
    ) -> None:
        assignments: dict[str, Any] = {
            "continuation_split_generation": continuation_split_generation,
            "generations": (
                generations if case_generations is None else case_generations
            ),
            "mode": mode,
            "partition": partition,
            "physical_batch_cells": physical_batch_cells,
            "repeat": repeat,
            "reporting_enabled": reporting,
            "ticks_per_generation": ticks,
        }
        if tick_native_selected:
            assignments.update(
                {
                    "checkpoint_ticks": ("17,160,161" if tick_native else "none"),
                    "deliberate_restore_tick": (17 if tick_native else "none"),
                    "total_ticks": (161 if tick_native else "none"),
                }
            )
        rows.append(
            {
                "id": case_id,
                "description": f"Standard execution-route canary {mode} branch.",
                "assignments": assignments,
                "prerequisites": prerequisites or [],
                "capability_requirements": sorted(obligations or set()),
            }
        )

    specialized = (
        _CONTINUATION_OBLIGATIONS
        | _TICK_NATIVE_OBLIGATIONS
        | {"reporting_parity", "alternate_batch_partition"}
    )
    base_obligations = selected - specialized
    if base_obligations or not tick_native_selected:
        add_case("fresh-repeat-a", mode="fresh", repeat=1, obligations=base_obligations)
        add_case(
            "fresh-repeat-b",
            mode="fresh",
            repeat=2,
            prerequisites=["fresh-repeat-a"],
            obligations=base_obligations,
        )
    continuation = selected & _CONTINUATION_OBLIGATIONS
    if continuation:
        segment_min = settings["resident_segment_generation_min"]
        continuation_generations = max(generations, segment_min * 2)
        add_case(
            "uninterrupted",
            mode="uninterrupted",
            repeat=1,
            case_generations=continuation_generations,
            continuation_split_generation=segment_min,
        )
        add_case(
            "disk-continuation",
            mode="disk-continuation",
            repeat=1,
            case_generations=continuation_generations,
            continuation_split_generation=segment_min,
            prerequisites=["uninterrupted"],
            obligations=continuation,
        )
    if "reporting_parity" in selected:
        add_case("reporting-disabled", mode="reporting", repeat=1)
        add_case(
            "reporting-enabled",
            mode="reporting",
            repeat=1,
            reporting=True,
            prerequisites=["reporting-disabled"],
            obligations={"reporting_parity"},
        )
    if "alternate_batch_partition" in selected:
        add_case("partition-default", mode="partition", repeat=1)
        add_case(
            "partition-alternate",
            mode="partition",
            repeat=1,
            partition="alternate",
            prerequisites=["partition-default"],
            obligations={"alternate_batch_partition"},
        )
    if tick_native_selected:
        add_case(
            "tick-native-partial-restore",
            mode="tick-native-continuation",
            repeat=1,
            case_generations=2,
            obligations=set(tick_native_selected),
            tick_native=True,
        )

    dimensions = []
    for dimension_id in sorted(rows[0]["assignments"]):
        values_by_key = {
            _value_key(row["assignments"][dimension_id]): row["assignments"][
                dimension_id
            ]
            for row in rows
        }
        dimensions.append(
            {
                "id": dimension_id,
                "description": "",
                "values": [values_by_key[key] for key in sorted(values_by_key)],
            }
        )
    plan = validate_plan(
        {
            "schema_version": SCHEMA_VERSION,
            "plan_id": f"route-canary-{route_sha256[:16]}",
            "description": (
                f"Obligation-driven canary for execution route {route_sha256}."
            ),
            "dimensions": dimensions,
            "capability_requirements": [],
            "cases": rows,
        }
    )
    case_generations = {row["id"]: row["assignments"]["generations"] for row in rows}
    maximum_generations = max(case_generations.values())
    return {
        "schema_version": EXECUTION_ROUTE_CANARY_SCHEMA_VERSION,
        "execution_route_sha256": route_sha256,
        "route_contract": route_contract_payload,
        "required_obligations": sorted(required_set),
        "missing_obligations": sorted(selected),
        "canary_requirements": {
            "admitted_physical_batch_sizes": settings["admitted_physical_batch_sizes"],
            "boundary_generations": settings["boundary_generations"],
            "odd_tail_generations": settings["odd_tail_generations"],
            "qualified_ticks_per_generation": ticks,
            "resident_segment_generation_min": settings[
                "resident_segment_generation_min"
            ],
        },
        "shape": {
            "physical_batch_cells": physical_batch_cells,
            "generations": maximum_generations,
            "ticks_per_generation": ticks,
            "max_physical_batch_cells": physical_batch_cells,
            "max_generations": maximum_generations,
            "case_generations": case_generations,
        },
        "qualification_plan": plan,
    }


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects duplicate mapping keys."""


def _unique_yaml_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise QualificationPlanError(f"duplicate YAML field {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_yaml_mapping,
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationPlanError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise QualificationPlanError(f"non-finite JSON number {value!r} is not allowed")


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise QualificationPlanError(f"{name} must be an object with string keys")
    return dict(value)


def _exact_keys(
    value: Mapping[str, Any], allowed: frozenset[str], *, name: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise QualificationPlanError(f"{name} contains unsupported fields: {unknown}")


def _identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise QualificationPlanError(f"{name} must be a lowercase stable identifier")
    return value


def _description(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise QualificationPlanError(f"{name} must be a string")
    return value


def _scalar(value: Any, *, name: str) -> bool | int | float | str:
    if value is None or isinstance(value, (dict, list)):
        raise QualificationPlanError(
            f"{name} must be a string, boolean, or finite number"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise QualificationPlanError(f"{name} must be finite")
    if not isinstance(value, (bool, int, float, str)):
        raise QualificationPlanError(f"{name} has an unsupported value type")
    return value


def _value_key(value: bool | int | float | str) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _selector(
    value: Any,
    *,
    name: str,
    allowed_values: Mapping[str, set[str]],
) -> dict[str, bool | int | float | str]:
    selector = _object(value, name=name)
    if not selector:
        raise QualificationPlanError(f"{name} must not be empty")
    unknown = sorted(set(selector) - set(allowed_values))
    if unknown:
        raise QualificationPlanError(f"{name} contains unknown dimensions: {unknown}")
    canonical: dict[str, bool | int | float | str] = {}
    for dimension_id in sorted(selector):
        selected = _scalar(selector[dimension_id], name=f"{name}.{dimension_id}")
        if _value_key(selected) not in allowed_values[dimension_id]:
            raise QualificationPlanError(
                f"{name} selects an unsupported {dimension_id!r} value"
            )
        canonical[dimension_id] = selected
    return canonical


def _selector_list(
    value: Any,
    *,
    name: str,
    allowed_values: Mapping[str, set[str]],
) -> list[dict[str, bool | int | float | str]]:
    if not isinstance(value, list):
        raise QualificationPlanError(f"{name} must be an array")
    selectors = [
        _selector(item, name=f"{name}[{index}]", allowed_values=allowed_values)
        for index, item in enumerate(value)
    ]
    keys = [_value_key(selector) for selector in selectors]
    if len(keys) != len(set(keys)):
        raise QualificationPlanError(f"{name} must not contain duplicate selectors")
    return selectors


def _matches_selector(
    assignments: Mapping[str, Any], selector: Mapping[str, Any]
) -> bool:
    return all(
        _value_key(assignments[dimension_id]) == _value_key(selected)
        for dimension_id, selected in selector.items()
    )


def _identifiers(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list):
        raise QualificationPlanError(f"{name} must be an array")
    result = [
        _identifier(item, name=f"{name}[{index}]") for index, item in enumerate(value)
    ]
    if len(set(result)) != len(result):
        raise QualificationPlanError(f"{name} must not contain duplicates")
    return sorted(result)


def _validate_prerequisite_graph(cases: list[dict[str, Any]]) -> None:
    case_ids = {case["id"] for case in cases}
    graph = {case["id"]: case["prerequisites"] for case in cases}
    for case_id, prerequisites in graph.items():
        unknown = sorted(set(prerequisites) - case_ids)
        if unknown:
            raise QualificationPlanError(
                f"case {case_id!r} has unknown prerequisites: {unknown}"
            )
        if case_id in prerequisites:
            raise QualificationPlanError(f"case {case_id!r} may not require itself")

    remaining_prerequisites = {
        case_id: len(prerequisites) for case_id, prerequisites in graph.items()
    }
    dependents: dict[str, list[str]] = {case_id: [] for case_id in graph}
    for case_id, prerequisites in graph.items():
        for prerequisite in prerequisites:
            dependents[prerequisite].append(case_id)
    ready = sorted(
        case_id for case_id, count in remaining_prerequisites.items() if count == 0
    )
    heapq.heapify(ready)
    visited_count = 0
    while ready:
        case_id = heapq.heappop(ready)
        visited_count += 1
        for dependent in sorted(dependents[case_id]):
            remaining_prerequisites[dependent] -= 1
            if remaining_prerequisites[dependent] == 0:
                heapq.heappush(ready, dependent)
    if visited_count != len(graph):
        raise QualificationPlanError("case prerequisites contain a cycle")


def validate_plan(value: Any) -> dict[str, Any]:
    """Return a deterministic canonical plan or raise on unsupported data."""

    plan = _object(value, name="plan")
    _exact_keys(plan, _TOP_KEYS, name="plan")
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise QualificationPlanError(f"schema_version must be {SCHEMA_VERSION!r}")
    plan_id = _identifier(plan.get("plan_id"), name="plan_id")
    description = _description(plan.get("description", ""), name="description")
    capabilities = _identifiers(
        plan.get("capability_requirements", []), name="capability_requirements"
    )

    raw_dimensions = plan.get("dimensions")
    if not isinstance(raw_dimensions, list) or not raw_dimensions:
        raise QualificationPlanError("dimensions must be a non-empty array")
    dimensions: list[dict[str, Any]] = []
    allowed_values: dict[str, set[str]] = {}
    for index, raw in enumerate(raw_dimensions):
        dimension = _object(raw, name=f"dimensions[{index}]")
        _exact_keys(dimension, _DIMENSION_KEYS, name=f"dimensions[{index}]")
        dimension_id = _identifier(dimension.get("id"), name=f"dimensions[{index}].id")
        if dimension_id in allowed_values:
            raise QualificationPlanError(f"duplicate dimension id: {dimension_id!r}")
        raw_values = dimension.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise QualificationPlanError(
                f"dimensions[{index}].values must be non-empty"
            )
        values = [
            _scalar(item, name=f"dimensions[{index}].values[{value_index}]")
            for value_index, item in enumerate(raw_values)
        ]
        keys = [_value_key(item) for item in values]
        if len(set(keys)) != len(keys):
            raise QualificationPlanError(
                f"dimension {dimension_id!r} has duplicate values"
            )
        allowed_values[dimension_id] = set(keys)
        dimensions.append(
            {
                "id": dimension_id,
                "description": _description(
                    dimension.get("description", ""),
                    name=f"dimensions[{index}].description",
                ),
                "values": values,
            }
        )
    dimensions.sort(key=lambda item: item["id"])

    raw_cases = plan.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise QualificationPlanError("cases must be a non-empty array")
    cases: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    dimension_ids = set(allowed_values)
    for index, raw in enumerate(raw_cases):
        case = _object(raw, name=f"cases[{index}]")
        _exact_keys(case, _CASE_KEYS, name=f"cases[{index}]")
        case_id = _identifier(case.get("id"), name=f"cases[{index}].id")
        if case_id in seen_cases:
            raise QualificationPlanError(f"duplicate case id: {case_id!r}")
        seen_cases.add(case_id)
        assignments = _object(
            case.get("assignments"), name=f"cases[{index}].assignments"
        )
        missing = sorted(dimension_ids - set(assignments))
        extra = sorted(set(assignments) - dimension_ids)
        if missing or extra:
            raise QualificationPlanError(
                f"case {case_id!r} assignments mismatch dimensions; missing={missing}, extra={extra}"
            )
        canonical_assignments: dict[str, Any] = {}
        for dimension_id in sorted(assignments):
            assigned = _scalar(
                assignments[dimension_id],
                name=f"cases[{index}].assignments.{dimension_id}",
            )
            if _value_key(assigned) not in allowed_values[dimension_id]:
                raise QualificationPlanError(
                    f"case {case_id!r} assigns an unsupported {dimension_id!r} value"
                )
            canonical_assignments[dimension_id] = assigned
        cases.append(
            {
                "id": case_id,
                "description": _description(
                    case.get("description", ""), name=f"cases[{index}].description"
                ),
                "assignments": canonical_assignments,
                "prerequisites": _identifiers(
                    case.get("prerequisites", []),
                    name=f"cases[{index}].prerequisites",
                ),
                "capability_requirements": _identifiers(
                    case.get("capability_requirements", []),
                    name=f"cases[{index}].capability_requirements",
                ),
            }
        )
    _validate_prerequisite_graph(cases)
    cases.sort(key=lambda item: item["id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan_id,
        "description": description,
        "dimensions": dimensions,
        "capability_requirements": capabilities,
        "cases": cases,
    }


def compile_matrix(value: Any) -> dict[str, Any]:
    """Compile a strict declarative matrix into a canonical v1 plan.

    Every rule is an equality selector over declared dimension values. No
    selector text is parsed or evaluated as an expression.
    """

    matrix = _object(value, name="matrix")
    _exact_keys(matrix, _MATRIX_KEYS, name="matrix")
    if matrix.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise QualificationPlanError(
            f"matrix.schema_version must be {MATRIX_SCHEMA_VERSION!r}"
        )
    plan_id = _identifier(matrix.get("plan_id"), name="matrix.plan_id")
    case_id_prefix = _identifier(
        matrix.get("case_id_prefix"), name="matrix.case_id_prefix"
    )
    description = _description(matrix.get("description", ""), name="matrix.description")
    plan_capabilities = _identifiers(
        matrix.get("capability_requirements", []),
        name="matrix.capability_requirements",
    )
    raw_max_cases = matrix.get("max_cases", DEFAULT_MAX_MATRIX_CASES)
    if (
        isinstance(raw_max_cases, bool)
        or not isinstance(raw_max_cases, int)
        or raw_max_cases <= 0
    ):
        raise QualificationPlanError("matrix.max_cases must be a positive integer")
    max_cases = raw_max_cases

    raw_dimensions = matrix.get("dimensions")
    if not isinstance(raw_dimensions, list) or not raw_dimensions:
        raise QualificationPlanError("matrix.dimensions must be a non-empty array")
    dimensions: list[dict[str, Any]] = []
    allowed_values: dict[str, set[str]] = {}
    ordered_values: dict[str, list[bool | int | float | str]] = {}
    for index, raw in enumerate(raw_dimensions):
        dimension = _object(raw, name=f"matrix.dimensions[{index}]")
        _exact_keys(dimension, _DIMENSION_KEYS, name=f"matrix.dimensions[{index}]")
        dimension_id = _identifier(
            dimension.get("id"), name=f"matrix.dimensions[{index}].id"
        )
        if dimension_id in allowed_values:
            raise QualificationPlanError(f"duplicate dimension id: {dimension_id!r}")
        raw_values = dimension.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise QualificationPlanError(
                f"matrix.dimensions[{index}].values must be non-empty"
            )
        values = [
            _scalar(
                item,
                name=f"matrix.dimensions[{index}].values[{value_index}]",
            )
            for value_index, item in enumerate(raw_values)
        ]
        keyed_values = sorted((_value_key(item), item) for item in values)
        if len({key for key, _item in keyed_values}) != len(keyed_values):
            raise QualificationPlanError(
                f"dimension {dimension_id!r} has duplicate values"
            )
        allowed_values[dimension_id] = {key for key, _item in keyed_values}
        ordered_values[dimension_id] = [item for _key, item in keyed_values]
        dimensions.append(
            {
                "id": dimension_id,
                "description": _description(
                    dimension.get("description", ""),
                    name=f"matrix.dimensions[{index}].description",
                ),
                "values": [item for _key, item in keyed_values],
            }
        )
    dimensions.sort(key=lambda item: item["id"])
    dimension_ids = [dimension["id"] for dimension in dimensions]
    combination_count = math.prod(
        len(ordered_values[dimension_id]) for dimension_id in dimension_ids
    )
    include = _selector_list(
        matrix.get("include", []),
        name="matrix.include",
        allowed_values=allowed_values,
    )
    exclude = _selector_list(
        matrix.get("exclude", []),
        name="matrix.exclude",
        allowed_values=allowed_values,
    )
    assignments: list[dict[str, bool | int | float | str]] = []
    fully_specified_include = bool(include) and all(
        set(selector) == set(dimension_ids) for selector in include
    )
    if fully_specified_include:
        if len(include) > max_cases:
            raise QualificationPlanError(
                f"matrix fully specified include exceeds max_cases ({len(include)} > {max_cases})"
            )
        candidate_rows = include
    else:
        if combination_count > max_cases:
            raise QualificationPlanError(
                "matrix Cartesian product exceeds max_cases "
                f"({combination_count} > {max_cases})"
            )
        candidate_rows = (
            dict(zip(dimension_ids, combination))
            for combination in product(
                *(ordered_values[item] for item in dimension_ids)
            )
        )
    for row in candidate_rows:
        if (
            include
            and not fully_specified_include
            and not any(_matches_selector(row, selector) for selector in include)
        ):
            continue
        if any(_matches_selector(row, selector) for selector in exclude):
            continue
        assignments.append(dict(row))
    if not assignments:
        raise QualificationPlanError("matrix selectors produce no cases")

    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for row in assignments:
        digest = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        case_id = f"{case_id_prefix}-{digest}"
        if case_id in case_ids:
            raise QualificationPlanError("generated case IDs collide")
        case_ids.add(case_id)
        cases.append(
            {
                "id": case_id,
                "description": "",
                "assignments": row,
                "prerequisites": [],
                "capability_requirements": [],
            }
        )

    raw_templates = matrix.get("templates", [])
    if not isinstance(raw_templates, list):
        raise QualificationPlanError("matrix.templates must be an array")
    for index, raw in enumerate(raw_templates):
        template = _object(raw, name=f"matrix.templates[{index}]")
        _exact_keys(template, _TEMPLATE_KEYS, name=f"matrix.templates[{index}]")
        selector = _selector(
            template.get("selector"),
            name=f"matrix.templates[{index}].selector",
            allowed_values=allowed_values,
        )
        matched = [
            case for case in cases if _matches_selector(case["assignments"], selector)
        ]
        if not matched:
            raise QualificationPlanError(
                f"matrix.templates[{index}].selector matches no generated cases"
            )
        template_description = _description(
            template.get("description", ""),
            name=f"matrix.templates[{index}].description",
        )
        template_capabilities = _identifiers(
            template.get("capability_requirements", []),
            name=f"matrix.templates[{index}].capability_requirements",
        )
        prerequisite_selectors = _selector_list(
            template.get("prerequisites", []),
            name=f"matrix.templates[{index}].prerequisites",
            allowed_values=allowed_values,
        )
        prerequisites: list[str] = []
        for prerequisite_index, prerequisite_selector in enumerate(
            prerequisite_selectors
        ):
            prerequisite_matches = [
                case
                for case in cases
                if _matches_selector(case["assignments"], prerequisite_selector)
            ]
            if len(prerequisite_matches) != 1:
                raise QualificationPlanError(
                    f"matrix.templates[{index}].prerequisites[{prerequisite_index}] "
                    "must match exactly one generated case"
                )
            prerequisites.append(prerequisite_matches[0]["id"])
        for case in matched:
            if (
                template_description
                and case["description"]
                and case["description"] != template_description
            ):
                raise QualificationPlanError(
                    f"conflicting template descriptions for case {case['id']!r}"
                )
            if template_description:
                case["description"] = template_description
            case["capability_requirements"] = sorted(
                set(case["capability_requirements"]) | set(template_capabilities)
            )
            case["prerequisites"] = sorted(
                set(case["prerequisites"]) | set(prerequisites)
            )

    return validate_plan(
        {
            "schema_version": SCHEMA_VERSION,
            "plan_id": plan_id,
            "description": description,
            "dimensions": dimensions,
            "capability_requirements": plan_capabilities,
            "cases": cases,
        }
    )


def _load_document(path: str | Path) -> Any:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise QualificationPlanError(f"cannot read plan: {source}") from exc
    try:
        if source.suffix.lower() == ".json":
            value = json.loads(
                text,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        elif source.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.load(text, Loader=_UniqueKeySafeLoader)
        else:
            raise QualificationPlanError(
                "definition filename must end in .json, .yaml, or .yml"
            )
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise QualificationPlanError(
            f"definition is not valid {source.suffix.lower()[1:].upper()}"
        ) from exc
    return value


def load_document(path: str | Path) -> Any:
    """Load strict JSON or YAML without imposing a qualification-plan schema."""

    return _load_document(path)


def load_plan(path: str | Path) -> dict[str, Any]:
    """Load strict JSON or YAML and return its canonical plan."""

    value = _load_document(path)
    return validate_plan(value)


def load_matrix(path: str | Path) -> dict[str, Any]:
    """Load and compile a strict versioned JSON/YAML matrix definition."""

    value = _load_document(path)
    return compile_matrix(value)
