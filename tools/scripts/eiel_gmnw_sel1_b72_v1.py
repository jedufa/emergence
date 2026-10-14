#!/usr/bin/env python3
"""Freeze and validate the preregistered GMNW SEL1 b72g12 harness."""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, fields
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Final, Iterator, Mapping, Sequence

import torch


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ENGINE: Final[Path] = ROOT / "engine"
for _path in (ROOT, ENGINE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SCHEMA_VERSION: Final[str] = "eiel_gmnw_sel1_b72_contract_v1"
RESULT_SCHEMA_VERSION: Final[str] = "eiel_gmnw_sel1_cell_result_v1"
ANALYSIS_SCHEMA_VERSION: Final[str] = "eiel_gmnw_sel1_analysis_v1"
POLICY_ARM: Final[str] = "mixed_selection"
FOUNDER_COUNTS: Final[tuple[int, ...]] = (0, 4, 8, 16)
SEEDS: Final[tuple[int, ...]] = tuple(range(101, 910, 101))
CONTEXTS: Final[tuple[str, ...]] = ("delayed", "immediate")
CONTEXT_CODES: Final[Mapping[str, str]] = {"delayed": "d", "immediate": "i"}
CUE_VISIBLE_AT_DECISION: Final[Mapping[str, bool]] = {
    "delayed": False,
    "immediate": True,
}
BATCH_CELLS: Final[int] = 72
INITIAL_POPULATION: Final[int] = 64
GENERATIONS: Final[int] = 12
TICKS_PER_GENERATION: Final[int] = 80
PAIR_COUNT: Final[int] = 36
CUE_ALPHABET: Final[int] = 4
DELAY_TICKS: Final[int] = 12
DISTRACTORS_ENABLED: Final[bool] = True
GRACE_GENERATION_COUNT: Final[int] = 4
SEVERE_GENERATION_COUNT: Final[int] = 4
GRACE_DAMAGE_FRACTION: Final[float] = 0.01
SEVERE_DAMAGE_FRACTION: Final[float] = 0.01
LETHAL_DAMAGE_FRACTION: Final[float] = 1.0
GRACE_REPRODUCTION_SCALE: Final[float] = 0.5
SOURCE_PROJECTION_SUFFIX: Final[str] = ".pt"
REPEAT_CELL_IDS: Final[tuple[str, ...]] = (
    "sel1_f00_s101_d",
    "sel1_f00_s101_i",
    "sel1_f04_s202_d",
    "sel1_f04_s202_i",
    "sel1_f08_s505_d",
    "sel1_f16_s909_d",
)
CONTINUATION_CELL_IDS: Final[tuple[str, ...]] = (
    "sel1_f08_s505_d",
    "sel1_f08_s505_i",
)
POLICY_FAMILIES: Final[tuple[str, ...]] = ("mlp", "gru")
COUNT_METRIC_NAMES: Final[tuple[str, ...]] = (
    "alive_pre_turnover",
    "alive_post_turnover",
    "decision_count",
    "correct_count",
    "task_death_count",
    "unique_selected_parent_pool_count",
    "committed_turnover_offspring_count",
)
COST_METRIC_NAME: Final[str] = "recurrent_upkeep_charge_total"
NATIVE_METRIC_NAMES: Final[tuple[str, ...]] = (
    *COUNT_METRIC_NAMES,
    COST_METRIC_NAME,
)
DERIVED_METRIC_NAMES: Final[tuple[str, ...]] = (
    "decision_accuracy",
    "alive_pre_turnover_share",
    "alive_post_turnover_share",
    "task_death_share",
    "task_death_rate",
    "selected_parent_pool_share",
    "selected_parent_pool_rate",
    "committed_turnover_offspring_share",
    "recurrent_upkeep_charge_share",
    "recurrent_upkeep_charge_per_alive_pre_turnover",
)
PENDING_NATIVE_MEASUREMENTS: Final[tuple[str, ...]] = (
    "ordinary_birth_count_by_family",
    "failed_birth_count_by_family",
    "reproduction_eligibility_by_family",
    "descendant_contribution_by_family",
    "founder_ancestry_fraction",
    "effective_lineage_count",
    "top_one_ancestry_concentration",
    "lineage_extinction_generation",
    "policy_id_integrity",
    "founder_reinsertion_integrity",
)

# Frozen before scientific execution. Counts are evaluated over all nine paired seeds.
THRESHOLDS: Final[Mapping[str, int | float | bool]] = {
    "required_completed_cells": 72,
    "required_complete_pairs": 36,
    "maximum_failed_cells": 0,
    "required_completed_generations_per_cell": 12,
    "required_repeat_cells": 6,
    "required_continuation_cells": 2,
    "require_exact_repeat_state_digest": True,
    "require_exact_continuation_state_digest": True,
    "chance_accuracy": 0.25,
    "minimum_gru_delayed_accuracy": 0.90,
    "maximum_mlp_delayed_accuracy": 0.35,
    "minimum_gru_immediate_accuracy": 0.90,
    "minimum_mlp_immediate_accuracy": 0.90,
    "require_zero_gru_controls_exact": True,
    "require_positive_recurrent_cost": True,
    "maximum_zero_founder_final_gru_fraction": 0.0,
    "minimum_delayed_minus_immediate_gru_fraction": 0.25,
    "minimum_delayed_final_minus_initial_gru_fraction": 0.25,
    "minimum_supporting_seed_pairs_per_founder_count": 7,
    "minimum_parent_advantage_seeds_per_founder_count": 7,
    "minimum_offspring_advantage_seeds_per_founder_count": 7,
    "minimum_supporting_nonzero_founder_counts": 2,
    "maximum_immediate_gru_fraction_increase": 0.10,
    "maximum_immediate_advantage_seeds_per_founder_count": 5,
}
A_GATE_RECEIPT_NAMES: Final[tuple[str, ...]] = (
    "immediate_control_calibration",
    "fork_identity",
    "b72_selection_envelope_admission",
    "policy_id_integrity",
    "no_policy_mutation",
    "founder_reinsertion_integrity",
    "rng_integrity",
    "physical_cell_identity",
    "causal_ordering_integrity",
    "stage5_exact_repeat",
    "stage5_exact_continuation",
)
C_PREREQUISITE_RECEIPT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "immediate_control_calibration",
        "fork_identity",
        "b72_selection_envelope_admission",
    }
)
DECISION_CODES: Final[Mapping[str, str]] = {
    "A": "GMNW-SEL1-B72-A/DELAYED-WORLD-AMPLIFIES-AVAILABLE-GRU",
    "B": "GMNW-SEL1-B72-B/CAPABILITY-PROVED-SELECTION-LIMITED",
    "C": "GMNW-SEL1-B72-C/ONE-PRECISE-PREREQUISITE",
    "D": "GMNW-SEL1-B72-D/NO-DELAYED-SPECIFIC-GRU-AMPLIFICATION",
}


class Sel1HarnessError(RuntimeError):
    """The SEL1 contract, adapter boundary, or result is invalid."""


@dataclass(frozen=True, slots=True)
class Sel1CellV1:
    ordinal: int
    cell_id: str
    pair_id: str
    founder_count: int
    context: str
    cue_visible_at_decision: bool
    seed: int
    policy_arm: str = POLICY_ARM

    def configuration_kwargs(self) -> dict[str, Any]:
        """Return only the parent-owned configuration additions for this cell."""

        return {
            "policy_arm": self.policy_arm,
            "gru_founder_count": self.founder_count,
            "cue_visible_at_decision": self.cue_visible_at_decision,
            "task_seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class Sel1PairV1:
    pair_id: str
    founder_count: int
    seed: int
    delayed_cell_id: str
    immediate_cell_id: str


@dataclass(frozen=True, slots=True)
class PreparedSel1B72V1:
    native: Any
    preparation_receipt: Mapping[str, Any]


def _cell_id(founder_count: int, seed: int, context: str) -> str:
    return f"sel1_f{founder_count:02d}_s{seed:03d}_{CONTEXT_CODES[context]}"


def _pair_id(founder_count: int, seed: int) -> str:
    return f"sel1_f{founder_count:02d}_s{seed:03d}"


def cells_v1() -> tuple[Sel1CellV1, ...]:
    """Return pair-adjacent cells in founder, seed, delayed/immediate order."""

    cells: list[Sel1CellV1] = []
    for founder_count in FOUNDER_COUNTS:
        for seed in SEEDS:
            pair_id = _pair_id(founder_count, seed)
            for context in CONTEXTS:
                cells.append(
                    Sel1CellV1(
                        ordinal=len(cells),
                        cell_id=_cell_id(founder_count, seed, context),
                        pair_id=pair_id,
                        founder_count=founder_count,
                        context=context,
                        cue_visible_at_decision=CUE_VISIBLE_AT_DECISION[context],
                        seed=seed,
                    )
                )
    return tuple(cells)


def pairs_v1() -> tuple[Sel1PairV1, ...]:
    by_pair: dict[str, list[Sel1CellV1]] = {}
    for cell in cells_v1():
        by_pair.setdefault(cell.pair_id, []).append(cell)
    return tuple(
        Sel1PairV1(
            pair_id=pair_id,
            founder_count=pair[0].founder_count,
            seed=pair[0].seed,
            delayed_cell_id=pair[0].cell_id,
            immediate_cell_id=pair[1].cell_id,
        )
        for pair_id, pair in by_pair.items()
    )


def repeat_cell_ids_v1() -> tuple[str, ...]:
    return REPEAT_CELL_IDS


def continuation_cell_ids_v1() -> tuple[str, ...]:
    return CONTINUATION_CELL_IDS


def configurations_v1() -> tuple[Any, ...]:
    """Return the frozen mixed-selection GMNW configuration in physical order."""

    module = importlib.import_module(
        "engine.perf.cuda_architecture_b_direct_resident_gmnw_gru_v1"
    )
    config_type = module.GmnwConfigurationV1
    return tuple(
        config_type(
            cue_alphabet=CUE_ALPHABET,
            cue_ticks=1,
            delay_ticks=DELAY_TICKS,
            consequence_ticks=1,
            distractors_enabled=DISTRACTORS_ENABLED,
            cue_visible_at_decision=cell.cue_visible_at_decision,
            reset_hidden_each_tick=False,
            policy_arm=POLICY_ARM,
            gru_founder_count=cell.founder_count,
            task_seed=cell.seed,
            grace_generation_count=GRACE_GENERATION_COUNT,
            severe_generation_count=SEVERE_GENERATION_COUNT,
            grace_damage_fraction=GRACE_DAMAGE_FRACTION,
            severe_damage_fraction=SEVERE_DAMAGE_FRACTION,
            lethal_damage_fraction=LETHAL_DAMAGE_FRACTION,
            grace_reproduction_scale=GRACE_REPRODUCTION_SCALE,
            terminal_low_population_no_reinit=True,
        )
        for cell in cells_v1()
    )


def source_projection_paths_v1(root: Path) -> tuple[Path, ...]:
    """Resolve the exact stable-ID per-cell source projection layout."""

    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise Sel1HarnessError(f"source projection root is not a directory: {resolved}")
    paths = tuple(
        resolved / f"{cell.cell_id}{SOURCE_PROJECTION_SUFFIX}" for cell in cells_v1()
    )
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise Sel1HarnessError(
            "missing per-cell projections: " + ", ".join(missing[:8])
        )
    return paths


def _projection_namespace_v1(value: Any, *, cell_id: str) -> SimpleNamespace:
    if isinstance(value, Mapping) and "projection" in value:
        value = value["projection"]
    if not isinstance(value, Mapping):
        raise Sel1HarnessError(f"{cell_id}: source projection must be an object")
    schema_version = value.get("schema_version")
    metadata = value.get("metadata")
    tensors = value.get("tensors")
    if type(schema_version) is not str or not schema_version:
        raise Sel1HarnessError(f"{cell_id}: source projection schema is invalid")
    if not isinstance(metadata, Mapping) or not isinstance(tensors, Mapping):
        raise Sel1HarnessError(f"{cell_id}: source projection body is incomplete")
    if not tensors:
        raise Sel1HarnessError(f"{cell_id}: source projection has no tensors")
    typed_tensors: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise Sel1HarnessError(f"{cell_id}: source projection tensor is invalid")
        if tensor.device.type != "cpu" or tensor.requires_grad:
            raise Sel1HarnessError(
                f"{cell_id}: source projection tensors must be detached CPU storage"
            )
        typed_tensors[name] = tensor
    runtime_fields = {
        name: copy.deepcopy(member)
        for name, member in value.items()
        if name not in {"schema_version", "metadata", "tensors"}
    }
    return SimpleNamespace(
        schema_version=schema_version,
        metadata=copy.deepcopy(dict(metadata)),
        tensors=typed_tensors,
        **runtime_fields,
    )


def load_source_projections_v1(root: Path) -> tuple[SimpleNamespace, ...]:
    """Load exactly one private CPU projection for every preregistered cell."""

    loaded: list[SimpleNamespace] = []
    occupied: dict[tuple[str, int], tuple[str, str]] = {}
    for cell, path in zip(cells_v1(), source_projection_paths_v1(root), strict=True):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        projection = _projection_namespace_v1(payload, cell_id=cell.cell_id)
        for name, tensor in projection.tensors.items():
            if not tensor.numel():
                continue
            storage = (str(tensor.device), int(tensor.untyped_storage().data_ptr()))
            previous = occupied.get(storage)
            if previous is not None:
                raise Sel1HarnessError(
                    f"{cell.cell_id}: source tensor {name!r} aliases "
                    f"{previous[0]} tensor {previous[1]!r}"
                )
            occupied[storage] = (cell.cell_id, name)
        loaded.append(projection)
    if len({id(projection) for projection in loaded}) != BATCH_CELLS:
        raise Sel1HarnessError("per-cell projections are not distinct objects")
    return tuple(loaded)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _family_total_schema_v1(*, integer: bool) -> dict[str, Any]:
    value_schema: dict[str, Any] = {"type": "integer", "minimum": 0}
    if not integer:
        value_schema = {"type": "number", "minimum": 0.0}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [*POLICY_FAMILIES, "total"],
        "properties": {
            family: dict(value_schema) for family in (*POLICY_FAMILIES, "total")
        },
    }


def _selection_generation_schema_v1() -> dict[str, Any]:
    nullable_rate = {
        "anyOf": [
            {"type": "number", "minimum": 0.0, "maximum": 1.0},
            {"type": "null"},
        ]
    }
    nullable_nonnegative = {
        "anyOf": [{"type": "number", "minimum": 0.0}, {"type": "null"}]
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["generation_index", "native", "derived"],
        "properties": {
            "generation_index": {"type": "integer", "minimum": 0, "maximum": 11},
            "native": {
                "type": "object",
                "additionalProperties": False,
                "required": list(NATIVE_METRIC_NAMES),
                "properties": {
                    **{
                        name: _family_total_schema_v1(integer=True)
                        for name in COUNT_METRIC_NAMES
                    },
                    COST_METRIC_NAME: _family_total_schema_v1(integer=False),
                },
            },
            "derived": {
                "type": "object",
                "additionalProperties": False,
                "required": list(POLICY_FAMILIES),
                "properties": {
                    family: {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(DERIVED_METRIC_NAMES),
                        "properties": {
                            **{
                                name: dict(nullable_rate)
                                for name in DERIVED_METRIC_NAMES
                                if name
                                != "recurrent_upkeep_charge_per_alive_pre_turnover"
                            },
                            "recurrent_upkeep_charge_per_alive_pre_turnover": dict(
                                nullable_nonnegative
                            ),
                        },
                    }
                    for family in POLICY_FAMILIES
                },
            },
        },
    }


def result_schema_v1() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": RESULT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "cell_id",
            "pair_id",
            "founder_count",
            "context",
            "cue_visible_at_decision",
            "seed",
            "policy_arm",
            "status",
            "completed_generations",
            "execution_receipt",
            "metrics",
            "authoritative_state_sha256",
        ],
        "properties": {
            "schema_version": {"const": RESULT_SCHEMA_VERSION},
            "cell_id": {"type": "string"},
            "pair_id": {"type": "string"},
            "founder_count": {"enum": list(FOUNDER_COUNTS)},
            "context": {"enum": list(CONTEXTS)},
            "cue_visible_at_decision": {"type": "boolean"},
            "seed": {"enum": list(SEEDS)},
            "policy_arm": {"const": POLICY_ARM},
            "status": {"const": "completed"},
            "completed_generations": {"const": GENERATIONS},
            "execution_receipt": {
                "type": "object",
                "required": [
                    "native_window_loop_owner",
                    "runtime_jit",
                    "hidden_fallback",
                    "python_tick_or_generation_ownership",
                    "native_dispatches",
                    "physical_batch_cells",
                    "logical_chunking",
                ],
                "properties": {
                    "native_window_loop_owner": {"const": "native_extension"},
                    "runtime_jit": {"const": False},
                    "hidden_fallback": {"const": False},
                    "python_tick_or_generation_ownership": {"const": False},
                    "native_dispatches": {"const": 1},
                    "physical_batch_cells": {"const": BATCH_CELLS},
                    "logical_chunking": {"const": False},
                },
            },
            "metrics": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "initial_gru_count",
                    "final_gru_count",
                    "final_alive_count",
                    "final_gru_fraction",
                    "selection_generations",
                ],
                "properties": {
                    "initial_gru_count": {"type": "integer", "minimum": 0},
                    "final_gru_count": {"type": "integer", "minimum": 0},
                    "final_alive_count": {"type": "integer", "minimum": 0},
                    "final_gru_fraction": {
                        "anyOf": [
                            {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            {"type": "null"},
                        ]
                    },
                    "selection_generations": {
                        "type": "array",
                        "minItems": GENERATIONS,
                        "maxItems": GENERATIONS,
                        "items": _selection_generation_schema_v1(),
                    },
                },
            },
            "authoritative_state_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
    }


def contract_v1() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_label": "gmnw-sel1-b72g12t80",
        "scientific_scope": (
            "bounded paired selection comparison only; no intelligence, emergence, "
            "consciousness, broad cognition, or broad adaptive-memory claim"
        ),
        "matrix_order": "founder_count_then_seed_then_delayed_immediate",
        "policy_arm": POLICY_ARM,
        "founder_counts": list(FOUNDER_COUNTS),
        "contexts": [
            {
                "name": context,
                "code": CONTEXT_CODES[context],
                "cue_visible_at_decision": CUE_VISIBLE_AT_DECISION[context],
            }
            for context in CONTEXTS
        ],
        "seeds": list(SEEDS),
        "batch_cells": BATCH_CELLS,
        "physical_dispatch_count": 1,
        "logical_chunking": False,
        "native_window_count": GENERATIONS // 2,
        "generations": GENERATIONS,
        "ticks_per_generation": TICKS_PER_GENERATION,
        "gmnw_configuration": {
            "cue_alphabet": CUE_ALPHABET,
            "cue_ticks": 1,
            "delay_ticks": DELAY_TICKS,
            "consequence_ticks": 1,
            "distractors_enabled": DISTRACTORS_ENABLED,
            "grace_generation_count": GRACE_GENERATION_COUNT,
            "severe_generation_count": SEVERE_GENERATION_COUNT,
            "grace_damage_fraction": GRACE_DAMAGE_FRACTION,
            "severe_damage_fraction": SEVERE_DAMAGE_FRACTION,
            "lethal_damage_fraction": LETHAL_DAMAGE_FRACTION,
            "grace_reproduction_scale": GRACE_REPRODUCTION_SCALE,
        },
        "cells": [asdict(cell) for cell in cells_v1()],
        "pairs": [asdict(pair) for pair in pairs_v1()],
        "repeat_cell_ids": list(repeat_cell_ids_v1()),
        "continuation_cell_ids": list(continuation_cell_ids_v1()),
        "thresholds": dict(THRESHOLDS),
        "result_schema": result_schema_v1(),
        "selection_metrics": {
            "required": True,
            "family_axis": list(POLICY_FAMILIES),
            "count_metrics": list(COUNT_METRIC_NAMES),
            "cost_metric": COST_METRIC_NAME,
            "generation_count": GENERATIONS,
            "pending_non_native_measurements": list(PENDING_NATIVE_MEASUREMENTS),
        },
        "interfaces": {
            "preparation": "prepare_physical_b72_v1",
            "execution": "execute_prepared_b72_v1",
            "complete_run": "execute_physical_b72_v1",
        },
    }
    return {**payload, "contract_sha256": canonical_sha256(payload)}


def validate_contract_v1(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    candidate = dict(contract_v1() if value is None else value)
    if candidate.get("schema_version") != SCHEMA_VERSION:
        raise Sel1HarnessError("schema_version mismatch")
    actual_hash = candidate.pop("contract_sha256", None)
    expected_hash = canonical_sha256(candidate)
    if actual_hash != expected_hash:
        raise Sel1HarnessError("contract_sha256 mismatch")
    expected = contract_v1()
    if {**candidate, "contract_sha256": actual_hash} != expected:
        raise Sel1HarnessError(
            "contract does not match the frozen SEL1 preregistration"
        )
    return expected


def _accepted_integer_membership(function: Any, argument: str) -> set[int]:
    accepted: set[int] = set()
    tree = ast.parse(inspect.getsource(function))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name):
            continue
        if node.left.id != argument or len(node.comparators) != 1:
            continue
        values = node.comparators[0]
        if not isinstance(values, (ast.Set, ast.Tuple, ast.List)):
            continue
        for item in values.elts:
            if isinstance(item, ast.Constant) and type(item.value) is int:
                accepted.add(item.value)
    return accepted


def engine_readiness_v1() -> dict[str, Any]:
    """Inspect the CPU-visible engine API without preparing or running a cell."""

    blockers: list[str] = []
    module_name = "engine.perf.cuda_architecture_b_direct_resident_gmnw_gru_v1"
    try:
        module = importlib.import_module(module_name)
        config_type = module.GmnwConfigurationV1
        build = module.build_prepared_direct_resident_gmnw_gru_v1
        execute = module.execute_direct_resident_gmnw_gru_v1
        allocator_module = importlib.import_module(
            "engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1"
        )
        allocator = allocator_module.allocate_direct_resident_source_v1
    except (AttributeError, ImportError) as exc:
        return {
            "ok": False,
            "module": module_name,
            "batch_cells": BATCH_CELLS,
            "generations": GENERATIONS,
            "blockers": [f"engine import surface unavailable: {exc}"],
        }

    config_fields = {field.name for field in fields(config_type)}
    for required in ("policy_arm", "gru_founder_count", "cue_visible_at_decision"):
        if required not in config_fields:
            blockers.append(f"GmnwConfigurationV1.{required} is unavailable")
    if not blockers:
        try:
            for cell in cells_v1():
                config_type(**cell.configuration_kwargs())
        except (TypeError, ValueError) as exc:
            blockers.append(f"SEL1 configuration construction failed: {exc}")

    for name, function in (("prepare", build), ("execute", execute)):
        accepted = _accepted_integer_membership(function, "logical_generation_count")
        if GENERATIONS not in accepted:
            blockers.append(
                f"{name} surface does not advertise resident g{GENERATIONS} support"
            )
    allocator_signature = inspect.signature(allocator)
    parameter = allocator_signature.parameters.get("per_cell_requests")
    if (
        parameter is None
        or parameter.kind is not inspect.Parameter.KEYWORD_ONLY
        or parameter.default is not None
    ):
        blockers.append("engine per-cell source allocator API is unavailable")
    build_parameter = inspect.signature(build).parameters.get(
        "selection_metrics_enabled"
    )
    if build_parameter is None or build_parameter.default is not False:
        blockers.append("engine optional selection metrics API is unavailable")
    for name, expected in (
        ("GMNW_SELECTION_POLICY_FAMILIES_V1", POLICY_FAMILIES),
        ("GMNW_SELECTION_COUNT_METRIC_NAMES_V1", COUNT_METRIC_NAMES),
    ):
        if tuple(getattr(module, name, ())) != expected:
            blockers.append(f"engine {name} contract mismatch")
    return {
        "ok": not blockers,
        "module": module_name,
        "batch_cells": BATCH_CELLS,
        "generations": GENERATIONS,
        "blockers": blockers,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_projection_metadata_v1(projection: SimpleNamespace, gru: Any) -> None:
    projection.metadata["native_runtime_architecture_b_compact_reporting_only_v1"] = (
        False
    )
    projection.metadata["import_digest"] = gru.projection_digest(
        {"metadata": projection.metadata, "tensors": projection.tensors}
    )


def _per_cell_request_v1(
    primary: Any,
    projection: SimpleNamespace,
    *,
    gru: Any,
) -> Any:
    boundary = gru.import_cpu_owner_boundary_to_cuda(
        projection,
        event_capacity=primary.event_capacity,
        device=primary.cuda_device,
        policy_family="stable_gru",
    )
    max_births = int(primary.envelope.max_births_per_tick)
    birth_state = gru.prepare_generation_birth_state_cuda(
        boundary,
        projection,
        steps=gru.PHYSICAL_WINDOW_TICKS_V1,
        max_births_per_tick=max_births,
        turnover_refill_capacity=gru.PHYSICAL_WINDOW_TURNOVER_EPISODES_V1,
        resident_policy_episode_capacity=(
            gru.required_physical_window_policy_episodes_v1(max_births)
        ),
    )
    return gru.replace(
        primary,
        import_projection=projection,
        cuda_boundary=boundary,
        generation_birth_state=birth_state,
    )


@contextmanager
def _bind_per_cell_source_api_v1(
    projections: Sequence[SimpleNamespace],
) -> Iterator[dict[str, Any]]:
    """Import one verified source identity into private batched device banks."""

    if len(projections) != BATCH_CELLS:
        raise Sel1HarnessError(f"expected exactly {BATCH_CELLS} source projections")
    gru = importlib.import_module(
        "engine.perf.cuda_architecture_b_direct_resident_gru_v3"
    )
    allocator_module = importlib.import_module(
        "engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1"
    )
    original_projection_builder = gru.build_import_projection
    original_allocator = gru.allocate_direct_resident_source_v1
    engine_allocator = allocator_module.allocate_direct_resident_source_v1
    audit: dict[str, Any] = {
        "projection_builder_calls": 0,
        "allocator_calls": 0,
        "per_cell_request_count": 0,
        "source_identity_count": 1,
        "engine_allocator": (
            "engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1."
            "allocate_direct_resident_source_v1"
        ),
    }

    for projection in projections:
        _prepare_projection_metadata_v1(projection, gru)

    def projection_builder(*args: Any, **kwargs: Any) -> SimpleNamespace:
        del args, kwargs
        audit["projection_builder_calls"] += 1
        if audit["projection_builder_calls"] != 1:
            raise Sel1HarnessError("logical chunking attempted multiple source builds")
        return projections[0]

    def allocator(primary: Any) -> Any:
        audit["allocator_calls"] += 1
        if audit["allocator_calls"] != 1:
            raise Sel1HarnessError("fallback attempted multiple source allocations")
        audit["per_cell_request_count"] = 1
        return engine_allocator(primary)

    gru.build_import_projection = projection_builder
    gru.allocate_direct_resident_source_v1 = allocator
    try:
        yield audit
    finally:
        gru.build_import_projection = original_projection_builder
        gru.allocate_direct_resident_source_v1 = original_allocator


def _validate_preparation_receipt_v1(receipt: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "physical_batch_cells": BATCH_CELLS,
        "logical_generation_count": GENERATIONS,
        "ticks_per_generation": TICKS_PER_GENERATION,
        "logical_chunking": False,
        "source_projection_count": BATCH_CELLS,
        "source_identity_count": 1,
        "per_cell_request_count": 1,
        "projection_builder_calls": 1,
        "allocator_calls": 1,
        "selection_metrics_enabled": True,
        "hidden_fallback": False,
    }
    mismatches = {
        name: {"expected": value, "actual": receipt.get(name)}
        for name, value in expected.items()
        if type(receipt.get(name)) is not type(value) or receipt.get(name) != value
    }
    cell_ids = receipt.get("source_projection_cell_ids")
    expected_ids = [cell.cell_id for cell in cells_v1()]
    if cell_ids != expected_ids:
        mismatches["source_projection_cell_ids"] = {
            "expected": expected_ids,
            "actual": cell_ids,
        }
    if mismatches:
        raise Sel1HarnessError(
            f"physical B72 preparation receipt mismatch: {mismatches}"
        )
    return dict(receipt)


def prepare_physical_b72_v1(
    module: Any,
    *,
    source_projection_root: Path,
    run_root: Path,
    witness: Path,
    mlp: Path,
    runner_factory: Callable[[Path], Any] | None = None,
) -> PreparedSel1B72V1:
    """Prepare one physical b72g12t80 batch from 72 private source projections."""

    readiness = engine_readiness_v1()
    if not readiness["ok"]:
        raise Sel1HarnessError(
            "engine is not SEL1-ready: " + "; ".join(readiness["blockers"])
        )
    for name, path in (("witness", witness), ("MLP", mlp)):
        if not path.expanduser().resolve().is_file():
            raise Sel1HarnessError(f"{name} artifact is missing: {path}")
    resolved_run_root = run_root.expanduser().resolve()
    if resolved_run_root.exists():
        raise FileExistsError(resolved_run_root)
    projections = load_source_projections_v1(source_projection_root)
    projection_paths = source_projection_paths_v1(source_projection_root)
    if runner_factory is None:
        gate = importlib.import_module(
            "tools.scripts.eiel_cuda_architecture_b_gru_gate_v3"
        )
        runner_factory = gate._new_gru_runner
    runner = runner_factory(resolved_run_root / "source_bootstrap")
    gmnw = importlib.import_module(
        "engine.perf.cuda_architecture_b_direct_resident_gmnw_gru_v1"
    )
    configs = configurations_v1()
    with _bind_per_cell_source_api_v1(projections) as audit:
        prepared = gmnw.build_prepared_direct_resident_gmnw_gru_v1(
            runner,
            qualified_cuda_module=module,
            config=configs[0],
            cell_configs=configs,
            batch_cells=BATCH_CELLS,
            control_matrix_enabled=True,
            logical_generation_count=GENERATIONS,
            witness_path=witness.expanduser().resolve(),
            matched_mlp_path=mlp.expanduser().resolve(),
            qualification_output_ticks=1,
            selection_metrics_enabled=True,
        )
    live = prepared.patch.source.live_state
    alive = live.get("alive_mask") if isinstance(live, Mapping) else None
    if not isinstance(alive, torch.Tensor) or tuple(alive.shape[:1]) != (BATCH_CELLS,):
        raise Sel1HarnessError("prepared source does not expose physical B72")
    if prepared.patch.source.logical_generation_count != GENERATIONS:
        raise Sel1HarnessError("prepared source does not expose the exact g12 schedule")
    _validated_native_selection_metrics_v1(prepared.selection_metrics)
    receipt = _validate_preparation_receipt_v1(
        {
            "physical_batch_cells": BATCH_CELLS,
            "logical_generation_count": GENERATIONS,
            "ticks_per_generation": TICKS_PER_GENERATION,
            "logical_chunking": False,
            "source_projection_count": len(projections),
            "source_projection_cell_ids": [cell.cell_id for cell in cells_v1()],
            "source_projection_files": [
                {
                    "cell_id": cell.cell_id,
                    "path": str(path),
                    "sha256": _file_sha256(path),
                }
                for cell, path in zip(cells_v1(), projection_paths, strict=True)
            ],
            "per_cell_request_count": audit["per_cell_request_count"],
            "source_identity_count": audit["source_identity_count"],
            "projection_builder_calls": audit["projection_builder_calls"],
            "allocator_calls": audit["allocator_calls"],
            "engine_allocator": audit["engine_allocator"],
            "selection_metrics_enabled": True,
            "hidden_fallback": False,
        }
    )
    return PreparedSel1B72V1(native=prepared, preparation_receipt=receipt)


def _ratio_or_none(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    value = float(numerator / denominator)
    if not math.isfinite(value):
        raise Sel1HarnessError("derived metric is not finite")
    return value


def _validated_native_selection_metrics_v1(
    value: Any,
) -> dict[str, torch.Tensor]:
    expected_names = {*NATIVE_METRIC_NAMES, "status"}
    if not isinstance(value, Mapping) or set(value) != expected_names:
        raise Sel1HarnessError("native selection metrics mapping is missing or invalid")
    expected_shape = (BATCH_CELLS, GENERATIONS, len(POLICY_FAMILIES))
    metrics: dict[str, torch.Tensor] = {}
    for name in COUNT_METRIC_NAMES:
        tensor = value[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.int64
            or tuple(tensor.shape) != expected_shape
        ):
            raise Sel1HarnessError(
                f"native selection metric {name} must be int64[72,12,2]"
            )
        if bool(torch.any(tensor < 0).item()):
            raise Sel1HarnessError(f"native selection metric {name} is negative")
        metrics[name] = tensor
    upkeep = value[COST_METRIC_NAME]
    if (
        not isinstance(upkeep, torch.Tensor)
        or upkeep.dtype != torch.float64
        or tuple(upkeep.shape) != expected_shape
    ):
        raise Sel1HarnessError(
            "native selection metric recurrent_upkeep_charge_total must be "
            "float64[72,12,2]"
        )
    if not bool(torch.isfinite(upkeep).all().item()) or bool(
        torch.any(upkeep < 0.0).item()
    ):
        raise Sel1HarnessError("native recurrent upkeep must be finite and non-negative")
    metrics[COST_METRIC_NAME] = upkeep
    status = value["status"]
    if (
        not isinstance(status, torch.Tensor)
        or status.dtype != torch.int32
        or tuple(status.shape) != (BATCH_CELLS,)
    ):
        raise Sel1HarnessError("native selection metrics status must be int32[72]")
    if bool(torch.count_nonzero(status).item()):
        raise Sel1HarnessError("native selection metrics status is nonzero")
    metrics["status"] = status
    if bool(torch.any(metrics["correct_count"] > metrics["decision_count"]).item()):
        raise Sel1HarnessError("native correct_count exceeds decision_count")
    if bool(
        torch.any(
            metrics["unique_selected_parent_pool_count"]
            > metrics["alive_pre_turnover"]
        ).item()
    ):
        raise Sel1HarnessError("native selected parent pool exceeds alive population")
    zero_founder_rows = [
        cell.ordinal for cell in cells_v1() if cell.founder_count == 0
    ]
    for name in NATIVE_METRIC_NAMES:
        gru_values = metrics[name][zero_founder_rows, :, 1]
        if bool(torch.count_nonzero(gru_values).item()):
            raise Sel1HarnessError(f"zero-GRU integrity failed for native metric {name}")
    return metrics


def _family_values_v1(
    metrics: Mapping[str, torch.Tensor],
    *,
    cell_index: int,
    generation_index: int,
    name: str,
) -> dict[str, int | float]:
    values: list[int | float] = []
    for family_index in range(len(POLICY_FAMILIES)):
        raw = metrics[name][cell_index, generation_index, family_index].item()
        values.append(int(raw) if name in COUNT_METRIC_NAMES else float(raw))
    total: int | float
    if name in COUNT_METRIC_NAMES:
        total = sum(int(value) for value in values)
    else:
        total = float(sum(float(value) for value in values))
    return {**dict(zip(POLICY_FAMILIES, values, strict=True)), "total": total}


def _derived_family_metrics_v1(
    native: Mapping[str, Mapping[str, int | float]],
    family: str,
) -> dict[str, float | None]:
    alive_pre = int(native["alive_pre_turnover"][family])
    task_deaths = int(native["task_death_count"][family])
    return {
        "decision_accuracy": _ratio_or_none(
            int(native["correct_count"][family]),
            int(native["decision_count"][family]),
        ),
        "alive_pre_turnover_share": _ratio_or_none(
            alive_pre, int(native["alive_pre_turnover"]["total"])
        ),
        "alive_post_turnover_share": _ratio_or_none(
            int(native["alive_post_turnover"][family]),
            int(native["alive_post_turnover"]["total"]),
        ),
        "task_death_share": _ratio_or_none(
            task_deaths, int(native["task_death_count"]["total"])
        ),
        "task_death_rate": _ratio_or_none(task_deaths, task_deaths + alive_pre),
        "selected_parent_pool_share": _ratio_or_none(
            int(native["unique_selected_parent_pool_count"][family]),
            int(native["unique_selected_parent_pool_count"]["total"]),
        ),
        "selected_parent_pool_rate": _ratio_or_none(
            int(native["unique_selected_parent_pool_count"][family]), alive_pre
        ),
        "committed_turnover_offspring_share": _ratio_or_none(
            int(native["committed_turnover_offspring_count"][family]),
            int(native["committed_turnover_offspring_count"]["total"]),
        ),
        "recurrent_upkeep_charge_share": _ratio_or_none(
            float(native[COST_METRIC_NAME][family]),
            float(native[COST_METRIC_NAME]["total"]),
        ),
        "recurrent_upkeep_charge_per_alive_pre_turnover": _ratio_or_none(
            float(native[COST_METRIC_NAME][family]), alive_pre
        ),
    }


def _selection_generation_v1(
    metrics: Mapping[str, torch.Tensor],
    *,
    cell_index: int,
    generation_index: int,
) -> dict[str, Any]:
    native = {
        name: _family_values_v1(
            metrics,
            cell_index=cell_index,
            generation_index=generation_index,
            name=name,
        )
        for name in NATIVE_METRIC_NAMES
    }
    return {
        "generation_index": generation_index,
        "native": native,
        "derived": {
            family: _derived_family_metrics_v1(native, family)
            for family in POLICY_FAMILIES
        },
    }


def _validated_selection_generation_v1(
    source: Any,
    *,
    cell_id: str,
    generation_index: int,
) -> dict[str, Any]:
    if not isinstance(source, Mapping) or set(source) != {
        "generation_index",
        "native",
        "derived",
    }:
        raise Sel1HarnessError(f"{cell_id}: selection generation schema mismatch")
    if source["generation_index"] != generation_index:
        raise Sel1HarnessError(f"{cell_id}: selection generation index mismatch")
    native = source["native"]
    if not isinstance(native, Mapping) or set(native) != set(NATIVE_METRIC_NAMES):
        raise Sel1HarnessError(f"{cell_id}: native selection metric fields mismatch")
    normalized_native: dict[str, dict[str, int | float]] = {}
    for name in NATIVE_METRIC_NAMES:
        values = native[name]
        if not isinstance(values, Mapping) or set(values) != {
            *POLICY_FAMILIES,
            "total",
        }:
            raise Sel1HarnessError(f"{cell_id}: {name} family fields mismatch")
        expected_type = int if name in COUNT_METRIC_NAMES else float
        if any(
            type(values[family]) is not expected_type or values[family] < 0
            for family in (*POLICY_FAMILIES, "total")
        ):
            raise Sel1HarnessError(f"{cell_id}: {name} values are invalid")
        if expected_type is float and any(
            not math.isfinite(values[family])
            for family in (*POLICY_FAMILIES, "total")
        ):
            raise Sel1HarnessError(f"{cell_id}: {name} values are not finite")
        expected_total = values["mlp"] + values["gru"]
        if values["total"] != expected_total:
            raise Sel1HarnessError(f"{cell_id}: {name} family sum mismatch")
        normalized_native[name] = dict(values)
    if normalized_native["correct_count"]["mlp"] > normalized_native[
        "decision_count"
    ]["mlp"] or normalized_native["correct_count"]["gru"] > normalized_native[
        "decision_count"
    ]["gru"]:
        raise Sel1HarnessError(f"{cell_id}: decision correctness bounds failed")
    for family in POLICY_FAMILIES:
        if normalized_native["unique_selected_parent_pool_count"][family] > (
            normalized_native["alive_pre_turnover"][family]
        ):
            raise Sel1HarnessError(f"{cell_id}: selected parent pool bound failed")
    expected_derived = {
        family: _derived_family_metrics_v1(normalized_native, family)
        for family in POLICY_FAMILIES
    }
    if source["derived"] != expected_derived:
        raise Sel1HarnessError(f"{cell_id}: derived selection metrics mismatch")
    return {
        "generation_index": generation_index,
        "native": normalized_native,
        "derived": expected_derived,
    }


def validate_result_rows_v1(
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    expected = cells_v1()
    if len(results) != BATCH_CELLS:
        raise Sel1HarnessError(f"expected exactly {BATCH_CELLS} result rows")
    normalized: list[dict[str, Any]] = []
    for cell, source in zip(expected, results, strict=True):
        row = dict(source)
        expected_identity = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "cell_id": cell.cell_id,
            "pair_id": cell.pair_id,
            "founder_count": cell.founder_count,
            "context": cell.context,
            "cue_visible_at_decision": cell.cue_visible_at_decision,
            "seed": cell.seed,
            "policy_arm": POLICY_ARM,
            "status": "completed",
            "completed_generations": GENERATIONS,
        }
        for name, expected_value in expected_identity.items():
            if (
                type(row.get(name)) is not type(expected_value)
                or row.get(name) != expected_value
            ):
                raise Sel1HarnessError(f"{cell.cell_id}: {name} mismatch")
        receipt = row.get("execution_receipt")
        required_receipt = {
            "native_window_loop_owner": "native_extension",
            "runtime_jit": False,
            "hidden_fallback": False,
            "python_tick_or_generation_ownership": False,
            "native_dispatches": 1,
            "physical_batch_cells": BATCH_CELLS,
            "logical_chunking": False,
        }
        if not isinstance(receipt, Mapping) or any(
            type(receipt.get(name)) is not type(value) or receipt.get(name) != value
            for name, value in required_receipt.items()
        ):
            raise Sel1HarnessError(
                f"{cell.cell_id}: resident execution receipt mismatch"
            )
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != {
            "initial_gru_count",
            "final_gru_count",
            "final_alive_count",
            "final_gru_fraction",
            "selection_generations",
        }:
            raise Sel1HarnessError(f"{cell.cell_id}: metrics must be an object")
        for name in ("initial_gru_count", "final_gru_count", "final_alive_count"):
            if type(metrics.get(name)) is not int or metrics[name] < 0:
                raise Sel1HarnessError(f"{cell.cell_id}: {name} must be non-negative")
        if metrics["final_gru_count"] > metrics["final_alive_count"]:
            raise Sel1HarnessError(f"{cell.cell_id}: final family counts disagree")
        expected_fraction = _ratio_or_none(
            metrics["final_gru_count"], metrics["final_alive_count"]
        )
        fraction = metrics.get("final_gru_fraction")
        if fraction != expected_fraction:
            raise Sel1HarnessError(f"{cell.cell_id}: final_gru_fraction is invalid")
        if metrics["initial_gru_count"] != cell.founder_count:
            raise Sel1HarnessError(f"{cell.cell_id}: initial_gru_count mismatch")
        generations = metrics["selection_generations"]
        if not isinstance(generations, Sequence) or len(generations) != GENERATIONS:
            raise Sel1HarnessError(
                f"{cell.cell_id}: expected exactly {GENERATIONS} selection generations"
            )
        normalized_generations = [
            _validated_selection_generation_v1(
                generation,
                cell_id=cell.cell_id,
                generation_index=generation_index,
            )
            for generation_index, generation in enumerate(generations)
        ]
        final_native = normalized_generations[-1]["native"]
        if (
            final_native["alive_post_turnover"]["total"]
            != metrics["final_alive_count"]
            or final_native["alive_post_turnover"]["gru"]
            != metrics["final_gru_count"]
        ):
            raise Sel1HarnessError(f"{cell.cell_id}: final alive agreement failed")
        if cell.founder_count == 0:
            for generation in normalized_generations:
                if any(
                    generation["native"][name]["gru"] != 0
                    for name in NATIVE_METRIC_NAMES
                ):
                    raise Sel1HarnessError(
                        f"{cell.cell_id}: zero-GRU integrity failed"
                    )
        digest = row.get("authoritative_state_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise Sel1HarnessError(
                f"{cell.cell_id}: authoritative state digest is invalid"
            )
        normalized.append(
            {
                **row,
                "metrics": {
                    **dict(metrics),
                    "selection_generations": normalized_generations,
                },
            }
        )
    if len({row["cell_id"] for row in normalized}) != BATCH_CELLS:
        raise Sel1HarnessError("duplicate result cell_id")
    return tuple(normalized)


def validate_native_dispatch_receipt_v1(
    receipt: Mapping[str, Any],
    *,
    dispatch_calls: int,
) -> dict[str, Any]:
    """Require one b72g12t80 dispatch with the native loop as sole owner."""

    expected = {
        "completed_generations": GENERATIONS,
        "logical_generation_count": GENERATIONS,
        "window_count": GENERATIONS // 2,
        "settled_window_count": GENERATIONS // 2,
        "external_python_window_boundaries": 0,
        "native_window_loop_owner": "native_extension",
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
        "runtime_jit": False,
        "hidden_fallback": False,
        "gru_python_tick_or_generation_ownership": False,
        "gmnw_python_tick_or_generation_ownership": False,
    }
    mismatches = {
        name: {"expected": value, "actual": receipt.get(name)}
        for name, value in expected.items()
        if type(receipt.get(name)) is not type(value) or receipt.get(name) != value
    }
    if type(dispatch_calls) is not int or dispatch_calls != 1:
        mismatches["harness_dispatch_calls"] = {
            "expected": 1,
            "actual": dispatch_calls,
        }
    reported_dispatches = receipt.get("native_dispatches")
    if reported_dispatches is not None and (
        type(reported_dispatches) is not int or reported_dispatches != dispatch_calls
    ):
        mismatches["native_dispatches"] = {
            "expected": dispatch_calls,
            "actual": reported_dispatches,
        }
    if receipt.get("logical_chunking") is True:
        mismatches["logical_chunking"] = {"expected": False, "actual": True}
    if receipt.get("chunk_count") not in (None, 1):
        mismatches["chunk_count"] = {"expected": 1, "actual": receipt["chunk_count"]}
    if mismatches:
        raise Sel1HarnessError(
            f"native b72g12t80 dispatch receipt mismatch: {mismatches}"
        )
    return {
        **expected,
        "native_dispatches": dispatch_calls,
        "physical_batch_cells": BATCH_CELLS,
        "ticks_per_generation": TICKS_PER_GENERATION,
        "logical_chunking": False,
        "chunk_count": 1,
        "harness_dispatch_calls": dispatch_calls,
    }


def _result_rows_from_prepared_v1(
    prepared: PreparedSel1B72V1,
    *,
    state_digests: Sequence[str],
    native_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    if len(state_digests) != BATCH_CELLS:
        raise Sel1HarnessError("missing per-cell authoritative state digests")
    selection_metrics = _validated_native_selection_metrics_v1(
        getattr(prepared.native, "selection_metrics", None)
    )
    live = prepared.native.patch.source.live_state
    alive = live.get("alive_mask") if isinstance(live, Mapping) else None
    family = live.get("policy_use_gru") if isinstance(live, Mapping) else None
    if (
        not isinstance(alive, torch.Tensor)
        or not isinstance(family, torch.Tensor)
        or tuple(alive.shape) != tuple(family.shape)
        or alive.ndim != 2
        or int(alive.size(0)) != BATCH_CELLS
    ):
        raise Sel1HarnessError("final per-cell family projection is missing")
    alive_cpu = alive.detach().to(device="cpu", dtype=torch.bool)
    family_cpu = family.detach().to(device="cpu", dtype=torch.bool)
    execution_receipt = {
        "native_window_loop_owner": native_receipt["native_window_loop_owner"],
        "runtime_jit": native_receipt["runtime_jit"],
        "hidden_fallback": native_receipt["hidden_fallback"],
        "python_tick_or_generation_ownership": False,
        "native_dispatches": native_receipt["native_dispatches"],
        "physical_batch_cells": BATCH_CELLS,
        "logical_chunking": False,
    }
    rows: list[dict[str, Any]] = []
    for index, (cell, digest) in enumerate(zip(cells_v1(), state_digests, strict=True)):
        final_alive = int(alive_cpu[index].sum().item())
        final_gru = int((alive_cpu[index] & family_cpu[index]).sum().item())
        rows.append(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "cell_id": cell.cell_id,
                "pair_id": cell.pair_id,
                "founder_count": cell.founder_count,
                "context": cell.context,
                "cue_visible_at_decision": cell.cue_visible_at_decision,
                "seed": cell.seed,
                "policy_arm": POLICY_ARM,
                "status": "completed",
                "completed_generations": GENERATIONS,
                "execution_receipt": dict(execution_receipt),
                "metrics": {
                    "initial_gru_count": cell.founder_count,
                    "final_gru_count": final_gru,
                    "final_alive_count": final_alive,
                    "final_gru_fraction": _ratio_or_none(final_gru, final_alive),
                    "selection_generations": [
                        _selection_generation_v1(
                            selection_metrics,
                            cell_index=index,
                            generation_index=generation_index,
                        )
                        for generation_index in range(GENERATIONS)
                    ],
                },
                "authoritative_state_sha256": digest,
            }
        )
    return validate_result_rows_v1(rows)


def _pooled_metric_v1(
    row: Mapping[str, Any], name: str, family: str
) -> int | float:
    values = [
        generation["native"][name][family]
        for generation in row["metrics"]["selection_generations"]
    ]
    return sum(values)


def _cell_accuracy_v1(row: Mapping[str, Any], family: str) -> float | None:
    return _ratio_or_none(
        int(_pooled_metric_v1(row, "correct_count", family)),
        int(_pooled_metric_v1(row, "decision_count", family)),
    )


def _cell_parent_advantage_v1(row: Mapping[str, Any]) -> bool:
    for generation in row["metrics"]["selection_generations"]:
        derived = generation["derived"]["gru"]
        population_share = derived["alive_pre_turnover_share"]
        parent_share = derived["selected_parent_pool_share"]
        if (
            population_share is not None
            and parent_share is not None
            and population_share < 1.0
            and parent_share > population_share
        ):
            return True
    return False


def _cell_turnover_offspring_advantage_v1(row: Mapping[str, Any]) -> bool:
    offspring_share = _ratio_or_none(
        int(_pooled_metric_v1(row, "committed_turnover_offspring_count", "gru")),
        int(_pooled_metric_v1(row, "committed_turnover_offspring_count", "total")),
    )
    initial_share = row["founder_count"] / INITIAL_POPULATION
    return offspring_share is not None and offspring_share > initial_share


def _pooled_accuracy_v1(
    rows: Sequence[Mapping[str, Any]], family: str
) -> float | None:
    return _ratio_or_none(
        sum(int(_pooled_metric_v1(row, "correct_count", family)) for row in rows),
        sum(int(_pooled_metric_v1(row, "decision_count", family)) for row in rows),
    )


def _complete_median_v1(values: Sequence[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return float(statistics.median(value for value in values if value is not None))


def _endpoint_population_contribution_v1(row: Mapping[str, Any]) -> float:
    """Return GRU endpoint share, treating complete population extinction as zero."""

    metrics = row["metrics"]
    fraction = metrics["final_gru_fraction"]
    if fraction is not None:
        return float(fraction)
    if metrics["final_alive_count"] != 0 or metrics["final_gru_count"] != 0:
        raise Sel1HarnessError(
            f"{row['cell_id']}: null final_gru_fraction without extinction"
        )
    return 0.0


def _family_extinction_generation_v1(
    row: Mapping[str, Any], family: str
) -> int | None:
    """Return first post-turnover extinction generation and reject reappearance."""

    if family not in POLICY_FAMILIES:
        raise Sel1HarnessError(f"unknown policy family: {family}")
    initial_count = (
        row["founder_count"]
        if family == "gru"
        else INITIAL_POPULATION - row["founder_count"]
    )
    extinct_at = 0 if initial_count == 0 else None
    for generation_index, generation in enumerate(
        row["metrics"]["selection_generations"], start=1
    ):
        alive = int(generation["native"]["alive_post_turnover"][family])
        if alive == 0 and extinct_at is None:
            extinct_at = generation_index
        elif alive > 0 and extinct_at is not None:
            raise Sel1HarnessError(
                f"{row['cell_id']}: {family} family reappeared after extinction "
                f"at generation {generation_index}"
            )
    return extinct_at


def _a_gate_receipt_status_v1(
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    missing = [
        name
        for name in A_GATE_RECEIPT_NAMES
        if receipt is None or type(receipt.get(name)) is not bool
    ]
    failed = [
        name
        for name in A_GATE_RECEIPT_NAMES
        if receipt is not None and receipt.get(name) is False
    ]
    return {
        "required": list(A_GATE_RECEIPT_NAMES),
        "missing": missing,
        "failed": failed,
        "all_supplied_and_green": not missing and not failed,
    }


def _select_decision_v1(
    *,
    native_a_gate_pass: bool,
    directional_supporting_founder_count: int,
    required_supporting_founder_count: int,
    a_gate_receipt_status: Mapping[str, Any],
) -> dict[str, Any]:
    directional = (
        directional_supporting_founder_count >= required_supporting_founder_count
    )
    scientific_letter = "A" if native_a_gate_pass else ("B" if directional else "D")
    receipt_green = bool(a_gate_receipt_status["all_supplied_and_green"])
    missing = list(a_gate_receipt_status["missing"])
    failed = list(a_gate_receipt_status["failed"])

    if scientific_letter == "A" and receipt_green:
        final_letter: str | None = "A"
        status = "final"
    elif (
        scientific_letter == "A"
        and not missing
        and len(failed) == 1
        and failed[0] in C_PREREQUISITE_RECEIPT_NAMES
    ):
        final_letter = "C"
        status = "final_one_precise_prerequisite"
    elif scientific_letter == "A":
        final_letter = None
        status = "provisional_matrix_a_final_a_gated"
    else:
        final_letter = scientific_letter
        status = (
            "final" if receipt_green else "provisional_scientific_matrix_decision"
        )

    return {
        "letter": final_letter,
        "code": DECISION_CODES[final_letter] if final_letter is not None else None,
        "status": status,
        "scientific_matrix_letter": scientific_letter,
        "scientific_matrix_code": DECISION_CODES[scientific_letter],
        "directional_supporting_founder_count": (
            directional_supporting_founder_count
        ),
        "a_eligible": final_letter == "A",
        "a_gate_receipt": dict(a_gate_receipt_status),
    }


def analyze_result_rows_v1(
    results: Sequence[Mapping[str, Any]],
    *,
    a_gate_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the retained matrix and fail closed on external Decision-A gates."""

    rows = validate_result_rows_v1(results)
    by_id = {row["cell_id"]: row for row in rows}
    competence_by_founder: list[dict[str, Any]] = []
    amplification_by_founder: list[dict[str, Any]] = []
    for founder_count in FOUNDER_COUNTS[1:]:
        delayed = [
            row
            for row in rows
            if row["founder_count"] == founder_count and row["context"] == "delayed"
        ]
        immediate = [
            row
            for row in rows
            if row["founder_count"] == founder_count
            and row["context"] == "immediate"
        ]
        delayed_gru_accuracy = _pooled_accuracy_v1(delayed, "gru")
        delayed_mlp_accuracy = _pooled_accuracy_v1(delayed, "mlp")
        immediate_gru_accuracy = _pooled_accuracy_v1(immediate, "gru")
        immediate_mlp_accuracy = _pooled_accuracy_v1(immediate, "mlp")
        delayed_competence_pass = (
            delayed_gru_accuracy is not None
            and delayed_gru_accuracy >= THRESHOLDS["minimum_gru_delayed_accuracy"]
            and delayed_mlp_accuracy is not None
            and delayed_mlp_accuracy <= THRESHOLDS["maximum_mlp_delayed_accuracy"]
        )
        immediate_competence_pass = (
            immediate_gru_accuracy is not None
            and immediate_gru_accuracy
            >= THRESHOLDS["minimum_gru_immediate_accuracy"]
            and immediate_mlp_accuracy is not None
            and immediate_mlp_accuracy
            >= THRESHOLDS["minimum_mlp_immediate_accuracy"]
        )
        competence_by_founder.append(
            {
                "founder_count": founder_count,
                "delayed": {
                    "gru_accuracy": delayed_gru_accuracy,
                    "mlp_accuracy": delayed_mlp_accuracy,
                    "pass": delayed_competence_pass,
                },
                "immediate": {
                    "gru_accuracy": immediate_gru_accuracy,
                    "mlp_accuracy": immediate_mlp_accuracy,
                    "pass": immediate_competence_pass,
                },
            }
        )

        paired_deltas: list[dict[str, Any]] = []
        for pair in pairs_v1():
            if pair.founder_count != founder_count:
                continue
            delayed_row = by_id[pair.delayed_cell_id]
            immediate_row = by_id[pair.immediate_cell_id]
            delayed_share = delayed_row["metrics"]["final_gru_fraction"]
            immediate_share = immediate_row["metrics"]["final_gru_fraction"]
            delayed_contribution = _endpoint_population_contribution_v1(delayed_row)
            immediate_contribution = _endpoint_population_contribution_v1(
                immediate_row
            )
            delta = delayed_contribution - immediate_contribution
            paired_deltas.append(
                {
                    "seed": pair.seed,
                    "delayed_final_gru_share": delayed_share,
                    "immediate_final_gru_share": immediate_share,
                    "delayed_endpoint_gru_population_contribution": (
                        delayed_contribution
                    ),
                    "immediate_endpoint_gru_population_contribution": (
                        immediate_contribution
                    ),
                    "delayed_minus_immediate": delta,
                }
            )
        deltas = [item["delayed_minus_immediate"] for item in paired_deltas]
        delayed_changes = [
            _endpoint_population_contribution_v1(row)
            - founder_count / INITIAL_POPULATION
            for row in delayed
        ]
        immediate_changes = [
            _endpoint_population_contribution_v1(row)
            - founder_count / INITIAL_POPULATION
            for row in immediate
        ]
        positive_pair_count = sum(delta is not None and delta > 0.0 for delta in deltas)
        delayed_parent_count = sum(_cell_parent_advantage_v1(row) for row in delayed)
        delayed_turnover_offspring_count = sum(
            _cell_turnover_offspring_advantage_v1(row) for row in delayed
        )
        immediate_parent_count = sum(
            _cell_parent_advantage_v1(row) for row in immediate
        )
        immediate_turnover_offspring_count = sum(
            _cell_turnover_offspring_advantage_v1(row) for row in immediate
        )
        recurrent_positive_count = sum(
            float(_pooled_metric_v1(row, COST_METRIC_NAME, "gru")) > 0.0
            for row in delayed
        )
        median_delta = _complete_median_v1(deltas)
        median_delayed_change = _complete_median_v1(delayed_changes)
        median_immediate_change = _complete_median_v1(immediate_changes)
        delayed_pass = (
            delayed_competence_pass
            and positive_pair_count
            >= THRESHOLDS["minimum_supporting_seed_pairs_per_founder_count"]
            and median_delta is not None
            and median_delta
            >= THRESHOLDS["minimum_delayed_minus_immediate_gru_fraction"]
            and median_delayed_change is not None
            and median_delayed_change
            >= THRESHOLDS["minimum_delayed_final_minus_initial_gru_fraction"]
            and delayed_parent_count
            >= THRESHOLDS["minimum_parent_advantage_seeds_per_founder_count"]
            and delayed_turnover_offspring_count
            >= THRESHOLDS["minimum_offspring_advantage_seeds_per_founder_count"]
            and recurrent_positive_count == len(delayed)
        )
        immediate_pass = (
            immediate_competence_pass
            and median_immediate_change is not None
            and median_immediate_change
            <= THRESHOLDS["maximum_immediate_gru_fraction_increase"]
            and immediate_parent_count
            <= THRESHOLDS["maximum_immediate_advantage_seeds_per_founder_count"]
            and immediate_turnover_offspring_count
            <= THRESHOLDS["maximum_immediate_advantage_seeds_per_founder_count"]
        )
        directional_pass = (
            delayed_competence_pass
            and immediate_competence_pass
            and positive_pair_count > len(deltas) // 2
            and median_delta is not None
            and median_delta > 0.0
            and median_delayed_change is not None
            and median_delayed_change > 0.0
            and recurrent_positive_count == len(delayed)
            and immediate_pass
        )
        amplification_by_founder.append(
            {
                "founder_count": founder_count,
                "initial_gru_fraction": founder_count / INITIAL_POPULATION,
                "paired_delayed_minus_immediate": paired_deltas,
                "positive_paired_seed_count": positive_pair_count,
                "directional_delayed_amplification": directional_pass,
                "median_delayed_minus_immediate_final_gru_fraction": median_delta,
                "median_delayed_minus_immediate_endpoint_gru_population_contribution": (
                    median_delta
                ),
                "median_delayed_final_minus_initial_gru_fraction": (
                    median_delayed_change
                ),
                "delayed_parent_advantage_seed_count": delayed_parent_count,
                "delayed_committed_turnover_offspring_advantage_seed_count": (
                    delayed_turnover_offspring_count
                ),
                "delayed_positive_recurrent_cost_seed_count": (
                    recurrent_positive_count
                ),
                "delayed_7_of_9_gate": delayed_pass,
                "median_immediate_final_minus_initial_gru_fraction": (
                    median_immediate_change
                ),
                "immediate_parent_advantage_seed_count": immediate_parent_count,
                "immediate_committed_turnover_offspring_advantage_seed_count": (
                    immediate_turnover_offspring_count
                ),
                "immediate_specificity_gate": immediate_pass,
            }
        )

    zero_rows = [row for row in rows if row["founder_count"] == 0]
    zero_gru_pass = all(
        row["metrics"]["final_gru_count"] == 0
        and all(
            generation["native"][name]["gru"] == 0
            for generation in row["metrics"]["selection_generations"]
            for name in NATIVE_METRIC_NAMES
        )
        for row in zero_rows
    )
    competence_pass = all(
        item[context]["pass"]
        for item in competence_by_founder
        for context in CONTEXTS
    )
    delayed_supporting = sum(
        item["delayed_7_of_9_gate"] for item in amplification_by_founder
    )
    immediate_supporting = sum(
        item["immediate_specificity_gate"] for item in amplification_by_founder
    )
    required_founders = THRESHOLDS["minimum_supporting_nonzero_founder_counts"]
    native_gate_pass = (
        zero_gru_pass
        and competence_pass
        and delayed_supporting >= required_founders
        and immediate_supporting >= required_founders
    )
    directional_supporting = sum(
        item["directional_delayed_amplification"]
        for item in amplification_by_founder
    )
    extinction_cells = []
    for row in rows:
        extinction_cells.append(
            {
                "cell_id": row["cell_id"],
                "mlp_extinction_generation": _family_extinction_generation_v1(
                    row, "mlp"
                ),
                "gru_extinction_generation": _family_extinction_generation_v1(
                    row, "gru"
                ),
                "endpoint_gru_population_contribution": (
                    _endpoint_population_contribution_v1(row)
                ),
            }
        )
    receipt_status = _a_gate_receipt_status_v1(a_gate_receipt)
    decision = _select_decision_v1(
        native_a_gate_pass=native_gate_pass,
        directional_supporting_founder_count=directional_supporting,
        required_supporting_founder_count=required_founders,
        a_gate_receipt_status=receipt_status,
    )
    measurement_inventory = [
        {
            "measurement": name,
            "status": (
                "derived_policy_family_proxy"
                if name == "lineage_extinction_generation"
                else "pending"
            ),
            "reason": (
                "derived from per-generation post-turnover policy-family counts; "
                "individual founder-lineage extinction remains unavailable"
                if name == "lineage_extinction_generation"
                else "not exposed by the native SEL1 selection_metrics mapping"
            ),
        }
        for name in PENDING_NATIVE_MEASUREMENTS
    ]
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "completed",
        "competence": {
            "by_founder_count": competence_by_founder,
            "all_nonzero_founder_contexts_pass": competence_pass,
        },
        "zero_gru_integrity": {
            "cell_count": len(zero_rows),
            "exact_native_zero": zero_gru_pass,
        },
        "founder_fraction_gates": amplification_by_founder,
        "replication_and_specificity": {
            "required_supporting_nonzero_founder_counts": required_founders,
            "delayed_7_of_9_supporting_founder_count": delayed_supporting,
            "immediate_specificity_supporting_founder_count": immediate_supporting,
            "pass": delayed_supporting >= required_founders
            and immediate_supporting >= required_founders,
        },
        "directional_amplification": {
            "required_supporting_nonzero_founder_counts": required_founders,
            "supporting_founder_counts": [
                item["founder_count"]
                for item in amplification_by_founder
                if item["directional_delayed_amplification"]
            ],
            "supporting_founder_count": directional_supporting,
            "pass": directional_supporting >= required_founders,
        },
        "policy_family_extinction": {
            "basis": "alive_post_turnover; generation 0 means absent initially",
            "no_reappearance": True,
            "cells": extinction_cells,
        },
        "native_metric_gate_decision": "pass" if native_gate_pass else "fail",
        "decision": decision,
        "preregistered_a_decision": (
            "pass" if decision["a_eligible"] else "fail_closed"
        ),
        "measurement_inventory": measurement_inventory,
        "pending_non_native_measurements": [
            item for item in measurement_inventory if item["status"] == "pending"
        ],
    }


def execute_prepared_b72_v1(
    module: Any,
    prepared: PreparedSel1B72V1,
    *,
    run_id: str,
    execute_fn: Callable[..., Mapping[str, Any]] | None = None,
    digest_fn: Callable[[Any], Sequence[str]] | None = None,
) -> Mapping[str, Any]:
    """Execute exactly one native dispatch and return its compact cell report."""

    if not isinstance(prepared, PreparedSel1B72V1):
        raise Sel1HarnessError("prepared must be PreparedSel1B72V1")
    _validate_preparation_receipt_v1(prepared.preparation_receipt)
    if type(run_id) is not str or not run_id.strip():
        raise Sel1HarnessError("run_id must be a nonempty string")
    if execute_fn is None:
        gmnw = importlib.import_module(
            "engine.perf.cuda_architecture_b_direct_resident_gmnw_gru_v1"
        )
        execute_fn = gmnw.execute_direct_resident_gmnw_gru_v1
    if digest_fn is None:
        matrix_runner = importlib.import_module(
            "tools.scripts.eiel_gmnw_capability_matrix_runner_v1"
        )
        digest_fn = matrix_runner.authoritative_state_digests_by_cell_v1

    dispatch_calls = 1
    started = time.perf_counter_ns()
    raw_receipt = execute_fn(
        module,
        prepared.native,
        logical_generation_count=GENERATIONS,
        run_id=run_id,
    )
    if not isinstance(raw_receipt, Mapping):
        raise Sel1HarnessError("native dispatch returned a non-object receipt")
    native_receipt = validate_native_dispatch_receipt_v1(
        raw_receipt,
        dispatch_calls=dispatch_calls,
    )
    rows = _result_rows_from_prepared_v1(
        prepared,
        state_digests=tuple(digest_fn(prepared.native)),
        native_receipt=native_receipt,
    )
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": contract_v1()["contract_sha256"],
        "run_id": run_id,
        "run_label": "gmnw-sel1-b72g12t80",
        "physical_batch_cells": BATCH_CELLS,
        "logical_worlds": BATCH_CELLS,
        "logical_chunking": False,
        "chunk_count": 1,
        "native_dispatches": 1,
        "generations": GENERATIONS,
        "ticks_per_generation": TICKS_PER_GENERATION,
        "preparation_receipt": dict(prepared.preparation_receipt),
        "native_receipt": native_receipt,
        "repeat_cell_ids": list(repeat_cell_ids_v1()),
        "continuation_cell_ids": list(continuation_cell_ids_v1()),
        "results": list(rows),
        "analysis": analyze_result_rows_v1(rows),
        "wall_time_ns": time.perf_counter_ns() - started,
    }


def execute_physical_b72_v1(
    *,
    extension: Path,
    witness: Path,
    mlp: Path,
    source_projection_root: Path,
    run_root: Path,
    run_id: str,
) -> Mapping[str, Any]:
    """Load, prepare, dispatch, and persist one concrete physical b72g12t80 run."""

    resolved_extension = extension.expanduser().resolve()
    if not resolved_extension.is_file():
        raise Sel1HarnessError(f"CUDA extension is missing: {resolved_extension}")
    admission = importlib.import_module(
        "tools.scripts.eiel_gmnw_physical_batch_admission_v1"
    )
    gate = importlib.import_module("tools.scripts.eiel_cuda_architecture_b_gru_gate_v3")
    artifact = admission.validate_immutable_cuda_artifact_v1(resolved_extension)
    module = gate._load_cuda_module(resolved_extension)
    prepared = prepare_physical_b72_v1(
        module,
        source_projection_root=source_projection_root,
        run_root=run_root,
        witness=witness,
        mlp=mlp,
    )
    report = dict(execute_prepared_b72_v1(module, prepared, run_id=run_id))
    report["artifacts"] = {
        "extension": {"path": str(resolved_extension), **dict(artifact)},
        "witness": {
            "path": str(witness.expanduser().resolve()),
            "sha256": _file_sha256(witness.expanduser().resolve()),
        },
        "mlp": {
            "path": str(mlp.expanduser().resolve()),
            "sha256": _file_sha256(mlp.expanduser().resolve()),
        },
    }
    resolved_run_root = run_root.expanduser().resolve()
    resolved_run_root.mkdir(parents=True, exist_ok=True)
    output = resolved_run_root / "sel1_b72_result.json"
    report["result_path"] = str(output)
    output.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return report


def execute_sel1_v1(**kwargs: Any) -> Mapping[str, Any]:
    """Compatibility name for the concrete complete-run API."""

    required = {
        "extension",
        "witness",
        "mlp",
        "source_projection_root",
        "run_root",
        "run_id",
    }
    missing = sorted(required - set(kwargs))
    if missing:
        raise Sel1HarnessError("execution inputs are not wired: " + ", ".join(missing))
    return execute_physical_b72_v1(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate-contract", action="store_true")
    action.add_argument("--check-engine-ready", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--print-contract", action="store_true")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--witness", type=Path)
    parser.add_argument("--mlp", type=Path)
    parser.add_argument("--source-projection-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.print_contract:
            result = contract_v1()
        elif args.check_engine_ready:
            result = engine_readiness_v1()
        elif args.execute:
            values = {
                "extension": args.extension,
                "witness": args.witness,
                "mlp": args.mlp,
                "source_projection_root": args.source_projection_root,
                "run_root": args.run_root,
                "run_id": args.run_id,
            }
            missing = sorted(name for name, value in values.items() if value is None)
            if missing:
                raise Sel1HarnessError(
                    "execution inputs are not wired: " + ", ".join(missing)
                )
            result = execute_sel1_v1(**values)
        else:
            value = None
            if args.contract is not None:
                loaded = json.loads(args.contract.read_text(encoding="ascii"))
                if not isinstance(loaded, Mapping):
                    raise Sel1HarnessError("contract JSON must be an object")
                value = loaded
            contract = validate_contract_v1(value)
            result = {
                "ok": True,
                "schema_version": SCHEMA_VERSION,
                "contract_sha256": contract["contract_sha256"],
                "cells": len(contract["cells"]),
                "pairs": len(contract["pairs"]),
            }
    except Exception as exc:  # noqa: BLE001 - fail-closed machine receipt.
        result = {
            "ok": False,
            "schema_version": SCHEMA_VERSION,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
