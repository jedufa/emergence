"""Typed Patch configuration adapters for mixed cross-path inheritance batches.

The immutable provider registry owns public parameter names and native integer
IDs.  This module translates those declarations into per-cell resident
controls and configures the existing Patch runner construction boundary.  It
does not execute a tensor, choose parents, or implement inheritance.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

from engine.env.gridworld_pkg.constants import ACT_REST

from .patch_runner_factory import (
    PATCH_FIXED_PARAMETER_NAMES_V1,
    PatchFixedRunnerBuildV1,
    PatchRunnerFactoryError,
    build_patch_fixed_runner_v1,
    derive_patch_master_seed_v1,
    prepare_patch_generation_zero_v1,
)


ADAPTER_SCHEMA_VERSION: Final[str] = "eiel_inheritance_cross_path_patch_adapters_v1"
CELL_CONTROL_SCHEMA_VERSION: Final[str] = "eiel_inheritance_cross_path_cell_controls_v1"

GENERATION_BOUNDARY_REFILL: Final[str] = "generation_boundary_refill"
ORDINARY_LIVE_BIRTH: Final[str] = "ordinary_live_birth"
GENOTYPE_BLIND_NEUTRAL: Final[str] = "genotype_blind_neutral"

_ROLE_SUFFIXES: Final[dict[str, str]] = {
    "inheritance_mode": ".inheritance_mode",
    "founder_pattern": ".founder_pattern",
    "reproduction_path": ".reproduction_path",
    "mutation_enabled": ".mutation_enabled",
    "neutral_fixture": ".neutral_fixture",
    "mutation_weather_enabled": ".mutation_weather_enabled",
    "selective_mating_enabled": ".selective_mating_enabled",
    "ecological_trait_authority_enabled": ".ecological_trait_authority_enabled",
    "family_head_authority_enabled": ".family_head_authority_enabled",
    "modular_output_authority_enabled": ".modular_output_authority_enabled",
}
_PER_CELL_NATIVE_ROLES: Final[frozenset[str]] = frozenset(
    {"inheritance_mode", "founder_pattern", "reproduction_path"}
)


class InheritanceCrossPathPatchAdapterError(ValueError):
    """The registry or cell cannot be translated without changing semantics."""


@dataclass(frozen=True, slots=True)
class SemanticAxisV1:
    """One registry-owned semantic axis and its public parameter binding."""

    role: str
    semantic: str
    parameter_name: str
    value_type: str
    ids: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class NeutralSelectionControlV1:
    """Selection policy without ownership of production eligibility."""

    mode: str
    algorithm: str
    eligible_parent_set_owner: str
    eligible_parent_set_override: bool
    uniform_over_eligible_set: bool
    deterministic_schedule_required: bool
    genotype_blind: bool
    lineage_blind: bool
    policy_blind: bool
    strategy_blind: bool


@dataclass(frozen=True, slots=True)
class CrossPathCellControlV1:
    """Tombstone row for the retired standalone cross-path controller."""

    schema_version: str
    cell_id: str
    semantic_values: Mapping[str, Any]
    semantic_ids: Mapping[str, int]
    inheritance_mode: str
    founder_pattern: str
    reproduction_path: str
    inheritance_mode_id: int
    founder_pattern_id: int
    reproduction_path_id: int
    inheritance_donor_draw_budget: int
    generation_boundary_refill_enabled: bool
    ordinary_live_birth_enabled: bool
    mutation_enabled: bool
    neutral_fixture_enabled: bool
    mutation_weather_enabled: bool
    selective_mating_enabled: bool
    ecological_trait_authority_enabled: bool
    family_head_authority_enabled: bool
    modular_output_authority_enabled: bool
    mate_required_for_ordinary_birth: bool
    selection: NeutralSelectionControlV1

    def __post_init__(self) -> None:
        raise InheritanceCrossPathPatchAdapterError(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "cross-path cell-control construction is disabled"
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cell_id": self.cell_id,
            "semantic_values": dict(self.semantic_values),
            "semantic_ids": dict(self.semantic_ids),
            "inheritance_mode": self.inheritance_mode,
            "founder_pattern": self.founder_pattern,
            "reproduction_path": self.reproduction_path,
            "inheritance_mode_id": self.inheritance_mode_id,
            "founder_pattern_id": self.founder_pattern_id,
            "reproduction_path_id": self.reproduction_path_id,
            "inheritance_donor_draw_budget": (self.inheritance_donor_draw_budget),
            # Both fields are deliberately serialized.  The false field is a
            # write-isolation assertion, not an omitted default.
            "generation_boundary_refill_enabled": (
                self.generation_boundary_refill_enabled
            ),
            "ordinary_live_birth_enabled": self.ordinary_live_birth_enabled,
            "mutation_enabled": self.mutation_enabled,
            "neutral_fixture_enabled": self.neutral_fixture_enabled,
            "mutation_weather_enabled": self.mutation_weather_enabled,
            "selective_mating_enabled": self.selective_mating_enabled,
            "ecological_trait_authority_enabled": (
                self.ecological_trait_authority_enabled
            ),
            "family_head_authority_enabled": (self.family_head_authority_enabled),
            "modular_output_authority_enabled": (self.modular_output_authority_enabled),
            "mate_required_for_ordinary_birth": (self.mate_required_for_ordinary_birth),
            "selection": {
                field: getattr(self.selection, field)
                for field in self.selection.__dataclass_fields__
            },
        }


@dataclass(frozen=True, slots=True)
class CrossPathPatchRunnerBuildV1:
    """Configured real Patch runner plus registry-derived control receipt."""

    runner: Any
    receipt: Mapping[str, Any]
    cell_control: CrossPathCellControlV1


class CrossPathPatchRunnerAdapterV1:
    """Tombstone for the retired standalone Patch runner adapter."""

    def __init__(
        self,
        registry: Mapping[str, Any],
        *,
        runner_constructor: Callable[..., Any] | None = None,
    ) -> None:
        raise InheritanceCrossPathPatchAdapterError(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "cross-path runner-adapter construction is disabled"
        )
        # Historical implementation retained temporarily for evidence reading.
        snapshot = copy.deepcopy(dict(registry))
        # Validate all roles and selection during binding, before any runner is
        # created.  Canonical JSON also rejects non-declarative registry values.
        _axis_by_role(snapshot)
        _selection_control(snapshot)
        try:
            canonical = json.dumps(
                snapshot,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise InheritanceCrossPathPatchAdapterError(
                "provider registry is not canonical JSON data"
            ) from exc
        self._registry = snapshot
        self.registry_sha256 = hashlib.sha256(canonical).hexdigest()
        if runner_constructor is None:
            from engine.simulation.runner import SimulationRunner

            runner_constructor = SimulationRunner
        if not callable(runner_constructor):
            raise InheritanceCrossPathPatchAdapterError(
                "runner_constructor must be callable"
            )
        self._runner_constructor = runner_constructor

    def __call__(
        self,
        cell: Mapping[str, Any],
        run_dir: str | Path,
        *,
        generations: int,
        ticks_per_generation: int,
        artifact_path: str,
        qualification_receipt_path: str,
        reporting_receipt_path: str,
        max_births_per_generation: int = 64,
    ) -> CrossPathPatchRunnerBuildV1:
        built = build_cross_path_patch_runner_v1(
            cell,
            self._registry,
            run_dir,
            generations=generations,
            ticks_per_generation=ticks_per_generation,
            artifact_path=artifact_path,
            qualification_receipt_path=qualification_receipt_path,
            reporting_receipt_path=reporting_receipt_path,
            max_births_per_generation=max_births_per_generation,
            runner_constructor=self._runner_constructor,
        )
        return CrossPathPatchRunnerBuildV1(
            runner=built.runner,
            receipt={
                **dict(built.receipt),
                "provider_registry_sha256": self.registry_sha256,
            },
            cell_control=built.cell_control,
        )


def _canonical_token(value: Any, value_type: str, *, field: str) -> str:
    if value_type == "string" and isinstance(value, str) and value:
        return value
    if value_type == "boolean" and type(value) is bool:
        return "true" if value else "false"
    if value_type == "integer" and type(value) is int:
        return str(value)
    if value_type == "number" and type(value) in {int, float}:
        return str(value)
    raise InheritanceCrossPathPatchAdapterError(
        f"{field} has the wrong registry-declared type"
    )


def _axis_by_role(
    registry: Mapping[str, Any],
) -> dict[str, SemanticAxisV1]:
    bindings = registry.get("parameter_bindings")
    semantic_types = registry.get("semantic_types")
    semantic_ids = registry.get("semantic_ids")
    if not all(
        isinstance(value, Mapping) for value in (bindings, semantic_types, semantic_ids)
    ):
        raise InheritanceCrossPathPatchAdapterError(
            "provider registry lacks semantic bindings, types, or IDs"
        )
    result: dict[str, SemanticAxisV1] = {}
    for role, suffix in _ROLE_SUFFIXES.items():
        matches = [
            str(semantic)
            for semantic, parameter in bindings.items()
            if isinstance(semantic, str)
            and isinstance(parameter, str)
            and parameter.endswith(suffix)
        ]
        if len(matches) > 1 or (role in _PER_CELL_NATIVE_ROLES and len(matches) != 1):
            raise InheritanceCrossPathPatchAdapterError(
                f"provider registry has an ambiguous or missing {role} semantic"
            )
        if not matches:
            continue
        semantic = matches[0]
        value_type = semantic_types.get(semantic)
        ids = semantic_ids.get(semantic)
        if (
            value_type not in {"string", "boolean", "integer", "number"}
            or not isinstance(ids, Mapping)
            or not ids
            or any(
                not isinstance(token, str)
                or type(identifier) is not int
                or identifier < 0
                for token, identifier in ids.items()
            )
            or len(set(ids.values())) != len(ids)
        ):
            raise InheritanceCrossPathPatchAdapterError(
                f"provider registry has invalid {role} semantic IDs"
            )
        result[role] = SemanticAxisV1(
            role=role,
            semantic=semantic,
            parameter_name=str(bindings[semantic]),
            value_type=str(value_type),
            ids={str(token): int(identifier) for token, identifier in ids.items()},
        )
    return result


def semantic_axes_v1(
    registry: Mapping[str, Any],
) -> Mapping[str, SemanticAxisV1]:
    """Resolve registry-declared native axes, never campaign names.

    Core mixed-batch roles are mandatory.  Constant arm switches may instead
    live in ``fixed_parameters`` and therefore have no native integer axis.
    """

    return _axis_by_role(registry)


def _cell_parts(
    cell: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any], Mapping[str, int]]:
    cell_id = cell.get("cell_id")
    parameters = cell.get("parameters")
    seeds = cell.get("seeds")
    if (
        not isinstance(cell_id, str)
        or not cell_id
        or not isinstance(parameters, Mapping)
        or not isinstance(seeds, Mapping)
    ):
        raise InheritanceCrossPathPatchAdapterError(
            "cross-path cell lacks an ID, parameters, or named seeds"
        )
    if any(
        not isinstance(owner, str) or not owner or type(seed) is not int or seed < 0
        for owner, seed in seeds.items()
    ):
        raise InheritanceCrossPathPatchAdapterError(
            "cross-path named seeds are invalid"
        )
    return cell_id, parameters, seeds


def _selection_control(registry: Mapping[str, Any]) -> NeutralSelectionControlV1:
    identity = registry.get("runtime_identity")
    selection_mode = (
        identity.get("selection_mode") if isinstance(identity, Mapping) else None
    )
    if selection_mode != GENOTYPE_BLIND_NEUTRAL:
        raise InheritanceCrossPathPatchAdapterError(
            "Patch cross-path adapter supports only genotype_blind_neutral selection"
        )
    return NeutralSelectionControlV1(
        mode=selection_mode,
        algorithm="uniform_deterministic_over_production_eligible_set_v1",
        eligible_parent_set_owner="production_reproduction_lifecycle",
        eligible_parent_set_override=False,
        uniform_over_eligible_set=True,
        deterministic_schedule_required=True,
        genotype_blind=True,
        lineage_blind=True,
        policy_blind=True,
        strategy_blind=True,
    )


def build_cell_control_v1(
    cell: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> CrossPathCellControlV1:
    """Reject construction of retired standalone cross-path controls."""

    raise InheritanceCrossPathPatchAdapterError(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "cross-path cell-control construction is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    cell_id, parameters, _seeds = _cell_parts(cell)
    axes = _axis_by_role(registry)
    semantic_values: dict[str, Any] = {}
    semantic_ids: dict[str, int] = {}
    role_values: dict[str, Any] = {}
    for role, axis in axes.items():
        if axis.parameter_name not in parameters:
            raise InheritanceCrossPathPatchAdapterError(
                f"cell {cell_id} lacks registry parameter {axis.parameter_name}"
            )
        value = parameters[axis.parameter_name]
        token = _canonical_token(value, axis.value_type, field=axis.parameter_name)
        if token not in axis.ids:
            raise InheritanceCrossPathPatchAdapterError(
                f"cell {cell_id} uses unsupported {role} value"
            )
        semantic_values[axis.semantic] = value
        semantic_ids[axis.semantic] = axis.ids[token]
        role_values[role] = value

    fixed = registry.get("fixed_parameters")
    if not isinstance(fixed, Mapping):
        raise InheritanceCrossPathPatchAdapterError(
            "provider registry fixed_parameters must be an object"
        )
    for role, suffix in _ROLE_SUFFIXES.items():
        if role in role_values:
            continue
        matches = [
            str(parameter_name)
            for parameter_name in fixed
            if isinstance(parameter_name, str) and parameter_name.endswith(suffix)
        ]
        if len(matches) != 1:
            raise InheritanceCrossPathPatchAdapterError(
                f"provider registry must fix or bind exactly one {role} parameter"
            )
        parameter_name = matches[0]
        expected = fixed[parameter_name]
        if parameters.get(parameter_name) != expected:
            raise InheritanceCrossPathPatchAdapterError(
                f"cell {cell_id} differs from fixed {role}"
            )
        role_values[role] = expected

    path = role_values["reproduction_path"]
    if path not in {GENERATION_BOUNDARY_REFILL, ORDINARY_LIVE_BIRTH}:
        raise InheritanceCrossPathPatchAdapterError(
            "registry reproduction path is unsupported by the Patch adapter"
        )
    required_switches = {
        "mutation_enabled": False,
        "neutral_fixture": True,
        "mutation_weather_enabled": False,
        "selective_mating_enabled": False,
        "ecological_trait_authority_enabled": False,
        "family_head_authority_enabled": False,
        "modular_output_authority_enabled": False,
    }
    for role, expected in required_switches.items():
        if role_values[role] is not expected:
            raise InheritanceCrossPathPatchAdapterError(
                f"Patch neutral cross-path adapter requires {role}={expected}"
            )

    inheritance_mode = role_values["inheritance_mode"]
    founder_pattern = role_values["founder_pattern"]
    if not isinstance(inheritance_mode, str) or not isinstance(founder_pattern, str):
        raise InheritanceCrossPathPatchAdapterError(
            "Patch inheritance mode and founder pattern must be strings"
        )
    inheritance_semantic = registry.get("inheritance_semantic")
    inheritance_axis = axes["inheritance_mode"]
    budgets = registry.get("draw_budgets")
    inheritance_token = _canonical_token(
        inheritance_mode,
        inheritance_axis.value_type,
        field=inheritance_axis.parameter_name,
    )
    if (
        inheritance_semantic != inheritance_axis.semantic
        or not isinstance(budgets, Mapping)
        or set(budgets) != set(inheritance_axis.ids)
        or type(budgets.get(inheritance_token)) is not int
        or budgets[inheritance_token] < 0
    ):
        raise InheritanceCrossPathPatchAdapterError(
            "registry inheritance semantic or donor draw budget is invalid"
        )

    return CrossPathCellControlV1(
        schema_version=CELL_CONTROL_SCHEMA_VERSION,
        cell_id=cell_id,
        semantic_values=semantic_values,
        semantic_ids=semantic_ids,
        inheritance_mode=inheritance_mode,
        founder_pattern=founder_pattern,
        reproduction_path=path,
        inheritance_mode_id=semantic_ids[axes["inheritance_mode"].semantic],
        founder_pattern_id=semantic_ids[axes["founder_pattern"].semantic],
        reproduction_path_id=semantic_ids[axes["reproduction_path"].semantic],
        inheritance_donor_draw_budget=budgets[inheritance_token],
        generation_boundary_refill_enabled=path == GENERATION_BOUNDARY_REFILL,
        ordinary_live_birth_enabled=path == ORDINARY_LIVE_BIRTH,
        mutation_enabled=False,
        neutral_fixture_enabled=True,
        mutation_weather_enabled=False,
        selective_mating_enabled=False,
        ecological_trait_authority_enabled=False,
        family_head_authority_enabled=False,
        modular_output_authority_enabled=False,
        # A turnover-only cell still carries the requirement so enabling its
        # dormant ordinary path cannot silently revert to asexual births.
        mate_required_for_ordinary_birth=True,
        selection=_selection_control(registry),
    )


def build_mixed_batch_controls_v1(
    cells: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> tuple[CrossPathCellControlV1, ...]:
    """Reject construction of retired standalone mixed-batch controls."""

    raise InheritanceCrossPathPatchAdapterError(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "cross-path mixed-batch control construction is disabled"
    )


def _patch_factory_cell(
    cell: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    cell_id, parameters, seeds = _cell_parts(cell)
    fixed = registry.get("fixed_parameters")
    fixed = fixed if isinstance(fixed, Mapping) else {}
    patch_parameters: dict[str, Any] = {}
    for name in PATCH_FIXED_PARAMETER_NAMES_V1:
        value = parameters.get(name, fixed.get(name))
        if value is None:
            raise InheritanceCrossPathPatchAdapterError(
                f"cell {cell_id} lacks Patch factory parameter {name}"
            )
        patch_parameters[name] = value
    return {
        "cell_id": cell_id,
        "parameters": patch_parameters,
        "seeds": dict(seeds),
    }


def _configure_patch_contract(
    *,
    env_cfg: Any,
    sim_cfg: Any,
    control: CrossPathCellControlV1,
) -> None:
    # The real production birth boundary must see a mate-required contract even
    # in cells whose ordinary path is dormant.
    env_cfg.reproduce_requires_mate = True

    # Existing generic fixed-action controls are used without enabling the old
    # RECOMB-72 turnover owner.  Native preparation must project and verify
    # these values before granting neutral-route authority.
    sim_cfg.recomb72_neutral_actions_enable = True
    sim_cfg.recomb72_neutral_action_id = int(ACT_REST)
    sim_cfg.recomb72_neutral_environment_enable = True
    sim_cfg.recomb72_audit_enable = False
    sim_cfg.recomb72_disable_ordinary_tick_births = (
        not control.ordinary_live_birth_enabled
    )
    sim_cfg.recomb72_mutation_off = True

    # Preserve the qualified Patch import ABI.  The independent neutral-action
    # switch above returns ACT_REST before policy sampling.
    sim_cfg.policy_deterministic_actions = False
    sim_cfg.policy_tick_stride = 1
    sim_cfg.policy_weight_mutation_enable = False
    sim_cfg.mutation_weather_enabled = False
    sim_cfg.genome_c1_enable = False
    sim_cfg.genome_c2_enable = False
    # The representation remains present and inheritable; fixed actions and
    # the neutral environment remove its behavioral/ecological authority.
    sim_cfg.resource_handling_representation_enable = True
    sim_cfg.resource_handling_representation_reporting_enable = True
    sim_cfg.policy_family_head_residual_authority_enable = False
    sim_cfg.policy_family_head_residual_mutation_enable = False
    sim_cfg.policy_family_head_shadow_enable = False
    sim_cfg.native_runtime_modular_storage_enable = False

    # Keep the qualified Patch P2 observation/layout flags intact.  The host
    # source runner executes no ticks; the resident owner must honor the
    # explicit neutral-environment cell control before dispatch.
    env_cfg.enable_biomes = False
    env_cfg.enable_combat = False
    env_cfg.enable_injury = False
    env_cfg.enable_scar = False
    env_cfg.food_spawn_per_step = 0
    env_cfg.max_food = 0
    env_cfg.move_cost = 0.0
    env_cfg.rest_cost = 0.0
    env_cfg.gather_cost = 0.0
    env_cfg.attack_cost = 0.0
    env_cfg.food_energy = 0.0


def build_cross_path_patch_runner_v1(
    cell: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_dir: str | Path,
    *,
    generations: int,
    ticks_per_generation: int,
    artifact_path: str,
    qualification_receipt_path: str,
    reporting_receipt_path: str,
    max_births_per_generation: int = 64,
    runner_constructor: Callable[..., Any],
) -> CrossPathPatchRunnerBuildV1:
    """Reject construction of the retired standalone Patch runner."""

    raise InheritanceCrossPathPatchAdapterError(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "cross-path Patch runner construction is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    cell_id, _parameters, seeds = _cell_parts(cell)
    control = build_cell_control_v1(cell, registry)
    patch_cell = _patch_factory_cell(cell, registry)

    def configured_constructor(**kwargs: Any) -> Any:
        _configure_patch_contract(
            env_cfg=kwargs["env_cfg"],
            sim_cfg=kwargs["sim_cfg"],
            control=control,
        )
        return runner_constructor(**kwargs)

    try:
        base: PatchFixedRunnerBuildV1 = build_patch_fixed_runner_v1(
            patch_cell,
            run_dir,
            generations=generations,
            ticks_per_generation=ticks_per_generation,
            artifact_path=artifact_path,
            qualification_receipt_path=qualification_receipt_path,
            reporting_receipt_path=reporting_receipt_path,
            max_births_per_generation=max_births_per_generation,
            runner_constructor=configured_constructor,
        )
    except PatchRunnerFactoryError as exc:
        raise InheritanceCrossPathPatchAdapterError(str(exc)) from exc

    # A real SimulationRunner is initialized only after every cross-path flag is
    # installed.  Injectable test constructors intentionally have no lifecycle.
    from engine.simulation.runner import SimulationRunner

    if isinstance(base.runner, SimulationRunner):
        prepare_patch_generation_zero_v1(base.runner)
        from .inheritance_cross_path_source_materializer_v1 import (
            apply_inheritance_cross_path_founders_v1,
        )

        apply_inheritance_cross_path_founders_v1(
            base.runner,
            cell_id=cell_id,
            founder_pattern=control.founder_pattern,
            selection_seed=derive_patch_master_seed_v1(dict(seeds)),
        )
    setattr(
        base.runner,
        "_inheritance_cross_path_cell_control_v1",
        control.payload(),
    )

    receipt = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "patch_factory_receipt": dict(base.receipt),
        "cell_control": control.payload(),
        "configuration": {
            "mate_required": True,
            "neutral_fixed_action": int(ACT_REST),
            "production_eligible_parent_set_preserved": True,
            "host_source_device": "cpu",
            "execution_device": "cuda",
            "mutation_authority": False,
            "mutation_enforcement_owner": "resident_cell_control",
            "ecological_trait_authority": False,
            "family_head_authority": False,
            "modular_output_authority": False,
        },
    }
    return CrossPathPatchRunnerBuildV1(
        runner=base.runner,
        receipt=receipt,
        cell_control=control,
    )


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "CELL_CONTROL_SCHEMA_VERSION",
    "CrossPathCellControlV1",
    "CrossPathPatchRunnerAdapterV1",
    "CrossPathPatchRunnerBuildV1",
    "InheritanceCrossPathPatchAdapterError",
    "NeutralSelectionControlV1",
    "SemanticAxisV1",
    "build_cell_control_v1",
    "build_cross_path_patch_runner_v1",
    "build_mixed_batch_controls_v1",
    "semantic_axes_v1",
]
