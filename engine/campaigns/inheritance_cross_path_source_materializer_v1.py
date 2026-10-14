"""Typed source materialization for the Patch cross-path resident overlay.

The materializer consumes a versioned source authority embedded in each
canonical Patch checkpoint projection.  That authority carries the hereditary
state which is not represented by the scalar resident banks (including
resource and family-head records), so unsupported or incomplete sources fail
closed instead of producing linkage codes detached from the actual founders.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import torch

from engine.agents.genome import Genome
from engine.evolution.inheritance_linkage_digest_table_v1 import (
    LinkageDigestTableV1,
    build_linkage_digest_table_v1,
    founder_codes_v1,
)
from engine.evolution.inheritance_registry_v1 import canonical_parent_order
from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
    PreparedDirectResidentPatchMulticellV1,
)
from engine.perf.cuda_architecture_b_resident_reproduction_exposure_sidecar_v1 import (
    STORAGE_MODE_COMPACT_COMPLETE,
    STORAGE_MODE_EXACT_FULL,
    STORAGE_MODE_CODES,
    ResidentReproductionExposureShapeV1,
)

from .inheritance_cross_path_patch_adapters import (
    CELL_CONTROL_SCHEMA_VERSION,
    CrossPathCellControlV1,
)
from .inheritance_cross_path_patch_execution_v1 import (
    CrossPathBatchSourceMaterializationV1,
)


SCHEMA_VERSION: Final[str] = "eiel_inheritance_cross_path_source_materializer_v1"
SOURCE_AUTHORITY_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_source_authority_v1"
)
FOUNDER_PATTERN_REGISTRY_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_founder_patterns_v1"
)
SOURCE_AUTHORITY_FIELD: Final[str] = "inheritance_cross_path_source_authority_v1"
FOUNDER_ASSIGNMENT_VERSION: Final[str] = "balanced_sorted_agent_id_two_founders_v1"

_AUDITED_PATTERN_FIELDS: Final[tuple[str, ...]] = (
    "exploration_bias",
    "aggression_bias",
    "action_temperature",
    "resource_handling_axis",
    "drive_inertia",
    "pred_lr",
)
_AUDITED_RESIDENT_GENOME_BANKS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "exploration_bias": "genome_exploration_bias",
        "aggression_bias": "genome_aggression_bias",
        "action_temperature": "genome_action_temperature",
        "drive_inertia": "genome_drive_inertia",
        "pred_lr": "genome_pred_lr",
    }
)
_NEUTRAL_PATTERN_VALUES: Final[Mapping[str, float]] = MappingProxyType(
    {
        "exploration_bias": 0.0,
        "aggression_bias": 0.0,
        "action_temperature": 1.0,
        "resource_handling_axis": 0.0,
        "drive_inertia": 0.85,
        "pred_lr": 0.08,
    }
)


@dataclass(frozen=True, slots=True)
class FounderPatternDefinitionV1:
    """Registry-owned expected A/B values for one founder pattern."""

    pattern_id: str
    parent_a_values: Mapping[str, float]
    parent_b_values: Mapping[str, float]


def _pattern_values(**overrides: float) -> Mapping[str, float]:
    values = dict(_NEUTRAL_PATTERN_VALUES)
    values.update(overrides)
    return MappingProxyType(values)


FOUNDER_PATTERN_REGISTRY_V1: Final[Mapping[str, FounderPatternDefinitionV1]] = (
    MappingProxyType(
        {
            "single_extreme_scalar": FounderPatternDefinitionV1(
                pattern_id="single_extreme_scalar",
                parent_a_values=_pattern_values(exploration_bias=-1.0),
                parent_b_values=_pattern_values(exploration_bias=1.0),
            ),
            "resource_specialization_extremes": FounderPatternDefinitionV1(
                pattern_id="resource_specialization_extremes",
                parent_a_values=_pattern_values(resource_handling_axis=-1.0),
                parent_b_values=_pattern_values(resource_handling_axis=1.0),
            ),
            "correlated_multi_trait_packages": FounderPatternDefinitionV1(
                pattern_id="correlated_multi_trait_packages",
                parent_a_values=MappingProxyType(
                    {
                        "exploration_bias": -1.0,
                        "aggression_bias": -2.0,
                        "action_temperature": 0.25,
                        "resource_handling_axis": -1.0,
                        "drive_inertia": 0.0,
                        "pred_lr": 0.0,
                    }
                ),
                parent_b_values=MappingProxyType(
                    {
                        "exploration_bias": 1.0,
                        "aggression_bias": 2.0,
                        "action_temperature": 2.5,
                        "resource_handling_axis": 1.0,
                        "drive_inertia": 0.999,
                        "pred_lr": 1.0,
                    }
                ),
            ),
        }
    )
)


class InheritanceCrossPathSourceMaterializationError(ValueError):
    """A Patch source cannot be translated without inventing authority."""


def _fail(message: str) -> InheritanceCrossPathSourceMaterializationError:
    return InheritanceCrossPathSourceMaterializationError(message)


def _exact_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(f"{name} must be an exact integer >= {minimum}")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise _fail(f"{name} must be a finite number")
    return float(value)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{name} must be a mapping")
    return value


def apply_inheritance_cross_path_founders_v1(
    runner: Any,
    *,
    cell_id: str,
    founder_pattern: str,
    selection_seed: int,
) -> Mapping[str, Any]:
    """Install a balanced two-founder authority without consuming RNG.

    The population size is deliberately not fixed here. An even live
    population is divided by stable agent ID, so later campaigns can reuse the
    same source owner with another admitted capacity.
    """

    if type(cell_id) is not str or not cell_id:
        raise _fail("source-authority cell_id must be nonempty")
    definition = FOUNDER_PATTERN_REGISTRY_V1.get(founder_pattern)
    if definition is None:
        raise _fail(
            f"founder pattern is not registered in "
            f"{FOUNDER_PATTERN_REGISTRY_VERSION}: {founder_pattern!r}"
        )
    _exact_int(selection_seed, name="selection_seed")
    if selection_seed >= 1 << 64:
        raise _fail("selection_seed exceeds uint64")
    population = getattr(runner, "pop", None)
    alive_agents = getattr(population, "alive_agents", None)
    if not callable(alive_agents):
        raise _fail("runner lacks a live population source")
    agents = sorted(
        tuple(alive_agents()),
        key=lambda agent: int(getattr(agent, "agent_id", -1)),
    )
    if (
        len(agents) < 2
        or len(agents) % 2 != 0
        or any(int(getattr(agent, "agent_id", -1)) < 0 for agent in agents)
        or len({int(agent.agent_id) for agent in agents}) != len(agents)
    ):
        raise _fail(
            "balanced two-founder authority requires an even live population "
            "with unique nonnegative agent IDs"
        )
    baseline_genome = getattr(agents[0], "genome", None)
    baseline_policy = getattr(agents[0], "policy", None)
    banked_policy = getattr(runner, "_mlp_pack", None)
    genome_types: tuple[type[Any], ...] = (Genome,)
    try:
        from agents.genome import Genome as RuntimeGenome  # noqa: PLC0415

        if RuntimeGenome is not Genome:
            genome_types = (*genome_types, RuntimeGenome)
    except ImportError:
        pass
    if not isinstance(baseline_genome, genome_types) or (
        baseline_policy is None
        and not isinstance(banked_policy, torch.Tensor)
    ):
        raise _fail(
            "founder source lacks a canonical Genome or policy authority"
        )

    founders: list[Genome] = []
    for values in (definition.parent_a_values, definition.parent_b_values):
        founder = copy.deepcopy(baseline_genome)
        for field, value in values.items():
            setattr(founder, field, float(value))
        founder.mutation_rate = 0.0
        founder.refresh_canonical_payload_cache()
        founders.append(founder)

    split = len(agents) // 2
    ordinal_by_agent_id: dict[int, int] = {}
    for index, agent in enumerate(agents):
        ordinal = 0 if index < split else 1
        agent.genome = copy.deepcopy(founders[ordinal])
        if baseline_policy is not None:
            agent.policy = copy.deepcopy(baseline_policy)
        ordinal_by_agent_id[int(agent.agent_id)] = ordinal

    mark_dirty = getattr(runner, "_mark_bank_dirty", None)
    if callable(mark_dirty):
        mark_dirty()
    sync = getattr(runner, "_phase3_pre_start_generation_sync", None)
    if callable(sync):
        sync(agents)

    sim_cfg = getattr(runner, "sim_cfg", None)
    birth_capacity = getattr(sim_cfg, "cuda_architecture_b_max_births_per_tick", None)
    birth_capacity = _exact_int(
        birth_capacity, name="cuda_architecture_b_max_births_per_tick", minimum=1
    )
    threshold = _finite_float(
        founders[0].reproduction_threshold,
        name="neutral_repro_threshold",
    )
    if threshold < 0.0 or not math.isclose(
        threshold,
        float(founders[1].reproduction_threshold),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise _fail("founders must share a nonnegative reproduction threshold")
    authority = {
        "schema_version": SOURCE_AUTHORITY_SCHEMA_VERSION,
        "founder_pattern_registry_version": FOUNDER_PATTERN_REGISTRY_VERSION,
        "founder_assignment_version": FOUNDER_ASSIGNMENT_VERSION,
        "cell_id": cell_id,
        "founder_pattern": founder_pattern,
        "founders": [founder.to_dict() for founder in founders],
        "founder_ordinal_by_agent_id": ordinal_by_agent_id,
        "selection_seed": selection_seed,
        "neutral_repro_threshold": threshold,
        "birth_capacity": birth_capacity,
    }
    setattr(runner, SOURCE_AUTHORITY_FIELD, authority)
    return authority


def bind_source_authority_to_projection_v1(
    projection: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind stable founder agent IDs to the projection's exact slot layout."""

    validate = _mapping(authority, name="source authority")
    if (
        validate.get("schema_version") != SOURCE_AUTHORITY_SCHEMA_VERSION
        or validate.get("founder_pattern_registry_version")
        != FOUNDER_PATTERN_REGISTRY_VERSION
        or validate.get("founder_assignment_version") != FOUNDER_ASSIGNMENT_VERSION
    ):
        raise _fail("source authority identity mismatch")
    tensors = _mapping(projection.get("tensors"), name="projection tensors")
    alive = tensors.get("alive_mask")
    agent_ids = tensors.get("slot_to_agent_id")
    if (
        not isinstance(alive, torch.Tensor)
        or alive.dtype != torch.bool
        or alive.device.type != "cpu"
        or alive.ndim != 1
        or not isinstance(agent_ids, torch.Tensor)
        or agent_ids.dtype != torch.int64
        or agent_ids.device.type != "cpu"
        or tuple(agent_ids.shape) != tuple(alive.shape)
    ):
        raise _fail(
            "source projection requires exact-shape CPU alive_mask and "
            "slot_to_agent_id tensors"
        )
    assignments = _mapping(
        validate.get("founder_ordinal_by_agent_id"),
        name="founder_ordinal_by_agent_id",
    )
    normalized: dict[int, int] = {}
    for agent_id, ordinal in assignments.items():
        if (
            type(agent_id) is not int
            or agent_id < 0
            or type(ordinal) is not int
            or ordinal not in {0, 1}
        ):
            raise _fail("founder agent assignments must map nonnegative IDs to 0/1")
        normalized[agent_id] = ordinal
    ordinals = torch.full(tuple(alive.shape), -1, dtype=torch.int64)
    for slot in torch.nonzero(alive, as_tuple=False).flatten().tolist():
        agent_id = int(agent_ids[slot].item())
        if agent_id not in normalized:
            raise _fail(
                f"live source agent lacks founder assignment: agent_id={agent_id}"
            )
        ordinals[slot] = normalized[agent_id]
    if any(agent_id not in set(agent_ids[alive].tolist()) for agent_id in normalized):
        raise _fail("founder assignments contain agents absent from the live source")

    bound_authority = {
        name: copy.deepcopy(value)
        for name, value in validate.items()
        if name != "founder_ordinal_by_agent_id"
    }
    bound_authority["founder_ordinal_by_slot"] = ordinals
    bound = dict(projection)
    bound[SOURCE_AUTHORITY_FIELD] = bound_authority
    return bound


def bind_runner_source_authority_to_projection_v1(
    projection: Mapping[str, Any],
    runner: Any,
) -> dict[str, Any]:
    """Persist the runner-owned founder authority inside a cached G0 source."""

    authority = getattr(runner, SOURCE_AUTHORITY_FIELD, None)
    if not isinstance(authority, Mapping):
        raise _fail("runner lacks exact cross-path founder source authority")
    return bind_source_authority_to_projection_v1(projection, authority)


def _sequence(value: Any, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise _fail(f"{name} must be a sequence")
    return tuple(value)


def _founder(
    value: Any,
    *,
    name: str,
    expected: Mapping[str, float],
) -> Genome:
    payload = _mapping(value, name=name)
    try:
        genome = Genome.from_dict(dict(payload))
    except Exception as exc:
        raise _fail(f"{name} is not a canonical Genome payload") from exc
    for field in _AUDITED_PATTERN_FIELDS:
        actual = getattr(genome, field, None)
        if (
            type(actual) not in {int, float}
            or not math.isfinite(float(actual))
            or not math.isclose(
                float(actual),
                float(expected[field]),
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
        ):
            raise _fail(f"{name}.{field} does not match the registered founder pattern")
    return genome


def _live_mask(
    patch: PreparedDirectResidentPatchMulticellV1,
) -> torch.Tensor:
    banks = getattr(patch.patch, "live_banks", None)
    alive = banks.get("alive_mask") if isinstance(banks, Mapping) else None
    if (
        not isinstance(alive, torch.Tensor)
        or alive.dtype != torch.bool
        or alive.ndim != 2
        or int(alive.shape[0]) != patch.batch_cells
        or int(alive.shape[1]) < 1
    ):
        raise _fail("prepared Patch alive_mask must be a [cells, slots] bool tensor")
    return alive.detach().to(device="cpu")


def _validate_checkpoint_restore_state_v1(
    checkpoint_projection: Mapping[str, Any],
    *,
    live: torch.Tensor,
    controls: tuple[CrossPathCellControlV1, ...],
    expected_exposure_shape: ResidentReproductionExposureShapeV1,
    expected_exposure_enabled: bool,
) -> None:
    linkage = _mapping(
        checkpoint_projection.get("linkage"),
        name="checkpoint linkage projection",
    )
    linkage_tensors = _mapping(
        linkage.get("tensors"),
        name="checkpoint linkage tensors",
    )
    genotype_codes = linkage_tensors.get("genotype_codes")
    if (
        not isinstance(genotype_codes, torch.Tensor)
        or genotype_codes.device.type != "cpu"
        or genotype_codes.dtype != torch.int64
        or genotype_codes.requires_grad
        or not genotype_codes.is_contiguous()
        or tuple(genotype_codes.shape) != tuple(live.shape)
    ):
        raise _fail(
            "checkpoint linkage genotype_codes must be contiguous CPU int64 "
            "matching the prepared Patch [cell, slot] shape"
        )
    if bool(genotype_codes.lt(-1).any().item()):
        raise _fail(
            "checkpoint linkage genotype_codes contain an invalid negative code"
        )
    linked_rows = torch.tensor(
        [
            control.inheritance_mode == "linked_parent_copy_v1"
            for control in controls
        ],
        dtype=torch.bool,
    ).unsqueeze(1)
    if tuple(linked_rows.shape[:1]) != tuple(live.shape[:1]):
        raise _fail("checkpoint linkage controls differ from physical cell order")
    if bool((live & linked_rows & genotype_codes.eq(-1)).any().item()):
        raise _fail(
            "checkpoint linkage omits a genotype for a live linked-inheritance "
            "Patch slot"
        )

    exposure = _mapping(
        checkpoint_projection.get("exposure"),
        name="checkpoint exposure projection",
    )
    exposure_shape = _mapping(
        exposure.get("shape"),
        name="checkpoint exposure shape",
    )
    eligible_capacity = exposure_shape.get("eligible_capacity")
    maximum_live_count = int(live.sum(dim=1, dtype=torch.int64).max().item())
    if type(eligible_capacity) is not int or eligible_capacity < maximum_live_count:
        raise _fail(
            "checkpoint exposure eligible capacity is below the current "
            "per-cell live population bound"
        )
    if exposure_shape != expected_exposure_shape.payload():
        raise _fail(
            "checkpoint exposure shape differs from the declared campaign policy"
        )
    exposure_tensors = _mapping(
        exposure.get("tensors"),
        name="checkpoint exposure tensors",
    )
    cell_enabled = exposure_tensors.get("cell_enabled")
    expected_enabled = torch.full(
        (int(live.shape[0]),),
        expected_exposure_enabled,
        dtype=torch.bool,
    )
    if (
        not isinstance(cell_enabled, torch.Tensor)
        or cell_enabled.device.type != "cpu"
        or cell_enabled.dtype != torch.bool
        or cell_enabled.requires_grad
        or not cell_enabled.is_contiguous()
        or tuple(cell_enabled.shape) != tuple(expected_enabled.shape)
        or not torch.equal(cell_enabled, expected_enabled)
    ):
        raise _fail(
            "checkpoint exposure enabled cells differ from the declared "
            "campaign policy"
        )


def _source_ordinals(
    authority: Mapping[str, Any],
    *,
    cell_id: str,
    live: torch.Tensor,
) -> torch.Tensor:
    ordinals = authority.get("founder_ordinal_by_slot")
    if (
        not isinstance(ordinals, torch.Tensor)
        or ordinals.device.type != "cpu"
        or ordinals.dtype != torch.int64
        or ordinals.ndim != 1
        or tuple(ordinals.shape) != tuple(live.shape)
    ):
        raise _fail(f"{cell_id} founder_ordinal_by_slot must be exact-shape CPU int64")
    if bool(torch.any(ordinals[live] < 0).item()) or bool(
        torch.any(ordinals[live] > 1).item()
    ):
        raise _fail(f"{cell_id} live founder ordinals must be binary")
    if bool(torch.any(ordinals[~live] != -1).item()):
        raise _fail(f"{cell_id} dead slots must use founder ordinal -1")
    live_values = set(int(value) for value in ordinals[live].tolist())
    if live_values != {0, 1}:
        raise _fail(f"{cell_id} must contain both registered founders")
    return ordinals


def _materialize_cell(
    *,
    control: CrossPathCellControlV1,
    projection: Mapping[str, Any],
    live: torch.Tensor,
    resident_banks: Mapping[str, Any],
    cell_index: int,
    fresh_source: bool,
) -> tuple[LinkageDigestTableV1, torch.Tensor | None, int, float, int]:
    authority = _mapping(
        projection.get(SOURCE_AUTHORITY_FIELD),
        name=f"{control.cell_id} source authority",
    )
    if (
        authority.get("schema_version") != SOURCE_AUTHORITY_SCHEMA_VERSION
        or authority.get("founder_pattern_registry_version")
        != FOUNDER_PATTERN_REGISTRY_VERSION
        or authority.get("cell_id") != control.cell_id
        or authority.get("founder_pattern") != control.founder_pattern
    ):
        raise _fail(f"{control.cell_id} source authority identity mismatch")
    definition = FOUNDER_PATTERN_REGISTRY_V1.get(control.founder_pattern)
    if definition is None:
        raise _fail(
            f"{control.cell_id} founder pattern is not registered in "
            f"{FOUNDER_PATTERN_REGISTRY_VERSION}"
        )
    founders = _sequence(authority.get("founders"), name=f"{control.cell_id} founders")
    if len(founders) != 2:
        raise _fail(f"{control.cell_id} must declare exactly two founders")
    parent_a = _founder(
        founders[0],
        name=f"{control.cell_id} founder A",
        expected=definition.parent_a_values,
    )
    parent_b = _founder(
        founders[1],
        name=f"{control.cell_id} founder B",
        expected=definition.parent_b_values,
    )
    ordered, _digests = canonical_parent_order(parent_a, parent_b)
    if ordered[0] is parent_a:
        table = build_linkage_digest_table_v1(parent_a, parent_b)
        ordinal_to_code = founder_codes_v1(table)
    else:
        table = build_linkage_digest_table_v1(parent_b, parent_a)
        canonical_codes = founder_codes_v1(table)
        ordinal_to_code = (canonical_codes[1], canonical_codes[0])
    codes = None
    if fresh_source:
        ordinals = _source_ordinals(
            authority,
            cell_id=control.cell_id,
            live=live,
        )
        for field, bank_name in _AUDITED_RESIDENT_GENOME_BANKS.items():
            bank = resident_banks.get(bank_name)
            if (
                not isinstance(bank, torch.Tensor)
                or not bank.dtype.is_floating_point
                or bank.ndim != 2
                or int(bank.shape[0]) <= cell_index
                or tuple(bank[cell_index].shape) != tuple(live.shape)
            ):
                raise _fail(
                    f"{control.cell_id} audited resident genome bank "
                    f"{bank_name} is missing or malformed"
                )
            actual = bank[cell_index].detach().to(device="cpu")
            expected = torch.empty_like(actual)
            expected.fill_(float(getattr(parent_a, field)))
            expected[ordinals == 1] = float(getattr(parent_b, field))
            if not torch.equal(actual[live], expected[live]):
                raise _fail(
                    f"{control.cell_id} resident genome plane {bank_name} "
                    "does not match the frozen founder payload"
                )
        codes = torch.full_like(ordinals, -1)
        for ordinal, code in enumerate(ordinal_to_code):
            codes[ordinals == ordinal] = code
    selection_seed = _exact_int(
        authority.get("selection_seed"),
        name=f"{control.cell_id} selection_seed",
    )
    if selection_seed >= 1 << 64:
        raise _fail(f"{control.cell_id} selection_seed exceeds uint64")
    neutral_threshold = _finite_float(
        authority.get("neutral_repro_threshold"),
        name=f"{control.cell_id} neutral_repro_threshold",
    )
    birth_capacity = _exact_int(
        authority.get("birth_capacity"),
        name=f"{control.cell_id} birth_capacity",
        minimum=1,
    )
    return table, codes, selection_seed, neutral_threshold, birth_capacity


@dataclass(frozen=True, slots=True)
class InheritanceCrossPathSourceMaterializerV1:
    """Tombstone for the retired standalone inheritance source materializer."""

    device: str = "cuda"
    exposure_attempt_capacity: int | None = None
    exposure_snapshot_capacity: int | None = None
    exposure_storage_mode: str = STORAGE_MODE_EXACT_FULL
    exposure_aggregate_capacity: int = 1

    def __post_init__(self) -> None:
        raise _fail(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "cross-path source materialization is disabled"
        )
        if type(self.device) is not str or not self.device.strip():
            raise _fail("device must be a nonempty string")
        capacities = (
            self.exposure_attempt_capacity,
            self.exposure_snapshot_capacity,
        )
        if (capacities[0] is None) != (capacities[1] is None):
            raise _fail(
                "reproduction exposure attempt and snapshot capacities "
                "must be declared together"
            )
        if capacities[0] is not None and any(
            type(value) is not int or value < 1 for value in capacities
        ):
            raise _fail(
                "reproduction exposure capacities must be positive integers"
            )
        if self.exposure_storage_mode not in STORAGE_MODE_CODES:
            raise _fail("reproduction exposure storage mode is unsupported")
        if (
            type(self.exposure_aggregate_capacity) is not int
            or self.exposure_aggregate_capacity < 1
        ):
            raise _fail(
                "reproduction exposure aggregate capacity must be positive"
            )

    def __call__(
        self,
        *,
        patch: PreparedDirectResidentPatchMulticellV1,
        controls: tuple[CrossPathCellControlV1, ...],
        source_projections: tuple[Mapping[str, Any], ...],
        checkpoint_projection: Mapping[str, Any] | None = None,
    ) -> CrossPathBatchSourceMaterializationV1:
        if not isinstance(patch, PreparedDirectResidentPatchMulticellV1):
            raise _fail("patch must be PreparedDirectResidentPatchMulticellV1")
        if (
            not controls
            or any(
                not isinstance(control, CrossPathCellControlV1)
                or control.schema_version != CELL_CONTROL_SCHEMA_VERSION
                for control in controls
            )
            or tuple(control.cell_id for control in controls) != patch.cell_ids
        ):
            raise _fail("controls must exactly match prepared Patch cell order")
        if (
            len(source_projections) != patch.batch_cells
            or len(patch.source_projections) != patch.batch_cells
            or any(
                source is not canonical
                for source, canonical in zip(
                    source_projections,
                    patch.source_projections,
                    strict=True,
                )
            )
        ):
            raise _fail(
                "source projections must be the exact ordered Patch authorities"
            )
        live = _live_mask(patch)
        resident_banks = getattr(patch.patch, "live_banks", None)
        if not isinstance(resident_banks, Mapping):
            raise _fail("prepared Patch lacks resident live banks")
        if checkpoint_projection is not None and not isinstance(
            checkpoint_projection,
            Mapping,
        ):
            raise _fail("checkpoint_projection must be a mapping when supplied")
        fresh_source = checkpoint_projection is None
        tables: list[LinkageDigestTableV1] = []
        code_rows: list[torch.Tensor] = []
        seeds: dict[str, int] = {}
        thresholds: dict[str, float] = {}
        capacities: list[int] = []
        for index, (control, projection) in enumerate(
            zip(controls, source_projections, strict=True)
        ):
            table, codes, seed, threshold, capacity = _materialize_cell(
                control=control,
                projection=_mapping(projection, name=f"source_projections[{index}]"),
                live=live[index],
                resident_banks=resident_banks,
                cell_index=index,
                fresh_source=fresh_source,
            )
            tables.append(table)
            if codes is not None:
                code_rows.append(codes)
            seeds[control.cell_id] = seed
            thresholds[control.cell_id] = threshold
            capacities.append(capacity)
        if len(set(capacities)) != 1:
            raise _fail("birth_capacity must be identical across the physical batch")
        source_codes = None
        if fresh_source:
            if len(code_rows) != patch.batch_cells:
                raise _fail("fresh source omitted genotype-code rows")
            source_codes = (
                torch.stack(code_rows, dim=0)
                .to(device=self.device)
                .contiguous()
            )
            if tuple(source_codes.shape) != tuple(live.shape):
                raise _fail("materialized genotype-code shape drifted")
        elif code_rows:
            raise _fail("checkpoint restore invented fresh genotype-code rows")
        exposure_shape = None
        if self.exposure_attempt_capacity is not None:
            if self.exposure_storage_mode == STORAGE_MODE_COMPACT_COMPLETE:
                banks = getattr(patch.patch, "live_banks", None)
                agent_id_to_slot = (
                    banks.get("agent_id_to_slot")
                    if isinstance(banks, Mapping)
                    else None
                )
                if (
                    not isinstance(agent_id_to_slot, torch.Tensor)
                    or agent_id_to_slot.ndim != 2
                    or int(agent_id_to_slot.shape[0]) != patch.batch_cells
                ):
                    raise _fail(
                        "compact reproduction exposure requires the resident "
                        "agent-ID allocation bank"
                    )
                required_agent_id_capacity = int(agent_id_to_slot.shape[1])
                if self.exposure_aggregate_capacity < required_agent_id_capacity:
                    raise _fail(
                        "compact reproduction exposure aggregate capacity is "
                        "below the resident agent-ID allocation bound: "
                        f"declared={self.exposure_aggregate_capacity}, "
                        f"required={required_agent_id_capacity}"
                    )
            exposure_shape = ResidentReproductionExposureShapeV1(
                batch_cells=patch.batch_cells,
                attempt_capacity=self.exposure_attempt_capacity,
                snapshot_capacity=self.exposure_snapshot_capacity,
                eligible_capacity=int(live.shape[1]),
                storage_mode=self.exposure_storage_mode,
                aggregate_capacity=self.exposure_aggregate_capacity,
            )
        expected_exposure_shape = exposure_shape
        if expected_exposure_shape is None:
            expected_exposure_shape = ResidentReproductionExposureShapeV1(
                batch_cells=patch.batch_cells,
                attempt_capacity=1,
                snapshot_capacity=2,
                eligible_capacity=int(live.shape[1]),
            )
        if not fresh_source:
            _validate_checkpoint_restore_state_v1(
                checkpoint_projection,
                live=live,
                controls=controls,
                expected_exposure_shape=expected_exposure_shape,
                expected_exposure_enabled=exposure_shape is not None,
            )
        return CrossPathBatchSourceMaterializationV1(
            tables=tuple(tables),
            source_genotype_codes=source_codes,
            selection_seed_by_cell=MappingProxyType(seeds),
            neutral_repro_threshold_by_cell=MappingProxyType(thresholds),
            birth_capacity=capacities[0],
            exposure_shape=exposure_shape,
            device=self.device,
        )


def materialize_inheritance_cross_path_sources_v1(
    *,
    patch: PreparedDirectResidentPatchMulticellV1,
    controls: tuple[CrossPathCellControlV1, ...],
    source_projections: tuple[Mapping[str, Any], ...],
    checkpoint_projection: Mapping[str, Any] | None = None,
    device: str = "cuda",
) -> CrossPathBatchSourceMaterializationV1:
    """Reject the retired standalone source-materialization route."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "cross-path source materialization is disabled"
    )


__all__ = [
    "FOUNDER_PATTERN_REGISTRY_V1",
    "FOUNDER_PATTERN_REGISTRY_VERSION",
    "FOUNDER_ASSIGNMENT_VERSION",
    "FounderPatternDefinitionV1",
    "InheritanceCrossPathSourceMaterializationError",
    "InheritanceCrossPathSourceMaterializerV1",
    "SCHEMA_VERSION",
    "SOURCE_AUTHORITY_FIELD",
    "SOURCE_AUTHORITY_SCHEMA_VERSION",
    "apply_inheritance_cross_path_founders_v1",
    "bind_source_authority_to_projection_v1",
    "bind_runner_source_authority_to_projection_v1",
    "materialize_inheritance_cross_path_sources_v1",
]
