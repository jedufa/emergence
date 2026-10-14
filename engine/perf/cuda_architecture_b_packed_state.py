from __future__ import annotations

"""Versioned byte layout for the Architecture B packed-state challenger.

The layout is a transport and storage contract only.  It does not route a
simulation, load a CUDA module, or compile native code.
"""

from engine.runtime_roots import public_runtime_path

import json
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import torch

from perf.cuda_architecture_b_state_bank import (
    ACCEPTED_MANIFEST_SHA256,
    ACCEPTED_TENSOR_FIELD_COUNT,
    load_accepted_state_bank_manifest,
)


PACKED_STATE_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_packed_state_layout_v2"
)
PACKED_STATE_ALIGNMENT_BYTES: Final[int] = 64
STAGE2_EXTENSION: Final[Path] = Path(
    public_runtime_path('evidence_raw/eiel_archb_cuda_stage2_20260717/09_immutable/20260717T133943/eiel_cuda_architecture_b_resident_generation_v5.so')
)
STAGE2_EXTENSION_SHA256: Final[str] = (
    "456de172b28ac65a7d27438b9b918e4571d8c381d71bd3e1198ad12e0c8f7a5d"
)
STAGE2_EXTENSION_ABI_VERSION: Final[str] = "eiel_cuda_architecture_b_abi_v3"

FIELD_CATEGORIES: Final[tuple[str, ...]] = (
    "agent_scalars",
    "world_patch",
    "policy_parameters_cache",
    "rng",
    "identity_slot_lineage",
    "event_lifecycle",
    "accumulators",
    "reusable_scratch",
)
STATE_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "hot_persistent",
        "hot_read_only",
        "hot_derived_scratch",
        "cold_persistent_checkpoint",
        "reporting_only",
        "inactive_capacity",
        "runtime_auxiliary",
    }
)

PRODUCTION_PATH_COUNTS: Final[dict[str, int]] = {"flat": 293, "patch": 295}
PRODUCTION_STORAGE_COUNTS: Final[dict[str, int]] = {"flat": 183, "patch": 185}
PRODUCTION_STAGE1B_ALIAS_COUNTS: Final[dict[str, int]] = {
    "flat": 110,
    "patch": 110,
}
_FLAT_ABSENT_ACCEPTED_FIELDS: Final[frozenset[str]] = frozenset(
    {"patch_recovery_activity", "patch_visibility"}
)
_ACCEPTED_PRODUCTION_SHAPE_OVERRIDES: Final[
    dict[str, tuple[ShapeDimension, ...]]
] = {
    "agent_id_to_slot_agent_ids": ("cell", "active_population_count"),
    "agent_id_to_slot_slots": ("cell", "active_population_count"),
    "agent_ids_by_alive_order": ("cell", "active_population_count"),
    "slot_free_queue": ("cell", "resident_free_queue_count"),
}

_TORCH_DTYPE_BY_NAME: Final[dict[str, torch.dtype]] = {
    "bool": torch.bool,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
}
_DTYPE_BYTES: Final[dict[str, int]] = {
    name: torch.empty((), dtype=dtype).element_size()
    for name, dtype in _TORCH_DTYPE_BY_NAME.items()
}

_EVENT_PREFIXES: Final[tuple[str, ...]] = (
    "event_tape_",
    "typed_lifecycle_event_",
)
_REPORTING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "attacks",
        "dev_stage_initialized",
        "early_energy_collected",
        "early_energy_collected_exact",
        "early_life_weight",
        "early_survival_steps",
        "energy_collected",
        "energy_collected_exact",
        "episode0_pred_err_count",
        "episode0_pred_err_sum",
        "first_repro_step",
        "late_pred_err_count",
        "late_pred_err_sum",
        "pred_err_count",
        "pred_err_sum",
    }
)
_INACTIVE_FIXED_MLP_FIELDS: Final[frozenset[str]] = frozenset(
    {"biome_grid", "gru_pack", "hazard_rate_map", "hmem", "policy_use_gru"}
)
_DERIVED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "action_cache_actions",
        "action_cache_cursor",
        "action_cache_next",
        "action_cache_token",
        "agent_grid",
        "patch_density",
        "patch_recovery_activity",
        "patch_scent_emit_pending",
        "patch_visibility",
        "patch_waste_emit_pending",
    }
)
_NO_STAGE2_CONSUMER_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "agent_id_to_slot_agent_ids",
        "agent_id_to_slot_slots",
        "biome_grid",
        "gru_pack",
        "hazard_rate_map",
        "patch_recovery_activity",
        "patch_visibility",
    }
)
_RESIDENT_SEMANTIC_HOT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "action_cache_actions",
        "action_cache_cursor",
        "action_cache_next",
        "action_cache_token",
        "action_oscillations",
        "action_switches",
        "age",
        "agent_grid",
        "agent_id_to_slot",
        "agent_ids_by_alive_order",
        "alive_mask",
        "cost_charge_maturation_scalar",
        "cost_charge_scalar",
        "drive_inertia",
        "drive_jitter",
        "drive_noise",
        "drive_volatility",
        "drives",
        "energy",
        "episode0_drive_jitter_sum",
        "episode0_oscillations",
        "episode0_selfharm_events",
        "episode0_switches",
        "food_grid",
        "gru_upkeep_cost",
        "health",
        "injury",
        "last_action",
        "last_energy_delta",
        "last_health_delta",
        "last_injury_delta",
        "last_last_action",
        "lineage_depth",
        "lineage_founder_id",
        "mlp_pack",
        "patch_cover",
        "patch_density",
        "patch_fertility",
        "patch_food_amount",
        "patch_movement_cost",
        "patch_scent",
        "patch_scent_emit_pending",
        "patch_shelter",
        "patch_waste",
        "patch_waste_emit_pending",
        "policy_action_temperature",
        "policy_attack_bias",
        "policy_drive_logit_gain",
        "policy_energy_observation",
        "pos_x",
        "pos_y",
        "prev_drive_delta",
        "rest_streak",
        "selfharm_events",
        "slot_to_agent_id",
        "slot_tokens",
        "survival_steps",
    }
)


class PackedStateLayoutError(ValueError):
    """Raised when a packed-state layout or tensor bank is contradictory."""


ShapeDimension = str | int


@dataclass(frozen=True)
class PackedStateProductionBinding:
    world: str
    cell_count: int
    shape_bindings: tuple[tuple[str, int], ...]
    path_to_canonical: tuple[tuple[str, str], ...]
    representative_by_canonical: tuple[tuple[str, str], ...]
    absent_accepted_fields: tuple[str, ...]
    source_path_count: int
    source_storage_count: int
    stage1b_alias_count: int

    @property
    def shape_bindings_by_name(self) -> dict[str, int]:
        return dict(self.shape_bindings)

    @property
    def path_to_canonical_by_name(self) -> dict[str, str]:
        return dict(self.path_to_canonical)

    @property
    def representative_by_canonical_name(self) -> dict[str, str]:
        return dict(self.representative_by_canonical)

    @property
    def layout_aliases(self) -> dict[str, str]:
        return {
            path: canonical
            for path, canonical in self.path_to_canonical
            if path != canonical
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "world": self.world,
            "cell_count": self.cell_count,
            "shape_bindings": dict(self.shape_bindings),
            "source_path_count": self.source_path_count,
            "source_storage_count": self.source_storage_count,
            "stage1b_alias_count": self.stage1b_alias_count,
            "packed_layout_alias_count": len(self.layout_aliases),
            "absent_accepted_fields": list(self.absent_accepted_fields),
            "path_to_canonical": dict(self.path_to_canonical),
            "representative_by_canonical": dict(
                self.representative_by_canonical
            ),
        }


@dataclass(frozen=True)
class _FieldTemplate:
    name: str
    dtype_name: str
    shape_template: tuple[ShapeDimension, ...]
    category: str
    state_class: str
    source: str = "stage2_runtime_contract"


@dataclass(frozen=True)
class PackedStateField:
    name: str
    dtype_name: str
    shape_template: tuple[ShapeDimension, ...]
    shape: tuple[int, ...]
    category: str
    state_class: str
    source: str
    offset_bytes: int
    nbytes: int
    storage_nbytes: int
    alignment_bytes: int

    @property
    def end_offset_bytes(self) -> int:
        return self.offset_bytes + self.storage_nbytes

    @property
    def torch_dtype(self) -> torch.dtype:
        return _TORCH_DTYPE_BY_NAME[self.dtype_name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype_name,
            "shape_template": list(self.shape_template),
            "shape": list(self.shape),
            "category": self.category,
            "state_class": self.state_class,
            "source": self.source,
            "offset_bytes": self.offset_bytes,
            "nbytes": self.nbytes,
            "storage_nbytes": self.storage_nbytes,
            "alignment_bytes": self.alignment_bytes,
        }


@dataclass(frozen=True)
class PackedStateLayout:
    schema_version: str
    source_manifest_sha256: str
    source_extension_sha256: str
    source_extension_abi_version: str
    shape_bindings: tuple[tuple[str, int], ...]
    fields: tuple[PackedStateField, ...]
    aliases: tuple[tuple[str, str], ...]
    alignment_bytes: int
    total_bytes: int
    manifest_sha256: str

    @property
    def fields_by_name(self) -> dict[str, PackedStateField]:
        return {field.name: field for field in self.fields}

    @property
    def aliases_by_name(self) -> dict[str, str]:
        return dict(self.aliases)

    @property
    def accepted_fields(self) -> tuple[PackedStateField, ...]:
        return tuple(field for field in self.fields if field.source == "accepted_manifest")

    def resolve_name(self, name: str) -> str:
        if name in self.fields_by_name:
            return name
        try:
            return self.aliases_by_name[name]
        except KeyError as exc:
            raise PackedStateLayoutError(f"unknown packed-state field or alias: {name!r}") from exc

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_contract": {
                "accepted_manifest_sha256": self.source_manifest_sha256,
                "accepted_tensor_field_count": ACCEPTED_TENSOR_FIELD_COUNT,
                "stage2_extension_sha256": self.source_extension_sha256,
                "stage2_extension_abi_version": self.source_extension_abi_version,
            },
            "shape_bindings": dict(self.shape_bindings),
            "alignment_bytes": self.alignment_bytes,
            "total_bytes": self.total_bytes,
            "fields": [field.to_dict() for field in self.fields],
            "aliases": dict(self.aliases),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["manifest_sha256"] = self.manifest_sha256
        return payload


def _runtime_templates() -> tuple[_FieldTemplate, ...]:
    scalar = ("cell", 1)
    templates = [
        _FieldTemplate(
            "birth.event_tape_active_count",
            "int64",
            scalar,
            "event_lifecycle",
            "reporting_only",
        ),
        _FieldTemplate(
            "birth.event_tape_status",
            "int32",
            scalar,
            "event_lifecycle",
            "reporting_only",
        ),
        _FieldTemplate(
            "birth.identity.agent_id_to_slot",
            "int32",
            ("cell", "birth_agent_id_capacity"),
            "identity_slot_lineage",
            "runtime_auxiliary",
        ),
        _FieldTemplate(
            "birth.identity.population_next_id",
            "int64",
            scalar,
            "identity_slot_lineage",
            "runtime_auxiliary",
        ),
        _FieldTemplate(
            "birth.identity.slot_free_queue",
            "int64",
            ("cell", "capacity"),
            "identity_slot_lineage",
            "runtime_auxiliary",
        ),
        _FieldTemplate(
            "birth.identity.slot_free_queue_count",
            "int64",
            scalar,
            "identity_slot_lineage",
            "runtime_auxiliary",
        ),
        _FieldTemplate(
            "resident.pending_free_agent_ids",
            "int64",
            ("cell", "capacity"),
            "reusable_scratch",
            "hot_derived_scratch",
        ),
        _FieldTemplate(
            "resident.pending_free_slots",
            "int64",
            ("cell", "capacity"),
            "reusable_scratch",
            "hot_derived_scratch",
        ),
        _FieldTemplate(
            "resident.pending_free_count",
            "int64",
            scalar,
            "reusable_scratch",
            "hot_derived_scratch",
        ),
        _FieldTemplate(
            "resident.drive_noise_samples_by_tick_slot",
            "float32",
            ("cell", "tick_capacity", "capacity", "drive_channels"),
            "rng",
            "hot_read_only",
        ),
        _FieldTemplate(
            "resident.drive_noise_samples_valid_by_tick_slot",
            "bool",
            ("cell", "tick_capacity", "capacity"),
            "rng",
            "hot_read_only",
        ),
        _FieldTemplate(
            "birth.scalar_rng.python_random_words",
            "uint32",
            ("cell", 624),
            "rng",
            "runtime_auxiliary",
        ),
        _FieldTemplate(
            "birth.scalar_rng.python_random_index",
            "int64",
            scalar,
            "rng",
            "runtime_auxiliary",
        ),
        _FieldTemplate(
            "birth.scalar_rng.python_random_has_gauss",
            "bool",
            scalar,
            "rng",
            "runtime_auxiliary",
        ),
        _FieldTemplate(
            "birth.scalar_rng.python_random_gauss_next",
            "float64",
            scalar,
            "rng",
            "runtime_auxiliary",
        ),
        _FieldTemplate(
            "birth.scalar_rng.python_random_gauss_primary_tape",
            "float64",
            ("cell", "cpython_gauss_tape_capacity"),
            "rng",
            "runtime_auxiliary",
        ),
        _FieldTemplate(
            "birth.scalar_rng.python_random_gauss_secondary_tape",
            "float64",
            ("cell", "cpython_gauss_tape_capacity"),
            "rng",
            "runtime_auxiliary",
        ),
        _FieldTemplate(
            "birth.policy.rng_manager_mutation_torch_cpu_state",
            "uint8",
            ("cell", "mutation_rng_state_bytes"),
            "rng",
            "runtime_auxiliary",
        ),
    ]
    for stem in (
        "resident.env_pcg64",
        "birth.planner_pcg64",
        "birth.drive.pcg64",
        "birth.runner_pcg64",
        "resident.birth_spawn_env_pcg64",
    ):
        templates.extend(
            (
                _FieldTemplate(
                    f"{stem}_state_words",
                    "uint64",
                    ("cell", 4),
                    "rng",
                    "hot_persistent",
                ),
                _FieldTemplate(
                    f"{stem}_buffer_words",
                    "int64",
                    ("cell", 2),
                    "rng",
                    "hot_persistent",
                ),
            )
        )
    for name in (
        "birth.metrics_np_semantic_draws",
        "birth.policy_np_semantic_draws",
        "birth.drive.mutation_np_semantic_draws",
        "birth.scalar_rng.mutation_py_semantic_draws",
        "birth.scalar_rng.python_random_raw_draws",
        "birth.policy.mutation_torch_cpu_semantic_draws",
    ):
        templates.append(_FieldTemplate(name, "int64", scalar, "rng", "runtime_auxiliary"))
    for prefix, capacity_symbol, state_class in (
        ("birth.policy", "ordinary_birth_capacity", "runtime_auxiliary"),
        (
            "birth.turnover_policy",
            "turnover_capacity",
            "cold_persistent_checkpoint",
        ),
    ):
        templates.extend(
            (
                _FieldTemplate(
                    f"{prefix}.policy_birth_delta_tape",
                    "float32",
                    ("cell", capacity_symbol, "mlp_pack_width"),
                    "policy_parameters_cache",
                    state_class,
                ),
                _FieldTemplate(
                    f"{prefix}.policy_birth_mutation_mask_tape",
                    "bool",
                    ("cell", capacity_symbol, "mlp_pack_width"),
                    "policy_parameters_cache",
                    state_class,
                ),
                _FieldTemplate(
                    f"{prefix}.policy_birth_crossover_mask_tape",
                    "bool",
                    ("cell", capacity_symbol, "mlp_pack_width"),
                    "policy_parameters_cache",
                    state_class,
                ),
                _FieldTemplate(
                    f"{prefix}.policy_birth_post_rng_state_tape",
                    "uint8",
                    ("cell", capacity_symbol, "mutation_rng_state_bytes"),
                    "rng",
                    state_class,
                ),
                _FieldTemplate(
                    f"{prefix}.policy_birth_semantic_draws_tape",
                    "int64",
                    ("cell", capacity_symbol),
                    "rng",
                    state_class,
                ),
                _FieldTemplate(
                    f"{prefix}.policy_birth_tape_cursor",
                    "int64",
                    scalar,
                    "policy_parameters_cache",
                    state_class,
                ),
            )
        )
    return tuple(templates)


_RUNTIME_TEMPLATES: Final[tuple[_FieldTemplate, ...]] = _runtime_templates()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _numel(shape: Iterable[int]) -> int:
    result = 1
    for size in shape:
        result *= size
    return result


def _accepted_access_partition(name: str) -> str:
    if name in _NO_STAGE2_CONSUMER_FIELDS:
        return "no_stage2_consumer"
    if name in _RESIDENT_SEMANTIC_HOT_FIELDS or name.startswith(_EVENT_PREFIXES):
        return "semantic_hot"
    return "birth_turnover_only"


def _accepted_state_class(name: str, semantic_owner: str) -> str:
    partition = _accepted_access_partition(name)
    if name in _INACTIVE_FIXED_MLP_FIELDS:
        return "inactive_capacity"
    if name in _REPORTING_FIELDS or name.startswith(_EVENT_PREFIXES):
        return "reporting_only"
    if name == "rng_manager_mutation_torch_cpu_state" or partition == "no_stage2_consumer":
        return "cold_persistent_checkpoint"
    if name in _DERIVED_FIELDS:
        return "hot_derived_scratch"
    if name in {"patch_cover", "patch_movement_cost", "patch_shelter"}:
        return "hot_read_only"
    if partition == "semantic_hot":
        return "hot_persistent"
    if "policy and genome" in semantic_owner or "organism" in semantic_owner:
        return "cold_persistent_checkpoint"
    return "cold_persistent_checkpoint"


def _accepted_category(name: str, semantic_owner: str) -> str:
    if name.startswith(_EVENT_PREFIXES):
        return "event_lifecycle"
    if name == "rng_manager_mutation_torch_cpu_state":
        return "rng"
    if name in _REPORTING_FIELDS:
        return "accumulators"
    if name in _DERIVED_FIELDS and not name.startswith("action_cache_"):
        return "reusable_scratch"
    if "population identity and slot" in semantic_owner:
        return "identity_slot_lineage"
    if "environment" in semantic_owner or semantic_owner.startswith("Patch"):
        return "world_patch"
    if "policy and genome" in semantic_owner:
        return "policy_parameters_cache"
    if "organism" in semantic_owner:
        return "agent_scalars"
    raise PackedStateLayoutError(
        f"accepted field {name!r} has no unambiguous packed category: {semantic_owner!r}"
    )


@lru_cache(maxsize=1)
def _accepted_templates() -> tuple[_FieldTemplate, ...]:
    manifest = load_accepted_state_bank_manifest()
    templates = tuple(
        _FieldTemplate(
            name=field.name,
            dtype_name=field.dtype_name,
            shape_template=_ACCEPTED_PRODUCTION_SHAPE_OVERRIDES.get(
                field.name, field.device_neutral_shape_variables
            ),
            category=_accepted_category(field.name, field.semantic_owner),
            state_class=_accepted_state_class(field.name, field.semantic_owner),
            source="accepted_manifest",
        )
        for field in manifest.fields
    )
    if len(templates) != ACCEPTED_TENSOR_FIELD_COUNT:
        raise PackedStateLayoutError("accepted packed field inventory is incomplete")
    partition_counts = {
        partition: sum(
            _accepted_access_partition(field.name) == partition for field in manifest.fields
        )
        for partition in ("semantic_hot", "birth_turnover_only", "no_stage2_consumer")
    }
    if partition_counts != {
        "semantic_hot": 79,
        "birth_turnover_only": 53,
        "no_stage2_consumer": 7,
    }:
        raise PackedStateLayoutError(
            f"Stage2 accepted field partition changed: {partition_counts}"
        )
    return templates


def _templates_by_name() -> dict[str, _FieldTemplate]:
    templates = (*_accepted_templates(), *_RUNTIME_TEMPLATES)
    result = {template.name: template for template in templates}
    if len(result) != ACCEPTED_TENSOR_FIELD_COUNT + len(_RUNTIME_TEMPLATES):
        raise PackedStateLayoutError("packed canonical field names are not unique")
    return result


def _bind_shape_symbol(
    bindings: dict[str, int],
    symbol: str,
    size: int,
    *,
    field_name: str,
) -> None:
    previous = bindings.setdefault(symbol, size)
    if previous != size:
        raise PackedStateLayoutError(
            f"production shape symbol {symbol!r} conflicts at {field_name!r}: "
            f"expected {previous}, got {size}"
        )


def bind_production_stage1b_cells(
    cells: Sequence[Any],
) -> PackedStateProductionBinding:
    """Bind exact Stage1B paths and dimensions from prepared production cells."""

    from perf import cuda_architecture_b_stage1b as stage1b

    source_cells = tuple(cells)
    if not source_cells:
        raise PackedStateLayoutError("at least one production Stage1B cell is required")
    worlds = {
        getattr(getattr(cell.birth_state, "receipt", None), "world", None)
        for cell in source_cells
    }
    if len(worlds) != 1 or next(iter(worlds)) not in PRODUCTION_PATH_COUNTS:
        raise PackedStateLayoutError(
            f"production cells must share one exact flat or patch world: {sorted(worlds, key=str)}"
        )
    world = str(next(iter(worlds)))

    flattened = [stage1b._flatten_cell(cell) for cell in source_cells]
    first = flattened[0]
    expected_path_count = PRODUCTION_PATH_COUNTS[world]
    if len(first) != expected_path_count:
        raise PackedStateLayoutError(
            f"production {world} Stage1B path count changed: "
            f"expected {expected_path_count}, got {len(first)}"
        )
    aliases = stage1b._alias_map(first)
    expected_alias_count = PRODUCTION_STAGE1B_ALIAS_COUNTS[world]
    actual_alias_count = sum(path != representative for path, representative in aliases.items())
    if actual_alias_count != expected_alias_count:
        raise PackedStateLayoutError(
            f"production {world} Stage1B alias count changed: "
            f"expected {expected_alias_count}, got {actual_alias_count}"
        )
    representatives = set(aliases.values())
    if len(representatives) != PRODUCTION_STORAGE_COUNTS[world]:
        raise PackedStateLayoutError(
            f"production {world} Stage1B storage count changed: "
            f"expected {PRODUCTION_STORAGE_COUNTS[world]}, got {len(representatives)}"
        )

    for index, flat in enumerate(flattened[1:], start=1):
        if set(flat) != set(first):
            raise PackedStateLayoutError(
                f"production Stage1B paths differ at cell {index}"
            )
        if stage1b._alias_map(flat) != aliases:
            raise PackedStateLayoutError(
                f"production Stage1B alias graph differs at cell {index}"
            )

    accepted_names = {template.name for template in _accepted_templates()}
    representative_to_canonical: dict[str, str] = {}
    observed_accepted: set[str] = set()
    for name in sorted(accepted_names):
        resident_path = f"resident.{name}"
        if resident_path not in first:
            continue
        representative = aliases[resident_path]
        previous = representative_to_canonical.setdefault(representative, name)
        if previous != name:
            raise PackedStateLayoutError(
                "accepted canonical fields collapse onto one production view: "
                f"{previous!r}, {name!r}"
            )
        observed_accepted.add(name)

    absent_accepted = accepted_names - observed_accepted
    expected_absent = _FLAT_ABSENT_ACCEPTED_FIELDS if world == "flat" else frozenset()
    if absent_accepted != expected_absent:
        raise PackedStateLayoutError(
            f"production {world} accepted-field presence changed: "
            f"expected absent={sorted(expected_absent)}, got={sorted(absent_accepted)}"
        )

    runtime_representatives = representatives - set(representative_to_canonical)
    expected_runtime_representatives = {
        template.name for template in _RUNTIME_TEMPLATES
    }
    if runtime_representatives != expected_runtime_representatives:
        raise PackedStateLayoutError(
            "production Stage1B runtime representatives changed: "
            f"missing={sorted(expected_runtime_representatives - runtime_representatives)}, "
            f"unexpected={sorted(runtime_representatives - expected_runtime_representatives)}"
        )
    representative_to_canonical.update(
        {representative: representative for representative in runtime_representatives}
    )
    if set(representative_to_canonical) != representatives:
        raise PackedStateLayoutError("production Stage1B representative binding is incomplete")

    path_to_canonical = {
        path: representative_to_canonical[representative]
        for path, representative in aliases.items()
    }
    if len(path_to_canonical) != expected_path_count:
        raise PackedStateLayoutError("production Stage1B path binding is incomplete")

    templates = _templates_by_name()
    shape_bindings: dict[str, int] = {"cell": len(source_cells)}
    devices: set[str] = set()
    source_storage_owners: dict[tuple[str, int], tuple[int, str]] = {}
    for cell_index, flat in enumerate(flattened):
        for representative, canonical in sorted(representative_to_canonical.items()):
            tensor = flat[representative]
            template = templates[canonical]
            if not isinstance(tensor, torch.Tensor):
                raise PackedStateLayoutError(
                    f"production path {representative!r} is not a tensor"
                )
            if tensor.dtype != _TORCH_DTYPE_BY_NAME[template.dtype_name]:
                raise PackedStateLayoutError(
                    f"production dtype mismatch for {representative!r}: "
                    f"expected {template.dtype_name}, got "
                    f"{str(tensor.dtype).removeprefix('torch.')}"
                )
            source_template = template.shape_template[1:]
            if tensor.ndim != len(source_template):
                raise PackedStateLayoutError(
                    f"production rank mismatch for {representative!r}: "
                    f"expected {len(source_template)}, got {tensor.ndim}"
                )
            if not tensor.is_contiguous() or tensor.requires_grad:
                raise PackedStateLayoutError(
                    f"production tensor {representative!r} must be contiguous and detached"
                )
            if tensor.device.type != "cuda":
                raise PackedStateLayoutError(
                    f"production tensor {representative!r} must be CUDA-resident"
                )
            devices.add(str(tensor.device))
            storage_key = (str(tensor.device), int(tensor.untyped_storage()._cdata))
            previous_owner = source_storage_owners.setdefault(
                storage_key, (cell_index, representative)
            )
            if previous_owner != (cell_index, representative):
                raise PackedStateLayoutError(
                    "distinct production canonical fields or cells share source storage: "
                    f"{previous_owner} and {(cell_index, representative)}"
                )
            for dimension, size in zip(source_template, tensor.shape, strict=True):
                actual = int(size)
                if isinstance(dimension, int):
                    if actual != dimension:
                        raise PackedStateLayoutError(
                            f"production shape mismatch for {representative!r}: "
                            f"expected fixed dimension {dimension}, got {actual}"
                        )
                else:
                    _bind_shape_symbol(
                        shape_bindings,
                        dimension,
                        actual,
                        field_name=representative,
                    )
    if len(devices) != 1:
        raise PackedStateLayoutError(
            f"production Stage1B cells must use one CUDA device: {sorted(devices)}"
        )

    gauss_capacity = 512 * (
        shape_bindings["ordinary_birth_capacity"]
        + shape_bindings["turnover_capacity"]
    )
    if shape_bindings.get("cpython_gauss_tape_capacity") != gauss_capacity:
        raise PackedStateLayoutError(
            "production CPython Gaussian tape capacity contradicts birth capacities"
        )
    return PackedStateProductionBinding(
        world=world,
        cell_count=len(source_cells),
        shape_bindings=tuple(sorted(shape_bindings.items())),
        path_to_canonical=tuple(sorted(path_to_canonical.items())),
        representative_by_canonical=tuple(
            sorted(
                (canonical, representative)
                for representative, canonical in representative_to_canonical.items()
            )
        ),
        absent_accepted_fields=tuple(sorted(absent_accepted)),
        source_path_count=len(first),
        source_storage_count=len(representatives),
        stage1b_alias_count=actual_alias_count,
    )


def _resolve_shape(
    template: tuple[ShapeDimension, ...], bindings: Mapping[str, int]
) -> tuple[int, ...]:
    return tuple(bindings[value] if isinstance(value, str) else value for value in template)


def build_packed_state_layout(
    shape_bindings: Mapping[str, int],
    *,
    include_runtime_fields: bool = True,
    verify_source_artifacts: bool = True,
    production_binding: PackedStateProductionBinding | None = None,
) -> PackedStateLayout:
    """Build one deterministic same-shape packed layout.

    Offsets are fixed for the supplied dimensions.  Runtime fields are the
    unique Stage2 tensor owners needed by the persistent challenger; aliases
    are compatibility names and never receive storage.
    """

    if verify_source_artifacts:
        try:
            extension_sha256 = _file_sha256(STAGE2_EXTENSION)
        except OSError as exc:
            raise PackedStateLayoutError(
                f"unable to verify immutable Stage2 extension: {STAGE2_EXTENSION}"
            ) from exc
        if extension_sha256 != STAGE2_EXTENSION_SHA256:
            raise PackedStateLayoutError(
                "immutable Stage2 extension digest mismatch: "
                f"expected {STAGE2_EXTENSION_SHA256}, got {extension_sha256}"
            )

    accepted = _accepted_templates()
    templates = accepted + (_RUNTIME_TEMPLATES if include_runtime_fields else ())
    required_symbols = {
        dimension
        for template in templates
        for dimension in template.shape_template
        if isinstance(dimension, str)
    }
    provided = dict(shape_bindings)
    auto_symbols = {"cpython_gauss_tape_capacity"} if include_runtime_fields else set()
    missing = sorted(required_symbols - set(provided) - auto_symbols)
    unexpected = sorted(set(provided) - required_symbols)
    if missing or unexpected:
        raise PackedStateLayoutError(
            f"shape bindings mismatch: missing={missing}, unexpected={unexpected}"
        )
    for symbol, value in sorted(provided.items()):
        if type(value) is not int or value < 0:
            raise PackedStateLayoutError(
                f"shape binding {symbol!r} must be an exact non-negative integer"
            )
        if symbol not in {
            "ordinary_birth_capacity",
            "turnover_capacity",
            "cpython_gauss_tape_capacity",
            "resident_free_queue_count",
        } and value == 0:
            raise PackedStateLayoutError(f"shape binding {symbol!r} must be positive")
    if provided["cell"] <= 0:
        raise PackedStateLayoutError("shape binding 'cell' must be positive")
    if (
        provided["active_population_count"]
        + provided["resident_free_queue_count"]
        != provided["capacity"]
    ):
        raise PackedStateLayoutError(
            "active population plus resident free queue must equal capacity"
        )
    if include_runtime_fields:
        if provided["turnover_capacity"] > provided["capacity"]:
            raise PackedStateLayoutError("turnover_capacity cannot exceed capacity")
        expected_gauss_capacity = 512 * (
            provided["ordinary_birth_capacity"] + provided["turnover_capacity"]
        )
        if (
            "cpython_gauss_tape_capacity" in provided
            and provided["cpython_gauss_tape_capacity"] != expected_gauss_capacity
        ):
            raise PackedStateLayoutError(
                "cpython_gauss_tape_capacity contradicts birth capacities"
            )
        provided["cpython_gauss_tape_capacity"] = expected_gauss_capacity
    if production_binding is not None:
        if not include_runtime_fields:
            raise PackedStateLayoutError(
                "production binding requires the complete runtime packed layout"
            )
        if dict(production_binding.shape_bindings) != provided:
            raise PackedStateLayoutError(
                "production binding shape dimensions differ from layout bindings"
            )

    category_order = {name: index for index, name in enumerate(FIELD_CATEGORIES)}
    templates = tuple(sorted(templates, key=lambda row: (category_order[row.category], row.name)))
    fields: list[PackedStateField] = []
    cursor = 0
    seen_names: set[str] = set()
    for template in templates:
        if template.name in seen_names:
            raise PackedStateLayoutError(
                f"duplicate canonical packed-state field: {template.name!r}"
            )
        seen_names.add(template.name)
        if template.dtype_name not in _DTYPE_BYTES:
            raise PackedStateLayoutError(
                f"unsupported dtype for {template.name}: {template.dtype_name}"
            )
        if template.category not in category_order or template.state_class not in STATE_CLASSES:
            raise PackedStateLayoutError(
                f"invalid classification for packed-state field {template.name!r}"
            )
        shape = _resolve_shape(template.shape_template, provided)
        nbytes = _numel(shape) * _DTYPE_BYTES[template.dtype_name]
        storage_nbytes = nbytes if nbytes else PACKED_STATE_ALIGNMENT_BYTES
        cursor = _align(cursor, PACKED_STATE_ALIGNMENT_BYTES)
        fields.append(
            PackedStateField(
                name=template.name,
                dtype_name=template.dtype_name,
                shape_template=template.shape_template,
                shape=shape,
                category=template.category,
                state_class=template.state_class,
                source=template.source,
                offset_bytes=cursor,
                nbytes=nbytes,
                storage_nbytes=storage_nbytes,
                alignment_bytes=PACKED_STATE_ALIGNMENT_BYTES,
            )
        )
        cursor += storage_nbytes

    aliases = production_binding.layout_aliases if production_binding is not None else {}
    missing_alias_targets = sorted(set(aliases.values()) - seen_names)
    if missing_alias_targets:
        raise PackedStateLayoutError(
            f"production aliases target absent canonical fields: {missing_alias_targets}"
        )
    total_bytes = _align(cursor, PACKED_STATE_ALIGNMENT_BYTES)
    provisional = PackedStateLayout(
        schema_version=PACKED_STATE_SCHEMA_VERSION,
        source_manifest_sha256=ACCEPTED_MANIFEST_SHA256,
        source_extension_sha256=STAGE2_EXTENSION_SHA256,
        source_extension_abi_version=STAGE2_EXTENSION_ABI_VERSION,
        shape_bindings=tuple(sorted(provided.items())),
        fields=tuple(fields),
        aliases=tuple(sorted(aliases.items())),
        alignment_bytes=PACKED_STATE_ALIGNMENT_BYTES,
        total_bytes=total_bytes,
        manifest_sha256="",
    )
    layout = PackedStateLayout(
        **{
            **provisional.__dict__,
            "manifest_sha256": _canonical_sha256(provisional._payload()),
        }
    )
    validate_packed_state_layout(layout)
    return layout


def validate_packed_state_layout(layout: PackedStateLayout) -> None:
    """Validate schema, hashing, ordering, alignment, shapes, and aliases."""

    if not isinstance(layout, PackedStateLayout):
        raise PackedStateLayoutError("layout must be a PackedStateLayout")
    if layout.schema_version != PACKED_STATE_SCHEMA_VERSION:
        raise PackedStateLayoutError("packed-state schema version mismatch")
    if layout.alignment_bytes != PACKED_STATE_ALIGNMENT_BYTES:
        raise PackedStateLayoutError("packed-state alignment contract mismatch")
    if (
        layout.source_manifest_sha256 != ACCEPTED_MANIFEST_SHA256
        or layout.source_extension_sha256 != STAGE2_EXTENSION_SHA256
        or layout.source_extension_abi_version != STAGE2_EXTENSION_ABI_VERSION
    ):
        raise PackedStateLayoutError("packed-state source identity mismatch")
    expected_hash = _canonical_sha256(layout._payload())
    if layout.manifest_sha256 != expected_hash:
        raise PackedStateLayoutError(
            "packed-state manifest hash mismatch: "
            f"expected {expected_hash}, got {layout.manifest_sha256}"
        )
    bindings = dict(layout.shape_bindings)
    if len(bindings) != len(layout.shape_bindings):
        raise PackedStateLayoutError("shape bindings must be unique")
    names: set[str] = set()
    cursor = 0
    previous_category = -1
    category_order = {name: index for index, name in enumerate(FIELD_CATEGORIES)}
    expected_order = sorted(
        layout.fields,
        key=lambda field: (category_order.get(field.category, -1), field.name),
    )
    if list(layout.fields) != expected_order:
        raise PackedStateLayoutError("packed-state fields are not in canonical order")
    for field in layout.fields:
        if field.name in names:
            raise PackedStateLayoutError(f"duplicate packed-state field: {field.name!r}")
        names.add(field.name)
        if field.dtype_name not in _DTYPE_BYTES:
            raise PackedStateLayoutError(f"unsupported dtype for {field.name!r}")
        if field.category not in category_order or field.state_class not in STATE_CLASSES:
            raise PackedStateLayoutError(f"invalid classification for {field.name!r}")
        category_index = category_order[field.category]
        if category_index < previous_category:
            raise PackedStateLayoutError("packed-state categories are not monotonic")
        previous_category = category_index
        expected_shape = _resolve_shape(field.shape_template, bindings)
        if field.shape != expected_shape:
            raise PackedStateLayoutError(f"shape does not match template for {field.name!r}")
        expected_nbytes = _numel(field.shape) * _DTYPE_BYTES[field.dtype_name]
        if field.nbytes != expected_nbytes:
            raise PackedStateLayoutError(f"byte size mismatch for {field.name!r}")
        expected_storage_nbytes = (
            expected_nbytes if expected_nbytes else field.alignment_bytes
        )
        if field.storage_nbytes != expected_storage_nbytes:
            raise PackedStateLayoutError(
                f"storage byte size mismatch for {field.name!r}"
            )
        if field.alignment_bytes != layout.alignment_bytes:
            raise PackedStateLayoutError(f"alignment mismatch for {field.name!r}")
        expected_offset = _align(cursor, layout.alignment_bytes)
        if field.offset_bytes != expected_offset:
            raise PackedStateLayoutError(f"offset mismatch for {field.name!r}")
        if field.offset_bytes % field.alignment_bytes:
            raise PackedStateLayoutError(f"misaligned offset for {field.name!r}")
        cursor = field.end_offset_bytes
    expected_accepted = {template.name: template for template in _accepted_templates()}
    actual_accepted = {
        field.name: field
        for field in layout.fields
        if field.source == "accepted_manifest"
    }
    if set(actual_accepted) != set(expected_accepted):
        raise PackedStateLayoutError("layout must contain all 139 accepted fields exactly once")
    for name, template in expected_accepted.items():
        field = actual_accepted[name]
        if (
            field.dtype_name != template.dtype_name
            or field.shape_template != template.shape_template
            or field.category != template.category
            or field.state_class != template.state_class
        ):
            raise PackedStateLayoutError(
                f"accepted field contract mismatch for {name!r}"
            )
    expected_runtime = {template.name: template for template in _RUNTIME_TEMPLATES}
    actual_runtime = {
        field.name: field
        for field in layout.fields
        if field.source == "stage2_runtime_contract"
    }
    if actual_runtime and set(actual_runtime) != set(expected_runtime):
        raise PackedStateLayoutError(
            "runtime layout must contain every Stage2 runtime field exactly once"
        )
    for name, field in actual_runtime.items():
        template = expected_runtime[name]
        if (
            field.dtype_name != template.dtype_name
            or field.shape_template != template.shape_template
            or field.category != template.category
            or field.state_class != template.state_class
        ):
            raise PackedStateLayoutError(
                f"Stage2 runtime field contract mismatch for {name!r}"
            )
    expected_total = _align(cursor, layout.alignment_bytes)
    if layout.total_bytes != expected_total:
        raise PackedStateLayoutError("packed-state total byte size mismatch")
    alias_names: set[str] = set()
    for alias, target in layout.aliases:
        if alias in alias_names or alias in names:
            raise PackedStateLayoutError(f"duplicate or shadowing alias: {alias!r}")
        alias_names.add(alias)
        if target not in names:
            raise PackedStateLayoutError(f"alias {alias!r} targets absent field {target!r}")


def _validate_buffer(layout: PackedStateLayout, buffer: torch.Tensor) -> None:
    validate_packed_state_layout(layout)
    if not isinstance(buffer, torch.Tensor):
        raise PackedStateLayoutError("packed buffer must be a torch.Tensor")
    if buffer.dtype != torch.uint8 or buffer.ndim != 1 or not buffer.is_contiguous():
        raise PackedStateLayoutError("packed buffer must be contiguous rank-1 torch.uint8")
    if buffer.requires_grad:
        raise PackedStateLayoutError("packed buffer cannot require gradients")
    if buffer.numel() != layout.total_bytes:
        raise PackedStateLayoutError(
            f"packed buffer byte size mismatch: expected {layout.total_bytes}, got {buffer.numel()}"
        )


def packed_state_views(
    layout: PackedStateLayout,
    buffer: torch.Tensor,
    *,
    include_aliases: bool = False,
    accepted_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Return zero-copy typed views over a packed uint8 buffer."""

    _validate_buffer(layout, buffer)
    fields = layout.accepted_fields if accepted_only else layout.fields
    views = {
        field.name: buffer.narrow(0, field.offset_bytes, field.nbytes)
        .view(field.torch_dtype)
        .reshape(field.shape)
        for field in fields
    }
    if include_aliases:
        for alias, canonical in layout.aliases:
            if canonical in views:
                views[alias] = views[canonical]
    return views


def _pack_fields(
    layout: PackedStateLayout,
    tensors: Mapping[str, torch.Tensor],
    fields: tuple[PackedStateField, ...],
) -> torch.Tensor:
    validate_packed_state_layout(layout)
    expected_names = {field.name for field in fields}
    normalized: dict[str, tuple[str, torch.Tensor]] = {}
    for supplied_name, tensor in tensors.items():
        canonical = layout.resolve_name(supplied_name)
        if canonical not in expected_names:
            raise PackedStateLayoutError(
                f"field {supplied_name!r} is outside the requested packed surface"
            )
        previous = normalized.get(canonical)
        if previous is not None:
            raise PackedStateLayoutError(
                f"duplicate storage input for {canonical!r}: {previous[0]!r} and {supplied_name!r}"
            )
        if not isinstance(tensor, torch.Tensor):
            raise PackedStateLayoutError(f"packed-state input {supplied_name!r} is not a tensor")
        normalized[canonical] = (supplied_name, tensor)
    missing = sorted(expected_names - set(normalized))
    if missing:
        raise PackedStateLayoutError(f"packed-state tensor bank is missing fields: {missing}")

    first = normalized[fields[0].name][1]
    device = first.device
    buffer = torch.zeros(layout.total_bytes, dtype=torch.uint8, device=device)
    views = packed_state_views(layout, buffer)
    for field in fields:
        supplied_name, tensor = normalized[field.name]
        if tensor.device != device:
            raise PackedStateLayoutError(
                f"packed-state input {supplied_name!r} is on {tensor.device}, expected {device}"
            )
        if tensor.dtype != field.torch_dtype:
            raise PackedStateLayoutError(
                f"dtype mismatch for {supplied_name!r}: expected {field.dtype_name}, "
                f"got {str(tensor.dtype).removeprefix('torch.')}"
            )
        if tuple(tensor.shape) != field.shape:
            raise PackedStateLayoutError(
                f"shape mismatch for {supplied_name!r}: expected {field.shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise PackedStateLayoutError(f"packed-state input {supplied_name!r} must be contiguous")
        if tensor.requires_grad:
            raise PackedStateLayoutError(
                f"packed-state input {supplied_name!r} cannot require gradients"
            )
        views[field.name].copy_(tensor)
    return buffer


def pack_packed_state(
    layout: PackedStateLayout, tensors: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Pack the complete canonical layout into one uint8 tensor."""

    return _pack_fields(layout, tensors, layout.fields)


def pack_boundary_state(
    layout: PackedStateLayout, tensors: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Pack exactly the accepted 139-field boundary surface.

    Runtime-only and reusable-scratch ranges remain zero initialized.
    """

    return _pack_fields(layout, tensors, layout.accepted_fields)


def unpack_packed_state(
    layout: PackedStateLayout,
    buffer: torch.Tensor,
    *,
    accepted_only: bool = False,
    clone: bool = False,
    include_aliases: bool = False,
) -> dict[str, torch.Tensor]:
    """Expose compatibility views, optionally detached from packed storage."""

    views = packed_state_views(
        layout,
        buffer,
        include_aliases=include_aliases,
        accepted_only=accepted_only,
    )
    if clone:
        return {name: tensor.clone() for name, tensor in views.items()}
    return views


def build_production_packed_state_layout(
    cells: Sequence[Any],
    *,
    verify_source_artifacts: bool = True,
) -> tuple[PackedStateLayout, PackedStateProductionBinding]:
    binding = bind_production_stage1b_cells(cells)
    layout = build_packed_state_layout(
        binding.shape_bindings_by_name,
        verify_source_artifacts=verify_source_artifacts,
        production_binding=binding,
    )
    return layout, binding


def _validate_production_layout_binding(
    layout: PackedStateLayout,
    binding: PackedStateProductionBinding,
) -> None:
    validate_packed_state_layout(layout)
    if dict(layout.shape_bindings) != binding.shape_bindings_by_name:
        raise PackedStateLayoutError(
            "packed layout dimensions differ from the production binding"
        )
    if dict(layout.aliases) != binding.layout_aliases:
        raise PackedStateLayoutError(
            "packed layout aliases differ from the production binding"
        )
    expected_fields = set(_templates_by_name())
    if set(layout.fields_by_name) != expected_fields:
        raise PackedStateLayoutError(
            "production packed layout must contain every canonical storage field"
        )
    targets = set(binding.path_to_canonical_by_name.values())
    if targets != set(binding.representative_by_canonical_name):
        raise PackedStateLayoutError(
            "production path targets differ from canonical representatives"
        )
    if not targets <= expected_fields:
        raise PackedStateLayoutError(
            "production path binding targets absent canonical storage"
        )


def pack_production_stage1b_cells(
    layout: PackedStateLayout,
    binding: PackedStateProductionBinding,
    cells: Sequence[Any],
) -> torch.Tensor:
    """Copy each unique prepared production tensor once into packed storage."""

    from perf import cuda_architecture_b_stage1b as stage1b

    source_cells = tuple(cells)
    fresh_binding = bind_production_stage1b_cells(source_cells)
    if fresh_binding != binding:
        raise PackedStateLayoutError(
            "production cells changed after packed-state shape/alias binding"
        )
    _validate_production_layout_binding(layout, binding)
    flattened = [stage1b._flatten_cell(cell) for cell in source_cells]
    representative_by_canonical = binding.representative_by_canonical_name
    first_representative = next(iter(representative_by_canonical.values()))
    device = flattened[0][first_representative].device
    buffer = torch.zeros(layout.total_bytes, dtype=torch.uint8, device=device)
    views = packed_state_views(layout, buffer)
    for canonical, representative in sorted(representative_by_canonical.items()):
        target = views[canonical]
        for cell_index, flat in enumerate(flattened):
            source = flat[representative]
            destination = target[cell_index]
            if source.dtype != destination.dtype or source.shape != destination.shape:
                raise PackedStateLayoutError(
                    f"production source changed at {representative!r} cell {cell_index}"
                )
            destination.copy_(source)
    return buffer


def packed_production_path_views(
    layout: PackedStateLayout,
    binding: PackedStateProductionBinding,
    buffer: torch.Tensor,
    *,
    cell_index: int,
) -> dict[str, torch.Tensor]:
    """Return all real Stage1B paths as zero-copy packed cell views."""

    _validate_production_layout_binding(layout, binding)
    if type(cell_index) is not int or not 0 <= cell_index < binding.cell_count:
        raise PackedStateLayoutError(
            f"packed production cell index must be in [0, {binding.cell_count})"
        )
    banks = packed_state_views(layout, buffer)
    cell_views = {
        canonical: banks[canonical][cell_index]
        for canonical in binding.representative_by_canonical_name
    }
    result = {
        path: cell_views[canonical]
        for path, canonical in binding.path_to_canonical
    }
    if len(result) != binding.source_path_count:
        raise PackedStateLayoutError("packed production path view surface is incomplete")
    return result


def reconstruct_packed_stage1b_cell(
    source_cell: Any,
    layout: PackedStateLayout,
    binding: PackedStateProductionBinding,
    buffer: torch.Tensor,
    *,
    cell_index: int,
) -> Any:
    """Rebuild resident tensors and CudaGenerationBirthState from packed views."""

    from perf import cuda_architecture_b_stage1b as stage1b

    paths = packed_production_path_views(
        layout,
        binding,
        buffer,
        cell_index=cell_index,
    )
    source_flat = stage1b._flatten_cell(source_cell)
    source_aliases = stage1b._alias_map(source_flat)
    if set(paths) != set(source_flat):
        raise PackedStateLayoutError(
            "packed reconstruction paths differ from the source Stage1B cell"
        )
    path_to_canonical = binding.path_to_canonical_by_name
    for path, representative in source_aliases.items():
        if paths[path] is not paths[representative]:
            raise PackedStateLayoutError(
                f"packed reconstruction lost exact source alias identity at {path!r}"
            )
        if path_to_canonical[path] != path_to_canonical[representative]:
            raise PackedStateLayoutError(
                f"packed path binding contradicts Stage1B alias identity at {path!r}"
            )
    return stage1b.CudaStage1BResidentCell(
        cell_id=source_cell.cell_id,
        tensors=stage1b._mapped_values(paths, "resident"),
        birth_state=stage1b._rebuild_birth_state(source_cell.birth_state, paths),
        config=source_cell.config,
        deterministic=source_cell.deterministic,
    )


def packed_production_allocation_receipt(
    layout: PackedStateLayout,
    binding: PackedStateProductionBinding,
    buffer: torch.Tensor,
) -> dict[str, Any]:
    _validate_production_layout_binding(layout, binding)
    _validate_buffer(layout, buffer)
    paths = packed_production_path_views(
        layout,
        binding,
        buffer,
        cell_index=0,
    )
    canonical_views = {
        canonical: paths[representative]
        for canonical, representative in binding.representative_by_canonical
    }
    storage_ids = {
        (str(tensor.device), int(tensor.untyped_storage()._cdata))
        for tensor in canonical_views.values()
    }
    zero_copy = all(
        paths[path] is canonical_views[canonical]
        for path, canonical in binding.path_to_canonical
    )
    logical_bytes = sum(field.nbytes for field in layout.fields)
    return {
        "schema_version": "eiel_cuda_architecture_b_packed_allocation_receipt_v1",
        "world": binding.world,
        "cell_count": binding.cell_count,
        "source_path_count": binding.source_path_count,
        "source_unique_storage_count": binding.source_storage_count,
        "source_stage1b_alias_count": binding.stage1b_alias_count,
        "packed_canonical_field_count": len(layout.fields),
        "packed_present_source_field_count": len(canonical_views),
        "packed_absent_accepted_field_count": len(binding.absent_accepted_fields),
        "packed_layout_alias_count": len(layout.aliases),
        "packed_buffer_count": 1,
        "packed_unique_storage_count": len(storage_ids),
        "packed_buffer_bytes": int(buffer.numel()),
        "packed_logical_tensor_bytes": logical_bytes,
        "packed_alignment_and_empty_identity_bytes": layout.total_bytes - logical_bytes,
        "source_to_packed_copy_operations": binding.source_storage_count,
        "all_stage1b_paths_zero_copy_views": zero_copy,
        "all_canonical_views_share_packed_storage": len(storage_ids) == 1,
        "empty_fields_have_distinct_aligned_offsets": len(
            {
                field.offset_bytes
                for field in layout.fields
                if field.nbytes == 0
            }
        )
        == sum(field.nbytes == 0 for field in layout.fields),
    }


__all__ = [
    "FIELD_CATEGORIES",
    "PACKED_STATE_ALIGNMENT_BYTES",
    "PACKED_STATE_SCHEMA_VERSION",
    "PRODUCTION_PATH_COUNTS",
    "PRODUCTION_STAGE1B_ALIAS_COUNTS",
    "PRODUCTION_STORAGE_COUNTS",
    "STAGE2_EXTENSION",
    "STAGE2_EXTENSION_ABI_VERSION",
    "STAGE2_EXTENSION_SHA256",
    "PackedStateField",
    "PackedStateLayout",
    "PackedStateLayoutError",
    "PackedStateProductionBinding",
    "bind_production_stage1b_cells",
    "build_packed_state_layout",
    "build_production_packed_state_layout",
    "pack_boundary_state",
    "pack_packed_state",
    "pack_production_stage1b_cells",
    "packed_production_allocation_receipt",
    "packed_production_path_views",
    "packed_state_views",
    "reconstruct_packed_stage1b_cell",
    "unpack_packed_state",
    "validate_packed_state_layout",
]
