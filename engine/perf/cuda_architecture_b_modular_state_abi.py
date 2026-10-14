"""Pure-CPU fixed-shape modular storage overlay for Phase E.

This module is a storage contract only.  It does not load native code, move a
tensor to CUDA, execute a module, or change the accepted Architecture B state
manifest.  Hereditary strings and canonical payloads remain in an exact cold
sidecar; resident hot planes refer to them by checked integer indices.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Final, Mapping, Sequence

import torch

from agents.modular_genome_v1 import (
    B2_CAPACITY_PROFILES,
    b2_capacity_profile_id,
    canonical_json_bytes,
    canonical_payload_from_genome_dict,
    resolve_b2_caps,
)
from evolution.structural_development_v1 import (
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    effective_developmental_expression_record,
    validate_agent_developmental_graph,
)
from evolution.structural_gated_recurrent_v1 import (
    GATED_NUMERIC_FIELDS,
    GATED_OPERATOR,
    GATED_RECORD_KEY,
    normalize_gated_recurrent_record,
)
from evolution.structural_observation_routing_v1 import (
    ADMITTED_SEMANTIC_IDS,
    MAX_BINDINGS_PER_MODULE,
    ROUTING_BLOCK_KEY,
    normalize_routing_block,
)
from evolution.structural_output_routing_v1 import (
    ADMITTED_ACTION_SEMANTIC_IDS,
    MAX_OUTPUT_BINDINGS_PER_MODULE,
    OUTPUT_ROUTING_BLOCK_KEY,
    _runtime_action_index_by_semantic_id,
    normalize_output_routing_block,
)
from evolution.structural_sustainable_growth_v1 import (
    COST_COEFFICIENTS,
    module_is_dormant,
    normalize_activity_module_params,
    structural_cost_summary,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    ModularRecurrentMutationReceiptV1,
    ModularRecurrentMutationReceiptValidationError,
    mutation_ancestry_sha256_v1,
    validate_modular_recurrent_mutation_receipt_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_replay_v1 import (
    ModularRecurrentMutationReplayError,
    replay_modular_recurrent_mutation_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_sequence_v1 import (
    SCHEMA_VERSION as MODULAR_MUTATION_SEQUENCE_SCHEMA_VERSION_V1,
    ModularRecurrentMutationSequenceV1,
    ModularRecurrentMutationSequenceValidationError,
    validate_modular_recurrent_mutation_sequence_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_sequence_replay_v1 import (
    ModularRecurrentMutationSequenceReplayError,
    replay_modular_recurrent_mutation_sequence_v1,
)


MODULAR_STORAGE_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_modular_storage_overlay_v1"
)
MODULAR_STORAGE_PROJECTION_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_modular_storage_projection_v1"
)
MODULAR_STORAGE_PROJECTION_PREFIX: Final[str] = "phase_e_modular_v1__"
MODULAR_STORAGE_METADATA_KEY: Final[str] = "phase_e_modular_storage_v1"
PHYSICAL_MODULE_SLOTS: Final[int] = 16
PHYSICAL_CONNECTION_SLOTS: Final[int] = 48
MODULE_STATE_WIDTH: Final[int] = 1
MODULAR_RUNTIME_MUTABLE_FIELD_NAMES: Final[tuple[str, ...]] = (
    "module_state_present",
    "module_state",
    "module_output_present",
    "module_previous_output",
    "development_latch_present",
    "development_latch_state",
    "development_effective_gain_present",
    "development_effective_gain",
)
_MODULAR_RUNTIME_SIDECAR_MAP_NAMES: Final[tuple[str, ...]] = (
    "structural_live_module_state",
    "structural_live_module_outputs",
    "structural_development_latch_by_instance_id",
    "structural_development_effective_gain_by_instance_id",
)
INTEGRATOR_FIELDS: Final[tuple[str, ...]] = (
    "expression_scale",
    "live_retention",
    "live_input_gain",
    "live_output_gain",
)
COST_COUNT_FIELDS: Final[tuple[str, ...]] = tuple(COST_COEFFICIENTS)

_INPUT_SEMANTIC_CODE: Final[dict[str, int]] = {
    name: index for index, name in enumerate(ADMITTED_SEMANTIC_IDS)
}
_OUTPUT_SEMANTIC_CODE: Final[dict[str, int]] = {
    name: index for index, name in enumerate(ADMITTED_ACTION_SEMANTIC_IDS)
}
_OUTPUT_ACTION_INDEX: Final[dict[str, int]] = (
    _runtime_action_index_by_semantic_id()
)
_MODULE_TYPE_CODE: Final[dict[str, int]] = {
    "generic": 1,
    "policy_core": 2,
    "behavior_bias": 3,
    "life_history": 4,
    "identity_regulation": 5,
    "temperament_priors": 6,
}
_IDENTITY_ORIGIN_CODE: Final[dict[str, int]] = {
    "legacy_derived_unresolved": 1,
    "native": 2,
}
_OPERATOR_NONE = 0
_OPERATOR_INTEGRATOR = 1
_OPERATOR_GATED = 2


class ModularStorageError(RuntimeError):
    """A fixed-shape modular overlay was malformed or contradictory."""


@dataclass(frozen=True)
class ModularStorageFieldV1:
    name: str
    dtype_name: str
    shape: tuple[int, ...]
    fill_value: int | float | bool

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "bool": torch.bool,
            "float32": torch.float32,
            "float64": torch.float64,
            "int8": torch.int8,
            "int32": torch.int32,
            "int64": torch.int64,
        }[self.dtype_name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype_name,
            "shape": list(self.shape),
            "fill_value": self.fill_value,
        }


@dataclass(frozen=True)
class ModularStorageLayoutV1:
    schema_version: str
    profile_id: str
    agent_capacity: int
    logical_module_slots: int
    logical_connection_slots: int
    physical_module_slots: int
    physical_connection_slots: int
    fields: tuple[ModularStorageFieldV1, ...]
    manifest_sha256: str

    @property
    def fields_by_name(self) -> dict[str, ModularStorageFieldV1]:
        return {field.name: field for field in self.fields}

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "agent_capacity": self.agent_capacity,
            "logical_module_slots": self.logical_module_slots,
            "logical_connection_slots": self.logical_connection_slots,
            "physical_module_slots": self.physical_module_slots,
            "physical_connection_slots": self.physical_connection_slots,
            "module_state_width": MODULE_STATE_WIDTH,
            "fields": [field.to_dict() for field in self.fields],
        }

    def to_dict(self) -> dict[str, Any]:
        out = self._payload()
        out["manifest_sha256"] = self.manifest_sha256
        return out


@dataclass(frozen=True)
class PackedModularStorageV1:
    layout: ModularStorageLayoutV1
    tensors: dict[str, torch.Tensor]
    identity_table: tuple[str, ...]
    records_sidecar: tuple[dict[str, Any] | None, ...]
    sidecar_sha256: str
    packed_sha256: str

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(_packed_payload(self))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _field_specs(agent_capacity: int) -> tuple[ModularStorageFieldV1, ...]:
    a = agent_capacity
    m = PHYSICAL_MODULE_SLOTS
    c = PHYSICAL_CONNECTION_SLOTS
    i = MAX_BINDINGS_PER_MODULE
    o = MAX_OUTPUT_BINDINGS_PER_MODULE
    return (
        ModularStorageFieldV1("agent_occupied", "bool", (a,), False),
        ModularStorageFieldV1("agent_id", "int64", (a,), -1),
        ModularStorageFieldV1("agent_slot_token", "int64", (a,), -1),
        ModularStorageFieldV1("module_count", "int32", (a,), 0),
        ModularStorageFieldV1("connection_count", "int32", (a,), 0),
        ModularStorageFieldV1("module_occupied", "bool", (a, m), False),
        ModularStorageFieldV1("module_local_id_index", "int32", (a, m), -1),
        ModularStorageFieldV1("module_instance_id_index", "int32", (a, m), -1),
        ModularStorageFieldV1("module_homology_id_index", "int32", (a, m), -1),
        ModularStorageFieldV1("module_type_index", "int32", (a, m), -1),
        ModularStorageFieldV1("module_type_code", "int8", (a, m), 0),
        ModularStorageFieldV1("identity_origin_code", "int8", (a, m), 0),
        ModularStorageFieldV1("topology_position", "float32", (a, m, 2), 0.0),
        ModularStorageFieldV1("module_execution_order", "int32", (a, m), -1),
        ModularStorageFieldV1("dependency_topological_order", "int32", (a, m), -1),
        ModularStorageFieldV1("operator_code", "int8", (a, m), 0),
        ModularStorageFieldV1("integrator_parameters", "float32", (a, m, 4), 0.0),
        ModularStorageFieldV1(
            "gated_parameters", "float32", (a, m, len(GATED_NUMERIC_FIELDS)), 0.0
        ),
        ModularStorageFieldV1("module_dormant", "bool", (a, m), False),
        ModularStorageFieldV1("module_state_present", "bool", (a, m), False),
        ModularStorageFieldV1("module_state", "float32", (a, m, 1), 0.0),
        ModularStorageFieldV1("module_output_present", "bool", (a, m), False),
        ModularStorageFieldV1("module_previous_output", "float32", (a, m, 1), 0.0),
        ModularStorageFieldV1("input_binding_count", "int8", (a, m), 0),
        ModularStorageFieldV1("input_semantic_code", "int8", (a, m, i), -1),
        ModularStorageFieldV1("input_semantic_id_index", "int32", (a, m, i), -1),
        ModularStorageFieldV1("input_gain", "float32", (a, m, i), 0.0),
        ModularStorageFieldV1("input_enabled", "bool", (a, m, i), False),
        ModularStorageFieldV1("output_binding_count", "int8", (a, m), 0),
        ModularStorageFieldV1("output_semantic_code", "int8", (a, m, o), -1),
        ModularStorageFieldV1("output_semantic_id_index", "int32", (a, m, o), -1),
        ModularStorageFieldV1("output_action_index", "int8", (a, m, o), -1),
        ModularStorageFieldV1("output_gain", "float32", (a, m, o), 0.0),
        ModularStorageFieldV1("output_enabled", "bool", (a, m, o), False),
        ModularStorageFieldV1("development_explicit", "bool", (a, m), False),
        ModularStorageFieldV1("development_stage_gains", "float32", (a, m, 3), 0.0),
        ModularStorageFieldV1("development_prerequisite_count", "int8", (a, m), 0),
        ModularStorageFieldV1(
            "development_prerequisite_homology_index", "int32", (a, m, 2), -1
        ),
        ModularStorageFieldV1(
            "development_prerequisite_module_slot", "int32", (a, m, 2), -1
        ),
        ModularStorageFieldV1("development_latch_enabled", "bool", (a, m), False),
        ModularStorageFieldV1("development_latch_present", "bool", (a, m), False),
        ModularStorageFieldV1("development_latch_state", "bool", (a, m), False),
        ModularStorageFieldV1("development_effective_gain_present", "bool", (a, m), False),
        ModularStorageFieldV1("development_effective_gain", "float32", (a, m), 0.0),
        ModularStorageFieldV1("connection_occupied", "bool", (a, c), False),
        ModularStorageFieldV1("connection_local_id_index", "int32", (a, c), -1),
        ModularStorageFieldV1("connection_src_module_slot", "int32", (a, c), -1),
        ModularStorageFieldV1("connection_dst_module_slot", "int32", (a, c), -1),
        ModularStorageFieldV1("connection_src_slot_index", "int32", (a, c), -1),
        ModularStorageFieldV1("connection_dst_slot_index", "int32", (a, c), -1),
        ModularStorageFieldV1("connection_value_type_index", "int32", (a, c), -1),
        ModularStorageFieldV1("connection_channel_index", "int32", (a, c), -1),
        ModularStorageFieldV1("connection_weight", "float32", (a, c), 0.0),
        ModularStorageFieldV1("connection_execution_order", "int32", (a, c), -1),
        ModularStorageFieldV1(
            "structural_cost_counts", "int64", (a, len(COST_COUNT_FIELDS)), 0
        ),
        ModularStorageFieldV1(
            "structural_cost_components", "float64", (a, len(COST_COUNT_FIELDS)), 0.0
        ),
        ModularStorageFieldV1("structural_cost_total", "float64", (a,), 0.0),
    )


def build_modular_storage_layout(
    agent_capacity: int,
    profile_id: str = "b2-12x32-v1",
) -> ModularStorageLayoutV1:
    if type(agent_capacity) is not int or agent_capacity <= 0:
        raise ModularStorageError("agent_capacity must be a positive exact integer")
    profile = str(profile_id)
    if profile not in B2_CAPACITY_PROFILES:
        raise ModularStorageError(f"unsupported modular capacity profile: {profile}")
    caps = resolve_b2_caps(B2_CAPACITY_PROFILES[profile])
    provisional = ModularStorageLayoutV1(
        schema_version=MODULAR_STORAGE_SCHEMA_VERSION,
        profile_id=profile,
        agent_capacity=agent_capacity,
        logical_module_slots=int(caps["max_modules_per_agent"]),
        logical_connection_slots=int(caps["max_connections_per_agent"]),
        physical_module_slots=PHYSICAL_MODULE_SLOTS,
        physical_connection_slots=PHYSICAL_CONNECTION_SLOTS,
        fields=_field_specs(agent_capacity),
        manifest_sha256="",
    )
    return ModularStorageLayoutV1(
        **{**provisional.__dict__, "manifest_sha256": _digest(provisional._payload())}
    )


def _new_tensors(layout: ModularStorageLayoutV1) -> dict[str, torch.Tensor]:
    return {
        field.name: torch.full(
            field.shape,
            field.fill_value,
            dtype=field.torch_dtype,
            device="cpu",
        )
        for field in layout.fields
    }


def _finite_f32(value: Any, *, field: str, lower: float, upper: float) -> float:
    if isinstance(value, bool):
        raise ModularStorageError(f"{field} must be finite numeric")
    try:
        result = float(torch.tensor(float(value), dtype=torch.float32).item())
    except Exception as exc:
        raise ModularStorageError(f"{field} must be finite numeric") from exc
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ModularStorageError(f"{field} is outside [{lower}, {upper}]")
    return result


def _runtime_map(
    value: Any,
    *,
    field: str,
    valid_ids: set[str],
    lower: float,
    upper: float,
) -> dict[str, float]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ModularStorageError(f"{field} must be a mapping")
    out: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if key not in valid_ids:
            raise ModularStorageError(f"{field} contains orphan identity {key!r}")
        out[key] = _finite_f32(raw_value, field=f"{field}.{key}", lower=lower, upper=upper)
    return dict(sorted(out.items()))


def _runtime_bool_map(value: Any, *, field: str, valid_ids: set[str]) -> dict[str, bool]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ModularStorageError(f"{field} must be a mapping")
    out: dict[str, bool] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if key not in valid_ids:
            raise ModularStorageError(f"{field} contains orphan identity {key!r}")
        if type(raw_value) is not bool:
            raise ModularStorageError(f"{field}.{key} must be bool")
        out[key] = bool(raw_value)
    return dict(sorted(out.items()))


def _nonnegative_exact_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise ModularStorageError(
            f"{field} must be an exact nonnegative int64 value"
        )
    return int(value)


def _normalize_cost(
    raw: Any,
    *,
    modules: list[dict[str, Any]],
    connections: list[dict[str, Any]],
    serialized_genome_bytes: int,
    saturated_count: int,
) -> dict[str, Any]:
    if raw is not None:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("counts"), Mapping):
            raise ModularStorageError("cost_summary must contain a counts mapping")
        counts = dict(raw["counts"])
        if set(counts) != set(COST_COUNT_FIELDS):
            raise ModularStorageError("cost_summary count fields mismatch")
        result = structural_cost_summary(
            allocated_module_count=counts["allocated_module"],
            executing_integrator_count=counts["executing_integrator"],
            executing_gated_count=counts["executing_gated"],
            evaluated_connection_count=counts["evaluated_connection"],
            active_input_binding_count=counts["active_input_binding"],
            active_output_binding_count=counts["active_output_binding"],
            active_dependency_edge_count=counts["active_dependency_edge"],
            dormant_module_count=counts["dormant_module_maintenance"],
            serialized_genome_bytes=counts["serialized_genome_byte"],
            saturated_or_rejected_activity_count=counts[
                "saturated_or_rejected_activity"
            ],
        )
        return result

    generic = [m for m in modules if str(m.get("module_type")) == "generic"]
    dormant = [m for m in generic if module_is_dormant(m)]
    active = [m for m in generic if not module_is_dormant(m)]
    gated = [
        m
        for m in active
        if str(dict(m.get("params") or {}).get("live_operator", "bounded_scalar_integrator_v1"))
        == GATED_OPERATOR
    ]
    input_count = 0
    output_count = 0
    dependency_count = 0
    for module in active:
        params = dict(module.get("params") or {})
        if ROUTING_BLOCK_KEY in params:
            input_count += sum(
                bool(row["enabled"])
                for row in normalize_routing_block(params[ROUTING_BLOCK_KEY])["bindings"]
            )
        if OUTPUT_ROUTING_BLOCK_KEY in params:
            output_count += sum(
                bool(row["enabled"])
                for row in normalize_output_routing_block(params[OUTPUT_ROUTING_BLOCK_KEY])[
                    "bindings"
                ]
            )
        if DEVELOPMENTAL_EXPRESSION_BLOCK_KEY in params:
            dependency_count += len(
                effective_developmental_expression_record(
                    params[DEVELOPMENTAL_EXPRESSION_BLOCK_KEY]
                )["prerequisite_homology_ids"]
            )
    return structural_cost_summary(
        allocated_module_count=len(generic),
        executing_integrator_count=len(active) - len(gated),
        executing_gated_count=len(gated),
        evaluated_connection_count=len(connections),
        active_input_binding_count=input_count,
        active_output_binding_count=output_count,
        active_dependency_edge_count=dependency_count,
        dormant_module_count=len(dormant),
        serialized_genome_bytes=serialized_genome_bytes,
        saturated_or_rejected_activity_count=saturated_count,
    )


def _normalize_record(
    record: Mapping[str, Any], layout: ModularStorageLayoutV1
) -> dict[str, Any]:
    agent_id = _nonnegative_exact_int(record.get("agent_id"), field="agent_id")
    agent_slot_token = _nonnegative_exact_int(
        record.get("agent_slot_token"), field="agent_slot_token"
    )
    if "genome" not in record or not isinstance(record["genome"], Mapping):
        raise ModularStorageError("occupied agent record requires a genome mapping")
    raw_genome = deepcopy(dict(record["genome"]))
    raw_modules = raw_genome.get("modules")
    raw_connections = raw_genome.get("connections")
    if isinstance(raw_modules, list) and len(raw_modules) > layout.logical_module_slots:
        raise ModularStorageError("logical module capacity exceeded")
    if isinstance(raw_connections, list) and len(raw_connections) > layout.logical_connection_slots:
        raise ModularStorageError("logical connection capacity exceeded")
    try:
        canonical = canonical_payload_from_genome_dict(raw_genome)
    except Exception as exc:
        raise ModularStorageError(f"genome canonicalization failed: {exc}") from exc
    if b2_capacity_profile_id(canonical["b2_caps"]) != layout.profile_id:
        raise ModularStorageError("genome capacity profile disagrees with overlay layout")
    modules = [dict(value) for value in canonical["modules"]]
    connections = [dict(value) for value in canonical["connections"]]
    if len(modules) > layout.logical_module_slots or len(connections) > layout.logical_connection_slots:
        raise ModularStorageError("canonical genome exceeds logical profile")
    if isinstance(raw_modules, list) and len(modules) != len(raw_modules):
        raise ModularStorageError("canonicalization dropped or clipped a module")
    if isinstance(raw_connections, list) and len(connections) != len(raw_connections):
        raise ModularStorageError("canonicalization dropped or clipped a connection")

    instance_ids = [str(module.get("module_instance_id", "")) for module in modules]
    homology_ids = [str(module.get("homology_id", "")) for module in modules]
    local_ids = [str(module.get("module_id", "")) for module in modules]
    if any(not value for value in (*instance_ids, *homology_ids, *local_ids)):
        raise ModularStorageError("module identities must be nonempty")
    if len(set(instance_ids)) != len(instance_ids):
        raise ModularStorageError("duplicate module instance identity")
    if len(set(homology_ids)) != len(homology_ids):
        raise ModularStorageError("duplicate module homology identity")
    if len(set(local_ids)) != len(local_ids):
        raise ModularStorageError("duplicate module local identity")
    local_set = set(local_ids)
    for connection in connections:
        if (
            str(connection.get("src_module_id", "")) not in local_set
            or str(connection.get("dst_module_id", "")) not in local_set
        ):
            raise ModularStorageError("connection has a dangling module endpoint")

    valid_ids = set(instance_ids)
    states = _runtime_map(
        record.get("structural_live_module_state", {}),
        field="structural_live_module_state",
        valid_ids=valid_ids,
        lower=-1.0,
        upper=1.0,
    )
    outputs = _runtime_map(
        record.get("structural_live_module_outputs", {}),
        field="structural_live_module_outputs",
        valid_ids=valid_ids,
        lower=-1.0,
        upper=1.0,
    )
    if set(states) != set(outputs):
        raise ModularStorageError("module state and output identity sets differ")
    latches = _runtime_bool_map(
        record.get("structural_development_latch_by_instance_id", {}),
        field="structural_development_latch_by_instance_id",
        valid_ids=valid_ids,
    )
    effective = _runtime_map(
        record.get("structural_development_effective_gain_by_instance_id", {}),
        field="structural_development_effective_gain_by_instance_id",
        valid_ids=valid_ids,
        lower=0.0,
        upper=1.0,
    )
    gated_enabled = record.get("gated_recurrent_enabled", any(
        str(dict(module.get("params") or {}).get("live_operator", "")) == GATED_OPERATOR
        for module in modules
    ))
    if type(gated_enabled) is not bool:
        raise ModularStorageError("gated_recurrent_enabled must be bool")
    saturated = record.get("saturated_or_rejected_activity_count", 0)
    if type(saturated) is not int or saturated < 0:
        raise ModularStorageError("saturated_or_rejected_activity_count must be nonnegative int")
    cost = _normalize_cost(
        record.get("cost_summary"),
        modules=modules,
        connections=connections,
        serialized_genome_bytes=len(canonical_json_bytes(canonical)),
        saturated_count=saturated,
    )
    return {
        "agent_id": agent_id,
        "agent_slot_token": agent_slot_token,
        "genome": canonical,
        "structural_live_module_state": states,
        "structural_live_module_outputs": outputs,
        "structural_development_latch_by_instance_id": latches,
        "structural_development_effective_gain_by_instance_id": effective,
        "gated_recurrent_enabled": bool(gated_enabled),
        "saturated_or_rejected_activity_count": saturated,
        "cost_summary": cost,
    }


def _identity_strings(records: Sequence[dict[str, Any] | None]) -> tuple[str, ...]:
    strings: set[str] = set()
    for record in records:
        if record is None:
            continue
        genome = record["genome"]
        for module in genome["modules"]:
            for name in (
                "module_id",
                "module_instance_id",
                "homology_id",
                "module_type",
            ):
                strings.add(str(module.get(name, "")))
            params = dict(module.get("params") or {})
            for block_name in (ROUTING_BLOCK_KEY, OUTPUT_ROUTING_BLOCK_KEY):
                block = params.get(block_name)
                if isinstance(block, Mapping):
                    for binding in block.get("bindings", []):
                        strings.add(str(binding.get("semantic_id", "")))
        for connection in genome["connections"]:
            for name in (
                "connection_id",
                "src_slot",
                "dst_slot",
                "value_type",
                "channel",
            ):
                strings.add(str(connection.get(name, "")))
    strings.discard("")
    return tuple(sorted(strings))


def _stable_module_key(module: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(module.get("homology_id", "")),
        str(module.get("module_instance_id", "")),
        str(module.get("module_id", "")),
    )


def _fill_record(
    tensors: dict[str, torch.Tensor],
    agent_slot: int,
    record: dict[str, Any],
    identity_index: Mapping[str, int],
) -> None:
    modules = [dict(value) for value in record["genome"]["modules"]]
    connections = [dict(value) for value in record["genome"]["connections"]]
    physical_modules = sorted(modules, key=_stable_module_key)
    physical_by_local = {
        str(module["module_id"]): index for index, module in enumerate(physical_modules)
    }
    physical_by_homology = {
        str(module["homology_id"]): index for index, module in enumerate(physical_modules)
    }
    tensors["agent_occupied"][agent_slot] = True
    tensors["agent_id"][agent_slot] = int(record["agent_id"])
    tensors["agent_slot_token"][agent_slot] = int(record["agent_slot_token"])
    tensors["module_count"][agent_slot] = len(modules)
    tensors["connection_count"][agent_slot] = len(connections)

    gated_enabled = bool(record["gated_recurrent_enabled"])
    if gated_enabled:
        execution_modules = sorted(
            (m for m in modules if str(m.get("module_type")) == "generic"),
            key=_stable_module_key,
        )
    else:
        execution_modules = sorted(
            (m for m in modules if str(m.get("module_type")) == "generic"),
            key=lambda value: str(value.get("module_id", "")),
        )
    for order, module in enumerate(execution_modules):
        tensors["module_execution_order"][agent_slot, order] = physical_by_local[
            str(module["module_id"])
        ]

    dependency_order = validate_agent_developmental_graph(modules)
    for order, module in enumerate(dependency_order):
        tensors["dependency_topological_order"][agent_slot, order] = physical_by_local[
            str(module["module_id"])
        ]

    states = record["structural_live_module_state"]
    outputs = record["structural_live_module_outputs"]
    latches = record["structural_development_latch_by_instance_id"]
    effective = record["structural_development_effective_gain_by_instance_id"]
    for slot, module in enumerate(physical_modules):
        local_id = str(module["module_id"])
        instance_id = str(module["module_instance_id"])
        homology_id = str(module["homology_id"])
        module_type = str(module["module_type"])
        params = normalize_activity_module_params(module.get("params", {}))
        tensors["module_occupied"][agent_slot, slot] = True
        tensors["module_local_id_index"][agent_slot, slot] = identity_index[local_id]
        tensors["module_instance_id_index"][agent_slot, slot] = identity_index[instance_id]
        tensors["module_homology_id_index"][agent_slot, slot] = identity_index[homology_id]
        tensors["module_type_index"][agent_slot, slot] = identity_index[module_type]
        tensors["module_type_code"][agent_slot, slot] = _MODULE_TYPE_CODE.get(module_type, 127)
        tensors["identity_origin_code"][agent_slot, slot] = _IDENTITY_ORIGIN_CODE.get(
            str(module.get("identity_origin", "")), 127
        )
        tensors["topology_position"][agent_slot, slot] = torch.tensor(
            module.get("topology_pos", (0.0, 0.0)), dtype=torch.float32
        )
        operator = str(params.get("live_operator", "bounded_scalar_integrator_v1"))
        if module_type == "generic" and operator == GATED_OPERATOR:
            tensors["operator_code"][agent_slot, slot] = _OPERATOR_GATED
            gated = normalize_gated_recurrent_record(params.get(GATED_RECORD_KEY))
            tensors["gated_parameters"][agent_slot, slot] = torch.tensor(
                [gated[name] for name in GATED_NUMERIC_FIELDS], dtype=torch.float32
            )
            tensors["integrator_parameters"][agent_slot, slot, 0] = float(
                params.get("expression_scale", 0.0)
            )
        elif module_type == "generic" and operator == "bounded_scalar_integrator_v1":
            tensors["operator_code"][agent_slot, slot] = _OPERATOR_INTEGRATOR
            tensors["integrator_parameters"][agent_slot, slot] = torch.tensor(
                [float(params.get(name, 0.0)) for name in INTEGRATOR_FIELDS],
                dtype=torch.float32,
            )
        elif module_type == "generic":
            raise ModularStorageError(f"unsupported live operator: {operator}")
        tensors["module_dormant"][agent_slot, slot] = module_is_dormant(module)
        if instance_id in states:
            tensors["module_state_present"][agent_slot, slot] = True
            tensors["module_state"][agent_slot, slot, 0] = states[instance_id]
            tensors["module_output_present"][agent_slot, slot] = True
            tensors["module_previous_output"][agent_slot, slot, 0] = outputs[instance_id]

        input_block = params.get(ROUTING_BLOCK_KEY)
        if input_block is not None:
            bindings = normalize_routing_block(input_block)["bindings"]
            tensors["input_binding_count"][agent_slot, slot] = len(bindings)
            for binding_slot, binding in enumerate(bindings):
                semantic = str(binding["semantic_id"])
                tensors["input_semantic_code"][agent_slot, slot, binding_slot] = (
                    _INPUT_SEMANTIC_CODE[semantic]
                )
                tensors["input_semantic_id_index"][agent_slot, slot, binding_slot] = (
                    identity_index[semantic]
                )
                tensors["input_gain"][agent_slot, slot, binding_slot] = binding["gain"]
                tensors["input_enabled"][agent_slot, slot, binding_slot] = binding["enabled"]

        output_block = params.get(OUTPUT_ROUTING_BLOCK_KEY)
        if output_block is not None:
            bindings = normalize_output_routing_block(output_block)["bindings"]
            tensors["output_binding_count"][agent_slot, slot] = len(bindings)
            for binding_slot, binding in enumerate(bindings):
                semantic = str(binding["semantic_id"])
                code = _OUTPUT_SEMANTIC_CODE[semantic]
                tensors["output_semantic_code"][agent_slot, slot, binding_slot] = code
                tensors["output_action_index"][agent_slot, slot, binding_slot] = (
                    _OUTPUT_ACTION_INDEX[semantic]
                )
                tensors["output_semantic_id_index"][agent_slot, slot, binding_slot] = (
                    identity_index[semantic]
                )
                tensors["output_gain"][agent_slot, slot, binding_slot] = binding["gain"]
                tensors["output_enabled"][agent_slot, slot, binding_slot] = binding["enabled"]

        development_raw = params.get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY)
        development = effective_developmental_expression_record(development_raw)
        tensors["development_explicit"][agent_slot, slot] = development_raw is not None
        tensors["development_stage_gains"][agent_slot, slot] = torch.tensor(
            development["stage_gains"], dtype=torch.float32
        )
        prerequisites = list(development["prerequisite_homology_ids"])
        tensors["development_prerequisite_count"][agent_slot, slot] = len(prerequisites)
        for prerequisite_slot, prerequisite in enumerate(prerequisites):
            tensors["development_prerequisite_homology_index"][
                agent_slot, slot, prerequisite_slot
            ] = identity_index[prerequisite]
            tensors["development_prerequisite_module_slot"][
                agent_slot, slot, prerequisite_slot
            ] = physical_by_homology[prerequisite]
        tensors["development_latch_enabled"][agent_slot, slot] = development[
            "latch_dependency_readiness"
        ]
        if instance_id in latches:
            tensors["development_latch_present"][agent_slot, slot] = True
            tensors["development_latch_state"][agent_slot, slot] = latches[instance_id]
        if instance_id in effective:
            tensors["development_effective_gain_present"][agent_slot, slot] = True
            tensors["development_effective_gain"][agent_slot, slot] = effective[instance_id]

    def connection_key(connection: Mapping[str, Any]) -> tuple[Any, ...]:
        if not gated_enabled:
            return (str(connection.get("connection_id", "")),)
        src = physical_modules[physical_by_local[str(connection["src_module_id"])]]
        dst = physical_modules[physical_by_local[str(connection["dst_module_id"])]]
        return (
            _stable_module_key(src),
            _stable_module_key(dst),
            str(connection.get("src_slot", "")),
            str(connection.get("dst_slot", "")),
            str(connection.get("value_type", "")),
            str(connection.get("channel", "")),
            float(connection.get("weight", 0.0)),
            str(connection.get("connection_id", "")),
        )

    physical_connections = sorted(connections, key=connection_key)
    for slot, connection in enumerate(physical_connections):
        tensors["connection_occupied"][agent_slot, slot] = True
        tensors["connection_local_id_index"][agent_slot, slot] = identity_index[
            str(connection["connection_id"])
        ]
        tensors["connection_src_module_slot"][agent_slot, slot] = physical_by_local[
            str(connection["src_module_id"])
        ]
        tensors["connection_dst_module_slot"][agent_slot, slot] = physical_by_local[
            str(connection["dst_module_id"])
        ]
        for field_name, source_name in (
            ("connection_src_slot_index", "src_slot"),
            ("connection_dst_slot_index", "dst_slot"),
            ("connection_value_type_index", "value_type"),
            ("connection_channel_index", "channel"),
        ):
            tensors[field_name][agent_slot, slot] = identity_index[str(connection[source_name])]
        tensors["connection_weight"][agent_slot, slot] = float(connection["weight"])
        tensors["connection_execution_order"][agent_slot, slot] = slot

    cost = record["cost_summary"]
    for index, name in enumerate(COST_COUNT_FIELDS):
        tensors["structural_cost_counts"][agent_slot, index] = int(cost["counts"][name])
        tensors["structural_cost_components"][agent_slot, index] = float(
            cost["components"][name]
        )
    tensors["structural_cost_total"][agent_slot] = float(cost["total"])


def _tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    raw = tensor.detach().contiguous().numpy().tobytes().hex()
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "bytes_hex": raw,
    }


def _sidecar_payload(packed: PackedModularStorageV1) -> dict[str, Any]:
    return {
        "identity_table": list(packed.identity_table),
        "records": list(packed.records_sidecar),
    }


def _packed_payload(packed: PackedModularStorageV1) -> dict[str, Any]:
    return {
        "layout": packed.layout.to_dict(),
        "sidecar": _sidecar_payload(packed),
        "sidecar_sha256": packed.sidecar_sha256,
        "tensors": {
            name: _tensor_payload(tensor)
            for name, tensor in sorted(packed.tensors.items())
        },
    }


def pack_modular_storage(
    records_by_agent_slot: Sequence[Mapping[str, Any] | None],
    *,
    agent_capacity: int,
    profile_id: str = "b2-12x32-v1",
) -> PackedModularStorageV1:
    layout = build_modular_storage_layout(agent_capacity, profile_id)
    records = tuple(records_by_agent_slot)
    if len(records) > agent_capacity:
        raise ModularStorageError("agent record count exceeds agent_capacity")
    normalized: list[dict[str, Any] | None] = []
    for record in records:
        if record is None:
            normalized.append(None)
        elif isinstance(record, Mapping):
            normalized.append(_normalize_record(record, layout))
        else:
            raise ModularStorageError("agent record must be a mapping or None")
    normalized.extend([None] * (agent_capacity - len(normalized)))
    normalized_tuple = tuple(normalized)
    occupied_records = [record for record in normalized_tuple if record is not None]
    agent_ids = [int(record["agent_id"]) for record in occupied_records]
    if len(set(agent_ids)) != len(agent_ids):
        raise ModularStorageError("duplicate occupied agent_id binding")
    identity_table = _identity_strings(normalized_tuple)
    identity_index = {value: index for index, value in enumerate(identity_table)}
    tensors = _new_tensors(layout)
    for agent_slot, record in enumerate(normalized_tuple):
        if record is not None:
            _fill_record(tensors, agent_slot, record, identity_index)
    provisional = PackedModularStorageV1(
        layout=layout,
        tensors=tensors,
        identity_table=identity_table,
        records_sidecar=deepcopy(normalized_tuple),
        sidecar_sha256="",
        packed_sha256="",
    )
    sidecar_sha = _digest(_sidecar_payload(provisional))
    with_sidecar = PackedModularStorageV1(
        **{**provisional.__dict__, "sidecar_sha256": sidecar_sha}
    )
    packed = PackedModularStorageV1(
        **{**with_sidecar.__dict__, "packed_sha256": _digest(_packed_payload(with_sidecar))}
    )
    validate_modular_storage(packed, verify_semantic_repack=False)
    return packed


def maybe_pack_modular_storage(
    enabled: bool,
    records_by_agent_slot: Any,
    *,
    agent_capacity: Any = None,
    profile_id: Any = None,
) -> PackedModularStorageV1 | None:
    """Return before inspecting any other input when the overlay is disabled."""

    if enabled is False:
        return None
    if enabled is not True:
        raise ModularStorageError("enabled must be an exact bool")
    return pack_modular_storage(
        records_by_agent_slot,
        agent_capacity=agent_capacity,
        profile_id="b2-12x32-v1" if profile_id is None else profile_id,
    )


def _validate_layout(layout: ModularStorageLayoutV1) -> None:
    if layout.schema_version != MODULAR_STORAGE_SCHEMA_VERSION:
        raise ModularStorageError("layout schema mismatch")
    if layout.physical_module_slots != 16 or layout.physical_connection_slots != 48:
        raise ModularStorageError("physical modular envelope mismatch")
    expected = build_modular_storage_layout(layout.agent_capacity, layout.profile_id)
    if layout != expected:
        raise ModularStorageError("layout digest or field contract mismatch")


def validate_modular_storage(
    packed: PackedModularStorageV1,
    *,
    verify_semantic_repack: bool = True,
) -> None:
    if not isinstance(packed, PackedModularStorageV1):
        raise ModularStorageError("packed value has the wrong type")
    _validate_layout(packed.layout)
    expected_names = set(packed.layout.fields_by_name)
    if set(packed.tensors) != expected_names:
        raise ModularStorageError("packed tensor field set mismatch")
    for field in packed.layout.fields:
        tensor = packed.tensors[field.name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != field.torch_dtype
            or tuple(tensor.shape) != field.shape
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ModularStorageError(f"invalid tensor plane: {field.name}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ModularStorageError(f"nonfinite tensor plane: {field.name}")
    if len(packed.records_sidecar) != packed.layout.agent_capacity:
        raise ModularStorageError("sidecar agent capacity mismatch")
    if tuple(sorted(set(packed.identity_table))) != packed.identity_table:
        raise ModularStorageError("identity sidecar is not unique canonical UTF-8 order")
    if any(not isinstance(value, str) or not value for value in packed.identity_table):
        raise ModularStorageError("identity sidecar contains an invalid string")
    if _digest(_sidecar_payload(packed)) != packed.sidecar_sha256:
        raise ModularStorageError("identity/payload sidecar digest mismatch")
    if _digest(_packed_payload(packed)) != packed.packed_sha256:
        raise ModularStorageError("packed modular storage digest mismatch")
    occupied = packed.tensors["agent_occupied"]
    for slot, record in enumerate(packed.records_sidecar):
        if bool(occupied[slot].item()) != (record is not None):
            raise ModularStorageError("agent occupancy contradicts sidecar")
        expected_agent_id = -1 if record is None else int(record["agent_id"])
        expected_slot_token = -1 if record is None else int(record["agent_slot_token"])
        if int(packed.tensors["agent_id"][slot].item()) != expected_agent_id:
            raise ModularStorageError("agent_id plane contradicts sidecar")
        if (
            int(packed.tensors["agent_slot_token"][slot].item())
            != expected_slot_token
        ):
            raise ModularStorageError("agent_slot_token plane contradicts sidecar")
        module_count = int(packed.tensors["module_count"][slot].item())
        connection_count = int(packed.tensors["connection_count"][slot].item())
        if module_count > packed.layout.logical_module_slots:
            raise ModularStorageError("logical module cap violated")
        if connection_count > packed.layout.logical_connection_slots:
            raise ModularStorageError("logical connection cap violated")
        module_mask = packed.tensors["module_occupied"][slot]
        connection_mask = packed.tensors["connection_occupied"][slot]
        expected_module_mask = torch.arange(PHYSICAL_MODULE_SLOTS) < module_count
        expected_connection_mask = torch.arange(PHYSICAL_CONNECTION_SLOTS) < connection_count
        if not torch.equal(module_mask, expected_module_mask):
            raise ModularStorageError("module occupancy is not canonical prefix form")
        if not torch.equal(connection_mask, expected_connection_mask):
            raise ModularStorageError("connection occupancy is not canonical prefix form")
    if verify_semantic_repack:
        expected = pack_modular_storage(
            packed.records_sidecar,
            agent_capacity=packed.layout.agent_capacity,
            profile_id=packed.layout.profile_id,
        )
        if expected.identity_table != packed.identity_table:
            raise ModularStorageError("identity table contradicts canonical sidecar")
        for name in sorted(expected.tensors):
            if not torch.equal(expected.tensors[name], packed.tensors[name]):
                raise ModularStorageError(f"tensor plane contradicts sidecar: {name}")


def unpack_modular_storage(
    packed: PackedModularStorageV1,
) -> tuple[dict[str, Any] | None, ...]:
    validate_modular_storage(packed)
    return deepcopy(packed.records_sidecar)


def _plane_sha256(tensor: torch.Tensor) -> str:
    if not isinstance(tensor, torch.Tensor):
        raise ModularStorageError("projection plane must be a tensor")
    return sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def _projection_manifest(packed: PackedModularStorageV1) -> list[dict[str, Any]]:
    return [
        {
            "logical_name": field.name,
            "dtype": field.dtype_name,
            "shape": list(field.shape),
            "sha256": _plane_sha256(packed.tensors[field.name]),
        }
        for field in packed.layout.fields
    ]


def _projection_envelope_without_digest(
    packed: PackedModularStorageV1,
) -> dict[str, Any]:
    return {
        "projection_schema_version": MODULAR_STORAGE_PROJECTION_SCHEMA_VERSION,
        "storage_schema_version": MODULAR_STORAGE_SCHEMA_VERSION,
        "layout": packed.layout.to_dict(),
        "profile_id": packed.layout.profile_id,
        "identity_table": list(packed.identity_table),
        "records_sidecar": deepcopy(list(packed.records_sidecar)),
        "sidecar_sha256": packed.sidecar_sha256,
        "packed_sha256": packed.packed_sha256,
        "tensor_manifest": _projection_manifest(packed),
    }


def modular_storage_projection_payload(
    packed: PackedModularStorageV1,
) -> dict[str, dict[str, Any]]:
    """Encode one exact namespaced projection without aliasing packed storage."""

    validate_modular_storage(packed)
    envelope = _projection_envelope_without_digest(packed)
    envelope["envelope_sha256"] = _digest(envelope)
    return {
        "metadata": {MODULAR_STORAGE_METADATA_KEY: envelope},
        "tensors": {
            f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}": tensor.clone()
            for name, tensor in packed.tensors.items()
        },
    }


def rebind_modular_runtime_storage(
    authority: PackedModularStorageV1,
    live_tensors: Mapping[str, Any],
) -> PackedModularStorageV1:
    """Rebind the eight live runtime planes and refresh checkpoint digests.

    The cold genome, identity table, slot ownership, topology, costs, and all
    other planes remain exact to a previously validated packed authority.  The
    result must be reproducible from the updated runtime sidecar byte-for-byte;
    otherwise the live state is not checkpoint-representable and is rejected.
    """

    validate_modular_storage(authority)
    if not isinstance(live_tensors, Mapping):
        raise ModularStorageError("live modular tensors must be a mapping")
    logical_names = tuple(field.name for field in authority.layout.fields)
    logical_set = set(logical_names)
    projected_set = {
        f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}" for name in logical_names
    }
    keys = {str(name) for name in live_tensors}
    logical_present = keys & logical_set
    projected_present = keys & projected_set
    mixed = sorted(
        name
        for name in keys
        if name.startswith("phase_e_modular_") and name not in projected_set
    )
    if mixed or (logical_present and projected_present):
        raise ModularStorageError("mixed live modular tensor namespaces")
    if logical_present == logical_set:
        live = {name: live_tensors[name] for name in logical_names}
    elif projected_present == projected_set:
        live = {
            name: live_tensors[f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}"]
            for name in logical_names
        }
    else:
        raise ModularStorageError("live modular tensor field set is incomplete")

    mutable = set(MODULAR_RUNTIME_MUTABLE_FIELD_NAMES)
    for field in authority.layout.fields:
        tensor = live[field.name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != field.torch_dtype
            or tuple(tensor.shape) != field.shape
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ModularStorageError(
                f"invalid live modular tensor plane: {field.name}"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ModularStorageError(
                f"nonfinite live modular tensor plane: {field.name}"
            )
        if field.name not in mutable and not torch.equal(
            tensor, authority.tensors[field.name]
        ):
            raise ModularStorageError(
                f"live checkpoint changed cold/static plane: {field.name}"
            )

    records = deepcopy(list(authority.records_sidecar))
    occupied = live["module_occupied"]
    for agent_slot, record in enumerate(records):
        if record is None:
            continue
        state: dict[str, float] = {}
        output: dict[str, float] = {}
        latch: dict[str, bool] = {}
        effective: dict[str, float] = {}
        for module_slot in range(PHYSICAL_MODULE_SLOTS):
            if not bool(occupied[agent_slot, module_slot].item()):
                continue
            identity_index = int(
                live["module_instance_id_index"][agent_slot, module_slot].item()
            )
            if not 0 <= identity_index < len(authority.identity_table):
                raise ModularStorageError(
                    "live module instance identity index is out of range"
                )
            instance_id = authority.identity_table[identity_index]
            if bool(live["module_state_present"][agent_slot, module_slot].item()):
                state[instance_id] = float(
                    live["module_state"][agent_slot, module_slot, 0].item()
                )
            if bool(live["module_output_present"][agent_slot, module_slot].item()):
                output[instance_id] = float(
                    live["module_previous_output"][agent_slot, module_slot, 0].item()
                )
            if bool(
                live["development_latch_present"][agent_slot, module_slot].item()
            ):
                latch[instance_id] = bool(
                    live["development_latch_state"][agent_slot, module_slot].item()
                )
            if bool(
                live["development_effective_gain_present"][
                    agent_slot, module_slot
                ].item()
            ):
                effective[instance_id] = float(
                    live["development_effective_gain"][
                        agent_slot, module_slot
                    ].item()
                )
        if set(state) != set(output):
            raise ModularStorageError(
                "live module state and output identity sets differ"
            )
        record["structural_live_module_state"] = state
        record["structural_live_module_outputs"] = output
        record["structural_development_latch_by_instance_id"] = latch
        record["structural_development_effective_gain_by_instance_id"] = effective

    refreshed = pack_modular_storage(
        records,
        agent_capacity=authority.layout.agent_capacity,
        profile_id=authority.layout.profile_id,
    )
    if refreshed.identity_table != authority.identity_table:
        raise ModularStorageError("live checkpoint changed the identity table")
    for name in logical_names:
        if not torch.equal(refreshed.tensors[name], live[name]):
            raise ModularStorageError(
                f"live runtime plane is not sidecar-representable: {name}"
            )
    return refreshed


def modular_storage_live_projection_payload(
    authority: PackedModularStorageV1,
    live_tensors: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Refresh live runtime authority and encode an exact checkpoint overlay."""

    return modular_storage_projection_payload(
        rebind_modular_runtime_storage(authority, live_tensors)
    )


def rebind_modular_lifecycle_storage(
    authority: PackedModularStorageV1,
    *,
    profile_id: str,
    terminal_alive_mask: torch.Tensor,
    terminal_slot_to_agent_id: torch.Tensor,
    terminal_slot_tokens: torch.Tensor,
    selected_parent_slot_by_child_slot: Mapping[int, int],
    live_tensors: Mapping[str, Any] | None = None,
    mutation_receipt: (
        Mapping[str, object]
        | ModularRecurrentMutationReceiptV1
        | ModularRecurrentMutationSequenceV1
        | None
    ) = None,
) -> PackedModularStorageV1:
    """Rebind cold modular authority after births, replacement, and death.

    Parent slots address the validated pre-lifecycle authority.  Each newly
    occupied or replaced terminal child row must have exactly one explicit
    parent entry.  Unchanged owners preserve their complete cold record, while
    children inherit the selected parent's hereditary structure with neutral
    runtime and developmental runtime maps.
    """

    validate_modular_storage(authority)
    if type(profile_id) is not str or profile_id != authority.layout.profile_id:
        raise ModularStorageError("modular lifecycle profile mismatch")

    capacity = authority.layout.agent_capacity
    terminal_fields = (
        ("terminal_alive_mask", terminal_alive_mask, torch.bool),
        ("terminal_slot_to_agent_id", terminal_slot_to_agent_id, torch.int64),
        ("terminal_slot_tokens", terminal_slot_tokens, torch.int64),
    )
    for name, tensor, dtype in terminal_fields:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != dtype
            or tuple(tensor.shape) != (capacity,)
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ModularStorageError(f"invalid modular lifecycle tensor: {name}")

    if bool(torch.any(terminal_slot_to_agent_id[terminal_alive_mask] < 0).item()):
        raise ModularStorageError("occupied terminal owner ID must be nonnegative")
    if bool(torch.any(terminal_slot_tokens[terminal_alive_mask] < 0).item()):
        raise ModularStorageError("occupied terminal slot token must be nonnegative")
    if bool(torch.any(terminal_slot_to_agent_id[~terminal_alive_mask] != -1).item()):
        raise ModularStorageError("retired terminal owner ID must be -1")
    occupied_ids = terminal_slot_to_agent_id[terminal_alive_mask].tolist()
    if len(set(occupied_ids)) != len(occupied_ids):
        raise ModularStorageError("duplicate occupied terminal owner ID")

    if not isinstance(selected_parent_slot_by_child_slot, Mapping):
        raise ModularStorageError("selected-parent mapping must be a mapping")
    selected_parents: dict[int, int] = {}
    for child_slot, parent_slot in selected_parent_slot_by_child_slot.items():
        if type(child_slot) is not int or type(parent_slot) is not int:
            raise ModularStorageError(
                "selected-parent mapping slots must be exact integers"
            )
        if not 0 <= child_slot < capacity:
            raise ModularStorageError("selected-parent child slot is out of range")
        if not 0 <= parent_slot < capacity:
            raise ModularStorageError("selected-parent slot is out of range")
        if authority.records_sidecar[parent_slot] is None:
            raise ModularStorageError("selected-parent slot has no cold authority")
        selected_parents[child_slot] = parent_slot

    changed_child_slots: set[int] = set()
    for slot in range(capacity):
        if not bool(terminal_alive_mask[slot].item()):
            continue
        prior = authority.records_sidecar[slot]
        if (
            prior is None
            or int(prior["agent_id"])
            != int(terminal_slot_to_agent_id[slot].item())
            or int(prior["agent_slot_token"])
            != int(terminal_slot_tokens[slot].item())
        ):
            changed_child_slots.add(slot)
    if set(selected_parents) != changed_child_slots:
        missing = sorted(changed_child_slots - set(selected_parents))
        extra = sorted(set(selected_parents) - changed_child_slots)
        raise ModularStorageError(
            "selected-parent mapping does not exactly cover changed child slots "
            f"missing={missing!r};extra={extra!r}"
        )

    validated_receipt: ModularRecurrentMutationReceiptV1 | None = None
    validated_sequence: ModularRecurrentMutationSequenceV1 | None = None
    receipt_ordinal_by_child_slot: dict[int, int] = {}
    if mutation_receipt is not None:
        to_dict = getattr(mutation_receipt, "to_dict", None)
        mutation_wire = to_dict() if callable(to_dict) else mutation_receipt
        mutation_schema = (
            mutation_wire.get("schema_version")
            if isinstance(mutation_wire, Mapping)
            else None
        )
    else:
        mutation_wire = None
        mutation_schema = None
    if mutation_schema == MODULAR_MUTATION_SEQUENCE_SCHEMA_VERSION_V1:
        try:
            validated_sequence = validate_modular_recurrent_mutation_sequence_v1(
                mutation_wire
            )
        except ModularRecurrentMutationSequenceValidationError as exc:
            raise ModularStorageError(
                f"modular lifecycle mutation sequence is invalid: {exc}"
            ) from exc
        if validated_sequence.profile_id != authority.layout.profile_id:
            raise ModularStorageError(
                "modular lifecycle mutation sequence profile mismatch"
            )
        if validated_sequence.source_packed_sha256 != authority.packed_sha256:
            raise ModularStorageError(
                "modular lifecycle mutation sequence source authority mismatch"
            )
    elif mutation_receipt is not None:
        try:
            validated_receipt = validate_modular_recurrent_mutation_receipt_v1(
                mutation_wire
            )
        except ModularRecurrentMutationReceiptValidationError as exc:
            raise ModularStorageError(
                f"modular lifecycle mutation receipt is invalid: {exc}"
            ) from exc
        if validated_receipt.profile_id != authority.layout.profile_id:
            raise ModularStorageError(
                "modular lifecycle mutation receipt profile mismatch"
            )
        if validated_receipt.source_packed_sha256 != authority.packed_sha256:
            raise ModularStorageError(
                "modular lifecycle mutation receipt source authority mismatch"
            )
        if len(validated_receipt.transactions) != 1:
            raise ModularStorageError(
                "bounded modular lifecycle mutation checkpoint requires exactly one "
                "transaction"
            )
        transaction = validated_receipt.transactions[0]
        if transaction.transaction_kind not in {"ordinary_birth", "turnover"}:
            raise ModularStorageError(
                "modular lifecycle mutation receipt has an unsupported transaction kind"
            )
        if set(selected_parents) != {transaction.child_slot}:
            raise ModularStorageError(
                "modular lifecycle mutation receipt does not exactly cover changed "
                "child slots"
            )
        expected_parent_slot = selected_parents[transaction.child_slot]
        parent = authority.records_sidecar[expected_parent_slot]
        if parent is None:
            raise ModularStorageError("selected-parent slot has no cold authority")
        if (
            transaction.parent_slot != expected_parent_slot
            or transaction.parent_agent_id != int(parent["agent_id"])
            or transaction.parent_slot_token != int(parent["agent_slot_token"])
        ):
            raise ModularStorageError(
                "modular lifecycle mutation receipt parent ownership mismatch"
            )
        if (
            transaction.child_agent_id
            != int(terminal_slot_to_agent_id[transaction.child_slot].item())
            or transaction.child_slot_token
            != int(terminal_slot_tokens[transaction.child_slot].item())
        ):
            raise ModularStorageError(
                "modular lifecycle mutation receipt child ownership mismatch"
            )
        expected_ancestry_sha256 = mutation_ancestry_sha256_v1(
            transaction_kind=transaction.transaction_kind,
            parent_slot=transaction.parent_slot,
            child_slot=transaction.child_slot,
            parent_agent_id=transaction.parent_agent_id,
            child_agent_id=transaction.child_agent_id,
            child_slot_token=transaction.child_slot_token,
        )
        if validated_receipt.ancestry_tape_sha256 != expected_ancestry_sha256:
            raise ModularStorageError(
                "modular lifecycle mutation receipt ancestry mismatch"
            )
        receipt_ordinal_by_child_slot[transaction.child_slot] = 0

    if validated_sequence is not None:
        try:
            sequence_replay = replay_modular_recurrent_mutation_sequence_v1(
                validated_sequence,
                authority.records_sidecar,
                identity_table=authority.identity_table,
            )
        except ModularRecurrentMutationSequenceReplayError as exc:
            raise ModularStorageError(
                f"modular lifecycle mutation sequence replay failed: {exc}"
            ) from exc
        records = deepcopy(list(sequence_replay.final_records))
        for slot, record in enumerate(records):
            alive = bool(terminal_alive_mask[slot].item())
            if not alive:
                records[slot] = None
                continue
            if record is None:
                raise ModularStorageError(
                    "modular lifecycle mutation sequence terminal occupancy mismatch"
                )
            if (
                int(record["agent_id"])
                != int(terminal_slot_to_agent_id[slot].item())
                or int(record["agent_slot_token"])
                != int(terminal_slot_tokens[slot].item())
            ):
                raise ModularStorageError(
                    "modular lifecycle mutation sequence terminal owner mismatch"
                )
            if slot in changed_child_slots:
                origin_slot = sequence_replay.final_origin_slot_by_record_slot[slot]
                if origin_slot != selected_parents[slot]:
                    raise ModularStorageError(
                        "modular lifecycle mutation sequence origin mismatch"
                    )
                for name in _MODULAR_RUNTIME_SIDECAR_MAP_NAMES:
                    record[name] = {}
    else:
        records: list[dict[str, Any] | None] = [None] * capacity
        for slot in range(capacity):
            if not bool(terminal_alive_mask[slot].item()):
                continue
            if slot not in changed_child_slots:
                survivor = authority.records_sidecar[slot]
                if survivor is None:
                    raise ModularStorageError(
                        "unchanged terminal owner has no cold authority"
                    )
                records[slot] = deepcopy(survivor)
                continue
            child = deepcopy(authority.records_sidecar[selected_parents[slot]])
            if child is None:
                raise ModularStorageError("selected-parent slot has no cold authority")
            if slot in receipt_ordinal_by_child_slot:
                if validated_receipt is None:
                    raise ModularStorageError(
                        "validated mutation receipt is unavailable"
                    )
                try:
                    child["genome"] = replay_modular_recurrent_mutation_v1(
                        child["genome"],
                        validated_receipt,
                        identity_table=authority.identity_table,
                        transaction_ordinal=receipt_ordinal_by_child_slot[slot],
                    )
                except ModularRecurrentMutationReplayError as exc:
                    raise ModularStorageError(
                        f"modular lifecycle mutation replay failed: {exc}"
                    ) from exc
            child["agent_id"] = int(terminal_slot_to_agent_id[slot].item())
            child["agent_slot_token"] = int(terminal_slot_tokens[slot].item())
            for name in _MODULAR_RUNTIME_SIDECAR_MAP_NAMES:
                child[name] = {}
            records[slot] = child

    expected_live_structure: PackedModularStorageV1 | None = None
    if validated_sequence is not None:
        expected_live_structure = pack_modular_storage(
            records,
            agent_capacity=capacity,
            profile_id=profile_id,
        )
        if expected_live_structure.identity_table != authority.identity_table:
            raise ModularStorageError(
                "modular lifecycle mutation sequence changed the bounded identity table"
            )
    elif validated_receipt is not None:
        comparison_records = deepcopy(list(authority.records_sidecar))
        for slot in changed_child_slots:
            comparison_records[slot] = deepcopy(records[slot])
        expected_live_structure = pack_modular_storage(
            comparison_records,
            agent_capacity=capacity,
            profile_id=profile_id,
        )
        if expected_live_structure.identity_table != authority.identity_table:
            raise ModularStorageError(
                "modular lifecycle mutation changed the bounded identity table"
            )

    if live_tensors is not None:
        if not isinstance(live_tensors, Mapping):
            raise ModularStorageError("live modular tensors must be a mapping")
        logical_names = tuple(field.name for field in authority.layout.fields)
        logical_set = set(logical_names)
        projected_set = {
            f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}" for name in logical_names
        }
        keys = {str(name) for name in live_tensors}
        if keys & logical_set == logical_set:
            live = {name: live_tensors[name] for name in logical_names}
        elif keys & projected_set == projected_set:
            live = {
                name: live_tensors[f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}"]
                for name in logical_names
            }
        else:
            raise ModularStorageError("live modular tensor field set is incomplete")
        normalized_tokens = torch.where(
            terminal_alive_mask,
            terminal_slot_tokens,
            torch.full_like(terminal_slot_tokens, -1),
        )
        if (
            not torch.equal(live["agent_occupied"], terminal_alive_mask)
            or not torch.equal(live["agent_id"], terminal_slot_to_agent_id)
            or not torch.equal(live["agent_slot_token"], normalized_tokens)
        ):
            raise ModularStorageError(
                "live modular membership contradicts lifecycle terminal owner"
            )
        mutable = set(MODULAR_RUNTIME_MUTABLE_FIELD_NAMES)
        inactive = ~terminal_alive_mask
        for field in authority.layout.fields:
            tensor = live[field.name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != field.torch_dtype
                or tuple(tensor.shape) != field.shape
                or not tensor.is_contiguous()
                or tensor.requires_grad
            ):
                raise ModularStorageError(
                    f"invalid live modular lifecycle plane: {field.name}"
                )
            if tensor.is_floating_point() and not bool(
                torch.isfinite(tensor).all().item()
            ):
                raise ModularStorageError(
                    f"nonfinite live modular lifecycle plane: {field.name}"
                )
            expected_dead = torch.full_like(tensor[inactive], field.fill_value)
            if not torch.equal(tensor[inactive], expected_dead):
                raise ModularStorageError(
                    f"lifecycle boundary left a dead modular row uncleared: {field.name}"
                )
            if field.name in mutable or field.name in {
                "agent_occupied",
                "agent_id",
                "agent_slot_token",
            }:
                continue
            for slot in range(capacity):
                if not bool(terminal_alive_mask[slot].item()):
                    continue
                authority_slot = selected_parents.get(slot, slot)
                expected_tensor = (
                    authority.tensors[field.name][authority_slot]
                    if expected_live_structure is None
                    else expected_live_structure.tensors[field.name][slot]
                )
                if not torch.equal(tensor[slot], expected_tensor):
                    raise ModularStorageError(
                        "lifecycle boundary changed hereditary modular plane: "
                        f"{field.name} slot={slot} parent={authority_slot}"
                    )

        occupied_modules = live["module_occupied"]
        for agent_slot, record in enumerate(records):
            if record is None:
                continue
            state: dict[str, float] = {}
            output: dict[str, float] = {}
            latch: dict[str, bool] = {}
            effective: dict[str, float] = {}
            for module_slot in range(PHYSICAL_MODULE_SLOTS):
                if not bool(occupied_modules[agent_slot, module_slot].item()):
                    continue
                identity_index = int(
                    live["module_instance_id_index"][
                        agent_slot, module_slot
                    ].item()
                )
                if not 0 <= identity_index < len(authority.identity_table):
                    raise ModularStorageError(
                        "live lifecycle module identity index is out of range"
                    )
                instance_id = authority.identity_table[identity_index]
                if bool(
                    live["module_state_present"][agent_slot, module_slot].item()
                ):
                    state[instance_id] = float(
                        live["module_state"][agent_slot, module_slot, 0].item()
                    )
                if bool(
                    live["module_output_present"][agent_slot, module_slot].item()
                ):
                    output[instance_id] = float(
                        live["module_previous_output"][
                            agent_slot, module_slot, 0
                        ].item()
                    )
                if bool(
                    live["development_latch_present"][
                        agent_slot, module_slot
                    ].item()
                ):
                    latch[instance_id] = bool(
                        live["development_latch_state"][
                            agent_slot, module_slot
                        ].item()
                    )
                if bool(
                    live["development_effective_gain_present"][
                        agent_slot, module_slot
                    ].item()
                ):
                    effective[instance_id] = float(
                        live["development_effective_gain"][
                            agent_slot, module_slot
                        ].item()
                    )
            if set(state) != set(output):
                raise ModularStorageError(
                    "live lifecycle module state and output identity sets differ"
                )
            record["structural_live_module_state"] = state
            record["structural_live_module_outputs"] = output
            record["structural_development_latch_by_instance_id"] = latch
            record["structural_development_effective_gain_by_instance_id"] = effective

    return pack_modular_storage(
        records,
        agent_capacity=capacity,
        profile_id=profile_id,
    )


def rebind_modular_generation_boundary_storage(
    authority: PackedModularStorageV1,
    live_tensors: Mapping[str, Any],
    *,
    alive_mask: torch.Tensor,
    slot_to_agent_id: torch.Tensor,
    slot_tokens: torch.Tensor,
) -> PackedModularStorageV1:
    """Rebind one live generation boundary, including ordinary deaths.

    The native generation lane forbids births and turnover.  A boundary may
    therefore retain an existing occupied row or remove it after an ordinary
    death, but it may not introduce or replace an owner.  Surviving cold rows
    remain byte-exact to the prior authority.  Dead rows must already have
    been cleared to the committed ABI fill values by the resident lane.

    Repacking is intentional here: removing a cold record may change canonical
    identity-table indexes for surviving rows.  That index migration happens
    only at the generation/checkpoint boundary, never between resident ticks.
    """

    validate_modular_storage(authority)
    if not isinstance(live_tensors, Mapping):
        raise ModularStorageError("live modular tensors must be a mapping")
    logical_names = tuple(field.name for field in authority.layout.fields)
    logical_set = set(logical_names)
    projected_set = {
        f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}" for name in logical_names
    }
    keys = {str(name) for name in live_tensors}
    logical_present = keys & logical_set
    projected_present = keys & projected_set
    mixed = sorted(
        name
        for name in keys
        if name.startswith("phase_e_modular_") and name not in projected_set
    )
    if mixed or (logical_present and projected_present):
        raise ModularStorageError("mixed live modular tensor namespaces")
    if logical_present == logical_set:
        live = {name: live_tensors[name] for name in logical_names}
    elif projected_present == projected_set:
        live = {
            name: live_tensors[f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}"]
            for name in logical_names
        }
    else:
        raise ModularStorageError("live modular tensor field set is incomplete")

    capacity = authority.layout.agent_capacity
    owner_fields = (
        ("alive_mask", alive_mask, torch.bool),
        ("slot_to_agent_id", slot_to_agent_id, torch.int64),
        ("slot_tokens", slot_tokens, torch.int64),
    )
    for name, tensor, dtype in owner_fields:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != dtype
            or tuple(tensor.shape) != (capacity,)
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ModularStorageError(
                f"invalid generation-boundary owner tensor: {name}"
            )
    normalized_tokens = torch.where(
        alive_mask,
        slot_tokens,
        torch.full_like(slot_tokens, -1),
    )
    if (
        not torch.equal(live["agent_occupied"], alive_mask)
        or not torch.equal(live["agent_id"], slot_to_agent_id)
        or not torch.equal(live["agent_slot_token"], normalized_tokens)
    ):
        raise ModularStorageError(
            "live modular membership contradicts generation-boundary owner"
        )
    prior_occupied = authority.tensors["agent_occupied"]
    if bool(torch.any(torch.logical_and(alive_mask, ~prior_occupied)).item()):
        raise ModularStorageError(
            "generation boundary introduced an owner while births are forbidden"
        )
    if not torch.equal(
        slot_to_agent_id[alive_mask],
        authority.tensors["agent_id"][alive_mask],
    ) or not torch.equal(
        normalized_tokens[alive_mask],
        authority.tensors["agent_slot_token"][alive_mask],
    ):
        raise ModularStorageError(
            "generation boundary replaced a surviving modular owner"
        )

    mutable = set(MODULAR_RUNTIME_MUTABLE_FIELD_NAMES)
    inactive = ~alive_mask
    for field in authority.layout.fields:
        tensor = live[field.name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != field.torch_dtype
            or tuple(tensor.shape) != field.shape
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ModularStorageError(
                f"invalid live modular tensor plane: {field.name}"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ModularStorageError(
                f"nonfinite live modular tensor plane: {field.name}"
            )
        if field.name not in mutable and not torch.equal(
            tensor[alive_mask], authority.tensors[field.name][alive_mask]
        ):
            raise ModularStorageError(
                f"generation boundary changed surviving cold/static plane: {field.name}"
            )
        expected_dead = torch.full_like(tensor[inactive], field.fill_value)
        if not torch.equal(tensor[inactive], expected_dead):
            raise ModularStorageError(
                f"generation boundary left a dead modular row uncleared: {field.name}"
            )

    records = deepcopy(list(authority.records_sidecar))
    occupied_modules = live["module_occupied"]
    for agent_slot in range(capacity):
        if not bool(alive_mask[agent_slot].item()):
            records[agent_slot] = None
            continue
        record = records[agent_slot]
        if record is None:
            raise ModularStorageError("surviving modular owner has no cold record")
        state: dict[str, float] = {}
        output: dict[str, float] = {}
        latch: dict[str, bool] = {}
        effective: dict[str, float] = {}
        for module_slot in range(PHYSICAL_MODULE_SLOTS):
            if not bool(occupied_modules[agent_slot, module_slot].item()):
                continue
            identity_index = int(
                live["module_instance_id_index"][agent_slot, module_slot].item()
            )
            if not 0 <= identity_index < len(authority.identity_table):
                raise ModularStorageError(
                    "live module instance identity index is out of range"
                )
            instance_id = authority.identity_table[identity_index]
            if bool(live["module_state_present"][agent_slot, module_slot].item()):
                state[instance_id] = float(
                    live["module_state"][agent_slot, module_slot, 0].item()
                )
            if bool(live["module_output_present"][agent_slot, module_slot].item()):
                output[instance_id] = float(
                    live["module_previous_output"][agent_slot, module_slot, 0].item()
                )
            if bool(
                live["development_latch_present"][agent_slot, module_slot].item()
            ):
                latch[instance_id] = bool(
                    live["development_latch_state"][agent_slot, module_slot].item()
                )
            if bool(
                live["development_effective_gain_present"][
                    agent_slot, module_slot
                ].item()
            ):
                effective[instance_id] = float(
                    live["development_effective_gain"][agent_slot, module_slot].item()
                )
        if set(state) != set(output):
            raise ModularStorageError(
                "live module state and output identity sets differ"
            )
        record["structural_live_module_state"] = state
        record["structural_live_module_outputs"] = output
        record["structural_development_latch_by_instance_id"] = latch
        record["structural_development_effective_gain_by_instance_id"] = effective

    refreshed = pack_modular_storage(
        records,
        agent_capacity=capacity,
        profile_id=authority.layout.profile_id,
    )
    for name in sorted(mutable):
        if not torch.equal(
            refreshed.tensors[name][alive_mask], live[name][alive_mask]
        ):
            raise ModularStorageError(
                "live generation-boundary runtime plane is not exactly "
                f"sidecar-representable: {name}"
            )
    if (
        not torch.equal(refreshed.tensors["agent_occupied"], alive_mask)
        or not torch.equal(refreshed.tensors["agent_id"], slot_to_agent_id)
        or not torch.equal(
            refreshed.tensors["agent_slot_token"], normalized_tokens
        )
    ):
        raise ModularStorageError(
            "repacked generation boundary contradicts owner identity"
        )
    return refreshed


def modular_storage_generation_boundary_projection_payload(
    authority: PackedModularStorageV1,
    live_tensors: Mapping[str, Any],
    *,
    alive_mask: torch.Tensor,
    slot_to_agent_id: torch.Tensor,
    slot_tokens: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    """Encode an exact checkpoint projection after a live generation."""

    return modular_storage_projection_payload(
        rebind_modular_generation_boundary_storage(
            authority,
            live_tensors,
            alive_mask=alive_mask,
            slot_to_agent_id=slot_to_agent_id,
            slot_tokens=slot_tokens,
        )
    )


def modular_storage_lifecycle_projection_payload(
    authority: PackedModularStorageV1,
    live_tensors: Mapping[str, Any],
    *,
    profile_id: str,
    alive_mask: torch.Tensor,
    slot_to_agent_id: torch.Tensor,
    slot_tokens: torch.Tensor,
    selected_parent_slot_by_child_slot: Mapping[int, int],
    mutation_receipt: (
        Mapping[str, object]
        | ModularRecurrentMutationReceiptV1
        | ModularRecurrentMutationSequenceV1
        | None
    ) = None,
) -> dict[str, dict[str, Any]]:
    """Encode a lifecycle-aware generation-boundary modular overlay."""

    return modular_storage_projection_payload(
        rebind_modular_lifecycle_storage(
            authority,
            profile_id=profile_id,
            terminal_alive_mask=alive_mask,
            terminal_slot_to_agent_id=slot_to_agent_id,
            terminal_slot_tokens=slot_tokens,
            selected_parent_slot_by_child_slot=selected_parent_slot_by_child_slot,
            live_tensors=live_tensors,
            mutation_receipt=mutation_receipt,
        )
    )


def maybe_modular_storage_projection_payload(
    enabled: bool,
    records_by_agent_slot: Any,
    *,
    agent_capacity: Any = None,
    profile_id: Any = None,
) -> dict[str, dict[str, Any]]:
    """Build the projection only when explicitly enabled.

    The disabled return occurs before any other argument is inspected.
    """

    if enabled is False:
        return {"metadata": {}, "tensors": {}}
    if enabled is not True:
        raise ModularStorageError("enabled must be an exact bool")
    packed = pack_modular_storage(
        records_by_agent_slot,
        agent_capacity=agent_capacity,
        profile_id="b2-12x32-v1" if profile_id is None else profile_id,
    )
    return modular_storage_projection_payload(packed)


def _mixed_projection_keys(
    metadata: Mapping[str, Any], tensors: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    mixed_metadata = sorted(
        str(name)
        for name in metadata
        if str(name).startswith("phase_e_modular_storage_")
        and str(name) != MODULAR_STORAGE_METADATA_KEY
    )
    mixed_tensors = sorted(
        str(name)
        for name in tensors
        if str(name).startswith("phase_e_modular_")
        and not str(name).startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    )
    return mixed_metadata, mixed_tensors


def modular_storage_from_projection(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> PackedModularStorageV1 | None:
    """Decode and fully validate an overlay from an overall projection."""

    if not isinstance(metadata, Mapping) or not isinstance(tensors, Mapping):
        raise ModularStorageError("projection metadata and tensors must be mappings")
    mixed_metadata, mixed_tensors = _mixed_projection_keys(metadata, tensors)
    if mixed_metadata or mixed_tensors:
        raise ModularStorageError(
            "mixed modular projection version "
            f"metadata={mixed_metadata!r};tensors={mixed_tensors!r}"
        )
    projected_names = {
        str(name): value
        for name, value in tensors.items()
        if str(name).startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    }
    if MODULAR_STORAGE_METADATA_KEY not in metadata:
        if projected_names:
            raise ModularStorageError("modular projection tensors exist without metadata")
        return None
    raw_envelope = metadata.get(MODULAR_STORAGE_METADATA_KEY)
    if not isinstance(raw_envelope, Mapping):
        raise ModularStorageError("modular projection envelope must be a mapping")
    envelope = deepcopy(dict(raw_envelope))
    expected_envelope_fields = {
        "projection_schema_version",
        "storage_schema_version",
        "layout",
        "profile_id",
        "identity_table",
        "records_sidecar",
        "sidecar_sha256",
        "packed_sha256",
        "tensor_manifest",
        "envelope_sha256",
    }
    if set(envelope) != expected_envelope_fields:
        raise ModularStorageError("modular projection envelope fields mismatch")
    if (
        envelope["projection_schema_version"]
        != MODULAR_STORAGE_PROJECTION_SCHEMA_VERSION
        or envelope["storage_schema_version"] != MODULAR_STORAGE_SCHEMA_VERSION
    ):
        raise ModularStorageError("modular projection schema mismatch")
    supplied_envelope_digest = envelope.pop("envelope_sha256")
    if type(supplied_envelope_digest) is not str or _digest(envelope) != supplied_envelope_digest:
        raise ModularStorageError("modular projection envelope digest mismatch")
    raw_layout = envelope.get("layout")
    if not isinstance(raw_layout, Mapping):
        raise ModularStorageError("modular projection layout must be a mapping")
    raw_agent_capacity = raw_layout.get("agent_capacity")
    raw_profile_id = envelope.get("profile_id")
    if type(raw_agent_capacity) is not int or type(raw_profile_id) is not str:
        raise ModularStorageError("modular projection layout binding is malformed")
    layout = build_modular_storage_layout(raw_agent_capacity, raw_profile_id)
    if dict(raw_layout) != layout.to_dict():
        raise ModularStorageError("modular projection layout is not exact")

    manifest = envelope.get("tensor_manifest")
    if not isinstance(manifest, list) or any(not isinstance(row, Mapping) for row in manifest):
        raise ModularStorageError("modular projection tensor manifest is malformed")
    expected_logical_names = [field.name for field in layout.fields]
    manifest_names = [str(row.get("logical_name", "")) for row in manifest]
    if manifest_names != expected_logical_names or len(set(manifest_names)) != len(manifest_names):
        raise ModularStorageError("modular projection tensor manifest field order mismatch")
    expected_projected_names = {
        f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}" for name in expected_logical_names
    }
    actual_projected_names = set(projected_names)
    if actual_projected_names != expected_projected_names:
        raise ModularStorageError(
            "modular projection tensor field set mismatch "
            f"missing={sorted(expected_projected_names-actual_projected_names)!r};"
            f"extra={sorted(actual_projected_names-expected_projected_names)!r}"
        )

    cloned: dict[str, torch.Tensor] = {}
    for field, row in zip(layout.fields, manifest):
        if set(row) != {"logical_name", "dtype", "shape", "sha256"}:
            raise ModularStorageError("modular projection plane manifest fields mismatch")
        if row["dtype"] != field.dtype_name or row["shape"] != list(field.shape):
            raise ModularStorageError(f"modular projection plane descriptor mismatch: {field.name}")
        projected_name = f"{MODULAR_STORAGE_PROJECTION_PREFIX}{field.name}"
        tensor = projected_names[projected_name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != field.torch_dtype
            or tuple(tensor.shape) != field.shape
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ModularStorageError(f"invalid projected tensor plane: {field.name}")
        if type(row["sha256"]) is not str or _plane_sha256(tensor) != row["sha256"]:
            raise ModularStorageError(f"projected tensor plane digest mismatch: {field.name}")
        cloned[field.name] = tensor.clone()

    identity_table_raw = envelope.get("identity_table")
    records_raw = envelope.get("records_sidecar")
    if not isinstance(identity_table_raw, list) or not isinstance(records_raw, list):
        raise ModularStorageError("modular projection sidecar is malformed")
    packed = PackedModularStorageV1(
        layout=layout,
        tensors=cloned,
        identity_table=tuple(identity_table_raw),
        records_sidecar=tuple(deepcopy(records_raw)),
        sidecar_sha256=envelope.get("sidecar_sha256"),
        packed_sha256=envelope.get("packed_sha256"),
    )
    validate_modular_storage(packed)
    return packed


__all__ = [
    "COST_COUNT_FIELDS",
    "MODULAR_STORAGE_SCHEMA_VERSION",
    "MODULAR_STORAGE_METADATA_KEY",
    "MODULAR_STORAGE_PROJECTION_PREFIX",
    "MODULAR_STORAGE_PROJECTION_SCHEMA_VERSION",
    "MODULAR_RUNTIME_MUTABLE_FIELD_NAMES",
    "PHYSICAL_CONNECTION_SLOTS",
    "PHYSICAL_MODULE_SLOTS",
    "ModularStorageError",
    "ModularStorageFieldV1",
    "ModularStorageLayoutV1",
    "PackedModularStorageV1",
    "build_modular_storage_layout",
    "maybe_pack_modular_storage",
    "maybe_modular_storage_projection_payload",
    "modular_storage_generation_boundary_projection_payload",
    "modular_storage_lifecycle_projection_payload",
    "modular_storage_live_projection_payload",
    "modular_storage_from_projection",
    "modular_storage_projection_payload",
    "pack_modular_storage",
    "rebind_modular_generation_boundary_storage",
    "rebind_modular_lifecycle_storage",
    "rebind_modular_runtime_storage",
    "unpack_modular_storage",
    "validate_modular_storage",
]
