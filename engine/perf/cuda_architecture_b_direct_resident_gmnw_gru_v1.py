"""Prepare and execute the bounded GMNW stable-GRU resident CUDA owner."""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Final

import torch

from engine.evolution.gmnw_v1 import GMNW_TASK_OBSERVATION_COLUMNS
from engine.perf.cuda_architecture_b_direct_resident_execution_v2 import (
    create_run_control_v2,
)
from engine.perf.cuda_architecture_b_direct_resident_gru_v3 import (
    GRU_ACTION_COUNT_V3,
    GRU_HIDDEN_WIDTH_V3,
    GRU_SCRATCH_WIDTH_V3,
    PreparedDirectResidentGruV3,
    _set_checkpoint_policy_family_metadata_v3,
    build_prepared_direct_resident_gru_v3,
    project_direct_resident_gru_checkpoint_v3,
    project_direct_resident_gru_multicell_checkpoints_v3,
    stable_gru_packed_width_v3,
)
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
    GENERATION_ARGUMENT_INDEX_V1,
    GENERATION_COUNT_ARGUMENT_INDEX_V1,
    PreparedDirectResidentTrajectoryV1,
    prepare_direct_resident_trajectory_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    ORDINARY_BIRTH_OUTPUT_FIELDS,
    PHYSICAL_WINDOW_TICKS_V1,
    _allocate_birth_outputs,  # noqa: F401 - retained test hook
    _allocate_gmnw_qualification_trace_outputs,
    _allocate_tick_outputs,  # noqa: F401 - retained test hook
)
from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
    DirectResidentSourceAllocationRequestV1,
)
from engine.perf.cxxbackbone10_resident_state_owner import (
    _with_checkpoint_contract_metadata,
)


GMNW_OBSERVATION_COLUMNS_V1: Final[int] = 12
GMNW_COUNTER_COLUMNS_V1: Final[int] = 6
PATCH_GRU_OBSERVATION_CHANNELS_V1: Final[tuple[str, ...]] = (
    "legacy_occupancy_or_food",
    "patch_food_norm",
    "patch_cover",
    "patch_shelter",
    "patch_scent_norm",
    "patch_waste_norm",
    "patch_density_norm",
    "patch_movement_cost_norm",
)
GMNW_GRU_OBSERVATION_COLUMNS_V1: Final[tuple[str, ...]] = tuple(
    f"{channel}[dy={dy},dx={dx}]"
    for channel in PATCH_GRU_OBSERVATION_CHANNELS_V1
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
) + (
    "agent_energy_over_max_energy",
    "agent_health_over_start_health",
    *GMNW_TASK_OBSERVATION_COLUMNS,
)
GMNW_GRU_OBSERVATION_REGISTRY_SHA256_V1: Final[str] = (
    "486f868d8a169583030a019bbeb0a3de66ff6db20364fce140d81a9591818066"
)
GMNW_WITNESS_SCHEMA_VERSION_V1: Final[str] = "eiel_gmnw_stable_gru_witness_v1"
GMNW_WITNESS_SCHEMA_VERSION_V2: Final[str] = "eiel_gmnw_stable_gru_witness_v2"
GMNW_WITNESS_V2_PARENT_SHA256: Final[str] = (
    "908207785421f2dc8dc347cd165b4876f8d025b11d22ca424b2a440298aea115"
)
GMNW_CAPABILITY_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_flattened_gmnw_gru_capability_v1"
)
GMNW_DIRECT_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_gmnw_gru_v1"
)
GMNW_PHYSICAL_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_gmnw_gru_physical_v1"
)
DEFAULT_WITNESS_PATH_V1: Final[Path] = Path(
    public_runtime_path('evidence_raw/eiel_gru_memory_necessity_20260721/04_witness/stable_gru_witness_v1.pt')
)
DEFAULT_MATCHED_MLP_PATH_V1: Final[Path] = Path(
    public_runtime_path('evidence_raw/eiel_gru_memory_necessity_20260721/04_witness/matched_mlp_control_v1.pt')
)
GMNW_MATCHED_MLP_SCHEMA_VERSION_V1: Final[str] = "eiel_gmnw_matched_mlp_control_v1"
GMNW_SEL1_IMMEDIATE_MLP_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_gmnw_sel1_immediate_mlp_v1"
)
GMNW_MATCHED_MLP_PACKED_WIDTH_V1: Final[int] = 272_903
GMNW_MATCHED_MLP_NEURAL_SCRATCH_WIDTH_V1: Final[int] = 1_024
GMNW_MATCHED_MLP_AUDIT_TICKS_V1: Final[int] = 160
GMNW_MATCHED_MLP_AUDIT_WORDS_PER_TICK_V1: Final[int] = 2
GMNW_MATCHED_MLP_SCRATCH_WIDTH_V1: Final[int] = (
    GMNW_MATCHED_MLP_NEURAL_SCRATCH_WIDTH_V1
    + GMNW_MATCHED_MLP_AUDIT_TICKS_V1 * GMNW_MATCHED_MLP_AUDIT_WORDS_PER_TICK_V1
)
GMNW_VENV_CONSEQUENCE_PROFILE_TENSOR_INDEX_V1: Final[int] = 22
GMNW_CONTROL_POLICY_ARM_TENSOR_INDEX_V1: Final[int] = 23
GMNW_CONTROL_MLP_PACK_TENSOR_INDEX_V1: Final[int] = 24
GMNW_CONTROL_MLP_SCRATCH_TENSOR_INDEX_V1: Final[int] = 25
GMNW_POLICY_ARMS_V1: Final[tuple[str, ...]] = (
    "persistent_gru",
    "reset_gru",
    "matched_mlp",
    "mixed_selection",
)
GMNW_POLICY_ARM_CODES_V1: Final[dict[tuple[str, bool], int]] = {
    ("persistent_gru", False): 0,
    ("reset_gru", False): 1,
    ("matched_mlp", False): 2,
    ("mixed_selection", False): 3,
    ("persistent_gru", True): 4,
    ("matched_mlp", True): 5,
    ("mixed_selection", True): 6,
}
GMNW_VENV_CONSEQUENCE_PROFILE_CODES_V1: Final[dict[str, int]] = {
    "legacy": 0,
    "P1": 1,
    "P2": 2,
    "P3": 3,
}
GMNW_QUALIFICATION_TRACE_VERSION_V1: Final[int] = 1
GMNW_QUALIFICATION_TRACE_TASK_STATE_COLUMNS_V1: Final[tuple[str, ...]] = (
    "cue_identity",
    "cue_phase",
    "episode_index",
    "phase_tick",
    "delay_remaining",
    "decision_due",
    "correct_action",
    "selected_private_action",
    "correctness",
)
GMNW_QUALIFICATION_TRACE_TENSOR_NAMES_V1: Final[tuple[str, ...]] = (
    "gmnw_trace_active_count",
    "gmnw_trace_ordered_agent_ids",
    "gmnw_trace_ordered_slots",
    "gmnw_trace_hidden_before",
    "gmnw_trace_hidden_after",
    "gmnw_trace_task_state_before",
    "gmnw_trace_task_state_after",
)
GMNW_QUALIFICATION_INITIAL_STREAM_COUNT_V1: Final[int] = 64
GMNW_QUALIFICATION_EPISODES_PER_STREAM_V1: Final[int] = 48
GMNW_QUALIFICATION_DECISION_ROWS_V1: Final[int] = (
    GMNW_QUALIFICATION_INITIAL_STREAM_COUNT_V1
    * GMNW_QUALIFICATION_EPISODES_PER_STREAM_V1
)
GMNW_QUALIFICATION_DECISION_COLUMNS_V1: Final[tuple[str, ...]] = (
    "stream_id",
    "episode_index",
    "cue_identity",
    "selected_private_action",
    "correct_action",
    "correctness",
)
GMNW_QUALIFICATION_DECISION_TAPE_VERSION_V1: Final[int] = 1
GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1: Final[tuple[str, ...]] = (
    "gmnw_qualification_decision_present",
    "gmnw_qualification_decision_rows",
    "gmnw_qualification_decision_status",
)
GMNW_SELECTION_METRICS_VERSION_V1: Final[int] = 1
GMNW_SELECTION_POLICY_FAMILIES_V1: Final[tuple[str, ...]] = ("mlp", "gru")
GMNW_SELECTION_COUNT_METRIC_NAMES_V1: Final[tuple[str, ...]] = (
    "alive_pre_turnover",
    "alive_post_turnover",
    "decision_count",
    "correct_count",
    "task_death_count",
    "unique_selected_parent_pool_count",
    "committed_turnover_offspring_count",
)
GMNW_SELECTION_METRIC_NAMES_V1: Final[tuple[str, ...]] = (
    *GMNW_SELECTION_COUNT_METRIC_NAMES_V1,
    "recurrent_upkeep_charge_total",
)
GMNW_SELECTION_METRICS_STATUS_NAME_V1: Final[str] = "status"
GMNW_MINIMUM_RESIDENT_SEGMENT_GENERATIONS_V1: Final[int] = 3
GMNW_MAXIMUM_RESIDENT_SEGMENT_GENERATIONS_V1: Final[int] = 128
GMNW_ALLOCATION_GENERATION_CAPACITIES_V1: Final[tuple[int, ...]] = (
    4,
    8,
    12,
    32,
    100,
)
MLP_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "live_state.mlp_pack"
)
OBSERVATION_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "outputs.observations"
)
CACHE_WIDTH_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "config.cache_width"
)
PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V1: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_runtime_bookkeeping")
)
PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V1: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_full_tick_output_tensors")
)
PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V1: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_patch_stats_tensors")
)
TICKS_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "ticks_per_generation"
)


class CudaArchitectureBDirectResidentGmnwGruV1Error(RuntimeError):
    """The bounded integrated GMNW+GRU resident boundary is invalid."""


@dataclass(frozen=True, slots=True)
class GmnwConfigurationV1:
    cue_alphabet: int = 2
    cue_ticks: int = 1
    delay_ticks: int = 4
    consequence_ticks: int = 1
    distractors_enabled: bool = False
    cue_visible_at_decision: bool = False
    reset_hidden_each_tick: bool = False
    policy_arm: str = "persistent_gru"
    gru_founder_count: int | None = None
    task_seed: int = 7001
    grace_generation_count: int = 4
    severe_generation_count: int = 4
    grace_damage_fraction: float = 0.15
    severe_damage_fraction: float = 0.45
    lethal_damage_fraction: float = 1.0
    grace_reproduction_scale: float = 0.5
    venv_consequence_profile: str = "legacy"
    terminal_low_population_no_reinit: bool = False


@dataclass(frozen=True, slots=True)
class PreparedDirectResidentGmnwGruV1:
    patch: PreparedDirectResidentTrajectoryV1
    gru_tensors: tuple[torch.Tensor, ...]
    gmnw_tensors: tuple[torch.Tensor, ...]
    gru_config: Mapping[str, Any]
    gmnw_config: Mapping[str, Any]
    initial_projection: Any
    base_observation_size: int
    observation_size: int
    source_projections: tuple[Any, ...] = dataclass_field(default_factory=tuple)
    cell_ids: tuple[str, ...] = dataclass_field(default_factory=tuple)
    qualification_trace: Mapping[str, torch.Tensor] = dataclass_field(
        default_factory=dict
    )
    qualification_decision_tape: Mapping[str, torch.Tensor] = dataclass_field(
        default_factory=dict
    )
    selection_metrics: Mapping[str, torch.Tensor] = dataclass_field(
        default_factory=dict
    )
    allocation_memory_receipt: Mapping[str, Any] = dataclass_field(default_factory=dict)
    checkpoint_source_projections: tuple[Mapping[str, Any], ...] = ()


def _fail(message: str) -> CudaArchitectureBDirectResidentGmnwGruV1Error:
    return CudaArchitectureBDirectResidentGmnwGruV1Error(message)


def gmnw_allocation_generation_capacity_v1(logical_generation_count: int) -> int:
    """Return storage capacity for an admitted logical GMNW horizon."""

    if (
        type(logical_generation_count) is not int
        or not 4 <= logical_generation_count <= 100
    ):
        raise _fail("GMNW+GRU preparation requires generations in [4,100]")
    return next(
        count
        for count in GMNW_ALLOCATION_GENERATION_CAPACITIES_V1
        if count >= logical_generation_count
    )


def _allocation_memory_snapshot(device: torch.device, *, phase: str) -> dict[str, Any]:
    receipt: dict[str, Any] = {"phase": phase, "device": str(device)}
    if device.type != "cuda":
        receipt["cuda_memory_available"] = False
        return receipt
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    receipt.update(
        {
            "cuda_memory_available": True,
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
    )
    return receipt


def gmnw_gru_observation_contract_v1() -> dict[str, Any]:
    """Return the frozen 86-column Patch plus GMNW observation contract."""

    return {
        "schema_version": "eiel_gmnw_gru_observation_registry_v1",
        "column_count": len(GMNW_GRU_OBSERVATION_COLUMNS_V1),
        "columns": list(GMNW_GRU_OBSERVATION_COLUMNS_V1),
        "registry_sha256": GMNW_GRU_OBSERVATION_REGISTRY_SHA256_V1,
        "spatial_order": (
            "channel_major_then_dy_minus1_to_plus1_then_dx_minus1_to_plus1"
        ),
        "base_patch_column_count": 74,
        "gmnw_column_start": 74,
        "gmnw_column_count": GMNW_OBSERVATION_COLUMNS_V1,
        "gru_input": "all_86_columns_in_registry_order",
        "row_mask": "alive_slots_only",
        "disabled_fields": [
            "wdl_cue",
            "jungle_observations",
            "prediction_inputs",
            "modular_recurrent_metadata",
        ],
        "normalization": {
            "legacy_occupancy_or_food": "exact_-1_0_1",
            "patch_food_scent_waste_density": "safe_norm_clamped_0_1",
            "patch_cover_shelter": "finite_clamped_0_1",
            "patch_movement_cost": "min_max_clamped_0_1",
            "agent_energy": "energy_over_max_energy",
            "agent_health": "health_over_start_health",
            "gmnw_cues_phases_distractors": "exact_binary_0_1",
        },
    }


def _packed_tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(cpu.numpy().tobytes(order="C")).hexdigest()


def _stable_gru_layout_blocks(observation_size: int) -> list[dict[str, Any]]:
    hidden = GRU_HIDDEN_WIDTH_V3
    actions = GRU_ACTION_COUNT_V3
    definitions = (
        ("encoder_weight", (hidden, observation_size)),
        ("encoder_bias", (hidden,)),
        ("gru_input_weight", (3 * hidden, hidden)),
        ("gru_input_bias", (3 * hidden,)),
        ("gru_recurrent_weight", (3 * hidden, hidden)),
        ("gru_recurrent_bias", (3 * hidden,)),
        ("action_head_weight", (actions, hidden)),
        ("action_head_bias", (actions,)),
    )
    offset = 0
    blocks: list[dict[str, Any]] = []
    for name, shape in definitions:
        extent = 1
        for dimension in shape:
            extent *= dimension
        blocks.append(
            {
                "semantic_name": name,
                "offset": offset,
                "shape": list(shape),
                "extent": extent,
            }
        )
        offset += extent
    return blocks


def gmnw_witness_production_layout_crosswalk_v1(
    witness: torch.Tensor,
    production: torch.Tensor,
    *,
    base_observation_size: int = 74,
) -> dict[str, Any]:
    """Prove the frozen 12-column witness is losslessly embedded in 86 columns."""

    production_observation_size = base_observation_size + GMNW_OBSERVATION_COLUMNS_V1
    if witness.ndim != 1 or witness.numel() != stable_gru_packed_width_v3(
        GMNW_OBSERVATION_COLUMNS_V1
    ):
        raise _fail("witness layout crosswalk received an invalid witness pack")
    if production.ndim != 1 or production.numel() != stable_gru_packed_width_v3(
        production_observation_size
    ):
        raise _fail("witness layout crosswalk received an invalid production pack")
    witness_cpu = witness.detach().to(device="cpu", dtype=torch.float32).contiguous()
    production_cpu = (
        production.detach().to(device="cpu", dtype=torch.float32).contiguous()
    )
    witness_blocks = _stable_gru_layout_blocks(GMNW_OBSERVATION_COLUMNS_V1)
    production_blocks = _stable_gru_layout_blocks(production_observation_size)
    rows: list[dict[str, Any]] = []
    for witness_block, production_block in zip(
        witness_blocks, production_blocks, strict=True
    ):
        name = str(witness_block["semantic_name"])
        witness_values = witness_cpu[
            int(witness_block["offset"]) : int(witness_block["offset"])
            + int(witness_block["extent"])
        ].view(tuple(int(value) for value in witness_block["shape"]))
        production_values = production_cpu[
            int(production_block["offset"]) : int(production_block["offset"])
            + int(production_block["extent"])
        ].view(tuple(int(value) for value in production_block["shape"]))
        if name == "encoder_weight":
            copied = production_values[:, base_observation_size:]
            padding = production_values[:, :base_observation_size]
        else:
            copied = production_values
            padding = production_values.new_empty((0,))
        copied_exact = torch.equal(witness_values, copied)
        padding_zero = bool(torch.count_nonzero(padding).item() == 0)
        rows.append(
            {
                "semantic_name": name,
                "witness_offset": int(witness_block["offset"]),
                "witness_shape": list(witness_block["shape"]),
                "production_offset": int(production_block["offset"]),
                "production_shape": list(production_block["shape"]),
                "copied_values": int(copied.numel()),
                "padded_values": int(padding.numel()),
                "disabled_values": 0,
                "hash_before_projection": _packed_tensor_sha256(witness_values),
                "hash_after_projection": _packed_tensor_sha256(copied),
                "copied_exact": copied_exact,
                "padding_zero": padding_zero,
            }
        )
    empty_hash = hashlib.sha256(b"").hexdigest()
    for name in ("inactive_prediction_state", "inactive_modular_recurrent_state"):
        rows.append(
            {
                "semantic_name": name,
                "witness_offset": None,
                "witness_shape": [0],
                "production_offset": None,
                "production_shape": [0],
                "copied_values": 0,
                "padded_values": 0,
                "disabled_values": 0,
                "hash_before_projection": empty_hash,
                "hash_after_projection": empty_hash,
                "copied_exact": True,
                "padding_zero": True,
                "storage": "disabled_outside_stable_gru_pack",
            }
        )
    encoded_registry = json.dumps(
        GMNW_GRU_OBSERVATION_COLUMNS_V1,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return {
        "schema_version": "eiel_gmnw_witness_production_layout_crosswalk_v1",
        "ok": all(
            bool(row["copied_exact"]) and bool(row["padding_zero"]) for row in rows
        ),
        "witness_observation_size": GMNW_OBSERVATION_COLUMNS_V1,
        "production_observation_size": production_observation_size,
        "witness_packed_width": int(witness.numel()),
        "production_packed_width": int(production.numel()),
        "observation_registry_sha256": hashlib.sha256(encoded_registry).hexdigest(),
        "witness_packed_sha256": _packed_tensor_sha256(witness_cpu),
        "production_packed_sha256": _packed_tensor_sha256(production_cpu),
        "blocks": rows,
    }


def _validated_capability(module: Any) -> dict[str, Any]:
    entrypoint = getattr(module, GMNW_CAPABILITY_ENTRYPOINT_V1, None)
    if not callable(entrypoint):
        raise _fail(
            f"CUDA extension is missing callable {GMNW_CAPABILITY_ENTRYPOINT_V1}"
        )
    raw = entrypoint()
    if not isinstance(raw, Mapping):
        raise _fail("GMNW+GRU CUDA capability must be a mapping")
    capability = dict(raw)
    expected = {
        "abi_version": 2,
        "world": "GRU_MEMORY_NECESSITY_WORLD_V1",
        "policy_family": "stable_gru",
        "resident_private_task_state": True,
        "resident_hidden_state": True,
        "per_cell_task_configuration": True,
        "exact_consequence_lifecycle": True,
        "exact_reproduction_opportunity": True,
        "public_world_action_neutralization": True,
        "native_tick_ownership": True,
        "native_generation_ownership": True,
        "runtime_jit_required": False,
        "hidden_fallback": False,
        "production_qualified": False,
    }
    for name, expected_value in expected.items():
        actual = capability.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"GMNW+GRU capability {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return capability


def _validated_configuration(config: GmnwConfigurationV1) -> None:
    if not isinstance(config, GmnwConfigurationV1):
        raise _fail("config must be GmnwConfigurationV1")
    if config.cue_alphabet not in {2, 4}:
        raise _fail("cue_alphabet must be 2 or 4")
    if config.policy_arm not in GMNW_POLICY_ARMS_V1:
        raise _fail(f"policy_arm must be one of {GMNW_POLICY_ARMS_V1}")
    if type(config.cue_visible_at_decision) is not bool:
        raise _fail("cue_visible_at_decision must be an exact boolean")
    if (config.policy_arm, config.cue_visible_at_decision) not in (
        GMNW_POLICY_ARM_CODES_V1
    ):
        raise _fail("reset_gru does not support immediate cue visibility")
    if config.reset_hidden_each_tick is not (config.policy_arm == "reset_gru"):
        raise _fail("reset_hidden_each_tick must be true exactly for reset_gru")
    if type(config.terminal_low_population_no_reinit) is not bool:
        raise _fail("terminal_low_population_no_reinit must be an exact boolean")
    if config.venv_consequence_profile not in GMNW_VENV_CONSEQUENCE_PROFILE_CODES_V1:
        raise _fail(
            "venv_consequence_profile must be one of "
            f"{tuple(GMNW_VENV_CONSEQUENCE_PROFILE_CODES_V1)}"
        )
    if config.policy_arm == "mixed_selection":
        if (
            type(config.gru_founder_count) is not int
            or not 0 <= config.gru_founder_count <= 64
        ):
            raise _fail(
                "mixed_selection gru_founder_count must be an integer in [0,64]"
            )
    elif config.gru_founder_count is not None:
        raise _fail("gru_founder_count is valid only for mixed_selection")
    for name in (
        "cue_ticks",
        "delay_ticks",
        "consequence_ticks",
        "grace_generation_count",
        "severe_generation_count",
    ):
        if type(getattr(config, name)) is not int or getattr(config, name) < 1:
            raise _fail(f"{name} must be a positive exact integer")
    for name in (
        "grace_damage_fraction",
        "severe_damage_fraction",
        "lethal_damage_fraction",
        "grace_reproduction_scale",
    ):
        value = getattr(config, name)
        if type(value) is not float or not 0.0 <= value <= 1.0:
            raise _fail(f"{name} must be an exact float in [0,1]")


def _load_witness(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise _fail("GMNW witness payload must be a mapping")
    schema_version = payload.get("schema_version")
    if schema_version not in {
        GMNW_WITNESS_SCHEMA_VERSION_V1,
        GMNW_WITNESS_SCHEMA_VERSION_V2,
    }:
        raise _fail("GMNW witness schema_version mismatch")
    if schema_version == GMNW_WITNESS_SCHEMA_VERSION_V2 and (
        payload.get("continuous_episode_training") is not True
        or payload.get("parent_weights_sha256") != GMNW_WITNESS_V2_PARENT_SHA256
    ):
        raise _fail("GMNW continuous witness qualification metadata mismatch")
    expected = {
        "observation_width": GMNW_OBSERVATION_COLUMNS_V1,
        "hidden_width": GRU_HIDDEN_WIDTH_V3,
        "action_count": GRU_ACTION_COUNT_V3,
    }
    for name, expected_value in expected.items():
        if payload.get(name) != expected_value:
            raise _fail(f"GMNW witness {name} mismatch")
    packed = payload.get("packed")
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.float32:
        raise _fail("GMNW witness packed weights must be float32")
    expected_width = stable_gru_packed_width_v3(GMNW_OBSERVATION_COLUMNS_V1)
    if packed.ndim != 1 or packed.numel() != expected_width:
        raise _fail("GMNW witness packed width mismatch")
    return packed.contiguous()


def _load_matched_mlp(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise _fail("GMNW matched MLP payload must be a mapping")
    schema_version = payload.get("schema_version")
    if schema_version not in {
        GMNW_MATCHED_MLP_SCHEMA_VERSION_V1,
        GMNW_SEL1_IMMEDIATE_MLP_SCHEMA_VERSION_V1,
    }:
        raise _fail("GMNW matched MLP schema version mismatch")
    expected_architecture = {
        "observation_width": GMNW_OBSERVATION_COLUMNS_V1,
        "hidden_widths": [512, 512],
        "action_count": GRU_ACTION_COUNT_V3,
        "packed_tensor_order": [
            "network.0.weight",
            "network.0.bias",
            "network.2.weight",
            "network.2.bias",
            "network.4.weight",
            "network.4.bias",
        ],
    }
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping) or any(
        architecture.get(name) != expected_value
        for name, expected_value in expected_architecture.items()
    ):
        raise _fail("GMNW matched MLP architecture mismatch")
    expected_temporal_role = (
        "pre_outcome_frozen_control"
        if schema_version == GMNW_MATCHED_MLP_SCHEMA_VERSION_V1
        else "pre_selection_matrix_frozen_control"
    )
    if payload.get("temporal_role") != expected_temporal_role:
        raise _fail("GMNW matched MLP is not frozen before outcomes")
    if schema_version == GMNW_SEL1_IMMEDIATE_MLP_SCHEMA_VERSION_V1:
        qualification = payload.get("qualification")
        if (
            not isinstance(qualification, Mapping)
            or qualification.get("accepted") is not True
            or qualification.get("same_process_repeat_exact") is not True
            or qualification.get("minimum_immediate_expected_accuracy", 0.0) < 0.90
            or qualification.get("minimum_immediate_stochastic_accuracy", 0.0) < 0.90
            or qualification.get("maximum_delayed_expected_accuracy", 1.0) > 0.35
            or qualification.get("maximum_delayed_stochastic_accuracy", 1.0) > 0.35
        ):
            raise _fail("GMNW SEL1 immediate MLP qualification mismatch")
    packed = payload.get("packed")
    if (
        not isinstance(packed, torch.Tensor)
        or packed.dtype != torch.float32
        or packed.ndim != 1
        or packed.numel() != GMNW_MATCHED_MLP_PACKED_WIDTH_V1
        or not bool(torch.isfinite(packed).all().item())
    ):
        raise _fail("GMNW matched MLP packed weights are invalid")
    if payload.get("packed_width") != GMNW_MATCHED_MLP_PACKED_WIDTH_V1:
        raise _fail("GMNW matched MLP packed-width metadata mismatch")
    if payload.get("packed_sha256") != _packed_tensor_sha256(packed):
        raise _fail("GMNW matched MLP packed hash mismatch")
    return packed.contiguous()


def _expand_witness_pack(
    witness: torch.Tensor,
    *,
    base_observation_size: int,
    cells: int,
    capacity: int,
    alive_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    observation_size = base_observation_size + GMNW_OBSERVATION_COLUMNS_V1
    packed_width = stable_gru_packed_width_v3(observation_size)
    output = torch.zeros(
        (cells, capacity, packed_width), dtype=torch.float32, device=device
    )
    encoder_rows = witness[: GRU_HIDDEN_WIDTH_V3 * GMNW_OBSERVATION_COLUMNS_V1].view(
        GRU_HIDDEN_WIDTH_V3, GMNW_OBSERVATION_COLUMNS_V1
    )
    one = torch.zeros((packed_width,), dtype=torch.float32)
    encoder = one[: GRU_HIDDEN_WIDTH_V3 * observation_size].view(
        GRU_HIDDEN_WIDTH_V3, observation_size
    )
    encoder[:, base_observation_size:] = encoder_rows
    one[GRU_HIDDEN_WIDTH_V3 * observation_size :] = witness[
        GRU_HIDDEN_WIDTH_V3 * GMNW_OBSERVATION_COLUMNS_V1 :
    ]
    alive = alive_mask.to(device="cpu", dtype=torch.bool)
    output[alive.to(device=device)] = one.to(device=device)
    return output.contiguous()


def _shared_witness_pack(
    witness: torch.Tensor,
    *,
    base_observation_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the single immutable GMNW GRU policy row retained by all worlds."""

    dense = _expand_witness_pack(
        witness,
        base_observation_size=base_observation_size,
        cells=1,
        capacity=1,
        alive_mask=torch.ones((1, 1), dtype=torch.bool),
        device=device,
    )
    return dense[0, 0].contiguous()


def _materialize_frozen_gru_checkpoint_rows_cpu(
    prepared: PreparedDirectResidentGmnwGruV1,
) -> torch.Tensor:
    shared = prepared.gru_tensors[0]
    alive = prepared.patch.live_banks.get("alive_mask")
    policy_use_gru = prepared.patch.live_banks.get("policy_use_gru")
    if (
        prepared.gru_config.get("immutable_shared_policy") is not True
        or shared.dtype != torch.float32
        or shared.ndim != 1
        or not isinstance(alive, torch.Tensor)
        or not isinstance(policy_use_gru, torch.Tensor)
        or alive.shape != policy_use_gru.shape
        or alive.ndim != 2
    ):
        raise _fail("frozen shared GRU checkpoint authority is invalid")
    active = (alive & policy_use_gru).detach().to(device="cpu")
    shared_cpu = shared.detach().to(device="cpu")
    dense = torch.zeros(
        (*active.shape, shared_cpu.numel()), dtype=torch.float32, device="cpu"
    )
    dense[active] = shared_cpu
    return dense.contiguous()


def _expand_fixed_mlp_pack(
    original: torch.Tensor, *, base_observation_size: int
) -> torch.Tensor:
    hidden = GRU_HIDDEN_WIDTH_V3
    actions = GRU_ACTION_COUNT_V3
    observation_size = base_observation_size + GMNW_OBSERVATION_COLUMNS_V1
    expected_original = (
        hidden * base_observation_size
        + hidden
        + hidden * hidden
        + hidden
        + actions * hidden
        + actions
    )
    expected_expanded = (
        hidden * observation_size
        + hidden
        + hidden * hidden
        + hidden
        + actions * hidden
        + actions
    )
    if original.ndim not in {2, 3}:
        raise _fail("resident fixed-MLP compatibility pack has an invalid width")
    if int(original.shape[-1]) == expected_expanded:
        return original
    if int(original.shape[-1]) != expected_original:
        raise _fail("resident fixed-MLP compatibility pack has an invalid width")
    expanded = torch.zeros(
        (*original.shape[:-1], expected_expanded),
        dtype=original.dtype,
        device=original.device,
    )
    expanded[..., : hidden * observation_size].view(
        *original.shape[:-1], hidden, observation_size
    )[..., :base_observation_size] = original[
        ..., : hidden * base_observation_size
    ].view(*original.shape[:-1], hidden, base_observation_size)
    expanded[..., hidden * observation_size :] = original[
        ..., hidden * base_observation_size :
    ]
    return expanded.contiguous()


def _contract_fixed_mlp_pack(
    expanded: torch.Tensor, *, base_observation_size: int
) -> torch.Tensor:
    hidden = GRU_HIDDEN_WIDTH_V3
    actions = GRU_ACTION_COUNT_V3
    observation_size = base_observation_size + GMNW_OBSERVATION_COLUMNS_V1
    original_width = (
        hidden * base_observation_size
        + hidden
        + hidden * hidden
        + hidden
        + actions * hidden
        + actions
    )
    expanded_width = original_width + hidden * GMNW_OBSERVATION_COLUMNS_V1
    if expanded.ndim not in {2, 3} or int(expanded.shape[-1]) != expanded_width:
        raise _fail("resident fixed-MLP checkpoint pack has an invalid width")
    contracted = torch.zeros(
        (*expanded.shape[:-1], original_width),
        dtype=expanded.dtype,
        device=expanded.device,
    )
    contracted[..., : hidden * base_observation_size].view(
        *expanded.shape[:-1], hidden, base_observation_size
    ).copy_(
        expanded[..., : hidden * observation_size].view(
            *expanded.shape[:-1], hidden, observation_size
        )[..., :base_observation_size]
    )
    contracted[..., hidden * base_observation_size :].copy_(
        expanded[..., hidden * observation_size :]
    )
    return contracted.contiguous()


def _contract_private_gru_pack(
    expanded: torch.Tensor, *, base_observation_size: int
) -> torch.Tensor:
    observation_size = base_observation_size + GMNW_OBSERVATION_COLUMNS_V1
    expanded_width = stable_gru_packed_width_v3(observation_size)
    public_width = stable_gru_packed_width_v3(base_observation_size)
    hidden = GRU_HIDDEN_WIDTH_V3
    if expanded.ndim not in {2, 3}:
        raise _fail("resident private-GRU checkpoint pack has an invalid width")
    if int(expanded.shape[-1]) == public_width:
        return expanded.contiguous()
    if int(expanded.shape[-1]) != expanded_width:
        raise _fail("resident private-GRU checkpoint pack has an invalid width")
    contracted = torch.zeros(
        (*expanded.shape[:-1], public_width),
        dtype=expanded.dtype,
        device=expanded.device,
    )
    contracted[..., : hidden * base_observation_size].view(
        *expanded.shape[:-1], hidden, base_observation_size
    ).copy_(
        expanded[..., : hidden * observation_size].view(
            *expanded.shape[:-1], hidden, observation_size
        )[..., :base_observation_size]
    )
    contracted[..., hidden * base_observation_size :].copy_(
        expanded[..., hidden * observation_size :]
    )
    return contracted.contiguous()


def _canonicalize_exhausted_private_action_cache(
    tensors: dict[str, Any],
) -> None:
    """Project the one-action private cache into the public inactive form."""
    cache_token = tensors["action_cache_token"].clone()
    cache_cursor = tensors["action_cache_cursor"].clone()
    cache_next = tensors["action_cache_next"].clone()
    exhausted = cache_cursor >= 1
    cache_token[exhausted] = -1
    cache_cursor[exhausted] = -1
    cache_next[exhausted] = -1
    tensors["action_cache_token"] = cache_token
    tensors["action_cache_cursor"] = cache_cursor
    tensors["action_cache_next"] = cache_next


def _allocate_gmnw_tensors(
    *,
    cells: int,
    capacity: int,
    generations: int,
    configs: Sequence[GmnwConfigurationV1] | None = None,
    config: GmnwConfigurationV1 | None = None,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if configs is not None and config is not None:
        raise _fail("provide either configs or config, not both")
    if configs is None:
        if config is None:
            raise _fail("GMNW configuration is required")
        configs = tuple(config for _ in range(cells))
    if len(configs) != cells:
        raise _fail("GMNW cell configuration count must match physical cells")
    for config in configs:
        _validated_configuration(config)
    rows = (cells, capacity)
    return (
        torch.full(rows, -1, dtype=torch.int64, device=device),
        torch.full(rows, -1, dtype=torch.int64, device=device),
        torch.full(rows, -1, dtype=torch.int8, device=device),
        torch.zeros(rows, dtype=torch.int8, device=device),
        torch.zeros(rows, dtype=torch.int32, device=device),
        torch.zeros(rows, dtype=torch.int32, device=device),
        torch.zeros(rows, dtype=torch.bool, device=device),
        torch.full(rows, -1, dtype=torch.int8, device=device),
        torch.full(rows, -1, dtype=torch.int8, device=device),
        torch.full(rows, -1, dtype=torch.int8, device=device),
        torch.zeros(rows, dtype=torch.int8, device=device),
        torch.zeros(rows, dtype=torch.int64, device=device),
        torch.zeros(rows, dtype=torch.int64, device=device),
        torch.ones(rows, dtype=torch.float32, device=device),
        torch.zeros(
            (cells, generations, GMNW_COUNTER_COLUMNS_V1),
            dtype=torch.int64,
            device=device,
        ),
        torch.zeros((cells, generations), dtype=torch.float64, device=device),
        torch.zeros((cells,), dtype=torch.int32, device=device),
        torch.tensor(
            [config.cue_alphabet for config in configs],
            dtype=torch.int32,
            device=device,
        ),
        torch.tensor(
            [config.delay_ticks for config in configs],
            dtype=torch.int32,
            device=device,
        ),
        torch.tensor(
            [config.distractors_enabled for config in configs],
            dtype=torch.bool,
            device=device,
        ),
        torch.tensor(
            [config.reset_hidden_each_tick for config in configs],
            dtype=torch.bool,
            device=device,
        ),
        torch.tensor(
            [config.task_seed for config in configs],
            dtype=torch.int64,
            device=device,
        ),
        torch.tensor(
            [
                GMNW_VENV_CONSEQUENCE_PROFILE_CODES_V1[config.venv_consequence_profile]
                for config in configs
            ],
            dtype=torch.int32,
            device=device,
        ).contiguous(),
    )


def _allocate_gmnw_selection_metrics_v1(
    *,
    cells: int,
    generations: int,
    device: torch.device,
    initial: Mapping[str, torch.Tensor] | None = None,
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    expected_shape = (cells, generations, len(GMNW_SELECTION_POLICY_FAMILIES_V1))
    counts = torch.zeros(
        (*expected_shape, len(GMNW_SELECTION_COUNT_METRIC_NAMES_V1)),
        dtype=torch.int64,
        device=device,
    )
    upkeep = torch.zeros(expected_shape, dtype=torch.float64, device=device)
    status = torch.zeros((cells,), dtype=torch.int32, device=device)
    if initial is not None:
        if not isinstance(initial, Mapping):
            raise _fail("GMNW selection metrics continuation must be a mapping")
        expected_names = {
            *GMNW_SELECTION_METRIC_NAMES_V1,
            GMNW_SELECTION_METRICS_STATUS_NAME_V1,
        }
        if set(initial) != expected_names:
            raise _fail("GMNW selection metrics continuation fields mismatch")
        for column, name in enumerate(GMNW_SELECTION_COUNT_METRIC_NAMES_V1):
            tensor = initial[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.int64
                or tuple(tensor.shape) != expected_shape
            ):
                raise _fail(
                    f"GMNW selection metric {name} must be int64[cells,generations,2]"
                )
            counts[..., column].copy_(tensor.to(device=device))
        initial_upkeep = initial["recurrent_upkeep_charge_total"]
        if (
            not isinstance(initial_upkeep, torch.Tensor)
            or initial_upkeep.dtype != torch.float64
            or tuple(initial_upkeep.shape) != expected_shape
        ):
            raise _fail(
                "GMNW selection metric recurrent_upkeep_charge_total must be "
                "float64[cells,generations,2]"
            )
        upkeep.copy_(initial_upkeep.to(device=device))
        initial_status = initial[GMNW_SELECTION_METRICS_STATUS_NAME_V1]
        if (
            not isinstance(initial_status, torch.Tensor)
            or initial_status.dtype != torch.int32
            or tuple(initial_status.shape) != (cells,)
        ):
            raise _fail("GMNW selection metrics status must be int32[cells]")
        if bool(torch.count_nonzero(initial_status).item()):
            raise _fail("GMNW selection metrics continuation status is nonzero")
        status.copy_(initial_status.to(device=device))
    named = {
        name: counts[..., column]
        for column, name in enumerate(GMNW_SELECTION_COUNT_METRIC_NAMES_V1)
    }
    named["recurrent_upkeep_charge_total"] = upkeep
    named[GMNW_SELECTION_METRICS_STATUS_NAME_V1] = status
    return (counts, upkeep, status), named


def _neutralize_gmnw_policy_modifiers(live_state: dict[str, Any]) -> None:
    """Freeze active policy modifiers without changing inherited genomes."""
    live_state["action_temperature"].fill_(0.05)
    live_state["attack_bias"].zero_()
    live_state["drive_logit_gain"].zero_()


def _assign_mixed_selection_founders(
    live_state: dict[str, Any],
    configs: Sequence[GmnwConfigurationV1],
) -> None:
    """Assign spatially stratified founder families without changing ecology."""

    for cell, config in enumerate(configs):
        if config.policy_arm != "mixed_selection":
            continue
        alive = live_state["alive_mask"][cell]
        gru_upkeep_cost = live_state["gru_upkeep_cost"][cell]
        original_gru_upkeep_cost = gru_upkeep_cost.clone()
        alive_slots = torch.nonzero(alive, as_tuple=False).flatten()
        founder_count = int(config.gru_founder_count or 0)
        if int(alive_slots.numel()) != 64 or founder_count > int(alive_slots.numel()):
            raise _fail("SEL1 mixed founders require exactly 64 live founders")
        live_state["policy_use_gru"][cell].masked_fill_(alive, False)
        live_state["policy_mem_dim"][cell].masked_fill_(alive, 0)
        live_state["hmem"][cell].zero_()
        gru_upkeep_cost.masked_fill_(alive, 0.0)
        if founder_count == 0:
            continue
        pos_x = live_state["pos_x"][cell, alive_slots].to(torch.int64)
        spatial_stride = int(pos_x.max().item()) + 2
        spatial = (
            live_state["pos_y"][cell, alive_slots].to(torch.int64) * spatial_stride
            + pos_x
        )
        agent_ids = live_state["slot_to_agent_id"][cell, alive_slots]
        key_stride = int(agent_ids.max().item()) + 2
        ordered = alive_slots[torch.argsort(spatial * key_stride + agent_ids)]
        ranks = torch.div(
            (2 * torch.arange(founder_count, device=ordered.device) + 1)
            * int(ordered.numel()),
            2 * founder_count,
            rounding_mode="floor",
        )
        rotation = int(config.task_seed % int(ordered.numel()))
        selected = ordered[(ranks + rotation) % int(ordered.numel())]
        live_state["policy_use_gru"][cell, selected] = True
        live_state["policy_mem_dim"][cell, selected] = GRU_HIDDEN_WIDTH_V3
        gru_upkeep_cost[selected] = original_gru_upkeep_cost[selected]


def _resolved_cell_configs(
    config: GmnwConfigurationV1,
    cell_configs: Sequence[GmnwConfigurationV1] | None,
    *,
    batch_cells: int,
) -> tuple[GmnwConfigurationV1, ...]:
    if type(batch_cells) is not int or batch_cells < 1 or batch_cells > 72:
        raise _fail("GMNW batch_cells must be an exact integer in [1,72]")
    resolved = (
        tuple(config for _ in range(batch_cells))
        if cell_configs is None
        else tuple(cell_configs)
    )
    if len(resolved) != batch_cells:
        raise _fail("GMNW cell_configs length must equal batch_cells")
    for item in resolved:
        _validated_configuration(item)
    shared_fields = (
        "cue_ticks",
        "consequence_ticks",
        "grace_generation_count",
        "severe_generation_count",
        "grace_damage_fraction",
        "severe_damage_fraction",
        "lethal_damage_fraction",
        "grace_reproduction_scale",
        "terminal_low_population_no_reinit",
    )
    first = resolved[0]
    for field in shared_fields:
        expected = getattr(first, field)
        if any(getattr(item, field) != expected for item in resolved[1:]):
            raise _fail(f"GMNW physical cells must share scalar field {field}")
    return resolved


def build_prepared_direct_resident_gmnw_gru_v1(
    runner: Any | None = None,
    *,
    qualified_cuda_module: Any,
    config: GmnwConfigurationV1 = GmnwConfigurationV1(),
    cell_configs: Sequence[GmnwConfigurationV1] | None = None,
    batch_cells: int = 1,
    control_matrix_enabled: bool = False,
    logical_generation_count: int = 4,
    agent_id_allocation_generation_count: int | None = None,
    initial_generation: int = 0,
    gmnw_generation_base: int | None = None,
    gmnw_generation_rows: int | None = None,
    source_checkpoint_projection: Mapping[str, Any] | None = None,
    per_cell_requests: Sequence[DirectResidentSourceAllocationRequestV1] | None = None,
    source_checkpoint_projections: Sequence[Mapping[str, Any]] | None = None,
    cell_ids: Sequence[str] | None = None,
    witness_path: Path = DEFAULT_WITNESS_PATH_V1,
    matched_mlp_path: Path = DEFAULT_MATCHED_MLP_PATH_V1,
    cuda_device: str = "cuda",
    qualification_output_ticks: int = 1,
    qualification_trace_enabled: bool = False,
    qualification_decision_tape_enabled: bool = False,
    selection_metrics_enabled: bool = False,
    selection_metrics_state: Mapping[str, torch.Tensor] | None = None,
) -> PreparedDirectResidentGmnwGruV1:
    """Overlay the frozen GMNW task and witness on the exact Patch GRU owner."""

    if per_cell_requests is None:
        if runner is None:
            raise _fail("GMNW preparation requires runner or per_cell_requests")
        if source_checkpoint_projections is not None:
            raise _fail("B1 runner route uses source_checkpoint_projection")
        if type(batch_cells) is not int or batch_cells != 1:
            raise _fail(
                "runner convenience route is B1; use per_cell_requests for multicell"
            )
        resolved_batch_cells = 1
    else:
        if runner is not None or source_checkpoint_projection is not None:
            raise _fail(
                "multicell request route cannot also supply runner/B1 checkpoint"
            )
        if isinstance(per_cell_requests, (str, bytes, bytearray)) or not isinstance(
            per_cell_requests, Sequence
        ):
            raise _fail("per_cell_requests must be a sequence")
        resolved_batch_cells = len(per_cell_requests)
        if not 1 <= resolved_batch_cells <= 72:
            raise _fail("per_cell_requests must contain B1..B72 public requests")

    configs = _resolved_cell_configs(
        config,
        cell_configs,
        batch_cells=resolved_batch_cells,
    )
    config = configs[0]
    if type(control_matrix_enabled) is not bool:
        raise _fail("control_matrix_enabled must be an exact boolean")
    if any(
        item.policy_arm != "persistent_gru" or item.cue_visible_at_decision
        for item in configs
    ) or any(item.venv_consequence_profile != "legacy" for item in configs):
        control_matrix_enabled = True
    if type(qualification_trace_enabled) is not bool:
        raise _fail("qualification_trace_enabled must be an exact boolean")
    if type(qualification_decision_tape_enabled) is not bool:
        raise _fail("qualification_decision_tape_enabled must be an exact boolean")
    if type(selection_metrics_enabled) is not bool:
        raise _fail("selection_metrics_enabled must be an exact boolean")
    if selection_metrics_state is not None:
        selection_metrics_enabled = True
    capability = _validated_capability(qualified_cuda_module)
    if (
        qualification_trace_enabled
        and capability.get("qualification_trace_version")
        != GMNW_QUALIFICATION_TRACE_VERSION_V1
    ):
        raise _fail("CUDA extension does not expose GMNW qualification trace v1")
    if (
        qualification_decision_tape_enabled
        and capability.get("qualification_decision_tape_version")
        != GMNW_QUALIFICATION_DECISION_TAPE_VERSION_V1
    ):
        raise _fail(
            "CUDA extension does not expose GMNW qualification decision tape v1"
        )
    if (
        selection_metrics_enabled
        and capability.get("selection_metrics_version")
        != GMNW_SELECTION_METRICS_VERSION_V1
    ):
        raise _fail("CUDA extension does not expose GMNW selection metrics v1")
    allocation_generation_count = gmnw_allocation_generation_capacity_v1(
        logical_generation_count
    )
    if gmnw_generation_base is None:
        gmnw_generation_base = initial_generation
    if (
        type(gmnw_generation_base) is not int
        or gmnw_generation_base < 0
        or gmnw_generation_base > initial_generation
    ):
        raise _fail(
            "gmnw_generation_base must be an integer in [0, initial_generation]"
        )
    maximum_gmnw_rows = (
        initial_generation + logical_generation_count - gmnw_generation_base
    )
    if gmnw_generation_rows is None:
        gmnw_generation_rows = maximum_gmnw_rows
    minimum_gmnw_rows = initial_generation + 1 - gmnw_generation_base
    if (
        type(gmnw_generation_rows) is not int
        or not minimum_gmnw_rows
        <= gmnw_generation_rows
        <= maximum_gmnw_rows
    ):
        raise _fail(
            "gmnw_generation_rows must cover the next absolute generation "
            "within the allocated logical horizon"
        )
    witness = _load_witness(witness_path)
    shared_gru_pack_cpu = _shared_witness_pack(
        witness,
        base_observation_size=len(GMNW_GRU_OBSERVATION_COLUMNS_V1)
        - GMNW_OBSERVATION_COLUMNS_V1,
        device=torch.device("cpu"),
    )
    base: PreparedDirectResidentGruV3 = build_prepared_direct_resident_gru_v3(
        runner,
        qualified_cuda_module=qualified_cuda_module,
        logical_generation_count=allocation_generation_count,
        agent_id_allocation_generation_count=agent_id_allocation_generation_count,
        initial_generation=initial_generation,
        source_checkpoint_projection=source_checkpoint_projection,
        per_cell_requests=per_cell_requests,
        source_checkpoint_projections=source_checkpoint_projections,
        cell_ids=cell_ids,
        cuda_device=cuda_device,
        batch_cells=resolved_batch_cells,
        observation_extra_columns=GMNW_OBSERVATION_COLUMNS_V1,
        output_ticks=qualification_output_ticks,
        immutable_shared_gru_pack=shared_gru_pack_cpu,
    )
    source = base.patch.source
    initial_projection = base.initial_projection
    source_projections = base.source_projections
    prepared_cell_ids = base.cell_ids
    device = source.live_state["energy"].device
    allocation_phases = [
        _allocation_memory_snapshot(device, phase="dense_base_allocated")
    ]
    base_metadata_value = (
        base.initial_projection.get("metadata")
        if isinstance(base.initial_projection, Mapping)
        else getattr(base.initial_projection, "metadata", None)
    )
    if not isinstance(base_metadata_value, Mapping):
        raise _fail("GMNW source projection is missing metadata")
    base_metadata = dict(base_metadata_value)
    if base_metadata.get("policy_weight_mutation_enable") is not False:
        raise _fail(
            "frozen shared GMNW GRU policy requires policy mutation to be disabled"
        )
    cells = int(source.live_state["energy"].size(0))
    capacity = int(source.live_state["energy"].size(1))
    observation_size = base.observation_size
    base_observation_size = observation_size - GMNW_OBSERVATION_COLUMNS_V1
    gru_pack = source.live_state["gru_pack"]
    live_state = dict(source.live_state)
    live_state["mlp_pack"] = _expand_fixed_mlp_pack(
        live_state["mlp_pack"], base_observation_size=base_observation_size
    )
    live_state["gru_pack"] = gru_pack
    # GMNW consumes the policy head privately and publishes REST to Patch.
    # The kernel reapplies these active values before each GMNW policy call so
    # newborns retain the contract without changing ordinary genome evolution.
    _neutralize_gmnw_policy_modifiers(live_state)
    live_state["policy_mem_dim"].masked_fill_(
        live_state["alive_mask"], GRU_HIDDEN_WIDTH_V3
    )
    live_state["policy_use_gru"].masked_fill_(live_state["alive_mask"], True)
    for cell, item in enumerate(configs):
        if item.policy_arm == "matched_mlp":
            alive = live_state["alive_mask"][cell]
            live_state["policy_mem_dim"][cell].masked_fill_(alive, 0)
            live_state["policy_use_gru"][cell].masked_fill_(alive, False)
            live_state["hmem"][cell].zero_()
            live_state["gru_upkeep_cost"][cell].masked_fill_(alive, 0.0)
    _assign_mixed_selection_founders(live_state, configs)
    if initial_generation == 0:
        live_state["hmem"].zero_()
    birth_state_tensors = list(source.ordinary_birth_state_tensors)
    if len(birth_state_tensors) < 89:
        raise _fail("resident birth state does not expose the policy tape contract")
    birth_state_tensors[83] = live_state["mlp_pack"]
    for index in (86, 87, 88):
        birth_state_tensors[index] = _expand_fixed_mlp_pack(
            birth_state_tensors[index],
            base_observation_size=base_observation_size,
        )
    turnover_state_tensors = list(source.turnover_state_tensors)
    turnover_shadow_tensors = list(source.turnover_shadow_tensors)
    if len(turnover_state_tensors) <= 86 or len(turnover_shadow_tensors) <= 86:
        raise _fail("resident turnover state does not expose its MLP authority")
    turnover_state_tensors[86] = live_state["mlp_pack"]
    turnover_shadow_tensors[86] = torch.empty(
        (
            cells,
            int(source.configuration.population_size),
            int(live_state["mlp_pack"].size(2)),
        ),
        dtype=live_state["mlp_pack"].dtype,
        device=device,
    )
    live_banks = dict(source.live_banks)
    live_banks.update(live_state)
    live_banks["birth_state_tensors"] = birth_state_tensors
    live_banks["turnover_state_tensors"] = turnover_state_tensors
    live_banks["turnover_shadow_tensors"] = turnover_shadow_tensors
    if qualification_output_ticks not in {1, 80, 160}:
        raise _fail("GMNW qualification_output_ticks must be 1, 80, or 160")
    if qualification_trace_enabled and qualification_output_ticks not in {80, 160}:
        raise _fail("GMNW qualification tracing requires 80 or 160 output ticks")
    configuration = replace(source.configuration, cache_width=1)
    outputs = dict(source.outputs)
    if tuple(outputs["observations"].shape) != (
        cells,
        qualification_output_ticks,
        capacity,
        observation_size,
    ):
        raise _fail("direct GMNW output allocation did not retain the final layout")
    ordinary_birth_output_tensors = [
        outputs[name] for name in ORDINARY_BIRTH_OUTPUT_FIELDS
    ]
    event_state_tensors = [
        outputs["event_status_by_tick"]
        if tensor is source.outputs["event_status_by_tick"]
        else tensor
        for tensor in source.event_state_tensors
    ]
    full_tick_output_names = (
        "chemistry_order_slots",
        "runner_shuffle_draws",
        "runner_shuffle_status",
        "drive_noise_draws",
        "drive_noise_status",
        "chemistry_draws",
        "chemistry_status",
        "active_mechanics_status",
        "active_mechanics_counts",
        "residual_death_agent_ids",
        "residual_death_slots",
        "residual_death_reason_codes",
        "residual_death_count",
        "residual_release_count",
        "residual_release_status",
        "policy_pcg64_state_after_action",
        "policy_pcg64_buffer_after_action",
        "food_spawn_stats",
        "generation_summary_scalar",
    )
    full_tick_output_tensors = [outputs[name] for name in full_tick_output_names]
    updated_source = replace(
        source,
        configuration=configuration,
        live_state=live_state,
        outputs=outputs,
        full_tick_output_tensors=full_tick_output_tensors,
        ordinary_birth_output_tensors=ordinary_birth_output_tensors,
        event_state_tensors=event_state_tensors,
        ordinary_birth_state_tensors=birth_state_tensors,
        turnover_state_tensors=turnover_state_tensors,
        turnover_shadow_tensors=turnover_shadow_tensors,
        live_banks=live_banks,
    )
    patch = prepare_direct_resident_trajectory_v1(updated_source)
    if patch.arguments[MLP_ARGUMENT_INDEX_V1] is not live_state["mlp_pack"]:
        raise _fail("GMNW prepared ABI did not retain the expanded MLP pack")
    if patch.arguments[OBSERVATION_ARGUMENT_INDEX_V1] is not outputs["observations"]:
        raise _fail("GMNW prepared ABI did not retain the expanded observations")
    if patch.arguments[CACHE_WIDTH_ARGUMENT_INDEX_V1] != 1:
        raise _fail("GMNW prepared ABI did not retain cache_width=1")
    # The final source owns every tensor still needed below. Release the dense
    # pre-overlay source before allocating GRU/GMNW sidecars so preparation does
    # not retain both complete layouts at the allocation peak.
    del base, source
    gru_tensors = (
        gru_pack,
        torch.empty(
            (cells, capacity, GRU_SCRATCH_WIDTH_V3),
            dtype=torch.float32,
            device=device,
        ),
        torch.zeros((cells,), dtype=torch.int32, device=device),
    )
    gmnw_tensors = _allocate_gmnw_tensors(
        cells=cells,
        capacity=capacity,
        generations=gmnw_generation_rows,
        configs=configs,
        device=device,
    )
    if control_matrix_enabled:
        policy_arm = torch.tensor(
            [
                GMNW_POLICY_ARM_CODES_V1[
                    (item.policy_arm, item.cue_visible_at_decision)
                ]
                for item in configs
            ],
            dtype=torch.int8,
            device=device,
        )
        matched_mlp = _load_matched_mlp(matched_mlp_path).to(device=device)
        gmnw_tensors = (
            *gmnw_tensors,
            policy_arm,
            matched_mlp,
            torch.empty(
                (cells, GMNW_MATCHED_MLP_SCRATCH_WIDTH_V1),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros((cells,), dtype=torch.int32, device=device),
        )
    qualification_trace: dict[str, torch.Tensor] = {}
    if qualification_trace_enabled:
        trace_allocations = _allocate_gmnw_qualification_trace_outputs(
            capacity=capacity,
            hidden_width=GRU_HIDDEN_WIDTH_V3,
            task_state_width=len(GMNW_QUALIFICATION_TRACE_TASK_STATE_COLUMNS_V1),
            device=device,
            output_ticks=qualification_output_ticks,
            batch_cells=cells,
        )
        qualification_trace = {
            "gmnw_trace_active_count": outputs["active_count"],
            "gmnw_trace_ordered_agent_ids": outputs["ordered_agent_ids"],
            "gmnw_trace_ordered_slots": outputs["ordered_slots"],
            **trace_allocations,
        }
        expected_trace_shapes = {
            "gmnw_trace_active_count": (cells, qualification_output_ticks),
            "gmnw_trace_ordered_agent_ids": (
                cells,
                qualification_output_ticks,
                capacity,
            ),
            "gmnw_trace_ordered_slots": (
                cells,
                qualification_output_ticks,
                capacity,
            ),
            "gmnw_trace_hidden_before": (
                cells,
                qualification_output_ticks,
                capacity,
                GRU_HIDDEN_WIDTH_V3,
            ),
            "gmnw_trace_hidden_after": (
                cells,
                qualification_output_ticks,
                capacity,
                GRU_HIDDEN_WIDTH_V3,
            ),
            "gmnw_trace_task_state_before": (
                cells,
                qualification_output_ticks,
                capacity,
                len(GMNW_QUALIFICATION_TRACE_TASK_STATE_COLUMNS_V1),
            ),
            "gmnw_trace_task_state_after": (
                cells,
                qualification_output_ticks,
                capacity,
                len(GMNW_QUALIFICATION_TRACE_TASK_STATE_COLUMNS_V1),
            ),
        }
        for name in GMNW_QUALIFICATION_TRACE_TENSOR_NAMES_V1:
            tensor = qualification_trace.get(name)
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != (expected_trace_shapes[name])
            ):
                raise _fail(f"GMNW qualification trace tensor {name} shape mismatch")
        gmnw_tensors = (
            *gmnw_tensors,
            *(
                qualification_trace[name]
                for name in GMNW_QUALIFICATION_TRACE_TENSOR_NAMES_V1
            ),
        )
    qualification_decision_tape: dict[str, torch.Tensor] = {}
    if qualification_decision_tape_enabled:
        qualification_decision_tape = {
            "gmnw_qualification_decision_present": torch.zeros(
                (cells, GMNW_QUALIFICATION_DECISION_ROWS_V1),
                dtype=torch.int8,
                device=device,
            ),
            "gmnw_qualification_decision_rows": torch.full(
                (
                    cells,
                    GMNW_QUALIFICATION_DECISION_ROWS_V1,
                    len(GMNW_QUALIFICATION_DECISION_COLUMNS_V1),
                ),
                -1,
                dtype=torch.int64,
                device=device,
            ),
            "gmnw_qualification_decision_status": torch.zeros(
                (cells,), dtype=torch.int32, device=device
            ),
        }
        gmnw_tensors = (
            *gmnw_tensors,
            *(
                qualification_decision_tape[name]
                for name in GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1
            ),
        )
    selection_metrics: dict[str, torch.Tensor] = {}
    if selection_metrics_enabled:
        selection_tensors, selection_metrics = _allocate_gmnw_selection_metrics_v1(
            cells=cells,
            generations=gmnw_generation_rows,
            device=device,
            initial=selection_metrics_state,
        )
        gmnw_tensors = (*gmnw_tensors, *selection_tensors)
    allocation_phases.append(
        _allocation_memory_snapshot(device, phase="shared_layout_retained")
    )
    return PreparedDirectResidentGmnwGruV1(
        patch=patch,
        gru_tensors=gru_tensors,
        gmnw_tensors=gmnw_tensors,
        gru_config={
            "upkeep_cost_base": float(base_metadata.get("gru_cost_base", 0.0)),
            "upkeep_cost_per_dim": float(base_metadata.get("gru_cost_per_dim", 0.0)),
            "immutable_shared_policy": True,
        },
        gmnw_config={
            "enabled": True,
            "generation_base": gmnw_generation_base,
            "generation_rows_per_cell": gmnw_generation_rows,
            "cue_ticks": config.cue_ticks,
            "consequence_ticks": config.consequence_ticks,
            "grace_generation_count": config.grace_generation_count,
            "severe_generation_count": config.severe_generation_count,
            "grace_damage_fraction": config.grace_damage_fraction,
            "severe_damage_fraction": config.severe_damage_fraction,
            "lethal_damage_fraction": config.lethal_damage_fraction,
            "grace_reproduction_scale": config.grace_reproduction_scale,
            "terminal_low_population_no_reinit": (
                config.terminal_low_population_no_reinit
            ),
            "control_matrix_enabled": control_matrix_enabled,
            "policy_arm_codes": tuple(
                GMNW_POLICY_ARM_CODES_V1[
                    (item.policy_arm, item.cue_visible_at_decision)
                ]
                for item in configs
            ),
            "cue_visible_at_decision": tuple(
                item.cue_visible_at_decision for item in configs
            ),
            "gru_founder_count": tuple(item.gru_founder_count for item in configs),
            "venv_consequence_profile_codes": tuple(
                GMNW_VENV_CONSEQUENCE_PROFILE_CODES_V1[item.venv_consequence_profile]
                for item in configs
            ),
            "qualification_trace_enabled": qualification_trace_enabled,
            "qualification_trace_output_ticks": (
                qualification_output_ticks if qualification_trace_enabled else 0
            ),
            "qualification_decision_tape_enabled": (
                qualification_decision_tape_enabled
            ),
            "selection_metrics_enabled": selection_metrics_enabled,
        },
        initial_projection=initial_projection,
        base_observation_size=base_observation_size,
        observation_size=observation_size,
        source_projections=source_projections,
        cell_ids=prepared_cell_ids,
        qualification_trace=qualification_trace,
        qualification_decision_tape=qualification_decision_tape,
        selection_metrics=selection_metrics,
        allocation_memory_receipt={
            "schema_version": "eiel_gmnw_allocation_memory_receipt_v1",
            "policy_storage": "immutable_shared_gru_v1",
            "shared_policy_count": 1,
            "shared_policy_bytes": int(gru_pack.numel() * gru_pack.element_size()),
            "phases": allocation_phases,
        },
    )


def _validate_result(result: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(result)
    expected = {
        "gru_native_owner": True,
        "gru_native_controller_abi_version": 3,
        "gmnw_native_owner": True,
        "gmnw_native_controller_abi_version": 2,
        "gru_python_tick_or_generation_ownership": False,
        "gmnw_python_tick_or_generation_ownership": False,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    for name, expected_value in expected.items():
        actual = out.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"GMNW+GRU result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return out


def _require_internal_status_clean(
    prepared: PreparedDirectResidentGmnwGruV1,
) -> None:
    failures: dict[str, Any] = {}
    for name, tensor in prepared.patch.source.outputs.items():
        if not isinstance(tensor, torch.Tensor) or "status" not in name:
            continue
        cpu = tensor.detach().cpu()
        nonzero = torch.nonzero(cpu, as_tuple=False)
        if nonzero.numel():
            index = tuple(int(value) for value in nonzero[0].tolist())
            failures[name] = {
                "first_index": list(index),
                "first_value": int(cpu[index].item()),
            }
    if failures:
        first_failure = prepared.patch.source.outputs["first_failure_records"]
        raise _fail(
            "GMNW+GRU resident internal status is nonzero: "
            f"failures={failures} "
            f"first_failure={first_failure.detach().cpu().tolist()}"
        )


def _attach_gmnw_selection_metrics_v1(
    result: dict[str, Any],
    prepared: PreparedDirectResidentGmnwGruV1,
) -> None:
    if not prepared.selection_metrics:
        return
    version = result.get("gmnw_selection_metrics_version")
    if type(version) is not int or version != GMNW_SELECTION_METRICS_VERSION_V1:
        raise _fail("GMNW selection metrics result version mismatch")
    status = prepared.selection_metrics.get(GMNW_SELECTION_METRICS_STATUS_NAME_V1)
    if (
        not isinstance(status, torch.Tensor)
        or status.dtype != torch.int32
        or bool(torch.count_nonzero(status).item())
    ):
        raise _fail("GMNW selection metrics device status is nonzero or invalid")
    result["gmnw_selection_metrics"] = {
        name: prepared.selection_metrics[name]
        for name in GMNW_SELECTION_METRIC_NAMES_V1
    }
    result["gmnw_selection_metric_families"] = GMNW_SELECTION_POLICY_FAMILIES_V1


def execute_direct_resident_gmnw_gru_physical_v1(
    module: Any,
    prepared: PreparedDirectResidentGmnwGruV1,
    *,
    generation_count: int,
    ticks_per_generation: int = PHYSICAL_WINDOW_TICKS_V1 // 2,
) -> dict[str, Any]:
    if generation_count not in {1, 2}:
        raise _fail("physical GMNW+GRU qualification requires generation_count 1 or 2")
    if not 1 <= ticks_per_generation <= PHYSICAL_WINDOW_TICKS_V1 // 2:
        raise _fail("physical GMNW+GRU ticks must be in [1,80]")
    entrypoint = getattr(module, GMNW_PHYSICAL_ENTRYPOINT_V1, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {GMNW_PHYSICAL_ENTRYPOINT_V1}")
    arguments = list(prepared.patch.arguments)
    physical_ticks = generation_count * ticks_per_generation
    arguments[PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V1] = arguments[
        PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V1
    ][:, :physical_ticks].contiguous()
    full_tick_outputs = list(arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V1])
    full_tick_outputs[-1] = full_tick_outputs[-1][:generation_count]
    arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V1] = full_tick_outputs
    patch_stats = list(arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V1])
    patch_stats[0] = patch_stats[0][:, :physical_ticks].contiguous()
    arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V1] = patch_stats
    arguments[TICKS_ARGUMENT_INDEX_V1] = ticks_per_generation
    arguments[GENERATION_COUNT_ARGUMENT_INDEX_V1] = generation_count
    raw = entrypoint(
        list(prepared.gru_tensors),
        dict(prepared.gru_config),
        list(prepared.gmnw_tensors),
        dict(prepared.gmnw_config),
        *arguments,
    )
    if not isinstance(raw, Mapping):
        raise _fail("GMNW+GRU physical entrypoint returned a non-mapping result")
    result = _validate_result(raw)
    _require_internal_status_clean(prepared)
    if result.get("gmnw_gru_physical_gate_entrypoint") is not True:
        raise _fail("GMNW+GRU physical receipt is missing its gate marker")
    if bool(torch.count_nonzero(prepared.gru_tensors[2]).item()):
        raise _fail("GMNW+GRU device policy status is nonzero")
    if bool(torch.count_nonzero(prepared.gmnw_tensors[16]).item()):
        raise _fail("GMNW+GRU device task status is nonzero")
    if prepared.gmnw_config.get("control_matrix_enabled") is True and bool(
        torch.count_nonzero(prepared.gmnw_tensors[26]).item()
    ):
        raise _fail("GMNW control-matrix device status is nonzero")
    if prepared.qualification_decision_tape and bool(
        torch.count_nonzero(
            prepared.qualification_decision_tape["gmnw_qualification_decision_status"]
        ).item()
    ):
        raise _fail("GMNW qualification decision tape status is nonzero")
    _attach_gmnw_selection_metrics_v1(result, prepared)
    return result


def _finalize_gmnw_gru_checkpoint_projection_v1(
    projection: Mapping[str, Any],
    *,
    base_observation_size: int,
    policy_family: str | None = None,
) -> Mapping[str, Any]:
    projection = dict(projection)
    tensors = dict(projection["tensors"])
    tensors["mlp_pack"] = _contract_fixed_mlp_pack(
        tensors["mlp_pack"],
        base_observation_size=base_observation_size,
    )
    metadata = dict(projection["metadata"])
    if metadata.get("policy_weight_mutation_enable") is not False:
        raise _fail("GMNW checkpoint projection requires frozen private GRU weights")
    tensors["gru_pack"] = _contract_private_gru_pack(
        tensors["gru_pack"],
        base_observation_size=base_observation_size,
    )
    _canonicalize_exhausted_private_action_cache(tensors)
    if policy_family is not None:
        metadata = _set_checkpoint_policy_family_metadata_v3(metadata, policy_family)
    projection["metadata"] = metadata
    projection["tensors"] = tensors
    return _with_checkpoint_contract_metadata(projection)


def _gmnw_checkpoint_gru_prepared_v1(
    prepared: PreparedDirectResidentGmnwGruV1,
) -> PreparedDirectResidentGruV3:
    checkpoint_live_banks = dict(prepared.patch.live_banks)
    checkpoint_live_banks["gru_pack"] = _materialize_frozen_gru_checkpoint_rows_cpu(
        prepared
    )
    return PreparedDirectResidentGruV3(
        patch=replace(prepared.patch, live_banks=checkpoint_live_banks),
        gru_tensors=prepared.gru_tensors,
        initial_projection=prepared.initial_projection,
        observation_size=prepared.observation_size,
    )


def project_direct_resident_gmnw_gru_checkpoint_v1(
    prepared: PreparedDirectResidentGmnwGruV1,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
) -> Mapping[str, Any]:
    """Project resident GMNW state through the canonical public GRU layout."""

    if not isinstance(prepared, PreparedDirectResidentGmnwGruV1):
        raise _fail("prepared must be PreparedDirectResidentGmnwGruV1")
    projection = project_direct_resident_gru_checkpoint_v3(
        _gmnw_checkpoint_gru_prepared_v1(prepared),
        direct_result=direct_result,
        completed_generation_count=completed_generation_count,
        pre_materialized_cpu_fields=frozenset({"gru_pack"}),
    )
    arm_codes = tuple(prepared.gmnw_config.get("policy_arm_codes", ()))
    policy_family = (
        _checkpoint_policy_family_for_arm_code_v1(arm_codes[0])
        if len(arm_codes) == 1
        else None
    )
    return _finalize_gmnw_gru_checkpoint_projection_v1(
        projection,
        base_observation_size=prepared.base_observation_size,
        policy_family=policy_family,
    )


def _checkpoint_policy_family_for_arm_code_v1(code: int) -> str:
    if code in {
        GMNW_POLICY_ARM_CODES_V1[("persistent_gru", False)],
        GMNW_POLICY_ARM_CODES_V1[("reset_gru", False)],
        GMNW_POLICY_ARM_CODES_V1[("persistent_gru", True)],
    }:
        return "gru"
    if code in {
        GMNW_POLICY_ARM_CODES_V1[("matched_mlp", False)],
        GMNW_POLICY_ARM_CODES_V1[("matched_mlp", True)],
    }:
        return "mlp"
    if code in {
        GMNW_POLICY_ARM_CODES_V1[("mixed_selection", False)],
        GMNW_POLICY_ARM_CODES_V1[("mixed_selection", True)],
    }:
        return "mixed"
    raise _fail(f"unknown GMNW policy arm code at checkpoint: {code!r}")


def project_direct_resident_gmnw_gru_multicell_checkpoints_v1(
    prepared: PreparedDirectResidentGmnwGruV1,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
    ordered_source_projections: Sequence[Any],
) -> tuple[Mapping[str, Any], ...]:
    """Project one canonical, storage-private owner checkpoint per GMNW cell."""

    if not isinstance(prepared, PreparedDirectResidentGmnwGruV1):
        raise _fail("prepared must be PreparedDirectResidentGmnwGruV1")
    arm_codes = tuple(prepared.gmnw_config.get("policy_arm_codes", ()))
    if not arm_codes:
        policy_families: tuple[str | None, ...] = (None,) * len(
            ordered_source_projections
        )
    elif len(arm_codes) != len(ordered_source_projections):
        raise _fail("GMNW checkpoint policy arm count differs from resident cells")
    else:
        policy_families = tuple(
            _checkpoint_policy_family_for_arm_code_v1(code) for code in arm_codes
        )
    projections = project_direct_resident_gru_multicell_checkpoints_v3(
        _gmnw_checkpoint_gru_prepared_v1(prepared),
        direct_result=direct_result,
        completed_generation_count=completed_generation_count,
        ordered_source_projections=ordered_source_projections,
        pre_materialized_cpu_fields=frozenset({"gru_pack"}),
        checkpoint_policy_families=policy_families,
    )
    return tuple(
        _finalize_gmnw_gru_checkpoint_projection_v1(
            projection,
            base_observation_size=prepared.base_observation_size,
            policy_family=policy_family,
        )
        for projection, policy_family in zip(projections, policy_families, strict=True)
    )


def _validated_resident_segment_range_v1(
    prepared: PreparedDirectResidentGmnwGruV1,
    *,
    logical_generation_count: int,
) -> tuple[int, int]:
    if (
        type(logical_generation_count) is not int
        or logical_generation_count < GMNW_MINIMUM_RESIDENT_SEGMENT_GENERATIONS_V1
        or logical_generation_count > GMNW_MAXIMUM_RESIDENT_SEGMENT_GENERATIONS_V1
    ):
        raise _fail(
            "resident GMNW+GRU segment generations must be an integer in [3,128]"
        )

    # Older admission tests use a structural stub. Real owners always take the
    # strict path below; retaining the stub path avoids changing that test-only API.
    if not isinstance(prepared, PreparedDirectResidentGmnwGruV1):
        if not 4 <= logical_generation_count <= 100:
            raise _fail("resident GMNW+GRU requires generations in [4,100]")
        return 0, logical_generation_count

    arguments = prepared.patch.arguments
    if (
        not isinstance(arguments, Sequence)
        or len(arguments) <= GENERATION_ARGUMENT_INDEX_V1
    ):
        raise _fail("prepared GMNW+GRU owner omits its absolute generation argument")
    initial_generation = arguments[GENERATION_ARGUMENT_INDEX_V1]
    source_generation = getattr(prepared.patch.source, "generation_base", None)
    metadata = (
        prepared.initial_projection.get("metadata")
        if isinstance(prepared.initial_projection, Mapping)
        else getattr(prepared.initial_projection, "metadata", None)
    )
    metadata_generation = (
        metadata.get("generation") if isinstance(metadata, Mapping) else None
    )
    if (
        type(initial_generation) is not int
        or initial_generation < 0
        or source_generation != initial_generation
        or metadata_generation != initial_generation
    ):
        raise _fail(
            "prepared GMNW+GRU owner has inconsistent absolute generation state"
        )

    generation_base = prepared.gmnw_config.get("generation_base")
    generation_rows = prepared.gmnw_config.get("generation_rows_per_cell")
    if (
        type(generation_base) is not int
        or generation_base < 0
        or type(generation_rows) is not int
        or generation_rows < 1
    ):
        raise _fail("prepared GMNW+GRU owner has an invalid declared terminal horizon")
    terminal_generation = generation_base + generation_rows
    if terminal_generation < generation_base:
        raise _fail("prepared GMNW+GRU terminal generation overflowed")
    target_generation = initial_generation + logical_generation_count
    if (
        target_generation < initial_generation
        or initial_generation < generation_base
        or target_generation > terminal_generation
    ):
        raise _fail(
            "resident GMNW+GRU segment exceeds its declared absolute generation range"
        )
    return initial_generation, target_generation


def execute_direct_resident_gmnw_gru_v1(
    module: Any,
    prepared: PreparedDirectResidentGmnwGruV1,
    *,
    logical_generation_count: int,
    run_id: str,
    run_control: Any | None = None,
) -> dict[str, Any]:
    initial_generation, target_generation = _validated_resident_segment_range_v1(
        prepared,
        logical_generation_count=logical_generation_count,
    )
    entrypoint = getattr(module, GMNW_DIRECT_ENTRYPOINT_V1, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {GMNW_DIRECT_ENTRYPOINT_V1}")
    control = (
        create_run_control_v2(module, run_id=run_id)
        if run_control is None
        else run_control
    )
    fresh = dict(control.manifest())
    if (
        fresh.get("run_id") != run_id
        or fresh.get("status") != "initialized"
        or fresh.get("run_started") is not False
        or fresh.get("sealed") is not False
    ):
        raise _fail("GMNW+GRU run control must be fresh and match run_id")
    raw_metadata = (
        prepared.initial_projection.get("metadata")
        if isinstance(prepared.initial_projection, Mapping)
        else getattr(prepared.initial_projection, "metadata", None)
    )
    if not isinstance(raw_metadata, Mapping):
        raise _fail("GMNW initial projection metadata is not a mapping")
    metadata = dict(raw_metadata)
    raw = entrypoint(
        control,
        list(prepared.gru_tensors),
        dict(prepared.gru_config),
        list(prepared.gmnw_tensors),
        dict(prepared.gmnw_config),
        *prepared.patch.arguments,
        logical_generation_count,
        metadata.get("policy_weight_mutation_enable"),
        metadata.get("policy_weight_mutation_mode"),
        metadata.get("policy_weight_mutation_rate"),
        metadata.get("policy_weight_mutation_std"),
    )
    if not isinstance(raw, Mapping):
        raise _fail("GMNW+GRU resident entrypoint returned a non-mapping result")
    completed = raw.get("completed_generations")
    if (
        type(completed) is not int
        or completed < 1
        or completed > logical_generation_count
    ):
        raise _fail("GMNW+GRU resident result is not an exact completed prefix")
    result = _validate_result(raw)
    _require_internal_status_clean(prepared)
    manifest = dict(control.manifest())
    if isinstance(prepared, PreparedDirectResidentGmnwGruV1):
        absolute_coordinates_valid = (
            manifest.get("start_generation") == initial_generation
            and manifest.get("target_next_generation") == target_generation
        )
    else:
        absolute_coordinates_valid = manifest.get("start_generation") in {
            None,
            initial_generation,
        } and manifest.get("target_next_generation") in {None, target_generation}
    if (
        manifest.get("status") != "generation_complete"
        or manifest.get("requested_generations") != logical_generation_count
        or not absolute_coordinates_valid
        or manifest.get("active_generation") != initial_generation + completed - 1
        or manifest.get("sealed") is not False
    ):
        raise _fail("GMNW+GRU run-control manifest is not at a completed boundary")
    if completed == logical_generation_count:
        if (
            manifest.get("awaiting_final_output_ack") is not True
            or manifest.get("awaiting_cancellation_checkpoint") is not False
            or manifest.get("cancel_requested") is not False
        ):
            raise _fail("GMNW+GRU complete run is not awaiting final output ack")
    elif (
        manifest.get("awaiting_final_output_ack") is not False
        or manifest.get("awaiting_cancellation_checkpoint") is not True
        or manifest.get("cancel_requested") is not True
    ):
        raise _fail("GMNW+GRU partial run is not awaiting cancellation checkpoint")
    result["native_run_control_v2"] = manifest
    result["cancelled_prefix"] = completed < logical_generation_count
    _attach_gmnw_selection_metrics_v1(result, prepared)
    return result


__all__ = [
    "DEFAULT_MATCHED_MLP_PATH_V1",
    "DEFAULT_WITNESS_PATH_V1",
    "GMNW_GRU_OBSERVATION_COLUMNS_V1",
    "GMNW_GRU_OBSERVATION_REGISTRY_SHA256_V1",
    "GMNW_ALLOCATION_GENERATION_CAPACITIES_V1",
    "GMNW_CAPABILITY_ENTRYPOINT_V1",
    "GMNW_DIRECT_ENTRYPOINT_V1",
    "GMNW_MATCHED_MLP_PACKED_WIDTH_V1",
    "GMNW_MATCHED_MLP_NEURAL_SCRATCH_WIDTH_V1",
    "GMNW_MATCHED_MLP_AUDIT_TICKS_V1",
    "GMNW_MATCHED_MLP_AUDIT_WORDS_PER_TICK_V1",
    "GMNW_CONTROL_MLP_SCRATCH_TENSOR_INDEX_V1",
    "GMNW_CONTROL_MLP_PACK_TENSOR_INDEX_V1",
    "GMNW_CONTROL_POLICY_ARM_TENSOR_INDEX_V1",
    "GMNW_MATCHED_MLP_SCRATCH_WIDTH_V1",
    "GMNW_PHYSICAL_ENTRYPOINT_V1",
    "GMNW_POLICY_ARMS_V1",
    "GMNW_QUALIFICATION_DECISION_COLUMNS_V1",
    "GMNW_QUALIFICATION_DECISION_ROWS_V1",
    "GMNW_QUALIFICATION_DECISION_TAPE_VERSION_V1",
    "GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1",
    "GMNW_QUALIFICATION_EPISODES_PER_STREAM_V1",
    "GMNW_QUALIFICATION_INITIAL_STREAM_COUNT_V1",
    "GMNW_QUALIFICATION_TRACE_TASK_STATE_COLUMNS_V1",
    "GMNW_QUALIFICATION_TRACE_TENSOR_NAMES_V1",
    "GMNW_QUALIFICATION_TRACE_VERSION_V1",
    "GMNW_SELECTION_COUNT_METRIC_NAMES_V1",
    "GMNW_SELECTION_METRIC_NAMES_V1",
    "GMNW_SELECTION_METRICS_STATUS_NAME_V1",
    "GMNW_SELECTION_METRICS_VERSION_V1",
    "GMNW_SELECTION_POLICY_FAMILIES_V1",
    "GMNW_VENV_CONSEQUENCE_PROFILE_CODES_V1",
    "GMNW_VENV_CONSEQUENCE_PROFILE_TENSOR_INDEX_V1",
    "CudaArchitectureBDirectResidentGmnwGruV1Error",
    "GmnwConfigurationV1",
    "PreparedDirectResidentGmnwGruV1",
    "build_prepared_direct_resident_gmnw_gru_v1",
    "execute_direct_resident_gmnw_gru_physical_v1",
    "execute_direct_resident_gmnw_gru_v1",
    "gmnw_allocation_generation_capacity_v1",
    "project_direct_resident_gmnw_gru_checkpoint_v1",
    "project_direct_resident_gmnw_gru_multicell_checkpoints_v1",
    "gmnw_gru_observation_contract_v1",
    "gmnw_witness_production_layout_crosswalk_v1",
]
