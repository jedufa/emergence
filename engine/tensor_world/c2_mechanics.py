"""Static C2 mechanic declarations frozen from the source-current route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


MECHANIC_CONTRACT_VERSION_V1: Final[str] = "tensor_world_mechanic_contract_v1"


@dataclass(frozen=True, slots=True)
class TensorMechanicV1:
    capability_id: str
    capability_version: int
    required_fields: tuple[str, ...]
    mutated_fields: tuple[str, ...]
    logical_phase: str
    dependencies: tuple[str, ...]
    persistent_fields: tuple[str, ...]
    event_outputs: tuple[str, ...]
    checkpoint_fields: tuple[str, ...]
    comparator_surfaces: tuple[str, ...]


PATCH_TENSOR_MECHANIC_V1: Final = TensorMechanicV1(
    "patch_tensor_mechanic", 1,
    ("world.occupancy", "patch.fertility", "patch.pending_emissions"),
    ("world.food_amount", "patch.scent", "patch.waste", "patch.density", "patch.recovery_activity"),
    "pre_observation_world", ("eed_tensor_mechanic",),
    ("patch.scent", "patch.waste", "patch.recovery_activity"),
    ("patch.tick_stats",),
    ("world.food_amount", "patch.scent", "patch.waste", "patch.recovery_activity"),
    ("world.food_amount", "patch.scent", "patch.waste", "patch.density", "patch.recovery_activity"),
)

JUNGLE_TENSOR_MECHANIC_V1: Final = TensorMechanicV1(
    "jungle_tensor_mechanic", 1,
    ("patch.scent", "patch.waste", "patch.density", "patch.cover", "patch.shelter", "agents.position", "rng.jungle_fixture"),
    ("jungle.visibility", "effects.pre_policy_health_delta"),
    "pre_observation_agent_effect", ("patch_tensor_mechanic",),
    ("jungle.visibility",), ("jungle.tick_metrics",), (),
    ("jungle.visibility", "effects.pre_policy_health_delta", "agents.health"),
)

WDL_TENSOR_MECHANIC_V1: Final = TensorMechanicV1(
    "wdl_tensor_mechanic", 1,
    ("wdl.temporal_state", "wdl.schedule", "agents.position"),
    ("wdl.temporal_state", "wdl.schedule", "effects.post_action_health_delta"),
    "split_pre_observation_and_post_action", (),
    ("wdl.cue", "wdl.timer", "wdl.active", "wdl.intensity", "wdl.age", "wdl.event_id", "wdl.prior_cue", "wdl.schedule"),
    ("wdl.tick_metrics", "wdl.action_metrics"),
    ("wdl.cue", "wdl.timer", "wdl.active", "wdl.intensity", "wdl.age", "wdl.event_id", "wdl.prior_cue", "wdl.schedule"),
    ("wdl.temporal_state", "transactions.observations", "effects.post_action_health_delta"),
)

EED_TENSOR_MECHANIC_V1: Final = TensorMechanicV1(
    "eed_tensor_mechanic", 1,
    ("eed.resource_state", "eed.pending_impact", "eed.resource_class"),
    ("eed.resource_amount", "eed.impact_load", "world.food_amount"),
    "pre_patch_resource_and_action_gather", ("wdl_tensor_mechanic",),
    ("eed.resource_amount", "eed.impact_load", "eed.pending_impact", "eed.handling_state"),
    ("eed.hot_path_receipt",),
    ("eed.resource_amount", "eed.impact_load", "eed.pending_impact", "eed.handling_state"),
    ("eed.resource_amount", "eed.impact_load", "eed.hot_path_receipt"),
)

C2_MECHANICS_V1: Final = (
    WDL_TENSOR_MECHANIC_V1,
    EED_TENSOR_MECHANIC_V1,
    PATCH_TENSOR_MECHANIC_V1,
    JUNGLE_TENSOR_MECHANIC_V1,
)
