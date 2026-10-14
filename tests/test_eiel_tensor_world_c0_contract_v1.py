from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from engine.tensor_world.comparator import ComparisonRuleV1, OracleComparatorV1
from engine.tensor_world.contracts import (
    BackendCapabilitiesV1,
    EquivalenceClassV1,
    TensorWorldContractError,
    TickReceiptV1,
    TickRequestV1,
)
from engine.tensor_world.state import (
    AgentTensorStateV1,
    HotEventAccumulatorStateV1,
    IdentityTensorStateV1,
    OccupancyTensorStateV1,
    PATCH_OBSERVATION_WIDTH_V1,
    PATCH_POLICY_PACKED_WIDTH_V1,
    PolicyTensorStateV1,
    RngTensorStateV1,
    TensorWorldStateBundleV1,
    TransactionTensorStateV1,
    WorldTensorStateV1,
    state_tensor_names_v1,
)


ROOT = Path(__file__).resolve().parents[1]
C0_ARTIFACT_ROOT = ROOT / "_artifacts/tensor_world_c0_c7_v3/c0"


def _artifact(name: str) -> dict[str, object]:
    value = json.loads((C0_ARTIFACT_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _capabilities() -> BackendCapabilitiesV1:
    return BackendCapabilitiesV1(
        backend_id="tensor_world_c1_v1",
        mechanics=(
            "observation",
            "fixed_mlp",
            "action_selection",
            "movement",
            "occupancy_conflict",
            "gather",
            "energy",
        ),
        batch_sizes=(1, 4, 48),
        grid_sizes=(32,),
        initial_populations=(256,),
        slot_capacities=(512,),
        policy_families=("fixed_mlp",),
        observation_abi="patch_radius1_observation_8x9_plus2_v1",
        policy_abi="fixed_mlp_74x64x64x7_fp32_pack_v1",
        determinism="fixture_trajectory_exact_c1",
        rng_mode="device_resident_fixture_injection_c1",
        residency_mode="gpu_resident_cross_world",
        checkpoint_replay=False,
        status="experimental",
    )


def test_c0_capabilities_are_narrow_and_fail_closed() -> None:
    capabilities = _capabilities()
    assert capabilities.supports(
        batch=48,
        grid_size=32,
        initial_population=256,
        slot_capacity=512,
        policy_family="fixed_mlp",
        mechanics=("observation", "movement"),
    )
    assert not capabilities.supports(
        batch=48,
        grid_size=32,
        initial_population=256,
        slot_capacity=512,
        policy_family="fixed_mlp",
        mechanics=("wdl",),
    )
    assert not capabilities.checkpoint_replay


def test_c0_request_rejects_multitick_and_host_controls() -> None:
    with pytest.raises(TensorWorldContractError, match="exactly one tick"):
        TickRequestV1(tick_count=2)
    with pytest.raises(TensorWorldContractError, match="host controls"):
        TickRequestV1(controls={"pause": True})


def test_c0_receipt_is_bounded_and_nonnegative() -> None:
    receipt = TickReceiptV1(
        backend_id="tensor_world_c1_v1",
        completed_ticks=1,
        host_to_device_bytes=0,
        device_to_host_bytes=0,
        host_callback_count=0,
        synchronization_count=0,
        materialization_count=0,
    )
    assert set(receipt.to_dict()) == {
        "schema_version",
        "backend_id",
        "completed_ticks",
        "host_to_device_bytes",
        "device_to_host_bytes",
        "host_callback_count",
        "synchronization_count",
        "materialization_count",
    }
    with pytest.raises(TensorWorldContractError, match="backend identity"):
        TickReceiptV1("", 1, 0, 0, 0, 0, 0)


def test_c0_state_abi_matches_source_current_patch_fixed_mlp() -> None:
    assert PATCH_OBSERVATION_WIDTH_V1 == 8 * 9 + 2 == 74
    assert PATCH_POLICY_PACKED_WIDTH_V1 == (64 * 74 + 64 + 64 * 64 + 64 + 7 * 64 + 7)
    names = state_tensor_names_v1()
    assert "IdentityTensorStateV1.domain_world_ids" in names
    assert "OccupancyTensorStateV1.slot_grid" in names
    assert "PolicyTensorStateV1.packed_weights" in names


def test_c0_oracle_comparator_classifies_field_mismatches() -> None:
    comparator = OracleComparatorV1(
        (
            ComparisonRuleV1("agents.positions", EquivalenceClassV1.BIT_EXACT_REQUIRED),
            ComparisonRuleV1(
                "policy.logits",
                EquivalenceClassV1.TRAJECTORY_EXACT_REQUIRED,
                atol=1.0e-6,
            ),
            ComparisonRuleV1("diagnostics.wall_ms", EquivalenceClassV1.DIAGNOSTIC_ONLY),
        )
    )
    result = comparator.compare(
        {
            "agents": {"positions": torch.tensor([[1, 2]])},
            "policy": {"logits": torch.tensor([[0.1, 0.2]])},
            "diagnostics": {"wall_ms": 2.0},
        },
        {
            "agents": {"positions": torch.tensor([[1, 3]])},
            "policy": {"logits": torch.tensor([[0.1, 0.2]])},
            "diagnostics": {"wall_ms": 999.0},
        },
    )
    assert not result.passed
    assert result.mismatches[0].subsystem == "agents"
    assert result.mismatches[0].mismatch_count == 1


def test_c0_oracle_comparator_rejects_duplicate_rules() -> None:
    rule = ComparisonRuleV1("agents.energy", EquivalenceClassV1.BIT_EXACT_REQUIRED)
    with pytest.raises(TensorWorldContractError, match="unique"):
        OracleComparatorV1((rule, rule))


def test_c0_bit_exact_comparison_uses_storage_identity() -> None:
    comparator = OracleComparatorV1(
        (ComparisonRuleV1("value", EquivalenceClassV1.BIT_EXACT_REQUIRED),)
    )
    signed_zero = comparator.compare(
        {"value": torch.tensor([0.0])},
        {"value": torch.tensor([-0.0])},
    )
    assert not signed_zero.passed
    same_nan_bits = torch.tensor([float("nan")])
    assert comparator.compare(
        {"value": same_nan_bits.clone()},
        {"value": same_nan_bits.clone()},
    ).passed


def test_c0_state_bundle_validates_the_frozen_layout() -> None:
    batch, capacity, width = 1, 2, 3
    grid = (batch, width, width)
    slots = (batch, capacity)
    state = TensorWorldStateBundleV1(
        world=WorldTensorStateV1(*[torch.zeros(grid) for _ in range(7)]),
        agents=AgentTensorStateV1(
            alive=torch.ones(slots, dtype=torch.bool),
            agent_id=torch.arange(capacity, dtype=torch.int64).reshape(slots),
            pos_y=torch.zeros(slots, dtype=torch.int64),
            pos_x=torch.zeros(slots, dtype=torch.int64),
            energy=torch.zeros(slots),
            health=torch.zeros(slots),
            age=torch.zeros(slots, dtype=torch.int64),
            survival_steps=torch.zeros(slots, dtype=torch.int64),
            energy_collected=torch.zeros(slots, dtype=torch.float64),
        ),
        occupancy=OccupancyTensorStateV1(
            torch.full(grid, -1, dtype=torch.int64),
            torch.full(grid, -1, dtype=torch.int64),
        ),
        policy=PolicyTensorStateV1(
            packed_weights=torch.zeros(batch, capacity, PATCH_POLICY_PACKED_WIDTH_V1),
            drives=torch.zeros(batch, capacity, 4),
            attack_bias=torch.zeros(slots),
            action_temperature=torch.ones(slots),
            drive_logit_gain=torch.zeros(slots),
        ),
        identity=IdentityTensorStateV1(
            torch.arange(batch, dtype=torch.int64),
            torch.arange(capacity, dtype=torch.int64).reshape(slots),
        ),
        rng=RngTensorStateV1(
            torch.zeros(batch, capacity, 7, dtype=torch.float64),
            torch.zeros(batch, dtype=torch.int64),
        ),
        transactions=TransactionTensorStateV1(
            actions=torch.zeros(slots, dtype=torch.int64),
            moved=torch.zeros(slots, dtype=torch.bool),
            gathered=torch.zeros(slots, dtype=torch.bool),
            food_gained=torch.zeros(slots),
            observations=torch.zeros(batch, capacity, PATCH_OBSERVATION_WIDTH_V1),
            logits=torch.zeros(batch, capacity, 7),
        ),
        hot_events=HotEventAccumulatorStateV1(
            torch.zeros(batch, dtype=torch.int64),
            torch.zeros(batch, dtype=torch.int64),
            torch.zeros(batch, dtype=torch.int64),
        ),
    )
    state.validate()
    state.world.food_amount = torch.zeros(batch, width, width + 1)
    with pytest.raises(TensorWorldContractError, match="square world"):
        state.validate()


def test_c0_machine_contracts_are_source_bound_and_internally_consistent() -> None:
    oracle = _artifact("oracle_binding.json")
    capabilities = _artifact("backend_capabilities.json")
    budget = _artifact("latency_budget.json")
    callback = _artifact("callback_rng_decision.json")
    state_schema = _artifact("state_schema.json")
    capability_matrix = _artifact("c1_c2_capability_matrix.json")

    assert oracle["frozen_source_commit"] == "d161ad630d890a2faec2cbb6666b9570782b59c3"
    for relative, expected in oracle["source_files"].items():  # type: ignore[union-attr]
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected
    assert capabilities["abi"] == {
        "observation": "patch_radius1_observation_8x9_plus2_v1",
        "policy": "fixed_mlp_74x64x64x7_fp32_pack_v1",
        "observation_width": PATCH_OBSERVATION_WIDTH_V1,
        "policy_packed_width": PATCH_POLICY_PACKED_WIDTH_V1,
        "action_count": 7,
    }
    allocations = budget["frozen_allocations_ms"]
    assert isinstance(allocations, list)
    assert sum(float(item["incremental"]) for item in allocations) == pytest.approx(
        26.18109479
    )
    assert callback["gate_result"] == "not_host_boundary_blocked"
    assert capability_matrix["capabilities"][-1]["capability"] == "production admission"  # type: ignore[index]
    assert set(state_schema["partitions"]) >= {  # type: ignore[arg-type]
        "world",
        "agents",
        "occupancy",
        "policy",
        "rng",
        "identity",
        "transactions",
        "hot_events",
        "population",
        "mechanics",
    }
