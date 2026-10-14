from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from agents.agent import ACTION_VOCABULARY, DRIVE_VOCABULARY
from agents.chemistry import CHEMISTRY_SLOT_NAMES
from agents.modular_genome_v1 import B2_FROZEN_CAPS, canonical_payload_from_genome_dict
from agents.perception import perception_channel_registry
from evolution.capacity_contract import (
    REASON_MODULE_CAPACITY,
    build_capacity_contract,
    canonical_json_bytes,
    capacity_headroom_summary,
    validate_active_hereditary_payload,
)
from evolution.capacity_errors import CapacityIntegrityError
from evolution.operators_c1 import apply_operator_pipeline
from evolution.population_pkg.spawn import SpawnMixin
from simulation.capacity_epoch import (
    CapacityEpochState,
    capacity_epoch_migration_contract,
    checkpoint_capacity_epoch_metadata,
    neutral_pad_sequence,
    validate_transition,
)
from simulation.checkpoint_bundle_v2 import _build_metadata
from simulation.metrics_pkg.trait_capacity_saturation import trait_capacity_saturation_payloads
from simulation.runner_pkg.runner_base import RunnerBase


def _module(index: int) -> dict:
    return {
        "module_id": f"m{index:03d}",
        "module_type": "generic",
        "params": {},
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
    }


def _payload(module_count: int) -> dict:
    return {
        "genes": {"hidden_size": 64, "memory_dim": 0, "use_gru": False},
        "modules": [_module(i) for i in range(module_count)],
        "connections": [],
    }


class _FixedRng:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def random(self) -> float:
        return float(next(self.values))


def test_contract_matches_authoritative_registries_and_is_canonical() -> None:
    contract = build_capacity_contract()
    active_channels = sum(
        1 for channel in perception_channel_registry() if channel.active_now and not channel.future_reserved
    )
    assert contract["observations"]["implemented_active_channel_count"] == active_channels
    assert contract["actions"]["count"] == len(ACTION_VOCABULARY)
    assert contract["drives"]["count"] == len(DRIVE_VOCABULARY)
    assert contract["chemistry"]["slot_count"] == len(CHEMISTRY_SLOT_NAMES)
    assert contract["structure"]["limits"] == dict(sorted(B2_FROZEN_CAPS.items()))
    assert canonical_json_bytes(contract) == canonical_json_bytes(build_capacity_contract())


def test_empty_founder_is_not_bootstrap_reachable_under_current_operators() -> None:
    contract = build_capacity_contract()
    reachability = contract["structure"]["reachability"]
    assert reachability["ordinary_founder_payload"] == "empty"
    assert reachability["classification"] == "not_bootstrap_reachable"


def test_active_overflow_rejects_while_explicit_legacy_compatibility_clips() -> None:
    overflow = _payload(int(B2_FROZEN_CAPS["max_modules_per_agent"]) + 1)
    with pytest.raises(RuntimeError, match="DET_GENOME_B2_BUDGET_EXCEEDED"):
        validate_active_hereditary_payload(overflow)

    compatibility = canonical_payload_from_genome_dict({**overflow, "b2_invalid_policy": "clip"})
    assert len(compatibility["modules"]) == int(B2_FROZEN_CAPS["max_modules_per_agent"])
    assert compatibility["b2_invalid_policy"] == "clip"


def test_strict_runner_upgrades_active_policy_to_reject(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    cfg = SimpleNamespace(
        genome_b2_interfaces_enable=True,
        genome_b2_invalid_policy="clip",
        **{f"genome_b2_{name}": value for name, value in B2_FROZEN_CAPS.items()},
    )
    owner = SimpleNamespace(sim_cfg=cfg)
    RunnerBase._enforce_b2_campaign_freeze(owner)
    assert cfg.genome_b2_invalid_policy == "reject"


def test_capacity_blocked_operator_event_is_deterministic_and_classified() -> None:
    payload = canonical_payload_from_genome_dict(
        {**_payload(int(B2_FROZEN_CAPS["max_modules_per_agent"])), "b2_invalid_policy": "reject"}
    )

    def run_once() -> tuple[dict, dict]:
        return apply_operator_pipeline(
            payload=payload,
            rng=_FixedRng([0.2, 0.0, 0.0, 0.0]),
            max_ops=1,
            invalid_operator_policy="reject",
        )

    first_payload, first_telemetry = run_once()
    second_payload, second_telemetry = run_once()
    assert first_payload == second_payload
    assert first_telemetry == second_telemetry
    assert first_telemetry["capacity_blocked_count"] == 1
    assert first_telemetry["capacity_blocked_reason_counts"] == {REASON_MODULE_CAPACITY: 1}
    assert len(first_payload["modules"]) == int(B2_FROZEN_CAPS["max_modules_per_agent"])
    assert first_payload["genes"]["c1_capacity_blocked_count"] == 1


def test_spawn_capacity_shape_failure_is_not_swallowed() -> None:
    owner = SpawnMixin()
    owner.env = SimpleNamespace(
        agent_pos_y=np.zeros((2,), dtype=np.int64),
        agent_pos_x=np.zeros((3,), dtype=np.int64),
        agent_alive=np.zeros((2,), dtype=bool),
    )
    with pytest.raises(CapacityIntegrityError, match="capacity.runtime_allocation.failed"):
        owner._ensure_env_capacity_for_id(4)


def test_headroom_and_reporting_use_contract_not_stale_counts() -> None:
    cfg = SimpleNamespace(
        genome_c1_enable=True,
        genome_c2_enable=True,
        structural_payload_activation_enable=False,
        enable_evolvable_sensor_channel_mask=True,
    )
    records = [
        {
            "module_count": 0,
            "connection_count": 0,
            "effective_memory": 32,
            "sensor_channel_count": 3,
            "offspring": 0,
            "c1_capacity_blocked_count": 0,
            "c1_capacity_blocked_reason_counts_json": "{}",
        }
    ]
    contract = build_capacity_contract(sim_cfg=cfg)
    headroom = capacity_headroom_summary(records, contract=contract)
    assert headroom["allocation"]["perception_channels"]["available"] == contract["observations"]["implemented_active_channel_count"]
    assert headroom["representation_limited"] is True

    summary_json, _, structural_json = trait_capacity_saturation_payloads(records, sim_cfg=cfg)
    summary = json.loads(summary_json)
    structural = json.loads(structural_json)
    sensor = summary["traits"]["sensor_channel_active_count"]
    assert sensor["cap_value"] == contract["observations"]["implemented_active_channel_count"]
    assert sensor["cap_source"] == "capacity_contract.observations.implemented_active_channel_count"
    assert structural["structural_reachability_classification"] == "not_bootstrap_reachable"


def test_capacity_epoch_metadata_and_neutral_expansion_are_rng_free() -> None:
    cfg = SimpleNamespace(
        capacity_epoch_metadata_enable=True,
        capacity_epoch_id="capacity-epoch-2",
        capacity_epoch_sequence=2,
        capacity_epoch_parent_id="capacity-epoch-1",
        structural_payload_activation_enable=False,
        genome_c1_enable=True,
        device="cpu",
    )
    metadata = checkpoint_capacity_epoch_metadata(sim_cfg=cfg)
    assert metadata is not None
    assert metadata["capacity_epoch_id"] == "capacity-epoch-2"
    assert metadata["rng_draws_consumed_by_migration"] == 0
    assert neutral_pad_sequence((1, 2), new_capacity=5, neutral_value=0) == (1, 2, 0, 0, 0)
    with pytest.raises(ValueError, match="cannot truncate"):
        neutral_pad_sequence((1, 2), new_capacity=1, neutral_value=0)
    validate_transition(CapacityEpochState.MONITORING, CapacityEpochState.WARNING)
    with pytest.raises(ValueError, match="illegal capacity-epoch transition"):
        validate_transition(CapacityEpochState.MONITORING, CapacityEpochState.MIGRATING)
    migration = capacity_epoch_migration_contract()
    assert migration["rng_draws"] == 0
    assert "action_vocabulary" in migration["semantic_intervention_fields"]

    sim = SimpleNamespace(sim_cfg=cfg, env_cfg=None)
    checkpoint_metadata = _build_metadata(
        sim,
        generation_index=4,
        strict_replay_mode=True,
        world_hash="world",
        population_hash="population",
        rng_snapshot_hash="rng",
        runner_state_hash="runner",
        bank_state_hash="bank",
        resident_runtime_state_hash="resident",
        native_runtime_checkpoint_hash="native",
        event_log_hash="events",
        wal_hash="wal",
    )
    assert checkpoint_metadata["capacity_epoch"]["capacity_epoch_id"] == "capacity-epoch-2"
    assert checkpoint_metadata["capacity_epoch"]["capacity_contract_hash"] == metadata["capacity_contract_hash"]
