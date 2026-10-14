from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import perf.cuda_architecture_b_modular_policy_step as policy_step
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_PROJECTION_PREFIX,
)
from perf.cuda_architecture_b_modular_state_bank import (
    MODULAR_STATE_BANK_FIELD_COUNT,
    build_modular_state_bank_manifest,
)


CAPACITY = 4
OBSERVATION_WIDTH = 27
HIDDEN_SIZE = 4
ACTION_COUNT = 7
PROFILE_ID = "b2-12x32-v1"
MODULE = (
    Path(__file__).resolve().parents[2]
    / "engine/perf/cuda_architecture_b_modular_policy_step.py"
)


def _resident(profile_id: str = PROFILE_ID) -> dict[str, torch.Tensor]:
    manifest = build_modular_state_bank_manifest(CAPACITY, profile_id)
    resident = {
        field.projected_name: torch.full(
            field.shape,
            field.fill_value,
            dtype=field.torch_dtype,
        )
        for field in manifest.fields
    }
    agent_ids = torch.tensor([-1, 11, -1, 17], dtype=torch.int64)
    slot_tokens = torch.tensor([70, 110, 130, 170], dtype=torch.int64)
    alive = torch.tensor([False, True, False, True], dtype=torch.bool)
    prefix = MODULAR_STORAGE_PROJECTION_PREFIX
    resident[f"{prefix}agent_occupied"].copy_(alive)
    resident[f"{prefix}agent_id"].copy_(agent_ids)
    resident[f"{prefix}agent_slot_token"].copy_(slot_tokens)
    packed_width = (
        HIDDEN_SIZE * OBSERVATION_WIDTH
        + HIDDEN_SIZE
        + HIDDEN_SIZE * HIDDEN_SIZE
        + HIDDEN_SIZE
        + ACTION_COUNT * HIDDEN_SIZE
        + ACTION_COUNT
    )
    resident.update(
        {
            "slot_to_agent_id": agent_ids,
            "slot_tokens": slot_tokens,
            "alive_mask": alive,
            "survival_steps": torch.tensor([0, 19, 20, 40], dtype=torch.int32),
            "mlp_pack": torch.zeros(
                (CAPACITY, packed_width), dtype=torch.float32
            ),
            "drives": torch.zeros((CAPACITY, 4), dtype=torch.float32),
            "attack_bias": torch.zeros(CAPACITY, dtype=torch.float32),
            "action_temperature": torch.ones(CAPACITY, dtype=torch.float32),
            "drive_logit_gain": torch.ones(CAPACITY, dtype=torch.float32),
            "action_cache_actions": torch.full(
                (CAPACITY, 1), -1, dtype=torch.int32
            ),
            "action_cache_token": torch.full(
                (CAPACITY,), -1, dtype=torch.int64
            ),
            "action_cache_cursor": torch.full(
                (CAPACITY,), -1, dtype=torch.int32
            ),
            "action_cache_next": torch.full(
                (CAPACITY,), -1, dtype=torch.int32
            ),
            "pcg64_state_words": torch.empty(0, dtype=torch.uint64),
            "pcg64_buffer_words": torch.empty(0, dtype=torch.int64),
        }
    )
    return resident


def _arguments(resident: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "resident_tensors": resident,
        "enabled": True,
        "profile_id": PROFILE_ID,
        "observations": torch.arange(
            CAPACITY * OBSERVATION_WIDTH, dtype=torch.float32
        ).reshape(CAPACITY, OBSERVATION_WIDTH),
        "fatigue": torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float32),
        "miss_slots": torch.tensor([3, 1], dtype=torch.int32),
        "active_count": torch.tensor([2], dtype=torch.int64),
        "hidden_size": HIDDEN_SIZE,
        "action_count": ACTION_COUNT,
        "cache_width": 1,
        "policy_stride": 1,
        "max_births": 0,
        "turnover_enabled": False,
        "early_to_adult_step": 20,
        "adult_to_late_step": 40,
        "deterministic": True,
        "allow_cpu_validation": True,
    }


def test_tick_then_compact_residual_mlp_order_and_receipt(monkeypatch) -> None:
    resident = _resident()
    arguments = _arguments(resident)
    observations = arguments["observations"]
    miss_slots = arguments["miss_slots"]
    active_count = arguments["active_count"]
    full_residuals = torch.arange(
        CAPACITY * ACTION_COUNT, dtype=torch.float32
    ).reshape(CAPACITY, ACTION_COUNT)
    calls: list[str] = []
    captured: dict[str, object] = {}
    policy_sentinel = object()

    def fake_tick(planes, tick_observations, fatigue, stage_indices, *, action_width):
        calls.append("tick")
        assert planes is resident
        assert tick_observations is observations
        assert tuple(fatigue.shape) == (CAPACITY,)
        assert torch.equal(stage_indices, torch.tensor([0, 0, 1, 2]))
        assert action_width == ACTION_COUNT
        return SimpleNamespace(
            action_residuals=full_residuals,
            audit_tensors={"tick_probe": torch.tensor([41])},
            receipt=SimpleNamespace(lane="fake-modular-tick"),
        )

    def fake_mlp(**kwargs):
        calls.append("mlp")
        captured.update(kwargs)
        return policy_sentinel

    monkeypatch.setattr(policy_step, "execute_modular_tick", fake_tick)
    monkeypatch.setattr(
        policy_step,
        "fixed_mlp_refill_counted_with_residual_cuda",
        fake_mlp,
    )
    result = policy_step.execute_modular_policy_step(**arguments)

    assert result is not None
    assert calls == ["tick", "mlp"]
    assert result.policy_result is policy_sentinel
    assert captured["slots"] is miss_slots
    assert captured["active_count"] is active_count
    index = miss_slots.to(dtype=torch.int64)
    assert torch.equal(
        captured["observations"], observations.index_select(0, index)
    )
    assert torch.equal(
        captured["logit_residuals"], full_residuals.index_select(0, index)
    )
    assert captured["mlp_pack"] is resident["mlp_pack"]
    assert captured["pcg64_state_words"] is resident["pcg64_state_words"]
    assert torch.equal(
        result.audit_tensors["stage_indices"], torch.tensor([0, 0, 1, 2])
    )
    assert result.modular_tick_audit_tensors["tick_probe"].item() == 41
    receipt = result.receipt.to_dict()
    assert receipt["compact_row_count"] == 2
    assert receipt["all_active_policy_rows_are_misses"] is True
    assert receipt["residual_applied_before_action_selection"] is True
    assert receipt["device_to_host_synchronizations"] == 0
    assert receipt["hidden_fallback_count"] == 0
    assert receipt["lifecycle_supported"] is False
    assert len(result.entry_receipt.tensor_argument_names) == (
        MODULAR_STATE_BANK_FIELD_COUNT
    )


def test_default_off_returns_before_reading_modular_arguments(monkeypatch) -> None:
    class MustNotBeRead:
        def __getattribute__(self, name):
            raise AssertionError(f"disabled argument was read: {name}")

    def must_not_run(*args, **kwargs):
        raise AssertionError("disabled policy step executed work")

    monkeypatch.setattr(policy_step, "execute_modular_tick", must_not_run)
    monkeypatch.setattr(
        policy_step,
        "fixed_mlp_refill_counted_with_residual_cuda",
        must_not_run,
    )
    assert policy_step.execute_modular_policy_step(
        MustNotBeRead(),
        enabled=False,
        profile_id=MustNotBeRead(),
        observations=MustNotBeRead(),
        fatigue=MustNotBeRead(),
        miss_slots=MustNotBeRead(),
        active_count=MustNotBeRead(),
        hidden_size=-1,
        action_count=-1,
        cache_width=-1,
        policy_stride=-1,
        max_births=99,
        turnover_enabled=True,
        early_to_adult_step=-1,
        adult_to_late_step=-1,
        deterministic=MustNotBeRead(),
    ) is None


@pytest.mark.parametrize(
    ("update", "message"),
    (
        ({"cache_width": 2}, "cache_width"),
        ({"policy_stride": 2}, "policy_stride"),
        ({"max_births": 1}, "max_births=0"),
        ({"turnover_enabled": True}, "turnover disabled"),
        ({"profile_id": "b2-13x33-v1"}, "profile"),
    ),
)
def test_first_live_scope_rejects_policy_or_lifecycle_drift(update, message) -> None:
    arguments = _arguments(_resident())
    arguments.update(update)
    with pytest.raises(RuntimeError, match=message):
        policy_step.execute_modular_policy_step(**arguments)


@pytest.mark.parametrize(
    ("damage", "message"),
    (
        ("active_count", "every compact miss row"),
        ("dead_slot", "alive slot"),
        ("duplicate_slot", "each active slot exactly once"),
        ("cached_alive", "every alive policy row"),
        ("negative_survival", "nonnegative"),
    ),
)
def test_resident_miss_and_stage_inputs_fail_closed(damage: str, message: str) -> None:
    resident = _resident()
    arguments = _arguments(resident)
    if damage == "active_count":
        arguments["active_count"] = torch.tensor([1], dtype=torch.int64)
    elif damage == "dead_slot":
        resident["alive_mask"][3] = False
        resident[f"{MODULAR_STORAGE_PROJECTION_PREFIX}agent_occupied"][3] = False
    elif damage == "duplicate_slot":
        arguments["miss_slots"] = torch.tensor([1, 1], dtype=torch.int32)
    elif damage == "cached_alive":
        arguments["miss_slots"] = torch.tensor([1], dtype=torch.int32)
        arguments["active_count"] = torch.tensor([1], dtype=torch.int64)
    else:
        resident["survival_steps"][0] = -1
    with pytest.raises(policy_step.ModularPolicyStepError, match=message):
        policy_step.execute_modular_policy_step(**arguments)


@pytest.mark.parametrize(
    ("first", "second"),
    ((0, 40), (20, 20), (41, 40), (True, 40)),
)
def test_stage_thresholds_are_explicit_positive_and_ordered(first, second) -> None:
    arguments = _arguments(_resident())
    arguments["early_to_adult_step"] = first
    arguments["adult_to_late_step"] = second
    with pytest.raises(policy_step.ModularPolicyStepError, match="stage|early"):
        policy_step.execute_modular_policy_step(**arguments)


def test_production_path_has_no_host_transfer_or_scalar_extraction() -> None:
    source = MODULE.read_text(encoding="utf-8")
    forbidden = (".cpu(", ".item(", ".tolist(", "numpy", "torch.cuda.synchronize")
    assert not any(token in source for token in forbidden)
