from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import perf.cuda_architecture_b_generation as generation
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_PROJECTION_PREFIX,
    pack_modular_storage,
)
from perf.eiel_phase_e_modular_tick_fixture_v1 import (
    _development as _fixture_development,
    _module as _fixture_module,
)
from tests.perf.test_eiel_cuda_architecture_b_generation import (
    ACTION_COUNT,
    CAPACITY,
    HIDDEN_SIZE,
    REST_ACTION,
    SLOTS,
    _fixture as _generation_fixture,
    _generation_config,
    generation_build,
)
from tests.perf.test_eiel_cuda_architecture_b_modular_state_abi import _record


FORMAL_FLOAT_ATOL = 2.0e-7


def _config() -> generation.CudaGenerationConfig:
    return generation.CudaGenerationConfig(
        steps=1,
        patch_enabled=False,
        hidden_size=4,
    )


def _phase_e_generation_fixture(
    *, policy_action: int | None
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    tensors, row_state = _generation_fixture(
        patch=False,
        policy_action=policy_action,
    )
    observation_width = 27
    packed_width = (
        HIDDEN_SIZE * observation_width
        + HIDDEN_SIZE
        + HIDDEN_SIZE * HIDDEN_SIZE
        + HIDDEN_SIZE
        + ACTION_COUNT * HIDDEN_SIZE
        + ACTION_COUNT
    )
    tensors["mlp_pack"] = torch.zeros(
        (CAPACITY, packed_width),
        dtype=torch.float32,
        device="cuda",
    )
    if policy_action is not None:
        tensors["mlp_pack"][:, -ACTION_COUNT + policy_action] = 10.0
    tensors["action_cache_actions"] = torch.full(
        (CAPACITY, 1),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    return tensors, row_state


def _phase_e_config() -> generation.CudaGenerationConfig:
    return replace(
        _generation_config(patch=False),
        cache_width=1,
        view_radius=2,
        active_mechanics_enabled=False,
    )


def _modular_bank(
    generation_tensors: dict[str, torch.Tensor],
    row_state: dict[str, np.ndarray],
    *,
    structured: bool,
    development_gains: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, torch.Tensor]:
    records: list[dict | None] = [None] * CAPACITY
    for row, (agent_id, slot) in enumerate(
        zip(row_state["agent_ids"], SLOTS, strict=True)
    ):
        modules = []
        if structured and row == 0:
            modules = [
                _fixture_module(
                    row,
                    0,
                    "generation-owner",
                    input_semantic="self_energy",
                    input_gain=1.0,
                    output_semantic="rest",
                    output_gain=1.0,
                    development=_fixture_development(development_gains),
                )
            ]
        records[int(slot)] = _record(
            modules,
            agent_id=int(agent_id),
            agent_slot_token=100 + int(slot),
        )
    packed = pack_modular_storage(
        records,
        agent_capacity=CAPACITY,
        profile_id="b2-12x32-v1",
    )
    bank = {
        f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}": tensor.cuda().contiguous()
        for name, tensor in packed.tensors.items()
    }
    bank.update(
        slot_to_agent_id=generation_tensors["slot_to_agent_id"],
        slot_tokens=generation_tensors["slot_tokens"],
        alive_mask=generation_tensors["alive_mask"],
    )
    return bank


def _assert_generation_result_exact(actual, expected) -> None:
    assert tuple(vars(actual)) == tuple(vars(expected))
    for name in vars(actual):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


def _assert_generation_tensor_bank_exact(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert tuple(actual) == tuple(expected)
    for name in actual:
        assert torch.equal(actual[name], expected[name]), name


def test_disabled_path_delegates_to_v3_before_reading_modular_arguments(
    monkeypatch,
) -> None:
    class MustNotBeRead:
        def __getattribute__(self, name):
            raise AssertionError(f"disabled modular argument was read: {name}")

    sentinel = object()
    captured = {}

    def fake_legacy(tensors, config, **kwargs):
        captured.update(tensors=tensors, config=config, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(generation, "execute_cuda_generation", fake_legacy)
    tensors = {"legacy": torch.tensor([1])}
    config = _config()
    result = generation.execute_cuda_modular_generation(
        tensors,
        config,
        enabled=False,
        modular_tensors=MustNotBeRead(),
        profile_id=MustNotBeRead(),
    )

    assert result is sentinel
    assert captured["tensors"] is tensors
    assert captured["config"] is config


def test_enabled_path_calls_only_explicit_phase_e_export(monkeypatch) -> None:
    tensor_arguments = (torch.tensor([1]), torch.tensor([2]))
    scalar_values = tuple(range(13))
    entry = SimpleNamespace(
        tensor_arguments=tensor_arguments,
        config_scalar_values=scalar_values,
    )
    calls = []

    class Module:
        def eiel_cuda_architecture_b_generation_phase_e_v1(self, *arguments):
            calls.append(arguments)
            return ("native-output",)

    config = _config()
    native_call = SimpleNamespace(
        module=Module(),
        arguments=("legacy-argument-0", "legacy-argument-1"),
        config=config,
    )
    monkeypatch.setattr(
        generation,
        "prepare_modular_generation_entry",
        lambda *args, **kwargs: entry,
    )
    monkeypatch.setattr(
        generation,
        "_prepare_cuda_generation_call",
        lambda *args, **kwargs: native_call,
    )
    sentinel = object()
    monkeypatch.setattr(
        generation,
        "_cuda_generation_result",
        lambda outputs, accepted_config: sentinel,
    )

    result = generation.execute_cuda_modular_generation(
        {"alive_mask": torch.tensor([True, True])},
        config,
        enabled=True,
        modular_tensors={"modular": "authority"},
        profile_id="b2-12x32-v1",
    )

    assert result is sentinel
    assert calls == [
        (
            "legacy-argument-0",
            "legacy-argument-1",
            list(tensor_arguments),
            *scalar_values,
        )
    ]


def test_enabled_path_forbids_hidden_v3_fallback(monkeypatch) -> None:
    entry = SimpleNamespace(tensor_arguments=(), config_scalar_values=tuple(range(13)))
    monkeypatch.setattr(
        generation,
        "prepare_modular_generation_entry",
        lambda *args, **kwargs: entry,
    )
    monkeypatch.setattr(
        generation,
        "_prepare_cuda_generation_call",
        lambda *args, **kwargs: SimpleNamespace(
            module=SimpleNamespace(), arguments=(), config=_config()
        ),
    )

    with pytest.raises(RuntimeError, match="hidden v3 fallback is forbidden"):
        generation.execute_cuda_modular_generation(
            {"alive_mask": torch.tensor([True])},
            _config(),
            enabled=True,
            modular_tensors={},
            profile_id="b2-12x32-v1",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("structural_development_mature_start_step", 20.0),
        ("structural_development_mature_start_step", "20"),
        ("structural_development_mature_start_step", True),
        ("structural_development_late_start_step", 40.0),
        ("structural_development_late_start_step", "40"),
        ("structural_development_late_start_step", True),
    ),
)
def test_enabled_path_rejects_coerced_development_boundaries(
    field: str,
    value: object,
) -> None:
    config = replace(_config(), **{field: value})

    with pytest.raises(RuntimeError, match="must be an exact int"):
        generation.execute_cuda_modular_generation(
            {"alive_mask": torch.tensor([True])},
            config,
            enabled=True,
            modular_tensors={},
            profile_id="b2-12x32-v1",
        )


def test_modular_generation_capability_is_bounded_and_explicit() -> None:
    capability = generation.CUDA_MODULAR_GENERATION_CAPABILITIES
    assert capability["explicit_native_owner"] is True
    assert capability["native_cpp_tick_loop"] is True
    assert capability["hidden_fallback"] is False
    assert capability["births"] is False
    assert capability["turnover"] is False
    assert capability["multi_cell"] is False
    assert capability["promotion_qualified"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_default_disabled_path_is_bit_exact_to_generation_v3(
    generation_build,
) -> None:
    legacy_tensors, _ = _generation_fixture(patch=False)
    disabled_tensors, _ = _generation_fixture(patch=False)
    config = _generation_config(patch=False)

    expected = generation.execute_cuda_generation(
        legacy_tensors,
        config,
        build_directory=generation_build,
    )
    actual = generation.execute_cuda_modular_generation(
        disabled_tensors,
        config,
        build_directory=generation_build,
    )

    _assert_generation_result_exact(actual, expected)
    _assert_generation_tensor_bank_exact(disabled_tensors, legacy_tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_enabled_neutral_modular_bank_is_bit_exact_to_generation_v3(
    generation_build,
) -> None:
    legacy_tensors, _ = _phase_e_generation_fixture(policy_action=REST_ACTION)
    modular_tensors, row_state = _phase_e_generation_fixture(
        policy_action=REST_ACTION
    )
    neutral_bank = _modular_bank(
        modular_tensors,
        row_state,
        structured=False,
    )
    modular_before = {
        name: tensor.clone()
        for name, tensor in neutral_bank.items()
        if name.startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    }
    config = _phase_e_config()

    expected = generation.execute_cuda_generation(
        legacy_tensors,
        config,
        build_directory=generation_build,
    )
    actual = generation.execute_cuda_modular_generation(
        modular_tensors,
        config,
        enabled=True,
        modular_tensors=neutral_bank,
        profile_id="b2-12x32-v1",
        build_directory=generation_build,
    )

    _assert_generation_result_exact(actual, expected)
    _assert_generation_tensor_bank_exact(modular_tensors, legacy_tensors)
    for name, before in modular_before.items():
        assert torch.equal(neutral_bank[name], before), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_enabled_structured_bank_changes_state_output_and_action_evidence(
    generation_build,
) -> None:
    source_tensors, row_state = _phase_e_generation_fixture(policy_action=None)
    neutral_tensors = {
        name: tensor.clone() for name, tensor in source_tensors.items()
    }
    structured_tensors = {
        name: tensor.clone() for name, tensor in source_tensors.items()
    }
    neutral_bank = _modular_bank(
        neutral_tensors,
        row_state,
        structured=False,
    )
    structured_bank = _modular_bank(
        structured_tensors,
        row_state,
        structured=True,
    )
    state_name = f"{MODULAR_STORAGE_PROJECTION_PREFIX}module_state"
    output_name = (
        f"{MODULAR_STORAGE_PROJECTION_PREFIX}module_previous_output"
    )
    present_name = f"{MODULAR_STORAGE_PROJECTION_PREFIX}module_state_present"
    output_present_name = (
        f"{MODULAR_STORAGE_PROJECTION_PREFIX}module_output_present"
    )
    structured_state_before = structured_bank[state_name].clone()
    structured_output_before = structured_bank[output_name].clone()
    policy_rng_before = {
        name: structured_tensors[name].clone()
        for name in ("pcg64_state_words", "pcg64_buffer_words")
    }
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state().clone()
    config = replace(_phase_e_config(), steps=1)

    neutral = generation.execute_cuda_modular_generation(
        neutral_tensors,
        config,
        enabled=True,
        modular_tensors=neutral_bank,
        profile_id="b2-12x32-v1",
        deterministic=True,
        build_directory=generation_build,
    )
    structured = generation.execute_cuda_modular_generation(
        structured_tensors,
        config,
        enabled=True,
        modular_tensors=structured_bank,
        profile_id="b2-12x32-v1",
        deterministic=True,
        build_directory=generation_build,
    )

    structured_slot = int(SLOTS[0])
    assert not torch.equal(
        structured_bank[state_name][structured_slot],
        structured_state_before[structured_slot],
    )
    assert not torch.equal(
        structured_bank[output_name][structured_slot],
        structured_output_before[structured_slot],
    )
    assert bool(structured_bank[present_name][structured_slot, 0])
    assert bool(structured_bank[output_present_name][structured_slot, 0])
    assert torch.count_nonzero(
        structured_bank[output_name][structured_slot]
    ) > 0
    assert torch.equal(
        structured.slots_by_alive_order,
        neutral.slots_by_alive_order,
    )

    structured_rows = torch.nonzero(
        structured.slots_by_alive_order == structured_slot,
        as_tuple=False,
    ).flatten()
    assert structured_rows.numel() == 1
    structured_row = int(structured_rows.item())
    module_output = structured_bank[output_name][structured_slot, 0, 0]
    assert int(
        structured_bank[
            f"{MODULAR_STORAGE_PROJECTION_PREFIX}output_binding_count"
        ][structured_slot, 0]
    ) == 1
    assert bool(
        structured_bank[f"{MODULAR_STORAGE_PROJECTION_PREFIX}output_enabled"][
            structured_slot, 0, 0
        ]
    )
    assert int(
        structured_bank[
            f"{MODULAR_STORAGE_PROJECTION_PREFIX}output_action_index"
        ][structured_slot, 0, 0]
    ) == REST_ACTION
    assert float(
        structured_bank[f"{MODULAR_STORAGE_PROJECTION_PREFIX}output_gain"][
            structured_slot, 0, 0
        ]
    ) == 1.0
    expected_residuals = torch.zeros_like(neutral.logits)
    expected_residuals[structured_row, REST_ACTION] = torch.clamp(
        module_output + module_output,
        min=-0.25,
        max=0.25,
    )
    expected_logits = neutral.logits + expected_residuals
    torch.testing.assert_close(
        structured.logits,
        expected_logits,
        rtol=0.0,
        atol=FORMAL_FLOAT_ATOL,
    )
    expected_actions = expected_logits.argmax(dim=1).to(torch.int32)
    active_row_count = int(structured.slots_by_alive_order.numel())
    expected_live_actions = expected_actions[:active_row_count]
    expected_action_slots = torch.full_like(
        structured.actions_by_tick_slot,
        REST_ACTION,
    )
    expected_action_slots[0, structured.slots_by_alive_order.long()] = (
        expected_live_actions
    )
    assert torch.equal(
        structured.actions_by_tick_slot,
        expected_action_slots,
    )
    active_slots = structured.slots_by_alive_order.long()
    assert torch.equal(
        structured_tensors["action_cache_next"].index_select(0, active_slots),
        expected_live_actions,
    )
    assert torch.equal(
        structured_tensors["action_cache_actions"][active_slots, 0],
        expected_live_actions,
    )
    assert int(expected_actions[structured_row]) == REST_ACTION
    assert not torch.equal(
        structured.actions_by_tick_slot,
        neutral.actions_by_tick_slot,
    )

    gain_name = (
        f"{MODULAR_STORAGE_PROJECTION_PREFIX}development_effective_gain"
    )
    for age, expected_gain in ((19, 0.25), (20, 0.5), (39, 0.5), (40, 0.75)):
        boundary_tensors, boundary_row_state = _phase_e_generation_fixture(
            policy_action=None
        )
        boundary_bank = _modular_bank(
            boundary_tensors,
            boundary_row_state,
            structured=True,
            development_gains=(0.25, 0.5, 0.75),
        )
        boundary_tensors["survival_steps"][structured_slot] = age
        generation.execute_cuda_modular_generation(
            boundary_tensors,
            config,
            enabled=True,
            modular_tensors=boundary_bank,
            profile_id="b2-12x32-v1",
            deterministic=True,
            build_directory=generation_build,
        )
        torch.testing.assert_close(
            boundary_bank[gain_name][structured_slot, 0],
            torch.tensor(expected_gain, device="cuda"),
            rtol=0.0,
            atol=0.0,
        )

    assert int(structured.next_rng_counter.item()) == config.rng_counter
    for name in (
        "next_rng_counter",
        "pcg64_state_words",
        "pcg64_buffer_words",
        "policy_pcg64_state_after_action_by_tick",
        "policy_pcg64_buffer_after_action_by_tick",
        "drive_noise_rng_semantic_draws_by_tick",
        "drive_noise_rng_status_by_tick",
        "runner_shuffle_semantic_draws_by_tick",
        "runner_shuffle_status_by_tick",
        "chemistry_semantic_draws_by_tick",
        "chemistry_status_by_tick",
        "metrics_np_semantic_draws",
    ):
        assert torch.equal(getattr(structured, name), getattr(neutral, name)), name
    for name, before in policy_rng_before.items():
        assert torch.equal(getattr(structured, name), before), name
    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng_before)

    capability = generation.CUDA_MODULAR_GENERATION_CAPABILITIES
    receipt = generation.CUDA_GENERATION_PREBUILT_CAPABILITY_CONTRACT[
        "phase_e_modular_generation"
    ]
    assert receipt == capability
    assert capability["lane"] == "single_cell_native_generation_phase_e_v1"
    assert capability["explicit_native_owner"] is True
    assert capability["native_cpp_tick_loop"] is True
    assert capability["modular_residual_before_action_selection"] is True
    assert capability["hidden_fallback"] is False
