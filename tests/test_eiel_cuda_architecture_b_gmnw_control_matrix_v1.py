from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from engine.perf import cuda_architecture_b_direct_resident_gmnw_gru_v1 as gmnw


ROOT = Path(__file__).resolve().parents[1]
KERNELS = (
    ROOT / "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"
)
GMNW_PERSISTENT = ROOT / "engine/perf/cuda_architecture_b_persistent_gmnw_v1.cuh"
GMNW_CONTEXT = ROOT / "engine/perf/cuda_architecture_b_flattened_gmnw_controller_v1.hpp"
BINDING = (
    ROOT / "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_binding.cpp"
)
TURNOVER = ROOT / "engine/perf/cuda_architecture_b_flattened_turnover_v1_kernels.cuh"
TURNOVER_SCALAR = (
    ROOT / "engine/perf/cuda_architecture_b_persistent_turnover_scalar_genome.cuh"
)


@dataclass(frozen=True)
class _Configuration:
    cache_width: int = 8
    max_births_per_tick: int = 2
    population_size: int = 2


@dataclass(frozen=True)
class _Source:
    configuration: _Configuration
    live_state: dict[str, torch.Tensor]
    outputs: dict[str, torch.Tensor]
    full_tick_output_tensors: list[torch.Tensor]
    ordinary_birth_output_tensors: list[torch.Tensor]
    event_state_tensors: list[torch.Tensor]
    ordinary_birth_state_tensors: list[torch.Tensor]
    turnover_state_tensors: list[torch.Tensor]
    turnover_shadow_tensors: list[torch.Tensor]
    live_banks: dict[str, Any]


@pytest.fixture
def cpu_builder(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    capacity = 2
    matched_mlp = torch.arange(
        gmnw.GMNW_MATCHED_MLP_PACKED_WIDTH_V1,
        dtype=torch.float32,
    )
    loaded_paths: list[Any] = []
    base_calls: list[dict[str, Any]] = []
    projection = SimpleNamespace(
        metadata={
            "gru_cost_base": 0.0,
            "gru_cost_per_dim": 0.0,
            "policy_weight_mutation_enable": False,
        }
    )

    def build_base(
        _runner: Any,
        *,
        logical_generation_count: int,
        batch_cells: int,
        **_kwargs: Any,
    ) -> Any:
        assert logical_generation_count == 4
        requests = tuple(_kwargs.get("per_cell_requests") or ())
        cells = len(requests) if requests else batch_cells
        checkpoints = tuple(_kwargs.get("source_checkpoint_projections") or ())
        source_projections = checkpoints or (projection,) * cells
        cell_ids = tuple(_kwargs.get("cell_ids") or map(str, range(cells)))
        base_calls.append(
            {
                "runner": _runner,
                "batch_cells": batch_cells,
                "per_cell_requests": _kwargs.get("per_cell_requests"),
                "source_checkpoint_projections": _kwargs.get(
                    "source_checkpoint_projections"
                ),
                "cell_ids": _kwargs.get("cell_ids"),
            }
        )
        alive = torch.ones((cells, capacity), dtype=torch.bool)
        mlp_pack = torch.zeros((cells, capacity, 1), dtype=torch.float32)
        observation_size = 74 + int(_kwargs["observation_extra_columns"])
        shared_pack = _kwargs["immutable_shared_gru_pack"]
        live_state = {
            "energy": torch.ones((cells, capacity), dtype=torch.float32),
            "alive_mask": alive,
            "mlp_pack": mlp_pack,
            "policy_mem_dim": torch.zeros((cells, capacity), dtype=torch.int32),
            "policy_use_gru": torch.zeros((cells, capacity), dtype=torch.bool),
            "hmem": torch.ones((cells, capacity, 64), dtype=torch.float32),
            "gru_upkeep_cost": torch.ones((cells, capacity), dtype=torch.float32),
            "action_temperature": torch.ones((cells, capacity), dtype=torch.float32),
            "attack_bias": torch.ones((cells, capacity), dtype=torch.float32),
            "drive_logit_gain": torch.ones((cells, capacity), dtype=torch.float32),
            "gru_pack": shared_pack,
        }
        outputs = allocate_tick_outputs(
            capacity=capacity,
            observation_size=observation_size,
            output_ticks=int(_kwargs["output_ticks"]),
            batch_cells=cells,
        )
        outputs.update(allocate_birth_outputs())
        event_status = outputs["event_status_by_tick"]
        placeholder = torch.empty(0)
        source = _Source(
            configuration=_Configuration(),
            live_state=live_state,
            outputs=outputs,
            full_tick_output_tensors=[],
            ordinary_birth_output_tensors=[],
            event_state_tensors=[event_status],
            ordinary_birth_state_tensors=[placeholder] * 89,
            turnover_state_tensors=[placeholder] * 87,
            turnover_shadow_tensors=[placeholder] * 87,
            live_banks={},
        )
        return SimpleNamespace(
            patch=SimpleNamespace(source=source),
            initial_projection=source_projections[0],
            observation_size=observation_size,
            source_projections=source_projections,
            cell_ids=cell_ids,
        )

    def allocate_tick_outputs(
        *,
        capacity: int,
        observation_size: int,
        output_ticks: int,
        batch_cells: int,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        outputs = {
            "ordered_agent_ids": torch.full(
                (batch_cells, output_ticks, capacity), -1, dtype=torch.int64
            ),
            "ordered_slots": torch.full(
                (batch_cells, output_ticks, capacity), -1, dtype=torch.int32
            ),
            "active_count": torch.zeros((batch_cells, output_ticks), dtype=torch.int64),
            "observations": torch.zeros(
                (batch_cells, output_ticks, capacity, observation_size),
                dtype=torch.float32,
            ),
            "event_status_by_tick": torch.zeros(
                (batch_cells, output_ticks), dtype=torch.int32
            ),
        }
        for name in (
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
        ):
            outputs[name] = torch.zeros(1)
        return outputs

    def allocate_birth_outputs(**_kwargs: Any) -> dict[str, torch.Tensor]:
        return {name: torch.zeros(1) for name in gmnw.ORDINARY_BIRTH_OUTPUT_FIELDS}

    def prepare(source: _Source) -> Any:
        arguments: list[Any] = [None] * len(gmnw.DIRECT_RESIDENT_ARGUMENT_NAMES_V1)
        arguments[gmnw.MLP_ARGUMENT_INDEX_V1] = source.live_state["mlp_pack"]
        arguments[gmnw.OBSERVATION_ARGUMENT_INDEX_V1] = source.outputs["observations"]
        arguments[gmnw.CACHE_WIDTH_ARGUMENT_INDEX_V1] = source.configuration.cache_width
        return SimpleNamespace(
            source=source,
            arguments=tuple(arguments),
            live_banks=source.live_banks,
        )

    def load_matched_mlp(path: Any) -> torch.Tensor:
        loaded_paths.append(path)
        return matched_mlp

    monkeypatch.setattr(
        gmnw,
        "_validated_capability",
        lambda _module: {
            "qualification_trace_version": 1,
            "qualification_decision_tape_version": 1,
        },
    )
    monkeypatch.setattr(gmnw, "build_prepared_direct_resident_gru_v3", build_base)
    monkeypatch.setattr(gmnw, "_load_witness", lambda _path: torch.zeros(1))
    monkeypatch.setattr(
        gmnw,
        "_expand_witness_pack",
        lambda _witness, *, cells, capacity, device, **_kwargs: torch.zeros(
            (cells, capacity, 1), dtype=torch.float32, device=device
        ),
    )
    monkeypatch.setattr(
        gmnw,
        "_shared_witness_pack",
        lambda _witness, *, device, **_kwargs: torch.zeros(
            (1,), dtype=torch.float32, device=device
        ),
    )
    monkeypatch.setattr(
        gmnw,
        "_expand_fixed_mlp_pack",
        lambda tensor, **_kwargs: tensor,
    )
    monkeypatch.setattr(gmnw, "_allocate_tick_outputs", allocate_tick_outputs)
    monkeypatch.setattr(gmnw, "_allocate_birth_outputs", allocate_birth_outputs)
    monkeypatch.setattr(gmnw, "prepare_direct_resident_trajectory_v1", prepare)
    monkeypatch.setattr(gmnw, "_load_matched_mlp", load_matched_mlp)
    return {
        "loaded_paths": loaded_paths,
        "matched_mlp": matched_mlp,
        "projection": projection,
        "base_calls": base_calls,
    }


@pytest.mark.parametrize(
    ("policy_arm", "reset_hidden_each_tick"),
    [
        ("persistent_gru", False),
        ("reset_gru", True),
        ("matched_mlp", False),
        ("mixed_selection", False),
    ],
)
def test_configuration_accepts_each_control_arm_with_exact_reset_contract(
    policy_arm: str,
    reset_hidden_each_tick: bool,
) -> None:
    config = gmnw.GmnwConfigurationV1(
        policy_arm=policy_arm,
        reset_hidden_each_tick=reset_hidden_each_tick,
        gru_founder_count=4 if policy_arm == "mixed_selection" else None,
    )

    gmnw._validated_configuration(config)


@pytest.mark.parametrize(
    "config",
    [
        gmnw.GmnwConfigurationV1(
            policy_arm="persistent_gru", reset_hidden_each_tick=True
        ),
        gmnw.GmnwConfigurationV1(policy_arm="reset_gru", reset_hidden_each_tick=False),
        gmnw.GmnwConfigurationV1(policy_arm="matched_mlp", reset_hidden_each_tick=True),
    ],
)
def test_configuration_rejects_reset_flag_unless_and_until_arm_is_reset(
    config: gmnw.GmnwConfigurationV1,
) -> None:
    with pytest.raises(
        gmnw.CudaArchitectureBDirectResidentGmnwGruV1Error,
        match="reset_hidden_each_tick must be true exactly for reset_gru",
    ):
        gmnw._validated_configuration(config)


def test_configuration_rejects_unknown_policy_arm() -> None:
    config = gmnw.GmnwConfigurationV1(policy_arm="unknown")

    with pytest.raises(
        gmnw.CudaArchitectureBDirectResidentGmnwGruV1Error,
        match="policy_arm must be one of",
    ):
        gmnw._validated_configuration(config)


def test_immediate_and_mixed_arm_codes_preserve_delayed_abi_values() -> None:
    assert gmnw.GMNW_POLICY_ARM_CODES_V1[("persistent_gru", False)] == 0
    assert gmnw.GMNW_POLICY_ARM_CODES_V1[("reset_gru", False)] == 1
    assert gmnw.GMNW_POLICY_ARM_CODES_V1[("matched_mlp", False)] == 2
    assert gmnw.GMNW_POLICY_ARM_CODES_V1[("mixed_selection", False)] == 3
    assert gmnw.GMNW_POLICY_ARM_CODES_V1[("persistent_gru", True)] == 4
    assert gmnw.GMNW_POLICY_ARM_CODES_V1[("matched_mlp", True)] == 5
    assert gmnw.GMNW_POLICY_ARM_CODES_V1[("mixed_selection", True)] == 6


def test_mixed_founders_are_exact_spatially_stratified_and_pair_stable() -> None:
    rows = 64
    base = {
        "alive_mask": torch.ones((2, rows), dtype=torch.bool),
        "policy_use_gru": torch.ones((2, rows), dtype=torch.bool),
        "policy_mem_dim": torch.full((2, rows), 64, dtype=torch.int64),
        "hmem": torch.ones((2, rows, 64), dtype=torch.float32),
        "gru_upkeep_cost": (
            torch.arange(1, rows + 1, dtype=torch.float32).view(1, rows).repeat(2, 1)
        ),
        "pos_y": (
            torch.arange(rows, dtype=torch.int32).view(1, rows).repeat(2, 1) // 8
        ),
        "pos_x": (torch.arange(rows, dtype=torch.int32).view(1, rows).repeat(2, 1) % 8),
        "slot_to_agent_id": (
            torch.arange(rows, dtype=torch.int64).view(1, rows).repeat(2, 1)
        ),
    }
    configs = (
        gmnw.GmnwConfigurationV1(
            policy_arm="mixed_selection",
            gru_founder_count=8,
            task_seed=505,
        ),
        gmnw.GmnwConfigurationV1(
            policy_arm="mixed_selection",
            gru_founder_count=8,
            cue_visible_at_decision=True,
            task_seed=505,
        ),
    )

    gmnw._assign_mixed_selection_founders(base, configs)

    assert base["policy_use_gru"].sum(dim=1).tolist() == [8, 8]
    assert torch.equal(base["policy_use_gru"][0], base["policy_use_gru"][1])
    assert torch.equal(base["policy_mem_dim"][0], base["policy_mem_dim"][1])
    assert torch.count_nonzero(base["hmem"]) == 0
    assert torch.equal(
        base["gru_upkeep_cost"] > 0,
        base["policy_use_gru"],
    )
    assert torch.equal(base["gru_upkeep_cost"][0], base["gru_upkeep_cost"][1])


def test_mixed_native_path_preserves_row_family_and_selected_parent_inheritance() -> (
    None
):
    kernel = KERNELS.read_text()
    turnover = TURNOVER.read_text()
    scalar = TURNOVER_SCALAR.read_text()

    assert "kMixedSelectionArmV1" in kernel
    assert "kMixedSelectionImmediateArmV1" in kernel
    assert "mixed_selection_cell" in kernel
    assert "ordinary_birth_state.policy_use_gru[" in kernel
    assert "cell_capacity_offset + slot" in kernel
    assert "service_matched_mlp_gumbel_policy_serial_v1" in kernel
    assert "service_stable_gru_gumbel_policy_serial" in kernel
    assert (
        "gmnw_policy_arm\n                    > "
        "gmnw_controls::kMixedSelectionImmediateArmV1"
    ) in kernel
    assert "const bool child_use_gru = state.policy_use_gru[parent_slot];" in scalar
    assert "if (!execution.immutable_shared_gru_policy)" in turnover
    assert "rebound_policy_use_gru[parent_slot]" in turnover
    assert "!= rebound_policy_use_gru[child_slot]" in turnover


def test_three_cell_control_matrix_retains_order_and_independent_task_tensors(
    cpu_builder: dict[str, Any],
) -> None:
    configs = (
        gmnw.GmnwConfigurationV1(
            cue_alphabet=2,
            delay_ticks=3,
            distractors_enabled=False,
            policy_arm="persistent_gru",
            task_seed=101,
        ),
        gmnw.GmnwConfigurationV1(
            cue_alphabet=4,
            delay_ticks=5,
            distractors_enabled=True,
            reset_hidden_each_tick=True,
            policy_arm="reset_gru",
            task_seed=202,
        ),
        gmnw.GmnwConfigurationV1(
            cue_alphabet=2,
            delay_ticks=7,
            distractors_enabled=True,
            policy_arm="matched_mlp",
            task_seed=303,
        ),
    )

    resolved = gmnw._resolved_cell_configs(
        configs[0], configs, batch_cells=len(configs)
    )
    prepared = gmnw.build_prepared_direct_resident_gmnw_gru_v1(
        qualified_cuda_module=object(),
        per_cell_requests=(object(), object(), object()),
        cell_configs=configs,
        cell_ids=("persistent", "reset", "mlp"),
        cuda_device="cpu",
    )

    assert resolved == configs
    assert len(prepared.gmnw_tensors) == 27
    cue_alphabet, delay, distractors, reset, task_seed = prepared.gmnw_tensors[17:22]
    assert cue_alphabet.tolist() == [2, 4, 2]
    assert delay.tolist() == [3, 5, 7]
    assert distractors.tolist() == [False, True, True]
    assert reset.tolist() == [False, True, False]
    assert task_seed.tolist() == [101, 202, 303]

    arm, packed, scratch, status = prepared.gmnw_tensors[-4:]
    assert arm.tolist() == [0, 1, 2]
    assert arm.dtype == torch.int8
    assert arm.shape == (3,)
    assert packed is cpu_builder["matched_mlp"]
    assert packed.dtype == torch.float32
    assert packed.shape == (272_903,)
    assert scratch.dtype == torch.float32
    assert scratch.shape == (3, 1_344)
    assert status.dtype == torch.int32
    assert status.shape == (3,)
    assert prepared.gru_tensors[0].ndim == 1
    assert prepared.gru_config["immutable_shared_policy"] is True
    assert prepared.allocation_memory_receipt["shared_policy_count"] == 1
    assert prepared.patch.source.live_state["gru_upkeep_cost"].tolist() == [
        [1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ]
    assert prepared.cell_ids == ("persistent", "reset", "mlp")


def test_shared_policy_fails_closed_when_mutation_is_enabled(
    cpu_builder: dict[str, Any],
) -> None:
    cpu_builder["projection"].metadata["policy_weight_mutation_enable"] = True

    with pytest.raises(
        gmnw.CudaArchitectureBDirectResidentGmnwGruV1Error,
        match="requires policy mutation to be disabled",
    ):
        gmnw.build_prepared_direct_resident_gmnw_gru_v1(
            object(),
            qualified_cuda_module=object(),
            batch_cells=1,
            control_matrix_enabled=False,
            cuda_device="cpu",
        )


def test_runner_convenience_route_fails_closed_above_b1() -> None:
    with pytest.raises(
        gmnw.CudaArchitectureBDirectResidentGmnwGruV1Error,
        match="runner convenience route is B1",
    ):
        gmnw.build_prepared_direct_resident_gmnw_gru_v1(
            object(),
            qualified_cuda_module=object(),
            batch_cells=2,
            cuda_device="cpu",
        )


@pytest.mark.parametrize(
    ("requests", "cell_configs", "match"),
    [
        (
            tuple(object() for _ in range(73)),
            None,
            r"per_cell_requests must contain B1\.\.B72 public requests",
        ),
        (
            (object(), object(), object()),
            (gmnw.GmnwConfigurationV1(), gmnw.GmnwConfigurationV1()),
            "cell_configs length must equal batch_cells",
        ),
    ],
)
def test_public_multicell_route_fails_closed_for_b73_or_cell_count_mismatch(
    requests: tuple[object, ...],
    cell_configs: tuple[gmnw.GmnwConfigurationV1, ...] | None,
    match: str,
) -> None:
    with pytest.raises(
        gmnw.CudaArchitectureBDirectResidentGmnwGruV1Error,
        match=match,
    ):
        gmnw.build_prepared_direct_resident_gmnw_gru_v1(
            qualified_cuda_module=object(),
            per_cell_requests=requests,
            cell_configs=cell_configs,
            cuda_device="cpu",
        )


def test_public_multicell_route_delegates_ordered_sources_and_cell_ids(
    cpu_builder: dict[str, Any],
) -> None:
    requests = (object(), object(), object())
    checkpoints = tuple(
        {
            "metadata": {
                "generation": 8,
                "tick": 0,
                "native_runtime_fixed_policy_family": "gru",
                "policy_weight_mutation_enable": False,
            }
        }
        for _ in requests
    )
    cell_ids = ("cell-c", "cell-a", "cell-b")

    prepared = gmnw.build_prepared_direct_resident_gmnw_gru_v1(
        qualified_cuda_module=object(),
        per_cell_requests=requests,
        source_checkpoint_projections=checkpoints,
        cell_ids=cell_ids,
        initial_generation=8,
        gmnw_generation_base=0,
        cuda_device="cpu",
    )

    call = cpu_builder["base_calls"][-1]
    assert call == {
        "runner": None,
        "batch_cells": 3,
        "per_cell_requests": requests,
        "source_checkpoint_projections": checkpoints,
        "cell_ids": cell_ids,
    }
    assert prepared.source_projections == checkpoints
    assert prepared.cell_ids == cell_ids
    assert prepared.gmnw_config["generation_base"] == 0
    assert prepared.gmnw_config["generation_rows_per_cell"] == 12
    assert prepared.gmnw_tensors[14].shape[1] == 12


def test_legacy_b1_persistent_route_keeps_23_tensors_without_matched_mlp(
    cpu_builder: dict[str, Any],
) -> None:
    prepared = gmnw.build_prepared_direct_resident_gmnw_gru_v1(
        object(),
        qualified_cuda_module=object(),
        config=gmnw.GmnwConfigurationV1(),
        batch_cells=1,
        control_matrix_enabled=False,
        cuda_device="cpu",
    )

    assert len(prepared.gmnw_tensors) == 23
    assert prepared.qualification_trace == {}
    assert cpu_builder["loaded_paths"] == []


def test_opt_in_trace_appends_exact_suffix_without_entering_source_outputs(
    cpu_builder: dict[str, Any],
) -> None:
    prepared = gmnw.build_prepared_direct_resident_gmnw_gru_v1(
        object(),
        qualified_cuda_module=object(),
        config=gmnw.GmnwConfigurationV1(),
        batch_cells=1,
        control_matrix_enabled=False,
        cuda_device="cpu",
        qualification_output_ticks=80,
        qualification_trace_enabled=True,
    )

    assert len(prepared.gmnw_tensors) == 30
    assert tuple(prepared.qualification_trace) == (
        gmnw.GMNW_QUALIFICATION_TRACE_TENSOR_NAMES_V1
    )
    for index, name in enumerate(gmnw.GMNW_QUALIFICATION_TRACE_TENSOR_NAMES_V1):
        assert prepared.gmnw_tensors[23 + index] is prepared.qualification_trace[name]
        assert name not in prepared.patch.source.outputs
        assert name not in prepared.patch.source.live_banks
    assert prepared.qualification_trace["gmnw_trace_active_count"].shape == (1, 80)
    assert prepared.qualification_trace["gmnw_trace_hidden_before"].shape == (
        1,
        80,
        2,
        64,
    )
    assert prepared.qualification_trace["gmnw_trace_task_state_after"].shape == (
        1,
        80,
        2,
        9,
    )
    assert prepared.gmnw_config["qualification_trace_enabled"] is True
    assert prepared.gmnw_config["qualification_trace_output_ticks"] == 80


def test_opt_in_decision_tape_uses_direct_stream_episode_indexing(
    cpu_builder: dict[str, Any],
) -> None:
    prepared = gmnw.build_prepared_direct_resident_gmnw_gru_v1(
        object(),
        qualified_cuda_module=object(),
        config=gmnw.GmnwConfigurationV1(),
        batch_cells=1,
        control_matrix_enabled=False,
        cuda_device="cpu",
        qualification_decision_tape_enabled=True,
    )

    assert len(prepared.gmnw_tensors) == 26
    tape = prepared.qualification_decision_tape
    assert tuple(tape) == gmnw.GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1
    assert tape["gmnw_qualification_decision_present"].shape == (1, 3_072)
    assert tape["gmnw_qualification_decision_present"].dtype == torch.int8
    assert tape["gmnw_qualification_decision_rows"].shape == (1, 3_072, 6)
    assert torch.all(tape["gmnw_qualification_decision_rows"] == -1)
    assert tape["gmnw_qualification_decision_status"].shape == (1,)
    assert prepared.gmnw_config["qualification_decision_tape_enabled"] is True
    assert prepared.gmnw_config["control_matrix_enabled"] is False


def test_binding_accepts_legacy_control_and_versioned_trace_tensor_sets() -> None:
    source = BINDING.read_text(encoding="utf-8")

    assert "const std::size_t expected_tensor_count = 23U" in source
    assert "(has_control_tensors ? 4U : 0U)" in source
    assert "(has_trace_tensors ? 7U : 0U)" in source
    assert "(has_decision_tape_tensors ? 3U : 0U)" in source
    assert "const bool has_control_tensors" in source
    assert "const bool has_trace_tensors" in source
    assert "const bool has_decision_tape_tensors" in source
    assert "values[22].min().item<std::int32_t>()" in source
    assert "values[22].max().item<std::int32_t>()" in source
    assert "GMNW VENV consequence profiles contain an unknown code" in source
    assert "gmnw_controller::kAbiVersionV2" in source


def test_matched_mlp_birth_keeps_shared_policy_immutable_and_uses_no_tape() -> None:
    source = KERNELS.read_text(encoding="utf-8")
    start = source.index("if (!mini_policy_use_gru[0]) {")
    end = source.index("} else if (!mini_policy_use_gru[1]) {", start)
    matched_birth = source[start:end]

    assert "no policy-mutation tape is consumed" in matched_birth
    assert "mini_mlp[arguments.packed_width + column]" in matched_birth
    assert "policy_birth_delta_tape" not in matched_birth
    assert "matched_mlp_pack" not in matched_birth
    assert "mini_policy_use_gru[1] = false" in matched_birth


def test_control_scratch_retains_pre_action_rng_buffer_for_exact_replay() -> None:
    source = KERNELS.read_text(encoding="utf-8")

    assert gmnw.GMNW_MATCHED_MLP_NEURAL_SCRATCH_WIDTH_V1 == 1_024
    assert gmnw.GMNW_MATCHED_MLP_AUDIT_TICKS_V1 == 160
    assert gmnw.GMNW_MATCHED_MLP_AUDIT_WORDS_PER_TICK_V1 == 2
    assert gmnw.GMNW_MATCHED_MLP_SCRATCH_WIDTH_V1 == 1_344
    assert "rng_audit[0] = static_cast<float>(policy_rng_buffer[0])" in source
    assert "rng_audit[1] = __uint_as_float(" in source


def test_birth_planner_offsets_reproduction_scale_for_each_physical_cell() -> None:
    source = KERNELS.read_text(encoding="utf-8")

    assert (
        "runner_metric_draws,\n        reproduction_scale + cell_capacity_offset,"
        in (source)
    )
    assert "runner_metric_draws,\n        reproduction_scale," not in source


def test_cuda_damage_uses_cpu_oracle_double_then_float_contract() -> None:
    context = GMNW_CONTEXT.read_text(encoding="utf-8")
    binding = BINDING.read_text(encoding="utf-8")
    persistent = GMNW_PERSISTENT.read_text(encoding="utf-8")

    for name in ("grace", "severe", "lethal"):
        assert f"double {name}_damage_fraction;" in context
        assert (
            f"const double {name}_damage = exact_double(config, "
            f'"{name}_damage_fraction");'
        ) in binding
        assert (
            f"context.{name}_damage_fraction * static_cast<double>(start_health)"
        ) in persistent
