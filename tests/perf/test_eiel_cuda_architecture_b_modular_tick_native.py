from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load_inline

from perf.cuda_architecture_b_modular_tick import ModularTickError, execute_modular_tick
from perf.cuda_architecture_b_modular_state_abi import build_modular_storage_layout
from perf.cuda_architecture_b_modular_tick_native import (
    AUDIT_TENSOR_NAMES,
    NATIVE,
    NATIVE_MODULAR_TICK_LANE,
    SOURCE,
    execute_modular_tick_native,
    load_native_modular_tick_extension,
)
from perf.eiel_phase_e_modular_tick_fixture_v1 import (
    AGENT_COUNT,
    LOCAL_SIZE,
    TICK_AGES,
    _fixed_inputs,
)
from tests.perf.test_eiel_cuda_architecture_b_modular_tick import (
    _MUTABLE_PLANES,
    _packed_fixture,
)


@pytest.fixture(scope="session")
def native_build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    build = tmp_path_factory.mktemp("phase_e_native_modular_tick")
    load_native_modular_tick_extension(build_directory=build)
    return build


@pytest.fixture(scope="session")
def native_repair_module(tmp_path_factory: pytest.TempPathFactory):
    source = SOURCE.read_text(encoding="utf-8")
    wrappers = r"""
    torch::Tensor phase_e_test_scatter_active_rows(
        torch::Tensor rows,
        torch::Tensor slots,
        torch::Tensor active_mask,
        int64_t capacity
    ) {
        return eiel::cuda_architecture_b::scatter_active_rows_by_slot_v1(
            rows, slots, active_mask, capacity
        );
    }

    void phase_e_test_clear_modular_rows(
        const std::vector<torch::Tensor>& planes,
        torch::Tensor dead_slots,
        torch::Tensor dead_count
    ) {
        eiel::cuda_architecture_b::clear_modular_rows_v1(
            planes, dead_slots, dead_count
        );
    }

    void phase_e_test_validate_modular_planes(
        const std::vector<torch::Tensor>& planes
    ) {
        eiel::cuda_architecture_b::validate_modular_planes_v1(planes);
    }
    """
    digest = sha256((source + wrappers).encode("utf-8")).hexdigest()[:16]
    return load_inline(
        name=f"eiel_phase_e_native_repair_{digest}",
        cpp_sources=[source, wrappers],
        functions=[
            "phase_e_test_scatter_active_rows",
            "phase_e_test_clear_modular_rows",
            "phase_e_test_validate_modular_planes",
        ],
        build_directory=str(tmp_path_factory.mktemp("phase-e-native-repair")),
        extra_include_paths=[str(NATIVE)],
        extra_cflags=["-O0", "-g0", "-ffp-contract=off"],
        with_cuda=False,
    )


def _clone_planes(planes: dict[str, torch.Tensor], device: str = "cpu"):
    return {name: tensor.clone().to(device) for name, tensor in planes.items()}


def test_native_tick_matches_python_executor_exactly_for_three_cpu_ticks(
    native_build: Path,
) -> None:
    packed, _, _ = _packed_fixture()
    python_planes = _clone_planes(packed.tensors)
    native_planes = _clone_planes(packed.tensors)
    observations, _, _, _ = _fixed_inputs()

    for tick, _age in enumerate(TICK_AGES):
        obs = torch.from_numpy(observations[tick].copy())
        fatigue = torch.from_numpy(observations[tick, :, LOCAL_SIZE].copy())
        stages = torch.full((AGENT_COUNT,), tick, dtype=torch.int64)
        python_result = execute_modular_tick(
            python_planes, obs.clone(), fatigue.clone(), stages.clone()
        )
        native_result = execute_modular_tick_native(
            native_planes,
            obs.clone(),
            fatigue.clone(),
            stages.clone(),
            build_directory=native_build,
        )

        assert torch.equal(native_result.action_residuals, python_result.action_residuals)
        assert tuple(native_result.audit_tensors) == AUDIT_TENSOR_NAMES
        for name in AUDIT_TENSOR_NAMES:
            assert torch.equal(
                native_result.audit_tensors[name], python_result.audit_tensors[name]
            ), name
        for name in native_planes:
            assert torch.equal(native_planes[name], python_planes[name]), name

    assert native_result.receipt.lane == NATIVE_MODULAR_TICK_LANE
    assert native_result.receipt.rng_draws == 0
    assert native_result.receipt.device_to_host_synchronizations == 0


def test_native_tick_mutates_only_the_eight_committed_dynamic_planes(
    native_build: Path,
) -> None:
    packed, _, _ = _packed_fixture()
    planes = _clone_planes(packed.tensors)
    before = {name: tensor.clone() for name, tensor in planes.items()}
    observations, _, _, _ = _fixed_inputs()

    execute_modular_tick_native(
        planes,
        torch.from_numpy(observations[0].copy()),
        torch.from_numpy(observations[0, :, LOCAL_SIZE].copy()),
        torch.zeros((AGENT_COUNT,), dtype=torch.int64),
        build_directory=native_build,
    )

    for name in planes:
        if name not in _MUTABLE_PLANES:
            assert torch.equal(planes[name], before[name]), name


def test_native_slot_scatter_is_an_exact_linear_row_copy(
    native_repair_module,
) -> None:
    rows = torch.tensor(
        (
            (0.125, -0.25, 1.0),
            (3.5, -7.25, 9.0),
            (-11.0, 13.0, 0.03125),
            (101.0, 102.0, 103.0),
        ),
        dtype=torch.float32,
    )
    slots = torch.tensor((3, 1, 0, 0), dtype=torch.int64)
    active = torch.tensor((True, True, True, False), dtype=torch.bool)

    actual = native_repair_module.phase_e_test_scatter_active_rows(
        rows, slots, active, 4
    )
    expected = torch.zeros_like(rows)
    expected.index_copy_(0, slots[:3], rows[:3])
    assert torch.equal(actual, expected)

    duplicate_slots = torch.tensor((3, 3, 0, 0), dtype=torch.int64)
    with pytest.raises(RuntimeError, match="active slots must be unique"):
        native_repair_module.phase_e_test_scatter_active_rows(
            rows, duplicate_slots, active, 4
        )


def test_native_row_teardown_restores_all_57_abi_fill_values(
    native_repair_module,
) -> None:
    packed, _, _ = _packed_fixture()
    planes = _clone_planes(packed.tensors)
    before = _clone_planes(packed.tensors)
    layout = build_modular_storage_layout(packed.layout.agent_capacity)
    ordered = [planes[field.name] for field in layout.fields]
    native_repair_module.phase_e_test_clear_modular_rows(
        ordered,
        torch.full((packed.layout.agent_capacity,), -1, dtype=torch.int64),
        torch.zeros((1,), dtype=torch.int64),
    )
    for name in planes:
        assert torch.equal(planes[name], before[name]), name

    dead_rows = (0, packed.layout.agent_capacity - 1)
    dead_slots = torch.full(
        (packed.layout.agent_capacity,), -1, dtype=torch.int64
    )
    dead_slots[: len(dead_rows)] = torch.tensor(dead_rows, dtype=torch.int64)
    dead_count = torch.tensor([len(dead_rows)], dtype=torch.int64)

    native_repair_module.phase_e_test_clear_modular_rows(
        ordered, dead_slots, dead_count
    )

    live_rows = tuple(
        row
        for row in range(packed.layout.agent_capacity)
        if row not in dead_rows
    )
    for field in layout.fields:
        tensor = planes[field.name]
        expected_dead = torch.full_like(tensor[list(dead_rows)], field.fill_value)
        assert torch.equal(tensor[list(dead_rows)], expected_dead), field.name
        assert torch.equal(tensor[list(live_rows)], before[field.name][list(live_rows)]), (
            field.name
        )


def test_native_plane_preflight_rejects_before_mutating_runtime_planes(
    native_repair_module,
) -> None:
    packed, _, _ = _packed_fixture()
    planes = _clone_planes(packed.tensors)
    before = _clone_planes(packed.tensors)
    layout = build_modular_storage_layout(packed.layout.agent_capacity)
    agent_slot, module_slot = torch.nonzero(
        planes["module_occupied"].logical_and(
            planes["module_type_code"].eq(1)
        ),
        as_tuple=False,
    )[0].tolist()
    planes["operator_code"][agent_slot, module_slot] = 127
    before["operator_code"][agent_slot, module_slot] = 127

    with pytest.raises(RuntimeError, match="supported operator code"):
        native_repair_module.phase_e_test_validate_modular_planes(
            [planes[field.name] for field in layout.fields]
        )

    for name in planes:
        assert torch.equal(planes[name], before[name]), name


def test_phase_e_generation_preflights_before_context_and_uses_linear_scatter() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "engine/perf/native/cuda_architecture_b_generation.cpp"
    ).read_text(encoding="utf-8")
    phase_e = source.split(
        "std::vector<torch::Tensor> eiel_cuda_architecture_b_generation_phase_e_v1(",
        1,
    )[1]
    assert phase_e.index("validate_modular_planes_v1(modular_planes)") < phase_e.index(
        "auto context = eiel::cuda_architecture_b::GenerationContext::prepare"
    )
    assert "scatter_active_rows_by_slot_v1" in source
    assert "at::one_hot" not in source
    assert "torch::matmul" not in source
    assert phase_e.count("execute_post_policy_modular(tick, modular_planes)") == 1
    modular_policy = source.split(
        "void execute_single_cell_modular_policy(",
        1,
    )[1].split("GenerationPolicyBatchView policy_batch_view()", 1)[0]
    observation_handoff = "observations = std::move(modular_observations);"
    assert observation_handoff in modular_policy
    assert modular_policy.index(observation_handoff) < modular_policy.index(
        "execute_single_cell_policy_with_residual("
    )


@pytest.mark.parametrize(
    "damage",
    (
        "plane_count",
        "observation_shape",
        "fatigue_dtype",
        "stage_dtype",
        "observation_nonfinite",
        "fatigue_range",
        "stage_range",
        "operator_code",
        "development_order",
        "module_order",
        "input_code",
        "output_action",
        "connection_endpoint",
    ),
)
def test_native_tick_rejects_malformed_boundary(
    native_build: Path,
    damage: str,
) -> None:
    packed, _, _ = _packed_fixture()
    planes = _clone_planes(packed.tensors)
    observations, _, _, _ = _fixed_inputs()
    obs = torch.from_numpy(observations[0].copy())
    fatigue = torch.from_numpy(observations[0, :, LOCAL_SIZE].copy())
    stages = torch.zeros((AGENT_COUNT,), dtype=torch.int64)
    if damage == "plane_count":
        del planes["agent_id"]
    elif damage == "observation_shape":
        obs = obs[:, :-1].contiguous()
    elif damage == "fatigue_dtype":
        fatigue = fatigue.to(torch.float64)
    elif damage == "stage_dtype":
        stages = stages.to(torch.int32)
    elif damage == "observation_nonfinite":
        obs[0, 0] = torch.nan
    elif damage == "fatigue_range":
        fatigue[0] = 2.0
    elif damage == "stage_range":
        stages[0] = 3
    elif damage == "operator_code":
        planes["operator_code"][0, 0] = 127
    elif damage == "development_order":
        planes["dependency_topological_order"][0, 0] = -1
    elif damage == "module_order":
        planes["module_execution_order"][0, 0] = -1
    elif damage == "input_code":
        planes["input_semantic_code"][0, 0, 0] = 127
    elif damage == "output_action":
        planes["output_action_index"][0, 0, 0] = 127
    else:
        agent_slot, connection_slot = torch.nonzero(
            planes["connection_occupied"], as_tuple=False
        )[0].tolist()
        planes["connection_src_module_slot"][agent_slot, connection_slot] = 127

    with pytest.raises(ModularTickError):
        execute_modular_tick_native(
            planes,
            obs,
            fatigue,
            stages,
            build_directory=native_build,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_tick_cuda_matches_python_formal_tolerance(native_build: Path) -> None:
    packed, _, _ = _packed_fixture()
    python_planes = _clone_planes(packed.tensors, "cuda")
    native_planes = _clone_planes(packed.tensors, "cuda")
    observations, _, _, _ = _fixed_inputs()
    obs = torch.from_numpy(observations[0].copy()).cuda()
    fatigue = torch.from_numpy(observations[0, :, LOCAL_SIZE].copy()).cuda()
    stages = torch.zeros((AGENT_COUNT,), dtype=torch.int64, device="cuda")

    python_result = execute_modular_tick(
        python_planes, obs.clone(), fatigue.clone(), stages.clone()
    )
    native_result = execute_modular_tick_native(
        native_planes,
        obs.clone(),
        fatigue.clone(),
        stages.clone(),
        build_directory=native_build,
    )
    torch.testing.assert_close(
        native_result.action_residuals,
        python_result.action_residuals,
        atol=2e-7,
        rtol=0.0,
    )
