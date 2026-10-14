from __future__ import annotations

from pathlib import Path

import pytest
import torch

from env.gridworld_pkg.constants import ACT_REST
from perf.cuda_architecture_b_modular_lifecycle import (
    prune_and_reuse_modular_slots,
)
from perf.cuda_architecture_b_modular_state_abi import (
    modular_storage_from_projection,
)
from perf.cuda_architecture_b_modular_tick import execute_modular_tick
from perf.cuda_architecture_b_modular_tick_native import (
    execute_modular_tick_native,
    load_native_modular_tick_extension,
)
from perf.eiel_phase_e_modular_tick_fixture_v1 import (
    AGENT_COUNT,
    LOCAL_SIZE,
    TICK_AGES,
    _fixed_inputs,
)
from tests.perf.test_eiel_cuda_architecture_b_modular_capacity_epoch import (
    _migrate,
)
from tests.perf.test_eiel_cuda_architecture_b_modular_lifecycle import (
    _fixture as _lifecycle_fixture,
)
from tests.perf.test_eiel_cuda_architecture_b_modular_tick import (
    _packed_fixture,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Phase E 12-arm parity requires CUDA",
)

FORMAL_FLOAT_ATOL = 2.0e-7
RUNTIME_PLANES = (
    "module_state_present",
    "module_state",
    "module_output_present",
    "module_previous_output",
    "development_latch_present",
    "development_latch_state",
    "development_effective_gain_present",
    "development_effective_gain",
)
DEVELOPMENT_AUDIT_TENSORS = (
    "development_effective_gain",
    "development_dependency_ready",
    "development_latch_state",
    "previous_tick_connection_input",
    "evaluated_connection_count",
)
COST_AUDIT_TENSORS = (
    "structural_cost_counts",
    "structural_cost_components",
    "structural_cost_total",
)


@pytest.fixture(scope="module")
def native_build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    build = tmp_path_factory.mktemp("phase_e_12_arm_native_tick")
    load_native_modular_tick_extension(build_directory=build)
    return build


def _clone_planes(
    planes: dict[str, torch.Tensor],
    *,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone().to(device=device).contiguous()
        for name, tensor in planes.items()
    }


def _assert_formal_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
) -> None:
    observed = actual.detach().cpu()
    reference = expected.detach().cpu()
    if reference.dtype.is_floating_point:
        torch.testing.assert_close(
            observed,
            reference,
            rtol=0.0,
            atol=FORMAL_FLOAT_ATOL,
            msg=name,
        )
    else:
        assert torch.equal(observed, reference), name


def _rng_snapshot() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()


def _assert_rng_unchanged(before: tuple[torch.Tensor, torch.Tensor]) -> None:
    cpu_before, cuda_before = before
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_before)


def _tick_inputs(
    observations,
    tick: int,
    *,
    capacity: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observation = torch.from_numpy(observations[tick, :capacity].copy()).to(device)
    fatigue = torch.from_numpy(
        observations[tick, :capacity, LOCAL_SIZE].copy()
    ).to(device)
    stages = torch.full((capacity,), tick, dtype=torch.int64, device=device)
    return observation, fatigue, stages


def test_cuda_native_development_dependency_and_dormancy_reactivation_match_cpu(
    native_build: Path,
) -> None:
    cpu_packed, _, _ = _packed_fixture()
    cuda_packed, _, _ = _packed_fixture()
    cpu_planes = _clone_planes(cpu_packed.tensors)
    cuda_planes = _clone_planes(cuda_packed.tensors, device="cuda")
    observations, _, _, _ = _fixed_inputs()
    dormant_locations = torch.nonzero(
        torch.logical_and(
            cpu_planes["module_occupied"],
            cpu_planes["module_dormant"],
        ),
        as_tuple=False,
    )
    assert tuple(dormant_locations.shape) == (1, 2)
    agent_slot, module_slot = (
        int(value) for value in dormant_locations[0].tolist()
    )
    dormant_state_before = cpu_planes["module_state"][
        agent_slot, module_slot
    ].clone()
    effective_gain_by_tick: list[torch.Tensor] = []
    dependency_ready_by_tick: list[torch.Tensor] = []
    rng_before = _rng_snapshot()

    for tick, _age in enumerate(TICK_AGES):
        if tick == len(TICK_AGES) - 1:
            cpu_planes["module_dormant"][agent_slot, module_slot] = False
            cuda_planes["module_dormant"][agent_slot, module_slot] = False

        cpu_observation, cpu_fatigue, cpu_stages = _tick_inputs(
            observations,
            tick,
            capacity=AGENT_COUNT,
            device="cpu",
        )
        cuda_observation, cuda_fatigue, cuda_stages = _tick_inputs(
            observations,
            tick,
            capacity=AGENT_COUNT,
            device="cuda",
        )
        cpu_result = execute_modular_tick(
            cpu_planes,
            cpu_observation,
            cpu_fatigue,
            cpu_stages,
        )
        cuda_result = execute_modular_tick_native(
            cuda_planes,
            cuda_observation,
            cuda_fatigue,
            cuda_stages,
            build_directory=native_build,
        )

        for name in DEVELOPMENT_AUDIT_TENSORS:
            _assert_formal_parity(
                cuda_result.audit_tensors[name],
                cpu_result.audit_tensors[name],
                name=f"tick={tick} audit={name}",
            )
        for name in RUNTIME_PLANES:
            _assert_formal_parity(
                cuda_planes[name],
                cpu_planes[name],
                name=f"tick={tick} plane={name}",
            )
        _assert_formal_parity(
            cuda_result.audit_tensors["module_output"],
            cpu_result.audit_tensors["module_output"],
            name=f"tick={tick} audit=module_output",
        )

        effective_gain_by_tick.append(
            cpu_result.audit_tensors["development_effective_gain"].clone()
        )
        dependency_ready_by_tick.append(
            cpu_result.audit_tensors["development_dependency_ready"].clone()
        )
        assert cuda_result.receipt.lane == (
            "native_aten_device_resident_modular_tick_v1"
        )
        assert cuda_result.receipt.hidden_fallback_count == 0
        assert cuda_result.receipt.rng_draws == 0
        if tick < len(TICK_AGES) - 1:
            assert torch.equal(
                cpu_planes["module_state"][agent_slot, module_slot],
                dormant_state_before,
            )
            assert torch.count_nonzero(
                cpu_result.audit_tensors["module_output"][agent_slot, module_slot]
            ) == 0

    assert not torch.equal(effective_gain_by_tick[0], effective_gain_by_tick[-1])
    assert any(bool(value.any()) for value in dependency_ready_by_tick)
    assert bool(cpu_planes["development_latch_state"].any())
    assert int(cpu_result.audit_tensors["evaluated_connection_count"].sum()) > 0
    _assert_formal_parity(
        cuda_planes["dependency_topological_order"],
        cpu_planes["dependency_topological_order"],
        name="dependency topological order",
    )
    assert not torch.equal(
        cpu_planes["module_state"][agent_slot, module_slot],
        dormant_state_before,
    )
    assert torch.count_nonzero(
        cpu_result.audit_tensors["module_output"][agent_slot, module_slot]
    ) > 0
    for identity_name in (
        "agent_id",
        "agent_slot_token",
        "module_instance_id_index",
        "module_homology_id_index",
    ):
        _assert_formal_parity(
            cuda_planes[identity_name],
            cpu_planes[identity_name],
            name=f"reactivation identity={identity_name}",
        )
    _assert_rng_unchanged(rng_before)


def test_cuda_native_gated_projection_and_routing_match_cpu_without_rng_drift(
    native_build: Path,
) -> None:
    cpu_packed, _, _ = _packed_fixture()
    cuda_packed, _, _ = _packed_fixture()
    cpu_planes = _clone_planes(cpu_packed.tensors)
    cuda_planes = _clone_planes(cuda_packed.tensors, device="cuda")
    observations, _, _, _ = _fixed_inputs()

    input_positions = torch.arange(
        cpu_planes["input_enabled"].shape[-1], dtype=torch.int64
    ).reshape(1, 1, -1)
    cpu_input_mask = torch.logical_and(
        input_positions < cpu_planes["input_binding_count"].unsqueeze(2),
        cpu_planes["input_enabled"],
    )
    cuda_input_mask = torch.logical_and(
        input_positions.cuda()
        < cuda_planes["input_binding_count"].unsqueeze(2),
        cuda_planes["input_enabled"],
    )
    output_positions = torch.arange(
        cpu_planes["output_enabled"].shape[-1], dtype=torch.int64
    ).reshape(1, 1, -1)
    cpu_output_mask = torch.logical_and(
        output_positions < cpu_planes["output_binding_count"].unsqueeze(2),
        cpu_planes["output_enabled"],
    )
    cuda_output_mask = torch.logical_and(
        output_positions.cuda()
        < cuda_planes["output_binding_count"].unsqueeze(2),
        cuda_planes["output_enabled"],
    )
    assert bool(cpu_input_mask.any())
    assert bool(cpu_output_mask.any())
    assert torch.equal(cuda_input_mask.cpu(), cpu_input_mask)
    assert torch.equal(cuda_output_mask.cpu(), cpu_output_mask)

    exact_routing_and_identity_planes = (
        "agent_id",
        "agent_slot_token",
        "module_local_id_index",
        "module_instance_id_index",
        "module_homology_id_index",
        "module_execution_order",
        "operator_code",
        "input_binding_count",
        "input_semantic_code",
        "input_semantic_id_index",
        "input_enabled",
        "output_binding_count",
        "output_semantic_code",
        "output_semantic_id_index",
        "output_action_index",
        "output_enabled",
        "connection_local_id_index",
        "connection_src_module_slot",
        "connection_dst_module_slot",
        "connection_src_slot_index",
        "connection_dst_slot_index",
        "connection_execution_order",
    )
    for name in exact_routing_and_identity_planes:
        assert torch.equal(cuda_planes[name].cpu(), cpu_planes[name]), name

    cpu_inputs = _tick_inputs(observations, 0, capacity=AGENT_COUNT, device="cpu")
    cuda_inputs = _tick_inputs(observations, 0, capacity=AGENT_COUNT, device="cuda")
    rng_before = _rng_snapshot()

    cpu_result = execute_modular_tick(cpu_planes, *cpu_inputs)
    cuda_result = execute_modular_tick_native(
        cuda_planes,
        *cuda_inputs,
        build_directory=native_build,
    )

    semantic_sources = torch.stack(
        (
            cpu_inputs[0][:, LOCAL_SIZE],
            cpu_inputs[0][:, LOCAL_SIZE + 1],
        ),
        dim=1,
    )
    expanded_sources = semantic_sources.unsqueeze(1).expand(
        -1, cpu_planes["module_occupied"].shape[1], -1
    )
    routed_sources = expanded_sources.gather(
        2,
        cpu_planes["input_semantic_code"].to(dtype=torch.int64).clamp(0, 1),
    )
    input_contributions = torch.where(
        cpu_input_mask,
        cpu_planes["input_gain"] * routed_sources,
        torch.zeros_like(cpu_planes["input_gain"]),
    )
    reconstructed_routed_observation = torch.zeros_like(
        cpu_result.audit_tensors["routed_observation"]
    )
    for binding in range(input_contributions.shape[2]):
        reconstructed_routed_observation = (
            reconstructed_routed_observation + input_contributions[:, :, binding]
        )
    reconstructed_routed_observation = torch.clamp(
        reconstructed_routed_observation,
        -1.0,
        1.0,
    )
    assert int(cpu_input_mask.sum()) >= 2
    _assert_formal_parity(
        cpu_result.audit_tensors["routed_observation"],
        reconstructed_routed_observation,
        name="independently reconstructed input routing",
    )

    integrator_mask = torch.logical_and(
        cpu_planes["module_occupied"],
        torch.logical_and(
            cpu_planes["module_type_code"] == 1,
            torch.logical_and(
                cpu_planes["operator_code"] == 1,
                torch.logical_and(
                    torch.logical_not(cpu_planes["module_dormant"]),
                    cpu_planes["development_effective_gain"] > 0.0,
                ),
            ),
        ),
    )
    integrator_count = int(integrator_mask.sum())
    assert integrator_count >= 1
    assert (
        int(cpu_result.audit_tensors["executing_integrator_count"].sum())
        == integrator_count
    )
    assert torch.equal(
        cuda_result.audit_tensors["executing_integrator_count"].cpu(),
        cpu_result.audit_tensors["executing_integrator_count"],
    )
    routed_observation = cpu_result.audit_tensors["routed_observation"]
    fatigue = cpu_inputs[1].unsqueeze(1).expand_as(routed_observation)
    combined_external = torch.where(
        routed_observation == 0.0,
        fatigue,
        fatigue + routed_observation,
    )
    integrator_parameters = cpu_planes["integrator_parameters"]
    expected_integrator_state = torch.clamp(
        integrator_parameters[:, :, 1]
        * cpu_result.audit_tensors["state_before"]
        + integrator_parameters[:, :, 2] * combined_external
        + cpu_result.audit_tensors["previous_tick_connection_input"],
        -1.0,
        1.0,
    )
    expected_integrator_output = torch.clamp(
        integrator_parameters[:, :, 0]
        * integrator_parameters[:, :, 3]
        * expected_integrator_state,
        -1.0,
        1.0,
    )
    development_gain = cpu_planes["development_effective_gain"]
    expected_integrator_output = torch.where(
        development_gain == 1.0,
        expected_integrator_output,
        torch.clamp(
            development_gain * expected_integrator_output,
            -1.0,
            1.0,
        ),
    )
    _assert_formal_parity(
        cpu_result.audit_tensors["state_after"][integrator_mask],
        expected_integrator_state[integrator_mask],
        name="independently reconstructed integrator state",
    )
    _assert_formal_parity(
        cpu_result.audit_tensors["module_output"][integrator_mask],
        expected_integrator_output[integrator_mask],
        name="independently reconstructed integrator output",
    )
    for name in ("state_before", "state_after", "module_output"):
        _assert_formal_parity(
            cuda_result.audit_tensors[name].cpu()[integrator_mask],
            cpu_result.audit_tensors[name][integrator_mask],
            name=f"integrator projection={name}",
        )
    assert bool(
        torch.any(
            cpu_result.audit_tensors["state_after"][integrator_mask]
            != cpu_result.audit_tensors["state_before"][integrator_mask]
        )
    )

    gated_mask = torch.logical_and(
        cpu_planes["module_occupied"],
        torch.logical_and(
            cpu_planes["module_type_code"] == 1,
            torch.logical_and(
                cpu_planes["operator_code"] == 2,
                torch.logical_and(
                    torch.logical_not(cpu_planes["module_dormant"]),
                    cpu_planes["development_effective_gain"] > 0.0,
                ),
            ),
        ),
    )
    assert bool(gated_mask.any())
    assert int(cpu_result.audit_tensors["executing_gated_count"].sum()) == int(
        gated_mask.sum()
    )
    assert torch.equal(
        cuda_result.audit_tensors["executing_gated_count"].cpu(),
        cpu_result.audit_tensors["executing_gated_count"],
    )
    for name in ("state_before", "state_after", "module_output"):
        _assert_formal_parity(
            cuda_result.audit_tensors[name].cpu()[gated_mask],
            cpu_result.audit_tensors[name][gated_mask],
            name=f"gated projection={name}",
        )
    assert bool(
        torch.any(
            cpu_result.audit_tensors["state_after"][gated_mask]
            != cpu_result.audit_tensors["state_before"][gated_mask]
        )
    )

    for name in ("routed_observation",):
        _assert_formal_parity(
            cuda_result.audit_tensors[name],
            cpu_result.audit_tensors[name],
            name=f"input routing value={name}",
        )
    assert torch.equal(
        cuda_result.audit_tensors["active_input_binding_count"].cpu(),
        cpu_result.audit_tensors["active_input_binding_count"],
    )
    source_enabled = torch.logical_and(
        cpu_planes["module_occupied"],
        torch.logical_and(
            cpu_planes["module_type_code"] == 1,
            torch.logical_and(
                torch.logical_not(cpu_planes["module_dormant"]),
                cpu_planes["development_effective_gain"] > 0.0,
            ),
        ),
    )
    reconstructed_active_input_count = torch.sum(
        torch.logical_and(cpu_input_mask, source_enabled.unsqueeze(2)).to(
            dtype=torch.int64
        ),
        dim=(1, 2),
    )
    assert torch.equal(
        cpu_result.audit_tensors["active_input_binding_count"],
        reconstructed_active_input_count,
    )
    active_action_indexes = cpu_planes["output_action_index"][cpu_output_mask]
    assert bool(torch.all(active_action_indexes >= 0))
    assert bool(
        torch.all(active_action_indexes < cpu_result.action_residuals.shape[1])
    )

    reconstructed_legacy_rest = torch.zeros(
        (AGENT_COUNT,), dtype=torch.float32
    )
    reconstructed_routed_action = torch.zeros_like(cpu_result.action_residuals)
    non_rest_nonzero_contributions = 0
    generic_modules = torch.logical_and(
        cpu_planes["module_occupied"],
        cpu_planes["module_type_code"] == 1,
    )
    module_output = cpu_result.audit_tensors["module_output"]
    action_width = cpu_result.action_residuals.shape[1]
    for module_slot in range(generic_modules.shape[1]):
        active_module = generic_modules[:, module_slot]
        selected_output = module_output[:, module_slot]
        reconstructed_legacy_rest = reconstructed_legacy_rest + torch.where(
            active_module,
            selected_output,
            torch.zeros_like(selected_output),
        )
        for binding in range(cpu_output_mask.shape[2]):
            active_binding = torch.logical_and(
                active_module,
                cpu_output_mask[:, module_slot, binding],
            )
            action_index = cpu_planes["output_action_index"][
                :, module_slot, binding
            ].to(dtype=torch.int64)
            contribution = torch.where(
                active_binding,
                cpu_planes["output_gain"][:, module_slot, binding]
                * selected_output,
                torch.zeros_like(selected_output),
            )
            action_one_hot = torch.nn.functional.one_hot(
                action_index.clamp(0, action_width - 1),
                num_classes=action_width,
            ).to(dtype=torch.float32)
            reconstructed_routed_action = (
                reconstructed_routed_action
                + action_one_hot * contribution.unsqueeze(1)
            )
            non_rest_nonzero_contributions += int(
                torch.sum(
                    torch.logical_and(
                        active_binding,
                        torch.logical_and(
                            action_index != int(ACT_REST),
                            contribution != 0.0,
                        ),
                    )
                )
            )
    assert non_rest_nonzero_contributions >= 1
    _assert_formal_parity(
        cpu_result.audit_tensors["routed_action_residual_before_rest_combine"],
        reconstructed_routed_action,
        name="independently reconstructed output routing scatter",
    )
    reconstructed_legacy_rest = torch.clamp(
        reconstructed_legacy_rest,
        -0.25,
        0.25,
    )
    reconstructed_action_residuals = torch.clamp(
        reconstructed_routed_action,
        -0.25,
        0.25,
    )
    routed_rest = reconstructed_routed_action[:, int(ACT_REST)]
    combined_rest = torch.where(
        routed_rest == 0.0,
        reconstructed_legacy_rest,
        torch.clamp(reconstructed_legacy_rest + routed_rest, -0.25, 0.25),
    )
    reconstructed_action_residuals[:, int(ACT_REST)] = combined_rest
    _assert_formal_parity(
        cpu_result.action_residuals,
        reconstructed_action_residuals,
        name="independently reconstructed REST combine",
    )
    for name in (
        "routed_action_residual_before_rest_combine",
        "action_residuals",
    ):
        _assert_formal_parity(
            cuda_result.audit_tensors[name],
            cpu_result.audit_tensors[name],
            name=f"output routing value={name}",
        )
    _assert_formal_parity(
        cuda_result.action_residuals,
        cpu_result.action_residuals,
        name="output routing final action residual",
    )
    assert torch.equal(
        cuda_result.audit_tensors["active_output_binding_count"].cpu(),
        cpu_result.audit_tensors["active_output_binding_count"],
    )
    reconstructed_active_output_count = torch.sum(
        torch.logical_and(cpu_output_mask, source_enabled.unsqueeze(2)).to(
            dtype=torch.int64
        ),
        dim=(1, 2),
    )
    assert torch.equal(
        cpu_result.audit_tensors["active_output_binding_count"],
        reconstructed_active_output_count,
    )

    mutable_float_planes = {
        "module_state",
        "module_previous_output",
        "development_effective_gain",
    }
    for name in cpu_planes:
        if name in mutable_float_planes:
            continue
        assert torch.equal(cuda_planes[name].cpu(), cpu_planes[name]), name
    for name in mutable_float_planes:
        _assert_formal_parity(
            cuda_planes[name],
            cpu_planes[name],
            name=f"mutable resident plane={name}",
        )
    inactive_modules = torch.logical_not(cpu_planes["module_occupied"])
    for name in ("module_state", "module_previous_output"):
        assert torch.equal(
            cuda_planes[name].cpu()[inactive_modules],
            cpu_planes[name][inactive_modules],
        ), f"inactive stable state/output={name}"
    assert cuda_result.receipt.lane == "native_aten_device_resident_modular_tick_v1"
    assert cuda_result.receipt.hidden_fallback_count == 0
    assert cuda_result.receipt.rng_draws == 0
    _assert_rng_unchanged(rng_before)


def test_cuda_native_structural_cost_matches_cpu_and_remains_reporting_only(
    native_build: Path,
) -> None:
    baseline_packed, _, _ = _packed_fixture()
    altered_packed, _, _ = _packed_fixture()
    cpu_planes = _clone_planes(baseline_packed.tensors)
    cuda_baseline = _clone_planes(baseline_packed.tensors, device="cuda")
    cuda_altered = _clone_planes(altered_packed.tensors, device="cuda")
    occupied = cuda_altered["agent_occupied"]
    cuda_altered["structural_cost_counts"][occupied] = (
        cuda_altered["structural_cost_counts"][occupied] + 7
    )
    cuda_altered["structural_cost_components"][occupied] = (
        cuda_altered["structural_cost_components"][occupied] + 0.5
    )
    cuda_altered["structural_cost_total"][occupied] = (
        cuda_altered["structural_cost_total"][occupied] + 5.0
    )
    observations, _, _, _ = _fixed_inputs()
    cpu_inputs = _tick_inputs(observations, 0, capacity=AGENT_COUNT, device="cpu")
    cuda_inputs = _tick_inputs(observations, 0, capacity=AGENT_COUNT, device="cuda")
    rng_before = _rng_snapshot()

    cpu_result = execute_modular_tick(cpu_planes, *cpu_inputs)
    cuda_result = execute_modular_tick_native(
        cuda_baseline,
        *cuda_inputs,
        build_directory=native_build,
    )
    altered_result = execute_modular_tick_native(
        cuda_altered,
        *(tensor.clone() for tensor in cuda_inputs),
        build_directory=native_build,
    )

    for name in COST_AUDIT_TENSORS:
        assert torch.equal(
            cuda_result.audit_tensors[name].cpu(),
            cpu_result.audit_tensors[name],
        ), name
        assert torch.equal(
            altered_result.audit_tensors[name],
            cuda_altered[name],
        ), f"altered cost telemetry={name}"
    _assert_formal_parity(
        altered_result.action_residuals,
        cuda_result.action_residuals,
        name="structural cost reporting-only action residual",
    )
    for name in RUNTIME_PLANES:
        _assert_formal_parity(
            cuda_altered[name],
            cuda_baseline[name],
            name=f"structural cost reporting-only plane={name}",
        )
    assert cuda_result.receipt.hidden_fallback_count == 0
    assert altered_result.receipt.hidden_fallback_count == 0
    assert cuda_result.receipt.rng_draws == altered_result.receipt.rng_draws == 0
    _assert_rng_unchanged(rng_before)


def test_cuda_prune_and_slot_reuse_matches_cpu_exactly_without_rng_drift() -> None:
    _, cpu_planes, cpu_owner, cpu_replacement, cpu_slot = _lifecycle_fixture()
    _, cuda_planes, cuda_owner, cuda_replacement, cuda_slot = _lifecycle_fixture()
    cuda_planes = _clone_planes(cuda_planes, device="cuda")
    cuda_owner = _clone_planes(cuda_owner, device="cuda")
    cuda_replacement = _clone_planes(cuda_replacement, device="cuda")
    assert cuda_slot == cpu_slot
    rng_before = _rng_snapshot()

    cpu_result = prune_and_reuse_modular_slots(
        cpu_planes,
        cpu_owner,
        profile_id="b2-12x32-v1",
        agent_rows=torch.tensor([0], dtype=torch.int64),
        module_slots=torch.tensor([cpu_slot], dtype=torch.int64),
        replacement=cpu_replacement,
        cpu_validation_mode=True,
    )
    cuda_result = prune_and_reuse_modular_slots(
        cuda_planes,
        cuda_owner,
        profile_id="b2-12x32-v1",
        agent_rows=torch.tensor([0], dtype=torch.int64, device="cuda"),
        module_slots=torch.tensor([cuda_slot], dtype=torch.int64, device="cuda"),
        replacement=cuda_replacement,
    )

    assert tuple(cuda_result.planes) == tuple(cpu_result.planes)
    for name in cpu_result.planes:
        assert torch.equal(
            cuda_result.planes[name].cpu(),
            cpu_result.planes[name],
        ), name
    assert tuple(cuda_result.audit_tensors) == tuple(cpu_result.audit_tensors)
    for name in cpu_result.audit_tensors:
        assert torch.equal(
            cuda_result.audit_tensors[name].cpu(),
            cpu_result.audit_tensors[name],
        ), name
    assert torch.equal(
        cuda_result.planes["agent_id"].cpu(),
        cpu_result.planes["agent_id"],
    )
    assert torch.equal(
        cuda_result.planes["agent_slot_token"].cpu(),
        cpu_result.planes["agent_slot_token"],
    )
    assert cuda_result.receipt.device.startswith("cuda")
    assert cuda_result.receipt.device_to_host_synchronizations == 0
    _assert_rng_unchanged(rng_before)


def test_cuda_capacity_migration_first_continuation_matches_cpu_without_rng_drift(
    native_build: Path,
) -> None:
    _, _, _, _, migration = _migrate()
    migrated = modular_storage_from_projection(
        migration.metadata,
        migration.tensors,
    )
    assert migrated is not None
    assert migrated.layout.profile_id == "b2-16x48-v1"
    assert migrated.layout.physical_module_slots == 16
    assert migrated.layout.physical_connection_slots == 48
    assert migration.receipt.old_occupied_subspace_exact is True
    assert migration.receipt.newly_legal_tail_canonical_empty is True
    assert migration.receipt.migration_rng_draws == 0

    cpu_planes = _clone_planes(migrated.tensors)
    cuda_planes = _clone_planes(migrated.tensors, device="cuda")
    capacity = migrated.layout.agent_capacity
    observations, _, _, _ = _fixed_inputs()
    cpu_inputs = _tick_inputs(observations, 0, capacity=capacity, device="cpu")
    cuda_inputs = _tick_inputs(observations, 0, capacity=capacity, device="cuda")
    rng_before = _rng_snapshot()

    cpu_result = execute_modular_tick(cpu_planes, *cpu_inputs)
    cuda_result = execute_modular_tick_native(
        cuda_planes,
        *cuda_inputs,
        build_directory=native_build,
    )

    _assert_formal_parity(
        cuda_result.action_residuals,
        cpu_result.action_residuals,
        name="first expanded action residual",
    )
    for name in cpu_planes:
        _assert_formal_parity(
            cuda_planes[name],
            cpu_planes[name],
            name=f"first expanded plane={name}",
        )
    for name in cuda_result.audit_tensors:
        _assert_formal_parity(
            cuda_result.audit_tensors[name],
            cpu_result.audit_tensors[name],
            name=f"first expanded audit={name}",
        )
    assert cuda_result.receipt.lane == "native_aten_device_resident_modular_tick_v1"
    assert cuda_result.receipt.hidden_fallback_count == 0
    _assert_rng_unchanged(rng_before)
