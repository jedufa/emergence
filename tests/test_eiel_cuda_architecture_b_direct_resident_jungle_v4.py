from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from engine.perf import cuda_architecture_b_direct_resident_jungle_v4 as jungle


EXPECTED_CONFIG_KEYS = {
    "enabled",
    "ticks_per_generation",
    "generation_base",
    "predator_pressure_enabled",
    "juvenile_vulnerability_enabled",
    "terrain_protection_enabled",
    "terrain_protection_apply_to_damage_enabled",
    "birth_emissions_enabled",
    "birth_emission_clamp_enabled",
    "ecological_recovery_enabled",
    "visibility_from_cover_enabled",
    "recovery_regrowth_boost_enabled",
    "juvenile_age_threshold",
    "scent_max",
    "waste_max",
    "density_denominator",
    "visibility_min",
    "visibility_max",
    "visibility_default",
    "predator_base_risk",
    "predator_scent_weight",
    "predator_waste_weight",
    "predator_density_weight",
    "predator_visibility_weight",
    "predator_cover_protection_weight",
    "predator_shelter_protection_weight",
    "predator_risk_min",
    "predator_risk_cap",
    "predator_health_damage",
    "juvenile_damage_multiplier",
    "cover_damage_reduction_weight",
    "shelter_damage_reduction_weight",
    "terrain_damage_multiplier_min",
    "birth_scent_emit",
    "birth_waste_emit",
    "recovery_activity_min",
    "recovery_activity_max",
    "fertility_min",
    "fertility_max",
    "recovery_decay_factor",
    "recovery_base_growth",
    "recovery_cover_weight",
    "recovery_fertility_weight",
    "recovery_waste_penalty_weight",
    "recovery_density_penalty_weight",
    "recovery_scent_penalty_weight",
    "recovery_regrowth_boost_scale",
    "recovery_regrowth_boost_max",
}


def _metadata() -> dict[str, Any]:
    values: dict[str, Any] = {
        "native_runtime_jungle_execution_enable": True,
        "native_runtime_fixed_policy_family": "mlp",
        "native_runtime_allow_runtime_jit_build": False,
        "native_runtime_patch_profile": "P2",
        "generation": 0,
        "tick": 0,
        "policy_weight_mutation_enable": False,
        "policy_weight_mutation_mode": "disabled",
        "policy_weight_mutation_rate": 0.0,
        "policy_weight_mutation_std": 0.0,
    }
    values.update({name: False for name in jungle._JUNGLE_BOOL_METADATA_FIELDS_V4})
    values.update(
        {
            "jungle_predator_pressure_enable": True,
            "jungle_reproduction_emissions_enable": True,
            "jungle_birth_emission_at_child_cell_enable": True,
            "jungle_birth_emission_clamp_enable": True,
            "jungle_juvenile_predator_vulnerability_enable": True,
            "jungle_terrain_protection_enable": True,
            "jungle_terrain_protection_apply_to_damage_enable": True,
            "jungle_ecological_recovery_enable": True,
            "jungle_recovery_regrowth_boost_enable": True,
            "patch_density_include_self": True,
            "patch_visibility_from_cover_enable": True,
        }
    )
    values.update({name: 1 for name in jungle._JUNGLE_INT_METADATA_FIELDS_V4})
    values.update({name: 0.5 for name in jungle._JUNGLE_FLOAT_METADATA_FIELDS_V4})
    values.update(
        {
            "patch_scent_max": 2.0,
            "patch_waste_max": 4.0,
            "patch_visibility_min": 0.0,
            "patch_visibility_max": 1.0,
            "patch_visibility_default": 0.75,
            "patch_fertility_min": 0.0,
            "patch_fertility_max": 1.0,
            "jungle_juvenile_age_threshold": 10,
            "jungle_juvenile_predator_damage_multiplier": 1.5,
            "jungle_juvenile_vulnerability_cap": 3.0,
            "jungle_recovery_activity_min": 0.0,
            "jungle_recovery_activity_max": 1.0,
            "jungle_recovery_activity_decay_per_step": 0.2,
        }
    )
    return values


class _CapabilityModule:
    def eiel_cuda_architecture_b_flattened_jungle_capability_v4(
        self,
    ) -> dict[str, Any]:
        return {
            "abi_version": 4,
            "world": "patch_jungle",
            "resident_state": True,
            "exact_predator_pressure": True,
            "exact_birth_emissions": True,
            "exact_ecological_recovery": True,
            "native_tick_ownership": True,
            "runtime_jit_required": False,
            "production_qualified": False,
        }


@dataclass(frozen=True)
class _GenerationConfig:
    steps: int
    generation: int = 9
    start_tick: int = 9
    max_births_per_tick: int = 9
    ordinary_births_enabled: bool = True
    generation_turnover_enabled: bool = False


@pytest.mark.parametrize(
    ("initial_generation", "use_checkpoint", "logical_generations"),
    ((0, False, 1), (0, False, 2), (0, False, 3), (0, False, 32), (7, True, 8)),
)
def test_build_reuses_patch_source_and_allocates_exact_v4_context(
    monkeypatch: pytest.MonkeyPatch,
    initial_generation: int,
    use_checkpoint: bool,
    logical_generations: int,
) -> None:
    metadata = _metadata()
    projection = SimpleNamespace(
        metadata=metadata,
        tensors={
            "patch_recovery_activity": torch.zeros((3, 4), dtype=torch.float32),
            "patch_visibility": torch.ones((3, 4), dtype=torch.float32),
        },
    )
    boundary = SimpleNamespace(tensors=projection.tensors)
    source = SimpleNamespace(
        live_state={"energy": torch.zeros((1, 6), dtype=torch.float64)},
        logical_generation_count=max(3, logical_generations),
    )
    seen: dict[str, Any] = {}

    def derive(proxy: Any) -> SimpleNamespace:
        seen["masked_flags"] = tuple(
            getattr(proxy.env_cfg, name) for name in jungle._JUNGLE_ENVELOPE_FLAGS_V4
        )
        return SimpleNamespace(
            max_births_per_tick=2,
            generation_count=max(3, logical_generations),
        )

    def build_projection(
        _runner: Any,
        capabilities: dict[str, bool],
        **_kwargs: Any,
    ) -> Any:
        seen["capabilities"] = capabilities
        seen["projection_kwargs"] = _kwargs
        return projection

    def allocate(request: Any) -> Any:
        seen["request"] = request
        return source

    monkeypatch.setattr(jungle, "derive_cuda_architecture_b_runner_envelope", derive)
    monkeypatch.setattr(jungle, "build_import_projection", build_projection)
    monkeypatch.setattr(jungle, "projection_digest", lambda _value: "digest")
    monkeypatch.setattr(
        jungle,
        "import_cpu_owner_boundary_to_cuda",
        lambda *_args, **_kwargs: boundary,
    )
    monkeypatch.setattr(
        jungle,
        "prepare_generation_birth_state_cuda",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(jungle, "allocate_direct_resident_source_v1", allocate)
    monkeypatch.setattr(
        jungle,
        "prepare_direct_resident_trajectory_v1",
        lambda actual: SimpleNamespace(source=actual, arguments=()),
    )
    monkeypatch.setattr(
        jungle,
        "CudaGenerationConfig",
        SimpleNamespace(
            from_metadata=lambda _metadata, *, steps: _GenerationConfig(steps=steps)
        ),
    )
    env_cfg = SimpleNamespace(
        **{name: True for name in jungle._JUNGLE_ENVELOPE_FLAGS_V4}
    )
    runner = SimpleNamespace(env_cfg=env_cfg, sim_cfg=SimpleNamespace())
    checkpoint_projection = {
        "metadata": {**metadata, "generation": initial_generation, "tick": 0},
        "tensors": projection.tensors,
    }
    source_projection = checkpoint_projection if use_checkpoint else None

    prepared = jungle.build_prepared_direct_resident_jungle_v4(
        runner,
        qualified_cuda_module=_CapabilityModule(),
        logical_generation_count=logical_generations,
        initial_generation=initial_generation,
        source_checkpoint_projection=source_projection,
    )

    assert seen["masked_flags"] == (False,) * 5
    assert all(getattr(env_cfg, name) for name in jungle._JUNGLE_ENVELOPE_FLAGS_V4)
    assert seen["capabilities"] == jungle.JUNGLE_CAPABILITIES_V4
    assert seen["projection_kwargs"] == {"generation": initial_generation, "tick": 0}
    assert seen["request"].logical_generation_count == max(3, logical_generations)
    assert seen["request"].envelope.generation_count == max(3, logical_generations)
    assert seen["request"].configuration.generation == initial_generation
    expected_projection = checkpoint_projection if use_checkpoint else projection
    assert seen["request"].import_projection is expected_projection
    assert tuple(tensor.shape for tensor in prepared.state_tensors) == (
        (1, 3, 4),
        (1, 3, 4),
    )
    assert tuple(tensor.shape for tensor in prepared.reporting_tensors) == (
        (1, 2, 3, 4),
        (1, logical_generations * 80, 7),
        (1, logical_generations * 80, 9),
        (1, logical_generations * 80, 7),
        (1, logical_generations * 80, 6),
    )
    assert set(prepared.config) == EXPECTED_CONFIG_KEYS
    assert prepared.config["ticks_per_generation"] == 80
    assert prepared.config["generation_base"] == initial_generation
    assert prepared.config["recovery_decay_factor"] == pytest.approx(0.8)
    assert prepared.initial_projection is expected_projection


def test_normalized_config_matches_cpu_bounding_and_dependency_semantics() -> None:
    metadata = _metadata()
    metadata.update(
        {
            "jungle_birth_emission_at_child_cell_enable": False,
            "jungle_juvenile_age_threshold": -4,
            "patch_density_radius": 0,
            "patch_density_include_self": False,
            "patch_visibility_min": 0.9,
            "patch_visibility_max": 0.1,
            "patch_visibility_default": 2.0,
            "jungle_predator_risk_min": 0.8,
            "jungle_predator_risk_cap": -2.0,
            "jungle_juvenile_predator_damage_multiplier": 5.0,
            "jungle_juvenile_vulnerability_cap": 2.0,
            "jungle_predator_health_damage": -1.0,
            "jungle_terrain_damage_multiplier_min": 3.0,
            "jungle_recovery_activity_min": 2.0,
            "jungle_recovery_activity_max": -1.0,
            "patch_fertility_min": 4.0,
            "patch_fertility_max": 1.0,
            "jungle_recovery_activity_decay_per_step": 5.0,
            "jungle_recovery_activity_cover_weight": -1.0,
        }
    )

    config = jungle._normalized_jungle_config(metadata)

    assert set(config) == EXPECTED_CONFIG_KEYS
    assert config["birth_emissions_enabled"] is False
    assert config["juvenile_age_threshold"] == 0
    assert config["density_denominator"] == 1.0
    assert (config["visibility_min"], config["visibility_max"]) == (0.1, 0.9)
    assert config["visibility_default"] == 0.9
    assert (config["predator_risk_min"], config["predator_risk_cap"]) == (0.0, 0.8)
    assert config["juvenile_damage_multiplier"] == 2.0
    assert config["predator_health_damage"] == 0.0
    assert config["terrain_damage_multiplier_min"] == 1.0
    assert (config["recovery_activity_min"], config["recovery_activity_max"]) == (
        -1.0,
        2.0,
    )
    assert (config["fertility_min"], config["fertility_max"]) == (1.0, 4.0)
    assert config["recovery_decay_factor"] == 0.0
    assert config["recovery_cover_weight"] == 0.0


def test_jungle_reporting_uses_the_exact_t160_owner_geometry() -> None:
    reporting = jungle._allocate_reporting(
        device=torch.device("cpu"),
        cells=2,
        height=3,
        width=4,
        capacity=6,
        logical_generation_count=3,
        ticks_per_generation=160,
    )

    assert tuple(tensor.shape for tensor in reporting) == (
        (2, 2, 3, 4),
        (2, 480, 7),
        (2, 480, 9),
        (2, 480, 7),
        (2, 480, 6),
    )
    assert jungle._normalized_jungle_config(
        _metadata(), ticks_per_generation=160
    )["ticks_per_generation"] == 160


class _Control:
    def __init__(self, run_id: str, owner_epoch: int) -> None:
        self.state = {
            "run_id": run_id,
            "owner_epoch": owner_epoch,
            "status": "initialized",
            "run_started": False,
            "sealed": False,
        }

    def request_cancel(self, *_args: Any) -> None: ...

    def acknowledge_cancellation_checkpoint(self, *_args: Any) -> None: ...

    def acknowledge_run_complete(self, *_args: Any) -> None: ...

    def publish_failure(self, *_args: Any) -> None: ...

    def final_result(self) -> None: ...

    def manifest(self) -> dict[str, Any]:
        return dict(self.state)


class _ExecutionModule:
    RunControlV2 = _Control

    def __init__(self, *, runtime_jit: bool = False) -> None:
        self.runtime_jit = runtime_jit
        self.direct_calls: list[tuple[Any, ...]] = []
        self.physical_calls: list[tuple[Any, ...]] = []

    def eiel_cuda_architecture_b_execute_direct_resident_jungle_v4(
        self,
        control: _Control,
        *_arguments: Any,
    ) -> dict[str, Any]:
        generations = int(_arguments[-5])
        generation_base = int(_arguments[2]["generation_base"])
        self.direct_calls.append((control, *_arguments))
        control.state.update(
            {
                "status": "generation_complete",
                "run_started": True,
                "active_generation": generation_base + generations - 1,
                "requested_generations": generations,
                "awaiting_final_output_ack": True,
            }
        )
        return {
            "logical_generation_count": generations,
            "window_count": (generations + 1) // 2,
            "physical_window_alignment": "absolute_generation_lattice_v1",
            "completed_generations": generations,
            "external_python_window_boundaries": 0,
            "direct_cpp_window_core_dispatch": True,
            "production_native_chunk_loop": True,
            "hidden_fallback": False,
            "runtime_jit": self.runtime_jit,
            "jungle_native_owner": True,
            "jungle_native_controller_abi_version": 4,
            "jungle_python_tick_or_generation_ownership": False,
        }

    def eiel_cuda_architecture_b_execute_direct_resident_jungle_physical_v4(
        self,
        *_arguments: Any,
    ) -> dict[str, Any]:
        self.physical_calls.append(_arguments)
        return {
            "hidden_fallback": False,
            "runtime_jit": False,
            "jungle_native_owner": True,
            "jungle_native_controller_abi_version": 4,
            "jungle_physical_gate_entrypoint": True,
            "jungle_python_tick_or_generation_ownership": False,
        }


def _arguments() -> tuple[Any, ...]:
    arguments: list[Any] = [object() for _ in range(160)]
    arguments[jungle.PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V4] = torch.zeros(
        (1, 160, 5), dtype=torch.int64
    )
    arguments[jungle.PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V4] = [
        torch.zeros((2, 1, 28), dtype=torch.float64)
    ]
    arguments[jungle.PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V4] = [
        torch.zeros((1, 160, 25), dtype=torch.float64)
    ]
    arguments[jungle.TICKS_ARGUMENT_INDEX_V4] = 80
    arguments[jungle.GENERATION_COUNT_ARGUMENT_INDEX_V1] = 2
    return tuple(arguments)


def _prepared(
    generations: int, *, initial_generation: int = 0
) -> jungle.PreparedDirectResidentJungleV4:
    ticks = generations * 80
    patch = SimpleNamespace(
        source=SimpleNamespace(
            logical_generation_count=generations,
            generation_base=initial_generation,
            outputs={},
        ),
        arguments=_arguments(),
        live_banks={},
        checkpoint_projection_inputs={
            "pre_generation_canonical_projection": {"initial": True},
            "resident_birth_state": object(),
        },
    )
    return jungle.PreparedDirectResidentJungleV4(
        patch=patch,
        state_tensors=(torch.zeros((1, 2, 3)), torch.ones((1, 2, 3))),
        reporting_tensors=(
            torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            torch.zeros((1, ticks, 7), dtype=torch.float64),
            torch.zeros((1, ticks, 9), dtype=torch.int64),
            torch.zeros((1, ticks, 7), dtype=torch.float64),
            torch.zeros((1, ticks, 3), dtype=torch.float32),
        ),
        config={
            "enabled": True,
            "ticks_per_generation": 80,
            "generation_base": initial_generation,
        },
        logical_generation_count=generations,
        initial_projection=SimpleNamespace(
            metadata={**_metadata(), "generation": initial_generation}
        ),
    )


@pytest.mark.parametrize("generations", (1, 2))
def test_physical_entrypoint_is_one_call_with_exact_prefix_slices(
    generations: int,
) -> None:
    module = _ExecutionModule()

    result = jungle.execute_direct_resident_jungle_physical_v4(
        module,
        _prepared(2),
        generation_count=generations,
        ticks_per_generation=40,
    )

    assert result["jungle_physical_gate_entrypoint"] is True
    assert len(module.physical_calls) == 1
    call = module.physical_calls[0]
    assert len(call) == 163
    reporting = call[1]
    assert tuple(tensor.shape[1] for tensor in reporting[1:]) == (generations * 40,) * 4
    assert call[2]["ticks_per_generation"] == 40
    assert call[2]["generation_base"] == 0
    physical_arguments = call[3:]
    assert physical_arguments[
        jungle.PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V4
    ].shape == (1, generations * 40, 5)
    assert (
        physical_arguments[jungle.PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V4][
            -1
        ].shape[0]
        == generations
    )
    assert physical_arguments[jungle.PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V4][
        0
    ].shape == (1, generations * 40, 25)
    assert physical_arguments[jungle.TICKS_ARGUMENT_INDEX_V4] == 40
    assert physical_arguments[jungle.GENERATION_COUNT_ARGUMENT_INDEX_V1] == generations


@pytest.mark.parametrize("generations", (3, 4, 7, 14, 32))
def test_resident_entrypoint_owns_all_windows_in_one_dispatch(generations: int) -> None:
    module = _ExecutionModule()

    result = jungle.execute_direct_resident_jungle_v4(
        module,
        _prepared(generations),
        logical_generation_count=generations,
        run_id=f"jungle-g{generations}",
    )

    assert result["completed_generations"] == generations
    assert result["external_python_window_boundaries"] == 0
    assert result["jungle_python_tick_or_generation_ownership"] is False
    assert len(module.direct_calls) == 1
    call = module.direct_calls[0]
    reporting = call[2]
    assert tuple(tensor.shape[1] for tensor in reporting[1:]) == (generations * 80,) * 4
    assert call[3]["ticks_per_generation"] == 80
    assert call[3]["generation_base"] == 0


def test_resident_entrypoint_uses_absolute_resume_generation() -> None:
    module = _ExecutionModule()

    result = jungle.execute_direct_resident_jungle_v4(
        module,
        _prepared(7, initial_generation=7),
        logical_generation_count=7,
        run_id="jungle-resume-g7",
    )

    assert result["window_count"] == 4
    assert result["native_run_control_v2"]["active_generation"] == 13
    assert module.direct_calls[0][3]["generation_base"] == 7


def test_resident_entrypoint_fails_closed_on_runtime_jit_receipt() -> None:
    module = _ExecutionModule(runtime_jit=True)

    with pytest.raises(
        jungle.CudaArchitectureBDirectResidentJungleV4Error,
        match="runtime_jit mismatch",
    ):
        jungle.execute_direct_resident_jungle_v4(
            module,
            _prepared(4),
            logical_generation_count=4,
            run_id="jungle-jit-rejected",
        )

    assert len(module.direct_calls) == 1


def test_capability_fails_closed_when_runtime_jit_is_required() -> None:
    module = _CapabilityModule()
    capability = module.eiel_cuda_architecture_b_flattened_jungle_capability_v4
    module.eiel_cuda_architecture_b_flattened_jungle_capability_v4 = lambda: {
        **capability(),
        "runtime_jit_required": True,
    }

    with pytest.raises(
        jungle.CudaArchitectureBDirectResidentJungleV4Error,
        match="runtime_jit_required mismatch",
    ):
        jungle._validated_capability(module)


def test_checkpoint_projects_both_jungle_state_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(4, initial_generation=7)
    captured: dict[str, Any] = {}

    def project(
        canonical: Any,
        birth_state: Any,
        outputs: dict[str, Any],
        persistent_state: dict[str, Any],
        *,
        completed_generation_count: int,
        source_generation: int,
    ) -> Any:
        captured.update(
            canonical=canonical,
            birth_state=birth_state,
            outputs=outputs,
            persistent_state=persistent_state,
            completed_generation_count=completed_generation_count,
            source_generation=source_generation,
        )
        return SimpleNamespace(
            projection={
                "metadata": {
                    "generation": source_generation + completed_generation_count
                },
                "tensors": {
                    "pred_lr": torch.zeros(1),
                    "alive_mask": torch.ones(1, dtype=torch.bool),
                },
            }
        )

    monkeypatch.setattr(
        jungle,
        "_with_materialized_event_history",
        lambda live_banks, _result: live_banks,
    )
    monkeypatch.setattr(
        jungle,
        "_materialize_disabled_prediction_state_v3",
        lambda _prepared, persistent_state: dict(persistent_state),
    )
    monkeypatch.setattr(
        jungle,
        "_with_checkpoint_contract_metadata",
        lambda projection: projection,
    )
    monkeypatch.setattr(jungle, "_project_terminal_checkpoint", project)
    result = {
        "generation_summary_scalar": torch.zeros((4, 1, 28)),
        "runtime_bookkeeping_by_tick": torch.zeros((1, 320, 5), dtype=torch.int64),
    }

    projection = jungle.project_direct_resident_jungle_checkpoint_v4(
        prepared,
        direct_result=result,
        completed_generation_count=4,
    )

    assert projection["metadata"]["generation"] == 11
    assert captured["completed_generation_count"] == 4
    assert captured["source_generation"] == 7
    assert (
        captured["persistent_state"]["patch_recovery_activity"]
        is prepared.state_tensors[0]
    )
    assert captured["persistent_state"]["patch_visibility"] is prepared.state_tensors[1]


@pytest.mark.parametrize("generations", (0, 2, 33, 64, True))
def test_resident_entrypoint_rejects_unqualified_horizon(generations: Any) -> None:
    module = _ExecutionModule()

    with pytest.raises(
        jungle.CudaArchitectureBDirectResidentJungleV4Error,
        match=r"logical_generation_count in \[3,32\]",
    ):
        jungle.execute_direct_resident_jungle_v4(
            module,
            _prepared(4),
            logical_generation_count=generations,
            run_id="jungle-invalid",
        )

    assert module.direct_calls == []
