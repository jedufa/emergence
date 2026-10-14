from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from engine.native_runtime.cuda_architecture_b_backend_factory import (
    CudaArchitectureBBackendFactoryError,
    build_cuda_architecture_b_backend,
    derive_cuda_architecture_b_production_runner_envelope,
)
from engine.native_runtime.cuda_architecture_b_routing import (
    CPP_CPU_BACKEND,
    CUDA_BACKEND,
    NO_BACKEND,
    QUALIFIED_OUTPUT_SCHEMA,
)
from env.env_config import EnvConfig
from evolution.population import PopConfig
from simulation.runner_pkg.runner_config import SimConfig


class _AdapterSpy:
    def __init__(self, backend: str, *, reason: str = "test_route") -> None:
        self.backend = backend
        self.reason = reason
        self.dispatch_started = False
        self.calls: list[dict[str, Any]] = []

    def resolve_once(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {
            "contract_status": "artifact_bound_adapter_resolved",
            "routing_integrated": True,
            "request_mode": kwargs["request_mode"],
            "resolved_backend": self.backend,
            "reason": self.reason,
            "dispatch_started": False,
            "no_hidden_fallback": True,
            "post_dispatch_fallback_allowed": False,
        }


class _ConstructorSpies:
    def __init__(self) -> None:
        self.cpu_calls: list[Any] = []
        self.cuda_calls: list[tuple[Any, Any]] = []
        self.cpu_owner = object()
        self.cuda_owner = object()
        self.cpu_error: BaseException | None = None
        self.cuda_error: BaseException | None = None

    def cpu(self, runner: Any) -> Any:
        self.cpu_calls.append(runner)
        if self.cpu_error is not None:
            raise self.cpu_error
        return self.cpu_owner

    def cuda(self, runner: Any, adapter: Any) -> Any:
        self.cuda_calls.append((runner, adapter))
        if self.cuda_error is not None:
            raise self.cuda_error
        return self.cuda_owner


def _qualified_runner(
    *,
    width: int = 16,
    height: int = 16,
    population_size: int = 64,
    steps_per_generation: int = 80,
) -> SimpleNamespace:
    sim_cfg = SimConfig(
        generations=4,
        steps_per_generation=steps_per_generation,
        device="cpu",
        bank_device="cpu",
        campaign_output_profile="compact_default",
        allow_large_output_profile=False,
        allow_generation_checkpointing=False,
        native_runtime_allow_runtime_jit_build=False,
        native_runtime_fixed_policy_family="mlp",
        checkpoint_period_gen=0,
        checkpoint_bundle_v2_period_gen=0,
        policy_deterministic_actions=False,
        shock_period=0,
        shock_duration=0,
        shock_start_prob=0.0,
        pressure_autopilot_allow_shock_enable=False,
    )
    env_cfg = EnvConfig(
        width=width,
        height=height,
        patch_world_enable=True,
        patch_world_schema_version=1,
        patch_world_observation_enable=True,
        patch_world_observation_schema_version=1,
        patch_world_observation_stacked_gather_enable=False,
        patch_world_observation_numba_gather_enable=False,
        patch_world_diagnostics_enable=True,
        patch_world_lineage_descriptors_enable=True,
        patch_world_descriptor_schema_version=1,
        patch_world_waste_consequence_enable=True,
        patch_movement_cost_apply_enable=True,
        reproduce_requires_mate=False,
        tier4_juvenile_vulnerability_enable=False,
        tier4_juvenile_steps=0,
    )
    return SimpleNamespace(
        sim_cfg=sim_cfg,
        env_cfg=env_cfg,
        pop_cfg=PopConfig(population_size=population_size),
        _policy_use_gru=[False] * population_size,
    )


def _build(
    runner: Any,
    adapter: _AdapterSpy,
    spies: _ConstructorSpies,
    **kwargs: Any,
) -> Any:
    return build_cuda_architecture_b_backend(
        runner,
        production_adapter=adapter,  # type: ignore[arg-type]
        cpu_constructor=spies.cpu,
        cuda_constructor=spies.cuda,  # type: ignore[arg-type]
        **kwargs,
    )


def test_auto_unmeasured_constructs_only_tagged_cpp_cpu_owner() -> None:
    runner = _qualified_runner()
    adapter = _AdapterSpy(CPP_CPU_BACKEND)
    spies = _ConstructorSpies()

    backend = _build(runner, adapter, spies)

    assert backend.tag == CPP_CPU_BACKEND
    assert backend.owner is spies.cpu_owner
    assert backend.envelope.grid == (16, 16)
    assert backend.envelope.population == 64
    assert backend.envelope.generation_count == 4
    assert backend.envelope.ticks_per_generation == 80
    assert backend.envelope.max_births_per_tick == 1
    assert backend.envelope.batch_cells == 1
    assert backend.envelope.checkpoint_schedule == "terminal_only"
    assert backend.envelope.output_schema == QUALIFIED_OUTPUT_SCHEMA
    assert spies.cpu_calls == [runner]
    assert spies.cuda_calls == []
    assert adapter.calls == [
        {
            "request_mode": "auto",
            "allow_cpp_cpu_fallback": True,
            "workload": backend.envelope.routing_workload(),
            "allow_explicit_cuda_exactness_override": False,
            "unsupported_mechanic_flags": {},
            "performance_decision": "unmeasured",
            "performance_decision_measured": False,
        }
    ]


def test_w32p256_runner_geometry_is_admitted_without_route_authority() -> None:
    runner = _qualified_runner(width=32, height=32, population_size=256)
    adapter = _AdapterSpy(CPP_CPU_BACKEND)
    spies = _ConstructorSpies()

    backend = _build(runner, adapter, spies)

    assert backend.envelope.grid == (32, 32)
    assert backend.envelope.population == 256
    assert adapter.calls[0]["workload"]["grid"] == [32, 32]
    assert adapter.calls[0]["workload"]["population"] == 256


def test_w32p512_runner_geometry_is_admitted_for_dense_route_delta() -> None:
    runner = _qualified_runner(width=32, height=32, population_size=512)
    adapter = _AdapterSpy(CPP_CPU_BACKEND)
    spies = _ConstructorSpies()

    backend = _build(runner, adapter, spies)

    assert backend.envelope.grid == (32, 32)
    assert backend.envelope.population == 512
    assert adapter.calls[0]["workload"]["grid"] == [32, 32]
    assert adapter.calls[0]["workload"]["population"] == 512


def test_t160_runner_ticks_are_admitted_runtime_route_data() -> None:
    runner = _qualified_runner(steps_per_generation=160)
    adapter = _AdapterSpy(CPP_CPU_BACKEND)
    spies = _ConstructorSpies()

    backend = _build(runner, adapter, spies)

    assert backend.envelope.ticks_per_generation == 160
    assert adapter.calls[0]["workload"]["ticks_per_generation"] == 160


def test_tick_native_runner_envelope_uses_absolute_control_for_workload() -> None:
    runner = _qualified_runner()
    runner.sim_cfg.ordinary_births_enabled = True
    runner.sim_cfg.population_size = 64
    runner.sim_cfg.lifecycle_mode = 1
    runner.sim_cfg.generation_turnover_enabled = False
    runner.sim_cfg.cull_fraction = 0.0
    runner.sim_cfg.parent_fraction = 1.0
    runner.sim_cfg.turnover_selection_mode = 1
    runner.sim_cfg.turnover_selection_seed = 101
    native_control = {
        "time_authority": "absolute_tick_v1",
        "total_ticks": 5000,
        "resident_owner_span_ticks": 1120,
        "native_dispatch_window_ticks": 1120,
        "refill_chunk_ticks": 160,
        "source_window_ticks": 160,
        "event_window_capacity": 65536,
        "checkpoint_ticks": [5000],
    }

    envelope = derive_cuda_architecture_b_production_runner_envelope(
        runner,
        tick_native_control=native_control,
    )
    workload = envelope.routing_workload()

    assert envelope.time_authority == "absolute_tick_v1"
    assert envelope.total_ticks == 5000
    assert envelope.generation_count == 1
    assert envelope.ticks_per_generation == 160
    assert workload["time_authority"] == "absolute_tick_v1"
    assert workload["total_ticks"] == 5000
    assert workload["checkpoint_ticks"] == [5000]
    assert not {"generation_count", "ticks_per_generation"}.intersection(workload)


def test_w16p128_geometry_is_admitted_as_explicit_turnover_profile() -> None:
    runner = _qualified_runner(width=16, height=16, population_size=128)
    adapter = _AdapterSpy(CPP_CPU_BACKEND)
    spies = _ConstructorSpies()

    backend = _build(runner, adapter, spies)

    assert backend.envelope.grid == (16, 16)
    assert backend.envelope.population == 128
    assert adapter.calls[0]["workload"]["grid"] == [16, 16]
    assert adapter.calls[0]["workload"]["population"] == 128


@pytest.mark.parametrize(
    ("width", "height", "population_size"),
    [
        (16, 16, 256),
        (32, 32, 64),
    ],
)
def test_mixed_geometry_population_pairs_construct_zero_owners(
    width: int,
    height: int,
    population_size: int,
) -> None:
    runner = _qualified_runner(
        width=width,
        height=height,
        population_size=population_size,
    )
    adapter = _AdapterSpy(CUDA_BACKEND)
    spies = _ConstructorSpies()

    with pytest.raises(
        CudaArchitectureBBackendFactoryError,
        match="width/height/population geometry must match an admitted pair",
    ):
        _build(runner, adapter, spies)

    assert adapter.calls == []
    assert spies.cpu_calls == []
    assert spies.cuda_calls == []


@pytest.mark.parametrize("generations", (3, 7, 14, 37, 128))
def test_runner_envelope_treats_generation_count_as_bounded_runtime_data(
    generations: int,
) -> None:
    runner = _qualified_runner()
    runner.sim_cfg.generations = generations
    adapter = _AdapterSpy(CPP_CPU_BACKEND)
    spies = _ConstructorSpies()

    backend = _build(runner, adapter, spies)

    assert backend.envelope.generation_count == generations


def test_hybrid_runner_envelope_admits_p192_initial_live_count() -> None:
    runner = _qualified_runner()
    runner.pop_cfg = PopConfig(population_size=192, max_slots_cap=256)
    runner.sim_cfg.ordinary_births_enabled = True
    runner.sim_cfg.population_size = 192
    runner.sim_cfg.lifecycle_mode = 1
    runner.sim_cfg.generation_turnover_enabled = False
    runner.sim_cfg.cull_fraction = 0.0
    runner.sim_cfg.parent_fraction = 1.0
    runner.sim_cfg.turnover_selection_mode = 1
    runner.sim_cfg.turnover_selection_seed = 101
    adapter = _AdapterSpy(CUDA_BACKEND)
    spies = _ConstructorSpies()

    backend = _build(
        runner,
        adapter,
        spies,
        request_mode="explicit_cuda",
        allow_explicit_cuda_exactness_override=True,
    )

    assert backend.envelope.population == 192
    assert adapter.calls[0]["workload"]["population"] == 192


def test_nonhybrid_runner_envelope_keeps_b64_population_freeze() -> None:
    runner = _qualified_runner()
    runner.pop_cfg = PopConfig(population_size=192, max_slots_cap=256)

    with pytest.raises(
        CudaArchitectureBBackendFactoryError,
        match="population_size mismatch",
    ):
        _build(runner, _AdapterSpy(CUDA_BACKEND), _ConstructorSpies())


@pytest.mark.parametrize("generations", (3, 5, 7, 128))
def test_production_runner_envelope_requires_qualified_generation_count(
    generations: int,
) -> None:
    runner = _qualified_runner()
    runner.sim_cfg.generations = generations

    with pytest.raises(
        CudaArchitectureBBackendFactoryError,
        match="production runner generation count mismatch",
    ):
        derive_cuda_architecture_b_production_runner_envelope(runner)


def test_explicit_exact_override_constructs_only_tagged_cuda_owner() -> None:
    runner = _qualified_runner()
    adapter = _AdapterSpy(CUDA_BACKEND)
    spies = _ConstructorSpies()

    backend = _build(
        runner,
        adapter,
        spies,
        request_mode="explicit_cuda",
        allow_explicit_cuda_exactness_override=True,
    )

    assert backend.tag == CUDA_BACKEND
    assert backend.owner is spies.cuda_owner
    assert spies.cpu_calls == []
    assert spies.cuda_calls == [(runner, adapter)]
    assert adapter.calls[0]["allow_cpp_cpu_fallback"] is False
    assert adapter.calls[0]["allow_explicit_cuda_exactness_override"] is True


def test_live_native_birth_limit_flows_unchanged_through_b64_route() -> None:
    runner = _qualified_runner()
    runner.sim_cfg.native_runtime_max_births_per_generation = 64
    adapter = _AdapterSpy(CUDA_BACKEND)
    spies = _ConstructorSpies()

    backend = _build(
        runner,
        adapter,
        spies,
        request_mode="explicit_cuda",
        allow_explicit_cuda_exactness_override=True,
    )

    assert backend.envelope.max_births_per_tick == 64
    assert adapter.calls[0]["workload"]["max_births_per_tick"] == 64


def test_g1_runner_requires_short_technical_gate_admission() -> None:
    runner = _qualified_runner()
    runner.sim_cfg.generations = 1

    with pytest.raises(CudaArchitectureBBackendFactoryError, match=r"\[3, 128\]"):
        _build(runner, _AdapterSpy(CUDA_BACKEND), _ConstructorSpies())

    backend = _build(
        runner,
        _AdapterSpy(CUDA_BACKEND),
        _ConstructorSpies(),
        request_mode="explicit_cuda",
        allow_explicit_cuda_exactness_override=True,
        allow_short_technical_gate_generation_count=True,
    )

    assert backend.envelope.generation_count == 1


@pytest.mark.parametrize("births", [0, 65, True, 1.0])
def test_birth_limit_outside_qualified_range_constructs_zero_owners(
    births: object,
) -> None:
    runner = _qualified_runner()
    runner.sim_cfg.native_runtime_max_births_per_generation = births
    adapter = _AdapterSpy(CUDA_BACKEND)
    spies = _ConstructorSpies()

    with pytest.raises(CudaArchitectureBBackendFactoryError, match="must be in"):
        _build(runner, adapter, spies)

    assert adapter.calls == []
    assert spies.cpu_calls == []
    assert spies.cuda_calls == []


def test_configured_cuda_ceiling_cannot_change_native_birth_semantics() -> None:
    runner = _qualified_runner()
    runner.sim_cfg.native_runtime_max_births_per_generation = 3
    runner.sim_cfg.cuda_architecture_b_max_births_per_tick = 2

    with pytest.raises(
        CudaArchitectureBBackendFactoryError,
        match="exceeds the configured Architecture B CUDA ceiling",
    ):
        _build(runner, _AdapterSpy(CUDA_BACKEND), _ConstructorSpies())


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda runner: setattr(runner.sim_cfg, "generations", 2), "generations"),
        (lambda runner: setattr(runner.env_cfg, "width", 15), "width"),
        (
            lambda runner: setattr(runner.sim_cfg, "checkpoint_period_gen", 1),
            "checkpoint_period_gen",
        ),
        (
            lambda runner: setattr(runner.sim_cfg, "campaign_output_profile", "full"),
            "campaign_output_profile",
        ),
        (
            lambda runner: setattr(
                runner.env_cfg,
                "jungle_predator_pressure_enable",
                True,
            ),
            "unsupported_active_jungle",
        ),
    ],
)
def test_nonqualified_runner_constructs_zero_owners(
    mutate: Any,
    match: str,
) -> None:
    runner = _qualified_runner()
    mutate(runner)
    adapter = _AdapterSpy(CUDA_BACKEND)
    spies = _ConstructorSpies()

    with pytest.raises(CudaArchitectureBBackendFactoryError, match=match):
        _build(runner, adapter, spies)

    assert adapter.calls == []
    assert spies.cpu_calls == []
    assert spies.cuda_calls == []


def test_unsupported_mechanic_and_rejected_route_construct_zero_owners() -> None:
    runner = _qualified_runner()
    adapter = _AdapterSpy(NO_BACKEND, reason="explicit override required")
    spies = _ConstructorSpies()

    with pytest.raises(
        CudaArchitectureBBackendFactoryError, match="unsupported mechanic"
    ):
        _build(
            runner,
            adapter,
            spies,
            unsupported_mechanic_flags={"unqualified_mechanic": True},
        )
    assert adapter.calls == []

    with pytest.raises(CudaArchitectureBBackendFactoryError, match="override required"):
        _build(
            runner,
            adapter,
            spies,
            request_mode="explicit_cuda",
        )
    assert len(adapter.calls) == 1
    assert spies.cpu_calls == []
    assert spies.cuda_calls == []


@pytest.mark.parametrize("selected_backend", [CPP_CPU_BACKEND, CUDA_BACKEND])
def test_selected_constructor_failure_has_no_post_dispatch_fallback(
    selected_backend: str,
) -> None:
    runner = _qualified_runner()
    adapter = _AdapterSpy(selected_backend)
    spies = _ConstructorSpies()
    failure = RuntimeError(f"{selected_backend} construction failed")
    if selected_backend == CPP_CPU_BACKEND:
        spies.cpu_error = failure
    else:
        spies.cuda_error = failure

    with pytest.raises(RuntimeError) as captured:
        _build(
            runner,
            adapter,
            spies,
            request_mode=(
                "auto" if selected_backend == CPP_CPU_BACKEND else "explicit_cuda"
            ),
            allow_explicit_cuda_exactness_override=(selected_backend == CUDA_BACKEND),
        )

    assert captured.value is failure
    assert len(spies.cpu_calls) == (1 if selected_backend == CPP_CPU_BACKEND else 0)
    assert len(spies.cuda_calls) == (1 if selected_backend == CUDA_BACKEND else 0)
