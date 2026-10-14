"""Pre-owner backend factory for the exact Patch P2 CUDA envelope."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from .cuda_architecture_b_production_adapter import (
    CudaArchitectureBProductionAdapter,
)
from .cuda_architecture_b_routing import (
    CPP_CPU_BACKEND,
    CUDA_BACKEND,
    QUALIFIED_OUTPUT_SCHEMA,
)
from .patch_profile import PATCH_PROFILE_P2, classify_patch_architecture_b_profile
from .patch_profile import classify_patch_profile


QUALIFIED_POPULATION: Final[int] = 64
QUALIFIED_HYBRID_INITIAL_LIVE_COUNTS: Final[frozenset[int]] = frozenset({64, 192})
QUALIFIED_GRID: Final[tuple[int, int]] = (16, 16)
QUALIFIED_GEOMETRIES: Final[frozenset[tuple[int, tuple[int, int]]]] = frozenset(
    {
        (64, (16, 16)),
        (128, (16, 16)),
        (256, (32, 32)),
        (512, (32, 32)),
    }
)
QUALIFIED_GENERATION_COUNT: Final[int] = 4
MIN_RESIDENT_GENERATION_COUNT: Final[int] = 3
MAX_RESIDENT_GENERATION_COUNT: Final[int] = 128
QUALIFIED_TICKS_PER_GENERATION: Final[int] = 80
ADMITTED_TICKS_PER_GENERATION: Final[tuple[int, ...]] = (80, 160)
QUALIFIED_MIN_BIRTHS_PER_TICK: Final[int] = 1
QUALIFIED_MAX_BIRTHS_PER_TICK: Final[int] = 64
QUALIFIED_BATCH_CELLS: Final[int] = 1
QUALIFIED_POLICY_FAMILY: Final[str] = "stochastic_fixed_mlp"
QUALIFIED_GRU_POLICY_FAMILY: Final[str] = "stochastic_stable_gru"
QUALIFIED_CHECKPOINT_SCHEDULE: Final[str] = "terminal_only"
QUALIFIED_CAMPAIGN_OUTPUT_PROFILE: Final[str] = "compact_default"


class CudaArchitectureBBackendFactoryError(RuntimeError):
    """The live runner or resolved route cannot select exactly one owner."""


@dataclass(frozen=True, slots=True)
class CudaArchitectureBRunnerEnvelope:
    """Runner-derived portion of the one qualified production envelope."""

    world: str
    profile: str
    population: int
    grid: tuple[int, int]
    policy_family: str
    generation_count: int
    ticks_per_generation: int
    max_births_per_tick: int
    batch_cells: int
    checkpoint_schedule: str
    output_schema: str
    unsupported_mechanic_flags: tuple[tuple[str, bool], ...]
    # The generation-shaped members above are retained for the existing CUDA
    # owner ABI.  A native route binds the active contract below and never
    # uses those members as horizon or residency authority.
    time_authority: str = "legacy_generation_v1"
    total_ticks: int | None = None
    resident_owner_span_ticks: int | None = None
    native_dispatch_window_ticks: int | None = None
    refill_chunk_ticks: int | None = None
    source_window_ticks: int | None = None
    event_window_capacity: int | None = None
    checkpoint_ticks: tuple[int, ...] = ()

    def routing_workload(self) -> dict[str, Any]:
        common = {
            "world": self.world,
            "profile": self.profile,
            "population": self.population,
            "grid": list(self.grid),
            "policy_family": self.policy_family,
            "max_births_per_tick": self.max_births_per_tick,
            "batch_cells": self.batch_cells,
            "checkpoint_schedule": self.checkpoint_schedule,
            "output_schema": self.output_schema,
        }
        if self.time_authority == "absolute_tick_v1":
            return {
                **common,
                "time_authority": self.time_authority,
                "total_ticks": self.total_ticks,
                "resident_owner_span_ticks": self.resident_owner_span_ticks,
                "native_dispatch_window_ticks": self.native_dispatch_window_ticks,
                "source_window_ticks": self.source_window_ticks,
                "event_window_capacity": self.event_window_capacity,
                "checkpoint_ticks": list(self.checkpoint_ticks),
            }
        return {
            **common,
            "generation_count": self.generation_count,
            "ticks_per_generation": self.ticks_per_generation,
        }


@dataclass(frozen=True, slots=True)
class CudaArchitectureBBackend:
    """One constructed backend tagged with its immutable pre-owner decision."""

    tag: str
    owner: Any
    envelope: CudaArchitectureBRunnerEnvelope
    route_receipt: dict[str, Any]

    @property
    def backend_tag(self) -> str:
        return self.tag

    @property
    def resolved_backend(self) -> str:
        return self.tag


_MISSING: Final[object] = object()


def _fail(message: str) -> CudaArchitectureBBackendFactoryError:
    return CudaArchitectureBBackendFactoryError(message)


def _require_exact_field(
    value: Any,
    name: str,
    expected: Any,
    *,
    source: str,
) -> Any:
    actual = getattr(value, name, _MISSING)
    if actual is _MISSING:
        raise _fail(f"qualified runner is missing {source}.{name}")
    if type(actual) is not type(expected) or actual != expected:
        raise _fail(
            f"qualified runner {source}.{name} mismatch: "
            f"expected={expected!r} actual={actual!r}"
        )
    return actual


def _require_initial_live_count(pop_cfg: Any, sim_cfg: Any) -> int:
    actual = getattr(pop_cfg, "population_size", _MISSING)
    if actual is _MISSING:
        raise _fail("qualified runner is missing pop_cfg.population_size")
    if type(actual) is not int:
        raise _fail(
            "qualified runner pop_cfg.population_size must be an exact integer"
        )
    lifecycle_mode = getattr(sim_cfg, "lifecycle_mode", _MISSING)
    generation_turnover = getattr(sim_cfg, "generation_turnover_enabled", _MISSING)
    hybrid_lifecycle = lifecycle_mode == 1 and generation_turnover is False
    if hybrid_lifecycle:
        if actual not in QUALIFIED_HYBRID_INITIAL_LIVE_COUNTS:
            raise _fail(
                "qualified hybrid runner pop_cfg.population_size must be 64 or 192; "
                f"actual={actual!r}"
            )
        return actual
    if actual != QUALIFIED_POPULATION:
        raise _fail(
            f"qualified runner pop_cfg.population_size mismatch: "
            f"expected={QUALIFIED_POPULATION!r} actual={actual!r}"
        )
    return actual


def _require_admitted_ticks_per_generation(value: Any, *, source: str) -> int:
    actual = getattr(value, "steps_per_generation", _MISSING)
    if (
        type(actual) is not int
        or actual not in ADMITTED_TICKS_PER_GENERATION
    ):
        raise _fail(
            f"qualified runner {source}.steps_per_generation must be one of "
            f"{list(ADMITTED_TICKS_PER_GENERATION)!r}, actual={actual!r}"
        )
    return actual


def _require_exact_int_field(value: Any, name: str, *, source: str) -> int:
    actual = getattr(value, name, _MISSING)
    if actual is _MISSING:
        raise _fail(f"qualified runner is missing {source}.{name}")
    if type(actual) is not int:
        raise _fail(
            f"qualified runner {source}.{name} must be an exact integer, "
            f"actual={actual!r}"
        )
    return actual


def _require_birth_bound(sim_cfg: Any) -> int:
    births = getattr(sim_cfg, "native_runtime_max_births_per_generation", _MISSING)
    if type(births) is not int or not (
        QUALIFIED_MIN_BIRTHS_PER_TICK <= births <= QUALIFIED_MAX_BIRTHS_PER_TICK
    ):
        raise _fail(
            "qualified runner sim_cfg.native_runtime_max_births_per_generation "
            f"must be in [{QUALIFIED_MIN_BIRTHS_PER_TICK}, "
            f"{QUALIFIED_MAX_BIRTHS_PER_TICK}], actual={births!r}"
        )
    configured_ceiling = getattr(
        sim_cfg, "cuda_architecture_b_max_births_per_tick", _MISSING
    )
    if type(configured_ceiling) is not int or not (
        QUALIFIED_MIN_BIRTHS_PER_TICK
        <= configured_ceiling
        <= QUALIFIED_MAX_BIRTHS_PER_TICK
    ):
        raise _fail(
            "qualified runner sim_cfg.cuda_architecture_b_max_births_per_tick "
            f"must be in [{QUALIFIED_MIN_BIRTHS_PER_TICK}, "
            f"{QUALIFIED_MAX_BIRTHS_PER_TICK}], actual={configured_ceiling!r}"
        )
    if births > configured_ceiling:
        raise _fail(
            "native birth limit exceeds the configured Architecture B CUDA ceiling: "
            f"births={births} ceiling={configured_ceiling}"
        )
    return births


def _require_native_birth_bound(sim_cfg: Any) -> int:
    """Read the tick-native per-tick birth ceiling without a generation axis."""

    births = getattr(sim_cfg, "cuda_architecture_b_max_births_per_tick", _MISSING)
    if type(births) is not int or not (
        QUALIFIED_MIN_BIRTHS_PER_TICK <= births <= QUALIFIED_MAX_BIRTHS_PER_TICK
    ):
        raise _fail(
            "qualified native runner sim_cfg.cuda_architecture_b_max_births_per_tick "
            f"must be in [{QUALIFIED_MIN_BIRTHS_PER_TICK}, "
            f"{QUALIFIED_MAX_BIRTHS_PER_TICK}], actual={births!r}"
        )
    configured_ceiling = getattr(
        sim_cfg, "cuda_architecture_b_max_births_per_tick", _MISSING
    )
    if births > configured_ceiling:
        raise _fail(
            "native birth limit exceeds the configured Architecture B CUDA ceiling: "
            f"births={births} ceiling={configured_ceiling}"
        )
    return births


def _validated_unsupported_flags(
    value: Mapping[str, bool] | None,
) -> tuple[tuple[str, bool], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise _fail("unsupported_mechanic_flags must be a mapping")
    rows: list[tuple[str, bool]] = []
    for name, enabled in sorted(value.items(), key=lambda item: str(item[0])):
        if type(name) is not str or not name.strip() or type(enabled) is not bool:
            raise _fail("unsupported_mechanic_flags must contain named booleans")
        if enabled:
            raise _fail(f"qualified runner enables unsupported mechanic {name!r}")
        rows.append((name, enabled))
    return tuple(rows)


def derive_cuda_architecture_b_runner_envelope(
    runner: Any,
    *,
    policy_family: str = "mlp",
    allow_short_technical_gate_generation_count: bool = False,
    unsupported_mechanic_flags: Mapping[str, bool] | None = None,
    tick_native_control: Mapping[str, Any] | None = None,
) -> CudaArchitectureBRunnerEnvelope:
    """Derive the runner-owned Patch envelope without granting qualification.

    Legacy callers retain a generation-shaped owner ABI.  When
    ``tick_native_control`` is supplied, absolute ticks bind the active
    horizon/residency contract and any generation-shaped values returned for
    the old owner ABI are bounded compatibility dimensions only.  The
    production artifact registry still decides whether any particular request
    is qualified before dispatch.
    """

    sim_cfg = getattr(runner, "sim_cfg", _MISSING)
    env_cfg = getattr(runner, "env_cfg", _MISSING)
    pop_cfg = getattr(runner, "pop_cfg", _MISSING)
    if sim_cfg is _MISSING or env_cfg is _MISSING or pop_cfg is _MISSING:
        raise _fail("factory requires a fully initialized live runner")

    population = _require_exact_int_field(
        pop_cfg,
        "population_size",
        source="pop_cfg",
    )
    width = _require_exact_int_field(env_cfg, "width", source="env_cfg")
    height = _require_exact_int_field(env_cfg, "height", source="env_cfg")
    grid = (width, height)
    hybrid_geometry = (
        population == 192
        and grid == QUALIFIED_GRID
        and getattr(sim_cfg, "lifecycle_mode", _MISSING) == 1
        and getattr(sim_cfg, "generation_turnover_enabled", _MISSING) is False
    )
    if (population, grid) not in QUALIFIED_GEOMETRIES and not hybrid_geometry:
        if population == 192 and grid == QUALIFIED_GRID:
            raise _fail(
                "qualified runner pop_cfg.population_size mismatch: "
                f"expected={QUALIFIED_POPULATION!r} actual={population!r}"
            )
        admitted = ", ".join(
            f"w{geometry[0]}x{geometry[1]}p{population_size}"
            for population_size, geometry in sorted(QUALIFIED_GEOMETRIES)
        )
        raise _fail(
            "qualified runner width/height/population geometry must match an admitted "
            f"pair ({admitted}), actual=w{width}x{height}p{population}"
        )
    native_control = None
    if tick_native_control is not None:
        from engine.campaigns.tick_native_control import TickNativeControlV1

        try:
            native_control = TickNativeControlV1.from_execution(
                {"tick_native_control": dict(tick_native_control)}
            )
        except (TypeError, ValueError) as exc:
            raise _fail(f"tick-native runner control is invalid: {exc}") from exc
        if getattr(sim_cfg, "generation_turnover_enabled", _MISSING) is not False:
            raise _fail(
                "generation turnover cannot be active on tick-native route"
            )
        source_window = native_control.source_window_ticks
        compatibility_chunks = max(
            1,
            (
                native_control.native_dispatch_window_ticks
                + source_window
                - 1
            )
            // source_window,
        )
        if compatibility_chunks > MAX_RESIDENT_GENERATION_COUNT:
            raise _fail("tick-native compatibility chunk count exceeds owner ABI limit")
        generations = compatibility_chunks
        ticks = source_window
        max_births_per_tick = _require_native_birth_bound(sim_cfg)
    else:
        generations = getattr(sim_cfg, "generations", _MISSING)
        minimum_generations = (
            1
            if allow_short_technical_gate_generation_count
            else MIN_RESIDENT_GENERATION_COUNT
        )
        if (
            type(generations) is not int
            or generations < minimum_generations
            or generations > MAX_RESIDENT_GENERATION_COUNT
        ):
            raise _fail(
                "qualified runner sim_cfg.generations must be an exact integer in "
                f"[{minimum_generations}, {MAX_RESIDENT_GENERATION_COUNT}], "
                f"actual={generations!r}"
            )
        ticks = _require_admitted_ticks_per_generation(sim_cfg, source="sim_cfg")
        max_births_per_tick = _require_birth_bound(sim_cfg)
    if policy_family not in {"mlp", "gru"}:
        raise _fail("policy_family must be exactly mlp or gru")
    _require_exact_field(
        sim_cfg,
        "native_runtime_fixed_policy_family",
        policy_family,
        source="sim_cfg",
    )
    _require_exact_field(
        sim_cfg,
        "policy_deterministic_actions",
        False,
        source="sim_cfg",
    )
    _require_exact_field(
        sim_cfg,
        "campaign_output_profile",
        QUALIFIED_CAMPAIGN_OUTPUT_PROFILE,
        source="sim_cfg",
    )
    _require_exact_field(
        sim_cfg,
        "allow_large_output_profile",
        False,
        source="sim_cfg",
    )
    if native_control is None:
        _require_exact_field(
            sim_cfg,
            "allow_generation_checkpointing",
            False,
            source="sim_cfg",
        )
        _require_exact_field(
            sim_cfg,
            "checkpoint_period_gen",
            0,
            source="sim_cfg",
        )
        _require_exact_field(
            sim_cfg,
            "checkpoint_bundle_v2_period_gen",
            0,
            source="sim_cfg",
        )

    if policy_family == "mlp":
        classification = classify_patch_architecture_b_profile(
            sim_cfg,
            env_cfg,
            runner,
            world="patch",
        )
    else:
        classification = classify_patch_profile(
            sim_cfg,
            env_cfg,
            runner,
            world="patch",
            accepted_policy_families=("gru",),
        )
    if not classification.supported or classification.profile != PATCH_PROFILE_P2:
        reasons = ",".join(classification.unsupported_reasons) or "profile_mismatch"
        raise _fail(f"runner is outside the exact Patch P2 envelope: {reasons}")

    flags = _validated_unsupported_flags(unsupported_mechanic_flags)
    return CudaArchitectureBRunnerEnvelope(
        world="patch",
        profile=PATCH_PROFILE_P2,
        population=population,
        grid=grid,
        policy_family=(
            QUALIFIED_POLICY_FAMILY
            if policy_family == "mlp"
            else QUALIFIED_GRU_POLICY_FAMILY
        ),
        generation_count=generations,
        ticks_per_generation=ticks,
        max_births_per_tick=max_births_per_tick,
        batch_cells=QUALIFIED_BATCH_CELLS,
        checkpoint_schedule=QUALIFIED_CHECKPOINT_SCHEDULE,
        output_schema=QUALIFIED_OUTPUT_SCHEMA,
        unsupported_mechanic_flags=flags,
        time_authority=(
            "absolute_tick_v1" if native_control is not None else "legacy_generation_v1"
        ),
        total_ticks=(native_control.total_ticks if native_control is not None else None),
        resident_owner_span_ticks=(
            native_control.resident_owner_span_ticks
            if native_control is not None
            else None
        ),
        native_dispatch_window_ticks=(
            native_control.native_dispatch_window_ticks
            if native_control is not None
            else None
        ),
        refill_chunk_ticks=None,
        source_window_ticks=(
            native_control.source_window_ticks if native_control is not None else None
        ),
        event_window_capacity=(
            native_control.event_window_capacity if native_control is not None else None
        ),
        checkpoint_ticks=(
            native_control.resolved_checkpoint_ticks
            if native_control is not None
            else ()
        ),
    )


def derive_cuda_architecture_b_production_runner_envelope(
    runner: Any,
    *,
    unsupported_mechanic_flags: Mapping[str, bool] | None = None,
    tick_native_control: Mapping[str, Any] | None = None,
) -> CudaArchitectureBRunnerEnvelope:
    """Require the legacy owner count, or validate the native control contract."""

    envelope = derive_cuda_architecture_b_runner_envelope(
        runner,
        unsupported_mechanic_flags=unsupported_mechanic_flags,
        tick_native_control=tick_native_control,
    )
    if tick_native_control is None and envelope.generation_count != QUALIFIED_GENERATION_COUNT:
        raise _fail(
            "production runner generation count mismatch: "
            f"expected={QUALIFIED_GENERATION_COUNT!r} "
            f"actual={envelope.generation_count!r}"
        )
    return envelope


def _validated_route_receipt(
    value: Any,
    *,
    request_mode: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("production adapter returned a non-mapping route receipt")
    receipt = dict(value)
    exact_fields = {
        "contract_status": "artifact_bound_adapter_resolved",
        "routing_integrated": True,
        "request_mode": request_mode,
        "dispatch_started": False,
        "no_hidden_fallback": True,
        "post_dispatch_fallback_allowed": False,
    }
    for name, expected in exact_fields.items():
        actual = receipt.get(name, _MISSING)
        if type(actual) is not type(expected) or actual != expected:
            raise _fail(
                f"production route receipt {name} mismatch: "
                f"expected={expected!r} actual={actual!r}"
            )
    return receipt


def build_cuda_architecture_b_backend(
    runner: Any,
    *,
    production_adapter: CudaArchitectureBProductionAdapter,
    cpu_constructor: Callable[[Any], Any],
    cuda_constructor: Callable[[Any, CudaArchitectureBProductionAdapter], Any],
    request_mode: str = "auto",
    allow_explicit_cuda_exactness_override: bool = False,
    allow_short_technical_gate_generation_count: bool = False,
    unsupported_mechanic_flags: Mapping[str, bool] | None = None,
    performance_decision: str = "unmeasured",
    performance_decision_measured: bool = False,
) -> CudaArchitectureBBackend:
    """Resolve before construction, then construct exactly one selected owner.

    An owner-constructor exception is terminal. The factory deliberately has no
    retry or alternate-constructor path after the route has selected a backend.
    """

    if not callable(cpu_constructor) or not callable(cuda_constructor):
        raise _fail("CPU and CUDA owner constructors must both be callable")
    resolve_once = getattr(production_adapter, "resolve_once", None)
    if not callable(resolve_once):
        raise _fail("production_adapter must provide resolve_once")
    if getattr(production_adapter, "dispatch_started", _MISSING) is not False:
        raise _fail("backend selection must precede CUDA dispatch")

    envelope = derive_cuda_architecture_b_runner_envelope(
        runner,
        allow_short_technical_gate_generation_count=(
            allow_short_technical_gate_generation_count
        ),
        unsupported_mechanic_flags=unsupported_mechanic_flags,
    )
    route = _validated_route_receipt(
        resolve_once(
            request_mode=request_mode,
            allow_cpp_cpu_fallback=request_mode == "auto",
            workload=envelope.routing_workload(),
            allow_explicit_cuda_exactness_override=(
                allow_explicit_cuda_exactness_override
            ),
            unsupported_mechanic_flags=dict(envelope.unsupported_mechanic_flags),
            performance_decision=performance_decision,
            performance_decision_measured=performance_decision_measured,
        ),
        request_mode=request_mode,
    )
    backend = route.get("resolved_backend")
    if backend == CPP_CPU_BACKEND:
        owner = cpu_constructor(runner)
    elif backend == CUDA_BACKEND:
        owner = cuda_constructor(runner, production_adapter)
    else:
        reason = route.get("reason", "route_rejected")
        raise _fail(f"backend route rejected before owner construction: {reason}")
    return CudaArchitectureBBackend(
        tag=backend,
        owner=owner,
        envelope=envelope,
        route_receipt=route,
    )


__all__ = [
    "QUALIFIED_BATCH_CELLS",
    "QUALIFIED_CAMPAIGN_OUTPUT_PROFILE",
    "QUALIFIED_CHECKPOINT_SCHEDULE",
    "QUALIFIED_GENERATION_COUNT",
    "QUALIFIED_GEOMETRIES",
    "QUALIFIED_GRID",
    "QUALIFIED_MAX_BIRTHS_PER_TICK",
    "QUALIFIED_MIN_BIRTHS_PER_TICK",
    "QUALIFIED_POLICY_FAMILY",
    "QUALIFIED_POPULATION",
    "QUALIFIED_TICKS_PER_GENERATION",
    "CudaArchitectureBBackend",
    "CudaArchitectureBBackendFactoryError",
    "CudaArchitectureBRunnerEnvelope",
    "build_cuda_architecture_b_backend",
    "derive_cuda_architecture_b_production_runner_envelope",
    "derive_cuda_architecture_b_runner_envelope",
]
