from __future__ import annotations

"""Standalone native ATen implementation of the Phase E modular tick."""

from hashlib import sha256
from pathlib import Path
from typing import Any, Final, MutableMapping

import torch
from torch.utils.cpp_extension import load

from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_PROJECTION_PREFIX,
    build_modular_storage_layout,
)
from perf.cuda_architecture_b_modular_tick import (
    ACTION_WIDTH,
    MODULAR_TICK_SCHEMA_VERSION,
    ModularTickError,
    ModularTickReceipt,
    ModularTickResult,
)


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "engine" / "perf" / "native"
SOURCE = NATIVE / "cuda_architecture_b_modular_tick.cpp"
BINDING = NATIVE / "cuda_architecture_b_modular_tick_bindings.cpp"
HEADER = NATIVE / "cuda_architecture_b_modular_tick.hpp"
SOURCES: Final[tuple[Path, ...]] = (SOURCE, BINDING)
HASH_SOURCES: Final[tuple[Path, ...]] = (*SOURCES, HEADER)
NATIVE_MODULAR_TICK_LANE: Final[str] = "native_aten_device_resident_modular_tick_v1"
AUDIT_TENSOR_NAMES: Final[tuple[str, ...]] = (
    "state_before",
    "state_after",
    "module_output",
    "development_effective_gain",
    "development_dependency_ready",
    "development_latch_state",
    "routed_observation",
    "previous_tick_connection_input",
    "legacy_rest_residual",
    "routed_action_residual_before_rest_combine",
    "action_residuals",
    "allocated_module_count",
    "executing_module_count",
    "executing_integrator_count",
    "executing_gated_count",
    "dormant_module_count",
    "evaluated_connection_count",
    "active_input_binding_count",
    "active_output_binding_count",
    "active_dependency_edge_count",
    "changing_state_module_count",
    "structured_agent_count",
    "neutral_agent_count",
    "structural_cost_counts",
    "structural_cost_components",
    "structural_cost_total",
    "rng_draw_count",
)

_MODULES: dict[str, Any] = {}


def _source_digest() -> str:
    digest = sha256()
    for source in HASH_SOURCES:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_native_modular_tick_extension(
    *, build_directory: str | Path | None = None, verbose: bool = False
) -> Any:
    digest = _source_digest()
    cached = _MODULES.get(digest)
    if cached is not None:
        return cached
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    module = load(
        name=f"eiel_cuda_architecture_b_modular_tick_{digest[:16]}",
        sources=[str(source) for source in SOURCES],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(NATIVE)],
        extra_cflags=["-O0", "-g0", "-ffp-contract=off"],
        with_cuda=False,
        verbose=bool(verbose),
    )
    _MODULES[digest] = module
    return module


def _ordered_planes(
    planes: MutableMapping[str, torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], str]:
    if not isinstance(planes, MutableMapping):
        raise ModularTickError("modular planes must be a mutable mapping")
    names = tuple(field.name for field in build_modular_storage_layout(1).fields)
    plain = set(names)
    prefixed = {f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}" for name in names}
    keys = {str(name) for name in planes}
    plain_present = keys & plain
    prefixed_present = keys & prefixed
    other_modular = {
        name
        for name in keys
        if name.startswith("phase_e_modular_") and name not in prefixed
    }
    if other_modular or (plain_present and prefixed_present):
        raise ModularTickError("mixed modular plane namespaces are forbidden")
    if plain_present:
        if plain_present != plain or keys != plain:
            raise ModularTickError("unprefixed modular plane set must be exact and complete")
        return tuple(planes[name] for name in names), "unprefixed"
    if prefixed_present:
        if prefixed_present != prefixed:
            raise ModularTickError("prefixed modular plane set is incomplete")
        return (
            tuple(planes[f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}"] for name in names),
            MODULAR_STORAGE_PROJECTION_PREFIX,
        )
    raise ModularTickError("complete modular plane set is required")


def execute_modular_tick_native(
    planes: MutableMapping[str, torch.Tensor],
    observations: torch.Tensor,
    fatigue: torch.Tensor,
    stage_indices: torch.Tensor,
    *,
    action_width: int = ACTION_WIDTH,
    build_directory: str | Path | None = None,
) -> ModularTickResult:
    ordered, namespace = _ordered_planes(planes)
    module = load_native_modular_tick_extension(build_directory=build_directory)
    try:
        outputs = module.eiel_cuda_architecture_b_modular_tick_v1(
            list(ordered), observations, fatigue, stage_indices, action_width
        )
    except RuntimeError as exc:
        raise ModularTickError(str(exc)) from exc
    if len(outputs) != 1 + len(AUDIT_TENSOR_NAMES):
        raise ModularTickError("native modular tick returned an invalid tensor surface")
    action_residuals = outputs[0]
    audit = dict(zip(AUDIT_TENSOR_NAMES, outputs[1:], strict=True))
    return ModularTickResult(
        planes=planes,
        action_residuals=action_residuals,
        audit_tensors=audit,
        receipt=ModularTickReceipt(
            schema_version=MODULAR_TICK_SCHEMA_VERSION,
            lane=NATIVE_MODULAR_TICK_LANE,
            field_namespace=namespace,
            plane_count=len(ordered),
            action_width=ACTION_WIDTH,
        ),
    )


__all__ = [
    "AUDIT_TENSOR_NAMES",
    "HASH_SOURCES",
    "NATIVE_MODULAR_TICK_LANE",
    "SOURCES",
    "execute_modular_tick_native",
    "load_native_modular_tick_extension",
]
