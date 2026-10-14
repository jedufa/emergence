"""Python boundary for the native GMNW-V1 private task CPU owner."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from engine.evolution.gmnw_v1 import GmnwConfig
from engine.perf.cxxbackbone10_resident_state_owner import (
    load_resident_state_owner_extension,
)


class GmnwCpuOwnerError(RuntimeError):
    """The bounded native GMNW CPU owner contract is unavailable."""


def _config_payload(config: GmnwConfig) -> dict[str, object]:
    return asdict(config)


class GmnwCpuOwnerV1:
    def __init__(self, native_owner: Any, config: GmnwConfig) -> None:
        self._native_owner = native_owner
        self.config = config

    @classmethod
    def create(
        cls,
        *,
        capacity: int,
        config: GmnwConfig,
        build_directory: str | Path | None = None,
        native_module: Any | None = None,
    ) -> "GmnwCpuOwnerV1":
        if type(capacity) is not int or capacity < 1:
            raise GmnwCpuOwnerError("capacity must be a positive exact integer")
        module = native_module or load_resident_state_owner_extension(
            build_directory=build_directory
        )
        capability = dict(getattr(module, "EIEL_GMNW_CPU_CAPABILITIES_V1", {}))
        expected = {
            "abi_version": 1,
            "world": "GRU_MEMORY_NECESSITY_WORLD_V1",
            "private_state_resident": True,
            "exact_phase_machine": True,
            "checkpoint_roundtrip": True,
            "birth_death_slot_reuse": True,
            "consequence_lifecycle": True,
            "runtime_jit_required": False,
        }
        if capability != expected:
            raise GmnwCpuOwnerError(
                f"native GMNW CPU capability mismatch: {capability!r}"
            )
        native = module.GmnwPrivateTaskCpuOwnerV1(
            capacity,
            _config_payload(config),
        )
        return cls(native, config)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        build_directory: str | Path | None = None,
        native_module: Any | None = None,
    ) -> "GmnwCpuOwnerV1":
        metadata = dict(checkpoint.get("metadata") or {})
        config = GmnwConfig(
            cue_alphabet=metadata.get("cue_alphabet"),
            cue_ticks=metadata.get("cue_ticks"),
            delay_ticks=metadata.get("delay_ticks"),
            consequence_ticks=metadata.get("consequence_ticks"),
            distractors_enabled=metadata.get("distractors_enabled"),
            task_seed=metadata.get("task_seed"),
            grace_generation_count=metadata.get("grace_generation_count"),
            severe_generation_count=metadata.get("severe_generation_count"),
            grace_damage_fraction=metadata.get("grace_damage_fraction"),
            severe_damage_fraction=metadata.get("severe_damage_fraction"),
            lethal_damage_fraction=metadata.get("lethal_damage_fraction"),
            grace_reproduction_scale=metadata.get("grace_reproduction_scale"),
        )
        owner = cls.create(
            capacity=metadata.get("capacity"),
            config=config,
            build_directory=build_directory,
            native_module=native_module,
        )
        owner._native_owner.restore_checkpoint(dict(checkpoint))
        return owner

    def initialize_slots(
        self, slots: torch.Tensor, stream_ids: torch.Tensor
    ) -> dict[str, Any]:
        return dict(self._native_owner.initialize_slots(slots, stream_ids))

    def initialize_birth(self, *, slot: int, stream_id: int) -> dict[str, Any]:
        return dict(self._native_owner.initialize_birth(slot, stream_id))

    def clear_slots(self, slots: torch.Tensor) -> dict[str, Any]:
        return dict(self._native_owner.clear_slots(slots))

    def append_task_observation(self, base_observation: torch.Tensor) -> torch.Tensor:
        return self._native_owner.append_task_observation(base_observation)

    def step(
        self,
        actions: torch.Tensor,
        *,
        cancel_requested: bool = False,
    ) -> dict[str, Any]:
        return dict(self._native_owner.step(actions, cancel_requested))

    def step_with_consequences(
        self,
        actions: torch.Tensor,
        health: torch.Tensor,
        reproduction_scale: torch.Tensor,
        *,
        generation_index: int,
        start_health: float = 1.0,
        cancel_requested: bool = False,
    ) -> dict[str, Any]:
        return dict(
            self._native_owner.step_with_consequences(
                actions,
                health,
                reproduction_scale,
                generation_index,
                start_health,
                cancel_requested,
            )
        )

    def compact_counters(self) -> dict[str, Any]:
        return dict(self._native_owner.compact_counters())

    def evaluate_turnover(
        self,
        selected_parent_slots: torch.Tensor,
        ordinary_parent_eligible: torch.Tensor,
        reproduction_scale: torch.Tensor,
        policy_family: torch.Tensor,
    ) -> dict[str, Any]:
        return dict(
            self._native_owner.evaluate_turnover(
                selected_parent_slots,
                ordinary_parent_eligible,
                reproduction_scale,
                policy_family,
            )
        )

    def export_checkpoint(self) -> dict[str, Any]:
        checkpoint = dict(self._native_owner.export_checkpoint())
        checkpoint["metadata"] = dict(checkpoint["metadata"])
        checkpoint["tensors"] = dict(checkpoint["tensors"])
        return checkpoint

    def owner_manifest(self) -> dict[str, Any]:
        return dict(self._native_owner.owner_manifest())


__all__ = ["GmnwCpuOwnerError", "GmnwCpuOwnerV1"]
