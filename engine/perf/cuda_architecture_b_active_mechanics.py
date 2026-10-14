from __future__ import annotations

"""Exact resident-CUDA active mechanics for the accepted Flat/Patch P2 envelope.

The surface begins after base action mutation and ends before lifecycle. It
owns the accepted CPU resident owner's rest-charge gate, fixed-MLP upkeep,
survival-step advancement, policy-energy refresh, and drive update. `age` and
injury remain unchanged: the accepted owner does not tick `age` here and the
accepted Flat/P2 profile disables injury, scar, combat, and biome mechanics.
"""

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Any, Final, Mapping

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CUDA = (
    ROOT / "engine" / "perf" / "native" / "cuda_architecture_b_active_mechanics.cu"
)
ACTIVE_MECHANICS_STATUS: Final[dict[int, str]] = {
    0: "ok",
    1: "invalid_slot_partition",
    2: "nonfinite_live_state",
    3: "invalid_drive_control",
    4: "missing_precomputed_drive_noise",
    5: "counter_overflow",
}
ACTIVE_MECHANICS_COUNT_FIELDS: Final[tuple[str, ...]] = (
    "processed_rows",
    "survival_advances",
    "rest_charged_rows",
    "rest_deficit_damage_rows",
    "drive_noise_rows",
)
ACTIVE_MECHANICS_CAPABILITY_RECEIPT: Final[dict[str, bool | str]] = {
    "schema_version": "eiel_cuda_architecture_b_active_mechanics_capability_v1",
    "decision": "ARCHB-CUDA/EXACT-FLAT-PATCH-P2-ACTIVE-MECHANICS",
    "single_cell_cuda_resident_state": True,
    "in_place_stable_input_pointers": True,
    "fail_closed_row_preflight_before_mutation": True,
    "cpu_owner_float32_operation_order": True,
    "flat_and_patch_p2_shared_mechanics": True,
    "rest_charge_and_deficit_damage": True,
    "fixed_mlp_upkeep_and_survival_steps": True,
    "policy_energy_observation_refresh": True,
    "drive_state_and_diagnostics": True,
    "precomputed_drive_noise": True,
    "accepted_flat_patch_p2_rest_charge_enabled": False,
    "age_tensor_tick": False,
    "injury_or_scar_mutation": False,
    "scar_flag_inert_when_injury_disabled": True,
    "combat_or_biome_damage": False,
    "live_rng_generation": False,
    "action_application": False,
    "chemistry_or_maintenance": False,
    "lifecycle_or_reproduction": False,
    "checkpoint_or_reporting": False,
    "gru_policy": False,
    "multi_cell_batching": False,
    "jungle_or_wdl": False,
    "throughput_qualification": False,
}

_MODULE: Any | None = None


@dataclass(frozen=True)
class CudaActiveMechanicsConfig:
    """Accepted mechanics controls copied from an owner metadata boundary."""

    world: str = "flat"
    patch_profile: str | None = None
    fixed_policy_family: str = "mlp"
    enable_rest_charge: bool = True
    rest_streak_free: int = 8
    rest_charge_energy: float = 0.25
    rest_charge_growth: float = 0.10
    rest_deficit_to_health: float = 1.0
    developmental_maturity_age: int = 20
    episode0_k: int = 20
    max_energy: float = 20.0
    start_health: float = 1.0
    drive_hunger_gain: float = 0.08
    drive_hunger_decay: float = 0.02
    drive_pain_gain: float = 0.18
    drive_pain_decay: float = 0.03
    drive_fatigue_gain: float = 0.05
    drive_fatigue_rest_relief: float = 0.10
    drive_fatigue_decay: float = 0.01
    drive_fear_gain: float = 0.22
    drive_fear_decay: float = 0.03
    enable_injury: bool = False
    enable_scar: bool = False
    enable_combat: bool = False
    enable_biomes: bool = False
    chemistry_functional_coupling_enable: bool = False
    maintenance_enable: bool = False

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "CudaActiveMechanicsConfig":
        world = str(metadata.get("world", "flat") or "flat").strip().lower()
        return cls(
            world=world,
            patch_profile=(
                str(metadata.get("native_runtime_patch_profile", "") or "") or None
            ),
            fixed_policy_family=str(
                metadata.get("native_runtime_fixed_policy_family", "mlp") or "mlp"
            ),
            enable_rest_charge=bool(metadata.get("enable_rest_charge", True)),
            rest_streak_free=int(metadata.get("rest_streak_free", 8) or 0),
            rest_charge_energy=float(metadata.get("rest_charge_energy", 0.25) or 0.0),
            rest_charge_growth=float(metadata.get("rest_charge_growth", 0.10) or 0.0),
            rest_deficit_to_health=float(
                metadata.get("rest_deficit_to_health", 1.0) or 0.0
            ),
            developmental_maturity_age=int(
                metadata.get("developmental_maturity_age", 20) or 0
            ),
            episode0_k=int(metadata.get("episode0_k", 20) or 0),
            max_energy=float(metadata.get("max_energy", 20.0) or 0.0),
            start_health=float(metadata.get("start_health", 1.0) or 0.0),
            drive_hunger_gain=float(metadata.get("drive_hunger_gain", 0.08) or 0.0),
            drive_hunger_decay=float(metadata.get("drive_hunger_decay", 0.02) or 0.0),
            drive_pain_gain=float(metadata.get("drive_pain_gain", 0.18) or 0.0),
            drive_pain_decay=float(metadata.get("drive_pain_decay", 0.03) or 0.0),
            drive_fatigue_gain=float(metadata.get("drive_fatigue_gain", 0.05) or 0.0),
            drive_fatigue_rest_relief=float(
                metadata.get("drive_fatigue_rest_relief", 0.10) or 0.0
            ),
            drive_fatigue_decay=float(metadata.get("drive_fatigue_decay", 0.01) or 0.0),
            drive_fear_gain=float(metadata.get("drive_fear_gain", 0.22) or 0.0),
            drive_fear_decay=float(metadata.get("drive_fear_decay", 0.03) or 0.0),
            enable_injury=bool(metadata.get("enable_injury", False)),
            enable_scar=bool(metadata.get("enable_scar", False)),
            enable_combat=bool(metadata.get("enable_combat", False)),
            enable_biomes=bool(metadata.get("enable_biomes", False)),
            chemistry_functional_coupling_enable=bool(
                metadata.get("chemistry_functional_coupling_enable", False)
            ),
            maintenance_enable=bool(metadata.get("maintenance_enable", False)),
        )

    def validate(self) -> None:
        if self.world not in {"flat", "patch"}:
            raise RuntimeError(
                "Architecture B active mechanics supports only Flat and accepted Patch P2"
            )
        if self.world == "patch" and self.patch_profile != "P2":
            raise RuntimeError(
                "Architecture B Patch active mechanics requires the accepted P2 profile"
            )
        if self.fixed_policy_family.strip().lower() not in {"mlp", "fixed_mlp"}:
            raise RuntimeError(
                "Architecture B active mechanics supports fixed MLP rows only; GRU is excluded"
            )
        unsupported = tuple(
            name
            for name, active in (
                ("enable_injury", self.enable_injury),
                ("enable_combat", self.enable_combat),
                ("enable_biomes", self.enable_biomes),
                (
                    "chemistry_functional_coupling_enable",
                    self.chemistry_functional_coupling_enable,
                ),
                ("maintenance_enable", self.maintenance_enable),
            )
            if active
        )
        if unsupported:
            raise RuntimeError(
                "Architecture B active mechanics excludes active accepted-envelope gates: "
                + ", ".join(unsupported)
            )
        integer_controls = {
            "rest_streak_free": self.rest_streak_free,
            "developmental_maturity_age": self.developmental_maturity_age,
            "episode0_k": self.episode0_k,
        }
        for name, value in integer_controls.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        scalar_controls = {
            "rest_charge_energy": self.rest_charge_energy,
            "rest_charge_growth": self.rest_charge_growth,
            "rest_deficit_to_health": self.rest_deficit_to_health,
            "drive_hunger_gain": self.drive_hunger_gain,
            "drive_hunger_decay": self.drive_hunger_decay,
            "drive_pain_gain": self.drive_pain_gain,
            "drive_pain_decay": self.drive_pain_decay,
            "drive_fatigue_gain": self.drive_fatigue_gain,
            "drive_fatigue_rest_relief": self.drive_fatigue_rest_relief,
            "drive_fatigue_decay": self.drive_fatigue_decay,
            "drive_fear_gain": self.drive_fear_gain,
            "drive_fear_decay": self.drive_fear_decay,
        }
        for name, value in scalar_controls.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("max_energy", self.max_energy),
            ("start_health", self.start_health),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class CudaActiveMechanicsResult:
    """Asynchronous device result; callers decide where status synchronization occurs."""

    status: torch.Tensor
    counts: torch.Tensor

    def status_code(self) -> int:
        return int(self.status.item())

    def status_name(self) -> str:
        code = self.status_code()
        return ACTIVE_MECHANICS_STATUS.get(code, f"unknown_status_{code}")

    def count_dict(self) -> dict[str, int]:
        values = [int(value) for value in self.counts.detach().cpu().tolist()]
        return dict(zip(ACTIVE_MECHANICS_COUNT_FIELDS, values, strict=True))

    def raise_for_status(self) -> None:
        code = self.status_code()
        if code != 0:
            raise RuntimeError(
                "Architecture B active mechanics failed closed: "
                f"{ACTIVE_MECHANICS_STATUS.get(code, f'unknown_status_{code}')}"
            )


def load_cuda_active_mechanics_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Architecture B active mechanics requires an available CUDA device"
        )
    cuda_source = SOURCE_CUDA.read_text(encoding="utf-8")
    module_suffix = sha256(cuda_source.encode("utf-8")).hexdigest()[:12]
    cpp_source = r"""
    #include <torch/extension.h>

    std::vector<torch::Tensor> eiel_cuda_architecture_b_active_mechanics_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor alive_mask,
        torch::Tensor actions_by_slot,
        torch::Tensor energy,
        torch::Tensor health,
        torch::Tensor injury,
        torch::Tensor age,
        torch::Tensor policy_energy_observation,
        torch::Tensor rest_streak,
        torch::Tensor survival_steps,
        torch::Tensor gru_upkeep_cost,
        torch::Tensor cost_charge_scalar,
        torch::Tensor cost_charge_maturation_scalar,
        torch::Tensor drives,
        torch::Tensor drive_inertia,
        torch::Tensor drive_noise,
        torch::Tensor prev_drive_delta,
        torch::Tensor drive_volatility,
        torch::Tensor drive_jitter,
        torch::Tensor episode0_drive_jitter_sum,
        torch::Tensor last_energy_delta,
        torch::Tensor last_health_delta,
        torch::Tensor last_injury_delta,
        torch::Tensor selfharm_events,
        torch::Tensor episode0_selfharm_events,
        torch::Tensor pre_energy,
        torch::Tensor pre_health,
        torch::Tensor pre_injury,
        torch::Tensor drive_noise_samples,
        torch::Tensor drive_noise_sample_valid,
        bool enable_rest_charge,
        int64_t rest_streak_free,
        double rest_charge_energy,
        double rest_charge_growth,
        double rest_deficit_to_health,
        int64_t developmental_maturity_age,
        int64_t episode0_k,
        double max_energy,
        double start_health,
        double drive_hunger_gain,
        double drive_hunger_decay,
        double drive_pain_gain,
        double drive_pain_decay,
        double drive_fatigue_gain,
        double drive_fatigue_rest_relief,
        double drive_fatigue_decay,
        double drive_fear_gain,
        double drive_fear_decay
    );
    """
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load_inline(
        name=f"eiel_cuda_architecture_b_active_mechanics_{module_suffix}",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=["eiel_cuda_architecture_b_active_mechanics_v1"],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def execute_active_mechanics_cuda(
    *,
    slot_to_agent_id: torch.Tensor,
    alive_mask: torch.Tensor,
    actions_by_slot: torch.Tensor,
    energy: torch.Tensor,
    health: torch.Tensor,
    injury: torch.Tensor,
    age: torch.Tensor,
    policy_energy_observation: torch.Tensor,
    rest_streak: torch.Tensor,
    survival_steps: torch.Tensor,
    gru_upkeep_cost: torch.Tensor,
    cost_charge_scalar: torch.Tensor,
    cost_charge_maturation_scalar: torch.Tensor,
    drives: torch.Tensor,
    drive_inertia: torch.Tensor,
    drive_noise: torch.Tensor,
    prev_drive_delta: torch.Tensor,
    drive_volatility: torch.Tensor,
    drive_jitter: torch.Tensor,
    episode0_drive_jitter_sum: torch.Tensor,
    last_energy_delta: torch.Tensor,
    last_health_delta: torch.Tensor,
    last_injury_delta: torch.Tensor,
    selfharm_events: torch.Tensor,
    episode0_selfharm_events: torch.Tensor,
    pre_energy: torch.Tensor,
    pre_health: torch.Tensor,
    pre_injury: torch.Tensor,
    drive_noise_samples: torch.Tensor,
    drive_noise_sample_valid: torch.Tensor,
    config: CudaActiveMechanicsConfig,
    build_directory: str | Path | None = None,
) -> CudaActiveMechanicsResult:
    """Apply one exact post-action/pre-lifecycle mechanics tick in place.

    The returned status remains on CUDA. A nonzero status guarantees that the
    kernel performed no authoritative writes. Call `raise_for_status()` only at
    a boundary where a host synchronization is acceptable.
    """

    config.validate()
    module = load_cuda_active_mechanics_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_active_mechanics_v1(
        slot_to_agent_id,
        alive_mask,
        actions_by_slot,
        energy,
        health,
        injury,
        age,
        policy_energy_observation,
        rest_streak,
        survival_steps,
        gru_upkeep_cost,
        cost_charge_scalar,
        cost_charge_maturation_scalar,
        drives,
        drive_inertia,
        drive_noise,
        prev_drive_delta,
        drive_volatility,
        drive_jitter,
        episode0_drive_jitter_sum,
        last_energy_delta,
        last_health_delta,
        last_injury_delta,
        selfharm_events,
        episode0_selfharm_events,
        pre_energy,
        pre_health,
        pre_injury,
        drive_noise_samples,
        drive_noise_sample_valid,
        bool(config.enable_rest_charge),
        int(config.rest_streak_free),
        float(config.rest_charge_energy),
        float(config.rest_charge_growth),
        float(config.rest_deficit_to_health),
        int(config.developmental_maturity_age),
        int(config.episode0_k),
        float(config.max_energy),
        float(config.start_health),
        float(config.drive_hunger_gain),
        float(config.drive_hunger_decay),
        float(config.drive_pain_gain),
        float(config.drive_pain_decay),
        float(config.drive_fatigue_gain),
        float(config.drive_fatigue_rest_relief),
        float(config.drive_fatigue_decay),
        float(config.drive_fear_gain),
        float(config.drive_fear_decay),
    )
    return CudaActiveMechanicsResult(*outputs)


__all__ = [
    "ACTIVE_MECHANICS_CAPABILITY_RECEIPT",
    "ACTIVE_MECHANICS_COUNT_FIELDS",
    "ACTIVE_MECHANICS_STATUS",
    "CudaActiveMechanicsConfig",
    "CudaActiveMechanicsResult",
    "execute_active_mechanics_cuda",
    "load_cuda_active_mechanics_extension",
]
