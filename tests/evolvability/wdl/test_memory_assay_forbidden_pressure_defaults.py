from __future__ import annotations

from pathlib import Path

import yaml

from simulation.runner_pkg.runner_config import SimConfig


ROOT = Path(__file__).resolve().parents[3]
REGISTRIES = (
    ROOT / "_spec/registry/v1/phaseIV_crb_m1_memory_readiness_bridge.yaml",
    ROOT / "_spec/registry/v1/wdl_cog_m1_memory_selection_probe.yaml",
    ROOT / "_spec/registry/v1/wdl_cog_m1r_lineage_aware_confirmation.yaml",
)

FORBIDDEN_DEFAULTS = {
    "sim_cfg.disable_prediction_error_selection_pressure": "true",
    "sim_cfg.episode0_fitness_weight": "0.0",
    "sim_cfg.episode0_pred_err_fitness_weight": "0.0",
    "sim_cfg.episode0_thrashing_fitness_weight": "0.0",
    "sim_cfg.episode0_drive_jitter_fitness_weight": "0.0",
    "sim_cfg.episode0_selfharm_fitness_weight": "0.0",
    "sim_cfg.late_pred_cost_enable": "false",
    "sim_cfg.late_coherence_eff_enable": "false",
    "sim_cfg.adult_move_penalty_enable": "false",
    "sim_cfg.identity_drift_enable": "false",
}


def _default_knobs(registry: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in registry["cli_defaults"]:
        if pair[0] == "--knob":
            key, _, value = str(pair[1]).partition("=")
            out[key] = value
    return out


def test_runtime_defaults_do_not_enable_late_prediction_or_coherence_pressure() -> None:
    cfg = SimConfig()

    assert cfg.late_pred_cost_enable is False
    assert cfg.late_coherence_eff_enable is False


def test_memory_selection_and_delayed_hazard_registries_freeze_forbidden_pressure() -> None:
    for registry_path in REGISTRIES:
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        defaults = _default_knobs(registry)
        for key, expected in FORBIDDEN_DEFAULTS.items():
            assert defaults[key] == expected, f"{registry_path.name} {key}"
