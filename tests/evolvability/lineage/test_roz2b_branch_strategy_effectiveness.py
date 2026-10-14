from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = PROJECT_ROOT / "_spec/registry/v1/roz2b_branch_strategy_effectiveness.yaml"
SCRIPT = PROJECT_ROOT / "tools/bin/roz2b_branch_strategy_effectiveness_audit.py"
SPEC = importlib.util.spec_from_file_location("roz2b_branch_strategy_effectiveness_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
roz2b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(roz2b)


def _registry() -> dict:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))


def _knobs(flags: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    idx = 0
    while idx < len(flags):
        if flags[idx] == "--knob":
            key, value = flags[idx + 1].split("=", 1)
            out[key] = value
            idx += 2
        else:
            idx += 1
    return out


def test_roz2b_registry_matrix_is_explicit_and_complete() -> None:
    registry = _registry()

    assert registry["matrix_order"] == ["setting", "init_arm", "seed"]
    assert registry["seeds"] == [101, 202, 303, 404, 505]
    assert len(registry["axes"]["setting"]["values"]) == 9
    assert len(registry["axes"]["init_arm"]["values"]) == 2
    assert len(registry["axes"]["setting"]["values"]) * len(registry["axes"]["init_arm"]["values"]) * len(registry["seeds"]) == 90
    assert registry["naming"]["run_id_format"] == "{name}__{setting}__init_{init_arm}__seed{seed:06d}"


def test_roz2b_true_flat_active_is_explicit_noop_and_matches_reporting_flags() -> None:
    flags = _registry()["axes"]["setting"]["flags"]
    reporting = _knobs(flags["arm_reporting_only__context_true_flat_noop"])
    active = _knobs(flags["arm_active_roz2a__context_true_flat_noop"])

    assert active == reporting
    assert active["sim_cfg.path2_roz_reporting_enable"] == "true"
    assert active["env_cfg.roz2a_patch_niche_enable"] == "false"
    assert active["env_cfg.patch_world_observation_enable"] == "false"
    assert active["env_cfg.patch_movement_cost_apply_enable"] == "false"
    assert active["env_cfg.patch_world_waste_consequence_enable"] == "false"
    assert active["env_cfg.jungle_ecological_recovery_enable"] == "false"


def test_roz2b_active_niche_separates_mechanics_without_direct_reproduction_knobs() -> None:
    flags = _registry()["axes"]["setting"]["flags"]
    reporting = _knobs(flags["arm_reporting_only__context_roz2a_niche_context"])
    active = _knobs(flags["arm_active_roz2a__context_roz2a_niche_context"])

    assert reporting["env_cfg.roz2a_patch_niche_enable"] == "false"
    assert active["env_cfg.roz2a_patch_niche_enable"] == "true"
    assert active["env_cfg.patch_movement_cost_apply_enable"] == "true"
    assert active["env_cfg.patch_world_waste_consequence_enable"] == "true"
    assert active["env_cfg.jungle_ecological_recovery_enable"] == "true"
    forbidden = {key for key in active if "reproduce" in key or "birth" in key}
    assert forbidden == set()


def test_roz2b_run_id_parser_recovers_arm_context_init_seed() -> None:
    parsed = roz2b._parse_run_id(
        "roz2b_branch_strategy_effectiveness__arm_active_roz2a__context_roz2a_niche_context__init_founder_balanced__seed000505"
    )

    assert parsed == {
        "arm": "active_roz2a",
        "context": "roz2a_niche_context",
        "init_arm": "founder_balanced",
        "seed": "505",
    }


def test_roz2b_specialization_jsd_is_deterministic_and_bounded() -> None:
    left = {"zone_a": 1.0}
    right = {"zone_b": 1.0}

    assert roz2b._jsd(left, right) == roz2b._jsd(right, left)
    assert 0.0 <= roz2b._jsd(left, right) <= 1.0
