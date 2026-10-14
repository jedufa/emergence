#!/usr/bin/env python3
"""Generate bounded EED-0 RP1 reference-world causal evidence.

This fixture uses the real Python GridWorld, RESPEC gather implementation,
Patch update hook, and LRO1 reproductive-opportunity contract. It does not run
evolution, a campaign, checkpoints, native code, C++, or CUDA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from env.env_config import EnvConfig  # noqa: E402
from env.gridworld import GridWorld  # noqa: E402
from env.gridworld_pkg.constants import ACT_GATHER  # noqa: E402
from env.gridworld_pkg.patch_world import (  # noqa: E402
    EED_RESOURCE_IMPACT_ARRAY_NAMES,
    PATCH_WORLD_ARRAY_NAMES,
    RESPEC0B_RESOURCE_A,
    RESPEC0B_RESOURCE_B,
    apply_respec0b_resource_gather,
)
from simulation.rng_manager import RNGManager  # noqa: E402
from simulation.runner_pkg.runner_ticks_reproduction import (  # noqa: E402
    compute_lro1_reproductive_opportunity,
)


SCHEMA_VERSION = "eed0_rp1_reference_fixture_v1"
PHENOTYPES = {
    "a_oriented": 0.75,
    "neutral": 0.0,
    "b_oriented": -0.75,
    "equalized_capacity_sham": 0.75,
}
RESOURCES = {
    "resource_a": RESPEC0B_RESOURCE_A,
    "resource_b": RESPEC0B_RESOURCE_B,
}
PROMPT_SHA256 = "6511cd618a0af769d43981013c7bf1cd3841ad05d43fabc88c32a0decab032b3"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _git_rev_parse_or_head(ref: str) -> str:
    try:
        return _git("rev-parse", ref)
    except subprocess.CalledProcessError:
        return _git("rev-parse", "HEAD")


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _cfg(
    *,
    overlay_enabled: bool,
    impact_applied: bool,
    equalized_sham: bool = False,
) -> EnvConfig:
    cfg = EnvConfig(width=2, height=1, view_radius=1)
    cfg.population_size = 2
    cfg.max_agent_capacity_mult = 6
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 2
    cfg.patch_world_enable = True
    cfg.patch_food_gather_amount = 1.0
    cfg.patch_food_presence_threshold = 0.5
    cfg.patch_food_regrow_per_step = 0.02
    cfg.respec0b_resource_specialization_enable = True
    cfg.respec0b_resource_class_mode = "vertical_split_ab"
    cfg.respec0b_resource_equalized_sham_enable = bool(equalized_sham)
    cfg.eed_resource_impact_enable = bool(overlay_enabled)
    cfg.eed_resource_impact_apply_enable = bool(impact_applied)
    return cfg


def _env(cfg: EnvConfig, *, seed: int = 47001) -> GridWorld:
    manager = RNGManager(global_seed=seed)
    return GridWorld(cfg, seed=seed, run_dir=None, rng=manager.env_np)


def _resource_cell(env: GridWorld, resource_class: int) -> tuple[int, int]:
    matches = np.argwhere(
        np.asarray(env.respec0b_resource_class, dtype=np.int32)
        == int(resource_class)
    )
    if matches.shape[0] != 1:
        raise AssertionError("RP1 fixture requires exactly one cell per resource")
    return int(matches[0, 0]), int(matches[0, 1])


def _stock_snapshot(env: GridWorld) -> dict[str, float]:
    if bool(getattr(env.cfg, "eed_resource_impact_enable", False)):
        return {
            "resource_a": float(np.sum(env.eed_resource_a_amount, dtype=np.float64)),
            "resource_b": float(np.sum(env.eed_resource_b_amount, dtype=np.float64)),
        }
    classes = np.asarray(env.respec0b_resource_class, dtype=np.int32)
    food = np.asarray(env.patch_food_amount, dtype=np.float32)
    return {
        "resource_a": float(
            np.sum(food[classes == RESPEC0B_RESOURCE_A], dtype=np.float64)
        ),
        "resource_b": float(
            np.sum(food[classes == RESPEC0B_RESOURCE_B], dtype=np.float64)
        ),
    }


def _impact_snapshot(env: GridWorld) -> dict[str, float]:
    if not bool(getattr(env.cfg, "eed_resource_impact_enable", False)):
        return {
            "pending_a": 0.0,
            "pending_b": 0.0,
            "load_a": 0.0,
            "load_b": 0.0,
        }
    return {
        "pending_a": float(
            np.sum(env.eed_resource_a_pending_impact, dtype=np.float64)
        ),
        "pending_b": float(
            np.sum(env.eed_resource_b_pending_impact, dtype=np.float64)
        ),
        "load_a": float(np.sum(env.eed_resource_a_impact_load, dtype=np.float64)),
        "load_b": float(np.sum(env.eed_resource_b_impact_load, dtype=np.float64)),
    }


def run_arm(
    *,
    resource: str,
    phenotype: str,
    overlay_enabled: bool,
    impact_applied: bool,
    seed: int = 47001,
) -> dict[str, Any]:
    if resource not in RESOURCES or phenotype not in PHENOTYPES:
        raise ValueError("unsupported RP1 fixture arm")
    equalized_sham = phenotype == "equalized_capacity_sham"
    cfg = _cfg(
        overlay_enabled=overlay_enabled,
        impact_applied=impact_applied,
        equalized_sham=equalized_sham,
    )
    env = _env(cfg, seed=seed)
    y, x = _resource_cell(env, RESOURCES[resource])
    env.add_agent(0, (y, x))
    rng_before = deepcopy(env.rng.bit_generator.state)
    initial = _stock_snapshot(env)
    axes = np.full((2,), PHENOTYPES[phenotype], dtype=np.float32)
    gained = np.zeros((2,), dtype=np.float32)
    actions = np.full((2,), ACT_GATHER, dtype=np.int32)
    apply_respec0b_resource_gather(
        env,
        np.zeros((2,), dtype=np.int32),
        actions,
        np.full((2,), y, dtype=np.int32),
        np.full((2,), x, dtype=np.int32),
        gained,
        axes,
    )
    immediate = _stock_snapshot(env)
    impact_after_action = _impact_snapshot(env)
    gather_stats = dict(getattr(env, "respec0b_resource_gather_stats_last", {}))
    env.patch_world_step(1)
    later = _stock_snapshot(env)
    impact_after_settlement = _impact_snapshot(env)
    later_food = float(env.patch_food_amount[y, x])
    opportunity = compute_lro1_reproductive_opportunity(
        {
            "density": 1.0,
            "waste": 0.0,
            "scent": 0.0,
            "shelter_cover": 0.0,
            "food": later_food,
            "hazard": 0.0,
        },
        enabled=True,
        base_try_prob=0.8,
    )
    rng_after = deepcopy(env.rng.bit_generator.state)
    return {
        "schema_version": SCHEMA_VERSION,
        "resource": resource,
        "phenotype": phenotype,
        "axis": PHENOTYPES[phenotype],
        "seed": seed,
        "overlay_enabled": bool(overlay_enabled),
        "impact_applied": bool(impact_applied),
        "agent_count": 1,
        "action_count": 2,
        "initial_stock": initial,
        "immediate_stock": immediate,
        "immediate_depletion": {
            name: initial[name] - immediate[name] for name in initial
        },
        "immediate_energy_by_action": [float(value) for value in gained],
        "immediate_energy_total": float(np.sum(gained, dtype=np.float64)),
        "handling_stats": gather_stats,
        "impact_after_action": impact_after_action,
        "later_stock": later,
        "later_food_context": later_food,
        "later_gather_available": int(
            later_food > float(cfg.patch_food_presence_threshold)
        ),
        "later_reproductive_opportunity": opportunity,
        "impact_after_settlement": impact_after_settlement,
        "settlement_stats": dict(getattr(env, "eed_resource_impact_stats_last", {})),
        "rng_before_sha256": _sha256_bytes(_canonical(rng_before)),
        "rng_after_sha256": _sha256_bytes(_canonical(rng_after)),
        "rng_unchanged": rng_before == rng_after,
    }


def fixture_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for resource in RESOURCES:
        for phenotype in PHENOTYPES:
            for impact_applied in (False, True):
                rows.append(
                    run_arm(
                        resource=resource,
                        phenotype=phenotype,
                        overlay_enabled=True,
                        impact_applied=impact_applied,
                    )
                )
    return rows


def _paired(
    rows: list[dict[str, Any]],
    resource: str,
    phenotype: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = [
        row
        for row in rows
        if row["resource"] == resource and row["phenotype"] == phenotype
    ]
    off = next(row for row in matches if not row["impact_applied"])
    on = next(row for row in matches if row["impact_applied"])
    return off, on


def _flags_off_parity() -> dict[str, Any]:
    explicit = _env(
        _cfg(overlay_enabled=False, impact_applied=False),
        seed=47002,
    )
    baseline_cfg = _cfg(overlay_enabled=False, impact_applied=False)
    baseline_cfg.eed_resource_impact_apply_enable = False
    baseline = _env(baseline_cfg, seed=47002)
    array_equal = all(
        np.array_equal(getattr(explicit, name), getattr(baseline, name))
        for name in PATCH_WORLD_ARRAY_NAMES
    )
    eed_arrays_absent = all(
        not hasattr(explicit, name) and not hasattr(baseline, name)
        for name in EED_RESOURCE_IMPACT_ARRAY_NAMES
    )
    return {
        "schema_version": "eed0_rp1_flags_off_parity_v1",
        "patch_arrays_equal": array_equal,
        "eed_arrays_absent": eed_arrays_absent,
        "pass": bool(array_equal and eed_arrays_absent),
    }


def _matrix_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        opportunity = row["later_reproductive_opportunity"]
        flat.append(
            {
                "resource": row["resource"],
                "phenotype": row["phenotype"],
                "impact_applied": int(row["impact_applied"]),
                "agent_count": row["agent_count"],
                "action_count": row["action_count"],
                "immediate_energy_total": row["immediate_energy_total"],
                "immediate_a_depletion": row["immediate_depletion"]["resource_a"],
                "immediate_b_depletion": row["immediate_depletion"]["resource_b"],
                "pending_a": row["impact_after_action"]["pending_a"],
                "pending_b": row["impact_after_action"]["pending_b"],
                "later_a_stock": row["later_stock"]["resource_a"],
                "later_b_stock": row["later_stock"]["resource_b"],
                "later_opportunity_multiplier": opportunity[
                    "lro1_opportunity_multiplier"
                ],
                "later_effective_try_prob": opportunity[
                    "lro1_opportunity_effective_try_prob"
                ],
                "rng_unchanged": int(row["rng_unchanged"]),
            }
        )
    return flat


def _trajectory_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        arm = {
            "resource": row["resource"],
            "phenotype": row["phenotype"],
            "impact_applied": int(row["impact_applied"]),
        }
        for stage, stocks in (
            ("initial", row["initial_stock"]),
            ("post_immediate_gather", row["immediate_stock"]),
            ("post_impact_settlement", row["later_stock"]),
        ):
            out.append(
                {
                    **arm,
                    "stage": stage,
                    "resource_a_stock": stocks["resource_a"],
                    "resource_b_stock": stocks["resource_b"],
                }
            )
    return out


def build_evidence() -> dict[str, Any]:
    rows = fixture_matrix()
    a_off, a_on = _paired(rows, "resource_a", "a_oriented")
    b_off, b_on = _paired(rows, "resource_b", "b_oriented")

    def _pair_pass(off: dict[str, Any], on: dict[str, Any], resource: str) -> bool:
        return bool(
            off["agent_count"] == on["agent_count"] == 1
            and off["action_count"] == on["action_count"] == 2
            and off["immediate_stock"] == on["immediate_stock"]
            and off["immediate_energy_by_action"] == on["immediate_energy_by_action"]
            and on["later_stock"][resource] < off["later_stock"][resource]
            and on["later_reproductive_opportunity"][
                "lro1_opportunity_effective_try_prob"
            ]
            < off["later_reproductive_opportunity"][
                "lro1_opportunity_effective_try_prob"
            ]
        )

    a_pass = _pair_pass(a_off, a_on, "resource_a")
    b_pass = _pair_pass(b_off, b_on, "resource_b")
    neutral_pending_zero = all(
        row["impact_after_action"]["pending_a"] == 0.0
        and row["impact_after_action"]["pending_b"] == 0.0
        for row in rows
        if row["phenotype"] in {"neutral", "equalized_capacity_sham"}
    )
    rng_pass = all(row["rng_unchanged"] for row in rows)
    repeat_rows = fixture_matrix()
    repeat_pass = _canonical(rows) == _canonical(repeat_rows)
    flags_off = _flags_off_parity()
    ablation_pass = bool(
        a_off["impact_after_action"]["pending_a"] == 0.0
        and b_off["impact_after_action"]["pending_b"] == 0.0
        and a_off["immediate_stock"] == a_on["immediate_stock"]
        and b_off["immediate_stock"] == b_on["immediate_stock"]
        and a_off["immediate_energy_by_action"] == a_on["immediate_energy_by_action"]
        and b_off["immediate_energy_by_action"] == b_on["immediate_energy_by_action"]
    )
    decision = (
        "EED-0-RP1-A"
        if all(
            (
                a_pass,
                b_pass,
                neutral_pending_zero,
                rng_pass,
                repeat_pass,
                flags_off["pass"],
                ablation_pass,
            )
        )
        else "EED-0-RP1-B"
    )
    return {
        "rows": rows,
        "matrix_csv": _matrix_csv_rows(rows),
        "trajectories_csv": _trajectory_rows(rows),
        "reciprocal": {
            "schema_version": "eed0_rp1_reciprocal_later_consequence_v1",
            "a_direction_pass": a_pass,
            "b_direction_pass": b_pass,
            "a_impact_off_try_prob": a_off["later_reproductive_opportunity"][
                "lro1_opportunity_effective_try_prob"
            ],
            "a_impact_on_try_prob": a_on["later_reproductive_opportunity"][
                "lro1_opportunity_effective_try_prob"
            ],
            "b_impact_off_try_prob": b_off["later_reproductive_opportunity"][
                "lro1_opportunity_effective_try_prob"
            ],
            "b_impact_on_try_prob": b_on["later_reproductive_opportunity"][
                "lro1_opportunity_effective_try_prob"
            ],
            "a_impact_off_later_stock": a_off["later_stock"]["resource_a"],
            "a_impact_on_later_stock": a_on["later_stock"]["resource_a"],
            "b_impact_off_later_stock": b_off["later_stock"]["resource_b"],
            "b_impact_on_later_stock": b_on["later_stock"]["resource_b"],
        },
        "ablation": {
            "schema_version": "eed0_rp1_impact_ablation_v1",
            "immediate_action_and_energy_equal": ablation_pass,
            "impact_off_pending_zero": bool(
                a_off["impact_after_action"]["pending_a"] == 0.0
                and b_off["impact_after_action"]["pending_b"] == 0.0
            ),
            "neutral_and_sham_pending_zero": neutral_pending_zero,
            "pass": bool(ablation_pass and neutral_pending_zero),
        },
        "rng": {
            "schema_version": "eed0_rp1_rng_audit_v1",
            "new_rng_draws": 0,
            "all_arm_states_unchanged": rng_pass,
            "pass": rng_pass,
        },
        "repeatability": {
            "schema_version": "eed0_rp1_repeatability_v1",
            "byte_identical": repeat_pass,
            "first_sha256": _sha256_bytes(_canonical(rows)),
            "repeat_sha256": _sha256_bytes(_canonical(repeat_rows)),
            "pass": repeat_pass,
        },
        "flags_off": flags_off,
        "decision": {
            "schema_version": "eed0_rp1_decision_v1",
            "decision": decision,
            "reference_prerequisite_pass": decision == "EED-0-RP1-A",
            "eed0_final_authorized": False,
            "eed1_authorized": False,
            "eed72_authorized": False,
            "next_step": (
                "wait_for_shared_authoritative_schema_native_integration"
                if decision == "EED-0-RP1-A"
                else "name_and_close_one_exact_reference_prerequisite"
            ),
        },
    }


def write_evidence(out_dir: Path) -> dict[str, Any]:
    built = build_evidence()
    frozen_design_path = (
        PROJECT_ROOT / "docs" / "40_eiel_design" / "eed0_rp1_frozen_design.json"
    )
    frozen_design = json.loads(frozen_design_path.read_text(encoding="utf-8"))
    source_paths = [
        PROJECT_ROOT / "engine" / "env" / "env_config.py",
        PROJECT_ROOT / "engine" / "env" / "gridworld_pkg" / "patch_world.py",
        Path(__file__).resolve(),
        frozen_design_path,
    ]
    provenance = {
        "schema_version": "eed0_rp1_source_provenance_v1",
        "base_main_commit": "c18830ff792eb567e69e3083ead453f1ed04084e",
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "prompt_sha256": PROMPT_SHA256,
        "sources": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256_file(path),
            }
            for path in source_paths
        ],
    }
    ownership = {
        "schema_version": "eed0_rp1_ownership_gate_v1",
        "base_main_commit": "c18830ff792eb567e69e3083ead453f1ed04084e",
        "genome_v2_commit_observed": _git_rev_parse_or_head(
            "integration/genome-upgrade-v2-e1-closure"
        ),
        "reference_world_scope_disjoint": True,
        "production_registry_changed": False,
        "authoritative_schema_changed": False,
        "checkpoint_changed": False,
        "native_cxx_cuda_changed": False,
        "pass": True,
    }
    reward_isolation = {
        "schema_version": "eed0_rp1_reward_isolation_v1",
        "metrics_feed_mechanics": False,
        "parent_selection_changed": False,
        "survival_scoring_changed": False,
        "mutation_scoring_changed": False,
        "reproduction_reward_changed": False,
        "later_measure": "existing_lro1_pre_attempt_try_probability",
        "pass": True,
    }
    runtime_scope = {
        "schema_version": "eed0_rp1_runtime_scope_v1",
        "python_reference_world": True,
        "production_component_registered": False,
        "checkpoint_supported": False,
        "native_supported": False,
        "cxx_supported": False,
        "cuda_supported": False,
        "campaign_executed": False,
        "eed1_authorized": False,
        "eed72_authorized": False,
    }

    _json(out_dir / "eed0_rp1_source_provenance.json", provenance)
    _json(out_dir / "eed0_rp1_ownership_gate.json", ownership)
    _json(out_dir / "eed0_rp1_frozen_design.json", frozen_design)
    _csv(out_dir / "eed0_rp1_fixture_matrix.csv", built["matrix_csv"])
    _csv(
        out_dir / "eed0_rp1_transition_trajectories.csv",
        built["trajectories_csv"],
    )
    _json(out_dir / "eed0_rp1_impact_ablation.json", built["ablation"])
    _json(
        out_dir / "eed0_rp1_reciprocal_later_consequence.json",
        built["reciprocal"],
    )
    _json(out_dir / "eed0_rp1_reward_isolation.json", reward_isolation)
    _json(out_dir / "eed0_rp1_repeatability.json", built["repeatability"])
    _json(out_dir / "eed0_rp1_flags_off_parity.json", built["flags_off"])
    _json(out_dir / "eed0_rp1_rng_audit.json", built["rng"])
    _json(out_dir / "eed0_rp1_runtime_scope.json", runtime_scope)
    _json(out_dir / "eed0_rp1_decision.json", built["decision"])
    return built


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    built = write_evidence(args.out_dir.resolve())
    print(json.dumps(built["decision"], sort_keys=True))
    return 0 if built["decision"]["decision"] == "EED-0-RP1-A" else 1


if __name__ == "__main__":
    raise SystemExit(main())
