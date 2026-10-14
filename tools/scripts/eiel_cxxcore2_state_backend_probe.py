#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "engine"))

from env.env_config import EnvConfig  # noqa: E402
from env.gridworld_pkg.world import GridWorld  # noqa: E402
from perf.cxxcore_env_apply_shadow import config_from_env  # noqa: E402
from perf.cxxcore2_state_backend import (  # noqa: E402
    as_backend_inputs,
    compare_backend_states,
    run_backend_python_reference,
)
from simulation.rng_manager import RNGManager  # noqa: E402
from tools.scripts.eiel_cxxcore1_env_apply_shadow_native_smoke import (  # noqa: E402
    compile_cpp_kernel as compile_cxxcore1_kernel,
    load_env_apply_no_rng,
)
from tools.scripts.eiel_cxxcore2_state_backend_native_smoke import (  # noqa: E402
    compile_cpp_kernel,
    load_backend,
)


def _env_cfg(width: int, height: int) -> EnvConfig:
    return EnvConfig(
        width=width,
        height=height,
        food_spawn_per_step=0,
        max_food=0,
        enable_biomes=False,
        enable_combat=False,
        enable_injury=False,
        enable_scar=False,
        migration_pressure_enable=False,
        tier3_validation_force_random_walk=False,
        gru_cost_base=0.0,
        gru_cost_per_dim=0.0,
    )


def _fixture(*, n: int, width: int, height: int, ticks: int, seed: int):
    if n > width * height:
        raise ValueError("fixture requires n <= width * height")
    rm = RNGManager(global_seed=seed)
    env = GridWorld(_env_cfg(width, height), seed=seed, rng=rm.env_np)
    for aid in range(n):
        env.add_agent(aid, pos=(int(aid // width), int(aid % width)))
    env.grid[:, :] = 0
    agent_ids = np.arange(n, dtype=np.int32)
    actions = np.empty((ticks, n), dtype=np.int32)
    for tick in range(ticks):
        actions[tick, :] = ((agent_ids + tick) % 7).astype(np.int32)
    if ticks > 0:
        gather0 = agent_ids[actions[0] == 4]
        env.grid[env.agent_pos_y[gather0], env.agent_pos_x[gather0]] = 1
    energy = np.full((n,), 120.0, dtype=np.float32)
    health = np.full((n,), 10.0, dtype=np.float32)
    injury = np.zeros((n,), dtype=np.float32)
    scar = np.zeros((n,), dtype=np.float32)
    inputs = as_backend_inputs(
        config=config_from_env(env),
        agent_ids=agent_ids,
        initial_energy=energy,
        initial_health=health,
        initial_injury=injury,
        initial_scar=scar,
        pos_y_by_id=env.agent_pos_y.copy(),
        pos_x_by_id=env.agent_pos_x.copy(),
        agent_grid=env.agent_grid.copy(),
        grid=env.grid.copy(),
        actions_by_tick=actions,
        move_cost_mults=np.ones((n,), dtype=np.float32),
        juvenile_harvest_mults=np.ones((n,), dtype=np.float32),
    )
    return env, inputs


def _python_gridworld_reference(env: GridWorld, inputs: Any) -> tuple[dict[str, np.ndarray], float, float]:
    energy = inputs.initial_energy.copy()
    health = inputs.initial_health.copy()
    injury = inputs.initial_injury.copy()
    scar = inputs.initial_scar.copy()
    py_env_s = 0.0
    prep_s = 0.0
    died = np.zeros_like(energy, dtype=bool)
    for tick in range(int(inputs.actions_by_tick.shape[0])):
        t0 = time.perf_counter()
        actions = inputs.actions_by_tick[tick].astype(np.int32, copy=False)
        prep_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        died, energy, health, food_gained, injury, scar = env.step_agents_batch(
            agent_ids=inputs.agent_ids,
            actions=actions,
            energies=energy,
            healths=health,
            injuries=injury,
            scars=scar,
            move_cost_mults=inputs.move_cost_mults,
            juvenile_harvest_mults=inputs.juvenile_harvest_mults,
        )
        py_env_s += time.perf_counter() - t0
    return {
        "new_energy": energy,
        "new_health": health,
        "new_injury": injury,
        "new_scar": scar,
        "died_mask": died,
        "new_pos_y": env.agent_pos_y[inputs.agent_ids].astype(np.int32, copy=True),
        "new_pos_x": env.agent_pos_x[inputs.agent_ids].astype(np.int32, copy=True),
        "grid": env.grid.copy(),
    }, py_env_s, prep_s


def _arm(name: str, *, shape: str, n: int, cells: int, ticks: int) -> dict[str, Any]:
    return {
        "shape": shape,
        "arm": name,
        "agents": int(n),
        "cells": int(cells),
        "ticks": int(ticks),
        "wall_s": 0.0,
        "python_env_apply_s": 0.0,
        "python_action_prep_s": 0.0,
        "initial_sync_s": 0.0,
        "action_transfer_s": 0.0,
        "cpp_compute_s": 0.0,
        "export_s": 0.0,
        "periodic_export_s": 0.0,
        "compare_s": 0.0,
        "diagnostics_s": 0.0,
        "parity": "not_checked",
        "first_divergence": None,
        "max_abs_diff": 0.0,
        "mismatch_count": 0,
        "fields_compared": 0,
        "interpretation": "",
    }


def run_shape(shape_name: str, *, n: int, width: int, height: int, ticks: int, backend: Any, cxxcore1: Any) -> dict[str, Any]:
    cells = width * height
    out: dict[str, Any] = {"shape": {"name": shape_name, "agents": n, "width": width, "height": height, "cells": cells, "ticks": ticks}, "arms": []}

    env, inputs = _fixture(n=n, width=width, height=height, ticks=ticks, seed=9001)
    env.step_agents_batch(
        agent_ids=inputs.agent_ids,
        actions=inputs.actions_by_tick[0],
        energies=inputs.initial_energy.copy(),
        healths=inputs.initial_health.copy(),
        injuries=inputs.initial_injury.copy(),
        scars=inputs.initial_scar.copy(),
        move_cost_mults=inputs.move_cost_mults,
        juvenile_harvest_mults=inputs.juvenile_harvest_mults,
    )

    env, inputs = _fixture(n=n, width=width, height=height, ticks=ticks, seed=9001)
    arm = _arm("baseline_python_authoritative", shape=shape_name, n=n, cells=cells, ticks=ticks)
    t0 = time.perf_counter()
    python_final, py_env_s, prep_s = _python_gridworld_reference(env, inputs)
    arm["wall_s"] = time.perf_counter() - t0
    arm["python_env_apply_s"] = py_env_s
    arm["python_action_prep_s"] = prep_s
    arm["parity"] = "not_applicable"
    arm["interpretation"] = "warmed Python authoritative no-RNG fixture"
    out["arms"].append(arm)

    env, inputs = _fixture(n=n, width=width, height=height, ticks=ticks, seed=9001)
    arm = _arm("cxxcore1_shadow_cpp_no_compare_reference", shape=shape_name, n=n, cells=cells, ticks=ticks)
    if cxxcore1 is None:
        arm["parity"] = "unavailable"
        arm["interpretation"] = "CXXCORE-1 native boundary unavailable"
    else:
        energy = inputs.initial_energy.copy()
        health = inputs.initial_health.copy()
        injury = inputs.initial_injury.copy()
        scar = inputs.initial_scar.copy()
        t_wall = time.perf_counter()
        for tick in range(ticks):
            t = time.perf_counter()
            snap = {
                "config": inputs.config,
                "agent_ids": inputs.agent_ids,
                "actions": inputs.actions_by_tick[tick],
                "energies": energy,
                "healths": health,
                "injuries": injury,
                "scars": scar,
                "pos_y_by_id": env.agent_pos_y.copy(),
                "pos_x_by_id": env.agent_pos_x.copy(),
                "agent_grid": env.agent_grid.copy(),
                "grid": env.grid.copy(),
                "move_cost_mults": inputs.move_cost_mults,
                "juvenile_harvest_mults": inputs.juvenile_harvest_mults,
            }
            arm["initial_sync_s"] += time.perf_counter() - t
            t = time.perf_counter()
            actual = cxxcore1(**snap)
            arm["cpp_compute_s"] += time.perf_counter() - t
            energy = actual["new_energy"]
            health = actual["new_health"]
            injury = actual["new_injury"]
            scar = actual["new_scar"]
            env.agent_pos_y[inputs.agent_ids] = actual["new_pos_y"]
            env.agent_pos_x[inputs.agent_ids] = actual["new_pos_x"]
        arm["wall_s"] = time.perf_counter() - t_wall
        arm["parity"] = "not_checked"
        arm["interpretation"] = "CXXCORE-1 per-tick copied boundary lower-bound reference"
    out["arms"].append(arm)

    env, inputs = _fixture(n=n, width=width, height=height, ticks=0, seed=9001)
    arm = _arm("cxxcore2_initial_sync_only", shape=shape_name, n=n, cells=cells, ticks=0)
    t0 = time.perf_counter()
    backend(inputs, checkpoint_interval=0, export_final=True)
    arm["wall_s"] = time.perf_counter() - t0
    arm["initial_sync_s"] = arm["wall_s"]
    arm["parity"] = "not_applicable"
    arm["interpretation"] = "one-time Python-to-C++ copied state initialization plus export"
    out["arms"].append(arm)

    env, inputs = _fixture(n=n, width=width, height=height, ticks=ticks, seed=9001)
    arm = _arm("cxxcore2_persistent_backend_final_compare", shape=shape_name, n=n, cells=cells, ticks=ticks)
    t0 = time.perf_counter()
    actual = backend(inputs, checkpoint_interval=0, export_final=True)
    arm["wall_s"] = time.perf_counter() - t0
    arm["cpp_compute_s"] = arm["wall_s"]
    t = time.perf_counter()
    report = compare_backend_states(python_final, actual)
    arm["compare_s"] = time.perf_counter() - t
    arm["wall_s"] = float(arm["wall_s"] + arm["compare_s"])
    arm["parity"] = "pass" if report["ok"] else "fail"
    arm["first_divergence"] = report.get("first_divergence")
    arm["max_abs_diff"] = float(report.get("max_abs_diff", 0.0))
    arm["mismatch_count"] = int(report.get("mismatch_count", 0))
    arm["fields_compared"] = len(report.get("fields_compared", []))
    arm["interpretation"] = "persistent backend main break-even arm"
    out["arms"].append(arm)

    interval = 100 if ticks >= 100 else max(1, ticks)
    env, inputs = _fixture(n=n, width=width, height=height, ticks=ticks, seed=9001)
    arm = _arm("cxxcore2_persistent_backend_periodic_compare", shape=shape_name, n=n, cells=cells, ticks=ticks)
    t0 = time.perf_counter()
    actual = backend(inputs, checkpoint_interval=interval, export_final=True)
    arm["wall_s"] = time.perf_counter() - t0
    arm["cpp_compute_s"] = arm["wall_s"]
    t = time.perf_counter()
    expected = run_backend_python_reference(inputs, checkpoint_interval=interval)
    report = compare_backend_states(expected, actual)
    arm["compare_s"] = time.perf_counter() - t
    arm["wall_s"] = float(arm["wall_s"] + arm["compare_s"])
    arm["periodic_export_s"] = max(0.0, arm["wall_s"] - out["arms"][-1]["wall_s"])
    arm["parity"] = "pass" if report["ok"] else "fail"
    arm["first_divergence"] = report.get("first_divergence")
    arm["max_abs_diff"] = float(report.get("max_abs_diff", 0.0))
    arm["mismatch_count"] = int(report.get("mismatch_count", 0))
    arm["fields_compared"] = len(report.get("fields_compared", []))
    arm["interpretation"] = f"periodic checkpoint/export every {interval} ticks"
    out["arms"].append(arm)

    env, inputs = _fixture(n=n, width=width, height=height, ticks=ticks, seed=9001)
    arm = _arm("cxxcore2_kernel_loop_only_microbench", shape=shape_name, n=n, cells=cells, ticks=ticks)
    t0 = time.perf_counter()
    actual = backend(inputs, checkpoint_interval=0, export_final=False)
    arm["wall_s"] = time.perf_counter() - t0
    arm["cpp_compute_s"] = arm["wall_s"]
    arm["parity"] = "not_applicable"
    arm["interpretation"] = f"checksum-only backend loop checksum={float(actual['checksum']):.6f}"
    out["arms"].append(arm)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="CXXCORE-2 persistent state-backend timing probe.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()
    compile_info = compile_cpp_kernel(args.build_dir / "cxxcore2")
    compile1_info = compile_cxxcore1_kernel(args.build_dir / "cxxcore1")
    if not compile_info.get("ok"):
        payload = {"schema_version": "cxxcore2_state_backend_probe_v1", "native_compile": compile_info, "shapes": []}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    backend = load_backend(Path(str(compile_info["library_path"])))
    cxxcore1 = load_env_apply_no_rng(Path(str(compile1_info["library_path"]))) if compile1_info.get("ok") else None
    shapes = [
        ("small", 256, 32, 32),
        ("medium", 1024, 64, 64),
    ]
    if not args.skip_large:
        shapes.append(("large", 4096, 64, 64))
    results = [
        run_shape(name, n=n, width=w, height=h, ticks=int(args.ticks), backend=backend, cxxcore1=cxxcore1)
        for name, n, w, h in shapes
    ]
    payload = {
        "schema_version": "cxxcore2_state_backend_probe_v1",
        "native_compile": compile_info,
        "cxxcore1_compile": compile1_info,
        "shapes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
