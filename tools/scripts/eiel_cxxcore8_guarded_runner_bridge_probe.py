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

from perf.cxxcore8_guarded_runner_bridge import (  # noqa: E402
    Cxxcore8BridgeConfig,
    bridge_audit,
    result_payload,
    run_guarded_runner_bridge,
)
from simulation.rng_manager import RNGManager  # noqa: E402
from perf.cxxcore7_rng_precompute import (  # noqa: E402
    apply_validation_action_override,
    build_rng_payload,
    precompute_validation_action_override_ints,
)
from tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe import _capture_fixture, _python_authoritative  # noqa: E402
from tools.scripts.eiel_cxxcore7_rng_precompute_runner_bridge_probe import (  # noqa: E402
    compile_cpp_kernel,
    load_rng_precompute_backend,
)


def _arm(shape: str, name: str, *, n: int, cells: int, ticks: int) -> dict[str, Any]:
    return {
        "shape": shape,
        "arm": name,
        "agents": int(n),
        "cells": int(cells),
        "ticks": int(ticks),
        "actions_covered": int(n) * int(ticks),
        "wall_s": 0.0,
        "python_baseline_included_slice_s": 0.0,
        "bridge_snapshot_copy_s": 0.0,
        "rng_precompute_s": 0.0,
        "native_compute_s": 0.0,
        "materialization_s": 0.0,
        "sparse_compare_s": 0.0,
        "final_compare_s": 0.0,
        "writeback_s": 0.0,
        "fallback_overhead_s": 0.0,
        "diagnostics_s": 0.0,
        "parity": "not_checked",
        "sparse_parity": "not_checked",
        "first_divergence": None,
        "fallback_occurred": False,
        "fallback_reason": "",
        "writeback_applied": False,
        "state_digest": "",
        "interpretation": "",
    }


def _apply_result(arm: dict[str, Any], result: Any) -> None:
    payload = result_payload(result)
    timing = payload["timing"]
    parity = payload.get("parity") or {}
    sparse = payload.get("sparse_parity") or {}
    diagnostics = payload.get("diagnostics") or {}
    arm["wall_s"] = float(timing.get("total_wall_s", 0.0))
    arm["bridge_snapshot_copy_s"] = float(timing.get("snapshot_copy_s", 0.0))
    arm["native_compute_s"] = float(timing.get("native_compute_s", 0.0))
    arm["materialization_s"] = float(timing.get("materialization_s", 0.0))
    arm["sparse_compare_s"] = float(timing.get("sparse_compare_s", 0.0))
    arm["final_compare_s"] = float(timing.get("final_compare_s", 0.0))
    arm["writeback_s"] = float(timing.get("writeback_s", 0.0))
    arm["fallback_overhead_s"] = float(timing.get("fallback_s", 0.0))
    arm["diagnostics_s"] = float(timing.get("diagnostics_s", 0.0))
    arm["fallback_occurred"] = bool(payload.get("fallback_occurred", False))
    arm["fallback_reason"] = str(payload.get("fallback_reason", ""))
    arm["writeback_applied"] = bool(payload.get("writeback_applied", False))
    arm["state_digest"] = str(diagnostics.get("state_digest", ""))
    if parity:
        arm["parity"] = "pass" if bool(parity.get("ok", False)) else "fail"
        arm["first_divergence"] = parity.get("first_divergence")
    elif payload["status"] == "fallback":
        arm["parity"] = "fallback"
    else:
        arm["parity"] = payload["status"]
    if sparse:
        arm["sparse_parity"] = "pass" if bool(sparse.get("ok", False)) else "fail"
    elif payload["status"] == "fallback":
        arm["sparse_parity"] = "fallback"
    else:
        arm["sparse_parity"] = payload["status"]
    arm["result"] = payload


def run_shape(shape_name: str, *, n: int, width: int, height: int, ticks: int, native: Any) -> dict[str, Any]:
    cells = width * height
    inputs, _, full_snapshot, _, _, _, _, _ = _capture_fixture(
        n=n,
        width=width,
        height=height,
        ticks=ticks,
        seed=81008,
    )
    result: dict[str, Any] = {
        "shape": {"name": shape_name, "agents": n, "width": width, "height": height, "cells": cells, "ticks": ticks},
        "arms": [],
    }

    t0 = time.perf_counter()
    _, py_timing = _python_authoritative(inputs.backend)
    py_wall = time.perf_counter() - t0
    arm = _arm(shape_name, "flags_off_python_baseline", n=n, cells=cells, ticks=ticks)
    arm["wall_s"] = py_wall + full_snapshot.production_cost_s
    arm["python_baseline_included_slice_s"] = py_timing["py_env_s"]
    arm["interpretation"] = "normal Python baseline with bridge flags off"
    arm["parity"] = "not_applicable"
    arm["sparse_parity"] = "not_applicable"
    result["arms"].append(arm)

    cfg_disabled = Cxxcore8BridgeConfig(cxxcore8_guarded_runner_bridge_enable=False)
    arm = _arm(shape_name, "bridge_disabled_noop", n=n, cells=cells, ticks=ticks)
    bridge = run_guarded_runner_bridge(inputs, config=cfg_disabled, native_backend=native)
    _apply_result(arm, bridge)
    arm["interpretation"] = "default-off bridge returns disabled and does not write back"
    result["arms"].append(arm)

    cfg_shadow = Cxxcore8BridgeConfig(cxxcore8_guarded_runner_bridge_enable=True, cxxcore8_guarded_runner_bridge_mode="shadow")
    arm = _arm(shape_name, "bridge_shadow_no_rng", n=n, cells=cells, ticks=ticks)
    bridge = run_guarded_runner_bridge(inputs, config=cfg_shadow, native_backend=native)
    _apply_result(arm, bridge)
    arm["interpretation"] = "Python authoritative; C++ copied-state output compared only"
    result["arms"].append(arm)

    cfg_auth = Cxxcore8BridgeConfig(
        cxxcore8_guarded_runner_bridge_enable=True,
        cxxcore8_guarded_runner_bridge_mode="authoritative_no_rng",
    )
    arm = _arm(shape_name, "bridge_authoritative_no_rng", n=n, cells=cells, ticks=ticks)
    bridge = run_guarded_runner_bridge(inputs, config=cfg_auth, native_backend=native)
    _apply_result(arm, bridge)
    arm["interpretation"] = "C++ output retained after Python-reference parity"
    result["arms"].append(arm)

    rng_t0 = time.perf_counter()
    rng_payload = build_rng_payload(
        validation_action_override_ints=precompute_validation_action_override_ints(
            rng=RNGManager(global_seed=81008).env_np,
            ticks=ticks,
            agents=n,
        )
    )
    rng_s = time.perf_counter() - rng_t0
    cfg_rng = Cxxcore8BridgeConfig(
        cxxcore8_guarded_runner_bridge_enable=True,
        cxxcore8_guarded_runner_bridge_mode="authoritative_precomputed_rng",
        cxxcore8_runner_bridge_allow_validation_rng_precompute=True,
    )
    arm = _arm(shape_name, "bridge_authoritative_validation_rng_precompute", n=n, cells=cells, ticks=ticks)
    bridge = run_guarded_runner_bridge(inputs, config=cfg_rng, native_backend=native, rng_payload=rng_payload)
    _apply_result(arm, bridge)
    arm["rng_precompute_s"] = rng_s
    arm["interpretation"] = "C++ consumes Python-precomputed validation action RNG and writes back after parity"
    result["arms"].append(arm)

    arm = _arm(shape_name, "missing_native_fallback", n=n, cells=cells, ticks=ticks)
    bridge = run_guarded_runner_bridge(inputs, config=cfg_auth, native_backend=None)
    _apply_result(arm, bridge)
    arm["interpretation"] = "native unavailable fallback uses Python reference"
    result["arms"].append(arm)

    if shape_name == "tiny":
        arm = _arm(shape_name, "missing_rng_payload_fallback", n=n, cells=cells, ticks=ticks)
        bridge = run_guarded_runner_bridge(inputs, config=cfg_rng, native_backend=native, rng_payload=None)
        _apply_result(arm, bridge)
        arm["interpretation"] = "precomputed RNG mode without payload falls back"
        result["arms"].append(arm)

        arm = _arm(shape_name, "injected_divergence_fallback", n=n, cells=cells, ticks=ticks)
        bridge = run_guarded_runner_bridge(inputs, config=cfg_rng, native_backend=native, rng_payload=rng_payload, inject_divergence=True)
        _apply_result(arm, bridge)
        arm["rng_precompute_s"] = rng_s
        arm["interpretation"] = "injected C++ divergence falls back to Python reference"
        result["arms"].append(arm)

    ref_inputs = apply_validation_action_override(inputs, rng_payload)
    expected, _ = _python_authoritative(ref_inputs.backend)
    result["validation_rng_reference_digest"] = str(result["arms"][4].get("state_digest", ""))
    result["metrics_final_row_compatibility"] = {
        "implemented": "summary_digest",
        "energy_sum": float(np.sum(expected["new_energy"], dtype=np.float64)),
        "died_count": float(np.sum(expected["died_mask"])),
        "cost_total": float(np.sum(expected["cost_total_by_agent"], dtype=np.float64)),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="CXXCORE-8 guarded runner bridge probe.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=240)
    parser.add_argument("--skip-medium", action="store_true")
    args = parser.parse_args()

    compile8 = compile_cpp_kernel(args.build_dir / "cxxcore8")
    payload: dict[str, Any] = {
        "schema_version": "cxxcore8_guarded_runner_bridge_probe_v1",
        "git": {
            "base_commit": "a7d80bfdd45486b6c48f354412a062c5c82896f0",
            "depends_on": [
                "a7d80bfdd45486b6c48f354412a062c5c82896f0",
                "32cd7d88e0c7edd2e40ce92c59e91200bf9cc4fe",
            ],
        },
        "native_compile": compile8,
        "bridge_audit": bridge_audit(),
        "optional_expansions": {
            "biome_hazard_precompute": "deferred; requires extra copied biome/hazard state",
            "tier3_transition_precompute": "design_only; branch-dependent crossing schedule needs dedicated preflight",
        },
        "shapes": [],
    }
    if not compile8.get("ok"):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    native = load_rng_precompute_backend(Path(str(compile8["library_path"])))
    shapes = [("tiny", 16, 8, 8, min(48, int(args.ticks))), ("compact_runner", 128, 16, 16, min(240, int(args.ticks)))]
    if not args.skip_medium:
        shapes.append(("medium_runner", 512, 32, 32, min(240, int(args.ticks))))
    payload["shapes"] = [
        run_shape(name, n=n, width=w, height=h, ticks=t, native=native)
        for name, n, w, h, t in shapes
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
