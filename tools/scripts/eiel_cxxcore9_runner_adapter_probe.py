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

from perf.cxxcore7_rng_precompute import (  # noqa: E402
    build_rng_payload,
    precompute_validation_action_override_ints,
)
from perf.cxxcore9_runner_adapter import (  # noqa: E402
    Cxxcore9AdapterConfig,
    adapter_boundary_map,
    result_payload,
    run_runner_adapter,
)
from simulation.rng_manager import RNGManager  # noqa: E402
from tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe import _capture_fixture, _python_authoritative  # noqa: E402
from tools.scripts.eiel_cxxcore7_rng_precompute_runner_bridge_probe import (  # noqa: E402
    compile_cpp_kernel,
    load_rng_precompute_backend,
)


def _empty_arm(shape: str, arm: str, *, n: int, cells: int, ticks: int) -> dict[str, Any]:
    return {
        "shape": shape,
        "arm": arm,
        "agents": int(n),
        "cells": int(cells),
        "ticks": int(ticks),
        "actions_covered": int(n) * int(ticks),
        "total_wall_s": 0.0,
        "python_reference_s": 0.0,
        "snapshot_restore_token_s": 0.0,
        "cost_layout_cache_update_s": 0.0,
        "rng_precompute_s": 0.0,
        "native_backend_s": 0.0,
        "comparison_parity_s": 0.0,
        "writeback_materialization_s": 0.0,
        "diagnostics_s": 0.0,
        "fallback_restoration_s": 0.0,
        "parity": "not_checked",
        "fallback_reason": "",
        "first_divergence": None,
        "writeback_applied": False,
        "retained_native_output": False,
        "state_digest": "",
        "interpretation": "",
    }


def _apply_adapter_result(arm: dict[str, Any], result: Any) -> None:
    payload = result_payload(result)
    timing = payload["timing"]
    arm["total_wall_s"] = float(timing.get("total_wall_s", 0.0))
    arm["python_reference_s"] = float(timing.get("python_reference_s", 0.0))
    arm["snapshot_restore_token_s"] = float(timing.get("snapshot_restore_token_s", 0.0))
    arm["native_backend_s"] = float(timing.get("native_backend_s", 0.0))
    arm["comparison_parity_s"] = float(timing.get("sparse_compare_s", 0.0)) + float(timing.get("final_compare_s", 0.0))
    arm["writeback_materialization_s"] = float(timing.get("materialization_s", 0.0)) + float(timing.get("writeback_s", 0.0))
    arm["diagnostics_s"] = float(timing.get("diagnostics_s", 0.0))
    arm["fallback_restoration_s"] = float(timing.get("fallback_restoration_s", 0.0))
    arm["parity"] = "pass" if payload.get("parity_passed") else "fail" if payload.get("parity_checked") else "not_checked"
    if payload.get("fallback_occurred"):
        arm["parity"] = "fallback" if not payload.get("parity_checked") else arm["parity"]
    arm["fallback_reason"] = str(payload.get("fallback_reason") or "")
    arm["first_divergence"] = payload.get("first_divergence")
    arm["writeback_applied"] = bool(payload.get("writeback_applied", False))
    arm["retained_native_output"] = bool(payload.get("retained_native_output", False))
    arm["state_digest"] = str(payload.get("state_digest", ""))
    arm["result"] = payload


def run_shape(shape_name: str, *, n: int, width: int, height: int, ticks: int, native: Any) -> dict[str, Any]:
    cells = width * height
    inputs, _, full_snapshot, _, _, _, _, _ = _capture_fixture(
        n=n,
        width=width,
        height=height,
        ticks=ticks,
        seed=91009,
    )
    result: dict[str, Any] = {
        "shape": {"name": shape_name, "agents": n, "width": width, "height": height, "cells": cells, "ticks": ticks},
        "arms": [],
    }

    t0 = time.perf_counter()
    _, py_timing = _python_authoritative(inputs.backend)
    py_wall = time.perf_counter() - t0
    arm = _empty_arm(shape_name, "baseline_python_flags_off", n=n, cells=cells, ticks=ticks)
    arm["total_wall_s"] = py_wall + full_snapshot.production_cost_s
    arm["python_reference_s"] = float(py_timing["py_env_s"])
    arm["cost_layout_cache_update_s"] = float(full_snapshot.production_cost_s)
    arm["parity"] = "not_applicable"
    arm["interpretation"] = "normal Python path; all CXXCORE-9 flags off"
    result["arms"].append(arm)

    cfg_disabled = Cxxcore9AdapterConfig(cxxcore9_runner_adapter_enable=False)
    arm = _empty_arm(shape_name, "adapter_disabled_noop", n=n, cells=cells, ticks=ticks)
    _apply_adapter_result(arm, run_runner_adapter(inputs, config=cfg_disabled, native_backend=native))
    arm["interpretation"] = "default-off adapter creates no native output and does not write back"
    result["arms"].append(arm)

    cfg_shadow = Cxxcore9AdapterConfig(
        cxxcore9_runner_adapter_enable=True,
        cxxcore9_runner_adapter_mode="shadow",
        cxxcore9_runner_adapter_parity_policy="every_tick",
        cxxcore9_runner_adapter_parity_interval=1,
    )
    arm = _empty_arm(shape_name, "adapter_shadow_every_tick", n=n, cells=cells, ticks=ticks)
    _apply_adapter_result(arm, run_runner_adapter(inputs, config=cfg_shadow, native_backend=native))
    arm["interpretation"] = "Python authoritative; native candidate compared every tick/final via copied-state bridge"
    result["arms"].append(arm)

    cfg_no_rng = Cxxcore9AdapterConfig(
        cxxcore9_runner_adapter_enable=True,
        cxxcore9_runner_adapter_mode="guarded_authoritative_no_rng",
        cxxcore9_runner_adapter_parity_policy="interval",
        cxxcore9_runner_adapter_parity_interval=8,
    )
    arm = _empty_arm(shape_name, "adapter_guarded_authoritative_no_rng_interval_parity", n=n, cells=cells, ticks=ticks)
    _apply_adapter_result(arm, run_runner_adapter(inputs, config=cfg_no_rng, native_backend=native))
    arm["interpretation"] = "copied-state native output retained only after parity; not live runner object writeback"
    result["arms"].append(arm)

    rng_t0 = time.perf_counter()
    rng_payload = build_rng_payload(
        validation_action_override_ints=precompute_validation_action_override_ints(
            rng=RNGManager(global_seed=91009).env_np,
            ticks=ticks,
            agents=n,
        )
    )
    rng_s = time.perf_counter() - rng_t0
    cfg_rng = Cxxcore9AdapterConfig(
        cxxcore9_runner_adapter_enable=True,
        cxxcore9_runner_adapter_mode="guarded_authoritative_validation_rng",
        cxxcore9_runner_adapter_parity_policy="interval",
        cxxcore9_runner_adapter_parity_interval=8,
    )
    arm = _empty_arm(shape_name, "adapter_guarded_authoritative_validation_rng_interval_parity", n=n, cells=cells, ticks=ticks)
    _apply_adapter_result(arm, run_runner_adapter(inputs, config=cfg_rng, native_backend=native, rng_payload=rng_payload))
    arm["rng_precompute_s"] = rng_s
    arm["interpretation"] = "C++ consumes Python-precomputed validation action override array"
    result["arms"].append(arm)

    if shape_name == "tiny":
        arm = _empty_arm(shape_name, "adapter_fallback_injected_divergence", n=n, cells=cells, ticks=ticks)
        _apply_adapter_result(
            arm,
            run_runner_adapter(
                inputs,
                config=cfg_rng,
                native_backend=native,
                rng_payload=rng_payload,
                inject_divergence=True,
            ),
        )
        arm["rng_precompute_s"] = rng_s
        arm["interpretation"] = "injected native divergence falls back to Python reference output"
        result["arms"].append(arm)

        arm = _empty_arm(shape_name, "adapter_missing_native_fallback", n=n, cells=cells, ticks=ticks)
        _apply_adapter_result(arm, run_runner_adapter(inputs, config=cfg_no_rng, native_backend=None))
        arm["interpretation"] = "missing native backend returns Python reference and retains no native output"
        result["arms"].append(arm)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="CXXCORE-9 guarded runner adapter probe.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=180)
    parser.add_argument("--skip-medium", action="store_true")
    args = parser.parse_args()

    compile9 = compile_cpp_kernel(args.build_dir / "cxxcore9")
    payload: dict[str, Any] = {
        "schema_version": "cxxcore9_runner_adapter_probe_v1",
        "git": {
            "base_commit": "2176fa3d639cece34e924c3e5732761a54562841",
            "dependency": "CXXCORE-8 guarded copied-state bridge",
        },
        "native_compile": compile9,
        "boundary_map": adapter_boundary_map(),
        "decision_scope": "default-off adapter guardrails; no live Agent/GridWorld writeback claim",
        "shapes": [],
    }
    if not compile9.get("ok"):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    native = load_rng_precompute_backend(Path(str(compile9["library_path"])))
    shapes = [
        ("tiny", 16, 8, 8, min(48, int(args.ticks))),
        ("compact_runner", 128, 16, 16, min(180, int(args.ticks))),
    ]
    if not args.skip_medium:
        shapes.append(("medium_runner", 512, 32, 32, min(180, int(args.ticks))))
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
