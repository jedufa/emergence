#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from perf.cxxcore10_live_env_tick_side_effect_split import (  # noqa: E402
    Cxxcore10SplitConfig,
    result_payload,
    run_env_tick_split_transaction,
    side_effect_taxonomy_payload,
)
from simulation.rng_manager import RNGManager  # noqa: E402
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
        "total_wall_s": 0.0,
        "python_direct_ref_s": 0.0,
        "python_split_materialize_s": 0.0,
        "native_s": 0.0,
        "writeback_s": 0.0,
        "compare_s": 0.0,
        "fallback_restore_s": 0.0,
        "rng_precompute_s": 0.0,
        "diagnostics_s": 0.0,
        "cache_layout_refresh_s": 0.0,
        "parity_fallback": "",
        "first_divergence": None,
        "writeback_mode": "",
        "retained_native_output": False,
        "interpretation": "",
    }


def _apply_result(arm: dict[str, Any], result: Any) -> None:
    payload = result_payload(result)
    timing = payload["timing"]
    arm["total_wall_s"] = float(timing.get("total_wall_s", 0.0))
    arm["python_direct_ref_s"] = float(timing.get("python_direct_reference_s", 0.0))
    arm["python_split_materialize_s"] = float(timing.get("python_split_materialize_s", 0.0))
    arm["native_s"] = float(timing.get("native_backend_s", 0.0))
    arm["writeback_s"] = float(timing.get("python_writeback_s", 0.0))
    arm["compare_s"] = float(timing.get("comparison_s", 0.0))
    arm["fallback_restore_s"] = float(timing.get("fallback_restore_s", 0.0))
    arm["diagnostics_s"] = float(timing.get("diagnostics_s", 0.0))
    arm["cache_layout_refresh_s"] = float(timing.get("cache_layout_refresh_s", 0.0))
    if payload.get("fallback_reason"):
        arm["parity_fallback"] = str(payload["fallback_reason"])
    elif payload.get("parity_passed"):
        arm["parity_fallback"] = "pass"
    elif payload.get("parity") is None:
        arm["parity_fallback"] = str(payload.get("status", "not_checked"))
    else:
        arm["parity_fallback"] = "fail"
    arm["first_divergence"] = payload.get("first_divergence")
    arm["writeback_mode"] = str((payload.get("diagnostics") or {}).get("writeback", {}).get("writeback_mode", ""))
    arm["retained_native_output"] = bool(payload.get("retained_native_output", False))
    arm["result"] = payload


def run_shape(shape_name: str, *, n: int, width: int, height: int, ticks: int, native: Any) -> dict[str, Any]:
    cells = width * height
    inputs, _, full_snapshot, _, _, _, _, _ = _capture_fixture(
        n=n,
        width=width,
        height=height,
        ticks=ticks,
        seed=101010,
    )
    result: dict[str, Any] = {
        "shape": {"name": shape_name, "agents": n, "width": width, "height": height, "cells": cells, "ticks": ticks},
        "arms": [],
    }

    t0 = time.perf_counter()
    _, py_timing = _python_authoritative(inputs.backend)
    py_wall = time.perf_counter() - t0
    arm = _arm(shape_name, "baseline_python_flags_off", n=n, cells=cells, ticks=ticks)
    arm["total_wall_s"] = py_wall + full_snapshot.production_cost_s
    arm["python_direct_ref_s"] = float(py_timing["py_env_s"])
    arm["cache_layout_refresh_s"] = float(full_snapshot.production_cost_s)
    arm["parity_fallback"] = "not_applicable"
    arm["interpretation"] = "current Python reference path with split disabled"
    result["arms"].append(arm)

    cfg_py = Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=True,
        cxxcore10_env_tick_split_mode="python_split_reference",
    )
    arm = _arm(shape_name, "python_split_transaction_reference", n=n, cells=cells, ticks=ticks)
    _apply_result(arm, run_env_tick_split_transaction(inputs, config=cfg_py, native_backend=None))
    arm["interpretation"] = "Python materialized result feeds Python-owned copied-state writeback boundary"
    result["arms"].append(arm)

    cfg_shadow = Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=True,
        cxxcore10_env_tick_split_mode="native_shadow_transaction",
    )
    arm = _arm(shape_name, "native_shadow_transaction_every_tick", n=n, cells=cells, ticks=ticks)
    _apply_result(arm, run_env_tick_split_transaction(inputs, config=cfg_shadow, native_backend=native))
    arm["interpretation"] = "native materialized candidate compared; Python split output remains retained"
    result["arms"].append(arm)

    cfg_guard = Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=True,
        cxxcore10_env_tick_split_mode="native_guarded_transaction_no_rng",
        cxxcore10_env_tick_split_parity_policy="interval",
        cxxcore10_env_tick_split_parity_interval=8,
    )
    arm = _arm(shape_name, "native_guarded_transaction_no_rng_interval_parity", n=n, cells=cells, ticks=ticks)
    _apply_result(arm, run_env_tick_split_transaction(inputs, config=cfg_guard, native_backend=native))
    arm["interpretation"] = "native materialized result feeds Python-owned copied-state writeback after parity"
    result["arms"].append(arm)

    rng_t0 = time.perf_counter()
    rng_payload = build_rng_payload(
        validation_action_override_ints=precompute_validation_action_override_ints(
            rng=RNGManager(global_seed=101010).env_np,
            ticks=ticks,
            agents=n,
        )
    )
    rng_s = time.perf_counter() - rng_t0
    cfg_rng = Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=True,
        cxxcore10_env_tick_split_mode="native_guarded_transaction_validation_rng",
        cxxcore10_env_tick_split_parity_policy="interval",
        cxxcore10_env_tick_split_parity_interval=8,
    )
    arm = _arm(shape_name, "native_guarded_transaction_validation_rng_interval_parity", n=n, cells=cells, ticks=ticks)
    _apply_result(arm, run_env_tick_split_transaction(inputs, config=cfg_rng, native_backend=native, rng_payload=rng_payload))
    arm["rng_precompute_s"] = rng_s
    arm["interpretation"] = "C++ consumes Python-precomputed validation action override arrays"
    result["arms"].append(arm)

    if shape_name == "tiny":
        arm = _arm(shape_name, "native_guarded_injected_divergence_fallback", n=n, cells=cells, ticks=ticks)
        _apply_result(
            arm,
            run_env_tick_split_transaction(
                inputs,
                config=cfg_guard,
                native_backend=native,
                inject_divergence=True,
            ),
        )
        arm["interpretation"] = "injected native divergence restores Python reference transaction output"
        result["arms"].append(arm)

        arm = _arm(shape_name, "native_missing_fallback", n=n, cells=cells, ticks=ticks)
        _apply_result(arm, run_env_tick_split_transaction(inputs, config=cfg_guard, native_backend=None))
        arm["interpretation"] = "missing native backend falls back before native materialization"
        result["arms"].append(arm)

    return result


def _write_flat_csv(payload: dict[str, Any], path: Path) -> None:
    fields = [
        "shape",
        "arm",
        "agents",
        "cells",
        "ticks",
        "total_wall_s",
        "python_direct_ref_s",
        "python_split_materialize_s",
        "native_s",
        "writeback_s",
        "compare_s",
        "fallback_restore_s",
        "rng_precompute_s",
        "diagnostics_s",
        "cache_layout_refresh_s",
        "parity_fallback",
        "retained_native_output",
        "interpretation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for shape in payload["shapes"]:
            for arm in shape["arms"]:
                writer.writerow({field: arm.get(field, "") for field in fields})


def _write_taxonomy_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "classification", "notes"])
        writer.writeheader()
        for row in side_effect_taxonomy_payload()["rows"]:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="CXXCORE-10 live env-tick side-effect split probe.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=180)
    parser.add_argument("--skip-medium", action="store_true")
    args = parser.parse_args()

    compile10 = compile_cpp_kernel(args.build_dir / "cxxcore10")
    payload: dict[str, Any] = {
        "schema_version": "cxxcore10_live_env_tick_side_effect_split_probe_v1",
        "git": {
            "base_commit": "278c528a0dbb80741ea3d43b3527e202a2d67203",
            "dependency": "CXXCORE-9 adapter guardrail contract",
        },
        "native_compile": compile10,
        "side_effect_taxonomy": side_effect_taxonomy_payload(),
        "decision_scope": "default-off transaction side-effect split; no live Agent/GridWorld mutation by native",
        "shapes": [],
    }
    if not compile10.get("ok"):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    native = load_rng_precompute_backend(Path(str(compile10["library_path"])))
    shapes = [
        ("tiny", 16, 8, 8, min(48, int(args.ticks))),
        ("compact_runner", 128, 16, 16, min(180, int(args.ticks))),
    ]
    if not args.skip_medium:
        shapes.append(("medium_runner", 512, 32, 32, min(180, int(args.ticks))))
    payload["shapes"] = [run_shape(name, n=n, width=w, height=h, ticks=t, native=native) for name, n, w, h, t in shapes]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_flat_csv(payload, args.output.with_name("cxxcore10_timing_parity_summary.csv"))
    _write_taxonomy_csv(args.output.with_name("cxxcore10_side_effect_taxonomy.csv"))
    args.output.with_name("cxxcore10_side_effect_taxonomy.json").write_text(
        json.dumps(side_effect_taxonomy_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
