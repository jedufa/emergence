#!/usr/bin/env python3
"""Measure, but never alter, the physical-B48 HAPB policy callback.

This is an M0 attribution probe.  It deliberately profiles the current sealed
prepared-boundary route before any fixed-MLP architecture decision.  The
default fixture admits every physical row as a miss so that it exercises both
the hot-prefix cache and the pageable overflow tile.  It is not a trajectory
or qualification runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
# This must match the compiled PyInit symbol of the sealed extension, not the
# profiler's filename.
MODULE_NAME = "eiel_cuda_architecture_b_wdl_jungle_eed_hapb_v1"

from engine.perf.cuda_architecture_b_hapb_v1 import (
    CELL_COUNT_V1,
    HOT_PREFIX_ROWS_V1,
    POLICY_WIDTH_V1,
    SLOT_CAPACITY_V1,
    execute_hapb_prepared_policy_boundary_v1,
    initialize_selected_hapb_owner_v1,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("HAPB artifact cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(samples_ns: list[int]) -> dict[str, int]:
    if not samples_ns:
        raise ValueError("at least one timing sample is required")
    ordered = sorted(samples_ns)
    return {
        "count": len(samples_ns),
        "min_ns": ordered[0],
        "median_ns": int(statistics.median(ordered)),
        "p95_ns": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "max_ns": ordered[-1],
    }


def _timed_cuda_call(
    device: torch.device,
    call: Callable[[], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, int]]:
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    started_ns = time.perf_counter_ns()
    result = call()
    end_event.record()
    end_event.synchronize()
    return result, {
        "host_wall_ns": time.perf_counter_ns() - started_ns,
        "cuda_stream_ns": int(round(start_event.elapsed_time(end_event) * 1_000_000)),
    }


def _fixture(
    *, active_rows: int, miss_rows: int, first_active_slot: int
) -> dict[str, Any]:
    if (
        not 0 < active_rows <= SLOT_CAPACITY_V1
        or not 0 < miss_rows <= active_rows
        or first_active_slot < 0
        or first_active_slot + active_rows > SLOT_CAPACITY_V1
    ):
        raise ValueError("require 1 <= miss rows <= active rows <= 512")
    canonical = torch.zeros(
        (CELL_COUNT_V1, SLOT_CAPACITY_V1, POLICY_WIDTH_V1), dtype=torch.float32
    )
    # Keep the fixture deterministic while ensuring the compact MLP reads
    # non-zero policy parameters on both sides of the hot-prefix boundary.
    canonical[:, :, 0] = 0.125
    canonical[:, HOT_PREFIX_ROWS_V1:, 1] = -0.25
    owner = initialize_selected_hapb_owner_v1(canonical, device="cuda")
    device = owner.cuda_tile_arena.device
    ordered = torch.full((48, 512), -1, dtype=torch.int32, device=device)
    ordered[:, :active_rows] = torch.arange(
        first_active_slot, first_active_slot + active_rows,
        dtype=torch.int32, device=device,
    )
    observations = torch.zeros((48, 512, 83), dtype=torch.float32, device=device)
    misses = torch.zeros((48, 512), dtype=torch.int64, device=device)
    misses[:, :miss_rows] = torch.arange(miss_rows, dtype=torch.int64, device=device)
    active = torch.full((48,), active_rows, dtype=torch.int64, device=device)
    miss_count = torch.full((48,), miss_rows, dtype=torch.int64, device=device)
    payload: dict[str, Any] = {
        "owner": owner,
        "ordered_slots": ordered,
        "observations": observations,
        "stable_miss_rows": misses,
        "active_row_counts": active,
        "miss_row_counts": miss_count,
        "drives": torch.zeros((48, 512, 4), dtype=torch.float32, device=device),
        "attack_bias": torch.zeros((48, 512), dtype=torch.float32, device=device),
        "action_temperature": torch.ones((48, 512), dtype=torch.float32, device=device),
        "drive_logit_gain": torch.ones((48, 512), dtype=torch.float32, device=device),
        "slot_tokens": torch.arange(48 * 512, dtype=torch.int64, device=device).reshape(48, 512),
        "action_cache_actions": torch.zeros((48, 512, 2), dtype=torch.int32, device=device),
        "action_cache_token": torch.full((48, 512), -1, dtype=torch.int64, device=device),
        "action_cache_cursor": torch.zeros((48, 512), dtype=torch.int32, device=device),
        "action_cache_next": torch.zeros((48, 512), dtype=torch.int32, device=device),
        "row_actions": torch.full((48, 512), -1, dtype=torch.int32, device=device),
        "policy_pcg64_state_words": torch.zeros((48, 4), dtype=torch.uint64, device=device),
        "policy_pcg64_buffer_words": torch.zeros((48, 2), dtype=torch.int64, device=device),
    }
    payload["policy_pcg64_state_words"][:, 0] = 1
    payload["policy_pcg64_state_words"][:, 2] = 1
    return payload


def _policy_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "w_hunger_gather": 0.1,
        "w_pain_rest": 0.2,
        "w_fatigue_rest": 0.3,
        "w_pain_attack": 0.4,
        "w_fear_attack": 0.5,
    }


def _split_callback(module: Any, payload: dict[str, Any]) -> tuple[torch.Tensor, dict[str, dict[str, int]]]:
    owner = payload["owner"]
    device = owner.cuda_tile_arena.device
    timings: dict[str, dict[str, int]] = {}
    started_ns = time.perf_counter_ns()
    ordered_cpu = payload["ordered_slots"].cpu()
    miss_cpu = payload["stable_miss_rows"].cpu()
    active_cpu = payload["active_row_counts"].cpu()
    miss_count_cpu = payload["miss_row_counts"].cpu()
    timings["control_maps_d2h"] = {"host_wall_ns": time.perf_counter_ns() - started_ns}
    started_ns = time.perf_counter_ns()
    hot_rows: list[int] = []
    hot_active_rows: list[int] = []
    overflow_rows: list[int] = []
    overflow_active_rows: list[int] = []
    for cell in range(CELL_COUNT_V1):
        active_count = int(active_cpu[cell].item())
        for ordinal in range(int(miss_count_cpu[cell].item())):
            active_row = int(miss_cpu[cell, ordinal].item())
            slot = int(ordered_cpu[cell, active_row].item())
            if slot < HOT_PREFIX_ROWS_V1:
                hot_rows.append(cell * SLOT_CAPACITY_V1 + slot)
                hot_active_rows.append(cell * SLOT_CAPACITY_V1 + active_row)
            else:
                overflow_rows.append(cell * SLOT_CAPACITY_V1 + slot)
                overflow_active_rows.append(cell * SLOT_CAPACITY_V1 + active_row)
        if active_count < 0 or active_count > SLOT_CAPACITY_V1:
            raise RuntimeError("profile fixture produced an invalid active count")
    timings["host_plan_decode"] = {"host_wall_ns": time.perf_counter_ns() - started_ns}
    started_ns = time.perf_counter_ns()
    stable_overflow_rows = torch.tensor(overflow_rows, dtype=torch.int64)
    stable_overflow_active_rows = torch.tensor(overflow_active_rows, dtype=torch.int64)
    hot_rows_tensor = torch.tensor(hot_rows, dtype=torch.int64)
    hot_active_tensor = torch.tensor(hot_active_rows, dtype=torch.int64)
    timings["host_row_map_materialization"] = {"host_wall_ns": time.perf_counter_ns() - started_ns}
    context_arguments = (
        owner.canonical_host_bank, owner.pinned_staging_buffers[0],
        owner.require_hot_prefix_cache_v1(), owner.cuda_tile_arena,
        stable_overflow_rows, stable_overflow_active_rows,
        payload["policy_pcg64_state_words"], payload["policy_pcg64_buffer_words"],
    )
    policy_arguments = (
        payload["observations"].reshape(-1, 83), hot_rows_tensor, hot_active_tensor,
        payload["drives"].reshape(-1, 4), payload["attack_bias"].reshape(-1),
        payload["action_temperature"].reshape(-1), payload["drive_logit_gain"].reshape(-1),
    )
    weights = (0.1, 0.2, 0.3, 0.4, 0.5)
    logits, timings["native_refill"] = _timed_cuda_call(
        device,
        lambda: module.eiel_cuda_architecture_b_hapb_refill_policy_logits_from_context_v1(
            *context_arguments, *policy_arguments, 64, 7, *weights
        ),
    )
    settlement_arguments = (
        payload["ordered_slots"], payload["slot_tokens"],
        payload["action_cache_actions"], payload["action_cache_token"],
        payload["action_cache_cursor"], payload["action_cache_next"], logits,
        payload["row_actions"], payload["stable_miss_rows"],
        payload["active_row_counts"], payload["miss_row_counts"],
        payload["policy_pcg64_state_words"], payload["policy_pcg64_buffer_words"],
    )
    counters, timings["native_settlement"] = _timed_cuda_call(
        device,
        lambda: module.eiel_cuda_architecture_b_hapb_settle_prepared_policy_rows_v1(
            *settlement_arguments, *weights
        ),
    )
    timings["split_callback"] = {
        "host_wall_ns": sum(
            values["host_wall_ns"]
            for values in timings.values()
            if "host_wall_ns" in values
        )
    }
    return counters, timings


def _run_total(module: Any, payload: dict[str, Any]) -> tuple[torch.Tensor, dict[str, int]]:
    kwargs = _policy_kwargs(payload)
    owner = kwargs.pop("owner")
    device = owner.cuda_tile_arena.device
    return _timed_cuda_call(
        device,
        lambda: execute_hapb_prepared_policy_boundary_v1(
            owner,
            refill_logits_export=module.eiel_cuda_architecture_b_hapb_refill_policy_logits_from_context_v1,
            settle_prepared_export=module.eiel_cuda_architecture_b_hapb_settle_prepared_policy_rows_v1,
            native_boundary_export=module.eiel_cuda_architecture_b_hapb_execute_prepared_policy_boundary_v1,
            **kwargs,
        )[0],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--active-rows-per-cell", type=int, default=512)
    parser.add_argument("--miss-rows-per-cell", type=int, default=512)
    parser.add_argument("--first-active-slot", type=int, default=0)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite immutable M0 callback profile")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    module = _load(args.artifact)
    # One exact fixed-fixture equivalence check proves that the profiling split
    # did not change the callback's settled counters or action materialization.
    fixture_kwargs = {
        "active_rows": args.active_rows_per_cell,
        "miss_rows": args.miss_rows_per_cell,
        "first_active_slot": args.first_active_slot,
    }
    total_fixture = _fixture(**fixture_kwargs)
    total_counters, _ = _run_total(module, total_fixture)
    split_fixture = _fixture(**fixture_kwargs)
    split_counters, _ = _split_callback(module, split_fixture)
    torch.cuda.synchronize(total_fixture["owner"].cuda_tile_arena.device)
    if not torch.equal(total_counters, split_counters) or not torch.equal(
        total_fixture["row_actions"], split_fixture["row_actions"]
    ):
        raise RuntimeError("profile split is not exact to the combined callback fixture")
    samples: dict[str, list[int]] = {}
    for _ in range(args.repeats):
        total_counters, total_timing = _run_total(
            module, _fixture(**fixture_kwargs)
        )
        if tuple(total_counters.shape) != (48, 8):
            raise RuntimeError("combined callback returned an invalid counter ABI")
        for name, value in total_timing.items():
            samples.setdefault(f"combined_callback.{name}", []).append(value)
        split_counters, split_timings = _split_callback(
            module, _fixture(**fixture_kwargs)
        )
        if tuple(split_counters.shape) != (48, 8):
            raise RuntimeError("split callback returned an invalid counter ABI")
        for stage, values in split_timings.items():
            for name, value in values.items():
                samples.setdefault(f"{stage}.{name}", []).append(value)
    args.output_root.mkdir(parents=True)
    receipt = {
        "schema_version": "eiel_b48_hapb_fixed_mlp_policy_callback_profile_v1",
        "purpose": "M0 measurement-only decomposition before fixed-MLP architecture selection",
        "artifact_sha256": _sha256(args.artifact),
        "artifact_source_commit": json.loads(args.manifest.read_text())["committed_source_snapshot"]["build_source_commit"],
        "host_boundary_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "profile_script_sha256": _sha256(Path(__file__).resolve()),
        "fixture": {
            "physical_cells": CELL_COUNT_V1,
            "active_rows_per_cell": args.active_rows_per_cell,
            "miss_rows_per_cell": args.miss_rows_per_cell,
            "first_active_slot": args.first_active_slot,
            "hot_prefix_misses_per_cell": max(
                0,
                min(args.first_active_slot + args.miss_rows_per_cell, HOT_PREFIX_ROWS_V1)
                - args.first_active_slot,
            ),
            "overflow_misses_per_cell": max(
                0,
                args.first_active_slot + args.miss_rows_per_cell - HOT_PREFIX_ROWS_V1,
            ),
            "owner_reinitialized_per_sample": True,
        },
        "exact_split_equivalence": {
            "counters_equal": True,
            "row_actions_equal": True,
        },
        "timing_ns": {name: _summary(values) for name, values in sorted(samples.items())},
        "scope_limit": "isolated prepared-policy callback; not an end-to-end tick, trajectory, or qualification result",
    }
    receipt_path = args.output_root / "m0_policy_callback_profile_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_sha256"] = _sha256(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
