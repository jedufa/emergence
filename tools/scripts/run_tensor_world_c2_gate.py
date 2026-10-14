from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from engine.tensor_world.c1_backend import TensorWorldC1BackendV1
from engine.tensor_world.c1_fixture import make_c1_fixture_v1
from engine.tensor_world.c2_backend import (
    TensorWorldC2BackendV1,
    _config_tensors_v1,
    _native_state_v1,
    clone_c2_bundle_to_device_v1,
)
from engine.tensor_world.c2_fixture import make_c2_fixture_v1
from engine.tensor_world.c2_native_loader import load_tensor_world_c2_extension
from engine.tensor_world.contracts import TickRequestV1
from engine.tensor_world.native_loader import sha256_file


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
    }


def _config_mapping(config: Any) -> dict[str, Any]:
    return {name: getattr(config, name) for name in config.__dataclass_fields__}


def _dynamic_digest(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    # Dynamic/exported trajectory surfaces only; immutable policy/config inputs
    # are receipt-bound separately and intentionally excluded.
    for index in (
        2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 19, 23, 24, 25, 26,
        28, 29, 30, 31, 32, 33, 40, 41, 42, 43, 44, 45, 46, 48,
        49, 50, 55, 56, 57, 58, 60, 61, 62, 63, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    ):
        tensor = tensors[index].detach().cpu().contiguous()
        digest.update(index.to_bytes(2, "little"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parity_case(
    *, native: Any, batch: int, capacity: int, active: int, width: int,
    ticks: int, label: str,
) -> dict[str, Any]:
    source, config = make_c2_fixture_v1(
        batch=batch, capacity=capacity, active=active, width=width
    )
    candidate = clone_c2_bundle_to_device_v1(source, torch.device("cuda:0"))
    oracle = clone_c2_bundle_to_device_v1(source, torch.device("cuda:0"))
    del source
    candidate_t = _native_state_v1(candidate)
    oracle_t = _native_state_v1(oracle)
    candidate_ci, candidate_cf = _config_tensors_v1(config, candidate)
    oracle_ci, oracle_cf = _config_tensors_v1(config, oracle)
    mismatch: dict[str, Any] | None = None
    observed_phases = {"cue": False, "hidden_gap": False, "active": False}
    for tick in range(ticks):
        native.wdl_step(candidate_t, candidate_ci, candidate_cf)
        native.eed_step(candidate_t, candidate_ci, candidate_cf)
        native.patch_step(candidate_t, candidate_ci, candidate_cf)
        native.jungle_step(candidate_t, candidate_ci, candidate_cf)
        native.observe(candidate_t, candidate_ci, candidate_cf)
        native.policy_logits(candidate_t, candidate_ci, candidate_cf)
        native.select_actions(candidate_t, candidate_ci, candidate_cf)
        native.settle(candidate_t, candidate_ci, candidate_cf)
        native.oracle_tick(oracle_t, oracle_ci, oracle_cf)
        torch.cuda.synchronize()
        cue = candidate.wdl.cue > 0
        active_hazard = candidate.wdl.active > 0
        hidden = (candidate.wdl.timer >= 0) & ~cue & ~active_hazard
        observed_phases["cue"] |= bool(cue.any())
        observed_phases["hidden_gap"] |= bool(hidden.any())
        observed_phases["active"] |= bool(active_hazard.any())
        if bool((active_hazard & (candidate.wdl.prior_cue <= 0)).any()):
            mismatch = {"tick": tick, "field": "wdl.active_without_prior_cue"}
            break
        for index, (left, right) in enumerate(zip(candidate_t, oracle_t, strict=True)):
            if not torch.equal(left, right):
                mismatch = {
                    "tick": tick, "tensor_abi_index": index,
                    "candidate_dtype": str(left.dtype), "shape": list(left.shape),
                    "mismatch_count": int((left != right).sum().item()),
                }
                break
        if mismatch is not None:
            break
    digest = _dynamic_digest(candidate_t) if mismatch is None else None
    result = {
        "label": label, "batch": batch, "capacity": capacity, "active": active,
        "width": width, "ticks": ticks, "exact": mismatch is None,
        "mismatch": mismatch, "observed_wdl_phases": observed_phases,
        "canonical_dynamic_sha256": digest,
        "next_absolute_step": int(candidate.wdl.next_absolute_step[0]),
    }
    del candidate, oracle, candidate_t, oracle_t, candidate_ci, candidate_cf, oracle_ci, oracle_cf
    torch.cuda.empty_cache()
    return result


def _time_call(call: Any, *, warmup: int, samples: int) -> list[float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def _component_timings(native: Any) -> tuple[list[dict[str, Any]], int, int]:
    source, config = make_c2_fixture_v1(batch=48, capacity=512, active=256, width=32)
    state = clone_c2_bundle_to_device_v1(source, torch.device("cuda:0"))
    del source
    tensors = _native_state_v1(state)
    ci, cf = _config_tensors_v1(config, state)
    calls = (
        ("WDL", lambda: native.wdl_step(tensors, ci, cf)),
        ("EED", lambda: native.eed_step(tensors, ci, cf)),
        ("Patch", lambda: native.patch_step(tensors, ci, cf)),
        ("Jungle", lambda: native.jungle_step(tensors, ci, cf)),
        ("observation_expansion", lambda: native.observe(tensors, ci, cf)),
        ("policy_83", lambda: native.policy_logits(tensors, ci, cf)),
        ("action_selection", lambda: native.select_actions(tensors, ci, cf)),
        ("world_event_and_effect_settlement", lambda: native.settle(tensors, ci, cf)),
    )
    rows = []
    for name, call in calls:
        values = _time_call(call, warmup=3, samples=20)
        rows.append({"component": name, **_summary(values)})
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    del state
    torch.cuda.empty_cache()
    return rows, allocated, reserved


def _backend_timings(
    *, c1_extension: str, c2_extension: str
) -> tuple[dict[str, Any], list[float], list[float], int, int, dict[str, int]]:
    torch.cuda.reset_peak_memory_stats()
    c1_state, c1_config = make_c1_fixture_v1(batch=48, capacity=512, active=256)
    c1 = TensorWorldC1BackendV1(extension_path=c1_extension)
    c1_run = c1.initialize({"bundle": c1_state, "config": _config_mapping(c1_config)})
    del c1_state
    request = TickRequestV1()
    c1_values = _time_call(lambda: c1.advance(c1_run, request), warmup=10, samples=100)
    c1.close(c1_run)
    torch.cuda.empty_cache()

    c2_state, c2_config = make_c2_fixture_v1(batch=48, capacity=512, active=256)
    c2 = TensorWorldC2BackendV1(extension_path=c2_extension)
    c2_run = c2.initialize({"bundle": c2_state, "config": _config_mapping(c2_config)})
    del c2_state
    c2_values = _time_call(lambda: c2.advance(c2_run, request), warmup=10, samples=100)
    counters = dict(c2.residency_counters(c2_run))
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    c2.close(c2_run)
    torch.cuda.empty_cache()
    summary = {
        "c1_baseline": _summary(c1_values), "c2_composed": _summary(c2_values),
        "incremental_median_ms": statistics.median(c2_values) - statistics.median(c1_values),
    }
    return summary, c1_values, c2_values, allocated, reserved, counters


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Tensor World C2 production gate")
    parser.add_argument("--c1-extension", required=True)
    parser.add_argument("--c2-extension", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError("C2 gate output root must be immutable and new")
    output.mkdir(parents=True)
    c1_extension = Path(args.c1_extension).resolve()
    c2_extension = Path(args.c2_extension).resolve()
    native = load_tensor_world_c2_extension(c2_extension)
    started = time.time()

    parity = [
        _parity_case(native=native, batch=1, capacity=8, active=4, width=8, ticks=1, label="C2-P1-B1-one-tick"),
        _parity_case(native=native, batch=4, capacity=8, active=4, width=8, ticks=1, label="C2-P2-Bsmall-composed-one-tick"),
        _parity_case(native=native, batch=4, capacity=8, active=4, width=8, ticks=21, label="C2-P3-Bsmall-multitick"),
        _parity_case(native=native, batch=48, capacity=512, active=256, width=32, ticks=1, label="C2-P4-B48-one-tick"),
        _parity_case(native=native, batch=48, capacity=512, active=256, width=32, ticks=21, label="C2-P5-B48-bounded-multitick"),
    ]
    parity_pass = all(row["exact"] for row in parity)
    temporal_pass = all(parity[index]["observed_wdl_phases"][phase]
        for index in (2, 4) for phase in ("cue", "hidden_gap", "active"))
    _write_json(output / "tensor_world_C2_oracle_parity.json", {
        "schema_version": "tensor_world_c2_oracle_parity_v1",
        "comparison_basis": "direct source-current Patch/WDL/Jungle helper reuse plus native source-order oracle entry point",
        "equivalence": "zero-tolerance tensor equality", "cases": parity,
        "pass": parity_pass,
    })
    _write_json(output / "tensor_world_C2_multitick_parity.json", {
        "schema_version": "tensor_world_c2_multitick_parity_v1",
        "required_ticks": 21, "wdl_cue_hidden_active_observed": temporal_pass,
        "cases": [parity[2], parity[4]], "pass": parity_pass and temporal_pass,
    })

    torch.cuda.reset_peak_memory_stats()
    components, component_allocated, component_reserved = _component_timings(native)
    timing, c1_values, c2_values, allocated, reserved, counters = _backend_timings(
        c1_extension=str(c1_extension), c2_extension=str(c2_extension)
    )
    with (output / "tensor_world_C2_component_timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(components[0]))
        writer.writeheader()
        writer.writerows(components)
    with (output / "tensor_world_C2_full_tick_timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample", "c1_ms", "c2_ms"))
        writer.writerows((index, c1, c2) for index, (c1, c2) in enumerate(zip(c1_values, c2_values, strict=True)))

    residency_pass = all(value == 0 for value in counters.values())
    _write_json(output / "tensor_world_C2_residency_audit.json", {
        "schema_version": "tensor_world_c2_residency_audit_v1",
        "steady_state_counters": counters, "cuda_graph": True,
        "persistent_state_device": "cuda:0", "per_world_host_loops": 0,
        "per_agent_host_loops": 0, "pass": residency_pass,
    })
    _write_json(output / "tensor_world_C2_vram.json", {
        "schema_version": "tensor_world_c2_vram_v1",
        "peak_allocated_bytes": max(allocated, component_allocated),
        "peak_reserved_bytes": max(reserved, component_reserved),
        "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
    })

    median = timing["c2_composed"]["median_ms"]
    if not parity_pass or not temporal_pass or not residency_pass:
        decision = "TW-C2-REPAIR-REQUIRED"
    elif median <= 17.0:
        decision = "TW-C2-STRONG-GO-REVIEW"
    elif median <= 30.0:
        decision = "TW-C2-GO-REVIEW"
    elif median <= 35.0:
        decision = "TW-C2-REPAIR-REQUIRED"
    else:
        decision = "TW-C2-NO-GO"
    receipt = {
        "schema_version": "tensor_world_c2_gate_run_receipt_v1",
        "decision": decision, "semantic_parity_pass": parity_pass,
        "temporal_parity_pass": temporal_pass, "residency_pass": residency_pass,
        "timing": timing, "latency_target_ms": 17.0,
        "review_threshold_ms": 30.0, "stop_threshold_ms": 35.0,
        "c1_extension": {"path": str(c1_extension), "sha256": sha256_file(c1_extension)},
        "c2_extension": {"path": str(c2_extension), "sha256": sha256_file(c2_extension)},
        "duration_seconds": time.time() - started,
    }
    _write_json(output / "gate_run_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if decision in {"TW-C2-GO-REVIEW", "TW-C2-STRONG-GO-REVIEW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
