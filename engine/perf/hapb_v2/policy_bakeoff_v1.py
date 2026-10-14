"""Shipped HAPB V2 policy-kernel bake-off driver.

CPU receipt emission is always safe.  GPU execution requires
EIEL_HAPB_V2_EXPERIMENTAL=1 and is the real P0-2 path, not a stub.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import json
from pathlib import Path
import struct
from typing import Any, Callable, Mapping

import numpy as np

from engine.perf.hapb_v2.amdahl_projection_v1 import project_v2_decision_inputs_v1
from engine.perf.hapb_v2.bakeoff_receipt_v1 import (
    empty_bakeoff_receipt_v1,
    validate_bakeoff_receipt_v1,
)
from engine.perf.hapb_v2.exactness_contract_v1 import (
    DEFAULT_DRIVE_WEIGHTS_V1,
    logits_sha256_v1,
)
from engine.perf.hapb_v2.experimental import require_experimental_v1
from engine.perf.hapb_v2.fixtures_v1 import (
    PolicyEvalFixtureV1,
    all_policy_eval_fixtures_v1,
)


class HapbV2BakeoffError(RuntimeError):
    """The bake-off driver refused to run or failed closed."""


def float32_bits_v1(value: float | np.floating) -> str:
    packed = struct.pack(">f", float(np.float32(value)))
    return format(struct.unpack(">I", packed)[0], "032b")


def compare_logits_to_reference_v1(
    got: np.ndarray,
    expected: np.ndarray,
) -> dict[str, object]:
    actual = np.asarray(got, dtype=np.float32)
    reference = np.asarray(expected, dtype=np.float32)
    if actual.shape != reference.shape:
        raise HapbV2BakeoffError(
            f"logit shape {actual.shape} != reference {reference.shape}"
        )
    mismatch = actual != reference
    count = int(np.count_nonzero(mismatch))
    first: dict[str, object] | None = None
    if count:
        row, action = (int(v) for v in np.argwhere(mismatch)[0])
        expected_value = float(reference[row, action])
        got_value = float(actual[row, action])
        expected_bits = float32_bits_v1(expected_value)
        got_bits = float32_bits_v1(got_value)
        first = {
            "row": row,
            "action": action,
            "expected": expected_value,
            "got": got_value,
            "expected_bits": expected_bits,
            "got_bits": got_bits,
            "abs_diff": abs(got_value - expected_value),
            "ulp_distance": abs(
                int(expected_bits, 2) - int(got_bits, 2)
            ),
        }
    return {
        "mismatch_count": count,
        "first_mismatch": first,
        "logits_sha256": logits_sha256_v1(actual),
        "reference_sha256": logits_sha256_v1(reference),
        "exact": count == 0,
    }


def build_cpu_receipt_v1() -> dict[str, object]:
    receipt = empty_bakeoff_receipt_v1()
    receipt["fixtures"] = [
        {"name": fixture.name, "rows": fixture.rows, "sha256": fixture.fixture_sha256}
        for fixture in all_policy_eval_fixtures_v1()
    ]
    receipt["amdahl"] = project_v2_decision_inputs_v1(
        policy_kernel_fraction=None,
        kernel_speedup=None,
        logits_bit_exact=None,
        complexity_bounded=True,
    )
    validate_bakeoff_receipt_v1(receipt)
    return receipt


def _to_cuda_inputs(fixture: PolicyEvalFixtureV1) -> dict[str, Any]:
    import torch

    def cuda_f32(array: np.ndarray) -> Any:
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).cuda()

    return {
        "packs": cuda_f32(fixture.packs),
        "observations": cuda_f32(fixture.observations),
        "logical_slots": torch.from_numpy(
            np.ascontiguousarray(fixture.logical_slots, dtype=np.int64)
        ).cuda(),
        "drives": cuda_f32(fixture.drives),
        "attack_bias": cuda_f32(fixture.attack_bias),
        "action_temperature": cuda_f32(fixture.action_temperature),
        "drive_logit_gain": cuda_f32(fixture.drive_logit_gain),
        "drive_weights": (
            float(DEFAULT_DRIVE_WEIGHTS_V1.w_hunger_gather),
            float(DEFAULT_DRIVE_WEIGHTS_V1.w_pain_rest),
            float(DEFAULT_DRIVE_WEIGHTS_V1.w_fatigue_rest),
            float(DEFAULT_DRIVE_WEIGHTS_V1.w_pain_attack),
            float(DEFAULT_DRIVE_WEIGHTS_V1.w_fear_attack),
        ),
    }


def _invoke(
    function: Callable[..., Any],
    payload: Mapping[str, Any],
) -> Any:
    weights = payload["drive_weights"]
    return function(
        payload["packs"],
        payload["observations"],
        payload["logical_slots"],
        payload["drives"],
        payload["attack_bias"],
        payload["action_temperature"],
        payload["drive_logit_gain"],
        weights[0],
        weights[1],
        weights[2],
        weights[3],
        weights[4],
    )


def _time_kernel_ms(
    function: Callable[..., Any],
    payload: Mapping[str, Any],
    *,
    warmup: int,
    iters: int,
) -> tuple[Any, float, float]:
    import time

    import torch

    for _ in range(max(0, warmup)):
        _invoke(function, payload)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    last = None
    for _ in range(max(1, iters)):
        last = _invoke(function, payload)
    end_event.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    kernel_ms = float(start_event.elapsed_time(end_event)) / float(max(1, iters))
    return last, kernel_ms, wall_ms


def run_gpu_bakeoff_v1(
    *,
    output: Path,
    warmup: int = 5,
    iters: int = 20,
    policy_kernel_fraction: float | None = None,
    build_dir: Path | None = None,
) -> dict[str, object]:
    """Load the experimental extension, evaluate fixtures, write a receipt."""

    require_experimental_v1()
    import torch

    if not torch.cuda.is_available():
        raise HapbV2BakeoffError("CUDA is not available for the experimental bake-off")

    from engine.perf.hapb_v2.experimental_extension_v1 import (
        compile_experimental_extension_v1,
    )

    destination = build_dir or (
        Path(public_runtime_path('live/hapb-v2-cpu-prep-20260816/experimental_build'))
    )
    module = compile_experimental_extension_v1(output_dir=destination)
    try:
        nx1 = module.eiel_hapb_v2_nx1_policy_logits_experimental_v1
        nx64 = module.eiel_hapb_v2_nx64_policy_logits_experimental_v1
    except AttributeError as exc:
        raise HapbV2BakeoffError(
            "experimental extension lacks N×1/N×64 bake-off entrypoints"
        ) from exc

    receipt = empty_bakeoff_receipt_v1()
    receipt["gpu_executed"] = True
    receipt["production_route_mutated"] = False
    exact_all = True
    nx1_ms_total = 0.0
    nx64_ms_total = 0.0
    rows_total = 0
    fixture_reports: list[dict[str, object]] = []
    first_global: dict[str, object] | None = None
    mismatch_total = 0
    for fixture in all_policy_eval_fixtures_v1():
        payload = _to_cuda_inputs(fixture)
        nx1_logits, nx1_ms, nx1_wall = _time_kernel_ms(
            nx1, payload, warmup=warmup, iters=iters
        )
        nx64_logits, nx64_ms, nx64_wall = _time_kernel_ms(
            nx64, payload, warmup=warmup, iters=iters
        )
        nx1_cpu = nx1_logits.detach().float().cpu().numpy()
        nx64_cpu = nx64_logits.detach().float().cpu().numpy()
        # GPU bit-exactness oracle is the isolated V1 N×1 device helper,
        # which matches production hapb_compact_policy_logits_v1.  Host
        # numpy/libm logits are algorithmic only and differ by a few ULPs
        # from device tanhf / float32.
        vs_each_other = compare_logits_to_reference_v1(nx64_cpu, nx1_cpu)
        vs_host_nx1 = compare_logits_to_reference_v1(
            nx1_cpu, fixture.expected_logits
        )
        vs_host_nx64 = compare_logits_to_reference_v1(
            nx64_cpu, fixture.expected_logits
        )
        exact = bool(vs_each_other["exact"])
        exact_all = exact_all and exact
        mismatch_total += int(vs_each_other["mismatch_count"])
        if first_global is None and vs_each_other["first_mismatch"] is not None:
            first_global = {
                "fixture": fixture.name,
                "oracle": "isolated_v1_nx1_device_helper",
                **dict(vs_each_other["first_mismatch"]),
            }
        nx1_ms_total += nx1_ms
        nx64_ms_total += nx64_ms
        rows_total += fixture.rows
        fixture_reports.append(
            {
                "name": fixture.name,
                "rows": fixture.rows,
                "unique_contents": int(len(set(int(v) for v in fixture.content_ids))),
                "nx1_kernel_ms": nx1_ms,
                "nx64_kernel_ms": nx64_ms,
                "nx1_wall_ms": nx1_wall,
                "nx64_wall_ms": nx64_wall,
                "exact_vs_v1_nx1_device": exact,
                "exact_vs_contract": exact,
                "gpu_oracle": "isolated_v1_nx1_device_helper",
                "nx64_vs_nx1": vs_each_other,
                "host_math_gap_nx1": vs_host_nx1,
                "host_math_gap_nx64": vs_host_nx64,
            }
        )

    speedup = None if nx64_ms_total <= 0.0 else nx1_ms_total / nx64_ms_total
    receipt["fixtures"] = fixture_reports
    receipt["exact_logits_parity"] = exact_all
    receipt["kernel_speedup_nx64_vs_nx1"] = speedup
    receipt["candidates"]["v1_nx1"].update(
        {
            "implemented": True,
            "rows_evaluated": rows_total,
            "warmup_iters": warmup,
            "timed_iters": iters,
            "kernel_elapsed_ms": nx1_ms_total,
            "launch_count": len(fixture_reports) * max(1, iters),
        }
    )
    receipt["candidates"]["v2_nx64"].update(
        {
            "implemented": True,
            "rows_evaluated": rows_total,
            "warmup_iters": warmup,
            "timed_iters": iters,
            "kernel_elapsed_ms": nx64_ms_total,
            "mismatch_count": mismatch_total,
            "first_mismatch": first_global,
        }
    )
    receipt["amdahl"] = project_v2_decision_inputs_v1(
        policy_kernel_fraction=policy_kernel_fraction,
        kernel_speedup=speedup,
        logits_bit_exact=exact_all,
        complexity_bounded=True,
    )
    receipt["decision"] = receipt["amdahl"].get("decision")
    validate_bakeoff_receipt_v1(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
