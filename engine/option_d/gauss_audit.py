"""Bounded bitwise audit of CUDA Gaussian strategies against current CPython."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import json
import math
import platform
import random
import sys
from typing import Any

import torch

from .contracts import (
    CpythonRNGState,
    DrawOperation,
    DrawPredicate,
    DrawSchedule,
    GenerationResult,
    RNGStateBundle,
    canonical_json_bytes,
    float64_bits,
)
from .cpu import execute_cpu, state_from_seed
from .cuda import cuda_source_digest, execute_cuda, probe_cuda_math


def _neighbors(value: float) -> tuple[float, float, float]:
    return (
        math.nextafter(value, -math.inf),
        value,
        math.nextafter(value, math.inf),
    )


def _math_probe_inputs() -> tuple[tuple[float, float], ...]:
    angles = (
        0.0,
        *_neighbors(float.fromhex("0x1p-26")),
        *_neighbors(0.126),
        *_neighbors(float.fromhex("0x1.eb6p-1")),
        *_neighbors(math.pi / 2.0),
        *_neighbors(float.fromhex("0x1.368fdp+1")),
        *_neighbors(math.pi),
        math.nextafter(2.0 * math.pi, 0.0),
    )
    logarithms = (
        float.fromhex("0x1p-53"),
        math.nextafter(float.fromhex("0x1p-53"), math.inf),
        0.125,
        0.5,
        *tuple(value for value in _neighbors(0.9375) if 0.0 < value <= 1.0),
        math.nextafter(1.0, 0.0),
        1.0,
    )
    boundary = tuple(
        (angle, logarithm)
        for angle in angles
        for logarithm in logarithms
        if 0.0 <= angle < 2.0 * math.pi
    )
    generator = random.Random(0xD3A)
    random_domain = tuple(
        (generator.random() * (2.0 * math.pi), 1.0 - generator.random())
        for _ in range(65_536)
    )
    return boundary + random_domain


def _audit_math_probe(
    *,
    strategy: str,
    build_root: str | Path | None,
) -> dict[str, Any]:
    inputs = _math_probe_inputs()
    outputs = probe_cuda_math(
        inputs,
        gaussian_strategy=strategy,
        build_root=build_root,
    )
    functions = (math.sin, math.cos, math.log)
    names = ("sin", "cos", "log")
    mismatches = {name: 0 for name in names}
    first_mismatch: dict[str, Any] = {}
    for index, ((angle, logarithm), actual) in enumerate(
        zip(inputs, outputs, strict=True)
    ):
        arguments = (angle, angle, logarithm)
        for name, function, argument, actual_value in zip(
            names,
            functions,
            arguments,
            actual,
            strict=True,
        ):
            expected_bits = float64_bits(function(argument))
            actual_bits = float64_bits(actual_value)
            if expected_bits != actual_bits:
                mismatches[name] += 1
                first_mismatch.setdefault(
                    name,
                    {
                        "actual_bits": actual_bits,
                        "argument_bits": float64_bits(argument),
                        "expected_bits": expected_bits,
                        "index": index,
                        "xor": actual_bits ^ expected_bits,
                    },
                )
    return {
        "first_mismatch": first_mismatch,
        "mismatches": mismatches,
        "vectors": len(inputs),
    }


def _undo_right_xor(value: int, shift: int) -> int:
    result = value
    for _ in range(32 // shift + 1):
        result = value ^ (result >> shift)
    return result & 0xFFFFFFFF


def _undo_left_xor_mask(value: int, shift: int, mask: int) -> int:
    result = value
    for _ in range(32 // shift + 1):
        result = value ^ ((result << shift) & mask)
    return result & 0xFFFFFFFF


def _untemper(value: int) -> int:
    result = _undo_right_xor(value, 18)
    result = _undo_left_xor_mask(result, 15, 0xEFC60000)
    result = _undo_left_xor_mask(result, 7, 0x9D2C5680)
    return _undo_right_xor(result, 11)


def _boundary_state(*tempered_words: int) -> CpythonRNGState:
    words = tuple(_untemper(value) for value in tempered_words)
    return CpythonRNGState(words=words + (0,) * (624 - len(words)), index=0)


def _schedule_cases() -> tuple[
    tuple[
        str,
        tuple[tuple[DrawOperation, ...], ...],
        tuple[CpythonRNGState, ...] | None,
    ],
    ...,
]:
    mixed = tuple(
        operation
        for _ in range(257)
        for operation in (
            DrawOperation.random(),
            DrawOperation.gauss(),
            DrawOperation.getrandbits(0),
            DrawOperation.getrandbits(64),
            DrawOperation.gauss(mu=-3.25, sigma=0.125),
            DrawOperation.randbelow(17),
        )
    )
    predicated: list[DrawOperation] = []
    for _ in range(1024):
        source = len(predicated)
        predicated.append(DrawOperation.random())
        predicated.append(
            DrawOperation.gauss(
                sigma=0.005,
                predicate=DrawPredicate.random_lt(source, 0.2),
            )
        )
    return (
        (
            "cache_miss_hit",
            ((DrawOperation.gauss(), DrawOperation.gauss()),),
            None,
        ),
        (
            "odd_cache_live_257",
            (tuple(DrawOperation.gauss() for _ in range(257)),),
            None,
        ),
        (
            "even_cache_empty_258",
            (tuple(DrawOperation.gauss(mu=2.0, sigma=-0.25) for _ in range(258)),),
            None,
        ),
        ("long_mixed", (mixed,), None),
        ("predicated_gauss", (tuple(predicated),), None),
        (
            "multiworld_variable",
            tuple(
                tuple(
                    DrawOperation.gauss(mu=world * 0.25, sigma=1.0 / (world + 1))
                    for _ in range(1 + world * 17)
                )
                for world in range(8)
            ),
            None,
        ),
        (
            "boundary_angle_zero_uniform_zero",
            ((DrawOperation.gauss(), DrawOperation.gauss()),),
            (_boundary_state(0, 0, 0, 0),),
        ),
        (
            "boundary_angle_max_uniform_max",
            ((DrawOperation.gauss(), DrawOperation.gauss()),),
            (_boundary_state(*(0xFFFFFFFF,) * 4),),
        ),
        (
            "boundary_angle_zero_uniform_max",
            ((DrawOperation.gauss(), DrawOperation.gauss()),),
            (_boundary_state(0, 0, 0xFFFFFFFF, 0xFFFFFFFF),),
        ),
        (
            "boundary_angle_max_uniform_zero",
            ((DrawOperation.gauss(), DrawOperation.gauss()),),
            (_boundary_state(0xFFFFFFFF, 0xFFFFFFFF, 0, 0),),
        ),
    )


def _first_mismatch(
    expected: GenerationResult,
    actual: GenerationResult,
) -> dict[str, Any] | None:
    for world, (expected_values, actual_values) in enumerate(
        zip(expected.material.worlds, actual.material.worlds, strict=True)
    ):
        for operation, (expected_value, actual_value) in enumerate(
            zip(expected_values, actual_values, strict=True)
        ):
            if expected_value != actual_value:
                return {
                    "actual_bits": actual_value.value_bits,
                    "actual_executed": actual_value.executed,
                    "expected_bits": expected_value.value_bits,
                    "expected_executed": expected_value.executed,
                    "operation": operation,
                    "operator": expected_value.operator.name.lower(),
                    "world": world,
                    "xor": expected_value.value_bits ^ actual_value.value_bits,
                }
    return None


def audit_gaussian_strategy(
    *,
    strategy: str,
    source_binding: str,
    build_root: str | Path | None = None,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    totals = {
        "cache_mismatches": 0,
        "cos_mismatches": 0,
        "counter_mismatches": 0,
        "log_mismatches": 0,
        "material_mismatches": 0,
        "mt_state_mismatches": 0,
        "operations": 0,
        "sin_mismatches": 0,
    }
    math_probe = _audit_math_probe(strategy=strategy, build_root=build_root)
    for name, count in math_probe["mismatches"].items():
        totals[f"{name}_mismatches"] = count
    for case_index, (name, worlds, explicit_states) in enumerate(_schedule_cases()):
        bundle = RNGStateBundle(
            explicit_states
            or tuple(
                state_from_seed(0xD3000000 + case_index * 100 + world)
                for world in range(len(worlds))
            ),
            source_binding=source_binding,
        )
        schedule = DrawSchedule(worlds)
        expected = execute_cpu(bundle, schedule)
        actual = execute_cuda(
            bundle,
            schedule,
            gaussian_strategy=strategy,
            build_root=build_root,
        )
        first = _first_mismatch(expected, actual)
        material_match = first is None
        mt_state_match = all(
            (left.words, left.index) == (right.words, right.index)
            for left, right in zip(
                expected.successor.worlds,
                actual.successor.worlds,
                strict=True,
            )
        )
        cache_match = all(
            left.gauss_next_bits == right.gauss_next_bits
            for left, right in zip(
                expected.successor.worlds,
                actual.successor.worlds,
                strict=True,
            )
        )
        counters_match = all(
            left.counters == right.counters
            for left, right in zip(
                expected.successor.worlds,
                actual.successor.worlds,
                strict=True,
            )
        )
        operations = sum(len(world) for world in worlds)
        totals["operations"] += operations
        totals["material_mismatches"] += int(not material_match)
        totals["mt_state_mismatches"] += int(not mt_state_match)
        totals["cache_mismatches"] += int(not cache_match)
        totals["counter_mismatches"] += int(not counters_match)
        cases.append(
            {
                "cache_match": cache_match,
                "counters_match": counters_match,
                "first_material_mismatch": first,
                "initial_bundle_sha256": bundle.digest(),
                "material_match": material_match,
                "mt_state_match": mt_state_match,
                "name": name,
                "operations": operations,
                "schedule_sha256": schedule.digest(),
                "worlds": len(worlds),
            }
        )
    exact = all(value == 0 for key, value in totals.items() if key != "operations")
    payload: dict[str, Any] = {
        "cases": cases,
        "cuda": {
            "device_name": torch.cuda.get_device_name(),
            "device_capability": list(torch.cuda.get_device_capability()),
            "source_sha256": cuda_source_digest(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "decision": "exact" if exact else "mismatch",
        "host": {
            "libm_sha256": sha256(Path("/lib/x86_64-linux-gnu/libm.so.6").read_bytes()).hexdigest(),
            "platform": platform.platform(),
            "python": sys.version,
            "random_py_sha256": sha256(Path("/usr/lib/python3.14/random.py").read_bytes()).hexdigest(),
        },
        "math_probe": math_probe,
        "schema_version": "eiel_option_d_gaussian_strategy_audit_v1",
        "source_binding": source_binding,
        "strategy": strategy,
        "totals": totals,
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("libdevice", "software"), required=True)
    parser.add_argument("--source-binding", required=True)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = audit_gaussian_strategy(
        strategy=arguments.strategy,
        source_binding=arguments.source_binding,
        build_root=arguments.build_root,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(payload) + b"\n")
    print(json.dumps(payload["totals"], sort_keys=True))
    print(payload["decision"])
    return 0 if payload["decision"] == "exact" else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
