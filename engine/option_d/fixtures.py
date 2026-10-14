"""Regenerate current-runtime D1 fixture corpora for later GPU parity gates."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable

from .contracts import (
    CpythonRNGState,
    DrawOperation,
    DrawPredicate,
    DrawSchedule,
    Operator,
    RNGStateBundle,
    canonical_json_bytes,
    canonical_sha256,
    float64_bits,
    float64_from_bits,
)
from .cpu import execute_cpu, state_from_seed


def _python_state_payload(state: tuple[Any, ...]) -> dict[str, Any]:
    frozen = CpythonRNGState.from_python_state(state)
    return {
        "gauss_next_bits": frozen.gauss_next_bits,
        "index": frozen.index,
        "words": list(frozen.words),
    }


def _official_execute(
    initial: CpythonRNGState,
    operations: Iterable[DrawOperation],
) -> tuple[list[int], list[bool], tuple[Any, ...]]:
    rng = random.Random()
    rng.setstate(initial.to_python_state())
    outputs: list[int] = []
    executed: list[bool] = []
    for operation in operations:
        predicate = operation.predicate
        if predicate is not None:
            if not executed[predicate.source_operation]:
                raise RuntimeError("fixture predicate source was skipped")
            source_bits = outputs[predicate.source_operation]
            if not (
                float64_from_bits(source_bits)
                < float64_from_bits(predicate.threshold_bits)
            ):
                outputs.append(0)
                executed.append(False)
                continue
        if operation.operator is Operator.RAW32:
            outputs.append(rng.getrandbits(32))
        elif operation.operator is Operator.RANDOM:
            outputs.append(float64_bits(rng.random()))
        elif operation.operator is Operator.GETRANDBITS:
            outputs.append(rng.getrandbits(operation.argument))
        elif operation.operator is Operator.RANDBELOW:
            outputs.append(rng._randbelow(operation.argument))
        elif operation.operator is Operator.GAUSS:
            outputs.append(
                float64_bits(
                    rng.gauss(
                        float64_from_bits(operation.mu_bits),
                        float64_from_bits(operation.sigma_bits),
                    )
                )
            )
        else:  # pragma: no cover - frozen contract validates first
            raise AssertionError(operation)
        executed.append(True)
    return outputs, executed, rng.getstate()


def _case(
    name: str,
    seed: int | str,
    operations: tuple[DrawOperation, ...],
    *,
    source_binding: str,
) -> dict[str, Any]:
    initial = state_from_seed(seed)
    schedule = DrawSchedule((operations,))
    candidate = execute_cpu(
        RNGStateBundle((initial,), source_binding=source_binding),
        schedule,
    )
    expected_outputs, expected_executed, expected_state = _official_execute(
        initial, operations
    )
    actual_outputs = [value.value_bits for value in candidate.material.worlds[0]]
    actual_executed = [value.executed for value in candidate.material.worlds[0]]
    if actual_outputs != expected_outputs:
        raise RuntimeError(f"D1 fixture candidate output mismatch in {name}")
    if actual_executed != expected_executed:
        raise RuntimeError(f"D1 fixture candidate branch mismatch in {name}")
    if candidate.successor.worlds[0].to_python_state() != expected_state:
        raise RuntimeError(f"D1 fixture candidate successor mismatch in {name}")
    expected_state_payload = _python_state_payload(expected_state)
    return {
        "candidate_counters": candidate.successor.worlds[0].counters.to_dict(),
        "candidate_receipt_sha256": candidate.receipt.digest(),
        "expected_outputs_bits": expected_outputs,
        "expected_outputs_sha256": canonical_sha256(expected_outputs),
        "expected_executed": expected_executed,
        "expected_executed_sha256": canonical_sha256(expected_executed),
        "expected_successor": expected_state_payload,
        "expected_successor_sha256": canonical_sha256(expected_state_payload),
        "name": name,
        "operations": [operation.to_dict() for operation in operations],
        "schedule_sha256": schedule.digest(),
        "seed": seed,
    }


def build_d1_fixture_corpus(
    *,
    source_binding: str,
    random_py_path: str | Path = "/usr/lib/python3.14/random.py",
) -> dict[str, Any]:
    random_path = Path(random_py_path)
    executable_path = Path("/usr/bin/python3.14")
    cases = [
        _case(
            "raw_seed_42_twist_boundary_625",
            42,
            tuple(DrawOperation.raw32() for _ in range(625)),
            source_binding=source_binding,
        ),
        _case(
            "raw_seed_1_double_twist_1249",
            1,
            tuple(DrawOperation.raw32() for _ in range(1249)),
            source_binding=source_binding,
        ),
        _case(
            "getrandbits_boundaries_and_zero",
            314159,
            tuple(
                DrawOperation.getrandbits(width)
                for width in (0, 1, 2, 7, 8, 15, 16, 31, 32, 33, 47, 63, 64)
            ),
            source_binding=source_binding,
        ),
        _case(
            "randbelow_rejection_boundaries",
            271828,
            tuple(
                DrawOperation.randbelow(upper)
                for _ in range(32)
                for upper in (1, 2, 3, 5, 7, 255, 256, 257, 65537, (1 << 63) - 1)
            ),
            source_binding=source_binding,
        ),
        _case(
            "gauss_odd_cache_live_257",
            1618033,
            tuple(DrawOperation.gauss(mu=-0.25, sigma=1.75) for _ in range(257)),
            source_binding=source_binding,
        ),
        _case(
            "gauss_even_cache_empty_258",
            1414213,
            tuple(DrawOperation.gauss(mu=2.0, sigma=0.125) for _ in range(258)),
            source_binding=source_binding,
        ),
        _case(
            "mixed_long_continuation",
            "option-d-current-source",
            tuple(
                operation
                for _ in range(257)
                for operation in (
                    DrawOperation.random(),
                    DrawOperation.getrandbits(0),
                    DrawOperation.randbelow(17),
                    DrawOperation.gauss(),
                    DrawOperation.getrandbits(64),
                    DrawOperation.gauss(mu=3.5, sigma=-0.25),
                )
            ),
            source_binding=source_binding,
        ),
        _case(
            "resident_predicated_branches",
            "option-d-device-branches",
            tuple(
                operation
                for branch in range(257)
                for operation in (
                    DrawOperation.random(),
                    DrawOperation.gauss(
                        sigma=0.005,
                        predicate=DrawPredicate.random_lt(branch * 4, 0.2),
                    ),
                    DrawOperation.random(),
                    DrawOperation.randbelow(
                        2,
                        predicate=DrawPredicate.random_lt(branch * 4 + 2, 0.05),
                    ),
                )
            ),
            source_binding=source_binding,
        ),
    ]
    payload = {
        "cases": cases,
        "current_authority": {
            "executable": str(executable_path),
            "executable_sha256": sha256(executable_path.read_bytes()).hexdigest(),
            "python_version": sys.version,
            "random_py": str(random_path),
            "random_py_sha256": sha256(random_path.read_bytes()).hexdigest(),
        },
        "exactness": "integer and binary64 payload, successor words/index/cache",
        "schema_version": "eiel_option_d_d1_fixture_corpus_v1",
        "source_binding": source_binding,
    }
    payload["corpus_sha256"] = canonical_sha256(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-binding", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = build_d1_fixture_corpus(source_binding=arguments.source_binding)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(payload) + b"\n")
    print(
        json.dumps(
            {
                "cases": len(payload["cases"]),
                "corpus_sha256": payload["corpus_sha256"],
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
