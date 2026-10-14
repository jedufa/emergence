from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GETRANDBITS_ZERO_SURFACES = (
    "engine/perf/cuda_architecture_b_persistent_birth_scalar_genome.cuh",
    "engine/perf/native/cuda_architecture_b_lifecycle.cu",
    "engine/perf/cuda_architecture_b_persistent_turnover_plan.cuh",
    "engine/perf/native/cuda_architecture_b_turnover.cu",
)


def _getrandbits_body(source: str) -> str:
    match = re.search(
        r"__device__\s+__forceinline__\s+uint64_t\s+getrandbits\s*\(",
        source,
    )
    assert match is not None, "getrandbits definition is missing"
    open_brace = source.find("{", match.end())
    assert open_brace != -1, "getrandbits body is missing"

    depth = 0
    for index in range(open_brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace + 1 : index]
    raise AssertionError("getrandbits body is unterminated")


def test_cpython_getrandbits_zero_returns_zero_without_raw_draw() -> None:
    for relative in GETRANDBITS_ZERO_SURFACES:
        source = (ROOT / relative).read_text(encoding="utf-8")
        body = _getrandbits_body(source)

        zero_guard = body.find("bit_count == 0")
        assert zero_guard != -1, relative

        return_zero = body.find("return 0;", zero_guard)
        assert return_zero != -1, relative

        small_width_path = body.find("bit_count <= 32")
        first_raw_draw = body.find("raw32()")
        assert small_width_path != -1, relative
        assert first_raw_draw != -1, relative
        assert zero_guard < return_zero < small_width_path <= first_raw_draw, relative
