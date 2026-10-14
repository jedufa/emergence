"""Localize HAPB-FRAG RNGV2 first-turnover 5021 without skipping status 21 globally."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TURNOVER = ROOT / "engine/perf/cuda_architecture_b_flattened_turnover_v1_kernels.cuh"
BINDING = ROOT / "engine/perf/cuda_architecture_b_flattened_controller_v1_binding.cpp"
WDL = ROOT / "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"
SCALAR = ROOT / "engine/perf/cuda_architecture_b_persistent_turnover_scalar_genome.cuh"

# Live 128f diagnostic: leftover MT words were present, index=592, gauss
# tape width 0, all 48 cells published 5021 at the first generation boundary.
LIVE_FIRST_TURNOVER_5021 = {
    "python_random_index": 592,
    "python_random_raw_draws_before_pairing": 0,
    "gauss_tape_capacity": 0,
    "leftover_pair_raw_draws": 200,
    "published_status": 5021,
    "phase": 5,
    "substatus": 21,
}


def _assignments(path: Path) -> list[tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    hits: list[tuple[int, str]] = []
    for index, line in enumerate(lines, start=1):
        if "kStatusInvalidPythonRandomState" in line and (
            "=" in line or "return" in line
        ):
            hits.append((index, line.strip()))
    return hits


def test_enumerate_invalid_python_random_state_producers() -> None:
    turnover = _assignments(TURNOVER)
    scalar = _assignments(SCALAR)
    assert turnover, "turnover kernel must still name status 21"
    assert scalar, "persistent scalar helper still owns a V1 status-21 check"
    # Ordinary-birth helpers also alias 21; they are not the first-turnover path.
    assert all("kStatusInvalidPythonRandomState" in line for _, line in turnover)


def test_turnover_status_21_tape_checks_are_v2_guarded() -> None:
    source = TURNOVER.read_text(encoding="utf-8")
    pre, post = source.split("counters[kScalarSemanticDrawsAdded]", 1)
    assert "if (execution.rng_semantics_version == 2)" in pre
    assert "Leftover CPython choice-RNG draws are not a V1 Gaussian-tape cursor." in pre
    assert "execution.gauss_tape_capacity" in pre
    assert "if (execution.rng_semantics_version == 2)" in post
    # V1 tape overflow remains a real 21; V2 leftover pair draws must not.
    v1_post = post.split("if (execution.rng_semantics_version == 2)", 1)[1]
    assert "> execution.gauss_tape_capacity" in v1_post
    assert "kStatusInvalidPythonRandomState" in v1_post


def test_first_turnover_leftover_pair_draws_must_not_be_gauss_tape_overflow() -> None:
    """Would have failed on the live 128f 5021 before the post-scalar V2 guard."""

    capacity = LIVE_FIRST_TURNOVER_5021["gauss_tape_capacity"]
    leftover = LIVE_FIRST_TURNOVER_5021["leftover_pair_raw_draws"]
    index = LIVE_FIRST_TURNOVER_5021["python_random_index"]
    assert 0 <= index <= 624
    assert leftover > capacity
    v1_would_fail = leftover > capacity
    assert v1_would_fail is True
    source = TURNOVER.read_text(encoding="utf-8")
    post = source.split("turnover_python_raw_draws[0] = rng.raw_draws;", 1)[1]
    guarded = post.split("cross_control.committed_counts[cell]", 1)[0]
    assert "execution.rng_semantics_version == 2" in guarded
    assert "turnover_python_raw_draws[0] < 0" in guarded
    # The V2 branch must not use gauss_tape_capacity as the leftover-draw bound.
    v2_branch = guarded.split("execution.rng_semantics_version == 2", 1)[1]
    v2_branch = v2_branch.split("} else if", 1)[0]
    assert "gauss_tape_capacity" not in v2_branch


def test_zero_gauss_tape_infers_rng_semantics_version_two_into_turnover() -> None:
    binding = BINDING.read_text(encoding="utf-8")
    assert "ordinary_birth_config.gauss_tape_capacity = state.at(39).size(1);" in binding
    assert (
        "if (ordinary_birth_config.gauss_tape_capacity == 0) {\n"
        "            ordinary_birth_config.rng_semantics_version = 2;\n"
        "        }"
    ) in binding
    wdl = WDL.read_text(encoding="utf-8")
    assert "ordinary_birth_config.rng_semantics_version," in wdl
    turnover = TURNOVER.read_text(encoding="utf-8")
    assert re.search(
        r"int32_t rng_semantics_version;\n    uint32_t world_uid;",
        turnover,
    )


def test_persistent_scalar_status_21_is_not_the_v2_diploid_turnover_path() -> None:
    source = TURNOVER.read_text(encoding="utf-8")
    assert "commit_turnover_fixed_mlp_child_scalar_genome_with_ranked_inheritance_v2_serial" in source
    assert "if (statuses[kScalarStatus] == scalar::kStatusOk && diploid_mode)" in source
    assert "diploid::stage_child_v2(" in source
    scalar = SCALAR.read_text(encoding="utf-8")
    assert "kStatusInvalidPythonRandomState = 21" in scalar
