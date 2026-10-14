from __future__ import annotations

from pathlib import Path

from perf.cxxbackbone1_rng_schedule import (
    authority_surfaces_payload,
    compare_rng_schedule_results,
    compile_rng_schedule_kernel,
    load_rng_schedule_backend,
    make_rng_schedule_fixture,
    result_digest,
    run_rng_schedule_python_reference,
)


def test_cxxbackbone1_rng_schedule_reference_is_deterministic() -> None:
    inputs = make_rng_schedule_fixture(streams=4, events=12, seed=301)
    a = run_rng_schedule_python_reference(inputs)
    b = run_rng_schedule_python_reference(inputs)

    assert result_digest(a) == result_digest(b)
    assert compare_rng_schedule_results(a, b)["ok"]


def test_cxxbackbone1_rng_schedule_native_exact_parity(tmp_path: Path) -> None:
    inputs = make_rng_schedule_fixture(streams=5, events=21, seed=302)
    compile_info = compile_rng_schedule_kernel(tmp_path / "build")

    assert compile_info["ok"], compile_info

    backend = load_rng_schedule_backend(compile_info["library_path"])
    expected = run_rng_schedule_python_reference(inputs)
    observed = backend(inputs)
    parity = compare_rng_schedule_results(expected, observed)
    payload = authority_surfaces_payload(parity)

    assert observed.executed_native is True
    assert observed.hidden_fallback is False
    assert parity["ok"], parity
    assert result_digest(expected) == result_digest(observed)
    assert payload["native_authority_surfaces"]["rng_state_indices"] is True
    assert payload["native_authority_surfaces"]["mutation_rng_draw_schedule"] is True
