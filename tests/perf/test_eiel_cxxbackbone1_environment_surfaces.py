from __future__ import annotations

from pathlib import Path

from perf.cxxbackbone1_environment_surfaces import (
    authority_surfaces_payload,
    compare_environment_surfaces_results,
    compile_environment_surfaces_kernel,
    load_environment_surfaces_backend,
    make_environment_surfaces_fixture,
    result_digest,
    run_environment_surfaces_python_reference,
)


def test_cxxbackbone1_environment_reference_is_deterministic() -> None:
    inputs = make_environment_surfaces_fixture(cells=64, seed=611)
    a = run_environment_surfaces_python_reference(inputs)
    b = run_environment_surfaces_python_reference(inputs)

    assert result_digest(a) == result_digest(b)
    assert compare_environment_surfaces_results(a, b, atol=0.0)["ok"]


def test_cxxbackbone1_environment_native_patch_jungle_wdl_parity(tmp_path: Path) -> None:
    inputs = make_environment_surfaces_fixture(cells=96, seed=612)
    compile_info = compile_environment_surfaces_kernel(tmp_path / "build")

    assert compile_info["ok"], compile_info

    backend = load_environment_surfaces_backend(compile_info["library_path"])
    expected = run_environment_surfaces_python_reference(inputs)
    observed = backend(inputs)
    parity = compare_environment_surfaces_results(expected, observed, atol=0.0)
    payload = authority_surfaces_payload(parity)

    assert observed.executed_native is True
    assert observed.hidden_fallback is False
    assert parity["ok"], parity
    assert result_digest(expected) == result_digest(observed)
    assert payload["native_authority_surfaces"]["patch_world_state"] is True
    assert payload["native_authority_surfaces"]["jungle_state"] is True
    assert payload["native_authority_surfaces"]["wdl_state"] is True
