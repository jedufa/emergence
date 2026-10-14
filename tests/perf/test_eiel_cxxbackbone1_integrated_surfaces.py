from __future__ import annotations

from pathlib import Path

from perf.cxxbackbone1_integrated_surfaces import (
    authority_surfaces_payload,
    compare_integrated_surfaces_results,
    compile_integrated_surfaces_kernel,
    load_integrated_surfaces_backend,
    make_integrated_surfaces_fixture,
    result_digest,
    run_integrated_surfaces_python_reference,
)


def test_cxxbackbone1_integrated_surfaces_reference_is_deterministic() -> None:
    fixture = make_integrated_surfaces_fixture(slots=20, cells=4, seed=811)
    a = run_integrated_surfaces_python_reference(fixture)
    b = run_integrated_surfaces_python_reference(fixture)

    assert result_digest(a) == result_digest(b)
    assert compare_integrated_surfaces_results(a, b)["ok"]


def test_cxxbackbone1_integrated_surfaces_native_parity(tmp_path: Path) -> None:
    fixture = make_integrated_surfaces_fixture(slots=24, cells=4, seed=812)
    compile_info = compile_integrated_surfaces_kernel(tmp_path / "build")

    assert compile_info["ok"], compile_info

    backend = load_integrated_surfaces_backend(compile_info["library_path"])
    expected = run_integrated_surfaces_python_reference(fixture)
    observed = backend(fixture)
    parity = compare_integrated_surfaces_results(expected, observed)
    payload = authority_surfaces_payload(parity)

    assert observed.executed_native is True
    assert observed.hidden_fallback is False
    assert parity["ok"], parity
    assert result_digest(expected) == result_digest(observed)
    assert payload["native_authority_surfaces"]["lineage_founder_state"] is True
    assert payload["native_authority_surfaces"]["lifecycle_reproduction_mutation_state"] is True
    assert payload["native_authority_surfaces"]["full_resident_checkpoint"] is True
    assert payload["native_authority_surfaces"]["live_lifecycle_execution"] is True
    assert payload["native_authority_surfaces"]["public_output_compatibility"] is True
    assert payload["native_authority_surfaces"]["multi_cell_batching"] is True
