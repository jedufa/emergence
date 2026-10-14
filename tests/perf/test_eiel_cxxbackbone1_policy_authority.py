from __future__ import annotations

from pathlib import Path

from perf.cxxbackbone1_policy_authority import (
    authority_surfaces_payload,
    compare_policy_authority_results,
    compile_policy_authority_kernel,
    load_policy_authority_backend,
    make_policy_authority_fixture,
    result_digest,
    run_policy_authority_python_reference,
)


def test_cxxbackbone1_policy_reference_is_deterministic() -> None:
    inputs = make_policy_authority_fixture(rows=8, slots=11, seed=211)
    a = run_policy_authority_python_reference(inputs)
    b = run_policy_authority_python_reference(inputs)

    assert result_digest(a) == result_digest(b)
    assert compare_policy_authority_results(a, b, atol=0.0)["ok"]


def test_cxxbackbone1_policy_native_mixed_mlp_gru_hmem_cache_parity(tmp_path: Path) -> None:
    inputs = make_policy_authority_fixture(rows=9, slots=13, seed=212)
    compile_info = compile_policy_authority_kernel(tmp_path / "build")

    assert compile_info["ok"], compile_info

    backend = load_policy_authority_backend(compile_info["library_path"])
    expected = run_policy_authority_python_reference(inputs)
    observed = backend(inputs)
    parity = compare_policy_authority_results(expected, observed, atol=0.0)
    payload = authority_surfaces_payload(parity)

    assert observed.executed_native is True
    assert observed.hidden_fallback is False
    assert parity["ok"], parity
    assert result_digest(expected) == result_digest(observed)
    assert payload["native_authority_surfaces"]["action_cache"] is True
    assert payload["native_authority_surfaces"]["hmem"] is True
    assert payload["native_authority_surfaces"]["mlp_gru_policy_banks"] is True
    assert payload["native_authority_surfaces"]["policy_row_mappings"] is True
