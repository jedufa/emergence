from __future__ import annotations

from engine.option_d.fixtures import build_d1_fixture_corpus


def test_current_source_fixture_corpus_regenerates_exactly() -> None:
    corpus = build_d1_fixture_corpus(source_binding="fixture-test-source")

    assert corpus["schema_version"] == "eiel_option_d_d1_fixture_corpus_v1"
    assert len(corpus["cases"]) == 8
    assert len(corpus["corpus_sha256"]) == 64
    assert {case["name"] for case in corpus["cases"]} == {
        "raw_seed_42_twist_boundary_625",
        "raw_seed_1_double_twist_1249",
        "getrandbits_boundaries_and_zero",
        "randbelow_rejection_boundaries",
        "gauss_odd_cache_live_257",
        "gauss_even_cache_empty_258",
        "mixed_long_continuation",
        "resident_predicated_branches",
    }
