from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("tools/scripts/recomb1_inheritance_variance_audit.py").resolve()
SPEC = importlib.util.spec_from_file_location("recomb1_inheritance_variance_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_analytic_variance_and_covariance_predictions_are_explicit() -> None:
    models = AUDIT.analytic_predictions()["models"]

    assert models["exact_midpoint"]["marginal_variance_multiplier"] == 0.5
    assert models["uniform_interpolation"]["marginal_variance_multiplier"] == 2.0 / 3.0
    assert models["parent_copy_per_locus"]["marginal_variance_multiplier"] == 1.0
    assert models["parent_copy_per_locus"]["cross_locus_covariance_multiplier"] == 0.5
    assert models["linkage_block_copy"]["marginal_variance_multiplier"] == 1.0
    assert models["linkage_block_copy"]["cross_locus_covariance_multiplier"] == 1.0
    assert models["analysis_only_diploid"]["scope"].startswith("analysis-only")


def test_fixture_matrix_is_fixed_and_complete() -> None:
    fixtures = AUDIT.fixture_definitions()

    assert set(fixtures) == {
        "single_trait_bipolar",
        "single_trait_three_cluster",
        "single_trait_continuous_two_cluster",
        "single_trait_bounded_edge",
        "multi_trait_two_locus",
        "multi_trait_four_locus",
    }
    assert all(len(fixture["initial"]) == 96 for fixture in fixtures.values())
    assert len(AUDIT.MODELS) == 6


def test_audit_is_exact_repeatable_and_exposes_required_metrics() -> None:
    first = AUDIT.run_audit(seeds=(7100,))
    second = AUDIT.run_audit(seeds=(7100,))

    assert AUDIT.canonical_bytes(first) == AUDIT.canonical_bytes(second)
    assert first["all_checks_pass"] is True
    history = first["fixtures"]["multi_trait_four_locus"]["parent_copy_per_locus"][
        "7100"
    ]["history"]
    assert len(history) == 11
    final = history[-1]
    assert {
        "mean_by_locus",
        "variance_by_locus",
        "extreme_fraction",
        "center_fraction",
        "bimodality_proxy",
        "unique_phenotype_count",
        "parent_offspring_distance",
        "lineage_conditioned_mean_by_locus",
        "covariance_locus_0_1",
        "correlation_locus_0_1",
        "package_mixing",
    } <= final.keys()


def test_checkpoint_resume_and_novelty_measurement_gates() -> None:
    payload = AUDIT.run_audit(seeds=(7100,))

    proofs = payload["determinism"]["checkpoint_resume_proofs"]
    assert set(proofs) == set(AUDIT.MODELS)
    assert all(proof["checkpoint_generation"] == 2 for proof in proofs.values())
    assert all(proof["byte_identical_continuation"] for proof in proofs.values())

    novelty = payload["scalar_mutation_novelty_survival"]
    assert novelty["measurement_generations"] == [1, 2, 5, 10]
    assert all(
        set(novelty["results"][model]["7100"]) == {"1", "2", "5", "10"}
        for model in AUDIT.MODELS
    )
    assert all(
        "novelty_survival" in novelty["results"][model]["7100"]["1"]
        for model in AUDIT.MODELS
    )


def test_cli_writes_machine_readable_payload(tmp_path: Path) -> None:
    output = tmp_path / "recomb1.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), "--seeds", "7100"],
        check=True,
    )

    decoded = json.loads(output.read_text(encoding="utf-8"))
    assert decoded["schema_version"] == AUDIT.SCHEMA_VERSION
    assert decoded["rules"]["population_size"] == 96
    assert decoded["rules"]["generations"] == 10
    assert decoded["all_checks_pass"] is True
