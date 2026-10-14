from __future__ import annotations

from pathlib import Path

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.evolution.genome_upgrade_v2_hereditary_capacity_gate_v1 import (
    DECISION_A,
    DECISION_B,
    DECISION_D,
    EVIDENCE_SCHEMA_VERSION,
    METRIC_FIELDS,
    PREREGISTRATION_SHA256,
    evaluate_hereditary_capacity_v1,
    preregistration_payload_v1,
)
from tools.scripts.genome_upgrade_v2_hereditary_capacity_gate_v1 import (
    materialize,
)


def _authority(**updates: str) -> dict[str, str]:
    value = {
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "source_commit": "1" * 40,
        "package_id_sha256": "2" * 64,
        "package_receipt_sha256": "3" * 64,
        "artifact_sha256": "4" * 64,
        "execution_route_sha256": "5" * 64,
    }
    return {**value, **updates}


def _row(repeat: int, **updates: object) -> dict[str, object]:
    metrics = {
        "allele_count": 12,
        "genotype_count": 8,
        "heterozygosity_q10": 384,
        "hidden_allele_count": 4,
        "retained_hidden_allele_count": 3,
        "fixed_allele_count": 1,
        "lost_allele_count": 0,
        "retained_package_count": 2,
        "linkage_disequilibrium_abs_q10": 128,
        "homogenization_generation": None,
        "expression_cost_q10": 12,
        "runtime_cost_q10": 7,
    }
    assert set(metrics) == set(METRIC_FIELDS)
    value = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "authority": _authority(),
        "repeat_index": repeat,
        "metrics": metrics,
        "causal_effect_observed": True,
        "cost_tradeoff_observed": True,
        "mutation_reachable": True,
        "immediate_cap_saturation": False,
        "no_fallback": True,
        "no_jit": True,
    }
    value.update(updates)
    return {**value, "evidence_sha256": content_sha256(value)}


def test_hereditary_capacity_terminal_decisions_and_identity() -> None:
    rows = [_row(0), _row(1)]
    assert (
        evaluate_hereditary_capacity_v1(rows, authority=_authority())["decision"]
        == DECISION_A
    )

    negative = [_row(0), _row(1)]
    negative[1]["metrics"] = {**negative[1]["metrics"], "heterozygosity_q10": 0}
    unsigned = dict(negative[1])
    unsigned.pop("evidence_sha256")
    negative[1]["evidence_sha256"] = content_sha256(unsigned)
    assert (
        evaluate_hereditary_capacity_v1(negative, authority=_authority())["decision"]
        == DECISION_B
    )

    mixed = [_row(0), _row(1, authority=_authority(artifact_sha256="6" * 64))]
    assert (
        evaluate_hereditary_capacity_v1(mixed, authority=_authority())["decision"]
        == DECISION_D
    )


def test_hereditary_capacity_preregistration_freezes_production_design() -> None:
    payload = preregistration_payload_v1()

    assert content_sha256(payload) == PREREGISTRATION_SHA256
    assert payload["campaign_design"] == {
        "world": "patch",
        "overlays": [],
        "policy_family": "fixed_mlp",
        "selection_mode": "standard",
        "grid_size": 16,
        "initial_population": 64,
        "resident_capacity": 256,
        "founder_placement": "balanced",
        "genome_profile": "genome_upgrade_v2_fixed_q10_v1",
        "founder_profile": "pleiotropic_tradeoffs",
        "mutation_rate_q10": 20,
        "mutation_step_q10": 64,
        "ecology_seeds": [101, 202],
        "task_seeds": [3003],
        "backend": "cuda_resident",
        "logical_cells": 2,
        "physical_batch_maximum": 2,
        "generations": 3,
        "ticks_per_generation": 80,
        "checkpoint_generations": [0, 3],
        "digest_every_generation": True,
        "checkpoint_storage_mode": "disk",
        "fresh_repeats": 0,
        "continuation_cells": 0,
    }


def test_hereditary_capacity_cli_outputs_are_immutable(tmp_path: Path) -> None:
    output = tmp_path / "result"
    manifest = materialize(
        authority=_authority(),
        evidence=[_row(0), _row(1)],
        output_directory=output,
    )
    assert manifest["decision"] == DECISION_A
    assert len(list(output.iterdir())) == 3
    with pytest.raises(FileExistsError):
        materialize(
            authority=_authority(),
            evidence=[_row(0), _row(1)],
            output_directory=output,
        )
