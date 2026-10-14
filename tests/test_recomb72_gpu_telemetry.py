from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.recomb72_gpu_telemetry import (
    ANCESTRY_SUM_ABS_TOLERANCE,
    BOUNDARY_SCHEMA_VERSION,
    CHECKPOINT_GENERATIONS,
    INHERITANCE_EVENT_SUMMARY_SCHEMA_VERSION,
    SOURCE_COMPLETENESS,
    Recomb72GpuTelemetryError,
    build_checkpoint_catalog,
    build_generation_rows,
    build_gpu_telemetry,
    checkpoint_state_sha256,
    checkpoint_tensor_sha256,
    materialize_scientific_chronicle,
    validate_generation_rows,
)
from engine.evolution.recomb72_inheritance import (
    AUDITED_LOCI,
    founder_packages,
)
from engine.perf.recomb72_gpu_native import SCHEMA_VERSION as NATIVE_STATE_SCHEMA


CELL_ID = "cell-gpu-001"
MODE = "current_authoritative"
PATTERN = "correlated_multi_trait_packages"


def _boundaries(tmp_path: Path) -> list[dict]:
    package_a, package_b = founder_packages(PATTERN)
    package_a = {
        locus: torch.tensor(value, dtype=torch.float32).item()
        for locus, value in package_a.items()
    }
    package_b = {
        locus: torch.tensor(value, dtype=torch.float32).item()
        for locus, value in package_b.items()
    }
    packages = [package_a] * 32 + [package_b] * 32
    founder_ids = [f"founder_{index:02d}" for index in range(64)]
    ancestry = [
        [1.0 if founder == agent else 0.0 for founder in range(64)]
        for agent in range(64)
    ]
    rows = []
    for generation in range(31):
        inheritance_events = (
            {
                "schema_version": INHERITANCE_EVENT_SUMMARY_SCHEMA_VERSION,
                "source": "native_gpu_event_tensors",
                "event_count": 0,
                "loci": {},
                "full_package_preservation_count": 0,
                "full_package_preservation_fraction": 0.0,
                "not_applicable": True,
            }
            if generation == 0
            else {
                "schema_version": INHERITANCE_EVENT_SUMMARY_SCHEMA_VERSION,
                "source": "native_gpu_event_tensors",
                "event_count": 32,
                "loci": {
                    locus: {
                        "parent_pair_distance_mean": 0.5,
                        "child_to_parent_a_distance_mean": 0.25,
                        "child_to_parent_b_distance_mean": 0.25,
                        "child_to_parental_midpoint_distance_mean": 0.1,
                        "exact_parent_count": 16,
                        "strictly_between_count": 16,
                    }
                    for locus in AUDITED_LOCI
                },
                "full_package_preservation_count": 8,
                "full_package_preservation_fraction": 0.25,
            }
        )
        checkpoint = None
        if generation in CHECKPOINT_GENERATIONS:
            checkpoint_path = tmp_path / f"checkpoint_gen{generation:06d}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": NATIVE_STATE_SCHEMA,
                "locus_names": list(AUDITED_LOCI),
                "tensors": {
                    "loci": torch.zeros((1, 64, 6), dtype=torch.float32),
                    "alive": torch.ones((1, 64), dtype=torch.bool),
                    "agent_ids": torch.arange(64, dtype=torch.int64).reshape(1, 64),
                    "ancestry": torch.eye(64, dtype=torch.float32).reshape(1, 64, 64),
                    "next_id": torch.tensor([64], dtype=torch.int64),
                    "generation": torch.tensor([generation], dtype=torch.int64),
                    "seeds": torch.tensor([101], dtype=torch.int64),
                    "modes": torch.tensor([0], dtype=torch.int32),
                    "patterns": torch.tensor([2], dtype=torch.int32),
                    "status": torch.tensor([0], dtype=torch.int32),
                },
            }
            torch.save(payload, checkpoint_path)
            checkpoint = {
                "schema_version": "recomb72_gpu_checkpoint_v1",
                "payload_schema_version": NATIVE_STATE_SCHEMA,
                "generation": generation,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "tensor_sha256": checkpoint_tensor_sha256(payload),
                "state_sha256": checkpoint_state_sha256(payload),
                "source_projection_sha256": canonical_sha256(
                    {"generation": generation, "kind": "projection"}
                ),
                "fresh_owner_cuda_restore": True,
            }
        rows.append(
            {
                "schema_version": BOUNDARY_SCHEMA_VERSION,
                "cell_id": CELL_ID,
                "generation": generation,
                "mode": MODE,
                "pattern": PATTERN,
                "paired_seed": 101,
                "mechanics_digest_sha256": canonical_sha256(
                    {"generation": generation, "kind": "mechanics"}
                ),
                "state": {
                    "alive_agent_ids": list(range(64)),
                    "locus_values": {
                        locus: [package[locus] for package in packages]
                        for locus in AUDITED_LOCI
                    },
                    "founder_ids": founder_ids,
                    "founder_ancestry_weights": ancestry,
                },
                "population": (
                    {
                        "before_cull": 64,
                        "eligible_parents": 64,
                        "selected_parent_events": 0,
                        "pairings": 0,
                        "children": 0,
                        "after_refill": 64,
                        "failed_refill_events": 0,
                        "ordinary_births": 0,
                        "mutation_events": 0,
                    }
                    if generation == 0
                    else {
                        "before_cull": 64,
                        "eligible_parents": 32,
                        "selected_parent_events": 64,
                        "pairings": 32,
                        "children": 32,
                        "after_refill": 64,
                        "failed_refill_events": 0,
                        "ordinary_births": 0,
                        "mutation_events": 0,
                    }
                ),
                "parent_contribution_counts": (
                    {} if generation == 0 else {str(index): 2 for index in range(32)}
                ),
                "inheritance_events": inheritance_events,
                "integrity": {
                    "genotype_blind_cull": True,
                    "all_survivors_eligible": True,
                    "actual_ticks_per_generation": 0 if generation == 0 else 80,
                    "no_fallback": True,
                    "native_gpu_owner": True,
                },
                **({"checkpoint": checkpoint} if checkpoint is not None else {}),
            }
        )
    return rows


def test_gpu_boundaries_project_exact_six_locus_and_ancestry_rows(
    tmp_path: Path,
) -> None:
    rows = build_generation_rows(_boundaries(tmp_path))

    validate_generation_rows(rows)
    assert [row["generation"] for row in rows] == list(range(31))
    assert all(set(row["loci"]) == set(AUDITED_LOCI) for row in rows)
    assert rows[0]["packages"]["intact_package_fraction"] == 1.0
    assert rows[0]["ancestry"]["active_founder_ancestry_count"] == 64
    assert rows[0]["ancestry"]["effective_founder_ancestry_count"] == pytest.approx(
        64.0
    )
    assert rows[0]["integrity"]["founder_split"] == {"a": 32, "b": 32}
    assert rows[0]["inheritance_events"]["not_applicable"] is True
    assert rows[1]["inheritance_events"]["event_count"] == 32
    assert (
        rows[1]["inheritance_events"]["loci"]["pred_lr"][
            "child_to_parental_midpoint_distance_mean"
        ]
        == 0.1
    )
    assert rows[30]["inheritance_events"]["full_package_preservation_count"] == 8
    assert rows[30]["population"]["children"] == 32


def test_gpu_telemetry_rejects_incomplete_or_noncanonical_boundaries(
    tmp_path: Path,
) -> None:
    missing = _boundaries(tmp_path / "missing")
    missing.pop(17)
    with pytest.raises(Recomb72GpuTelemetryError, match="exactly G0 through G30"):
        build_generation_rows(missing)

    ancestry = _boundaries(tmp_path / "ancestry")
    ancestry[8]["state"]["founder_ancestry_weights"][0] = [1.0] * 63
    with pytest.raises(Recomb72GpuTelemetryError, match="complete 64-founder"):
        build_generation_rows(ancestry)

    out_of_bounds = _boundaries(tmp_path / "bounds")
    out_of_bounds[2]["state"]["locus_values"]["exploration_bias"][0] = 2.0
    with pytest.raises(Recomb72GpuTelemetryError, match="out-of-bounds"):
        build_generation_rows(out_of_bounds)

    wrong_founder = _boundaries(tmp_path / "wrong-founder")
    wrong_founder[0]["state"]["locus_values"]["exploration_bias"][0] += 0.001
    with pytest.raises(Recomb72GpuTelemetryError, match="exact frozen 32/32"):
        build_generation_rows(wrong_founder)


def test_gpu_telemetry_accepts_float32_ancestry_sum_roundoff_only(
    tmp_path: Path,
) -> None:
    normalized = torch.arange(1, 65, dtype=torch.float32)
    normalized /= normalized.sum()
    weights = normalized.tolist()
    assert abs(sum(weights) - 1.0) > 1e-12

    rounded = _boundaries(tmp_path / "rounded")
    rounded[29]["state"]["founder_ancestry_weights"] = copy.deepcopy(
        rounded[29]["state"]["founder_ancestry_weights"]
    )
    rounded[29]["state"]["founder_ancestry_weights"][0] = weights
    build_generation_rows(rounded)

    corrupted = _boundaries(tmp_path / "corrupted")
    corrupted[29]["state"]["founder_ancestry_weights"] = copy.deepcopy(
        corrupted[29]["state"]["founder_ancestry_weights"]
    )
    corrupted_weights = list(weights)
    corrupted_weights[0] += 2.0 * ANCESTRY_SUM_ABS_TOLERANCE
    corrupted[29]["state"]["founder_ancestry_weights"][0] = corrupted_weights
    with pytest.raises(Recomb72GpuTelemetryError, match="sum to one"):
        build_generation_rows(corrupted)

    negative = _boundaries(tmp_path / "negative")
    negative[29]["state"]["founder_ancestry_weights"] = copy.deepcopy(
        negative[29]["state"]["founder_ancestry_weights"]
    )
    negative_weights = list(weights)
    negative_weights[0] = -1e-9
    negative_weights[1] += weights[0] + 1e-9
    negative[29]["state"]["founder_ancestry_weights"][0] = negative_weights
    with pytest.raises(Recomb72GpuTelemetryError, match="non-negative"):
        build_generation_rows(negative)


def test_generation_row_validation_rejects_post_projection_tampering(
    tmp_path: Path,
) -> None:
    rows = build_generation_rows(_boundaries(tmp_path))
    rows[5]["loci"]["exploration_bias"]["variance_population"] = 0.0

    with pytest.raises(Recomb72GpuTelemetryError, match="row digest mismatch"):
        validate_generation_rows(rows)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("event_count", 31, "exactly 32 children"),
        ("full_package_preservation_fraction", float("nan"), "must be finite"),
    ],
)
def test_gpu_telemetry_rejects_invalid_gpu_inheritance_event_summary(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    boundaries = _boundaries(tmp_path)
    boundaries[7]["inheritance_events"][field] = value

    with pytest.raises(Recomb72GpuTelemetryError, match=message):
        build_generation_rows(boundaries)


def test_gpu_telemetry_rejects_out_of_range_inheritance_counts(
    tmp_path: Path,
) -> None:
    boundaries = _boundaries(tmp_path)
    boundaries[3]["inheritance_events"]["loci"]["pred_lr"][
        "strictly_between_count"
    ] = 17

    with pytest.raises(
        Recomb72GpuTelemetryError,
        match="inheritance counts exceed 32 events",
    ):
        build_generation_rows(boundaries)


def test_gpu_checkpoint_catalog_binds_all_declared_recomb_milestones(
    tmp_path: Path,
) -> None:
    catalog = build_checkpoint_catalog(_boundaries(tmp_path))

    assert catalog["generations"] == [0, 5, 10, 20, 30]
    assert sorted(catalog["checkpoints"], key=int) == ["0", "5", "10", "20", "30"]
    for generation, row in catalog["checkpoints"].items():
        assert row["generation"] == int(generation)
        assert row["schema_version"] == "recomb72_gpu_checkpoint_v1"
        assert row["payload_schema_version"] == NATIVE_STATE_SCHEMA
        assert Path(row["checkpoint_path"]).is_absolute()
        assert set(row["tensor_sha256"]) == {
            "loci", "alive", "agent_ids", "ancestry", "next_id",
            "generation", "seeds", "modes", "patterns", "status",
        }
        assert len(row["state_sha256"]) == 64
        assert row["fresh_owner_cuda_restore"] is True
        assert len(row["mechanics_digest_sha256"]) == 64


def test_gpu_checkpoint_catalog_rejects_architecture_b_mislabel(
    tmp_path: Path,
) -> None:
    boundaries = _boundaries(tmp_path)
    boundaries[5]["checkpoint"] = {
        "schema_version": "checkpoint_bundle_v2",
        "bundle_kind": "architecture_b",
        "generation": 5,
        "bundle_path": str(tmp_path / "architecture-b"),
    }

    with pytest.raises(
        Recomb72GpuTelemetryError,
        match="not a RECOMB-72 GPU checkpoint v1",
    ):
        build_checkpoint_catalog(boundaries)


def test_reporting_toggle_cannot_change_mechanics_digest_or_input(
    tmp_path: Path,
) -> None:
    boundaries = _boundaries(tmp_path)
    before = copy.deepcopy(boundaries)

    enabled = build_gpu_telemetry(boundaries, reporting_enabled=True)
    disabled = build_gpu_telemetry(boundaries, reporting_enabled=False)

    assert boundaries == before
    assert (
        enabled["mechanics_digest_chain_sha256"]
        == disabled["mechanics_digest_chain_sha256"]
    )
    assert len(enabled["generation_metrics_rows"]) == 31
    assert disabled["generation_metrics_rows"] == []
    assert enabled["checkpoint_catalog"] == disabled["checkpoint_catalog"]


def test_materializes_truthful_native_resident_scientific_full_chronicle(
    tmp_path: Path,
) -> None:
    boundaries = _boundaries(tmp_path / "bundles")
    telemetry = build_gpu_telemetry(boundaries, reporting_enabled=True)

    receipt = materialize_scientific_chronicle(
        telemetry["generation_metrics_rows"],
        run_root=tmp_path / "cell",
        campaign_id="recomb72-gpu-test",
        cell_id=CELL_ID,
        mechanics_digest_chain_sha256=telemetry[
            "mechanics_digest_chain_sha256"
        ],
    )

    chronicle_root = Path(receipt["chronicle_root"])
    manifest = json.loads(
        (chronicle_root / "chronicle_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (chronicle_root / "scientific_chronicle_summary.json").read_text(
            encoding="utf-8"
        )
    )
    events = [
        json.loads(line)
        for line in (chronicle_root / "formal_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    recomb_events = [
        event
        for event in events
        if event["event_type"] == "recomb72_native_generation_boundary"
    ]

    assert manifest["profile"] == "scientific_full"
    assert manifest["finalized"] is True
    assert manifest["source_completeness"] == SOURCE_COMPLETENESS
    assert summary["last_generation"] == 30
    assert summary["write_failures"] == 0
    assert len(recomb_events) == 31
    assert recomb_events[0]["measurements"]["generation_metrics"]["generation"] == 0
    assert (
        recomb_events[1]["measurements"]["generation_metrics"][
            "inheritance_events"
        ]["event_count"]
        == 32
    )
    assert receipt["source_completeness"] == SOURCE_COMPLETENESS
