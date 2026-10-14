"""Focused generic tests for exact natural source-state materialization."""

from __future__ import annotations

import hashlib
import struct
from collections import Counter
from pathlib import Path

import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from emergence_contracts import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
    SCIENCE_HEREDITARY_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_HEREDITARY_ATTRIBUTION_IDENTITY_MODE_V1,
    SCIENCE_HEREDITARY_ATTRIBUTION_SCHEMA_V1,
    SCIENCE_HEREDITARY_ATTRIBUTION_TRACKED_IDENTITY_MODE_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1,
    canonical_json_bytes,
    canonical_sha256,
    build_exact_natural_source_state_bank_v1,
    build_exact_natural_source_state_ref_v1,
    full_genome_state_id_v1,
    materialization_profile_id_v1,
    tracked_genotype_state_id_v1,
    validate_hereditary_attribution_configuration_v1,
)
from engine.campaigns.science_cuda_v2_native_world_v1 import (
    project_science_cuda_v2_native_world_v1,
)
from eiel_composition.exact_natural_source_state_recovery_v1 import (
    ExactNaturalSourceStateRecoveryError,
    enumerate_exact_natural_snapshot_candidates_v1,
    expand_packed_birth_provenance_v1,
    native_genome_bytes_from_row_v1,
    reconstruct_exact_natural_birth_v1,
    replayed_hereditary_agent_state_id_v1,
)
from eiel_composition.exact_natural_source_state_v1 import (
    ExactNaturalSourceStateMaterializationError,
    load_and_authenticate_source_state_bank_v1,
    materialize_exact_natural_source_initialization_v1,
    validate_materialized_exact_natural_source_initialization_v1,
)
from emergence_evidence.science_compact_world_atlas_v1 import (
    _ActiveGenomeV1,
    _record_hereditary_metrics_v1,
    _hereditary_snapshot_projection_v1,
    _snapshot_distribution,
)
from engine.perf.cuda_full_evolution_v2.layout import state_layout_v2
from engine.perf.cuda_full_evolution_v2.science_native_world_runtime_v1 import (
    ScienceNativeWorldRuntimeProjectionV1,
)


_NATIVE_GENOME = struct.Struct("<224b32h")


def _genome(seed: int) -> bytes:
    policy = [((seed + index) % 11) - 5 for index in range(224)]
    trait = [((seed + index) % 17) - 8 for index in range(32)]
    return _NATIVE_GENOME.pack(*policy, *trait)


def _fixture_ref(
    source_path: Path,
    *,
    agent_id: int,
    genome: bytes,
    tracked: str | None = None,
    state_class: str = "BRANCHABLE_EXACT",
    birth_tick: int = 0,
    parents: tuple[int, int] = (0, 0),
    kind: str = "tick_zero_genome_catalog_v1",
) -> dict[str, object]:
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    values = _NATIVE_GENOME.unpack(genome)
    lineage = "sha256:" + (f"{agent_id:064x}")
    agent_state = "sha256:" + (f"{agent_id + 100:064x}")
    return build_exact_natural_source_state_ref_v1(
        source={
            "campaign_id": "EIEL-EXACT-FIXTURE",
            "run_id": "fixture-run",
            "capture_index_path": str(source_path),
            "capture_index_sha256": {
                "algorithm": "sha256",
                "hex": source_digest,
            },
            "artifact_digests": [
                {
                    "kind": "fixture-capture",
                    "path": str(source_path),
                    "sha256": source_digest,
                }
            ],
        },
        world={
            "cell_id": f"fixture-cell-{agent_id}",
            "world_id": f"fixture-world-{agent_id}",
            "source_seed": agent_id,
        },
        absolute_tick=birth_tick,
        agent={
            "agent_id": agent_id,
            "agent_state_id": agent_state,
            "lineage_id": lineage,
            "lineage_depth": 0 if birth_tick == 0 else 1,
            "parent_a_id": parents[0],
            "parent_b_id": parents[1],
        },
        policy_bytes=genome[:224],
        trait_bytes=genome[224:],
        tracked_genotype_state_id=tracked,
        state_class=state_class,
        birth_provenance={
            "kind": kind,
            "birth_tick": birth_tick,
            "parent_agent_ids": list(parents),
            "packed_hex": "" if birth_tick == 0 else bytes(68).hex(),
            "mutation_step": 1,
        },
    )


def _fixture_bank(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    source_path = tmp_path / "capture.bin"
    source_path.write_bytes(b"authenticated fixture capture")
    genome_a = _genome(1)
    genome_b = _genome(2)
    tracked_a = tracked_genotype_state_id_v1(
        genome_a, policy_loci=(0,), trait_loci=(1,)
    )
    tracked_b = tracked_genotype_state_id_v1(
        genome_b, policy_loci=(0,), trait_loci=(1,)
    )
    ref_a = _fixture_ref(
        source_path, agent_id=1, genome=genome_a, tracked=tracked_a
    )
    ref_b = _fixture_ref(
        source_path, agent_id=2, genome=genome_b, tracked=tracked_b
    )
    bank = build_exact_natural_source_state_bank_v1(
        [ref_a, ref_b],
        bank_id="fixture-bank-v1",
        panels={
            "fixture-panel": {"A": ref_a["ref_id"], "B": ref_b["ref_id"]}
        },
    )
    bank_path = tmp_path / "source-state-bank.json"
    bank_bytes = canonical_json_bytes(bank)
    bank_path.write_bytes(bank_bytes)
    pointer = {
        "path": str(bank_path),
        "file_sha256": hashlib.sha256(bank_bytes).hexdigest(),
        "bank_sha256": bank["bank_sha256"]["hex"],
    }
    authenticated, binding = load_and_authenticate_source_state_bank_v1(pointer)
    return authenticated, binding, {"A": ref_a, "B": ref_b}


def test_tick_zero_reference_is_full_state_and_source_authenticated(tmp_path: Path) -> None:
    bank, _binding, refs = _fixture_bank(tmp_path)
    ref = refs["A"]
    assert ref["absolute_tick"] == 0
    assert len(ref["genome"]["policy_hex"]) == 448
    assert len(ref["genome"]["trait_hex"]) == 128
    assert ref["genome_state_id"] == full_genome_state_id_v1(
        bytes.fromhex(ref["genome"]["policy_hex"] + ref["genome"]["trait_hex"])
    )
    assert len(bank["refs"]) == 2


def test_authenticated_replay_recovers_later_natural_child_without_editing() -> None:
    parent_a = {"policy": [0] * 224, "trait": [0] * 32}
    parent_b = {"policy": [0] * 224, "trait": [0] * 32}
    packed = bytes(68)
    expanded = expand_packed_birth_provenance_v1(packed)
    assert len(expanded) == 132
    replay = reconstruct_exact_natural_birth_v1(
        parent_a=parent_a,
        parent_b=parent_b,
        packed_birth_provenance=packed,
        mutation_step=1,
    )
    assert replay["child_genome_state_id"] == full_genome_state_id_v1(
        replay["child_genome_bytes"]
    )
    assert replay["mutation_origin_count"] == 0
    replayed_id = replayed_hereditary_agent_state_id_v1(
        world_id="world-fixture",
        agent_id=7,
        parent_agent_ids=(3, 4),
        birth_tick=11,
        genome_state_id=replay["child_genome_state_id"],
    )
    assert replayed_id.startswith("sha256:")
    with pytest.raises(ExactNaturalSourceStateRecoveryError, match="expected"):
        reconstruct_exact_natural_birth_v1(
            parent_a=parent_a,
            parent_b=parent_b,
            packed_birth_provenance=packed,
            mutation_step=1,
            expected_child_genome_state_id="sha256:" + "f" * 64,
        )


def test_source_digest_binding_rejects_stale_capture(tmp_path: Path) -> None:
    _bank, _binding, _refs = _fixture_bank(tmp_path)
    source_path = tmp_path / "capture.bin"
    source_path.write_bytes(b"changed after authentication")
    bank_path = tmp_path / "source-state-bank.json"
    raw = bank_path.read_bytes()
    pointer = {
        "path": str(bank_path),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "bank_sha256": __import__("json").loads(raw)["bank_sha256"]["hex"],
    }
    with pytest.raises(ExactNaturalSourceStateMaterializationError, match="digest"):
        load_and_authenticate_source_state_bank_v1(pointer)


def test_ambiguous_tracked_projection_returns_all_full_candidates() -> None:
    first = bytearray(_genome(3))
    second = bytearray(first)
    second[10] += 1
    first = bytes(first)
    second = bytes(second)
    rows = []
    for encoded in (first, second):
        values = _NATIVE_GENOME.unpack(encoded)
        rows.append(
            {
                "genome_state_id": full_genome_state_id_v1(encoded),
                "policy": list(values[:224]),
                "trait": list(values[224:]),
            }
        )
    tracked = tracked_genotype_state_id_v1(
        first, policy_loci=(0,), trait_loci=()
    )
    candidates = enumerate_exact_natural_snapshot_candidates_v1(
        catalog_rows=rows,
        active_rows=[
            {"agent_id": 1, "genome_state_id": rows[0]["genome_state_id"]},
            {"agent_id": 2, "genome_state_id": rows[1]["genome_state_id"]},
        ],
        tracked_identity=tracked,
        policy_loci=(0,),
        trait_loci=(),
    )
    assert len(candidates) == 2
    assert {item["genome_state_id"] for item in candidates} == {
        row["genome_state_id"] for row in rows
    }


def test_mixed_exact_materialization_is_repeatably_content_bound(tmp_path: Path) -> None:
    bank, binding, refs = _fixture_bank(tmp_path)
    initialization = {
        "initial_population": 4,
        "initial_live_count": 4,
        "founder_profiles": ["legacy-placeholder"],
        "exact_founder_composition": [
            {"founder_profile_id": "legacy-placeholder", "count": 4}
        ],
        "spatial_placement_profile": "fixture-placement",
        "standing_variation_profile": "fixture-variation",
        "exact_natural_source_composition": [
            {
                "source_state_ref": "fixture-panel",
                "source_state_role": "A",
                "count": 2,
            },
            {
                "source_state_ref": "fixture-panel",
                "source_state_role": "B",
                "count": 2,
            },
        ],
    }
    first = materialize_exact_natural_source_initialization_v1(
        initialization, bank=bank, bank_binding=binding
    )
    second = materialize_exact_natural_source_initialization_v1(
        initialization, bank=bank, bank_binding=binding
    )
    assert first == second
    assert initialization["founder_profiles"] == ["legacy-placeholder"]
    assert first["founder_profiles"] == [
        materialization_profile_id_v1(refs["A"]["genome_state_id"]),
        materialization_profile_id_v1(refs["B"]["genome_state_id"]),
    ]
    assert [row["count"] for row in first["exact_founder_composition"]] == [2, 2]
    assert {
        row["ref_id"] for row in first["exact_natural_source_states"]
    } == {refs["A"]["ref_id"], refs["B"]["ref_id"]}
    validated = validate_materialized_exact_natural_source_initialization_v1(first)
    assert {row["ref_id"] for row in validated} == {
        refs["A"]["ref_id"],
        refs["B"]["ref_id"],
    }


@pytest.mark.skipif(torch is None, reason="requires torch for resident-state projection")
def test_exact_natural_runtime_copies_all_hereditary_loci_and_keeps_legacy_isolated(
    tmp_path: Path,
) -> None:
    bank, binding, refs = _fixture_bank(tmp_path)
    exact_initialization = materialize_exact_natural_source_initialization_v1(
        {
            "initial_population": 256,
            "initial_live_count": 256,
            "founder_profiles": ["legacy-placeholder"],
            "exact_founder_composition": [
                {"founder_profile_id": "legacy-placeholder", "count": 256}
            ],
            "spatial_placement_profile": "fixture-placement",
            "standing_variation_profile": "fixture-variation",
            "exact_natural_source_composition": [
                {"source_state_ref": "fixture-panel", "source_state_role": "A", "count": 128},
                {"source_state_ref": "fixture-panel", "source_state_role": "B", "count": 128},
            ],
        },
        bank=bank,
        bank_binding=binding,
    )
    legacy_initialization = {
        "initial_population": 256,
        "initial_live_count": 256,
        "founder_profiles": ["founder-neutral-v1"],
        "exact_founder_composition": [
            {"founder_profile_id": "founder-neutral-v1", "count": 256}
        ],
        "spatial_placement_profile": "fixture-placement",
        "standing_variation_profile": "fixture-variation",
    }

    def native_control(initialization: dict[str, object], index: int) -> dict[str, object]:
        semantic = {
            "world": {
                "world_profile_id": "generic-runtime-fixture-v1",
                "geometry": {"width": 32, "height": 32},
                "resident_capacity": 512,
            },
            "initialization": initialization,
            "dimension_bindings": {},
            "control_bindings": [],
            "seed_bindings": {"founder": 7000 + index, "initialization": 7100 + index},
            "time": {
                "authority": "absolute_tick_v1",
                "start_next_tick": 0,
                "total_ticks": 2,
                "checkpoint_ticks": [0, 2],
            },
            "initial_state_identity": {
                "algorithm": "sha256",
                "hex": canonical_sha256({"fixture_world": index}).hex,
            },
        }
        science = {
            "test_id": "GENERIC-EXACT-FOUNDER",
            "test_version": "1",
            "study_id": "EXACT-FOUNDER-RUNTIME-FIXTURE",
            "arm_id": f"arm-{index}",
            "replicate": index,
        }
        return project_science_cuda_v2_native_world_v1(
            {"semantic_inputs": semantic, "science": science}
        ).to_dict()

    projection = ScienceNativeWorldRuntimeProjectionV1.from_controls(
        [
            native_control(exact_initialization, 0),
            native_control(legacy_initialization, 1),
        ]
    )
    layout = state_layout_v2()
    state = torch.zeros(
        (2, layout.state_bytes_per_world), dtype=torch.uint8
    )
    traits = state[:, 158976:191744].view(torch.int16).reshape(2, 2, 16, 512)
    traits[0, :, 0, :256] = 77
    traits[1, :, 0, :256] = 88

    binding_result = projection.apply_founders_to_state(state)

    assert binding_result["hereditary_identity_application_mode"] == "mixed_per_world_v1"
    assert binding_result["preserved_trait_loci_by_world"] == [[], [0]]
    assert traits[1, :, 0, :256].unique().tolist() == [88]

    policy = state[:, 44288:158976].view(torch.int8).reshape(2, 2, 112, 512)
    traits = state[:, 158976:191744].view(torch.int16).reshape(2, 2, 16, 512)
    actual_ids = []
    for slot in range(256):
        encoded = (
            policy[0, :, :, slot].contiguous().numpy().tobytes()
            + traits[0, :, :, slot].contiguous().numpy().tobytes()
        )
        actual_ids.append(full_genome_state_id_v1(encoded))
    expected_ids = [
        row["genome_state_id"]
        for row in exact_initialization["exact_natural_source_states"]
        for _ in range(128)
    ]
    assert Counter(actual_ids) == Counter(expected_ids)


def test_full_hereditary_contract_is_pre_gpu_admission_boundary() -> None:
    focal = ["sha256:" + "1" * 64, "sha256:" + "2" * 64]
    binding = {
        "schema_version": SCIENCE_HEREDITARY_ATTRIBUTION_SCHEMA_V1,
        "event_kinds": list(SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1),
        "identity_mode": SCIENCE_HEREDITARY_ATTRIBUTION_IDENTITY_MODE_V1,
        "tracked_identity_mode": SCIENCE_HEREDITARY_ATTRIBUTION_TRACKED_IDENTITY_MODE_V1,
        "lineage_mode": "parent_ancestry_v1",
        "focal_tracked_genotype_state_ids": focal,
        "denominator_fields": list(SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1),
    }
    configuration = {"hereditary_attribution": binding}
    required = set(SCIENCE_HEREDITARY_ATTRIBUTION_CAPABILITIES_V1)
    assert validate_hereditary_attribution_configuration_v1(
        configuration, required_observations=required
    ) == ()
    missing_full_identity = required - {"hereditary.full_identity"}
    errors = validate_hereditary_attribution_configuration_v1(
        configuration, required_observations=missing_full_identity
    )
    assert "required_observations:hereditary.full_identity" in errors


def test_full_identity_projection_keeps_parent_lineage_and_focal_frequency() -> None:
    genome = _genome(9)
    focal = tracked_genotype_state_id_v1(genome, policy_loci=(0,), trait_loci=())
    active = {
        1: _ActiveGenomeV1(
            encoded=genome,
            generation=0,
            mutation_lineage=False,
            born=False,
            lineage_id="sha256:" + "a" * 64,
            parent_a_id=0,
            parent_b_id=0,
        ),
        2: _ActiveGenomeV1(
            encoded=genome,
            generation=1,
            mutation_lineage=False,
            born=True,
            lineage_id="sha256:" + "b" * 64,
            parent_a_id=1,
            parent_b_id=1,
        ),
    }
    projection = _hereditary_snapshot_projection_v1(
        {
            "hereditary_contract": {},
            "hereditary_focal_ids": (focal,),
            "policy_loci": (0,),
            "trait_loci": (),
        },
        active,
    )
    assert projection is not None
    assert projection["focal_live_counts"] == {focal: 2}
    assert projection["focal_live_total"] == 2
    assert projection["live_full_genome_state_counts"][0]["count"] == 2


def test_capture_genome_row_requires_complete_native_state() -> None:
    with pytest.raises(ExactNaturalSourceStateRecoveryError, match="complete"):
        native_genome_bytes_from_row_v1({"policy": [0], "trait": [0]})


def test_hereditary_metrics_accept_signed_energy_delta_only() -> None:
    genome = _genome(12)
    record = _ActiveGenomeV1(
        encoded=genome,
        generation=0,
        mutation_lineage=False,
        born=False,
    )
    metrics = {name: 0 for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1}
    metrics["energy_delta_total"] = -17
    reconstruction = {
        "hereditary_contract": {},
        "policy_loci": (),
        "trait_loci": (),
        "hereditary_metric_totals": {},
        "hereditary_metric_agent_counts": {},
        "hereditary_completed_agents": 0,
    }
    _record_hereditary_metrics_v1(reconstruction, record, metrics)
    tracked = full_genome_state_id_v1(genome)
    assert reconstruction["hereditary_metric_totals"][tracked][
        "energy_delta_total"
    ] == -17
    assert reconstruction["hereditary_completed_agents"] == 1


def test_snapshot_lineage_marker_does_not_overwrite_reconstructed_depth() -> None:
    genome = _genome(15)
    state_id = full_genome_state_id_v1(genome)
    record = _ActiveGenomeV1(
        encoded=genome,
        generation=3,
        mutation_lineage=False,
        born=True,
        parent_a_id=2,
        parent_b_id=3,
    )
    reconstruction = {
        "snapshots": {
            100: {
                "agent_state_ids": {7: state_id},
                "agent_metadata": {
                    7: {
                        "agent_state_id": "sha256:" + "c" * 64,
                        "lineage_id": "sha256:" + "d" * 64,
                        "lineage_depth": 1,
                        "parent_a_id": 2,
                        "parent_b_id": 3,
                        "absolute_birth_tick": 20,
                    }
                },
                "agent_metrics": {},
            }
        },
        "milestones": {100: {}},
        "active": {7: record},
        "policy_loci": (),
        "trait_loci": (),
        "hereditary_contract": {},
        "hereditary_focal_ids": (),
    }
    result = {
        "integrity": {
            "genome_state_identity_checks": 0,
            "hereditary_full_identity_checks": 0,
        }
    }

    _snapshot_distribution(
        reconstruction,
        result=result,
        absolute_tick=100,
    )

    assert record.generation == 3
    assert reconstruction["milestones"][100]["genome_reconstruction"][
        "maximum_active_lineage_depth"
    ] == 3
