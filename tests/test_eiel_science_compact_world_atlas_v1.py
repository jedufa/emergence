from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
import os
from pathlib import Path
import random
import shutil
import signal
import struct
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    SchemaVersion,
    ScienceWorldBundleV1,
    ScienceWorldRecordV1,
    SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_SCHEMA_V1,
    SCIENCE_HEREDITARY_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_HEREDITARY_ATTRIBUTION_IDENTITY_MODE_V1,
    SCIENCE_HEREDITARY_ATTRIBUTION_SCHEMA_V1,
    SCIENCE_HEREDITARY_ATTRIBUTION_TRACKED_IDENTITY_MODE_V1,
    canonical_sha256,
)
from emergence_evidence.science_compact_world_atlas_v1 import (
    COMPACT_WORLD_ATLAS_REQUIRED_CAPABILITIES,
    CompactWorldAtlasAsyncSidecarV1,
    CompactWorldAtlasIncrementalReducerV1,
    _HereditaryBirthRecordSpoolV1,
    NATIVE_REDUCER_ARTIFACT_USE_ENGINEERING_REAL_USE_V1,
    _canonical_snapshot_manifest_paths_v1,
    reduce_compact_science_world_atlas_v1,
    validate_hereditary_birth_record_artifact_v1,
)
from emergence_evidence.native_compact_reducer_v1 import (
    build_native_compact_reducer_v1,
    load_native_compact_reducer_v1,
)
from emergence_evidence.science_strategy_attribution_v1 import (
    analyze_compact_science_strategy_attribution_streaming_v1,
)
from engine.native_runtime.cuda_v2_compact_science_capture_v1 import (
    reconstruct_compact_birth_genome_v1,
    write_compact_science_dispatch_v1,
)
from engine.perf.cuda_full_evolution_v2.direct_sparse_runtime import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
    COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1,
    CudaFullEvolutionCompactScienceCaptureV1,
)
from engine.perf.cuda_full_evolution_v2.runtime import (
    CudaFullEvolutionSparseEventDispatchV3,
)


_STRATEGY_POLICY_IDS = (
    "sha256:" + "a" * 64,
    "sha256:" + "b" * 64,
    "sha256:" + "c" * 64,
)


def _payload(schema: str, value: object) -> CanonicalPayloadV1:
    return CanonicalPayloadV1.from_value(SchemaVersion(schema), value)


def _world(
    world_id: str = "generic-world-0001",
    *,
    seed_offset: int = 0,
    strategy_attribution: bool = False,
    hereditary_attribution: bool = False,
) -> ScienceWorldRecordV1:
    science = {
        "test_id": "GENERIC-HEREDITY",
        "test_version": "1",
        "study_id": "GENERIC-MIXED-WORLD",
        "arm_id": "mutation-on",
        "replicate": 7 + seed_offset,
    }
    world = {
        "world_profile_id": "generic-w32",
        "geometry": {"width": 32, "height": 32},
        "resident_capacity": 512,
        "overlay_profile_ids": [],
    }
    initialization = {
        "initial_population": 2,
        "exact_founder_composition": {"founder-a": 1, "founder-b": 1},
        "founder_profiles": ["founder-a", "founder-b"],
        "spatial_placement_profile": "deterministic",
        "standing_variation_profile": "paired",
    }
    seeds = {
        "ecology": 11 + seed_offset,
        "founder": 12 + seed_offset,
        "mutation": 13 + seed_offset,
    }
    semantic = {
        "world": world,
        "initialization": initialization,
        "dimension_bindings": {"condition": "mutation-on"},
        "control_bindings": [
            {
                "control_id": "mutation.step",
                "resolved_value": 2,
                "schedule": {"kind": "static"},
                "descriptor_sha256": canonical_sha256(
                    {"control_id": "mutation.step"}
                ).to_dict(),
            }
        ],
        "seed_bindings": seeds,
        "time": {
            "authority": "absolute_tick_v1",
            "start_next_tick": 0,
            "total_ticks": 1,
            "checkpoint_ticks": [1],
            "deliberate_restore_ticks": [],
        },
        "initial_state_identity": canonical_sha256(
            {"world": world, "initialization": initialization, "seed_bindings": seeds}
        ).to_dict(),
    }
    acquisition = {
        "profile_id": "SCIENCE_COMPACT",
        "event_observations": ["compact.lifecycle_events"],
        "state_observations": ["compact.state_snapshots"],
        "runtime_counters": ["compact.agent_lifetime_metrics"],
        "snapshot_ticks": [0, 1],
        "windows": [],
    }
    reducer_configuration: dict[str, object] = {
        "tracked_policy_loci": [0],
        "tracked_trait_loci": [0],
    }
    required_observations = list(COMPACT_WORLD_ATLAS_REQUIRED_CAPABILITIES)
    if strategy_attribution:
        reducer_configuration["strategy_attribution"] = {
            "schema_version": SCIENCE_STRATEGY_ATTRIBUTION_SCHEMA_V1,
            "event_kinds": list(SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1),
            "identity_mode": "declared_policy_mapping_v1",
            "lineage_mode": "parent_ancestry_v1",
            "strategy_ids": ["alpha", "beta"],
            "policy_id_to_strategy_id": {
                "sha256:" + "a" * 64: "alpha",
                "sha256:" + "b" * 64: "beta",
            },
            "denominator_fields": list(
                SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1
            ),
        }
        required_observations = sorted(
            set(required_observations)
            | set(SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1)
        )
    if hereditary_attribution:
        reducer_configuration["hereditary_attribution"] = {
            "schema_version": SCIENCE_HEREDITARY_ATTRIBUTION_SCHEMA_V1,
            "event_kinds": list(SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1),
            "identity_mode": SCIENCE_HEREDITARY_ATTRIBUTION_IDENTITY_MODE_V1,
            "tracked_identity_mode": (
                SCIENCE_HEREDITARY_ATTRIBUTION_TRACKED_IDENTITY_MODE_V1
            ),
            "lineage_mode": "parent_ancestry_v1",
            "focal_tracked_genotype_state_ids": [
                "sha256:" + "1" * 64,
                "sha256:" + "2" * 64,
            ],
            "denominator_fields": list(
                SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1
            ),
        }
        required_observations = sorted(
            set(required_observations)
            | set(SCIENCE_HEREDITARY_ATTRIBUTION_CAPABILITIES_V1)
        )
    reducers = [
        {
            "reducer_id": "science.compact.world_atlas.v1",
            "contract_version": "science_compact_world_atlas_v1",
            "required_observations": required_observations,
            "output_contract_id": "eiel_science_compact_world_atlas_result_v1",
            "configuration": reducer_configuration,
        }
    ]
    semantic_identity = canonical_sha256(
        {
            "schema_version": "eiel_science_world_record_v1",
            "science": science,
            "semantic_inputs": semantic,
        }
    )
    acquisition_identity = canonical_sha256(
        {
            "schema_version": "eiel_science_world_record_v1",
            "semantic_world_sha256": semantic_identity.to_dict(),
            "acquisition": acquisition,
            "reducers": reducers,
        }
    )
    return ScienceWorldRecordV1(
        world_id=world_id,
        science=_payload("eiel_science_ownership_v1", science),
        semantic_inputs=_payload("eiel_science_resolved_inputs_v1", semantic),
        acquisition=_payload("eiel_science_acquisition_contract_v1", acquisition),
        reducers=_payload("eiel_science_world_reducers_v1", reducers),
        semantic_world_sha256=semantic_identity,
        acquisition_sha256=acquisition_identity,
    )


def _state_id(policy: list[int], trait: list[int]) -> str:
    policy_bytes = bytes(value & 0xFF for value in policy)
    trait_bytes = b"".join(struct.pack("<h", value) for value in trait)
    return (
        "sha256:"
        + canonical_sha256(
            {"policy": policy_bytes.hex(), "trait": trait_bytes.hex()}
        ).hex
    )


def _snapshot(
    root: Path,
    *,
    world: ScienceWorldRecordV1,
    tick: int,
    agents: list[tuple[int, str]],
    catalog: dict[str, tuple[list[int], list[int]]],
    world_uid: int = 101,
    world_index: int = 0,
    agent_policy_ids: dict[int, str] | None = None,
    agent_parent_ids: dict[int, tuple[int, int]] | None = None,
) -> dict[str, object]:
    metrics = {name: 0 for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1}
    rows: list[dict[str, object]] = [
        {
            "row_kind": "world_state",
            "cell_id": world.world_id,
            "world_uid": world_uid,
            "absolute_tick": tick,
            "population_count": len(agents),
            "qualified_field_values": {
                "population_count": len(agents),
                "birth_count": max(0, len(agents) - 2),
                "death_count": 0,
                "mutation_count": max(0, len(agents) - 2),
                "resource_total": 0,
                "resource_mean": 0,
                "resource_min": 0,
                "resource_max": 0,
                "resource_nonzero_count": 0,
            },
        }
    ]
    rows.extend(
        {
            "row_kind": "active_agent_state",
            "cell_id": world.world_id,
            "world_uid": world_uid,
            "absolute_tick": tick,
            "agent_id": agent_id,
            "agent_state_id": "sha256:" + f"{agent_id + 100:064x}",
            "lineage_id": "sha256:" + f"{agent_id + 200:064x}",
            "lineage_depth": int(
                (agent_parent_ids or {}).get(agent_id, (0, 0)) != (0, 0)
            ),
            "absolute_birth_tick": (
                tick
                if (agent_parent_ids or {}).get(agent_id, (0, 0)) != (0, 0)
                else 0
            ),
            "genome_state_id": state_id,
            "lifetime_metrics": metrics,
            "policy_id": (agent_policy_ids or {}).get(agent_id),
            "parent_a_id": (agent_parent_ids or {}).get(agent_id, (0, 0))[0],
            "parent_b_id": (agent_parent_ids or {}).get(agent_id, (0, 0))[1],
        }
        for agent_id, state_id in agents
    )
    rows.extend(
        {
            "row_kind": "genome_catalog",
            "cell_id": world.world_id,
            "world_uid": world_uid,
            "absolute_tick": tick,
            "genome_state_id": state_id,
            "policy": policy,
            "trait": trait,
        }
        for state_id, (policy, trait) in catalog.items()
    )
    uncompressed = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    compressed = gzip.compress(uncompressed, mtime=0)
    root.mkdir(parents=True)
    (root / "snapshot.jsonl.gz").write_bytes(compressed)
    body = {
        "schema_version": "eiel_cuda_v2_compact_science_snapshot_artifact_v1",
        "absolute_tick": tick,
        "row_count": len(rows),
        "captured_world_count": 1,
        "captured_worlds": [
            {
                "world_index": world_index,
                "world_uid": world_uid,
                "cell_id": world.world_id,
                "science_ownership": world.science.to_value(),
                "science_world_sha256": world.semantic_world_sha256.to_dict(),
                "science_acquisition_sha256": world.acquisition_sha256.to_dict(),
                "active_agent_count": len(agents),
                "canonical_state_sha256": "0" * 64,
            }
        ],
        "metric_fields": list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1),
        "genome_catalog_included": bool(catalog),
        "uncompressed_content_sha256": hashlib.sha256(uncompressed).hexdigest(),
        "compressed_file_sha256": hashlib.sha256(compressed).hexdigest(),
        "compressed_bytes": len(compressed),
        "data_file": "snapshot.jsonl.gz",
        "native_world_control_set_sha256": "3" * 64,
        "dropped_record_count": 0,
    }
    manifest = {**body, "artifact_sha256": canonical_sha256(body).hex}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return {
        "path": str(root),
        "artifact_sha256": manifest["artifact_sha256"],
    }


def _copy_sealed_artifact(
    pointer: dict[str, object],
    destination: Path,
) -> dict[str, object]:
    shutil.copytree(Path(str(pointer["path"])), destination)
    for path in destination.iterdir():
        if path.is_file():
            path.chmod(0o444)
    return {
        "path": str(destination),
        "artifact_sha256": pointer["artifact_sha256"],
    }


def test_generic_reducer_reconstructs_birth_and_preserves_world_ownership(
    tmp_path: Path,
) -> None:
    world = _world(strategy_attribution=True)
    second_world = _world(
        "generic-world-0002", seed_offset=1, strategy_attribution=True
    )
    bundle = ScienceWorldBundleV1(
        bundle_id="GENERIC-COMPACT-BUNDLE",
        compact_source_specs=_payload(
            "eiel_science_compact_source_specs_v1", [{"source": "test"}]
        ),
        compiler_component_sha256=ContentDigest("sha256", "4" * 64),
        worlds=(world, second_world),
    )
    parent_a = ([0] * 224, [0] * 32)
    parent_b = ([10] * 224, [20] * 32)
    parent_a_id = _state_id(*parent_a)
    parent_b_id = _state_id(*parent_b)
    initial = _snapshot(
        tmp_path / "snapshot-0",
        world=world,
        tick=0,
        agents=[(1, parent_a_id), (2, parent_b_id)],
        catalog={parent_a_id: parent_a, parent_b_id: parent_b},
        agent_policy_ids={
            1: "sha256:" + "a" * 64,
            2: "sha256:" + "b" * 64,
        },
    )
    second_initial = _snapshot(
        tmp_path / "snapshot-0-second",
        world=second_world,
        tick=0,
        agents=[(1, parent_a_id), (2, parent_b_id)],
        catalog={parent_a_id: parent_a, parent_b_id: parent_b},
        world_uid=202,
        world_index=1,
        agent_policy_ids={
            1: "sha256:" + "a" * 64,
            2: "sha256:" + "b" * 64,
        },
    )

    provenance = bytearray(132)
    provenance[0] = 1
    capture = CudaFullEvolutionCompactScienceCaptureV1(
        event_agent_metrics=torch.zeros((2, 21), dtype=torch.int64),
        event_birth_provenance=torch.tensor(
            [list(provenance), list(provenance)], dtype=torch.uint8
        ),
        metric_fields=COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
        birth_provenance_encoding=COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1,
        start_next_absolute_tick=0,
        end_next_absolute_tick=1,
        world_uids=(101, 202),
        world_offsets=(0, 1, 2),
        event_content_sha256="5" * 64,
    )
    dispatch = CudaFullEvolutionSparseEventDispatchV3(
        packed_events=(
            struct.pack("<QQQQ", 3, 1, 2, 2 << 6)
            + struct.pack("<QQQQ", 3, 1, 2, 2 << 6)
        ),
        world_offsets=(0, 1, 2),
        event_counts=(1, 1),
        final_next_agent_ids=(4, 4),
        world_uids=(101, 202),
        start_next_absolute_tick=0,
        end_next_absolute_tick=1,
        tick_count=1,
        drain_elapsed_seconds=0.0,
        count_projection_elapsed_seconds=0.0,
        compaction_and_transfer_elapsed_seconds=0.0,
        content_hash_elapsed_seconds=0.0,
        content_sha256="5" * 64,
        artifact_sha256="6" * 64,
        manifest_sha256="7" * 64,
    )
    dispatch_receipt = write_compact_science_dispatch_v1(
        tmp_path / "dispatch",
        dispatch=dispatch,
        capture=capture,
        world_cells={
            world.world_id: {
                "science_logical_world_id": 101,
                "science_ownership": world.science.to_value(),
                "science_world_sha256": world.semantic_world_sha256.to_dict(),
                "science_acquisition_sha256": world.acquisition_sha256.to_dict(),
            },
            second_world.world_id: {
                "science_logical_world_id": 202,
                "science_ownership": second_world.science.to_value(),
                "science_world_sha256": second_world.semantic_world_sha256.to_dict(),
                "science_acquisition_sha256": second_world.acquisition_sha256.to_dict(),
            },
        },
        native_world_control_set_sha256="3" * 64,
    )
    child = reconstruct_compact_birth_genome_v1(
        parent_a_genome={"policy": parent_a[0], "trait": parent_a[1]},
        parent_b_genome={"policy": parent_b[0], "trait": parent_b[1]},
        birth_provenance=provenance,
        mutation_step=2,
    )["child_genome"]
    child_id = _state_id(child["policy"], child["trait"])
    terminal = _snapshot(
        tmp_path / "snapshot-1",
        world=world,
        tick=1,
        agents=[(1, parent_a_id), (2, parent_b_id), (3, child_id)],
        catalog={},
        agent_policy_ids={
            1: "sha256:" + "a" * 64,
            2: "sha256:" + "b" * 64,
            3: "sha256:" + "a" * 64,
        },
        agent_parent_ids={3: (1, 2)},
    )
    second_terminal = _snapshot(
        tmp_path / "snapshot-1-second",
        world=second_world,
        tick=1,
        agents=[(1, parent_a_id), (2, parent_b_id), (3, child_id)],
        catalog={},
        world_uid=202,
        world_index=1,
        agent_policy_ids={
            1: "sha256:" + "a" * 64,
            2: "sha256:" + "b" * 64,
            3: "sha256:" + "a" * 64,
        },
        agent_parent_ids={3: (1, 2)},
    )
    dispatch_pointer = {
        "path": dispatch_receipt["path"],
        "artifact_sha256": dispatch_receipt["artifact_sha256"],
    }
    body = {
        "schema_version": "eiel_cuda_v2_compact_science_capture_index_v1",
        "observation_abi": "eiel_cuda_v2_compact_science_capture_v1",
        "batch_id": "generic-b1",
        "ordered_cell_ids": [world.world_id, second_world.world_id],
        "native_world_control_set_sha256": "3" * 64,
        "terminal_next_absolute_tick": 1,
        "dispatch_artifacts": [dispatch_pointer],
        "snapshot_artifacts": [
            initial,
            second_initial,
            terminal,
            second_terminal,
        ],
        "checkpoint_frontier": None,
        "dropped_record_count": 0,
    }
    index = {**body, "index_sha256": canonical_sha256(body).hex}
    result = reduce_compact_science_world_atlas_v1(index, bundle)
    row = result["world_results"][0]
    assert row["resident_capacity"] == 512
    assert row["birth_count"] == row["reconstructed_birth_count"] == 1
    assert row["mutation_origin_count"] == 1
    assert row["mutation_origin_counts_by_linked_locus"][0] == 1
    assert row["maximum_lineage_depth"] == 1
    assert (
        row["milestones"][-1]["genome_reconstruction"]["state_identity_check_count"]
        == 3
    )
    assert result["physical_batch_count"] == 1
    assert result["physical_composition_used"] is False
    assert len(result["world_results"]) == 2
    assert result["world_results"][1]["resident_capacity"] == 512
    assert result["world_results"][1]["birth_count"] == 1

    parallel = reduce_compact_science_world_atlas_v1(
        index,
        bundle,
        max_workers=2,
    )
    assert parallel == result

    incremental = CompactWorldAtlasIncrementalReducerV1(bundle)
    assert incremental.consume_snapshot(initial) is True
    assert incremental.consume_snapshot(second_initial) is True
    assert incremental.consume_dispatch(dispatch_pointer) is True
    assert incremental.next_absolute_tick == 1
    assert incremental.consumed_dispatch_count == 1
    assert incremental.finalize(index) == result

    parallel_incremental = CompactWorldAtlasIncrementalReducerV1(
        bundle,
        max_workers=2,
    )
    assert parallel_incremental.consume_snapshot(initial) is True
    assert parallel_incremental.consume_snapshot(second_initial) is True
    assert parallel_incremental.consume_dispatch(dispatch_pointer) is True
    assert parallel_incremental.finalize(index) == result

    native_path, native_manifest = build_native_compact_reducer_v1(
        tmp_path / "native-reducer"
    )
    native_artifact = load_native_compact_reducer_v1(
        native_path,
        native_manifest,
        require_production_qualified=False,
    )
    with pytest.raises(
        ValueError,
        match="native async reducer preflight failed: .*not production-routing",
    ):
        CompactWorldAtlasAsyncSidecarV1(
            tmp_path / "native-sidecar-preflight",
            bundle,
            native_artifact_path=native_path,
            native_manifest_path=native_manifest,
            native_manifest_sha256=hashlib.sha256(
                native_manifest.read_bytes()
            ).hexdigest(),
        )
    assert (
        reduce_compact_science_world_atlas_v1(
            index,
            bundle,
            max_workers=2,
            native_artifact=native_artifact,
        )
        == result
    )
    native_incremental = CompactWorldAtlasIncrementalReducerV1(
        bundle,
        max_workers=2,
        native_artifact=native_artifact,
    )
    assert native_incremental.execution_route == "native_cpp_v1"
    assert native_incremental.consume_snapshot(initial) is True
    assert native_incremental.consume_snapshot(second_initial) is True
    assert native_incremental.consume_dispatch(dispatch_pointer) is True
    assert native_incremental.consume_snapshot(terminal) is True
    assert native_incremental.consume_snapshot(second_terminal) is True
    assert native_incremental.finalize(index) == result
    assert native_incremental.strategy_attribution_result == (
        analyze_compact_science_strategy_attribution_streaming_v1(index, bundle)
    )

    science_root = tmp_path / "async-science-compact"
    async_initial = _copy_sealed_artifact(
        initial,
        science_root / "snapshots" / "tick-000000000000-a",
    )
    async_second_initial = _copy_sealed_artifact(
        second_initial,
        science_root / "snapshots" / "tick-000000000000-b",
    )
    async_terminal = _copy_sealed_artifact(
        terminal,
        science_root / "snapshots" / "tick-000000000001-a",
    )
    async_second_terminal = _copy_sealed_artifact(
        second_terminal,
        science_root / "snapshots" / "tick-000000000001-b",
    )
    async_dispatch = _copy_sealed_artifact(
        dispatch_pointer,
        science_root / "dispatches" / "ticks-000000000000-000000000001",
    )
    async_body = {
        **body,
        "dispatch_artifacts": [async_dispatch],
        "snapshot_artifacts": [
            async_initial,
            async_second_initial,
            async_terminal,
            async_second_terminal,
        ],
    }
    async_index = {
        **async_body,
        "index_sha256": canonical_sha256(async_body).hex,
    }
    async_expected = reduce_compact_science_world_atlas_v1(
        async_index,
        bundle,
    )
    terminal_index_path = science_root / "capture-index-terminal.json"
    async_dispatch_manifest = Path(str(async_dispatch["path"])) / "manifest.json"
    async_dispatch_manifest.chmod(0o644)
    with CompactWorldAtlasAsyncSidecarV1(
        science_root,
        bundle,
        max_workers=2,
        poll_interval_seconds=0.01,
    ) as sidecar:
        assert sidecar.pid is not None
        assert sidecar.is_alive
        assert not terminal_index_path.exists()
        terminal_index_path.write_text(json.dumps(async_index), encoding="utf-8")
        terminal_index_path.chmod(0o444)
        with pytest.raises(
            ValueError,
            match="async compact reducer result timed out",
        ):
            sidecar.result(timeout_seconds=0.1)
        assert sidecar.is_alive
        async_dispatch_manifest.chmod(0o444)
        assert sidecar.result(timeout_seconds=15) == async_expected
        progress = sidecar.progress_snapshot()
        assert progress["phase"] == "COMPLETED"
        assert progress["campaign_caught_up_fraction"] == 1.0
        assert progress["producer_reducer_frontier_lag_ticks"] == 0
        assert progress["sidecar_process"]["alive"] is False
        assert progress["process_cleanup"]["clean"] is True

    with CompactWorldAtlasAsyncSidecarV1(
        science_root,
        bundle,
        max_workers=2,
        poll_interval_seconds=0.01,
        native_artifact_path=native_path,
        native_manifest_path=native_manifest,
        native_manifest_sha256=hashlib.sha256(
            native_manifest.read_bytes()
        ).hexdigest(),
        native_artifact_use_class=(
            NATIVE_REDUCER_ARTIFACT_USE_ENGINEERING_REAL_USE_V1
        ),
    ) as native_sidecar:
        native_bundle = native_sidecar.result_bundle(timeout_seconds=15)
        assert native_bundle["atlas_result"] == async_expected
        assert native_bundle["strategy_attribution_result"] == (
            analyze_compact_science_strategy_attribution_streaming_v1(
                async_index, bundle
            )
        )
        assert native_bundle["second_python_attribution_pass"] is False
        assert native_bundle["native_artifact_use_class"] == (
            NATIVE_REDUCER_ARTIFACT_USE_ENGINEERING_REAL_USE_V1
        )
        assert native_bundle["operational_metrics"]["peak_rss_kib"] > 0


@dataclass(slots=True)
class _GeneratedAgent:
    genome: bytes
    policy_id: str
    parent_a_id: int = 0
    parent_b_id: int = 0


@dataclass(slots=True)
class _GeneratedWorldState:
    world: ScienceWorldRecordV1
    world_uid: int
    active: dict[int, _GeneratedAgent]
    next_agent_id: int
    birth_count: int = 0
    death_count: int = 0
    mutation_count: int = 0


def _randomized_world(
    world_id: str,
    *,
    world_index: int,
    total_ticks: int,
    snapshot_ticks: list[int],
) -> ScienceWorldRecordV1:
    science = {
        "test_id": "NATIVE-REDUCER-DIFFERENTIAL",
        "test_version": "1",
        "study_id": "NATIVE-REDUCER-RANDOMIZED",
        "arm_id": "engineering",
        "replicate": world_index,
    }
    world = {
        "world_profile_id": "randomized-w16",
        "geometry": {"width": 16, "height": 16},
        "resident_capacity": 16,
        "overlay_profile_ids": [],
    }
    initialization = {
        "initial_population": 6,
        "exact_founder_composition": ["alpha", "beta", "gamma"],
        "founder_profiles": ["alpha", "beta", "gamma"],
        "spatial_placement_profile": "deterministic",
        "standing_variation_profile": "randomized-differential",
    }
    seeds = {
        "ecology": 100 + world_index,
        "founder": 200 + world_index,
        "mutation": 300 + world_index,
    }
    mutation_schedule = {
        "kind": "piecewise_constant_absolute_ticks",
        "segments": [
            {"start_absolute_tick": 0, "value": 1},
            {"start_absolute_tick": 64, "value": 3},
            {"start_absolute_tick": 128, "value": 127},
        ],
    }
    semantic = {
        "world": world,
        "initialization": initialization,
        "dimension_bindings": {
            "pressure_regime": "off",
            "randomized_panel": world_index % 2,
        },
        "control_bindings": [
            {
                "control_id": "mutation.step",
                "resolved_value": 1,
                "schedule": mutation_schedule,
                "descriptor_sha256": canonical_sha256(
                    {"control_id": "mutation.step", "schedule": mutation_schedule}
                ).to_dict(),
            }
        ],
        "seed_bindings": seeds,
        "time": {
            "authority": "absolute_tick_v1",
            "start_next_tick": 0,
            "total_ticks": total_ticks,
            "checkpoint_ticks": [total_ticks],
            "deliberate_restore_ticks": [],
        },
        "initial_state_identity": canonical_sha256(
            {"world": world, "initialization": initialization, "seeds": seeds}
        ).to_dict(),
    }
    acquisition = {
        "profile_id": "SCIENCE_COMPACT",
        "event_observations": ["compact.lifecycle_events"],
        "state_observations": ["compact.state_snapshots"],
        "runtime_counters": ["compact.agent_lifetime_metrics"],
        "snapshot_ticks": snapshot_ticks,
        "windows": [],
    }
    strategy = {
        "schema_version": SCIENCE_STRATEGY_ATTRIBUTION_SCHEMA_V1,
        "event_kinds": list(SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1),
        "identity_mode": "declared_policy_mapping_v1",
        "lineage_mode": "parent_ancestry_v1",
        "strategy_ids": ["alpha", "beta", "gamma"],
        "policy_id_to_strategy_id": dict(
            zip(_STRATEGY_POLICY_IDS, ("alpha", "beta", "gamma"), strict=True)
        ),
        "denominator_fields": list(
            SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1
        ),
    }
    reducers = [
        {
            "reducer_id": "science.compact.world_atlas.v1",
            "contract_version": "science_compact_world_atlas_v1",
            "required_observations": sorted(
                set(COMPACT_WORLD_ATLAS_REQUIRED_CAPABILITIES)
                | set(SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1)
            ),
            "output_contract_id": "eiel_science_compact_world_atlas_result_v1",
            "configuration": {
                "tracked_policy_loci": [0, 31, 111],
                "tracked_trait_loci": [0, 7, 15],
                "strategy_attribution": strategy,
            },
        }
    ]
    semantic_identity = canonical_sha256(
        {
            "schema_version": "eiel_science_world_record_v1",
            "science": science,
            "semantic_inputs": semantic,
        }
    )
    acquisition_identity = canonical_sha256(
        {
            "schema_version": "eiel_science_world_record_v1",
            "semantic_world_sha256": semantic_identity.to_dict(),
            "acquisition": acquisition,
            "reducers": reducers,
        }
    )
    return ScienceWorldRecordV1(
        world_id=world_id,
        science=_payload("eiel_science_ownership_v1", science),
        semantic_inputs=_payload("eiel_science_resolved_inputs_v1", semantic),
        acquisition=_payload("eiel_science_acquisition_contract_v1", acquisition),
        reducers=_payload("eiel_science_world_reducers_v1", reducers),
        semantic_world_sha256=semantic_identity,
        acquisition_sha256=acquisition_identity,
    )


def _random_metric_row(world_uid: int, agent_id: int, tick: int) -> list[int]:
    base = (world_uid * 97 + agent_id * 193 + tick * 389) % 1_000_000
    values = [base + field * 17 for field in range(21)]
    values[17] = -(base * 31 + 17)
    return values


def _generated_genome_value(encoded: bytes) -> dict[str, list[int]]:
    unpacked = struct.unpack("<224b32h", encoded)
    return {
        "policy": list(unpacked[:224]),
        "trait": list(unpacked[224:]),
    }


def _pack_random_provenance(
    rng: random.Random,
    *,
    dispatch_index: int,
    birth_index: int,
) -> tuple[bytes, bytes]:
    packed = bytearray(68)
    for linked_locus in range(128):
        allele_zero = rng.randrange(3)
        allele_one = rng.randrange(3)
        if birth_index == 0 and linked_locus == dispatch_index:
            allele_zero, allele_one = 1, 2
        nibble = allele_zero | (allele_one << 2)
        packed[linked_locus // 2] |= nibble << (4 * (linked_locus % 2))
    recombination = (0, 0xFFFFFFFF, rng.getrandbits(32))[dispatch_index % 3]
    packed[64:68] = recombination.to_bytes(4, "little")
    expanded = bytes(
        (packed[linked_locus // 2] >> (4 * (linked_locus % 2))) & 0x0F
        for linked_locus in range(128)
    ) + packed[64:68]
    return bytes(packed), expanded


def _mutation_step_for_tick(tick: int) -> int:
    if tick >= 128:
        return 127
    if tick >= 64:
        return 3
    return 1


def _write_randomized_snapshot(
    root: Path,
    *,
    states: list[_GeneratedWorldState],
    tick: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    captured_worlds: list[dict[str, object]] = []
    for world_index, state in enumerate(states):
        rows.append(
            {
                "row_kind": "world_state",
                "cell_id": state.world.world_id,
                "world_uid": state.world_uid,
                "absolute_tick": tick,
                "qualified_field_values": {
                    "population_count": len(state.active),
                    "birth_count": state.birth_count,
                    "death_count": state.death_count,
                    "mutation_count": state.mutation_count,
                    "resource_total": tick + world_index,
                    "resource_mean": tick / 2 + world_index,
                    "resource_min": world_index,
                    "resource_max": tick + world_index,
                    "resource_nonzero_count": int(tick > 0),
                },
            }
        )
        for agent_id, agent in sorted(state.active.items()):
            metrics = dict(
                zip(
                    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
                    _random_metric_row(state.world_uid, agent_id, tick),
                    strict=True,
                )
            )
            rows.append(
                {
                    "row_kind": "active_agent_state",
                    "cell_id": state.world.world_id,
                    "world_uid": state.world_uid,
                    "absolute_tick": tick,
                    "agent_id": agent_id,
                    "genome_state_id": _state_id(
                        list(struct.unpack("<224b32h", agent.genome)[:224]),
                        list(struct.unpack("<224b32h", agent.genome)[224:]),
                    ),
                    "lifetime_metrics": metrics,
                    "policy_id": agent.policy_id,
                    "parent_a_id": agent.parent_a_id,
                    "parent_b_id": agent.parent_b_id,
                    "energy": (agent_id * 3 + tick) % 101,
                    "health": (agent_id * 5 + tick) % 103,
                    "injury": (agent_id * 7 + tick) % 107,
                }
            )
        if tick == 0:
            catalog: dict[str, bytes] = {}
            for agent in state.active.values():
                unpacked = struct.unpack("<224b32h", agent.genome)
                state_id = _state_id(
                    list(unpacked[:224]), list(unpacked[224:])
                )
                catalog[state_id] = agent.genome
            for state_id, genome in sorted(catalog.items()):
                unpacked = struct.unpack("<224b32h", genome)
                rows.append(
                    {
                        "row_kind": "genome_catalog",
                        "cell_id": state.world.world_id,
                        "world_uid": state.world_uid,
                        "absolute_tick": tick,
                        "genome_state_id": state_id,
                        "policy": list(unpacked[:224]),
                        "trait": list(unpacked[224:]),
                    }
                )
        captured_worlds.append(
            {
                "world_index": world_index,
                "world_uid": state.world_uid,
                "cell_id": state.world.world_id,
                "science_ownership": state.world.science.to_value(),
                "science_world_sha256": state.world.semantic_world_sha256.to_dict(),
                "science_acquisition_sha256": (
                    state.world.acquisition_sha256.to_dict()
                ),
                "active_agent_count": len(state.active),
                "canonical_state_sha256": hashlib.sha256(
                    f"{state.world.world_id}:{tick}".encode()
                ).hexdigest(),
            }
        )
    uncompressed = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    compressed = gzip.compress(uncompressed, mtime=0)
    root.mkdir(parents=True)
    (root / "snapshot.jsonl.gz").write_bytes(compressed)
    body = {
        "schema_version": "eiel_cuda_v2_compact_science_snapshot_artifact_v1",
        "absolute_tick": tick,
        "row_count": len(rows),
        "captured_world_count": len(captured_worlds),
        "captured_worlds": captured_worlds,
        "metric_fields": list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1),
        "genome_catalog_included": tick == 0,
        "uncompressed_content_sha256": hashlib.sha256(uncompressed).hexdigest(),
        "compressed_file_sha256": hashlib.sha256(compressed).hexdigest(),
        "compressed_bytes": len(compressed),
        "data_file": "snapshot.jsonl.gz",
        "native_world_control_set_sha256": "3" * 64,
        "dropped_record_count": 0,
    }
    manifest = {**body, "artifact_sha256": canonical_sha256(body).hex}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {"path": str(root), "artifact_sha256": manifest["artifact_sha256"]}


def _write_randomized_dispatch(
    root: Path,
    *,
    states: list[_GeneratedWorldState],
    start: int,
    end: int,
    dispatch_index: int,
    rng: random.Random,
) -> dict[str, object]:
    event_rows: list[bytes] = []
    metric_rows: list[list[int]] = []
    provenance_rows: list[bytes] = []
    world_offsets = [0]
    event_counts: list[int] = []
    for state in states:
        world_event_count = 0
        birth_index = 0
        offsets = sorted(rng.randrange(end - start) for _ in range(28))
        for event_index, offset in enumerate(offsets):
            absolute_tick = start + offset
            should_birth = (
                len(state.active) < 16
                and (len(state.active) <= 2 or event_index % 3 != 1)
            )
            if should_birth:
                parent_a_id = rng.choice(tuple(state.active))
                parent_b_id = rng.choice(tuple(state.active))
                parent_a = state.active[parent_a_id]
                parent_b = state.active[parent_b_id]
                _, expanded = _pack_random_provenance(
                    rng,
                    dispatch_index=dispatch_index,
                    birth_index=birth_index,
                )
                oracle = reconstruct_compact_birth_genome_v1(
                    parent_a_genome=_generated_genome_value(parent_a.genome),
                    parent_b_genome=_generated_genome_value(parent_b.genome),
                    birth_provenance=expanded,
                    mutation_step=_mutation_step_for_tick(absolute_tick),
                )
                agent_id = state.next_agent_id
                state.next_agent_id += 1
                state.active[agent_id] = _GeneratedAgent(
                    genome=struct.pack(
                        "<224b32h",
                        *oracle["child_genome"]["policy"],
                        *oracle["child_genome"]["trait"],
                    ),
                    policy_id=parent_a.policy_id,
                    parent_a_id=parent_a_id,
                    parent_b_id=parent_b_id,
                )
                state.birth_count += 1
                state.mutation_count += int(oracle["mutation_origin_count"])
                event_rows.append(
                    struct.pack(
                        "<QQQQ",
                        agent_id,
                        parent_a_id,
                        parent_b_id,
                        (2 << 6) | offset,
                    )
                )
                metric_rows.append([0] * 21)
                provenance_rows.append(expanded)
                birth_index += 1
            else:
                agent_id = rng.choice(tuple(state.active))
                event_rows.append(
                    struct.pack("<QQQQ", agent_id, 0, 0, (1 << 6) | offset)
                )
                metric_rows.append(
                    _random_metric_row(state.world_uid, agent_id, absolute_tick)
                )
                provenance_rows.append(bytes(132))
                del state.active[agent_id]
                state.death_count += 1
            world_event_count += 1
        event_counts.append(world_event_count)
        world_offsets.append(len(event_rows))

    packed_events = b"".join(event_rows)
    event_digest = hashlib.sha256(packed_events).hexdigest()
    capture = CudaFullEvolutionCompactScienceCaptureV1(
        event_agent_metrics=torch.tensor(metric_rows, dtype=torch.int64),
        event_birth_provenance=torch.tensor(provenance_rows, dtype=torch.uint8),
        metric_fields=COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
        birth_provenance_encoding=COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1,
        start_next_absolute_tick=start,
        end_next_absolute_tick=end,
        world_uids=tuple(state.world_uid for state in states),
        world_offsets=tuple(world_offsets),
        event_content_sha256=event_digest,
    )
    dispatch = CudaFullEvolutionSparseEventDispatchV3(
        packed_events=packed_events,
        world_offsets=tuple(world_offsets),
        event_counts=tuple(event_counts),
        final_next_agent_ids=tuple(state.next_agent_id for state in states),
        world_uids=tuple(state.world_uid for state in states),
        start_next_absolute_tick=start,
        end_next_absolute_tick=end,
        tick_count=end - start,
        drain_elapsed_seconds=0.0,
        count_projection_elapsed_seconds=0.0,
        compaction_and_transfer_elapsed_seconds=0.0,
        content_hash_elapsed_seconds=0.0,
        content_sha256=event_digest,
        artifact_sha256=hashlib.sha256((event_digest + "artifact").encode()).hexdigest(),
        manifest_sha256=hashlib.sha256((event_digest + "manifest").encode()).hexdigest(),
    )
    receipt = write_compact_science_dispatch_v1(
        root,
        dispatch=dispatch,
        capture=capture,
        world_cells={
            state.world.world_id: {
                "science_logical_world_id": state.world_uid,
                "science_ownership": state.world.science.to_value(),
                "science_world_sha256": state.world.semantic_world_sha256.to_dict(),
                "science_acquisition_sha256": (
                    state.world.acquisition_sha256.to_dict()
                ),
            }
            for state in states
        },
        native_world_control_set_sha256="3" * 64,
    )
    return {"path": receipt["path"], "artifact_sha256": receipt["artifact_sha256"]}


def _randomized_capture(
    root: Path,
    *,
    seed: int,
) -> tuple[ScienceWorldBundleV1, dict[str, object]]:
    rng = random.Random(seed)
    total_ticks = 192
    snapshot_ticks = [0, 64, 128, total_ticks]
    worlds = tuple(
        _randomized_world(
            f"randomized-world-{index:04d}",
            world_index=index,
            total_ticks=total_ticks,
            snapshot_ticks=snapshot_ticks,
        )
        for index in range(4)
    )
    bundle = ScienceWorldBundleV1(
        bundle_id=f"RANDOMIZED-NATIVE-REDUCER-{seed}",
        compact_source_specs=_payload(
            "eiel_science_compact_source_specs_v1", [{"seed": seed}]
        ),
        compiler_component_sha256=ContentDigest("sha256", "4" * 64),
        worlds=worlds,
    )
    states: list[_GeneratedWorldState] = []
    for world_index, world in enumerate(worlds):
        active: dict[int, _GeneratedAgent] = {}
        for agent_id in range(1, 7):
            policy = [rng.randint(-127, 127) for _ in range(224)]
            trait = [rng.randint(-32767, 32767) for _ in range(32)]
            if world_index == 0 and agent_id == 1:
                policy[:4] = [-127, -126, 126, 127]
                trait[:4] = [-32767, -32766, 32766, 32767]
            active[agent_id] = _GeneratedAgent(
                genome=struct.pack("<224b32h", *policy, *trait),
                policy_id=_STRATEGY_POLICY_IDS[(agent_id - 1) % 3],
            )
        states.append(
            _GeneratedWorldState(
                world=world,
                world_uid=10_000 + world_index,
                active=active,
                next_agent_id=7,
            )
        )
    snapshots = [
        _write_randomized_snapshot(root / "snapshot-000", states=states, tick=0)
    ]
    dispatches: list[dict[str, object]] = []
    for dispatch_index, start in enumerate((0, 64, 128)):
        end = start + 64
        dispatches.append(
            _write_randomized_dispatch(
                root / f"dispatch-{dispatch_index:03d}",
                states=states,
                start=start,
                end=end,
                dispatch_index=dispatch_index,
                rng=rng,
            )
        )
        snapshots.append(
            _write_randomized_snapshot(
                root / f"snapshot-{end:03d}", states=states, tick=end
            )
        )
    body = {
        "schema_version": "eiel_cuda_v2_compact_science_capture_index_v1",
        "observation_abi": "eiel_cuda_v2_compact_science_capture_v1",
        "batch_id": f"randomized-{seed}",
        "ordered_cell_ids": [world.world_id for world in worlds],
        "native_world_control_set_sha256": "3" * 64,
        "terminal_next_absolute_tick": total_ticks,
        "dispatch_artifacts": dispatches,
        "snapshot_artifacts": snapshots,
        "checkpoint_frontier": None,
        "dropped_record_count": 0,
    }
    return bundle, {**body, "index_sha256": canonical_sha256(body).hex}


@pytest.fixture(scope="module")
def randomized_native_artifact(
    tmp_path_factory: pytest.TempPathFactory,
):
    artifact, manifest = build_native_compact_reducer_v1(
        tmp_path_factory.mktemp("randomized-native-reducer")
    )
    return load_native_compact_reducer_v1(
        artifact, manifest, require_production_qualified=False
    )


@pytest.mark.parametrize(
    "seed",
    [0xE1E1, 0xA7A5, 0x5EED, 0xC0DE, 0xB17B, 0xD15C, 0x5A17, 0xF00D],
)
def test_randomized_native_atlas_and_strategy_are_exact_across_threads_and_restart(
    tmp_path: Path,
    randomized_native_artifact,
    seed: int,
) -> None:
    bundle, index = _randomized_capture(tmp_path / f"capture-{seed}", seed=seed)
    python_atlas = reduce_compact_science_world_atlas_v1(index, bundle)
    python_strategy = analyze_compact_science_strategy_attribution_streaming_v1(
        index, bundle
    )

    for worker_threads in (1, 2, 4, 8):
        reducer = CompactWorldAtlasIncrementalReducerV1(
            bundle,
            max_workers=worker_threads,
            native_artifact=randomized_native_artifact,
        )
        assert reducer.finalize(index) == python_atlas
        assert reducer.strategy_attribution_result == python_strategy

    interrupted = CompactWorldAtlasIncrementalReducerV1(
        bundle,
        max_workers=4,
        native_artifact=randomized_native_artifact,
    )
    assert interrupted.consume_snapshot(index["snapshot_artifacts"][0]) is True
    assert interrupted.consume_dispatch(index["dispatch_artifacts"][0]) is True
    assert interrupted.consume_snapshot(index["snapshot_artifacts"][1]) is True
    interrupted.close()

    restarted = CompactWorldAtlasIncrementalReducerV1(
        bundle,
        max_workers=4,
        native_artifact=randomized_native_artifact,
    )
    assert restarted.finalize(index) == python_atlas
    assert restarted.strategy_attribution_result == python_strategy


def _async_capture_target_plan(
    source_index: dict[str, object],
    root: Path,
) -> tuple[
    list[tuple[dict[str, object], Path, dict[str, object]]],
    list[tuple[dict[str, object], Path, dict[str, object]]],
    dict[str, object],
]:
    dispatch_plan = []
    for pointer_value in source_index["dispatch_artifacts"]:
        pointer = dict(pointer_value)
        manifest = json.loads(
            (Path(str(pointer["path"])) / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        destination = root / "dispatches" / (
            f"ticks-{manifest['start_next_absolute_tick']:012d}-"
            f"{manifest['end_next_absolute_tick']:012d}"
        )
        target = {
            "path": str(destination),
            "artifact_sha256": pointer["artifact_sha256"],
        }
        dispatch_plan.append((pointer, destination, target))
    snapshot_plan = []
    for index, pointer_value in enumerate(source_index["snapshot_artifacts"]):
        pointer = dict(pointer_value)
        manifest = json.loads(
            (Path(str(pointer["path"])) / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        destination = root / "snapshots" / (
            f"tick-{manifest['absolute_tick']:012d}-{index:03d}"
        )
        target = {
            "path": str(destination),
            "artifact_sha256": pointer["artifact_sha256"],
        }
        snapshot_plan.append((pointer, destination, target))
    body = {
        key: value
        for key, value in source_index.items()
        if key not in {"index_sha256", "path"}
    }
    body["dispatch_artifacts"] = [row[2] for row in dispatch_plan]
    body["snapshot_artifacts"] = [row[2] for row in snapshot_plan]
    terminal_index = {**body, "index_sha256": canonical_sha256(body).hex}
    return dispatch_plan, snapshot_plan, terminal_index


def _publish_async_plan_rows(
    rows: list[tuple[dict[str, object], Path, dict[str, object]]],
    indices: tuple[int, ...],
) -> None:
    for index in indices:
        source, destination, _ = rows[index]
        _copy_sealed_artifact(source, destination)


def _write_sealed_terminal_index(root: Path, index: dict[str, object]) -> Path:
    path = root / "capture-index-terminal.json"
    path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    path.chmod(0o444)
    return path


def _wait_for_sidecar_progress(
    sidecar: CompactWorldAtlasAsyncSidecarV1,
    predicate,
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        progress = sidecar.progress_snapshot()
        if predicate(progress):
            return progress
        if not sidecar.is_alive:
            break
        time.sleep(0.01)
    raise AssertionError(f"sidecar did not reach expected progress: {progress}")


def _native_engineering_sidecar_kwargs(artifact) -> dict[str, object]:
    return {
        "max_workers": 4,
        "poll_interval_seconds": 0.01,
        "native_artifact_path": artifact.artifact_path,
        "native_manifest_path": artifact.manifest_path,
        "native_manifest_sha256": artifact.manifest_sha256,
        "native_artifact_use_class": (
            NATIVE_REDUCER_ARTIFACT_USE_ENGINEERING_REAL_USE_V1
        ),
    }


def test_async_snapshot_discovery_excludes_native_staging_roots(
    tmp_path: Path,
) -> None:
    snapshots_root = tmp_path / "science-compact" / "snapshots"
    canonical = snapshots_root / "tick-000000007500"
    native_stage = (
        snapshots_root
        / ".tick-000000007500.011cf3c4a7e7068505c2.native-snapshot-v1.stage"
    )
    unrelated = snapshots_root / "tick-not-a-native-root"
    suffixed_canonical = snapshots_root / "tick-000000007500.extra"
    for root in (canonical, native_stage, unrelated, suffixed_canonical):
        root.mkdir(parents=True)
        (root / "manifest.json").write_text("{}", encoding="ascii")

    assert _canonical_snapshot_manifest_paths_v1(snapshots_root) == (
        canonical / "manifest.json",
    )


def test_async_native_restart_replays_sealed_prefix_and_cleans_processes(
    tmp_path: Path,
    randomized_native_artifact,
) -> None:
    bundle, source_index = _randomized_capture(
        tmp_path / "source-capture", seed=0x51DEC4
    )
    root = tmp_path / "async-capture"
    dispatch_plan, snapshot_plan, terminal_index = _async_capture_target_plan(
        source_index, root
    )
    _publish_async_plan_rows(snapshot_plan, (0, 1))
    _publish_async_plan_rows(dispatch_plan, (0,))

    first = CompactWorldAtlasAsyncSidecarV1(
        root,
        bundle,
        **_native_engineering_sidecar_kwargs(randomized_native_artifact),
    )
    first_progress = _wait_for_sidecar_progress(
        first,
        lambda value: value["reducer_frontier_tick"] >= 64,
    )
    assert first_progress["producer_reducer_frontier_lag_ticks"] == 0
    first.cancel()
    cancelled = first.progress_snapshot()
    assert cancelled["phase"] == "CANCELLED"
    assert cancelled["process_cleanup"]["clean"] is True
    assert cancelled["process_cleanup"]["orphaned_worker_count"] == 0

    _publish_async_plan_rows(snapshot_plan, (2, 3))
    _publish_async_plan_rows(dispatch_plan, (1, 2))
    _write_sealed_terminal_index(root, terminal_index)
    python_atlas = reduce_compact_science_world_atlas_v1(terminal_index, bundle)
    python_strategy = analyze_compact_science_strategy_attribution_streaming_v1(
        terminal_index, bundle
    )
    with CompactWorldAtlasAsyncSidecarV1(
        root,
        bundle,
        **_native_engineering_sidecar_kwargs(randomized_native_artifact),
    ) as restarted:
        execution = restarted.result_bundle(timeout_seconds=15)
        assert execution["atlas_result"] == python_atlas
        assert execution["strategy_attribution_result"] == python_strategy
        assert execution["second_python_attribution_pass"] is False
        metrics = execution["operational_metrics"]
        assert metrics["phase"] == "COMPLETED"
        assert metrics["campaign_caught_up_fraction"] == 1.0
        assert metrics["producer_reducer_frontier_lag_ticks"] == 0
        assert metrics["terminal_backlog_ticks"] in {0, 192}
        assert metrics["terminal_reconciliation_seconds"] >= 0
        assert metrics["process_cleanup"]["clean"] is True
        assert metrics["process_cleanup"]["orphaned_worker_count"] == 0


def test_async_native_malformed_input_fails_only_the_consumer(
    tmp_path: Path,
    randomized_native_artifact,
) -> None:
    bundle, source_index = _randomized_capture(
        tmp_path / "source-capture", seed=0xBAD1
    )
    root = tmp_path / "malformed-capture"
    dispatch_plan, snapshot_plan, _ = _async_capture_target_plan(source_index, root)
    _publish_async_plan_rows(snapshot_plan, (0,))
    _publish_async_plan_rows(dispatch_plan, (0,))
    malformed_root = dispatch_plan[0][1]
    manifest = json.loads(
        (malformed_root / "manifest.json").read_text(encoding="utf-8")
    )
    event_path = malformed_root / str(manifest["event_file"])
    event_path.chmod(0o644)
    corrupted = bytearray(event_path.read_bytes())
    corrupted[len(corrupted) // 2] ^= 1
    event_path.write_bytes(corrupted)
    event_path.chmod(0o444)
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(str(snapshot_plan[0][0]["path"]))).iterdir()
        if path.is_file()
    }

    with CompactWorldAtlasAsyncSidecarV1(
        root,
        bundle,
        **_native_engineering_sidecar_kwargs(randomized_native_artifact),
    ) as sidecar:
        with pytest.raises(
            ValueError,
            match="async compact reducer failed.*SHA-256 differs",
        ):
            sidecar.result_bundle(timeout_seconds=10)
        progress = sidecar.progress_snapshot()
        assert progress["phase"] == "FAILED"
        assert progress["sidecar_process"]["alive"] is False
        assert progress["process_cleanup"]["clean"] is True
        assert progress["process_cleanup"]["orphaned_worker_count"] == 0

    assert source_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(str(snapshot_plan[0][0]["path"]))).iterdir()
        if path.is_file()
    }


def test_async_native_lag_timeout_and_cancellation_are_bounded(
    tmp_path: Path,
    randomized_native_artifact,
) -> None:
    bundle, _ = _randomized_capture(tmp_path / "source-capture", seed=0x1A6)
    sidecar = CompactWorldAtlasAsyncSidecarV1(
        tmp_path / "empty-capture",
        bundle,
        **_native_engineering_sidecar_kwargs(randomized_native_artifact),
    )
    with pytest.raises(ValueError, match="result timed out"):
        sidecar.result_bundle(timeout_seconds=0.05)
    cancel_started = time.monotonic()
    sidecar.cancel()
    assert time.monotonic() - cancel_started < 4.0
    progress = sidecar.progress_snapshot()
    assert progress["sidecar_process"]["alive"] is False
    assert progress["process_cleanup"]["clean"] is True
    assert progress["process_cleanup"]["orphaned_worker_count"] == 0


def test_full_hereditary_reducer_uses_exact_contract_and_native_stream(
    tmp_path: Path,
    randomized_native_artifact,
) -> None:
    world = _world(hereditary_attribution=True)
    bundle = ScienceWorldBundleV1(
        bundle_id="GENERIC-HEREDITARY-BUNDLE",
        compact_source_specs=_payload(
            "eiel_science_compact_source_specs_v1", [{"source": "test"}]
        ),
        compiler_component_sha256=ContentDigest("sha256", "8" * 64),
        worlds=(world,),
    )
    parent_a = ([0] * 224, [0] * 32)
    parent_b = ([10] * 224, [20] * 32)
    parent_a_id = _state_id(*parent_a)
    parent_b_id = _state_id(*parent_b)
    initial = _snapshot(
        tmp_path / "hereditary-snapshot-0",
        world=world,
        tick=0,
        agents=[(1, parent_a_id), (2, parent_b_id)],
        catalog={parent_a_id: parent_a, parent_b_id: parent_b},
    )
    provenance = bytearray(132)
    provenance[0] = 1
    child = reconstruct_compact_birth_genome_v1(
        parent_a_genome={"policy": parent_a[0], "trait": parent_a[1]},
        parent_b_genome={"policy": parent_b[0], "trait": parent_b[1]},
        birth_provenance=provenance,
        mutation_step=2,
    )["child_genome"]
    child_id = _state_id(child["policy"], child["trait"])
    terminal = _snapshot(
        tmp_path / "hereditary-snapshot-1",
        world=world,
        tick=1,
        agents=[(1, parent_a_id), (2, parent_b_id), (3, child_id)],
        catalog={},
        agent_parent_ids={3: (1, 2)},
    )
    capture = CudaFullEvolutionCompactScienceCaptureV1(
        event_agent_metrics=torch.zeros((1, 21), dtype=torch.int64),
        event_birth_provenance=torch.tensor(
            [list(provenance)], dtype=torch.uint8
        ),
        metric_fields=COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
        birth_provenance_encoding=COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1,
        start_next_absolute_tick=0,
        end_next_absolute_tick=1,
        world_uids=(101,),
        world_offsets=(0, 1),
        event_content_sha256="5" * 64,
    )
    dispatch = CudaFullEvolutionSparseEventDispatchV3(
        packed_events=struct.pack("<QQQQ", 3, 1, 2, 2 << 6),
        world_offsets=(0, 1),
        event_counts=(1,),
        final_next_agent_ids=(4,),
        world_uids=(101,),
        start_next_absolute_tick=0,
        end_next_absolute_tick=1,
        tick_count=1,
        drain_elapsed_seconds=0.0,
        count_projection_elapsed_seconds=0.0,
        compaction_and_transfer_elapsed_seconds=0.0,
        content_hash_elapsed_seconds=0.0,
        content_sha256="5" * 64,
        artifact_sha256="6" * 64,
        manifest_sha256="7" * 64,
    )
    dispatch_receipt = write_compact_science_dispatch_v1(
        tmp_path / "hereditary-dispatch",
        dispatch=dispatch,
        capture=capture,
        world_cells={
            world.world_id: {
                "science_logical_world_id": 101,
                "science_ownership": world.science.to_value(),
                "science_world_sha256": world.semantic_world_sha256.to_dict(),
                "science_acquisition_sha256": world.acquisition_sha256.to_dict(),
            }
        },
        native_world_control_set_sha256="3" * 64,
    )
    body = {
        "schema_version": "eiel_cuda_v2_compact_science_capture_index_v1",
        "observation_abi": "eiel_cuda_v2_compact_science_capture_v1",
        "batch_id": "generic-hereditary-b1",
        "ordered_cell_ids": [world.world_id],
        "native_world_control_set_sha256": "3" * 64,
        "terminal_next_absolute_tick": 1,
        "dispatch_artifacts": [
            {
                "path": dispatch_receipt["path"],
                "artifact_sha256": dispatch_receipt["artifact_sha256"],
            }
        ],
        "snapshot_artifacts": [initial, terminal],
        "checkpoint_frontier": None,
        "dropped_record_count": 0,
    }
    index = {**body, "index_sha256": canonical_sha256(body).hex}
    result = reduce_compact_science_world_atlas_v1(
        index,
        bundle,
        max_workers=2,
        output_root=tmp_path / "hereditary-result",
    )
    attribution = result["world_results"][0]["hereditary_attribution"]
    assert len(attribution["birth_parent_child_records"]) == 1
    assert attribution["integrity"]["parent_child_record_count"] == 1
    assert attribution["integrity"]["funnel_reconciled"] is True
    assert (
        result["world_results"][0]["milestones"][-1][
            "hereditary_frequency_trajectory"
        ]["total_live_count"]
        == 3
    )

    native_reducer = CompactWorldAtlasIncrementalReducerV1(
        bundle,
        max_workers=1,
        native_artifact=randomized_native_artifact,
        output_root=tmp_path / "native-hereditary-result",
    )
    native_route = native_reducer.execution_route
    native_result = native_reducer.finalize(index)
    native_attribution = native_result["world_results"][0][
        "hereditary_attribution"
    ]
    assert native_route == "native_cpp_v1"
    assert native_attribution["frequency_trajectory"] == attribution[
        "frequency_trajectory"
    ]
    assert native_attribution["integrity"]["parent_child_record_count"] == 1
    assert native_attribution["integrity"]["funnel_reconciled"] is True
    native_artifact = native_attribution["birth_parent_child_records"]
    assert isinstance(native_artifact, list)
    assert len(native_artifact) == 1
    assert native_artifact[0]["parent_agent_ids"] == [1, 2]


def test_async_native_abrupt_crash_is_detected_without_orphans(
    tmp_path: Path,
    randomized_native_artifact,
) -> None:
    bundle, _ = _randomized_capture(tmp_path / "source-capture", seed=0xC2A5)
    sidecar = CompactWorldAtlasAsyncSidecarV1(
        tmp_path / "crash-capture",
        bundle,
        **_native_engineering_sidecar_kwargs(randomized_native_artifact),
    )
    pid = sidecar.pid
    assert pid is not None
    os.kill(pid, signal.SIGKILL)
    with pytest.raises(ValueError, match="exited without a result"):
        sidecar.result_bundle(timeout_seconds=5)
    progress = sidecar.progress_snapshot()
    assert progress["sidecar_process"]["alive"] is False
    assert progress["process_cleanup"]["clean"] is True
    assert progress["process_cleanup"]["orphaned_worker_count"] == 0
    assert not Path(f"/proc/{pid}").exists()


def test_full_hereditary_birth_records_are_streamed_and_authenticated(
    tmp_path: Path,
) -> None:
    spool = _HereditaryBirthRecordSpoolV1(
        tmp_path / "reducer-artifacts" / "births.bin.gz",
        world_ids=("world-a", "world-b"),
    )
    for index in range(12_000):
        identity = f"sha256:{index + 1:064x}"
        spool.append(
            {
                "absolute_tick": index,
                "world_id": "world-a" if index % 2 else "world-b",
                "child_agent_id": index + 1,
                "parent_agent_ids": [1, 2],
                "parent_agent_state_ids": [None, None],
                "parent_full_genome_state_ids": [identity, identity],
                "child_full_genome_state_id": identity,
                "parent_tracked_genotype_state_ids": [identity, identity],
                "child_tracked_genotype_state_id": identity,
                "lineage_depth": 1,
                "mutation_origin_count": 0,
                "recombination_pattern": 0,
                "birth_provenance_hex": "00" * 68,
            }
        )
    assert len(spool._buffer) < 1024 * 1024
    artifact = spool.seal()
    artifact.update({"world_id": "world-a", "world_record_count": 6_000})
    verified = validate_hereditary_birth_record_artifact_v1(
        artifact,
        expected_world_id="world-a",
    )
    assert verified["record_count"] == 12_000
    assert verified["world_record_counts"] == {
        "world-a": 6_000,
        "world-b": 6_000,
    }


def test_native_hereditary_terminal_finalization_uses_configured_parallelism(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(2)
    finalized: list[int] = []

    class _FakeNativeState:
        def world_receipt(self, world_index: int):
            return SimpleNamespace(
                birth_count=0,
                death_count=0,
                active_agent_count=1,
            )

        def finalize_hereditary_birth_output(self, world_index: int) -> None:
            barrier.wait(timeout=5)
            finalized.append(world_index)

        def finalize_strategy(self, world_index: int, *, terminal_tick: int) -> None:
            assert terminal_tick == 10

        def strategy_receipt(self, world_index: int):
            return SimpleNamespace(
                genotype_reproductive_attribution={
                    "totals": {
                        "agent_count": 1,
                        "offspring_birth_count": 0,
                    },
                    "descendant_birth_counts_by_parent_tracked_identity": {},
                }
            )

    world_ids = ("generic-world-a", "generic-world-b")
    output_paths = {}
    for world_id in world_ids:
        path = tmp_path / f"{world_id}.bin.gz"
        path.write_bytes(b"generic-hereditary-artifact")
        output_paths[world_id] = path

    reducer = object.__new__(CompactWorldAtlasIncrementalReducerV1)
    reducer._native_state = _FakeNativeState()
    reducer._native_max_workers = 2
    reducer.world_ids = world_ids
    reducer._native_hereditary_worlds = set(world_ids)
    reducer.reconstructions = {
        world_id: {"native_hereditary_birth_output_path": output_paths[world_id]}
        for world_id in world_ids
    }
    reducer.results = {world_id: {"integrity": {}} for world_id in world_ids}

    reducer._finalize_native_hereditary_attribution(terminal_tick=10)

    assert sorted(finalized) == [0, 1]
    assert all(
        reducer.reconstructions[world_id]["native_hereditary_birth_artifact"][
            "immutable"
        ]
        for world_id in world_ids
    )
