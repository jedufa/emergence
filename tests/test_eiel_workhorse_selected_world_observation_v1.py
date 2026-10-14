from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import struct

import pytest

from engine.campaigns import cuda_full_evolution_v2_workhorse_provider as provider_module
from engine.perf.cuda_full_evolution_v2.workhorse import (
    BACKEND_FUSED_CTA_TEMPORAL_V2,
    SUPPORTED_WORKHORSE_SHAPES_V2,
    FullEvolutionCudaWorkhorseContractV2,
    WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1,
    WorkhorseSelectedWorldObservationError,
    seal_workhorse_selected_world_observation_v1,
    validate_workhorse_selected_world_observation_v1,
    workhorse_layout_v2,
    write_workhorse_indexed_checkpoint_set_v1,
)
from engine.perf.cuda_full_evolution_v2.workhorse import indexed_checkpoint
from engine.perf.cuda_full_evolution_v2.workhorse import (
    selected_world_observation as observation_module,
)


def _pack(state: bytearray, layout: object, field: str, values: list[int]) -> None:
    descriptor = layout.field(field)
    wire = {"u8": "B", "u16": "H", "u32": "I", "i32": "i", "u64": "Q"}[
        descriptor.dtype
    ]
    struct.pack_into(
        f"<{len(values)}{wire}", state, descriptor.offset, *values
    )


def _indexed_terminal(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], FullEvolutionCudaWorkhorseContractV2, Path]:
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=1,
        temporal_k=1,
        total_ticks=5,
        start_next_tick=25,
    )
    layout = workhorse_layout_v2(shape)
    state = bytearray(layout.state_bytes_per_world)
    header = [0] * 32
    header[0] = 1902
    header[1] = 30
    header[2] = 203
    header[3] = 2
    header[4] = 0
    header[5] = 30
    header[6:13] = [11, 12, 13, 14, 3, 2, 1]
    header[13] = 302
    header[14] = 4
    header[15:20] = [3, 1, 2, 1, 4]
    _pack(state, layout, "header", header)
    occupancy = [0xFFFFFFFF] * shape.sites
    occupancy[3] = 0
    occupancy[7] = 1
    _pack(state, layout, "occupancy", occupancy)
    _pack(state, layout, "resource", [2] * shape.sites)
    _pack(state, layout, "patch_field", [1] * shape.sites)
    _pack(state, layout, "alive", [1, 1] + [0] * (shape.capacity - 2))
    _pack(
        state,
        layout,
        "active_slots_by_uid",
        [0, 1] + [0xFFFFFFFF] * (shape.capacity - 2),
    )
    for field, values in (
        ("position", [3, 7]),
        ("energy", [90, 120]),
        ("health", [200, 230]),
        ("age", [8, 3]),
        ("birth_tick", [2, 27]),
        ("agent_id", [101, 202]),
        ("parent_a_id", [0, 101]),
        ("parent_b_id", [0, 88]),
    ):
        _pack(state, layout, field, values + [0] * (shape.capacity - len(values)))
    genome_a = [0] * (shape.capacity * 32)
    genome_b = [0] * (shape.capacity * 32)
    expressed = [0] * (shape.capacity * 32)
    for slot, offset in ((0, 10), (1, 40)):
        for locus in range(32):
            genome_a[slot * 32 + locus] = (offset + locus) & 0xFF
            genome_b[slot * 32 + locus] = (offset + locus + 1) & 0xFF
            expressed[slot * 32 + locus] = (offset + locus) & 0xFF
    _pack(state, layout, "genome_a", genome_a)
    _pack(state, layout, "genome_b", genome_b)
    _pack(state, layout, "expressed", expressed)

    physical_root = root / "physical"
    physical_root.mkdir(parents=True)
    state_path = physical_root / "canonical_state.bin"
    state_path.write_bytes(state)
    state_path.chmod(0o444)
    manifest_path = physical_root / "checkpoint.json"
    manifest_path.write_text("{}\n", encoding="ascii")
    manifest_path.chmod(0o444)
    state_sha256 = hashlib.sha256(state).hexdigest()
    metadata = {
        "contract": contract.to_dict(),
        "checkpoint_at_tick": 30,
        "bundle_body_sha256": "1" * 64,
        "checkpoint_identity": {"checkpoint_identity_sha256": "2" * 64},
        "canonical_state": {"sha256": state_sha256, "bytes": len(state)},
        "world_uids": [1902],
        "rng_keys": [302],
        "artifact": {
            "identity_sha256": "3" * 64,
            "sealed_artifact_identity_sha256": "4" * 64,
        },
        "attempt_lineage_sha256": "5" * 64,
        "dispatch_identity_sha256": "6" * 64,
        "event_sink_ack": {
            "receipt_path": str((physical_root / "event.json").resolve()),
            "receipt_file_sha256": "7" * 64,
        },
    }
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda _path, **_kwargs: (deepcopy(metadata), state_path),
    )
    indexed = write_workhorse_indexed_checkpoint_set_v1(
        manifest_path, root / "indexed"
    )
    return indexed, contract, state_path


def _result(root: Path, contract: FullEvolutionCudaWorkhorseContractV2) -> dict[str, object]:
    result_path = root / "continuation-result.json"
    result_path.write_text("{}\n", encoding="ascii")
    result_path.chmod(0o444)
    logical = {
        "science_continuation_identity_sha256": "8" * 64,
        "world_count": 1,
        "source_checkpoint_occurrence_sha256": "9" * 64,
        "source_parent_occurrence_sha256": "a" * 64,
        "source_semantic_world_sha256": "b" * 64,
        "source_control_history_sha256": "c" * 64,
        "source_rng_state_sha256": "d" * 64,
        "source_event_prefix_sha256": "e" * 64,
        "source_active_control_state_sha256": "f" * 64,
    }
    measurements = []
    for ordinal, (start, end, count, digest) in enumerate(
        ((25, 28, 2, "1"), (28, 30, 2, "2"))
    ):
        measurements.append(
            {
                "sample_index": ordinal,
                "event_sink_receipt": {
                    "world_uids": [1902],
                    "event_counts": [count],
                    "start_next_absolute_tick": start,
                    "end_next_absolute_tick": end,
                    "receipt_file_sha256": digest * 64,
                    "receipt_body_sha256": "3" * 64,
                    "dispatch_content_sha256": "4" * 64,
                    "post_state_sha256": "5" * 64,
                },
            }
        )
    return {
        "receipt_path": str(result_path.resolve()),
        "receipt_file_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "receipt_content_sha256": "6" * 64,
        "outcome": "CONTINUED",
        "continuation_contract": {
            "science_continuation_identity": logical,
            "physical_continuation_plan": {"target_contract": contract.to_dict()},
        },
        "absolute_frontiers": {
            "start_next_tick": 25,
            "target_next_tick": 30,
            "committed_next_tick": 30,
        },
        "restored_world_uids": [1902],
        "dispatch_measurements": measurements,
        "control_application": {
            "control_contract_sha256": "7" * 64,
            "applied_next_absolute_tick": 25,
            "first_permitted_divergence_tick": 25,
            "ordered_native_fields": [
                "resource_abundance_multiplier_q16",
                "mutation_threshold_u16",
            ],
            "requested_native_rows": [[65536, 8192]],
            "readback_native_rows": [[65536, 8192]],
            "control_application_sha256": "8" * 64,
        },
    }


def test_selected_world_host_observation_is_exact_compact_and_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexed, contract, _state_path = _indexed_terminal(tmp_path, monkeypatch)
    result = _result(tmp_path, contract)
    monkeypatch.setattr(
        observation_module,
        "validate_workhorse_indexed_continuation_result_v2",
        lambda _path: deepcopy(result),
    )
    observation = seal_workhorse_selected_world_observation_v1(
        tmp_path / "selected-world-observation.json",
        execution_result_path=result["receipt_path"],
        terminal_indexed_checkpoint_manifest_path=indexed["manifest_path"],
        source_checkpoint_occurrence_sha256="0" * 64,
        source_checkpoint_occurrence_projection_sha256="1" * 64,
        cell_id="cell-selected-world",
    )

    assert (
        observation["observation_mode"]
        == WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1
    )
    assert observation["integrity"] == {
        "world_count": 1,
        "dropped_record_count": 0,
        "checkpoint_state_reopened_exact": True,
        "event_acknowledgements_reopened_exact": True,
        "control_readback_authenticated": True,
        "runtime_observation_mode_remains_off": True,
        "host_boundary_adapter_is_observation_only": True,
    }
    world = observation["world_observations"][0]
    assert world["active_population"] == world["alive_count"] == 2
    assert world["cumulative_counters"]["birth_count"] == 3
    assert world["cumulative_counters"]["death_count"] == 1
    assert world["cumulative_counters"]["mutation_count"] == 2
    examples = world["bounded_active_agent_examples"]
    assert examples["complete"] is True
    assert examples["rows"][1]["parent_a_id"] == 101
    assert world["active_genotype_summary"]["unique_genotype_count"] == 2
    assert world["active_lineage_summary"]["active_founder_count"] == 1
    coverage = observation["event_observation_coverage"]
    assert coverage["total_event_count"] == 4
    assert coverage["event_kind_rows_available"] is False
    assert coverage["birth_death_causal_rows_claimed"] is False
    assert observation["named_phenotype_reward_authority_granted"] is False

    reopened = validate_workhorse_selected_world_observation_v1(
        observation["observation_path"],
        expected_checkpoint_occurrence_sha256="0" * 64,
        expected_checkpoint_occurrence_projection_sha256="1" * 64,
    )
    assert reopened["observation_content_sha256"] == observation[
        "observation_content_sha256"
    ]
    resumed = seal_workhorse_selected_world_observation_v1(
        observation["observation_path"],
        execution_result_path=result["receipt_path"],
        terminal_indexed_checkpoint_manifest_path=indexed["manifest_path"],
        source_checkpoint_occurrence_sha256="0" * 64,
        source_checkpoint_occurrence_projection_sha256="1" * 64,
        cell_id="cell-selected-world",
    )
    assert resumed == reopened
    with pytest.raises(
        WorkhorseSelectedWorldObservationError,
        match="another terminal occurrence",
    ):
        validate_workhorse_selected_world_observation_v1(
            observation["observation_path"],
            expected_checkpoint_occurrence_sha256="2" * 64,
        )


def test_selected_world_host_observation_fails_closed_on_source_identity_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexed, contract, _state_path = _indexed_terminal(tmp_path, monkeypatch)
    result = _result(tmp_path, contract)
    current = deepcopy(result)
    monkeypatch.setattr(
        observation_module,
        "validate_workhorse_indexed_continuation_result_v2",
        lambda _path: deepcopy(current),
    )
    observation = seal_workhorse_selected_world_observation_v1(
        tmp_path / "selected-world-observation.json",
        execution_result_path=result["receipt_path"],
        terminal_indexed_checkpoint_manifest_path=indexed["manifest_path"],
        source_checkpoint_occurrence_sha256="0" * 64,
        source_checkpoint_occurrence_projection_sha256="1" * 64,
        cell_id="cell-selected-world",
    )
    current["continuation_contract"]["science_continuation_identity"][
        "source_control_history_sha256"
    ] = "2" * 64
    with pytest.raises(
        WorkhorseSelectedWorldObservationError,
        match="does not rederive",
    ):
        validate_workhorse_selected_world_observation_v1(
            observation["observation_path"]
        )


def test_provider_terminalization_reopens_and_binds_selected_world_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=1,
        temporal_k=1,
        total_ticks=5,
        start_next_tick=25,
    )
    batch_root = tmp_path / "batch"
    batch_root.mkdir()
    manifest_path = batch_root / "manifest.json"
    result_path = batch_root / "workhorse-indexed-continuation-result.json"
    observation_path = batch_root / "workhorse-selected-world-host-observation.json"
    result_path.write_text("{}\n", encoding="ascii")
    observation_path.write_text("{}\n", encoding="ascii")
    result_path.chmod(0o444)
    observation_path.chmod(0o444)
    logical = {
        "science_continuation_identity_sha256": "1" * 64,
        "source_semantic_world_sha256": "2" * 64,
    }
    continuation = {
        "science_continuation_identity": logical,
        "physical_continuation_plan": {
            "target_contract": contract.to_dict(),
            "physical_continuation_plan_sha256": "3" * 64,
        },
        "checkpoint_ticks": [30],
    }
    operation = {
        "operation": "INDEXED_ACS_CONTINUATION",
        "contract": contract.to_dict(),
        "source_checkpoint_identity_sha256": "4" * 64,
        "continuation_contract": continuation,
    }
    result = {
        "receipt_file_sha256": "5" * 64,
        "receipt_content_sha256": "6" * 64,
        "absolute_frontiers": {
            "start_next_tick": 25,
            "target_next_tick": 30,
            "committed_next_tick": 30,
        },
        "aggregate_world_ticks_per_second": 12.5,
    }
    occurrence = "7" * 64
    projection = "8" * 64
    observation = {
        "observation_mode": WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1,
        "path": str(observation_path.resolve()),
        "file_sha256": "9" * 64,
        "content_sha256": "a" * 64,
        "file_bytes": observation_path.stat().st_size,
        "checkpoint_occurrence_sha256": occurrence,
        "checkpoint_occurrence_projection_sha256": projection,
    }
    provider_result = {
        "schema_version": "eiel_workhorse_indexed_continuation_provider_result_v2",
        "operation": operation["operation"],
        "shape_id": contract.shape_id,
        "backend_id": contract.backend_id,
        "B": 1,
        "preferred_temporal_k": 1,
        "source_world_state_set_sha256": "4" * 64,
        "science_continuation_identity_sha256": "1" * 64,
        "physical_continuation_plan_sha256": "3" * 64,
        "execution_result_path": str(result_path.resolve()),
        "execution_result_file_sha256": "5" * 64,
        "execution_result_content_sha256": "6" * 64,
        "absolute_frontiers": result["absolute_frontiers"],
        "checkpoint_ticks": [30],
        "aggregate_world_ticks_per_second": 12.5,
        "indexed_checkpoint_receipts": [{}],
        "acs_checkpoint_staging": {},
        "host_boundary_observation": observation,
    }
    publication = {
        "occurrence_digest": occurrence,
        "checkpoint_occurrence_projection_content_sha256": projection,
    }
    monkeypatch.setattr(
        provider_module,
        "validate_workhorse_indexed_continuation_result_v2",
        lambda _path: deepcopy(result),
    )
    monkeypatch.setattr(
        provider_module,
        "_indexed_continuation_publication_seals_v2",
        lambda **_kwargs: [deepcopy(publication)],
    )
    monkeypatch.setattr(
        provider_module,
        "validate_workhorse_selected_world_observation_v1",
        lambda _path, **_kwargs: {
            "observation_file_sha256": "9" * 64,
            "observation_content_sha256": "a" * 64,
            "observation_file_bytes": observation_path.stat().st_size,
        },
    )
    terminal_receipt = {
        "receipt_path": str((batch_root / "terminal.json").resolve()),
        "receipt_file_sha256": "b" * 64,
        "receipt_content_sha256": "c" * 64,
        "indexed_checkpoint_publications": [publication],
    }
    monkeypatch.setattr(
        provider_module,
        "seal_workhorse_terminalized_indexed_continuation_receipt_v2",
        lambda *_args, **_kwargs: deepcopy(terminal_receipt),
    )

    finalized = provider_module._finalize_indexed_continuation_provider_result_v2(
        provider_result=provider_result,
        operation=operation,
        manifest_path=manifest_path,
        canonical_terminal_binding={"attempt_id": "ATTEMPT-SELECTED-WORLD"},
    )

    binding = finalized["host_boundary_observation"]
    assert binding["terminally_reopened"] is True
    assert binding["runtime_observation_mode"] == "OFF"
    assert binding["host_adapter_observation_only"] is True
