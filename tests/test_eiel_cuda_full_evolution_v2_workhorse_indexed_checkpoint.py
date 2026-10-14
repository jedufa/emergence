from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import struct

import pytest

from engine.campaigns import cuda_full_evolution_v2_workhorse_provider as provider_module
from engine.perf.cuda_full_evolution_v2.workhorse import (
    BACKEND_FUSED_CTA_TEMPORAL_V2,
    SUPPORTED_WORKHORSE_SHAPES_V2,
    FullEvolutionCudaWorkhorseContractV2,
    WorkhorseIndexedContinuationError,
    WorkhorseIndexedCheckpointError,
    WORKHORSE_INDEXED_CHECKPOINT_PUBLICATION_SCHEMA_V2,
    compile_workhorse_absolute_dispatch_plan_v2,
    inspect_workhorse_indexed_checkpoint_set_v1,
    validate_workhorse_repack_binding_v2,
    validate_workhorse_indexed_continuation_contract_v2,
    validate_workhorse_indexed_checkpoint_publication_seals_v2,
    workhorse_backend_v2,
    workhorse_layout_v2,
    workhorse_repack_binding_v2,
    workhorse_indexed_continuation_contract_v2,
    write_workhorse_indexed_checkpoint_set_v1,
)
from engine.perf.cuda_full_evolution_v2.workhorse.contract import canonical_sha256
from engine.perf.cuda_full_evolution_v2.workhorse import indexed_checkpoint
from engine.perf.cuda_full_evolution_v2.workhorse import continuation as continuation_module
from engine.perf.cuda_full_evolution_v2.workhorse import canonical_rows
from engine.perf.cuda_full_evolution_v2.workhorse import authority as workhorse_authority
from engine.perf.cuda_full_evolution_v2.workhorse import layout as workhorse_layout
from engine.perf.cuda_full_evolution_v2.workhorse import probe
from engine.perf.cuda_full_evolution_v2.workhorse.scientific_control import (
    workhorse_active_scientific_control_state_v1,
)
from engine.perf.cuda_full_evolution_v2 import benchmark as workhorse_benchmark


def _physical_checkpoint(
    root: Path,
    *,
    world_uids: list[int],
    rng_keys: list[int],
    temporal_k: int,
    tick: int = 25,
) -> tuple[dict[str, object], Path, Path]:
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=len(world_uids),
        temporal_k=temporal_k,
        total_ticks=tick,
        start_next_tick=0,
    )
    row_bytes = workhorse_layout_v2(shape).state_bytes_per_world
    root.mkdir(parents=True)
    state_path = root / "canonical_state.bin"
    aggregate = hashlib.sha256()
    with state_path.open("xb") as handle:
        for world_uid, rng_key in zip(world_uids, rng_keys, strict=True):
            state = bytearray(row_bytes)
            header = [0] * 32
            header[0] = world_uid
            header[1] = tick
            header[4] = 0
            header[13] = rng_key
            header[14] = 4
            struct.pack_into("<32Q", state, 0, *header)
            state[32 * 8 : 32 * 8 + 8] = world_uid.to_bytes(8, "little")
            handle.write(state)
            aggregate.update(state)
    state_path.chmod(0o444)
    manifest_path = root / "checkpoint.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest_path.chmod(0o444)
    metadata: dict[str, object] = {
        "contract": contract.to_dict(),
        "checkpoint_at_tick": tick,
        "bundle_body_sha256": "1" * 64,
        "checkpoint_identity": {"checkpoint_identity_sha256": "2" * 64},
        "canonical_state": {
            "sha256": aggregate.hexdigest(),
            "bytes": len(world_uids) * row_bytes,
        },
        "world_uids": list(world_uids),
        "rng_keys": list(rng_keys),
        "artifact": {
            "identity_sha256": "3" * 64,
            "sealed_artifact_identity_sha256": "4" * 64,
        },
        "attempt_lineage_sha256": "5" * 64,
        "dispatch_identity_sha256": "6" * 64,
        "event_sink_ack": {
            "receipt_path": str((root / "event-ack.json").resolve()),
            "receipt_file_sha256": "7" * 64,
        },
    }
    return metadata, manifest_path, state_path


def test_indexed_checkpoint_keeps_one_payload_and_reopens_every_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, manifest_path, state_path = _physical_checkpoint(
        tmp_path / "physical",
        world_uids=[903, 901, 902],
        rng_keys=[73, 71, 72],
        temporal_k=4,
    )
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (deepcopy(metadata), state_path),
    )

    written = write_workhorse_indexed_checkpoint_set_v1(
        manifest_path, tmp_path / "indexed"
    )
    reopened, reopened_state = inspect_workhorse_indexed_checkpoint_set_v1(
        written["manifest_path"]
    )

    assert reopened_state == state_path
    assert reopened["storage_model"] == {
        "canonical_state_payload_copies": 1,
        "indexed_world_payload_copies": 0,
        "world_rows_are_range_references": True,
        "source_payload_must_remain_live": True,
    }
    assert [row["world_uid"] for row in reopened["world_index_by_uid"]] == [
        901,
        902,
        903,
    ]
    assert list((tmp_path / "indexed").iterdir()) == [
        tmp_path / "indexed" / "checkpoint-index.json"
    ]


def test_selected_world_checkpoint_is_independent_and_repackable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, manifest_path, state_path = _physical_checkpoint(
        tmp_path / "physical-selected",
        world_uids=[1901, 1902, 1903],
        rng_keys=[301, 302, 303],
        temporal_k=4,
    )
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (deepcopy(metadata), state_path),
    )
    indexed = write_workhorse_indexed_checkpoint_set_v1(
        manifest_path, tmp_path / "indexed-selected"
    )
    projection_core = {
        "schema_version": "eiel_workhorse_row_checkpoint_occurrence_projection_v2",
        "claim_class": "TECHNICAL_ONLY",
        "science_authority_granted": False,
        "qualification_authority_granted": False,
        "time_authority": "absolute_tick_v1",
        "checkpoint_at_tick": 25,
        "checkpoint_manifest_path": str(manifest_path),
        "checkpoint_manifest_file_sha256": "1" * 64,
        "checkpoint_bundle_body_sha256": "2" * 64,
        "checkpoint_identity_sha256": indexed["source_physical_checkpoint"][
            "checkpoint_identity_sha256"
        ],
        "canonical_state_sha256": "3" * 64,
        "canonical_parent_binding_sha256": "4" * 64,
        "operation_contract_sha256": "5" * 64,
        "absolute_tick_bundle_path": str(tmp_path / "bundle"),
        "absolute_tick_bundle_digest": {"algorithm": "sha256", "hex": "6" * 64},
        "occurrence_root": str(tmp_path / "occurrence"),
        "occurrence_digest": {"algorithm": "sha256", "hex": "8" * 64},
        "occurrence_descriptor_path": str(tmp_path / "occurrence" / "descriptor.json"),
        "occurrence_descriptor_file_sha256": "a" * 64,
        "occurrence_commit_path": str(tmp_path / "occurrence" / "commit.json"),
        "occurrence_commit_file_sha256": "b" * 64,
        "parent_checkpoint_digest": {"algorithm": "sha256", "hex": "9" * 64},
        "canonical_visible": True,
        "restore_admitted_exact": False,
    }
    projection = {
        **projection_core,
        "projection_content_sha256": canonical_sha256(projection_core),
    }
    selected = indexed_checkpoint.materialize_workhorse_selected_world_checkpoint_v1(
        indexed["manifest_path"],
        tmp_path / "retained-parent",
        world_uid=1902,
        checkpoint_occurrence_projection=projection,
        semantic_world_sha256="a" * 64,
        control_history_sha256="b" * 64,
        rng_state_sha256="c" * 64,
        event_prefix_sha256="d" * 64,
        active_scientific_control_state=(
            workhorse_active_scientific_control_state_v1(
                captured_next_absolute_tick=25,
                resource_multiplier=1.0,
                mutation_rate=0.125,
            )
        ),
    )
    reopened, selected_state = (
        indexed_checkpoint.inspect_workhorse_selected_world_checkpoint_v1(
            selected["manifest_path"]
        )
    )
    row_bytes = workhorse_layout_v2(SUPPORTED_WORKHORSE_SHAPES_V2[0]).state_bytes_per_world
    with state_path.open("rb") as handle:
        handle.seek(row_bytes)
        expected = handle.read(row_bytes)
    assert selected_state.read_bytes() == expected
    assert reopened["selected_world"]["world_uid"] == 1902
    assert reopened["selected_world"]["rng_key"] == 302
    assert reopened["source_indexed_checkpoint"]["world_count"] == 3
    assert json.loads(Path(selected["authority_receipt_path"]).read_text()) == projection

    # Retention is deliberately self-contained: neither the aggregate payload
    # nor its range index is a restore dependency after exact materialization.
    Path(indexed["manifest_path"]).parent.rename(tmp_path / "released-index")
    state_path.parent.rename(tmp_path / "released-physical")
    assert (
        indexed_checkpoint.inspect_workhorse_selected_world_checkpoint_v1(
            selected["manifest_path"]
        )[1].read_bytes()
        == expected
    )
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    target = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=1,
        temporal_k=2,
        total_ticks=5,
        start_next_tick=25,
    )
    repack = workhorse_repack_binding_v2(
        [selected["manifest_path"]],
        ordered_world_uids=[1902],
        target_contract=target,
    )
    assert validate_workhorse_repack_binding_v2(repack) == repack
    assert repack["continuation_scope"] == "EXPERIMENTAL_SUBSET_FORK"
    assert repack["source_world_count"] == 3
    assert repack["source_locators"][0]["source_state_path"] == str(
        selected_state
    )

    dispatch = compile_workhorse_absolute_dispatch_plan_v2(
        start_next_tick=25,
        target_next_tick=30,
        preferred_temporal_k=2,
        admitted_temporal_k=[1, 2],
        boundary_next_ticks=[28, 30],
    )
    with pytest.raises(
        WorkhorseIndexedContinuationError,
        match="requires explicit resource and mutation controls",
    ):
        workhorse_indexed_continuation_contract_v2(
            science_continuation_contract_sha256="e" * 64,
            repack_binding=repack,
            dispatch_plan=dispatch,
            checkpoint_ticks=[28, 30],
        )
    controls = continuation_module.workhorse_scientific_control_contract_v1(
        effective_next_absolute_tick=25,
        resource_multiplier=1.5,
        mutation_rate=0.25,
    )
    continuation = workhorse_indexed_continuation_contract_v2(
        science_continuation_contract_sha256="e" * 64,
        repack_binding=repack,
        dispatch_plan=dispatch,
        checkpoint_ticks=[28, 30],
        scientific_control_contract=controls,
    )
    assert validate_workhorse_indexed_continuation_contract_v2(continuation) == (
        continuation
    )
    logical = continuation["science_continuation_identity"]
    assert logical["source_checkpoint_occurrence_sha256"] == "8" * 64
    assert logical["source_parent_occurrence_sha256"] == "9" * 64
    assert logical["source_semantic_world_sha256"] == "a" * 64
    assert logical["source_control_history_sha256"] == "b" * 64
    assert logical["source_rng_state_sha256"] == "c" * 64
    assert logical["source_event_prefix_sha256"] == "d" * 64
    assert logical["source_active_control_state_sha256"] == reopened[
        "active_scientific_control_state"
    ]["active_control_state_sha256"]
    assert controls["native_mapping"]["ordered_values"] == [16384, 98304]

    unsupported = deepcopy(controls)
    unsupported["controls"]["named.target.phenotype"] = 1.0
    with pytest.raises(
        WorkhorseIndexedContinuationError, match="unsupported controls"
    ):
        continuation_module.validate_workhorse_scientific_control_contract_v1(
            unsupported
        )


def test_native_scientific_control_source_has_closed_tick_bound_one_shot_abi() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_full_evolution_v2/workhorse/native/full_evolution_workhorse_v2.cu"
    ).read_text(encoding="utf-8")
    assert "kScienceMutationThresholdControl = 16" in source
    assert "kScienceResourceMultiplierQ16Control = 17" in source
    assert "kScienceControlsApplied = 18" in source
    assert "header[kNextAbsoluteTick] != expected_next_absolute_tick" in source
    assert "prior[kScienceControlsApplied] != 0" in source
    assert "scientific controls require a fresh owner before graph preparation" in source
    assert "science_control_abi_version_v1" in source
    assert "source[kScienceControlsApplied]" in source
    assert "const bool default_rate = mutation_threshold ==" in source
    assert "(mutation_draw & 7u) == 0u" in source


def test_checkpoint_publication_seal_survives_payload_release_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, manifest_path, state_path = _physical_checkpoint(
        tmp_path / "physical",
        world_uids=[2101, 2102],
        rng_keys=[171, 172],
        temporal_k=2,
    )
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (deepcopy(metadata), state_path),
    )
    indexed = write_workhorse_indexed_checkpoint_set_v1(
        manifest_path, tmp_path / "indexed"
    )
    contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
        metadata["contract"]
    )
    body = {
        "schema_version": WORKHORSE_INDEXED_CHECKPOINT_PUBLICATION_SCHEMA_V2,
        "checkpoint_at_tick": 25,
        "canonical_state_bytes": metadata["canonical_state"]["bytes"],
        "canonical_state_sha256": metadata["canonical_state"]["sha256"],
        "checkpoint_bundle_body_sha256": metadata["bundle_body_sha256"],
        "indexed_manifest_file_sha256": indexed["manifest_file_sha256"],
        "indexed_manifest_body_sha256": indexed["manifest_body_sha256"],
        "world_state_set_identity": indexed["world_state_set_identity"],
        "checkpoint_occurrence_projection_content_sha256": "8" * 64,
        "absolute_tick_bundle_digest": "9" * 64,
        "occurrence_digest": "a" * 64,
    }
    seal = {**body, "publication_seal_sha256": canonical_sha256(body)}
    assert validate_workhorse_indexed_checkpoint_publication_seals_v2(
        [seal], contract=contract, checkpoint_ticks=[25]
    ) == [seal]

    state_path.unlink()
    assert validate_workhorse_indexed_checkpoint_publication_seals_v2(
        [seal], contract=contract, checkpoint_ticks=[25]
    ) == [seal]

    hostile = deepcopy(seal)
    hostile["world_state_set_identity"]["worlds_by_uid"][0][
        "state_sha256"
    ] = "b" * 64
    hostile_body = {
        key: value
        for key, value in hostile.items()
        if key != "publication_seal_sha256"
    }
    hostile["publication_seal_sha256"] = canonical_sha256(hostile_body)
    with pytest.raises(
        WorkhorseIndexedContinuationError,
        match="publication identity changed",
    ):
        validate_workhorse_indexed_checkpoint_publication_seals_v2(
            [hostile], contract=contract, checkpoint_ticks=[25]
        )


def test_provider_reopens_live_publications_before_compacting_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, manifest_path, state_path = _physical_checkpoint(
        tmp_path / "physical-provider",
        world_uids=[2201, 2202],
        rng_keys=[181, 182],
        temporal_k=2,
    )
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (deepcopy(metadata), state_path),
    )
    monkeypatch.setattr(
        provider_module,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (deepcopy(metadata), state_path),
    )
    indexed = write_workhorse_indexed_checkpoint_set_v1(
        manifest_path, tmp_path / "indexed-provider"
    )
    contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
        metadata["contract"]
    )
    detached = {
        "schema_version": "eiel_cuda_full_evolution_v2_workhorse_detached_checkpoint_result_v1",
        "status": "complete",
        "path": str(manifest_path.parent.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "state_path": str(state_path.resolve()),
        "checkpoint_at_tick": 25,
        "canonical_state_bytes": metadata["canonical_state"]["bytes"],
        "canonical_state_sha256": metadata["canonical_state"]["sha256"],
        "bundle_body_sha256": metadata["bundle_body_sha256"],
    }
    projection = {
        "checkpoint_at_tick": 25,
        "canonical_state_sha256": metadata["canonical_state"]["sha256"],
        "projection_content_sha256": "c" * 64,
        "absolute_tick_bundle_digest": {"algorithm": "sha256", "hex": "d" * 64},
        "occurrence_digest": {"algorithm": "sha256", "hex": "e" * 64},
    }
    receipt = {
        **detached,
        "checkpoint_occurrence_projection": projection,
        "indexed_checkpoint": {
            "manifest_path": indexed["manifest_path"],
            "manifest_file_sha256": indexed["manifest_file_sha256"],
            "manifest_body_sha256": indexed["manifest_body_sha256"],
            "world_state_set_identity": indexed["world_state_set_identity"],
            "storage_model": indexed["storage_model"],
        },
    }
    result = {
        "indexed_checkpoint_receipts": [receipt],
        "acs_checkpoint_staging": {
            "closed": True,
            "writer_alive": False,
            "failure_latched": False,
            "checkpoint_receipts": [detached],
        },
    }
    publications = provider_module._indexed_continuation_publication_seals_v2(
        provider_result=result,
        contract=contract,
        checkpoint_ticks=[25],
    )
    assert publications[0]["world_state_set_identity"] == indexed[
        "world_state_set_identity"
    ]
    assert publications[0]["absolute_tick_bundle_digest"] == "d" * 64
    assert publications[0]["occurrence_digest"] == "e" * 64
    assert publications[0]["publication_seal_sha256"]


def test_world_state_set_identity_excludes_physical_order_and_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, first_manifest, first_state = _physical_checkpoint(
        tmp_path / "physical-a",
        world_uids=[1001, 1002, 1003],
        rng_keys=[81, 82, 83],
        temporal_k=1,
    )
    second, second_manifest, second_state = _physical_checkpoint(
        tmp_path / "physical-b",
        world_uids=[1003, 1001, 1002],
        rng_keys=[83, 81, 82],
        temporal_k=4,
    )
    by_path = {
        first_manifest.resolve(): (first, first_state),
        second_manifest.resolve(): (second, second_state),
    }
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (
            deepcopy(by_path[Path(path).resolve()][0]),
            by_path[Path(path).resolve()][1],
        ),
    )

    indexed_a = write_workhorse_indexed_checkpoint_set_v1(
        first_manifest, tmp_path / "indexed-a"
    )
    indexed_b = write_workhorse_indexed_checkpoint_set_v1(
        second_manifest, tmp_path / "indexed-b"
    )

    assert indexed_a["world_state_set_identity"] == indexed_b[
        "world_state_set_identity"
    ]
    assert indexed_a["source_physical_checkpoint"] != indexed_b[
        "source_physical_checkpoint"
    ]


def test_indexed_checkpoint_detects_source_world_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, manifest_path, state_path = _physical_checkpoint(
        tmp_path / "physical",
        world_uids=[1201, 1202],
        rng_keys=[91, 92],
        temporal_k=2,
    )
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (deepcopy(metadata), state_path),
    )
    written = write_workhorse_indexed_checkpoint_set_v1(
        manifest_path, tmp_path / "indexed"
    )
    state_path.chmod(0o644)
    with state_path.open("r+b") as handle:
        handle.seek(32 * 8 + 3)
        handle.write(b"\xff")
    state_path.chmod(0o444)

    with pytest.raises(WorkhorseIndexedCheckpointError, match="world row changed"):
        inspect_workhorse_indexed_checkpoint_set_v1(written["manifest_path"])


def test_indexed_checkpoint_manifest_tampering_fails_before_source_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, manifest_path, state_path = _physical_checkpoint(
        tmp_path / "physical",
        world_uids=[1301],
        rng_keys=[101],
        temporal_k=1,
    )
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (deepcopy(metadata), state_path),
    )
    written = write_workhorse_indexed_checkpoint_set_v1(
        manifest_path, tmp_path / "indexed"
    )
    index_path = Path(written["manifest_path"])
    index_path.parent.chmod(0o755)
    index_path.chmod(0o644)
    value = json.loads(index_path.read_text(encoding="utf-8"))
    value["world_index_by_uid"][0]["state_offset"] = 1
    index_path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    index_path.chmod(0o444)
    index_path.parent.chmod(0o555)

    with pytest.raises(WorkhorseIndexedCheckpointError, match="schema, seal"):
        inspect_workhorse_indexed_checkpoint_set_v1(index_path)


def test_repack_binding_gathers_multiple_sources_in_new_physical_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, first_manifest, first_state = _physical_checkpoint(
        tmp_path / "physical-a",
        world_uids=[1401, 1402],
        rng_keys=[111, 112],
        temporal_k=4,
    )
    second, second_manifest, second_state = _physical_checkpoint(
        tmp_path / "physical-b",
        world_uids=[1403, 1404],
        rng_keys=[113, 114],
        temporal_k=2,
    )
    by_path = {
        first_manifest.resolve(): (first, first_state),
        second_manifest.resolve(): (second, second_state),
    }
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (
            deepcopy(by_path[Path(path).resolve()][0]),
            by_path[Path(path).resolve()][1],
        ),
    )
    indexed_a = write_workhorse_indexed_checkpoint_set_v1(
        first_manifest, tmp_path / "indexed-a"
    )
    indexed_b = write_workhorse_indexed_checkpoint_set_v1(
        second_manifest, tmp_path / "indexed-b"
    )
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    target = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=4,
        temporal_k=1,
        total_ticks=17,
        start_next_tick=25,
    )

    binding = workhorse_repack_binding_v2(
        [indexed_a["manifest_path"], indexed_b["manifest_path"]],
        ordered_world_uids=[1404, 1401, 1403, 1402],
        target_contract=target,
    )

    assert validate_workhorse_repack_binding_v2(binding) == binding
    assert binding["continuation_scope"] == "FULL_SET_CONTINUATION"
    assert binding["checkpoint_at_tick"] == 25
    assert [
        row["world_uid"]
        for row in binding["selected_worlds_in_physical_order"]
    ] == [1404, 1401, 1403, 1402]
    assert [row["target_physical_slot"] for row in binding["source_locators"]] == [
        0,
        1,
        2,
        3,
    ]

    reordered = workhorse_repack_binding_v2(
        [indexed_a["manifest_path"], indexed_b["manifest_path"]],
        ordered_world_uids=[1401, 1402, 1403, 1404],
        target_contract=target,
    )
    assert (
        binding["selected_world_state_set_identity"]
        == reordered["selected_world_state_set_identity"]
    )
    assert binding["repack_binding_sha256"] != reordered["repack_binding_sha256"]


def test_continuation_science_identity_excludes_physical_order_and_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, manifest_path, state_path = _physical_checkpoint(
        tmp_path / "physical",
        world_uids=[1451, 1452, 1453, 1454],
        rng_keys=[211, 212, 213, 214],
        temporal_k=4,
    )
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (deepcopy(metadata), state_path),
    )
    indexed = write_workhorse_indexed_checkpoint_set_v1(
        manifest_path, tmp_path / "indexed"
    )
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]

    def build(order: list[int], preferred_k: int) -> dict[str, object]:
        target = FullEvolutionCudaWorkhorseContractV2(
            shape_id=shape.shape_id,
            backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
            B=4,
            temporal_k=preferred_k,
            total_ticks=17,
            start_next_tick=25,
        )
        repack = workhorse_repack_binding_v2(
            [indexed["manifest_path"]],
            ordered_world_uids=order,
            target_contract=target,
        )
        dispatch = compile_workhorse_absolute_dispatch_plan_v2(
            start_next_tick=25,
            target_next_tick=42,
            preferred_temporal_k=preferred_k,
            admitted_temporal_k=[1, 2, 4],
            boundary_next_ticks=[32, 42],
        )
        return workhorse_indexed_continuation_contract_v2(
            science_continuation_contract_sha256="a" * 64,
            repack_binding=repack,
            dispatch_plan=dispatch,
            checkpoint_ticks=[32, 42],
        )

    first = build([1454, 1451, 1453, 1452], 4)
    second = build([1451, 1452, 1453, 1454], 2)

    assert validate_workhorse_indexed_continuation_contract_v2(first) == first
    assert (
        first["science_continuation_identity"]
        == second["science_continuation_identity"]
    )
    assert (
        first["physical_continuation_plan"]
        != second["physical_continuation_plan"]
    )
    assert first["continuation_contract_sha256"] != second[
        "continuation_contract_sha256"
    ]


def test_continuation_rejects_checkpoint_schedule_that_differs_from_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, manifest_path, state_path = _physical_checkpoint(
        tmp_path / "physical",
        world_uids=[1461],
        rng_keys=[221],
        temporal_k=1,
    )
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (deepcopy(metadata), state_path),
    )
    indexed = write_workhorse_indexed_checkpoint_set_v1(
        manifest_path, tmp_path / "indexed"
    )
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    target = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=1,
        temporal_k=1,
        total_ticks=5,
        start_next_tick=25,
    )
    repack = workhorse_repack_binding_v2(
        [indexed["manifest_path"]],
        ordered_world_uids=[1461],
        target_contract=target,
    )
    dispatch = compile_workhorse_absolute_dispatch_plan_v2(
        start_next_tick=25,
        target_next_tick=30,
        preferred_temporal_k=1,
        admitted_temporal_k=[1],
        boundary_next_ticks=[27, 30],
    )

    with pytest.raises(
        WorkhorseIndexedContinuationError,
        match="frontiers disagree",
    ):
        workhorse_indexed_continuation_contract_v2(
            science_continuation_contract_sha256="b" * 64,
            repack_binding=repack,
            dispatch_plan=dispatch,
            checkpoint_ticks=[30],
        )


def test_repack_binding_distinguishes_subset_fork_and_rejects_unused_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, first_manifest, first_state = _physical_checkpoint(
        tmp_path / "physical-a",
        world_uids=[1501, 1502],
        rng_keys=[121, 122],
        temporal_k=2,
    )
    second, second_manifest, second_state = _physical_checkpoint(
        tmp_path / "physical-b",
        world_uids=[1503],
        rng_keys=[123],
        temporal_k=1,
    )
    by_path = {
        first_manifest.resolve(): (first, first_state),
        second_manifest.resolve(): (second, second_state),
    }
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (
            deepcopy(by_path[Path(path).resolve()][0]),
            by_path[Path(path).resolve()][1],
        ),
    )
    indexed_a = write_workhorse_indexed_checkpoint_set_v1(
        first_manifest, tmp_path / "indexed-a"
    )
    indexed_b = write_workhorse_indexed_checkpoint_set_v1(
        second_manifest, tmp_path / "indexed-b"
    )
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    subset_target = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=1,
        temporal_k=1,
        total_ticks=5,
        start_next_tick=25,
    )

    subset = workhorse_repack_binding_v2(
        [indexed_a["manifest_path"]],
        ordered_world_uids=[1502],
        target_contract=subset_target,
    )
    assert subset["continuation_scope"] == "EXPERIMENTAL_SUBSET_FORK"
    assert validate_workhorse_repack_binding_v2(subset) == subset

    with pytest.raises(
        WorkhorseIndexedCheckpointError,
        match="every repack source manifest",
    ):
        workhorse_repack_binding_v2(
            [indexed_a["manifest_path"], indexed_b["manifest_path"]],
            ordered_world_uids=[1502],
            target_contract=subset_target,
        )


def test_repack_binding_rejects_duplicate_missing_and_wrong_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, first_manifest, first_state = _physical_checkpoint(
        tmp_path / "physical-a",
        world_uids=[1601, 1602],
        rng_keys=[131, 132],
        temporal_k=2,
    )
    duplicate, duplicate_manifest, duplicate_state = _physical_checkpoint(
        tmp_path / "physical-b",
        world_uids=[1602, 1603],
        rng_keys=[132, 133],
        temporal_k=1,
    )
    by_path = {
        first_manifest.resolve(): (first, first_state),
        duplicate_manifest.resolve(): (duplicate, duplicate_state),
    }
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (
            deepcopy(by_path[Path(path).resolve()][0]),
            by_path[Path(path).resolve()][1],
        ),
    )
    indexed_a = write_workhorse_indexed_checkpoint_set_v1(
        first_manifest, tmp_path / "indexed-a"
    )
    indexed_b = write_workhorse_indexed_checkpoint_set_v1(
        duplicate_manifest, tmp_path / "indexed-b"
    )
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    target = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=2,
        temporal_k=1,
        total_ticks=5,
        start_next_tick=25,
    )

    with pytest.raises(WorkhorseIndexedCheckpointError, match="duplicate a world UID"):
        workhorse_repack_binding_v2(
            [indexed_a["manifest_path"], indexed_b["manifest_path"]],
            ordered_world_uids=[1601, 1603],
            target_contract=target,
        )
    with pytest.raises(WorkhorseIndexedCheckpointError, match="absent world UIDs"):
        workhorse_repack_binding_v2(
            [indexed_a["manifest_path"]],
            ordered_world_uids=[1601, 9999],
            target_contract=target,
        )

    wrong_tick = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=2,
        temporal_k=1,
        total_ticks=5,
        start_next_tick=24,
    )
    with pytest.raises(WorkhorseIndexedCheckpointError, match="target differs"):
        workhorse_repack_binding_v2(
            [indexed_a["manifest_path"]],
            ordered_world_uids=[1601, 1602],
            target_contract=wrong_tick,
        )


def test_repack_restore_streams_rows_into_a_fresh_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    first, first_manifest, first_state = _physical_checkpoint(
        tmp_path / "physical-a",
        world_uids=[1701, 1702],
        rng_keys=[141, 142],
        temporal_k=4,
    )
    second, second_manifest, second_state = _physical_checkpoint(
        tmp_path / "physical-b",
        world_uids=[1703, 1704],
        rng_keys=[143, 144],
        temporal_k=2,
    )
    by_path = {
        first_manifest.resolve(): (first, first_state),
        second_manifest.resolve(): (second, second_state),
    }
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (
            deepcopy(by_path[Path(path).resolve()][0]),
            by_path[Path(path).resolve()][1],
        ),
    )
    indexed_a = write_workhorse_indexed_checkpoint_set_v1(
        first_manifest, tmp_path / "indexed-a"
    )
    indexed_b = write_workhorse_indexed_checkpoint_set_v1(
        second_manifest, tmp_path / "indexed-b"
    )
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    target = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=4,
        temporal_k=1,
        total_ticks=7,
        start_next_tick=25,
    )
    binding = workhorse_repack_binding_v2(
        [indexed_a["manifest_path"], indexed_b["manifest_path"]],
        ordered_world_uids=[1704, 1701, 1703, 1702],
        target_contract=target,
    )
    layout = workhorse_layout_v2(shape)

    class RestoreModule:
        calls = 0

        def restore_derived(self, state: object, scratch: object) -> None:
            self.calls += 1
            scratch.fill_(9)

    module = RestoreModule()
    loaded = SimpleNamespace(
        shape=shape,
        backend=workhorse_backend_v2(BACKEND_FUSED_CTA_TEMPORAL_V2),
        layout=layout,
        module=module,
    )
    owner = probe._OwnedBuffers(
        loaded=loaded,
        state=torch.zeros((4, layout.state_bytes_per_world), dtype=torch.uint8),
        scratch=torch.zeros((4, layout.scratch_bytes_per_world), dtype=torch.uint8),
        events=torch.ones((4, 2, 48), dtype=torch.uint8),
        counts=torch.ones((4,), dtype=torch.int64),
        temporal_k=1,
        world_uids=(),
        rng_keys=(),
        committed_next_tick=0,
    )

    restored = probe._restore_owner_from_repack_binding_v2(
        torch,
        loaded,
        owner,
        repack_binding=binding,
    )

    assert restored == binding
    assert owner.world_uids == (1704, 1701, 1703, 1702)
    assert owner.rng_keys == (144, 141, 143, 142)
    assert owner.committed_next_tick == 25
    assert owner.poisoned_status == ()
    assert module.calls == 1
    assert torch.count_nonzero(owner.events).item() == 0
    assert torch.count_nonzero(owner.counts).item() == 0
    assert torch.all(owner.scratch == 9)
    assert [
        struct.unpack("<Q", owner.state[slot, :8].numpy().tobytes())[0]
        for slot in range(4)
    ] == [1704, 1701, 1703, 1702]


def test_repack_restore_poisoned_owner_cannot_hide_target_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    metadata, manifest_path, state_path = _physical_checkpoint(
        tmp_path / "physical",
        world_uids=[1801],
        rng_keys=[151],
        temporal_k=1,
    )
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (deepcopy(metadata), state_path),
    )
    indexed = write_workhorse_indexed_checkpoint_set_v1(
        manifest_path, tmp_path / "indexed"
    )
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    target = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=1,
        temporal_k=1,
        total_ticks=7,
        start_next_tick=25,
    )
    binding = workhorse_repack_binding_v2(
        [indexed["manifest_path"]],
        ordered_world_uids=[1801],
        target_contract=target,
    )
    layout = workhorse_layout_v2(shape)
    loaded = SimpleNamespace(
        shape=shape,
        backend=workhorse_backend_v2(BACKEND_FUSED_CTA_TEMPORAL_V2),
        layout=layout,
        module=SimpleNamespace(restore_derived=lambda *_args: None),
    )
    owner = probe._OwnedBuffers(
        loaded=loaded,
        state=torch.zeros((1, layout.state_bytes_per_world), dtype=torch.uint8),
        scratch=torch.zeros((1, layout.scratch_bytes_per_world), dtype=torch.uint8),
        events=torch.zeros((1, 2, 48), dtype=torch.uint8),
        counts=torch.zeros((1,), dtype=torch.int64),
        temporal_k=2,
        world_uids=(),
        rng_keys=(),
        committed_next_tick=0,
    )

    with pytest.raises(
        probe.FullEvolutionWorkhorseProbeError,
        match="targets another owner",
    ):
        probe._restore_owner_from_repack_binding_v2(
            torch,
            loaded,
            owner,
            repack_binding=binding,
        )
    assert owner.poisoned_status == (-1,)


def test_indexed_continuation_runner_restores_dispatches_and_captures_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, manifest_path, state_path = _physical_checkpoint(
        tmp_path / "physical",
        world_uids=[1901, 1902],
        rng_keys=[161, 162],
        temporal_k=2,
    )
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda path, **_kwargs: (deepcopy(metadata), state_path),
    )
    indexed = write_workhorse_indexed_checkpoint_set_v1(
        manifest_path, tmp_path / "indexed"
    )
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    target = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=2,
        temporal_k=2,
        total_ticks=5,
        start_next_tick=25,
    )
    repack = workhorse_repack_binding_v2(
        [indexed["manifest_path"]],
        ordered_world_uids=[1902, 1901],
        target_contract=target,
    )
    dispatch = compile_workhorse_absolute_dispatch_plan_v2(
        start_next_tick=25,
        target_next_tick=30,
        preferred_temporal_k=2,
        admitted_temporal_k=[1, 2],
        boundary_next_ticks=[28, 30],
    )
    continuation = workhorse_indexed_continuation_contract_v2(
        science_continuation_contract_sha256="c" * 64,
        repack_binding=repack,
        dispatch_plan=dispatch,
        checkpoint_ticks=[28, 30],
    )
    artifact = {
        "path": str((tmp_path / "artifact.so").resolve()),
        "sha256": "d" * 64,
        "sealed_artifact_identity_sha256": "e" * 64,
    }
    device = {"uuid": "GPU-TEST", "total_memory_bytes": 10_000}
    envelope = {
        "artifact": artifact,
        "device_identity": device,
        "expected_rows": [
            {"B": 2, "temporal_k": value, "planned_admission": {"admitted": True}}
            for value in (1, 2)
        ],
        "owner_components_by_k": {"2": {}},
        "capacity_basis": {"reserve_bytes_requested": 100},
        "receipt_path": str((tmp_path / "envelope.json").resolve()),
        "receipt_file_sha256": "f" * 64,
        "receipt_content_sha256": "1" * 64,
    }
    parent = {
        "contract": target.to_dict(),
        "science_continuation_contract_sha256": "c" * 64,
        "source_checkpoint_identity_sha256": continuation[
            "science_continuation_identity"
        ]["source_world_state_set_sha256"],
        "compiled_temporal_k": [1, 2],
        "artifact_sha256": artifact["sha256"],
        "sealed_artifact_identity_sha256": artifact[
            "sealed_artifact_identity_sha256"
        ],
        "device_uuid": "GPU-TEST",
    }

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True

        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, _other: object) -> float:
            return 2.0

    fake_cuda = SimpleNamespace(
        current_device=lambda: 0,
        mem_get_info=lambda _device: (9_000, 10_000),
        Event=FakeEvent,
        empty_cache=lambda: None,
        synchronize=lambda: None,
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda)
    owner = SimpleNamespace(
        state=object(),
        world_uids=(),
        committed_next_tick=0,
        attempt_authority_binding=None,
        pending=None,
    )
    loaded = SimpleNamespace(
        module=SimpleNamespace(runtime_inspect=lambda: {"ready": True})
    )
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_canonical_parent_authority_binding_v2",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_canonical_terminal_binding_v2",
        lambda value, **_kwargs: dict(value),
    )
    reopened_event_receipts: list[dict[str, object]] = []

    def reopen_event_receipts(
        values: object,
    ) -> list[dict[str, object]]:
        rows = [dict(value) for value in values]
        reopened_event_receipts.extend(rows)
        return rows

    monkeypatch.setattr(
        canonical_rows,
        "_reopen_row_event_sink_receipts_v2",
        reopen_event_receipts,
    )
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_device_envelope_calibration_receipt_v2",
        lambda _path: deepcopy(envelope),
    )
    monkeypatch.setattr(
        canonical_rows,
        "workhorse_artifact_binding_v2",
        lambda _path: deepcopy(artifact),
    )
    monkeypatch.setattr(
        workhorse_authority,
        "validate_workhorse_semantic_gate_receipt_v2",
        lambda *_args, **_kwargs: {"semantic": "exact"},
    )
    monkeypatch.setattr(
        workhorse_authority,
        "validate_authoritative_micro_denominator_v2",
        lambda *_args, **_kwargs: {"micro": "exact"},
    )
    monkeypatch.setattr(
        canonical_rows,
        "_owner_components",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        canonical_rows,
        "_envelope_row",
        lambda *_args, **_kwargs: {"planned_admission": {"admitted": True}},
    )
    monkeypatch.setattr(
        workhorse_layout,
        "estimate_workhorse_resources_v2",
        lambda *_args, **_kwargs: SimpleNamespace(to_dict=lambda: {"bytes": 100}),
    )
    monkeypatch.setattr(
        workhorse_layout,
        "admit_workhorse_vram_v2",
        lambda *_args, **_kwargs: SimpleNamespace(to_dict=lambda: {"admitted": True}),
    )
    monkeypatch.setattr(probe, "_torch", lambda: fake_torch)
    monkeypatch.setattr(probe, "_load", lambda *_args: loaded)
    monkeypatch.setattr(probe, "_allocate_owner", lambda *_args: owner)

    def restore(*_args: object, **_kwargs: object) -> dict[str, object]:
        owner.world_uids = (1902, 1901)
        owner.committed_next_tick = 25
        return repack

    monkeypatch.setattr(probe, "_restore_owner_from_repack_binding_v2", restore)
    monkeypatch.setattr(probe, "_prepare_graph_if_needed", lambda *_args: None)
    monkeypatch.setattr(probe, "_release_graph", lambda *_args: None)

    def prepare(_loaded: object, _owner: object, temporal_k: int, _mask: int) -> None:
        owner.pending = {"temporal_k": temporal_k}

    def acknowledge(*_args: object, **_kwargs: object) -> dict[str, object]:
        owner.committed_next_tick += owner.pending["temporal_k"]
        owner.pending = None
        sample = owner.committed_next_tick
        return {
            "receipt_path": str((tmp_path / f"event-{sample}.json").resolve()),
            "receipt_file_sha256": f"{sample:064x}",
            "receipt_body_sha256": f"{sample + 1:064x}",
            "dispatch_content_sha256": f"{sample + 2:064x}",
        }

    monkeypatch.setattr(probe, "_prepare_dispatch_v2", prepare)
    monkeypatch.setattr(probe, "_advance", lambda *_args: None)
    monkeypatch.setattr(probe, "_acknowledge_events", acknowledge)
    monkeypatch.setattr(
        workhorse_benchmark,
        "cuda_device_identity_v2",
        lambda *_args: deepcopy(device),
    )
    captured: list[int] = []

    result = continuation_module.run_workhorse_indexed_continuation_v2(
        artifact["path"],
        envelope_receipt_path=envelope["receipt_path"],
        canonical_parent_authority_binding=parent,
        authoritative_micro_receipt_path=tmp_path / "micro.json",
        semantic_gate_receipt_path=tmp_path / "semantic.json",
        continuation_contract=continuation,
        event_sink_root=tmp_path / "events",
        execution_result_path=tmp_path / "continuation-result.json",
        checkpoint_boundary_handler=lambda **kwargs: captured.append(
            kwargs["checkpoint_at_tick"]
        ),
    )

    assert result["outcome"] == "CONTINUED"
    assert result["restored_world_uids"] == [1902, 1901]
    assert result["checkpoint_capture_ticks"] == [28, 30]
    assert captured == [28, 30]
    assert len(reopened_event_receipts) == 3
    assert [row["temporal_k"] for row in result["dispatch_measurements"]] == [
        2,
        1,
        2,
    ]
    assert result["aggregate_world_ticks_per_second"] == pytest.approx(10_000 / 6)

    publications: list[dict[str, object]] = []
    for tick in (28, 30):
        identity = deepcopy(indexed["world_state_set_identity"])
        identity["checkpoint_at_tick"] = tick
        identity_body = {
            key: value
            for key, value in identity.items()
            if key != "world_state_set_sha256"
        }
        identity["world_state_set_sha256"] = canonical_sha256(identity_body)
        publication_body = {
            "schema_version": WORKHORSE_INDEXED_CHECKPOINT_PUBLICATION_SCHEMA_V2,
            "checkpoint_at_tick": tick,
            "canonical_state_bytes": metadata["canonical_state"]["bytes"],
            "canonical_state_sha256": f"{tick:064x}",
            "checkpoint_bundle_body_sha256": f"{tick + 1:064x}",
            "indexed_manifest_file_sha256": f"{tick + 2:064x}",
            "indexed_manifest_body_sha256": f"{tick + 3:064x}",
            "world_state_set_identity": identity,
            "checkpoint_occurrence_projection_content_sha256": f"{tick + 4:064x}",
            "absolute_tick_bundle_digest": f"{tick + 5:064x}",
            "occurrence_digest": f"{tick + 6:064x}",
        }
        publications.append(
            {
                **publication_body,
                "publication_seal_sha256": canonical_sha256(publication_body),
            }
        )
    terminal = {
        "attempt_terminal_status": "COMPLETED",
        "final_frontiers": {
            field: 30
            for field in (
                "last_dispatched_next_tick",
                "last_computed_next_tick",
                "durable_checkpoint_next_tick",
                "published_artifact_next_tick",
            )
        },
    }
    terminal_receipt = (
        continuation_module.seal_workhorse_terminalized_indexed_continuation_receipt_v2(
            tmp_path / "terminal-continuation.json",
            execution_result_path=result["receipt_path"],
            canonical_terminal_binding=terminal,
            indexed_checkpoint_publications=publications,
        )
    )
    state_path.chmod(0o600)
    state_path.unlink()
    reopened_terminal = (
        continuation_module.validate_workhorse_terminalized_indexed_continuation_receipt_v2(
            terminal_receipt["receipt_path"]
        )
    )
    assert reopened_terminal["indexed_checkpoint_publications"] == publications
