from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import threading
import time

import pytest
import torch

from emergence_contracts.science_compact_capture import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
)
from engine.native_runtime.cuda_v2_compact_science_capture_v1 import (
    write_compact_science_snapshot_v1,
)
from engine.native_runtime.cuda_snapshot_plan_v1 import (
    CudaSnapshotPlanError,
    EielSnapshotCapsuleError,
    compile_snapshot_plan_v1 as compile_semantic_snapshot_plan_v1,
    snapshot_capsule_section_bytes_v1 as semantic_section_bytes_v1,
    validate_snapshot_capsule_v1 as semantic_validate_capsule_v1,
)
from engine.native_runtime.cuda_snapshot_v1 import (
    NativeSnapshotLoadError,
    compile_snapshot_plan_v1,
    convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1,
    extract_compact_snapshot_capsule_section_v1,
    extract_snapshot_capsule_section_v1,
    inspect_snapshot_capsule_v1,
    inspect_compact_snapshot_capsule_v1,
    load_prebuilt_snapshot_extension_v1,
    normalize_snapshot_plan_v1,
    write_owned_compact_snapshot_to_v1_gzip_stage_file_v1,
    write_owned_compact_snapshot_capsule_stage_file_v1,
)
from engine.native_runtime.cuda_snapshot_v1.default_binding import (
    default_snapshot_extension_path_v1,
    resolve_snapshot_extension_path_v1,
    validate_resolved_snapshot_extension_v1,
)
from engine.native_runtime.cuda_snapshot_v1.build import (
    build_native_snapshot_extension_v1,
)


CONTROL_SHA = "a" * 64
EXECUTION_PUBLICATION_CONTEXT_SHA = "b" * 64


def _cells() -> dict[str, dict[str, object]]:
    return {
        "cell-two": {
            "science_logical_world_id": 202,
            "science_ownership": {"study": "study-beta", "arm": "two"},
            "science_world_sha256": {"algorithm": "sha256", "hex": "2" * 64},
            "science_acquisition_sha256": {
                "algorithm": "sha256",
                "hex": "3" * 64,
            },
            "science_acquisition": {"snapshot_ticks": [5, 9]},
        },
        "cell-one": {
            "science_logical_world_id": 101,
            "science_ownership": {"arm": "one", "study": "study-alpha"},
            "science_world_sha256": {"algorithm": "sha256", "hex": "1" * 64},
            "science_acquisition_sha256": {
                "algorithm": "sha256",
                "hex": "2" * 64,
            },
            "science_acquisition": {"snapshot_ticks": [0, 5]},
        },
    }


def _cells_for_count(world_count: int) -> tuple[tuple[int, ...], dict[str, dict[str, object]]]:
    world_uids = tuple(1000 + index for index in range(world_count))
    cells: dict[str, dict[str, object]] = {}
    for index, uid in enumerate(world_uids):
        cells[f"cell-{index}"] = {
            "science_logical_world_id": uid,
            "science_ownership": {"ordinal": index},
            "science_world_sha256": {
                "algorithm": "sha256",
                "hex": f"{index + 1:064x}",
            },
            "science_acquisition_sha256": {
                "algorithm": "sha256",
                "hex": f"{index + 100:064x}",
            },
            "science_acquisition": {"snapshot_ticks": [5]},
        }
    return world_uids, cells


@pytest.fixture(scope="session")
def native_snapshot_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("eiel-native-snapshot-extension") / "artifact"
    return build_native_snapshot_extension_v1(output)


@pytest.fixture(scope="session")
def semantic_plan():
    return compile_semantic_snapshot_plan_v1(
        world_uids=(101, 202),
        world_cells=_cells(),
        native_world_control_set_sha256=CONTROL_SHA,
        execution_publication_context_sha256=(
            EXECUTION_PUBLICATION_CONTEXT_SHA
        ),
        catalog_absolute_tick=0,
    )


@pytest.fixture(scope="session")
def native_plan(native_snapshot_artifact: Path, semantic_plan):
    return compile_snapshot_plan_v1(
        semantic_plan.canonical_plan_bytes,
        extension_path=native_snapshot_artifact,
    )


@pytest.fixture(scope="session")
def host_inputs(semantic_plan):
    mapping = semantic_plan.to_mapping()
    state_shape = tuple(mapping["state_input"]["shape"])
    metrics_shape = tuple(mapping["metrics_input"]["shape"])
    raw_state = bytearray(state_shape[0] * state_shape[1])
    fields = {row["name"]: row for row in mapping["layout"]["fields"]}
    stride = state_shape[1]
    header_offset = int(fields["scalar_header_counters"]["offset_bytes"])
    assert header_offset == 0
    for world_index, world in enumerate(mapping["worlds"]):
        base = world_index * stride + header_offset
        struct.pack_into("<Q", raw_state, base, int(world["world_uid"]))
        struct.pack_into("<Q", raw_state, base + 8, 5)

    def put_agent(world: int, slot: int, agent_id: int, *, alive: bool) -> None:
        base = world * stride
        struct.pack_into(
            "<Q",
            raw_state,
            base + fields["agent_id"]["offset_bytes"] + slot * 8,
            agent_id,
        )
        if alive:
            word_offset = (
                base
                + fields["alive_words"]["offset_bytes"]
                + (slot // 32) * 4
            )
            word = struct.unpack_from("<I", raw_state, word_offset)[0]
            struct.pack_into("<I", raw_state, word_offset, word | (1 << (slot % 32)))

    put_agent(0, 0, 10, alive=True)
    put_agent(0, 1, 20, alive=True)
    put_agent(0, 2, 30, alive=False)
    put_agent(0, 3, 25, alive=True)
    put_agent(1, 0, 40, alive=True)
    # World zero has two identical active genomes plus one distinct genome.
    raw_state[
        fields["policy_alleles"]["offset_bytes"] + 1
    ] = 1
    # World one is also distinct, though catalog dedup remains per world.
    raw_state[
        stride + fields["policy_alleles"]["offset_bytes"]
    ] = 1
    state = torch.frombuffer(raw_state, dtype=torch.uint8).clone().reshape(state_shape)
    metrics = torch.arange(
        metrics_shape[0] * metrics_shape[1] * metrics_shape[2],
        dtype=torch.int64,
    ).reshape(metrics_shape).contiguous()
    return state, metrics


def _state_at_tick(state: torch.Tensor, tick: int) -> torch.Tensor:
    updated = state.clone()
    # The frozen V1 scalar-header registry starts each row with world_uid and
    # next_absolute_tick, both little-endian u64 values.
    updated[:, 8:16] = torch.tensor(
        list(struct.pack("<Q", tick)), dtype=torch.uint8
    )
    return updated


def _capsule(native_plan, host_inputs, *, tick: int) -> bytes:
    state, metrics = host_inputs
    return bytes(
        native_plan.pack_capsule(
            _state_at_tick(state, tick),
            metrics,
            absolute_tick=tick,
        )
    )


def _rewrite_global_sha(value: bytearray) -> None:
    trailer_offset = struct.unpack_from("<Q", value, 120)[0]
    value[trailer_offset : trailer_offset + 32] = hashlib.sha256(
        value[:trailer_offset]
    ).digest()


def _rewrite_section_sha(value: bytearray, descriptor_offset: int) -> None:
    payload_offset = struct.unpack_from("<Q", value, descriptor_offset + 48)[0]
    payload_length = struct.unpack_from("<Q", value, descriptor_offset + 56)[0]
    value[descriptor_offset + 136 : descriptor_offset + 168] = hashlib.sha256(
        value[payload_offset : payload_offset + payload_length]
    ).digest()


def _unpack_column(
    capsule: bytes,
    native_plan,
    name: str,
    code: str,
) -> tuple[int, ...]:
    payload = extract_compact_snapshot_capsule_section_v1(
        capsule, name, expected_plan=native_plan
    )
    width = struct.calcsize(code)
    assert len(payload) % width == 0
    return tuple(
        value[0] for value in struct.iter_unpack("<" + code, payload)
    )


def _digest_column(capsule: bytes, native_plan, name: str) -> tuple[str, ...]:
    payload = extract_compact_snapshot_capsule_section_v1(
        capsule, name, expected_plan=native_plan
    )
    assert len(payload) % 32 == 0
    return tuple(
        payload[offset : offset + 32].hex()
        for offset in range(0, len(payload), 32)
    )


def _canonical_identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def _reconstruct_compact_rows(
    capsule: bytes,
    native_plan,
    semantic_plan,
) -> list[dict[str, object]]:
    """Offline parity converter; publication remains native descriptor-only."""

    inspection = inspect_compact_snapshot_capsule_v1(
        capsule, expected_plan=native_plan
    )
    embedded_plan = extract_compact_snapshot_capsule_section_v1(
        capsule, "canonical_plan_json", expected_plan=native_plan
    )
    assert embedded_plan == semantic_plan.canonical_plan_bytes
    mapping = json.loads(embedded_plan)
    tick = int(inspection["absolute_tick"])
    selected = _unpack_column(
        capsule, native_plan, "selected_world_indices", "Q"
    )
    offsets = _unpack_column(capsule, native_plan, "world_offsets", "Q")
    world_uids = _unpack_column(capsule, native_plan, "world_uid", "Q")
    canonical_digests = _digest_column(
        capsule, native_plan, "canonical_state_sha256"
    )
    resource_digests = _digest_column(
        capsule, native_plan, "resource_state_sha256"
    )
    headers = _unpack_column(capsule, native_plan, "world_header_counts", "Q")
    resources = _unpack_column(capsule, native_plan, "resource_summary", "Q")
    agent_ids = _unpack_column(capsule, native_plan, "agent_id", "Q")
    positions = _unpack_column(capsule, native_plan, "position", "H")
    energies = _unpack_column(capsule, native_plan, "energy", "h")
    healths = _unpack_column(capsule, native_plan, "health", "h")
    ages = _unpack_column(capsule, native_plan, "age_ticks", "H")
    births = _unpack_column(capsule, native_plan, "absolute_birth_tick", "Q")
    parent_as = _unpack_column(capsule, native_plan, "parent_a_id", "Q")
    parent_bs = _unpack_column(capsule, native_plan, "parent_b_id", "Q")
    genome_digests = _digest_column(capsule, native_plan, "genome_state_sha256")
    policy_digests = _digest_column(capsule, native_plan, "policy_sha256")
    metrics = _unpack_column(capsule, native_plan, "lifetime_metrics", "q")
    metric_fields = tuple(mapping["metric_fields"])
    resource_count = int(mapping["layout"]["resource_cell_count"])
    rows: list[dict[str, object]] = []
    for selection, world_index in enumerate(selected):
        world = mapping["worlds"][world_index]
        world_uid = world_uids[selection]
        header = headers[selection * 6 : selection * 6 + 6]
        resource = resources[selection * 4 : selection * 4 + 4]
        world_state_id = _canonical_identity(
            {
                "world_uid": world_uid,
                "tick": tick,
                "canonical_state_sha256": canonical_digests[selection],
            }
        )
        rows.append(
            {
                "absolute_tick": tick,
                "cell_id": json.loads(world["cell_id_json"]),
                "field_set_version": "cuda_full_evolution_v2_state_projection_v1",
                "qualified_field_values": {
                    "absolute_tick": tick,
                    "active_count": header[1],
                    "birth_count": header[2],
                    "death_count": header[3],
                    "mutation_count": header[4],
                    "next_absolute_tick": header[0],
                    "population_count": header[1],
                    "recombination_count": header[5],
                    "resource_max": resource[2],
                    "resource_mean": resource[0] / resource_count,
                    "resource_min": resource[1],
                    "resource_nonzero_count": resource[3],
                    "resource_state_sha256": {
                        "algorithm": "sha256",
                        "hex": resource_digests[selection],
                    },
                    "resource_total": resource[0],
                    "world_uid": world_uid,
                },
                "row_kind": "world_state",
                "snapshot_reason": "native_cuda_v2_science_capture_boundary",
                "world_state_id": world_state_id,
                "world_uid": world_uid,
            }
        )
        assert canonical_digests[selection]
        for row_index in range(offsets[selection], offsets[selection + 1]):
            metric_start = row_index * len(metric_fields)
            genome_state_id = "sha256:" + genome_digests[row_index]
            lineage_id = _canonical_identity(
                {
                    "world_uid": world_uid,
                    "agent_id": agent_ids[row_index],
                    "parent_a": parent_as[row_index],
                    "parent_b": parent_bs[row_index],
                }
            )
            founder_id = _canonical_identity(
                {
                    "world_uid": world_uid,
                    "root_parent_a": (
                        parent_as[row_index] or agent_ids[row_index]
                    ),
                    "root_parent_b": parent_bs[row_index],
                }
            )
            agent_state_id = _canonical_identity(
                {
                    "world_uid": world_uid,
                    "tick": tick,
                    "agent_id": agent_ids[row_index],
                    "alive": True,
                    "position": positions[row_index],
                    "energy": energies[row_index],
                    "health": healths[row_index],
                    "age_ticks": ages[row_index],
                    "genome_state_id": genome_state_id,
                }
            )
            rows.append(
                {
                    "absolute_birth_tick": births[row_index],
                    "absolute_tick": tick,
                    "age_ticks": ages[row_index],
                    "agent_id": agent_ids[row_index],
                    "agent_state_id": agent_state_id,
                    "alive": True,
                    "cell_id": json.loads(world["cell_id_json"]),
                    "energy": energies[row_index],
                    "founder_id": founder_id,
                    "genome_state_id": genome_state_id,
                    "health": healths[row_index],
                    "hereditary_package_state_ids": [],
                    "injury": 0,
                    "lifetime_metrics": dict(
                        zip(
                            metric_fields,
                            metrics[metric_start : metric_start + len(metric_fields)],
                            strict=True,
                        )
                    ),
                    "lineage_depth": (
                        0
                        if parent_as[row_index] == 0 and parent_bs[row_index] == 0
                        else 1
                    ),
                    "lineage_id": lineage_id,
                    "linkage_state_id": genome_state_id,
                    "parent_a_id": parent_as[row_index],
                    "parent_b_id": parent_bs[row_index],
                    "policy_id": "sha256:" + policy_digests[row_index],
                    "position": positions[row_index],
                    "row_kind": "active_agent_state",
                    "world_uid": world_uid,
                }
            )
    section_names = {row["name"] for row in inspection["sections"]}
    if "catalog_world_offsets" in section_names:
        catalog_offsets = _unpack_column(
            capsule, native_plan, "catalog_world_offsets", "Q"
        )
        catalog_digests = _digest_column(
            capsule, native_plan, "catalog_genome_sha256"
        )
        policy_loci = int(mapping["layout"]["policy_geometry"]["bytes"]) // int(
            mapping["layout"]["agent_capacity"]
        )
        trait_loci = int(mapping["layout"]["trait_geometry"]["bytes"]) // (
            2 * int(mapping["layout"]["agent_capacity"])
        )
        policy = _unpack_column(capsule, native_plan, "catalog_policy", "b")
        trait = _unpack_column(capsule, native_plan, "catalog_trait", "h")
        # Legacy ordering is world block: move each world's catalog directly
        # after that world's agents rather than appending all worlds at end.
        rebuilt: list[dict[str, object]] = []
        cursor = 0
        for selection, world_index in enumerate(selected):
            world_row_count = 1 + offsets[selection + 1] - offsets[selection]
            rebuilt.extend(rows[cursor : cursor + world_row_count])
            cursor += world_row_count
            world = mapping["worlds"][world_index]
            for catalog_index in range(
                catalog_offsets[selection], catalog_offsets[selection + 1]
            ):
                rebuilt.append(
                    {
                        "absolute_tick": tick,
                        "cell_id": json.loads(world["cell_id_json"]),
                        "genome_state_id": "sha256:" + catalog_digests[catalog_index],
                        "policy": list(
                            policy[
                                catalog_index * policy_loci :
                                (catalog_index + 1) * policy_loci
                            ]
                        ),
                        "row_kind": "genome_catalog",
                        "trait": list(
                            trait[
                                catalog_index * trait_loci :
                                (catalog_index + 1) * trait_loci
                            ]
                        ),
                        "world_uid": world_uids[selection],
                    }
                )
        rows = rebuilt
    return rows


def test_sealed_extension_manifest_and_truthful_backend(
    native_snapshot_artifact: Path,
) -> None:
    loaded = load_prebuilt_snapshot_extension_v1(native_snapshot_artifact)
    assert loaded.manifest["runtime_jit"] is False
    assert loaded.manifest["compiled_from_sealed_snapshot"] is True
    assert loaded.manifest["projection_backend"] == "HOST_CONTIGUOUS_SYNC_V1"
    assert loaded.module.supports_cuda_projection_v1() is False
    capsule_format = loaded.module.capsule_format_manifest_v1()
    assert capsule_format["canonical_plan_embedded"] is False
    assert capsule_format["plan_identity_payload_bytes"] == 128
    assert capsule_format["materialization_profile"] == (
        "DENSE_SELECTED_WORLD_PARITY_ORACLE_V1"
    )
    assert capsule_format["active_row_compaction"] is False
    assert capsule_format["production_compaction_ready"] is False
    assert capsule_format["compact_active_row_compaction"] is True
    assert capsule_format["compact_production_publication_ready"] is False
    assert capsule_format["compact_canonical_plan_embedded"] is True


def test_canonical_snapshot_binding_is_source_bound_and_validated_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EIEL_NATIVE_SNAPSHOT_EXTENSION_V1", raising=False)
    path = default_snapshot_extension_path_v1()
    assert path.is_file()
    assert resolve_snapshot_extension_path_v1() == path
    artifact, manifest_path, manifest = validate_resolved_snapshot_extension_v1()
    assert artifact == path.resolve()
    assert manifest_path == path.with_suffix(path.suffix + ".manifest.json")
    assert manifest["module_name"] == path.stem


def test_loader_rejects_supplied_symlink_and_unbound_module_name(
    native_snapshot_artifact: Path,
    tmp_path: Path,
) -> None:
    symlink = tmp_path / "snapshot-link.so"
    symlink.symlink_to(native_snapshot_artifact)
    with pytest.raises(NativeSnapshotLoadError, match="symlink"):
        load_prebuilt_snapshot_extension_v1(symlink)

    copied = tmp_path / native_snapshot_artifact.name
    copied_manifest = copied.with_suffix(copied.suffix + ".manifest.json")
    shutil.copy2(native_snapshot_artifact, copied)
    shutil.copy2(
        native_snapshot_artifact.with_suffix(
            native_snapshot_artifact.suffix + ".manifest.json"
        ),
        copied_manifest,
    )
    copied_manifest.chmod(0o600)
    manifest = json.loads(copied_manifest.read_text(encoding="utf-8"))
    manifest.pop("manifest_sha256")
    manifest["module_name"] = "eiel_native_snapshot_v1_deadbeefdeadbeef"
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    copied_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(NativeSnapshotLoadError, match="source inventory digest"):
        load_prebuilt_snapshot_extension_v1(copied)


@pytest.mark.parametrize("world_count", [1, 3, 9])
def test_native_plan_world_and_input_shapes_are_dynamic(
    native_snapshot_artifact: Path,
    world_count: int,
) -> None:
    world_uids, cells = _cells_for_count(world_count)
    semantic = compile_semantic_snapshot_plan_v1(
        world_uids=world_uids,
        world_cells=cells,
        native_world_control_set_sha256=CONTROL_SHA,
        execution_publication_context_sha256=(
            EXECUTION_PUBLICATION_CONTEXT_SHA
        ),
        catalog_absolute_tick=None,
    )
    native = compile_snapshot_plan_v1(
        semantic, extension_path=native_snapshot_artifact
    )
    mapping = semantic.to_mapping()
    state_shape = tuple(mapping["state_input"]["shape"])
    raw_state = bytearray(state_shape[0] * state_shape[1])
    for world_index, world_uid in enumerate(world_uids):
        base = world_index * state_shape[1]
        struct.pack_into("<Q", raw_state, base, world_uid)
        struct.pack_into("<Q", raw_state, base + 8, 5)
    state = torch.frombuffer(raw_state, dtype=torch.uint8).clone().reshape(state_shape)
    metrics = torch.zeros(
        tuple(mapping["metrics_input"]["shape"]), dtype=torch.int64
    )
    capsule = bytes(native.pack_capsule(state, metrics, absolute_tick=5))
    inspection = native.inspect_capsule(capsule)
    assert inspection["world_count"] == world_count
    assert inspection["row_count"] == world_count
    assert extract_snapshot_capsule_section_v1(
        capsule, "selected_world_indices", expected_plan=native
    ) == b"".join(struct.pack("<Q", index) for index in range(world_count))


def test_native_plan_binds_declared_state_capture_transport(
    native_snapshot_artifact: Path,
    semantic_plan,
) -> None:
    loaded = load_prebuilt_snapshot_extension_v1(native_snapshot_artifact)
    normalized = normalize_snapshot_plan_v1(semantic_plan.to_mapping())
    normalized["state_capture_transport"]["accepted_modes"][0][
        "world_stride_bytes"
    ] += 1
    with pytest.raises(RuntimeError, match="capture mode geometry differs"):
        loaded.module.compile_snapshot_plan_v1(
            semantic_plan.canonical_plan_bytes,
            normalized,
        )


def test_native_capsule_is_deterministic_bounded_and_semantically_accepted(
    native_plan,
    semantic_plan,
    host_inputs,
) -> None:
    first = _capsule(native_plan, host_inputs, tick=5)
    second = _capsule(native_plan, host_inputs, tick=5)
    assert first == second
    inspection = inspect_snapshot_capsule_v1(first, expected_plan=native_plan)
    assert inspection["absolute_tick"] == 5
    assert inspection["world_count"] == 2
    assert inspection["row_count"] == 6
    assert inspection["section_count"] == len(
        semantic_plan.to_mapping()["sections"]
    ) + 2
    assert inspection["integrity_trailer_sha256"] == hashlib.sha256(
        first[: inspection["trailer_offset"]]
    ).hexdigest()
    assert inspection["capsule_sha256"] == hashlib.sha256(first).hexdigest()
    first_section = inspection["sections"][0]
    assert first_section["name"] == "plan_identity"
    assert first_section["kind"] == 6
    assert first_section["dtype"] == "u8"
    assert first_section["shape"] == [128]
    assert first_section["payload_length"] == 128
    assert semantic_plan.canonical_plan_bytes not in first
    assert len(semantic_plan.canonical_plan_bytes) > 128
    semantic_view = semantic_validate_capsule_v1(
        first, expected_plan=semantic_plan
    )
    assert semantic_view.absolute_tick == 5
    assert semantic_section_bytes_v1(
        first, "selected_world_indices", expected_plan=semantic_plan
    ) == struct.pack("<QQ", 0, 1)


def test_native_world_selection_and_world_major_gather_are_exact(
    native_plan,
    semantic_plan,
    host_inputs,
) -> None:
    state, metrics = host_inputs
    capsule = _capsule(native_plan, host_inputs, tick=9)
    inspection = native_plan.inspect_capsule(capsule)
    assert inspection["world_count"] == 1
    assert inspection["row_count"] == 2
    assert extract_snapshot_capsule_section_v1(
        capsule, "selected_world_indices", expected_plan=native_plan
    ) == struct.pack("<Q", 1)
    assert extract_snapshot_capsule_section_v1(
        capsule, "scalar_header_counters", expected_plan=native_plan
    ) == _state_at_tick(state, 9)[1, : 24 * 8].numpy().tobytes()
    assert extract_snapshot_capsule_section_v1(
        capsule, "lifetime_metrics", expected_plan=native_plan
    ) == metrics[1].numpy().tobytes()
    assert semantic_section_bytes_v1(
        capsule,
        "selected_world_indices",
        expected_plan=semantic_plan,
    ) == struct.pack("<Q", 1)


def test_native_row_count_is_derived_from_alive_and_catalog_truth(
    native_plan,
    semantic_plan,
    host_inputs,
    tmp_path: Path,
) -> None:
    catalog_capsule = _capsule(native_plan, host_inputs, tick=0)
    catalog_inspection = native_plan.inspect_capsule(catalog_capsule)
    # One world row, three active-agent rows, and two deduplicated genome rows.
    assert catalog_inspection["row_count"] == 6

    state, metrics = host_inputs
    state_at_catalog_tick = _state_at_tick(state, 0)
    legacy = write_compact_science_snapshot_v1(
        tmp_path / "legacy-v1",
        state=state_at_catalog_tick,
        active_metrics_projection={
            "next_absolute_tick": 0,
            "world_uids": (101, 202),
            "metric_fields": COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
            "tensor": metrics,
        },
        world_cells=_cells(),
        absolute_tick=0,
        native_world_control_set_sha256=CONTROL_SHA,
        include_genome_catalog=True,
    )
    assert catalog_inspection["row_count"] == legacy["row_count"]

    zero_state = state.clone()
    mapping = semantic_plan.to_mapping()
    fields = {row["name"]: row for row in mapping["layout"]["fields"]}
    for name in ("alive_words", "agent_id"):
        start = int(fields[name]["offset_bytes"])
        stop = start + int(fields[name]["bytes"])
        zero_state[:, start:stop] = 0
    zero_capsule = bytes(
        native_plan.pack_capsule(zero_state, metrics, absolute_tick=5)
    )
    assert native_plan.inspect_capsule(zero_capsule)["row_count"] == 2

    mapping = native_plan.plan_manifest()
    assert mapping["materialization_profile"] == (
        "DENSE_SELECTED_WORLD_PARITY_ORACLE_V1"
    )


def test_native_row_count_handles_all_alive_capacity(
    native_plan,
    semantic_plan,
    host_inputs,
) -> None:
    state, metrics = host_inputs
    mapping = semantic_plan.to_mapping()
    fields = {row["name"]: row for row in mapping["layout"]["fields"]}
    stride = int(mapping["state_input"]["shape"][1])
    capacity = int(mapping["layout"]["agent_capacity"])
    raw = bytearray(state.numpy().tobytes())
    for world in range(2):
        base = world * stride
        for word in range(capacity // 32):
            struct.pack_into(
                "<I",
                raw,
                base + int(fields["alive_words"]["offset_bytes"]) + word * 4,
                (1 << 32) - 1,
            )
        for slot in range(capacity):
            struct.pack_into(
                "<Q",
                raw,
                base + int(fields["agent_id"]["offset_bytes"]) + slot * 8,
                world * capacity + slot + 1,
            )
    all_alive = torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(state.shape)
    capsule = bytes(
        native_plan.pack_capsule(all_alive, metrics, absolute_tick=5)
    )
    assert native_plan.inspect_capsule(capsule)["row_count"] == 2 + 2 * capacity


def test_native_row_accounting_rejects_duplicate_nonzero_identity(
    native_plan,
    semantic_plan,
    host_inputs,
) -> None:
    state, metrics = host_inputs
    duplicate = state.clone()
    mapping = semantic_plan.to_mapping()
    fields = {row["name"]: row for row in mapping["layout"]["fields"]}
    ids_offset = int(fields["agent_id"]["offset_bytes"])
    duplicate[0, ids_offset + 8 : ids_offset + 16] = torch.tensor(
        list(struct.pack("<Q", 10)), dtype=torch.uint8
    )
    with pytest.raises(RuntimeError, match="duplicate agent identity 10"):
        native_plan.pack_capsule(duplicate, metrics, absolute_tick=5)


def test_native_rejects_swapped_world_state_and_wrong_frontier(
    native_plan,
    host_inputs,
) -> None:
    state, metrics = host_inputs
    with pytest.raises(RuntimeError, match="world identity differs"):
        native_plan.pack_capsule(
            state.flip(0).contiguous(), metrics, absolute_tick=5
        )
    with pytest.raises(RuntimeError, match="next_absolute_tick differs"):
        native_plan.pack_capsule(
            _state_at_tick(state, 4), metrics, absolute_tick=5
        )


def test_native_plan_rejects_wrong_inputs_tick_and_exact_integer_domain(
    native_plan,
    host_inputs,
) -> None:
    state, metrics = host_inputs
    with pytest.raises(RuntimeError, match="dtype differs"):
        native_plan.pack_capsule(
            state.to(torch.int16), metrics, absolute_tick=5
        )
    with pytest.raises(RuntimeError, match="contiguous"):
        native_plan.pack_capsule(
            state[:, ::2], metrics, absolute_tick=5
        )
    with pytest.raises(RuntimeError, match="absent from compiled plan"):
        native_plan.pack_capsule(state, metrics, absolute_tick=7)
    with pytest.raises(RuntimeError, match="must be an integer"):
        native_plan.pack_capsule(state, metrics, absolute_tick=True)


def test_native_capsule_rejects_truncation_global_and_section_corruption(
    native_plan,
    host_inputs,
) -> None:
    capsule = _capsule(native_plan, host_inputs, tick=5)
    with pytest.raises(RuntimeError, match="byte count|truncated"):
        native_plan.inspect_capsule(capsule[:-1])
    global_corrupt = bytearray(capsule)
    global_corrupt[-1] ^= 1
    with pytest.raises(RuntimeError, match="global SHA-256"):
        native_plan.inspect_capsule(bytes(global_corrupt))
    section_corrupt = bytearray(capsule)
    first_data_payload_offset = struct.unpack_from(
        "<Q", section_corrupt, 384 + 2 * 256 + 48
    )[0]
    section_corrupt[first_data_payload_offset] ^= 1
    _rewrite_global_sha(section_corrupt)
    with pytest.raises(RuntimeError, match="section SHA-256"):
        native_plan.inspect_capsule(bytes(section_corrupt))


def test_python_and_native_validators_share_adversarial_abi(
    native_snapshot_artifact: Path,
    native_plan,
    host_inputs,
) -> None:
    capsule = _capsule(native_plan, host_inputs, tick=5)
    mutations: list[tuple[str, bytearray]] = []

    row_count = bytearray(capsule)
    struct.pack_into("<Q", row_count, 56, 1)
    _rewrite_global_sha(row_count)
    mutations.append(("row_count", row_count))

    tick = bytearray(capsule)
    struct.pack_into("<Q", tick, 40, 1 << 63)
    _rewrite_global_sha(tick)
    mutations.append(("signed_tick", tick))

    selection = bytearray(capsule)
    selection_descriptor = 384 + 256
    selected_offset = struct.unpack_from(
        "<Q", selection, selection_descriptor + 48
    )[0]
    selection[selected_offset : selected_offset + 16] = struct.pack("<QQ", 1, 0)
    _rewrite_section_sha(selection, selection_descriptor)
    _rewrite_global_sha(selection)
    mutations.append(("selected_order", selection))

    rank = bytearray(capsule)
    first_data_descriptor = 384 + 2 * 256
    struct.pack_into("<H", rank, first_data_descriptor + 6, 0)
    _rewrite_global_sha(rank)
    mutations.append(("rank_zero", rank))

    product_overflow = bytearray(capsule)
    struct.pack_into("<H", product_overflow, first_data_descriptor + 6, 2)
    struct.pack_into("<Q", product_overflow, first_data_descriptor + 72, 2)
    struct.pack_into("<Q", product_overflow, first_data_descriptor + 80, 1 << 63)
    _rewrite_global_sha(product_overflow)
    mutations.append(("shape_product_overflow", product_overflow))

    directory_overflow = bytearray(capsule)
    struct.pack_into("<Q", directory_overflow, 64, 1 << 63)
    _rewrite_global_sha(directory_overflow)
    mutations.append(("directory_overflow", directory_overflow))

    for _label, mutation in mutations:
        with pytest.raises(EielSnapshotCapsuleError):
            semantic_validate_capsule_v1(bytes(mutation))
        with pytest.raises(RuntimeError):
            inspect_snapshot_capsule_v1(
                bytes(mutation), extension_path=native_snapshot_artifact
            )


def test_native_capsule_rejects_schema_reserved_and_expected_plan_mismatch(
    native_snapshot_artifact: Path,
    native_plan,
    semantic_plan,
    host_inputs,
) -> None:
    capsule = _capsule(native_plan, host_inputs, tick=5)
    wrong_version = bytearray(capsule)
    struct.pack_into("<H", wrong_version, 16, 2)
    _rewrite_global_sha(wrong_version)
    with pytest.raises(RuntimeError, match="version differs"):
        inspect_snapshot_capsule_v1(
            bytes(wrong_version), extension_path=native_snapshot_artifact
        )
    reserved = bytearray(capsule)
    reserved[280] = 1
    _rewrite_global_sha(reserved)
    with pytest.raises(RuntimeError, match="reserved header bytes"):
        native_plan.inspect_capsule(bytes(reserved))
    wrong_mapping = semantic_plan.to_mapping()
    wrong_mapping["snapshot_schema_id"] = "different_snapshot_schema_v1"
    wrong_mapping["schema_sha256"] = hashlib.sha256(
        b"different_snapshot_schema_v1"
    ).hexdigest()
    wrong_bytes = json.dumps(
        wrong_mapping,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    with pytest.raises(CudaSnapshotPlanError, match="schema identity"):
        compile_snapshot_plan_v1(
            wrong_bytes, extension_path=native_snapshot_artifact
        )
    wrong_law = semantic_plan.to_mapping()
    wrong_law["selection_laws"]["alive_selection"] = "publish_every_slot"
    with pytest.raises(CudaSnapshotPlanError, match="selection laws"):
        compile_snapshot_plan_v1(
            wrong_law, extension_path=native_snapshot_artifact
        )


def test_noncanonical_plan_bytes_fail_before_native_compile(
    native_snapshot_artifact: Path,
    semantic_plan,
) -> None:
    noncanonical = json.dumps(
        semantic_plan.to_mapping(), indent=2, ensure_ascii=True
    ).encode("ascii")
    with pytest.raises(CudaSnapshotPlanError, match="not exact canonical"):
        compile_snapshot_plan_v1(
            noncanonical, extension_path=native_snapshot_artifact
        )


def test_compact_stage_capsule_exactly_reconstructs_sparse_legacy_rows(
    native_plan,
    semantic_plan,
    host_inputs,
    tmp_path: Path,
) -> None:
    state, metrics = host_inputs
    state_at_tick = _state_at_tick(state, 5)
    canonical_bytes = int(
        semantic_plan.to_mapping()["layout"]["canonical_state_bytes_per_world"]
    )
    stage_path = (tmp_path / "sparse.compact.capsule").absolute()
    receipt = write_owned_compact_snapshot_capsule_stage_file_v1(
        native_plan,
        state=state_at_tick[:, :canonical_bytes].contiguous(),
        metrics=metrics,
        absolute_tick=5,
        stage_path=stage_path,
    )
    capsule = stage_path.read_bytes()
    inspection = inspect_compact_snapshot_capsule_v1(
        capsule, expected_plan=native_plan
    )
    assert receipt["stage_file_state"] == (
        "CONTENT_FSYNCED_CLOSED_STAGE_ONLY_NOT_PUBLISHED"
    )
    assert receipt["stage_durability_scope"] == (
        "FILE_CONTENT_ONLY_DIRECTORY_ENTRY_NOT_FSYNCED_V1"
    )
    assert receipt["stage_parent_directory_fsync_performed"] is False
    assert receipt["stage_path_identity_scheme"] == "POSIX_ST_DEV_ST_INO_V1"
    stage_stat = os.stat(stage_path, follow_symlinks=False)
    assert receipt["stage_file_device"] == stage_stat.st_dev
    assert receipt["stage_file_inode"] == stage_stat.st_ino
    parent_stat = os.stat(stage_path.parent, follow_symlinks=False)
    assert receipt["stage_parent_directory_device"] == parent_stat.st_dev
    assert receipt["stage_parent_directory_inode"] == parent_stat.st_ino
    assert receipt["stage_path_revalidated_after_close"] is True
    assert receipt["state_capture_mode"] == "canonical_checkpoint_prefix_v1"
    assert receipt["state_capture_preferred_mode"] == (
        "canonical_checkpoint_prefix_v1"
    )
    assert receipt["state_capture_mode_is_preferred"] is True
    assert receipt["payload_bytes_returned_to_python"] == 0
    assert receipt["materialization_profile"] == "ACTIVE_COLUMNAR_STAGE_V1"
    assert receipt["capsule_bytes"] == len(capsule) == stage_path.stat().st_size
    assert receipt["capsule_sha256"] == hashlib.sha256(capsule).hexdigest()
    assert receipt["world_count"] == 2
    assert receipt["active_agent_count"] == 4
    assert receipt["catalog_row_count"] == 0
    assert receipt["active_row_payload_bytes_per_agent"] == 276
    assert receipt["catalog_row_payload_bytes"] == 320
    assert receipt["row_count"] == 6 == inspection["row_count"]
    assert receipt["self_describing_reconstruction"] is True
    assert receipt["embedded_canonical_plan_bytes"] == len(
        semantic_plan.canonical_plan_bytes
    )
    assert inspection["canonical_plan_index_plus_one"] == 4
    assert extract_compact_snapshot_capsule_section_v1(
        capsule, "canonical_plan_json", expected_plan=native_plan
    ) == semantic_plan.canonical_plan_bytes
    section_names = {row["name"] for row in inspection["sections"]}
    assert {
        "agent_state_sha256",
        "founder_sha256",
        "lineage_sha256",
        "world_state_sha256",
    }.isdisjoint(section_names)
    metrics_descriptor = next(
        row for row in inspection["sections"] if row["name"] == "lifetime_metrics"
    )
    assert receipt["active_metrics_content_sha256"] == metrics_descriptor[
        "payload_sha256"
    ]
    assert receipt["memory"]["borrowed_state_row_bytes"] == canonical_bytes
    assert receipt["memory"]["horizon_dependent_bytes"] == 0
    assert receipt["memory"]["native_heap_peak_bytes_upper_bound"] >= len(capsule)
    assert _unpack_column(capsule, native_plan, "world_offsets", "Q") == (
        0,
        3,
        4,
    )
    assert _unpack_column(capsule, native_plan, "active_slots", "I") == (
        0,
        1,
        3,
        0,
    )
    assert _unpack_column(capsule, native_plan, "agent_id", "Q") == (
        10,
        20,
        25,
        40,
    )

    legacy_root = tmp_path / "legacy-sparse"
    legacy = write_compact_science_snapshot_v1(
        legacy_root,
        state=state_at_tick,
        active_metrics_projection={
            "next_absolute_tick": 5,
            "world_uids": (101, 202),
            "metric_fields": COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
            "tensor": metrics,
        },
        world_cells=_cells(),
        absolute_tick=5,
        native_world_control_set_sha256=CONTROL_SHA,
        include_genome_catalog=False,
    )
    canonical_jsonl = gzip.decompress(
        (legacy_root / str(legacy["data_file"])).read_bytes()
    )
    legacy_gzip = (legacy_root / str(legacy["data_file"])).read_bytes()
    converted_path = (tmp_path / "sparse.converted.jsonl.gz").absolute()
    converted = (
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            native_plan,
            capsule_stage_path=stage_path,
            gzip_stage_path=converted_path,
        )
    )
    assert converted_path.read_bytes() == legacy_gzip
    assert gzip.decompress(converted_path.read_bytes()) == canonical_jsonl
    assert converted["row_count"] == 6
    assert converted["captured_world_count"] == 2
    assert converted["active_agent_count"] == 4
    assert converted["catalog_row_count"] == 0
    assert converted["genome_catalog_included"] is False
    assert converted["payload_bytes_returned_to_python"] == 0
    assert converted["python_row_materialization"] is False
    assert converted["compressed_file_sha256"] == hashlib.sha256(
        legacy_gzip
    ).hexdigest()
    assert converted["uncompressed_content_sha256"] == hashlib.sha256(
        canonical_jsonl
    ).hexdigest()
    assert converted["output_stage_file_state"] == (
        "CONTENT_FSYNCED_CLOSED_STAGE_ONLY_NOT_PUBLISHED"
    )
    assert converted["output_stage_durability_scope"] == (
        "FILE_CONTENT_ONLY_DIRECTORY_ENTRY_NOT_FSYNCED_V1"
    )
    assert converted["output_parent_directory_fsync_performed"] is False
    assert converted["input_path_revalidated_after_close"] is True
    assert converted["output_path_revalidated_after_close"] is True
    assert converted["memory"]["horizon_dependent_bytes"] == 0
    legacy_rows = [json.loads(line) for line in canonical_jsonl.splitlines()]
    assert _reconstruct_compact_rows(
        capsule, native_plan, semantic_plan
    ) == legacy_rows
    dense = _capsule(native_plan, host_inputs, tick=5)
    assert len(capsule) < len(dense)
    assert receipt["compact_payload_bytes"] < receipt["dense_oracle_payload_bytes"]

    fused_path = (tmp_path / "sparse.fused.jsonl.gz").absolute()
    fused = write_owned_compact_snapshot_to_v1_gzip_stage_file_v1(
        native_plan,
        state=state_at_tick[:, :canonical_bytes].contiguous(),
        metrics=metrics,
        absolute_tick=5,
        stage_path=fused_path,
    )
    assert fused_path.read_bytes() == legacy_gzip
    assert fused["schema_version"] == "eiel_native_snapshot_v1_stage_file_receipt_v1"
    assert fused["implementation"] == (
        "NATIVE_COMPACT_CAPSULE_FUSED_TO_EXACT_CANONICAL_V1_V1"
    )
    assert fused["stage_path"] == str(fused_path)
    assert fused["stage_file_state"] == "FSYNCED_STAGE_ONLY_NOT_PUBLISHED"
    assert fused["stage_file_fsync_completed"] is True
    assert fused["stage_file_close_completed"] is True
    assert fused["atomic_publication_performed"] is False
    assert fused["directory_fsync_performed"] is False
    assert fused["capsule_stage_written"] is False
    assert fused["python_row_materialization"] is False
    assert fused["row_count"] == 6
    assert fused["active_agent_count"] == 4
    assert fused["catalog_row_count"] == 0
    assert fused["compressed_file_sha256"] == hashlib.sha256(
        legacy_gzip
    ).hexdigest()
    assert fused["uncompressed_content_sha256"] == hashlib.sha256(
        canonical_jsonl
    ).hexdigest()
    assert fused["timing"]["native_encode_write_fsync_close_ns"] >= 0

    undeclared_width = canonical_bytes + 1
    rejected_path = (tmp_path / "undeclared-width.compact.capsule").absolute()
    with pytest.raises(RuntimeError, match="plan-declared.*capture transport"):
        write_owned_compact_snapshot_capsule_stage_file_v1(
            native_plan,
            state=state_at_tick[:, :undeclared_width].contiguous(),
            metrics=metrics,
            absolute_tick=5,
            stage_path=rejected_path,
        )
    assert not rejected_path.exists()


def test_compact_catalog_capsule_reconstructs_deduplicated_alleles(
    native_plan,
    semantic_plan,
    host_inputs,
    tmp_path: Path,
) -> None:
    state, metrics = host_inputs
    state_at_tick = _state_at_tick(state, 0)
    stage_path = (tmp_path / "catalog.compact.capsule").absolute()
    receipt = write_owned_compact_snapshot_capsule_stage_file_v1(
        native_plan,
        state=state_at_tick,
        metrics=metrics,
        absolute_tick=0,
        stage_path=stage_path,
    )
    capsule = stage_path.read_bytes()
    assert receipt["state_capture_mode"] == "full_resident_row_v1"
    assert receipt["state_capture_mode_is_preferred"] is False
    assert receipt["world_count"] == 1
    assert receipt["active_agent_count"] == 3
    assert receipt["catalog_row_count"] == 2
    assert receipt["row_count"] == 6
    assert _unpack_column(
        capsule, native_plan, "catalog_world_offsets", "Q"
    ) == (0, 2)
    catalog_digests = _digest_column(
        capsule, native_plan, "catalog_genome_sha256"
    )
    assert catalog_digests == tuple(sorted(set(catalog_digests)))

    legacy_root = tmp_path / "legacy-catalog"
    legacy = write_compact_science_snapshot_v1(
        legacy_root,
        state=state_at_tick,
        active_metrics_projection={
            "next_absolute_tick": 0,
            "world_uids": (101, 202),
            "metric_fields": COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
            "tensor": metrics,
        },
        world_cells=_cells(),
        absolute_tick=0,
        native_world_control_set_sha256=CONTROL_SHA,
        include_genome_catalog=True,
    )
    legacy_gzip = (legacy_root / str(legacy["data_file"])).read_bytes()
    converted_path = (tmp_path / "catalog.converted.jsonl.gz").absolute()
    converted = (
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            native_plan,
            capsule_stage_path=stage_path,
            gzip_stage_path=converted_path,
        )
    )
    assert converted_path.read_bytes() == legacy_gzip
    assert converted["row_count"] == 6
    assert converted["active_agent_count"] == 3
    assert converted["catalog_row_count"] == 2
    assert converted["genome_catalog_included"] is True
    legacy_rows = [
        json.loads(line)
        for line in gzip.decompress(legacy_gzip).splitlines()
    ]
    assert _reconstruct_compact_rows(
        capsule, native_plan, semantic_plan
    ) == legacy_rows


def test_compact_zero_extent_collision_and_corruption_fail_closed(
    native_plan,
    semantic_plan,
    host_inputs,
    tmp_path: Path,
) -> None:
    state, metrics = host_inputs
    swapped_path = (tmp_path / "swapped.compact.capsule").absolute()
    with pytest.raises(RuntimeError, match="world identity differs"):
        write_owned_compact_snapshot_capsule_stage_file_v1(
            native_plan,
            state=_state_at_tick(state, 5).flip(0).contiguous(),
            metrics=metrics,
            absolute_tick=5,
            stage_path=swapped_path,
        )
    assert not swapped_path.exists()
    frontier_path = (tmp_path / "frontier.compact.capsule").absolute()
    with pytest.raises(RuntimeError, match="next_absolute_tick differs"):
        write_owned_compact_snapshot_capsule_stage_file_v1(
            native_plan,
            state=_state_at_tick(state, 4),
            metrics=metrics,
            absolute_tick=5,
            stage_path=frontier_path,
        )
    assert not frontier_path.exists()
    zero = _state_at_tick(state, 5)
    fields = {
        row["name"]: row for row in semantic_plan.to_mapping()["layout"]["fields"]
    }
    for name in ("alive_words", "agent_id"):
        start = int(fields[name]["offset_bytes"])
        stop = start + int(fields[name]["bytes"])
        zero[:, start:stop] = 0
    stage_path = (tmp_path / "zero.compact.capsule").absolute()
    receipt = write_owned_compact_snapshot_capsule_stage_file_v1(
        native_plan,
        state=zero,
        metrics=metrics,
        absolute_tick=5,
        stage_path=stage_path,
    )
    capsule = stage_path.read_bytes()
    inspection = inspect_compact_snapshot_capsule_v1(
        capsule, expected_plan=native_plan
    )
    assert receipt["active_agent_count"] == 0
    assert receipt["row_count"] == 2
    assert next(
        row for row in inspection["sections"] if row["name"] == "agent_id"
    )["shape"] == [0]
    zero_legacy_root = tmp_path / "legacy-zero"
    zero_legacy = write_compact_science_snapshot_v1(
        zero_legacy_root,
        state=zero,
        active_metrics_projection={
            "next_absolute_tick": 5,
            "world_uids": (101, 202),
            "metric_fields": COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
            "tensor": metrics,
        },
        world_cells=_cells(),
        absolute_tick=5,
        native_world_control_set_sha256=CONTROL_SHA,
        include_genome_catalog=False,
    )
    zero_legacy_gzip = (
        zero_legacy_root / str(zero_legacy["data_file"])
    ).read_bytes()
    zero_gzip_path = (tmp_path / "zero.converted.jsonl.gz").absolute()
    zero_converted = (
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            native_plan,
            capsule_stage_path=stage_path,
            gzip_stage_path=zero_gzip_path,
        )
    )
    assert zero_gzip_path.read_bytes() == zero_legacy_gzip
    assert zero_converted["row_count"] == 2
    assert zero_converted["active_agent_count"] == 0
    assert zero_converted["catalog_row_count"] == 0
    assert zero_converted["genome_catalog_included"] is False

    zero_catalog_state = _state_at_tick(zero, 0)
    zero_catalog_capsule_path = (
        tmp_path / "zero-catalog.compact.capsule"
    ).absolute()
    zero_catalog_receipt = write_owned_compact_snapshot_capsule_stage_file_v1(
        native_plan,
        state=zero_catalog_state,
        metrics=metrics,
        absolute_tick=0,
        stage_path=zero_catalog_capsule_path,
    )
    assert zero_catalog_receipt["catalog_row_count"] == 0
    zero_catalog_legacy_root = tmp_path / "legacy-zero-catalog"
    zero_catalog_legacy = write_compact_science_snapshot_v1(
        zero_catalog_legacy_root,
        state=zero_catalog_state,
        active_metrics_projection={
            "next_absolute_tick": 0,
            "world_uids": (101, 202),
            "metric_fields": COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
            "tensor": metrics,
        },
        world_cells=_cells(),
        absolute_tick=0,
        native_world_control_set_sha256=CONTROL_SHA,
        include_genome_catalog=True,
    )
    zero_catalog_legacy_gzip = (
        zero_catalog_legacy_root / str(zero_catalog_legacy["data_file"])
    ).read_bytes()
    zero_catalog_gzip_path = (
        tmp_path / "zero-catalog.converted.jsonl.gz"
    ).absolute()
    zero_catalog_converted = (
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            native_plan,
            capsule_stage_path=zero_catalog_capsule_path,
            gzip_stage_path=zero_catalog_gzip_path,
        )
    )
    assert zero_catalog_gzip_path.read_bytes() == zero_catalog_legacy_gzip
    assert zero_catalog_converted["row_count"] == 1
    assert zero_catalog_converted["catalog_row_count"] == 0
    assert zero_catalog_converted["genome_catalog_included"] is True
    before = stage_path.read_bytes()
    with pytest.raises(RuntimeError, match="exclusive create"):
        write_owned_compact_snapshot_capsule_stage_file_v1(
            native_plan,
            state=zero,
            metrics=metrics,
            absolute_tick=5,
            stage_path=stage_path,
        )
    assert stage_path.read_bytes() == before
    victim = tmp_path / "victim"
    victim.write_bytes(b"preserve-me")
    symlink_stage = tmp_path / "symlink.compact.capsule"
    symlink_stage.symlink_to(victim)
    with pytest.raises(RuntimeError, match="exclusive create"):
        write_owned_compact_snapshot_capsule_stage_file_v1(
            native_plan,
            state=zero,
            metrics=metrics,
            absolute_tick=5,
            stage_path=symlink_stage,
        )
    assert victim.read_bytes() == b"preserve-me"
    real_parent = tmp_path / "real-stage-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-stage-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="parent component open failed"):
        write_owned_compact_snapshot_capsule_stage_file_v1(
            native_plan,
            state=zero,
            metrics=metrics,
            absolute_tick=5,
            stage_path=linked_parent / "through-parent-symlink.capsule",
        )
    assert not (real_parent / "through-parent-symlink.capsule").exists()

    with pytest.raises(RuntimeError, match="byte count|truncated"):
        inspect_compact_snapshot_capsule_v1(
            capsule[:-1], expected_plan=native_plan
        )
    corrupted = bytearray(capsule)
    corrupted[-1] ^= 1
    with pytest.raises(RuntimeError, match="global SHA-256"):
        inspect_compact_snapshot_capsule_v1(
            bytes(corrupted), expected_plan=native_plan
        )
    wrong_version = bytearray(capsule)
    struct.pack_into("<H", wrong_version, 16, 2)
    _rewrite_global_sha(wrong_version)
    with pytest.raises(RuntimeError, match="version differs"):
        inspect_compact_snapshot_capsule_v1(
            bytes(wrong_version), expected_plan=native_plan
        )
    wrong_profile = bytearray(capsule)
    profile_descriptor = 384 + 2 * 256
    profile_payload = struct.unpack_from(
        "<Q", wrong_profile, profile_descriptor + 48
    )[0]
    wrong_profile[profile_payload] ^= 1
    _rewrite_section_sha(wrong_profile, profile_descriptor)
    _rewrite_global_sha(wrong_profile)
    with pytest.raises(RuntimeError, match="materialization profile"):
        inspect_compact_snapshot_capsule_v1(
            bytes(wrong_profile), expected_plan=native_plan
        )
    wrong_plan = bytearray(capsule)
    plan_descriptor = 384 + 3 * 256
    plan_payload = struct.unpack_from("<Q", wrong_plan, plan_descriptor + 48)[0]
    wrong_plan[plan_payload] ^= 1
    _rewrite_section_sha(wrong_plan, plan_descriptor)
    _rewrite_global_sha(wrong_plan)
    with pytest.raises(RuntimeError, match="canonical plan section"):
        inspect_compact_snapshot_capsule_v1(
            bytes(wrong_plan), expected_plan=native_plan
        )


def test_compact_native_converter_rejects_bad_capsules_plans_and_stage_paths(
    native_snapshot_artifact: Path,
    native_plan,
    host_inputs,
    tmp_path: Path,
) -> None:
    state, metrics = host_inputs
    capsule_path = (tmp_path / "source.compact.capsule").absolute()
    write_owned_compact_snapshot_capsule_stage_file_v1(
        native_plan,
        state=_state_at_tick(state, 5),
        metrics=metrics,
        absolute_tick=5,
        stage_path=capsule_path,
    )
    capsule = capsule_path.read_bytes()

    truncated_path = (tmp_path / "truncated.compact.capsule").absolute()
    truncated_path.write_bytes(capsule[:-1])
    truncated_output = (tmp_path / "truncated.jsonl.gz").absolute()
    with pytest.raises(RuntimeError, match="byte count|truncated"):
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            native_plan,
            capsule_stage_path=truncated_path,
            gzip_stage_path=truncated_output,
        )
    assert not truncated_output.exists()

    corrupted = bytearray(capsule)
    corrupted[-1] ^= 1
    corrupt_path = (tmp_path / "corrupt.compact.capsule").absolute()
    corrupt_path.write_bytes(corrupted)
    corrupt_output = (tmp_path / "corrupt.jsonl.gz").absolute()
    with pytest.raises(RuntimeError, match="global SHA-256"):
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            native_plan,
            capsule_stage_path=corrupt_path,
            gzip_stage_path=corrupt_output,
        )
    assert not corrupt_output.exists()

    other_semantic = compile_semantic_snapshot_plan_v1(
        world_uids=(101, 202),
        world_cells=_cells(),
        native_world_control_set_sha256=CONTROL_SHA,
        execution_publication_context_sha256="c" * 64,
        catalog_absolute_tick=0,
    )
    other_native = compile_snapshot_plan_v1(
        other_semantic,
        extension_path=native_snapshot_artifact,
    )
    wrong_plan_output = (tmp_path / "wrong-plan.jsonl.gz").absolute()
    with pytest.raises(RuntimeError, match="identities differ"):
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            other_native,
            capsule_stage_path=capsule_path,
            gzip_stage_path=wrong_plan_output,
        )
    assert not wrong_plan_output.exists()

    collision = (tmp_path / "collision.jsonl.gz").absolute()
    collision.write_bytes(b"preserve-existing-output")
    with pytest.raises(RuntimeError, match="exclusive create"):
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            native_plan,
            capsule_stage_path=capsule_path,
            gzip_stage_path=collision,
        )
    assert collision.read_bytes() == b"preserve-existing-output"

    linked_input = (tmp_path / "linked-input.compact.capsule").absolute()
    linked_input.symlink_to(capsule_path)
    linked_input_output = (tmp_path / "linked-input.jsonl.gz").absolute()
    with pytest.raises(RuntimeError, match="capsule stage open failed"):
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            native_plan,
            capsule_stage_path=linked_input,
            gzip_stage_path=linked_input_output,
        )
    assert not linked_input_output.exists()

    output_victim = (tmp_path / "output-victim").absolute()
    output_victim.write_bytes(b"preserve-symlink-target")
    linked_output = (tmp_path / "linked-output.jsonl.gz").absolute()
    linked_output.symlink_to(output_victim)
    with pytest.raises(RuntimeError, match="exclusive create"):
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            native_plan,
            capsule_stage_path=capsule_path,
            gzip_stage_path=linked_output,
        )
    assert output_victim.read_bytes() == b"preserve-symlink-target"

    real_parent = (tmp_path / "converter-real-parent").absolute()
    real_parent.mkdir()
    linked_parent = (tmp_path / "converter-linked-parent").absolute()
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="parent component open failed"):
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            native_plan,
            capsule_stage_path=capsule_path,
            gzip_stage_path=linked_parent / "through-symlink.jsonl.gz",
        )
    assert not (real_parent / "through-symlink.jsonl.gz").exists()

    with pytest.raises(ValueError, match="must be distinct"):
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            native_plan,
            capsule_stage_path=capsule_path,
            gzip_stage_path=capsule_path,
        )

    unsupported_output = (tmp_path / "unsupported.jsonl.gz").absolute()
    with pytest.raises(RuntimeError, match="policy is unsupported"):
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            native_plan,
            capsule_stage_path=capsule_path,
            gzip_stage_path=unsupported_output,
            encoding_policy="UNSUPPORTED",
        )
    assert not unsupported_output.exists()


@pytest.mark.parametrize("install_foreign_replacement", [False, True])
def test_compact_stage_path_identity_race_preserves_moved_and_foreign_entries(
    native_plan,
    semantic_plan,
    host_inputs,
    tmp_path: Path,
    install_foreign_replacement: bool,
) -> None:
    state, metrics = host_inputs
    mapping = semantic_plan.to_mapping()
    fields = {row["name"]: row for row in mapping["layout"]["fields"]}
    stride = int(mapping["state_input"]["shape"][1])
    capacity = int(mapping["layout"]["agent_capacity"])
    raw = bytearray(_state_at_tick(state, 5).numpy().tobytes())
    for world in range(int(mapping["world_count"])):
        base = world * stride
        for word in range(capacity // 32):
            struct.pack_into(
                "<I",
                raw,
                base + int(fields["alive_words"]["offset_bytes"]) + word * 4,
                (1 << 32) - 1,
            )
        for slot in range(capacity):
            struct.pack_into(
                "<Q",
                raw,
                base + int(fields["agent_id"]["offset_bytes"]) + slot * 8,
                slot + 1,
            )
    full_state = torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(
        state.shape
    )
    canonical_bytes = int(mapping["layout"]["canonical_state_bytes_per_world"])
    captured_state = full_state[:, :canonical_bytes].contiguous()
    foreign_content = b"foreign-stage-replacement-v1"

    for attempt in range(32):
        stage_path = (tmp_path / f"race-{attempt}.capsule").absolute()
        moved_path = (tmp_path / f"race-{attempt}.owned-moved").absolute()
        racer_started = threading.Event()
        racer_finished = threading.Event()
        racer_errors: list[BaseException] = []

        def race_path() -> None:
            try:
                racer_started.set()
                deadline = time.monotonic() + 5.0
                while True:
                    try:
                        os.rename(stage_path, moved_path)
                        break
                    except FileNotFoundError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("stage path was never created")
                        time.sleep(0)
                if install_foreign_replacement:
                    descriptor = os.open(
                        stage_path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                    try:
                        assert os.write(descriptor, foreign_content) == len(
                            foreign_content
                        )
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            except BaseException as error:  # pragma: no cover - surfaced below
                racer_errors.append(error)
            finally:
                racer_finished.set()

        racer = threading.Thread(target=race_path, daemon=True)
        racer.start()
        assert racer_started.wait(timeout=1.0)
        observed_error: RuntimeError | None = None
        try:
            write_owned_compact_snapshot_capsule_stage_file_v1(
                native_plan,
                state=captured_state,
                metrics=metrics,
                absolute_tick=5,
                stage_path=stage_path,
            )
        except RuntimeError as error:
            observed_error = error
        assert racer_finished.wait(timeout=6.0)
        racer.join(timeout=1.0)
        assert not racer.is_alive()
        if racer_errors:
            raise racer_errors[0]
        if observed_error is not None:
            assert moved_path.is_file()
            assert moved_path.stat().st_size > len(foreign_content)
            if install_foreign_replacement:
                assert stage_path.read_bytes() == foreign_content
                assert "FOREIGN_REPLACEMENT_PRESERVED" in str(observed_error)
            else:
                assert not stage_path.exists()
                assert "OWNED_ENTRY_ABSENT_NO_CLEANUP" in str(observed_error)
            break
        if stage_path.exists():
            stage_path.unlink()
        if moved_path.exists():
            moved_path.unlink()
    else:  # pragma: no cover - scheduling failure is a substantive test failure
        pytest.fail("could not interleave the adversarial stage rename race")


def test_compact_all_alive_extrema_and_raw_size_reduction(
    native_plan,
    semantic_plan,
    host_inputs,
    tmp_path: Path,
) -> None:
    state, metrics = host_inputs
    mapping = semantic_plan.to_mapping()
    fields = {row["name"]: row for row in mapping["layout"]["fields"]}
    stride = int(mapping["state_input"]["shape"][1])
    capacity = int(mapping["layout"]["agent_capacity"])
    raw = bytearray(_state_at_tick(state, 9).numpy().tobytes())
    world = 1
    base = world * stride
    for word in range(capacity // 32):
        struct.pack_into(
            "<I",
            raw,
            base + int(fields["alive_words"]["offset_bytes"]) + word * 4,
            (1 << 32) - 1,
        )
    for slot in range(capacity):
        agent_id = (1 << 64) - 1 if slot == capacity - 1 else slot + 1
        struct.pack_into(
            "<Q",
            raw,
            base + int(fields["agent_id"]["offset_bytes"]) + slot * 8,
            agent_id,
        )
    struct.pack_into(
        "<h", raw, base + int(fields["energy"]["offset_bytes"]), -(1 << 15)
    )
    struct.pack_into(
        "<h",
        raw,
        base + int(fields["health"]["offset_bytes"]) + 2,
        (1 << 15) - 1,
    )
    full_state = torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(state.shape)
    extrema_metrics = metrics.clone()
    extrema_metrics[1, 0, 0] = -(1 << 63)
    extrema_metrics[1, 1, 0] = (1 << 63) - 1
    canonical_bytes = int(mapping["layout"]["canonical_state_bytes_per_world"])
    stage_path = (tmp_path / "all-alive.compact.capsule").absolute()
    receipt = write_owned_compact_snapshot_capsule_stage_file_v1(
        native_plan,
        state=full_state[:, :canonical_bytes].contiguous(),
        metrics=extrema_metrics,
        absolute_tick=9,
        stage_path=stage_path,
    )
    capsule = stage_path.read_bytes()
    assert receipt["active_agent_count"] == capacity
    assert _unpack_column(capsule, native_plan, "agent_id", "Q")[-1] == (
        1 << 64
    ) - 1
    compact_metrics = _unpack_column(
        capsule, native_plan, "lifetime_metrics", "q"
    )
    assert compact_metrics[0] == -(1 << 63)
    assert compact_metrics[len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)] == (
        1 << 63
    ) - 1

    legacy_root = tmp_path / "legacy-all-alive"
    legacy = write_compact_science_snapshot_v1(
        legacy_root,
        state=full_state,
        active_metrics_projection={
            "next_absolute_tick": 9,
            "world_uids": (101, 202),
            "metric_fields": COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
            "tensor": extrema_metrics,
        },
        world_cells=_cells(),
        absolute_tick=9,
        native_world_control_set_sha256=CONTROL_SHA,
        include_genome_catalog=False,
    )
    legacy_gzip = (legacy_root / str(legacy["data_file"])).read_bytes()
    canonical_jsonl = gzip.decompress(legacy_gzip)
    converted_path = (tmp_path / "all-alive.converted.jsonl.gz").absolute()
    converted = (
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            native_plan,
            capsule_stage_path=stage_path,
            gzip_stage_path=converted_path,
        )
    )
    assert converted_path.read_bytes() == legacy_gzip
    assert converted["active_agent_count"] == capacity
    assert converted["row_count"] == capacity + 1
    dense = bytes(
        native_plan.pack_capsule(full_state, extrema_metrics, absolute_tick=9)
    )
    assert len(capsule) < len(dense)
    assert len(capsule) < len(canonical_jsonl)
    assert _reconstruct_compact_rows(
        capsule, native_plan, semantic_plan
    ) == [json.loads(line) for line in canonical_jsonl.splitlines()]


def test_compact_stage_supports_larger_dynamic_world_count(
    native_snapshot_artifact: Path,
    tmp_path: Path,
) -> None:
    world_count = 9
    world_uids, cells = _cells_for_count(world_count)
    semantic = compile_semantic_snapshot_plan_v1(
        world_uids=world_uids,
        world_cells=cells,
        native_world_control_set_sha256=CONTROL_SHA,
        execution_publication_context_sha256=EXECUTION_PUBLICATION_CONTEXT_SHA,
        catalog_absolute_tick=None,
    )
    native = compile_snapshot_plan_v1(
        semantic, extension_path=native_snapshot_artifact
    )
    mapping = semantic.to_mapping()
    canonical_bytes = int(mapping["layout"]["canonical_state_bytes_per_world"])
    raw = bytearray(world_count * canonical_bytes)
    for index, uid in enumerate(world_uids):
        struct.pack_into("<Q", raw, index * canonical_bytes, uid)
        struct.pack_into("<Q", raw, index * canonical_bytes + 8, 5)
    state = torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(
        world_count, canonical_bytes
    )
    metrics = torch.zeros(tuple(mapping["metrics_input"]["shape"]), dtype=torch.int64)
    path = (tmp_path / "w9.compact.capsule").absolute()
    receipt = write_owned_compact_snapshot_capsule_stage_file_v1(
        native,
        state=state,
        metrics=metrics,
        absolute_tick=5,
        stage_path=path,
    )
    assert receipt["world_count"] == world_count
    assert receipt["row_count"] == world_count
    assert inspect_compact_snapshot_capsule_v1(
        path.read_bytes(), expected_plan=native
    )["world_count"] == world_count


def test_compact_stage_preserves_unsigned_world_and_agent_extrema(
    native_snapshot_artifact: Path,
    tmp_path: Path,
) -> None:
    maximum_u64 = (1 << 64) - 1
    cells = {
        "u64-world": {
            "science_logical_world_id": maximum_u64,
            "science_ownership": {"arm": "extrema"},
            "science_world_sha256": {
                "algorithm": "sha256",
                "hex": "4" * 64,
            },
            "science_acquisition_sha256": {
                "algorithm": "sha256",
                "hex": "5" * 64,
            },
            "science_acquisition": {"snapshot_ticks": [5]},
        }
    }
    semantic = compile_semantic_snapshot_plan_v1(
        world_uids=(maximum_u64,),
        world_cells=cells,
        native_world_control_set_sha256=CONTROL_SHA,
        execution_publication_context_sha256=EXECUTION_PUBLICATION_CONTEXT_SHA,
        catalog_absolute_tick=None,
    )
    native = compile_snapshot_plan_v1(
        semantic, extension_path=native_snapshot_artifact
    )
    mapping = semantic.to_mapping()
    canonical_bytes = int(mapping["layout"]["canonical_state_bytes_per_world"])
    resident_bytes = int(mapping["state_input"]["shape"][1])
    fields = {row["name"]: row for row in mapping["layout"]["fields"]}
    raw = bytearray(resident_bytes)
    struct.pack_into("<Q", raw, 0, maximum_u64)
    struct.pack_into("<Q", raw, 8, 5)
    struct.pack_into(
        "<I", raw, int(fields["alive_words"]["offset_bytes"]), 1
    )
    struct.pack_into(
        "<Q", raw, int(fields["agent_id"]["offset_bytes"]), maximum_u64
    )
    state = torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(
        1, resident_bytes
    )
    metrics = torch.zeros(tuple(mapping["metrics_input"]["shape"]), dtype=torch.int64)
    metrics[0, 0, 0] = -(1 << 63)
    metrics[0, 0, 1] = (1 << 63) - 1
    path = (tmp_path / "u64.compact.capsule").absolute()
    write_owned_compact_snapshot_capsule_stage_file_v1(
        native,
        state=state[:, :canonical_bytes].contiguous(),
        metrics=metrics,
        absolute_tick=5,
        stage_path=path,
    )
    capsule = path.read_bytes()
    assert _unpack_column(capsule, native, "world_uid", "Q") == (maximum_u64,)
    assert _unpack_column(capsule, native, "agent_id", "Q") == (maximum_u64,)
    assert _unpack_column(capsule, native, "lifetime_metrics", "q")[:2] == (
        -(1 << 63),
        (1 << 63) - 1,
    )
    legacy_root = tmp_path / "legacy-u64-extrema"
    legacy = write_compact_science_snapshot_v1(
        legacy_root,
        state=state,
        active_metrics_projection={
            "next_absolute_tick": 5,
            "world_uids": (maximum_u64,),
            "metric_fields": COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
            "tensor": metrics,
        },
        world_cells=cells,
        absolute_tick=5,
        native_world_control_set_sha256=CONTROL_SHA,
        include_genome_catalog=False,
    )
    legacy_gzip = (legacy_root / str(legacy["data_file"])).read_bytes()
    converted_path = (tmp_path / "u64.converted.jsonl.gz").absolute()
    converted = (
        convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
            native,
            capsule_stage_path=path,
            gzip_stage_path=converted_path,
        )
    )
    assert converted_path.read_bytes() == legacy_gzip
    assert converted["active_agent_count"] == 1
    converted_rows = [
        json.loads(line)
        for line in gzip.decompress(legacy_gzip).splitlines()
    ]
    assert converted_rows[0]["world_uid"] == maximum_u64
    assert converted_rows[1]["agent_id"] == maximum_u64
    assert converted_rows[1]["lifetime_metrics"][
        COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1[0]
    ] == -(1 << 63)
