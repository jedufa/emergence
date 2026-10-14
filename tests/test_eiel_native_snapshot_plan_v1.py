from __future__ import annotations

import hashlib
import json
import math
import struct

import pytest

from engine.native_runtime.cuda_snapshot_plan_v1 import (
    EIEL_SNAPSHOT_CAPSULE_ALIGNMENT_BYTES_V1,
    EIEL_SNAPSHOT_CAPSULE_DTYPES_V1,
    EIEL_SNAPSHOT_CAPSULE_ENDIAN_MARKER_V1,
    EIEL_SNAPSHOT_CAPSULE_HEADER_BYTES_V1,
    EIEL_SNAPSHOT_CAPSULE_MAGIC_V1,
    EIEL_SNAPSHOT_CAPSULE_PROJECTION_HOST_CONTIGUOUS_SYNC_V1,
    EIEL_SNAPSHOT_CAPSULE_SECTION_DESCRIPTOR_BYTES_V1,
    EIEL_SNAPSHOT_CAPSULE_SECTION_KINDS_V1,
    EIEL_SNAPSHOT_CAPSULE_VERSION_V1,
    CudaSnapshotPlanError,
    EielSnapshotCapsuleError,
    NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1,
    NATIVE_SNAPSHOT_STATE_CAPTURE_FULL_RESIDENT_V1,
    NATIVE_SNAPSHOT_STATE_CAPTURE_TRANSPORT_SCHEMA_V1,
    compile_snapshot_plan_v1,
    snapshot_capsule_section_bytes_v1,
    validate_snapshot_capsule_v1,
    validate_snapshot_plan_v1,
)
from engine.perf.cuda_full_evolution_v2.layout import state_layout_v2

CONTROL_SHA256 = "a" * 64
EXECUTION_PUBLICATION_CONTEXT_SHA256 = "b" * 64


def _cells() -> dict[str, dict[str, object]]:
    # Deliberately opposite to native projection order.
    return {
        "cell-two": {
            "science_logical_world_id": 202,
            "science_ownership": {"study": "study-β", "arm": "two"},
            "science_world_sha256": {
                "algorithm": "sha256",
                "hex": "2" * 64,
            },
            "science_acquisition_sha256": {
                "algorithm": "sha256",
                "hex": "3" * 64,
            },
            "science_acquisition": {"snapshot_ticks": [5, 9]},
        },
        "cell-one": {
            "science_logical_world_id": 101,
            "science_ownership": {"arm": "one", "study": "study-α"},
            "science_world_sha256": {
                "algorithm": "sha256",
                "hex": "1" * 64,
            },
            "science_acquisition_sha256": {
                "algorithm": "sha256",
                "hex": "2" * 64,
            },
            "science_acquisition": {"snapshot_ticks": [0, 5]},
        },
    }


def _plan(*, control_sha256: str = CONTROL_SHA256):
    return compile_snapshot_plan_v1(
        world_uids=(101, 202),
        world_cells=_cells(),
        native_world_control_set_sha256=control_sha256,
        execution_publication_context_sha256=(
            EXECUTION_PUBLICATION_CONTEXT_SHA256
        ),
        catalog_absolute_tick=0,
    )


def test_plan_compiles_exact_dynamic_selection_and_layout_geometry() -> None:
    compiled = _plan()
    value = compiled.to_mapping()
    layout = state_layout_v2()

    assert hashlib.sha256(compiled.canonical_plan_bytes).hexdigest() == (
        compiled.plan_sha256
    )
    assert value["world_count"] == 2
    assert [world["world_uid"] for world in value["worlds"]] == [101, 202]
    assert value["snapshot_boundaries"] == [
        {
            "absolute_tick": 0,
            "world_indices": [0],
            "world_uids": [101],
            "include_genome_catalog": True,
        },
        {
            "absolute_tick": 5,
            "world_indices": [0, 1],
            "world_uids": [101, 202],
            "include_genome_catalog": False,
        },
        {
            "absolute_tick": 9,
            "world_indices": [1],
            "world_uids": [202],
            "include_genome_catalog": False,
        },
    ]
    assert value["layout"]["state_bytes_per_world"] == layout.state_bytes_per_world
    assert value["layout"]["canonical_state_bytes_per_world"] == (
        layout.canonical_checkpoint_bytes_per_world
    )
    assert value["state_input"] == {
        "dtype": "u8",
        "shape": [2, layout.state_bytes_per_world],
        "contiguity": "C",
        "world_stride_bytes": layout.state_bytes_per_world,
    }
    assert value["state_capture_transport"] == {
        "schema_version": NATIVE_SNAPSHOT_STATE_CAPTURE_TRANSPORT_SCHEMA_V1,
        "resident_source_contract": "state_input",
        "prefix_semantics": "leading_bytes_of_each_resident_world_row_v1",
        "preferred_mode": NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1,
        "accepted_modes": [
            {
                "mode": NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1,
                "dtype": "u8",
                "shape": [2, layout.canonical_checkpoint_bytes_per_world],
                "contiguity": "C",
                "world_stride_bytes": (
                    layout.canonical_checkpoint_bytes_per_world
                ),
                "total_bytes": (
                    2 * layout.canonical_checkpoint_bytes_per_world
                ),
            },
            {
                "mode": NATIVE_SNAPSHOT_STATE_CAPTURE_FULL_RESIDENT_V1,
                "dtype": "u8",
                "shape": [2, layout.state_bytes_per_world],
                "contiguity": "C",
                "world_stride_bytes": layout.state_bytes_per_world,
                "total_bytes": 2 * layout.state_bytes_per_world,
            },
        ],
    }
    assert value["layout"]["agent_capacity"] == layout.field("agent_id").shape[0]
    assert value["layout"]["resource_cell_count"] == layout.field("resource").shape[0]
    assert value["layout"]["policy_geometry"]["shape"] == list(
        layout.field("policy_alleles").shape
    )
    assert value["layout"]["trait_geometry"]["shape"] == list(
        layout.field("trait_alleles").shape
    )
    assert len(value["metric_fields"]) == 21
    assert value["sections"][-1]["source_shape"] == value["metrics_input"]["shape"]
    assert all(section["world_major"] is True for section in value["sections"])
    assert all("element_shape" in section for section in value["sections"])
    assert b'"row_kind"' not in compiled.canonical_plan_bytes

    first_world = value["worlds"][0]
    assert json.loads(first_world["cell_id_json"]) == "cell-one"
    assert json.loads(first_world["science_ownership_json"]) == {
        "arm": "one",
        "study": "study-α",
    }
    assert "α" not in first_world["science_ownership_json"]


def test_plan_is_deterministic_and_canonical() -> None:
    first = _plan()
    cells = _cells()
    reordered = {"cell-one": cells["cell-one"], "cell-two": cells["cell-two"]}
    second = compile_snapshot_plan_v1(
        world_uids=[101, 202],
        world_cells=reordered,
        native_world_control_set_sha256=CONTROL_SHA256,
        execution_publication_context_sha256=(
            EXECUTION_PUBLICATION_CONTEXT_SHA256
        ),
        catalog_absolute_tick=0,
    )
    assert first == second
    assert validate_snapshot_plan_v1(first.canonical_plan_bytes) == first
    assert validate_snapshot_plan_v1(first.to_mapping()) == first
    noncanonical = json.dumps(first.to_mapping(), indent=2).encode("utf-8")
    with pytest.raises(CudaSnapshotPlanError, match="not exact canonical"):
        validate_snapshot_plan_v1(noncanonical)


def test_execution_publication_context_is_required_and_plan_bound() -> None:
    first = _plan()
    second = compile_snapshot_plan_v1(
        world_uids=(101, 202),
        world_cells=_cells(),
        native_world_control_set_sha256=CONTROL_SHA256,
        execution_publication_context_sha256="c" * 64,
        catalog_absolute_tick=0,
    )
    assert first.plan_sha256 != second.plan_sha256
    assert first.to_mapping()["execution_publication_context_sha256"] == (
        EXECUTION_PUBLICATION_CONTEXT_SHA256
    )
    with pytest.raises(CudaSnapshotPlanError, match="execution/publication"):
        compile_snapshot_plan_v1(
            world_uids=(101, 202),
            world_cells=_cells(),
            native_world_control_set_sha256=CONTROL_SHA256,
            execution_publication_context_sha256="c" * 63,
            catalog_absolute_tick=0,
        )


@pytest.mark.parametrize("world_count", [1, 3, 9])
def test_plan_batch_and_schedule_geometry_remain_dynamic(world_count: int) -> None:
    cells: dict[str, dict[str, object]] = {}
    world_uids: list[int] = []
    for index in range(world_count):
        uid = 1000 + index
        world_uids.append(uid)
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
            "science_acquisition": {
                "snapshot_ticks": [5] if index % 2 == 0 else [5, 9]
            },
        }
    plan = compile_snapshot_plan_v1(
        world_uids=world_uids,
        world_cells=cells,
        native_world_control_set_sha256=CONTROL_SHA256,
        execution_publication_context_sha256=(
            EXECUTION_PUBLICATION_CONTEXT_SHA256
        ),
        catalog_absolute_tick=None,
    ).to_mapping()
    assert plan["world_count"] == world_count
    assert plan["state_input"]["shape"][0] == world_count
    assert all(
        mode["shape"][0] == world_count
        for mode in plan["state_capture_transport"]["accepted_modes"]
    )
    assert plan["state_capture_transport"]["accepted_modes"][0][
        "total_bytes"
    ] == math.prod(
        plan["state_capture_transport"]["accepted_modes"][0]["shape"]
    )
    assert plan["metrics_input"]["shape"][0] == world_count
    assert plan["snapshot_boundaries"][0]["world_indices"] == list(
        range(world_count)
    )
    if world_count == 1:
        assert len(plan["snapshot_boundaries"]) == 1
    else:
        assert plan["snapshot_boundaries"][-1]["world_indices"] == [
            index for index in range(world_count) if index % 2 == 1
        ]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda cells: cells["cell-one"]["science_acquisition"].update(
                {"snapshot_ticks": [5, 0]}
            ),
            "strictly increasing",
        ),
        (
            lambda cells: cells["cell-one"].update({"science_logical_world_id": True}),
            "logical-world identities",
        ),
        (
            lambda cells: cells["cell-one"].update(
                {
                    "science_world_sha256": {
                        "algorithm": "sha256",
                        "hex": "A" * 64,
                    }
                }
            ),
            "digest object",
        ),
        (
            lambda cells: cells["cell-one"].update(
                {"science_ownership": {"bad": math.nan}}
            ),
            "canonical JSON",
        ),
    ],
)
def test_plan_compilation_fails_closed(mutation, match: str) -> None:
    cells = _cells()
    mutation(cells)
    with pytest.raises(CudaSnapshotPlanError, match=match):
        compile_snapshot_plan_v1(
            world_uids=(101, 202),
            world_cells=cells,
            native_world_control_set_sha256=CONTROL_SHA256,
            execution_publication_context_sha256=(
                EXECUTION_PUBLICATION_CONTEXT_SHA256
            ),
            catalog_absolute_tick=0,
        )


def test_plan_rejects_coverage_digest_and_semantic_tampering() -> None:
    cells = _cells()
    with pytest.raises(CudaSnapshotPlanError, match="exactly cover"):
        compile_snapshot_plan_v1(
            world_uids=(101,),
            world_cells=cells,
            native_world_control_set_sha256=CONTROL_SHA256,
            execution_publication_context_sha256=(
                EXECUTION_PUBLICATION_CONTEXT_SHA256
            ),
            catalog_absolute_tick=0,
        )
    with pytest.raises(CudaSnapshotPlanError, match="lowercase SHA-256"):
        compile_snapshot_plan_v1(
            world_uids=(101, 202),
            world_cells=cells,
            native_world_control_set_sha256="f" * 63,
            execution_publication_context_sha256=(
                EXECUTION_PUBLICATION_CONTEXT_SHA256
            ),
            catalog_absolute_tick=0,
        )
    value = _plan().to_mapping()
    value["snapshot_boundaries"][0]["world_uids"] = [202]
    with pytest.raises(CudaSnapshotPlanError, match="boundary selection"):
        validate_snapshot_plan_v1(value)
    value = _plan().to_mapping()
    value["selection_laws"]["alive_selection"] = "publish_every_slot"
    with pytest.raises(CudaSnapshotPlanError, match="selection laws"):
        validate_snapshot_plan_v1(value)
    value = _plan().to_mapping()
    value["layout"]["fields"][0]["offset_bytes"] = 64
    with pytest.raises(CudaSnapshotPlanError, match="source layout"):
        validate_snapshot_plan_v1(value)
    value = _plan().to_mapping()
    value["state_capture_transport"]["preferred_mode"] = (
        NATIVE_SNAPSHOT_STATE_CAPTURE_FULL_RESIDENT_V1
    )
    with pytest.raises(CudaSnapshotPlanError, match="capture transport"):
        validate_snapshot_plan_v1(value)
    value = _plan().to_mapping()
    value["state_capture_transport"]["accepted_modes"][0][
        "world_stride_bytes"
    ] += 1
    with pytest.raises(CudaSnapshotPlanError, match="capture transport"):
        validate_snapshot_plan_v1(value)
    value = _plan().to_mapping()
    value["execution_publication_context_sha256"] = "B" * 64
    with pytest.raises(CudaSnapshotPlanError, match="execution/publication"):
        validate_snapshot_plan_v1(value)


def _align(value: int) -> int:
    alignment = EIEL_SNAPSHOT_CAPSULE_ALIGNMENT_BYTES_V1
    return ((value + alignment - 1) // alignment) * alignment


def _build_capsule(plan, *, absolute_tick: int) -> tuple[bytes, dict[str, bytes]]:
    """Independent test encoder for the frozen binary ABI."""

    mapping = plan.to_mapping()
    boundary = next(
        item
        for item in mapping["snapshot_boundaries"]
        if item["absolute_tick"] == absolute_tick
    )
    selected_indices = tuple(boundary["world_indices"])
    selected_payload = b"".join(struct.pack("<Q", index) for index in selected_indices)
    identity_payload = b"".join(
        bytes.fromhex(value)
        for value in (
            plan.plan_sha256,
            mapping["schema_sha256"],
            mapping["layout_sha256"],
            mapping["native_world_control_set_sha256"],
        )
    )
    declarations: list[dict[str, object]] = [
        {
            "name": "plan_identity",
            "kind": "identity",
            "dtype": "u8",
            "shape": [len(identity_payload)],
            "payload": identity_payload,
        },
        {
            "name": "selected_world_indices",
            "kind": "identity",
            "dtype": "u64",
            "shape": [len(selected_indices)],
            "payload": selected_payload,
        },
    ]
    for ordinal, section in enumerate(mapping["sections"], start=2):
        shape = [len(selected_indices), *section["element_shape"]]
        length = len(selected_indices) * section["bytes_per_world"]
        declarations.append(
            {
                "name": section["name"],
                "kind": section["kind"],
                "dtype": section["dtype"],
                "shape": shape,
                "payload": bytes([ordinal % 251]) * length,
            }
        )

    names = b"".join(str(row["name"]).encode("utf-8") for row in declarations)
    section_count = len(declarations)
    directory_offset = EIEL_SNAPSHOT_CAPSULE_HEADER_BYTES_V1
    directory_bytes = section_count * EIEL_SNAPSHOT_CAPSULE_SECTION_DESCRIPTOR_BYTES_V1
    names_offset = directory_offset + directory_bytes
    payload_offset = _align(names_offset + len(names))

    name_cursor = names_offset
    payload_cursor = payload_offset
    descriptors: list[bytes] = []
    payload_regions: list[tuple[int, bytes]] = []
    payload_by_name: dict[str, bytes] = {}
    for row in declarations:
        name_bytes = str(row["name"]).encode("utf-8")
        payload = bytes(row["payload"])
        section_offset = _align(payload_cursor)
        shape = tuple(int(value) for value in row["shape"])
        dtype_code, element_width = EIEL_SNAPSHOT_CAPSULE_DTYPES_V1[str(row["dtype"])]
        element_count = math.prod(shape)
        assert len(payload) == element_count * element_width
        descriptor = bytearray(EIEL_SNAPSHOT_CAPSULE_SECTION_DESCRIPTOR_BYTES_V1)
        struct.pack_into(
            "<IHHII" + "Q" * 7 + "Q" * 8 + "32s",
            descriptor,
            0,
            EIEL_SNAPSHOT_CAPSULE_SECTION_KINDS_V1[str(row["kind"])],
            dtype_code,
            len(shape),
            0,
            0,
            element_width,
            element_count,
            name_cursor,
            len(name_bytes),
            section_offset,
            len(payload),
            len(payload),
            *shape,
            *([0] * (8 - len(shape))),
            hashlib.sha256(payload).digest(),
        )
        descriptors.append(bytes(descriptor))
        payload_regions.append((section_offset, payload))
        payload_by_name[str(row["name"])] = payload
        name_cursor += len(name_bytes)
        payload_cursor = section_offset + len(payload)
    trailer_offset = _align(payload_cursor)
    capsule_bytes = trailer_offset + 32
    raw = bytearray(capsule_bytes)
    raw[directory_offset : directory_offset + directory_bytes] = b"".join(descriptors)
    raw[names_offset : names_offset + len(names)] = names
    for offset, payload in payload_regions:
        raw[offset : offset + len(payload)] = payload
    struct.pack_into(
        "<16sHHIII" + "Q" * 13 + "32s32s32s32sIIQ",
        raw,
        0,
        EIEL_SNAPSHOT_CAPSULE_MAGIC_V1,
        EIEL_SNAPSHOT_CAPSULE_VERSION_V1,
        EIEL_SNAPSHOT_CAPSULE_HEADER_BYTES_V1,
        EIEL_SNAPSHOT_CAPSULE_ENDIAN_MARKER_V1,
        EIEL_SNAPSHOT_CAPSULE_SECTION_DESCRIPTOR_BYTES_V1,
        EIEL_SNAPSHOT_CAPSULE_ALIGNMENT_BYTES_V1,
        0,
        absolute_tick,
        len(selected_indices),
        len(selected_indices),
        section_count,
        directory_offset,
        directory_bytes,
        names_offset,
        len(names),
        payload_offset,
        trailer_offset - payload_offset,
        trailer_offset,
        capsule_bytes,
        bytes.fromhex(plan.plan_sha256),
        bytes.fromhex(mapping["schema_sha256"]),
        bytes.fromhex(mapping["layout_sha256"]),
        bytes.fromhex(mapping["native_world_control_set_sha256"]),
        EIEL_SNAPSHOT_CAPSULE_PROJECTION_HOST_CONTIGUOUS_SYNC_V1,
        0,
        0,
    )
    raw[trailer_offset:] = hashlib.sha256(raw[:trailer_offset]).digest()
    return bytes(raw), payload_by_name


def _reseal(value: bytearray) -> None:
    trailer_offset = struct.unpack_from("<Q", value, 120)[0]
    value[trailer_offset:] = hashlib.sha256(value[:trailer_offset]).digest()


def test_capsule_fixed_header_directory_and_selected_world_round_trip() -> None:
    plan = _plan()
    capsule, payloads = _build_capsule(plan, absolute_tick=0)
    view = validate_snapshot_capsule_v1(capsule, expected_plan=plan)

    assert view.absolute_tick == 0
    assert view.world_count == 1
    assert view.row_count == 1
    assert view.plan_sha256 == plan.plan_sha256
    assert view.sections[0].name == "plan_identity"
    assert view.sections[0].payload_length == 128
    assert view.sections[1].name == "selected_world_indices"
    assert view.sections[1].shape == (1,)
    assert snapshot_capsule_section_bytes_v1(
        capsule, "selected_world_indices", expected_plan=plan
    ) == struct.pack("<Q", 0)
    assert (
        snapshot_capsule_section_bytes_v1(
            capsule, "lifetime_metrics", expected_plan=plan
        )
        == payloads["lifetime_metrics"]
    )
    assert all(section.payload_offset % 64 == 0 for section in view.sections)
    assert all(section.shape[0] == 1 for section in view.sections[2:])


def test_capsule_heterogeneous_tick_selects_the_other_world() -> None:
    plan = _plan()
    capsule, _payloads = _build_capsule(plan, absolute_tick=9)
    view = validate_snapshot_capsule_v1(capsule)
    assert view.plan is None
    assert view.world_count == 1
    assert snapshot_capsule_section_bytes_v1(
        capsule, "selected_world_indices"
    ) == struct.pack("<Q", 1)


def test_capsule_rejects_global_section_reserved_padding_and_truncation_tamper() -> (
    None
):
    plan = _plan()
    capsule, _payloads = _build_capsule(plan, absolute_tick=5)

    global_tamper = bytearray(capsule)
    global_tamper[-1] ^= 1
    with pytest.raises(EielSnapshotCapsuleError, match="global SHA-256"):
        validate_snapshot_capsule_v1(global_tamper)

    section_tamper = bytearray(capsule)
    view = validate_snapshot_capsule_v1(section_tamper)
    target = view.sections[-1]
    section_tamper[target.payload_offset] ^= 1
    _reseal(section_tamper)
    with pytest.raises(EielSnapshotCapsuleError, match="section .* SHA-256"):
        validate_snapshot_capsule_v1(section_tamper)

    header_reserved = bytearray(capsule)
    header_reserved[280] = 1
    _reseal(header_reserved)
    with pytest.raises(EielSnapshotCapsuleError, match="fixed header"):
        validate_snapshot_capsule_v1(header_reserved)

    descriptor_reserved = bytearray(capsule)
    descriptor_reserved[EIEL_SNAPSHOT_CAPSULE_HEADER_BYTES_V1 + 168] = 1
    _reseal(descriptor_reserved)
    with pytest.raises(EielSnapshotCapsuleError, match="reserved bytes"):
        validate_snapshot_capsule_v1(descriptor_reserved)

    names_offset = struct.unpack_from("<Q", capsule, 88)[0]
    names_bytes = struct.unpack_from("<Q", capsule, 96)[0]
    payload_offset = struct.unpack_from("<Q", capsule, 104)[0]
    assert names_offset + names_bytes < payload_offset
    padding_tamper = bytearray(capsule)
    padding_tamper[names_offset + names_bytes] = 1
    _reseal(padding_tamper)
    with pytest.raises(EielSnapshotCapsuleError, match="name-table padding"):
        validate_snapshot_capsule_v1(padding_tamper)

    with pytest.raises(EielSnapshotCapsuleError, match="truncated|region"):
        validate_snapshot_capsule_v1(capsule[:-1])


def test_capsule_rejects_wrong_expected_plan_and_unscheduled_tick() -> None:
    plan = _plan()
    capsule, _payloads = _build_capsule(plan, absolute_tick=0)
    other = _plan(control_sha256="b" * 64)
    with pytest.raises(EielSnapshotCapsuleError, match="expected plan"):
        validate_snapshot_capsule_v1(capsule, expected_plan=other)

    tampered = bytearray(capsule)
    struct.pack_into("<Q", tampered, 40, 7)
    _reseal(tampered)
    with pytest.raises(EielSnapshotCapsuleError, match="not requested"):
        validate_snapshot_capsule_v1(tampered, expected_plan=plan)
