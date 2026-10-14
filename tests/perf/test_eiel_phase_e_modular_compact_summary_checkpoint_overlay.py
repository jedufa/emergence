from __future__ import annotations

import copy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch

from engine.native_runtime.compact_summary_checkpoint_descriptor_plan import (
    PHASE_E_MODULAR_OVERLAY_PLAN_SCHEMA_VERSION,
    CompactSummaryCheckpointDescriptorPlanError,
    generate_phase_e_modular_checkpoint_descriptor_overlay,
    maybe_generate_phase_e_modular_checkpoint_descriptor_overlay,
    validate_phase_e_modular_checkpoint_descriptor_overlay,
)
from engine.perf.cuda_architecture_b_compact_summary_checkpoint_runtime import (
    ACTIVE_LENGTH_TENSOR_INDEX_V1,
    ALIVE_MASK_TENSOR_INDEX_V1,
    CANONICALIZATION_MODE_V1,
    MODE_CODES,
    MODE_EXACT,
    NULL_TENSOR_INDEX,
    PHASE_E_MODULAR_RUNTIME_SCHEMA_VERSION,
    bind_phase_e_modular_checkpoint_runtime,
    maybe_bind_phase_e_modular_checkpoint_runtime,
    validate_phase_e_modular_checkpoint_runtime,
)
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_METADATA_KEY,
    MODULAR_STORAGE_PROJECTION_PREFIX,
    build_modular_storage_layout,
    maybe_modular_storage_projection_payload,
)


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V1_PLAN = (
    ROOT / "docs/300_architecture-specifications/370_specifications/EIEL-DOC-000840--compact-summary-checkpoint-descriptor-plan-v1.json"
)
FROZEN_V1_PLAN_SHA256 = (
    "0a778c063cb6ff2c63e9721d25c1c1c937e44723863ac92938aa3f185c9d044d"
)


def _projection(capacity: int = 3) -> tuple[dict, dict]:
    payload = maybe_modular_storage_projection_payload(
        True,
        [None] * capacity,
        agent_capacity=capacity,
        profile_id="b2-12x32-v1",
    )
    metadata = {"unrelated_base_metadata": "preserved-outside-overlay"}
    metadata.update(payload["metadata"])
    tensors = {"energy": torch.zeros((capacity,), dtype=torch.float32)}
    tensors.update(payload["tensors"])
    return metadata, tensors


def test_absent_overlay_returns_no_plan_runtime_or_overlay_metadata() -> None:
    metadata = {"world": "flat"}
    tensors = {"energy": torch.zeros((3,), dtype=torch.float32)}

    assert (
        maybe_generate_phase_e_modular_checkpoint_descriptor_overlay(
            metadata, tensors
        )
        is None
    )
    assert (
        maybe_bind_phase_e_modular_checkpoint_runtime(
            None,
            metadata,
            tensors,
            generation_count=2,
        )
        is None
    )
    assert MODULAR_STORAGE_METADATA_KEY not in metadata
    assert not any(
        name.startswith(MODULAR_STORAGE_PROJECTION_PREFIX) for name in tensors
    )


def test_overlay_selects_exact_metadata_and_lexicographic_fixed_planes() -> None:
    metadata, tensors = _projection()
    overlay = generate_phase_e_modular_checkpoint_descriptor_overlay(
        metadata, tensors
    )
    layout = build_modular_storage_layout(3, "b2-12x32-v1")

    assert overlay["schema_version"] == PHASE_E_MODULAR_OVERLAY_PLAN_SCHEMA_VERSION
    assert set(overlay["source_contract"]["projection_metadata"]) == {
        MODULAR_STORAGE_METADATA_KEY
    }
    assert overlay["source_contract"]["layout"] == layout.to_dict()
    expected_names = tuple(
        sorted(f"{MODULAR_STORAGE_PROJECTION_PREFIX}{field.name}" for field in layout.fields)
    )
    descriptors = overlay["descriptors"]
    assert tuple(row["tensor_name"] for row in descriptors) == expected_names
    assert [row["descriptor_index"] for row in descriptors] == list(
        range(len(descriptors))
    )
    by_logical_name = {row["logical_tensor_name"]: row for row in descriptors}
    for field in layout.fields:
        row = by_logical_name[field.name]
        assert row["dtype"] == field.dtype_name
        assert row["fixed_shape"] == list(field.shape)
        assert row["rank"] == len(field.shape)
        assert row["raw_canonical_mode"] == MODE_EXACT
    assert [batch["count"] for batch in overlay["batches"]] == [16, 16, 16, 9]
    assert all(batch["count"] <= 16 for batch in overlay["batches"])
    validate_phase_e_modular_checkpoint_descriptor_overlay(
        overlay, metadata, tensors
    )


def test_overlay_runtime_binds_only_exact_prefixed_cpu_planes() -> None:
    metadata, tensors = _projection()
    overlay = generate_phase_e_modular_checkpoint_descriptor_overlay(
        metadata, tensors
    )

    runtime = bind_phase_e_modular_checkpoint_runtime(
        overlay,
        metadata,
        tensors,
        generation_count=2,
    )

    assert runtime.schema_version == PHASE_E_MODULAR_RUNTIME_SCHEMA_VERSION
    assert runtime.backend == "torch"
    assert runtime.device == "cpu"
    assert runtime.cell_count == 1
    assert runtime.descriptor_count == 57
    assert runtime.descriptor_names == tuple(sorted(runtime.descriptor_names))
    assert runtime.raw_digests.shape == (2, 1, 57, 32)
    assert runtime.canonical_digests.shape == (2, 1, 57, 32)
    assert runtime.status.shape == (2, 1, 57)
    assert runtime.validity.shape == (2, 1, 57)
    assert [batch.count for batch in runtime.batches] == [16, 16, 16, 9]
    assert len(runtime.tensor_table) == runtime.descriptor_count
    assert all(
        name.startswith(f"projection.{MODULAR_STORAGE_PROJECTION_PREFIX}")
        for name in runtime.tensor_names
    )
    assert np.all(
        runtime.descriptor_specs[:, ALIVE_MASK_TENSOR_INDEX_V1]
        == NULL_TENSOR_INDEX
    )
    assert np.all(
        runtime.descriptor_specs[:, ACTIVE_LENGTH_TENSOR_INDEX_V1]
        == NULL_TENSOR_INDEX
    )
    assert np.all(
        runtime.descriptor_specs[:, CANONICALIZATION_MODE_V1]
        == MODE_CODES[MODE_EXACT]
    )
    validate_phase_e_modular_checkpoint_runtime(runtime)


@pytest.mark.parametrize("case", ["metadata_only", "tensors_only", "mixed"])
def test_partial_or_mixed_projection_presence_fails_closed(case: str) -> None:
    metadata, tensors = _projection()
    if case == "metadata_only":
        tensors = {"energy": tensors["energy"]}
    elif case == "tensors_only":
        metadata.pop(MODULAR_STORAGE_METADATA_KEY)
    else:
        tensors["phase_e_modular_v2__agent_id"] = torch.zeros(
            (3,), dtype=torch.int64
        )

    with pytest.raises(RuntimeError):
        maybe_generate_phase_e_modular_checkpoint_descriptor_overlay(
            metadata, tensors
        )


@pytest.mark.parametrize("case", ["missing", "extra", "plane", "envelope"])
def test_missing_extra_or_tampered_modular_authority_fails_closed(case: str) -> None:
    metadata, tensors = _projection()
    if case == "missing":
        tensors.pop(f"{MODULAR_STORAGE_PROJECTION_PREFIX}agent_id")
    elif case == "extra":
        tensors[f"{MODULAR_STORAGE_PROJECTION_PREFIX}unexpected"] = torch.zeros(
            (3,), dtype=torch.int64
        )
    elif case == "plane":
        tensors[f"{MODULAR_STORAGE_PROJECTION_PREFIX}agent_id"][0] += 1
    else:
        metadata[MODULAR_STORAGE_METADATA_KEY]["envelope_sha256"] = "0" * 64

    with pytest.raises(RuntimeError):
        generate_phase_e_modular_checkpoint_descriptor_overlay(metadata, tensors)


def test_overlay_plan_tamper_and_partial_runtime_composition_fail_closed() -> None:
    metadata, tensors = _projection()
    overlay = generate_phase_e_modular_checkpoint_descriptor_overlay(
        metadata, tensors
    )
    malformed = copy.deepcopy(overlay)
    malformed["descriptors"][0]["fixed_shape"][0] += 1
    with pytest.raises(
        CompactSummaryCheckpointDescriptorPlanError,
        match="does not exactly match",
    ):
        validate_phase_e_modular_checkpoint_descriptor_overlay(
            malformed, metadata, tensors
        )
    with pytest.raises(ValueError, match="without its descriptor overlay"):
        maybe_bind_phase_e_modular_checkpoint_runtime(
            None,
            metadata,
            tensors,
            generation_count=1,
        )
    with pytest.raises(ValueError, match="authority is absent"):
        maybe_bind_phase_e_modular_checkpoint_runtime(
            overlay,
            {"world": "flat"},
            {"energy": tensors["energy"]},
            generation_count=1,
        )


def test_frozen_v1_descriptor_artifact_bytes_remain_exact() -> None:
    payload = FROZEN_V1_PLAN.read_bytes()
    assert len(payload) == 260_532
    assert sha256(payload).hexdigest() == FROZEN_V1_PLAN_SHA256
