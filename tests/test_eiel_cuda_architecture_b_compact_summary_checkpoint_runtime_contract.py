from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT
    / "engine"
    / "perf"
    / "cuda_architecture_b_compact_summary_checkpoint_runtime_v1.hpp"
)


def test_descriptor_counts_are_exact_per_world_with_patch_sized_capacity() -> None:
    source = HEADER.read_text(encoding="utf-8")

    assert "kFlatDescriptorCountV1 = 132" in source
    assert "kPatchP2DescriptorCountV1 = 144" in source
    assert "kDescriptorCapacityV1 = kPatchP2DescriptorCountV1" in source
    assert "kDescriptorCountV1 = kFlatDescriptorCountV1" in source
    assert "expected_descriptor_count_v1(false) == 132" in source
    assert "expected_descriptor_count_v1(true) == 144" in source
    assert "kAgentIdToSlotAgentIdsDescriptorIndexV1 = 8" in source
    assert "kAgentIdToSlotSlotsDescriptorIndexV1 = 9" in source
    assert "descriptor_count == kFlatDescriptorCountV1" in source
    assert "descriptor_count == kPatchP2DescriptorCountV1" in source


def test_pointer_spec_contract_is_frozen_at_thirteen_int64_columns() -> None:
    source = HEADER.read_text(encoding="utf-8")
    assert "eiel_cuda_architecture_b_compact_summary_checkpoint_runtime_v1" in source
    assert "enum DescriptorSpecColumnV1 : std::int64_t" in source
    assert "static_assert(kDescriptorSpecWidthV1 == 13)" in source
    names = re.findall(r"^\s+(k[A-Za-z0-9]+V1)(?:\s*=\s*0)?,?$", source, re.MULTILINE)
    assert names == [
        "kDataTensorIndexV1",
        "kAliveMaskTensorIndexV1",
        "kActiveLengthTensorIndexV1",
        "kCellStrideBytesV1",
        "kBytesPerCellV1",
        "kRowCapacityV1",
        "kRowBytesV1",
        "kAliveMaskCellStrideV1",
        "kDtypeBytesV1",
        "kCanonicalizationModeV1",
        "kRawOutputIndexV1",
        "kCanonicalOutputIndexV1",
        "kDescriptorIndexV1",
        "kDescriptorSpecWidthV1",
    ]
