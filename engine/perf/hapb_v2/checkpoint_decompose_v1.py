"""CPU-side decomposition of HAPB midpoint/checkpoint artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final
import zipfile

from engine.perf.hapb_v2.exactness_contract_v1 import POLICY_WIDTH_V1


DECOMPOSE_SCHEMA_V1: Final[str] = "eiel_hapb_v2_checkpoint_decompose_v1"
CANONICAL_POLICY_BANK_BYTES_V1: Final[int] = 48 * 512 * POLICY_WIDTH_V1 * 4
HOT_PREFIX_CACHE_BYTES_V1: Final[int] = 48 * 256 * POLICY_WIDTH_V1 * 4
CONTENT_MAP_BYTES_V1: Final[int] = 48 * 512 * 8
KNOWN_SIZE_LABELS_V1: Final[dict[int, str]] = {
    CANONICAL_POLICY_BANK_BYTES_V1: "canonical_host_policy_bank_b48_c512_w9991_f32",
    HOT_PREFIX_CACHE_BYTES_V1: "hot_prefix_cache_b48_p256_w9991_f32",
    48 * 512 * POLICY_WIDTH_V1 * 4: "canonical_or_full_slot_policy_bank",
    63_059_968: "per_cell_large_tensor_x48_unclassified",
    2_621_440: "per_cell_2621440_tensor_x48_unclassified",
    1_650_127_656: "torch_data_pkl_embedded_python_state",
}


class HapbV2CheckpointDecomposeError(ValueError):
    """Checkpoint decomposition rejected its input."""


def _label_size(size: int, count: int) -> str:
    known = KNOWN_SIZE_LABELS_V1.get(size)
    if known is not None:
        return known
    if count == 48:
        return f"per_cell_repeated_{size}_bytes"
    if count == 1 and size > 1_000_000_000:
        return "large_singleton_blob"
    return "unclassified_storage"


def decompose_torch_zip_v1(path: str | Path) -> dict[str, object]:
    artifact = Path(path)
    if not artifact.is_file():
        raise HapbV2CheckpointDecomposeError(f"checkpoint path is not a file: {artifact}")
    file_bytes = artifact.stat().st_size
    if not zipfile.is_zipfile(artifact):
        return {
            "schema_version": DECOMPOSE_SCHEMA_V1,
            "path": str(artifact),
            "file_bytes": file_bytes,
            "format": "not_zip_torch_archive",
            "members": 0,
            "attribution": [],
            "limitations": ["file is not a zip-based torch archive"],
        }
    with zipfile.ZipFile(artifact) as archive:
        infos = archive.infolist()
        size_counts = Counter(info.file_size for info in infos)
        uncompressed = sum(info.file_size for info in infos)
        compressed = sum(info.compress_size for info in infos)
        pickle_members = [
            info.filename for info in infos if info.filename.endswith("data.pkl")
        ]
        pickle_bytes = sum(
            info.file_size for info in infos if info.filename.endswith("data.pkl")
        )
    attribution = []
    for size, count in size_counts.most_common():
        if size == 0:
            continue
        attribution.append(
            {
                "member_size_bytes": int(size),
                "member_count": int(count),
                "total_bytes": int(size) * int(count),
                "label": _label_size(int(size), int(count)),
            }
        )
    classified = {
        "canonical_host_policy_bank_bytes": CANONICAL_POLICY_BANK_BYTES_V1,
        "hot_prefix_or_reconstructible_cache_bytes": HOT_PREFIX_CACHE_BYTES_V1,
        "alias_map_int64_bytes": CONTENT_MAP_BYTES_V1,
        "observed_pickle_bytes": int(pickle_bytes),
        "observed_file_bytes": int(file_bytes),
        "observed_uncompressed_bytes": int(uncompressed),
    }
    one_snapshot_overlap = pickle_bytes >= CANONICAL_POLICY_BANK_BYTES_V1
    return {
        "schema_version": DECOMPOSE_SCHEMA_V1,
        "path": str(artifact),
        "file_bytes": int(file_bytes),
        "uncompressed_bytes": int(uncompressed),
        "compressed_bytes": int(compressed),
        "format": "torch_zip",
        "members": len(infos),
        "pickle_members": pickle_members,
        "source_derived_sizes": classified,
        "attribution": attribution,
        "candidate_savings": {
            "exclude_reconstructible_cuda_cache": HOT_PREFIX_CACHE_BYTES_V1,
            "stream_canonical_policy_bank_out_of_pickle": CANONICAL_POLICY_BANK_BYTES_V1
            if one_snapshot_overlap
            else 0,
            "one_snapshot_overlap_likely": bool(one_snapshot_overlap),
        },
        "limitations": [
            "zip inventory does not name tensors; labels are size-class hypotheses",
            "do not assume everything above 0.9 GiB is waste",
            "active Gate-6 checkpoint authority is not modified",
        ],
    }


def streaming_release_plan_v1(report: dict[str, object]) -> dict[str, object]:
    savings = dict(report.get("candidate_savings") or {})
    justified = bool(savings.get("one_snapshot_overlap_likely"))
    return {
        "schema_version": "eiel_hapb_v2_checkpoint_streaming_plan_v1",
        "justified": justified,
        "plan": [
            "keep host-canonical policy bank as the sole authoritative payload",
            "do not serialize reconstructible CUDA hot-prefix or tile arenas",
            "if pickle embeds the bank bytes, store the bank as a raw sidecar",
            "stream one generation snapshot rather than overlapping midpoint copies",
        ]
        if justified
        else [
            "no streaming implementation is justified until named-tensor attribution "
            "confirms overlap"
        ],
        "implementation_safe_now": False,
        "reason": (
            "Gate 6 is still the checkpoint-authority owner; V2 prep only plans"
        ),
    }
