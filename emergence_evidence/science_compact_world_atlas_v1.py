"""Pure campaign-neutral reduction of CUDA V2 compact science artifacts.

The reducer consumes only an immutable compact-capture index and an exact
``EIEL_SCIENCE_WORLD_BUNDLE_V1``.  It preserves logical-world ownership while
summarising lifecycle, action, ecology, reproduction, mutation, recombination,
and declared milestone state.  Test/panel meanings remain opaque configuration
owned by each world; physical batch slots never become scientific identity.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import resource
import signal
import stat
import struct
import tempfile
import time
import traceback
import zlib
from collections import Counter
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Final, Mapping, Sequence

from emergence_contracts import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
    COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1,
    COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1,
    COMPACT_SCIENCE_DEATH_METRIC_ENCODING_V1,
    COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1,
    COMPACT_SCIENCE_DISPATCH_GZIP_ENCODING_V1,
    COMPACT_SCIENCE_DISPATCH_SCHEMA_V1,
    COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_ENCODING_V1,
    COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1,
    COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1,
    ContentDigest,
    SchemaVersion,
    ScienceReducerDescriptorV1,
    ScienceWorldBundleV1,
    canonical_json_bytes,
    canonical_sha256,
    reconstruct_compact_birth_genome_v1,
    tracked_genotype_state_id_v1,
    hereditary_attribution_configuration_v1,
)
from emergence_evidence.native_compact_reducer_v1 import (
    NativeCompactReducerArtifactV1,
    NativeCompactReducerStateV1,
    create_native_compact_reducer_state_v1,
    load_native_compact_reducer_v1,
)
from emergence_evidence.science_strategy_attribution_v1 import (
    StrategyAttributionError,
    render_native_strategy_attribution_v1,
    strategy_attribution_contract_v1,
    strategy_threshold_ticks_v1,
)

COMPACT_WORLD_ATLAS_REDUCER_ID: Final[str] = "science.compact.world_atlas.v1"
COMPACT_WORLD_ATLAS_CONTRACT_VERSION: Final[SchemaVersion] = SchemaVersion(
    "science_compact_world_atlas_v1"
)
COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID: Final[str] = (
    "eiel_science_compact_world_atlas_result_v1"
)
COMPACT_WORLD_ATLAS_CONFIGURATION_SCHEMA: Final[dict[str, Any]] = {
    "schema_version": "eiel_science_compact_world_atlas_config_v1",
    "fields": {
        "analysis_dimensions": "sorted array of opaque dimension names",
        "decision_family": "opaque nonempty study decision token",
        "reporting_group": "opaque nonempty aggregation token",
    },
    "additional_properties": True,
}
COMPACT_WORLD_ATLAS_REQUIRED_CAPABILITIES: Final[tuple[str, ...]] = (
    "compact.agent_lifetime_metrics",
    "compact.birth_provenance",
    "compact.genome_reconstruction",
    "compact.lifecycle_events",
    "compact.state_snapshots",
)
_ASYNC_REDUCER_NICE_INCREMENT_V1: Final[int] = 5
_ASYNC_PROGRESS_FIELD_COUNT_V1: Final[int] = 20
_ASYNC_PROGRESS_PHASES_V1: Final[dict[int, str]] = {
    0: "STARTING",
    1: "FOLLOWING_SEALED_PREFIX",
    2: "TERMINAL_RECONCILIATION",
    3: "COMPLETED",
    4: "FAILED",
    5: "CANCELLED",
}
NATIVE_REDUCER_ARTIFACT_USE_PRODUCTION_V1: Final[str] = "production_routing_v1"
NATIVE_REDUCER_ARTIFACT_USE_ENGINEERING_REAL_USE_V1: Final[str] = (
    "engineering_real_use_v1"
)
NATIVE_REDUCER_ARTIFACT_USE_DEFAULT_TECHNICAL_V1: Final[str] = (
    "default_technical_v1"
)
_NATIVE_REDUCER_ARTIFACT_USE_CLASSES_V1: Final[frozenset[str]] = frozenset(
    {
        NATIVE_REDUCER_ARTIFACT_USE_PRODUCTION_V1,
        NATIVE_REDUCER_ARTIFACT_USE_ENGINEERING_REAL_USE_V1,
        NATIVE_REDUCER_ARTIFACT_USE_DEFAULT_TECHNICAL_V1,
    }
)

HEREDITARY_BIRTH_RECORD_ARTIFACT_SCHEMA_V1: Final[str] = (
    "eiel_science_hereditary_birth_parent_child_records_v1"
)
HEREDITARY_BIRTH_RECORD_STORAGE_ENCODING_V1: Final[str] = (
    "gzip_fixed_binary_v1"
)
_HEREDITARY_BIRTH_RECORD_BUFFER_BYTES_V1: Final[int] = 1024 * 1024
_HEREDITARY_BIRTH_RECORD_INLINE_LIMIT_V1: Final[int] = 4096
_HEREDITARY_BIRTH_RECORD_MAGIC_V1: Final[bytes] = (
    b"EIEL_HEREDITARY_BIRTH_RECORDS_V1\n"
)
_HEREDITARY_BIRTH_RECORD_STRUCT_V1: Final[struct.Struct] = struct.Struct(
    "<HQQQQ" + "32s" * 8 + "QII68s"
)


class CompactWorldAtlasReducerError(ValueError):
    """A sealed compact artifact or its logical ownership is inconsistent."""


class _DigestingWriterV1:
    """Hash compressed bytes without a second read of a large artifact."""

    def __init__(self, raw: Any, digest: Any) -> None:
        self._raw = raw
        self._digest = digest

    def write(self, value: bytes) -> int:
        self._digest.update(value)
        return self._raw.write(value)

    def flush(self) -> None:
        self._raw.flush()

    def fileno(self) -> int:
        return self._raw.fileno()


class _DigestingReaderV1:
    """Hash compressed bytes as a gzip stream is validated."""

    def __init__(self, raw: Any, digest: Any) -> None:
        self._raw = raw
        self._digest = digest

    def read(self, size: int = -1) -> bytes:
        value = self._raw.read(size)
        self._digest.update(value)
        return value

    def read1(self, size: int = -1) -> bytes:
        reader = getattr(self._raw, "read1", self._raw.read)
        value = reader(size)
        self._digest.update(value)
        return value

    def prepend(self, value: bytes) -> None:
        del value

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._raw.tell()


def _hereditary_unsigned_v1(
    value: Any,
    *,
    label: str,
    maximum: int,
    positive: bool = False,
) -> int:
    if type(value) is not int or value < (1 if positive else 0) or value > maximum:
        raise CompactWorldAtlasReducerError(f"hereditary birth {label} is invalid")
    return value


def _hereditary_id_bytes_v1(value: Any, *, label: str, nullable: bool) -> bytes:
    if value is None and nullable:
        return bytes(32)
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise CompactWorldAtlasReducerError(
            f"hereditary birth {label} is not an exact sha256 identity"
        )
    try:
        raw = bytes.fromhex(value[7:])
    except ValueError as exc:
        raise CompactWorldAtlasReducerError(
            f"hereditary birth {label} is not an exact sha256 identity"
        ) from exc
    if nullable and raw == bytes(32):
        raise CompactWorldAtlasReducerError(
            f"hereditary birth {label} uses the reserved null identity"
        )
    return raw


def _hereditary_id_from_bytes_v1(value: bytes, *, nullable: bool) -> str | None:
    if len(value) != 32:
        raise CompactWorldAtlasReducerError(
            "hereditary birth identity field width differs"
        )
    if nullable and value == bytes(32):
        return None
    return f"sha256:{value.hex()}"


def _hereditary_binary_record_v1(
    record: Mapping[str, Any], world_index_by_id: Mapping[str, int]
) -> bytes:
    world_id = record.get("world_id")
    if not isinstance(world_id, str) or world_id not in world_index_by_id:
        raise CompactWorldAtlasReducerError(
            "hereditary birth record has an unowned logical world identity"
        )
    parent_agent_ids = record.get("parent_agent_ids")
    parent_agent_state_ids = record.get("parent_agent_state_ids")
    parent_full_ids = record.get("parent_full_genome_state_ids")
    parent_tracked_ids = record.get("parent_tracked_genotype_state_ids")
    for value, label in (
        (parent_agent_ids, "parent agent identities"),
        (parent_agent_state_ids, "parent agent-state identities"),
        (parent_full_ids, "parent full-genome identities"),
        (parent_tracked_ids, "parent tracked identities"),
    ):
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or len(value) != 2
        ):
            raise CompactWorldAtlasReducerError(
                f"hereditary birth {label} are invalid"
            )
    parent_agents = [
        _hereditary_unsigned_v1(
            value,
            label="parent agent identity",
            maximum=(1 << 64) - 1,
            positive=True,
        )
        for value in parent_agent_ids
    ]
    parent_states = [
        _hereditary_id_bytes_v1(
            value,
            label="parent agent-state identity",
            nullable=True,
        )
        for value in parent_agent_state_ids
    ]
    parent_full = [
        _hereditary_id_bytes_v1(
            value,
            label="parent full-genome identity",
            nullable=False,
        )
        for value in parent_full_ids
    ]
    parent_tracked = [
        _hereditary_id_bytes_v1(
            value,
            label="parent tracked identity",
            nullable=False,
        )
        for value in parent_tracked_ids
    ]
    child_full = _hereditary_id_bytes_v1(
        record.get("child_full_genome_state_id"),
        label="child full-genome identity",
        nullable=False,
    )
    child_tracked = _hereditary_id_bytes_v1(
        record.get("child_tracked_genotype_state_id"),
        label="child tracked identity",
        nullable=False,
    )
    birth_provenance_hex = record.get("birth_provenance_hex")
    if (
        not isinstance(birth_provenance_hex, str)
            or len(birth_provenance_hex)
            != COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1 * 2
        or any(
            character not in "0123456789abcdef"
            for character in birth_provenance_hex
        )
    ):
        raise CompactWorldAtlasReducerError(
            "hereditary birth provenance encoding is invalid"
        )
    birth_provenance = bytes.fromhex(birth_provenance_hex)
    try:
        return _HEREDITARY_BIRTH_RECORD_STRUCT_V1.pack(
            _hereditary_unsigned_v1(
                world_index_by_id[world_id],
                label="world index",
                maximum=0xFFFF,
            ),
            _hereditary_unsigned_v1(
                record.get("absolute_tick"),
                label="absolute tick",
                maximum=(1 << 64) - 1,
            ),
            _hereditary_unsigned_v1(
                record.get("child_agent_id"),
                label="child agent identity",
                maximum=(1 << 64) - 1,
                positive=True,
            ),
            parent_agents[0],
            parent_agents[1],
            parent_states[0],
            parent_states[1],
            parent_full[0],
            parent_full[1],
            child_full,
            parent_tracked[0],
            parent_tracked[1],
            child_tracked,
            _hereditary_unsigned_v1(
                record.get("lineage_depth"),
                label="lineage depth",
                maximum=(1 << 64) - 1,
            ),
            _hereditary_unsigned_v1(
                record.get("mutation_origin_count"),
                label="mutation origin count",
                maximum=0xFFFFFFFF,
            ),
            _hereditary_unsigned_v1(
                record.get("recombination_pattern"),
                label="recombination pattern",
                maximum=0xFFFFFFFF,
            ),
            birth_provenance,
        )
    except struct.error as exc:
        raise CompactWorldAtlasReducerError(
            "hereditary birth record exceeds binary ABI"
        ) from exc


def _decode_hereditary_binary_record_v1(
    encoded: bytes, world_ids: Sequence[str]
) -> dict[str, Any]:
    if len(encoded) != _HEREDITARY_BIRTH_RECORD_STRUCT_V1.size:
        raise CompactWorldAtlasReducerError(
            "hereditary birth record has the wrong binary width"
        )
    try:
        values = _HEREDITARY_BIRTH_RECORD_STRUCT_V1.unpack(encoded)
    except struct.error as exc:
        raise CompactWorldAtlasReducerError(
            "hereditary birth record binary decoding failed"
        ) from exc
    world_index = values[0]
    if world_index >= len(world_ids):
        raise CompactWorldAtlasReducerError(
            "hereditary birth record world index is unowned"
        )
    parent_state_a = _hereditary_id_from_bytes_v1(values[5], nullable=True)
    parent_state_b = _hereditary_id_from_bytes_v1(values[6], nullable=True)
    parent_full_a = _hereditary_id_from_bytes_v1(values[7], nullable=False)
    parent_full_b = _hereditary_id_from_bytes_v1(values[8], nullable=False)
    child_full = _hereditary_id_from_bytes_v1(values[9], nullable=False)
    parent_tracked_a = _hereditary_id_from_bytes_v1(values[10], nullable=False)
    parent_tracked_b = _hereditary_id_from_bytes_v1(values[11], nullable=False)
    child_tracked = _hereditary_id_from_bytes_v1(values[12], nullable=False)
    assert (
        parent_full_a is not None
        and parent_full_b is not None
        and child_full is not None
        and parent_tracked_a is not None
        and parent_tracked_b is not None
        and child_tracked is not None
    )
    return {
        "absolute_tick": values[1],
        "world_id": world_ids[world_index],
        "child_agent_id": values[2],
        "parent_agent_ids": [values[3], values[4]],
        "parent_agent_state_ids": [parent_state_a, parent_state_b],
        "parent_full_genome_state_ids": [parent_full_a, parent_full_b],
        "child_full_genome_state_id": child_full,
        "parent_tracked_genotype_state_ids": [parent_tracked_a, parent_tracked_b],
        "child_tracked_genotype_state_id": child_tracked,
        "lineage_depth": values[13],
        "mutation_origin_count": values[14],
        "recombination_pattern": values[15],
        "birth_provenance_hex": values[16].hex(),
    }


class _HereditaryBirthRecordSpoolV1:
    """Bounded-memory, immutable-at-seal storage for exact birth joins.

    A full-hereditary result must retain every parent/child identity and birth
    provenance row. Keeping those rows as Python dictionaries is not viable
    for a large physical batch. This spool keeps only a small write buffer in
    memory and exposes a content-addressed artifact reference in the
    canonical result. Records use a fixed-width binary ABI so large birth
    streams avoid per-record JSON encoding/allocation costs; decoding restores
    the exact public record shape for small compatibility results.
    """

    def __init__(self, path: Path, *, world_ids: Sequence[str]) -> None:
        self.path = path.expanduser().resolve()
        self.world_ids = tuple(world_ids)
        if not self.world_ids or len(set(self.world_ids)) != len(self.world_ids):
            raise CompactWorldAtlasReducerError(
                "hereditary birth artifact world ownership is invalid"
            )
        if self.path.is_symlink() or self.path.exists():
            raise CompactWorldAtlasReducerError(
                f"hereditary birth artifact path is not fresh: {self.path}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.parent.is_symlink():
            raise CompactWorldAtlasReducerError(
                "hereditary birth artifact parent is a symbolic link"
            )
        self._world_index_by_id = {
            world_id: index for index, world_id in enumerate(self.world_ids)
        }
        self.stage_path = self.path.with_name(f".{self.path.name}.stage")
        if self.stage_path.is_symlink() or self.stage_path.exists():
            raise CompactWorldAtlasReducerError(
                "hereditary birth artifact staging path is not fresh"
            )
        try:
            descriptor = os.open(
                self.stage_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            self._raw = os.fdopen(descriptor, "wb")
        except OSError as exc:
            raise CompactWorldAtlasReducerError(
                f"cannot create hereditary birth artifact: {self.stage_path}"
            ) from exc
        self._digest = hashlib.sha256()
        self._writer = _DigestingWriterV1(self._raw, self._digest)
        self._gzip = gzip.GzipFile(
            fileobj=self._writer, mode="wb", compresslevel=1, mtime=0
        )
        self._gzip.write(_HEREDITARY_BIRTH_RECORD_MAGIC_V1)
        self._buffer = bytearray()
        self.record_count = 0
        self._world_record_counts: dict[str, int] = {
            world_id: 0 for world_id in self.world_ids
        }
        self._sealed = False
        self._aborted = False
        self._artifact: dict[str, Any] | None = None

    def append(self, record: Mapping[str, Any]) -> None:
        if self._sealed or self._aborted:
            raise CompactWorldAtlasReducerError(
                "hereditary birth artifact is no longer writable"
            )
        record_world_id = record.get("world_id")
        encoded = _hereditary_binary_record_v1(record, self._world_index_by_id)
        self._buffer.extend(encoded)
        self.record_count += 1
        self._world_record_counts[record_world_id] = (
            self._world_record_counts.get(record_world_id, 0) + 1
        )
        if len(self._buffer) >= _HEREDITARY_BIRTH_RECORD_BUFFER_BYTES_V1:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if self._buffer:
            self._gzip.write(self._buffer)
            self._buffer.clear()

    def seal(self) -> dict[str, Any]:
        if self._aborted:
            raise CompactWorldAtlasReducerError(
                "cannot seal an aborted hereditary birth artifact"
            )
        if self._sealed:
            assert self._artifact is not None
            return dict(self._artifact)
        try:
            self._flush_buffer()
            self._gzip.close()
            self._raw.flush()
            os.fsync(self._raw.fileno())
            self._raw.close()
            os.chmod(self.stage_path, 0o444)
            if self.path.exists() or self.path.is_symlink():
                raise CompactWorldAtlasReducerError(
                    f"hereditary birth artifact appeared before publication: {self.path}"
                )
            # Hard-link publication is atomic and refuses to overwrite an
            # artifact that another writer may have created.
            os.link(self.stage_path, self.path)
            self.stage_path.unlink()
        except CompactWorldAtlasReducerError:
            self.abort()
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            self.abort()
            raise CompactWorldAtlasReducerError(
                f"cannot seal hereditary birth artifact: {self.path}"
            ) from exc
        self._sealed = True
        self._artifact = {
            "schema_version": HEREDITARY_BIRTH_RECORD_ARTIFACT_SCHEMA_V1,
            "storage_encoding": HEREDITARY_BIRTH_RECORD_STORAGE_ENCODING_V1,
            "world_ids": list(self.world_ids),
            "path": str(self.path),
            "file_sha256": self._digest.hexdigest(),
            "bytes": self.path.stat().st_size,
            "record_count": self.record_count,
            "world_record_counts": dict(sorted(self._world_record_counts.items())),
            "record_width": _HEREDITARY_BIRTH_RECORD_STRUCT_V1.size,
            "header": _HEREDITARY_BIRTH_RECORD_MAGIC_V1.decode("ascii").rstrip(
                "\n"
            ),
            "immutable": True,
        }
        return dict(self._artifact)

    def records_for_world(self, world_id: str) -> list[dict[str, Any]]:
        """Retain the legacy inline shape for small bounded results only."""

        if not self._sealed or self._artifact is None:
            raise CompactWorldAtlasReducerError(
                "hereditary birth artifact must be sealed before reading"
            )
        if self.record_count > _HEREDITARY_BIRTH_RECORD_INLINE_LIMIT_V1:
            raise CompactWorldAtlasReducerError(
                "hereditary birth artifact exceeds inline result limit"
            )
        if world_id not in self.world_ids:
            raise CompactWorldAtlasReducerError(
                "requested hereditary birth artifact world is unowned"
            )
        records: list[dict[str, Any]] = []
        observed_total = 0
        try:
            with gzip.open(self.path, mode="rb") as compressed:
                if compressed.read(len(_HEREDITARY_BIRTH_RECORD_MAGIC_V1)) != (
                    _HEREDITARY_BIRTH_RECORD_MAGIC_V1
                ):
                    raise CompactWorldAtlasReducerError(
                        "hereditary birth artifact header differs"
                    )
                while True:
                    encoded = compressed.read(_HEREDITARY_BIRTH_RECORD_STRUCT_V1.size)
                    if not encoded:
                        break
                    if len(encoded) != _HEREDITARY_BIRTH_RECORD_STRUCT_V1.size:
                        raise CompactWorldAtlasReducerError(
                            "hereditary birth artifact record is truncated"
                        )
                    record = _decode_hereditary_binary_record_v1(
                        encoded, self.world_ids
                    )
                    observed_total += 1
                    if record.get("world_id") == world_id:
                        records.append(dict(record))
        except (
            OSError,
            EOFError,
            UnicodeError,
            gzip.BadGzipFile,
            struct.error,
            zlib.error,
        ) as exc:
            raise CompactWorldAtlasReducerError(
                "hereditary birth artifact cannot be read"
            ) from exc
        if observed_total != self.record_count:
            raise CompactWorldAtlasReducerError(
                "hereditary birth artifact record count differs while reading"
            )
        return records

    def abort(self) -> None:
        if self._sealed or self._aborted:
            return
        self._aborted = True
        try:
            self._gzip.close()
        except (OSError, ValueError):
            pass
        try:
            self._raw.close()
        except (OSError, ValueError):
            pass
        try:
            self.stage_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def validate_hereditary_birth_record_artifact_v1(
    artifact: Mapping[str, Any],
    *,
    expected_world_id: str | None = None,
) -> dict[str, Any]:
    """Authenticate and stream-validate one exact birth-record artifact."""

    if artifact.get("schema_version") != HEREDITARY_BIRTH_RECORD_ARTIFACT_SCHEMA_V1:
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact schema is invalid"
        )
    if artifact.get("storage_encoding") != HEREDITARY_BIRTH_RECORD_STORAGE_ENCODING_V1:
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact storage encoding is invalid"
        )
    if artifact.get("record_width") != _HEREDITARY_BIRTH_RECORD_STRUCT_V1.size:
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact record width is invalid"
        )
    if artifact.get("header") != _HEREDITARY_BIRTH_RECORD_MAGIC_V1.decode(
        "ascii"
    ).rstrip("\n"):
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact header is invalid"
        )
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact path is invalid"
        )
    path = Path(raw_path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact is not a regular file"
        )
    expected_digest = artifact.get("file_sha256")
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact digest is invalid"
        )
    expected_total = artifact.get("record_count")
    if type(expected_total) is not int or expected_total < 0:
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact record count is invalid"
        )
    raw_world_ids = artifact.get("world_ids")
    raw_world_counts = artifact.get("world_record_counts")
    if (
        not isinstance(raw_world_ids, list)
        or any(not isinstance(value, str) or not value for value in raw_world_ids)
        or len(raw_world_ids) != len(set(raw_world_ids))
        or not isinstance(raw_world_counts, Mapping)
        or set(raw_world_counts) != set(raw_world_ids)
        or any(type(value) is not int or value < 0 for value in raw_world_counts.values())
    ):
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact world counts are invalid"
        )
    if expected_world_id is not None and expected_world_id not in raw_world_ids:
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact does not own the requested world"
        )
    pointer_world_id = artifact.get("world_id")
    if pointer_world_id is not None and pointer_world_id not in raw_world_ids:
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact pointer world is not owned"
        )
    pointer_world_count = artifact.get("world_record_count")
    if pointer_world_count is not None:
        if (
            type(pointer_world_count) is not int
            or pointer_world_id is None
            or pointer_world_count != raw_world_counts[pointer_world_id]
        ):
            raise CompactWorldAtlasReducerError(
                "hereditary birth artifact pointer count is invalid"
            )
    if artifact.get("immutable") is not True:
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact is not immutable"
        )

    digest = hashlib.sha256()
    counts = {world_id: 0 for world_id in raw_world_ids}
    total = 0
    try:
        with path.open("rb") as raw:
            reader = _DigestingReaderV1(raw, digest)
            with gzip.GzipFile(fileobj=reader, mode="rb") as compressed:
                if compressed.read(len(_HEREDITARY_BIRTH_RECORD_MAGIC_V1)) != (
                    _HEREDITARY_BIRTH_RECORD_MAGIC_V1
                ):
                    raise CompactWorldAtlasReducerError(
                        "hereditary birth artifact header differs"
                    )
                while True:
                    encoded = compressed.read(_HEREDITARY_BIRTH_RECORD_STRUCT_V1.size)
                    if not encoded:
                        break
                    if len(encoded) != _HEREDITARY_BIRTH_RECORD_STRUCT_V1.size:
                        raise CompactWorldAtlasReducerError(
                            "hereditary birth artifact record is truncated"
                        )
                    record = _decode_hereditary_binary_record_v1(
                        encoded, raw_world_ids
                    )
                    world_id = record.get("world_id")
                    if not isinstance(world_id, str) or world_id not in counts:
                        raise CompactWorldAtlasReducerError(
                            "hereditary birth artifact record world is unowned"
                        )
                    counts[world_id] += 1
                    total += 1
    except (
        OSError,
        EOFError,
        UnicodeError,
        gzip.BadGzipFile,
        struct.error,
        zlib.error,
    ) as exc:
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact gzip stream is invalid"
        ) from exc
    if digest.hexdigest() != expected_digest:
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact digest differs"
        )
    if artifact.get("bytes") != path.stat().st_size:
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact byte count differs"
        )
    if total != expected_total or counts != dict(raw_world_counts):
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact record counts differ"
        )
    return {
        "record_count": total,
        "world_record_counts": counts,
        "file_sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def _records_for_hereditary_birth_artifact_world_v1(
    artifact: Mapping[str, Any], world_id: str
) -> list[dict[str, Any]]:
    """Decode one bounded inline view from a sealed binary birth artifact."""

    validate_hereditary_birth_record_artifact_v1(
        artifact,
        expected_world_id=world_id,
    )
    raw_path = artifact.get("path")
    raw_world_ids = artifact.get("world_ids")
    if not isinstance(raw_path, str) or not isinstance(raw_world_ids, list):
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact inline source is invalid"
        )
    records: list[dict[str, Any]] = []
    try:
        with gzip.open(Path(raw_path).expanduser(), mode="rb") as compressed:
            if compressed.read(len(_HEREDITARY_BIRTH_RECORD_MAGIC_V1)) != (
                _HEREDITARY_BIRTH_RECORD_MAGIC_V1
            ):
                raise CompactWorldAtlasReducerError(
                    "hereditary birth artifact header differs"
                )
            while True:
                encoded = compressed.read(_HEREDITARY_BIRTH_RECORD_STRUCT_V1.size)
                if not encoded:
                    break
                if len(encoded) != _HEREDITARY_BIRTH_RECORD_STRUCT_V1.size:
                    raise CompactWorldAtlasReducerError(
                        "hereditary birth artifact record is truncated"
                    )
                record = _decode_hereditary_binary_record_v1(encoded, raw_world_ids)
                if record["world_id"] == world_id:
                    records.append(record)
    except (OSError, EOFError, gzip.BadGzipFile, struct.error, zlib.error) as exc:
        raise CompactWorldAtlasReducerError(
            "hereditary birth artifact inline source cannot be read"
        ) from exc
    return records


_POLICY_LOCI_PER_HAPLOTYPE: Final[int] = 112
_TRAIT_LOCI_PER_HAPLOTYPE: Final[int] = 16
_POLICY_VALUE_COUNT: Final[int] = 224
_TRAIT_VALUE_COUNT: Final[int] = 32
_GENOME_ROW_STRUCT: Final[struct.Struct] = struct.Struct("<224b32h")
_EVENT_ROW_STRUCT: Final[struct.Struct] = struct.Struct("<QQQQ")


@dataclass(slots=True)
class _ActiveGenomeV1:
    encoded: bytes
    generation: int
    mutation_lineage: bool
    born: bool
    reproduced: bool = False
    agent_state_id: str | None = None
    lineage_id: str | None = None
    full_genome_state_id: str | None = None
    tracked_genotype_state_id: str | None = None
    parent_a_id: int = 0
    parent_b_id: int = 0
    birth_tick: int = 0
    latest_metrics: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class _DispatchWorldWorkV1:
    """One logical world's slice of one already-verified dispatch."""

    cell_id: str
    first: int
    stop: int
    birth_base: int
    death_base: int
    birth_count: int
    death_count: int


@dataclass(frozen=True, slots=True)
class _DispatchWorkFrameV1:
    """Dispatch metadata delivered to one reducer worker."""

    start: int
    end: int
    tick_count: int
    worlds: tuple[_DispatchWorldWorkV1, ...]


@dataclass(frozen=True, slots=True)
class _PreparedDispatchV1:
    """A dispatch whose manifests and payload hashes were checked once."""

    work: _DispatchWorkFrameV1
    events: bytes
    metrics: bytes
    provenance: bytes
    world_offsets: tuple[int, ...]

    def for_indices(self, indices: Sequence[int]) -> _DispatchWorkFrameV1:
        return _DispatchWorkFrameV1(
            start=self.work.start,
            end=self.work.end,
            tick_count=self.work.tick_count,
            worlds=tuple(self.work.worlds[index] for index in indices),
        )


@dataclass(frozen=True, slots=True)
class _PreparedSnapshotV1:
    """One authenticated and decompressed canonical snapshot payload."""

    manifest: Mapping[str, Any]
    uncompressed: bytes
    absolute_tick: int
    row_count: int


def _encode_genome(value: Mapping[str, Any], *, label: str) -> bytes:
    policy = value.get("policy")
    trait = value.get("trait")
    if (
        isinstance(policy, (str, bytes))
        or not isinstance(policy, Sequence)
        or len(policy) != _POLICY_VALUE_COUNT
        or isinstance(trait, (str, bytes))
        or not isinstance(trait, Sequence)
        or len(trait) != _TRAIT_VALUE_COUNT
    ):
        raise CompactWorldAtlasReducerError(f"{label} genome shape differs")
    try:
        return _GENOME_ROW_STRUCT.pack(
            *(int(item) for item in policy),
            *(int(item) for item in trait),
        )
    except (ValueError, TypeError, struct.error) as exc:
        raise CompactWorldAtlasReducerError(
            f"{label} genome values exceed native storage"
        ) from exc


def _decode_genome(value: bytes) -> dict[str, list[int]]:
    if len(value) != _GENOME_ROW_STRUCT.size:
        raise CompactWorldAtlasReducerError("compact genome row width differs")
    row = _GENOME_ROW_STRUCT.unpack(value)
    return {
        "policy": list(row[:_POLICY_VALUE_COUNT]),
        "trait": list(row[_POLICY_VALUE_COUNT:]),
    }


def _genome_state_id(value: bytes) -> str:
    policy = value[:_POLICY_VALUE_COUNT]
    trait = value[_POLICY_VALUE_COUNT:]
    digest = canonical_sha256({"policy": policy.hex(), "trait": trait.hex()})
    return f"sha256:{digest.hex}"


def _tracked_genome_state_id(
    value: bytes,
    *,
    policy_loci: Sequence[int],
    trait_loci: Sequence[int],
) -> str:
    """Use the same tracked projection as the native compact carrier."""

    try:
        return tracked_genotype_state_id_v1(
            value,
            policy_loci=policy_loci,
            trait_loci=trait_loci,
        )
    except Exception as exc:
        raise CompactWorldAtlasReducerError(
            f"tracked hereditary identity cannot be derived: {exc}"
        ) from exc


def _tracked_loci(world: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    configuration: Mapping[str, Any] = {}
    for binding in world.reducers.to_value():
        if binding.get("reducer_id") == COMPACT_WORLD_ATLAS_REDUCER_ID:
            candidate = binding.get("configuration")
            if isinstance(candidate, Mapping):
                configuration = candidate
            break

    def parse(name: str, limit: int) -> tuple[int, ...]:
        raw = configuration.get(name, [])
        if not isinstance(raw, list) or any(
            type(item) is not int or not 0 <= item < limit for item in raw
        ):
            raise CompactWorldAtlasReducerError(
                f"reducer configuration {name} is invalid"
            )
        result = tuple(raw)
        if result != tuple(sorted(set(result))):
            raise CompactWorldAtlasReducerError(
                f"reducer configuration {name} must be sorted and unique"
            )
        return result

    return (
        parse("tracked_policy_loci", _POLICY_LOCI_PER_HAPLOTYPE),
        parse("tracked_trait_loci", _TRAIT_LOCI_PER_HAPLOTYPE),
    )


def _control_value_at_tick(
    world: Any, control_id: str, absolute_tick: int, *, default: object
) -> object:
    semantic = (
        world
        if isinstance(world, Mapping)
        else world.semantic_inputs.to_value()
    )
    row = next(
        (
            value
            for value in semantic["control_bindings"]
            if value.get("control_id") == control_id
        ),
        None,
    )
    if row is None:
        return default
    value = row["resolved_value"]
    schedule = row["schedule"]
    kind = schedule["kind"]
    if kind == "static":
        return value
    if kind == "piecewise_constant_absolute_ticks":
        for segment in schedule["segments"]:
            if segment["start_absolute_tick"] <= absolute_tick:
                value = segment["value"]
            else:
                break
        return value
    if kind == "switch_absolute_tick":
        if type(value) is not bool:
            raise CompactWorldAtlasReducerError(
                f"{control_id} uses a non-boolean switch"
            )
        return not value if absolute_tick >= schedule["absolute_tick"] else value
    if kind == "pulse_absolute_ticks":
        start = schedule["start_absolute_tick"]
        return (
            value
            if start <= absolute_tick < start + schedule["duration_ticks"]
            else default
        )
    raise CompactWorldAtlasReducerError(
        f"{control_id} uses an unsupported native schedule"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CompactWorldAtlasReducerError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompactWorldAtlasReducerError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise CompactWorldAtlasReducerError(f"{label} must be an object")
    return value


def _bound_file(
    root: Path, name: object, expected_sha256: object, *, label: str
) -> bytes:
    if not isinstance(name, str) or Path(name).name != name:
        raise CompactWorldAtlasReducerError(f"{label} file name is invalid")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise CompactWorldAtlasReducerError(f"{label} file is absent")
    value = path.read_bytes()
    if not isinstance(expected_sha256, str) or _sha256_bytes(value) != expected_sha256:
        raise CompactWorldAtlasReducerError(f"{label} SHA-256 differs")
    return value


def _bound_dispatch_payload_v1(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    prefix: str,
    label: str,
) -> bytes:
    """Open either the original identity payload or deterministic gzip V1."""

    stored = _bound_file(
        root,
        manifest.get(f"{prefix}_file"),
        manifest.get(f"{prefix}_file_sha256"),
        label=label,
    )
    encoding = manifest.get("dispatch_payload_storage_encoding")
    if encoding is None:
        return stored
    if encoding != COMPACT_SCIENCE_DISPATCH_GZIP_ENCODING_V1:
        raise CompactWorldAtlasReducerError(
            "compact dispatch payload storage encoding differs"
        )
    if not str(manifest.get(f"{prefix}_file", "")).endswith(".gz") or manifest.get(
        f"{prefix}_stored_bytes"
    ) != len(stored):
        raise CompactWorldAtlasReducerError(f"{label} stored identity differs")
    try:
        value = gzip.decompress(stored)
    except (OSError, EOFError, zlib.error) as exc:
        raise CompactWorldAtlasReducerError(
            f"{label} compressed payload is invalid"
        ) from exc
    expected_uncompressed = manifest.get(f"{prefix}_uncompressed_sha256")
    if (
        not isinstance(expected_uncompressed, str)
        or _sha256_bytes(value) != expected_uncompressed
    ):
        raise CompactWorldAtlasReducerError(f"{label} uncompressed SHA-256 differs")
    return value


def _artifact_manifest(
    pointer: Mapping[str, Any], *, label: str
) -> tuple[Path, dict[str, Any]]:
    raw_path = pointer.get("path")
    if not isinstance(raw_path, str):
        raise CompactWorldAtlasReducerError(f"{label} path is invalid")
    root = Path(raw_path).expanduser()
    if root.is_symlink():
        raise CompactWorldAtlasReducerError(f"{label} root is a symbolic link")
    root = root.resolve()
    manifest = _read_json(root / "manifest.json", label=f"{label} manifest")
    expected = pointer.get("artifact_sha256")
    if manifest.get("artifact_sha256") != expected:
        raise CompactWorldAtlasReducerError(f"{label} pointer identity differs")
    body = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    if canonical_sha256(body).hex != expected:
        raise CompactWorldAtlasReducerError(f"{label} manifest digest differs")
    return root, manifest


def compact_world_atlas_reducer_descriptor_v1() -> ScienceReducerDescriptorV1:
    source = Path(__file__).resolve()
    return ScienceReducerDescriptorV1(
        reducer_id=COMPACT_WORLD_ATLAS_REDUCER_ID,
        contract_version=COMPACT_WORLD_ATLAS_CONTRACT_VERSION,
        implementation_sha256=ContentDigest(
            "sha256", _sha256_bytes(source.read_bytes())
        ),
        configuration_schema_sha256=canonical_sha256(
            COMPACT_WORLD_ATLAS_CONFIGURATION_SCHEMA
        ),
        required_observation_capability_ids=(COMPACT_WORLD_ATLAS_REQUIRED_CAPABILITIES),
        output_contract_id=COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID,
        purity_declarations=(
            "CONTENT_ADDRESSED_IMPLEMENTATION",
            "DETERMINISTIC_CANONICAL_OUTPUT",
            "NO_NETWORK",
            "NO_SIMULATOR_IMPORT",
            "NO_TRAJECTORY_MUTATION",
            "OFFLINE_ONLY",
            "SEALED_ARTIFACT_INPUT_ONLY",
            "TERMINAL_INDEX_RECONCILIATION_REQUIRED",
        ),
    )


def _blank_world(world: Any) -> dict[str, Any]:
    science = world.science.to_value()
    semantic = world.semantic_inputs.to_value()
    return {
        "world_id": world.world_id,
        "science": science,
        "dimension_bindings": semantic["dimension_bindings"],
        "resident_capacity": int(semantic["world"]["resident_capacity"]),
        "semantic_world_sha256": world.semantic_world_sha256.to_dict(),
        "acquisition_sha256": world.acquisition_sha256.to_dict(),
        "birth_count": 0,
        "death_count": 0,
        "reconstructed_birth_count": 0,
        "mutation_origin_count": 0,
        "mutation_saturated_noop_count": 0,
        "mutation_origin_counts_by_linked_locus": [0] * 128,
        "mutation_delta_totals_by_linked_locus": [0] * 128,
        "recombination_birth_count": 0,
        "recombination_nonzero_pattern_count": 0,
        "recombination_choice_one_counts_by_block": [[0, 0] for _ in range(16)],
        "carrier_reproduction_birth_count": 0,
        "mutation_carrier_reproduction_birth_count": 0,
        "unique_born_agents_reproduced_count": 0,
        "unique_mutation_carriers_reproduced_count": 0,
        "maximum_lineage_depth": 0,
        "completed_lifetime_metric_totals": {
            name: 0 for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
        },
        "milestones": [],
        "integrity": {
            "dispatch_count": 0,
            "event_count": 0,
            "snapshot_count": 0,
            "dropped_record_count": 0,
            "dispatch_tick_count": 0,
            "genome_state_identity_checks": 0,
            "hereditary_full_identity_checks": 0,
            "hereditary_parent_child_records": 0,
            "hereditary_funnel_reconciled": False,
        },
    }


def _blank_reconstruction(world: Any) -> dict[str, Any]:
    semantic = world.semantic_inputs.to_value()
    time = semantic["time"]
    start = int(time.get("start_next_tick", 0))
    policy_loci, trait_loci = _tracked_loci(world)
    try:
        strategy_contract = strategy_attribution_contract_v1(world)
    except StrategyAttributionError as exc:
        raise CompactWorldAtlasReducerError(str(exc)) from exc
    reducer_configuration: Mapping[str, Any] = {}
    for reducer in world.reducers.to_value():
        if reducer.get("reducer_id") == COMPACT_WORLD_ATLAS_REDUCER_ID:
            candidate = reducer.get("configuration")
            if isinstance(candidate, Mapping):
                reducer_configuration = candidate
    hereditary_contract = hereditary_attribution_configuration_v1(
        reducer_configuration
    )
    focal_ids: tuple[str, ...] = ()
    if isinstance(hereditary_contract, Mapping):
        raw_focal_ids = hereditary_contract.get(
            "focal_tracked_genotype_state_ids", ()
        )
        if isinstance(raw_focal_ids, list):
            focal_ids = tuple(raw_focal_ids)
    return {
        "world": world,
        "semantic_inputs": semantic,
        "start_next_tick": start,
        "next_tick": start,
        "active": {},
        "snapshots": {},
        "milestones": {},
        "policy_loci": policy_loci,
        "trait_loci": trait_loci,
        "strategy_contract": strategy_contract,
        "hereditary_contract": hereditary_contract,
        "hereditary_focal_ids": focal_ids,
        "hereditary_birth_spool": None,
        "hereditary_descendant_counts": {},
        "hereditary_metric_totals": {},
        "hereditary_metric_agent_counts": {},
        "hereditary_completed_agents": 0,
        "strategy_snapshots": {},
        "strategy_initial_masks": (),
        "initialized": False,
        "mutation_step_static": (
            next(
                (
                    value["resolved_value"]
                    for value in semantic["control_bindings"]
                    if value.get("control_id") == "mutation.step"
                    and value.get("schedule", {}).get("kind") == "static"
                ),
                None,
            )
        ),
    }


def _configure_hereditary_birth_spool_v1(
    reconstructions: Mapping[str, dict[str, Any]],
    *,
    world_ids: Sequence[str],
    output_root: str | Path | None,
    artifact_suffix: str = "",
) -> None:
    """Attach one bounded-memory birth artifact to a physical reduction.

    One shared stream avoids opening one file per logical world while retaining
    a per-world filtered reference in each world result.  The stream is
    created only for the opt-in full-hereditary contract.
    """

    hereditary_world_ids = tuple(
        world_id
        for world_id in world_ids
        if reconstructions[world_id]["hereditary_contract"] is not None
    )
    if not hereditary_world_ids:
        return
    if output_root is None:
        root = Path(tempfile.mkdtemp(prefix="eiel-hereditary-reducer-"))
    else:
        root = Path(output_root).expanduser()
        if root.is_symlink():
            raise CompactWorldAtlasReducerError(
                "hereditary reducer output root is a symbolic link"
            )
        if root.exists() and root.is_file():
            raise CompactWorldAtlasReducerError(
                "hereditary reducer output root is a regular file"
            )
        root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise CompactWorldAtlasReducerError(
            "hereditary reducer output root is a symbolic link"
        )
    artifact_root = root / "reducer-artifacts" / "hereditary-attribution"
    spool = _HereditaryBirthRecordSpoolV1(
        artifact_root / f"birth-parent-child-records{artifact_suffix}.bin.gz",
        world_ids=hereditary_world_ids,
    )
    for world_id in hereditary_world_ids:
        reconstructions[world_id]["hereditary_birth_spool"] = spool


def _configure_native_hereditary_birth_output_v1(
    reconstructions: Mapping[str, dict[str, Any]],
    *,
    world_ids: Sequence[str],
    output_root: str | Path | None,
) -> None:
    """Reserve one exact native birth stream per full-identity world.

    Native reduction owns the stream writer so Python never materializes one
    birth dictionary or genome join at a time.  The paths are still ordinary
    reducer artifacts and are sealed into the same result contract.
    """

    hereditary_world_ids = tuple(
        world_id
        for world_id in world_ids
        if reconstructions[world_id]["hereditary_contract"] is not None
    )
    if not hereditary_world_ids:
        return
    if output_root is None:
        root = Path(tempfile.mkdtemp(prefix="eiel-native-hereditary-reducer-"))
    else:
        root = Path(output_root).expanduser()
        if root.is_symlink():
            raise CompactWorldAtlasReducerError(
                "native hereditary reducer output root is a symbolic link"
            )
        if root.exists() and root.is_file():
            raise CompactWorldAtlasReducerError(
                "native hereditary reducer output root is a regular file"
            )
        root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise CompactWorldAtlasReducerError(
            "native hereditary reducer output root is a symbolic link"
        )
    artifact_root = root / "reducer-artifacts" / "hereditary-attribution"
    artifact_root.mkdir(parents=True, exist_ok=True)
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise CompactWorldAtlasReducerError(
            "native hereditary reducer artifact root is invalid"
        )
    world_index_by_id = {world_id: index for index, world_id in enumerate(world_ids)}
    for world_id in hereditary_world_ids:
        world_index = world_index_by_id[world_id]
        path = artifact_root / (
            f"birth-parent-child-records-native-world-{world_index:04d}.bin.gz"
        )
        if path.is_symlink() or path.exists():
            raise CompactWorldAtlasReducerError(
                f"native hereditary birth artifact path is not fresh: {path}"
            )
        reconstructions[world_id]["native_hereditary_birth_output_path"] = path


def _snapshot_numeric_stats_v1(
    values: Sequence[object],
) -> dict[str, float | int | None]:
    numeric = [
        value
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        "mean": (sum(numeric) / len(numeric)) if numeric else None,
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
    }


def _native_snapshot_numeric_stats_v1(
    value: object,
) -> dict[str, float | int | None]:
    if not isinstance(value, Mapping):
        raise CompactWorldAtlasReducerError("native snapshot statistics are invalid")
    count = value.get("count")
    integer_only = value.get("integer_only")
    if type(count) is not int or count < 0 or type(integer_only) is not bool:
        raise CompactWorldAtlasReducerError("native snapshot statistics drifted")
    if integer_only:
        total = value.get("sum")
        minimum = value.get("minimum")
        maximum = value.get("maximum")
        if (
            type(total) is not int
            or (count == 0 and (minimum is not None or maximum is not None))
            or (
                count > 0
                and (type(minimum) is not int or type(maximum) is not int)
            )
        ):
            raise CompactWorldAtlasReducerError(
                "native integer snapshot statistics drifted"
            )
        return {
            "mean": (total / count) if count else None,
            "min": minimum,
            "max": maximum,
        }
    values = value.get("values")
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(
            not isinstance(item, (int, float)) or isinstance(item, bool)
            for item in values
        )
    ):
        raise CompactWorldAtlasReducerError(
            "native floating snapshot statistics drifted"
        )
    return _snapshot_numeric_stats_v1(values)


def _native_snapshot_histograms_v1(
    value: object,
    *,
    expected_loci: Sequence[int],
) -> dict[str, list[dict[str, int]]]:
    if not isinstance(value, list) or len(value) != len(expected_loci):
        raise CompactWorldAtlasReducerError("native snapshot histograms drifted")
    result: dict[str, list[dict[str, int]]] = {}
    for expected_locus, row in zip(expected_loci, value, strict=True):
        if (
            not isinstance(row, Mapping)
            or row.get("locus") != expected_locus
            or not isinstance(row.get("alleles"), list)
            or len(row["alleles"]) != 2
        ):
            raise CompactWorldAtlasReducerError(
                "native snapshot histogram locus drifted"
            )
        alleles: list[dict[str, int]] = []
        for entries in row["alleles"]:
            if not isinstance(entries, list):
                raise CompactWorldAtlasReducerError(
                    "native snapshot histogram allele drifted"
                )
            histogram: dict[str, int] = {}
            prior: int | None = None
            for entry in entries:
                if (
                    not isinstance(entry, list)
                    or len(entry) != 2
                    or type(entry[0]) is not int
                    or type(entry[1]) is not int
                    or entry[1] < 1
                    or (prior is not None and entry[0] <= prior)
                ):
                    raise CompactWorldAtlasReducerError(
                        "native snapshot histogram entry drifted"
                    )
                prior = entry[0]
                histogram[str(entry[0])] = entry[1]
            alleles.append(histogram)
        result[str(expected_locus)] = alleles
    return result


def _encode_lifetime_metric_row_v1(metrics: Mapping[str, Any]) -> bytes:
    row = bytearray(COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1)
    cursor = 0
    for index, name in enumerate(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1):
        value = metrics.get(name)
        if type(value) is not int:
            raise CompactWorldAtlasReducerError(
                "active lifetime metric value is invalid"
            )
        if index == 17:
            try:
                struct.pack_into("<q", row, cursor, value)
            except struct.error as exc:
                raise CompactWorldAtlasReducerError(
                    "active signed lifetime metric exceeds its ABI"
                ) from exc
            cursor += 8
        else:
            try:
                struct.pack_into("<I", row, cursor, value)
            except struct.error as exc:
                raise CompactWorldAtlasReducerError(
                    "active unsigned lifetime metric exceeds its ABI"
                ) from exc
            cursor += 4
    return bytes(row)


def _snapshot_distribution(
    reconstruction: dict[str, Any],
    *,
    result: dict[str, Any],
    absolute_tick: int,
    active_override: Mapping[int, _ActiveGenomeV1] | None = None,
) -> None:
    snapshot = reconstruction["snapshots"].get(absolute_tick)
    milestone = reconstruction["milestones"].get(absolute_tick)
    if snapshot is None or milestone is None:
        return
    expected = snapshot["agent_state_ids"]
    active = reconstruction["active"] if active_override is None else active_override
    if set(expected) != set(active):
        missing = sorted(set(expected) - set(active))[:8]
        extra = sorted(set(active) - set(expected))[:8]
        raise CompactWorldAtlasReducerError(
            "reconstructed active ownership differs from snapshot at tick "
            f"{absolute_tick}: missing={missing}, extra={extra}"
        )
    policy_loci: tuple[int, ...] = reconstruction["policy_loci"]
    trait_loci: tuple[int, ...] = reconstruction["trait_loci"]
    policy_histograms = {str(locus): [{}, {}] for locus in policy_loci}
    trait_histograms = {str(locus): [{}, {}] for locus in trait_loci}
    metadata_by_agent = snapshot.get("agent_metadata", {})
    metrics_by_agent = snapshot.get("agent_metrics", {})
    mutation_lineage_count = 0
    lineage_depth_max = 0
    for agent_id, expected_state_id in expected.items():
        record = active[agent_id]
        metadata = metadata_by_agent.get(agent_id)
        if isinstance(metadata, Mapping):
            record.agent_state_id = metadata.get("agent_state_id")
            record.lineage_id = metadata.get("lineage_id")
            record.parent_a_id = metadata.get("parent_a_id", record.parent_a_id)
            record.parent_b_id = metadata.get("parent_b_id", record.parent_b_id)
            record.birth_tick = metadata.get("absolute_birth_tick", record.birth_tick)
            depth = metadata.get("lineage_depth")
            if type(depth) is int and depth >= 0:
                # The compact native snapshot ABI exposes lineage_depth as a
                # founder/non-founder marker (the encoder emits 0 when both
                # parent IDs are zero and 1 otherwise).  It is not the
                # transitive ancestry depth reconstructed from birth events.
                snapshot_depth = int(
                    record.parent_a_id != 0 or record.parent_b_id != 0
                )
                if depth != snapshot_depth:
                    raise CompactWorldAtlasReducerError(
                        "snapshot lineage-depth marker differs from parentage"
                    )
        metrics = metrics_by_agent.get(agent_id)
        if isinstance(metrics, Mapping):
            record.latest_metrics = dict(metrics)
        actual_state_id = _genome_state_id(record.encoded)
        if actual_state_id != expected_state_id:
            raise CompactWorldAtlasReducerError(
                "reconstructed genome identity differs from snapshot at tick "
                f"{absolute_tick} for agent {agent_id}"
            )
        result["integrity"]["genome_state_identity_checks"] += 1
        if reconstruction["hereditary_contract"] is not None:
            result["integrity"]["hereditary_full_identity_checks"] += 1
        mutation_lineage_count += int(record.mutation_lineage)
        lineage_depth_max = max(lineage_depth_max, record.generation)
        values = _GENOME_ROW_STRUCT.unpack(record.encoded)
        for locus in policy_loci:
            for allele, index in enumerate((locus, _POLICY_LOCI_PER_HAPLOTYPE + locus)):
                key = str(values[index])
                histogram = policy_histograms[str(locus)][allele]
                histogram[key] = histogram.get(key, 0) + 1
        for locus in trait_loci:
            for allele, index in enumerate(
                (
                    _POLICY_VALUE_COUNT + locus,
                    _POLICY_VALUE_COUNT + _TRAIT_LOCI_PER_HAPLOTYPE + locus,
                )
            ):
                key = str(values[index])
                histogram = trait_histograms[str(locus)][allele]
                histogram[key] = histogram.get(key, 0) + 1
    milestone["genome_reconstruction"] = {
        "active_agent_count": len(active),
        "state_identity_check_count": len(active),
        "mutation_lineage_active_count": mutation_lineage_count,
        "maximum_active_lineage_depth": lineage_depth_max,
        "tracked_policy_locus_histograms": policy_histograms,
        "tracked_trait_locus_histograms": trait_histograms,
    }
    hereditary_projection = _hereditary_snapshot_projection_v1(
        reconstruction, active
    )
    if hereditary_projection is not None:
        hereditary_projection["absolute_tick"] = absolute_tick
        milestone["hereditary_frequency_trajectory"] = hereditary_projection


def _initialize_reconstruction(
    reconstruction: dict[str, Any], *, result: dict[str, Any]
) -> None:
    start = reconstruction["start_next_tick"]
    snapshot = reconstruction["snapshots"].get(start)
    if snapshot is None:
        raise CompactWorldAtlasReducerError(
            "exact founder genome reconstruction requires the initial snapshot"
        )
    catalog = snapshot.get("genome_catalog", {})
    if not catalog:
        raise CompactWorldAtlasReducerError(
            "initial snapshot lacks the founder genome catalog"
        )
    active: dict[int, _ActiveGenomeV1] = {}
    for agent_id, state_id in snapshot["agent_state_ids"].items():
        encoded = catalog.get(state_id)
        if encoded is None:
            raise CompactWorldAtlasReducerError(
                "initial active agent refers to an absent founder genome"
            )
        if _genome_state_id(encoded) != state_id:
            raise CompactWorldAtlasReducerError(
                "founder genome catalog identity differs"
            )
        founder_metadata = snapshot.get("agent_metadata", {}).get(agent_id, {})
        founder_tracked_id = (
            _tracked_genome_state_id(
                encoded,
                policy_loci=reconstruction["policy_loci"],
                trait_loci=reconstruction["trait_loci"],
            )
            if reconstruction["hereditary_contract"] is not None
            else None
        )
        active[agent_id] = _ActiveGenomeV1(
            encoded=encoded,
            generation=0,
            mutation_lineage=False,
            born=False,
            agent_state_id=founder_metadata.get("agent_state_id"),
            lineage_id=founder_metadata.get("lineage_id"),
            full_genome_state_id=state_id,
            tracked_genotype_state_id=founder_tracked_id,
            parent_a_id=founder_metadata.get("parent_a_id", 0),
            parent_b_id=founder_metadata.get("parent_b_id", 0),
            birth_tick=founder_metadata.get("absolute_birth_tick", 0),
            latest_metrics=snapshot.get("agent_metrics", {}).get(agent_id),
        )
    reconstruction["active"] = active
    reconstruction["initialized"] = True
    _snapshot_distribution(reconstruction, result=result, absolute_tick=start)


def _prepare_dispatch(
    pointer: Mapping[str, Any],
    *,
    reconstruction_by_cell: Mapping[str, dict[str, Any]],
    scan_event_kinds: bool = True,
) -> _PreparedDispatchV1:
    """Verify one dispatch and derive immutable per-world work descriptors.

    The event, metric, and provenance payloads are read and authenticated here
    exactly once.  World reduction is deliberately separate so the same frame
    can be consumed by several isolated workers without rereading or
    re-decompressing the source artifact.
    """
    root, manifest = _artifact_manifest(pointer, label="compact dispatch")
    if (
        manifest.get("schema_version") != COMPACT_SCIENCE_DISPATCH_SCHEMA_V1
        or manifest.get("metric_fields") != list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
        or manifest.get("birth_provenance_encoding")
        != COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1
        or manifest.get("metric_storage_encoding")
        != COMPACT_SCIENCE_DEATH_METRIC_ENCODING_V1
        or manifest.get("birth_provenance_storage_encoding")
        != COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_ENCODING_V1
        or manifest.get("dropped_record_count") != 0
    ):
        raise CompactWorldAtlasReducerError("compact dispatch ABI differs")
    source_sink_receipt_sha256 = manifest.get("source_sink_receipt_sha256")
    if source_sink_receipt_sha256 is not None and (
        not isinstance(source_sink_receipt_sha256, str)
        or len(source_sink_receipt_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_sink_receipt_sha256
        )
    ):
        raise CompactWorldAtlasReducerError(
            "compact dispatch source sink receipt binding is invalid"
        )
    event_count = manifest.get("event_count")
    logical_worlds = manifest.get("logical_worlds")
    offsets = manifest.get("world_offsets")
    if (
        type(event_count) is not int
        or event_count < 0
        or not isinstance(logical_worlds, list)
        or not isinstance(offsets, list)
        or len(offsets) != len(logical_worlds) + 1
        or offsets[0] != 0
        or offsets[-1] != event_count
    ):
        raise CompactWorldAtlasReducerError("compact dispatch ownership differs")
    if any(
        type(offset) is not int or offset < 0 or (index and offset < offsets[index - 1])
        for index, offset in enumerate(offsets)
    ):
        raise CompactWorldAtlasReducerError("compact dispatch offsets are invalid")
    events = _bound_dispatch_payload_v1(
        root,
        manifest,
        prefix="event",
        label="sparse lifecycle event",
    )
    metrics_raw = _bound_dispatch_payload_v1(
        root,
        manifest,
        prefix="metric",
        label="event metric",
    )
    provenance = _bound_dispatch_payload_v1(
        root,
        manifest,
        prefix="birth_provenance",
        label="birth provenance",
    )
    birth_count = manifest.get("birth_event_count")
    death_count = manifest.get("death_event_count")
    if (
        type(birth_count) is not int
        or type(death_count) is not int
        or birth_count + death_count != event_count
        or len(events) != event_count * 32
        or len(metrics_raw) != death_count * COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1
        or len(provenance)
        != birth_count * COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1
    ):
        raise CompactWorldAtlasReducerError("compact event row count differs")
    start = int(manifest["start_next_absolute_tick"])
    end = int(manifest["end_next_absolute_tick"])
    tick_count = int(manifest["tick_count"])
    if end - start != tick_count or not 1 <= tick_count <= 64:
        raise CompactWorldAtlasReducerError("compact dispatch tick span differs")
    birth_cursor = 0
    death_cursor = 0
    world_work: list[_DispatchWorldWorkV1] = []
    for logical_index, owner in enumerate(logical_worlds):
        if not isinstance(owner, Mapping):
            raise CompactWorldAtlasReducerError(
                "dispatch logical-world owner is invalid"
            )
        cell_id_value = owner.get("cell_id")
        if cell_id_value not in reconstruction_by_cell:
            raise CompactWorldAtlasReducerError(
                "dispatch names an unknown logical world"
            )
        cell_id = str(cell_id_value)
        reconstruction = reconstruction_by_cell[cell_id]
        world = reconstruction["world"]
        if (
            owner.get("science_world_sha256") != world.semantic_world_sha256.to_dict()
            or owner.get("science_acquisition_sha256")
            != world.acquisition_sha256.to_dict()
        ):
            raise CompactWorldAtlasReducerError(
                "dispatch logical-world identity differs from the bundle"
            )
        first, stop = int(offsets[logical_index]), int(offsets[logical_index + 1])
        birth_count_for_world = 0
        death_count_for_world = 0
        if scan_event_kinds:
            for row_index in range(first, stop):
                packed = _EVENT_ROW_STRUCT.unpack_from(events, row_index * 32)[3]
                event_kind = (packed >> 6) & 0x3
                if event_kind == 1:
                    death_count_for_world += 1
                elif event_kind == 2:
                    birth_count_for_world += 1
                else:
                    raise CompactWorldAtlasReducerError(
                        "unsupported compact lifecycle event kind"
                    )
        world_work.append(
            _DispatchWorldWorkV1(
                cell_id=cell_id,
                first=first,
                stop=stop,
                birth_base=birth_cursor,
                death_base=death_cursor,
                birth_count=birth_count_for_world,
                death_count=death_count_for_world,
            )
        )
        birth_cursor += world_work[-1].birth_count
        death_cursor += world_work[-1].death_count
    if scan_event_kinds and (
        birth_cursor != birth_count or death_cursor != death_count
    ):
        raise CompactWorldAtlasReducerError(
            "compact event-kind auxiliary cursors are incomplete"
        )
    return _PreparedDispatchV1(
        work=_DispatchWorkFrameV1(
            start=start,
            end=end,
            tick_count=tick_count,
            worlds=tuple(world_work),
        ),
        events=events,
        metrics=metrics_raw,
        provenance=provenance,
        world_offsets=tuple(int(offset) for offset in offsets),
    )


def _reduce_dispatch_worlds(
    work: _DispatchWorkFrameV1,
    *,
    events: bytes | memoryview,
    metrics_raw: bytes | memoryview,
    provenance: bytes | memoryview,
    worlds_by_cell: Mapping[str, dict[str, Any]],
    reconstruction_by_cell: Mapping[str, dict[str, Any]],
) -> None:
    """Apply one verified dispatch to the disjoint logical-world states."""
    for world_work in work.worlds:
        cell_id = world_work.cell_id
        result = worlds_by_cell[cell_id]
        reconstruction = reconstruction_by_cell[cell_id]
        world = reconstruction["world"]
        if (
            not reconstruction["initialized"]
            or reconstruction["next_tick"] != work.start
        ):
            raise CompactWorldAtlasReducerError(
                "compact dispatches are not a contiguous absolute-tick stream"
            )
        active: dict[int, _ActiveGenomeV1] = reconstruction["active"]
        result["integrity"]["dispatch_count"] += 1
        result["integrity"]["dispatch_tick_count"] += work.tick_count
        result["integrity"]["event_count"] += world_work.stop - world_work.first
        birth_cursor = world_work.birth_base
        death_cursor = world_work.death_base
        birth_used = 0
        death_used = 0
        prior_tick = work.start - 1
        for row_index in range(world_work.first, world_work.stop):
            agent_id, parent_a, parent_b, packed = _EVENT_ROW_STRUCT.unpack_from(
                events, row_index * 32
            )
            event_kind = (packed >> 6) & 0x3
            tick = work.start + (packed & 0x3F)
            if tick < prior_tick or not work.start <= tick < work.end:
                raise CompactWorldAtlasReducerError(
                    "compact lifecycle event order or tick differs"
                )
            prior_tick = tick
            if event_kind == 1:
                if agent_id not in active:
                    raise CompactWorldAtlasReducerError(
                        f"death names an inactive agent at tick {tick}"
                    )
                deceased = active[agent_id]
                death_metrics: dict[str, int] = {}
                result["death_count"] += 1
                metric_offset = (
                    death_cursor + death_used
                ) * COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1
                for metric_index, name in enumerate(
                    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
                ):
                    if metric_index == 17:
                        value = struct.unpack_from("<q", metrics_raw, metric_offset)[0]
                        metric_offset += 8
                    else:
                        value = struct.unpack_from("<I", metrics_raw, metric_offset)[0]
                        metric_offset += 4
                    death_metrics[name] = int(value)
                    result["completed_lifetime_metric_totals"][name] += int(value)
                _record_hereditary_metrics_v1(
                    reconstruction, deceased, death_metrics
                )
                death_used += 1
                del active[agent_id]
            elif event_kind == 2:
                if not agent_id or not parent_a or not parent_b:
                    raise CompactWorldAtlasReducerError("birth parentage is incomplete")
                if (
                    agent_id in active
                    or parent_a not in active
                    or parent_b not in active
                ):
                    raise CompactWorldAtlasReducerError(
                        f"birth ancestry is not active at tick {tick}"
                    )
                result["birth_count"] += 1
                row_start = (
                    birth_cursor + birth_used
                ) * COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1
                row = provenance[
                    row_start : row_start
                    + COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1
                ]
                provenance_row = bytearray(COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1)
                for packed_index, packed_loci in enumerate(row[:64]):
                    provenance_row[packed_index * 2] = packed_loci & 0xF
                    provenance_row[packed_index * 2 + 1] = packed_loci >> 4
                provenance_row[128:132] = row[64:68]
                mutation_step = reconstruction["mutation_step_static"]
                if mutation_step is None:
                    mutation_step = _control_value_at_tick(
                        reconstruction["semantic_inputs"],
                        "mutation.step",
                        tick,
                        default=1,
                    )
                if type(mutation_step) is not int:
                    raise CompactWorldAtlasReducerError(
                        "resolved mutation step is not an integer"
                    )
                try:
                    reconstructed = reconstruct_compact_birth_genome_v1(
                        parent_a_genome=_decode_genome(active[parent_a].encoded),
                        parent_b_genome=_decode_genome(active[parent_b].encoded),
                        birth_provenance=provenance_row,
                        mutation_step=mutation_step,
                    )
                except Exception as exc:
                    raise CompactWorldAtlasReducerError(
                        f"birth genome reconstruction failed at tick {tick}: {exc}"
                    ) from exc
                child_encoded = _encode_genome(
                    reconstructed["child_genome"], label="reconstructed child"
                )
                mutation_count = int(reconstructed["mutation_origin_count"])
                saturated_noops = int(reconstructed["saturated_noop_count"])
                result["reconstructed_birth_count"] += 1
                result["mutation_origin_count"] += mutation_count
                result["mutation_saturated_noop_count"] += saturated_noops
                for change in reconstructed["changed_alleles"]:
                    linked_locus = int(change["linked_locus"])
                    result["mutation_origin_counts_by_linked_locus"][linked_locus] += 1
                    result["mutation_delta_totals_by_linked_locus"][
                        linked_locus
                    ] += int(change["delta"])
                recombination = int.from_bytes(row[64:68], "little")
                result["recombination_birth_count"] += 1
                result["recombination_nonzero_pattern_count"] += int(recombination != 0)
                for block in range(16):
                    result["recombination_choice_one_counts_by_block"][block][0] += (
                        recombination >> (2 * block)
                    ) & 1
                    result["recombination_choice_one_counts_by_block"][block][1] += (
                        recombination >> (2 * block + 1)
                    ) & 1
                parents = (active[parent_a], active[parent_b])
                parent_full_ids: list[str] = []
                parent_tracked_ids: list[str] = []
                child_full_id: str | None = None
                child_tracked_id: str | None = None
                if reconstruction["hereditary_contract"] is not None:
                    parent_full_ids = [
                        parent.full_genome_state_id
                        or _genome_state_id(parent.encoded)
                        for parent in parents
                    ]
                    parent_tracked_ids = [
                        parent.tracked_genotype_state_id
                        or _hereditary_tracked_id_v1(reconstruction, parent)
                        for parent in parents
                    ]
                    child_full_id = _genome_state_id(child_encoded)
                    child_tracked_id = _tracked_genome_state_id(
                        child_encoded,
                        policy_loci=reconstruction["policy_loci"],
                        trait_loci=reconstruction["trait_loci"],
                    )
                if any(parent.born for parent in parents):
                    result["carrier_reproduction_birth_count"] += 1
                if any(parent.mutation_lineage for parent in parents):
                    result["mutation_carrier_reproduction_birth_count"] += 1
                for parent in parents:
                    if parent.born and not parent.reproduced:
                        parent.reproduced = True
                        result["unique_born_agents_reproduced_count"] += 1
                        if parent.mutation_lineage:
                            result["unique_mutation_carriers_reproduced_count"] += 1
                generation = max(parent.generation for parent in parents) + 1
                mutation_lineage = mutation_count > 0 or any(
                    parent.mutation_lineage for parent in parents
                )
                if reconstruction["hereditary_contract"] is not None:
                    birth_record = {
                        "absolute_tick": tick,
                        "world_id": cell_id,
                        "child_agent_id": agent_id,
                        "parent_agent_ids": [parent_a, parent_b],
                        "parent_agent_state_ids": [
                            parents[0].agent_state_id,
                            parents[1].agent_state_id,
                        ],
                        "parent_full_genome_state_ids": [
                            parent_full_ids[0],
                            parent_full_ids[1],
                        ],
                        "child_full_genome_state_id": child_full_id,
                        "parent_tracked_genotype_state_ids": parent_tracked_ids,
                        "child_tracked_genotype_state_id": child_tracked_id,
                        "lineage_depth": generation,
                        "mutation_origin_count": mutation_count,
                        "recombination_pattern": recombination,
                        "birth_provenance_hex": bytes(row).hex(),
                    }
                    spool = reconstruction["hereditary_birth_spool"]
                    if not isinstance(spool, _HereditaryBirthRecordSpoolV1):
                        raise CompactWorldAtlasReducerError(
                            "full-hereditary attribution birth spool is absent"
                        )
                    spool.append(birth_record)
                    parent_tracked_ids = birth_record[
                        "parent_tracked_genotype_state_ids"
                    ]
                    descendant_counts = reconstruction[
                        "hereditary_descendant_counts"
                    ]
                    for identity in sorted(set(parent_tracked_ids)):
                        descendant_counts[identity] = (
                            descendant_counts.get(identity, 0) + 1
                        )
                    result["integrity"]["hereditary_parent_child_records"] += 1
                active[agent_id] = _ActiveGenomeV1(
                    encoded=child_encoded,
                    generation=generation,
                    mutation_lineage=mutation_lineage,
                    born=True,
                    full_genome_state_id=(
                        child_full_id
                        if reconstruction["hereditary_contract"] is not None
                        else None
                    ),
                    tracked_genotype_state_id=(
                        child_tracked_id
                        if reconstruction["hereditary_contract"] is not None
                        else None
                    ),
                    parent_a_id=parent_a,
                    parent_b_id=parent_b,
                    birth_tick=tick,
                )
                result["maximum_lineage_depth"] = max(
                    result["maximum_lineage_depth"], generation
                )
                birth_used += 1
            else:
                raise CompactWorldAtlasReducerError(
                    f"unsupported compact lifecycle event kind at tick {tick}"
                )
        if birth_used != world_work.birth_count or death_used != world_work.death_count:
            raise CompactWorldAtlasReducerError(
                "compact event-kind auxiliary cursors are incomplete"
            )
        capacity = int(world.semantic_inputs.to_value()["world"]["resident_capacity"])
        if len(active) > capacity:
            raise CompactWorldAtlasReducerError(
                "reconstructed active population exceeds resident capacity"
            )
        reconstruction["next_tick"] = work.end
        _snapshot_distribution(reconstruction, result=result, absolute_tick=work.end)


def _reduce_dispatch(
    pointer: Mapping[str, Any],
    *,
    worlds_by_cell: Mapping[str, dict[str, Any]],
    reconstruction_by_cell: Mapping[str, dict[str, Any]],
) -> None:
    frame = _prepare_dispatch(
        pointer,
        reconstruction_by_cell=reconstruction_by_cell,
    )
    _reduce_dispatch_worlds(
        frame.work,
        events=frame.events,
        metrics_raw=frame.metrics,
        provenance=frame.provenance,
        worlds_by_cell=worlds_by_cell,
        reconstruction_by_cell=reconstruction_by_cell,
    )


def _prepare_snapshot(
    pointer: Mapping[str, Any],
) -> _PreparedSnapshotV1:
    root, manifest = _artifact_manifest(pointer, label="compact snapshot")
    if (
        manifest.get("schema_version") != COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1
        or manifest.get("metric_fields")
        != list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
        or manifest.get("dropped_record_count") != 0
    ):
        raise CompactWorldAtlasReducerError("compact snapshot ABI differs")
    compressed = _bound_file(
        root,
        manifest.get("data_file"),
        manifest.get("compressed_file_sha256"),
        label="snapshot data",
    )
    try:
        uncompressed = gzip.decompress(compressed)
    except (OSError, EOFError) as exc:
        raise CompactWorldAtlasReducerError("snapshot data is invalid") from exc
    expected_uncompressed_sha256 = manifest.get("uncompressed_content_sha256")
    if (
        not isinstance(expected_uncompressed_sha256, str)
        or _sha256_bytes(uncompressed) != expected_uncompressed_sha256
    ):
        raise CompactWorldAtlasReducerError("snapshot uncompressed SHA-256 differs")
    absolute_tick = manifest.get("absolute_tick")
    row_count = manifest.get("row_count")
    if (
        type(absolute_tick) is not int
        or absolute_tick < 0
        or type(row_count) is not int
        or row_count < 1
    ):
        raise CompactWorldAtlasReducerError("snapshot extent is invalid")
    return _PreparedSnapshotV1(
        manifest=manifest,
        uncompressed=uncompressed,
        absolute_tick=absolute_tick,
        row_count=row_count,
    )


def _reduce_snapshot(
    pointer: Mapping[str, Any],
    *,
    worlds_by_cell: Mapping[str, dict[str, Any]],
    reconstruction_by_cell: Mapping[str, dict[str, Any]],
) -> tuple[int, tuple[str, ...]]:
    prepared = _prepare_snapshot(pointer)
    manifest = prepared.manifest
    uncompressed = prepared.uncompressed
    try:
        lines = uncompressed.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise CompactWorldAtlasReducerError("snapshot data is invalid") from exc
    if len(lines) != prepared.row_count:
        raise CompactWorldAtlasReducerError("snapshot row count differs")
    absolute_tick = prepared.absolute_tick
    grouped: dict[str, dict[str, Any]] = {}
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompactWorldAtlasReducerError("snapshot JSON row is invalid") from exc
        cell_id = row.get("cell_id")
        if cell_id not in worlds_by_cell:
            raise CompactWorldAtlasReducerError(
                "snapshot names an unknown logical world"
            )
        if row.get("absolute_tick") != absolute_tick:
            raise CompactWorldAtlasReducerError(
                "snapshot row tick differs from its manifest"
            )
        cell_id_text = str(cell_id)
        strategy_enabled = (
            reconstruction_by_cell[cell_id_text]["strategy_contract"] is not None
        )
        group = grouped.setdefault(
            cell_id_text,
            {
                "absolute_tick": absolute_tick,
                "world_state": None,
                "active_agent_count": 0,
                "active_lifetime_metric_totals": {
                    name: 0 for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
                },
                "genome_catalog_size": 0,
                "_agent_state_ids": {},
                "_agent_metadata": {},
                "_agent_metrics": {},
                "_genome_catalog": {},
                "_strategy_enabled": strategy_enabled,
                "_hereditary_enabled": (
                    reconstruction_by_cell[cell_id_text]["hereditary_contract"]
                    is not None
                ),
                "_strategy_agent_ids": [],
                "_strategy_metric_rows": bytearray(),
                "_strategy_policy_ids": {},
                "_strategy_parent_ids": {},
                "_strategy_energy": [],
                "_strategy_health": [],
                "_strategy_injury": [],
            },
        )
        if row.get("row_kind") == "world_state":
            if group["world_state"] is not None:
                raise CompactWorldAtlasReducerError(
                    "snapshot duplicates one logical world state"
                )
            group["world_state"] = {
                key: value
                for key, value in row.items()
                if key not in {"row_kind", "cell_id", "world_uid", "absolute_tick"}
            }
        elif row.get("row_kind") == "active_agent_state":
            agent_id = row.get("agent_id")
            state_id = row.get("genome_state_id")
            if (
                type(agent_id) is not int
                or agent_id <= 0
                or not isinstance(state_id, str)
                or not state_id.startswith("sha256:")
                or agent_id in group["_agent_state_ids"]
            ):
                raise CompactWorldAtlasReducerError(
                    "active snapshot agent identity differs"
                )
            group["_agent_state_ids"][agent_id] = state_id
            group["_agent_metadata"][agent_id] = {
                "agent_state_id": row.get("agent_state_id"),
                "lineage_id": row.get("lineage_id"),
                "lineage_depth": row.get("lineage_depth"),
                "parent_a_id": row.get("parent_a_id"),
                "parent_b_id": row.get("parent_b_id"),
                "absolute_birth_tick": row.get("absolute_birth_tick"),
            }
            group["active_agent_count"] += 1
            metrics = row.get("lifetime_metrics")
            if not isinstance(metrics, Mapping):
                raise CompactWorldAtlasReducerError(
                    "active lifetime metrics are absent"
                )
            group["_agent_metrics"][agent_id] = dict(metrics)
            encoded_metrics = _encode_lifetime_metric_row_v1(metrics)
            if group["_hereditary_enabled"]:
                if (
                    not isinstance(row.get("agent_state_id"), str)
                    or not row["agent_state_id"].startswith("sha256:")
                    or not isinstance(row.get("lineage_id"), str)
                    or not row["lineage_id"].startswith("sha256:")
                    or type(row.get("lineage_depth")) is not int
                    or row["lineage_depth"] < 0
                    or type(row.get("parent_a_id")) is not int
                    or row["parent_a_id"] < 0
                    or type(row.get("parent_b_id")) is not int
                    or row["parent_b_id"] < 0
                    or type(row.get("absolute_birth_tick")) is not int
                    or row["absolute_birth_tick"] < 0
                ):
                    raise CompactWorldAtlasReducerError(
                        "full-hereditary snapshot identity is incomplete"
                    )
            for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1:
                group["active_lifetime_metric_totals"][name] += int(metrics[name])
            if group["_strategy_enabled"]:
                group["_strategy_agent_ids"].append(agent_id)
                group["_strategy_metric_rows"].extend(encoded_metrics)
                group["_strategy_policy_ids"][agent_id] = row.get("policy_id")
                group["_strategy_parent_ids"][agent_id] = (
                    row.get("parent_a_id"),
                    row.get("parent_b_id"),
                )
                group["_strategy_energy"].append(row.get("energy"))
                group["_strategy_health"].append(row.get("health"))
                group["_strategy_injury"].append(row.get("injury"))
        elif row.get("row_kind") == "genome_catalog":
            state_id = row.get("genome_state_id")
            if not isinstance(state_id, str) or state_id in group["_genome_catalog"]:
                raise CompactWorldAtlasReducerError(
                    "snapshot genome catalog identity differs"
                )
            encoded = _encode_genome(row, label="snapshot catalog")
            if _genome_state_id(encoded) != state_id:
                raise CompactWorldAtlasReducerError(
                    "snapshot genome catalog does not bind its alleles"
                )
            group["_genome_catalog"][state_id] = encoded
            group["genome_catalog_size"] += 1
        else:
            raise CompactWorldAtlasReducerError("snapshot row kind is unsupported")
    captured_worlds = manifest.get("captured_worlds")
    if (
        not isinstance(captured_worlds, list)
        or manifest.get("captured_world_count") != len(captured_worlds)
        or any(not isinstance(row, Mapping) for row in captured_worlds)
        or {row.get("cell_id") for row in captured_worlds if isinstance(row, Mapping)}
        != set(grouped)
    ):
        raise CompactWorldAtlasReducerError("snapshot logical-world ownership differs")
    captured_by_cell = {str(row["cell_id"]): row for row in captured_worlds}
    for cell_id, milestone in grouped.items():
        result = worlds_by_cell[cell_id]
        reconstruction = reconstruction_by_cell[cell_id]
        world = reconstruction["world"]
        owner = captured_by_cell[cell_id]
        if (
            owner.get("science_world_sha256") != world.semantic_world_sha256.to_dict()
            or owner.get("science_acquisition_sha256")
            != world.acquisition_sha256.to_dict()
            or owner.get("active_agent_count") != milestone["active_agent_count"]
        ):
            raise CompactWorldAtlasReducerError(
                "snapshot world identity differs from the bundle"
            )
        if milestone["world_state"] is None:
            raise CompactWorldAtlasReducerError(
                "snapshot lacks one logical world state"
            )
        expected_ticks = world.acquisition.to_value()["snapshot_ticks"]
        if absolute_tick not in expected_ticks:
            raise CompactWorldAtlasReducerError(
                "snapshot tick is not owned by the logical world"
            )
        if absolute_tick in reconstruction["snapshots"]:
            raise CompactWorldAtlasReducerError(
                "snapshot duplicates a logical world milestone"
            )
        strategy_enabled = bool(milestone.pop("_strategy_enabled"))
        hereditary_enabled = bool(milestone.pop("_hereditary_enabled"))
        if hereditary_enabled != (
            reconstruction["hereditary_contract"] is not None
        ):
            raise CompactWorldAtlasReducerError(
                "snapshot hereditary-attribution mode differs from the world"
            )
        strategy_agent_ids = tuple(milestone.pop("_strategy_agent_ids"))
        strategy_metric_rows = bytes(milestone.pop("_strategy_metric_rows"))
        strategy_policy_ids = dict(milestone.pop("_strategy_policy_ids"))
        strategy_parent_ids = dict(milestone.pop("_strategy_parent_ids"))
        strategy_energy = tuple(milestone.pop("_strategy_energy"))
        strategy_health = tuple(milestone.pop("_strategy_health"))
        strategy_injury = tuple(milestone.pop("_strategy_injury"))
        reconstruction["snapshots"][absolute_tick] = {
            "agent_state_ids": dict(milestone.pop("_agent_state_ids")),
            "agent_metadata": dict(milestone.pop("_agent_metadata")),
            "agent_metrics": dict(milestone.pop("_agent_metrics")),
            "genome_catalog": dict(milestone.pop("_genome_catalog")),
        }
        if strategy_enabled:
            fields = milestone["world_state"].get("qualified_field_values")
            if not isinstance(fields, Mapping):
                raise CompactWorldAtlasReducerError(
                    "strategy snapshot world fields are absent"
                )
            reconstruction["snapshots"][absolute_tick].update(
                {
                    "strategy_agent_ids": strategy_agent_ids,
                    "strategy_metric_rows": strategy_metric_rows,
                    "strategy_policy_ids": strategy_policy_ids,
                    "strategy_parent_ids": strategy_parent_ids,
                }
            )
            reconstruction["strategy_snapshots"][absolute_tick] = {
                "absolute_tick": absolute_tick,
                "active_count": milestone["active_agent_count"],
                "population_count": fields.get(
                    "population_count", milestone["active_agent_count"]
                ),
                "birth_count": fields.get("birth_count"),
                "death_count": fields.get("death_count"),
                "mutation_count": fields.get("mutation_count"),
                "resource_total": fields.get("resource_total"),
                "resource_mean": fields.get("resource_mean"),
                "resource_min": fields.get("resource_min"),
                "resource_max": fields.get("resource_max"),
                "resource_nonzero_count": fields.get("resource_nonzero_count"),
                "energy": _snapshot_numeric_stats_v1(strategy_energy),
                "health": _snapshot_numeric_stats_v1(strategy_health),
                "injury": _snapshot_numeric_stats_v1(strategy_injury),
                "active_by_strategy": {},
            }
        reconstruction["milestones"][absolute_tick] = milestone
        result["milestones"].append(milestone)
        result["integrity"]["snapshot_count"] += 1
    return absolute_tick, tuple(grouped)


def _activate_snapshot_frontier_v1(
    reconstruction: dict[str, Any],
    *,
    result: dict[str, Any],
    absolute_tick: int,
) -> None:
    """Initialize from, or reconcile against, a newly delivered snapshot."""

    if reconstruction["initialized"]:
        next_tick = int(reconstruction["next_tick"])
        if next_tick > absolute_tick:
            raise CompactWorldAtlasReducerError(
                "snapshot arrived after the reducer passed its absolute tick"
            )
        if next_tick == absolute_tick:
            _snapshot_distribution(
                reconstruction,
                result=result,
                absolute_tick=absolute_tick,
            )
        return
    if absolute_tick == reconstruction["start_next_tick"]:
        _initialize_reconstruction(reconstruction, result=result)


def _apply_snapshot_payload_v1(
    absolute_tick: int,
    payload_by_cell: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
    *,
    worlds_by_cell: Mapping[str, dict[str, Any]],
    reconstruction_by_cell: Mapping[str, dict[str, Any]],
) -> None:
    """Apply one parent-verified snapshot slice inside an isolated worker."""

    for cell_id, payload in payload_by_cell.items():
        if cell_id not in worlds_by_cell or cell_id not in reconstruction_by_cell:
            raise CompactWorldAtlasReducerError(
                "streamed snapshot names an unowned logical world"
            )
        snapshot, milestone = payload
        result = worlds_by_cell[cell_id]
        reconstruction = reconstruction_by_cell[cell_id]
        if absolute_tick in reconstruction["snapshots"]:
            raise CompactWorldAtlasReducerError(
                "snapshot duplicates a logical world milestone"
            )
        reconstruction["snapshots"][absolute_tick] = snapshot
        reconstruction["milestones"][absolute_tick] = milestone
        result["milestones"].append(milestone)
        result["integrity"]["snapshot_count"] += 1
        _activate_snapshot_frontier_v1(
            reconstruction,
            result=result,
            absolute_tick=absolute_tick,
        )


def _attach_shared_memory(name: str) -> SharedMemory:
    """Attach without registering a worker-owned unlink obligation when possible."""
    try:
        return SharedMemory(name=name, create=False, track=False)
    except TypeError:  # pragma: no cover - compatibility with Python < 3.13
        return SharedMemory(name=name, create=False)


def _parallel_worker_error(
    worker_index: int,
    error_type: str,
    message: str,
    worker_traceback: str,
) -> CompactWorldAtlasReducerError:
    return CompactWorldAtlasReducerError(
        "parallel reducer worker "
        f"{worker_index} failed with {error_type}: {message}\n{worker_traceback}"
    )


def _compact_world_atlas_parallel_worker(
    worker_index: int,
    world_ids: tuple[str, ...],
    result_templates: Mapping[str, dict[str, Any]],
    reconstruction_templates: Mapping[str, dict[str, Any]],
    initialize_at_start: bool,
    output_root: str | Path | None,
    command_queue: Any,
    result_queue: Any,
) -> None:
    """Own one fixed logical-world shard until every dispatch is consumed."""
    local_birth_spool: _HereditaryBirthRecordSpoolV1 | None = None
    try:
        local_world_ids = tuple(world_ids)
        local_results = {
            world_id: result_templates[world_id] for world_id in local_world_ids
        }
        local_reconstructions = {
            world_id: reconstruction_templates[world_id] for world_id in local_world_ids
        }
        if any(
            reconstruction["hereditary_contract"] is not None
            for reconstruction in local_reconstructions.values()
        ):
            _configure_hereditary_birth_spool_v1(
                local_reconstructions,
                world_ids=local_world_ids,
                output_root=output_root,
                artifact_suffix=f"-worker-{worker_index}",
            )
            local_birth_spool = next(
                (
                    reconstruction["hereditary_birth_spool"]
                    for reconstruction in local_reconstructions.values()
                    if isinstance(
                        reconstruction["hereditary_birth_spool"],
                        _HereditaryBirthRecordSpoolV1,
                    )
                ),
                None,
            )
        if initialize_at_start:
            for world_id in local_world_ids:
                _initialize_reconstruction(
                    local_reconstructions[world_id],
                    result=local_results[world_id],
                )

        while True:
            command = command_queue.get()
            if command is None:
                if local_birth_spool is not None:
                    local_birth_spool.seal()
                break
            if not isinstance(command, tuple) or not command:
                raise CompactWorldAtlasReducerError(
                    "parallel reducer command is malformed"
                )
            command_kind = command[0]
            if command_kind == "snapshot":
                if len(command) != 3:
                    raise CompactWorldAtlasReducerError(
                        "parallel reducer snapshot command is malformed"
                    )
                _apply_snapshot_payload_v1(
                    command[1],
                    command[2],
                    worlds_by_cell=local_results,
                    reconstruction_by_cell=local_reconstructions,
                )
                result_queue.put(("ack", worker_index, command_kind))
                continue
            if command_kind != "dispatch" or len(command) != 6:
                raise CompactWorldAtlasReducerError(
                    "parallel reducer dispatch command is malformed"
                )
            _, work, shared_name, event_size, metric_size, provenance_size = command
            shared: SharedMemory | None = None
            buffer: memoryview | None = None
            event_view: memoryview | None = None
            metric_view: memoryview | None = None
            provenance_view: memoryview | None = None
            try:
                shared = _attach_shared_memory(shared_name)
                buffer = shared.buf
                event_view = buffer[:event_size]
                metric_view = buffer[event_size : event_size + metric_size]
                provenance_start = event_size + metric_size
                provenance_view = buffer[
                    provenance_start : provenance_start + provenance_size
                ]
                _reduce_dispatch_worlds(
                    work,
                    events=event_view,
                    metrics_raw=metric_view,
                    provenance=provenance_view,
                    worlds_by_cell=local_results,
                    reconstruction_by_cell=local_reconstructions,
                )
            finally:
                for view in (event_view, metric_view, provenance_view):
                    if view is not None:
                        view.release()
                if buffer is not None:
                    buffer.release()
                if shared is not None:
                    shared.close()
            result_queue.put(("ack", worker_index, command_kind))

        result_queue.put(
            (
                "done",
                worker_index,
                local_results,
                {
                    world_id: local_reconstructions[world_id]["next_tick"]
                    for world_id in local_world_ids
                },
                {
                    world_id: dict(local_birth_spool._artifact)
                    for world_id in local_world_ids
                }
                if local_birth_spool is not None
                else {},
            )
        )
    except BaseException as exc:  # pragma: no cover - exercised through IPC tests
        if local_birth_spool is not None:
            local_birth_spool.abort()
        result_queue.put(
            (
                "error",
                worker_index,
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
        )


def _parallel_message(
    result_queue: Any,
    processes: Sequence[multiprocessing.Process],
) -> tuple[Any, ...]:
    """Wait for a worker message while turning a silent worker death into failure."""
    while True:
        try:
            message = result_queue.get(timeout=5.0)
        except queue.Empty:
            if not any(process.is_alive() for process in processes):
                raise CompactWorldAtlasReducerError(
                    "parallel reducer workers exited without a result"
                )
            continue
        if not isinstance(message, tuple) or not message:
            raise CompactWorldAtlasReducerError(
                "parallel reducer worker message is malformed"
            )
        if message[0] == "error":
            raise _parallel_worker_error(*message[1:])
        return message


def _terminate_parallel_workers(
    processes: Sequence[multiprocessing.Process],
) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10)


class _ParallelReducerPoolV1:
    """Persistent, bounded process pool for both offline and streamed dispatches."""

    def __init__(
        self,
        *,
        world_ids: tuple[str, ...],
        worlds_by_cell: Mapping[str, dict[str, Any]],
        reconstruction_by_cell: Mapping[str, dict[str, Any]],
        max_workers: int,
        initialize_at_start: bool,
        output_root: str | Path | None,
    ) -> None:
        try:
            context_name = "forkserver" if os.name == "posix" else "spawn"
            context = multiprocessing.get_context(context_name)
        except ValueError as exc:  # pragma: no cover - POSIX without fork
            raise CompactWorldAtlasReducerError(
                "parallel compact reduction cannot acquire a safe process context"
            ) from exc
        self.world_ids = world_ids
        self.worlds_by_cell = worlds_by_cell
        self.reconstruction_by_cell = reconstruction_by_cell
        self.worker_count = min(
            max_workers,
            len(world_ids),
            os.cpu_count() or 1,
        )
        self.groups = tuple(
            tuple(range(worker_index, len(world_ids), self.worker_count))
            for worker_index in range(self.worker_count)
        )
        self.command_queues = [context.Queue(maxsize=1) for _ in self.groups]
        self.result_queue = context.Queue()
        self.processes: list[multiprocessing.Process] = []
        self.closed = False
        try:
            for worker_index, group in enumerate(self.groups):
                group_world_ids = tuple(world_ids[index] for index in group)
                process = context.Process(
                    target=_compact_world_atlas_parallel_worker,
                    args=(
                        worker_index,
                        group_world_ids,
                        {
                            world_id: worlds_by_cell[world_id]
                            for world_id in group_world_ids
                        },
                        {
                            world_id: reconstruction_by_cell[world_id]
                            for world_id in group_world_ids
                        },
                        initialize_at_start,
                        output_root,
                        self.command_queues[worker_index],
                        self.result_queue,
                    ),
                    name=f"compact-world-atlas-reducer-{worker_index}",
                )
                process.start()
                self.processes.append(process)
        except BaseException:
            self.terminate()
            raise

    def _await_acknowledgements(self, command_kind: str) -> None:
        pending = set(range(self.worker_count))
        while pending:
            message = _parallel_message(self.result_queue, self.processes)
            if (
                len(message) != 3
                or message[0] != "ack"
                or message[1] not in pending
                or message[2] != command_kind
            ):
                raise CompactWorldAtlasReducerError(
                    f"parallel reducer {command_kind} acknowledgement is malformed"
                )
            pending.remove(message[1])

    def submit_snapshot(
        self,
        absolute_tick: int,
        cell_ids: Sequence[str],
    ) -> None:
        """Deliver one already-verified snapshot to every disjoint shard."""

        if self.closed:
            raise CompactWorldAtlasReducerError("parallel reducer pool is closed")
        selected = set(cell_ids)
        try:
            for worker_index, group in enumerate(self.groups):
                payload: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
                for index in group:
                    world_id = self.world_ids[index]
                    if world_id not in selected:
                        continue
                    reconstruction = self.reconstruction_by_cell[world_id]
                    payload[world_id] = (
                        reconstruction["snapshots"][absolute_tick],
                        reconstruction["milestones"][absolute_tick],
                    )
                self.command_queues[worker_index].put(
                    ("snapshot", absolute_tick, payload)
                )
            self._await_acknowledgements("snapshot")
        except BaseException:
            self.terminate()
            raise

    def submit_dispatch(self, prepared: _PreparedDispatchV1) -> None:
        """Expose one verified dispatch once, then wait for every shard."""

        if self.closed:
            raise CompactWorldAtlasReducerError("parallel reducer pool is closed")
        event_size = len(prepared.events)
        metric_size = len(prepared.metrics)
        provenance_size = len(prepared.provenance)
        shared = SharedMemory(
            create=True,
            size=max(1, event_size + metric_size + provenance_size),
        )
        try:
            buffer = shared.buf
            if event_size:
                buffer[:event_size] = prepared.events
            if metric_size:
                buffer[event_size : event_size + metric_size] = prepared.metrics
            if provenance_size:
                provenance_start = event_size + metric_size
                buffer[provenance_start : provenance_start + provenance_size] = (
                    prepared.provenance
                )
            buffer.release()
            for worker_index, group in enumerate(self.groups):
                self.command_queues[worker_index].put(
                    (
                        "dispatch",
                        prepared.for_indices(group),
                        shared.name,
                        event_size,
                        metric_size,
                        provenance_size,
                    )
                )
            self._await_acknowledgements("dispatch")
        except BaseException:
            self.terminate()
            raise
        finally:
            shared.close()
            shared.unlink()

    def finish(self) -> None:
        """Collect canonical world results and close every worker cleanly."""

        if self.closed:
            raise CompactWorldAtlasReducerError("parallel reducer pool is closed")
        completed_results: dict[str, dict[str, Any]] = {}
        completed_ticks: dict[str, int] = {}
        try:
            for command_queue in self.command_queues:
                command_queue.put(None)
            pending = set(range(self.worker_count))
            while pending:
                message = _parallel_message(self.result_queue, self.processes)
                if (
                    len(message) != 5
                    or message[0] != "done"
                    or message[1] not in pending
                ):
                    raise CompactWorldAtlasReducerError(
                        "parallel reducer completion message is malformed"
                    )
                pending.remove(message[1])
                worker_results = message[2]
                worker_ticks = message[3]
                worker_artifacts = message[4]
                if not isinstance(worker_results, Mapping) or not isinstance(
                    worker_ticks, Mapping
                ) or not isinstance(worker_artifacts, Mapping):
                    raise CompactWorldAtlasReducerError(
                        "parallel reducer completion payload is malformed"
                    )
                completed_results.update(worker_results)
                completed_ticks.update(worker_ticks)
                for world_id, artifact in worker_artifacts.items():
                    if world_id not in self.reconstruction_by_cell or not isinstance(
                        artifact, Mapping
                    ):
                        raise CompactWorldAtlasReducerError(
                            "parallel reducer hereditary artifact ownership is malformed"
                        )
                    self.reconstruction_by_cell[world_id][
                        "hereditary_birth_artifact"
                    ] = dict(artifact)
            for process in self.processes:
                process.join(timeout=10)
                if process.is_alive() or process.exitcode != 0:
                    raise CompactWorldAtlasReducerError(
                        "parallel reducer worker did not exit cleanly"
                    )
            for world_id in self.world_ids:
                if world_id not in completed_results or world_id not in completed_ticks:
                    raise CompactWorldAtlasReducerError(
                        "parallel reducer did not return every logical world"
                    )
                self.worlds_by_cell[world_id].clear()
                self.worlds_by_cell[world_id].update(completed_results[world_id])
                self.reconstruction_by_cell[world_id]["next_tick"] = completed_ticks[
                    world_id
                ]
        except BaseException:
            self.terminate()
            raise
        self._close_queues()

    def _close_queues(self) -> None:
        if self.closed:
            return
        self.closed = True
        for command_queue in self.command_queues:
            command_queue.close()
        self.result_queue.close()

    def terminate(self) -> None:
        if self.closed:
            return
        _terminate_parallel_workers(self.processes)
        self._close_queues()


def _reduce_dispatches_parallel(
    dispatch_pointers: Sequence[Mapping[str, Any]],
    *,
    world_ids: tuple[str, ...],
    worlds_by_cell: Mapping[str, dict[str, Any]],
    reconstruction_by_cell: Mapping[str, dict[str, Any]],
    max_workers: int,
) -> None:
    """Reduce disjoint logical-world shards with one shared frame per dispatch."""

    worker_count = min(max_workers, len(world_ids), os.cpu_count() or 1)
    if worker_count < 2 and len(world_ids) > 1:
        for world_id in world_ids:
            _initialize_reconstruction(
                reconstruction_by_cell[world_id],
                result=worlds_by_cell[world_id],
            )
        for pointer in dispatch_pointers:
            _reduce_dispatch(
                pointer,
                worlds_by_cell=worlds_by_cell,
                reconstruction_by_cell=reconstruction_by_cell,
            )
        return
    pool = _ParallelReducerPoolV1(
        world_ids=world_ids,
        worlds_by_cell=worlds_by_cell,
        reconstruction_by_cell=reconstruction_by_cell,
        max_workers=max_workers,
        initialize_at_start=True,
        output_root=None,
    )
    try:
        for pointer in dispatch_pointers:
            pool.submit_dispatch(
                _prepare_dispatch(
                    pointer,
                    reconstruction_by_cell=reconstruction_by_cell,
                )
            )
        pool.finish()
    except BaseException:
        pool.terminate()
        raise


def _load_capture_index_v1(
    value: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    return (
        dict(value)
        if isinstance(value, Mapping)
        else _read_json(
            Path(value).expanduser().resolve(),
            label="capture index",
        )
    )


def _load_world_bundle_v1(
    value: str | Path | ScienceWorldBundleV1,
) -> ScienceWorldBundleV1:
    return (
        value
        if isinstance(value, ScienceWorldBundleV1)
        else ScienceWorldBundleV1.from_dict(
            _read_json(
                Path(value).expanduser().resolve(),
                label="world bundle",
            )
        )
    )


def _validate_capture_index_v1(
    index: Mapping[str, Any],
    bundle: ScienceWorldBundleV1,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    if index.get("schema_version") != "eiel_cuda_v2_compact_science_capture_index_v1":
        raise CompactWorldAtlasReducerError("compact capture index schema differs")
    if index.get("dropped_record_count") != 0:
        raise CompactWorldAtlasReducerError(
            "compact capture index reports dropped rows"
        )
    index_body = {
        key: value
        for key, value in index.items()
        if key not in {"index_sha256", "path"}
    }
    if index.get("index_sha256") != canonical_sha256(index_body).hex:
        raise CompactWorldAtlasReducerError("compact capture index digest differs")
    if index.get("ordered_cell_ids") != [world.world_id for world in bundle.worlds]:
        raise CompactWorldAtlasReducerError(
            "compact physical order differs from the exact bundle world order"
        )
    dispatches = index.get("dispatch_artifacts")
    snapshots = index.get("snapshot_artifacts")
    if (
        not isinstance(dispatches, list)
        or not isinstance(snapshots, list)
        or any(not isinstance(pointer, Mapping) for pointer in dispatches)
        or any(not isinstance(pointer, Mapping) for pointer in snapshots)
    ):
        raise CompactWorldAtlasReducerError(
            "compact capture index artifact ownership is invalid"
        )
    return tuple(dispatches), tuple(snapshots)


def _finalize_world_atlas_v1(
    index: Mapping[str, Any],
    bundle: ScienceWorldBundleV1,
    *,
    results: Mapping[str, dict[str, Any]],
    reconstructions: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    terminal = index.get("terminal_next_absolute_tick")
    if type(terminal) is not int or terminal < 1:
        raise CompactWorldAtlasReducerError("compact terminal tick is invalid")
    rows = []
    for world in bundle.worlds:
        result = results[world.world_id]
        reconstruction = reconstructions[world.world_id]
        semantic_time = world.semantic_inputs.to_value()["time"]
        expected_terminal = int(semantic_time["total_ticks"])
        expected_snapshots = list(world.acquisition.to_value()["snapshot_ticks"])
        actual_snapshots = sorted(reconstruction["snapshots"])
        if (
            terminal != expected_terminal
            or reconstruction["next_tick"] != terminal
            or result["integrity"]["dispatch_tick_count"]
            != terminal - reconstruction["start_next_tick"]
            or result["reconstructed_birth_count"] != result["birth_count"]
            or actual_snapshots != expected_snapshots
        ):
            raise CompactWorldAtlasReducerError(
                f"logical world {world.world_id} terminal coverage differs"
            )
        if reconstruction["hereditary_contract"] is not None:
            native_genotype_result = reconstruction.get(
                "native_hereditary_receipt"
            )
            if isinstance(native_genotype_result, Mapping):
                native_groups = native_genotype_result.get("groups")
                if not isinstance(native_groups, list):
                    raise CompactWorldAtlasReducerError(
                        "native hereditary attribution groups are absent"
                    )
                metric_totals: dict[str, dict[str, int]] = {}
                metric_counts: dict[str, int] = {}
                for group in native_groups:
                    if not isinstance(group, Mapping):
                        raise CompactWorldAtlasReducerError(
                            "native hereditary attribution group is invalid"
                        )
                    identity = group.get("genome_state_id")
                    raw_metrics = group.get("metric_totals")
                    agent_count = group.get("agent_count")
                    death_count = group.get("death_count")
                    terminal_alive_count = group.get("terminal_alive_count")
                    if (
                        not isinstance(identity, str)
                        or len(identity) != 71
                        or not identity.startswith("sha256:")
                        or any(
                            character not in "0123456789abcdef"
                            for character in identity[7:]
                        )
                        or identity in metric_totals
                        or not isinstance(raw_metrics, Mapping)
                        or set(raw_metrics)
                        != set(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
                        or any(type(value) is not int for value in raw_metrics.values())
                        or type(agent_count) is not int
                        or agent_count < 0
                        or type(death_count) is not int
                        or type(terminal_alive_count) is not int
                        or death_count + terminal_alive_count != agent_count
                    ):
                        raise CompactWorldAtlasReducerError(
                            "native hereditary attribution group identity or counts "
                            "are invalid"
                        )
                    metric_totals[identity] = dict(raw_metrics)
                    metric_counts[identity] = agent_count
                reconstruction["hereditary_metric_totals"] = metric_totals
                reconstruction["hereditary_metric_agent_counts"] = metric_counts
            else:
                for record in tuple(reconstruction["active"].values()):
                    _record_hereditary_metrics_v1(
                        reconstruction, record, record.latest_metrics
                    )
            trajectory = []
            for milestone in sorted(
                result["milestones"], key=lambda row: int(row["absolute_tick"])
            ):
                projection = milestone.get("hereditary_frequency_trajectory")
                if isinstance(projection, Mapping):
                    trajectory.append(dict(projection))
            metric_totals = reconstruction["hereditary_metric_totals"]
            metric_counts = reconstruction["hereditary_metric_agent_counts"]
            funnel = {}
            for identity in sorted(metric_totals):
                metrics = dict(metric_totals[identity])
                metrics["agent_count"] = metric_counts.get(identity, 0)
                eligible = metrics["reproduction_eligible_count"]
                births = metrics["births_as_parent_count"]
                metrics["reproductive_rate"] = (
                    births / eligible if eligible else None
                )
                funnel[identity] = metrics
            spool = reconstruction["hereditary_birth_spool"]
            stored_artifact = reconstruction.get("hereditary_birth_artifact")
            if isinstance(spool, _HereditaryBirthRecordSpoolV1):
                birth_artifact = spool.seal()
            elif isinstance(stored_artifact, Mapping):
                birth_artifact = dict(stored_artifact)
            else:
                raise CompactWorldAtlasReducerError(
                    "full-hereditary attribution birth artifact is absent"
                )
            world_id = world.world_id
            world_record_counts = birth_artifact.get("world_record_counts")
            if (
                not isinstance(world_record_counts, Mapping)
                or type(world_record_counts.get(world_id, 0)) is not int
                or world_record_counts.get(world_id, 0)
                != result["integrity"]["hereditary_parent_child_records"]
            ):
                raise CompactWorldAtlasReducerError(
                    "full-hereditary birth artifact count differs"
                )
            world_birth_artifact = {
                **birth_artifact,
                "world_id": world_id,
                "world_record_count": world_record_counts.get(world_id, 0),
            }
            birth_parent_child_records: object = world_birth_artifact
            if (
                isinstance(spool, _HereditaryBirthRecordSpoolV1)
                and spool.record_count <= _HEREDITARY_BIRTH_RECORD_INLINE_LIMIT_V1
            ):
                birth_parent_child_records = spool.records_for_world(world_id)
            elif (
                isinstance(stored_artifact, Mapping)
                and stored_artifact.get("record_count")
                <= _HEREDITARY_BIRTH_RECORD_INLINE_LIMIT_V1
            ):
                birth_parent_child_records = (
                    _records_for_hereditary_birth_artifact_world_v1(
                        stored_artifact, world_id
                    )
                )
            focal_ids = reconstruction["hereditary_focal_ids"]
            result["hereditary_attribution"] = {
                "schema_version": "eiel_science_hereditary_attribution_v1",
                "identity_mode": "full_genome_state_id_v1",
                "tracked_identity_mode": "exact_tracked_genotype_state_id_v1",
                "reproductive_attribution_group_scope": (
                    "requested_focal_ids_v1"
                    if native_genotype_result is not None
                    and native_genotype_result.get("group_scope")
                    == "requested_focal_ids_v1"
                    else "all_tracked_identities_v1"
                ),
                "tracked_policy_loci": list(reconstruction["policy_loci"]),
                "tracked_trait_loci": list(reconstruction["trait_loci"]),
                "focal_tracked_genotype_state_ids": list(focal_ids),
                "frequency_trajectory": trajectory,
                "reproduction_funnel": funnel,
                "birth_parent_child_records": birth_parent_child_records,
                "descendant_birth_counts_by_parent_tracked_identity": dict(
                    sorted(reconstruction["hereditary_descendant_counts"].items())
                ),
                "lineage": {
                    "completed_agent_count": reconstruction[
                        "hereditary_completed_agents"
                    ],
                    "maximum_lineage_depth": result["maximum_lineage_depth"],
                },
                "integrity": {
                    "full_identity_checks": result["integrity"][
                        "hereditary_full_identity_checks"
                    ],
                    "parent_child_record_count": result["integrity"][
                        "hereditary_parent_child_records"
                    ],
                    "funnel_agent_count": sum(metric_counts.values()),
                    "funnel_reconciled": (
                        reconstruction["hereditary_completed_agents"]
                        == result["death_count"]
                        + int(
                            reconstruction.get(
                                "native_terminal_active_count",
                                len(reconstruction["active"]),
                            )
                        )
                    ),
                },
            }
            result["integrity"]["hereditary_funnel_reconciled"] = bool(
                result["hereditary_attribution"]["integrity"]["funnel_reconciled"]
            )
            if not result["integrity"]["hereditary_funnel_reconciled"]:
                raise CompactWorldAtlasReducerError(
                    "full-hereditary lifetime funnel does not reconcile"
                )
        result["milestones"].sort(key=lambda row: int(row["absolute_tick"]))
        result["result_sha256"] = canonical_sha256(result).to_dict()
        rows.append(result)
    body = {
        "schema_version": COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID,
        "reducer_id": COMPACT_WORLD_ATLAS_REDUCER_ID,
        "reducer_contract_version": str(COMPACT_WORLD_ATLAS_CONTRACT_VERSION),
        "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
        "capture_index_sha256": index["index_sha256"],
        "logical_world_count": len(rows),
        "physical_batch_count": 1,
        "physical_composition_used": False,
        "world_results": rows,
    }
    return {**body, "result_sha256": canonical_sha256(body).to_dict()}


def _artifact_pointer_key_v1(
    pointer: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, str]:
    path = pointer.get("path")
    artifact_sha256 = pointer.get("artifact_sha256")
    if (
        not isinstance(path, str)
        or not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
    ):
        raise CompactWorldAtlasReducerError(f"{label} pointer identity is invalid")
    return str(Path(path).expanduser().resolve()), artifact_sha256


def _native_mutation_segments_v1(world: Any) -> tuple[tuple[int, int], ...]:
    """Compile one integer control into ordered native tick segments."""

    semantic = world.semantic_inputs.to_value()
    time_contract = semantic["time"]
    start = int(time_contract.get("start_next_tick", 0))
    terminal = int(time_contract["total_ticks"])
    boundaries = {start}
    row = next(
        (
            value
            for value in semantic["control_bindings"]
            if value.get("control_id") == "mutation.step"
        ),
        None,
    )
    if row is not None:
        schedule = row["schedule"]
        kind = schedule["kind"]
        if kind == "piecewise_constant_absolute_ticks":
            boundaries.update(
                int(segment["start_absolute_tick"]) for segment in schedule["segments"]
            )
        elif kind == "switch_absolute_tick":
            boundaries.add(int(schedule["absolute_tick"]))
        elif kind == "pulse_absolute_ticks":
            pulse_start = int(schedule["start_absolute_tick"])
            boundaries.update(
                (pulse_start, pulse_start + int(schedule["duration_ticks"]))
            )
        elif kind != "static":
            raise CompactWorldAtlasReducerError(
                "mutation.step uses an unsupported native schedule"
            )
    segments: list[tuple[int, int]] = []
    for boundary in sorted(
        {start, *(value for value in boundaries if start < value < terminal)}
    ):
        step = _control_value_at_tick(world, "mutation.step", boundary, default=1)
        if type(step) is not int or not 1 <= step < 2**31:
            raise CompactWorldAtlasReducerError(
                "resolved mutation step is outside the native integer ABI"
            )
        if not segments or segments[-1][1] != step:
            segments.append((boundary, step))
    return tuple(segments)


class CompactWorldAtlasIncrementalReducerV1:
    """Consume sealed snapshots and dispatches while their campaign continues.

    Prefix work is provisional.  :meth:`finalize` reconciles every consumed
    artifact against the immutable terminal capture index before producing a
    scientific result, so early execution changes latency but not authority.
    """

    def __init__(
        self,
        world_bundle: str | Path | ScienceWorldBundleV1,
        *,
        max_workers: int = 1,
        native_artifact: NativeCompactReducerArtifactV1 | None = None,
        output_root: str | Path | None = None,
        strategy_early_life_tick: int = 10_000,
        strategy_pressure_factor: str = "pressure_regime",
    ) -> None:
        if type(max_workers) is not int or max_workers < 1:
            raise CompactWorldAtlasReducerError(
                "max_workers must be a positive integer"
            )
        if type(strategy_early_life_tick) is not int or strategy_early_life_tick < 1:
            raise CompactWorldAtlasReducerError(
                "strategy_early_life_tick must be a positive integer"
            )
        if not isinstance(strategy_pressure_factor, str) or not strategy_pressure_factor:
            raise CompactWorldAtlasReducerError(
                "strategy_pressure_factor must be a nonempty string"
            )
        self.bundle = _load_world_bundle_v1(world_bundle)
        self.world_ids = tuple(world.world_id for world in self.bundle.worlds)
        self._world_index_by_id = {
            world_id: index for index, world_id in enumerate(self.world_ids)
        }
        self.results = {
            world.world_id: _blank_world(world) for world in self.bundle.worlds
        }
        self.reconstructions = {
            world.world_id: _blank_reconstruction(world) for world in self.bundle.worlds
        }
        starts = {
            int(world.semantic_inputs.to_value()["time"].get("start_next_tick", 0))
            for world in self.bundle.worlds
        }
        if len(starts) != 1:
            raise CompactWorldAtlasReducerError(
                "incremental physical reduction requires one absolute start frontier"
            )
        self.next_absolute_tick = starts.pop()
        self._dispatch_keys: list[tuple[str, str]] = []
        self._snapshot_keys: dict[tuple[str, str], int] = {}
        self._closed = False
        self._finalized = False
        self._native_initialized_worlds: set[str] = set()
        self._native_reconciled_snapshots: set[tuple[str, int]] = set()
        self._native_state: NativeCompactReducerStateV1 | None = None
        self._native_max_workers = max_workers
        self._strategy_early_life_tick = strategy_early_life_tick
        self._strategy_pressure_factor = strategy_pressure_factor
        self._strategy_result: dict[str, Any] | None = None
        self._native_hereditary_worlds: set[str] = set()
        hereditary_worlds = {
            world_id
            for world_id, reconstruction in self.reconstructions.items()
            if reconstruction["hereditary_contract"] is not None
        }
        if native_artifact is not None:
            try:
                self._native_state = create_native_compact_reducer_state_v1(
                    native_artifact,
                    resident_capacities=tuple(
                        int(
                            world.semantic_inputs.to_value()["world"][
                                "resident_capacity"
                            ]
                        )
                        for world in self.bundle.worlds
                    ),
                    start_next_absolute_ticks=tuple(
                        int(
                            world.semantic_inputs.to_value()["time"].get(
                                "start_next_tick", 0
                            )
                        )
                        for world in self.bundle.worlds
                    ),
                    mutation_segments=tuple(
                        _native_mutation_segments_v1(world)
                        for world in self.bundle.worlds
                    ),
                    max_worker_threads=max_workers,
                )
            except Exception as exc:
                raise CompactWorldAtlasReducerError(
                    f"native compact reducer initialization failed: {exc}"
                ) from exc
        configured_strategy_worlds = {
            world_id
            for world_id, reconstruction in self.reconstructions.items()
            if reconstruction["strategy_contract"] is not None
        }
        if self._native_state is not None and configured_strategy_worlds and (
            configured_strategy_worlds != set(self.world_ids)
        ):
            self._native_state.close()
            self._native_state = None
            raise CompactWorldAtlasReducerError(
                "fused native strategy attribution must own the complete physical batch"
            )
        self._native_strategy_enabled = (
            self._native_state is not None
            and configured_strategy_worlds == set(self.world_ids)
        )
        if self._native_state is not None:
            try:
                for world_index, world_id in enumerate(self.world_ids):
                    reconstruction = self.reconstructions[world_id]
                    contract = reconstruction["strategy_contract"]
                    strategy_count = 0
                    strategy_policy_masks: dict[str, int] = {}
                    threshold_ticks: tuple[int, ...] = ()
                    if isinstance(contract, Mapping):
                        strategy_ids = tuple(contract["strategy_ids"])
                        strategy_index = {
                            strategy_id: index
                            for index, strategy_id in enumerate(strategy_ids)
                        }
                        strategy_count = len(strategy_ids)
                        strategy_policy_masks = {
                            policy_id: 1 << strategy_index[strategy_id]
                            for policy_id, strategy_id in contract[
                                "policy_id_to_strategy_id"
                            ].items()
                        }
                        terminal = int(
                            reconstruction["world"].semantic_inputs.to_value()[
                                "time"
                            ]["total_ticks"]
                        )
                        threshold_ticks = strategy_threshold_ticks_v1(
                            terminal, self._strategy_early_life_tick
                        )
                    self._native_state.configure_snapshot_world(
                        world_index,
                        cell_id=world_id,
                        tracked_policy_loci=reconstruction["policy_loci"],
                        tracked_trait_loci=reconstruction["trait_loci"],
                        strategy_count=strategy_count,
                        strategy_policy_masks=strategy_policy_masks,
                        threshold_ticks=threshold_ticks,
                        early_life_tick=(
                            self._strategy_early_life_tick
                            if strategy_count
                            else 0
                        ),
                    )
                    if world_id in hereditary_worlds:
                        self._native_state.configure_genotype_attribution(
                            world_index,
                            focal_ids=reconstruction["hereditary_focal_ids"],
                        )
                if hereditary_worlds:
                    _configure_native_hereditary_birth_output_v1(
                        self.reconstructions,
                        world_ids=self.world_ids,
                        output_root=output_root,
                    )
                    for world_index, world_id in enumerate(self.world_ids):
                        path = self.reconstructions[world_id].get(
                            "native_hereditary_birth_output_path"
                        )
                        if path is not None:
                            self._native_state.configure_hereditary_birth_output(
                                world_index, path
                            )
                    self._native_hereditary_worlds = set(hereditary_worlds)
            except Exception as exc:
                self._native_state.close()
                self._native_state = None
                raise CompactWorldAtlasReducerError(
                    f"native snapshot configuration failed: {exc}"
                ) from exc
        self._pool = (
            None
            if max_workers == 1 or self._native_state is not None
            else _ParallelReducerPoolV1(
                world_ids=self.world_ids,
                worlds_by_cell=self.results,
                reconstruction_by_cell=self.reconstructions,
                max_workers=max_workers,
                initialize_at_start=False,
                output_root=output_root,
            )
        )
        if self._pool is None and self._native_state is None:
            _configure_hereditary_birth_spool_v1(
                self.reconstructions,
                world_ids=self.world_ids,
                output_root=output_root,
            )

    @property
    def execution_route(self) -> str:
        return "native_cpp_v1" if self._native_state is not None else "python_v1"

    @property
    def consumed_dispatch_count(self) -> int:
        return len(self._dispatch_keys)

    @property
    def consumed_snapshot_count(self) -> int:
        return len(self._snapshot_keys)

    @property
    def consumed_artifact_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._dispatch_keys) | frozenset(self._snapshot_keys)

    @property
    def strategy_attribution_result(self) -> dict[str, Any] | None:
        return self._strategy_result

    def _require_open(self) -> None:
        if self._closed or self._finalized:
            raise CompactWorldAtlasReducerError(
                "incremental compact reducer is already closed"
            )

    def _reduce_native_snapshot(
        self,
        pointer: Mapping[str, Any],
    ) -> tuple[int, tuple[str, ...]]:
        if self._native_state is None:
            raise CompactWorldAtlasReducerError("native reducer state is absent")
        prepared = _prepare_snapshot(pointer)
        native = self._native_state.process_snapshot(
            absolute_tick=prepared.absolute_tick,
            canonical_json_lines=prepared.uncompressed,
            row_count=prepared.row_count,
        )
        native_worlds = native.get("worlds")
        if not isinstance(native_worlds, list):
            raise CompactWorldAtlasReducerError(
                "native snapshot world projection is absent"
            )
        captured_worlds = prepared.manifest.get("captured_worlds")
        if (
            not isinstance(captured_worlds, list)
            or prepared.manifest.get("captured_world_count")
            != len(captured_worlds)
            or any(not isinstance(row, Mapping) for row in captured_worlds)
        ):
            raise CompactWorldAtlasReducerError(
                "snapshot logical-world ownership differs"
            )
        captured_by_cell = {
            str(row["cell_id"]): row
            for row in captured_worlds
            if isinstance(row.get("cell_id"), str)
        }
        projected_ids: list[str] = []
        projected_indices: set[int] = set()
        for projection in native_worlds:
            if not isinstance(projection, Mapping) or set(projection) != {
                "active_agent_count",
                "active_by_strategy",
                "active_lifetime_metric_totals",
                "energy",
                "genome_catalog_size",
                "genome_reconstruction",
                "health",
                "initial_strategy_masks",
                "injury",
                "world_index",
                "world_state_row",
            }:
                raise CompactWorldAtlasReducerError(
                    "native snapshot projection contract drifted"
                )
            world_index = projection["world_index"]
            if (
                type(world_index) is not int
                or not 0 <= world_index < len(self.world_ids)
                or world_index in projected_indices
            ):
                raise CompactWorldAtlasReducerError(
                    "native snapshot world index drifted"
                )
            projected_indices.add(world_index)
            world_id = self.world_ids[world_index]
            projected_ids.append(world_id)
            reconstruction = self.reconstructions[world_id]
            result = self.results[world_id]
            world = reconstruction["world"]
            owner = captured_by_cell.get(world_id)
            active_count = projection["active_agent_count"]
            catalog_size = projection["genome_catalog_size"]
            totals = projection["active_lifetime_metric_totals"]
            world_state_row = projection["world_state_row"]
            if (
                not isinstance(owner, Mapping)
                or owner.get("science_world_sha256")
                != world.semantic_world_sha256.to_dict()
                or owner.get("science_acquisition_sha256")
                != world.acquisition_sha256.to_dict()
                or owner.get("active_agent_count") != active_count
                or type(active_count) is not int
                or active_count < 0
                or type(catalog_size) is not int
                or catalog_size < 0
                or not isinstance(totals, list)
                or len(totals) != len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
                or any(type(value) is not int for value in totals)
                or not isinstance(world_state_row, Mapping)
                or world_state_row.get("row_kind") != "world_state"
                or world_state_row.get("cell_id") != world_id
                or world_state_row.get("absolute_tick")
                != prepared.absolute_tick
            ):
                raise CompactWorldAtlasReducerError(
                    "native snapshot world identity differs from the bundle"
                )
            expected_ticks = world.acquisition.to_value()["snapshot_ticks"]
            if prepared.absolute_tick not in expected_ticks:
                raise CompactWorldAtlasReducerError(
                    "snapshot tick is not owned by the logical world"
                )
            if prepared.absolute_tick in reconstruction["snapshots"]:
                raise CompactWorldAtlasReducerError(
                    "snapshot duplicates a logical world milestone"
                )
            genome = projection["genome_reconstruction"]
            if (
                not isinstance(genome, Mapping)
                or set(genome) != {
                    "active_agent_count",
                    "maximum_active_lineage_depth",
                    "mutation_lineage_active_count",
                    "state_identity_check_count",
                    "tracked_policy_locus_histograms",
                    "tracked_trait_locus_histograms",
                }
                or genome.get("active_agent_count") != active_count
                or genome.get("state_identity_check_count") != active_count
                or type(genome.get("maximum_active_lineage_depth")) is not int
                or genome["maximum_active_lineage_depth"] < 0
                or type(genome.get("mutation_lineage_active_count")) is not int
                or not 0 <= genome["mutation_lineage_active_count"] <= active_count
            ):
                raise CompactWorldAtlasReducerError(
                    "native genome reconstruction projection drifted"
                )
            policy_histograms = _native_snapshot_histograms_v1(
                genome["tracked_policy_locus_histograms"],
                expected_loci=reconstruction["policy_loci"],
            )
            trait_histograms = _native_snapshot_histograms_v1(
                genome["tracked_trait_locus_histograms"],
                expected_loci=reconstruction["trait_loci"],
            )
            for histograms in (policy_histograms, trait_histograms):
                if any(
                    sum(histogram.values()) != active_count
                    for alleles in histograms.values()
                    for histogram in alleles
                ):
                    raise CompactWorldAtlasReducerError(
                        "native snapshot histogram population differs"
                    )
            milestone = {
                "absolute_tick": prepared.absolute_tick,
                "world_state": {
                    key: value
                    for key, value in world_state_row.items()
                    if key
                    not in {"row_kind", "cell_id", "world_uid", "absolute_tick"}
                },
                "active_agent_count": active_count,
                "active_lifetime_metric_totals": dict(
                    zip(
                        COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
                        totals,
                        strict=True,
                    )
                ),
                "genome_catalog_size": catalog_size,
                "genome_reconstruction": {
                    "active_agent_count": active_count,
                    "state_identity_check_count": active_count,
                    "mutation_lineage_active_count": genome[
                        "mutation_lineage_active_count"
                    ],
                    "maximum_active_lineage_depth": genome[
                        "maximum_active_lineage_depth"
                    ],
                    "tracked_policy_locus_histograms": policy_histograms,
                    "tracked_trait_locus_histograms": trait_histograms,
                },
            }
            if world_id in self._native_hereditary_worlds:
                active_records = self._native_state.export_active(world_index)
                if len(active_records) != active_count:
                    raise CompactWorldAtlasReducerError(
                        "native hereditary active-state count differs"
                    )
                active = {
                    record.agent_id: _ActiveGenomeV1(
                        encoded=record.genome,
                        generation=record.generation,
                        mutation_lineage=record.mutation_lineage,
                        born=record.born,
                        reproduced=record.reproduced,
                    )
                    for record in active_records
                }
                if len(active) != active_count:
                    raise CompactWorldAtlasReducerError(
                        "native hereditary active-state identities are not unique"
                    )
                hereditary_projection = _hereditary_snapshot_projection_v1(
                    reconstruction, active
                )
                if hereditary_projection is None:
                    raise CompactWorldAtlasReducerError(
                        "native hereditary projection unexpectedly absent"
                    )
                hereditary_projection["absolute_tick"] = prepared.absolute_tick
                milestone["hereditary_frequency_trajectory"] = (
                    hereditary_projection
                )
                result["integrity"]["hereditary_full_identity_checks"] += (
                    active_count
                )
            reconstruction["snapshots"][prepared.absolute_tick] = {
                "native_reconciled": True
            }
            reconstruction["milestones"][prepared.absolute_tick] = milestone
            result["milestones"].append(milestone)
            result["integrity"]["snapshot_count"] += 1
            result["integrity"]["genome_state_identity_checks"] += active_count
            contract = reconstruction["strategy_contract"]
            if isinstance(contract, Mapping):
                strategy_ids = tuple(contract["strategy_ids"])
                active_by_strategy = projection["active_by_strategy"]
                initial_masks = projection["initial_strategy_masks"]
                if (
                    not isinstance(active_by_strategy, list)
                    or len(active_by_strategy) != len(strategy_ids)
                    or any(
                        type(value) is not int or value < 0
                        for value in active_by_strategy
                    )
                    or not isinstance(initial_masks, list)
                    or any(
                        type(mask) is not int
                        or mask <= 0
                        or mask >> len(strategy_ids)
                        for mask in initial_masks
                    )
                ):
                    raise CompactWorldAtlasReducerError(
                        "native strategy snapshot projection drifted"
                    )
                fields = milestone["world_state"].get("qualified_field_values")
                if not isinstance(fields, Mapping):
                    raise CompactWorldAtlasReducerError(
                        "strategy snapshot world fields are absent"
                    )
                reconstruction["strategy_snapshots"][prepared.absolute_tick] = {
                    "absolute_tick": prepared.absolute_tick,
                    "active_count": active_count,
                    "population_count": fields.get(
                        "population_count", active_count
                    ),
                    "birth_count": fields.get("birth_count"),
                    "death_count": fields.get("death_count"),
                    "mutation_count": fields.get("mutation_count"),
                    "resource_total": fields.get("resource_total"),
                    "resource_mean": fields.get("resource_mean"),
                    "resource_min": fields.get("resource_min"),
                    "resource_max": fields.get("resource_max"),
                    "resource_nonzero_count": fields.get(
                        "resource_nonzero_count"
                    ),
                    "energy": _native_snapshot_numeric_stats_v1(
                        projection["energy"]
                    ),
                    "health": _native_snapshot_numeric_stats_v1(
                        projection["health"]
                    ),
                    "injury": _native_snapshot_numeric_stats_v1(
                        projection["injury"]
                    ),
                    "active_by_strategy": {
                        strategy_id: active_by_strategy[index]
                        for index, strategy_id in enumerate(strategy_ids)
                        if active_by_strategy[index] > 0
                    },
                }
                start = int(reconstruction["start_next_tick"])
                if prepared.absolute_tick == start:
                    if len(initial_masks) != active_count:
                        raise CompactWorldAtlasReducerError(
                            "native strategy founder projection drifted"
                        )
                    reconstruction["strategy_initial_masks"] = tuple(
                        initial_masks
                    )
                elif initial_masks:
                    raise CompactWorldAtlasReducerError(
                        "native strategy founders leaked past the initial snapshot"
                    )
            self._native_initialized_worlds.add(world_id)
            self._native_reconciled_snapshots.add(
                (world_id, prepared.absolute_tick)
            )
        if set(projected_ids) != set(captured_by_cell):
            raise CompactWorldAtlasReducerError(
                "snapshot logical-world ownership differs"
            )
        return prepared.absolute_tick, tuple(projected_ids)

    def _record_native_strategy_snapshot(
        self,
        world_id: str,
        absolute_tick: int,
        strategy_masks: Sequence[int],
    ) -> None:
        reconstruction = self.reconstructions[world_id]
        contract = reconstruction["strategy_contract"]
        summary = reconstruction["strategy_snapshots"].get(absolute_tick)
        if not isinstance(contract, Mapping) or not isinstance(summary, dict):
            raise CompactWorldAtlasReducerError(
                "native strategy snapshot projection is absent"
            )
        strategy_ids = tuple(contract["strategy_ids"])
        counts = [0] * len(strategy_ids)
        for mask in strategy_masks:
            if type(mask) is not int or mask <= 0 or mask >> len(strategy_ids):
                raise CompactWorldAtlasReducerError(
                    "native strategy snapshot mask is invalid"
                )
            for index in range(len(strategy_ids)):
                if mask & (1 << index):
                    counts[index] += 1
        summary["active_by_strategy"] = {
            strategy_id: counts[index]
            for index, strategy_id in enumerate(strategy_ids)
            if counts[index] > 0
        }

    def _reconcile_native_snapshot(self, world_id: str, absolute_tick: int) -> None:
        if self._native_state is None:
            raise CompactWorldAtlasReducerError("native reducer state is absent")
        key = (world_id, absolute_tick)
        if key in self._native_reconciled_snapshots:
            return
        world_index = self._world_index_by_id[world_id]
        reconstruction = self.reconstructions[world_id]
        snapshot = reconstruction["snapshots"][absolute_tick]
        if self._native_strategy_enabled:
            self._native_state.observe_strategy_snapshot(
                world_index,
                active_agent_ids=tuple(snapshot["strategy_agent_ids"]),
                active_metric_rows=bytes(snapshot["strategy_metric_rows"]),
            )
        records = self._native_state.export_active(world_index)
        active = {
            record.agent_id: _ActiveGenomeV1(
                encoded=record.genome,
                generation=record.generation,
                mutation_lineage=record.mutation_lineage,
                born=record.born,
                reproduced=record.reproduced,
            )
            for record in records
        }
        _snapshot_distribution(
            reconstruction,
            result=self.results[world_id],
            absolute_tick=absolute_tick,
            active_override=active,
        )
        if self._native_strategy_enabled:
            self._record_native_strategy_snapshot(
                world_id,
                absolute_tick,
                tuple(record.strategy_mask for record in records),
            )
        snapshot.clear()
        snapshot["native_reconciled"] = True
        self._native_reconciled_snapshots.add(key)

    def _activate_native_snapshot(self, world_id: str, absolute_tick: int) -> None:
        if self._native_state is None:
            raise CompactWorldAtlasReducerError("native reducer state is absent")
        reconstruction = self.reconstructions[world_id]
        start = int(reconstruction["start_next_tick"])
        if world_id not in self._native_initialized_worlds:
            if absolute_tick != start:
                return
            _initialize_reconstruction(
                reconstruction,
                result=self.results[world_id],
            )
            active: Mapping[int, _ActiveGenomeV1] = reconstruction["active"]
            ordered = sorted(active.items())
            self._native_state.initialize_world(
                self._world_index_by_id[world_id],
                agent_ids=tuple(agent_id for agent_id, _ in ordered),
                packed_genomes=b"".join(record.encoded for _, record in ordered),
            )
            snapshot = reconstruction["snapshots"][absolute_tick]
            if self._native_strategy_enabled:
                contract = reconstruction["strategy_contract"]
                if not isinstance(contract, Mapping):
                    raise CompactWorldAtlasReducerError(
                        "native strategy contract is absent"
                    )
                strategy_ids = tuple(contract["strategy_ids"])
                strategy_index = {
                    strategy_id: index
                    for index, strategy_id in enumerate(strategy_ids)
                }
                policy_mapping = contract["policy_id_to_strategy_id"]
                founder_ids = tuple(snapshot["strategy_agent_ids"])
                if set(founder_ids) != set(active):
                    raise CompactWorldAtlasReducerError(
                        "native strategy founder ownership differs"
                    )
                masks: list[int] = []
                for agent_id in founder_ids:
                    policy_id = snapshot["strategy_policy_ids"].get(agent_id)
                    parent_ids = snapshot["strategy_parent_ids"].get(agent_id)
                    if (
                        not isinstance(policy_id, str)
                        or policy_id not in policy_mapping
                        or parent_ids != (0, 0)
                    ):
                        raise CompactWorldAtlasReducerError(
                            "native strategy founder identity differs"
                        )
                    strategy_id = policy_mapping[policy_id]
                    masks.append(1 << strategy_index[strategy_id])
                terminal = int(
                    reconstruction["world"].semantic_inputs.to_value()["time"][
                        "total_ticks"
                    ]
                )
                self._native_state.configure_strategy(
                    self._world_index_by_id[world_id],
                    strategy_count=len(strategy_ids),
                    founder_agent_ids=founder_ids,
                    founder_strategy_masks=tuple(masks),
                    threshold_ticks=strategy_threshold_ticks_v1(
                        terminal, self._strategy_early_life_tick
                    ),
                    early_life_tick=self._strategy_early_life_tick,
                )
                self._native_state.observe_strategy_snapshot(
                    self._world_index_by_id[world_id],
                    active_agent_ids=founder_ids,
                    active_metric_rows=bytes(snapshot["strategy_metric_rows"]),
                )
                reconstruction["strategy_initial_masks"] = tuple(masks)
                self._record_native_strategy_snapshot(
                    world_id, absolute_tick, tuple(masks)
                )
            reconstruction["active"] = {}
            snapshot.clear()
            snapshot["native_reconciled"] = True
            self._native_initialized_worlds.add(world_id)
            self._native_reconciled_snapshots.add((world_id, absolute_tick))
            return
        next_tick = int(reconstruction["next_tick"])
        if next_tick > absolute_tick:
            raise CompactWorldAtlasReducerError(
                "snapshot arrived after the native reducer passed its absolute tick"
            )
        if next_tick == absolute_tick:
            self._reconcile_native_snapshot(world_id, absolute_tick)

    def _sync_native_results(self) -> None:
        if self._native_state is None:
            return
        for world_index, world_id in enumerate(self.world_ids):
            receipt = self._native_state.world_receipt(world_index)
            result = self.results[world_id]
            result["birth_count"] = receipt.birth_count
            result["death_count"] = receipt.death_count
            result["reconstructed_birth_count"] = (
                receipt.births.reconstructed_birth_count
            )
            result["mutation_origin_count"] = receipt.births.mutation_origin_count
            result["mutation_saturated_noop_count"] = (
                receipt.births.saturated_noop_count
            )
            result["mutation_origin_counts_by_linked_locus"] = list(
                receipt.births.mutation_origin_counts_by_linked_locus
            )
            result["mutation_delta_totals_by_linked_locus"] = list(
                receipt.births.mutation_delta_totals_by_linked_locus
            )
            result["recombination_birth_count"] = receipt.recombination_birth_count
            result["recombination_nonzero_pattern_count"] = (
                receipt.births.recombination_nonzero_pattern_count
            )
            result["recombination_choice_one_counts_by_block"] = [
                [
                    receipt.births.recombination_choice_one_counts[2 * block],
                    receipt.births.recombination_choice_one_counts[2 * block + 1],
                ]
                for block in range(16)
            ]
            result["carrier_reproduction_birth_count"] = (
                receipt.carrier_reproduction_birth_count
            )
            result["mutation_carrier_reproduction_birth_count"] = (
                receipt.mutation_carrier_reproduction_birth_count
            )
            result["unique_born_agents_reproduced_count"] = (
                receipt.unique_born_agents_reproduced_count
            )
            result["unique_mutation_carriers_reproduced_count"] = (
                receipt.unique_mutation_carriers_reproduced_count
            )
            result["maximum_lineage_depth"] = receipt.maximum_lineage_depth
            result["completed_lifetime_metric_totals"] = dict(
                zip(
                    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
                    receipt.completed_lifetime_metric_totals,
                    strict=True,
                )
            )
            result["integrity"]["dispatch_count"] = receipt.dispatch_count
            result["integrity"]["event_count"] = receipt.event_count
            result["integrity"]["dispatch_tick_count"] = receipt.dispatch_tick_count
            self.reconstructions[world_id]["next_tick"] = receipt.next_absolute_tick

    def _finalize_native_hereditary_attribution(self, *, terminal_tick: int) -> None:
        if self._native_state is None:
            return
        rows = [
            (world_index, world_id)
            for world_index, world_id in enumerate(self.world_ids)
            if world_id in self._native_hereditary_worlds
        ]
        if not rows:
            return

        def finalize_world(
            row: tuple[int, str],
        ) -> tuple[int, str, Any, dict[str, Any], dict[str, Any]]:
            world_index, world_id = row
            reconstruction = self.reconstructions[world_id]
            world_receipt = self._native_state.world_receipt(world_index)
            self._native_state.finalize_hereditary_birth_output(world_index)
            self._native_state.finalize_strategy(
                world_index, terminal_tick=terminal_tick
            )
            strategy_receipt = self._native_state.strategy_receipt(world_index)
            genotype_result = strategy_receipt.genotype_reproductive_attribution
            if not isinstance(genotype_result, Mapping):
                raise CompactWorldAtlasReducerError(
                    "native hereditary attribution result is absent"
                )
            totals = genotype_result.get("totals")
            focal_only = (
                genotype_result.get("group_scope")
                == "requested_focal_ids_v1"
            )
            if (
                not isinstance(totals, Mapping)
                or (
                    not focal_only
                    and (
                        totals.get("agent_count")
                        != world_receipt.death_count
                        + world_receipt.active_agent_count
                        or totals.get("offspring_birth_count")
                        != world_receipt.birth_count
                    )
                )
                or totals.get("parent_birth_event_count")
                is not None
                and type(totals.get("parent_birth_event_count")) is not int
            ):
                raise CompactWorldAtlasReducerError(
                    "native hereditary attribution totals do not reconcile"
                )
            artifact = _native_hereditary_birth_artifact_v1(
                reconstruction.get("native_hereditary_birth_output_path"),
                world_id=world_id,
                record_count=world_receipt.birth_count,
            )
            return world_index, world_id, world_receipt, dict(genotype_result), artifact

        worker_count = min(max(1, self._native_max_workers), len(rows))
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="native-hereditary-finalize",
        ) as executor:
            finalized = tuple(executor.map(finalize_world, rows))

        # Apply projections in canonical world order after concurrent native
        # finalization. This keeps result serialization deterministic while
        # allowing per-world gzip/fsync/hash work to overlap.
        for world_index, world_id, world_receipt, genotype_result, artifact in finalized:
            reconstruction = self.reconstructions[world_id]
            result = self.results[world_id]
            reconstruction["native_hereditary_receipt"] = dict(genotype_result)
            reconstruction["native_hereditary_birth_artifact"] = artifact
            reconstruction["hereditary_birth_artifact"] = artifact
            reconstruction["native_terminal_active_count"] = (
                world_receipt.active_agent_count
            )
            reconstruction["hereditary_completed_agents"] = (
                world_receipt.death_count + world_receipt.active_agent_count
            )
            descendant_counts = genotype_result.get(
                "descendant_birth_counts_by_parent_tracked_identity", {}
            )
            if not isinstance(descendant_counts, Mapping) or any(
                not isinstance(identity, str)
                or type(count) is not int
                or count < 0
                for identity, count in descendant_counts.items()
            ):
                raise CompactWorldAtlasReducerError(
                    "native hereditary descendant counts are invalid"
                )
            reconstruction["hereditary_descendant_counts"] = dict(
                descendant_counts
            )
            result["integrity"]["hereditary_parent_child_records"] = (
                world_receipt.birth_count
            )

    def consume_snapshot(self, pointer: Mapping[str, Any]) -> bool:
        """Consume one fully sealed snapshot; exact repeats are idempotent."""

        self._require_open()
        key = _artifact_pointer_key_v1(pointer, label="compact snapshot")
        if key in self._snapshot_keys:
            return False
        try:
            if self._native_state is not None:
                absolute_tick, cell_ids = self._reduce_native_snapshot(pointer)
            else:
                absolute_tick, cell_ids = _reduce_snapshot(
                    pointer,
                    worlds_by_cell=self.results,
                    reconstruction_by_cell=self.reconstructions,
                )
            if self._native_state is not None:
                pass
            elif self._pool is None:
                for world_id in cell_ids:
                    _activate_snapshot_frontier_v1(
                        self.reconstructions[world_id],
                        result=self.results[world_id],
                        absolute_tick=absolute_tick,
                    )
            else:
                self._pool.submit_snapshot(absolute_tick, cell_ids)
        except BaseException:
            self.close()
            raise
        self._snapshot_keys[key] = absolute_tick
        return True

    def consume_dispatch(self, pointer: Mapping[str, Any]) -> bool:
        """Consume the next sealed dispatch in absolute-tick order."""

        self._require_open()
        key = _artifact_pointer_key_v1(pointer, label="compact dispatch")
        if key in self._dispatch_keys:
            return False
        try:
            prepared = _prepare_dispatch(
                pointer,
                reconstruction_by_cell=self.reconstructions,
                scan_event_kinds=self._native_state is None,
            )
            if prepared.work.start != self.next_absolute_tick:
                raise CompactWorldAtlasReducerError(
                    "incremental compact dispatch is outside the next frontier"
                )
            if any(
                reconstruction["start_next_tick"] not in reconstruction["snapshots"]
                for reconstruction in self.reconstructions.values()
            ):
                raise CompactWorldAtlasReducerError(
                    "incremental reduction requires the sealed initial snapshot"
                )
            if self._native_state is not None:
                if set(self._native_initialized_worlds) != set(self.world_ids):
                    raise CompactWorldAtlasReducerError(
                        "native incremental reduction requires every initial world"
                    )
                if tuple(world.cell_id for world in prepared.work.worlds) != (
                    self.world_ids
                ):
                    raise CompactWorldAtlasReducerError(
                        "native dispatch world order differs from the bundle"
                    )
                self._native_state.process_dispatch(
                    start_next_absolute_tick=prepared.work.start,
                    end_next_absolute_tick=prepared.work.end,
                    event_rows=prepared.events,
                    death_metric_rows=prepared.metrics,
                    packed_birth_provenance_rows=prepared.provenance,
                    world_offsets=prepared.world_offsets,
                )
                for world_id in self.world_ids:
                    reconstruction = self.reconstructions[world_id]
                    reconstruction["next_tick"] = prepared.work.end
            elif self._pool is None:
                _reduce_dispatch_worlds(
                    prepared.work,
                    events=prepared.events,
                    metrics_raw=prepared.metrics,
                    provenance=prepared.provenance,
                    worlds_by_cell=self.results,
                    reconstruction_by_cell=self.reconstructions,
                )
            else:
                self._pool.submit_dispatch(prepared)
        except BaseException:
            self.close()
            raise
        self.next_absolute_tick = prepared.work.end
        self._dispatch_keys.append(key)
        return True

    def finalize(
        self,
        capture_index: str | Path | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Backfill any sealed suffix and bind the result to the terminal index."""

        self._require_open()
        index = _load_capture_index_v1(capture_index)
        dispatches, snapshots = _validate_capture_index_v1(index, self.bundle)
        try:
            snapshot_rows: list[tuple[int, Mapping[str, Any], tuple[str, str]]] = []
            for pointer in snapshots:
                _, manifest = _artifact_manifest(pointer, label="compact snapshot")
                absolute_tick = manifest.get("absolute_tick")
                if type(absolute_tick) is not int or absolute_tick < 0:
                    raise CompactWorldAtlasReducerError(
                        "compact snapshot tick is invalid"
                    )
                snapshot_rows.append(
                    (
                        absolute_tick,
                        pointer,
                        _artifact_pointer_key_v1(
                            pointer,
                            label="compact snapshot",
                        ),
                    )
                )
            snapshot_rows.sort(key=lambda row: row[0])

            snapshot_cursor = 0
            for dispatch_index, pointer in enumerate(dispatches):
                while (
                    snapshot_cursor < len(snapshot_rows)
                    and snapshot_rows[snapshot_cursor][0] <= self.next_absolute_tick
                ):
                    _, snapshot_pointer, snapshot_key = snapshot_rows[snapshot_cursor]
                    if snapshot_key not in self._snapshot_keys:
                        self.consume_snapshot(snapshot_pointer)
                    snapshot_cursor += 1
                dispatch_key = _artifact_pointer_key_v1(
                    pointer,
                    label="compact dispatch",
                )
                if dispatch_index < len(self._dispatch_keys):
                    if self._dispatch_keys[dispatch_index] != dispatch_key:
                        raise CompactWorldAtlasReducerError(
                            "incremental dispatch prefix differs from terminal index"
                        )
                else:
                    self.consume_dispatch(pointer)
            while snapshot_cursor < len(snapshot_rows):
                _, snapshot_pointer, snapshot_key = snapshot_rows[snapshot_cursor]
                if snapshot_key not in self._snapshot_keys:
                    self.consume_snapshot(snapshot_pointer)
                snapshot_cursor += 1

            terminal_dispatch_keys = [
                _artifact_pointer_key_v1(pointer, label="compact dispatch")
                for pointer in dispatches
            ]
            terminal_snapshot_keys = {row[2] for row in snapshot_rows}
            if (
                self._dispatch_keys != terminal_dispatch_keys
                or set(self._snapshot_keys) != terminal_snapshot_keys
            ):
                raise CompactWorldAtlasReducerError(
                    "incremental artifact prefix differs from terminal ownership"
                )
            if self._pool is not None:
                self._pool.finish()
                self._pool = None
            self._sync_native_results()
            if self._native_state is not None:
                expected_reconciliations = {
                    (world_id, tick)
                    for world_id in self.world_ids
                    for tick in self.reconstructions[world_id]["snapshots"]
                }
                if self._native_reconciled_snapshots != expected_reconciliations:
                    raise CompactWorldAtlasReducerError(
                        "native reducer did not reconcile every sealed snapshot"
                    )
                self._finalize_native_hereditary_attribution(
                    terminal_tick=int(index["terminal_next_absolute_tick"])
                )
            result = _finalize_world_atlas_v1(
                index,
                self.bundle,
                results=self.results,
                reconstructions=self.reconstructions,
            )
            if self._native_strategy_enabled:
                receipts = {}
                for world_index, world_id in enumerate(self.world_ids):
                    self._native_state.finalize_strategy(
                        world_index,
                        terminal_tick=int(index["terminal_next_absolute_tick"]),
                    )
                    receipts[world_id] = self._native_state.strategy_receipt(
                        world_index
                    )
                try:
                    self._strategy_result = render_native_strategy_attribution_v1(
                        index,
                        self.bundle,
                        receipts_by_world=receipts,
                        snapshots_by_world={
                            world_id: tuple(
                                self.reconstructions[world_id][
                                    "strategy_snapshots"
                                ][tick]
                                for tick in sorted(
                                    self.reconstructions[world_id][
                                        "strategy_snapshots"
                                    ]
                                )
                            )
                            for world_id in self.world_ids
                        },
                        initial_strategy_masks_by_world={
                            world_id: self.reconstructions[world_id][
                                "strategy_initial_masks"
                            ]
                            for world_id in self.world_ids
                        },
                        atlas_integrity_by_world={
                            world_id: self.results[world_id]["integrity"]
                            for world_id in self.world_ids
                        },
                        early_life_tick=self._strategy_early_life_tick,
                        pressure_factor=self._strategy_pressure_factor,
                    )
                except StrategyAttributionError as exc:
                    raise CompactWorldAtlasReducerError(
                        f"native strategy rendering failed: {exc}"
                    ) from exc
        except BaseException:
            self.close()
            raise
        if self._native_state is not None:
            self._native_state.close()
            self._native_state = None
        self._finalized = True
        self._closed = True
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None
        if self._native_state is not None:
            self._native_state.close()
            self._native_state = None
        for reconstruction in self.reconstructions.values():
            spool = reconstruction.get("hereditary_birth_spool")
            if isinstance(spool, _HereditaryBirthRecordSpoolV1):
                spool.abort()

    def __enter__(self) -> "CompactWorldAtlasIncrementalReducerV1":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _read_only_regular_file_v1(path: Path) -> bool:
    """Treat publication races as not-yet-sealed, never as partial input."""

    try:
        if path.is_symlink():
            return False
        mode = path.stat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) and not (mode & 0o222)


def _sealed_artifact_pointer_v1(root: Path) -> dict[str, Any] | None:
    """Return a pointer only after every manifest-owned file is read-only."""

    if root.is_symlink() or not root.is_dir():
        return None
    manifest_path = root / "manifest.json"
    if not _read_only_regular_file_v1(manifest_path):
        return None
    manifest = _read_json(manifest_path, label="sealed compact artifact manifest")
    owned_names = {
        value
        for key, value in manifest.items()
        if (key == "data_file" or key.endswith("_file")) and isinstance(value, str)
    }
    if not owned_names:
        raise CompactWorldAtlasReducerError(
            "sealed compact artifact manifest owns no payload files"
        )
    for name in owned_names:
        if Path(name).name != name:
            raise CompactWorldAtlasReducerError(
                "sealed compact artifact owns an invalid payload path"
            )
        path = root / name
        if not _read_only_regular_file_v1(path):
            return None
    artifact_sha256 = manifest.get("artifact_sha256")
    if not isinstance(artifact_sha256, str):
        raise CompactWorldAtlasReducerError(
            "sealed compact artifact lacks its identity"
        )
    pointer = {
        "path": str(root.resolve()),
        "artifact_sha256": artifact_sha256,
    }
    _artifact_manifest(pointer, label="sealed compact artifact")
    return pointer


def _canonical_snapshot_manifest_paths_v1(
    snapshots_root: Path,
) -> tuple[Path, ...]:
    """Enumerate only atomically published native snapshot roots.

    Native snapshot publication seals a hidden ``.tick-...stage`` directory
    before renaming it to its canonical ``tick-<12-digit-tick>`` root.  The
    replay harness may add a bounded ``-<3-digit-index>`` disambiguator to a
    copied snapshot root.  A reducer must never register the hidden stage:
    the publisher may rename it immediately after the manifest becomes
    read-only, leaving a stale path in the reducer's ownership ledger.
    """

    if snapshots_root.is_symlink() or not snapshots_root.is_dir():
        return ()
    result: list[Path] = []
    for candidate in snapshots_root.iterdir():
        name = candidate.name
        canonical_name = len(name) == 17 and name.startswith("tick-")
        indexed_name = (
            len(name) == 21
            and name.startswith("tick-")
            and name[17] == "-"
            and all("0" <= value <= "9" for value in name[18:])
        )
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or not (canonical_name or indexed_name)
            or not all("0" <= value <= "9" for value in name[5:17])
        ):
            continue
        result.append(candidate / "manifest.json")
    return tuple(sorted(result))


def _sealed_terminal_index_v1(
    path: Path,
    bundle: ScienceWorldBundleV1,
) -> dict[str, Any] | None:
    """Load terminal ownership only when every referenced artifact is sealed."""

    if not _read_only_regular_file_v1(path):
        return None
    index = _load_capture_index_v1(path)
    dispatches, snapshots = _validate_capture_index_v1(index, bundle)
    for pointer in (*dispatches, *snapshots):
        raw_path = pointer.get("path")
        expected_sha256 = pointer.get("artifact_sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_sha256, str):
            raise CompactWorldAtlasReducerError(
                "terminal compact artifact pointer is invalid"
            )
        sealed = _sealed_artifact_pointer_v1(Path(raw_path).expanduser())
        if sealed is None:
            return None
        if sealed["artifact_sha256"] != expected_sha256:
            raise CompactWorldAtlasReducerError(
                "terminal compact artifact identity differs"
            )
    return index


def _snapshot_frontier_ready_v1(
    reducer: CompactWorldAtlasIncrementalReducerV1,
) -> bool:
    tick = reducer.next_absolute_tick
    for world_id in reducer.world_ids:
        reconstruction = reducer.reconstructions[world_id]
        expected = reconstruction["world"].acquisition.to_value()["snapshot_ticks"]
        if tick in expected and tick not in reconstruction["snapshots"]:
            return False
    return True


@dataclass(slots=True)
class _AsyncReducerProgressV1:
    """Single-writer operational counters shared without a control channel."""

    phase: int
    started_monotonic_ns: int
    start_next_absolute_tick: int
    planned_terminal_tick: int
    reducer_frontier_tick: int
    producer_frontier_tick: int
    current_frontier_lag_ticks: int = 0
    maximum_frontier_lag_ticks: int = 0
    current_oldest_unreduced_age_ns: int = 0
    maximum_oldest_unreduced_age_ns: int = 0
    terminal_backlog_ticks: int = -1
    terminal_reconciliation_ns: int = 0
    peak_rss_kib: int = 0
    consumed_dispatch_count: int = 0
    observed_sealed_dispatch_count: int = 0
    consumed_snapshot_count: int = 0
    child_pid: int = 0
    active_child_process_count: int = 0
    terminal_seen_monotonic_ns: int = 0

    def publish(self, shared: Any) -> None:
        values = (
            self.phase,
            self.started_monotonic_ns,
            self.start_next_absolute_tick,
            self.planned_terminal_tick,
            self.reducer_frontier_tick,
            self.producer_frontier_tick,
            self.current_frontier_lag_ticks,
            self.maximum_frontier_lag_ticks,
            self.current_oldest_unreduced_age_ns,
            self.maximum_oldest_unreduced_age_ns,
            self.terminal_backlog_ticks,
            self.terminal_reconciliation_ns,
            self.peak_rss_kib,
            self.consumed_dispatch_count,
            self.observed_sealed_dispatch_count,
            self.consumed_snapshot_count,
            self.child_pid,
            self.active_child_process_count,
            self.terminal_seen_monotonic_ns,
        )
        sequence = int(shared[0])
        if sequence % 2:
            sequence += 1
        shared[0] = sequence + 1
        for index, value in enumerate(values, start=1):
            shared[index] = int(value)
        shared[0] = sequence + 2


def _async_progress_projection_v1(
    shared: Any,
    *,
    sidecar_pid: int | None,
    sidecar_alive: bool,
    sidecar_exitcode: int | None,
) -> dict[str, Any]:
    values: tuple[int, ...] | None = None
    consistent = False
    for _ in range(16):
        first = int(shared[0])
        candidate = tuple(
            int(shared[index])
            for index in range(1, _ASYNC_PROGRESS_FIELD_COUNT_V1)
        )
        second = int(shared[0])
        if first == second and first % 2 == 0:
            values = candidate
            consistent = True
            break
    if values is None:
        values = tuple(
            int(shared[index])
            for index in range(1, _ASYNC_PROGRESS_FIELD_COUNT_V1)
        )
    (
        phase,
        started_ns,
        start_tick,
        planned_terminal_tick,
        reducer_frontier,
        producer_frontier,
        current_lag,
        maximum_lag,
        current_oldest_ns,
        maximum_oldest_ns,
        terminal_backlog,
        terminal_reconciliation_ns,
        peak_rss_kib,
        consumed_dispatches,
        observed_dispatches,
        consumed_snapshots,
        child_pid,
        active_children,
        terminal_seen_ns,
    ) = values
    denominator = planned_terminal_tick - start_tick
    caught_up_fraction = (
        0.0
        if denominator <= 0
        else min(
            1.0,
            max(0.0, (reducer_frontier - start_tick) / denominator),
        )
    )
    producer_denominator = producer_frontier - start_tick
    producer_caught_up_fraction = (
        1.0
        if producer_denominator <= 0
        else min(
            1.0,
            max(0.0, (reducer_frontier - start_tick) / producer_denominator),
        )
    )
    return {
        "schema_version": "eiel_native_compact_reducer_operational_metrics_v1",
        "observer_only": True,
        "producer_notification_required": False,
        "producer_backpressure_path": False,
        "snapshot_consistent": consistent,
        "phase": _ASYNC_PROGRESS_PHASES_V1.get(phase, "UNKNOWN"),
        "started_monotonic_ns": started_ns,
        "start_next_absolute_tick": start_tick,
        "planned_terminal_tick": planned_terminal_tick,
        "reducer_frontier_tick": reducer_frontier,
        "producer_frontier_tick": producer_frontier,
        "producer_reducer_frontier_lag_ticks": current_lag,
        "maximum_producer_reducer_frontier_lag_ticks": maximum_lag,
        "oldest_unreduced_artifact_age_seconds": current_oldest_ns / 1e9,
        "maximum_oldest_unreduced_artifact_age_seconds": maximum_oldest_ns / 1e9,
        "campaign_caught_up_fraction": caught_up_fraction,
        "sealed_producer_frontier_caught_up_fraction": (
            producer_caught_up_fraction
        ),
        "terminal_backlog_ticks": (
            None if terminal_backlog < 0 else terminal_backlog
        ),
        "terminal_reconciliation_seconds": terminal_reconciliation_ns / 1e9,
        "peak_rss_kib": peak_rss_kib,
        "consumed_dispatch_count": consumed_dispatches,
        "observed_sealed_dispatch_count": observed_dispatches,
        "consumed_snapshot_count": consumed_snapshots,
        "terminal_seen_monotonic_ns": terminal_seen_ns,
        "sidecar_process": {
            "pid": sidecar_pid if sidecar_pid is not None else child_pid or None,
            "alive": sidecar_alive,
            "exitcode": sidecar_exitcode,
            "active_child_process_count_at_last_projection": active_children,
        },
    }


def _hereditary_tracked_id_v1(
    reconstruction: Mapping[str, Any], record: _ActiveGenomeV1
) -> str:
    if record.tracked_genotype_state_id is not None:
        return record.tracked_genotype_state_id
    return _tracked_genome_state_id(
        record.encoded,
        policy_loci=reconstruction["policy_loci"],
        trait_loci=reconstruction["trait_loci"],
    )


def _record_hereditary_metrics_v1(
    reconstruction: dict[str, Any],
    record: _ActiveGenomeV1,
    metrics: Mapping[str, Any] | None,
) -> None:
    """Attribute one complete lifetime to its exact tracked projection."""

    if reconstruction["hereditary_contract"] is None:
        return
    if not isinstance(metrics, Mapping):
        raise CompactWorldAtlasReducerError(
            "full-hereditary attribution lacks lifetime metrics"
        )
    tracked_id = _hereditary_tracked_id_v1(reconstruction, record)
    target = reconstruction["hereditary_metric_totals"].setdefault(
        tracked_id,
        {name: 0 for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1},
    )
    for index, name in enumerate(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1):
        value = metrics.get(name)
        # ``energy_delta_total`` is the one signed field in the compact
        # lifetime-metric ABI.  All counters remain non-negative; rejecting
        # a negative energy delta would make exact hereditary reduction fail
        # on ordinary deaths.
        if type(value) is not int or (index != 17 and value < 0):
            raise CompactWorldAtlasReducerError(
                "full-hereditary lifetime metric is invalid"
            )
        target[name] += value
    counts = reconstruction["hereditary_metric_agent_counts"]
    counts[tracked_id] = counts.get(tracked_id, 0) + 1
    reconstruction["hereditary_completed_agents"] += 1


def _hereditary_snapshot_projection_v1(
    reconstruction: Mapping[str, Any],
    active: Mapping[int, _ActiveGenomeV1],
) -> dict[str, Any] | None:
    if reconstruction["hereditary_contract"] is None:
        return None
    full_counts = Counter(_genome_state_id(record.encoded) for record in active.values())
    tracked_counts = Counter(
        _hereditary_tracked_id_v1(reconstruction, record)
        for record in active.values()
    )
    focal_ids = reconstruction["hereditary_focal_ids"]
    focal_counts = {focal_id: tracked_counts.get(focal_id, 0) for focal_id in focal_ids}
    return {
        "absolute_tick": None,
        "live_full_genome_state_counts": [
            {"genome_state_id": identity, "count": full_counts[identity]}
            for identity in sorted(full_counts)
        ],
        "live_tracked_genotype_state_counts": [
            {"tracked_genotype_state_id": identity, "count": tracked_counts[identity]}
            for identity in sorted(tracked_counts)
        ],
        "focal_live_counts": focal_counts,
        "focal_live_total": sum(focal_counts.values()),
        "total_live_count": len(active),
    }


def _native_hereditary_birth_artifact_v1(
    path_value: object,
    *,
    world_id: str,
    record_count: int,
) -> dict[str, Any]:
    """Seal the native stream into the common immutable artifact contract."""

    if not isinstance(path_value, (str, Path)):
        raise CompactWorldAtlasReducerError(
            "native hereditary birth artifact path is absent"
        )
    path = Path(path_value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise CompactWorldAtlasReducerError(
            "native hereditary birth artifact is not a regular file"
        )
    try:
        os.chmod(path, 0o444, follow_symlinks=False)
    except (NotImplementedError, OSError) as exc:
        raise CompactWorldAtlasReducerError(
            "native hereditary birth artifact could not be sealed read-only"
        ) from exc
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        size = path.stat().st_size
    except OSError as exc:
        raise CompactWorldAtlasReducerError(
            "native hereditary birth artifact could not be hashed"
        ) from exc
    return {
        "schema_version": HEREDITARY_BIRTH_RECORD_ARTIFACT_SCHEMA_V1,
        "storage_encoding": HEREDITARY_BIRTH_RECORD_STORAGE_ENCODING_V1,
        "world_ids": [world_id],
        "path": str(path),
        "file_sha256": digest.hexdigest(),
        "bytes": size,
        "record_count": record_count,
        "world_record_counts": {world_id: record_count},
        "record_width": _HEREDITARY_BIRTH_RECORD_STRUCT_V1.size,
        "header": _HEREDITARY_BIRTH_RECORD_MAGIC_V1.decode("ascii").rstrip(
            "\n"
        ),
        "immutable": True,
    }


def _compact_world_atlas_async_entry_v1(
    science_compact_root: str,
    world_bundle: str | Path | ScienceWorldBundleV1,
    reducer_output_root: str | None,
    max_workers: int,
    poll_interval_seconds: float,
    native_artifact_path: str | None,
    native_manifest_path: str | None,
    native_manifest_sha256: str | None,
    native_artifact_use_class: str,
    cancel_event: Any,
    result_queue: Any,
    progress_shared: Any,
) -> None:
    """Independent process entrypoint; producer notification is not required."""

    reducer: CompactWorldAtlasIncrementalReducerV1 | None = None
    started_ns = time.monotonic_ns()
    progress = _AsyncReducerProgressV1(
        phase=0,
        started_monotonic_ns=started_ns,
        start_next_absolute_tick=0,
        planned_terminal_tick=0,
        reducer_frontier_tick=0,
        producer_frontier_tick=0,
        child_pid=os.getpid(),
        terminal_backlog_ticks=-1,
    )
    progress.publish(progress_shared)
    try:
        if os.name == "posix":
            try:
                os.nice(_ASYNC_REDUCER_NICE_INCREMENT_V1)
            except OSError:
                pass
        native_artifact = None
        if native_artifact_path is not None:
            if native_manifest_path is None or native_manifest_sha256 is None:
                raise CompactWorldAtlasReducerError(
                    "native sidecar identity is incomplete"
                )
            native_artifact = load_native_compact_reducer_v1(
                native_artifact_path,
                native_manifest_path,
                require_production_qualified=(
                    native_artifact_use_class
                    == NATIVE_REDUCER_ARTIFACT_USE_PRODUCTION_V1
                ),
                expected_manifest_sha256=native_manifest_sha256,
            )
        reducer = CompactWorldAtlasIncrementalReducerV1(
            world_bundle,
            max_workers=max_workers,
            native_artifact=native_artifact,
            output_root=reducer_output_root,
        )
        terminal_ticks = {
            int(world.semantic_inputs.to_value()["time"]["total_ticks"])
            for world in reducer.bundle.worlds
        }
        if len(terminal_ticks) != 1:
            raise CompactWorldAtlasReducerError(
                "async physical reduction requires one terminal tick"
            )
        progress.phase = 1
        progress.start_next_absolute_tick = reducer.next_absolute_tick
        progress.planned_terminal_tick = terminal_ticks.pop()
        progress.reducer_frontier_tick = reducer.next_absolute_tick
        progress.producer_frontier_tick = reducer.next_absolute_tick
        root = Path(science_compact_root)
        terminal_index = root / "capture-index-terminal.json"
        sealed_dispatches: dict[
            tuple[str, str], tuple[dict[str, Any], int, int, int]
        ] = {}
        sealed_snapshots: dict[
            tuple[str, str], tuple[dict[str, Any], int, int]
        ] = {}

        def register_snapshot(pointer: Mapping[str, Any]) -> None:
            key = _artifact_pointer_key_v1(pointer, label="sealed compact snapshot")
            if key not in sealed_snapshots:
                _, manifest = _artifact_manifest(
                    pointer,
                    label="sealed compact snapshot",
                )
                absolute_tick = manifest.get("absolute_tick")
                if type(absolute_tick) is not int or absolute_tick < 0:
                    raise CompactWorldAtlasReducerError(
                        "sealed compact snapshot tick is invalid"
                    )
                sealed_snapshots[key] = (
                    dict(pointer),
                    absolute_tick,
                    int((Path(key[0]) / "manifest.json").stat().st_mtime_ns),
                )

        def discover_dispatches() -> None:
            dispatches_root = root / "dispatches"
            if not dispatches_root.is_dir() or dispatches_root.is_symlink():
                return
            for candidate in sorted(dispatches_root.glob("ticks-*-*")):
                pointer = _sealed_artifact_pointer_v1(candidate)
                if pointer is None:
                    continue
                key = _artifact_pointer_key_v1(
                    pointer,
                    label="sealed compact dispatch",
                )
                if key in sealed_dispatches:
                    continue
                _, manifest = _artifact_manifest(
                    pointer,
                    label="sealed compact dispatch",
                )
                start = manifest.get("start_next_absolute_tick")
                end = manifest.get("end_next_absolute_tick")
                if (
                    type(start) is not int
                    or type(end) is not int
                    or start < progress.start_next_absolute_tick
                    or end <= start
                    or candidate.name != f"ticks-{start:012d}-{end:012d}"
                ):
                    raise CompactWorldAtlasReducerError(
                        "sealed compact dispatch frontier identity differs"
                    )
                sealed_dispatches[key] = (
                    pointer,
                    start,
                    end,
                    int((candidate / "manifest.json").stat().st_mtime_ns),
                )

        def publish_progress() -> None:
            assert reducer is not None
            progress.reducer_frontier_tick = reducer.next_absolute_tick
            progress.producer_frontier_tick = max(
                [progress.start_next_absolute_tick]
                + [row[2] for row in sealed_dispatches.values()]
            )
            progress.current_frontier_lag_ticks = max(
                0,
                progress.producer_frontier_tick - progress.reducer_frontier_tick,
            )
            progress.maximum_frontier_lag_ticks = max(
                progress.maximum_frontier_lag_ticks,
                progress.current_frontier_lag_ticks,
            )
            consumed = reducer.consumed_artifact_keys
            pending_mtimes = [
                row[3]
                for key, row in sealed_dispatches.items()
                if key not in consumed
            ] + [
                row[2]
                for key, row in sealed_snapshots.items()
                if key not in consumed
            ]
            progress.current_oldest_unreduced_age_ns = (
                max(0, time.time_ns() - min(pending_mtimes))
                if pending_mtimes
                else 0
            )
            progress.maximum_oldest_unreduced_age_ns = max(
                progress.maximum_oldest_unreduced_age_ns,
                progress.current_oldest_unreduced_age_ns,
            )
            progress.peak_rss_kib = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            progress.consumed_dispatch_count = reducer.consumed_dispatch_count
            progress.observed_sealed_dispatch_count = len(sealed_dispatches)
            progress.consumed_snapshot_count = reducer.consumed_snapshot_count
            progress.active_child_process_count = len(
                multiprocessing.active_children()
            )
            progress.publish(progress_shared)

        publish_progress()
        while True:
            if cancel_event.is_set():
                reducer.close()
                progress.phase = 5
                publish_progress()
                result_queue.put(("cancelled",))
                return

            made_progress = False
            snapshots_root = root / "snapshots"
            for manifest_path in _canonical_snapshot_manifest_paths_v1(
                snapshots_root
            ):
                pointer = _sealed_artifact_pointer_v1(manifest_path.parent)
                if pointer is not None:
                    register_snapshot(pointer)

            snapshot_candidates = sorted(
                (
                    row
                    for key, row in sealed_snapshots.items()
                    if key not in reducer.consumed_artifact_keys
                    and row[1] <= reducer.next_absolute_tick
                ),
                key=lambda row: (row[1], str(row[0].get("path", ""))),
            )
            for snapshot_pointer, snapshot_tick, _ in snapshot_candidates:
                if snapshot_tick < reducer.next_absolute_tick:
                    raise CompactWorldAtlasReducerError(
                        "sealed compact snapshot fell behind reducer frontier"
                    )
                if reducer.consume_snapshot(snapshot_pointer):
                    made_progress = True

            discover_dispatches()
            publish_progress()

            sealed_terminal = _sealed_terminal_index_v1(
                terminal_index,
                reducer.bundle,
            )
            if sealed_terminal is not None:
                progress.phase = 2
                progress.terminal_seen_monotonic_ns = time.monotonic_ns()
                progress.terminal_backlog_ticks = max(
                    0,
                    int(sealed_terminal["terminal_next_absolute_tick"])
                    - reducer.next_absolute_tick,
                )
                publish_progress()
                result = reducer.finalize(sealed_terminal)
                progress.terminal_reconciliation_ns = (
                    time.monotonic_ns() - progress.terminal_seen_monotonic_ns
                )
                progress.phase = 3
                publish_progress()
                if progress.active_child_process_count != 0:
                    raise CompactWorldAtlasReducerError(
                        "async compact reducer retained child workers after finalization"
                    )
                result_queue.put(
                    (
                        "result_bundle",
                        {
                            "schema_version": (
                                "eiel_native_compact_reducer_execution_bundle_v1"
                            ),
                            "atlas_result": result,
                            "strategy_attribution_result": (
                                reducer.strategy_attribution_result
                            ),
                            "second_python_attribution_pass": False,
                            "native_artifact_use_class": (
                                native_artifact_use_class
                                if native_artifact is not None
                                else None
                            ),
                            "operational_metrics": _async_progress_projection_v1(
                                progress_shared,
                                sidecar_pid=os.getpid(),
                                sidecar_alive=True,
                                sidecar_exitcode=None,
                            ),
                        },
                    )
                )
                return

            if _snapshot_frontier_ready_v1(reducer):
                candidates = [
                    row[0]
                    for key, row in sealed_dispatches.items()
                    if row[1] == reducer.next_absolute_tick
                    and key not in reducer.consumed_artifact_keys
                ]
                if len(candidates) > 1:
                    raise CompactWorldAtlasReducerError(
                        "multiple sealed dispatches own one incremental frontier"
                    )
                if candidates:
                    reducer.consume_dispatch(candidates[0])
                    made_progress = True
                    publish_progress()
            if not made_progress:
                time.sleep(poll_interval_seconds)
    except BaseException as exc:  # pragma: no cover - surfaced through IPC tests
        if reducer is not None:
            reducer.close()
            progress.reducer_frontier_tick = reducer.next_absolute_tick
            progress.consumed_dispatch_count = reducer.consumed_dispatch_count
            progress.consumed_snapshot_count = reducer.consumed_snapshot_count
        progress.phase = 4
        progress.peak_rss_kib = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        progress.active_child_process_count = len(multiprocessing.active_children())
        progress.publish(progress_shared)
        result_queue.put(
            (
                "error",
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
        )


class CompactWorldAtlasAsyncSidecarV1:
    """Nonblocking process wrapper around the sealed-artifact reducer.

    The producer never submits work to this object.  The sidecar discovers
    read-only artifacts beneath its exact batch-local root, so a slow or dead
    reducer cannot create CUDA or ACS2 backpressure.
    """

    def __init__(
        self,
        science_compact_root: str | Path,
        world_bundle: str | Path | ScienceWorldBundleV1,
        *,
        max_workers: int = 1,
        poll_interval_seconds: float = 0.25,
        native_artifact_path: str | Path | None = None,
        native_manifest_path: str | Path | None = None,
        native_manifest_sha256: str | None = None,
        native_artifact_use_class: str = NATIVE_REDUCER_ARTIFACT_USE_PRODUCTION_V1,
        start_immediately: bool = True,
    ) -> None:
        if type(max_workers) is not int or max_workers < 1:
            raise CompactWorldAtlasReducerError(
                "max_workers must be a positive integer"
            )
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or not 0.01 <= float(poll_interval_seconds) <= 60.0
        ):
            raise CompactWorldAtlasReducerError(
                "poll_interval_seconds must be between 0.01 and 60"
            )
        try:
            context_name = "forkserver" if os.name == "posix" else "spawn"
            context = multiprocessing.get_context(context_name)
        except ValueError as exc:  # pragma: no cover - POSIX without fork
            raise CompactWorldAtlasReducerError(
                "async compact reducer cannot acquire a safe process context"
            ) from exc
        root = Path(science_compact_root).expanduser()
        if root.is_symlink():
            raise CompactWorldAtlasReducerError(
                "async compact reducer root must not be a symbolic link"
            )
        native_values = (
            native_artifact_path,
            native_manifest_path,
            native_manifest_sha256,
        )
        if any(value is not None for value in native_values) and not all(
            value is not None for value in native_values
        ):
            raise CompactWorldAtlasReducerError(
                "native async reducer requires artifact, manifest, and digest"
            )
        if native_manifest_sha256 is not None and (
            len(native_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in native_manifest_sha256
            )
        ):
            raise CompactWorldAtlasReducerError(
                "native async reducer manifest digest is invalid"
            )
        if native_artifact_use_class not in _NATIVE_REDUCER_ARTIFACT_USE_CLASSES_V1:
            raise CompactWorldAtlasReducerError(
                "native async reducer artifact use class is invalid"
            )
        if native_artifact_path is None and (
            native_artifact_use_class
            != NATIVE_REDUCER_ARTIFACT_USE_PRODUCTION_V1
        ):
            raise CompactWorldAtlasReducerError(
                "native artifact use class requires a native artifact"
            )
        if type(start_immediately) is not bool:
            raise CompactWorldAtlasReducerError(
                "start_immediately must be a boolean"
            )
        if native_artifact_path is not None:
            try:
                load_native_compact_reducer_v1(
                    native_artifact_path,
                    native_manifest_path,
                    require_production_qualified=(
                        native_artifact_use_class
                        == NATIVE_REDUCER_ARTIFACT_USE_PRODUCTION_V1
                    ),
                    expected_manifest_sha256=native_manifest_sha256,
                )
            except Exception as exc:
                raise CompactWorldAtlasReducerError(
                    f"native async reducer preflight failed: {exc}"
                ) from exc
        self.execution_route = (
            "native_cpp_v1" if native_artifact_path is not None else "python_v1"
        )
        self.native_artifact_use_class = (
            native_artifact_use_class if native_artifact_path is not None else None
        )
        self._cancel_event = context.Event()
        self._result_queue = context.Queue(maxsize=1)
        self._progress_shared = context.RawArray(
            "q", _ASYNC_PROGRESS_FIELD_COUNT_V1
        )
        self._progress_shared[11] = -1
        self._message: tuple[Any, ...] | None = None
        self._closed = False
        self._started = False
        self._cleanup_outcome: dict[str, Any] | None = None
        self._process = context.Process(
            target=_compact_world_atlas_async_entry_v1,
            args=(
                str(root.resolve()),
                world_bundle,
                str(root.parent.resolve()),
                max_workers,
                float(poll_interval_seconds),
                (
                    str(Path(native_artifact_path).expanduser().resolve())
                    if native_artifact_path is not None
                    else None
                ),
                (
                    str(Path(native_manifest_path).expanduser().resolve())
                    if native_manifest_path is not None
                    else None
                ),
                native_manifest_sha256,
                native_artifact_use_class,
                self._cancel_event,
                self._result_queue,
                self._progress_shared,
            ),
            name="compact-world-atlas-async-sidecar",
        )
        if start_immediately:
            self.start()

    def start(self) -> None:
        """Start the observer after its owning execution root is bound."""

        if self._closed:
            raise CompactWorldAtlasReducerError(
                "async compact reducer is closed"
            )
        if self._started:
            return
        self._process.start()
        self._started = True

    @property
    def pid(self) -> int | None:
        return self._process.pid

    @property
    def is_alive(self) -> bool:
        return self._started and self._process.is_alive()

    def progress_snapshot(self) -> dict[str, Any]:
        """Return observer-only counters; this never signals the reducer."""

        projection = _async_progress_projection_v1(
            self._progress_shared,
            sidecar_pid=self._process.pid,
            sidecar_alive=self.is_alive,
            sidecar_exitcode=self._process.exitcode,
        )
        if self._cleanup_outcome is not None:
            projection["process_cleanup"] = dict(self._cleanup_outcome)
        return projection

    def result_bundle(
        self, *, timeout_seconds: float | None = None
    ) -> dict[str, Any]:
        """Wait only at the explicit post-GPU consumer boundary."""

        if not self._started:
            raise CompactWorldAtlasReducerError(
                "async compact reducer has not started"
            )
        if self._closed and self._message is None:
            raise CompactWorldAtlasReducerError("async compact reducer is closed")
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
        ):
            raise CompactWorldAtlasReducerError(
                "timeout_seconds must be a positive finite number"
            )
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + float(timeout_seconds)
        )
        while self._message is None:
            wait_seconds = 0.25
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CompactWorldAtlasReducerError(
                        "async compact reducer result timed out"
                    )
                wait_seconds = min(wait_seconds, remaining)
            try:
                self._message = self._result_queue.get(timeout=wait_seconds)
            except queue.Empty:
                if not self._process.is_alive():
                    self._process.join(timeout=10)
                    self._cleanup_outcome = {
                        "clean": True,
                        "forced_termination": False,
                        "orphaned_worker_count": 0,
                        "unexpected_exit": True,
                    }
                    self._close_queue()
                    raise CompactWorldAtlasReducerError(
                        "async compact reducer exited without a result"
                    )
        message = self._message
        if message[0] == "error":
            self._process.join(timeout=10)
            self._cleanup_outcome = {
                "clean": not self._process.is_alive(),
                "forced_termination": False,
                "orphaned_worker_count": int(
                    self.progress_snapshot()["sidecar_process"][
                        "active_child_process_count_at_last_projection"
                    ]
                ),
            }
            self._close_queue()
            raise CompactWorldAtlasReducerError(
                f"async compact reducer failed with {message[1]}: "
                f"{message[2]}\n{message[3]}"
            )
        if message[0] not in {"result", "result_bundle"} or len(message) != 2:
            self.cancel()
            raise CompactWorldAtlasReducerError(
                "async compact reducer did not produce a result"
            )
        self._process.join(timeout=10)
        if self._process.is_alive() or self._process.exitcode != 0:
            raise CompactWorldAtlasReducerError(
                "async compact reducer did not exit cleanly"
            )
        self._close_queue()
        self._cleanup_outcome = {
            "clean": True,
            "forced_termination": False,
            "orphaned_worker_count": 0,
        }
        result = message[1]
        if not isinstance(result, dict):
            raise CompactWorldAtlasReducerError(
                "async compact reducer result is not an object"
            )
        if message[0] == "result":
            result = {
                "schema_version": "eiel_native_compact_reducer_execution_bundle_v1",
                "atlas_result": result,
                "strategy_attribution_result": None,
                "second_python_attribution_pass": False,
                "native_artifact_use_class": self.native_artifact_use_class,
                "operational_metrics": self.progress_snapshot(),
            }
            return result
        if (
            result.get("schema_version")
            != "eiel_native_compact_reducer_execution_bundle_v1"
            or not isinstance(result.get("atlas_result"), Mapping)
            or result.get("second_python_attribution_pass") is not False
        ):
            raise CompactWorldAtlasReducerError(
                "async compact reducer execution bundle is invalid"
            )
        result = dict(result)
        result["operational_metrics"] = self.progress_snapshot()
        return result

    def result(self, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        """Return the legacy atlas projection from the fused execution bundle."""

        return dict(
            self.result_bundle(timeout_seconds=timeout_seconds)["atlas_result"]
        )

    def _close_queue(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._result_queue.close()

    @staticmethod
    def _descendant_pids(root_pid: int | None) -> tuple[int, ...]:
        if root_pid is None or os.name != "posix":
            return ()
        discovered: list[int] = []
        pending = [root_pid]
        while pending:
            parent = pending.pop()
            children_path = Path(f"/proc/{parent}/task/{parent}/children")
            try:
                children = [
                    int(value)
                    for value in children_path.read_text(encoding="ascii").split()
                ]
            except (FileNotFoundError, OSError, UnicodeError, ValueError):
                continue
            for child in children:
                if child not in discovered:
                    discovered.append(child)
                    pending.append(child)
        return tuple(discovered)

    @staticmethod
    def _signal_exact_processes(pids: Sequence[int], value: int) -> None:
        for pid in reversed(tuple(pids)):
            try:
                os.kill(pid, value)
            except (ProcessLookupError, PermissionError):
                pass

    def cancel(self) -> None:
        """Stop only the sidecar; producer state and artifacts are untouched."""

        if self._closed:
            return
        if not self._started:
            self._cleanup_outcome = {
                "clean": True,
                "forced_termination": False,
                "orphaned_worker_count": 0,
                "not_started": True,
            }
            self._close_queue()
            return
        self._cancel_event.set()
        self._process.join(timeout=2)
        descendants = self._descendant_pids(self._process.pid)
        forced = False
        if self._process.is_alive():
            forced = True
            self._signal_exact_processes(descendants, signal.SIGTERM)
            self._process.terminate()
            self._process.join(timeout=1)
        if self._process.is_alive():
            self._signal_exact_processes(descendants, signal.SIGKILL)
            self._process.kill()
            self._process.join(timeout=1)
        remaining_descendants = tuple(
            pid for pid in descendants if Path(f"/proc/{pid}").exists()
        )
        self._cleanup_outcome = {
            "clean": not self._process.is_alive() and not remaining_descendants,
            "forced_termination": forced,
            "orphaned_worker_count": len(remaining_descendants),
            "resolved_descendant_pids": list(descendants),
        }
        self._close_queue()

    def __enter__(self) -> "CompactWorldAtlasAsyncSidecarV1":
        return self

    def __exit__(self, *_: object) -> None:
        self.cancel()


def reduce_compact_science_world_atlas_v1(
    capture_index: str | Path | Mapping[str, Any],
    world_bundle: str | Path | ScienceWorldBundleV1,
    *,
    max_workers: int = 1,
    native_artifact: NativeCompactReducerArtifactV1 | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Reduce one physical compact index into exact logical-world results.

    ``max_workers=1`` retains the original serial execution.  Values greater
    than one enable opt-in logical-world sharding; dispatch payloads are
    authenticated and decompressed once by the parent, then exposed to fixed
    worker shards through bounded shared memory.  Worker count is an execution
    detail and never enters the canonical result identity.
    """
    if type(max_workers) is not int or max_workers < 1:
        raise CompactWorldAtlasReducerError("max_workers must be a positive integer")

    if native_artifact is not None:
        with CompactWorldAtlasIncrementalReducerV1(
            world_bundle,
            max_workers=max_workers,
            native_artifact=native_artifact,
            output_root=output_root,
        ) as reducer:
            return reducer.finalize(capture_index)

    index = _load_capture_index_v1(capture_index)
    bundle = _load_world_bundle_v1(world_bundle)
    dispatches, snapshots = _validate_capture_index_v1(index, bundle)
    results = {world.world_id: _blank_world(world) for world in bundle.worlds}
    reconstructions = {
        world.world_id: _blank_reconstruction(world) for world in bundle.worlds
    }
    hereditary_worlds = {
        world_id
        for world_id, reconstruction in reconstructions.items()
        if reconstruction["hereditary_contract"] is not None
    }
    if hereditary_worlds:
        max_workers = 1
    _configure_hereditary_birth_spool_v1(
        reconstructions,
        world_ids=tuple(world.world_id for world in bundle.worlds),
        output_root=output_root,
    )
    for pointer in snapshots:
        _reduce_snapshot(
            pointer,
            worlds_by_cell=results,
            reconstruction_by_cell=reconstructions,
        )
    world_ids = tuple(world.world_id for world in bundle.worlds)
    try:
        if max_workers == 1:
            for world_id in world_ids:
                _initialize_reconstruction(
                    reconstructions[world_id], result=results[world_id]
                )
            for pointer in dispatches:
                _reduce_dispatch(
                    pointer,
                    worlds_by_cell=results,
                    reconstruction_by_cell=reconstructions,
                )
        else:
            _reduce_dispatches_parallel(
                dispatches,
                world_ids=world_ids,
                worlds_by_cell=results,
                reconstruction_by_cell=reconstructions,
                max_workers=max_workers,
            )
        return _finalize_world_atlas_v1(
            index,
            bundle,
            results=results,
            reconstructions=reconstructions,
        )
    except BaseException:
        for reconstruction in reconstructions.values():
            spool = reconstruction.get("hereditary_birth_spool")
            if isinstance(spool, _HereditaryBirthRecordSpoolV1):
                spool.abort()
        raise


__all__ = (
    "COMPACT_WORLD_ATLAS_CONFIGURATION_SCHEMA",
    "COMPACT_WORLD_ATLAS_CONTRACT_VERSION",
    "COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID",
    "COMPACT_WORLD_ATLAS_REDUCER_ID",
    "COMPACT_WORLD_ATLAS_REQUIRED_CAPABILITIES",
    "CompactWorldAtlasAsyncSidecarV1",
    "CompactWorldAtlasIncrementalReducerV1",
    "CompactWorldAtlasReducerError",
    "HEREDITARY_BIRTH_RECORD_ARTIFACT_SCHEMA_V1",
    "HEREDITARY_BIRTH_RECORD_STORAGE_ENCODING_V1",
    "compact_world_atlas_reducer_descriptor_v1",
    "reduce_compact_science_world_atlas_v1",
    "validate_hereditary_birth_record_artifact_v1",
)
