"""Fail-closed ingestion of canonical CUDA V2 science sidecars.

The CUDA provider owns creation and lifecycle registration of the raw sidecar.
This module is the effect-free offline boundary used by science tooling: it
verifies the sealed bytes, reconstructs every canonical observation record,
checks the per-world frontier, and then delegates publication to the generic
route-neutral evidence API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from emergence_contracts import (
    ContentDigest,
    SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_VERSION,
    ScienceObservationRecordV1,
    canonical_json_bytes,
    canonical_science_record_stream_sha256_v1,
    canonical_sha256,
    validate_science_observation_stream_v1,
    validate_science_native_control_receipt_v1,
)

from .science_publication_v1 import (
    ScienceObservationPublicationPlanV1,
    ScienceWorldEvidencePublicationV1,
    publish_science_world_evidence_v1,
)


SCIENCE_NATIVE_SIDECAR_SCHEMA_V1 = (
    "eiel_cuda_full_evolution_science_observation_sidecar_v1"
)
SCIENCE_NATIVE_SIDECAR_FRONTIER_SCHEMA_V1 = (
    "eiel_cuda_full_evolution_science_observation_frontier_v1"
)
SCIENCE_NATIVE_SIDECAR_OBSERVATION_ABI_V1 = "eiel_science_observation_record_v1"
SCIENCE_NATIVE_SIDECAR_MODE_V1 = "TYPED_SIDECAR_V1"


class ScienceNativeSidecarError(ValueError):
    """A native sidecar cannot be trusted as canonical science evidence."""


def _validate_control_binding_join_v1(
    rows: tuple[ScienceObservationRecordV1, ...], *, cell_id: str
) -> None:
    """Require every sidecar row to join one control-binding provenance row."""

    controls = tuple(row for row in rows if row.record_kind.value == "control_binding")
    if len(controls) != 1:
        raise ScienceNativeSidecarError(
            f"science sidecar must contain exactly one control binding for {cell_id}"
        )
    control = controls[0]
    if any(
        row.control_binding_sha256 != control.control_binding_sha256 for row in rows
    ):
        raise ScienceNativeSidecarError(
            f"science sidecar control-binding join differs for {cell_id}"
        )
    payload = control.payload.to_value()
    payload_schema = str(control.payload.schema_version)
    applied = (
        payload.get("applied_native_control") if isinstance(payload, dict) else None
    )
    if payload_schema != str(SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_VERSION):
        if applied is not None:
            raise ScienceNativeSidecarError(
                f"science sidecar native control payload schema differs for {cell_id}"
            )
        return
    if applied is None:
        raise ScienceNativeSidecarError(
            f"science sidecar native control provenance is missing for {cell_id}"
        )
    try:
        validate_science_native_control_receipt_v1(
            applied, field=f"sidecar[{cell_id}].applied_native_control"
        )
    except Exception as exc:
        raise ScienceNativeSidecarError(
            f"science sidecar native control provenance is invalid for {cell_id}"
        ) from exc
    if canonical_sha256(payload) != control.control_binding_sha256:
        raise ScienceNativeSidecarError(
            f"science sidecar native control binding digest differs for {cell_id}"
        )


def _digest(value: object, *, field: str) -> ContentDigest:
    if not isinstance(value, Mapping) or set(value) != {"algorithm", "hex"}:
        raise ScienceNativeSidecarError(f"{field} is not an exact SHA-256 digest")
    try:
        return ContentDigest(str(value["algorithm"]), str(value["hex"]))
    except (TypeError, ValueError) as exc:
        raise ScienceNativeSidecarError(
            f"{field} is not a valid SHA-256 digest"
        ) from exc


@dataclass(frozen=True, slots=True)
class NativeScienceSidecarV1:
    path: Path
    sidecar_sha256: ContentDigest
    batch_id: str
    cell_ids: tuple[str, ...]
    records_by_cell: tuple[tuple[str, tuple[ScienceObservationRecordV1, ...]], ...]
    next_absolute_tick: int
    frontier: dict[str, object]

    def records_for_cell(self, cell_id: str) -> tuple[ScienceObservationRecordV1, ...]:
        for candidate, rows in self.records_by_cell:
            if candidate == cell_id:
                return rows
        raise ScienceNativeSidecarError(f"sidecar does not contain cell {cell_id!r}")

    def records_mapping(self) -> dict[str, tuple[ScienceObservationRecordV1, ...]]:
        return dict(self.records_by_cell)


def _frontier_for_records(
    records_by_cell: Mapping[str, tuple[ScienceObservationRecordV1, ...]],
    *,
    next_absolute_tick: int,
) -> dict[str, object]:
    per_world: dict[str, object] = {}
    for cell_id in sorted(records_by_cell):
        rows = records_by_cell[cell_id]
        if not rows:
            raise ScienceNativeSidecarError(
                f"typed science world {cell_id} has no canonical observations"
            )
        per_world[cell_id] = {
            "cell_id": cell_id,
            "logical_world_id": rows[0].logical_world_id,
            "world_uid": rows[0].world_uid.to_dict(),
            "record_count": len(rows),
            "next_world_local_sequence": len(rows),
            "last_absolute_tick": rows[-1].absolute_tick,
            "prefix_chain_sha256": canonical_science_record_stream_sha256_v1(
                rows
            ).to_dict(),
        }
    return {
        "schema_version": SCIENCE_NATIVE_SIDECAR_FRONTIER_SCHEMA_V1,
        "observation_mode": SCIENCE_NATIVE_SIDECAR_MODE_V1,
        "observation_abi": SCIENCE_NATIVE_SIDECAR_OBSERVATION_ABI_V1,
        "next_absolute_tick": next_absolute_tick,
        "per_world": per_world,
    }


def load_native_science_sidecar_v1(
    path: Path,
    *,
    expected_sha256: str | ContentDigest | None = None,
    expected_batch_id: str | None = None,
    expected_cell_ids: tuple[str, ...] | None = None,
    expected_next_absolute_tick: int | None = None,
    producer_phase_registry_ordinals: Mapping[str, int],
) -> NativeScienceSidecarV1:
    """Load one immutable terminal sidecar and verify its complete frontier."""

    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ScienceNativeSidecarError(
            "science sidecar is unavailable or is a symlink"
        )
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ScienceNativeSidecarError(
            "science sidecar is unavailable or is a symlink"
        )
    payload = resolved.read_bytes()
    observed_sha256 = ContentDigest("sha256", hashlib.sha256(payload).hexdigest())
    if expected_sha256 is not None:
        expected = (
            expected_sha256
            if isinstance(expected_sha256, ContentDigest)
            else ContentDigest("sha256", expected_sha256)
        )
        if observed_sha256 != expected:
            raise ScienceNativeSidecarError("science sidecar digest differs")
    try:
        body = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ScienceNativeSidecarError(
            "science sidecar is not canonical JSON data"
        ) from exc
    if canonical_json_bytes(body) != payload:
        raise ScienceNativeSidecarError("science sidecar bytes are not canonical")
    required = {
        "schema_version",
        "observation_abi",
        "observation_mode",
        "batch_id",
        "cell_ids",
        "records_by_cell",
        "frontier",
    }
    if not isinstance(body, dict) or set(body) != required:
        raise ScienceNativeSidecarError("science sidecar object shape is invalid")
    if (
        body["schema_version"] != SCIENCE_NATIVE_SIDECAR_SCHEMA_V1
        or body["observation_abi"] != SCIENCE_NATIVE_SIDECAR_OBSERVATION_ABI_V1
        or body["observation_mode"] != SCIENCE_NATIVE_SIDECAR_MODE_V1
    ):
        raise ScienceNativeSidecarError("science sidecar contract identity is invalid")
    batch_id = body["batch_id"]
    if not isinstance(batch_id, str) or not batch_id:
        raise ScienceNativeSidecarError("science sidecar batch identity is invalid")
    if expected_batch_id is not None and batch_id != expected_batch_id:
        raise ScienceNativeSidecarError("science sidecar names another physical batch")
    cell_values = body["cell_ids"]
    if (
        not isinstance(cell_values, list)
        or any(not isinstance(value, str) or not value for value in cell_values)
        or len(cell_values) != len(set(cell_values))
    ):
        raise ScienceNativeSidecarError("science sidecar cell identities are invalid")
    cell_ids = tuple(cell_values)
    if expected_cell_ids is not None and cell_ids != expected_cell_ids:
        raise ScienceNativeSidecarError("science sidecar cell set or order differs")
    raw_rows = body["records_by_cell"]
    if not isinstance(raw_rows, dict) or set(raw_rows) != set(cell_ids):
        raise ScienceNativeSidecarError("science sidecar record ownership differs")

    records: dict[str, tuple[ScienceObservationRecordV1, ...]] = {}
    for cell_id in cell_ids:
        values = raw_rows[cell_id]
        if not isinstance(values, list):
            raise ScienceNativeSidecarError(
                f"science sidecar rows are invalid for {cell_id}"
            )
        try:
            rows = tuple(ScienceObservationRecordV1.from_dict(row) for row in values)
        except Exception as exc:
            raise ScienceNativeSidecarError(
                f"science sidecar row validation failed for {cell_id}"
            ) from exc
        if not rows or any(row.cell_id != cell_id for row in rows):
            raise ScienceNativeSidecarError(
                f"science sidecar record ownership differs for {cell_id}"
            )
        if tuple(row.world_local_sequence for row in rows) != tuple(range(len(rows))):
            raise ScienceNativeSidecarError(
                f"science sidecar sequence is not contiguous for {cell_id}"
            )
        if any(
            row.instrument_manifest_sha256 != rows[0].instrument_manifest_sha256
            or row.science_semantic_acquisition_sha256
            != rows[0].science_semantic_acquisition_sha256
            or row.world_uid != rows[0].world_uid
            or row.logical_world_id != rows[0].logical_world_id
            for row in rows
        ):
            raise ScienceNativeSidecarError(
                f"science sidecar record identity changes within {cell_id}"
            )
        try:
            validate_science_observation_stream_v1(
                rows,
                producer_phase_registry_ordinals=dict(producer_phase_registry_ordinals),
                expected_start_world_local_sequence=0,
            )
        except Exception as exc:
            raise ScienceNativeSidecarError(
                f"science sidecar canonical stream validation failed for {cell_id}"
            ) from exc
        _validate_control_binding_join_v1(rows, cell_id=cell_id)
        records[cell_id] = rows

    frontier = body["frontier"]
    if not isinstance(frontier, dict):
        raise ScienceNativeSidecarError("science sidecar frontier is invalid")
    next_tick = frontier.get("next_absolute_tick")
    if type(next_tick) is not int or next_tick < 0:
        raise ScienceNativeSidecarError("science sidecar frontier tick is invalid")
    if (
        expected_next_absolute_tick is not None
        and next_tick != expected_next_absolute_tick
    ):
        raise ScienceNativeSidecarError("science sidecar frontier tick differs")
    observed_frontier = _frontier_for_records(records, next_absolute_tick=next_tick)
    if frontier != observed_frontier:
        raise ScienceNativeSidecarError(
            "science sidecar frontier is not self-consistent"
        )
    return NativeScienceSidecarV1(
        path=resolved,
        sidecar_sha256=observed_sha256,
        batch_id=batch_id,
        cell_ids=cell_ids,
        records_by_cell=tuple((cell_id, records[cell_id]) for cell_id in cell_ids),
        next_absolute_tick=next_tick,
        frontier=observed_frontier,
    )


def publish_native_science_sidecar_v1(
    sidecar: NativeScienceSidecarV1,
    plans_by_cell: Mapping[str, ScienceObservationPublicationPlanV1],
) -> tuple[tuple[str, ScienceWorldEvidencePublicationV1], ...]:
    """Publish each semantic world independently of its physical batch."""

    if set(plans_by_cell) != set(sidecar.cell_ids):
        raise ScienceNativeSidecarError(
            "publication plans do not exactly cover the sidecar worlds"
        )
    result: list[tuple[str, ScienceWorldEvidencePublicationV1]] = []
    for cell_id in sorted(sidecar.cell_ids):
        result.append(
            (
                cell_id,
                publish_science_world_evidence_v1(
                    (sidecar.records_for_cell(cell_id),), plans_by_cell[cell_id]
                ),
            )
        )
    return tuple(result)


__all__ = (
    "SCIENCE_NATIVE_SIDECAR_FRONTIER_SCHEMA_V1",
    "SCIENCE_NATIVE_SIDECAR_MODE_V1",
    "SCIENCE_NATIVE_SIDECAR_OBSERVATION_ABI_V1",
    "SCIENCE_NATIVE_SIDECAR_SCHEMA_V1",
    "NativeScienceSidecarV1",
    "ScienceNativeSidecarError",
    "load_native_science_sidecar_v1",
    "publish_native_science_sidecar_v1",
)
