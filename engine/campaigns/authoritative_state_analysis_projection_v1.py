"""Immutable production projection from persisted schema-bound analysis inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from engine.authoritative_state.admission import (
    canonical_json_bytes,
    canonical_sha256,
    require_sha256,
)
from engine.authoritative_state.chronicle import validate_chronicle_event
from engine.authoritative_state.conversions import (
    checked_finite_float32,
    checked_integer,
    checked_lineage_founder_id,
)
from engine.authoritative_state.projections import validate_projection_envelope
from engine.authoritative_state.schema import load_schema_v1

from .analysis import (
    AUTHORITATIVE_BINDING_KEYS,
    CampaignAnalysisError,
    authoritative_analysis_projection_v1,
)


SOURCE_SCHEMA_VERSION = "eiel_authoritative_analysis_source_v1"
SOURCE_RECORD_SCHEMA_VERSION = "eiel_authoritative_analysis_source_record_v1"
RECEIPT_SCHEMA_VERSION = "eiel_authoritative_analysis_projection_receipt_v1"
SOURCE_COORDINATE_KEYS = (
    "cell_id",
    "start_generation",
    "terminal_generation",
    "population_semantics",
)
SOURCE_FIELD_IDS = (
    "reporting.injury_start.v1",
    "reporting.injury_end.v1",
    "identity.lineage_founder_id.v1",
    "reporting.offspring_count.v1",
)
ANALYSIS_DTYPES = {
    "analysis.injury_delta.v1": "float32",
    "analysis.lineage_id.v1": "string",
    "analysis.offspring_count.v1": "int64",
}
_CANONICAL_SIGNED_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)")


class AuthoritativeAnalysisProjectionError(ValueError):
    """A persisted source, Chronicle event, or immutable receipt is invalid."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuthoritativeAnalysisProjectionError(f"{path} must be an object")
    return value


def _identity_bindings(value: Any) -> dict[str, str]:
    row = _mapping(value, path="analysis source identity_bindings")
    if set(row) != AUTHORITATIVE_BINDING_KEYS:
        raise AuthoritativeAnalysisProjectionError(
            "analysis source identity binding keys differ"
        )
    return {
        key: require_sha256(
            item, path=f"analysis_source.identity_bindings.{key}"
        )
        for key, item in row.items()
    }


def source_record_sha256(field_id: str, values: Sequence[Any]) -> str:
    """Hash one ordered persisted reporting column under a typed domain."""

    return canonical_sha256(
        {
            "schema_version": SOURCE_RECORD_SCHEMA_VERSION,
            "field_id": field_id,
            "values": list(values),
        }
    )


def authoritative_analysis_source_payload_v1(
    *,
    source_values: Mapping[str, Sequence[Any]],
    identity_bindings: Mapping[str, str],
    source_coordinates: Mapping[str, Any],
    row_agent_ids: Sequence[Any],
) -> dict[str, Any]:
    """Build the exact self-hashed payload that a reporting owner persists."""

    if set(source_values) != set(SOURCE_FIELD_IDS):
        raise AuthoritativeAnalysisProjectionError(
            "analysis source field set differs"
        )
    raw_values: dict[str, list[Any]] = {}
    for field_id in SOURCE_FIELD_IDS:
        raw = source_values[field_id]
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(
            raw, Sequence
        ):
            raise AuthoritativeAnalysisProjectionError(
                f"analysis source {field_id} must be an array"
            )
        raw_values[field_id] = list(raw)
    counts = {len(row) for row in raw_values.values()}
    if len(counts) != 1 or not counts or next(iter(counts)) < 1:
        raise AuthoritativeAnalysisProjectionError(
            "analysis source row counts differ"
        )
    raw_agent_ids = list(row_agent_ids)
    if len(raw_agent_ids) != next(iter(counts)):
        raise AuthoritativeAnalysisProjectionError(
            "analysis source agent IDs differ from row count"
        )
    agent_ids = [
        checked_integer(value, bits=64, signed=True, path=f"row_agent_ids[{index}]")
        for index, value in enumerate(raw_agent_ids)
    ]
    if any(value < 0 for value in agent_ids) or agent_ids != sorted(set(agent_ids)):
        raise AuthoritativeAnalysisProjectionError(
            "analysis source agent IDs must be unique ascending nonnegative int64"
        )
    coordinates = _mapping(
        source_coordinates, path="analysis source coordinates"
    )
    if set(coordinates) != set(SOURCE_COORDINATE_KEYS):
        raise AuthoritativeAnalysisProjectionError(
            "analysis source coordinate fields differ"
        )
    normalized_coordinates = {
        key: coordinates[key] for key in SOURCE_COORDINATE_KEYS
    }
    cell_id = normalized_coordinates["cell_id"]
    start_generation = normalized_coordinates["start_generation"]
    terminal_generation = normalized_coordinates["terminal_generation"]
    population_semantics = normalized_coordinates["population_semantics"]
    if (
        not isinstance(cell_id, str)
        or not cell_id
        or cell_id.strip() != cell_id
        or type(start_generation) is not int
        or start_generation < 0
        or type(terminal_generation) is not int
        or terminal_generation <= start_generation
        or population_semantics
        != "surviving_source_cohort_after_turnover_v1"
    ):
        raise AuthoritativeAnalysisProjectionError(
            "analysis source coordinates are invalid"
        )
    values = {
        "reporting.injury_start.v1": [
            checked_finite_float32(
                value, path=f"reporting.injury_start.v1[{index}]"
            )
            for index, value in enumerate(
                raw_values["reporting.injury_start.v1"]
            )
        ],
        "reporting.injury_end.v1": [
            checked_finite_float32(
                value, path=f"reporting.injury_end.v1[{index}]"
            )
            for index, value in enumerate(
                raw_values["reporting.injury_end.v1"]
            )
        ],
        "identity.lineage_founder_id.v1": [
            checked_lineage_founder_id(
                value, path=f"identity.lineage_founder_id.v1[{index}]"
            )
            for index, value in enumerate(
                raw_values["identity.lineage_founder_id.v1"]
            )
        ],
        "reporting.offspring_count.v1": [
            _checked_nonnegative_int32(
                value, path=f"reporting.offspring_count.v1[{index}]"
            )
            for index, value in enumerate(
                raw_values["reporting.offspring_count.v1"]
            )
        ],
    }
    unsigned = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "identity_bindings": _identity_bindings(identity_bindings),
        "source_coordinates": normalized_coordinates,
        "row_agent_ids": agent_ids,
        "source_values": values,
        "source_record_hashes": {
            field_id: source_record_sha256(field_id, row)
            for field_id, row in values.items()
        },
    }
    return {**unsigned, "source_payload_sha256": canonical_sha256(unsigned)}


def load_analysis_source_v1(path: Path) -> dict[str, Any]:
    """Load and fully validate an already-persisted analysis source artifact."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        source = _mapping(raw, path="analysis source")
        expected_fields = {
            "schema_version",
            "identity_bindings",
            "source_coordinates",
            "row_agent_ids",
            "source_values",
            "source_record_hashes",
            "source_payload_sha256",
        }
        if set(source) != expected_fields:
            raise AuthoritativeAnalysisProjectionError(
                "analysis source field set differs"
            )
        if source.get("schema_version") != SOURCE_SCHEMA_VERSION:
            raise AuthoritativeAnalysisProjectionError(
                "analysis source schema version differs"
            )
        supplied_hash = require_sha256(
            source.get("source_payload_sha256"),
            path="analysis_source.source_payload_sha256",
        )
        unsigned = dict(source)
        unsigned.pop("source_payload_sha256")
        if canonical_sha256(unsigned) != supplied_hash:
            raise AuthoritativeAnalysisProjectionError(
                "analysis source payload hash mismatch"
            )
        rebuilt = authoritative_analysis_source_payload_v1(
            source_values=_mapping(
                source.get("source_values"), path="analysis source values"
            ),
            identity_bindings=_mapping(
                source.get("identity_bindings"),
                path="analysis source identity_bindings",
            ),
            source_coordinates=_mapping(
                source.get("source_coordinates"),
                path="analysis source coordinates",
            ),
            row_agent_ids=source.get("row_agent_ids"),
        )
        if rebuilt != source:
            raise AuthoritativeAnalysisProjectionError(
                "analysis source record hashes differ"
            )
        return dict(source)
    except AuthoritativeAnalysisProjectionError:
        raise
    except (CampaignAnalysisError, OSError, TypeError, ValueError) as exc:
        raise AuthoritativeAnalysisProjectionError(
            f"analysis source rejected: {exc}"
        ) from exc


def _validate_registry_projection_dtypes() -> None:
    schema = load_schema_v1()
    registry = schema.components["field_registry.json"]
    contract = schema.components["analysis_projection_schema.json"]
    registry_types = {
        row["field_id"]: row["physical_type"] for row in registry["fields"]
    }
    contract_types = {
        row["field_id"]: row["dtype"] for row in contract["fields"]
    }
    if contract_types != ANALYSIS_DTYPES:
        raise AuthoritativeAnalysisProjectionError(
            "analysis projection contract dtypes differ"
        )
    if {
        field_id: registry_types.get(field_id) for field_id in ANALYSIS_DTYPES
    } != ANALYSIS_DTYPES:
        raise AuthoritativeAnalysisProjectionError(
            "analysis registry and projection dtypes differ"
        )
    offspring = next(
        row
        for row in registry["fields"]
        if row["field_id"] == "reporting.offspring_count.v1"
    )
    if (
        offspring.get("physical_type") != "int32"
        or offspring.get("signedness") != "signed"
        or offspring.get("valid_range")
        != {"minimum": 0, "maximum": 2**31 - 1}
    ):
        raise AuthoritativeAnalysisProjectionError(
            "reporting offspring count is not bounded signed int32"
        )
    for field_id in (
        "reporting.injury_start.v1",
        "reporting.injury_end.v1",
    ):
        if registry_types.get(field_id) != "float32":
            raise AuthoritativeAnalysisProjectionError(
                "reporting injury source is not float32"
            )


def _checked_nonnegative_int32(value: Any, *, path: str) -> int:
    result = checked_integer(value, bits=32, signed=True, path=path)
    if result < 0:
        raise AuthoritativeAnalysisProjectionError(f"{path} must be nonnegative")
    return result


def _exact_authority_from_event(
    event: Mapping[str, Any],
    *,
    expected_event_id: str,
    identity_bindings: Mapping[str, str],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    authority = _mapping(
        event.get("authoritative_state_schema_v1"),
        path="Chronicle authoritative_state_schema_v1",
    )
    if set(authority) != {"provider_projection", "chronicle_event"}:
        raise AuthoritativeAnalysisProjectionError(
            "Chronicle authoritative projection field set differs"
        )
    persisted_provider = _mapping(
        authority["provider_projection"], path="Chronicle provider projection"
    )
    persisted_chronicle = _mapping(
        authority["chronicle_event"], path="Chronicle schema event"
    )
    if persisted_chronicle.get("event_id") != expected_event_id:
        raise AuthoritativeAnalysisProjectionError(
            "Chronicle exact event ID differs"
        )
    schema = load_schema_v1()
    provider_contract = schema.components["provider_projection_schema.json"]
    chronicle_contract = schema.components["chronicle_schema.json"]
    provider = dict(persisted_provider)
    provider_field_ids = [
        row["field_id"] for row in provider_contract["fields"]
    ]
    provider_values = _mapping(
        provider.get("values"), path="Chronicle provider values"
    )
    provider_provenance = _mapping(
        provider.get("provenance"), path="Chronicle provider provenance"
    )
    if set(provider_values) != set(provider_field_ids) or set(
        provider_provenance
    ) != set(provider_field_ids):
        raise AuthoritativeAnalysisProjectionError(
            "Chronicle provider projection field IDs differ"
        )
    provider["values"] = {
        field_id: provider_values[field_id] for field_id in provider_field_ids
    }
    provider["provenance"] = {
        field_id: provider_provenance[field_id]
        for field_id in provider_field_ids
    }
    chronicle = dict(persisted_chronicle)
    chronicle_field_ids = [
        row["field_id"] for row in chronicle_contract["fields"]
    ]
    chronicle_values = _mapping(
        chronicle.get("values"), path="Chronicle schema values"
    )
    if set(chronicle_values) != set(chronicle_field_ids):
        raise AuthoritativeAnalysisProjectionError(
            "Chronicle schema event field IDs differ"
        )
    chronicle["values"] = {
        field_id: chronicle_values[field_id]
        for field_id in chronicle_field_ids
    }
    validate_projection_envelope(
        provider,
        contract=provider_contract,
        expected_bindings={
            **identity_bindings,
            "chronicle_schema_hash": chronicle_contract["contract_hash"],
        },
    )
    validate_chronicle_event(
        chronicle,
        contract=chronicle_contract,
        expected_bindings={
            **identity_bindings,
            "provider_projection_hash": provider_contract["contract_hash"],
        },
    )
    measurements = chronicle.get("open_values", {}).get("measurements")
    provenance = chronicle.get("open_value_provenance", {}).get("measurements")
    provider_hash = canonical_sha256(persisted_provider)
    if measurements != provider.get("values") or not isinstance(
        provenance, Mapping
    ):
        raise AuthoritativeAnalysisProjectionError(
            "Chronicle measurements differ from provider projection"
        )
    if any(
        not isinstance(row, Mapping)
        or row.get("source_field_id") != field_id
        or row.get("source_record_hash") != provider_hash
        or row.get("transform") != "copied"
        for field_id, row in provenance.items()
    ):
        raise AuthoritativeAnalysisProjectionError(
            "Chronicle provider provenance differs"
        )
    formal_bindings = {
        "run_id": "chronicle.run_id.v1",
        "generation": "chronicle.generation.v1",
        "step": "chronicle.step.v1",
        "phase_order": "chronicle.phase_order.v1",
        "intra_step_sequence": "chronicle.intra_step_sequence.v1",
        "event_type": "chronicle.event_type.v1",
        "subject_id": "chronicle.subject_id.v1",
        "source_component": "chronicle.source_component.v1",
    }
    values = chronicle["values"]
    if any(event.get(key) != values[field_id] for key, field_id in formal_bindings.items()):
        raise AuthoritativeAnalysisProjectionError(
            "formal and schema-bound Chronicle event identities differ"
        )
    return provider, chronicle


def stream_exact_chronicle_authority_v1(
    path: Path,
    *,
    expected_event_id: str,
    identity_bindings: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Stream JSONL and admit exactly one event with the requested schema ID."""

    if not isinstance(expected_event_id, str) or not expected_event_id.startswith(
        "evt_"
    ):
        raise AuthoritativeAnalysisProjectionError(
            "expected Chronicle event ID is invalid"
        )
    found: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
    found_line = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, Mapping):
                    raise AuthoritativeAnalysisProjectionError(
                        f"Chronicle line {line_number} is not an object"
                    )
                authority = event.get("authoritative_state_schema_v1")
                if not isinstance(authority, Mapping):
                    continue
                chronicle = authority.get("chronicle_event")
                if (
                    not isinstance(chronicle, Mapping)
                    or chronicle.get("event_id") != expected_event_id
                ):
                    continue
                if found is not None:
                    raise AuthoritativeAnalysisProjectionError(
                        "Chronicle exact event ID is duplicated"
                    )
                found = _exact_authority_from_event(
                    event,
                    expected_event_id=expected_event_id,
                    identity_bindings=identity_bindings,
                )
                found_line = line_number
    except AuthoritativeAnalysisProjectionError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise AuthoritativeAnalysisProjectionError(
            f"Chronicle stream rejected: {exc}"
        ) from exc
    if found is None:
        raise AuthoritativeAnalysisProjectionError(
            "Chronicle exact event ID is absent"
        )
    return dict(found[0]), dict(found[1]), found_line


def build_analysis_projection_receipt_v1(
    *,
    analysis_source_path: Path,
    chronicle_events_path: Path,
    chronicle_event_id: str,
) -> dict[str, Any]:
    """Validate persisted inputs and return one deterministic self-hashed receipt."""

    _validate_registry_projection_dtypes()
    source = load_analysis_source_v1(analysis_source_path)
    bindings = _identity_bindings(source["identity_bindings"])
    provider, chronicle, line_number = stream_exact_chronicle_authority_v1(
        chronicle_events_path,
        expected_event_id=chronicle_event_id,
        identity_bindings=bindings,
    )
    projection = authoritative_analysis_projection_v1(
        source_values=source["source_values"],
        source_record_hashes=source["source_record_hashes"],
        provider_projection=provider,
        identity_bindings=bindings,
    )
    unsigned = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "analysis_source": {
            "file_sha256": _file_sha256(analysis_source_path),
            "payload_sha256": source["source_payload_sha256"],
        },
        "chronicle_source": {
            "file_sha256": _file_sha256(chronicle_events_path),
            "line_number": line_number,
            "event_id": chronicle_event_id,
            "event_sha256": canonical_sha256(chronicle),
        },
        "identity_bindings": bindings,
        "analysis_projection": projection,
    }
    return {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}


def validate_analysis_projection_receipt_v1(
    receipt: Mapping[str, Any],
) -> str:
    """Validate the receipt's exact shape and self-hash."""

    if set(receipt) != {
        "schema_version",
        "analysis_source",
        "chronicle_source",
        "identity_bindings",
        "analysis_projection",
        "receipt_sha256",
    }:
        raise AuthoritativeAnalysisProjectionError(
            "analysis projection receipt field set differs"
        )
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise AuthoritativeAnalysisProjectionError(
            "analysis projection receipt schema differs"
        )
    supplied = require_sha256(
        receipt.get("receipt_sha256"), path="receipt.receipt_sha256"
    )
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    observed = canonical_sha256(unsigned)
    if supplied != observed:
        raise AuthoritativeAnalysisProjectionError(
            "analysis projection receipt hash mismatch"
        )
    analysis_source = _mapping(
        receipt.get("analysis_source"), path="receipt.analysis_source"
    )
    if set(analysis_source) != {"file_sha256", "payload_sha256"}:
        raise AuthoritativeAnalysisProjectionError(
            "receipt analysis source field set differs"
        )
    require_sha256(
        analysis_source["file_sha256"],
        path="receipt.analysis_source.file_sha256",
    )
    require_sha256(
        analysis_source["payload_sha256"],
        path="receipt.analysis_source.payload_sha256",
    )
    chronicle_source = _mapping(
        receipt.get("chronicle_source"), path="receipt.chronicle_source"
    )
    if set(chronicle_source) != {
        "file_sha256",
        "line_number",
        "event_id",
        "event_sha256",
    }:
        raise AuthoritativeAnalysisProjectionError(
            "receipt Chronicle source field set differs"
        )
    require_sha256(
        chronicle_source["file_sha256"],
        path="receipt.chronicle_source.file_sha256",
    )
    require_sha256(
        chronicle_source["event_sha256"],
        path="receipt.chronicle_source.event_sha256",
    )
    if (
        type(chronicle_source["line_number"]) is not int
        or chronicle_source["line_number"] < 1
        or not isinstance(chronicle_source["event_id"], str)
        or not chronicle_source["event_id"].startswith("evt_")
    ):
        raise AuthoritativeAnalysisProjectionError(
            "receipt Chronicle source identity is invalid"
        )
    bindings = _identity_bindings(receipt.get("identity_bindings"))
    projection = _mapping(
        receipt.get("analysis_projection"),
        path="receipt.analysis_projection",
    )
    schema = load_schema_v1()
    contract = schema.components["analysis_projection_schema.json"]
    validate_projection_envelope(
        projection,
        contract=contract,
        expected_bindings={
            **bindings,
            "provider_projection_hash": schema.components[
                "provider_projection_schema.json"
            ]["contract_hash"],
        },
    )
    _validate_projection_values(projection.get("values"))
    return supplied


def _validate_projection_values(value: Any) -> None:
    values = _mapping(value, path="receipt.analysis_projection.values")
    if tuple(values) != tuple(ANALYSIS_DTYPES):
        raise AuthoritativeAnalysisProjectionError(
            "analysis projection value fields differ"
        )
    injury = values["analysis.injury_delta.v1"]
    lineage = values["analysis.lineage_id.v1"]
    offspring = values["analysis.offspring_count.v1"]
    if any(
        isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence)
        for row in (injury, lineage, offspring)
    ):
        raise AuthoritativeAnalysisProjectionError(
            "analysis projection values must be arrays"
        )
    counts = {len(injury), len(lineage), len(offspring)}
    if len(counts) != 1 or next(iter(counts), 0) < 1:
        raise AuthoritativeAnalysisProjectionError(
            "analysis projection row counts differ"
        )
    for index, item in enumerate(injury):
        if checked_finite_float32(
            item, path=f"analysis.injury_delta.v1[{index}]"
        ) != item:
            raise AuthoritativeAnalysisProjectionError(
                "analysis injury delta is not exact float32"
            )
    for index, item in enumerate(lineage):
        if (
            not isinstance(item, str)
            or _CANONICAL_SIGNED_DECIMAL.fullmatch(item) is None
            or str(
                checked_lineage_founder_id(
                    int(item), path=f"analysis.lineage_id.v1[{index}]"
                )
            )
            != item
        ):
            raise AuthoritativeAnalysisProjectionError(
                "analysis lineage ID is not canonical decimal int32"
            )
    for index, item in enumerate(offspring):
        resolved = checked_integer(
            item,
            bits=64,
            signed=True,
            path=f"analysis.offspring_count.v1[{index}]",
        )
        if resolved < 0:
            raise AuthoritativeAnalysisProjectionError(
                "analysis offspring count must be nonnegative"
            )


def write_immutable_analysis_projection_receipt_v1(
    path: Path, receipt: Mapping[str, Any]
) -> str:
    """Atomically publish one receipt without ever replacing a prior path."""

    digest = validate_analysis_projection_receipt_v1(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(receipt) + b"\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    except FileExistsError as exc:
        raise AuthoritativeAnalysisProjectionError(
            f"immutable analysis projection output already exists: {path}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return digest


def write_immutable_analysis_source_v1(
    path: Path, source: Mapping[str, Any]
) -> str:
    """Atomically publish one fully validated analysis source artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(source) + b"\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        loaded = json.loads(temporary_path.read_text(encoding="utf-8"))
        rebuilt = authoritative_analysis_source_payload_v1(
            source_values=_mapping(
                loaded.get("source_values"), path="analysis source values"
            ),
            identity_bindings=_mapping(
                loaded.get("identity_bindings"),
                path="analysis source identity_bindings",
            ),
            source_coordinates=_mapping(
                loaded.get("source_coordinates"),
                path="analysis source coordinates",
            ),
            row_agent_ids=loaded.get("row_agent_ids"),
        )
        if rebuilt != loaded:
            raise AuthoritativeAnalysisProjectionError(
                "analysis source output differs after serialization"
            )
        os.link(temporary_path, path)
    except FileExistsError as exc:
        raise AuthoritativeAnalysisProjectionError(
            f"immutable analysis source output already exists: {path}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return require_sha256(
        source.get("source_payload_sha256"),
        path="analysis_source.source_payload_sha256",
    )


__all__ = [
    "AuthoritativeAnalysisProjectionError",
    "SOURCE_FIELD_IDS",
    "SOURCE_SCHEMA_VERSION",
    "authoritative_analysis_source_payload_v1",
    "build_analysis_projection_receipt_v1",
    "load_analysis_source_v1",
    "source_record_sha256",
    "stream_exact_chronicle_authority_v1",
    "validate_analysis_projection_receipt_v1",
    "write_immutable_analysis_source_v1",
    "write_immutable_analysis_projection_receipt_v1",
]
