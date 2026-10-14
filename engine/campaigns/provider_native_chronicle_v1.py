"""Generic durable Chronicle consumer for provider-native segment artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from engine.authoritative_state.chronicle import (
    chronicle_event_id,
    validate_chronicle_event,
)
from engine.authoritative_state.conversions import checked_integer
from engine.authoritative_state.projections import validate_projection_envelope
from engine.authoritative_state.schema import content_sha256, load_schema_v1
from engine.evolution.genome_upgrade_v2 import GenomeUpgradeV2Error
from engine.evolution.genome_upgrade_v2_telemetry import (
    CHRONICLE_SCHEMA_VERSION as GENOME_V2_CHRONICLE_SCHEMA_VERSION,
    project_genome_upgrade_v2_checkpoint_summary_v1,
)
from engine.perf.cuda_architecture_b_resident_reproduction_exposure_sidecar_v1 import (
    CONTINUITY_SCHEMA_VERSION as NATIVE_CONTINUITY_SCHEMA_VERSION,
)
from simulation.chronicle.schema import (
    MANIFEST_SCHEMA_VERSION,
    canonical_json,
    canonical_payload,
    event_order_key,
    make_event,
)
from simulation.chronicle.storage import ARTIFACT_NAMES, write_json, write_jsonl


RECEIPT_SCHEMA_VERSION = "eiel_provider_native_chronicle_receipt_v1"
STATE_SCHEMA_VERSION = "eiel_provider_native_chronicle_state_v1"
SUMMARY_SCHEMA_VERSION = "eiel_scientific_chronicle_summary_v1"
SOURCE_COMPLETENESS = "provider_native_segment_artifact_complete_v1"
CONTINUATION_REBIND_SCHEMA_VERSION = (
    "eiel_provider_native_chronicle_continuation_rebind_v1"
)
GENOME_V2_G0_BINDING_SCHEMA_VERSION = (
    "eiel_provider_native_chronicle_genome_v2_g0_binding_v1"
)
DURABLE_AUTHORITY_FILENAMES = (
    "provider_native_chronicle_state.json",
    ARTIFACT_NAMES["events"],
    ARTIFACT_NAMES["entries"],
    ARTIFACT_NAMES["episodes"],
    ARTIFACT_NAMES["contexts"],
    ARTIFACT_NAMES["summary"],
    ARTIFACT_NAMES["manifest"],
)
AUTHORITATIVE_BINDING_KEYS = frozenset(
    {
        "authoritative_schema_hash",
        "binary_schema_hash",
        "checkpoint_schema_hash",
    }
)


class ProviderNativeChronicleError(ValueError):
    """Provider-native Chronicle input or durable continuation is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _safe_name(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise ProviderNativeChronicleError(f"{field} is unsafe")
    return value


def _object(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderNativeChronicleError(f"{field} is malformed")
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProviderNativeChronicleError(f"{field} is invalid")
    return value


def _lower_digest(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProviderNativeChronicleError(f"{field} is invalid")
    return value


class _CursorProjectionIterable:
    """Expose one cursor payload at a time to an existing Chronicle reducer.

    This adapter deliberately is not a ``Sequence`` and never stores the
    payloads it yields.  It is used only at the consumer boundary so the
    durable Chronicle implementation can retain its existing validation and
    replay rules while source/checkpoint tensors remain cursor-owned.
    """

    __slots__ = ("_cursor", "_cell_ids", "_consumed")

    def __init__(self, cursor: Any, cell_ids: Sequence[str]) -> None:
        if cursor is None:
            raise ProviderNativeChronicleError(
                "Chronicle projection cursor is absent"
            )
        ordered_cell_ids = tuple(cell_ids)
        if tuple(getattr(cursor, "cell_ids", ())) != ordered_cell_ids:
            raise ProviderNativeChronicleError(
                "Chronicle projection cursor cell order differs"
            )
        if not callable(getattr(cursor, "next_cell", None)):
            raise ProviderNativeChronicleError(
                "Chronicle projection cursor lacks next_cell"
            )
        if not callable(getattr(cursor, "close", None)):
            raise ProviderNativeChronicleError(
                "Chronicle projection cursor lacks close"
            )
        if not callable(getattr(cursor, "fail", None)):
            raise ProviderNativeChronicleError(
                "Chronicle projection cursor lacks fail"
            )
        if not (
            callable(getattr(cursor, "release_cell", None))
            or callable(getattr(cursor, "import_and_release", None))
        ):
            raise ProviderNativeChronicleError(
                "Chronicle projection cursor lacks a release operation"
            )
        self._cursor = cursor
        self._cell_ids = ordered_cell_ids
        self._consumed = False

    def __len__(self) -> int:
        return len(self._cell_ids)

    def __iter__(self):
        if self._consumed:
            raise ProviderNativeChronicleError(
                "Chronicle projection cursor input was consumed twice"
            )
        self._consumed = True
        cursor = self._cursor
        try:
            for index, cell_id in enumerate(self._cell_ids):
                detached = cursor.next_cell()
                if detached is None:
                    raise ProviderNativeChronicleError(
                        "Chronicle projection cursor ended before every cell"
                    )
                if (
                    getattr(detached, "cell_id", None) != cell_id
                    or getattr(detached, "cell_index", None) != index
                ):
                    raise ProviderNativeChronicleError(
                        "Chronicle projection cursor emitted unordered cell"
                    )
                require_payload = getattr(detached, "require_payload", None)
                if not callable(require_payload):
                    raise ProviderNativeChronicleError(
                        "Chronicle projection cursor payload is not detachable"
                    )
                payload = require_payload()
                if not isinstance(payload, Mapping):
                    raise ProviderNativeChronicleError(
                        "Chronicle projection cursor payload is not a mapping"
                    )
                try:
                    yield payload
                finally:
                    release_cell = getattr(cursor, "release_cell", None)
                    if callable(release_cell):
                        release_cell(cell_id)
                    else:
                        cursor.import_and_release(cell_id, None)
            receipt = cursor.close()
            if getattr(receipt, "state", None) != "SEALED":
                raise ProviderNativeChronicleError(
                    "Chronicle projection cursor did not seal"
                )
        except BaseException as exc:
            if getattr(cursor, "state", None) not in {
                "SEALED",
                "FAILED_UNPUBLISHED",
            }:
                try:
                    cursor.fail(str(exc))
                except Exception:
                    pass
            raise


def _authoritative_contracts() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    schema = load_schema_v1()
    provider = schema.components["provider_projection_schema.json"]
    chronicle = schema.components["chronicle_schema.json"]
    return provider, chronicle


def _authoritative_identity_bindings(
    value: Mapping[str, Any],
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != AUTHORITATIVE_BINDING_KEYS:
        raise ProviderNativeChronicleError(
            "authoritative identity binding keys differ"
        )
    bindings = {
        key: _lower_digest(item, field=f"authoritative bindings.{key}")
        for key, item in value.items()
    }
    schema = load_schema_v1()
    expected = {
        "authoritative_schema_hash": schema.schema_hash,
        "checkpoint_schema_hash": schema.components["checkpoint_schema.json"][
            "contract_hash"
        ],
    }
    for key, wanted in expected.items():
        if bindings[key] != wanted:
            raise ProviderNativeChronicleError(
                f"authoritative identity binding mismatch: {key}"
            )
    return bindings


def _selected_parent_id(value: Any, *, index: int) -> int:
    result = checked_integer(
        value,
        bits=64,
        signed=True,
        path=f"reproduction.selected_parent_id.v1[{index}]",
    )
    if result < -1:
        raise ProviderNativeChronicleError(
            "selected parent ID must be -1 or nonnegative"
        )
    return result


def authoritative_provider_projection_v1(
    *,
    attempt: Mapping[str, Any],
    source_record_sha256: str,
    identity_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one exact native attempt through the canonical provider contract."""

    provider_contract, chronicle_contract = _authoritative_contracts()
    base = _authoritative_identity_bindings(identity_bindings)
    source_hash = _lower_digest(
        source_record_sha256, field="provider source record"
    )
    if not isinstance(attempt, Mapping):
        raise ProviderNativeChronicleError("reproduction attempt is malformed")
    eligible = attempt.get("eligible_parent_counts")
    selected = attempt.get("selected_parent_ids")
    if (
        not isinstance(eligible, Sequence)
        or isinstance(eligible, (str, bytes))
        or not isinstance(selected, Sequence)
        or isinstance(selected, (str, bytes))
        or len(eligible) != len(selected)
        or not eligible
    ):
        raise ProviderNativeChronicleError(
            "reproduction parent-role values are malformed"
        )
    success = attempt.get("success")
    if type(success) is not bool:
        raise ProviderNativeChronicleError(
            "reproduction success must be an exact boolean"
        )
    source_values = (
        checked_integer(
            attempt.get("attempt_index"),
            bits=64,
            signed=False,
            path="reproduction.attempt_index.v1",
        ),
        [
            checked_integer(
                value,
                bits=64,
                signed=False,
                path=f"reproduction.eligible_parent_count.v1[{index}]",
            )
            for index, value in enumerate(eligible)
        ],
        [
            _selected_parent_id(value, index=index)
            for index, value in enumerate(selected)
        ],
        success,
    )
    values = {}
    provenance = {}
    for contract_row, source_value in zip(
        provider_contract["fields"], source_values, strict=True
    ):
        output_field_id = contract_row["field_id"]
        values[output_field_id] = canonical_payload(source_value)
        provenance[output_field_id] = {
            "transform": contract_row["transform"],
            "source_field_ids": list(contract_row["source_field_ids"]),
            "source_record_hashes": [source_hash],
        }
    expected_bindings = {
        **base,
        "chronicle_schema_hash": chronicle_contract["contract_hash"],
    }
    envelope = {
        "projection_schema_version": provider_contract["schema_version"],
        "projection_schema_hash": provider_contract["contract_hash"],
        "bindings": expected_bindings,
        "output_manifest": canonical_payload(provider_contract["fields"]),
        "values": values,
        "provenance": provenance,
    }
    validate_projection_envelope(
        envelope,
        contract=provider_contract,
        expected_bindings=expected_bindings,
    )
    return envelope


def authoritative_chronicle_event_v1(
    *,
    run_id: str,
    generation: int,
    step: int,
    intra_step_sequence: int,
    subject_id: str,
    provider_projection: Mapping[str, Any],
    identity_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a native provider projection to one canonical Chronicle event."""

    provider_contract, chronicle_contract = _authoritative_contracts()
    base = _authoritative_identity_bindings(identity_bindings)
    provider_bindings = {
        **base,
        "chronicle_schema_hash": chronicle_contract["contract_hash"],
    }
    validate_projection_envelope(
        provider_projection,
        contract=provider_contract,
        expected_bindings=provider_bindings,
    )
    if not isinstance(run_id, str) or not run_id:
        raise ProviderNativeChronicleError("Chronicle run ID is invalid")
    if not isinstance(subject_id, str) or not subject_id:
        raise ProviderNativeChronicleError("Chronicle subject ID is invalid")
    values = {
        "chronicle.run_id.v1": run_id,
        "chronicle.generation.v1": checked_integer(
            generation,
            bits=64,
            signed=False,
            path="chronicle.generation.v1",
        ),
        "chronicle.step.v1": checked_integer(
            step, bits=64, signed=False, path="chronicle.step.v1"
        ),
        "chronicle.phase_order.v1": 720,
        "chronicle.intra_step_sequence.v1": checked_integer(
            intra_step_sequence,
            bits=64,
            signed=False,
            path="chronicle.intra_step_sequence.v1",
        ),
        "chronicle.event_type.v1": "provider_native_reproduction_attempt",
        "chronicle.subject_id.v1": subject_id,
        "chronicle.source_component.v1": (
            "engine.campaigns.segment_artifact_observer"
        ),
    }
    provider_hash = _digest(provider_projection)
    measurements = canonical_payload(provider_projection["values"])
    measurement_provenance = {
        field_id: {
            "source_field_id": field_id,
            "source_record_hash": provider_hash,
            "transform": "copied",
        }
        for field_id in measurements
    }
    expected_bindings = {
        **base,
        "provider_projection_hash": provider_contract["contract_hash"],
    }
    event = {
        "chronicle_schema_version": chronicle_contract["schema_version"],
        "chronicle_schema_hash": chronicle_contract["contract_hash"],
        "bindings": expected_bindings,
        "field_manifest": canonical_payload(chronicle_contract["fields"]),
        "values": values,
        "open_values": {"measurements": measurements},
        "open_value_provenance": {
            "measurements": measurement_provenance
        },
    }
    event["event_id"] = chronicle_event_id(
        contract=chronicle_contract, values=values
    )
    validate_chronicle_event(
        event,
        contract=chronicle_contract,
        expected_bindings=expected_bindings,
    )
    return event


@dataclass(frozen=True)
class ProviderNativeChronicleReceiptV1:
    """Validated batch receipt returned by the generic segment consumer."""

    campaign_id: str
    batch_id: str
    profile: str
    finalized: bool
    cell_receipts: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        _safe_name(self.campaign_id, field="campaign_id")
        _safe_name(self.batch_id, field="batch_id")
        if (
            self.profile != "scientific_full"
            or type(self.finalized) is not bool
            or not isinstance(self.cell_receipts, Mapping)
            or not self.cell_receipts
            or any(
                not isinstance(cell_id, str)
                or not isinstance(receipt, Mapping)
                or receipt.get("cell_id") != cell_id
                for cell_id, receipt in self.cell_receipts.items()
            )
        ):
            raise ProviderNativeChronicleError(
                "provider-native Chronicle receipt is malformed"
            )

    def payload(self) -> dict[str, Any]:
        return canonical_payload(
            {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "campaign_id": self.campaign_id,
                "batch_id": self.batch_id,
                "profile": self.profile,
                "source_completeness": SOURCE_COMPLETENESS,
                "finalized": self.finalized,
                "cell_receipts": self.cell_receipts,
            }
        )

    @property
    def digest_sha256(self) -> str:
        return _digest(self.payload())

    @classmethod
    def from_payload(
        cls, value: Mapping[str, Any]
    ) -> "ProviderNativeChronicleReceiptV1":
        fields = {
            "schema_version",
            "campaign_id",
            "batch_id",
            "profile",
            "source_completeness",
            "finalized",
            "cell_receipts",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ProviderNativeChronicleError(
                "provider-native Chronicle receipt field set differs"
            )
        if (
            value.get("schema_version") != RECEIPT_SCHEMA_VERSION
            or value.get("source_completeness") != SOURCE_COMPLETENESS
        ):
            raise ProviderNativeChronicleError(
                "provider-native Chronicle receipt schema differs"
            )
        return cls(
            campaign_id=value["campaign_id"],
            batch_id=value["batch_id"],
            profile=value["profile"],
            finalized=value["finalized"],
            cell_receipts=canonical_payload(value["cell_receipts"]),
        )


@dataclass(frozen=True)
class ProviderNativeChronicleContinuationSeedV1:
    """Verified immutable prefix for a fresh provider-native Chronicle owner."""

    prior_receipt: ProviderNativeChronicleReceiptV1
    prior_authority_sha256: str
    prior_continuity_sha256: str
    next_segment_index: int
    cell_ids: tuple[str, ...]
    cell_chain_sha256: tuple[str, ...]
    source_states: tuple[Mapping[str, Any], ...]

    @property
    def binding_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CONTINUATION_REBIND_SCHEMA_VERSION,
            "source_campaign_id": self.prior_receipt.campaign_id,
            "source_batch_id": self.prior_receipt.batch_id,
            "prior_authority_sha256": self.prior_authority_sha256,
            "prior_receipt_sha256": self.prior_receipt.digest_sha256,
            "prior_continuity_sha256": self.prior_continuity_sha256,
            "next_segment_index": self.next_segment_index,
            "cell_ids": list(self.cell_ids),
            "cell_chain_sha256": list(self.cell_chain_sha256),
        }


def _verified_continuation(
    value: Mapping[str, Any],
    *,
    cell_count: int,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "segment_count",
        "cell_chain_sha256",
        "cumulative_attempt_counts",
        "cumulative_success_counts",
        "cumulative_failure_counts",
        "cumulative_snapshot_counts",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema_version") != NATIVE_CONTINUITY_SCHEMA_VERSION
    ):
        raise ProviderNativeChronicleError(
            "prior native continuity schema differs"
        )
    row = canonical_payload(value)
    _integer(
        row["segment_count"], field="prior continuity segment_count", minimum=1
    )
    sequences = (
        row["cell_chain_sha256"],
        row["cumulative_attempt_counts"],
        row["cumulative_success_counts"],
        row["cumulative_failure_counts"],
        row["cumulative_snapshot_counts"],
    )
    if any(not isinstance(values, list) or len(values) != cell_count for values in sequences):
        raise ProviderNativeChronicleError(
            "prior native continuity cell cardinality differs"
        )
    for digest in row["cell_chain_sha256"]:
        _lower_digest(digest, field="prior continuity cell chain")
    for field in (
        "cumulative_attempt_counts",
        "cumulative_success_counts",
        "cumulative_failure_counts",
        "cumulative_snapshot_counts",
    ):
        for value_item in row[field]:
            _integer(value_item, field=f"prior continuity {field}")
    if any(
        attempts != successes + failures
        for attempts, successes, failures in zip(
            row["cumulative_attempt_counts"],
            row["cumulative_success_counts"],
            row["cumulative_failure_counts"],
            strict=True,
        )
    ):
        raise ProviderNativeChronicleError(
            "prior native continuity counters contradict"
        )
    return row


def _verified_segment_bindings(
    value: Any,
    *,
    segment_count: int,
    final_chain_sha256: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != segment_count:
        raise ProviderNativeChronicleError(
            "prior Chronicle segment count differs"
        )
    bindings: list[dict[str, Any]] = []
    previous_chain = "0" * 64
    previous_end: int | None = None
    required_fields = {
        "segment_index",
        "start_generation",
        "end_generation",
        "source_segment_sha256",
        "cell_segment_sha256",
        "previous_chain_sha256",
        "chain_sha256_after_segment",
    }
    legacy_summary_fields = {"genome_v2_summary_sha256"}
    exact_summary_fields = {
        "genome_v2_summary_sha256",
        "genome_v2_projection_sha256",
        "genome_v2_generation",
        "genome_v2_tick",
        "genome_v2_sidecar_payload_sha256",
    }
    for expected_index, raw in enumerate(value):
        if (
            not isinstance(raw, Mapping)
            or not required_fields <= set(raw)
            or set(raw) - required_fields - exact_summary_fields
        ):
            raise ProviderNativeChronicleError(
                "prior Chronicle segment binding fields differ"
            )
        row = canonical_payload(raw)
        if _integer(row["segment_index"], field="segment_index") != expected_index:
            raise ProviderNativeChronicleError(
                "prior Chronicle segment index has a gap"
            )
        start = _integer(row["start_generation"], field="start_generation")
        end = _integer(
            row["end_generation"],
            field="end_generation",
            minimum=start + 1,
        )
        if previous_end is not None and start != previous_end:
            raise ProviderNativeChronicleError(
                "prior Chronicle generation range has a gap"
            )
        source_digest = _lower_digest(
            row["source_segment_sha256"], field="source_segment_sha256"
        )
        del source_digest
        segment_digest = _lower_digest(
            row["cell_segment_sha256"], field="cell_segment_sha256"
        )
        supplied_previous = _lower_digest(
            row["previous_chain_sha256"], field="previous_chain_sha256"
        )
        supplied_after = _lower_digest(
            row["chain_sha256_after_segment"],
            field="chain_sha256_after_segment",
        )
        supplied_summary_fields = set(row) - required_fields
        if supplied_summary_fields not in (
            set(),
            legacy_summary_fields,
            exact_summary_fields,
        ):
            raise ProviderNativeChronicleError(
                "prior Chronicle Genome V2 segment binding fields differ"
            )
        if supplied_summary_fields:
            _lower_digest(
                row["genome_v2_summary_sha256"],
                field="genome_v2_summary_sha256",
            )
        if supplied_summary_fields == exact_summary_fields:
            _lower_digest(
                row["genome_v2_projection_sha256"],
                field="genome_v2_projection_sha256",
            )
            _lower_digest(
                row["genome_v2_sidecar_payload_sha256"],
                field="genome_v2_sidecar_payload_sha256",
            )
            if (
                _integer(
                    row["genome_v2_generation"],
                    field="genome_v2_generation",
                )
                != end
                or _integer(
                    row["genome_v2_tick"],
                    field="genome_v2_tick",
                )
                != 0
            ):
                raise ProviderNativeChronicleError(
                    "prior Chronicle Genome V2 segment milestone differs"
                )
        expected_after = _digest(
            {
                "schema_version": NATIVE_CONTINUITY_SCHEMA_VERSION,
                "segment_index": expected_index,
                "previous_chain_sha256": previous_chain,
                "segment_sha256": segment_digest,
            }
        )
        if supplied_previous != previous_chain or supplied_after != expected_after:
            raise ProviderNativeChronicleError(
                "prior Chronicle native segment chain differs"
            )
        bindings.append(row)
        previous_chain = supplied_after
        previous_end = end
    if previous_chain != final_chain_sha256:
        raise ProviderNativeChronicleError(
            "prior Chronicle terminal chain differs from native continuity"
        )
    return bindings


def _verified_genome_v2_summaries(
    value: Any,
    *,
    cell_id: str,
    segment_bindings: Sequence[Mapping[str, Any]],
    generation_zero_binding: Any = None,
) -> list[dict[str, Any]]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise ProviderNativeChronicleError(
            "prior Chronicle Genome V2 summaries are malformed"
        )
    required_fields = {
        "schema_version",
        "cell_id",
        "generation",
        "tick",
        "sidecar_payload_sha256",
        "summary_sha256",
        "summary",
    }
    summaries: list[dict[str, Any]] = []
    previous_generation = -1
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != required_fields:
            raise ProviderNativeChronicleError(
                "prior Chronicle Genome V2 summary fields differ"
            )
        row = canonical_payload(raw)
        generation = _integer(
            row["generation"], field="Genome V2 summary generation"
        )
        _integer(row["tick"], field="Genome V2 summary tick")
        if (
            row["schema_version"] != GENOME_V2_CHRONICLE_SCHEMA_VERSION
            or row["cell_id"] != cell_id
            or generation <= previous_generation
        ):
            raise ProviderNativeChronicleError(
                "prior Chronicle Genome V2 summary identity or order differs"
            )
        _lower_digest(
            row["sidecar_payload_sha256"],
            field="Genome V2 summary sidecar_payload_sha256",
        )
        supplied_summary_sha256 = _lower_digest(
            row["summary_sha256"],
            field="Genome V2 summary summary_sha256",
        )
        if (
            not isinstance(row["summary"], Mapping)
            or content_sha256(row["summary"]) != supplied_summary_sha256
        ):
            raise ProviderNativeChronicleError(
                "prior Chronicle Genome V2 summary content differs"
            )
        summaries.append(row)
        previous_generation = generation

    has_generation_zero = bool(summaries and summaries[0]["generation"] == 0)
    if generation_zero_binding is None:
        if has_generation_zero:
            raise ProviderNativeChronicleError(
                "prior Chronicle Genome V2 G0 binding is absent"
            )
    else:
        g0_fields = {
            "schema_version",
            "generation",
            "tick",
            "source_projection_sha256",
            "sidecar_payload_sha256",
            "summary_sha256",
            "projection_sha256",
        }
        if (
            not isinstance(generation_zero_binding, Mapping)
            or set(generation_zero_binding) != g0_fields
            or generation_zero_binding.get("schema_version")
            != GENOME_V2_G0_BINDING_SCHEMA_VERSION
            or generation_zero_binding.get("generation") != 0
            or generation_zero_binding.get("tick") != 0
            or not has_generation_zero
        ):
            raise ProviderNativeChronicleError(
                "prior Chronicle Genome V2 G0 binding differs"
            )
        for field in (
            "source_projection_sha256",
            "sidecar_payload_sha256",
            "summary_sha256",
            "projection_sha256",
        ):
            _lower_digest(
                generation_zero_binding[field],
                field=f"Genome V2 G0 binding {field}",
            )
        first = summaries[0]
        if (
            generation_zero_binding["sidecar_payload_sha256"]
            != first["sidecar_payload_sha256"]
            or generation_zero_binding["summary_sha256"]
            != first["summary_sha256"]
            or generation_zero_binding["projection_sha256"]
            != content_sha256(first)
        ):
            raise ProviderNativeChronicleError(
                "prior Chronicle Genome V2 G0 projection differs"
            )
    segment_summaries = summaries[1:] if has_generation_zero else summaries
    expected_bindings = [
        row
        for row in segment_bindings
        if "genome_v2_summary_sha256" in row
    ]
    if len(segment_summaries) != len(expected_bindings):
        raise ProviderNativeChronicleError(
            "prior Chronicle Genome V2 summary count differs from segments"
        )
    if has_generation_zero and summaries[0]["tick"] != 0:
        raise ProviderNativeChronicleError(
            "prior Chronicle Genome V2 G0 tick differs"
        )
    if has_generation_zero and len(expected_bindings) != len(segment_bindings):
        raise ProviderNativeChronicleError(
            "prior Chronicle Genome V2 segment summary is absent"
        )
    for summary, binding in zip(
        segment_summaries, expected_bindings, strict=True
    ):
        if (
            summary["generation"] != binding["end_generation"]
            or summary["tick"] != 0
            or summary["summary_sha256"]
            != binding["genome_v2_summary_sha256"]
        ):
            raise ProviderNativeChronicleError(
                "prior Chronicle Genome V2 summary substitutes a segment"
            )
        exact_fields = {
            "genome_v2_summary_sha256",
            "genome_v2_projection_sha256",
            "genome_v2_generation",
            "genome_v2_tick",
            "genome_v2_sidecar_payload_sha256",
        }
        if exact_fields <= set(binding) and (
            summary["generation"] != binding["genome_v2_generation"]
            or summary["tick"] != binding["genome_v2_tick"]
            or summary["sidecar_payload_sha256"]
            != binding["genome_v2_sidecar_payload_sha256"]
            or content_sha256(summary)
            != binding["genome_v2_projection_sha256"]
        ):
            raise ProviderNativeChronicleError(
                "prior Chronicle Genome V2 projection substitutes a segment"
            )
    return summaries


def _verified_source_state(
    *,
    receipt: ProviderNativeChronicleReceiptV1,
    cell_id: str,
    cell_receipt: Mapping[str, Any],
    segment_count: int,
    attempt_cursor: int,
    final_chain_sha256: str,
) -> dict[str, Any]:
    root_value = cell_receipt.get("chronicle_root")
    if not isinstance(root_value, str):
        raise ProviderNativeChronicleError("prior Chronicle root is absent")
    root = Path(root_value)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ProviderNativeChronicleError("prior Chronicle root is unsafe")
    state_path = root / "provider_native_chronicle_state.json"
    manifest_path = root / ARTIFACT_NAMES["manifest"]
    summary_path = root / ARTIFACT_NAMES["summary"]
    authority_paths = {
        filename: root / filename
        for filename in DURABLE_AUTHORITY_FILENAMES
    }
    if any(
        path.is_symlink() or not path.is_file()
        for path in authority_paths.values()
    ):
        raise ProviderNativeChronicleError(
            "prior Chronicle durable authority is incomplete"
        )
    durable_digests = cell_receipt.get("durable_artifact_sha256")
    if (
        not isinstance(durable_digests, Mapping)
        or set(durable_digests) != set(DURABLE_AUTHORITY_FILENAMES)
        or any(
            _sha256(path)
            != _lower_digest(
                durable_digests.get(filename),
                field=f"durable_artifact_sha256.{filename}",
            )
            for filename, path in authority_paths.items()
        )
    ):
        raise ProviderNativeChronicleError(
            "prior Chronicle durable authority digest differs"
        )
    if (
        _sha256(manifest_path)
        != _lower_digest(
            cell_receipt.get("chronicle_manifest_sha256"),
            field="chronicle_manifest_sha256",
        )
        or _sha256(summary_path)
        != _lower_digest(
            cell_receipt.get("chronicle_summary_sha256"),
            field="chronicle_summary_sha256",
        )
    ):
        raise ProviderNativeChronicleError(
            "prior Chronicle durable artifact digest differs"
        )
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProviderNativeChronicleError(
            "prior Chronicle state is unreadable"
        ) from exc
    identity = {
        "schema_version": STATE_SCHEMA_VERSION,
        "campaign_id": receipt.campaign_id,
        "batch_id": receipt.batch_id,
        "cell_id": cell_id,
        "profile": receipt.profile,
        "finalized": True,
        "next_segment_index": segment_count,
        "attempt_cursor": attempt_cursor,
    }
    if not isinstance(state, dict) or any(
        state.get(field) != expected for field, expected in identity.items()
    ):
        raise ProviderNativeChronicleError(
            "prior Chronicle state identity or cursor differs"
        )
    bindings = _verified_segment_bindings(
        state.get("segments"),
        segment_count=segment_count,
        final_chain_sha256=final_chain_sha256,
    )
    if cell_receipt.get("segment_bindings") != bindings:
        raise ProviderNativeChronicleError(
            "prior Chronicle receipt substitutes segment bindings"
        )
    summaries = _verified_genome_v2_summaries(
        state.get("genome_v2_summaries"),
        cell_id=cell_id,
        segment_bindings=bindings,
        generation_zero_binding=state.get(
            "genome_v2_generation_zero_binding"
        ),
    )
    try:
        durable_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProviderNativeChronicleError(
            "prior Chronicle summary is unreadable"
        ) from exc
    if (
        not isinstance(durable_summary, Mapping)
        or durable_summary.get("genome_v2_generation_summaries") != summaries
    ):
        raise ProviderNativeChronicleError(
            "prior Chronicle durable Genome V2 summary differs from state"
        )
    if (
        cell_receipt.get("cell_id") != cell_id
        or cell_receipt.get("finalized") is not True
        or cell_receipt.get("segment_count") != segment_count
        or cell_receipt.get("attempt_cursor") != attempt_cursor
        or cell_receipt.get("event_cursor") != state.get("event_cursor")
    ):
        raise ProviderNativeChronicleError(
            "prior Chronicle cell receipt differs from durable state"
        )
    return canonical_payload(state)


def bind_provider_native_chronicle_continuation_v1(
    *,
    prior_receipt: ProviderNativeChronicleReceiptV1,
    prior_continuity: Mapping[str, Any],
    prior_authority_sha256: str,
    cell_ids: Sequence[str],
) -> ProviderNativeChronicleContinuationSeedV1:
    """Bind a verified Chronicle prefix to exact native checkpoint continuity."""

    if not isinstance(prior_receipt, ProviderNativeChronicleReceiptV1):
        raise ProviderNativeChronicleError(
            "prior_receipt must be a typed verified Chronicle receipt"
        )
    if prior_receipt.finalized is not True:
        raise ProviderNativeChronicleError(
            "prior Chronicle receipt must be finalized"
        )
    authority = _lower_digest(
        prior_authority_sha256, field="prior_authority_sha256"
    )
    cells = tuple(_safe_name(value, field="cell_id") for value in cell_ids)
    if (
        not cells
        or len(set(cells)) != len(cells)
        or set(prior_receipt.cell_receipts) != set(cells)
    ):
        raise ProviderNativeChronicleError(
            "prior Chronicle receipt cell identities differ"
        )
    continuity = _verified_continuation(
        prior_continuity,
        cell_count=len(cells),
    )
    segment_count = continuity["segment_count"]
    source_states = tuple(
        _verified_source_state(
            receipt=prior_receipt,
            cell_id=cell_id,
            cell_receipt=prior_receipt.cell_receipts[cell_id],
            segment_count=segment_count,
            attempt_cursor=continuity["cumulative_attempt_counts"][index],
            final_chain_sha256=continuity["cell_chain_sha256"][index],
        )
        for index, cell_id in enumerate(cells)
    )
    return ProviderNativeChronicleContinuationSeedV1(
        prior_receipt=prior_receipt,
        prior_authority_sha256=authority,
        prior_continuity_sha256=_digest(continuity),
        next_segment_index=segment_count,
        cell_ids=cells,
        cell_chain_sha256=tuple(continuity["cell_chain_sha256"]),
        source_states=source_states,
    )


class ProviderNativeChronicleConsumerV1:
    """Read-only, RNG-free projection of exact settlement artifacts."""

    def __init__(
        self,
        *,
        batch_root: Path,
        campaign_id: str,
        batch_id: str,
        cell_ids: Sequence[str],
        profile: str,
        authoritative_identity_bindings: Mapping[str, Any] | None = None,
    ) -> None:
        if profile != "scientific_full":
            raise ProviderNativeChronicleError(
                "provider-native segment Chronicles require scientific_full"
            )
        self.batch_root = Path(batch_root).resolve()
        self.campaign_id = _safe_name(campaign_id, field="campaign_id")
        self.batch_id = _safe_name(batch_id, field="batch_id")
        self.cell_ids = tuple(_safe_name(value, field="cell_id") for value in cell_ids)
        if not self.cell_ids or len(set(self.cell_ids)) != len(self.cell_ids):
            raise ProviderNativeChronicleError("cell identities are empty or duplicated")
        self.profile = profile
        self._authoritative_identity_bindings = (
            _authoritative_identity_bindings(authoritative_identity_bindings)
            if authoritative_identity_bindings is not None
            else None
        )
        self._continuation_binding: Mapping[str, Any] | None = None

    @classmethod
    def from_checkpoint_continuation(
        cls,
        *,
        batch_root: Path,
        campaign_id: str,
        batch_id: str,
        cell_ids: Sequence[str],
        profile: str,
        prior_receipt: ProviderNativeChronicleReceiptV1,
        prior_continuity: Mapping[str, Any],
        prior_authority_sha256: str,
        authoritative_identity_bindings: Mapping[str, Any] | None = None,
    ) -> "ProviderNativeChronicleConsumerV1":
        """Create a fresh Chronicle owner seeded from one verified exact prefix."""

        consumer = cls(
            batch_root=batch_root,
            campaign_id=campaign_id,
            batch_id=batch_id,
            cell_ids=cell_ids,
            profile=profile,
            authoritative_identity_bindings=authoritative_identity_bindings,
        )
        if (
            prior_receipt.campaign_id != consumer.campaign_id
            or prior_receipt.profile != consumer.profile
            or prior_receipt.batch_id == consumer.batch_id
        ):
            raise ProviderNativeChronicleError(
                "Chronicle continuation owner identity is not a fresh exact rebind"
            )
        seed = bind_provider_native_chronicle_continuation_v1(
            prior_receipt=prior_receipt,
            prior_continuity=prior_continuity,
            prior_authority_sha256=prior_authority_sha256,
            cell_ids=consumer.cell_ids,
        )
        if any(consumer._root(cell_id).exists() for cell_id in consumer.cell_ids):
            raise ProviderNativeChronicleError(
                "Chronicle continuation target already exists"
            )
        binding = {
            **seed.binding_payload,
            "restored_batch_id": consumer.batch_id,
        }
        consumer._continuation_binding = canonical_payload(binding)
        states: list[tuple[str, dict[str, Any]]] = []
        for cell_id, source_state in zip(
            consumer.cell_ids, seed.source_states, strict=True
        ):
            state = canonical_payload(source_state)
            state.update(
                {
                    "campaign_id": consumer.campaign_id,
                    "batch_id": consumer.batch_id,
                    "cell_id": cell_id,
                    "profile": consumer.profile,
                    "finalized": False,
                    "continuation_binding": binding,
                }
            )
            states.append((cell_id, state))
        for cell_id, state in states:
            consumer._persist(cell_id, state, finalized=False)
        return consumer

    def _root(self, cell_id: str) -> Path:
        return self.batch_root / "cells" / cell_id / "chronicle"

    def _state_path(self, cell_id: str) -> Path:
        return self._root(cell_id) / "provider_native_chronicle_state.json"

    def _empty_state(self, cell_id: str) -> dict[str, Any]:
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "batch_id": self.batch_id,
            "cell_id": cell_id,
            "profile": self.profile,
            "next_segment_index": 0,
            "event_cursor": 0,
            "attempt_cursor": 0,
            "segments": [],
            "events": [],
            "finalized": False,
        }
        if self._authoritative_identity_bindings is not None:
            state["authoritative_identity_bindings"] = dict(
                self._authoritative_identity_bindings
            )
        return state

    def _load(self, cell_id: str) -> dict[str, Any]:
        path = self._state_path(cell_id)
        if not path.exists():
            return self._empty_state(cell_id)
        if path.is_symlink() or not path.is_file():
            raise ProviderNativeChronicleError("Chronicle state path is unsafe")
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProviderNativeChronicleError("Chronicle state is unreadable") from exc
        expected = {
            "schema_version": STATE_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "batch_id": self.batch_id,
            "cell_id": cell_id,
            "profile": self.profile,
        }
        if not isinstance(state, dict) or any(
            state.get(name) != value for name, value in expected.items()
        ):
            raise ProviderNativeChronicleError("Chronicle continuation identity differs")
        if state.get("authoritative_identity_bindings") != (
            self._authoritative_identity_bindings
        ):
            raise ProviderNativeChronicleError(
                "Chronicle authoritative identity binding differs"
            )
        stored_binding = state.get("continuation_binding")
        if (
            self._continuation_binding is None
            and stored_binding is not None
        ):
            raise ProviderNativeChronicleError(
                "checkpoint-bound Chronicle requires its verified rebind factory"
            )
        if (
            self._continuation_binding is not None
            and stored_binding != self._continuation_binding
        ):
            raise ProviderNativeChronicleError(
                "Chronicle continuation authority or digest differs"
            )
        segments = state.get("segments")
        if not isinstance(segments, list):
            raise ProviderNativeChronicleError(
                "Chronicle segment authority is malformed"
            )
        final_chain = (
            _lower_digest(
                segments[-1].get("chain_sha256_after_segment"),
                field="Chronicle terminal chain",
            )
            if segments and isinstance(segments[-1], Mapping)
            else "0" * 64
        )
        verified_segments = _verified_segment_bindings(
            segments,
            segment_count=len(segments),
            final_chain_sha256=final_chain,
        )
        _verified_genome_v2_summaries(
            state.get("genome_v2_summaries"),
            cell_id=cell_id,
            segment_bindings=verified_segments,
            generation_zero_binding=state.get(
                "genome_v2_generation_zero_binding"
            ),
        )
        return state

    @staticmethod
    def _cell_segment_sha256(
        artifact: Mapping[str, Any], cell: Mapping[str, Any]
    ) -> str:
        supplied = cell.get("segment_sha256")
        if not isinstance(supplied, str) or len(supplied) != 64:
            raise ProviderNativeChronicleError("cell segment hash is missing")
        unsigned = dict(cell)
        unsigned.pop("segment_sha256", None)
        unsigned.pop("previous_chain_sha256", None)
        unsigned.pop("chain_sha256_after_segment", None)
        expected = _digest(
            {
                "schema_version": artifact.get("schema_version"),
                "segment_index": artifact.get("segment_index"),
                "cell": unsigned,
            }
        )
        if supplied != expected:
            raise ProviderNativeChronicleError("cell segment hash does not verify")
        return supplied

    def observe_generation_zero(
        self,
        *,
        cell_ids: Sequence[str],
        checkpoint_projections: Sequence[Mapping[str, Any]] | None = None,
        checkpoint_projection_cursor: Any | None = None,
        source_projection_sha256_by_cell: Mapping[str, str],
        source_projection_digest: Callable[[Mapping[str, Any]], str],
    ) -> None:
        """Bind an exact Genome V2 G0 projection before the first segment."""

        if (checkpoint_projections is None) == (
            checkpoint_projection_cursor is None
        ):
            raise ProviderNativeChronicleError(
                "G0 requires exactly one projection input"
            )
        projection_input = (
            _CursorProjectionIterable(
                checkpoint_projection_cursor, self.cell_ids
            )
            if checkpoint_projection_cursor is not None
            else checkpoint_projections
        )

        if (
            tuple(cell_ids) != self.cell_ids
            or len(projection_input) != len(self.cell_ids)
            or not isinstance(source_projection_sha256_by_cell, Mapping)
            or set(source_projection_sha256_by_cell) != set(self.cell_ids)
        ):
            raise ProviderNativeChronicleError(
                "G0 checkpoints differ from physical cell order"
            )
        source_projection_sha256 = {
            cell_id: _lower_digest(
                source_projection_sha256_by_cell[cell_id],
                field=f"G0 source projection {cell_id}",
            )
            for cell_id in self.cell_ids
        }
        recomputed_source_sha256: dict[str, str] = {}
        try:
            projected: list[Mapping[str, Any] | None] = []
            for cell_id, checkpoint_projection in zip(
                self.cell_ids, projection_input, strict=True
            ):
                projected_row = project_genome_upgrade_v2_checkpoint_summary_v1(
                    checkpoint_projection,
                    expected_cell_id=cell_id,
                )
                projected.append(projected_row)
                if projected_row is not None and callable(
                    source_projection_digest
                ):
                    recomputed_source_sha256[cell_id] = _lower_digest(
                        source_projection_digest(checkpoint_projection),
                        field=f"recomputed G0 source projection {cell_id}",
                    )
        except GenomeUpgradeV2Error as exc:
            raise ProviderNativeChronicleError(
                f"Genome V2 Chronicle G0 projection is invalid: {exc}"
            ) from exc
        except Exception as exc:
            raise ProviderNativeChronicleError(
                f"Genome V2 Chronicle G0 source digest is invalid: {exc}"
            ) from exc
        present = [row is not None for row in projected]
        if not any(present):
            return
        if not all(present):
            raise ProviderNativeChronicleError(
                "Genome V2 Chronicle G0 presence differs across physical cells"
            )
        if not callable(source_projection_digest):
            raise ProviderNativeChronicleError(
                "Genome V2 Chronicle G0 source digest authority is absent"
            )
        if recomputed_source_sha256 != source_projection_sha256:
            raise ProviderNativeChronicleError(
                "Genome V2 Chronicle G0 source projection digest differs"
            )
        generation_zero = [
            dict(row) for row in projected if row is not None
        ]
        if any(
            row["generation"] != 0 or row["tick"] != 0
            for row in generation_zero
        ):
            raise ProviderNativeChronicleError(
                "Genome V2 Chronicle G0 generation or tick differs"
            )

        states = [
            (cell_id, self._load(cell_id))
            for cell_id in self.cell_ids
        ]
        actions: list[
            tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]
        ] = []
        for (cell_id, state), row in zip(
            states, generation_zero, strict=True
        ):
            segments = list(state.get("segments", []))
            summaries = _verified_genome_v2_summaries(
                state.get("genome_v2_summaries"),
                cell_id=cell_id,
                segment_bindings=segments,
                generation_zero_binding=state.get(
                    "genome_v2_generation_zero_binding"
                ),
            )
            binding = {
                "schema_version": GENOME_V2_G0_BINDING_SCHEMA_VERSION,
                "generation": 0,
                "tick": 0,
                "source_projection_sha256": (
                    source_projection_sha256[cell_id]
                ),
                "sidecar_payload_sha256": row[
                    "sidecar_payload_sha256"
                ],
                "summary_sha256": row["summary_sha256"],
                "projection_sha256": content_sha256(row),
            }
            if summaries and summaries[0]["generation"] == 0:
                if (
                    summaries[0] != row
                    or state.get("genome_v2_generation_zero_binding")
                    != binding
                ):
                    raise ProviderNativeChronicleError(
                        "replayed Chronicle Genome V2 G0 differs from durable authority"
                    )
                continue
            if segments or summaries:
                raise ProviderNativeChronicleError(
                    "Genome V2 Chronicle G0 arrived after segment authority"
                )
            if state.get("finalized") is True:
                raise ProviderNativeChronicleError(
                    "cannot append G0 to a finalized provider-native Chronicle"
                )
            actions.append((cell_id, state, row, binding))
        for cell_id, state, row, binding in actions:
            state["genome_v2_summaries"] = [row]
            state["genome_v2_generation_zero_binding"] = binding
            self._persist(cell_id, state, finalized=False)

    def observe_segment(
        self,
        *,
        segment_ordinal: int,
        start_generation: int,
        end_generation: int,
        cell_ids: Sequence[str],
        segment_artifacts: Sequence[Mapping[str, Any]],
        checkpoint_projections: Sequence[Mapping[str, Any]] | None = None,
        checkpoint_projection_cursor: Any | None = None,
    ) -> None:
        _integer(segment_ordinal, field="segment_ordinal", minimum=1)
        _integer(start_generation, field="start_generation")
        _integer(end_generation, field="end_generation", minimum=start_generation + 1)
        if (checkpoint_projections is None) == (
            checkpoint_projection_cursor is None
        ):
            raise ProviderNativeChronicleError(
                "segment requires exactly one projection input"
            )
        projection_input = (
            _CursorProjectionIterable(
                checkpoint_projection_cursor, self.cell_ids
            )
            if checkpoint_projection_cursor is not None
            else checkpoint_projections
        )
        if (
            tuple(cell_ids) != self.cell_ids
            or len(segment_artifacts) != len(self.cell_ids)
            or len(projection_input) != len(self.cell_ids)
        ):
            raise ProviderNativeChronicleError(
                "segment artifacts or checkpoints differ from physical cell order"
            )
        for cell_id, artifact, checkpoint_projection in zip(
            self.cell_ids,
            segment_artifacts,
            projection_input,
            strict=True,
        ):
            if artifact.get("cell_id") != cell_id:
                raise ProviderNativeChronicleError("segment artifact cell order differs")
            segment_index = _integer(
                artifact.get("segment_index"), field="segment_index"
            )
            source_hash = artifact.get("segment_output_sha256")
            if not isinstance(source_hash, str) or len(source_hash) != 64:
                raise ProviderNativeChronicleError("source segment hash is missing")
            cell = _object(artifact.get("cell"), field="segment cell")
            cell_hash = self._cell_segment_sha256(artifact, cell)
            previous_chain = _lower_digest(
                cell.get("previous_chain_sha256"),
                field="previous_chain_sha256",
            )
            chain_after = _lower_digest(
                cell.get("chain_sha256_after_segment"),
                field="chain_sha256_after_segment",
            )
            expected_chain_after = _digest(
                {
                    "schema_version": NATIVE_CONTINUITY_SCHEMA_VERSION,
                    "segment_index": segment_index,
                    "previous_chain_sha256": previous_chain,
                    "segment_sha256": cell_hash,
                }
            )
            if chain_after != expected_chain_after:
                raise ProviderNativeChronicleError(
                    "native segment chain hash does not verify"
                )
            state = self._load(cell_id)
            committed = list(state.get("segments", []))
            summaries = _verified_genome_v2_summaries(
                state.get("genome_v2_summaries"),
                cell_id=cell_id,
                segment_bindings=committed,
                generation_zero_binding=state.get(
                    "genome_v2_generation_zero_binding"
                ),
            )
            try:
                genome_v2_summary = (
                    project_genome_upgrade_v2_checkpoint_summary_v1(
                        checkpoint_projection,
                        expected_cell_id=cell_id,
                    )
                )
            except GenomeUpgradeV2Error as exc:
                raise ProviderNativeChronicleError(
                    f"Genome V2 Chronicle projection is invalid: {exc}"
                ) from exc
            existing = next(
                (
                    row
                    for row in committed
                    if isinstance(row, Mapping)
                    and row.get("segment_index") == segment_index
                ),
                None,
            )
            binding = {
                "segment_index": segment_index,
                "start_generation": start_generation,
                "end_generation": end_generation,
                "source_segment_sha256": source_hash,
                "cell_segment_sha256": cell_hash,
                "previous_chain_sha256": previous_chain,
                "chain_sha256_after_segment": chain_after,
            }
            if genome_v2_summary is not None:
                if (
                    genome_v2_summary["generation"] != end_generation
                    or genome_v2_summary["tick"] != 0
                ):
                    raise ProviderNativeChronicleError(
                        "Genome V2 Chronicle segment milestone differs"
                    )
                if committed and any(
                    "genome_v2_summary_sha256" not in row
                    for row in committed
                ):
                    raise ProviderNativeChronicleError(
                        "Genome V2 Chronicle summary presence changed across segments"
                    )
                binding["genome_v2_summary_sha256"] = genome_v2_summary[
                    "summary_sha256"
                ]
                binding["genome_v2_projection_sha256"] = content_sha256(
                    genome_v2_summary
                )
                binding["genome_v2_generation"] = genome_v2_summary[
                    "generation"
                ]
                binding["genome_v2_tick"] = genome_v2_summary["tick"]
                binding["genome_v2_sidecar_payload_sha256"] = (
                    genome_v2_summary["sidecar_payload_sha256"]
                )
            elif summaries or any(
                "genome_v2_summary_sha256" in row for row in committed
            ):
                raise ProviderNativeChronicleError(
                    "Genome V2 Chronicle segment summary is absent"
                )
            if existing is not None:
                if dict(existing) != binding:
                    raise ProviderNativeChronicleError(
                        "replayed Chronicle segment differs from durable authority"
                    )
                continue
            if state.get("finalized") is True:
                raise ProviderNativeChronicleError(
                    "cannot append to a finalized provider-native Chronicle"
                )
            if segment_index != state.get("next_segment_index"):
                raise ProviderNativeChronicleError(
                    "provider-native Chronicle segment gap or reordering"
                )
            expected_previous = (
                committed[-1].get("chain_sha256_after_segment")
                if committed
                else "0" * 64
            )
            if previous_chain != expected_previous:
                raise ProviderNativeChronicleError(
                    "provider-native Chronicle prior native authority differs"
                )
            attempts = cell.get("attempts")
            if not isinstance(attempts, list):
                raise ProviderNativeChronicleError("segment attempts are malformed")
            capacity = _object(
                artifact.get("capacity_semantics"), field="capacity semantics"
            )
            storage_mode = capacity.get("storage_mode")
            if storage_mode not in {"exact_full", "compact_complete"}:
                raise ProviderNativeChronicleError("storage mode is unsupported")
            events = list(state.get("events", []))
            cursor = _integer(state.get("event_cursor"), field="event_cursor")
            for attempt in attempts:
                row = _object(attempt, field="reproduction attempt")
                generation = _integer(row.get("generation"), field="generation")
                tick = _integer(row.get("tick"), field="tick")
                path_id = _integer(row.get("path"), field="path")
                measurements = {
                    "provider_native_event_cursor": cursor,
                    "source_segment_sha256": source_hash,
                    "cell_segment_sha256": cell_hash,
                    "segment_index": segment_index,
                    "storage_mode": storage_mode,
                    "reproduction_path_id": path_id,
                    "attempt_index": row.get("attempt_index"),
                    "eligible_parent_counts": row.get("eligible_parent_counts"),
                    "selected_parent_ids": row.get("selected_parent_ids"),
                    "success": row.get("success"),
                    "failure_code": row.get("failure_code"),
                    "commit_ordinal": row.get("commit_ordinal"),
                    "authoritative_commit": row.get("authoritative_commit"),
                    "eligible_parent_snapshot_claims_present": (
                        storage_mode == "exact_full"
                    ),
                }
                if storage_mode == "exact_full":
                    if not isinstance(row.get("eligible_snapshot_refs"), list):
                        raise ProviderNativeChronicleError(
                            "exact_full attempt lacks eligible snapshot references"
                        )
                    measurements["eligible_snapshot_refs"] = row[
                        "eligible_snapshot_refs"
                    ]
                elif "eligible_snapshot_refs" in row:
                    raise ProviderNativeChronicleError(
                        "compact attempt makes a fabricated snapshot claim"
                    )
                event = make_event(
                        run_id=cell_id,
                        campaign_id=self.campaign_id,
                        generation=generation,
                        step=tick,
                        phase_order=720,
                        intra_step_sequence=cursor,
                        event_type="provider_native_reproduction_attempt",
                        event_category="inheritance",
                        subject_type="reproduction_attempt",
                        subject_id=f"{segment_index}:{row.get('attempt_index')}",
                        confidence_class="measured",
                        causality_class="direct_mechanism",
                        source_component="engine.campaigns.segment_artifact_observer",
                        source_record_refs=(
                            f"segment:{segment_index}:sha256:{source_hash}",
                            f"cell-segment:sha256:{cell_hash}",
                        ),
                        measurements=measurements,
                        factual_summary_fields={
                            "reproduction_path_id": path_id,
                            "success": row.get("success"),
                        },
                        tags=("provider_native", "reproduction", storage_mode),
                    )
                if self._authoritative_identity_bindings is not None:
                    provider_projection = authoritative_provider_projection_v1(
                        attempt=row,
                        source_record_sha256=cell_hash,
                        identity_bindings=self._authoritative_identity_bindings,
                    )
                    event["authoritative_state_schema_v1"] = {
                        "provider_projection": provider_projection,
                        "chronicle_event": authoritative_chronicle_event_v1(
                            run_id=cell_id,
                            generation=generation,
                            step=tick,
                            intra_step_sequence=cursor,
                            subject_id=(
                                f"{segment_index}:{row.get('attempt_index')}"
                            ),
                            provider_projection=provider_projection,
                            identity_bindings=(
                                self._authoritative_identity_bindings
                            ),
                        ),
                    }
                events.append(event)
                cursor += 1
            cumulative = _object(
                cell.get("cumulative_counts_after_segment"),
                field="cumulative counts",
            )
            attempt_cursor = _integer(
                cumulative.get("attempts"), field="attempt cursor"
            )
            if attempt_cursor < state.get("attempt_cursor", 0):
                raise ProviderNativeChronicleError("attempt cursor regressed")
            state.update(
                {
                    "next_segment_index": segment_index + 1,
                    "event_cursor": cursor,
                    "attempt_cursor": attempt_cursor,
                    "segments": [*committed, binding],
                    "events": sorted(events, key=event_order_key),
                }
            )
            if genome_v2_summary is not None:
                summaries.append(genome_v2_summary)
                state["genome_v2_summaries"] = summaries
            self._persist(cell_id, state, finalized=False)

    def _persist(
        self, cell_id: str, state: Mapping[str, Any], *, finalized: bool
    ) -> None:
        root = self._root(cell_id)
        events = list(state.get("events", []))
        segments = list(state.get("segments", []))
        write_jsonl(root / ARTIFACT_NAMES["events"], events)
        write_jsonl(root / ARTIFACT_NAMES["entries"], ())
        write_json(root / ARTIFACT_NAMES["episodes"], [])
        write_jsonl(root / ARTIFACT_NAMES["contexts"], ())
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "run_id": cell_id,
            "campaign_id": self.campaign_id,
            "profile": self.profile,
            "source_completeness": SOURCE_COMPLETENESS,
            "finalized": finalized,
            "write_failures": 0,
            "last_generation": (
                max((int(row["end_generation"]) for row in segments), default=0)
            ),
            "event_count": len(events),
            "segment_count": len(segments),
            "event_cursor": state.get("event_cursor", 0),
            "attempt_cursor": state.get("attempt_cursor", 0),
            "genome_v2_generation_summaries": list(
                state.get("genome_v2_summaries", [])
            ),
        }
        write_json(root / ARTIFACT_NAMES["summary"], summary)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "recorder_version": "provider_native_segment_consumer_v1",
            "run_id": cell_id,
            "campaign_id": self.campaign_id,
            "profile": self.profile,
            "source_completeness": SOURCE_COMPLETENESS,
            "event_count": len(events),
            "segment_count": len(segments),
            "finalized": finalized,
            "artifact_names": dict(sorted(ARTIFACT_NAMES.items())),
        }
        write_json(root / ARTIFACT_NAMES["manifest"], manifest)
        durable_state = dict(state)
        durable_state["finalized"] = finalized
        write_json(self._state_path(cell_id), durable_state)

    def finalize(self) -> ProviderNativeChronicleReceiptV1:
        receipts: dict[str, Mapping[str, Any]] = {}
        for cell_id in self.cell_ids:
            state = self._load(cell_id)
            if not state.get("segments"):
                raise ProviderNativeChronicleError(
                    f"cell {cell_id!r} has no provider-native segments"
                )
            self._persist(cell_id, state, finalized=True)
            root = self._root(cell_id).resolve()
            manifest_path = root / ARTIFACT_NAMES["manifest"]
            summary_path = root / ARTIFACT_NAMES["summary"]
            segments = list(state["segments"])
            receipts[cell_id] = {
                "cell_id": cell_id,
                "chronicle_root": str(root),
                "segment_count": len(segments),
                "segment_bindings": segments,
                "event_cursor": state["event_cursor"],
                "attempt_cursor": state["attempt_cursor"],
                "chronicle_manifest_sha256": _sha256(manifest_path),
                "chronicle_summary_sha256": _sha256(summary_path),
                "durable_artifact_sha256": {
                    filename: _sha256(root / filename)
                    for filename in DURABLE_AUTHORITY_FILENAMES
                },
                "finalized": True,
            }
            if isinstance(state.get("continuation_binding"), Mapping):
                receipts[cell_id]["continuation_binding"] = canonical_payload(
                    state["continuation_binding"]
                )
        return ProviderNativeChronicleReceiptV1(
            campaign_id=self.campaign_id,
            batch_id=self.batch_id,
            profile=self.profile,
            finalized=True,
            cell_receipts=receipts,
        )

    def checkpoint_receipt(
        self, *, snapshot_batch_root: Path
    ) -> ProviderNativeChronicleReceiptV1:
        """Finalize an immutable prefix copy without closing the live consumer."""

        snapshot = ProviderNativeChronicleConsumerV1(
            batch_root=snapshot_batch_root,
            campaign_id=self.campaign_id,
            batch_id=self.batch_id,
            cell_ids=self.cell_ids,
            profile=self.profile,
            authoritative_identity_bindings=(
                self._authoritative_identity_bindings
            ),
        )
        snapshot._continuation_binding = self._continuation_binding
        if any(snapshot._root(cell_id).exists() for cell_id in self.cell_ids):
            raise ProviderNativeChronicleError(
                "Chronicle checkpoint snapshot target already exists"
            )
        states = [
            (cell_id, canonical_payload(self._load(cell_id)))
            for cell_id in self.cell_ids
        ]
        if any(not state.get("segments") for _cell_id, state in states):
            raise ProviderNativeChronicleError(
                "Chronicle checkpoint snapshot has no observed prefix"
            )
        for cell_id, state in states:
            snapshot._persist(cell_id, state, finalized=False)
        return snapshot.finalize()


__all__ = [
    "CONTINUATION_REBIND_SCHEMA_VERSION",
    "DURABLE_AUTHORITY_FILENAMES",
    "ProviderNativeChronicleContinuationSeedV1",
    "ProviderNativeChronicleConsumerV1",
    "ProviderNativeChronicleError",
    "ProviderNativeChronicleReceiptV1",
    "RECEIPT_SCHEMA_VERSION",
    "SOURCE_COMPLETENESS",
    "authoritative_chronicle_event_v1",
    "authoritative_provider_projection_v1",
    "bind_provider_native_chronicle_continuation_v1",
]
