"""Pure offline MUTSUP direct-retention reduction.

This module deliberately sits on the route-neutral science evidence boundary.
It consumes only sealed :class:`ScienceEvidencePartitionV1` values and never
imports, calls, or mutates a simulator/provider.  The reducer answers the
small question that the current V1 observation ABI can actually represent:

``applied mutation -> attributed child/genome -> later carrier -> later birth``

The result is descriptive.  In particular, a paired OFF/ON difference is not
treated as a causal effect, a fitness estimate, or a mutation-rate estimate.
Ownership (study, arm, replicate) is required explicitly in the reducer
configuration or in the study aggregator; it is never inferred from a cell
name, partition, slot, or physical packing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_VERSION,
    SchemaVersion,
    ScienceEvidencePartitionV1,
    ScienceRecordKindV1,
    ScienceReducerDescriptorV1,
    canonical_json_bytes,
    canonical_sha256,
    validate_science_native_control_receipt_v1,
)

from .science_reducer_protocol_v1 import (
    ScienceReducerCheckpointV1,
    ScienceReducerError,
    ScienceReducerProgressV1,
    execute_science_reducer_v1,
)


MUTSUP_DIRECT_RETENTION_REDUCER_ID = "science.mutsup.direct_retention.v1"
MUTSUP_DIRECT_RETENTION_CONTRACT_VERSION = SchemaVersion(
    "science_mutsup_direct_retention_v1"
)
MUTSUP_DIRECT_RETENTION_OUTPUT_CONTRACT_ID = (
    "eiel_science_mutsup_direct_retention_result_v1"
)
MUTSUP_DIRECT_RETENTION_STATE_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_mutsup_direct_retention_state_v1"
)
MUTSUP_DIRECT_RETENTION_CONFIGURATION_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_mutsup_direct_retention_config_v1"
)

MUTSUP_MUTATION_OBSERVATION_SEMANTICS_APPLIED_OUTCOMES_ONLY_COMPLETE = (
    "applied_outcomes_only_complete"
)

# These are the minimum ABI capabilities named by the existing MUTSUP
# observation-resolution artifacts.  Lineage/death evidence is used when it
# is present, but is not silently substituted for the exact parent/genome
# identity needed by the direct-retention question.
MUTSUP_DIRECT_RETENTION_REQUIRED_CAPABILITY_IDS = (
    "birth_parentage",
    "mutation_provenance",
)
MUTSUP_DIRECT_RETENTION_REQUIRED_RECORD_KINDS = (
    ScienceRecordKindV1.CONTROL_BINDING,
    ScienceRecordKindV1.BIRTH_COMMITTED,
    ScienceRecordKindV1.MUTATION_OUTCOME,
    ScienceRecordKindV1.WORLD_COMPLETION,
)
MUTSUP_DIRECT_RETENTION_OPTIONAL_RECORD_KINDS = (
    ScienceRecordKindV1.AGENT_STATE_SNAPSHOT,
    ScienceRecordKindV1.BIRTH_ATTEMPT,
    ScienceRecordKindV1.DEATH,
    ScienceRecordKindV1.REPRODUCTION_OPPORTUNITY,
    ScienceRecordKindV1.CHECKPOINT_MARKER,
)
MUTSUP_DIRECT_RETENTION_CLAIM_BOUNDARY = (
    "Descriptive direct mutation-origin attribution and reproductive-carrier "
    "observations for explicitly paired worlds; no causal effect, fitness, "
    "mutation-opportunity rate, genome-distance, or universal retention claim."
)

_KIND_ORDINAL = {kind: ordinal for ordinal, kind in enumerate(ScienceRecordKindV1)}
_REQUIRED_KIND_NAMES = tuple(
    kind.value for kind in MUTSUP_DIRECT_RETENTION_REQUIRED_RECORD_KINDS
)
_OPTIONAL_KIND_NAMES = tuple(
    kind.value for kind in MUTSUP_DIRECT_RETENTION_OPTIONAL_RECORD_KINDS
)
_APPLIED_RESULTS = frozenset(
    {
        "ACCEPTED",
        "APPLIED",
        "MUTATION_APPLIED",
        "MUTATION_APPLIED_WITH_SATURATED_NOOPS",
    }
)
_KNOWN_RESULTS = frozenset(
    {
        "DISABLED",
        "NO_OP",
        "REJECTED",
        "ACCEPTED",
        "APPLIED",
        "MUTATION_APPLIED",
        "MUTATION_APPLIED_WITH_SATURATED_NOOPS",
        "MUTATION_SATURATED_NOOP",
        "CATASTROPHIC",
        "INVALID",
        "FAILED",
    }
)
_OFF_ARM_DEFAULTS = frozenset(
    {"off", "mutation_off", "mutations_off", "m0", "control_off", "no_mutation"}
)
_ON_ARM_DEFAULTS = frozenset(
    {"on", "mutation_on", "mutations_on", "m1", "control_on", "baseline"}
)
_INTEGRITY_BAD_WORDS = frozenset(
    {"DROP", "DROPPED", "OVERFLOW", "OVERFLOWED", "TRUNCATED", "LOSS"}
)


def _fail(message: str) -> None:
    raise ScienceReducerError(message)


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    result = dict(value)
    if any(type(key) is not str for key in result):
        _fail(f"{field} keys must be strings")
    return result


def _nonempty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        _fail(f"{field} must be a non-empty exact string")
    return value


def _identity_key(value: object, *, field: str) -> str:
    """Return a typed, deterministic key for opaque agent/genome identities."""

    if value is None or isinstance(value, bool):
        _fail(f"{field} is missing or has an invalid identity")
    if isinstance(value, str):
        if not value or value.strip() != value:
            _fail(f"{field} is an invalid identity")
        return "str:" + value
    if type(value) is int:
        return "int:" + str(value)
    if isinstance(value, Mapping):
        try:
            return "json:" + canonical_json_bytes(dict(value)).decode("ascii")
        except (TypeError, ValueError) as error:
            raise ScienceReducerError(
                f"{field} is not canonical identity data"
            ) from error
    _fail(f"{field} is an invalid identity")
    raise AssertionError("unreachable")


def _digest_hex(value: object, *, field: str) -> str:
    if isinstance(value, ContentDigest):
        return value.hex
    if isinstance(value, Mapping):
        data = _mapping(value, field=field)
        if set(data) != {"algorithm", "hex"}:
            _fail(f"{field} is not a SHA-256 digest")
        try:
            return ContentDigest(data["algorithm"], data["hex"]).hex
        except (TypeError, ValueError) as error:
            raise ScienceReducerError(f"{field} is not a SHA-256 digest") from error
    if isinstance(value, str) and value.startswith("sha256:"):
        try:
            return ContentDigest.parse(value).hex
        except (TypeError, ValueError) as error:
            raise ScienceReducerError(f"{field} is not a SHA-256 digest") from error
    _fail(f"{field} is not a SHA-256 digest")
    raise AssertionError("unreachable")


def _payload(row: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    raw = row.get("payload")
    if isinstance(raw, Mapping) and set(raw) == {"schema_version", "payload"}:
        raw = raw["payload"]
    return _mapping(raw, field=field)


def _native_control_provenance(
    payload: Mapping[str, Any], *, field: str
) -> dict[str, Any] | None:
    """Validate the applied native mask/vector receipt carried by a control row."""

    raw = payload.get("applied_native_control")
    if raw is None:
        return None
    try:
        return validate_science_native_control_receipt_v1(
            raw, field=f"{field}.applied_native_control"
        )
    except Exception as exc:
        _fail(f"{field}.applied_native_control is invalid: {exc}")
    raise AssertionError("unreachable")


def _record_kind(row: Mapping[str, Any]) -> str:
    return _nonempty_text(row.get("record_kind"), field="row.record_kind")


def _event_key(row: Mapping[str, Any]) -> str:
    raw = row.get("deterministic_event_id")
    return _digest_hex(raw, field="row.deterministic_event_id")


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    tick = row.get("absolute_tick")
    sequence = row.get("world_local_sequence")
    if type(tick) is not int or tick < 0:
        _fail("row absolute_tick is invalid")
    if type(sequence) is not int or sequence < 0:
        _fail("row world_local_sequence is invalid")
    return tick, sequence, _event_key(row)


def _normalise_result(value: object, *, field: str) -> str:
    result = _nonempty_text(value, field=field).upper().replace("-", "_")
    if result not in _KNOWN_RESULTS:
        _fail(f"{field} has unsupported mutation result class {result!r}")
    return result


def _normalise_replicate(value: object, *, field: str) -> str | int:
    if type(value) is int or (
        isinstance(value, str) and value and value.strip() == value
    ):
        return value
    _fail(f"{field} must be an explicit string or integer replicate")
    raise AssertionError("unreachable")


@dataclass(frozen=True, slots=True)
class _WorldOwner:
    world_uid_hex: str
    study_id: str
    arm_id: str
    replicate: str | int
    pair_id: str | None = None
    cell_id: str | None = None

    @property
    def pair_key(self) -> tuple[str, str | int, str | None]:
        return self.study_id, self.replicate, self.pair_id


def _owner_world_key(
    raw: Mapping[str, Any], *, field: str, alias: str | None = None
) -> str:
    for name in ("world_uid", "world_id", "world_hex", "world_key"):
        if name in raw:
            value = raw[name]
            if name == "world_uid":
                return _digest_hex(value, field=f"{field}.{name}")
            return _nonempty_text(value, field=f"{field}.{name}")
    if alias is not None:
        return _nonempty_text(alias, field=f"{field}.world_key")
    _fail(f"{field} requires an explicit world_uid")
    raise AssertionError("unreachable")


def _parse_owners(
    config: Mapping[str, Any], world_rows: Mapping[str, Mapping[str, Any]]
) -> dict[str, _WorldOwner]:
    """Parse explicit world ownership in several canonical-compatible forms."""

    raw_source: object = None
    for name in (
        "world_ownership",
        "world_metadata",
        "worlds",
        "ownership",
        "science_ownership",
        "science",
    ):
        if name in config:
            raw_source = config[name]
            break
    owners: list[tuple[str | None, Mapping[str, Any]]] = []
    if isinstance(raw_source, list):
        for index, item in enumerate(raw_source):
            owner = _mapping(item, field=f"configuration.{name}[{index}]")
            nested = owner.get("science_ownership", owner.get("science"))
            if isinstance(nested, Mapping):
                owner = {**dict(nested), **owner}
            owners.append((None, owner))
    elif isinstance(raw_source, Mapping):
        source = dict(raw_source)
        # A single owner object is accepted for a one-world reduction.  A
        # mapping keyed by world UID/cell ID is the normal multi-world form.
        if {"study_id", "arm_id", "replicate"}.issubset(source):
            owners.append((None, source))
        else:
            for alias, item in source.items():
                owner = _mapping(item, field=f"configuration.{name}[{alias}]")
                nested = owner.get("science_ownership", owner.get("science"))
                if isinstance(nested, Mapping):
                    owner = {**dict(nested), **owner}
                owners.append((alias, owner))
    elif raw_source is not None:
        _fail("configuration world ownership must be an array or object")

    if not owners and {"study_id", "arm_id", "replicate"}.issubset(config):
        if len(world_rows) != 1:
            _fail("multi-world reduction requires explicit world ownership")
        owners.append((next(iter(world_rows)), config))
    if not owners:
        _fail("explicit world/study/arm/replicate ownership is required")

    result: dict[str, _WorldOwner] = {}
    for index, (alias, raw) in enumerate(owners):
        field = f"configuration.world_ownership[{index}]"
        implicit_world = (
            next(iter(world_rows)) if alias is None and len(world_rows) == 1 else None
        )
        world_key = _owner_world_key(
            raw,
            field=field,
            alias=alias if alias is not None else implicit_world,
        )
        # world_uid is the canonical key; a configured cell/world alias is
        # resolved below without allowing it to become an identity source.
        matching_world = world_key if world_key in world_rows else None
        if matching_world is None:
            for candidate, row in world_rows.items():
                if world_key == row.get("cell_id") or world_key == row.get("world_id"):
                    matching_world = candidate
                    break
        if matching_world is None:
            _fail(f"world ownership does not name an observed world: {world_key!r}")
        if matching_world in result:
            _fail(f"duplicate world ownership for {matching_world}")
        study_id = _nonempty_text(raw.get("study_id"), field=f"{field}.study_id")
        arm_id = _nonempty_text(raw.get("arm_id"), field=f"{field}.arm_id")
        replicate = _normalise_replicate(
            raw.get("replicate"), field=f"{field}.replicate"
        )
        pair_id = raw.get("pair_id")
        if pair_id is not None:
            pair_id = _nonempty_text(pair_id, field=f"{field}.pair_id")
        cell_id = raw.get("cell_id")
        if cell_id is not None:
            cell_id = _nonempty_text(cell_id, field=f"{field}.cell_id")
        result[matching_world] = _WorldOwner(
            world_uid_hex=matching_world,
            study_id=study_id,
            arm_id=arm_id,
            replicate=replicate,
            pair_id=pair_id,
            cell_id=cell_id,
        )
    missing = sorted(set(world_rows) - set(result))
    if missing:
        _fail(f"world ownership is missing observed worlds: {missing}")
    return result


def _pairing_arm_sets(
    config: Mapping[str, Any],
) -> tuple[frozenset[str], frozenset[str]]:
    pairing = config.get("pairing", config.get("paired_contrasts"))
    if pairing is None:
        return _OFF_ARM_DEFAULTS, _ON_ARM_DEFAULTS
    if isinstance(pairing, list):
        # A list is the compact explicit world-pair form.  Arm roles are
        # still checked from each sealed world's ownership below.
        return _OFF_ARM_DEFAULTS, _ON_ARM_DEFAULTS
    data = _mapping(pairing, field="configuration.pairing")
    off = data.get("off_arm_ids", data.get("off_arms", data.get("off")))
    on = data.get("on_arm_ids", data.get("on_arms", data.get("on")))
    if off is None and on is None and any(key in data for key in ("pairs", "pair_map")):
        return _OFF_ARM_DEFAULTS, _ON_ARM_DEFAULTS
    if isinstance(off, str):
        off = [off]
    if isinstance(on, str):
        on = [on]
    if not isinstance(off, list) or not isinstance(on, list) or not off or not on:
        _fail("pairing requires non-empty off_arm_ids and on_arm_ids")
    off_set = frozenset(
        _nonempty_text(item, field="configuration.pairing.off") for item in off
    )
    on_set = frozenset(
        _nonempty_text(item, field="configuration.pairing.on") for item in on
    )
    if off_set & on_set:
        _fail("pairing OFF and ON arm sets overlap")
    return off_set, on_set


def _arm_role(arm_id: str, off: frozenset[str], on: frozenset[str]) -> str | None:
    if arm_id in off or arm_id.lower() in {item.lower() for item in off}:
        return "OFF"
    if arm_id in on or arm_id.lower() in {item.lower() for item in on}:
        return "ON"
    folded = arm_id.lower().replace("-", "_").replace(" ", "_")
    if folded in _OFF_ARM_DEFAULTS:
        return "OFF"
    if folded in _ON_ARM_DEFAULTS:
        return "ON"
    return None


def _world_reference(
    value: object,
    worlds: Mapping[str, Mapping[str, Any]],
    *,
    field: str,
) -> str:
    """Resolve an explicit pair endpoint without treating aliases as identity."""

    if isinstance(value, Mapping):
        if {"algorithm", "hex"}.issubset(value):
            candidate = _digest_hex(value, field=field)
            if candidate in worlds:
                return candidate
        if "world_uid" in value:
            return _world_reference(
                value["world_uid"], worlds, field=f"{field}.world_uid"
            )
        if "world_id" in value:
            return _world_reference(
                value["world_id"], worlds, field=f"{field}.world_id"
            )
        _fail(f"{field} does not identify an observed world")
    if isinstance(value, str) and value.strip() == value and value:
        if value in worlds:
            return value
        for world_uid, world in worlds.items():
            if value == world.get("cell_id") or value == world.get("world_id"):
                return world_uid
    _fail(f"{field} does not identify an observed world")
    raise AssertionError("unreachable")


def _explicit_pair_specs(config: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    """Return explicit world pair rows when the configuration supplies them."""

    pairing = config.get("pairing", config.get("paired_contrasts"))
    source: object = None
    if isinstance(pairing, list):
        source = pairing
    elif isinstance(pairing, Mapping):
        data = dict(pairing)
        if "pairs" in data:
            source = data["pairs"]
        elif "pair_map" in data:
            source = data["pair_map"]
    elif pairing is not None:
        _fail("configuration pairing must be an object or explicit pair list")
    if source is None and "pairs" in config:
        source = config["pairs"]
    if source is None:
        return None
    if isinstance(source, Mapping):
        source = [
            {
                **_mapping(item, field=f"configuration.pairing.pair_map[{pair_id}]"),
                "pair_id": _nonempty_text(
                    pair_id, field="configuration.pairing.pair_map key"
                ),
            }
            for pair_id, item in source.items()
        ]
    if not isinstance(source, list) or not source:
        _fail("explicit pairing requires a non-empty pairs list")
    return [
        _mapping(item, field=f"configuration.pairing.pairs[{index}]")
        for index, item in enumerate(source)
    ]


def _reject_config_generation_keys(
    value: object, *, path: str = "configuration"
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(f"{path} contains a non-string key")
            if "generation" in key.lower():
                _fail(f"generation-shaped configuration key is forbidden: {path}.{key}")
            _reject_config_generation_keys(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_config_generation_keys(item, path=f"{path}[{index}]")


def _integrity_bad(value: object, *, path: str = "configuration") -> str | None:
    """Find explicit dropped/overflow markers without scanning free text."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_upper = str(key).upper()
            if key_upper in {
                "DROPPED",
                "DROPPED_RECORDS",
                "DROPPED_EVENT_COUNT",
                "DROPPED_COUNT",
                "DROPPED_EVENTS",
                "OVERFLOW",
                "OVERFLOWED",
                "OVERFLOW_COUNT",
                "OVERFLOW_RECORD_COUNT",
                "OVERFLOW_EVENT_COUNT",
                "TRUNCATED_RECORD_COUNT",
                "LOSS_COUNT",
                "EVIDENCE_DROPPED",
                "EVIDENCE_OVERFLOW",
                "EVIDENCE_LOSS",
            }:
                if (type(item) is bool and item) or (type(item) is int and item > 0):
                    return f"{path}.{key}={item!r}"
            if key_upper in {"STATUS", "EVIDENCE_STATUS", "INTEGRITY_STATUS"}:
                text = str(item).upper()
                if any(word in text for word in _INTEGRITY_BAD_WORDS):
                    return f"{path}.{key}={item!r}"
            found = _integrity_bad(item, path=f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _integrity_bad(item, path=f"{path}[{index}]")
            if found:
                return found
    return None


def _stream_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        encoded = canonical_json_bytes(dict(row))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_world_stream(
    world_uid_hex: str,
    world: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_raw = world.get("rows")
    if not isinstance(rows_raw, list):
        _fail(f"world {world_uid_hex} accumulator rows are malformed")
    rows = [_mapping(item, field=f"world[{world_uid_hex}].rows[]") for item in rows_raw]
    for index, row in enumerate(rows):
        bad = _integrity_bad(row, path=f"world[{world_uid_hex}].rows[{index}]")
        if bad:
            _fail(f"dropped/overflow evidence is not reducible: {bad}")
    kinds = set(world.get("partition_record_kinds", []))
    if any(not isinstance(kind, str) for kind in kinds):
        _fail(f"world {world_uid_hex} partition record-kind registry is malformed")
    try:
        for kind in kinds:
            ScienceRecordKindV1(kind)
    except ValueError as error:
        raise ScienceReducerError(
            f"world {world_uid_hex} partition record-kind registry is unknown"
        ) from error
    row_kinds = {_record_kind(row) for row in rows}
    if not row_kinds.issubset(kinds):
        _fail(f"world {world_uid_hex} has a row outside its record-kind registry")
    missing = sorted(set(_REQUIRED_KIND_NAMES) - kinds)
    if missing:
        _fail(f"world {world_uid_hex} missing required record kind(s): {missing}")
    rows.sort(key=_row_sort_key)
    sequences = [row["world_local_sequence"] for row in rows]
    if sequences != list(range(len(rows))):
        _fail(f"world {world_uid_hex} sequence gap or duplicate")
    event_ids = [_event_key(row) for row in rows]
    if len(event_ids) != len(set(event_ids)):
        _fail(f"world {world_uid_hex} duplicate deterministic event ID")
    for row in rows:
        if _digest_hex(row.get("world_uid"), field="row.world_uid") != world_uid_hex:
            _fail(f"world {world_uid_hex} row world identity differs")
    completion_rows = [
        row
        for row in rows
        if _record_kind(row) == ScienceRecordKindV1.WORLD_COMPLETION.value
    ]
    if len(completion_rows) != 1:
        _fail(f"world {world_uid_hex} requires exactly one world_completion record")
    completion = _payload(completion_rows[0], field="world_completion.payload")
    if completion_rows[0].get("absolute_tick") != completion.get(
        "terminal_absolute_tick"
    ):
        _fail(f"world {world_uid_hex} completion terminal tick does not join row tick")
    if completion.get("completion_class") not in {
        "EMPTY_INITIAL",
        "EXTINCT",
        "NO_BIRTHS",
        "TERMINAL_HORIZON",
    }:
        _fail(f"world {world_uid_hex} completion class is unsupported")
    population_count = completion.get("terminal_population_count")
    if type(population_count) is not int or population_count < 0:
        _fail(f"world {world_uid_hex} terminal population count is invalid")
    # The producer seals the terminal marker *after* it computes the stream
    # prefix.  Consequently ``terminal_record_count`` and
    # ``canonical_record_stream_sha256`` describe the rows preceding the
    # marker, while the marker itself is part of the published stream.  A
    # small number of offline fixtures historically used the opposite
    # convention; accepting that exact, self-consistent form keeps the
    # reducer ABI-compatible without weakening dropped-row detection.
    terminal_count = completion.get("terminal_record_count")
    if type(terminal_count) is not int or terminal_count < 0:
        _fail(
            f"world {world_uid_hex} dropped evidence: terminal record count is invalid"
        )
    try:
        stream_digest = _digest_hex(
            completion.get("canonical_record_stream_sha256"),
            field="world_completion.canonical_record_stream_sha256",
        )
    except ScienceReducerError:
        raise
    completion_index = rows.index(completion_rows[0])
    if completion_index != len(rows) - 1:
        _fail(f"world {world_uid_hex} has evidence after world completion")
    prefix_rows = rows[:completion_index]
    stream_matches = (
        terminal_count == len(prefix_rows)
        and stream_digest == _stream_sha256(prefix_rows)
    ) or (terminal_count == len(rows) and stream_digest == _stream_sha256(rows))
    if not stream_matches:
        _fail(f"world {world_uid_hex} dropped evidence: canonical stream/count differs")
    terminal_tick = completion.get("terminal_absolute_tick")
    next_tick = completion.get("next_absolute_tick")
    if (
        type(terminal_tick) is not int
        or terminal_tick < 0
        or type(next_tick) is not int
        or next_tick < terminal_tick
    ):
        _fail(f"world {world_uid_hex} completion absolute tick frontier is invalid")
    return rows, completion


@dataclass(frozen=True, slots=True)
class _Mutation:
    row: Mapping[str, Any]
    payload: Mapping[str, Any]
    mutation_attempt_id: str
    mutation_opportunity_id: str
    subject_agent_key: str
    origin_agent_key: str
    origin_absolute_tick: int
    event_tick: int
    pre_genome_key: str
    post_genome_key: str
    result_class: str
    attempted: bool

    @property
    def applied(self) -> bool:
        return self.attempted and self.result_class in _APPLIED_RESULTS


@dataclass(frozen=True, slots=True)
class _Birth:
    row: Mapping[str, Any]
    payload: Mapping[str, Any]
    birth_attempt_id: str
    opportunity_id: str
    child_agent_key: str
    child_pre_genome_key: str
    child_post_genome_key: str
    parent_pairs: tuple[tuple[str, str], ...]
    event_tick: int


def _mutation(row: Mapping[str, Any]) -> _Mutation:
    payload = _payload(row, field="mutation_outcome.payload")
    attempted = payload.get("attempted")
    if type(attempted) is not bool:
        _fail("mutation_outcome.attempted must be boolean")
    result = _normalise_result(
        payload.get("result_class"), field="mutation_outcome.result_class"
    )
    origin_tick = payload.get("origin_absolute_tick")
    event_tick = row.get("absolute_tick")
    if type(origin_tick) is not int or origin_tick < 0 or type(event_tick) is not int:
        _fail("mutation_outcome origin/event tick is invalid")
    if origin_tick > event_tick:
        _fail("mutation_outcome origin tick is after event tick")
    mutation_attempt_id = _identity_key(
        payload.get("mutation_attempt_id"), field="mutation_outcome.mutation_attempt_id"
    )
    opportunity_id = _identity_key(
        payload.get("mutation_opportunity_id"),
        field="mutation_outcome.mutation_opportunity_id",
    )
    subject = _identity_key(
        payload.get("subject_agent_id"), field="mutation_outcome.subject_agent_id"
    )
    origin = _identity_key(
        payload.get("origin_agent_id"), field="mutation_outcome.origin_agent_id"
    )
    pre = _identity_key(
        payload.get("pre_genome_state_id"), field="mutation_outcome.pre_genome_state_id"
    )
    post = _identity_key(
        payload.get("post_genome_state_id"),
        field="mutation_outcome.post_genome_state_id",
    )
    if (
        payload.get("mutation_class_id") == "native_mutation_count"
        or payload.get("reason_code") == "native_mutation_count"
    ):
        _fail(
            "native mutation outcome has aggregate-only provenance; exact "
            "mutation locus, child pre/post genome, and origin are required"
        )
    if (
        payload.get("mutation_class_id") == "native_genome_v2_point_mutation"
        and origin != subject
    ):
        _fail(
            "native exact mutation provenance must identify the mutated child as origin"
        )
    if result == "MUTATION_SATURATED_NOOP" and payload.get(
        "mutation_class_id"
    ) != "native_genome_v2_point_mutation":
        _fail("saturated mutation outcome lacks explicit native provenance")
    if (
        payload.get("mutation_class_id") == "native_genome_v2_point_mutation"
        and not attempted
    ):
        _fail("native exact mutation has attempted=false")
    if payload.get("mutation_class_id") == "native_genome_v2_point_mutation":
        for digest_field, identity in (
            ("pre_value_or_state_digest", payload.get("pre_genome_state_id")),
            ("post_value_or_state_digest", payload.get("post_genome_state_id")),
        ):
            observed = _digest_hex(
                payload.get(digest_field),
                field=f"mutation_outcome.{digest_field}",
            )
            expected = canonical_sha256({"genome_state_id": identity}).hex
            if observed != expected:
                _fail(
                    f"native exact mutation {digest_field} does not bind its genome identity"
                )
        coordinates = payload.get("affected_coordinates")
        if not isinstance(coordinates, Mapping):
            _fail("native exact mutation lacks affected allele coordinates")
        required_coordinate_fields = {
            "accepted_draws",
            "accepted_draw_details",
            "changed_alleles",
            "native_mutation_count",
            "saturated_noop_count",
        }
        if set(coordinates) != required_coordinate_fields:
            _fail("native exact mutation allele provenance fields are invalid")
        changed = coordinates.get("changed_alleles")
        details = coordinates.get("accepted_draw_details")
        accepted_draws = coordinates.get("accepted_draws")
        native_count = coordinates.get("native_mutation_count")
        saturated_count = coordinates.get("saturated_noop_count")
        if (
            not isinstance(changed, list)
            or not isinstance(details, list)
            or type(accepted_draws) is not int
            or accepted_draws <= 0
            or type(native_count) is not int
            or native_count <= 0
            or accepted_draws != native_count
            or type(saturated_count) is not int
            or saturated_count < 0
            or saturated_count > accepted_draws
            or len(details) != accepted_draws
        ):
            _fail("native exact mutation accepted-draw provenance is invalid")

        def validate_coordinate(
            allele: Mapping[str, Any], *, index: int, detail: bool
        ) -> tuple[int, int, int, int, int]:
            expected = (
                {
                    "genome_region",
                    "haplotype",
                    "linked_locus",
                    "pre_value",
                    "post_value",
                    "requested_delta",
                    "effective_delta",
                    "saturated_noop",
                }
                if detail
                else {
                    "genome_region",
                    "haplotype",
                    "linked_locus",
                    "pre_value",
                    "post_value",
                }
            )
            if set(allele) != expected:
                _fail(
                    f"native exact mutation {'draw' if detail else 'allele'} "
                    f"{index} fields are invalid"
                )
            region = allele["genome_region"]
            haplotype = allele["haplotype"]
            locus = allele["linked_locus"]
            pre_value = allele["pre_value"]
            post_value = allele["post_value"]
            if (
                type(region) is not int
                or region not in (0, 1)
                or type(haplotype) is not int
                or haplotype not in (0, 1)
                or type(locus) is not int
                or locus < 0
                or locus >= 128
                or (region == 0 and locus >= 112)
                or (region == 1 and locus < 112)
                or type(pre_value) is not int
                or type(post_value) is not int
                or (
                    not -128 <= pre_value <= 127
                    if region == 0
                    else not -32768 <= pre_value <= 32767
                )
                or (
                    not -127 <= post_value <= 127
                    if region == 0
                    else not -32767 <= post_value <= 32767
                )
            ):
                _fail(
                    f"native exact mutation {'draw' if detail else 'allele'} "
                    f"{index} values are invalid"
                )
            return region, haplotype, locus, pre_value, post_value

        detail_keys: list[tuple[int, int, int]] = []
        changed_keys: list[tuple[int, int, int]] = []
        nonzero_details: list[dict[str, int]] = []
        detail_saturated_count = 0
        for index, raw_detail in enumerate(details):
            if not isinstance(raw_detail, Mapping):
                _fail(f"native exact mutation draw {index} is invalid")
            region, haplotype, locus, pre_value, post_value = validate_coordinate(
                raw_detail, index=index, detail=True
            )
            requested_delta = raw_detail["requested_delta"]
            effective_delta = raw_detail["effective_delta"]
            saturated_noop = raw_detail["saturated_noop"]
            if (
                type(requested_delta) is not int
                or requested_delta not in (-1, 1)
                or type(effective_delta) is not int
                or effective_delta != post_value - pre_value
                or type(saturated_noop) is not bool
                or saturated_noop != (effective_delta == 0)
            ):
                _fail(f"native exact mutation draw {index} delta is invalid")
            if not saturated_noop and effective_delta != requested_delta:
                _fail(
                    f"native exact mutation draw {index} effective delta does not "
                    "match its requested delta"
                )
            if saturated_noop:
                native_boundary = (
                    (127 if requested_delta == 1 else -127)
                    if region == 0
                    else (32767 if requested_delta == 1 else -32767)
                )
                if pre_value != native_boundary or post_value != native_boundary:
                    _fail(
                        f"native exact mutation draw {index} saturated value is "
                        "not at the native boundary"
                    )
            key = (region, haplotype, locus)
            if key in detail_keys:
                _fail("native exact mutation has duplicate allele coordinates")
            detail_keys.append(key)
            if saturated_noop:
                detail_saturated_count += 1
            else:
                nonzero_details.append(
                    {
                        "genome_region": region,
                        "haplotype": haplotype,
                        "linked_locus": locus,
                        "pre_value": pre_value,
                        "post_value": post_value,
                    }
                )
        if detail_saturated_count != saturated_count:
            _fail("native exact mutation saturated-draw count is not bound")
        if len(changed) != len(nonzero_details):
            _fail("native exact mutation changed-allele count is not bound")
        for index, raw_allele in enumerate(changed):
            if not isinstance(raw_allele, Mapping):
                _fail(f"native exact mutation allele {index} is invalid")
            region, haplotype, locus, pre_value, post_value = validate_coordinate(
                raw_allele, index=index, detail=False
            )
            if pre_value == post_value:
                _fail(f"native exact mutation allele {index} is not changed")
            key = (region, haplotype, locus)
            if key in changed_keys:
                _fail("native exact mutation has duplicate changed coordinates")
            changed_keys.append(key)
            if dict(raw_allele) != nonzero_details[index]:
                _fail(
                    "native exact mutation changed alleles do not equal the "
                    "non-saturated accepted-draw subset"
                )
        if changed_keys != [key for key, raw in zip(
            detail_keys,
            details,
            strict=True,
        ) if not raw["saturated_noop"]]:
            _fail("native exact mutation changed coordinates are not canonical")
        if result == "MUTATION_SATURATED_NOOP":
            if saturated_count != accepted_draws or changed:
                _fail("native saturated mutation result does not describe saturation")
        elif result == "MUTATION_APPLIED_WITH_SATURATED_NOOPS":
            if not changed or saturated_count <= 0:
                _fail("native mixed mutation result does not describe saturation")
        elif result == "MUTATION_APPLIED":
            if not changed or saturated_count != 0:
                _fail("native applied mutation result does not bind its draws")
        else:
            _fail(f"native exact mutation result class {result!r} is invalid")
        if result == "MUTATION_SATURATED_NOOP":
            if pre != post:
                _fail("native saturated mutation has different pre/post genome identity")
        elif pre == post:
            _fail("native applied mutation has identical pre/post genome identity")
        bounded = payload.get("bounded_delta")
        if not isinstance(bounded, Mapping):
            _fail("native exact mutation lacks bounded delta")
        if any(
            bounded.get(field) != expected
            for field, expected in (
                ("mutation_count", native_count),
                ("accepted_draws", accepted_draws),
                ("changed_allele_count", len(changed)),
                ("saturated_noop_count", saturated_count),
            )
        ):
            _fail("native exact mutation bounded delta does not bind coordinates")
    if result in _APPLIED_RESULTS:
        if not attempted:
            _fail("applied mutation has attempted=false")
        if pre == post:
            _fail("applied mutation has identical pre/post genome identity")
    return _Mutation(
        row=row,
        payload=payload,
        mutation_attempt_id=mutation_attempt_id,
        mutation_opportunity_id=opportunity_id,
        subject_agent_key=subject,
        origin_agent_key=origin,
        origin_absolute_tick=origin_tick,
        event_tick=event_tick,
        pre_genome_key=pre,
        post_genome_key=post,
        result_class=result,
        attempted=attempted,
    )


def _birth(row: Mapping[str, Any]) -> _Birth:
    payload = _payload(row, field="birth_committed.payload")
    birth_attempt = _identity_key(
        payload.get("birth_attempt_id"), field="birth_committed.birth_attempt_id"
    )
    opportunity = _identity_key(
        payload.get("opportunity_id"), field="birth_committed.opportunity_id"
    )
    child = _identity_key(
        payload.get("child_agent_id"), field="birth_committed.child_agent_id"
    )
    parent_agents = payload.get("parent_agent_ids")
    parent_genomes = payload.get("parent_genome_state_ids")
    if (
        not isinstance(parent_agents, list)
        or not isinstance(parent_genomes, list)
        or not parent_agents
        or len(parent_agents) != len(parent_genomes)
    ):
        _fail("birth_committed parent agent/genome arrays do not join")
    parent_pairs = tuple(
        (
            _identity_key(agent, field="birth_committed.parent_agent_ids[]"),
            _identity_key(genome, field="birth_committed.parent_genome_state_ids[]"),
        )
        for agent, genome in zip(parent_agents, parent_genomes, strict=True)
    )
    event_tick = row.get("absolute_tick")
    if type(event_tick) is not int or event_tick < 0:
        _fail("birth_committed event tick is invalid")
    return _Birth(
        row=row,
        payload=payload,
        birth_attempt_id=birth_attempt,
        opportunity_id=opportunity,
        child_agent_key=child,
        child_pre_genome_key=_identity_key(
            payload.get("child_pre_mutation_genome_state_id"),
            field="birth_committed.child_pre_mutation_genome_state_id",
        ),
        child_post_genome_key=_identity_key(
            payload.get("child_post_birth_genome_state_id"),
            field="birth_committed.child_post_birth_genome_state_id",
        ),
        parent_pairs=parent_pairs,
        event_tick=event_tick,
    )


def _death(row: Mapping[str, Any]) -> tuple[str, str, int]:
    payload = _payload(row, field="death.payload")
    tick = row.get("absolute_tick")
    if type(tick) is not int or tick < 0:
        _fail("death event tick is invalid")
    return (
        _identity_key(payload.get("agent_id"), field="death.agent_id"),
        _identity_key(payload.get("genome_state_id"), field="death.genome_state_id"),
        tick,
    )


def _count_kinds(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        kind = _record_kind(row)
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def _milestone_offsets(config: Mapping[str, Any]) -> tuple[int, ...]:
    raw = config.get(
        "milestone_offsets_ticks", config.get("carrier_milestone_offsets_ticks")
    )
    if raw is None:
        return (1, 2, 5, 10)
    if (
        not isinstance(raw, list)
        or not raw
        or any(type(item) is not int or item <= 0 for item in raw)
    ):
        _fail("carrier milestone offsets must be positive absolute-tick offsets")
    result = tuple(sorted(set(raw)))
    if result != tuple(raw):
        _fail("carrier milestone offsets must be sorted and unique")
    return result


def _world_result(
    world_uid_hex: str,
    rows: Sequence[Mapping[str, Any]],
    completion: Mapping[str, Any],
    owner: _WorldOwner,
    config: Mapping[str, Any],
    *,
    off_arm_ids: frozenset[str],
    on_arm_ids: frozenset[str],
) -> dict[str, Any]:
    kinds_present = set(config.get("_kinds_present", []))
    # The state carries partition kinds; this value is replaced by the caller
    # before this function is reached.  Keeping the check here makes malformed
    # direct calls fail closed too.
    if not kinds_present:
        kinds_present = {row["record_kind"] for row in rows}

    controls = [
        row
        for row in rows
        if _record_kind(row) == ScienceRecordKindV1.CONTROL_BINDING.value
    ]
    if not controls:
        _fail(f"world {world_uid_hex} has no control binding evidence")
    acquisition_ids = {
        _digest_hex(
            row.get("science_semantic_acquisition_sha256"),
            field="row.science_semantic_acquisition_sha256",
        )
        for row in rows
    }
    instrument_ids = {
        _digest_hex(
            row.get("instrument_manifest_sha256"),
            field="row.instrument_manifest_sha256",
        )
        for row in rows
    }
    if len(acquisition_ids) != 1:
        _fail(f"world {world_uid_hex} mixes science acquisition identities")
    if len(instrument_ids) != 1:
        _fail(f"world {world_uid_hex} mixes instrument identities")
    cell_ids = {_nonempty_text(row.get("cell_id"), field="row.cell_id") for row in rows}
    if len(cell_ids) != 1:
        _fail(f"world {world_uid_hex} mixes cell identities")
    if owner.cell_id is not None and owner.cell_id not in cell_ids:
        _fail(f"world {world_uid_hex} ownership cell identity differs")
    logical_world_ids = {row.get("logical_world_id") for row in rows}
    if len(logical_world_ids) != 1 or not all(
        type(item) is int for item in logical_world_ids
    ):
        _fail(f"world {world_uid_hex} mixes logical world identities")
    semantic_execution_ids = {
        _digest_hex(row.get("semantic_execution_id"), field="row.semantic_execution_id")
        for row in rows
    }
    if len(semantic_execution_ids) != 1:
        _fail(f"world {world_uid_hex} mixes semantic execution identities")
    role = _arm_role(owner.arm_id, off_arm_ids, on_arm_ids)
    control_binding_ids = {
        _digest_hex(
            row.get("control_binding_sha256"), field="row.control_binding_sha256"
        )
        for row in controls
    }
    native_controls: list[dict[str, Any]] = []
    for control_index, row in enumerate(controls):
        raw_payload = row.get("payload")
        payload_schema = (
            raw_payload.get("schema_version")
            if isinstance(raw_payload, Mapping)
            else None
        )
        payload = _payload(
            row, field=f"world[{world_uid_hex}].control[{control_index}].payload"
        )
        applied = _native_control_provenance(
            payload,
            field=f"world[{world_uid_hex}].control[{control_index}].payload",
        )
        if applied is None:
            if payload_schema == str(
                SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_VERSION
            ):
                _fail(
                    f"world {world_uid_hex} native control payload lacks applied provenance"
                )
            continue
        if payload_schema != str(SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_VERSION):
            _fail(
                f"world {world_uid_hex} applied native control uses an unsupported payload schema"
            )
        try:
            payload_digest = canonical_sha256(payload).hex
            control_digest = _digest_hex(
                row.get("control_binding_sha256"),
                field=f"world[{world_uid_hex}].control[{control_index}].control_binding_sha256",
            )
        except ScienceReducerError:
            raise
        if payload_digest != control_digest:
            _fail(
                f"world {world_uid_hex} native control binding digest does not bind its payload"
            )
        native_controls.append(applied)
    if native_controls and len(native_controls) != len(controls):
        _fail(f"world {world_uid_hex} mixes native and legacy control bindings")
    if native_controls and any(
        item != native_controls[0] for item in native_controls[1:]
    ):
        _fail(f"world {world_uid_hex} mixes applied native control receipts")
    if native_controls:
        mutation_enabled = _mapping(
            native_controls[0].get("control_vector"),
            field=f"world[{world_uid_hex}].native_control.control_vector",
        ).get("mutation.enabled")
        if role == "OFF" and mutation_enabled is not False:
            _fail(f"OFF arm {owner.arm_id!r} did not execute mutation.disabled")
        if role == "ON" and mutation_enabled is not True:
            _fail(f"ON arm {owner.arm_id!r} did not execute mutation.enabled")
    if any(
        _digest_hex(
            row.get("control_binding_sha256"), field="row.control_binding_sha256"
        )
        not in control_binding_ids
        for row in rows
    ):
        _fail(f"world {world_uid_hex} has a record without a control-binding join")

    mutation_rows = [
        row
        for row in rows
        if _record_kind(row) == ScienceRecordKindV1.MUTATION_OUTCOME.value
    ]
    birth_rows = [
        row
        for row in rows
        if _record_kind(row) == ScienceRecordKindV1.BIRTH_COMMITTED.value
    ]
    death_rows = [
        row for row in rows if _record_kind(row) == ScienceRecordKindV1.DEATH.value
    ]
    mutations = [_mutation(row) for row in mutation_rows]
    births = [_birth(row) for row in birth_rows]
    deaths = [_death(row) for row in death_rows]

    def unique(values: Iterable[str], what: str) -> None:
        values_list = list(values)
        if len(values_list) != len(set(values_list)):
            _fail(f"world {world_uid_hex} duplicate {what}")

    unique((item.mutation_attempt_id for item in mutations), "mutation_attempt_id")
    unique(
        (item.mutation_opportunity_id for item in mutations), "mutation_opportunity_id"
    )
    unique((item.birth_attempt_id for item in births), "committed birth ID")
    unique((item.opportunity_id for item in births), "committed birth opportunity ID")
    unique((item.child_agent_key for item in births), "child agent ID")

    applied = [item for item in mutations if item.applied]
    if role == "OFF" and mutations:
        _fail(f"mutation in OFF arm {owner.arm_id!r} is not permitted")

    outcomes: dict[str, int] = {}
    for item in mutations:
        outcomes[item.result_class] = outcomes.get(item.result_class, 0) + 1
    outcomes = dict(sorted(outcomes.items()))

    # A mutation is directly attributed only when its subject and post-mutation
    # genome occur on the committed birth that created that child.  Native exact
    # rows identify the mutated child as origin; legacy fixture rows identify a
    # parent.  This is intentionally stricter than matching a genome digest
    # anywhere in a stream.
    attributed: list[tuple[_Mutation, _Birth]] = []
    for mutation in applied:
        candidates = [
            birth
            for birth in births
            if birth.child_agent_key == mutation.subject_agent_key
            and birth.event_tick >= mutation.event_tick
            and birth.child_post_genome_key == mutation.post_genome_key
            and birth.child_pre_genome_key == mutation.pre_genome_key
            and (
                (
                    mutation.payload.get("mutation_class_id")
                    == "native_genome_v2_point_mutation"
                    and mutation.origin_agent_key == mutation.subject_agent_key
                )
                or mutation.origin_agent_key
                in {agent for agent, _ in birth.parent_pairs}
            )
        ]
        if len(candidates) != 1:
            _fail(
                f"world {world_uid_hex} missing or ambiguous mutation-to-child/genome join "
                f"for {mutation.mutation_attempt_id}"
            )
        attributed.append((mutation, candidates[0]))

    # Every committed birth may optionally have a matching attempt.  If an
    # attempt family was acquired, partial/mismatched attempt joins are an
    # integrity failure; the direct metric itself does not require that
    # optional family.
    attempt_rows = [
        row
        for row in rows
        if _record_kind(row) == ScienceRecordKindV1.BIRTH_ATTEMPT.value
    ]
    if attempt_rows or ScienceRecordKindV1.BIRTH_ATTEMPT.value in kinds_present:
        attempts: dict[str, Mapping[str, Any]] = {}
        for row in attempt_rows:
            payload = _payload(row, field="birth_attempt.payload")
            attempt_id = _identity_key(
                payload.get("birth_attempt_id"), field="birth_attempt.birth_attempt_id"
            )
            if attempt_id in attempts:
                _fail(f"world {world_uid_hex} duplicate birth_attempt_id")
            attempts[attempt_id] = payload
        for birth in births:
            attempt = attempts.get(birth.birth_attempt_id)
            if attempt is None:
                _fail(
                    f"world {world_uid_hex} missing birth attempt join for committed birth"
                )
            if (
                _identity_key(
                    attempt.get("opportunity_id"), field="birth_attempt.opportunity_id"
                )
                != birth.opportunity_id
            ):
                _fail(f"world {world_uid_hex} birth attempt opportunity join differs")

    # A later carrier is an exact (agent, genome) parent pair on a committed
    # birth at a strictly later absolute tick.  We do not infer carriers from
    # an aggregate population or from a genome delta.
    carrier_rows: dict[str, list[_Birth]] = {}
    retention: list[dict[str, Any]] = []
    for mutation, child_birth in attributed:
        later = [
            birth
            for birth in births
            if birth.event_tick > mutation.event_tick
            and (mutation.subject_agent_key, mutation.post_genome_key)
            in birth.parent_pairs
        ]
        later.sort(
            key=lambda birth: (birth.event_tick, birth.row["world_local_sequence"])
        )
        carrier_rows[mutation.mutation_attempt_id] = later
        first = later[0] if later else None
        retention.append(
            {
                "mutation_attempt_id": mutation.payload["mutation_attempt_id"],
                "mutation_opportunity_id": mutation.payload["mutation_opportunity_id"],
                "subject_agent_id": mutation.payload["subject_agent_id"],
                "post_genome_state_id": mutation.payload["post_genome_state_id"],
                "origin_absolute_tick": mutation.origin_absolute_tick,
                "attributed_child_agent_id": child_birth.payload["child_agent_id"],
                "attributed_child_birth_tick": child_birth.event_tick,
                "later_carrier_birth_tick": None if first is None else first.event_tick,
                "later_carrier_birth_attempt_id": (
                    None if first is None else first.payload["birth_attempt_id"]
                ),
                "later_committed_birth": first is not None,
            }
        )

    offsets = _milestone_offsets(config)
    milestones: list[dict[str, Any]] = []
    for offset in offsets:
        retained = [
            item
            for mutation, item in zip(applied, retention, strict=True)
            if any(
                birth.event_tick <= mutation.origin_absolute_tick + offset
                for birth in carrier_rows[
                    _identity_key(
                        mutation.payload["mutation_attempt_id"],
                        field="mutation_attempt_id",
                    )
                ]
            )
        ]
        milestones.append(
            {
                "offset_ticks": offset,
                "mutations_with_later_carrier": len(retained),
                "applied_mutation_count": len(applied),
                "retention_fraction": (
                    None if not applied else len(retained) / len(applied)
                ),
            }
        )

    deaths_before = 0
    death_supported = bool(death_rows)
    for mutation, item in zip(applied, retention, strict=True):
        carrier_tick = item["later_carrier_birth_tick"]
        if carrier_tick is not None:
            continue
        subject = mutation.subject_agent_key
        post = mutation.post_genome_key
        if any(
            agent == subject and genome == post and tick > mutation.event_tick
            for agent, genome, tick in deaths
        ):
            deaths_before += 1

    mutation_observation_semantics = config.get("mutation_observation_semantics")
    if not mutation_rows:
        if mutation_observation_semantics == (
            MUTSUP_MUTATION_OBSERVATION_SEMANTICS_APPLIED_OUTCOMES_ONLY_COMPLETE
        ):
            mutation_opportunity_status = (
                "MUTATION_DISABLED_CONTROL"
                if role == "OFF"
                else "NO_OBSERVED_MUTATIONS"
            )
        else:
            mutation_opportunity_status = "INSUFFICIENT_OPPORTUNITY_EVIDENCE"
    elif not applied:
        mutation_opportunity_status = "NO_OBSERVED_MUTATIONS"
    else:
        mutation_opportunity_status = "APPLIED_MUTATIONS_OBSERVED"
    direct_count = sum(item["later_committed_birth"] for item in retention)
    terminal_tick = completion["terminal_absolute_tick"]
    terminal_count = sum(
        item["later_carrier_birth_tick"] is not None
        and item["later_carrier_birth_tick"] <= terminal_tick
        for item in retention
    )
    if direct_count != terminal_count:
        _fail(f"world {world_uid_hex} later carrier exceeds terminal evidence")

    completeness = {
        "status": "COMPLETE",
        "evidence_status": "SUFFICIENT",
        "record_count": len(rows),
        "record_kinds_present": sorted(
            kinds_present,
            key=lambda item: _KIND_ORDINAL.get(ScienceRecordKindV1(item), 10_000),
        ),
        "required_record_kinds": list(_REQUIRED_KIND_NAMES),
        "sequence_contiguous": True,
        "terminal_absolute_tick": terminal_tick,
        "terminal_record_count_verified": True,
        "canonical_record_stream_verified": True,
    }
    counts = {
        "mutation_outcome_count": len(mutations),
        "mutation_opportunity_count": len(mutations),
        "applied_mutation_count": len(applied),
        "birth_committed_count": len(births),
        "death_count": len(deaths),
        "direct_reproductive_retention_count": direct_count,
        "later_carrier_count": direct_count,
    }
    fraction = None if not applied else direct_count / len(applied)
    death_metric = {
        "evidence_supported": death_supported,
        "count": deaths_before if death_supported else None,
        "denominator": len(applied) if death_supported else None,
        "fraction": (
            None if not death_supported or not applied else deaths_before / len(applied)
        ),
    }
    world = {
        "world_uid": {"algorithm": "sha256", "hex": world_uid_hex},
        "instrument_manifest_sha256": {
            "algorithm": "sha256",
            "hex": next(iter(instrument_ids)),
        },
        "science_semantic_acquisition_sha256": {
            "algorithm": "sha256",
            "hex": next(iter(acquisition_ids)),
        },
        "cell_id": next(iter(cell_ids)),
        "study_id": owner.study_id,
        "arm_id": owner.arm_id,
        "replicate": owner.replicate,
        "pair_id": owner.pair_id,
        "arm_role": role,
        "completeness": completeness,
        "record_counts_by_kind": _count_kinds(rows),
        "mutation_opportunity": {
            "status": mutation_opportunity_status,
            "observed_record_count": len(mutation_rows),
            "applied_record_count": len(applied),
            "opportunity_evidence_present": bool(mutation_rows),
        },
        "mutation_outcomes_by_class": outcomes,
        "counts": counts,
        "applied_mutation_count": len(applied),
        "direct_reproductive_retention_count": direct_count,
        "direct_reproductive_retention_fraction": fraction,
        "direct_reproductive_retention": {
            "count": direct_count,
            "denominator": len(applied),
            "fraction": fraction,
            "claim": "exact later committed parent carrier only",
        },
        "carrier_milestones": milestones,
        "deaths_before_retention": death_metric,
        "retained_mutations": retention,
    }
    return world


def _state_value(state: CanonicalPayloadV1) -> dict[str, Any]:
    if state.schema_version != MUTSUP_DIRECT_RETENTION_STATE_SCHEMA_VERSION:
        _fail("direct-retention reducer state schema differs")
    value = _mapping(state.to_value(), field="direct-retention state")
    worlds = value.get("worlds")
    if not isinstance(worlds, Mapping):
        _fail("direct-retention reducer state worlds are malformed")
    return value


def _configuration(configuration: CanonicalPayloadV1) -> dict[str, Any]:
    value = _mapping(configuration.to_value(), field="direct-retention configuration")
    _reject_config_generation_keys(value)
    bad = _integrity_bad(value)
    if bad:
        _fail(f"dropped/overflow evidence is not reducible: {bad}")
    semantics = value.get("mutation_observation_semantics")
    if semantics is not None and semantics != (
        MUTSUP_MUTATION_OBSERVATION_SEMANTICS_APPLIED_OUTCOMES_ONLY_COMPLETE
    ):
        _fail(
            "unsupported mutation_observation_semantics; expected "
            f"{MUTSUP_MUTATION_OBSERVATION_SEMANTICS_APPLIED_OUTCOMES_ONLY_COMPLETE}"
        )
    return value


def _result_identity(core: Mapping[str, Any]) -> dict[str, str]:
    return canonical_sha256(dict(core)).to_dict()


def _classification(
    worlds: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]]
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    statuses = {world["mutation_opportunity"]["status"] for world in worlds}
    if "INSUFFICIENT_OPPORTUNITY_EVIDENCE" in statuses:
        reasons.append(
            "at least one world has no observed mutation-opportunity records"
        )
    if "NO_OBSERVED_MUTATIONS" in statuses:
        reasons.append(
            "at least one world has mutation evidence but no applied mutation"
        )
    if "INSUFFICIENT_OPPORTUNITY_EVIDENCE" in statuses:
        return "INSUFFICIENT_OPPORTUNITY_EVIDENCE", reasons
    if contrasts:
        return "DESCRIPTIVE_PAIRED_OFF_ON", reasons
    if "NO_OBSERVED_MUTATIONS" in statuses:
        return "NO_OBSERVED_MUTATIONS", reasons
    return "DIRECT_RETENTION_OBSERVED", reasons


@dataclass(frozen=True, slots=True)
class MutsupDirectRetentionReducerV1:
    """Protocol implementation for ``science.mutsup.direct_retention.v1``."""

    implementation_sha256: ContentDigest
    reducer_id: str = MUTSUP_DIRECT_RETENTION_REDUCER_ID

    def initial_state(self, configuration: CanonicalPayloadV1) -> CanonicalPayloadV1:
        _configuration(configuration)
        return CanonicalPayloadV1.from_value(
            MUTSUP_DIRECT_RETENTION_STATE_SCHEMA_VERSION,
            {"worlds": {}},
        )

    def consume_partition(
        self,
        state: CanonicalPayloadV1,
        partition: ScienceEvidencePartitionV1,
    ) -> CanonicalPayloadV1:
        value = _state_value(state)
        worlds = {
            str(key): dict(item)
            for key, item in _mapping(value["worlds"], field="state.worlds").items()
        }
        world_key = partition.world_uid.hex
        world = _mapping(worlds.get(world_key, {}), field=f"state.worlds[{world_key}]")
        if (
            "world_uid" in world
            and _digest_hex(world["world_uid"], field="state.world_uid") != world_key
        ):
            _fail(f"partition world identity differs for {world_key}")
        rows = list(world.get("rows", []))
        rows.extend(row.semantic_dict() for row in partition.rows)
        kinds = set(world.get("partition_record_kinds", []))
        kinds.add(partition.record_kind.value)
        world["world_uid"] = partition.world_uid.to_dict()
        world["rows"] = rows
        world["partition_record_kinds"] = sorted(
            kinds, key=lambda item: _KIND_ORDINAL.get(ScienceRecordKindV1(item), 10_000)
        )
        world["cell_id"] = (
            partition.rows[0].cell_id if partition.rows else world.get("cell_id", "")
        )
        worlds[world_key] = world
        return CanonicalPayloadV1.from_value(
            MUTSUP_DIRECT_RETENTION_STATE_SCHEMA_VERSION,
            {"worlds": dict(sorted(worlds.items()))},
        )

    def finalize(
        self,
        state: CanonicalPayloadV1,
        configuration: CanonicalPayloadV1,
        *,
        output_contract_id: str,
    ) -> CanonicalPayloadV1:
        if output_contract_id != MUTSUP_DIRECT_RETENTION_OUTPUT_CONTRACT_ID:
            _fail("direct-retention output contract differs")
        config = _configuration(configuration)
        value = _state_value(state)
        raw_worlds = _mapping(value["worlds"], field="state.worlds")
        if not raw_worlds:
            _fail("direct-retention evidence contains no worlds")
        world_rows = {
            str(key): _mapping(item, field=f"state.worlds[{key}]")
            for key, item in raw_worlds.items()
        }
        owners = _parse_owners(config, world_rows)
        off_arm_ids, on_arm_ids = _pairing_arm_sets(config)
        worlds: list[dict[str, Any]] = []
        for world_uid_hex in sorted(world_rows):
            raw = world_rows[world_uid_hex]
            rows, completion = _validate_world_stream(world_uid_hex, raw)
            local_config = dict(config)
            local_config["_kinds_present"] = list(raw.get("partition_record_kinds", []))
            worlds.append(
                _world_result(
                    world_uid_hex,
                    rows,
                    completion,
                    owners[world_uid_hex],
                    local_config,
                    off_arm_ids=off_arm_ids,
                    on_arm_ids=on_arm_ids,
                )
            )
        worlds.sort(key=lambda item: item["world_uid"]["hex"])
        # Cross-arm pairing is intentionally outside the stream reducer.  The
        # protocol's one-acquisition identity and exact-prefix resume apply to
        # this per-world reduction; the study aggregator combines sealed
        # outputs using an explicit ownership/pair map.
        contrasts: list[dict[str, Any]] = []
        classification, reasons = _classification(worlds, contrasts)
        core: dict[str, Any] = {
            "reducer_id": self.reducer_id,
            "contract_version": str(MUTSUP_DIRECT_RETENTION_CONTRACT_VERSION),
            "world_count": len(worlds),
            "worlds": worlds,
            "paired_contrasts": contrasts,
            "decision": {
                "classification": classification,
                "reasons": reasons,
                "scientific_claims_made": False,
                "claim_boundary": MUTSUP_DIRECT_RETENTION_CLAIM_BOUNDARY,
            },
            "claim_boundary": MUTSUP_DIRECT_RETENTION_CLAIM_BOUNDARY,
            "physical_partition_invariant": True,
            "trajectory_effects_performed": False,
        }
        return CanonicalPayloadV1.from_value(
            SchemaVersion(output_contract_id),
            {**core, "result_identity_sha256": _result_identity(core)},
        )


def mutsup_direct_retention_reducer_descriptor_v1() -> ScienceReducerDescriptorV1:
    return ScienceReducerDescriptorV1(
        reducer_id=MUTSUP_DIRECT_RETENTION_REDUCER_ID,
        contract_version=MUTSUP_DIRECT_RETENTION_CONTRACT_VERSION,
        implementation_sha256=MUTSUP_DIRECT_RETENTION_IMPLEMENTATION_SHA256,
        configuration_schema_sha256=MUTSUP_DIRECT_RETENTION_CONFIGURATION_SHA256,
        required_observation_capability_ids=(
            MUTSUP_DIRECT_RETENTION_REQUIRED_CAPABILITY_IDS
        ),
        output_contract_id=MUTSUP_DIRECT_RETENTION_OUTPUT_CONTRACT_ID,
        purity_declarations=(
            "CONTENT_ADDRESSED_IMPLEMENTATION",
            "DETERMINISTIC_CANONICAL_OUTPUT",
            "NO_NETWORK",
            "NO_SIMULATOR_IMPORT",
            "NO_TRAJECTORY_MUTATION",
            "OFFLINE_ONLY",
        ),
    )


def default_mutsup_direct_retention_reducer_descriptor_v1() -> (
    ScienceReducerDescriptorV1
):
    return mutsup_direct_retention_reducer_descriptor_v1()


def mutsup_direct_retention_reducer_v1() -> MutsupDirectRetentionReducerV1:
    return MutsupDirectRetentionReducerV1(
        implementation_sha256=MUTSUP_DIRECT_RETENTION_IMPLEMENTATION_SHA256
    )


def execute_mutsup_direct_retention_v1(
    configuration: CanonicalPayloadV1,
    partitions: tuple[ScienceEvidencePartitionV1, ...],
    *,
    evidence_capability_ids: tuple[str, ...],
    campaign_spec_sha256: ContentDigest,
    instrument_manifest_sha256: ContentDigest,
    route_binding_sha256: ContentDigest,
    science_semantic_acquisition_sha256: ContentDigest,
    physical_execution_plan_sha256: ContentDigest,
    science_execution_plan_sha256: ContentDigest,
    reduction_plan_sha256: ContentDigest,
    resume: ScienceReducerCheckpointV1 | None = None,
    stop_after_partitions: int | None = None,
) -> ScienceReducerProgressV1:
    """Execute through the canonical protocol using the registered reducer."""

    return execute_science_reducer_v1(
        mutsup_direct_retention_reducer_descriptor_v1(),
        mutsup_direct_retention_reducer_v1(),
        configuration,
        partitions,
        evidence_capability_ids=evidence_capability_ids,
        campaign_spec_sha256=campaign_spec_sha256,
        instrument_manifest_sha256=instrument_manifest_sha256,
        route_binding_sha256=route_binding_sha256,
        science_semantic_acquisition_sha256=science_semantic_acquisition_sha256,
        physical_execution_plan_sha256=physical_execution_plan_sha256,
        science_execution_plan_sha256=science_execution_plan_sha256,
        reduction_plan_sha256=reduction_plan_sha256,
        resume=resume,
        stop_after_partitions=stop_after_partitions,
    )


def _pair_key(world: Mapping[str, Any]) -> tuple[str, str | int, str | None]:
    return (
        _nonempty_text(world.get("study_id"), field="world.study_id"),
        _normalise_replicate(world.get("replicate"), field="world.replicate"),
        None
        if world.get("pair_id") is None
        else _nonempty_text(world.get("pair_id"), field="world.pair_id"),
    )


def _contrast(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    pair_key: tuple[str, str | int, str | None],
) -> dict[str, Any]:
    def metric(name: str) -> dict[str, Any]:
        left_value = left.get(name)
        right_value = right.get(name)
        if left_value is None or right_value is None:
            difference = None
        elif type(left_value) not in (int, float) or type(right_value) not in (
            int,
            float,
        ):
            _fail(f"paired metric {name} is not numeric")
        else:
            difference = right_value - left_value
        return {"off": left_value, "on": right_value, "on_minus_off": difference}

    return {
        "pair_key": {
            "study_id": pair_key[0],
            "replicate": pair_key[1],
            "pair_id": pair_key[2],
        },
        "off_world_uid": left["world_uid"],
        "on_world_uid": right["world_uid"],
        "off_arm_id": left["arm_id"],
        "on_arm_id": right["arm_id"],
        "off_science_semantic_acquisition_sha256": left.get(
            "science_semantic_acquisition_sha256"
        ),
        "on_science_semantic_acquisition_sha256": right.get(
            "science_semantic_acquisition_sha256"
        ),
        "status": "DESCRIPTIVE_ONLY",
        "metrics": {
            "applied_mutation_count": metric("applied_mutation_count"),
            "direct_reproductive_retention_count": metric(
                "direct_reproductive_retention_count"
            ),
            "direct_reproductive_retention_fraction": metric(
                "direct_reproductive_retention_fraction"
            ),
        },
        "claim_boundary": MUTSUP_DIRECT_RETENTION_CLAIM_BOUNDARY,
    }


def _paired_contrasts_from_worlds(
    worlds: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    off_arm_ids: frozenset[str],
    on_arm_ids: frozenset[str],
    require_pairs: bool,
) -> list[dict[str, Any]]:
    explicit = _explicit_pair_specs(config)
    if explicit is not None:
        world_by_uid = {
            _digest_hex(world.get("world_uid"), field="world.world_uid"): world
            for world in worlds
        }
        used: set[str] = set()
        contrasts: list[dict[str, Any]] = []
        for index, spec in enumerate(explicit):
            off_ref = None
            on_ref = None
            for key in ("off_world_uid", "off_world_id", "off_world"):
                if key in spec:
                    off_ref = spec[key]
                    break
            for key in ("on_world_uid", "on_world_id", "on_world"):
                if key in spec:
                    on_ref = spec[key]
                    break
            if off_ref is None or on_ref is None:
                _fail(
                    f"configuration.pairing.pairs[{index}] requires OFF and ON worlds"
                )
            off_uid = _world_reference(
                off_ref,
                world_by_uid,
                field=f"configuration.pairing.pairs[{index}].off_world",
            )
            on_uid = _world_reference(
                on_ref,
                world_by_uid,
                field=f"configuration.pairing.pairs[{index}].on_world",
            )
            if off_uid == on_uid:
                _fail(f"configuration.pairing.pairs[{index}] repeats one world")
            if off_uid in used or on_uid in used:
                _fail(f"configuration.pairing.pairs[{index}] reuses a world")
            used.update((off_uid, on_uid))
            left, right = world_by_uid[off_uid], world_by_uid[on_uid]
            if _arm_role(left["arm_id"], off_arm_ids, on_arm_ids) != "OFF":
                _fail(f"explicit OFF endpoint {off_uid} does not have OFF arm role")
            if _arm_role(right["arm_id"], off_arm_ids, on_arm_ids) != "ON":
                _fail(f"explicit ON endpoint {on_uid} does not have ON arm role")
            if _pair_key(left)[:2] != _pair_key(right)[:2]:
                _fail(f"explicit pair {index} crosses study or replicate ownership")
            pair_id = spec.get("pair_id", _pair_key(left)[2])
            if pair_id is not None:
                pair_id = _nonempty_text(
                    pair_id, field=f"configuration.pairing.pairs[{index}].pair_id"
                )
            configured_study = spec.get("study_id")
            if configured_study is not None and configured_study != _pair_key(left)[0]:
                _fail(f"explicit pair {index} study ownership differs")
            configured_replicate = spec.get("replicate")
            if (
                configured_replicate is not None
                and _normalise_replicate(
                    configured_replicate,
                    field=f"configuration.pairing.pairs[{index}].replicate",
                )
                != _pair_key(left)[1]
            ):
                _fail(f"explicit pair {index} replicate ownership differs")
            left_pair_id = _pair_key(left)[2]
            right_pair_id = _pair_key(right)[2]
            if (
                pair_id is not None
                and left_pair_id is not None
                and pair_id != left_pair_id
            ) or (
                pair_id is not None
                and right_pair_id is not None
                and pair_id != right_pair_id
            ):
                _fail(f"explicit pair {index} pair_id ownership differs")
            contrasts.append(
                _contrast(
                    left,
                    right,
                    pair_key=(_pair_key(left)[0], _pair_key(left)[1], pair_id),
                )
            )
        if require_pairs and used != set(world_by_uid):
            missing = sorted(set(world_by_uid) - used)
            _fail(f"invalid pairing: explicit map leaves worlds unpaired: {missing}")
        contrasts.sort(
            key=lambda item: (
                item["pair_key"]["study_id"],
                str(item["pair_key"]["replicate"]),
                ""
                if item["pair_key"]["pair_id"] is None
                else item["pair_key"]["pair_id"],
                _digest_hex(item["off_world_uid"], field="contrast.off_world_uid"),
            )
        )
        if require_pairs and not contrasts:
            _fail("invalid pairing: no complete OFF/ON pair")
        return contrasts

    groups: dict[tuple[str, str | int, str | None], list[Mapping[str, Any]]] = {}
    for world in worlds:
        groups.setdefault(_pair_key(world), []).append(world)
    contrasts: list[dict[str, Any]] = []
    for key in sorted(
        groups,
        key=lambda item: (item[0], str(item[1]), "" if item[2] is None else item[2]),
    ):
        rows = groups[key]
        off = [
            world
            for world in rows
            if _arm_role(world["arm_id"], off_arm_ids, on_arm_ids) == "OFF"
        ]
        on = [
            world
            for world in rows
            if _arm_role(world["arm_id"], off_arm_ids, on_arm_ids) == "ON"
        ]
        if not off and not on:
            if require_pairs:
                _fail(f"pairing has no OFF/ON arm for {key!r}")
            continue
        if len(off) != 1 or len(on) != 1:
            _fail(f"invalid OFF/ON pairing for study/replicate {key!r}")
        left, right = off[0], on[0]

        contrasts.append(_contrast(left, right, pair_key=key))
    if require_pairs and not contrasts:
        _fail("invalid pairing: no complete OFF/ON pair")
    return contrasts


def _verify_sealed_world_result(
    data: Mapping[str, Any],
    *,
    field: str,
) -> list[Mapping[str, Any]]:
    """Validate the immutable identity envelope before study aggregation."""

    if data.get("reducer_id") != MUTSUP_DIRECT_RETENTION_REDUCER_ID:
        _fail(f"{field} reducer identity differs")
    if data.get("physical_partition_invariant") is not True:
        _fail(f"{field} does not declare physical partition invariance")
    if data.get("trajectory_effects_performed") is not False:
        _fail(f"{field} is not an offline result")
    worlds = data.get("worlds")
    if not isinstance(worlds, list) or not worlds:
        _fail(f"{field}.worlds is empty or malformed")
    identity = data.get("result_identity_sha256")
    if _digest_hex(identity, field=f"{field}.result_identity_sha256") != _digest_hex(
        _result_identity(
            {
                key: value
                for key, value in data.items()
                if key != "result_identity_sha256"
            }
        ),
        field=f"{field}.result_identity_sha256",
    ):
        _fail(f"{field} result identity differs")
    for index, world in enumerate(worlds):
        world_data = _mapping(world, field=f"{field}.worlds[{index}]")
        completeness = _mapping(
            world_data.get("completeness"),
            field=f"{field}.worlds[{index}].completeness",
        )
        if completeness.get("status") != "COMPLETE":
            _fail(f"{field}.worlds[{index}] is not complete")
        _digest_hex(
            world_data.get("world_uid"), field=f"{field}.worlds[{index}].world_uid"
        )
        _digest_hex(
            world_data.get("instrument_manifest_sha256"),
            field=f"{field}.worlds[{index}].instrument_manifest_sha256",
        )
        _digest_hex(
            world_data.get("science_semantic_acquisition_sha256"),
            field=f"{field}.worlds[{index}].science_semantic_acquisition_sha256",
        )
    return worlds


def _apply_aggregate_ownership(
    worlds: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> None:
    ownership_keys = (
        "world_ownership",
        "world_metadata",
        "worlds",
        "ownership",
        "science_ownership",
        "science",
    )
    if not any(key in config for key in ownership_keys):
        return
    world_rows = {
        _digest_hex(world.get("world_uid"), field="world.world_uid"): world
        for world in worlds
    }
    owners = _parse_owners(config, world_rows)
    for world_uid, owner in owners.items():
        world = world_rows[world_uid]
        expected = {
            "study_id": owner.study_id,
            "arm_id": owner.arm_id,
            "replicate": owner.replicate,
            "pair_id": owner.pair_id,
        }
        for key, value in expected.items():
            if key in world and world[key] != value:
                _fail(f"aggregate ownership differs for world {world_uid}: {key}")
            world[key] = value
        if owner.cell_id is not None:
            if world.get("cell_id") != owner.cell_id:
                _fail(
                    f"aggregate ownership cell identity differs for world {world_uid}"
                )


def aggregate_mutsup_direct_retention_v1(
    world_results: Iterable[CanonicalPayloadV1 | Mapping[str, Any]],
    *,
    pairing: CanonicalPayloadV1 | Mapping[str, Any] | None = None,
) -> CanonicalPayloadV1:
    """Aggregate sealed per-world results into descriptive OFF/ON contrasts.

    Pairing is intentionally separate from the stream reducer.  It consumes
    only already reduced per-world payloads and explicit ownership/mapping, so
    cross-arm grouping cannot accidentally depend on evidence partition order
    or physical placement.
    """

    raw_worlds: list[dict[str, Any]] = []
    source_result_identities: list[str] = []
    inherited_config: dict[str, Any] = {}
    for index, item in enumerate(world_results):
        if isinstance(item, CanonicalPayloadV1):
            if item.schema_version != SchemaVersion(
                MUTSUP_DIRECT_RETENTION_OUTPUT_CONTRACT_ID
            ):
                _fail(f"world result {index} output contract differs")
            data = _mapping(item.to_value(), field=f"world_results[{index}]")
        else:
            data = _mapping(item, field=f"world_results[{index}]")
        worlds = _verify_sealed_world_result(data, field=f"world_results[{index}]")
        source_result_identities.append(
            _digest_hex(
                data.get("result_identity_sha256"),
                field=f"world_results[{index}].result_identity_sha256",
            )
        )
        for world in worlds:
            raw_worlds.append(dict(world))
    if not raw_worlds:
        _fail("study aggregation requires at least one world result")
    seen = [
        _digest_hex(world.get("world_uid"), field="world.world_uid")
        for world in raw_worlds
    ]
    if len(seen) != len(set(seen)):
        _fail("duplicate world identity in study aggregation")
    if pairing is not None:
        if isinstance(pairing, CanonicalPayloadV1):
            inherited_config = _configuration(pairing)
        else:
            inherited_config = _configuration(
                CanonicalPayloadV1.from_value(
                    MUTSUP_DIRECT_RETENTION_CONFIGURATION_SCHEMA_VERSION,
                    pairing,
                )
            )
        off_arm_ids, on_arm_ids = _pairing_arm_sets(inherited_config)
    else:
        off_arm_ids, on_arm_ids = _OFF_ARM_DEFAULTS, _ON_ARM_DEFAULTS
    _apply_aggregate_ownership(raw_worlds, inherited_config)
    contrasts = _paired_contrasts_from_worlds(
        raw_worlds,
        inherited_config,
        off_arm_ids=off_arm_ids,
        on_arm_ids=on_arm_ids,
        require_pairs=True,
    )
    classifications = {
        _mapping(
            world.get("mutation_opportunity"), field="world.mutation_opportunity"
        ).get("status")
        for world in raw_worlds
    }
    decision_reasons: list[str] = []
    if "INSUFFICIENT_OPPORTUNITY_EVIDENCE" in classifications:
        decision_reasons.append(
            "at least one world has no observed mutation-opportunity records"
        )
    if "NO_OBSERVED_MUTATIONS" in classifications:
        decision_reasons.append(
            "at least one world has mutation evidence but no applied mutation"
        )
    if "INSUFFICIENT_OPPORTUNITY_EVIDENCE" in classifications:
        classification = "INSUFFICIENT_OPPORTUNITY_EVIDENCE"
    elif contrasts:
        classification = "DESCRIPTIVE_PAIRED_OFF_ON"
    else:
        classification = "NOT_INTERPRETABLE"
    core = {
        "reducer_id": MUTSUP_DIRECT_RETENTION_REDUCER_ID,
        "aggregation_contract_version": "science_mutsup_direct_retention_study_aggregate_v1",
        "world_count": len(raw_worlds),
        "worlds": sorted(
            raw_worlds,
            key=lambda world: _digest_hex(world["world_uid"], field="world.world_uid"),
        ),
        "source_result_identities": [
            {"algorithm": "sha256", "hex": value}
            for value in sorted(source_result_identities)
        ],
        "science_semantic_acquisition_identities": [
            world["science_semantic_acquisition_sha256"]
            for world in sorted(
                raw_worlds,
                key=lambda world: _digest_hex(
                    world["world_uid"], field="world.world_uid"
                ),
            )
        ],
        "paired_contrasts": contrasts,
        "decision": {
            "classification": classification,
            "reasons": decision_reasons,
            "scientific_claims_made": False,
            "claim_boundary": MUTSUP_DIRECT_RETENTION_CLAIM_BOUNDARY,
        },
        "claim_boundary": MUTSUP_DIRECT_RETENTION_CLAIM_BOUNDARY,
        "physical_partition_invariant": True,
        "trajectory_effects_performed": False,
    }
    return CanonicalPayloadV1.from_value(
        SchemaVersion(MUTSUP_DIRECT_RETENTION_OUTPUT_CONTRACT_ID),
        {**core, "result_identity_sha256": _result_identity(core)},
    )


# The implementation identity is source-bound to the reducer body.  Excluding
# this declaration avoids a self-referential digest while ensuring that a
# reducer semantic repair changes its descriptor without invalidating the
# native/CUDA artifact identity.
MUTSUP_DIRECT_RETENTION_CONFIGURATION_SHA256 = canonical_sha256(
    {
        "schema_version": str(MUTSUP_DIRECT_RETENTION_CONFIGURATION_SCHEMA_VERSION),
        "required": ["world_ownership", "pairing", "milestone_offsets_ticks"],
        "ownership_fields": ["world_uid", "study_id", "arm_id", "replicate", "pair_id"],
        "evidence_integrity": "dropped_or_overflow_is_fail_closed",
    }
)
_IMPLEMENTATION_SOURCE_MARKER = b"\n# The implementation identity is source-bound"


def _implementation_source_sha256() -> str:
    source = Path(__file__).read_bytes()
    prefix, separator, _ = source.partition(_IMPLEMENTATION_SOURCE_MARKER)
    if not separator:
        raise RuntimeError("MUTSUP reducer implementation marker is missing")
    return hashlib.sha256(prefix).hexdigest()


MUTSUP_DIRECT_RETENTION_IMPLEMENTATION_SHA256 = canonical_sha256(
    {
        "reducer_id": MUTSUP_DIRECT_RETENTION_REDUCER_ID,
        "contract_version": str(MUTSUP_DIRECT_RETENTION_CONTRACT_VERSION),
        "implementation_source_sha256": _implementation_source_sha256(),
        "claim_boundary": MUTSUP_DIRECT_RETENTION_CLAIM_BOUNDARY,
    }
)

# Friendly aliases used by callers that spell MUTSUP in uppercase.
MUTSUPDirectRetentionReducerV1 = MutsupDirectRetentionReducerV1
default_mutsup_direct_retention_descriptor_v1 = (
    mutsup_direct_retention_reducer_descriptor_v1
)
reduce_mutsup_direct_retention_study_v1 = aggregate_mutsup_direct_retention_v1


__all__ = (
    "MUTSUP_DIRECT_RETENTION_CLAIM_BOUNDARY",
    "MUTSUP_DIRECT_RETENTION_CONFIGURATION_SCHEMA_VERSION",
    "MUTSUP_DIRECT_RETENTION_CONFIGURATION_SHA256",
    "MUTSUP_DIRECT_RETENTION_CONTRACT_VERSION",
    "MUTSUP_DIRECT_RETENTION_IMPLEMENTATION_SHA256",
    "MUTSUP_DIRECT_RETENTION_OUTPUT_CONTRACT_ID",
    "MUTSUP_DIRECT_RETENTION_REDUCER_ID",
    "MUTSUP_DIRECT_RETENTION_REQUIRED_CAPABILITY_IDS",
    "MUTSUP_DIRECT_RETENTION_REQUIRED_RECORD_KINDS",
    "MUTSUP_DIRECT_RETENTION_STATE_SCHEMA_VERSION",
    "MUTSUP_MUTATION_OBSERVATION_SEMANTICS_APPLIED_OUTCOMES_ONLY_COMPLETE",
    "MUTSUPDirectRetentionReducerV1",
    "MutsupDirectRetentionReducerV1",
    "aggregate_mutsup_direct_retention_v1",
    "default_mutsup_direct_retention_descriptor_v1",
    "default_mutsup_direct_retention_reducer_descriptor_v1",
    "mutsup_direct_retention_reducer_descriptor_v1",
    "mutsup_direct_retention_reducer_v1",
    "reduce_mutsup_direct_retention_study_v1",
)
