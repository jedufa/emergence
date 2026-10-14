"""Generic strategy attribution over sealed compact science captures.

This module is deliberately independent of any study or founder panel.  A
world declares its strategy identity mapping and denominator contract in
``EIEL_SCIENCE_WORLD_BUNDLE_V1``; the analyzer follows parent ancestry through
the compact lifecycle stream and emits exact, logical-world-owned funnels.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from emergence_contracts import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
    COMPACT_SCIENCE_DEATH_METRIC_ENCODING_V1,
    COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1,
    COMPACT_SCIENCE_DISPATCH_GZIP_ENCODING_V1,
    COMPACT_SCIENCE_DISPATCH_SCHEMA_V1,
    COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1,
    COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1,
    SCIENCE_GENOTYPE_ATTRIBUTION_SCHEMA_V1,
    SCIENCE_GENOTYPE_IDENTITY_MODE_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1,
    ScienceWorldBundleV1,
    ScienceWorldRecordV1,
    canonical_sha256,
)

if TYPE_CHECKING:
    from emergence_evidence.native_compact_reducer_v1 import (
        NativeStrategyReceiptResultV1,
    )


STRATEGY_ATTRIBUTION_RESULT_SCHEMA_V1 = (
    "eiel_science_strategy_attribution_result_v1"
)
_EVENT_ROW_STRUCT = struct.Struct("<QQQQ")
_STRATEGY_CONTRACT_SCHEMA = "eiel_science_strategy_attribution_v1"


class StrategyAttributionError(ValueError):
    """A sealed capture or declared strategy contract is inconsistent."""


def strategy_union_v1(*strategy_sets: Sequence[str]) -> tuple[str, ...]:
    """Return the deterministic ancestry label for one or more parents."""

    values = {
        str(value)
        for group in strategy_sets
        for value in group
        if isinstance(value, str) and value
    }
    if not values:
        raise StrategyAttributionError("strategy ancestry is empty")
    return tuple(sorted(values))


def strategy_group_id_v1(strategy_ids: Sequence[str]) -> str:
    values = strategy_union_v1(strategy_ids)
    return "+".join(values)


def _as_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise StrategyAttributionError(f"{label} must be an integer")
    return int(value)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StrategyAttributionError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StrategyAttributionError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise StrategyAttributionError(f"{label} must be an object")
    return value


def _load_bundle(value: str | Path | ScienceWorldBundleV1) -> ScienceWorldBundleV1:
    if isinstance(value, ScienceWorldBundleV1):
        return value
    return ScienceWorldBundleV1.from_dict(
        _read_json(Path(value).expanduser().resolve(), label="world bundle")
    )


def _load_index(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return _read_json(Path(value).expanduser().resolve(), label="capture index")


def _verify_artifact(
    pointer: Mapping[str, Any], *, label: str
) -> tuple[Path, dict[str, Any]]:
    raw_path = pointer.get("path")
    expected = pointer.get("artifact_sha256")
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        raise StrategyAttributionError(f"{label} pointer identity is invalid")
    root = Path(raw_path).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise StrategyAttributionError(f"{label} root is not a directory")
    manifest = _read_json(root / "manifest.json", label=f"{label} manifest")
    if manifest.get("artifact_sha256") != expected:
        raise StrategyAttributionError(f"{label} pointer identity differs")
    body = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    if canonical_sha256(body).hex != expected:
        raise StrategyAttributionError(f"{label} manifest digest differs")
    return root, manifest


def _payload(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    prefix: str,
    label: str,
) -> bytes:
    name = manifest.get(f"{prefix}_file")
    expected = manifest.get(f"{prefix}_file_sha256")
    if not isinstance(name, str) or Path(name).name != name:
        raise StrategyAttributionError(f"{label} file name is invalid")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise StrategyAttributionError(f"{label} file is absent")
    value = path.read_bytes()
    if not isinstance(expected, str) or hashlib.sha256(value).hexdigest() != expected:
        raise StrategyAttributionError(f"{label} file digest differs")
    encoding = manifest.get("dispatch_payload_storage_encoding")
    if encoding is None:
        return value
    if encoding != COMPACT_SCIENCE_DISPATCH_GZIP_ENCODING_V1:
        raise StrategyAttributionError(f"{label} storage encoding differs")
    if manifest.get(f"{prefix}_stored_bytes") != len(value):
        raise StrategyAttributionError(f"{label} stored length differs")
    try:
        decoded = gzip.decompress(value)
    except (OSError, EOFError) as exc:
        raise StrategyAttributionError(f"{label} compressed payload is invalid") from exc
    expected_decoded = manifest.get(f"{prefix}_uncompressed_sha256")
    if (
        not isinstance(expected_decoded, str)
        or hashlib.sha256(decoded).hexdigest() != expected_decoded
    ):
        raise StrategyAttributionError(f"{label} uncompressed digest differs")
    return decoded


def strategy_attribution_contract_v1(
    world: ScienceWorldRecordV1,
) -> dict[str, Any] | None:
    """Return one validated explicit strategy binding, if the world opts in."""

    for binding in world.reducers.to_value():
        configuration = binding.get("configuration")
        if not isinstance(configuration, Mapping):
            continue
        candidate = configuration.get("strategy_attribution")
        if isinstance(candidate, Mapping):
            value = dict(candidate)
            if value.get("schema_version") != _STRATEGY_CONTRACT_SCHEMA:
                raise StrategyAttributionError("strategy contract schema differs")
            ids = value.get("strategy_ids")
            mapping = value.get("policy_id_to_strategy_id")
            denominators = value.get("denominator_fields")
            if (
                not isinstance(ids, list)
                or ids != sorted(set(ids))
                or not ids
                or any(not isinstance(item, str) or not item for item in ids)
                or not isinstance(mapping, Mapping)
                or any(
                    not isinstance(key, str)
                    or not isinstance(strategy, str)
                    or strategy not in ids
                    for key, strategy in mapping.items()
                )
                or not isinstance(denominators, list)
                or any(
                    field not in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
                    for field in denominators
                )
            ):
                raise StrategyAttributionError("strategy contract fields are invalid")
            direct_genotype = value.get("direct_genotype_attribution", False)
            if type(direct_genotype) is not bool:
                raise StrategyAttributionError(
                    "strategy contract genotype mode is invalid"
                )
            if direct_genotype and (
                value.get("genotype_identity_mode")
                not in {
                    SCIENCE_GENOTYPE_IDENTITY_MODE_V1,
                    # A failed pre-repair run may be replayed from its
                    # source-bound materialization.  That materialization
                    # used this token before the direct identity was made
                    # explicit as the declared tracked-genotype projection.
                    "exact_genome_state_id_v1",
                }
                or value.get("genotype_denominator_fields")
                != list(SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1)
            ):
                raise StrategyAttributionError(
                    "strategy contract genotype binding is invalid"
                )
            if value.get("genotype_identity_mode") == "exact_genome_state_id_v1":
                value["genotype_identity_mode"] = SCIENCE_GENOTYPE_IDENTITY_MODE_V1
            return value
    return None


def strategy_attribution_required_for_bundle_v1(
    world_bundle: ScienceWorldBundleV1,
) -> bool:
    """Return whether the bundle requires one strategy result per batch."""

    configured = []
    for world in world_bundle.worlds:
        configured.append(strategy_attribution_contract_v1(world) is not None)
    if any(configured) and not all(configured):
        raise StrategyAttributionError(
            "strategy attribution must be configured for every bundle world"
        )
    return bool(configured and configured[0])


def genotype_attribution_required_v1(contract: Mapping[str, Any]) -> bool:
    return contract.get("direct_genotype_attribution", False) is True


def validate_genotype_reproductive_attribution_v1(
    value: Mapping[str, Any],
) -> None:
    """Validate exact reproductive attribution at the declared loci."""

    if not isinstance(value, Mapping) or set(value) not in ({
        "schema_version",
        "identity_mode",
        "identity_scope",
        "group_count",
        "groups",
        "totals",
    }, {
        "schema_version",
        "identity_mode",
        "identity_scope",
        "group_scope",
        "group_count",
        "groups",
        "totals",
    }, {
        "schema_version",
        "identity_mode",
        "identity_scope",
        "group_count",
        "groups",
        "descendant_birth_counts_by_parent_tracked_identity",
        "totals",
    }, {
        "schema_version",
        "identity_mode",
        "identity_scope",
        "group_scope",
        "group_count",
        "groups",
        "descendant_birth_counts_by_parent_tracked_identity",
        "totals",
    }):
        raise StrategyAttributionError("genotype attribution result fields are invalid")
    if value.get("group_scope") is not None:
        raise StrategyAttributionError(
            "focal-only genotype attribution cannot satisfy complete direct attribution"
        )
    if value.get("schema_version") != SCIENCE_GENOTYPE_ATTRIBUTION_SCHEMA_V1:
        raise StrategyAttributionError("genotype attribution result schema differs")
    if value.get("identity_mode") != SCIENCE_GENOTYPE_IDENTITY_MODE_V1:
        raise StrategyAttributionError("genotype attribution identity mode differs")
    scope = value.get("identity_scope")
    if not isinstance(scope, Mapping) or set(scope) != {
        "policy_loci",
        "trait_loci",
    }:
        raise StrategyAttributionError("genotype attribution identity scope is invalid")
    policy_loci = scope.get("policy_loci")
    trait_loci = scope.get("trait_loci")
    if (
        not isinstance(policy_loci, list)
        or policy_loci != sorted(set(policy_loci))
        or any(type(locus) is not int or not 0 <= locus < 112 for locus in policy_loci)
        or not isinstance(trait_loci, list)
        or trait_loci != sorted(set(trait_loci))
        or any(type(locus) is not int or not 0 <= locus < 16 for locus in trait_loci)
    ):
        raise StrategyAttributionError("genotype attribution identity scope is invalid")
    groups = value.get("groups")
    group_count = value.get("group_count")
    if (
        type(group_count) is not int
        or group_count < 0
        or not isinstance(groups, list)
        or group_count != len(groups)
    ):
        raise StrategyAttributionError("genotype attribution group count is invalid")
    aggregate = {
        "agent_count": 0,
        "founder_count": 0,
        "offspring_birth_count": 0,
        "parent_birth_event_count": 0,
        "reproductive_agent_count": 0,
        "death_count": 0,
        "terminal_alive_count": 0,
    }
    aggregate_metrics = {
        field: 0 for field in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
    }
    prior_digest = ""
    for group in groups:
        if not isinstance(group, Mapping) or set(group) != {
            "genome_state_id",
            "agent_count",
            "founder_count",
            "offspring_birth_count",
            "parent_birth_event_count",
            "reproductive_agent_count",
            "death_count",
            "terminal_alive_count",
            "reproduction_funnel",
            "metric_totals",
        }:
            raise StrategyAttributionError("genotype attribution group fields are invalid")
        digest = group.get("genome_state_id")
        if (
            not isinstance(digest, str)
            or len(digest) != 71
            or not digest.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in digest[7:])
            or digest <= prior_digest
        ):
            raise StrategyAttributionError("genotype attribution identity ordering is invalid")
        prior_digest = digest
        for field in aggregate:
            number = group.get(field)
            if type(number) is not int or number < 0:
                raise StrategyAttributionError(
                    f"genotype attribution {field} is invalid"
                )
            aggregate[field] += number
        if group["agent_count"] != (
            group["founder_count"] + group["offspring_birth_count"]
        ) or group["death_count"] + group["terminal_alive_count"] != group[
            "agent_count"
        ]:
            raise StrategyAttributionError("genotype attribution population accounting differs")
        if group["reproductive_agent_count"] > group["agent_count"]:
            raise StrategyAttributionError("genotype reproductive-agent count exceeds population")
        funnel = group.get("reproduction_funnel")
        if not isinstance(funnel, Mapping) or set(funnel) != set(
            SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1
        ):
            raise StrategyAttributionError("genotype reproduction funnel fields are invalid")
        metrics = group.get("metric_totals")
        if not isinstance(metrics, Mapping) or set(metrics) != set(
            COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
        ):
            raise StrategyAttributionError("genotype metric totals are invalid")
        for field in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1:
            number = metrics.get(field)
            if type(number) is not int:
                raise StrategyAttributionError("genotype metric total is invalid")
            aggregate_metrics[field] += number
            if field in funnel and funnel[field] != number:
                raise StrategyAttributionError(
                    "genotype reproduction funnel denominator differs from metric total"
                )
        for field in SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1:
            if type(funnel[field]) is not int or funnel[field] < 0:
                raise StrategyAttributionError(
                    "genotype reproduction funnel denominator is invalid"
                )
    totals = value.get("totals")
    if not isinstance(totals, Mapping) or set(totals) != {
        *aggregate,
        "reproduction_funnel",
        "metric_totals",
    }:
        raise StrategyAttributionError("genotype attribution totals are invalid")
    for field, expected in aggregate.items():
        if totals.get(field) != expected:
            raise StrategyAttributionError("genotype attribution count totals differ")
    total_funnel = totals.get("reproduction_funnel")
    total_metrics = totals.get("metric_totals")
    if not isinstance(total_funnel, Mapping) or set(total_funnel) != set(
        SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1
    ) or not isinstance(total_metrics, Mapping) or set(total_metrics) != set(
        COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
    ):
        raise StrategyAttributionError("genotype attribution total funnel is invalid")
    for field, expected in aggregate_metrics.items():
        if total_metrics.get(field) != expected:
            raise StrategyAttributionError("genotype attribution metric totals differ")
        if field in total_funnel and total_funnel[field] != expected:
            raise StrategyAttributionError(
                "genotype attribution total funnel differs from metrics"
            )
    descendant_counts = value.get(
        "descendant_birth_counts_by_parent_tracked_identity", {}
    )
    if (
        not isinstance(descendant_counts, Mapping)
        or list(descendant_counts) != sorted(descendant_counts)
        or any(
            not isinstance(identity, str)
            or len(identity) != 71
            or not identity.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in identity[7:])
            or type(count) is not int
            or count < 0
            for identity, count in descendant_counts.items()
        )
    ):
        raise StrategyAttributionError(
            "genotype descendant counts are invalid"
        )


def _strategy_contract(world: ScienceWorldRecordV1) -> dict[str, Any]:
    value = strategy_attribution_contract_v1(world)
    if value is not None:
        return value
    raise StrategyAttributionError(
        f"logical world {world.world_id} lacks strategy attribution configuration"
    )


def _control_row(world: ScienceWorldRecordV1, control_id: str) -> Mapping[str, Any] | None:
    semantic = world.semantic_inputs.to_value()
    for row in semantic.get("control_bindings", []):
        if isinstance(row, Mapping) and row.get("control_id") == control_id:
            return row
    return None


def _pressure_exposure(world: ScienceWorldRecordV1) -> dict[str, Any]:
    row = _control_row(world, "selection.pressure")
    if row is None:
        return {
            "control_id": "selection.pressure",
            "onset_absolute_tick": None,
            "maximum_value": 0.0,
            "schedule": None,
        }
    schedule = row.get("schedule")
    resolved = row.get("resolved_value")
    segments: list[Mapping[str, Any]] = []
    if isinstance(schedule, Mapping) and schedule.get("kind") == "piecewise_constant_absolute_ticks":
        raw_segments = schedule.get("segments")
        if not isinstance(raw_segments, list):
            raise StrategyAttributionError("pressure schedule segments are invalid")
        segments = [segment for segment in raw_segments if isinstance(segment, Mapping)]
    else:
        segments = [{"start_absolute_tick": 0, "value": resolved}]
    positive = [segment for segment in segments if isinstance(segment.get("value"), (int, float)) and segment.get("value") != 0]
    onset = min(
        (int(segment["start_absolute_tick"]) for segment in positive if type(segment.get("start_absolute_tick")) is int),
        default=None,
    )
    maximum = max(
        (float(segment["value"]) for segment in segments if isinstance(segment.get("value"), (int, float))),
        default=0.0,
    )
    return {
        "control_id": "selection.pressure",
        "onset_absolute_tick": onset,
        "maximum_value": maximum,
        "schedule": [dict(segment) for segment in segments],
    }


def _metric_row(raw: bytes, offset: int) -> dict[str, int]:
    values: dict[str, int] = {}
    cursor = offset
    for index, name in enumerate(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1):
        width = 8 if index == 17 else 4
        if cursor + width > len(raw):
            raise StrategyAttributionError("death metric row is truncated")
        values[name] = int(
            struct.unpack_from("<q" if index == 17 else "<I", raw, cursor)[0]
        )
        cursor += width
    return values


@dataclass(slots=True)
class _Agent:
    strategy_ids: tuple[str, ...]
    birth_tick: int
    lineage_depth: int
    founder: bool = False
    alive: bool = True
    death_tick: int | None = None
    death_metrics: dict[str, int] | None = None
    latest_metrics: dict[str, int] | None = None
    latest_energy: int | float | None = None
    latest_health: int | float | None = None
    latest_injury: int | float | None = None
    latest_lineage_id: str | None = None
    observed_policy_ids: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _WorldState:
    world: ScienceWorldRecordV1
    contract: dict[str, Any]
    agents: dict[int, _Agent] = field(default_factory=dict)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    births: int = 0
    deaths: int = 0
    first_birth_tick: int | None = None
    first_death_tick: int | None = None
    extinction_tick: int | None = None
    parent_birth_events: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    offspring_birth_events: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    parent_pair_events: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    mixed_parent_births: int = 0
    death_events_by_strategy: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    integrity: dict[str, int] = field(
        default_factory=lambda: {
            "dispatch_count": 0,
            "event_count": 0,
            "snapshot_count": 0,
            "dropped_record_count": 0,
        }
    )


def _world_states(bundle: ScienceWorldBundleV1) -> dict[str, _WorldState]:
    states: dict[str, _WorldState] = {}
    for world in bundle.worlds:
        if world.world_id in states:
            raise StrategyAttributionError("world IDs are not unique")
        states[world.world_id] = _WorldState(world, _strategy_contract(world))
    return states


def _validate_index(index: Mapping[str, Any], bundle: ScienceWorldBundleV1) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if index.get("schema_version") != "eiel_cuda_v2_compact_science_capture_index_v1":
        raise StrategyAttributionError("compact capture index schema differs")
    if index.get("dropped_record_count") != 0:
        raise StrategyAttributionError("compact capture index reports dropped rows")
    body = {key: value for key, value in index.items() if key not in {"index_sha256", "path"}}
    if index.get("index_sha256") != canonical_sha256(body).hex:
        raise StrategyAttributionError("compact capture index digest differs")
    expected = [world.world_id for world in bundle.worlds]
    if index.get("ordered_cell_ids") != expected:
        raise StrategyAttributionError("compact physical order differs from bundle")
    dispatches = index.get("dispatch_artifacts")
    snapshots = index.get("snapshot_artifacts")
    if (
        not isinstance(dispatches, list)
        or not isinstance(snapshots, list)
        or any(not isinstance(item, Mapping) for item in dispatches)
        or any(not isinstance(item, Mapping) for item in snapshots)
    ):
        raise StrategyAttributionError("compact artifact ownership is invalid")
    return list(dispatches), list(snapshots)


def _read_snapshot(pointer: Mapping[str, Any], states: Mapping[str, _WorldState]) -> int:
    root, manifest = _verify_artifact(pointer, label="compact snapshot")
    if (
        manifest.get("schema_version") != COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1
        or manifest.get("metric_fields") != list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
        or manifest.get("dropped_record_count") != 0
    ):
        raise StrategyAttributionError("compact snapshot ABI differs")
    name = manifest.get("data_file")
    expected = manifest.get("compressed_file_sha256")
    if not isinstance(name, str) or Path(name).name != name:
        raise StrategyAttributionError("snapshot data file name is invalid")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise StrategyAttributionError("snapshot data file is absent")
    compressed = path.read_bytes()
    if not isinstance(expected, str) or hashlib.sha256(compressed).hexdigest() != expected:
        raise StrategyAttributionError("snapshot compressed digest differs")
    absolute_tick = _as_int(manifest.get("absolute_tick"), label="snapshot tick")
    grouped: dict[str, dict[str, Any]] = {
        cell_id: {
            "world_state": None,
            "active_ids": set(),
            "energy": [],
            "health": [],
            "injury": [],
            "strategy_counts": defaultdict(int),
            "active_count": 0,
        }
        for cell_id in states
    }
    row_count = 0
    digest = hashlib.sha256()
    try:
        stream = gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb")
        while True:
            raw = stream.readline()
            if not raw:
                break
            digest.update(raw)
            row_count += 1
            try:
                row = json.loads(raw)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise StrategyAttributionError("snapshot JSON row is invalid") from exc
            if not isinstance(row, Mapping) or row.get("absolute_tick") != absolute_tick:
                raise StrategyAttributionError("snapshot row tick differs from manifest")
            cell_id = row.get("cell_id")
            if not isinstance(cell_id, str) or cell_id not in grouped:
                raise StrategyAttributionError("snapshot names an unknown world")
            group = grouped[cell_id]
            state = states[cell_id]
            kind = row.get("row_kind")
            if kind == "world_state":
                if group["world_state"] is not None:
                    raise StrategyAttributionError("snapshot duplicates world state")
                group["world_state"] = dict(row)
            elif kind == "active_agent_state":
                agent_id = row.get("agent_id")
                if type(agent_id) is not int or agent_id <= 0 or agent_id in group["active_ids"]:
                    raise StrategyAttributionError("snapshot agent identity differs")
                group["active_ids"].add(agent_id)
                group["active_count"] += 1
                metrics = row.get("lifetime_metrics")
                if not isinstance(metrics, Mapping) or any(type(metrics.get(name)) is not int for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1):
                    raise StrategyAttributionError("snapshot lifetime metrics are invalid")
                if absolute_tick == int(state.world.semantic_inputs.to_value()["time"].get("start_next_tick", 0)) and agent_id not in state.agents:
                    policy_id = row.get("policy_id")
                    mapping = state.contract["policy_id_to_strategy_id"]
                    if not isinstance(policy_id, str) or policy_id not in mapping:
                        raise StrategyAttributionError("founder policy lacks declared strategy identity")
                    if row.get("parent_a_id") != 0 or row.get("parent_b_id") != 0:
                        raise StrategyAttributionError("founder parentage is not null")
                    state.agents[agent_id] = _Agent(
                        strategy_ids=(str(mapping[policy_id]),),
                        birth_tick=absolute_tick,
                        lineage_depth=0,
                        founder=True,
                    )
                agent = state.agents.get(agent_id)
                if agent is None or not agent.alive:
                    raise StrategyAttributionError("snapshot names an inactive agent")
                policy_id = row.get("policy_id")
                if isinstance(policy_id, str):
                    agent.observed_policy_ids.add(policy_id)
                agent.latest_metrics = {name: int(metrics[name]) for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1}
                for attr, key in (("latest_energy", "energy"), ("latest_health", "health"), ("latest_injury", "injury")):
                    value = row.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        setattr(agent, attr, value)
                lineage_id = row.get("lineage_id")
                if isinstance(lineage_id, str):
                    agent.latest_lineage_id = lineage_id
                group["energy"].append(agent.latest_energy)
                group["health"].append(agent.latest_health)
                group["injury"].append(agent.latest_injury)
                for strategy in agent.strategy_ids:
                    group["strategy_counts"][strategy] += 1
            elif kind == "genome_catalog":
                continue
            else:
                raise StrategyAttributionError("snapshot row kind is unsupported")
        stream.close()
    except OSError as exc:
        raise StrategyAttributionError("snapshot compression is invalid") from exc
    if row_count != manifest.get("row_count"):
        raise StrategyAttributionError("snapshot row count differs")
    expected_uncompressed = manifest.get("uncompressed_content_sha256")
    if not isinstance(expected_uncompressed, str) or digest.hexdigest() != expected_uncompressed:
        raise StrategyAttributionError("snapshot uncompressed digest differs")
    captured = manifest.get("captured_worlds")
    captured_by_cell = {
        str(row["cell_id"]): row
        for row in captured
        if isinstance(row, Mapping) and isinstance(row.get("cell_id"), str)
    } if isinstance(captured, list) else {}
    if len(captured_by_cell) != len(states):
        raise StrategyAttributionError("snapshot captured-world ownership differs")
    for cell_id, group in grouped.items():
        world_state = group["world_state"]
        if world_state is None:
            raise StrategyAttributionError("snapshot lacks world state")
        active_count = group["active_count"]
        state = states[cell_id]
        owner = captured_by_cell[cell_id]
        if (
            owner.get("active_agent_count") != active_count
            or owner.get("science_world_sha256")
            != state.world.semantic_world_sha256.to_dict()
            or owner.get("science_acquisition_sha256")
            != state.world.acquisition_sha256.to_dict()
        ):
            raise StrategyAttributionError("snapshot active count differs")
        active_ids = {agent_id for agent_id, agent in state.agents.items() if agent.alive}
        if active_ids != group["active_ids"]:
            raise StrategyAttributionError("snapshot active ownership differs")
        fields = world_state.get("qualified_field_values")
        if not isinstance(fields, Mapping):
            raise StrategyAttributionError("snapshot world fields are absent")
        def _stats(values: Sequence[object]) -> dict[str, float | int | None]:
            numeric = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
            return {
                "mean": (sum(numeric) / len(numeric)) if numeric else None,
                "min": min(numeric) if numeric else None,
                "max": max(numeric) if numeric else None,
            }
        state.snapshots.append({
            "absolute_tick": absolute_tick,
            "active_count": active_count,
            "population_count": fields.get("population_count", active_count),
            "birth_count": fields.get("birth_count"),
            "death_count": fields.get("death_count"),
            "mutation_count": fields.get("mutation_count"),
            "resource_total": fields.get("resource_total"),
            "resource_mean": fields.get("resource_mean"),
            "resource_min": fields.get("resource_min"),
            "resource_max": fields.get("resource_max"),
            "resource_nonzero_count": fields.get("resource_nonzero_count"),
            "energy": _stats(group["energy"]),
            "health": _stats(group["health"]),
            "injury": _stats(group["injury"]),
            "active_by_strategy": dict(sorted(group["strategy_counts"].items())),
        })
        state.integrity["snapshot_count"] += 1
    return absolute_tick


def _read_dispatch(pointer: Mapping[str, Any], states: Mapping[str, _WorldState]) -> tuple[int, int]:
    root, manifest = _verify_artifact(pointer, label="compact dispatch")
    if (
        manifest.get("schema_version") != COMPACT_SCIENCE_DISPATCH_SCHEMA_V1
        or manifest.get("metric_fields") != list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
        or manifest.get("metric_storage_encoding") != COMPACT_SCIENCE_DEATH_METRIC_ENCODING_V1
        or manifest.get("dropped_record_count") != 0
    ):
        raise StrategyAttributionError("compact dispatch ABI differs")
    events = _payload(root, manifest, prefix="event", label="sparse lifecycle events")
    metrics = _payload(root, manifest, prefix="metric", label="death metrics")
    provenance = _payload(root, manifest, prefix="birth_provenance", label="birth provenance")
    event_count = _as_int(manifest.get("event_count"), label="dispatch event count")
    birth_count = _as_int(manifest.get("birth_event_count"), label="dispatch birth count")
    death_count = _as_int(manifest.get("death_event_count"), label="dispatch death count")
    if birth_count + death_count != event_count or len(events) != event_count * _EVENT_ROW_STRUCT.size:
        raise StrategyAttributionError("compact dispatch event row count differs")
    if len(metrics) != death_count * COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1:
        raise StrategyAttributionError("compact dispatch death metric count differs")
    if len(provenance) != birth_count * (COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1):
        raise StrategyAttributionError("compact dispatch birth provenance count differs")
    start = _as_int(manifest.get("start_next_absolute_tick"), label="dispatch start tick")
    end = _as_int(manifest.get("end_next_absolute_tick"), label="dispatch end tick")
    if end <= start or end - start != _as_int(manifest.get("tick_count"), label="dispatch tick count"):
        raise StrategyAttributionError("compact dispatch tick span differs")
    owners = manifest.get("logical_worlds")
    offsets = manifest.get("world_offsets")
    if not isinstance(owners, list) or not isinstance(offsets, list) or len(offsets) != len(owners) + 1 or offsets[0] != 0 or offsets[-1] != event_count:
        raise StrategyAttributionError("compact dispatch world offsets differ")
    if any(
        type(offset) is not int
        or offset < 0
        or (index and offset < offsets[index - 1])
        for index, offset in enumerate(offsets)
    ):
        raise StrategyAttributionError("compact dispatch offsets are invalid")
    birth_cursor = 0
    death_cursor = 0
    for index, owner in enumerate(owners):
        if not isinstance(owner, Mapping) or not isinstance(owner.get("cell_id"), str) or owner["cell_id"] not in states:
            raise StrategyAttributionError("dispatch names an unknown world")
        cell_id = str(owner["cell_id"])
        state = states[cell_id]
        if (
            owner.get("science_world_sha256")
            != state.world.semantic_world_sha256.to_dict()
            or owner.get("science_acquisition_sha256")
            != state.world.acquisition_sha256.to_dict()
        ):
            raise StrategyAttributionError("dispatch logical-world identity differs")
        first, stop = int(offsets[index]), int(offsets[index + 1])
        world_births = world_deaths = 0
        prior_tick = start - 1
        for row_index in range(first, stop):
            agent_id, parent_a, parent_b, packed = _EVENT_ROW_STRUCT.unpack_from(events, row_index * _EVENT_ROW_STRUCT.size)
            event_kind = (packed >> 6) & 0x3
            tick = start + (packed & 0x3F)
            if tick < prior_tick or not start <= tick < end:
                raise StrategyAttributionError("compact event order or tick differs")
            prior_tick = tick
            if event_kind == 1:
                agent = state.agents.get(agent_id)
                if agent is None or not agent.alive:
                    raise StrategyAttributionError("death names an inactive agent")
                agent.alive = False
                agent.death_tick = tick
                agent.death_metrics = _metric_row(metrics, death_cursor * COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1)
                death_cursor += 1
                world_deaths += 1
                state.deaths += 1
                state.integrity["event_count"] += 1
                state.death_events_by_strategy.update({strategy: 1 for strategy in agent.strategy_ids})
                if state.first_death_tick is None:
                    state.first_death_tick = tick
                if state.extinction_tick is None and not any(item.alive for item in state.agents.values()):
                    state.extinction_tick = tick
            elif event_kind == 2:
                if not agent_id or not parent_a or not parent_b:
                    raise StrategyAttributionError("birth parentage is incomplete")
                if agent_id in state.agents or parent_a not in state.agents or parent_b not in state.agents:
                    raise StrategyAttributionError("birth ancestry is not active")
                first_parent = state.agents[parent_a]
                second_parent = state.agents[parent_b]
                if not first_parent.alive or not second_parent.alive:
                    raise StrategyAttributionError("birth parent is inactive")
                child_strategy = strategy_union_v1(first_parent.strategy_ids, second_parent.strategy_ids)
                child = _Agent(
                    strategy_ids=child_strategy,
                    birth_tick=tick,
                    lineage_depth=max(first_parent.lineage_depth, second_parent.lineage_depth) + 1,
                )
                state.agents[agent_id] = child
                state.births += 1
                state.integrity["event_count"] += 1
                world_births += 1
                if state.first_birth_tick is None:
                    state.first_birth_tick = tick
                for strategy in first_parent.strategy_ids:
                    state.parent_birth_events[strategy] += 1
                for strategy in second_parent.strategy_ids:
                    state.parent_birth_events[strategy] += 1
                child_group = strategy_group_id_v1(child.strategy_ids)
                state.offspring_birth_events[child_group] += 1
                parent_group = "+".join(sorted((strategy_group_id_v1(first_parent.strategy_ids), strategy_group_id_v1(second_parent.strategy_ids))))
                state.parent_pair_events[parent_group] += 1
                if first_parent.strategy_ids != second_parent.strategy_ids:
                    state.mixed_parent_births += 1
                birth_cursor += 1
            else:
                raise StrategyAttributionError("unsupported compact lifecycle event kind")
        if world_births + world_deaths != stop - first:
            raise StrategyAttributionError("compact event auxiliary cursors differ")
        state.integrity["dispatch_count"] += 1
        state.integrity["dispatch_tick_count"] = state.integrity.get("dispatch_tick_count", 0) + (end - start)
    if birth_cursor != birth_count or death_cursor != death_count:
        raise StrategyAttributionError("compact dispatch auxiliary cursors incomplete")
    return start, end


def _sum_metrics(state: _WorldState) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    agent_counts: dict[str, int] = defaultdict(int)
    for agent in state.agents.values():
        metrics = agent.latest_metrics if agent.alive else agent.death_metrics
        if metrics is None:
            raise StrategyAttributionError("agent lacks terminal lifetime metrics")
        for strategy in agent.strategy_ids:
            agent_counts[strategy] += 1
            for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1:
                totals[strategy][name] += int(metrics[name])
    return {key: dict(sorted(value.items())) for key, value in sorted(totals.items())}, dict(sorted(agent_counts.items()))


def _offspring_survival(state: _WorldState, terminal: int, early_life_tick: int) -> dict[str, Any]:
    thresholds = sorted({tick for tick in (*[100, 250, 500, 1000, 2500, 5000, 7500, early_life_tick], terminal) if 0 <= tick <= terminal})
    by_strategy: dict[str, dict[str, Any]] = defaultdict(lambda: {"born": 0, "survived_to": defaultdict(int), "lineage_depth_max": 0})
    for agent in state.agents.values():
        if agent.founder:
            continue
        for strategy in agent.strategy_ids:
            row = by_strategy[strategy]
            row["born"] += 1
            row["lineage_depth_max"] = max(row["lineage_depth_max"], agent.lineage_depth)
            for tick in thresholds:
                if agent.death_tick is None or agent.death_tick > tick:
                    row["survived_to"][str(tick)] += 1
    return {
        "threshold_ticks": thresholds,
        "by_strategy": {
            strategy: {
                "born": row["born"],
                "survived_to": dict(sorted(row["survived_to"].items())),
                "lineage_depth_max": row["lineage_depth_max"],
            }
            for strategy, row in sorted(by_strategy.items())
        },
    }


def _early_active(state: _WorldState, tick: int) -> int:
    candidates = [row for row in state.snapshots if int(row["absolute_tick"]) <= tick]
    return int(candidates[-1]["active_count"]) if candidates else 0


def _initially_mixed(state: _WorldState) -> bool:
    initialization = state.world.semantic_inputs.to_value().get("initialization", {})
    composition = initialization.get("exact_founder_composition")
    if isinstance(composition, list) and len(composition) > 1:
        return True
    return len({strategy for agent in state.agents.values() if agent.founder for strategy in agent.strategy_ids}) > 1


def classify_causal_class_v1(
    result: Mapping[str, Any],
    *,
    early_life_tick: int = 10_000,
    pressure_control_survives: bool | None = None,
) -> dict[str, Any]:
    """Apply the preregistered A--E decision rules without tuning a result."""

    first_death = result.get("time_to_first_death", {}).get("absolute_tick")
    first_birth = result.get("time_to_first_birth", {}).get("absolute_tick")
    extinction = result.get("extinction", {}).get("absolute_tick")
    survival_curve = result.get("survival_curve", [])
    early_rows = [
        row
        for row in survival_curve
        if isinstance(row, Mapping)
        and type(row.get("absolute_tick")) is int
        and int(row["absolute_tick"]) <= early_life_tick
    ]
    early_active = int(early_rows[-1].get("active_count", 0)) if early_rows else 0
    funnel = result.get("reproduction_funnel", {})
    totals = funnel.get("all_agents", {}) if isinstance(funnel, Mapping) else {}
    eligible = int(totals.get("reproduction_eligible_count", 0))
    candidates = int(totals.get("mate_candidate_count", 0))
    attempts = int(totals.get("birth_attempt_count", 0))
    births = int(result.get("event_counts", {}).get("birth", 0))
    mixed = bool(result.get("initial_composition", {}).get("mixed", False))
    pressure = result.get("pressure_exposure", {})
    onset = pressure.get("onset_absolute_tick")
    reasons: list[str] = []
    classification = "UNRESOLVED"
    if (
        first_death is not None
        and (first_birth is None or int(first_death) < int(first_birth))
        and eligible == 0
        and candidates == 0
        and attempts == 0
    ):
        classification = "A"
        reasons.append("death precedes any birth and all early reproduction denominators are zero")
    elif early_active > 0 and births == 0 and eligible == 0 and candidates == 0 and attempts == 0:
        classification = "B"
        reasons.append("lineages remain alive through the early-life horizon without eligibility or opportunity")
    elif (
        births > 0
        and early_active == 0
        and (onset is None or extinction is None or int(extinction) <= int(onset))
    ):
        classification = "C"
        reasons.append("births occur but no population remains through the early-life horizon outside pressure exposure")
    elif (
        onset is not None
        and extinction is not None
        and int(extinction) > int(onset)
        and pressure_control_survives is True
    ):
        classification = "D"
        reasons.append("a paired no-pressure control survives while the exposed lineage goes extinct after pressure onset")
    elif early_active > 0 and mixed and births > 0:
        classification = "E"
        reasons.append("mixed lineages persist with attributed reproduction; directional contrast requires a later preregistered test")
    else:
        reasons.append("available observations do not uniquely separate A-E")
    return {
        "class": classification,
        "early_life_tick": early_life_tick,
        "reasons": reasons,
        "pressure_control_survives": pressure_control_survives,
    }


def _world_result(state: _WorldState, *, terminal: int, early_life_tick: int) -> dict[str, Any]:
    totals, agent_counts = _sum_metrics(state)
    strategies = list(state.contract["strategy_ids"])
    all_totals = {name: 0 for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1}
    for values in totals.values():
        for name in all_totals:
            all_totals[name] += int(values.get(name, 0))
    pressure = _pressure_exposure(state.world)
    pressure_end = state.extinction_tick if state.extinction_tick is not None else terminal
    pressure["exposed_tick_span"] = (
        max(0, int(pressure_end) - int(pressure["onset_absolute_tick"]))
        if pressure["onset_absolute_tick"] is not None
        else 0
    )
    initial_mixed = _initially_mixed(state)
    initial_profiles = state.world.semantic_inputs.to_value().get("initialization", {}).get("exact_founder_composition", [])
    return {
        "world_id": state.world.world_id,
        "science": state.world.science.to_value(),
        "dimension_bindings": state.world.semantic_inputs.to_value().get("dimension_bindings", {}),
        "semantic_world_sha256": state.world.semantic_world_sha256.to_dict(),
        "acquisition_sha256": state.world.acquisition_sha256.to_dict(),
        "strategy_contract": {
            "schema_version": state.contract["schema_version"],
            "strategy_ids": strategies,
            "identity_mode": state.contract.get("identity_mode"),
            "lineage_mode": state.contract.get("lineage_mode"),
            "denominator_fields": list(state.contract["denominator_fields"]),
            "policy_mapping_count": len(state.contract["policy_id_to_strategy_id"]),
        },
        "initial_composition": {
            "mixed": initial_mixed,
            "founder_profiles": initial_profiles,
        },
        "event_counts": {
            "birth": state.births,
            "death": state.deaths,
            "lineage": len(state.agents),
        },
        "time_to_first_birth": {
            "absolute_tick": state.first_birth_tick,
            "value": None if state.first_birth_tick is None else state.first_birth_tick,
        },
        "time_to_first_death": {
            "absolute_tick": state.first_death_tick,
            "value": None if state.first_death_tick is None else state.first_death_tick,
        },
        "extinction": {
            "absolute_tick": state.extinction_tick,
            "survives_terminal": state.extinction_tick is None,
        },
        "survival_curve": sorted(state.snapshots, key=lambda row: int(row["absolute_tick"])),
        "reproduction_funnel": {
            "all_agents": all_totals,
            "by_strategy": totals,
            "agent_counts_by_strategy": agent_counts,
            "parent_birth_events_by_strategy": dict(sorted(state.parent_birth_events.items())),
            "offspring_birth_events_by_strategy": dict(sorted(state.offspring_birth_events.items())),
            "parent_pair_events": dict(sorted(state.parent_pair_events.items())),
            "mixed_parent_birth_count": state.mixed_parent_births,
            "assignment_mode": "parent_ancestry_union_v1",
        },
        "offspring_survival": _offspring_survival(state, terminal, early_life_tick),
        "lineage": {
            "agent_count": len(state.agents),
            "founder_count": sum(int(agent.founder) for agent in state.agents.values()),
            "maximum_depth": max((agent.lineage_depth for agent in state.agents.values()), default=0),
            "maximum_depth_by_strategy": {
                strategy: max((agent.lineage_depth for agent in state.agents.values() if strategy in agent.strategy_ids), default=0)
                for strategy in strategies
            },
            "terminal_alive_by_strategy": {
                strategy: sum(int(agent.alive and strategy in agent.strategy_ids) for agent in state.agents.values())
                for strategy in strategies
            },
            "persistence_to_early_life_by_strategy": {
                strategy: sum(
                    int(
                        agent.birth_tick <= early_life_tick
                        and (agent.death_tick is None or agent.death_tick > early_life_tick)
                        and strategy in agent.strategy_ids
                    )
                    for agent in state.agents.values()
                )
                for strategy in strategies
            },
        },
        "pressure_exposure": pressure,
        "integrity": {
            **state.integrity,
            "terminal_next_absolute_tick": terminal,
            "all_agents_have_terminal_metrics": True,
            "strategy_identity_exact": True,
        },
    }


def analyze_compact_science_strategy_attribution_v1(
    capture_index: str | Path | Mapping[str, Any],
    world_bundle: str | Path | ScienceWorldBundleV1,
    *,
    early_life_tick: int = 10_000,
    pressure_factor: str = "pressure_regime",
) -> dict[str, Any]:
    """Reduce a sealed compact capture into generic strategy/viability facts."""

    if type(early_life_tick) is not int or early_life_tick < 1:
        raise StrategyAttributionError("early_life_tick must be positive")
    bundle = _load_bundle(world_bundle)
    index = _load_index(capture_index)
    dispatches, snapshots = _validate_index(index, bundle)
    states = _world_states(bundle)
    ordered_snapshots = sorted(snapshots, key=lambda pointer: _as_int(_verify_artifact(pointer, label="snapshot")[1].get("absolute_tick"), label="snapshot tick"))
    expected_starts = {
        int(world.semantic_inputs.to_value()["time"].get("start_next_tick", 0))
        for world in bundle.worlds
    }
    if len(expected_starts) != 1:
        raise StrategyAttributionError("strategy attribution requires one absolute start frontier")
    start_tick = expected_starts.pop()
    if not ordered_snapshots or _read_snapshot(ordered_snapshots[0], states) != start_tick:
        raise StrategyAttributionError("strategy attribution requires the absolute start snapshot")
    ordered_dispatches = sorted(dispatches, key=lambda pointer: _as_int(_verify_artifact(pointer, label="dispatch")[1].get("start_next_absolute_tick"), label="dispatch start tick"))
    snapshot_cursor = 1
    for pointer in ordered_dispatches:
        start, end = _read_dispatch(pointer, states)
        while snapshot_cursor < len(ordered_snapshots):
            _, manifest = _verify_artifact(ordered_snapshots[snapshot_cursor], label="snapshot")
            tick = _as_int(manifest.get("absolute_tick"), label="snapshot tick")
            if tick > end:
                break
            _read_snapshot(ordered_snapshots[snapshot_cursor], states)
            snapshot_cursor += 1
        for state in states.values():
            if state.snapshots and int(state.snapshots[-1]["absolute_tick"]) == end:
                state.integrity["dispatch_tick_count"] = state.integrity.get("dispatch_tick_count", 0)
    while snapshot_cursor < len(ordered_snapshots):
        _read_snapshot(ordered_snapshots[snapshot_cursor], states)
        snapshot_cursor += 1
    terminal = _as_int(index.get("terminal_next_absolute_tick"), label="terminal tick")
    if snapshot_cursor != len(ordered_snapshots):
        raise StrategyAttributionError("snapshot ownership is incomplete")
    results = [_world_result(states[world.world_id], terminal=terminal, early_life_tick=early_life_tick) for world in bundle.worlds]
    pressure_controls: dict[tuple[tuple[str, Any], ...], bool] = {}
    for result in results:
        dimensions = result.get("dimension_bindings", {})
        key = tuple(sorted((str(name), value) for name, value in dimensions.items() if name not in {pressure_factor, "factorial_index"}))
        if dimensions.get(pressure_factor) in {"off", "none", "disabled"}:
            pressure_controls[key] = _early_active(states[result["world_id"]], early_life_tick) > 0
    for result in results:
        dimensions = result.get("dimension_bindings", {})
        key = tuple(sorted((str(name), value) for name, value in dimensions.items() if name not in {pressure_factor, "factorial_index"}))
        control = pressure_controls.get(key)
        result["causal_classification"] = classify_causal_class_v1(
            result,
            early_life_tick=early_life_tick,
            pressure_control_survives=control,
        )
    body = {
        "schema_version": STRATEGY_ATTRIBUTION_RESULT_SCHEMA_V1,
        "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
        "capture_index_sha256": index["index_sha256"],
        "logical_world_count": len(results),
        "physical_batch_count": 1,
        "physical_composition_used": False,
        "early_life_tick": early_life_tick,
        "pressure_pairing_factor": pressure_factor,
        "world_results": results,
        "classification_counts": {
            value: sum(int(result["causal_classification"]["class"] == value) for result in results)
            for value in ("A", "B", "C", "D", "E", "UNRESOLVED")
        },
    }
    return {**body, "result_sha256": canonical_sha256(body).to_dict()}


# The first implementation above intentionally mirrors the reference reducer
# and retains every historical agent.  That is useful for small fixtures but
# makes a large, birth-heavy capture needlessly expensive: terminal death
# metrics and threshold survival can be accumulated when a death is observed.
# The streaming implementation below keeps only active ancestry plus those
# exact aggregates.  It is part of the generic instrument contract; no study
# dimensions or founder names are involved.


@dataclass(slots=True)
class _StreamingAgent:
    strategy_ids: tuple[str, ...]
    birth_tick: int
    lineage_depth: int
    founder: bool = False
    latest_metrics: dict[str, int] | None = None
    latest_energy: int | float | None = None
    latest_health: int | float | None = None
    latest_injury: int | float | None = None


@dataclass(slots=True)
class _StreamingWorldState:
    world: ScienceWorldRecordV1
    contract: dict[str, Any]
    start_tick: int
    active: dict[int, _StreamingAgent] = field(default_factory=dict)
    seen_ids: set[int] = field(default_factory=set)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    metric_totals: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    agent_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    offspring_born: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    offspring_survived: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    max_depth: int = 0
    max_depth_by_strategy: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    total_agents: int = 0
    founder_count: int = 0
    births: int = 0
    deaths: int = 0
    first_birth_tick: int | None = None
    first_death_tick: int | None = None
    extinction_tick: int | None = None
    parent_birth_events: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    offspring_birth_events: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    parent_pair_events: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    mixed_parent_births: int = 0
    death_events_by_strategy: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    persistence_to_early: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    integrity: dict[str, int] = field(
        default_factory=lambda: {
            "dispatch_count": 0,
            "event_count": 0,
            "snapshot_count": 0,
            "dropped_record_count": 0,
        }
    )


def strategy_threshold_ticks_v1(
    terminal: int, early_life_tick: int
) -> tuple[int, ...]:
    if (
        type(terminal) is not int
        or terminal < 0
        or type(early_life_tick) is not int
        or early_life_tick < 1
    ):
        raise StrategyAttributionError("strategy threshold horizon is invalid")
    return tuple(
        sorted(
            {
                tick
                for tick in (
                    100,
                    250,
                    500,
                    1000,
                    2500,
                    5000,
                    7500,
                    early_life_tick,
                    terminal,
                )
                if 0 <= tick <= terminal
            }
        )
    )


def _stream_thresholds(terminal: int, early_life_tick: int) -> tuple[int, ...]:
    return strategy_threshold_ticks_v1(terminal, early_life_tick)


def _stream_add_terminal_metrics(
    state: _StreamingWorldState,
    agent: _StreamingAgent,
    metrics: Mapping[str, int],
) -> None:
    for strategy in agent.strategy_ids:
        totals = state.metric_totals[strategy]
        for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1:
            totals[name] += int(metrics[name])


def _stream_add_offspring_survival(
    state: _StreamingWorldState,
    agent: _StreamingAgent,
    *,
    death_tick: int | None,
    thresholds: Sequence[int],
    early_life_tick: int,
) -> None:
    # Persistence-to-early-life includes both founders and descendants.  The
    # separate offspring-survival denominator below intentionally excludes
    # founders, matching the reference reducer's offspring contract.
    for strategy in agent.strategy_ids:
        if agent.birth_tick <= early_life_tick and (
            death_tick is None or death_tick > early_life_tick
        ):
            state.persistence_to_early[strategy] += 1
        if agent.founder:
            continue
        for tick in thresholds:
            # Preserve the reference reducer's survival denominator semantics:
            # a recorded lineage is counted as surviving a threshold when its
            # death is after that threshold (or it is alive at terminal).
            if death_tick is None or death_tick > tick:
                state.offspring_survived[strategy][str(tick)] += 1


def _stream_stats(values: Sequence[object]) -> dict[str, float | int | None]:
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


def _stream_read_snapshot(
    pointer: Mapping[str, Any],
    states: Mapping[str, _StreamingWorldState],
) -> int:
    root, manifest = _verify_artifact(pointer, label="compact snapshot")
    if (
        manifest.get("schema_version") != COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1
        or manifest.get("metric_fields")
        != list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
        or manifest.get("dropped_record_count") != 0
    ):
        raise StrategyAttributionError("compact snapshot ABI differs")
    name = manifest.get("data_file")
    expected = manifest.get("compressed_file_sha256")
    if not isinstance(name, str) or Path(name).name != name:
        raise StrategyAttributionError("snapshot data file name is invalid")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise StrategyAttributionError("snapshot data file is absent")
    compressed = path.read_bytes()
    if not isinstance(expected, str) or hashlib.sha256(compressed).hexdigest() != expected:
        raise StrategyAttributionError("snapshot compressed digest differs")
    absolute_tick = _as_int(manifest.get("absolute_tick"), label="snapshot tick")
    captured = manifest.get("captured_worlds")
    if (
        not isinstance(captured, list)
        or not captured
        or manifest.get("captured_world_count") != len(captured)
        or any(not isinstance(row, Mapping) for row in captured)
    ):
        raise StrategyAttributionError("snapshot captured-world ownership differs")
    captured_by_cell = {
        str(row["cell_id"]): row
        for row in captured
        if isinstance(row.get("cell_id"), str)
    }
    if (
        len(captured_by_cell) != len(captured)
        or not set(captured_by_cell).issubset(states)
    ):
        raise StrategyAttributionError("snapshot captured-world ownership differs")
    grouped: dict[str, dict[str, Any]] = {
        cell_id: {
            "world_state": None,
            "active_ids": set(),
            "energy": [],
            "health": [],
            "injury": [],
            "strategy_counts": defaultdict(int),
            "active_count": 0,
        }
        for cell_id in captured_by_cell
    }
    row_count = 0
    digest = hashlib.sha256()
    try:
        stream = gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb")
        while True:
            raw = stream.readline()
            if not raw:
                break
            digest.update(raw)
            row_count += 1
            try:
                row = json.loads(raw)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise StrategyAttributionError("snapshot JSON row is invalid") from exc
            if not isinstance(row, Mapping) or row.get("absolute_tick") != absolute_tick:
                raise StrategyAttributionError("snapshot row tick differs from manifest")
            cell_id = row.get("cell_id")
            if not isinstance(cell_id, str) or cell_id not in grouped:
                raise StrategyAttributionError("snapshot names an unknown world")
            group = grouped[cell_id]
            state = states[cell_id]
            kind = row.get("row_kind")
            if kind == "world_state":
                if group["world_state"] is not None:
                    raise StrategyAttributionError("snapshot duplicates world state")
                group["world_state"] = dict(row)
            elif kind == "active_agent_state":
                agent_id = row.get("agent_id")
                if (
                    type(agent_id) is not int
                    or agent_id <= 0
                    or agent_id in group["active_ids"]
                ):
                    raise StrategyAttributionError("snapshot agent identity differs")
                group["active_ids"].add(agent_id)
                group["active_count"] += 1
                metrics = row.get("lifetime_metrics")
                if not isinstance(metrics, Mapping) or any(
                    type(metrics.get(name)) is not int
                    for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
                ):
                    raise StrategyAttributionError("snapshot lifetime metrics are invalid")
                if absolute_tick == state.start_tick and agent_id not in state.active:
                    policy_id = row.get("policy_id")
                    mapping = state.contract["policy_id_to_strategy_id"]
                    if not isinstance(policy_id, str) or policy_id not in mapping:
                        raise StrategyAttributionError(
                            "founder policy lacks declared strategy identity"
                        )
                    if row.get("parent_a_id") != 0 or row.get("parent_b_id") != 0:
                        raise StrategyAttributionError("founder parentage is not null")
                    strategy_ids = (str(mapping[policy_id]),)
                    agent = _StreamingAgent(
                        strategy_ids=strategy_ids,
                        birth_tick=absolute_tick,
                        lineage_depth=0,
                        founder=True,
                    )
                    state.active[agent_id] = agent
                    state.seen_ids.add(agent_id)
                    state.total_agents += 1
                    state.founder_count += 1
                    for strategy in strategy_ids:
                        state.agent_counts[strategy] += 1
                agent = state.active.get(agent_id)
                if agent is None:
                    raise StrategyAttributionError("snapshot names an inactive agent")
                agent.latest_metrics = {
                    name: int(metrics[name])
                    for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
                }
                for attr, key in (
                    ("latest_energy", "energy"),
                    ("latest_health", "health"),
                    ("latest_injury", "injury"),
                ):
                    value = row.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        setattr(agent, attr, value)
                for strategy in agent.strategy_ids:
                    group["strategy_counts"][strategy] += 1
                group["energy"].append(agent.latest_energy)
                group["health"].append(agent.latest_health)
                group["injury"].append(agent.latest_injury)
            elif kind == "genome_catalog":
                continue
            else:
                raise StrategyAttributionError("snapshot row kind is unsupported")
        stream.close()
    except OSError as exc:
        raise StrategyAttributionError("snapshot compression is invalid") from exc
    if row_count != manifest.get("row_count"):
        raise StrategyAttributionError("snapshot row count differs")
    expected_uncompressed = manifest.get("uncompressed_content_sha256")
    if not isinstance(expected_uncompressed, str) or digest.hexdigest() != expected_uncompressed:
        raise StrategyAttributionError("snapshot uncompressed digest differs")
    for cell_id, group in grouped.items():
        world_state = group["world_state"]
        if world_state is None:
            raise StrategyAttributionError("snapshot lacks world state")
        state = states[cell_id]
        owner = captured_by_cell[cell_id]
        if (
            owner.get("active_agent_count") != group["active_count"]
            or owner.get("science_world_sha256")
            != state.world.semantic_world_sha256.to_dict()
            or owner.get("science_acquisition_sha256")
            != state.world.acquisition_sha256.to_dict()
        ):
            raise StrategyAttributionError("snapshot active count differs")
        if set(state.active) != group["active_ids"]:
            raise StrategyAttributionError("snapshot active ownership differs")
        fields = world_state.get("qualified_field_values")
        if not isinstance(fields, Mapping):
            raise StrategyAttributionError("snapshot world fields are absent")
        state.snapshots.append(
            {
                "absolute_tick": absolute_tick,
                "active_count": int(group["active_count"]),
                "population_count": fields.get("population_count", group["active_count"]),
                "birth_count": fields.get("birth_count"),
                "death_count": fields.get("death_count"),
                "mutation_count": fields.get("mutation_count"),
                "resource_total": fields.get("resource_total"),
                "resource_mean": fields.get("resource_mean"),
                "resource_min": fields.get("resource_min"),
                "resource_max": fields.get("resource_max"),
                "resource_nonzero_count": fields.get("resource_nonzero_count"),
                "energy": _stream_stats(group["energy"]),
                "health": _stream_stats(group["health"]),
                "injury": _stream_stats(group["injury"]),
                "active_by_strategy": dict(sorted(group["strategy_counts"].items())),
            }
        )
        state.integrity["snapshot_count"] += 1
    return absolute_tick


def _stream_read_dispatch(
    pointer: Mapping[str, Any],
    states: Mapping[str, _StreamingWorldState],
    *,
    thresholds: Sequence[int],
    early_life_tick: int,
) -> tuple[int, int]:
    root, manifest = _verify_artifact(pointer, label="compact dispatch")
    if (
        manifest.get("schema_version") != COMPACT_SCIENCE_DISPATCH_SCHEMA_V1
        or manifest.get("metric_fields")
        != list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
        or manifest.get("metric_storage_encoding")
        != COMPACT_SCIENCE_DEATH_METRIC_ENCODING_V1
        or manifest.get("dropped_record_count") != 0
    ):
        raise StrategyAttributionError("compact dispatch ABI differs")
    events = _payload(root, manifest, prefix="event", label="sparse lifecycle events")
    metrics = _payload(root, manifest, prefix="metric", label="death metrics")
    provenance = _payload(
        root, manifest, prefix="birth_provenance", label="birth provenance"
    )
    event_count = _as_int(manifest.get("event_count"), label="dispatch event count")
    birth_count = _as_int(manifest.get("birth_event_count"), label="dispatch birth count")
    death_count = _as_int(manifest.get("death_event_count"), label="dispatch death count")
    if birth_count + death_count != event_count or len(events) != event_count * _EVENT_ROW_STRUCT.size:
        raise StrategyAttributionError("compact dispatch event row count differs")
    if len(metrics) != death_count * COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1:
        raise StrategyAttributionError("compact dispatch death metric count differs")
    if len(provenance) != birth_count * COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1:
        raise StrategyAttributionError("compact dispatch birth provenance count differs")
    start = _as_int(manifest.get("start_next_absolute_tick"), label="dispatch start tick")
    end = _as_int(manifest.get("end_next_absolute_tick"), label="dispatch end tick")
    if end <= start or end - start != _as_int(manifest.get("tick_count"), label="dispatch tick count"):
        raise StrategyAttributionError("compact dispatch tick span differs")
    owners = manifest.get("logical_worlds")
    offsets = manifest.get("world_offsets")
    if (
        not isinstance(owners, list)
        or not isinstance(offsets, list)
        or len(offsets) != len(owners) + 1
        or offsets[0] != 0
        or offsets[-1] != event_count
    ):
        raise StrategyAttributionError("compact dispatch world offsets differ")
    if any(
        type(offset) is not int
        or offset < 0
        or (index and offset < offsets[index - 1])
        for index, offset in enumerate(offsets)
    ):
        raise StrategyAttributionError("compact dispatch offsets are invalid")
    birth_cursor = death_cursor = 0
    for index, owner in enumerate(owners):
        if (
            not isinstance(owner, Mapping)
            or not isinstance(owner.get("cell_id"), str)
            or owner["cell_id"] not in states
        ):
            raise StrategyAttributionError("dispatch names an unknown world")
        cell_id = str(owner["cell_id"])
        state = states[cell_id]
        if (
            owner.get("science_world_sha256")
            != state.world.semantic_world_sha256.to_dict()
            or owner.get("science_acquisition_sha256")
            != state.world.acquisition_sha256.to_dict()
        ):
            raise StrategyAttributionError("dispatch logical-world identity differs")
        first, stop = int(offsets[index]), int(offsets[index + 1])
        world_births = world_deaths = 0
        prior_tick = start - 1
        for row_index in range(first, stop):
            agent_id, parent_a, parent_b, packed = _EVENT_ROW_STRUCT.unpack_from(
                events, row_index * _EVENT_ROW_STRUCT.size
            )
            event_kind = (packed >> 6) & 0x3
            tick = start + (packed & 0x3F)
            if tick < prior_tick or not start <= tick < end:
                raise StrategyAttributionError("compact event order or tick differs")
            prior_tick = tick
            if event_kind == 1:
                agent = state.active.pop(agent_id, None)
                if agent is None:
                    raise StrategyAttributionError("death names an inactive agent")
                death_metrics = _metric_row(
                    metrics,
                    death_cursor * COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1,
                )
                death_cursor += 1
                _stream_add_terminal_metrics(state, agent, death_metrics)
                _stream_add_offspring_survival(
                    state,
                    agent,
                    death_tick=tick,
                    thresholds=thresholds,
                    early_life_tick=early_life_tick,
                )
                for strategy in agent.strategy_ids:
                    state.death_events_by_strategy[strategy] += 1
                state.deaths += 1
                world_deaths += 1
                state.integrity["event_count"] += 1
                if state.first_death_tick is None:
                    state.first_death_tick = tick
                if state.extinction_tick is None and not state.active:
                    state.extinction_tick = tick
            elif event_kind == 2:
                if not agent_id or not parent_a or not parent_b:
                    raise StrategyAttributionError("birth parentage is incomplete")
                if (
                    agent_id in state.seen_ids
                    or parent_a not in state.active
                    or parent_b not in state.active
                ):
                    raise StrategyAttributionError("birth ancestry is not active")
                first_parent = state.active[parent_a]
                second_parent = state.active[parent_b]
                child_strategy = strategy_union_v1(
                    first_parent.strategy_ids, second_parent.strategy_ids
                )
                child = _StreamingAgent(
                    strategy_ids=child_strategy,
                    birth_tick=tick,
                    lineage_depth=max(
                        first_parent.lineage_depth, second_parent.lineage_depth
                    )
                    + 1,
                )
                state.active[agent_id] = child
                state.seen_ids.add(agent_id)
                state.total_agents += 1
                state.births += 1
                world_births += 1
                state.integrity["event_count"] += 1
                if state.first_birth_tick is None:
                    state.first_birth_tick = tick
                state.max_depth = max(state.max_depth, child.lineage_depth)
                child_group = strategy_group_id_v1(child.strategy_ids)
                state.offspring_birth_events[child_group] += 1
                for strategy in child.strategy_ids:
                    state.agent_counts[strategy] += 1
                    state.offspring_born[strategy] += 1
                    state.max_depth_by_strategy[strategy] = max(
                        state.max_depth_by_strategy[strategy], child.lineage_depth
                    )
                for strategy in first_parent.strategy_ids:
                    state.parent_birth_events[strategy] += 1
                for strategy in second_parent.strategy_ids:
                    state.parent_birth_events[strategy] += 1
                parent_group = "+".join(
                    sorted(
                        (
                            strategy_group_id_v1(first_parent.strategy_ids),
                            strategy_group_id_v1(second_parent.strategy_ids),
                        )
                    )
                )
                state.parent_pair_events[parent_group] += 1
                if first_parent.strategy_ids != second_parent.strategy_ids:
                    state.mixed_parent_births += 1
                birth_cursor += 1
            else:
                raise StrategyAttributionError("unsupported compact lifecycle event kind")
        if world_births + world_deaths != stop - first:
            raise StrategyAttributionError("compact event auxiliary cursors differ")
        state.integrity["dispatch_count"] += 1
        state.integrity["dispatch_tick_count"] = state.integrity.get(
            "dispatch_tick_count", 0
        ) + (end - start)
    if birth_cursor != birth_count or death_cursor != death_count:
        raise StrategyAttributionError("compact dispatch auxiliary cursors incomplete")
    return start, end


def _stream_initial_mixed(state: _StreamingWorldState) -> bool:
    initialization = state.world.semantic_inputs.to_value().get("initialization", {})
    composition = initialization.get("exact_founder_composition")
    if isinstance(composition, list) and len(composition) > 1:
        return True
    return len(
        {
            strategy
            for agent in state.active.values()
            if agent.founder
            for strategy in agent.strategy_ids
        }
    ) > 1


def _stream_world_result(
    state: _StreamingWorldState,
    *,
    terminal: int,
    early_life_tick: int,
    thresholds: Sequence[int],
) -> dict[str, Any]:
    strategies = list(state.contract["strategy_ids"])
    pressure = _pressure_exposure(state.world)
    pressure_end = state.extinction_tick if state.extinction_tick is not None else terminal
    pressure["exposed_tick_span"] = (
        max(0, int(pressure_end) - int(pressure["onset_absolute_tick"]))
        if pressure["onset_absolute_tick"] is not None
        else 0
    )
    # Terminal active agents are included in offspring-survival and early
    # persistence denominators without retaining dead historical agents.
    for agent in state.active.values():
        _stream_add_offspring_survival(
            state,
            agent,
            death_tick=None,
            thresholds=thresholds,
            early_life_tick=early_life_tick,
        )
        if agent.latest_metrics is None:
            raise StrategyAttributionError("terminal active agent lacks metrics")
        _stream_add_terminal_metrics(state, agent, agent.latest_metrics)
    totals = {
        strategy: dict(sorted(state.metric_totals[strategy].items()))
        for strategy in sorted(state.metric_totals)
    }
    all_totals = {
        name: sum(int(values.get(name, 0)) for values in totals.values())
        for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
    }
    initial_profiles = state.world.semantic_inputs.to_value().get(
        "initialization", {}
    ).get("exact_founder_composition", [])
    return {
        "world_id": state.world.world_id,
        "science": state.world.science.to_value(),
        "dimension_bindings": state.world.semantic_inputs.to_value().get(
            "dimension_bindings", {}
        ),
        "semantic_world_sha256": state.world.semantic_world_sha256.to_dict(),
        "acquisition_sha256": state.world.acquisition_sha256.to_dict(),
        "strategy_contract": {
            "schema_version": state.contract["schema_version"],
            "strategy_ids": strategies,
            "identity_mode": state.contract.get("identity_mode"),
            "lineage_mode": state.contract.get("lineage_mode"),
            "denominator_fields": list(state.contract["denominator_fields"]),
            "policy_mapping_count": len(state.contract["policy_id_to_strategy_id"]),
        },
        "initial_composition": {
            "mixed": _stream_initial_mixed(state),
            "founder_profiles": initial_profiles,
        },
        "event_counts": {
            "birth": state.births,
            "death": state.deaths,
            "lineage": state.total_agents,
        },
        "time_to_first_birth": {
            "absolute_tick": state.first_birth_tick,
            "value": state.first_birth_tick,
        },
        "time_to_first_death": {
            "absolute_tick": state.first_death_tick,
            "value": state.first_death_tick,
        },
        "extinction": {
            "absolute_tick": state.extinction_tick,
            "survives_terminal": state.extinction_tick is None,
        },
        "survival_curve": sorted(
            state.snapshots, key=lambda row: int(row["absolute_tick"])
        ),
        "reproduction_funnel": {
            "all_agents": all_totals,
            "by_strategy": totals,
            "agent_counts_by_strategy": dict(sorted(state.agent_counts.items())),
            "parent_birth_events_by_strategy": dict(
                sorted(state.parent_birth_events.items())
            ),
            "offspring_birth_events_by_strategy": dict(
                sorted(state.offspring_birth_events.items())
            ),
            "parent_pair_events": dict(sorted(state.parent_pair_events.items())),
            "mixed_parent_birth_count": state.mixed_parent_births,
            "assignment_mode": "parent_ancestry_union_v1",
        },
        "offspring_survival": {
            "threshold_ticks": list(thresholds),
            "by_strategy": {
                strategy: {
                    "born": state.offspring_born.get(strategy, 0),
                    "survived_to": dict(
                        sorted(state.offspring_survived.get(strategy, {}).items())
                    ),
                    "lineage_depth_max": state.max_depth_by_strategy.get(strategy, 0),
                }
                for strategy in sorted(state.offspring_born)
            },
        },
        "lineage": {
            "agent_count": state.total_agents,
            "founder_count": state.founder_count,
            "maximum_depth": state.max_depth,
            "maximum_depth_by_strategy": {
                strategy: state.max_depth_by_strategy.get(strategy, 0)
                for strategy in strategies
            },
            "terminal_alive_by_strategy": {
                strategy: sum(
                    int(strategy in agent.strategy_ids)
                    for agent in state.active.values()
                )
                for strategy in strategies
            },
            "persistence_to_early_life_by_strategy": {
                strategy: state.persistence_to_early.get(strategy, 0)
                for strategy in strategies
            },
        },
        "pressure_exposure": pressure,
        "integrity": {
            **state.integrity,
            "terminal_next_absolute_tick": terminal,
            "all_agents_have_terminal_metrics": True,
            "strategy_identity_exact": True,
        },
    }


def _strategy_ids_for_mask_v1(
    mask: int,
    strategy_ids: Sequence[str],
) -> tuple[str, ...]:
    if (
        type(mask) is not int
        or mask <= 0
        or mask >> len(strategy_ids)
    ):
        raise StrategyAttributionError("native strategy mask is invalid")
    return tuple(
        strategy_id
        for index, strategy_id in enumerate(strategy_ids)
        if mask & (1 << index)
    )


def render_native_strategy_attribution_v1(
    capture_index: Mapping[str, Any],
    world_bundle: ScienceWorldBundleV1,
    *,
    receipts_by_world: Mapping[str, "NativeStrategyReceiptResultV1"],
    snapshots_by_world: Mapping[str, Sequence[Mapping[str, Any]]],
    initial_strategy_masks_by_world: Mapping[str, Sequence[int]],
    atlas_integrity_by_world: Mapping[str, Mapping[str, Any]],
    early_life_tick: int = 10_000,
    pressure_factor: str = "pressure_regime",
) -> dict[str, Any]:
    """Render the canonical strategy result from fused native statistics.

    This function performs no lifecycle-event replay.  Authentication,
    decompression, lineage reconstruction, ancestry propagation and metric
    accumulation have already occurred in the same native pass as the atlas.
    The Python work here is a bounded canonical projection over one receipt
    and the declared snapshot summaries for each logical world.
    """

    if type(early_life_tick) is not int or early_life_tick < 1:
        raise StrategyAttributionError("early_life_tick must be positive")
    index = dict(capture_index)
    _validate_index(index, world_bundle)
    terminal = _as_int(
        index.get("terminal_next_absolute_tick"), label="terminal tick"
    )
    world_ids = tuple(world.world_id for world in world_bundle.worlds)
    expected_world_ids = set(world_ids)
    for label, values in (
        ("native receipts", receipts_by_world),
        ("strategy snapshots", snapshots_by_world),
        ("initial strategy masks", initial_strategy_masks_by_world),
        ("atlas integrity", atlas_integrity_by_world),
    ):
        if set(values) != expected_world_ids:
            raise StrategyAttributionError(f"{label} logical-world ownership differs")

    threshold_ticks = _stream_thresholds(terminal, early_life_tick)
    results: list[dict[str, Any]] = []
    for world in world_bundle.worlds:
        contract = _strategy_contract(world)
        strategy_ids = tuple(str(value) for value in contract["strategy_ids"])
        receipt = receipts_by_world[world.world_id]
        if (
            not receipt.enabled
            or not receipt.finalized
            or receipt.strategy_count != len(strategy_ids)
            or receipt.threshold_ticks != threshold_ticks
            or receipt.early_life_tick != early_life_tick
            or len(receipt.agent_counts_by_strategy) != len(strategy_ids)
            or len(receipt.terminal_metric_totals_by_strategy)
            != len(strategy_ids)
        ):
            raise StrategyAttributionError("native strategy receipt contract differs")
        direct_genotype = genotype_attribution_required_v1(contract)
        genotype_result = receipt.genotype_reproductive_attribution
        if direct_genotype:
            if not isinstance(genotype_result, Mapping):
                raise StrategyAttributionError(
                    "native strategy receipt lacks direct genotype attribution"
                )
            validate_genotype_reproductive_attribution_v1(genotype_result)
        if any(
            len(row) != len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
            for row in receipt.terminal_metric_totals_by_strategy
        ):
            raise StrategyAttributionError("native strategy metric width differs")

        snapshots = [dict(row) for row in snapshots_by_world[world.world_id]]
        snapshots.sort(key=lambda row: _as_int(row.get("absolute_tick"), label="snapshot tick"))
        expected_snapshots = list(world.acquisition.to_value()["snapshot_ticks"])
        if (
            [row["absolute_tick"] for row in snapshots] != expected_snapshots
            or not snapshots
            or snapshots[-1]["absolute_tick"] != terminal
        ):
            raise StrategyAttributionError("native strategy snapshot coverage differs")

        totals = {
            strategy_id: dict(
                sorted(
                    zip(
                        COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
                        receipt.terminal_metric_totals_by_strategy[index_value],
                        strict=True,
                    )
                )
            )
            for index_value, strategy_id in enumerate(strategy_ids)
            if receipt.agent_counts_by_strategy[index_value] > 0
        }
        all_totals = {
            name: sum(int(values.get(name, 0)) for values in totals.values())
            for name in COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
        }
        agent_counts = {
            strategy_id: receipt.agent_counts_by_strategy[index_value]
            for index_value, strategy_id in enumerate(strategy_ids)
            if receipt.agent_counts_by_strategy[index_value] > 0
        }
        parent_birth_events = {
            strategy_id: receipt.parent_birth_events_by_strategy[index_value]
            for index_value, strategy_id in enumerate(strategy_ids)
            if receipt.parent_birth_events_by_strategy[index_value] > 0
        }
        offspring_birth_events: dict[str, int] = {}
        for mask, count in receipt.offspring_group_counts:
            group = strategy_group_id_v1(
                _strategy_ids_for_mask_v1(mask, strategy_ids)
            )
            offspring_birth_events[group] = offspring_birth_events.get(group, 0) + count
        parent_pair_events: dict[str, int] = {}
        for first_mask, second_mask, count in receipt.parent_pair_counts:
            first_group = strategy_group_id_v1(
                _strategy_ids_for_mask_v1(first_mask, strategy_ids)
            )
            second_group = strategy_group_id_v1(
                _strategy_ids_for_mask_v1(second_mask, strategy_ids)
            )
            group = "+".join(sorted((first_group, second_group)))
            parent_pair_events[group] = parent_pair_events.get(group, 0) + count

        pressure = _pressure_exposure(world)
        pressure_end = (
            receipt.extinction_tick
            if receipt.extinction_tick is not None
            else terminal
        )
        pressure["exposed_tick_span"] = (
            max(0, int(pressure_end) - int(pressure["onset_absolute_tick"]))
            if pressure["onset_absolute_tick"] is not None
            else 0
        )
        initialization = world.semantic_inputs.to_value().get("initialization", {})
        initial_profiles = initialization.get("exact_founder_composition", [])
        founder_masks = tuple(initial_strategy_masks_by_world[world.world_id])
        initial_strategy_count = len(
            {
                strategy_id
                for mask in founder_masks
                for strategy_id in _strategy_ids_for_mask_v1(mask, strategy_ids)
            }
        )
        initial_mixed = (
            isinstance(initial_profiles, list) and len(initial_profiles) > 1
        ) or initial_strategy_count > 1
        integrity = atlas_integrity_by_world[world.world_id]
        if (
            integrity.get("event_count")
            != receipt.birth_count + receipt.death_count
            or integrity.get("snapshot_count") != len(snapshots)
        ):
            raise StrategyAttributionError("native strategy integrity differs")

        strategy_contract_projection = {
            "schema_version": contract["schema_version"],
            "strategy_ids": list(strategy_ids),
            "identity_mode": contract.get("identity_mode"),
            "lineage_mode": contract.get("lineage_mode"),
            "denominator_fields": list(contract["denominator_fields"]),
            "policy_mapping_count": len(contract["policy_id_to_strategy_id"]),
        }
        if direct_genotype:
            strategy_contract_projection.update(
                {
                    "direct_genotype_attribution": True,
                    "genotype_identity_mode": contract["genotype_identity_mode"],
                    "genotype_denominator_fields": list(
                        contract["genotype_denominator_fields"]
                    ),
                }
            )
        result = {
            "world_id": world.world_id,
            "science": world.science.to_value(),
            "dimension_bindings": world.semantic_inputs.to_value().get(
                "dimension_bindings", {}
            ),
            "semantic_world_sha256": world.semantic_world_sha256.to_dict(),
            "acquisition_sha256": world.acquisition_sha256.to_dict(),
            "strategy_contract": strategy_contract_projection,
            "initial_composition": {
                "mixed": initial_mixed,
                "founder_profiles": initial_profiles,
            },
            "event_counts": {
                "birth": receipt.birth_count,
                "death": receipt.death_count,
                "lineage": receipt.total_agent_count,
            },
            "time_to_first_birth": {
                "absolute_tick": receipt.first_birth_tick,
                "value": receipt.first_birth_tick,
            },
            "time_to_first_death": {
                "absolute_tick": receipt.first_death_tick,
                "value": receipt.first_death_tick,
            },
            "extinction": {
                "absolute_tick": receipt.extinction_tick,
                "survives_terminal": receipt.extinction_tick is None,
            },
            "survival_curve": snapshots,
            "reproduction_funnel": {
                "all_agents": all_totals,
                "by_strategy": dict(sorted(totals.items())),
                "agent_counts_by_strategy": dict(sorted(agent_counts.items())),
                "parent_birth_events_by_strategy": dict(
                    sorted(parent_birth_events.items())
                ),
                "offspring_birth_events_by_strategy": dict(
                    sorted(offspring_birth_events.items())
                ),
                "parent_pair_events": dict(sorted(parent_pair_events.items())),
                "mixed_parent_birth_count": receipt.mixed_parent_birth_count,
                "assignment_mode": "parent_ancestry_union_v1",
            },
            "offspring_survival": {
                "threshold_ticks": list(threshold_ticks),
                "by_strategy": {
                    strategy_id: {
                        "born": receipt.offspring_born_by_strategy[index_value],
                        "survived_to": {
                            str(tick): count
                            for tick, count in zip(
                                threshold_ticks,
                                receipt.offspring_survived_by_strategy_threshold[
                                    index_value
                                ],
                                strict=True,
                            )
                            if count > 0
                        },
                        "lineage_depth_max": (
                            receipt.maximum_lineage_depth_by_strategy[index_value]
                        ),
                    }
                    for index_value, strategy_id in enumerate(strategy_ids)
                    if receipt.offspring_born_by_strategy[index_value] > 0
                },
            },
            "lineage": {
                "agent_count": receipt.total_agent_count,
                "founder_count": receipt.founder_count,
                "maximum_depth": receipt.maximum_lineage_depth,
                "maximum_depth_by_strategy": {
                    strategy_id: receipt.maximum_lineage_depth_by_strategy[index_value]
                    for index_value, strategy_id in enumerate(strategy_ids)
                },
                "terminal_alive_by_strategy": {
                    strategy_id: receipt.terminal_alive_by_strategy[index_value]
                    for index_value, strategy_id in enumerate(strategy_ids)
                },
                "persistence_to_early_life_by_strategy": {
                    strategy_id: (
                        receipt.persistence_to_early_life_by_strategy[index_value]
                    )
                    for index_value, strategy_id in enumerate(strategy_ids)
                },
            },
            "pressure_exposure": pressure,
            "integrity": {
                "dispatch_count": int(integrity["dispatch_count"]),
                "event_count": int(integrity["event_count"]),
                "snapshot_count": int(integrity["snapshot_count"]),
                "dropped_record_count": 0,
                "dispatch_tick_count": int(integrity["dispatch_tick_count"]),
                "terminal_next_absolute_tick": terminal,
                "all_agents_have_terminal_metrics": True,
                "strategy_identity_exact": True,
            },
        }
        if direct_genotype:
            result["genotype_reproductive_attribution"] = dict(genotype_result)
            result["integrity"]["genotype_identity_exact"] = True
        results.append(result)

    pressure_controls: dict[tuple[tuple[str, Any], ...], bool] = {}
    for result in results:
        dimensions = result.get("dimension_bindings", {})
        key = tuple(
            sorted(
                (str(name), value)
                for name, value in dimensions.items()
                if name not in {pressure_factor, "factorial_index"}
            )
        )
        if dimensions.get(pressure_factor) in {"off", "none", "disabled"}:
            pressure_controls[key] = result["survival_curve"][-1]["active_count"] > 0
    for result in results:
        dimensions = result.get("dimension_bindings", {})
        key = tuple(
            sorted(
                (str(name), value)
                for name, value in dimensions.items()
                if name not in {pressure_factor, "factorial_index"}
            )
        )
        result["causal_classification"] = classify_causal_class_v1(
            result,
            early_life_tick=early_life_tick,
            pressure_control_survives=pressure_controls.get(key),
        )
    body = {
        "schema_version": STRATEGY_ATTRIBUTION_RESULT_SCHEMA_V1,
        "bundle_content_sha256": world_bundle.bundle_content_sha256().to_dict(),
        "capture_index_sha256": index["index_sha256"],
        "logical_world_count": len(results),
        "physical_batch_count": 1,
        "physical_composition_used": False,
        "early_life_tick": early_life_tick,
        "pressure_pairing_factor": pressure_factor,
        "world_results": results,
        "classification_counts": {
            value: sum(
                int(result["causal_classification"]["class"] == value)
                for result in results
            )
            for value in ("A", "B", "C", "D", "E", "UNRESOLVED")
        },
    }
    return {**body, "result_sha256": canonical_sha256(body).to_dict()}


def analyze_compact_science_strategy_attribution_streaming_v1(
    capture_index: str | Path | Mapping[str, Any],
    world_bundle: str | Path | ScienceWorldBundleV1,
    *,
    early_life_tick: int = 10_000,
    pressure_factor: str = "pressure_regime",
) -> dict[str, Any]:
    """Reduce a sealed capture with bounded active-lineage memory.

    The output schema and ownership/hash checks are identical to the reference
    analyzer.  Only the internal retention strategy differs: dead agents are
    folded into exact strategy totals and threshold counters immediately.
    """

    if type(early_life_tick) is not int or early_life_tick < 1:
        raise StrategyAttributionError("early_life_tick must be positive")
    bundle = _load_bundle(world_bundle)
    index = _load_index(capture_index)
    dispatches, snapshots = _validate_index(index, bundle)
    terminal = _as_int(index.get("terminal_next_absolute_tick"), label="terminal tick")
    states = {
        world.world_id: _StreamingWorldState(
            world=world,
            contract=_strategy_contract(world),
            start_tick=_as_int(
                world.semantic_inputs.to_value()["time"].get("start_next_tick", 0),
                label="world start tick",
            ),
        )
        for world in bundle.worlds
    }
    expected_starts = {state.start_tick for state in states.values()}
    if len(expected_starts) != 1:
        raise StrategyAttributionError("strategy attribution requires one absolute start frontier")
    start_tick = expected_starts.pop()
    threshold_ticks = _stream_thresholds(terminal, early_life_tick)
    ordered_snapshots = sorted(
        snapshots,
        key=lambda pointer: _as_int(
            _verify_artifact(pointer, label="snapshot")[1].get("absolute_tick"),
            label="snapshot tick",
        ),
    )
    snapshot_ticks = [
        _as_int(
            _verify_artifact(pointer, label="snapshot")[1].get("absolute_tick"),
            label="snapshot tick",
        )
        for pointer in ordered_snapshots
    ]
    if not ordered_snapshots or snapshot_ticks[0] != start_tick:
        raise StrategyAttributionError("strategy attribution requires the absolute start snapshot")
    if snapshot_ticks[-1] != terminal:
        raise StrategyAttributionError("strategy attribution requires the terminal snapshot")
    ordered_dispatches = sorted(
        dispatches,
        key=lambda pointer: _as_int(
            _verify_artifact(pointer, label="dispatch")[1].get(
                "start_next_absolute_tick"
            ),
            label="dispatch start tick",
        ),
    )
    # Establish every founder shard at the absolute start frontier before the
    # first dispatch.  Snapshot publication may seal one full physical batch
    # or several disjoint logical-world artifacts at the same tick.
    snapshot_cursor = 0
    while (
        snapshot_cursor < len(ordered_snapshots)
        and snapshot_ticks[snapshot_cursor] == start_tick
    ):
        _stream_read_snapshot(ordered_snapshots[snapshot_cursor], states)
        snapshot_cursor += 1
    dispatch_frontier = start_tick
    for pointer in ordered_dispatches:
        start, end = _stream_read_dispatch(
            pointer,
            states,
            thresholds=threshold_ticks,
            early_life_tick=early_life_tick,
        )
        if start != dispatch_frontier:
            raise StrategyAttributionError("dispatch frontier is not contiguous")
        dispatch_frontier = end
        while snapshot_cursor < len(ordered_snapshots):
            _, manifest = _verify_artifact(ordered_snapshots[snapshot_cursor], label="snapshot")
            tick = _as_int(manifest.get("absolute_tick"), label="snapshot tick")
            if tick > end:
                break
            _stream_read_snapshot(ordered_snapshots[snapshot_cursor], states)
            snapshot_cursor += 1
    while snapshot_cursor < len(ordered_snapshots):
        _stream_read_snapshot(ordered_snapshots[snapshot_cursor], states)
        snapshot_cursor += 1
    if dispatch_frontier != terminal:
        raise StrategyAttributionError("dispatch frontier does not reach terminal tick")
    for state in states.values():
        if [row["absolute_tick"] for row in state.snapshots] != list(
            state.world.acquisition.to_value()["snapshot_ticks"]
        ):
            raise StrategyAttributionError(
                "strategy attribution snapshot coverage differs"
            )
    results = [
        _stream_world_result(
            states[world.world_id],
            terminal=terminal,
            early_life_tick=early_life_tick,
            thresholds=threshold_ticks,
        )
        for world in bundle.worlds
    ]
    pressure_controls: dict[tuple[tuple[str, Any], ...], bool] = {}
    for result in results:
        dimensions = result.get("dimension_bindings", {})
        key = tuple(
            sorted(
                (str(name), value)
                for name, value in dimensions.items()
                if name not in {pressure_factor, "factorial_index"}
            )
        )
        if dimensions.get(pressure_factor) in {"off", "none", "disabled"}:
            pressure_controls[key] = (
                states[result["world_id"]].snapshots[-1]["active_count"] > 0
            )
    for result in results:
        dimensions = result.get("dimension_bindings", {})
        key = tuple(
            sorted(
                (str(name), value)
                for name, value in dimensions.items()
                if name not in {pressure_factor, "factorial_index"}
            )
        )
        result["causal_classification"] = classify_causal_class_v1(
            result,
            early_life_tick=early_life_tick,
            pressure_control_survives=pressure_controls.get(key),
        )
    body = {
        "schema_version": STRATEGY_ATTRIBUTION_RESULT_SCHEMA_V1,
        "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
        "capture_index_sha256": index["index_sha256"],
        "logical_world_count": len(results),
        "physical_batch_count": 1,
        "physical_composition_used": False,
        "early_life_tick": early_life_tick,
        "pressure_pairing_factor": pressure_factor,
        "world_results": results,
        "classification_counts": {
            value: sum(
                int(result["causal_classification"]["class"] == value)
                for result in results
            )
            for value in ("A", "B", "C", "D", "E", "UNRESOLVED")
        },
    }
    return {**body, "result_sha256": canonical_sha256(body).to_dict()}


__all__ = (
    "genotype_attribution_required_v1",
    "validate_genotype_reproductive_attribution_v1",
    "STRATEGY_ATTRIBUTION_RESULT_SCHEMA_V1",
    "StrategyAttributionError",
    "analyze_compact_science_strategy_attribution_v1",
    "analyze_compact_science_strategy_attribution_streaming_v1",
    "classify_causal_class_v1",
    "render_native_strategy_attribution_v1",
    "strategy_attribution_contract_v1",
    "strategy_attribution_required_for_bundle_v1",
    "strategy_group_id_v1",
    "strategy_threshold_ticks_v1",
    "strategy_union_v1",
)
