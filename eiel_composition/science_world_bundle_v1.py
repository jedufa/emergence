"""Pure materialization, admission, packing, and demultiplexing for science worlds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    SchemaVersion,
    ScienceAdmissionOutcomeV1,
    ScienceCampaignV1,
    ScienceWorldBundleV1,
    ScienceWorldRecordV1,
    SCIENCE_GENOTYPE_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_HEREDITARY_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1,
    validate_hereditary_attribution_configuration_v1,
    strategy_attribution_configuration_v1,
    validate_strategy_attribution_configuration_v1,
    canonical_json_bytes,
    canonical_sha256,
)

from .science_instrument import ScienceCompilationBundleV1


SCIENCE_WORLD_BUNDLE_COMPILER_ID = "eiel_science_world_bundle_compiler_v1"
SCIENCE_WORLD_PACKING_SCHEMA_VERSION = "eiel_science_world_physical_packing_v1"
SCIENCE_WORLD_ADMISSION_SCHEMA_VERSION = "eiel_science_world_bundle_admission_v1"


class ScienceWorldBundleError(ValueError):
    """A compact campaign cannot be materialized as exact explicit worlds."""


SCIENCE_WORLD_BUNDLE_FILE_SCHEMA_V1 = "eiel_science_world_bundle_file_v1"


def write_science_world_bundle_v1(
    bundle: ScienceWorldBundleV1,
    path: str | Path,
) -> dict[str, object]:
    """Materialize one immutable, content-addressed bundle file.

    The file is written with exclusive creation and made read-only after the
    canonical bytes are durable.  Callers retain the returned digest as the
    immutable execution binding; this helper never overwrites an existing
    path or silently relocates a bundle.
    """

    if not isinstance(bundle, ScienceWorldBundleV1):
        raise ScienceWorldBundleError("bundle must be a ScienceWorldBundleV1")
    target = Path(path).expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise ScienceWorldBundleError(f"bundle output must be a fresh regular path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = bundle.to_dict()
    encoded = canonical_json_bytes(payload)
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        target.chmod(0o444)
    except (OSError, ValueError) as exc:
        try:
            target.unlink()
        except OSError:
            pass
        raise ScienceWorldBundleError(f"cannot materialize bundle: {exc}") from exc
    return {
        "schema_version": SCIENCE_WORLD_BUNDLE_FILE_SCHEMA_V1,
        "path": str(target),
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
        "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
        "world_count": bundle.world_count,
        "immutable": True,
    }


def load_science_world_bundle_v1(path: str | Path) -> ScienceWorldBundleV1:
    """Load and validate canonical bytes from one materialized bundle file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ScienceWorldBundleError(f"bundle path is not a regular file: {source}")
    try:
        raw_bytes = source.read_bytes()
        raw = json.loads(raw_bytes.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ScienceWorldBundleError(f"bundle file is not canonical JSON: {source}") from exc
    if canonical_json_bytes(raw) != raw_bytes:
        raise ScienceWorldBundleError("bundle file is not in canonical JSON form")
    try:
        return ScienceWorldBundleV1.from_dict(raw)
    except Exception as exc:
        if isinstance(exc, ScienceWorldBundleError):
            raise
        raise ScienceWorldBundleError(f"bundle file is invalid: {exc}") from exc


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScienceWorldBundleError(f"{field} must be an object")
    return value


def _list(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ScienceWorldBundleError(f"{field} must be an array")
    return value


def _science_ownership(
    campaign: ScienceCampaignV1, cell: Mapping[str, Any]
) -> dict[str, object]:
    question = _mapping(campaign.question.to_value(), field="question")
    ownership = _mapping(question.get("science_ownership"), field="question.science_ownership")
    dimensions = _mapping(cell.get("dimension_bindings"), field="dimension_bindings")
    for field in ("test_id", "test_version", "study_id"):
        if not isinstance(ownership.get(field), str) or not ownership[field]:
            raise ScienceWorldBundleError(f"science ownership requires {field}")

    arm_dimensions = ownership.get("arm_dimension_ids", [])
    if not isinstance(arm_dimensions, list) or any(
        not isinstance(item, str) or item not in dimensions for item in arm_dimensions
    ):
        raise ScienceWorldBundleError("arm_dimension_ids must name resolved dimensions")
    if arm_dimensions:
        arm_id = "+".join(
            f"{dimension}={dimensions[dimension]}" for dimension in arm_dimensions
        )
    else:
        arm_id = ownership.get("fixed_arm_id")
    if not isinstance(arm_id, str) or not arm_id:
        raise ScienceWorldBundleError("science ownership requires an explicit arm")

    replicate_dimension = ownership.get("replicate_dimension_id")
    if replicate_dimension is not None:
        if not isinstance(replicate_dimension, str) or replicate_dimension not in dimensions:
            raise ScienceWorldBundleError(
                "replicate_dimension_id must name a resolved dimension"
            )
        replicate: object = dimensions[replicate_dimension]
    else:
        replicate = ownership.get("fixed_replicate")
    if not isinstance(replicate, (str, int)) or isinstance(replicate, bool):
        raise ScienceWorldBundleError("science ownership requires an explicit replicate")
    return {
        "test_id": ownership["test_id"],
        "test_version": ownership["test_version"],
        "study_id": ownership["study_id"],
        "arm_id": arm_id,
        "replicate": replicate,
    }


def _acquisition(campaign: ScienceCampaignV1) -> dict[str, object]:
    observations = _mapping(campaign.observations.to_value(), field="observations")
    time = _mapping(campaign.time.to_value(), field="time")
    snapshot_ticks = observations.get(
        "milestone_ticks", time.get("observation_ticks", [])
    )
    return {
        "profile_id": observations.get("profile_id"),
        "event_observations": sorted(observations.get("event_observations", [])),
        "state_observations": sorted(observations.get("state_observations", [])),
        "runtime_counters": sorted(observations.get("runtime_counters", [])),
        "snapshot_ticks": sorted(snapshot_ticks),
        "windows": sorted(
            observations.get("observation_windows", []),
            key=lambda row: (row.get("start_tick"), row.get("end_tick")),
        ),
    }


def _reducer_bindings(
    campaign: ScienceCampaignV1, compilation: ScienceCompilationBundleV1
) -> list[dict[str, object]]:
    configs = {
        row["reducer_id"]: row["configuration"]
        for row in _list(
            compilation.reduction_plan.normalized_configurations.to_value(),
            field="normalized_configurations",
        )
    }
    bindings: list[dict[str, object]] = []
    for descriptor in compilation.reduction_plan.reducer_descriptors:
        configuration = configs[descriptor.reducer_id]
        required_observations = list(descriptor.required_observation_capability_ids)
        # Strategy attribution is an explicit, reusable reducer mode.  It is
        # added to the per-world binding only when configuration opts in, so
        # ordinary compact reductions retain their smaller observation ABI.
        if strategy_attribution_configuration_v1(configuration) is not None:
            required_observations = sorted(
                set(required_observations)
                | set(SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1)
            )
            strategy_binding = strategy_attribution_configuration_v1(configuration)
            if isinstance(strategy_binding, Mapping) and strategy_binding.get(
                "direct_genotype_attribution", False
            ):
                required_observations = sorted(
                    set(required_observations)
                    | set(SCIENCE_GENOTYPE_ATTRIBUTION_CAPABILITIES_V1)
                )
        hereditary_binding = configuration.get("hereditary_attribution")
        if hereditary_binding is not None:
            required_observations = sorted(
                set(required_observations)
                | set(SCIENCE_HEREDITARY_ATTRIBUTION_CAPABILITIES_V1)
            )
        bindings.append(
            {
            "reducer_id": descriptor.reducer_id,
            "contract_version": str(descriptor.contract_version),
            "required_observations": required_observations,
            "output_contract_id": descriptor.output_contract_id,
            "configuration": configuration,
            }
        )
    return bindings


def materialize_science_world_bundle_v1(
    campaigns: Iterable[tuple[ScienceCampaignV1, ScienceCompilationBundleV1]],
    *,
    bundle_id: str,
) -> ScienceWorldBundleV1:
    """Expand compact compiled campaigns into one heterogeneous explicit bundle."""

    pairs = tuple(campaigns)
    if not pairs:
        raise ScienceWorldBundleError("at least one compiled campaign is required")
    worlds: list[ScienceWorldRecordV1] = []
    source_specs: list[dict[str, object]] = []
    compiler_digests: list[dict[str, str]] = []
    for campaign, compilation in pairs:
        if compilation.campaign_spec_sha256 != campaign.campaign_spec_sha256():
            raise ScienceWorldBundleError("campaign and compilation identities differ")
        source_specs.append(
            {
                "campaign_id": campaign.campaign_id,
                "campaign_spec_sha256": compilation.campaign_spec_sha256.to_dict(),
                "compilation_bundle_sha256": compilation.compilation_bundle_sha256().to_dict(),
            }
        )
        compiler_digests.append(compilation.compiler_component_sha256.to_dict())
        acquisition = _acquisition(campaign)
        reducers = _reducer_bindings(campaign, compilation)
        time = campaign.time.to_value()
        for raw_cell in _list(compilation.semantic_cells.to_value(), field="semantic_cells"):
            cell = _mapping(raw_cell, field="semantic_cell")
            science = _science_ownership(campaign, cell)
            resolved = {
                "world": cell["world"],
                "initialization": cell["initialization"],
                "dimension_bindings": cell["dimension_bindings"],
                "control_bindings": cell["control_bindings"],
                "seed_bindings": cell["seed_bindings"],
                "time": time,
                "initial_state_identity": canonical_sha256(
                    {
                        "world": cell["world"],
                        "initialization": cell["initialization"],
                        "seed_bindings": cell["seed_bindings"],
                    }
                ).to_dict(),
            }
            science_payload = CanonicalPayloadV1.from_value(
                SchemaVersion("eiel_science_ownership_v1"), science
            )
            semantic_payload = CanonicalPayloadV1.from_value(
                SchemaVersion("eiel_science_resolved_inputs_v1"), resolved
            )
            acquisition_payload = CanonicalPayloadV1.from_value(
                SchemaVersion("eiel_science_acquisition_contract_v1"), acquisition
            )
            reducer_payload = CanonicalPayloadV1.from_value(
                SchemaVersion("eiel_science_world_reducers_v1"), reducers
            )
            semantic_identity = canonical_sha256(
                {
                    "schema_version": "eiel_science_world_record_v1",
                    "science": science,
                    "semantic_inputs": resolved,
                }
            )
            acquisition_identity = canonical_sha256(
                {
                    "schema_version": "eiel_science_world_record_v1",
                    "semantic_world_sha256": semantic_identity.to_dict(),
                    "acquisition": acquisition,
                    "reducers": reducers,
                }
            )
            worlds.append(
                ScienceWorldRecordV1(
                    world_id=f"world-{semantic_identity.hex[:32]}",
                    science=science_payload,
                    semantic_inputs=semantic_payload,
                    acquisition=acquisition_payload,
                    reducers=reducer_payload,
                    semantic_world_sha256=semantic_identity,
                    acquisition_sha256=acquisition_identity,
                )
            )
    source_specs.sort(key=lambda row: str(row["campaign_id"]))
    worlds.sort(key=lambda world: world.world_id)
    compiler_component_sha256 = canonical_sha256(
        {
            "compiler_id": SCIENCE_WORLD_BUNDLE_COMPILER_ID,
            "source_compiler_component_sha256s": sorted(
                compiler_digests, key=lambda row: row["hex"]
            ),
        }
    )
    return ScienceWorldBundleV1(
        bundle_id=bundle_id,
        compact_source_specs=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_science_compact_source_specs_v1"), source_specs
        ),
        compiler_component_sha256=compiler_component_sha256,
        worlds=tuple(worlds),
    )


@dataclass(frozen=True, slots=True)
class ScienceWorldAdmissionFindingV1:
    outcome: ScienceAdmissionOutcomeV1
    world_id: str
    test_id: str
    field: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "outcome": self.outcome.value,
            "world_id": self.world_id,
            "test_id": self.test_id,
            "field": self.field,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ScienceWorldBundleAdmissionV1:
    bundle_content_sha256: ContentDigest
    admitted_world_ids: tuple[str, ...]
    findings: tuple[ScienceWorldAdmissionFindingV1, ...]

    @property
    def outcome(self) -> ScienceAdmissionOutcomeV1:
        if self.findings:
            return self.findings[0].outcome
        return ScienceAdmissionOutcomeV1.ADMITTED

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCIENCE_WORLD_ADMISSION_SCHEMA_VERSION,
            "bundle_content_sha256": self.bundle_content_sha256.to_dict(),
            "outcome": self.outcome.value,
            "admitted_world_ids": list(self.admitted_world_ids),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def admit_science_world_bundle_v1(
    bundle: ScienceWorldBundleV1,
    *,
    supported_observations: Iterable[str],
    supported_profiles: Iterable[str],
    maximum_total_ticks: int,
    nonexecuted_reducer_ids: Iterable[str] = (),
) -> ScienceWorldBundleAdmissionV1:
    """Fail closed on unavailable acquisition or absolute-time requirements.

    ``nonexecuted_reducer_ids`` is an internal lowering-boundary declaration,
    not reducer authority.  It permits a caller that separately proves a
    reducer cannot execute for this physical route to retain and validate the
    immutable reducer binding without pretending its input ABI is available.
    """

    supported = set(supported_observations)
    profiles = set(supported_profiles)
    nonexecuted = tuple(nonexecuted_reducer_ids)
    if (
        any(not isinstance(value, str) or not value for value in nonexecuted)
        or nonexecuted != tuple(sorted(set(nonexecuted)))
    ):
        raise ScienceWorldBundleError(
            "nonexecuted reducer IDs must be sorted unique nonempty tokens"
        )
    bound_reducer_ids = {
        reducer.get("reducer_id")
        for world in bundle.worlds
        for reducer in world.reducers.to_value()
        if isinstance(reducer, Mapping)
        and isinstance(reducer.get("reducer_id"), str)
    }
    if not set(nonexecuted).issubset(bound_reducer_ids):
        raise ScienceWorldBundleError(
            "nonexecuted reducer declaration is absent from the bundle"
        )
    findings: list[ScienceWorldAdmissionFindingV1] = []
    admitted: list[str] = []
    for world in bundle.worlds:
        science = world.science.to_value()
        acquisition = world.acquisition.to_value()
        semantic = world.semantic_inputs.to_value()
        required = set(acquisition["event_observations"])
        required.update(acquisition["state_observations"])
        required.update(acquisition["runtime_counters"])
        reducer_contract_invalid = False
        for reducer in world.reducers.to_value():
            reducer_id = reducer.get("reducer_id")
            required_observations = reducer.get("required_observations")
            if (
                not isinstance(reducer_id, str)
                or not reducer_id
                or not isinstance(required_observations, list)
                or any(
                    not isinstance(value, str) or not value
                    for value in required_observations
                )
                or required_observations != sorted(set(required_observations))
            ):
                findings.append(
                    ScienceWorldAdmissionFindingV1(
                        outcome=ScienceAdmissionOutcomeV1.INVALID_CAMPAIGN,
                        world_id=world.world_id,
                        test_id=f"{science['test_id']}_V{science['test_version']}",
                        field="reducers.required_observations",
                        detail="required observations must be a sorted unique token list",
                    )
                )
                reducer_contract_invalid = True
                break
            configuration = reducer.get("configuration")
            if not isinstance(configuration, Mapping):
                findings.append(
                    ScienceWorldAdmissionFindingV1(
                        outcome=ScienceAdmissionOutcomeV1.INVALID_CAMPAIGN,
                        world_id=world.world_id,
                        test_id=f"{science['test_id']}_V{science['test_version']}",
                        field="reducers.configuration",
                        detail="reducer configuration must be an object",
                    )
                )
                reducer_contract_invalid = True
                break
            if reducer_id not in nonexecuted:
                required.update(required_observations)
            try:
                strategy_errors = validate_strategy_attribution_configuration_v1(
                    configuration,
                    required_observations=required_observations,
                )
            except ValueError as exc:
                strategy_errors = (str(exc),)
            if strategy_errors:
                findings.append(
                    ScienceWorldAdmissionFindingV1(
                        outcome=ScienceAdmissionOutcomeV1.MISSING_OBSERVATION_CAPABILITY,
                        world_id=world.world_id,
                        test_id=f"{science['test_id']}_V{science['test_version']}",
                        field="reducers.strategy_attribution",
                        detail="invalid or incomplete strategy attribution contract: "
                        + ", ".join(strategy_errors),
                    )
                )
                reducer_contract_invalid = True
                break
            try:
                hereditary_errors = validate_hereditary_attribution_configuration_v1(
                    configuration,
                    required_observations=required_observations,
                )
            except ValueError as exc:
                hereditary_errors = (str(exc),)
            if hereditary_errors:
                findings.append(
                    ScienceWorldAdmissionFindingV1(
                        outcome=ScienceAdmissionOutcomeV1.MISSING_OBSERVATION_CAPABILITY,
                        world_id=world.world_id,
                        test_id=f"{science['test_id']}_V{science['test_version']}",
                        field="reducers.hereditary_attribution",
                        detail="invalid or incomplete full-hereditary attribution contract: "
                        + ", ".join(hereditary_errors),
                    )
                )
                reducer_contract_invalid = True
                break
        if reducer_contract_invalid:
            continue
        missing = sorted(required - supported)
        if missing:
            findings.append(
                ScienceWorldAdmissionFindingV1(
                    outcome=ScienceAdmissionOutcomeV1.MISSING_OBSERVATION_CAPABILITY,
                    world_id=world.world_id,
                    test_id=f"{science['test_id']}_V{science['test_version']}",
                    field="acquisition.required_observations",
                    detail=f"missing: {', '.join(missing)}",
                )
            )
            continue
        if acquisition["profile_id"] not in profiles:
            findings.append(
                ScienceWorldAdmissionFindingV1(
                    outcome=ScienceAdmissionOutcomeV1.MISSING_OBSERVATION_CAPABILITY,
                    world_id=world.world_id,
                    test_id=f"{science['test_id']}_V{science['test_version']}",
                    field="acquisition.profile_id",
                    detail=f"unsupported profile: {acquisition['profile_id']}",
                )
            )
            continue
        time = semantic["time"]
        start = time["start_next_tick"]
        target = start + time["total_ticks"]
        requested_ticks = [*acquisition["snapshot_ticks"], *time.get("checkpoint_ticks", [])]
        windows = acquisition["windows"]
        invalid_timing = (
            time["total_ticks"] > maximum_total_ticks
            or any(tick < start or tick > target for tick in requested_ticks)
            or any(
                row["start_tick"] < start
                or row["end_tick"] > target
                or row["end_tick"] < row["start_tick"]
                for row in windows
            )
        )
        if invalid_timing:
            findings.append(
                ScienceWorldAdmissionFindingV1(
                    outcome=ScienceAdmissionOutcomeV1.OUTSIDE_QUALIFIED_ENVELOPE,
                    world_id=world.world_id,
                    test_id=f"{science['test_id']}_V{science['test_version']}",
                    field="semantic_inputs.time",
                    detail="absolute-tick request is outside the engineering envelope",
                )
            )
            continue
        admitted.append(world.world_id)
    findings.sort(key=lambda item: (item.outcome.value, item.world_id, item.field))
    return ScienceWorldBundleAdmissionV1(
        bundle_content_sha256=bundle.bundle_content_sha256(),
        admitted_world_ids=tuple(sorted(admitted)) if not findings else (),
        findings=tuple(findings),
    )


@dataclass(frozen=True, slots=True)
class SciencePhysicalPackingV1:
    bundle_content_sha256: ContentDigest
    carrier_id: str
    carrier_capacity: int
    batches: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if self.carrier_capacity < 1 or not self.batches:
            raise ScienceWorldBundleError("physical packing is empty or invalid")
        flattened = [world_id for batch in self.batches for world_id in batch]
        if len(flattened) != len(set(flattened)):
            raise ScienceWorldBundleError("physical packing duplicates a world")
        if any(not batch or len(batch) > self.carrier_capacity for batch in self.batches):
            raise ScienceWorldBundleError("physical batch exceeds carrier capacity")

    def packing_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": SCIENCE_WORLD_PACKING_SCHEMA_VERSION,
            "bundle_content_sha256": self.bundle_content_sha256.to_dict(),
            "carrier_id": self.carrier_id,
            "carrier_capacity": self.carrier_capacity,
            "batches": [
                {
                    "physical_batch_index": index,
                    "world_ids": list(batch),
                }
                for index, batch in enumerate(self.batches)
            ],
        }
        if include_identity:
            result["physical_packing_sha256"] = self.packing_sha256().to_dict()
        return result


def pack_admitted_science_worlds_v1(
    bundle: ScienceWorldBundleV1,
    admission: ScienceWorldBundleAdmissionV1,
    *,
    carrier_id: str,
    carrier_capacity: int,
    order: Iterable[str] | None = None,
) -> SciencePhysicalPackingV1:
    if admission.bundle_content_sha256 != bundle.bundle_content_sha256():
        raise ScienceWorldBundleError("admission does not bind the bundle")
    if admission.outcome is not ScienceAdmissionOutcomeV1.ADMITTED:
        raise ScienceWorldBundleError("non-admitted bundle cannot be packed")
    world_ids = list(admission.admitted_world_ids)
    if order is not None:
        candidate = list(order)
        if sorted(candidate) != sorted(world_ids):
            raise ScienceWorldBundleError("packing order is not an exact world permutation")
        world_ids = candidate
    batches = tuple(
        tuple(world_ids[index : index + carrier_capacity])
        for index in range(0, len(world_ids), carrier_capacity)
    )
    return SciencePhysicalPackingV1(
        bundle_content_sha256=bundle.bundle_content_sha256(),
        carrier_id=carrier_id,
        carrier_capacity=carrier_capacity,
        batches=batches,
    )


def demultiplex_world_results_v1(
    bundle: ScienceWorldBundleV1,
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Regroup world-keyed results without inspecting physical placement."""

    by_world = {world.world_id: world for world in bundle.worlds}
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        world_id = row.get("world_id")
        if not isinstance(world_id, str) or world_id not in by_world:
            raise ScienceWorldBundleError("result names an unknown world")
        world = by_world[world_id]
        science = world.science.to_value()
        for reducer in world.reducers.to_value():
            key = "/".join(
                str(item)
                for item in (
                    science["study_id"], science["test_id"], science["arm_id"],
                    science["replicate"], world_id, reducer["reducer_id"],
                )
            )
            groups.setdefault(key, []).append(dict(row))
    return {
        "schema_version": "eiel_science_result_demultiplex_v1",
        "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
        "groups": [
            {"group_key": key, "rows": groups[key]} for key in sorted(groups)
        ],
    }


__all__ = (
    "SCIENCE_WORLD_BUNDLE_COMPILER_ID",
    "SCIENCE_WORLD_BUNDLE_FILE_SCHEMA_V1",
    "SciencePhysicalPackingV1",
    "ScienceWorldAdmissionFindingV1",
    "ScienceWorldBundleAdmissionV1",
    "ScienceWorldBundleError",
    "admit_science_world_bundle_v1",
    "demultiplex_world_results_v1",
    "materialize_science_world_bundle_v1",
    "load_science_world_bundle_v1",
    "pack_admitted_science_worlds_v1",
    "write_science_world_bundle_v1",
)
