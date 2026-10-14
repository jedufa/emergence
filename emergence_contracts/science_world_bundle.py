"""Canonical explicit-world materialization contract for the science instrument."""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import canonical_sha256
from .errors import contract_invalid
from .ids import ContentDigest, SchemaVersion
from .payloads import CanonicalPayloadV1


SCIENCE_WORLD_BUNDLE_SCHEMA = "EIEL_SCIENCE_WORLD_BUNDLE_V1"
SCIENCE_WORLD_BUNDLE_SCHEMA_VERSION = SchemaVersion("eiel_science_world_bundle_v1")
SCIENCE_WORLD_RECORD_SCHEMA_VERSION = SchemaVersion("eiel_science_world_record_v1")
SCIENCE_WORLD_BUNDLE_INSTRUMENT_CONTRACT = "EIEL_SCIENCE_INSTRUMENT_V1"


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise contract_invalid("non-empty text is required", field=field)
    return value


def _digest(value: object, *, field: str) -> ContentDigest:
    if not isinstance(value, dict) or set(value) != {"algorithm", "hex"}:
        raise contract_invalid("digest object is invalid", field=field)
    if value.get("algorithm") != "sha256":
        raise contract_invalid("digest algorithm is invalid", field=field)
    return ContentDigest(algorithm="sha256", hex=value["hex"])


@dataclass(frozen=True, slots=True)
class ScienceWorldRecordV1:
    """One fully resolved scientific unit, free of physical placement."""

    world_id: str
    science: CanonicalPayloadV1
    semantic_inputs: CanonicalPayloadV1
    acquisition: CanonicalPayloadV1
    reducers: CanonicalPayloadV1
    semantic_world_sha256: ContentDigest
    acquisition_sha256: ContentDigest
    schema_version: SchemaVersion = SCIENCE_WORLD_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.world_id, field="world_id")
        if self.schema_version != SCIENCE_WORLD_RECORD_SCHEMA_VERSION:
            raise contract_invalid("world record schema is invalid", field="schema_version")
        science = self.science.to_value()
        if not isinstance(science, dict):
            raise contract_invalid("science ownership must be an object", field="science")
        required = {"test_id", "test_version", "study_id", "arm_id", "replicate"}
        if set(science) != required:
            raise contract_invalid("science ownership fields are not exact", field="science")
        for name in required - {"replicate"}:
            _require_text(science[name], field=f"science.{name}")
        if not isinstance(science["replicate"], (str, int)) or isinstance(
            science["replicate"], bool
        ):
            raise contract_invalid("replicate must be text or integer", field="science.replicate")
        semantic = self.semantic_inputs.to_value()
        if not isinstance(semantic, dict):
            raise contract_invalid("semantic inputs must be an object", field="semantic_inputs")
        required_semantic = {
            "world",
            "initialization",
            "dimension_bindings",
            "control_bindings",
            "seed_bindings",
            "time",
            "initial_state_identity",
        }
        if set(semantic) != required_semantic:
            raise contract_invalid(
                "resolved semantic input fields are not exact", field="semantic_inputs"
            )
        initialization = semantic["initialization"]
        selected_world_initialization = (
            isinstance(initialization, dict)
            and initialization.get("schema_version")
            == "eiel_science_selected_world_initialization_v1"
        )
        if not semantic["seed_bindings"] and not selected_world_initialization:
            raise contract_invalid("resolved seeds are required", field="semantic_inputs.seed_bindings")
        if selected_world_initialization and semantic["seed_bindings"] != {}:
            raise contract_invalid(
                "selected-world initialization must inherit rather than derive RNG seeds",
                field="semantic_inputs.seed_bindings",
            )
        time = semantic["time"]
        if not isinstance(time, dict) or time.get("authority") != "absolute_tick_v1":
            raise contract_invalid("absolute-tick time authority is required", field="semantic_inputs.time")
        acquisition = self.acquisition.to_value()
        if not isinstance(acquisition, dict):
            raise contract_invalid("acquisition must be an object", field="acquisition")
        if set(acquisition) != {
            "profile_id",
            "event_observations",
            "state_observations",
            "runtime_counters",
            "snapshot_ticks",
            "windows",
        }:
            raise contract_invalid("acquisition fields are not exact", field="acquisition")
        reducers = self.reducers.to_value()
        if not isinstance(reducers, list):
            raise contract_invalid("reducers must be an array", field="reducers")
        for index, reducer in enumerate(reducers):
            if not isinstance(reducer, dict) or set(reducer) != {
                "reducer_id",
                "contract_version",
                "required_observations",
                "output_contract_id",
                "configuration",
            }:
                raise contract_invalid("reducer binding fields are not exact", field=f"reducers[{index}]")
            _require_text(reducer["reducer_id"], field=f"reducers[{index}].reducer_id")
        if self.semantic_world_sha256 != self.expected_semantic_world_sha256():
            raise contract_invalid("semantic world identity mismatch", field="semantic_world_sha256")
        if self.acquisition_sha256 != self.expected_acquisition_sha256():
            raise contract_invalid("acquisition identity mismatch", field="acquisition_sha256")

    def expected_semantic_world_sha256(self) -> ContentDigest:
        return canonical_sha256(
            {
                "schema_version": str(SCIENCE_WORLD_RECORD_SCHEMA_VERSION),
                "science": self.science.to_value(),
                "semantic_inputs": self.semantic_inputs.to_value(),
            }
        )

    def expected_acquisition_sha256(self) -> ContentDigest:
        return canonical_sha256(
            {
                "schema_version": str(SCIENCE_WORLD_RECORD_SCHEMA_VERSION),
                "semantic_world_sha256": self.semantic_world_sha256.to_dict(),
                "acquisition": self.acquisition.to_value(),
                "reducers": self.reducers.to_value(),
            }
        )

    def world_contract_sha256(self) -> ContentDigest:
        return canonical_sha256(
            {
                "schema_version": str(self.schema_version),
                "world_id": self.world_id,
                "semantic_world_sha256": self.semantic_world_sha256.to_dict(),
                "acquisition_sha256": self.acquisition_sha256.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "world_id": self.world_id,
            "science": self.science.to_value(),
            "semantic_inputs": self.semantic_inputs.to_value(),
            "acquisition": self.acquisition.to_value(),
            "reducers": self.reducers.to_value(),
            "identities": {
                "semantic_world_sha256": self.semantic_world_sha256.to_dict(),
                "acquisition_sha256": self.acquisition_sha256.to_dict(),
                "world_contract_sha256": self.world_contract_sha256().to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, value: object) -> "ScienceWorldRecordV1":
        if not isinstance(value, dict) or set(value) != {
            "schema_version", "world_id", "science", "semantic_inputs",
            "acquisition", "reducers", "identities",
        }:
            raise contract_invalid("world record fields are not exact", field="world_record")
        identities = value["identities"]
        if not isinstance(identities, dict) or set(identities) != {
            "semantic_world_sha256", "acquisition_sha256", "world_contract_sha256"
        }:
            raise contract_invalid("world identity fields are not exact", field="identities")
        result = cls(
            world_id=value["world_id"],
            science=CanonicalPayloadV1.from_value(SchemaVersion("eiel_science_ownership_v1"), value["science"]),
            semantic_inputs=CanonicalPayloadV1.from_value(SchemaVersion("eiel_science_resolved_inputs_v1"), value["semantic_inputs"]),
            acquisition=CanonicalPayloadV1.from_value(SchemaVersion("eiel_science_acquisition_contract_v1"), value["acquisition"]),
            reducers=CanonicalPayloadV1.from_value(SchemaVersion("eiel_science_world_reducers_v1"), value["reducers"]),
            semantic_world_sha256=_digest(identities["semantic_world_sha256"], field="semantic_world_sha256"),
            acquisition_sha256=_digest(identities["acquisition_sha256"], field="acquisition_sha256"),
            schema_version=SchemaVersion(value["schema_version"]),
        )
        if result.world_contract_sha256() != _digest(
            identities["world_contract_sha256"], field="world_contract_sha256"
        ):
            raise contract_invalid("world contract identity mismatch", field="world_contract_sha256")
        return result


@dataclass(frozen=True, slots=True)
class ScienceWorldBundleV1:
    """Execution truth: a deterministic immutable set of explicit worlds."""

    bundle_id: str
    compact_source_specs: CanonicalPayloadV1
    compiler_component_sha256: ContentDigest
    worlds: tuple[ScienceWorldRecordV1, ...]
    instrument_contract: str = SCIENCE_WORLD_BUNDLE_INSTRUMENT_CONTRACT
    schema: str = SCIENCE_WORLD_BUNDLE_SCHEMA
    schema_version: SchemaVersion = SCIENCE_WORLD_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.bundle_id, field="bundle_id")
        if self.schema != SCIENCE_WORLD_BUNDLE_SCHEMA:
            raise contract_invalid("bundle schema is invalid", field="schema")
        if self.schema_version != SCIENCE_WORLD_BUNDLE_SCHEMA_VERSION:
            raise contract_invalid("bundle schema version is invalid", field="schema_version")
        if self.instrument_contract != SCIENCE_WORLD_BUNDLE_INSTRUMENT_CONTRACT:
            raise contract_invalid("instrument contract is invalid", field="instrument_contract")
        if not self.worlds:
            raise contract_invalid("bundle must contain worlds", field="worlds")
        world_ids = [world.world_id for world in self.worlds]
        if world_ids != sorted(world_ids) or len(world_ids) != len(set(world_ids)):
            raise contract_invalid("world IDs must be sorted and unique", field="worlds")
        semantic_keys = [world.semantic_world_sha256.hex for world in self.worlds]
        if len(semantic_keys) != len(set(semantic_keys)):
            raise contract_invalid("semantic world identities must be unique", field="worlds")

    @property
    def world_count(self) -> int:
        return len(self.worlds)

    def bundle_content_sha256(self) -> ContentDigest:
        return canonical_sha256(
            {
                "schema": self.schema,
                "schema_version": str(self.schema_version),
                "instrument_contract": self.instrument_contract,
                "compact_source_specs": self.compact_source_specs.to_value(),
                "compiler_component_sha256": self.compiler_component_sha256.to_dict(),
                "worlds": [world.to_dict() for world in self.worlds],
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": str(self.schema_version),
            "bundle_id": self.bundle_id,
            "instrument_contract": self.instrument_contract,
            "world_count": self.world_count,
            "compact_source_specs": self.compact_source_specs.to_value(),
            "compiler_component_sha256": self.compiler_component_sha256.to_dict(),
            "worlds": [world.to_dict() for world in self.worlds],
            "bundle_content_sha256": self.bundle_content_sha256().to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ScienceWorldBundleV1":
        if not isinstance(value, dict) or set(value) != {
            "schema", "schema_version", "bundle_id", "instrument_contract",
            "world_count", "compact_source_specs", "compiler_component_sha256",
            "worlds", "bundle_content_sha256",
        }:
            raise contract_invalid("bundle fields are not exact", field="world_bundle")
        worlds_raw = value["worlds"]
        if not isinstance(worlds_raw, list):
            raise contract_invalid("worlds must be an array", field="worlds")
        result = cls(
            bundle_id=value["bundle_id"],
            compact_source_specs=CanonicalPayloadV1.from_value(
                SchemaVersion("eiel_science_compact_source_specs_v1"), value["compact_source_specs"]
            ),
            compiler_component_sha256=_digest(value["compiler_component_sha256"], field="compiler_component_sha256"),
            worlds=tuple(ScienceWorldRecordV1.from_dict(row) for row in worlds_raw),
            instrument_contract=value["instrument_contract"],
            schema=value["schema"],
            schema_version=SchemaVersion(value["schema_version"]),
        )
        if value["world_count"] != result.world_count:
            raise contract_invalid("world count mismatch", field="world_count")
        if _digest(value["bundle_content_sha256"], field="bundle_content_sha256") != result.bundle_content_sha256():
            raise contract_invalid("bundle content identity mismatch", field="bundle_content_sha256")
        return result


__all__ = (
    "SCIENCE_WORLD_BUNDLE_INSTRUMENT_CONTRACT",
    "SCIENCE_WORLD_BUNDLE_SCHEMA",
    "SCIENCE_WORLD_BUNDLE_SCHEMA_VERSION",
    "SCIENCE_WORLD_RECORD_SCHEMA_VERSION",
    "ScienceWorldBundleV1",
    "ScienceWorldRecordV1",
)
