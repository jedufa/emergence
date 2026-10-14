"""Physical CUDA V2 carrier contracts for explicit science worlds.

These records deliberately keep semantic and acquisition identities separate
from native route, backend, device-independent packing, physical B and temporal
K.  They grant neither execution nor Qualification authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Final

from .canonical import canonical_sha256
from .errors import contract_invalid
from .ids import ContentDigest, SchemaVersion


SCIENCE_CUDA_V2_CARRIER_FAMILY_ID: Final[str] = (
    "eiel_science_cuda_full_evolution_v2_carrier_family_v1"
)
SCIENCE_CUDA_V2_PROFILE_SCHEMA_VERSION: Final[SchemaVersion] = SchemaVersion(
    "eiel_science_cuda_v2_carrier_profile_v1"
)
SCIENCE_CUDA_V2_WORLD_MAPPING_SCHEMA_VERSION: Final[SchemaVersion] = SchemaVersion(
    "eiel_science_cuda_v2_world_mapping_v1"
)
SCIENCE_CUDA_V2_PHYSICAL_BATCH_SCHEMA_VERSION: Final[SchemaVersion] = SchemaVersion(
    "eiel_science_cuda_v2_physical_batch_v1"
)
SCIENCE_CUDA_V2_EXECUTION_PLAN_SCHEMA_VERSION: Final[SchemaVersion] = SchemaVersion(
    "eiel_science_cuda_v2_execution_plan_v1"
)
SCIENCE_CUDA_V2_FALLBACK_POLICY: Final[str] = "FORBIDDEN"
SCIENCE_CUDA_V2_NATIVE_WORLD_SCHEMA_VERSION: Final[str] = (
    "eiel_science_cuda_v2_native_world_v1"
)


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise contract_invalid("non-empty text is required", field=field)
    return value


def _positive(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise contract_invalid("positive integer is required", field=field)
    return value


@dataclass(frozen=True, slots=True)
class ScienceCudaV2CarrierProfileV1:
    """One closed physical carrier profile within the single adapter family."""

    carrier_profile_id: str
    native_shape_id: str
    provider_family: str
    provider_id: str
    native_profile_id: str
    native_route_id: str
    width: int
    height: int
    initial_population: int
    capacity: int
    backend_ids: tuple[str, ...]
    temporal_k_values: tuple[int, ...]
    target_physical_batch: int
    schema_version: SchemaVersion = SCIENCE_CUDA_V2_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "carrier_profile_id",
            "native_shape_id",
            "provider_family",
            "provider_id",
            "native_profile_id",
            "native_route_id",
        ):
            _text(getattr(self, name), field=name)
        for name in (
            "width",
            "height",
            "initial_population",
            "capacity",
            "target_physical_batch",
        ):
            _positive(getattr(self, name), field=name)
        expected_profile = (
            f"w{self.width}x{self.height}p{self.initial_population}"
            f"c{self.capacity}_gpuv2"
        )
        if self.carrier_profile_id != expected_profile:
            raise contract_invalid(
                f"carrier profile must equal {expected_profile}",
                field="carrier_profile_id",
            )
        expected_shape = (
            f"w{self.width}h{self.height}p{self.initial_population}c{self.capacity}"
        )
        if self.native_shape_id != expected_shape:
            raise contract_invalid(
                f"native shape must equal {expected_shape}", field="native_shape_id"
            )
        if self.initial_population > min(self.width * self.height, self.capacity):
            raise contract_invalid(
                "initial population exceeds sites or capacity",
                field="initial_population",
            )
        if (
            not self.backend_ids
            or len(self.backend_ids) != len(set(self.backend_ids))
            or any(not isinstance(value, str) or not value for value in self.backend_ids)
        ):
            raise contract_invalid("backend IDs must be unique text", field="backend_ids")
        if (
            not self.temporal_k_values
            or self.temporal_k_values != tuple(sorted(set(self.temporal_k_values)))
            or self.temporal_k_values[0] != 1
            or any(type(value) is not int or value <= 0 for value in self.temporal_k_values)
        ):
            raise contract_invalid(
                "temporal K values must be unique ascending integers beginning at 1",
                field="temporal_k_values",
            )
        if self.schema_version != SCIENCE_CUDA_V2_PROFILE_SCHEMA_VERSION:
            raise contract_invalid("profile schema changed", field="schema_version")

    def profile_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": str(self.schema_version),
            "carrier_family_id": SCIENCE_CUDA_V2_CARRIER_FAMILY_ID,
            "carrier_profile_id": self.carrier_profile_id,
            "native_shape_id": self.native_shape_id,
            "provider_family": self.provider_family,
            "provider_id": self.provider_id,
            "native_profile_id": self.native_profile_id,
            "native_route_id": self.native_route_id,
            "geometry": {
                "width": self.width,
                "height": self.height,
                "initial_population": self.initial_population,
                "capacity": self.capacity,
            },
            "backend_ids": list(self.backend_ids),
            "temporal_k_values": list(self.temporal_k_values),
            "target_physical_batch": self.target_physical_batch,
            "batch_is_semantic": False,
        }
        if include_identity:
            value["carrier_profile_sha256"] = self.profile_sha256().to_dict()
        return value


@dataclass(frozen=True, slots=True)
class ScienceCudaV2WorldMappingV1:
    """Stable semantic-to-numeric mapping; no physical slot is present."""

    world_id: str
    world_uid: int
    semantic_world_sha256: ContentDigest
    acquisition_sha256: ContentDigest
    requested_observations: tuple[str, ...]
    schema_version: SchemaVersion = SCIENCE_CUDA_V2_WORLD_MAPPING_SCHEMA_VERSION
    # Native RNG identity is deliberately separate from the arm-specific
    # semantic world digest.  These fields are optional for legacy host-only
    # carrier projections; native-bound science plans populate all of them.
    native_rng_world_uid: int | None = None
    native_rng_identity_sha256: ContentDigest | None = None
    native_control_vector: dict[str, object] | None = None
    native_control_digest: ContentDigest | None = None
    native_capability_mask: int | None = None
    native_world_control: dict[str, object] | None = None

    def __post_init__(self) -> None:
        _text(self.world_id, field="world_id")
        if type(self.world_uid) is not int or not 0 < self.world_uid < (1 << 64):
            raise contract_invalid("world UID must fit nonzero u64", field="world_uid")
        if (
            self.requested_observations
            != tuple(sorted(set(self.requested_observations)))
            or any(not isinstance(value, str) or not value for value in self.requested_observations)
        ):
            raise contract_invalid(
                "requested observations must be unique sorted text",
                field="requested_observations",
            )
        if (self.native_rng_world_uid is None) != (
            self.native_rng_identity_sha256 is None
        ):
            raise contract_invalid(
                "native RNG UID and identity digest must be supplied together",
                field="native_rng_world_uid",
            )
        if self.native_rng_world_uid is not None:
            if type(self.native_rng_world_uid) is not int or not 0 < self.native_rng_world_uid <= (
                1 << 63
            ) - 1:
                raise contract_invalid(
                    "native RNG UID must fit positive signed int64",
                    field="native_rng_world_uid",
                )
            if not isinstance(self.native_rng_identity_sha256, ContentDigest):
                raise contract_invalid(
                    "native RNG identity must be a content digest",
                    field="native_rng_identity_sha256",
                )
        control_fields = (
            self.native_control_vector,
            self.native_control_digest,
            self.native_capability_mask,
        )
        if any(field is not None for field in control_fields) and not all(
            field is not None for field in control_fields
        ):
            raise contract_invalid(
                "native control vector, digest, and capability mask are atomic",
                field="native_control_vector",
            )
        if self.native_control_vector is not None:
            vector = self.native_control_vector
            if not isinstance(vector, Mapping) or set(vector) != {"mutation.enabled"} or type(
                vector["mutation.enabled"]
            ) is not bool:
                raise contract_invalid(
                    "native control vector is not the closed mutation contract",
                    field="native_control_vector",
                )
            expected_mask = 127 if vector["mutation.enabled"] else 95
            if self.native_capability_mask != expected_mask:
                raise contract_invalid(
                    "native capability mask differs from mutation.enabled",
                    field="native_capability_mask",
                )
            expected_digest = canonical_sha256(
                {
                    "schema_version": "eiel_science_cuda_v2_native_control_v1",
                    "capability_mask": self.native_capability_mask,
                    "control_vector": vector,
                }
            )
            if self.native_control_digest != expected_digest:
                raise contract_invalid(
                    "native control digest does not match applied vector",
                    field="native_control_digest",
                )
        if self.native_world_control is not None:
            value = self.native_world_control
            required = {
                "schema_version",
                "semantic_inputs_sha256",
                "science_ownership_sha256",
                "capability_mask_at_start",
                "world",
                "initialization",
                "seed_bindings",
                "control_bindings",
                "time",
                "control_digest",
            }
            if not isinstance(value, Mapping) or set(value) != required:
                raise contract_invalid(
                    "native world control fields are not exact",
                    field="native_world_control",
                )
            if value.get("schema_version") != SCIENCE_CUDA_V2_NATIVE_WORLD_SCHEMA_VERSION:
                raise contract_invalid(
                    "native world control schema changed",
                    field="native_world_control.schema_version",
                )
            mask = value.get("capability_mask_at_start")
            if type(mask) is not int or not 0 <= mask <= 127:
                raise contract_invalid(
                    "native world capability mask is invalid",
                    field="native_world_control.capability_mask_at_start",
                )
            for name in (
                "semantic_inputs_sha256",
                "science_ownership_sha256",
                "control_digest",
            ):
                digest = value.get(name)
                if (
                    not isinstance(digest, Mapping)
                    or set(digest) != {"algorithm", "hex"}
                    or digest.get("algorithm") != "sha256"
                ):
                    raise contract_invalid(
                        "native world digest is invalid",
                        field=f"native_world_control.{name}",
                    )
            body = {
                key: value[key]
                for key in required
                if key != "control_digest"
            }
            expected_world_digest = canonical_sha256(body)
            if value["control_digest"] != expected_world_digest.to_dict():
                raise contract_invalid(
                    "native world control digest does not match its body",
                    field="native_world_control.control_digest",
                )
        if self.schema_version != SCIENCE_CUDA_V2_WORLD_MAPPING_SCHEMA_VERSION:
            raise contract_invalid("world mapping schema changed", field="schema_version")

    def mapping_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": str(self.schema_version),
            "world_id": self.world_id,
            "world_uid": self.world_uid,
            "semantic_world_sha256": self.semantic_world_sha256.to_dict(),
            "acquisition_sha256": self.acquisition_sha256.to_dict(),
            "requested_observations": list(self.requested_observations),
        }
        if self.native_rng_world_uid is not None:
            value.update(
                {
                    "native_rng_world_uid": self.native_rng_world_uid,
                    "native_rng_identity_sha256": self.native_rng_identity_sha256.to_dict(),
                }
            )
        if self.native_control_vector is not None:
            value["native_control"] = {
                "schema_version": "eiel_science_cuda_v2_native_control_v1",
                "capability_mask": self.native_capability_mask,
                "control_vector": dict(self.native_control_vector),
                "control_digest": self.native_control_digest.to_dict(),
            }
        if self.native_world_control is not None:
            value["native_world_control"] = dict(self.native_world_control)
        if include_identity:
            value["world_mapping_sha256"] = self.mapping_sha256().to_dict()
        return value


@dataclass(frozen=True, slots=True)
class ScienceCudaV2PhysicalBatchV1:
    """One physical native launch partition; all placement is explicit here."""

    physical_batch_index: int
    carrier_profile: ScienceCudaV2CarrierProfileV1
    backend_id: str
    temporal_k: int
    world_slots: tuple[tuple[int, ScienceCudaV2WorldMappingV1], ...]
    native_observation_union: tuple[str, ...]
    schema_version: SchemaVersion = SCIENCE_CUDA_V2_PHYSICAL_BATCH_SCHEMA_VERSION
    native_control_vector: dict[str, object] | None = None
    native_control_digest: ContentDigest | None = None
    native_capability_mask: int | None = None

    def __post_init__(self) -> None:
        if type(self.physical_batch_index) is not int or self.physical_batch_index < 0:
            raise contract_invalid(
                "physical batch index must be nonnegative", field="physical_batch_index"
            )
        if self.backend_id not in self.carrier_profile.backend_ids:
            raise contract_invalid("backend is not admitted by profile", field="backend_id")
        if self.temporal_k not in self.carrier_profile.temporal_k_values:
            raise contract_invalid("temporal K is not admitted by profile", field="temporal_k")
        if not self.world_slots:
            raise contract_invalid("physical batch cannot be empty", field="world_slots")
        slots = [slot for slot, _ in self.world_slots]
        if slots != list(range(len(slots))):
            raise contract_invalid("physical slots must be contiguous from zero", field="world_slots")
        ids = [world.world_id for _, world in self.world_slots]
        uids = [world.world_uid for _, world in self.world_slots]
        if len(ids) != len(set(ids)) or len(uids) != len(set(uids)):
            raise contract_invalid("physical batch duplicates world identity", field="world_slots")
        native_rng_bindings: dict[int, tuple[ContentDigest, set[str]]] = {}
        for _, world in self.world_slots:
            if world.native_rng_world_uid is None:
                continue
            if world.native_rng_identity_sha256 is None:
                raise contract_invalid(
                    "native RNG identity digest is absent", field="world_slots"
                )
            control_digest = (
                str(world.native_world_control["control_digest"]["hex"])
                if world.native_world_control is not None
                else (
                    world.native_control_digest.hex
                    if world.native_control_digest is not None
                    else ""
                )
            )
            prior = native_rng_bindings.get(world.native_rng_world_uid)
            if prior is None:
                native_rng_bindings[world.native_rng_world_uid] = (
                    world.native_rng_identity_sha256,
                    {control_digest},
                )
                continue
            if (
                prior[0] != world.native_rng_identity_sha256
                or not control_digest
                or control_digest in prior[1]
            ):
                raise contract_invalid(
                    "physical batch duplicates native RNG identity outside paired controls",
                    field="world_slots",
                )
            prior[1].add(control_digest)
        control_fields = (
            self.native_control_vector,
            self.native_control_digest,
            self.native_capability_mask,
        )
        if any(field is not None for field in control_fields) and not all(
            field is not None for field in control_fields
        ):
            raise contract_invalid(
                "physical batch native control fields are not atomic",
                field="native_control_vector",
            )
        if any(
            world.native_control_vector is not None
            and not isinstance(world.native_control_vector, Mapping)
            for _, world in self.world_slots
        ):
            raise contract_invalid(
                "world native control vectors must be objects",
                field="world_slots",
            )
        mapped_controls = {
            (
                tuple(sorted(world.native_control_vector.items()))
                if world.native_control_vector is not None
                else None,
                world.native_control_digest,
                world.native_capability_mask,
            )
            for _, world in self.world_slots
        }
        mapped_control = next(iter(mapped_controls)) if len(mapped_controls) == 1 else None
        if self.native_control_vector is None:
            # A batch-level scalar receipt is a legacy optimization only.  A
            # heterogeneous native batch deliberately omits it and carries the
            # authoritative control independently on every logical world.
            pass
        else:
            expected_batch_control = (
                tuple(sorted(self.native_control_vector.items())),
                self.native_control_digest,
                self.native_capability_mask,
            )
            if mapped_control != expected_batch_control:
                raise contract_invalid(
                    "physical batch native control differs from uniform world mappings",
                    field="native_control_vector",
                )
        if any(
            world.native_world_control is None for _, world in self.world_slots
        ) and any(
            world.native_world_control is not None for _, world in self.world_slots
        ):
            raise contract_invalid(
                "physical batch partially binds native world controls",
                field="world_slots",
            )
        expected_union = tuple(
            sorted(
                {
                    observation
                    for _, world in self.world_slots
                    for observation in world.requested_observations
                }
            )
        )
        if self.native_observation_union != expected_union:
            raise contract_invalid(
                "native observation union differs from per-world requests",
                field="native_observation_union",
            )
        if self.schema_version != SCIENCE_CUDA_V2_PHYSICAL_BATCH_SCHEMA_VERSION:
            raise contract_invalid("physical batch schema changed", field="schema_version")

    @property
    def physical_batch_size(self) -> int:
        return len(self.world_slots)

    def batch_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": str(self.schema_version),
            "physical_batch_index": self.physical_batch_index,
            "carrier_profile": self.carrier_profile.to_dict(),
            "backend_id": self.backend_id,
            "temporal_k": self.temporal_k,
            "physical_batch_size": self.physical_batch_size,
            "world_slots": [
                {"physical_slot": slot, "world": world.to_dict()}
                for slot, world in self.world_slots
            ],
            "native_observation_union": list(self.native_observation_union),
        }
        if self.native_control_vector is not None:
            value["native_control"] = {
                "schema_version": "eiel_science_cuda_v2_native_control_v1",
                "capability_mask": self.native_capability_mask,
                "control_vector": dict(self.native_control_vector),
                "control_digest": self.native_control_digest.to_dict(),
            }
        if include_identity:
            value["physical_batch_sha256"] = self.batch_sha256().to_dict()
        return value


@dataclass(frozen=True, slots=True)
class ScienceCudaV2ExecutionPlanV1:
    """A complete physical plan bound to an admitted explicit bundle."""

    bundle_content_sha256: ContentDigest
    bundle_admission_sha256: ContentDigest
    carrier_profile_catalog_sha256: ContentDigest
    logical_bundle_size: int
    batches: tuple[ScienceCudaV2PhysicalBatchV1, ...]
    execution_quantum_id: str | None = None
    fallback_policy: str = SCIENCE_CUDA_V2_FALLBACK_POLICY
    schema_version: SchemaVersion = SCIENCE_CUDA_V2_EXECUTION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _positive(self.logical_bundle_size, field="logical_bundle_size")
        if not self.batches:
            raise contract_invalid("execution plan cannot be empty", field="batches")
        indices = [batch.physical_batch_index for batch in self.batches]
        if indices != list(range(len(indices))):
            raise contract_invalid("batch indices must be contiguous", field="batches")
        mappings = [world for batch in self.batches for _, world in batch.world_slots]
        if len(mappings) != self.logical_bundle_size:
            raise contract_invalid("logical bundle size differs from mapping count", field="logical_bundle_size")
        ids = [world.world_id for world in mappings]
        uids = [world.world_uid for world in mappings]
        if len(ids) != len(set(ids)) or len(uids) != len(set(uids)):
            raise contract_invalid("execution plan duplicates world identity", field="batches")
        # A paired OFF/ON experiment intentionally reuses one *native* RNG UID
        # in two differently masked batches while retaining distinct semantic
        # world UIDs.  Same-control or unbound native collisions remain
        # invalid, as do collisions with different RNG identity digests.
        uid_bindings: dict[
            int, tuple[ContentDigest | None, ContentDigest | None, str]
        ] = {}
        for world in mappings:
            if world.native_rng_world_uid is None:
                continue
            prior = uid_bindings.get(world.native_rng_world_uid)
            current = (
                (
                    ContentDigest(
                        algorithm="sha256",
                        hex=world.native_world_control["control_digest"]["hex"],
                    )
                    if world.native_world_control is not None
                    else world.native_control_digest
                ),
                world.native_rng_identity_sha256,
                world.world_id,
            )
            if prior is not None:
                if (
                    prior[1] is None
                    or current[1] is None
                    or prior[1] != current[1]
                    or prior[0] is None
                    or current[0] is None
                    or prior[0] == current[0]
                ):
                    raise contract_invalid(
                        "execution plan duplicates native UID outside paired control batches",
                        field="batches",
                    )
            else:
                uid_bindings[world.native_rng_world_uid] = current
        if self.execution_quantum_id is not None:
            _text(self.execution_quantum_id, field="execution_quantum_id")
        if self.fallback_policy != SCIENCE_CUDA_V2_FALLBACK_POLICY:
            raise contract_invalid("fallback policy must be FORBIDDEN", field="fallback_policy")
        if self.schema_version != SCIENCE_CUDA_V2_EXECUTION_PLAN_SCHEMA_VERSION:
            raise contract_invalid("execution plan schema changed", field="schema_version")

    def plan_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": str(self.schema_version),
            "carrier_family_id": SCIENCE_CUDA_V2_CARRIER_FAMILY_ID,
            "bundle_content_sha256": self.bundle_content_sha256.to_dict(),
            "bundle_admission_sha256": self.bundle_admission_sha256.to_dict(),
            "carrier_profile_catalog_sha256": self.carrier_profile_catalog_sha256.to_dict(),
            "logical_bundle_size": self.logical_bundle_size,
            "execution_quantum_id": self.execution_quantum_id,
            "fallback_policy": self.fallback_policy,
            "batches": [batch.to_dict() for batch in self.batches],
        }
        if include_identity:
            value["physical_execution_plan_sha256"] = self.plan_sha256().to_dict()
        return value


__all__ = (
    "SCIENCE_CUDA_V2_CARRIER_FAMILY_ID",
    "SCIENCE_CUDA_V2_EXECUTION_PLAN_SCHEMA_VERSION",
    "SCIENCE_CUDA_V2_FALLBACK_POLICY",
    "SCIENCE_CUDA_V2_NATIVE_WORLD_SCHEMA_VERSION",
    "SCIENCE_CUDA_V2_PHYSICAL_BATCH_SCHEMA_VERSION",
    "SCIENCE_CUDA_V2_PROFILE_SCHEMA_VERSION",
    "SCIENCE_CUDA_V2_WORLD_MAPPING_SCHEMA_VERSION",
    "ScienceCudaV2CarrierProfileV1",
    "ScienceCudaV2ExecutionPlanV1",
    "ScienceCudaV2PhysicalBatchV1",
    "ScienceCudaV2WorldMappingV1",
)
