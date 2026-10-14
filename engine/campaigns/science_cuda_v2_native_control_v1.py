"""Closed native-control projection at the CUDA V2 campaign boundary.

The science instrument intentionally has a richer control vocabulary than the
native Full Evolution V2 ABI.  This module is the small, explicit projection
between the two: the only semantic control currently admitted is a static
boolean ``mutation.enabled`` binding.  Everything else is rejected by the
strict entry point before a native launch is allowed.

Native RNG identity is kept separate from the arm-specific semantic world
identity.  It is derived from the materialized initial-state identity and the
explicit science test/study/replicate ownership, deliberately excluding the
arm so paired OFF/ON worlds address the same native RNG subject.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping

from .canonical import canonical_sha256


SCIENCE_CUDA_V2_NATIVE_CONTROL_SCHEMA_VERSION: Final[str] = (
    "eiel_science_cuda_v2_native_control_v1"
)
SCIENCE_CUDA_V2_MUTATION_CONTROL_ID: Final[str] = "mutation.enabled"
SCIENCE_CUDA_V2_MUTATION_CAPABILITY_BIT: Final[int] = 1 << 5
SCIENCE_CUDA_V2_FULL_CAPABILITY_MASK: Final[int] = (1 << 7) - 1
SCIENCE_CUDA_V2_MUTATION_OFF_CAPABILITY_MASK: Final[int] = (
    SCIENCE_CUDA_V2_FULL_CAPABILITY_MASK & ~SCIENCE_CUDA_V2_MUTATION_CAPABILITY_BIT
)


class ScienceCudaV2NativeControlError(ValueError):
    """A science semantic control cannot be represented by the native ABI."""


def _semantic_inputs(value: object) -> Mapping[str, Any]:
    if hasattr(value, "semantic_inputs"):
        value = getattr(value, "semantic_inputs")
    if hasattr(value, "to_value"):
        value = value.to_value()
    if not isinstance(value, Mapping):
        raise ScienceCudaV2NativeControlError(
            "native control projection requires semantic inputs"
        )
    # Accept a world record or its already-unwrapped semantic_inputs payload.
    candidate = value.get("semantic_inputs")
    if isinstance(candidate, Mapping):
        value = candidate
    return value


def _science_ownership(value: object) -> Mapping[str, Any]:
    if hasattr(value, "science"):
        value = getattr(value, "science")
    if hasattr(value, "to_value"):
        value = value.to_value()
    if not isinstance(value, Mapping):
        raise ScienceCudaV2NativeControlError(
            "native RNG projection requires science ownership"
        )
    candidate = value.get("science")
    if isinstance(candidate, Mapping):
        value = candidate
    required = {"test_id", "test_version", "study_id", "arm_id", "replicate"}
    if set(value) != required:
        raise ScienceCudaV2NativeControlError(
            "science ownership is not the exact materialized contract"
        )
    return value


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"algorithm", "hex"}
        or value.get("algorithm") != "sha256"
    ):
        raise ScienceCudaV2NativeControlError(f"{field} is not a digest object")
    digest = value.get("hex")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ScienceCudaV2NativeControlError(f"{field} is not a SHA-256 digest")
    return digest


def _control_rows(value: object) -> tuple[Mapping[str, Any], ...]:
    semantic = _semantic_inputs(value)
    required = {
        "world",
        "initialization",
        "dimension_bindings",
        "control_bindings",
        "seed_bindings",
        "time",
        "initial_state_identity",
    }
    if set(semantic) != required:
        raise ScienceCudaV2NativeControlError(
            "resolved semantic input fields are not the exact materialized contract"
        )
    rows = semantic.get("control_bindings")
    if not isinstance(rows, list):
        raise ScienceCudaV2NativeControlError(
            "semantic control_bindings must be an array"
        )
    result: list[Mapping[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ScienceCudaV2NativeControlError(
                f"control_bindings[{index}] is not an object"
            )
        result.append(raw)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ScienceCudaV2NativeControlV1:
    """One exact scalar capability-mask projection for a physical batch."""

    mutation_enabled: bool
    capability_mask: int
    control_digest: str

    def __post_init__(self) -> None:
        if type(self.mutation_enabled) is not bool:
            raise ScienceCudaV2NativeControlError(
                "mutation.enabled must resolve to a boolean"
            )
        expected = (
            SCIENCE_CUDA_V2_FULL_CAPABILITY_MASK
            if self.mutation_enabled
            else SCIENCE_CUDA_V2_MUTATION_OFF_CAPABILITY_MASK
        )
        if type(self.capability_mask) is not int or self.capability_mask != expected:
            raise ScienceCudaV2NativeControlError(
                "native capability mask does not match mutation.enabled"
            )
        expected_digest = canonical_sha256(self._digest_body())
        if self.control_digest != expected_digest:
            raise ScienceCudaV2NativeControlError(
                "native control digest does not match the applied vector"
            )

    @property
    def control_vector(self) -> dict[str, bool]:
        return {SCIENCE_CUDA_V2_MUTATION_CONTROL_ID: self.mutation_enabled}

    def _digest_body(self) -> dict[str, Any]:
        return {
            "schema_version": SCIENCE_CUDA_V2_NATIVE_CONTROL_SCHEMA_VERSION,
            "capability_mask": self.capability_mask,
            "control_vector": self.control_vector,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._digest_body(),
            "control_digest": {"algorithm": "sha256", "hex": self.control_digest},
        }

    @classmethod
    def from_values(
        cls, *, mutation_enabled: bool, capability_mask: int | None = None
    ) -> "ScienceCudaV2NativeControlV1":
        if type(mutation_enabled) is not bool:
            raise ScienceCudaV2NativeControlError(
                "mutation.enabled must resolve to a boolean"
            )
        expected_mask = (
            SCIENCE_CUDA_V2_FULL_CAPABILITY_MASK
            if mutation_enabled
            else SCIENCE_CUDA_V2_MUTATION_OFF_CAPABILITY_MASK
        )
        if capability_mask is not None and capability_mask != expected_mask:
            raise ScienceCudaV2NativeControlError(
                "supplied native capability mask differs from mutation.enabled"
            )
        body = {
            "schema_version": SCIENCE_CUDA_V2_NATIVE_CONTROL_SCHEMA_VERSION,
            "capability_mask": expected_mask,
            "control_vector": {SCIENCE_CUDA_V2_MUTATION_CONTROL_ID: mutation_enabled},
        }
        return cls(
            mutation_enabled=mutation_enabled,
            capability_mask=expected_mask,
            control_digest=canonical_sha256(body),
        )

    @classmethod
    def from_dict(cls, value: object) -> "ScienceCudaV2NativeControlV1":
        if not isinstance(value, Mapping):
            raise ScienceCudaV2NativeControlError(
                "native control receipt is not an object"
            )
        required = {
            "schema_version",
            "capability_mask",
            "control_vector",
            "control_digest",
        }
        if set(value) != required:
            raise ScienceCudaV2NativeControlError(
                "native control receipt fields are not exact"
            )
        if value.get("schema_version") != SCIENCE_CUDA_V2_NATIVE_CONTROL_SCHEMA_VERSION:
            raise ScienceCudaV2NativeControlError(
                "native control schema is unsupported"
            )
        vector = value.get("control_vector")
        if not isinstance(vector, Mapping) or set(vector) != {
            SCIENCE_CUDA_V2_MUTATION_CONTROL_ID
        }:
            raise ScienceCudaV2NativeControlError("native control vector is not exact")
        mutation_enabled = vector[SCIENCE_CUDA_V2_MUTATION_CONTROL_ID]
        digest = _digest(value.get("control_digest"), field="control_digest")
        return cls(
            mutation_enabled=mutation_enabled,
            capability_mask=value.get("capability_mask"),
            control_digest=digest,
        )


def project_science_cuda_v2_native_control_v1(
    value: object,
) -> ScienceCudaV2NativeControlV1:
    """Strictly project resolved controls into the scalar native mask.

    An empty control list means the existing native default (all capabilities)
    and is therefore represented explicitly as mutation ON.  A non-empty list
    must contain exactly one static boolean ``mutation.enabled`` row; numeric
    rates, schedules, recombination/founder/environment/selection controls, and
    unknown controls all fail closed.
    """

    rows = _control_rows(value)
    if not rows:
        return ScienceCudaV2NativeControlV1.from_values(mutation_enabled=True)
    mutation: Mapping[str, Any] | None = None
    for index, row in enumerate(rows):
        control_id = row.get("control_id")
        if control_id != SCIENCE_CUDA_V2_MUTATION_CONTROL_ID:
            raise ScienceCudaV2NativeControlError(
                f"unsupported native science control {control_id!r} at control_bindings[{index}]"
            )
        if mutation is not None:
            raise ScienceCudaV2NativeControlError(
                "mutation.enabled is duplicated in native science controls"
            )
        mutation = row
    assert mutation is not None  # rows is non-empty and all rows are checked above
    if set(mutation) != {
        "control_id",
        "resolved_value",
        "schedule",
        "descriptor_sha256",
    }:
        raise ScienceCudaV2NativeControlError(
            "mutation.enabled binding fields are not exact"
        )
    schedule = mutation.get("schedule")
    if schedule != {"kind": "static"}:
        raise ScienceCudaV2NativeControlError(
            "mutation.enabled requires the static schedule only"
        )
    resolved = mutation.get("resolved_value")
    if type(resolved) is not bool:
        raise ScienceCudaV2NativeControlError(
            "mutation.enabled must resolve to a boolean; numeric rates are unsupported"
        )
    _digest(
        mutation.get("descriptor_sha256"), field="mutation.enabled descriptor_sha256"
    )
    return ScienceCudaV2NativeControlV1.from_values(mutation_enabled=resolved)


def derive_science_cuda_v2_native_rng_identity_v1(
    value: object,
) -> tuple[int, str]:
    """Derive the arm-independent native RNG subject and its binding digest."""

    semantic = _semantic_inputs(value)
    initial_identity = _digest(
        semantic.get("initial_state_identity"), field="initial_state_identity"
    )
    world = semantic.get("world")
    initialization = semantic.get("initialization")
    seed_bindings = semantic.get("seed_bindings")
    if not isinstance(world, Mapping) or not isinstance(initialization, Mapping):
        raise ScienceCudaV2NativeControlError(
            "native RNG identity requires world and initialization objects"
        )
    if not isinstance(seed_bindings, Mapping) or not seed_bindings:
        raise ScienceCudaV2NativeControlError(
            "native RNG identity requires resolved seed bindings"
        )
    expected_initial = canonical_sha256(
        {
            "world": world,
            "initialization": initialization,
            "seed_bindings": seed_bindings,
        }
    )
    if initial_identity != expected_initial:
        raise ScienceCudaV2NativeControlError(
            "initial_state_identity does not match world, initialization, and seeds"
        )
    science = _science_ownership(value)
    identity_body = {
        "schema_version": SCIENCE_CUDA_V2_NATIVE_CONTROL_SCHEMA_VERSION,
        "initial_state_identity": {"algorithm": "sha256", "hex": initial_identity},
        # arm_id is intentionally absent: paired OFF/ON worlds share this UID.
        "science_test": {
            "test_id": science["test_id"],
            "test_version": science["test_version"],
        },
        "science_study": science["study_id"],
        "replicate": science["replicate"],
    }
    identity_digest = canonical_sha256(identity_body)
    native_uid = int(identity_digest[:16], 16) & ((1 << 63) - 1)
    if native_uid == 0:
        raise ScienceCudaV2NativeControlError(
            "native RNG identity produced reserved UID zero"
        )
    return native_uid, identity_digest


__all__ = (
    "SCIENCE_CUDA_V2_FULL_CAPABILITY_MASK",
    "SCIENCE_CUDA_V2_MUTATION_CAPABILITY_BIT",
    "SCIENCE_CUDA_V2_MUTATION_CONTROL_ID",
    "SCIENCE_CUDA_V2_MUTATION_OFF_CAPABILITY_MASK",
    "SCIENCE_CUDA_V2_NATIVE_CONTROL_SCHEMA_VERSION",
    "ScienceCudaV2NativeControlError",
    "ScienceCudaV2NativeControlV1",
    "derive_science_cuda_v2_native_rng_identity_v1",
    "project_science_cuda_v2_native_control_v1",
)
