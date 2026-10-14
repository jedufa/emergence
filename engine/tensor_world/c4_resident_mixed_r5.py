"""CPU-only preparation surfaces for the C4 R5 mixed-world rung.

This module deliberately contains no resident/native executor.  It creates a
true B4 oracle fixture and an exact tensor-leaf comparator so a future resident
R5 transaction can be judged at one full-authoritative boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from typing import Final, Mapping

import torch

from .c4_fixture import make_c4_fixture_v1
from .c4_state import TensorWorldStateBundleV4
from .c5_determinism import (
    TensorWorldC5ReplayContextV1,
    make_c5_pcg64_context_v1,
)
from .contracts import TensorWorldContractError


C4_RESIDENT_MIXED_R5_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_mixed_r5_preparation_v1"
)
C4_RESIDENT_MIXED_R5_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_MIXED_R5_COUNTS_V1: Final[tuple[int, int, int, int]] = (0, 1, 2, 1)
C4_RESIDENT_MIXED_R5_IDENTITY_LIMITATION_V1: Final[str] = (
    "fixture_only_world_local_biological_ids_repeat_across_worlds; "
    "the capacity-sized identity map cannot represent globally unique B4 IDs"
)


@dataclass(frozen=True, slots=True)
class C4ResidentMixedR5FixtureV1:
    bundle: TensorWorldStateBundleV4
    replay: TensorWorldC5ReplayContextV1
    expected_transaction_counts: tuple[int, int, int, int]
    identity_map_limitation: str
    schema_version: str = C4_RESIDENT_MIXED_R5_SCHEMA_VERSION_V1
    decision: str = C4_RESIDENT_MIXED_R5_DECISION_V1
    gate_claim: bool = False

    def validate(self) -> None:
        if self.schema_version != C4_RESIDENT_MIXED_R5_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 R5 fixture schema is unsupported")
        if self.bundle.batch != 4 or self.bundle.device.type != "cpu":
            raise TensorWorldContractError("C4 R5 preparation fixture requires CPU B4")
        if self.expected_transaction_counts != C4_RESIDENT_MIXED_R5_COUNTS_V1:
            raise TensorWorldContractError("C4 R5 transaction counts diverge")
        if self.decision != C4_RESIDENT_MIXED_R5_DECISION_V1 or self.gate_claim:
            raise TensorWorldContractError("C4 R5 preparation cannot claim a gate")
        self.bundle.validate()
        self.replay.validate_pcg64_for_bundle(self.bundle)


@dataclass(frozen=True, slots=True)
class C4ResidentMixedR5AuthoritativeSnapshotV1:
    tensors: Mapping[str, torch.Tensor]
    digest: str
    schema_version: str = C4_RESIDENT_MIXED_R5_SCHEMA_VERSION_V1
    gate_claim: bool = False

    def validate(self) -> None:
        if self.schema_version != C4_RESIDENT_MIXED_R5_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 R5 snapshot schema is unsupported")
        if not self.tensors or self.gate_claim:
            raise TensorWorldContractError("C4 R5 snapshot is incomplete")
        if len(self.digest) != 64:
            raise TensorWorldContractError("C4 R5 snapshot digest is invalid")
        for path, tensor in self.tensors.items():
            if not path or tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise TensorWorldContractError("C4 R5 snapshot tensor is invalid")


def make_c4_resident_mixed_r5_fixture_v1(
    *, capacity: int = 8, active: int = 3, width: int = 8, seed: int = 29
) -> C4ResidentMixedR5FixtureV1:
    """Create B4 with deterministic world-local transaction counts 0/1/2/1."""

    if capacity < 8 or active < 3:
        raise TensorWorldContractError("C4 R5 fixture requires capacity>=8 and active>=3")
    bundle, _ = make_c4_fixture_v1(
        batch=4,
        capacity=capacity,
        active=active,
        width=width,
        seed=seed,
        world_local_identity=True,
    )
    lifecycle = bundle.base.lifecycle
    health = bundle.base.base.core.agents.health
    lifecycle.death_candidate.zero_()
    lifecycle.death_candidate[1, 1] = True
    lifecycle.death_candidate[2, 1:3] = True
    lifecycle.death_candidate[3, 0] = True
    health[1, 1] = 0.0
    health[2, 2] = 0.0
    health[3, 0] = 0.0
    replay = make_c5_pcg64_context_v1(bundle, streams=2, seed=seed)
    fixture = C4ResidentMixedR5FixtureV1(
        bundle=bundle,
        replay=replay,
        expected_transaction_counts=C4_RESIDENT_MIXED_R5_COUNTS_V1,
        identity_map_limitation=C4_RESIDENT_MIXED_R5_IDENTITY_LIMITATION_V1,
    )
    fixture.validate()
    return fixture


def _collect_tensor_leaves_v1(
    value: object, *, path: str, output: dict[str, torch.Tensor]
) -> None:
    if isinstance(value, torch.Tensor):
        output[path] = value.detach().clone().contiguous()
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _collect_tensor_leaves_v1(
                getattr(value, field.name), path=f"{path}.{field.name}", output=output
            )


def capture_c4_resident_mixed_r5_authoritative_v1(
    bundle: TensorWorldStateBundleV4,
    replay: TensorWorldC5ReplayContextV1,
) -> C4ResidentMixedR5AuthoritativeSnapshotV1:
    """Capture every tensor leaf owned by the C4 bundle and C5 handoff."""

    bundle.validate()
    replay.validate_pcg64_for_bundle(bundle)
    if bundle.device.type != "cpu":
        raise TensorWorldContractError("C4 R5 authoritative capture is CPU-only")
    tensors: dict[str, torch.Tensor] = {}
    _collect_tensor_leaves_v1(bundle, path="bundle", output=tensors)
    _collect_tensor_leaves_v1(replay, path="replay", output=tensors)
    digest = sha256()
    for path, tensor in tensors.items():
        digest.update(path.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    snapshot = C4ResidentMixedR5AuthoritativeSnapshotV1(
        tensors=tensors, digest=digest.hexdigest()
    )
    snapshot.validate()
    return snapshot


def compare_c4_resident_mixed_r5_authoritative_v1(
    expected: C4ResidentMixedR5AuthoritativeSnapshotV1,
    actual: C4ResidentMixedR5AuthoritativeSnapshotV1,
) -> dict[str, object]:
    """Byte-compare full authoritative tensor leaves without tolerance."""

    expected.validate()
    actual.validate()
    mismatches: list[dict[str, object]] = []
    paths = tuple(dict.fromkeys((*expected.tensors, *actual.tensors)))
    for path in paths:
        left = expected.tensors.get(path)
        right = actual.tensors.get(path)
        if left is None or right is None:
            mismatches.append({"path": path, "reason": "missing"})
        elif left.dtype != right.dtype or left.shape != right.shape:
            mismatches.append({"path": path, "reason": "dtype_or_shape"})
        elif not torch.equal(left.view(torch.uint8), right.view(torch.uint8)):
            mismatches.append({"path": path, "reason": "bytes"})
    return {
        "pass": not mismatches,
        "tensor_count": len(paths),
        "expected_digest": expected.digest,
        "actual_digest": actual.digest,
        "mismatches": mismatches,
        "gate_claim": False,
    }


__all__ = [
    "C4_RESIDENT_MIXED_R5_COUNTS_V1",
    "C4_RESIDENT_MIXED_R5_DECISION_V1",
    "C4_RESIDENT_MIXED_R5_IDENTITY_LIMITATION_V1",
    "C4_RESIDENT_MIXED_R5_SCHEMA_VERSION_V1",
    "C4ResidentMixedR5AuthoritativeSnapshotV1",
    "C4ResidentMixedR5FixtureV1",
    "capture_c4_resident_mixed_r5_authoritative_v1",
    "compare_c4_resident_mixed_r5_authoritative_v1",
    "make_c4_resident_mixed_r5_fixture_v1",
]
