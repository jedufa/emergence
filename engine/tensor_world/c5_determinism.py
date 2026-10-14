"""Preparation-bound C5 digest and checkpoint ownership.

The C3/C4 slices deliberately do not own a resident production RNG or event
buffer.  This module makes that omission explicit instead of silently hashing
an incomplete continuation state.  Callers must provide an explicit replay
context containing RNG words/cursors and deterministic event-order surfaces.
The context is intentionally small and representation-opaque; C5 can replace
the fixture tensors with the source-current resident implementation without
changing the checkpoint boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Final, Mapping

import torch

from engine.perf.cuda_architecture_b_resident_window_runtime_v1 import (
    refill_cpython_gaussian_window_,
)

from .contracts import TensorWorldContractError
from .c4_state import TensorWorldStateBundleV4


C5_DIGEST_SCHEMA_VERSION_V1: Final[str] = "tensor_world_c5_canonical_digest_v1"
C5_CHECKPOINT_SCHEMA_VERSION_V1: Final[str] = "tensor_world_c5_checkpoint_v1"
C5_REPLAY_CONTEXT_SCHEMA_VERSION_V1: Final[str] = "tensor_world_c5_replay_context_v1"
C5_TYPED_CONTINUATION_SCHEMA_VERSION_R11_V1: Final[str] = (
    "tensor_world_c5_typed_continuation_r11_v1"
)
C5_CONTINUATION_AUTHORITY_SCHEMA_VERSION_R11_V1: Final[str] = (
    "tensor_world_c5_continuation_authority_r11_v1"
)
C5_CONTINUATION_AUTHORITY_DECISION_R11_V1: Final[str] = "NO_GATE_DECISION"
C5_GAUSSIAN_WINDOW_SCHEMA_VERSION_R11_V1: Final[str] = (
    "tensor_world_c5_gaussian_window_r11_v1"
)
C5_GAUSSIAN_WINDOW_AUTHORITATIVE_R11_V1: Final[int] = 0
C5_GAUSSIAN_WINDOW_EXHAUSTED_R11_V1: Final[int] = 1
C5_GAUSSIAN_WINDOW_REFILL_IN_PROGRESS_R11_V1: Final[int] = 2
C5_GAUSSIAN_WINDOW_INVALID_R11_V1: Final[int] = 3
ROOT = Path(__file__).resolve().parents[2]
EVENT_RING_SOURCE = ROOT / "engine/tensor_world/c4_event_ring_r11.py"


@dataclass(slots=True)
class TensorWorldC5PCG64RoleStateR11V1:
    """Typed PCG64 continuation view for one trajectory-driving role."""

    state_words: torch.Tensor
    buffer_words: torch.Tensor
    cursor: torch.Tensor
    raw64_transition_count: torch.Tensor
    output32_return_count: torch.Tensor
    semantic_draw_count: torch.Tensor
    role: str

    def validate(self, *, batch: int, device: torch.device) -> None:
        expected = (
            ("state_words", self.state_words, (batch, 4), torch.int64),
            ("buffer_words", self.buffer_words, (batch, 2), torch.int64),
            ("cursor", self.cursor, (batch,), torch.int64),
            (
                "raw64_transition_count",
                self.raw64_transition_count,
                (batch,),
                torch.int64,
            ),
            (
                "output32_return_count",
                self.output32_return_count,
                (batch,),
                torch.int64,
            ),
            ("semantic_draw_count", self.semantic_draw_count, (batch,), torch.int64),
        )
        for name, tensor, shape, dtype in expected:
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.device != device
            ):
                raise TensorWorldContractError(
                    f"C5 typed role {self.role} {name} has an invalid ABI"
                )
        if not self.role:
            raise TensorWorldContractError("C5 typed RNG role name is required")
        if device.type == "cpu" and (
            bool((self.cursor < 0).any().item())
            or bool((self.raw64_transition_count < 0).any().item())
            or bool((self.output32_return_count < 0).any().item())
            or bool((self.semantic_draw_count < 0).any().item())
        ):
            raise TensorWorldContractError(
                f"C5 typed role {self.role} counters must be nonnegative"
            )


@dataclass(slots=True)
class TensorWorldC5GaussianWindowR11V1:
    """Published caller-owned CPython Gaussian window authority.

    The two float64 tapes remain in ``bundle.scalar_rng``.  This compact
    continuation state binds those physical rows to an absolute raw-draw
    interval and records the refill identity needed by checkpoint/replay.
    ``status`` is publish-last: any interrupted refill remains fail-closed.
    """

    base_raw_draws: torch.Tensor
    valid_raw_draw_ends: torch.Tensor
    refill_generations: torch.Tensor
    provenance_sha256: torch.Tensor
    status: torch.Tensor
    exhaustion_raw_draws: torch.Tensor
    schema_version: str = C5_GAUSSIAN_WINDOW_SCHEMA_VERSION_R11_V1

    @property
    def device(self) -> torch.device:
        return self.base_raw_draws.device

    def validate(self, *, batch: int, tape_capacity: int) -> None:
        if self.schema_version != C5_GAUSSIAN_WINDOW_SCHEMA_VERSION_R11_V1:
            raise TensorWorldContractError(
                "C5 R11 Gaussian-window schema is missing or unsupported"
            )
        expected = (
            ("base_raw_draws", self.base_raw_draws, (batch,), torch.int64),
            (
                "valid_raw_draw_ends",
                self.valid_raw_draw_ends,
                (batch,),
                torch.int64,
            ),
            (
                "refill_generations",
                self.refill_generations,
                (batch,),
                torch.int64,
            ),
            (
                "provenance_sha256",
                self.provenance_sha256,
                (batch, 32),
                torch.uint8,
            ),
            ("status", self.status, (batch,), torch.int32),
            (
                "exhaustion_raw_draws",
                self.exhaustion_raw_draws,
                (batch,),
                torch.int64,
            ),
        )
        devices = {self.device}
        for name, tensor, shape, dtype in expected:
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise TensorWorldContractError(
                    f"C5 R11 Gaussian-window {name} has an invalid ABI"
                )
            devices.add(tensor.device)
        if len(devices) != 1:
            raise TensorWorldContractError(
                "C5 R11 Gaussian-window tensors must share one device"
            )
        if type(tape_capacity) is not int or tape_capacity < 0:
            raise TensorWorldContractError(
                "C5 R11 Gaussian-window physical capacity must be nonnegative"
            )
        base = self.base_raw_draws.detach().cpu()
        end = self.valid_raw_draw_ends.detach().cpu()
        generation = self.refill_generations.detach().cpu()
        statuses = self.status.detach().cpu()
        exhausted = self.exhaustion_raw_draws.detach().cpu()
        provenance = self.provenance_sha256.detach().cpu()
        allowed = {
            C5_GAUSSIAN_WINDOW_AUTHORITATIVE_R11_V1,
            C5_GAUSSIAN_WINDOW_EXHAUSTED_R11_V1,
            C5_GAUSSIAN_WINDOW_REFILL_IN_PROGRESS_R11_V1,
            C5_GAUSSIAN_WINDOW_INVALID_R11_V1,
        }
        for world in range(batch):
            window_base = int(base[world])
            window_end = int(end[world])
            refill_generation = int(generation[world])
            window_status = int(statuses[world])
            exhaustion_raw = int(exhausted[world])
            if (
                window_base < 0
                or window_end < window_base
                or window_end - window_base > tape_capacity
                or refill_generation < 0
                or window_status not in allowed
            ):
                raise TensorWorldContractError(
                    "C5 R11 Gaussian-window semantic state is invalid"
                )
            if window_status == C5_GAUSSIAN_WINDOW_AUTHORITATIVE_R11_V1:
                if (
                    (window_end == window_base and tape_capacity != 0)
                    or exhaustion_raw != -1
                ):
                    raise TensorWorldContractError(
                        "C5 R11 authoritative Gaussian window is empty or exhausted"
                    )
                if not bool(provenance[world].any().item()):
                    raise TensorWorldContractError(
                        "C5 R11 authoritative Gaussian window lacks provenance"
                    )
            elif window_status == C5_GAUSSIAN_WINDOW_EXHAUSTED_R11_V1:
                if exhaustion_raw < window_base:
                    raise TensorWorldContractError(
                        "C5 R11 Gaussian exhaustion cursor is invalid"
                    )


@dataclass(slots=True)
class TensorWorldC5ReplayContextV1:
    """Explicit continuation-owned RNG and event-order state.

    ``rng_state_words`` and ``rng_cursors`` are opaque resident stream
    surfaces.  The implementation does not infer an algorithm or consume a
    draw.  ``event_order_keys`` is the stable per-slot ordering surface used
    by population/lifecycle transactions; ``event_sequence`` seals the
    world-local append order.
    """

    rng_state_words: torch.Tensor
    rng_cursors: torch.Tensor
    event_sequence: torch.Tensor
    event_order_keys: torch.Tensor
    rng_algorithm: str = "pcg64_words_v1"
    event_order_schema: str = "tensor_world_event_order_v1"
    schema_version: str = C5_REPLAY_CONTEXT_SCHEMA_VERSION_V1
    # Optional explicit PCG64 cache words.  Older preparation contexts may
    # omit this field; production-bound callers must provide it through
    # ``validate_pcg64_for_bundle``.
    rng_buffer_words: torch.Tensor | None = None
    semantic_draw_counters: torch.Tensor | None = None
    # R11 typed continuation uses dedicated role streams after the
    # caller-visible prefix. Historical contexts may leave these unset; the
    # composed R11 owner invokes the strict validator below.
    typed_continuation_schema: str | None = None
    typed_role_indices: tuple[int, int, int, int, int] | None = None
    raw64_transition_counts: torch.Tensor | None = None
    output32_return_counts: torch.Tensor | None = None
    gaussian_window: TensorWorldC5GaussianWindowR11V1 | None = None

    @property
    def device(self) -> torch.device:
        return self.rng_state_words.device

    def validate_for_bundle(self, bundle: TensorWorldStateBundleV4) -> None:
        if self.schema_version != C5_REPLAY_CONTEXT_SCHEMA_VERSION_V1:
            raise TensorWorldContractError(
                "C5 replay context schema version is unsupported"
            )
        if not self.rng_algorithm or not self.event_order_schema:
            raise TensorWorldContractError("C5 replay context identity is incomplete")
        if self.rng_state_words.ndim != 3:
            raise TensorWorldContractError(
                "C5 rng_state_words must have shape [batch, streams, words]"
            )
        expected_slots = (bundle.batch, bundle.capacity)
        if tuple(self.rng_cursors.shape) != (
            bundle.batch,
            self.rng_state_words.shape[1],
        ):
            raise TensorWorldContractError(
                "C5 rng_cursors must have shape [batch, streams] matching rng_state_words"
            )
        if tuple(self.rng_state_words.shape[:1]) != (bundle.batch,):
            raise TensorWorldContractError(
                "C5 rng_state_words batch dimension does not match the state"
            )
        if tuple(self.event_sequence.shape) != (bundle.batch,):
            raise TensorWorldContractError("C5 event_sequence must have shape [batch]")
        if tuple(self.event_order_keys.shape) != expected_slots:
            raise TensorWorldContractError(
                "C5 event_order_keys must have shape [batch, capacity]"
            )
        expected_dtypes = {
            "rng_state_words": torch.int64,
            "rng_cursors": torch.int64,
            "event_sequence": torch.int64,
            "event_order_keys": torch.int64,
        }
        devices = {bundle.device}
        for name, expected_dtype in expected_dtypes.items():
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor):
                raise TensorWorldContractError(f"C5 {name} is not a tensor")
            if tensor.dtype != expected_dtype:
                raise TensorWorldContractError(
                    f"C5 {name} requires dtype={expected_dtype}, got {tensor.dtype}"
                )
            if not tensor.is_contiguous():
                raise TensorWorldContractError(f"C5 {name} must be contiguous")
            devices.add(tensor.device)
        if len(devices) != 1:
            raise TensorWorldContractError(
                "C5 replay context and state must share one device"
            )
        if self.rng_buffer_words is not None:
            expected = (bundle.batch, self.rng_state_words.shape[1], 2)
            if (
                not isinstance(self.rng_buffer_words, torch.Tensor)
                or tuple(self.rng_buffer_words.shape) != expected
                or self.rng_buffer_words.dtype != torch.int64
                or not self.rng_buffer_words.is_contiguous()
                or self.rng_buffer_words.device != bundle.device
            ):
                raise TensorWorldContractError(
                    "C5 rng_buffer_words requires contiguous int64 [batch, streams, 2]"
                )
        if self.semantic_draw_counters is not None:
            expected = (bundle.batch, self.rng_state_words.shape[1])
            if (
                not isinstance(self.semantic_draw_counters, torch.Tensor)
                or tuple(self.semantic_draw_counters.shape) != expected
                or self.semantic_draw_counters.dtype != torch.int64
                or not self.semantic_draw_counters.is_contiguous()
                or self.semantic_draw_counters.device != bundle.device
            ):
                raise TensorWorldContractError(
                    "C5 semantic_draw_counters requires contiguous int64 [batch, streams]"
                )
        for name, tensor in (
            ("raw64_transition_counts", self.raw64_transition_counts),
            ("output32_return_counts", self.output32_return_counts),
        ):
            if tensor is not None:
                expected = (bundle.batch, self.rng_state_words.shape[1])
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tuple(tensor.shape) != expected
                    or tensor.dtype != torch.int64
                    or not tensor.is_contiguous()
                    or tensor.device != bundle.device
                ):
                    raise TensorWorldContractError(
                        f"C5 {name} requires contiguous int64 [batch, streams]"
                    )
        if self.gaussian_window is not None:
            if bundle.scalar_rng is None:
                raise TensorWorldContractError(
                    "C5 Gaussian-window authority requires scalar RNG state"
                )
            self.gaussian_window.validate(
                batch=bundle.batch,
                tape_capacity=bundle.scalar_rng.tape_capacity,
            )
        event_cursor: torch.Tensor | None = None
        event_cursor_name = ""
        if bundle.event_ring is not None:
            event_cursor = bundle.event_ring.producer_cursor
            event_cursor_name = "R11 event_ring.producer_cursor"
        elif bundle.events is not None:
            event_cursor = bundle.events.active_count
            event_cursor_name = "resident C4 event cursor"
        if event_cursor is not None and not torch.equal(
            self.event_sequence, event_cursor
        ):
            raise TensorWorldContractError(
                f"C5 event_sequence must match the {event_cursor_name}"
            )

    def validate_pcg64_for_bundle(self, bundle: TensorWorldStateBundleV4) -> None:
        """Validate the concrete resident PCG64 state/cursor representation."""

        self.validate_for_bundle(bundle)
        if self.rng_algorithm != "pcg64_words_v1":
            raise TensorWorldContractError(
                "C5 production RNG requires rng_algorithm=pcg64_words_v1"
            )
        if self.rng_state_words.shape[2] != 4 or self.rng_buffer_words is None:
            raise TensorWorldContractError(
                "C5 production PCG64 requires state words [batch, streams, 4] "
                "and buffer words [batch, streams, 2]"
            )
        if self.semantic_draw_counters is None:
            raise TensorWorldContractError(
                "C5 production PCG64 requires semantic draw counters"
            )
        # The increment is an odd 128-bit value.  Values are stored as signed
        # int64 bit patterns, so only the low word's parity is observable here.
        if bool(((self.rng_state_words[:, :, 2] & 1) == 0).any().item()):
            raise TensorWorldContractError("C5 PCG64 increments must be odd")
        if (
            bool((self.rng_buffer_words[:, :, 0] < 0).any().item())
            or bool((self.rng_buffer_words[:, :, 0] > 1).any().item())
            or bool((self.rng_buffer_words[:, :, 1] < 0).any().item())
        ):
            raise TensorWorldContractError("C5 PCG64 cache words are invalid")

    def validate_typed_continuation_for_bundle(
        self, bundle: TensorWorldStateBundleV4
    ) -> None:
        """Require the complete R11 typed continuation owner."""

        self.validate_pcg64_for_bundle(bundle)
        if (
            self.typed_continuation_schema
            != C5_TYPED_CONTINUATION_SCHEMA_VERSION_R11_V1
        ):
            raise TensorWorldContractError(
                "C5 R11 typed continuation schema is missing or unsupported"
            )
        indices = self.typed_role_indices
        if indices is None or len(indices) != 5:
            raise TensorWorldContractError("C5 R11 typed role mapping is incomplete")
        if len(set(indices)) != 5 or any(
            type(index) is not int
            or index < 0
            or index >= self.rng_state_words.shape[1]
            for index in indices
        ):
            raise TensorWorldContractError("C5 R11 typed role mapping is invalid")
        if self.raw64_transition_counts is None or self.output32_return_counts is None:
            raise TensorWorldContractError(
                "C5 R11 raw/output PCG counters are required"
            )
        if self.semantic_draw_counters is None:
            raise TensorWorldContractError("C5 R11 semantic PCG counters are required")
        for name, index in zip(
            ("environment", "runner", "population_mutation", "policy", "drive_noise"),
            indices,
            strict=True,
        ):
            for counter_name, counter in (
                ("raw64_transition_count", self.raw64_transition_counts),
                ("output32_return_count", self.output32_return_counts),
                ("semantic_draw_count", self.semantic_draw_counters),
            ):
                if tuple(counter[:, index].shape) != (bundle.batch,):
                    raise TensorWorldContractError(
                        f"C5 R11 {name} {counter_name} is unavailable"
                    )
        if bundle.scalar_rng is None:
            raise TensorWorldContractError(
                "C5 R11 typed continuation requires resident CPython MT19937 state"
            )
        bundle.scalar_rng.validate(expected_batch=bundle.batch)
        if self.gaussian_window is None:
            raise TensorWorldContractError(
                "C5 R11 typed continuation requires Gaussian-window authority"
            )
        self.gaussian_window.validate(
            batch=bundle.batch,
            tape_capacity=bundle.scalar_rng.tape_capacity,
        )
        raw = bundle.scalar_rng.python_random_raw_draws.detach().cpu()
        base = self.gaussian_window.base_raw_draws.detach().cpu()
        end = self.gaussian_window.valid_raw_draw_ends.detach().cpu()
        if bool(((raw < base) | (raw > end)).any().item()):
            raise TensorWorldContractError(
                "C5 R11 scalar RNG cursor is outside its Gaussian window"
            )


@dataclass(frozen=True, slots=True)
class TensorWorldC5CheckpointV1:
    """Immutable-at-boundary CPU snapshot used by the C5 preparation seam."""

    state: TensorWorldStateBundleV4
    replay_context: TensorWorldC5ReplayContextV1
    digest: str
    backend_id: str
    source_commit: str
    provider_contract_id: str
    metadata: Mapping[str, Any]
    schema_version: str = C5_CHECKPOINT_SCHEMA_VERSION_V1

    def validate(self) -> None:
        if self.schema_version != C5_CHECKPOINT_SCHEMA_VERSION_V1:
            raise TensorWorldContractError(
                "C5 checkpoint schema version is unsupported"
            )
        if (
            not self.backend_id
            or not self.source_commit
            or not self.provider_contract_id
        ):
            raise TensorWorldContractError("C5 checkpoint identity is incomplete")
        if len(self.digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.digest
        ):
            raise TensorWorldContractError("C5 checkpoint digest is not SHA256")
        self.state.validate()
        self.replay_context.validate_for_bundle(self.state)
        if self.state.device.type != "cpu" or self.replay_context.device.type != "cpu":
            raise TensorWorldContractError(
                "C5 checkpoint snapshots must be CPU resident"
            )
        try:
            json.dumps(self.metadata, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise TensorWorldContractError(
                "C5 checkpoint metadata must be JSON-compatible"
            ) from exc


def _canonical_update(hasher: "hashlib._Hash", value: Any, path: str) -> None:
    """Encode dataclasses/tensors/scalars with explicit field boundaries."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", copy=True).contiguous()
        hasher.update(b"TENSOR\0")
        hasher.update(path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(
            json.dumps(tuple(tensor.shape), separators=(",", ":")).encode("ascii")
        )
        hasher.update(b"\0")
        hasher.update(tensor.view(torch.uint8).numpy().tobytes())
        hasher.update(b"\0")
        return
    if is_dataclass(value) and not isinstance(value, type):
        hasher.update(b"DATACLASS\0")
        hasher.update(path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(type(value).__qualname__.encode("utf-8"))
        hasher.update(b"\0")
        for field in fields(value):
            _canonical_update(
                hasher, getattr(value, field.name), f"{path}.{field.name}"
            )
        return
    if isinstance(value, Mapping):
        hasher.update(b"MAPPING\0")
        hasher.update(path.encode("utf-8"))
        hasher.update(b"\0")
        for key in sorted(value, key=lambda item: str(item)):
            _canonical_update(hasher, key, f"{path}.key")
            _canonical_update(hasher, value[key], f"{path}[{key!r}]")
        return
    if isinstance(value, (tuple, list)):
        hasher.update(b"SEQUENCE\0")
        hasher.update(path.encode("utf-8"))
        hasher.update(b"\0")
        for index, item in enumerate(value):
            _canonical_update(hasher, item, f"{path}[{index}]")
        return
    if isinstance(value, bool):
        encoded = "true" if value else "false"
    elif value is None:
        encoded = "null"
    elif isinstance(value, (int, float, str)):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    else:
        raise TensorWorldContractError(
            f"C5 canonical digest cannot encode {type(value).__qualname__} at {path}"
        )
    hasher.update(b"SCALAR\0")
    hasher.update(path.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(encoded.encode("utf-8"))
    hasher.update(b"\0")


def canonical_tensor_world_digest_v1(
    bundle: TensorWorldStateBundleV4,
    replay_context: TensorWorldC5ReplayContextV1 | None = None,
) -> str:
    """Return a canonical digest, requiring all C5 continuation surfaces."""

    bundle.validate()
    if replay_context is None:
        raise TensorWorldContractError(
            "C5 canonical digest requires explicit RNG/event replay context"
        )
    replay_context.validate_for_bundle(bundle)
    hasher = hashlib.sha256()
    hasher.update(C5_DIGEST_SCHEMA_VERSION_V1.encode("ascii"))
    _canonical_update(hasher, bundle, "bundle")
    _canonical_update(hasher, replay_context, "replay_context")
    return hasher.hexdigest()


def _clone_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True).contiguous()
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: _clone_to_device(getattr(value, field.name), device)
                for field in fields(value)
            }
        )
    if isinstance(value, tuple):
        return tuple(_clone_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_clone_to_device(item, device) for item in value]
    if isinstance(value, Mapping):
        return {key: _clone_to_device(item, device) for key, item in value.items()}
    return value


def _clone_to_cpu(value: Any) -> Any:
    return _clone_to_device(value, torch.device("cpu"))


def capture_c5_checkpoint_v1(
    bundle: TensorWorldStateBundleV4,
    replay_context: TensorWorldC5ReplayContextV1,
    *,
    backend_id: str,
    source_commit: str,
    provider_contract_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> TensorWorldC5CheckpointV1:
    """Capture a coherent checkpoint at an explicit safe boundary."""

    bundle.validate()
    replay_context.validate_for_bundle(bundle)
    if not backend_id or not source_commit or not provider_contract_id:
        raise TensorWorldContractError("C5 checkpoint identity is incomplete")
    snapshot = _clone_to_cpu(bundle)
    context_snapshot = _clone_to_cpu(replay_context)
    checkpoint = TensorWorldC5CheckpointV1(
        state=snapshot,
        replay_context=context_snapshot,
        digest=canonical_tensor_world_digest_v1(snapshot, context_snapshot),
        backend_id=backend_id,
        source_commit=source_commit,
        provider_contract_id=provider_contract_id,
        metadata=dict(metadata or {}),
    )
    checkpoint.validate()
    return checkpoint


def _restore_in_place(destination: Any, source: Any) -> None:
    if type(destination) is not type(source):
        raise TensorWorldContractError(
            "C5 restore destination type does not match checkpoint source"
        )
    if isinstance(destination, torch.Tensor):
        destination.copy_(source.to(device=destination.device))
        return
    if is_dataclass(destination) and not isinstance(destination, type):
        for field in fields(destination):
            target = getattr(destination, field.name)
            value = getattr(source, field.name)
            if isinstance(target, torch.Tensor):
                _restore_in_place(target, value)
            elif is_dataclass(target) and not isinstance(target, type):
                _restore_in_place(target, value)
            elif isinstance(target, tuple) and isinstance(value, tuple):
                if len(target) != len(value):
                    raise TensorWorldContractError(
                        "C5 restore tuple length does not match checkpoint source"
                    )
                if all(
                    is_dataclass(target_item)
                    and not isinstance(target_item, type)
                    and type(target_item) is type(value_item)
                    for target_item, value_item in zip(target, value, strict=True)
                ):
                    # CUDA graphs retain raw pointers into the fixed event-ring
                    # bank tuple.  Restore its tensor leaves in place so a
                    # post-capture control-boundary restore cannot invalidate
                    # the graph's addresses.
                    for target_item, value_item in zip(target, value, strict=True):
                        _restore_in_place(target_item, value_item)
                else:
                    target_device = getattr(destination, "device", torch.device("cpu"))
                    setattr(
                        destination,
                        field.name,
                        tuple(_clone_to_device(item, target_device) for item in value),
                    )
            elif isinstance(target, list) and isinstance(value, list):
                target_device = getattr(destination, "device", torch.device("cpu"))
                setattr(
                    destination,
                    field.name,
                    [_clone_to_device(item, target_device) for item in value],
                )
            else:
                setattr(destination, field.name, value)
        return
    raise TensorWorldContractError(
        f"C5 restore cannot update {type(destination).__qualname__}"
    )


def restore_c5_checkpoint_v1(
    checkpoint: TensorWorldC5CheckpointV1,
    destination: TensorWorldStateBundleV4,
    replay_context: TensorWorldC5ReplayContextV1,
    *,
    expected_backend_id: str,
    expected_source_commit: str,
    expected_provider_contract_id: str,
) -> str:
    """Restore state/context without consuming RNG or changing draw cursors."""

    checkpoint.validate()
    if checkpoint.backend_id != expected_backend_id:
        raise TensorWorldContractError("C5 checkpoint backend identity is incompatible")
    if checkpoint.source_commit != expected_source_commit:
        raise TensorWorldContractError("C5 checkpoint source commit is incompatible")
    if checkpoint.provider_contract_id != expected_provider_contract_id:
        raise TensorWorldContractError(
            "C5 checkpoint provider contract identity is incompatible"
        )
    destination.validate()
    replay_context.validate_for_bundle(destination)
    if destination.schema_version != checkpoint.state.schema_version:
        raise TensorWorldContractError("C5 checkpoint state schema is incompatible")
    expected_digest = canonical_tensor_world_digest_v1(
        checkpoint.state, checkpoint.replay_context
    )
    if expected_digest != checkpoint.digest:
        raise TensorWorldContractError("C5 checkpoint digest is corrupted")
    # Stage the complete restore on same-device shadows.  A checkpoint may be
    # structurally incompatible with a live destination even after the
    # destination's own validation succeeds (for example, one side has an
    # optional resident state and the other does not).  Applying directly to
    # the live bundle would then leave a partially restored trajectory.  The
    # shadow must pass every post-restore check before any caller-owned object
    # is mutated.
    staged_destination = _clone_to_device(destination, destination.device)
    staged_replay_context = _clone_to_device(replay_context, replay_context.device)
    _restore_in_place(staged_destination, checkpoint.state)
    _restore_in_place(staged_replay_context, checkpoint.replay_context)
    staged_destination.validate()
    staged_replay_context.validate_for_bundle(staged_destination)
    restored_digest = canonical_tensor_world_digest_v1(
        staged_destination, staged_replay_context
    )
    if restored_digest != checkpoint.digest:
        raise TensorWorldContractError("C5 restore digest does not match checkpoint")
    # This commit is structurally identical to the staged operation above.  A
    # successful stage therefore makes this caller-owned update atomic with
    # respect to all contract failures.
    _restore_in_place(destination, staged_destination)
    _restore_in_place(replay_context, staged_replay_context)
    return restored_digest


_U64_MASK: Final[int] = (1 << 64) - 1
_U64_SIGN: Final[int] = 1 << 63
_PCG64_MULTIPLIER_LO: Final[int] = 4865540595714422341
_PCG64_MULTIPLIER_HI: Final[int] = 2549297995355413924


def _as_u64(value: int) -> int:
    return int(value) & _U64_MASK


def _as_i64(value: int) -> int:
    value = _as_u64(value)
    return value - (1 << 64) if value >= _U64_SIGN else value


def _pcg64_raw64_words(words: list[int]) -> int:
    old_lo, old_hi, inc_lo, inc_hi = (_as_u64(value) for value in words[:4])
    next_lo = (old_lo * _PCG64_MULTIPLIER_LO) & _U64_MASK
    next_hi = (
        ((old_lo * _PCG64_MULTIPLIER_LO) >> 64)
        + old_lo * _PCG64_MULTIPLIER_HI
        + old_hi * _PCG64_MULTIPLIER_LO
    ) & _U64_MASK
    before_increment = next_lo
    next_lo = (next_lo + inc_lo) & _U64_MASK
    next_hi = (next_hi + inc_hi + int(next_lo < before_increment)) & _U64_MASK
    words[0] = _as_i64(next_lo)
    words[1] = _as_i64(next_hi)
    xorshifted = next_hi ^ next_lo
    rotation = next_hi >> 58
    if rotation:
        return (
            (xorshifted >> rotation) | (xorshifted << ((64 - rotation) & 63))
        ) & _U64_MASK
    return xorshifted


def _pcg64_raw32_words(
    words: list[int], buffer: list[int], counters: list[int] | None = None
) -> int:
    if int(buffer[0]) != 0:
        buffer[0] = 0
        if counters is not None:
            counters[1] += 1
        return int(buffer[1]) & 0xFFFFFFFF
    raw = _pcg64_raw64_words(words)
    buffer[1] = _as_i64(raw >> 32)
    buffer[0] = 1
    if counters is not None:
        counters[0] += 1
        counters[1] += 1
    return raw & 0xFFFFFFFF


def _pcg64_bounded_uint32_words(
    words: list[int],
    buffer: list[int],
    max_value: int,
    counters: list[int] | None = None,
) -> int:
    if max_value < 0 or max_value > 0xFFFFFFFF:
        raise TensorWorldContractError("C5 PCG64 bounded maximum is outside uint32")
    if max_value == 0xFFFFFFFF:
        return _pcg64_raw32_words(words, buffer, counters)
    mask = int(max_value)
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16
    while True:
        value = _pcg64_raw32_words(words, buffer, counters) & mask
        if value <= max_value:
            return value


def make_c5_pcg64_context_v1(
    bundle: TensorWorldStateBundleV4,
    *,
    streams: int = 2,
    seed: int = 0,
    typed: bool = False,
    source_commit: str = "UNSPECIFIED_TEST_SOURCE",
) -> TensorWorldC5ReplayContextV1:
    """Create explicit deterministic PCG64 streams for a CPU oracle fixture.

    The seed mixer only initializes the state; all subsequent draws use the
    source-current PCG64 transition and cache semantics.  A production CUDA
    route must bind its state initialization to the authoritative seed
    contract before changing the preparation status of C4/C5.
    """

    if type(streams) is not int or streams <= 0:
        raise TensorWorldContractError("C5 PCG64 streams must be a positive int")
    if type(seed) is not int:
        raise TensorWorldContractError("C5 PCG64 seed must be an int")
    bundle.validate()
    if type(typed) is not bool:
        raise TensorWorldContractError("C5 typed continuation selector must be bool")
    if not isinstance(source_commit, str) or not source_commit:
        raise TensorWorldContractError("C5 source commit identity is incomplete")
    # Preserve the historical prefix for preparation/R9 callers.  The R11
    # owner opts into three additional role streams so all five typed PCG64
    # roles have distinct resident state and counters.
    total_streams = streams + 3 if typed else streams
    shape = (bundle.batch, total_streams, 4)
    state = torch.empty(shape, dtype=torch.int64, device=bundle.device)
    buffer = torch.zeros(
        (bundle.batch, total_streams, 2), dtype=torch.int64, device=bundle.device
    )
    cursors = torch.zeros(
        (bundle.batch, total_streams), dtype=torch.int64, device=bundle.device
    )
    semantic = torch.zeros_like(cursors)

    def mix(value: int) -> int:
        value = (value + 0x9E3779B97F4A7C15) & _U64_MASK
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64_MASK
        return (value ^ (value >> 31)) & _U64_MASK

    for world in range(bundle.batch):
        for stream in range(total_streams):
            base = mix(
                seed ^ (world * 0xD1B54A32D192ED03) ^ (stream * 0x8CB92BAA7F3D8DD7)
            )
            inc = (mix(base ^ 0xDB4F0B9175AE2165) | 1) & _U64_MASK
            state[world, stream, 0] = _as_i64(mix(base ^ 0xA24BAED4963EE407))
            state[world, stream, 1] = _as_i64(mix(base ^ 0x9FB21C651E98DF25))
            state[world, stream, 2] = _as_i64(inc)
            state[world, stream, 3] = _as_i64(mix(inc ^ 0xC13FA9A902A6328F))
    if bundle.event_ring is not None:
        initial_event_sequence = bundle.event_ring.producer_cursor.clone()
    elif bundle.events is not None:
        initial_event_sequence = bundle.events.active_count.clone()
    else:
        initial_event_sequence = torch.zeros(
            bundle.batch, dtype=torch.int64, device=bundle.device
        )
    gaussian_window: TensorWorldC5GaussianWindowR11V1 | None = None
    if typed:
        if bundle.scalar_rng is None:
            raise TensorWorldContractError(
                "C5 typed continuation requires scalar RNG state"
            )
        scalar = bundle.scalar_rng
        scalar.validate(expected_batch=bundle.batch)
        capacity = scalar.tape_capacity
        base = torch.zeros(
            bundle.batch, dtype=torch.int64, device=bundle.device
        )
        if capacity > 0:
            refill_cpython_gaussian_window_(
                {
                    "python_random_words": scalar.python_random_words,
                    "python_random_index": scalar.python_random_index,
                    "python_random_has_gauss": scalar.python_random_has_gauss,
                    "python_random_gauss_next": scalar.python_random_gauss_next,
                    "python_random_gauss_primary_tape": (
                        scalar.python_random_gauss_primary_tape
                    ),
                    "python_random_gauss_secondary_tape": (
                        scalar.python_random_gauss_secondary_tape
                    ),
                    "python_random_raw_draws": scalar.python_random_raw_draws,
                    "mutation_py_semantic_draws": (
                        scalar.mutation_py_semantic_draws
                    ),
                },
                python_random_window_base=base,
            )
        else:
            base.copy_(scalar.python_random_raw_draws)
        end = base + capacity
        provenance_rows: list[list[int]] = []
        for world in range(bundle.batch):
            hasher = hashlib.sha256()
            hasher.update(C5_GAUSSIAN_WINDOW_SCHEMA_VERSION_R11_V1.encode("ascii"))
            hasher.update(b"\0exact_cpython_initialization_window\0")
            hasher.update(source_commit.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(int(world).to_bytes(8, "little", signed=False))
            hasher.update(int(capacity).to_bytes(8, "little", signed=False))
            hasher.update(int(base[world]).to_bytes(8, "little", signed=True))
            hasher.update(int(end[world]).to_bytes(8, "little", signed=True))
            hasher.update(
                int(capacity > 0).to_bytes(8, "little", signed=False)
            )
            for tensor in (
                scalar.python_random_words[world],
                scalar.python_random_index[world : world + 1],
                scalar.python_random_has_gauss[world : world + 1],
                scalar.python_random_gauss_next[world : world + 1],
                scalar.python_random_gauss_primary_tape[world],
                scalar.python_random_gauss_secondary_tape[world],
            ):
                host = tensor.detach().to(device="cpu", copy=True).contiguous()
                hasher.update(host.view(torch.uint8).numpy().tobytes())
            provenance_rows.append(list(hasher.digest()))
        gaussian_window = TensorWorldC5GaussianWindowR11V1(
            base_raw_draws=base,
            valid_raw_draw_ends=end,
            refill_generations=(
                torch.ones_like(base) if capacity > 0 else torch.zeros_like(base)
            ),
            provenance_sha256=torch.tensor(
                provenance_rows, dtype=torch.uint8, device=bundle.device
            ),
            status=torch.full(
                (bundle.batch,),
                C5_GAUSSIAN_WINDOW_AUTHORITATIVE_R11_V1,
                dtype=torch.int32,
                device=bundle.device,
            ),
            exhaustion_raw_draws=torch.full_like(base, -1),
        )
    context = TensorWorldC5ReplayContextV1(
        rng_state_words=state,
        rng_cursors=cursors,
        event_sequence=initial_event_sequence,
        event_order_keys=torch.arange(
            bundle.batch * bundle.capacity, dtype=torch.int64, device=bundle.device
        ).reshape(bundle.batch, bundle.capacity),
        rng_buffer_words=buffer,
        semantic_draw_counters=semantic,
        typed_continuation_schema=(
            C5_TYPED_CONTINUATION_SCHEMA_VERSION_R11_V1 if typed else None
        ),
        typed_role_indices=(0, 1, streams, streams + 1, streams + 2) if typed else None,
        raw64_transition_counts=torch.zeros_like(cursors) if typed else None,
        output32_return_counts=torch.zeros_like(cursors) if typed else None,
        gaussian_window=gaussian_window,
    )
    context.validate_pcg64_for_bundle(bundle)
    return context


def c5_pcg64_uniform_float32_v1(
    context: TensorWorldC5ReplayContextV1,
    bundle: TensorWorldStateBundleV4,
    *,
    world: int,
    stream: int,
    count: int,
) -> torch.Tensor:
    """Draw source-current NumPy-compatible float32 values from one stream."""

    context.validate_pcg64_for_bundle(bundle)
    if bundle.device.type != "cpu":
        raise TensorWorldContractError("C5 Python PCG64 oracle requires CPU state")
    if not (
        0 <= world < bundle.batch and 0 <= stream < context.rng_state_words.shape[1]
    ):
        raise TensorWorldContractError("C5 PCG64 world/stream is outside context")
    if type(count) is not int or count < 0:
        raise TensorWorldContractError("C5 PCG64 count must be a nonnegative int")
    words = [int(value) for value in context.rng_state_words[world, stream].tolist()]
    buffer = [int(value) for value in context.rng_buffer_words[world, stream].tolist()]
    values: list[float] = []
    counters = [0, 0]
    for _ in range(count):
        raw = _pcg64_raw32_words(words, buffer, counters)
        values.append(float(raw >> 8) * (2.0**-24))
        context.rng_cursors[world, stream] += 1
    context.rng_state_words[world, stream].copy_(torch.tensor(words, dtype=torch.int64))
    context.rng_buffer_words[world, stream].copy_(
        torch.tensor(buffer, dtype=torch.int64)
    )
    if context.raw64_transition_counts is not None:
        context.raw64_transition_counts[world, stream] += counters[0]
    if context.output32_return_counts is not None:
        context.output32_return_counts[world, stream] += counters[1]
    return torch.tensor(values, dtype=torch.float32)


def c5_pcg64_bounded_uint32_v1(
    context: TensorWorldC5ReplayContextV1,
    bundle: TensorWorldStateBundleV4,
    *,
    world: int,
    stream: int,
    max_value: int,
) -> int:
    """Draw one bounded Fisher-Yates value and advance its semantic counter."""

    context.validate_pcg64_for_bundle(bundle)
    if bundle.device.type != "cpu":
        raise TensorWorldContractError("C5 Python PCG64 oracle requires CPU state")
    if not (
        0 <= world < bundle.batch and 0 <= stream < context.rng_state_words.shape[1]
    ):
        raise TensorWorldContractError("C5 PCG64 world/stream is outside context")
    words = [int(value) for value in context.rng_state_words[world, stream].tolist()]
    buffer = [int(value) for value in context.rng_buffer_words[world, stream].tolist()]
    counters = [0, 0]
    value = _pcg64_bounded_uint32_words(words, buffer, int(max_value), counters)
    context.rng_cursors[world, stream] += 1
    context.semantic_draw_counters[world, stream] += 1
    context.rng_state_words[world, stream].copy_(torch.tensor(words, dtype=torch.int64))
    context.rng_buffer_words[world, stream].copy_(
        torch.tensor(buffer, dtype=torch.int64)
    )
    if context.raw64_transition_counts is not None:
        context.raw64_transition_counts[world, stream] += counters[0]
    if context.output32_return_counts is not None:
        context.output32_return_counts[world, stream] += counters[1]
    return value


def run_c5_replay_resume_parity_v1(
    bundle: TensorWorldStateBundleV4,
    replay_context: TensorWorldC5ReplayContextV1,
    *,
    ticks: int,
    checkpoint_ticks: tuple[int, ...] = (1,),
    advance,
    backend_id: str = "tensor_world_c5_preparation_v1",
    source_commit: str = "source-current-test",
    provider_contract_id: str = "tensor_world_provider_preparation_v1",
) -> dict[str, object]:
    """Run a bounded fresh/resume digest comparison at safe boundaries.

    ``advance`` is deliberately an injected state-machine callback.  This
    keeps the harness independent of a provider while requiring every
    trajectory mutation to be visible in the bundle/context digest.  The
    callback is called as ``advance(state, context)`` once per tick.
    """

    if type(ticks) is not int or ticks < 0:
        raise TensorWorldContractError("C5 replay ticks must be a nonnegative int")
    if not callable(advance):
        raise TensorWorldContractError("C5 replay advance must be callable")
    if bundle.device.type != "cpu" or replay_context.device.type != "cpu":
        raise TensorWorldContractError(
            "C5 replay harness currently requires CPU snapshots"
        )
    replay_context.validate_for_bundle(bundle)
    boundaries = tuple(sorted(set(checkpoint_ticks)))
    if any(type(tick) is not int or tick < 0 or tick > ticks for tick in boundaries):
        raise TensorWorldContractError("C5 checkpoint tick is outside replay horizon")

    def clone(value):
        return _clone_to_cpu(value)

    fresh_state = clone(bundle)
    fresh_context = clone(replay_context)
    fresh_digests = [canonical_tensor_world_digest_v1(fresh_state, fresh_context)]
    checkpoints: dict[int, TensorWorldC5CheckpointV1] = {}
    for tick in range(1, ticks + 1):
        advance(fresh_state, fresh_context)
        fresh_digests.append(
            canonical_tensor_world_digest_v1(fresh_state, fresh_context)
        )
        if tick in boundaries:
            checkpoints[tick] = capture_c5_checkpoint_v1(
                fresh_state,
                fresh_context,
                backend_id=backend_id,
                source_commit=source_commit,
                provider_contract_id=provider_contract_id,
                metadata={"safe_boundary": f"tick_{tick}"},
            )

    resumed_digests: dict[int, list[str]] = {}
    for checkpoint_tick, checkpoint in checkpoints.items():
        resumed_state = clone(checkpoint.state)
        resumed_context = clone(checkpoint.replay_context)
        values = [checkpoint.digest]
        for _ in range(checkpoint_tick, ticks):
            advance(resumed_state, resumed_context)
            values.append(
                canonical_tensor_world_digest_v1(resumed_state, resumed_context)
            )
        resumed_digests[checkpoint_tick] = values

    comparisons: dict[str, bool] = {}
    for checkpoint_tick, values in resumed_digests.items():
        comparisons[str(checkpoint_tick)] = values == fresh_digests[checkpoint_tick:]
    return {
        "fresh_digests": fresh_digests,
        "resumed_digests": resumed_digests,
        "checkpoint_ticks": list(boundaries),
        "comparisons": comparisons,
        "all_equal": all(comparisons.values()) if comparisons else True,
    }


def c5_continuation_authority_source_receipt_r11_v1() -> dict[str, object]:
    """Describe the source-current event cursor authority at C5 boundaries."""

    sources = {
        "continuation_owner": Path(__file__),
        "event_ring": EVENT_RING_SOURCE,
    }
    return {
        "schema_version": C5_CONTINUATION_AUTHORITY_SCHEMA_VERSION_R11_V1,
        "decision": C5_CONTINUATION_AUTHORITY_DECISION_R11_V1,
        "legacy_without_event_ring": "bundle.events.active_count",
        "typed_r11_with_event_ring": "bundle.event_ring.producer_cursor",
        "legacy_tape_with_event_ring": "transaction_scratch_count_masked",
        "event_sequence_mismatch": "fail_closed_before_checkpoint_mutation",
        "checkpoint_restore": "stage_validate_then_exact_commit",
        "gaussian_window_authority": (
            "caller_owned_absolute_base_end_generation_provenance_status"
        ),
        "gaussian_refill_publication": "publish_authoritative_last",
        "gate_claim": False,
        "qualification_impact": {
            "legacy_event_cursor_contract": "PRESERVED_COMPATIBLE_INTERFACE",
            "r11_typed_event_continuation": "DELTA_QUALIFICATION_REQUIRED",
            "r11_gaussian_window_continuation": "DELTA_QUALIFICATION_REQUIRED",
            "broad_requalification_required": False,
        },
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


__all__ = [
    "C5_CHECKPOINT_SCHEMA_VERSION_V1",
    "C5_CONTINUATION_AUTHORITY_DECISION_R11_V1",
    "C5_CONTINUATION_AUTHORITY_SCHEMA_VERSION_R11_V1",
    "C5_DIGEST_SCHEMA_VERSION_V1",
    "C5_GAUSSIAN_WINDOW_AUTHORITATIVE_R11_V1",
    "C5_GAUSSIAN_WINDOW_EXHAUSTED_R11_V1",
    "C5_GAUSSIAN_WINDOW_INVALID_R11_V1",
    "C5_GAUSSIAN_WINDOW_REFILL_IN_PROGRESS_R11_V1",
    "C5_GAUSSIAN_WINDOW_SCHEMA_VERSION_R11_V1",
    "C5_REPLAY_CONTEXT_SCHEMA_VERSION_V1",
    "C5_TYPED_CONTINUATION_SCHEMA_VERSION_R11_V1",
    "TensorWorldC5CheckpointV1",
    "TensorWorldC5PCG64RoleStateR11V1",
    "TensorWorldC5GaussianWindowR11V1",
    "TensorWorldC5ReplayContextV1",
    "canonical_tensor_world_digest_v1",
    "c5_continuation_authority_source_receipt_r11_v1",
    "capture_c5_checkpoint_v1",
    "c5_pcg64_bounded_uint32_v1",
    "c5_pcg64_uniform_float32_v1",
    "make_c5_pcg64_context_v1",
    "run_c5_replay_resume_parity_v1",
    "restore_c5_checkpoint_v1",
]
