"""HAPB CPU oracle. Shared-domain arithmetic is TensorWorld V2, not a fork."""

from __future__ import annotations

from dataclasses import dataclass

from engine.tw_rngv2.bounded import bounded_uint32
from engine.tw_rngv2.normal import box_muller_bits
from engine.tw_rngv2.oracle import PrototypeOracle
from engine.tw_rngv2.philox import philox4x32_10
from engine.tw_rngv2.status import Status
from engine.tw_rngv2.uniforms import f64_bits, u32_to_f32_bits, u64_from_lanes, u64_to_f64_half_open

from .contract import NAMESPACE_DEFAULT, SEMANTIC_EPOCH
from .identity import hapb_counter_words, hapb_key_words, hapb_validate_fields


def hapb_word_at(
    *,
    world_uid: int,
    domain_id: int,
    subject_uid: int,
    absolute_tick: int,
    operation_id: int,
    namespace: int,
    ordinal: int,
    semantic_epoch: int = SEMANTIC_EPOCH,
) -> int:
    block, lane = divmod(int(ordinal), 4)
    words = philox4x32_10(
        hapb_counter_words(
            subject_uid=subject_uid,
            absolute_tick=absolute_tick,
            operation_id=operation_id,
            draw_block=block,
        ),
        hapb_key_words(
            rng_namespace_key64=namespace,
            world_uid=world_uid,
            domain_id=domain_id,
            semantic_epoch=semantic_epoch,
        ),
    )
    return words[lane]


@dataclass(slots=True)
class HapbEvent:
    world_uid: int
    domain_id: int
    subject_uid: int
    absolute_tick: int
    operation_id: int
    semantic_epoch: int = SEMANTIC_EPOCH


@dataclass(slots=True)
class HapbDrawResult:
    event: HapbEvent
    status: Status
    words: tuple[int, int, int, int]
    bounded: int
    accepted_ordinal: int
    uniform_f32_bits: int
    uniform_f64_bits: int
    z0_bits: int
    z1_bits: int


class HapbOracle:
    """CPU reference for migrated HAPB domains.

    Shared TensorWorld domains are bitwise identical to ``PrototypeOracle``.
    Physical slot / host page are intentionally absent from the address.
    """

    def __init__(self, rng_namespace_key64: int = NAMESPACE_DEFAULT) -> None:
        self.namespace = int(rng_namespace_key64) & ((1 << 64) - 1)
        self._parent = PrototypeOracle(self.namespace)

    def generate(self, event: HapbEvent, *, n: int = 7) -> HapbDrawResult:
        status = hapb_validate_fields(
            world_uid=event.world_uid,
            domain_id=event.domain_id,
            semantic_epoch=event.semantic_epoch,
            subject_uid=event.subject_uid,
            absolute_tick=event.absolute_tick,
            operation_id=event.operation_id,
            draw_block=0,
        )
        if status is not Status.RNG_OK:
            empty = (0, 0, 0, 0)
            return HapbDrawResult(event, status, empty, 0, 0, 0, 0, 0, 0)
        words = philox4x32_10(
            hapb_counter_words(
                subject_uid=event.subject_uid,
                absolute_tick=event.absolute_tick,
                operation_id=event.operation_id,
                draw_block=0,
            ),
            hapb_key_words(
                rng_namespace_key64=self.namespace,
                world_uid=event.world_uid,
                domain_id=event.domain_id,
                semantic_epoch=event.semantic_epoch,
            ),
        )
        bounded, ordinal, bstatus = bounded_uint32(
            n,
            lambda retry: hapb_word_at(
                world_uid=event.world_uid,
                domain_id=event.domain_id,
                subject_uid=event.subject_uid,
                absolute_tick=event.absolute_tick,
                operation_id=event.operation_id,
                namespace=self.namespace,
                ordinal=retry,
                semantic_epoch=event.semantic_epoch,
            ),
        )
        z0, z1, nstatus = box_muller_bits(words)
        status = bstatus if bstatus is not Status.RNG_OK else nstatus
        return HapbDrawResult(
            event,
            status,
            words,
            bounded,
            ordinal,
            u32_to_f32_bits(words[0]),
            f64_bits(u64_to_f64_half_open(u64_from_lanes(words[0], words[1]))),
            z0,
            z1,
        )

    def parent_generate(self, address, *, n: int = 7):
        return self._parent.generate(address, n=n)
