"""Standalone CPU oracle for address + Philox + samplers + digest."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Sequence

from .address import EventAddress
from .bounded import bounded_uint32
from .contract import CONTRACT_HASH, CONTRACT_IDENTITY, NAMESPACE_DEFAULT
from .normal import box_muller_bits
from .philox import philox4x32_10
from .status import Status
from .uniforms import f64_bits, u32_to_f32_bits, u64_from_lanes, u64_to_f64_half_open


def _word_at(address: EventAddress, namespace: int, ordinal: int) -> int:
    block, lane = divmod(int(ordinal), 4)
    words = philox4x32_10(
        address.counter_words(block),
        address.key_words(namespace),
    )
    return words[lane]


def consumer_record(
    address: EventAddress,
    *,
    draw_block: int,
    lane: int,
    status: Status,
    payload_u64: int,
) -> bytes:
    return b"".join(
        (
            int(address.world_uid).to_bytes(4, "little"),
            int(address.domain_id).to_bytes(2, "little"),
            int(address.semantic_epoch).to_bytes(2, "little"),
            int(address.subject_uid).to_bytes(8, "little"),
            int(address.absolute_tick).to_bytes(8, "little"),
            int(address.operation_id).to_bytes(2, "little"),
            int(draw_block).to_bytes(2, "little"),
            bytes((int(lane) & 0xFF, int(status) & 0xFF)),
            int(payload_u64).to_bytes(8, "little"),
        )
    )


def consumer_digest(records: Iterable[bytes]) -> str:
    hasher = sha256()
    hasher.update(CONTRACT_IDENTITY.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(bytes.fromhex(CONTRACT_HASH))
    for record in records:
        hasher.update(record)
    return hasher.hexdigest()


@dataclass(slots=True)
class DrawResult:
    address: EventAddress
    status: Status
    words: tuple[int, int, int, int]
    bounded: int
    accepted_ordinal: int
    uniform_f32_bits: int
    uniform_f64_bits: int
    z0_bits: int
    z1_bits: int


class PrototypeOracle:
    def __init__(self, rng_namespace_key64: int = NAMESPACE_DEFAULT) -> None:
        self.namespace = int(rng_namespace_key64) & ((1 << 64) - 1)
        self._seen: set[tuple[int, ...]] = set()

    def forget(self) -> None:
        self._seen.clear()

    def generate(
        self,
        address: EventAddress,
        *,
        n: int = 7,
        allow_duplicate: bool = False,
    ) -> DrawResult:
        status = address.validate()
        if status is not Status.RNG_OK:
            empty = (0, 0, 0, 0)
            return DrawResult(address, status, empty, 0, 0, 0, 0, 0, 0)
        self._seen.add(address.identity_tuple(0, 0))
        words = philox4x32_10(
            address.counter_words(0),
            address.key_words(self.namespace),
        )
        bounded, ordinal, bstatus = bounded_uint32(
            n, lambda r: _word_at(address, self.namespace, r)
        )
        z0, z1, nstatus = box_muller_bits(words)
        status = bstatus if bstatus is not Status.RNG_OK else nstatus
        u32_bits = u32_to_f32_bits(words[0])
        u64_bits = f64_bits(u64_to_f64_half_open(u64_from_lanes(words[0], words[1])))
        return DrawResult(
            address,
            status,
            words,
            bounded,
            ordinal,
            u32_bits,
            u64_bits,
            z0,
            z1,
        )

    def digest(self, results: Sequence[DrawResult]) -> str:
        records = [
            consumer_record(
                result.address,
                draw_block=0,
                lane=0,
                status=result.status,
                payload_u64=(
                    result.words[0]
                    | (result.bounded << 32)
                    if result.status is Status.RNG_OK
                    else 0
                ),
            )
            + result.z0_bits.to_bytes(8, "little")
            + result.z1_bits.to_bytes(8, "little")
            + result.accepted_ordinal.to_bytes(4, "little")
            for result in results
        ]
        return consumer_digest(records)
