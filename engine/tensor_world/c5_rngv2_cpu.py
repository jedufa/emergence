"""CPU production reference for V2 resident mutation/birth draws."""

from __future__ import annotations

from dataclasses import dataclass

from engine.tw_rngv2.address import EventAddress, birth_proposal_rank
from engine.tw_rngv2.bounded import bounded_uint32
from engine.tw_rngv2.contract import NAMESPACE_DEFAULT, SEMANTIC_EPOCH
from engine.tw_rngv2.normal import box_muller_pair
from engine.tw_rngv2.philox import philox4x32_10
from engine.tw_rngv2.status import Status
from engine.tw_rngv2.uniforms import u64_from_lanes, u64_to_f64_half_open

from .c5_rngv2_route import RNG_SEMANTICS_V2
from .contracts import TensorWorldContractError


DOMAIN_BERNOULLI = 0x0330
DOMAIN_NORMAL = 0x0331
DOMAIN_DISCRETE = 0x0332

GENE_SPECS: tuple[tuple[float, float, float], ...] = (
    (0.10, -1.0, 1.0),
    (0.15, -2.0, 2.0),
    (2.0, 2.0, 90.0),
    (0.10, 0.25, 2.5),
    (0.05, 0.0, 0.999),
    (0.01, 0.0, 0.10),
    (0.20, 0.0, 3.0),
    (0.03, 0.0, 1.0),
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _words(address: EventAddress, namespace: int, ordinal: int = 0) -> tuple[int, int, int, int]:
    block, _lane = divmod(int(ordinal), 4)
    return philox4x32_10(address.counter_words(block), address.key_words(namespace))


def proposal_subject(
    parent_uid: int,
    mate_uid: int,
    occurrence: int,
    peers: list[tuple[tuple[int, int], int]],
) -> int:
    return birth_proposal_rank((parent_uid, mate_uid), occurrence, peers)


@dataclass(slots=True)
class V2BirthMutation:
    child_genes: tuple[float, ...]
    child_rate: float
    accepted_retries: tuple[int, ...]
    status: Status


def mutate_one_birth_v2(
    *,
    world_uid: int,
    absolute_tick: int,
    subject_uid: int,
    parent_genes: tuple[float, ...],
    mate_genes: tuple[float, ...],
    parent_rate: float,
    mate_rate: float,
    namespace: int = NAMESPACE_DEFAULT,
    semantic_epoch: int = SEMANTIC_EPOCH,
) -> V2BirthMutation:
    if len(parent_genes) != 8 or len(mate_genes) != 8:
        raise TensorWorldContractError("V2 mutation requires 8 scalar genes")
    rate = _clamp(0.5 * (parent_rate + mate_rate), 0.0, 1.0)
    genes: list[float] = []
    retries: list[int] = []
    for field, spec in enumerate(GENE_SPECS):
        mid = 0.5 * (parent_genes[field] + mate_genes[field])
        bernoulli_addr = EventAddress(
            world_uid, DOMAIN_BERNOULLI, subject_uid, absolute_tick, field, semantic_epoch
        )
        words = _words(bernoulli_addr, namespace)
        uniform = u64_to_f64_half_open(u64_from_lanes(words[0], words[1]))
        value = mid
        if uniform < rate:
            normal_addr = EventAddress(
                world_uid, DOMAIN_NORMAL, subject_uid, absolute_tick, field, semantic_epoch
            )
            z0, _z1, status = box_muller_pair(_words(normal_addr, namespace))
            if status is not Status.RNG_OK:
                return V2BirthMutation((), 0.0, (), status)
            value = mid + spec[0] * z0
        genes.append(_clamp(value, spec[1], spec[2]))
        discrete_addr = EventAddress(
            world_uid, DOMAIN_DISCRETE, subject_uid, absolute_tick, field, semantic_epoch
        )
        _value, ordinal, status = bounded_uint32(
            7, lambda r, addr=discrete_addr: _words(addr, namespace, r)[r % 4]
        )
        if status is not Status.RNG_OK:
            return V2BirthMutation((), 0.0, (), status)
        retries.append(ordinal)
    rate_addr = EventAddress(
        world_uid, DOMAIN_BERNOULLI, subject_uid, absolute_tick, 8, semantic_epoch
    )
    rate_words = _words(rate_addr, namespace)
    child_rate = rate
    if u64_to_f64_half_open(u64_from_lanes(rate_words[0], rate_words[1])) < 0.2:
        normal_addr = EventAddress(
            world_uid, DOMAIN_NORMAL, subject_uid, absolute_tick, 8, semantic_epoch
        )
        z0, _z1, status = box_muller_pair(_words(normal_addr, namespace))
        if status is not Status.RNG_OK:
            return V2BirthMutation((), 0.0, (), status)
        child_rate = _clamp(rate + 0.005 * z0, 0.0, 1.0)
    return V2BirthMutation(tuple(genes), child_rate, tuple(retries), Status.RNG_OK)


def mutate_planned_births_v2(
    *,
    world_uid: int,
    absolute_tick: int,
    plans: tuple[tuple[int, int, tuple[float, ...], tuple[float, ...], float, float, int], ...],
    namespace: int = NAMESPACE_DEFAULT,
) -> list[V2BirthMutation]:
    """plans: (parent_uid, mate_uid, parent_genes, mate_genes, parent_rate, mate_rate, occurrence)."""

    peers = [((int(row[0]), int(row[1])), int(row[6])) for row in plans]
    results: list[V2BirthMutation] = []
    for row in plans:
        subject = proposal_subject(row[0], row[1], row[6], peers)
        results.append(
            mutate_one_birth_v2(
                world_uid=world_uid,
                absolute_tick=absolute_tick,
                subject_uid=subject,
                parent_genes=row[2],
                mate_genes=row[3],
                parent_rate=row[4],
                mate_rate=row[5],
                namespace=namespace,
            )
        )
    return results


def require_v2_semantics(version: int) -> None:
    if version != RNG_SEMANTICS_V2:
        raise TensorWorldContractError("CPU V2 mutation requires rng_semantics_version=2")
