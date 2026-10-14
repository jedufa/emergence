"""Lower additive CUDA V2 capture primitives into canonical science rows.

The native owner records only causal primitives that are already computed by
the transition kernel (actions, births, parents, mutation counts, and sparse
death/birth events).  Mutation counts are native integrity counters, not
standalone mutation provenance.  When mutation or birth genome provenance is
requested, this module reconstructs the inherited child from authoritative
native state and the committed absolute-tick RNG address, then emits a
mutation outcome only for an exact, bounded allele delta.  This module is
deliberately host-only: it reads detached capture/state projections after the
normal durable event acknowledgement and does not feed observations back into
the simulation.

The first engineering route uses one-tick science dispatches.  That gives the
adapter an exact pre/post state pair for every action and ecological delta;
the provider rejects typed capture for a larger temporal K until a native
per-tick state projection is added rather than manufacturing an inexact
causal join.  Each lowered world carries its own semantic observation
contract.  The adapter filters detailed rows at this boundary, while retaining
the small control and world-state integrity rows needed for frontier and
terminal validation.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from emergence_contracts import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
    COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1,
    COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1,
    COMPACT_SCIENCE_CAPTURE_SCHEMA_V1,
    COMPACT_SCIENCE_CHECKPOINT_FRONTIER_SCHEMA_V1,
    COMPACT_SCIENCE_DEATH_METRIC_ENCODING_V1,
    COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1,
    COMPACT_SCIENCE_DISPATCH_GZIP_ENCODING_V1,
    COMPACT_SCIENCE_DISPATCH_SCHEMA_V1,
    COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_ENCODING_V1,
    COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1,
    COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1,
    ContentDigest,
    SCIENCE_NATIVE_AGENT_STATE_SNAPSHOT_PAYLOAD_SCHEMA_VERSION,
    SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_VERSION,
    SchemaVersion,
    ScienceCompactCaptureContractError,
    canonical_sha256 as contract_canonical_sha256,
    reconstruct_compact_birth_genome_v1 as contract_reconstruct_compact_birth_genome_v1,
    validate_science_native_control_receipt_v1,
)
from emergence_contracts.science_instrument import ScienceRecordKindV1

from engine.campaigns.canonical import canonical_sha256
from engine.perf.cuda_full_evolution_v2.direct_sparse_runtime import (
    CudaFullEvolutionScienceCaptureV1,
)
from engine.perf.cuda_full_evolution_v2.layout import state_layout_v2
from engine.perf.cuda_full_evolution_v2.runtime import (
    CudaFullEvolutionSparseEventDispatchV3,
)
SCIENCE_CAPTURE_ADAPTER_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_science_capture_adapter_v1"
)
SCIENCE_CAPTURE_PRODUCER_COMPONENT_V1: Final[str] = (
    "cuda_full_evolution_v2_science_capture_v1"
)
SCIENCE_CAPTURE_PRODUCER_PHASES_V1: Final[tuple[int, ...]] = (
    5,
    10,
    11,
    12,
    18,
    19,
    20,
    21,
    22,
    23,
    40,
    50,
)
SCIENCE_CAPTURE_PRODUCER_CONTRACT_V1: Final[SchemaVersion] = SchemaVersion(
    "eiel_cuda_full_evolution_v2_science_capture_v1"
)
SCIENCE_CAPTURE_PAYLOAD_SCHEMA_V1: Final[SchemaVersion] = SchemaVersion(
    "eiel_cuda_full_evolution_v2_science_payload_v1"
)
_STATE_LAYOUT = state_layout_v2()
_STATE_BYTES = _STATE_LAYOUT.state_bytes_per_world
_CANONICAL_STATE_BYTES = _STATE_LAYOUT.canonical_checkpoint_bytes_per_world
_CAPACITY = 512
_ALIVE_WORDS = _CAPACITY // 32
_HEADER_WORDS = 24
_POLICY_HAPLOTYPES = 2
_POLICY_LOCI_PER_HAPLOTYPE = 112
_TRAIT_LOCI_PER_HAPLOTYPE = 16
_LINKED_LOCI = 128
_RECOMBINATION_BLOCKS = 16
_RECOMBINATION_DOMAIN = 0x41
_MUTATION_CAPABILITY_BIT = 1 << 5
_RECOMBINATION_CAPABILITY_BIT = 1 << 6
_RNG_SEMANTIC_EPOCH = 3
_HEADER_NEXT_ABSOLUTE_TICK = 1
_HEADER_BIRTH_CAPABILITY_COUNTER = 10
_HEADER_MUTATION_CAPABILITY_COUNTER = 11
_HEADER_RECOMBINATION_CAPABILITY_COUNTER = 12
_HEADER_BIRTH_COUNT = 15
_HEADER_MUTATION_COUNT = 17
_HEADER_RECOMBINATION_COUNT = 18
_MASK32 = 0xFFFFFFFF
_PHILOX_MULTIPLIER_0 = 0xD2511F53
_PHILOX_MULTIPLIER_1 = 0xCD9E8D57
_PHILOX_WEYL_0 = 0x9E3779B9
_PHILOX_WEYL_1 = 0xBB67AE85
_MUTATION_DOMAIN = 0x42
_MUTATION_THRESHOLD = 2048


def science_capture_producer_component_id_v1(phase: int) -> str:
    """Return the stable producer identity for one canonical capture phase.

    Canonical observation records do not retain the phase ordinal as a public
    field after finalization.  The observation registry reconstructs it from
    the producer component identity, so one component identity cannot safely
    span several phases.
    """

    if type(phase) is not int or phase not in SCIENCE_CAPTURE_PRODUCER_PHASES_V1:
        raise ValueError(f"unsupported CUDA V2 science capture phase: {phase!r}")
    return f"{SCIENCE_CAPTURE_PRODUCER_COMPONENT_V1}_phase_{phase:02d}"


def science_capture_producer_phase_registry_v1() -> dict[str, int]:
    """Return the exact registry required to validate native capture rows."""

    return {
        science_capture_producer_component_id_v1(phase): phase
        for phase in SCIENCE_CAPTURE_PRODUCER_PHASES_V1
    }


def _requested_observation_capabilities(
    cell: Mapping[str, object],
) -> frozenset[str] | None:
    """Return the explicit per-world acquisition contract.

    Lowered Science Instrument cells carry ``science_requested_observations``
    so a heterogeneous bundle can request only the causal primitives needed by
    each reducer.  A missing field is retained as a compatibility mode for
    direct callers from before the explicit bundle boundary; those callers get
    the complete engineering projection rather than silently losing records.
    An explicit empty sequence is meaningful and requests no detailed event or
    agent rows.
    """

    if "science_requested_observations" not in cell:
        return None
    raw = cell.get("science_requested_observations")
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("science_requested_observations must be a sequence")
    return frozenset(str(value) for value in raw)


def _observation_requested(
    cell: Mapping[str, object], *capability_ids: str
) -> bool:
    """Check a semantic capability without coupling to native field names."""

    requested = _requested_observation_capabilities(cell)
    if requested is None:
        return True
    if "causal_lifecycle" in requested:
        return True
    return bool(requested.intersection(capability_ids))


def _snapshot_requested(cell: Mapping[str, object], tick: int) -> bool:
    """Honor the world's resolved absolute-tick state snapshot contract.

    Event rows remain causal and may be emitted at every native dispatch, but
    state observations are acquisition points.  A missing acquisition block is
    retained as compatibility mode for direct callers and therefore permits
    every boundary; an explicit empty snapshot list requests no agent-state
    snapshots.
    """

    acquisition = cell.get("science_acquisition")
    if not isinstance(acquisition, Mapping):
        return True
    ticks = acquisition.get("snapshot_ticks")
    if ticks is None:
        return True
    if isinstance(ticks, (str, bytes)) or not isinstance(ticks, Sequence):
        raise ValueError("science acquisition snapshot_ticks must be a sequence")
    return tick in {int(value) for value in ticks}


def _digest(value: object) -> dict[str, str]:
    if isinstance(value, ContentDigest):
        return value.to_dict()
    if isinstance(value, Mapping) and set(value) == {"algorithm", "hex"}:
        if value.get("algorithm") == "sha256" and isinstance(value.get("hex"), str):
            return {"algorithm": "sha256", "hex": str(value["hex"])}
    encoded = repr(value).encode("utf-8")
    return {"algorithm": "sha256", "hex": hashlib.sha256(encoded).hexdigest()}


def _digest_text(value: object) -> str:
    if isinstance(value, bytes):
        digest = hashlib.sha256(value).hexdigest()
    else:
        digest = canonical_sha256(value)
    return f"sha256:{digest}"


def _canonical_digest(value: object) -> dict[str, str]:
    return contract_canonical_sha256(value).to_dict()


def _payload(
    value: Mapping[str, object],
    *,
    schema_version: SchemaVersion = SCIENCE_CAPTURE_PAYLOAD_SCHEMA_V1,
) -> dict[str, object]:
    return {
        "schema_version": str(schema_version),
        "payload": dict(value),
    }


def _normalise_applied_native_control(
    value: object,
) -> dict[str, object] | None:
    """Validate the exact native receipt carried by a science control row."""

    if value is None:
        return None
    try:
        return validate_science_native_control_receipt_v1(value)
    except Exception as exc:
        raise ValueError(f"applied native control receipt is invalid: {exc}") from exc


def _tensor_bytes(value: Any, *, field: str) -> bytes:
    if isinstance(value, bytes):
        return value
    try:
        tensor = value.detach().contiguous().cpu()
        return tensor.numpy().tobytes(order="C")
    except Exception as exc:  # pragma: no cover - defensive provider boundary
        raise ValueError(f"{field} is not a detached tensor projection") from exc


def _digest_bytes(value: bytes) -> dict[str, str]:
    return {"algorithm": "sha256", "hex": hashlib.sha256(value).hexdigest()}


def _u16(raw: bytes, offset: int, index: int = 0) -> int:
    return int(struct.unpack_from("<H", raw, offset + 2 * index)[0])


def _i16(raw: bytes, offset: int, index: int = 0) -> int:
    return int(struct.unpack_from("<h", raw, offset + 2 * index)[0])


def _u32(raw: bytes, offset: int, index: int = 0) -> int:
    return int(struct.unpack_from("<I", raw, offset + 4 * index)[0])


def _u64(raw: bytes, offset: int, index: int = 0) -> int:
    return int(struct.unpack_from("<Q", raw, offset + 8 * index)[0])


def _validate_native_state_header(
    raw: bytes,
    *,
    world_uid: int,
    expected_next_tick: int,
    field: str,
) -> tuple[int, ...]:
    """Validate the identity/frontier words used by native RNG and capture.

    The science adapter must never reconstruct a mutation from a state pair
    belonging to another world or another absolute-tick boundary.  The native
    state header is the authoritative binding for both checks.
    """

    header = _field_offset("scalar_header_counters")
    required_bytes = header + 8 * _HEADER_WORDS
    if len(raw) < required_bytes:
        raise ValueError(f"{field} is shorter than the native state header")
    words = tuple(_u64(raw, header, index) for index in range(_HEADER_WORDS))
    if words[0] != int(world_uid):
        raise ValueError(f"{field} world identity differs from the requested world")
    if words[_HEADER_NEXT_ABSOLUTE_TICK] != int(expected_next_tick):
        raise ValueError(
            f"{field} next absolute tick does not equal {expected_next_tick}"
        )
    if words[14] != _RNG_SEMANTIC_EPOCH:
        raise ValueError(f"{field} RNG semantic epoch is unsupported")
    return words


def _rotate_left32(value: int, shift: int) -> int:
    value &= _MASK32
    return ((value << shift) | (value >> (32 - shift))) & _MASK32


def _native_rng_tick_key(
    raw: bytes,
    *,
    world_uid: int,
    absolute_tick: int,
) -> tuple[int, int]:
    """Mirror the native absolute-tick RNG key derivation.

    This is intentionally kept local to the capture adapter.  It is an
    observation-side reconstruction of the already committed native address,
    not a second RNG authority and not a source of simulation draws.
    """

    words = _validate_native_state_header(
        raw,
        world_uid=world_uid,
        expected_next_tick=absolute_tick,
        field="native pre-dispatch state",
    )
    rng_key = words[13]
    key_low = rng_key & _MASK32
    key_high = (rng_key >> 32) & _MASK32
    uid_low = int(world_uid) & _MASK32
    uid_high = (int(world_uid) >> 32) & _MASK32
    tick_low = int(absolute_tick) & _MASK32
    tick_high = (int(absolute_tick) >> 32) & _MASK32
    first = (
        key_low
        ^ _rotate_left32(key_high, 7)
        ^ _rotate_left32(uid_low, 13)
        ^ _rotate_left32(uid_high, 23)
        ^ _rotate_left32(tick_low, 3)
        ^ _rotate_left32(tick_high, 17)
        ^ 0x9E3779B9
        ^ _RNG_SEMANTIC_EPOCH
    ) & _MASK32
    second = (
        key_high
        ^ _rotate_left32(key_low, 11)
        ^ _rotate_left32(uid_high, 5)
        ^ _rotate_left32(uid_low, 19)
        ^ _rotate_left32(tick_high, 29)
        ^ _rotate_left32(tick_low, 9)
        ^ 0xBB67AE85
    ) & _MASK32
    first = (first + _rotate_left32(second, 15)) & _MASK32
    second = (second ^ _rotate_left32(first, 21)) & _MASK32
    first = (first + _rotate_left32(second, 9)) & _MASK32
    return first, second


def _native_rng_address_counter(
    domain: int,
    subject: int,
    operation: int,
    draw: int,
) -> tuple[int, int, int, int]:
    """Mirror the native four-word counter address construction."""

    domain_low = int(domain) & _MASK32
    domain_high = (int(domain) >> 32) & _MASK32
    subject_low = int(subject) & _MASK32
    subject_high = (int(subject) >> 32) & _MASK32
    operation_low = int(operation) & _MASK32
    operation_high = (int(operation) >> 32) & _MASK32
    draw_low = int(draw) & _MASK32
    draw_high = (int(draw) >> 32) & _MASK32
    return (
        (
            subject_low
            ^ _rotate_left32(domain_high, 5)
            ^ _rotate_left32(operation_low, 17)
            ^ _rotate_left32(draw_high, 27)
        )
        & _MASK32,
        (
            subject_high
            ^ _rotate_left32(domain_low, 7)
            ^ _rotate_left32(operation_high, 19)
            ^ _rotate_left32(draw_low, 29)
        )
        & _MASK32,
        (
            domain_low
            ^ _rotate_left32(subject_high, 11)
            ^ _rotate_left32(operation_low, 23)
            ^ _rotate_left32(draw_high, 3)
        )
        & _MASK32,
        (
            domain_high
            ^ _rotate_left32(subject_low, 13)
            ^ _rotate_left32(operation_high, 25)
            ^ _rotate_left32(draw_low, 5)
        )
        & _MASK32,
    )


def _native_philox4x32_10(
    counter: Sequence[int], key: Sequence[int]
) -> tuple[int, int, int, int]:
    """Mirror the native Philox4x32-10 core for read-only reconstruction."""

    ctr = [int(value) & _MASK32 for value in counter]
    k0 = int(key[0]) & _MASK32
    k1 = int(key[1]) & _MASK32
    for _ in range(10):
        product0 = _PHILOX_MULTIPLIER_0 * ctr[0]
        product1 = _PHILOX_MULTIPLIER_1 * ctr[2]
        low0 = product0 & _MASK32
        high0 = (product0 >> 32) & _MASK32
        low1 = product1 & _MASK32
        high1 = (product1 >> 32) & _MASK32
        ctr = [
            (high1 ^ ctr[1] ^ k0) & _MASK32,
            low1,
            (high0 ^ ctr[3] ^ k1) & _MASK32,
            low0,
        ]
        k0 = (k0 + _PHILOX_WEYL_0) & _MASK32
        k1 = (k1 + _PHILOX_WEYL_1) & _MASK32
    return tuple(ctr)  # type: ignore[return-value]


def _native_child_recombination_bits(
    tick_key: Sequence[int],
    *,
    child_id: int,
    parent_a_id: int,
    parent_b_id: int,
    linked_block: int,
) -> int:
    domain = _RECOMBINATION_DOMAIN | (int(linked_block) << 32)
    block = _native_philox4x32_10(
        _native_rng_address_counter(
            domain, int(child_id), int(parent_a_id), int(parent_b_id)
        ),
        tick_key,
    )
    return (block[0] ^ block[1] ^ block[2] ^ block[3]) & _MASK32


def _native_child_mutation_draws(
    tick_key: Sequence[int], *, child_id: int, linked_locus: int
) -> tuple[int, int]:
    """Mirror the native domain-0x42 diploid mutation draws."""

    block = _native_philox4x32_10(
        _native_rng_address_counter(
            _MUTATION_DOMAIN, int(child_id), int(linked_locus), 0
        ),
        tick_key,
    )
    return (
        (block[0] ^ block[2]) & _MASK32,
        (block[1] ^ block[3]) & _MASK32,
    )


def _clamp_i8(value: int) -> int:
    # The native genome operator deliberately leaves one sentinel value on
    # either side of the signed storage range.  Storage decoding accepts the
    # full representation, but mutation application does not.
    return max(-127, min(127, int(value)))


def _clamp_i16(value: int) -> int:
    return max(-32767, min(32767, int(value)))


def _genome_state_id(projection: Mapping[str, Sequence[int]]) -> str:
    policy = projection.get("policy")
    trait = projection.get("trait")
    if (
        policy is None
        or trait is None
        or len(policy) != _POLICY_HAPLOTYPES * _POLICY_LOCI_PER_HAPLOTYPE
        or len(trait) != _POLICY_HAPLOTYPES * _TRAIT_LOCI_PER_HAPLOTYPE
    ):
        raise ValueError("native genome projection has an invalid shape")
    policy_bytes = bytes(int(value) & 0xFF for value in policy)
    trait_bytes = b"".join(
        struct.pack("<h", int(value)) for value in trait
    )
    return _digest_text(
        {"policy": policy_bytes.hex(), "trait": trait_bytes.hex()}
    )


def _native_genome_projection(
    agent: Mapping[str, object], *, field: str
) -> dict[str, tuple[int, ...]]:
    value = agent.get("_native_genome_projection")
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} lacks authoritative native genome projection")
    policy = value.get("policy")
    trait = value.get("trait")
    if (
        isinstance(policy, (str, bytes))
        or not isinstance(policy, Sequence)
        or isinstance(trait, (str, bytes))
        or not isinstance(trait, Sequence)
    ):
        raise ValueError(f"{field} has an invalid native genome projection")
    if any(type(item) is not int for item in policy) or any(
        type(item) is not int for item in trait
    ):
        raise ValueError(f"{field} genome projection values must be integers")
    projection = {
        "policy": tuple(policy),
        "trait": tuple(trait),
    }
    if any(value < -128 or value > 127 for value in projection["policy"]):
        raise ValueError(f"{field} has an out-of-range policy allele")
    if any(value < -32768 or value > 32767 for value in projection["trait"]):
        raise ValueError(f"{field} has an out-of-range trait allele")
    _genome_state_id(projection)
    return projection


def _authoritative_native_genome_state_id(
    agent: Mapping[str, object], *, field: str
) -> str:
    projection = _native_genome_projection(agent, field=field)
    state_id = agent.get("genome_state_id")
    if not isinstance(state_id, str):
        raise ValueError(f"{field} lacks an authoritative genome state identity")
    if _genome_state_id(projection) != state_id:
        raise ValueError(f"{field} genome state identity does not bind alleles")
    return state_id


@dataclass(frozen=True, slots=True)
class _NativeGenomeReconstruction:
    pre_state_id: str
    post_state_id: str
    changed_alleles: tuple[dict[str, int], ...]
    accepted_draw_details: tuple[dict[str, object], ...]
    accepted_draw_count: int
    saturated_noop_count: int


def _reconstruct_child_pre_mutation_genome(
    *,
    state_before: bytes,
    state_after: bytes,
    world_uid: int,
    absolute_tick: int,
    child_id: int,
    parent_a_id: int,
    parent_b_id: int,
    parent_a: Mapping[str, object],
    parent_b: Mapping[str, object],
    child: Mapping[str, object],
    native_control: Mapping[str, object],
    native_mutation_count: int,
) -> _NativeGenomeReconstruction:
    """Rebuild native inheritance and prove the observed child delta.

    The returned pre/post identities are derived from exact native allele
    projections.  Mutation draws use the native domain, threshold, direction,
    and integer clamp semantics.  The observed child must equal the fully
    replayed post-mutation projection and the native count must equal the
    number of accepted draws, including accepted draws that saturate at a
    representable boundary.
    """

    if native_mutation_count < 0:
        raise ValueError("native mutation count is negative")
    _validate_native_state_header(
        state_before,
        world_uid=world_uid,
        expected_next_tick=absolute_tick,
        field="native pre-dispatch state",
    )
    _validate_native_state_header(
        state_after,
        world_uid=world_uid,
        expected_next_tick=absolute_tick + 1,
        field="native post-dispatch state",
    )
    mask = native_control.get("capability_mask")
    if type(mask) is not int:
        raise ValueError("native mutation provenance control mask is invalid")
    mutation_enabled = bool(mask & _MUTATION_CAPABILITY_BIT)
    if native_mutation_count > 0 and not mutation_enabled:
        raise ValueError(
            "native mutation provenance count is nonzero while mutation is disabled"
        )
    parent_a_projection = _native_genome_projection(
        parent_a, field="native parent A"
    )
    parent_b_projection = _native_genome_projection(
        parent_b, field="native parent B"
    )
    child_projection = _native_genome_projection(child, field="native child")
    if parent_a.get("agent_id") != parent_a_id:
        raise ValueError("native parent A genome projection has the wrong agent identity")
    if parent_b.get("agent_id") != parent_b_id:
        raise ValueError("native parent B genome projection has the wrong agent identity")
    if child.get("agent_id") != child_id:
        raise ValueError("native child genome projection has the wrong agent identity")
    child_state_id = _authoritative_native_genome_state_id(
        child, field="native child"
    )

    tick_key = _native_rng_tick_key(
        state_before,
        world_uid=world_uid,
        absolute_tick=absolute_tick,
    )
    recombination_enabled = bool(mask & _RECOMBINATION_CAPABILITY_BIT)
    pre_policy = [0] * (_POLICY_HAPLOTYPES * _POLICY_LOCI_PER_HAPLOTYPE)
    pre_trait = [0] * (_POLICY_HAPLOTYPES * _TRAIT_LOCI_PER_HAPLOTYPE)
    for block in range(_RECOMBINATION_BLOCKS):
        recombination = (
            _native_child_recombination_bits(
                tick_key,
                child_id=child_id,
                parent_a_id=parent_a_id,
                parent_b_id=parent_b_id,
                linked_block=block,
            )
            if recombination_enabled
            else 0
        )
        hap_a = recombination & 1
        hap_b = (recombination >> 1) & 1
        for lane in range(8):
            linked_locus = block * 8 + lane
            if linked_locus < _POLICY_LOCI_PER_HAPLOTYPE:
                pre_policy[linked_locus] = parent_a_projection["policy"][
                    hap_a * _POLICY_LOCI_PER_HAPLOTYPE + linked_locus
                ]
                pre_policy[_POLICY_LOCI_PER_HAPLOTYPE + linked_locus] = (
                    parent_b_projection["policy"][
                        hap_b * _POLICY_LOCI_PER_HAPLOTYPE + linked_locus
                    ]
                )
            else:
                trait_locus = linked_locus - _POLICY_LOCI_PER_HAPLOTYPE
                pre_trait[trait_locus] = parent_a_projection["trait"][
                    hap_a * _TRAIT_LOCI_PER_HAPLOTYPE + trait_locus
                ]
                pre_trait[_TRAIT_LOCI_PER_HAPLOTYPE + trait_locus] = (
                    parent_b_projection["trait"][
                        hap_b * _TRAIT_LOCI_PER_HAPLOTYPE + trait_locus
                    ]
                )

    pre_projection = {"policy": tuple(pre_policy), "trait": tuple(pre_trait)}
    pre_state_id = _genome_state_id(pre_projection)
    expected_policy = list(pre_policy)
    expected_trait = list(pre_trait)
    changed: list[dict[str, int]] = []
    accepted_details: list[dict[str, object]] = []
    saturated_noop_count = 0
    accepted_draw_count = 0
    for linked_locus in range(_LINKED_LOCI):
        if mutation_enabled:
            draws = _native_child_mutation_draws(
                tick_key,
                child_id=child_id,
                linked_locus=linked_locus,
            )
        else:
            draws = (0, 0)
        for haplotype, draw in enumerate(draws):
            if not mutation_enabled or (draw & 0xFFFF) >= _MUTATION_THRESHOLD:
                continue
            if linked_locus < _POLICY_LOCI_PER_HAPLOTYPE:
                index = haplotype * _POLICY_LOCI_PER_HAPLOTYPE + linked_locus
                pre_value = expected_policy[index]
                post_value = _clamp_i8(
                    pre_value
                    + (1 if ((draw >> 16) & 1) != 0 else -1)
                )
                expected_policy[index] = post_value
                region = 0
            else:
                trait_locus = linked_locus - _POLICY_LOCI_PER_HAPLOTYPE
                index = haplotype * _TRAIT_LOCI_PER_HAPLOTYPE + trait_locus
                pre_value = expected_trait[index]
                post_value = _clamp_i16(
                    pre_value
                    + (1 if ((draw >> 16) & 1) != 0 else -1)
                )
                expected_trait[index] = post_value
                region = 1
            saturated_noop = post_value == pre_value
            accepted_draw_count += 1
            requested_delta = 1 if ((draw >> 16) & 1) != 0 else -1
            # ``post_value - pre_value`` is the applied/effective delta.  Keep
            # the requested direction as a separate field so a saturated
            # accepted draw remains truthful rather than looking unattempted.
            detail = {
                "genome_region": region,
                "haplotype": haplotype,
                "linked_locus": linked_locus,
                "pre_value": pre_value,
                "post_value": post_value,
                "requested_delta": requested_delta,
                "effective_delta": post_value - pre_value,
                "saturated_noop": saturated_noop,
            }
            accepted_details.append(detail)
            if saturated_noop:
                saturated_noop_count += 1
            else:
                changed.append(
                    {
                        "genome_region": region,
                        "haplotype": haplotype,
                        "linked_locus": linked_locus,
                        "pre_value": pre_value,
                        "post_value": post_value,
                    }
                )

    if accepted_draw_count != native_mutation_count:
        raise ValueError(
            "native mutation provenance accepted-draw count differs from native counter"
        )
    expected_post_projection = {
        "policy": tuple(expected_policy),
        "trait": tuple(expected_trait),
    }
    if expected_post_projection != child_projection:
        raise ValueError(
            "native mutation provenance replayed child genome differs from native post-state"
        )
    expected_post_state_id = _genome_state_id(expected_post_projection)
    if expected_post_state_id != child_state_id:
        raise ValueError(
            "native mutation provenance replayed post-genome digest differs from native state"
        )
    return _NativeGenomeReconstruction(
        pre_state_id=pre_state_id,
        post_state_id=expected_post_state_id,
        changed_alleles=tuple(changed),
        accepted_draw_details=tuple(accepted_details),
        accepted_draw_count=accepted_draw_count,
        saturated_noop_count=saturated_noop_count,
    )


def _field_offset(name: str) -> int:
    return _STATE_LAYOUT.field(name).offset_bytes


def _world_segment(raw: bytes, *, world_index: int) -> bytes:
    start = world_index * _STATE_BYTES
    end = start + _STATE_BYTES
    if start < 0 or end > len(raw):
        raise ValueError("state projection does not contain the requested world")
    return raw[start:end]


def _agent_rows(
    raw: bytes,
    *,
    world_uid: int,
    tick: int,
    selected_agent_ids: Collection[int] | None = None,
) -> dict[int, dict[str, object]]:
    """Decode all agents for a snapshot, or selected causal event subjects."""

    selected = None if selected_agent_ids is None else frozenset(selected_agent_ids)
    alive_offset = _field_offset("alive_words")
    ids_offset = _field_offset("agent_id")
    parents_a_offset = _field_offset("parent_a_id")
    parents_b_offset = _field_offset("parent_b_id")
    positions = _field_offset("position")
    energies = _field_offset("energy")
    healths = _field_offset("health")
    ages = _field_offset("age")
    birth_ticks = _field_offset("absolute_birth_tick")
    policy_offset = _field_offset("policy_alleles")
    policy_field = _STATE_LAYOUT.field("policy_alleles")
    policy_loci = int(policy_field.shape[0] * policy_field.shape[1])
    trait_offset = _field_offset("trait_alleles")
    trait_field = _STATE_LAYOUT.field("trait_alleles")
    trait_loci = int(trait_field.shape[0] * trait_field.shape[1])
    rows: dict[int, dict[str, object]] = {}
    alive_words = tuple(_u32(raw, alive_offset, index) for index in range(_ALIVE_WORDS))
    seen_agent_ids: set[int] = set()
    for slot in range(_CAPACITY):
        agent_id = _u64(raw, ids_offset, slot)
        if agent_id == 0:
            continue
        if agent_id in seen_agent_ids:
            raise ValueError(
                f"native state contains duplicate agent identity {agent_id}"
            )
        seen_agent_ids.add(agent_id)
        if selected is not None and agent_id not in selected:
            continue
        alive = bool(alive_words[slot // 32] & (1 << (slot % 32)))
        parent_a = _u64(raw, parents_a_offset, slot)
        parent_b = _u64(raw, parents_b_offset, slot)
        # The allele arrays are stored [haplotype, locus, slot].  Selecting the
        # slot by striding preserves the exact native genome bytes without
        # interpreting policy or trait semantics in this adapter.
        policy = bytearray()
        policy_values: list[int] = []
        policy_stride = _CAPACITY
        for index in range(policy_loci):
            raw_value = raw[policy_offset + index * policy_stride + slot]
            policy.append(raw_value)
            policy_values.append(raw_value if raw_value < 128 else raw_value - 256)
        trait = bytearray()
        trait_values: list[int] = []
        trait_stride = _CAPACITY
        for index in range(trait_loci):
            value = _i16(raw, trait_offset + 2 * index * trait_stride, slot)
            trait_values.append(value)
            trait.extend(
                raw[
                    trait_offset + 2 * (index * trait_stride + slot) :
                    trait_offset + 2 * (index * trait_stride + slot) + 2
                ]
            )
        genome_id = _digest_text({"policy": bytes(policy).hex(), "trait": bytes(trait).hex()})
        policy_id = _digest_text(bytes(policy))
        lineage_id = _digest_text(
            {"world_uid": world_uid, "agent_id": agent_id, "parent_a": parent_a, "parent_b": parent_b}
        )
        founder_id = _digest_text(
            {"world_uid": world_uid, "root_parent_a": parent_a or agent_id, "root_parent_b": parent_b}
        )
        agent_state_id = _digest_text(
            {
                "world_uid": world_uid,
                "tick": tick,
                "agent_id": agent_id,
                "alive": alive,
                "position": _u16(raw, positions, slot),
                "energy": _i16(raw, energies, slot),
                "health": _i16(raw, healths, slot),
                "age_ticks": _u16(raw, ages, slot),
                "genome_state_id": genome_id,
            }
        )
        rows[agent_id] = {
            "agent_state_id": agent_state_id,
            "agent_id": agent_id,
            "alive": alive,
            "position": _u16(raw, positions, slot),
            "energy": _i16(raw, energies, slot),
            "health": _i16(raw, healths, slot),
            "injury": 0,
            "age_ticks": _u16(raw, ages, slot),
            "absolute_birth_tick": _u64(raw, birth_ticks, slot),
            "genome_state_id": genome_id,
            "policy_id": policy_id,
            "founder_id": founder_id,
            "lineage_id": lineage_id,
            "lineage_depth": 0 if parent_a == 0 and parent_b == 0 else 1,
            "hereditary_package_state_ids": [],
            "linkage_state_id": genome_id,
            "parent_a_id": parent_a,
            "parent_b_id": parent_b,
            "slot": slot,
            "_native_genome_projection": {
                "policy": tuple(policy_values),
                "trait": tuple(trait_values),
            },
        }
    return rows


def _state_summary(raw: bytes, *, world_uid: int, tick: int) -> dict[str, object]:
    header = tuple(_u64(raw, _field_offset("scalar_header_counters"), index) for index in range(_HEADER_WORDS))
    canonical = raw[:_CANONICAL_STATE_BYTES]
    resource = raw[_field_offset("resource") : _field_offset("resource") + _STATE_LAYOUT.field("resource").bytes]
    resource_values = struct.unpack("<1024H", resource)
    return {
        "world_state_id": _digest_text({"world_uid": world_uid, "tick": tick, "canonical_state_sha256": hashlib.sha256(canonical).hexdigest()}),
        "field_set_version": "cuda_full_evolution_v2_state_projection_v1",
        "qualified_field_values": {
            "world_uid": world_uid,
            "absolute_tick": tick,
            "next_absolute_tick": header[1],
            "active_count": header[3],
            "population_count": header[3],
            "birth_count": header[15],
            "death_count": header[16],
            "mutation_count": header[17],
            "recombination_count": header[18],
            "resource_state_sha256": _digest_bytes(resource),
            "resource_total": sum(resource_values),
            "resource_mean": sum(resource_values) / len(resource_values),
            "resource_min": min(resource_values),
            "resource_max": max(resource_values),
            "resource_nonzero_count": sum(value != 0 for value in resource_values),
        },
        "snapshot_reason": "native_cuda_v2_science_capture_boundary",
    }


def _event_rows(dispatch: CudaFullEvolutionSparseEventDispatchV3) -> list[dict[str, int]]:
    if len(dispatch.packed_events) % 32:
        raise ValueError("sparse dispatch payload is not aligned to 32-byte rows")
    if len(dispatch.world_offsets) != len(dispatch.event_counts) + 1:
        raise ValueError("sparse dispatch world offsets do not match event counts")
    row_count = len(dispatch.packed_events) // 32
    if dispatch.world_offsets[-1] != row_count:
        raise ValueError("sparse dispatch world offsets do not cover packed events")
    if row_count == 0:
        return []
    rows: list[dict[str, int]] = []
    for world_index in range(len(dispatch.event_counts)):
        begin = dispatch.world_offsets[world_index]
        end = dispatch.world_offsets[world_index + 1]
        for ordinal in range(begin, end):
            agent_id, parent_a, parent_b, packed = struct.unpack_from(
                "<QQQQ", dispatch.packed_events, ordinal * 32
            )
            rows.append(
                {
                    "world_index": world_index,
                    "row_ordinal": ordinal,
                    "agent_id": int(agent_id),
                    "parent_a_id": int(parent_a),
                    "parent_b_id": int(parent_b),
                    "tick_offset": int(packed & 0x3F),
                    "event_kind": int((packed >> 6) & 0x3),
                    "subject_slot": int((packed >> 8) & 0x1FF),
                    "site_index": int((packed >> 17) & 0x3FF),
                }
            )
    return rows


def _validate_native_birth_binding(
    *,
    sparse_rows: Sequence[Mapping[str, int]],
    child_id: int,
    parent_a_id: int,
    parent_b_id: int,
    absolute_tick: int,
    child_slot: int,
    birth_site: int,
    before_agents: Mapping[int, Mapping[str, object]],
    after_agents: Mapping[int, Mapping[str, object]],
) -> Mapping[str, int]:
    """Bind one science birth to exactly one native event and state transition."""

    if child_id == 0 or parent_a_id == 0 or parent_b_id == 0:
        raise ValueError("native birth provenance contains a zero agent identity")
    if not 0 <= child_slot < _CAPACITY:
        raise ValueError("native birth provenance child slot is outside C512")
    if not 0 <= birth_site < 1024:
        raise ValueError("native birth provenance site is outside the native world")
    child_events = [
        event
        for event in sparse_rows
        if event.get("event_kind") == 2 and event.get("agent_id") == child_id
    ]
    matching = [
        event
        for event in child_events
        if event.get("parent_a_id") == parent_a_id
        and event.get("parent_b_id") == parent_b_id
        and event.get("tick_offset") == 0
        and event.get("subject_slot") == child_slot
        and event.get("site_index") == birth_site
    ]
    if len(child_events) != 1 or len(matching) != 1:
        raise ValueError(
            "native birth provenance requires exactly one sparse birth event "
            "matching child, parents, tick, slot, and site"
        )
    event = matching[0]
    if child_id in before_agents:
        raise ValueError("native birth child is present in the pre-dispatch state")
    for label, parent_id in (("A", parent_a_id), ("B", parent_b_id)):
        parent = before_agents.get(parent_id)
        if parent is None or parent.get("alive") is not True:
            raise ValueError(f"native birth parent {label} is absent or not alive")
    child = after_agents.get(child_id)
    if child is None or child.get("alive") is not True:
        raise ValueError("native birth child is absent or not alive post-dispatch")
    if (
        child.get("slot") != child_slot
        or child.get("parent_a_id") != parent_a_id
        or child.get("parent_b_id") != parent_b_id
        or child.get("absolute_birth_tick") != absolute_tick
        or child.get("position") != birth_site
    ):
        raise ValueError(
            "native birth child state does not bind parent IDs, birth tick, slot, or site"
        )
    return event


def _row(
    *,
    kind: ScienceRecordKindV1,
    tick: int,
    address: str,
    payload: Mapping[str, object],
    control_binding_sha256: Mapping[str, str],
    phase: int,
    before_sha256: Mapping[str, str] | None,
    after_sha256: Mapping[str, str] | None,
    cause_event_ids: Sequence[Mapping[str, str]] = (),
    payload_schema_version: SchemaVersion = SCIENCE_CAPTURE_PAYLOAD_SCHEMA_V1,
) -> dict[str, object]:
    return {
        "producer_component_id": science_capture_producer_component_id_v1(phase),
        "producer_contract_version": str(SCIENCE_CAPTURE_PRODUCER_CONTRACT_V1),
        "producer_phase_registry_ordinal": phase,
        "absolute_tick": tick,
        "semantic_source_address": address,
        "record_kind": kind.value,
        "record_kind_version": 1,
        "control_binding_sha256": dict(control_binding_sha256),
        "payload": _payload(payload, schema_version=payload_schema_version),
        "source_state_before_sha256": None if before_sha256 is None else dict(before_sha256),
        "source_state_after_sha256": None if after_sha256 is None else dict(after_sha256),
        "cause_event_ids": [dict(item) for item in sorted(cause_event_ids, key=lambda item: str(item.get("hex")))],
        "lifecycle_epoch": 0,
        "legacy_compatibility": None,
    }


def build_native_cuda_v2_observation_outputs_v1(
    *,
    dispatch: CudaFullEvolutionSparseEventDispatchV3,
    capture: CudaFullEvolutionScienceCaptureV1,
    state_before: Any,
    state_after: Any,
    world_cells: Mapping[str, Mapping[str, object]],
    applied_native_control: Mapping[str, object] | None = None,
    emit_control_binding: bool = True,
) -> dict[str, dict[str, object]]:
    """Build one route-neutral provider envelope per semantic world.

    The function is intentionally strict about one-tick capture.  A larger
    native temporal window would require per-intermediate-tick state fields to
    make the decision/consequence join truthful.
    """

    if capture.start_next_absolute_tick != dispatch.start_next_absolute_tick:
        raise ValueError("science capture and sparse dispatch start ticks differ")
    if capture.end_next_absolute_tick != capture.start_next_absolute_tick + dispatch.tick_count:
        raise ValueError("science capture and sparse dispatch frontiers differ")
    if dispatch.tick_count != 1:
        raise ValueError("typed science capture currently requires temporal K=1")
    if tuple(capture.world_uids) != tuple(dispatch.world_uids):
        raise ValueError("science capture world identities differ from dispatch")
    if type(emit_control_binding) is not bool:
        raise ValueError("emit_control_binding must be a boolean")
    resolved_native_control = _normalise_applied_native_control(applied_native_control)
    # Direct compatibility callers may already carry the provider receipt on
    # each lowered cell.  Reuse it when the explicit argument is omitted, but
    # reject any disagreement rather than allowing one world to carry a
    # different applied mask/vector provenance.
    for world_uid in dispatch.world_uids:
        cell_id = next(
            (
                key
                for key, cell in world_cells.items()
                if cell.get("seeds", {}).get("task") == world_uid
            ),
            None,
        )
        if cell_id is None:
            raise ValueError(f"no semantic cell is bound to CUDA world UID {world_uid}")
        cell_native_control = _normalise_applied_native_control(
            world_cells[cell_id].get("native_control")
        )
        if cell_native_control is None:
            continue
        if resolved_native_control is None:
            resolved_native_control = cell_native_control
        elif cell_native_control != resolved_native_control:
            raise ValueError(
                "science cells disagree on the applied native control receipt"
            )
    actions_ids = capture.action_agent_ids.tolist()
    actions = capture.action_values.tolist()
    birth_child_ids = capture.birth_child_ids.tolist()
    birth_parent_a_ids = capture.birth_parent_a_ids.tolist()
    birth_parent_b_ids = capture.birth_parent_b_ids.tolist()
    mutation_counts = capture.birth_mutation_counts.tolist()
    birth_child_slots = capture.birth_child_slots.tolist()
    birth_sites = capture.birth_sites.tolist()
    birth_counts = capture.birth_counts.tolist()
    sparse_rows = _event_rows(dispatch)
    sparse_rows_by_world: list[list[dict[str, int]]] = [
        [] for _ in dispatch.world_uids
    ]
    for event in sparse_rows:
        sparse_rows_by_world[event["world_index"]].append(event)
    state_before_raw = _tensor_bytes(state_before, field="state_before_projection")
    state_after_raw = _tensor_bytes(state_after, field="state_after_projection")
    outputs: dict[str, dict[str, object]] = {}
    for world_index, world_uid in enumerate(dispatch.world_uids):
        cell_id = next(
            (key for key, cell in world_cells.items() if cell.get("seeds", {}).get("task") == world_uid),
            None,
        )
        if cell_id is None:
            raise ValueError(f"no semantic cell is bound to CUDA world UID {world_uid}")
        cell = world_cells[cell_id]
        before = _world_segment(state_before_raw, world_index=world_index)
        after = _world_segment(state_after_raw, world_index=world_index)
        start_tick = capture.start_next_absolute_tick
        end_tick = capture.end_next_absolute_tick
        before_sha = _digest_bytes(before[:_CANONICAL_STATE_BYTES])
        after_sha = _digest_bytes(after[:_CANONICAL_STATE_BYTES])
        agent_snapshot_requested = _observation_requested(
            cell,
            "agent_state",
            "genome_identity",
            "lineage",
            "lineage_ancestry",
            "population_composition",
        )
        before_snapshot_requested = agent_snapshot_requested and _snapshot_requested(
            cell, start_tick
        )
        after_snapshot_requested = agent_snapshot_requested and _snapshot_requested(
            cell, end_tick
        )
        decision_requested = _observation_requested(
            cell, "decision", "action", "executed_action"
        )
        ecology_requested = _observation_requested(
            cell, "ecological_consequence", "ecology"
        )
        death_requested = _observation_requested(
            cell, "death", "lineage", "lineage_ancestry"
        )
        reproduction_requested = _observation_requested(
            cell,
            "reproductive_funnel",
            "reproduction_opportunity",
            "selection",
        )
        birth_requested = _observation_requested(
            cell,
            "birth",
            "birth_parentage",
            "parentage",
            "genome_identity",
            "lineage",
            "lineage_ancestry",
        )
        mutation_requested = _observation_requested(
            cell, "mutation", "mutation_provenance", "genome_delta"
        )
        provenance_requested = birth_requested or mutation_requested
        world_birth_count = int(birth_counts[world_index][0])
        if world_birth_count < 0:
            raise ValueError("native birth provenance birth count is negative")
        world_birth_events = [
            event
            for event in sparse_rows_by_world[world_index]
            if event["event_kind"] == 2
        ]
        if len(world_birth_events) != world_birth_count:
            raise ValueError(
                "native birth count differs from the exact sparse birth-event count"
            )
        native_pre_header: tuple[int, ...] | None = None
        native_post_header: tuple[int, ...] | None = None
        if resolved_native_control is not None and (
            provenance_requested or world_birth_count
        ):
            native_pre_header = _validate_native_state_header(
                before,
                world_uid=world_uid,
                expected_next_tick=start_tick,
                field="native pre-dispatch state",
            )
            native_post_header = _validate_native_state_header(
                after,
                world_uid=world_uid,
                expected_next_tick=end_tick,
                field="native post-dispatch state",
            )
            birth_count_delta = (
                native_post_header[_HEADER_BIRTH_COUNT]
                - native_pre_header[_HEADER_BIRTH_COUNT]
            )
            birth_capability_delta = (
                native_post_header[_HEADER_BIRTH_CAPABILITY_COUNTER]
                - native_pre_header[_HEADER_BIRTH_CAPABILITY_COUNTER]
            )
            expected_recombination_count = (
                world_birth_count * _RECOMBINATION_BLOCKS
                if resolved_native_control.get("capability_mask", 0)
                & _RECOMBINATION_CAPABILITY_BIT
                else 0
            )
            recombination_capability_delta = (
                native_post_header[_HEADER_RECOMBINATION_CAPABILITY_COUNTER]
                - native_pre_header[_HEADER_RECOMBINATION_CAPABILITY_COUNTER]
            )
            recombination_count_delta = (
                native_post_header[_HEADER_RECOMBINATION_COUNT]
                - native_pre_header[_HEADER_RECOMBINATION_COUNT]
            )
            if (
                birth_count_delta != world_birth_count
                or birth_capability_delta != world_birth_count
                or recombination_capability_delta != expected_recombination_count
                or recombination_count_delta != expected_recombination_count
            ):
                raise ValueError(
                    "native birth/recombination counter delta differs from "
                    "captured birth count or applied capability mask"
                )
        if provenance_requested and resolved_native_control is not None:
            assert native_pre_header is not None and native_post_header is not None
            expected_mutation_count = sum(
                int(mutation_counts[world_index][0][rank])
                for rank in range(world_birth_count)
            )
            mutation_count_delta = (
                native_post_header[_HEADER_MUTATION_COUNT]
                - native_pre_header[_HEADER_MUTATION_COUNT]
            )
            mutation_capability_delta = (
                native_post_header[_HEADER_MUTATION_CAPABILITY_COUNTER]
                - native_pre_header[_HEADER_MUTATION_CAPABILITY_COUNTER]
            )
            if (
                mutation_count_delta != expected_mutation_count
                or mutation_capability_delta != expected_mutation_count
            ):
                raise ValueError(
                    "native mutation counter delta differs from captured accepted draws"
                )
        if provenance_requested and world_birth_count and resolved_native_control is None:
            raise ValueError(
                "native birth provenance requires the applied native control receipt"
            )
        before_agent_ids: set[int] | None = (
            None if before_snapshot_requested else set()
        )
        after_agent_ids: set[int] | None = (
            None if after_snapshot_requested else set()
        )
        if decision_requested or ecology_requested:
            action_agent_ids = {
                int(agent_id)
                for agent_id in actions_ids[world_index][0]
                if int(agent_id) != 0
            }
            if before_agent_ids is not None:
                before_agent_ids.update(action_agent_ids)
            if after_agent_ids is not None:
                after_agent_ids.update(action_agent_ids)
        if death_requested and before_agent_ids is not None:
            before_agent_ids.update(
                event["agent_id"]
                for event in sparse_rows_by_world[world_index]
                if event["event_kind"] == 1 and event["agent_id"] != 0
            )
        if reproduction_requested or provenance_requested:
            for rank in range(world_birth_count):
                parent_a = int(birth_parent_a_ids[world_index][0][rank])
                parent_b = int(birth_parent_b_ids[world_index][0][rank])
                child_id = int(birth_child_ids[world_index][0][rank])
                mutation_count = int(mutation_counts[world_index][0][rank])
                if reproduction_requested and before_agent_ids is not None:
                    before_agent_ids.add(parent_a)
                if provenance_requested:
                    if before_agent_ids is not None:
                        before_agent_ids.update((parent_a, parent_b))
                        before_agent_ids.add(child_id)
                    if after_agent_ids is not None:
                        after_agent_ids.add(child_id)
        before_agents = (
            {}
            if before_agent_ids == set()
            else _agent_rows(
                before,
                world_uid=world_uid,
                tick=start_tick,
                selected_agent_ids=before_agent_ids,
            )
        )
        after_agents = (
            {}
            if after_agent_ids == set()
            else _agent_rows(
                after,
                world_uid=world_uid,
                tick=end_tick,
                selected_agent_ids=after_agent_ids,
            )
        )
        semantic_inputs = cell.get("science_semantic_inputs", {})
        if not isinstance(semantic_inputs, Mapping):
            semantic_inputs = {}
        acquisition = cell.get("science_acquisition", {})
        if not isinstance(acquisition, Mapping):
            acquisition = {}
        control_payload = {
            "binding_id": f"science-control-{cell_id}",
            "semantic_cell_id": cell_id,
            "seed_bindings": cell.get("science_seed_bindings", {}),
            "control_values": semantic_inputs.get("control_bindings", {}),
            "control_schedules": semantic_inputs.get("time", {}),
            "registered_profile_ids": [acquisition.get("profile_id")],
            "semantic_envelope_resolution_sha256": _canonical_digest(semantic_inputs),
            "semantic_applied_readback_sha256": _canonical_digest(
                {
                    "semantic_control_bindings": semantic_inputs.get(
                        "control_bindings", {}
                    ),
                    "applied_native_control": resolved_native_control,
                }
            ),
            "continuation_parent_execution_identity": None,
        }
        control_payload_schema = SCIENCE_CAPTURE_PAYLOAD_SCHEMA_V1
        if resolved_native_control is not None:
            control_payload["applied_native_control"] = dict(resolved_native_control)
            control_payload_schema = SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_VERSION
        control_digest = _canonical_digest(control_payload)
        state_rows: list[dict[str, object]] = []
        event_output: list[dict[str, object]] = []
        checkpoint_output: list[dict[str, object]] = []
        if emit_control_binding:
            state_rows.append(
                _row(
                    kind=ScienceRecordKindV1.CONTROL_BINDING,
                    tick=start_tick,
                    address=f"world/{cell_id}/control-binding",
                    payload=control_payload,
                    control_binding_sha256=control_digest,
                    phase=5,
                    before_sha256=before_sha,
                    after_sha256=None,
                    payload_schema_version=control_payload_schema,
                )
            )
        before_summary = _state_summary(before, world_uid=world_uid, tick=start_tick)
        after_summary = _state_summary(after, world_uid=world_uid, tick=end_tick)
        for tick, snapshot, state_sha, reason in (
            (start_tick, before_summary, before_sha, "native_pre_dispatch"),
            (end_tick, after_summary, after_sha, "native_post_dispatch"),
        ):
            snapshot = {**snapshot, "snapshot_reason": reason}
            state_rows.append(
                _row(
                    kind=ScienceRecordKindV1.WORLD_STATE_SNAPSHOT,
                    tick=tick,
                    address=f"world/{cell_id}/tick/{tick}/world-state/{reason}",
                    payload=snapshot,
                    control_binding_sha256=control_digest,
                    phase=10,
                    before_sha256=before_sha if tick == start_tick else None,
                    after_sha256=after_sha if tick == end_tick else None,
                )
            )
        if agent_snapshot_requested:
            if before_snapshot_requested:
                for agent_id, agent in sorted(before_agents.items()):
                    payload = {
                        key: value
                        for key, value in agent.items()
                        if key
                        not in {
                            "parent_a_id",
                            "parent_b_id",
                            "slot",
                            "_native_genome_projection",
                        }
                    }
                    state_rows.append(
                        _row(
                            kind=ScienceRecordKindV1.AGENT_STATE_SNAPSHOT,
                            tick=start_tick,
                            address=f"world/{cell_id}/tick/{start_tick}/agent/{agent_id}/pre-state",
                            payload=payload,
                            control_binding_sha256=control_digest,
                            phase=11,
                            before_sha256=before_sha,
                            after_sha256=None,
                            payload_schema_version=(
                                SCIENCE_NATIVE_AGENT_STATE_SNAPSHOT_PAYLOAD_SCHEMA_VERSION
                            ),
                        )
                    )
            if after_snapshot_requested:
                for agent_id, agent in sorted(after_agents.items()):
                    payload = {
                        key: value
                        for key, value in agent.items()
                        if key
                        not in {
                            "parent_a_id",
                            "parent_b_id",
                            "slot",
                            "_native_genome_projection",
                        }
                    }
                    state_rows.append(
                        _row(
                            kind=ScienceRecordKindV1.AGENT_STATE_SNAPSHOT,
                            tick=end_tick,
                            address=f"world/{cell_id}/tick/{end_tick}/agent/{agent_id}/post-state",
                            payload=payload,
                            control_binding_sha256=control_digest,
                            phase=12,
                            before_sha256=None,
                            after_sha256=after_sha,
                            payload_schema_version=(
                                SCIENCE_NATIVE_AGENT_STATE_SNAPSHOT_PAYLOAD_SCHEMA_VERSION
                            ),
                        )
                    )

        def event_identity(event: Mapping[str, int], tick: int, kind: str) -> dict[str, str]:
            return _canonical_digest(
                {
                    "schema_version": "eiel_cuda_v2_science_native_event_id_v1",
                    "world_uid": world_uid,
                    "tick": tick,
                    "kind": kind,
                    "ordinal": event.get("row_ordinal"),
                    "agent_id": event.get("agent_id"),
                }
            )

        for event in sparse_rows_by_world[world_index] if death_requested else ():
            tick = start_tick + event["tick_offset"]
            event_kind = event["event_kind"]
            event_id = event_identity(event, tick, "birth" if event_kind == 2 else "death")
            if event_kind == 1:
                agent = before_agents.get(event["agent_id"], {})
                payload = {
                    "agent_id": event["agent_id"],
                    "terminal_agent_state_id": agent.get("agent_state_id", _digest_text({"agent_id": event["agent_id"], "tick": tick})),
                    "genome_state_id": agent.get("genome_state_id", _digest_text(event["agent_id"])),
                    "lineage_id": agent.get("lineage_id", _digest_text(event["agent_id"])),
                    "cause_class": "native_sparse_death_event",
                    "reason_code": "native_transition",
                    "causing_event_ids": [event_id],
                }
                event_output.append(
                    _row(
                        kind=ScienceRecordKindV1.DEATH,
                        tick=tick,
                        address=f"world/{cell_id}/tick/{tick}/death/{event['agent_id']}",
                        payload=payload,
                        control_binding_sha256=control_digest,
                        phase=20,
                        before_sha256=before_sha,
                        after_sha256=after_sha,
                        cause_event_ids=(event_id,),
                    )
                )

        for slot in range(_CAPACITY) if decision_requested or ecology_requested else ():
            agent_id = int(actions_ids[world_index][0][slot])
            if agent_id == 0:
                continue
            action = int(actions[world_index][0][slot])
            before_agent = before_agents.get(agent_id)
            after_agent = after_agents.get(agent_id)
            if before_agent is None:
                continue
            decision_id = _digest_text({"world_uid": world_uid, "tick": start_tick, "agent_id": agent_id, "slot": slot})
            decision_payload = {
                "decision_id": decision_id,
                "agent_id": agent_id,
                "agent_state_before_id": before_agent["agent_state_id"],
                "genome_state_id": before_agent["genome_state_id"],
                "policy_id": before_agent["policy_id"],
                "decision_input_state_id": before_agent["agent_state_id"],
                "qualified_policy_output_summary": {"native_action_code": action},
                "selected_action_id": f"native-action-{action}",
                "executed_action_id": f"native-action-{action}",
                "execution_status": "EXECUTED" if after_agent is not None else "TERMINATED_AFTER_EXECUTION",
                "execution_reason_code": "native_controller_action",
            }
            if decision_requested:
                event_output.append(
                    _row(
                        kind=ScienceRecordKindV1.DECISION_ACTION,
                        tick=start_tick,
                        address=f"world/{cell_id}/tick/{start_tick}/decision/{agent_id}/{slot}",
                        payload=decision_payload,
                        control_binding_sha256=control_digest,
                        phase=21,
                        before_sha256=before_sha,
                        after_sha256=after_sha,
                    )
                )
            if ecology_requested:
                after_or_before = after_agent or before_agent
                event_output.append(
                    _row(
                        kind=ScienceRecordKindV1.ECOLOGICAL_TRANSACTION,
                        tick=end_tick,
                        address=f"world/{cell_id}/tick/{end_tick}/consequence/{agent_id}/{slot}",
                        payload={
                            "transaction_id": _digest_text({"decision_id": decision_id, "tick": end_tick}),
                            "decision_id": decision_id,
                            "actor_agent_id": agent_id,
                            "affected_subject_ids": [agent_id],
                            "executed_action_id": f"native-action-{action}",
                            "phenotype_state_id": after_or_before["genome_state_id"],
                            "world_state_before_id": before_summary["world_state_id"],
                            "world_state_after_id": after_summary["world_state_id"],
                            "resource_deltas": {},
                            "energy_delta": (0 if after_agent is None else int(after_agent["energy"]) - int(before_agent["energy"])),
                            "health_delta": (0 if after_agent is None else int(after_agent["health"]) - int(before_agent["health"])),
                            "injury_delta": 0,
                            "movement_cost": (0 if after_agent is None else abs(int(after_agent["position"]) - int(before_agent["position"]))),
                            "outcome_class": "EXECUTED_NATIVE_ACTION",
                        },
                        control_binding_sha256=control_digest,
                        phase=22,
                        before_sha256=before_sha,
                        after_sha256=after_sha,
                    )
                )

        for rank in (
            range(world_birth_count)
            if reproduction_requested or provenance_requested
            else ()
        ):
            child_id = int(birth_child_ids[world_index][0][rank])
            parent_a = int(birth_parent_a_ids[world_index][0][rank])
            parent_b = int(birth_parent_b_ids[world_index][0][rank])
            mutation_count = int(mutation_counts[world_index][0][rank])
            child_slot = int(birth_child_slots[world_index][0][rank])
            birth_site = int(birth_sites[world_index][0][rank])
            tick = start_tick
            if provenance_requested:
                birth_event = _validate_native_birth_binding(
                    sparse_rows=sparse_rows_by_world[world_index],
                    child_id=child_id,
                    parent_a_id=parent_a,
                    parent_b_id=parent_b,
                    absolute_tick=tick,
                    child_slot=child_slot,
                    birth_site=birth_site,
                    before_agents=before_agents,
                    after_agents=after_agents,
                )
            else:
                birth_event = next(
                    (
                        item
                        for item in sparse_rows_by_world[world_index]
                        if item["world_index"] == world_index
                        and item["event_kind"] == 2
                        and item["agent_id"] == child_id
                    ),
                    {"row_ordinal": rank, "agent_id": child_id},
                )
            event_id = event_identity(birth_event, tick, "birth")
            opportunity_id = _digest_text({"world_uid": world_uid, "tick": tick, "child_id": child_id, "rank": rank})
            attempt_id = _digest_text({"opportunity_id": opportunity_id, "birth": True})
            parent_a_agent = before_agents.get(parent_a)
            parent_b_agent = before_agents.get(parent_b)
            child_state = after_agents.get(child_id)
            strict_genome_provenance = provenance_requested
            if strict_genome_provenance:
                if (
                    parent_a_agent is None
                    or parent_b_agent is None
                    or child_state is None
                ):
                    raise ValueError(
                        "native birth provenance unavailable: authoritative "
                        "parent and child genome state is missing"
                    )
                parent_a_state = _authoritative_native_genome_state_id(
                    parent_a_agent, field="native parent A"
                )
                parent_b_state = _authoritative_native_genome_state_id(
                    parent_b_agent, field="native parent B"
                )
                child_genome = _authoritative_native_genome_state_id(
                    child_state, field="native child"
                )
            else:
                parent_a_state = parent_a_agent.get(
                    "genome_state_id", _digest_text({"agent_id": parent_a, "tick": tick})
                ) if parent_a_agent is not None else _digest_text(
                    {"agent_id": parent_a, "tick": tick}
                )
                parent_b_state = parent_b_agent.get(
                    "genome_state_id", _digest_text({"agent_id": parent_b, "tick": tick})
                ) if parent_b_agent is not None else _digest_text(
                    {"agent_id": parent_b, "tick": tick}
                )
                child_state = {} if child_state is None else child_state
                child_genome = child_state.get(
                    "genome_state_id",
                    _digest_text(
                        {"child_id": child_id, "tick": tick, "mutation_count": mutation_count}
                    ),
                )
            pre_mutation_genome = child_genome
            reconstruction: _NativeGenomeReconstruction | None = None
            if provenance_requested:
                if resolved_native_control is None:
                    raise ValueError(
                        "native birth/mutation provenance unavailable: applied native "
                        "control receipt is required for exact reconstruction"
                    )
                reconstruction = _reconstruct_child_pre_mutation_genome(
                    state_before=before,
                    state_after=after,
                    world_uid=world_uid,
                    absolute_tick=tick,
                    child_id=child_id,
                    parent_a_id=parent_a,
                    parent_b_id=parent_b,
                    parent_a=parent_a_agent,
                    parent_b=parent_b_agent,
                    child=child_state,
                    native_control=resolved_native_control,
                    native_mutation_count=mutation_count,
                )
                pre_mutation_genome = reconstruction.pre_state_id
                child_genome = reconstruction.post_state_id
            common = {
                "opportunity_id": opportunity_id,
                "subject_agent_id": parent_a,
                "subject_agent_state_id": before_agents.get(parent_a, {}).get("agent_state_id", _digest_text(parent_a)),
                "eligible": True,
                "candidate_parent_ids": [parent_a, parent_b],
                "selected_parent_ids": [parent_a, parent_b],
                "selection_mode_id": "native_reproduction_selection",
                "outcome_class": "COMMITTED",
                "reason_code": "native_birth_committed",
            }
            if reproduction_requested:
                event_output.append(
                    _row(
                        kind=ScienceRecordKindV1.REPRODUCTION_OPPORTUNITY,
                        tick=tick,
                        address=f"world/{cell_id}/tick/{tick}/reproduction-opportunity/{rank}",
                        payload=common,
                        control_binding_sha256=control_digest,
                        phase=18,
                        before_sha256=before_sha,
                        after_sha256=after_sha,
                        cause_event_ids=(event_id,),
                    )
                )
            if birth_requested:
                event_output.append(
                    _row(
                        kind=ScienceRecordKindV1.BIRTH_ATTEMPT,
                        tick=tick,
                        address=f"world/{cell_id}/tick/{tick}/birth-attempt/{rank}",
                        payload={
                            "birth_attempt_id": attempt_id,
                            "opportunity_id": opportunity_id,
                            "parent_agent_ids": [parent_a, parent_b],
                            "parent_genome_state_ids": [parent_a_state, parent_b_state],
                            "birth_mode_id": "native_ordinary_birth",
                            "inheritance_operator_id": "native_genome_v2_inheritance",
                            "recombination_operator_id": "native_recombination_profile",
                            "attempt_result_class": "COMMITTED",
                            "reason_code": "native_birth_committed",
                        },
                        control_binding_sha256=control_digest,
                        phase=19,
                        before_sha256=before_sha,
                        after_sha256=after_sha,
                        cause_event_ids=(event_id,),
                    )
                )
                event_output.append(
                    _row(
                        kind=ScienceRecordKindV1.BIRTH_COMMITTED,
                        tick=tick,
                        address=f"world/{cell_id}/tick/{tick}/birth-committed/{child_id}/{rank}",
                        payload={
                            "birth_attempt_id": attempt_id,
                            "opportunity_id": opportunity_id,
                            "child_agent_id": child_id,
                            "parent_agent_ids": [parent_a, parent_b],
                            "parent_genome_state_ids": [parent_a_state, parent_b_state],
                            "birth_mode_id": "native_ordinary_birth",
                            "inheritance_operator_id": "native_genome_v2_inheritance",
                            "recombination_operator_id": "native_recombination_profile",
                            "parent_hereditary_package_state_ids": [],
                            "child_pre_mutation_genome_state_id": pre_mutation_genome,
                            "child_post_birth_genome_state_id": child_genome,
                            "child_hereditary_package_state_ids": [],
                            "child_linkage_state_id": child_state.get("linkage_state_id", child_genome),
                            "founder_id": child_state.get("founder_id", _digest_text(child_id)),
                            "lineage_id": child_state.get("lineage_id", _digest_text({"child": child_id, "parents": [parent_a, parent_b]})),
                            "lineage_depth": child_state.get("lineage_depth", 1),
                        },
                        control_binding_sha256=control_digest,
                        phase=20,
                        before_sha256=before_sha,
                        after_sha256=after_sha,
                        cause_event_ids=(event_id,),
                    )
                )
            if mutation_count > 0 and mutation_requested:
                if reconstruction is None:
                    raise ValueError(
                        "native mutation provenance reconstruction is unavailable"
                    )
                result_class = (
                    "MUTATION_SATURATED_NOOP"
                    if not reconstruction.changed_alleles
                    else (
                        "MUTATION_APPLIED_WITH_SATURATED_NOOPS"
                        if reconstruction.saturated_noop_count
                        else "MUTATION_APPLIED"
                    )
                )
                event_output.append(
                    _row(
                        kind=ScienceRecordKindV1.MUTATION_OUTCOME,
                        tick=tick,
                        address=f"world/{cell_id}/tick/{tick}/mutation/{child_id}/{rank}",
                        payload={
                            "mutation_opportunity_id": opportunity_id,
                            "mutation_attempt_id": _digest_text({"attempt_id": attempt_id, "mutation": True}),
                            "subject_agent_id": child_id,
                            "origin_agent_id": child_id,
                            "origin_absolute_tick": tick,
                            "attempted": True,
                            "result_class": result_class,
                            "operator_id": "native_genome_v2_mutation",
                            "mutation_class_id": "native_genome_v2_point_mutation",
                            "affected_coordinates": {
                                "accepted_draws": reconstruction.accepted_draw_count,
                                "changed_alleles": [
                                    dict(item)
                                    for item in reconstruction.changed_alleles
                                ],
                                "accepted_draw_details": [
                                    dict(item)
                                    for item in reconstruction.accepted_draw_details
                                ],
                                "saturated_noop_count": reconstruction.saturated_noop_count,
                                "native_mutation_count": mutation_count,
                            },
                            "pre_value_or_state_digest": _canonical_digest(
                                {"genome_state_id": pre_mutation_genome}
                            ),
                            # Keep the pre/post value digest shape symmetric:
                            # the state identity is the value being compared,
                            # while child/physical identity is already carried
                            # by the surrounding event fields.
                            "post_value_or_state_digest": _canonical_digest(
                                {"genome_state_id": child_genome}
                            ),
                            "bounded_delta": {
                                "mutation_count": mutation_count,
                                "accepted_draws": reconstruction.accepted_draw_count,
                                "changed_allele_count": len(
                                    reconstruction.changed_alleles
                                ),
                                "saturated_noop_count": reconstruction.saturated_noop_count,
                            },
                            "pre_genome_state_id": pre_mutation_genome,
                            "post_genome_state_id": child_genome,
                            "reason_code": "native_exact_genome_delta",
                        },
                        control_binding_sha256=control_digest,
                        phase=23,
                        before_sha256=before_sha,
                        after_sha256=after_sha,
                        cause_event_ids=(event_id,),
                    )
                )

        outputs[cell_id] = {
            "schema_version": "eiel_science_native_observation_output_v1",
            "event_dispatch": event_output,
            "state_projection": state_rows,
            "checkpoint_projection": checkpoint_output,
            "effects_performed": False,
        }
    return outputs


def build_native_cuda_v2_checkpoint_marker_output_v1(
    *,
    cell_id: str,
    absolute_tick: int,
    checkpoint_occurrence_id: str,
    authoritative_state_sha256: Mapping[str, str],
    observation_prefix_sha256: Mapping[str, str],
    next_world_local_sequence: int,
    continuation_parent_identity: Mapping[str, str] | None,
    control_binding_sha256: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build one canonical checkpoint marker without touching native state."""

    if control_binding_sha256 is None:
        # Retain the historical direct-call shape for callers that have no
        # preceding control row.  A native-bound provider must pass the exact
        # semantic control-binding digest, and validates that requirement at
        # its publication boundary.
        control_digest = _canonical_digest(
            {
                "cell_id": cell_id,
                "absolute_tick": absolute_tick,
                "checkpoint_occurrence_id": checkpoint_occurrence_id,
            }
        )
    else:
        control_digest = _digest(control_binding_sha256)
    row = _row(
        kind=ScienceRecordKindV1.CHECKPOINT_MARKER,
        tick=absolute_tick,
        address=f"world/{cell_id}/tick/{absolute_tick}/checkpoint-marker",
        payload={
            "checkpoint_occurrence_id": checkpoint_occurrence_id,
            "checkpoint_contract_id": "eiel_cuda_full_evolution_science_checkpoint_extension_v1",
            "authoritative_state_sha256": dict(authoritative_state_sha256),
            "observation_prefix_sha256": dict(observation_prefix_sha256),
            "next_world_local_sequence": next_world_local_sequence,
            "continuation_parent_identity": (
                None
                if continuation_parent_identity is None
                else dict(continuation_parent_identity)
            ),
        },
        control_binding_sha256=control_digest,
        phase=40,
        before_sha256=None,
        after_sha256=dict(authoritative_state_sha256),
    )
    return {
        "schema_version": "eiel_science_native_observation_output_v1",
        "event_dispatch": [],
        "state_projection": [],
        "checkpoint_projection": [row],
        "effects_performed": False,
    }


def build_native_cuda_v2_completion_output_v1(
    *,
    cell_id: str,
    absolute_tick: int,
    completion_class: str,
    terminal_state_sha256: Mapping[str, str],
    terminal_population_count: int,
    terminal_record_count: int,
    canonical_record_stream_sha256: Mapping[str, str],
    checkpoint_occurrence_ids: Sequence[str],
    control_binding_sha256: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the terminal world marker after the final frontier is committed."""

    completion_id = _digest_text(
        {
            "cell_id": cell_id,
            "absolute_tick": absolute_tick,
            "terminal_record_count": terminal_record_count,
            "canonical_record_stream_sha256": dict(canonical_record_stream_sha256),
        }
    )
    if control_binding_sha256 is None:
        # See the checkpoint-marker compatibility note above.  Provider-owned
        # native science output always supplies the exact control digest.
        control_digest = _canonical_digest(
            {
                "cell_id": cell_id,
                "absolute_tick": absolute_tick,
                "completion_id": completion_id,
            }
        )
    else:
        control_digest = _digest(control_binding_sha256)
    row = _row(
        kind=ScienceRecordKindV1.WORLD_COMPLETION,
        tick=absolute_tick,
        address=f"world/{cell_id}/tick/{absolute_tick}/world-completion",
        payload={
            "completion_id": completion_id,
            "completion_class": completion_class,
            "terminal_state_sha256": dict(terminal_state_sha256),
            "terminal_population_count": terminal_population_count,
            "terminal_absolute_tick": absolute_tick,
            "next_absolute_tick": absolute_tick,
            "terminal_record_count": terminal_record_count,
            "canonical_record_stream_sha256": dict(canonical_record_stream_sha256),
            "checkpoint_occurrence_ids": list(checkpoint_occurrence_ids),
            "reason_code": "native_terminal_horizon",
        },
        control_binding_sha256=control_digest,
        phase=50,
        before_sha256=None,
        after_sha256=dict(terminal_state_sha256),
    )
    return {
        "schema_version": "eiel_science_native_observation_output_v1",
        "event_dispatch": [row],
        "state_projection": [],
        "checkpoint_projection": [],
        "effects_performed": False,
    }


__all__ = (
    "SCIENCE_CAPTURE_ADAPTER_SCHEMA_V1",
    "SCIENCE_CAPTURE_PAYLOAD_SCHEMA_V1",
    "SCIENCE_CAPTURE_PRODUCER_COMPONENT_V1",
    "SCIENCE_CAPTURE_PRODUCER_CONTRACT_V1",
    "SCIENCE_CAPTURE_PRODUCER_PHASES_V1",
    "build_native_cuda_v2_checkpoint_marker_output_v1",
    "build_native_cuda_v2_completion_output_v1",
    "build_native_cuda_v2_observation_outputs_v1",
    "science_capture_producer_component_id_v1",
    "science_capture_producer_phase_registry_v1",
)
