"""Default-off V2 oracles for remaining EIEL_FULL_EVOLUTION_V1 RNG domains.

Does not change the qualified 0x0330-0x0332 mutation/birth consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from agents.modular_genome_v1 import canon_float
from evolution.diploid_inheritance_v1 import (
    DIPLOID_INHERITANCE_PROFILE,
    DIPLOID_LINEAGE_SCHEMA_VERSION,
    DIPLOID_LINKAGE_BLOCKS,
    DIPLOID_SCALAR_LOCI,
    LOCUS_BOUNDS,
    MUTATION_SIGMA,
    DiploidScalarGenomeV1,
    diploid_genome_from_genome,
)
from evolution.genome_upgrade_v2 import (
    HEREDITARY_LOCUS_COUNT,
    MEIOSIS_DRAW_COUNT,
    MUTATION_SEMANTIC_DRAW_COUNT,
    Q10_SCALE,
    GenomeUpgradeV2V1,
    assemble_child_from_tape_v1,
)
from engine.tw_rngv2.address import EventAddress
from engine.tw_rngv2.bounded import bounded_uint32
from engine.tw_rngv2.contract import NAMESPACE_DEFAULT, SEMANTIC_EPOCH
from engine.tw_rngv2.normal import box_muller_pair
from engine.tw_rngv2.philox import philox4x32_10
from engine.tw_rngv2.status import Status
from engine.tw_rngv2.uniforms import u64_from_lanes, u64_to_f64_half_open

from .contracts import TensorWorldContractError


DOMAIN_ENVIRONMENT = 0x0100
DOMAIN_ACTION_CHOICE = 0x0210
DOMAIN_MATING = 0x0300
DOMAIN_BIRTH_PROPOSAL = 0x0310
DOMAIN_RECOMBINATION = 0x0320
OP_DIPLOID_MEIOSIS = 0
OP_DIPLOID_GATE = 32
OP_DIPLOID_HOMOLOG = 96
OP_DIPLOID_DELTA = 160
DOMAIN_GV2_HOMOLOG = 0x0340
DOMAIN_GV2_GATE = 0x0341
DOMAIN_GV2_DELTA = 0x0342
DOMAIN_JUNGLE_HIT = 0x0360
DOMAIN_POLICY_BIRTH = 0x0370
DOMAIN_LIFECYCLE = 0x0400
OP_FOOD_SPAWN = 0


def _words(address: EventAddress, namespace: int, ordinal: int = 0) -> tuple[int, int, int, int]:
    block, _lane = divmod(int(ordinal), 4)
    return philox4x32_10(address.counter_words(block), address.key_words(namespace))


def _uniform(address: EventAddress, namespace: int) -> float:
    words = _words(address, namespace)
    return u64_to_f64_half_open(u64_from_lanes(words[0], words[1]))


def _bounded(address: EventAddress, namespace: int, upper: int) -> tuple[int, int, Status]:
    return bounded_uint32(upper, lambda r, addr=address: _words(addr, namespace, r)[r % 4])


@dataclass(slots=True)
class V2Gv2InheritanceTape:
    donor_homologs: tuple[int, ...]
    mutation_homologs: tuple[int, ...]
    mutation_enabled: tuple[int, ...]
    mutation_delta_q10: tuple[int, ...]
    status: Status


def draw_gv2_inheritance_tape_v2(
    *,
    world_uid: int,
    absolute_tick: int,
    subject_uid: int,
    mutation_rate_q10: int,
    mutation_step_q10: int,
    namespace: int = NAMESPACE_DEFAULT,
) -> V2Gv2InheritanceTape:
    if not 0 <= mutation_rate_q10 <= Q10_SCALE or not 0 <= mutation_step_q10 <= Q10_SCALE:
        raise TensorWorldContractError("GV2 V2 mutation Q10 is out of range")
    donors: list[int] = []
    for draw in range(MEIOSIS_DRAW_COUNT):
        addr = EventAddress(
            world_uid, DOMAIN_RECOMBINATION, subject_uid, absolute_tick, draw, SEMANTIC_EPOCH
        )
        value, _ordinal, status = _bounded(addr, namespace, 2)
        if status is not Status.RNG_OK:
            return V2Gv2InheritanceTape((), (), (), (), status)
        donors.append(int(value))
    homologs: list[int] = []
    enabled: list[int] = []
    deltas: list[int] = []
    threshold = mutation_rate_q10 / Q10_SCALE
    width = mutation_step_q10 * 2 + 1
    for locus in range(HEREDITARY_LOCUS_COUNT):
        h_addr = EventAddress(
            world_uid, DOMAIN_GV2_HOMOLOG, subject_uid, absolute_tick, locus, SEMANTIC_EPOCH
        )
        homolog, _o, status = _bounded(h_addr, namespace, 2)
        if status is not Status.RNG_OK:
            return V2Gv2InheritanceTape((), (), (), (), status)
        g_addr = EventAddress(
            world_uid, DOMAIN_GV2_GATE, subject_uid, absolute_tick, locus, SEMANTIC_EPOCH
        )
        gate = _uniform(g_addr, namespace)
        d_addr = EventAddress(
            world_uid, DOMAIN_GV2_DELTA, subject_uid, absolute_tick, locus, SEMANTIC_EPOCH
        )
        raw, _o, status = _bounded(d_addr, namespace, width)
        if status is not Status.RNG_OK:
            return V2Gv2InheritanceTape((), (), (), (), status)
        homologs.append(int(homolog))
        enabled.append(int(gate < threshold))
        deltas.append(int(raw) - mutation_step_q10)
    return V2Gv2InheritanceTape(
        tuple(donors), tuple(homologs), tuple(enabled), tuple(deltas), Status.RNG_OK
    )


def assemble_child_from_v2_v1(
    parent_a: GenomeUpgradeV2V1,
    parent_b: GenomeUpgradeV2V1,
    *,
    world_uid: int,
    absolute_tick: int,
    subject_uid: int,
    mutation_rate_q10: int,
    mutation_step_q10: int,
    namespace: int = NAMESPACE_DEFAULT,
) -> tuple[GenomeUpgradeV2V1, dict[str, object]]:
    tape = draw_gv2_inheritance_tape_v2(
        world_uid=world_uid,
        absolute_tick=absolute_tick,
        subject_uid=subject_uid,
        mutation_rate_q10=mutation_rate_q10,
        mutation_step_q10=mutation_step_q10,
        namespace=namespace,
    )
    if tape.status is not Status.RNG_OK:
        raise TensorWorldContractError(f"GV2 V2 inheritance failed: {tape.status.name}")
    child, receipt = assemble_child_from_tape_v1(
        parent_a,
        parent_b,
        donor_homologs=tape.donor_homologs,
        mutation_homologs=tape.mutation_homologs,
        mutation_enabled=tape.mutation_enabled,
        mutation_delta_q10=tape.mutation_delta_q10,
    )
    receipt.update(
        {
            "rng_semantics_version": 2,
            "rng_family": "philox4x32_10",
            "host_tape": False,
            "total_semantic_draws": MEIOSIS_DRAW_COUNT + MUTATION_SEMANTIC_DRAW_COUNT,
            "domains": {
                "recombination": DOMAIN_RECOMBINATION,
                "gv2_homolog": DOMAIN_GV2_HOMOLOG,
                "gv2_gate": DOMAIN_GV2_GATE,
                "gv2_delta": DOMAIN_GV2_DELTA,
            },
            "subject_uid": int(subject_uid),
            "absolute_tick": int(absolute_tick),
            "world_uid": int(world_uid),
        }
    )
    return child, receipt


def _diploid_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assemble_diploid_child_from_v2_v1(
    parent_a: object,
    parent_b: object,
    *,
    world_uid: int,
    absolute_tick: int,
    subject_uid: int,
    mutation_enabled: bool,
    resource_handling_enabled: bool,
    mutation_sigma_multiplier: float = 1.0,
    namespace: int = NAMESPACE_DEFAULT,
) -> tuple[DiploidScalarGenomeV1, dict[str, float], dict[str, object]]:
    parent_genomes = (
        diploid_genome_from_genome(parent_a),
        diploid_genome_from_genome(parent_b),
    )
    gametes: list[list[float]] = [[], []]
    donor_rows: list[dict[str, object]] = []
    selected_by_parent: list[dict[str, int]] = [{}, {}]
    draw_index = 0
    for parent_index, parent_genome in enumerate(parent_genomes):
        for block_id, loci in DIPLOID_LINKAGE_BLOCKS:
            addr = EventAddress(
                world_uid,
                DOMAIN_BIRTH_PROPOSAL,
                subject_uid,
                absolute_tick,
                OP_DIPLOID_MEIOSIS + draw_index,
                SEMANTIC_EPOCH,
            )
            homolog, _ordinal, status = _bounded(addr, namespace, 2)
            if status is not Status.RNG_OK:
                raise TensorWorldContractError(
                    f"diploid V2 meiosis failed: {status.name}"
                )
            selected_by_parent[parent_index][block_id] = int(homolog)
            donor_rows.append(
                {
                    "draw_index": draw_index,
                    "parent": "A" if parent_index == 0 else "B",
                    "block_id": block_id,
                    "homolog": int(homolog),
                }
            )
            draw_index += 1
        locus_block = {
            locus: block_id
            for block_id, block_loci in DIPLOID_LINKAGE_BLOCKS
            for locus in block_loci
        }
        gametes[parent_index] = [
            parent_genome.allele(selected_by_parent[parent_index][locus_block[locus]], locus)
            for locus in DIPLOID_SCALAR_LOCI
        ]

    assembled = DiploidScalarGenomeV1(
        allele_0=tuple(gametes[0]),
        allele_1=tuple(gametes[1]),
    )
    allele_rows = [list(assembled.allele_0), list(assembled.allele_1)]
    pre_mutation_rate = assembled.phenotype()["mutation_rate"]
    mutation_rows: list[dict[str, object]] = []
    if mutation_enabled:
        for locus_index, locus in enumerate(DIPLOID_SCALAR_LOCI):
            gate_addr = EventAddress(
                world_uid,
                DOMAIN_BIRTH_PROPOSAL,
                subject_uid,
                absolute_tick,
                OP_DIPLOID_GATE + locus_index,
                SEMANTIC_EPOCH,
            )
            homolog_addr = EventAddress(
                world_uid,
                DOMAIN_BIRTH_PROPOSAL,
                subject_uid,
                absolute_tick,
                OP_DIPLOID_HOMOLOG + locus_index,
                SEMANTIC_EPOCH,
            )
            delta_addr = EventAddress(
                world_uid,
                DOMAIN_BIRTH_PROPOSAL,
                subject_uid,
                absolute_tick,
                OP_DIPLOID_DELTA + locus_index,
                SEMANTIC_EPOCH,
            )
            gate_draw = _uniform(gate_addr, namespace)
            homolog, _o, status = _bounded(homolog_addr, namespace, 2)
            if status is not Status.RNG_OK:
                raise TensorWorldContractError(
                    f"diploid V2 homolog failed: {status.name}"
                )
            z0, _z1, nstatus = box_muller_pair(_words(delta_addr, namespace))
            if nstatus is not Status.RNG_OK:
                raise TensorWorldContractError(
                    f"diploid V2 delta failed: {nstatus.name}"
                )
            gate_rate = 0.2 if locus == "mutation_rate" else pre_mutation_rate
            operator_enabled = locus in MUTATION_SIGMA and (
                locus != "resource_handling_axis" or resource_handling_enabled
            )
            pre_value = float(allele_rows[int(homolog)][locus_index])
            semantic_change = bool(operator_enabled and gate_draw < gate_rate)
            delta = (
                float(z0)
                * float(MUTATION_SIGMA.get(locus, 0.0))
                * float(mutation_sigma_multiplier)
                if semantic_change
                else 0.0
            )
            lo, hi = LOCUS_BOUNDS[locus]
            post_value = canon_float(
                min(hi, max(lo, pre_value + delta)),
                field=f"diploid.mutation.{locus}",
            )
            allele_rows[int(homolog)][locus_index] = post_value
            mutation_rows.append(
                {
                    "locus_id": locus,
                    "selected_homolog": int(homolog),
                    "pre_value": pre_value,
                    "delta": canon_float(delta, field=f"diploid.delta.{locus}"),
                    "post_value": post_value,
                    "semantic_change": bool(post_value != pre_value),
                    "operator_enabled": operator_enabled,
                }
            )

    child = DiploidScalarGenomeV1(
        allele_0=tuple(allele_rows[0]),
        allele_1=tuple(allele_rows[1]),
    )
    phenotype = child.phenotype()
    receipt: dict[str, object] = {
        "schema_version": DIPLOID_LINEAGE_SCHEMA_VERSION,
        "profile": DIPLOID_INHERITANCE_PROFILE,
        "rng_semantics_version": 2,
        "rng_family": "philox4x32_10",
        "host_tape": False,
        "parent_order": "canonical_hereditary_payload_sha256_v1",
        "meiosis": {
            "draw_count": draw_index,
            "donors": donor_rows,
            "domain": DOMAIN_BIRTH_PROPOSAL,
        },
        "mutation": {
            "enabled": bool(mutation_enabled),
            "semantic_draw_count": (
                3 * len(DIPLOID_SCALAR_LOCI) if mutation_enabled else 0
            ),
            "events": mutation_rows,
            "domain": DOMAIN_BIRTH_PROPOSAL,
        },
        "genotype_sha256": child.to_payload()["genotype_sha256"],
        "phenotype_sha256": _diploid_sha256(phenotype),
        "subject_uid": int(subject_uid),
        "absolute_tick": int(absolute_tick),
        "world_uid": int(world_uid),
    }
    return child, phenotype, receipt


def action_gumbel_uniforms_v2(
    *,
    world_uid: int,
    absolute_tick: int,
    subject_uid: int,
    action_count: int,
    namespace: int = NAMESPACE_DEFAULT,
) -> tuple[float, ...]:
    return tuple(
        _uniform(
            EventAddress(
                world_uid, DOMAIN_ACTION_CHOICE, subject_uid, absolute_tick, action, SEMANTIC_EPOCH
            ),
            namespace,
        )
        for action in range(action_count)
    )


def food_spawn_uniform_v2(
    *,
    world_uid: int,
    absolute_tick: int,
    attempt: int,
    namespace: int = NAMESPACE_DEFAULT,
) -> float:
    return _uniform(
        EventAddress(
            world_uid, DOMAIN_ENVIRONMENT, 0, absolute_tick, attempt, SEMANTIC_EPOCH
        ),
        namespace,
    )


def jungle_hit_uniform_v2(
    *,
    world_uid: int,
    absolute_tick: int,
    subject_uid: int,
    namespace: int = NAMESPACE_DEFAULT,
) -> float:
    return _uniform(
        EventAddress(
            world_uid, DOMAIN_JUNGLE_HIT, subject_uid, absolute_tick, 0, SEMANTIC_EPOCH
        ),
        namespace,
    )


def mating_shuffle_choice_v2(
    *,
    world_uid: int,
    absolute_tick: int,
    subject_uid: int,
    remaining: int,
    namespace: int = NAMESPACE_DEFAULT,
) -> tuple[int, Status]:
    if remaining <= 1:
        return 0, Status.RNG_OK
    addr = EventAddress(
        world_uid, DOMAIN_MATING, subject_uid, absolute_tick, remaining - 1, SEMANTIC_EPOCH
    )
    value, _ordinal, status = _bounded(addr, namespace, remaining)
    return int(value), status


def drive_noise_pair_v2(
    *,
    world_uid: int,
    absolute_tick: int,
    subject_uid: int,
    channel: int,
    namespace: int = NAMESPACE_DEFAULT,
) -> tuple[float, float, Status]:
    addr = EventAddress(
        world_uid, DOMAIN_LIFECYCLE, subject_uid, absolute_tick, channel, SEMANTIC_EPOCH
    )
    return box_muller_pair(_words(addr, namespace))


def domain_ids_unique() -> bool:
    ids = (
        DOMAIN_ENVIRONMENT,
        DOMAIN_ACTION_CHOICE,
        DOMAIN_MATING,
        DOMAIN_BIRTH_PROPOSAL,
        DOMAIN_RECOMBINATION,
        DOMAIN_GV2_HOMOLOG,
        DOMAIN_GV2_GATE,
        DOMAIN_GV2_DELTA,
        DOMAIN_JUNGLE_HIT,
        DOMAIN_POLICY_BIRTH,
        DOMAIN_LIFECYCLE,
        0x0330,
        0x0331,
        0x0332,
    )
    return len(ids) == len(set(ids))
