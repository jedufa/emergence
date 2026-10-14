"""Cacheless V2 policy-window scratch. Addresses are semantic episode ordinals."""

from __future__ import annotations

from dataclasses import dataclass

from engine.tw_rngv2.normal import box_muller_pair
from engine.tw_rngv2.philox import philox4x32_10
from engine.tw_rngv2.status import Status
from engine.tw_rngv2.uniforms import u32_to_f32_half_open

from .contract import (
    DOMAIN_HAPB_POLICY_WINDOW_CROSSOVER,
    DOMAIN_HAPB_POLICY_WINDOW_MASK,
    DOMAIN_HAPB_POLICY_WINDOW_MUTATION,
    NAMESPACE_DEFAULT,
    SEMANTIC_EPOCH,
)
from .identity import hapb_counter_words, hapb_key_words, hapb_validate_fields


@dataclass(slots=True)
class HapbPolicyWindowScratch:
    crossover: list[list[bool]]
    mutation: list[list[float]]
    mask: list[list[bool]]
    host_tape_bytes: int
    host_gaussian_cache_bytes: int
    status: Status


def _words(
    *,
    world_uid: int,
    domain_id: int,
    subject_uid: int,
    absolute_tick: int,
    operation_id: int,
    namespace: int,
) -> tuple[int, int, int, int]:
    return philox4x32_10(
        hapb_counter_words(
            subject_uid=subject_uid,
            absolute_tick=absolute_tick,
            operation_id=operation_id,
            draw_block=0,
        ),
        hapb_key_words(
            rng_namespace_key64=namespace,
            world_uid=world_uid,
            domain_id=domain_id,
            semantic_epoch=SEMANTIC_EPOCH,
        ),
    )


def fill_policy_window_v2(
    *,
    world_uid: int,
    semantic_origin: int,
    episodes: int,
    width: int,
    mutation_enable: bool = False,
    mutation_rate: float = 0.0,
    mutation_std: float = 0.0,
    namespace: int = NAMESPACE_DEFAULT,
    absolute_tick: int = 0,
) -> HapbPolicyWindowScratch:
    if episodes < 0 or width < 0 or semantic_origin < 0:
        return HapbPolicyWindowScratch([], [], [], 0, 0, Status.RNG_FIELD_OVERFLOW)
    crossover: list[list[bool]] = []
    mutation: list[list[float]] = []
    mask: list[list[bool]] = []
    for episode in range(episodes):
        subject = semantic_origin + episode
        cross_row: list[bool] = []
        for column in range(width):
            status = hapb_validate_fields(
                world_uid=world_uid,
                domain_id=DOMAIN_HAPB_POLICY_WINDOW_CROSSOVER,
                semantic_epoch=SEMANTIC_EPOCH,
                subject_uid=subject,
                absolute_tick=absolute_tick,
                operation_id=column,
                draw_block=0,
            )
            if status is not Status.RNG_OK:
                return HapbPolicyWindowScratch([], [], [], 0, 0, status)
            words = _words(
                world_uid=world_uid,
                domain_id=DOMAIN_HAPB_POLICY_WINDOW_CROSSOVER,
                subject_uid=subject,
                absolute_tick=absolute_tick,
                operation_id=column,
                namespace=namespace,
            )
            cross_row.append(u32_to_f32_half_open(words[0]) < 0.5)
        crossover.append(cross_row)
        if mutation_enable and mutation_std > 0.0:
            mut_row: list[float] = []
            mask_row: list[bool] = []
            for column in range(width):
                keep = True
                if mutation_rate < 1.0:
                    mask_words = _words(
                        world_uid=world_uid,
                        domain_id=DOMAIN_HAPB_POLICY_WINDOW_MASK,
                        subject_uid=subject,
                        absolute_tick=absolute_tick,
                        operation_id=column,
                        namespace=namespace,
                    )
                    keep = u32_to_f32_half_open(mask_words[0]) < mutation_rate
                mask_row.append(keep)
                if keep:
                    mut_words = _words(
                        world_uid=world_uid,
                        domain_id=DOMAIN_HAPB_POLICY_WINDOW_MUTATION,
                        subject_uid=subject,
                        absolute_tick=absolute_tick,
                        operation_id=column,
                        namespace=namespace,
                    )
                    z0, _z1, nstatus = box_muller_pair(mut_words)
                    if nstatus is not Status.RNG_OK:
                        return HapbPolicyWindowScratch([], [], [], 0, 0, nstatus)
                    mut_row.append(float(mutation_std) * float(z0))
                else:
                    mut_row.append(0.0)
            mutation.append(mut_row)
            mask.append(mask_row)
        else:
            mutation.append([0.0] * width)
            mask.append([False] * width)
    return HapbPolicyWindowScratch(
        crossover=crossover,
        mutation=mutation,
        mask=mask,
        host_tape_bytes=0,
        host_gaussian_cache_bytes=0,
        status=Status.RNG_OK,
    )
