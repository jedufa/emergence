"""Deterministic CPU fixtures for later HAPB V2 kernel bake-offs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Final, Iterator

import numpy as np

from engine.perf.hapb_v2.exactness_contract_v1 import (
    ACTION_COUNT_V1,
    CONTRACT_ID_V1,
    DEFAULT_DRIVE_WEIGHTS_V1,
    DRIVE_WIDTH_MIN_V1,
    HIDDEN_SIZE_V1,
    OBSERVATION_SIZE_V1,
    POLICY_WIDTH_V1,
    DriveWeightsV1,
    evaluate_fixed_mlp_policy_logits_from_pack_exact_v1,
    exactness_contract_sha256_v1,
    logits_sha256_v1,
)


FIXTURE_SCHEMA_V1: Final[str] = "eiel_hapb_v2_policy_eval_fixture_v1"
FIXTURE_SEED_V1: Final[int] = 0x48415042  # 'HAPB'
B48_REPRESENTATIVE_ROWS_V1: Final[int] = 48


@dataclass(frozen=True)
class PolicyEvalFixtureV1:
    name: str
    packs: np.ndarray
    observations: np.ndarray
    logical_slots: np.ndarray
    content_ids: np.ndarray
    drives: np.ndarray
    attack_bias: np.ndarray
    action_temperature: np.ndarray
    drive_logit_gain: np.ndarray
    expected_logits: np.ndarray
    expected_logits_sha256: tuple[str, ...]
    fixture_sha256: str

    @property
    def rows(self) -> int:
        return int(self.packs.shape[0])


def _rng(name: str) -> np.random.Generator:
    seed = hashlib.sha256(f"{FIXTURE_SEED_V1}:{name}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(seed[:8], "little"))


def _pack(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(POLICY_WIDTH_V1, dtype=np.float32) * np.float32(0.05)


def _observation(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(OBSERVATION_SIZE_V1, dtype=np.float32) * np.float32(0.25)


def _slot_scalars(
    rng: np.random.Generator,
    slot_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drives = rng.random((slot_count, DRIVE_WIDTH_MIN_V1), dtype=np.float32)
    attack = rng.standard_normal(slot_count, dtype=np.float32) * np.float32(0.1)
    temperature = np.float32(0.2) + rng.random(slot_count, dtype=np.float32)
    gain = rng.random(slot_count, dtype=np.float32)
    return drives, attack, temperature, gain


def _digest_fixture(
    name: str,
    packs: np.ndarray,
    observations: np.ndarray,
    logical_slots: np.ndarray,
    content_ids: np.ndarray,
    expected_logits: np.ndarray,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(name.encode("utf-8"))
    hasher.update(CONTRACT_ID_V1.encode("utf-8"))
    hasher.update(exactness_contract_sha256_v1().encode("utf-8"))
    for array in (packs, observations, logical_slots, content_ids, expected_logits):
        hasher.update(np.ascontiguousarray(array).tobytes(order="C"))
    return hasher.hexdigest()


def _finalize(
    name: str,
    packs: np.ndarray,
    observations: np.ndarray,
    logical_slots: np.ndarray,
    content_ids: np.ndarray,
    drives: np.ndarray,
    attack_bias: np.ndarray,
    action_temperature: np.ndarray,
    drive_logit_gain: np.ndarray,
    drive_weights: DriveWeightsV1 = DEFAULT_DRIVE_WEIGHTS_V1,
) -> PolicyEvalFixtureV1:
    expected = np.empty((int(packs.shape[0]), ACTION_COUNT_V1), dtype=np.float32)
    hashes: list[str] = []
    for row in range(int(packs.shape[0])):
        logits = evaluate_fixed_mlp_policy_logits_from_pack_exact_v1(
            packs[row],
            observations[row],
            slot=int(logical_slots[row]),
            drives=drives,
            attack_bias=attack_bias,
            action_temperature=action_temperature,
            drive_logit_gain=drive_logit_gain,
            drive_weights=drive_weights,
        )
        expected[row] = logits
        hashes.append(logits_sha256_v1(logits))
    digest = _digest_fixture(
        name, packs, observations, logical_slots, content_ids, expected
    )
    return PolicyEvalFixtureV1(
        name=name,
        packs=packs,
        observations=observations,
        logical_slots=logical_slots,
        content_ids=content_ids,
        drives=drives,
        attack_bias=attack_bias,
        action_temperature=action_temperature,
        drive_logit_gain=drive_logit_gain,
        expected_logits=expected,
        expected_logits_sha256=tuple(hashes),
        fixture_sha256=digest,
    )


def tiny_single_row_fixture_v1() -> PolicyEvalFixtureV1:
    rng = _rng("tiny_single_row")
    drives, attack, temperature, gain = _slot_scalars(rng, 1)
    return _finalize(
        "tiny_single_row",
        packs=_pack(rng).reshape(1, POLICY_WIDTH_V1),
        observations=_observation(rng).reshape(1, OBSERVATION_SIZE_V1),
        logical_slots=np.asarray([0], dtype=np.int64),
        content_ids=np.asarray([1001], dtype=np.int64),
        drives=drives,
        attack_bias=attack,
        action_temperature=temperature,
        drive_logit_gain=gain,
    )


def small_multi_row_fixture_v1() -> PolicyEvalFixtureV1:
    rng = _rng("small_multi_row")
    rows = 4
    drives, attack, temperature, gain = _slot_scalars(rng, rows)
    packs = np.stack([_pack(rng) for _ in range(rows)], axis=0)
    observations = np.stack([_observation(rng) for _ in range(rows)], axis=0)
    return _finalize(
        "small_multi_row",
        packs=packs,
        observations=observations,
        logical_slots=np.arange(rows, dtype=np.int64),
        content_ids=np.arange(2001, 2001 + rows, dtype=np.int64),
        drives=drives,
        attack_bias=attack,
        action_temperature=temperature,
        drive_logit_gain=gain,
    )


def b48_like_fixture_v1() -> PolicyEvalFixtureV1:
    rng = _rng("b48_like")
    rows = B48_REPRESENTATIVE_ROWS_V1
    slot_count = 64
    drives, attack, temperature, gain = _slot_scalars(rng, slot_count)
    packs = np.stack([_pack(rng) for _ in range(rows)], axis=0)
    observations = np.stack([_observation(rng) for _ in range(rows)], axis=0)
    slots = (np.arange(rows, dtype=np.int64) * 3) % slot_count
    return _finalize(
        "b48_like",
        packs=packs,
        observations=observations,
        logical_slots=slots,
        content_ids=np.arange(3000, 3000 + rows, dtype=np.int64),
        drives=drives,
        attack_bias=attack,
        action_temperature=temperature,
        drive_logit_gain=gain,
    )


def duplicate_content_fixture_v1() -> PolicyEvalFixtureV1:
    rng = _rng("duplicate_content")
    shared = _pack(rng)
    rows = 5
    drives, attack, temperature, gain = _slot_scalars(rng, rows)
    packs = np.stack([shared for _ in range(rows)], axis=0)
    observations = np.stack([_observation(rng) for _ in range(rows)], axis=0)
    return _finalize(
        "duplicate_content",
        packs=packs,
        observations=observations,
        logical_slots=np.arange(rows, dtype=np.int64),
        content_ids=np.full((rows,), 4001, dtype=np.int64),
        drives=drives,
        attack_bias=attack,
        action_temperature=temperature,
        drive_logit_gain=gain,
    )


def mixed_content_fixture_v1() -> PolicyEvalFixtureV1:
    rng = _rng("mixed_content")
    unique_a = _pack(rng)
    unique_b = _pack(rng)
    rows = 6
    packs = np.stack(
        [unique_a, unique_a, unique_b, unique_a, unique_b, _pack(rng)],
        axis=0,
    )
    observations = np.stack([_observation(rng) for _ in range(rows)], axis=0)
    drives, attack, temperature, gain = _slot_scalars(rng, rows)
    return _finalize(
        "mixed_content",
        packs=packs,
        observations=observations,
        logical_slots=np.arange(rows, dtype=np.int64),
        content_ids=np.asarray([5001, 5001, 5002, 5001, 5002, 5003], dtype=np.int64),
        drives=drives,
        attack_bias=attack,
        action_temperature=temperature,
        drive_logit_gain=gain,
    )


def all_policy_eval_fixtures_v1() -> tuple[PolicyEvalFixtureV1, ...]:
    return (
        tiny_single_row_fixture_v1(),
        small_multi_row_fixture_v1(),
        b48_like_fixture_v1(),
        duplicate_content_fixture_v1(),
        mixed_content_fixture_v1(),
    )


def iter_fixture_metadata_v1() -> Iterator[dict[str, object]]:
    for fixture in all_policy_eval_fixtures_v1():
        yield {
            "schema_version": FIXTURE_SCHEMA_V1,
            "name": fixture.name,
            "rows": fixture.rows,
            "observation_size": OBSERVATION_SIZE_V1,
            "hidden_size": HIDDEN_SIZE_V1,
            "action_count": ACTION_COUNT_V1,
            "policy_width": POLICY_WIDTH_V1,
            "unique_content_ids": int(np.unique(fixture.content_ids).size),
            "fixture_sha256": fixture.fixture_sha256,
            "expected_logits_sha256": list(fixture.expected_logits_sha256),
            "contract_id": CONTRACT_ID_V1,
            "contract_sha256": exactness_contract_sha256_v1(),
        }
