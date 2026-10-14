"""Shared deterministic control schedule for cross-path inheritance owners."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Mapping, Sequence


SCHEDULE_SCHEMA_VERSION = "inheritance_cross_path_control_schedule_v1"


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def canonical_package_sha256(
    locus_names: Sequence[str], values: Sequence[float]
) -> str:
    if len(locus_names) != len(values):
        raise ValueError("locus names and values are misaligned")
    return canonical_json_sha256(
        {
            str(name): float(value)
            for name, value in zip(locus_names, values, strict=True)
        }
    )


def schedule_word(
    contract_sha256: str,
    schedule_namespace: str,
    seed: int,
    generation: int,
    opportunity_ordinal: int,
    channel: str,
    *extra: Any,
) -> int:
    """Return one versioned word shared by Python, CPU tensor, and CUDA tensor."""

    if (
        len(str(contract_sha256)) != 64
        or not str(schedule_namespace)
        or int(generation) < 0
        or int(opportunity_ordinal) < 0
        or not str(channel)
    ):
        raise ValueError("cross-path schedule identity is invalid")
    payload = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "contract_sha256": str(contract_sha256),
        "schedule_namespace": str(schedule_namespace),
        "seed": int(seed),
        "generation": int(generation),
        "opportunity_ordinal": int(opportunity_ordinal),
        "channel": str(channel),
        "extra": list(extra),
    }
    return int.from_bytes(
        hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).digest()[:8],
        "big",
    )


def scheduled_parent_ids(
    eligible_parent_ids: Sequence[int],
    *,
    contract_sha256: str,
    schedule_namespace: str,
    seed: int,
    generation: int,
    opportunity_ordinal: int,
) -> tuple[int, int]:
    parents = tuple(sorted(int(value) for value in eligible_parent_ids))
    if len(parents) < 2 or len(set(parents)) != len(parents):
        raise ValueError("eligible parent IDs must be unique and contain two parents")
    first = schedule_word(
        contract_sha256,
        schedule_namespace,
        seed,
        generation,
        opportunity_ordinal,
        "parent_a",
    ) % len(parents)
    offset = 1 + (
        schedule_word(
            contract_sha256,
            schedule_namespace,
            seed,
            generation,
            opportunity_ordinal,
            "parent_b",
        )
        % (len(parents) - 1)
    )
    return parents[first], parents[(first + offset) % len(parents)]


def donor_ordinals(
    draw_count: int,
    *,
    contract_sha256: str,
    schedule_namespace: str,
    seed: int,
    generation: int,
    opportunity_ordinal: int,
) -> tuple[int, ...]:
    """Use the same Python Random semantics as accepted linked_parent_copy_v1."""

    if int(draw_count) < 0:
        raise ValueError("draw_count must be non-negative")
    rng = random.Random(
        schedule_word(
            contract_sha256,
            schedule_namespace,
            seed,
            generation,
            opportunity_ordinal,
            "inheritance",
        )
    )
    return tuple(0 if rng.random() < 0.5 else 1 for _ in range(draw_count))


def neutral_cull_rank(
    agent_id: int,
    *,
    contract_sha256: str,
    schedule_namespace: str,
    seed: int,
    generation: int,
) -> tuple[int, int]:
    return (
        schedule_word(
            contract_sha256,
            schedule_namespace,
            seed,
            generation,
            int(agent_id),
            "neutral_cull_rank",
        ),
        int(agent_id),
    )


__all__ = [
    "SCHEDULE_SCHEMA_VERSION",
    "canonical_json_sha256",
    "canonical_package_sha256",
    "donor_ordinals",
    "neutral_cull_rank",
    "schedule_word",
    "scheduled_parent_ids",
]
