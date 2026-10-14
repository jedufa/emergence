#!/usr/bin/env python3
"""Freeze the device-neutral GMNW VENV-1 physical B72 matrix contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Final, Mapping


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ENGINE: Final[Path] = ROOT / "engine"
for _path in (ROOT, ENGINE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from engine.evolution import gmnw_v1 as gmnw  # noqa: E402


SCHEMA_VERSION: Final[str] = "eiel_gmnw_venv1_matrix_contract_v1"
RESULT_SCHEMA_VERSION: Final[str] = "eiel_gmnw_venv1_matrix_result_v1"
RUN_LABEL: Final[str] = "gmnw-venv1-b72g100t80"
PROFILE_IDS: Final[tuple[str, ...]] = ("P1", "P2", "P3")
ARM_IDS: Final[tuple[str, ...]] = ("G-D", "R-D", "F-D", "F-I")
SOURCE_SEEDS: Final[tuple[int, ...]] = (101, 202, 303, 404, 505, 606)
GENERATIONS: Final[int] = 100
TICKS_PER_GENERATION: Final[int] = 80
CHECKPOINT_GENERATIONS: Final[tuple[int, ...]] = (25, 50, 75, 100)
PHYSICAL_CELLS: Final[int] = 72
FORKS_PER_SOURCE: Final[int] = 12
INITIAL_POPULATION: Final[int] = 64
SOURCE_SET_SCHEMA_VERSION: Final[str] = (
    "eiel_gmnw_ub100_source_projection_set_v1"
)
CLAIM_BOUNDARY: Final[str] = (
    "bounded homogeneous-policy viability-envelope comparison only; no selection, "
    "de novo memory discovery, intelligence, emergence, consciousness, broad "
    "cognition, or broad adaptive-memory claim"
)

FROZEN_CONTRACT_SHA256: Final[str] = (
    "455dc811ba5c9f03598e313c4f183169f40110a53c008588eb33d8a26368236a"
)


class Venv1MatrixContractError(RuntimeError):
    """The VENV-1 matrix differs from its frozen physical contract."""


@dataclass(frozen=True, slots=True)
class Venv1ArmV1:
    arm_id: str
    arm_code: str
    policy_arm: str
    context: str
    cue_visible_at_decision: bool
    reset_hidden_each_tick: bool


@dataclass(frozen=True, slots=True)
class Venv1SourceV1:
    source_group: int
    source_seed: int
    source_id: str
    source_branch_id: str


@dataclass(frozen=True, slots=True)
class Venv1CellV1:
    ordinal: int
    cell_id: str
    profile_id: str
    profile_code: int
    profile_config_hash: str
    arm_id: str
    policy_arm: str
    context: str
    cue_visible_at_decision: bool
    reset_hidden_each_tick: bool
    source_seed: int
    task_seed: int
    source_group: int
    source_id: str
    source_branch_id: str
    fork_branch_id: str
    fork_reseeded: bool
    seed_derivation: None


def canonical_sha256_v1(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def arms_v1() -> tuple[Venv1ArmV1, ...]:
    """Return the four frozen homogeneous arms in physical matrix order."""

    return (
        Venv1ArmV1("G-D", "gd", "persistent_gru", "delayed", False, False),
        Venv1ArmV1("R-D", "rd", "reset_gru", "delayed", False, True),
        Venv1ArmV1("F-D", "fd", "matched_mlp", "delayed", False, False),
        Venv1ArmV1("F-I", "fi", "matched_mlp", "immediate", True, False),
    )


def sources_v1() -> tuple[Venv1SourceV1, ...]:
    """Return the ordered six-source subset of the accepted UB100 source set."""

    return tuple(
        Venv1SourceV1(
            source_group=source_group,
            source_seed=seed,
            source_id=f"gmnw-ub100-source-s{seed:03d}",
            source_branch_id=f"gmnw-ub100-source-s{seed:03d}",
        )
        for source_group, seed in enumerate(SOURCE_SEEDS)
    )


def _matrix_gmnw_config_v1() -> gmnw.GmnwConfig:
    return gmnw.GmnwConfig(
        cue_alphabet=4,
        cue_ticks=1,
        delay_ticks=12,
        consequence_ticks=1,
        distractors_enabled=True,
        cue_visible_at_decision=False,
        task_seed=SOURCE_SEEDS[0],
        grace_generation_count=4,
        severe_generation_count=4,
        grace_damage_fraction=0.15,
        severe_damage_fraction=0.45,
        lethal_damage_fraction=1.0,
        grace_reproduction_scale=0.5,
    )


def profile_contracts_v1() -> tuple[dict[str, Any], ...]:
    """Resolve P1/P2/P3 only through the canonical engine registry."""

    config = _matrix_gmnw_config_v1()
    rows: list[dict[str, Any]] = []
    for profile_id in PROFILE_IDS:
        profile = gmnw.resolve_gmnw_venv1_profile(profile_id)
        if profile.profile_id != profile_id:
            raise Venv1MatrixContractError("canonical profile ID resolution drifted")
        resolved = gmnw.gmnw_venv1_profile_config(profile_id, config=config)
        rows.append(
            {
                "profile_id": profile.profile_id,
                "profile_code": profile.profile_code,
                "profile_config": resolved,
            }
        )
    if [row["profile_code"] for row in rows] != [1, 2, 3]:
        raise Venv1MatrixContractError("canonical P1/P2/P3 profile codes drifted")
    return tuple(rows)


def cells_v1() -> tuple[Venv1CellV1, ...]:
    """Return cells in profile, arm, then accepted-source order."""

    profiles = profile_contracts_v1()
    cells: list[Venv1CellV1] = []
    for profile in profiles:
        profile_id = str(profile["profile_id"])
        profile_code = int(profile["profile_code"])
        profile_config = profile["profile_config"]
        if not isinstance(profile_config, Mapping):
            raise Venv1MatrixContractError("canonical profile config is not an object")
        profile_config_hash = str(profile_config["config_hash"])
        for arm in arms_v1():
            for source in sources_v1():
                cell_id = (
                    f"venv1_{profile_id.lower()}_{arm.arm_code}_"
                    f"s{source.source_seed:03d}"
                )
                cells.append(
                    Venv1CellV1(
                        ordinal=len(cells),
                        cell_id=cell_id,
                        profile_id=profile_id,
                        profile_code=profile_code,
                        profile_config_hash=profile_config_hash,
                        arm_id=arm.arm_id,
                        policy_arm=arm.policy_arm,
                        context=arm.context,
                        cue_visible_at_decision=arm.cue_visible_at_decision,
                        reset_hidden_each_tick=arm.reset_hidden_each_tick,
                        source_seed=source.source_seed,
                        task_seed=source.source_seed,
                        source_group=source.source_group,
                        source_id=source.source_id,
                        source_branch_id=source.source_branch_id,
                        fork_branch_id=f"gmnw-{cell_id.replace('_', '-')}",
                        fork_reseeded=False,
                        seed_derivation=None,
                    )
                )
    if len(cells) != PHYSICAL_CELLS:
        raise Venv1MatrixContractError("matrix must contain exactly 72 cells")
    if len({cell.cell_id for cell in cells}) != PHYSICAL_CELLS:
        raise Venv1MatrixContractError("matrix cell IDs must be unique")
    return tuple(cells)


def _checkpoint_result_schema_v1() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "generation",
            "alive_count",
            "decision_count",
            "correct_count",
            "incorrect_count",
            "task_death_count",
            "authoritative_state_sha256",
        ],
        "properties": {
            "generation": {"enum": list(CHECKPOINT_GENERATIONS)},
            "alive_count": {
                "type": "integer",
                "minimum": 0,
                "maximum": INITIAL_POPULATION,
            },
            "decision_count": {"type": "integer", "minimum": 0},
            "correct_count": {"type": "integer", "minimum": 0},
            "incorrect_count": {"type": "integer", "minimum": 0},
            "task_death_count": {"type": "integer", "minimum": 0},
            "authoritative_state_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
    }


def result_schema_v1() -> dict[str, Any]:
    identity_properties: dict[str, Any] = {
        "ordinal": {"type": "integer", "minimum": 0, "maximum": 71},
        "cell_id": {"type": "string"},
        "profile_id": {"enum": list(PROFILE_IDS)},
        "profile_code": {"enum": [1, 2, 3]},
        "arm_id": {"enum": list(ARM_IDS)},
        "policy_arm": {"enum": ["persistent_gru", "reset_gru", "matched_mlp"]},
        "context": {"enum": ["delayed", "immediate"]},
        "cue_visible_at_decision": {"type": "boolean"},
        "source_seed": {"enum": list(SOURCE_SEEDS)},
        "source_group": {"type": "integer", "minimum": 0, "maximum": 5},
        "fork_reseeded": {"const": False},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": RESULT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "contract_sha256",
            "run_label",
            "claim_boundary",
            "status",
            "completed_generations",
            "checkpoint_generations",
            "cells",
            "result_sha256",
        ],
        "properties": {
            "schema_version": {"const": RESULT_SCHEMA_VERSION},
            "contract_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "run_label": {"const": RUN_LABEL},
            "claim_boundary": {"const": CLAIM_BOUNDARY},
            "status": {"const": "completed"},
            "completed_generations": {"const": GENERATIONS},
            "checkpoint_generations": {"const": list(CHECKPOINT_GENERATIONS)},
            "cells": {
                "type": "array",
                "minItems": PHYSICAL_CELLS,
                "maxItems": PHYSICAL_CELLS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        *identity_properties,
                        "status",
                        "completed_generations",
                        "checkpoints",
                    ],
                    "properties": {
                        **identity_properties,
                        "status": {"const": "completed"},
                        "completed_generations": {"const": GENERATIONS},
                        "checkpoints": {
                            "type": "array",
                            "prefixItems": [
                                {
                                    **_checkpoint_result_schema_v1(),
                                    "properties": {
                                        **_checkpoint_result_schema_v1()["properties"],
                                        "generation": {"const": generation},
                                    },
                                }
                                for generation in CHECKPOINT_GENERATIONS
                            ],
                            "items": False,
                            "minItems": len(CHECKPOINT_GENERATIONS),
                            "maxItems": len(CHECKPOINT_GENERATIONS),
                        },
                    },
                },
            },
            "result_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
    }


def _contract_material_v1() -> dict[str, Any]:
    config = _matrix_gmnw_config_v1()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_label": RUN_LABEL,
        "device_binding": None,
        "execution_mechanics": "not_defined_by_this_contract",
        "claim_boundary": CLAIM_BOUNDARY,
        "matrix_order": "profile_then_arm_then_source_seed",
        "physical_batch_cells": PHYSICAL_CELLS,
        "logical_worlds": PHYSICAL_CELLS,
        "logical_chunking": False,
        "generations": GENERATIONS,
        "ticks_per_generation": TICKS_PER_GENERATION,
        "checkpoint_generations": list(CHECKPOINT_GENERATIONS),
        "canonical_profile_registry": {
            "schema_version": gmnw.GMNW_VENV1_PROFILE_CONTRACT_VERSION,
            "contract_hash": gmnw.gmnw_venv1_profile_contract_hash(),
            "selected_profile_ids": list(PROFILE_IDS),
            "selected_profile_codes": [1, 2, 3],
            "profiles": list(profile_contracts_v1()),
        },
        "arms": [asdict(arm) for arm in arms_v1()],
        "accepted_sources": {
            "source_set_schema_version": SOURCE_SET_SCHEMA_VERSION,
            "subset_rule": "ordered_prefix_6_of_accepted_ub100_source_seeds",
            "source_seeds": list(SOURCE_SEEDS),
            "source_count": len(SOURCE_SEEDS),
            "forks_per_source": FORKS_PER_SOURCE,
            "pre_configuration_fork_identity_required": True,
            "fork_rng_inheritance": "exact_source_rng_state",
            "fork_reseeding": False,
            "seed_derivation": None,
            "sources": [asdict(source) for source in sources_v1()],
        },
        "gmnw_configuration": {
            "cue_alphabet": config.cue_alphabet,
            "cue_ticks": config.cue_ticks,
            "delay_ticks": config.delay_ticks,
            "consequence_ticks": config.consequence_ticks,
            "distractors_enabled": config.distractors_enabled,
            "initial_population": INITIAL_POPULATION,
            "terminal_low_population_no_reinit": True,
        },
        "cells": [asdict(cell) for cell in cells_v1()],
        "result_schema": result_schema_v1(),
        "hash": {
            "algorithm": "sha256",
            "canonical_json": "utf8_sorted_keys_compact_ascii_finite",
        },
    }


def contract_v1() -> dict[str, Any]:
    material = _contract_material_v1()
    contract_hash = canonical_sha256_v1(material)
    if FROZEN_CONTRACT_SHA256 and contract_hash != FROZEN_CONTRACT_SHA256:
        raise Venv1MatrixContractError(
            "generated contract does not match FROZEN_CONTRACT_SHA256"
        )
    return {**material, "contract_sha256": contract_hash}


def validate_contract_v1(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if value is not None and not isinstance(value, Mapping):
        raise Venv1MatrixContractError("contract must be an object")
    candidate = dict(contract_v1() if value is None else value)
    if candidate.get("schema_version") != SCHEMA_VERSION:
        raise Venv1MatrixContractError("schema_version mismatch")
    actual_hash = candidate.pop("contract_sha256", None)
    calculated_hash = canonical_sha256_v1(candidate)
    if actual_hash != calculated_hash:
        raise Venv1MatrixContractError("contract_sha256 mismatch")
    if FROZEN_CONTRACT_SHA256 and actual_hash != FROZEN_CONTRACT_SHA256:
        raise Venv1MatrixContractError("contract does not use the frozen hash")
    expected = contract_v1()
    if {**candidate, "contract_sha256": actual_hash} != expected:
        raise Venv1MatrixContractError("contract differs from the frozen VENV-1 matrix")
    return expected
