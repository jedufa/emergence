"""Fail-closed seed authority for the Genome V2 RECOMB-72 campaign."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from typing import Any, Final, Mapping, Sequence

from engine.evolution.genome_upgrade_v2_recomb72_founders_v1 import (
    FOUNDER_PROFILE_IDS,
)
from engine.simulation.rng_manager import _seed_for_domain, _seed_for_stream

from .canonical import canonical_sha256
from .patch_runner_factory import (
    derive_patch_master_seed_v1,
    derive_production_turnover_seed_v1,
)


SCHEMA_VERSION: Final[str] = "genome_v2_recomb72_seed_authority_v1"
DESIGN_LABEL: Final[str] = (
    "Ecology-blocked deterministic design with independent profile-specific "
    "selection tapes."
)
ECOLOGY_SEEDS: Final[tuple[int, ...]] = (101, 202, 303, 404, 505, 606)
TASK_SEED: Final[int] = 0
EXPECTED_CELL_COUNT: Final[int] = 72
PRODUCTION_PROFILE_ID: Final[str] = "eiel-production-runtime-v1"
WORLD_IDENTITY: Final[str] = "patch"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CELL_ID_RE = re.compile(r"cell-[0-9a-f]{20}\Z")
_U32_MASK = (1 << 32) - 1
_I63_MASK = (1 << 63) - 1
_U64_MASK = (1 << 64) - 1


class GenomeV2Recomb72SeedAuthorityError(ValueError):
    """The compiled plan cannot support the frozen RECOMB-72 seed design."""


def _fail(message: str) -> GenomeV2Recomb72SeedAuthorityError:
    return GenomeV2Recomb72SeedAuthorityError(message)


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _fail(f"{name} must be a lowercase SHA-256")
    return value


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise _fail(f"{name} must be an object with string keys")
    return dict(value)


def _sequence(value: Any, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail(f"{name} must be a sequence")
    return tuple(value)


def _plan_payload(plan: Any) -> dict[str, Any]:
    payload = getattr(plan, "payload", None)
    if callable(payload):
        value = payload()
    elif isinstance(plan, Mapping):
        value = dict(plan)
    else:
        raise _fail("plan must be a compiled campaign plan or its payload")
    result = _mapping(value, name="plan")
    if result.get("schema_version") != "eiel_campaign_plan_v1":
        raise _fail("plan schema is not eiel_campaign_plan_v1")
    plan_sha256 = _sha256(result.get("plan_sha256"), name="plan.plan_sha256")
    unsigned = dict(result)
    unsigned.pop("plan_sha256")
    if canonical_sha256(unsigned) != plan_sha256:
        raise _fail("plan.plan_sha256 does not bind the supplied compiled plan")
    if result.get("resolved_profile") != PRODUCTION_PROFILE_ID:
        raise _fail(
            f"plan must resolve the production profile {PRODUCTION_PROFILE_ID!r}"
        )
    return result


def _master_seed_raw_u64(seeds: Mapping[str, int]) -> int:
    payload = {
        "schema_version": "eiel_named_seed_owners_v1",
        "owners": dict(sorted(seeds.items())),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _active_seed_domains(master_seed: int) -> dict[str, Any]:
    policy_stream = _seed_for_stream(master_seed, "policy")
    mutation_stream = _seed_for_stream(master_seed, "mutation")
    metrics_stream = _seed_for_stream(master_seed, "metrics")
    mutation_py_u64 = _seed_for_domain(mutation_stream, "py")
    return {
        "policy": {
            "stream_seed_u64": policy_stream,
            "np_seed_u64": _seed_for_domain(policy_stream, "np"),
        },
        "mutation": {
            "stream_seed_u64": mutation_stream,
            "py_seed_u64": mutation_py_u64,
            "py_seed_folded_u32": (
                mutation_py_u64 ^ (mutation_py_u64 >> 32)
            )
            & _U32_MASK,
            "np_seed_u64": _seed_for_domain(mutation_stream, "np"),
            "torch_cpu_seed_u64": _seed_for_domain(
                mutation_stream, "torch", "cpu"
            ),
        },
        "metrics": {
            "stream_seed_u64": metrics_stream,
            "np_seed_u64": _seed_for_domain(metrics_stream, "np"),
        },
        "legacy_construction": {
            "python_seed_u32": (
                _seed_for_stream(master_seed, "legacy_globals") & _U32_MASK
            ),
            "numpy_seed_u32": (
                _seed_for_stream(master_seed, "legacy_numpy") & _U32_MASK
            ),
            "torch_global_seed_i63": (
                _seed_for_stream(master_seed, "legacy_torch_global") & _I63_MASK
            ),
            "torch_cuda_seed_i63": (
                _seed_for_stream(master_seed, "legacy_torch_cuda") & _I63_MASK
            ),
        },
    }


def _cell_row(cell: Mapping[str, Any]) -> dict[str, Any]:
    cell_id = cell.get("cell_id")
    if not isinstance(cell_id, str) or _CELL_ID_RE.fullmatch(cell_id) is None:
        raise _fail("every cell_id must be a compiler-derived cell-<20 hex> identity")
    parameters = _mapping(cell.get("parameters"), name=f"{cell_id}.parameters")
    seeds = _mapping(cell.get("seeds"), name=f"{cell_id}.seeds")
    if set(seeds) != {"ecology", "task"}:
        raise _fail(f"{cell_id} must have exactly ecology and task seed owners")
    ecology = seeds["ecology"]
    task = seeds["task"]
    if type(ecology) is not int or ecology not in ECOLOGY_SEEDS:
        raise _fail(f"{cell_id} has an unsupported ecology seed")
    if type(task) is not int or task != TASK_SEED:
        raise _fail(f"{cell_id} must use task seed {TASK_SEED}")
    profile = parameters.get("genome_upgrade_v2.founder_profile")
    if profile not in FOUNDER_PROFILE_IDS:
        raise _fail(f"{cell_id} has an unsupported Genome V2 founder profile")

    named_seeds = {"ecology": ecology, "task": task}
    raw_master = _master_seed_raw_u64(named_seeds)
    master = derive_patch_master_seed_v1(named_seeds)
    if master != raw_master & _I63_MASK:
        raise _fail(f"{cell_id} master-seed signed63 projection drifted")
    turnover = derive_production_turnover_seed_v1(
        cell_id=cell_id,
        seeds=named_seeds,
    )
    if turnover != derive_production_turnover_seed_v1(
        cell_id=cell_id,
        seeds=named_seeds,
    ):
        raise _fail(f"{cell_id} turnover seed derivation is not deterministic")
    if not 0 <= turnover <= _U64_MASK:
        raise _fail(f"{cell_id} turnover seed is not an exact uint64")
    active = _active_seed_domains(master)
    if active != _active_seed_domains(master):
        raise _fail(f"{cell_id} active seed derivation is not deterministic")
    return {
        "founder_profile": profile,
        "ecology_seed": ecology,
        "task_seed": task,
        "cell_id": cell_id,
        "master_seed_raw_u64": raw_master,
        "master_seed_i63": master,
        "turnover_selection_seed_u64": turnover,
        "active_seed_domains": active,
    }


def _flatten_seed_widths(
    rows_by_ecology: Mapping[int, Mapping[str, Any]],
) -> tuple[list[int], list[int], list[int]]:
    u64: list[int] = []
    u32: list[int] = []
    i63: list[int] = []
    for ecology in ECOLOGY_SEEDS:
        row = rows_by_ecology[ecology]
        domains = row["active_seed_domains"]
        u64.extend(
            (
                domains["policy"]["stream_seed_u64"],
                domains["policy"]["np_seed_u64"],
                domains["mutation"]["stream_seed_u64"],
                domains["mutation"]["py_seed_u64"],
                domains["mutation"]["np_seed_u64"],
                domains["mutation"]["torch_cpu_seed_u64"],
                domains["metrics"]["stream_seed_u64"],
                domains["metrics"]["np_seed_u64"],
            )
        )
        u32.extend(
            (
                domains["mutation"]["py_seed_folded_u32"],
                domains["legacy_construction"]["python_seed_u32"],
                domains["legacy_construction"]["numpy_seed_u32"],
            )
        )
        i63.extend(
            (
                row["master_seed_i63"],
                domains["legacy_construction"]["torch_global_seed_i63"],
                domains["legacy_construction"]["torch_cuda_seed_i63"],
            )
        )
    return u64, u32, i63


def audit_genome_v2_recomb72_seed_authority_v1(
    plan: Any,
    *,
    package_id_sha256: str,
    execution_route_sha256: str,
    world_identity: str,
    configuration_sha256: str,
) -> dict[str, Any]:
    """Validate and bind the exact ecology-blocked B72 seed topology."""

    payload = _plan_payload(plan)
    package_id = _sha256(package_id_sha256, name="package_id_sha256")
    route_id = _sha256(
        execution_route_sha256, name="execution_route_sha256"
    )
    config_id = _sha256(configuration_sha256, name="configuration_sha256")
    if world_identity != WORLD_IDENTITY:
        raise _fail(f"world_identity must be {WORLD_IDENTITY!r}")

    raw_cells = _sequence(payload.get("cells"), name="plan.cells")
    if len(raw_cells) != EXPECTED_CELL_COUNT:
        raise _fail(f"plan must contain exactly {EXPECTED_CELL_COUNT} cells")
    cells = tuple(
        _mapping(cell, name=f"plan.cells[{index}]")
        for index, cell in enumerate(raw_cells)
    )
    rows = tuple(_cell_row(cell) for cell in cells)
    cell_ids = tuple(row["cell_id"] for row in rows)
    if len(set(cell_ids)) != EXPECTED_CELL_COUNT:
        raise _fail("plan cell IDs must be unique")

    expected_pairs = {
        (profile, ecology)
        for ecology in ECOLOGY_SEEDS
        for profile in FOUNDER_PROFILE_IDS
    }
    actual_pairs = {
        (row["founder_profile"], row["ecology_seed"]) for row in rows
    }
    if len(actual_pairs) != EXPECTED_CELL_COUNT or actual_pairs != expected_pairs:
        raise _fail("plan must contain every founder-profile/ecology pair exactly once")

    raw_batches = _sequence(
        payload.get("physical_batches"), name="plan.physical_batches"
    )
    if len(raw_batches) != 1:
        raise _fail("plan must contain exactly one physical B72 batch")
    batch = _mapping(raw_batches[0], name="plan.physical_batches[0]")
    batch_ids = _sequence(
        batch.get("cell_ids"), name="plan.physical_batches[0].cell_ids"
    )
    if (
        batch.get("ordinal") != 0
        or batch.get("batch_id") != "batch-0000"
        or len(batch_ids) != EXPECTED_CELL_COUNT
        or len(set(batch_ids)) != EXPECTED_CELL_COUNT
        or set(batch_ids) != set(cell_ids)
    ):
        raise _fail("plan physical batch must be one exact identity-preserving B72")

    row_by_id = {row["cell_id"]: row for row in rows}
    profile_rank = {
        profile: ordinal for ordinal, profile in enumerate(FOUNDER_PROFILE_IDS)
    }
    canonical_rows = tuple(
        sorted(
            rows,
            key=lambda row: (
                row["ecology_seed"],
                profile_rank[row["founder_profile"]],
                row["cell_id"],
            ),
        )
    )
    if canonical_rows != tuple(
        sorted(
            reversed(rows),
            key=lambda row: (
                row["ecology_seed"],
                profile_rank[row["founder_profile"]],
                row["cell_id"],
            ),
        )
    ):
        raise _fail("canonical rows depend on cell declaration order")

    master_counts = Counter(row["master_seed_i63"] for row in rows)
    if len(master_counts) != len(ECOLOGY_SEEDS) or set(master_counts.values()) != {
        len(FOUNDER_PROFILE_IDS)
    }:
        raise _fail("master seeds must be six unique values repeated 12 times")
    turnover_seeds = tuple(
        row["turnover_selection_seed_u64"] for row in rows
    )
    if len(set(turnover_seeds)) != EXPECTED_CELL_COUNT:
        raise _fail(
            "profile-specific turnover selection seeds must be 72 unique values"
        )

    representative_by_ecology: dict[int, Mapping[str, Any]] = {}
    for row in canonical_rows:
        representative = representative_by_ecology.setdefault(
            row["ecology_seed"], row
        )
        if (
            row["master_seed_raw_u64"] != representative["master_seed_raw_u64"]
            or row["master_seed_i63"] != representative["master_seed_i63"]
            or row["active_seed_domains"] != representative["active_seed_domains"]
        ):
            raise _fail("engine stream seeds are not shared within an ecology block")
    raw_masters = [
        representative_by_ecology[ecology]["master_seed_raw_u64"]
        for ecology in ECOLOGY_SEEDS
    ]
    masked_masters = [
        representative_by_ecology[ecology]["master_seed_i63"]
        for ecology in ECOLOGY_SEEDS
    ]
    if len(set(raw_masters)) != 6 or len(set(masked_masters)) != 6:
        raise _fail("master seed truncation aliases ecology owners")

    active_u64, active_u32, active_i63 = _flatten_seed_widths(
        representative_by_ecology
    )
    if len(set(active_u64)) != len(active_u64):
        raise _fail("active uint64 stream/domain seeds collide")
    if len(set(active_u32)) != len(active_u32):
        raise _fail("active uint32 projections collide")
    if len(set(active_i63)) != len(active_i63):
        raise _fail("active signed63 projections collide")
    if set(active_u64) & set(turnover_seeds):
        raise _fail("turnover seeds collide with active uint64 stream/domain seeds")

    turnover_by_id = {
        row["cell_id"]: row["turnover_selection_seed_u64"] for row in rows
    }
    batch_seed_vector = tuple(turnover_by_id[cell_id] for cell_id in batch_ids)
    rederived_after_position_change = {}
    for cell_id in reversed(batch_ids):
        row = row_by_id[cell_id]
        rederived_after_position_change[cell_id] = (
            derive_production_turnover_seed_v1(
                cell_id=cell_id,
                seeds={
                    "ecology": row["ecology_seed"],
                    "task": row["task_seed"],
                },
            )
        )
    if rederived_after_position_change != turnover_by_id:
        raise _fail("turnover seeds depend on physical batch position")

    authority = {
        "package_id_sha256": package_id,
        "execution_route_sha256": route_id,
        "world_identity": world_identity,
        "configuration_sha256": config_id,
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "design_label": DESIGN_LABEL,
        "fully_paired_claims_allowed": False,
        "claims_boundary": (
            "Named-owner engine streams are shared within ecology blocks, but "
            "turnover selection tapes are profile-specific. Fully paired "
            "trajectory claims are prohibited."
        ),
        "authority": authority,
        "plan_identity": {
            "campaign_id": payload.get("campaign_id"),
            "plan_sha256": payload["plan_sha256"],
            "resolved_profile": payload["resolved_profile"],
            "physical_batch_id": batch["batch_id"],
        },
        "seed_input_authority": {
            "engine_streams": ["ecology", "task"],
            "turnover_selection": ["cell_id", "ecology", "task"],
            "audit_only_not_seed_inputs": sorted(authority),
        },
        "matrix_contract": {
            "cell_count": EXPECTED_CELL_COUNT,
            "founder_profile_ids": list(FOUNDER_PROFILE_IDS),
            "ecology_seeds": list(ECOLOGY_SEEDS),
            "task_seed": TASK_SEED,
            "physical_batch_shape": "b72",
            "physical_order": (
                "compiler_canonical_identity_order_with_explicit_position_map"
            ),
        },
        "rows": list(canonical_rows),
        "rows_sha256": canonical_sha256(list(canonical_rows)),
        "physical_batch_mapping": [
            {
                "position": position,
                "cell_id": cell_id,
                "ecology_seed": row_by_id[cell_id]["ecology_seed"],
                "turnover_selection_seed_u64": batch_seed_vector[position],
            }
            for position, cell_id in enumerate(batch_ids)
        ],
        "physical_batch_identity_mapping_sha256": canonical_sha256(
            sorted(turnover_by_id.items())
        ),
        "checks": {
            "exact_72_cell_matrix": True,
            "unique_compiler_cell_ids": True,
            "single_physical_b72": True,
            "ecology_blocks_complete_independent_of_physical_order": True,
            "six_master_seeds_repeated_12_times": True,
            "seventy_two_profile_specific_turnover_seeds": True,
            "master_signed63_projection_alias_free": True,
            "active_uint64_domain_collision_free": True,
            "active_uint32_projection_alias_free": True,
            "active_signed63_projection_alias_free": True,
            "deterministic_seed_rerun": True,
            "cell_declaration_order_invariant": True,
            "physical_batch_position_mapping_invariant": True,
        },
    }
    result["audit_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "DESIGN_LABEL",
    "ECOLOGY_SEEDS",
    "EXPECTED_CELL_COUNT",
    "GenomeV2Recomb72SeedAuthorityError",
    "SCHEMA_VERSION",
    "TASK_SEED",
    "WORLD_IDENTITY",
    "audit_genome_v2_recomb72_seed_authority_v1",
]
