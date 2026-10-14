"""Contract-driven CPU/CUDA tensor engine for neutral inheritance path audits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

from engine.evolution.inheritance_cross_path_schedule import (
    canonical_package_sha256,
    donor_ordinals,
    neutral_cull_rank,
    scheduled_parent_ids,
    schedule_word,
)


SCHEMA_VERSION = "eiel_inheritance_cross_path_tensor_v2"
CHECKPOINT_SCHEMA_VERSION = "eiel_inheritance_cross_path_checkpoint_v2"
MUTATION_OFF_OPERATOR = "off"
MUTATION_DETERMINISTIC_DELTA_OPERATOR = "deterministic_delta"


class CrossPathTensorError(ValueError):
    """A cross-path inheritance contract or state is invalid."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _word(*parts: Any) -> int:
    return int.from_bytes(
        hashlib.sha256(
            json.dumps(
                {"schema": SCHEMA_VERSION, "parts": list(parts)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).digest()[:8],
        "big",
    )


@dataclass(frozen=True)
class ProfilePolicyV1:
    profile_id: str
    operator: str
    draw_count: int

    def __post_init__(self) -> None:
        if not self.profile_id or self.operator not in {
            "midpoint",
            "linked_block_copy",
            "linked_parent_copy",
        }:
            raise CrossPathTensorError("inheritance profile is invalid")
        if self.draw_count < 0:
            raise CrossPathTensorError("inheritance draw count is invalid")
        if self.operator == "midpoint" and self.draw_count != 0:
            raise CrossPathTensorError("midpoint must add zero linkage draws")


@dataclass(frozen=True)
class ReproductionPathPolicyV1:
    path_id: str
    replacement_mode: str
    successful_births: int
    birth_opportunities: int
    schedule_namespace: str = ""
    attempt_selection: str = "prefix"
    capacity_failure_reason: str = "capacity_full"

    def __post_init__(self) -> None:
        if not self.path_id or self.replacement_mode not in {
            "batch_survivor_refill",
            "sequential_live_birth",
        }:
            raise CrossPathTensorError("reproduction path policy is invalid")
        if (
            self.attempt_selection != "prefix"
            or not str(self.capacity_failure_reason)
        ):
            raise CrossPathTensorError("reproduction path schedule is invalid")
        if (
            self.successful_births < 1
            or self.birth_opportunities < self.successful_births
        ):
            raise CrossPathTensorError("reproduction opportunity counts are invalid")


@dataclass(frozen=True)
class FounderGroupPolicyV1:
    group_id: str
    count: int

    def __post_init__(self) -> None:
        if not str(self.group_id):
            raise CrossPathTensorError("founder group ID must be non-empty")
        if type(self.count) is not int or self.count < 1:
            raise CrossPathTensorError("founder group count must be positive")


@dataclass(frozen=True)
class TensorMutationPolicyV1:
    policy_id: str
    operator: str = MUTATION_OFF_OPERATOR
    locus_deltas: tuple[tuple[str, float], ...] = ()
    event_modulus: int = 1
    event_residue: int = 0
    draw_budget: int = 0
    schedule_channel: str = "mutation"

    def __post_init__(self) -> None:
        if not str(self.policy_id) or self.operator not in {
            MUTATION_OFF_OPERATOR,
            MUTATION_DETERMINISTIC_DELTA_OPERATOR,
        }:
            raise CrossPathTensorError("tensor mutation policy is invalid")
        if (
            type(self.event_modulus) is not int
            or self.event_modulus < 1
            or type(self.event_residue) is not int
            or not 0 <= self.event_residue < self.event_modulus
            or type(self.draw_budget) is not int
            or self.draw_budget < 0
            or not str(self.schedule_channel)
        ):
            raise CrossPathTensorError("tensor mutation schedule is invalid")
        names = tuple(str(name) for name, _delta in self.locus_deltas)
        if len(set(names)) != len(names):
            raise CrossPathTensorError("tensor mutation loci must be unique")
        try:
            deltas = tuple(float(delta) for _name, delta in self.locus_deltas)
        except (TypeError, ValueError) as exc:
            raise CrossPathTensorError("tensor mutation deltas must be numeric") from exc
        if any(not torch.isfinite(torch.tensor(delta)).item() for delta in deltas):
            raise CrossPathTensorError("tensor mutation deltas must be finite")
        if self.operator == MUTATION_OFF_OPERATOR:
            if names or self.draw_budget:
                raise CrossPathTensorError(
                    "mutation-off policy cannot declare deltas or draws"
                )
        elif not names or not any(delta != 0.0 for delta in deltas):
            raise CrossPathTensorError(
                "deterministic mutation requires a nonzero locus delta"
            )
        elif self.draw_budget != 0:
            raise CrossPathTensorError(
                "deterministic-delta mutation consumes zero RNG draws"
            )

    @property
    def enabled(self) -> bool:
        return self.operator != MUTATION_OFF_OPERATOR

    def payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "operator": self.operator,
            "enabled": self.enabled,
            "locus_deltas": {
                str(name): float(delta) for name, delta in self.locus_deltas
            },
            "event_modulus": self.event_modulus,
            "event_residue": self.event_residue,
            "draw_budget": self.draw_budget,
            "schedule_channel": self.schedule_channel,
        }


@dataclass(frozen=True)
class TensorInheritanceContractV1:
    locus_names: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    linkage_block_ids: tuple[str, ...]
    linked_locus_block_indices: tuple[int, ...]
    profiles: tuple[ProfilePolicyV1, ...]
    paths: tuple[ReproductionPathPolicyV1, ...]
    default_profile_id: str
    population_size: int
    capacity: int
    schedule_contract_sha256: str | None = None
    founder_groups: tuple[FounderGroupPolicyV1, ...] = ()
    founder_group_a_count: int | None = None
    mutation_policy: TensorMutationPolicyV1 = TensorMutationPolicyV1(
        "mutation_off"
    )

    def __post_init__(self) -> None:
        locus_count = len(self.locus_names)
        if (
            locus_count < 1
            or len(set(self.locus_names)) != locus_count
            or len(self.lower_bounds) != locus_count
            or len(self.upper_bounds) != locus_count
            or len(self.linked_locus_block_indices) != locus_count
        ):
            raise CrossPathTensorError("locus registry dimensions are invalid")
        if any(
            not float(lower) < float(upper)
            for lower, upper in zip(
                self.lower_bounds, self.upper_bounds, strict=True
            )
        ):
            raise CrossPathTensorError("locus bounds are invalid")
        block_count = len(self.linkage_block_ids)
        if block_count < 1 or len(set(self.linkage_block_ids)) != block_count:
            raise CrossPathTensorError("linkage block registry is invalid")
        if any(
            index < -1 or index >= block_count
            for index in self.linked_locus_block_indices
        ):
            raise CrossPathTensorError("locus linkage assignment is invalid")
        profile_ids = [row.profile_id for row in self.profiles]
        path_ids = [row.path_id for row in self.paths]
        if (
            not self.profiles
            or len(set(profile_ids)) != len(profile_ids)
            or self.default_profile_id not in profile_ids
            or not self.paths
            or len(set(path_ids)) != len(path_ids)
        ):
            raise CrossPathTensorError("profile or path registry is invalid")
        for profile in self.profiles:
            if (
                profile.operator == "linked_block_copy"
                and profile.draw_count != block_count
            ):
                raise CrossPathTensorError(
                    "linked profile draw count must equal linkage block count"
                )
        if (
            self.population_size < 4
            or self.capacity < self.population_size
            or any(
                row.successful_births > self.population_size - 2
                for row in self.paths
            )
        ):
            raise CrossPathTensorError("population contract is invalid")
        if self.founder_group_a_count is not None and not (
            0 < self.founder_group_a_count < self.population_size
        ):
            raise CrossPathTensorError(
                "legacy founder group split must be inside the population"
            )
        groups = self.resolved_founder_groups()
        if sum(group.count for group in groups) != self.population_size:
            raise CrossPathTensorError(
                "founder group counts must equal the population size"
            )
        mutation_loci = {
            str(name) for name, _delta in self.mutation_policy.locus_deltas
        }
        unknown_mutation_loci = sorted(mutation_loci - set(self.locus_names))
        if unknown_mutation_loci:
            raise CrossPathTensorError(
                f"mutation policy references unknown loci: "
                f"{unknown_mutation_loci!r}"
            )
        if self.schedule_contract_sha256 is not None and not (
            len(self.schedule_contract_sha256) == 64
            and all(
                character in "0123456789abcdef"
                for character in self.schedule_contract_sha256
            )
        ):
            raise CrossPathTensorError("schedule contract hash is invalid")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "loci": [
                {
                    "name": name,
                    "lower": lower,
                    "upper": upper,
                    "linked_block_index": block,
                }
                for name, lower, upper, block in zip(
                    self.locus_names,
                    self.lower_bounds,
                    self.upper_bounds,
                    self.linked_locus_block_indices,
                    strict=True,
                )
            ],
            "linkage_block_ids": list(self.linkage_block_ids),
            "profiles": [
                {
                    "profile_id": row.profile_id,
                    "operator": row.operator,
                    "draw_count": row.draw_count,
                }
                for row in self.profiles
            ],
            "paths": [
                {
                    "path_id": row.path_id,
                    "replacement_mode": row.replacement_mode,
                    "successful_births": row.successful_births,
                    "birth_opportunities": row.birth_opportunities,
                    "schedule_namespace": (
                        row.schedule_namespace or row.path_id
                    ),
                    "attempt_selection": row.attempt_selection,
                    "capacity_failure_reason": row.capacity_failure_reason,
                }
                for row in self.paths
            ],
            "default_profile_id": self.default_profile_id,
            "population_size": self.population_size,
            **(
                {"founder_group_a_count": self.founder_group_a_count}
                if self.founder_group_a_count is not None
                else {}
            ),
            "founder_groups": [
                {"group_id": group.group_id, "count": group.count}
                for group in self.resolved_founder_groups()
            ],
            "capacity": self.capacity,
            "schedule_contract_sha256": self.schedule_contract_sha256,
            "mutation_policy": self.mutation_policy.payload(),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.payload())

    @property
    def control_schedule_sha256(self) -> str:
        return self.schedule_contract_sha256 or self.sha256

    def resolved_founder_groups(self) -> tuple[FounderGroupPolicyV1, ...]:
        groups = self.founder_groups
        if not groups:
            if self.founder_group_a_count is None:
                raise CrossPathTensorError(
                    "founder groups must be declared explicitly"
                )
            groups = (
                FounderGroupPolicyV1("a", self.founder_group_a_count),
                FounderGroupPolicyV1(
                    "b", self.population_size - self.founder_group_a_count
                ),
            )
        group_ids = tuple(group.group_id for group in groups)
        if len(set(group_ids)) != len(group_ids):
            raise CrossPathTensorError("founder group IDs must be unique")
        return groups


@dataclass(slots=True)
class TensorCrossPathStateV1:
    contract: TensorInheritanceContractV1
    loci: torch.Tensor
    alive: torch.Tensor
    agent_ids: torch.Tensor
    ancestry: torch.Tensor
    next_id: torch.Tensor
    generation: torch.Tensor
    seeds: tuple[int, ...]
    profile_ids: tuple[str, ...]
    path_ids: tuple[str, ...]
    pattern_ids: tuple[str, ...]

    @property
    def batch_cells(self) -> int:
        return int(self.loci.shape[0])

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "contract": self.contract.payload(),
            "contract_sha256": self.contract.sha256,
            "seeds": list(self.seeds),
            "profile_ids": list(self.profile_ids),
            "path_ids": list(self.path_ids),
            "pattern_ids": list(self.pattern_ids),
            "tensors": {
                name: getattr(self, name).detach().cpu().clone()
                for name in (
                    "loci",
                    "alive",
                    "agent_ids",
                    "ancestry",
                    "next_id",
                    "generation",
                )
            },
        }


def _profile(
    contract: TensorInheritanceContractV1, profile_id: str
) -> ProfilePolicyV1:
    try:
        return next(row for row in contract.profiles if row.profile_id == profile_id)
    except StopIteration as exc:
        raise CrossPathTensorError(f"unsupported profile: {profile_id!r}") from exc


def _path(
    contract: TensorInheritanceContractV1, path_id: str
) -> ReproductionPathPolicyV1:
    try:
        return next(row for row in contract.paths if row.path_id == path_id)
    except StopIteration as exc:
        raise CrossPathTensorError(f"unsupported path: {path_id!r}") from exc


def initialize_tensor_cross_path_state_v1(
    contract: TensorInheritanceContractV1,
    *,
    seeds: Sequence[int],
    profile_ids: Sequence[str],
    path_ids: Sequence[str],
    pattern_ids: Sequence[str],
    founder_packages: Mapping[str, Mapping[str, Sequence[float]]],
    device: str | torch.device,
) -> TensorCrossPathStateV1:
    raise CrossPathTensorError(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "standalone tensor inheritance initialization is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    batch = len(seeds)
    if (
        batch < 1
        or len(profile_ids) != batch
        or len(path_ids) != batch
        or len(pattern_ids) != batch
    ):
        raise CrossPathTensorError("cell identity arrays are not aligned")
    for profile_id in profile_ids:
        _profile(contract, str(profile_id))
    for path_id in path_ids:
        _path(contract, str(path_id))
    founder_groups = contract.resolved_founder_groups()
    group_ids = tuple(group.group_id for group in founder_groups)
    packages: list[dict[str, Sequence[float]]] = []
    for pattern_id in pattern_ids:
        raw = founder_packages.get(str(pattern_id))
        if not isinstance(raw, Mapping) or set(raw) != set(group_ids):
            raise CrossPathTensorError(f"founder pattern is invalid: {pattern_id!r}")
        normalized = {group_id: raw[group_id] for group_id in group_ids}
        for values in normalized.values():
            if len(values) != len(contract.locus_names):
                raise CrossPathTensorError("founder package width mismatch")
            for value, lower, upper in zip(
                values, contract.lower_bounds, contract.upper_bounds, strict=True
            ):
                if not float(lower) <= float(value) <= float(upper):
                    raise CrossPathTensorError("founder package is outside bounds")
        packages.append(normalized)

    resolved = torch.device(device)
    shape = (batch, contract.capacity, len(contract.locus_names))
    loci = torch.zeros(shape, dtype=torch.float32, device=resolved)
    alive = torch.zeros(
        (batch, contract.capacity), dtype=torch.bool, device=resolved
    )
    agent_ids = torch.full(
        (batch, contract.capacity), -1, dtype=torch.int64, device=resolved
    )
    ancestry = torch.zeros(
        (batch, contract.capacity, contract.population_size),
        dtype=torch.float32,
        device=resolved,
    )
    for cell, pattern_packages in enumerate(packages):
        start = 0
        for group in founder_groups:
            stop = start + group.count
            loci[cell, start:stop] = torch.tensor(
                pattern_packages[group.group_id],
                dtype=torch.float32,
                device=resolved,
            )
            start = stop
        alive[cell, : contract.population_size] = True
        agent_ids[cell, : contract.population_size] = torch.arange(
            contract.population_size, dtype=torch.int64, device=resolved
        )
        ancestry[cell, : contract.population_size] = torch.eye(
            contract.population_size, dtype=torch.float32, device=resolved
        )
    return TensorCrossPathStateV1(
        contract=contract,
        loci=loci,
        alive=alive,
        agent_ids=agent_ids,
        ancestry=ancestry,
        next_id=torch.full(
            (batch,),
            contract.population_size,
            dtype=torch.int64,
            device=resolved,
        ),
        generation=torch.zeros(batch, dtype=torch.int64, device=resolved),
        seeds=tuple(int(value) for value in seeds),
        profile_ids=tuple(str(value) for value in profile_ids),
        path_ids=tuple(str(value) for value in path_ids),
        pattern_ids=tuple(str(value) for value in pattern_ids),
    )


def restore_tensor_cross_path_state_v1(
    contract: TensorInheritanceContractV1,
    payload: Mapping[str, Any],
    *,
    device: str | torch.device,
) -> TensorCrossPathStateV1:
    raise CrossPathTensorError(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "standalone tensor inheritance restoration is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("contract_sha256") != contract.sha256
        or payload.get("contract") != contract.payload()
    ):
        raise CrossPathTensorError("checkpoint contract mismatch")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise CrossPathTensorError("checkpoint tensors are missing")
    return TensorCrossPathStateV1(
        contract=contract,
        seeds=tuple(int(value) for value in payload["seeds"]),
        profile_ids=tuple(str(value) for value in payload["profile_ids"]),
        path_ids=tuple(str(value) for value in payload["path_ids"]),
        pattern_ids=tuple(str(value) for value in payload["pattern_ids"]),
        **{
            name: torch.as_tensor(tensors[name]).to(device=device).contiguous()
            for name in (
                "loci",
                "alive",
                "agent_ids",
                "ancestry",
                "next_id",
                "generation",
            )
        },
    )


def _canonical_pair(
    loci_a: torch.Tensor,
    loci_b: torch.Tensor,
    id_a: torch.Tensor,
    id_b: torch.Tensor,
    *,
    digest_a: str,
    digest_b: str,
    integer_id_a: int,
    integer_id_b: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a_first = (digest_a, integer_id_a) <= (digest_b, integer_id_b)
    if a_first:
        return loci_a, loci_b, id_a, id_b
    return (
        loci_b,
        loci_a,
        id_b,
        id_a,
    )


def advance_tensor_cross_path_state_v1(
    state: TensorCrossPathStateV1,
    *,
    generations: int,
) -> dict[str, torch.Tensor]:
    raise CrossPathTensorError(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "standalone tensor inheritance execution is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    if generations < 1:
        raise CrossPathTensorError("generation count must be positive")
    contract = state.contract
    batch = state.batch_cells
    max_events = max(row.birth_opportunities for row in contract.paths)
    locus_count = len(contract.locus_names)
    block_count = len(contract.linkage_block_ids)
    device = state.loci.device
    event_shape = (batch, generations, max_events)
    outputs = {
        "event_valid": torch.zeros(event_shape, dtype=torch.bool, device=device),
        "event_success": torch.zeros(event_shape, dtype=torch.bool, device=device),
        "failure_reason_code": torch.zeros(
            event_shape, dtype=torch.int8, device=device
        ),
        "parent_agent_ids": torch.full(
            event_shape, -1, dtype=torch.int64, device=device
        ),
        "mate_agent_ids": torch.full(
            event_shape, -1, dtype=torch.int64, device=device
        ),
        "child_agent_ids": torch.full(
            event_shape, -1, dtype=torch.int64, device=device
        ),
        "parent_loci": torch.zeros(
            (*event_shape, locus_count), dtype=torch.float32, device=device
        ),
        "mate_loci": torch.zeros(
            (*event_shape, locus_count), dtype=torch.float32, device=device
        ),
        "child_loci": torch.zeros(
            (*event_shape, locus_count), dtype=torch.float32, device=device
        ),
        "pre_mutation_child_loci": torch.zeros(
            (*event_shape, locus_count), dtype=torch.float32, device=device
        ),
        "linkage_donor_ordinals": torch.full(
            (*event_shape, block_count), -1, dtype=torch.int8, device=device
        ),
        "inheritance_draw_count": torch.zeros(
            event_shape, dtype=torch.int32, device=device
        ),
        "linked_package_integrity": torch.ones(
            event_shape, dtype=torch.bool, device=device
        ),
        "birth_opportunities": torch.zeros(
            (batch, generations), dtype=torch.int32, device=device
        ),
        "successful_births": torch.zeros(
            (batch, generations), dtype=torch.int32, device=device
        ),
        "failed_birth_attempts": torch.zeros(
            (batch, generations), dtype=torch.int32, device=device
        ),
        "refill_children": torch.zeros(
            (batch, generations), dtype=torch.int32, device=device
        ),
        "ordinary_births": torch.zeros(
            (batch, generations), dtype=torch.int32, device=device
        ),
        "mutation_events": torch.zeros(
            event_shape, dtype=torch.int32, device=device
        ),
        "mutation_draw_count": torch.zeros(
            event_shape, dtype=torch.int32, device=device
        ),
        "mutated_locus_mask": torch.zeros(
            (*event_shape, locus_count), dtype=torch.bool, device=device
        ),
    }
    bounds_low = torch.tensor(
        contract.lower_bounds, dtype=torch.float32, device=device
    )
    bounds_high = torch.tensor(
        contract.upper_bounds, dtype=torch.float32, device=device
    )
    mutation = contract.mutation_policy
    mutation_delta_by_locus = dict(mutation.locus_deltas)
    mutation_delta = torch.tensor(
        [
            float(mutation_delta_by_locus.get(name, 0.0))
            for name in contract.locus_names
        ],
        dtype=torch.float32,
        device=device,
    )
    starting_generations = tuple(
        int(value) for value in state.generation.detach().cpu().tolist()
    )

    for step in range(generations):
        for cell in range(batch):
            generation = starting_generations[cell] + step + 1
            seed = state.seeds[cell]
            profile = _profile(contract, state.profile_ids[cell])
            path = _path(contract, state.path_ids[cell])
            population = contract.population_size
            schedule_namespace = path.schedule_namespace or path.path_id
            ids_cpu = [
                int(value)
                for value in state.agent_ids[cell, :population]
                .detach()
                .cpu()
                .tolist()
            ]
            loci_cpu = [
                [float(value) for value in row]
                for row in state.loci[cell, :population].detach().cpu().tolist()
            ]
            ranked_slots = sorted(
                range(population),
                key=lambda slot: neutral_cull_rank(
                    ids_cpu[slot],
                    contract_sha256=contract.control_schedule_sha256,
                    schedule_namespace=schedule_namespace,
                    seed=seed,
                    generation=generation,
                ),
            )
            survivor_count = population - path.successful_births
            parent_pool_slots = ranked_slots[:survivor_count]
            victim_slots = ranked_slots[survivor_count:]
            eligible_ids = tuple(ids_cpu[slot] for slot in parent_pool_slots)
            slot_by_id = {
                ids_cpu[slot]: slot for slot in parent_pool_slots
            }
            package_digest_by_slot = {
                slot: canonical_package_sha256(
                    contract.locus_names, loci_cpu[slot]
                )
                for slot in parent_pool_slots
            }

            for event in range(path.birth_opportunities):
                parent_id, mate_id = scheduled_parent_ids(
                    eligible_ids,
                    contract_sha256=contract.control_schedule_sha256,
                    schedule_namespace=schedule_namespace,
                    seed=seed,
                    generation=generation,
                    opportunity_ordinal=event,
                )
                parent_slot = slot_by_id[parent_id]
                mate_slot = slot_by_id[mate_id]
                a, b, id_a, id_b = _canonical_pair(
                    state.loci[cell, parent_slot],
                    state.loci[cell, mate_slot],
                    state.agent_ids[cell, parent_slot],
                    state.agent_ids[cell, mate_slot],
                    digest_a=package_digest_by_slot[parent_slot],
                    digest_b=package_digest_by_slot[mate_slot],
                    integer_id_a=parent_id,
                    integer_id_b=mate_id,
                )
                outputs["event_valid"][cell, step, event] = True
                outputs["parent_agent_ids"][cell, step, event] = id_a
                outputs["mate_agent_ids"][cell, step, event] = id_b
                outputs["parent_loci"][cell, step, event] = a
                outputs["mate_loci"][cell, step, event] = b
                if event >= path.successful_births:
                    outputs["failure_reason_code"][cell, step, event] = 1
                    continue

                victim = victim_slots[event]
                if profile.operator == "midpoint":
                    child = 0.5 * (a + b)
                    donors = [-1] * block_count
                    integrity = torch.ones(
                        (), dtype=torch.bool, device=device
                    )
                else:
                    donors = list(
                        donor_ordinals(
                            profile.draw_count,
                            contract_sha256=contract.control_schedule_sha256,
                            schedule_namespace=schedule_namespace,
                            seed=seed,
                            generation=generation,
                            opportunity_ordinal=event,
                        )
                    )
                    values = []
                    for locus, block in enumerate(
                        contract.linked_locus_block_indices
                    ):
                        values.append(
                            0.5 * (a[locus] + b[locus])
                            if block == -1
                            else (a[locus] if donors[block] == 0 else b[locus])
                        )
                    child = torch.stack(values)
                    integrity = torch.ones(
                        (), dtype=torch.bool, device=device
                    )
                    for locus, block in enumerate(
                        contract.linked_locus_block_indices
                    ):
                        expected = (
                            0.5 * (a[locus] + b[locus])
                            if block == -1
                            else (
                                a[locus]
                                if donors[block] == 0
                                else b[locus]
                            )
                        )
                        integrity = integrity & (child[locus] == expected)
                pre_mutation_child = torch.minimum(
                    bounds_high, torch.maximum(bounds_low, child)
                )
                child = pre_mutation_child
                mutation_triggered = (
                    mutation.enabled
                    and schedule_word(
                        contract.control_schedule_sha256,
                        schedule_namespace,
                        seed,
                        generation,
                        event,
                        mutation.schedule_channel,
                    )
                    % mutation.event_modulus
                    == mutation.event_residue
                )
                if mutation_triggered:
                    child = torch.minimum(
                        bounds_high,
                        torch.maximum(
                            bounds_low, pre_mutation_child + mutation_delta
                        ),
                    )
                mutated_locus_mask = child != pre_mutation_child
                mutation_event_count = mutated_locus_mask.any().to(torch.int32)
                child_id = state.next_id[cell].clone()
                ancestry = 0.5 * (
                    state.ancestry[cell, parent_slot]
                    + state.ancestry[cell, mate_slot]
                )
                state.loci[cell, victim] = child
                state.ancestry[cell, victim] = ancestry
                state.agent_ids[cell, victim] = child_id
                state.next_id[cell] += 1

                outputs["event_success"][cell, step, event] = True
                outputs["child_agent_ids"][cell, step, event] = child_id
                outputs["child_loci"][cell, step, event] = child
                outputs["pre_mutation_child_loci"][
                    cell, step, event
                ] = pre_mutation_child
                outputs["linkage_donor_ordinals"][cell, step, event] = (
                    torch.tensor(donors, dtype=torch.int8, device=device)
                )
                outputs["inheritance_draw_count"][cell, step, event] = (
                    profile.draw_count
                )
                outputs["linked_package_integrity"][cell, step, event] = (
                    integrity
                )
                outputs["mutation_events"][
                    cell, step, event
                ] = mutation_event_count
                outputs["mutation_draw_count"][
                    cell, step, event
                ] = mutation.draw_budget
                outputs["mutated_locus_mask"][
                    cell, step, event
                ] = mutated_locus_mask

            state.generation[cell] += 1
            outputs["birth_opportunities"][cell, step] = path.birth_opportunities
            outputs["successful_births"][cell, step] = path.successful_births
            outputs["failed_birth_attempts"][cell, step] = (
                path.birth_opportunities - path.successful_births
            )
            if path.replacement_mode == "batch_survivor_refill":
                outputs["refill_children"][cell, step] = path.successful_births
            else:
                outputs["ordinary_births"][cell, step] = path.successful_births
    return outputs


def summarize_tensor_cross_path_state_v1(
    state: TensorCrossPathStateV1,
) -> dict[str, torch.Tensor]:
    population = state.contract.population_size
    values = state.loci[:, :population]
    total = torch.zeros_like(values[:, 0])
    minimum = values[:, 0].clone()
    maximum = values[:, 0].clone()
    for index in range(population):
        total = total + values[:, index]
        minimum = torch.minimum(minimum, values[:, index])
        maximum = torch.maximum(maximum, values[:, index])
    mean = total / population
    squared_delta_total = torch.zeros_like(mean)
    for index in range(population):
        squared_delta_total = squared_delta_total + (
            values[:, index] - mean
        ).square()
    ancestry_mean = torch.zeros_like(state.ancestry[:, 0])
    for index in range(population):
        ancestry_mean = ancestry_mean + state.ancestry[:, index]
    ancestry_mean = ancestry_mean / population
    ancestry_square_total = torch.zeros(
        ancestry_mean.shape[0],
        dtype=ancestry_mean.dtype,
        device=ancestry_mean.device,
    )
    dominant_ancestry = ancestry_mean[:, 0].clone()
    for index in range(population):
        ancestry_square_total = (
            ancestry_square_total + ancestry_mean[:, index].square()
        )
        dominant_ancestry = torch.maximum(
            dominant_ancestry, ancestry_mean[:, index]
        )
    return {
        "mean": mean,
        "variance_population": squared_delta_total / population,
        "minimum": minimum,
        "maximum": maximum,
        "alive": state.alive[:, :population].sum(dim=1),
        "effective_founder_ancestry_count": 1.0 / ancestry_square_total,
        "dominant_founder_ancestry_share": dominant_ancestry,
    }


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CrossPathTensorError",
    "FounderGroupPolicyV1",
    "MUTATION_DETERMINISTIC_DELTA_OPERATOR",
    "MUTATION_OFF_OPERATOR",
    "ProfilePolicyV1",
    "ReproductionPathPolicyV1",
    "SCHEMA_VERSION",
    "TensorCrossPathStateV1",
    "TensorInheritanceContractV1",
    "TensorMutationPolicyV1",
    "advance_tensor_cross_path_state_v1",
    "initialize_tensor_cross_path_state_v1",
    "restore_tensor_cross_path_state_v1",
    "summarize_tensor_cross_path_state_v1",
]
