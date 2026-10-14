"""Generic immutable reference mechanics for cross-path inheritance parity.

The module owns no campaign dimensions or production routing. Callers provide
the loci, linkage blocks, profiles, founders, path policies, and each bounded
execution request.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence

from engine.evolution.inheritance_cross_path_schedule import schedule_word


MIDPOINT_OPERATOR = "midpoint"
LINKED_PARENT_COPY_OPERATOR = "linked_parent_copy"
PROFILE_OPERATORS = (MIDPOINT_OPERATOR, LINKED_PARENT_COPY_OPERATOR)
ATTEMPT_PREFIX = "prefix"
ATTEMPT_HASH_RANKED = "hash_ranked"
ATTEMPT_SELECTION_POLICIES = (ATTEMPT_PREFIX, ATTEMPT_HASH_RANKED)
MUTATION_OFF_OPERATOR = "off"
MUTATION_DETERMINISTIC_DELTA_OPERATOR = "deterministic_delta"
MUTATION_OPERATORS = (
    MUTATION_OFF_OPERATOR,
    MUTATION_DETERMINISTIC_DELTA_OPERATOR,
)


class InheritanceCrossPathError(ValueError):
    """Raised when a reference contract or execution request is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InheritanceCrossPathError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise InheritanceCrossPathError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class LocusSpec:
    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not str(self.name):
            raise InheritanceCrossPathError("locus name must be non-empty")
        lower = _finite(self.lower, label=f"{self.name}.lower")
        upper = _finite(self.upper, label=f"{self.name}.upper")
        if lower > upper:
            raise InheritanceCrossPathError(f"{self.name} bounds are reversed")


@dataclass(frozen=True)
class LinkageBlockSpec:
    block_id: str
    locus_names: tuple[str, ...]
    consume_draw: bool = True

    def __post_init__(self) -> None:
        if not str(self.block_id):
            raise InheritanceCrossPathError("linkage block ID must be non-empty")
        if len(set(self.locus_names)) != len(self.locus_names):
            raise InheritanceCrossPathError(
                f"duplicate locus in linkage block {self.block_id!r}"
            )
        if self.locus_names and not self.consume_draw:
            raise InheritanceCrossPathError(
                f"non-empty linkage block {self.block_id!r} must consume a donor draw"
            )


@dataclass(frozen=True)
class ProfileSpec:
    profile_id: str
    operator: str
    draw_budget: int
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.profile_id):
            raise InheritanceCrossPathError("profile ID must be non-empty")
        if self.operator not in PROFILE_OPERATORS:
            raise InheritanceCrossPathError(
                f"unsupported inheritance operator {self.operator!r}"
            )
        if int(self.draw_budget) < 0:
            raise InheritanceCrossPathError("profile draw budget must be non-negative")


@dataclass(frozen=True)
class FounderPackageSpec:
    founder_id: str
    values: tuple[tuple[str, float], ...]

    @classmethod
    def from_mapping(
        cls, founder_id: str, values: Mapping[str, Any]
    ) -> "FounderPackageSpec":
        return cls(
            str(founder_id),
            tuple((str(key), float(value)) for key, value in sorted(values.items())),
        )

    def as_mapping(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.values}


@dataclass(frozen=True)
class FounderGroupSpec:
    group_id: str
    founder_id: str
    count: int

    def __post_init__(self) -> None:
        if not str(self.group_id) or not str(self.founder_id):
            raise InheritanceCrossPathError(
                "founder group and package IDs must be non-empty"
            )
        if type(self.count) is not int or self.count < 1:
            raise InheritanceCrossPathError("founder group count must be positive")


@dataclass(frozen=True)
class PathPolicy:
    path_id: str
    schedule_namespace: str
    attempt_selection: str = ATTEMPT_PREFIX
    capacity_failure_reason: str = "capacity_full"
    ancestry_parent_weights: tuple[float, float] = (0.5, 0.5)

    def __post_init__(self) -> None:
        if not str(self.path_id) or not str(self.schedule_namespace):
            raise InheritanceCrossPathError(
                "path ID and schedule namespace must be non-empty"
            )
        if self.attempt_selection not in ATTEMPT_SELECTION_POLICIES:
            raise InheritanceCrossPathError(
                f"unsupported attempt selection {self.attempt_selection!r}"
            )
        weights = tuple(
            _finite(value, label=f"{self.path_id}.ancestry_weight")
            for value in self.ancestry_parent_weights
        )
        if len(weights) != 2 or any(value < 0.0 for value in weights):
            raise InheritanceCrossPathError(
                "path ancestry weights must be two non-negative values"
            )
        if sum(weights) <= 0.0:
            raise InheritanceCrossPathError("path ancestry weights need positive mass")


@dataclass(frozen=True)
class MutationPolicy:
    policy_id: str
    operator: str = MUTATION_OFF_OPERATOR
    locus_deltas: tuple[tuple[str, float], ...] = ()
    event_modulus: int = 1
    event_residue: int = 0
    draw_budget: int = 0
    schedule_channel: str = "mutation"

    def __post_init__(self) -> None:
        if not str(self.policy_id):
            raise InheritanceCrossPathError("mutation policy ID must be non-empty")
        if self.operator not in MUTATION_OPERATORS:
            raise InheritanceCrossPathError(
                f"unsupported mutation operator {self.operator!r}"
            )
        if (
            type(self.event_modulus) is not int
            or self.event_modulus < 1
            or type(self.event_residue) is not int
            or not 0 <= self.event_residue < self.event_modulus
            or type(self.draw_budget) is not int
            or self.draw_budget < 0
            or not str(self.schedule_channel)
        ):
            raise InheritanceCrossPathError("mutation schedule contract is invalid")
        names = tuple(str(name) for name, _value in self.locus_deltas)
        if len(set(names)) != len(names):
            raise InheritanceCrossPathError("mutation loci must be unique")
        deltas = tuple(
            _finite(value, label=f"mutation.{name}")
            for name, value in self.locus_deltas
        )
        if self.operator == MUTATION_OFF_OPERATOR:
            if names or self.draw_budget:
                raise InheritanceCrossPathError(
                    "mutation-off policy cannot declare deltas or draws"
                )
        elif not names or not any(value != 0.0 for value in deltas):
            raise InheritanceCrossPathError(
                "deterministic mutation requires a nonzero locus delta"
            )
        elif self.draw_budget != 0:
            raise InheritanceCrossPathError(
                "deterministic-delta mutation consumes zero RNG draws"
            )

    @property
    def enabled(self) -> bool:
        return self.operator != MUTATION_OFF_OPERATOR

    def delta_mapping(self) -> dict[str, float]:
        return {str(name): float(value) for name, value in self.locus_deltas}

    def payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "operator": self.operator,
            "enabled": self.enabled,
            "locus_deltas": self.delta_mapping(),
            "event_modulus": self.event_modulus,
            "event_residue": self.event_residue,
            "draw_budget": self.draw_budget,
            "schedule_channel": self.schedule_channel,
        }


@dataclass(frozen=True)
class InheritanceContract:
    schema_version: str
    registry_version: str
    loci: tuple[LocusSpec, ...]
    linkage_blocks: tuple[LinkageBlockSpec, ...]
    profiles: tuple[ProfileSpec, ...]
    default_profile_id: str
    founders: tuple[FounderPackageSpec, ...]
    path_policies: tuple[PathPolicy, ...]
    mutation_policy: MutationPolicy
    founder_groups: tuple[FounderGroupSpec, ...] = ()
    parent_order_version: str = "canonical_package_sha256_then_parent_id_v1"
    event_schema_version: str = "inheritance_cross_path_event_v1"
    capacity_accounting_version: str = "fixed_capacity_accounting_v1"
    ancestry_version: str = "weighted_normalized_parent_ancestry_v1"

    def __post_init__(self) -> None:
        if not self.schema_version or not self.registry_version:
            raise InheritanceCrossPathError(
                "schema and registry versions must be non-empty"
            )
        locus_names = tuple(spec.name for spec in self.loci)
        if not locus_names or len(set(locus_names)) != len(locus_names):
            raise InheritanceCrossPathError("contract loci must be non-empty and unique")
        block_ids = tuple(block.block_id for block in self.linkage_blocks)
        if len(set(block_ids)) != len(block_ids):
            raise InheritanceCrossPathError("linkage block IDs must be unique")
        projected = tuple(
            name for block in self.linkage_blocks for name in block.locus_names
        )
        if sorted(projected) != sorted(locus_names) or len(projected) != len(
            set(projected)
        ):
            raise InheritanceCrossPathError(
                "every locus must occur in exactly one linkage block"
            )
        profiles = {profile.profile_id: profile for profile in self.profiles}
        if len(profiles) != len(self.profiles) or self.default_profile_id not in profiles:
            raise InheritanceCrossPathError(
                "profile IDs must be unique and include the default"
            )
        aliases = [
            alias for profile in self.profiles for alias in profile.aliases if alias
        ]
        if len(set(aliases)) != len(aliases) or set(aliases) & set(profiles):
            raise InheritanceCrossPathError("profile aliases must be unique")
        linked_budget = sum(
            1 for block in self.linkage_blocks if block.consume_draw
        )
        for profile in self.profiles:
            expected = (
                0 if profile.operator == MIDPOINT_OPERATOR else linked_budget
            )
            if int(profile.draw_budget) != expected:
                raise InheritanceCrossPathError(
                    f"profile {profile.profile_id!r} draw budget must be {expected}"
                )
        founder_ids = tuple(founder.founder_id for founder in self.founders)
        if not founder_ids or len(set(founder_ids)) != len(founder_ids):
            raise InheritanceCrossPathError(
                "founder IDs must be non-empty and unique"
            )
        for founder in self.founders:
            self.canonical_package(founder.as_mapping())
        group_ids = tuple(group.group_id for group in self.founder_groups)
        if len(set(group_ids)) != len(group_ids):
            raise InheritanceCrossPathError("founder group IDs must be unique")
        unknown_founders = sorted(
            group.founder_id
            for group in self.founder_groups
            if group.founder_id not in founder_ids
        )
        if unknown_founders:
            raise InheritanceCrossPathError(
                f"founder groups reference unknown packages: {unknown_founders!r}"
            )
        mutation_loci = set(self.mutation_policy.delta_mapping())
        unknown_mutation_loci = sorted(mutation_loci - set(locus_names))
        if unknown_mutation_loci:
            raise InheritanceCrossPathError(
                f"mutation policy references unknown loci: "
                f"{unknown_mutation_loci!r}"
            )
        path_ids = tuple(policy.path_id for policy in self.path_policies)
        namespaces = tuple(policy.schedule_namespace for policy in self.path_policies)
        if (
            not path_ids
            or len(set(path_ids)) != len(path_ids)
            or len(set(namespaces)) != len(namespaces)
        ):
            raise InheritanceCrossPathError(
                "path IDs and schedule namespaces must be non-empty and unique"
            )

    def canonical_package(self, raw: Mapping[str, Any]) -> dict[str, float]:
        if not isinstance(raw, Mapping):
            raise InheritanceCrossPathError("package must be a mapping")
        locus_names = tuple(spec.name for spec in self.loci)
        missing = [name for name in locus_names if name not in raw]
        extra = sorted(str(key) for key in raw if key not in locus_names)
        if missing or extra:
            raise InheritanceCrossPathError(
                f"package fields mismatch: missing={missing!r} extra={extra!r}"
            )
        output = {}
        for spec in self.loci:
            value = _finite(raw[spec.name], label=spec.name)
            if value < spec.lower or value > spec.upper:
                raise InheritanceCrossPathError(
                    f"{spec.name}={value!r} is outside [{spec.lower}, {spec.upper}]"
                )
            output[spec.name] = value
        return output

    def resolve_profile(self, value: Any = None) -> ProfileSpec:
        selected = str(value or self.default_profile_id).strip()
        for profile in self.profiles:
            if selected == profile.profile_id or selected in profile.aliases:
                return profile
        raise InheritanceCrossPathError(f"unsupported profile {selected!r}")

    def path_policy(self, path_id: Any) -> PathPolicy:
        selected = str(path_id or "").strip()
        for policy in self.path_policies:
            if policy.path_id == selected:
                return policy
        raise InheritanceCrossPathError(f"unsupported reproduction path {selected!r}")

    def founder_package(self, founder_id: Any) -> dict[str, float]:
        selected = str(founder_id or "")
        for founder in self.founders:
            if founder.founder_id == selected:
                return self.canonical_package(founder.as_mapping())
        raise InheritanceCrossPathError(f"unknown founder {selected!r}")

    def founder_group_package(self, group_id: Any) -> dict[str, float]:
        selected = str(group_id or "")
        for group in self.founder_groups:
            if group.group_id == selected:
                return self.founder_package(group.founder_id)
        raise InheritanceCrossPathError(f"unknown founder group {selected!r}")

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "loci": [
                {"name": spec.name, "bounds": [spec.lower, spec.upper]}
                for spec in self.loci
            ],
            "linkage_blocks": [
                {
                    "block_id": block.block_id,
                    "locus_names": list(block.locus_names),
                    "consume_draw": block.consume_draw,
                }
                for block in self.linkage_blocks
            ],
            "profiles": [
                {
                    "profile_id": profile.profile_id,
                    "operator": profile.operator,
                    "draw_budget": profile.draw_budget,
                    "aliases": list(profile.aliases),
                }
                for profile in self.profiles
            ],
            "default_profile_id": self.default_profile_id,
            "founders": [
                {
                    "founder_id": founder.founder_id,
                    "values": self.canonical_package(founder.as_mapping()),
                }
                for founder in self.founders
            ],
            "founder_groups": [
                {
                    "group_id": group.group_id,
                    "founder_id": group.founder_id,
                    "count": group.count,
                }
                for group in self.founder_groups
            ],
            "path_policies": [
                {
                    "path_id": policy.path_id,
                    "schedule_namespace": policy.schedule_namespace,
                    "attempt_selection": policy.attempt_selection,
                    "capacity_failure_reason": policy.capacity_failure_reason,
                    "ancestry_parent_weights": list(policy.ancestry_parent_weights),
                }
                for policy in self.path_policies
            ],
            "mutation_policy": self.mutation_policy.payload(),
            "parent_order_version": self.parent_order_version,
            "event_schema_version": self.event_schema_version,
            "capacity_accounting_version": self.capacity_accounting_version,
            "ancestry_version": self.ancestry_version,
        }

    def registry_sha256(self) -> str:
        return _sha256(self.metadata())

    def validate_registry_sha256(self, supplied: Any) -> str:
        value = str(supplied or "")
        expected = self.registry_sha256()
        if value != expected:
            raise InheritanceCrossPathError(
                f"registry mismatch: expected={expected} supplied={value}"
            )
        return value


@dataclass(frozen=True)
class ReferenceParent:
    parent_id: int
    package: tuple[tuple[str, float], ...]
    ancestry: tuple[tuple[str, float], ...]

    @classmethod
    def from_mappings(
        cls,
        parent_id: int,
        package: Mapping[str, Any],
        ancestry: Mapping[str, Any],
    ) -> "ReferenceParent":
        return cls(
            int(parent_id),
            tuple((str(key), float(value)) for key, value in sorted(package.items())),
            tuple((str(key), float(value)) for key, value in sorted(ancestry.items())),
        )

    def package_mapping(self) -> dict[str, float]:
        return dict(self.package)

    def ancestry_mapping(self) -> dict[str, float]:
        return dict(self.ancestry)


@dataclass(frozen=True)
class ProjectionResult:
    profile_id: str
    child: tuple[tuple[str, float], ...]
    draw_count: int
    donor_ordinals: tuple[int, ...]
    donor_parent_id_by_block: tuple[tuple[str, int], ...]
    donor_parent_id_by_locus: tuple[tuple[str, int], ...]
    canonical_parent_ids: tuple[int, int]
    canonical_parent_package_sha256: tuple[str, str]

    def child_mapping(self) -> dict[str, float]:
        return dict(self.child)


@dataclass(frozen=True)
class PathExecutionRequest:
    path_id: str
    seed: int
    generation: int
    opportunity_count: int
    birth_attempt_count: int
    starting_population: int
    capacity: int
    first_child_id: int

    def __post_init__(self) -> None:
        integers = {
            "generation": self.generation,
            "opportunity_count": self.opportunity_count,
            "birth_attempt_count": self.birth_attempt_count,
            "starting_population": self.starting_population,
            "capacity": self.capacity,
            "first_child_id": self.first_child_id,
        }
        if any(int(value) < 0 for value in integers.values()):
            raise InheritanceCrossPathError(
                f"execution request values must be non-negative: {integers!r}"
            )
        if self.birth_attempt_count > self.opportunity_count:
            raise InheritanceCrossPathError(
                "birth attempts cannot exceed available opportunities"
            )
        if self.starting_population > self.capacity:
            raise InheritanceCrossPathError(
                "starting population exceeds fixed capacity"
            )


@dataclass(frozen=True)
class ScheduledOpportunity:
    opportunity_ordinal: int
    parent_ids: tuple[int, int]


@dataclass(frozen=True)
class ChildEvent:
    schema_version: str
    contract_schema_version: str
    registry_version: str
    registry_sha256: str
    path_id: str
    schedule_namespace: str
    profile_id: str
    seed: int
    generation: int
    opportunity_ordinal: int
    attempted: bool
    scheduled_parent_ids: tuple[int, int]
    population_before: int
    population_after: int
    capacity: int
    capacity_accounting_version: str
    mutation_policy: MutationPolicy
    mutation_event_count: int
    mutation_draw_count: int
    mutated_loci: tuple[str, ...]
    pre_mutation_child_sha256: str | None
    success: bool
    failure_reason: str | None
    child_id: int | None
    child: tuple[tuple[str, float], ...] | None
    child_sha256: str | None
    canonical_parent_ids: tuple[int, int] | None
    canonical_parent_package_sha256: tuple[str, str] | None
    inheritance_draw_count: int
    donor_ordinals: tuple[int, ...]
    donor_parent_id_by_block: tuple[tuple[str, int], ...]
    donor_parent_id_by_locus: tuple[tuple[str, int], ...]
    ancestry: tuple[tuple[str, float], ...] | None

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_schema_version": self.contract_schema_version,
            "registry_version": self.registry_version,
            "registry_sha256": self.registry_sha256,
            "path_id": self.path_id,
            "schedule_namespace": self.schedule_namespace,
            "profile_id": self.profile_id,
            "seed": self.seed,
            "generation": self.generation,
            "opportunity_ordinal": self.opportunity_ordinal,
            "attempted": self.attempted,
            "scheduled_parent_ids": list(self.scheduled_parent_ids),
            "population_before": self.population_before,
            "population_after": self.population_after,
            "capacity": self.capacity,
            "capacity_accounting_version": self.capacity_accounting_version,
            "mutation": {
                "policy_id": self.mutation_policy.policy_id,
                "operator": self.mutation_policy.operator,
                "enabled": self.mutation_policy.enabled,
                "event_count": self.mutation_event_count,
                "rng_draw_count": self.mutation_draw_count,
                "mutated_loci": list(self.mutated_loci),
                "pre_mutation_child_sha256": self.pre_mutation_child_sha256,
            },
            "success": self.success,
            "failure_reason": self.failure_reason,
            "child_id": self.child_id,
            "child": None if self.child is None else dict(self.child),
            "child_sha256": self.child_sha256,
            "canonical_parent_ids": (
                None
                if self.canonical_parent_ids is None
                else list(self.canonical_parent_ids)
            ),
            "canonical_parent_package_sha256": (
                None
                if self.canonical_parent_package_sha256 is None
                else list(self.canonical_parent_package_sha256)
            ),
            "inheritance_draw_count": self.inheritance_draw_count,
            "donor_ordinals": list(self.donor_ordinals),
            "donor_parent_id_by_block": dict(self.donor_parent_id_by_block),
            "donor_parent_id_by_locus": dict(self.donor_parent_id_by_locus),
            "ancestry": None if self.ancestry is None else dict(self.ancestry),
        }


@dataclass(frozen=True)
class PathExecutionResult:
    path_id: str
    profile_id: str
    opportunity_count: int
    birth_attempts: int
    non_birth_opportunities: int
    successes: int
    failures: int
    starting_population: int
    ending_population: int
    capacity: int
    events: tuple[ChildEvent, ...]
    registry_sha256: str


def _schedule_word(contract: InheritanceContract, *parts: Any) -> int:
    if len(parts) != 5:
        raise InheritanceCrossPathError(
            "cross-path schedule requires namespace, seed, generation, "
            "opportunity, and channel"
        )
    namespace, seed, generation, opportunity, channel = parts
    return schedule_word(
        contract.registry_sha256(),
        str(namespace),
        int(seed),
        int(generation),
        int(opportunity),
        str(channel),
    )


def _canonical_parent_order(
    contract: InheritanceContract,
    parent_a: ReferenceParent,
    parent_b: ReferenceParent,
) -> tuple[tuple[ReferenceParent, ReferenceParent], tuple[str, str]]:
    if parent_a.parent_id == parent_b.parent_id:
        raise InheritanceCrossPathError("two distinct parent IDs are required")
    rows = []
    for parent in (parent_a, parent_b):
        package = contract.canonical_package(parent.package_mapping())
        rows.append((_sha256(package), parent.parent_id, parent))
    rows.sort(key=lambda row: (row[0], row[1]))
    return (rows[0][2], rows[1][2]), (rows[0][0], rows[1][0])


def project_inheritance(
    contract: InheritanceContract,
    parent_a: ReferenceParent,
    parent_b: ReferenceParent,
    *,
    rng: random.Random,
    profile_id: Any = None,
    expected_registry_sha256: str | None = None,
) -> ProjectionResult:
    """Reject execution of the retired standalone reference inheritance path."""

    raise InheritanceCrossPathError(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "standalone reference inheritance execution is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    profile = contract.resolve_profile(profile_id)
    if expected_registry_sha256 is not None:
        contract.validate_registry_sha256(expected_registry_sha256)
    if rng is None or not callable(getattr(rng, "random", None)):
        raise InheritanceCrossPathError("inheritance requires a random-compatible RNG")
    parents, digests = _canonical_parent_order(contract, parent_a, parent_b)
    packages = tuple(
        contract.canonical_package(parent.package_mapping()) for parent in parents
    )
    child: dict[str, float] = {}
    ordinals: list[int] = []
    donor_blocks: dict[str, int] = {}
    donor_loci: dict[str, int] = {}

    if profile.operator == MIDPOINT_OPERATOR:
        for locus in contract.loci:
            child[locus.name] = 0.5 * (
                packages[0][locus.name] + packages[1][locus.name]
            )
    else:
        for block in contract.linkage_blocks:
            if not block.consume_draw:
                continue
            draw = _finite(rng.random(), label=f"{block.block_id}.donor_draw")
            if not 0.0 <= draw < 1.0:
                raise InheritanceCrossPathError(
                    "inheritance RNG returned a value outside [0, 1)"
                )
            ordinal = 0 if draw < 0.5 else 1
            ordinals.append(ordinal)
            donor_id = parents[ordinal].parent_id
            donor_blocks[block.block_id] = donor_id
            for locus_name in block.locus_names:
                child[locus_name] = packages[ordinal][locus_name]
                donor_loci[locus_name] = donor_id

    child = contract.canonical_package(child)
    if len(ordinals) != profile.draw_budget:
        raise RuntimeError(
            f"inheritance draw mismatch: expected={profile.draw_budget} "
            f"observed={len(ordinals)}"
        )
    return ProjectionResult(
        profile_id=profile.profile_id,
        child=tuple(child.items()),
        draw_count=len(ordinals),
        donor_ordinals=tuple(ordinals),
        donor_parent_id_by_block=tuple(donor_blocks.items()),
        donor_parent_id_by_locus=tuple(donor_loci.items()),
        canonical_parent_ids=(parents[0].parent_id, parents[1].parent_id),
        canonical_parent_package_sha256=digests,
    )


def deterministic_path_schedule(
    contract: InheritanceContract,
    eligible_parent_ids: Sequence[int],
    request: PathExecutionRequest,
) -> tuple[ScheduledOpportunity, ...]:
    """Choose birth opportunities and genotype-blind parent pairs."""

    policy = contract.path_policy(request.path_id)
    supplied = tuple(int(value) for value in eligible_parent_ids)
    if len(set(supplied)) != len(supplied):
        raise InheritanceCrossPathError("eligible parent IDs must be unique")
    parent_ids = tuple(sorted(supplied))
    if len(parent_ids) < 2:
        raise InheritanceCrossPathError("path schedule requires two parents")
    candidates = list(range(request.opportunity_count))
    if policy.attempt_selection == ATTEMPT_HASH_RANKED:
        candidates.sort(
            key=lambda ordinal: (
                _schedule_word(
                    contract,
                    policy.schedule_namespace,
                    request.seed,
                    request.generation,
                    ordinal,
                    "attempt_rank",
                ),
                ordinal,
            )
        )
    selected = sorted(candidates[: request.birth_attempt_count])
    output = []
    count = len(parent_ids)
    for ordinal in selected:
        first = _schedule_word(
            contract,
            policy.schedule_namespace,
            request.seed,
            request.generation,
            ordinal,
            "parent_a",
        ) % count
        offset = 1 + (
            _schedule_word(
                contract,
                policy.schedule_namespace,
                request.seed,
                request.generation,
                ordinal,
                "parent_b",
            )
            % (count - 1)
        )
        output.append(
            ScheduledOpportunity(
                opportunity_ordinal=ordinal,
                parent_ids=(parent_ids[first], parent_ids[(first + offset) % count]),
            )
        )
    return tuple(output)


def merge_ancestry(
    parent_a: Mapping[str, Any],
    parent_b: Mapping[str, Any],
    *,
    weights: tuple[float, float] = (0.5, 0.5),
) -> dict[str, float]:
    """Merge normalized ancestry using configured parent contributions."""

    def normalized(raw: Mapping[str, Any], label: str) -> dict[str, float]:
        if not raw:
            raise InheritanceCrossPathError(f"{label} ancestry must be non-empty")
        values = {
            str(key): _finite(value, label=f"{label}.{key}")
            for key, value in raw.items()
        }
        if any(value < 0.0 for value in values.values()) or sum(values.values()) <= 0:
            raise InheritanceCrossPathError(
                f"{label} ancestry must have non-negative positive mass"
            )
        total = sum(values.values())
        return {key: value / total for key, value in values.items()}

    a = normalized(parent_a, "parent_a")
    b = normalized(parent_b, "parent_b")
    weight_values = tuple(_finite(value, label="ancestry_weight") for value in weights)
    total_weight = sum(weight_values)
    if (
        len(weight_values) != 2
        or any(value < 0 for value in weight_values)
        or total_weight <= 0
    ):
        raise InheritanceCrossPathError("ancestry weights are invalid")
    normalized_weights = (
        weight_values[0] / total_weight,
        weight_values[1] / total_weight,
    )
    output: dict[str, float] = {}
    for source, weight in zip((a, b), normalized_weights):
        for key, value in source.items():
            output[key] = output.get(key, 0.0) + weight * value
    return {key: output[key] for key in sorted(output)}


def _apply_mutation(
    contract: InheritanceContract,
    child: Mapping[str, Any],
    *,
    policy: PathPolicy,
    request: PathExecutionRequest,
    opportunity_ordinal: int,
) -> tuple[dict[str, float], int, int, tuple[str, ...], str]:
    """Apply the declared post-inheritance mutation arm deterministically."""

    output = contract.canonical_package(child)
    before_sha256 = _sha256(output)
    mutation = contract.mutation_policy
    if not mutation.enabled:
        return output, 0, 0, (), before_sha256
    schedule_value = _schedule_word(
        contract,
        policy.schedule_namespace,
        request.seed,
        request.generation,
        opportunity_ordinal,
        mutation.schedule_channel,
    )
    if schedule_value % mutation.event_modulus != mutation.event_residue:
        return output, 0, mutation.draw_budget, (), before_sha256
    bounds = {spec.name: (spec.lower, spec.upper) for spec in contract.loci}
    mutated = []
    for locus_name, delta in mutation.locus_deltas:
        lower, upper = bounds[locus_name]
        original = output[locus_name]
        value = min(upper, max(lower, original + float(delta)))
        output[locus_name] = value
        if value != original:
            mutated.append(locus_name)
    return (
        contract.canonical_package(output),
        1 if mutated else 0,
        mutation.draw_budget,
        tuple(mutated),
        before_sha256,
    )


def execute_reference_path(
    contract: InheritanceContract,
    parents: Sequence[ReferenceParent],
    request: PathExecutionRequest,
    *,
    profile_id: Any = None,
    expected_registry_sha256: str | None = None,
) -> PathExecutionResult:
    """Reject execution of the retired standalone reference lifecycle path."""

    raise InheritanceCrossPathError(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "standalone reference lifecycle execution is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    policy = contract.path_policy(request.path_id)
    profile = contract.resolve_profile(profile_id)
    expected_hash = contract.registry_sha256()
    contract.validate_registry_sha256(
        expected_hash
        if expected_registry_sha256 is None
        else expected_registry_sha256
    )
    by_id = {parent.parent_id: parent for parent in parents}
    if len(by_id) != len(parents):
        raise InheritanceCrossPathError("reference parent IDs must be unique")
    schedule = deterministic_path_schedule(contract, tuple(by_id), request)
    population = request.starting_population
    successes = 0
    events: list[ChildEvent] = []

    for scheduled in schedule:
        before = population
        base = {
            "schema_version": contract.event_schema_version,
            "contract_schema_version": contract.schema_version,
            "registry_version": contract.registry_version,
            "registry_sha256": expected_hash,
            "path_id": policy.path_id,
            "schedule_namespace": policy.schedule_namespace,
            "profile_id": profile.profile_id,
            "seed": request.seed,
            "generation": request.generation,
            "opportunity_ordinal": scheduled.opportunity_ordinal,
            "attempted": True,
            "scheduled_parent_ids": scheduled.parent_ids,
            "population_before": before,
            "capacity": request.capacity,
            "capacity_accounting_version": contract.capacity_accounting_version,
            "mutation_policy": contract.mutation_policy,
        }
        if population >= request.capacity:
            events.append(
                ChildEvent(
                    **base,
                    success=False,
                    failure_reason=policy.capacity_failure_reason,
                    population_after=before,
                    mutation_event_count=0,
                    mutation_draw_count=0,
                    mutated_loci=(),
                    pre_mutation_child_sha256=None,
                    child_id=None,
                    child=None,
                    child_sha256=None,
                    canonical_parent_ids=None,
                    canonical_parent_package_sha256=None,
                    inheritance_draw_count=0,
                    donor_ordinals=(),
                    donor_parent_id_by_block=(),
                    donor_parent_id_by_locus=(),
                    ancestry=None,
                )
            )
            continue
        inheritance_rng = random.Random(
            _schedule_word(
                contract,
                policy.schedule_namespace,
                request.seed,
                request.generation,
                scheduled.opportunity_ordinal,
                "inheritance",
            )
        )
        projection = project_inheritance(
            contract,
            by_id[scheduled.parent_ids[0]],
            by_id[scheduled.parent_ids[1]],
            rng=inheritance_rng,
            profile_id=profile.profile_id,
            expected_registry_sha256=expected_hash,
        )
        child, mutation_events, mutation_draws, mutated_loci, pre_mutation_sha = (
            _apply_mutation(
                contract,
                projection.child_mapping(),
                policy=policy,
                request=request,
                opportunity_ordinal=scheduled.opportunity_ordinal,
            )
        )
        ancestry = merge_ancestry(
            by_id[scheduled.parent_ids[0]].ancestry_mapping(),
            by_id[scheduled.parent_ids[1]].ancestry_mapping(),
            weights=policy.ancestry_parent_weights,
        )
        population += 1
        child_id = request.first_child_id + successes
        successes += 1
        events.append(
            ChildEvent(
                **base,
                success=True,
                failure_reason=None,
                population_after=population,
                mutation_event_count=mutation_events,
                mutation_draw_count=mutation_draws,
                mutated_loci=mutated_loci,
                pre_mutation_child_sha256=pre_mutation_sha,
                child_id=child_id,
                child=tuple(child.items()),
                child_sha256=_sha256(child),
                canonical_parent_ids=projection.canonical_parent_ids,
                canonical_parent_package_sha256=(
                    projection.canonical_parent_package_sha256
                ),
                inheritance_draw_count=projection.draw_count,
                donor_ordinals=projection.donor_ordinals,
                donor_parent_id_by_block=projection.donor_parent_id_by_block,
                donor_parent_id_by_locus=projection.donor_parent_id_by_locus,
                ancestry=tuple(ancestry.items()),
            )
        )

    failures = len(schedule) - successes
    return PathExecutionResult(
        path_id=policy.path_id,
        profile_id=profile.profile_id,
        opportunity_count=request.opportunity_count,
        birth_attempts=len(schedule),
        non_birth_opportunities=request.opportunity_count - len(schedule),
        successes=successes,
        failures=failures,
        starting_population=request.starting_population,
        ending_population=population,
        capacity=request.capacity,
        events=tuple(events),
        registry_sha256=expected_hash,
    )


__all__ = [
    "ATTEMPT_HASH_RANKED",
    "ATTEMPT_PREFIX",
    "ChildEvent",
    "FounderGroupSpec",
    "FounderPackageSpec",
    "InheritanceContract",
    "InheritanceCrossPathError",
    "LINKED_PARENT_COPY_OPERATOR",
    "LinkageBlockSpec",
    "LocusSpec",
    "MIDPOINT_OPERATOR",
    "MUTATION_DETERMINISTIC_DELTA_OPERATOR",
    "MUTATION_OFF_OPERATOR",
    "MutationPolicy",
    "PathExecutionRequest",
    "PathExecutionResult",
    "PathPolicy",
    "ProfileSpec",
    "ProjectionResult",
    "ReferenceParent",
    "ScheduledOpportunity",
    "deterministic_path_schedule",
    "execute_reference_path",
    "merge_ancestry",
    "project_inheritance",
]
