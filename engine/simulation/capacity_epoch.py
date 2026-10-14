from __future__ import annotations

"""Capacity-epoch metadata and deterministic neutral migration authority.

The manager changes only a predeclared structural envelope at a completed
generation boundary. It preserves the old subspace, runtime state, identities,
ordering, and RNG state, and it fails closed outside the CPU reference lane.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence, TypeVar

from agents.modular_genome_v1 import (
    B2_CAPACITY_PROFILES,
    b2_capacity_profile_id,
    resolve_b2_caps,
)
from evolution.capacity_contract import build_capacity_contract


CAPACITY_EPOCH_SCHEMA_VERSION = "eiel_capacity_epoch_v1"
CAPACITY_EPOCH_TRANSITION_SCHEMA_VERSION = "eiel_capacity_epoch_transition_v1"
CAPACITY_MANAGER_SCHEMA_VERSION = "eiel_substrate_capacity_manager_v1"
CAPACITY_MIGRATION_SCHEMA_VERSION = "eiel_capacity_epoch_neutral_migration_v1"
LEGACY_CAPACITY_PROFILE_ID = "b2-12x32-v1"
EXPANDED_CAPACITY_PROFILE_ID = "b2-16x48-v1"
SAFE_BOUNDARY_KIND = "completed_generation_before_next_generation_rng_draw"

MECHANICAL_EXPANSION_FIELDS: tuple[str, ...] = (
    "structural_module_slots",
    "structural_connection_slots",
    "recurrent_state_storage",
    "hidden_unit_arena",
    "developmental_rule_storage",
    "structural_lineage_metadata",
    "runtime_live_slots",
)
SEMANTIC_INTERVENTION_FIELDS: tuple[str, ...] = (
    "action_vocabulary",
    "sensor_vocabulary",
    "chemistry_species",
    "drive_vocabulary",
    "world_interaction_semantics",
    "communication_primitives",
)


class CapacityEpochState(str, Enum):
    MONITORING = "monitoring"
    WARNING = "warning"
    ELIGIBLE = "eligible"
    AUTHORIZED = "authorized"
    PAUSING_AT_SAFE_BOUNDARY = "pausing_at_safe_boundary"
    CHECKPOINTED = "checkpointed"
    MIGRATING = "migrating"
    VALIDATING = "validating"
    READY_TO_RESUME = "ready_to_resume"
    RESUMED = "resumed"
    FAILED_ROLLBACK = "failed_rollback"


_ALLOWED_TRANSITIONS: dict[CapacityEpochState, tuple[CapacityEpochState, ...]] = {
    CapacityEpochState.MONITORING: (CapacityEpochState.WARNING,),
    CapacityEpochState.WARNING: (CapacityEpochState.MONITORING, CapacityEpochState.ELIGIBLE),
    CapacityEpochState.ELIGIBLE: (CapacityEpochState.WARNING, CapacityEpochState.AUTHORIZED),
    CapacityEpochState.AUTHORIZED: (CapacityEpochState.PAUSING_AT_SAFE_BOUNDARY,),
    CapacityEpochState.PAUSING_AT_SAFE_BOUNDARY: (
        CapacityEpochState.CHECKPOINTED,
        CapacityEpochState.FAILED_ROLLBACK,
    ),
    CapacityEpochState.CHECKPOINTED: (
        CapacityEpochState.MIGRATING,
        CapacityEpochState.FAILED_ROLLBACK,
    ),
    CapacityEpochState.MIGRATING: (
        CapacityEpochState.VALIDATING,
        CapacityEpochState.FAILED_ROLLBACK,
    ),
    CapacityEpochState.VALIDATING: (
        CapacityEpochState.READY_TO_RESUME,
        CapacityEpochState.FAILED_ROLLBACK,
    ),
    CapacityEpochState.READY_TO_RESUME: (
        CapacityEpochState.RESUMED,
        CapacityEpochState.FAILED_ROLLBACK,
    ),
    CapacityEpochState.RESUMED: (CapacityEpochState.MONITORING,),
    CapacityEpochState.FAILED_ROLLBACK: (),
}


@dataclass(frozen=True)
class CapacityEpochMetadata:
    schema_version: str
    capacity_epoch_id: str
    capacity_epoch_sequence: int
    capacity_contract_hash: str
    capacity_profile_id: str
    capacity_profile_caps: dict[str, int]
    parent_capacity_epoch_id: str | None
    transition_state: str
    rng_draws_consumed_by_migration: int
    phenotype_equivalence_required: bool
    biological_identity_preserved: bool
    ancestry_preserved: bool
    ordering_preserved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_transition(source: CapacityEpochState | str, target: CapacityEpochState | str) -> None:
    src = source if isinstance(source, CapacityEpochState) else CapacityEpochState(str(source))
    dst = target if isinstance(target, CapacityEpochState) else CapacityEpochState(str(target))
    if dst not in _ALLOWED_TRANSITIONS[src]:
        raise ValueError(f"illegal capacity-epoch transition: {src.value}->{dst.value}")


def checkpoint_capacity_epoch_metadata(*, sim_cfg: Any, env_cfg: Any = None) -> dict[str, Any] | None:
    if sim_cfg is None or not bool(getattr(sim_cfg, "capacity_epoch_metadata_enable", False)):
        return None
    epoch_id = str(getattr(sim_cfg, "capacity_epoch_id", "capacity-epoch-0") or "capacity-epoch-0")
    sequence = int(getattr(sim_cfg, "capacity_epoch_sequence", 0) or 0)
    if sequence < 0:
        raise ValueError("capacity_epoch_sequence must be nonnegative")
    contract = build_capacity_contract(sim_cfg=sim_cfg, env_cfg=env_cfg, capacity_epoch_id=epoch_id)
    profile_caps = resolve_b2_caps(
        {
            "max_modules_per_agent": getattr(sim_cfg, "genome_b2_max_modules_per_agent", 12),
            "max_connections_per_agent": getattr(sim_cfg, "genome_b2_max_connections_per_agent", 32),
            "max_module_fanin": getattr(sim_cfg, "genome_b2_max_module_fanin", 6),
            "max_module_fanout": getattr(sim_cfg, "genome_b2_max_module_fanout", 6),
        }
    )
    transition_state = CapacityEpochState(
        str(getattr(sim_cfg, "capacity_epoch_transition_state", CapacityEpochState.MONITORING.value))
    )
    metadata = CapacityEpochMetadata(
        schema_version=CAPACITY_EPOCH_SCHEMA_VERSION,
        capacity_epoch_id=epoch_id,
        capacity_epoch_sequence=sequence,
        capacity_contract_hash=str(contract["contract_hash"]),
        capacity_profile_id=b2_capacity_profile_id(profile_caps),
        capacity_profile_caps=profile_caps,
        parent_capacity_epoch_id=(
            str(getattr(sim_cfg, "capacity_epoch_parent_id", "") or "") or None
        ),
        transition_state=transition_state.value,
        rng_draws_consumed_by_migration=0,
        phenotype_equivalence_required=True,
        biological_identity_preserved=True,
        ancestry_preserved=True,
        ordering_preserved=True,
    )
    return metadata.to_dict()


T = TypeVar("T")


def neutral_pad_sequence(values: Sequence[T], *, new_capacity: int, neutral_value: T) -> tuple[T, ...]:
    """Return deterministic neutral padding without RNG draws or truncation."""

    old = tuple(values)
    target = int(new_capacity)
    if target < len(old):
        raise ValueError(f"capacity migration cannot truncate: old={len(old)} new={target}")
    return old + tuple(neutral_value for _ in range(target - len(old)))


def capacity_epoch_migration_contract() -> dict[str, Any]:
    return {
        "schema_version": CAPACITY_EPOCH_TRANSITION_SCHEMA_VERSION,
        "mechanical_expansion_fields": list(MECHANICAL_EXPANSION_FIELDS),
        "semantic_intervention_fields": list(SEMANTIC_INTERVENTION_FIELDS),
        "safe_boundary": "completed_generation_before_next_generation_rng_draw",
        "trigger_evidence": [
            "sustained_headroom_warning",
            "population_or_lineage_prevalence",
            "repeated_blocked_legal_growth",
            "functional_activity_or_ablation_relevance",
            "viability",
            "anti_bloat_check",
        ],
        "tier_selection": "predeclared_smallest_compatible_profile",
        "neutral_initialization": True,
        "rng_draws": 0,
        "required_invariants": [
            "biological_identity",
            "ancestry",
            "active_values",
            "runtime_state",
            "ordering",
            "phenotype_at_boundary",
        ],
        "validation": [
            "old_envelope_replay_digest",
            "expanded_envelope_neutral_projection_digest",
            "first_post_resume_action_equivalence",
            "checkpoint_hash_and_provenance",
        ],
        "rollback": "retain_source_checkpoint_and_refuse_resume_on_any_mismatch",
        "cross_epoch_reporting": "report_epoch_local_shapes_and_join_by_provenance_not_raw_byte_equivalence",
        "cuda_interaction": "select_or_build_a_prevalidated_fixed_shape_profile_before_authorization",
        "guidance_boundary": "integrity_only_no_fitness_or_reward_authority",
    }


def validate_capacity_epoch_checkpoint_compatibility(sim: Any, payload: Mapping[str, Any]) -> None:
    """Reject epoch/profile drift before checkpoint population materialization."""

    metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    expected = metadata.get("capacity_epoch")
    sim_cfg = getattr(sim, "sim_cfg", None)
    enabled = bool(getattr(sim_cfg, "capacity_epoch_metadata_enable", False))
    if expected is None:
        if enabled:
            raise RuntimeError("capacity epoch metadata missing from checkpoint")
        return
    if not enabled:
        raise RuntimeError("capacity epoch checkpoint cannot resume with epoch metadata disabled")
    if not isinstance(expected, Mapping):
        raise RuntimeError("capacity epoch metadata malformed")
    expected_base = dict(expected)
    expected_manager = expected_base.pop("manager", None)
    actual = checkpoint_capacity_epoch_metadata(sim_cfg=sim_cfg, env_cfg=getattr(sim, "env_cfg", None))
    if expected_base != dict(actual or {}):
        raise RuntimeError(f"capacity epoch metadata/config mismatch expected={expected_base!r} actual={actual!r}")
    manager = getattr(sim, "substrate_capacity_manager", None)
    if expected_manager is None:
        if manager is not None:
            raise RuntimeError("capacity epoch manager state missing from checkpoint")
    else:
        if manager is None:
            raise RuntimeError("capacity epoch checkpoint requires enabled manager")
        restored_manager = SubstrateCapacityManager.from_checkpoint_dict(expected_manager)
        if restored_manager.state.value != str(expected_base.get("transition_state", "")):
            raise RuntimeError("capacity epoch manager/config transition mismatch")
        setattr(sim, "substrate_capacity_manager", restored_manager)
    expected_caps = resolve_b2_caps(expected.get("capacity_profile_caps"))
    population = payload.get("population", {})
    agents = population.get("agents", []) if isinstance(population, Mapping) else []
    for row in list(agents or []):
        if not isinstance(row, Mapping):
            continue
        genome = row.get("genome", {})
        if not isinstance(genome, Mapping) or resolve_b2_caps(genome.get("b2_caps", None)) != expected_caps:
            raise RuntimeError(f"capacity epoch genome profile mismatch agent_id={row.get('agent_id', -1)}")


@dataclass(frozen=True)
class CapacityPressureSample:
    generation: int
    module_cap_contacts: int
    connection_cap_contacts: int
    blocked_legal_mutations: int
    lineage_ids_near_cap: tuple[str, ...]
    active_capacity_fraction: float
    low_bloat: bool
    compression_relief_failed: bool

    def __post_init__(self) -> None:
        if int(self.generation) < 0:
            raise ValueError("capacity pressure generation must be nonnegative")
        for name in (
            "module_cap_contacts",
            "connection_cap_contacts",
            "blocked_legal_mutations",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"capacity pressure {name} must be nonnegative")
        fraction = float(self.active_capacity_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("active_capacity_fraction must be in [0,1]")
        if tuple(self.lineage_ids_near_cap) != tuple(sorted(set(self.lineage_ids_near_cap))):
            raise ValueError("lineage_ids_near_cap must be unique and sorted")


def _capacity_pressure_counter_snapshot(agents: Sequence[Any]) -> dict[str, int]:
    counters: dict[str, int] = {}
    for agent in sorted(agents, key=lambda item: int(getattr(item, "agent_id", -1))):
        genome = getattr(agent, "genome", None)
        if genome is None:
            continue
        stats = getattr(agent, "stats", None)
        lineage_id = str(
            getattr(stats, "lineage_id", "")
            or f"agent-{int(getattr(agent, 'agent_id', -1))}"
        )
        telemetry = dict(getattr(agent, "structural_live_module_telemetry", {}) or {})
        for suffix, value in (
            ("c1", getattr(genome, "c1_capacity_blocked_count", 0)),
            ("dup", telemetry.get("blocked_duplication_attempts", 0)),
            ("prune", telemetry.get("blocked_prune_attempts", 0)),
        ):
            key = f"{lineage_id}:{suffix}"
            counters[key] = max(counters.get(key, 0), int(value or 0))
    return counters


def capacity_pressure_sample_from_agents(
    agents: Sequence[Any],
    *,
    generation: int,
    previous_counters: Mapping[str, int] | None = None,
) -> CapacityPressureSample:
    """Build one deterministic, reporting-only pressure sample from live agents."""

    module_contacts = 0
    connection_contacts = 0
    blocked_legal = 0
    near_lineages: set[str] = set()
    active_fractions: list[float] = []
    near_bloat_labels: list[str] = []
    near_compression_failures: list[bool] = []
    previous = dict(previous_counters or {})
    current = _capacity_pressure_counter_snapshot(agents)
    counted_lineages: set[str] = set()
    for agent in sorted(agents, key=lambda item: int(getattr(item, "agent_id", -1))):
        genome = getattr(agent, "genome", None)
        if genome is None:
            continue
        caps = resolve_b2_caps(getattr(genome, "b2_caps", None))
        modules = list(getattr(genome, "modules", []) or [])
        connections = list(getattr(genome, "connections", []) or [])
        module_fraction = len(modules) / float(caps["max_modules_per_agent"])
        connection_fraction = len(connections) / float(caps["max_connections_per_agent"])
        module_contacts += int(len(modules) >= caps["max_modules_per_agent"])
        connection_contacts += int(len(connections) >= caps["max_connections_per_agent"])
        telemetry = dict(getattr(agent, "structural_live_module_telemetry", {}) or {})
        stats = getattr(agent, "stats", None)
        lineage_id = str(
            getattr(stats, "lineage_id", "")
            or f"agent-{int(getattr(agent, 'agent_id', -1))}"
        )
        if lineage_id not in counted_lineages:
            for suffix in ("c1", "dup"):
                key = f"{lineage_id}:{suffix}"
                blocked_legal += max(0, int(current.get(key, 0)) - int(previous.get(key, 0)))
            counted_lineages.add(lineage_id)
        if max(module_fraction, connection_fraction) < 0.80:
            continue
        near_lineages.add(lineage_id)
        executing = int(telemetry.get("executing_modules", 0) or 0)
        active_fractions.append(executing / float(caps["max_modules_per_agent"]))
        bloat = telemetry.get("bloat_classification", {})
        bloat = dict(bloat) if isinstance(bloat, Mapping) else {}
        near_bloat_labels.append(str(bloat.get("classification", "missing") or "missing"))
        near_compression_failures.append(
            int(current.get(f"{lineage_id}:prune", 0))
            > int(previous.get(f"{lineage_id}:prune", 0))
            and not list(telemetry.get("prunable_module_instance_ids", []) or [])
        )
    low_bloat = bool(near_bloat_labels) and all(
        label not in {"missing", "possible_bloat", "growth_without_activity", "pruning_pressure"}
        for label in near_bloat_labels
    )
    return CapacityPressureSample(
        generation=int(generation),
        module_cap_contacts=int(module_contacts),
        connection_cap_contacts=int(connection_contacts),
        blocked_legal_mutations=int(blocked_legal),
        lineage_ids_near_cap=tuple(sorted(near_lineages)),
        active_capacity_fraction=max(active_fractions, default=0.0),
        low_bloat=low_bloat,
        compression_relief_failed=bool(near_compression_failures) and all(near_compression_failures),
    )


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def capacity_epoch_rng_digest(snapshot: Mapping[str, Any]) -> str:
    return _canonical_digest(dict(snapshot))


def _genome_old_subspace(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(payload))
    out.pop("b2_caps", None)
    return out


@dataclass(frozen=True)
class NeutralCapacityEpochMigrator:
    """Mechanical executor; it has no pressure, guidance, or authorization role."""

    def prepare(
        self,
        *,
        agents: Sequence[Any],
        source_caps: Mapping[str, Any],
        target_caps: Mapping[str, Any],
    ) -> list[tuple[Any, Any, dict[str, Any], dict[str, Any]]]:
        source = resolve_b2_caps(dict(source_caps))
        target = resolve_b2_caps(dict(target_caps))
        staged: list[tuple[Any, Any, dict[str, Any], dict[str, Any]]] = []
        for agent in sorted(agents, key=lambda item: int(getattr(item, "agent_id", -1))):
            genome = getattr(agent, "genome", None)
            if genome is None:
                continue
            if resolve_b2_caps(getattr(genome, "b2_caps", None)) != source:
                raise RuntimeError(f"capacity source profile mismatch agent_id={getattr(agent, 'agent_id', -1)}")
            source_payload = genome.to_dict()
            target_payload = deepcopy(source_payload)
            target_payload["b2_caps"] = dict(target)
            migrated = type(genome).from_dict(target_payload)
            migrated_payload = migrated.to_dict()
            if _genome_old_subspace(source_payload) != _genome_old_subspace(migrated_payload):
                raise RuntimeError(f"capacity old-subspace mismatch agent_id={getattr(agent, 'agent_id', -1)}")
            staged.append((agent, migrated, source_payload, migrated_payload))
        return staged

    def apply(
        self,
        *,
        staged: Sequence[tuple[Any, Any, dict[str, Any], dict[str, Any]]],
        sim_cfg: Any,
        target_caps: Mapping[str, Any],
    ) -> None:
        target = resolve_b2_caps(dict(target_caps))
        for agent, migrated, _source, _target in staged:
            agent.genome = migrated
        for cap_name, field_name in (
            ("max_modules_per_agent", "genome_b2_max_modules_per_agent"),
            ("max_connections_per_agent", "genome_b2_max_connections_per_agent"),
            ("max_module_fanin", "genome_b2_max_module_fanin"),
            ("max_module_fanout", "genome_b2_max_module_fanout"),
        ):
            setattr(sim_cfg, field_name, int(target[cap_name]))
        parent_epoch = str(getattr(sim_cfg, "capacity_epoch_id", "capacity-epoch-0") or "capacity-epoch-0")
        setattr(sim_cfg, "capacity_epoch_parent_id", parent_epoch)
        setattr(sim_cfg, "capacity_epoch_sequence", int(getattr(sim_cfg, "capacity_epoch_sequence", 0) or 0) + 1)
        setattr(sim_cfg, "capacity_epoch_id", f"capacity-epoch-{int(getattr(sim_cfg, 'capacity_epoch_sequence'))}")
        setattr(sim_cfg, "capacity_epoch_metadata_enable", True)
        setattr(sim_cfg, "capacity_epoch_transition_state", CapacityEpochState.RESUMED.value)


@dataclass
class SubstrateCapacityManager:
    """Integrity-only manager for one predeclared mechanical capacity epoch."""

    state: CapacityEpochState = CapacityEpochState.MONITORING
    pressure_window: list[CapacityPressureSample] = field(default_factory=list)
    source_profile_id: str = LEGACY_CAPACITY_PROFILE_ID
    target_profile_id: str | None = None
    authorized_generation: int | None = None
    last_receipt: dict[str, Any] | None = None
    counter_snapshot: dict[str, int] = field(default_factory=dict)
    migrator: NeutralCapacityEpochMigrator = field(default_factory=NeutralCapacityEpochMigrator)

    def to_checkpoint_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CAPACITY_MANAGER_SCHEMA_VERSION,
            "state": self.state.value,
            "pressure_window": [asdict(sample) for sample in self.pressure_window],
            "source_profile_id": self.source_profile_id,
            "target_profile_id": self.target_profile_id,
            "authorized_generation": self.authorized_generation,
            "last_receipt": deepcopy(self.last_receipt),
            "counter_snapshot": {
                str(key): int(value) for key, value in sorted(self.counter_snapshot.items())
            },
        }

    @classmethod
    def from_checkpoint_dict(cls, payload: Mapping[str, Any]) -> "SubstrateCapacityManager":
        if not isinstance(payload, Mapping) or str(payload.get("schema_version", "")) != CAPACITY_MANAGER_SCHEMA_VERSION:
            raise RuntimeError("capacity manager checkpoint schema mismatch")
        samples = []
        for raw in list(payload.get("pressure_window", []) or []):
            if not isinstance(raw, Mapping):
                raise RuntimeError("capacity manager pressure window malformed")
            samples.append(
                CapacityPressureSample(
                    generation=int(raw.get("generation", -1)),
                    module_cap_contacts=int(raw.get("module_cap_contacts", -1)),
                    connection_cap_contacts=int(raw.get("connection_cap_contacts", -1)),
                    blocked_legal_mutations=int(raw.get("blocked_legal_mutations", -1)),
                    lineage_ids_near_cap=tuple(str(value) for value in raw.get("lineage_ids_near_cap", [])),
                    active_capacity_fraction=float(raw.get("active_capacity_fraction", -1.0)),
                    low_bloat=bool(raw.get("low_bloat", False)),
                    compression_relief_failed=bool(raw.get("compression_relief_failed", False)),
                )
            )
        if len(samples) > 3 or any(
            samples[index].generation >= samples[index + 1].generation
            for index in range(len(samples) - 1)
        ):
            raise RuntimeError("capacity manager pressure window order mismatch")
        last_receipt = payload.get("last_receipt")
        if last_receipt is not None and not isinstance(last_receipt, Mapping):
            raise RuntimeError("capacity manager receipt malformed")
        target = payload.get("target_profile_id")
        authorized = payload.get("authorized_generation")
        manager = cls(
            state=CapacityEpochState(str(payload.get("state", ""))),
            pressure_window=samples,
            source_profile_id=str(payload.get("source_profile_id", LEGACY_CAPACITY_PROFILE_ID)),
            target_profile_id=str(target) if target is not None else None,
            authorized_generation=int(authorized) if authorized is not None else None,
            last_receipt=deepcopy(dict(last_receipt)) if isinstance(last_receipt, Mapping) else None,
            counter_snapshot={
                str(key): int(value)
                for key, value in dict(payload.get("counter_snapshot", {}) or {}).items()
            },
        )
        if manager.source_profile_id not in B2_CAPACITY_PROFILES:
            raise RuntimeError("capacity manager source profile unsupported")
        if manager.target_profile_id is not None and manager.target_profile_id not in B2_CAPACITY_PROFILES:
            raise RuntimeError("capacity manager target profile unsupported")
        return manager

    def observe_agents(self, agents: Sequence[Any], *, generation: int) -> tuple[CapacityEpochState, CapacityPressureSample]:
        sample = capacity_pressure_sample_from_agents(
            agents,
            generation=int(generation),
            previous_counters=self.counter_snapshot,
        )
        self.counter_snapshot = _capacity_pressure_counter_snapshot(agents)
        return self.observe(sample), sample

    def observe(self, sample: CapacityPressureSample) -> CapacityEpochState:
        if self.state not in {
            CapacityEpochState.MONITORING,
            CapacityEpochState.WARNING,
            CapacityEpochState.ELIGIBLE,
        }:
            raise RuntimeError(f"capacity manager cannot observe in state={self.state.value}")
        if self.pressure_window and int(sample.generation) <= int(self.pressure_window[-1].generation):
            raise ValueError("capacity pressure generations must be strictly increasing")
        self.pressure_window.append(sample)
        self.pressure_window = self.pressure_window[-3:]
        touches = sum(
            int(row.module_cap_contacts) + int(row.connection_cap_contacts)
            for row in self.pressure_window
        )
        blocked = sum(int(row.blocked_legal_mutations) for row in self.pressure_window)
        lineage_ids = {
            lineage_id
            for row in self.pressure_window
            for lineage_id in row.lineage_ids_near_cap
        }
        sustained = bool(
            len(self.pressure_window) == 3
            and touches >= 3
            and blocked >= 2
            and len(lineage_ids) >= 2
            and all(float(row.active_capacity_fraction) >= 0.80 for row in self.pressure_window)
            and all(bool(row.low_bloat) for row in self.pressure_window)
            and all(bool(row.compression_relief_failed) for row in self.pressure_window)
        )
        if sustained:
            if self.state == CapacityEpochState.MONITORING:
                validate_transition(self.state, CapacityEpochState.WARNING)
                self.state = CapacityEpochState.WARNING
            if self.state == CapacityEpochState.WARNING:
                validate_transition(self.state, CapacityEpochState.ELIGIBLE)
                self.state = CapacityEpochState.ELIGIBLE
        else:
            if self.state == CapacityEpochState.ELIGIBLE:
                validate_transition(self.state, CapacityEpochState.WARNING)
                self.state = CapacityEpochState.WARNING
            if touches > 0 and self.state == CapacityEpochState.MONITORING:
                validate_transition(self.state, CapacityEpochState.WARNING)
                self.state = CapacityEpochState.WARNING
            elif touches == 0 and self.state == CapacityEpochState.WARNING:
                validate_transition(self.state, CapacityEpochState.MONITORING)
                self.state = CapacityEpochState.MONITORING
        return self.state

    def authorize(self, *, target_profile_id: str, generation: int) -> None:
        if self.state != CapacityEpochState.ELIGIBLE:
            raise RuntimeError("capacity migration is not eligible")
        latest_generation = int(self.pressure_window[-1].generation) if self.pressure_window else -1
        if int(generation) != latest_generation:
            raise RuntimeError(
                f"capacity authorization generation mismatch latest={latest_generation} requested={int(generation)}"
            )
        if target_profile_id != EXPANDED_CAPACITY_PROFILE_ID:
            raise ValueError(f"unsupported capacity target profile={target_profile_id}")
        validate_transition(self.state, CapacityEpochState.AUTHORIZED)
        self.state = CapacityEpochState.AUTHORIZED
        self.target_profile_id = str(target_profile_id)
        self.authorized_generation = int(generation)

    def migrate_at_safe_boundary(
        self,
        *,
        agents: Sequence[Any],
        sim_cfg: Any,
        next_generation: int,
        boundary_kind: str,
        write_rollback_checkpoint: Callable[[], str],
        verify_rollback_checkpoint: Callable[[str], bool],
        rng_digest_before: str,
        rng_digest_after: Callable[[], str],
    ) -> dict[str, Any]:
        if self.state != CapacityEpochState.AUTHORIZED:
            raise RuntimeError("capacity migration is not authorized")
        if str(boundary_kind) != SAFE_BOUNDARY_KIND:
            raise RuntimeError(f"capacity migration unsafe boundary={boundary_kind}")
        if self.authorized_generation is None or int(next_generation) != int(self.authorized_generation) + 1:
            raise RuntimeError(
                "capacity migration generation mismatch "
                f"authorized={self.authorized_generation} next={int(next_generation)}"
            )
        if str(getattr(sim_cfg, "device", "cpu") or "cpu").lower() != "cpu":
            raise RuntimeError("capacity migration CPU reference only")
        prohibited = (
            "native_runtime_enable",
            "runnercore_native_enable",
            "cxxrewrite_native_tick_authoritative_enable",
            "cxxrewrite_resident_generation_runner_enable",
            "cxxwave1g_full_cpp_backbone_enable",
        )
        active = [name for name in prohibited if bool(getattr(sim_cfg, name, False))]
        if active:
            raise RuntimeError(f"capacity migration native lane unsupported flags={','.join(active)}")

        source_caps = resolve_b2_caps(B2_CAPACITY_PROFILES[self.source_profile_id])
        configured_caps = resolve_b2_caps(
            {
                "max_modules_per_agent": getattr(sim_cfg, "genome_b2_max_modules_per_agent", 12),
                "max_connections_per_agent": getattr(sim_cfg, "genome_b2_max_connections_per_agent", 32),
                "max_module_fanin": getattr(sim_cfg, "genome_b2_max_module_fanin", 6),
                "max_module_fanout": getattr(sim_cfg, "genome_b2_max_module_fanout", 6),
            }
        )
        if configured_caps != source_caps:
            raise RuntimeError(
                f"capacity source config profile mismatch expected={source_caps!r} actual={configured_caps!r}"
            )

        rollback_path = str(write_rollback_checkpoint())
        if not rollback_path or not bool(verify_rollback_checkpoint(rollback_path)):
            self.state = CapacityEpochState.FAILED_ROLLBACK
            raise RuntimeError("capacity rollback checkpoint verification failed")
        validate_transition(self.state, CapacityEpochState.PAUSING_AT_SAFE_BOUNDARY)
        self.state = CapacityEpochState.PAUSING_AT_SAFE_BOUNDARY
        validate_transition(self.state, CapacityEpochState.CHECKPOINTED)
        self.state = CapacityEpochState.CHECKPOINTED
        validate_transition(self.state, CapacityEpochState.MIGRATING)
        self.state = CapacityEpochState.MIGRATING

        target_caps = resolve_b2_caps(B2_CAPACITY_PROFILES[str(self.target_profile_id)])
        try:
            staged = self.migrator.prepare(
                agents=agents,
                source_caps=source_caps,
                target_caps=target_caps,
            )

            validate_transition(self.state, CapacityEpochState.VALIDATING)
            self.state = CapacityEpochState.VALIDATING
            self.migrator.apply(staged=staged, sim_cfg=sim_cfg, target_caps=target_caps)
            if str(rng_digest_after()) != str(rng_digest_before):
                raise RuntimeError("capacity migration consumed RNG")
            receipt = {
                "schema_version": CAPACITY_MIGRATION_SCHEMA_VERSION,
                "manager_schema_version": CAPACITY_MANAGER_SCHEMA_VERSION,
                "source_profile_id": self.source_profile_id,
                "target_profile_id": self.target_profile_id,
                "source_caps": source_caps,
                "target_caps": target_caps,
                "next_generation": int(next_generation),
                "safe_boundary": SAFE_BOUNDARY_KIND,
                "rollback_checkpoint_path": rollback_path,
                "migrated_agent_count": len(staged),
                "neutral_unused_module_slots": list(
                    range(source_caps["max_modules_per_agent"], target_caps["max_modules_per_agent"])
                ),
                "neutral_unused_connection_slots": list(
                    range(source_caps["max_connections_per_agent"], target_caps["max_connections_per_agent"])
                ),
                "migration_rng_draws": 0,
                "old_subspace_digest": _canonical_digest(
                    [_genome_old_subspace(source) for _agent, _migrated, source, _target in staged]
                ),
                "identity_preserved": True,
                "homology_preserved": True,
                "topology_preserved": True,
                "runtime_state_preserved": True,
                "ordering_preserved": True,
                "reward_fitness_selection_authority": False,
            }
            validate_transition(self.state, CapacityEpochState.READY_TO_RESUME)
            self.state = CapacityEpochState.READY_TO_RESUME
            validate_transition(self.state, CapacityEpochState.RESUMED)
            self.state = CapacityEpochState.RESUMED
            self.last_receipt = dict(receipt)
            return receipt
        except Exception:
            self.state = CapacityEpochState.FAILED_ROLLBACK
            raise
