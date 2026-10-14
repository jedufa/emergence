"""Executable C4 R2: B1 resident ordinary birth-only transaction.

The CPU ordinary-birth oracle remains the semantic authority.  This module
provides a separate, bounded CUDA implementation for the no-mutation,
no-mate R2 rung.  Planning consumes a caller-owned resident spawn tape and all
transaction scratch is fixed-capacity and reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load

from .c4_population import TensorWorldC4BirthConfigV1
from .c4_state import TensorWorldStateBundleV4
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
PERF_ROOT = ROOT / "engine" / "perf"
NATIVE_ROOT = ROOT / "engine" / "tensor_world" / "native"
NATIVE_BINDING = NATIVE_ROOT / "tensor_world_c4_resident_birth_r2_binding.cpp"
NATIVE_KERNEL = NATIVE_ROOT / "tensor_world_c4_resident_birth_r2.cu"
SOURCE_POPULATION = ROOT / "engine" / "tensor_world" / "c4_population.py"
SOURCE_EVENTS = ROOT / "engine" / "tensor_world" / "c4_events.py"
SOURCE_REPAIR = ROOT / "engine" / "tensor_world" / "c4_resident_repair.py"
SOURCE_IDENTITY = PERF_ROOT / "cuda_architecture_b_persistent_birth_identity.cuh"
SOURCE_GENOME = PERF_ROOT / "cuda_architecture_b_persistent_birth_scalar_genome.cuh"
SOURCE_POLICY = PERF_ROOT / "cuda_architecture_b_persistent_birth_policy.cuh"

C4_RESIDENT_BIRTH_R2_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_birth_r2_v1"
)
C4_RESIDENT_BIRTH_R2_STAGING_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_birth_r2_staging_v1"
)
C4_RESIDENT_BIRTH_R2_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_BIRTH_R2_STATUS_V1: Final[dict[int, str]] = {
    0: "ok",
    1: "invalid_plan_configuration",
    2: "invalid_free_queue",
    3: "invalid_parent_order_or_state",
    4: "invalid_child_reservation",
    5: "identity_capacity_exceeded",
    6: "event_capacity_exceeded",
    7: "event_semantics_invalid",
    8: "unsupported_gru_parent",
    9: "spawn_tape_invalid",
}


@dataclass(slots=True)
class TensorWorldC4ResidentBirthR2StagingV1:
    """Caller-owned resident plan and commit scratch."""

    spawn_uniforms: torch.Tensor
    parent_slots: torch.Tensor
    child_slots: torch.Tensor
    child_pos_y: torch.Tensor
    child_pos_x: torch.Tensor
    child_energy: torch.Tensor
    planned_count: torch.Tensor
    decision_parent_slots: torch.Tensor
    decision_adjacent_counts: torch.Tensor
    decision_reason_codes: torch.Tensor
    decision_count: torch.Tensor
    planned_repro_cooldown: torch.Tensor
    spawn_rng_draws_added: torch.Tensor
    spawn_rng_semantic_draws_added: torch.Tensor
    child_agent_ids: torch.Tensor
    queue_cursor: torch.Tensor
    event_sequence: torch.Tensor
    status: torch.Tensor
    schema_version: str = C4_RESIDENT_BIRTH_R2_STAGING_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.spawn_uniforms.device

    @property
    def batch(self) -> int:
        return int(self.spawn_uniforms.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.spawn_uniforms.shape[1])

    def validate(self, *, batch: int | None = None, capacity: int | None = None) -> None:
        if self.schema_version != C4_RESIDENT_BIRTH_R2_STAGING_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 R2 birth staging schema is unsupported")
        if self.spawn_uniforms.ndim != 3 or self.spawn_uniforms.shape[2] != 3:
            raise TensorWorldContractError("C4 R2 spawn tape requires shape [batch,capacity,3]")
        actual_batch, actual_capacity, _ = self.spawn_uniforms.shape
        if batch is not None and actual_batch != batch:
            raise TensorWorldContractError("C4 R2 birth staging batch mismatch")
        if capacity is not None and actual_capacity != capacity:
            raise TensorWorldContractError("C4 R2 birth staging capacity mismatch")
        if self.device.type != "cuda":
            raise TensorWorldContractError(
                "C4 R2 birth staging requires CUDA; CPU fallback is forbidden"
            )
        slots = (actual_batch, actual_capacity)
        expected = (
            ("spawn_uniforms", self.spawn_uniforms, (*slots, 3), torch.float64),
            ("parent_slots", self.parent_slots, slots, torch.int32),
            ("child_slots", self.child_slots, slots, torch.int32),
            ("child_pos_y", self.child_pos_y, slots, torch.int32),
            ("child_pos_x", self.child_pos_x, slots, torch.int32),
            ("child_energy", self.child_energy, slots, torch.float32),
            ("planned_count", self.planned_count, (actual_batch,), torch.int64),
            ("decision_parent_slots", self.decision_parent_slots, slots, torch.int64),
            ("decision_adjacent_counts", self.decision_adjacent_counts, slots, torch.int32),
            ("decision_reason_codes", self.decision_reason_codes, slots, torch.int32),
            ("decision_count", self.decision_count, (actual_batch,), torch.int64),
            ("planned_repro_cooldown", self.planned_repro_cooldown, slots, torch.int32),
            ("spawn_rng_draws_added", self.spawn_rng_draws_added, (actual_batch,), torch.int64),
            ("spawn_rng_semantic_draws_added", self.spawn_rng_semantic_draws_added, (actual_batch,), torch.int64),
            ("child_agent_ids", self.child_agent_ids, slots, torch.int64),
            ("queue_cursor", self.queue_cursor, (actual_batch,), torch.int64),
            ("event_sequence", self.event_sequence, (actual_batch,), torch.int64),
            ("status", self.status, (actual_batch,), torch.int32),
        )
        devices = {self.device}
        for name, tensor, shape, dtype in expected:
            if tuple(tensor.shape) != shape or tensor.dtype != dtype or not tensor.is_contiguous():
                raise TensorWorldContractError(
                    f"C4 R2 birth staging {name} requires contiguous "
                    f"shape={shape!r} dtype={dtype}"
                )
            devices.add(tensor.device)
        if len(devices) != 1:
            raise TensorWorldContractError("C4 R2 birth staging must share one device")


def make_c4_resident_birth_r2_staging_v1(
    *, batch: int, capacity: int, device: torch.device | str
) -> TensorWorldC4ResidentBirthR2StagingV1:
    """Allocate the fixed-capacity R2 staging envelope once."""

    target = torch.device(device)
    if type(batch) is not int or batch <= 0 or type(capacity) is not int or capacity <= 0:
        raise TensorWorldContractError("C4 R2 staging dimensions must be positive ints")
    if target.type != "cuda":
        raise TensorWorldContractError(
            "C4 R2 birth staging requires CUDA; CPU fallback is forbidden"
        )
    slots = (batch, capacity)
    staging = TensorWorldC4ResidentBirthR2StagingV1(
        spawn_uniforms=torch.zeros((*slots, 3), dtype=torch.float64, device=target),
        parent_slots=torch.full(slots, -1, dtype=torch.int32, device=target),
        child_slots=torch.full(slots, -1, dtype=torch.int32, device=target),
        child_pos_y=torch.full(slots, -1, dtype=torch.int32, device=target),
        child_pos_x=torch.full(slots, -1, dtype=torch.int32, device=target),
        child_energy=torch.zeros(slots, dtype=torch.float32, device=target),
        planned_count=torch.zeros(batch, dtype=torch.int64, device=target),
        decision_parent_slots=torch.full(slots, -1, dtype=torch.int64, device=target),
        decision_adjacent_counts=torch.zeros(slots, dtype=torch.int32, device=target),
        decision_reason_codes=torch.full(slots, -1, dtype=torch.int32, device=target),
        decision_count=torch.zeros(batch, dtype=torch.int64, device=target),
        planned_repro_cooldown=torch.zeros(slots, dtype=torch.int32, device=target),
        spawn_rng_draws_added=torch.zeros(batch, dtype=torch.int64, device=target),
        spawn_rng_semantic_draws_added=torch.zeros(batch, dtype=torch.int64, device=target),
        child_agent_ids=torch.full(slots, -1, dtype=torch.int64, device=target),
        queue_cursor=torch.zeros(batch, dtype=torch.int64, device=target),
        event_sequence=torch.zeros(batch, dtype=torch.int64, device=target),
        status=torch.zeros(batch, dtype=torch.int32, device=target),
    )
    staging.validate(batch=batch, capacity=capacity)
    return staging


_MODULE: Any | None = None


def c4_resident_birth_r2_source_receipt_v1() -> dict[str, object]:
    """Return source-bound implementation and reset-coverage information."""

    sources = {
        "adapter": Path(__file__),
        "binding": NATIVE_BINDING,
        "kernel": NATIVE_KERNEL,
        "cpu_population_oracle": SOURCE_POPULATION,
        "typed_event_oracle": SOURCE_EVENTS,
        "reset_manifest": SOURCE_REPAIR,
        "architecture_b_identity": SOURCE_IDENTITY,
        "architecture_b_genome": SOURCE_GENOME,
        "architecture_b_policy": SOURCE_POLICY,
    }
    return {
        "schema_version": C4_RESIDENT_BIRTH_R2_SCHEMA_VERSION_V1,
        "staging_schema_version": C4_RESIDENT_BIRTH_R2_STAGING_SCHEMA_VERSION_V1,
        "route": "R2_B1_CUDA_ORDINARY_BIRTH_ONLY_NO_MUTATION",
        "decision": C4_RESIDENT_BIRTH_R2_DECISION_V1,
        "resident_spawn_tape": "caller_owned_float64_[batch,capacity,3]",
        "batch_commit": "all_worlds_preflight_then_every_commit_kernel_status_guarded",
        "reset_coverage": "all_slot_scoped_current_V4_manifest_classes",
        "handling_axis_newborn": (
            "zero_is_the_neutral_axis_when_active_y_active_x_are_minus_one_and_"
            "handling_progress_is_zero"
        ),
        "jungle_uniforms_newborn": "preserve_existing_slot_value_pending_C5_handoff",
        "physical_identity_projection": "preserve_domain_world_ids_and_slot_ids",
        "mutation": "fail_closed",
        "mate_or_recombination": "fail_closed",
        "cpu_fallback": False,
        "python_world_or_agent_loops": False,
        "gate_claim": False,
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


def load_tensor_world_c4_resident_birth_r2_extension(
    *, build_directory: str | Path | None = None, verbose: bool = False
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise TensorWorldContractError("C4 R2 resident birth requires CUDA")
    digest = sha256(
        b"\0".join(path.read_bytes() for path in (NATIVE_BINDING, NATIVE_KERNEL))
    ).hexdigest()[:16]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=f"tensor_world_c4_resident_birth_r2_{digest}",
        sources=[str(NATIVE_BINDING), str(NATIVE_KERNEL)],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(PERF_ROOT)],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def _primary_tensors(bundle: TensorWorldStateBundleV4) -> list[torch.Tensor]:
    core = bundle.base.base.core
    c2 = bundle.base.base
    c3 = bundle.base
    p = bundle.population
    tape = bundle.events
    if tape is None:
        raise TensorWorldContractError("C4 R2 typed birth event boundary is required")
    return [
        core.agents.alive, core.agents.agent_id, core.agents.pos_y, core.agents.pos_x,
        core.agents.energy, core.agents.health, core.agents.age,
        core.agents.survival_steps, core.agents.energy_collected,
        core.occupancy.agent_id_grid, core.occupancy.slot_grid, c2.patch.agent_grid_i32,
        core.policy.drives, core.policy.attack_bias, core.policy.action_temperature,
        core.policy.drive_logit_gain, p.agent_id_to_slot, p.slot_tokens, p.next_agent_id,
        p.free_queue, p.free_queue_count, p.parent_slots, p.child_slots, p.child_pos_y,
        p.child_pos_x, p.child_energy, p.mate_slots, p.planned_count, p.plan_status,
        p.scar, p.repro_cooldown, p.offspring_count, p.first_repro_step,
        p.lineage_birth_generation, p.lineage_birth_tick, p.lineage_founder_id,
        p.lineage_depth, p.lineage_parent_agent_id, p.patch_habitat_lineage_id,
        p.decision_parent_slots, p.decision_mate_slots, p.decision_adjacent_counts,
        p.decision_reason_codes, p.decision_count, p.planned_repro_cooldown,
        p.birth_parent_ids, p.birth_child_ids, p.birth_count, p.mutation_occurred,
        p.mutation_delta_total, p.spawn_rng_draws_added,
        p.spawn_rng_semantic_draws_added, p.runner_rng_draws_added,
        p.runner_rng_semantic_draws_added, p.active_agent_ids, p.active_agent_slots,
        p.active_count, p.survivor_agent_ids,
        c3.agent.age, c3.agent.survival_steps, c3.agent.policy_energy_observation,
        c3.agent.drives, c3.agent.drive_inertia, c3.agent.drive_noise,
        c3.agent.repro_threshold, c3.adaptation.chemistry,
        c3.genome.reproduction_threshold, c3.genome.aggression_bias,
        c3.genome.action_temperature, c3.genome.drive_inertia, c3.genome.drive_noise,
        c3.genome.drive_logit_gain, c3.genome.drive_setpoints,
        c3.transactions.pre_energy, c3.transactions.pre_health,
        c3.transactions.pre_injury, c3.transactions.pre_survival_steps,
        tape.event_type, tape.generation, tape.tick, tape.actor_id, tape.target_id,
        tape.value, tape.typed_event_type, tape.typed_generation, tape.typed_tick,
        tape.typed_sequence, tape.typed_actor_id, tape.typed_target_id,
        tape.typed_actor_slot, tape.typed_target_slot, tape.typed_parent_id,
        tape.typed_child_id, tape.typed_founder_id, tape.typed_lineage_depth,
        tape.typed_subtype, tape.typed_reason, tape.typed_success, tape.typed_value,
        tape.active_count, tape.status,
    ]


def _inherit_tensors(bundle: TensorWorldStateBundleV4) -> list[torch.Tensor]:
    core = bundle.base.base.core
    c3 = bundle.base
    g = c3.genome
    return [
        core.policy.packed_weights, bundle.base.base.policy.packed_weights,
        g.hidden_size, g.memory_dim, g.sensor_channel_mask, g.sensor_radius,
        g.introspection_channel_mask, g.self_regulation_control_mask,
        g.self_modification_target_mask, g.directed_inheritance_channel_mask,
        g.policy_mem_dim, g.policy_use_gru, g.exploration_bias, g.aggression_bias,
        g.mutation_rate, g.reproduction_threshold, g.action_temperature,
        g.drive_inertia, g.drive_noise, g.drive_logit_gain, g.pred_lr,
        g.drive_setpoints, g.genome_drive_sp_hunger, g.genome_drive_sp_pain,
        g.genome_drive_sp_fatigue, g.genome_drive_sp_fear, g.exp_prior_strength,
        g.exp_prior_noise, g.drive_birth_noise,
    ]


def _zero_tensors(bundle: TensorWorldStateBundleV4) -> list[torch.Tensor]:
    core = bundle.base.base.core
    c2 = bundle.base.base
    a = bundle.base.agent
    ad = bundle.base.adaptation
    tx = bundle.base.transactions
    return [
        core.transactions.moved, core.transactions.gathered,
        core.transactions.food_gained, core.transactions.observations,
        core.transactions.logits, c2.eed.active_class, c2.eed.handling_progress,
        c2.eed.handling_axis, c2.effects.pre_policy_health_delta,
        c2.effects.post_action_health_delta, c2.effects.hazard_flags,
        c2.world_events.gather_amount, c2.world_events.gather_resource_class,
        c2.transactions.observations, c2.transactions.logits,
        a.injury, a.energy_collected, a.energy_collected_exact, a.rest_streak,
        a.action_switches, a.action_oscillations, a.episode0_switches,
        a.episode0_oscillations, a.attacks, a.gru_upkeep_cost,
        a.cost_charge_scalar, a.cost_charge_maturation_scalar,
        a.prev_drive_delta, a.drive_volatility, a.drive_jitter,
        a.episode0_drive_jitter_sum, a.last_energy_delta, a.last_health_delta,
        a.last_injury_delta, a.selfharm_events, a.episode0_selfharm_events,
        ad.pred_err_sum, ad.pred_err_count, ad.episode0_pred_err_sum,
        ad.episode0_pred_err_count, ad.late_pred_err_sum, ad.late_pred_err_count,
        ad.dev_stage_initialized, ad.chemistry_observation_bias,
        bundle.base.lifecycle.death_candidate, bundle.base.lifecycle.death_reason,
        bundle.base.lifecycle.maturity_factor, bundle.base.lifecycle.fertility_factor,
        bundle.base.lifecycle.senescence_factor, tx.observations, tx.logits,
        tx.chemistry_noise_samples, tx.chemistry_noise_samples_valid,
        tx.drive_noise_samples, tx.drive_noise_samples_valid,
    ]


def _minus_one_tensors(bundle: TensorWorldStateBundleV4) -> list[torch.Tensor]:
    core = bundle.base.base.core
    c2 = bundle.base.base
    a = bundle.base.agent
    runtime = bundle.base.policy_runtime
    tx = bundle.base.transactions
    return [
        core.transactions.actions, c2.eed.active_y, c2.eed.active_x,
        a.last_action, a.last_last_action, runtime.action_cache_actions,
        runtime.action_cache_token, runtime.action_cache_cursor,
        runtime.action_cache_next, tx.row_actions, tx.actions_by_slot,
    ]


def _half_tensors(bundle: TensorWorldStateBundleV4) -> list[torch.Tensor]:
    # The CPU oracle freezes action uniforms at the neutral 0.5 value. Jungle
    # uniforms belong to the unresolved C5 continuation and are preserved.
    return [bundle.base.base.core.rng.action_uniforms]


def _one_tensors(bundle: TensorWorldStateBundleV4) -> list[torch.Tensor]:
    return [bundle.base.adaptation.chemistry_observation_fidelity]


def execute_c4_resident_birth_r2_cuda_v1(
    bundle: TensorWorldStateBundleV4,
    staging: TensorWorldC4ResidentBirthR2StagingV1,
    config: TensorWorldC4BirthConfigV1,
    *,
    mate_slots: torch.Tensor | None = None,
    recombination: bool = False,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> TensorWorldC4ResidentBirthR2StagingV1:
    """Plan and commit the B1 no-mutation ordinary birth transaction."""

    if not isinstance(bundle, TensorWorldStateBundleV4) or bundle.device.type != "cuda":
        raise TensorWorldContractError("C4 R2 birth requires a CUDA V4 bundle")
    if bundle.batch != 1:
        raise TensorWorldContractError("C4 R2 is bounded to B1")
    if not isinstance(config, TensorWorldC4BirthConfigV1):
        raise TensorWorldContractError("C4 R2 birth config is required")
    if config.allow_mutation:
        raise TensorWorldContractError("C4 R2 mutation is unsupported and fail-closed")
    if mate_slots is not None or recombination:
        raise TensorWorldContractError("C4 R2 mate/recombination is unsupported and fail-closed")
    if bundle.events is None:
        raise TensorWorldContractError("C4 R2 typed birth event boundary is required")
    staging.validate(batch=1, capacity=bundle.capacity)
    if staging.device != bundle.device:
        raise TensorWorldContractError("C4 R2 staging must share the bundle device")
    max_births = bundle.capacity if config.max_births is None else config.max_births
    module = load_tensor_world_c4_resident_birth_r2_extension(
        build_directory=build_directory, verbose=verbose
    )
    module.birth_transaction(
        _primary_tensors(bundle),
        _inherit_tensors(bundle),
        _zero_tensors(bundle),
        _minus_one_tensors(bundle),
        _half_tensors(bundle),
        _one_tensors(bundle),
        [
            staging.spawn_uniforms, staging.parent_slots, staging.child_slots,
            staging.child_pos_y, staging.child_pos_x, staging.child_energy,
            staging.planned_count, staging.decision_parent_slots,
            staging.decision_adjacent_counts, staging.decision_reason_codes,
            staging.decision_count, staging.planned_repro_cooldown,
            staging.spawn_rng_draws_added, staging.spawn_rng_semantic_draws_added,
            staging.child_agent_ids, staging.queue_cursor, staging.event_sequence,
            staging.status,
        ],
        int(max_births), float(config.parent_energy_share),
        float(config.minimum_energy_buffer), int(config.cooldown_steps),
        float(config.start_health), float(config.max_energy),
        int(config.life_history_mode), int(config.maturity_start_ticks),
        int(config.maturity_rise_ticks), int(config.fertility_decline_start_ticks),
        int(config.fertility_decline_ticks), float(config.elder_fertility_floor),
        int(config.generation), int(bundle.base.tick),
    )
    return staging


__all__ = [
    "C4_RESIDENT_BIRTH_R2_DECISION_V1",
    "C4_RESIDENT_BIRTH_R2_SCHEMA_VERSION_V1",
    "C4_RESIDENT_BIRTH_R2_STAGING_SCHEMA_VERSION_V1",
    "C4_RESIDENT_BIRTH_R2_STATUS_V1",
    "TensorWorldC4ResidentBirthR2StagingV1",
    "c4_resident_birth_r2_source_receipt_v1",
    "execute_c4_resident_birth_r2_cuda_v1",
    "load_tensor_world_c4_resident_birth_r2_extension",
    "make_c4_resident_birth_r2_staging_v1",
]
