"""Experimental C3 backend layered over the resident C2 CUDA route.

The C3 path is intentionally separate from C2 admission.  It adds resident
compiled genome parameters, active-mechanics state, and eligibility outputs;
it does not settle C4 population transactions or claim a C3 gate decision.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import secrets
from typing import Any, Mapping

import torch

from .c2_backend import (
    TensorWorldC2ConfigV1,
    _config_tensors_v1,
    _native_state_v1,
    clone_c2_bundle_to_device_v1,
)
from .c2_native_loader import load_tensor_world_c2_extension
from .c3_mechanics import TensorWorldC3MechanicsConfigV1
from .c3_state import TensorWorldStateBundleV3
from .contracts import (
    BackendCapabilitiesV1,
    BackendRunStateV1,
    TensorWorldContractError,
    TickReceiptV1,
    TickRequestV1,
)


@dataclass(frozen=True, slots=True)
class TensorWorldC3ConfigV1:
    c2: TensorWorldC2ConfigV1 = TensorWorldC2ConfigV1()
    mechanics: TensorWorldC3MechanicsConfigV1 = TensorWorldC3MechanicsConfigV1()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TensorWorldC3ConfigV1":
        if not value:
            return cls()
        unknown = set(value) - {"c2", "mechanics"}
        if unknown:
            raise TensorWorldContractError(
                f"unknown C3 configuration fields: {sorted(unknown)}"
            )
        c2_value = value.get("c2", {})
        mechanics_value = value.get("mechanics", {})
        if isinstance(c2_value, TensorWorldC2ConfigV1):
            c2 = c2_value
        elif isinstance(c2_value, Mapping):
            c2 = TensorWorldC2ConfigV1.from_mapping(c2_value)
        else:
            raise TensorWorldContractError("C3 c2 configuration must be a mapping")
        if isinstance(mechanics_value, TensorWorldC3MechanicsConfigV1):
            mechanics = mechanics_value
        elif isinstance(mechanics_value, Mapping):
            mechanics = TensorWorldC3MechanicsConfigV1(**mechanics_value)
        else:
            raise TensorWorldContractError(
                "C3 mechanics configuration must be a mapping"
            )
        if c2.max_energy != mechanics.max_energy or c2.start_health != mechanics.start_health:
            raise TensorWorldContractError(
                "C3 and C2 normalization maxima must agree"
            )
        return cls(c2=c2, mechanics=mechanics)


def _copy_owner(owner: Any, device: torch.device) -> Any:
    return type(owner)(
        *(
            getattr(owner, field.name)
            .detach()
            .to(device=device, copy=True)
            .contiguous()
            for field in fields(owner)
        )
    )


def clone_c3_bundle_to_device_v1(
    source: TensorWorldStateBundleV3, device: torch.device
) -> TensorWorldStateBundleV3:
    source.validate()
    result = TensorWorldStateBundleV3(
        base=clone_c2_bundle_to_device_v1(source.base, device),
        agent=_copy_owner(source.agent, device),
        genome=_copy_owner(source.genome, device),
        adaptation=_copy_owner(source.adaptation, device),
        policy_runtime=_copy_owner(source.policy_runtime, device),
        lifecycle=_copy_owner(source.lifecycle, device),
        transactions=_copy_owner(source.transactions, device),
    )
    result.validate()
    return result


def _sync_compiled_policy_v1(state: TensorWorldStateBundleV3) -> None:
    """Copy the resident C3 execution view into the inherited C2 policy ABI."""

    # These resident values are the C3 compiled-genome boundary.  Keeping the
    # execution copies synchronized here makes policy/mechanics invalidation
    # explicit instead of letting stale fixture defaults survive a genome
    # update.
    state.agent.drive_inertia.copy_(state.genome.drive_inertia)
    state.agent.drive_noise.copy_(state.genome.drive_noise)
    state.agent.repro_threshold.copy_(state.genome.reproduction_threshold)
    state.base.core.policy.drives.copy_(state.agent.drives)
    state.base.core.policy.attack_bias.copy_(state.genome.aggression_bias)
    state.base.core.policy.action_temperature.copy_(state.genome.action_temperature)
    state.base.core.policy.drive_logit_gain.copy_(state.genome.drive_logit_gain)


def _validate_c3_noise_samples_v1(
    state: TensorWorldStateBundleV3,
    config: TensorWorldC3MechanicsConfigV1,
) -> None:
    """Fail closed before CUDA graph capture when a stochastic row lacks draws."""

    core = state.base.core
    live = core.agents.alive & (core.agents.agent_id >= 0)
    drive_required = live & (state.agent.drive_noise > 0.0)
    if bool(torch.any(drive_required & ~state.transactions.drive_noise_samples_valid).item()):
        raise TensorWorldContractError(
            "C3 drive mechanics require receipt-bound resident noise samples"
        )
    if bool(torch.any(drive_required).item()) and not bool(
        torch.isfinite(state.transactions.drive_noise_samples[drive_required]).all().item()
    ):
        raise TensorWorldContractError("C3 drive noise samples must be finite")
    chemistry_required = live if config.chemistry_noise_required else torch.zeros_like(live)
    if config.enable_chemistry and bool(
        torch.any(chemistry_required & ~state.transactions.chemistry_noise_samples_valid).item()
    ):
        raise TensorWorldContractError(
            "C3 chemistry requires receipt-bound resident noise samples"
        )
    if config.enable_chemistry and bool(torch.any(chemistry_required).item()) and not bool(
        torch.isfinite(state.transactions.chemistry_noise_samples[chemistry_required]).all().item()
    ):
        raise TensorWorldContractError("C3 chemistry noise samples must be finite")


def _validate_c3_active_mechanics_state_v1(
    state: TensorWorldStateBundleV3,
) -> None:
    """Validate the source active-mechanics pre-write contract.

    The source CUDA helper validates every assigned row before its first
    authoritative write.  Keep the same fail-closed boundary in the Python
    admission path so the resident vector implementation cannot silently
    normalize malformed controls that the source route would reject.
    """

    core = state.base.core
    agent = state.agent
    assigned = core.agents.agent_id >= 0
    if not bool(torch.equal(core.agents.alive, assigned)):
        raise TensorWorldContractError(
            "C3 active mechanics requires alive and identity partitions to match"
        )
    live = assigned

    finite_fields = (
        ("energy", core.agents.energy),
        ("health", core.agents.health),
        ("injury", agent.injury),
        ("pre_energy", state.transactions.pre_energy),
        ("pre_health", state.transactions.pre_health),
        ("pre_injury", state.transactions.pre_injury),
        ("gru_upkeep_cost", agent.gru_upkeep_cost),
        ("cost_charge_scalar", agent.cost_charge_scalar),
        ("cost_charge_maturation_scalar", agent.cost_charge_maturation_scalar),
        ("drive_volatility", agent.drive_volatility),
        ("drive_jitter", agent.drive_jitter),
        ("episode0_drive_jitter_sum", agent.episode0_drive_jitter_sum),
        ("last_energy_delta", agent.last_energy_delta),
        ("last_health_delta", agent.last_health_delta),
        ("last_injury_delta", agent.last_injury_delta),
    )
    for name, tensor in finite_fields:
        if bool(torch.any(~torch.isfinite(tensor[live])).item()):
            raise TensorWorldContractError(
                f"C3 active mechanics live {name} is nonfinite"
            )
    for name, tensor in (
        ("drives", agent.drives),
        ("prev_drive_delta", agent.prev_drive_delta),
    ):
        if bool(torch.any(~torch.isfinite(tensor[live])).item()):
            raise TensorWorldContractError(
                f"C3 active mechanics live {name} is nonfinite"
            )
    for name, tensor in (
        ("drive_inertia", agent.drive_inertia),
        ("drive_noise", agent.drive_noise),
    ):
        values = tensor[live]
        if bool(torch.any(~torch.isfinite(values)).item()) or bool(
            torch.any(values < 0.0).item()
        ):
            raise TensorWorldContractError(
                f"C3 active mechanics {name} must be finite and nonnegative"
            )
    int32_max = torch.iinfo(torch.int32).max
    for name, tensor in (
        ("rest_streak", agent.rest_streak),
        ("survival_steps", agent.survival_steps),
        ("selfharm_events", agent.selfharm_events),
        ("episode0_selfharm_events", agent.episode0_selfharm_events),
    ):
        if bool(torch.any(tensor[live] >= int32_max).item()):
            raise TensorWorldContractError(
                f"C3 active mechanics {name} counter overflow"
            )


def _chemistry_update_cuda_v1(
    state: TensorWorldStateBundleV3,
    config: TensorWorldC3MechanicsConfigV1,
) -> None:
    """Vectorized source-current chemistry update with an explicit C5 draw boundary."""

    if not config.enable_chemistry:
        return
    core = state.base.core
    agent = state.agent
    adaptation = state.adaptation
    tx = state.transactions
    live = core.agents.alive & (core.agents.agent_id >= 0)
    max_energy = float(config.max_energy)
    start_health = float(config.start_health)
    energy_n = (core.agents.energy.to(torch.float64) / max_energy).clamp(0.0, 1.0)
    health_n = (core.agents.health.to(torch.float64) / start_health).clamp(0.0, 1.0)
    injury_n = (agent.injury.to(torch.float64) / start_health).clamp(0.0, 1.0)
    stress = (
        0.55 * injury_n
        + 0.30 * agent.drives[..., 1].to(torch.float64)
        + 0.15 * (1.0 - energy_n)
    ).clamp(0.0, 1.0)
    current = adaptation.chemistry.to(torch.float64)
    energy_store, repair, acute, chronic, reg1, reg2 = current.unbind(dim=-1)
    acute_target = (1.25 * injury_n + 0.45 * stress + 0.20 * (1.0 - health_n)).clamp(0.0, 1.0)
    acute = (acute + 0.50 * (acute_target - acute) + 0.03 * reg1).clamp(0.0, 1.0)
    chronic_target = (0.75 * chronic + 0.25 * acute).clamp(0.0, 1.0)
    chronic = (chronic + 0.07 * (chronic_target - chronic)).clamp(0.0, 1.0)
    energy_store = (energy_store + 0.03 * energy_n - 0.02 * stress - 0.01 * acute).clamp(0.0, 1.0)
    repair = (repair + 0.04 * energy_n - 0.03 * acute - 0.01 * chronic).clamp(0.0, 1.0)
    if config.chemistry_noise_required:
        noise = tx.chemistry_noise_samples.to(torch.float64)
    else:
        noise = torch.zeros_like(tx.chemistry_noise_samples, dtype=torch.float64)
    reg1 = (0.92 * reg1 + 0.08 * (1.0 - acute) + 0.01 * noise[..., 0]).clamp(0.0, 1.0)
    reg2 = (0.97 * reg2 + 0.03 * chronic + 0.01 * noise[..., 1]).clamp(0.0, 1.0)
    acute_resolve = torch.minimum(acute, 0.10 * repair + 0.04 * energy_store)
    acute = (acute - acute_resolve).clamp(0.0, 1.0)
    chronic = (chronic + 0.25 * acute - 0.002 * repair).clamp(0.0, 1.0)
    updated = torch.stack((energy_store, repair, acute, chronic, reg1, reg2), dim=-1).to(torch.float32)
    adaptation.chemistry.copy_(torch.where(live.unsqueeze(-1), updated, adaptation.chemistry))
    tx.hot_counters[:, 5].copy_(live.sum(dim=1, dtype=torch.int64) * 2)


def _active_mechanics_cuda_v1(
    state: TensorWorldStateBundleV3,
    config: TensorWorldC3MechanicsConfigV1,
) -> None:
    """Vectorized resident active-mechanics phase for graph capture.

    This is an engineering implementation, not a parity receipt.  The scalar
    source transcription in :mod:`c3_mechanics` remains the comparator oracle.
    """

    core = state.base.core
    agent = state.agent
    tx = state.transactions
    lifecycle = state.lifecycle

    # Keep the explicit C3 source fields aligned with the inherited C2
    # transport state before the resident mechanics writes begin.
    agent.age.copy_(core.agents.age.to(torch.int32))
    agent.survival_steps.copy_(core.agents.survival_steps.to(torch.int32))
    agent.energy_collected.copy_(core.agents.energy_collected.to(torch.float32))
    agent.energy_collected_exact.copy_(core.agents.energy_collected)
    alive = core.agents.alive
    assigned = core.agents.agent_id >= 0
    live = alive & assigned
    tx.observations.copy_(state.base.transactions.observations)
    tx.logits.copy_(state.base.transactions.logits)
    tx.row_actions.copy_(core.transactions.actions)
    tx.actions_by_slot.copy_(core.transactions.actions)

    raw_action = tx.actions_by_slot
    action = torch.where(
        (raw_action < 0) | (raw_action > 6),
        torch.full_like(raw_action, 5),
        raw_action,
    )

    # The source fixed-MLP policy service owns a two-entry cache keyed by the
    # stable slot token.  The inherited C2 policy phase has selected the
    # action; retain it in both source-shaped entries without inventing a
    # per-tick token.  C4 invalidates the token when it settles a slot.
    cache_actions = state.policy_runtime.action_cache_actions
    cache_actions.copy_(torch.where(
        live.unsqueeze(-1),
        action.unsqueeze(-1).expand_as(cache_actions),
        cache_actions,
    ))
    slot_tokens = state.base.core.identity.slot_ids
    state.policy_runtime.action_cache_token.copy_(torch.where(
        live,
        slot_tokens,
        state.policy_runtime.action_cache_token,
    ))
    state.policy_runtime.action_cache_cursor.copy_(torch.where(
        live,
        torch.ones_like(state.policy_runtime.action_cache_cursor),
        state.policy_runtime.action_cache_cursor,
    ))
    state.policy_runtime.action_cache_next.copy_(torch.where(
        live,
        action,
        state.policy_runtime.action_cache_next,
    ))
    # Attack count is a resident C3 metric, independent of whether the
    # target/injury branch later produces self-harm.
    agent.attacks.add_((live & (action == 6)).to(torch.int32))

    # The source route records action-history counters immediately after
    # action selection and before active mechanics.  Keep this resident and
    # vectorized so the fixed-MLP target does not silently lose those C3
    # outputs while C4 remains the owner of population transactions.
    old_last = agent.last_action
    old_last_last = agent.last_last_action
    switched = live & (old_last != -1) & (action != old_last)
    oscillated = live & (old_last_last != -1) & (action == old_last_last) & (action != old_last)
    agent.action_switches.add_(switched.to(torch.int32))
    agent.action_oscillations.add_(oscillated.to(torch.int32))
    early = core.agents.survival_steps < config.episode0_k
    agent.episode0_switches.add_((switched & early).to(torch.int32))
    agent.episode0_oscillations.add_((oscillated & early).to(torch.int32))
    agent.last_last_action.copy_(torch.where(live, old_last, old_last_last))
    agent.last_action.copy_(torch.where(live, action, old_last))
    rest = action == 5
    if config.enable_rest_charge:
        agent.rest_streak.copy_(torch.where(
            live,
            torch.where(rest, agent.rest_streak + 1, torch.zeros_like(agent.rest_streak)),
            agent.rest_streak,
        ))
        over = (agent.rest_streak - config.rest_streak_free).clamp_min(0)
        charge = config.rest_charge_energy + config.rest_charge_growth * (over - 1).clamp_min(0)
        charge_mask = (agent.rest_streak > config.rest_streak_free) & live
        charge = torch.where(charge_mask, charge, torch.zeros_like(charge))
        charged_energy = core.agents.energy - charge
        deficit = (-charged_energy).clamp_min(0.0) * config.rest_deficit_to_health
        core.agents.health.sub_(torch.where(charge_mask, deficit, torch.zeros_like(deficit))).clamp_min_(0.0)
        core.agents.energy.copy_(torch.where(charge_mask, charged_energy.clamp_min(0.0), core.agents.energy))
        agent.policy_energy_observation.copy_(torch.where(
            charge_mask,
            core.agents.energy.to(torch.float64),
            agent.policy_energy_observation,
        ))

    terminal = live & ((core.agents.energy <= 0.0) | (core.agents.health <= 0.0))
    active = live & ~terminal
    core.agents.energy.sub_(torch.where(
        active,
        agent.gru_upkeep_cost.clamp_min(0.0),
        torch.zeros_like(agent.gru_upkeep_cost),
    )).clamp_min_(0.0)
    charge = agent.cost_charge_scalar.clamp_min(0.0)
    if config.developmental_maturity_age > 0:
        charge = charge + torch.where(
            core.agents.survival_steps < config.developmental_maturity_age,
            agent.cost_charge_maturation_scalar.clamp_min(0.0),
            torch.zeros_like(charge),
        )
    core.agents.energy.sub_(torch.where(active, charge, torch.zeros_like(charge))).clamp_min_(0.0)

    # The source lifecycle slice uses the pre-action survival value for
    # senescence/maturity calculations; the resident counter is advanced
    # before the drive update, so use the captured transaction snapshot here.
    age = tx.pre_survival_steps
    if config.life_history_mode == 1:
        if config.life_history_senescence_ramp_ticks <= 0:
            senescence = (age >= config.life_history_senescence_start_ticks).to(torch.float32)
        else:
            senescence = (
                (age - config.life_history_senescence_start_ticks + 1).to(torch.float32)
                / float(config.life_history_senescence_ramp_ticks)
            ).clamp(0.0, 1.0)
        senescence = torch.where(age >= config.life_history_senescence_start_ticks, senescence, torch.zeros_like(senescence))
        core.agents.energy.sub_(torch.where(active, senescence * config.life_history_senescence_energy_cost, torch.zeros_like(senescence))).clamp_min_(0.0)
        core.agents.health.sub_(torch.where(active, senescence * config.life_history_senescence_health_cost, torch.zeros_like(senescence)))
        if config.life_history_max_age_ticks > 0:
            core.agents.health.masked_fill_(active & (age >= config.life_history_max_age_ticks), 0.0)
    else:
        senescence = torch.zeros_like(core.agents.energy)

    core.agents.survival_steps.add_(active.to(core.agents.survival_steps.dtype))
    # Age is a separate source lifecycle counter.  C3 advances it at the
    # same active-row boundary as survival_steps; C4 birth settlement owns
    # resetting it for a newly assigned slot.
    core.agents.age.add_(active.to(core.agents.age.dtype))
    # Keep the explicit C3 execution projection source-ordered with the
    # inherited transport fields after the active counter advances.  C4/C5
    # consume this resident view directly; they must not infer it from core.
    agent.survival_steps.copy_(core.agents.survival_steps.to(torch.int32))
    agent.age.copy_(core.agents.age.to(torch.int32))
    agent.energy_collected.copy_(core.agents.energy_collected.to(torch.float32))
    agent.energy_collected_exact.copy_(core.agents.energy_collected)
    agent.policy_energy_observation.copy_(torch.where(
        active,
        core.agents.energy.to(torch.float64),
        agent.policy_energy_observation,
    ))
    lifecycle.senescence_factor.copy_(torch.where(active, senescence, torch.zeros_like(senescence)))

    if config.life_history_mode == 1:
        maturity = torch.where(
            age < config.life_history_maturity_start_ticks,
            torch.zeros_like(age, dtype=torch.float32),
            torch.ones_like(age, dtype=torch.float32),
        )
        if config.life_history_maturity_rise_ticks > 0:
            maturity = torch.where(
                age >= config.life_history_maturity_start_ticks,
                ((age - config.life_history_maturity_start_ticks + 1).to(torch.float32) / float(config.life_history_maturity_rise_ticks)).clamp(0.0, 1.0),
                torch.zeros_like(maturity),
            )
        fertility = maturity.clone()
        if config.life_history_fertility_decline_start_ticks > 0:
            if config.life_history_fertility_decline_ticks <= 0:
                aged = torch.full_like(fertility, config.life_history_elder_fertility_floor)
            else:
                aged = 1.0 - ((age - config.life_history_fertility_decline_start_ticks + 1).to(torch.float32) / float(config.life_history_fertility_decline_ticks)).clamp(0.0, 1.0) * (1.0 - config.life_history_elder_fertility_floor)
            fertility = torch.minimum(fertility, torch.where(age >= config.life_history_fertility_decline_start_ticks, aged, torch.ones_like(aged)))
        fertility.clamp_(0.0, 1.0)
    else:
        maturity = torch.ones_like(core.agents.energy)
        fertility = torch.ones_like(core.agents.energy)
    lifecycle.maturity_factor.copy_(torch.where(active, maturity, torch.zeros_like(maturity)))
    lifecycle.fertility_factor.copy_(torch.where(active, fertility, torch.zeros_like(fertility)))

    de = core.agents.energy - tx.pre_energy
    dh = core.agents.health - tx.pre_health
    dinj = agent.injury - tx.pre_injury
    agent.last_energy_delta.copy_(torch.where(active, de, agent.last_energy_delta))
    agent.last_health_delta.copy_(torch.where(active, dh, agent.last_health_delta))
    agent.last_injury_delta.copy_(torch.where(active, dinj, agent.last_injury_delta))
    energy_norm = (core.agents.energy / config.max_energy).clamp(0.0, 1.0)
    health_norm = (core.agents.health / config.start_health).clamp(0.0, 1.0)
    previous = agent.drives
    hunger = previous[..., 0] + config.drive_hunger_gain * (1.0 - energy_norm) + 0.04 * (-de / config.max_energy) - config.drive_hunger_decay
    hunger = torch.where(action == 4, hunger - 0.20 * (de / config.max_energy).clamp_min(0.0), hunger).clamp(0.0, 1.0)
    pain = previous[..., 1] + torch.where(dinj > 0.0, config.drive_pain_gain * dinj.clamp_max(1.0), torch.zeros_like(dinj))
    pain = pain + torch.where(dh < 0.0, 0.10 * (-dh / config.start_health).clamp_max(1.0), torch.zeros_like(dh))
    pain = (pain - config.drive_pain_decay).clamp(0.0, 1.0)
    fatigue = torch.where(action == 5, previous[..., 2] - config.drive_fatigue_rest_relief, previous[..., 2] + config.drive_fatigue_gain)
    fatigue = (fatigue - config.drive_fatigue_decay).clamp(0.0, 1.0)
    fear = previous[..., 3] + torch.where(dinj > 0.0, config.drive_fear_gain * dinj.clamp_max(1.0), torch.zeros_like(dinj))
    fear = (fear + 0.05 * (1.0 - health_norm) - config.drive_fear_decay).clamp(0.0, 1.0)
    targets = torch.stack((hunger, pain, fatigue, fear), dim=-1)
    inertia = agent.drive_inertia.clamp(0.0, 0.999).unsqueeze(-1)
    next_drives = (inertia * previous + (1.0 - inertia) * targets).clamp(0.0, 1.0)
    noise_scale = agent.drive_noise.clamp_min(0.0).unsqueeze(-1)
    # Zero-valued noise is the C3 fixture default.  C5 will bind the resident
    # sample validity/cursor contract before admitting nonzero noise.
    next_drives = (next_drives + noise_scale * tx.drive_noise_samples).clamp(0.0, 1.0)
    deltas = next_drives - previous
    # The source sums absolute drive terms in double precision and casts the
    # result back to f32.  Preserve that accumulation order at the ABI
    # boundary instead of relying on a f32 reduction's rounding.
    volatility = deltas.abs().to(torch.float64).sum(dim=-1).to(torch.float32)
    jitter = (deltas - agent.prev_drive_delta).abs().to(torch.float64).sum(dim=-1).to(torch.float32)
    agent.drive_volatility.add_(torch.where(active, volatility, torch.zeros_like(volatility)))
    agent.drive_jitter.add_(torch.where(active, jitter, torch.zeros_like(jitter)))
    agent.episode0_drive_jitter_sum.add_(torch.where(active & (core.agents.survival_steps <= config.episode0_k), jitter, torch.zeros_like(jitter)))
    agent.prev_drive_delta.copy_(torch.where(active.unsqueeze(-1), deltas, agent.prev_drive_delta))
    agent.drives.copy_(torch.where(active.unsqueeze(-1), next_drives, previous))
    agent.selfharm_events.add_((active & (action == 6) & (dinj > 0.0)).to(torch.int32))
    agent.episode0_selfharm_events.add_((active & (action == 6) & (dinj > 0.0) & (core.agents.survival_steps <= config.episode0_k)).to(torch.int32))

    lifecycle.death_candidate.copy_(live & ((core.agents.energy <= 0.0) | (core.agents.health <= 0.0)))
    reasons = torch.where(
        core.agents.health <= 0.0,
        torch.full_like(lifecycle.death_reason, 2),
        torch.where(
            core.agents.energy <= 0.0,
            torch.ones_like(lifecycle.death_reason),
            torch.zeros_like(lifecycle.death_reason),
        ),
    )
    lifecycle.death_reason.copy_(torch.where(
        live,
        reasons,
        torch.zeros_like(lifecycle.death_reason),
    ))
    tx.hot_counters[:, :5].zero_()
    tx.hot_counters[:, 0].copy_(active.sum(dim=1, dtype=torch.int64))
    tx.hot_counters[:, 1].copy_(active.sum(dim=1, dtype=torch.int64))
    tx.hot_counters[:, 2].copy_((rest & live).sum(dim=1, dtype=torch.int64))
    tx.hot_counters[:, 3].copy_((rest & live & (core.agents.energy <= 0.0)).sum(dim=1, dtype=torch.int64))
    tx.hot_counters[:, 4].copy_((active & (noise_scale.squeeze(-1) > 0.0)).sum(dim=1, dtype=torch.int64))


class TensorWorldC3BackendV1:
    _capabilities = BackendCapabilitiesV1(
        backend_id="tensor_world_c3_v1",
        mechanics=(
            "patch", "jungle", "wdl", "eed", "observation", "fixed_mlp",
            "movement", "gather", "energy", "agent_active_mechanics",
            "genome_execution_view", "death_eligibility",
        ),
        batch_sizes=(1, 4, 48),
        grid_sizes=(32,),
        initial_populations=(256,),
        slot_capacities=(512,),
        policy_families=("fixed_mlp",),
        observation_abi="patch_wdl_radius1_observation_9x9_plus2_v1",
        policy_abi="fixed_mlp_83x64x64x7_fp32_pack_v1",
        determinism="fixture_trajectory_exact_c3_engineering",
        rng_mode="inherited_c2_fixture_until_c5",
        residency_mode="gpu_resident_c2_graph_plus_c3_active_mechanics_v1",
        checkpoint_replay=False,
        status="experimental",
    )

    def __init__(self, *, device: str | torch.device = "cuda:0", extension_path: str | None = None) -> None:
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise TensorWorldContractError("Tensor World C3 is a CUDA-only backend")
        self._native = load_tensor_world_c2_extension(extension_path)
        self._runs: dict[str, dict[str, Any]] = {}

    @property
    def capabilities(self) -> BackendCapabilitiesV1:
        return self._capabilities

    def initialize(self, canonical_state: Mapping[str, Any]) -> BackendRunStateV1:
        source = canonical_state.get("bundle")
        if not isinstance(source, TensorWorldStateBundleV3):
            raise TensorWorldContractError("canonical C3 state requires TensorWorldStateBundleV3")
        config = TensorWorldC3ConfigV1.from_mapping(canonical_state.get("config", {}))
        fixture_mode = bool(canonical_state.get("fixture_mode", False))
        with torch.cuda.device(self._device):
            state = clone_c3_bundle_to_device_v1(source, self._device)
            active = state.base.core.agents.alive.sum(dim=1)
            if not bool(torch.all(active == active[0]).item()):
                raise TensorWorldContractError("C3 requires equal active counts across worlds")
            shape = (state.batch, state.grid_size, int(active[0]), state.capacity)
            admitted = shape in {(1, 32, 256, 512), (4, 32, 256, 512), (48, 32, 256, 512)}
            if not admitted and (not fixture_mode or shape not in {(1, 8, 4, 8), (4, 8, 4, 8)}):
                raise TensorWorldContractError("C3 shape is outside the fail-closed capability")
            # Compile the resident genome view before checking stochastic
            # inputs; otherwise a newly enabled genome drive-noise field could
            # bypass the pre-capture validity fence through a stale agent copy.
            _sync_compiled_policy_v1(state)
            _validate_c3_active_mechanics_state_v1(state)
            _validate_c3_noise_samples_v1(state, config.mechanics)
            native_state = _native_state_v1(state.base)
            ci, cf = _config_tensors_v1(config.c2, state.base)
            snapshot_native = tuple(t.clone() for t in native_state)
            snapshot_c3 = tuple(
                tensor.clone()
                for owner in (
                    state.agent, state.genome, state.adaptation,
                    state.policy_runtime, state.lifecycle, state.transactions,
                )
                for tensor in (getattr(owner, field.name) for field in fields(owner))
            )
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize(self._device)
            self._execute(state, native_state, ci, cf, config.mechanics)
            torch.cuda.synchronize(self._device)
            with torch.cuda.graph(graph):
                self._execute(state, native_state, ci, cf, config.mechanics)
            torch.cuda.synchronize(self._device)
            for destination, saved in zip(native_state, snapshot_native, strict=True):
                destination.copy_(saved)
            index = 0
            for owner in (
                state.agent, state.genome, state.adaptation,
                state.policy_runtime, state.lifecycle, state.transactions,
            ):
                for field in fields(owner):
                    getattr(owner, field.name).copy_(snapshot_c3[index])
                    index += 1
        token = secrets.token_hex(16)
        self._runs[token] = {
            "bundle": state,
            "config": config,
            "graph": graph,
            "steady_state_counters": {
                "host_to_device_bytes": 0,
                "device_to_host_bytes": 0,
                "host_callback_count": 0,
                "synchronization_count": 0,
                "materialization_count": 0,
                "cuda_allocator_call_count": 0,
                "python_per_world_loop_count": 0,
                "python_per_agent_loop_count": 0,
            },
        }
        return BackendRunStateV1(self._capabilities.backend_id, token)

    def _execute(
        self,
        state: TensorWorldStateBundleV3,
        native_state: list[torch.Tensor],
        ci: torch.Tensor,
        cf: torch.Tensor,
        mechanics: TensorWorldC3MechanicsConfigV1,
    ) -> None:
        state.transactions.pre_energy.copy_(state.base.core.agents.energy)
        state.transactions.pre_health.copy_(state.base.core.agents.health)
        state.transactions.pre_injury.copy_(state.agent.injury)
        state.transactions.pre_survival_steps.copy_(state.base.core.agents.survival_steps.to(torch.int32))
        _sync_compiled_policy_v1(state)
        self._native.wdl_step(native_state, ci, cf)
        self._native.eed_step(native_state, ci, cf)
        self._native.patch_step(native_state, ci, cf)
        self._native.jungle_step(native_state, ci, cf)
        self._native.observe(native_state, ci, cf)
        self._native.policy_logits(native_state, ci, cf)
        self._native.select_actions(native_state, ci, cf)
        self._native.settle(native_state, ci, cf)
        _active_mechanics_cuda_v1(state, mechanics)
        _chemistry_update_cuda_v1(state, mechanics)

    def _owned(self, state: BackendRunStateV1) -> dict[str, Any]:
        if state.closed or state.backend_id != self._capabilities.backend_id or state.token not in self._runs:
            raise TensorWorldContractError("Tensor World C3 run handle is invalid or closed")
        return self._runs[state.token]

    def advance(self, state: BackendRunStateV1, request: TickRequestV1) -> TickReceiptV1:
        if not isinstance(request, TickRequestV1):
            raise TensorWorldContractError("C3 advance requires TickRequestV1")
        owned = self._owned(state)
        owned["graph"].replay()
        owned["bundle"].base.core.tick += 1
        return TickReceiptV1(self._capabilities.backend_id, 1, 0, 0, 0, 0, 0)

    def export_state(self, state: BackendRunStateV1) -> Mapping[str, Any]:
        owned = self._owned(state)
        torch.cuda.synchronize(self._device)
        return {"bundle": clone_c3_bundle_to_device_v1(owned["bundle"], torch.device("cpu")), "config": owned["config"]}

    def residency_counters(self, state: BackendRunStateV1) -> Mapping[str, int]:
        return dict(self._owned(state)["steady_state_counters"])

    def close(self, state: BackendRunStateV1) -> None:
        if not state.closed:
            self._runs.pop(state.token, None)
            state.closed = True


__all__ = [
    "TensorWorldC3BackendV1",
    "TensorWorldC3ConfigV1",
    "clone_c3_bundle_to_device_v1",
]
