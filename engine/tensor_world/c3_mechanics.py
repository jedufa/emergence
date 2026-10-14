"""Source-transcribed C3 active-agent mechanics for bounded oracle fixtures.

The implementation is intentionally CPU/scalar and is only an oracle/fixture
surface.  It is not admissible as a steady-state backend path: a production
backend must provide the same ordering in resident CUDA state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import torch

from .c3_state import TensorWorldStateBundleV3
from .contracts import TensorWorldContractError


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _sub(left: float, right: float) -> float:
    return _f32(_f32(left) - _f32(right))


def _mul(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _div(left: float, right: float) -> float:
    return _f32(_f32(left) / _f32(right))


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _max_zero(value: float) -> float:
    return value if value > 0.0 else 0.0


def _bounded_ramp(age: int, start: int, ramp: int) -> float:
    if age < start:
        return 0.0
    if ramp <= 0:
        return 1.0
    return _clamp01(float(age - start + 1) / float(ramp))


def _fertility_scale(age: int, config: "TensorWorldC3MechanicsConfigV1") -> float:
    if config.life_history_mode == 0:
        return 1.0
    if age < config.life_history_maturity_start_ticks:
        return 0.0
    scale = 1.0
    if config.life_history_maturity_rise_ticks > 0:
        scale = min(
            1.0,
            float(age - config.life_history_maturity_start_ticks + 1)
            / float(config.life_history_maturity_rise_ticks),
        )
    if (
        config.life_history_fertility_decline_start_ticks > 0
        and age >= config.life_history_fertility_decline_start_ticks
    ):
        if config.life_history_fertility_decline_ticks <= 0:
            scale = min(scale, config.life_history_elder_fertility_floor)
        else:
            decline = min(
                1.0,
                float(age - config.life_history_fertility_decline_start_ticks + 1)
                / float(config.life_history_fertility_decline_ticks),
            )
            scale = min(
                scale,
                1.0 - decline * (1.0 - config.life_history_elder_fertility_floor),
            )
    return _clamp01(scale)


def apply_c3_chemistry_oracle_v1(
    bundle: TensorWorldStateBundleV3,
    config: "TensorWorldC3MechanicsConfigV1",
) -> None:
    """Apply the source-current six-channel chemistry update on a CPU fixture.

    The source consumes two standard-normal draws per live row.  C5 owns the
    resident RNG stream, so this oracle accepts the already materialized
    per-slot samples and fails closed when the caller has not supplied them.
    """

    bundle.validate()
    if bundle.device.type != "cpu":
        raise TensorWorldContractError("C3 scalar oracle requires a CPU fixture")
    if not config.enable_chemistry:
        return
    core = bundle.base.core
    adaptation = bundle.adaptation
    tx = bundle.transactions
    for world in range(bundle.batch):
        for slot in range(bundle.capacity):
            if not bool(core.agents.alive[world, slot]):
                continue
            if int(core.agents.agent_id[world, slot]) < 0:
                raise TensorWorldContractError("C3 alive row has no biological identity")
            if config.chemistry_noise_required and not bool(
                tx.chemistry_noise_samples_valid[world, slot]
            ):
                raise TensorWorldContractError("C3 chemistry noise sample is missing")

            energy_n = _clamp01(
                _div(float(core.agents.energy[world, slot]), config.max_energy)
            )
            health_n = _clamp01(
                _div(float(core.agents.health[world, slot]), config.start_health)
            )
            injury_n = _clamp01(
                _div(float(bundle.agent.injury[world, slot]), config.start_health)
            )
            stress = _clamp01(
                0.55 * injury_n
                + 0.30 * float(bundle.agent.drives[world, slot, 1])
                + 0.15 * (1.0 - energy_n)
            )
            current = [
                float(adaptation.chemistry[world, slot, index])
                for index in range(6)
            ]
            energy_store, repair, acute, chronic, reg1, reg2 = current
            acute_target = _clamp01(
                1.25 * injury_n + 0.45 * stress + 0.20 * (1.0 - health_n)
            )
            acute = _clamp01(
                acute + 0.50 * (acute_target - acute) + 0.03 * reg1
            )
            chronic_target = _clamp01(0.75 * chronic + 0.25 * acute)
            chronic = _clamp01(chronic + 0.07 * (chronic_target - chronic))
            energy_store = _clamp01(
                energy_store + 0.03 * energy_n - 0.02 * stress - 0.01 * acute
            )
            repair = _clamp01(
                repair + 0.04 * energy_n - 0.03 * acute - 0.01 * chronic
            )
            noise0 = (
                float(tx.chemistry_noise_samples[world, slot, 0])
                if config.chemistry_noise_required
                else 0.0
            )
            noise1 = (
                float(tx.chemistry_noise_samples[world, slot, 1])
                if config.chemistry_noise_required
                else 0.0
            )
            reg1 = _clamp01(0.92 * reg1 + 0.08 * (1.0 - acute) + 0.01 * noise0)
            reg2 = _clamp01(0.97 * reg2 + 0.03 * chronic + 0.01 * noise1)
            acute_resolve = min(acute, 0.10 * repair + 0.04 * energy_store)
            acute = _clamp01(acute - acute_resolve)
            chronic = _clamp01(chronic + 0.25 * acute - 0.002 * repair)
            for index, value in enumerate(
                (energy_store, repair, acute, chronic, reg1, reg2)
            ):
                adaptation.chemistry[world, slot, index] = _f32(value)


@dataclass(frozen=True, slots=True)
class TensorWorldC3MechanicsConfigV1:
    enable_rest_charge: bool = False
    enable_chemistry: bool = False
    chemistry_noise_required: bool = True
    rest_streak_free: int = 3
    rest_charge_energy: float = 0.0
    rest_charge_growth: float = 0.0
    rest_deficit_to_health: float = 0.0
    developmental_maturity_age: int = 20
    life_history_mode: int = 0
    life_history_maturity_start_ticks: int = 0
    life_history_maturity_rise_ticks: int = 0
    life_history_fertility_decline_start_ticks: int = 0
    life_history_fertility_decline_ticks: int = 0
    life_history_elder_fertility_floor: float = 0.0
    life_history_senescence_start_ticks: int = 0
    life_history_senescence_ramp_ticks: int = 0
    life_history_senescence_energy_cost: float = 0.0
    life_history_senescence_health_cost: float = 0.0
    life_history_max_age_ticks: int = 0
    episode0_k: int = 10
    max_energy: float = 160.0
    start_health: float = 12.0
    drive_hunger_gain: float = 0.1
    drive_hunger_decay: float = 0.01
    drive_pain_gain: float = 0.2
    drive_pain_decay: float = 0.01
    drive_fatigue_gain: float = 0.05
    drive_fatigue_rest_relief: float = 0.1
    drive_fatigue_decay: float = 0.01
    drive_fear_gain: float = 0.2
    drive_fear_decay: float = 0.01

    def __post_init__(self) -> None:
        if type(self.enable_rest_charge) is not bool:
            raise TensorWorldContractError("C3 enable_rest_charge must be bool")
        if type(self.enable_chemistry) is not bool:
            raise TensorWorldContractError("C3 enable_chemistry must be bool")
        if type(self.chemistry_noise_required) is not bool:
            raise TensorWorldContractError("C3 chemistry_noise_required must be bool")
        if self.life_history_mode not in (0, 1):
            raise TensorWorldContractError("C3 life_history_mode must be 0 or 1")
        for name in (
            "rest_streak_free",
            "developmental_maturity_age",
            "life_history_maturity_start_ticks",
            "life_history_maturity_rise_ticks",
            "life_history_fertility_decline_start_ticks",
            "life_history_fertility_decline_ticks",
            "life_history_senescence_start_ticks",
            "life_history_senescence_ramp_ticks",
            "life_history_max_age_ticks",
            "episode0_k",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise TensorWorldContractError(f"C3 {name} must be nonnegative int")
        for name in (
            "life_history_elder_fertility_floor",
            "life_history_senescence_energy_cost",
            "life_history_senescence_health_cost",
            "max_energy",
            "start_health",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise TensorWorldContractError(f"C3 {name} must be finite and nonnegative")
        if not 0.0 <= self.life_history_elder_fertility_floor <= 1.0:
            raise TensorWorldContractError("C3 fertility floor must be in [0, 1]")
        if self.max_energy <= 0.0 or self.start_health <= 0.0:
            raise TensorWorldContractError("C3 normalization maxima must be positive")


def capture_c3_pre_action_v1(bundle: TensorWorldStateBundleV3) -> None:
    """Capture the source active-mechanics pre-action snapshot."""

    bundle.validate()
    if bundle.device.type != "cpu":
        raise TensorWorldContractError("C3 scalar oracle requires a CPU fixture")
    bundle.transactions.pre_energy.copy_(bundle.base.core.agents.energy)
    bundle.transactions.pre_health.copy_(bundle.base.core.agents.health)
    bundle.transactions.pre_injury.copy_(bundle.agent.injury)
    bundle.transactions.pre_survival_steps.copy_(
        bundle.base.core.agents.survival_steps.to(torch.int32)
    )
    bundle.transactions.actions_by_slot.copy_(
        bundle.base.core.transactions.actions.to(torch.int32)
    )


def apply_c3_active_mechanics_oracle_v1(
    bundle: TensorWorldStateBundleV3,
    config: TensorWorldC3MechanicsConfigV1,
) -> None:
    """Apply the source-current active-mechanics phase without C4 culling.

    The function emits ``lifecycle.death_candidate`` and reason codes but does
    not clear ``alive``/identity/occupancy.  That boundary is intentionally
    reserved for the C4 population transaction.
    """

    bundle.validate()
    if bundle.device.type != "cpu":
        raise TensorWorldContractError("C3 scalar oracle requires a CPU fixture")
    if config.max_energy <= 0.0 or config.start_health <= 0.0:
        raise TensorWorldContractError("C3 normalization maxima must be positive")

    core = bundle.base.core
    agent = bundle.agent
    tx = bundle.transactions
    lifecycle = bundle.lifecycle

    # These source-current fields are owned by the C3 execution view even
    # though the inherited C2 bundle stores the transport copies.  Synchronize
    # them at the phase boundary so C4/C5 never has to infer age or exact
    # accumulation from a slot-local implementation detail.
    agent.age.copy_(core.agents.age.to(torch.int32))
    agent.survival_steps.copy_(core.agents.survival_steps.to(torch.int32))
    agent.energy_collected.copy_(core.agents.energy_collected.to(torch.float32))
    agent.energy_collected_exact.copy_(core.agents.energy_collected)

    # These transaction/reporting surfaces are part of the source active
    # mechanics boundary, not optional diagnostics.  Capture the raw selected
    # action before normalizing invalid values for the mechanics walk.
    tx.row_actions.copy_(tx.actions_by_slot)
    active_rows = torch.zeros_like(core.agents.alive)
    live_rows = core.agents.alive & (core.agents.agent_id >= 0)
    rest_rows = torch.zeros_like(core.agents.alive)

    # The source controller records action-history counters immediately after
    # action selection, before the active-mechanics gate.  C4 owns population
    # mutation, but these resident C3 counters belong to the agent execution
    # surface and therefore remain unchanged by slot settlement.
    for world in range(bundle.batch):
        for slot in range(bundle.capacity):
            if not bool(core.agents.alive[world, slot]):
                continue
            if int(core.agents.agent_id[world, slot]) < 0:
                raise TensorWorldContractError("C3 alive row has no biological identity")
            action = int(tx.actions_by_slot[world, slot])
            if action < 0 or action > 6:
                action = 5
            # The source policy service refills both cache entries and binds
            # the cache to the stable slot token.  C3's inherited C2 policy
            # phase has already selected ``action``; retain that selection in
            # the source-shaped cache without inventing a per-tick token.
            bundle.policy_runtime.action_cache_actions[world, slot, :] = action
            bundle.policy_runtime.action_cache_token[world, slot] = int(
                bundle.base.core.identity.slot_ids[world, slot]
            )
            bundle.policy_runtime.action_cache_cursor[world, slot] = 1
            bundle.policy_runtime.action_cache_next[world, slot] = action
            if action == 6:
                agent.attacks[world, slot] += 1
            rest_rows[world, slot] = action == 5
            old_last = int(agent.last_action[world, slot])
            old_last_last = int(agent.last_last_action[world, slot])
            switched = old_last != -1 and action != old_last
            oscillated = old_last_last != -1 and action == old_last_last and action != old_last
            if switched:
                agent.action_switches[world, slot] += 1
                if int(core.agents.survival_steps[world, slot]) < config.episode0_k:
                    agent.episode0_switches[world, slot] += 1
            if oscillated:
                agent.action_oscillations[world, slot] += 1
                if int(core.agents.survival_steps[world, slot]) < config.episode0_k:
                    agent.episode0_oscillations[world, slot] += 1
            agent.last_last_action[world, slot] = old_last
            agent.last_action[world, slot] = action

    for world in range(bundle.batch):
        for slot in range(bundle.capacity):
            if not bool(core.agents.alive[world, slot]):
                lifecycle.death_candidate[world, slot] = False
                lifecycle.death_reason[world, slot] = 0
                continue
            if int(core.agents.agent_id[world, slot]) < 0:
                raise TensorWorldContractError("C3 alive row has no biological identity")

            energy = float(core.agents.energy[world, slot])
            health = float(core.agents.health[world, slot])
            injury = float(agent.injury[world, slot])
            pre_energy = float(tx.pre_energy[world, slot])
            pre_health = float(tx.pre_health[world, slot])
            pre_injury = float(tx.pre_injury[world, slot])
            for name, value in (
                ("energy", energy),
                ("health", health),
                ("injury", injury),
                ("pre_energy", pre_energy),
                ("pre_health", pre_health),
                ("pre_injury", pre_injury),
            ):
                if not math.isfinite(value):
                    raise TensorWorldContractError(f"C3 live {name} is nonfinite")

            action = int(tx.actions_by_slot[world, slot])
            if action < 0 or action > 6:
                action = 5
            if config.enable_rest_charge:
                if action == 5:
                    streak = int(agent.rest_streak[world, slot]) + 1
                else:
                    streak = 0
                agent.rest_streak[world, slot] = streak
                if streak > config.rest_streak_free:
                    over = streak - config.rest_streak_free
                    charge = _add(
                        config.rest_charge_energy,
                        _mul(config.rest_charge_growth, float(over - 1)),
                    )
                    energy = _sub(energy, charge)
                    if energy < 0.0:
                        health = _sub(
                            health,
                            _mul(-energy, config.rest_deficit_to_health),
                        )
                        energy = 0.0
                    agent.policy_energy_observation[world, slot] = float(energy)

            # The source terminal gate precedes upkeep, life-history, and
            # drive updates; C4 receives the resulting eligibility later.
            if energy > 0.0 and health > 0.0:
                active_rows[world, slot] = True
                energy = max(0.0, _sub(energy, _max_zero(float(agent.gru_upkeep_cost[world, slot]))))
                charge = _max_zero(float(agent.cost_charge_scalar[world, slot]))
                survival = int(core.agents.survival_steps[world, slot])
                if (
                    config.developmental_maturity_age > 0
                    and survival < config.developmental_maturity_age
                ):
                    charge = _add(
                        charge,
                        _max_zero(float(agent.cost_charge_maturation_scalar[world, slot])),
                    )
                if charge > 0.0:
                    energy = max(0.0, _sub(energy, charge))

                age = survival
                senescence = (
                    _bounded_ramp(
                        age,
                        config.life_history_senescence_start_ticks,
                        config.life_history_senescence_ramp_ticks,
                    )
                    if config.life_history_mode == 1
                    else 0.0
                )
                if config.life_history_mode == 1:
                    if senescence > 0.0:
                        energy = max(
                            0.0,
                            _sub(
                                energy,
                                _mul(config.life_history_senescence_energy_cost, senescence),
                            ),
                        )
                        health = _sub(
                            health,
                            _mul(config.life_history_senescence_health_cost, senescence),
                        )
                    if (
                        config.life_history_max_age_ticks > 0
                        and age >= config.life_history_max_age_ticks
                    ):
                        health = 0.0

                core.agents.survival_steps[world, slot] = survival + 1
                core.agents.age[world, slot] = int(core.agents.age[world, slot]) + 1
                agent.survival_steps[world, slot] = survival + 1
                agent.age[world, slot] = int(core.agents.age[world, slot])
                agent.energy_collected[world, slot] = float(core.agents.energy_collected[world, slot])
                agent.energy_collected_exact[world, slot] = core.agents.energy_collected[world, slot]
                agent.policy_energy_observation[world, slot] = float(energy)
                lifecycle.maturity_factor[world, slot] = _bounded_ramp(
                    age,
                    config.life_history_maturity_start_ticks,
                    config.life_history_maturity_rise_ticks,
                ) if config.life_history_mode == 1 else 1.0
                lifecycle.fertility_factor[world, slot] = _fertility_scale(age, config)
                lifecycle.senescence_factor[world, slot] = senescence

                de = _sub(energy, pre_energy)
                dh = _sub(health, pre_health)
                dinj = _sub(injury, pre_injury)
                agent.last_energy_delta[world, slot] = de
                agent.last_health_delta[world, slot] = dh
                agent.last_injury_delta[world, slot] = dinj

                drives = [float(agent.drives[world, slot, index]) for index in range(4)]
                previous_delta = [
                    float(agent.prev_drive_delta[world, slot, index])
                    for index in range(4)
                ]
                energy_norm = _clamp01(_div(energy, config.max_energy))
                health_norm = _clamp01(_div(health, config.start_health))

                hunger = _sub(
                    _add(
                        _add(
                            drives[0],
                            _mul(config.drive_hunger_gain, _sub(1.0, energy_norm)),
                        ),
                        _mul(0.04, _div(-de, config.max_energy)),
                    ),
                    config.drive_hunger_decay,
                )
                if action == 4 and de > 0.0:
                    hunger = _sub(hunger, _mul(0.20, _div(de, config.max_energy)))
                hunger = _clamp01(hunger)

                pain = drives[1]
                if dinj > 0.0:
                    pain = _add(pain, _mul(config.drive_pain_gain, min(dinj, 1.0)))
                if dh < 0.0:
                    pain = _add(
                        pain,
                        _mul(0.10, min(_div(-dh, config.start_health), 1.0)),
                    )
                pain = _clamp01(_sub(pain, config.drive_pain_decay))

                fatigue = drives[2]
                fatigue = _sub(
                    fatigue,
                    config.drive_fatigue_rest_relief if action == 5 else -config.drive_fatigue_gain,
                )
                fatigue = _clamp01(_sub(fatigue, config.drive_fatigue_decay))

                fear = drives[3]
                if dinj > 0.0:
                    fear = _add(fear, _mul(config.drive_fear_gain, min(dinj, 1.0)))
                fear = _clamp01(
                    _sub(
                        _add(fear, _mul(0.05, _sub(1.0, health_norm))),
                        config.drive_fear_decay,
                    )
                )

                targets = [hunger, pain, fatigue, fear]
                inertia = min(0.999, max(0.0, float(agent.drive_inertia[world, slot])))
                inverse = _sub(1.0, inertia)
                next_drives = [
                    _add(_mul(inertia, previous), _mul(inverse, target))
                    for previous, target in zip(drives, targets, strict=True)
                ]
                noise_scale = max(0.0, float(agent.drive_noise[world, slot]))
                if noise_scale > 0.0:
                    if not bool(tx.drive_noise_samples_valid[world, slot]):
                        raise TensorWorldContractError("C3 drive noise sample is missing")
                    next_drives = [
                        _add(next_value, _mul(noise_scale, float(tx.drive_noise_samples[world, slot, index])))
                        for index, next_value in enumerate(next_drives)
                    ]
                next_drives = [_clamp01(value) for value in next_drives]
                deltas = [_sub(next_value, previous) for next_value, previous in zip(next_drives, drives, strict=True)]
                volatility = sum(abs(value) for value in deltas)
                jitter = sum(abs(value - previous) for value, previous in zip(deltas, previous_delta, strict=True))
                agent.drive_volatility[world, slot] = _add(float(agent.drive_volatility[world, slot]), volatility)
                agent.drive_jitter[world, slot] = _add(float(agent.drive_jitter[world, slot]), jitter)
                if int(core.agents.survival_steps[world, slot]) <= config.episode0_k:
                    agent.episode0_drive_jitter_sum[world, slot] = _add(
                        float(agent.episode0_drive_jitter_sum[world, slot]), jitter
                    )
                for index, value in enumerate(deltas):
                    agent.prev_drive_delta[world, slot, index] = value
                    agent.drives[world, slot, index] = next_drives[index]
                if action == 6 and dinj > 0.0:
                    agent.selfharm_events[world, slot] += 1
                    if int(core.agents.survival_steps[world, slot]) <= config.episode0_k:
                        agent.episode0_selfharm_events[world, slot] += 1
            else:
                lifecycle.maturity_factor[world, slot] = 0.0
                lifecycle.fertility_factor[world, slot] = 0.0
                lifecycle.senescence_factor[world, slot] = 0.0

            core.agents.energy[world, slot] = _f32(energy)
            core.agents.health[world, slot] = _f32(health)
            lifecycle.death_candidate[world, slot] = energy <= 0.0 or health <= 0.0
            lifecycle.death_reason[world, slot] = (
                2 if health <= 0.0 else (1 if energy <= 0.0 else 0)
            )

    tx.hot_counters[:, :5].zero_()
    tx.hot_counters[:, 0].copy_(active_rows.sum(dim=1, dtype=torch.int64))
    tx.hot_counters[:, 1].copy_(active_rows.sum(dim=1, dtype=torch.int64))
    tx.hot_counters[:, 2].copy_(
        (rest_rows & live_rows).sum(dim=1, dtype=torch.int64)
    )
    tx.hot_counters[:, 3].copy_(
        (rest_rows & live_rows & (core.agents.energy <= 0.0)).sum(
            dim=1, dtype=torch.int64
        )
    )
    tx.hot_counters[:, 4].copy_(
        (
            active_rows
            & (agent.drive_noise > 0.0)
        ).sum(dim=1, dtype=torch.int64)
    )

    apply_c3_chemistry_oracle_v1(bundle, config)


__all__ = [
    "TensorWorldC3MechanicsConfigV1",
    "apply_c3_chemistry_oracle_v1",
    "apply_c3_active_mechanics_oracle_v1",
    "capture_c3_pre_action_v1",
]
