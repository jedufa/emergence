"""Source-shaped fixed-MLP scalar genome birth mutation oracle.

The native source uses CPython MT19937 words plus a generation-boundary
Gaussian tape.  This module owns those exact continuation surfaces on CPU and
keeps the mutation pass transactional: a tape exhaustion or unsupported GRU
parent fails before authoritative genome/RNG state is changed.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import torch

from .c4_state import C4PythonRandomTensorStateV1, TensorWorldStateBundleV4
from .contracts import TensorWorldContractError


@dataclass(slots=True)
class _CpythonCursor:
    words: list[int]
    index: int
    has_gauss: bool
    gauss_next: float
    primary: list[float]
    secondary: list[float]
    window_base_raw_draws: int
    raw_draws: int
    semantic_draws: int

    def twist(self) -> None:
        matrix_a = 0x9908B0DF
        upper = 0x80000000
        lower = 0x7FFFFFFF
        for index in range(624 - 397):
            y = (self.words[index] & upper) | (self.words[index + 1] & lower)
            self.words[index] = (
                self.words[index + 397] ^ (y >> 1) ^ (matrix_a if y & 1 else 0)
            ) & 0xFFFFFFFF
        for index in range(624 - 397, 623):
            y = (self.words[index] & upper) | (self.words[index + 1] & lower)
            self.words[index] = (
                self.words[index - 227] ^ (y >> 1) ^ (matrix_a if y & 1 else 0)
            ) & 0xFFFFFFFF
        y = (self.words[623] & upper) | (self.words[0] & lower)
        self.words[623] = (
            self.words[397] ^ (y >> 1) ^ (matrix_a if y & 1 else 0)
        ) & 0xFFFFFFFF
        self.index = 0

    def raw32(self) -> int:
        if self.index >= 624:
            self.twist()
        value = int(self.words[self.index])
        self.index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        self.raw_draws += 1
        return value & 0xFFFFFFFF

    def random_double(self) -> float:
        a = self.raw32() >> 5
        b = self.raw32() >> 6
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)

    def consume_gauss(self, sigma: float) -> float:
        if self.has_gauss:
            value = self.gauss_next * sigma
            self.has_gauss = False
            self.gauss_next = 0.0
            return value
        tape_index = self.raw_draws - self.window_base_raw_draws
        if tape_index < 0 or tape_index >= len(self.primary):
            raise TensorWorldContractError(
                "C4 scalar mutation exhausted the generation Gaussian tape"
            )
        _ = self.random_double()
        _ = self.random_double()
        value = float(self.primary[tape_index]) * sigma
        self.gauss_next = float(self.secondary[tape_index])
        self.has_gauss = True
        return value

    def getrandbits(self, bit_count: int) -> int:
        if bit_count == 0:
            return 0
        if bit_count <= 32:
            return self.raw32() >> (32 - bit_count)
        value = 0
        filled = 0
        remaining = bit_count
        while remaining > 0:
            take = min(remaining, 32)
            word = self.raw32()
            if take < 32:
                word >>= 32 - take
            value |= word << filled
            filled += take
            remaining -= take
        return value

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise TensorWorldContractError("C4 randbelow upper bound must be positive")
        bit_count = int(upper).bit_length()
        value = self.getrandbits(bit_count)
        while value >= upper:
            value = self.getrandbits(bit_count)
        return value


def _load_cursor(
    state: C4PythonRandomTensorStateV1,
    world: int,
    *,
    window_base_raw_draws: int = 0,
) -> _CpythonCursor:
    absolute_raw_draws = int(state.python_random_raw_draws[world])
    if (
        type(window_base_raw_draws) is not int
        or window_base_raw_draws < 0
        or window_base_raw_draws > absolute_raw_draws
    ):
        raise TensorWorldContractError(
            "C4 scalar mutation Gaussian window base is invalid"
        )
    return _CpythonCursor(
        words=[int(value) for value in state.python_random_words[world].tolist()],
        index=int(state.python_random_index[world]),
        has_gauss=bool(state.python_random_has_gauss[world]),
        gauss_next=float(state.python_random_gauss_next[world]),
        primary=[float(value) for value in state.python_random_gauss_primary_tape[world].tolist()],
        secondary=[float(value) for value in state.python_random_gauss_secondary_tape[world].tolist()],
        window_base_raw_draws=window_base_raw_draws,
        raw_draws=absolute_raw_draws,
        semantic_draws=int(state.mutation_py_semantic_draws[world]),
    )


def _store_cursor(state: C4PythonRandomTensorStateV1, world: int, cursor: _CpythonCursor) -> None:
    state.python_random_words[world].copy_(torch.tensor(cursor.words, dtype=torch.uint32))
    state.python_random_index[world] = cursor.index
    state.python_random_has_gauss[world] = cursor.has_gauss
    state.python_random_gauss_next[world] = cursor.gauss_next
    state.python_random_raw_draws[world] = cursor.raw_draws
    state.mutation_py_semantic_draws[world] = cursor.semantic_draws


def clone_c4_cpython_random_state_v1(
    state: C4PythonRandomTensorStateV1,
) -> C4PythonRandomTensorStateV1:
    return copy.deepcopy(state)


def _canon_double(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    value32 = float(torch.tensor(float(value), dtype=torch.float32).item())
    return float(round(value32 * 100000000.0) / 100000000.0)


def _canon_float(value: float) -> float:
    return _canon_double(value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _snap_memory_dim(value: int) -> int:
    ladder = (0, 8, 16, 32, 64)
    return min(ladder, key=lambda item: (abs(value - item), ladder.index(item)))


def _memory_index(value: int) -> int:
    if value <= 0:
        return 0
    if value <= 8:
        return 1
    if value <= 16:
        return 2
    if value <= 32:
        return 3
    return 4


def _snap_sensor_radius(value: int, maximum: int = 3) -> int:
    best = min((1, 2, 3), key=lambda item: (abs(value - item), item))
    if maximum > 0:
        best = min(best, max(1, maximum))
    return max(1, best)


def _mutate_scalar_gene(
    cursor: _CpythonCursor,
    value: float,
    rate: float,
    sigma: float,
    lower: float,
    upper: float,
) -> tuple[float, bool, float]:
    cursor.semantic_draws += 1
    if cursor.random_double() < rate:
        delta = cursor.consume_gauss(sigma)
        cursor.semantic_draws += 1
        return _clamp(value + delta, lower, upper), True, delta
    return _clamp(value, lower, upper), False, 0.0


def _scalar_mutation_values(
    bundle: TensorWorldStateBundleV4,
    world: int,
    parent_slot: int,
    mate_slot: int,
    cursor: _CpythonCursor,
    *,
    write_slot: int | None,
) -> tuple[bool, float]:
    c3 = bundle.base
    genome = c3.genome
    parent = parent_slot
    mate = mate_slot
    if bool(genome.policy_use_gru[world, parent]) or bool(genome.policy_use_gru[world, mate]):
        raise TensorWorldContractError(
            "C4 fixed-MLP scalar mutation cannot commit a GRU parent/mate"
        )

    cursor.semantic_draws += 1  # hidden-size decision
    _ = cursor.random_double()
    parent_rate = _canon_double(float(genome.mutation_rate[world, parent]))
    mate_rate = _canon_double(float(genome.mutation_rate[world, mate]))
    base_rate = 0.5 * (parent_rate + mate_rate)
    scalar_rate = _clamp(base_rate, 0.0, 1.0)

    def average(name: str) -> float:
        return 0.5 * (
            _canon_double(float(getattr(genome, name)[world, parent]))
            + _canon_double(float(getattr(genome, name)[world, mate]))
        )

    exploration, m1, d1 = _mutate_scalar_gene(
        cursor, average("exploration_bias"), scalar_rate, 0.10, -1.0, 1.0
    )
    aggression, m2, d2 = _mutate_scalar_gene(
        cursor, average("aggression_bias"), scalar_rate, 0.15, -2.0, 2.0
    )
    threshold, m3, d3 = _mutate_scalar_gene(
        cursor, average("reproduction_threshold"), scalar_rate, 2.0, 2.0, 90.0
    )
    temperature, m4, d4 = _mutate_scalar_gene(
        cursor, average("action_temperature"), scalar_rate, 0.10, 0.25, 2.5
    )
    inertia, m5, d5 = _mutate_scalar_gene(
        cursor, average("drive_inertia"), scalar_rate, 0.05, 0.0, 0.999
    )
    noise, m6, d6 = _mutate_scalar_gene(
        cursor, average("drive_noise"), scalar_rate, 0.01, 0.0, 0.10
    )
    gain, m7, d7 = _mutate_scalar_gene(
        cursor, average("drive_logit_gain"), scalar_rate, 0.20, 0.0, 3.0
    )
    pred_lr, m8, d8 = _mutate_scalar_gene(
        cursor, average("pred_lr"), scalar_rate, 0.03, 0.0, 1.0
    )

    child_mutation_rate = base_rate
    cursor.semantic_draws += 1
    if cursor.random_double() < 0.2:
        child_mutation_rate += cursor.consume_gauss(0.005)
        cursor.semantic_draws += 1
    child_mutation_rate = _clamp(child_mutation_rate, 0.001, 0.20)

    parent_memory = int(genome.memory_dim[world, parent])
    mate_memory = int(genome.memory_dim[world, mate])
    inherited_memory = _snap_memory_dim(round(0.5 * (parent_memory + mate_memory)))
    memory_probability = _clamp(0.25 * child_mutation_rate, 0.002, 0.05)
    cursor.semantic_draws += 1
    child_memory = inherited_memory
    if cursor.random_double() < memory_probability:
        memory_idx = _memory_index(inherited_memory)
        memory_idx += -1 if cursor.randbelow(2) == 0 else 1
        cursor.semantic_draws += 1
        child_memory = (0, 8, 16, 32, 64)[min(4, max(0, memory_idx))]

    child_use_gru = bool(genome.policy_use_gru[world, parent])
    child_hidden = int(genome.hidden_size[world, parent])
    child_sensor_mask = (int(genome.sensor_channel_mask[world, parent]) & 1023) | 1
    child_sensor_radius = _snap_sensor_radius(int(genome.sensor_radius[world, parent]))
    child_sets = torch.stack(
        (
            genome.drive_setpoints[world, parent],
            genome.drive_setpoints[world, mate],
        )
    ).to(torch.float64).mean(dim=0).clamp(0.0, 1.0).to(torch.float32)
    child_exp_strength = _canon_float(_clamp(average("exp_prior_strength"), 0.0, 1.0))
    child_exp_noise = _canon_float(_clamp(average("exp_prior_noise"), 0.0, 0.5))
    child_birth_noise = _canon_float(_clamp(average("drive_birth_noise"), 0.0, 0.2))
    child_masks = {
        "introspection_channel_mask": int(genome.introspection_channel_mask[world, parent]) & 32767,
        "self_regulation_control_mask": int(genome.self_regulation_control_mask[world, parent]) & 31,
        "self_modification_target_mask": int(genome.self_modification_target_mask[world, parent]) & 31,
        "directed_inheritance_channel_mask": int(genome.directed_inheritance_channel_mask[world, parent]) & 63,
    }
    if write_slot is not None:
        slot = write_slot
        genome.hidden_size[world, slot] = child_hidden
        genome.memory_dim[world, slot] = child_memory if child_use_gru else 0
        genome.policy_mem_dim[world, slot] = child_memory if child_use_gru else 0
        genome.policy_use_gru[world, slot] = child_use_gru
        genome.sensor_channel_mask[world, slot] = child_sensor_mask
        genome.sensor_radius[world, slot] = child_sensor_radius
        for name, value in child_masks.items():
            getattr(genome, name)[world, slot] = value
        for name, value in (
            ("exploration_bias", exploration),
            ("aggression_bias", aggression),
            ("mutation_rate", child_mutation_rate),
            ("reproduction_threshold", threshold),
            ("action_temperature", temperature),
            ("drive_inertia", inertia),
            ("drive_noise", noise),
            ("drive_logit_gain", gain),
            ("pred_lr", pred_lr),
            ("exp_prior_strength", child_exp_strength),
            ("exp_prior_noise", child_exp_noise),
            ("drive_birth_noise", child_birth_noise),
        ):
            getattr(genome, name)[world, slot] = value
        genome.drive_setpoints[world, slot].copy_(child_sets)
        c3.agent.repro_threshold[world, slot] = threshold
        c3.agent.drive_inertia[world, slot] = inertia
        c3.agent.drive_noise[world, slot] = noise
        c3.base.core.policy.attack_bias[world, slot] = aggression
        c3.base.core.policy.action_temperature[world, slot] = temperature
        c3.base.core.policy.drive_logit_gain[world, slot] = gain
    mutation_delta = sum(abs(value) for value in (d1, d2, d3, d4, d5, d6, d7, d8))
    return any((m1, m2, m3, m4, m5, m6, m7, m8)), float(mutation_delta)


def preflight_c4_scalar_genome_mutations_v1(
    bundle: TensorWorldStateBundleV4,
    *,
    world: int,
    parent_slots: list[int],
    mate_slots: list[int],
    gaussian_window_base_raw_draws: int = 0,
) -> tuple[C4PythonRandomTensorStateV1, list[tuple[bool, float]]]:
    """Run the source draw/tape pass on a cloned RNG without state mutation."""

    state = bundle.scalar_rng
    if state is None:
        raise TensorWorldContractError("C4 scalar mutation requires explicit CPython RNG state")
    state.validate(expected_batch=bundle.batch)
    trial = clone_c4_cpython_random_state_v1(state)
    cursor = _load_cursor(
        trial, world, window_base_raw_draws=gaussian_window_base_raw_draws
    )
    outputs: list[tuple[bool, float]] = []
    for parent, mate in zip(parent_slots, mate_slots, strict=True):
        outputs.append(
            _scalar_mutation_values(
                bundle, world, parent, mate, cursor, write_slot=None
            )
        )
    _store_cursor(trial, world, cursor)
    return trial, outputs


def apply_c4_scalar_genome_mutations_v1(
    bundle: TensorWorldStateBundleV4,
    *,
    world: int,
    parent_slots: list[int],
    mate_slots: list[int],
    child_slots: list[int],
    gaussian_window_base_raw_draws: int = 0,
) -> list[tuple[bool, float]]:
    """Apply a preflighted scalar mutation sequence and advance RNG state."""

    state = bundle.scalar_rng
    if state is None:
        raise TensorWorldContractError("C4 scalar mutation requires explicit CPython RNG state")
    cursor = _load_cursor(
        state, world, window_base_raw_draws=gaussian_window_base_raw_draws
    )
    outputs: list[tuple[bool, float]] = []
    for parent, mate, child in zip(parent_slots, mate_slots, child_slots, strict=True):
        outputs.append(
            _scalar_mutation_values(
                bundle, world, parent, mate, cursor, write_slot=child
            )
        )
    _store_cursor(state, world, cursor)
    return outputs


__all__ = [
    "apply_c4_scalar_genome_mutations_v1",
    "clone_c4_cpython_random_state_v1",
    "preflight_c4_scalar_genome_mutations_v1",
]
