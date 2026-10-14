from __future__ import annotations

from engine.option_d.contracts import DrawSchedule, RNGStateBundle
from engine.option_d.cpu import execute_cpu, state_from_seed
from engine.option_d.gauss_audit import _schedule_cases, _untemper


def _temper(value: int) -> int:
    value ^= value >> 11
    value ^= (value << 7) & 0x9D2C5680
    value ^= (value << 15) & 0xEFC60000
    value ^= value >> 18
    return value & 0xFFFFFFFF


def test_boundary_states_invert_the_cpython_temper_transform() -> None:
    for target in (0, 1, 0x12345678, 0x80000000, 0xFFFFFFFF):
        assert _temper(_untemper(target)) == target


def test_gaussian_audit_cases_are_cpu_executable() -> None:
    for case_index, (_name, worlds, explicit_states) in enumerate(_schedule_cases()):
        states = explicit_states or tuple(
            state_from_seed(0xD3000000 + case_index * 100 + world)
            for world in range(len(worlds))
        )
        bundle = RNGStateBundle(states, source_binding="gauss-audit-contract-test")

        result = execute_cpu(bundle, DrawSchedule(worlds))

        result.successor.validate()
