from __future__ import annotations

from tools.evidence.profiles import MEMORY_ASSAY_CONFIG_EXPECTATIONS, get_profile


FORBIDDEN_MEMORY_PRESSURE_EXPECTATIONS = {
    "disable_prediction_error_selection_pressure": True,
    "episode0_fitness_weight": 0.0,
    "episode0_pred_err_fitness_weight": 0.0,
    "episode0_thrashing_fitness_weight": 0.0,
    "episode0_drive_jitter_fitness_weight": 0.0,
    "episode0_selfharm_fitness_weight": 0.0,
    "late_pred_cost_enable": False,
    "late_coherence_eff_enable": False,
    "adult_move_penalty_enable": False,
    "identity_drift_enable": False,
}


def _expectations_by_field(profile_name: str) -> dict[str, object]:
    profile = get_profile(profile_name)
    return {expectation.field: expectation.expected for expectation in profile.config_expectations}


def test_memory_assay_profile_freezes_forbidden_pressure_expectations() -> None:
    expectations = _expectations_by_field("CRB-M1")

    for field, expected in FORBIDDEN_MEMORY_PRESSURE_EXPECTATIONS.items():
        assert expectations[field] == expected


def test_memory_assay_expectation_tuple_is_stable() -> None:
    fields = {expectation.field for expectation in MEMORY_ASSAY_CONFIG_EXPECTATIONS}

    assert set(FORBIDDEN_MEMORY_PRESSURE_EXPECTATIONS).issubset(fields)
