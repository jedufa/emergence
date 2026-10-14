from __future__ import annotations

import copy
import json

import pytest

from engine.campaigns.production_engine_lifecycle_accounting_v1 import (
    ACCOUNTING_COUNTER_NAMES_V1,
    PRIMARY_TERMINAL_CAUSES_V1,
    QACCT_DIRECT_RESIDENT_ABI_VARIANT_V1,
    LifecyclePopulationAccountingError,
    build_lifecycle_population_accounting_v1,
    lifecycle_qacct_receipt_v1,
    primary_terminal_cause_v1,
    validate_lifecycle_population_accounting_v1,
    validate_lifecycle_qacct_restore_v1,
)
from engine.perf import cuda_architecture_b_direct_resident_preparation_v1 as preparation
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
    DirectResidentPreparationError,
    validate_direct_resident_argument_abi_v1,
)
from tests.test_authoritative_state_runtime_abi_v1 import _prepared


LIFE_HISTORY_ABI_DEFAULTS = {
    "config.life_history_mode": 0,
    "config.life_history_maturity_start_ticks": 0,
    "config.life_history_maturity_rise_ticks": 0,
    "config.life_history_fertility_decline_start_ticks": 0,
    "config.life_history_fertility_decline_ticks": 0,
    "config.life_history_elder_fertility_floor": 1.0,
    "config.life_history_senescence_start_ticks": 0,
    "config.life_history_senescence_ramp_ticks": 0,
    "config.life_history_senescence_energy_cost": 0.0,
    "config.life_history_senescence_health_cost": 0.0,
    "config.life_history_max_age_ticks": 0,
}


def test_qacct_counter_surface_is_closed_and_ordered() -> None:
    assert ACCOUNTING_COUNTER_NAMES_V1 == (
        "live_start",
        "ecological_deaths",
        "senescence_deaths",
        "max_age_deaths",
        "legacy_turnover_culls",
        "hard_cap_removals",
        "integrity_admin_removals",
        "admitted_births",
        "immature_parent_rejections",
        "fertility_age_rejections",
        "no_capacity_rejections",
        "dead_slots_released",
        "newborn_slot_reuses",
        "live_end",
        "residual",
    )
    assert PRIMARY_TERMINAL_CAUSES_V1 == (
        "none",
        "ecological_death",
        "senescence_death",
        "max_age_death",
        "legacy_turnover_cull",
        "hard_cap_removal",
        "integrity_admin_removal",
    )


def test_life_history_abi_slots_have_unique_fail_closed_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = preparation._require_turnover_selection_seed_by_cell

    def validate(value, **kwargs):
        if value.device.type == "cpu":
            return value
        return original(value, **kwargs)

    monkeypatch.setattr(
        preparation,
        "_require_turnover_selection_seed_by_cell",
        validate,
    )
    prepared = _prepared("cpu")
    arguments = tuple(prepared.arguments)
    indices = {
        name: DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(name)
        for name in LIFE_HISTORY_ABI_DEFAULTS
    }

    assert tuple(indices.values()) == tuple(range(111, 122))
    for name, expected in LIFE_HISTORY_ABI_DEFAULTS.items():
        assert arguments[indices[name]] == expected

    bad_values = {
        "config.life_history_mode": 2,
        "config.life_history_maturity_start_ticks": -1,
        "config.life_history_maturity_rise_ticks": -2,
        "config.life_history_fertility_decline_start_ticks": -3,
        "config.life_history_fertility_decline_ticks": -4,
        "config.life_history_elder_fertility_floor": 1.25,
        "config.life_history_senescence_start_ticks": -5,
        "config.life_history_senescence_ramp_ticks": -6,
        "config.life_history_senescence_energy_cost": -0.25,
        "config.life_history_senescence_health_cost": -0.5,
        "config.life_history_max_age_ticks": -7,
    }
    for name, bad_value in bad_values.items():
        mutated = list(arguments)
        mutated[indices[name]] = bad_value
        with pytest.raises(DirectResidentPreparationError, match=name):
            validate_direct_resident_argument_abi_v1(mutated)


def test_population_accounting_partitions_live_count_changes() -> None:
    accounting = build_lifecycle_population_accounting_v1(
        live_start=64,
        ecological_deaths=3,
        senescence_deaths=2,
        max_age_deaths=1,
        admitted_births=6,
        dead_slots_released=6,
        newborn_slot_reuses=6,
        live_end=64,
    )

    assert accounting.terminal_removals == 6
    assert accounting.residual == 0
    assert validate_lifecycle_population_accounting_v1(accounting) is accounting


def test_population_accounting_fails_closed_on_residual_or_cap_activation() -> None:
    residual = build_lifecycle_population_accounting_v1(
        live_start=5,
        ecological_deaths=1,
        admitted_births=1,
        live_end=6,
    )
    with pytest.raises(LifecyclePopulationAccountingError, match="residual"):
        validate_lifecycle_population_accounting_v1(residual)

    hard_cap = build_lifecycle_population_accounting_v1(
        live_start=5,
        hard_cap_removals=1,
        live_end=4,
    )
    with pytest.raises(LifecyclePopulationAccountingError, match="hard-cap"):
        validate_lifecycle_population_accounting_v1(hard_cap)


def test_terminal_cause_precedence_produces_one_primary_cause() -> None:
    assert primary_terminal_cause_v1({}) == "none"
    assert primary_terminal_cause_v1({"ecological_death": True}) == (
        "ecological_death"
    )
    assert primary_terminal_cause_v1(
        {
            "ecological_death": True,
            "senescence_death": True,
            "max_age_death": True,
        }
    ) == "max_age_death"
    assert primary_terminal_cause_v1(
        {"hard_cap_removal": True, "ecological_death": True}
    ) == "hard_cap_removal"
    with pytest.raises(LifecyclePopulationAccountingError, match="unknown"):
        primary_terminal_cause_v1({"predator_death": True})
    with pytest.raises(LifecyclePopulationAccountingError, match="boolean"):
        primary_terminal_cause_v1({"ecological_death": 1})


def test_qacct_receipt_is_stable_and_restore_matches_exact_schema_and_profile() -> None:
    accounting = build_lifecycle_population_accounting_v1(
        live_start=10,
        ecological_deaths=2,
        admitted_births=2,
        dead_slots_released=2,
        newborn_slot_reuses=2,
        live_end=10,
    )
    receipt = lifecycle_qacct_receipt_v1(
        accounting=accounting,
        lifecycle_control_sha256="a" * 64,
        configuration_sha256="b" * 64,
        profile_sha256="c" * 64,
        checkpoint_schema_version="checkpoint_bundle_v2",
    )

    json.dumps(receipt, sort_keys=True, allow_nan=False)
    assert receipt == lifecycle_qacct_receipt_v1(
        accounting=accounting,
        lifecycle_control_sha256="a" * 64,
        configuration_sha256="b" * 64,
        profile_sha256="c" * 64,
        checkpoint_schema_version="checkpoint_bundle_v2",
    )
    assert receipt["direct_resident_abi_variant"] == (
        QACCT_DIRECT_RESIDENT_ABI_VARIANT_V1
    )
    assert validate_lifecycle_qacct_restore_v1(
        receipt,
        expected_checkpoint_schema_version="checkpoint_bundle_v2",
        expected_configuration_sha256="b" * 64,
        expected_profile_sha256="c" * 64,
    ) is receipt

    mismatched = copy.deepcopy(receipt)
    mismatched["profile_sha256"] = "d" * 64
    with pytest.raises(LifecyclePopulationAccountingError, match="profile"):
        validate_lifecycle_qacct_restore_v1(
            mismatched,
            expected_checkpoint_schema_version="checkpoint_bundle_v2",
            expected_configuration_sha256="b" * 64,
            expected_profile_sha256="c" * 64,
        )
