"""Lifecycle V2 population accounting and receipt authority."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

from .canonical import canonical_sha256


QACCT_SCHEMA_VERSION: Final[str] = "eiel_gv2_lifecycle_v2_qacct1_population_accounting_v1"
QACCT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_gv2_lifecycle_v2_qacct1_receipt_v1"
)
QACCT_CHECKPOINT_CONTRACT_VERSION: Final[str] = (
    "eiel_gv2_lifecycle_v2_qacct1_checkpoint_contract_v1"
)
QACCT_DIRECT_RESIDENT_ABI_VARIANT_V1: Final[str] = "direct_resident_outer_172"
CAPACITY_ACCOUNTING_SCHEMA_VERSION: Final[str] = (
    "eiel_gv2_lifecycle_v2_capacity_accounting_v1"
)
CAPACITY_REASON_AUTHORITY_FULL_PREFIX_V1: Final[str] = (
    "authoritative_full_prefix_route_counts"
)
CAPACITY_REASON_AUTHORITY_DIAGNOSTIC_V1: Final[str] = (
    "diagnostic_inferred_saturated_capacity"
)

PRIMARY_TERMINAL_CAUSES_V1: Final[tuple[str, ...]] = (
    "none",
    "ecological_death",
    "senescence_death",
    "max_age_death",
    "legacy_turnover_cull",
    "hard_cap_removal",
    "integrity_admin_removal",
)
CAUSE_PRECEDENCE_V1: Final[tuple[str, ...]] = (
    "integrity_admin_removal",
    "hard_cap_removal",
    "legacy_turnover_cull",
    "max_age_death",
    "senescence_death",
    "ecological_death",
)
ACCOUNTING_COUNTER_NAMES_V1: Final[tuple[str, ...]] = (
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


class LifecyclePopulationAccountingError(ValueError):
    """Lifecycle accounting is incomplete, ambiguous, or internally inconsistent."""


def _fail(message: str) -> LifecyclePopulationAccountingError:
    return LifecyclePopulationAccountingError(message)


def _strict_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(f"{name} must be an integer")
    if value < 0:
        raise _fail(f"{name} must be non-negative")
    return value


def primary_terminal_cause_v1(active_causes: Mapping[str, object]) -> str:
    """Return exactly one primary terminal/removal cause from boolean flags."""

    admitted = set(PRIMARY_TERMINAL_CAUSES_V1) - {"none"}
    unknown = set(active_causes) - admitted
    if unknown:
        raise _fail(f"unknown terminal cause flags: {sorted(unknown)}")
    non_boolean = [
        cause for cause, value in active_causes.items() if not isinstance(value, bool)
    ]
    if non_boolean:
        raise _fail(f"terminal cause flags must be boolean: {sorted(non_boolean)}")
    selected = [
        cause
        for cause in CAUSE_PRECEDENCE_V1
        if active_causes.get(cause, False)
    ]
    if len(selected) > 1:
        return selected[0]
    if len(selected) == 1:
        return selected[0]
    return "none"


@dataclass(frozen=True, slots=True)
class LifecyclePopulationAccountingV1:
    schema_version: str
    live_start: int
    ecological_deaths: int
    senescence_deaths: int
    max_age_deaths: int
    legacy_turnover_culls: int
    hard_cap_removals: int
    integrity_admin_removals: int
    admitted_births: int
    immature_parent_rejections: int
    fertility_age_rejections: int
    no_capacity_rejections: int
    dead_slots_released: int
    newborn_slot_reuses: int
    live_end: int
    residual: int

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def terminal_removals(self) -> int:
        return (
            self.ecological_deaths
            + self.senescence_deaths
            + self.max_age_deaths
            + self.legacy_turnover_culls
            + self.hard_cap_removals
            + self.integrity_admin_removals
        )


def build_lifecycle_population_accounting_v1(
    *,
    live_start: int,
    ecological_deaths: int = 0,
    senescence_deaths: int = 0,
    max_age_deaths: int = 0,
    legacy_turnover_culls: int = 0,
    hard_cap_removals: int = 0,
    integrity_admin_removals: int = 0,
    admitted_births: int = 0,
    immature_parent_rejections: int = 0,
    fertility_age_rejections: int = 0,
    no_capacity_rejections: int = 0,
    dead_slots_released: int = 0,
    newborn_slot_reuses: int = 0,
    live_end: int,
) -> LifecyclePopulationAccountingV1:
    values = {
        "live_start": live_start,
        "ecological_deaths": ecological_deaths,
        "senescence_deaths": senescence_deaths,
        "max_age_deaths": max_age_deaths,
        "legacy_turnover_culls": legacy_turnover_culls,
        "hard_cap_removals": hard_cap_removals,
        "integrity_admin_removals": integrity_admin_removals,
        "admitted_births": admitted_births,
        "immature_parent_rejections": immature_parent_rejections,
        "fertility_age_rejections": fertility_age_rejections,
        "no_capacity_rejections": no_capacity_rejections,
        "dead_slots_released": dead_slots_released,
        "newborn_slot_reuses": newborn_slot_reuses,
        "live_end": live_end,
    }
    checked = {
        name: _strict_nonnegative_int(value, name=name)
        for name, value in values.items()
    }
    terminal_removals = (
        checked["ecological_deaths"]
        + checked["senescence_deaths"]
        + checked["max_age_deaths"]
        + checked["legacy_turnover_culls"]
        + checked["hard_cap_removals"]
        + checked["integrity_admin_removals"]
    )
    expected_live_end = (
        checked["live_start"] - terminal_removals + checked["admitted_births"]
    )
    residual = checked["live_end"] - expected_live_end
    return LifecyclePopulationAccountingV1(
        schema_version=QACCT_SCHEMA_VERSION,
        residual=residual,
        **checked,
    )


def validate_lifecycle_population_accounting_v1(
    accounting: LifecyclePopulationAccountingV1,
) -> LifecyclePopulationAccountingV1:
    if not isinstance(accounting, LifecyclePopulationAccountingV1):
        raise _fail("accounting must be LifecyclePopulationAccountingV1")
    if accounting.schema_version != QACCT_SCHEMA_VERSION:
        raise _fail("accounting schema_version mismatch")
    for name in ACCOUNTING_COUNTER_NAMES_V1:
        _strict_nonnegative_int(getattr(accounting, name), name=name)
    if accounting.residual != 0:
        raise _fail("lifecycle population accounting residual must be zero")
    if accounting.dead_slots_released > accounting.terminal_removals:
        raise _fail("dead slot releases exceed terminal removals")
    if accounting.newborn_slot_reuses > accounting.admitted_births:
        raise _fail("newborn slot reuses exceed admitted births")
    if accounting.hard_cap_removals > 0:
        raise _fail("hard-cap removals are reportable safety-fuse activations")
    return accounting


def lifecycle_qacct_receipt_v1(
    *,
    accounting: LifecyclePopulationAccountingV1,
    lifecycle_control_sha256: str,
    configuration_sha256: str,
    profile_sha256: str,
    checkpoint_schema_version: str,
    capacity_accounting: Mapping[str, Any] | None = None,
    generation_turnover_enabled: bool | None = None,
    lifecycle_mode_abi: int | None = None,
    population_authority_source: str | None = None,
    legacy_turnover_final_live_authority: str | None = None,
    legacy_turnover_final_live_value: int | None = None,
    direct_resident_abi_variant: str = QACCT_DIRECT_RESIDENT_ABI_VARIANT_V1,
) -> dict[str, Any]:
    validate_lifecycle_population_accounting_v1(accounting)
    payload: dict[str, Any] = {
        "schema_version": QACCT_RECEIPT_SCHEMA_VERSION,
        "accounting_schema_version": accounting.schema_version,
        "checkpoint_contract_version": QACCT_CHECKPOINT_CONTRACT_VERSION,
        "checkpoint_schema_version": checkpoint_schema_version,
        "direct_resident_abi_variant": direct_resident_abi_variant,
        "lifecycle_control_sha256": lifecycle_control_sha256,
        "configuration_sha256": configuration_sha256,
        "profile_sha256": profile_sha256,
        "terminal_cause_taxonomy": list(PRIMARY_TERMINAL_CAUSES_V1),
        "cause_precedence": list(CAUSE_PRECEDENCE_V1),
        "counters": {
            name: getattr(accounting, name) for name in ACCOUNTING_COUNTER_NAMES_V1
        },
    }
    if capacity_accounting is not None:
        if (
            not isinstance(capacity_accounting, Mapping)
            or capacity_accounting.get("schema_version")
            != CAPACITY_ACCOUNTING_SCHEMA_VERSION
        ):
            raise _fail("capacity_accounting schema_version mismatch")
        required_ints = (
            "initial_population_configured",
            "physical_resident_capacity",
            "hard_cap_configured",
            "live_start",
            "live_end",
            "free_slots_start",
            "free_slots_end",
            "pending_terminal_slots_start",
            "pending_terminal_slots_end",
            "birth_attempts_total",
            "admitted_births",
            "failed_births_total",
            "failed_birth_reason_sum",
            "ecological_deaths",
            "senescence_deaths",
            "max_age_deaths",
            "legacy_turnover_culls",
            "hard_cap_removals",
            "integrity_admin_removals",
            "terminal_removals",
            "population_conservation_expected_live_end",
        )
        for name in required_ints:
            _strict_nonnegative_int(capacity_accounting.get(name), name=name)
        reason_authority_kind = capacity_accounting.get("reason_authority_kind")
        if reason_authority_kind not in {
            CAPACITY_REASON_AUTHORITY_FULL_PREFIX_V1,
            CAPACITY_REASON_AUTHORITY_DIAGNOSTIC_V1,
        }:
            raise _fail("capacity_accounting reason_authority_kind is invalid")
        reason_qualification_authority = capacity_accounting.get(
            "reason_qualification_authority"
        )
        if type(reason_qualification_authority) is not bool:
            raise _fail(
                "capacity_accounting reason_qualification_authority must be boolean"
            )
        if (
            reason_authority_kind == CAPACITY_REASON_AUTHORITY_FULL_PREFIX_V1
            and reason_qualification_authority is not True
        ):
            raise _fail("authoritative full-prefix reasons must qualify")
        if (
            reason_authority_kind == CAPACITY_REASON_AUTHORITY_DIAGNOSTIC_V1
            and reason_qualification_authority is not False
        ):
            raise _fail("diagnostic inferred reasons must not qualify")
        reason_denominator = capacity_accounting.get("reason_denominator")
        if not isinstance(reason_denominator, str) or not reason_denominator:
            raise _fail("capacity_accounting reason_denominator is invalid")
        reasons = capacity_accounting.get("failed_birth_reasons")
        if not isinstance(reasons, Mapping) or not reasons:
            raise _fail("capacity_accounting failed_birth_reasons must be a mapping")
        reason_sum = 0
        for name, value in reasons.items():
            if not isinstance(name, str) or not name:
                raise _fail("capacity_accounting failed-birth reason names are invalid")
            reason_sum += _strict_nonnegative_int(
                value,
                name=f"capacity_accounting.failed_birth_reasons.{name}",
            )
        if reason_sum != capacity_accounting["failed_birth_reason_sum"]:
            raise _fail("capacity_accounting failed-birth reason sum mismatch")
        if reason_sum != capacity_accounting["failed_births_total"]:
            raise _fail("capacity_accounting failed-birth reasons do not close")
        if (
            capacity_accounting["population_conservation_expected_live_end"]
            != capacity_accounting["live_end"]
        ):
            raise _fail("capacity_accounting population conservation mismatch")
        payload["capacity_accounting"] = dict(capacity_accounting)
        payload["capacity_accounting_reason_qualification_authority"] = (
            reason_qualification_authority
        )
    if generation_turnover_enabled is not None:
        if not isinstance(generation_turnover_enabled, bool):
            raise _fail("generation_turnover_enabled must be boolean when supplied")
        payload["generation_turnover_enabled"] = generation_turnover_enabled
    if lifecycle_mode_abi is not None:
        if isinstance(lifecycle_mode_abi, bool) or lifecycle_mode_abi not in {0, 1}:
            raise _fail("lifecycle_mode_abi must be 0 or 1 when supplied")
        payload["lifecycle_mode_abi"] = int(lifecycle_mode_abi)
    if population_authority_source is not None:
        if (
            not isinstance(population_authority_source, str)
            or not population_authority_source
        ):
            raise _fail("population_authority_source must be a non-empty string")
        payload["population_authority_source"] = population_authority_source
    if legacy_turnover_final_live_authority is not None:
        if (
            not isinstance(legacy_turnover_final_live_authority, str)
            or not legacy_turnover_final_live_authority
        ):
            raise _fail(
                "legacy_turnover_final_live_authority must be a non-empty string"
            )
        payload["legacy_turnover_final_live_authority"] = (
            legacy_turnover_final_live_authority
        )
    if legacy_turnover_final_live_value is not None:
        payload["legacy_turnover_final_live_value"] = _strict_nonnegative_int(
            legacy_turnover_final_live_value,
            name="legacy_turnover_final_live_value",
        )
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def validate_lifecycle_qacct_restore_v1(
    receipt: Mapping[str, Any],
    *,
    expected_checkpoint_schema_version: str,
    expected_configuration_sha256: str,
    expected_profile_sha256: str,
    expected_direct_resident_abi_variant: str = QACCT_DIRECT_RESIDENT_ABI_VARIANT_V1,
) -> Mapping[str, Any]:
    if receipt.get("schema_version") != QACCT_RECEIPT_SCHEMA_VERSION:
        raise _fail("receipt schema_version mismatch")
    expected = {
        "checkpoint_contract_version": QACCT_CHECKPOINT_CONTRACT_VERSION,
        "checkpoint_schema_version": expected_checkpoint_schema_version,
        "configuration_sha256": expected_configuration_sha256,
        "profile_sha256": expected_profile_sha256,
        "direct_resident_abi_variant": expected_direct_resident_abi_variant,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise _fail(f"receipt {key} mismatch")
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_sha256", None)
    if digest != canonical_sha256(unsigned):
        raise _fail("receipt_sha256 mismatch")
    counters = receipt.get("counters")
    if not isinstance(counters, Mapping):
        raise _fail("receipt counters are absent")
    if tuple(counters) != ACCOUNTING_COUNTER_NAMES_V1:
        raise _fail("receipt counter surface changed")
    return receipt


def accounting_counter_names_v1() -> tuple[str, ...]:
    return ACCOUNTING_COUNTER_NAMES_V1


__all__ = [
    "ACCOUNTING_COUNTER_NAMES_V1",
    "CAPACITY_ACCOUNTING_SCHEMA_VERSION",
    "CAPACITY_REASON_AUTHORITY_DIAGNOSTIC_V1",
    "CAPACITY_REASON_AUTHORITY_FULL_PREFIX_V1",
    "CAUSE_PRECEDENCE_V1",
    "PRIMARY_TERMINAL_CAUSES_V1",
    "QACCT_CHECKPOINT_CONTRACT_VERSION",
    "QACCT_DIRECT_RESIDENT_ABI_VARIANT_V1",
    "QACCT_RECEIPT_SCHEMA_VERSION",
    "QACCT_SCHEMA_VERSION",
    "LifecyclePopulationAccountingError",
    "LifecyclePopulationAccountingV1",
    "accounting_counter_names_v1",
    "build_lifecycle_population_accounting_v1",
    "lifecycle_qacct_receipt_v1",
    "primary_terminal_cause_v1",
    "validate_lifecycle_population_accounting_v1",
    "validate_lifecycle_qacct_restore_v1",
]
