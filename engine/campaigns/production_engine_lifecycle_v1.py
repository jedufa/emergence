"""Generic production-engine lifecycle configuration authority."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .canonical import canonical_sha256
from .production_engine_configuration import production_engine_configuration_v1
from .providers import CampaignProviderError

SCHEMA_VERSION: Final[str] = "eiel_production_engine_lifecycle_control_v2"
LEGACY_SYNCHRONIZED_TRUNCATION_V1: Final[str] = "legacy_synchronized_truncation_v1"
HYBRID_OVERLAPPING_V1: Final[str] = "hybrid_overlapping_v1"
_ADMITTED_LIFECYCLE_MODES: Final[frozenset[str]] = frozenset(
    {LEGACY_SYNCHRONIZED_TRUNCATION_V1, HYBRID_OVERLAPPING_V1}
)
LIFECYCLE_PARAMETER_NAMES: Final[tuple[str, ...]] = (
    "production_lifecycle.turnover_enabled",
    "production_lifecycle.ordinary_births_enabled",
    "production_lifecycle.turnover_cull_fraction",
    "production_lifecycle.turnover_parent_fraction",
)
OPTIONAL_LIFECYCLE_PARAMETER_NAMES: Final[tuple[str, ...]] = (
    "production_lifecycle.profile",
    "production_lifecycle.lifecycle_mode",
    "production_lifecycle.life_history_profile",
)
PATCH_FIXED_MLP_TURNOVER_ONLY_W32P256C256_V1: Final[str] = (
    "patch_fixed_mlp_turnover_only_w32p256c256_v1"
)
PATCH_FIXED_MLP_TURNOVER_ONLY_W32P256C512_V1: Final[str] = (
    "patch_fixed_mlp_turnover_only_w32p256c512_v1"
)
PATCH_FIXED_MLP_TURNOVER_ONLY_W32P512C768_V1: Final[str] = (
    "patch_fixed_mlp_turnover_only_w32p512c768_v1"
)
PATCH_FIXED_MLP_TURNOVER_ONLY_W32P512C1024_V1: Final[str] = (
    "patch_fixed_mlp_turnover_only_w32p512c1024_v1"
)
PATCH_FIXED_MLP_TURNOVER_ONLY_W16P64C128_V1: Final[str] = (
    "patch_fixed_mlp_turnover_only_w16p64c128_v1"
)
PATCH_FIXED_MLP_TURNOVER_ONLY_W16P128C256_V1: Final[str] = (
    "patch_fixed_mlp_turnover_only_w16p128c256_v1"
)
PATCH_FIXED_MLP_TURNOVER_ONLY_W32_PROFILE_SHAPES_V1: Final[
    dict[str, tuple[int, int, int]]
] = {
    PATCH_FIXED_MLP_TURNOVER_ONLY_W32P256C256_V1: (32, 256, 256),
    PATCH_FIXED_MLP_TURNOVER_ONLY_W32P256C512_V1: (32, 256, 512),
    PATCH_FIXED_MLP_TURNOVER_ONLY_W32P512C768_V1: (32, 512, 768),
    PATCH_FIXED_MLP_TURNOVER_ONLY_W32P512C1024_V1: (32, 512, 1024),
}
PATCH_FIXED_MLP_TURNOVER_ONLY_PROFILE_SHAPES_V1: Final[
    dict[str, tuple[int, int, int]]
] = {
    PATCH_FIXED_MLP_TURNOVER_ONLY_W16P64C128_V1: (16, 64, 128),
    PATCH_FIXED_MLP_TURNOVER_ONLY_W16P128C256_V1: (16, 128, 256),
    **PATCH_FIXED_MLP_TURNOVER_ONLY_W32_PROFILE_SHAPES_V1,
}
PATCH_FIXED_MLP_TURNOVER_ONLY_W32_PROFILE_IDS_V1: Final[tuple[str, ...]] = tuple(
    PATCH_FIXED_MLP_TURNOVER_ONLY_W32_PROFILE_SHAPES_V1
)
PATCH_FIXED_MLP_TURNOVER_ONLY_PROFILE_IDS_V1: Final[tuple[str, ...]] = tuple(
    PATCH_FIXED_MLP_TURNOVER_ONLY_PROFILE_SHAPES_V1
)
_EXPECTED_LIFECYCLE_VALUES: Final[dict[str, Any]] = {
    "production_lifecycle.turnover_enabled": True,
    "production_lifecycle.turnover_parent_fraction": 1.0,
}
_RETIRED_PARAMETER_PREFIXES: Final[tuple[str, ...]] = (
    "inheritance_cross_path.",
    "recomb72.",
)
_ADMITTED_WORLD_ENVELOPES: Final[frozenset[tuple[str, tuple[str, ...], str]]] = (
    frozenset(
        {
            ("patch", (), "fixed_mlp"),
            ("patch", ("eed_resource_impact",), "fixed_mlp"),
            (
                "patch",
                ("eed_resource_impact", "jungle", "wdl_delayed_hazard"),
                "fixed_mlp",
            ),
            ("patch", ("jungle", "wdl_delayed_hazard"), "fixed_mlp"),
            ("patch", ("wdl_delayed_hazard",), "fixed_mlp"),
        }
    )
)
LIFE_HISTORY_PROFILE_INERT_V1: Final[str] = "legacy_inert_v1"
LIFE_HISTORY_PROFILE_STAGE1R_QUALIFICATION_V1: Final[str] = (
    "stage1r_tick_life_history_v1"
)
LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1: Final[str] = (
    "stage1s_overlap_tick_life_history_v1"
)
PRODUCTION_DEFAULT_LIFECYCLE_MODE_V1: Final[str] = HYBRID_OVERLAPPING_V1
PRODUCTION_DEFAULT_LIFE_HISTORY_PROFILE_V1: Final[str] = (
    LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1
)
_ADMITTED_LIFE_HISTORY_PROFILES: Final[frozenset[str]] = frozenset(
    {
        LIFE_HISTORY_PROFILE_INERT_V1,
        LIFE_HISTORY_PROFILE_STAGE1R_QUALIFICATION_V1,
        LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1,
    }
)
_ADMITTED_HYBRID_INITIAL_POPULATIONS_V1: Final[frozenset[int]] = frozenset(
    {64, 192, 256, 512}
)


def _fail(message: str) -> CampaignProviderError:
    return CampaignProviderError(f"production lifecycle control {message}")


@dataclass(frozen=True, slots=True)
class ProductionEngineLifecycleControlV1:
    """Frozen neutral turnover policy selected by campaign configuration."""

    turnover_enabled: bool
    ordinary_births_enabled: bool
    lifecycle_mode: str
    turnover_cull_fraction: float
    turnover_parent_fraction: float
    selection_mode: str
    turnover_selection_mode_abi: int
    target_population: int
    retained_population: int
    refill_population: int
    genome_v2_mutation_regimes_q10: tuple[tuple[int, int], ...]
    life_history_profile_id: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "turnover": {
                "enabled": self.turnover_enabled,
                "cull_fraction": self.turnover_cull_fraction,
                "parent_fraction": self.turnover_parent_fraction,
                "target_population": self.target_population,
                "retained_population": self.retained_population,
                "refill_population": self.refill_population,
            },
            "ordinary_births_enabled": self.ordinary_births_enabled,
            "lifecycle_mode": self.lifecycle_mode,
            "selection_mode": self.selection_mode,
            "turnover_selection_mode_abi": self.turnover_selection_mode_abi,
            "genome_v2_mutation": {
                "enabled": self.genome_v2_mutation_enabled,
                "admitted_regimes_q10": [
                    {"rate_q10": rate, "step_q10": step}
                    for rate, step in self.genome_v2_mutation_regimes_q10
                ],
                "regime_policy": "fixed_q10_off_control_or_low_admitted",
            },
            "life_history": production_lifecycle_life_history_profile_v1(
                self.life_history_profile_id
            ),
        }

    @property
    def control_sha256(self) -> str:
        return canonical_sha256(self.payload())

    @property
    def genome_v2_mutation_enabled(self) -> bool:
        return any(
            rate > 0 and step > 0 for rate, step in self.genome_v2_mutation_regimes_q10
        )


def _cell_parameters(plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_cells = plan.get("cells")
    if (
        not isinstance(raw_cells, Sequence)
        or isinstance(raw_cells, (str, bytes))
        or not raw_cells
    ):
        raise _fail("requires at least one compiled cell")
    parameters: list[Mapping[str, Any]] = []
    for ordinal, cell in enumerate(raw_cells):
        row = cell.get("parameters") if isinstance(cell, Mapping) else None
        if not isinstance(row, Mapping):
            raise _fail(f"cell {ordinal} lacks parameters")
        parameters.append(row)
    return tuple(parameters)


def _reject_retired_authority(parameters: Mapping[str, Any]) -> None:
    for raw_name, value in parameters.items():
        name = str(raw_name)
        if name.startswith(_RETIRED_PARAMETER_PREFIXES):
            raise _fail(f"contains retired parameter {name!r}")
        if "linked_parent_copy_v1" in name or (
            isinstance(value, str) and "linked_parent_copy_v1" in value
        ):
            raise _fail("contains retired linked_parent_copy_v1 authority")


def _lifecycle_mode(parameters: Mapping[str, Any]) -> str:
    raw_mode = parameters.get(
        "production_lifecycle.lifecycle_mode",
        PRODUCTION_DEFAULT_LIFECYCLE_MODE_V1,
    )
    if type(raw_mode) is not str or raw_mode not in _ADMITTED_LIFECYCLE_MODES:
        raise _fail("requires admitted production_lifecycle.lifecycle_mode")
    return raw_mode


def _expected_cull_fraction(mode: str) -> float:
    if mode == HYBRID_OVERLAPPING_V1:
        return 0.0
    return 0.5


def _selection_mode_for_lifecycle(
    mode: str, *, target_population: int
) -> tuple[str, int, int, int]:
    if mode == HYBRID_OVERLAPPING_V1:
        return "genotype_blind_neutral", 1, target_population, 0
    retained = target_population // 2
    return "genotype_blind_neutral", 1, retained, target_population - retained


def _lifecycle_profile_id(parameters: Mapping[str, Any]) -> str | None:
    raw_profile = parameters.get("production_lifecycle.profile")
    if raw_profile is None:
        return None
    if raw_profile not in PATCH_FIXED_MLP_TURNOVER_ONLY_PROFILE_SHAPES_V1:
        raise _fail("requires admitted production_lifecycle.profile")
    return raw_profile


def _life_history_profile_id(parameters: Mapping[str, Any]) -> str:
    raw_profile = parameters.get(
        "production_lifecycle.life_history_profile",
        PRODUCTION_DEFAULT_LIFE_HISTORY_PROFILE_V1,
    )
    if (
        type(raw_profile) is not str
        or raw_profile not in _ADMITTED_LIFE_HISTORY_PROFILES
    ):
        raise _fail("requires admitted production_lifecycle.life_history_profile")
    return raw_profile


def production_lifecycle_life_history_profile_v1(profile_id: str) -> dict[str, Any]:
    """Return one frozen tick-life-history profile for the production engine."""

    if profile_id == LIFE_HISTORY_PROFILE_INERT_V1:
        return {
            "profile_id": profile_id,
            "schema_version": "eiel_gv2_lifecycle_v2_life_history_profile_v1",
            "developmental_maturity_age": 20,
            "life_history_mode": 0,
            "life_history_maturity_start_ticks": 0,
            "life_history_maturity_rise_ticks": 0,
            "life_history_fertility_decline_start_ticks": 0,
            "life_history_fertility_decline_ticks": 0,
            "life_history_elder_fertility_floor": 1.0,
            "life_history_senescence_start_ticks": 0,
            "life_history_senescence_ramp_ticks": 0,
            "life_history_senescence_energy_cost": 0.0,
            "life_history_senescence_health_cost": 0.0,
            "life_history_max_age_ticks": 0,
        }
    if profile_id == LIFE_HISTORY_PROFILE_STAGE1R_QUALIFICATION_V1:
        return {
            "profile_id": profile_id,
            "schema_version": "eiel_gv2_lifecycle_v2_life_history_profile_v1",
            "developmental_maturity_age": 20,
            "life_history_mode": 1,
            "life_history_maturity_start_ticks": 20,
            "life_history_maturity_rise_ticks": 80,
            "life_history_fertility_decline_start_ticks": 1200,
            "life_history_fertility_decline_ticks": 800,
            "life_history_elder_fertility_floor": 0.05,
            "life_history_senescence_start_ticks": 1600,
            "life_history_senescence_ramp_ticks": 800,
            "life_history_senescence_energy_cost": 0.002,
            "life_history_senescence_health_cost": 0.0005,
            "life_history_max_age_ticks": 2400,
        }
    if profile_id == LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1:
        return {
            "profile_id": profile_id,
            "schema_version": "eiel_gv2_lifecycle_v2_life_history_profile_v1",
            "developmental_maturity_age": 20,
            "life_history_mode": 1,
            "life_history_maturity_start_ticks": 20,
            "life_history_maturity_rise_ticks": 80,
            "life_history_fertility_decline_start_ticks": 1200,
            "life_history_fertility_decline_ticks": 800,
            "life_history_elder_fertility_floor": 0.35,
            "life_history_senescence_start_ticks": 1600,
            "life_history_senescence_ramp_ticks": 800,
            "life_history_senescence_energy_cost": 0.002,
            "life_history_senescence_health_cost": 0.0005,
            "life_history_max_age_ticks": 2400,
        }
    raise _fail("requires admitted production_lifecycle.life_history_profile")


def production_engine_lifecycle_control_v1(
    plan: Mapping[str, Any],
) -> ProductionEngineLifecycleControlV1 | None:
    """Return the exact frozen turnover control, or ``None`` when unselected."""

    cell_parameters = _cell_parameters(plan)
    selected = tuple(
        any(
            name in parameters
            for name in (
                *LIFECYCLE_PARAMETER_NAMES,
                *OPTIONAL_LIFECYCLE_PARAMETER_NAMES,
            )
        )
        for parameters in cell_parameters
    )
    if not any(selected):
        return None
    if not all(selected):
        raise _fail("is not selected identically by every cell")

    for parameters in cell_parameters:
        _reject_retired_authority(parameters)
        missing = tuple(
            name for name in LIFECYCLE_PARAMETER_NAMES if name not in parameters
        )
        if missing:
            raise _fail(f"lacks required parameters {missing!r}")
        ordinary_births_enabled = parameters[
            "production_lifecycle.ordinary_births_enabled"
        ]
        if type(ordinary_births_enabled) is not bool:
            raise _fail(
                "requires production_lifecycle.ordinary_births_enabled to be bool"
            )
        mode = _lifecycle_mode(parameters)
        for name, expected in _EXPECTED_LIFECYCLE_VALUES.items():
            actual = parameters[name]
            if type(actual) is not type(expected) or actual != expected:
                raise _fail(f"requires {name}={expected!r}")
        expected_cull_fraction = _expected_cull_fraction(mode)
        actual_cull_fraction = parameters["production_lifecycle.turnover_cull_fraction"]
        if (
            type(actual_cull_fraction) is not float
            or actual_cull_fraction != expected_cull_fraction
        ):
            raise _fail(
                "requires production_lifecycle.turnover_cull_fraction="
                f"{expected_cull_fraction!r} for lifecycle mode {mode!r}"
            )

    reference = {name: cell_parameters[0][name] for name in LIFECYCLE_PARAMETER_NAMES}
    reference["production_lifecycle.lifecycle_mode"] = _lifecycle_mode(
        cell_parameters[0]
    )
    reference["production_lifecycle.life_history_profile"] = _life_history_profile_id(
        cell_parameters[0]
    )
    reference["production_lifecycle.profile"] = _lifecycle_profile_id(
        cell_parameters[0]
    )
    if any(
        {
            **{name: parameters[name] for name in LIFECYCLE_PARAMETER_NAMES},
            "production_lifecycle.lifecycle_mode": _lifecycle_mode(parameters),
            "production_lifecycle.life_history_profile": _life_history_profile_id(
                parameters
            ),
            "production_lifecycle.profile": _lifecycle_profile_id(parameters),
        }
        != reference
        for parameters in cell_parameters[1:]
    ):
        raise _fail("differs across compiled cells")

    configuration = production_engine_configuration_v1(plan)
    world_envelope = (
        configuration.world_base,
        configuration.world_overlays,
        configuration.controller_family,
    )
    if (
        world_envelope not in _ADMITTED_WORLD_ENVELOPES
        or configuration.selection_mode != "genotype_blind_neutral"
    ):
        raise _fail(
            "requires admitted patch/fixed_mlp turnover envelope and "
            "genotype_blind_neutral"
        )

    lifecycle_mode = str(reference["production_lifecycle.lifecycle_mode"])
    life_history_profile_id = str(
        reference["production_lifecycle.life_history_profile"]
    )
    profile_id = reference["production_lifecycle.profile"]
    if (
        lifecycle_mode == LEGACY_SYNCHRONIZED_TRUNCATION_V1
        and life_history_profile_id != LIFE_HISTORY_PROFILE_INERT_V1
    ):
        raise _fail("requires inert life history for legacy synchronized mode")

    target_population = 64
    if profile_id in PATCH_FIXED_MLP_TURNOVER_ONLY_PROFILE_SHAPES_V1:
        grid_size, initial_population, resident_capacity = (
            PATCH_FIXED_MLP_TURNOVER_ONLY_PROFILE_SHAPES_V1[profile_id]
        )
        if (
            lifecycle_mode != LEGACY_SYNCHRONIZED_TRUNCATION_V1
            or reference["production_lifecycle.ordinary_births_enabled"] is not False
            or life_history_profile_id != LIFE_HISTORY_PROFILE_INERT_V1
        ):
            raise _fail(
                "requires exact turnover-only legacy no-birth lifecycle"
            )
        for parameters in cell_parameters:
            required = {
                "patch.grid_size": grid_size,
                "patch.initial_population": initial_population,
                "patch.capacity": resident_capacity,
            }
            for name, expected in required.items():
                actual = parameters.get(name)
                if type(actual) is not type(expected) or actual != expected:
                    raise _fail(f"requires {name}={expected!r}")
            if any(str(name).startswith("genome_upgrade_v2.") for name in parameters):
                raise _fail("turnover-only profile excludes Genome V2")
        target_population = initial_population
        mutation_regimes = ((0, 0),)
    else:
        hybrid_initial_populations: set[int] = set()
        for parameters in cell_parameters:
            required = {
                "genome_upgrade_v2.enable": True,
                "genome_upgrade_v2.profile": "genome_upgrade_v2_fixed_q10_v1",
            }
            for name, expected in required.items():
                actual = parameters.get(name)
                if type(actual) is not type(expected) or actual != expected:
                    raise _fail(f"requires {name}={expected!r}")
            initial_population = parameters.get("patch.initial_population")
            if (
                type(initial_population) is not int
                or initial_population not in _ADMITTED_HYBRID_INITIAL_POPULATIONS_V1
            ):
                raise _fail(
                    "requires patch.initial_population in "
                    f"{sorted(_ADMITTED_HYBRID_INITIAL_POPULATIONS_V1)!r}"
                )
            hybrid_initial_populations.add(initial_population)
            rate_q10 = parameters.get("genome_upgrade_v2.mutation_rate_q10")
            step_q10 = parameters.get("genome_upgrade_v2.mutation_step_q10")
            if (
                type(rate_q10) is not int
                or type(step_q10) is not int
                or not 0 <= rate_q10 <= 1024
                or not 0 <= step_q10 <= 1024
            ):
                raise _fail("requires typed Genome V2 Q10 mutation bounds")
            if (rate_q10 == 0) != (step_q10 == 0):
                raise _fail(
                    "requires Genome V2 mutation rate/step to be both zero or both positive"
                )
            capacity = parameters.get("patch.capacity")
            if type(capacity) is not int or capacity < initial_population:
                raise _fail("requires patch.capacity >= patch.initial_population")
        if len(hybrid_initial_populations) != 1:
            raise _fail("requires uniform patch.initial_population")
        target_population = next(iter(hybrid_initial_populations))
        mutation_regimes = tuple(
            sorted(
                {
                    (
                        int(parameters["genome_upgrade_v2.mutation_rate_q10"]),
                        int(parameters["genome_upgrade_v2.mutation_step_q10"]),
                    )
                    for parameters in cell_parameters
                }
            )
        )
    selection_mode, selection_mode_abi, retained_population, refill_population = (
        _selection_mode_for_lifecycle(
            lifecycle_mode, target_population=target_population
        )
    )
    turnover_cull_fraction = _expected_cull_fraction(lifecycle_mode)

    return ProductionEngineLifecycleControlV1(
        turnover_enabled=True,
        ordinary_births_enabled=bool(
            reference["production_lifecycle.ordinary_births_enabled"]
        ),
        lifecycle_mode=lifecycle_mode,
        turnover_cull_fraction=turnover_cull_fraction,
        turnover_parent_fraction=1.0,
        selection_mode=selection_mode,
        turnover_selection_mode_abi=selection_mode_abi,
        target_population=target_population,
        retained_population=retained_population,
        refill_population=refill_population,
        genome_v2_mutation_regimes_q10=mutation_regimes,
        life_history_profile_id=life_history_profile_id,
    )


__all__ = [
    "LIFECYCLE_PARAMETER_NAMES",
    "OPTIONAL_LIFECYCLE_PARAMETER_NAMES",
    "LEGACY_SYNCHRONIZED_TRUNCATION_V1",
    "HYBRID_OVERLAPPING_V1",
    "LIFE_HISTORY_PROFILE_INERT_V1",
    "LIFE_HISTORY_PROFILE_STAGE1R_QUALIFICATION_V1",
    "LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1",
    "PRODUCTION_DEFAULT_LIFECYCLE_MODE_V1",
    "PRODUCTION_DEFAULT_LIFE_HISTORY_PROFILE_V1",
    "PATCH_FIXED_MLP_TURNOVER_ONLY_W16P64C128_V1",
    "PATCH_FIXED_MLP_TURNOVER_ONLY_W16P128C256_V1",
    "PATCH_FIXED_MLP_TURNOVER_ONLY_PROFILE_IDS_V1",
    "PATCH_FIXED_MLP_TURNOVER_ONLY_PROFILE_SHAPES_V1",
    "PATCH_FIXED_MLP_TURNOVER_ONLY_W32P256C256_V1",
    "PATCH_FIXED_MLP_TURNOVER_ONLY_W32P256C512_V1",
    "PATCH_FIXED_MLP_TURNOVER_ONLY_W32P512C768_V1",
    "PATCH_FIXED_MLP_TURNOVER_ONLY_W32P512C1024_V1",
    "PATCH_FIXED_MLP_TURNOVER_ONLY_W32_PROFILE_IDS_V1",
    "PATCH_FIXED_MLP_TURNOVER_ONLY_W32_PROFILE_SHAPES_V1",
    "ProductionEngineLifecycleControlV1",
    "SCHEMA_VERSION",
    "production_engine_lifecycle_control_v1",
    "production_lifecycle_life_history_profile_v1",
]
