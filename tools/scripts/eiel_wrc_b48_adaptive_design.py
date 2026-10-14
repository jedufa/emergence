#!/usr/bin/env python3
"""Generate the frozen 16-configuration B48 adaptive experiment design."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Final, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.registry import default_profile_registry  # noqa: E402
from engine.campaigns.canonical import canonical_sha256  # noqa: E402
from engine.campaigns.compiler import compile_campaign, load_campaign_spec  # noqa: E402
from engine.campaigns.orchestration import atomic_write_json  # noqa: E402
from tools.scripts.eiel_wrc_b48_adaptive_paths import (  # noqa: E402
    INITIAL_DESIGN_PATH,
)


PROFILE_ID: Final[str] = "eiel-production-runtime-v1"
OUTPUT_FIELDS: Final[tuple[str, ...]] = (
    "config_id",
    "seed_tuple",
    "physical_cell",
    "parameter",
    "value",
    "source_WRC_ID",
    "evidence_rationale",
    "expected_regime",
)
SEED_TUPLES: Final[tuple[tuple[str, int, int], ...]] = (
    ("S1", 101, 1001),
    ("S2", 202, 2002),
    ("S3", 303, 3003),
)


# Every row is a covered EXPERIMENT_VARIABLE. Activation controls are explicit
# so the three replicates of a configuration share one complete active bundle.
BASE_PANEL: Final[dict[str, Any]] = {
    "env_cfg.jungle_ecological_recovery_enable": True,
    "env_cfg.jungle_recovery_activity_base_growth": 0.0,
    "env_cfg.jungle_recovery_activity_decay_per_step": 0.02,
    "env_cfg.jungle_recovery_activity_fertility_weight": 0.02,
    "env_cfg.jungle_recovery_activity_waste_penalty_weight": 0.05,
    "env_cfg.move_cost": 0.2,
    "env_cfg.patch_movement_cost_apply_enable": True,
    "env_cfg.patch_movement_cost_energy_scale": 1.0,
    "env_cfg.patch_food_gather_amount": 1.0,
    "env_cfg.patch_food_regrow_per_step": 0.02,
    "env_cfg.jungle_predator_pressure_enable": True,
    "env_cfg.jungle_predator_base_risk": 0.0,
    "env_cfg.jungle_predator_health_damage": 0.25,
    "env_cfg.jungle_predator_risk_cap": 0.05,
    "env_cfg.reproduce_cooldown_steps": 35,
    "env_cfg.reproduce_min_energy_buffer": 20.0,
    "env_cfg.reproduce_parent_energy_share": 0.2,
    "env_cfg.reproduce_try_prob": 0.15,
    "env_cfg.wdl_delayed_hazard_event_period_steps": 20,
    "wdl.active_duration_ticks": 6,
    "wdl.delay_ticks": 8,
    "env_cfg.patch_world_waste_consequence_enable": True,
}

WRC_IDS: Final[dict[str, str]] = {
    "env_cfg.jungle_ecological_recovery_enable": "P011",
    "env_cfg.jungle_recovery_activity_base_growth": "P012",
    "env_cfg.jungle_recovery_activity_decay_per_step": "P014",
    "env_cfg.jungle_recovery_activity_fertility_weight": "P016",
    "env_cfg.jungle_recovery_activity_waste_penalty_weight": "P018",
    "env_cfg.move_cost": "P023",
    "env_cfg.patch_food_gather_amount": "P030",
    "env_cfg.patch_food_regrow_per_step": "P032",
    "env_cfg.jungle_predator_base_risk": "P039",
    "env_cfg.jungle_predator_health_damage": "P042",
    "env_cfg.jungle_predator_pressure_enable": "P043",
    "env_cfg.jungle_predator_risk_cap": "P044",
    "env_cfg.patch_movement_cost_apply_enable": "P056",
    "env_cfg.patch_movement_cost_energy_scale": "P057",
    "env_cfg.reproduce_cooldown_steps": "P058",
    "env_cfg.reproduce_min_energy_buffer": "P059",
    "env_cfg.reproduce_parent_energy_share": "P060",
    "env_cfg.reproduce_try_prob": "P061",
    "env_cfg.wdl_delayed_hazard_event_period_steps": "P067",
    "wdl.active_duration_ticks": "P070",
    "wdl.delay_ticks": "P073",
    "env_cfg.patch_world_waste_consequence_enable": "P083",
}

PARAMETER_EVIDENCE: Final[dict[str, str]] = {
    "P011": "strong active recovery switch in the P011-P083 synthesis",
    "P012": "first effective 0.1; monotonic recovery response over tested 0-4",
    "P014": "active decay threshold at 0.02; tested 0-1",
    "P016": "active fertility-weight threshold near 0.04; tested 0-0.32",
    "P018": "active waste penalty from 0.025; tested 0-0.8",
    "P023": "movement response begins at 0.2 with discrimination at 0.2-0.4",
    "P030": "strong food-throughput response with threshold near 0.5; tested 0-4",
    "P032": "active regrowth response; tested 0-0.32",
    "P039": "predator response threshold at 0.1; tested 0-1",
    "P042": "predator damage active from 0.125; tested 0-4",
    "P043": "strong active predator-pressure switch in the synthesis",
    "P044": "strong risk-cap response with threshold at 0.05; tested 0-1",
    "P056": "covered movement-cost activation switch",
    "P057": "movement burden active from scale 2; tested 0-4",
    "P058": "reproduction cooldown spans favorable 18 to saturation near 70",
    "P059": "strong non-monotonic reproduction buffer response; tested 0-80",
    "P060": "strong non-monotonic parent-share response; tested 0-1",
    "P061": "strong reproduction-attempt response; tested 0-1",
    "P067": "hazard-period response discriminates around 20-40; tested 1-80",
    "P070": "non-monotonic hazard-duration regimes; tested 1-96",
    "P073": "non-monotonic WDL delay regimes around 8; tested 1-8",
    "P083": "active waste/crowding consequence switch in the synthesis",
}


CONFIGURATIONS: Final[tuple[tuple[str, str, Mapping[str, Any]], ...]] = (
    ("C01", "reference-like full-stack regime", {}),
    (
        "C02",
        "moderate ecological-recovery support",
        {
            "env_cfg.jungle_recovery_activity_base_growth": 0.1,
            "env_cfg.jungle_recovery_activity_fertility_weight": 0.04,
        },
    ),
    (
        "C03",
        "moderate resource-throughput support",
        {
            "env_cfg.patch_food_gather_amount": 2.0,
            "env_cfg.patch_food_regrow_per_step": 0.08,
        },
    ),
    (
        "C04",
        "moderately elevated movement burden",
        {
            "env_cfg.move_cost": 0.4,
            "env_cfg.patch_movement_cost_energy_scale": 2.0,
        },
    ),
    (
        "C05",
        "high but bounded movement burden",
        {
            "env_cfg.move_cost": 0.8,
            "env_cfg.patch_movement_cost_energy_scale": 2.0,
        },
    ),
    (
        "C06",
        "severe pre-collapse movement probe",
        {
            "env_cfg.move_cost": 1.6,
            "env_cfg.patch_movement_cost_energy_scale": 4.0,
        },
    ),
    (
        "C07",
        "moderate predator pressure",
        {
            "env_cfg.jungle_predator_base_risk": 0.1,
            "env_cfg.jungle_predator_risk_cap": 0.1,
        },
    ),
    (
        "C08",
        "high predator pressure and damage",
        {
            "env_cfg.jungle_predator_base_risk": 0.2,
            "env_cfg.jungle_predator_health_damage": 0.5,
            "env_cfg.jungle_predator_risk_cap": 0.2,
        },
    ),
    (
        "C09",
        "bounded severe predator regime",
        {
            "env_cfg.jungle_predator_base_risk": 0.4,
            "env_cfg.jungle_predator_health_damage": 1.0,
            "env_cfg.jungle_predator_risk_cap": 0.4,
        },
    ),
    (
        "C10",
        "strong ecological recovery support",
        {
            "env_cfg.jungle_recovery_activity_base_growth": 0.4,
            "env_cfg.jungle_recovery_activity_decay_per_step": 0.0,
            "env_cfg.jungle_recovery_activity_fertility_weight": 0.08,
            "env_cfg.jungle_recovery_activity_waste_penalty_weight": 0.025,
        },
    ),
    (
        "C11",
        "strong resource-throughput support",
        {
            "env_cfg.patch_food_gather_amount": 4.0,
            "env_cfg.patch_food_regrow_per_step": 0.32,
        },
    ),
    (
        "C12",
        "reproduction-favorable opportunity regime",
        {
            "env_cfg.reproduce_cooldown_steps": 18,
            "env_cfg.reproduce_min_energy_buffer": 0.0,
            "env_cfg.reproduce_try_prob": 0.3,
        },
    ),
    (
        "C13",
        "reproduction-constrained regime",
        {
            "env_cfg.reproduce_cooldown_steps": 70,
            "env_cfg.reproduce_min_energy_buffer": 40.0,
            "env_cfg.reproduce_parent_energy_share": 0.1,
            "env_cfg.reproduce_try_prob": 0.1,
        },
    ),
    (
        "C14",
        "movement and predator pressure compensated by recovery",
        {
            "env_cfg.move_cost": 0.8,
            "env_cfg.patch_movement_cost_energy_scale": 2.0,
            "env_cfg.jungle_predator_base_risk": 0.2,
            "env_cfg.jungle_predator_health_damage": 0.5,
            "env_cfg.jungle_predator_risk_cap": 0.2,
            "env_cfg.jungle_recovery_activity_base_growth": 0.1,
            "env_cfg.patch_food_gather_amount": 2.0,
        },
    ),
    (
        "C15",
        "resource support opposed by reproduction pressure",
        {
            "env_cfg.patch_food_gather_amount": 4.0,
            "env_cfg.patch_food_regrow_per_step": 0.08,
            "env_cfg.reproduce_cooldown_steps": 70,
            "env_cfg.reproduce_min_energy_buffer": 40.0,
        },
    ),
    (
        "C16",
        "unusual high-information compensated severe regime",
        {
            "env_cfg.move_cost": 1.6,
            "env_cfg.patch_movement_cost_energy_scale": 4.0,
            "env_cfg.jungle_predator_base_risk": 0.2,
            "env_cfg.jungle_predator_health_damage": 1.0,
            "env_cfg.jungle_predator_risk_cap": 0.2,
            "env_cfg.jungle_recovery_activity_base_growth": 0.4,
            "env_cfg.jungle_recovery_activity_decay_per_step": 0.0,
            "env_cfg.jungle_recovery_activity_fertility_weight": 0.08,
            "env_cfg.patch_food_gather_amount": 4.0,
            "env_cfg.patch_food_regrow_per_step": 0.32,
            "env_cfg.patch_world_waste_consequence_enable": True,
        },
    ),
)


class AdaptiveDesignError(ValueError):
    """The proposed design does not preserve the qualified WRC contract."""


def configuration_bundles() -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    for config_id, _regime, overrides in CONFIGURATIONS:
        bundle = {**BASE_PANEL, **dict(overrides)}
        if config_id in bundles:
            raise AdaptiveDesignError(f"duplicate configuration {config_id}")
        bundles[config_id] = bundle
    return bundles


def validate_design() -> None:
    profiles = {
        profile.profile_id: profile for profile in default_profile_registry().profiles()
    }
    profile = profiles.get(PROFILE_ID)
    if profile is None:
        raise AdaptiveDesignError(f"missing runtime profile {PROFILE_ID}")
    bundles = configuration_bundles()
    if len(bundles) != 16 or len(SEED_TUPLES) != 3:
        raise AdaptiveDesignError("design must be exactly 16 configurations x 3 seeds")
    if set(BASE_PANEL) != set(WRC_IDS):
        raise AdaptiveDesignError("base panel and WRC mapping differ")
    prohibited = {"P020", "P045", "P074", "P076", "P081"}
    if prohibited & set(WRC_IDS.values()):
        raise AdaptiveDesignError("design uses an unresolved control or telemetry row")
    for config_id, bundle in bundles.items():
        if set(bundle) != set(BASE_PANEL):
            raise AdaptiveDesignError(f"{config_id} does not define the complete panel")
        for parameter, value in bundle.items():
            contract = profile.parameter_contracts.get(parameter)
            if (
                contract is None
                or contract.variability_semantics != "experiment_variable"
                or contract.qualification_affecting
                or not contract.per_cell
                or not contract.accepts(value)
            ):
                raise AdaptiveDesignError(
                    f"{config_id} value is outside the qualified envelope: "
                    f"{parameter}={value!r}"
                )
        delay = bundle["wdl.delay_ticks"]
        active_duration = bundle["wdl.active_duration_ticks"]
        event_period = bundle["env_cfg.wdl_delayed_hazard_event_period_steps"]
        if event_period < delay + active_duration + 1:
            raise AdaptiveDesignError(
                f"{config_id} WDL event period is shorter than its delay and "
                "active-duration context"
            )
        if (
            bundle["env_cfg.patch_movement_cost_apply_enable"]
            is not bundle["env_cfg.patch_world_waste_consequence_enable"]
        ):
            raise AdaptiveDesignError(
                f"{config_id} leaves the exact Patch P2 toggle context"
            )


def build_rows() -> list[dict[str, str | int]]:
    validate_design()
    regimes = {config_id: regime for config_id, regime, _ in CONFIGURATIONS}
    rows: list[dict[str, str | int]] = []
    physical_cell = 0
    for config_id, bundle in configuration_bundles().items():
        for seed_id, ecology_seed, task_seed in SEED_TUPLES:
            seed_tuple = json.dumps(
                {
                    "id": seed_id,
                    "ecology": ecology_seed,
                    "task": task_seed,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            for parameter, value in bundle.items():
                wrc_id = WRC_IDS[parameter]
                rows.append(
                    {
                        "config_id": config_id,
                        "seed_tuple": seed_tuple,
                        "physical_cell": physical_cell,
                        "parameter": parameter,
                        "value": json.dumps(value, separators=(",", ":")),
                        "source_WRC_ID": wrc_id,
                        "evidence_rationale": (
                            f"{regimes[config_id]}; {PARAMETER_EVIDENCE[wrc_id]}"
                        ),
                        "expected_regime": regimes[config_id],
                    }
                )
            physical_cell += 1
    if physical_cell != 48:
        raise AdaptiveDesignError("design did not occupy exactly 48 physical cells")
    return rows


def materialize_initial_campaign_spec(
    base_spec: Mapping[str, Any], *, runtime_root: Path
) -> dict[str, Any]:
    """Project the committed paired-replicate design into one exact B48 spec."""

    validate_design()
    spec = deepcopy(dict(base_spec))
    spec["campaign_id"] = "b48-wrc-adaptive-g25-g125-20260811"
    spec["description"] = (
        "Exploratory B48 WRC adaptive experiment, initial C1 segment through G25."
    )
    spec["no_claims"] = (
        "Exploratory execution; no scientific or production claim follows without "
        "the declared checkpoint analyses and final review."
    )
    parameters = dict(spec.get("parameters") or {})
    # The former 3-way founder axis is fixed as neutral context so the committed
    # design remains exactly 16 configurations x 3 paired seed replicates.
    parameters["eed.founder_resource_axis_profile"] = "symmetric_mix"
    spec["parameters"] = parameters
    bundles = configuration_bundles()
    dimensions: list[dict[str, Any]] = []
    parameter_bindings: dict[str, str] = {}
    dimension_by_parameter: dict[str, str] = {}
    for ordinal, parameter in enumerate(sorted(BASE_PANEL)):
        dimension_id = f"wrc_parameter_{ordinal:02d}"
        values_by_identity: dict[str, Any] = {}
        for bundle in bundles.values():
            value = bundle[parameter]
            identity = f"{type(value).__name__}:{json.dumps(value, sort_keys=True)}"
            values_by_identity[identity] = value
        dimensions.append(
            {
                "id": dimension_id,
                "values": [values_by_identity[key] for key in sorted(values_by_identity)],
            }
        )
        parameter_bindings[dimension_id] = parameter
        dimension_by_parameter[parameter] = dimension_id
    dimensions.extend(
        [
            {"id": "ecology_seed", "values": [row[1] for row in SEED_TUPLES]},
            {"id": "task_seed", "values": [row[2] for row in SEED_TUPLES]},
        ]
    )
    include: list[dict[str, Any]] = []
    for config_id, bundle in bundles.items():
        del config_id
        for _seed_id, ecology_seed, task_seed in SEED_TUPLES:
            include.append(
                {
                    **{
                        dimension_by_parameter[parameter]: value
                        for parameter, value in bundle.items()
                    },
                    "ecology_seed": ecology_seed,
                    "task_seed": task_seed,
                }
            )
    spec["matrix"] = {
        "dimensions": dimensions,
        "parameter_bindings": parameter_bindings,
        "seed_bindings": {"ecology_seed": "ecology", "task_seed": "task"},
        "include": include,
        "exclude": [],
        "expected_cells": 48,
        "max_cells": 48,
    }
    execution = dict(spec.get("execution") or {})
    execution["generations"] = 25
    execution["ticks_per_generation"] = 160
    execution["batch"] = {"mode": "automatic", "maximum": 48}
    execution["cancellation"] = False
    execution["technical_gate"] = {"selectors": [], "partition": False}
    spec["execution"] = execution
    spec["checkpoints"] = {
        "milestones": [25],
        "digest_every_generation": True,
        "storage_mode": "disk",
    }
    spec["comparisons"] = {
        "fresh_repeats": 0,
        "continuation_splits": [],
    }
    output = dict(spec.get("output") or {})
    output["runtime_root"] = str(runtime_root.resolve())
    output["chronicle_profile"] = "scientific_full"
    output["chronicle_archive_enabled"] = False
    output["evidence_closeout_enabled"] = False
    spec["output"] = output
    return spec


def bind_design_to_compiled_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Bind physical design ordinals to compiler-generated stable cell IDs."""

    cells = plan.get("cells")
    plan_sha256 = plan.get("plan_sha256")
    if not isinstance(cells, list) or len(cells) != 48 or not isinstance(
        plan_sha256, str
    ):
        raise AdaptiveDesignError("compiled plan is not the exact physical B48")
    expected: dict[tuple[str, str], tuple[str, int]] = {}
    for config_id, bundle in configuration_bundles().items():
        for seed_id, ecology_seed, task_seed in SEED_TUPLES:
            identity = canonical_sha256(
                {
                    "parameters": bundle,
                    "seeds": {"ecology": ecology_seed, "task": task_seed},
                }
            )
            expected[(config_id, seed_id)] = (identity, len(expected))
    rows: list[dict[str, Any]] = []
    observed: set[tuple[str, str]] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise AdaptiveDesignError("compiled cell is malformed")
        identity = canonical_sha256(
            {
                "parameters": {
                    name: dict(cell.get("parameters") or {}).get(name)
                    for name in BASE_PANEL
                },
                "seeds": dict(cell.get("seeds") or {}),
            }
        )
        matches = [key for key, (digest, _ordinal) in expected.items() if digest == identity]
        if len(matches) != 1:
            raise AdaptiveDesignError("compiled cell does not map uniquely to the design")
        config_id, seed_id = matches[0]
        observed.add(matches[0])
        rows.append(
            {
                "physical_cell": expected[matches[0]][1],
                "config_id": config_id,
                "seed_tuple": seed_id,
                "compiled_ordinal": cell.get("ordinal"),
                "cell_id": cell.get("cell_id"),
                "branch_id": cell.get("branch_id"),
                "rng_lineage_id": cell.get("rng_lineage_id"),
            }
        )
    if observed != set(expected):
        raise AdaptiveDesignError("compiled plan does not cover the full design")
    rows.sort(key=lambda row: int(row["physical_cell"]))
    receipt = {
        "schema_version": "eiel_wrc_b48_adaptive_design_binding_v1",
        "design_source_commit": "d5a0f3ad118848d67f3c16cb4caf62d0a0ad83bc",
        "design_dependency_repairs": [
            {
                "affected_configurations": ["C15", "C16"],
                "changed_parameter": "wdl.active_duration_ticks",
                "from": 24,
                "to": 6,
                "preserved_replicates_per_configuration": 3,
                "reason": (
                    "preserve the existing fixed-MLP WDL batch ABI, which "
                    "requires one shared WDL mechanics context across B48"
                ),
            },
            {
                "affected_configurations": [
                    f"C{ordinal:02d}" for ordinal in range(1, 16)
                ],
                "changed_parameter": (
                    "env_cfg.patch_world_waste_consequence_enable"
                ),
                "from": False,
                "to": True,
                "preserved_replicates_per_configuration": 3,
                "reason": (
                    "preserve the exact full Patch P2 toggle context while "
                    "movement-cost application is enabled"
                ),
            },
        ],
        "plan_sha256": plan_sha256,
        "physical_cells": 48,
        "rows": rows,
    }
    return {**receipt, "receipt_sha256": canonical_sha256(receipt)}


def write_design(path: Path) -> None:
    rows = build_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=INITIAL_DESIGN_PATH,
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--base-spec", type=Path)
    parser.add_argument("--spec-output", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--binding-output", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_design()
    materialization_receipt: dict[str, Any] = {}
    requested_materialization = any(
        value is not None
        for value in (
            args.base_spec,
            args.spec_output,
            args.plan_output,
            args.binding_output,
            args.runtime_root,
        )
    )
    if requested_materialization:
        if any(
            value is None
            for value in (
                args.base_spec,
                args.spec_output,
                args.plan_output,
                args.binding_output,
                args.runtime_root,
            )
        ):
            raise AdaptiveDesignError(
                "campaign materialization requires --base-spec, --spec-output, "
                "--plan-output, --binding-output, and --runtime-root"
            )
        spec = materialize_initial_campaign_spec(
            load_campaign_spec(args.base_spec),
            runtime_root=args.runtime_root,
        )
        plan = compile_campaign(spec).payload()
        binding = bind_design_to_compiled_plan(plan)
        atomic_write_json(args.spec_output, spec)
        atomic_write_json(args.plan_output, plan)
        atomic_write_json(args.binding_output, binding)
        materialization_receipt = {
            "spec_output": str(args.spec_output.resolve()),
            "plan_output": str(args.plan_output.resolve()),
            "binding_output": str(args.binding_output.resolve()),
            "plan_sha256": plan["plan_sha256"],
            "design_binding_sha256": binding["receipt_sha256"],
        }
    if not args.validate_only:
        write_design(args.output)
    print(
        json.dumps(
            {
                "ok": True,
                "configurations": len(CONFIGURATIONS),
                "seed_tuples": len(SEED_TUPLES),
                "physical_cells": len(CONFIGURATIONS) * len(SEED_TUPLES),
                "parameters_per_cell": len(BASE_PANEL),
                "output": None if args.validate_only else str(args.output.resolve()),
                **materialization_receipt,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
