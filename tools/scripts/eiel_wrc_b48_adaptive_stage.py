#!/usr/bin/env python3
"""Compile one reviewed B48 intervention manifest into the next stage plan."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Final


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import canonical_sha256  # noqa: E402
from engine.campaigns.orchestration import atomic_write_json  # noqa: E402
from engine.campaigns.registry import default_profile_registry  # noqa: E402
from engine.campaigns.wrc_adaptive_rebinding import (  # noqa: E402
    WRCCheckpointAdaptiveRebindingError,
    materialize_wrc_adaptive_restore_plan_v1,
)
from engine.campaigns.wrc_variability import (  # noqa: E402
    INVARIANT_NAME,
    WRCVariabilityEnvelopeError,
    validate_wrc_parameter_variability_envelope_v1,
)
from tools.scripts.eiel_wrc_b48_adaptive_design import (  # noqa: E402
    CONFIGURATIONS,
    PROFILE_ID,
    SEED_TUPLES,
    WRC_IDS,
)


INTERVENTION_FIELDS: Final[tuple[str, ...]] = (
    "stage",
    "config_id",
    "tick_start",
    "tick_end",
    "parameter",
    "previous_value",
    "new_value",
    "source_WRC_ID",
    "reason",
    "evidence_used",
    "envelope_status",
    "receipt_action",
)
STAGES: Final[dict[str, tuple[int, int, int, int]]] = {
    "C2": (25, 50, 4_000, 8_000),
    "C3": (50, 75, 8_000, 12_000),
    "C4": (75, 100, 12_000, 16_000),
    "C5": (100, 125, 16_000, 20_000),
}
SOURCE_ANALYSIS_STAGE: Final[dict[str, str]] = {
    "C2": "C1",
    "C3": "C2",
    "C4": "C3",
    "C5": "C4",
}
ANALYSIS_OUTPUT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "world_summary",
        "configuration_summary",
        "seed_divergence",
        "checkpoints",
        "world_reductions",
    }
)


class AdaptiveStageError(ValueError):
    """A stage intervention violates the frozen B48/WRC contract."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdaptiveStageError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AdaptiveStageError(f"{label} must be an object")
    return dict(value)


def validate_design_binding(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    binding = dict(value)
    declared = binding.pop("receipt_sha256", None)
    rows = binding.get("rows")
    if (
        binding.get("schema_version")
        != "eiel_wrc_b48_adaptive_design_binding_v1"
        or binding.get("physical_cells") != 48
        or not isinstance(declared, str)
        or canonical_sha256(binding) != declared
        or not isinstance(rows, list)
        or len(rows) != 48
    ):
        raise AdaptiveStageError("design binding is invalid")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def validate_analysis_receipt(
    value: Mapping[str, Any],
    *,
    source_plan: Mapping[str, Any],
    design_binding: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    receipt = dict(value)
    declared = receipt.pop("receipt_sha256", None)
    outputs = receipt.get("outputs")
    source_generation, _restore_generation, source_tick_end, _tick_end = STAGES[stage]
    hash_fields = (
        "analysis_sha256",
        "checkpoint_sha256",
        "reducer_config_sha256",
    )
    outputs_valid = isinstance(outputs, Mapping) and set(outputs) == ANALYSIS_OUTPUT_NAMES
    if outputs_valid:
        outputs_valid = all(
            isinstance(row, Mapping)
            and isinstance(row.get("path"), str)
            and Path(row["path"]).is_absolute()
            and isinstance(row.get("sha256"), str)
            and len(row["sha256"]) == 64
            and type(row.get("bytes")) is int
            and row["bytes"] > 0
            for row in outputs.values()
        )
    if (
        receipt.get("schema_version")
        != "eiel_wrc_b48_adaptive_analysis_outputs_v1"
        or receipt.get("stage") != SOURCE_ANALYSIS_STAGE[stage]
        or receipt.get("plan_sha256") != source_plan.get("plan_sha256")
        or receipt.get("design_binding_receipt_sha256")
        != design_binding.get("receipt_sha256")
        or receipt.get("terminal_generation") != source_generation
        or receipt.get("tick_end") != source_tick_end
        or receipt.get("qualification_receipt_action") != "unchanged"
        or any(
            not isinstance(receipt.get(name), str) or len(receipt[name]) != 64
            for name in hash_fields
        )
        or not isinstance(receipt.get("reducer_version"), str)
        or not outputs_valid
        or not isinstance(declared, str)
        or canonical_sha256(receipt) != declared
    ):
        raise AdaptiveStageError("source stage analysis receipt is invalid")
    for name, row in sorted(outputs.items()):
        path = Path(str(row["path"]))
        if not path.is_file() or path.is_symlink():
            raise AdaptiveStageError(f"source analysis artifact is absent: {name}")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise AdaptiveStageError(
                f"cannot read source analysis artifact {name}: {exc}"
            ) from exc
        if (
            len(raw) != row["bytes"]
            or hashlib.sha256(raw).hexdigest() != row["sha256"]
        ):
            raise AdaptiveStageError(
                f"source analysis artifact identity differs: {name}"
            )
    return {**receipt, "receipt_sha256": declared}


def _parse_interventions(path: Path, *, stage: str) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != INTERVENTION_FIELDS:
                raise AdaptiveStageError("intervention CSV fields differ")
            raw_rows = [dict(row) for row in reader]
    except OSError as exc:
        raise AdaptiveStageError(f"cannot read intervention CSV: {exc}") from exc
    rows = [row for row in raw_rows if row.get("stage") == stage]
    if not rows:
        raise AdaptiveStageError(f"intervention CSV has no {stage} rows")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            previous = json.loads(row["previous_value"])
            new = json.loads(row["new_value"])
            tick_start = int(row["tick_start"])
            tick_end = int(row["tick_end"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AdaptiveStageError("intervention row is malformed") from exc
        parsed.append(
            {
                **row,
                "previous_value": previous,
                "new_value": new,
                "tick_start": tick_start,
                "tick_end": tick_end,
            }
        )
    return parsed


def materialize_stage_plan(
    source_plan: Mapping[str, Any],
    design_binding: Mapping[str, Any],
    intervention_rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    analysis_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one config-level manifest and preserve all three replicates."""

    if stage not in STAGES:
        raise AdaptiveStageError(f"unsupported stage {stage!r}")
    source_generation, restore_generation, tick_start, tick_end = STAGES[stage]
    execution = source_plan.get("execution")
    cells = source_plan.get("cells")
    if (
        not isinstance(execution, Mapping)
        or execution.get("generations") != source_generation
        or not isinstance(cells, list)
        or len(cells) != 48
    ):
        raise AdaptiveStageError("source plan is not the expected B48 stage")
    binding_rows = validate_design_binding(design_binding)
    source_analysis = validate_analysis_receipt(
        analysis_receipt,
        source_plan=source_plan,
        design_binding=design_binding,
        stage=stage,
    )
    cells_by_id = {
        str(cell.get("cell_id")): dict(cell)
        for cell in cells
        if isinstance(cell, Mapping) and isinstance(cell.get("cell_id"), str)
    }
    bindings_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in binding_rows:
        cell_id = row.get("cell_id")
        cell = cells_by_id.get(str(cell_id))
        if (
            not isinstance(cell, Mapping)
            or cell.get("branch_id") != row.get("branch_id")
            or cell.get("rng_lineage_id") != row.get("rng_lineage_id")
        ):
            raise AdaptiveStageError("source plan identity differs from design binding")
        bindings_by_config[str(row.get("config_id"))].append(row)
    expected_configs = {row[0] for row in CONFIGURATIONS}
    expected_seeds = {row[0] for row in SEED_TUPLES}
    if set(bindings_by_config) != expected_configs or any(
        len(rows) != 3
        or {str(row.get("seed_tuple")) for row in rows} != expected_seeds
        for rows in bindings_by_config.values()
    ):
        raise AdaptiveStageError("design binding does not preserve 16 x 3 replication")
    profile = next(
        (
            candidate
            for candidate in default_profile_registry().profiles()
            if candidate.profile_id == PROFILE_ID
        ),
        None,
    )
    if profile is None:
        raise AdaptiveStageError("production WRC profile is unavailable")
    qualification = source_plan.get("qualification_key")
    if (
        not isinstance(qualification, Mapping)
        or canonical_sha256(qualification)
        != source_plan.get("qualification_key_sha256")
    ):
        raise AdaptiveStageError("source qualification binding is invalid")
    envelope_payload = qualification.get("variability_envelope")
    fixed_invariants = qualification.get("fixed_invariants")
    if not isinstance(envelope_payload, Mapping) or not isinstance(
        fixed_invariants, Mapping
    ):
        raise AdaptiveStageError("source WRC envelope binding is absent")
    try:
        envelope = validate_wrc_parameter_variability_envelope_v1(envelope_payload)
    except WRCVariabilityEnvelopeError as exc:
        raise AdaptiveStageError(f"source WRC envelope is invalid: {exc}") from exc
    if (
        envelope.profile_id != PROFILE_ID
        or fixed_invariants.get(INVARIANT_NAME) != envelope.envelope_sha256
    ):
        raise AdaptiveStageError("source WRC envelope identity differs")
    artifacts = source_plan.get("artifacts")
    qualification_receipt = (
        artifacts.get("qualification_receipt")
        if isinstance(artifacts, Mapping)
        else None
    )
    if not isinstance(qualification_receipt, Mapping):
        raise AdaptiveStageError("source qualification receipt binding is absent")
    envelope_rows = {
        str(row.get("name")): dict(row) for row in envelope.parameter_contracts
    }
    normalized_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    changes_by_cell: dict[str, dict[str, Any]] = defaultdict(dict)
    changed_configs: set[str] = set()
    for raw in intervention_rows:
        row = dict(raw)
        if set(row) != set(INTERVENTION_FIELDS):
            raise AdaptiveStageError("intervention fields differ")
        config_id = row.get("config_id")
        parameter = row.get("parameter")
        key = (str(config_id), str(parameter))
        if key in seen:
            raise AdaptiveStageError(f"duplicate intervention {key!r}")
        seen.add(key)
        if (
            row.get("stage") != stage
            or config_id not in expected_configs
            or row.get("tick_start") != tick_start
            or row.get("tick_end") != tick_end
            or not isinstance(parameter, str)
            or WRC_IDS.get(parameter) != row.get("source_WRC_ID")
            or row.get("envelope_status") != "COVERED_APPROVED"
            or row.get("receipt_action") != "unchanged"
            or not isinstance(row.get("reason"), str)
            or not row["reason"].strip()
            or not isinstance(row.get("evidence_used"), str)
            or row["evidence_used"] != source_analysis["receipt_sha256"]
        ):
            raise AdaptiveStageError(f"intervention authority is invalid: {key!r}")
        contract = profile.parameter_contracts.get(parameter)
        new_value = row.get("new_value")
        previous_value = row.get("previous_value")
        if (
            contract is None
            or envelope_rows.get(parameter)
            != contract.variability_contract_payload_v1()
            or contract.variability_semantics != "experiment_variable"
            or contract.qualification_affecting
            or not contract.per_cell
            or not contract.accepts(new_value)
            or new_value == previous_value
        ):
            raise AdaptiveStageError(f"intervention value is invalid: {key!r}")
        for binding in bindings_by_config[str(config_id)]:
            cell_id = str(binding["cell_id"])
            parameters = cells_by_id[cell_id].get("parameters")
            if (
                not isinstance(parameters, Mapping)
                or parameters.get(parameter) != previous_value
            ):
                raise AdaptiveStageError(
                    f"previous value differs across {config_id} replicates: {parameter}"
                )
            changes_by_cell[cell_id][parameter] = new_value
        changed_configs.add(str(config_id))
        normalized_rows.append(row)
    if len(changed_configs) > 16:
        raise AdaptiveStageError("stage exceeds 16 configuration bundles")
    parameter_names = sorted({str(row["parameter"]) for row in normalized_rows})
    arrays = {
        parameter: [
            changes_by_cell[cell["cell_id"]].get(
                parameter, dict(cell["parameters"])[parameter]
            )
            for cell in cells
        ]
        for parameter in parameter_names
    }
    try:
        restored = materialize_wrc_adaptive_restore_plan_v1(
            source_plan,
            restore_generation=restore_generation,
            parameter_arrays=arrays,
        )
    except WRCCheckpointAdaptiveRebindingError as exc:
        raise AdaptiveStageError(str(exc)) from exc
    # Validate all dependency-bearing experiment variables after applying the
    # complete stage bundle, including controls unchanged by this transition.
    for cell in restored["cells"]:
        parameters = dict(cell["parameters"])
        for name, contract in profile.parameter_contracts.items():
            if name not in parameters or not contract.activation_contract:
                continue
            required = contract.activation_contract.get(
                "required_effective_controls"
            )
            if not isinstance(required, Mapping):
                raise AdaptiveStageError(f"activation contract is malformed: {name}")
            for control, expected in required.items():
                actual = (
                    parameters.get(control)
                    if control in parameters
                    else True
                    if control == "env_cfg.patch_world_enable"
                    else None
                )
                if actual != expected:
                    raise AdaptiveStageError(
                        f"PARAMETER_CONTEXT_INACTIVE: {name} requires "
                        f"{control}={expected!r}, got {actual!r}"
                    )
    normalized_rows.sort(key=lambda row: (row["config_id"], row["parameter"]))
    binding_by_cell = {str(row["cell_id"]): row for row in binding_rows}
    cell_overrides = [
        {
            "cell_id": cell["cell_id"],
            "config_id": binding_by_cell[cell["cell_id"]]["config_id"],
            "seed_tuple": binding_by_cell[cell["cell_id"]]["seed_tuple"],
            "overrides": {
                name: changes_by_cell[cell["cell_id"]][name]
                for name in sorted(changes_by_cell[cell["cell_id"]])
            },
        }
        for cell in restored["cells"]
        if changes_by_cell[cell["cell_id"]]
    ]
    manifest = {
        "schema_version": "eiel_wrc_b48_adaptive_stage_manifest_v1",
        "stage": stage,
        "source_generation": source_generation,
        "restore_generation": restore_generation,
        "tick_start": tick_start,
        "tick_end": tick_end,
        "source_plan_sha256": source_plan.get("plan_sha256"),
        "restore_plan_sha256": restored["plan_sha256"],
        "design_binding_receipt_sha256": design_binding.get("receipt_sha256"),
        "source_analysis_receipt_sha256": source_analysis["receipt_sha256"],
        "qualification_key_sha256": source_plan["qualification_key_sha256"],
        "variability_envelope_id": envelope.envelope_id,
        "variability_envelope_sha256": envelope.envelope_sha256,
        "qualification_receipt": dict(qualification_receipt),
        "production_artifacts_sha256": canonical_sha256(artifacts),
        "configuration_count": 16,
        "seed_replicates_per_configuration": 3,
        "physical_cell_count": 48,
        "changed_configuration_count": len(changed_configs),
        "changed_cell_count": len(cell_overrides),
        "rows": normalized_rows,
        "cell_overrides": cell_overrides,
        "qualification_receipt_action": "unchanged",
    }
    return restored, {**manifest, "receipt_sha256": canonical_sha256(manifest)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--design-binding", type=Path, required=True)
    parser.add_argument("--interventions", type=Path, required=True)
    parser.add_argument("--analysis-receipt", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = _load_json(args.source_plan, label="source plan")
    binding = _load_json(args.design_binding, label="design binding")
    rows = _parse_interventions(args.interventions, stage=args.stage)
    analysis_receipt = _load_json(args.analysis_receipt, label="analysis receipt")
    plan, manifest = materialize_stage_plan(
        source,
        binding,
        rows,
        stage=args.stage,
        analysis_receipt=analysis_receipt,
    )
    atomic_write_json(args.plan_output, plan)
    atomic_write_json(args.manifest_output, manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "stage": args.stage,
                "plan_output": str(args.plan_output.resolve()),
                "manifest_output": str(args.manifest_output.resolve()),
                "plan_sha256": plan["plan_sha256"],
                "manifest_sha256": manifest["receipt_sha256"],
                "qualification_receipt_action": "unchanged",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
