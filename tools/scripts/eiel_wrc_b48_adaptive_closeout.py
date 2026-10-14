#!/usr/bin/env python3
"""Verify and consolidate one complete G125/T20000 B48 adaptive chain."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import sys
from typing import Any, Final
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import canonical_sha256  # noqa: E402
from engine.campaigns.orchestration import atomic_write_json  # noqa: E402
from tools.scripts.eiel_wrc_b48_adaptive_analyze import (  # noqa: E402
    CHECKPOINT_FIELDS,
    CONFIG_FIELDS,
    DIVERGENCE_FIELDS,
    STAGE_WINDOWS,
    WORLD_FIELDS,
)
from tools.scripts.eiel_wrc_b48_adaptive_stage import (  # noqa: E402
    ANALYSIS_OUTPUT_NAMES,
    INTERVENTION_FIELDS,
    STAGES,
    validate_design_binding,
)


STAGE_ORDER: Final[tuple[str, ...]] = ("C1", "C2", "C3", "C4", "C5")
TRANSITION_ORDER: Final[tuple[str, ...]] = ("C2", "C3", "C4", "C5")
OUTPUT_NAMES: Final[dict[str, str]] = {
    "initial_design": "B48_EXPLORATORY_INITIAL_DESIGN.csv",
    "interventions": "B48_EXPLORATORY_STAGE_INTERVENTIONS.csv",
    "checkpoints": "B48_EXPLORATORY_CHECKPOINTS.csv",
    "configuration_summary": "B48_EXPLORATORY_CONFIGURATION_SUMMARY.csv",
    "world_summary": "B48_EXPLORATORY_WORLD_SUMMARY.csv",
    "seed_divergence": "B48_EXPLORATORY_SEED_DIVERGENCE.csv",
    "world_reductions": "B48_EXPLORATORY_WORLD_REDUCTIONS.jsonl",
    "report_input": "WRC_B48_20K_ADAPTIVE_EXPLORATORY_REPORT_INPUT.json",
    "exploratory_report": "WRC_B48_20K_ADAPTIVE_EXPLORATORY_REPORT.md",
    "integration_record": "WRC_B48_20K_ADAPTIVE_EXPLORATORY_INTEGRATION_RECORD.md",
}


class AdaptiveCloseoutError(ValueError):
    """The five-stage adaptive chain is incomplete or internally inconsistent."""


def _signed_receipt(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    receipt = dict(value)
    declared = receipt.pop("receipt_sha256", None)
    if not isinstance(declared, str) or canonical_sha256(receipt) != declared:
        raise AdaptiveCloseoutError(f"{label} identity hash differs")
    return {**receipt, "receipt_sha256": declared}


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdaptiveCloseoutError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AdaptiveCloseoutError(f"{label} must be an object")
    return dict(value)


def _descriptor_bytes(value: Mapping[str, Any], *, label: str) -> bytes:
    path_value = value.get("path")
    digest = value.get("sha256")
    size = value.get("bytes")
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or not isinstance(digest, str)
        or len(digest) != 64
        or type(size) is not int
        or size < 1
    ):
        raise AdaptiveCloseoutError(f"{label} descriptor is invalid")
    path = Path(path_value)
    if not path.is_file() or path.is_symlink():
        raise AdaptiveCloseoutError(f"{label} artifact is absent or unsafe")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AdaptiveCloseoutError(f"cannot read {label}: {exc}") from exc
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise AdaptiveCloseoutError(f"{label} artifact identity differs")
    return raw


def _csv_rows(
    raw: bytes, *, fields: Sequence[str], label: str
) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(StringIO(raw.decode("utf-8"), newline=""))
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise AdaptiveCloseoutError(f"{label} CSV fields differ")
        return [dict(row) for row in reader]
    except UnicodeError as exc:
        raise AdaptiveCloseoutError(f"{label} is not UTF-8") from exc


def _analysis_artifacts(
    value: Mapping[str, Any],
    *,
    stage: str,
    design_binding_sha256: str,
    binding_by_cell: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = _signed_receipt(value, label=f"{stage} analysis receipt")
    _start, generation, _tick_start, tick_end = STAGE_WINDOWS[stage]
    outputs = receipt.get("outputs")
    if (
        receipt.get("schema_version")
        != "eiel_wrc_b48_adaptive_analysis_outputs_v1"
        or receipt.get("stage") != stage
        or receipt.get("terminal_generation") != generation
        or receipt.get("tick_end") != tick_end
        or receipt.get("design_binding_receipt_sha256")
        != design_binding_sha256
        or receipt.get("qualification_receipt_action") != "unchanged"
        or not isinstance(receipt.get("plan_sha256"), str)
        or len(receipt["plan_sha256"]) != 64
        or not isinstance(outputs, Mapping)
        or set(outputs) != ANALYSIS_OUTPUT_NAMES
    ):
        raise AdaptiveCloseoutError(f"{stage} analysis binding differs")
    raw = {
        name: _descriptor_bytes(outputs[name], label=f"{stage}/{name}")
        for name in sorted(ANALYSIS_OUTPUT_NAMES)
    }
    world_rows = _csv_rows(
        raw["world_summary"], fields=WORLD_FIELDS, label=f"{stage} world summary"
    )
    config_rows = _csv_rows(
        raw["configuration_summary"],
        fields=CONFIG_FIELDS,
        label=f"{stage} configuration summary",
    )
    divergence_rows = _csv_rows(
        raw["seed_divergence"],
        fields=DIVERGENCE_FIELDS,
        label=f"{stage} seed divergence",
    )
    checkpoint_rows = _csv_rows(
        raw["checkpoints"],
        fields=CHECKPOINT_FIELDS,
        label=f"{stage} checkpoints",
    )
    if (
        len(world_rows) != 48
        or len(config_rows) != 16
        or len(checkpoint_rows) != 1
        or not divergence_rows
        or {row.get("stage") for row in world_rows} != {stage}
        or {row.get("stage") for row in config_rows} != {stage}
        or {row.get("stage") for row in divergence_rows} != {stage}
        or checkpoint_rows[0].get("stage") != stage
        or checkpoint_rows[0].get("absolute_tick") != str(tick_end)
        or checkpoint_rows[0].get("generation") != str(generation)
        or checkpoint_rows[0].get("checkpoint_sha256")
        != receipt.get("checkpoint_sha256")
        or checkpoint_rows[0].get("plan_sha256") != receipt.get("plan_sha256")
        or checkpoint_rows[0].get("continuity_status") != "VERIFIED_EXACT_48"
        or checkpoint_rows[0].get("qualification_receipt_action") != "unchanged"
    ):
        raise AdaptiveCloseoutError(f"{stage} analysis row coverage differs")
    worlds_by_cell = {row.get("cell_id"): row for row in world_rows}
    if set(worlds_by_cell) != set(binding_by_cell):
        raise AdaptiveCloseoutError(f"{stage} world identities differ")
    for cell_id, binding in binding_by_cell.items():
        row = worlds_by_cell[cell_id]
        if (
            row.get("config_id") != str(binding.get("config_id"))
            or row.get("seed_tuple") != str(binding.get("seed_tuple"))
            or row.get("branch_id") != str(binding.get("branch_id"))
            or row.get("rng_lineage_id") != str(binding.get("rng_lineage_id"))
        ):
            raise AdaptiveCloseoutError(f"{stage}/{cell_id} identity differs")
    expected_configs = {f"C{index:02d}" for index in range(1, 17)}
    if (
        {row.get("config_id") for row in config_rows} != expected_configs
        or any(row.get("replicate_count") != "3" for row in config_rows)
        or {row.get("config_id") for row in divergence_rows} != expected_configs
        or not all(
            any(
                row.get("config_id") == config_id and row.get("metric") == "alive"
                for row in divergence_rows
            )
            for config_id in expected_configs
        )
    ):
        raise AdaptiveCloseoutError(f"{stage} 16 x 3 aggregation differs")
    reduction_rows: list[dict[str, Any]] = []
    try:
        for line in raw["world_reductions"].decode("utf-8").splitlines():
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise AdaptiveCloseoutError(
                    f"{stage} world reduction row is malformed"
                )
            material = dict(row)
            declared = material.pop("result_sha256", None)
            if not isinstance(declared, str) or canonical_sha256(material) != declared:
                raise AdaptiveCloseoutError(
                    f"{stage} world reduction identity differs"
                )
            reduction_rows.append(dict(row))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AdaptiveCloseoutError(
            f"{stage} world reductions are malformed"
        ) from exc
    reduction_cells = {
        row.get("identity", {}).get("world_id")
        for row in reduction_rows
        if isinstance(row.get("identity"), Mapping)
    }
    if len(reduction_rows) != 48 or reduction_cells != set(binding_by_cell):
        raise AdaptiveCloseoutError(f"{stage} world reductions differ")
    reductions_by_cell = {
        str(row["identity"]["world_id"]): row for row in reduction_rows
    }
    if any(
        worlds_by_cell[cell_id].get("reducer_result_sha256")
        != reductions_by_cell[cell_id].get("result_sha256")
        for cell_id in binding_by_cell
    ):
        raise AdaptiveCloseoutError(f"{stage} world summary reduction binding differs")
    return {
        "receipt": receipt,
        "world_rows": world_rows,
        "configuration_rows": config_rows,
        "seed_divergence_rows": divergence_rows,
        "checkpoint_row": checkpoint_rows[0],
        "world_reductions_raw": raw["world_reductions"],
    }


def _validate_stage_manifest(
    value: Mapping[str, Any],
    *,
    stage: str,
    source_analysis: Mapping[str, Any],
    restore_analysis: Mapping[str, Any],
    design_binding_sha256: str,
    bindings_by_config: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    manifest = _signed_receipt(value, label=f"{stage} intervention manifest")
    source_generation, restore_generation, tick_start, tick_end = STAGES[stage]
    rows = manifest.get("rows")
    cell_overrides = manifest.get("cell_overrides")
    qualification_receipt = manifest.get("qualification_receipt")
    if (
        manifest.get("schema_version")
        != "eiel_wrc_b48_adaptive_stage_manifest_v1"
        or manifest.get("stage") != stage
        or manifest.get("source_generation") != source_generation
        or manifest.get("restore_generation") != restore_generation
        or manifest.get("tick_start") != tick_start
        or manifest.get("tick_end") != tick_end
        or manifest.get("source_plan_sha256")
        != source_analysis.get("plan_sha256")
        or manifest.get("restore_plan_sha256")
        != restore_analysis.get("plan_sha256")
        or manifest.get("source_analysis_receipt_sha256")
        != source_analysis.get("receipt_sha256")
        or manifest.get("design_binding_receipt_sha256")
        != design_binding_sha256
        or manifest.get("configuration_count") != 16
        or manifest.get("seed_replicates_per_configuration") != 3
        or manifest.get("physical_cell_count") != 48
        or manifest.get("qualification_receipt_action") != "unchanged"
        or not isinstance(manifest.get("qualification_key_sha256"), str)
        or len(manifest["qualification_key_sha256"]) != 64
        or not isinstance(manifest.get("variability_envelope_id"), str)
        or not manifest["variability_envelope_id"]
        or not isinstance(manifest.get("variability_envelope_sha256"), str)
        or len(manifest["variability_envelope_sha256"]) != 64
        or not isinstance(qualification_receipt, Mapping)
        or not isinstance(qualification_receipt.get("sha256"), str)
        or len(qualification_receipt["sha256"]) != 64
        or not isinstance(manifest.get("production_artifacts_sha256"), str)
        or len(manifest["production_artifacts_sha256"]) != 64
        or not isinstance(rows, list)
        or not rows
        or not isinstance(cell_overrides, list)
    ):
        raise AdaptiveCloseoutError(f"{stage} intervention binding differs")
    by_config: dict[str, dict[str, Any]] = defaultdict(dict)
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != set(INTERVENTION_FIELDS):
            raise AdaptiveCloseoutError(f"{stage} intervention row is malformed")
        config_id = str(raw.get("config_id"))
        parameter = str(raw.get("parameter"))
        if (
            raw.get("stage") != stage
            or config_id not in bindings_by_config
            or raw.get("tick_start") != tick_start
            or raw.get("tick_end") != tick_end
            or raw.get("evidence_used") != source_analysis.get("receipt_sha256")
            or raw.get("envelope_status") != "COVERED_APPROVED"
            or raw.get("receipt_action") != "unchanged"
            or parameter in by_config[config_id]
        ):
            raise AdaptiveCloseoutError(f"{stage} intervention authority differs")
        by_config[config_id][parameter] = raw.get("new_value")
    observed_cells: dict[str, dict[str, Any]] = {}
    for raw in cell_overrides:
        if not isinstance(raw, Mapping):
            raise AdaptiveCloseoutError(f"{stage} cell override is malformed")
        cell_id = str(raw.get("cell_id"))
        config_id = str(raw.get("config_id"))
        if (
            cell_id in observed_cells
            or config_id not in by_config
            or raw.get("overrides") != by_config[config_id]
        ):
            raise AdaptiveCloseoutError(f"{stage} cell override differs")
        expected = {
            str(row["cell_id"]): str(row["seed_tuple"])
            for row in bindings_by_config[config_id]
        }
        if cell_id not in expected or raw.get("seed_tuple") != expected[cell_id]:
            raise AdaptiveCloseoutError(f"{stage} replicate mapping differs")
        observed_cells[cell_id] = dict(raw)
    expected_cells = {
        str(row["cell_id"])
        for config_id in by_config
        for row in bindings_by_config[config_id]
    }
    if (
        set(observed_cells) != expected_cells
        or manifest.get("changed_configuration_count") != len(by_config)
        or manifest.get("changed_cell_count") != len(expected_cells)
    ):
        raise AdaptiveCloseoutError(f"{stage} 3-seed expansion differs")
    return manifest


def _initial_design(
    path: Path, *, bindings_by_physical: Mapping[int, Mapping[str, Any]]
) -> tuple[bytes, list[dict[str, str]]]:
    try:
        raw = path.resolve().read_bytes()
    except OSError as exc:
        raise AdaptiveCloseoutError(f"cannot read initial design: {exc}") from exc
    fields = (
        "config_id",
        "seed_tuple",
        "physical_cell",
        "parameter",
        "value",
        "source_WRC_ID",
        "evidence_rationale",
        "expected_regime",
    )
    rows = _csv_rows(raw, fields=fields, label="initial design")
    if len(rows) != 48 * 22:
        raise AdaptiveCloseoutError("initial design does not contain 48 x 22 rows")
    counts: dict[int, int] = defaultdict(int)
    for row in rows:
        try:
            physical = int(row["physical_cell"])
            seed = json.loads(row["seed_tuple"])
        except (ValueError, json.JSONDecodeError) as exc:
            raise AdaptiveCloseoutError("initial design row is malformed") from exc
        binding = bindings_by_physical.get(physical)
        if (
            binding is None
            or row["config_id"] != binding.get("config_id")
            or not isinstance(seed, Mapping)
            or seed.get("id") != binding.get("seed_tuple")
        ):
            raise AdaptiveCloseoutError("initial design mapping differs")
        counts[physical] += 1
    if counts != {index: 22 for index in range(48)}:
        raise AdaptiveCloseoutError("initial design physical coverage differs")
    return raw, rows


def consolidate_adaptive_chain(
    *,
    design_binding: Mapping[str, Any],
    initial_design_path: Path,
    analysis_receipts: Sequence[Mapping[str, Any]],
    stage_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify one complete chain and return cumulative materialization inputs."""

    binding_rows = validate_design_binding(design_binding)
    binding_sha = str(design_binding["receipt_sha256"])
    binding_by_cell = {str(row["cell_id"]): row for row in binding_rows}
    bindings_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bindings_by_physical = {
        int(row["physical_cell"]): row for row in binding_rows
    }
    for row in binding_rows:
        bindings_by_config[str(row["config_id"])].append(row)
    if len(binding_by_cell) != 48 or len(bindings_by_physical) != 48:
        raise AdaptiveCloseoutError("design binding identity is incomplete")
    initial_raw, _initial_rows = _initial_design(
        initial_design_path, bindings_by_physical=bindings_by_physical
    )
    raw_analysis_by_stage = {
        str(value.get("stage")): value for value in analysis_receipts
    }
    if len(analysis_receipts) != 5 or set(raw_analysis_by_stage) != set(STAGE_ORDER):
        raise AdaptiveCloseoutError("exactly C1-C5 analysis receipts are required")
    analyses = {
        stage: _analysis_artifacts(
            raw_analysis_by_stage[stage],
            stage=stage,
            design_binding_sha256=binding_sha,
            binding_by_cell=binding_by_cell,
        )
        for stage in STAGE_ORDER
    }
    analysis_identity: tuple[str, str, str, str, str] | None = None
    for stage in STAGE_ORDER:
        checkpoint = analyses[stage]["checkpoint_row"]
        identity = (
            str(checkpoint.get("qualification_key_sha256")),
            str(checkpoint.get("variability_envelope_id")),
            str(checkpoint.get("variability_envelope_sha256")),
            str(checkpoint.get("qualification_receipt_sha256")),
            str(checkpoint.get("execution_route_sha256")),
        )
        if any(not value for value in identity):
            raise AdaptiveCloseoutError(f"{stage} runtime identity is incomplete")
        if analysis_identity is None:
            analysis_identity = identity
        elif identity != analysis_identity:
            raise AdaptiveCloseoutError(
                "qualification, envelope, receipt, or route identity changed"
            )
    raw_manifests_by_stage = {
        str(value.get("stage")): value for value in stage_manifests
    }
    if (
        len(stage_manifests) != 4
        or set(raw_manifests_by_stage) != set(TRANSITION_ORDER)
    ):
        raise AdaptiveCloseoutError("exactly C2-C5 stage manifests are required")
    manifests: dict[str, dict[str, Any]] = {}
    qualification_identity: tuple[Any, ...] | None = None
    for index, stage in enumerate(TRANSITION_ORDER):
        source_stage = STAGE_ORDER[index]
        restore_stage = STAGE_ORDER[index + 1]
        manifest = _validate_stage_manifest(
            raw_manifests_by_stage[stage],
            stage=stage,
            source_analysis=analyses[source_stage]["receipt"],
            restore_analysis=analyses[restore_stage]["receipt"],
            design_binding_sha256=binding_sha,
            bindings_by_config=bindings_by_config,
        )
        identity = (
            manifest.get("qualification_key_sha256"),
            manifest.get("variability_envelope_id"),
            manifest.get("variability_envelope_sha256"),
            manifest["qualification_receipt"]["sha256"],
            manifest.get("production_artifacts_sha256"),
        )
        if analysis_identity is None or identity[:4] != analysis_identity[:4]:
            raise AdaptiveCloseoutError(
                "intervention and checkpoint qualification identity differ"
            )
        if qualification_identity is None:
            qualification_identity = identity
        elif identity != qualification_identity:
            raise AdaptiveCloseoutError("qualification or production identity changed")
        manifests[stage] = manifest
    checkpoint_hashes = [
        analyses[stage]["receipt"].get("checkpoint_sha256") for stage in STAGE_ORDER
    ]
    if len(set(checkpoint_hashes)) != 5:
        raise AdaptiveCloseoutError("checkpoint identities are not stage-distinct")

    world_rows = [
        row for stage in STAGE_ORDER for row in analyses[stage]["world_rows"]
    ]
    config_rows = [
        row
        for stage in STAGE_ORDER
        for row in analyses[stage]["configuration_rows"]
    ]
    divergence_rows = [
        row
        for stage in STAGE_ORDER
        for row in analyses[stage]["seed_divergence_rows"]
    ]
    checkpoint_rows = [analyses[stage]["checkpoint_row"] for stage in STAGE_ORDER]
    intervention_rows = [
        dict(row)
        for stage in TRANSITION_ORDER
        for row in manifests[stage]["rows"]
    ]
    reductions_raw = b"".join(
        analyses[stage]["world_reductions_raw"] for stage in STAGE_ORDER
    )
    c5_configs = {
        str(row["config_id"]): row
        for row in analyses["C5"]["configuration_rows"]
    }
    alive_trajectories: dict[str, list[float]] = defaultdict(list)
    for stage in STAGE_ORDER:
        for row in analyses[stage]["configuration_rows"]:
            alive_trajectories[str(row["config_id"])].append(
                float(row["alive_terminal_mean"])
            )
    alive_divergence = {
        str(row["config_id"]): row
        for row in analyses["C5"]["seed_divergence_rows"]
        if row.get("metric") == "alive"
    }
    viable_configs = sorted(
        config_id
        for config_id, row in c5_configs.items()
        if int(row["viable_count"]) >= 2
    )
    ranked_resilience = sorted(
        c5_configs,
        key=lambda config_id: (
            -float(c5_configs[config_id]["alive_terminal_mean"]),
            config_id,
        ),
    )
    ranked_divergence = sorted(
        alive_divergence,
        key=lambda config_id: (
            -float(alive_divergence[config_id]["terminal_relative_range"]),
            config_id,
        ),
    )
    report_input = {
        "schema_version": "eiel_wrc_b48_g125_adaptive_report_input_v1",
        "interpretation": "EXPLORATORY_ADAPTIVE_INTERVENTION",
        "terminal_generation": 125,
        "terminal_tick": 20_000,
        "world_count": 48,
        "configuration_count": 16,
        "seed_replicates_per_configuration": 3,
        "continuous_checkpoint_count": 5,
        "qualification_receipt_action": "unchanged",
        "viable_configuration_count_at_c5": len(viable_configs),
        "viable_configurations_at_c5": viable_configs,
        "resilience_ranking_by_c5_alive_mean": ranked_resilience,
        "fragility_ranking_by_c5_alive_mean": list(reversed(ranked_resilience)),
        "alive_seed_divergence_ranking_at_c5": ranked_divergence,
        "configuration_alive_trajectories": dict(sorted(alive_trajectories.items())),
        "limitations": [
            "Rankings are descriptive and path-dependent.",
            "They do not identify isolated knob causality.",
            "Final scientific synthesis must inspect reducer drill-back sources.",
            "This chain verification is not a qualification action.",
        ],
    }
    source_receipts = {
        "design_binding": binding_sha,
        "analysis": {
            stage: analyses[stage]["receipt"]["receipt_sha256"]
            for stage in STAGE_ORDER
        },
        "interventions": {
            stage: manifests[stage]["receipt_sha256"] for stage in TRANSITION_ORDER
        },
    }
    chain = {
        "schema_version": "eiel_wrc_b48_g125_adaptive_chain_v1",
        "status": "CHAIN_VERIFIED_G125_T20000",
        "design_binding_receipt_sha256": binding_sha,
        "initial_design_sha256": hashlib.sha256(initial_raw).hexdigest(),
        "stage_order": list(STAGE_ORDER),
        "transition_order": list(TRANSITION_ORDER),
        "checkpoint_sha256s": checkpoint_hashes,
        "qualification_identity_sha256": canonical_sha256(
            {
                "qualification_and_artifacts": list(qualification_identity or ()),
                "execution_route_sha256": (
                    None if analysis_identity is None else analysis_identity[4]
                ),
            }
        ),
        "source_receipts": source_receipts,
        "qualification_receipt_action": "unchanged",
    }
    chain["chain_sha256"] = canonical_sha256(chain)
    consolidated = {
        "chain": chain,
        "initial_design_raw": initial_raw,
        "intervention_rows": intervention_rows,
        "checkpoint_rows": checkpoint_rows,
        "configuration_rows": config_rows,
        "world_rows": world_rows,
        "seed_divergence_rows": divergence_rows,
        "world_reductions_raw": reductions_raw,
        "report_input": report_input,
    }
    consolidated["consolidated_sha256"] = _consolidated_sha256(consolidated)
    return consolidated


def _consolidated_sha256(value: Mapping[str, Any]) -> str:
    initial = value.get("initial_design_raw")
    reductions = value.get("world_reductions_raw")
    if not isinstance(initial, bytes) or not isinstance(reductions, bytes):
        raise AdaptiveCloseoutError("consolidated byte artifacts are absent")
    chain = value.get("chain")
    if not isinstance(chain, Mapping):
        raise AdaptiveCloseoutError("consolidated chain is absent")
    return canonical_sha256(
        {
            "chain_sha256": chain.get("chain_sha256"),
            "initial_design_sha256": hashlib.sha256(initial).hexdigest(),
            "intervention_rows": value.get("intervention_rows"),
            "checkpoint_rows": value.get("checkpoint_rows"),
            "configuration_rows": value.get("configuration_rows"),
            "world_rows": value.get("world_rows"),
            "seed_divergence_rows": value.get("seed_divergence_rows"),
            "world_reductions_sha256": hashlib.sha256(reductions).hexdigest(),
            "report_input": value.get("report_input"),
        }
    )


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _integration_record(chain: Mapping[str, Any]) -> str:
    rows = [
        "# WRC B48 G125 / 20,000-tick Adaptive Integration Record",
        "",
        "Status: `CHAIN_VERIFIED_G125_T20000`",
        "",
        "This record binds one exploratory 48-world chain. It is not a new "
        "qualification action or an isolated-causality claim.",
        "",
        "| Stage | Analysis receipt | Checkpoint |",
        "|---|---|---|",
    ]
    analyses = chain["source_receipts"]["analysis"]
    for stage, checkpoint in zip(
        STAGE_ORDER, chain["checkpoint_sha256s"], strict=True
    ):
        rows.append(f"| {stage} | `{analyses[stage]}` | `{checkpoint}` |")
    rows.extend(
        [
            "",
            "## Preserved chain invariants",
            "",
            "- 48 physical worlds and 16 configuration classes x 3 seed replicates.",
            "- Generation-boundary stages G25, G50, G75, G100, and G125 at T160.",
            "- Five distinct verified checkpoint identities.",
            "- One unchanged qualification, envelope, production-artifact, and route lineage.",
            "- Every C2-C5 intervention is bound to the preceding analysis receipt.",
            "- Every configuration intervention expands identically to its three seeds.",
            "",
            "## Claim boundary",
            "",
            "Outputs are exploratory and path-dependent. Reducer summaries remain "
            "indexes into authoritative runtime/checkpoint sources.",
            "",
            f"Chain SHA-256: `{chain['chain_sha256']}`",
            "",
        ]
    )
    return "\n".join(rows)


def _config_list(values: Sequence[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "None"


def _exploratory_report(consolidated: Mapping[str, Any]) -> str:
    """Render descriptive answers without treating stages as causal treatments."""

    chain = consolidated["chain"]
    config_rows = consolidated["configuration_rows"]
    divergence_rows = consolidated["seed_divergence_rows"]
    interventions = consolidated["intervention_rows"]
    by_stage_config = {
        (str(row["stage"]), str(row["config_id"])): row for row in config_rows
    }
    configs = sorted({str(row["config_id"]) for row in config_rows})

    def alive(stage: str, config_id: str) -> float:
        return float(by_stage_config[(stage, config_id)]["alive_terminal_mean"])

    trajectories = {
        config_id: [alive(stage, config_id) for stage in STAGE_ORDER]
        for config_id in configs
    }
    c5_rows = {
        config_id: by_stage_config[("C5", config_id)] for config_id in configs
    }
    viable = sorted(
        config_id
        for config_id, row in c5_rows.items()
        if int(row["viable_count"]) >= 2
    )
    resilient = sorted(
        configs, key=lambda config_id: (-alive("C5", config_id), config_id)
    )
    fragile = list(reversed(resilient))
    rebound_candidates = sorted(
        config_id
        for config_id in configs
        if alive("C2", config_id) < alive("C1", config_id)
        and alive("C3", config_id) > alive("C2", config_id)
    )
    nonmonotonic: list[str] = []
    for config_id, values in trajectories.items():
        signs = {
            1 if right > left else -1
            for left, right in zip(values, values[1:])
            if right != left
        }
        if len(signs) > 1:
            nonmonotonic.append(config_id)
    alive_divergence = {
        str(row["config_id"]): row
        for row in divergence_rows
        if row.get("stage") == "C5" and row.get("metric") == "alive"
    }
    low_agreement = sorted(
        config_id
        for config_id, row in alive_divergence.items()
        if row.get("agreement_band") == "LOW"
    )
    high_agreement = sorted(
        config_id
        for config_id, row in alive_divergence.items()
        if row.get("agreement_band") == "HIGH"
    )
    final_by_config: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    all_parameters_by_config: dict[str, set[str]] = defaultdict(set)
    for row in interventions:
        config_id = str(row["config_id"])
        all_parameters_by_config[config_id].add(str(row["parameter"]))
        if row.get("stage") == "C5":
            final_by_config[config_id].append(row)
    final_scores: list[tuple[float, str]] = []
    for config_id in sorted(final_by_config):
        baseline = alive("C4", config_id)
        change = abs(alive("C5", config_id) - baseline) / max(abs(baseline), 1.0)
        divergence = float(
            alive_divergence.get(config_id, {}).get("terminal_relative_range") or 0.0
        )
        final_scores.append((change + divergence, config_id))
    informative = [
        config_id
        for _score, config_id in sorted(
            final_scores, key=lambda item: (-item[0], item[1])
        )[:5]
    ]
    mixed_candidates = sorted(
        config_id
        for config_id, parameters in all_parameters_by_config.items()
        if len(parameters) >= 2
    )
    follow_up_reasons: dict[str, list[str]] = defaultdict(list)
    for config_id in fragile[:3]:
        follow_up_reasons[config_id].append("low C5 alive mean")
    for config_id in low_agreement:
        follow_up_reasons[config_id].append("large three-seed alive divergence")
    for config_id in rebound_candidates:
        follow_up_reasons[config_id].append("descriptive C2-to-C3 rebound")
    for config_id in nonmonotonic:
        follow_up_reasons[config_id].append("non-monotonic stage trajectory")
    follow_up = sorted(follow_up_reasons)
    final_intervention_lines: list[str] = []
    for config_id in informative:
        rows = final_by_config.get(config_id, [])
        parameters = ", ".join(
            f"`{row['parameter']}` {row['previous_value']!r}->{row['new_value']!r}"
            for row in sorted(rows, key=lambda row: str(row["parameter"]))
        )
        final_intervention_lines.append(
            f"- `{config_id}`: {parameters or 'no C5 override row'}; "
            f"C4/C5 alive mean {alive('C4', config_id):.6g}/"
            f"{alive('C5', config_id):.6g}."
        )
    follow_up_lines = [
        f"- `{config_id}`: {', '.join(follow_up_reasons[config_id])}."
        for config_id in follow_up
    ]
    checkpoint_rows = consolidated["checkpoint_rows"]
    checkpoint_ids = [str(row["checkpoint_sha256"]) for row in checkpoint_rows]
    rows = [
        "# WRC B48 G125 / 20,000-tick Adaptive Exploratory Report",
        "",
        "Interpretation: `EXPLORATORY_ADAPTIVE_INTERVENTION`",
        "",
        "This report describes one path-dependent 48-world continuation. It does "
        "not establish isolated knob causality, general evolutionary conclusions, "
        "or new qualification authority.",
        "",
        "## Chain summary",
        "",
        "- 48 worlds; 16 configuration identities x 3 fixed seed replicates.",
        "- Generation-boundary checkpoints at G25/G50/G75/G100/G125, T160.",
        "- Terminal horizon: G125 / 20,000 ticks per world.",
        f"- Chain identity: `{chain['chain_sha256']}`.",
        f"- Qualification receipt action: `{chain['qualification_receipt_action']}`.",
        "",
        "## Required questions",
        "",
        "1. **Did all 48 worlds remain continuous through all five checkpoints?** "
        f"Yes under the enforced chain contract: five distinct checkpoint identities "
        f"were verified ({', '.join(f'`{value}`' for value in checkpoint_ids)}).",
        "",
        "2. **Did every intervention reuse the same qualification receipt?** Yes. "
        "All four transition manifests and five checkpoint analyses resolved to one "
        "qualification/envelope/route/production identity with action `unchanged`.",
        "",
        f"3. **How many configurations remained viable?** {len(viable)} of 16 met "
        "the declared descriptive rule of at least two `VIABLE` seed replicates at C5: "
        f"{_config_list(viable)}.",
        "",
        "4. **Which configurations were most resilient?** By C5 mean alive count, "
        f"the top five were {_config_list(resilient[:5])}. This is a descriptive "
        "ranking, not a treatment-effect estimate.",
        "",
        "5. **Which were most fragile?** By the same C5 alive-count ranking, the "
        f"bottom five were {_config_list(fragile[:5])}.",
        "",
        "6. **Which recovered after Stage 3 relief?** The configurations showing a "
        "C1-to-C2 decline followed by a C2-to-C3 alive-mean increase were "
        f"{_config_list(rebound_candidates)}. They are rebound candidates; the "
        "continued-history design cannot attribute recovery to one control.",
        "",
        "7. **Which showed persistent history effects?** History dependence is not "
        "identified without counterfactual continuations. Non-monotonic trajectory "
        f"candidates for follow-up were {_config_list(nonmonotonic)}.",
        "",
        "8. **Which showed large three-seed divergence?** Using the reducer's C5 "
        f"alive `LOW` agreement band: {_config_list(low_agreement)}.",
        "",
        "9. **Which showed strong agreement across all three seeds?** Using the C5 "
        f"alive `HIGH` agreement band: {_config_list(high_agreement)}.",
        "",
        "10. **Did mixed pressure create outcomes distinct from simple pressure or "
        "relief?** The single-history design cannot establish that contrast. "
        "Configurations receiving two or more distinct intervention parameters, and "
        f"therefore suitable as mixed-pressure candidates, were {_config_list(mixed_candidates)}.",
        "",
        "11. **Which final adaptive interventions were most informative?** A "
        "descriptive ranking combining absolute C4-to-C5 alive movement and C5 seed "
        f"divergence selected {_config_list(informative)}:",
        "",
        *(final_intervention_lines or ["- None."]),
        "",
        "12. **Which trajectories deserve formal follow-up science?** Candidates "
        "were selected for low terminal viability, large seed divergence, rebound, "
        "or non-monotonic behavior:",
        "",
        *(follow_up_lines or ["- None under these descriptive rules."]),
        "",
        "13. **Did the experiment expose a control, checkpoint, reducer, or "
        "qualification defect?** No defect was detected by the enforced chain, "
        "runtime-readback, reducer-identity, or qualification-identity checks. This "
        "does not exclude defects outside those observed contracts; inspect the "
        "canonical reducer drill-back and runtime receipts for anomaly review.",
        "",
        "## Limitations",
        "",
        "- Stages inherit ecology, population, lineages, and adaptation from prior stages.",
        "- Stage summaries are not independent treatments.",
        "- Alive-count rankings are descriptive and do not replace multi-metric review.",
        "- Unavailable metrics were not inferred.",
        "- The reducer output remains an index into authoritative raw/checkpoint sources.",
        "",
    ]
    return "\n".join(rows)


def materialize_closeout_outputs(
    consolidated: Mapping[str, Any], *, output_dir: Path
) -> dict[str, Any]:
    root = output_dir.resolve()
    chain = consolidated["chain"]
    declared_consolidated = consolidated.get("consolidated_sha256")
    if (
        not isinstance(declared_consolidated, str)
        or _consolidated_sha256(consolidated) != declared_consolidated
    ):
        raise AdaptiveCloseoutError("consolidated closeout identity differs")
    report_input = {
        **dict(consolidated["report_input"]),
        "chain_sha256": chain["chain_sha256"],
        "consolidated_sha256": declared_consolidated,
    }
    payloads: dict[str, bytes] = {
        "initial_design": consolidated["initial_design_raw"],
        "interventions": _csv_text(
            consolidated["intervention_rows"], INTERVENTION_FIELDS
        ).encode(),
        "checkpoints": _csv_text(
            consolidated["checkpoint_rows"], CHECKPOINT_FIELDS
        ).encode(),
        "configuration_summary": _csv_text(
            consolidated["configuration_rows"], CONFIG_FIELDS
        ).encode(),
        "world_summary": _csv_text(
            consolidated["world_rows"], WORLD_FIELDS
        ).encode(),
        "seed_divergence": _csv_text(
            consolidated["seed_divergence_rows"], DIVERGENCE_FIELDS
        ).encode(),
        "world_reductions": consolidated["world_reductions_raw"],
        "report_input": (
            json.dumps(report_input, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "exploratory_report": _exploratory_report(consolidated).encode(),
        "integration_record": _integration_record(chain).encode(),
    }
    descriptors: dict[str, dict[str, Any]] = {}
    for name, raw in payloads.items():
        path = root / OUTPUT_NAMES[name]
        _atomic_write(path, raw)
        descriptors[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    receipt = {
        "schema_version": "eiel_wrc_b48_g125_adaptive_closeout_outputs_v1",
        "status": "CHAIN_VERIFIED_G125_T20000",
        "chain_sha256": chain["chain_sha256"],
        "outputs": descriptors,
        "qualification_receipt_action": "unchanged",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    atomic_write_json(root / "B48_ADAPTIVE_CHAIN_RECEIPT.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-binding", type=Path, required=True)
    parser.add_argument("--initial-design", type=Path, required=True)
    parser.add_argument(
        "--analysis-receipt", type=Path, action="append", required=True
    )
    parser.add_argument("--stage-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    consolidated = consolidate_adaptive_chain(
        design_binding=_load_json(args.design_binding, label="design binding"),
        initial_design_path=args.initial_design,
        analysis_receipts=[
            _load_json(path, label=f"analysis receipt {path}")
            for path in args.analysis_receipt
        ],
        stage_manifests=[
            _load_json(path, label=f"stage manifest {path}")
            for path in args.stage_manifest
        ],
    )
    receipt = materialize_closeout_outputs(
        consolidated, output_dir=args.output_dir
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
