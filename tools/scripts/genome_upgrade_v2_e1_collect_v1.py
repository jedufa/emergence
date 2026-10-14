#!/usr/bin/env python3
"""Collect exact Genome V2 E1 cell evidence from completed campaign roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Final, Mapping, Sequence


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.authoritative_state.schema import (  # noqa: E402
    content_sha256,
    file_sha256,
)
from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (  # noqa: E402
    PREREGISTRATION_SHA256 as C1_PREREGISTRATION_SHA256,
)
from engine.evolution.genome_upgrade_v2_e1_gate_v1 import (  # noqa: E402
    ARMS,
    CAMPAIGNS,
    CAMPAIGN_FOUNDER_PAYLOADS,
    OBSERVATION_COUNTS,
    PRIMARY_METRICS,
    REPRESENTATIVE_SEED,
    SEEDS,
    TASK_SEED,
    DECISION_D,
    E1AuthorityBindingV1,
    E1CellEvidenceV1,
    evaluate_e1_v1,
    founder_payload_sha256s_v1,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (  # noqa: E402
    checkpoint_authority_binding_v1,
    package_identity_sha256_v1,
    validate_checkpoint_authority_binding_v1,
)
from tools.scripts.genome_upgrade_v2_e1_gate_v1 import (  # noqa: E402
    GenomeV2E1RunnerError,
    authority_from_payload_v1,
)


COLLECTOR_SCHEMA_VERSION: Final[str] = "genome_v2_p10_e1_collector_v1"
PROVIDER_ID: Final[str] = "eiel-production-provider-v1"
RUNTIME_PROFILE_ID: Final[str] = "eiel-production-runtime-v1"


class GenomeV2E1CollectorError(RuntimeError):
    """A completed campaign root cannot support exact E1 evidence."""


def _fail(message: str) -> None:
    raise GenomeV2E1CollectorError(message)


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenomeV2E1CollectorError(
            f"{label} is not readable JSON: {path}"
        ) from exc


def _object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a JSON object")
    return dict(value)


def _array(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be a JSON array")
    return value


def _hash(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _join(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    mismatches = [
        name for name, item in expected.items() if value.get(name) != item
    ]
    if mismatches:
        _fail(f"{label} identity join mismatch: {sorted(mismatches)!r}")


def _declared_path(
    value: Any,
    *,
    expected: Path,
    label: str,
) -> None:
    if not isinstance(value, str) or Path(value).resolve() != expected.resolve():
        _fail(f"{label} path does not identify {expected}")


def _plan_payload(
    root: Path,
    authority: E1AuthorityBindingV1 | None,
) -> dict[str, Any]:
    plan = _object(
        _read_json(root / "execution_plan.json", label="execution plan"),
        label="execution plan",
    )
    campaign = plan.get("campaign_id")
    if campaign not in CAMPAIGNS:
        _fail(f"unexpected E1 campaign id: {campaign!r}")
    supplied_hash = _hash(plan.get("plan_sha256"), label="plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    if content_sha256(unsigned) != supplied_hash:
        _fail(f"{campaign}: execution plan content hash mismatch")
    if (
        authority is not None
        and authority.campaign_plan_sha256s.get(campaign) != supplied_hash
    ):
        _fail(f"{campaign}: execution plan differs from E1 authority")
    execution = _object(plan.get("execution"), label=f"{campaign} execution")
    _join(
        plan,
        {
            "resolved_profile": RUNTIME_PROFILE_ID,
        },
        label=f"{campaign} plan",
    )
    _join(
        execution,
        {
            "generations": 32,
            "ticks_per_generation": 80,
        },
        label=f"{campaign} execution",
    )
    qualification = _object(
        plan.get("qualification_key"), label=f"{campaign} qualification key"
    )
    _join(
        qualification,
        {
            "schema_version": "eiel_runtime_qualification_key_v1",
            "profile_id": RUNTIME_PROFILE_ID,
            "backend": "cuda_resident",
        },
        label=f"{campaign} qualification key",
    )
    if authority is not None:
        artifacts = _object(
            qualification.get("artifact_hashes"),
            label=f"{campaign} qualified artifacts",
        )
        _join(
            artifacts,
            {
                "cuda_extension": authority.artifact_sha256,
                "genome_upgrade_v2_package_receipt": (
                    authority.package_receipt_sha256
                ),
            },
            label=f"{campaign} qualified artifacts",
        )
    return plan


def _cell_identity(
    campaign: str, raw: Any
) -> tuple[str, str, int, dict[str, Any]]:
    cell = _object(raw, label=f"{campaign} compiled cell")
    cell_id = cell.get("cell_id")
    if not isinstance(cell_id, str) or not cell_id:
        _fail(f"{campaign}: compiled cell has no identity")
    seeds = _object(cell.get("seeds"), label=f"{cell_id} seeds")
    dimensions = _object(
        cell.get("dimensions"), label=f"{cell_id} dimensions"
    )
    parameters = _object(
        cell.get("parameters"), label=f"{cell_id} parameters"
    )
    ecology_seed = seeds.get("ecology")
    if (
        type(ecology_seed) is not int
        or ecology_seed not in SEEDS
        or dimensions.get("ecology_seed") != ecology_seed
        or seeds.get("task") != TASK_SEED
        or dimensions.get("task_seed") != TASK_SEED
    ):
        _fail(f"{campaign}/{cell_id}: seed identity mismatch")
    if campaign == CAMPAIGNS[0]:
        arm = dimensions.get("e1_arm")
        if (
            arm not in ARMS
            or parameters.get("genome_upgrade_v2.e1_arm_label") != arm
            or parameters.get("genome_upgrade_v2.founder_profile") != "neutral"
        ):
            _fail(f"{campaign}/{cell_id}: neutral arm identity mismatch")
    else:
        profile = dimensions.get("founder_profile")
        arm = "control" if profile == "neutral" else "treatment"
        if (
            arm not in ARMS
            or parameters.get("genome_upgrade_v2.founder_profile") != profile
            or CAMPAIGN_FOUNDER_PAYLOADS[campaign][arm] != profile
        ):
            _fail(f"{campaign}/{cell_id}: founder arm identity mismatch")
    return cell_id, arm, ecology_seed, cell


def _completed_campaign_manifest(
    root: Path,
    plan: Mapping[str, Any],
    execution_route_sha256s: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    campaign = str(plan["campaign_id"])
    route = execution_route_sha256s[campaign]
    manifest = _object(
        _read_json(root / "campaign_manifest.json", label="campaign manifest"),
        label="campaign manifest",
    )
    _join(
        manifest,
        {
            "schema_version": "eiel_campaign_execution_manifest_v1",
            "campaign_id": campaign,
            "plan_sha256": plan["plan_sha256"],
            "profile_id": RUNTIME_PROFILE_ID,
            "provider_id": PROVIDER_ID,
            "execution_route_sha256": route,
            "status": "completed",
        },
        label=f"{campaign} manifest",
    )
    batches = _array(plan.get("physical_batches"), label=f"{campaign} batches")
    summaries = _array(
        manifest.get("batches"), label=f"{campaign} batch summaries"
    )
    if len(batches) != len(summaries):
        _fail(f"{campaign}: batch manifest cardinality mismatch")
    result: dict[str, dict[str, Any]] = {}
    summary_by_id = {
        row.get("batch_id"): _object(row, label="batch summary")
        for row in summaries
        if isinstance(row, Mapping)
    }
    for batch in batches:
        batch = _object(batch, label=f"{campaign} physical batch")
        batch_id = batch.get("batch_id")
        if not isinstance(batch_id, str) or batch_id in result:
            _fail(f"{campaign}: invalid or duplicate physical batch")
        path = root / "batches" / batch_id / "manifest.json"
        summary = summary_by_id.get(batch_id)
        if summary is None:
            _fail(f"{campaign}: missing batch summary for {batch_id}")
        _join(
            summary,
            {"status": "completed"},
            label=f"{campaign}/{batch_id} summary",
        )
        _declared_path(
            summary.get("manifest_path"),
            expected=path,
            label=f"{campaign}/{batch_id} summary",
        )
        value = _object(
            _read_json(path, label=f"{campaign}/{batch_id} manifest"),
            label=f"{campaign}/{batch_id} manifest",
        )
        _join(
            value,
            {
                "schema_version": "eiel_campaign_batch_manifest_v1",
                "campaign_id": campaign,
                "plan_sha256": plan["plan_sha256"],
                "profile_id": RUNTIME_PROFILE_ID,
                "provider_id": PROVIDER_ID,
                "execution_route_sha256": route,
                "batch_id": batch_id,
                "cell_ids": batch.get("cell_ids"),
                "status": "completed",
            },
            label=f"{campaign}/{batch_id} manifest",
        )
        result[batch_id] = value
    if set(summary_by_id) != set(result):
        _fail(f"{campaign}: unexpected batch summaries")
    return manifest, result


def _cell_files(
    root: Path,
    plan: Mapping[str, Any],
    *,
    route_sha256: str,
    cells: Mapping[str, Mapping[str, Any]],
) -> None:
    campaign = str(plan["campaign_id"])
    index = _object(
        _read_json(root / "cell_receipts.json", label="cell receipt index"),
        label="cell receipt index",
    )
    _join(
        index,
        {
            "schema_version": "eiel_campaign_cell_receipt_index_v1",
            "campaign_id": campaign,
            "plan_sha256": plan["plan_sha256"],
            "completed_cell_count": len(cells),
        },
        label=f"{campaign} cell receipt index",
    )
    rows = _array(index.get("rows"), label=f"{campaign} cell receipt rows")
    if len(rows) != len(cells):
        _fail(f"{campaign}: cell receipt count differs from plan")
    seen: set[str] = set()
    for raw in rows:
        row = _object(raw, label=f"{campaign} cell receipt")
        cell_id = row.get("cell_id")
        if cell_id not in cells or cell_id in seen:
            _fail(f"{campaign}: invalid or duplicate cell receipt")
        seen.add(str(cell_id))
        cell_root = root / "cells" / str(cell_id)
        manifest_path = cell_root / "cell_manifest.json"
        completion_path = cell_root / "completion.json"
        _declared_path(
            row.get("cell_manifest_path"),
            expected=manifest_path,
            label=f"{campaign}/{cell_id} cell manifest",
        )
        _declared_path(
            row.get("completion_path"),
            expected=completion_path,
            label=f"{campaign}/{cell_id} completion",
        )
        manifest = _object(
            _read_json(manifest_path, label=f"{cell_id} cell manifest"),
            label=f"{cell_id} cell manifest",
        )
        expected_batch = next(
            batch["batch_id"]
            for batch in plan["physical_batches"]
            if cell_id in batch["cell_ids"]
        )
        _join(
            manifest,
            {
                "schema_version": "eiel_campaign_cell_manifest_v1",
                "campaign_id": campaign,
                "plan_sha256": plan["plan_sha256"],
                "profile_id": RUNTIME_PROFILE_ID,
                "provider_id": PROVIDER_ID,
                "execution_route_sha256": route_sha256,
                "batch_id": expected_batch,
                "cell": cells[str(cell_id)],
            },
            label=f"{campaign}/{cell_id} cell manifest",
        )
        _declared_path(
            manifest.get("batch_manifest_path"),
            expected=root / "batches" / expected_batch / "manifest.json",
            label=f"{campaign}/{cell_id} batch manifest",
        )
        completion = _object(
            _read_json(completion_path, label=f"{cell_id} completion"),
            label=f"{cell_id} completion",
        )
        _join(
            completion,
            {
                "schema_version": "eiel_campaign_cell_completion_v1",
                "campaign_id": campaign,
                "plan_sha256": plan["plan_sha256"],
                "cell_id": cell_id,
                "status": "completed",
                "terminal_checkpoint_generation": 32,
                "execution_route_sha256": route_sha256,
            },
            label=f"{campaign}/{cell_id} completion",
        )
    if seen != set(cells):
        _fail(f"{campaign}: cell receipts do not cover the plan")


def _public_metrics(
    root: Path,
    plan: Mapping[str, Any],
    cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    campaign = str(plan["campaign_id"])
    payload = _object(
        _read_json(root / "cell_metrics.json", label="cell metrics"),
        label="cell metrics",
    )
    _join(
        payload,
        {
            "schema_version": "eiel_campaign_cell_metrics_v1",
            "campaign_id": campaign,
            "plan_sha256": plan["plan_sha256"],
        },
        label=f"{campaign} cell metrics",
    )
    rows = _array(payload.get("rows"), label=f"{campaign} metric rows")
    result: dict[str, dict[str, int]] = {}
    required = (*PRIMARY_METRICS, *OBSERVATION_COUNTS)
    for raw in rows:
        row = _object(raw, label=f"{campaign} metric row")
        if set(row) != {"cell_id", "metrics"}:
            _fail(f"{campaign}: public metric row fields differ from contract")
        cell_id = row.get("cell_id")
        metrics = _object(row.get("metrics"), label=f"{cell_id} metrics")
        if cell_id not in cells or cell_id in result:
            _fail(f"{campaign}: invalid or duplicate public metric row")
        selected: dict[str, int] = {}
        for name in required:
            value = metrics.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail(f"{campaign}/{cell_id}: invalid E1 metric {name}")
            selected[name] = value
        result[str(cell_id)] = selected
    if set(result) != set(cells):
        _fail(f"{campaign}: public metrics do not exactly cover plan cells")
    return result


def _batch_evidence(
    campaign: str,
    batch: Mapping[str, Any],
    *,
    cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = _object(batch.get("result"), label=f"{campaign} batch result")
    terminal = _object(
        _object(
            result.get("checkpoint_catalog"),
            label=f"{campaign} checkpoint catalog",
        ).get("32"),
        label=f"{campaign} terminal checkpoint",
    )
    _join(
        terminal,
        {
            "generation": 32,
            "execution_route_sha256": batch["execution_route_sha256"],
        },
        label=f"{campaign} terminal checkpoint",
    )
    checkpoint_hashes = _object(
        terminal.get("canonical_checkpoint_storage_digests"),
        label=f"{campaign} checkpoint digests",
    )
    mechanics_hashes = _object(
        terminal.get("mechanics_checkpoint_digests"),
        label=f"{campaign} terminal mechanics digests",
    )
    runner_rows = _array(
        result.get("runner_receipts"), label=f"{campaign} runner receipts"
    )
    expected = set(str(value) for value in batch["cell_ids"])
    runners: dict[str, dict[str, Any]] = {}
    runner_counts: dict[str, int] = {}
    for raw in runner_rows:
        row = _object(raw, label=f"{campaign} runner receipt")
        cell_id = row.get("cell_id")
        if cell_id not in expected:
            _fail(f"{campaign}: runner receipt identifies an unknown cell")
        key = str(cell_id)
        if key in runners and runners[key] != row:
            _fail(f"{campaign}/{key}: segmented runner receipts differ")
        runners[key] = row
        runner_counts[key] = runner_counts.get(key, 0) + 1
    if (
        set(runners) != expected
        or len(set(runner_counts.values())) != 1
        or next(iter(runner_counts.values()), 0) not in {1, 2}
    ):
        _fail(f"{campaign}: runner receipts do not exactly cover batch cells")
    native = [
        _object(row, label=f"{campaign} native receipt")
        for row in _array(
            result.get("native_execution_receipts"),
            label=f"{campaign} native receipts",
        )
    ]
    if not native:
        _fail(f"{campaign}: native execution receipts are absent")
    for row in native:
        if (
            row.get("execution_device") != "cuda"
            or type(row.get("explicit_native_no_fallback")) is not bool
            or type(row.get("hidden_fallback")) is not bool
            or type(row.get("runtime_jit")) is not bool
        ):
            _fail(f"{campaign}: malformed native CUDA execution receipt")
    no_fallback = all(
        row["explicit_native_no_fallback"] and not row["hidden_fallback"]
        for row in native
    ) and result.get("fallback_used") is False
    no_jit = all(not row["runtime_jit"] for row in native)
    metric_rows = _array(
        batch.get("metrics_rows"), label=f"{campaign} native metric rows"
    )
    raw_metrics: dict[str, dict[str, Any]] = {}
    if (
        len(metric_rows) == len(expected)
        and all(
            isinstance(row, Mapping) and "generation_metrics" in row
            for row in metric_rows
        )
    ):
        for raw in metric_rows:
            row = _object(raw, label=f"{campaign} native metric row")
            cell_id = row.get("cell_id")
            if cell_id not in expected or cell_id in raw_metrics:
                _fail(f"{campaign}: invalid or duplicate native metric row")
            raw_metrics[str(cell_id)] = row
    else:
        grouped: dict[str, list[dict[str, Any]]] = {
            cell_id: [] for cell_id in expected
        }
        for raw in metric_rows:
            row = _object(raw, label=f"{campaign} generation metric row")
            cell_id = row.get("cell_id")
            if cell_id not in grouped:
                _fail(f"{campaign}: generation metric identifies an unknown cell")
            grouped[str(cell_id)].append(row)
        if any(len(rows) != 32 for rows in grouped.values()):
            _fail(
                f"{campaign}: flat generation metrics do not exactly cover "
                "32 generations per cell"
            )
        raw_metrics = {
            cell_id: {
                "cell_id": cell_id,
                "generation_metrics": rows,
            }
            for cell_id, rows in grouped.items()
        }
    if (
        set(checkpoint_hashes) != expected
        or set(mechanics_hashes) != expected
        or set(runners) != expected
        or set(raw_metrics) != expected
    ):
        _fail(f"{campaign}: batch evidence does not exactly cover batch cells")
    batch_sha256 = content_sha256(batch)
    native_sha256 = content_sha256(native)
    collected: dict[str, dict[str, Any]] = {}
    for cell_id in expected:
        checkpoint_sha256 = _hash(
            checkpoint_hashes[cell_id],
            label=f"{campaign}/{cell_id} checkpoint digest",
        )
        terminal_mechanics_sha256 = _hash(
            mechanics_hashes[cell_id],
            label=f"{campaign}/{cell_id} terminal mechanics digest",
        )
        runner = runners[cell_id]
        genome = _object(
            runner.get("genome"), label=f"{campaign}/{cell_id} genome receipt"
        )
        metrics = raw_metrics[cell_id]
        generation_rows = _array(
            metrics.get("generation_metrics"),
            label=f"{campaign}/{cell_id} generation metrics",
        )
        if len(generation_rows) != 32:
            _fail(f"{campaign}/{cell_id}: generation trajectory is incomplete")
        sequence: list[dict[str, Any]] = []
        for generation, raw in enumerate(generation_rows):
            row = _object(
                raw, label=f"{campaign}/{cell_id}/G{generation} metric"
            )
            if row.get("generation") != generation:
                _fail(
                    f"{campaign}/{cell_id}: generation trajectory is noncontiguous"
                )
            sequence.append(
                {
                    "generation": generation,
                    "generation_digest_sha256": content_sha256(row),
                }
            )
        collected[cell_id] = {
            "checkpoint_sha256": checkpoint_sha256,
            "runner": runner,
            "founder_profile_id": genome.get("founder_profile_id"),
            "founder_payload_sha256": genome.get("founder_profile_sha256"),
            "execution_receipt_sha256": content_sha256(
                {
                    "schema_version": (
                        "genome_v2_p10_e1_cell_execution_receipt_v1"
                    ),
                    "campaign_id": campaign,
                    "cell_id": cell_id,
                    "batch_manifest_sha256": batch_sha256,
                    "runner_receipt_sha256": content_sha256(runner),
                    "native_execution_receipts_sha256": native_sha256,
                    "terminal_checkpoint_sha256": checkpoint_sha256,
                }
            ),
            "trajectory_sha256": content_sha256(
                {
                    "schema_version": (
                        "genome_v2_p10_e1_bounded_trajectory_digest_v1"
                    ),
                    "campaign_id": campaign,
                    "cell_id": cell_id,
                    "generation_metric_row_digests": sequence,
                    "terminal_mechanics_checkpoint_sha256": (
                        terminal_mechanics_sha256
                    ),
                }
            ),
            "no_fallback": no_fallback,
            "no_jit": no_jit,
        }
    return collected


def _comparison_evidence(
    root: Path,
    plan: Mapping[str, Any],
    *,
    route_sha256: str,
    cell_arms: Mapping[str, tuple[str, int]],
) -> tuple[dict[str, dict[str, Any]], str]:
    campaign = str(plan["campaign_id"])
    summary = _object(
        _read_json(
            root / "comparisons_manifest.json", label="comparisons manifest"
        ),
        label="comparisons manifest",
    )
    comparison_plan = _object(
        plan.get("comparisons"), label=f"{campaign} comparison plan"
    )
    branches = _array(
        comparison_plan.get("branches"),
        label=f"{campaign} comparison branches",
    )
    reporting_requirement = _object(
        comparison_plan.get("reporting_parity"),
        label=f"{campaign} reporting parity requirement",
    )
    accepted_reporting_modes = _array(
        reporting_requirement.get("accepted_evidence_modes"),
        label=f"{campaign} accepted reporting parity modes",
    )
    if (
        reporting_requirement.get("required") is not True
        or "enabled_disabled_exact_terminal_state"
        not in accepted_reporting_modes
    ):
        _fail(f"{campaign}: reporting parity requirement is not frozen")
    _join(
        summary,
        {
            "schema_version": "eiel_campaign_comparisons_manifest_v1",
            "campaign_id": campaign,
            "plan_sha256": plan["plan_sha256"],
            "profile_id": RUNTIME_PROFILE_ID,
            "provider_id": PROVIDER_ID,
            "execution_route_sha256": route_sha256,
            "status": "completed",
        },
        label=f"{campaign} comparisons manifest",
    )
    summaries = _array(
        summary.get("branches"), label=f"{campaign} comparison summaries"
    )
    planned = {
        row.get("branch_id"): _object(row, label="planned comparison")
        for row in branches
        if isinstance(row, Mapping)
    }
    indexed = {
        row.get("branch_id"): _object(row, label="comparison summary")
        for row in summaries
        if isinstance(row, Mapping)
    }
    kind_counts = {
        kind: sum(row.get("kind") == kind for row in planned.values())
        for kind in (
            "fresh_repeat",
            "checkpoint_continuation",
            "reporting_parity",
        )
    }
    if (
        len(branches) != 5
        or len(planned) != 5
        or len(indexed) != 5
        or set(indexed) != set(planned)
        or kind_counts
        != {
            "fresh_repeat": 2,
            "checkpoint_continuation": 2,
            "reporting_parity": 1,
        }
    ):
        _fail(f"{campaign}: representative comparison matrix differs from plan")
    per_cell: dict[str, dict[str, Any]] = {}
    reporting_receipt_sha256: str | None = None
    for branch_id, branch in planned.items():
        source = branch.get("source_cell_id")
        if source not in cell_arms:
            _fail(f"{campaign}: comparison source cell is unknown")
        arm, seed = cell_arms[str(source)]
        kind = branch.get("kind")
        if (
            kind not in {
                "fresh_repeat",
                "checkpoint_continuation",
                "reporting_parity",
            }
            or (
                kind == "checkpoint_continuation"
                and branch.get("split_generation") != 16
            )
            or (
                kind in {"fresh_repeat", "checkpoint_continuation"}
                and seed != REPRESENTATIVE_SEED
            )
            or (
                kind == "reporting_parity"
                and (
                    branch.get("split_generation") is not None
                    or branch.get("repeat_ordinal") is not None
                )
            )
        ):
            _fail(f"{campaign}: comparison does not match frozen parity plan")
        path = root / "comparisons" / str(branch_id) / "manifest.json"
        row = indexed[branch_id]
        _join(
            row,
            {
                "kind": kind,
                "source_cell_id": source,
                "status": "completed",
            },
            label=f"{campaign}/{branch_id} comparison summary",
        )
        _declared_path(
            row.get("manifest_path"),
            expected=path,
            label=f"{campaign}/{branch_id} comparison",
        )
        manifest = _object(
            _read_json(path, label=f"{campaign}/{branch_id} comparison"),
            label=f"{campaign}/{branch_id} comparison",
        )
        _join(
            manifest,
            {
                "schema_version": "eiel_campaign_comparison_manifest_v1",
                "campaign_id": campaign,
                "plan_sha256": plan["plan_sha256"],
                "profile_id": RUNTIME_PROFILE_ID,
                "provider_id": PROVIDER_ID,
                "execution_route_sha256": route_sha256,
                "branch_id": branch_id,
                "kind": kind,
                "source_cell_id": source,
                "split_generation": branch.get("split_generation"),
                "repeat_ordinal": branch.get("repeat_ordinal"),
                "status": "completed",
            },
            label=f"{campaign}/{branch_id} comparison",
        )
        comparison = _object(
            manifest.get("comparison"),
            label=f"{campaign}/{branch_id} comparison receipt",
        )
        if kind == "reporting_parity":
            enabled_digest = _hash(
                comparison.get("reporting_enabled_terminal_digest"),
                label=f"{campaign}/{branch_id} reporting-enabled digest",
            )
            disabled_digest = _hash(
                comparison.get("reporting_disabled_terminal_digest"),
                label=f"{campaign}/{branch_id} reporting-disabled digest",
            )
            compared_surfaces = comparison.get("compared_surfaces")
            reporting_exact = (
                isinstance(compared_surfaces, list)
                and "terminal_checkpoint_digest" in compared_surfaces
                and set(compared_surfaces)
                <= {
                    "terminal_checkpoint_digest",
                    "transition_sentinels_sha256",
                }
                and enabled_digest == disabled_digest
                and comparison.get("state_mutation_sentinel")
                == enabled_digest
                and comparison.get("match") is True
                and comparison.get("first_difference") is None
                and comparison.get("mismatches") == []
                and comparison.get("missing_surfaces") == []
            )
            _join(
                comparison,
                {
                    "schema_version": (
                        "eiel_campaign_reporting_parity_comparison_v1"
                    ),
                    "source_cell_id": source,
                    "evidence_mode": (
                        "enabled_disabled_exact_terminal_state"
                    ),
                    "evidence_status": "reporting_parity_proven",
                    "execution_route_sha256": route_sha256,
                    "branch_role": "reporting_parity",
                },
                label=f"{campaign}/{branch_id} reporting parity",
            )
            if not reporting_exact:
                _fail(f"{campaign}/{branch_id}: reporting parity is not exact")
            receipt = _object(
                _read_json(
                    root / "reporting_parity_receipt.json",
                    label=f"{campaign} reporting parity receipt",
                ),
                label=f"{campaign} reporting parity receipt",
            )
            _join(
                receipt,
                {
                    "schema_version": (
                        "eiel_campaign_reporting_parity_receipt_v1"
                    ),
                    "campaign_id": campaign,
                    "plan_sha256": plan["plan_sha256"],
                    "profile_id": RUNTIME_PROFILE_ID,
                    "provider_id": PROVIDER_ID,
                    "execution_route_sha256": route_sha256,
                    "comparison_branch_id": branch_id,
                    "comparison_receipt_sha256": content_sha256(comparison),
                    "comparison_first_difference": None,
                    "qualification_evidence_status": (
                        "reporting_parity_proven"
                    ),
                    "status": "supported",
                    "evidence_mode": (
                        "enabled_disabled_exact_terminal_state"
                    ),
                    "observational_contract": (
                        "reporting_disabled_skips_compact_materialization_only_v1"
                    ),
                    "state_mutation_sentinel": enabled_digest,
                    "reason_code": None,
                    "reason": None,
                },
                label=f"{campaign} reporting parity receipt",
            )
            reporting_receipt_sha256 = content_sha256(receipt)
            continue
        baseline = _object(
            comparison.get("baseline"), label=f"{branch_id} baseline"
        )
        candidate = _object(
            comparison.get("candidate"), label=f"{branch_id} candidate"
        )
        common = comparison.get("compared_surfaces")
        calculated_exact = (
            isinstance(common, list)
            and bool(common)
            and set(common) == set(baseline) == set(candidate)
            and baseline == candidate
            and comparison.get("missing_surfaces") == []
            and comparison.get("mismatches") == []
            and comparison.get("first_difference") is None
        )
        if comparison.get("match") is not calculated_exact:
            _fail(f"{campaign}/{branch_id}: comparison exactness is inconsistent")
        target = per_cell.setdefault(
            str(source),
            {
                "arm": arm,
                "seed": seed,
                "fresh_repeat_exact": None,
                "continuation_g16_exact": None,
                "manifests": {},
            },
        )
        field = (
            "fresh_repeat_exact"
            if kind == "fresh_repeat"
            else "continuation_g16_exact"
        )
        if target[field] is not None:
            _fail(f"{campaign}/{source}: duplicate {kind} comparison")
        target[field] = calculated_exact
        target["manifests"][kind] = {
            "manifest_sha256": content_sha256(manifest),
            "comparison_receipt_sha256": content_sha256(comparison),
        }
    if reporting_receipt_sha256 is None:
        _fail(f"{campaign}: reporting parity evidence is incomplete")
    expected_sources = {
        cell_id
        for cell_id, (_arm, seed) in cell_arms.items()
        if seed == REPRESENTATIVE_SEED
    }
    if set(per_cell) != expected_sources:
        _fail(f"{campaign}: representative comparison cells are incomplete")
    for cell_id, value in per_cell.items():
        if (
            value["fresh_repeat_exact"] is None
            or value["continuation_g16_exact"] is None
            or set(value["manifests"])
            != {"fresh_repeat", "checkpoint_continuation"}
        ):
            _fail(f"{campaign}/{cell_id}: parity evidence is incomplete")
        value["parity_receipt_sha256"] = content_sha256(
            {
                "schema_version": (
                    "genome_v2_p10_e1_representative_parity_receipt_v1"
                ),
                "campaign_id": campaign,
                "cell_id": cell_id,
                "arm": value["arm"],
                "ecology_seed": value["seed"],
                "execution_route_sha256": route_sha256,
                "fresh_repeat": value["manifests"]["fresh_repeat"],
                "continuation_g16": value["manifests"][
                    "checkpoint_continuation"
                ],
            }
        )
    return per_cell, reporting_receipt_sha256


def _validated_package_authority_v1(
    package_receipt_path: Path,
) -> tuple[dict[str, Any], str, str]:
    receipt = _object(
        _read_json(package_receipt_path, label="package identity receipt"),
        label="package identity receipt",
    )
    if (
        receipt.get("schema_version")
        != "genome_v2_package_identity_receipt_v1"
        or receipt.get("ok") is not True
        or receipt.get("authority_state") != "package_candidate_complete"
    ):
        _fail("package identity receipt is not a complete T2 candidate")
    package_id = _hash(
        receipt.get("package_id_sha256"), label="package_id_sha256"
    )
    if package_identity_sha256_v1(receipt) != package_id:
        _fail("package identity receipt self-address is invalid")
    identity = _object(
        receipt.get("identity"), label="package receipt identity"
    )
    _join(
        identity,
        {
            "provider_id": PROVIDER_ID,
            "runtime_profile_id": RUNTIME_PROFILE_ID,
            "global_genome_profile_id": "genome_upgrade_v2",
            "source_clean": True,
            "no_fallback": True,
            "no_jit": True,
        },
        label="package receipt identity",
    )
    source_commit = identity.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        _fail("package source commit is invalid")
    _hash(identity.get("cuda_artifact_sha256"), label="package artifact")
    return identity, package_id, file_sha256(package_receipt_path)


def _validated_route_bindings_v1(
    value: Any,
    *,
    package_identity: Mapping[str, Any],
    package_id_sha256: str,
    package_receipt_sha256: str,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    payload = _object(value, label="campaign route bindings")
    if set(payload) != set(CAMPAIGNS):
        _fail("route binding campaigns differ from E1 contract")
    bindings: dict[str, dict[str, str]] = {}
    routes: dict[str, str] = {}
    for campaign in CAMPAIGNS:
        try:
            binding = validate_checkpoint_authority_binding_v1(
                payload[campaign]
            )
        except ValueError as exc:
            raise GenomeV2E1CollectorError(
                f"{campaign}: route binding is invalid"
            ) from exc
        route = binding["execution_route_sha256"]
        expected = checkpoint_authority_binding_v1(
            package_identity,
            package_id_sha256=package_id_sha256,
            package_receipt_sha256=package_receipt_sha256,
            execution_route_sha256=route,
        )
        if binding != expected:
            _fail(f"{campaign}: route binding differs from package authority")
        bindings[campaign] = binding
        routes[campaign] = route
    return bindings, routes


def _validated_c1_admission_v1(value: Any) -> tuple[str, str]:
    decision = _object(value, label="C1 decision")
    supplied = _hash(
        decision.get("decision_sha256"), label="C1 decision_sha256"
    )
    unsigned = dict(decision)
    unsigned.pop("decision_sha256")
    if (
        decision.get("decision") != "C1-A"
        or decision.get("preregistration_sha256")
        != C1_PREREGISTRATION_SHA256
        or content_sha256(unsigned) != supplied
    ):
        _fail("C1 decision is not exact content-addressed C1-A admission")
    return C1_PREREGISTRATION_SHA256, supplied


def issue_e1_authority_v1(
    *,
    package_receipt_path: Path,
    route_bindings: Any,
    c1_decision: Any,
    run_roots: Sequence[Path],
) -> E1AuthorityBindingV1:
    """Issue E1 authority only from exact package, route, C1, and run evidence."""

    if len(run_roots) != 3 or len({root.resolve() for root in run_roots}) != 3:
        _fail("authority issuance requires exactly three distinct run roots")
    identity, package_id, package_receipt_sha = (
        _validated_package_authority_v1(package_receipt_path)
    )
    bindings, routes = _validated_route_bindings_v1(
        route_bindings,
        package_identity=identity,
        package_id_sha256=package_id,
        package_receipt_sha256=package_receipt_sha,
    )
    c1_preregistration, c1_decision_sha = _validated_c1_admission_v1(
        c1_decision
    )
    plans: dict[str, str] = {}
    checkpoints: dict[str, str] = {}
    parity_receipts: dict[str, str] = {}
    reporting_parity_receipts: dict[str, str] = {}
    seen: set[str] = set()
    for root in run_roots:
        plan = _plan_payload(root, None)
        campaign = str(plan["campaign_id"])
        if campaign in seen:
            _fail(f"duplicate campaign run root: {campaign}")
        seen.add(campaign)
        plans[campaign] = str(plan["plan_sha256"])
        raw_cells = _array(plan.get("cells"), label=f"{campaign} cells")
        cell_arms: dict[str, tuple[str, int]] = {}
        for raw in raw_cells:
            cell_id, arm, seed, _cell = _cell_identity(campaign, raw)
            if cell_id in cell_arms:
                _fail(f"{campaign}: duplicate compiled cell")
            cell_arms[cell_id] = (arm, seed)
        _manifest, batches = _completed_campaign_manifest(
            root, plan, routes
        )
        for batch in batches.values():
            for cell_id, evidence in _batch_evidence(
                campaign, batch, cells={}
            ).items():
                if cell_id not in cell_arms:
                    _fail(f"{campaign}: checkpoint covers unknown cell")
                arm, seed = cell_arms[cell_id]
                key = f"{campaign}:{arm}:{seed}"
                if key in checkpoints:
                    _fail(f"{campaign}: duplicate checkpoint identity")
                checkpoints[key] = evidence["checkpoint_sha256"]
        parity, reporting_parity_sha256 = _comparison_evidence(
            root,
            plan,
            route_sha256=routes[campaign],
            cell_arms=cell_arms,
        )
        reporting_parity_receipts[campaign] = (
            reporting_parity_sha256
        )
        for cell_id, evidence in parity.items():
            arm, seed = cell_arms[cell_id]
            parity_receipts[
                f"{campaign}:{arm}:{seed}"
            ] = evidence["parity_receipt_sha256"]
    if seen != set(CAMPAIGNS):
        _fail("authority run roots differ from the E1 campaign set")
    authority = E1AuthorityBindingV1(
        source_commit=str(identity["source_commit"]),
        package_id_sha256=package_id,
        package_receipt_sha256=package_receipt_sha,
        artifact_sha256=str(identity["cuda_artifact_sha256"]),
        execution_route_sha256s=routes,
        checkpoint_authority_bindings=bindings,
        c1_preregistration_sha256=c1_preregistration,
        c1_decision_sha256=c1_decision_sha,
        c1_decision="C1-A",
        campaign_plan_sha256s=plans,
        founder_payload_sha256s=founder_payload_sha256s_v1(),
        checkpoint_sha256s=checkpoints,
        parity_receipt_sha256s=parity_receipts,
        reporting_parity_receipt_sha256s=(
            reporting_parity_receipts
        ),
    )
    probe = evaluate_e1_v1([], authority=authority)
    if probe.get("decision") == DECISION_D:
        _fail(f"issued E1 authority is invalid: {probe.get('reasons', [])!r}")
    # Re-run the complete collector against the issued identities before
    # allowing the authority to leave this process.
    collect_e1_cell_evidence_v1(authority=authority, run_roots=run_roots)
    return authority


def materialize_e1_authority_v1(
    *,
    package_receipt_path: Path,
    route_bindings: Any,
    c1_decision: Any,
    run_roots: Sequence[Path],
    output_path: Path,
) -> E1AuthorityBindingV1:
    authority = issue_e1_authority_v1(
        package_receipt_path=package_receipt_path,
        route_bindings=route_bindings,
        c1_decision=c1_decision,
        run_roots=run_roots,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="ascii") as stream:
            json.dump(
                authority.payload(),
                stream,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
    except FileExistsError as exc:
        raise GenomeV2E1CollectorError(
            "E1 authority output already exists; immutable overwrite refused"
        ) from exc
    return authority


def collect_e1_cell_evidence_v1(
    *,
    authority: E1AuthorityBindingV1,
    run_roots: Sequence[Path],
) -> list[dict[str, Any]]:
    """Validate three completed campaign roots and emit exact gate input rows."""

    if len(run_roots) != 3 or len({root.resolve() for root in run_roots}) != 3:
        _fail("collector requires exactly three distinct campaign run roots")
    authority_probe = evaluate_e1_v1([], authority=authority)
    if authority_probe.get("decision") == DECISION_D:
        _fail(
            "E1 authority is invalid: "
            f"{authority_probe.get('reasons', [])!r}"
        )
    collected: list[E1CellEvidenceV1] = []
    seen_campaigns: set[str] = set()
    for root in run_roots:
        plan = _plan_payload(root, authority)
        campaign = str(plan["campaign_id"])
        if campaign in seen_campaigns:
            _fail(f"duplicate campaign run root: {campaign}")
        seen_campaigns.add(campaign)
        route = authority.execution_route_sha256s[campaign]
        raw_cells = _array(plan.get("cells"), label=f"{campaign} cells")
        if len(raw_cells) != 12:
            _fail(f"{campaign}: expected exactly 12 compiled cells")
        cells: dict[str, dict[str, Any]] = {}
        identity: dict[str, tuple[str, int]] = {}
        expected_keys = {(arm, seed) for arm in ARMS for seed in SEEDS}
        observed_keys: set[tuple[str, int]] = set()
        for raw in raw_cells:
            cell_id, arm, seed, cell = _cell_identity(campaign, raw)
            if cell_id in cells or (arm, seed) in observed_keys:
                _fail(f"{campaign}: duplicate E1 cell identity")
            cells[cell_id] = cell
            identity[cell_id] = (arm, seed)
            observed_keys.add((arm, seed))
        if observed_keys != expected_keys:
            _fail(f"{campaign}: compiled E1 cell matrix is incomplete")
        campaign_manifest, batches = _completed_campaign_manifest(
            root, plan, authority.execution_route_sha256s
        )
        del campaign_manifest
        planned_batch_cells = [
            cell_id
            for batch in plan["physical_batches"]
            for cell_id in batch["cell_ids"]
        ]
        if (
            len(planned_batch_cells) != len(set(planned_batch_cells))
            or set(planned_batch_cells) != set(cells)
        ):
            _fail(f"{campaign}: physical batches do not partition cells")
        _cell_files(
            root,
            plan,
            route_sha256=route,
            cells=cells,
        )
        metrics = _public_metrics(root, plan, cells)
        native: dict[str, dict[str, Any]] = {}
        for batch in batches.values():
            for cell_id, evidence in _batch_evidence(
                campaign, batch, cells=cells
            ).items():
                if cell_id in native:
                    _fail(f"{campaign}: duplicate native cell evidence")
                native[cell_id] = evidence
        if set(native) != set(cells):
            _fail(f"{campaign}: native evidence does not cover plan cells")
        parity, reporting_parity_sha256 = _comparison_evidence(
            root,
            plan,
            route_sha256=route,
            cell_arms=identity,
        )
        if (
            reporting_parity_sha256
            != authority.reporting_parity_receipt_sha256s.get(campaign)
        ):
            _fail(f"{campaign}: reporting parity authority mismatch")
        route_authority_receipt_sha256 = content_sha256(
            {
                "schema_version": (
                    "genome_v2_p10_e1_route_authority_receipt_v1"
                ),
                "campaign_id": campaign,
                "execution_route_sha256": route,
                "checkpoint_authority_binding": (
                    authority.checkpoint_authority_bindings[campaign]
                ),
            }
        )
        for cell_id, (arm, seed) in identity.items():
            expected_profile = CAMPAIGN_FOUNDER_PAYLOADS[campaign][arm]
            evidence = native[cell_id]
            if (
                evidence["founder_profile_id"] != expected_profile
                or evidence["founder_payload_sha256"]
                != authority.founder_payload_sha256s[expected_profile]
            ):
                _fail(f"{campaign}/{cell_id}: founder receipt join mismatch")
            checkpoint_key = f"{campaign}:{arm}:{seed}"
            if (
                evidence["checkpoint_sha256"]
                != authority.checkpoint_sha256s.get(checkpoint_key)
            ):
                _fail(f"{campaign}/{cell_id}: checkpoint authority mismatch")
            representative = parity.get(cell_id)
            parity_sha256 = (
                representative["parity_receipt_sha256"]
                if representative is not None
                else route_authority_receipt_sha256
            )
            if representative is not None and (
                parity_sha256
                != authority.parity_receipt_sha256s.get(checkpoint_key)
            ):
                _fail(f"{campaign}/{cell_id}: parity authority mismatch")
            collected.append(
                E1CellEvidenceV1(
                    campaign_id=campaign,
                    arm=arm,
                    ecology_seed=seed,
                    task_seed=TASK_SEED,
                    authority_sha256=authority.authority_sha256(),
                    campaign_plan_sha256=str(plan["plan_sha256"]),
                    execution_route_sha256=route,
                    founder_payload_id=expected_profile,
                    founder_payload_sha256=evidence[
                        "founder_payload_sha256"
                    ],
                    checkpoint_sha256=evidence["checkpoint_sha256"],
                    execution_receipt_sha256=evidence[
                        "execution_receipt_sha256"
                    ],
                    trajectory_sha256=evidence["trajectory_sha256"],
                    route_parity_receipt_sha256=parity_sha256,
                    fresh_repeat_exact=(
                        representative["fresh_repeat_exact"]
                        if representative is not None
                        else False
                    ),
                    continuation_g16_exact=(
                        representative["continuation_g16_exact"]
                        if representative is not None
                        else False
                    ),
                    no_fallback=evidence["no_fallback"],
                    no_jit=evidence["no_jit"],
                    metrics=metrics[cell_id],
                )
            )
    if seen_campaigns != set(CAMPAIGNS) or len(collected) != 36:
        _fail("campaign roots do not form the exact 36-cell E1 matrix")
    ordered = sorted(
        collected,
        key=lambda row: (
            CAMPAIGNS.index(row.campaign_id),
            ARMS.index(row.arm),
            SEEDS.index(row.ecology_seed),
        ),
    )
    return _gate_payload([cell.payload() for cell in ordered])


def _gate_payload(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key not in {"schema_version", "cell_sha256"}
        }
        for row in rows
    ]


def materialize_e1_cell_evidence_v1(
    *,
    authority: E1AuthorityBindingV1,
    run_roots: Sequence[Path],
    output_path: Path,
) -> list[dict[str, Any]]:
    rows = collect_e1_cell_evidence_v1(
        authority=authority,
        run_roots=run_roots,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="ascii") as stream:
            json.dump(
                rows,
                stream,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
    except FileExistsError as exc:
        raise GenomeV2E1CollectorError(
            "cell evidence output already exists; immutable overwrite refused"
        ) from exc
    return rows


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--authority-output", type=Path)
    parser.add_argument("--package-receipt", type=Path)
    parser.add_argument("--route-bindings", type=Path)
    parser.add_argument("--c1-decision", type=Path)
    parser.add_argument(
        "--run-root", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        issuing = args.authority is None
        if issuing:
            if any(
                value is None
                for value in (
                    args.authority_output,
                    args.package_receipt,
                    args.route_bindings,
                    args.c1_decision,
                )
            ):
                _fail(
                    "authority issuance requires --authority-output, "
                    "--package-receipt, --route-bindings, and --c1-decision"
                )
            authority = materialize_e1_authority_v1(
                package_receipt_path=args.package_receipt,
                route_bindings=_read_json(
                    args.route_bindings, label="campaign route bindings"
                ),
                c1_decision=_read_json(
                    args.c1_decision, label="C1 decision"
                ),
                run_roots=args.run_root,
                output_path=args.authority_output,
            )
        else:
            if any(
                value is not None
                for value in (
                    args.authority_output,
                    args.package_receipt,
                    args.route_bindings,
                    args.c1_decision,
                )
            ):
                _fail(
                    "--authority cannot be combined with authority issuance inputs"
                )
            authority = authority_from_payload_v1(
                _read_json(args.authority, label="E1 authority")
            )
        rows = materialize_e1_cell_evidence_v1(
            authority=authority,
            run_roots=args.run_root,
            output_path=args.output,
        )
    except (GenomeV2E1CollectorError, GenomeV2E1RunnerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": COLLECTOR_SCHEMA_VERSION,
                "output_path": str(args.output),
                "authority_path": str(
                    args.authority_output if issuing else args.authority
                ),
                "authority_sha256": authority.authority_sha256(),
                "cell_count": len(rows),
                "cell_evidence_sha256": content_sha256(rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
