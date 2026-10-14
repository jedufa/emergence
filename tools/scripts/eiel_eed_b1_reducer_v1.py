from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping


class EedB1ReductionError(ValueError):
    pass


TREATMENTS = (
    "impact_on",
    "impact_off",
    "ecology_frozen",
    "phenotype_fixed",
)
FOUNDERS = ("a_high", "b_high", "symmetric_mix")
ECOLOGY_SEEDS = (101, 202, 303, 404, 505, 606)
TASK_SEED = 29741
PROVIDER_ID = "eiel-production-provider-v1"
PROFILE_ID = "eiel-production-runtime-v1"
SOURCE_COMMIT = "b479c70e83a7f7ad5fe04a8d5f70213cdaf585eb"
ARTIFACT_SHA256 = "d2643e53578d0706f2711c9a9790a036e78774424be5c77f03ca228507db2d76"
MANIFEST_SHA256 = "42a8d4af7377245d8f32c9cac3bf8be14e588f3bc69d3a0d0a6fc186affa1cfa"
PACKAGE_ID = "fa8eed7de804cf85220508884aff00c3746455fdcddc900181fc0f7864a15968"
PACKAGE_RECEIPT_SHA256 = "f059f79afc31f32e7cff9e381e93e71a0799ca6246aa80a01c7c80ce6b06b5c2"
QUALIFICATION_RECEIPT_SHA256 = "d32840ca14663ecc99f359497bd45383f36978cc4f43a56f09e2bdfdc31c847b"
REPORTING_RECEIPT_SHA256 = "6cf2eabfc5076cee4230066c76d18151d6e3f017cafed6a9e5ba40468370875a"
CONTRACT_SHA256 = QUALIFICATION_RECEIPT_SHA256
EXPECTED_COMPLETED_COMPARISONS = 6
ROUTES = {
    "impact_on": "3c67e91c937070e1db33615b99a362dc16cc54a2146d1f16874bbe6498cb3818",
    "impact_off": "a299962d77350b58f673cfdd153d9534b70055bda360077e7b3da842c555c898",
    "ecology_frozen": "9a11d1e30392fb69836496af4b23c13c4f3969fe0bed46886428ac435d621861",
    "phenotype_fixed": "3aa8bd8fec3a7839fb02e3608734f5bc98a889f3fb63c1558b68043cad2ee216",
}
CAMPAIGN_IDS = {
    treatment: f"eed-b1-science-{treatment}-g100" for treatment in TREATMENTS
}
REQUIRED_METRICS = (
    "population.alive",
    "population.survival_auc",
    "population.successful_births_cumulative",
    "eed.resource_a_stock_total",
    "eed.resource_b_stock_total",
    "eed.resource_a_impact_load_total",
    "eed.resource_b_impact_load_total",
    "eed.resource_a_stock_per_capita",
    "eed.resource_b_stock_per_capita",
    "eed.resource_a_impact_load_per_capita",
    "eed.resource_b_impact_load_per_capita",
    "eed.reproductive_opportunity_successful_births",
    "eed.reproductive_opportunity_survival_auc",
    "eed.resource_axis_mean",
)
ZERO_POPULATION_UNDEFINED_METRICS = frozenset(
    {
        "eed.resource_a_stock_per_capita",
        "eed.resource_b_stock_per_capita",
        "eed.resource_a_impact_load_per_capita",
        "eed.resource_b_impact_load_per_capita",
        "eed.resource_axis_mean",
    }
)
CONTRASTS = {
    "total_enabled_impact": ("impact_on", "impact_off"),
    "persistent_ecological_state": ("impact_on", "ecology_frozen"),
    "phenotype_specific_per_capita": ("impact_on", "phenotype_fixed"),
    "direct_handling_context": ("impact_off", "ecology_frozen"),
}


def _read_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EedB1ReductionError(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EedB1ReductionError(f"{name} must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EedB1ReductionError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EedB1ReductionError(f"{name} is nonfinite")
    return result


def _optional_number(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _number(value, name=name)


def _metric_is_zero_population_undefined(
    metrics: Mapping[str, Any],
    metric: str,
) -> bool:
    return (
        metric in ZERO_POPULATION_UNDEFINED_METRICS
        and metrics.get(metric) is None
        and _number(metrics.get("population.alive"), name="population.alive") == 0.0
    )


def validate_required_metrics(
    metrics: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    """Validate B1 EED metrics without converting extinction into fake data.

    Terminal per-capita and live-agent phenotype-axis metrics are undefined
    when no agents are alive.  That is a real scientific state, not missing
    runtime evidence.  The same null values remain invalid for any non-extinct
    metric row.
    """

    missing = [metric for metric in REQUIRED_METRICS if metric not in metrics]
    if missing:
        raise EedB1ReductionError(f"{name} lacks required metrics: {missing}")
    alive = _number(metrics.get("population.alive"), name=f"{name}/population.alive")
    if alive < 0 or not alive.is_integer():
        raise EedB1ReductionError(f"{name}/population.alive is invalid")
    finite_metrics: list[str] = []
    undefined_metrics: list[str] = []
    for metric in REQUIRED_METRICS:
        value = _optional_number(metrics.get(metric), name=f"{name}/{metric}")
        if value is not None:
            finite_metrics.append(metric)
            continue
        if metric in ZERO_POPULATION_UNDEFINED_METRICS and alive == 0:
            undefined_metrics.append(metric)
            continue
        raise EedB1ReductionError(f"{name}/{metric} is missing or undefined")
    return {
        "alive": int(alive),
        "finite_required_metrics": finite_metrics,
        "undefined_zero_population_metrics": undefined_metrics,
        "zero_population_terminal": alive == 0,
    }


def paired_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        raise EedB1ReductionError("paired summary has no values")
    finite = [_number(value, name="paired delta") for value in values]
    return {
        "count": len(finite),
        "values": finite,
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "minimum": min(finite),
        "maximum": max(finite),
        "positive": sum(value > 0 for value in finite),
        "zero": sum(value == 0 for value in finite),
        "negative": sum(value < 0 for value in finite),
    }


def paired_summary_with_undefined(
    values: list[float],
    *,
    expected_count: int,
    undefined_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    if values:
        summary = paired_summary(values)
    else:
        summary = {
            "count": 0,
            "values": [],
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "positive": 0,
            "zero": 0,
            "negative": 0,
        }
    return {
        **summary,
        "expected_count": expected_count,
        "finite_pair_count": len(values),
        "undefined_pair_count": len(undefined_pairs),
        "undefined_pairs": undefined_pairs,
    }


def paired_contrasts(
    cells: Mapping[str, Mapping[tuple[str, int], Mapping[str, Any]]],
    *,
    metrics: tuple[str, ...] = REQUIRED_METRICS,
    allow_zero_population_undefined: bool = False,
) -> dict[str, Any]:
    expected = {(founder, seed) for founder in FOUNDERS for seed in ECOLOGY_SEEDS}
    for treatment in TREATMENTS:
        observed = set(cells.get(treatment, {}))
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise EedB1ReductionError(
                f"{treatment} pairing mismatch: missing={missing}, extra={extra}"
            )
    output: dict[str, Any] = {}
    for name, (left, right) in CONTRASTS.items():
        founder_rows: dict[str, Any] = {}
        for founder in FOUNDERS:
            metric_rows: dict[str, Any] = {}
            for metric in metrics:
                deltas: list[float] = []
                undefined_pairs: list[dict[str, Any]] = []
                for seed in ECOLOGY_SEEDS:
                    left_cell = cells[left][(founder, seed)]
                    right_cell = cells[right][(founder, seed)]
                    left_value = _optional_number(
                        left_cell.get(metric),
                        name=f"{left}/{founder}/{seed}/{metric}",
                    )
                    right_value = _optional_number(
                        right_cell.get(metric),
                        name=f"{right}/{founder}/{seed}/{metric}",
                    )
                    if left_value is not None and right_value is not None:
                        deltas.append(left_value - right_value)
                        continue
                    left_undefined = _metric_is_zero_population_undefined(
                        left_cell, metric
                    )
                    right_undefined = _metric_is_zero_population_undefined(
                        right_cell, metric
                    )
                    if (
                        allow_zero_population_undefined
                        and (left_undefined or right_undefined)
                        and (left_value is not None or left_undefined)
                        and (right_value is not None or right_undefined)
                    ):
                        undefined_pairs.append(
                            {
                                "founder_profile": founder,
                                "ecology_seed": seed,
                                "left_zero_population_undefined": left_undefined,
                                "right_zero_population_undefined": right_undefined,
                            }
                        )
                        continue
                    _number(left_cell.get(metric), name=f"{left}/{founder}/{seed}/{metric}")
                    _number(right_cell.get(metric), name=f"{right}/{founder}/{seed}/{metric}")
                metric_rows[metric] = (
                    paired_summary_with_undefined(
                        deltas,
                        expected_count=len(ECOLOGY_SEEDS),
                        undefined_pairs=undefined_pairs,
                    )
                    if allow_zero_population_undefined
                    else paired_summary(deltas)
                )
            founder_rows[founder] = metric_rows
        output[name] = {
            "left_treatment": left,
            "right_treatment": right,
            "operation": "left_minus_right",
            "founders": founder_rows,
        }
    return output


def _checkpoint_metrics(
    batch: Mapping[str, Any],
    *,
    generation: int,
    cell_id: str,
) -> dict[str, Any]:
    result = batch.get("result")
    catalog = result.get("checkpoint_catalog") if isinstance(result, Mapping) else None
    checkpoint = catalog.get(str(generation)) if isinstance(catalog, Mapping) else None
    rows = checkpoint.get("metrics_rows") if isinstance(checkpoint, Mapping) else None
    matches = (
        [row for row in rows if isinstance(row, Mapping) and row.get("cell_id") == cell_id]
        if isinstance(rows, list)
        else []
    )
    if len(matches) != 1 or not isinstance(matches[0].get("metrics"), Mapping):
        raise EedB1ReductionError(
            f"{cell_id} lacks one G{generation} checkpoint metric row"
        )
    return dict(matches[0]["metrics"])


def _generation_windows(batch: Mapping[str, Any], *, cell_id: str) -> dict[str, Any]:
    rows = batch.get("metrics_rows")
    matches = (
        [row for row in rows if isinstance(row, Mapping) and row.get("cell_id") == cell_id]
        if isinstance(rows, list)
        else []
    )
    generation_rows = (
        matches[0].get("generation_metrics") if len(matches) == 1 else None
    )
    if not isinstance(generation_rows, list) or len(generation_rows) != 100:
        raise EedB1ReductionError(f"{cell_id} lacks 100 generation metric rows")
    by_generation = {
        row.get("generation"): row
        for row in generation_rows
        if isinstance(row, Mapping)
    }
    if set(by_generation) != set(range(100)):
        raise EedB1ReductionError(f"{cell_id} generation metric coverage is not G0-G99")

    def window(start: int, stop: int) -> dict[str, Any]:
        selected = [by_generation[generation] for generation in range(start, stop)]
        alive = [_number(row.get("alive"), name=f"{cell_id}/G{row.get('generation')}/alive") for row in selected]
        births = [
            _number(
                row.get("successful_births"),
                name=f"{cell_id}/G{row.get('generation')}/successful_births",
            )
            for row in selected
        ]
        return {
            "start_generation": start,
            "stop_generation_exclusive": stop,
            "alive_mean": statistics.fmean(alive),
            "alive_minimum": min(alive),
            "alive_maximum": max(alive),
            "successful_births_total": sum(births),
            "extinct_generation_count": sum(value == 0 for value in alive),
        }

    extinction_generations = [
        int(generation)
        for generation, row in sorted(by_generation.items())
        if _number(row.get("alive"), name=f"{cell_id}/G{generation}/alive") == 0
    ]

    return {
        "g0_g50": window(0, 50),
        "g50_g100": window(50, 100),
        "first_extinction_generation": (
            min(extinction_generations) if extinction_generations else None
        ),
        "terminal_alive": _number(
            by_generation[99].get("alive"), name=f"{cell_id}/G99/alive"
        ),
    }


def _artifact_hashes_from_batch(batch: Mapping[str, Any]) -> Mapping[str, Any]:
    checkpoint = batch.get("checkpoint")
    binding = (
        checkpoint.get("checkpoint_binding") if isinstance(checkpoint, Mapping) else None
    )
    hashes = binding.get("artifact_sha256s") if isinstance(binding, Mapping) else None
    if not isinstance(hashes, Mapping):
        raise EedB1ReductionError("batch lacks checkpoint artifact hash binding")
    expected = {
        "cuda_extension": ARTIFACT_SHA256,
        "extension_manifest": MANIFEST_SHA256,
        "genome_upgrade_v2_package_receipt": PACKAGE_RECEIPT_SHA256,
        "qualification_receipt": QUALIFICATION_RECEIPT_SHA256,
        "reporting_receipt": REPORTING_RECEIPT_SHA256,
    }
    for name, expected_sha256 in expected.items():
        if hashes.get(name) != expected_sha256:
            raise EedB1ReductionError(f"batch artifact hash binding mismatch: {name}")
    return hashes


def _runner_artifact_paths(batch: Mapping[str, Any]) -> Mapping[str, Path]:
    result = batch.get("result")
    receipts = result.get("runner_receipts") if isinstance(result, Mapping) else None
    if not isinstance(receipts, list) or not receipts:
        raise EedB1ReductionError("batch lacks runner artifact path bindings")
    rows: list[Mapping[str, Any]] = [
        receipt.get("artifact_bindings")
        for receipt in receipts
        if isinstance(receipt, Mapping)
        and isinstance(receipt.get("artifact_bindings"), Mapping)
    ]
    if not rows:
        raise EedB1ReductionError("batch runner artifact path binding is missing")
    first = rows[0]
    if any(row != first for row in rows):
        raise EedB1ReductionError("batch runner artifact path bindings disagree")
    required = ("cuda_extension", "qualification_receipt", "reporting_receipt")
    if any(not isinstance(first.get(name), str) for name in required):
        raise EedB1ReductionError("batch runner artifact path binding is incomplete")
    cuda_extension = Path(str(first["cuda_extension"]))
    return {
        "cuda_extension": cuda_extension,
        "extension_manifest": Path(str(first.get("extension_manifest", f"{cuda_extension}.manifest.json"))),
        "qualification_receipt": Path(str(first["qualification_receipt"])),
        "reporting_receipt": Path(str(first["reporting_receipt"])),
    }


def _verify_batch_artifact_authority(batch: Mapping[str, Any]) -> dict[str, Any]:
    _artifact_hashes_from_batch(batch)
    paths = _runner_artifact_paths(batch)
    for name, expected in (
        ("cuda_extension", ARTIFACT_SHA256),
        ("extension_manifest", MANIFEST_SHA256),
        ("qualification_receipt", QUALIFICATION_RECEIPT_SHA256),
        ("reporting_receipt", REPORTING_RECEIPT_SHA256),
    ):
        path = paths[name]
        if _sha256(path) != expected:
            raise EedB1ReductionError(f"{name} file hash mismatch: {path}")
    artifact_manifest = _read_object(paths["extension_manifest"], name="CUDA artifact manifest")
    snapshot = artifact_manifest.get("committed_source_snapshot")
    if not isinstance(snapshot, Mapping) or snapshot.get("build_source_commit") != SOURCE_COMMIT:
        raise EedB1ReductionError("CUDA artifact source commit mismatch")
    return {
        "source_commit": SOURCE_COMMIT,
        "cuda_artifact_sha256": ARTIFACT_SHA256,
        "extension_manifest_sha256": MANIFEST_SHA256,
        "package_id_sha256": PACKAGE_ID,
        "package_receipt_sha256": PACKAGE_RECEIPT_SHA256,
        "qualification_receipt_sha256": QUALIFICATION_RECEIPT_SHA256,
        "reporting_receipt_sha256": REPORTING_RECEIPT_SHA256,
    }


def _verify_artifact_authority(
    plan: Mapping[str, Any],
    *,
    fallback_batch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping):
        if fallback_batch is None:
            raise EedB1ReductionError("execution plan lacks artifacts")
        return _verify_batch_artifact_authority(fallback_batch)
    extension = artifacts.get("cuda_extension")
    manifest_row = artifacts.get("extension_manifest")
    package_row = artifacts.get("genome_upgrade_v2_package_receipt")
    for name, row, expected in (
        ("CUDA artifact", extension, ARTIFACT_SHA256),
        ("extension manifest", manifest_row, MANIFEST_SHA256),
    ):
        if not isinstance(row, Mapping) or row.get("sha256") != expected:
            raise EedB1ReductionError(f"{name} plan binding mismatch")
        path = Path(str(row.get("path")))
        if _sha256(path) != expected:
            raise EedB1ReductionError(f"{name} file hash mismatch: {path}")
    if not isinstance(package_row, Mapping):
        raise EedB1ReductionError("package receipt plan binding is missing")
    package_path = Path(str(package_row.get("path")))
    if _sha256(package_path) != package_row.get("sha256"):
        raise EedB1ReductionError("package receipt file hash mismatch")
    package = _read_object(package_path, name="Genome V2 package receipt")
    if package.get("package_id_sha256") != PACKAGE_ID:
        raise EedB1ReductionError("Genome V2 package identity mismatch")
    artifact_manifest = _read_object(
        Path(str(manifest_row.get("path"))), name="CUDA artifact manifest"
    )
    snapshot = artifact_manifest.get("committed_source_snapshot")
    if not isinstance(snapshot, Mapping) or snapshot.get("build_source_commit") != SOURCE_COMMIT:
        raise EedB1ReductionError("CUDA artifact source commit mismatch")
    return {
        "source_commit": SOURCE_COMMIT,
        "cuda_artifact_sha256": ARTIFACT_SHA256,
        "extension_manifest_sha256": MANIFEST_SHA256,
        "package_id_sha256": PACKAGE_ID,
        "package_receipt_sha256": package_row.get("sha256"),
    }


def _ordinary_genome_v2_birth_evidence(
    batch: Mapping[str, Any],
    terminal: Mapping[str, Any],
    *,
    cell_id: str,
) -> dict[str, Any]:
    births = _number(
        terminal.get("population.successful_births_cumulative"),
        name=f"{cell_id}/ordinary births",
    )
    eed_births = _number(
        terminal.get("eed.successful_births"),
        name=f"{cell_id}/EED successful births",
    )
    expression_count = _number(
        terminal.get("genome_v2.expression_observation_count"),
        name=f"{cell_id}/Genome V2 expression observations",
    )
    mutation_events = _number(
        terminal.get("eed.committed_mutation_events"),
        name=f"{cell_id}/committed mutation events",
    )
    hereditary_outcomes = _number(
        terminal.get("eed.hereditary_change_outcomes"),
        name=f"{cell_id}/hereditary change outcomes",
    )
    result = batch.get("result")
    receipts = result.get("runner_receipts") if isinstance(result, Mapping) else None
    if not isinstance(receipts, list) or not receipts:
        raise EedB1ReductionError(f"{cell_id} lacks runner receipts")
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise EedB1ReductionError(f"{cell_id} runner receipt is invalid")
        genome = receipt.get("genome")
        lifecycle = receipt.get("production_lifecycle")
        if (
            not isinstance(genome, Mapping)
            or genome.get("enabled") is not True
            or genome.get("inheritance_profile") != "diploid_linkage_v1"
            or not isinstance(lifecycle, Mapping)
            or lifecycle.get("ordinary_births_enabled") is not True
            or lifecycle.get("lifecycle_mode") != "hybrid_overlapping_v1"
        ):
            raise EedB1ReductionError(
                f"{cell_id} lacks bound Genome V2 ordinary-birth configuration"
            )
    if births <= 0 or eed_births <= 0:
        raise EedB1ReductionError(
            f"{cell_id} lacks positive ordinary Genome V2 birth evidence"
        )
    return {
        "ordinary_births_positive": True,
        "population_successful_births_cumulative": births,
        "eed_successful_births": eed_births,
        "genome_v2_enabled": True,
        "inheritance_profile": "diploid_linkage_v1",
        "ordinary_births_enabled": True,
        "genome_v2_expression_observation_count": expression_count,
        "committed_mutation_events": mutation_events,
        "hereditary_change_outcomes": hereditary_outcomes,
    }


def _positive_admission_index(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    if (
        manifest.get("schema_version")
        != "eiel_eed_b1_positive_scientific_admission_manifest_v1"
        or manifest.get("status") != "frozen"
        or manifest.get("scientific_cell_count") != 72
    ):
        raise EedB1ReductionError("positive scientific admission manifest is invalid")
    censoring = manifest.get("censoring_and_interruption_rules")
    if not isinstance(censoring, Mapping) or any(
        censoring.get(name) != disposition
        for name, disposition in {
            "incomplete_cells": "excluded_zero_science",
            "interrupted_attempts": "excluded_zero_science",
            "missing_terminal_generations": "excluded_zero_science",
            "superseded_attempts": "excluded_zero_science",
            "unpublished_checkpoints": "excluded_zero_science",
            "runtime_complete_reporting_incomplete": "excluded_zero_science",
            "integrity_excluded_cells": "excluded_zero_science",
        }.items()
    ):
        raise EedB1ReductionError("censoring and interruption rules are not frozen")
    items = manifest.get("admitted_items")
    if not isinstance(items, list) or len(items) != 72:
        raise EedB1ReductionError("positive admission item count mismatch")
    required = {
        "logical_cell_id",
        "execution_attempt_id",
        "treatment",
        "founder_profile",
        "seed",
        "source_commit",
        "artifact_sha256",
        "package_id",
        "route_sha256",
        "qualification_key_sha256",
        "path",
        "file_size",
        "sha256",
        "scientific_role",
        "admission_status",
    }
    indexed: dict[str, dict[str, Mapping[str, Any]]] = {
        treatment: {} for treatment in TREATMENTS
    }
    for item in items:
        if not isinstance(item, Mapping) or not required.issubset(item):
            raise EedB1ReductionError("positive admission item is incomplete")
        treatment = item.get("treatment")
        cell_id = item.get("logical_cell_id")
        if treatment not in indexed or not isinstance(cell_id, str):
            raise EedB1ReductionError("positive admission identity is invalid")
        if cell_id in indexed[str(treatment)]:
            raise EedB1ReductionError("duplicate positive scientific admission")
        if (
            item.get("source_commit") != SOURCE_COMMIT
            or item.get("artifact_sha256") != ARTIFACT_SHA256
            or item.get("package_id") != PACKAGE_ID
            or item.get("route_sha256") != ROUTES[str(treatment)]
            or item.get("scientific_role") != "primary_terminal_science_cell"
            or item.get("admission_status") != "admitted"
        ):
            raise EedB1ReductionError("positive admission authority mismatch")
        path = Path(str(item.get("path"))).resolve()
        if any(
            component in path.parts
            for component in (
                "comparison_recovery_archive",
                "batch_attempt_recovery_archive",
            )
        ):
            raise EedB1ReductionError("recovery path entered positive admission")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise EedB1ReductionError(
                f"positive admission path is unavailable: {path}"
            ) from exc
        if size != item.get("file_size") or _sha256(path) != item.get("sha256"):
            raise EedB1ReductionError("positive admission file identity mismatch")
        indexed[str(treatment)][cell_id] = item
    for treatment, rows in indexed.items():
        if len(rows) != 18:
            raise EedB1ReductionError(
                f"{treatment} positive admission count is not 18"
            )
    return indexed


def _load_treatment(
    run_root: Path,
    treatment: str,
    admissions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    plan = _read_object(run_root / "execution_plan.json", name="execution plan")
    campaign = _read_object(run_root / "campaign_manifest.json", name="campaign manifest")
    comparisons = _read_object(
        run_root / "comparisons_manifest.json", name="comparisons manifest"
    )
    reporting = _read_object(
        run_root / "reporting_parity_receipt.json", name="reporting parity receipt"
    )
    expected_campaign = CAMPAIGN_IDS[treatment]
    if plan.get("campaign_id") != expected_campaign:
        raise EedB1ReductionError(f"{treatment} campaign identity mismatch")
    if campaign.get("status") != "completed":
        raise EedB1ReductionError(f"{treatment} primary campaign is not completed")
    if campaign.get("counts") != {
        "completed": 18,
        "failed": 0,
        "interrupted": 0,
        "pending": 0,
        "running": 0,
    }:
        raise EedB1ReductionError(f"{treatment} primary campaign counts mismatch")
    if comparisons.get("status") != "completed":
        raise EedB1ReductionError(f"{treatment} comparisons are not completed")
    comparison_counts = comparisons.get("counts")
    if comparison_counts != {
        "completed": EXPECTED_COMPLETED_COMPARISONS,
        "failed": 0,
        "interrupted": 0,
        "pending": 0,
        "running": 0,
    }:
        raise EedB1ReductionError(f"{treatment} comparison count mismatch")
    if reporting.get("qualification_evidence_status") != "reporting_parity_proven":
        raise EedB1ReductionError(f"{treatment} reporting parity is not proven")
    route = ROUTES[treatment]
    for owner in (campaign, comparisons, reporting):
        if owner.get("provider_id") != PROVIDER_ID or owner.get("profile_id") != PROFILE_ID:
            raise EedB1ReductionError(f"{treatment} provider/profile binding mismatch")
        if owner.get("execution_route_sha256") != route:
            raise EedB1ReductionError(f"{treatment} route binding mismatch")
    cells = plan.get("cells")
    batches = plan.get("physical_batches")
    if not isinstance(cells, list) or not isinstance(batches, list):
        raise EedB1ReductionError(f"{treatment} plan matrix is invalid")
    cell_by_id = {
        row.get("cell_id"): row for row in cells if isinstance(row, Mapping)
    }
    batch_by_cell: dict[str, Mapping[str, Any]] = {}
    admission_by_cell: dict[str, Mapping[str, Any]] = {}
    for batch_row in batches:
        if not isinstance(batch_row, Mapping):
            continue
        cell_ids = batch_row.get("cell_ids")
        if not isinstance(cell_ids, list) or len(cell_ids) != 1:
            raise EedB1ReductionError(f"{treatment} is not physical B1")
        cell_id = str(cell_ids[0])
        admission = admissions.get(cell_id)
        if not isinstance(admission, Mapping):
            raise EedB1ReductionError(
                f"{treatment}/{cell_id} lacks explicit positive admission"
            )
        batch_path = Path(str(admission.get("path"))).resolve()
        expected_batch_path = (
            run_root
            / "batches"
            / str(batch_row.get("batch_id"))
            / "manifest.json"
        ).resolve()
        if batch_path != expected_batch_path:
            raise EedB1ReductionError(
                f"{treatment}/{cell_id} admission path is not canonical"
            )
        batch = _read_object(batch_path, name="batch manifest")
        if (
            admission.get("execution_attempt_id")
            != f"{batch.get('batch_id')}:attempt-{batch.get('attempts')}"
            or admission.get("qualification_key_sha256")
            != plan.get("qualification_key_sha256")
            or admission.get("founder_profile")
            != cell_by_id.get(cell_id, {}).get("dimensions", {}).get(
                "founder_profile"
            )
            or admission.get("seed")
            != cell_by_id.get(cell_id, {}).get("seeds")
        ):
            raise EedB1ReductionError(
                f"{treatment}/{cell_id} positive admission binding mismatch"
            )
        batch_by_cell[cell_id] = batch
        admission_by_cell[cell_id] = admission
    try:
        fallback_batch = next(iter(batch_by_cell.values()))
    except StopIteration as exc:
        raise EedB1ReductionError(f"{treatment} lacks physical batch manifests") from exc
    authority = _verify_artifact_authority(plan, fallback_batch=fallback_batch)
    expected_keys = {(founder, seed) for founder in FOUNDERS for seed in ECOLOGY_SEEDS}
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    intermediate: dict[tuple[str, int], dict[str, Any]] = {}
    windows: dict[tuple[str, int], dict[str, Any]] = {}
    cell_audit: list[dict[str, Any]] = []
    for cell_id, cell in cell_by_id.items():
        dimensions = cell.get("dimensions")
        if not isinstance(cell_id, str) or not isinstance(dimensions, Mapping):
            raise EedB1ReductionError(f"{treatment} cell identity is invalid")
        key = (dimensions.get("founder_profile"), dimensions.get("ecology_seed"))
        if key not in expected_keys or dimensions.get("task_seed") != TASK_SEED:
            raise EedB1ReductionError(f"{treatment}/{cell_id} matrix dimensions mismatch")
        if key in observed:
            raise EedB1ReductionError(f"{treatment} duplicate scientific cell: {key}")
        batch = batch_by_cell.get(cell_id)
        if not isinstance(batch, Mapping):
            raise EedB1ReductionError(f"{treatment}/{cell_id} lacks one physical batch")
        result = batch.get("result")
        checkpoint = batch.get("checkpoint")
        if (
            batch.get("status") != "completed"
            or batch.get("provider_id") != PROVIDER_ID
            or batch.get("profile_id") != PROFILE_ID
            or batch.get("execution_route_sha256") != route
            or not isinstance(result, Mapping)
            or result.get("fallback_used") is not False
            or result.get("execution_device") != "cuda"
            or result.get("terminal_generation") != 100
            or not isinstance(checkpoint, Mapping)
            or checkpoint.get("generation") != 100
            or checkpoint.get("execution_route_sha256") != route
        ):
            raise EedB1ReductionError(f"{treatment}/{cell_id} execution audit failed")
        terminal = _checkpoint_metrics(batch, generation=100, cell_id=cell_id)
        g50 = _checkpoint_metrics(batch, generation=50, cell_id=cell_id)
        terminal_metric_audit = validate_required_metrics(
            terminal,
            name=f"{treatment}/{cell_id}/G100",
        )
        g50_metric_audit = validate_required_metrics(
            g50,
            name=f"{treatment}/{cell_id}/G50",
        )
        birth_evidence = _ordinary_genome_v2_birth_evidence(
            batch,
            terminal,
            cell_id=f"{treatment}/{cell_id}",
        )
        observed[key] = terminal
        intermediate[key] = g50
        windows[key] = _generation_windows(batch, cell_id=cell_id)
        cell_audit.append(
            {
                "cell_id": cell_id,
                "founder_profile": key[0],
                "ecology_seed": key[1],
                "task_seed": TASK_SEED,
                "batch_id": batch.get("batch_id"),
                "execution_attempt_id": admission_by_cell[cell_id].get(
                    "execution_attempt_id"
                ),
                "admission_status": "admitted",
                "admitted_manifest_path": admission_by_cell[cell_id].get("path"),
                "admitted_manifest_sha256": admission_by_cell[cell_id].get("sha256"),
                "plan_sha256": plan.get("plan_sha256"),
                "qualification_key_sha256": plan.get("qualification_key_sha256"),
                "execution_route_sha256": route,
                "terminal_generation": 100,
                "fallback_used": False,
                "finite_required_metrics": not terminal_metric_audit[
                    "undefined_zero_population_metrics"
                ],
                "undefined_zero_population_metrics": terminal_metric_audit[
                    "undefined_zero_population_metrics"
                ],
                "g50_undefined_zero_population_metrics": g50_metric_audit[
                    "undefined_zero_population_metrics"
                ],
                "ordinary_genome_v2_birth_evidence": birth_evidence,
            }
        )
    if set(observed) != expected_keys or len(cell_by_id) != 18:
        raise EedB1ReductionError(f"{treatment} does not contain the frozen 18-cell matrix")
    return {
        "authority": {
            **authority,
            "campaign_id": expected_campaign,
            "plan_sha256": plan.get("plan_sha256"),
            "qualification_key_sha256": plan.get("qualification_key_sha256"),
            "execution_route_sha256": route,
            "provider_id": PROVIDER_ID,
            "profile_id": PROFILE_ID,
        },
        "cells": observed,
        "g50_cells": intermediate,
        "generation_windows": windows,
        "cell_audit": sorted(
            cell_audit,
            key=lambda row: (
                FOUNDERS.index(str(row["founder_profile"])),
                int(row["ecology_seed"]),
            ),
        ),
    }


def reduce_evidence(
    evidence_root: Path,
    positive_admission_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    reducer_source_path = Path(__file__).resolve()
    admission_index = _positive_admission_index(positive_admission_manifest)
    loaded: dict[str, dict[str, Any]] = {}
    for treatment in TREATMENTS:
        run_root = (
            evidence_root
            / "10_b1_science_runs"
            / treatment
            / CAMPAIGN_IDS[treatment]
        )
        loaded[treatment] = _load_treatment(
            run_root,
            treatment,
            admission_index[treatment],
        )
    terminal_cells = {name: row["cells"] for name, row in loaded.items()}
    g50_cells = {name: row["g50_cells"] for name, row in loaded.items()}
    window_metrics: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for treatment, row in loaded.items():
        window_metrics[treatment] = {}
        for key, windows in row["generation_windows"].items():
            window_metrics[treatment][key] = {
                "population.alive_mean_g0_g50": windows["g0_g50"]["alive_mean"],
                "population.alive_mean_g50_g100": windows["g50_g100"]["alive_mean"],
                "population.successful_births_g0_g50": windows["g0_g50"][
                    "successful_births_total"
                ],
                "population.successful_births_g50_g100": windows["g50_g100"][
                    "successful_births_total"
                ],
                "population.extinct_generation_count_g0_g50": windows["g0_g50"][
                    "extinct_generation_count"
                ],
                "population.extinct_generation_count_g50_g100": windows["g50_g100"][
                    "extinct_generation_count"
                ],
            }
    return {
        "schema_version": "eiel_eed_b1_reduction_v1",
        "decision_authority": "descriptive_evidence_only_no_new_significance_threshold",
        "contract": {
            "schema_version": "eiel_eed_b1_reducer_source_contract_binding_v1",
            "reducer_source_path": str(reducer_source_path),
            "reducer_source_sha256": _sha256(reducer_source_path),
            "legacy_external_contract_file_status": "absent_from_current_authority_tree",
            "expected_completed_comparisons_per_treatment": EXPECTED_COMPLETED_COMPARISONS,
        },
        "matrix": {
            "logical_cells": 72,
            "physical_batch_cells": 1,
            "generations": 100,
            "ticks_per_generation": 80,
            "treatments": list(TREATMENTS),
            "founder_profiles": list(FOUNDERS),
            "ecology_seeds": list(ECOLOGY_SEEDS),
            "task_seed": TASK_SEED,
        },
        "authority": {
            treatment: loaded[treatment]["authority"] for treatment in TREATMENTS
        },
        "provenance_audit": {
            "status": "pass",
            "audited_primary_cells": 72,
            "duplicate_scientific_contributions": 0,
            "exclusions": [],
            "positive_admission_manifest_required": True,
            "all_scientific_contributions_explicitly_admitted": True,
            "cells": [
                {"treatment": treatment, **row}
                for treatment in TREATMENTS
                for row in loaded[treatment]["cell_audit"]
            ],
        },
        "paired_contrasts": {
            "g50_intermediate": paired_contrasts(
                g50_cells,
                allow_zero_population_undefined=True,
            ),
            "g100_terminal": paired_contrasts(
                terminal_cells,
                allow_zero_population_undefined=True,
            ),
            "generation_windows": paired_contrasts(
                window_metrics,
                metrics=(
                    "population.alive_mean_g0_g50",
                    "population.alive_mean_g50_g100",
                    "population.successful_births_g0_g50",
                    "population.successful_births_g50_g100",
                    "population.extinct_generation_count_g0_g50",
                    "population.extinct_generation_count_g50_g100",
                ),
            ),
        },
        "interpretation_boundary": {
            "formal_mediation_claim": False,
            "causal_wording_requires_temporal_ordering": True,
            "unsupported_claims": [
                "intelligence",
                "emergence",
                "consciousness",
                "open-ended evolution",
                "stable polymorphism",
                "broad ecological feedback",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reduce the frozen EED B1 matrix")
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--admission-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    admission_manifest = _read_object(
        args.admission_manifest.resolve(),
        name="positive scientific admission manifest",
    )
    result = reduce_evidence(args.evidence_root.resolve(), admission_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, "output": str(args.output), "status": "pass"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
