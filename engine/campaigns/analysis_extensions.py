"""Registered typed extensions for provider-owned scientific campaign analysis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import canonical_sha256


class CampaignAnalysisExtensionError(ValueError):
    """An analysis extension or its authoritative inputs are malformed."""


AnalysisCallable = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]], Path | None, Mapping[str, Any]],
    dict[str, Any],
]
RenderCallable = Callable[[Mapping[str, Any]], str]


@dataclass(frozen=True, slots=True)
class CampaignAnalysisExtensionV1:
    extension_id: str
    analyze: AnalysisCallable
    render_markdown: RenderCallable


class CampaignAnalysisExtensionRegistryV1:
    def __init__(self) -> None:
        self._extensions: dict[str, CampaignAnalysisExtensionV1] = {}

    def register(self, extension: CampaignAnalysisExtensionV1) -> None:
        if (
            not isinstance(extension.extension_id, str)
            or not extension.extension_id
            or extension.extension_id in self._extensions
        ):
            raise CampaignAnalysisExtensionError(
                "analysis extension ID is invalid or duplicated"
            )
        self._extensions[extension.extension_id] = extension

    def resolve(self, extension_id: str) -> CampaignAnalysisExtensionV1:
        try:
            return self._extensions[extension_id]
        except KeyError as exc:
            raise CampaignAnalysisExtensionError(
                f"analysis extension is not registered: {extension_id!r}"
            ) from exc


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _bound(
    value: Mapping[str, Any] | None,
    plan: Mapping[str, Any],
) -> bool:
    return bool(
        value
        and value.get("campaign_id") == plan.get("campaign_id")
        and value.get("plan_sha256") == plan.get("plan_sha256")
    )


def _recomb72_identity_join(
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, tuple[str, str, int]], bool]:
    plan_cells = plan.get("cells")
    expected: dict[str, tuple[str, str, int]] = {}
    if not isinstance(plan_cells, list):
        return {}, False
    for cell in plan_cells:
        dimensions = cell.get("dimensions") if isinstance(cell, Mapping) else None
        cell_id = cell.get("cell_id") if isinstance(cell, Mapping) else None
        if not isinstance(cell_id, str) or not isinstance(dimensions, Mapping):
            return {}, False
        try:
            identity = (
                str(dimensions.get("founder_pattern") or ""),
                str(dimensions.get("inheritance_mode") or ""),
                int(dimensions.get("paired_seed", -1)),
            )
        except (TypeError, ValueError):
            return {}, False
        if cell_id in expected:
            return {}, False
        expected[cell_id] = identity
    scientific: dict[str, tuple[str, str, int]] = {}
    for row in rows:
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str):
            return expected, False
        try:
            identity = (
                str(row.get("pattern") or ""),
                str(row.get("mode") or ""),
                int(row.get("paired_seed", -1)),
            )
        except (TypeError, ValueError):
            return expected, False
        previous = scientific.setdefault(cell_id, identity)
        if previous != identity:
            return expected, False
    return expected, bool(
        len(expected) == 72
        and set(scientific) == set(expected)
        and scientific == expected
    )


def _recomb72_external_gates(
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    run_root: Path | None,
    context: Mapping[str, Any],
) -> dict[str, bool]:
    from engine.evolution.recomb72_decision import REQUIRED_EXTERNAL_GATES

    supplied = context.get("external_gates")
    if isinstance(supplied, Mapping):
        return {
            name: supplied.get(name) is True for name in REQUIRED_EXTERNAL_GATES
        }
    gates = {name: False for name in REQUIRED_EXTERNAL_GATES}
    if run_root is None:
        return gates
    root = run_root.expanduser().resolve()
    campaign = _read_json(root / "campaign_manifest.json")
    technical = _read_json(root / "technical_gate_receipt.json")
    comparisons = _read_json(root / "comparisons_manifest.json")
    reporting = _read_json(root / "reporting_parity_receipt.json")
    cells = _read_json(root / "cell_receipts.json")
    chronicle = _read_json(root / "chronicle_archive_receipt.json")

    plan_cells = plan.get("cells")
    expected_identity, exact_scientific_join = _recomb72_identity_join(plan, rows)
    expected_ids = set(expected_identity)

    counts = campaign.get("counts") if isinstance(campaign, Mapping) else None
    plan_batches = plan.get("physical_batches")
    campaign_batches = (
        campaign.get("batches") if isinstance(campaign, Mapping) else None
    )
    expected_batches = {
        str(batch["batch_id"]): tuple(str(value) for value in batch["cell_ids"])
        for batch in plan_batches
        if isinstance(batch, Mapping)
        and isinstance(batch.get("batch_id"), str)
        and isinstance(batch.get("cell_ids"), list)
    } if isinstance(plan_batches, list) else {}
    observed_batches = {
        str(batch["batch_id"]): str(batch.get("status"))
        for batch in campaign_batches
        if isinstance(batch, Mapping) and isinstance(batch.get("batch_id"), str)
    } if isinstance(campaign_batches, list) else {}
    exact_completed_batches = bool(
        isinstance(plan_batches, list)
        and isinstance(campaign_batches, list)
        and len(expected_batches) == len(plan_batches) == len(observed_batches)
        and set(observed_batches) == set(expected_batches)
        and all(status == "completed" for status in observed_batches.values())
        and len(
            [
                cell_id
                for cell_ids in expected_batches.values()
                for cell_id in cell_ids
            ]
        )
        == 72
        and {
            cell_id
            for cell_ids in expected_batches.values()
            for cell_id in cell_ids
        }
        == expected_ids
    )
    gates["authoritative_execution"] = bool(
        _bound(campaign, plan)
        and campaign.get("status") == "completed"
        and isinstance(counts, Mapping)
        and counts.get("completed") == len(expected_batches)
        and all(
            counts.get(name, 0) == 0
            for name in ("pending", "running", "interrupted", "failed")
        )
        and exact_completed_batches
        and exact_scientific_join
    )
    neutral_cells = bool(
        isinstance(plan_cells, list)
        and len(plan_cells) == 72
        and all(
            isinstance(cell, Mapping)
            and isinstance(cell.get("parameters"), Mapping)
            and cell["parameters"].get("recomb72.neutral_fixture") is True
            and cell["parameters"].get("recomb72.mutation_enabled") is False
            and cell["parameters"].get("recomb72.ordinary_births_enabled") is False
            for cell in plan_cells
        )
    )
    technical_evidence = (
        technical.get("evidence") if isinstance(technical, Mapping) else None
    )
    technical_checks = (
        technical_evidence.get("checks")
        if isinstance(technical_evidence, Mapping)
        else None
    )
    required_gate_checks = {
        "identity_exact",
        "telemetry_complete_g0_g5",
        "founder_split_exact",
        "population_64",
        "turnover_32",
        "mutation_zero",
        "ordinary_births_zero",
        "mode_reported",
        "registry_exact",
        "native_owner_no_fallback",
        "g0_g5_checkpoints",
        "fresh_owner_restore_exact",
        "production_sources_route_bound",
        "current_production_parity_evidence",
    }
    technical_cells_clean = bool(
        isinstance(technical_checks, Mapping)
        and len(technical_checks) == 4
        and all(
            isinstance(checks, Mapping)
            and required_gate_checks <= set(checks)
            and all(checks.get(name) is True for name in required_gate_checks)
            for checks in technical_checks.values()
        )
    )
    gates["trait_neutrality"] = bool(
        _bound(technical, plan)
        and technical.get("passed") is True
        and isinstance(technical_evidence, Mapping)
        and technical_evidence.get("passed") is True
        and technical_cells_clean
        and neutral_cells
    )
    gates["no_fallback"] = bool(rows) and all(
        isinstance(row.get("integrity"), Mapping)
        and row["integrity"].get("no_fallback") is True
        and row["integrity"].get("native_gpu_owner") is True
        for row in rows
    )
    comparison_contract = (
        plan.get("comparisons") if isinstance(plan.get("comparisons"), Mapping) else {}
    )
    expected_branches = {
        branch["branch_id"]: branch
        for branch in comparison_contract.get("branches", [])
        if isinstance(branch, Mapping) and isinstance(branch.get("branch_id"), str)
    }
    comparison_summaries = (
        comparisons.get("branches") if isinstance(comparisons, Mapping) else None
    )
    observed_branches = {
        row["branch_id"]: row
        for row in comparison_summaries
        if isinstance(comparison_summaries, list)
        and isinstance(row, Mapping)
        and isinstance(row.get("branch_id"), str)
    } if isinstance(comparison_summaries, list) else {}
    checkpoint_branches = [
        branch
        for branch in expected_branches.values()
        if branch.get("kind") == "checkpoint_continuation"
    ]
    checkpoint_manifests_clean = True
    for branch in checkpoint_branches:
        manifest = _read_json(
            root / "comparisons" / str(branch["branch_id"]) / "manifest.json"
        )
        comparison = (
            manifest.get("comparison") if isinstance(manifest, Mapping) else None
        )
        compared_surfaces = (
            comparison.get("compared_surfaces")
            if isinstance(comparison, Mapping)
            else None
        )
        if not (
            _bound(manifest, plan)
            and manifest.get("status") == "completed"
            and manifest.get("kind") == "checkpoint_continuation"
            and manifest.get("source_cell_id") == branch.get("source_cell_id")
            and manifest.get("split_generation") == branch.get("split_generation")
            and isinstance(comparison, Mapping)
            and comparison.get("match") is True
            and comparison.get("branch_role") == "checkpoint_continuation"
            and {
                "terminal_checkpoint_digest",
                "metrics_rows_sha256",
                "transition_sentinels_sha256",
            }
            <= set(compared_surfaces or [])
        ):
            checkpoint_manifests_clean = False
            break
    checkpoint_coverage = {
        (
            expected_identity.get(str(branch.get("source_cell_id")), ("", "", -1))[1],
            expected_identity.get(str(branch.get("source_cell_id")), ("", "", -1))[0],
            expected_identity.get(str(branch.get("source_cell_id")), ("", "", -1))[2],
            branch.get("split_generation"),
        )
        for branch in checkpoint_branches
    }
    gates["checkpoint_parity"] = bool(
        _bound(comparisons, plan)
        and comparisons.get("status") == "completed"
        and set(observed_branches) == set(expected_branches)
        and all(row.get("status") == "completed" for row in observed_branches.values())
        and len(checkpoint_branches) == 8
        and {row[0] for row in checkpoint_coverage}
        == {
            "current_authoritative",
            "parent_copy_per_locus",
            "bounded_interpolation",
            "whole_package_single_parent",
        }
        and {row[1] for row in checkpoint_coverage}
        == {
            "single_extreme_scalar",
            "resource_specialization_extremes",
            "correlated_multi_trait_packages",
        }
        and len({row[2] for row in checkpoint_coverage}) >= 3
        and {row[3] for row in checkpoint_coverage} == {10, 20}
        and checkpoint_manifests_clean
    )
    reporting_branches = [
        branch
        for branch in expected_branches.values()
        if branch.get("kind") == "reporting_parity"
    ]
    reporting_manifest = (
        _read_json(
            root
            / "comparisons"
            / str(reporting_branches[0]["branch_id"])
            / "manifest.json"
        )
        if len(reporting_branches) == 1
        else None
    )
    reporting_comparison = (
        reporting_manifest.get("comparison")
        if isinstance(reporting_manifest, Mapping)
        else None
    )
    gates["reporting_parity"] = bool(
        _bound(reporting, plan)
        and reporting.get("qualification_evidence_status")
        == "reporting_parity_proven"
        and len(reporting_branches) == 1
        and _bound(reporting_manifest, plan)
        and reporting_manifest.get("status") == "completed"
        and reporting_manifest.get("source_cell_id")
        == reporting_branches[0].get("source_cell_id")
        and isinstance(reporting_comparison, Mapping)
        and reporting_comparison.get("match") is True
        and reporting_comparison.get("evidence_status")
        == "reporting_parity_proven"
        and "terminal_checkpoint_digest"
        in reporting_comparison.get("compared_surfaces", [])
        and reporting.get("comparison_receipt_sha256")
        == canonical_sha256(reporting_comparison)
    )
    provenance_ids: set[str] = set()
    batch_ids: set[str] = set()
    for manifest_path in sorted((root / "batches").glob("*/manifest.json")):
        manifest = _read_json(manifest_path)
        result = manifest.get("result") if isinstance(manifest, Mapping) else None
        cell_results = (
            result.get("cell_results") if isinstance(result, Mapping) else None
        )
        if not isinstance(cell_results, Mapping):
            continue
        manifest_cell_ids = (
            manifest.get("cell_ids") if isinstance(manifest, Mapping) else None
        )
        if (
            manifest.get("status") != "completed"
            or not isinstance(manifest_cell_ids, list)
            or set(cell_results) != set(manifest_cell_ids)
            or batch_ids & set(manifest_cell_ids)
        ):
            continue
        batch_ids.update(manifest_cell_ids)
        for cell_id, cell_result in cell_results.items():
            provenance = (
                cell_result.get("provenance")
                if isinstance(cell_result, Mapping)
                else None
            )
            if (
                isinstance(cell_id, str)
                and isinstance(provenance, Mapping)
                and provenance.get("cell_id") == cell_id
                and isinstance(provenance.get("provider_capability_sha256"), str)
                and isinstance(provenance.get("verified_prebuilt_artifact"), Mapping)
                and provenance["verified_prebuilt_artifact"].get("verified") is True
                and isinstance(provenance.get("mode_contract_sha256"), str)
                and isinstance(provenance.get("checkpoint_contract_sha256"), str)
                and isinstance(provenance.get("terminal_checkpoint_identity"), Mapping)
            ):
                provenance_ids.add(cell_id)
    receipt_rows = cells.get("rows") if isinstance(cells, Mapping) else None
    receipt_ids = {
        row.get("cell_id")
        for row in receipt_rows
        if isinstance(receipt_rows, list)
        and isinstance(row, Mapping)
        and isinstance(row.get("cell_id"), str)
    } if isinstance(receipt_rows, list) else set()
    gates["cell_provenance_complete"] = bool(
        _bound(cells, plan)
        and cells.get("completed_cell_count") == 72
        and len(receipt_rows or []) == 72
        and receipt_ids == expected_ids
        and provenance_ids == expected_ids
        and batch_ids == expected_ids
        and exact_scientific_join
    )
    chronicle_ids = set(chronicle.get("world_ids", [])) if isinstance(
        chronicle, Mapping
    ) and isinstance(chronicle.get("world_ids"), list) else set()
    archive_path = (
        Path(str(chronicle.get("archive_path") or "")).resolve()
        if isinstance(chronicle, Mapping)
        else Path()
    )
    gates["chronicle_authority"] = bool(
        _bound(chronicle, plan)
        and chronicle.get("verified") is True
        and chronicle_ids == expected_ids
        and _file_sha256(archive_path) == chronicle.get("archive_sha256")
        and exact_scientific_join
    )
    artifacts = plan.get("artifacts")
    qualification = (
        artifacts.get("qualification_receipt")
        if isinstance(artifacts, Mapping)
        else None
    )
    if isinstance(qualification, Mapping):
        qualification_path = Path(str(qualification.get("path") or "")).resolve()
        qualification_payload = _read_json(qualification_path)
        qualification_bound = bool(
            _file_sha256(qualification_path) == qualification.get("sha256")
            and qualification_payload
            and qualification_payload.get("current_mode_matches_production") is True
        )
        gates["current_mode_matches_production"] = bool(
            qualification_bound
            or (
                gates["trait_neutrality"]
                and any(
                    isinstance(checks, Mapping)
                    and checks.get("current_production_parity_evidence") is True
                    for checks in (technical_checks or {}).values()
                )
            )
        )
    return gates


def _analyze_recomb72(
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    run_root: Path | None,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    from engine.evolution.recomb72_decision import (
        Recomb72DecisionError,
        analyze_recomb72,
    )

    analysis = plan.get("analysis")
    extension = analysis.get("extension") if isinstance(analysis, Mapping) else None
    if not isinstance(extension, Mapping):
        raise CampaignAnalysisExtensionError(
            "RECOMB-72 analysis extension contract is absent"
        )
    contract = extension.get("contract")
    contract_sha256 = extension.get("contract_sha256")
    if not isinstance(contract, Mapping) or not isinstance(contract_sha256, str):
        raise CampaignAnalysisExtensionError(
            "RECOMB-72 analysis extension contract is malformed"
        )
    if canonical_sha256(contract) != contract_sha256:
        raise CampaignAnalysisExtensionError(
            "RECOMB-72 analysis extension contract hash mismatch"
        )
    gates = _recomb72_external_gates(plan, rows, run_root, context)
    try:
        result = analyze_recomb72(
            rows,
            external_gates=gates,
            analysis_contract=contract,
            contract_sha256=contract_sha256,
        )
    except Recomb72DecisionError as exc:
        raise CampaignAnalysisExtensionError(str(exc)) from exc
    return {
        **result,
        "extension_id": "recomb72_scientific_v1",
        "external_gates_source": (
            "explicit_analysis_context"
            if isinstance(context.get("external_gates"), Mapping)
            else "bound_run_artifacts"
        ),
    }


def _render_recomb72(value: Mapping[str, Any]) -> str:
    scientific = value.get("scientific_analysis")
    if not isinstance(scientific, Mapping):
        raise CampaignAnalysisExtensionError(
            "RECOMB-72 scientific analysis result is absent"
        )
    paired = scientific.get("paired_milestones")
    summaries = (
        paired.get("summaries")
        if isinstance(paired, Mapping)
        and isinstance(paired.get("summaries"), list)
        else []
    )
    pattern3 = scientific.get("pattern3_locus_paired_milestones")
    pattern3_summaries = (
        pattern3.get("summaries")
        if isinstance(pattern3, Mapping)
        and isinstance(pattern3.get("summaries"), list)
        else []
    )
    correlations = scientific.get("pattern3_correlation_matrices")
    correlations = correlations if isinstance(correlations, list) else []
    homogenization = scientific.get("paired_time_to_homogenization")
    homogenization = homogenization if isinstance(homogenization, list) else []
    lineage = scientific.get("lineage_genetic_decoupling")
    lineage = lineage if isinstance(lineage, list) else []
    integrity = scientific.get("integrity")
    integrity = integrity if isinstance(integrity, Mapping) else {}
    external_gates = integrity.get("external_gates")
    external_gates = external_gates if isinstance(external_gates, Mapping) else {}

    def number(value: Any) -> str:
        return f"{float(value):.6g}"

    def paired_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
        table = [
            "| Pattern | Metric | G | Alternative | Mean delta | Median | Min | Max | Per-seed current→alternative (delta) |",
            "|---|---|---:|---|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            pairs = "; ".join(
                f"S{pair['seed']} {number(pair['current'])}→"
                f"{number(pair['alternative'])} "
                f"({number(pair['delta_alternative_minus_current'])})"
                for pair in row.get("pairs", [])
            )
            table.append(
                f"| {row.get('pattern')} | {row.get('metric')} | "
                f"{row.get('generation')} | {row.get('alternative_mode')} | "
                f"{number(row.get('delta_mean'))} | "
                f"{number(row.get('delta_median'))} | "
                f"{number(row.get('delta_minimum'))} | "
                f"{number(row.get('delta_maximum'))} | {pairs} |"
            )
        return table

    lines = [
        "# Campaign Analysis",
        "",
        f"Campaign: `{value.get('campaign_id', '')}`",
        "",
        f"Decision: **{scientific.get('decision')}**",
        "",
        f"Decision label: `{scientific.get('decision_label')}`",
        "",
        f"Interpretable: `{str(scientific.get('interpretable')).lower()}`",
        "",
        "## Scientific integrity",
        "",
        f"- Generation rows: `{integrity.get('observed_generation_rows', 0)}` / `2232`.",
        f"- Cells: `{integrity.get('observed_cells', 0)}` / `72`.",
        f"- Contract SHA-256: `{scientific.get('analysis_contract_sha256', '')}`.",
        "",
        "### External authority gates",
        "",
        "| Gate | Passed |",
        "|---|---|",
        *[
            f"| {name} | `{str(passed is True).lower()}` |"
            for name, passed in sorted(external_gates.items())
        ],
        "",
        "### Frozen definitions",
        "",
        "- Homogenization: variance below 10% of G0 for three consecutive generations.",
        "- Extreme retention: within 5% of the legal locus range from either frozen endpoint.",
        "- Package retention: exact canonical-float equality to all loci of Package A or Package B; endpoint mixtures are recombinant and any non-endpoint locus is neither.",
        "- No post-result numeric materiality cutoff is used.",
        "",
        "## Primary paired milestones",
        "",
        "G5/G10/G20/G30 and six seeds are shown explicitly; each row contains all paired seeds, not only a pooled mean.",
        "",
        *paired_table(summaries),
        "",
        "## Pattern-3 per-locus variance",
        "",
        *paired_table(pattern3_summaries),
        "",
        "### Pattern-3 trait-correlation matrices",
        "",
        "| Mode | Seed | G | Correlation matrix |",
        "|---|---:|---:|---|",
        *[
            f"| {row.get('mode')} | {row.get('seed')} | {row.get('generation')} | "
            f"`{json.dumps(row.get('correlation_matrix'), sort_keys=True, separators=(',', ':'))}` |"
            for row in correlations
        ],
        "",
        "## Paired time to homogenization",
        "",
        "| Pattern | Locus | Alternative | All six current-faster | Per-seed current→alternative first generation |",
        "|---|---|---|---|---|",
        *[
            (
                f"| {row.get('pattern')} | {row.get('locus')} | "
                f"{row.get('alternative_mode')} | "
                f"`{str(row.get('current_homogenizes_faster_all_seeds') is True).lower()}` | "
                + "; ".join(
                    f"S{pair.get('seed')} "
                    f"{pair.get('current_first_generation')}→"
                    f"{pair.get('alternative_first_generation')}"
                    for pair in row.get("pairs", [])
                )
                + " |"
            )
            for row in homogenization
        ],
        "",
        "## Terminal lineage/genetic decoupling",
        "",
        "| Pattern | Mode | Seed | Effective lineages | Top-1 ancestry | Trait distance | Package distance | Genealogy persists | Genetically decoupled |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
        *[
            (
                f"| {row.get('pattern')} | {row.get('mode')} | {row.get('seed')} | "
                f"{number(row.get('effective_alive_lineages'))} | "
                f"{number(row.get('top1_founder_ancestry_share'))} | "
                f"{number(row.get('trait_distance_between_branches'))} | "
                f"{number(row.get('package_distance_between_branches'))} | "
                f"`{str(row.get('genealogy_persists') is True).lower()}` | "
                f"`{str(row.get('genealogy_persists_while_genetic_distinction_disappears') is True).lower()}` |"
            )
            for row in lineage
        ],
        "",
        "## Decision predicate evidence",
        "",
        "```json",
        json.dumps(
            scientific.get("scientific_decision_evidence", {}),
            sort_keys=True,
            indent=2,
        ),
        "```",
        "",
        "## Fail-closed reasons",
        "",
    ]
    reasons = scientific.get("fail_closed_reasons")
    if isinstance(reasons, list) and reasons:
        lines.extend(f"- `{reason}`" for reason in reasons)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Deferred bounds",
            "",
            "- This lifecycle report does not generalize to hereditary surfaces outside the frozen audited loci.",
            "- It does not authorize a production inheritance repair or any cognition, emergence, adaptation, speciation, or open-ended-evolution claim.",
            "- Evidence packaging, branch/commit disposition, test inventory, and any repair remain final-closeout responsibilities.",
        ]
    )
    return "\n".join(lines) + "\n"


def _analyze_inheritance_cross_path(
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    _run_root: Path | None,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt the generic contract-driven analyzer to the campaign lifecycle."""

    analysis = plan.get("analysis")
    extension = analysis.get("extension") if isinstance(analysis, Mapping) else None
    if not isinstance(extension, Mapping):
        raise CampaignAnalysisExtensionError(
            "inheritance cross-path analysis contract is absent"
        )
    contract = extension.get("contract")
    contract_sha256 = extension.get("contract_sha256")
    if (
        extension.get("id") != "inheritance_cross_path_scientific_v1"
        or not isinstance(contract, Mapping)
        or not isinstance(contract_sha256, str)
        or canonical_sha256(contract) != contract_sha256
    ):
        raise CampaignAnalysisExtensionError(
            "inheritance cross-path analysis contract binding is invalid"
        )
    external_gates = context.get("external_gates")
    if not isinstance(external_gates, Mapping):
        external_gates = {}
    try:
        from engine.evolution.inheritance_cross_path_analysis import (
            InheritanceCrossPathAnalysisError,
            analyze_inheritance_cross_path,
        )

        result = analyze_inheritance_cross_path(
            rows,
            analysis_contract=contract,
            external_gates=external_gates,
        )
    except InheritanceCrossPathAnalysisError as exc:
        raise CampaignAnalysisExtensionError(str(exc)) from exc
    if result.get("contract_sha256") != contract_sha256:
        raise CampaignAnalysisExtensionError(
            "inheritance cross-path analyzer returned another contract identity"
        )
    return result


def _render_inheritance_cross_path(value: Mapping[str, Any]) -> str:
    """Render a stable bounded report without knowing campaign dimensions."""

    scientific = value.get("scientific_analysis")
    if not isinstance(scientific, Mapping):
        raise CampaignAnalysisExtensionError(
            "inheritance cross-path scientific analysis is absent"
        )
    decision = str(scientific.get("decision") or "")
    label = str(scientific.get("decision_label") or "")
    confirmations = scientific.get("path_confirmations")
    consistency = scientific.get("cross_path_consistency")
    locking = scientific.get("locking")
    reasons = scientific.get("fail_closed_reasons")
    lines = [
        "# Win/Loss",
        "",
        f"Decision: **{decision}**",
        "",
        f"Meaning: {label}",
        "",
        f"- Analysis contract SHA-256: `{scientific.get('contract_sha256', '')}`",
        f"- Expected cells: `{scientific.get('expected_cells', 0)}`",
        (
            "- Expected generation rows: "
            f"`{scientific.get('expected_generation_rows', 0)}`"
        ),
        f"- Interpretable: `{str(scientific.get('interpretable') is True).lower()}`",
        "",
        "## Reproduction-path results",
        "",
        "```json",
        json.dumps(
            confirmations if isinstance(confirmations, Mapping) else {},
            sort_keys=True,
            indent=2,
        ),
        "```",
        "",
        "## Cross-path consistency",
        "",
        "```json",
        json.dumps(
            consistency if isinstance(consistency, Mapping) else {},
            sort_keys=True,
            indent=2,
        ),
        "```",
        "",
        "## Locking check",
        "",
        "```json",
        json.dumps(
            locking if isinstance(locking, Mapping) else {},
            sort_keys=True,
            indent=2,
        ),
        "```",
        "",
        "## Fail-closed reasons",
        "",
    ]
    if isinstance(reasons, list) and reasons:
        lines.extend(f"- `{reason}`" for reason in reasons)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            (
                "This report is limited to the inheritance modes, founder patterns, "
                "reproduction paths, paired identities, loci, milestones, gates, and "
                "decision predicates frozen in the bound analysis contract."
            ),
            (
                "It makes no intelligence, cognition, emergence, speciation, "
                "adaptation, or open-ended-evolution claim."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def default_analysis_extension_registry() -> CampaignAnalysisExtensionRegistryV1:
    registry = CampaignAnalysisExtensionRegistryV1()
    registry.register(
        CampaignAnalysisExtensionV1(
            extension_id="inheritance_cross_path_scientific_v1",
            analyze=_analyze_inheritance_cross_path,
            render_markdown=_render_inheritance_cross_path,
        )
    )
    registry.register(
        CampaignAnalysisExtensionV1(
            extension_id="recomb72_scientific_v1",
            analyze=_analyze_recomb72,
            render_markdown=_render_recomb72,
        )
    )
    return registry


__all__ = [
    "CampaignAnalysisExtensionError",
    "CampaignAnalysisExtensionRegistryV1",
    "CampaignAnalysisExtensionV1",
    "default_analysis_extension_registry",
]
