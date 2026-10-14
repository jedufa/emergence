#!/usr/bin/env python3
"""Freeze GV2 B48-R2 seed-1708 localization preregistration artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "engine"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from engine.campaigns.canonical import canonical_sha256, pretty_json  # noqa: E402
from engine.campaigns.patch_runner_factory import (  # noqa: E402
    derive_patch_master_seed_v1,
    derive_production_turnover_seed_v1,
)


SCHEMA_VERSION = "gv2_b48_r2_seed1708_localization_prereg_v1"
R1_CAMPAIGN_ID = "gv2-b48-r1-neutral-package-identity-b24g30"
R2_PROGRAMME_NAME = (
    "GV2-B48-R2 -- SEED-1708 PACKAGE-IDENTITY LOCALIZATION AND FALSIFICATION"
)
R2_SCIENCE_CAMPAIGN_ID = "gv2-b48-r2-seed1708-wholeseed-boundary-b24g30"
R2_CONTROL_CAMPAIGN_ID = "gv2-b48-r2-seed1708-exact-repeat-control-b2g30"
PROMPT_SHA256 = "491ed83b9cc15c55a4841135233ef7f8d1fb10f6c36a0d5e23deef0ae96cbb2b"
BRIDGE_EXECUTION_PROMPT_SHA256 = (
    "86681b6e62e3abae7b7d621476ce51a60c9c3debde0d06307b8929ef427d4e95"
)
BRIDGE_EXECUTION_PROMPT_ZIP_SHA256 = (
    "1556008332bb169d59348b1e1b3ec8e0a255c635b947fbced18b5118d2dc835d"
)
PRIMARY_IDENTITY_TYPE = "preregistered_package_identity"
SECONDARY_IDENTITY_TYPE = "allele_genotype_state_identity"
R1_SEEDS = (707, 808, 909, 1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809)
R1_POSITIVE_SEED = 1708
REQUIRED_GENERATIONS = (0, 5, 10, 15, 20, 25, 30)


class B48R2PreregistrationError(RuntimeError):
    """R2 preregistration cannot be constructed from current evidence."""


def _fail(message: str) -> None:
    raise B48R2PreregistrationError(message)


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                value = json.load(handle)
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise B48R2PreregistrationError(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{name} must be a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty_json(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_heldout_seeds(
    *,
    prompt_sha256: str = PROMPT_SHA256,
    excluded: Sequence[int] = R1_SEEDS,
    count: int = 12,
) -> list[int]:
    """Select deterministic whole-seed held-outs without touching R1 seeds."""

    excluded_set = {int(seed) for seed in excluded}
    seeds: list[int] = []
    counter = 0
    while len(seeds) < count:
        material = f"{SCHEMA_VERSION}|{prompt_sha256}|heldout|{counter}".encode("ascii")
        candidate = 2000 + (int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 7000)
        counter += 1
        if candidate in excluded_set or candidate in seeds:
            continue
        seeds.append(candidate)
    return seeds


def _include_rows(seeds: Sequence[int]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for seed in seeds:
        for rate, step in ((0, 0), (16, 16)):
            rows.append(
                {
                    "ecology_seed": int(seed),
                    "task_seed": 0,
                    "founder_profile": "neutral",
                    "mutation_rate_q10": rate,
                    "mutation_step_q10": step,
                }
            )
    return rows


def _matrix(seeds: Sequence[int]) -> dict[str, Any]:
    return {
        "dimensions": [
            {"id": "ecology_seed", "values": [int(seed) for seed in seeds]},
            {"id": "task_seed", "values": [0]},
            {"id": "founder_profile", "values": ["neutral"]},
            {"id": "mutation_rate_q10", "values": [0, 16]},
            {"id": "mutation_step_q10", "values": [0, 16]},
        ],
        "parameter_bindings": {
            "founder_profile": "genome_upgrade_v2.founder_profile",
            "mutation_rate_q10": "genome_upgrade_v2.mutation_rate_q10",
            "mutation_step_q10": "genome_upgrade_v2.mutation_step_q10",
        },
        "seed_bindings": {"ecology_seed": "ecology", "task_seed": "task"},
        "include": _include_rows(seeds),
    }


def classify_positive_family(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives = [
        dict(row)
        for row in rows
        if row.get("identity_type") == PRIMARY_IDENTITY_TYPE
        and row.get("positive_mutation_persistence") is True
    ]
    families: dict[str, list[dict[str, Any]]] = {}
    for row in positives:
        carriers = row.get("terminal_descendant_carriers")
        carrier = carriers[0] if isinstance(carriers, list) and carriers else {}
        exact_values = carrier.get("exact_values") if isinstance(carrier, Mapping) else {}
        key = canonical_sha256(
            {
                "cell_id": row.get("cell_id"),
                "state_hash": row.get("state_hash"),
                "terminal_carrier_stable_id": carrier.get("carrier_stable_id")
                if isinstance(carrier, Mapping)
                else None,
                "terminal_carrier_slot": carrier.get("carrier_slot")
                if isinstance(carrier, Mapping)
                else None,
                "terminal_exact_hereditary_digest": exact_values.get(
                    "exact_hereditary_digest"
                )
                if isinstance(exact_values, Mapping)
                else None,
            }
        )
        families.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, family_rows in sorted(families.items()):
        first = family_rows[0]
        carrier = first.get("terminal_descendant_carriers", [{}])[0]
        exact_values = carrier.get("exact_values") if isinstance(carrier, Mapping) else {}
        summaries.append(
            {
                "family_id": key,
                "row_count": len(family_rows),
                "cell_id": first.get("cell_id"),
                "paired_off_cell_id": first.get("paired_off_cell_id"),
                "seed": int(first.get("seed")),
                "founder_profile": first.get("founder_profile"),
                "mutation_rate_q10": int(first.get("mutation_rate_q10")),
                "mutation_step_q10": int(first.get("mutation_step_q10")),
                "state_hash": first.get("state_hash"),
                "terminal_carrier_stable_id": carrier.get("carrier_stable_id")
                if isinstance(carrier, Mapping)
                else None,
                "terminal_carrier_slot": carrier.get("carrier_slot")
                if isinstance(carrier, Mapping)
                else None,
                "terminal_exact_hereditary_digest": exact_values.get(
                    "exact_hereditary_digest"
                )
                if isinstance(exact_values, Mapping)
                else None,
                "event_rows": [
                    {
                        "event_child_stable_id": row.get("event_child_stable_id"),
                        "event_generation": row.get("event_generation"),
                        "event_transaction_id": row.get("event_transaction_id"),
                    }
                    for row in sorted(
                        family_rows,
                        key=lambda item: (
                            int(item.get("event_generation", -1)),
                            int(item.get("event_child_stable_id", -1)),
                        ),
                    )
                ],
            }
        )
    return {
        "positive_row_count": len(positives),
        "positive_event_family_count": len(summaries),
        "families": summaries,
        "interpretation": (
            "one_package_identity_event_family_counted_across_four_event_rows"
            if len(summaries) == 1 and len(positives) == 4
            else "positive_rows_do_not_collapse_to_single_known_r1_family"
        ),
    }


def seed_authority_audit(*, sample_cell_id: str = "cell-72d2e7a8ed9bf6e53a08") -> dict[str, Any]:
    owners = {"ecology": R1_POSITIVE_SEED, "task": 0}
    return {
        "schema_version": "gv2_b48_r2_seed_authority_audit_v1",
        "campaign_seed_owners": sorted(owners),
        "campaign_seed_bindings": {"ecology_seed": "ecology", "task_seed": "task"},
        "compiler_allowed_seed_owners": ["ecology", "founder", "policy", "task"],
        "r1_exposed_named_seed_owners": owners,
        "derived_engine_master_seed_for_r1_positive_seed": derive_patch_master_seed_v1(owners),
        "derived_turnover_seed_for_r1_positive_cell": derive_production_turnover_seed_v1(
            cell_id=sample_cell_id,
            seeds=owners,
        ),
        "spatial_ecology_seed_owner": "ecology",
        "spatial_ecology_overlay_status": "disabled_in_current_qualified_patch_route",
        "wdl_base_seed_owner": "task",
        "component_swap_available_without_route_or_architecture_change": False,
        "component_swap_blocker": (
            "mutation, reproduction, founder, policy, checkpoint, and turnover RNG "
            "streams are not exposed as separately swappable campaign seed owners "
            "in the R1 production campaign contract; ecology and task are hashed "
            "into shared engine/runtime seed roots."
        ),
        "selected_r2_design_branch": "fallback_whole_seed_heldout_boundary_probe",
    }


def reducer_audit_contract() -> dict[str, Any]:
    return {
        "schema_version": "gv2_b48_r2_reducer_audit_contract_v1",
        "primary_scientific_count": "positive_event_family_count",
        "raw_positive_row_count_is_not_independent_origin_count": True,
        "event_family_key": {
            "schema_version": "gv2_b48_r2_event_family_key_v1",
            "fields": [
                "logical_cell_id",
                "identity_type",
                "package_state_hash",
                "terminal_exact_hereditary_digest",
                "terminal_carrier_stable_id",
                "terminal_carrier_slot",
            ],
            "excluded_from_family_key": [
                "event_transaction_id",
                "event_child_stable_id",
                "event_generation",
            ],
            "reason": (
                "R1 showed multiple event rows with the same state hash, terminal "
                "carrier, and hereditary digest; including row-local event IDs "
                "would incorrectly split one terminal package-identity family "
                "into multiple independent origins."
            ),
        },
        "mutation_origin_key": {
            "status": "claim_only_if_full_ordinary_birth_chain_reconstructed",
            "required_fields": [
                "first_novel_package_state_appearance",
                "ordinary_birth_receipt",
                "parent_id",
                "mate_parent_id",
                "child_id",
                "event_generation",
                "event_tick_or_transaction_index",
                "descendant_transfer_chain",
                "terminal_carrier_id",
                "terminal_exact_hereditary_digest",
            ],
            "fallback_claim_if_missing": (
                "terminal package-identity persistence under ordinary-birth "
                "authority; no reconstructed mutation-origin lineage claim"
            ),
        },
        "required_reducer_fields": [
            "classified_event_rows",
            "unique_event_identities",
            "unique_package_state_hashes",
            "unique_terminal_carriers",
            "unique_event_families",
            "unique_mutation_origins_where_reconstructable",
            "positive_event_rows",
            "positive_event_families",
            "positive_seeds",
            "positive_logical_cells",
            "total_low_mutation_classified_rows",
            "total_low_mutation_event_families",
            "mutation_off_false_positive_count",
            "positive_family_rate_per_classified_family",
            "positive_seed_fraction",
            "terminal_carrier_counts",
            "allele_genotype_positive_count",
            "per_seed_rates",
            "exact_repeat_control_separated_from_science",
        ],
        "independent_audit_requirements": [
            "all_positives",
            "at_least_24_stratified_negatives",
            "every_design_arm",
            "every_physical_partition",
            "paired_mutation_off_controls",
            "all_predicate_components",
            "zero_unresolved_rows",
            "zero_production_mismatches",
            "exact_predicate_equivalence",
            "exact_authority_binding",
        ],
    }


def retention_contract() -> dict[str, Any]:
    return {
        "schema_version": "gv2_b48_r2_retention_contract_v1",
        "audit_hold_status_required": "AUDIT_HOLD_ACTIVE",
        "class_a_chain_surfaces": [
            "g0_native_checkpoint_projections",
            "g5_g10_g15_g20_g25_g30_native_checkpoint_projections",
            "paired_off_control_projections",
            "ordinary_birth_receipts",
            "event_transaction_receipts",
            "first_novel_package_state_appearance_evidence",
            "parent_child_identity_evidence",
            "descendant_transfer_chain_evidence",
            "terminal_carrier_identity",
            "terminal_exact_hereditary_digest_continuity",
            "package_state_continuity",
            "route_package_artifact_source_authority",
            "reducer_inputs",
            "independent_audit_inputs",
        ],
        "release_receipt_required_before_prune": True,
        "release_receipt_requirements": [
            "execution_complete",
            "reduction_complete",
            "sample_manifest_frozen",
            "independent_reconstruction_complete",
            "predicate_equivalence_established",
            "terminal_decision_written",
            "compact_archive_verified",
        ],
        "claim_downgrade_if_chain_unavailable": (
            "terminal package-identity persistence under ordinary-birth authority"
        ),
    }


def resource_measurement_contract() -> dict[str, Any]:
    return {
        "schema_version": "gv2_b48_r2_resource_measurement_contract_v1",
        "gpu_exclusivity_required": True,
        "measurement_order": [
            "b2_exact_repeat_control_batch_0000",
            "b12_science_batch_0000",
            "b12_science_batch_0001",
        ],
        "dual_b12_measurement_default": True,
        "single_b12_measurement_allowed_only_with_canonical_signature_equality": True,
        "resource_signature_fields": [
            "logical_cell_count",
            "event_capacities",
            "mutation_tape_widths",
            "policy_state_widths",
            "checkpoint_schedule",
            "reporting_configuration",
            "founder_profile",
            "cell_specific_route_bindings",
            "per_cell_storage_capacities",
            "expected_receipt_event_limits",
            "physical_mapping_order",
            "late_generation_state_packing_sizes",
        ],
    }


def _load_primary_rows(r1_root: Path) -> list[dict[str, Any]]:
    payload = _read_json(
        r1_root / "06_reduction_r1" / "gv2_b48_r1_primary_classified_rows_v1.json.gz",
        name="R1 primary classified rows",
    )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        _fail("R1 primary classified rows payload lacks rows")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _cell_manifest(run_roots: Sequence[Path], cell_id: str) -> dict[str, Any] | None:
    for run_root in run_roots:
        plan_path = run_root / "execution_plan.json"
        if not plan_path.exists():
            continue
        plan = _read_json(plan_path, name="R1 execution plan")
        for cell in plan.get("cells", []):
            if isinstance(cell, Mapping) and cell.get("cell_id") == cell_id:
                return {
                    "run_root": str(run_root),
                    "execution_plan_path": str(plan_path),
                    "execution_plan_file_sha256": _sha256(plan_path),
                    "plan_sha256": plan.get("plan_sha256"),
                    "cell": dict(cell),
                }
    return None


def _projection_sidecars(run_roots: Sequence[Path], cell_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_root in run_roots:
        pattern = (
            f"batches/*/checkpoint_bundles_v2/{cell_id}/bundle_gen*/"
            "native_runtime_checkpoint_projection.json*"
        )
        for path in sorted(run_root.glob(pattern)):
            stem = path.parent.name.removeprefix("bundle_gen")
            generation = int(stem) if stem.isdigit() else None
            rows.append(
                {
                    "generation": generation,
                    "path": str(path),
                    "file_sha256": _sha256(path),
                    "compressed": path.name.endswith(".gz"),
                }
            )
    return sorted(rows, key=lambda row: (-1 if row["generation"] is None else row["generation"], row["path"]))


def positive_family_forensics(
    *,
    r1_root: Path,
    run_roots: Sequence[Path],
) -> dict[str, Any]:
    rows = _load_primary_rows(r1_root)
    family = classify_positive_family(rows)
    positive_cell_ids = sorted(
        {
            str(row["cell_id"])
            for row in rows
            if row.get("positive_mutation_persistence") is True
        }
    )
    paired_off_cell_ids = sorted(
        {
            str(row["paired_off_cell_id"])
            for row in rows
            if row.get("positive_mutation_persistence") is True
        }
    )
    target_cell_ids = positive_cell_ids + paired_off_cell_ids
    sidecars = {cell_id: _projection_sidecars(run_roots, cell_id) for cell_id in target_cell_ids}
    manifests = {cell_id: _cell_manifest(run_roots, cell_id) for cell_id in target_cell_ids}
    missing = {
        cell_id: sorted(set(REQUIRED_GENERATIONS) - {int(row["generation"]) for row in rows_ if row["generation"] is not None})
        for cell_id, rows_ in sidecars.items()
    }
    return {
        "schema_version": "gv2_b48_r2_seed1708_positive_family_forensics_v1",
        "r1_primary_rows_path": str(
            r1_root / "06_reduction_r1" / "gv2_b48_r1_primary_classified_rows_v1.json.gz"
        ),
        "r1_primary_rows_file_sha256": _sha256(
            r1_root / "06_reduction_r1" / "gv2_b48_r1_primary_classified_rows_v1.json.gz"
        ),
        "positive_family": family,
        "positive_cell_ids": positive_cell_ids,
        "paired_off_cell_ids": paired_off_cell_ids,
        "cell_manifests": manifests,
        "projection_sidecars_by_cell": sidecars,
        "missing_required_projection_generations_by_cell": missing,
        "ordinary_birth_lineage_chain_limitations": {
            "gen1_f1_birth_records_observed_empty": True,
            "lineage_events_observed_empty": True,
            "limitation": (
                "R1 retained terminal projection and reducer/audit evidence sufficient "
                "for terminal join reconstruction, but not a full ordinary-birth "
                "parent-chain chronicle for the four seed-1708 event rows."
            ),
        },
    }


def _with_runtime_root(spec: dict[str, Any], *, runtime_root: Path) -> dict[str, Any]:
    output = dict(spec.get("output") or {})
    output["runtime_root"] = str(runtime_root)
    spec["output"] = output
    return spec


def build_science_spec(
    template: Mapping[str, Any],
    *,
    seeds: Sequence[int],
    runtime_root: Path | None = None,
) -> dict[str, Any]:
    spec = dict(template)
    spec["campaign_id"] = R2_SCIENCE_CAMPAIGN_ID
    spec["description"] = (
        "GV2-B48-R2 whole-seed held-out boundary probe; 12 deterministic new "
        "paired seeds, neutral founders, mutation off versus low fixed Q10, "
        "logical b24g30t80, physical B12x2."
    )
    spec["no_claims"] = (
        "Seed-1708 localization/falsification only; no B72, ecology, adaptation, "
        "cognition, intelligence, emergence, profile-general persistence, "
        "allele-genotype-general persistence, or open-ended-evolution claim."
    )
    spec["matrix"] = _matrix(seeds)
    spec["execution"] = dict(template["execution"])
    spec["execution"]["batch"] = {"mode": "bounded", "cells": None, "maximum": 12}
    spec["execution"]["technical_gate"] = {
        **dict(template["execution"]["technical_gate"]),
        "selectors": [_include_rows([seeds[0]])[1]],
    }
    if runtime_root is not None:
        spec = _with_runtime_root(spec, runtime_root=runtime_root / R2_SCIENCE_CAMPAIGN_ID)
    return spec


def build_control_spec(
    template: Mapping[str, Any],
    *,
    runtime_root: Path | None = None,
) -> dict[str, Any]:
    spec = dict(template)
    spec["campaign_id"] = R2_CONTROL_CAMPAIGN_ID
    spec["description"] = (
        "GV2-B48-R2 exact seed-1708 two-cell repeat control; determinism and "
        "pipeline control only, not independent scientific replication."
    )
    spec["no_claims"] = (
        "Exact seed-1708 repeat control only; not independent science and no B72 "
        "or generalization claim."
    )
    spec["matrix"] = _matrix([R1_POSITIVE_SEED])
    spec["execution"] = dict(template["execution"])
    spec["execution"]["batch"] = {"mode": "bounded", "cells": None, "maximum": 2}
    spec["execution"]["technical_gate"] = {
        **dict(template["execution"]["technical_gate"]),
        "selectors": [_include_rows([R1_POSITIVE_SEED])[1]],
    }
    if runtime_root is not None:
        spec = _with_runtime_root(spec, runtime_root=runtime_root / R2_CONTROL_CAMPAIGN_ID)
    return spec


def write_specs(
    *,
    template_spec_path: Path,
    spec_output_root: Path,
    heldout_seeds: Sequence[int],
    runtime_root: Path,
) -> dict[str, Any]:
    template = yaml.safe_load(template_spec_path.read_text(encoding="utf-8"))
    if not isinstance(template, Mapping):
        _fail(f"template spec must be a mapping: {template_spec_path}")
    specs = {
        "science": (
            spec_output_root / "gv2_b48_r2_seed1708_wholeseed_boundary_b24g30.yaml",
            build_science_spec(template, seeds=heldout_seeds, runtime_root=runtime_root),
        ),
        "control": (
            spec_output_root / "gv2_b48_r2_seed1708_exact_repeat_control_b2g30.yaml",
            build_control_spec(template, runtime_root=runtime_root),
        ),
    }
    written: dict[str, Any] = {}
    spec_output_root.mkdir(parents=True, exist_ok=True)
    for name, (path, payload) in specs.items():
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        written[name] = {
            "path": str(path),
            "file_sha256": _sha256(path),
            "campaign_id": payload["campaign_id"],
            "logical_cells": len(payload["matrix"]["include"]),
            "physical_batch_maximum": payload["execution"]["batch"]["maximum"],
        }
    return written


def preregistration(
    *,
    r1_root: Path,
    r1_spec_paths: Sequence[Path],
    run_roots: Sequence[Path],
    runtime_root: Path,
    spec_output_root: Path | None,
) -> dict[str, Any]:
    heldout_seeds = deterministic_heldout_seeds()
    seed_audit = seed_authority_audit()
    forensics = positive_family_forensics(r1_root=r1_root, run_roots=run_roots)
    spec_outputs = None
    if spec_output_root is not None:
        if not r1_spec_paths:
            _fail("spec output requires at least one R1 template spec")
        spec_outputs = write_specs(
            template_spec_path=r1_spec_paths[0],
            spec_output_root=spec_output_root,
            heldout_seeds=heldout_seeds,
            runtime_root=runtime_root,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "programme": R2_PROGRAMME_NAME,
        "prompt_sha256": PROMPT_SHA256,
        "bridge_execution_prompt_sha256": BRIDGE_EXECUTION_PROMPT_SHA256,
        "bridge_execution_prompt_zip_sha256": BRIDGE_EXECUTION_PROMPT_ZIP_SHA256,
        "r1_campaign_id": R1_CAMPAIGN_ID,
        "r1_root": str(r1_root),
        "r1_spec_paths": [
            {"path": str(path), "file_sha256": _sha256(path)}
            for path in r1_spec_paths
        ],
        "r1_run_roots": [str(path) for path in run_roots],
        "r2_runtime_root": str(runtime_root),
        "stage0_seed_authority": seed_audit,
        "stage1_positive_family_forensics": forensics,
        "r2_design": {
            "selected_branch": seed_audit["selected_r2_design_branch"],
            "science_campaign_id": R2_SCIENCE_CAMPAIGN_ID,
            "control_campaign_id": R2_CONTROL_CAMPAIGN_ID,
            "heldout_whole_seeds": heldout_seeds,
            "logical_science_cells": 24,
            "logical_control_cells": 2,
            "founder_profile": "neutral",
            "mutation_arms": [
                {"mutation_rate_q10": 0, "mutation_step_q10": 0},
                {"mutation_rate_q10": 16, "mutation_step_q10": 16},
            ],
            "generations": 30,
            "ticks_per_generation": 80,
            "primary_identity_type": PRIMARY_IDENTITY_TYPE,
            "secondary_identity_type": SECONDARY_IDENTITY_TYPE,
            "exact_seed1708_repeat_status": (
                "allowed_as_determinism_pipeline_control_only_not_independent_science"
            ),
            "resource_measurement_contract": resource_measurement_contract(),
            "reducer_audit_contract": reducer_audit_contract(),
            "retention_contract": retention_contract(),
            "decision_family": {
                "GV2-B48-R2-A": "seed_component_or_substream_localized",
                "GV2-B48-R2-B": "whole_seed_boundary_signal_independently_reproduced",
                "GV2-B48-R2-C": "not_reproduced_seed1708_original_trajectory_conditional",
                "GV2-B48-R2-D": "insufficient_or_integrity_blocked",
                "GV2-B48-R2-E": "integrity_failure",
            },
            "claim_boundary": (
                "R2 localizes or falsifies the R1 seed-1708 package-identity "
                "family only; no B72, ecology, adaptation, cognition, intelligence, "
                "emergence, profile-general, or allele-genotype-general claim."
            ),
        },
        "spec_outputs": spec_outputs,
    }


def build_package(
    *,
    r1_root: Path,
    r1_spec_paths: Sequence[Path],
    run_roots: Sequence[Path],
    output_root: Path,
    spec_output_root: Path | None,
) -> dict[str, Any]:
    runtime_root = output_root.parent / "runs"
    payload = preregistration(
        r1_root=r1_root,
        r1_spec_paths=r1_spec_paths,
        run_roots=run_roots,
        runtime_root=runtime_root,
        spec_output_root=spec_output_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    seed_path = output_root / "gv2_b48_r2_seed_decomposition_v1.json"
    forensics_path = output_root / "gv2_b48_r2_seed1708_positive_family_forensics_v1.json"
    reducer_contract_path = output_root / "gv2_b48_r2_reducer_audit_contract_v1.json"
    retention_contract_path = output_root / "gv2_b48_r2_retention_contract_v1.json"
    resource_contract_path = output_root / "gv2_b48_r2_resource_measurement_contract_v1.json"
    prereg_path = output_root / "gv2_b48_r2_preregistration_v1.json"
    package_path = output_root / "gv2_b48_r2_stage0_stage1_package_v1.json"
    _write_json(seed_path, payload["stage0_seed_authority"])
    _write_json(forensics_path, payload["stage1_positive_family_forensics"])
    _write_json(reducer_contract_path, payload["r2_design"]["reducer_audit_contract"])
    _write_json(retention_contract_path, payload["r2_design"]["retention_contract"])
    _write_json(resource_contract_path, payload["r2_design"]["resource_measurement_contract"])
    _write_json(prereg_path, payload)
    package_payload = {
        "schema_version": "gv2_b48_r2_stage0_stage1_package_v1",
        "preregistration_sha256": canonical_sha256(payload),
        "seed_decomposition_path": str(seed_path),
        "seed_decomposition_sha256": _sha256(seed_path),
        "positive_family_forensics_path": str(forensics_path),
        "positive_family_forensics_sha256": _sha256(forensics_path),
        "reducer_audit_contract_path": str(reducer_contract_path),
        "reducer_audit_contract_sha256": _sha256(reducer_contract_path),
        "retention_contract_path": str(retention_contract_path),
        "retention_contract_sha256": _sha256(retention_contract_path),
        "resource_measurement_contract_path": str(resource_contract_path),
        "resource_measurement_contract_sha256": _sha256(resource_contract_path),
        "preregistration_path": str(prereg_path),
        "preregistration_file_sha256": _sha256(prereg_path),
        "spec_outputs": payload.get("spec_outputs"),
        "stage_decision": "fallback_whole_seed_heldout_boundary_probe_preregistered",
    }
    package_payload["package_sha256"] = canonical_sha256(package_payload)
    _write_json(package_path, package_payload)
    return {
        "ok": True,
        "stage_decision": package_payload["stage_decision"],
        "heldout_seeds": payload["r2_design"]["heldout_whole_seeds"],
        "positive_family_interpretation": payload["stage1_positive_family_forensics"][
            "positive_family"
        ]["interpretation"],
        "paths": {
            "seed_decomposition": str(seed_path),
            "positive_family_forensics": str(forensics_path),
            "reducer_audit_contract": str(reducer_contract_path),
            "retention_contract": str(retention_contract_path),
            "resource_measurement_contract": str(resource_contract_path),
            "preregistration": str(prereg_path),
            "package": str(package_path),
        },
        "spec_outputs": payload.get("spec_outputs"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-root", type=Path, required=True)
    parser.add_argument("--r1-spec", type=Path, action="append", required=True)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spec-output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_package(
        r1_root=args.r1_root.expanduser().resolve(),
        r1_spec_paths=[path.expanduser().resolve() for path in args.r1_spec],
        run_roots=[path.expanduser().resolve() for path in args.run_root],
        output_root=args.output_root.expanduser().resolve(),
        spec_output_root=args.spec_output_root.expanduser().resolve()
        if args.spec_output_root is not None
        else None,
    )
    print(pretty_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
