from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from engine.campaigns.production_engine_provider import (
    CHECKPOINT_SCHEMA_VERSION,
    FRESH_OWNER_CHECKPOINT_REBIND_SCHEMA_VERSION,
    FRESH_OWNER_RESTORE_RECEIPT_SCHEMA_VERSION,
    PROFILE_ID,
    PROVIDER_ID,
    TECHNICAL_GATE_EVIDENCE_SCHEMA_VERSION,
    EielProductionExecutionProviderV1,
)


ROUTE_SHA256 = "7" * 64


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def _fixture(
    *,
    cell_count: int = 3,
    generations: int = 3,
    milestones: tuple[int, ...] = (0, 2, 3),
) -> tuple[dict[str, Any], tuple[str, ...], list[dict[str, Any]]]:
    cell_ids = tuple(f"gate-cell-{index}" for index in range(cell_count))
    cells = []
    for index, cell_id in enumerate(cell_ids):
        mode = f"mode-{index % 2}"
        pattern = f"pattern-{index}"
        path = f"path-{index % 2}"
        cells.append(
            {
                "cell_id": cell_id,
                "dimensions": {
                    "inheritance_mode": mode,
                    "founder_pattern": pattern,
                    "reproduction_path": path,
                    "paired_seed": 1000 + index,
                },
                "parameters": {
                    "inheritance_cross_path.mutation_enabled": False,
                    "inheritance_cross_path.inheritance_mode": mode,
                    "inheritance_cross_path.founder_pattern": pattern,
                    "inheritance_cross_path.reproduction_path": path,
                },
                "seeds": {"ecology": 1000 + index},
            }
        )
    canary_batch_id = "gate-canary"
    restore_batch_id = "gate-restore"
    plan = {
        "campaign_id": "generic-cross-path-gate",
        "plan_sha256": "1" * 64,
        "qualification_key_sha256": "2" * 64,
        "resolved_profile": PROFILE_ID,
        "cells": cells,
        "artifacts": {
            f"artifact-{index}": {
                "path": f"/not-read/artifact-{index}",
                "sha256": str(index + 3) * 64,
                "abi": f"abi-{index}",
            }
            for index in range(4)
        },
        "execution": {
            "technical_gate": {
                "required": True,
                "execution_mode": "pre_primary_canary",
                "cell_ids": list(cell_ids),
                "canary": {
                    "batch_id": canary_batch_id,
                    "restore_batch_id": restore_batch_id,
                    "generations": generations,
                    "checkpoint_milestones": list(milestones),
                    "cell_ids": list(cell_ids),
                },
            }
        },
    }
    contract = {
        "execution_mode": "pre_primary_canary",
        "cell_ids": list(cell_ids),
        "batch_id": canary_batch_id,
        "restore_batch_id": restore_batch_id,
        "generations": generations,
        "checkpoint_milestones": list(milestones),
    }
    binding = {
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile_id": PROFILE_ID,
        "provider_id": PROVIDER_ID,
        "execution_route_sha256": ROUTE_SHA256,
        "contract": contract,
    }
    gate_contract_sha256 = _digest(binding)
    artifact_sha256s = {
        name: row["sha256"] for name, row in sorted(plan["artifacts"].items())
    }

    def checkpoint(generation: int, batch_id: str) -> dict[str, Any]:
        digests = {
            cell_id: hashlib.sha256(
                f"{cell_id}:{generation}".encode()
            ).hexdigest()
            for cell_id in cell_ids
        }
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "execution_route_sha256": ROUTE_SHA256,
            "checkpoint_binding": {
                "campaign_id": plan["campaign_id"],
                "plan_sha256": plan["plan_sha256"],
                "qualification_key_sha256": plan["qualification_key_sha256"],
                "profile_id": PROFILE_ID,
                "provider_id": PROVIDER_ID,
                "execution_route_sha256": ROUTE_SHA256,
                "batch_id": batch_id,
                "cell_ids": list(cell_ids),
                "artifact_sha256s": artifact_sha256s,
            },
            "generation": generation,
            "continuation_storage_mode": "disk",
            "cell_bundles": [
                {
                    "cell_id": cell_id,
                    "path": f"/durable/{batch_id}/{cell_id}/g{generation}",
                    "source_projection_sha256": digests[cell_id],
                }
                for cell_id in cell_ids
            ],
            "terminal_cell_digests": digests,
        }

    metrics_rows = []
    for cell in cells:
        for generation in range(generations + 1):
            metrics_rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "generation": generation,
                    "identity": dict(cell["dimensions"]),
                    "mutation_events": 0,
                    "reproduction": {
                        "birth_opportunities": int(generation > 0),
                        "successful_births": int(generation > 0),
                        "failed_birth_attempts": 0,
                        "refill_child_count": (
                            int(generation > 0)
                            if cell["dimensions"]["reproduction_path"] == "path-0"
                            else 0
                        ),
                        "ordinary_birth_count": (
                            int(generation > 0)
                            if cell["dimensions"]["reproduction_path"] == "path-1"
                            else 0
                        ),
                    },
                }
            )

    canary_checkpoint = checkpoint(generations, canary_batch_id)
    restore_checkpoint = checkpoint(generations, restore_batch_id)
    terminal_digests = canary_checkpoint["terminal_cell_digests"]
    source_generation = max(
        (
            generation
            for generation in milestones
            if 0 < generation < generations
        ),
        default=generations,
    )
    source_checkpoint = checkpoint(source_generation, canary_batch_id)
    source_digests = source_checkpoint["terminal_cell_digests"]
    restore_checkpoint["terminal_cell_digests"] = dict(terminal_digests)
    for row in restore_checkpoint["cell_bundles"]:
        row["source_projection_sha256"] = terminal_digests[row["cell_id"]]
    manifests = [
        {
            "schema_version": "eiel_campaign_technical_gate_manifest_v1",
            **{name: binding[name] for name in (
                "campaign_id",
                "plan_sha256",
                "profile_id",
                "provider_id",
                "execution_route_sha256",
            )},
            "gate_contract_sha256": gate_contract_sha256,
            "gate_role": "canary_execute",
            "batch_id": canary_batch_id,
            "cell_ids": list(cell_ids),
            "status": "completed",
            "checkpoint": canary_checkpoint,
            "metrics_rows": deepcopy(metrics_rows),
            "result": {
                "terminal_generation": generations,
                "native_dispatches": len(milestones) - 1,
                "runner_receipts": [{"owner": "native"}],
                "execution_device": "cuda:0",
                "fallback_used": False,
                "checkpoint_catalog": {
                    str(generation): checkpoint(generation, canary_batch_id)
                    for generation in milestones
                },
                "provider_native_chronicle": {
                    "finalized": True,
                    "source_completeness": (
                        "provider_native_segment_artifact_complete_v1"
                    ),
                    "cell_receipts": {
                        cell_id: {
                            "cell_id": cell_id,
                            "finalized": True,
                            "segment_count": len(milestones) - 1,
                            "attempt_cursor": generations,
                        }
                        for cell_id in cell_ids
                    },
                },
            },
        },
        {
            "schema_version": "eiel_campaign_technical_gate_manifest_v1",
            **{name: binding[name] for name in (
                "campaign_id",
                "plan_sha256",
                "profile_id",
                "provider_id",
                "execution_route_sha256",
            )},
            "gate_contract_sha256": gate_contract_sha256,
            "gate_role": "fresh_owner_restore",
            "batch_id": restore_batch_id,
            "cell_ids": list(cell_ids),
            "status": "completed",
            "checkpoint": restore_checkpoint,
            "metrics_rows": deepcopy(metrics_rows),
            "result": {
                "terminal_generation": generations,
                "native_dispatches": 0,
                "runner_receipts": [],
                "execution_device": "cuda:0",
                "fallback_used": False,
                "fresh_owner_restore_receipt": {
                    "schema_version": FRESH_OWNER_RESTORE_RECEIPT_SCHEMA_VERSION,
                    "projection_bundles_reloaded": True,
                    "source_batch_id": canary_batch_id,
                    "restored_batch_id": restore_batch_id,
                    "cell_ids": list(cell_ids),
                    "source_checkpoint_generation": source_generation,
                    "source_checkpoint_cell_digests": dict(source_digests),
                    "restored_terminal_cell_digests": dict(terminal_digests),
                    "checkpoint_rebind": {
                        "schema_version": (
                            FRESH_OWNER_CHECKPOINT_REBIND_SCHEMA_VERSION
                        ),
                        "source_batch_id": canary_batch_id,
                        "restored_batch_id": restore_batch_id,
                        "source_checkpoint_binding": source_checkpoint[
                            "checkpoint_binding"
                        ],
                        "source_checkpoint_sha256": _digest(source_checkpoint),
                    },
                    "reload_receipts": [
                        {
                            "cell_id": cell_id,
                            "source_projection_sha256": source_digests[cell_id],
                            "loaded_projection_sha256": source_digests[cell_id],
                        }
                        for cell_id in cell_ids
                    ],
                },
            },
        },
    ]
    return plan, cell_ids, manifests


def _evaluate(
    plan: dict[str, Any],
    cell_ids: tuple[str, ...],
    manifests: list[dict[str, Any]],
    tmp_path: Path,
) -> dict[str, Any]:
    provider = EielProductionExecutionProviderV1()
    return provider.evaluate_technical_gate(
        plan=plan,
        gate_cell_ids=cell_ids,
        batch_manifests=manifests,
        run_root=tmp_path,
    )


def _gv2_exposure_receipt(
    *,
    batch_id: str,
    cell_ids: tuple[str, ...],
    native_execution_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    cell_receipts = {
        cell_id: {
            "cell_id": cell_id,
            "segment_count": len(native_execution_receipts),
            "attempt_count": len(native_execution_receipts),
            "success_count": len(native_execution_receipts),
            "failure_count": 0,
            "eligible_snapshot_count": len(native_execution_receipts),
            "chain_sha256_after_last_segment": hashlib.sha256(
                f"{batch_id}:{cell_id}".encode()
            ).hexdigest(),
            "cumulative_counts_after_last_segment": {
                "attempts": len(native_execution_receipts),
                "successes": len(native_execution_receipts),
                "failures": 0,
                "eligible_snapshots": len(native_execution_receipts),
            },
        }
        for cell_id in cell_ids
    }
    payload = {
        "schema_version": "eiel_provider_native_exposure_receipt_v1",
        "batch_id": batch_id,
        "profile_id": PROFILE_ID,
        "execution_route_sha256": ROUTE_SHA256,
        "cell_ids": list(cell_ids),
        "segment_indices": list(range(len(native_execution_receipts))),
        "native_execution_receipts_sha256": _digest(native_execution_receipts),
        "cell_receipts": cell_receipts,
        "totals": {
            "attempt_count": len(cell_ids) * len(native_execution_receipts),
            "success_count": len(cell_ids) * len(native_execution_receipts),
            "failure_count": 0,
            "eligible_snapshot_count": len(cell_ids)
            * len(native_execution_receipts),
            "decision_bearing_drop_count": 0,
        },
    }
    return {**payload, "receipt_sha256": _digest(payload)}


def _gv2_fixture() -> tuple[dict[str, Any], tuple[str, ...], list[dict[str, Any]]]:
    cell_ids = ("gv2-cell-disabled", "gv2-cell-enabled")
    cells = [
        {
            "cell_id": cell_ids[0],
            "dimensions": {
                "ecology_seed": 101,
                "task_seed": 0,
                "founder_profile": "neutral",
                "mutation_rate_q10": 0,
                "mutation_step_q10": 0,
            },
            "parameters": {
                "genome_upgrade_v2.enable": True,
                "genome_upgrade_v2.founder_profile": "neutral",
                "genome_upgrade_v2.mutation_rate_q10": 0,
                "genome_upgrade_v2.mutation_step_q10": 0,
                "production_lifecycle.ordinary_births_enabled": True,
            },
            "seeds": {"ecology": 101, "task": 0},
        },
        {
            "cell_id": cell_ids[1],
            "dimensions": {
                "ecology_seed": 202,
                "task_seed": 0,
                "founder_profile": "heterozygous",
                "mutation_rate_q10": 16,
                "mutation_step_q10": 16,
            },
            "parameters": {
                "genome_upgrade_v2.enable": True,
                "genome_upgrade_v2.founder_profile": "heterozygous",
                "genome_upgrade_v2.mutation_rate_q10": 16,
                "genome_upgrade_v2.mutation_step_q10": 16,
                "production_lifecycle.ordinary_births_enabled": True,
            },
            "seeds": {"ecology": 202, "task": 0},
        },
    ]
    generations = 3
    milestones = (0, 1, 3)
    canary_batch_id = "gv2-gate-canary"
    restore_batch_id = "gv2-gate-restore"
    plan = {
        "campaign_id": "gv2-wdl-gate",
        "plan_sha256": "1" * 64,
        "qualification_key_sha256": "2" * 64,
        "resolved_profile": PROFILE_ID,
        "cells": cells,
        "artifacts": {
            "cuda_extension": {
                "path": "/not-read/extension.so",
                "sha256": "3" * 64,
                "abi": "architecture_b_wdl_fixed_mlp_v2",
            },
            "extension_manifest": {
                "path": "/not-read/manifest.json",
                "sha256": "4" * 64,
                "abi": "architecture_b_wdl_fixed_mlp_v2",
            },
            "genome_upgrade_v2_package_receipt": {
                "path": "/not-read/package.json",
                "sha256": "5" * 64,
                "abi": "genome_upgrade_v2_fixed_q10_v1",
            },
        },
        "execution": {
            "backend": "cuda_resident",
            "technical_gate": {
                "required": True,
                "execution_mode": "pre_primary_canary",
                "profile": "genome_v2_wdl_fixed_mutation_persistence_v1",
                "cell_ids": list(cell_ids),
                "canary": {
                    "batch_id": canary_batch_id,
                    "restore_batch_id": restore_batch_id,
                    "generations": generations,
                    "checkpoint_milestones": list(milestones),
                    "cell_ids": list(cell_ids),
                },
            },
        },
    }
    contract = {
        "execution_mode": "pre_primary_canary",
        "cell_ids": list(cell_ids),
        "batch_id": canary_batch_id,
        "restore_batch_id": restore_batch_id,
        "generations": generations,
        "checkpoint_milestones": list(milestones),
        "profile": "genome_v2_wdl_fixed_mutation_persistence_v1",
    }
    binding = {
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile_id": PROFILE_ID,
        "provider_id": PROVIDER_ID,
        "execution_route_sha256": ROUTE_SHA256,
        "contract": contract,
    }
    artifact_sha256s = {
        name: row["sha256"] for name, row in sorted(plan["artifacts"].items())
    }

    def checkpoint(generation: int, batch_id: str) -> dict[str, Any]:
        digests = {
            cell_id: hashlib.sha256(
                f"{cell_id}:{generation}:{batch_id}".encode()
            ).hexdigest()
            for cell_id in sorted(cell_ids)
        }
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "execution_route_sha256": ROUTE_SHA256,
            "checkpoint_binding": {
                "campaign_id": plan["campaign_id"],
                "plan_sha256": plan["plan_sha256"],
                "qualification_key_sha256": plan["qualification_key_sha256"],
                "profile_id": PROFILE_ID,
                "provider_id": PROVIDER_ID,
                "execution_route_sha256": ROUTE_SHA256,
                "batch_id": batch_id,
                "cell_ids": list(cell_ids),
                "artifact_sha256s": artifact_sha256s,
            },
            "generation": generation,
            "continuation_storage_mode": "disk",
            "cell_bundles": [
                {
                    "cell_id": cell_id,
                    "path": f"/durable/{batch_id}/{cell_id}/g{generation}",
                    "source_projection_sha256": digests[cell_id],
                }
                for cell_id in cell_ids
            ],
            "terminal_cell_digests": digests,
        }

    canary_terminal = checkpoint(generations, canary_batch_id)
    restore_terminal = checkpoint(generations, restore_batch_id)
    restore_terminal["terminal_cell_digests"] = dict(
        canary_terminal["terminal_cell_digests"]
    )
    for row in restore_terminal["cell_bundles"]:
        row["source_projection_sha256"] = restore_terminal[
            "terminal_cell_digests"
        ][row["cell_id"]]
    source_checkpoint = checkpoint(1, canary_batch_id)
    metrics_rows = [
        {
            "schema_version": "eiel_genome_v2_e1_native_cell_metrics_v1",
            "cell_id": cell_id,
            "generation_metrics": [
                {
                    "generation": generation,
                    "alive": 64,
                    "successful_births": 1,
                }
                for generation in range(generations)
            ],
            "metrics": {
                "population.births": 3,
                "population.terminal_alive": 64,
            },
            "genome_v2_attribution": {
                "schema_version": "genome_v2_e1_metric_attribution_v1",
                "cell_id": cell_id,
                "generation": generations,
                "summary_sha256": hashlib.sha256(cell_id.encode()).hexdigest(),
                "metrics": {"genome_v2.context_observation_count": 64},
            },
        }
        for cell_id in cell_ids
    ]
    canary_receipts = [
        {
            "segment_ordinal": 1,
            "start_generation": 0,
            "end_generation": 1,
            "execution_device": "cuda:0",
            "native_owner": "native_extension",
            "hidden_fallback": False,
            "runtime_jit": False,
            "explicit_native_no_fallback": True,
        },
        {
            "segment_ordinal": 2,
            "start_generation": 1,
            "end_generation": 3,
            "execution_device": "cuda:0",
            "native_owner": "native_extension",
            "hidden_fallback": False,
            "runtime_jit": False,
            "explicit_native_no_fallback": True,
        },
    ]
    restore_receipts = [
        {
            "segment_ordinal": 1,
            "start_generation": 1,
            "end_generation": 3,
            "execution_device": "cuda:0",
            "native_owner": "native_extension",
            "hidden_fallback": False,
            "runtime_jit": False,
            "explicit_native_no_fallback": True,
        }
    ]

    def runner_receipt(cell: dict[str, Any]) -> dict[str, Any]:
        dimensions = cell["dimensions"]
        rate = dimensions["mutation_rate_q10"]
        step = dimensions["mutation_step_q10"]
        return {
            "schema_version": "eiel_patch_wdl_fixed_mlp_runner_factory_v1",
            "cell_id": cell["cell_id"],
            "genome": {
                "enabled": True,
                "profile": "genome_upgrade_v2_fixed_q10_v1",
                "founder_profile_id": dimensions["founder_profile"],
                "mutation_rate_q10": rate,
                "mutation_step_q10": step,
            },
            "production_lifecycle": {
                "enabled": True,
                "ordinary_births_enabled": True,
                "genome_v2_causal_capture_enabled": True,
                "genome_v2_mutation_enabled": rate > 0,
                "diploid_mutation_enabled": rate > 0,
                "legacy_scalar_mutation_off": rate == 0,
                "genome_v2_mutation_rate_q10": rate,
                "genome_v2_mutation_step_q10": step,
            },
            "seed_ownership": {
                "named_owners": {
                    "ecology": dimensions["ecology_seed"],
                    "task": dimensions["task_seed"],
                }
            },
        }

    runner_receipts = [runner_receipt(cell) for cell in cells]
    manifests = [
        {
            "schema_version": "eiel_campaign_technical_gate_manifest_v1",
            **{
                name: binding[name]
                for name in (
                    "campaign_id",
                    "plan_sha256",
                    "profile_id",
                    "provider_id",
                    "execution_route_sha256",
                )
            },
            "gate_contract_sha256": _digest(binding),
            "gate_role": "canary_execute",
            "batch_id": canary_batch_id,
            "cell_ids": list(cell_ids),
            "status": "completed",
            "checkpoint": canary_terminal,
            "metrics_rows": deepcopy(metrics_rows),
            "result": {
                "terminal_generation": generations,
                "native_dispatches": 2,
                "execution_device": "cuda:0",
                "fallback_used": False,
                "native_execution_receipts": canary_receipts,
                "runner_receipts": deepcopy(runner_receipts),
                "checkpoint_catalog": {
                    str(generation): checkpoint(generation, canary_batch_id)
                    for generation in milestones
                },
                "provider_native_exposure_receipt": _gv2_exposure_receipt(
                    batch_id=canary_batch_id,
                    cell_ids=cell_ids,
                    native_execution_receipts=canary_receipts,
                ),
            },
        },
        {
            "schema_version": "eiel_campaign_technical_gate_manifest_v1",
            **{
                name: binding[name]
                for name in (
                    "campaign_id",
                    "plan_sha256",
                    "profile_id",
                    "provider_id",
                    "execution_route_sha256",
                )
            },
            "gate_contract_sha256": _digest(binding),
            "gate_role": "fresh_owner_restore",
            "batch_id": restore_batch_id,
            "cell_ids": list(cell_ids),
            "status": "completed",
            "checkpoint": restore_terminal,
            "metrics_rows": deepcopy(metrics_rows),
            "result": {
                "terminal_generation": generations,
                "native_dispatches": 1,
                "execution_device": "cuda:0",
                "fallback_used": False,
                "native_execution_receipts": restore_receipts,
                "runner_receipts": deepcopy(runner_receipts),
                "fresh_owner_restore_receipt": {
                    "schema_version": FRESH_OWNER_RESTORE_RECEIPT_SCHEMA_VERSION,
                    "projection_bundles_reloaded": True,
                    "source_batch_id": canary_batch_id,
                    "restored_batch_id": restore_batch_id,
                    "cell_ids": list(cell_ids),
                    "source_checkpoint_generation": 1,
                    "source_checkpoint_cell_digests": dict(
                        source_checkpoint["terminal_cell_digests"]
                    ),
                    "restored_terminal_cell_digests": dict(
                        restore_terminal["terminal_cell_digests"]
                    ),
                    "checkpoint_rebind": {
                        "schema_version": (
                            FRESH_OWNER_CHECKPOINT_REBIND_SCHEMA_VERSION
                        ),
                        "source_batch_id": canary_batch_id,
                        "restored_batch_id": restore_batch_id,
                        "source_checkpoint_binding": source_checkpoint[
                            "checkpoint_binding"
                        ],
                        "source_checkpoint_sha256": _digest(source_checkpoint),
                    },
                    "reload_receipts": [
                        {
                            "cell_id": cell_id,
                            "source_projection_sha256": source_checkpoint[
                                "terminal_cell_digests"
                            ][cell_id],
                            "loaded_projection_sha256": source_checkpoint[
                                "terminal_cell_digests"
                            ][cell_id],
                        }
                        for cell_id in cell_ids
                    ],
                },
                "provider_native_exposure_receipt": _gv2_exposure_receipt(
                    batch_id=restore_batch_id,
                    cell_ids=cell_ids,
                    native_execution_receipts=restore_receipts,
                ),
            },
        },
    ]
    return plan, cell_ids, manifests


def test_complete_dimension_driven_gate_evidence_passes(tmp_path: Path) -> None:
    plan, cell_ids, manifests = _fixture(
        cell_count=3,
        generations=3,
        milestones=(0, 2, 3),
    )

    receipt = _evaluate(plan, cell_ids, manifests, tmp_path)

    assert receipt["schema_version"] == TECHNICAL_GATE_EVIDENCE_SCHEMA_VERSION
    assert receipt["gate_cell_ids"] == list(cell_ids)
    assert receipt["checkpoint_milestones"] == [0, 2, 3]
    assert receipt["terminal_generation"] == 3
    assert receipt["passed"] is True
    assert receipt["reasons"] == []
    assert all(receipt["checks"].values())


def test_genome_v2_wdl_gate_accepts_nested_metrics_and_mutation_arms(
    tmp_path: Path,
) -> None:
    plan, cell_ids, manifests = _gv2_fixture()

    receipt = _evaluate(plan, cell_ids, manifests, tmp_path)

    assert receipt["passed"] is True
    assert receipt["route_family_profile"] == (
        "genome_v2_wdl_fixed_mutation_persistence_v1"
    )
    assert receipt["checks"]["mutation_zero"] is True
    assert receipt["checks"]["provider_native_exposure_receipts_complete"] is True


def test_genome_v2_wdl_gate_accepts_without_legacy_runner_receipts(
    tmp_path: Path,
) -> None:
    plan, cell_ids, manifests = _gv2_fixture()
    for manifest in manifests:
        manifest["result"].pop("runner_receipts", None)

    receipt = _evaluate(plan, cell_ids, manifests, tmp_path)

    assert receipt["passed"] is True


def test_genome_v2_wdl_gate_requires_explicit_compiled_profile(
    tmp_path: Path,
) -> None:
    plan, cell_ids, manifests = _gv2_fixture()
    del plan["execution"]["technical_gate"]["profile"]

    receipt = _evaluate(plan, cell_ids, manifests, tmp_path)

    assert receipt["passed"] is False
    assert any("explicit route family profile" in reason for reason in receipt["reasons"])


def test_genome_v2_wdl_gate_rejects_profile_mechanics_conflict(
    tmp_path: Path,
) -> None:
    plan, cell_ids, manifests = _fixture()
    plan["execution"]["technical_gate"][
        "profile"
    ] = "genome_v2_wdl_fixed_mutation_persistence_v1"

    receipt = _evaluate(plan, cell_ids, manifests, tmp_path)

    assert receipt["passed"] is False
    assert any("profile conflicts with compiled mechanics" in reason for reason in receipt["reasons"])


def test_genome_v2_wdl_gate_requires_native_exposure_receipts(
    tmp_path: Path,
) -> None:
    plan, cell_ids, manifests = _gv2_fixture()
    del manifests[1]["result"]["provider_native_exposure_receipt"]

    receipt = _evaluate(plan, cell_ids, manifests, tmp_path)

    assert receipt["passed"] is False
    assert receipt["checks"]["provider_native_exposure_receipts_complete"] is False
    assert any(
        "provider-native reproduction exposure receipts" in reason
        for reason in receipt["reasons"]
    )


def test_genome_v2_wdl_gate_rejects_historical_segments_counted_as_current(
    tmp_path: Path,
) -> None:
    plan, cell_ids, manifests = _gv2_fixture()
    receipt = manifests[1]["result"]["provider_native_exposure_receipt"]
    receipt["segment_indices"] = [0, 1]

    evaluated = _evaluate(plan, cell_ids, manifests, tmp_path)

    assert evaluated["passed"] is False
    assert evaluated["checks"]["provider_native_exposure_receipts_complete"] is False


def test_genome_v2_wdl_gate_requires_post_restore_execution(
    tmp_path: Path,
) -> None:
    plan, cell_ids, manifests = _gv2_fixture()
    manifests[1]["result"]["native_execution_receipts"][0][
        "start_generation"
    ] = 3

    receipt = _evaluate(plan, cell_ids, manifests, tmp_path)

    assert receipt["passed"] is False
    assert receipt["checks"]["cuda_without_fallback"] is False


def test_genome_v2_wdl_gate_rejects_malformed_mutation_arm(
    tmp_path: Path,
) -> None:
    plan, cell_ids, manifests = _gv2_fixture()
    plan["cells"][1]["parameters"]["genome_upgrade_v2.mutation_step_q10"] = 0

    receipt = _evaluate(plan, cell_ids, manifests, tmp_path)

    assert receipt["passed"] is False
    assert receipt["checks"]["mutation_zero"] is False
    assert any("mutation arm" in reason for reason in receipt["reasons"])


def test_missing_g0_checkpoint_fails_with_explicit_reason(tmp_path: Path) -> None:
    plan, cell_ids, manifests = _fixture()
    del manifests[0]["result"]["checkpoint_catalog"]["0"]

    receipt = _evaluate(plan, cell_ids, manifests, tmp_path)

    assert receipt["passed"] is False
    assert receipt["checks"]["checkpoint_milestones_exact"] is False
    assert any("G0 durable checkpoint proof is absent" in reason for reason in receipt["reasons"])


def test_terminal_fast_path_without_reload_receipt_fails(tmp_path: Path) -> None:
    plan, cell_ids, manifests = _fixture()
    del manifests[1]["result"]["fresh_owner_restore_receipt"]
    manifests[1]["result"]["terminal_checkpoint_recovered"] = True

    receipt = _evaluate(plan, cell_ids, manifests, tmp_path)

    assert receipt["passed"] is False
    assert (
        receipt["checks"]["fresh_owner_bundle_reload_and_equality_proven"]
        is False
    )
    assert any("terminal fast-path recovery is insufficient" in reason for reason in receipt["reasons"])


@pytest.mark.parametrize(
    ("mutate", "check_name", "reason_fragment"),
    [
        (
            lambda _plan, manifests: manifests[0]["metrics_rows"][1].__setitem__(
                "mutation_events", 1
            ),
            "mutation_zero",
            "records mutation",
        ),
        (
            lambda _plan, manifests: manifests[0]["result"].__setitem__(
                "fallback_used", True
            ),
            "cuda_without_fallback",
            "CUDA/no-fallback",
        ),
        (
            lambda plan, manifests: manifests[0]["metrics_rows"][0][
                "identity"
            ].__setitem__("reproduction_path", "wrong-path"),
            "path_and_mode_identity_exact",
            "telemetry identity differs",
        ),
        (
            lambda _plan, manifests: [
                row["reproduction"].__setitem__("birth_opportunities", 0)
                for row in manifests[0]["metrics_rows"]
            ],
            "real_reproduction_exposure",
            "lack measured reproduction exposure",
        ),
    ],
)
def test_scientific_and_route_evidence_fail_closed(
    tmp_path: Path,
    mutate: Any,
    check_name: str,
    reason_fragment: str,
) -> None:
    plan, cell_ids, manifests = _fixture()
    mutate(plan, manifests)

    receipt = _evaluate(plan, cell_ids, manifests, tmp_path)

    assert receipt["passed"] is False
    assert receipt["checks"][check_name] is False
    assert any(reason_fragment in reason for reason in receipt["reasons"])


def test_plan_and_artifact_binding_drift_fails(tmp_path: Path) -> None:
    plan, cell_ids, manifests = _fixture()
    manifests[0]["checkpoint"]["checkpoint_binding"]["artifact_sha256s"][
        "artifact-0"
    ] = "f" * 64

    receipt = _evaluate(plan, cell_ids, manifests, tmp_path)

    assert receipt["passed"] is False
    assert receipt["checks"]["checkpoint_milestones_exact"] is False
    assert receipt["checks"]["fresh_owner_terminal_checkpoint_bound"] is False
    assert any(
        "checkpoint evidence is incomplete or binding-invalid" in reason
        for reason in receipt["reasons"]
    )
