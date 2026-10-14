from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import eiel_composition.science_cuda_v2_generic_host_context_v1 as generic_context
import eiel_composition.science_request_v1_front_door as front_door
from eiel_composition.science_instrument_cli import main as science_cli_main
from eiel_composition.science_instrument_artifacts_v1 import (
    serialize_science_host_context_v1,
)
from eiel_composition.science_instrument_operator_v1 import (
    load_science_instrument_materialization_v1,
)
from eiel_composition.science_request_v1_front_door import (
    AMBIGUOUS_SCIENCE_AUTHORITY,
    MISSING_SCIENTIFIC_CAPABILITY,
    ScienceExecutionAuthorityV1,
    ScienceRequestV1FrontDoorError,
    activate_science_execution_authority_v1,
    plan_science_request_v1,
    resolve_current_science_authority_v1,
    run_science_request_v1,
    validate_science_request_v1,
)
from emergence_contracts import ContentDigest, canonical_sha256
from emergence_contracts.science_request_v1 import validate_eiel_science_request_v1
from engine.campaigns.artifact_lifecycle import (
    build_artifact_disposition_manifest_v1,
    build_sealed_artifact_inventory_row_v1,
    build_storage_disposition_receipt_v1,
    lifecycle_profile_contract_v1,
)
from engine.campaigns.baseline_v1 import (
    BaselineDeclarationV1,
    load_baseline_profile_v1,
)
from eiel_composition.science_request_v1_compiler import compile_science_request_v1
from tests.test_eiel_science_request_v1_compiler import _selected_world_request


_EXAMPLES = Path("_spec/campaigns/v1/eiel_science_request_v1/examples")
_W32 = _EXAMPLES / "EIEL_SCIENCE_REQUEST_V1_selection_window_w32_b1024_example.json"
_W128 = (
    _EXAMPLES
    / "EIEL_SCIENCE_REQUEST_V1_mutation_pressure_w128x80_multibatch_example.json"
)


def test_selected_world_compilation_selects_host_observation_boundary() -> None:
    compilation = compile_science_request_v1(
        validate_eiel_science_request_v1(_selected_world_request())
    )
    world_id = compilation.world_bundle.worlds[0].world_id

    assert front_door._compilation_uses_selected_world_continuation_v1(compilation)
    assert front_door._materialized_world_ids_v1(
        {
            "schema_version": "eiel_science_selected_world_technical_lowering_v1",
            "semantic_world_id": world_id,
        }
    ) == [world_id]


def test_selected_world_completed_child_resume_preserves_host_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    attempt = root / "batches" / "batch-0000" / "attempt-0001"
    attempt.mkdir(parents=True)
    binding = {"observation_mode": "authenticated-host-fixture"}
    result_core = {
        "ok": True,
        "status": "COMPLETED",
        "result_class": "SUCCESS",
        "selected_world_host_observation": binding,
    }
    result = {
        **result_core,
        "result_sha256": canonical_sha256(result_core).hex,
    }
    attempt_result = attempt / "batch_result.json"
    canonical_result = root / "batch_result_0000.json"
    _write_json(attempt_result, result)
    _write_json(canonical_result, result)
    child_lifecycle = {"authenticated": True}
    monkeypatch.setattr(
        front_door,
        "_verify_science_batch_result_v1",
        lambda **_kwargs: child_lifecycle,
    )

    loaded, receipt = front_door._load_completed_science_batch_v1(
        root=root,
        row={
            "attempt_root": str(attempt.relative_to(root)),
            "result": {
                "path": str(attempt_result.relative_to(root)),
                "file_sha256": _file_sha256(attempt_result),
                "result_sha256": result["result_sha256"],
            },
        },
        batch_index=0,
        lowered={
            "science_materialization": {
                "schema_version": (
                    "eiel_science_selected_world_technical_lowering_v1"
                ),
                "semantic_world_id": "world-selected",
            }
        },
        materialization_receipt={},
        require_launchpad_lifecycle=True,
    )

    assert loaded == result
    assert receipt["materialized_world_ids"] == ["world-selected"]
    assert receipt["selected_world_host_observation"] == binding
    assert receipt["child_lifecycle"] == child_lifecycle


def test_selected_world_fresh_child_skips_compact_atlas_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_envelope(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("selected-world child must not request a compact atlas")

    monkeypatch.setattr(
        front_door, "_science_batch_result_envelope_v1", forbidden_envelope
    )

    assert (
        front_door._science_batch_result_envelope_for_batch_v1(
            result={"ok": True},
            lowered={
                "science_materialization": {
                    "schema_version": (
                        "eiel_science_selected_world_technical_lowering_v1"
                    )
                }
            },
            physical_batch_index=0,
            request_sha256=ContentDigest("sha256", "1" * 64),
            compilation=object(),  # type: ignore[arg-type]
        )
        is None
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _small_w32_request(tmp_path: Path) -> Path:
    value = json.loads(_W32.read_text(encoding="utf-8"))
    value["campaign_id"] = "EIEL-SELECTION-WINDOW-W32-MULTIBATCH-TEST-V1"
    value["dimensions"] = {
        "pressure_onset": [0, 2500],
        "pressure_intensity": [0.0, 0.5],
        "mutation_enabled": [False],
        "mutation_step": [1],
        "seed": [1103, 2207],
    }
    value["run"]["batch"] = 3
    value["expect_worlds"] = 8
    path = tmp_path / "small_w32_request.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _synthetic_batch_result(
    materialization: Path, *, batch_index: int, runtime_root: Path
) -> dict[str, Any]:
    _, receipt, _, lowered = load_science_instrument_materialization_v1(materialization)
    identities = receipt["identities"]
    source = receipt["lineage"]
    return {
        "schema_version": "synthetic_science_batch_result_v1",
        "ok": True,
        "status": "completed",
        "result_class": "SUCCESS",
        "batch_index": batch_index,
        "runtime_root": str(runtime_root),
        "scientific_execution": True,
        "science_lineage": {
            "schema_version": "eiel_science_instrument_execution_lineage_v1",
            "instrument_contract": "EIEL_SCIENCE_INSTRUMENT_V1",
            "science_use_class": lowered[batch_index]["science_materialization"][
                "science_use_class"
            ],
            "observation_mode": lowered[batch_index]["science_materialization"][
                "observation_mode"
            ],
            "request_sha256": identities["request_sha256"],
            "campaign_spec_sha256": identities["campaign_spec_sha256"],
            "compilation_bundle_sha256": identities["compilation_bundle_sha256"],
            "batch_index": batch_index,
            "world_ids": lowered[batch_index]["science_materialization"][
                "semantic_world_ids"
            ],
            "bundle_content_sha256": identities["bundle_content_sha256"],
            "bundle_admission_sha256": identities["bundle_admission_sha256"],
            "carrier_plan_sha256": identities["carrier_plan_sha256"],
            "base_campaign_plan_sha256": identities["base_campaign_plan_sha256"],
            "lowered_plan_sha256": lowered[batch_index]["plan_sha256"],
        },
        "source_lineage": {
            **source,
            "compiled_plan_sha256": lowered[batch_index]["plan_sha256"],
            "bound_plan_sha256": "8" * 64,
        },
        "compact_reducer": {
            "status": "COMPLETED",
            "atlas_result": {"synthetic_batch": batch_index},
        },
    }


def _base_plan() -> dict[str, Any]:
    checkpoints = (25_000, 50_000, 75_000, 100_000)
    plan: dict[str, Any] = {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": "science-request-v1-front-door-test-base",
        "artifact_lifecycle": lifecycle_profile_contract_v1(
            "SCIENCE_COMPACT",
            total_max_bytes=512 * 1024**2,
            checkpoint_schedule=checkpoints,
            checkpoint_authority="absolute_tick",
        ),
        "resolved_profile": "eiel_full_evolution_cuda_native_v2",
        "execution": {
            "total_ticks": 100_000,
            "start_next_tick": 0,
            "checkpoint_ticks": list(checkpoints),
            "tick_native_control": {
                "schedules": {
                    "checkpoint_ticks": list(checkpoints),
                    "report_ticks": [100_000],
                    "science_sample_ticks": [100_000],
                    "reducer_ticks": [100_000],
                    "archive_ticks": list(checkpoints),
                }
            },
        },
        "output": {"runtime_root": "not-bound-yet"},
        "cells": [{"parameters": {}}],
    }
    plan["plan_sha256"] = canonical_sha256(plan).hex
    return plan


def test_workhorse_off_runtime_authority_is_capability_scoped_and_fail_closed() -> None:
    profile = type(
        "Profile",
        (),
        {"provider_id": front_door.science_carrier.WORKHORSE_PROVIDER_ID},
    )()
    plan = {
        "analysis": {
            "extension": {
                "contract": {"operation": "EXECUTION_ROW"},
            }
        },
        "execution": {
            "native_streamed_checkpoint": True,
            "acs_checkpoint_slots": 2,
        },
        "science_materialization": {"observation_mode": "OFF"},
    }
    assert front_door._workhorse_off_host_observation_authority_v1(
        plan,
        profile=profile,
    )
    for field, value in (
        ("operation", "SEMANTIC_EVIDENCE_CASE"),
        ("native_streamed_checkpoint", False),
        ("acs_checkpoint_slots", 1),
        ("observation_mode", "COMPACT_SIDECAR"),
    ):
        changed = json.loads(json.dumps(plan))
        if field == "operation":
            changed["analysis"]["extension"]["contract"][field] = value
        elif field == "observation_mode":
            changed["science_materialization"][field] = value
        else:
            changed["execution"][field] = value
        assert not front_door._workhorse_off_host_observation_authority_v1(
            changed,
            profile=profile,
        )


def test_development_baseline_is_bound_to_sealed_acs_slot_count() -> None:
    all_ids = tuple(
        row.requirement_id
        for row in BaselineDeclarationV1.from_value(
            front_door._development_baseline()
        ).deviations
    )
    acs2_ids = tuple(
        row.requirement_id
        for row in BaselineDeclarationV1.from_value(
            front_door._development_baseline(acs_checkpoint_slots=2)
        ).deviations
    )
    assert all_ids == (
        "evolution.full_evolution",
        "runtime.production_cuda",
        "science.wrc_variability",
        "checkpoint.acs_two_slot",
        "authority.device_fallback",
    )
    assert acs2_ids == tuple(
        value for value in all_ids if value != "checkpoint.acs_two_slot"
    )
    with pytest.raises(ScienceRequestV1FrontDoorError, match="ACS checkpoint slots"):
        front_door._development_baseline(acs_checkpoint_slots=3)


def test_one_child_lifecycle_stays_byte_identical_when_base_is_sufficient() -> None:
    base = _base_plan()["artifact_lifecycle"]
    child = lifecycle_profile_contract_v1(
        "SCIENCE_COMPACT",
        total_max_bytes=64 * 1024**2,
        checkpoint_schedule=(25_000, 50_000, 75_000, 100_000),
        checkpoint_authority="absolute_tick",
    )

    aggregated = front_door._aggregate_science_run_lifecycle_v1(
        base,
        [{"artifact_lifecycle": child}],
        coordinator_durable_allowance_bytes=16 * 1024**2,
    )

    assert aggregated == base
    assert aggregated["contract_sha256"] == base["contract_sha256"]


def _identity() -> dict[str, Any]:
    return {
        "route_id": "tensor_world_cuda_full_evolution_v2",
        "profile_id": "eiel_full_evolution_cuda_native_v2",
        "carrier_profile_id": "w32x32p256c512_gpuv2",
        "provider_id": "eiel-full-evolution-cuda-native-provider-v1",
        "provider_family": "cuda_full_evolution_v2",
        "execution_route_sha256": ContentDigest("sha256", "1" * 64),
        "binary_sha256": ContentDigest("sha256", "2" * 64),
        "manifest_sha256": ContentDigest("sha256", "3" * 64),
        "manifest_file_sha256": ContentDigest("sha256", "4" * 64),
        "artifact_path": "/immutable/test/native.so",
        "device": {
            "index": 0,
            "name": "TEST CUDA DEVICE",
            "uuid": "test-device-v1",
            "sm": "sm_75",
            "total_memory_bytes": 8 * 1024**3,
        },
        "device_selection_receipt_sha256": ContentDigest("sha256", "5" * 64),
        "native_abi": "eiel_full_evolution_cuda_native_abi_v1",
    }


def _test_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ScienceExecutionAuthorityV1:
    plan = _base_plan()
    plan_path = tmp_path / "base_plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        generic_context,
        "_plan_identity",
        lambda value: (_identity(), 1024, 64),
    )
    context = generic_context.materialize_cuda_v2_explicit_bundle_host_context_v1(
        plan_path,
        logical_world_capacity=1024,
        context_scope_id="science-request-v1-front-door-test",
    )
    authority_core = {
        "schema_version": "eiel_science_execution_authority_v1",
        "authority_id": "test-w32-b1024-authority-v1",
        "status": "CURRENT",
        "world": {
            "width": 32,
            "height": 32,
            "population": 256,
            "capacity": 512,
        },
        "base_campaign_plan": {
            "path": str(plan_path.resolve()),
            "file_sha256": _file_sha256(plan_path),
            "plan_sha256": plan["plan_sha256"],
        },
    }
    authority_row = {
        **authority_core,
        "authority_sha256": canonical_sha256(authority_core).hex,
    }
    authority_path = tmp_path / "science_authority.json"
    authority_path.write_text(
        json.dumps(authority_row, sort_keys=True), encoding="utf-8"
    )
    return ScienceExecutionAuthorityV1(
        authority_id="test-w32-b1024-authority-v1",
        authority_sha256=authority_row["authority_sha256"],
        authority_source=str(authority_path.resolve()),
        authority_source_file_sha256=_file_sha256(authority_path),
        base_campaign_plan_path=plan_path.resolve(),
        base_campaign_plan_file_sha256=_file_sha256(plan_path),
        base_campaign_plan_sha256=plan["plan_sha256"],
        baseline=front_door._development_baseline(),
        baseline_profile_sha256=load_baseline_profile_v1()["profile_sha256"],
        science_context=serialize_science_host_context_v1(context),
        available_physical_batch_capacity=1024,
        physical_batch_capacity=1024,
        temporal_k=64,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _authenticated_child_inventory_fixture(
    root: Path,
    *,
    inventory_complete: bool = True,
    release_first: bool = False,
) -> tuple[Path, dict[str, Any]]:
    attempt = root / "batches" / "batch-0000" / "attempt-0001"
    attempt.mkdir(parents=True)
    objects = (
        (
            attempt / "batches/science-batch-0000/scratch/events/"
            "dispatch-tick-000000000000-000000000064/direct-sparse-events-v3.bin",
            "raw-envelope",
            "cuda.direct_sparse_event_envelope",
            "R0",
        ),
        (
            attempt / "batches/science-batch-0000/science-compact/dispatches/"
            "ticks-000000000000-000000000064/sparse_lifecycle_events_v3.bin.gz",
            "compact-dispatch",
            "cuda.compact_science_dispatch",
            "R3",
        ),
        (
            attempt / "batches/science-batch-0000/checkpoints/"
            "checkpoint-tick-000000000064/canonical_state.bin",
            "checkpoint-state",
            "checkpoint.cuda_native_v2",
            "R4",
        ),
        (
            attempt / "control/provider_receipt.json",
            "provider-control",
            "lifecycle.control",
            "R5",
        ),
    )
    inventory: list[dict[str, Any]] = []
    for path, payload, family, retention_class in objects:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        inventory.append(
            build_sealed_artifact_inventory_row_v1(
                run_root=attempt,
                path=path,
                artifact_family=family,
                retention_class=retention_class,
                owner_writer="fixture-child-provider",
                release_predicate="fixture exact lifecycle predicate",
            )
        )
    child_plan = _base_plan()
    child_plan["campaign_id"] = "fixture-child-campaign"
    child_plan["output"] = {"runtime_root": str(attempt)}
    child_plan.pop("plan_sha256", None)
    child_plan["plan_sha256"] = canonical_sha256(child_plan).hex
    manifest = build_artifact_disposition_manifest_v1(
        child_plan,
        run_root=attempt,
        terminal_status="completed",
        live_reference_status="none",
        audit_hold_status="active",
        qualification_hold_status="not_required",
        artifact_inventory=inventory,
        inventory_complete=inventory_complete,
    )
    manifest_path = attempt / "artifact_disposition_manifest.json"
    _write_json(manifest_path, manifest)
    produced = int(manifest["summary"]["produced_bytes"])
    removed: list[dict[str, Any]] = []
    released = 0
    if release_first:
        released_artifact = manifest["artifacts"][0]
        released = int(released_artifact["bytes"])
        removed = [
            {
                "path": released_artifact["path"],
                "bytes": released,
                "category": "event_jsonl",
                "retention_class": released_artifact["retention_class"],
                "checkpoint_role": released_artifact["checkpoint_role"],
                "release_predicate": released_artifact["release_predicate"],
                "artifact_type": released_artifact["artifact_type"],
                "sha256": released_artifact["sha256"],
            }
        ]
        (attempt / released_artifact["path"]).unlink()
    storage = build_storage_disposition_receipt_v1(
        manifest=manifest,
        bytes_before=produced,
        bytes_retained=produced - released,
        bytes_released=released,
        bytes_unresolved_rx=0,
        removed=removed,
        authorization_reason="fixture dry run",
        plan_sha256=child_plan["plan_sha256"],
        result=("release_applied_archive_pending" if removed else "dry_run"),
    )
    storage_path = attempt / "storage_disposition_receipt.json"
    _write_json(storage_path, storage)
    result_core = {"schema_version": "fixture_child_result_v1", "ok": True}
    result = {**result_core, "result_sha256": canonical_sha256(result_core).hex}
    result_path = attempt / "execution_result.json"
    _write_json(result_path, result)
    marker_core = {
        "schema_version": "eiel_execution_terminal_lifecycle_commit_v1",
        "plan_sha256": child_plan["plan_sha256"],
        "request_sha256": "1" * 64,
        "source_identity_sha256": "2" * 64,
        "result": {
            "path": result_path.name,
            "file_sha256": _file_sha256(result_path),
            "result_sha256": result["result_sha256"],
        },
        "manifest": {
            "path": manifest_path.name,
            "file_sha256": _file_sha256(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "storage_receipt": {
            "path": storage_path.name,
            "file_sha256": _file_sha256(storage_path),
            "receipt_sha256": storage["receipt_sha256"],
        },
        "terminal_artifacts": [],
        "committed_at_utc": "2026-08-28T00:00:00+00:00",
    }
    marker = {**marker_core, "commit_sha256": canonical_sha256(marker_core).hex}
    marker_path = attempt / "terminal_lifecycle_commit.json"
    _write_json(marker_path, marker)
    child_lifecycle = {
        "terminal_lifecycle_commit_path": str(marker_path),
        "terminal_lifecycle_commit_file_sha256": _file_sha256(marker_path),
        "terminal_lifecycle_commit_sha256": marker["commit_sha256"],
        "artifact_disposition_manifest_path": str(manifest_path),
        "artifact_disposition_manifest_sha256": manifest["manifest_sha256"],
        "storage_disposition_receipt_path": str(storage_path),
        "storage_disposition_receipt_sha256": storage["receipt_sha256"],
        "storage_receipt_successor_after_terminal_commit": False,
    }
    journal = {
        "schema_version": front_door.SCIENCE_REQUEST_V1_RUN_JOURNAL_SCHEMA_VERSION,
        "identity": {"require_launchpad_lifecycle": True},
        "status": "COMPLETED",
        "batches": [
            {
                "batch_index": 0,
                "world_ids": ["world-fixture"],
                "state": "COMPLETED",
                "attempt": 1,
                "attempt_root": "batches/batch-0000/attempt-0001",
                "result": None,
                "child_lifecycle": child_lifecycle,
                "wait": None,
            }
        ],
        "aggregate": None,
        "last_event": "completed_receipt_sealed",
    }
    front_door._write_science_run_journal_v1(root, journal)
    return attempt, child_lifecycle


def _reseal_child_lifecycle_fixture(
    root: Path,
    attempt: Path,
    *,
    manifest: dict[str, Any],
    storage: dict[str, Any],
) -> None:
    manifest_path = attempt / "artifact_disposition_manifest.json"
    storage_path = attempt / "storage_disposition_receipt.json"
    marker_path = attempt / "terminal_lifecycle_commit.json"
    _write_json(manifest_path, manifest)
    _write_json(storage_path, storage)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_core = {
        key: value for key, value in marker.items() if key != "commit_sha256"
    }
    marker_core["manifest"] = {
        "path": manifest_path.name,
        "file_sha256": _file_sha256(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
    }
    marker_core["storage_receipt"] = {
        "path": storage_path.name,
        "file_sha256": _file_sha256(storage_path),
        "receipt_sha256": storage["receipt_sha256"],
    }
    marker = {**marker_core, "commit_sha256": canonical_sha256(marker_core).hex}
    _write_json(marker_path, marker)
    journal = front_door._read_science_run_journal_v1(root)
    lifecycle = journal["batches"][0]["child_lifecycle"]
    lifecycle.update(
        {
            "terminal_lifecycle_commit_file_sha256": _file_sha256(marker_path),
            "terminal_lifecycle_commit_sha256": marker["commit_sha256"],
            "artifact_disposition_manifest_sha256": manifest["manifest_sha256"],
            "storage_disposition_receipt_sha256": storage["receipt_sha256"],
            "storage_receipt_successor_after_terminal_commit": False,
        }
    )
    front_door._write_science_run_journal_v1(root, journal)


def _apply_child_storage_successor_fixture(
    root: Path,
    attempt: Path,
    *,
    journal_binding: str,
) -> dict[str, Any]:
    manifest = json.loads(
        (attempt / "artifact_disposition_manifest.json").read_text(encoding="utf-8")
    )
    released = manifest["artifacts"][0]
    removed = {
        "path": released["path"],
        "bytes": released["bytes"],
        "category": "event_jsonl",
        "retention_class": released["retention_class"],
        "checkpoint_role": released["checkpoint_role"],
        "release_predicate": released["release_predicate"],
        "artifact_type": released["artifact_type"],
        "sha256": released["sha256"],
    }
    (attempt / released["path"]).unlink()
    produced = int(manifest["summary"]["produced_bytes"])
    storage = build_storage_disposition_receipt_v1(
        manifest=manifest,
        bytes_before=produced,
        bytes_retained=produced - int(released["bytes"]),
        bytes_released=int(released["bytes"]),
        bytes_unresolved_rx=0,
        removed=[removed],
        authorization_reason="fixture successor release",
        plan_sha256=manifest["plan_sha256"],
        result="release_applied_archive_pending",
    )
    _write_json(attempt / "storage_disposition_receipt.json", storage)
    if journal_binding != "marker":
        journal = front_door._read_science_run_journal_v1(root)
        lifecycle = journal["batches"][0]["child_lifecycle"]
        lifecycle["storage_disposition_receipt_sha256"] = storage["receipt_sha256"]
        lifecycle["storage_receipt_successor_after_terminal_commit"] = (
            journal_binding == "successor"
        )
        front_door._write_science_run_journal_v1(root, journal)
    return storage


def test_public_validation_compiles_both_supplied_examples() -> None:
    selection = validate_science_request_v1(_W32)
    mutation = validate_science_request_v1(_W128)

    assert selection["status"] == "VALID"
    assert selection["logical_world_count"] == 1024
    assert [
        row["world_count"] for row in selection["nominal_batch_plan"]["batches"]
    ] == [1024]
    assert mutation["status"] == "VALID"
    assert mutation["logical_world_count"] == 240
    assert [
        row["world_count"] for row in mutation["nominal_batch_plan"]["batches"]
    ] == [32, 32, 32, 32, 32, 32, 32, 16]
    assert selection["execution_started"] is False
    assert mutation["execution_started"] is False


def test_public_cli_validate_needs_only_the_request(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert science_cli_main(["validate", str(_W32)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "VALID"
    assert result["logical_world_count"] == 1024


def test_canonical_admission_rejects_policy_attribution_when_literal_genotype_attribution_is_required(
) -> None:
    raw = json.loads(_W32.read_text(encoding="utf-8"))
    raw["reduce"]["configuration"].update(
        {
            "require_direct_genotype_reproductive_attribution": True,
            "require_reproduction_funnel_denominators": True,
        }
    )
    authority_called = False

    def forbidden_authority(*args: Any) -> ScienceExecutionAuthorityV1:
        nonlocal authority_called
        authority_called = True
        raise AssertionError("unsupported genotype attribution must fail first")

    with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
        plan_science_request_v1(
            raw,
            repo_root=Path.cwd(),
            authority_resolver=forbidden_authority,
        )

    assert captured.value.code == "INVALID_SCIENCE_REQUEST"
    assert "locus/allele genotype identity" in str(captured.value)
    assert captured.value.details == {
        "request_sha256": validate_eiel_science_request_v1(raw)
        .request_sha256()
        .to_dict(),
        "execution_started": False,
    }
    assert authority_called is False


def test_public_cli_run_w128_reports_exact_pre_execution_capability_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert science_cli_main(["run", str(_W128)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == MISSING_SCIENTIFIC_CAPABILITY
    assert result["result_class"] == MISSING_SCIENTIFIC_CAPABILITY
    assert result["execution_started"] is False


def test_public_cli_plan_w32_self_resolves_without_environment_or_injected_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    authority = _test_authority(tmp_path, monkeypatch)
    installed_path = tmp_path / "installed-science-authority.json"
    activated = activate_science_execution_authority_v1(
        authority.base_campaign_plan_path,
        authority_id="installed-test-w32-b1024-authority-v1",
        output_path=installed_path,
    )
    assert activated["status"] == "SCIENCE_AUTHORITY_ACTIVATED"
    assert activated["authority_path"] == str(installed_path.resolve())
    monkeypatch.delenv(front_door.SCIENCE_AUTHORITY_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(
        front_door,
        "DEFAULT_SCIENCE_AUTHORITY_PATHS",
        (installed_path,),
    )
    monkeypatch.setattr(
        front_door,
        "DEFAULT_SCIENCE_RUNTIME_ROOT",
        tmp_path / "self-resolving-runtime",
    )

    assert science_cli_main(["plan", str(_W32)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ADMITTED_EXECUTABLE_MATERIALIZATION"
    assert result["execution_started"] is False
    assert result["lifecycle_closeout"]["inventory_complete"] is True


def test_authority_activation_appends_idempotently_and_rejects_ambiguity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _test_authority(tmp_path, monkeypatch)
    installed = tmp_path / "bounded-authority-set.json"
    first = activate_science_execution_authority_v1(
        authority.base_campaign_plan_path,
        authority_id="bounded-w32-authority-v1",
        output_path=installed,
        append_to_set=True,
    )
    second = activate_science_execution_authority_v1(
        authority.base_campaign_plan_path,
        authority_id="bounded-w32-authority-v1",
        output_path=installed,
        append_to_set=True,
    )
    payload = json.loads(installed.read_text(encoding="utf-8"))
    assert first["status"] == "SCIENCE_AUTHORITY_SET_MEMBER_ACTIVATED"
    assert second["authority_set_sha256"] == first["authority_set_sha256"]
    assert payload["schema_version"] == "eiel_science_execution_authority_set_v1"
    assert [row["authority_id"] for row in payload["authorities"]] == [
        "bounded-w32-authority-v1"
    ]

    conflicting_plan = json.loads(authority.base_campaign_plan_path.read_bytes())
    conflicting_plan["campaign_id"] = "same-match-different-plan"
    conflicting_plan.pop("plan_sha256")
    conflicting_plan["plan_sha256"] = canonical_sha256(conflicting_plan).hex
    conflicting_path = tmp_path / "conflicting-base-plan.json"
    conflicting_path.write_text(
        json.dumps(conflicting_plan, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(
        ScienceRequestV1FrontDoorError,
        match="two distinct exact matches",
    ):
        activate_science_execution_authority_v1(
            conflicting_path,
            authority_id="ambiguous-w32-authority-v2",
            output_path=installed,
            append_to_set=True,
        )


def test_bulk_authority_set_activation_is_single_pass_and_preserves_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _test_authority(tmp_path, monkeypatch)
    second_plan = _base_plan()
    second_checkpoints = (25_000, 50_000, 75_000, 100_000, 125_000)
    second_plan["campaign_id"] = "science-request-v1-second-target"
    second_plan["artifact_lifecycle"] = lifecycle_profile_contract_v1(
        "SCIENCE_COMPACT",
        total_max_bytes=512 * 1024**2,
        checkpoint_schedule=second_checkpoints,
        checkpoint_authority="absolute_tick",
    )
    second_plan["execution"]["total_ticks"] = 125_000
    second_plan["execution"]["checkpoint_ticks"] = list(second_checkpoints)
    second_plan["execution"]["tick_native_control"]["schedules"] = {
        "checkpoint_ticks": list(second_checkpoints),
        "report_ticks": [125_000],
        "science_sample_ticks": [125_000],
        "reducer_ticks": [125_000],
        "archive_ticks": list(second_checkpoints),
    }
    second_plan.pop("plan_sha256")
    second_plan["plan_sha256"] = canonical_sha256(second_plan).hex
    second_path = tmp_path / "base_plan_second.json"
    second_path.write_text(json.dumps(second_plan, sort_keys=True), encoding="utf-8")
    installed = tmp_path / "bulk-authority-set.json"

    result = front_door.activate_science_execution_authority_set_v1(
        (
            ("bounded-w32-t125000-v1", second_path),
        ),
        output_path=installed,
        preserve_from=(authority.authority_source,),
    )
    payload = json.loads(installed.read_text(encoding="utf-8"))

    assert result["status"] == "SCIENCE_AUTHORITY_SET_ACTIVATED"
    assert result["authority_count"] == 2
    assert result["maximum_authority_count"] == 4096
    assert [row["authority_id"] for row in payload["authorities"]] == [
        "bounded-w32-t125000-v1",
        "test-w32-b1024-authority-v1",
    ]
    assert payload["authority_set_sha256"] == result["authority_set_sha256"]


def test_authority_set_reader_rejects_more_than_explicit_member_limit(
    tmp_path: Path,
) -> None:
    set_core = {
        "schema_version": "eiel_science_execution_authority_set_v1",
        "authorities": [
            {} for _ in range(front_door.MAX_SCIENCE_EXECUTION_AUTHORITIES_V1 + 1)
        ],
    }
    authority_path = tmp_path / "oversized-authority-set.json"
    authority_path.write_text(
        json.dumps(
            {
                **set_core,
                "authority_set_sha256": canonical_sha256(set_core).hex,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ScienceRequestV1FrontDoorError,
        match="exceeds its bounded member limit",
    ):
        front_door._authority_rows(authority_path)


def test_public_run_rejects_legacy_manual_authority_flags() -> None:
    with pytest.raises(SystemExit) as captured:
        science_cli_main(["run", str(_W32), "--context", "context.json"])
    assert captured.value.code == 2


def test_w128_plan_fails_at_capability_boundary_before_authority_or_output(
    tmp_path: Path,
) -> None:
    called = False

    def forbidden_resolver(*args: Any) -> ScienceExecutionAuthorityV1:
        nonlocal called
        called = True
        raise AssertionError("authority resolution must not run")

    with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
        plan_science_request_v1(
            _W128,
            repo_root=Path.cwd(),
            materialization_parent=tmp_path / "materializations",
            authority_resolver=forbidden_resolver,
        )

    assert captured.value.code == MISSING_SCIENTIFIC_CAPABILITY
    assert called is False
    assert not (tmp_path / "materializations").exists()


def test_w128_run_fails_before_execution_and_runtime_creation(tmp_path: Path) -> None:
    executed = False

    def forbidden_executor(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal executed
        executed = True
        raise AssertionError("execution must not start")

    with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
        run_science_request_v1(
            _W128,
            repo_root=Path.cwd(),
            materialization_parent=tmp_path / "materializations",
            runtime_parent=tmp_path / "runs",
            batch_executor=forbidden_executor,
        )

    assert captured.value.code == MISSING_SCIENTIFIC_CAPABILITY
    assert captured.value.to_result()["status"] == MISSING_SCIENTIFIC_CAPABILITY
    assert executed is False
    assert not (tmp_path / "runs").exists()


def test_current_authority_resolution_fails_closed_on_two_distinct_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows: list[dict[str, Any]] = []
    for ordinal in (1, 2):
        plan = _base_plan()
        plan["campaign_id"] = f"ambiguous-authority-{ordinal}"
        plan.pop("plan_sha256")
        plan["plan_sha256"] = canonical_sha256(plan).hex
        plan_path = tmp_path / f"base_plan_{ordinal}.json"
        plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
        core = {
            "schema_version": "eiel_science_execution_authority_v1",
            "authority_id": f"ambiguous-w32-authority-{ordinal}",
            "status": "CURRENT",
            "world": {
                "width": 32,
                "height": 32,
                "population": 256,
                "capacity": 512,
            },
            "base_campaign_plan": {
                "path": str(plan_path),
                "file_sha256": _file_sha256(plan_path),
                "plan_sha256": plan["plan_sha256"],
            },
        }
        rows.append({**core, "authority_sha256": canonical_sha256(core).hex})
    set_core = {
        "schema_version": "eiel_science_execution_authority_set_v1",
        "authorities": rows,
    }
    authority_path = tmp_path / "authorities.json"
    authority_path.write_text(
        json.dumps(
            {
                **set_core,
                "authority_set_sha256": canonical_sha256(set_core).hex,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        front_door.SCIENCE_AUTHORITY_ENVIRONMENT_VARIABLE, str(authority_path)
    )
    request = validate_eiel_science_request_v1(json.loads(_W32.read_text()))
    compilation = compile_science_request_v1(request)

    with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
        resolve_current_science_authority_v1(request, compilation, Path.cwd())

    assert captured.value.code == AMBIGUOUS_SCIENCE_AUTHORITY
    assert len(captured.value.details["matching_authority_sha256s"]) == 2


def test_authority_resolution_partitions_nominal_batch_to_sealed_physical_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nominal request batch is not a frozen physical carrier claim."""

    plan = _base_plan()
    plan_path = tmp_path / "base_plan_b16.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    authority_core = {
        "schema_version": "eiel_science_execution_authority_v1",
        "authority_id": "test-w32-b16-authority-v1",
        "status": "CURRENT",
        "world": {
            "width": 32,
            "height": 32,
            "population": 256,
            "capacity": 512,
        },
        "base_campaign_plan": {
            "path": str(plan_path.resolve()),
            "file_sha256": _file_sha256(plan_path),
            "plan_sha256": plan["plan_sha256"],
        },
    }
    authority_row = {
        **authority_core,
        "authority_sha256": canonical_sha256(authority_core).hex,
    }
    authority_path = tmp_path / "science_authority_b16.json"
    authority_path.write_text(
        json.dumps(authority_row, sort_keys=True), encoding="utf-8"
    )
    monkeypatch.setattr(
        front_door, "_default_authority_sources", lambda: (authority_path,)
    )

    calls: list[int] = []

    class _FixedControls:
        def to_value(self) -> dict[str, Any]:
            return {"cuda_native.temporal_k": 64}

    def fake_context(
        *_args: Any, logical_world_capacity: int, **_kwargs: Any
    ) -> Any:
        calls.append(logical_world_capacity)
        return SimpleNamespace(
            route_adapter=SimpleNamespace(
                maximum_batch_cells=16,
                fixed_flat_controls=_FixedControls(),
            )
        )

    monkeypatch.setattr(
        front_door, "_science_authority_host_context_v1", fake_context
    )
    monkeypatch.setattr(
        front_door,
        "serialize_science_host_context_v1",
        lambda _context: {
            "context_sha256": {"algorithm": "sha256", "hex": "a" * 64}
        },
    )

    request = validate_eiel_science_request_v1(json.loads(_W32.read_text()))
    compilation = compile_science_request_v1(request)
    resolved = resolve_current_science_authority_v1(
        request, compilation, repo_root=Path.cwd()
    )

    assert calls == [1, 16]
    assert resolved.available_physical_batch_capacity == 16
    assert resolved.physical_batch_capacity == 16


def test_materialized_ownership_accepts_physical_partition_of_nominal_batch(
    tmp_path: Path,
) -> None:
    request_path = _small_w32_request(tmp_path)
    value = json.loads(request_path.read_text(encoding="utf-8"))
    value["run"]["batch"] = 8
    request = validate_eiel_science_request_v1(value)
    compilation = compile_science_request_v1(request)
    world_ids = [world.world_id for world in compilation.world_bundle.worlds]
    lowered = [
        {
            "science_materialization": {
                "semantic_world_ids": world_ids[offset : offset + 3]
            }
        }
        for offset in range(0, len(world_ids), 3)
    ]

    observed = front_door._verify_materialized_batch_ownership_v1(
        compilation=compilation,
        materialization=tuple(lowered),
    )

    assert [list(row) for row in observed] == [
        world_ids[0:3],
        world_ids[3:6],
        world_ids[6:8],
    ]


def test_w32_plan_reaches_admitted_executable_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _test_authority(tmp_path, monkeypatch)

    result = plan_science_request_v1(
        _W32,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        authority_resolver=lambda *args: authority,
    )

    assert result["ok"] is True
    assert result["status"] == "ADMITTED_EXECUTABLE_MATERIALIZATION"
    assert result["materialization"]["bundle_admission"]["outcome"] == "ADMITTED"
    assert result["materialization"]["batch_count"] == 1
    assert Path(result["materialization"]["root"]).is_dir()
    assert result["execution_started"] is False
    assert result["qualification_authority_effect"] is False
    lifecycle = result["lifecycle_closeout"]
    assert lifecycle["inventory_complete"] is True
    assert Path(lifecycle["run_passport_path"]).is_file()
    assert Path(lifecycle["artifact_disposition_manifest_path"]).is_file()
    assert Path(lifecycle["storage_disposition_receipt_path"]).is_file()

    reused = plan_science_request_v1(
        _W32,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        authority_resolver=lambda *args: authority,
    )
    assert reused["reused"] is True
    assert reused["receipt_sha256"] == result["receipt_sha256"]


def test_outer_inventory_inherits_exact_authenticated_child_dispositions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    _, child_lifecycle = _authenticated_child_inventory_fixture(root)
    rows = front_door._collect_science_lifecycle_inventory_v1(root)
    by_path = {row["path"]: row for row in rows}
    prefix = "batches/batch-0000/attempt-0001/"

    assert (
        by_path[
            prefix + "batches/science-batch-0000/scratch/events/"
            "dispatch-tick-000000000000-000000000064/direct-sparse-events-v3.bin"
        ]["retention_class"]
        == "R0"
    )
    assert (
        by_path[
            prefix + "batches/science-batch-0000/science-compact/dispatches/"
            "ticks-000000000000-000000000064/sparse_lifecycle_events_v3.bin.gz"
        ]["retention_class"]
        == "R3"
    )
    assert (
        by_path[
            prefix + "batches/science-batch-0000/checkpoints/"
            "checkpoint-tick-000000000064/canonical_state.bin"
        ]["retention_class"]
        == "R4"
    )
    assert by_path[prefix + "control/provider_receipt.json"]["retention_class"] == "R5"
    assert by_path[prefix + "terminal_lifecycle_commit.json"]["retention_class"] == "R5"
    assert child_lifecycle["artifact_disposition_manifest_sha256"]


def test_outer_inventory_accepts_only_exact_receipt_proven_child_release(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    _, child_lifecycle = _authenticated_child_inventory_fixture(
        root,
        release_first=True,
    )

    rows = front_door._collect_science_lifecycle_inventory_v1(root)
    paths = {row["path"] for row in rows}
    prefix = "batches/batch-0000/attempt-0001/"

    assert not any("direct-sparse-events-v3.bin" in path for path in paths)
    assert prefix + "storage_disposition_receipt.json" in paths
    assert prefix + "artifact_disposition_manifest.json" in paths
    assert child_lifecycle["storage_receipt_successor_after_terminal_commit"] is False


def test_outer_inventory_rejects_authenticated_child_release_identity_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    attempt, _ = _authenticated_child_inventory_fixture(root, release_first=True)
    storage_path = attempt / "storage_disposition_receipt.json"
    storage = json.loads(storage_path.read_text(encoding="utf-8"))
    storage["removed"][0]["sha256"] = "0" * 64
    storage_core = {
        key: value for key, value in storage.items() if key != "receipt_sha256"
    }
    storage["receipt_sha256"] = canonical_sha256(storage_core).hex
    _write_json(storage_path, storage)
    journal = front_door._read_science_run_journal_v1(root)
    journal["batches"][0]["child_lifecycle"]["storage_disposition_receipt_sha256"] = (
        storage["receipt_sha256"]
    )
    journal["batches"][0]["child_lifecycle"][
        "storage_receipt_successor_after_terminal_commit"
    ] = True
    front_door._write_science_run_journal_v1(root, journal)

    with pytest.raises(
        ScienceRequestV1FrontDoorError,
        match="release row 0 differs",
    ):
        front_door._collect_science_lifecycle_inventory_v1(root)


def test_outer_inventory_rejects_missing_retained_child_row(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    attempt, _ = _authenticated_child_inventory_fixture(root)
    (
        attempt / "batches/science-batch-0000/science-compact/dispatches/"
        "ticks-000000000000-000000000064/sparse_lifecycle_events_v3.bin.gz"
    ).unlink()

    with pytest.raises(
        ScienceRequestV1FrontDoorError,
        match="no longer resolves exactly",
    ):
        front_door._collect_science_lifecycle_inventory_v1(root)


def test_outer_inventory_rejects_changed_retained_child_contents(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    attempt, _ = _authenticated_child_inventory_fixture(root)
    retained = (
        attempt / "batches/science-batch-0000/science-compact/dispatches/"
        "ticks-000000000000-000000000064/sparse_lifecycle_events_v3.bin.gz"
    )
    retained.write_text("tampered-compact", encoding="utf-8")

    with pytest.raises(
        ScienceRequestV1FrontDoorError,
        match="content identity differs",
    ):
        front_door._collect_science_lifecycle_inventory_v1(root)


@pytest.mark.parametrize("journal_binding", ["marker", "successor"])
def test_outer_inventory_accepts_exact_post_marker_release_successor_states(
    tmp_path: Path,
    journal_binding: str,
) -> None:
    root = tmp_path / "campaign"
    attempt, _ = _authenticated_child_inventory_fixture(root)
    _apply_child_storage_successor_fixture(
        root,
        attempt,
        journal_binding=journal_binding,
    )

    rows = front_door._collect_science_lifecycle_inventory_v1(root)

    assert not any("direct-sparse-events-v3.bin" in row["path"] for row in rows)


def test_outer_inventory_rejects_successor_digest_with_false_successor_flag(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    attempt, _ = _authenticated_child_inventory_fixture(root)
    _apply_child_storage_successor_fixture(
        root,
        attempt,
        journal_binding="successor_digest_false_flag",
    )

    with pytest.raises(
        ScienceRequestV1FrontDoorError,
        match="storage disposition is not authenticated",
    ):
        front_door._collect_science_lifecycle_inventory_v1(root)


@pytest.mark.parametrize(
    "failure",
    [
        "extra_removed",
        "duplicate_removed",
        "still_exists",
        "byte_accounting",
        "action_failure",
        "non_applied_result",
    ],
)
def test_outer_inventory_rejects_invalid_applied_release_successor(
    tmp_path: Path,
    failure: str,
) -> None:
    root = tmp_path / "campaign"
    attempt, _ = _authenticated_child_inventory_fixture(root)
    storage = _apply_child_storage_successor_fixture(
        root,
        attempt,
        journal_binding="successor",
    )
    storage_path = attempt / "storage_disposition_receipt.json"
    if failure == "extra_removed":
        extra = dict(storage["removed"][0])
        extra["path"] = "unmanifested-extra.bin"
        storage["removed"].append(extra)
    elif failure == "duplicate_removed":
        storage["removed"].append(dict(storage["removed"][0]))
    elif failure == "still_exists":
        released_path = attempt / storage["removed"][0]["path"]
        released_path.parent.mkdir(parents=True, exist_ok=True)
        released_path.write_text("raw-envelope", encoding="utf-8")
    elif failure == "byte_accounting":
        storage["bytes_released"] += 1
    elif failure == "action_failure":
        storage["action_failures"] = [{"error": "synthetic failure"}]
    else:
        storage["result"] = "dry_run"
    if failure != "still_exists":
        storage_core = {
            key: value for key, value in storage.items() if key != "receipt_sha256"
        }
        storage["receipt_sha256"] = canonical_sha256(storage_core).hex
        _write_json(storage_path, storage)
        journal = front_door._read_science_run_journal_v1(root)
        journal["batches"][0]["child_lifecycle"][
            "storage_disposition_receipt_sha256"
        ] = storage["receipt_sha256"]
        front_door._write_science_run_journal_v1(root, journal)

    with pytest.raises(
        ScienceRequestV1FrontDoorError,
        match="release row|release receipt|claimed release",
    ):
        front_door._collect_science_lifecycle_inventory_v1(root)


def test_outer_inventory_rejects_marker_receipt_reserialized_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    attempt, _ = _authenticated_child_inventory_fixture(root)
    storage_path = attempt / "storage_disposition_receipt.json"
    storage = json.loads(storage_path.read_text(encoding="utf-8"))
    storage_path.write_text(json.dumps(storage, indent=2), encoding="utf-8")

    with pytest.raises(
        ScienceRequestV1FrontDoorError,
        match="storage disposition is not authenticated",
    ):
        front_door._collect_science_lifecycle_inventory_v1(root)


@pytest.mark.parametrize("path_alias", [False, True])
def test_outer_inventory_rejects_duplicate_or_aliased_manifest_paths(
    tmp_path: Path,
    path_alias: bool,
) -> None:
    root = tmp_path / "campaign"
    attempt, _ = _authenticated_child_inventory_fixture(root)
    manifest_path = attempt / "artifact_disposition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    duplicate = dict(manifest["artifacts"][0])
    duplicate["artifact_id"] = "artifact-duplicate-fixture"
    if path_alias:
        duplicate["path"] = duplicate["path"].replace("/scratch/", "/scratch//")
    manifest["artifacts"].append(duplicate)
    size = int(duplicate["bytes"])
    manifest["summary"]["produced_bytes"] += size
    manifest["summary"]["release_eligible_bytes"] += size
    manifest["summary"]["by_retention_class"]["R0"]["bytes"] += size
    manifest["summary"]["by_retention_class"]["R0"]["objects"] += 1
    manifest_core = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest_core).hex
    produced = int(manifest["summary"]["produced_bytes"])
    storage = build_storage_disposition_receipt_v1(
        manifest=manifest,
        bytes_before=produced,
        bytes_retained=produced,
        bytes_released=0,
        bytes_unresolved_rx=0,
        authorization_reason="duplicate path adversarial fixture",
        plan_sha256=manifest["plan_sha256"],
    )
    _reseal_child_lifecycle_fixture(
        root,
        attempt,
        manifest=manifest,
        storage=storage,
    )

    with pytest.raises(
        ScienceRequestV1FrontDoorError,
        match="duplicates an object path|path is not canonical",
    ):
        front_door._collect_science_lifecycle_inventory_v1(root)


@pytest.mark.parametrize("failure", ["tampered", "incomplete"])
def test_outer_inventory_fails_closed_on_invalid_child_disposition(
    tmp_path: Path,
    failure: str,
) -> None:
    root = tmp_path / "campaign"
    attempt, _ = _authenticated_child_inventory_fixture(
        root,
        inventory_complete=failure != "incomplete",
    )
    if failure == "tampered":
        manifest_path = attempt / "artifact_disposition_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["terminal_status"] = "failed"
        _write_json(manifest_path, manifest)

    with pytest.raises(
        ScienceRequestV1FrontDoorError,
        match="inventory|identity|exact and complete",
    ):
        front_door._collect_science_lifecycle_inventory_v1(root)


def test_w32_run_uses_every_materialized_batch_then_aggregates_and_closes_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _test_authority(tmp_path, monkeypatch)
    observed: list[tuple[int, Path]] = []

    def fake_executor(
        materialization: Path,
        *,
        batch_index: int,
        runtime_root: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        observed.append((batch_index, runtime_root))
        return _synthetic_batch_result(
            materialization,
            batch_index=batch_index,
            runtime_root=runtime_root,
        )

    aggregate_result = {
        "schema_version": "synthetic_campaign_atlas_result_v1",
        "result_sha256": {"algorithm": "sha256", "hex": "6" * 64},
    }
    verification_receipt = {
        "schema_version": "synthetic_campaign_atlas_verification_v1",
        "receipt_sha256": {"algorithm": "sha256", "hex": "7" * 64},
    }

    def fake_aggregate(**kwargs: Any) -> dict[str, Any]:
        assert len(kwargs["batch_results"]) == 1
        assert kwargs["batch_results"][0]["physical_batch_index"] == 0
        return {
            "result": aggregate_result,
            "verification_receipt": verification_receipt,
        }

    monkeypatch.setattr(
        front_door, "aggregate_science_campaign_atlas_v1", fake_aggregate
    )
    result = run_science_request_v1(
        _W32,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        runtime_parent=tmp_path / "runs",
        authority_resolver=lambda *args: authority,
        batch_executor=fake_executor,
    )

    assert result["ok"] is True
    assert result["status"] == "COMPLETED"
    assert result["closeout_completed"] is True
    assert [index for index, _ in observed] == [0]
    assert observed[0][1].parts[-2:] == ("batch-0000", "attempt-0001")
    assert Path(result["runtime_root"]).is_dir()
    assert Path(result["receipt_path"]).is_file()
    assert Path(result["lifecycle_closeout"]["run_passport_path"]).is_file()
    assert Path(
        result["lifecycle_closeout"]["artifact_disposition_manifest_path"]
    ).is_file()
    assert Path(
        result["lifecycle_closeout"]["storage_disposition_receipt_path"]
    ).is_file()
    assert result["qualification_authority_effect"] is False


def test_w32_request_batch_capacity_is_preserved_and_every_batch_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = _small_w32_request(tmp_path)
    authority = replace(
        _test_authority(tmp_path, monkeypatch), physical_batch_capacity=3
    )
    observed: list[int] = []

    def fake_executor(
        materialization: Path,
        *,
        batch_index: int,
        runtime_root: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        observed.append(batch_index)
        return _synthetic_batch_result(
            materialization,
            batch_index=batch_index,
            runtime_root=runtime_root,
        )

    def fake_aggregate(**kwargs: Any) -> dict[str, Any]:
        assert [row["physical_batch_index"] for row in kwargs["batch_results"]] == [
            0,
            1,
            2,
        ]
        return {
            "result": {
                "schema_version": "synthetic_campaign_atlas_result_v1",
                "result_sha256": {"algorithm": "sha256", "hex": "6" * 64},
            },
            "verification_receipt": {
                "schema_version": "synthetic_campaign_atlas_verification_v1",
                "receipt_sha256": {"algorithm": "sha256", "hex": "7" * 64},
            },
        }

    monkeypatch.setattr(
        front_door, "aggregate_science_campaign_atlas_v1", fake_aggregate
    )
    result = run_science_request_v1(
        request_path,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        runtime_parent=tmp_path / "runs",
        authority_resolver=lambda *args: authority,
        batch_executor=fake_executor,
    )

    assert observed == [0, 1, 2]
    assert [
        len(row["materialized_world_ids"]) for row in result["completed_batches"]
    ] == [3, 3, 2]
    assert result["status"] == "COMPLETED"
    lifecycle_plan = json.loads(
        (Path(result["runtime_root"]) / "science_lifecycle_plan.json").read_text(
            encoding="utf-8"
        )
    )
    _, _, _, lowered = load_science_instrument_materialization_v1(
        Path(result["materialization_root"])
    )
    child_storage = [row["artifact_lifecycle"]["storage"] for row in lowered]
    storage = lifecycle_plan["artifact_lifecycle"]["storage"]
    assert storage["scratch_max_bytes"] >= sum(
        row["scratch_max_bytes"] for row in child_storage
    )
    assert storage["checkpoint_max_bytes"] >= sum(
        row["checkpoint_max_bytes"] for row in child_storage
    )
    assert storage["durable_max_bytes"] >= sum(
        row["durable_max_bytes"] for row in child_storage
    ) + front_door._science_coordinator_storage_allowance_v1(
        compile_science_request_v1(
            validate_eiel_science_request_v1(
                json.loads(request_path.read_text(encoding="utf-8"))
            )
        )
    )
    assert (
        lifecycle_plan["execution"]["budgets"]["disk_bytes"]
        == storage["total_max_bytes"]
    )


def test_run_rejects_swapped_batch_identity_and_closes_failed_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = _small_w32_request(tmp_path)
    authority = replace(
        _test_authority(tmp_path, monkeypatch), physical_batch_capacity=3
    )

    def swapped_executor(
        materialization: Path,
        *,
        batch_index: int,
        runtime_root: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        result = _synthetic_batch_result(
            materialization,
            batch_index=batch_index,
            runtime_root=runtime_root,
        )
        result["batch_index"] = batch_index + 1
        result["science_lineage"]["batch_index"] = batch_index + 1
        return result

    with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
        run_science_request_v1(
            request_path,
            repo_root=Path.cwd(),
            materialization_parent=tmp_path / "materializations",
            runtime_parent=tmp_path / "runs",
            authority_resolver=lambda *args: authority,
            batch_executor=swapped_executor,
        )

    assert captured.value.code == front_door.SCIENCE_BATCH_EXECUTION_FAILED
    run_root = next(
        path
        for path in (tmp_path / "runs").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    manifest = json.loads(
        (run_root / "artifact_disposition_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["terminal_status"] == "failed"
    assert manifest["inventory"]["complete"] is True


def test_batch_authentication_rejects_every_published_provenance_identity_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = _small_w32_request(tmp_path)
    authority = replace(
        _test_authority(tmp_path, monkeypatch), physical_batch_capacity=3
    )
    planned = plan_science_request_v1(
        request_path,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        authority_resolver=lambda *args: authority,
    )
    materialization = Path(planned["materialization"]["root"])
    _, receipt, _, lowered = load_science_instrument_materialization_v1(materialization)
    batch_root = tmp_path / "batch-0000"
    for field in (
        "request_sha256",
        "campaign_spec_sha256",
        "compilation_bundle_sha256",
        "bundle_admission_sha256",
    ):
        result = _synthetic_batch_result(
            materialization, batch_index=0, runtime_root=batch_root
        )
        result["science_lineage"][field] = {
            "algorithm": "sha256",
            "hex": "f" * 64,
        }
        with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
            front_door._verify_science_batch_result_v1(
                result=result,
                batch_index=0,
                batch_root=batch_root,
                lowered=lowered[0],
                materialization_receipt=receipt,
                require_launchpad_lifecycle=False,
            )
        assert captured.value.code == front_door.SCIENCE_BATCH_EXECUTION_FAILED

    result = _synthetic_batch_result(
        materialization, batch_index=0, runtime_root=batch_root
    )
    source_field = next(iter(receipt["lineage"]))
    result["source_lineage"][source_field] = {"tampered": True}
    with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
        front_door._verify_science_batch_result_v1(
            result=result,
            batch_index=0,
            batch_root=batch_root,
            lowered=lowered[0],
            materialization_receipt=receipt,
            require_launchpad_lifecycle=False,
        )
    assert captured.value.code == front_door.SCIENCE_BATCH_EXECUTION_FAILED


def test_failed_default_launchpad_result_cannot_bypass_terminal_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = _small_w32_request(tmp_path)
    authority = replace(
        _test_authority(tmp_path, monkeypatch), physical_batch_capacity=3
    )
    planned = plan_science_request_v1(
        request_path,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        authority_resolver=lambda *args: authority,
    )
    materialization = Path(planned["materialization"]["root"])
    _, receipt, _, lowered = load_science_instrument_materialization_v1(materialization)
    batch_root = tmp_path / "batch-0000"
    result = _synthetic_batch_result(
        materialization, batch_index=0, runtime_root=batch_root
    )
    result.update(
        {
            "ok": False,
            "status": "failed",
            "result_class": "PREFLIGHT_ERROR",
            "front_door_result": {
                "ok": False,
                "status": "failed",
                "result_class": "PREFLIGHT_ERROR",
                "warnings": [
                    "CampaignExecutionError: provider failed before terminal closeout"
                ],
            },
        }
    )

    with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
        front_door._verify_science_batch_result_v1(
            result=result,
            batch_index=0,
            batch_root=batch_root,
            lowered=lowered[0],
            materialization_receipt=receipt,
            require_launchpad_lifecycle=True,
        )

    assert captured.value.code == front_door.SCIENCE_BATCH_EXECUTION_FAILED
    assert "authenticated accepted Launchpad terminal lifecycle" in str(captured.value)
    assert "PREFLIGHT_ERROR" in str(captured.value)
    assert "provider failed before terminal closeout" in str(captured.value)
    assert captured.value.details["launchpad_result_class"] == "PREFLIGHT_ERROR"


def test_base_plan_replacement_fails_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _test_authority(tmp_path, monkeypatch)
    plan = json.loads(authority.base_campaign_plan_path.read_text(encoding="utf-8"))
    plan["description"] = "replacement after authority binding"
    plan.pop("plan_sha256")
    plan["plan_sha256"] = canonical_sha256(plan).hex
    authority.base_campaign_plan_path.write_text(
        json.dumps(plan, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
        plan_science_request_v1(
            _W32,
            repo_root=Path.cwd(),
            materialization_parent=tmp_path / "materializations",
            authority_resolver=lambda *args: authority,
        )

    assert captured.value.code == front_door.INVALID_SCIENCE_AUTHORITY
    assert not (tmp_path / "materializations").exists()


def test_reuse_rejects_tampered_materialized_base_plan_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _test_authority(tmp_path, monkeypatch)
    result = plan_science_request_v1(
        _W32,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        authority_resolver=lambda *args: authority,
    )
    snapshot = Path(result["resolved_inputs"]["base_campaign_plan"]["path"])
    snapshot.chmod(0o644)
    snapshot.write_text("{}", encoding="utf-8")

    with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
        plan_science_request_v1(
            _W32,
            repo_root=Path.cwd(),
            materialization_parent=tmp_path / "materializations",
            authority_resolver=lambda *args: authority,
        )

    assert captured.value.code == front_door.SCIENCE_MATERIALIZATION_FAILED


def _synthetic_aggregate(**kwargs: Any) -> dict[str, Any]:
    assert [row["physical_batch_index"] for row in kwargs["batch_results"]] == list(
        range(len(kwargs["batch_results"]))
    )
    return {
        "result": {
            "schema_version": "synthetic_campaign_atlas_result_v1",
            "result_sha256": {"algorithm": "sha256", "hex": "6" * 64},
        },
        "verification_receipt": {
            "schema_version": "synthetic_campaign_atlas_verification_v1",
            "receipt_sha256": {"algorithm": "sha256", "hex": "7" * 64},
        },
    }


def test_capability_resolution_uses_carrier_registry_not_a_geometry_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = validate_eiel_science_request_v1(json.loads(_W32.read_text()))
    normalized = request
    import eiel_composition.science_cuda_v2_carrier_v1 as carrier

    profiles = carrier.SCIENCE_CUDA_V2_PROFILES
    monkeypatch.setattr(carrier, "SCIENCE_CUDA_V2_PROFILES", tuple())
    with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
        front_door._check_scientific_capability(normalized)
    assert captured.value.code == MISSING_SCIENTIFIC_CAPABILITY
    monkeypatch.setattr(carrier, "SCIENCE_CUDA_V2_PROFILES", profiles)
    front_door._check_scientific_capability(normalized)


def test_authority_registry_data_changes_selection_without_source_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _test_authority(tmp_path, monkeypatch)
    authority_path = Path(authority.authority_source)
    row = json.loads(authority_path.read_text(encoding="utf-8"))
    row["status"] = "RETIRED"
    row_core = {key: value for key, value in row.items() if key != "authority_sha256"}
    row["authority_sha256"] = canonical_sha256(row_core).hex
    authority_path.write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
    monkeypatch.setenv(
        front_door.SCIENCE_AUTHORITY_ENVIRONMENT_VARIABLE, str(authority_path)
    )
    request = validate_eiel_science_request_v1(json.loads(_W32.read_text()))
    compilation = compile_science_request_v1(request)
    with pytest.raises(ScienceRequestV1FrontDoorError) as retired:
        resolve_current_science_authority_v1(request, compilation, Path.cwd())
    assert retired.value.code == front_door.UNRESOLVED_SCIENCE_AUTHORITY

    row["status"] = "CURRENT"
    row_core = {key: value for key, value in row.items() if key != "authority_sha256"}
    row["authority_sha256"] = canonical_sha256(row_core).hex
    authority_path.write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
    selected = resolve_current_science_authority_v1(request, compilation, Path.cwd())
    assert selected.authority_id == authority.authority_id


@pytest.mark.parametrize(
    "wait_code", [front_door.WAIT_RESOURCE, front_door.WAIT_STORAGE]
)
def test_run_resume_skips_authenticated_batches_and_keeps_wait_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wait_code: str,
) -> None:
    request_path = _small_w32_request(tmp_path)
    authority = replace(
        _test_authority(tmp_path, monkeypatch), physical_batch_capacity=3
    )
    calls: list[int] = []
    wait_once = True

    def executor(
        materialization: Path,
        *,
        batch_index: int,
        runtime_root: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal wait_once
        del kwargs
        calls.append(batch_index)
        if batch_index == 1 and wait_once:
            wait_once = False
            raise ScienceRequestV1FrontDoorError(
                wait_code,
                f"test {wait_code.lower()} wait",
            )
        return _synthetic_batch_result(
            materialization,
            batch_index=batch_index,
            runtime_root=runtime_root,
        )

    monkeypatch.setattr(
        front_door, "aggregate_science_campaign_atlas_v1", _synthetic_aggregate
    )
    first = run_science_request_v1(
        request_path,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        runtime_parent=tmp_path / "runs",
        authority_resolver=lambda *args: authority,
        batch_executor=executor,
    )
    assert first["status"] == wait_code
    assert calls == [0, 1]
    journal = json.loads((Path(first["runtime_root"]) / "run_journal.json").read_text())
    assert journal["status"] == wait_code
    assert journal["batches"][1]["attempt"] == 1

    resumed = run_science_request_v1(
        request_path,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        runtime_parent=tmp_path / "runs",
        resume_root=first["runtime_root"],
        authority_resolver=lambda *args: authority,
        batch_executor=executor,
    )
    assert resumed["status"] == "COMPLETED"
    assert calls == [0, 1, 1, 2]
    assert (
        Path(resumed["runtime_root"])
        .joinpath("batches", "batch-0001", "attempt-0001", "batch_result.json")
        .is_file()
    )


def test_scheduler_admission_rejection_resumes_as_a_fresh_batch_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CRCA precheck rejection must not replay earlier terminal batches."""

    request_path = _small_w32_request(tmp_path)
    authority = replace(
        _test_authority(tmp_path, monkeypatch), physical_batch_capacity=3
    )
    calls: list[tuple[int, Path]] = []
    reject_once = True

    def executor(
        materialization: Path,
        *,
        batch_index: int,
        runtime_root: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal reject_once
        del kwargs
        calls.append((batch_index, runtime_root))
        if batch_index == 1 and reject_once:
            reject_once = False
            rejected = {
                "schema_version": "eiel_execution_result_v2",
                "ok": False,
                "status": "precheck_failed",
                "result_class": "ATTEMPT_ADMISSION_UNAVAILABLE",
                "warnings": [
                    "CRCA_SCHEDULER_ATTEMPT_REJECTED:single-worker execution lock"
                ],
                "crca_attempt_authority": {
                    "execution_authority_granted": False,
                    "reason_codes": [
                        "CRCA_SCHEDULER_ATTEMPT_REJECTED:single-worker execution lock"
                    ],
                },
            }
            runtime_root.mkdir(parents=True, exist_ok=True)
            _write_json(runtime_root / "execution_result.json", rejected)
            return rejected
        return _synthetic_batch_result(
            materialization,
            batch_index=batch_index,
            runtime_root=runtime_root,
        )

    monkeypatch.setattr(
        front_door, "aggregate_science_campaign_atlas_v1", _synthetic_aggregate
    )
    first = run_science_request_v1(
        request_path,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        runtime_parent=tmp_path / "runs",
        authority_resolver=lambda *args: authority,
        batch_executor=executor,
    )
    assert first["status"] == front_door.WAIT_RESOURCE
    root = Path(first["runtime_root"])
    first_journal = front_door._read_science_run_journal_v1(root)
    assert first_journal["batches"][0]["state"] == "COMPLETED"
    assert first_journal["batches"][1]["state"] == front_door.WAIT_RESOURCE
    assert front_door._wait_status_from_result_v1(
        {
            "status": "precheck_failed",
            "result_class": "ATTEMPT_ADMISSION_UNAVAILABLE",
        }
    ) == front_door.WAIT_RESOURCE

    resumed = run_science_request_v1(
        request_path,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        runtime_parent=tmp_path / "runs",
        resume_root=root,
        authority_resolver=lambda *args: authority,
        batch_executor=executor,
    )
    assert resumed["status"] == "COMPLETED"
    assert [index for index, _ in calls] == [0, 1, 1, 2]
    assert calls[1][1].name == "attempt-0001"
    assert calls[2][1].name == "attempt-0002"
    assert (calls[1][1] / "execution_result.json").is_file()
    assert (calls[2][1] / "batch_result.json").is_file()
    journal = front_door._read_science_run_journal_v1(root)
    retry_row = journal["batches"][1]
    assert retry_row["state"] == "COMPLETED"
    assert retry_row["attempt"] == 2
    assert retry_row["retry_history"][0]["attempt_root"].endswith(
        "batch-0001/attempt-0001"
    )


@pytest.mark.parametrize(
    ("warning", "expected"),
    [
        (
            "CRCA_SCHEDULER_ATTEMPT_REJECTED:single-worker execution lock",
            True,
        ),
        ("CRCA_SCHEDULER_ATTEMPT_REJECTED:exclusive_gpu_conflict", True),
        (
            "CRCA_SCHEDULER_ATTEMPT_REJECTED:canonical continuation lineage already has a published terminal",
            False,
        ),
        ("CRCA_SCHEDULER_ATTEMPT_REJECTED:synthetic-owner-conflict", False),
    ],
)
def test_retryable_admission_warning_requires_authenticated_resource_marker(
    warning: str, expected: bool
) -> None:
    assert front_door._retryable_admission_warning_v1(warning) is expected


def test_resume_finishes_outer_closeout_without_reexecuting_completed_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _test_authority(tmp_path, monkeypatch)
    calls = 0

    def executor(
        materialization: Path,
        *,
        batch_index: int,
        runtime_root: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal calls
        del kwargs
        calls += 1
        return _synthetic_batch_result(
            materialization,
            batch_index=batch_index,
            runtime_root=runtime_root,
        )

    monkeypatch.setattr(
        front_door,
        "aggregate_science_campaign_atlas_v1",
        _synthetic_aggregate,
    )
    publish = front_door._publish_science_lifecycle_closeout_v1
    fail_run_closeout = True

    def interrupted_closeout(**kwargs: Any) -> dict[str, Any]:
        nonlocal fail_run_closeout
        if kwargs["plan"].get("phase") == "run" and fail_run_closeout:
            fail_run_closeout = False
            raise OSError("synthetic interruption after completed receipt")
        return publish(**kwargs)

    monkeypatch.setattr(
        front_door,
        "_publish_science_lifecycle_closeout_v1",
        interrupted_closeout,
    )
    with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
        run_science_request_v1(
            _W32,
            repo_root=Path.cwd(),
            materialization_parent=tmp_path / "materializations",
            runtime_parent=tmp_path / "runs",
            authority_resolver=lambda *args: authority,
            batch_executor=executor,
        )
    root = Path(captured.value.details["runtime_root"])
    journal = json.loads((root / "run_journal.json").read_text(encoding="utf-8"))
    assert journal["status"] == "COMPLETED"
    assert not (root / "artifact_disposition_manifest.json").exists()
    assert calls == 1

    resumed = run_science_request_v1(
        _W32,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        runtime_parent=tmp_path / "runs",
        resume_root=root,
        authority_resolver=lambda *args: authority,
        batch_executor=lambda *args, **kwargs: pytest.fail(
            "completed child simulation was reexecuted"
        ),
    )
    assert resumed["status"] == "COMPLETED"
    assert calls == 1
    assert (root / "artifact_disposition_manifest.json").is_file()
    assert (root / "storage_disposition_receipt.json").is_file()


def test_resume_preserves_precloseout_receipt_across_child_storage_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _test_authority(tmp_path, monkeypatch)
    calls = 0
    successor = False

    def executor(
        materialization: Path,
        *,
        batch_index: int,
        runtime_root: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal calls
        del kwargs
        calls += 1
        result = _synthetic_batch_result(
            materialization,
            batch_index=batch_index,
            runtime_root=runtime_root,
        )
        result["front_door_result"] = {"synthetic": True}
        return result

    def verified_child_lifecycle(**kwargs: Any) -> dict[str, Any]:
        batch_root = kwargs["batch_root"]
        return {
            "terminal_lifecycle_commit_path": str(
                batch_root / "terminal_lifecycle_commit.json"
            ),
            "terminal_lifecycle_commit_file_sha256": "1" * 64,
            "terminal_lifecycle_commit_sha256": "2" * 64,
            "artifact_disposition_manifest_path": str(
                batch_root / "artifact_disposition_manifest.json"
            ),
            "artifact_disposition_manifest_sha256": "3" * 64,
            "storage_disposition_receipt_path": str(
                batch_root / "storage_disposition_receipt.json"
            ),
            "storage_disposition_receipt_sha256": ("5" if successor else "4") * 64,
            "storage_receipt_successor_after_terminal_commit": successor,
        }

    monkeypatch.setattr(
        front_door,
        "execute_materialized_science_batch_v1",
        executor,
    )
    monkeypatch.setattr(
        front_door,
        "_verify_child_launchpad_lifecycle_v1",
        verified_child_lifecycle,
    )
    monkeypatch.setattr(
        front_door,
        "aggregate_science_campaign_atlas_v1",
        _synthetic_aggregate,
    )
    publish = front_door._publish_science_lifecycle_closeout_v1
    interrupt_run_closeout = True

    def interrupted_closeout(**kwargs: Any) -> dict[str, Any]:
        nonlocal interrupt_run_closeout
        if kwargs["plan"].get("phase") != "run":
            return publish(**kwargs)
        if interrupt_run_closeout:
            interrupt_run_closeout = False
            raise OSError("synthetic pre-closeout interruption")
        return {"inventory_complete": True, "synthetic_closeout": True}

    monkeypatch.setattr(
        front_door,
        "_publish_science_lifecycle_closeout_v1",
        interrupted_closeout,
    )
    with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
        run_science_request_v1(
            _W32,
            repo_root=Path.cwd(),
            materialization_parent=tmp_path / "materializations",
            runtime_parent=tmp_path / "runs",
            authority_resolver=lambda *args: authority,
        )
    root = Path(captured.value.details["runtime_root"])
    receipt_path = root / "science_campaign_run_receipt.json"
    original_receipt = receipt_path.read_bytes()
    assert calls == 1

    successor = True
    resumed = run_science_request_v1(
        _W32,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        runtime_parent=tmp_path / "runs",
        resume_root=root,
        authority_resolver=lambda *args: authority,
    )

    assert resumed["status"] == "COMPLETED"
    assert resumed["lifecycle_closeout"]["synthetic_closeout"] is True
    assert receipt_path.read_bytes() == original_receipt
    assert (
        resumed["completed_batches"][0]["child_lifecycle"][
            "storage_receipt_successor_after_terminal_commit"
        ]
        is False
    )
    assert calls == 1


def test_precloseout_receipt_reconciliation_rejects_semantic_change(
    tmp_path: Path,
) -> None:
    old_lifecycle = {
        "terminal_lifecycle_commit_path": "/run/terminal_lifecycle_commit.json",
        "terminal_lifecycle_commit_file_sha256": "1" * 64,
        "terminal_lifecycle_commit_sha256": "2" * 64,
        "artifact_disposition_manifest_path": "/run/artifact_disposition_manifest.json",
        "artifact_disposition_manifest_sha256": "3" * 64,
        "storage_disposition_receipt_path": "/run/storage_disposition_receipt.json",
        "storage_disposition_receipt_sha256": "4" * 64,
        "storage_receipt_successor_after_terminal_commit": False,
    }
    new_lifecycle = {
        **old_lifecycle,
        "storage_disposition_receipt_sha256": "5" * 64,
        "storage_receipt_successor_after_terminal_commit": True,
    }
    old_core = {
        "schema_version": front_door.SCIENCE_REQUEST_V1_RUN_RECEIPT_SCHEMA_VERSION,
        "status": "COMPLETED",
        "request_sha256": {"algorithm": "sha256", "hex": "6" * 64},
        "aggregate": {"result_sha256": {"algorithm": "sha256", "hex": "7" * 64}},
        "completed_batches": [
            {
                "batch_index": 0,
                "result_sha256": "8" * 64,
                "child_lifecycle": old_lifecycle,
            }
        ],
    }
    old_receipt = {
        **old_core,
        "receipt_sha256": canonical_sha256(old_core).to_dict(),
    }
    candidate_core = {
        **old_core,
        "completed_batches": [
            {
                "batch_index": 0,
                "result_sha256": "8" * 64,
                "child_lifecycle": new_lifecycle,
            }
        ],
    }
    candidate = {
        **candidate_core,
        "receipt_sha256": canonical_sha256(candidate_core).to_dict(),
    }
    path = tmp_path / "science_campaign_run_receipt.json"
    _write_json(path, old_receipt)
    journal = {"batches": [{"child_lifecycle": old_lifecycle}]}

    assert (
        front_door._reconcile_precloseout_run_receipt_v1(
            path=path,
            candidate=candidate,
            journal=journal,
        )
        == old_receipt
    )

    tampered_core = {**old_core, "aggregate": {"result_sha256": "9" * 64}}
    tampered = {
        **tampered_core,
        "receipt_sha256": canonical_sha256(tampered_core).to_dict(),
    }
    _write_json(tmp_path / "tampered_receipt.json", tampered)
    with pytest.raises(ScienceRequestV1FrontDoorError, match="outside child lifecycle"):
        front_door._reconcile_precloseout_run_receipt_v1(
            path=tmp_path / "tampered_receipt.json",
            candidate=candidate,
            journal=journal,
        )


def test_run_resume_repairs_unfinished_closeout_without_rewriting_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _test_authority(tmp_path, monkeypatch)

    def executor(
        materialization: Path,
        *,
        batch_index: int,
        runtime_root: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        return _synthetic_batch_result(
            materialization,
            batch_index=batch_index,
            runtime_root=runtime_root,
        )

    monkeypatch.setattr(front_door, "aggregate_science_campaign_atlas_v1", _synthetic_aggregate)
    first = run_science_request_v1(
        _W32,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        runtime_parent=tmp_path / "runs",
        authority_resolver=lambda *args: authority,
        batch_executor=executor,
    )
    root = Path(first["runtime_root"])
    receipt_before = (root / "science_campaign_run_receipt.json").read_bytes()
    (root / "artifact_disposition_manifest.json").unlink()
    (root / "storage_disposition_receipt.json").unlink()
    journal = front_door._read_science_run_journal_v1(root)
    journal["status"] = "RUNNING"
    journal["last_event"] = "interrupted_before_closeout"
    front_door._write_science_run_journal_v1(root, journal)

    resumed = run_science_request_v1(
        _W32,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        runtime_parent=tmp_path / "runs",
        resume_root=root,
        authority_resolver=lambda *args: authority,
        batch_executor=lambda *args, **kwargs: pytest.fail(
            "authenticated completed batch was executed"
        ),
    )

    assert resumed["status"] == "COMPLETED"
    assert (root / "science_campaign_run_receipt.json").read_bytes() == receipt_before
    assert (root / "artifact_disposition_manifest.json").is_file()
    assert (root / "storage_disposition_receipt.json").is_file()


def test_tampered_completed_batch_receipt_is_not_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _test_authority(tmp_path, monkeypatch)
    called = 0

    def executor(
        materialization: Path,
        *,
        batch_index: int,
        runtime_root: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal called
        del kwargs
        called += 1
        return _synthetic_batch_result(
            materialization,
            batch_index=batch_index,
            runtime_root=runtime_root,
        )

    monkeypatch.setattr(
        front_door, "aggregate_science_campaign_atlas_v1", _synthetic_aggregate
    )
    first = run_science_request_v1(
        _W32,
        repo_root=Path.cwd(),
        materialization_parent=tmp_path / "materializations",
        runtime_parent=tmp_path / "runs",
        authority_resolver=lambda *args: authority,
        batch_executor=executor,
    )
    result_path = (
        Path(first["runtime_root"])
        / "batches"
        / "batch-0000"
        / "attempt-0001"
        / "batch_result.json"
    )
    result_path.chmod(0o644)
    result_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
        run_science_request_v1(
            _W32,
            repo_root=Path.cwd(),
            materialization_parent=tmp_path / "materializations",
            runtime_parent=tmp_path / "runs",
            resume_root=first["runtime_root"],
            authority_resolver=lambda *args: authority,
            batch_executor=lambda *args, **kwargs: pytest.fail(
                "tampered child was executed"
            ),
        )
    assert captured.value.code == front_door.SCIENCE_BATCH_EXECUTION_FAILED
    assert called == 1


def test_terminal_batch_failure_is_not_automatically_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _test_authority(tmp_path, monkeypatch)

    def failing_executor(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise ScienceRequestV1FrontDoorError(
            front_door.SCIENCE_BATCH_EXECUTION_FAILED,
            "terminal synthetic failure",
        )

    with pytest.raises(ScienceRequestV1FrontDoorError) as captured:
        run_science_request_v1(
            _W32,
            repo_root=Path.cwd(),
            materialization_parent=tmp_path / "materializations",
            runtime_parent=tmp_path / "runs",
            authority_resolver=lambda *args: authority,
            batch_executor=failing_executor,
        )
    root = Path(captured.value.details["runtime_root"])
    journal = json.loads((root / "run_journal.json").read_text())
    assert journal["status"] == "FAILED"
    with pytest.raises(ScienceRequestV1FrontDoorError) as resumed:
        run_science_request_v1(
            _W32,
            repo_root=Path.cwd(),
            materialization_parent=tmp_path / "materializations",
            runtime_parent=tmp_path / "runs",
            resume_root=root,
            authority_resolver=lambda *args: authority,
            batch_executor=lambda *args, **kwargs: pytest.fail(
                "terminal child was retried"
            ),
        )
    assert resumed.value.code == front_door.SCIENCE_BATCH_EXECUTION_FAILED


def test_keyboard_interrupt_is_not_reclassified_as_batch_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _test_authority(tmp_path, monkeypatch)

    def interrupting_executor(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run_science_request_v1(
            _W32,
            repo_root=Path.cwd(),
            materialization_parent=tmp_path / "materializations",
            runtime_parent=tmp_path / "runs",
            authority_resolver=lambda *args: authority,
            batch_executor=interrupting_executor,
        )
    run_root = next(
        path
        for path in (tmp_path / "runs").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    assert (run_root / "run_journal.json").is_file()
    assert not (run_root / "run_artifact_failure_receipt.json").exists()
