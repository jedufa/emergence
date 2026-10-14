from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.contracts import ExecutionRouteContractV1
from engine.campaigns.long_horizon_qualification import (
    BOUNDED_ROUTE_QUALIFIED,
    CANONICAL_FULL_PRODUCTION_ROUTE_SCOPE,
    LONG_HORIZON_EVIDENCE_SCHEMA_VERSION,
    LONG_HORIZON_RUNABILITY_QUALIFIED,
    LONG_HORIZON_RUNABILITY_ROUTE_SCOPE,
    SCIENCE_ROUTE_ATTESTATION_SCHEMA_VERSION,
    SCIENCE_ROUTE_QUALIFIED,
    LongHorizonQualificationError,
    qualification_dimensions_v1,
    validate_long_horizon_evidence_v1,
)
from engine.campaigns.qualification import (
    EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
    execution_route_qualification_status_v1,
)
from engine.qrr.production_front_door_v1 import (
    PRODUCTION_SCIENCE_REQUEST_SCHEMA_VERSION,
    ProductionScienceFrontDoorError,
    activate_qrr_v1,
    execute_science_request_v1,
    resolve_active_qrr_v1,
)
from engine.qrr.qualified_execution_envelope_v1 import (
    EXPERIMENT_REPRESENTATION_CONTRACT_SCHEMA_VERSION,
    QualifiedExecutionEnvelopeV1,
    build_qualified_execution_envelope_v1,
    load_qualified_execution_envelope_v1,
)
from engine.qrr.qualified_runtime_release_v1 import (
    _exact_route_qualified_preflight_v1,
)

TERMINAL_DIGEST = "a" * 64
EVENT_DIGEST = "b" * 64
PLAN_SHA256 = "c" * 64
SOURCE_SHA256 = "d" * 64
ARTIFACT_SHA256 = "e" * 64
RUN_RECEIPT_SHA256 = "f" * 64


def _contract() -> ExecutionRouteContractV1:
    return ExecutionRouteContractV1(
        schema_version="eiel_execution_route_contract_v1",
        provider_id="provider-v1",
        profile_id="profile-v1",
        components=(),
        required_obligations=(),
    )


def _evidence(
    *,
    route_sha256: str | None = None,
    artifact_sha256s: dict[str, str] | None = None,
) -> dict:
    route = route_sha256 or _contract().execution_route_sha256
    windows = [
        {
            "start_tick": index * 1250,
            "end_tick": (index + 1) * 1250,
            "peak_gpu_allocated_bytes": 90,
            "peak_gpu_reserved_bytes": 100,
            "minimum_free_gpu_bytes": 500,
            "host_peak_rss_bytes": 100,
            "host_terminal_rss_bytes": 90,
            "resident_event_bytes": 64,
            "resident_source_bytes": 128,
            "event_ring_peak_occupancy": 10,
            "events_emitted": 100,
            "events_drained": 90 if index == 3 else 100,
        }
        for index in range(4)
    ]
    body = {
        "schema_version": LONG_HORIZON_EVIDENCE_SCHEMA_VERSION,
        "evidence_identity": {
            "execution_route_sha256": route,
            "provider_id": "provider-v1",
            "profile_id": "profile-v1",
            "source_sha256": SOURCE_SHA256,
            "artifact_sha256s": artifact_sha256s or {"cuda_extension": ARTIFACT_SHA256},
            "plan_sha256": PLAN_SHA256,
            "run_id": "run-b136-5000-v1",
            "run_receipt_sha256": RUN_RECEIPT_SHA256,
            "route_scope": LONG_HORIZON_RUNABILITY_ROUTE_SCOPE,
            "geometry": {
                "batch_cells": 136,
                "world_width": 32,
                "population_capacity": 256,
                "controller_capacity": 512,
            },
        },
        "resource_limits": {
            "gpu_budget_bytes": 1000,
            "host_budget_bytes": 1000,
            "minimum_free_gpu_bytes_required": 100,
            "maximum_gpu_memory_slope_bytes_per_window": 10,
            "maximum_host_memory_slope_bytes_per_window": 10,
            "maximum_event_ring_occupancy_slope_per_window": 10,
        },
        "ticks_requested": 5000,
        "ticks_committed": 5000,
        "owner_boundaries_crossed": 4,
        "peak_gpu_allocated_bytes": 90,
        "peak_gpu_reserved_bytes": 100,
        "minimum_free_gpu_bytes": 500,
        "gpu_memory_slope_bytes_per_window": 0.0,
        "gpu_allocated_memory_slope_bytes_per_window": 0.0,
        "host_peak_rss_bytes": 100,
        "host_terminal_rss_bytes": 90,
        "host_memory_slope_bytes_per_window": 0.0,
        "resident_event_bytes": 64,
        "resident_event_bytes_slope_per_window": 0.0,
        "resident_source_bytes": 128,
        "resident_source_bytes_slope_per_window": 0.0,
        "event_ring_capacity": 64,
        "event_ring_peak_occupancy": 10,
        "event_ring_occupancy_slope_per_window": 0.0,
        "events_emitted": 400,
        "events_drained": 390,
        "terminal_event_ring_occupancy": 10,
        "external_event_history_count": 390,
        "external_event_history_digest": EVENT_DIGEST,
        "checkpoint_frontier": {
            "next_absolute_tick": 5000,
            "terminal_digest": TERMINAL_DIGEST,
        },
        "checkpoint_restore": {
            "checkpoint_tick": 2501,
            "checkpoint_digest": "1" * 64,
            "restored_checkpoint_digest": "1" * 64,
            "fresh_owner_restore_tick": 3361,
            "fresh_owner_uninterrupted_terminal_digest": TERMINAL_DIGEST,
            "fresh_owner_restored_terminal_digest": TERMINAL_DIGEST,
            "fresh_owner_uninterrupted_event_digest": EVENT_DIGEST,
            "fresh_owner_restored_event_digest": EVENT_DIGEST,
        },
        "terminal_digest": TERMINAL_DIGEST,
        "event_digest": EVENT_DIGEST,
        "windows": windows,
        "errors": [],
        "ooms": [],
    }
    return {**body, "evidence_sha256": canonical_sha256(body)}


def _rehash(evidence: dict) -> dict:
    evidence["evidence_sha256"] = canonical_sha256(
        {name: value for name, value in evidence.items() if name != "evidence_sha256"}
    )
    return evidence


def _attestation(evidence: dict, *, source_route_receipt_sha256: str = "9" * 64) -> dict:
    validated = validate_long_horizon_evidence_v1(evidence)
    identity = evidence["evidence_identity"]
    body = {
        "schema_version": SCIENCE_ROUTE_ATTESTATION_SCHEMA_VERSION,
        "route_scope": CANONICAL_FULL_PRODUCTION_ROUTE_SCOPE,
        "claim_dimension": SCIENCE_ROUTE_QUALIFIED,
        "execution_route_sha256": identity["execution_route_sha256"],
        "provider_id": identity["provider_id"],
        "profile_id": identity["profile_id"],
        "source_sha256": identity["source_sha256"],
        "artifact_sha256s": dict(identity["artifact_sha256s"]),
        "plan_sha256": identity["plan_sha256"],
        "run_id": identity["run_id"],
        "run_receipt_sha256": identity["run_receipt_sha256"],
        "geometry": dict(identity["geometry"]),
        "evidence_sha256": evidence["evidence_sha256"],
        "evidence_binding_sha256": validated["evidence_binding"]["binding_sha256"],
        "source_route_receipt_sha256": source_route_receipt_sha256,
        "canonical_route_pass": True,
        "production_resource_safety_pass": True,
        "terminal_finalization_pass": True,
        "science_output_pass": True,
    }
    return {**body, "attestation_sha256": canonical_sha256(body)}


def _validated(*, science: bool = False) -> dict:
    evidence = _evidence()
    attestation = _attestation(evidence) if science else None
    return validate_long_horizon_evidence_v1(
        evidence,
        expected_identity={
            "execution_route_sha256": _contract().execution_route_sha256,
            "provider_id": "provider-v1",
            "profile_id": "profile-v1",
            "source_sha256": SOURCE_SHA256,
            "artifact_sha256s": {"cuda_extension": ARTIFACT_SHA256},
            "plan_sha256": PLAN_SHA256,
            "run_id": "run-b136-5000-v1",
            "run_receipt_sha256": RUN_RECEIPT_SHA256,
        },
        canonical_full_production_route_attestation=attestation,
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("execution_route_sha256", "7" * 64),
        ("source_sha256", "7" * 64),
        ("artifact_sha256s", {"cuda_extension": "7" * 64}),
        ("plan_sha256", "7" * 64),
        ("run_id", "other-run"),
        ("run_receipt_sha256", "7" * 64),
    ),
)
def test_every_immutable_evidence_identity_axis_is_exact(
    field: str, replacement
) -> None:
    evidence = _evidence()
    evidence["evidence_identity"][field] = replacement
    _rehash(evidence)
    expected = {
        "execution_route_sha256": _contract().execution_route_sha256,
        "source_sha256": SOURCE_SHA256,
        "artifact_sha256s": {"cuda_extension": ARTIFACT_SHA256},
        "plan_sha256": PLAN_SHA256,
        "run_id": "run-b136-5000-v1",
        "run_receipt_sha256": RUN_RECEIPT_SHA256,
    }
    with pytest.raises(LongHorizonQualificationError, match=field):
        validate_long_horizon_evidence_v1(evidence, expected_identity=expected)


def test_validated_evidence_is_the_only_science_dimension_source() -> None:
    validated = _validated()

    assert validated["event_accounting_exact"] is True
    assert validated["checkpoint_export_load_exact"] is True
    assert validated["fresh_owner_checkpoint_continuation_exact"] is True
    assert validated["qualification_dimensions"] == [
        LONG_HORIZON_RUNABILITY_QUALIFIED,
    ]
    assert qualification_dimensions_v1(
        bounded_route_pass=True,
        long_horizon_evidence=_evidence(),
    ) == (
        BOUNDED_ROUTE_QUALIFIED,
        LONG_HORIZON_RUNABILITY_QUALIFIED,
    )
    with pytest.raises(
        LongHorizonQualificationError, match="structured evidence_refs"
    ):
        qualification_dimensions_v1(
            bounded_route_pass=True,
            long_horizon_evidence=_evidence(),
            canonical_full_production_route_attestation=_attestation(_evidence()),
        )
    with pytest.raises(TypeError):
        qualification_dimensions_v1(  # type: ignore[call-arg]
            bounded_route_pass=True,
            long_horizon_evidence=_evidence(),
            production_resource_safety_pass=True,
        )


def test_noncanonical_long_horizon_evidence_cannot_grant_science() -> None:
    evidence = _evidence()
    evidence["evidence_identity"]["route_scope"] = LONG_HORIZON_RUNABILITY_ROUTE_SCOPE
    _rehash(evidence)
    assert qualification_dimensions_v1(
        bounded_route_pass=True,
        long_horizon_evidence=evidence,
    ) == (BOUNDED_ROUTE_QUALIFIED, LONG_HORIZON_RUNABILITY_QUALIFIED)


def test_canonical_scope_string_cannot_replace_the_gate27_attestation() -> None:
    evidence = _evidence()
    evidence["evidence_identity"]["route_scope"] = (
        CANONICAL_FULL_PRODUCTION_ROUTE_SCOPE
    )
    _rehash(evidence)
    assert qualification_dimensions_v1(
        bounded_route_pass=True,
        long_horizon_evidence=evidence,
    ) == (BOUNDED_ROUTE_QUALIFIED, LONG_HORIZON_RUNABILITY_QUALIFIED)
    with pytest.raises(LongHorizonQualificationError, match="unchanged long-horizon"):
        qualification_dimensions_v1(
            bounded_route_pass=True,
            long_horizon_evidence=evidence,
            canonical_full_production_route_attestation=_attestation(evidence),
        )


def test_science_route_status_rejects_evidence_without_separate_route_scope() -> None:
    receipt = _route_receipt(_evidence())
    receipt["qualification_dimensions"].append(SCIENCE_ROUTE_QUALIFIED)
    rejected = execution_route_qualification_status_v1(_contract(), receipt)
    assert rejected["qualification_dimensions"] == [BOUNDED_ROUTE_QUALIFIED]
    assert any(
        row["code"] == "execution_route_long_horizon_evidence_invalid"
        for row in rejected["reasons"]
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda row: row["windows"][3].__setitem__(
                "peak_gpu_reserved_bytes", 10**15
            ),
            "GPU budget",
        ),
        (
            lambda row: row.__setitem__(
                "host_memory_slope_bytes_per_window", "UNVALIDATED"
            ),
            "finite number",
        ),
        (
            lambda row: row["windows"][2].__setitem__(
                "event_ring_peak_occupancy", 10**12
            ),
            "event_ring_peak_occupancy differs",
        ),
        (
            lambda row: row.__setitem__("external_event_history_count", 389),
            "external event history",
        ),
        (
            lambda row: row["checkpoint_restore"].__setitem__(
                "fresh_owner_restored_terminal_digest", "9" * 64
            ),
            "continuation digests",
        ),
        (
            lambda row: row["evidence_identity"].__setitem__("source_sha256", "9" * 64),
            "source_sha256",
        ),
    ),
)
def test_fabricated_or_unbounded_evidence_is_rejected(mutate, message: str) -> None:
    evidence = _evidence()
    mutate(evidence)
    _rehash(evidence)
    with pytest.raises(LongHorizonQualificationError, match=message):
        validate_long_horizon_evidence_v1(
            evidence,
            expected_identity={"source_sha256": SOURCE_SHA256},
        )


def _route_receipt(evidence: dict | None = None, *, science: bool = False) -> dict:
    dimensions = [BOUNDED_ROUTE_QUALIFIED]
    scope: dict = {}
    science_scope: dict = {}
    if evidence is not None:
        validated = validate_long_horizon_evidence_v1(evidence)
        dimensions.extend(validated["qualification_dimensions"])
        scope = {
            "validated_evidence": evidence,
            "evidence_binding": validated["evidence_binding"],
        }
        if science:
            attestation = _attestation(evidence)
            dimensions.append(SCIENCE_ROUTE_QUALIFIED)
            science_scope = {
                "route_scope": CANONICAL_FULL_PRODUCTION_ROUTE_SCOPE,
                "validated_evidence": evidence,
                "evidence_binding": validated["evidence_binding"],
                "canonical_full_production_route_attestation": attestation,
            }
    receipt = {
        "schema_version": EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
        "authority_state": "route_qualified",
        "execution_route_sha256": _contract().execution_route_sha256,
        "provider_id": "provider-v1",
        "profile_id": "profile-v1",
        "covered_obligations": [],
        "route_obligation_results": [],
        "artifact_sha256": ARTIFACT_SHA256,
        "qualification_dimensions": dimensions,
        "long_horizon_scope": scope,
    }
    if science_scope:
        receipt["science_route_scope"] = science_scope
    if evidence is not None:
        receipt["artifact_sha256s"] = dict(
            evidence["evidence_identity"]["artifact_sha256s"]
        )
    return receipt


def test_route_status_propagates_only_revalidated_evidence_authority() -> None:
    status = execution_route_qualification_status_v1(
        _contract(), _route_receipt(_evidence(), science=True)
    )
    assert status["authority_state"] == "route_qualified"
    assert status["qualification_dimensions"] == [BOUNDED_ROUTE_QUALIFIED]
    assert status["qualification_evidence_binding"] is None
    assert any(
        row["code"] == "execution_route_long_horizon_evidence_invalid"
        for row in status["reasons"]
    )

    artifacts = {
        "cuda_extension": ARTIFACT_SHA256,
        "extension_manifest": "9" * 64,
    }
    multi_artifact = execution_route_qualification_status_v1(
        _contract(), _route_receipt(_evidence(artifact_sha256s=artifacts), science=True)
    )
    assert multi_artifact["qualification_dimensions"] == [BOUNDED_ROUTE_QUALIFIED]

    fabricated = _route_receipt()
    fabricated["qualification_dimensions"] += [
        LONG_HORIZON_RUNABILITY_QUALIFIED,
        SCIENCE_ROUTE_QUALIFIED,
    ]
    rejected = execution_route_qualification_status_v1(_contract(), fabricated)
    assert rejected["qualification_dimensions"] == [BOUNDED_ROUTE_QUALIFIED]
    assert rejected["qualification_evidence_binding"] is None
    assert any(
        row["code"] == "execution_route_long_horizon_evidence_invalid"
        for row in rejected["reasons"]
    )


def _envelope(*, science: bool) -> QualifiedExecutionEnvelopeV1:
    route = _contract().execution_route_sha256
    artifacts = {"cuda_extension": ARTIFACT_SHA256}
    validated = _validated(science=False)
    evidence_binding = dict(validated["evidence_binding"])
    if science:
        evidence_binding["qualification_dimensions"] = [
            LONG_HORIZON_RUNABILITY_QUALIFIED,
            SCIENCE_ROUTE_QUALIFIED,
        ]
        evidence_binding[
            "canonical_full_production_route_attestation_sha256"
        ] = "9" * 64
        evidence_binding["binding_sha256"] = canonical_sha256(
            {
                name: value
                for name, value in evidence_binding.items()
                if name != "binding_sha256"
            }
        )
    dimensions = (
        (
            BOUNDED_ROUTE_QUALIFIED,
            LONG_HORIZON_RUNABILITY_QUALIFIED,
            SCIENCE_ROUTE_QUALIFIED,
        )
        if science
        else (BOUNDED_ROUTE_QUALIFIED,)
    )
    return QualifiedExecutionEnvelopeV1(
        envelope_id="authority-test-v1",
        provider_id="provider-v1",
        profile_id="profile-v1",
        execution_route_sha256=route,
        qualification_key_sha256="3" * 64,
        artifact_sha256s=artifacts,
        experiment_variable_contracts=(
            {
                "name": "control",
                "value_contract": {"kind": "int", "minimum": 0, "maximum": 1},
            },
        ),
        fixed_parameter_values={},
        admitted_batch_range=(1, 136),
        admitted_generation_range=(1, 32),
        admitted_ticks_per_generation=(160,),
        checkpoint_storage_modes=("disk",),
        resource_contract={
            "declared_budgets": {
                "disk_bytes": 1,
                "vram_bytes": 1000,
                "wall_seconds": 1,
            },
            "provider_resource_estimator_id": "resource-v1",
            "reference_provider_estimate": {},
            "admission_policy": "release_local_preflight_current_resource_state_v1",
        },
        representation_contract={
            "schema_version": EXPERIMENT_REPRESENTATION_CONTRACT_SCHEMA_VERSION,
            "policy": "shared_scalar_configs_with_qualified_per_cell_exceptions_v1",
            "execution_route_sha256": route,
            "artifact_sha256s": artifacts,
            "per_cell_mixed_parameter_names": [],
            "batch_uniform_parameter_names": ["control"],
        },
        time_contract={
            "authority": "absolute_tick_v1",
            "arbitrary_positive_terminal_tick": True,
            "arbitrary_checkpoint_ticks": True,
        },
        qualification_dimensions=dimensions,
        qualification_evidence_binding=(
            evidence_binding if science else {}
        ),
    )


def test_tick_native_envelope_roundtrip_preserves_exact_authority() -> None:
    science = _envelope(science=True)
    assert (
        load_qualified_execution_envelope_v1(
            science.payload()
        ).qualification_evidence_binding
        == science.qualification_evidence_binding
    )

    bounded = _envelope(science=False)
    assert load_qualified_execution_envelope_v1(
        bounded.payload()
    ).qualification_dimensions == (BOUNDED_ROUTE_QUALIFIED,)


def test_envelope_builder_consumes_normalized_route_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = execution_route_qualification_status_v1(
        _contract(), _route_receipt(_evidence(), science=True)
    )
    plan = {
        "plan_sha256": PLAN_SHA256,
        "qualification_key_sha256": "3" * 64,
        "qualification_key": {
            "admitted_batch_range": [1, 136],
            "variability_envelope": {
                "envelope_id": "authority-test-v1",
                "parameter_contracts": [
                    {"name": "x", "value_contract": {"kind": "int"}}
                ],
            },
        },
        "artifacts": {
            "cuda_extension": {"sha256": ARTIFACT_SHA256},
        },
        "physical_batches": [{"batch_id": "batch-0", "cell_ids": ["cell-0"]}],
        "cells": [{"cell_id": "cell-0", "parameters": {"x": 1}}],
        "execution": {
            "execution_route_identity": "tick_native_execution_route_v1",
            "total_ticks": 5000,
            "checkpoint_storage_mode": "disk",
            "budgets": {
                "disk_bytes": 1,
                "vram_bytes": 1000,
                "wall_seconds": 1,
            },
            "tick_native_control": {
                "time_authority": "absolute_tick_v1",
                "total_ticks": 5000,
                "resident_owner_span_ticks": 1120,
                "native_dispatch_window_ticks": 1120,
                "source_window_ticks": 160,
                "event_window_capacity": 41804,
            },
        },
    }
    monkeypatch.setattr(
        "engine.qrr.qualified_execution_envelope_v1.build_execution_request_v1",
        lambda *_args: SimpleNamespace(
            execution_route_sha256=_contract().execution_route_sha256,
            execution_route_contract={
                "components": [
                    {"component_id": "native_mechanics_binding", "batch_max": 136}
                ]
            },
            resident_request=object(),
        ),
    )
    provider = SimpleNamespace(
        provider_id="provider-v1",
        estimate_resources=lambda *_args: {"resource_estimator_id": "resource-v1"},
    )
    profile = SimpleNamespace(
        profile_id="profile-v1",
        qualified_batch_min=1,
        qualified_batch_max=136,
    )
    envelope = build_qualified_execution_envelope_v1(
        plan,
        provider=provider,
        profile=profile,
        qualification_authority=status,
    )
    assert envelope.qualification_dimensions == (BOUNDED_ROUTE_QUALIFIED,)
    assert envelope.qualification_evidence_binding == {}


def _release(
    envelope: QualifiedExecutionEnvelopeV1, release_id: str
) -> SimpleNamespace:
    science = SCIENCE_ROUTE_QUALIFIED in envelope.qualification_dimensions
    return SimpleNamespace(
        release_id=release_id,
        qualified_execution_envelope=envelope.payload(),
        qualification_authority={
            "qualification_dimensions": list(envelope.qualification_dimensions),
            "qualification_evidence_binding": dict(
                envelope.qualification_evidence_binding
            ),
            "execution_lane": (
                "production_science" if science else "qualification_or_qualprep"
            ),
            "production_science_authorized": science,
        },
        payload=lambda: {"manifest_sha256": "4" * 64},
    )


def test_production_activation_rejects_bounded_and_accepts_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qrr_root = tmp_path / "qrr"
    bounded_root = qrr_root / "qrr1-bounded"
    science_root = qrr_root / "qrr1-science"
    bounded_root.mkdir(parents=True)
    science_root.mkdir()
    releases = {
        bounded_root: _release(_envelope(science=False), bounded_root.name),
        science_root: _release(_envelope(science=True), science_root.name),
    }
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.load_qualified_runtime_release_v1",
        lambda path: releases[path],
    )
    with pytest.raises(
        ProductionScienceFrontDoorError, match="SCIENCE_ROUTE_QUALIFIED"
    ):
        activate_qrr_v1(bounded_root, qrr_root=qrr_root)
    registry = activate_qrr_v1(science_root, qrr_root=qrr_root)
    assert registry["active_release_id"] == science_root.name


def test_resolution_rechecks_science_authority_geometry_and_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qrr_root = tmp_path / "qrr"
    release_root = qrr_root / "qrr1-science"
    release_root.mkdir(parents=True)
    science = _release(_envelope(science=True), release_root.name)
    bounded = _release(_envelope(science=False), release_root.name)
    current = {"release": science}
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.load_qualified_runtime_release_v1",
        lambda _path: current["release"],
    )
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.load_campaign_spec",
        lambda _path: {
            "world": {
                "base": "patch",
                "overlays": [
                    "eed_resource_impact",
                    "jungle",
                    "wdl_delayed_hazard",
                ],
            },
            "parameters": {"patch.grid_size": 32},
        },
    )
    activate_qrr_v1(release_root, qrr_root=qrr_root)
    request = {
        "schema_version": PRODUCTION_SCIENCE_REQUEST_SCHEMA_VERSION,
        "request_id": "b136-science",
        "world_width": 32,
        "batch_cells": 136,
        "total_ticks": 5000,
        "checkpoint_ticks": [2501, 3361, 5000],
        "seeds": list(range(136)),
        "configurations": [{"id": "science", "controls": {"control": 1}}],
        "deliberate_restore_ticks": [2501, 3361],
        "observability": {"living_lineage_canopy": True},
    }
    _, receipt = resolve_active_qrr_v1(request, qrr_root=qrr_root)
    assert receipt["status"] == "PASS"

    too_long = deepcopy(request)
    too_long["total_ticks"] = 5001
    too_long["checkpoint_ticks"][-1] = 5001
    with pytest.raises(
        ProductionScienceFrontDoorError, match="SCIENCE_TICK_HORIZON_NOT_QUALIFIED"
    ):
        resolve_active_qrr_v1(too_long, qrr_root=qrr_root)

    current["release"] = bounded
    with pytest.raises(
        ProductionScienceFrontDoorError, match="SCIENCE_ROUTE_QUALIFIED"
    ):
        resolve_active_qrr_v1(request, qrr_root=qrr_root)
    request_path = tmp_path / "science-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(
        ProductionScienceFrontDoorError, match="SCIENCE_ROUTE_QUALIFIED"
    ):
        execute_science_request_v1(
            "check-representability",
            request_path,
            qrr_root=qrr_root,
            run_root=tmp_path / "must-not-materialize",
        )
    assert not (tmp_path / "must-not-materialize").exists()


def test_bounded_route_remains_an_explicit_qualification_execution_lane() -> None:
    plan = {"plan_sha256": PLAN_SHA256}
    preflight = {
        "schema_version": "eiel_campaign_preflight_v1",
        "plan_sha256": PLAN_SHA256,
        "ready_for_execution": True,
        "execution_route": {
            "authority_state": "route_qualified",
            "artifact_conformance_exact": True,
            "qualification_dimensions": [BOUNDED_ROUTE_QUALIFIED],
            "qualification_authority_scope": "bounded_route",
        },
    }
    assert _exact_route_qualified_preflight_v1(preflight, plan) is True
