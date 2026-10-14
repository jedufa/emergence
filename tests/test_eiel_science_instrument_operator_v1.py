from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import eiel_composition.science_instrument_operator_v1 as science_operator
from eiel_composition.science_instrument_artifacts_v1 import (
    serialize_science_host_context_v1,
)
from eiel_composition.science_cuda_v2_carrier_v1 import (
    SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1,
    CudaV2ScienceCarrierFamilyAdapterV1,
)
from eiel_composition.science_world_bundle_v1 import (
    ScienceWorldBundleAdmissionV1,
    ScienceWorldBundleError,
    load_science_world_bundle_v1,
    write_science_world_bundle_v1,
)
from emergence_contracts import ContentDigest, ScienceWorldBundleV1, canonical_sha256
from emergence_contracts import (
    CanonicalPayloadV1,
    SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_VERSION,
    SchemaVersion,
    ScienceObservationDraftV1,
    ScienceRecordKindV1,
    canonical_json_bytes,
    canonical_science_record_stream_sha256_v1,
    finalize_science_observation_drafts_v1,
)
from emergence_contracts.science_instrument import _RECORD_REQUIRED_PAYLOAD_FIELDS
from eiel_composition.science_cuda_v2_host_context_v1 import (
    build_mutsup_w32_paired_seed_request_v1,
    build_w32_cuda_v2_mutsup_base_plan_v1,
    materialize_w32_cuda_v2_science_host_context_v1,
)
from engine.campaigns import cuda_full_evolution_v2_provider as cuda_provider
from engine.campaigns.cuda_full_evolution_v2_provider import (
    CUDA_FULL_EVOLUTION_V2_BASELINE_DEVIATION_REQUIREMENT_IDS_V1,
)
from engine.native_runtime.cuda_v2_science_capture_v1 import (
    build_native_cuda_v2_completion_output_v1,
)
from engine.native_runtime.science_observation_projection_v1 import (
    NativeCudaV2ObservationContextV1,
    finalize_native_cuda_v2_observations_v1,
)
from tests.test_eiel_science_instrument_host_front_door_v1 import (
    _campaign,
    _context,
    _request,
)
from tests.test_eiel_science_compact_world_atlas_v1 import _payload, _world
from eiel_composition.science_instrument_operator_v1 import (
    SCIENCE_CUDA_V2_OBSERVATION_TYPED_SIDECAR_V1,
    SCIENCE_CUDA_V2_USE_CLASS_EXPLORATORY_SCIENCE_V1,
    ScienceInstrumentOperatorError,
    execute_materialized_science_batch_v1,
    load_science_instrument_materialization_v1,
    materialize_science_instrument_v1,
    reduce_mutsup_science_instrument_v1,
)


def _science_request_and_context() -> tuple[dict[str, Any], Any]:
    context = _context()
    campaign = _campaign(context.instrument_manifest)
    campaign["design"]["dimensions"][0]["canonical_values"] = [7, 11, 13, 17, 19, 23]
    campaign["design"]["expected_cell_count"] = 6
    campaign["execution"]["logical_world_count"] = 6
    campaign["question"]["science_ownership"] = {
        "test_id": "MUTSUP",
        "test_version": "1",
        "study_id": "MUTSUP-OPERATOR-TEST",
        "fixed_arm_id": "ordinary_birth",
        "fixed_replicate": 1,
    }
    campaign["seeds"] += [
        {"owner": owner, "binding_kind": "literal", "literal": 7}
        for owner in ("ecology", "founder", "policy", "mutation")
    ]
    request = _request(campaign)
    request["baseline"]["deviations"] = [
        {
            "requirement_id": requirement_id,
            "reason": "CUDA V2 exploratory operator fixture declares its route delta.",
        }
        for requirement_id in (
            CUDA_FULL_EVOLUTION_V2_BASELINE_DEVIATION_REQUIREMENT_IDS_V1
        )
    ]
    return request, context


def test_selected_world_host_binding_is_reopened_without_runtime_observation_upgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "attempt"
    observation_path = runtime / "batches" / "selected" / "observation.json"
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text("{}\n", encoding="ascii")
    binding = {
        "observation_mode": (
            science_operator.WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1
        ),
        "path": str(observation_path.resolve()),
        "file_sha256": "1" * 64,
        "content_sha256": "2" * 64,
        "file_bytes": observation_path.stat().st_size,
        "checkpoint_occurrence_sha256": "3" * 64,
        "checkpoint_occurrence_projection_sha256": "4" * 64,
        "terminally_reopened": True,
        "runtime_observation_mode": "OFF",
        "host_adapter_observation_only": True,
    }
    terminal = {
        "operation": "INDEXED_ACS_CONTINUATION",
        "host_boundary_observation": binding,
    }
    front_door_result = {
        "workhorse_terminalized_operation": terminal,
        "workhorse_terminalized_indexed_continuation": terminal,
    }
    monkeypatch.setattr(
        science_operator,
        "validate_workhorse_selected_world_observation_v1",
        lambda *_args, **_kwargs: {
            "cell_id": "world-selected",
            "execution_authority_granted": False,
            "qualification_authority_granted": False,
            "named_phenotype_reward_authority_granted": False,
            "integrity": {
                "world_count": 1,
                "dropped_record_count": 0,
                "runtime_observation_mode_remains_off": True,
                "host_boundary_adapter_is_observation_only": True,
            },
            "event_observation_coverage": {
                "event_kind_rows_available": False,
                "birth_death_causal_rows_claimed": False,
            },
            "world_observations": [{}],
            "observation_path": str(observation_path.resolve()),
            "observation_file_sha256": "1" * 64,
            "observation_content_sha256": "2" * 64,
            "observation_file_bytes": observation_path.stat().st_size,
        },
    )

    assert science_operator._authenticated_selected_world_host_observation_binding_v1(
        front_door_result,
        runtime_root=runtime,
        expected_world_id="world-selected",
    ) == binding

    del terminal["host_boundary_observation"]
    with pytest.raises(
        ScienceInstrumentOperatorError,
        match="lacks its terminally reopened host observation",
    ):
        science_operator._authenticated_selected_world_host_observation_binding_v1(
            front_door_result,
            runtime_root=runtime,
            expected_world_id="world-selected",
        )


def _base_plan() -> dict[str, Any]:
    plan: dict[str, Any] = {
        "schema_version": "eiel_campaign_plan_v1",
        "artifact_lifecycle": {
            "profile_id": "SCIENCE_COMPACT",
            "storage": {"total_max_bytes": 8 * 1024**3},
        },
        "resolved_profile": "eiel_full_evolution_cuda_native_v2",
        "execution": {"tick_native_control": {}},
        "output": {"runtime_root": "not-bound-yet"},
        "cells": [{"parameters": {}}],
    }
    plan["plan_sha256"] = canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ).hex
    return plan


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def test_compact_reducer_tail_budget_scales_with_exact_batch_and_horizon() -> None:
    assert science_operator.resolve_compact_reducer_tail_seconds_v1(
        None,
        world_count=1,
        total_ticks=5_000,
    ) == (60.0, "auto_world_count_tick_horizon")
    assert science_operator.resolve_compact_reducer_tail_seconds_v1(
        None,
        world_count=1_024,
        total_ticks=100_000,
    ) == (960.0, "auto_world_count_tick_horizon")
    assert science_operator.resolve_compact_reducer_tail_seconds_v1(
        0.01,
        world_count=1_024,
        total_ticks=100_000,
    ) == (0.01, "explicit")
    with pytest.raises(ValueError, match="at most 3600"):
        science_operator.resolve_compact_reducer_tail_seconds_v1(
            3_601.0,
            world_count=1,
            total_ticks=100_000,
        )


def _materialize_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Any]:
    request, context = _science_request_and_context()
    request_path = _write_json(tmp_path / "request.json", request)
    context_path = _write_json(
        tmp_path / "science_host_context.json",
        serialize_science_host_context_v1(context),
    )
    base_path = _write_json(tmp_path / "base_plan.json", _base_plan())
    return request_path, context_path, base_path, context


def test_materialize_expands_six_explicit_worlds_and_lowers_typed_batches(
    tmp_path: Path,
) -> None:
    request_path, context_path, base_path, context = _materialize_inputs(tmp_path)

    result = materialize_science_instrument_v1(
        request_path,
        context_path,
        base_path,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=5_000,
        physical_batch_capacity=1024,
        temporal_k=1,
    )

    assert result["world_bundle"]["world_count"] == 6
    assert result["bundle_admission"]["outcome"] == "ADMITTED"
    assert result["qualified_envelope_classification"]["primary_outcome"] == (
        "OUTSIDE_QUALIFIED_ENVELOPE"
    )
    assert result["qualification_authority_effect"] is False
    assert len(result["lowered_batches"]) == 1
    lowered = result["lowered_batches"][0]
    expected_manifest = context.instrument_manifest.manifest_sha256().to_dict()
    expected_crca = context.crca_semantic_binding.semantic_run_digest.to_dict()
    mapping_world_uids = {
        row["world"]["world_id"]: row["world"]["world_uid"]
        for row in result["carrier_plan"]["batches"][0]["world_slots"]
    }
    for cell in lowered["cells"]:
        assert cell["science_instrument_manifest_sha256"] == expected_manifest
        assert cell["science_crca_semantic_run_digest"] == expected_crca
        assert cell["science_logical_world_id"] == mapping_world_uids[cell["cell_id"]]
        assert cell["science_logical_world_id"] != cell["seeds"]["task"]
    assert (
        lowered["science_use_class"] == SCIENCE_CUDA_V2_USE_CLASS_EXPLORATORY_SCIENCE_V1
    )
    assert lowered["science_materialization"]["observation_mode"] == (
        SCIENCE_CUDA_V2_OBSERVATION_TYPED_SIDECAR_V1
    )
    assert result["engineering_only"] is True
    assert result["prequal_started"] is False
    assert result["qualification_started"] is False

    root, receipt, normalized, lowered_batches = (
        load_science_instrument_materialization_v1(result["receipt_path"])
    )
    assert root == tmp_path / "materialization"
    assert (
        receipt["identities"]["base_campaign_plan_sha256"]
        == _base_plan()["plan_sha256"]
    )
    assert normalized["request_id"] == "host-front-door-request-v1"
    assert len(lowered_batches) == 1


@pytest.mark.parametrize("which", ["request", "base"])
def test_materialization_requires_science_compact_for_both_rich_and_base_inputs(
    tmp_path: Path, which: str
) -> None:
    request_path, context_path, base_path, context = _materialize_inputs(tmp_path)
    if which == "request":
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["science_campaign"]["artifact_lifecycle"] = {
            "profile_id": "DEBUG_FORENSIC",
            "contract_ref": "eiel_run_artifact_lifecycle_contract_v1",
            "explicit_intent_allowed_override": None,
        }
        _write_json(request_path, request)
    else:
        base = _base_plan()
        base["artifact_lifecycle"] = {"profile_id": "DEBUG_FORENSIC"}
        base["plan_sha256"] = canonical_sha256(
            {key: value for key, value in base.items() if key != "plan_sha256"}
        ).hex
        _write_json(base_path, base)

    with pytest.raises(ScienceInstrumentOperatorError, match="SCIENCE_COMPACT"):
        materialize_science_instrument_v1(
            request_path,
            context_path,
            base_path,
            output_directory=tmp_path / "rejected",
            maximum_total_ticks=5_000,
        )


def test_materialization_fails_closed_when_tick_budget_is_exceeded(
    tmp_path: Path,
) -> None:
    request_path, context_path, base_path, _ = _materialize_inputs(tmp_path)

    with pytest.raises(
        ScienceInstrumentOperatorError, match="OUTSIDE_QUALIFIED_ENVELOPE"
    ):
        materialize_science_instrument_v1(
            request_path,
            context_path,
            base_path,
            output_directory=tmp_path / "rejected",
            maximum_total_ticks=4_999,
        )
    assert not (tmp_path / "rejected").exists()


def test_execute_one_materialized_batch_uses_only_bound_engineering_front_door(
    tmp_path: Path,
) -> None:
    request_path, context_path, base_path, _ = _materialize_inputs(tmp_path)
    materialization = materialize_science_instrument_v1(
        request_path,
        context_path,
        base_path,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=5_000,
    )
    observed: dict[str, Any] = {}

    def fake_front_door(
        request: Any, plan: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        observed["request"] = request
        observed["plan"] = plan
        observed["kwargs"] = kwargs
        return {
            "ok": True,
            "status": "completed",
            "result_class": "SUCCESS",
        }

    result = execute_materialized_science_batch_v1(
        materialization["receipt_path"],
        batch_index=0,
        runtime_root=tmp_path / "runtime",
        repo_root=tmp_path,
        crca_adapter=object(),
        execution_front_door=fake_front_door,
    )

    assert result["front_door_result"]["status"] == "completed"
    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["result_class"] == "SUCCESS"
    assert observed["request"].purpose == "engineering"
    assert observed["request"].qualification_profile is None
    assert (
        tuple(row.requirement_id for row in observed["request"].baseline.deviations)
        == CUDA_FULL_EVOLUTION_V2_BASELINE_DEVIATION_REQUIREMENT_IDS_V1
    )
    assert observed["plan"]["science_use_class"] == (
        SCIENCE_CUDA_V2_USE_CLASS_EXPLORATORY_SCIENCE_V1
    )
    assert observed["kwargs"]["source_lineage"]["compiled_plan_sha256"] == (
        observed["plan"]["plan_sha256"]
    )
    assert observed["kwargs"]["science_lineage"]["world_ids"]
    assert observed["kwargs"]["cuda_full_evolution_v2_crca_adapter"] is not None
    assert (
        observed["kwargs"]["cuda_full_evolution_v2_checkpoint_occurrence_adapter"]
        is not None
    )
    assert result["prequal_started"] is False
    assert result["qualification_started"] is False


def test_execute_overlaps_compact_reducer_and_joins_only_after_front_door(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, context_path, base_path, _ = _materialize_inputs(tmp_path)
    materialization = materialize_science_instrument_v1(
        request_path,
        context_path,
        base_path,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=5_000,
    )
    runtime_root = tmp_path / "runtime"
    compact_root = runtime_root / "batches" / "science-batch-0000" / "science-compact"
    events: list[str] = []

    class FakeAsyncSidecar:
        pid = 4321

        def __init__(
            self,
            root: Path,
            bundle: object,
            *,
            max_workers: int,
            native_artifact_path: Path,
            native_manifest_path: Path,
            native_manifest_sha256: str,
            native_artifact_use_class: str,
        ) -> None:
            assert root == compact_root
            assert bundle is fake_bundle
            assert max_workers == 3
            assert native_artifact_path.name == "libeiel_native_compact_reducer_v1.so"
            assert (
                native_manifest_path.name
                == "eiel_native_compact_reducer_build_manifest_v1.json"
            )
            assert len(native_manifest_sha256) == 64
            assert (
                native_artifact_use_class
                == science_operator.NATIVE_REDUCER_ARTIFACT_USE_DEFAULT_TECHNICAL_V1
            )
            events.append("sidecar-started")

        def result(self, *, timeout_seconds: float) -> dict[str, Any]:
            assert timeout_seconds == 60.0
            events.append("sidecar-joined")
            return {"schema_version": "synthetic-atlas-result-v1"}

        def cancel(self) -> None:
            pytest.fail("successful async reducer must not be cancelled")

    fake_bundle = object()
    monkeypatch.setattr(
        science_operator,
        "_compact_world_atlas_sidecar_target_v1",
        lambda *args: (fake_bundle, compact_root, "science-batch-0000"),
    )
    monkeypatch.setattr(
        science_operator,
        "CompactWorldAtlasAsyncSidecarV1",
        FakeAsyncSidecar,
    )

    def fake_front_door(
        request: Any, plan: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        del request, plan, kwargs
        events.append("front-door-returned")
        return {
            "ok": True,
            "status": "completed",
            "result_class": "SUCCESS",
        }

    result = execute_materialized_science_batch_v1(
        materialization["receipt_path"],
        batch_index=0,
        runtime_root=runtime_root,
        repo_root=tmp_path,
        crca_adapter=object(),
        execution_front_door=fake_front_door,
        compact_reducer_workers=3,
    )

    assert events == [
        "sidecar-started",
        "front-door-returned",
        "sidecar-joined",
    ]
    assert result["ok"] is True
    assert result["compact_reducer"]["status"] == "COMPLETED"
    assert result["compact_reducer"]["tail_slo_met"] is True
    assert result["compact_reducer"]["execution_route"] == "native_cpp_v1"
    assert result["compact_reducer"]["native_binding"] == "checked_in_default"
    assert result["compact_reducer"]["atlas_result"] == {
        "schema_version": "synthetic-atlas-result-v1"
    }


def test_canonical_front_door_binds_root_before_deferred_compact_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, context_path, base_path, _ = _materialize_inputs(tmp_path)
    materialization = materialize_science_instrument_v1(
        request_path,
        context_path,
        base_path,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=5_000,
    )
    compact_root = tmp_path / "runtime" / "science-compact"
    events: list[str] = []
    lifecycle_hook_seen: list[bool] = []

    class DeferredSidecar:
        pid = 2468

        def __init__(self, root: Path, bundle: object, **kwargs: Any) -> None:
            assert root == compact_root
            assert bundle is fake_bundle
            assert kwargs["start_immediately"] is False
            events.append("sidecar-constructed")

        def start(self) -> None:
            events.append("sidecar-started")

        def result(self, *, timeout_seconds: float) -> dict[str, Any]:
            assert timeout_seconds == 60.0
            events.append("sidecar-joined")
            return {"schema_version": "synthetic-atlas-result-v1"}

        def cancel(self) -> None:
            pytest.fail("successful deferred sidecar must not be cancelled")

    fake_bundle = object()
    monkeypatch.setattr(
        science_operator,
        "_compact_world_atlas_sidecar_target_v1",
        lambda *args: (fake_bundle, compact_root, "science-batch-0000"),
    )
    monkeypatch.setattr(science_operator, "CompactWorldAtlasAsyncSidecarV1", DeferredSidecar)

    def fake_bound_front_door(
        request: Any, plan: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        del request, plan
        events.append("request-bound")
        assert callable(kwargs["before_lifecycle_closeout_fn"])
        lifecycle_hook_seen.append(True)
        kwargs["before_execute_fn"]()
        events.append("producer-started")
        return {"ok": True, "status": "completed", "result_class": "SUCCESS"}

    monkeypatch.setattr(
        science_operator,
        "execute_bound_campaign_plan_v2",
        fake_bound_front_door,
    )

    result = execute_materialized_science_batch_v1(
        materialization["receipt_path"],
        batch_index=0,
        runtime_root=tmp_path / "runtime",
        repo_root=tmp_path,
        crca_adapter=object(),
        compact_reducer_workers=3,
    )

    assert events == [
        "sidecar-constructed",
        "request-bound",
        "sidecar-started",
        "producer-started",
        "sidecar-joined",
    ]
    assert lifecycle_hook_seen == [True]
    assert result["compact_reducer"]["sidecar_pid"] == 2468


def test_resumable_front_door_wait_does_not_seal_attempt_reducer_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, context_path, base_path, _ = _materialize_inputs(tmp_path)
    materialization = materialize_science_instrument_v1(
        request_path,
        context_path,
        base_path,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=5_000,
    )
    compact_root = tmp_path / "runtime" / "science-compact"
    events: list[str] = []

    class DeferredSidecar:
        pid = 8642

        def __init__(self, root: Path, bundle: object, **kwargs: Any) -> None:
            assert root == compact_root
            assert bundle is fake_bundle
            assert kwargs["start_immediately"] is False
            events.append("sidecar-constructed")

        def start(self) -> None:
            pytest.fail("storage wait must happen before producer start")

        def cancel(self) -> None:
            events.append("sidecar-cancelled")

    fake_bundle = object()
    monkeypatch.setattr(
        science_operator,
        "_compact_world_atlas_sidecar_target_v1",
        lambda *args: (fake_bundle, compact_root, "science-batch-0000"),
    )
    monkeypatch.setattr(science_operator, "CompactWorldAtlasAsyncSidecarV1", DeferredSidecar)

    def fake_bound_front_door(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        events.append("front-door-waited")
        return {"ok": False, "status": "WAIT_STORAGE", "result_class": "STORAGE_WAIT"}

    monkeypatch.setattr(
        science_operator,
        "execute_bound_campaign_plan_v2",
        fake_bound_front_door,
    )

    result = execute_materialized_science_batch_v1(
        materialization["receipt_path"],
        batch_index=0,
        runtime_root=tmp_path / "runtime",
        repo_root=tmp_path,
        crca_adapter=object(),
        compact_reducer_workers=3,
    )

    assert events == ["sidecar-constructed", "front-door-waited", "sidecar-cancelled"]
    assert result["status"] == "WAIT_STORAGE"
    assert "execution_receipt_path" not in result["compact_reducer"]
    assert not (tmp_path / "runtime" / "compact_reducer_execution_receipt.json").exists()


def test_execute_binds_native_reducer_identity_without_producer_coupling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, context_path, base_path, _ = _materialize_inputs(tmp_path)
    materialization = materialize_science_instrument_v1(
        request_path,
        context_path,
        base_path,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=5_000,
        science_use_class=(
            SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1
        ),
    )
    compact_root = tmp_path / "runtime" / "science-compact"
    native_artifact = tmp_path / "libnative.so"
    native_manifest = tmp_path / "native-manifest.json"
    native_digest = "a" * 64
    events: list[str] = []

    class NativeSidecar:
        pid = 8765

        def __init__(
            self,
            root: Path,
            bundle: object,
            **kwargs: Any,
        ) -> None:
            assert root == compact_root
            assert bundle is fake_bundle
            assert kwargs == {
                "max_workers": 6,
                "native_artifact_path": native_artifact,
                "native_manifest_path": native_manifest,
                "native_manifest_sha256": native_digest,
                "native_artifact_use_class": "engineering_real_use_v1",
            }
            events.append("native-sidecar-started")

        def result_bundle(self, *, timeout_seconds: float) -> dict[str, Any]:
            assert timeout_seconds == 60.0
            events.append("native-sidecar-joined")
            return {
                "schema_version": "eiel_native_compact_reducer_execution_bundle_v1",
                "atlas_result": {"schema_version": "native-atlas-result-v1"},
                "strategy_attribution_result": {
                    "schema_version": "native-strategy-result-v1"
                },
                "second_python_attribution_pass": False,
            }

        def cancel(self) -> None:
            pytest.fail("successful native sidecar must not be cancelled")

    fake_bundle = object()
    monkeypatch.setattr(
        science_operator,
        "_compact_world_atlas_sidecar_target_v1",
        lambda *args: (fake_bundle, compact_root, "science-batch-0000"),
    )
    monkeypatch.setattr(
        science_operator,
        "CompactWorldAtlasAsyncSidecarV1",
        NativeSidecar,
    )

    def fake_front_door(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        events.append("front-door-returned")
        return {"ok": True, "status": "completed", "result_class": "SUCCESS"}

    result = execute_materialized_science_batch_v1(
        materialization["receipt_path"],
        batch_index=0,
        runtime_root=tmp_path / "runtime",
        repo_root=tmp_path,
        crca_adapter=object(),
        execution_front_door=fake_front_door,
        compact_reducer_workers=6,
        compact_reducer_native_artifact=native_artifact,
        compact_reducer_native_manifest=native_manifest,
        compact_reducer_native_manifest_sha256=native_digest,
        compact_reducer_native_use_class="engineering_real_use_v1",
    )
    assert events == [
        "native-sidecar-started",
        "front-door-returned",
        "native-sidecar-joined",
    ]
    assert result["compact_reducer"]["execution_route"] == "native_cpp_v1"
    assert result["compact_reducer"]["native_manifest_sha256"] == native_digest
    assert result["compact_reducer"]["native_artifact_use_class"] == (
        "engineering_real_use_v1"
    )
    assert result["compact_reducer"]["strategy_attribution_result"] == {
        "schema_version": "native-strategy-result-v1"
    }
    assert result["compact_reducer"]["second_python_attribution_pass"] is False
    reducer_receipt = Path(
        result["compact_reducer"]["execution_receipt_path"]
    )
    assert reducer_receipt.is_file()
    persisted = json.loads(reducer_receipt.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == (
        "eiel_native_compact_reducer_execution_receipt_v1"
    )
    assert persisted["reducer"]["status"] == "COMPLETED"
    assert "atlas_result" not in persisted["reducer"]


def test_execute_rejects_partial_native_reducer_identity(tmp_path: Path) -> None:
    with pytest.raises(
        ScienceInstrumentOperatorError,
        match="requires artifact, manifest, and digest",
    ):
        execute_materialized_science_batch_v1(
            tmp_path / "unused-materialization",
            batch_index=0,
            runtime_root=tmp_path / "runtime",
            repo_root=tmp_path,
            compact_reducer_native_artifact=tmp_path / "native.so",
        )


def test_compact_reducer_target_selects_one_exact_physical_batch(
    tmp_path: Path,
) -> None:
    worlds = tuple(
        _world(f"generic-world-{index:04d}", seed_offset=index) for index in range(1, 4)
    )
    bundle = ScienceWorldBundleV1(
        bundle_id="GENERIC-COMPACT-BUNDLE",
        compact_source_specs=_payload(
            "eiel_science_compact_source_specs_v1", [{"source": "test"}]
        ),
        compiler_component_sha256=ContentDigest("sha256", "4" * 64),
        worlds=worlds,
    )
    materialization_root = tmp_path / "materialization"
    bundle_info = write_science_world_bundle_v1(
        bundle,
        materialization_root / "world_bundle.json",
    )
    world_ids = [world.world_id for world in worlds[:2]]
    lowered = {
        "science_materialization": {
            "observation_mode": (
                science_operator.SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1
            ),
            "semantic_world_ids": world_ids,
        },
        "physical_batches": [
            {
                "batch_id": "science-batch-0000",
                "cell_ids": world_ids,
            }
        ],
    }
    runtime_root = tmp_path / "runtime"

    target = science_operator._compact_world_atlas_sidecar_target_v1(
        materialization_root,
        {"artifacts": {"world_bundle": bundle_info}},
        lowered,
        runtime_root,
    )

    assert target is not None
    batch_bundle, compact_root, batch_id = target
    assert tuple(world.world_id for world in batch_bundle.worlds) == tuple(world_ids)
    assert batch_bundle.bundle_content_sha256() != bundle.bundle_content_sha256()
    assert batch_id == "science-batch-0000"
    assert compact_root == (
        runtime_root / "batches" / "science-batch-0000" / "science-compact"
    )


def test_execute_reports_compact_reducer_tail_slo_miss_after_front_door(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, context_path, base_path, _ = _materialize_inputs(tmp_path)
    materialization = materialize_science_instrument_v1(
        request_path,
        context_path,
        base_path,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=5_000,
    )
    events: list[str] = []

    class TimeoutSidecar:
        pid = 4321

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            events.append("sidecar-started")

        def result(self, *, timeout_seconds: float) -> dict[str, Any]:
            assert timeout_seconds == 0.01
            events.append("sidecar-tail-wait")
            raise science_operator.CompactWorldAtlasReducerError(
                "async compact reducer result timed out"
            )

        def cancel(self) -> None:
            events.append("sidecar-cancelled")

    monkeypatch.setattr(
        science_operator,
        "_compact_world_atlas_sidecar_target_v1",
        lambda *args: (
            object(),
            tmp_path / "runtime" / "science-compact",
            "science-batch-0000",
        ),
    )
    monkeypatch.setattr(
        science_operator,
        "CompactWorldAtlasAsyncSidecarV1",
        TimeoutSidecar,
    )

    def fake_front_door(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        events.append("front-door-returned")
        return {
            "ok": True,
            "status": "completed",
            "result_class": "SUCCESS",
        }

    result = execute_materialized_science_batch_v1(
        materialization["receipt_path"],
        batch_index=0,
        runtime_root=tmp_path / "runtime",
        repo_root=tmp_path,
        crca_adapter=object(),
        execution_front_door=fake_front_door,
        compact_reducer_workers=2,
        compact_reducer_tail_seconds=0.01,
    )

    assert events == [
        "sidecar-started",
        "front-door-returned",
        "sidecar-tail-wait",
        "sidecar-cancelled",
    ]
    assert result["front_door_result"]["ok"] is True
    assert result["ok"] is False
    assert result["status"] == "compact_reducer_tail_timeout"
    assert result["result_class"] == "REDUCER_TAIL_SLO_MISS"
    assert result["compact_reducer"]["tail_slo_met"] is False


def test_execute_tail_fallback_preserves_configured_reducer_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, context_path, base_path, _ = _materialize_inputs(tmp_path)
    materialization = materialize_science_instrument_v1(
        request_path,
        context_path,
        base_path,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=5_000,
    )
    seen: dict[str, Any] = {}

    class TimeoutSidecar:
        pid = 4321

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def result(self, *, timeout_seconds: float) -> dict[str, Any]:
            assert timeout_seconds == 0.01
            raise science_operator.CompactWorldAtlasReducerError(
                "async compact reducer result timed out"
            )

        def cancel(self) -> None:
            seen["cancelled"] = True

    class FakeReducer:
        strategy_attribution_result = None

        def __init__(
            self,
            bundle: object,
            *,
            max_workers: int,
            native_artifact: object,
            output_root: Path,
        ) -> None:
            del bundle, native_artifact, output_root
            seen["max_workers"] = max_workers

        def finalize(self, capture_index: Path) -> dict[str, str]:
            del capture_index
            return {"schema_version": "synthetic-atlas-result-v1"}

        def close(self) -> None:
            seen["closed"] = True

    monkeypatch.setattr(
        science_operator,
        "_compact_world_atlas_sidecar_target_v1",
        lambda *args: (
            object(),
            tmp_path / "runtime" / "science-compact",
            "science-batch-0000",
        ),
    )
    monkeypatch.setattr(
        science_operator,
        "CompactWorldAtlasAsyncSidecarV1",
        TimeoutSidecar,
    )
    monkeypatch.setattr(
        science_operator,
        "load_native_compact_reducer_v1",
        lambda *args, **kwargs: "native-binding",
    )
    monkeypatch.setattr(
        science_operator,
        "CompactWorldAtlasIncrementalReducerV1",
        FakeReducer,
    )

    def fake_front_door(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {"ok": True, "status": "completed", "result_class": "SUCCESS"}

    result = execute_materialized_science_batch_v1(
        materialization["receipt_path"],
        batch_index=0,
        runtime_root=tmp_path / "runtime",
        repo_root=tmp_path,
        crca_adapter=object(),
        execution_front_door=fake_front_door,
        compact_reducer_workers=3,
        compact_reducer_tail_seconds=0.01,
    )

    assert seen == {"max_workers": 3, "cancelled": True, "closed": True}
    assert result["ok"] is True
    assert result["compact_reducer"]["status"] == (
        "COMPLETED_AFTER_TAIL_FALLBACK"
    )
    assert result["compact_reducer"]["fallback_worker_count"] == 3


def test_execute_reducer_failure_is_reported_after_independent_producer_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, context_path, base_path, _ = _materialize_inputs(tmp_path)
    materialization = materialize_science_instrument_v1(
        request_path,
        context_path,
        base_path,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=5_000,
    )
    events: list[str] = []

    class FailedSidecar:
        pid = 9917

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            events.append("sidecar-started")

        def result_bundle(self, *, timeout_seconds: float) -> dict[str, Any]:
            assert timeout_seconds == 60.0
            events.append("reducer-failure-observed")
            raise science_operator.CompactWorldAtlasReducerError(
                "async compact reducer failed with malformed sealed input"
            )

        def cancel(self) -> None:
            events.append("sidecar-cleaned")

        def progress_snapshot(self) -> dict[str, Any]:
            return {
                "phase": "FAILED",
                "process_cleanup": {
                    "clean": True,
                    "orphaned_worker_count": 0,
                },
            }

    monkeypatch.setattr(
        science_operator,
        "_compact_world_atlas_sidecar_target_v1",
        lambda *args: (
            object(),
            tmp_path / "runtime" / "science-compact",
            "science-batch-0000",
        ),
    )
    monkeypatch.setattr(
        science_operator,
        "CompactWorldAtlasAsyncSidecarV1",
        FailedSidecar,
    )

    def fake_front_door(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        events.append("front-door-completed")
        return {
            "ok": True,
            "status": "completed",
            "result_class": "SUCCESS",
            "producer_receipt": "unchanged",
        }

    result = execute_materialized_science_batch_v1(
        materialization["receipt_path"],
        batch_index=0,
        runtime_root=tmp_path / "runtime",
        repo_root=tmp_path,
        crca_adapter=object(),
        execution_front_door=fake_front_door,
        compact_reducer_workers=4,
    )

    assert events == [
        "sidecar-started",
        "front-door-completed",
        "reducer-failure-observed",
        "sidecar-cleaned",
    ]
    assert result["front_door_result"] == {
        "ok": True,
        "status": "completed",
        "result_class": "SUCCESS",
        "producer_receipt": "unchanged",
    }
    assert result["ok"] is False
    assert result["status"] == "compact_reducer_failed"
    assert result["result_class"] == "REDUCER_ERROR"
    assert result["compact_reducer"]["operational_metrics"] == {
        "phase": "FAILED",
        "process_cleanup": {"clean": True, "orphaned_worker_count": 0},
    }


def test_execute_rejects_missing_cuda_v2_baseline_delta_before_front_door(
    tmp_path: Path,
) -> None:
    request_path, context_path, base_path, _ = _materialize_inputs(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["baseline"]["deviations"] = []
    _write_json(request_path, request)
    materialization = materialize_science_instrument_v1(
        request_path,
        context_path,
        base_path,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=5_000,
    )

    with pytest.raises(ScienceInstrumentOperatorError, match="exact five explicit"):
        execute_materialized_science_batch_v1(
            materialization["receipt_path"],
            batch_index=0,
            runtime_root=tmp_path / "runtime",
            repo_root=tmp_path,
            crca_adapter=object(),
            execution_front_door=lambda *args, **kwargs: pytest.fail(
                "front door must not run"
            ),
        )


def test_execute_derives_acs2_conformant_baseline_delta_from_bound_plan(
    tmp_path: Path,
) -> None:
    request_path, context_path, base_path, _ = _materialize_inputs(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["baseline"]["deviations"] = [
        row
        for row in request["baseline"]["deviations"]
        if row["requirement_id"] != "checkpoint.acs_two_slot"
    ]
    _write_json(request_path, request)
    base_plan = json.loads(base_path.read_text(encoding="utf-8"))
    base_plan["execution"]["acs_checkpoint_slots"] = 2
    base_plan.pop("plan_sha256")
    base_plan["plan_sha256"] = canonical_sha256(base_plan).hex
    _write_json(base_path, base_plan)
    materialization = materialize_science_instrument_v1(
        request_path,
        context_path,
        base_path,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=5_000,
    )
    observed: dict[str, Any] = {}

    def fake_front_door(
        request: Any, plan: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        del kwargs
        observed["request"] = request
        observed["plan"] = plan
        return {"ok": True, "status": "completed", "result_class": "SUCCESS"}

    result = execute_materialized_science_batch_v1(
        materialization["receipt_path"],
        batch_index=0,
        runtime_root=tmp_path / "runtime",
        repo_root=tmp_path,
        crca_adapter=object(),
        execution_front_door=fake_front_door,
    )

    assert result["ok"] is True
    assert observed["plan"]["execution"]["acs_checkpoint_slots"] == 2
    assert tuple(
        row.requirement_id for row in observed["request"].baseline.deviations
    ) == tuple(
        requirement_id
        for requirement_id in (
            CUDA_FULL_EVOLUTION_V2_BASELINE_DEVIATION_REQUIREMENT_IDS_V1
        )
        if requirement_id != "checkpoint.acs_two_slot"
    )


def test_execute_front_door_failure_is_propagated_and_cli_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path, context_path, base_path, _ = _materialize_inputs(tmp_path)
    materialization = materialize_science_instrument_v1(
        request_path,
        context_path,
        base_path,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=5_000,
    )

    def failing_front_door(
        request: Any, plan: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        del request, plan, kwargs
        return {
            "ok": False,
            "status": "failed",
            "result_class": "PREFLIGHT_ERROR",
        }

    result = execute_materialized_science_batch_v1(
        materialization["receipt_path"],
        batch_index=0,
        runtime_root=tmp_path / "runtime",
        repo_root=tmp_path,
        crca_adapter=object(),
        execution_front_door=failing_front_door,
    )
    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["result_class"] == "PREFLIGHT_ERROR"
    assert result["front_door_result"]["ok"] is False

    monkeypatch.setattr(
        science_operator,
        "execute_materialized_science_batch_v1",
        lambda *args, **kwargs: result,
    )
    assert (
        science_operator.main(
            [
                "execute",
                "--materialization",
                str(materialization["receipt_path"]),
                "--batch-index",
                "0",
                "--runtime-root",
                str(tmp_path / "runtime-cli"),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 2
    )


def test_exploratory_lowering_requires_exact_route_identity_digests(
    tmp_path: Path,
) -> None:
    request_path, context_path, base_path, _ = _materialize_inputs(tmp_path)
    materialization = materialize_science_instrument_v1(
        request_path,
        context_path,
        base_path,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=5_000,
    )
    bundle = load_science_world_bundle_v1(
        tmp_path / "materialization" / "world_bundle.json"
    )
    raw_admission = materialization["bundle_admission"]
    admission = ScienceWorldBundleAdmissionV1(
        bundle_content_sha256=ContentDigest(
            raw_admission["bundle_content_sha256"]["algorithm"],
            raw_admission["bundle_content_sha256"]["hex"],
        ),
        admitted_world_ids=tuple(raw_admission["admitted_world_ids"]),
        findings=(),
    )
    carrier = CudaV2ScienceCarrierFamilyAdapterV1()
    carrier_plan = carrier.plan(bundle, admission)
    with pytest.raises(ScienceWorldBundleError, match="exact instrument manifest"):
        # This direct call confirms the carrier's exploratory contract is
        # fail-closed when the route identities are not supplied.
        carrier.lower_all(
            _base_plan(),
            bundle,
            admission,
            carrier_plan,
            observation_mode=SCIENCE_CUDA_V2_OBSERVATION_TYPED_SIDECAR_V1,
            science_use_class=SCIENCE_CUDA_V2_USE_CLASS_EXPLORATORY_SCIENCE_V1,
        )


def test_materialization_input_mutation_does_not_modify_helper_fixture() -> None:
    request, context = _science_request_and_context()
    copy = deepcopy(request)
    copy["science_campaign"]["question"]["science_ownership"]["study_id"] = "other"
    assert request["science_campaign"]["question"]["science_ownership"]["study_id"] == (
        "MUTSUP-OPERATOR-TEST"
    )
    assert context.instrument_manifest.manifest_sha256().hex


def _digest_from_dict(value: dict[str, str]) -> ContentDigest:
    return ContentDigest(value["algorithm"], value["hex"])


def _synthetic_native_mutsup_rows(
    cell: dict[str, Any], *, mutation_enabled: bool, terminal_tick: int
):
    native_control = dict(cell["native_control"])
    control_value = {
        "binding_id": f"science-control-{cell['cell_id']}",
        "semantic_cell_id": cell["cell_id"],
        "seed_bindings": dict(cell["science_seed_bindings"]),
        "control_values": list(cell["science_semantic_inputs"]["control_bindings"]),
        "control_schedules": dict(cell["science_semantic_inputs"]["time"]),
        "registered_profile_ids": [cell["science_acquisition"]["profile_id"]],
        "semantic_envelope_resolution_sha256": canonical_sha256(
            cell["science_semantic_inputs"]
        ).to_dict(),
        "semantic_applied_readback_sha256": canonical_sha256(
            {"cell_id": cell["cell_id"], "terminal_tick": terminal_tick}
        ).to_dict(),
        "continuation_parent_execution_identity": None,
        "applied_native_control": native_control,
    }
    control_payload = CanonicalPayloadV1.from_value(
        SCIENCE_NATIVE_CONTROL_BINDING_PAYLOAD_SCHEMA_VERSION,
        control_value,
    )
    control_digest = canonical_sha256(control_value)

    def payload(kind: ScienceRecordKindV1, **overrides: object) -> CanonicalPayloadV1:
        value = {field: None for field in _RECORD_REQUIRED_PAYLOAD_FIELDS[kind]}
        value.update(overrides)
        return CanonicalPayloadV1.from_value(
            SchemaVersion("mutsup_operator_fixture_payload_v1"), value
        )

    def draft(
        kind: ScienceRecordKindV1,
        tick: int,
        phase: int,
        address: str,
        row_payload: CanonicalPayloadV1,
    ) -> ScienceObservationDraftV1:
        return ScienceObservationDraftV1(
            producer_component_id=(
                f"cuda_full_evolution_v2_science_capture_v1_phase_{phase:02d}"
            ),
            producer_contract_version=SchemaVersion(
                "eiel_cuda_full_evolution_v2_science_capture_v1"
            ),
            producer_phase_registry_ordinal=phase,
            absolute_tick=tick,
            semantic_source_address=f"world/{cell['cell_id']}/{address}",
            record_kind=kind,
            record_kind_version=1,
            control_binding_sha256=control_digest,
            payload=row_payload,
        )

    drafts = [
        draft(
            ScienceRecordKindV1.CONTROL_BINDING,
            0,
            5,
            "control-binding",
            control_payload,
        )
    ]
    if mutation_enabled:
        drafts.extend(
            (
                draft(
                    ScienceRecordKindV1.BIRTH_ATTEMPT,
                    1,
                    20,
                    "birth/20/attempt",
                    payload(
                        ScienceRecordKindV1.BIRTH_ATTEMPT,
                        birth_attempt_id="birth-attempt-20",
                        opportunity_id="birth-opportunity-20",
                        parent_agent_ids=[10, 11],
                        parent_genome_state_ids=["p0", "p1"],
                        birth_mode_id="ordinary_birth",
                        inheritance_operator_id="fixture_inheritance",
                        recombination_operator_id="fixture_recombination",
                        attempt_result_class="COMMITTED",
                        reason_code="fixture_committed",
                    ),
                ),
                draft(
                    ScienceRecordKindV1.BIRTH_COMMITTED,
                    1,
                    21,
                    "birth/20",
                    payload(
                        ScienceRecordKindV1.BIRTH_COMMITTED,
                        birth_attempt_id="birth-attempt-20",
                        opportunity_id="birth-opportunity-20",
                        child_agent_id=20,
                        parent_agent_ids=[10, 11],
                        parent_genome_state_ids=["p0", "p1"],
                        parent_hereditary_package_state_ids=[],
                        child_pre_mutation_genome_state_id="g0",
                        child_post_birth_genome_state_id="g1",
                        child_hereditary_package_state_ids=[],
                        child_linkage_state_id="link-20",
                        founder_id="founder-a",
                        lineage_id="lineage-20",
                        lineage_depth=1,
                        birth_mode_id="ordinary_birth",
                        inheritance_operator_id="fixture_inheritance",
                        recombination_operator_id="fixture_recombination",
                    ),
                ),
                draft(
                    ScienceRecordKindV1.MUTATION_OUTCOME,
                    1,
                    22,
                    "mutation/20",
                    payload(
                        ScienceRecordKindV1.MUTATION_OUTCOME,
                        mutation_opportunity_id="mutation-opportunity-20",
                        mutation_attempt_id="mutation-attempt-20",
                        subject_agent_id=20,
                        origin_agent_id=10,
                        origin_absolute_tick=1,
                        attempted=True,
                        result_class="MUTATION_APPLIED",
                        operator_id="genome_v2_mutation",
                        mutation_class_id="fixture_point_mutation",
                        affected_coordinates={"site": 7},
                        pre_value_or_state_digest=canonical_sha256("g0").to_dict(),
                        post_value_or_state_digest=canonical_sha256("g1").to_dict(),
                        bounded_delta={"site": 7},
                        pre_genome_state_id="g0",
                        post_genome_state_id="g1",
                        reason_code="fixture_mutation",
                    ),
                ),
                draft(
                    ScienceRecordKindV1.BIRTH_ATTEMPT,
                    3,
                    20,
                    "birth/40/attempt",
                    payload(
                        ScienceRecordKindV1.BIRTH_ATTEMPT,
                        birth_attempt_id="birth-attempt-40",
                        opportunity_id="birth-opportunity-40",
                        parent_agent_ids=[20, 30],
                        parent_genome_state_ids=["g1", "p2"],
                        birth_mode_id="ordinary_birth",
                        inheritance_operator_id="fixture_inheritance",
                        recombination_operator_id="fixture_recombination",
                        attempt_result_class="COMMITTED",
                        reason_code="fixture_committed",
                    ),
                ),
                draft(
                    ScienceRecordKindV1.BIRTH_COMMITTED,
                    3,
                    21,
                    "birth/40",
                    payload(
                        ScienceRecordKindV1.BIRTH_COMMITTED,
                        birth_attempt_id="birth-attempt-40",
                        opportunity_id="birth-opportunity-40",
                        child_agent_id=40,
                        parent_agent_ids=[20, 30],
                        parent_genome_state_ids=["g1", "p2"],
                        parent_hereditary_package_state_ids=[],
                        child_pre_mutation_genome_state_id="g1",
                        child_post_birth_genome_state_id="g1",
                        child_hereditary_package_state_ids=[],
                        child_linkage_state_id="link-40",
                        founder_id="founder-a",
                        lineage_id="lineage-40",
                        lineage_depth=2,
                        birth_mode_id="ordinary_birth",
                        inheritance_operator_id="fixture_inheritance",
                        recombination_operator_id="fixture_recombination",
                    ),
                ),
            )
        )
    context = NativeCudaV2ObservationContextV1(
        instrument_manifest_sha256=_digest_from_dict(
            cell["science_instrument_manifest_sha256"]
        ),
        crca_semantic_run_digest=_digest_from_dict(
            cell["science_crca_semantic_run_digest"]
        ),
        science_semantic_acquisition_sha256=_digest_from_dict(
            cell["science_acquisition_sha256"]
        ),
        cell_id=cell["cell_id"],
        logical_world_id=cell["science_logical_world_id"],
        semantic_world_key=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_science_semantic_world_key_v1"),
            cell["science_semantic_world_key"],
        ),
    )
    prefix = finalize_science_observation_drafts_v1(
        tuple(drafts),
        instrument_manifest_sha256=context.instrument_manifest_sha256,
        crca_semantic_run_digest=context.crca_semantic_run_digest,
        science_semantic_acquisition_sha256=(
            context.science_semantic_acquisition_sha256
        ),
        cell_id=context.cell_id,
        logical_world_id=context.logical_world_id,
        semantic_world_key=context.semantic_world_key,
    )
    completion = finalize_native_cuda_v2_observations_v1(
        build_native_cuda_v2_completion_output_v1(
            cell_id=cell["cell_id"],
            absolute_tick=terminal_tick,
            completion_class="TERMINAL_HORIZON",
            terminal_state_sha256=canonical_sha256(
                {"cell_id": cell["cell_id"], "tick": terminal_tick}
            ).to_dict(),
            terminal_population_count=256,
            terminal_record_count=len(prefix),
            canonical_record_stream_sha256=(
                canonical_science_record_stream_sha256_v1(prefix).to_dict()
            ),
            checkpoint_occurrence_ids=[f"checkpoint-{terminal_tick:012d}"],
            control_binding_sha256=control_digest.to_dict(),
        ),
        context=context,
        start_world_local_sequence=len(prefix),
    )
    return (*prefix, *completion)


def _synthetic_completed_batches(
    root: Path, materialization: dict[str, Any], *, terminal_tick: int
) -> list[Path]:
    receipts: list[Path] = []
    for batch_index, lowered in enumerate(materialization["lowered_batches"]):
        batch = lowered["physical_batches"][0]
        rows_by_cell = {}
        for cell in lowered["cells"]:
            enabled = cell["native_control"]["control_vector"]["mutation.enabled"]
            rows_by_cell[cell["cell_id"]] = _synthetic_native_mutsup_rows(
                cell,
                mutation_enabled=enabled,
                terminal_tick=terminal_tick,
            )
        frontier = cuda_provider._science_frontier_v1(
            rows_by_cell, next_absolute_tick=terminal_tick
        )
        sidecar_body = {
            "schema_version": cuda_provider.SCIENCE_OBSERVATION_SIDECAR_SCHEMA_V1,
            "observation_abi": cuda_provider.SCIENCE_OBSERVATION_ABI_V1,
            "observation_mode": cuda_provider.SCIENCE_OBSERVATION_TYPED_SIDECAR_V1,
            "batch_id": batch["batch_id"],
            "cell_ids": list(batch["cell_ids"]),
            "records_by_cell": {
                cell_id: [row.to_dict() for row in rows_by_cell[cell_id]]
                for cell_id in batch["cell_ids"]
            },
            "frontier": frontier,
        }
        sidecar_path = root / f"sidecar-{batch_index}.json"
        sidecar_payload = canonical_json_bytes(sidecar_body)
        sidecar_path.write_bytes(sidecar_payload)
        terminal_frontier = {
            **frontier,
            "records_path": str(sidecar_path),
            "records_sha256": hashlib.sha256(sidecar_payload).hexdigest(),
            "terminal": True,
        }
        manifest = {
            "schema_version": "eiel_campaign_batch_manifest_v1",
            "status": "completed",
            "batch_id": batch["batch_id"],
            "cell_ids": list(batch["cell_ids"]),
            "plan_sha256": lowered["plan_sha256"],
            "execution_route_sha256": f"{batch_index + 1}" * 64,
            "result": {
                "technical_only": True,
                "formal_qualification_authority": False,
                "time_authority": "absolute_tick_v1",
                "science_observation_frontier": terminal_frontier,
                "science_observation_record_counts": {
                    key: len(value) for key, value in rows_by_cell.items()
                },
            },
        }
        manifest_path = root / f"batch-{batch_index}.json"
        _write_json(manifest_path, manifest)
        receipts.append(manifest_path)
    return receipts


def test_reduce_mutsup_consumes_sealed_batches_and_proves_restart_equivalence(
    tmp_path: Path,
) -> None:
    context = materialize_w32_cuda_v2_science_host_context_v1(
        total_ticks=160,
        checkpoint_ticks=(160,),
    )
    request = build_mutsup_w32_paired_seed_request_v1(
        context,
        total_ticks=160,
        checkpoint_ticks=(160,),
        observation_ticks=(0, 80, 160),
    )
    base_plan = build_w32_cuda_v2_mutsup_base_plan_v1(
        context,
        total_ticks=160,
        checkpoint_ticks=(160,),
    )
    materialization = materialize_science_instrument_v1(
        request,
        serialize_science_host_context_v1(context),
        base_plan,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=160,
        physical_batch_capacity=3,
        temporal_k=1,
    )
    receipts = _synthetic_completed_batches(
        tmp_path, materialization, terminal_tick=160
    )

    result = reduce_mutsup_science_instrument_v1(
        materialization["receipt_path"],
        list(reversed(receipts)),
        output_directory=tmp_path / "reduction",
    )

    assert result["status"] == "COMPLETED"
    assert result["world_count"] == 6
    assert result["pair_count"] == 3
    assert result["reducer_restart_equivalent"] is True
    aggregate = result["aggregate_result"]["payload"]
    assert aggregate["decision"]["classification"] == "DESCRIPTIVE_PAIRED_OFF_ON"
    statuses = {
        world["arm_role"]: world["mutation_opportunity"]["status"]
        for world in aggregate["worlds"]
    }
    assert statuses == {
        "OFF": "MUTATION_DISABLED_CONTROL",
        "ON": "APPLIED_MUTATIONS_OBSERVED",
    }
    assert all(
        pair["metrics"]["direct_reproductive_retention_count"]
        == {
            "off": 0,
            "on": 1,
            "on_minus_off": 1,
        }
        for pair in aggregate["paired_contrasts"]
    )
    assert Path(result["receipt_path"]).is_file()


def test_reduce_mutsup_rejects_receipt_for_another_lowered_plan(
    tmp_path: Path,
) -> None:
    context = materialize_w32_cuda_v2_science_host_context_v1(
        total_ticks=160,
        checkpoint_ticks=(160,),
    )
    request = build_mutsup_w32_paired_seed_request_v1(
        context,
        total_ticks=160,
        checkpoint_ticks=(160,),
        observation_ticks=(0, 80, 160),
    )
    base_plan = build_w32_cuda_v2_mutsup_base_plan_v1(
        context,
        total_ticks=160,
        checkpoint_ticks=(160,),
    )
    materialization = materialize_science_instrument_v1(
        request,
        serialize_science_host_context_v1(context),
        base_plan,
        output_directory=tmp_path / "materialization",
        maximum_total_ticks=160,
        physical_batch_capacity=3,
        temporal_k=1,
    )
    receipts = _synthetic_completed_batches(
        tmp_path, materialization, terminal_tick=160
    )
    manifest = json.loads(receipts[0].read_text(encoding="utf-8"))
    manifest["plan_sha256"] = "0" * 64
    _write_json(receipts[0], manifest)

    with pytest.raises(ScienceInstrumentOperatorError, match="plan identity differs"):
        reduce_mutsup_science_instrument_v1(
            materialization["receipt_path"],
            receipts,
            output_directory=tmp_path / "reduction",
        )
