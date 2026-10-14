from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch
import yaml

from eiel_composition import cuda_full_evolution_v2_launchpad_adapter as adapter_module
from eiel_composition import science_cuda_v2_carrier_v1 as science_carrier
from eiel_composition import workhorse_v2_materialize as materializer_module
from eiel_composition.cuda_full_evolution_v2_launchpad_adapter import (
    CanonicalCudaFullEvolutionV2LaunchpadAdapterV1,
    CudaFullEvolutionV2LaunchpadAdapterError,
)
from emergence_contracts import (
    ScienceAdmissionOutcomeV1,
    canonical_sha256 as crca_sha256,
)
from emergence_contracts.science_request_v1 import validate_eiel_science_request_v1
from engine.campaigns import cuda_full_evolution_v2_workhorse_provider as provider_module
from engine.campaigns import cuda_full_evolution_v2_acs_adapter as acs_adapter_module
from engine.campaigns import execution_front_door_v1 as front_door
from engine.campaigns.canonical import canonical_sha256 as campaign_sha256
from engine.campaigns.compiler import CampaignError, compile_campaign
from engine.campaigns.cuda_full_evolution_v2_workhorse_provider import (
    DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2,
    INDEXED_ACS_CONTINUATION_AGGREGATE_WTPS_METRIC_V2,
    INDEXED_ACS_CONTINUATION_OPERATION_V2,
    OPERATION_EXTENSION_ID,
    OPERATION_SCHEMA_V2,
    OPERATION_SCHEMA_V3,
    RUNTIME_HEADROOM_REJECTION_CODE_V2,
    SEMANTIC_EVIDENCE_CASE_OPERATION_V2,
    WORLD_BASE,
    CampaignProviderError,
    cuda_full_evolution_workhorse_v2_profile_registry_v1,
    cuda_full_evolution_workhorse_v2_provider_registry_v1,
)
from engine.campaigns.execution_front_door_v1 import ExecutionRequestV2
from engine.campaigns.cuda_full_evolution_v2_workhorse_resources import (
    workhorse_checkpoint_storage_accounting_v2,
)
from engine.campaigns.providers import (
    ACS_NATIVE_RUNTIME_ABI_V1,
    ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1,
    ACS_NATIVE_RUNTIME_EXTENSION_KIND_V1,
    ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1,
    ACS_NATIVE_RUNTIME_MANIFEST_KIND_V1,
    ACS_NATIVE_RUNTIME_MODULE_NAME_V1,
    ACS_NATIVE_RUNTIME_VERSION_V1,
)
from engine.perf.cuda_full_evolution_v2.workhorse import canonical_rows
from engine.perf.cuda_full_evolution_v2.workhorse import authority as workhorse_authority
from engine.perf.cuda_full_evolution_v2.workhorse import probe as workhorse_probe
from engine.perf.cuda_full_evolution_v2.workhorse.canonical_rows import (
    WorkhorseCanonicalRowError,
    seal_workhorse_technical_row_receipt_v2,
    validate_workhorse_canonical_parent_authority_binding_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.contract import (
    FullEvolutionCudaWorkhorseContractV2,
    canonical_sha256,
    workhorse_artifact_identity_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.layout import workhorse_layout_v2
from engine.perf.cuda_full_evolution_v2.workhorse.semantic_cases import (
    WORKHORSE_EXTERNAL_CHECKPOINT_AUTHORIZATION_SOURCE_V2,
    WORKHORSE_EXTERNAL_CHECKPOINT_SOURCE_SCHEMA_V2,
    workhorse_science_checkpoint_compatibility_binding_v2,
    workhorse_semantic_case_law_v2,
    workhorse_semantic_case_operation_contract_v2,
)
from engine.qrr.execution_request_v1 import build_execution_request_v1
from eiel_composition.workhorse_v2_materialize import (
    _bind_acs_native_runtime_artifacts,
    _parser as workhorse_materializer_parser,
    _row_checkpoint_ticks_v1,
    _workhorse_wall_seconds_upper_bound_v2,
)
from eiel_composition.science_workhorse_selected_host_context_v1 import (
    WorkhorseSelectedHostContextError,
    materialize_workhorse_selected_host_context_v1,
)
from eiel_composition.science_cuda_v2_carrier_v1 import (
    ScienceWorldBundleError,
    _selected_workhorse_control_values_v1,
)
from eiel_composition.science_instrument_operator_v1 import (
    _engineering_bundle_admission,
)
from eiel_composition.science_request_v1_compiler import compile_science_request_v1
from tests.test_eiel_science_request_v1_compiler import _selected_world_request


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / (
    "_spec/campaigns/v1/tw_cuda_full_evolution_v2/"
    "micro_qualprep_v1.template.yaml"
)
T0 = "2026-08-26T03:00:00Z"
T1 = "2026-08-26T03:00:01Z"
T2 = "2026-08-26T03:00:02Z"
T3 = "2026-08-26T03:00:03Z"


def test_workhorse_materializer_covers_k64_provider_wall_bound() -> None:
    assert _workhorse_wall_seconds_upper_bound_v2(1) == 300
    assert _workhorse_wall_seconds_upper_bound_v2(64) == 3840
    with pytest.raises(ValueError, match="positive integer"):
        _workhorse_wall_seconds_upper_bound_v2(0)


def test_workhorse_row_exact_founder_bootstrap_is_explicit_and_two_checkpoint() -> None:
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id="w64h64p1024c2048",
        backend_id="phase_graph_v2",
        B=64,
        temporal_k=8,
        total_ticks=40,
        start_next_tick=0,
    )
    assert _row_checkpoint_ticks_v1(
        contract,
        exact_founder_bootstrap=False,
        acs_slots=0,
        checkpoint_retention_class="R2",
    ) == (40,)
    assert _row_checkpoint_ticks_v1(
        contract,
        exact_founder_bootstrap=True,
        acs_slots=2,
        checkpoint_retention_class="R4",
    ) == (8, 40)

    parsed = workhorse_materializer_parser().parse_args(
        [
            "row",
            "--artifact",
            "/tmp/workhorse.so",
            "--envelope",
            "/tmp/envelope.json",
            "--micro-receipt",
            "/tmp/micro.json",
            "--semantic-gate",
            "/tmp/semantic.json",
            "--B",
            "64",
            "--temporal-k",
            "8",
            "--acs-slots",
            "2",
            "--checkpoint-retention-class",
            "R4",
            "--exact-founder-bootstrap",
            "--control-root",
            "/tmp/control",
            "--run-root",
            "/tmp/run",
            "--campaign-id",
            "founder-bootstrap-v1",
        ]
    )
    assert parsed.exact_founder_bootstrap is True


@pytest.mark.parametrize(
    ("start_next_tick", "acs_slots", "retention", "total_ticks", "match"),
    (
        (1, 2, "R4", 40, "start-next-tick 0"),
        (0, 1, "R4", 40, "ACS2"),
        (0, 2, "R2", 40, "R4"),
        (0, 2, "R4", 8, "terminal tick after first dispatch"),
    ),
)
def test_workhorse_row_exact_founder_bootstrap_fails_closed(
    start_next_tick: int,
    acs_slots: int,
    retention: str,
    total_ticks: int,
    match: str,
) -> None:
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id="w64h64p1024c2048",
        backend_id="phase_graph_v2",
        B=64,
        temporal_k=8,
        total_ticks=total_ticks,
        start_next_tick=start_next_tick,
    )
    with pytest.raises(ValueError, match=match):
        _row_checkpoint_ticks_v1(
            contract,
            exact_founder_bootstrap=True,
            acs_slots=acs_slots,
            checkpoint_retention_class=retention,
        )


def test_materialize_row_emits_exact_founder_schedule_and_storage_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _operation_fixture(tmp_path, monkeypatch, acs_slots=2, B=64)
    operation = fixture["spec"]["analysis"]["extension"]["contract"]
    envelope = deepcopy(fixture["envelope"])
    envelope["receipt_path"] = operation["envelope"]["path"]
    envelope["capacity_basis"] = {
        "planning_available_bytes": 8 * 1024**3,
        "effective_reserve_bytes": 512 * 1024**2,
    }
    monkeypatch.setattr(
        materializer_module,
        "workhorse_artifact_binding_v2",
        lambda _path: deepcopy(fixture["artifact_binding"]),
    )
    monkeypatch.setattr(
        materializer_module,
        "validate_workhorse_device_envelope_calibration_receipt_v2",
        lambda _path: deepcopy(envelope),
    )
    micro = deepcopy(operation["authoritative_micro_receipt"])
    micro["receipt_path"] = micro.pop("path")
    micro["receipt_file_sha256"] = micro.pop("file_sha256")
    micro["receipt_content_sha256"] = micro.pop("content_sha256")
    monkeypatch.setattr(
        materializer_module,
        "validate_authoritative_micro_denominator_v2",
        lambda _path, **_kwargs: deepcopy(micro),
    )
    semantic = deepcopy(operation["semantic_gate_receipt"])
    semantic["receipt_path"] = semantic.pop("path")
    semantic["receipt_file_sha256"] = semantic.pop("file_sha256")
    semantic["semantic_gate_receipt_sha256"] = semantic.pop("content_sha256")
    monkeypatch.setattr(
        materializer_module,
        "validate_workhorse_semantic_gate_receipt_v2",
        lambda _path, **_kwargs: deepcopy(semantic),
    )
    monkeypatch.setattr(
        materializer_module,
        "workhorse_science_checkpoint_compatibility_binding_v2",
        lambda _root: {
            "science_consumer_extension_contract_sha256": (
                operation["science_continuation_contract_sha256"]
            )
        },
    )
    captured: dict[str, Any] = {}

    def capture_outputs(_control_root: Path, **values: Any) -> None:
        captured.update(values)

    monkeypatch.setattr(materializer_module, "_write_outputs", capture_outputs)
    native_extension = fixture["spec"]["artifacts"][
        ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1
    ]["path"]
    materializer_module.materialize_row(
        SimpleNamespace(
            artifact=fixture["artifact_binding"]["path"],
            envelope="envelope.json",
            micro_receipt="micro.json",
            semantic_gate="semantic.json",
            B=64,
            temporal_k=1,
            warmup_ticks=1,
            measurement_window_ticks=1,
            measured_windows=5,
            start_next_tick=0,
            acs_slots=2,
            acs_native_runtime_extension=native_extension,
            exact_founder_bootstrap=True,
            checkpoint_retention_class="R4",
            control_root=str(tmp_path / "bootstrap-control"),
            run_root=str(tmp_path / "bootstrap-run"),
            campaign_id="workhorse-founder-bootstrap-v1",
        )
    )

    spec = captured["spec"]
    assert captured["acs_slots"] == 2
    assert spec["execution"]["checkpoint_ticks"] == [1, 6]
    assert spec["execution"]["technical_gate"]["checkpoint_ticks"] == [1, 6]
    assert spec["artifact_lifecycle"]["checkpoints"] == {
        "authority": "absolute_tick",
        "checkpoint_ticks": [1, 6],
        "roles": [
            {
                "identity": "checkpoint-tick-000000000001",
                "checkpoint_tick": 1,
                "role": "intermediate",
                "retention_class": "R4",
            },
            {
                "identity": "checkpoint-tick-000000000006",
                "checkpoint_tick": 6,
                "role": "terminal",
                "retention_class": "R4",
            },
        ],
    }
    contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
        spec["analysis"]["extension"]["contract"]["contract"]
    )
    schedule = spec["analysis"]["extension"]["contract"]["measurement_schedule"]
    accounting = workhorse_checkpoint_storage_accounting_v2(
        contract=contract,
        checkpoint_ticks=(1, 6),
        acs_checkpoint_slots=2,
        dispatch_count=(
            schedule["warmup_dispatches"] + schedule["measured_dispatches"]
        ),
    )
    storage = spec["artifact_lifecycle"]["storage"]
    assert storage["scratch_max_bytes"] == accounting[
        "acs_pending_capture_file_bytes_upper_bound"
    ]
    assert storage["checkpoint_max_bytes"] == accounting[
        "checkpoint_bytes_upper_bound"
    ]
    bootstrap_operation = spec["analysis"]["extension"]["contract"]
    assert bootstrap_operation["schema_version"] == OPERATION_SCHEMA_V3
    assert bootstrap_operation["exact_founder_bootstrap"] == (
        provider_module.workhorse_exact_founder_bootstrap_contract_v1()
    )

    compiled = compile_campaign(
        spec,
        registry=cuda_full_evolution_workhorse_v2_profile_registry_v1(
            shape_id=contract.shape_id,
            operation="EXECUTION_ROW",
        ),
        planning_resource_accounting=(
            provider_module.cuda_full_evolution_workhorse_v2_signed_planning_resource_accounting_v2
        ),
    ).payload()
    assert "science_materialization" not in compiled
    plan = provider_module.materialize_workhorse_exact_founder_plan_v1(compiled)
    assert plan["science_materialization"]["schema_version"] == (
        provider_module.EXACT_FOUNDER_MATERIALIZATION_SCHEMA_V1
    )
    assert plan["science_materialization"]["world_identity_binding"][
        "world_uids"
    ] == list(range(210_000, 210_064))
    assert plan["science_materialization"]["world_identity_binding"][
        "rng_keys"
    ] == [83] * 64
    assert "world_identity_binding" not in plan["cells"][0][
        "native_world_control"
    ]["initialization"]
    assert plan["cells"][0]["native_world_control"]["seed_bindings"] == {
        "ecology": 101,
        "task": 100001,
    }
    controls = {
        row["control_id"]: row
        for row in plan["cells"][0]["native_world_control"]["control_bindings"]
    }
    assert {
        control_id: (row["resolved_value"], row["schedule"])
        for control_id, row in controls.items()
    } == {
        "environment.resource.multiplier": (1.0, {"kind": "static"}),
        "mutation.rate": (0.125, {"kind": "static"}),
    }
    assert set(plan["cells"][0]["science_world_sha256"]) == {
        "algorithm",
        "hex",
    }
    assert provider_module.materialize_workhorse_exact_founder_plan_v1(plan) == plan
    assert provider_module._exact_founder_runtime_world_identity_v1(plan) == (
        plan["science_materialization"]["world_identity_binding"]
    )
    changed_control = deepcopy(plan)
    changed_control["cells"][0]["native_world_control"]["control_bindings"][0][
        "resolved_value"
    ] = 1.5
    with pytest.raises(CampaignProviderError, match="native_world_control changed"):
        provider_module.materialize_workhorse_exact_founder_plan_v1(
            changed_control
        )
    changed_runtime_identity = deepcopy(plan)
    changed_runtime_identity["science_materialization"]["world_identity_binding"][
        "rng_keys"
    ][0] = 84
    with pytest.raises(CampaignProviderError, match="changed"):
        provider_module.materialize_workhorse_exact_founder_plan_v1(
            changed_runtime_identity
        )

    ordinary = fixture["plan"]
    assert provider_module.materialize_workhorse_exact_founder_plan_v1(
        ordinary
    ) == ordinary
    assert provider_module._exact_founder_runtime_world_identity_v1(ordinary) is None
    ordinary_science_projection = deepcopy(ordinary)
    ordinary_science_projection["cells"][0].update(
        {
            "native_world_control": plan["cells"][0]["native_world_control"],
            "science_world_sha256": plan["cells"][0]["science_world_sha256"],
        }
    )
    assert provider_module.materialize_workhorse_exact_founder_plan_v1(
        ordinary_science_projection
    ) == ordinary_science_projection

    spec_path = tmp_path / "exact-founder-campaign.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    request = ExecutionRequestV2.from_mapping(
        {
            "schema_version": "eiel_execution_request_v2",
            "purpose": "benchmark",
            "campaign_spec": str(spec_path),
            "output_root": str(tmp_path / "canonical-bootstrap-run"),
            "resource_wait": {
                "enabled": True,
                "poll_seconds": 300,
                "max_wait_seconds": None,
            },
            "qualification_profile": None,
            "baseline": materializer_module._baseline(acs_slots=2),
        }
    )
    canonical_plan, _source, _root, _explicit_lifecycle = (
        front_door._resolve_compiled_plan(request, repo_root=REPO_ROOT)
    )
    assert canonical_plan["science_materialization"] == (
        plan["science_materialization"]
    )
    assert canonical_plan["cells"][0]["native_world_control"] == (
        plan["cells"][0]["native_world_control"]
    )
    assert canonical_plan["cells"][0]["science_world_sha256"] == (
        plan["cells"][0]["science_world_sha256"]
    )


def test_workhorse_materializer_binds_exact_native_acs_runtime(
    tmp_path: Path,
) -> None:
    extension = tmp_path / "eiel_native_runtime_ext.so"
    manifest = extension.with_suffix(extension.suffix + ".manifest.json")
    extension.write_bytes(b"native-acs-runtime")
    manifest.write_text("{}\n", encoding="utf-8")
    spec: dict[str, Any] = {"artifacts": {}}

    _bind_acs_native_runtime_artifacts(
        spec,
        acs_slots=2,
        extension_path=str(extension),
    )

    assert spec["artifacts"][ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1] == {
        "path": str(extension.resolve()),
        "sha256": _file_sha256(extension),
        "kind": ACS_NATIVE_RUNTIME_EXTENSION_KIND_V1,
        "abi": ACS_NATIVE_RUNTIME_ABI_V1,
    }
    assert spec["artifacts"][ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1][
        "sha256"
    ] == _file_sha256(manifest)
    with pytest.raises(ValueError, match="requires --acs-native"):
        _bind_acs_native_runtime_artifacts(
            {"artifacts": {}},
            acs_slots=1,
            extension_path=None,
        )
    with pytest.raises(ValueError, match="requires --acs-slots"):
        _bind_acs_native_runtime_artifacts(
            {"artifacts": {}},
            acs_slots=0,
            extension_path=str(extension),
        )


def test_no_execution_science_authority_base_plan_is_science_compact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _operation_fixture(tmp_path, monkeypatch, acs_slots=2, B=1)
    spec = deepcopy(fixture["spec"])
    spec["artifact_lifecycle"]["lifecycle_profile"] = "SCIENCE_COMPACT"
    control_root = tmp_path / "authority-control"
    control_root.mkdir()

    plan_path = materializer_module._write_science_authority_base_plan_v1(
        control_root,
        spec=spec,
        shape_id="w64h64p1024c2048",
    )

    plan = json.loads(plan_path.read_bytes())
    assert plan["artifact_lifecycle"]["lifecycle_profile"] == "SCIENCE_COMPACT"
    assert plan["execution"]["acs_checkpoint_slots"] == 2
    assert plan["execution"]["native_streamed_checkpoint"] is True
    assert plan["analysis"]["extension"]["contract"]["operation"] == (
        "EXECUTION_ROW"
    )
    assert (control_root / "campaign.yaml").is_file()
    assert not (control_root / "execution-request-v2.json").exists()

    profile = next(
        row
        for row in science_carrier.SCIENCE_CUDA_V2_PROFILES
        if row.native_shape_id == "w64h64p1024c2048"
    )
    context = materialize_workhorse_selected_host_context_v1(
        plan,
        profile=profile,
        logical_world_capacity=1,
        context_scope_id="test-workhorse-selected-host-context-v1",
    )
    assert context.route_adapter.maximum_batch_cells == 1
    assert context.route_adapter.acs_checkpoint_slots == 2
    assert context.physical_execution_core.to_value()["carrier_profile_id"] == (
        "w64x64p1024c2048_gpuv2"
    )
    assert [row.control_id for row in context.control_descriptors] == [
        "environment.resource.multiplier",
        "mutation.rate",
    ]
    assert all(
        row.schedule_kinds
        == ("piecewise_constant_absolute_ticks", "static")
        for row in context.control_descriptors
    )
    assert {
        capability
        for capability in context.available_scientific_capability_ids
        if capability.startswith("schedule:")
    } == {
        "schedule:piecewise_constant_absolute_ticks",
        "schedule:static",
    }
    assert set(context.available_observation_capability_ids) == {
        "birth_count",
        "control_readback",
        "death_count",
        "genome_identity",
        "mutation_count",
        "population_count",
        "total_event_count",
    }
    assert not (
        {
            "birth",
            "death",
            "causal_lifecycle",
            "compact.agent_lifetime_metrics",
            "compact.birth_provenance",
            "compact.genome_reconstruction",
            "compact.lifecycle_events",
            "compact.state_snapshots",
        }
        & set(context.available_observation_capability_ids)
    )

    selected_request = _selected_world_request()
    selected_request["world"] = {
        "width": 64,
        "height": 64,
        "population": 1024,
        "capacity": 2048,
    }
    selected_request["run"] = {
        "ticks": 8,
        "batch": {"policy": "exact", "worlds": 1},
        "checkpoint_ticks": [8],
    }
    selected_request["observe"] = {
        "profile": "SCIENCE_STANDARD_V1",
        "events": ["total_event_count"],
        "state": ["genome_identity", "population_count"],
        "runtime_counters": [
            "birth_count",
            "death_count",
            "mutation_count",
        ],
        "snapshot_ticks": [6, 8],
    }
    selected_request["continuation"]["state_ref"]["next_absolute_tick"] = 6
    selected_request["world_template"]["controls"] = {
        "environment.resource.multiplier": {
            "value": 1.5,
            "schedule": {
                "kind": "step_absolute_tick",
                "onset_tick": 6,
                "baseline": 1.0,
            },
        },
        "mutation.rate": {
            "value": 0.125,
            "schedule": {
                "kind": "step_absolute_tick",
                "onset_tick": 6,
                "baseline": 0.125,
            },
        },
    }
    compilation = compile_science_request_v1(
        validate_eiel_science_request_v1(selected_request)
    )
    admission = _engineering_bundle_admission(
        compilation.world_bundle,
        context,
        None,
        maximum_total_ticks=8,
    )
    assert admission.outcome is ScienceAdmissionOutcomeV1.ADMITTED

    founder_compilation = compile_science_request_v1(
        validate_eiel_science_request_v1(
            json.loads(
                (
                    REPO_ROOT
                    / "_spec/adaptive_science_expedition/v1/examples/"
                    "EIEL_ADAPTIVE_EXPEDITION_G4_W64_FOUNDER_DECLARATION_V1.json"
                ).read_bytes()
            )
        )
    )
    founder_admission = _engineering_bundle_admission(
        founder_compilation.world_bundle,
        context,
        None,
        maximum_total_ticks=8,
    )
    assert founder_admission.outcome is not ScienceAdmissionOutcomeV1.ADMITTED
    assert any(
        finding.field == "native_projection.selected_world_initialization"
        for finding in founder_admission.findings
    )

    invalid = deepcopy(plan)
    invalid["execution"]["acs_checkpoint_slots"] = 1
    with pytest.raises(WorkhorseSelectedHostContextError, match="B1/ACS2"):
        materialize_workhorse_selected_host_context_v1(
            invalid,
            profile=profile,
            logical_world_capacity=1,
            context_scope_id="test-workhorse-selected-host-context-invalid-v1",
        )


def test_selected_workhorse_controls_are_static_or_single_branch_tick_only() -> None:
    controls = {
        "control_bindings": [
            {
                "control_id": "environment.resource.multiplier",
                "resolved_value": 1.5,
                "schedule": {
                    "kind": "piecewise_constant_absolute_ticks",
                    "segments": [{"start_absolute_tick": 6, "value": 1.5}],
                },
            },
            {
                "control_id": "mutation.rate",
                "resolved_value": 0.125,
                "schedule": {"kind": "static"},
            },
        ]
    }
    assert _selected_workhorse_control_values_v1(
        controls, branch_tick=6
    ) == (1.5, 0.125)

    invalid_schedules = (
        {
            "kind": "piecewise_constant_absolute_ticks",
            "segments": [
                {"start_absolute_tick": 6, "value": 1.5},
                {"start_absolute_tick": 7, "value": 1.0},
            ],
        },
        {
            "kind": "piecewise_constant_absolute_ticks",
            "segments": [{"start_absolute_tick": 7, "value": 1.5}],
        },
        {"kind": "pulse_absolute_ticks"},
        {"kind": "switch_absolute_tick"},
    )
    for schedule in invalid_schedules:
        invalid = deepcopy(controls)
        invalid["control_bindings"][0]["schedule"] = schedule
        with pytest.raises(
            ScienceWorldBundleError,
            match="branch tick|static or branch-tick-only schedule",
        ):
            _selected_workhorse_control_values_v1(invalid, branch_tick=6)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _acs_workhorse_state(
    contract: FullEvolutionCudaWorkhorseContractV2, *, tick: int
) -> torch.Tensor:
    row_bytes = workhorse_layout_v2(contract.shape).state_bytes_per_world
    raw = bytearray(contract.B * row_bytes)
    for world in range(contract.B):
        header = [0] * 32
        header[0] = 10_000 + world
        header[1] = tick
        header[4] = 0
        header[13] = 20_000 + world
        header[14] = 4
        struct.pack_into("<32Q", raw, world * row_bytes, *header)
    return torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(
        contract.B, row_bytes
    )


def _acs_workhorse_metadata(**values: Any) -> dict[str, Any]:
    headers = values["state_headers"]
    tick = values["checkpoint_at_tick"]
    assert all(
        header[1] == tick and header[4] == 0 and header[14] == 4
        for header in headers
    )
    body = {
        "schema_version": "eiel_full_evolution_cuda_workhorse_checkpoint_bundle_v2",
        "time_authority": "absolute_tick_v1",
        "contract": values["contract"].to_dict(),
        "checkpoint_at_tick": tick,
        "checkpoint_identity": {"checkpoint_identity_sha256": "9" * 64},
        "canonical_state": {
            "file_name": "canonical_state.bin",
            "bytes": values["state_nbytes"],
            "sha256": values["state_sha256"],
        },
        "world_uids": [int(header[0]) for header in headers],
        "rng_keys": [int(header[13]) for header in headers],
    }
    return {
        **body,
        "bundle_body_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def _write_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o444)


def _operation_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend_id: str = "phase_graph_v2",
    cuda_runtime: str = "12.8",
    shape_id: str = "w64h64p1024c2048",
    acs_slots: int | None = None,
    B: int = 2,
) -> dict[str, Any]:
    temporal_k = 1
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape_id,
        backend_id=backend_id,
        B=B,
        temporal_k=temporal_k,
        total_ticks=6,
    )
    artifact_path = tmp_path / "workhorse.so"
    envelope_path = tmp_path / "envelope.json"
    micro_path = tmp_path / "micro.json"
    semantic_path = tmp_path / "semantic.json"
    for path, payload in (
        (artifact_path, b"workhorse-host-artifact"),
        (envelope_path, b"{}"),
        (micro_path, b"{}"),
        (semantic_path, b"{}"),
    ):
        _write_file(path, payload)
    artifact_digest = _file_sha256(artifact_path)
    sealed_artifact_digest = hashlib.sha256(
        b"sealed-workhorse-host-artifact"
    ).hexdigest()
    artifact_binding = {
        "path": str(artifact_path.resolve()),
        "sha256": artifact_digest,
        "manifest_path": str((tmp_path / "workhorse.manifest.json").resolve()),
        "manifest_file_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "sealed_artifact_identity_sha256": sealed_artifact_digest,
        "source_inventory_sha256": "3" * 64,
        "identity": workhorse_artifact_identity_v2(
            contract.shape, contract.backend_id
        ),
    }
    device = {
        "index": 0,
        "name": "host-fixture-device",
        "uuid": "GPU-WORKHORSE-HOST",
        "pci_domain_id": 0,
        "pci_bus_id": 1,
        "pci_device_id": 0,
        "total_memory_bytes": 24 * 1024**3,
        "multiprocessor_count": 40,
        "sm": "sm_75",
        "cuda_runtime": cuda_runtime,
    }
    envelope = {
        "receipt_file_sha256": _file_sha256(envelope_path),
        "receipt_content_sha256": "4" * 64,
        "artifact": artifact_binding,
        "shape_id": shape_id,
        "backend_id": backend_id,
        "device_identity": device,
        "expected_rows": [
            {
                "B": B,
                "temporal_k": temporal_k,
                "planned_admission": {
                    "admitted": True,
                    "required_bytes": 64 * 1024**2,
                },
            }
        ],
    }
    monkeypatch.setattr(
        provider_module,
        "workhorse_artifact_binding_v2",
        lambda _path: deepcopy(artifact_binding),
    )
    monkeypatch.setattr(
        provider_module,
        "validate_workhorse_device_envelope_calibration_receipt_v2",
        lambda _path: deepcopy(envelope),
    )
    operation = {
        "schema_version": OPERATION_SCHEMA_V2,
        "operation": "EXECUTION_ROW",
        "contract": contract.to_dict(),
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": artifact_digest,
            "sealed_artifact_identity_sha256": sealed_artifact_digest,
        },
        "envelope": {
            "path": str(envelope_path.resolve()),
            "file_sha256": _file_sha256(envelope_path),
            "content_sha256": envelope["receipt_content_sha256"],
        },
        "authoritative_micro_receipt": {
            "path": str(micro_path.resolve()),
            "file_sha256": _file_sha256(micro_path),
            "content_sha256": "5" * 64,
        },
        "semantic_gate_receipt": {
            "path": str(semantic_path.resolve()),
            "file_sha256": _file_sha256(semantic_path),
            "content_sha256": "6" * 64,
        },
        "science_continuation_contract_sha256": "7" * 64,
        "source_checkpoint_identity_sha256": None,
        "measurement_schedule": (
            canonical_rows.workhorse_fixed_tick_measurement_schedule_v3(
                temporal_k=temporal_k,
                warmup_ticks=1,
                measurement_window_ticks=1,
                measured_windows=5,
            )
        ),
    }
    spec = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    spec["campaign_id"] = "workhorse-v2-one-row-host"
    spec["world"] = {"base": WORLD_BASE, "overlays": []}
    spec["parameters"] = {
        "workhorse.operation": "EXECUTION_ROW",
        "workhorse.shape_id": shape_id,
        "workhorse.backend_id": backend_id,
        "workhorse.B": B,
        "workhorse.temporal_k": temporal_k,
    }
    spec["matrix"]["dimensions"] = [
        {"id": "ecology_seed", "values": [101]},
        {"id": "task_seed", "values": [100001]},
    ]
    spec["matrix"]["expected_cells"] = 1
    spec["matrix"]["max_cells"] = 1
    spec["execution"].update(
        {
            "total_ticks": 6,
            "checkpoint_ticks": [6],
            "resident_owner_span_ticks": 6,
            "native_dispatch_window_ticks": temporal_k,
            "source_window_ticks": temporal_k,
            "event_window_capacity": 2 * contract.shape.capacity,
            "batch": {"mode": "explicit", "cells": 1},
        }
    )
    if acs_slots is not None:
        spec["checkpoints"]["acs_slots"] = acs_slots
    spec["execution"]["technical_gate"].update(
        {"total_ticks": 6, "checkpoint_ticks": [6]}
    )
    spec["artifact_lifecycle"]["checkpoints"] = {
        "authority": "absolute_tick",
        "checkpoint_ticks": [6],
        "roles": [
            {
                "identity": "checkpoint-tick-000000000006",
                "checkpoint_tick": 6,
                "role": "terminal",
                "retention_class": "R2",
            }
        ],
    }
    spec["artifacts"] = {
        "workhorse_artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": artifact_digest,
            "kind": "cuda_extension",
            "abi": "eiel_full_evolution_cuda_workhorse_abi_v2",
        }
    }
    if acs_slots:
        native_extension = tmp_path / "eiel_native_runtime_ext.so"
        _write_file(native_extension, b"workhorse-native-acs-host-fixture")
        native_manifest = native_extension.with_suffix(
            native_extension.suffix + ".manifest.json"
        )
        _write_file(
            native_manifest,
            (
                json.dumps(
                    {
                        "schema_version": "eiel_native_runtime_build_manifest_v1",
                        "stable_module_name": ACS_NATIVE_RUNTIME_MODULE_NAME_V1,
                        "extension_abi_version": ACS_NATIVE_RUNTIME_ABI_V1,
                        "extension_version": ACS_NATIVE_RUNTIME_VERSION_V1,
                        "output_sha256": _file_sha256(native_extension),
                        "runtime_jit_in_measured_execution": False,
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
        spec["artifacts"].update(
            {
                ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1: {
                    "path": str(native_extension.resolve()),
                    "sha256": _file_sha256(native_extension),
                    "kind": ACS_NATIVE_RUNTIME_EXTENSION_KIND_V1,
                    "abi": ACS_NATIVE_RUNTIME_ABI_V1,
                },
                ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1: {
                    "path": str(native_manifest.resolve()),
                    "sha256": _file_sha256(native_manifest),
                    "kind": ACS_NATIVE_RUNTIME_MANIFEST_KIND_V1,
                    "abi": ACS_NATIVE_RUNTIME_ABI_V1,
                },
            }
        )
    spec["analysis"] = {
        "required_metrics": ["workhorse.p10_world_ticks_per_second"],
        "optional_metrics": [],
        "rules": [],
        "extension": {
            "id": OPERATION_EXTENSION_ID,
            "contract": operation,
            "contract_sha256": canonical_sha256(operation),
        },
    }
    run_root = (tmp_path / "run").resolve()
    spec["output"]["runtime_root"] = str(run_root)
    plan = compile_campaign(
        spec,
        registry=cuda_full_evolution_workhorse_v2_profile_registry_v1(
            shape_id=shape_id
        ),
    ).payload()
    batch_id = str(plan["physical_batches"][0]["batch_id"])
    return {
        "plan": plan,
        "batch_id": batch_id,
        "run_root": run_root,
        "contract": contract,
        "artifact_binding": artifact_binding,
        "device": device,
        "envelope": envelope,
        "spec": spec,
    }


def _indexed_continuation_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected_world_observation: bool = False,
) -> dict[str, Any]:
    fixture = _operation_fixture(
        tmp_path,
        monkeypatch,
        acs_slots=2,
        B=1 if selected_world_observation else 2,
    )
    spec = deepcopy(fixture["spec"])
    old_contract = fixture["contract"]
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=old_contract.shape_id,
        backend_id=old_contract.backend_id,
        B=old_contract.B,
        temporal_k=2,
        total_ticks=5,
        start_next_tick=25,
    )
    science_sha256 = "7" * 64
    source_state_set_sha256 = "8" * 64
    logical = {
        "schema_version": "eiel_workhorse_science_continuation_identity_v2",
        "time_authority": "absolute_tick_v1",
        "science_continuation_contract_sha256": science_sha256,
        "source_world_state_set_sha256": source_state_set_sha256,
        "shape_id": contract.shape_id,
        "semantic_transition_sha256": contract.semantic_transition_sha256,
        "start_next_tick": 25,
        "target_next_tick": 30,
        "checkpoint_ticks": [28, 30],
        "world_count": 2,
        "continuation_scope": "FULL_SET_CONTINUATION",
        "packing_excluded_fields": [],
        "science_continuation_identity_sha256": "9" * 64,
    }
    if selected_world_observation:
        logical.update(
            {
                "source_checkpoint_occurrence_sha256": "1" * 64,
                "source_parent_occurrence_sha256": "2" * 64,
                "source_semantic_world_sha256": "3" * 64,
                "source_control_history_sha256": "4" * 64,
                "source_rng_state_sha256": "5" * 64,
                "source_event_prefix_sha256": "6" * 64,
                "source_active_control_state_sha256": "7" * 64,
            }
        )
    physical = {
        "schema_version": "eiel_workhorse_physical_continuation_plan_v2",
        "target_contract": contract.to_dict(),
        "repack_binding_sha256": "a" * 64,
        "dispatch_plan_sha256": "b" * 64,
        "source_world_state_set_sha256": source_state_set_sha256,
        "identity_scope": "PHYSICAL_EXECUTION_ONLY",
        "physical_continuation_plan_sha256": "c" * 64,
    }
    dispatch = {
        "schema_version": "eiel_workhorse_absolute_dispatch_plan_v2",
        "time_authority": "absolute_tick_v1",
        "start_next_tick": 25,
        "target_next_tick": 30,
        "total_ticks": 5,
        "preferred_temporal_k": 2,
        "admitted_temporal_k": [1, 2],
        "boundary_next_ticks": [28, 30],
        "segments": [
            {
                "start_next_tick": 25,
                "end_next_tick": 27,
                "temporal_k": 2,
                "dispatch_count": 1,
            },
            {
                "start_next_tick": 27,
                "end_next_tick": 28,
                "temporal_k": 1,
                "dispatch_count": 1,
            },
            {
                "start_next_tick": 28,
                "end_next_tick": 30,
                "temporal_k": 2,
                "dispatch_count": 1,
            },
        ],
        "dispatch_count": 3,
        "identity_scope": "PHYSICAL_PACKING_LOCAL_EXCLUDED_FROM_SCIENCE_IDENTITY",
        "dispatch_plan_sha256": "b" * 64,
    }
    continuation = {
        "schema_version": "eiel_workhorse_indexed_continuation_contract_v2",
        "operation": INDEXED_ACS_CONTINUATION_OPERATION_V2,
        "time_authority": "absolute_tick_v1",
        "science_continuation_identity": logical,
        "physical_continuation_plan": physical,
        "repack_binding": {
            "repack_binding_sha256": "a" * 64,
        },
        "dispatch_plan": dispatch,
        "checkpoint_ticks": [28, 30],
        "continuation_contract_sha256": "d" * 64,
    }
    row_operation = spec["analysis"]["extension"]["contract"]
    operation = {
        "schema_version": OPERATION_SCHEMA_V3,
        "operation": INDEXED_ACS_CONTINUATION_OPERATION_V2,
        "contract": contract.to_dict(),
        "artifact": deepcopy(row_operation["artifact"]),
        "envelope": deepcopy(row_operation["envelope"]),
        "authoritative_micro_receipt": deepcopy(
            row_operation["authoritative_micro_receipt"]
        ),
        "semantic_gate_receipt": deepcopy(row_operation["semantic_gate_receipt"]),
        "science_continuation_contract_sha256": science_sha256,
        "source_checkpoint_identity_sha256": source_state_set_sha256,
        "continuation_contract": continuation,
    }
    monkeypatch.setattr(
        provider_module,
        "validate_workhorse_indexed_continuation_contract_v2",
        lambda value: deepcopy(dict(value)),
    )
    fixture["envelope"]["expected_rows"].append(
        {
            "B": contract.B,
            "temporal_k": 2,
            "planned_admission": {
                "admitted": True,
                "required_bytes": 64 * 1024**2,
            },
        }
    )
    spec["campaign_id"] = "workhorse-v2-indexed-continuation-host"
    spec["parameters"].update(
        {
            "workhorse.operation": INDEXED_ACS_CONTINUATION_OPERATION_V2,
            "workhorse.temporal_k": 2,
        }
    )
    spec["execution"].update(
        {
            "start_next_tick": 25,
            "total_ticks": 30,
            "checkpoint_ticks": [28, 30],
            "resident_owner_span_ticks": 5,
            "native_dispatch_window_ticks": 2,
            "source_window_ticks": 2,
        }
    )
    spec["execution"]["technical_gate"].update(
        {"total_ticks": 30, "checkpoint_ticks": [28, 30]}
    )
    spec["checkpoints"]["acs_slots"] = 2
    spec["artifact_lifecycle"]["checkpoints"] = {
        "authority": "absolute_tick",
        "checkpoint_ticks": [28, 30],
        "roles": [
            {
                "identity": "checkpoint-tick-000000000028",
                "checkpoint_tick": 28,
                "role": "intermediate",
                "retention_class": "R2",
            },
            {
                "identity": "checkpoint-tick-000000000030",
                "checkpoint_tick": 30,
                "role": "terminal",
                "retention_class": "R2",
            },
        ],
    }
    spec["analysis"]["extension"] = {
        "id": OPERATION_EXTENSION_ID,
        "contract": operation,
        "contract_sha256": canonical_sha256(operation),
    }
    spec["analysis"]["required_metrics"] = [
        INDEXED_ACS_CONTINUATION_AGGREGATE_WTPS_METRIC_V2
    ]
    plan = compile_campaign(
        spec,
        registry=cuda_full_evolution_workhorse_v2_profile_registry_v1(
            shape_id=contract.shape_id,
            operation=INDEXED_ACS_CONTINUATION_OPERATION_V2,
        ),
    ).payload()
    return {
        **fixture,
        "spec": spec,
        "plan": plan,
        "batch_id": str(plan["physical_batches"][0]["batch_id"]),
        "contract": contract,
        "operation": operation,
        "continuation": continuation,
    }


def test_indexed_continuation_compiles_as_one_acs2_attempt_with_split_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _indexed_continuation_fixture(tmp_path, monkeypatch)
    provider = provider_module.CudaFullEvolutionWorkhorseV2ExecutionProviderV1(
        shape=fixture["contract"].shape
    )
    request = provider.translate_batch(fixture["plan"], fixture["batch_id"])

    assert request.acs_checkpoint_slots == 2
    assert request.start_next_tick == 25
    assert request.total_ticks == 30
    assert request.checkpoint_ticks == (28, 30)
    assert request.tick_native_control["dispatch_plan"]["dispatch_count"] == 3
    route = provider.execution_route_contract(fixture["plan"]).payload()
    operation_component = next(
        row
        for row in route["components"]
        if row["component_id"] == "cuda_full_evolution_workhorse_v2_operation"
    )
    assert operation_component["one_operation_per_attempt"] is True
    assert operation_component["science_continuation_identity"] == fixture[
        "continuation"
    ]["science_continuation_identity"]
    assert operation_component["physical_continuation_plan"] == fixture[
        "continuation"
    ]["physical_continuation_plan"]
    assert operation_component["compiled_temporal_k"] == [1, 2]
    bound_request = build_execution_request_v1(
        provider, fixture["plan"], fixture["batch_id"]
    ).payload()
    route_binding = adapter_module._route_bindings(bound_request)
    assert route_binding["event_transport"] == (
        "workhorse_external_exact_ack_indexed_acs2_v2"
    )
    assert route_binding["ticks_per_dispatch"] == 2
    resources = provider.estimate_resources(fixture["plan"], [request])
    assert resources["resource_estimator_id"] == (
        "workhorse_v2_indexed_acs_continuation_v2"
    )
    assert resources["event_dispatch_count_upper_bound"] == 3


@pytest.mark.parametrize(
    "shape_id",
    ("w64h64p1024c2048", "w128h80p2560c5120"),
)
def test_workhorse_acs_resource_bounds_scale_by_slots_without_extra_vram(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape_id: str,
) -> None:
    estimates: dict[int, Mapping[str, Any]] = {}
    for slots in (0, 1, 2):
        fixture = _operation_fixture(
            tmp_path / f"slots-{slots}",
            monkeypatch,
            shape_id=shape_id,
            acs_slots=slots,
        )
        registry = cuda_full_evolution_workhorse_v2_provider_registry_v1()
        provider = registry.get(fixture["plan"]["resolved_profile"])
        assert provider is not None
        request = provider.translate_batch(fixture["plan"], fixture["batch_id"])
        estimates[slots] = provider.estimate_resources(
            fixture["plan"], (request,)
        )
        signed = compile_campaign(
            fixture["spec"],
            registry=cuda_full_evolution_workhorse_v2_profile_registry_v1(
                shape_id=shape_id
            ),
            planning_resource_accounting=(
                provider_module.cuda_full_evolution_workhorse_v2_signed_planning_resource_accounting_v2
            ),
        ).payload()
        assert signed["resource_estimate"]["disk_bytes_upper_bound"] == estimates[
            slots
        ]["disk_bytes_upper_bound"]
        assert signed["resource_estimate"]["authority"] == (
            "planning_upper_bound_with_signed_workhorse_storage_v2"
        )

    payload_bytes = 2 * workhorse_layout_v2(
        FullEvolutionCudaWorkhorseContractV2(
            shape_id=shape_id,
            backend_id="phase_graph_v2",
            B=2,
            temporal_k=1,
            total_ticks=6,
        ).shape
    ).state_bytes_per_world
    assert {row["vram_bytes_upper_bound"] for row in estimates.values()} == {
        64 * 1024**2
    }
    for slots, row in estimates.items():
        assert row["acs_checkpoint_slots"] == slots
        assert row["acs_capture_pool_count"] == (1 if slots else 0)
        assert row["acs_pinned_slab_count"] == (2 if slots else 0)
        assert row["acs_pinned_host_bytes_upper_bound"] == (
            128 * 1024**2 if slots else 0
        )
        assert row["acs_pending_capture_files_upper_bound"] == min(slots, 1)
        assert row["acs_pending_capture_file_bytes_upper_bound"] == (
            min(slots, 1) * payload_bytes
        )
        assert row["indexed_world_payload_copies"] == 0
        assert row["physical_checkpoint_payload_copies"] == (1 if slots else 0)
        assert row["absolute_tick_projection_payload_copies"] == (
            1 if slots else 0
        )
        assert row["acs_producer_full_state_host_bytes"] == 0
        assert row["acs_producer_compacted_cuda_bytes"] == 0
        assert row["acs_worker_full_state_bytes"] == 0
        assert row["acs_worker_cuda_references"] == 0


@pytest.mark.parametrize(
    ("storage_field", "estimate_field", "message"),
    (
        (
            "scratch_max_bytes",
            "acs_pending_capture_file_bytes_upper_bound",
            "ACS scratch upper bound",
        ),
        (
            "checkpoint_max_bytes",
            "checkpoint_bytes_upper_bound",
            "checkpoint upper bound",
        ),
        (
            "durable_max_bytes",
            "durable_noncheckpoint_bytes_upper_bound",
            "durable output upper bound",
        ),
    ),
)
def test_workhorse_signed_storage_rejects_each_lifecycle_partition_one_byte_low(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage_field: str,
    estimate_field: str,
    message: str,
) -> None:
    fixture = _operation_fixture(
        tmp_path,
        monkeypatch,
        shape_id="w128h80p2560c5120",
        acs_slots=2,
    )
    provider = cuda_full_evolution_workhorse_v2_provider_registry_v1().get(
        fixture["plan"]["resolved_profile"]
    )
    assert provider is not None
    request = provider.translate_batch(fixture["plan"], fixture["batch_id"])
    estimate = provider.estimate_resources(fixture["plan"], (request,))
    hostile = deepcopy(fixture["spec"])
    hostile["artifact_lifecycle"]["storage"][storage_field] = (
        estimate[estimate_field] - 1
    )
    with pytest.raises(CampaignError, match=message):
        compile_campaign(
            hostile,
            registry=cuda_full_evolution_workhorse_v2_profile_registry_v1(
                shape_id="w128h80p2560c5120"
            ),
            planning_resource_accounting=(
                provider_module.cuda_full_evolution_workhorse_v2_signed_planning_resource_accounting_v2
            ),
        )


@pytest.mark.parametrize(
    "shape_id",
    ("w64h64p1024c2048", "w128h80p2560c5120"),
)
@pytest.mark.parametrize("fail_lifecycle_commit", (False, True))
def test_workhorse_row_acs_commits_occurrence_before_ack_and_quarantines_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape_id: str,
    fail_lifecycle_commit: bool,
) -> None:
    fixture = _bound_attempt(
        tmp_path,
        monkeypatch,
        shape_id=shape_id,
        acs_slots=2,
    )
    events: list[str] = []

    class FakeToken:
        sequence = 1

    class FakeCoordinator:
        def __init__(self, **values: Any) -> None:
            self.slots = int(values["slots"])
            self.checkpoint_root = Path(values["checkpoint_root"])
            self.pending: list[FakeToken] = []
            self.receipts: list[dict[str, Any]] = []
            self.closed = False

        def capture_workhorse_checkpoint(
            self,
            _state: object,
            *,
            contract: FullEvolutionCudaWorkhorseContractV2,
            checkpoint_at_tick: int,
            event_sink_receipt_path: str | Path,
        ) -> FakeToken:
            assert checkpoint_at_tick == contract.start_next_tick + contract.total_ticks
            assert Path(event_sink_receipt_path).is_file()
            bundle = self.checkpoint_root / (
                f"checkpoint-next-tick-{checkpoint_at_tick:012d}"
            )
            bundle.mkdir(parents=True)
            state = b"workhorse-row-acs-state"
            (bundle / "canonical_state.bin").write_bytes(state)
            (bundle / "checkpoint.json").write_text("{}\n", encoding="ascii")
            token = FakeToken()
            self.pending.append(token)
            return token

        def pending_tokens(self) -> tuple[FakeToken, ...]:
            return tuple(self.pending)

        def settle_checkpoint(self, token: FakeToken) -> dict[str, Any]:
            assert self.pending == [token]
            bundle = next(self.checkpoint_root.glob("checkpoint-next-tick-*"))
            state = (bundle / "canonical_state.bin").read_bytes()
            return {
                "path": str(bundle),
                "manifest_path": str(bundle / "checkpoint.json"),
                "checkpoint_at_tick": fixture["contract"].total_ticks,
                "canonical_state_sha256": hashlib.sha256(state).hexdigest(),
                "canonical_state_bytes": len(state),
            }

        def acknowledge_checkpoint(self, token: FakeToken) -> None:
            assert self.pending == [token]
            events.append("ack")
            self.receipts.append(self.settle_checkpoint(token))
            self.pending.clear()

        def close(self) -> None:
            assert not self.pending
            self.closed = True

        def abort(self) -> None:
            events.append("abort")
            for bundle in self.checkpoint_root.glob("checkpoint-next-tick-*"):
                bundle.rename(bundle.with_name(bundle.name + ".publication.failed.fake"))
            self.pending.clear()
            self.closed = True

        def telemetry(self) -> dict[str, Any]:
            return {
                "slots": self.slots,
                "closed": self.closed,
                "writer_alive": False,
                "failure_latched": False,
                "checkpoint_receipts": [dict(row) for row in self.receipts],
            }

    monkeypatch.setattr(
        acs_adapter_module,
        "CudaFullEvolutionV2WorkhorseACSCoordinatorV1",
        FakeCoordinator,
    )

    state_identity: dict[str, Any] = {}

    def inspect_checkpoint(path: str | Path) -> tuple[dict[str, Any], Path]:
        manifest_path = Path(path)
        state_path = manifest_path.parent / "canonical_state.bin"
        state = state_path.read_bytes()
        state_identity.update(
            {
                "bytes": len(state),
                "sha256": hashlib.sha256(state).hexdigest(),
            }
        )
        return {
            "checkpoint_at_tick": fixture["contract"].total_ticks,
            "canonical_state": dict(state_identity),
        }, state_path

    monkeypatch.setattr(
        provider_module,
        "inspect_workhorse_checkpoint_bundle_v2",
        inspect_checkpoint,
    )

    def write_index(
        _manifest_path: str | Path, destination: str | Path
    ) -> dict[str, Any]:
        events.append("index")
        root = Path(destination)
        tick = int(root.name.rsplit("-", 1)[-1])
        root.mkdir(parents=True)
        manifest = root / "checkpoint-index.json"
        _write_file(manifest, b"{}\n")
        return {
            "manifest_path": str(manifest.resolve()),
            "manifest_file_sha256": _file_sha256(manifest),
            "manifest_body_sha256": "8" * 64,
            "world_state_set_identity": {
                "checkpoint_at_tick": fixture["contract"].total_ticks,
                "shape_id": fixture["contract"].shape_id,
                "world_count": fixture["contract"].B,
                "semantic_transition_sha256": fixture[
                    "contract"
                ].semantic_transition_sha256,
            },
            "storage_model": {
                "canonical_state_payload_copies": 1,
                "indexed_world_payload_copies": 0,
                "world_rows_are_range_references": True,
                "source_payload_must_remain_live": True,
            },
        }

    monkeypatch.setattr(
        provider_module,
        "write_workhorse_indexed_checkpoint_set_v1",
        write_index,
    )

    def materialize_occurrence(**values: Any) -> dict[str, Any]:
        events.append("occurrence")
        absolute_root = Path(values["absolute_bundle_root"])
        occurrence_root = Path(values["occurrence_root"])
        absolute_root.mkdir(parents=True)
        occurrence_root.mkdir(parents=True)
        for name in ("canonical_state.bin", "metadata.json", "bundle_complete.json"):
            (absolute_root / name).write_bytes(name.encode("ascii"))
        for name in (
            "absolute_tick_checkpoint_occurrence_v1.json",
            "absolute_tick_checkpoint_occurrence_commit_v1.json",
        ):
            (occurrence_root / name).write_bytes(name.encode("ascii"))
        projection = {
            "schema_version": (
                "eiel_workhorse_row_checkpoint_occurrence_projection_v2"
            ),
            "checkpoint_at_tick": fixture["contract"].total_ticks,
            "canonical_state_sha256": state_identity["sha256"],
            "canonical_visible": True,
            "restore_admitted_exact": False,
            "absolute_tick_bundle_path": str(absolute_root.resolve()),
            "occurrence_root": str(occurrence_root.resolve()),
        }
        return {
            "canonical_state": None,
            "canonical_state_nbytes": state_identity["bytes"],
            "canonical_state_sha256": state_identity["sha256"],
            "projection": projection,
        }

    monkeypatch.setattr(
        fixture["adapter"],
        "materialize_workhorse_row_checkpoint_occurrence_v2",
        materialize_occurrence,
    )

    def run_row(_artifact_path: str | Path, **values: Any) -> dict[str, Any]:
        event_path = Path(values["event_sink_root"]) / "event.json"
        _write_file(event_path, b"{}\n")
        transport = values["checkpoint_transport"]
        assert transport is not None
        transport.capture_workhorse_checkpoint(
            object(),
            contract=fixture["contract"],
            checkpoint_at_tick=fixture["contract"].total_ticks,
            event_sink_receipt_path=event_path,
        )
        result_path = Path(values["execution_result_path"])
        _write_file(result_path, b'{"outcome":"COMPLETED"}\n')
        return {
            "outcome": "COMPLETED",
            "receipt_file_sha256": _file_sha256(result_path),
            "receipt_content_sha256": "a" * 64,
            "absolute_frontiers": {
                "start_next_tick": 0,
                "target_next_tick": 6,
                "committed_next_tick": 6,
            },
            "checkpoint_binding": canonical_rows.workhorse_row_checkpoint_binding_v2(
                fixture["contract"],
                science_continuation_contract_sha256="7" * 64,
            ),
            "p10_world_ticks_per_second": 1.0,
        }

    monkeypatch.setattr(
        provider_module,
        "run_workhorse_canonical_admitted_row_v2",
        run_row,
    )

    recorded: list[Path] = []

    def record_artifact(path: str | Path, **_metadata: Any) -> None:
        artifact = Path(path)
        if "workhorse-checkpoint-occurrences" in artifact.parts:
            events.append("lifecycle-occurrence")
            if fail_lifecycle_commit:
                raise RuntimeError("injected lifecycle occurrence failure")
        recorded.append(artifact)

    context = SimpleNamespace(
        run_root=fixture["run_root"],
        batch_root=fixture["run_root"] / "batches" / fixture["batch_id"],
        record_artifact=record_artifact,
    )
    request = fixture["provider"].translate_batch(
        fixture["plan"], fixture["batch_id"]
    )
    if fail_lifecycle_commit:
        with pytest.raises(RuntimeError, match="injected lifecycle"):
            fixture["provider"].execute_batch(
                fixture["plan"], request, context
            )
        assert "ack" not in events
        assert "abort" in events
        assert not (
            context.batch_root
            / "workhorse-absolute-checkpoints"
            / "checkpoint-tick-000000000006"
        ).exists()
        assert not (
            context.batch_root
            / "workhorse-checkpoint-occurrences"
            / "checkpoint-tick-000000000006"
        ).exists()
        assert list(
            (context.batch_root / "workhorse-absolute-checkpoints").glob(
                "checkpoint-tick-000000000006.publication.failed.*"
            )
        )
        assert list(
            (context.batch_root / "workhorse-checkpoint-occurrences").glob(
                "checkpoint-tick-000000000006.publication.failed.*"
            )
        )
        return

    result = fixture["provider"].execute_batch(fixture["plan"], request, context)
    assert events.index("index") < events.index("occurrence")
    assert events.index("occurrence") < events.index("lifecycle-occurrence")
    assert events.index("lifecycle-occurrence") < events.index("ack")
    projection = result["checkpoint"]["acs_checkpoint_receipt"][
        "checkpoint_occurrence_projection"
    ]
    assert projection["canonical_visible"] is True
    assert projection["restore_admitted_exact"] is False
    assert any("workhorse-checkpoint-occurrences" in path.parts for path in recorded)


@pytest.mark.parametrize(
    "shape_id",
    ("w64h64p1024c2048", "w128h80p2560c5120"),
)
def test_workhorse_row_acs_two_slot_backpressure_waits_for_occurrence_lifecycle_ack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape_id: str,
) -> None:
    fixture = _operation_fixture(
        tmp_path,
        monkeypatch,
        shape_id=shape_id,
        acs_slots=2,
    )
    contract = fixture["contract"]
    monkeypatch.setattr(
        acs_adapter_module,
        "workhorse_checkpoint_metadata_from_state_identity_v2",
        _acs_workhorse_metadata,
    )
    monkeypatch.setattr(
        workhorse_authority,
        "workhorse_checkpoint_metadata_from_state_identity_v2",
        _acs_workhorse_metadata,
    )

    def inspect_checkpoint(path: str | Path) -> tuple[dict[str, Any], Path]:
        manifest_path = Path(path)
        metadata = json.loads(manifest_path.read_text(encoding="ascii"))
        return metadata, manifest_path.parent / "canonical_state.bin"

    monkeypatch.setattr(
        provider_module,
        "inspect_workhorse_checkpoint_bundle_v2",
        inspect_checkpoint,
    )

    def write_index(
        _manifest_path: str | Path, destination: str | Path
    ) -> dict[str, Any]:
        events.append("index")
        root = Path(destination)
        destination_tick = int(root.name.rsplit("-", 1)[-1])
        root.mkdir(parents=True)
        manifest = root / "checkpoint-index.json"
        _write_file(manifest, b"{}\n")
        return {
            "manifest_path": str(manifest.resolve()),
            "manifest_file_sha256": _file_sha256(manifest),
            "manifest_body_sha256": "8" * 64,
            "world_state_set_identity": {
                "checkpoint_at_tick": destination_tick,
                "shape_id": contract.shape_id,
                "world_count": contract.B,
                "semantic_transition_sha256": contract.semantic_transition_sha256,
            },
            "storage_model": {
                "canonical_state_payload_copies": 1,
                "indexed_world_payload_copies": 0,
                "world_rows_are_range_references": True,
                "source_payload_must_remain_live": True,
            },
        }

    monkeypatch.setattr(
        provider_module,
        "write_workhorse_indexed_checkpoint_set_v1",
        write_index,
    )
    artifact_path = Path(fixture["spec"]["artifacts"]["workhorse_artifact"]["path"])
    checkpoint_root = tmp_path / "integrated-two-slot-checkpoints"
    coordinator = (
        acs_adapter_module.CudaFullEvolutionV2WorkhorseACSCoordinatorV1(
            slots=2,
            checkpoint_root=checkpoint_root,
            artifact_path=artifact_path,
            attempt_authority_binding={"attempt": "integrated-two-slot"},
            chunk_bytes=64 * 1024,
            require_native_pool=False,
        )
    )
    event_paths: dict[int, Path] = {}
    for tick in (2, 4, 6):
        event_paths[tick] = tmp_path / f"event-{tick}.json"
        event_paths[tick].write_text("{}\n", encoding="ascii")
    states = {
        tick: _acs_workhorse_state(contract, tick=tick) for tick in (2, 4, 6)
    }
    first = coordinator.capture_workhorse_checkpoint(
        states[2],
        contract=contract,
        checkpoint_at_tick=2,
        event_sink_receipt_path=event_paths[2],
    )
    second = coordinator.capture_workhorse_checkpoint(
        states[4],
        contract=contract,
        checkpoint_at_tick=4,
        event_sink_receipt_path=event_paths[4],
    )
    with pytest.raises(
        acs_adapter_module.CudaFullEvolutionV2ACSError,
        match="capacity requires the FIFO head",
    ):
        coordinator.capture_workhorse_checkpoint(
            states[6],
            contract=contract,
            checkpoint_at_tick=6,
            event_sink_receipt_path=event_paths[6],
        )

    events: list[str] = []

    class RecordingCoordinator:
        def settle_checkpoint(self, token: Any) -> Mapping[str, Any]:
            return coordinator.settle_checkpoint(token)

        def acknowledge_checkpoint(self, token: Any) -> None:
            events.append("ack")
            coordinator.acknowledge_checkpoint(token)

    class OccurrenceAdapter:
        @staticmethod
        def materialize_workhorse_row_checkpoint_occurrence_v2(
            **values: Any,
        ) -> dict[str, Any]:
            events.append("occurrence")
            manifest_path = Path(values["checkpoint_manifest_path"])
            state_path = manifest_path.parent / "canonical_state.bin"
            state_nbytes = state_path.stat().st_size
            state_sha256 = _file_sha256(state_path)
            metadata = json.loads(manifest_path.read_text(encoding="ascii"))
            tick = int(metadata["checkpoint_at_tick"])
            absolute_root = Path(values["absolute_bundle_root"])
            occurrence_root = Path(values["occurrence_root"])
            absolute_root.mkdir(parents=True)
            occurrence_root.mkdir(parents=True)
            _write_file(absolute_root / "canonical_state.bin", b"projection")
            _write_file(absolute_root / "bundle_complete.json", b"{}\n")
            _write_file(
                occurrence_root
                / "absolute_tick_checkpoint_occurrence_commit_v1.json",
                b"{}\n",
            )
            return {
                "canonical_state": None,
                "canonical_state_nbytes": state_nbytes,
                "canonical_state_sha256": state_sha256,
                "projection": {
                    "schema_version": (
                        "eiel_workhorse_row_checkpoint_occurrence_projection_v2"
                    ),
                    "checkpoint_at_tick": tick,
                    "canonical_state_sha256": state_sha256,
                    "canonical_visible": True,
                    "restore_admitted_exact": False,
                    "absolute_tick_bundle_path": str(absolute_root.resolve()),
                    "occurrence_root": str(occurrence_root.resolve()),
                },
            }

    third_token: list[Any] = []
    checked_before_ack = False
    recorded_lifecycle: list[tuple[Path, dict[str, Any]]] = []

    def record_artifact(path: str | Path, **_metadata: Any) -> None:
        nonlocal checked_before_ack
        artifact = Path(path)
        recorded_lifecycle.append((artifact, dict(_metadata)))
        if (
            not checked_before_ack
            and "workhorse-checkpoint-occurrences" in artifact.parts
        ):
            checked_before_ack = True
            events.append("lifecycle")
            assert "ack" not in events
            with pytest.raises(
                acs_adapter_module.CudaFullEvolutionV2ACSError,
                match="capacity requires the FIFO head",
            ):
                coordinator.capture_workhorse_checkpoint(
                    states[6],
                    contract=contract,
                    checkpoint_at_tick=6,
                    event_sink_receipt_path=event_paths[6],
                )

    context = SimpleNamespace(
        batch_root=tmp_path / "integrated-two-slot-batch",
        record_artifact=record_artifact,
        lifecycle_contract={
            "checkpoints": {
                "roles": [
                    {
                        "identity": f"checkpoint-tick-{tick:012d}",
                        "checkpoint_tick": tick,
                        "role": "terminal" if tick == 6 else "intermediate",
                        "retention_class": "R4" if tick == 2 else "R2",
                    }
                    for tick in (2, 4, 6)
                ]
            }
        },
    )
    recording = RecordingCoordinator()
    first_receipt = provider_module._commit_workhorse_row_acs_checkpoint_v2(
        context=context,
        coordinator=recording,
        token=first,
        operation_adapter=OccurrenceAdapter(),
        canonical_parent_authority_binding={"attempt": "integrated-two-slot"},
        contract=contract,
        expected_checkpoint_tick=2,
    )
    assert events.index("index") < events.index("occurrence")
    assert events.index("occurrence") < events.index("lifecycle") < events.index("ack")
    assert first_receipt["checkpoint_at_tick"] == 2
    first_tick_rows = [
        (path, metadata)
        for path, metadata in recorded_lifecycle
        if metadata.get("checkpoint_at_tick") == 2
    ]
    assert first_tick_rows
    assert all(
        metadata["retention_class"] == (
            "R3"
            if "workhorse-checkpoint-occurrences" in path.parts
            else "R4"
        )
        for path, metadata in first_tick_rows
    )
    third_token.append(
        coordinator.capture_workhorse_checkpoint(
            states[6],
            contract=contract,
            checkpoint_at_tick=6,
            event_sink_receipt_path=event_paths[6],
        )
    )
    provider_module._commit_workhorse_row_acs_checkpoint_v2(
        context=context,
        coordinator=recording,
        token=second,
        operation_adapter=OccurrenceAdapter(),
        canonical_parent_authority_binding={"attempt": "integrated-two-slot"},
        contract=contract,
        expected_checkpoint_tick=4,
    )
    provider_module._commit_workhorse_row_acs_checkpoint_v2(
        context=context,
        coordinator=recording,
        token=third_token[0],
        operation_adapter=OccurrenceAdapter(),
        canonical_parent_authority_binding={"attempt": "integrated-two-slot"},
        contract=contract,
        expected_checkpoint_tick=6,
    )
    coordinator.close()
    telemetry = coordinator.telemetry()
    assert telemetry["queue_occupancy_high_water"] == 2
    assert telemetry["capture_pool"]["route_capture_count"] == 3
    assert telemetry["queued_cuda_references"] == 0
    assert telemetry["writer_alive"] is False


@pytest.mark.parametrize("fail_lifecycle_commit", (False, True))
def test_workhorse_science_acs_external_commit_is_atomic_before_ack(
    tmp_path: Path,
    fail_lifecycle_commit: bool,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    event_root = tmp_path / "events"
    boundary_root = tmp_path / "boundaries"
    checkpoint_bundle = checkpoint_root / "checkpoint-next-tick-000000000016"
    absolute_root = checkpoint_root / "crca-bundles/checkpoint-tick-000000000016"
    occurrence_root = (
        checkpoint_root / "crca-occurrences/checkpoint-tick-000000000016"
    )
    for root, name in (
        (checkpoint_bundle, "checkpoint.json"),
        (absolute_root, "bundle_complete.json"),
        (occurrence_root, "absolute_tick_checkpoint_occurrence_commit_v1.json"),
        (event_root, "event.json"),
        (boundary_root, "boundary.json"),
    ):
        _write_file(root / name, b"{}\n")

    events: list[str] = []

    class Coordinator:
        @staticmethod
        def acknowledge_checkpoint(_token: object) -> None:
            events.append("ack")

    def record_artifact(path: str | Path, **_metadata: Any) -> None:
        artifact = Path(path)
        if occurrence_root in artifact.parents:
            events.append("lifecycle-occurrence")
            if fail_lifecycle_commit:
                raise RuntimeError("injected science lifecycle failure")
        events.append("lifecycle")

    values = {
        "context": SimpleNamespace(record_artifact=record_artifact),
        "coordinator": Coordinator(),
        "token": object(),
        "checkpoint_manifest_path": checkpoint_bundle / "checkpoint.json",
        "occurrence_projection": {
            "occurrence_root": str(occurrence_root.resolve()),
            "absolute_tick_bundle_path": str(absolute_root.resolve()),
        },
        "event_sink_root": event_root,
        "checkpoint_root": checkpoint_root,
        "boundary_root": boundary_root,
    }
    if fail_lifecycle_commit:
        with pytest.raises(RuntimeError, match="injected science lifecycle"):
            provider_module._commit_workhorse_science_acs_checkpoint_v1(**values)
        assert "ack" not in events
        assert not absolute_root.exists()
        assert not occurrence_root.exists()
        assert list(
            absolute_root.parent.glob(
                "checkpoint-tick-000000000016.publication.failed.*"
            )
        )
        assert list(
            occurrence_root.parent.glob(
                "checkpoint-tick-000000000016.publication.failed.*"
            )
        )
        return

    provider_module._commit_workhorse_science_acs_checkpoint_v1(**values)
    assert events.index("lifecycle-occurrence") < events.index("ack")
    assert absolute_root.is_dir()
    assert occurrence_root.is_dir()


def _calibration_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend_id: str = "phase_graph_v2",
    cuda_runtime: str = "12.8",
) -> dict[str, Any]:
    fixture = _operation_fixture(
        tmp_path,
        monkeypatch,
        backend_id=backend_id,
        cuda_runtime=cuda_runtime,
    )
    spec = deepcopy(fixture["spec"])
    shape = fixture["contract"].shape
    compiled_k = list(shape.temporal_k_candidates)
    total_ticks = len(compiled_k)
    row_operation = spec["analysis"]["extension"]["contract"]
    calibration = {
        "schema_version": OPERATION_SCHEMA_V2,
        "operation": DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2,
        "shape_id": shape.shape_id,
        "backend_id": fixture["contract"].backend_id,
        "artifact": deepcopy(row_operation["artifact"]),
        "compiled_temporal_k": compiled_k,
        "device_identity": deepcopy(fixture["device"]),
        "reserve_policy": {
            "mode": "STABLE_TOTAL_MINUS_EFFECTIVE_RESERVE_V2",
            "reserve_bytes_requested": 64 * 1024**2,
            "minimum_fraction_numerator": 1,
            "minimum_fraction_denominator": 4,
            "transient_free_bytes_is_plan_authority": False,
        },
        "absolute_tick_span": {
            "time_authority": "absolute_tick_v1",
            "start_next_tick": 0,
            "target_next_tick": total_ticks,
            "total_ticks": total_ticks,
        },
        "observation_mode": "OFF",
        "science_continuation_claimed": False,
        "science_authority_granted": False,
        "qualification_authority_granted": False,
    }
    spec["campaign_id"] = "workhorse-v2-one-calibration-host"
    spec["parameters"] = {
        "workhorse.operation": DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2,
        "workhorse.shape_id": shape.shape_id,
        "workhorse.backend_id": fixture["contract"].backend_id,
        "workhorse.compiled_temporal_k_sha256": canonical_sha256(compiled_k),
        "workhorse.reserve_bytes": calibration["reserve_policy"][
            "reserve_bytes_requested"
        ],
    }
    spec["execution"].update(
        {
            "total_ticks": total_ticks,
            "checkpoint_ticks": [total_ticks],
            "resident_owner_span_ticks": total_ticks,
            "native_dispatch_window_ticks": total_ticks,
            "source_window_ticks": total_ticks,
            "event_window_capacity": 1,
        }
    )
    spec["execution"]["technical_gate"].update(
        {"total_ticks": total_ticks, "checkpoint_ticks": [total_ticks]}
    )
    spec["artifact_lifecycle"]["checkpoints"] = {
        "authority": "absolute_tick",
        "checkpoint_ticks": [total_ticks],
        "roles": [
            {
                "identity": f"checkpoint-tick-{total_ticks:012d}",
                "checkpoint_tick": total_ticks,
                "role": "terminal",
                "retention_class": "R2",
            }
        ],
    }
    spec["analysis"] = {
        "required_metrics": [
            "workhorse.device_envelope_calibration_completed"
        ],
        "optional_metrics": [],
        "rules": [],
        "extension": {
            "id": OPERATION_EXTENSION_ID,
            "contract": calibration,
            "contract_sha256": canonical_sha256(calibration),
        },
    }
    plan = compile_campaign(
        spec,
        registry=cuda_full_evolution_workhorse_v2_profile_registry_v1(
            shape_id=shape.shape_id,
            operation=DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2,
        ),
    ).payload()
    return {
        **fixture,
        "spec": spec,
        "plan": plan,
        "batch_id": str(plan["physical_batches"][0]["batch_id"]),
        "operation": calibration,
    }


def _semantic_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    case_id: str,
    semantic_temporal_k: int = 2,
) -> dict[str, Any]:
    law = workhorse_semantic_case_law_v2(
        case_id, semantic_temporal_k=semantic_temporal_k
    )
    backend_id = (
        "reference_phased_v2"
        if law.artifact_role == "REFERENCE"
        else "phase_graph_v2"
    )
    fixture = _operation_fixture(
        tmp_path, monkeypatch, backend_id=backend_id
    )
    shape = fixture["contract"].shape
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=backend_id,
        B=law.B,
        temporal_k=law.dispatch_temporal_k,
        total_ticks=law.total_ticks,
        start_next_tick=law.start_next_absolute_tick,
    )
    artifact_binding = {
        **fixture["artifact_binding"],
        "identity": workhorse_artifact_identity_v2(shape, backend_id),
    }
    envelope_path = Path(
        fixture["spec"]["analysis"]["extension"]["contract"]["envelope"][
            "path"
        ]
    )
    envelope = {
        "receipt_file_sha256": _file_sha256(envelope_path),
        "receipt_content_sha256": "4" * 64,
        "artifact": artifact_binding,
        "shape_id": shape.shape_id,
        "backend_id": backend_id,
        "device_identity": fixture["device"],
        "expected_rows": [
            {
                "B": law.B,
                "temporal_k": law.dispatch_temporal_k,
                "planned_admission": {
                    "admitted": True,
                    "required_bytes": 64 * 1024**2,
                },
            }
        ],
    }
    monkeypatch.setattr(
        provider_module,
        "workhorse_artifact_binding_v2",
        lambda _path: deepcopy(artifact_binding),
    )
    monkeypatch.setattr(
        provider_module,
        "validate_workhorse_device_envelope_calibration_receipt_v2",
        lambda _path: deepcopy(envelope),
    )
    science = workhorse_science_checkpoint_compatibility_binding_v2(REPO_ROOT)
    source_identity = "e" * 64 if law.requires_source_checkpoint else None
    external = None
    if law.requires_source_checkpoint:
        occurrence_path = tmp_path / "source-occurrence.json"
        manifest_path = tmp_path / "source-base-manifest.json"
        _write_file(occurrence_path, b'{"occurrence":"exact"}')
        _write_file(manifest_path, b'{"base":"exact"}')
        external_body = {
            "schema_version": WORKHORSE_EXTERNAL_CHECKPOINT_SOURCE_SCHEMA_V2,
            "authorization_source": (
                WORKHORSE_EXTERNAL_CHECKPOINT_AUTHORIZATION_SOURCE_V2
            ),
            "checkpoint_occurrence_record_path": str(occurrence_path.resolve()),
            "checkpoint_occurrence_record_file_sha256": _file_sha256(
                occurrence_path
            ),
            "checkpoint_occurrence_identity_sha256": "d" * 64,
            "base_checkpoint_manifest_path": str(manifest_path.resolve()),
            "base_checkpoint_manifest_file_sha256": _file_sha256(manifest_path),
            "base_checkpoint_identity_sha256": source_identity,
        }
        external = {
            **external_body,
            "binding_sha256": canonical_sha256(external_body),
        }
    semantic_case = workhorse_semantic_case_operation_contract_v2(
        case_id=case_id,
        shape_id=shape.shape_id,
        semantic_temporal_k=semantic_temporal_k,
    )
    base_operation = fixture["spec"]["analysis"]["extension"]["contract"]
    operation = {
        "schema_version": OPERATION_SCHEMA_V2,
        "operation": SEMANTIC_EVIDENCE_CASE_OPERATION_V2,
        "semantic_case_contract": semantic_case,
        "contract": contract.to_dict(),
        "artifact": deepcopy(base_operation["artifact"]),
        "envelope": deepcopy(base_operation["envelope"]),
        "science_checkpoint_compatibility_binding": science,
        "science_continuation_contract_sha256": science[
            "science_consumer_extension_contract_sha256"
        ],
        "source_checkpoint_identity_sha256": source_identity,
        "external_source_checkpoint_binding": external,
    }
    spec = deepcopy(fixture["spec"])
    target = law.start_next_absolute_tick + law.total_ticks
    spec["campaign_id"] = f"workhorse-semantic-{law.ordinal}-host"
    spec["parameters"] = {
        "workhorse.operation": SEMANTIC_EVIDENCE_CASE_OPERATION_V2,
        "workhorse.shape_id": shape.shape_id,
        "workhorse.backend_id": backend_id,
        "workhorse.semantic_case_id": case_id,
        "workhorse.B": law.B,
        "workhorse.temporal_k": law.dispatch_temporal_k,
        "workhorse.semantic_temporal_k": semantic_temporal_k,
        "workhorse.attempt_terminal_status": law.expected_attempt_terminal_status,
    }
    spec["execution"].update(
        {
            "start_next_tick": law.start_next_absolute_tick,
            "total_ticks": target,
            "checkpoint_ticks": [target],
            "resident_owner_span_ticks": law.total_ticks,
            "native_dispatch_window_ticks": law.dispatch_temporal_k,
            "source_window_ticks": law.dispatch_temporal_k,
            "event_window_capacity": 2 * shape.capacity,
        }
    )
    spec["execution"]["technical_gate"].update(
        {"total_ticks": target, "checkpoint_ticks": [target]}
    )
    spec["artifact_lifecycle"]["checkpoints"] = {
        "authority": "absolute_tick",
        "checkpoint_ticks": [target],
        "roles": [
            {
                "identity": f"checkpoint-tick-{target:012d}",
                "checkpoint_tick": target,
                "role": "terminal",
                "retention_class": "R2",
            }
        ],
    }
    spec["analysis"] = {
        "required_metrics": ["workhorse.semantic_case_provisional_completed"],
        "optional_metrics": [],
        "rules": [],
        "extension": {
            "id": OPERATION_EXTENSION_ID,
            "contract": operation,
            "contract_sha256": canonical_sha256(operation),
        },
    }
    plan = compile_campaign(
        spec,
        registry=cuda_full_evolution_workhorse_v2_profile_registry_v1(
            shape_id=shape.shape_id,
            operation=SEMANTIC_EVIDENCE_CASE_OPERATION_V2,
        ),
        planning_resource_accounting=(
            provider_module.cuda_full_evolution_workhorse_v2_signed_planning_resource_accounting_v2
        ),
    ).payload()
    return {
        **fixture,
        "spec": spec,
        "plan": plan,
        "batch_id": str(plan["physical_batches"][0]["batch_id"]),
        "contract": contract,
        "artifact_binding": artifact_binding,
        "envelope": envelope,
        "operation": operation,
        "law": law,
    }


def _launchpad_request_for_fixture(
    tmp_path: Path, fixture: Mapping[str, Any]
) -> ExecutionRequestV2:
    spec_path = tmp_path / "campaign.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(fixture["spec"]), encoding="utf-8")
    return ExecutionRequestV2.from_mapping(
        {
            "schema_version": "eiel_execution_request_v2",
            "purpose": "qualprep",
            "campaign_spec": str(spec_path),
            "output_root": str(fixture["run_root"]),
            "resource_wait": {
                "enabled": False,
                "poll_seconds": 300,
                "max_wait_seconds": None,
            },
            "qualification_profile": None,
            "baseline": {
                "schema_version": "eiel_baseline_declaration_v1",
                "profile_id": "EIEL_PRODUCTION_BASELINE_V1",
                "intent": "development",
                "deviations": [
                    {
                        "requirement_id": name,
                        "reason": "host-only Workhorse provider test",
                    }
                    for name in (
                        "evolution.full_evolution",
                        "runtime.production_cuda",
                        "science.wrc_variability",
                        "checkpoint.acs_two_slot",
                        "authority.device_fallback",
                    )
                ],
            },
        }
    )


def _bound_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    calibration: bool = False,
    backend_id: str = "phase_graph_v2",
    cuda_runtime: str = "12.8",
    shape_id: str = "w64h64p1024c2048",
    acs_slots: int | None = None,
    indexed_continuation: bool = False,
    selected_world_observation: bool = False,
) -> dict[str, Any]:
    fixture = (
        _calibration_fixture(
            tmp_path,
            monkeypatch,
            backend_id=backend_id,
            cuda_runtime=cuda_runtime,
        )
        if calibration
        else (
            _indexed_continuation_fixture(
                tmp_path,
                monkeypatch,
                selected_world_observation=selected_world_observation,
            )
            if indexed_continuation
            else _operation_fixture(
                tmp_path,
                monkeypatch,
                backend_id=backend_id,
                cuda_runtime=cuda_runtime,
                shape_id=shape_id,
                acs_slots=acs_slots,
            )
        )
    )
    times = iter((T0, T1, T2, T3))

    def resource_evidence(**values: Any) -> Mapping[str, Any]:
        route = values["route"]
        return {
            "resource_observation_id": "ROBS-WORKHORSE-HOST",
            "resource_observation": {
                "schema_version": "eiel_test_resource_observation_v1",
                "device_id": route["device_id"],
            },
            "resource_observed_at_utc": values["at_utc"],
            "resource_recommendation_id": "RADM-WORKHORSE-HOST",
            "resource_recommendation": {
                "schema_version": "eiel_test_resource_recommendation_v1",
                "decision": "ADMIT",
            },
            "live_provider_check": {
                "schema_version": "crca_test_live_provider_check_v1",
                "available": True,
                "device_id": route["device_id"],
                "event_transport": route["event_transport"],
            },
        }

    adapter = CanonicalCudaFullEvolutionV2LaunchpadAdapterV1(
        repo_root=REPO_ROOT,
        state_root=tmp_path / "scheduler-state",
        allow_test_state_root=True,
        resource_evidence_factory=resource_evidence,
        now_fn=lambda: next(times),
    )
    registry = cuda_full_evolution_workhorse_v2_provider_registry_v1(
        operation_adapter=adapter
    )
    provider = registry.get(fixture["plan"]["resolved_profile"])
    assert provider is not None
    bound_request = build_execution_request_v1(
        provider, fixture["plan"], fixture["batch_id"]
    ).payload()
    source_identity = {
        "campaign_spec_sha256": "8" * 64,
        "compiled_plan_sha256": fixture["plan"]["plan_sha256"],
        "bound_plan_sha256": fixture["plan"]["plan_sha256"],
        "git": {
            "commit": "a" * 40,
            "working_tree_clean": True,
            "working_tree_status_sha256": campaign_sha256([]),
        },
    }
    preflight = {
        "schema_version": "eiel_campaign_preflight_v1",
        "campaign_id": fixture["plan"]["campaign_id"],
        "plan_sha256": fixture["plan"]["plan_sha256"],
        "ready_for_execution": True,
        "reasons": [],
        "execution_authority_reasons": [],
        "artifacts": [],
        "runtime_environment": {},
    }
    request = _launchpad_request_for_fixture(tmp_path, fixture)
    projection = adapter.persist_shadow_projection_v1(
        target=fixture["run_root"]
        / "control/crca/shadow-execution-projection.json",
        campaign_plan=fixture["plan"],
        bound_execution_request=bound_request,
        baseline=request.baseline.payload(),
        source_identity=provider.crca_source_identity_v1(
            fixture["plan"], source_identity, preflight
        ),
        assurance=provider.crca_assurance_v1(fixture["plan"], preflight),
        preflight=preflight,
        attempt_nonce="launchpad:qualprep:workhorse-host",
        proposed_reason="one exact Workhorse row host Attempt",
    )
    authority = adapter.bind_attempt_authority_v1(
        target=fixture["run_root"] / "control/crca/attempt-authority.json",
        request=request.payload(),
        campaign_plan=fixture["plan"],
        bound_execution_request=bound_request,
        source_identity=source_identity,
        preflight=preflight,
        shadow_projection=projection,
    )
    assert authority["execution_authority_granted"] is True, authority
    return {
        **fixture,
        "adapter": adapter,
        "provider": provider,
        "bound_request": bound_request,
        "attempt_authority_path": Path(authority["attempt_authority_path"]),
    }


def _fake_shared_device_torch(
    device: Mapping[str, Any], *, free_bytes: int | None = None
) -> SimpleNamespace:
    properties = SimpleNamespace(
        name=device["name"],
        uuid=device["uuid"],
        pci_domain_id=device["pci_domain_id"],
        pci_bus_id=device["pci_bus_id"],
        pci_device_id=device["pci_device_id"],
        total_memory=device["total_memory_bytes"],
        multi_processor_count=device["multiprocessor_count"],
        major=7,
        minor=5,
    )
    total = int(device["total_memory_bytes"])
    free = total if free_bytes is None else free_bytes
    return SimpleNamespace(
        version=SimpleNamespace(cuda=device["cuda_runtime"]),
        cuda=SimpleNamespace(
            current_device=lambda: int(device["index"]),
            get_device_properties=lambda _index: properties,
            mem_get_info=lambda _index: (free, total),
            memory_reserved=lambda _index: 0,
            empty_cache=lambda: None,
            synchronize=lambda: None,
        ),
    )


def _fake_fused_loaded(fixture: Mapping[str, Any]) -> SimpleNamespace:
    from engine.perf.cuda_full_evolution_v2.workhorse.contract import (
        workhorse_backend_v2,
    )
    from engine.perf.cuda_full_evolution_v2.workhorse.layout import (
        workhorse_layout_v2,
    )

    shape = fixture["contract"].shape
    backend = workhorse_backend_v2(fixture["contract"].backend_id)
    kernels = backend.topology_payload(shape)["kernels_per_dispatch"]
    device = fixture["device"]
    runtime = {
        "runtime_inspected": True,
        "shape_id": shape.shape_id,
        "profile_id": shape.profile_id,
        "route_id": shape.route_id,
        "backend_id": backend.backend_id,
        "device_uuid": device["uuid"],
        "target_sm": device["sm"],
        "total_device_bytes": device["total_memory_bytes"],
        "multiprocessor_count": device["multiprocessor_count"],
        "maximum_grid_x": 2_147_483_647,
        "maximum_grid_y": 65_535,
        "maximum_threads_per_block": 1024,
        "maximum_threads_per_sm": 2048,
        "registers_per_sm": 65_536,
        "shared_memory_per_sm": 65_536,
        "maximum_blocks_per_sm": 16,
        "warp_size": 32,
        "compiled_kernel_resources": [
            {
                "kernel": kernel,
                "registers_per_thread": 32,
                "static_shared_bytes": 0,
                "local_bytes_per_thread": 0,
                "maximum_threads_per_block": 1024,
                "active_blocks_per_sm_at_compiled_block": 4,
            }
            for kernel in kernels
        ],
    }
    return SimpleNamespace(
        shape=shape,
        backend=backend,
        layout=workhorse_layout_v2(shape),
        module=SimpleNamespace(runtime_inspect=lambda: deepcopy(runtime)),
    )


def _bind_workhorse_parent(fixture: Mapping[str, Any]) -> dict[str, Any]:
    return dict(
        fixture["adapter"].bind_workhorse_parent_context_v2(
            attempt_authority_path=fixture["attempt_authority_path"],
            target=(
                fixture["run_root"]
                / "batches"
                / fixture["batch_id"]
                / "workhorse-parent-context.json"
            ),
            campaign_plan=fixture["plan"],
            bound_execution_request=fixture["bound_request"],
        )
    )


def test_exactly_one_workhorse_row_compiles_to_one_provider_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _operation_fixture(tmp_path, monkeypatch)
    provider = cuda_full_evolution_workhorse_v2_provider_registry_v1().get(
        fixture["plan"]["resolved_profile"]
    )
    assert provider is not None
    request = provider.translate_batch(fixture["plan"], fixture["batch_id"])
    assert request.cell_ids and len(request.cell_ids) == 1
    assert request.total_ticks == fixture["contract"].total_ticks
    assert request.start_next_tick == fixture["contract"].start_next_tick
    assert request.generations is None
    assert front_door._is_cuda_full_evolution_workhorse_v2_spec(fixture["spec"])
    assert (
        front_door._provider_registry_for_plan_v1(fixture["plan"])
        .get(fixture["plan"]["resolved_profile"])
        .provider_id
        == provider.provider_id
    )
    assert [
        row.profile_id
        for row in front_door._profile_registry_for_plan_v1(
            fixture["plan"]
        ).profiles()
    ] == [fixture["plan"]["resolved_profile"]]
    hostile = deepcopy(fixture["plan"])
    hostile["physical_batches"].append(deepcopy(hostile["physical_batches"][0]))
    with pytest.raises(CampaignProviderError, match="exactly one physical batch"):
        provider.translate_batch(hostile, fixture["batch_id"])


def test_exactly_one_calibration_compiles_and_bad_artifact_fails_before_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _calibration_fixture(tmp_path, monkeypatch)
    provider = cuda_full_evolution_workhorse_v2_provider_registry_v1(
        operation_adapter=SimpleNamespace()
    ).get(fixture["plan"]["resolved_profile"])
    assert provider is not None
    request = provider.translate_batch(fixture["plan"], fixture["batch_id"])
    payload = request.payload()
    assert payload["tick_native_control"]["compiled_temporal_k"] == list(
        fixture["contract"].shape.temporal_k_candidates
    )
    assert "workhorse.B" not in payload["parameter_arrays"]
    assert "workhorse.temporal_k" not in payload["parameter_arrays"]
    assert front_door._is_cuda_full_evolution_workhorse_v2_spec(fixture["spec"])
    compiled_profile = front_door._profile_registry_for_plan_v1(
        fixture["plan"]
    ).profiles()[0]
    assert set(compiled_profile.parameter_contracts) == {
        "workhorse.operation",
        "workhorse.shape_id",
        "workhorse.backend_id",
        "workhorse.compiled_temporal_k_sha256",
        "workhorse.reserve_bytes",
    }

    artifact_validation_called = False
    physical_run_called = False

    def invalid_artifact(_path: str | Path) -> Mapping[str, Any]:
        nonlocal artifact_validation_called
        artifact_validation_called = True
        raise ValueError("sealed artifact differs")

    def forbidden_physical_run(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal physical_run_called
        physical_run_called = True
        raise AssertionError("physical calibration ran before preflight passed")

    monkeypatch.setattr(provider_module, "workhorse_artifact_binding_v2", invalid_artifact)
    monkeypatch.setattr(
        provider_module,
        "run_and_seal_workhorse_device_envelope_calibration_v2",
        forbidden_physical_run,
    )
    reasons = provider.readiness_reasons(fixture["plan"], request)
    assert len(reasons) == 1
    assert "sealed artifact differs" in reasons[0].message
    assert artifact_validation_called is True
    assert physical_run_called is False
    assert not (fixture["run_root"] / "control/crca/attempt-authority.json").exists()


def test_calibration_device_identity_requires_exact_cuda_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _calibration_fixture(tmp_path, monkeypatch)
    provider = cuda_full_evolution_workhorse_v2_provider_registry_v1(
        operation_adapter=SimpleNamespace()
    ).get(fixture["plan"]["resolved_profile"])
    assert provider is not None
    for runtime in (None, "   ", " 12.8 "):
        hostile = deepcopy(fixture["plan"])
        extension = hostile["analysis"]["extension"]
        operation = extension["contract"]
        if runtime is None:
            operation["device_identity"].pop("cuda_runtime")
        else:
            operation["device_identity"]["cuda_runtime"] = runtime
        extension["contract_sha256"] = canonical_sha256(operation)
        with pytest.raises(CampaignProviderError, match="device identity|cuda_runtime"):
            provider.translate_batch(hostile, fixture["batch_id"])


def test_shared_cuda_device_identity_flows_through_calibration_and_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse import probe as probe_module

    fixture = _bound_attempt(
        tmp_path,
        monkeypatch,
        calibration=True,
        backend_id="fused_cta_temporal_v2",
    )
    parent = _bind_workhorse_parent(fixture)
    fake_torch = _fake_shared_device_torch(fixture["device"])
    loaded = _fake_fused_loaded(fixture)
    monkeypatch.setattr(probe_module, "_torch", lambda: fake_torch)
    monkeypatch.setattr(probe_module, "_load", lambda *_args: loaded)
    monkeypatch.setattr(probe_module, "_allocate_owner", lambda *_args: object())
    monkeypatch.setattr(probe_module, "_initialize_owner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        probe_module, "_prepare_graph_if_needed", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(probe_module, "_release_graph", lambda *_args: None)
    monkeypatch.setattr(
        canonical_rows,
        "workhorse_artifact_binding_v2",
        lambda _path: deepcopy(fixture["artifact_binding"]),
    )

    receipt = canonical_rows.run_and_seal_workhorse_device_envelope_calibration_v2(
        fixture["operation"]["artifact"]["path"],
        output_path=tmp_path / "calibration.json",
        source_identity={"source_sha256": "a" * 64},
        canonical_parent_authority_binding=parent,
        reserve_bytes=fixture["operation"]["reserve_policy"][
            "reserve_bytes_requested"
        ],
    )
    assert receipt["device_identity"] == fixture["device"]
    assert receipt["device_identity"]["cuda_runtime"] == "12.8"
    assert canonical_rows.validate_workhorse_device_envelope_calibration_receipt_v2(
        receipt["receipt_path"]
    )["device_identity"] == fixture["device"]


def test_shared_cuda_device_identity_flows_through_row_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse import authority as authority_module
    from engine.perf.cuda_full_evolution_v2.workhorse import probe as probe_module
    from engine.perf.cuda_full_evolution_v2.workhorse.layout import (
        WorkhorseOwnerComponentsV2,
        admit_workhorse_vram_v2,
    )

    fixture = _bound_attempt(
        tmp_path,
        monkeypatch,
        backend_id="fused_cta_temporal_v2",
    )
    parent = _bind_workhorse_parent(fixture)
    fake_torch = _fake_shared_device_torch(fixture["device"], free_bytes=0)
    loaded = _fake_fused_loaded(fixture)
    owner = WorkhorseOwnerComponentsV2(
        backend_id=fixture["contract"].backend_id,
        temporal_k=fixture["contract"].temporal_k,
        graph_object_bytes=0,
        graph_exec_bytes=0,
        allocator_owner_overhead_bytes=0,
        driver_unattributed_owner_bytes=0,
        process_baseline_bytes=0,
        checkpoint_device_staging_bytes_per_world=0,
        measurement_receipt_sha256="b" * 64,
    )
    planned = admit_workhorse_vram_v2(
        fixture["contract"],
        free_bytes=fixture["device"]["total_memory_bytes"],
        total_device_bytes=fixture["device"]["total_memory_bytes"],
        owner_components=owner,
    ).to_dict()
    assert planned["admitted"] is True
    envelope = {
        "receipt_path": str((tmp_path / "envelope.json").resolve()),
        "receipt_file_sha256": "c" * 64,
        "receipt_content_sha256": "d" * 64,
        "artifact": fixture["artifact_binding"],
        "source_identity": {"source_sha256": "e" * 64},
        "source_identity_sha256": "f" * 64,
        "device_identity": fixture["device"],
        "capacity_basis": {"reserve_bytes_requested": 0},
        "owner_components_by_k": {
            str(fixture["contract"].temporal_k): owner.to_dict()
        },
        "expected_rows": [
            {
                "B": fixture["contract"].B,
                "temporal_k": fixture["contract"].temporal_k,
                "planned_admission": planned,
            }
        ],
    }
    monkeypatch.setattr(probe_module, "_torch", lambda: fake_torch)
    monkeypatch.setattr(probe_module, "_load", lambda *_args: loaded)
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_device_envelope_calibration_receipt_v2",
        lambda _path: deepcopy(envelope),
    )
    monkeypatch.setattr(
        canonical_rows,
        "workhorse_artifact_binding_v2",
        lambda _path: deepcopy(fixture["artifact_binding"]),
    )
    monkeypatch.setattr(
        authority_module,
        "validate_workhorse_semantic_gate_receipt_v2",
        lambda *_args, **_kwargs: {"semantic": "exact"},
    )
    monkeypatch.setattr(
        authority_module,
        "validate_authoritative_micro_denominator_v2",
        lambda _path, *, expected_device_identity: {
            "device_identity": dict(expected_device_identity)
        },
    )
    captured: dict[str, Any] = {}

    def capture_result(_path: str | Path, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"outcome": kwargs["outcome"]}

    monkeypatch.setattr(canonical_rows, "_seal_row_execution_result", capture_result)
    result = canonical_rows.run_workhorse_canonical_admitted_row_v2(
        fixture["artifact_binding"]["path"],
        envelope_receipt_path=envelope["receipt_path"],
        canonical_parent_authority_binding=parent,
        authoritative_micro_receipt_path=tmp_path / "micro.json",
        semantic_gate_receipt_path=tmp_path / "semantic.json",
        science_continuation_contract_sha256="7" * 64,
        event_sink_root=tmp_path / "events",
        execution_result_path=tmp_path / "row.json",
        B=fixture["contract"].B,
        temporal_k=fixture["contract"].temporal_k,
        warmup_ticks=1,
        measurement_window_ticks=1,
        measured_windows=5,
    )
    assert result["outcome"] == "RUNTIME_HEADROOM_REJECTED"
    assert captured["envelope"]["device_identity"]["cuda_runtime"] == "12.8"
    assert captured["runtime_admission"]["admitted"] is False


def test_physical_calibration_rejects_substituted_cuda_runtime_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse import probe as probe_module

    fixture = _bound_attempt(
        tmp_path,
        monkeypatch,
        calibration=True,
        backend_id="fused_cta_temporal_v2",
        cuda_runtime="11.8",
    )
    parent = _bind_workhorse_parent(fixture)
    actual_device = {**fixture["device"], "cuda_runtime": "12.8"}
    fake_torch = _fake_shared_device_torch(actual_device)
    loaded = _fake_fused_loaded({**fixture, "device": actual_device})
    allocated = False

    def forbidden_allocate(*_args: Any) -> object:
        nonlocal allocated
        allocated = True
        return object()

    monkeypatch.setattr(probe_module, "_torch", lambda: fake_torch)
    monkeypatch.setattr(probe_module, "_load", lambda *_args: loaded)
    monkeypatch.setattr(probe_module, "_allocate_owner", forbidden_allocate)
    monkeypatch.setattr(
        canonical_rows,
        "workhorse_artifact_binding_v2",
        lambda _path: deepcopy(fixture["artifact_binding"]),
    )
    with pytest.raises(WorkhorseCanonicalRowError, match="differs from parent"):
        canonical_rows.run_and_seal_workhorse_device_envelope_calibration_v2(
            fixture["operation"]["artifact"]["path"],
            output_path=tmp_path / "forbidden-calibration.json",
            source_identity={"source_sha256": "a" * 64},
            canonical_parent_authority_binding=parent,
            reserve_bytes=0,
        )
    assert allocated is False
    assert not (tmp_path / "forbidden-calibration.json").exists()


def test_calibration_reopens_one_scheduler_before_one_physical_run_and_finalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _bound_attempt(tmp_path, monkeypatch, calibration=True)
    target = (
        fixture["run_root"]
        / "batches"
        / fixture["batch_id"]
        / "workhorse-parent-context.json"
    )

    def component(request: dict[str, Any], component_id: str) -> dict[str, Any]:
        return next(
            row
            for row in request["execution_route_contract"]["components"]
            if row["component_id"] == component_id
        )

    hostile_requests: list[dict[str, Any]] = []
    cross_operation = deepcopy(fixture["bound_request"])
    component(
        cross_operation, "cuda_full_evolution_workhorse_v2_operation"
    )["operation"] = "EXECUTION_ROW"
    cross_operation["resident_request"]["parameter_arrays"][
        "workhorse.operation"
    ] = ["EXECUTION_ROW"]
    hostile_requests.append(cross_operation)
    cross_artifact = deepcopy(fixture["bound_request"])
    component(
        cross_artifact, "cuda_full_evolution_workhorse_v2_artifact_unit"
    )["artifact_sha256"] = "f" * 64
    hostile_requests.append(cross_artifact)
    cross_k = deepcopy(fixture["bound_request"])
    component(cross_k, "cuda_full_evolution_workhorse_v2_operation")[
        "compiled_temporal_k"
    ] = [1]
    component(cross_k, "cuda_full_evolution_workhorse_v2_device_selection")[
        "compiled_temporal_k"
    ] = [1]
    hostile_requests.append(cross_k)
    cross_device = deepcopy(fixture["bound_request"])
    component(cross_device, "cuda_full_evolution_workhorse_v2_device_selection")[
        "device_identity"
    ]["uuid"] = "GPU-HOSTILE"
    hostile_requests.append(cross_device)
    for hostile in hostile_requests:
        with pytest.raises(CudaFullEvolutionV2LaunchpadAdapterError):
            fixture["adapter"].bind_workhorse_parent_context_v2(
                attempt_authority_path=fixture["attempt_authority_path"],
                target=target,
                campaign_plan=fixture["plan"],
                bound_execution_request=hostile,
            )
        assert not target.exists()

    physical_calls: list[dict[str, Any]] = []

    def physical_runner(
        artifact_path: str | Path,
        *,
        output_path: str | Path,
        source_identity: Mapping[str, Any],
        canonical_parent_authority_binding: Mapping[str, Any],
        reserve_bytes: int,
    ) -> dict[str, Any]:
        parent = validate_workhorse_canonical_parent_authority_binding_v2(
            dict(canonical_parent_authority_binding),
            expected_operation=DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2,
        )
        assert parent["compiled_temporal_k"] == list(
            fixture["contract"].shape.temporal_k_candidates
        )
        assert parent["device_identity"] == fixture["device"]
        assert Path(parent["attempt_authority_path"]) == fixture[
            "attempt_authority_path"
        ]
        output = Path(output_path)
        _write_file(output, b'{"host":"physical-calibration"}')
        physical_calls.append(parent)
        return {
            "shape_id": fixture["operation"]["shape_id"],
            "backend_id": fixture["operation"]["backend_id"],
            "device_identity": fixture["device"],
            "owner_components_by_k": {
                str(value): {} for value in fixture["operation"]["compiled_temporal_k"]
            },
            "capacity_basis": {
                "reserve_bytes_requested": reserve_bytes,
            },
            "receipt_path": str(output.resolve()),
            "receipt_file_sha256": _file_sha256(output),
            "receipt_content_sha256": "c" * 64,
            "source_identity": dict(source_identity),
        }

    monkeypatch.setattr(
        provider_module,
        "run_and_seal_workhorse_device_envelope_calibration_v2",
        physical_runner,
    )
    recorded: list[Path] = []
    batch_root = fixture["run_root"] / "batches" / fixture["batch_id"]
    context = SimpleNamespace(
        run_root=fixture["run_root"],
        batch_root=batch_root,
        record_artifact=lambda path, **_: recorded.append(Path(path)),
    )
    request = fixture["provider"].translate_batch(
        fixture["plan"], fixture["batch_id"]
    )
    batch_result = fixture["provider"].execute_batch(
        fixture["plan"], request, context
    )
    assert len(physical_calls) == 1
    assert target in recorded
    manifest_path = batch_root / "manifest.json"
    manifest_path.write_text(json.dumps(batch_result), encoding="ascii")
    terminal = fixture["adapter"].terminalize_attempt_authority_v1(
        attempt_authority_path=fixture["attempt_authority_path"],
        campaign_plan=fixture["plan"],
        execution_status="completed",
        execution_receipt={
            "status": "completed",
            "plan_sha256": fixture["plan"]["plan_sha256"],
        },
        result_binding={"execution_status": "completed"},
    )

    def final_sealer(
        output_path: str | Path,
        *,
        calibration_receipt_path: str | Path,
        canonical_terminal_binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        terminal_path = Path(canonical_terminal_binding["terminal_bundle_path"])
        reopened = json.loads(terminal_path.read_text(encoding="ascii"))
        assert reopened["attempt_terminal"]["status"] == "COMPLETED"
        target_tick = fixture["operation"]["absolute_tick_span"][
            "target_next_tick"
        ]
        assert all(
            value == target_tick
            for key, value in canonical_terminal_binding["final_frontiers"].items()
            if key != "schema_version"
        )
        assert Path(calibration_receipt_path).is_file()
        output = Path(output_path)
        _write_file(output, b'{"host":"terminalized-calibration"}')
        return {
            "receipt_path": str(output.resolve()),
            "receipt_file_sha256": _file_sha256(output),
            "receipt_content_sha256": "d" * 64,
        }

    monkeypatch.setattr(
        provider_module,
        "seal_workhorse_terminalized_device_envelope_calibration_receipt_v2",
        final_sealer,
    )
    finalized = fixture["provider"].finalize_attempt_terminal_v2(
        plan=fixture["plan"],
        run_root=fixture["run_root"],
        canonical_terminal_binding=terminal["workhorse_terminal_binding"],
    )
    assert finalized["operation"] == DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2
    assert finalized["lifecycle_binding"]["retention_class"] == "R3"
    assert Path(finalized["final_receipt_path"]).is_file()


def test_calibration_expired_and_partial_terminal_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expired = _bound_attempt(tmp_path / "expired", monkeypatch, calibration=True)
    expired["adapter"].now_fn = lambda: "2026-08-27T05:00:00Z"
    with pytest.raises(Exception, match="expired"):
        expired["adapter"].bind_workhorse_parent_context_v2(
            attempt_authority_path=expired["attempt_authority_path"],
            target=expired["run_root"]
            / "batches"
            / expired["batch_id"]
            / "workhorse-parent-context.json",
            campaign_plan=expired["plan"],
            bound_execution_request=expired["bound_request"],
        )

    partial = _bound_attempt(tmp_path / "partial", monkeypatch, calibration=True)
    batch_root = partial["run_root"] / "batches" / partial["batch_id"]
    receipt_path = batch_root / "workhorse-device-envelope-calibration-measurement.json"
    _write_file(receipt_path, b'{"host":"partial"}')
    target_tick = partial["operation"]["absolute_tick_span"]["target_next_tick"]
    manifest = {
        "status": "completed",
        "checkpoint": {"next_absolute_tick": target_tick},
        "result": {
            "schema_version": (
                "eiel_workhorse_device_envelope_calibration_provider_result_v2"
            ),
            "operation": DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2,
            "shape_id": partial["operation"]["shape_id"],
            "backend_id": partial["operation"]["backend_id"],
            "artifact_sha256": partial["operation"]["artifact"]["sha256"],
            "sealed_artifact_identity_sha256": partial["operation"]["artifact"][
                "sealed_artifact_identity_sha256"
            ],
            "compiled_temporal_k": partial["operation"]["compiled_temporal_k"],
            "device_identity": partial["operation"]["device_identity"],
            "reserve_policy": partial["operation"]["reserve_policy"],
            "calibration_receipt_path": str(receipt_path.resolve()),
            "calibration_receipt_file_sha256": _file_sha256(receipt_path),
            "calibration_receipt_content_sha256": "e" * 64,
            "absolute_frontiers": {
                "start_next_tick": 0,
                "target_next_tick": target_tick,
                "committed_next_tick": target_tick - 1,
            },
        },
    }
    batch_root.mkdir(parents=True, exist_ok=True)
    (batch_root / "manifest.json").write_text(json.dumps(manifest), encoding="ascii")
    with pytest.raises(
        CudaFullEvolutionV2LaunchpadAdapterError,
        match="terminal bindings",
    ):
        partial["adapter"].terminalize_attempt_authority_v1(
            attempt_authority_path=partial["attempt_authority_path"],
            campaign_plan=partial["plan"],
            execution_status="completed",
            execution_receipt={
                "status": "completed",
                "plan_sha256": partial["plan"]["plan_sha256"],
            },
            result_binding={"execution_status": "completed"},
        )


def test_persisted_row_validator_rejects_failed_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = {"parent": "exact"}
    schedule = canonical_rows.workhorse_fixed_tick_measurement_schedule_v3(
        temporal_k=1,
        warmup_ticks=1,
        measurement_window_ticks=1,
        measured_windows=5,
    )
    comparison = canonical_rows.workhorse_fixed_tick_comparison_identity_v3(
        schedule=schedule,
        start_next_tick=0,
        source_checkpoint_identity_sha256=None,
    )
    envelope = {
        "receipt_file_sha256": "1" * 64,
        "receipt_content_sha256": "2" * 64,
        "shape_id": "shape",
        "backend_id": "backend",
        "artifact": {"artifact": "exact"},
        "source_identity_sha256": "3" * 64,
        "device_identity": {"device": "exact"},
        "expected_rows": [
            {
                "B": 2,
                "temporal_k": 1,
                "planned_admission": {"admitted": True},
            }
        ],
    }
    value = {
        "claim_class": "TECHNICAL_ONLY",
        "execution_authority_granted": False,
        "qualification_authority_granted": False,
        "science_authority_granted": False,
        "row_kind": "ATTEMPT_EXECUTED",
        "shape_id": "shape",
        "backend_id": "backend",
        "B": 2,
        "temporal_k": 1,
        "envelope_binding": {
            "receipt_path": "envelope.json",
            "receipt_file_sha256": "1" * 64,
            "receipt_content_sha256": "2" * 64,
        },
        "artifact": envelope["artifact"],
        "source_identity_sha256": "3" * 64,
        "device_identity": envelope["device_identity"],
        "outcome": "MEASURED",
        "p10_world_ticks_per_second": 1.0,
        "minimum_fixed_window_world_ticks_per_second": 1.0,
        "median_fixed_window_world_ticks_per_second": 1.0,
        "measurement_comparison_identity": comparison,
        "resource_ledger_conforms": True,
        "execution_result": {
            "path": "execution.json",
            "file_sha256": "4" * 64,
            "content_sha256": "5" * 64,
        },
        "canonical_parent_authority_binding": parent,
        "canonical_terminal_binding": {"terminal": "failed"},
        "canonical_preflight_rejection_binding": None,
    }
    execution = {
        "receipt_file_sha256": "4" * 64,
        "receipt_content_sha256": "5" * 64,
        "contract": {"B": 2, "temporal_k": 1, "start_next_tick": 0},
        "outcome": "MEASURED",
        "p10_world_ticks_per_second": 1.0,
        "minimum_fixed_window_world_ticks_per_second": 1.0,
        "median_fixed_window_world_ticks_per_second": 1.0,
        "measurement_schedule": schedule,
        "checkpoint_binding": {"source_checkpoint_identity_sha256": None},
        "resource_observation": {"resource_ledger_conforms": True},
        "canonical_parent_authority_binding": parent,
        "absolute_frontiers": {"target_next_tick": 6},
    }
    monkeypatch.setattr(
        canonical_rows,
        "_read_sealed_receipt",
        lambda *_args, **_kwargs: (tmp_path / "row.json", value),
    )
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_device_envelope_calibration_receipt_v2",
        lambda _path: envelope,
    )
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_canonical_row_execution_result_v2",
        lambda _path: execution,
    )
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_canonical_parent_authority_binding_v2",
        lambda *_args, **_kwargs: parent,
    )
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_canonical_terminal_binding_v2",
        lambda *_args, **_kwargs: {
            "attempt_terminal_status": "FAILED",
            "final_frontiers": {
                "last_dispatched_next_tick": 0,
                "last_computed_next_tick": 0,
                "durable_checkpoint_next_tick": 0,
                "published_artifact_next_tick": 0,
            },
        },
    )
    with pytest.raises(
        WorkhorseCanonicalRowError,
        match="persisted technical row requires exact completed terminal",
    ):
        canonical_rows.validate_workhorse_technical_row_receipt_v2(
            tmp_path / "row.json"
        )


def test_row_measurement_recomputes_rate_and_binds_exact_ack_index() -> None:
    ack = {
        "receipt_path": "/host/event-sink/sample-0/sink_receipt.json",
        "receipt_file_sha256": "f" * 64,
        "receipt_body_sha256": "e" * 64,
        "dispatch_content_sha256": "d" * 64,
        "start_next_absolute_tick": 11,
        "end_next_absolute_tick": 12,
    }
    body = {
        "sample_index": 0,
        "window_index": 0,
        "dispatch_index_in_window": 0,
        "start_next_absolute_tick": 11,
        "end_next_absolute_tick": 12,
        "timing_source": "cuda_event",
        "cuda_event_elapsed_ms": 2.0,
        "event_canonicalization_tail_ms": 7.5,
        "event_canonicalization_tail_scope": (
            "POST_CUDA_EVENT_EXCLUDED_FROM_WORLD_TICKS_PER_SECOND"
        ),
        "committed_world_ticks": 8,
        "world_ticks_per_second": 4_000.0,
        "event_sink_receipt_index": 1,
        "event_sink_receipt_path": ack["receipt_path"],
        "event_sink_receipt_file_sha256": ack["receipt_file_sha256"],
        "event_sink_receipt_body_sha256": ack["receipt_body_sha256"],
        "event_sink_dispatch_content_sha256": ack[
            "dispatch_content_sha256"
        ],
    }
    sample = {**body, "measurement_sha256": canonical_sha256(body)}
    validated = canonical_rows._validate_row_measurement_sample_v2(
        sample,
        sample_index=0,
        window_index=0,
        dispatch_index_in_window=0,
        start_next_absolute_tick=11,
        end_next_absolute_tick=12,
        committed_world_ticks=8,
        world_ticks_per_second=4_000.0,
        event_sink_receipt_index=1,
        event_sink_receipt=ack,
    )
    assert validated["event_canonicalization_tail_ms"] == 7.5
    assert validated["world_ticks_per_second"] == 8 * 1000.0 / 2.0
    for hostile_tail in (-0.1, float("inf"), float("nan")):
        hostile_body = {**body, "event_canonicalization_tail_ms": hostile_tail}
        hostile = {
            **hostile_body,
            "measurement_sha256": canonical_sha256(hostile_body),
        }
        with pytest.raises(WorkhorseCanonicalRowError, match="timing sample"):
            canonical_rows._validate_row_measurement_sample_v2(
                hostile,
                sample_index=0,
                window_index=0,
                dispatch_index_in_window=0,
                start_next_absolute_tick=11,
                end_next_absolute_tick=12,
                committed_world_ticks=8,
                world_ticks_per_second=4_000.0,
                event_sink_receipt_index=1,
                event_sink_receipt=ack,
            )

    hostile_rate_body = {**body, "world_ticks_per_second": 3_999.0}
    hostile_rate = {
        **hostile_rate_body,
        "measurement_sha256": canonical_sha256(hostile_rate_body),
    }
    with pytest.raises(WorkhorseCanonicalRowError, match="timing sample"):
        canonical_rows._validate_row_measurement_sample_v2(
            hostile_rate,
            sample_index=0,
            window_index=0,
            dispatch_index_in_window=0,
            start_next_absolute_tick=11,
            end_next_absolute_tick=12,
            committed_world_ticks=8,
            world_ticks_per_second=3_999.0,
            event_sink_receipt_index=1,
            event_sink_receipt=ack,
        )

    other_ack = {
        **ack,
        "receipt_path": "/host/event-sink/sample-1/sink_receipt.json",
        "receipt_file_sha256": "c" * 64,
    }
    with pytest.raises(WorkhorseCanonicalRowError, match="timing sample"):
        canonical_rows._validate_row_measurement_sample_v2(
            sample,
            sample_index=0,
            window_index=0,
            dispatch_index_in_window=0,
            start_next_absolute_tick=11,
            end_next_absolute_tick=12,
            committed_world_ticks=8,
            world_ticks_per_second=4_000.0,
            event_sink_receipt_index=2,
            event_sink_receipt=other_ack,
        )


@pytest.mark.parametrize("temporal_k", [1, 2, 4, 8, 16, 32, 64])
def test_fixed_tick_measurement_schedule_is_exactly_comparable_across_k(
    temporal_k: int,
) -> None:
    schedule = canonical_rows.workhorse_fixed_tick_measurement_schedule_v3(
        temporal_k=temporal_k,
        warmup_ticks=192,
        measurement_window_ticks=64,
        measured_windows=20,
    )
    assert schedule["warmup_dispatches"] == 192 // temporal_k
    assert schedule["dispatches_per_measurement_window"] == 64 // temporal_k
    assert schedule["measured_dispatches"] == 20 * 64 // temporal_k
    assert schedule["measured_ticks"] == 1280
    assert schedule["total_ticks"] == 1472
    assert schedule["percentile_method"] == "HYNDMAN_FAN_R7_LINEAR_INTERPOLATION"
    assert (
        canonical_rows.validate_workhorse_fixed_tick_measurement_schedule_v3(
            schedule, temporal_k=temporal_k
        )
        == schedule
    )


def test_fixed_tick_measurement_schedule_rejects_nondivisible_k() -> None:
    with pytest.raises(WorkhorseCanonicalRowError, match="must be divisible"):
        canonical_rows.workhorse_fixed_tick_measurement_schedule_v3(
            temporal_k=32,
            warmup_ticks=192,
            measurement_window_ticks=48,
            measured_windows=20,
        )


def test_offline_aggregate_rejects_mixed_fixed_tick_horizons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = {
        "receipt_path": str(tmp_path / "envelope.json"),
        "receipt_file_sha256": "1" * 64,
        "receipt_content_sha256": "2" * 64,
        "shape_id": "shape",
        "backend_id": "backend",
        "artifact": {"artifact": "exact"},
        "source_identity_sha256": "3" * 64,
        "device_identity": {"device": "exact"},
        "expected_rows": [
            {"B": 2, "temporal_k": 1},
            {"B": 2, "temporal_k": 2},
        ],
    }
    common = {
        "row_kind": "ATTEMPT_EXECUTED",
        "shape_id": "shape",
        "backend_id": "backend",
        "B": 2,
        "envelope_binding": {
            "receipt_path": envelope["receipt_path"],
            "receipt_file_sha256": envelope["receipt_file_sha256"],
            "receipt_content_sha256": envelope["receipt_content_sha256"],
        },
        "artifact": envelope["artifact"],
        "source_identity_sha256": envelope["source_identity_sha256"],
        "device_identity": envelope["device_identity"],
        "outcome": "MEASURED",
        "p10_world_ticks_per_second": 1.0,
        "minimum_fixed_window_world_ticks_per_second": 1.0,
        "median_fixed_window_world_ticks_per_second": 1.0,
        "resource_ledger_conforms": True,
    }
    rows = {}
    for temporal_k, warmup_ticks in ((1, 192), (2, 194)):
        schedule = canonical_rows.workhorse_fixed_tick_measurement_schedule_v3(
            temporal_k=temporal_k,
            warmup_ticks=warmup_ticks,
            measurement_window_ticks=64,
            measured_windows=20,
        )
        path = str(tmp_path / f"row-k{temporal_k}.json")
        rows[path] = {
            **common,
            "temporal_k": temporal_k,
            "measurement_comparison_identity": (
                canonical_rows.workhorse_fixed_tick_comparison_identity_v3(
                    schedule=schedule,
                    start_next_tick=0,
                    source_checkpoint_identity_sha256=None,
                )
            ),
        }
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_device_envelope_calibration_receipt_v2",
        lambda _path: envelope,
    )
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_technical_row_receipt_v2",
        lambda path: rows[str(path)],
    )
    with pytest.raises(
        WorkhorseCanonicalRowError,
        match="mixes fixed-tick horizon",
    ):
        canonical_rows.aggregate_workhorse_technical_rows_v2(
            tmp_path / "aggregate.json",
            envelope_receipt_path=envelope["receipt_path"],
            row_receipt_paths=list(rows),
        )


def test_fixed_tick_window_recomputes_summed_cuda_time() -> None:
    dispatches = []
    for sample_index, elapsed_ms in enumerate((2.0, 3.0)):
        body = {
            "sample_index": sample_index,
            "window_index": 0,
            "dispatch_index_in_window": sample_index,
            "start_next_absolute_tick": 100 + sample_index * 32,
            "end_next_absolute_tick": 132 + sample_index * 32,
            "timing_source": "cuda_event",
            "cuda_event_elapsed_ms": elapsed_ms,
            "event_canonicalization_tail_ms": 1.0,
            "event_canonicalization_tail_scope": (
                "POST_CUDA_EVENT_EXCLUDED_FROM_WORLD_TICKS_PER_SECOND"
            ),
            "committed_world_ticks": 64,
            "world_ticks_per_second": 64 * 1000.0 / elapsed_ms,
            "event_sink_receipt_index": sample_index,
            "event_sink_receipt_path": f"/sink/{sample_index}",
            "event_sink_receipt_file_sha256": "a" * 64,
            "event_sink_receipt_body_sha256": "b" * 64,
            "event_sink_dispatch_content_sha256": "c" * 64,
        }
        dispatches.append(
            {**body, "measurement_sha256": canonical_sha256(body)}
        )
    window_body = {
        "window_index": 0,
        "start_next_absolute_tick": 100,
        "end_next_absolute_tick": 164,
        "measurement_window_ticks": 64,
        "dispatch_measurement_start_index": 0,
        "dispatch_measurement_count": 2,
        "cuda_event_elapsed_ms_sum": 5.0,
        "committed_world_ticks": 128,
        "world_ticks_per_second": 25_600.0,
        "dispatch_measurement_sha256s": [
            row["measurement_sha256"] for row in dispatches
        ],
    }
    window = {
        **window_body,
        "window_measurement_sha256": canonical_sha256(window_body),
    }
    assert canonical_rows._validate_row_measurement_window_v3(
        window,
        window_index=0,
        start_next_absolute_tick=100,
        end_next_absolute_tick=164,
        measurement_window_ticks=64,
        dispatch_measurement_start_index=0,
        committed_world_ticks=128,
        dispatch_measurements=dispatches,
        world_ticks_per_second=25_600.0,
    ) == window
    hostile_body = {**window_body, "cuda_event_elapsed_ms_sum": 4.0}
    hostile = {
        **hostile_body,
        "window_measurement_sha256": canonical_sha256(hostile_body),
    }
    with pytest.raises(WorkhorseCanonicalRowError, match="window changed"):
        canonical_rows._validate_row_measurement_window_v3(
            hostile,
            window_index=0,
            start_next_absolute_tick=100,
            end_next_absolute_tick=164,
            measurement_window_ticks=64,
            dispatch_measurement_start_index=0,
            committed_world_ticks=128,
            dispatch_measurements=dispatches,
            world_ticks_per_second=25_600.0,
        )


def test_row_resource_conformance_counts_allocator_owner_overhead() -> None:
    ledger = {
        "tensor_resident_bytes": 100,
        "allocator_owner_overhead_bytes": 20,
        "incremental_device_bytes": 130,
    }
    conforms, allocator_delta, free_delta = (
        canonical_rows._row_resource_conformance_v2(
            ledger=ledger,
            allocator_baseline_bytes=10,
            allocator_peak_bytes=125,
            free_vram_before_bytes=1_000,
            free_vram_after_bytes=875,
        )
    )
    assert conforms is True
    assert allocator_delta == 115
    assert free_delta == 125

    assert canonical_rows._row_resource_conformance_v2(
        ledger=ledger,
        allocator_baseline_bytes=10,
        allocator_peak_bytes=131,
        free_vram_before_bytes=1_000,
        free_vram_after_bytes=875,
    )[0] is False
    assert canonical_rows._row_resource_conformance_v2(
        ledger=ledger,
        allocator_baseline_bytes=10,
        allocator_peak_bytes=125,
        free_vram_before_bytes=1_000,
        free_vram_after_bytes=869,
    )[0] is False

    # Regression: the physical W64/P1024/C2048 B96/K64 owner used native
    # allocator blocks 1,720,448 bytes above logical tensor bytes. Its B=1
    # reservation observation was only 1,590,524 bytes, so a B=1-only value
    # must be raised to the explicit allocator segment bound before admission.
    from engine.perf.cuda_full_evolution_v2.workhorse.layout import (
        workhorse_allocator_owner_overhead_bound_v2,
    )

    physical_tensor_bytes = 1_256_595_840
    physical_allocator_peak_bytes = 1_258_316_288
    bounded_overhead = workhorse_allocator_owner_overhead_bound_v2(1_590_524)
    assert bounded_overhead == 10 * 1024 * 1024
    assert canonical_rows._row_resource_conformance_v2(
        ledger={
            "tensor_resident_bytes": physical_tensor_bytes,
            "allocator_owner_overhead_bytes": bounded_overhead,
            "incremental_device_bytes": physical_tensor_bytes + bounded_overhead,
        },
        allocator_baseline_bytes=0,
        allocator_peak_bytes=physical_allocator_peak_bytes,
        free_vram_before_bytes=1_000_000_000,
        free_vram_after_bytes=1_000_000_000,
    )[0] is True


def test_row_event_sink_receipt_reopen_rejects_hostile_digest_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acknowledged = {
        "schema_version": (
            canonical_rows.WORKHORSE_SPARSE_EVENT_SINK_RECEIPT_SCHEMA_V3
        ),
        "receipt_path": "/host/event-sink/sample-0/sink_receipt.json",
        "receipt_file_sha256": "a" * 64,
        "receipt_body_sha256": "b" * 64,
        "dispatch_content_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_event_sink_receipt_v2",
        lambda path: deepcopy(acknowledged),
    )
    assert canonical_rows._reopen_row_event_sink_receipts_v2(
        [acknowledged]
    ) == [acknowledged]
    hostile = {**acknowledged, "receipt_file_sha256": "d" * 64}
    with pytest.raises(
        WorkhorseCanonicalRowError,
        match="changed after acknowledgement",
    ):
        canonical_rows._reopen_row_event_sink_receipts_v2([hostile])


def test_row_event_sink_receipt_reopen_accepts_compact_consumer_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acknowledged = {
        "schema_version": (
            canonical_rows.WORKHORSE_SPARSE_EVENT_CONSUMER_RECEIPT_SCHEMA_V4
        ),
        "receipt_path": "/host/event-sink/sample-0/sink_receipt.json",
        "receipt_file_sha256": "a" * 64,
        "receipt_body_sha256": "b" * 64,
        "dispatch_content_sha256": "c" * 64,
        "raw_payload_retained": False,
    }
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_event_sink_receipt_v2",
        lambda path: deepcopy(acknowledged),
    )
    assert canonical_rows._reopen_row_event_sink_receipts_v2(
        [acknowledged]
    ) == [acknowledged]


def test_row_event_sink_receipt_reopen_normalizes_legacy_v3_retention_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acknowledged = {
        "schema_version": (
            canonical_rows.WORKHORSE_SPARSE_EVENT_SINK_RECEIPT_SCHEMA_V3
        ),
        "receipt_path": "/host/event-sink/sample-0/sink_receipt.json",
        "receipt_file_sha256": "a" * 64,
        "receipt_body_sha256": "b" * 64,
        "dispatch_content_sha256": "c" * 64,
    }
    current = {**acknowledged, "raw_payload_retained": True}
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_event_sink_receipt_v2",
        lambda path: deepcopy(current),
    )
    assert canonical_rows._reopen_row_event_sink_receipts_v2(
        [acknowledged]
    ) == [current]

    hostile = {**acknowledged, "raw_payload_retained": False}
    with pytest.raises(
        WorkhorseCanonicalRowError,
        match="changed after acknowledgement",
    ):
        canonical_rows._reopen_row_event_sink_receipts_v2([hostile])


def test_provider_reopens_live_scheduler_and_rejects_hostile_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _bound_attempt(tmp_path, monkeypatch)
    target = (
        fixture["run_root"]
        / "batches"
        / fixture["batch_id"]
        / "workhorse-parent-context.json"
    )
    parent = fixture["adapter"].bind_workhorse_parent_context_v2(
        attempt_authority_path=fixture["attempt_authority_path"],
        target=target,
        campaign_plan=fixture["plan"],
        bound_execution_request=fixture["bound_request"],
    )
    validated = validate_workhorse_canonical_parent_authority_binding_v2(
        dict(parent), expected_operation="EXECUTION_ROW"
    )
    assert validated["execution_authority_granted"] is False
    assert validated["compiled_temporal_k"] == fixture["contract"].temporal_k
    assert validated["science_continuation_contract_sha256"] == "7" * 64

    other_row = deepcopy(fixture["bound_request"])
    other_row["resident_request"]["parameter_arrays"]["workhorse.B"] = [3]
    with pytest.raises(
        CudaFullEvolutionV2LaunchpadAdapterError,
        match="plan or request",
    ):
        fixture["adapter"].bind_workhorse_parent_context_v2(
            attempt_authority_path=fixture["attempt_authority_path"],
            target=target.parent / "other-parent.json",
            campaign_plan=fixture["plan"],
            bound_execution_request=other_row,
        )


def test_indexed_continuation_parent_binds_admitted_k_set_and_logical_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _bound_attempt(
        tmp_path,
        monkeypatch,
        indexed_continuation=True,
    )
    target = (
        fixture["run_root"]
        / "batches"
        / fixture["batch_id"]
        / "workhorse-parent-context.json"
    )
    parent = fixture["adapter"].bind_workhorse_parent_context_v2(
        attempt_authority_path=fixture["attempt_authority_path"],
        target=target,
        campaign_plan=fixture["plan"],
        bound_execution_request=fixture["bound_request"],
    )
    validated = validate_workhorse_canonical_parent_authority_binding_v2(
        dict(parent),
        expected_operation=INDEXED_ACS_CONTINUATION_OPERATION_V2,
    )
    event_attempt = workhorse_authority._event_attempt_view_v2(
        dict(parent), historical=False
    )
    assert event_attempt["binding"]["operation"] == (
        INDEXED_ACS_CONTINUATION_OPERATION_V2
    )
    assert event_attempt["start_next_tick"] == 25
    assert event_attempt["target_next_tick"] == 30
    assert validated["compiled_temporal_k"] == [1, 2]
    assert validated["source_checkpoint_identity_sha256"] == fixture[
        "continuation"
    ]["science_continuation_identity"]["source_world_state_set_sha256"]
    assert validated["absolute_tick_span"] == {
        "time_authority": "absolute_tick_v1",
        "start_next_tick": 25,
        "target_next_tick": 30,
        "total_ticks": 5,
    }
    loaded = SimpleNamespace(
        artifact=Path(fixture["artifact_binding"]["path"]),
    )
    owner = SimpleNamespace(
        loaded=loaded,
        state=torch.zeros((fixture["contract"].B, 8), dtype=torch.uint8),
        pending=None,
        poisoned_status=(),
        attempt_authority_binding=dict(parent),
        committed_next_tick=25,
        last_event_canonicalization_tail_ms=None,
    )
    workhorse_probe._prepare_dispatch_v2(loaded, owner, 1, 0)
    owner.pending = None
    workhorse_probe._prepare_dispatch_v2(loaded, owner, 2, 0)
    owner.pending = None
    with pytest.raises(
        workhorse_probe.FullEvolutionWorkhorseProbeError,
        match="differs from canonical parent",
    ):
        workhorse_probe._prepare_dispatch_v2(loaded, owner, 3, 0)

    pointer = json.loads(
        fixture["attempt_authority_path"].read_text(encoding="ascii")
    )
    pointer["attempt_id"] = "fabricated-attempt"
    pointer_core = {
        key: value for key, value in pointer.items() if key != "pointer_sha256"
    }
    pointer["pointer_sha256"] = crca_sha256(pointer_core).hex
    fabricated = tmp_path / "fabricated-pointer.json"
    _write_file(
        fabricated,
        json.dumps(pointer, sort_keys=True, separators=(",", ":")).encode("ascii"),
    )
    with pytest.raises(
        CudaFullEvolutionV2LaunchpadAdapterError,
        match="Scheduler bundle differs",
    ):
        fixture["adapter"].bind_workhorse_parent_context_v2(
            attempt_authority_path=fabricated,
            target=target.parent / "fabricated-parent.json",
            campaign_plan=fixture["plan"],
            bound_execution_request=fixture["bound_request"],
        )


def test_indexed_continuation_is_an_admitted_physical_acs2_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _bound_attempt(
        tmp_path,
        monkeypatch,
        indexed_continuation=True,
    )
    provider = fixture["provider"]
    request = provider.translate_batch(fixture["plan"], fixture["batch_id"])
    assert request.acs_checkpoint_slots == 2
    assert provider.readiness_reasons(fixture["plan"], request) == ()


@pytest.mark.parametrize("selected_world_observation", [False, True])
def test_indexed_continuation_provider_drains_and_commits_two_slot_fifo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_world_observation: bool,
) -> None:
    fixture = _bound_attempt(
        tmp_path,
        monkeypatch,
        indexed_continuation=True,
        selected_world_observation=selected_world_observation,
    )
    provider = fixture["provider"]
    request = provider.translate_batch(fixture["plan"], fixture["batch_id"])
    recorded: list[tuple[str, str]] = []
    context = SimpleNamespace(
        run_root=fixture["run_root"],
        batch_root=(
            fixture["run_root"] / "batches" / fixture["batch_id"]
        ),
        record_artifact=lambda path, *, artifact_family, retention_class: recorded.append(
            (artifact_family, retention_class)
        ),
    )
    Path(context.batch_root).mkdir(parents=True, exist_ok=True)

    class FakeCoordinator:
        def __init__(self, *, slots: int, **_kwargs: object) -> None:
            assert slots == 2
            self.pending: list[SimpleNamespace] = []
            self.sequence = 0
            self.closed = False

        def pending_tokens(self) -> tuple[SimpleNamespace, ...]:
            return tuple(self.pending)

        def capture_workhorse_checkpoint(
            self,
            _state: object,
            *,
            contract: object,
            checkpoint_at_tick: int,
            event_sink_receipt_path: object,
        ) -> SimpleNamespace:
            assert contract == fixture["contract"]
            assert event_sink_receipt_path
            token = SimpleNamespace(
                sequence=self.sequence,
                compute_frontier=checkpoint_at_tick,
            )
            self.sequence += 1
            self.pending.append(token)
            return token

        def acknowledge_checkpoint(self, token: SimpleNamespace) -> None:
            assert self.pending[0] is token
            self.pending.pop(0)

        def close(self) -> None:
            assert not self.pending
            self.closed = True

        def abort(self) -> None:
            self.pending.clear()

        def telemetry(self) -> dict[str, object]:
            return {
                "closed": self.closed,
                "writer_alive": False,
                "failure_latched": False,
                "checkpoint_receipts": [
                    {"checkpoint_at_tick": tick} for tick in (28, 30)
                ],
            }

    monkeypatch.setattr(
        acs_adapter_module,
        "CudaFullEvolutionV2WorkhorseACSCoordinatorV1",
        FakeCoordinator,
    )
    commit_ticks: list[int] = []
    occurrence_parents: list[object] = []

    def commit_checkpoint(**kwargs: object) -> dict[str, object]:
        tick = kwargs["expected_checkpoint_tick"]
        assert kwargs["expected_operation"] == INDEXED_ACS_CONTINUATION_OPERATION_V2
        coordinator = kwargs["coordinator"]
        token = kwargs["token"]
        assert token.compute_frontier == tick
        coordinator.acknowledge_checkpoint(token)
        commit_ticks.append(tick)
        occurrence_parents.append(kwargs["parent_checkpoint_digest"])
        return {
            "checkpoint_at_tick": tick,
            "checkpoint_occurrence_projection": {
                "projection_content_sha256": f"{tick + 2000:064x}",
                "occurrence_digest": {
                    "algorithm": "sha256",
                    "hex": f"{tick + 1000:064x}",
                }
            },
            "indexed_checkpoint": {
                "manifest_path": str(tmp_path / f"indexed-{tick}.json"),
                "world_state_set_identity": {
                    "checkpoint_at_tick": tick,
                    "world_state_set_sha256": f"{tick:064x}",
                }
            },
        }

    monkeypatch.setattr(
        provider_module,
        "_commit_workhorse_row_acs_checkpoint_v2",
        commit_checkpoint,
    )

    def run_continuation(*_args: object, **kwargs: object) -> dict[str, object]:
        handler = kwargs["checkpoint_boundary_handler"]
        for tick in (28, 30):
            handler(
                state=object(),
                contract=fixture["contract"],
                checkpoint_at_tick=tick,
                event_sink_receipt_path=tmp_path / f"event-{tick}.json",
            )
        result_path = Path(kwargs["execution_result_path"])
        _write_file(result_path, b"{}")
        return {
            "outcome": "CONTINUED",
            "checkpoint_capture_ticks": [28, 30],
            "receipt_content_sha256": "e" * 64,
            "receipt_file_sha256": _file_sha256(result_path),
            "aggregate_world_ticks_per_second": 1234.5,
            "absolute_frontiers": {
                "start_next_tick": 25,
                "target_next_tick": 30,
                "committed_next_tick": 30,
            },
        }

    monkeypatch.setattr(
        provider_module,
        "run_workhorse_indexed_continuation_v2",
        run_continuation,
    )
    if selected_world_observation:
        def seal_observation(path: str | Path, **_kwargs: object) -> dict[str, object]:
            output = Path(path)
            _write_file(output, b'{"observation":"selected-world"}\n')
            return {
                "observation_mode": (
                    provider_module.WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1
                ),
                "observation_path": str(output.resolve()),
                "observation_file_sha256": _file_sha256(output),
                "observation_content_sha256": "f" * 64,
                "observation_file_bytes": output.stat().st_size,
            }

        monkeypatch.setattr(
            provider_module,
            "seal_workhorse_selected_world_observation_v1",
            seal_observation,
        )

    executed = provider._execute_batch_with_bound_acs_runtime(
        fixture["plan"], request, context
    )

    assert commit_ticks == [28, 30]
    assert occurrence_parents == [
        (
            {"algorithm": "sha256", "hex": "1" * 64}
            if selected_world_observation
            else None
        ),
        {"algorithm": "sha256", "hex": f"{1028:064x}"},
    ]
    assert executed["status"] == "completed"
    assert executed["checkpoint"]["checkpoint_ticks"] == [28, 30]
    assert executed["result"]["aggregate_world_ticks_per_second"] == 1234.5
    assert executed["result"]["acs_checkpoint_staging"]["closed"] is True
    assert ("workhorse_indexed_continuation_result_v2", "R3") in recorded
    if selected_world_observation:
        assert executed["checkpoint"]["observation_mode"] == "OFF"
        assert executed["checkpoint"]["science_continuation_claimed"] is False
        pointer = executed["result"]["host_boundary_observation"]
        assert pointer["observation_mode"] == (
            provider_module.WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1
        )
        assert (
            "science.workhorse_selected_world_host_observation_v1",
            "R4",
        ) in recorded
    else:
        assert "host_boundary_observation" not in executed["result"]


def test_terminal_and_expired_attempts_cannot_reenter_one_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expired = _bound_attempt(tmp_path / "expired", monkeypatch)
    expired["adapter"].now_fn = lambda: "2026-08-27T05:00:00Z"
    with pytest.raises(Exception, match="expired"):
        expired["adapter"].bind_workhorse_parent_context_v2(
            attempt_authority_path=expired["attempt_authority_path"],
            target=expired["run_root"]
            / "batches"
            / expired["batch_id"]
            / "workhorse-parent-context.json",
            campaign_plan=expired["plan"],
            bound_execution_request=expired["bound_request"],
        )

    terminal = _bound_attempt(tmp_path / "terminal", monkeypatch)
    terminal["adapter"].terminalize_attempt_authority_v1(
        attempt_authority_path=terminal["attempt_authority_path"],
        campaign_plan=terminal["plan"],
        execution_status="failed",
        execution_receipt={
            "crca_final_frontiers": {
                "last_dispatched_next_tick": 0,
                "last_computed_next_tick": 0,
                "durable_checkpoint_next_tick": 0,
                "published_artifact_next_tick": 0,
            }
        },
        result_binding={"execution_status": "failed"},
    )
    with pytest.raises(Exception, match="terminal"):
        terminal["adapter"].bind_workhorse_parent_context_v2(
            attempt_authority_path=terminal["attempt_authority_path"],
            target=terminal["run_root"]
            / "batches"
            / terminal["batch_id"]
            / "workhorse-parent-context.json",
            campaign_plan=terminal["plan"],
            bound_execution_request=terminal["bound_request"],
        )


def test_runtime_headroom_rejection_terminalizes_real_scheduler_at_start_and_replay_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _bound_attempt(tmp_path, monkeypatch)
    request = fixture["provider"].translate_batch(
        fixture["plan"], fixture["batch_id"]
    )
    recorded: list[Path] = []
    context = SimpleNamespace(
        run_root=fixture["run_root"],
        batch_root=(
            fixture["run_root"] / "batches" / fixture["batch_id"]
        ),
        record_artifact=lambda path, **_: recorded.append(Path(path)),
    )
    physical_calls = 0

    def reject_runtime_headroom(
        _artifact_path: str | Path, **kwargs: object
    ) -> dict[str, Any]:
        nonlocal physical_calls
        physical_calls += 1
        parent = validate_workhorse_canonical_parent_authority_binding_v2(
            dict(kwargs["canonical_parent_authority_binding"]),
            expected_operation="EXECUTION_ROW",
        )
        assert parent["attempt_id"]
        output = Path(str(kwargs["execution_result_path"]))
        _write_file(output, b'{"outcome":"RUNTIME_HEADROOM_REJECTED"}')
        start_tick = fixture["contract"].start_next_tick
        target_tick = start_tick + fixture["contract"].total_ticks
        return {
            "outcome": "RUNTIME_HEADROOM_REJECTED",
            "receipt_file_sha256": _file_sha256(output),
            "receipt_content_sha256": "a" * 64,
            "absolute_frontiers": {
                "start_next_tick": start_tick,
                "target_next_tick": target_tick,
                "committed_next_tick": start_tick,
            },
        }

    monkeypatch.setattr(
        provider_module,
        "run_workhorse_canonical_admitted_row_v2",
        reject_runtime_headroom,
    )
    with pytest.raises(
        CampaignProviderError,
        match=RUNTIME_HEADROOM_REJECTION_CODE_V2,
    ):
        fixture["provider"].execute_batch(fixture["plan"], request, context)
    result_path = (
        context.batch_root / "workhorse-row-execution-result.json"
    )
    assert result_path in recorded
    assert physical_calls == 1

    start_tick = fixture["contract"].start_next_tick
    final_frontiers = {
        "last_dispatched_next_tick": start_tick,
        "last_computed_next_tick": start_tick,
        "durable_checkpoint_next_tick": start_tick,
        "published_artifact_next_tick": start_tick,
    }
    terminal = fixture["adapter"].terminalize_attempt_authority_v1(
        attempt_authority_path=fixture["attempt_authority_path"],
        campaign_plan=fixture["plan"],
        execution_status="failed",
        execution_receipt={"crca_final_frontiers": final_frontiers},
        result_binding={
            "execution_status": "failed",
            "result_class": RUNTIME_HEADROOM_REJECTION_CODE_V2,
        },
    )
    binding = terminal["workhorse_terminal_binding"]
    assert binding["attempt_terminal_status"] == "FAILED"
    assert {
        key: value
        for key, value in binding["final_frontiers"].items()
        if key != "schema_version"
    } == final_frontiers
    terminal_bundle = json.loads(
        Path(terminal["terminal_bundle_path"]).read_text(encoding="ascii")
    )
    assert terminal_bundle["attempt_terminal"]["status"] == "FAILED"
    assert {
        key: value
        for key, value in terminal_bundle["attempt_terminal"][
            "final_frontiers"
        ].items()
        if key != "schema_version"
    } == final_frontiers

    fixture["adapter"].now_fn = lambda: T3
    authority = fixture["adapter"]._authority(
        fixture["attempt_authority_path"]
    )
    claim = authority.raw["claim"]
    with pytest.raises(Exception, match="follows terminalization"):
        fixture["adapter"].issuer.heartbeat_or_renew(
            attempt_id=str(authority.attempt_start.attempt_id),
            process_owner_id=adapter_module.PROCESS_OWNER_ID_V1,
            process_owner_digest=adapter_module._scheduler_digest(
                claim["execution_owner_digest"]
            ),
            at_utc=T3,
        )
    with pytest.raises(Exception, match="terminal|live|lease|target already exists"):
        fixture["provider"].execute_batch(fixture["plan"], request, context)
    assert physical_calls == 1


def test_runtime_headroom_front_door_replay_reopens_failed_real_scheduler_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _operation_fixture(tmp_path, monkeypatch)
    request = _launchpad_request_for_fixture(tmp_path, fixture)
    times = iter((T0, T1, T2, T3))

    def resource_evidence(**values: Any) -> Mapping[str, Any]:
        route = values["route"]
        return {
            "resource_observation_id": "ROBS-WORKHORSE-FRONT-DOOR",
            "resource_observation": {
                "schema_version": "eiel_test_resource_observation_v1",
                "device_id": route["device_id"],
            },
            "resource_observed_at_utc": values["at_utc"],
            "resource_recommendation_id": "RADM-WORKHORSE-FRONT-DOOR",
            "resource_recommendation": {
                "schema_version": "eiel_test_resource_recommendation_v1",
                "decision": "ADMIT",
            },
            "live_provider_check": {
                "schema_version": "crca_test_live_provider_check_v1",
                "available": True,
                "device_id": route["device_id"],
                "event_transport": route["event_transport"],
            },
        }

    adapter = CanonicalCudaFullEvolutionV2LaunchpadAdapterV1(
        repo_root=REPO_ROOT,
        state_root=tmp_path / "scheduler-state",
        allow_test_state_root=True,
        resource_evidence_factory=resource_evidence,
        now_fn=lambda: next(times),
    )
    registry = cuda_full_evolution_workhorse_v2_provider_registry_v1(
        operation_adapter=adapter
    )
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _spec, **_kwargs: SimpleNamespace(
            payload=lambda: deepcopy(fixture["plan"])
        ),
    )
    monkeypatch.setattr(
        front_door,
        "_git_identity",
        lambda repo_root: {
            "repository_root": str(repo_root),
            "commit": "a" * 40,
            "working_tree_clean": True,
            "working_tree_status_sha256": campaign_sha256([]),
            "working_tree_status_lines": [],
        },
    )

    physical_calls = 0
    recorded: list[Path] = []

    def reject_runtime_headroom(
        _artifact_path: str | Path, **kwargs: object
    ) -> dict[str, Any]:
        nonlocal physical_calls
        physical_calls += 1
        validate_workhorse_canonical_parent_authority_binding_v2(
            dict(kwargs["canonical_parent_authority_binding"]),
            expected_operation="EXECUTION_ROW",
        )
        output = Path(str(kwargs["execution_result_path"]))
        _write_file(output, b'{"outcome":"RUNTIME_HEADROOM_REJECTED"}')
        start_tick = fixture["contract"].start_next_tick
        target_tick = start_tick + fixture["contract"].total_ticks
        return {
            "outcome": "RUNTIME_HEADROOM_REJECTED",
            "receipt_file_sha256": _file_sha256(output),
            "receipt_content_sha256": "b" * 64,
            "absolute_frontiers": {
                "start_next_tick": start_tick,
                "target_next_tick": target_tick,
                "committed_next_tick": start_tick,
            },
        }

    monkeypatch.setattr(
        provider_module,
        "run_workhorse_canonical_admitted_row_v2",
        reject_runtime_headroom,
    )

    def preflight(plan: Mapping[str, Any], **_kwargs: object) -> dict[str, Any]:
        return {
            "schema_version": "eiel_campaign_preflight_v1",
            "campaign_id": plan["campaign_id"],
            "plan_sha256": plan["plan_sha256"],
            "ready_for_execution": True,
            "reasons": [],
            "execution_authority_reasons": [],
            "artifacts": [],
            "runtime_environment": {},
        }

    def degraded_canopy(
        *, plan: Mapping[str, Any], run_root: Path, **_kwargs: object
    ) -> dict[str, Any]:
        return front_door.record_canopy_observability_degraded_v1(
            run_root,
            plan=plan,
            reason="host-only no-relay fixture",
        )

    def execute_provider_once(
        plan: Mapping[str, Any], *, run_root: Path, **_kwargs: object
    ) -> dict[str, Any]:
        provider = registry.get(str(plan["resolved_profile"]))
        assert provider is not None
        batch_id = str(plan["physical_batches"][0]["batch_id"])
        batch_root = run_root / "batches" / batch_id
        batch_root.mkdir(parents=True, exist_ok=True)
        context = SimpleNamespace(
            run_root=run_root,
            batch_root=batch_root,
            record_artifact=lambda path, **_: recorded.append(Path(path)),
        )
        resident_request = provider.translate_batch(plan, batch_id)
        return provider.execute_batch(plan, resident_request, context)

    call_kwargs = {
        "repo_root": tmp_path,
        "provider_registry": registry,
        "preflight_fn": preflight,
        "execute_fn": execute_provider_once,
        "mechanical_verifier_fn": lambda **_: {
            "schema_version": "eiel_execution_mechanical_verification_v1",
            "status": "PASS",
            "checks": [],
            "errors": [],
            "semantic_questions": [],
        },
        "canopy_start_fn": degraded_canopy,
        "cuda_full_evolution_v2_crca_adapter": adapter,
    }
    first = front_door.execute_request_v2(request, **call_kwargs)
    assert first["result_class"] == RUNTIME_HEADROOM_REJECTION_CODE_V2
    assert first["retry_contract"] == "FRESH_EXECUTION_REQUEST_V2_REQUIRED"
    assert first["crca_attempt_terminal"]["execution_status"] == "failed"
    terminal_binding = first["crca_attempt_terminal"][
        "workhorse_terminal_binding"
    ]
    assert terminal_binding["attempt_terminal_status"] == "FAILED"
    assert {
        value
        for key, value in terminal_binding["final_frontiers"].items()
        if key != "schema_version"
    } == {fixture["contract"].start_next_tick}
    assert any(path.name == "workhorse-row-execution-result.json" for path in recorded)

    authority_path = fixture["run_root"] / "control/crca/attempt-authority.json"
    authority = adapter._authority(authority_path)
    claim = authority.raw["claim"]
    adapter.now_fn = lambda: T3
    with pytest.raises(Exception, match="follows terminalization"):
        adapter.issuer.heartbeat_or_renew(
            attempt_id=str(authority.attempt_start.attempt_id),
            process_owner_id=adapter_module.PROCESS_OWNER_ID_V1,
            process_owner_digest=adapter_module._scheduler_digest(
                claim["execution_owner_digest"]
            ),
            at_utc=T3,
        )

    replay = front_door.execute_request_v2(request, **call_kwargs)
    assert replay == first
    assert physical_calls == 1


def test_real_scheduler_terminal_is_reopened_before_row_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _bound_attempt(tmp_path, monkeypatch)
    batch_root = (
        fixture["run_root"] / "batches" / fixture["batch_id"]
    )
    parent_path = batch_root / "workhorse-parent-context.json"
    fixture["adapter"].bind_workhorse_parent_context_v2(
        attempt_authority_path=fixture["attempt_authority_path"],
        target=parent_path,
        campaign_plan=fixture["plan"],
        bound_execution_request=fixture["bound_request"],
    )
    execution_result_path = batch_root / "workhorse-row-execution-result.json"
    _write_file(execution_result_path, b'{"host":"execution-result"}')
    target_tick = (
        fixture["contract"].start_next_tick
        + fixture["contract"].total_ticks
    )
    manifest = {
        "status": "completed",
        "checkpoint": {"next_absolute_tick": target_tick},
        "result": {
            "schema_version": "eiel_workhorse_single_row_provider_result_v2",
            "operation": "EXECUTION_ROW",
            "execution_result_path": str(execution_result_path.resolve()),
            "absolute_frontiers": {
                "start_next_tick": fixture["contract"].start_next_tick,
                "target_next_tick": target_tick,
                "committed_next_tick": target_tick,
            },
        },
    }
    manifest_path = batch_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")
    observed: list[dict[str, Any]] = []

    def seal_row(
        output_path: str | Path,
        *,
        execution_result_path: str | Path,
        canonical_terminal_binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        terminal_path = Path(
            str(canonical_terminal_binding["terminal_bundle_path"])
        )
        terminal_bundle = json.loads(terminal_path.read_text(encoding="ascii"))
        assert terminal_bundle["attempt_terminal"]["status"] == "COMPLETED"
        assert canonical_terminal_binding["final_frontiers"] == (
            terminal_bundle["attempt_terminal"]["final_frontiers"]
        )
        assert all(
            canonical_terminal_binding["final_frontiers"][field] == target_tick
            for field in (
                "last_dispatched_next_tick",
                "last_computed_next_tick",
                "durable_checkpoint_next_tick",
                "published_artifact_next_tick",
            )
        )
        output = Path(output_path)
        _write_file(output, b'{"host":"terminalized-row"}')
        observed.append(dict(canonical_terminal_binding))
        return {
            "receipt_path": str(output.resolve()),
            "receipt_file_sha256": _file_sha256(output),
            "receipt_content_sha256": "9" * 64,
        }

    monkeypatch.setattr(
        provider_module, "seal_workhorse_technical_row_receipt_v2", seal_row
    )
    terminal = fixture["adapter"].terminalize_attempt_authority_v1(
        attempt_authority_path=fixture["attempt_authority_path"],
        campaign_plan=fixture["plan"],
        execution_status="completed",
        execution_receipt={
            "status": "completed",
            "plan_sha256": fixture["plan"]["plan_sha256"],
        },
        result_binding={"execution_status": "completed"},
    )
    finalized = fixture["provider"].finalize_attempt_terminal_v2(
        plan=fixture["plan"],
        run_root=fixture["run_root"],
        canonical_terminal_binding=terminal["workhorse_terminal_binding"],
    )
    assert len(observed) == 1
    assert finalized["attempt_id"] == observed[0]["attempt_id"]
    assert Path(finalized["row_receipt_path"]).is_file()


def test_row_receipt_rejects_failed_or_partial_scheduler_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _bound_attempt(tmp_path, monkeypatch)
    parent_path = (
        fixture["run_root"]
        / "batches"
        / fixture["batch_id"]
        / "workhorse-parent-context.json"
    )
    parent = fixture["adapter"].bind_workhorse_parent_context_v2(
        attempt_authority_path=fixture["attempt_authority_path"],
        target=parent_path,
        campaign_plan=fixture["plan"],
        bound_execution_request=fixture["bound_request"],
    )
    target_tick = fixture["contract"].total_ticks

    def execution_result(_path: str | Path) -> Mapping[str, Any]:
        return {
            "canonical_parent_authority_binding": parent,
            "outcome": "MEASURED",
            "absolute_frontiers": {"target_next_tick": target_tick},
        }

    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_canonical_row_execution_result_v2",
        execution_result,
    )

    def terminal_binding(status: str, durable: int) -> dict[str, Any]:
        frontiers = {
            "schema_version": "crca_attempt_frontiers_v1",
            "last_dispatched_next_tick": target_tick,
            "last_computed_next_tick": target_tick,
            "durable_checkpoint_next_tick": durable,
            "published_artifact_next_tick": target_tick,
        }
        bundle = {
            "schema_version": "crca_launchpad_attempt_terminal_bundle_v1",
            "attempt_id": parent["attempt_id"],
            "authoritative_bundle_digest": {
                "algorithm": "sha256",
                "hex": parent["scheduler_bundle_digest"],
            },
            "attempt_terminal": {
                "schema_version": "crca_attempt_terminal_v1",
                "attempt_id": parent["attempt_id"],
                "status": status,
                "final_frontiers": frontiers,
            },
            "continuation_event": {},
            "terminalized_at_utc": T3,
        }
        path = tmp_path / f"terminal-{status.lower()}-{durable}.json"
        _write_file(
            path,
            json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            ),
        )
        core = {
            "schema_version": (
                "eiel_workhorse_canonical_parent_terminal_binding_v2"
            ),
            "authority_source": "CANONICAL_PROVIDER_AFTER_SCHEDULER_REOPEN",
            "status": "TERMINAL",
            "attempt_terminal_status": status,
            "run_id": parent["run_id"],
            "attempt_id": parent["attempt_id"],
            "terminal_bundle_path": str(path.resolve()),
            "terminal_bundle_file_sha256": _file_sha256(path),
            "terminal_bundle_digest": canonical_sha256(bundle),
            "final_frontiers": frontiers,
        }
        return {**core, "binding_sha256": canonical_sha256(core)}

    with pytest.raises(WorkhorseCanonicalRowError, match="completed exact-target"):
        seal_workhorse_technical_row_receipt_v2(
            tmp_path / "failed-row.json",
            execution_result_path=tmp_path / "unused.json",
            canonical_terminal_binding=terminal_binding("FAILED", 0),
        )
    with pytest.raises(WorkhorseCanonicalRowError, match="completed exact-target"):
        seal_workhorse_technical_row_receipt_v2(
            tmp_path / "partial-row.json",
            execution_result_path=tmp_path / "unused.json",
            canonical_terminal_binding=terminal_binding("COMPLETED", 0),
        )


@pytest.mark.parametrize(
    "case_id",
    (
        "parity_reference_k1_sequence",
        "checkpoint_same_restore_sequence",
        "fresh_restore_continuation",
        "invalid_restore_failstop",
        "transaction_failstop",
        "event_reuse_canonicalization_regression",
    ),
)
def test_semantic_cases_compile_as_one_envelope_admitted_operation_without_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
) -> None:
    fixture = _semantic_fixture(tmp_path, monkeypatch, case_id=case_id)

    class PreflightAdapter:
        @staticmethod
        def validate_workhorse_external_checkpoint_source_v2(
            binding: Mapping[str, Any], *, expected_identity_sha256: str
        ) -> Mapping[str, Any]:
            assert binding["base_checkpoint_identity_sha256"] == (
                expected_identity_sha256
            )
            return dict(binding)

    provider = cuda_full_evolution_workhorse_v2_provider_registry_v1(
        operation_adapter=PreflightAdapter()
    ).get(fixture["plan"]["resolved_profile"])
    assert provider is not None
    request = provider.translate_batch(fixture["plan"], fixture["batch_id"])
    assert len(request.cell_ids) == 1
    assert request.start_next_tick == fixture["law"].start_next_absolute_tick
    assert request.total_ticks == (
        fixture["law"].start_next_absolute_tick + fixture["law"].total_ticks
    )
    assert request.tick_native_control["attempt_terminal_contract"] == fixture[
        "operation"
    ]["semantic_case_contract"]["attempt_terminal_contract"]
    assert provider.readiness_reasons(fixture["plan"], request) == ()
    operation = fixture["plan"]["analysis"]["extension"]["contract"]
    assert "semantic_gate_receipt" not in operation
    assert operation["envelope"] == fixture["operation"]["envelope"]
    route_operation = next(
        row
        for row in provider.execution_route_contract(fixture["plan"]).components
        if row["component_id"]
        == "cuda_full_evolution_workhorse_v2_operation"
    )
    assert route_operation["semantic_case_id"] == case_id
    assert route_operation["one_operation_per_attempt"] is True
    hostile = deepcopy(fixture["plan"])
    extension = hostile["analysis"]["extension"]
    terminal = extension["contract"]["semantic_case_contract"][
        "attempt_terminal_contract"
    ]
    terminal["attempt_terminal_status"] = (
        "FAILED"
        if terminal["attempt_terminal_status"] == "COMPLETED"
        else "COMPLETED"
    )
    extension["contract_sha256"] = canonical_sha256(extension["contract"])
    with pytest.raises(CampaignProviderError, match="closed case table"):
        provider.translate_batch(hostile, fixture["batch_id"])


@pytest.mark.parametrize(
    "case_id",
    (
        "parity_reference_k1_sequence",
        "checkpoint_same_restore_sequence",
        "fresh_restore_continuation",
        "invalid_restore_failstop",
        "transaction_failstop",
        "event_reuse_canonicalization_regression",
    ),
)
def test_semantic_provider_binds_parent_then_calls_one_case_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
) -> None:
    fixture = _semantic_fixture(tmp_path, monkeypatch, case_id=case_id)
    calls: list[str] = []
    parent = {"attempt_id": f"ATTEMPT-{fixture['law'].ordinal}"}
    produced_binding: dict[str, Any] | None = None

    class SemanticAdapter:
        @staticmethod
        def validate_workhorse_external_checkpoint_source_v2(
            binding: Mapping[str, Any], *, expected_identity_sha256: str
        ) -> Mapping[str, Any]:
            assert binding["base_checkpoint_identity_sha256"] == (
                expected_identity_sha256
            )
            return dict(binding)

        @staticmethod
        def bind_workhorse_parent_context_v2(
            *, target: Path, **_kwargs: Any
        ) -> Mapping[str, Any]:
            calls.append("bind")
            _write_file(target, b'{"parent":"scheduler-reopened"}')
            return parent

        @staticmethod
        def materialize_workhorse_semantic_checkpoint_occurrence_v2(
            *, occurrence_root: Path, **_kwargs: Any
        ) -> Mapping[str, Any]:
            nonlocal produced_binding
            calls.append("occurrence")
            occurrence = occurrence_root / "occurrence.json"
            manifest = occurrence_root / "base-manifest.json"
            _write_file(occurrence, b'{"occurrence":"canonical"}')
            _write_file(manifest, b'{"base":"canonical"}')
            body = {
                "schema_version": WORKHORSE_EXTERNAL_CHECKPOINT_SOURCE_SCHEMA_V2,
                "authorization_source": (
                    WORKHORSE_EXTERNAL_CHECKPOINT_AUTHORIZATION_SOURCE_V2
                ),
                "checkpoint_occurrence_record_path": str(occurrence.resolve()),
                "checkpoint_occurrence_record_file_sha256": _file_sha256(
                    occurrence
                ),
                "checkpoint_occurrence_identity_sha256": "c" * 64,
                "base_checkpoint_manifest_path": str(manifest.resolve()),
                "base_checkpoint_manifest_file_sha256": _file_sha256(manifest),
                "base_checkpoint_identity_sha256": "b" * 64,
            }
            produced_binding = {
                **body,
                "binding_sha256": canonical_sha256(body),
            }
            return {
                "external_source_checkpoint_binding": produced_binding,
                "artifact_paths": [
                    str(occurrence.resolve()),
                    str(manifest.resolve()),
                ],
            }

    provider = cuda_full_evolution_workhorse_v2_provider_registry_v1(
        operation_adapter=SemanticAdapter()
    ).get(fixture["plan"]["resolved_profile"])
    assert provider is not None
    request = provider.translate_batch(fixture["plan"], fixture["batch_id"])

    def live_device(binding: Mapping[str, Any]) -> Mapping[str, Any]:
        assert binding == parent
        calls.append("live")
        return fixture["device"]

    fragment_holder: dict[str, Any] = {}

    def run_case(_artifact: str | Path, **kwargs: Any) -> dict[str, Any]:
        calls.append("run")
        assert kwargs["canonical_parent_authority_binding"] == parent
        fragment_path = Path(kwargs["fragment_output_root"]) / (
            "semantic_case_fragment.json"
        )
        _write_file(fragment_path, b'{"raw":"semantic"}')
        fragment_holder.update(
            {
                "fragment_body_sha256": "a" * 64,
                "case_id": case_id,
                "semantic_temporal_k": 2,
                "shape_id": fixture["contract"].shape_id,
                "artifact_sha256": fixture["operation"]["artifact"]["sha256"],
                "canonical_parent_binding": parent,
                "external_source_checkpoint_binding": fixture["operation"][
                    "external_source_checkpoint_binding"
                ],
                "base_checkpoint": (
                    {
                        "identity": {
                            "base_checkpoint_identity_sha256": "b" * 64
                        }
                    }
                    if fixture["law"].produces_base_checkpoint
                    else None
                ),
            }
        )
        terminal = fixture["operation"]["semantic_case_contract"][
            "attempt_terminal_contract"
        ]
        failed = terminal["attempt_terminal_status"] == "FAILED"
        return {
            "schema_version": (
                "eiel_workhorse_semantic_case_provisional_result_v2"
            ),
            "claim_class": "TECHNICAL_ONLY_PROVISIONAL",
            "qualification_authority_granted": False,
            "science_authority_granted": False,
            "case_id": case_id,
            "expected_attempt_terminal_status": terminal[
                "attempt_terminal_status"
            ],
            "attempt_terminal_contract": terminal,
            "provisional_result_class": (
                "EXPECTED_NEGATIVE_PENDING_FAILED_TERMINAL"
                if failed
                else "SUCCESS_PENDING_COMPLETED_TERMINAL"
            ),
            "provider_terminalization_action": (
                "RETURN_FOR_FAILED_ATTEMPT_TERMINALIZATION"
                if failed
                else "RETURN_FOR_COMPLETED_TERMINALIZATION"
            ),
            "case_fragment_path": str(fragment_path.resolve()),
            "case_fragment_file_sha256": _file_sha256(fragment_path),
            "case_fragment_body_sha256": "a" * 64,
            "source_checkpoint_identity_sha256": fixture["operation"][
                "source_checkpoint_identity_sha256"
            ],
        }

    monkeypatch.setattr(
        provider_module,
        "validate_workhorse_live_device_under_semantic_parent_v2",
        live_device,
    )
    monkeypatch.setattr(provider_module, "run_workhorse_semantic_case_v2", run_case)
    monkeypatch.setattr(
        provider_module,
        "reopen_workhorse_semantic_case_fragment_v2",
        lambda _path: (deepcopy(fragment_holder), {}),
    )
    recorded: list[Path] = []
    batch_root = fixture["run_root"] / "batches" / fixture["batch_id"]
    result = provider.execute_batch(
        fixture["plan"],
        request,
        SimpleNamespace(
            run_root=fixture["run_root"],
            batch_root=batch_root,
            record_artifact=lambda path, **_: recorded.append(Path(path)),
        ),
    )
    assert calls[:3] == ["bind", "live", "run"]
    assert calls.count("run") == 1
    assert calls.count("bind") == 1
    assert result["status"] == "completed"
    assert result["result"]["attempt_terminal_contract"] == fixture["operation"][
        "semantic_case_contract"
    ]["attempt_terminal_contract"]
    if fixture["law"].produces_base_checkpoint:
        assert calls == ["bind", "live", "run", "occurrence"]
        assert (
            result["result"]["produced_external_source_checkpoint_binding"]
            == produced_binding
        )
    else:
        assert "occurrence" not in calls
        assert (
            result["result"]["produced_external_source_checkpoint_binding"]
            is None
        )


def test_semantic_valid_negative_exists_only_after_exact_terminal_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _semantic_fixture(
        tmp_path,
        monkeypatch,
        case_id="invalid_restore_failstop",
    )
    provider = cuda_full_evolution_workhorse_v2_provider_registry_v1().get(
        fixture["plan"]["resolved_profile"]
    )
    assert provider is not None
    operation = fixture["operation"]
    terminal = operation["semantic_case_contract"]["attempt_terminal_contract"]
    fragment_path = (
        fixture["run_root"]
        / "batches"
        / fixture["batch_id"]
        / "semantic_case_fragment.json"
    )
    _write_file(fragment_path, b'{"raw":"case22"}')
    parent = {"attempt_id": "ATTEMPT-CASE22"}
    fragment = {
        "fragment_body_sha256": "a" * 64,
        "case_id": fixture["law"].case_id,
        "semantic_temporal_k": 2,
        "shape_id": fixture["contract"].shape_id,
        "artifact_sha256": operation["artifact"]["sha256"],
        "canonical_parent_binding": parent,
        "external_source_checkpoint_binding": operation[
            "external_source_checkpoint_binding"
        ],
        "base_checkpoint": None,
    }
    frontiers = terminal["final_frontiers"]
    provider_result = {
        "schema_version": "eiel_workhorse_semantic_case_provider_result_v2",
        "operation": SEMANTIC_EVIDENCE_CASE_OPERATION_V2,
        "case_id": fixture["law"].case_id,
        "shape_id": fixture["contract"].shape_id,
        "backend_id": fixture["contract"].backend_id,
        "B": fixture["contract"].B,
        "temporal_k": fixture["contract"].temporal_k,
        "semantic_temporal_k": 2,
        "artifact_sha256": operation["artifact"]["sha256"],
        "sealed_artifact_identity_sha256": operation["artifact"][
            "sealed_artifact_identity_sha256"
        ],
        "expected_attempt_terminal_status": "FAILED",
        "attempt_terminal_contract": terminal,
        "provisional_result_class": (
            "EXPECTED_NEGATIVE_PENDING_FAILED_TERMINAL"
        ),
        "provider_terminalization_action": (
            "RETURN_FOR_FAILED_ATTEMPT_TERMINALIZATION"
        ),
        "case_fragment_path": str(fragment_path.resolve()),
        "case_fragment_file_sha256": _file_sha256(fragment_path),
        "case_fragment_body_sha256": "a" * 64,
        "source_checkpoint_identity_sha256": operation[
            "source_checkpoint_identity_sha256"
        ],
        "external_source_checkpoint_binding": operation[
            "external_source_checkpoint_binding"
        ],
        "science_checkpoint_compatibility_binding": operation[
            "science_checkpoint_compatibility_binding"
        ],
        "produced_external_source_checkpoint_binding": None,
        "checkpoint_occurrence_artifact_paths": [],
        "absolute_frontiers": {
            "start_next_tick": fixture["contract"].start_next_tick,
            "target_next_tick": fixture["contract"].start_next_tick
            + fixture["contract"].total_ticks,
            "committed_next_tick": frontiers[
                "published_artifact_next_tick"
            ],
        },
    }
    manifest_path = fragment_path.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps({"result": provider_result}), encoding="ascii"
    )
    monkeypatch.setattr(
        provider_module,
        "reopen_workhorse_semantic_case_fragment_v2",
        lambda _path: (deepcopy(fragment), {}),
    )
    sealed: dict[str, Any] = {}
    seal_calls = 0

    def seal_terminal(
        output_path: str | Path, **_kwargs: Any
    ) -> dict[str, Any]:
        nonlocal seal_calls
        seal_calls += 1
        output = Path(output_path)
        _write_file(output, b'{"terminal":"case22"}')
        sealed.update(
            {
                "receipt_path": str(output.resolve()),
                "receipt_file_sha256": _file_sha256(output),
                "receipt_content_sha256": "f" * 64,
                "case_id": fixture["law"].case_id,
                "shape_id": fixture["contract"].shape_id,
                "artifact_sha256": operation["artifact"]["sha256"],
                "case_fragment_path": str(fragment_path.resolve()),
                "case_fragment_file_sha256": _file_sha256(fragment_path),
                "case_fragment_body_sha256": "a" * 64,
                "result_class": "VALID_NEGATIVE",
            }
        )
        return dict(sealed)

    monkeypatch.setattr(
        provider_module,
        "seal_workhorse_terminalized_semantic_case_receipt_v2",
        seal_terminal,
    )
    monkeypatch.setattr(
        provider_module,
        "validate_workhorse_terminalized_semantic_case_receipt_v2",
        lambda _path: dict(sealed),
    )
    finalized = provider.finalize_attempt_terminal_v2(
        plan=fixture["plan"],
        run_root=fixture["run_root"],
        canonical_terminal_binding={"attempt_id": "ATTEMPT-CASE22"},
    )
    assert finalized["result_class"] == "VALID_NEGATIVE"
    assert finalized["lifecycle_binding"]["retention_class"] == "R3"
    replayed = provider.finalize_attempt_terminal_v2(
        plan=fixture["plan"],
        run_root=fixture["run_root"],
        canonical_terminal_binding={"attempt_id": "ATTEMPT-CASE22"},
    )
    assert replayed == finalized
    assert seal_calls == 1

    hostile = deepcopy(provider_result)
    hostile["attempt_terminal_contract"] = deepcopy(terminal)
    hostile["attempt_terminal_contract"]["attempt_terminal_status"] = "COMPLETED"
    manifest_path.write_text(json.dumps({"result": hostile}), encoding="ascii")
    with pytest.raises(CampaignProviderError, match="not exact"):
        provider.finalize_attempt_terminal_v2(
            plan=fixture["plan"],
            run_root=fixture["run_root"],
            canonical_terminal_binding={"attempt_id": "ATTEMPT-CASE22"},
        )


def test_regular_science_founder_acs2_binds_real_genesis_schedule_and_defaults() -> None:
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id="w64h64p1024c2048",
        backend_id="phase_graph_v2",
        B=1,
        temporal_k=2,
        total_ticks=6,
    )
    operation = {
        "science_continuation_contract_sha256": "a" * 64,
        "source_checkpoint_identity_sha256": None,
    }
    plan = {
        "execution": {
            "checkpoint_ticks": [2, 6],
            "acs_checkpoint_slots": 2,
        },
        "science_materialization": {"schema_version": "fixture-v1"},
        "cells": [
            {
                "native_world_control": {
                    "control_bindings": [
                        {
                            "control_id": "environment.resource.multiplier",
                            "resolved_value": 1.0,
                            "schedule": {"kind": "static"},
                        },
                        {
                            "control_id": "mutation.rate",
                            "resolved_value": 0.125,
                            "schedule": {"kind": "static"},
                        },
                    ]
                }
            }
        ],
    }
    ticks = provider_module._execution_row_checkpoint_ticks_v2(
        plan,
        operation=operation,
        contract=contract,
    )
    assert ticks == (2, 6)
    provider_module._validate_science_founder_default_controls_v1(
        plan,
        checkpoint_ticks=ticks,
        acs_checkpoint_slots=2,
    )

    nondefault = deepcopy(plan)
    nondefault["cells"][0]["native_world_control"]["control_bindings"][0][
        "resolved_value"
    ] = 1.5
    with pytest.raises(CampaignProviderError, match="native defaults"):
        provider_module._validate_science_founder_default_controls_v1(
            nondefault,
            checkpoint_ticks=ticks,
            acs_checkpoint_slots=2,
        )
    missing = deepcopy(plan)
    missing["cells"][0]["native_world_control"]["control_bindings"].pop()
    with pytest.raises(CampaignProviderError, match="requires ACS2 and explicit"):
        provider_module._validate_science_founder_default_controls_v1(
            missing,
            checkpoint_ticks=ticks,
            acs_checkpoint_slots=2,
        )
    with pytest.raises(CampaignProviderError, match="requires ACS2 and explicit"):
        provider_module._validate_science_founder_default_controls_v1(
            plan,
            checkpoint_ticks=ticks,
            acs_checkpoint_slots=1,
        )
