from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import eiel_composition.science_cuda_v2_host_context_v1 as host_context_module

from eiel_composition.science_cuda_v2_host_context_v1 import (
    SCIENCE_CUDA_V2_MUTSUP_CHECKPOINT_TICKS,
    SCIENCE_CUDA_V2_MUTSUP_OBSERVATION_TICKS,
    SCIENCE_CUDA_V2_MUTSUP_TASK_SEEDS,
    SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS,
    SCIENCE_CUDA_V2_W32_CARRIER_PROFILE_ID,
    SCIENCE_CUDA_V2_W32_WORLD_PROFILE_ID,
    ScienceCudaV2HostContextError,
    build_mutsup_w32_paired_seed_request_v1,
    build_w32_cuda_v2_mutsup_base_plan_v1,
    load_mutsup_w32_authoring_spec_v1,
    main,
    materialize_w32_cuda_v2_science_host_context_v1,
)
from eiel_composition.science_instrument_artifacts_v1 import (
    parse_science_host_context_v1,
    serialize_science_host_context_v1,
)
from eiel_composition.science_request_v2 import (
    normalize_rich_production_science_request_v2,
)
from eiel_composition.science_world_bundle_v1 import materialize_science_world_bundle_v1
from eiel_composition.science_instrument import compile_science_campaign_v1
from emergence_contracts import ContentDigest, ScienceRecordKindV1
from engine.campaigns.cuda_full_evolution_v2_provider import (
    CUDA_FULL_EVOLUTION_V2_BASELINE_DEVIATION_REQUIREMENT_IDS_V1,
    CudaFullEvolutionV2ExecutionProviderV1,
)
from engine.perf.cuda_full_evolution_v2 import native_loader as cuda_v2_native_loader


def test_materialized_context_binds_exact_w32_route_and_standard_observations() -> None:
    context = materialize_w32_cuda_v2_science_host_context_v1()

    assert context.route_binding.route_id == "tensor_world_cuda_full_evolution_v2"
    physical = context.physical_execution_core.to_value()
    assert context.route_binding.route_profile_sha256 == ContentDigest(
        **physical["route_profile_sha256"]
    )
    assert (
        physical["route_lineage"]["current_host_execution_route_sha256"]
        == (physical["route_profile_sha256"])
    )
    assert physical["route_lineage"]["retained_engineering_packet_route_sha256"] == {
        "algorithm": "sha256",
        "hex": "8ca159dfa9795ce68de6572b9351a9d5f03a72feda4f472dd0ac694a1b98b83e",
    }
    assert context.route_binding.native_build_closure_sha256.hex == (
        "c30ae715b3ea5513b96b9def0ab837c539b91bfd02d90e25a1552150de142cd4"
    )
    assert (
        context.route_adapter.world_profile_id == SCIENCE_CUDA_V2_W32_WORLD_PROFILE_ID
    )
    assert context.physical_execution_core.to_value()["carrier_profile_id"] == (
        SCIENCE_CUDA_V2_W32_CARRIER_PROFILE_ID
    )
    assert physical["native_artifact"]["binary_sha256"]["hex"] == (
        "944b34559d829980ac0a4b62ebb39b2c0226241b9a835f7148910a9386391d2d"
    )
    assert physical["device_identity"]["uuid"] == (
        "7f52eb4f-61e3-8c2b-d463-eadb4254b011"
    )
    profile = context.observation_profile_descriptors[0]
    assert profile.record_kinds == tuple(ScienceRecordKindV1)
    assert profile.required_nonempty_record_kinds == (
        ScienceRecordKindV1.CONTROL_BINDING,
        ScienceRecordKindV1.WORLD_COMPLETION,
    )
    assert set(
        (
            "birth",
            "death",
            "mutation",
            "birth_parentage",
            "mutation_provenance",
            "genome_identity",
            "lineage_ancestry",
            "birth_count",
            "mutation_count",
            "population_count",
        )
    ).issubset(context.available_observation_capability_ids)
    assert context.qualified_envelope.rows == ()
    assert context.crca_semantic_binding.absolute_start_next_tick == 0
    assert context.crca_semantic_binding.absolute_target_next_tick == 10_000
    assert context.crca_semantic_binding.trajectory_checkpoint_ticks == (
        0,
        5000,
        10000,
    )


def test_observation_component_identities_bind_live_host_dependencies() -> None:
    context = materialize_w32_cuda_v2_science_host_context_v1()
    repository_root = Path(__file__).resolve().parents[1]

    assert context.route_binding.observation_adapter_sha256 == (
        host_context_module._observation_adapter_digest(repository_root)
    )
    abi_path = repository_root / (
        "_spec/science_instrument/v1/science_observation_abi_v1.json"
    )
    assert context.instrument_manifest.observation_abi_sha256 == ContentDigest(
        "sha256", hashlib.sha256(abi_path.read_bytes()).hexdigest()
    )


def test_native_artifact_and_host_route_use_separate_dependency_identities(
    monkeypatch,
) -> None:
    context = materialize_w32_cuda_v2_science_host_context_v1(
        total_ticks=160,
        checkpoint_ticks=(160,),
    )
    plan = build_w32_cuda_v2_mutsup_base_plan_v1(
        context,
        total_ticks=160,
        checkpoint_ticks=(160,),
    )
    physical = context.physical_execution_core.to_value()
    artifact_path = Path(physical["native_artifact"]["path"])
    cuda_v2_native_loader.validate_prebuilt_cuda_full_evolution_v2(artifact_path)
    provider = CudaFullEvolutionV2ExecutionProviderV1()
    original_route = provider.execution_route_contract(plan)

    original_file_sha256 = cuda_v2_native_loader.file_sha256
    provider_source = (
        Path(__file__).resolve().parents[1]
        / "engine/campaigns/cuda_full_evolution_v2_provider.py"
    ).resolve()

    def host_changed(path: Path) -> str:
        if Path(path).resolve() == provider_source:
            return "0" * 64
        return original_file_sha256(Path(path))

    monkeypatch.setattr(cuda_v2_native_loader, "file_sha256", host_changed)
    # A host-only source change alters the execution route but does not falsely
    # claim that the retained CUDA binary itself needs rebuilding.
    cuda_v2_native_loader.validate_prebuilt_cuda_full_evolution_v2(artifact_path)
    changed_route = provider.execution_route_contract(plan)
    assert changed_route.execution_route_sha256 != original_route.execution_route_sha256
    original_artifact = next(
        row
        for row in original_route.components
        if row["component_id"] == "cuda_full_evolution_v2_artifact_unit"
    )
    changed_artifact = next(
        row
        for row in changed_route.components
        if row["component_id"] == "cuda_full_evolution_v2_artifact_unit"
    )
    assert changed_artifact == original_artifact


def test_native_dependency_change_invalidates_retained_cuda_artifact(
    monkeypatch,
) -> None:
    context = materialize_w32_cuda_v2_science_host_context_v1(
        total_ticks=160,
        checkpoint_ticks=(160,),
    )
    artifact_path = Path(
        context.physical_execution_core.to_value()["native_artifact"]["path"]
    )
    original_file_sha256 = cuda_v2_native_loader.file_sha256
    native_source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_full_evolution_v2/native/full_evolution_v2.cu"
    ).resolve()

    def native_changed(path: Path) -> str:
        if Path(path).resolve() == native_source:
            return "0" * 64
        return original_file_sha256(Path(path))

    monkeypatch.setattr(cuda_v2_native_loader, "file_sha256", native_changed)
    with pytest.raises(
        cuda_v2_native_loader.CudaFullEvolutionNativeLoadError,
        match="prebuilt native dependency mismatch",
    ):
        cuda_v2_native_loader.validate_prebuilt_cuda_full_evolution_v2(artifact_path)


def test_context_artifact_round_trips_without_effects() -> None:
    context = materialize_w32_cuda_v2_science_host_context_v1()
    artifact = serialize_science_host_context_v1(context)
    assert artifact["context_sha256"]["algorithm"] == "sha256"
    assert parse_science_host_context_v1(artifact) == context


def test_mutsup_request_is_six_explicit_paired_worlds_and_normalizable() -> None:
    context = materialize_w32_cuda_v2_science_host_context_v1()
    request = build_mutsup_w32_paired_seed_request_v1(context)
    normalized = normalize_rich_production_science_request_v2(request)
    campaign = normalized.campaign
    assert campaign.time.to_value()["total_ticks"] == SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS
    assert campaign.time.to_value()["checkpoint_ticks"] == list(
        SCIENCE_CUDA_V2_MUTSUP_CHECKPOINT_TICKS
    )
    assert campaign.time.to_value()["observation_ticks"] == list(
        SCIENCE_CUDA_V2_MUTSUP_OBSERVATION_TICKS
    )
    assert campaign.design.to_value()["expected_cell_count"] == 6
    assert campaign.question.to_value()["no_claims"] is True
    assert "fixed native mutation law" in campaign.question.to_value()["description"]
    assert campaign.observations.to_value()["profile_id"] == "SCIENCE_STANDARD_V1"
    assert campaign.artifact_lifecycle.to_value()["profile_id"] == "SCIENCE_COMPACT"

    compilation = compile_science_campaign_v1(
        campaign,
        instrument_manifest=context.instrument_manifest,
        route_binding=context.route_binding,
        control_descriptors=context.control_descriptors,
        reducer_descriptors=context.reducer_descriptors,
        crca_semantic_binding=context.crca_semantic_binding,
        physical_execution_core=context.physical_execution_core,
        compiler_component_sha256=context.compiler_component_sha256,
    )
    bundle = materialize_science_world_bundle_v1(
        ((campaign, compilation),),
        bundle_id="MUTSUP-W32-PAIRED-SEED-10K-TEST",
    )
    assert bundle.world_count == 6
    by_replicate: dict[int, set[str]] = {}
    for world in bundle.worlds:
        science = world.science.to_value()
        by_replicate.setdefault(int(science["replicate"]), set()).add(science["arm_id"])
    assert set(by_replicate) == set(SCIENCE_CUDA_V2_MUTSUP_TASK_SEEDS)
    assert by_replicate == {
        seed: {"mutation_enabled=False", "mutation_enabled=True"}
        for seed in SCIENCE_CUDA_V2_MUTSUP_TASK_SEEDS
    }
    reducer_config = campaign.reducers.to_value()[0]["configuration"]
    assert len(reducer_config["world_ownership"]) == 6
    assert reducer_config["pairing"] == {
        "off_arm_ids": ["mutation_enabled=False"],
        "on_arm_ids": ["mutation_enabled=True"],
    }
    assert reducer_config["milestone_offsets_ticks"] == [
        tick for tick in SCIENCE_CUDA_V2_MUTSUP_OBSERVATION_TICKS if tick > 0
    ]


def test_canary_is_same_six_pairs_with_clipped_absolute_milestones() -> None:
    request = build_mutsup_w32_paired_seed_request_v1(
        total_ticks=160,
        checkpoint_ticks=(160,),
    )
    campaign = normalize_rich_production_science_request_v2(request).campaign
    assert request["request_id"] == "mutsup-w32-paired-seed-160-canary-v1"
    assert campaign.design.to_value()["expected_cell_count"] == 6
    assert campaign.time.to_value()["observation_ticks"] == [0, 80, 160]
    assert campaign.time.to_value()["checkpoint_ticks"] == [160]
    assert campaign.question.to_value()["no_claims"] is True


def test_default_base_plan_uses_canonical_science_compact_release_policy() -> None:
    context = materialize_w32_cuda_v2_science_host_context_v1()
    plan = build_w32_cuda_v2_mutsup_base_plan_v1(
        context,
        total_ticks=SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS,
        checkpoint_ticks=SCIENCE_CUDA_V2_MUTSUP_CHECKPOINT_TICKS,
    )

    lifecycle = plan["artifact_lifecycle"]
    assert lifecycle["lifecycle_profile"] == "SCIENCE_COMPACT"
    assert lifecycle["prune"]["automatic_prune_allowed"] is True
    assert plan["output"]["evidence_closeout_enabled"] is False
    assert lifecycle["checkpoints"]["roles"] == [
        {
            "identity": "checkpoint-tick-000000005000",
            "checkpoint_tick": 5000,
            "role": "intermediate",
            "retention_class": "R1",
        },
        {
            "identity": "checkpoint-tick-000000010000",
            "checkpoint_tick": 10000,
            "role": "terminal",
            "retention_class": "R4",
        },
    ]


def test_cli_writes_reusable_immutable_context_request_and_base_plan(
    tmp_path: Path,
) -> None:
    context_path = tmp_path / "context.json"
    request_path = tmp_path / "request.json"
    base_path = tmp_path / "base-plan.json"
    argv = [
        "--context-output",
        str(context_path),
        "--request-output",
        str(request_path),
        "--base-plan-output",
        str(base_path),
        "--total-ticks",
        "160",
        "--checkpoint-ticks",
        "160",
    ]
    assert main(argv) == 0
    first = {
        path.name: path.read_bytes() for path in (context_path, request_path, base_path)
    }
    assert main(argv) == 0
    assert first == {
        path.name: path.read_bytes() for path in (context_path, request_path, base_path)
    }
    assert context_path.stat().st_mode & 0o222 == 0
    assert request_path.stat().st_mode & 0o222 == 0
    assert base_path.stat().st_mode & 0o222 == 0
    base = json.loads(base_path.read_text())
    assert base["execution"]["checkpoint_ticks"] == [160]
    assert base["execution"]["total_ticks"] == 160
    assert base["execution"]["budgets"]["wall_seconds"] == 9_600


def test_default_base_plan_budget_covers_provider_fail_closed_wall_bound() -> None:
    context = materialize_w32_cuda_v2_science_host_context_v1()
    base = build_w32_cuda_v2_mutsup_base_plan_v1(
        context,
        total_ticks=SCIENCE_CUDA_V2_MUTSUP_TOTAL_TICKS,
        checkpoint_ticks=SCIENCE_CUDA_V2_MUTSUP_CHECKPOINT_TICKS,
    )

    assert base["execution"]["budgets"]["wall_seconds"] == 600_000


def test_authoring_spec_is_compact_and_hash_free() -> None:
    spec = load_mutsup_w32_authoring_spec_v1()
    assert spec["task_seeds"] == list(SCIENCE_CUDA_V2_MUTSUP_TASK_SEEDS)
    assert "sha256" not in json.dumps(spec)
    assert spec["qualification_claims"] is False


def test_mutsup_request_declares_exact_cuda_v2_engineering_baseline_delta() -> None:
    context = materialize_w32_cuda_v2_science_host_context_v1(
        total_ticks=160,
        checkpoint_ticks=(160,),
    )
    request = build_mutsup_w32_paired_seed_request_v1(
        context,
        total_ticks=160,
        checkpoint_ticks=(160,),
    )

    baseline = request["baseline"]
    assert baseline["intent"] == "development"
    assert tuple(
        row["requirement_id"] for row in baseline["deviations"]
    ) == CUDA_FULL_EVOLUTION_V2_BASELINE_DEVIATION_REQUIREMENT_IDS_V1
    assert all(row["reason"].strip() for row in baseline["deviations"])


def test_route_summary_identity_mismatch_fails_closed() -> None:
    packet = json.loads(
        Path(
            "_artifacts/eiel-science-instrument-v1-gpuv2-20260827/"
            "engineering-complete/engineering_complete_packet_v1.json"
        ).read_text()
    )
    packet = copy.deepcopy(packet)
    packet["physical_route"]["execution_route_sha256"] = "0" * 64
    with pytest.raises(ScienceCudaV2HostContextError, match="identity mismatch"):
        materialize_w32_cuda_v2_science_host_context_v1(packet)


def test_base_plan_binds_context_physical_artifact() -> None:
    context = materialize_w32_cuda_v2_science_host_context_v1(
        total_ticks=160,
        checkpoint_ticks=(160,),
    )
    base = build_w32_cuda_v2_mutsup_base_plan_v1(
        context,
        total_ticks=160,
        checkpoint_ticks=(160,),
    )
    assert base["resolved_profile"] == "eiel_full_evolution_cuda_native_v2"
    assert base["artifacts"]["cuda_extension"]["sha256"] == (
        "944b34559d829980ac0a4b62ebb39b2c0226241b9a835f7148910a9386391d2d"
    )
