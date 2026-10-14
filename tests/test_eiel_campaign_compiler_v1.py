from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from engine.campaigns.compiler import (
    CampaignError,
    compile_campaign,
    load_campaign_spec,
)
from engine.campaigns.providers import (
    ProductionEngineLifecycleV1,
    ResidentBatchRequestV1,
)
from engine.campaigns.registry import ProfileRegistryV1, default_profile_registry
from engine.campaigns.science_benchmark_clocks_v1 import (
    packed_tick_horizon_v1,
    terminal_only_acs_policy_v1,
)
from engine.campaigns.tick_native import TICK_NATIVE_EXECUTION_ROUTE_V1

ROOT = Path(__file__).resolve().parents[1]
SPEC_ROOT = ROOT / "_spec/campaigns/v1"


def _load(name: str) -> dict:
    return load_campaign_spec(SPEC_ROOT / name)


def test_fixed_patch_awkward_matrix_compiles_deterministically() -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    left = compile_campaign(spec)
    right = compile_campaign(copy.deepcopy(spec))

    assert left.payload() == right.payload()
    assert left.plan_sha256 == right.plan_sha256
    assert left.generations == 14
    assert len(left.cells) == 21
    assert [len(batch.cell_ids) for batch in left.batches] == [2] * 10 + [1]
    assert left.checkpoints == (7, 14)
    assert left.checkpoint_storage_mode == "disk"
    assert left.payload()["execution"]["checkpoint_storage_mode"] == "disk"
    assert len(left.comparison_branches) == 5
    assert [branch.kind for branch in left.comparison_branches] == [
        "fresh_repeat",
        "fresh_repeat",
        "checkpoint_continuation",
        "checkpoint_continuation",
        "reporting_parity",
    ]
    assert left.payload()["comparisons"]["reporting_parity"] == {
        "required": True,
        "accepted_evidence_modes": [
            "enabled_disabled_exact_terminal_state",
            "provider_observational_contract_with_state_mutation_sentinel",
        ],
    }
    assert left.payload()["execution"]["budgets"] == spec["execution"]["budgets"]
    assert left.payload()["output"] == {
        "runtime_root": spec["output"]["runtime_root"],
        "reporting_profile": spec["output"]["reporting_profile"],
        "chronicle_profile": "notable",
        "chronicle_history_root": "/tmp/emergence-test/runtime/chronicle_history",
        "chronicle_archive_enabled": True,
        "evidence_closeout_enabled": False,
    }


@pytest.mark.parametrize("profile", ["summary", "notable", "scientific_full"])
def test_chronicle_output_profile_and_history_root_are_plan_bound(
    profile: str,
) -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["output"].update(
        {
            "chronicle_profile": profile,
            "chronicle_history_root": "/data/test-chronicle-history",
            "chronicle_archive_enabled": False,
        }
    )

    plan = compile_campaign(spec)

    assert plan.output["chronicle_profile"] == profile
    assert plan.output["chronicle_history_root"] == "/data/test-chronicle-history"
    assert plan.output["chronicle_archive_enabled"] is False


@pytest.mark.parametrize("profile", ["off", "debug_trace", "verbose", True])
def test_chronicle_output_profile_fails_closed(profile: object) -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["output"]["chronicle_profile"] = profile

    with pytest.raises(CampaignError, match="output.chronicle_profile"):
        compile_campaign(spec)


def test_checkpoint_storage_mode_is_explicit_and_plan_bound() -> None:
    omitted = _load("patch_fixed_g14_irregular.yaml")
    explicit_disk = copy.deepcopy(omitted)
    explicit_disk["checkpoints"]["storage_mode"] = "disk"
    disk_and_memory = copy.deepcopy(omitted)
    disk_and_memory["checkpoints"]["storage_mode"] = "disk_and_memory"

    default_plan = compile_campaign(omitted)
    explicit_plan = compile_campaign(explicit_disk)
    retained_plan = compile_campaign(disk_and_memory)

    assert default_plan.payload() == explicit_plan.payload()
    assert default_plan.plan_sha256 == explicit_plan.plan_sha256
    assert retained_plan.checkpoint_storage_mode == "disk_and_memory"
    assert retained_plan.payload()["execution"]["checkpoint_storage_mode"] == (
        "disk_and_memory"
    )
    assert retained_plan.plan_sha256 != default_plan.plan_sha256


@pytest.mark.parametrize("value", ["memory", "disk_memory", "", True])
def test_checkpoint_storage_mode_fails_closed(value) -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["checkpoints"]["storage_mode"] = value

    with pytest.raises(CampaignError, match="checkpoints.storage_mode"):
        compile_campaign(spec)


def test_acs_checkpoint_slots_default_to_two_and_zero_is_sync_opt_out() -> None:
    omitted = _load("patch_fixed_g14_irregular.yaml")
    sync_spec = copy.deepcopy(omitted)
    sync_spec["checkpoints"]["acs_slots"] = 0

    default_plan = compile_campaign(omitted)
    sync_plan = compile_campaign(sync_spec)

    assert default_plan.acs_checkpoint_slots == 2
    assert default_plan.payload()["execution"]["acs_checkpoint_slots"] == 2
    assert default_plan.acs_capture_instrumentation is True
    assert sync_plan.acs_checkpoint_slots == 0
    assert sync_plan.payload()["execution"]["acs_checkpoint_slots"] == 0
    assert sync_plan.plan_sha256 != default_plan.plan_sha256


@pytest.mark.parametrize("value", [-1, 4, "2", True])
def test_acs_checkpoint_slots_fail_closed(value: object) -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["checkpoints"]["acs_slots"] = value

    with pytest.raises(CampaignError, match="checkpoints.acs_slots"):
        compile_campaign(spec)


def test_acs_capture_instrumentation_is_independently_opt_in() -> None:
    omitted = _load("patch_fixed_g14_irregular.yaml")
    instrumented = copy.deepcopy(omitted)
    instrumented["checkpoints"]["acs_capture_instrumentation"] = True

    default_plan = compile_campaign(omitted)
    instrumented_plan = compile_campaign(instrumented)

    assert default_plan.acs_checkpoint_slots == 2
    assert default_plan.acs_capture_instrumentation is True
    assert instrumented_plan.acs_checkpoint_slots == 2
    assert instrumented_plan.acs_capture_instrumentation is True
    assert (
        instrumented_plan.payload()["execution"]["acs_capture_instrumentation"] is True
    )
    sync_instrumented = copy.deepcopy(omitted)
    sync_instrumented["checkpoints"]["acs_slots"] = 0
    sync_instrumented["checkpoints"]["acs_capture_instrumentation"] = True
    sync_instrumented_plan = compile_campaign(sync_instrumented)
    assert sync_instrumented_plan.acs_checkpoint_slots == 0
    assert sync_instrumented_plan.acs_capture_instrumentation is True
    assert sync_instrumented_plan.plan_sha256 != default_plan.plan_sha256


def test_native_streamed_checkpoint_route_is_explicit_and_plan_bound() -> None:
    omitted = _load("patch_fixed_g14_irregular.yaml")
    native = copy.deepcopy(omitted)
    native["checkpoints"]["native_streamed_checkpoint"] = True

    default_plan = compile_campaign(omitted)
    native_plan = compile_campaign(native)

    assert default_plan.native_streamed_checkpoint is False
    assert "native_streamed_checkpoint" not in default_plan.payload()["execution"]
    assert native_plan.native_streamed_checkpoint is True
    assert native_plan.payload()["execution"]["native_streamed_checkpoint"] is True
    assert native_plan.plan_sha256 != default_plan.plan_sha256


@pytest.mark.parametrize("value", [0, 1, "true", None, [], {}])
def test_native_streamed_checkpoint_selection_fails_closed(value: object) -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["checkpoints"]["native_streamed_checkpoint"] = value

    with pytest.raises(CampaignError, match="checkpoints.native_streamed_checkpoint"):
        compile_campaign(spec)


def test_world_sharded_checkpoint_route_is_explicit_tick_native_and_plan_bound() -> None:
    spec = _tick_native_5000_spec()
    spec["checkpoints"]["native_streamed_checkpoint"] = True
    spec["checkpoints"]["world_sharded_checkpoint"] = True

    plan = compile_campaign(spec)

    assert plan.world_sharded_checkpoint is True
    assert plan.payload()["execution"]["world_sharded_checkpoint"] is True


def test_world_sharded_checkpoint_requires_native_streamed_and_absolute_ticks() -> None:
    legacy = _load("patch_fixed_g14_irregular.yaml")
    legacy["checkpoints"]["world_sharded_checkpoint"] = True
    with pytest.raises(CampaignError, match="requires native_streamed_checkpoint"):
        compile_campaign(legacy)

    legacy["checkpoints"]["native_streamed_checkpoint"] = True
    with pytest.raises(CampaignError, match="requires absolute-tick"):
        compile_campaign(legacy)


@pytest.mark.parametrize("value", [0, 1, "true", None, [], {}])
def test_world_sharded_checkpoint_selection_fails_closed(value: object) -> None:
    spec = _tick_native_5000_spec()
    spec["checkpoints"]["world_sharded_checkpoint"] = value

    with pytest.raises(CampaignError, match="checkpoints.world_sharded_checkpoint"):
        compile_campaign(spec)


@pytest.mark.parametrize("value", [0, 1, "true", None, [], {}])
def test_acs_capture_instrumentation_fails_closed(value: object) -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["checkpoints"]["acs_capture_instrumentation"] = value

    with pytest.raises(CampaignError, match="checkpoints.acs_capture_instrumentation"):
        compile_campaign(spec)


def _tick_native_5000_spec() -> dict:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["execution"].pop("generations")
    spec["execution"].pop("ticks_per_generation")
    spec["execution"]["execution_route_identity"] = TICK_NATIVE_EXECUTION_ROUTE_V1
    spec["execution"]["total_ticks"] = 5000
    spec["execution"]["checkpoint_ticks"] = [5000]
    spec["checkpoints"]["milestones"] = []
    spec["checkpoints"].pop("digest_every_generation", None)
    spec["checkpoints"]["acs_slots"] = 1
    spec["comparisons"]["fresh_repeats"] = 0
    spec["comparisons"]["continuation_splits"] = []
    spec["comparisons"]["continuation_cells"] = 0
    return spec


def test_tick_native_5000_terminal_acs_compiles_without_interior_checkpoints() -> None:
    plan = compile_campaign(_tick_native_5000_spec())
    payload = plan.payload(include_hash=False)
    execution = payload["execution"]
    policy = terminal_only_acs_policy_v1(plan)

    assert execution["execution_route_identity"] == TICK_NATIVE_EXECUTION_ROUTE_V1
    assert execution["total_ticks"] == 5000
    assert execution["checkpoint_ticks"] == [5000]
    assert "generations" not in execution
    assert "ticks_per_generation" not in execution
    assert execution["tick_native_control"]["native_dispatch_window_ticks"] == 1120
    assert execution["tick_native_control"]["source_window_ticks"] == 160
    assert execution["tick_native_control"]["time_authority"] == "absolute_tick_v1"
    assert "digest_every_generation" not in execution
    assert execution["acs_checkpoint_slots"] == 1
    assert packed_tick_horizon_v1(plan) == 5000
    assert policy["terminal_only"] is True
    assert policy["interior_checkpoint_ticks"] == []
    assert plan.generations is None
    assert plan.ticks_per_generation is None


def test_tick_native_native_checkpoint_keeps_absolute_tick_control_authoritative() -> (
    None
):
    spec = _tick_native_5000_spec()
    spec["checkpoints"]["native_streamed_checkpoint"] = True

    plan = compile_campaign(spec)
    execution = plan.payload(include_hash=False)["execution"]

    assert plan.native_streamed_checkpoint is True
    assert execution["native_streamed_checkpoint"] is True
    assert execution["time_authority"] == "absolute_tick_v1"
    assert execution["tick_native_control"]["resident_owner_span_ticks"] == 1120
    assert execution["tick_native_control"]["native_dispatch_window_ticks"] == 1120
    assert execution["tick_native_control"]["source_window_ticks"] == 160
    assert plan.comparison_branches == ()
    assert plan.resource_estimate["comparison_trajectories"] == 0


@pytest.mark.parametrize(
    "field",
    [
        "generations",
        "ticks_per_generation",
        "physical_window_generations",
        "internal_lifecycle_interval_ticks",
        "max_internal_lifecycle_segments_per_dispatch",
        "internal_lifecycle_schedule",
        "refill_chunk_ticks",
        "legacy_turnover_interval_ticks",
    ],
)
def test_tick_native_rejects_legacy_generation_and_lifecycle_authority(
    field: str,
) -> None:
    spec = _tick_native_5000_spec()
    spec["execution"][field] = (
        {"ticks": [160]} if field == "internal_lifecycle_schedule" else 32
    )

    with pytest.raises(CampaignError, match=field):
        compile_campaign(spec)


def test_science_representative_fe_spec_binds_one_physical_b136_batch() -> None:
    spec = _load("tw_rngv2/science_representative_5000tick_v1.yaml")
    plan = compile_campaign(spec)
    payload = plan.payload(include_hash=False)
    policy = terminal_only_acs_policy_v1(plan)

    assert spec["run_contract"]["profile_id"] == "eiel_full_evolution_v1"
    assert packed_tick_horizon_v1(plan) == 5000
    assert policy["terminal_only"] is False
    assert policy["interior_checkpoint_ticks"] == [2500]
    assert spec["checkpoints"]["acs_slots"] == 0
    assert payload["execution"]["acs_checkpoint_slots"] == 0
    assert "digest_every_generation" not in payload["execution"]
    assert len(plan.cells) == 136
    assert [len(batch.cell_ids) for batch in plan.batches] == [136]
    assert payload["execution"]["tick_native_control"]["event_window_capacity"] == (
        41_804
    )
    assert plan.resource_estimate["compiled_logical_cells"] == 136
    assert plan.resource_estimate["compiled_batch_count"] == 1
    assert plan.resource_estimate["compiled_physical_batch_cells_min"] == 136
    assert plan.resource_estimate["compiled_physical_batch_cells_max"] == 136
    assert plan.resource_estimate["compiled_physical_batch_cells_total"] == 136
    assert plan.resource_estimate["compiled_partition"] is False
    assert plan.resource_estimate["peak_host_checkpoint_import_scratch_bytes"] == (
        64 * 1024 * 1024
    )
    assert plan.resource_estimate["resident_event_capacity"] == 41_804
    assert plan.resource_estimate["resident_event_bytes_horizon_invariant"] is True
    assert plan.resource_estimate["resident_source_bytes_horizon_invariant"] is True
    assert plan.resource_estimate["checkpoint_accounting_basis"] == (
        "acs100x_fixed_mlp_b136_aggregate_projection_v1"
    )
    assert plan.resource_estimate["checkpoint_representation"] == (
        "aggregate_physical_batch_v1"
    )
    assert plan.resource_estimate["checkpoint_physical_object_count_upper_bound"] == 2
    assert plan.resource_estimate["checkpoint_bytes_upper_bound"] == 7_826_974_528
    assert plan.resource_estimate["disk_bytes_upper_bound"] == 7_848_734_528
    assert (
        plan.resource_estimate["disk_bytes_upper_bound"]
        <= spec["artifact_lifecycle"]["storage"]["total_max_bytes"]
    )
    assert (
        plan.resource_estimate["checkpoint_bytes_upper_bound"]
        <= spec["artifact_lifecycle"]["storage"]["checkpoint_max_bytes"]
    )
    assert payload["artifact_lifecycle"]["checkpoints"]["authority"] == "absolute_tick"
    assert payload["artifact_lifecycle"]["checkpoints"]["checkpoint_ticks"] == [
        2500,
        5000,
    ]
    assert (
        payload["artifact_lifecycle"]["checkpoints"]["roles"][0]["checkpoint_tick"]
        == 2500
    )
    assert "generation" not in payload["artifact_lifecycle"]["checkpoints"]["roles"][0]
    assert plan.run_contract["requested"]["profile_id"] == "eiel_full_evolution_v1"
    assert plan.run_contract["admission_status"] == "passed"
    assert (
        spec["execution"]["tick_native_control"]["native_dispatch_window_ticks"] == 1120
    )
    assert (
        payload["execution"]["tick_native_control"]["native_dispatch_window_ticks"]
        == 1120
    )
    assert plan.run_contract["required_capabilities"] == [
        "patch",
        "wdl_delayed_hazard",
        "jungle",
        "eed",
        "genome_v2_ordinary_birth",
        "genome_v2_mutation",
        "recombination_profile",
    ]
    assert all(
        row["configured"] and row["initializable"]
        for row in plan.run_contract["capabilities"].values()
    )


def test_science_representative_b2_control_is_explicitly_diagnostic() -> None:
    spec = _load(
        "tw_rngv2/science_representative_5000tick_b2_diagnostic_control_v1.yaml"
    )
    plan = compile_campaign(spec)

    assert "DIAGNOSTIC_PHYSICAL_B2_ONLY" in spec["description"]
    assert "DIAGNOSTIC_PHYSICAL_B2_ONLY" in spec["no_claims"]
    assert len(plan.cells) == 136
    assert [len(batch.cell_ids) for batch in plan.batches] == [2] * 68
    assert plan.resource_estimate["compiled_batch_count"] == 68
    assert plan.resource_estimate["compiled_partition"] is True
    assert plan.resource_estimate["physical_batch_cells"] == 2
    assert (
        plan.payload()["execution"]["tick_native_control"]["event_window_capacity"]
        == 41_804
    )
    assert plan.payload()["execution"]["acs_checkpoint_slots"] == 0
    assert plan.payload()["execution"]["native_streamed_checkpoint"] is True
    assert plan.payload()["execution"]["acs_capture_instrumentation"] is True


def test_terminal_b136_benchmark_is_terminal_only_and_observational() -> None:
    spec = _load("tw_rngv2/science_representative_5000tick_terminal_benchmark_v1.yaml")
    plan = compile_campaign(spec)
    payload = plan.payload()
    policy = terminal_only_acs_policy_v1(plan)

    assert "Observational physical-B136 timing benchmark only" in spec["no_claims"]
    assert [len(batch.cell_ids) for batch in plan.batches] == [136]
    assert payload["execution"]["checkpoint_ticks"] == [5000]
    assert payload["execution"]["acs_checkpoint_slots"] == 0
    assert payload["execution"]["acs_capture_instrumentation"] is True
    assert policy["terminal_only"] is True
    assert policy["interior_checkpoint_ticks"] == []
    assert plan.output["chronicle_profile"] == "scientific_full"
    assert plan.resource_estimate["compiled_partition"] is False


def test_tick_native_restore_schedule_survives_compilation() -> None:
    spec = _load("tw_rngv2/science_representative_5000tick_restore_v1.yaml")
    plan = compile_campaign(spec)
    execution = plan.payload()["execution"]

    assert plan.campaign_id.endswith("acs2-w32p256c512-20260824")
    assert [len(batch.cell_ids) for batch in plan.batches] == [136]
    assert execution["tick_native_control"]["event_window_capacity"] == 41_804
    assert execution["deliberate_restore_tick"] == 2500
    assert execution["deliberate_restore_ticks"] == [2500]
    assert execution["acs_checkpoint_slots"] == 2
    assert execution["acs_capture_instrumentation"] is True
    assert plan.payload()["artifact_lifecycle"]["lifecycle_profile"] == "QUALIFICATION"


def test_qualification_twins_compile_as_one_physical_b136_pair() -> None:
    restore = compile_campaign(
        _load("tw_rngv2/science_representative_5000tick_restore_v1.yaml")
    ).payload()
    uninterrupted = compile_campaign(
        _load(
            "tw_rngv2/science_representative_5000tick_qualification_uninterrupted_v1.yaml"
        )
    ).payload()

    def normalized(plan: dict) -> dict:
        value = copy.deepcopy(plan)
        for field in ("plan_sha256", "campaign_id", "description", "no_claims"):
            value.pop(field, None)
        value["output"].pop("runtime_root", None)
        value["execution"].pop("deliberate_restore_tick", None)
        value["execution"].pop("deliberate_restore_ticks", None)
        return value

    assert normalized(restore) == normalized(uninterrupted)
    for plan in (restore, uninterrupted):
        assert len(plan["cells"]) == 136
        assert [len(batch["cell_ids"]) for batch in plan["physical_batches"]] == [136]
        assert plan["resource_estimate"]["compiled_batch_count"] == 1
        assert plan["artifact_lifecycle"]["lifecycle_profile"] == "QUALIFICATION"
        assert (
            plan["artifact_lifecycle"]["retention"]["qualification_hold_binding"]
            == "required"
        )
        assert plan["artifact_lifecycle"]["prune"]["automatic_prune_allowed"] is True
        assert plan["execution"]["total_ticks"] == 5000
        assert plan["execution"]["checkpoint_ticks"] == [2500, 5000]
        assert plan["execution"]["acs_checkpoint_slots"] == 2


def test_historical_b48_50000_tick_restore_sentinel_compiles_exactly() -> None:
    spec = _load("tw_rngv2/historical_b48_w32_50000tick_restore_v1.yaml")
    plan = compile_campaign(spec)
    execution = plan.payload()["execution"]

    assert len(plan.cells) == 48
    assert len(plan.batches) == 24
    assert {len(batch.cell_ids) for batch in plan.batches} == {2}
    assert execution["total_ticks"] == 50_000
    assert execution["checkpoint_ticks"] == list(range(5_000, 50_001, 5_000))
    assert execution["deliberate_restore_ticks"] == [25_000]


def test_tick_native_absolute_dispatch_window_is_capacity_bounded() -> None:
    request = ResidentBatchRequestV1(
        profile_id="profile-v1",
        batch_id="batch-0000",
        cell_ids=("cell-a",),
        checkpoint_milestones=(),
        parameter_arrays=(),
        seed_arrays=(),
        total_ticks=5000,
        checkpoint_ticks=(5000,),
        resident_owner_span_ticks=1120,
        native_dispatch_window_ticks=1120,
        source_window_ticks=160,
        event_window_capacity=41804,
        tick_native_control={
            "time_authority": "absolute_tick_v1",
            "total_ticks": 5000,
            "resident_owner_span_ticks": 1120,
            "native_dispatch_window_ticks": 1120,
            "source_window_ticks": 160,
            "event_window_capacity": 41804,
        },
    )
    segments = ProductionEngineLifecycleV1._absolute_tick_segments_v1(
        request, start_next_tick=0
    )

    assert [row["end_next_tick"] for row in segments if row["publish_checkpoint"]] == [
        5000
    ]
    assert segments[-1]["end_next_tick"] == 5000
    assert all(row["absolute_tick_span"] <= 1120 for row in segments)
    assert segments[0]["end_next_tick"] == 1120
    assert all(row["dispatch_window_ticks"] == 1120 for row in segments)


def test_physical_window_generation_is_explicit_and_plan_bound() -> None:
    omitted = _load("patch_fixed_g14_irregular.yaml")
    g1 = copy.deepcopy(omitted)
    g1["execution"]["physical_window_generations"] = 1

    default_plan = compile_campaign(omitted)
    g1_plan = compile_campaign(g1)

    assert "physical_window_generations" not in default_plan.payload()["execution"]
    assert g1_plan.physical_window_generations == 1
    assert g1_plan.payload()["execution"]["physical_window_generations"] == 1
    assert g1_plan.plan_sha256 != default_plan.plan_sha256


@pytest.mark.parametrize("value", [0, 3, "1", True])
def test_physical_window_generation_fails_closed(value: object) -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["execution"]["physical_window_generations"] = value

    with pytest.raises(CampaignError, match="physical_window_generations"):
        compile_campaign(spec)


def test_stateful_g37_resolves_by_capabilities_not_lane_name() -> None:
    plan = compile_campaign(_load("wdl_gru_g37_irregular.yaml"))

    assert plan.profile_id == "eiel-production-runtime-v1"
    assert plan.generations == 37
    assert plan.checkpoints == (7, 19, 37)
    assert len(plan.cells) == 21
    assert [len(batch.cell_ids) for batch in plan.batches] == [1] * 21
    cells = {cell.cell_id: cell for cell in plan.cells}
    profile = next(
        profile
        for profile in default_profile_registry().profiles()
        if profile.profile_id == plan.profile_id
    )
    for batch in plan.batches:
        for parameter in profile.uniform_batch_parameters:
            values = {
                cells[cell_id].parameters[parameter] for cell_id in batch.cell_ids
            }
            assert len(values) == 1
    assert plan.resource_estimate["physical_batch_cells"] == 1
    assert profile.public_receipt()["uniform_batch_parameters"] == list(
        profile.uniform_batch_parameters
    )


def test_patch_stable_gru_resolves_without_wdl_overlay() -> None:
    spec = _load("wdl_gru_g37_irregular.yaml")
    spec["campaign_id"] = "patch-stable-gru-g37"
    spec["world"]["overlays"] = []
    for name in tuple(spec["parameters"]):
        if name.startswith("wdl."):
            del spec["parameters"][name]
    spec["matrix"]["dimensions"] = spec["matrix"]["dimensions"][:2]
    spec["matrix"]["parameter_bindings"] = {}
    spec["matrix"]["seed_bindings"] = {
        "ecology_seed": "ecology",
        "task_seed": "task",
    }
    spec["matrix"]["expected_cells"] = 7
    spec["analysis"] = {
        "required_metrics": ["policy.recurrent_cost"],
        "optional_metrics": ["lineage.active"],
        "rules": [],
    }

    plan = compile_campaign(spec)

    assert plan.profile_id == "eiel-production-runtime-v1"
    assert len(plan.cells) == 7
    assert [len(batch.cell_ids) for batch in plan.batches] == [1] * 7


@pytest.mark.parametrize("consequence_profile", ["legacy", "P1", "P2", "P3"])
@pytest.mark.parametrize("terminal_no_reinit", [False, True])
def test_gmnw_venv_mechanics_and_outcome_metrics_resolve_by_capability(
    consequence_profile: str, terminal_no_reinit: bool
) -> None:
    spec = _load("gmnw_gru_b3g9_qualification.yaml")
    spec["parameters"].update(
        {
            "gmnw.venv_consequence_profile": consequence_profile,
            "gmnw.terminal_low_population_no_reinit": terminal_no_reinit,
        }
    )
    spec["analysis"]["required_metrics"] = [
        "outcome.decision_count",
        "outcome.correct_count",
        "outcome.task_death_count",
    ]

    plan = compile_campaign(spec)

    assert plan.profile_id == "eiel-production-runtime-v1"
    assert {
        cell.parameters["gmnw.venv_consequence_profile"] for cell in plan.cells
    } == {consequence_profile}
    assert {
        cell.parameters["gmnw.terminal_low_population_no_reinit"] for cell in plan.cells
    } == {terminal_no_reinit}


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("gmnw.venv_consequence_profile", "P4"),
        ("gmnw.venv_consequence_profile", 1),
        ("gmnw.terminal_low_population_no_reinit", 1),
        ("gmnw.terminal_low_population_no_reinit", "true"),
    ],
)
def test_gmnw_venv_mechanics_fail_their_exact_typed_contract(
    parameter: str, value: object
) -> None:
    spec = _load("gmnw_gru_b3g9_qualification.yaml")
    spec["parameters"][parameter] = value

    with pytest.raises(
        CampaignError, match=rf"{parameter!r} violates its typed contract"
    ):
        compile_campaign(spec)


@pytest.mark.parametrize("consequence_profile", ["legacy", "P1", "P2", "P3"])
@pytest.mark.parametrize("terminal_no_reinit", [False, True])
def test_gmnw_venv_treatments_reuse_the_mechanic_qualification(
    consequence_profile: str, terminal_no_reinit: bool
) -> None:
    base = _load("gmnw_gru_b3g9_qualification.yaml")
    base["parameters"].update(
        {
            "gmnw.venv_consequence_profile": "legacy",
            "gmnw.terminal_low_population_no_reinit": False,
        }
    )
    changed = copy.deepcopy(base)
    changed["parameters"]["gmnw.venv_consequence_profile"] = consequence_profile
    changed["parameters"]["gmnw.terminal_low_population_no_reinit"] = terminal_no_reinit

    original = compile_campaign(base)
    changed_plan = compile_campaign(changed)

    assert (
        "parameter.gmnw.venv_consequence_profile"
        not in original.qualification_key["fixed_invariants"]
    )
    assert (
        "parameter.gmnw.terminal_low_population_no_reinit"
        not in original.qualification_key["fixed_invariants"]
    )
    assert changed_plan.qualification_key_sha256 == original.qualification_key_sha256
    if (consequence_profile, terminal_no_reinit) == ("legacy", False):
        assert changed_plan.plan_sha256 == original.plan_sha256
    else:
        assert changed_plan.plan_sha256 != original.plan_sha256

    profile = next(
        row
        for row in default_profile_registry().profiles()
        if row.profile_id == original.profile_id
    )
    for name in (
        "gmnw.venv_consequence_profile",
        "gmnw.terminal_low_population_no_reinit",
    ):
        contract = profile.parameter_contracts[name]
        assert contract.per_cell is False
        assert contract.qualification_affecting is False


def test_gmnw_outcome_metrics_are_owned_only_by_gmnw_capability() -> None:
    metric_names = {
        "outcome.decision_count",
        "outcome.correct_count",
        "outcome.task_death_count",
    }

    gmnw_spec = _load("gmnw_gru_b3g9_qualification.yaml")
    gmnw_spec["analysis"]["required_metrics"] = sorted(metric_names)
    gmnw_spec["analysis"]["rules"] = []
    gmnw_gpu = compile_campaign(gmnw_spec)
    assert metric_names <= set(gmnw_gpu.payload()["analysis"]["required_metrics"])

    gmnw_cpu_spec = copy.deepcopy(gmnw_spec)
    gmnw_cpu_spec["campaign_id"] = "gmnw-gru-b3g9-qualification-cpu-native"
    gmnw_cpu_spec["execution"]["backend"] = "cpu_native"
    gmnw_cpu_spec["execution"]["batch"] = {"mode": "bounded", "maximum": 1}
    gmnw_cpu = compile_campaign(gmnw_cpu_spec)
    assert gmnw_cpu.profile_id == "eiel-production-runtime-v1-cpu-native"
    assert metric_names <= set(gmnw_cpu.payload()["analysis"]["required_metrics"])

    patch_spec = _load("patch_fixed_g14_irregular.yaml")
    patch_spec["analysis"]["required_metrics"] = sorted(metric_names)
    patch_spec["analysis"]["rules"] = []
    with pytest.raises(CampaignError, match="missing_metric_provider"):
        compile_campaign(patch_spec)

    patch_cpu_spec = copy.deepcopy(patch_spec)
    patch_cpu_spec["campaign_id"] = "patch-fixed-g14-irregular-cpu-native"
    patch_cpu_spec["execution"]["backend"] = "cpu_native"
    patch_cpu_spec["execution"]["batch"] = {"mode": "bounded", "maximum": 1}
    with pytest.raises(CampaignError, match="missing_metric_provider"):
        compile_campaign(patch_cpu_spec)


def test_eed_overlay_resolves_metrics_and_b48_admission_envelope() -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["world"]["overlays"] = ["eed_resource_impact"]
    spec["parameters"].update(
        {
            "eed.founder_resource_axis_profile": "symmetric_mix",
            "eed.resource_impact_apply_enable": True,
            "eed.ecology_frozen": False,
            "eed.phenotype_fixed": True,
            "genome_upgrade_v2.profile": "genome_upgrade_v2_fixed_q10_v1",
            "genome_upgrade_v2.enable": True,
            "genome_upgrade_v2.mutation_rate_q10": 20,
            "genome_upgrade_v2.mutation_step_q10": 64,
        }
    )
    spec["analysis"]["required_metrics"] = [
        "population.alive",
        "population.successful_births_cumulative",
        "eed.resource_a_stock_total",
        "eed.resource_a_stock_per_capita",
        "eed.resource_b_stock_total",
        "eed.reproductive_opportunity_successful_births",
        "eed.pre_birth_opportunity_count",
        "eed.pre_birth_opportunity_mass",
        "eed.birth_attempts",
        "eed.successful_births",
        "eed.unique_opportunity_bearing_parents",
        "eed.committed_mutation_events",
        "eed.hereditary_change_outcomes",
        "eed.population_abundance",
        "eed.opportunity_mass_per_living_agent",
        "eed.opportunity_mass_per_opportunity_parent",
        "eed.resource_axis_mean",
    ]
    spec["analysis"]["rules"] = []
    spec["execution"]["batch"] = {"mode": "bounded", "maximum": 1}

    plan = compile_campaign(spec)

    assert plan.profile_id == "eiel-production-runtime-v1"
    assert plan.qualification_key["admitted_batch_range"] == [1, 48]
    fixed = plan.qualification_key["fixed_invariants"]
    assert "parameter.eed.founder_resource_axis_profile" not in fixed
    assert fixed["parameter.eed.resource_impact_apply_enable"] is True
    assert fixed["parameter.eed.ecology_frozen"] is False
    assert fixed["parameter.eed.phenotype_fixed"] is True


def test_checked_in_eed_b1_spec_binds_lifecycle_v2_hybrid_controls() -> None:
    spec = _load("eed_b48_reciprocal_feedback_g36.yaml")
    impact_off = copy.deepcopy(spec)
    impact_off["matrix"]["include"] = [
        {
            "impact_apply": False,
            "ecology_frozen": False,
            "phenotype_fixed": False,
        }
    ]
    impact_off["matrix"]["expected_cells"] = 12
    impact_off["comparisons"]["reporting_selector"]["impact_apply"] = False

    plan = compile_campaign(impact_off)

    assert len(plan.cells) == 12
    for cell in plan.cells:
        assert cell.parameters["production_lifecycle.turnover_enabled"] is True
        assert cell.parameters["production_lifecycle.ordinary_births_enabled"] is True
        assert (
            cell.parameters["production_lifecycle.lifecycle_mode"]
            == "hybrid_overlapping_v1"
        )
        assert (
            cell.parameters["production_lifecycle.life_history_profile"]
            == "stage1s_overlap_tick_life_history_v1"
        )
        assert cell.parameters["production_lifecycle.turnover_cull_fraction"] == 0.0
        assert cell.parameters["production_lifecycle.turnover_parent_fraction"] == 1.0
    fixed = plan.qualification_key["fixed_invariants"]
    assert fixed["runtime.selection_mode"] == "genotype_blind_neutral"
    assert fixed["parameter.production_lifecycle.ordinary_births_enabled"] is True
    assert fixed["parameter.production_lifecycle.lifecycle_mode"] == (
        "hybrid_overlapping_v1"
    )
    assert fixed["parameter.production_lifecycle.life_history_profile"] == (
        "stage1s_overlap_tick_life_history_v1"
    )
    assert fixed["parameter.production_lifecycle.turnover_cull_fraction"] == 0.0


def test_eed_treatment_controls_fail_closed_when_mixed_in_one_route() -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["campaign_id"] = "eed-treatment-matrix-b1"
    spec["world"]["overlays"] = ["eed_resource_impact"]
    spec["parameters"].update(
        {
            "genome_upgrade_v2.profile": "genome_upgrade_v2_fixed_q10_v1",
            "genome_upgrade_v2.enable": True,
            "genome_upgrade_v2.mutation_rate_q10": 20,
            "genome_upgrade_v2.mutation_step_q10": 64,
        }
    )
    spec["matrix"]["dimensions"] = [
        {
            "id": "impact_apply",
            "description": "Reciprocal EED impact control.",
            "values": [False, True],
        },
        {
            "id": "ecology_frozen",
            "description": "Frozen-stock EED control.",
            "values": [False, True],
        },
        {
            "id": "phenotype_fixed",
            "description": "Fixed-phenotype EED control.",
            "values": [False, True],
        },
        {
            "id": "founder_profile",
            "description": "Resource-axis founder profile.",
            "values": ["a_high", "b_high", "symmetric_mix"],
        },
        {"id": "ecology_seed", "values": [9741]},
        {"id": "task_seed", "values": [29741]},
    ]
    spec["matrix"]["parameter_bindings"] = {
        "impact_apply": "eed.resource_impact_apply_enable",
        "ecology_frozen": "eed.ecology_frozen",
        "phenotype_fixed": "eed.phenotype_fixed",
        "founder_profile": "eed.founder_resource_axis_profile",
    }
    spec["matrix"]["seed_bindings"] = {
        "ecology_seed": "ecology",
        "task_seed": "task",
    }
    spec["matrix"]["include"] = [
        {
            "impact_apply": False,
            "ecology_frozen": False,
            "phenotype_fixed": False,
        },
        {
            "impact_apply": True,
            "ecology_frozen": False,
            "phenotype_fixed": False,
        },
        {
            "impact_apply": True,
            "ecology_frozen": True,
            "phenotype_fixed": False,
        },
        {
            "impact_apply": True,
            "ecology_frozen": False,
            "phenotype_fixed": True,
        },
    ]
    spec["matrix"]["expected_cells"] = 12
    spec["matrix"]["max_cells"] = 24
    spec["execution"]["batch"] = {"mode": "bounded", "maximum": 1}
    spec["execution"]["technical_gate"] = {
        "selectors": [
            {
                "impact_apply": True,
                "ecology_frozen": False,
                "phenotype_fixed": False,
                "founder_profile": "a_high",
                "ecology_seed": 9741,
                "task_seed": 29741,
            }
        ],
        "partition": False,
    }
    spec["analysis"]["required_metrics"] = [
        "population.alive",
        "eed.resource_a_stock_total",
        "eed.resource_b_stock_total",
        "eed.resource_axis_mean",
    ]
    spec["analysis"]["rules"] = []

    with pytest.raises(
        CampaignError,
        match="qualification-affecting per-cell parameter .* varies",
    ):
        compile_campaign(spec)


def test_eed_uniform_treatment_controls_co_reside_across_founders() -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["campaign_id"] = "eed-impact-on-founders-b1"
    spec["world"]["overlays"] = ["eed_resource_impact"]
    spec["parameters"].update(
        {
            "genome_upgrade_v2.profile": "genome_upgrade_v2_fixed_q10_v1",
            "genome_upgrade_v2.enable": True,
            "genome_upgrade_v2.mutation_rate_q10": 20,
            "genome_upgrade_v2.mutation_step_q10": 64,
        }
    )
    spec["matrix"]["dimensions"] = [
        {
            "id": "founder_profile",
            "description": "Resource-axis founder profile.",
            "values": ["a_high", "b_high", "symmetric_mix"],
        },
        {"id": "ecology_seed", "values": [9741]},
        {"id": "task_seed", "values": [29741]},
    ]
    spec["matrix"]["parameter_bindings"] = {
        "founder_profile": "eed.founder_resource_axis_profile",
    }
    spec["matrix"]["seed_bindings"] = {
        "ecology_seed": "ecology",
        "task_seed": "task",
    }
    spec["matrix"]["expected_cells"] = 3
    spec["matrix"]["max_cells"] = 3
    spec["parameters"].update(
        {
            "eed.resource_impact_apply_enable": True,
            "eed.ecology_frozen": False,
            "eed.phenotype_fixed": False,
        }
    )
    spec["execution"]["batch"] = {"mode": "bounded", "maximum": 3}
    spec["execution"]["technical_gate"] = {
        "selectors": [
            {
                "founder_profile": "a_high",
                "ecology_seed": 9741,
                "task_seed": 29741,
            }
        ],
        "partition": False,
    }
    spec["analysis"]["required_metrics"] = [
        "population.alive",
        "eed.resource_a_stock_total",
        "eed.resource_b_stock_total",
        "eed.resource_axis_mean",
    ]
    spec["analysis"]["rules"] = []

    impact_on = compile_campaign(spec)
    impact_off_spec = copy.deepcopy(spec)
    impact_off_spec["parameters"]["eed.resource_impact_apply_enable"] = False
    impact_off = compile_campaign(impact_off_spec)

    assert impact_on.profile_id == "eiel-production-runtime-v1"
    assert len(impact_on.cells) == 3
    assert [len(batch.cell_ids) for batch in impact_on.batches] == [3]
    assert {
        cell.parameters["eed.resource_impact_apply_enable"] for cell in impact_on.cells
    } == {True}
    assert {
        cell.parameters["eed.founder_resource_axis_profile"] for cell in impact_on.cells
    } == {"a_high", "b_high", "symmetric_mix"}
    fixed = impact_on.qualification_key["fixed_invariants"]
    assert fixed["parameter.eed.resource_impact_apply_enable"] is True
    assert fixed["parameter.eed.ecology_frozen"] is False
    assert fixed["parameter.eed.phenotype_fixed"] is False
    assert "parameter.eed.founder_resource_axis_profile" not in fixed
    assert impact_on.qualification_key_sha256 != impact_off.qualification_key_sha256
    profile = next(
        row
        for row in default_profile_registry().profiles()
        if row.profile_id == impact_on.profile_id
    )
    for name in (
        "eed.resource_impact_apply_enable",
        "eed.ecology_frozen",
        "eed.phenotype_fixed",
    ):
        contract = profile.parameter_contracts[name]
        assert contract.per_cell is True
        assert contract.qualification_affecting is True
    founder_contract = profile.parameter_contracts["eed.founder_resource_axis_profile"]
    assert founder_contract.per_cell is True
    assert founder_contract.qualification_affecting is False


@pytest.mark.parametrize(
    ("spec_name", "route_parameters"),
    [
        (
            "eed_b48_technical_impact_off_w32p256c512g20t160gpu.yaml",
            {
                "eed.resource_impact_apply_enable": False,
                "eed.ecology_frozen": False,
                "eed.phenotype_fixed": False,
            },
        ),
        (
            "eed_b48_technical_impact_on_w32p256c512g20t160gpu.yaml",
            {
                "eed.resource_impact_apply_enable": True,
                "eed.ecology_frozen": False,
                "eed.phenotype_fixed": False,
            },
        ),
        (
            "eed_b48_technical_ecology_frozen_w32p256c512g20t160gpu.yaml",
            {
                "eed.resource_impact_apply_enable": True,
                "eed.ecology_frozen": True,
                "eed.phenotype_fixed": False,
            },
        ),
        (
            "eed_b48_technical_phenotype_fixed_w32p256c512g20t160gpu.yaml",
            {
                "eed.resource_impact_apply_enable": True,
                "eed.ecology_frozen": False,
                "eed.phenotype_fixed": True,
            },
        ),
    ],
)
def test_eed_b48_technical_specs_are_true_physical_b48(
    spec_name: str,
    route_parameters: dict[str, bool],
) -> None:
    plan = compile_campaign(_load(spec_name))

    assert plan.generations == 20
    assert plan.ticks_per_generation == 160
    assert plan.checkpoints == (10, 20)
    assert len(plan.cells) == 48
    assert [len(batch.cell_ids) for batch in plan.batches] == [48]
    assert plan.resource_estimate["physical_batch_cells"] == 48
    fixed = plan.qualification_key["fixed_invariants"]
    for name, expected in route_parameters.items():
        assert {cell.parameters[name] for cell in plan.cells} == {expected}
        assert fixed[f"parameter.{name}"] is expected
    assert {
        cell.parameters["eed.founder_resource_axis_profile"] for cell in plan.cells
    } == {"a_high", "b_high", "symmetric_mix"}
    assert "parameter.eed.founder_resource_axis_profile" not in fixed


def test_eed_b24_technical_rehearsal_is_one_standalone_resident_batch() -> None:
    spec = _load("eed_b24_technical_phenotype_fixed_w32p256c512g20t160gpu.yaml")
    plan = compile_campaign(spec)

    assert plan.campaign_id == "eed-b24-technical-phenotype-fixed-w32p256c512g20t160gpu"
    assert plan.generations == 20
    assert plan.ticks_per_generation == 160
    assert len(plan.cells) == 24
    assert [len(batch.cell_ids) for batch in plan.batches] == [24]
    assert plan.resource_estimate["physical_batch_cells"] == 24
    assert "not a B48 partition" in spec["no_claims"]
    assert "eed_b24_route_qualification" in spec["output"]["runtime_root"]


def test_uniform_groups_are_chunked_by_requested_maximum() -> None:
    spec = _load("wdl_gru_g37_irregular.yaml")
    spec["execution"]["batch"] = {"mode": "explicit", "cells": 4}

    plan = compile_campaign(spec)

    assert [len(batch.cell_ids) for batch in plan.batches] == [4, 3, 4, 3, 4, 3]
    assert plan.resource_estimate["physical_batch_cells"] == 4


def test_uniform_batch_values_change_plan_not_qualification_identity() -> None:
    base = _load("wdl_gru_g37_irregular.yaml")
    changed = copy.deepcopy(base)
    changed["matrix"]["dimensions"][2]["values"] = [0.05, 0.15, 0.2]

    left = compile_campaign(base)
    right = compile_campaign(changed)

    assert left.qualification_key_sha256 == right.qualification_key_sha256
    assert left.plan_sha256 != right.plan_sha256


def test_genome_v2_e1_arm_label_is_nonmechanical_per_cell_provenance() -> None:
    base = _load("patch_fixed_g14_irregular.yaml")
    e1_metrics = {
        "population.terminal_alive",
        "population.successful_births_cumulative",
        "genome_v2.expression_observation_count",
        "genome_v2.context_observation_count",
        "genome_v2.cost_observation_count",
        "genome_v2.cap_observation_count",
    }
    base["parameters"].update(
        {
            "genome_upgrade_v2.profile": "genome_upgrade_v2_fixed_q10_v1",
            "genome_upgrade_v2.enable": True,
            "genome_upgrade_v2.mutation_rate_q10": 20,
            "genome_upgrade_v2.mutation_step_q10": 64,
            "genome_upgrade_v2.founder_profile": "neutral",
            "genome_upgrade_v2.e1_arm_label": "control",
        }
    )
    base["analysis"]["required_metrics"] = sorted(e1_metrics)
    base["analysis"]["rules"] = []
    changed = copy.deepcopy(base)
    changed["parameters"]["genome_upgrade_v2.e1_arm_label"] = "treatment"

    control = compile_campaign(base)
    treatment = compile_campaign(changed)

    assert control.qualification_key_sha256 == treatment.qualification_key_sha256
    assert control.plan_sha256 != treatment.plan_sha256
    assert (
        "parameter.genome_upgrade_v2.e1_arm_label"
        not in control.qualification_key["fixed_invariants"]
    )
    assert (
        "parameter.genome_upgrade_v2.founder_profile"
        not in control.qualification_key["fixed_invariants"]
    )
    profile = next(
        row
        for row in default_profile_registry().profiles()
        if row.profile_id == control.profile_id
    )
    contract = profile.parameter_contracts["genome_upgrade_v2.e1_arm_label"]
    assert contract.per_cell is True
    assert contract.qualification_affecting is False
    founder_contract = profile.parameter_contracts["genome_upgrade_v2.founder_profile"]
    assert founder_contract.per_cell is True
    assert founder_contract.qualification_affecting is False
    assert e1_metrics <= profile.metric_names
    assert {family.family_id for family in profile.metric_families} >= {
        "population_v1",
        "genome_upgrade_v2_observation_v1",
    }


@pytest.mark.parametrize("generations", [13, 14, 37, 73, 100, 128, 129, 200])
def test_generation_values_are_not_a_qualification_whitelist(generations: int) -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["execution"]["generations"] = generations
    spec["checkpoints"]["milestones"] = [generations]
    spec["comparisons"]["continuation_splits"] = []

    plan = compile_campaign(spec)

    assert plan.generations == generations


def test_campaign_values_do_not_change_qualification_identity() -> None:
    base = _load("patch_fixed_g14_irregular.yaml")
    changed = copy.deepcopy(base)
    changed["execution"]["generations"] = 37
    changed["execution"]["batch"]["maximum"] = 7
    changed["checkpoints"]["milestones"] = [7, 19, 37]
    changed["comparisons"]["continuation_splits"] = [7, 19]
    changed["matrix"]["dimensions"][0]["values"] = [111, 222, 333]
    changed["matrix"]["expected_cells"] = 9

    left = compile_campaign(base)
    right = compile_campaign(changed)

    assert left.qualification_key_sha256 == right.qualification_key_sha256
    assert left.plan_sha256 != right.plan_sha256


def test_generation_zero_checkpoint_is_compiled_as_a_campaign_value() -> None:
    base = _load("patch_fixed_g14_irregular.yaml")
    changed = copy.deepcopy(base)
    changed["checkpoints"]["milestones"] = [
        0,
        *base["checkpoints"]["milestones"],
    ]

    original = compile_campaign(base)
    with_generation_zero = compile_campaign(changed)

    assert with_generation_zero.checkpoints[0] == 0
    assert (
        with_generation_zero.qualification_key_sha256
        == original.qualification_key_sha256
    )
    assert with_generation_zero.plan_sha256 != original.plan_sha256


def test_b72g100_plans_without_campaign_python() -> None:
    plan = compile_campaign(_load("patch_fixed_b72g100.yaml"))

    assert len(plan.cells) == 72
    assert plan.generations == 100
    assert len(plan.batches) == 1
    assert len(plan.batches[0].cell_ids) == 72
    assert tuple(
        cell_id for batch in plan.batches for cell_id in batch.cell_ids
    ) == tuple(cell.cell_id for cell in plan.cells)


def test_explicit_batch_preserves_planning_request_for_fail_closed_preflight() -> None:
    spec = _load("patch_fixed_b72g100.yaml")
    spec["execution"]["batch"] = {"mode": "explicit", "cells": 72}

    plan = compile_campaign(spec)

    assert len(plan.batches) == 1
    assert len(plan.batches[0].cell_ids) == 72


def test_patch_fixed_mlp_admits_the_audited_b72_batch_range() -> None:
    plan = compile_campaign(_load("patch_fixed_g14_irregular.yaml"))

    assert plan.profile_id == "eiel-production-runtime-v1"
    assert plan.qualification_key["admitted_batch_range"] == [1, 72]


def test_wdl_fixed_mlp_b48_enters_multicell_route_qualification() -> None:
    plan = compile_campaign(_load("gv2_b48_mutation_persistence_factorial_g30.yaml"))
    repeated = compile_campaign(
        _load("gv2_b48_mutation_persistence_factorial_g30.yaml")
    )

    assert plan.profile_id == "eiel-production-runtime-v1"
    assert plan.qualification_key["admitted_batch_range"] == [1, 72]
    assert len(plan.cells) == 48
    assert [cell.cell_id for cell in plan.cells] == [
        cell.cell_id for cell in repeated.cells
    ]
    assert len({cell.cell_id for cell in plan.cells}) == 48
    assert sorted(cell.ordinal for cell in plan.cells) == list(range(48))
    assert [len(batch.cell_ids) for batch in plan.batches] == [24, 24]
    assert tuple(
        cell_id for batch in plan.batches for cell_id in batch.cell_ids
    ) == tuple(cell.cell_id for cell in plan.cells)
    assert len({cell.dimensions["ecology_seed"] for cell in plan.cells}) == 6
    for ecology_seed in {cell.dimensions["ecology_seed"] for cell in plan.cells}:
        paired = [
            cell
            for cell in plan.cells
            if cell.dimensions["ecology_seed"] == ecology_seed
        ]
        assert len(paired) == 8
        assert {
            (
                cell.dimensions["founder_profile"],
                cell.dimensions["mutation_rate_q10"],
                cell.dimensions["mutation_step_q10"],
            )
            for cell in paired
        } == {
            ("neutral", 0, 0),
            ("neutral", 16, 16),
            (
                "recomb72_gv2_v1.single_locus_extreme_pair.all_heterozygous_ab",
                0,
                0,
            ),
            (
                "recomb72_gv2_v1.single_locus_extreme_pair.all_heterozygous_ab",
                16,
                16,
            ),
            (
                "recomb72_gv2_v1.single_locus_extreme_pair.balanced_homozygous_aa_bb",
                0,
                0,
            ),
            (
                "recomb72_gv2_v1.single_locus_extreme_pair.balanced_homozygous_aa_bb",
                16,
                16,
            ),
            (
                "recomb72_gv2_v1.single_locus_extreme_pair.homozygous_aa_control",
                0,
                0,
            ),
            (
                "recomb72_gv2_v1.single_locus_extreme_pair.homozygous_aa_control",
                16,
                16,
            ),
        }
    assert plan.technical_gate["required"] is True
    assert plan.technical_gate["execution_mode"] == "pre_primary_canary"
    assert plan.resource_estimate["physical_batch_cells"] == 24


def test_w32_patch_batch_specs_bind_requested_physical_batches() -> None:
    b24 = compile_campaign(
        _load("patch_fixed_structural_scale_w32p256c512_b24t80.yaml")
    )
    b48 = compile_campaign(
        _load("patch_fixed_structural_scale_w32p256c512_b48t160.yaml")
    )

    assert b24.qualification_key_sha256 == b48.qualification_key_sha256
    assert b24.qualification_key["admitted_batch_range"] == [1, 72]
    assert b48.qualification_key["admitted_batch_range"] == [1, 72]
    assert len(b24.cells) == 24
    assert len(b48.cells) == 48
    assert [len(batch.cell_ids) for batch in b24.batches] == [24]
    assert [len(batch.cell_ids) for batch in b48.batches] == [48]
    assert b24.ticks_per_generation == 80
    assert b48.ticks_per_generation == 160
    assert b24.resource_estimate["physical_batch_cells"] == 24
    assert b48.resource_estimate["physical_batch_cells"] == 48


def test_w32_c256_device_specs_bind_target_nomenclature_and_g1_gpu_route() -> None:
    cpu = compile_campaign(
        _load("patch_fixed_structural_scale_b1w32p256c256g200t80cpu.yaml")
    )
    b24 = compile_campaign(
        _load("patch_fixed_structural_scale_b24w32p256c256g200t80gpu.yaml")
    )
    b48 = compile_campaign(
        _load("patch_fixed_structural_scale_b48w32p256c256g200t80gpu.yaml")
    )

    assert cpu.campaign_id.endswith("b1w32p256c256g200t80cpu")
    assert b24.campaign_id.endswith("b24w32p256c256g200t80gpu")
    assert b48.campaign_id.endswith("b48w32p256c256g200t80gpu")
    assert cpu.profile_id == "eiel-production-runtime-v1-cpu-native"
    assert b24.profile_id == b48.profile_id == "eiel-production-runtime-v1"
    assert cpu.qualification_key["fixed_invariants"]["parameter.patch.capacity"] == 256
    assert b24.qualification_key_sha256 == b48.qualification_key_sha256
    assert len(cpu.cells) == 1
    assert len(b24.cells) == 24
    assert len(b48.cells) == 48
    assert [len(batch.cell_ids) for batch in cpu.batches] == [1]
    assert [len(batch.cell_ids) for batch in b24.batches] == [24]
    assert [len(batch.cell_ids) for batch in b48.batches] == [48]
    assert cpu.ticks_per_generation == b24.ticks_per_generation == 80
    assert b48.ticks_per_generation == 80
    assert "physical_window_generations" not in cpu.payload()["execution"]
    assert b24.payload()["execution"]["physical_window_generations"] == 1
    assert b48.payload()["execution"]["physical_window_generations"] == 1
    assert b24.resource_estimate["physical_batch_cells"] == 24
    assert b48.resource_estimate["physical_batch_cells"] == 48


def test_all_cuda_configurations_use_the_same_runtime_profile() -> None:
    paths = (
        "patch_fixed_g14_irregular.yaml",
        "patch_gru_b1g14_qualification.yaml",
        "wdl_gru_b1g14_qualification.yaml",
        "jungle_fixed_mlp_b1g8_qualification.yaml",
        "gmnw_gru_b3g9_qualification.yaml",
    )

    assert {compile_campaign(_load(path)).profile_id for path in paths} == {
        "eiel-production-runtime-v1"
    }


def test_jungle_admission_is_mechanics_configuration() -> None:
    plan = compile_campaign(_load("jungle_fixed_mlp_b1g8_qualification.yaml"))

    assert plan.profile_id == "eiel-production-runtime-v1"
    assert plan.qualification_key["admitted_batch_range"] == [1, 1]


def test_planning_batch_ceiling_is_not_part_of_qualification_identity() -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    source = next(
        row
        for row in default_profile_registry().profiles()
        if row.profile_id == "eiel-production-runtime-v1"
    )
    registry = ProfileRegistryV1()
    registry.register(replace(source, batch_max=70000))

    original = compile_campaign(spec)
    changed = compile_campaign(spec, registry=registry)

    assert original.qualification_key_sha256 == changed.qualification_key_sha256
    assert changed.qualification_key["admitted_batch_range"] == [1, 72]


def test_b72g200_is_configuration_only() -> None:
    plan = compile_campaign(_load("patch_fixed_b72g200.yaml"))

    assert len(plan.cells) == 72
    assert plan.generations == 200


def test_ambiguous_or_unbound_dimensions_fail_closed() -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["matrix"]["seed_bindings"]["founder_mode"] = "founder"

    with pytest.raises(CampaignError, match="ambiguous ownership"):
        compile_campaign(spec)


def test_duplicate_seed_owner_fails_closed() -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["matrix"]["seed_bindings"]["task_seed"] = "ecology"

    with pytest.raises(CampaignError, match="ambiguous duplicate owners"):
        compile_campaign(spec)


def test_unknown_campaign_fields_fail_closed() -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    spec["magic"] = True

    with pytest.raises(CampaignError, match="unsupported fields"):
        compile_campaign(spec)


def test_dynamic_batch_extent_is_not_qualification_identity() -> None:
    spec = _load("patch_fixed_g14_irregular.yaml")
    eleven = compile_campaign(spec)
    spec["execution"]["batch"]["maximum"] = 7
    seven = compile_campaign(spec)

    assert eleven.qualification_key["admitted_batch_range"] == [1, 72]
    assert seven.qualification_key_sha256 == eleven.qualification_key_sha256
    assert [len(batch.cell_ids) for batch in eleven.batches] == [2] * 10 + [1]
    assert [len(batch.cell_ids) for batch in seven.batches] == [7, 7, 7]
