from __future__ import annotations

from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from evolution.diploid_inheritance_v1 import DIPLOID_INHERITANCE_PROFILE
from evolution.genome_upgrade_v2 import (
    MEIOSIS_DRAW_COUNT,
    MUTATION_SEMANTIC_DRAW_COUNT,
    GenomeUpgradeV2V1,
    attach_genome_upgrade_v2,
    genome_upgrade_v2_from_genome,
)
from evolution.reproduction import crossover_genome
from engine.tensor_world.c5_rngv2_fullroute import (
    DOMAIN_BIRTH_PROPOSAL,
    DOMAIN_GV2_DELTA,
    action_gumbel_uniforms_v2,
    assemble_child_from_v2_v1,
    assemble_diploid_child_from_v2_v1,
    domain_ids_unique,
    draw_gv2_inheritance_tape_v2,
    food_spawn_uniform_v2,
    jungle_hit_uniform_v2,
)
from engine.tensor_world.c5_rngv2_route import V1_ROUTE_ID, resolve_rng_route
from engine.tw_rngv2.address import DOMAIN_BY_ID
from engine.tw_rngv2.status import Status
from simulation.rng_manager import RNGManager


def test_default_remains_v1() -> None:
    assert resolve_rng_route()["route_id"] == V1_ROUTE_ID


def test_new_domains_known_and_unique() -> None:
    assert domain_ids_unique()
    assert 0x0340 in DOMAIN_BY_ID
    assert 0x0341 in DOMAIN_BY_ID
    assert 0x0342 in DOMAIN_BY_ID
    assert 0x0360 in DOMAIN_BY_ID
    assert 0x0330 in DOMAIN_BY_ID
    assert 0x0310 in DOMAIN_BY_ID
    assert DOMAIN_BIRTH_PROPOSAL == 0x0310


def test_gv2_v2_tape_is_address_stable() -> None:
    a = draw_gv2_inheritance_tape_v2(
        world_uid=0, absolute_tick=4, subject_uid=1, mutation_rate_q10=20, mutation_step_q10=64
    )
    b = draw_gv2_inheritance_tape_v2(
        world_uid=0, absolute_tick=4, subject_uid=1, mutation_rate_q10=20, mutation_step_q10=64
    )
    assert a.status is Status.RNG_OK
    assert a.donor_homologs == b.donor_homologs
    assert a.mutation_delta_q10 == b.mutation_delta_q10
    other = draw_gv2_inheritance_tape_v2(
        world_uid=0, absolute_tick=4, subject_uid=2, mutation_rate_q10=20, mutation_step_q10=64
    )
    assert other.donor_homologs != a.donor_homologs or other.mutation_delta_q10 != a.mutation_delta_q10


def test_gv2_v2_assembly_no_host_tape() -> None:
    parent = GenomeUpgradeV2V1.neutral()
    child, receipt = assemble_child_from_v2_v1(
        parent,
        parent,
        world_uid=1,
        absolute_tick=3,
        subject_uid=7,
        mutation_rate_q10=20,
        mutation_step_q10=64,
    )
    assert receipt["host_tape"] is False
    assert receipt["rng_semantics_version"] == 2
    assert receipt["domains"]["gv2_delta"] == DOMAIN_GV2_DELTA
    replay, replay_receipt = assemble_child_from_v2_v1(
        parent,
        parent,
        world_uid=1,
        absolute_tick=3,
        subject_uid=7,
        mutation_rate_q10=20,
        mutation_step_q10=64,
    )
    assert child.to_payload()["genotype_sha256"] == replay.to_payload()["genotype_sha256"]
    assert replay_receipt["child_genotype_sha256"] == receipt["child_genotype_sha256"]


def _plain_genome() -> Genome:
    return Genome(
        hidden_size=64,
        exploration_bias=0.0,
        aggression_bias=0.0,
        reproduction_threshold=30.0,
        mutation_rate=0.02,
        action_temperature=1.0,
        use_gru=False,
        memory_dim=0,
        drive_inertia=0.8,
        drive_noise=0.01,
        drive_logit_gain=1.0,
        pred_lr=0.08,
        developmental_program=canonicalize_developmental_program(None),
    )


def test_diploid_v2_assembly_is_address_stable() -> None:
    parent = _plain_genome()
    child_a, phenotype_a, receipt_a = assemble_diploid_child_from_v2_v1(
        parent,
        parent,
        world_uid=2,
        absolute_tick=5,
        subject_uid=11,
        mutation_enabled=True,
        resource_handling_enabled=False,
    )
    child_b, phenotype_b, receipt_b = assemble_diploid_child_from_v2_v1(
        parent,
        parent,
        world_uid=2,
        absolute_tick=5,
        subject_uid=11,
        mutation_enabled=True,
        resource_handling_enabled=False,
    )
    assert receipt_a["host_tape"] is False
    assert receipt_a["meiosis"]["domain"] == DOMAIN_BIRTH_PROPOSAL
    assert child_a.to_payload()["genotype_sha256"] == child_b.to_payload()["genotype_sha256"]
    assert phenotype_a == phenotype_b
    _other, _ph, other_receipt = assemble_diploid_child_from_v2_v1(
        parent,
        parent,
        world_uid=2,
        absolute_tick=5,
        subject_uid=12,
        mutation_enabled=True,
        resource_handling_enabled=False,
    )
    assert other_receipt["meiosis"]["donors"] != receipt_a["meiosis"]["donors"]
    assert receipt_b["rng_semantics_version"] == 2


def test_crossover_dispatches_v2_without_host_tape() -> None:
    parent_a = _plain_genome()
    parent_b = _plain_genome()
    parent_b.aggression_bias = 0.1
    parent_b.refresh_canonical_payload_cache()
    attach_genome_upgrade_v2(parent_a, GenomeUpgradeV2V1.neutral())
    attach_genome_upgrade_v2(parent_b, GenomeUpgradeV2V1.neutral())
    manager = RNGManager(global_seed=4242)
    cfg = type(
        "Cfg",
        (),
        {
            "inheritance_profile": DIPLOID_INHERITANCE_PROFILE,
            "genome_c1_enable": False,
            "genome_c2_enable": False,
            "resource_handling_representation_enable": False,
            "diploid_mutation_enable": True,
            "genome_upgrade_v2_enable": True,
            "genome_upgrade_v2_mutation_rate_q10": 20,
            "genome_upgrade_v2_mutation_step_q10": 64,
            "rng_semantics_version": 2,
            "world_uid": 9,
        },
    )()
    child = crossover_genome(
        parent_a,
        parent_b,
        manager.mutation_py,
        rng_manager=manager,
        sim_cfg=cfg,
        structural_identity_context={"tick": 4, "child_agent_id": 17},
    )
    receipt = child._genome_upgrade_v2_inheritance_receipt_v1
    diploid_receipt = child._diploid_inheritance_receipt_v1
    assert receipt["rng_semantics_version"] == 2
    assert receipt["host_tape"] is False
    assert diploid_receipt["rng_semantics_version"] == 2
    assert diploid_receipt["host_tape"] is False
    assert receipt["total_semantic_draws"] == (
        MEIOSIS_DRAW_COUNT + MUTATION_SEMANTIC_DRAW_COUNT
    )
    expected, expected_receipt = assemble_child_from_v2_v1(
        GenomeUpgradeV2V1.neutral(),
        GenomeUpgradeV2V1.neutral(),
        world_uid=9,
        absolute_tick=4,
        subject_uid=17,
        mutation_rate_q10=20,
        mutation_step_q10=64,
    )
    value = genome_upgrade_v2_from_genome(child)
    assert value is not None
    assert value.to_payload()["genotype_sha256"] == expected.to_payload()["genotype_sha256"]
    assert receipt["child_genotype_sha256"] == expected_receipt["child_genotype_sha256"]
    replay_manager = RNGManager(global_seed=4242)
    replay = crossover_genome(
        parent_a,
        parent_b,
        replay_manager.mutation_py,
        rng_manager=replay_manager,
        sim_cfg=cfg,
        structural_identity_context={"tick": 4, "child_agent_id": 17},
    )
    replay_value = genome_upgrade_v2_from_genome(replay)
    assert replay_value is not None
    assert replay_value.to_payload()["genotype_sha256"] == expected.to_payload()["genotype_sha256"]


def test_crossover_v2_fails_closed_without_child_identity() -> None:
    parent = _plain_genome()
    attach_genome_upgrade_v2(parent, GenomeUpgradeV2V1.neutral())
    manager = RNGManager(global_seed=7)
    cfg = type(
        "Cfg",
        (),
        {
            "inheritance_profile": DIPLOID_INHERITANCE_PROFILE,
            "genome_c1_enable": False,
            "genome_c2_enable": False,
            "resource_handling_representation_enable": False,
            "diploid_mutation_enable": False,
            "genome_upgrade_v2_enable": True,
            "rng_semantics_version": 2,
        },
    )()
    try:
        crossover_genome(
            parent,
            parent,
            manager.mutation_py,
            rng_manager=manager,
            sim_cfg=cfg,
        )
    except RuntimeError as exc:
        assert "GENOME_UPGRADE_V2_RNGV2_REQUIRES_CHILD_IDENTITY" in str(exc)
    else:
        raise AssertionError("expected fail-closed V2 identity error")


def test_source_builder_v2_elides_giant_host_tapes() -> None:
    from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
        birth_state_capacity_overrides_v1,
    )

    overrides = birth_state_capacity_overrides_v1(
        type(
            "Cfg",
            (),
            {
                "rng_semantics_version": 2,
                "cuda_architecture_b_cpython_gauss_segment_capacity": 0,
                "policy_weight_mutation_enable": False,
                "policy_weight_mutation_std": 0.0,
                "policy_weight_mutation_rate": 0.0,
                "generation_turnover_enabled": True,
            },
        )(),
        ordinary_births_enabled=True,
        resident_policy_episode_capacity=736,
        turnover_refill_capacity=256,
        max_births_per_tick=3,
    )
    assert overrides["scalar_gauss_raw_capacity"] == 0
    assert overrides["resident_policy_episode_capacity"] == 736
    assert overrides["compact_policy_mutation_tape"] is True
    assert overrides["policy_birth_segment_refill_enabled"] is False


def test_wdl_tick_core_admits_zero_gauss_tape_on_v2() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"
    )
    text = source.read_text(encoding="utf-8")
    assert "rng_semantics_version != 2" in text
    assert "gauss_tape_capacity <= 0" in text
    marker = text.find("rng_semantics_version != 2")
    nearby = text[marker : marker + 240]
    assert "gauss_tape_capacity <= 0" in nearby


def test_wdl_birth_prepare_splits_status_112_conditions() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"
    )
    text = source.read_text(encoding="utf-8")
    for code in ("113", "114", "115", "119", "121", "122"):
        assert f"EIEL_RUNTIME_BIRTH_PREPARE_FAIL({code})" in text
    # Controller 112 on r12 is 90 + commit 22, not a restored prepare catch-all.
    prepare = text.split("prepare_flattened_runtime_birth_transaction_v2", 1)[1]
    prepare = prepare.split("flattened_controller_birth_prepare_kernel_v1", 1)[0]
    assert "EIEL_RUNTIME_BIRTH_PREPARE_FAIL(112)" not in prepare


def test_q3_tiny_wdl_envelope_is_config_valid() -> None:
    import math

    from engine.campaigns.patch_runner_factory import _configure_wdl_overlay_v1
    from engine.env.env_config import EnvConfig
    from engine.simulation.runner_pkg.runner_config import SimConfig

    env = EnvConfig()
    env.width = 16
    env.height = 16
    sim = SimConfig()
    _configure_wdl_overlay_v1(
        env,
        sim,
        {
            "wdl.delay_ticks": 12,
            "wdl.cue_duration_ticks": 1,
            "wdl.active_duration_ticks": 1,
            "wdl.intensity": 0.1,
            "wdl.cue_visible_at_decision": False,
        },
        policy_family="fixed_mlp",
    )
    delay = int(env.wdl_delayed_hazard_delay_steps)
    cue = int(env.wdl_delayed_hazard_cue_duration_steps)
    active = int(env.wdl_delayed_hazard_active_duration_steps)
    period = int(env.wdl_delayed_hazard_event_period_steps)
    intensity = float(env.wdl_delayed_hazard_intensity)
    dropout = float(env.wdl_delayed_hazard_observation_dropout_rate)
    flags = (
        int(bool(env.wdl_delayed_hazard_enable)),
        int(bool(env.wdl_delayed_hazard_cue_enable)),
        int(bool(env.wdl_delayed_hazard_observation_enable)),
        int(bool(env.wdl_delayed_hazard_damage_enable)),
        int(bool(env.wdl_delayed_hazard_deterministic_schedule_enable)),
    )
    # Same predicates as persistent_wdl::config_valid / native validate_config.
    assert int(env.wdl_delayed_hazard_schema_version) == 1
    assert env.height > 0 and env.width > 0
    assert delay >= 1
    assert cue >= 1 and cue <= delay
    assert active >= 1
    assert period >= delay + active + 1
    assert int(env.wdl_delayed_hazard_radius) >= 0
    assert math.isfinite(intensity) and 0.0 <= intensity <= 1.0
    assert math.isfinite(dropout) and 0.0 <= dropout <= 1.0
    assert all(flag in (0, 1) for flag in flags)
    assert delay == 12 and cue == 1 and active == 1 and period == 20


def test_window_controller_skips_zero_capacity_gauss_refill() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_architecture_b_resident_window_controller_v1.hpp"
    )
    text = source.read_text(encoding="utf-8")
    marker = text.find("const auto refill_gaussian_window")
    assert marker >= 0
    nearby = text[marker : marker + 900]
    assert "primary_tape.size(1) == 0" in nearby
    assert "v2_tape_free_zero_gauss_capacity" in nearby
    assert "refill_cpython_gaussian_window_v1(" in nearby


def test_v2_diploid_prepare_records_scalar_committed_count() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"
    )
    text = source.read_text(encoding="utf-8")
    marker = "if (rngv2_birth) {"
    diploid = text.split("if (diploid_mode) {", 1)[1]
    diploid = diploid.split("for (int64_t birth = 0; birth < birth_count; ++birth) {", 2)[2]
    assert marker in diploid
    nearby = diploid[diploid.find(marker) : diploid.find(marker) + 420]
    assert "scalar_committed_count = birth_count" in nearby
    assert "scalar_status = 0" in nearby


def test_world_action_oracles_are_addressed() -> None:
    u = action_gumbel_uniforms_v2(world_uid=0, absolute_tick=1, subject_uid=3, action_count=7)
    assert len(u) == 7
    assert u == action_gumbel_uniforms_v2(world_uid=0, absolute_tick=1, subject_uid=3, action_count=7)
    assert u != action_gumbel_uniforms_v2(world_uid=0, absolute_tick=1, subject_uid=4, action_count=7)
    spawn = food_spawn_uniform_v2(world_uid=0, absolute_tick=2, attempt=0)
    assert 0.0 <= spawn < 1.0
    hit = jungle_hit_uniform_v2(world_uid=0, absolute_tick=2, subject_uid=3)
    assert hit != spawn


def test_high_b_world_uids_do_not_alias_v2_draws() -> None:
    draws = [
        food_spawn_uniform_v2(world_uid=world, absolute_tick=7, attempt=0)
        for world in (1, 48, 144, 145)
    ]
    assert len(set(draws)) == 4
    assert food_spawn_uniform_v2(world_uid=144, absolute_tick=7, attempt=0) == draws[2]
