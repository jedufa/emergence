from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from engine.perf import cuda_architecture_b_direct_resident_settlement_v1 as settlement
from perf.cuda_architecture_b_boundary import EVENT_COLUMN_NAMES
from engine.perf.cuda_architecture_b_checkpoint import CudaBoundaryCheckpoint
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    PreparedDirectResidentTrajectoryV1,
)
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    RESULT_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_causal_cuda_output_v1 import (
    RECORD_CAPACITY,
    RECORD_WIDTH,
    UNAVAILABLE_VALUE,
)
from perf.cuda_genome_upgrade_v2_state_bank import (
    CudaGenomeUpgradeV2StateBank,
    GENOME_V2_FOUNDER_PROFILE_SCHEMA,
    resolve_genome_v2_founder_profile_v1,
    restore_cuda_genome_upgrade_v2_state_bank,
)


@dataclass(frozen=True)
class _BirthState:
    identity_state: dict[str, torch.Tensor]
    scalar_rng_state: dict[str, torch.Tensor]
    genome_state: dict[str, torch.Tensor]
    policy_mem_dim: torch.Tensor
    policy_use_gru: torch.Tensor
    drive_state: dict[str, torch.Tensor]
    policy_state: dict[str, torch.Tensor]
    turnover_policy_state: dict[str, torch.Tensor]
    planner_pcg64_state_words: torch.Tensor
    planner_pcg64_buffer_words: torch.Tensor
    policy_np_semantic_draws: torch.Tensor
    runner_pcg64_state_words: torch.Tensor
    runner_pcg64_buffer_words: torch.Tensor
    metrics_np_semantic_draws: torch.Tensor
    event_tape_columns: tuple[torch.Tensor, ...]
    event_tape_active_count: torch.Tensor
    event_tape_status: torch.Tensor


def _capability() -> dict[str, Any]:
    return {
        "generation_summary_scalar_layout_version": (
            settlement.GENERATION_SUMMARY_LAYOUT_VERSION
        ),
        "generation_summary_scalar_layout_width": len(
            settlement.GENERATION_SUMMARY_COLUMNS
        ),
        "generation_summary_scalar_columns": list(
            settlement.GENERATION_SUMMARY_COLUMNS
        ),
        "canonical_final_checkpoint_projection_supported": True,
        "patch_p2_qualification": {
            "qualified": True,
            "profile": "P2",
            "world": "patch",
            "policy": "stochastic_fixed_mlp",
            "batch_cells": 1,
            "population": 64,
            "grid_size": 16,
            "supported_geometry_envelopes": [
                {"population": 64, "grid_size": 16},
                {"population": 128, "grid_size": 16},
                {"population": 256, "grid_size": 32},
                {"population": 512, "grid_size": 32},
            ],
            "ticks_per_generation": 80,
            "generation_turnover": True,
        },
    }


def _direct_result(
    context: PreparedDirectResidentTrajectoryV1,
    **overrides: Any,
) -> dict[str, Any]:
    logical_generation_count = context.source.logical_generation_count
    physical_window_generations = context.source.physical_window_generations
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "owner": "native_extension",
        "physical_window_generations": physical_window_generations,
        "logical_generation_count": logical_generation_count,
        "window_count": settlement.resident_window_count_v1(
            initial_generation=context.source.generation_base,
            logical_generation_count=logical_generation_count,
            physical_window_generations=physical_window_generations,
        ),
        "physical_window_alignment": "absolute_generation_lattice_v1",
        "completed_generations": logical_generation_count,
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
        "generation_summary_scalar": context.source.outputs[
            "generation_summary_scalar"
        ],
        "runtime_bookkeeping_by_tick": context.source.outputs[
            "runtime_bookkeeping_by_tick"
        ],
    }
    result.update(overrides)
    return result


def test_w16p128_patch_geometry_is_admitted_for_settlement() -> None:
    assert (128, 16) in settlement.QUALIFIED_PATCH_GEOMETRIES
    assert settlement._capability_supports_patch_geometry(
        _capability()["patch_p2_qualification"],
        population=128,
        grid_size=16,
    )


def _context(
    source_generation: int = 0,
    logical_generation_count: int = 4,
    ticks_per_generation: int = settlement.TICKS_PER_GENERATION,
    physical_window_generations: int = 2,
) -> PreparedDirectResidentTrajectoryV1:
    columns = settlement.GENERATION_SUMMARY_COLUMNS
    index = {name: offset for offset, name in enumerate(columns)}
    scalar = torch.zeros(
        (logical_generation_count, 1, len(columns)), dtype=torch.float64
    )
    for generation_offset in range(logical_generation_count):
        generation = source_generation + generation_offset
        values = {
            "generation": generation,
            "event_cursor_end": 100 + generation_offset,
            "alive_after_turnover": 64,
            "births": 2,
            "deaths": 2 + generation_offset,
            "failed_births": generation_offset,
            "moved_total": 200 + generation_offset,
            "gathered_total": 300 + generation_offset,
        }
        if generation_offset > 0:
            values["births"] = values["deaths"]
        for name, value in values.items():
            scalar[generation_offset, 0, index[name]] = value
    birth_reason_counts = torch.zeros(
        (1, logical_generation_count * ticks_per_generation, 10),
        dtype=torch.int64,
    )
    for generation_offset in range(logical_generation_count):
        tick = generation_offset * ticks_per_generation
        birth_reason_counts[0, tick, 0] = int(
            scalar[generation_offset, 0, index["births"]].item()
        )
        birth_reason_counts[0, tick, 1] = int(
            scalar[generation_offset, 0, index["failed_births"]].item()
        )

    initial_drive_noise = torch.tensor([[1.25, 2.5]], dtype=torch.float32)
    initial_drive_noise_valid = torch.tensor([[True, False]])
    birth_state = _BirthState(
        identity_state={"slot_to_agent_id": torch.empty(0)},
        scalar_rng_state={"mutation_py_semantic_draws": torch.empty(0)},
        genome_state={"genome_pred_lr": torch.empty(0)},
        policy_mem_dim=torch.empty(0),
        policy_use_gru=torch.empty(0),
        drive_state={
            "pcg64_state_words": torch.empty(0),
            "pcg64_buffer_words": torch.empty(0),
        },
        policy_state={"mlp_pack": torch.empty(0)},
        turnover_policy_state={},
        planner_pcg64_state_words=torch.empty(0),
        planner_pcg64_buffer_words=torch.empty(0),
        policy_np_semantic_draws=torch.empty(0),
        runner_pcg64_state_words=torch.empty(0),
        runner_pcg64_buffer_words=torch.empty(0),
        metrics_np_semantic_draws=torch.empty(0),
        event_tape_columns=(),
        event_tape_active_count=torch.empty(0),
        event_tape_status=torch.empty(0),
    )
    cell = SimpleNamespace(
        canonical_projection={
            "metadata": {
                "world": "patch",
                "generation": source_generation,
                "tick": 0,
                "agent_grid_runner_phase": "generation_boundary",
                "capacity": 64,
                "population_size": 64,
                "native_runtime_use_precomputed_drive_noise": True,
            },
            "tensors": {
                "alive_mask": torch.ones(64, dtype=torch.bool),
                "drive_noise_samples_by_tick_slot": initial_drive_noise,
                "drive_noise_samples_valid_by_tick_slot": (initial_drive_noise_valid),
            },
        },
        birth_state=birth_state,
    )
    outputs = {
        "generation_summary_scalar": scalar,
        "runtime_bookkeeping_by_tick": torch.ones(
            (1, logical_generation_count * ticks_per_generation, 5),
            dtype=torch.int64,
        ),
        "lifecycle_status": torch.zeros(
            (1, logical_generation_count * ticks_per_generation), dtype=torch.int64
        ),
        "event_status_by_tick": torch.zeros(
            (1, logical_generation_count * ticks_per_generation), dtype=torch.int64
        ),
        "turnover_status": torch.zeros((1,), dtype=torch.int64),
        "turnover_next_generation": torch.tensor(
            [source_generation + logical_generation_count], dtype=torch.int64
        ),
        "turnover_next_tick": torch.tensor([0], dtype=torch.int64),
        "turnover_counts": torch.zeros((1, 8), dtype=torch.int64),
        "birth_reason_counts": birth_reason_counts,
    }
    persistent_state = {
        "slot_to_agent_id": torch.tensor([[1, 2, -1]], dtype=torch.int64),
        "mutation_py_semantic_draws": torch.tensor([[5]], dtype=torch.int64),
        "genome_pred_lr": torch.tensor([[0.1, 0.2, 0.9]], dtype=torch.float32),
        "mlp_pack": torch.zeros((1, 3, 2), dtype=torch.float32),
        "population_pcg64_state_words": torch.zeros((1, 4), dtype=torch.int64),
        "population_pcg64_buffer_words": torch.zeros((1, 3), dtype=torch.int64),
        "policy_mem_dim": torch.zeros((1, 3), dtype=torch.int64),
        "policy_use_gru": torch.zeros((1, 3), dtype=torch.bool),
        "pcg64_state_words": torch.zeros((1, 4), dtype=torch.int64),
        "pcg64_buffer_words": torch.zeros((1, 3), dtype=torch.int64),
        "policy_np_semantic_draws": torch.tensor([7], dtype=torch.int64),
        "runner_pcg64_state_words": torch.zeros((1, 4), dtype=torch.int64),
        "runner_pcg64_buffer_words": torch.zeros((1, 3), dtype=torch.int64),
        "metrics_np_semantic_draws": torch.tensor([11], dtype=torch.int64),
        "event_columns": [torch.tensor([[0, 1]], dtype=torch.int64)],
        "event_active_count": torch.tensor([2], dtype=torch.int64),
        "event_status": torch.tensor([0], dtype=torch.int64),
        "alive_mask": torch.tensor([[True, True, False]]),
        "pred_lr": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        "active_count_state": torch.tensor([2], dtype=torch.int64),
        "active_agent_ids": torch.tensor([[1, 2, -1]], dtype=torch.int64),
        "pending_free_count": torch.tensor([0], dtype=torch.int64),
        "drive_noise_samples_by_tick_slot": torch.zeros((1, 1, 2), dtype=torch.float64),
        "drive_noise_samples_valid_by_tick_slot": torch.zeros(
            (1, 1, 2), dtype=torch.bool
        ),
    }
    return PreparedDirectResidentTrajectoryV1(
        source=SimpleNamespace(
            outputs=outputs,
            generation_base=source_generation,
            logical_generation_count=logical_generation_count,
            ticks_per_generation=ticks_per_generation,
            generation_turnover_enabled=False,
            lifecycle_mode=1,
            population_size=64,
            physical_window_generations=physical_window_generations,
        ),
        arguments=(),
        physical_generation_summary=None,
        physical_runtime_bookkeeping=None,
        physical_patch_stats=None,
        physical_full_tick_output_tensors=(),
        live_banks=persistent_state,
        checkpoint_projection_inputs={
            "pre_generation_canonical_projection": cell.canonical_projection,
            "resident_birth_state": cell.birth_state,
        },
    )


def _checkpoint(
    generation: int = 4,
    *,
    authority_generation: int | None = None,
    alive_count: int = 64,
    generation_turnover_enabled: bool = False,
) -> CudaBoundaryCheckpoint:
    if authority_generation is None:
        authority_generation = generation
    lifecycle_mode_abi = 0 if generation_turnover_enabled else 1
    authority_source = (
        "legacy_turnover_final_live_plus_resident_identity"
        if generation_turnover_enabled
        else "resident_identity_plus_qacct"
    )
    final_live_authority = (
        "authoritative_turnover_counter"
        if generation_turnover_enabled
        else "inactive_zero_sentinel"
    )
    final_live_value = alive_count if generation_turnover_enabled else 0
    return CudaBoundaryCheckpoint(
        projection={
            "metadata": {
                "generation": generation,
                "tick": 0,
                "alive_count": alive_count,
                "generation_turnover_enabled": generation_turnover_enabled,
                "lifecycle_mode_abi": lifecycle_mode_abi,
                "lifecycle_population_authority_source": authority_source,
                "legacy_turnover_final_live_authority": final_live_authority,
                "legacy_turnover_final_live_value": final_live_value,
            },
            "tensors": {"alive_mask": torch.ones(alive_count, dtype=torch.bool)},
        },
        projection_digest="a" * 64,
        device="cuda:0",
        device_to_host_operations=1,
        device_to_host_bytes=1,
        device_synchronizations=1,
        source_projection_authority={
            "source_generation": authority_generation,
            "source_tick": 0,
            "source_phase": "generation_boundary",
        },
    )


def _genome_v2_persistent_state(
    bank: CudaGenomeUpgradeV2StateBank,
) -> dict[str, Any]:
    cells, capacity = bank.cells, bank.capacity
    scalar = torch.zeros((cells, capacity), dtype=torch.float32)
    integer = torch.zeros((cells, capacity), dtype=torch.int64)
    world = torch.zeros((cells, 2, 2), dtype=torch.float32)
    founder = resolve_genome_v2_founder_profile_v1("neutral")
    return {
        **bank.named_tensors(),
        "genome_v2_founder_profile": {
            "schema_version": GENOME_V2_FOUNDER_PROFILE_SCHEMA,
            "profile_id": founder["profile_id"],
            "payload_sha256": founder["payload_sha256"],
        },
        "genome_v2_observability_config": {
            "max_energy": 10.0,
            "start_health": 10.0,
            "developmental_maturity_age": 2,
            "patch_density_radius": 1,
            "patch_density_include_self": True,
            "patch_density_normalize": True,
            "patch_food_amount_max": 1.0,
            "patch_waste_max": 1.0,
        },
        "alive_mask": torch.ones((cells, capacity), dtype=torch.bool),
        "energy": scalar.clone().fill_(8.0),
        "health": scalar.clone().fill_(9.0),
        "injury": scalar.clone(),
        "age": integer.clone().fill_(3),
        "pos_y": integer.clone(),
        "pos_x": integer.clone(),
        "patch_density": world.clone().fill_(0.25),
        "patch_food_amount": world.clone().fill_(0.5),
        "patch_fertility": world.clone().fill_(0.75),
        "patch_waste": world.clone().fill_(0.1),
        "genome_reproduction_threshold": scalar.clone().fill_(4.0),
        "repro_threshold": scalar.clone().fill_(5.0),
        "repro_cooldown": integer.clone(),
        "policy_mem_dim": integer.clone().fill_(8),
        "lineage_founder_id": integer.clone(),
        "lineage_depth": integer.clone().fill_(1),
    }


def _with_disabled_genome_v2_causal_outputs(
    state: dict[str, Any],
) -> dict[str, Any]:
    cells, capacity = state["alive_mask"].shape
    state["slot_to_agent_id"] = (
        torch.arange(capacity, dtype=torch.int64).expand(cells, -1).contiguous()
    )
    for producer in ("birth", "turnover"):
        prefix = f"{producer}_genome_v2_causal"
        state[f"{prefix}_record_rows"] = torch.full(
            (cells, RECORD_CAPACITY, RECORD_WIDTH),
            UNAVAILABLE_VALUE,
            dtype=torch.int64,
        )
        state[f"{prefix}_record_valid"] = torch.zeros(
            (cells, RECORD_CAPACITY), dtype=torch.bool
        )
        state[f"{prefix}_record_count"] = torch.zeros((cells,), dtype=torch.int64)
        state[f"{prefix}_record_dropped_count"] = torch.zeros(
            (cells,), dtype=torch.int64
        )
        state[f"{prefix}_capture_enabled"] = torch.zeros((cells,), dtype=torch.bool)
    return state


def test_terminal_checkpoint_sidecar_captures_live_genome_v2_arrays() -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=3, device="cpu")
    bank.alleles_q10[0, 1, 6, 3, 2] = 1234
    bank.products_q10[0, 7, 1] = 567
    stale = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1, capacity=3, device="cpu"
    ).checkpoint_projection()
    checkpoint = _checkpoint()
    checkpoint.projection["genome_upgrade_v2_checkpoint_projection"] = stale

    projected = settlement._with_resident_inheritance_sidecars_v1(
        checkpoint,
        _genome_v2_persistent_state(bank),
        cell_id="cell-a",
    )
    restored = restore_cuda_genome_upgrade_v2_state_bank(
        projected.projection["genome_upgrade_v2_checkpoint_projection"],
        device="cpu",
    )

    assert restored.alleles_q10[0, 1, 6, 3, 2].item() == 1234
    assert restored.products_q10[0, 7, 1].item() == 567
    summary = projected.projection["genome_upgrade_v2_checkpoint_projection"][
        "compact_generation_summary"
    ]["cells"][0]
    assert summary["contexts"]["available"] is True
    assert summary["contexts"]["channels"]["0"]["mean"] == 204.0
    assert summary["contexts"]["channels"]["2"]["mean"] == 256.0
    assert summary["contexts"]["channels"]["3"]["mean"] == 512.0
    assert summary["contexts"]["channels"]["4"]["mean"] == 768.0
    assert summary["contexts"]["channels"]["5"]["mean"] == 102.0
    assert summary["consequences"]["observed_count"] == 2
    assert len(summary["selected_causal_events"]) == 3
    assert {row["event_kind"] for row in summary["selected_causal_events"]} == {
        "birth_or_turnover_child_terminal_snapshot"
    }
    assert projected.projection_digest != checkpoint.projection_digest


def test_terminal_checkpoint_sidecar_attaches_recomb72_longitudinal_summary() -> None:
    profile_id = (
        "recomb72_gv2_v1.correlated_multi_product_linkage_packages.all_heterozygous_ab"
    )
    founder = resolve_genome_v2_founder_profile_v1(profile_id)
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=4, device="cpu")
    alive = torch.ones((1, 4), dtype=torch.bool)
    bank.initialize_founder_populations_by_cell_v1(
        (profile_id,),
        alive_mask=alive,
    )
    bank.express(torch.zeros((1, 4, 8), dtype=torch.int64))
    persistent = _genome_v2_persistent_state(bank)
    persistent["genome_v2_founder_profile"] = {
        "schema_version": GENOME_V2_FOUNDER_PROFILE_SCHEMA,
        "profile_id": profile_id,
        "payload_sha256": founder["payload_sha256"],
    }

    projected = settlement._with_resident_inheritance_sidecars_v1(
        _checkpoint(),
        persistent,
        cell_id="cell-a",
    )
    sidecar = projected.projection["genome_upgrade_v2_checkpoint_projection"]
    recomb72 = sidecar["compact_generation_summary"]["cells"][0]["recomb72_gv2"]

    assert sidecar["founder_profile"] == persistent["genome_v2_founder_profile"]
    assert recomb72["profile_sha256"] == founder["payload_sha256"]
    assert (
        recomb72["founder_population_sha256"]
        == founder["founder_population"]["payload_sha256"]
    )
    assert recomb72["ordered_genotype_counts"] == {
        "aa": 0,
        "ab": 4,
        "ba": 0,
        "bb": 0,
        "other": 0,
    }


def test_terminal_checkpoint_sidecar_attaches_bounded_disabled_causal_projection() -> (
    None
):
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=3, device="cpu")
    persistent = _with_disabled_genome_v2_causal_outputs(
        _genome_v2_persistent_state(bank)
    )

    projected = settlement._with_resident_inheritance_sidecars_v1(
        _checkpoint(),
        persistent,
        cell_id="cell-a",
    )

    sidecar = projected.projection["genome_upgrade_v2_checkpoint_projection"]
    causal = sidecar["causal_transaction_projection"]
    assert causal["capture_status"] == "capture_disabled"
    assert causal["captured_counts"]["total"] == 0
    assert causal["evaluable"] is False
    compact = sidecar["compact_generation_summary"]["causal_transaction_projection"]
    assert compact["projection_sha256"] == causal["projection_sha256"]
    assert "event_receipts" not in compact


def test_terminal_checkpoint_sidecar_rejects_malformed_observability_state() -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=3, device="cpu")
    persistent = _genome_v2_persistent_state(bank)
    persistent["energy"] = torch.zeros((1, 2), dtype=torch.float32)

    with pytest.raises(
        settlement.CudaArchitectureBDirectResidentSettlementError,
        match="source tensor shape differs: energy",
    ):
        settlement._with_resident_inheritance_sidecars_v1(
            _checkpoint(),
            persistent,
            cell_id="cell-a",
        )


def test_terminal_checkpoint_sidecar_rejects_partial_genome_v2_arrays() -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=3, device="cpu")
    state = bank.named_tensors()
    del state["genome_v2_products_q10"]

    with pytest.raises(
        settlement.CudaArchitectureBDirectResidentSettlementError,
        match="sidecar is incomplete",
    ):
        settlement._with_resident_inheritance_sidecars_v1(
            _checkpoint(),
            state,
            cell_id="cell-a",
        )


def _materialized_event_result(*, sequence_start: int = 0) -> dict[str, Any]:
    rows = 3
    history = [torch.zeros(rows, dtype=torch.int64) for _ in EVENT_COLUMN_NAMES]
    sequence_index = EVENT_COLUMN_NAMES.index("typed_lifecycle_event_sequence")
    history[sequence_index] = torch.arange(
        sequence_start,
        sequence_start + rows,
        dtype=torch.int64,
    )
    return {"event_histories": [history], "event_rows_by_cell": [rows]}


def test_materialized_event_history_replaces_the_recycled_window() -> None:
    original = {"event_columns": [], "event_active_count": torch.tensor([1])}

    result = settlement._with_materialized_event_history(
        original,
        _materialized_event_result(),
    )

    assert result is not original
    assert int(result["event_active_count"].item()) == 3
    assert len(result["event_columns"]) == len(EVENT_COLUMN_NAMES)
    assert result["event_sequence_base"].tolist() == [0]


def test_materialized_event_history_rejects_nonzero_sequence_base() -> None:
    with pytest.raises(
        settlement.CudaArchitectureBDirectResidentSettlementError,
        match="zero-based contiguous",
    ):
        settlement._with_materialized_event_history(
            {},
            _materialized_event_result(sequence_start=5),
        )


def test_drained_event_history_rejoins_canonical_prefix_before_checkpoint() -> None:
    prefix_rows = 2
    local_rows = 3
    canonical = {
        "tensors": {
            name: torch.arange(
                prefix_rows,
                dtype=torch.int64,
            )
            + (100 * index)
            for index, name in enumerate(EVENT_COLUMN_NAMES)
        }
    }
    sequence_index = EVENT_COLUMN_NAMES.index("typed_lifecycle_event_sequence")
    canonical["tensors"][EVENT_COLUMN_NAMES[sequence_index]] = torch.arange(
        prefix_rows,
        dtype=torch.int64,
    )
    local_columns = [
        (
            torch.arange(local_rows, dtype=torch.int64)
            + (1_000 + 100 * index)
        ).unsqueeze(0)
        for index, _name in enumerate(EVENT_COLUMN_NAMES)
    ]
    local_columns[sequence_index] = torch.arange(
        local_rows,
        dtype=torch.int64,
    ).unsqueeze(0)
    persistent = {
        "event_history_drained": True,
        "event_columns": local_columns,
        "event_active_count": torch.tensor([local_rows], dtype=torch.int64),
    }

    merged = settlement._merge_drained_event_history_v1(canonical, persistent)

    assert int(merged["event_active_count"].item()) == prefix_rows + local_rows
    for index, name in enumerate(EVENT_COLUMN_NAMES):
        expected = torch.cat(
            (
                canonical["tensors"][name],
                local_columns[index][0],
            )
        )
        if index == sequence_index:
            expected = torch.arange(prefix_rows + local_rows, dtype=torch.int64)
        torch.testing.assert_close(merged["event_columns"][index][0], expected)
        torch.testing.assert_close(merged[name][0], expected)
    assert merged["event_sequence_base"].tolist() == [0]


def test_settlement_decodes_four_rows_and_projects_one_terminal_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    projected: list[dict[str, Any]] = []
    checkpoint = _checkpoint()

    def fake_project(**kwargs: Any) -> CudaBoundaryCheckpoint:
        projected.append(kwargs)
        return checkpoint

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )

    result = settlement.settle_patch_p2_b1_g4_direct_resident_v1(
        context,
        direct_result=_direct_result(context),
        capability=_capability(),
    )

    assert result.schema_version == settlement.SETTLEMENT_SCHEMA_VERSION
    assert result.terminal_generation == 4
    assert result.terminal_checkpoint is checkpoint
    assert len(result.compact_summary_rows) == 4
    expected_first_row = {
        "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
        "generation": 0,
        "world": "patch",
        "alive": 64,
        "births": 2,
        "failed_births": 0,
        "deaths": 2,
        "event_cursor": 100,
        "moved": 200,
        "gathered": 300,
        "patch_food_stock_after_turnover": 0.0,
        "energy_collected_exact_sum_after_turnover": 0.0,
        "policy_energy_live_sum_after_turnover": 0.0,
    }
    assert {
        name: result.compact_summary_rows[0][name]
        for name in expected_first_row
    } == expected_first_row
    assert result.lifecycle_qacct_receipt is not None
    assert result.lifecycle_qacct_receipt["counters"]["residual"] == 0
    assert result.lifecycle_qacct_receipt["counters"]["legacy_turnover_culls"] == 0
    assert result.lifecycle_qacct_receipt["counters"]["hard_cap_removals"] == 0
    assert result.lifecycle_qacct_receipt["counters"]["no_capacity_rejections"] == 6
    assert result.lifecycle_qacct_receipt["capacity_accounting"][
        "failed_birth_reason_sum"
    ] == 6
    assert result.lifecycle_qacct_receipt["capacity_accounting"][
        "failed_birth_reasons"
    ]["no_space"] == 6
    assert result.lifecycle_qacct_receipt["capacity_accounting"][
        "reason_authority_kind"
    ] == "authoritative_full_prefix_route_counts"
    assert result.lifecycle_qacct_receipt["capacity_accounting"][
        "reason_qualification_authority"
    ] is True
    assert result.lifecycle_qacct_receipt[
        "capacity_accounting_reason_qualification_authority"
    ] is True
    assert result.lifecycle_qacct_receipt["generation_turnover_enabled"] is False
    assert result.lifecycle_qacct_receipt["lifecycle_mode_abi"] == 1
    assert result.lifecycle_qacct_receipt["population_authority_source"] == (
        "resident_identity_plus_qacct"
    )
    assert result.lifecycle_qacct_receipt[
        "legacy_turnover_final_live_authority"
    ] == "inactive_zero_sentinel"
    assert result.compact_summary_rows[-1]["generation"] == 3
    assert len(projected) == 1
    call = projected[0]
    assert call["completed_generation_count"] == 4
    assert torch.equal(
        call["resident_boundary_tensors"]["drive_noise_samples_by_tick_slot"],
        context.checkpoint_projection_inputs["pre_generation_canonical_projection"][
            "tensors"
        ]["drive_noise_samples_by_tick_slot"].to(dtype=torch.float64),
    )
    assert torch.equal(
        call["resident_birth_state"].drive_state["pcg64_state_words"],
        context.live_banks["population_pcg64_state_words"][0],
    )
    assert int(call["generation_result"].applied_actions.item()) == 320


def test_hybrid_lifecycle_qacct_keeps_raw_deaths_separate_from_terminal_balance() -> None:
    source = SimpleNamespace(
        generation_turnover_enabled=False,
        lifecycle_mode=1,
        genome_upgrade_v2_founder_profile_sha256="b" * 64,
        population_size=64,
        resident_capacity=256,
    )
    source_projection = {
        "metadata": {
            "generation": 0,
            "tick": 0,
            "population_size": 64,
            "capacity": 256,
        },
        "tensors": {"alive_mask": torch.ones(64, dtype=torch.bool)},
    }

    receipt = settlement._lifecycle_qacct_receipt_from_route_v1(
        source,
        {
            "event_active_count": torch.tensor([0], dtype=torch.int64),
            "typed_lifecycle_event_event_type": torch.zeros((1, 0), dtype=torch.int32),
            "typed_lifecycle_event_reason_code": torch.zeros((1, 0), dtype=torch.int32),
            "typed_lifecycle_event_success": torch.zeros((1, 0), dtype=torch.int8),
        },
        {
            "turnover_counts": torch.zeros((1, 8), dtype=torch.int64),
            "birth_reason_counts": torch.zeros((1, 0, 10), dtype=torch.int64),
        },
        (
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": 0,
                "world": "patch",
                "alive": 64,
                "births": 0,
                "failed_births": 0,
                "deaths": 27,
                "event_cursor": 0,
                "moved": 0,
                "gathered": 0,
            },
        ),
        _checkpoint(1, alive_count=64, generation_turnover_enabled=False),
        source_projection,
    )

    counters = receipt["counters"]
    capacity = receipt["capacity_accounting"]
    assert counters["ecological_deaths"] == 0
    assert counters["legacy_turnover_culls"] == 0
    assert capacity["terminal_removals"] == 0
    assert counters["residual"] == 0
    assert capacity["raw_ecological_death_events"] == 27
    assert capacity["effective_ecological_deaths"] == 0
    assert capacity["ecological_death_accounting_authority"] == (
        "hybrid_lifecycle_population_balance_with_raw_event_count_retained"
    )


@pytest.mark.parametrize("completed", [1, 3])
def test_v2_settlement_projects_an_exact_completed_prefix(
    monkeypatch: pytest.MonkeyPatch,
    completed: int,
) -> None:
    context = _context()
    context.source.outputs["turnover_next_generation"] = torch.tensor(
        [completed], dtype=torch.int64
    )
    projected: list[dict[str, Any]] = []

    def fake_project(**kwargs: Any) -> CudaBoundaryCheckpoint:
        projected.append(kwargs)
        return _checkpoint(completed)

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )
    result = settlement.settle_patch_p2_b1_direct_resident_prefix_v2(
        context,
        direct_result=_direct_result(
            context,
            completed_generations=completed,
            settled_window_count=(completed + 1) // 2,
            generation_summary_scalar=context.source.outputs[
                "generation_summary_scalar"
            ][:completed],
            runtime_bookkeeping_by_tick=context.source.outputs[
                "runtime_bookkeeping_by_tick"
            ][:, : completed * settlement.TICKS_PER_GENERATION],
        ),
        capability=_capability(),
    )

    assert result.terminal_generation == completed
    assert len(result.compact_summary_rows) == completed
    assert result.compact_summary_rows[-1]["generation"] == completed - 1
    assert len(projected) == 1
    assert projected[0]["completed_generation_count"] == completed
    assert int(projected[0]["generation_result"].applied_actions.item()) == (
        completed * settlement.TICKS_PER_GENERATION
    )
    assert result.lifecycle_qacct_receipt is not None
    assert result.lifecycle_qacct_receipt["counters"]["residual"] == 0


def test_v2_settlement_omits_empty_drive_noise_placeholders_from_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    context.checkpoint_projection_inputs["pre_generation_canonical_projection"][
        "metadata"
    ]["native_runtime_use_precomputed_drive_noise"] = False
    context.live_banks["drive_noise_samples_by_tick_slot"] = torch.empty(
        (1, 0, 2), dtype=torch.float64
    )
    context.live_banks["drive_noise_samples_valid_by_tick_slot"] = torch.empty(
        (1, 0, 2), dtype=torch.bool
    )
    projected: list[dict[str, Any]] = []

    def fake_project(**kwargs: Any) -> CudaBoundaryCheckpoint:
        projected.append(kwargs)
        return _checkpoint(1)

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )
    settlement.settle_patch_p2_b1_direct_resident_prefix_v2(
        context,
        direct_result=_direct_result(
            context,
            completed_generations=1,
            settled_window_count=1,
            generation_summary_scalar=context.source.outputs[
                "generation_summary_scalar"
            ][:1],
            runtime_bookkeeping_by_tick=context.source.outputs[
                "runtime_bookkeeping_by_tick"
            ][:, : settlement.TICKS_PER_GENERATION],
        ),
        capability=_capability(),
    )

    assert len(projected) == 1
    resident_boundary_tensors = projected[0]["resident_boundary_tensors"]
    assert "drive_noise_samples_by_tick_slot" not in resident_boundary_tensors
    assert "drive_noise_samples_valid_by_tick_slot" not in resident_boundary_tensors


def test_v2_settlement_projects_exact_nonboundary_tick_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    projected: list[dict[str, Any]] = []

    def fake_project(**kwargs: Any) -> CudaBoundaryCheckpoint:
        projected.append(kwargs)
        checkpoint = _checkpoint(0)
        checkpoint.projection["metadata"].update(
            {"tick": 17, "agent_grid_runner_phase": "partial_generation"}
        )
        checkpoint.source_projection_authority.update(
            {
                "source_generation": 0,
                "source_tick": 17,
                "source_phase": "partial_generation",
            }
        )
        return checkpoint

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )
    direct_result = _direct_result(
        context,
        completed_generations=1,
        settled_window_count=1,
        runtime_bookkeeping_by_tick=context.source.outputs[
            "runtime_bookkeeping_by_tick"
        ][:, :17],
        absolute_tick_span_native_owner=True,
        tick_span_start=0,
        tick_span_count=17,
        tick_span_end=17,
        terminal_generation_boundary=False,
    )
    direct_result.pop("generation_summary_scalar")
    result = settlement.settle_patch_p2_b1_direct_resident_prefix_v2(
        context,
        direct_result=direct_result,
        capability=_capability(),
        reporting_enabled=False,
    )

    assert result.terminal_generation == 0
    assert result.terminal_tick == 17
    assert result.compact_summary_rows == ()
    assert projected[0]["completed_interval_steps"] == 17
    assert int(projected[0]["generation_result"].turnover_next_tick.item()) == 17


def test_private_absolute_tick_adapter_projects_tick_321_from_derived_local_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        source_generation=2,
        logical_generation_count=7,
        ticks_per_generation=160,
        physical_window_generations=1,
    )
    projected: list[dict[str, Any]] = []

    def fake_project(**kwargs: Any) -> CudaBoundaryCheckpoint:
        projected.append(kwargs)
        checkpoint = _checkpoint(2, authority_generation=2)
        checkpoint.projection["metadata"].update(
            {"tick": 1, "agent_grid_runner_phase": "partial_generation"}
        )
        checkpoint.source_projection_authority.update(
            {
                "source_generation": 2,
                "source_tick": 1,
                "source_phase": "partial_generation",
            }
        )
        return checkpoint

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )
    direct_result = _direct_result(
        context,
        completed_generations=1,
        settled_window_count=1,
        runtime_bookkeeping_by_tick=context.source.outputs[
            "runtime_bookkeeping_by_tick"
        ][:, :1],
        absolute_tick_span_native_owner=True,
        tick_native_start_next_tick=160,
        tick_native_end_next_tick=321,
        tick_span_start=0,
        tick_span_count=1,
        tick_span_end=1,
        terminal_generation_boundary=False,
    )
    direct_result.pop("generation_summary_scalar")
    result = settlement.settle_patch_p2_b1_direct_resident_prefix_v2(
        context,
        direct_result=direct_result,
        capability=_capability(),
        reporting_enabled=False,
    )

    assert result.terminal_generation == 2
    assert result.terminal_tick == 1
    assert projected[0]["completed_interval_steps"] == 1


def test_private_final_suffix_settlement_accepts_one_bookkeeping_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        source_generation=20,
        logical_generation_count=7,
        ticks_per_generation=160,
        physical_window_generations=1,
    )
    source_metadata = context.checkpoint_projection_inputs[
        "pre_generation_canonical_projection"
    ]["metadata"]
    source_metadata.update(
        {
            "tick": 159,
            "prediction_tick": 159,
            "agent_grid_runner_phase": "partial_generation",
        }
    )
    projected: list[dict[str, Any]] = []

    def fake_project(**kwargs: Any) -> CudaBoundaryCheckpoint:
        projected.append(kwargs)
        return _checkpoint(21)

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )
    direct_result = _direct_result(
        context,
        completed_generations=1,
        settled_window_count=1,
        runtime_bookkeeping_by_tick=context.source.outputs[
            "runtime_bookkeeping_by_tick"
        ][:, :1],
        absolute_tick_span_native_owner=True,
        tick_native_start_next_tick=3_359,
        tick_native_end_next_tick=3_360,
        tick_span_start=159,
        tick_span_count=1,
        tick_span_end=160,
        terminal_generation_boundary=True,
    )
    direct_result.pop("generation_summary_scalar")

    result = settlement.settle_patch_p2_b1_direct_resident_prefix_v2(
        context,
        direct_result=direct_result,
        capability=_capability(),
        reporting_enabled=False,
    )

    assert result.terminal_generation == 21
    assert result.terminal_tick == 0
    assert len(projected) == 1
    generation_result = projected[0]["generation_result"]
    assert tuple(generation_result.runtime_bookkeeping_by_tick.shape) == (1, 5)
    assert projected[0]["completed_interval_steps"] is None


def test_hybrid_settlement_derives_boundary_without_turnover_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    context.source.outputs["turnover_next_generation"] = torch.tensor(
        [0], dtype=torch.int64
    )
    context.source.outputs["turnover_next_tick"] = torch.tensor([999], dtype=torch.int64)
    projected: list[dict[str, Any]] = []

    def fake_project(**kwargs: Any) -> CudaBoundaryCheckpoint:
        projected.append(kwargs)
        return _checkpoint(4)

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )
    result = settlement.settle_patch_p2_b1_direct_resident_prefix_v2(
        context,
        direct_result=_direct_result(context),
        capability=_capability(),
    )

    assert result.terminal_generation == 4
    assert len(projected) == 1
    assert int(projected[0]["generation_result"].turnover_next_generation.item()) == 4
    assert int(projected[0]["generation_result"].turnover_next_tick.item()) == 0


def test_hybrid_settlement_ignores_stale_legacy_turnover_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    context.source.outputs["turnover_next_generation"] = torch.tensor(
        [5], dtype=torch.int64
    )
    projected: list[dict[str, Any]] = []

    def fake_project(**kwargs: Any) -> CudaBoundaryCheckpoint:
        projected.append(kwargs)
        return _checkpoint(4)

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )
    settlement.settle_patch_p2_b1_direct_resident_prefix_v2(
        context,
        direct_result=_direct_result(context),
        capability=_capability(),
    )

    assert len(projected) == 1
    assert int(projected[0]["generation_result"].turnover_next_generation.item()) == 4
    assert int(projected[0]["generation_result"].turnover_next_tick.item()) == 0


def test_hybrid_resume_qacct_uses_checkpoint_alive_mask_for_live_start() -> None:
    source = SimpleNamespace(
        generation_turnover_enabled=False,
        lifecycle_mode=1,
        genome_upgrade_v2_founder_profile_sha256="b" * 64,
    )
    source_projection = {
        "metadata": {
            "generation": 4,
            "tick": 0,
            "population_size": 64,
            "capacity": 256,
        },
        "tensors": {"alive_mask": torch.ones(256, dtype=torch.bool)},
    }

    receipt = settlement._lifecycle_qacct_receipt_from_route_v1(
        source,
        {},
        {
            "turnover_counts": torch.zeros((1, 8), dtype=torch.int64),
            "birth_reason_counts": torch.zeros((1, 80, 10), dtype=torch.int64),
        },
        (
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": 4,
                "world": "patch",
                "alive": 256,
                "births": 0,
                "failed_births": 0,
                "deaths": 0,
                "event_cursor": 1,
                "moved": 0,
                "gathered": 0,
            },
        ),
        _checkpoint(8, alive_count=256),
        source_projection,
    )

    counters = receipt["counters"]
    assert counters["live_start"] == 256
    assert counters["live_end"] == 256
    assert counters["residual"] == 0
    assert receipt["population_authority_source"] == "resident_identity_plus_qacct"
    assert receipt["legacy_turnover_final_live_value"] == 0


def test_turnover_qacct_uses_compact_prefix_population_balance_for_culls() -> None:
    source = SimpleNamespace(
        generation_turnover_enabled=True,
        lifecycle_mode=0,
        genome_upgrade_v2_founder_profile_sha256="b" * 64,
        population_size=256,
        resident_capacity=512,
    )
    source_projection = {
        "metadata": {
            "generation": 0,
            "tick": 0,
            "population_size": 256,
            "capacity": 512,
        },
        "tensors": {
            "alive_mask": torch.tensor(
                [*[True] * 256, *[False] * 256], dtype=torch.bool
            )
        },
    }
    birth_reason_counts = torch.zeros((1, 3 * 80, 10), dtype=torch.int64)
    for generation in range(3):
        birth_reason_counts[0, generation * 80, 0] = 256
    turnover_counts = torch.zeros((1, 8), dtype=torch.int64)
    turnover_counts[0, 1] = 256
    turnover_counts[0, 4] = 256
    turnover_counts[0, 7] = 256

    receipt = settlement._lifecycle_qacct_receipt_from_route_v1(
        source,
        {},
        {
            "turnover_counts": turnover_counts,
            "birth_reason_counts": birth_reason_counts,
        },
        (
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": 0,
                "world": "patch",
                "alive": 256,
                "births": 256,
                "failed_births": 0,
                "deaths": 2,
                "event_cursor": 362,
                "moved": 0,
                "gathered": 0,
            },
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": 1,
                "world": "patch",
                "alive": 256,
                "births": 256,
                "failed_births": 0,
                "deaths": 2,
                "event_cursor": 718,
                "moved": 0,
                "gathered": 0,
            },
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": 2,
                "world": "patch",
                "alive": 256,
                "births": 256,
                "failed_births": 0,
                "deaths": 1,
                "event_cursor": 1089,
                "moved": 0,
                "gathered": 0,
            },
        ),
        _checkpoint(3, alive_count=256, generation_turnover_enabled=True),
        source_projection,
    )

    counters = receipt["counters"]
    assert counters["live_start"] == 256
    assert counters["admitted_births"] == 768
    assert counters["ecological_deaths"] == 5
    assert counters["legacy_turnover_culls"] == 763
    assert counters["live_end"] == 256
    assert counters["residual"] == 0
    assert receipt["capacity_accounting"]["terminal_removals"] == 768
    assert receipt["capacity_accounting"][
        "population_conservation_expected_live_end"
    ] == 256


def test_provider_native_segment_artifact_splits_mixed_success_failure_rows() -> None:
    artifact, _metadata = settlement._provider_native_segment_artifact_v1(
        cell_id="cell-0",
        cell_index=0,
        source_projection={
            "metadata": {
                "native_next_segment_index": 0,
                "native_chain_sha256": "0" * 64,
            }
        },
        compact_rows=(
            {"generation": 0, "alive": 122, "births": 58, "failed_births": 0},
            {"generation": 1, "alive": 238, "births": 116, "failed_births": 475},
            {"generation": 2, "alive": 256, "births": 18, "failed_births": 14673},
            {"generation": 3, "alive": 256, "births": 0, "failed_births": 20480},
        ),
    )

    cell = artifact["cell"]
    assert cell["attempt_count"] == 6
    assert cell["success_count"] == 3
    assert cell["failure_count"] == 3
    assert cell["success_count"] + cell["failure_count"] == cell["attempt_count"]
    assert [attempt["success"] for attempt in cell["attempts"]] == [
        True,
        True,
        False,
        True,
        False,
        False,
    ]


def test_provider_native_segment_artifact_preserves_cumulative_counts() -> None:
    artifact, metadata = settlement._provider_native_segment_artifact_v1(
        cell_id="cell-0",
        cell_index=0,
        source_projection={
            "metadata": {
                "native_next_segment_index": 1,
                "native_chain_sha256": "a" * 64,
                "native_cumulative_attempt_count": 5,
                "native_cumulative_success_count": 3,
                "native_cumulative_failure_count": 2,
                "native_cumulative_snapshot_count": 0,
            }
        },
        compact_rows=(
            {"generation": 3, "alive": 43, "births": 0, "failed_births": 0},
            {"generation": 4, "alive": 43, "births": 1, "failed_births": 0},
        ),
    )

    assert artifact["segment_index"] == 1
    assert artifact["cell"]["cumulative_counts_after_segment"] == {
        "attempts": 7,
        "successes": 4,
        "failures": 3,
        "eligible_snapshots": 0,
    }
    assert metadata["native_next_segment_index"] == 2
    assert metadata["native_cumulative_attempt_count"] == 7
    assert metadata["native_cumulative_success_count"] == 4
    assert metadata["native_cumulative_failure_count"] == 3
    assert metadata["native_cumulative_snapshot_count"] == 0


def test_provider_native_segment_artifact_records_partial_tick_noop() -> None:
    artifact, metadata = settlement._provider_native_segment_artifact_v1(
        cell_id="cell-0",
        cell_index=0,
        source_projection={
            "metadata": {
                "native_next_segment_index": 2,
                "native_chain_sha256": "b" * 64,
                "native_cumulative_attempt_count": 7,
                "native_cumulative_success_count": 4,
                "native_cumulative_failure_count": 3,
                "native_cumulative_snapshot_count": 1,
            }
        },
        compact_rows=(),
    )

    assert artifact["segment_index"] == 2
    assert artifact["capacity_semantics"]["scope"] == (
        "partial_tick_segment_no_generation_summary_v1"
    )
    assert artifact["cell"]["attempts"] == []
    assert artifact["cell"]["cumulative_counts_after_segment"] == {
        "attempts": 7,
        "successes": 4,
        "failures": 3,
        "eligible_snapshots": 1,
    }
    assert metadata["native_next_segment_index"] == 3


def test_hybrid_qacct_accepts_mapping_backed_lifecycle_configuration() -> None:
    source = SimpleNamespace(
        configuration={
            "generation_turnover_enabled": False,
            "lifecycle_mode": 1,
            "genome_upgrade_v2_founder_profile_sha256": "b" * 64,
            "population_size": 64,
        },
    )
    source_projection = {
        "metadata": {
            "generation": 4,
            "tick": 0,
            "population_size": 64,
            "capacity": 256,
        },
        "tensors": {"alive_mask": torch.ones(256, dtype=torch.bool)},
    }

    receipt = settlement._lifecycle_qacct_receipt_from_route_v1(
        source,
        {},
        {
            "turnover_counts": torch.zeros((1, 8), dtype=torch.int64),
            "birth_reason_counts": torch.zeros((1, 80, 10), dtype=torch.int64),
        },
        (
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": 4,
                "world": "patch",
                "alive": 256,
                "births": 0,
                "failed_births": 0,
                "deaths": 0,
                "event_cursor": 1,
                "moved": 0,
                "gathered": 0,
            },
        ),
        _checkpoint(8, alive_count=256),
        source_projection,
    )

    assert receipt["generation_turnover_enabled"] is False
    assert receipt["lifecycle_mode_abi"] == 1
    assert receipt["population_authority_source"] == "resident_identity_plus_qacct"


def test_hybrid_qacct_uses_typed_event_reasons_when_tick_counts_are_short() -> None:
    source = SimpleNamespace(
        generation_turnover_enabled=False,
        lifecycle_mode=1,
        genome_upgrade_v2_founder_profile_sha256="b" * 64,
        population_size=64,
    )
    source_projection = {
        "metadata": {
            "generation": 4,
            "tick": 0,
            "population_size": 64,
            "capacity": 256,
        },
        "tensors": {
            "alive_mask": torch.tensor(
                [*[True] * 64, *[False] * 192], dtype=torch.bool
            )
        },
    }

    receipt = settlement._lifecycle_qacct_receipt_from_route_v1(
        source,
        {
            "event_active_count": torch.tensor([3], dtype=torch.int64),
            "typed_lifecycle_event_event_type": torch.tensor(
                [[3, 3, 1]], dtype=torch.int32
            ),
            "typed_lifecycle_event_reason_code": torch.tensor(
                [[1, 3, 0]], dtype=torch.int32
            ),
        },
        {
            "turnover_counts": torch.zeros((1, 8), dtype=torch.int64),
            "birth_reason_counts": torch.zeros((1, 80, 10), dtype=torch.int64),
        },
        (
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": 4,
                "world": "patch",
                "alive": 256,
                "births": 192,
                "failed_births": 2,
                "deaths": 0,
                "event_cursor": 3,
                "moved": 0,
                "gathered": 0,
            },
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": 5,
                "world": "patch",
                "alive": 256,
                "births": 0,
                "failed_births": 0,
                "deaths": 0,
                "event_cursor": 3,
                "moved": 0,
                "gathered": 0,
            },
        ),
        _checkpoint(6, alive_count=256),
        source_projection,
    )

    capacity = receipt["capacity_accounting"]
    assert capacity["reason_accounting_source"] == (
        "typed_lifecycle_event_reason_history"
    )
    assert capacity["reason_authority_kind"] == "authoritative_full_prefix_route_counts"
    assert capacity["reason_qualification_authority"] is True
    assert capacity["reason_accounted_count"] == 3
    assert capacity["failed_birth_reason_sum"] == 2
    assert capacity["failed_birth_reasons"]["no_space"] == 1
    assert capacity["failed_birth_reasons"]["no_free_slot"] == 1


def test_hybrid_qacct_saturates_unobserved_failed_reasons_only_at_full_capacity() -> None:
    source = SimpleNamespace(
        generation_turnover_enabled=False,
        lifecycle_mode=1,
        genome_upgrade_v2_founder_profile_sha256="b" * 64,
        population_size=64,
    )
    source_projection = {
        "metadata": {
            "generation": 4,
            "tick": 0,
            "population_size": 64,
            "capacity": 256,
        },
        "tensors": {"alive_mask": torch.ones(256, dtype=torch.bool)},
    }

    receipt = settlement._lifecycle_qacct_receipt_from_route_v1(
        source,
        {
            "event_active_count": torch.tensor([1], dtype=torch.int64),
            "typed_lifecycle_event_event_type": torch.tensor([[3]], dtype=torch.int32),
            "typed_lifecycle_event_reason_code": torch.tensor([[1]], dtype=torch.int32),
        },
        {
            "turnover_counts": torch.zeros((1, 8), dtype=torch.int64),
            "birth_reason_counts": torch.zeros((1, 80, 10), dtype=torch.int64),
        },
        (
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": 4,
                "world": "patch",
                "alive": 256,
                "births": 0,
                "failed_births": 10,
                "deaths": 0,
                "event_cursor": 1,
                "moved": 0,
                "gathered": 0,
            },
        ),
        _checkpoint(5, alive_count=256),
        source_projection,
    )

    capacity = receipt["capacity_accounting"]
    assert capacity["reason_accounting_source"].endswith(
        "_plus_saturated_capacity_closed_form"
    )
    assert capacity["reason_authority_kind"] == "diagnostic_inferred_saturated_capacity"
    assert capacity["reason_qualification_authority"] is False
    assert receipt["capacity_accounting_reason_qualification_authority"] is False
    assert capacity["failed_birth_reason_sum"] == 10
    assert capacity["failed_birth_reasons"]["no_space"] == 10


def test_hybrid_qacct_reconciles_mismatched_tick_reasons_with_typed_generation_filter() -> None:
    source = SimpleNamespace(
        generation_turnover_enabled=False,
        lifecycle_mode=1,
        genome_upgrade_v2_founder_profile_sha256="b" * 64,
        population_size=64,
    )
    source_projection = {
        "metadata": {
            "generation": 28,
            "tick": 0,
            "population_size": 64,
            "capacity": 256,
        },
        "tensors": {
            "alive_mask": torch.tensor(
                [*[True] * 255, False], dtype=torch.bool
            )
        },
    }

    receipt = settlement._lifecycle_qacct_receipt_from_route_v1(
        source,
        {
            "event_active_count": torch.tensor([5], dtype=torch.int64),
            "typed_lifecycle_event_event_type": torch.tensor(
                [[3, 3, 1, 3, 1]], dtype=torch.int32
            ),
            "typed_lifecycle_event_reason_code": torch.tensor(
                [[1, 3, -1, 1, -1]], dtype=torch.int32
            ),
            "typed_lifecycle_event_generation": torch.tensor(
                [[27, 28, 28, 29, 30]], dtype=torch.int32
            ),
            "typed_lifecycle_event_success": torch.tensor(
                [[0, 0, 1, 0, 1]], dtype=torch.int8
            ),
        },
        {
            "turnover_counts": torch.zeros((1, 8), dtype=torch.int64),
            "birth_reason_counts": torch.zeros((1, 320, 10), dtype=torch.int64),
        },
        (
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": 28,
                "world": "patch",
                "alive": 255,
                "births": 1,
                "failed_births": 1,
                "deaths": 1,
                "event_cursor": 4,
                "moved": 0,
                "gathered": 0,
            },
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": 29,
                "world": "patch",
                "alive": 255,
                "births": 0,
                "failed_births": 1,
                "deaths": 0,
                "event_cursor": 4,
                "moved": 0,
                "gathered": 0,
            },
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": 30,
                "world": "patch",
                "alive": 256,
                "births": 1,
                "failed_births": 0,
                "deaths": 0,
                "event_cursor": 5,
                "moved": 0,
                "gathered": 0,
            },
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": 31,
                "world": "patch",
                "alive": 256,
                "births": 0,
                "failed_births": 0,
                "deaths": 0,
                "event_cursor": 5,
                "moved": 0,
                "gathered": 0,
            },
        ),
        _checkpoint(32, alive_count=256),
        source_projection,
    )

    capacity = receipt["capacity_accounting"]
    assert capacity["reason_accounting_source"] == (
        "typed_lifecycle_event_reason_history_generation_filtered"
    )
    assert capacity["reason_authority_kind"] == "authoritative_full_prefix_route_counts"
    assert capacity["reason_qualification_authority"] is True
    assert capacity["failed_birth_reason_sum"] == 2
    assert capacity["failed_birth_reasons"]["no_space"] == 1
    assert capacity["failed_birth_reasons"]["no_free_slot"] == 1
    assert capacity["ordinary_birth_decisions_accepted"] == 2


def test_legacy_qacct_separates_turnover_births_from_ordinary_birth_events() -> None:
    source = SimpleNamespace(
        generation_turnover_enabled=True,
        lifecycle_mode=0,
        genome_upgrade_v2_founder_profile_sha256="b" * 64,
        population_size=64,
        resident_capacity=64,
    )
    source_projection = {
        "metadata": {
            "generation": 0,
            "tick": 0,
            "population_size": 64,
            "capacity": 64,
        },
        "tensors": {"alive_mask": torch.ones(64, dtype=torch.bool)},
    }
    turnover_counts = torch.zeros((1, 8), dtype=torch.int64)
    turnover_counts[0, 0] = 64
    turnover_counts[0, 1] = 32
    turnover_counts[0, 6] = 32
    turnover_counts[0, 7] = 64

    receipt = settlement._lifecycle_qacct_receipt_from_route_v1(
        source,
        {
            "event_active_count": torch.tensor([0], dtype=torch.int64),
            "typed_lifecycle_event_event_type": torch.zeros((1, 0), dtype=torch.int32),
            "typed_lifecycle_event_reason_code": torch.zeros((1, 0), dtype=torch.int32),
            "typed_lifecycle_event_success": torch.zeros((1, 0), dtype=torch.int8),
        },
        {
            "turnover_counts": turnover_counts,
            "birth_reason_counts": torch.zeros((1, 0, 10), dtype=torch.int64),
        },
        tuple(
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": generation,
                "world": "patch",
                "alive": 64,
                "births": 0,
                "failed_births": 0,
                "deaths": 0,
                "event_cursor": 0,
                "moved": 0,
                "gathered": 0,
            }
            for generation in range(4)
        ),
        _checkpoint(4, alive_count=64, generation_turnover_enabled=True),
        source_projection,
    )

    counters = receipt["counters"]
    capacity = receipt["capacity_accounting"]
    assert counters["admitted_births"] == 32
    assert counters["legacy_turnover_culls"] == 32
    assert counters["newborn_slot_reuses"] == 32
    assert counters["residual"] == 0
    assert capacity["admitted_births"] == 32
    assert capacity["ordinary_births_admitted"] == 0
    assert capacity["ordinary_birth_decisions_accepted"] == 0
    assert capacity["legacy_turnover_culls"] == 32


def test_legacy_qacct_reconciles_wd3_deaths_before_turnover_boundary() -> None:
    source = SimpleNamespace(
        generation_turnover_enabled=True,
        lifecycle_mode=0,
        genome_upgrade_v2_founder_profile_sha256="b" * 64,
        population_size=64,
        resident_capacity=256,
    )
    source_projection = {
        "metadata": {
            "generation": 0,
            "tick": 0,
            "population_size": 64,
            "capacity": 256,
        },
        "tensors": {"alive_mask": torch.ones(64, dtype=torch.bool)},
    }
    turnover_counts = torch.zeros((1, 8), dtype=torch.int64)
    turnover_counts[0, 0] = 57
    turnover_counts[0, 1] = 28
    turnover_counts[0, 4] = 28
    turnover_counts[0, 5] = 35
    turnover_counts[0, 6] = 35
    turnover_counts[0, 7] = 64

    receipt = settlement._lifecycle_qacct_receipt_from_route_v1(
        source,
        {
            "event_active_count": torch.tensor([19], dtype=torch.int64),
            "typed_lifecycle_event_event_type": torch.full(
                (1, 19), 4, dtype=torch.int32
            ),
            "typed_lifecycle_event_reason_code": torch.ones(
                (1, 19), dtype=torch.int32
            ),
            "typed_lifecycle_event_success": torch.ones((1, 19), dtype=torch.int8),
        },
        {
            "turnover_counts": turnover_counts,
            "birth_reason_counts": torch.zeros((1, 0, 10), dtype=torch.int64),
        },
        tuple(
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": generation,
                "world": "patch",
                "alive": 64,
                "births": 0,
                "failed_births": 0,
                "deaths": deaths,
                "event_cursor": cursor,
                "moved": 0,
                "gathered": 0,
            }
            for generation, deaths, cursor in (
                (0, 2, 2),
                (1, 7, 9),
                (2, 3, 12),
                (3, 7, 19),
            )
        ),
        _checkpoint(4, alive_count=64, generation_turnover_enabled=True),
        source_projection,
    )

    counters = receipt["counters"]
    capacity = receipt["capacity_accounting"]
    assert counters["residual"] == 0
    assert counters["ecological_deaths"] == 7
    assert counters["legacy_turnover_culls"] == 28
    assert counters["admitted_births"] == 35
    assert capacity["raw_ecological_death_events"] == 19
    assert capacity["effective_ecological_deaths"] == 7
    assert capacity["legacy_turnover_pre_live_count"] == 57
    assert capacity["legacy_turnover_ecological_deaths_reconciled"] == 12
    assert capacity["legacy_turnover_refill_committed"] == 35


def test_legacy_qacct_uses_compact_balance_when_ordinary_births_fill_capacity() -> None:
    source = SimpleNamespace(
        generation_turnover_enabled=True,
        lifecycle_mode=0,
        genome_upgrade_v2_founder_profile_sha256="b" * 64,
        population_size=256,
        resident_capacity=512,
    )
    source_projection = {
        "metadata": {
            "generation": 0,
            "tick": 0,
            "population_size": 256,
            "capacity": 512,
        },
        "tensors": {
            "alive_mask": torch.tensor(
                [*[True] * 256, *[False] * 256], dtype=torch.bool
            )
        },
    }
    birth_reason_counts = torch.zeros((1, 3 * 80, 10), dtype=torch.int64)
    for generation, births in enumerate((256, 256, 260)):
        birth_reason_counts[0, generation * 80, 0] = births
    turnover_counts = torch.zeros((1, 8), dtype=torch.int64)
    turnover_counts[0, 0] = 512
    turnover_counts[0, 1] = 256
    turnover_counts[0, 4] = 256
    turnover_counts[0, 7] = 256

    receipt = settlement._lifecycle_qacct_receipt_from_route_v1(
        source,
        {
            "event_active_count": torch.tensor([4], dtype=torch.int64),
            "typed_lifecycle_event_event_type": torch.full(
                (1, 4), 4, dtype=torch.int32
            ),
            "typed_lifecycle_event_reason_code": torch.ones(
                (1, 4), dtype=torch.int32
            ),
            "typed_lifecycle_event_success": torch.ones((1, 4), dtype=torch.int8),
        },
        {
            "turnover_counts": turnover_counts,
            "birth_reason_counts": birth_reason_counts,
        },
        tuple(
            {
                "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": generation,
                "world": "patch",
                "alive": alive,
                "births": births,
                "failed_births": 0,
                "deaths": deaths,
                "event_cursor": cursor,
                "moved": 0,
                "gathered": 0,
            }
            for generation, alive, births, deaths, cursor in (
                (0, 512, 256, 0, 0),
                (1, 512, 256, 0, 0),
                (2, 256, 260, 4, 4),
            )
        ),
        _checkpoint(3, alive_count=256, generation_turnover_enabled=True),
        source_projection,
    )

    counters = receipt["counters"]
    capacity = receipt["capacity_accounting"]
    assert counters["residual"] == 0
    assert counters["ecological_deaths"] == 4
    assert counters["legacy_turnover_culls"] == 768
    assert counters["admitted_births"] == 772
    assert capacity["ordinary_births_admitted"] == 772
    assert capacity["legacy_turnover_refill_committed"] == 0


def test_hybrid_qacct_rejects_stale_legacy_final_live_counter() -> None:
    source = SimpleNamespace(
        generation_turnover_enabled=False,
        lifecycle_mode=1,
        genome_upgrade_v2_founder_profile_sha256="b" * 64,
    )
    source_projection = {
        "metadata": {
            "generation": 4,
            "tick": 0,
            "population_size": 64,
            "capacity": 256,
        },
        "tensors": {"alive_mask": torch.ones(256, dtype=torch.bool)},
    }
    turnover_counts = torch.zeros((1, 8), dtype=torch.int64)
    turnover_counts[0, 7] = 256

    with pytest.raises(
        settlement.CudaArchitectureBDirectResidentSettlementError,
        match="inactive final-live",
    ):
        settlement._lifecycle_qacct_receipt_from_route_v1(
            source,
            {},
            {
                "turnover_counts": turnover_counts,
                "birth_reason_counts": torch.zeros((1, 80, 10), dtype=torch.int64),
            },
            (
                {
                    "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                    "generation": 4,
                    "world": "patch",
                    "alive": 256,
                    "births": 0,
                    "failed_births": 0,
                    "deaths": 0,
                    "event_cursor": 1,
                    "moved": 0,
                    "gathered": 0,
                },
            ),
            _checkpoint(8),
            source_projection,
        )


def test_hybrid_qacct_rejects_checkpoint_live_end_disagreement() -> None:
    source = SimpleNamespace(
        generation_turnover_enabled=False,
        lifecycle_mode=1,
        genome_upgrade_v2_founder_profile_sha256="b" * 64,
    )
    source_projection = {
        "metadata": {
            "generation": 4,
            "tick": 0,
            "population_size": 64,
            "capacity": 256,
        },
        "tensors": {"alive_mask": torch.ones(256, dtype=torch.bool)},
    }

    with pytest.raises(
        settlement.CudaArchitectureBDirectResidentSettlementError,
        match="live_end contradicts resident identity",
    ):
        settlement._lifecycle_qacct_receipt_from_route_v1(
            source,
            {},
            {
                "turnover_counts": torch.zeros((1, 8), dtype=torch.int64),
                "birth_reason_counts": torch.zeros((1, 80, 10), dtype=torch.int64),
            },
            (
                {
                    "schema_version": settlement.COMPACT_SUMMARY_SCHEMA_VERSION,
                    "generation": 4,
                    "world": "patch",
                    "alive": 256,
                    "births": 0,
                    "failed_births": 0,
                    "deaths": 0,
                    "event_cursor": 1,
                    "moved": 0,
                    "gathered": 0,
                },
            ),
            _checkpoint(8, alive_count=255),
            source_projection,
        )


def test_settlement_rejects_lifecycle_mode_turnover_mismatch() -> None:
    context = _context()
    context.source.lifecycle_mode = 0

    with pytest.raises(
        settlement.CudaArchitectureBDirectResidentSettlementError,
        match="lifecycle_mode contradicts generation_turnover_enabled",
    ):
        settlement.settle_patch_p2_b1_direct_resident_prefix_v2(
            context,
            direct_result=_direct_result(context),
            capability=_capability(),
        )


@pytest.mark.parametrize("logical_generation_count", (7, 14, 37))
@pytest.mark.parametrize("ticks_per_generation", (80, 160))
def test_v2_settlement_projects_irregular_qualified_horizons(
    monkeypatch: pytest.MonkeyPatch,
    logical_generation_count: int,
    ticks_per_generation: int,
) -> None:
    source_generation = 5
    context = _context(
        source_generation,
        logical_generation_count=logical_generation_count,
        ticks_per_generation=ticks_per_generation,
    )
    terminal_generation = source_generation + logical_generation_count
    projected: list[dict[str, Any]] = []

    def fake_project(**kwargs: Any) -> CudaBoundaryCheckpoint:
        projected.append(kwargs)
        return _checkpoint(terminal_generation)

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )
    result = settlement.settle_patch_p2_b1_direct_resident_prefix_v2(
        context,
        direct_result=_direct_result(context),
        capability=_capability(),
    )

    assert result.terminal_generation == terminal_generation
    assert len(result.compact_summary_rows) == logical_generation_count
    assert [row["generation"] for row in result.compact_summary_rows] == list(
        range(source_generation, terminal_generation)
    )
    assert projected[0]["completed_generation_count"] == logical_generation_count
    assert int(projected[0]["generation_result"].applied_actions.item()) == (
        logical_generation_count * ticks_per_generation
    )


@pytest.mark.parametrize("source_generation", (4, 8))
def test_v2_settlement_projects_fresh_absolute_generation_boundary(
    monkeypatch: pytest.MonkeyPatch,
    source_generation: int,
) -> None:
    context = _context(source_generation)
    terminal_generation = source_generation + 1
    context.source.outputs["turnover_next_generation"] = torch.tensor(
        [terminal_generation], dtype=torch.int64
    )
    projected: list[dict[str, Any]] = []

    def fake_project(**kwargs: Any) -> CudaBoundaryCheckpoint:
        projected.append(kwargs)
        return _checkpoint(terminal_generation)

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )
    result = settlement.settle_patch_p2_b1_direct_resident_prefix_v2(
        context,
        direct_result=_direct_result(
            context,
            completed_generations=1,
            settled_window_count=1,
            generation_summary_scalar=context.source.outputs[
                "generation_summary_scalar"
            ][:1],
            runtime_bookkeeping_by_tick=context.source.outputs[
                "runtime_bookkeeping_by_tick"
            ][:, : settlement.TICKS_PER_GENERATION],
        ),
        capability=_capability(),
    )

    assert result.terminal_generation == terminal_generation
    assert [row["generation"] for row in result.compact_summary_rows] == [
        source_generation
    ]
    assert len(projected) == 1
    assert (
        projected[0]["pre_generation_canonical_projection"]["metadata"]["generation"]
        == source_generation
    )
    assert projected[0]["completed_generation_count"] == 1


def test_v2_settlement_accepts_g1_physical_window_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(logical_generation_count=4, physical_window_generations=1)
    terminal_generation = 4
    projected: list[dict[str, Any]] = []

    def fake_project(**kwargs: Any) -> CudaBoundaryCheckpoint:
        projected.append(kwargs)
        return _checkpoint(terminal_generation)

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )
    result = settlement.settle_patch_p2_b1_direct_resident_prefix_v2(
        context,
        direct_result=_direct_result(context),
        capability=_capability(),
    )

    assert result.terminal_generation == terminal_generation
    assert len(result.compact_summary_rows) == 4
    assert projected[0]["completed_generation_count"] == 4


def test_v2_settlement_accepts_g1_result_when_source_window_metadata_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(logical_generation_count=4, physical_window_generations=1)
    result_payload = _direct_result(context)
    delattr(context.source, "physical_window_generations")
    projected: list[dict[str, Any]] = []

    def fake_project(**kwargs: Any) -> CudaBoundaryCheckpoint:
        projected.append(kwargs)
        return _checkpoint(4)

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )
    result = settlement.settle_patch_p2_b1_direct_resident_prefix_v2(
        context,
        direct_result=result_payload,
        capability=_capability(),
    )

    assert result.terminal_generation == 4
    assert projected[0]["completed_generation_count"] == 4


def test_v2_settlement_accepts_large_patch_source_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(4)
    context.checkpoint_projection_inputs["pre_generation_canonical_projection"][
        "metadata"
    ].update(
        {
            "population_size": 256,
            "capacity": 512,
        }
    )
    context.checkpoint_projection_inputs["pre_generation_canonical_projection"][
        "tensors"
    ]["alive_mask"] = torch.cat(
        (
            torch.ones(256, dtype=torch.bool),
            torch.zeros(256, dtype=torch.bool),
        )
    )
    alive_after_turnover = settlement.GENERATION_SUMMARY_COLUMNS.index(
        "alive_after_turnover"
    )
    context.source.outputs["generation_summary_scalar"][0, 0, alive_after_turnover] = (
        256
    )
    terminal_generation = 5
    projected: list[dict[str, Any]] = []

    def fake_project(**kwargs: Any) -> CudaBoundaryCheckpoint:
        projected.append(kwargs)
        return _checkpoint(terminal_generation, alive_count=256)

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )
    result = settlement.settle_patch_p2_b1_direct_resident_prefix_v2(
        context,
        direct_result=_direct_result(
            context,
            completed_generations=1,
            settled_window_count=1,
            generation_summary_scalar=context.source.outputs[
                "generation_summary_scalar"
            ][:1],
            runtime_bookkeeping_by_tick=context.source.outputs[
                "runtime_bookkeeping_by_tick"
            ][:, : settlement.TICKS_PER_GENERATION],
        ),
        capability=_capability(),
    )

    assert result.terminal_generation == terminal_generation
    assert projected[0]["pre_generation_canonical_projection"]["metadata"][
        "population_size"
    ] == 256
    assert projected[0]["pre_generation_canonical_projection"]["metadata"][
        "capacity"
    ] == 512


def test_settlement_rejects_source_generation_metadata_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(4)
    context.checkpoint_projection_inputs["pre_generation_canonical_projection"][
        "metadata"
    ]["generation"] = 3
    calls = 0

    def fake_project(**_kwargs: Any) -> CudaBoundaryCheckpoint:
        nonlocal calls
        calls += 1
        return _checkpoint(8)

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )

    with pytest.raises(
        settlement.CudaArchitectureBDirectResidentSettlementError,
        match="source-generation canonical boundary",
    ):
        settlement.settle_patch_p2_b1_g4_direct_resident_v1(
            context,
            direct_result=_direct_result(context),
            capability=_capability(),
        )
    assert calls == 0


@pytest.mark.parametrize(
    ("checkpoint_generation", "authority_generation"),
    ((4, 8), (8, 4)),
)
def test_settlement_rejects_nonabsolute_projected_checkpoint_boundary(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_generation: int,
    authority_generation: int,
) -> None:
    context = _context(4)
    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        lambda **_kwargs: _checkpoint(
            checkpoint_generation,
            authority_generation=authority_generation,
        ),
    )

    with pytest.raises(
        settlement.CudaArchitectureBDirectResidentSettlementError,
        match="exact completed-prefix boundary",
    ):
        settlement.settle_patch_p2_b1_g4_direct_resident_v1(
            context,
            direct_result=_direct_result(context),
            capability=_capability(),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_fallback": True}, "hidden_fallback mismatch"),
        ({"runtime_jit": True}, "runtime_jit mismatch"),
        ({"logical_generation_count": 8}, "logical_generation_count mismatch"),
    ],
)
def test_settlement_rejects_nonproduction_direct_results(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        lambda **_kwargs: _checkpoint(),
    )

    context = _context()
    with pytest.raises(
        settlement.CudaArchitectureBDirectResidentSettlementError,
        match=message,
    ):
        settlement.settle_patch_p2_b1_g4_direct_resident_v1(
            context,
            direct_result=_direct_result(context, **overrides),
            capability=_capability(),
        )


def test_settlement_rejects_nonterminal_generation_before_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    context.source.generation_turnover_enabled = True
    context.source.lifecycle_mode = 0
    context.source.outputs["turnover_next_generation"] = torch.tensor(
        [3], dtype=torch.int64
    )
    calls = 0

    def fake_project(**_kwargs: Any) -> CudaBoundaryCheckpoint:
        nonlocal calls
        calls += 1
        return _checkpoint()

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )

    with pytest.raises(
        settlement.CudaArchitectureBDirectResidentSettlementError,
        match="complete prefix boundary",
    ):
        settlement.settle_patch_p2_b1_g4_direct_resident_v1(
            context,
            direct_result=_direct_result(context),
            capability=_capability(),
        )
    assert calls == 0


def test_settlement_rejects_distinct_live_pred_lr_before_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    context.live_banks["genome_pred_lr"][0, 1] = 0.25
    calls = 0

    def fake_project(**_kwargs: Any) -> CudaBoundaryCheckpoint:
        nonlocal calls
        calls += 1
        return _checkpoint()

    monkeypatch.setattr(
        settlement,
        "project_post_generation_cuda_checkpoint",
        fake_project,
    )

    with pytest.raises(
        settlement.CudaArchitectureBDirectResidentSettlementError,
        match="distinct live runtime pred_lr",
    ):
        settlement.settle_patch_p2_b1_g4_direct_resident_v1(
            context,
            direct_result=_direct_result(context),
            capability=_capability(),
        )
    assert calls == 0
