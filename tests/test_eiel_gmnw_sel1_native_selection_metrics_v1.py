from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from engine.perf import cuda_architecture_b_direct_resident_gmnw_gru_v1 as subject


ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / "engine" / "perf"
CELLS = 3
GENERATIONS = 4
FAMILIES = 2


def _allocate(
    initial: dict[str, torch.Tensor] | None = None,
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    return subject._allocate_gmnw_selection_metrics_v1(
        cells=CELLS,
        generations=GENERATIONS,
        device=torch.device("cpu"),
        initial=initial,
    )


def test_optional_sidecar_has_exact_packed_layout_and_named_family_views() -> None:
    tensors, metrics = _allocate()

    assert subject.GMNW_SELECTION_METRICS_VERSION_V1 == 1
    assert subject.GMNW_SELECTION_POLICY_FAMILIES_V1 == ("mlp", "gru")
    assert subject.GMNW_SELECTION_COUNT_METRIC_NAMES_V1 == (
        "alive_pre_turnover",
        "alive_post_turnover",
        "decision_count",
        "correct_count",
        "task_death_count",
        "unique_selected_parent_pool_count",
        "committed_turnover_offspring_count",
    )
    assert subject.GMNW_SELECTION_METRIC_NAMES_V1 == (
        *subject.GMNW_SELECTION_COUNT_METRIC_NAMES_V1,
        "recurrent_upkeep_charge_total",
    )
    assert len(tensors) == 3
    counts, upkeep, status = tensors
    assert counts.shape == (CELLS, GENERATIONS, FAMILIES, 7)
    assert counts.dtype == torch.int64
    assert upkeep.shape == (CELLS, GENERATIONS, FAMILIES)
    assert upkeep.dtype == torch.float64
    assert status.shape == (CELLS,)
    assert status.dtype == torch.int32
    assert set(metrics) == {*subject.GMNW_SELECTION_METRIC_NAMES_V1, "status"}
    for column, name in enumerate(subject.GMNW_SELECTION_COUNT_METRIC_NAMES_V1):
        assert metrics[name].data_ptr() == counts[..., column].data_ptr()
        assert metrics[name].shape == (CELLS, GENERATIONS, FAMILIES)
    assert metrics["recurrent_upkeep_charge_total"] is upkeep
    assert metrics["status"] is status


def test_base_gmnw_inventory_has_23_tensors_with_legacy_venv_profile() -> None:
    tensors = subject._allocate_gmnw_tensors(
        cells=2,
        capacity=5,
        generations=4,
        config=subject.GmnwConfigurationV1(),
        device=torch.device("cpu"),
    )

    assert len(tensors) == 23
    assert tensors[14].shape == (2, 4, 6)
    assert tensors[14].dtype == torch.int64
    assert tensors[16].shape == (2,)
    assert tensors[16].dtype == torch.int32
    assert tensors[22].shape == (2,)
    assert tensors[22].dtype == torch.int32
    assert not torch.count_nonzero(tensors[22])


def test_continuation_rows_are_copied_exactly_without_aliasing() -> None:
    _, original = _allocate()
    for ordinal, name in enumerate(subject.GMNW_SELECTION_COUNT_METRIC_NAMES_V1, 1):
        original[name].copy_(
            torch.arange(CELLS * GENERATIONS * FAMILIES, dtype=torch.int64).reshape(
                CELLS, GENERATIONS, FAMILIES
            )
            + ordinal * 100
        )
    original["recurrent_upkeep_charge_total"].copy_(
        torch.arange(CELLS * GENERATIONS * FAMILIES, dtype=torch.float64).reshape(
            CELLS, GENERATIONS, FAMILIES
        )
        / 8.0
    )

    _, restored = _allocate(original)

    for name in (*subject.GMNW_SELECTION_METRIC_NAMES_V1, "status"):
        assert torch.equal(restored[name], original[name])
        assert restored[name].data_ptr() != original[name].data_ptr()
    original["alive_pre_turnover"].zero_()
    assert bool(torch.count_nonzero(restored["alive_pre_turnover"]).item())


@pytest.mark.parametrize(
    ("name", "replacement", "message"),
    (
        (
            "decision_count",
            torch.zeros((CELLS, GENERATIONS, FAMILIES), dtype=torch.int32),
            "decision_count must be int64",
        ),
        (
            "correct_count",
            torch.zeros((CELLS, GENERATIONS, 1), dtype=torch.int64),
            "correct_count must be int64",
        ),
        (
            "recurrent_upkeep_charge_total",
            torch.zeros((CELLS, GENERATIONS, FAMILIES), dtype=torch.float32),
            "recurrent_upkeep_charge_total must be float64",
        ),
        (
            "status",
            torch.zeros((CELLS, 1), dtype=torch.int32),
            "status must be int32",
        ),
    ),
)
def test_continuation_rejects_dtype_and_shape_drift(
    name: str,
    replacement: torch.Tensor,
    message: str,
) -> None:
    _, initial = _allocate()
    initial[name] = replacement

    with pytest.raises(
        subject.CudaArchitectureBDirectResidentGmnwGruV1Error, match=message
    ):
        _allocate(initial)


def test_continuation_rejects_nonzero_status_and_field_drift() -> None:
    _, initial = _allocate()
    initial["status"][1] = 7
    with pytest.raises(
        subject.CudaArchitectureBDirectResidentGmnwGruV1Error,
        match="continuation status is nonzero",
    ):
        _allocate(initial)

    _, initial = _allocate()
    initial["ordinary_birth_count"] = torch.zeros(
        (CELLS, GENERATIONS, FAMILIES), dtype=torch.int64
    )
    with pytest.raises(
        subject.CudaArchitectureBDirectResidentGmnwGruV1Error,
        match="continuation fields mismatch",
    ):
        _allocate(initial)


def test_result_mapping_requires_version_and_clean_status() -> None:
    _, metrics = _allocate()
    prepared = SimpleNamespace(selection_metrics=metrics)
    result = {"gmnw_selection_metrics_version": 1}

    subject._attach_gmnw_selection_metrics_v1(result, prepared)

    assert result["gmnw_selection_metric_families"] == ("mlp", "gru")
    assert list(result["gmnw_selection_metrics"]) == list(
        subject.GMNW_SELECTION_METRIC_NAMES_V1
    )
    assert "status" not in result["gmnw_selection_metrics"]

    metrics["status"][0] = 1
    with pytest.raises(
        subject.CudaArchitectureBDirectResidentGmnwGruV1Error,
        match="device status is nonzero",
    ):
        subject._attach_gmnw_selection_metrics_v1(
            {"gmnw_selection_metrics_version": 1}, prepared
        )


def test_cuda_source_hooks_are_authoritative_reporting_only_and_append_only() -> None:
    controller = (
        PERF / "cuda_architecture_b_flattened_gmnw_controller_v1.hpp"
    ).read_text(encoding="utf-8")
    persistent = (PERF / "cuda_architecture_b_persistent_gmnw_v1.cuh").read_text(
        encoding="utf-8"
    )
    kernels = (
        PERF / "cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"
    ).read_text(encoding="utf-8")
    binding = (
        PERF / "cuda_architecture_b_flattened_wdl_controller_v2_binding.cpp"
    ).read_text(encoding="utf-8")

    assert "kCounterColumnsV1 = 6" in controller
    assert "kSelectionPolicyFamilyCountV1 = 2" in controller
    assert "kSelectionCountMetricCountV1 = 7" in controller
    assert "GmnwSelectionMetricsContextV1 selection_metrics;" in controller
    assert "const bool policy_use_gru" in persistent
    assert "kSelectionDecisionCountV1" in persistent
    assert "kSelectionCorrectCountV1" in persistent
    assert "kSelectionTaskDeathCountV1" in persistent

    assert "ordinary_birth_state.policy_use_gru[slot_index]" in kernels
    assert "record_gmnw_selection_upkeep_v1" in kernels
    assert "record_gmnw_selection_pre_turnover_v1" in kernels
    assert "record_gmnw_selection_post_turnover_kernel_v1" in kernels
    assert "kOutputParentPoolCount" in kernels
    assert "kOutputChildCommittedCount" in kernels
    assert "kMixedSelectionImmediateArmV1" in kernels
    assert (
        "gmnw_policy_arm\n                    > gmnw_controls::kMixedSelectionImmediateArmV1"
        in kernels
    )

    assert "23U\n        + (has_control_tensors ? 4U : 0U)" in binding
    assert "+ (has_selection_metrics_tensors ? 3U : 0U)" in binding
    assert "GMNW counters must be int64[cells,generations,6]" in binding
    assert "GMNW selection counts must be int64[cells,generations,2,7]" in binding
    assert "selection_metrics_version" in binding


def test_only_the_eight_required_metrics_are_exposed() -> None:
    assert len(subject.GMNW_SELECTION_METRIC_NAMES_V1) == 8
    assert "ordinary_birth_count" not in subject.GMNW_SELECTION_METRIC_NAMES_V1
    assert "failed_birth_count" not in subject.GMNW_SELECTION_METRIC_NAMES_V1
