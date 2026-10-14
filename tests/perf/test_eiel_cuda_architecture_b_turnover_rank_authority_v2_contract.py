from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TURNOVER_SOURCE = (
    ROOT / "engine/perf/cuda_architecture_b_flattened_turnover_v1_kernels.cuh"
)
RANKED_ENTRY_POINT = (
    "commit_turnover_fixed_mlp_child_scalar_genome_"
    "with_ranked_inheritance_v2_serial"
)


def _balanced_call(source: str, name: str) -> str:
    start = source.index(name)
    open_paren = source.index("(", start + len(name))
    depth = 0
    for index in range(open_paren, len(source)):
        token = source[index]
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated call to {name}")


def test_turnover_rank_authority_is_bound_to_the_scalar_redraw_transaction() -> None:
    source = TURNOVER_SOURCE.read_text(encoding="utf-8")
    call = _balanced_call(source, RANKED_ENTRY_POINT)

    assert "FlatTurnoverWorkLayout::kParentPoolSlots" in call
    assert "FlatTurnoverWorkLayout::kCommittedChildSlots" in call
    assert "FlatTurnoverWorkLayout::kRefillParentSlots" not in call
    assert "FlatTurnoverWorkLayout::kRefillMateSlots" not in call

    required_rank_authority = (
        "cross_control.genotype_codes",
        "+ cell * execution.capacity",
        "cross_control.digest_rank_by_code",
        "+ cell * cross_control.genotype_count",
        "cross_control.genotype_count",
        "cross_control.canonical_parent_0_slots",
        "+ inheritance_row_base",
        "cross_control.canonical_parent_1_slots",
    )
    for fragment in required_rank_authority:
        assert fragment in call
    assert (
        "{\n            planned_birth_count,\n            execution.capacity,"
        in call
    )

    staging_start = source.index(
        "const int64_t inheritance_row_base", 0, source.index(RANKED_ENTRY_POINT)
    )
    staging = source[staging_start : source.index(RANKED_ENTRY_POINT)]
    assert "FlatTurnoverWorkLayout::kRefillParentSlots" not in staging
    assert "FlatTurnoverWorkLayout::kRefillMateSlots" not in staging
    assert "parent_rank" not in staging
    assert "mate_rank" not in staging


def test_turnover_sidecar_capacity_is_the_transaction_bound_not_slot_capacity() -> None:
    source = TURNOVER_SOURCE.read_text(encoding="utf-8")
    staging_start = source.index("const int64_t planned_birth_count")
    staging_end = source.index(RANKED_ENTRY_POINT, staging_start)
    staging = source[staging_start:staging_end]

    assert (
        "cross_control.inheritance_rows_per_cell < planned_birth_count"
        in staging
    )
    assert (
        "scalar::kStatusInheritanceSidecarCapacityExceeded"
        in staging
    )
    assert (
        "birth < cross_control.inheritance_rows_per_cell"
        in staging
    )
    assert (
        "cross_control.inheritance_rows_per_cell < execution.capacity"
        not in staging
    )


def test_disabled_turnover_settlement_mirrors_resident_live_identity() -> None:
    source = TURNOVER_SOURCE.read_text(encoding="utf-8")
    publish = source.index("flattened_turnover_publish_kernel_v1")
    start = source.index(
        "if (!turnover_cell_enabled_v1(config, cell))",
        publish,
    )
    end = source.index("const int64_t lanes", start)
    disabled = source[start:end]

    assert "output_counts[kOutputFinalLiveCount]" in disabled
    assert "tensors.source<int64_t>(kActiveCountState, cell)[0]" in disabled
    assert "outputs.status[cell] = 0" in disabled
