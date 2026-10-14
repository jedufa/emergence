from __future__ import annotations

import copy
import dataclasses
import json
import runpy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from engine.native_runtime import compact_summary_cpu as cpu
from engine.native_runtime import compact_summary_planes as abi
from engine.native_runtime import compact_summary_projection as projection
from engine.native_runtime.compact_summary_authority_adapter import (
    assemble_compact_summary_source,
)
from simulation.metrics_pkg.lineage import LineageMetricsState
from tools.scripts import (
    eiel_cuda_architecture_b_compact_summary_flat_g2_gate as gate,
)


ROOT = Path(__file__).resolve().parents[1]
CPU_FIXTURES = runpy.run_path(str(ROOT / "tests/test_eiel_compact_summary_cpu.py"))
ADAPTER_FIXTURES = runpy.run_path(
    str(ROOT / "tests/test_eiel_compact_summary_authority_adapter.py")
)


def _source(generation: int, *, final: bool) -> dict[str, Any]:
    source = CPU_FIXTURES["_source"](world="flat")
    source["pre_turnover"]["generation_start"]["generation"] = generation
    source["pre_turnover"]["generation_end"]["generation"] = generation
    source["post_turnover"]["next_generation_start"]["generation"] = generation + 1
    source["lineage"]["generation"] = generation
    for row in source["behavior"]["rows"]:
        row["generation"] = generation
    source["settled"]["terminal"]["status"] = (
        "run_complete" if final else "not_terminal"
    )
    return source


def _rows_and_sources(
    generation_count: int = gate.GENERATION_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = [
        _source(generation, final=generation == generation_count - 1)
        for generation in range(generation_count)
    ]
    return [cpu.build_compact_summary_row(source) for source in sources], sources


def _planes(
    generation_count: int = gate.GENERATION_COUNT,
    *,
    generation_capacity: int | None = None,
) -> abi.CompactSummaryPlanes:
    rows, _sources = _rows_and_sources(generation_count)
    if generation_capacity is None:
        generation_capacity = max(4, generation_count)
    planes = abi.allocate_compact_summary_planes(generation_capacity, 1)
    for generation, row in enumerate(rows):
        projection.encode_compact_summary_row(
            row,
            planes,
            generation=generation,
            cell=0,
        )
    return planes


def _clone_planes(planes: abi.CompactSummaryPlanes) -> abi.CompactSummaryPlanes:
    return abi.CompactSummaryPlanes(
        **{name: value.copy() for name, value in planes.as_mapping().items()}
    )


def _stored_adapter_inputs() -> Any:
    """Keep the accepted adapter fixture current with the live digest schema."""

    fixture = ADAPTER_FIXTURES["_inputs"]
    namespace = fixture.__globals__
    original_projection = namespace["_projection"]

    def projection_with_runtime_cursor(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = original_projection(*args, **kwargs)
        value.setdefault("runtime_event_cursor", value["event_cursor"])
        return value

    namespace["_projection"] = projection_with_runtime_cursor
    try:
        return fixture(world="flat")
    finally:
        namespace["_projection"] = original_projection


def _checkpoint() -> dict[str, Any]:
    inputs = _stored_adapter_inputs()
    return copy.deepcopy(
        inputs.run_result["boundaries"][0]["post_turnover_checkpoint_projection"]
    )


@dataclasses.dataclass(frozen=True)
class _FakeCpuAuthorityInput:
    run_result: dict[str, Any]
    boundary_index: int
    lineage_state_before: LineageMetricsState
    source: dict[str, Any]


class _FakeCpuRunner:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[int] = []

    def run_many(self, generation_count: int) -> dict[str, Any]:
        self.calls.append(generation_count)
        return self.result


def _cpu_run_many_result(
    generation_count: int = gate.GENERATION_COUNT,
) -> dict[str, Any]:
    checkpoint = _checkpoint()
    return {
        "schema_version": "eiel_native_architecture_b_run_result_v1",
        "terminal_status": "run_complete",
        "executed": True,
        "native_owner": True,
        "hidden_fallback": False,
        "runtime_jit": False,
        "start_generation": 0,
        "requested_generations": generation_count,
        "completed_generations": generation_count,
        "controller_invocations": 1,
        "python_generation_dispatches": 0,
        "native_generation_transactions": generation_count,
        "turnover_commits": generation_count,
        "boundaries": [
            {
                "completed_generation": generation,
                "next_generation": generation + 1,
                "generation_result": {
                    "generation_index": generation,
                    "executed": True,
                    "hidden_fallback": False,
                    "native_owner": True,
                },
                "turnover_result": {
                    "executed": True,
                    "hidden_fallback": False,
                    "native_owner": True,
                },
                "post_turnover_checkpoint_projection": copy.deepcopy(checkpoint),
            }
            for generation in range(generation_count)
        ],
        "final_checkpoint_projection": copy.deepcopy(checkpoint),
    }


def _advance_fake_lineage_state(inputs: _FakeCpuAuthorityInput) -> LineageMetricsState:
    state = LineageMetricsState.from_dict(inputs.lineage_state_before.to_dict())
    state.collapse_streak += 1
    state.seen_lineages.add(f"generation-{inputs.boundary_index}")
    return state


def _cpu_oracle_once(
    *,
    mutate_sources: Any = None,
    generation_count: int = gate.GENERATION_COUNT,
) -> tuple[gate.FlatG2CpuOracleRun, _FakeCpuRunner, dict[str, Any]]:
    _rows, sources = _rows_and_sources(generation_count)
    sources = copy.deepcopy(sources)
    if mutate_sources is not None:
        mutate_sources(sources)
    runner = _FakeCpuRunner(_cpu_run_many_result(generation_count))
    observations: dict[str, Any] = {
        "builder_generations": [],
        "builder_state_before": [],
        "adapter_generations": [],
    }

    def authority_input_builder(
        run_result: dict[str, Any],
        generation: int,
        state_before: LineageMetricsState,
    ) -> _FakeCpuAuthorityInput:
        observations["builder_generations"].append(generation)
        observations["builder_state_before"].append(state_before.to_dict())
        return _FakeCpuAuthorityInput(
            run_result=run_result,
            boundary_index=generation,
            lineage_state_before=state_before,
            source=sources[generation],
        )

    def authority_adapter(inputs: _FakeCpuAuthorityInput) -> dict[str, Any]:
        observations["adapter_generations"].append(inputs.boundary_index)
        return inputs.source

    oracle = gate._execute_cpu_oracle_once(
        runner,
        authority_input_builder=authority_input_builder,
        initial_lineage_state=LineageMetricsState(),
        authority_adapter=authority_adapter,
        row_builder=cpu.build_compact_summary_row,
        lineage_state_advance=_advance_fake_lineage_state,
        generation_count=generation_count,
    )
    return oracle, runner, observations


def _cpu_producer(
    mutations: list[Any] | None = None,
    *,
    generation_count: int = gate.GENERATION_COUNT,
) -> tuple[Any, list[dict[str, Any]]]:
    resolved = mutations or [None, None]
    calls: list[dict[str, Any]] = []

    def produce() -> gate.FlatG2CpuOracleRun:
        mutation = resolved[len(calls)]
        oracle, runner, observations = _cpu_oracle_once(
            mutate_sources=mutation,
            generation_count=generation_count,
        )
        calls.append(
            {
                "runner_calls": runner.calls,
                "observations": observations,
                "oracle": oracle,
            }
        )
        return oracle

    return produce, calls


def _receipt(
    generation_count: int = gate.GENERATION_COUNT,
) -> dict[str, Any]:
    return {
        "native_controller_receipt": {
            "native_dispatches": 1,
            "native_tick_iterations": generation_count * gate.TICKS_PER_GENERATION,
            "requested_generations": generation_count,
            "completed_generations": generation_count,
            "completed_generations_by_cell": [generation_count],
            "python_generation_iterations": 0,
            "python_tick_iterations": 0,
            "aten_operations_inside_tick_loop": 0,
            "runtime_jit": False,
            "hidden_fallback": False,
        },
        "aten_dispatches_in_extension_call": 0,
        "aten_allocations_in_extension_call": 0,
        "outputs_preallocated_before_timing": True,
        "scratch_preallocated_before_timing": True,
        "torch_allocated_delta_bytes": 0,
        "torch_peak_allocated_delta_bytes": 0,
        "pointer_stability_receipt": {
            "all_exact": True,
            "pointer_count": 12,
            "first_mismatch": None,
            "mismatches": [],
        },
        "compact_summary_planes_enabled": True,
        "compact_summary_rows_status": "retained_complete_unqualified",
        "summary_projection_mutates_simulation_state": False,
        "summary_projection_consumes_rng": False,
    }


def _capability() -> dict[str, Any]:
    return {
        "compact_summary_native_producer_status": "retained_complete_unqualified",
        "resident_generation_summaries_supported": False,
        "eiel_compact_generation_summary_v1_supported": False,
    }


def _run(
    *,
    generation_count: int = gate.GENERATION_COUNT,
    planes: abi.CompactSummaryPlanes | None = None,
    receipt: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
    state_after: Any | None = None,
    rng_after: Any | None = None,
) -> gate.FlatG2RetainedRun:
    state = {"authoritative_state_sha256": "1" * 64}
    rng = {"rng_authority_sha256": "2" * 64, "cursor": 9}
    return gate.FlatG2RetainedRun(
        compact_summary_planes=(
            _planes(generation_count) if planes is None else planes
        ),
        final_checkpoint_projection=_checkpoint() if checkpoint is None else checkpoint,
        receipt=_receipt(generation_count) if receipt is None else receipt,
        summary_state_before=state,
        summary_state_after=copy.deepcopy(state) if state_after is None else state_after,
        rng_state_before=rng,
        rng_state_after=copy.deepcopy(rng) if rng_after is None else rng_after,
        process_identity="stored-fixture-process",
    )


def _producer(runs: list[gate.FlatG2RetainedRun]) -> tuple[Any, list[int]]:
    iterator: Iterator[gate.FlatG2RetainedRun] = iter(runs)
    calls: list[int] = []

    def produce() -> gate.FlatG2RetainedRun:
        calls.append(len(calls))
        return next(iterator)

    return produce, calls


def test_stored_authority_adapter_fixture_is_consumed_read_only() -> None:
    first = _stored_adapter_inputs()
    second = _stored_adapter_inputs()

    rows, receipt = gate._expected_rows(
        [first, second],
        authority_adapter=assemble_compact_summary_source,
        row_builder=cpu.build_compact_summary_row,
    )

    assert len(rows) == 2
    assert all(row["summary_valid"] is True for row in rows)
    assert all(len(row["rng"]["authority_digest_sha256"]) == 64 for row in rows)
    assert receipt["read_only"] is True


def test_cpu_oracle_uses_run_many_g2_and_carries_lineage_state() -> None:
    oracle, runner, observations = _cpu_oracle_once()

    assert runner.calls == [2]
    assert observations["builder_generations"] == [0, 1]
    assert observations["adapter_generations"] == [0, 1]
    assert [
        row["generation_end"]["generation"] for row in oracle.canonical_rows
    ] == [0, 1]
    assert observations["builder_state_before"][0]["collapse_streak"] == 0
    assert observations["builder_state_before"][1]["collapse_streak"] == 1
    assert oracle.lineage_state_after.collapse_streak == 2
    flow = oracle.provenance["lineage_state_flow"]
    assert flow[0]["state_after_sha256"] == flow[1]["state_before_sha256"]
    assert oracle.provenance["authority_kind"] == "independent_cpu_oracle"
    assert oracle.provenance["execution_call"] == "run_many(2)"
    assert oracle.provenance["candidate_rows_consumed"] is False
    assert oracle.provenance["candidate_planes_consumed"] is False
    assert oracle.provenance["run_many_result_read_only"] is True


def test_cpu_oracle_rejects_any_generation_sequence_other_than_zero_one() -> None:
    result = _cpu_run_many_result()
    result["start_generation"] = 3
    runner = _FakeCpuRunner(result)

    with pytest.raises(ValueError, match="start_generation must be 0"):
        gate._execute_cpu_oracle_once(
            runner,
            authority_input_builder=lambda *_args: None,
            initial_lineage_state=LineageMetricsState(),
            authority_adapter=lambda value: value,
            row_builder=lambda value: value,
            lineage_state_advance=lambda _value: LineageMetricsState(),
        )

    assert runner.calls == [2]


def test_cpu_oracle_repeat_exposes_first_field_mismatch_before_candidate() -> None:
    def change_generation_one(sources: list[dict[str, Any]]) -> None:
        sources[1]["pre_turnover"]["generation_metrics"]["moved"] += 1

    cpu_producer, cpu_calls = _cpu_producer([None, change_generation_one])
    producer, candidate_calls = _producer([_run(), _run()])

    payload = gate.execute_live_gate(
        capability=_capability(),
        producer=producer,
        cpu_oracle_producer=cpu_producer,
    )

    assert payload["ok"] is False
    assert payload["decision"] == "flat_g2_cpu_oracle_repeat_rejected"
    assert len(cpu_calls) == 2
    assert candidate_calls == []
    assert payload["first_mismatch"]["authority"] == "cpu_oracle_repeat"
    assert payload["first_mismatch"]["generation_index"] == 1
    assert payload["first_mismatch"]["path"].endswith(
        ".generation_metrics.moved"
    )


def test_candidate_retention_has_no_expected_authority_input_surface() -> None:
    fields = {field.name for field in dataclasses.fields(gate.FlatG2RetainedRun)}

    assert "authority_inputs" not in fields
    assert fields == {
        "compact_summary_planes",
        "final_checkpoint_projection",
        "receipt",
        "summary_state_before",
        "summary_state_after",
        "rng_state_before",
        "rng_state_after",
        "process_identity",
    }


def test_synthetic_flat_g2_rows_close_every_exactness_check_without_promotion() -> None:
    first = _run()
    repeat = _run(planes=_clone_planes(first.compact_summary_planes))
    expected_checkpoint = _checkpoint()
    cpu_producer, cpu_calls = _cpu_producer()
    producer, calls = _producer([first, repeat])
    payload = gate.execute_live_gate(
        capability=_capability(),
        producer=producer,
        cpu_oracle_producer=cpu_producer,
        expected_final_checkpoint=expected_checkpoint,
    )

    assert payload["ok"] is True
    assert payload["decision"] == "flat_g2_compact_abi1_gate_closed_capability_false"
    assert calls == [0, 1]
    assert len(cpu_calls) == 2
    assert [call["runner_calls"] for call in cpu_calls] == [[2], [2]]
    assert payload["cpu_oracle_calls"] == 2
    assert payload["expected_authority"] == {
        "source": "independent_cpu_run_many",
        "candidate_derived": False,
        "generation_sequence": [0, 1],
    }
    assert payload["details"]["expected_authority"]["candidate_derived"] is False
    assert payload["canonical_native_capability"] is False
    assert payload["capability_promoted"] is False
    assert payload["claims"]["canonical_native_capability"] is False
    assert all(payload["checks"].values())

    receipt_checks = payload["details"]["first_run"]["receipt"]["checks"]
    required_receipt_checks = {
        "native_dispatches",
        "python_generation_iterations",
        "python_tick_iterations",
        "aten_operations_inside_tick_loop",
        "aten_dispatches_in_extension_call",
        "aten_allocations_in_extension_call",
        "outputs_preallocated_before_timing",
        "scratch_preallocated_before_timing",
        "torch_allocated_delta_bytes",
        "torch_peak_allocated_delta_bytes",
        "pointer_stability_receipt",
        "runtime_jit",
        "hidden_fallback",
        "summary_projection_mutates_simulation_state",
        "summary_projection_consumes_rng",
    }
    assert required_receipt_checks <= set(receipt_checks)
    assert all(receipt_checks[name]["exact"] for name in required_receipt_checks)

    for run_name in ("first_run", "repeat_run"):
        rows = payload["details"][run_name]["rows"]
        assert rows["checks"] == {
            "all_plane_prefix_bytes_exact": True,
            "all_semantic_rows_exact": True,
            "all_canonical_row_bytes_exact": True,
            "all_defined_masks_exact": True,
            "all_digest_fields_exact": True,
            "validity_exact": True,
            "tail_bit_zero_noop": True,
        }
        assert rows["validity"] == [1, 1, 0, 0]
        assert rows["digest_field_names"] == list(abi.DIGEST_FIELDS)
        assert all(
            all(field_checks.values())
            for field_checks in rows["digest_fields"]
        )
        assert all(check["exact"] for check in rows["defined_masks"])
        assert set(rows["planes"]) == set(gate.PLANE_NAMES)
        assert all(check["exact"] for check in rows["semantic_rows"])
        assert rows["abi_field_counts"]["digest"] == 10


@pytest.mark.parametrize("generation_count", [1, 8, 32, 64, 128])
def test_supported_generation_counts_close_exactly_with_zero_tail(
    generation_count: int,
) -> None:
    first_planes = _planes(
        generation_count,
        generation_capacity=generation_count + 1,
    )
    first = _run(generation_count=generation_count, planes=first_planes)
    repeat = _run(
        generation_count=generation_count,
        planes=_clone_planes(first_planes),
    )
    cpu_producer, cpu_calls = _cpu_producer(
        generation_count=generation_count
    )
    producer, candidate_calls = _producer([first, repeat])

    payload = gate.execute_live_gate(
        capability=_capability(),
        producer=producer,
        cpu_oracle_producer=cpu_producer,
        expected_final_checkpoint=_checkpoint(),
        generation_count=generation_count,
    )

    generation_label = f"g{generation_count}"
    expected_generations = list(range(generation_count))
    assert payload["ok"] is True
    assert payload["status"] == (
        f"exact_flat_{generation_label}_b1_compact_rows_capability_false"
    )
    assert payload["decision"] == (
        f"flat_{generation_label}_compact_abi1_gate_closed_capability_false"
    )
    assert payload["bounded_shape"]["generations"] == generation_count
    assert payload["expected_authority"]["generation_sequence"] == (
        expected_generations
    )
    assert payload["claims"][f"flat_{generation_label}_b1_rows_exact"] is True
    assert payload["claims"]["canonical_native_capability"] is False
    assert payload["canonical_native_capability"] is False
    assert payload["capability_promoted"] is False
    assert candidate_calls == [0, 1]
    assert [call["runner_calls"] for call in cpu_calls] == [
        [generation_count],
        [generation_count],
    ]
    assert all(
        call["oracle"].provenance["generation_sequence"]
        == expected_generations
        for call in cpu_calls
    )
    assert all(
        call["oracle"].provenance["execution_call"]
        == f"run_many({generation_count})"
        for call in cpu_calls
    )
    for run_name in ("first_run", "repeat_run"):
        rows = payload["details"][run_name]["rows"]
        assert rows["validity"] == [1] * generation_count + [0]
        assert rows["checks"]["validity_exact"] is True
        assert rows["checks"]["tail_bit_zero_noop"] is True
        assert len(rows["semantic_rows"]) == generation_count
        assert len(rows["defined_masks"]) == generation_count
        assert len(rows["digest_fields"]) == generation_count
        assert all(rows["canonical_bytes_exact"])


def test_g8_truncated_canonical_rows_fail_closed_without_candidate_dispatch() -> None:
    oracle, _runner, _observations = _cpu_oracle_once(generation_count=8)
    truncated = dataclasses.replace(
        oracle,
        canonical_rows=oracle.canonical_rows[:-1],
    )
    oracle_runs: Iterator[gate.FlatG2CpuOracleRun] = iter([truncated, truncated])
    candidate_producer, candidate_calls = _producer(
        [_run(generation_count=8), _run(generation_count=8)]
    )

    payload = gate.execute_live_gate(
        capability=_capability(),
        producer=candidate_producer,
        cpu_oracle_producer=lambda: next(oracle_runs),
        generation_count=8,
    )

    assert payload["ok"] is False
    assert payload["decision"] == "flat_g8_cpu_oracle_repeat_rejected"
    assert payload["first_mismatch"] == {
        "authority": "cpu_oracle_repeat",
        "path": "$.canonical_rows",
        "reason": "row_count_mismatch",
        "expected": 8,
        "actual": {"first": 7, "repeat": 7},
    }
    assert candidate_calls == []
    assert payload["canonical_native_capability"] is False


def test_unsupported_generation_count_fails_closed_without_dispatch() -> None:
    cpu_producer, cpu_calls = _cpu_producer()
    candidate_producer, candidate_calls = _producer([_run(), _run()])

    payload = gate.execute_live_gate(
        capability=_capability(),
        producer=candidate_producer,
        cpu_oracle_producer=cpu_producer,
        generation_count=3,
    )

    assert payload["ok"] is False
    assert payload["decision"] == "compact_generation_count_unsupported"
    assert payload["producer_calls"] == 0
    assert payload["cpu_oracle_calls"] == 0
    assert candidate_calls == []
    assert cpu_calls == []
    assert payload["canonical_native_capability"] is False


def test_digest_byte_mismatch_is_rejected_with_field_path() -> None:
    first = _run()
    repeat_planes = _clone_planes(first.compact_summary_planes)
    repeat_planes.digest_values[0, 0, 0, 0] ^= np.uint8(1)
    repeat = _run(planes=repeat_planes)
    cpu_producer, _cpu_calls = _cpu_producer()
    producer, _calls = _producer([first, repeat])

    payload = gate.execute_live_gate(
        capability=_capability(),
        producer=producer,
        cpu_oracle_producer=cpu_producer,
        expected_final_checkpoint=_checkpoint(),
    )

    assert payload["ok"] is False
    assert payload["decision"] == "flat_g2_compact_abi1_rejected"
    assert payload["details"]["repeat_run"]["rows"]["checks"][
        "all_digest_fields_exact"
    ] is False
    assert "digest_sha256" in payload["first_mismatch"]["path"]
    assert payload["capability_promoted"] is False


def test_invalid_tail_storage_fails_closed_before_projection() -> None:
    dirty = _planes()
    dirty.integer_values[2, 0, 0] = 1
    cpu_producer, _cpu_calls = _cpu_producer()
    producer, _calls = _producer([_run(planes=dirty), _run()])

    payload = gate.execute_live_gate(
        capability=_capability(),
        producer=producer,
        cpu_oracle_producer=cpu_producer,
        expected_final_checkpoint=_checkpoint(),
    )

    assert payload["ok"] is False
    assert payload["decision"] == "flat_g2_compact_evidence_invalid"
    assert "must remain a frozen no-op" in payload["reason"]


def test_python_loop_aten_and_summary_observation_contracts_are_required() -> None:
    receipt = _receipt()
    receipt["native_controller_receipt"]["python_generation_iterations"] = 1
    receipt["aten_dispatches_in_extension_call"] = 1
    first = _run(receipt=receipt, state_after={"authoritative_state_sha256": "3" * 64})
    repeat = _run(rng_after={"rng_authority_sha256": "4" * 64, "cursor": 10})
    cpu_producer, _cpu_calls = _cpu_producer()
    producer, _calls = _producer([first, repeat])

    payload = gate.execute_live_gate(
        capability=_capability(),
        producer=producer,
        cpu_oracle_producer=cpu_producer,
        expected_final_checkpoint=_checkpoint(),
    )

    assert payload["ok"] is False
    assert payload["details"]["first_run"]["receipt"]["failed"] == [
        "python_generation_iterations",
        "aten_dispatches_in_extension_call",
    ]
    assert payload["checks"]["first_summary_state_read_only"] is False
    assert payload["checks"]["repeat_summary_rng_non_consuming"] is False


def test_transient_allocation_and_pointer_change_fail_closed() -> None:
    receipt = _receipt()
    receipt["torch_peak_allocated_delta_bytes"] = 512
    receipt["pointer_stability_receipt"] = {
        "all_exact": False,
        "pointer_count": 12,
        "first_mismatch": {"surface": "status"},
        "mismatches": [{"surface": "status"}],
    }
    cpu_producer, _cpu_calls = _cpu_producer()
    producer, _candidate_calls = _producer([_run(receipt=receipt), _run()])

    payload = gate.execute_live_gate(
        capability=_capability(),
        producer=producer,
        cpu_oracle_producer=cpu_producer,
    )

    assert payload["ok"] is False
    assert payload["details"]["first_run"]["receipt"]["failed"] == [
        "torch_peak_allocated_delta_bytes",
        "pointer_stability_receipt",
    ]


def test_receipt_requires_explicit_non_mutating_non_consuming_flags() -> None:
    receipt = _receipt()
    receipt["summary_projection_mutates_simulation_state"] = True
    del receipt["summary_projection_consumes_rng"]
    cpu_producer, _cpu_calls = _cpu_producer()
    producer, _candidate_calls = _producer([_run(receipt=receipt), _run()])

    payload = gate.execute_live_gate(
        capability=_capability(),
        producer=producer,
        cpu_oracle_producer=cpu_producer,
    )

    assert payload["ok"] is False
    failed = payload["details"]["first_run"]["receipt"]["failed"]
    assert failed == [
        "summary_projection_mutates_simulation_state",
        "summary_projection_consumes_rng",
    ]
    assert payload["first_mismatch"] == {
        "path": "$.first_run.receipt.summary_projection_mutates_simulation_state",
        "reason": "receipt_contract_mismatch",
        "expected": False,
        "actual": True,
    }


def test_final_checkpoint_mismatch_and_repeat_are_gate_failures() -> None:
    expected = _checkpoint()
    mismatched = copy.deepcopy(expected)
    mismatched["tensors"]["alive_mask"][0] = False
    first = _run(checkpoint=mismatched)
    repeat = _run(checkpoint=copy.deepcopy(mismatched))
    cpu_producer, _cpu_calls = _cpu_producer()
    producer, _calls = _producer([first, repeat])

    payload = gate.execute_live_gate(
        capability=_capability(),
        producer=producer,
        cpu_oracle_producer=cpu_producer,
        expected_final_checkpoint=expected,
    )

    assert payload["ok"] is False
    assert payload["checks"]["first_final_checkpoint_exact"] is False
    assert payload["checks"]["repeat_final_checkpoint_exact"] is False
    assert payload["checks"]["same_process_checkpoint_repeat"] is True
    assert payload["first_mismatch"]["field"] == "alive_mask"


def test_current_live_entrypoint_fails_closed_as_precise_producer_pending(
    tmp_path: Path,
) -> None:
    payload = gate.execute_live_gate(
        capability={
            **_capability(),
            "compact_summary_native_producer_status": "producer_pending",
        }
    )

    assert payload["ok"] is False
    assert payload["status"] == "producer_pending"
    assert payload["decision"] == "producer_pending"
    assert payload["native_dispatches_attempted"] == 0
    assert payload["canonical_native_capability"] is False
    assert "retained_compact_summary_rows" in payload["missing_integration_inputs"]

    output = tmp_path / "pending.json"
    assert gate.main(["--output", str(output)]) == 1
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["decision"] == "producer_pending"
    assert written["native_dispatches_attempted"] == 0


def test_live_producer_with_missing_retention_returns_producer_pending() -> None:
    incomplete = _run()
    incomplete = gate.FlatG2RetainedRun(
        compact_summary_planes=None,
        final_checkpoint_projection=incomplete.final_checkpoint_projection,
        receipt=incomplete.receipt,
        summary_state_before=incomplete.summary_state_before,
        summary_state_after=incomplete.summary_state_after,
        rng_state_before=incomplete.rng_state_before,
        rng_state_after=incomplete.rng_state_after,
        process_identity=incomplete.process_identity,
    )
    cpu_producer, cpu_calls = _cpu_producer()
    producer, calls = _producer([incomplete])

    payload = gate.execute_live_gate(
        capability=_capability(),
        producer=producer,
        cpu_oracle_producer=cpu_producer,
        expected_final_checkpoint=_checkpoint(),
    )

    assert payload["decision"] == "producer_pending"
    assert payload["missing_integration_inputs"] == ["first.compact_summary_planes"]
    assert payload["native_dispatches_attempted"] == 1
    assert len(cpu_calls) == 2
    assert calls == [0]


def test_gate_rejects_premature_capability_promotion_without_calling_producer() -> None:
    capability = _capability()
    capability["eiel_compact_generation_summary_v1_supported"] = True
    cpu_producer, cpu_calls = _cpu_producer()
    producer, calls = _producer([_run(), _run()])

    payload = gate.execute_live_gate(
        capability=capability,
        producer=producer,
        cpu_oracle_producer=cpu_producer,
        expected_final_checkpoint=_checkpoint(),
    )

    assert payload["ok"] is False
    assert payload["decision"] == "compact_capability_boundary_violation"
    assert calls == []
    assert cpu_calls == []
    assert payload["capability_promoted"] is False
