"""Opt-in current-source C512 node attribution for C4 Performance Phase B.

The test prepares the admitted owner at a control boundary and marks exactly
one graph replay with an NVTX range.  Nsight's ``--cuda-graph-trace=node``
mode can then attribute the captured graph nodes without adding host work to
the measured replay window.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

import engine.tensor_world.c4_resident_composed_r11 as composed_r11
from engine.tensor_world.c4_mating import TensorWorldC4MateBirthConfigV1
from engine.tensor_world.c4_resident_composed_r11 import (
    bind_c4_resident_composed_r11_v1,
    capture_c4_resident_composed_r11_window_v1,
    execute_c4_resident_composed_r11_window_cuda_v1,
    make_c4_resident_composed_r11_staging_v1,
    prepare_c4_resident_composed_r11_window_control_v1,
)
from tests.test_eiel_tensor_world_c4_resident_composed_r11_production_cuda_v1 import (
    _cuda_graph_node_count,
    _move_in_place,
    _prepare_m64_fixture,
    _restore_qualification_sample,
)


NVTX_RANGE = "eiel_c4_phase_b_c512_graph_replay_v1"
SURFACE_TIMING_MACRO = "EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK"
R11_KERNEL = Path(__file__).resolve().parents[1] / (
    "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu"
)
R11_BINDING = Path(__file__).resolve().parents[1] / (
    "engine/tensor_world/native/tensor_world_c4_resident_graph_r11_binding.cpp"
)


def _load_surface_timing_extension(build_dir: Path) -> object:
    """Build the R11 source with the attribution-only marker ABI."""

    define = f"-D{SURFACE_TIMING_MACRO}"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="tensor_world_c4_resident_graph_r11_phase_b_surface_timing_v1",
        sources=[str(R11_BINDING), str(R11_KERNEL)],
        build_directory=str(build_dir),
        extra_include_paths=[
            str(Path(__file__).resolve().parents[1] / "engine/perf"),
            str(Path(__file__).resolve().parents[1] / "engine/tensor_world/native"),
        ],
        extra_cflags=["-O2", "-ffp-contract=off", define],
        extra_cuda_cflags=["-O2", "--fmad=false", define],
        with_cuda=True,
        verbose=False,
    )


def _surface_stats(markers: torch.Tensor, clock_rate_khz: int) -> dict[str, object]:
    """Convert per-world device clock pairs into auditable summary values."""

    names = {
        (0, 1): "full_gaussian_tape_validation",
        (2, 3): "gaussian_window_capacity_preflight",
        (4, 5): "r4_capacity_wide_staging_reset",
        (6, 7): "r4_ordered_reproduction_planner",
        (8, 9): "r5_post_plan_queue_death_preparation",
        (10, 11): "r4_r5_post_plan_materialization_commit",
        (12, 13): "r4_mt_gaussian_mutation_interpreter",
        (14, 15): "active_projection_rebuild",
    }
    values = markers.detach().cpu().to(torch.int64)
    result: dict[str, object] = {}
    for (start, end), name in names.items():
        cycles = (values[:, end] - values[:, start]).tolist()
        valid = [int(value) for value in cycles if value > 0]
        if not valid:
            raise AssertionError(f"no valid device timing markers for {name}")
        milliseconds = [value / float(clock_rate_khz) for value in valid]
        ordered = sorted(milliseconds)
        result[name] = {
            "worlds": len(valid),
            "min_ms": min(milliseconds),
            "median_ms": ordered[len(ordered) // 2],
            "max_ms": max(milliseconds),
            "mean_ms": sum(milliseconds) / len(milliseconds),
            "min_cycles": min(valid),
            "max_cycles": max(valid),
        }
    return result


def _time_graph_replay(graph: object, device: torch.device) -> float:
    """Time one already-captured graph at an explicit control boundary."""

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C4_PHASE_B_ATTRIBUTION") != "1",
    reason="Phase-B attribution is explicitly opt-in",
)
def test_c4_phase_b_c512_single_graph_node_attribution(tmp_path: Path) -> None:
    """Capture one current-source B48/W32/P256/C512 graph replay only."""

    device = torch.device("cuda")
    cpu_bundle, cpu_replay = _prepare_m64_fixture()
    bundle = _move_in_place(cpu_bundle, device)
    replay = _move_in_place(cpu_replay, device)
    staging = make_c4_resident_composed_r11_staging_v1(
        batch=48,
        capacity=512,
        grid_size=32,
        streams=int(replay.rng_state_words.shape[1]),
        device=device,
    )
    config = TensorWorldC4MateBirthConfigV1(
        max_births=64,
        allow_mutation=True,
        max_energy=160.0,
    )
    c2_path = os.environ.get(
        "TENSOR_WORLD_C2_PREBUILT",
        "/tmp/emergence-test/runtime/evidence_raw/tensor_world/production_c0_c7/"
        "20260815_c2_430489534_build_01/tensor_world_c2_ext.so",
    )
    build_dir = Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build"))
    extension = _load_surface_timing_extension(build_dir)
    markers = torch.full((48, 16), -1, dtype=torch.int64, device=device)
    extension._test_only_configure_phase_timing(markers)
    previous_module = composed_r11._R11_NATIVE_MODULE
    composed_r11._R11_NATIVE_MODULE = extension
    try:
        owner = bind_c4_resident_composed_r11_v1(
            bundle,
            replay,
            staging,
            enable_native=True,
            population_config=config,
            c2_extension_path=c2_path,
            native_build_directory=build_dir,
        )
        capture_c4_resident_composed_r11_window_v1(owner, config)

        # The production transaction graph is intentionally not retained with
        # raw graph nodes, but it is replayable and gives a direct one-tick
        # attribution boundary without mixing C2/C3 or transport work.
        _restore_qualification_sample(
            cpu_bundle=cpu_bundle,
            cpu_replay=cpu_replay,
            bundle=bundle,
            replay=replay,
        )
        owner.copy_graph.replay()
        owner.c3_graph.replay()
        prepare_c4_resident_composed_r11_window_control_v1(
            owner, generation=100, tick=1_000
        )
        torch.cuda.synchronize(device)
        owner.transaction_graph.replay()
        torch.cuda.synchronize(device)
        clock_rate_khz = int(torch.cuda.get_device_properties(device).clock_rate)
        marker_stats = _surface_stats(markers, clock_rate_khz)

        _restore_qualification_sample(
            cpu_bundle=cpu_bundle,
            cpu_replay=cpu_replay,
            bundle=bundle,
            replay=replay,
        )
        prepare_c4_resident_composed_r11_window_control_v1(
            owner, generation=100, tick=1_000
        )
        torch.cuda.synchronize(device)

        component_graphs = {
            "copy_graph": owner.copy_graph,
            "c3_graph": owner.c3_graph,
            "transaction_graph": owner.transaction_graph,
            "commit_graph": owner.commit_graph,
        }
        if any(graph is None for graph in component_graphs.values()):
            raise AssertionError("R11 component graph capture is incomplete")
        component_ms: dict[str, float] = {}
        # These component replays are attribution-only control-boundary probes.
        # The transaction follows copy+C3, and commit follows the transaction;
        # no component value is treated as an additive wall-clock decomposition.
        for name, graph in component_graphs.items():
            component_ms[name] = _time_graph_replay(graph, device)

        # Restore the canonical boundary before the qualified full-window replay.
        _restore_qualification_sample(
            cpu_bundle=cpu_bundle,
            cpu_replay=cpu_replay,
            bundle=bundle,
            replay=replay,
        )
        prepare_c4_resident_composed_r11_window_control_v1(
            owner, generation=100, tick=1_000
        )
        torch.cuda.synchronize(device)

        torch.cuda.nvtx.range_push(NVTX_RANGE)
        try:
            result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(device)

        assert result.global_status.cpu().tolist() == [0] * 48
        assert result.completed_ticks.cpu().tolist() == [5] * 48
        receipt_path = Path(
            os.environ.get(
                "EIEL_C4_PHASE_B_ATTRIBUTION_RECEIPT",
                tmp_path / "c4_phase_b_c512_attribution_receipt.json",
            )
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(
                {
                    "schema_version": "tensor_world_c4_phase_b_c512_attribution_v1",
                    "source_commit": os.environ.get(
                        "EIEL_C4_PHASE_B_SOURCE_COMMIT", "UNBOUND"
                    ),
                    "route": "R11_B48_W32_P256_C512_FIVE_TICK_GRAPH_WINDOW",
                    "device": torch.cuda.get_device_name(device),
                    "nvtx_range": NVTX_RANGE,
                    "graph_node_count": _cuda_graph_node_count(owner),
                    "transaction_graph_node_count": None,
                    "transaction_graph_node_count_note": (
                        "transaction_graph is intentionally not retained with "
                        "keep_graph=True by the production owner"
                    ),
                    "component_replay_ms": component_ms,
                    "surface_device_clock_rate_khz": clock_rate_khz,
                    "surface_attribution_ms": marker_stats,
                    "completed_ticks": int(result.completed_ticks[0].cpu()),
                    "global_status": int(result.global_status[0].cpu()),
                    "event_producer_cursor": int(
                        bundle.event_ring.producer_cursor[0].cpu()
                    )
                    if bundle.event_ring is not None
                    else None,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        extension._test_only_clear_phase_timing()
        composed_r11._R11_NATIVE_MODULE = previous_module
