"""Read-only current-source C512 component attribution probe.

This is intentionally a measurement harness, not an optimization or gate
implementation.  It profiles one qualified five-tick graph replay and emits
an external receipt with event-name/category totals.  The normal replay path
remains unchanged.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import runpy

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from engine.tensor_world.c4_mating import TensorWorldC4MateBirthConfigV1
from engine.tensor_world.c4_resident_composed_r11 import (
    bind_c4_resident_composed_r11_v1,
    capture_c4_resident_composed_r11_window_v1,
    execute_c4_resident_composed_r11_window_cuda_v1,
    make_c4_resident_composed_r11_staging_v1,
    prepare_c4_resident_composed_r11_window_control_v1,
)


_RESOURCE_TEST = Path(__file__).with_name(
    "test_eiel_tensor_world_c4_resident_composed_r11_production_cuda_v1.py"
)
_HELPERS = runpy.run_path(str(_RESOURCE_TEST), run_name="_c4_c512_attribution_helpers")


def _category(name: str) -> str:
    lowered = name.lower()
    if "policy_parallel_kernel" in lowered:
        return "c2_policy"
    if "observe_kernel" in lowered or "select_kernel" in lowered:
        return "c3_observe_select"
    if "settle_kernel" in lowered or "pre_tick_kernel" in lowered:
        return "c3_settle_mechanics"
    if "raw_transport_kernel" in lowered or "memcpy32_post" in lowered:
        return "r13_transport"
    if (
        "fused_newborn" in lowered
        or "r11::" in lowered
        or "r4::" in lowered
        or "r5::" in lowered
        or "r6::" in lowered
    ):
        return "r11_transaction"
    if name.startswith("void at::native::") or name.startswith("at::native::"):
        return "torch_framework"
    if "cudaGraphLaunch" in name:
        return "graph_launch"
    if name in {
        "ac2g",
        "process_sort_index",
        "process_name",
        "process_labels",
        "thread_name",
        "thread_sort_index",
    }:
        return "profiler_metadata"
    return "other"


def _event_row(event: object) -> tuple[str, float, int]:
    name = str(getattr(event, "name", "<unnamed>"))
    count = int(getattr(event, "count", 1) or 1)
    duration = 0.0
    for attribute in (
        "self_device_time_total",
        "self_cuda_time_total",
        "device_time_total",
        "cuda_time_total",
    ):
        value = getattr(event, attribute, None)
        if value is not None:
            duration = float(value)
            break
    return name, duration, count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C4_R11_ATTRIBUTION_PROBE") != "1",
    reason="C512 attribution is explicitly opt-in",
)
def test_c4_r11_c512_current_source_component_attribution(tmp_path: Path) -> None:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    cpu_bundle, cpu_replay = _HELPERS["_prepare_m64_fixture"]()
    bundle = _HELPERS["_move_in_place"](__import__("copy").deepcopy(cpu_bundle), device)
    replay = _HELPERS["_move_in_place"](__import__("copy").deepcopy(cpu_replay), device)
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
        "/tmp/emergence-test/runtime/evidence_raw/tensor_world/"
        "c2_policy_parallel_r12_4c15d001_20260815_02/tensor_world_c2_ext.so",
    )
    build_dir = Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build"))
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
    prepare_c4_resident_composed_r11_window_control_v1(owner, generation=100, tick=1000)
    torch.cuda.synchronize(device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
    torch.cuda.synchronize(device)
    assert result.global_status.cpu().tolist() == [0] * 48
    assert result.completed_ticks.cpu().tolist() == [5] * 48

    rows = [_event_row(event) for event in profiler.events()]
    durations: defaultdict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    names: Counter[str] = Counter()
    for name, duration, count in rows:
        category = _category(name)
        durations[category] += duration
        counts[category] += count
        names[name] += count
    graph_nodes = int(
        owner.r11_extension.captured_graph_node_count(
            owner.window_graph.raw_cuda_graph()
        )
    )
    assert graph_nodes > 0
    receipt = {
        "schema_version": "tensor_world_c4_r11_c512_component_attribution_v1",
        "source_commit": os.environ.get("EIEL_C4_R11_SOURCE_COMMIT", "UNBOUND"),
        "route": "R11_B48_W32_P256_C512_FIVE_TICK_GRAPH_WINDOW",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch": 48,
        "width": 32,
        "active_population": 256,
        "capacity": 512,
        "ticks_per_window": 5,
        "graph_node_count": graph_nodes,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "category_event_counts": dict(sorted(counts.items())),
        "category_device_time_total_us": dict(sorted(durations.items())),
        "top_event_names_by_count": names.most_common(80),
        "timing_scope": "one replay-only graph window; profiler attribution, not a performance gate",
        "optimization_run": False,
        "gate_claim": False,
    }
    receipt_path = Path(
        os.environ.get(
            "EIEL_C4_R11_ATTRIBUTION_RECEIPT",
            tmp_path / "c4_r11_c512_component_attribution.json",
        )
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trace_path = os.environ.get("EIEL_C4_R11_ATTRIBUTION_TRACE")
    if trace_path:
        profiler.export_chrome_trace(trace_path)


def test_c4_r11_c512_attribution_is_read_only_contract() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert "optimization_run" in source
    assert "gate_claim" in source
    assert "execute_c4_resident_composed_r11_window_cuda_v1" in source
