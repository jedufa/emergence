"""Static contract for the compile-gated R11 active-preflight diagnostic."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

import engine.tensor_world.c4_resident_composed_r11 as composed_r11
from engine.tensor_world.c4_event_ring_r11 import (
    _event_ring_tensors,
    make_c4_event_ring_r11_v1,
)
from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_identity_r11 import (
    _identity_staging_tensors_r11_v1,
    bind_c4_identity_authority_r11_v1,
)
from engine.tensor_world.c4_mating import TensorWorldC4MateBirthConfigV1
from engine.tensor_world.c4_resident_birth_r2 import (
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _primary_tensors,
    _zero_tensors,
)
from engine.tensor_world.c4_resident_composed_r11 import (
    capture_c4_resident_composed_r11_window_v1,
    execute_c4_resident_composed_r11_window_cuda_v1,
    make_c4_resident_composed_r11_staging_v1,
    prepare_c4_resident_composed_r11_window_control_v1,
)
from engine.tensor_world.c4_resident_mutation_r4 import (
    _replay_tensors,
    _scalar_rng_tensors,
    _staging_tensors,
)
from engine.tensor_world.c4_resident_population_r9 import _r9_turnover_tensors
from engine.tensor_world.c4_resident_turnover_r3 import _death_authoritative
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from tests.test_eiel_tensor_world_c4_resident_sequential_continuation_r11_cuda_v1 import (
    _bind_owner as _bind_sequential_owner,
    _inject_one_death_per_world,
)


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu"
BINDING = ROOT / (
    "engine/tensor_world/native/tensor_world_c4_resident_graph_r11_binding.cpp"
)
OWNER = ROOT / "engine/tensor_world/c4_resident_composed_r11.py"
MACRO = "EIEL_C4_R11_TEST_ONLY_ACTIVE_PREFLIGHT_HOOK"


def _load_active_preflight_extension(build_dir: Path):
    build_dir.mkdir(parents=True, exist_ok=True)
    define = f"-D{MACRO}"
    return load(
        name="tensor_world_c4_resident_graph_r11_active_preflight_probe_v1",
        sources=[str(BINDING), str(KERNEL)],
        build_directory=str(build_dir),
        extra_include_paths=[
            str(ROOT / "engine/perf"),
            str(ROOT / "engine/tensor_world/native"),
        ],
        extra_cflags=["-O2", "-ffp-contract=off", define],
        extra_cuda_cflags=["-O2", "--fmad=false", define],
        with_cuda=True,
        verbose=False,
    )


def _move_dataclass_tensors(value: object, device: torch.device) -> object:
    clone = copy.deepcopy(value)

    def move(owner: object) -> None:
        if not is_dataclass(owner):
            return
        for field in fields(owner):
            item = getattr(owner, field.name)
            if isinstance(item, torch.Tensor):
                setattr(owner, field.name, item.to(device=device))
            elif is_dataclass(item):
                move(item)

    move(clone)
    return clone


def _make_active_preflight_probe_state():
    cpu_bundle, _ = make_c4_fixture_v1(
        batch=48,
        capacity=32,
        active=3,
        width=8,
        seed=157,
        world_local_identity=True,
    )
    cpu_replay = make_c5_pcg64_context_v1(
        cpu_bundle, streams=5, seed=157, typed=True
    )
    bundle = _move_dataclass_tensors(cpu_bundle, torch.device("cuda"))
    replay = _move_dataclass_tensors(cpu_replay, torch.device("cuda"))
    assert replay.raw64_transition_counts is not None
    assert replay.output32_return_counts is not None
    assert bundle.events is not None
    bundle.event_ring = make_c4_event_ring_r11_v1(
        batch=48,
        capacity=32,
        event_capacity=bundle.events.capacity,
        device="cuda",
    )
    bind_c4_identity_authority_r11_v1(bundle, semantic=False)
    staging = make_c4_resident_composed_r11_staging_v1(
        batch=48,
        capacity=32,
        grid_size=8,
        streams=int(replay.rng_state_words.shape[1]),
        device="cuda",
    )
    staging.requested_ticks.fill_(1)
    return bundle, replay, staging


def _population_args(bundle, replay, staging, config) -> tuple[object, ...]:
    assert bundle.event_ring is not None
    assert replay.raw64_transition_counts is not None
    assert replay.output32_return_counts is not None
    max_births = bundle.capacity if config.max_births is None else config.max_births
    return (
        _primary_tensors(bundle),
        _death_authoritative(bundle),
        _inherit_tensors(bundle),
        _zero_tensors(bundle),
        _minus_one_tensors(bundle),
        _half_tensors(bundle),
        _one_tensors(bundle),
        _scalar_rng_tensors(bundle),
        [
            *_replay_tensors(replay),
            replay.raw64_transition_counts,
            replay.output32_return_counts,
        ],
        _staging_tensors(staging.r9.mutation),
        _r9_turnover_tensors(staging.r9),
        _identity_staging_tensors_r11_v1(staging.identity),
        [
            staging.rng.post_raw64_transition_counts,
            staging.rng.post_output32_return_counts,
        ],
        _event_ring_tensors(bundle.event_ring),
        staging.r9.preflight_status,
        staging.global_status,
        staging.generation,
        staging.tick,
        staging.requested_ticks,
        staging.completed_ticks,
        int(max_births),
        float(config.parent_energy_share),
        float(config.minimum_energy_buffer),
        int(config.cooldown_steps),
        float(config.start_health),
        float(config.max_energy),
        int(config.life_history_mode),
        int(config.maturity_start_ticks),
        int(config.maturity_rise_ticks),
        int(config.fertility_decline_start_ticks),
        int(config.fertility_decline_ticks),
        float(config.elder_fertility_floor),
        int(config.environment_stream),
        int(config.runner_stream),
    )


def test_r11_active_preflight_hook_is_test_only_and_immediately_before_r6() -> None:
    native = KERNEL.read_text(encoding="utf-8")
    binding = BINDING.read_text(encoding="utf-8")
    owner = OWNER.read_text(encoding="utf-8")
    transaction = native.split(
        "void tensor_world_c4_resident_population_r11_graph_cuda", 1
    )[1].split(
        "std::vector<int64_t> tensor_world_c4_resident_r11_newborn_fanout_contract",
        1,
    )[0]

    hook_call = "r11::capture_active_preflight_diagnostic<<<batch, 1, 0, stream>>>"
    r6_call = "r6::all_live_semantic_preflight<<<batch, 1, 0, stream>>>"
    assert transaction.count(hook_call) == 1
    assert transaction.count(r6_call) == 1
    assert transaction.index(hook_call) < transaction.index(r6_call)
    assert f"#ifdef {MACRO}" in transaction
    enqueue = transaction.split(
        "cudaStreamCaptureStatus capture_status", 1
    )[1].split("r11::reset_local_event_cursor<<<", 1)[0]
    assert "cudaStreamIsCapturing(stream.stream(), &capture_status)" in enqueue
    for pointer in (
        "live.alive",
        "live.active_count",
        "live.active_agent_ids",
        "live.active_agent_slots",
        "live.agent_id",
    ):
        assert f"reinterpret_cast<uintptr_t>({pointer})" in enqueue
    assert "reinterpret_cast<uintptr_t>(stream.stream())" in enqueue
    dispatch = transaction.split(
        "if (test_only_active_preflight_mode == 0)", 1
    )[1].split(r6_call, 1)[0]
    assert dispatch.count("capture_active_preflight_diagnostic<<<") == 1
    assert dispatch.count("active_preflight_noop<<<") == 1
    assert dispatch.count("active_preflight_system_fence<<<") == 1
    assert dispatch.count("capture_active_preflight_checksum<<<") == 1
    final_hook = "r11::capture_active_preflight_checksum<<<batch, 1, 0, stream>>>"
    between = transaction[
        transaction.index(final_hook) + len(final_hook) : transaction.index(r6_call)
    ]
    assert "<<<" not in between
    assert f"#ifdef {MACRO}" in binding
    assert '"_test_only_configure_active_preflight"' in binding
    assert '"_test_only_active_preflight_enqueue_metadata"' in binding
    assert '"_test_only_clear_active_preflight"' in binding
    assert "_test_only_active_preflight" not in owner
    owner_loader = owner.split("def _load_r11_graph_extension_v1", 1)[1].split(
        "def _r11_population_call_v1", 1
    )[0]
    assert MACRO not in owner_loader


def test_r11_active_preflight_hook_is_read_only_and_complete() -> None:
    native = KERNEL.read_text(encoding="utf-8")
    diagnostic = native.split(
        "__global__ void capture_active_preflight_diagnostic", 1
    )[1].split("#endif", 1)[0]

    assert "live.active_count[world]" in diagnostic
    assert "live.alive[offset + slot]" in diagnostic
    assert "live.active_agent_ids[offset + row]" in diagnostic
    assert "live.active_agent_slots[offset + row]" in diagnostic
    assert "live.agent_id[offset + slot] == id" in diagnostic
    assert "diagnostic.active_count[world] = count" in diagnostic
    assert "diagnostic.alive_sum[world] = alive" in diagnostic
    assert "diagnostic.tail_valid[world] = tail_ok ? 1 : 0" in diagnostic
    assert "diagnostic.order_valid[world] = order_ok ? 1 : 0" in diagnostic
    for pointer in (
        "live.alive",
        "live.active_count",
        "live.active_agent_ids",
        "live.active_agent_slots",
        "live.agent_id",
    ):
        assert f"reinterpret_cast<uintptr_t>({pointer})" in diagnostic
    assert "status[world]" not in diagnostic
    assert "live.active_count[world] =" not in diagnostic
    assert "live.alive[offset + slot] =" not in diagnostic
    checksum = native.split(
        "__global__ void capture_active_preflight_checksum", 1
    )[1].split("#endif", 1)[0]
    assert "volatile const bool* alive = live.alive" in checksum
    assert "volatile const int64_t* active_count = live.active_count" in checksum
    assert "volatile const int64_t* active_ids = live.active_agent_ids" in checksum
    assert "volatile const int32_t* active_slots = live.active_agent_slots" in checksum
    assert "volatile const int64_t* agent_ids = live.agent_id" in checksum
    assert "diagnostic.checksum[world]" in checksum


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C4_R11_ACTIVE_PREFLIGHT_PROBE") != "1",
    reason="R11 active-preflight diagnostic is explicitly opt-in",
)
def test_r11_active_preflight_hook_observes_valid_and_invalid_partitions(
    tmp_path: Path,
) -> None:
    bundle, replay, staging = _make_active_preflight_probe_state()
    config = TensorWorldC4MateBirthConfigV1(
        max_births=0,
        allow_mutation=True,
    )
    build_dir = Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build"))
    extension = _load_active_preflight_extension(build_dir)
    observed_count = torch.full((48,), -1, dtype=torch.int64, device="cuda")
    observed_alive = torch.full_like(observed_count, -1)
    tail_valid = torch.full((48,), -1, dtype=torch.int32, device="cuda")
    order_valid = torch.full_like(tail_valid, -1)
    device_pointers = torch.full((5,), -1, dtype=torch.int64, device="cuda")
    checksum = torch.full((48,), -1, dtype=torch.int64, device="cuda")
    extension._test_only_configure_active_preflight(
        observed_count, observed_alive, tail_valid, order_valid, device_pointers,
        checksum, 0,
    )

    extension.population_transaction_device_time(
        *_population_args(bundle, replay, staging, config)
    )
    torch.cuda.synchronize()
    assert observed_count.cpu().tolist() == [3] * 48
    assert observed_alive.cpu().tolist() == [3] * 48
    assert tail_valid.cpu().tolist() == [1] * 48
    assert order_valid.cpu().tolist() == [1] * 48
    assert staging.r9.preflight_status.cpu().tolist() == [0] * 48
    primary = _primary_tensors(bundle)
    expected_pointers = [
        primary[index].data_ptr() for index in (0, 56, 54, 55, 1)
    ]
    enqueue_metadata = (
        extension._test_only_active_preflight_enqueue_metadata()
    )
    assert enqueue_metadata[:5] == expected_pointers
    assert isinstance(enqueue_metadata[5], int)
    assert enqueue_metadata[6] == 0
    assert enqueue_metadata[7] == 0
    assert device_pointers.cpu().tolist() == expected_pointers

    invalid_world = 17
    bundle.population.active_count[invalid_world] = 2
    extension.population_transaction_device_time(
        *_population_args(bundle, replay, staging, config)
    )
    torch.cuda.synchronize()
    assert int(observed_count[invalid_world].cpu()) == 2
    assert int(observed_alive[invalid_world].cpu()) == 3
    assert int(tail_valid[invalid_world].cpu()) == 0
    assert int(order_valid[invalid_world].cpu()) == 1
    expected_status = [53] * 48
    expected_status[invalid_world] = 60
    assert staging.r9.preflight_status.cpu().tolist() == expected_status
    extension._test_only_clear_active_preflight()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C4_R11_ACTIVE_PREFLIGHT_PROBE") != "1",
    reason="R11 active-preflight diagnostic is explicitly opt-in",
)
@pytest.mark.parametrize(
    ("diagnostic_mode", "mode_name"),
    (
        (0, "read_write"),
        (1, "launch_only"),
        (2, "system_fence"),
        (3, "volatile_checksum"),
    ),
    ids=("read_write", "launch_only", "system_fence", "volatile_checksum"),
)
def test_r11_active_preflight_hook_observes_replayed_window_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    diagnostic_mode: int,
    mode_name: str,
) -> None:
    build_dir = Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build"))
    extension = _load_active_preflight_extension(build_dir)
    monkeypatch.setattr(composed_r11, "_R11_NATIVE_MODULE", extension)
    config = TensorWorldC4MateBirthConfigV1(
        max_births=0,
        allow_mutation=True,
        max_energy=1_000.0,
    )
    owner, bundle, _replay, _staging = _bind_sequential_owner(
        capacity=32,
        seed=401,
        config=config,
        c2_path=os.environ["TENSOR_WORLD_C2_PREBUILT"],
        build_dir=build_dir,
    )
    observed_count = torch.full((48,), -1, dtype=torch.int64, device="cuda")
    observed_alive = torch.full_like(observed_count, -1)
    tail_valid = torch.full((48,), -1, dtype=torch.int32, device="cuda")
    order_valid = torch.full_like(tail_valid, -1)
    device_pointers = torch.full((5,), -1, dtype=torch.int64, device="cuda")
    checksum = torch.full((48,), -1, dtype=torch.int64, device="cuda")
    extension._test_only_configure_active_preflight(
        observed_count, observed_alive, tail_valid, order_valid, device_pointers,
        checksum, diagnostic_mode,
    )
    capture_c4_resident_composed_r11_window_v1(owner, config)
    shadow_primary = _primary_tensors(owner.shadow_bundle)
    expected_pointers = [
        shadow_primary[index].data_ptr() for index in (0, 56, 54, 55, 1)
    ]
    enqueue_metadata = (
        extension._test_only_active_preflight_enqueue_metadata()
    )
    assert enqueue_metadata[:5] == expected_pointers
    assert enqueue_metadata[5] != 0
    assert torch.cuda.graph.default_capture_stream is not None
    assert enqueue_metadata[5] == int(
        torch.cuda.graph.default_capture_stream.cuda_stream
    )
    assert enqueue_metadata[6] == 1
    assert enqueue_metadata[7] == diagnostic_mode
    prepare_c4_resident_composed_r11_window_control_v1(
        owner, generation=20, tick=200
    )
    torch.cuda.synchronize()
    result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
    torch.cuda.synchronize()
    assert result.global_status.cpu().tolist() == [0] * 48

    assert bundle.event_ring is not None
    bundle.event_ring.consumer_cursor.copy_(bundle.event_ring.producer_cursor)
    _inject_one_death_per_world(bundle)
    prepare_c4_resident_composed_r11_window_control_v1(
        owner, generation=25, tick=205
    )
    observed_count.fill_(-1)
    observed_alive.fill_(-1)
    tail_valid.fill_(-1)
    order_valid.fill_(-1)
    device_pointers.fill_(-1)
    checksum.fill_(-1)
    torch.cuda.synchronize()
    result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
    torch.cuda.synchronize()
    status = int(result.c4_status[0].cpu())
    print(
        {
            "mode": diagnostic_mode,
            "mode_name": mode_name,
            "fixes_second_replay": status == 0,
            "status": status,
            "completed": int(result.completed_ticks[0].cpu()),
            "count": int(observed_count[0].cpu()),
            "alive": int(observed_alive[0].cpu()),
            "tail": int(tail_valid[0].cpu()),
            "order": int(order_valid[0].cpu()),
            "binding_pointers": enqueue_metadata[:5],
            "preflight_pointers": device_pointers.cpu().tolist(),
            "capture_stream": enqueue_metadata[5],
            "capture_status": enqueue_metadata[6],
            "checksum": int(checksum[0].cpu()),
        }
    )
    assert status in (0, 53, 60)
    if diagnostic_mode == 0:
        assert status == 0
        assert device_pointers.cpu().tolist() == expected_pointers
    elif diagnostic_mode in (1, 2):
        assert device_pointers.cpu().tolist() == [-1] * 5
        assert checksum.cpu().tolist() == [-1] * 48
    else:
        assert checksum.cpu().tolist() != [-1] * 48
    extension._test_only_clear_active_preflight()
