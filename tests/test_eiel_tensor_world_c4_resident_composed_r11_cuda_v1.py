"""Bounded opt-in CUDA smoke/parity seam for the repaired R11 owner."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import gc
import os
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_mating import TensorWorldC4MateBirthConfigV1
from engine.tensor_world.c4_resident_composed_r11 import (
    bind_c4_resident_composed_r11_v1,
    capture_c4_resident_composed_r11_window_v1,
    execute_c4_resident_composed_r11_cuda_v1,
    execute_c4_resident_composed_r11_window_cuda_v1,
    make_c4_resident_composed_r11_staging_v1,
    prepare_c4_resident_composed_r11_window_control_v1,
)
from engine.tensor_world.c4_resident_multitick_r10 import (
    make_c4_resident_multitick_r10_fixture_v1,
)
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1


def _move_dataclass_tensors(value: object, device: torch.device) -> object:
    clone = copy.deepcopy(value)

    def move(owner: object) -> None:
        if isinstance(owner, (tuple, list)):
            for item in owner:
                move(item)
            return
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


def _tensor_leaves(value: object, path: str = "") -> dict[str, torch.Tensor]:
    leaves: dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor):
        leaves[path] = value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            leaves.update(_tensor_leaves(item, f"{path}[{index}]"))
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            leaves.update(
                _tensor_leaves(getattr(value, field.name), f"{path}.{field.name}")
            )
    return leaves


def _make_bound_owner(
    *, seed: int, config: TensorWorldC4MateBirthConfigV1, c2_path: str,
    build_dir: Path,
) -> tuple[object, object, object]:
    fixture = make_c4_resident_multitick_r10_fixture_v1(seed=seed)
    bundle = _move_dataclass_tensors(fixture.bundle, torch.device("cuda"))
    replay = _move_dataclass_tensors(
        make_c5_pcg64_context_v1(
            fixture.bundle, streams=2, seed=seed, typed=True
        ),
        torch.device("cuda"),
    )
    staging = make_c4_resident_composed_r11_staging_v1(
        batch=48,
        capacity=32,
        grid_size=8,
        streams=int(replay.rng_state_words.shape[1]),
        device="cuda",
    )
    owner = bind_c4_resident_composed_r11_v1(
        bundle,
        replay,
        staging,
        enable_native=True,
        population_config=config,
        c2_extension_path=c2_path,
        native_build_directory=build_dir,
    )
    return owner, bundle, replay


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C4_R11_COMPOSED_PROBE") != "1",
    reason="R11 composed probe is explicitly opt-in",
)
def test_c4_r11_composed_b48_c32_device_shadow_smoke(tmp_path: Path) -> None:
    c2_path = os.environ.get("TENSOR_WORLD_C2_PREBUILT")
    if not c2_path:
        c2_path = (
            "/tmp/emergence-test/runtime/evidence_raw/tensor_world/production_c0_c7/"
            "20260815_c2_430489534_build_01/tensor_world_c2_ext.so"
        )
    fixture = make_c4_resident_multitick_r10_fixture_v1(seed=83)
    bundle = _move_dataclass_tensors(fixture.bundle, torch.device("cuda"))
    replay = _move_dataclass_tensors(
        make_c5_pcg64_context_v1(
            fixture.bundle, streams=2, seed=83, typed=True
        ),
        torch.device("cuda"),
    )
    ring_append_probe = os.environ.get("EIEL_C4_R11_RING_APPEND_PROBE") == "1"
    config = TensorWorldC4MateBirthConfigV1(
        max_births=1 if ring_append_probe else 0,
        allow_mutation=True,
        max_energy=1000.0,
    )
    staging = make_c4_resident_composed_r11_staging_v1(
        batch=48, capacity=32, grid_size=8,
        streams=int(replay.rng_state_words.shape[1]), device="cuda"
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
    if ring_append_probe:
        # Make the ring exercise deterministic even if the C3 fixture emits no
        # ordinary birth intent on this seed: the lifecycle candidate is an
        # explicit caller-owned input at the control boundary.
        bundle.base.lifecycle.death_candidate[:, 1] = True
        bundle.base.base.core.agents.health[:, 1] = 0.0
    result = execute_c4_resident_composed_r11_cuda_v1(
        owner, config, generation=11, tick=100
    )
    torch.cuda.synchronize()
    assert result.r9.preflight_status.cpu().tolist() == [0] * 48
    assert owner.native_bound is True

    if ring_append_probe:
        ring = bundle.event_ring
        assert ring is not None
        producer = ring.producer_cursor.cpu().tolist()
        counts = ring.bank_counts.cpu().tolist()
        assert all(value >= 0 for value in producer)
        assert any(value > 0 for value in producer)
        for world, value in enumerate(producer):
            assert sum(counts[world]) == value
            for bank_index, count in enumerate(counts[world]):
                if count == 0:
                    continue
                bank = ring.banks[bank_index]
                assert bank.typed_generation[world, :count].cpu().tolist() == [11] * count
                assert bank.typed_tick[world, :count].cpu().tolist() == [100] * count
        return

    window_probe = os.environ.get("EIEL_C4_R11_WINDOW_PROBE") == "1"
    window_failure_probe = (
        os.environ.get("EIEL_C4_R11_WINDOW_FAILURE_PROBE") == "1"
    )
    if window_probe or window_failure_probe:
        capture_c4_resident_composed_r11_window_v1(owner, config)
        if window_failure_probe:
            assert bundle.event_ring is not None
            replay.event_sequence[47] = bundle.event_ring.producer_cursor[47] + 1
        prepare_c4_resident_composed_r11_window_control_v1(
            owner, generation=20, tick=200
        )
        window_result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
        torch.cuda.synchronize()
        if window_failure_probe:
            assert window_result.completed_ticks.cpu().tolist() == [0] * 48
            assert window_result.first_error_world.cpu().tolist() == [47] * 48
            assert window_result.first_error_code.cpu().tolist() == [45] * 48
            return
        assert window_result.completed_ticks.cpu().tolist() == [5] * 48
        assert window_result.global_status.cpu().tolist() == [0] * 48
        return

    before_failure = {
        path: tensor.detach().cpu().clone()
        for path, tensor in (
            _tensor_leaves(bundle, "bundle") | _tensor_leaves(replay, "replay")
        ).items()
    }
    bundle.events.active_count[47] = bundle.events.capacity
    replay.event_sequence[47] = bundle.events.capacity
    bundle.base.base.core.agents.health[47, 1] = 0.0
    result = execute_c4_resident_composed_r11_cuda_v1(
        owner, config, generation=12, tick=101
    )
    torch.cuda.synchronize()
    assert int(result.r9.preflight_status[47].item()) != 0
    after_failure = {
        path: tensor.detach().cpu()
        for path, tensor in (
            _tensor_leaves(bundle, "bundle") | _tensor_leaves(replay, "replay")
        ).items()
    }
    assert after_failure.keys() == before_failure.keys()
    for path, saved in before_failure.items():
        # The injected prestate itself is intentionally after the first
        # successful tick; rollback must preserve it byte-for-byte.
        if path.endswith("core.agents.health"):
            # health is the only deliberate pre-failure mutation.
            expected = saved.clone()
            expected[47, 1] = 0.0
        elif path.endswith("active_count") and "events" in path:
            expected = saved.clone()
            expected[47] = bundle.events.capacity
        elif path.endswith("event_sequence"):
            expected = saved.clone()
            expected[47] = bundle.events.capacity
        else:
            expected = saved
        torch.testing.assert_close(after_failure[path], expected, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C4_R11_COMPOSED_PROBE") != "1"
    or os.environ.get("EIEL_C4_R11_EXACT_WINDOW_PARITY_PROBE") != "1",
    reason="R11 exact graph-window parity probe is explicitly opt-in",
)
def test_c4_r11_five_tick_graph_window_matches_composed_eager_boundary(
    tmp_path: Path,
) -> None:
    """Compare every resident owner leaf after the same five composed ticks.

    The five-call route is a qualification oracle executed only at an explicit
    test boundary.  Production continues to use the single graph-window call;
    no Python semantic work is introduced inside or between its normal ticks.
    """

    c2_path = os.environ.get(
        "TENSOR_WORLD_C2_PREBUILT",
        "/tmp/emergence-test/runtime/evidence_raw/tensor_world/production_c0_c7/"
        "20260815_c2_430489534_build_01/tensor_world_c2_ext.so",
    )
    config = TensorWorldC4MateBirthConfigV1(
        max_births=0,
        allow_mutation=True,
        max_energy=1000.0,
    )
    build_dir = Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build"))
    eager_owner, eager_bundle, eager_replay = _make_bound_owner(
        seed=89,
        config=config,
        c2_path=c2_path,
        build_dir=build_dir,
    )
    for offset in range(5):
        execute_c4_resident_composed_r11_cuda_v1(
            eager_owner,
            config,
            generation=20 + offset,
            tick=200 + offset,
        )
    # This synchronization is a qualification boundary, not part of the
    # production route.  It localizes any asynchronous fault to the five-call
    # reference before the independent graph-window owner is launched.
    torch.cuda.synchronize()
    assert eager_owner.shadow_staging is not None
    assert eager_owner.shadow_staging.global_status.cpu().tolist() == [0] * 48
    eager_leaves = {
        path: tensor.detach().cpu().clone()
        for path, tensor in (
            _tensor_leaves(eager_bundle, "bundle")
            | _tensor_leaves(eager_replay, "replay")
        ).items()
    }

    # The admitted runtime owns exactly one canonical graph owner.  Harvest
    # the qualification reference and retire it before constructing the
    # independent window owner so the test does not introduce an unsupported
    # multi-owner CUDA-graph lifetime.
    del eager_owner, eager_bundle, eager_replay
    gc.collect()
    torch.cuda.empty_cache()

    window_owner, window_bundle, window_replay = _make_bound_owner(
        seed=89,
        config=config,
        c2_path=c2_path,
        build_dir=build_dir,
    )
    capture_c4_resident_composed_r11_window_v1(window_owner, config)
    prepare_c4_resident_composed_r11_window_control_v1(
        window_owner,
        generation=20,
        tick=200,
    )
    window_result = execute_c4_resident_composed_r11_window_cuda_v1(window_owner)
    torch.cuda.synchronize()

    window_leaves = _tensor_leaves(window_bundle, "bundle") | _tensor_leaves(
        window_replay, "replay"
    )
    assert eager_leaves.keys() == window_leaves.keys()
    for path, expected in eager_leaves.items():
        torch.testing.assert_close(
            window_leaves[path].cpu(), expected, equal_nan=True,
            msg=lambda message, path=path: f"{path}: {message}",
        )
    assert window_result.completed_ticks.cpu().tolist() == [5] * 48
    assert window_result.current_tick.cpu().tolist() == [204] * 48
    assert window_result.tick.cpu().tolist() == [205] * 48
    assert window_result.generation.cpu().tolist() == [25] * 48
    assert window_result.global_status.cpu().tolist() == [0] * 48
