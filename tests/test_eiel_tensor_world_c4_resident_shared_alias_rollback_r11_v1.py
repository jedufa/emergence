"""Persistent-window guard proof for the two shared R11 policy aliases.

The CUDA case is explicitly opt-in.  It compiles a test-only failure hook,
captures the same five-step production graph, injects one local failure in the
first step, and verifies that a later locally valid/birth-producing step
reaches the shared writer but is blocked by the persistent outer status.
"""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import gc
import hashlib
import os
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

import engine.tensor_world.c4_resident_composed_r11 as composed_r11
from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_mating import TensorWorldC4MateBirthConfigV1
from engine.tensor_world.c4_resident_birth_r2 import (
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _zero_tensors,
)
from engine.tensor_world.c4_resident_composed_r11 import (
    bind_c4_resident_composed_r11_v1,
    capture_c4_resident_composed_r11_window_v1,
    execute_c4_resident_composed_r11_window_cuda_v1,
    make_c4_resident_composed_r11_staging_v1,
    prepare_c4_resident_composed_r11_window_control_v1,
)
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu"
BINDING = ROOT / (
    "engine/tensor_world/native/tensor_world_c4_resident_graph_r11_binding.cpp"
)
MACRO = "EIEL_C4_R11_TEST_ONLY_ALIAS_GUARD_HOOK"
TEST_STATUS = 72
WINDOW_TICKS = 5


def _capacities(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(","))
    if not result or any(item not in (32, 512) for item in result):
        raise ValueError("alias-guard capacities must be a nonempty subset of 32,512")
    if len(set(result)) != len(result):
        raise ValueError("alias-guard capacities must not contain duplicates")
    return result


PROBE_CAPACITIES = _capacities(
    os.environ.get("EIEL_C4_R11_ALIAS_GUARD_CAPACITIES", "32")
)


def _load_alias_guard_extension(build_dir: Path):
    define = f"-D{MACRO}"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="tensor_world_c4_resident_graph_r11_alias_guard_v1",
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


def _poison_free_slot_fanout(bundle: object, *, slot: int) -> None:
    groups = (
        (_inherit_tensors(bundle), 9.0),
        (_zero_tensors(bundle), 7.0),
        (_minus_one_tensors(bundle), 7.0),
        (_half_tensors(bundle), 0.125),
        (_one_tensors(bundle), 0.25),
    )
    for tensors, value in groups:
        for tensor in tensors:
            row = tensor[:, slot]
            if tensor.dtype == torch.bool:
                row.fill_(bool(value))
            else:
                row.fill_(value)


def _make_cpu_fixture(*, capacity: int, seed: int):
    bundle, _ = make_c4_fixture_v1(
        batch=48,
        capacity=capacity,
        active=3,
        width=8,
        seed=seed,
        world_local_identity=True,
    )
    core = bundle.base.base.core
    patch = bundle.base.base.patch.agent_grid_i32
    core.occupancy.agent_id_grid.fill_(-1)
    core.occupancy.slot_grid.fill_(-1)
    patch.fill_(-1)
    # The captured C2 policy phase advances slot 0 one cell east for this
    # deterministic seed.  Starting slot 0 west of slot 2 therefore leaves
    # an adjacent parent/mate pair at the R11 transaction boundary.
    positions = ((2, 1), (5, 5), (2, 3))
    for world in range(bundle.batch):
        for slot, (y, x) in enumerate(positions):
            agent_id = int(core.agents.agent_id[world, slot])
            core.agents.pos_y[world, slot] = y
            core.agents.pos_x[world, slot] = x
            core.occupancy.agent_id_grid[world, y, x] = agent_id
            core.occupancy.slot_grid[world, y, x] = slot
            patch[world, y, x] = agent_id
        mode = world % 4
        if mode in (1, 3):
            bundle.base.lifecycle.death_candidate[world, 1] = True
            core.agents.health[world, 1] = 0.0
        if mode == 2:
            core.agents.energy[world, :3] = 0.1
            bundle.base.agent.policy_energy_observation[world, :3] = 0.1
        if mode == 3:
            bundle.population.free_queue[world].fill_(-1)
            bundle.population.free_queue_count[world] = 0
    # Keep the full C2/C3 phase birth-producing.  The native direct
    # transaction fixtures historically set these views implicitly; the
    # composed graph must receive the same explicit authority after its C3
    # phase has run.
    core.agents.health[:, :3] = 1_000.0
    core.agents.energy[:, :3] = 1_000.0
    bundle.base.agent.policy_energy_observation[:, :3] = 1_000.0
    bundle.base.agent.repro_threshold[:, :3] = 0.0
    bundle.base.genome.reproduction_threshold[:, :3] = 0.0
    for world in range(bundle.batch):
        if world % 4 in (1, 3):
            core.agents.health[world, 1] = 0.0
    _poison_free_slot_fanout(bundle, slot=3)
    bundle.base.genome.mutation_rate[:, :3] = 1.0
    assert bundle.scalar_rng is not None
    bundle.scalar_rng.python_random_gauss_primary_tape.fill_(0.01)
    bundle.scalar_rng.python_random_gauss_secondary_tape.zero_()
    bundle.base.base.core.tick = 911
    replay = make_c5_pcg64_context_v1(bundle, streams=5, seed=seed, typed=True)
    config = TensorWorldC4MateBirthConfigV1(
        max_births=1,
        allow_mutation=True,
        generation=37,
        max_energy=160.0,
    )
    return bundle, replay, config


def _byte_digest(bundle: object, replay: object) -> str:
    """Stream a complete canonical/replay byte digest without retaining a copy."""

    def leaves(value: object, path: str, output: list[tuple[str, torch.Tensor]]) -> None:
        if isinstance(value, torch.Tensor):
            output.append((path, value))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                leaves(item, f"{path}[{index}]", output)
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                leaves(getattr(value, field.name), f"{path}.{field.name}", output)

    entries: list[tuple[str, torch.Tensor]] = []
    leaves(bundle, "bundle", entries)
    leaves(replay, "replay", entries)
    digest = hashlib.sha256()
    for path, tensor in sorted(entries):
        digest.update(path.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _shared_policy_digests(bundle: object) -> tuple[str, str]:
    return (
        _tensor_digest(bundle.base.base.core.policy.packed_weights),
        _tensor_digest(bundle.base.base.policy.packed_weights),
    )


def _configure_alias_hook(extension: object, *, fault_step: int):
    control = torch.tensor([0, fault_step, -1], dtype=torch.int64, device="cuda")
    local_status = torch.full((WINDOW_TICKS,), -1, dtype=torch.int32, device="cuda")
    planned_births = torch.full((WINDOW_TICKS,), -1, dtype=torch.int64, device="cuda")
    writer_reached = torch.full((WINDOW_TICKS,), -1, dtype=torch.int32, device="cuda")
    writer_skipped = torch.full((WINDOW_TICKS,), -1, dtype=torch.int32, device="cuda")
    outer_status = torch.full((WINDOW_TICKS,), -1, dtype=torch.int32, device="cuda")
    injected = torch.full((WINDOW_TICKS,), -1, dtype=torch.int32, device="cuda")
    extension._test_only_configure_alias_guard(
        control, local_status, planned_births, writer_reached,
        writer_skipped, outer_status, injected,
    )
    return control, local_status, planned_births, writer_reached, writer_skipped, outer_status, injected


def test_r11_shared_alias_guard_is_source_bound_and_two_level() -> None:
    source = (ROOT / "engine/tensor_world/c4_resident_composed_r11.py").read_text(
        encoding="utf-8"
    )
    native = KERNEL.read_text(encoding="utf-8")
    binding = BINDING.read_text(encoding="utf-8")
    assert len(composed_r11._R11_GUARDED_SHARED_TENSOR_PATHS_V1) == 2
    assert composed_r11._R11_GUARDED_SHARED_TENSOR_PATHS_V1 == frozenset({
        "base.base.core.policy.packed_weights",
        "base.base.policy.packed_weights",
    })
    assert source.count("shadow_staging.global_status,") >= 1
    assert "persistent_outer_status" in binding
    assert "persistent_window_all_ok(persistent_outer_status, batch)" in native
    assert "transaction-local success and the" in native
    transaction = native.split(
        "void tensor_world_c4_resident_population_r11_graph_cuda", 1
    )[1].split("std::vector<int64_t> tensor_world_c4_resident_r11_newborn_fanout_contract", 1)[0]
    assert transaction.count("fused_newborn_fanout_kernel<<<") == 1
    assert native.count("packed_weights") == 0
    assert f"#ifdef {MACRO}" in native
    assert f"#ifdef {MACRO}" in binding


def test_r11_shared_alias_guard_test_hook_is_not_in_production_owner() -> None:
    owner = (ROOT / "engine/tensor_world/c4_resident_composed_r11.py").read_text(
        encoding="utf-8"
    )
    loader = owner.split("def _load_r11_graph_extension_v1", 1)[1].split(
        "def _r11_population_call_v1", 1
    )[0]
    assert MACRO not in loader
    qualified = owner.split("def execute_c4_resident_composed_r11_window_cuda_v1(", 1)[1].split(
        "def prepare_c4_resident_composed_r11_window_control_v1(", 1
    )[0]
    assert qualified.count("owner.window_graph.replay()") == 1
    assert ".fill_(" not in qualified and ".copy_(" not in qualified


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C4_R11_ALIAS_GUARD_PROBE") != "1",
    reason="R11 shared-alias guard proof is explicitly opt-in",
)
@pytest.mark.parametrize("capacity", PROBE_CAPACITIES)
def test_r11_shared_alias_guard_fault_then_later_birth_rollback(
    capacity: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c2_path = os.environ.get(
        "TENSOR_WORLD_C2_PREBUILT",
        "/tmp/emergence-test/runtime/evidence_raw/tensor_world/c2_policy_parallel_r12_4c15d001_20260815_02/"
        "tensor_world_c2_ext.so",
    )
    build_dir = Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build"))
    extension = _load_alias_guard_extension(build_dir)
    previous_module = composed_r11._R11_NATIVE_MODULE
    monkeypatch.setattr(composed_r11, "_R11_NATIVE_MODULE", extension)
    try:
        cpu_bundle, cpu_replay, config = _make_cpu_fixture(
            capacity=capacity, seed=109 + capacity
        )
        bundle = _move_dataclass_tensors(cpu_bundle, torch.device("cuda"))
        replay = _move_dataclass_tensors(cpu_replay, torch.device("cuda"))
        staging = make_c4_resident_composed_r11_staging_v1(
            batch=48,
            capacity=capacity,
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
        diagnostics = _configure_alias_hook(extension, fault_step=0)
        capture_c4_resident_composed_r11_window_v1(owner, config)
        prepare_c4_resident_composed_r11_window_control_v1(
            owner, generation=37, tick=911
        )
        torch.cuda.synchronize()
        before_digest = _byte_digest(bundle, replay)
        before_policy = _shared_policy_digests(bundle)
        result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
        torch.cuda.synchronize()
        after_digest = _byte_digest(bundle, replay)
        after_policy = _shared_policy_digests(bundle)

        assert result.global_status.cpu().tolist() == [1] * 48
        assert result.completed_ticks.cpu().tolist() == [0] * 48
        assert result.first_error_code.cpu().tolist()[0] == TEST_STATUS
        assert result.first_error_tick.cpu().tolist()[0] == 911
        assert before_digest == after_digest
        assert before_policy == after_policy

        control, local_status, planned, reached, skipped, outer, injected = diagnostics
        assert control.cpu().tolist() == [WINDOW_TICKS, 0, WINDOW_TICKS - 1]
        assert injected.cpu().tolist() == [1, 0, 0, 0, 0]
        assert int(local_status[0].cpu()) == TEST_STATUS
        assert local_status[1:].cpu().tolist() == [0] * (WINDOW_TICKS - 1)
        planned_values = planned.cpu().tolist()
        assert all(int(value) > 0 for value in planned_values[1:]), planned_values
        assert reached[1:].cpu().tolist() == [1] * (WINDOW_TICKS - 1)
        assert skipped[1:].cpu().tolist() == [1] * (WINDOW_TICKS - 1)
        assert int(outer[0].cpu()) == 0
        assert outer[1:].cpu().tolist() == [1] * (WINDOW_TICKS - 1)

        extension._test_only_clear_alias_guard()
        del result, owner, staging, replay, bundle, cpu_replay, cpu_bundle
        gc.collect()
        torch.cuda.empty_cache()

        # Positive control: the same birth-producing fixture must still write
        # both shared rows when the persistent outer status remains successful.
        cpu_bundle, cpu_replay, config = _make_cpu_fixture(
            capacity=capacity, seed=109 + capacity
        )
        bundle = _move_dataclass_tensors(cpu_bundle, torch.device("cuda"))
        replay = _move_dataclass_tensors(cpu_replay, torch.device("cuda"))
        staging = make_c4_resident_composed_r11_staging_v1(
            batch=48,
            capacity=capacity,
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
        diagnostics = _configure_alias_hook(extension, fault_step=-1)
        capture_c4_resident_composed_r11_window_v1(owner, config)
        prepare_c4_resident_composed_r11_window_control_v1(
            owner, generation=37, tick=911
        )
        torch.cuda.synchronize()
        before_policy = _shared_policy_digests(bundle)
        result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
        torch.cuda.synchronize()
        after_policy = _shared_policy_digests(bundle)
        assert result.global_status.cpu().tolist() == [0] * 48
        assert result.completed_ticks.cpu().tolist() == [WINDOW_TICKS] * 48
        assert before_policy != after_policy
        _, local_status, planned, reached, skipped, outer, injected = diagnostics
        assert injected.cpu().tolist() == [0] * WINDOW_TICKS
        assert local_status.cpu().tolist() == [0] * WINDOW_TICKS
        assert all(int(value) > 0 for value in planned.cpu().tolist())
        assert reached.cpu().tolist() == [1] * WINDOW_TICKS
        assert skipped.cpu().tolist() == [0] * WINDOW_TICKS
        assert outer.cpu().tolist() == [0] * WINDOW_TICKS
        extension._test_only_clear_alias_guard()
    finally:
        monkeypatch.setattr(composed_r11, "_R11_NATIVE_MODULE", previous_module)
