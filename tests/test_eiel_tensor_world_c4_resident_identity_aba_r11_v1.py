"""Focused proof for the R11 split identity-token ABA guard."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

from engine.tensor_world.c4_event_ring_r11 import (
    _event_ring_tensors,
    make_c4_event_ring_r11_v1,
)
from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_identity_r11 import (
    _identity_staging_tensors_r11_v1,
    bind_c4_identity_authority_r11_v1,
)
from engine.tensor_world.c4_resident_birth_r2 import (
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _primary_tensors,
    _zero_tensors,
)
from engine.tensor_world.c4_resident_composed_r11 import (
    make_c4_resident_composed_r11_staging_v1,
)
from engine.tensor_world.c4_resident_mutation_r4 import (
    _replay_tensors,
    _scalar_rng_tensors,
    _staging_tensors,
)
from engine.tensor_world.c4_resident_population_r9 import _r9_turnover_tensors
from engine.tensor_world.c4_resident_turnover_r3 import _death_authoritative
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu"
BINDING = ROOT / (
    "engine/tensor_world/native/tensor_world_c4_resident_graph_r11_binding.cpp"
)
OWNER = ROOT / "engine/tensor_world/c4_resident_composed_r11.py"


def _load_identity_probe_extension(build_dir: Path):
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="tensor_world_c4_resident_graph_r11_identity_aba_probe_v1",
        sources=[str(BINDING), str(KERNEL)],
        build_directory=str(build_dir),
        extra_include_paths=[
            str(ROOT / "engine/perf"),
            str(ROOT / "engine/tensor_world/native"),
        ],
        extra_cflags=[
            "-O2", "-ffp-contract=off", "-DEIEL_C4_R11_TEST_ONLY_IDENTITY_HOOK"
        ],
        extra_cuda_cflags=[
            "-O2", "--fmad=false", "-DEIEL_C4_R11_TEST_ONLY_IDENTITY_HOOK"
        ],
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


def _make_probe_state():
    cpu_bundle, _ = make_c4_fixture_v1(
        batch=48,
        capacity=32,
        active=3,
        width=8,
        seed=151,
        world_local_identity=True,
    )
    bundle = _move_dataclass_tensors(cpu_bundle, torch.device("cuda"))
    replay = _move_dataclass_tensors(
        make_c5_pcg64_context_v1(
            cpu_bundle, streams=5, seed=151, typed=True
        ),
        torch.device("cuda"),
    )
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
    world = 17
    turn = staging.r9.turnover
    birth = staging.r9.mutation.birth
    live_ids = bundle.base.base.core.agents.agent_id
    turn.death_count[world] = 1
    turn.death_slots[world, 0] = 1
    turn.death_agent_ids[world, 0] = live_ids[world, 1]
    birth.planned_count[world] = 1
    birth.parent_slots[world, 0] = 0
    staging.r9.mutation.mate_slots[world, 0] = 2
    birth.child_slots[world, 0] = 3
    birth.child_agent_ids[world, 0] = bundle.population.next_agent_id[world]
    assert bool(bundle.base.base.core.agents.alive[world, 0])
    assert bool(bundle.base.base.core.agents.alive[world, 2])
    assert not bool(turn.shadow_alive[world, 3])
    # The production C3 single-shadow route validates identity after its
    # disposable post-death projection has cleared the released row.  Mirror
    # that boundary here so this test-only hook exercises the same authority
    # rather than the pre-death canonical fixture.
    dead_y = int(bundle.base.base.core.agents.pos_y[world, 1].item())
    dead_x = int(bundle.base.base.core.agents.pos_x[world, 1].item())
    bundle.base.base.core.agents.alive[world, 1] = False
    bundle.base.base.core.agents.agent_id[world, 1] = -1
    bundle.base.base.core.agents.pos_y[world, 1] = -1
    bundle.base.base.core.agents.pos_x[world, 1] = -1
    bundle.base.base.core.occupancy.agent_id_grid[world, dead_y, dead_x] = -1
    bundle.base.base.core.occupancy.slot_grid[world, dead_y, dead_x] = -1
    bundle.base.base.patch.agent_grid_i32[world, dead_y, dead_x] = -1
    return bundle, replay, staging, world


def _probe_args(bundle, replay, staging) -> tuple[object, ...]:
    assert replay.raw64_transition_counts is not None
    assert replay.output32_return_counts is not None
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
        staging.r9.preflight_status,
    )


def _rollback_tensors(bundle, replay) -> list[torch.Tensor]:
    assert bundle.event_ring is not None
    assert replay.raw64_transition_counts is not None
    assert replay.output32_return_counts is not None
    return [
        *_primary_tensors(bundle),
        *_death_authoritative(bundle),
        *_inherit_tensors(bundle),
        *_zero_tensors(bundle),
        *_minus_one_tensors(bundle),
        *_half_tensors(bundle),
        *_one_tensors(bundle),
        *_scalar_rng_tensors(bundle),
        *_replay_tensors(replay),
        replay.raw64_transition_counts,
        replay.output32_return_counts,
        *_event_ring_tensors(bundle.event_ring),
    ]


def _assert_byte_identical(
    actual: list[torch.Tensor], expected: list[torch.Tensor]
) -> None:
    assert len(actual) == len(expected)
    for value, frozen in zip(actual, expected, strict=True):
        assert value.dtype == frozen.dtype and value.shape == frozen.shape
        assert torch.equal(value.contiguous().view(torch.uint8), frozen.view(torch.uint8))


def _reset_probe_status(staging) -> None:
    staging.r9.preflight_status.zero_()
    staging.r9.turnover.death_status.zero_()
    staging.r9.mutation.birth.status.zero_()
    staging.identity.status.zero_()


def test_r11_aba_probe_is_split_test_only_and_uses_production_kernels() -> None:
    native = KERNEL.read_text(encoding="utf-8")
    binding = BINDING.read_text(encoding="utf-8")
    owner = OWNER.read_text(encoding="utf-8")
    capture = native.split(
        "void tensor_world_c4_resident_r11_test_only_identity_capture_cuda", 1
    )[1].split(
        "void tensor_world_c4_resident_r11_test_only_identity_validate_cuda", 1
    )[0]
    validate = native.split(
        "void tensor_world_c4_resident_r11_test_only_identity_validate_cuda", 1
    )[1]

    assert "validate_r11_identity_probe_abi" in capture
    assert capture.count("r11::capture_identity_tokens<<<") == 1
    assert "r11::validate_identity_tokens<<<" not in capture
    assert "validate_r11_identity_probe_abi" in validate
    assert validate.count("r11::validate_identity_tokens<<<") == 1
    assert validate.count("r11::merge_identity_status<<<") == 1
    assert "commit_deaths<<<" not in capture + validate
    assert "commit_kernel<<<" not in capture + validate
    assert '"_test_only_identity_capture"' in binding
    assert '"_test_only_identity_validate"' in binding
    assert "#ifdef EIEL_C4_R11_TEST_ONLY_IDENTITY_HOOK" in native
    assert "#ifdef EIEL_C4_R11_TEST_ONLY_IDENTITY_HOOK" in binding
    assert "_test_only_identity_" not in owner
    owner_loader = owner.split("def _load_r11_graph_extension_v1", 1)[1].split(
        "def _r11_population_call_v1", 1
    )[0]
    assert "EIEL_C4_R11_TEST_ONLY_IDENTITY_HOOK" not in owner_loader


def test_r11_aba_status_and_peer_contract_is_source_bound() -> None:
    native = KERNEL.read_text(encoding="utf-8")
    identity = native.split("__global__ void validate_identity_tokens", 1)[1].split(
        "__global__ void merge_identity_status", 1
    )[0]
    merge = native.split("__global__ void merge_identity_status", 1)[1].split(
        "__global__ void publish_first_status", 1
    )[0]

    assert "kSlotTokenMismatch = 65" in native
    assert "kSlotTokenOverflow = 66" in native
    assert "identity.child_tokens[offset + row] == INT64_MAX" in identity
    assert "code == 0 ? r6::kPeerFailed : code" in merge
    assert "turn.status[world] = r5::kPeerFailed" in merge
    assert "stage.base.status[world] = r4::kInvalidResident" in merge


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C4_R11_IDENTITY_ABA_PROBE") != "1",
    reason="R11 identity ABA split probe is explicitly opt-in",
)
def test_r11_aba_status65_status66_and_whole_batch_rollback(tmp_path: Path) -> None:
    bundle, replay, staging, world = _make_probe_state()
    build_dir = Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build"))
    extension = _load_identity_probe_extension(build_dir)
    rollback = _rollback_tensors(bundle, replay)
    frozen = [tensor.clone() for tensor in rollback]
    token_fields = (
        staging.identity.death_slot_tokens,
        staging.identity.parent_slot_tokens,
        staging.identity.mate_slot_tokens,
        staging.identity.child_slot_tokens,
    )

    extension._test_only_identity_capture(*_probe_args(bundle, replay, staging))
    extension._test_only_identity_validate(*_probe_args(bundle, replay, staging))
    torch.cuda.synchronize()
    assert staging.r9.preflight_status.cpu().tolist() == [0] * 48
    assert staging.identity.status.cpu().tolist() == [0] * 48
    assert staging.r9.turnover.death_status.cpu().tolist() == [0] * 48
    assert staging.r9.mutation.birth.status.cpu().tolist() == [0] * 48
    _assert_byte_identical(_rollback_tensors(bundle, replay), frozen)

    for expected_tokens in token_fields:
        _reset_probe_status(staging)
        extension._test_only_identity_capture(*_probe_args(bundle, replay, staging))
        torch.cuda.synchronize()
        live_tokens = bundle.population.slot_tokens[world]
        assert int(staging.identity.death_slot_tokens[world, 0]) == int(live_tokens[1])
        assert int(staging.identity.parent_slot_tokens[world, 0]) == int(live_tokens[0])
        assert int(staging.identity.mate_slot_tokens[world, 0]) == int(live_tokens[2])
        assert int(staging.identity.child_slot_tokens[world, 0]) == int(live_tokens[3])
        expected_tokens[world, 0].add_(1)
        extension._test_only_identity_validate(*_probe_args(bundle, replay, staging))
        torch.cuda.synchronize()
        expected_status = [53] * 48
        expected_status[world] = 65
        assert staging.r9.preflight_status.cpu().tolist() == expected_status
        assert staging.identity.status.cpu().tolist() == [
            65 if index == world else 0 for index in range(48)
        ]
        assert staging.r9.turnover.death_status.cpu().tolist() == [40] * 48
        assert staging.r9.mutation.birth.status.cpu().tolist() == [32] * 48
        _assert_byte_identical(_rollback_tensors(bundle, replay), frozen)

    _reset_probe_status(staging)
    bundle.population.slot_tokens[world, 3] = torch.iinfo(torch.int64).max
    overflow_rollback = _rollback_tensors(bundle, replay)
    overflow_frozen = [tensor.clone() for tensor in overflow_rollback]
    extension._test_only_identity_capture(*_probe_args(bundle, replay, staging))
    extension._test_only_identity_validate(*_probe_args(bundle, replay, staging))
    torch.cuda.synchronize()
    expected_status = [53] * 48
    expected_status[world] = 66
    assert staging.r9.preflight_status.cpu().tolist() == expected_status
    assert staging.identity.status.cpu().tolist() == [
        66 if index == world else 0 for index in range(48)
    ]
    assert staging.r9.turnover.death_status.cpu().tolist() == [40] * 48
    assert staging.r9.mutation.birth.status.cpu().tolist() == [32] * 48
    _assert_byte_identical(_rollback_tensors(bundle, replay), overflow_frozen)
