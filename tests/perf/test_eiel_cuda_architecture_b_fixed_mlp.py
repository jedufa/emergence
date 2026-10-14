from __future__ import annotations

from pathlib import Path
import copy

import numpy as np
import pytest
import torch

from perf.cuda_architecture_b_fixed_mlp import (
    CUDA_FIXED_MLP_CAPABILITIES,
    fixed_mlp_refill_batched_counted_cuda,
    fixed_mlp_refill_counted_cuda,
    fixed_mlp_refill_cuda,
)


OBSERVATION_SIZE = 6
HIDDEN_SIZE = 8
ACTION_COUNT = 7
CAPACITY = 11
ROW_COUNT = 7
CACHE_CAPACITY = 5
CACHE_WIDTH = 4
SLOTS = (8, 1, 10, 3, 0, 6, 4)


def _cuda_fixture() -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(910_202_607)
    packed_width = (
        HIDDEN_SIZE * OBSERVATION_SIZE
        + HIDDEN_SIZE
        + HIDDEN_SIZE * HIDDEN_SIZE
        + HIDDEN_SIZE
        + ACTION_COUNT * HIDDEN_SIZE
        + ACTION_COUNT
    )
    return {
        "observations": torch.randn(
            (ROW_COUNT, OBSERVATION_SIZE),
            generator=generator,
            device=device,
            dtype=torch.float32,
        ).contiguous(),
        "slots": torch.tensor(SLOTS, device=device, dtype=torch.int32),
        "mlp_pack": (
            torch.randn(
                (CAPACITY, packed_width),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            * 0.15
        ).contiguous(),
        "drives": torch.rand(
            (CAPACITY, 4), generator=generator, device=device, dtype=torch.float32
        ).contiguous(),
        "attack_bias": torch.linspace(-0.4, 0.5, CAPACITY, device=device),
        "action_temperature": torch.linspace(0.02, 1.4, CAPACITY, device=device),
        "drive_logit_gain": torch.linspace(-0.3, 1.1, CAPACITY, device=device),
        "slot_tokens": torch.arange(100, 100 + CAPACITY, device=device, dtype=torch.int64),
        "action_cache_actions": torch.full(
            (CAPACITY, CACHE_CAPACITY), -1, device=device, dtype=torch.int32
        ),
        "action_cache_token": torch.full((CAPACITY,), -1, device=device, dtype=torch.int64),
        "action_cache_cursor": torch.full((CAPACITY,), -1, device=device, dtype=torch.int32),
        "action_cache_next": torch.full((CAPACITY,), -1, device=device, dtype=torch.int32),
    }


def _fixed_mlp_oracle(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    slots = tensors["slots"].to(torch.int64)
    pack = tensors["mlp_pack"].index_select(0, slots)
    observations = tensors["observations"]
    offset = 0
    w1 = pack[:, offset : offset + HIDDEN_SIZE * OBSERVATION_SIZE].view(
        -1, HIDDEN_SIZE, OBSERVATION_SIZE
    )
    offset += HIDDEN_SIZE * OBSERVATION_SIZE
    b1 = pack[:, offset : offset + HIDDEN_SIZE]
    offset += HIDDEN_SIZE
    w2 = pack[:, offset : offset + HIDDEN_SIZE * HIDDEN_SIZE].view(
        -1, HIDDEN_SIZE, HIDDEN_SIZE
    )
    offset += HIDDEN_SIZE * HIDDEN_SIZE
    b2 = pack[:, offset : offset + HIDDEN_SIZE]
    offset += HIDDEN_SIZE
    w3 = pack[:, offset : offset + ACTION_COUNT * HIDDEN_SIZE].view(
        -1, ACTION_COUNT, HIDDEN_SIZE
    )
    offset += ACTION_COUNT * HIDDEN_SIZE
    b3 = pack[:, offset : offset + ACTION_COUNT]

    h1 = torch.tanh(torch.bmm(w1, observations.unsqueeze(-1)).squeeze(-1) + b1)
    h2 = torch.tanh(torch.bmm(w2, h1.unsqueeze(-1)).squeeze(-1) + b2)
    logits = torch.bmm(w3, h2.unsqueeze(-1)).squeeze(-1) + b3
    logits[:, 6] += tensors["attack_bias"].index_select(0, slots)
    temperature = tensors["action_temperature"].index_select(0, slots).clamp(min=0.05)
    logits = logits / temperature.unsqueeze(-1)
    drives = tensors["drives"].index_select(0, slots)
    gain = tensors["drive_logit_gain"].index_select(0, slots).clamp(min=0.0)
    logits[:, 4] += gain * (1.2 * drives[:, 0])
    logits[:, 5] += gain * (0.9 * drives[:, 1] + drives[:, 2])
    logits[:, 6] -= gain * (1.2 * drives[:, 1] + 1.4 * drives[:, 3])
    return logits


def _call(tensors: dict[str, torch.Tensor], build: Path, **overrides: object):
    arguments: dict[str, object] = {
        **tensors,
        "hidden_size": HIDDEN_SIZE,
        "action_count": ACTION_COUNT,
        "cache_width": CACHE_WIDTH,
        "deterministic": True,
        "rng_seed": 0x1234_5678,
        "rng_counter": 900,
        "build_directory": build,
    }
    arguments.update(overrides)
    return fixed_mlp_refill_cuda(**arguments)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fixed_mlp_deterministic_logits_actions_and_cache_match_oracle(tmp_path: Path) -> None:
    tensors = _cuda_fixture()
    expected_logits = _fixed_mlp_oracle(tensors)
    expected_actions = expected_logits.argmax(dim=1).to(torch.int32)
    untouched = sorted(set(range(CAPACITY)) - set(SLOTS))
    cache_pointer = tensors["action_cache_actions"].data_ptr()
    observation_pointer = tensors["observations"].data_ptr()
    policy_pointer = tensors["mlp_pack"].data_ptr()

    result = _call(tensors, tmp_path / "build")

    assert torch.equal(result.logits, expected_logits)
    assert torch.equal(result.actions, expected_actions)
    assert result.action_cache_actions.data_ptr() == cache_pointer
    assert tensors["observations"].data_ptr() == observation_pointer
    assert tensors["mlp_pack"].data_ptr() == policy_pointer
    assert tensors["observations"].is_cuda
    assert tensors["mlp_pack"].is_cuda
    assert result.action_cache_actions.is_cuda
    assert result.actions.is_cuda
    assert result.logits.is_cuda
    assert result.next_rng_counter.is_cuda
    selected_cache = result.action_cache_actions.index_select(
        0, tensors["slots"].to(torch.int64)
    )
    assert torch.equal(
        selected_cache[:, :CACHE_WIDTH],
        expected_actions.unsqueeze(1).expand(-1, CACHE_WIDTH),
    )
    assert torch.equal(selected_cache[:, CACHE_WIDTH:], torch.full_like(selected_cache[:, CACHE_WIDTH:], -1))
    assert torch.equal(
        result.action_cache_token.index_select(0, tensors["slots"].to(torch.int64)),
        tensors["slot_tokens"].index_select(0, tensors["slots"].to(torch.int64)),
    )
    assert torch.equal(
        result.action_cache_cursor.index_select(0, tensors["slots"].to(torch.int64)),
        torch.ones(ROW_COUNT, device="cuda", dtype=torch.int32),
    )
    assert torch.equal(
        result.action_cache_next.index_select(0, tensors["slots"].to(torch.int64)),
        expected_actions,
    )
    assert torch.equal(
        result.action_cache_actions[untouched],
        torch.full((len(untouched), CACHE_CAPACITY), -1, device="cuda", dtype=torch.int32),
    )
    assert torch.equal(result.next_rng_counter, torch.tensor([900], device="cuda", dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fixed_mlp_batched_counted_matches_isolated_cells_exactly(tmp_path: Path) -> None:
    sources = [_cuda_fixture(), _cuda_fixture()]
    sources[1]["observations"].mul_(0.75)
    sources[1]["mlp_pack"].mul_(1.1)
    active_rows = (ROW_COUNT, ROW_COUNT - 3)
    isolated = []
    for source, count in zip(sources, active_rows, strict=True):
        isolated.append(
            fixed_mlp_refill_counted_cuda(
                **source,
                active_count=torch.tensor([count], device="cuda", dtype=torch.int64),
                hidden_size=HIDDEN_SIZE,
                action_count=ACTION_COUNT,
                cache_width=CACHE_WIDTH,
                deterministic=True,
                rng_seed=0x1234_5678,
                rng_counter=900,
                build_directory=tmp_path / "build",
            )
        )

    batched_sources = [_cuda_fixture(), _cuda_fixture()]
    batched_sources[1]["observations"].mul_(0.75)
    batched_sources[1]["mlp_pack"].mul_(1.1)
    batched = fixed_mlp_refill_batched_counted_cuda(
        observations=torch.stack(
            [source["observations"] for source in batched_sources]
        ).contiguous(),
        slots=torch.stack([source["slots"] for source in batched_sources]).contiguous(),
        active_count=torch.tensor(
            [[count] for count in active_rows], device="cuda", dtype=torch.int64
        ),
        mlp_pack=torch.stack(
            [source["mlp_pack"] for source in batched_sources]
        ).contiguous(),
        drives=torch.stack([source["drives"] for source in batched_sources]).contiguous(),
        attack_bias=torch.stack(
            [source["attack_bias"] for source in batched_sources]
        ).contiguous(),
        action_temperature=torch.stack(
            [source["action_temperature"] for source in batched_sources]
        ).contiguous(),
        drive_logit_gain=torch.stack(
            [source["drive_logit_gain"] for source in batched_sources]
        ).contiguous(),
        slot_tokens=torch.stack(
            [source["slot_tokens"] for source in batched_sources]
        ).contiguous(),
        action_cache_actions=torch.stack(
            [source["action_cache_actions"] for source in batched_sources]
        ).contiguous(),
        action_cache_token=torch.stack(
            [source["action_cache_token"] for source in batched_sources]
        ).contiguous(),
        action_cache_cursor=torch.stack(
            [source["action_cache_cursor"] for source in batched_sources]
        ).contiguous(),
        action_cache_next=torch.stack(
            [source["action_cache_next"] for source in batched_sources]
        ).contiguous(),
        hidden_size=HIDDEN_SIZE,
        action_count=ACTION_COUNT,
        cache_width=CACHE_WIDTH,
        deterministic=True,
        rng_seed=0x1234_5678,
        rng_counter=900,
        build_directory=tmp_path / "build",
    )
    for cell, expected in enumerate(isolated):
        assert torch.equal(batched.logits[cell], expected.logits)
        assert torch.equal(batched.actions[cell], expected.actions)
        assert torch.equal(
            batched.action_cache_actions[cell], expected.action_cache_actions
        )
        assert torch.equal(batched.action_cache_token[cell], expected.action_cache_token)
        assert torch.equal(
            batched.action_cache_cursor[cell], expected.action_cache_cursor
        )
        assert torch.equal(batched.action_cache_next[cell], expected.action_cache_next)
        assert torch.equal(batched.next_rng_counter[cell], expected.next_rng_counter)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fixed_mlp_stochastic_cache_is_reproducible_and_counter_bounded(tmp_path: Path) -> None:
    first_tensors = _cuda_fixture()
    second_tensors = {name: tensor.clone() for name, tensor in first_tensors.items()}
    third_tensors = {name: tensor.clone() for name, tensor in first_tensors.items()}
    first = _call(first_tensors, tmp_path / "build", deterministic=False)
    second = _call(second_tensors, tmp_path / "build", deterministic=False)
    shifted = _call(third_tensors, tmp_path / "build", deterministic=False, rng_counter=901)
    selected_slots = first_tensors["slots"].to(torch.int64)
    consumed = ROW_COUNT * CACHE_WIDTH * ACTION_COUNT

    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.actions, second.actions)
    assert torch.equal(first.action_cache_actions, second.action_cache_actions)
    assert not torch.equal(first.action_cache_actions, shifted.action_cache_actions)
    assert torch.equal(
        first.actions,
        first.action_cache_actions.index_select(0, selected_slots)[:, 0],
    )
    assert bool(
        first.action_cache_actions.index_select(0, selected_slots)[:, :CACHE_WIDTH]
        .ge(0)
        .logical_and(
            first.action_cache_actions.index_select(0, selected_slots)[:, :CACHE_WIDTH]
            .lt(ACTION_COUNT)
        )
        .all()
    )
    assert torch.equal(
        first.next_rng_counter,
        torch.tensor([900 + consumed], device="cuda", dtype=torch.int64),
    )
    assert first.pcg64_state_dict() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fixed_mlp_stochastic_cache_and_pcg64_state_match_cpu_oracle(tmp_path: Path) -> None:
    tensors = _cuda_fixture()
    logits = _fixed_mlp_oracle(tensors).cpu().numpy()
    bitgen = np.random.PCG64(303)
    initial_state = copy.deepcopy(bitgen.state)
    generator = np.random.Generator(bitgen)
    uniforms = generator.random(ROW_COUNT * CACHE_WIDTH * ACTION_COUNT, dtype=np.float32)
    expected_state = copy.deepcopy(bitgen.state)
    expected = np.empty((ROW_COUNT, CACHE_WIDTH), dtype=np.int32)
    draw = 0
    for row in range(ROW_COUNT):
        for cache_index in range(CACHE_WIDTH):
            scores = []
            for action in range(ACTION_COUNT):
                uniform = np.clip(uniforms[draw], np.float32(1.0e-6), np.float32(1.0 - 1.0e-6))
                draw += 1
                gumbel = -np.log(-np.log(uniform))
                scores.append(np.float32(logits[row, action]) + np.float32(gumbel))
            expected[row, cache_index] = int(np.argmax(np.asarray(scores, dtype=np.float32)))

    result = _call(
        tensors,
        tmp_path / "build",
        deterministic=False,
        pcg64_state=initial_state,
    )
    selected = result.action_cache_actions.index_select(0, tensors["slots"].to(torch.int64))

    assert torch.equal(selected[:, :CACHE_WIDTH].cpu(), torch.from_numpy(expected))
    assert torch.equal(result.actions.cpu(), torch.from_numpy(expected[:, 0]))
    assert result.pcg64_state_dict() == expected_state
    assert torch.equal(result.next_rng_counter, torch.tensor([900], device="cuda", dtype=torch.int64))


def test_fixed_mlp_capability_excludes_gru_before_extension_load(tmp_path: Path) -> None:
    assert CUDA_FIXED_MLP_CAPABILITIES["fixed_mlp_policy_resident"] is True
    assert CUDA_FIXED_MLP_CAPABILITIES["gru_supported"] is False
    assert CUDA_FIXED_MLP_CAPABILITIES["host_sampling"] is False
    with pytest.raises(RuntimeError, match="GRU is excluded"):
        fixed_mlp_refill_cuda(
            observations=torch.empty(0),
            slots=torch.empty(0),
            mlp_pack=torch.empty(0),
            drives=torch.empty(0),
            attack_bias=torch.empty(0),
            action_temperature=torch.empty(0),
            drive_logit_gain=torch.empty(0),
            slot_tokens=torch.empty(0),
            action_cache_actions=torch.empty(0),
            action_cache_token=torch.empty(0),
            action_cache_cursor=torch.empty(0),
            action_cache_next=torch.empty(0),
            hidden_size=HIDDEN_SIZE,
            action_count=ACTION_COUNT,
            cache_width=CACHE_WIDTH,
            deterministic=True,
            policy_family="gru",
            build_directory=tmp_path / "unused",
        )
