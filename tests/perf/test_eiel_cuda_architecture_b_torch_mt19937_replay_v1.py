from __future__ import annotations

import struct
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.cpp_extension import load_inline


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engine/perf/cuda_architecture_b_torch_mt19937_replay_v1.cuh"
FULL_STATE_BYTES = 5056
MANDATORY_COUNTS = (0, 1, 2, 623, 624, 625, 1247, 1248, 1249, 9415, 18830)
QUALIFIED_SEEDS = (
    0,
    1,
    0x0000_0000_DEAD_BEEF,
    0x1234_5678_DEAD_BEEF,
    0xFFFF_FFFF_FFFF_FFFF,
)


def test_live_torch_identity_matches_qualified_d1_source() -> None:
    assert torch.__version__ == "2.10.0+cu128"
    assert torch.version.git_version == (
        "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
    )


@pytest.fixture(scope="module")
def replay_module(tmp_path_factory: pytest.TempPathFactory) -> Any:
    cpp_source = r"""
    #include <torch/extension.h>
    #include <ATen/CPUGeneratorImpl.h>

    #include <cstdint>

    std::vector<torch::Tensor> replay_states_cuda(
        torch::Tensor initial_states,
        torch::Tensor raw_counts
    );
    std::vector<torch::Tensor> replay_states_with_launch_threads_cuda(
        torch::Tensor initial_states,
        torch::Tensor raw_counts,
        int64_t launch_blocks,
        int64_t launch_threads
    );
    std::vector<torch::Tensor> replay_raw_uint32_cuda(
        torch::Tensor initial_states,
        torch::Tensor raw_counts
    );
    std::vector<torch::Tensor> replay_float32_bits_cuda(
        torch::Tensor initial_states,
        torch::Tensor raw_counts
    );
    std::vector<torch::Tensor> replay_packed_crossover_mask_cuda(
        torch::Tensor initial_states,
        int64_t logical_width
    );

    std::vector<torch::Tensor> raw_words_from_state_cpu(
        torch::Tensor state,
        int64_t count
    ) {
        TORCH_CHECK(state.device().is_cpu(), "state must be CPU resident");
        TORCH_CHECK(state.scalar_type() == torch::kUInt8, "state must be uint8");
        TORCH_CHECK(state.dim() == 1, "state must be rank 1");
        TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
        TORCH_CHECK(count >= 0, "count must be nonnegative");
        auto canonical_state = state.clone().contiguous();
        at::Generator generator = at::detail::createCPUGenerator(0);
        generator.set_state(canonical_state);
        auto* implementation = generator.get<at::CPUGeneratorImpl>();
        auto output = torch::empty(
            {count},
            torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU)
        );
        auto* words = output.data_ptr<std::uint32_t>();
        for (int64_t index = 0; index < count; ++index) {
            words[index] = implementation->random();
        }
        return {output, generator.get_state().clone().contiguous()};
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
        module.def("replay_states", &replay_states_cuda);
        module.def(
            "replay_states_with_launch_threads",
            &replay_states_with_launch_threads_cuda
        );
        module.def("replay_raw_uint32", &replay_raw_uint32_cuda);
        module.def("replay_float32_bits", &replay_float32_bits_cuda);
        module.def("replay_packed_crossover_mask", &replay_packed_crossover_mask_cuda);
        module.def("raw_words_from_state_cpu", &raw_words_from_state_cpu);
    }
    """
    cuda_source = r"""
    #include <torch/extension.h>

    #include <ATen/cuda/CUDAContext.h>
    #include <c10/cuda/CUDAGuard.h>
    #include <c10/cuda/CUDAException.h>
    #include <cuda.h>
    #include <cuda_runtime.h>

    #include <cstdint>
    #include <vector>

    #include "cuda_architecture_b_torch_mt19937_replay_v1.cuh"

    namespace replay =
        eiel::cuda_architecture_b::torch_mt19937_replay_v1;

    namespace {

    void check_state_rows(const torch::Tensor& states) {
        TORCH_CHECK(states.is_cuda(), "states must be a CUDA tensor");
        TORCH_CHECK(states.is_contiguous(), "states must be contiguous");
        TORCH_CHECK(states.scalar_type() == torch::kUInt8, "states must be uint8");
        TORCH_CHECK(
            states.dim() == 2 && states.size(1) == replay::kFullStateBytes,
            "states must be uint8[rows, 5056]"
        );
    }

    void check_counts(const torch::Tensor& counts, const std::int64_t rows) {
        TORCH_CHECK(counts.is_cuda(), "counts must be a CUDA tensor");
        TORCH_CHECK(counts.is_contiguous(), "counts must be contiguous");
        TORCH_CHECK(counts.scalar_type() == torch::kInt64, "counts must be int64");
        TORCH_CHECK(counts.dim() == 1 && counts.size(0) == rows, "bad counts shape");
    }

    __global__ void replay_states_kernel(
        const std::uint8_t* const initial,
        const std::int64_t* const counts,
        std::uint8_t* const output,
        bool* const status,
        const std::int64_t rows
    ) {
        const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
        if (row >= rows || threadIdx.x != 0) {
            return;
        }
        const std::uint8_t* const input_row =
            initial + row * replay::kFullStateBytes;
        std::uint8_t* const output_row =
            output + row * replay::kFullStateBytes;
        for (std::int64_t index = 0; index < replay::kFullStateBytes; ++index) {
            output_row[index] = input_row[index];
        }
        const std::int64_t count = counts[row];
        if (count < 0) {
            status[row] = false;
            return;
        }
        replay::TorchMt19937StateV1 state{};
        if (!replay::load_canonical_state_v1(
                input_row,
                replay::kFullStateBytes,
                &state
            )) {
            status[row] = false;
            return;
        }
        replay::discard_uint32_v1(
            &state,
            static_cast<std::uint64_t>(count)
        );
        status[row] = replay::store_engine_state_v1(
            &state,
            output_row,
            replay::kFullStateBytes
        );
    }

    __global__ void replay_float32_bits_kernel(
        const std::uint8_t* const initial,
        const std::int64_t* const counts,
        std::uint32_t* const bits,
        bool* const status,
        const std::int64_t rows,
        const std::int64_t columns
    ) {
        const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
        if (row >= rows || threadIdx.x != 0) {
            return;
        }
        replay::TorchMt19937StateV1 state{};
        if (!replay::load_canonical_state_v1(
                initial + row * replay::kFullStateBytes,
                replay::kFullStateBytes,
                &state
            )) {
            status[row] = false;
            return;
        }
        const std::int64_t count = counts[row];
        if (count < 0 || count > columns) {
            status[row] = false;
            return;
        }
        std::uint32_t* const row_bits = bits + row * columns;
        for (std::int64_t index = 0; index < count; ++index) {
            row_bits[index] = replay::torch_rand_float32_bits_v1(
                replay::next_uint32_v1(&state)
            );
        }
        status[row] = true;
    }

    __global__ void replay_raw_uint32_kernel(
        const std::uint8_t* const initial,
        const std::int64_t* const counts,
        std::int64_t* const raw_words,
        std::uint8_t* const terminal,
        bool* const status,
        const std::int64_t rows,
        const std::int64_t columns
    ) {
        const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
        if (row >= rows || threadIdx.x != 0) {
            return;
        }
        replay::TorchMt19937StateV1 state{};
        if (!replay::load_canonical_state_v1(
                initial + row * replay::kFullStateBytes,
                replay::kFullStateBytes,
                &state
            )) {
            status[row] = false;
            return;
        }
        const std::int64_t count = counts[row];
        if (count < 0 || count > columns) {
            status[row] = false;
            return;
        }
        std::uint8_t* const terminal_row =
            terminal + row * replay::kFullStateBytes;
        for (std::int64_t index = 0; index < replay::kFullStateBytes; ++index) {
            terminal_row[index] =
                initial[row * replay::kFullStateBytes + index];
        }
        std::int64_t* const row_raw = raw_words + row * columns;
        for (std::int64_t index = 0; index < count; ++index) {
            row_raw[index] =
                static_cast<std::int64_t>(replay::next_uint32_v1(&state));
        }
        status[row] = replay::store_engine_state_v1(
            &state,
            terminal_row,
            replay::kFullStateBytes
        );
    }

    __global__ void replay_packed_crossover_mask_kernel(
        const std::uint8_t* const initial,
        std::uint8_t* const packed,
        std::uint8_t* const terminal,
        bool* const status,
        const std::int64_t rows,
        const std::int64_t logical_width,
        const std::int64_t row_bytes
    ) {
        const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
        if (row >= rows || threadIdx.x != 0) {
            return;
        }
        replay::TorchMt19937StateV1 state{};
        if (!replay::load_canonical_state_v1(
                initial + row * replay::kFullStateBytes,
                replay::kFullStateBytes,
                &state
            )) {
            status[row] = false;
            return;
        }
        const std::uint8_t* const initial_row =
            initial + row * replay::kFullStateBytes;
        std::uint8_t* const terminal_row =
            terminal + row * replay::kFullStateBytes;
        for (std::int64_t index = 0; index < replay::kFullStateBytes; ++index) {
            terminal_row[index] = initial_row[index];
        }
        status[row] = replay::generate_packed_crossover_mask_lsb0_v1(
            &state,
            packed + row * row_bytes,
            logical_width
        ) && replay::store_engine_state_v1(&state, terminal_row, replay::kFullStateBytes);
    }

    }  // namespace

    std::vector<torch::Tensor> replay_states_cuda(
        torch::Tensor initial_states,
        torch::Tensor raw_counts
    ) {
        check_state_rows(initial_states);
        check_counts(raw_counts, initial_states.size(0));
        const c10::cuda::CUDAGuard device_guard(initial_states.device());
        auto output = torch::empty_like(initial_states);
        auto status = torch::zeros(
            {initial_states.size(0)}, initial_states.options().dtype(torch::kBool)
        );
        if (initial_states.size(0) > 0) {
            replay_states_kernel<<<
                static_cast<unsigned int>(initial_states.size(0)),
                1,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                initial_states.data_ptr<std::uint8_t>(),
                raw_counts.data_ptr<std::int64_t>(),
                output.data_ptr<std::uint8_t>(),
                status.data_ptr<bool>(),
                initial_states.size(0)
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return {output, status};
    }

    std::vector<torch::Tensor> replay_states_with_launch_threads_cuda(
        torch::Tensor initial_states,
        torch::Tensor raw_counts,
        int64_t launch_blocks,
        int64_t launch_threads
    ) {
        check_state_rows(initial_states);
        check_counts(raw_counts, initial_states.size(0));
        TORCH_CHECK(
            launch_blocks >= initial_states.size(0) && launch_blocks <= 65535,
            "launch_blocks must cover rows and remain bounded"
        );
        TORCH_CHECK(
            launch_threads >= 1 && launch_threads <= 1024,
            "launch_threads must be in [1, 1024]"
        );
        const c10::cuda::CUDAGuard device_guard(initial_states.device());
        auto output = torch::empty_like(initial_states);
        auto status = torch::zeros(
            {initial_states.size(0)}, initial_states.options().dtype(torch::kBool)
        );
        if (initial_states.size(0) > 0) {
            replay_states_kernel<<<
                static_cast<unsigned int>(launch_blocks),
                static_cast<unsigned int>(launch_threads),
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                initial_states.data_ptr<std::uint8_t>(),
                raw_counts.data_ptr<std::int64_t>(),
                output.data_ptr<std::uint8_t>(),
                status.data_ptr<bool>(),
                initial_states.size(0)
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return {output, status};
    }

    std::vector<torch::Tensor> replay_raw_uint32_cuda(
        torch::Tensor initial_states,
        torch::Tensor raw_counts
    ) {
        check_state_rows(initial_states);
        check_counts(raw_counts, initial_states.size(0));
        const c10::cuda::CUDAGuard device_guard(initial_states.device());
        const std::int64_t columns = raw_counts.numel() == 0
            ? 0
            : raw_counts.max().item<std::int64_t>();
        TORCH_CHECK(columns >= 0, "counts must be nonnegative");
        auto raw_words = torch::zeros(
            {initial_states.size(0), columns},
            initial_states.options().dtype(torch::kInt64)
        );
        auto terminal = torch::empty_like(initial_states);
        auto status = torch::zeros(
            {initial_states.size(0)}, initial_states.options().dtype(torch::kBool)
        );
        if (initial_states.size(0) > 0) {
            replay_raw_uint32_kernel<<<
                static_cast<unsigned int>(initial_states.size(0)),
                1,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                initial_states.data_ptr<std::uint8_t>(),
                raw_counts.data_ptr<std::int64_t>(),
                raw_words.data_ptr<std::int64_t>(),
                terminal.data_ptr<std::uint8_t>(),
                status.data_ptr<bool>(),
                initial_states.size(0),
                columns
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return {raw_words, terminal, status};
    }

    std::vector<torch::Tensor> replay_float32_bits_cuda(
        torch::Tensor initial_states,
        torch::Tensor raw_counts
    ) {
        check_state_rows(initial_states);
        check_counts(raw_counts, initial_states.size(0));
        const c10::cuda::CUDAGuard device_guard(initial_states.device());
        const std::int64_t columns = raw_counts.numel() == 0
            ? 0
            : raw_counts.max().item<std::int64_t>();
        TORCH_CHECK(columns >= 0, "counts must be nonnegative");
        auto bits = torch::zeros(
            {initial_states.size(0), columns},
            initial_states.options().dtype(torch::kInt32)
        );
        auto status = torch::zeros(
            {initial_states.size(0)}, initial_states.options().dtype(torch::kBool)
        );
        if (initial_states.size(0) > 0) {
            replay_float32_bits_kernel<<<
                static_cast<unsigned int>(initial_states.size(0)),
                1,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                initial_states.data_ptr<std::uint8_t>(),
                raw_counts.data_ptr<std::int64_t>(),
                reinterpret_cast<std::uint32_t*>(bits.data_ptr<std::int32_t>()),
                status.data_ptr<bool>(),
                initial_states.size(0),
                columns
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return {bits, status};
    }

    std::vector<torch::Tensor> replay_packed_crossover_mask_cuda(
        torch::Tensor initial_states,
        int64_t logical_width
    ) {
        check_state_rows(initial_states);
        TORCH_CHECK(logical_width >= 0, "logical_width must be nonnegative");
        const c10::cuda::CUDAGuard device_guard(initial_states.device());
        const std::int64_t row_bytes =
            replay::packed_lsb0_row_bytes_v1(logical_width);
        auto packed = torch::zeros(
            {initial_states.size(0), row_bytes},
            initial_states.options().dtype(torch::kUInt8)
        );
        auto terminal = torch::empty_like(initial_states);
        auto status = torch::zeros(
            {initial_states.size(0)}, initial_states.options().dtype(torch::kBool)
        );
        if (initial_states.size(0) > 0) {
            replay_packed_crossover_mask_kernel<<<
                static_cast<unsigned int>(initial_states.size(0)),
                1,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                initial_states.data_ptr<std::uint8_t>(),
                packed.data_ptr<std::uint8_t>(),
                terminal.data_ptr<std::uint8_t>(),
                status.data_ptr<bool>(),
                initial_states.size(0),
                logical_width,
                row_bytes
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return {packed, terminal, status};
    }
    """
    digest = sha256(
        HEADER.read_bytes() + cpp_source.encode() + cuda_source.encode()
    ).hexdigest()[:16]
    return load_inline(
        name=f"eiel_torch_mt19937_replay_v1_{digest}",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        build_directory=str(tmp_path_factory.mktemp("torch-mt19937-replay-build")),
        extra_include_paths=[str(HEADER.parent)],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
    )


def _seed_state(seed: int) -> torch.Tensor:
    return torch.Generator(device="cpu").manual_seed(seed).get_state().clone()


def _state_after(seed: int, draws: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.rand((draws,), generator=generator, dtype=torch.float32)
    return generator.get_state().clone()


def _state_after_state(initial_state: torch.Tensor, draws: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.set_state(initial_state.contiguous().cpu())
    torch.rand((draws,), generator=generator, dtype=torch.float32)
    return generator.get_state().clone()


def _float32_bits(seed: int, draws: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.rand((draws,), generator=generator, dtype=torch.float32)
    return values.view(torch.int32).cpu()


def _packed_crossover_mask(seed: int, draws: int) -> torch.Tensor:
    from engine.perf.cuda_architecture_b_child_policy import pack_crossover_mask_lsb0_v1

    generator = torch.Generator(device="cpu").manual_seed(seed)
    mask = torch.rand((draws,), generator=generator, dtype=torch.float32).lt(0.5)
    return pack_crossover_mask_lsb0_v1(mask).cpu()


def _raw_u32_sha256(values: list[int]) -> str:
    payload = b"".join(struct.pack("<I", int(value) & 0xFFFF_FFFF) for value in values)
    return sha256(payload).hexdigest()


@pytest.mark.parametrize("seed", QUALIFIED_SEEDS)
def test_device_replay_advances_canonical_state_at_twist_boundaries(
    replay_module: Any,
    seed: int,
) -> None:
    counts = list(MANDATORY_COUNTS)
    initial = torch.stack([_seed_state(seed) for _ in counts]).contiguous()
    raw_counts = torch.tensor(counts, dtype=torch.int64, device="cuda")

    actual, status = replay_module.replay_states(initial.cuda(), raw_counts)
    torch.cuda.synchronize()

    expected = torch.stack([_state_after(seed, count) for count in counts])
    assert status.cpu().tolist() == [True] * len(counts)
    assert torch.equal(actual.cpu(), expected)


def test_device_replay_state_only_discard_is_launch_and_permutation_invariant(
    replay_module: Any,
) -> None:
    seeds = list(QUALIFIED_SEEDS)
    counts = [0, 1, 623, 624, 18830]
    initial = torch.stack([_seed_state(seed) for seed in seeds]).contiguous()
    raw_counts = torch.tensor(counts, dtype=torch.int64, device="cuda")
    baseline, baseline_status = replay_module.replay_states(
        initial.cuda(),
        raw_counts,
    )
    torch.cuda.synchronize()
    assert baseline_status.cpu().tolist() == [True] * len(seeds)

    for launch_blocks, launch_threads in (
        (len(seeds), 1),
        (len(seeds) + 3, 32),
        (len(seeds) + 11, 256),
    ):
        actual, status = replay_module.replay_states_with_launch_threads(
            initial.cuda(),
            raw_counts,
            launch_blocks,
            launch_threads,
        )
        torch.cuda.synchronize()
        assert status.cpu().tolist() == [True] * len(seeds)
        assert torch.equal(actual.cpu(), baseline.cpu())

    permutation = [3, 0, 4, 1, 2]
    permuted, status = replay_module.replay_states_with_launch_threads(
        initial[permutation].contiguous().cuda(),
        raw_counts.cpu()[permutation].contiguous().cuda(),
        len(seeds) + 7,
        64,
    )
    torch.cuda.synchronize()
    assert status.cpu().tolist() == [True] * len(seeds)
    for permuted_row, original_row in enumerate(permutation):
        assert torch.equal(permuted.cpu()[permuted_row], baseline.cpu()[original_row])


@pytest.mark.parametrize(
    ("prelude", "first_count", "second_count"),
    [
        (0, 0, 1),
        (1, 622, 2),
        (623, 1, 624),
        (624, 625, 624),
        (9415, 623, 625),
    ],
)
def test_device_replay_state_only_discard_split_continuation(
    replay_module: Any,
    prelude: int,
    first_count: int,
    second_count: int,
) -> None:
    seed = 0x1234_5678_DEAD_BEEF
    initial = _state_after(seed, prelude).reshape(1, FULL_STATE_BYTES).cuda()
    first, first_status = replay_module.replay_states(
        initial,
        torch.tensor([first_count], dtype=torch.int64, device="cuda"),
    )
    terminal, second_status = replay_module.replay_states(
        first,
        torch.tensor([second_count], dtype=torch.int64, device="cuda"),
    )
    torch.cuda.synchronize()

    assert first_status.cpu().tolist() == [True]
    assert second_status.cpu().tolist() == [True]
    assert torch.equal(
        terminal.cpu()[0],
        _state_after(seed, prelude + first_count + second_count),
    )


def test_device_replay_state_only_discard_matches_ordinary_birth_schedule(
    replay_module: Any,
) -> None:
    birth_counts = [0, 1, 2, 8]
    initial = torch.stack([_seed_state(42) for _ in birth_counts]).contiguous()
    raw_counts = torch.tensor(
        [births * 9415 for births in birth_counts],
        dtype=torch.int64,
        device="cuda",
    )

    terminal, status = replay_module.replay_states(initial.cuda(), raw_counts)
    torch.cuda.synchronize()

    assert status.cpu().tolist() == [True] * len(birth_counts)
    expected = torch.stack(
        [_state_after(42, births * 9415) for births in birth_counts]
    )
    assert torch.equal(terminal.cpu(), expected)


def test_device_replay_preserves_normal_cache_bytes_during_uniform_discard(
    replay_module: Any,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(717)
    torch.randn((1,), generator=generator, dtype=torch.float32)
    initial = generator.get_state().clone()
    expected = _state_after_state(initial, 625)
    assert not torch.equal(initial[-40:], _seed_state(717)[-40:])

    terminal, status = replay_module.replay_states(
        initial.reshape(1, FULL_STATE_BYTES).cuda(),
        torch.tensor([625], dtype=torch.int64, device="cuda"),
    )
    torch.cuda.synchronize()

    assert status.cpu().tolist() == [True]
    assert torch.equal(terminal.cpu()[0], expected)
    assert torch.equal(terminal.cpu()[0, -40:], initial[-40:])


def test_device_replay_float32_conversion_matches_torch_rand_bits(
    replay_module: Any,
) -> None:
    counts = list(MANDATORY_COUNTS)
    seeds = list(QUALIFIED_SEEDS)
    initial_rows = []
    count_rows = []
    expected_rows = []
    max_count = max(counts)
    for seed in seeds:
        for count in counts:
            initial_rows.append(_seed_state(seed))
            count_rows.append(count)
            expected = torch.zeros((max_count,), dtype=torch.int32)
            expected[:count] = _float32_bits(seed, count)
            expected_rows.append(expected)
    initial = torch.stack(initial_rows).contiguous()
    raw_counts = torch.tensor(count_rows, dtype=torch.int64, device="cuda")

    actual, status = replay_module.replay_float32_bits(initial.cuda(), raw_counts)
    torch.cuda.synchronize()

    assert status.cpu().tolist() == [True] * len(count_rows)
    assert torch.equal(actual.cpu(), torch.stack(expected_rows))


def test_device_replay_raw_uint32_matches_live_cpu_generator_for_all_counts(
    replay_module: Any,
) -> None:
    rows = [(seed, count) for seed in QUALIFIED_SEEDS for count in MANDATORY_COUNTS]
    initial = torch.stack([_seed_state(seed) for seed, _ in rows]).contiguous()
    raw_counts = torch.tensor(
        [count for _, count in rows],
        dtype=torch.int64,
        device="cuda",
    )

    actual, terminal, status = replay_module.replay_raw_uint32(
        initial.cuda(),
        raw_counts,
    )
    torch.cuda.synchronize()

    actual_cpu = actual.cpu()
    terminal_cpu = terminal.cpu()
    assert status.cpu().tolist() == [True] * len(rows)
    for row, ((_, count), initial_state) in enumerate(zip(rows, initial, strict=True)):
        expected_raw, expected_terminal = replay_module.raw_words_from_state_cpu(
            initial_state,
            count,
        )
        assert torch.equal(
            actual_cpu[row, :count],
            expected_raw.to(torch.int64),
        )
        assert torch.equal(terminal_cpu[row], expected_terminal)


def test_device_replay_raw_uint32_matches_option_d_bootstrap_vectors(
    replay_module: Any,
) -> None:
    initial = torch.stack([_seed_state(42), _seed_state(42), _seed_state(1)]).contiguous()
    raw_counts = torch.tensor([625, 9415, 1249], dtype=torch.int64, device="cuda")

    actual, terminal, status = replay_module.replay_raw_uint32(
        initial.cuda(),
        raw_counts,
    )
    torch.cuda.synchronize()

    rows = actual.cpu().tolist()
    assert status.cpu().tolist() == [True, True, True]
    assert rows[0][:10] == [
        1608637542,
        3421126067,
        4083286876,
        787846414,
        3143890026,
        3348747335,
        2571218620,
        2563451924,
        670094950,
        1914837113,
    ]
    assert _raw_u32_sha256(rows[0][:625]) == (
        "b640f231a91a083a5487a2a103b79d09413d581f49d43907017a273f94781d2f"
    )
    assert _raw_u32_sha256(rows[1][:9415]) == (
        "6da64deab02bb16d963dd7de3338ab4932c163dfc2eb63c257619efb47b8e5d4"
    )
    assert _raw_u32_sha256(rows[2][:1249]) == (
        "6f85d69a34baff95d13f5186207d1b2b0c6d890397c31251be2b29925d98e7ea"
    )
    assert torch.equal(terminal.cpu()[0], _state_after(42, 625))
    assert torch.equal(terminal.cpu()[1], _state_after(42, 9415))
    assert torch.equal(terminal.cpu()[2], _state_after(1, 1249))


@pytest.mark.parametrize("width", MANDATORY_COUNTS)
def test_device_replay_generates_packed_crossover_mask_and_terminal_state(
    replay_module: Any,
    width: int,
) -> None:
    seeds = list(QUALIFIED_SEEDS)
    initial = torch.stack([_seed_state(seed) for seed in seeds]).contiguous()

    actual, terminal, status = replay_module.replay_packed_crossover_mask(
        initial.cuda(),
        width,
    )
    torch.cuda.synchronize()

    expected_masks = torch.stack([_packed_crossover_mask(seed, width) for seed in seeds])
    expected_states = torch.stack([_state_after(seed, width) for seed in seeds])
    assert status.cpu().tolist() == [True] * len(seeds)
    assert torch.equal(actual.cpu(), expected_masks)
    assert torch.equal(terminal.cpu(), expected_states)
    if width % 8:
        padding_mask = (~((1 << (width % 8)) - 1)) & 0xFF
        assert all((int(row[-1]) & padding_mask) == 0 for row in actual.cpu())


def test_device_replay_packed_crossover_split_continuation(
    replay_module: Any,
) -> None:
    seed = 42
    width = 9415
    initial = _seed_state(seed).reshape(1, FULL_STATE_BYTES).contiguous().cuda()

    first, mid_state, first_status = replay_module.replay_packed_crossover_mask(
        initial,
        width,
    )
    second, terminal, second_status = replay_module.replay_packed_crossover_mask(
        mid_state,
        width,
    )
    torch.cuda.synchronize()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    expected_first = torch.rand((width,), generator=generator, dtype=torch.float32).lt(0.5)
    expected_second = torch.rand((width,), generator=generator, dtype=torch.float32).lt(0.5)

    from engine.perf.cuda_architecture_b_child_policy import pack_crossover_mask_lsb0_v1

    assert first_status.cpu().tolist() == [True]
    assert second_status.cpu().tolist() == [True]
    assert torch.equal(first.cpu()[0], pack_crossover_mask_lsb0_v1(expected_first))
    assert torch.equal(second.cpu()[0], pack_crossover_mask_lsb0_v1(expected_second))
    assert torch.equal(terminal.cpu()[0], _state_after(seed, width * 2))


def test_device_replay_packed_crossover_uses_current_cuda_stream(
    replay_module: Any,
) -> None:
    seed = 1
    width = 625
    canonical = _seed_state(seed).reshape(1, FULL_STATE_BYTES).contiguous()
    initial = torch.zeros_like(canonical, device="cuda")

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        initial.copy_(canonical)
        actual, terminal, status = replay_module.replay_packed_crossover_mask(
            initial,
            width,
        )
    stream.synchronize()

    assert status.cpu().tolist() == [True]
    assert torch.equal(actual.cpu()[0], _packed_crossover_mask(seed, width))
    assert torch.equal(terminal.cpu()[0], _state_after(seed, width))


def test_device_replay_packed_crossover_supports_disjoint_concurrent_streams(
    replay_module: Any,
) -> None:
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    initial_a = _seed_state(1).reshape(1, FULL_STATE_BYTES).contiguous().cuda()
    initial_b = (
        _seed_state(0xFFFF_FFFF_FFFF_FFFF)
        .reshape(1, FULL_STATE_BYTES)
        .contiguous()
        .cuda()
    )

    with torch.cuda.stream(stream_a):
        actual_a, terminal_a, status_a = replay_module.replay_packed_crossover_mask(
            initial_a,
            1249,
        )
    with torch.cuda.stream(stream_b):
        actual_b, terminal_b, status_b = replay_module.replay_packed_crossover_mask(
            initial_b,
            9415,
        )
    stream_a.synchronize()
    stream_b.synchronize()

    assert status_a.cpu().tolist() == [True]
    assert status_b.cpu().tolist() == [True]
    assert torch.equal(actual_a.cpu()[0], _packed_crossover_mask(1, 1249))
    assert torch.equal(
        actual_b.cpu()[0],
        _packed_crossover_mask(0xFFFF_FFFF_FFFF_FFFF, 9415),
    )
    assert torch.equal(terminal_a.cpu()[0], _state_after(1, 1249))
    assert torch.equal(
        terminal_b.cpu()[0],
        _state_after(0xFFFF_FFFF_FFFF_FFFF, 9415),
    )


@pytest.mark.parametrize(
    ("offset", "replacement"),
    [
        (12, (0).to_bytes(4, "little")),
        (8, (0).to_bytes(4, "little")),
        (16, (625).to_bytes(8, "little")),
        (28, bytes([1, 0, 0, 0])),
    ],
)
def test_device_replay_rejects_noncanonical_states(
    replay_module: Any,
    offset: int,
    replacement: bytes,
) -> None:
    state = _seed_state(42)
    state[offset : offset + len(replacement)] = torch.tensor(
        list(replacement),
        dtype=torch.uint8,
    )

    actual, status = replay_module.replay_states(
        state.reshape(1, FULL_STATE_BYTES).cuda(),
        torch.tensor([1], dtype=torch.int64, device="cuda"),
    )
    torch.cuda.synchronize()

    assert status.cpu().tolist() == [False]
    assert not torch.equal(actual.cpu()[0], _state_after(42, 1))


def test_device_replay_rejects_unsupported_state_surfaces(
    replay_module: Any,
) -> None:
    counts = torch.tensor([1], dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match="uint8"):
        replay_module.replay_states(
            torch.zeros((1, FULL_STATE_BYTES), dtype=torch.int8, device="cuda"),
            counts,
        )
    with pytest.raises(RuntimeError, match=r"uint8\[rows, 5056\]"):
        replay_module.replay_states(
            torch.zeros((1, FULL_STATE_BYTES - 1), dtype=torch.uint8, device="cuda"),
            counts,
        )
    with pytest.raises(RuntimeError, match=r"launch_threads must be in \[1, 1024\]"):
        replay_module.replay_states_with_launch_threads(
            _seed_state(1).reshape(1, FULL_STATE_BYTES).cuda(),
            counts,
            1,
            0,
        )
