from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

pytest_plugins = ["tests.test_eiel_hapb_frag_a_policy_window_reuse_v1"]

from engine.hapb_rngv2.contract import NAMESPACE_DEFAULT, SEMANTIC_EPOCH
from engine.hapb_rngv2.cuda import generate_device, load_cuda_extension, native_sources_exist, philox_device
from engine.hapb_rngv2.oracle import HapbEvent, HapbOracle
from engine.hapb_rngv2.policy import fill_policy_window_v2
from engine.tw_rngv2.philox import PHILOX_KATS
from engine.tw_rngv2.status import Status


ROOT = Path(__file__).resolve().parents[1]
REFILL = ROOT / "engine/perf/cuda_architecture_b_resident_window_refill_v1.hpp"
WDL_RUNTIME = (
    ROOT / "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"
)


def test_native_hapb_rngv2_path_is_present() -> None:
    assert native_sources_exist() is True
    header = REFILL.read_text(encoding="utf-8")
    assert "make_fixed_mlp_policy_window_rngv2_v1" in header
    assert "hapb_rngv2_policy_window_enabled_v1" in header
    assert "hapb_rngv2_policy_window_enabled_v1()" in header
    assert "make_fixed_mlp_policy_window_v1(" in header


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device unavailable for HAPB RNGV2 Q3",
)


@cuda_required
def test_q3_cpu_gpu_shared_domain_parity_and_official_kats() -> None:
    module = load_cuda_extension()
    counters = torch.tensor(
        [list(row[1]) for row in PHILOX_KATS if row[0] == 10],
        dtype=torch.uint32,
        device="cuda",
    )
    keys = torch.tensor(
        [list(row[2]) for row in PHILOX_KATS if row[0] == 10],
        dtype=torch.uint32,
        device="cuda",
    )
    expected = torch.tensor(
        [list(row[3]) for row in PHILOX_KATS if row[0] == 10],
        dtype=torch.uint32,
        device="cuda",
    )
    got = philox_device(counters, keys, rounds=10, module=module)
    assert torch.equal(got, expected)

    oracle = HapbOracle()
    events = [
        HapbEvent(0, 0x0331, 3, 4, 1),
        HapbEvent(2, 0x0600, 10, 0, 0),
        HapbEvent(1, 0x0210, 8, 6, 0),
        HapbEvent(0, 0x0601, 11, 0, 2),
    ]
    cpu = [oracle.generate(event, n=7) for event in events]
    world = torch.tensor([e.world_uid for e in events], dtype=torch.uint32, device="cuda")
    domain = torch.tensor([e.domain_id for e in events], dtype=torch.uint32, device="cuda")
    epoch = torch.tensor(
        [SEMANTIC_EPOCH] * len(events), dtype=torch.uint32, device="cuda"
    )
    subject = torch.tensor(
        [e.subject_uid for e in events], dtype=torch.uint64, device="cuda"
    )
    tick = torch.tensor(
        [e.absolute_tick for e in events], dtype=torch.uint64, device="cuda"
    )
    operation = torch.tensor(
        [e.operation_id for e in events], dtype=torch.uint32, device="cuda"
    )
    n_values = torch.tensor([7] * len(events), dtype=torch.uint64, device="cuda")
    outputs = generate_device(
        world=world,
        domain=domain,
        epoch=epoch,
        subject=subject,
        tick=tick,
        operation=operation,
        n_values=n_values,
        namespace_key=NAMESPACE_DEFAULT,
        module=module,
    )
    w0, w1, w2, w3, bounded, accepted, _, _, z0, z1, status = outputs
    z0_host = z0.detach().cpu().numpy()
    z1_host = z1.detach().cpu().numpy()
    for index, result in enumerate(cpu):
        assert int(status[index]) == int(result.status)
        assert int(w0[index]) == result.words[0]
        assert int(w1[index]) == result.words[1]
        assert int(w2[index]) == result.words[2]
        assert int(w3[index]) == result.words[3]
        assert int(bounded[index]) == result.bounded
        assert int(accepted[index]) == result.accepted_ordinal
        assert int(z0_host[index]) == result.z0_bits
        assert int(z1_host[index]) == result.z1_bits


@cuda_required
def test_q3_policy_window_cpu_matches_semantic_event_and_no_tape() -> None:
    window = fill_policy_window_v2(
        world_uid=0, semantic_origin=3, episodes=1, width=4
    )
    assert window.status is Status.RNG_OK
    assert window.host_tape_bytes == 0
    oracle = HapbOracle()
    for column, bit in enumerate(window.crossover[0]):
        words = oracle.generate(HapbEvent(0, 0x0600, 3, 0, column), n=7).words
        from engine.tw_rngv2.uniforms import u32_to_f32_half_open

        assert bit is (u32_to_f32_half_open(words[0]) < 0.5)


def test_v2_turnover_diploid_uses_tape_free_stagers() -> None:
    source = (
        ROOT / "engine/perf/cuda_architecture_b_flattened_turnover_v1_kernels.cuh"
    ).read_text(encoding="utf-8")
    assert "cuda_diploid_scalar_genome_rngv2_v1.cuh" in source
    assert "genome_upgrade_v2_q10_rngv2_v1.cuh" in source
    assert "const bool rngv2_birth = execution.rng_semantics_version == 2;" in source
    assert "diploid::stage_child_v2(" in source
    assert "genome_v2::stage_child_v2(" in source
    assert "Leftover CPython choice-RNG draws are not a V1 Gaussian-tape cursor." in source
    post = source.split("counters[kScalarSemanticDrawsAdded] = semantic_draws_added;", 1)[1]
    assert "execution.rng_semantics_version == 2" in post.split(
        "cross_control.committed_counts[cell]", 1
    )[0]


def test_v2_birth_preserves_leftover_cpython_choice_rng_for_turnover() -> None:
    source = WDL_RUNTIME.read_text(encoding="utf-8")
    prepare = source.split("prepare_flattened_runtime_birth_transaction_v2", 1)[1]
    prepare = prepare.split("commit_flattened_runtime_birth_rng_v2", 1)[0]
    assert "if (rngv2_birth)" in prepare
    assert "python_post_words[word]" in prepare
    assert "scratch::kPythonRandomWordCount" in prepare
    commit = source.split("commit_flattened_runtime_birth_rng_v2", 1)[1]
    commit = commit.split("commit_flattened_runtime_birth_transaction_v2", 1)[0]
    assert "arguments.birth_config.rng_semantics_version != 2" in commit


def test_legacy_ensure_path_still_present_when_v2_env_unset() -> None:
    assert os.environ.get("EIEL_HAPB_RNG_SEMANTICS_VERSION") != "2"
    header = REFILL.read_text(encoding="utf-8")
    assert "hapb_rngv2_policy_window_enabled_v1()" in header
    assert header.count("make_fixed_mlp_policy_window_v1(") >= 2


def test_v2_policy_refill_does_not_address_physical_cell_as_world_uid() -> None:
    header = REFILL.read_text(encoding="utf-8")
    assert "static_cast<uint32_t>(cell)" not in header
    assert "Addressing by slot made force-vs-reuse diverge after turnover." in header


def test_q3_native_policy_window_matches_cpu_oracle() -> None:
    from hashlib import sha256
    from pathlib import Path

    import numpy as np
    from torch.utils.cpp_extension import load_inline

    from engine.tw_rngv2.uniforms import u32_to_f32_half_open

    digest = sha256(REFILL.read_bytes()).hexdigest()[:12]
    numpy_random_lib = Path(np.__file__).resolve().parent / "random/lib/libnpyrandom.a"
    module = load_inline(
        name=f"eiel_hapb_rngv2_policy_window_{digest}",
        cpp_sources=r"""
        #include <torch/extension.h>
        #include "cuda_architecture_b_resident_window_refill_v1.hpp"
        namespace refill = eiel::cuda_architecture_b::resident_window_refill_v1;
        torch::Tensor crossover_packed(
            int64_t episodes,
            int64_t width,
            uint32_t world_uid,
            uint64_t semantic_origin
        ) {
            auto empty = torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8));
            auto prepared = refill::make_fixed_mlp_policy_window_rngv2_v1(
                empty, episodes, width, 1, false, 0.0, 0.0, world_uid, semantic_origin
            );
            return prepared.crossover_mask_packed;
        }
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
            module.def("crossover_packed", &crossover_packed);
        }
        """,
        extra_include_paths=[str(ROOT / "engine/perf"), np.get_include()],
        extra_cflags=[
            "-O0",
            "-DEIEL_OPTION_D_M0_TELEMETRY_ENABLED=0",
            "-DEIEL_OPTION_D_M2_POLICY_SHADOW_ENABLED=0",
            "-DEIEL_OPTION_D_M4_DEVICE_POLICY_REFILL_ENABLED=0",
            "-DEIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED=0",
            "-DEIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED=0",
        ],
        extra_ldflags=[str(numpy_random_lib)],
        with_cuda=False,
        verbose=False,
    )
    packed = module.crossover_packed(1, 8, 2, 10)
    window = fill_policy_window_v2(world_uid=2, semantic_origin=10, episodes=1, width=8)
    assert window.status is Status.RNG_OK
    bits = packed[0].detach().cpu().numpy()
    for column, expected in enumerate(window.crossover[0]):
        native_bit = bool((int(bits[column >> 3]) >> (column & 7)) & 1)
        assert native_bit is expected
    oracle = HapbOracle()
    for column, expected in enumerate(window.crossover[0]):
        word = oracle.generate(HapbEvent(2, 0x0600, 10, 0, column), n=7).words[0]
        assert expected is (u32_to_f32_half_open(word) < 0.5)


def test_q3_ensure_v2_uses_frag_identity_and_semantic_bits(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_eiel_hapb_frag_a_policy_window_reuse_v1 import _call, _inputs

    from engine.tw_rngv2.uniforms import u32_to_f32_half_open

    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    monkeypatch.setenv("EIEL_HAPB_RNG_SEMANTICS_VERSION", "2")
    monkeypatch.delenv("EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL", raising=False)
    width = 17
    tensors = _inputs([0], capacity=8, width=width, bases=[0])
    first = _call(frag_a_ensure_module, tensors, required=8, width=width)
    assert first["physical_refill_performed"] is True
    assert first["frag_identity_result"] == "physical_refill_identity_published"
    expected = fill_policy_window_v2(
        world_uid=0,
        semantic_origin=int(tensors["counter"][0].item()),
        episodes=8,
        width=width,
    )
    assert expected.status is Status.RNG_OK
    packed = tensors["crossover"][0]
    for episode, row in enumerate(expected.crossover):
        bits = packed[episode].detach().cpu().numpy()
        for column, bit in enumerate(row):
            native_bit = bool((int(bits[column >> 3]) >> (column & 7)) & 1)
            assert native_bit is bit
            word = HapbOracle().generate(
                HapbEvent(0, 0x0600, 31 + episode, 0, column), n=7
            ).words[0]
            assert bit is (u32_to_f32_half_open(word) < 0.5)
    tape_before = tensors["crossover"].clone()
    second = _call(frag_a_ensure_module, tensors, required=2, width=width)
    assert second["resident_window_reused"] is True
    assert second["physical_refill_performed"] is False
    assert torch.equal(tensors["crossover"], tape_before)
