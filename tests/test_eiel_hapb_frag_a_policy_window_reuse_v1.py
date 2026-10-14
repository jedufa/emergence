from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "engine/perf/cuda_architecture_b_resident_window_refill_v1.hpp"


@pytest.fixture(scope="module")
def frag_a_ensure_module() -> Any:
    digest = sha256(HEADER.read_bytes()).hexdigest()[:12]
    source = r"""
    #include <torch/extension.h>
    #include "cuda_architecture_b_resident_window_refill_v1.hpp"

    namespace refill =
        eiel::cuda_architecture_b::resident_window_refill_v1;

    struct IdentityHandle {
        refill::PolicyWindowIdentityStateV1 state;
    };

    pybind11::dict ensure_policy_window(
        IdentityHandle& identity,
        torch::Tensor source_state,
        torch::Tensor delta,
        torch::Tensor mutation_mask,
        torch::Tensor crossover_mask_packed,
        torch::Tensor post_state_packed,
        torch::Tensor semantic_tape,
        torch::Tensor absolute_cursor,
        torch::Tensor semantic_counter,
        torch::Tensor window_base,
        int64_t policy_row_width,
        int64_t required_future_episodes
    ) {
        return refill::ensure_fixed_mlp_policy_window_with_identity_v1(
                source_state, delta, mutation_mask, crossover_mask_packed,
                post_state_packed, semantic_tape, absolute_cursor,
                semantic_counter, window_base, policy_row_width, false,
                "dense", 0.0, 0.0, required_future_episodes,
                refill::kFragIdentityProducerSchemaSha256V1,
                &identity.state
            );
    }

    void create_untrusted_policy_window_owner(
        IdentityHandle& identity,
        torch::Tensor source_state,
        torch::Tensor delta,
        torch::Tensor mutation_mask,
        torch::Tensor crossover_mask_packed,
        torch::Tensor post_state_packed,
        torch::Tensor semantic_tape,
        torch::Tensor absolute_cursor,
        torch::Tensor semantic_counter,
        torch::Tensor window_base,
        int64_t policy_row_width
    ) {
        auto capsule =
            refill::create_untrusted_policy_window_identity_owner_v1(
                source_state, delta, mutation_mask, crossover_mask_packed,
                post_state_packed, semantic_tape, absolute_cursor,
                semantic_counter, window_base, policy_row_width,
                "fresh_allocator_owner"
            );
        identity.state = refill::policy_window_identity_owner_from_capsule_v1(
            capsule
        )->state;
    }

    pybind11::dict owner_lease_contract(
        torch::Tensor source_state,
        torch::Tensor delta,
        torch::Tensor mutation_mask,
        torch::Tensor crossover_mask_packed,
        torch::Tensor post_state_packed,
        torch::Tensor semantic_tape,
        torch::Tensor absolute_cursor,
        torch::Tensor semantic_counter,
        torch::Tensor window_base,
        int64_t policy_row_width
    ) {
        auto first = refill::create_untrusted_policy_window_identity_owner_v1(
            source_state, delta, mutation_mask, crossover_mask_packed,
            post_state_packed, semantic_tape, absolute_cursor,
            semantic_counter, window_base, policy_row_width,
            "fresh_allocator_owner"
        );
        auto second = refill::create_untrusted_policy_window_identity_owner_v1(
            source_state, delta, mutation_mask, crossover_mask_packed,
            post_state_packed, semantic_tape, absolute_cursor,
            semantic_counter, window_base, policy_row_width,
            "fresh_allocator_owner"
        );
        auto* first_owner =
            refill::policy_window_identity_owner_from_capsule_v1(first);
        auto* second_owner =
            refill::policy_window_identity_owner_from_capsule_v1(second);
        bool duplicate_rejected = false;
        int64_t stale_token = 0;
        {
            refill::PolicyWindowIdentityOwnerExecutionLeaseV1 first_lease(
                first_owner
            );
            try {
                refill::PolicyWindowIdentityOwnerExecutionLeaseV1 second_lease(
                    second_owner
                );
            } catch (const c10::Error&) {
                duplicate_rejected = true;
            }
            first_owner->state.window_valid = {1U};
            stale_token = first_lease.
                mark_native_returned_pending_python_validation();
        }
        bool pending_invalidated = false;
        std::string boundary;
        int64_t successor_token = 0;
        {
            refill::PolicyWindowIdentityOwnerExecutionLeaseV1 retry(first_owner);
            pending_invalidated =
                first_owner->state.window_valid == std::vector<uint8_t>{0U};
            boundary = retry.boundary();
            successor_token = retry.
                mark_native_returned_pending_python_validation();
        }
        bool stale_commit_rejected = false;
        try {
            refill::commit_policy_window_identity_owner_entry_v1(
                first, stale_token
            );
        } catch (const c10::Error&) {
            stale_commit_rejected = true;
        }
        const bool successor_pending_after_stale_commit =
            first_owner->pending_validation_token == successor_token;
        refill::commit_policy_window_identity_owner_entry_v1(
            first, successor_token
        );
        first_owner->state.window_valid = {1U};
        {
            refill::PolicyWindowIdentityOwnerExecutionLeaseV1 failed_native(
                first_owner
            );
        }
        bool native_exception_invalidated = false;
        std::string native_exception_boundary;
        int64_t recovered_token = 0;
        {
            refill::PolicyWindowIdentityOwnerExecutionLeaseV1 recovered(
                first_owner
            );
            native_exception_invalidated =
                first_owner->state.window_valid == std::vector<uint8_t>{0U};
            native_exception_boundary = recovered.boundary();
            recovered_token = recovered.
                mark_native_returned_pending_python_validation();
        }
        refill::commit_policy_window_identity_owner_entry_v1(
            first, recovered_token
        );
        pybind11::dict out;
        out["duplicate_rejected"] = duplicate_rejected;
        out["pending_invalidated"] = pending_invalidated;
        out["retry_boundary"] = boundary;
        out["stale_commit_rejected"] = stale_commit_rejected;
        out["successor_pending_after_stale_commit"] =
            successor_pending_after_stale_commit;
        out["successor_committed"] =
            first_owner->pending_validation_token == 0;
        out["native_exception_invalidated"] = native_exception_invalidated;
        out["native_exception_boundary"] = native_exception_boundary;
        return out;
    }

    void consume_policy_rows(
        torch::Tensor source_state,
        torch::Tensor absolute_cursor,
        torch::Tensor semantic_counter,
        torch::Tensor counts,
        int64_t policy_row_width
    ) {
        TORCH_CHECK(
            !source_state.is_cuda() && source_state.scalar_type() == torch::kUInt8
                && source_state.dim() == 2 && source_state.is_contiguous(),
            "owned-consume source must be contiguous CPU uint8 rank-2"
        );
        TORCH_CHECK(
            !absolute_cursor.is_cuda() && absolute_cursor.scalar_type() == torch::kInt64
                && absolute_cursor.dim() == 1 && absolute_cursor.is_contiguous(),
            "owned-consume cursor must be contiguous CPU int64"
        );
        TORCH_CHECK(
            !semantic_counter.is_cuda() && semantic_counter.scalar_type() == torch::kInt64
                && semantic_counter.sizes() == absolute_cursor.sizes()
                && semantic_counter.is_contiguous(),
            "owned-consume semantic counter must match cursor"
        );
        counts = counts.to(torch::kCPU).to(torch::kInt64).contiguous();
        TORCH_CHECK(
            counts.sizes() == absolute_cursor.sizes() && policy_row_width > 0,
            "owned-consume counts/width mismatch"
        );
        auto* cursor = absolute_cursor.data_ptr<int64_t>();
        auto* semantic = semantic_counter.data_ptr<int64_t>();
        const auto* count = counts.data_ptr<int64_t>();
        const int64_t state_bytes = source_state.size(1);
        auto* source = source_state.data_ptr<uint8_t>();
        for (int64_t cell = 0; cell < source_state.size(0); ++cell) {
            TORCH_CHECK(count[cell] >= 0, "owned-consume count must be nonnegative");
            at::Generator generator = at::detail::createCPUGenerator(0);
            auto state_row = source_state[cell].clone().contiguous();
            generator.set_state(state_row);
            if (count[cell] > 0) {
                static_cast<void>(at::rand(
                    {count[cell], policy_row_width},
                    generator,
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
                ));
            }
            auto final_state = generator.get_state().contiguous();
            std::memcpy(
                source + cell * state_bytes,
                final_state.data_ptr<uint8_t>(),
                static_cast<std::size_t>(state_bytes)
            );
            cursor[cell] += count[cell];
            semantic[cell] += count[cell] * policy_row_width;
        }
    }

    pybind11::dict identity_snapshot(const IdentityHandle& identity) {
        const auto& state = identity.state;
        pybind11::dict out;
        out["initialized"] = state.initialized;
        out["tape_generation"] = state.tape_storage_generation;
        out["rng_lineage_epoch"] = state.rng_lineage_epoch;
        out["valid"] = state.window_valid;
        out["window_epoch"] = state.window_epoch;
        out["window_rng_lineage_epoch"] = state.window_rng_lineage_epoch;
        out["window_base"] = state.window_base;
        out["window_base_semantic_draws"] =
            state.window_base_semantic_draws;
        out["publication_origin"] = static_cast<int>(state.publication_origin);
        return out;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
        pybind11::class_<IdentityHandle>(module, "IdentityHandle")
            .def(pybind11::init<>())
            .def("invalidate", [](IdentityHandle& identity) {
                refill::invalidate_policy_window_identity_v1(
                    &identity.state, false, false
                );
            })
            .def("rotate_rng_lineage", [](IdentityHandle& identity) {
                refill::invalidate_policy_window_identity_v1(
                    &identity.state, true, false
                );
            })
            .def("rotate_tape_generation", [](IdentityHandle& identity) {
                refill::invalidate_policy_window_identity_v1(
                    &identity.state, false, true
                );
            })
            .def("drift_rng_lineage", [](IdentityHandle& identity) {
                identity.state.rng_lineage_epoch += 1;
            })
            .def("drift_tape_generation", [](IdentityHandle& identity) {
                identity.state.tape_storage_generation += 1;
            })
            .def("drift_configuration", [](IdentityHandle& identity) {
                identity.state.configuration.producer_schema_sha256 =
                    std::string(64, '0');
            })
            .def("snapshot", &identity_snapshot);
        module.def("ensure_policy_window", &ensure_policy_window);
        module.def(
            "create_untrusted_policy_window_owner",
            &create_untrusted_policy_window_owner
        );
        module.def("owner_lease_contract", &owner_lease_contract);
        module.def("consume_policy_rows", &consume_policy_rows);
    }
    """
    numpy_random_lib = (
        Path(np.__file__).resolve().parent / "random/lib/libnpyrandom.a"
    )
    return load_inline(
        name=f"eiel_hapb_frag_a_policy_reuse_{digest}",
        cpp_sources=[source],
        extra_include_paths=[str(ROOT / "engine/perf"), np.get_include()],
        extra_cflags=[
            "-O0",
            "-DEIEL_OPTION_D_M0_TELEMETRY_ENABLED=1",
            "-DEIEL_OPTION_D_M2_POLICY_SHADOW_ENABLED=0",
            "-DEIEL_OPTION_D_M4_DEVICE_POLICY_REFILL_ENABLED=0",
            "-DEIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED=0",
            "-DEIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED=0",
        ],
        extra_ldflags=[str(numpy_random_lib)],
        with_cuda=False,
        verbose=False,
    )


def _inputs(
    cursors: list[int],
    *,
    capacity: int = 8,
    width: int = 17,
    bases: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    cells = len(cursors)
    states = []
    for cell in range(cells):
        generator = torch.Generator(device="cpu").manual_seed(1701 + cell)
        states.append(generator.get_state())
    return {
        "source": torch.stack(states).contiguous(),
        "delta": torch.zeros((cells, capacity, 1), dtype=torch.float32),
        "mask": torch.zeros((cells, capacity, 1), dtype=torch.bool),
        "crossover": torch.zeros(
            (cells, capacity, (width + 7) // 8), dtype=torch.uint8
        ),
        "post": torch.empty((cells, capacity, 0), dtype=torch.uint8),
        "semantic": torch.full((cells, capacity), -1, dtype=torch.int64),
        "cursor": torch.tensor(cursors, dtype=torch.int64),
        "counter": torch.tensor([31 + i for i in range(cells)], dtype=torch.int64),
        "base": torch.tensor(bases if bases is not None else [0] * cells, dtype=torch.int64),
    }


def _call(
    module: Any,
    tensors: dict[str, torch.Tensor],
    required: int,
    width: int = 17,
) -> dict[str, Any]:
    identity = tensors.get("_identity")
    if identity is None:
        identity = module.IdentityHandle()
        tensors["_identity"] = identity
    return dict(
        module.ensure_policy_window(
            identity,
            tensors["source"],
            tensors["delta"],
            tensors["mask"],
            tensors["crossover"],
            tensors["post"],
            tensors["semantic"],
            tensors["cursor"],
            tensors["counter"],
            tensors["base"],
            width,
            required,
        )
    )


def _consume(
    module: Any,
    tensors: dict[str, Any],
    counts: list[int],
    *,
    width: int = 17,
) -> None:
    module.consume_policy_rows(
        tensors["source"],
        tensors["cursor"],
        tensors["counter"],
        torch.tensor(counts, dtype=torch.int64),
        width,
    )


def _snapshot(tensors: dict[str, Any]) -> dict[str, Any]:
    return dict(tensors["_identity"].snapshot())


def _assert_suffix_matches_current_rng(
    module: Any,
    tensors: dict[str, Any],
    *,
    required: int,
    width: int = 17,
) -> None:
    capacity = int(tensors["delta"].shape[1])
    cursors = [int(value) for value in tensors["cursor"].tolist()]
    bases = [int(value) for value in tensors["base"].tolist()]
    oracle = _inputs(cursors, capacity=capacity, width=width, bases=cursors)
    oracle["source"].copy_(tensors["source"])
    oracle["counter"].copy_(tensors["counter"])
    receipt = _call(module, oracle, required=required, width=width)
    assert receipt["physical_refill_performed"] is True
    for cell, (cursor, base) in enumerate(zip(cursors, bases, strict=True)):
        physical = cursor - base
        for name in ("delta", "mask", "crossover", "post", "semantic"):
            assert torch.equal(
                tensors[name][cell, physical : physical + required],
                oracle[name][cell, :required],
            ), (cell, name, physical, required)


def test_frag_identity_construction_then_first_hydration(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3 cases 1-2: untrusted construction is invalid until real refill."""

    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    tensors = _inputs([0, 4], capacity=8, bases=[0, 4])
    tensors["_identity"] = frag_a_ensure_module.IdentityHandle()
    assert _snapshot(tensors) == {
        "initialized": False,
        "tape_generation": 0,
        "rng_lineage_epoch": 0,
        "valid": [],
        "window_epoch": [],
        "window_rng_lineage_epoch": [],
        "window_base": [],
        "window_base_semantic_draws": [],
        "publication_origin": 0,
    }
    source_before = tensors["source"].clone()
    cursor_before = tensors["cursor"].clone()
    counter_before = tensors["counter"].clone()

    receipt = _call(frag_a_ensure_module, tensors, required=2)

    assert receipt["physical_refill_performed"] is True
    assert receipt["frag_identity_result"] == "physical_refill_identity_published"
    assert torch.equal(tensors["source"], source_before)
    assert torch.equal(tensors["cursor"], cursor_before)
    assert torch.equal(tensors["counter"], counter_before)
    assert torch.equal(tensors["base"], cursor_before)
    assert _snapshot(tensors) == {
        "initialized": True,
        "tape_generation": 1,
        "rng_lineage_epoch": 1,
        "valid": [1, 1],
        "window_epoch": [1, 1],
        "window_rng_lineage_epoch": [1, 1],
        "window_base": [0, 4],
        "window_base_semantic_draws": [31, 32],
        "publication_origin": 2,
    }


def test_frag_identity_untrusted_owner_forces_first_hydration(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3 case 2: post-hoc allocator bytes never authorize first reuse."""

    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    monkeypatch.setenv("EIEL_HAPB_FRAG_IDENTITY_TRACE", "1")
    tensors = _inputs([0], capacity=8, bases=[0])
    assert _call(frag_a_ensure_module, tensors, required=8)[
        "physical_refill_performed"
    ] is True
    tape_before = {
        name: tensors[name].clone()
        for name in ("source", "delta", "mask", "crossover", "post", "semantic")
    }
    tensors["_identity"] = frag_a_ensure_module.IdentityHandle()
    tensors["crossover"].bitwise_not_()
    frag_a_ensure_module.create_untrusted_policy_window_owner(
        tensors["_identity"],
        tensors["source"],
        tensors["delta"],
        tensors["mask"],
        tensors["crossover"],
        tensors["post"],
        tensors["semantic"],
        tensors["cursor"],
        tensors["counter"],
        tensors["base"],
        17,
    )
    assert _snapshot(tensors)["publication_origin"] == 0

    receipt = _call(frag_a_ensure_module, tensors, required=8)

    assert receipt["resident_window_reused"] is False
    assert receipt["physical_refill_performed"] is True
    assert receipt["frag_identity_trace"][0]["reason"] == (
        "identity_state_uninitialized"
    )
    assert receipt["frag_identity_trace"][0]["identity_publication_origin"] == (
        "uninitialized"
    )
    assert receipt["frag_identity_trace"][-1]["identity_publication_origin"] == (
        "physical_refill_published"
    )
    for name, value in tape_before.items():
        assert torch.equal(tensors[name], value), name


def test_frag_identity_owner_lease_rejects_duplicates_and_unvalidated_reentry(
    frag_a_ensure_module: Any,
) -> None:
    tensors = _inputs([0], capacity=8, bases=[0])
    receipt = frag_a_ensure_module.owner_lease_contract(
        tensors["source"],
        tensors["delta"],
        tensors["mask"],
        tensors["crossover"],
        tensors["post"],
        tensors["semantic"],
        tensors["cursor"],
        tensors["counter"],
        tensors["base"],
        17,
    )

    assert receipt == {
        "duplicate_rejected": True,
        "pending_invalidated": True,
        "retry_boundary": "unvalidated_prior_entry_reentry",
        "stale_commit_rejected": True,
        "successor_pending_after_stale_commit": True,
        "successor_committed": True,
        "native_exception_invalidated": True,
        "native_exception_boundary": "native_exception_reentry",
    }


def test_frag_identity_owner_rejects_live_storage_outside_its_lease(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    tensors = _inputs([0], capacity=8, bases=[0])
    tensors["_identity"] = frag_a_ensure_module.IdentityHandle()
    frag_a_ensure_module.create_untrusted_policy_window_owner(
        tensors["_identity"],
        tensors["source"],
        tensors["delta"],
        tensors["mask"],
        tensors["crossover"],
        tensors["post"],
        tensors["semantic"],
        tensors["cursor"],
        tensors["counter"],
        tensors["base"],
        17,
    )
    tensors["crossover"] = tensors["crossover"].clone()

    with pytest.raises(
        RuntimeError,
        match="live crossover storage differs from its execution lease",
    ):
        _call(frag_a_ensure_module, tensors, required=8)


@pytest.mark.parametrize(
    ("case", "consumed", "required", "expect_reuse"),
    [
        pytest.param("full_window", 0, 8, True, id="03-full-window"),
        pytest.param("partial_suffix", 2, 3, True, id="04-partial-suffix"),
        pytest.param("exact_exhaustion", 5, 3, True, id="05-exact-exhaustion"),
        pytest.param("one_past", 8, 1, False, id="06-one-past-refill"),
    ],
)
def test_frag_identity_physical_suffix_boundaries(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    consumed: int,
    required: int,
    expect_reuse: bool,
) -> None:
    """R3 cases 3-6: exact physical/logical suffix boundaries."""

    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    tensors = _inputs([0], capacity=8, bases=[0])
    assert _call(frag_a_ensure_module, tensors, required=8)[
        "physical_refill_performed"
    ] is True
    _consume(frag_a_ensure_module, tensors, [consumed])
    generation_before = _snapshot(tensors)["tape_generation"]
    source_before = tensors["source"].clone()
    cursor_before = tensors["cursor"].clone()
    counter_before = tensors["counter"].clone()
    if expect_reuse:
        _assert_suffix_matches_current_rng(
            frag_a_ensure_module, tensors, required=required
        )

    receipt = _call(frag_a_ensure_module, tensors, required=required)

    assert receipt["resident_window_reused"] is expect_reuse, case
    assert receipt["physical_refill_performed"] is (not expect_reuse), case
    assert torch.equal(tensors["source"], source_before)
    assert torch.equal(tensors["cursor"], cursor_before)
    assert torch.equal(tensors["counter"], counter_before)
    if expect_reuse:
        assert int(tensors["base"].item()) == 0
        assert _snapshot(tensors)["tape_generation"] == generation_before
        if case == "exact_exhaustion":
            _consume(frag_a_ensure_module, tensors, [required])
            one_past = _call(frag_a_ensure_module, tensors, required=1)
            assert one_past["physical_refill_performed"] is True
            assert torch.equal(tensors["base"], tensors["cursor"])
    else:
        assert torch.equal(tensors["base"], tensors["cursor"])
        assert _snapshot(tensors)["tape_generation"] == generation_before + 1


@pytest.mark.parametrize(
    ("case", "births", "expected_decisions"),
    [
        pytest.param("zero_birth", [0], ["reuse"], id="07-zero-birth"),
        pytest.param("nonzero_birth", [2], ["reuse"], id="08-nonzero-birth"),
        pytest.param(
            "consecutive_ticks",
            [1, 2, 2],
            ["reuse", "reuse", "physical"],
            id="09-consecutive-ticks",
        ),
    ],
)
def test_frag_identity_owned_consume_schedule(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    births: list[int],
    expected_decisions: list[str],
) -> None:
    """R3 cases 7-9: owned raw consume preserves lineage and RNG schedule."""

    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    tensors = _inputs([0], capacity=6, bases=[0])
    _call(frag_a_ensure_module, tensors, required=6)
    initial = {
        name: tensors[name].clone()
        for name in ("source", "cursor", "counter")
    }
    decisions: list[str] = []
    for birth_count in births:
        _consume(frag_a_ensure_module, tensors, [birth_count])
        source_before = tensors["source"].clone()
        cursor_before = tensors["cursor"].clone()
        counter_before = tensors["counter"].clone()
        if int(tensors["cursor"].item() - tensors["base"].item()) <= 4:
            _assert_suffix_matches_current_rng(
                frag_a_ensure_module, tensors, required=2
            )
        receipt = _call(frag_a_ensure_module, tensors, required=2)
        decisions.append(
            "physical" if receipt["physical_refill_performed"] else "reuse"
        )
        assert torch.equal(tensors["source"], source_before)
        assert torch.equal(tensors["cursor"], cursor_before)
        assert torch.equal(tensors["counter"], counter_before)
    assert decisions == expected_decisions, case
    if case == "zero_birth":
        for name, value in initial.items():
            assert torch.equal(tensors[name], value), name


def test_frag_identity_cursor_advance_and_reset_boundary(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3 case 15: owned advance is legal; unowned reset fails closed."""

    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    tensors = _inputs([0], capacity=8, bases=[0])
    _call(frag_a_ensure_module, tensors, required=8)
    _consume(frag_a_ensure_module, tensors, [2])
    assert _call(frag_a_ensure_module, tensors, required=2)[
        "resident_window_reused"
    ] is True
    # Same numeric value, but an external ATen write rotates the tensor version.
    tensors["cursor"].copy_(tensors["cursor"].clone())
    receipt = _call(frag_a_ensure_module, tensors, required=2)
    assert receipt["physical_refill_performed"] is True
    tensors["cursor"].copy_(torch.tensor([1], dtype=torch.int64))
    with pytest.raises(RuntimeError, match="outside the resident physical tape"):
        _call(frag_a_ensure_module, tensors, required=1)


@pytest.mark.parametrize(
    ("case", "drift", "reason"),
    [
        pytest.param(
            "rng_lineage",
            "drift_rng_lineage",
            "rng_lineage_mismatch",
            id="19-rng-identity-mismatch",
        ),
        pytest.param(
            "tape_generation",
            "drift_tape_generation",
            "tape_generation_mismatch",
            id="20-tape-identity-mismatch",
        ),
    ],
)
def test_frag_identity_epoch_mismatch_refills(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    drift: str,
    reason: str,
) -> None:
    """R3 cases 19-20: explicit RNG/tape identity mismatches fail closed."""

    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    monkeypatch.setenv("EIEL_HAPB_FRAG_IDENTITY_TRACE", "1")
    tensors = _inputs([0], capacity=8, bases=[0])
    _call(frag_a_ensure_module, tensors, required=8)
    getattr(tensors["_identity"], drift)()
    receipt = _call(frag_a_ensure_module, tensors, required=2)
    assert receipt["physical_refill_performed"] is True, case
    assert receipt["frag_identity_trace"][0]["reason"] == reason


@pytest.mark.parametrize(
    ("case", "mutate", "reason"),
    [
        pytest.param(
            "source_same_storage_write",
            lambda tensors: tensors["source"].copy_(tensors["source"].clone()),
            "tensor_storage_identity_mismatch",
            id="19-source-version-mismatch",
        ),
        pytest.param(
            "tape_same_storage_write",
            lambda tensors: tensors["crossover"].copy_(
                tensors["crossover"].clone()
            ),
            "tensor_storage_identity_mismatch",
            id="20-tape-version-mismatch",
        ),
        pytest.param(
            "producer_configuration",
            lambda tensors: tensors["_identity"].drift_configuration(),
            "configuration_identity_mismatch",
            id="20-configuration-mismatch",
        ),
    ],
)
def test_frag_identity_same_storage_and_configuration_writes_refill(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    mutate: Any,
    reason: str,
) -> None:
    """Direct tape dependencies reject version/config drift, not just reallocations."""

    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    monkeypatch.setenv("EIEL_HAPB_FRAG_IDENTITY_TRACE", "1")
    tensors = _inputs([0], capacity=8, bases=[0])
    _call(frag_a_ensure_module, tensors, required=8)
    mutate(tensors)

    receipt = _call(frag_a_ensure_module, tensors, required=2)

    assert receipt["physical_refill_performed"] is True, case
    assert receipt["frag_identity_trace"][0]["reason"] == reason


def test_frag_identity_stale_lifecycle_and_invalid_state_refill(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3 cases 21-22: stale-owner and explicit invalidity cannot reuse."""

    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    monkeypatch.setenv("EIEL_HAPB_FRAG_IDENTITY_TRACE", "1")
    old = _inputs([0], capacity=8, bases=[0])
    _call(frag_a_ensure_module, old, required=8)
    new = _inputs([0], capacity=8, bases=[0])
    new["_identity"] = old["_identity"]
    stale = _call(frag_a_ensure_module, new, required=2)
    assert stale["physical_refill_performed"] is True
    assert stale["frag_identity_trace"][0]["reason"] == (
        "tensor_storage_identity_mismatch"
    )
    new["_identity"].invalidate()
    invalid = _call(frag_a_ensure_module, new, required=2)
    assert invalid["physical_refill_performed"] is True
    assert invalid["frag_identity_trace"][0]["reason"] == "window_invalid"


def test_frag_identity_deterministic_repeated_execution(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3 case 23: decisions, bytes, RNG, counters, and epochs repeat exactly."""

    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    monkeypatch.delenv("EIEL_HAPB_FRAG_IDENTITY_TRACE", raising=False)

    def run() -> dict[str, Any]:
        tensors = _inputs([0, 0], capacity=5, bases=[0, 0])
        receipts: list[tuple[bool, int, tuple[int, ...], tuple[int, ...]]] = []
        first = _call(frag_a_ensure_module, tensors, required=5)
        receipts.append(
            (
                bool(first["physical_refill_performed"]),
                int(first["frag_identity_tape_generation"]),
                tuple(tensors["cursor"].tolist()),
                tuple(tensors["base"].tolist()),
            )
        )
        for counts in ([0, 1], [2, 1], [2, 3], [1, 0]):
            _consume(frag_a_ensure_module, tensors, list(counts))
            receipt = _call(frag_a_ensure_module, tensors, required=2)
            receipts.append(
                (
                    bool(receipt["physical_refill_performed"]),
                    int(receipt["frag_identity_tape_generation"]),
                    tuple(tensors["cursor"].tolist()),
                    tuple(tensors["base"].tolist()),
                )
            )
        digests = {
            name: sha256(tensors[name].contiguous().numpy().tobytes()).hexdigest()
            for name in (
                "source",
                "delta",
                "mask",
                "crossover",
                "post",
                "semantic",
                "cursor",
                "counter",
                "base",
            )
        }
        return {
            "receipts": receipts,
            "digests": digests,
            "identity": _snapshot(tensors),
        }

    assert run() == run()


def test_frag_a_production_default_is_physical_refill(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", raising=False)
    monkeypatch.delenv("EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL", raising=False)
    tensors = _inputs([0, 0], capacity=8)
    first = _call(frag_a_ensure_module, tensors, required=8)
    assert first["physical_refill_performed"] is True
    tensors["cursor"].copy_(torch.tensor([2, 1], dtype=torch.int64))
    second = _call(frag_a_ensure_module, tensors, required=3)
    assert second["physical_refill_performed"] is True
    assert second["resident_window_reused"] is False


def test_frag_a_reuses_remaining_suffix_without_mutating_tape(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    tensors = _inputs([2, 1, 0], capacity=8)
    seeded = _call(frag_a_ensure_module, tensors, required=8)
    assert seeded["physical_refill_performed"] is True
    before = {
        name: value.clone()
        for name, value in tensors.items()
        if isinstance(value, torch.Tensor)
    }
    receipt = _call(frag_a_ensure_module, tensors, required=3)
    assert receipt["physical_refill_performed"] is False
    assert receipt["resident_window_reused"] is True
    assert receipt["hapb_frag_a_policy_window_reuse_used"] is True
    assert receipt["physical_rows_generated"] == 0
    assert receipt["host_to_device_tape_bytes"] == 0
    assert receipt["minimum_remaining_episodes"] == 8
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            continue
        assert torch.equal(value, before[name]), name


def test_frag_a_unproved_zero_base_forces_first_hydration(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arithmetic equality alone cannot authorize a public-entry window."""

    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    tensors = _inputs([0, 0], capacity=8, bases=[0, 0])
    before = tensors["crossover"].clone()
    receipt = _call(frag_a_ensure_module, tensors, required=2)
    assert receipt["physical_refill_performed"] is True
    assert receipt["resident_window_reused"] is False
    assert receipt["frag_identity_result"] == "physical_refill_identity_published"
    assert not torch.equal(tensors["crossover"], before)
    assert torch.equal(tensors["base"], torch.tensor([0, 0], dtype=torch.int64))


def test_frag_identity_trace_observes_verified_reuse_without_mutation(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    monkeypatch.setenv("EIEL_HAPB_FRAG_IDENTITY_TRACE", "1")
    tensors = _inputs([2, 1], capacity=8, bases=[0, 0])
    hydration = _call(frag_a_ensure_module, tensors, required=3)
    assert hydration["physical_refill_performed"] is True
    before = {
        name: value.clone()
        for name, value in tensors.items()
        if isinstance(value, torch.Tensor)
    }

    receipt = _call(frag_a_ensure_module, tensors, required=3)

    assert receipt["resident_window_reused"] is True
    rows = list(receipt["frag_identity_trace"])
    assert [row["cell"] for row in rows] == [0, 1]
    assert [row["decision"] for row in rows] == ["reuse", "reuse"]
    assert [row["reason"] for row in rows] == [
        "identity_and_suffix_verified",
        "identity_and_suffix_verified",
    ]
    assert all(
        row["rng_identity"] == "lineage_and_schedule_verified"
        for row in rows
    )
    assert all(row["source_rng_state_fnv1a64"] is None for row in rows)
    assert all(row["source_rng_content_read_performed"] is False for row in rows)
    assert {
        row["source_rng_identity_basis"] for row in rows
    } == {"closed_writer_lineage_and_semantic_schedule"}
    assert receipt["device_to_host_operations_lower_bound"] == 3
    assert [row["semantic_counter"] for row in rows] == [31, 32]
    assert all(
        row["tape_identity"]
        == "storage_generation_and_configuration_verified"
        for row in rows
    )
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            continue
        assert torch.equal(value, before[name]), name


def test_frag_identity_trace_is_absent_by_default(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    monkeypatch.delenv("EIEL_HAPB_FRAG_IDENTITY_TRACE", raising=False)
    tensors = _inputs([0], capacity=8, bases=[0])
    receipt = _call(frag_a_ensure_module, tensors, required=2)
    assert "frag_identity_trace" not in receipt


def test_frag_a_negative_physical_base_is_rejected(
    frag_a_ensure_module: Any,
) -> None:
    tensors = _inputs([0, 0, 0], capacity=8, bases=[-1, -1, -1])
    with pytest.raises(RuntimeError, match="nonnegative physical origin"):
        _call(frag_a_ensure_module, tensors, required=3)


def test_frag_identity_negative_semantic_state_fails_closed(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    tensors = _inputs([0], capacity=8, bases=[0])
    _call(frag_a_ensure_module, tensors, required=8)
    tensors["counter"].copy_(torch.tensor([-1], dtype=torch.int64))
    with pytest.raises(RuntimeError, match="semantic_draws must be nonnegative"):
        _call(frag_a_ensure_module, tensors, required=2)


def test_frag_a_force_physical_refill_matches_old_every_tick_path(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    monkeypatch.setenv("EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL", "1")
    tensors = _inputs([0, 0], capacity=8)
    first = _call(frag_a_ensure_module, tensors, required=8)
    assert first["physical_refill_performed"] is True
    monkeypatch.delenv("EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL", raising=False)
    tape_after_first = tensors["crossover"].clone()
    _consume(frag_a_ensure_module, tensors, [2, 1])
    reuse = _call(frag_a_ensure_module, tensors, required=3)
    assert reuse["resident_window_reused"] is True
    assert torch.equal(tensors["crossover"], tape_after_first)
    monkeypatch.setenv("EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL", "1")
    forced = _inputs([2, 1], capacity=8, bases=[2, 1])
    forced["source"].copy_(tensors["source"])
    forced["counter"].copy_(tensors["counter"])
    forced_receipt = _call(frag_a_ensure_module, forced, required=3)
    assert forced_receipt["physical_refill_performed"] is True
    assert forced_receipt["resident_window_reused"] is False
    monkeypatch.delenv("EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL", raising=False)
    assert torch.equal(tensors["cursor"], torch.tensor([2, 1], dtype=torch.int64))
    assert torch.equal(tensors["source"], forced["source"])
    assert torch.equal(tensors["crossover"][0, 2:5], forced["crossover"][0, :3])
    assert torch.equal(tensors["crossover"][1, 1:4], forced["crossover"][1, :3])


def test_frag_a_walks_then_refills_when_coverage_is_exhausted(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    monkeypatch.setenv("EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL", "1")
    tensors = _inputs([0, 0], capacity=8)
    _call(frag_a_ensure_module, tensors, required=8)
    monkeypatch.delenv("EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL", raising=False)
    initial_base = tensors["base"].clone()
    physical = 0
    reused = 0
    for counts in ([0, 0], [2, 1], [3, 3]):
        _consume(frag_a_ensure_module, tensors, list(counts))
        before_source = tensors["source"].clone()
        before_counter = tensors["counter"].clone()
        receipt = _call(frag_a_ensure_module, tensors, required=3)
        if receipt["resident_window_reused"]:
            reused += 1
        else:
            physical += 1
        assert torch.equal(tensors["source"], before_source)
        assert torch.equal(tensors["counter"], before_counter)
        assert torch.equal(tensors["base"], initial_base)
    assert reused == 3
    assert physical == 0
    _consume(frag_a_ensure_module, tensors, [1, 1])
    before_source = tensors["source"].clone()
    receipt = _call(frag_a_ensure_module, tensors, required=3)
    assert receipt["physical_refill_performed"] is True
    assert int(receipt["refill_trigger_cell_count"]) >= 1
    assert torch.equal(tensors["source"], before_source)
    assert torch.equal(tensors["base"], tensors["cursor"])


def test_frag_a_repeated_window_demand_reduces_physical_refills(
    frag_a_ensure_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", "1")
    cells = 48
    capacity = 16
    required = 2
    ticks = 20
    os.environ["EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL"] = "1"
    tensors = _inputs([0] * cells, capacity=capacity)
    _call(frag_a_ensure_module, tensors, required=capacity)
    os.environ.pop("EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL", None)
    physical = 0
    reused = 0
    for _tick in range(ticks):
        receipt = _call(frag_a_ensure_module, tensors, required=required)
        if receipt["physical_refill_performed"]:
            physical += 1
        else:
            reused += 1
        _consume(frag_a_ensure_module, tensors, [required] * cells)
    assert reused >= 10
    assert physical * 5 <= ticks
    assert reused / max(physical, 1) >= 5.0
    os.environ.pop("EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL", None)
