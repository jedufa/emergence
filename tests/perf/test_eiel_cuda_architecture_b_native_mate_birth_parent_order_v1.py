from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


_HOST_PROBE = r"""
#include <cmath>

// CUDA exposes ::isfinite while strict C++17 hosts expose std::isfinite.
// Supply the CUDA spelling before including the device header so this probe
// tests the transaction contract rather than host-library namespace details.
inline bool isfinite(const double value) {
    return std::isfinite(value);
}

#include "engine/perf/cuda_architecture_b_flattened_birth_identity_v2.cuh"

#include <array>
#include <cstdint>

namespace identity =
    eiel::cuda_architecture_b::persistent_birth_identity;
namespace preflight =
    eiel::cuda_architecture_b::flattened_birth_identity_v2;

using ParentOrder = identity::BirthParentOrderModeV1;

int run_preflight(
    const ParentOrder order,
    const std::array<int64_t, 2>& planned_parent_slots
) {
    constexpr int64_t kSlots = 4;
    constexpr int64_t kBirths = 2;

    // Neutral selection owns this row order. It is deliberately not ordered
    // by agent ID: slot 0 has ID 20 and slot 1 has ID 10.
    std::array<int64_t, kSlots> slot_to_agent_id{20, 10, -1, -1};
    std::array<bool, kSlots> alive{true, true, false, false};
    std::array<int32_t, kSlots> grid{0, 1, -1, -1};
    std::array<float, kSlots> energy{20.0F, 20.0F, 0.0F, 0.0F};
    std::array<float, kSlots> health{10.0F, 10.0F, 0.0F, 0.0F};
    std::array<int32_t, kSlots> cooldown{};
    std::array<float, kSlots> threshold{1.0F, 1.0F, 0.0F, 0.0F};
    std::array<double, kSlots> observed_energy{20.0, 20.0, 0.0, 0.0};
    std::array<int64_t, kBirths> free_queue{2, 3};
    int64_t free_count = kBirths;
    int64_t next_id = 21;

    // Both target cells are empty and distinct.
    std::array<int32_t, kBirths * 2> spawn_positions{1, 0, 1, 1};
    int64_t planned_count = kBirths;

    std::array<int64_t, kBirths> staged_parents{};
    std::array<int64_t, kBirths> staged_children{};
    std::array<int64_t, kBirths> staged_child_ids{};
    std::array<double, kBirths> staged_child_energy{};
    int64_t staged_count = -1;
    int64_t queue_head = -1;
    int64_t queue_count = -1;

    identity::OrdinaryBirthIdentityConfig config{};
    config.plan_capacity = kBirths;
    config.slot_count = kSlots;
    // This is the low-ID compatibility-shadow width, not an ID ceiling.
    config.agent_id_capacity = kSlots;
    config.grid_height = 2;
    config.grid_width = 2;
    config.queue_capacity = kBirths;
    config.parent_energy_share = 0.25;
    config.minimum_energy_buffer = 1.0;
    config.start_health = 10.0F;
    config.cooldown_steps = 3;
    config.generation = 0;
    config.tick = 0;
    config.parent_order_mode = order;

    return static_cast<int>(preflight::
        preflight_ordinary_birth_identity_external_scratch_v2(
            {
                slot_to_agent_id.data(),
                alive.data(),
                grid.data(),
                energy.data(),
                health.data(),
                cooldown.data(),
                threshold.data(),
                observed_energy.data(),
                free_queue.data(),
                &free_count,
                &next_id,
            },
            {
                planned_parent_slots.data(),
                spawn_positions.data(),
                &planned_count,
            },
            {
                staged_parents.data(),
                staged_children.data(),
                staged_child_ids.data(),
                staged_child_energy.data(),
                &staged_count,
                &queue_head,
                &queue_count,
            },
            config
        )
    );
}

int main() {
    const std::array<int64_t, 2> reversed_unique{0, 1};
    const std::array<int64_t, 2> duplicate_parent{0, 0};

    if (
        run_preflight(ParentOrder::ascending_agent_id_v1, reversed_unique)
        != identity::kStatusInvalidParentOrder
    ) {
        return 1;
    }
    if (
        run_preflight(ParentOrder::authoritative_plan_v1, reversed_unique)
        != identity::kStatusOk
    ) {
        return 2;
    }
    if (
        run_preflight(ParentOrder::ascending_agent_id_v1, duplicate_parent)
        != identity::kStatusInvalidParentOrder
    ) {
        return 3;
    }
    if (
        run_preflight(ParentOrder::authoritative_plan_v1, duplicate_parent)
        != identity::kStatusInvalidParentOrder
    ) {
        return 4;
    }
    return 0;
}
"""


def test_native_mate_birth_parent_order_is_a_typed_transaction_contract(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 host compiler is required")

    (tmp_path / "cuda_runtime.h").write_text("#pragma once\n", encoding="ascii")
    source = tmp_path / "native_mate_birth_parent_order_v1.cpp"
    source.write_text(_HOST_PROBE, encoding="utf-8")
    executable = tmp_path / "native_mate_birth_parent_order_v1"
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-D__device__=",
            "-D__forceinline__=inline",
            "-I",
            str(tmp_path),
            "-I",
            str(PROJECT_ROOT),
            str(source),
            "-o",
            str(executable),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    result = subprocess.run(
        [str(executable)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"host transaction probe returned {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
