from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEADER = PROJECT_ROOT / "engine/perf/native/resident_state_contract.hpp"


def test_resident_state_contract_is_python_free_and_manifest_external() -> None:
    text = HEADER.read_text(encoding="utf-8")

    for forbidden in ("Python.h", "pybind", "torch/", "ATen/"):
        assert forbidden not in text
    assert "tensor_record_count" in text
    assert "ManifestAuthority" in text
    assert "139" not in text


def test_resident_state_contract_compiles_and_fails_closed(tmp_path: Path) -> None:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")

    source = tmp_path / "resident_state_contract_test.cpp"
    executable = tmp_path / "resident_state_contract_test"
    source.write_text(
        r'''
#include "engine/perf/native/resident_state_contract.hpp"

#include <cstdint>
#include <type_traits>

using namespace eiel::architecture_b;

namespace {

ManifestAuthority manifest(const std::uint32_t field_count) {
    ManifestAuthority value{};
    value.schema_version = 2;
    value.field_count = field_count;
    for (std::size_t index = 0; index < kSha256Bytes; ++index) {
        value.sha256[index] = static_cast<std::uint8_t>(index + 1);
    }
    return value;
}

CapabilityMask base_flags() {
    return kRequiredDispatchCapabilities
        | capability_mask(CapabilityFlag::kFlatWorld)
        | capability_mask(CapabilityFlag::kFixedMlpPolicy)
        | capability_mask(CapabilityFlag::kLifecycle)
        | capability_mask(CapabilityFlag::kExactRngSchedule);
}

ResidentDispatchContract valid_dispatch(
    const ManifestAuthority& authority,
    RngStreamPodState* rng_states
) {
    ResidentDispatchContract contract{};
    contract.capabilities.contract_version = kResidentStateContractVersion;
    contract.capabilities.backend = BackendKind::kCudaResident;
    contract.capabilities.device = DeviceKind::kCuda;
    contract.capabilities.device_ordinal = 0;
    contract.capabilities.max_cells = 1;
    contract.capabilities.flags = base_flags();

    contract.cell_bank.contract_version = kResidentStateContractVersion;
    contract.cell_bank.backend = BackendKind::kCudaResident;
    contract.cell_bank.device = DeviceKind::kCuda;
    contract.cell_bank.device_ordinal = 0;
    contract.cell_bank.layout = CellBankLayout::kContiguousCellMajor;
    contract.cell_bank.leading_cell_axis = 0;
    contract.cell_bank.cell_count = 1;
    contract.cell_bank.capacity_per_cell = 64;
    contract.cell_bank.tensor_record_count = authority.field_count;
    contract.cell_bank.state_epoch = 7;
    contract.cell_bank.state_address = 0x1000;
    contract.cell_bank.state_bytes = 4096;
    contract.cell_bank.manifest = authority;

    contract.rng_bank.contract_version = kResidentStateContractVersion;
    contract.rng_bank.algorithm = RngAlgorithm::kCounter64;
    contract.rng_bank.device = DeviceKind::kCuda;
    contract.rng_bank.device_ordinal = 0;
    contract.rng_bank.cell_count = 1;
    contract.rng_bank.streams_per_cell = 2;
    contract.rng_bank.state_count = 2;
    contract.rng_bank.states_address = reinterpret_cast<std::uintptr_t>(rng_states);
    contract.rng_bank.state_stride_bytes = sizeof(RngStreamPodState);
    contract.rng_bank.state_bytes = 2 * sizeof(RngStreamPodState);
    return contract;
}

GenerationBoundary valid_boundary() {
    GenerationBoundary boundary{};
    boundary.contract_version = kResidentStateContractVersion;
    boundary.boundary_ordinal = 5;
    boundary.completed_generation = 3;
    boundary.next_generation = 4;
    boundary.completed_ticks = 80;
    boundary.state_epoch_before = 7;
    boundary.state_epoch_after = 8;
    boundary.rng_draw_count_before = 100;
    boundary.rng_draw_count_after = 140;
    boundary.copies.boundary_snapshot_count = 1;
    boundary.copies.boundary_snapshot_bytes = 4096;
    boundary.copies.boundary_device_to_host_operations = 4;
    boundary.copies.boundary_device_to_host_bytes = 3072;
    boundary.copies.boundary_device_synchronizations = 1;
    return boundary;
}

}  // namespace

int main() {
    static_assert(std::is_trivial_v<RngStreamPodState>);
    static_assert(std::is_trivially_copyable_v<ResidentDispatchContract>);
    static_assert(!ValidationResult{}.ok());

    constexpr std::uint32_t external_manifest_field_count = 139;
    const ManifestAuthority expected = manifest(external_manifest_field_count);
    RngStreamPodState rng_states[2]{};
    ResidentDispatchContract contract = valid_dispatch(expected, rng_states);
    const CapabilityMask requested =
        capability_mask(CapabilityFlag::kFlatWorld)
        | capability_mask(CapabilityFlag::kFixedMlpPolicy)
        | capability_mask(CapabilityFlag::kLifecycle)
        | capability_mask(CapabilityFlag::kExactRngSchedule);
    if (!validate_dispatch_contract(contract, expected, requested).ok()) {
        return 1;
    }

    ResidentDispatchContract default_contract{};
    if (validate_dispatch_contract(default_contract, expected, requested).ok()) {
        return 2;
    }

    ResidentDispatchContract wrong_device = contract;
    wrong_device.capabilities.device = DeviceKind::kHostCpu;
    if (validate_dispatch_contract(wrong_device, expected, requested).error
        != ContractError::kBackendDeviceMismatch) {
        return 3;
    }

    ResidentDispatchContract missing_capability = contract;
    missing_capability.capabilities.flags &=
        ~capability_mask(CapabilityFlag::kLifecycle);
    if (validate_dispatch_contract(missing_capability, expected, requested).error
        != ContractError::kMissingCapability) {
        return 4;
    }

    ResidentDispatchContract unknown_capability = contract;
    unknown_capability.capabilities.flags |= 1ULL << 63;
    if (validate_dispatch_contract(unknown_capability, expected, requested).error
        != ContractError::kUnknownCapability) {
        return 5;
    }

    ManifestAuthority wrong_manifest = expected;
    wrong_manifest.sha256[0] ^= 0xFF;
    if (validate_dispatch_contract(contract, wrong_manifest, requested).error
        != ContractError::kManifestMismatch) {
        return 6;
    }

    ResidentDispatchContract mismatched_rng_cells = contract;
    mismatched_rng_cells.rng_bank.cell_count = 2;
    mismatched_rng_cells.rng_bank.state_count = 4;
    mismatched_rng_cells.rng_bank.state_bytes = 4 * sizeof(RngStreamPodState);
    if (validate_dispatch_contract(mismatched_rng_cells, expected, requested).error
        != ContractError::kRngCellCountMismatch) {
        return 7;
    }

    ResidentDispatchContract unqualified_multicell = contract;
    unqualified_multicell.capabilities.max_cells = 2;
    unqualified_multicell.cell_bank.cell_count = 2;
    unqualified_multicell.rng_bank.cell_count = 2;
    unqualified_multicell.rng_bank.state_count = 4;
    unqualified_multicell.rng_bank.state_bytes = 4 * sizeof(RngStreamPodState);
    if (validate_dispatch_contract(unqualified_multicell, expected, requested).error
        != ContractError::kMissingCapability) {
        return 8;
    }

    GenerationBoundary boundary = valid_boundary();
    if (!validate_generation_boundary(
            boundary,
            BackendKind::kCudaResident,
            BoundarySnapshotPolicy::kSingleImmutableSnapshot
        ).ok()) {
        return 9;
    }

    GenerationBoundary host_transfer = boundary;
    host_transfer.copies.trajectory_device_to_host_operations = 1;
    host_transfer.copies.trajectory_device_to_host_bytes = 8;
    if (validate_generation_boundary(
            host_transfer,
            BackendKind::kCudaResident,
            BoundarySnapshotPolicy::kSingleImmutableSnapshot
        ).error != ContractError::kTrajectoryHostTransfer) {
        return 10;
    }

    GenerationBoundary hidden_fallback = boundary;
    hidden_fallback.copies.hidden_fallbacks = 1;
    if (validate_generation_boundary(
            hidden_fallback,
            BackendKind::kCudaResident,
            BoundarySnapshotPolicy::kSingleImmutableSnapshot
        ).error != ContractError::kHiddenFallback) {
        return 11;
    }

    GenerationBoundary host_callback = boundary;
    host_callback.copies.host_trajectory_callbacks = 1;
    if (validate_generation_boundary(
            host_callback,
            BackendKind::kCudaResident,
            BoundarySnapshotPolicy::kSingleImmutableSnapshot
        ).error != ContractError::kTrajectoryHostCallback) {
        return 12;
    }

    GenerationBoundary missing_snapshot = boundary;
    missing_snapshot.copies.boundary_snapshot_count = 0;
    missing_snapshot.copies.boundary_snapshot_bytes = 0;
    if (validate_generation_boundary(
            missing_snapshot,
            BackendKind::kCudaResident,
            BoundarySnapshotPolicy::kSingleImmutableSnapshot
        ).error != ContractError::kBoundarySnapshotInvalid) {
        return 13;
    }

    return 0;
}
''',
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
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
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [str(executable)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run_result.returncode == 0, (
        f"contract executable failed with {run_result.returncode}: "
        f"{run_result.stdout}{run_result.stderr}"
    )


def test_resident_state_contract_compiles_in_cuda_translation_unit(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("NVCC unavailable")

    source = tmp_path / "resident_state_contract_test.cu"
    output = tmp_path / "resident_state_contract_test.o"
    source.write_text(
        r'''
#include "engine/perf/native/resident_state_contract.hpp"

using namespace eiel::architecture_b;

__global__ void validate_contract_kernel(ValidationResult* output) {
    *output = invalid_contract(ContractError::kUninitialized);
}

int main() {
    return ValidationResult{}.ok() ? 1 : 0;
}
''',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-Werror",
            "all-warnings",
            "-I",
            str(PROJECT_ROOT),
            "-c",
            str(source),
            "-o",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
