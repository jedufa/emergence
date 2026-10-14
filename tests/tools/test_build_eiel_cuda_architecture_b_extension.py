from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

import perf.cuda_architecture_b_generation as generation
from perf.cuda_architecture_b_generation import (
    COMPILE_SOURCES as GENERATION_COMPILE_SOURCES,
    HASH_SOURCES as GENERATION_HASH_SOURCES,
)
from perf.cuda_architecture_b_prebuilt import (
    EXTENSION_ABI_VERSION as PREBUILT_ABI_VERSION,
    EXTENSION_VERSION as PREBUILT_EXTENSION_VERSION,
)
from tools.scripts.build_eiel_cuda_architecture_b_extension import (
    CUDA_ARCHITECTURE_B_CAPABILITY_CONTRACT,
    CUDA_ARCHITECTURE_B_COMPILE_SOURCES,
    CUDA_ARCHITECTURE_B_SOURCE_INVENTORY,
    CXX_BUILD_FLAGS,
    EXPORTED_FUNCTIONS,
    EXTENSION_ABI_VERSION,
    EXTENSION_VERSION,
    NVCC_BUILD_FLAGS,
    _cuda_arch_from_capability,
    _canonical_json_sha256,
    _prepare_source_snapshots,
    _require_outputs_absent,
    _verify_build_ninja,
    _validate_module_name,
)


EXPECTED_COMPILE_SOURCE_NAMES = (
    "cuda_architecture_b_generation.cpp",
    "cuda_architecture_b_generation_multicell.cpp",
    "cxxbackbone8r_generation_island_cuda.cu",
    "cuda_architecture_b_fixed_mlp.cu",
    "cuda_architecture_b_modular_tick.cpp",
    "cuda_architecture_b_policy_cache.cu",
    "cuda_architecture_b_patch_context.cu",
    "cuda_architecture_b_flat_food_spawn.cu",
    "cuda_architecture_b_active_mechanics.cu",
    "cuda_architecture_b_lifecycle.cu",
    "cuda_architecture_b_death_lifecycle_events.cu",
    "cuda_architecture_b_birth_microcycle.cpp",
    "cuda_architecture_b_asexual_birth_plan.cu",
    "cuda_architecture_b_birth_transaction.cpp",
    "cuda_architecture_b_child_drives.cu",
    "cuda_architecture_b_child_policy.cu",
    "cuda_architecture_b_turnover.cu",
    "cuda_architecture_b_birth_lifecycle_events.cu",
    "cuda_architecture_b_death_class_concat.cu",
)


def test_cuda_architecture_b_builder_source_inventory_matches_generation_fail_closed() -> None:
    generation_compile_sources = tuple(source.resolve() for source in GENERATION_COMPILE_SOURCES)

    assert tuple(source.name for source in generation_compile_sources) == EXPECTED_COMPILE_SOURCE_NAMES
    assert CUDA_ARCHITECTURE_B_COMPILE_SOURCES == generation_compile_sources
    assert CUDA_ARCHITECTURE_B_SOURCE_INVENTORY == tuple(
        source.resolve() for source in GENERATION_HASH_SOURCES
    )
    assert "cuda_architecture_b_generation_context.hpp" in {
        source.name for source in CUDA_ARCHITECTURE_B_SOURCE_INVENTORY
    }
    assert "cuda_architecture_b_modular_tick.hpp" in {
        source.name for source in CUDA_ARCHITECTURE_B_SOURCE_INVENTORY
    }
    assert all(source.is_file() for source in CUDA_ARCHITECTURE_B_SOURCE_INVENTORY)
    assert "-DEIEL_CUDA_ARCHITECTURE_B_NO_PYBIND" in CXX_BUILD_FLAGS
    assert "-DEIEL_CUDA_ARCHITECTURE_B_NO_BIRTH_MICROCYCLE_PYBIND" in CXX_BUILD_FLAGS
    assert CXX_BUILD_FLAGS[2:] == NVCC_BUILD_FLAGS[2:]


def test_cuda_architecture_b_builder_validates_module_names_and_gpu_arch() -> None:
    _validate_module_name("eiel_cuda_architecture_b_ext")
    assert _cuda_arch_from_capability((7, 5)) == "7.5"
    assert EXTENSION_ABI_VERSION == "eiel_cuda_architecture_b_abi_v3"
    assert EXTENSION_VERSION == "eiel_cuda_architecture_b_resident_generation_v6"
    assert PREBUILT_ABI_VERSION == EXTENSION_ABI_VERSION
    assert PREBUILT_EXTENSION_VERSION == EXTENSION_VERSION
    assert EXPORTED_FUNCTIONS == (
        "eiel_cuda_architecture_b_generation_v3",
        "eiel_cuda_architecture_b_generation_multicell_v2",
        "eiel_cuda_architecture_b_generation_phase_e_v1",
    )
    assert generation._REQUIRED_EXTENSION_EXPORTS == EXPORTED_FUNCTIONS
    assert CUDA_ARCHITECTURE_B_CAPABILITY_CONTRACT == (
        generation.CUDA_GENERATION_PREBUILT_CAPABILITY_CONTRACT
    )
    assert CUDA_ARCHITECTURE_B_CAPABILITY_CONTRACT[
        "phase_e_modular_generation"
    ] == generation.CUDA_MODULAR_GENERATION_CAPABILITIES
    assert CUDA_ARCHITECTURE_B_CAPABILITY_CONTRACT[
        "phase_e_modular_generation"
    ]["native_modular_tick"] is True
    assert len(_canonical_json_sha256({"bounded": True})) == 64

    with pytest.raises(ValueError, match="valid Python extension module"):
        _validate_module_name("invalid-module")
    with pytest.raises(ValueError, match="invalid CUDA device capability"):
        _cuda_arch_from_capability((0, 0))


def test_generation_prebuilt_runtime_requires_phase_e_export_and_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "phase_e_generation.so"
    captured: dict[str, tuple[str, ...]] = {}
    source_records = [
        {
            "path": str(source),
            "sha256": sha256(source.read_bytes()).hexdigest(),
        }
        for source in GENERATION_HASH_SOURCES
    ]
    identity = SimpleNamespace(
        artifact=artifact,
        artifact_sha256="a" * 64,
        manifest={
            "capability_contract": dict(
                generation.CUDA_GENERATION_PREBUILT_CAPABILITY_CONTRACT
            ),
            "sources": source_records,
        },
    )

    def fake_validate(path, *, required_exports):
        assert Path(path) == artifact
        captured["validated"] = tuple(required_exports)
        return identity

    sentinel = object()

    def fake_load(path, *, required_exports):
        assert Path(path) == artifact
        captured["loaded"] = tuple(required_exports)
        return sentinel

    monkeypatch.setattr(generation.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generation, "validate_cuda_prebuilt_manifest", fake_validate)
    monkeypatch.setattr(
        generation,
        "load_prebuilt_cuda_architecture_b_extension",
        fake_load,
    )
    monkeypatch.setattr(generation, "_MODULES", {})
    monkeypatch.setenv("EIEL_CUDA_ARCHITECTURE_B_PREBUILT", str(artifact))

    assert generation.load_cuda_generation_extension() is sentinel
    assert captured == {
        "validated": EXPORTED_FUNCTIONS,
        "loaded": EXPORTED_FUNCTIONS,
    }

    identity.manifest["capability_contract"] = dict(
        generation.CUDA_GENERATION_CAPABILITIES
    )
    monkeypatch.setattr(generation, "_MODULES", {})
    with pytest.raises(RuntimeError, match="capability contract does not match"):
        generation.load_cuda_generation_extension()


def test_cuda_architecture_b_builder_refuses_to_replace_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "eiel_cuda_architecture_b_ext.so"
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    assert _require_outputs_absent(output) == manifest

    output.write_bytes(b"immutable")
    with pytest.raises(FileExistsError, match="immutable CUDA extension output already exists"):
        _require_outputs_absent(output)

    output.unlink()
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable CUDA extension output already exists"):
        _require_outputs_absent(output)


def test_cuda_architecture_b_builder_captures_immutable_source_snapshots(tmp_path: Path) -> None:
    binding = tmp_path / "binding.cpp"
    cuda = tmp_path / "kernel.cu"
    header = tmp_path / "shared.cuh"
    binding.write_text("binding-v1\n", encoding="utf-8")
    cuda.write_text("cuda-v1\n", encoding="utf-8")
    header.write_text("header-v1\n", encoding="utf-8")

    snapshots, records = _prepare_source_snapshots((binding, cuda, header), tmp_path / "build")

    assert [path.read_text(encoding="utf-8") for path in snapshots] == [
        "binding-v1\n",
        "cuda-v1\n",
        "header-v1\n",
    ]
    assert [record["path"] for record in records] == [str(binding), str(cuda), str(header)]

    cuda.write_text("cuda-v2\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="snapshot already exists with different content"):
        _prepare_source_snapshots((binding, cuda, header), tmp_path / "build")


def test_cuda_architecture_b_builder_rejects_empty_or_colliding_source_sets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        _prepare_source_snapshots((), tmp_path / "empty")

    left = tmp_path / "left" / "kernel.cu"
    right = tmp_path / "right" / "kernel.cu"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_text("left\n", encoding="utf-8")
    right.write_text("right\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be unique"):
        _prepare_source_snapshots((left, right), tmp_path / "collision")


def test_cuda_architecture_b_builder_verifies_deterministic_fp_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "kernel.cu"
    source.write_text("// bounded fixture\n", encoding="utf-8")
    torch_lib = tmp_path / "torch-lib"
    include = tmp_path / "torch-include"
    monkeypatch.setattr(
        "tools.scripts.build_eiel_cuda_architecture_b_extension.cpp_extension.TORCH_LIB_PATH",
        str(torch_lib),
    )
    monkeypatch.setattr(
        "tools.scripts.build_eiel_cuda_architecture_b_extension.cpp_extension.include_paths",
        lambda: [str(include)],
    )
    required = [
        str(torch_lib),
        str(include),
        str(source),
        *CXX_BUILD_FLAGS,
        *NVCC_BUILD_FLAGS,
        "nvcc",
        "compute_75",
    ]
    build_ninja = tmp_path / "build.ninja"
    build_ninja.write_text("\n".join(required) + "\n", encoding="utf-8")

    _verify_build_ninja(
        build_ninja=build_ninja,
        sources=(source,),
        cuda_arch="7.5",
    )

    build_ninja.write_text(
        "\n".join(value for value in required if value != "--fmad=false") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="missing required build identity"):
        _verify_build_ninja(
            build_ninja=build_ninja,
            sources=(source,),
            cuda_arch="7.5",
        )
